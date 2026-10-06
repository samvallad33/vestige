//! `vestige ingest-git` on a Strata log.
//!
//! Each commit becomes one `event` in the user scope, dated to the commit
//! time, with source key `(git, <repo identity>, <repo>#<sha>)`. The commit's
//! files become `touched` edges whose targets are repository-qualified file
//! handles, and each new-side hunk with lines becomes an anchor on that
//! commit. The anchor stores the hunk-header symbol. That is the symbol
//! structure: #382 has no owner decision, so this does not add a `calls` edge.
//!
//! Diff-body mentions and import lines stay in the record text when the
//! legacy record formatter includes them. They never become edges. A re-run
//! of the same commits appends no node, edge, or anchor frame.
//!
//! Git is local-only, through the same runner `ingest_repo` uses. Nothing is
//! fetched.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use strata_store::{
    AnchorRecord, ConnectionRecord, EdgeDirection, EdgeKind, IngestInput, SourceKey,
    effect_receipt_id, hunk_anchor_id, qualified_file_handle,
};
use vestige_core::advanced::git_records::{self, GitCommit, HunkSpan};
use vestige_core::{DEFAULT_MEMORY_SCOPE, Storage};

use super::repo_ingest::{self, GitRun};
use crate::strata_memory::{self, blocking_secrets_in};

/// What to read from a local checkout.
pub struct Request {
    /// Checkout path.
    pub repo_path: PathBuf,
    /// `git log --since`, when set.
    pub since: Option<String>,
    /// `git log --until`, when set.
    pub until: Option<String>,
    /// Newest-first cap passed to `git log -n`.
    pub max_commits: usize,
}

struct Prepared {
    display: String,
    identity: String,
    commits: Vec<GitCommit>,
}

struct EdgeOut {
    target: String,
    meta_sha: String,
    effect_seq: Option<u64>,
    data_seq: Option<u64>,
    written: bool,
}

struct AnchorOut {
    id: String,
    file: String,
    symbol: Option<String>,
    start_line: u32,
    end_line: u32,
    effect_seq: Option<u64>,
    data_seq: Option<u64>,
    written: bool,
}

struct CommitOut {
    sha: String,
    id: Option<String>,
    creating_seq: Option<u64>,
    data_seq: Option<u64>,
    /// This call admitted the node. A re-run is false even when edges are repaired.
    created_node: bool,
    unchanged: bool,
    skipped_secret: bool,
    extra_files: usize,
    extra_hunks: usize,
    edges: Vec<EdgeOut>,
    anchors: Vec<AnchorOut>,
}

/// Ingest `req` into the open Strata log behind `storage`.
///
/// A git failure writes nothing. The returned object is the `--json` body.
pub fn execute(storage: &Storage, req: Request) -> Result<Value, String> {
    let prepared = prepare(&req)?;
    let memory = strata_memory::live_memory(storage)
        .ok_or_else(|| "ingest-git needs the Strata log open in this process".to_string())?;
    let commits = memory.with_store_mut(|store| write_all(store, &prepared))?;
    Ok(report(&prepared, &commits))
}

/// Plain-text form of [`execute`]'s object. A re-run says that no new frame
/// was appended.
pub fn render_human(report: &Value) -> String {
    let mut lines = vec![
        format!("Repo: {}", report["repo"].as_str().unwrap_or("")),
        format!(
            "Identity: {}",
            report["repo_identity"].as_str().unwrap_or("")
        ),
        format!(
            "Commits seen: {}",
            report["commits_seen"].as_u64().unwrap_or(0)
        ),
        format!("Created: {}", report["created"].as_u64().unwrap_or(0)),
        format!("Unchanged: {}", report["unchanged"].as_u64().unwrap_or(0)),
        format!(
            "Edges written: {}",
            report["edges_written"].as_u64().unwrap_or(0)
        ),
        format!(
            "Anchors written: {}",
            report["anchors_written"].as_u64().unwrap_or(0)
        ),
        format!("status: {}", report["status"].as_str().unwrap_or("")),
    ];
    if report["status"].as_str() == Some("unchanged") {
        lines.push("unchanged (no new frames)".to_string());
    }
    if let Some(commits) = report["commits"].as_array() {
        for commit in commits {
            lines.push(format!(
                "commit {} id={} receipt={} extra_files={} extra_hunks={} unchanged={}",
                commit["sha"].as_str().unwrap_or(""),
                commit["id"].as_str().unwrap_or(""),
                commit["receipt"].as_str().unwrap_or(""),
                commit["extra_files"].as_u64().unwrap_or(0),
                commit["extra_hunks"].as_u64().unwrap_or(0),
                commit["unchanged"].as_bool().unwrap_or(false),
            ));
        }
    }
    lines.join("\n")
}

/// Remote identity, or the canonical path when the checkout has no origin.
///
/// The remote form matches the historical CLI normalization: strip one
/// protocol prefix and a trailing `.git`, then replace `:` with `/`.
pub fn repo_identity(path: &Path) -> String {
    if let Some(remote) = remote_identity(path) {
        return remote;
    }
    path.canonicalize()
        .map(|real| real.display().to_string())
        .unwrap_or_else(|_| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "repo".to_string())
        })
}

fn remote_identity(path: &Path) -> Option<String> {
    let run = repo_ingest::run_git(
        path,
        &["config".into(), "--get".into(), "remote.origin.url".into()],
    )
    .ok()?;
    if run.failure.is_some() {
        return None;
    }
    let url = String::from_utf8_lossy(&run.stdout).trim().to_string();
    if url.is_empty() {
        return None;
    }
    let stripped = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("git@")
        .trim_start_matches("ssh://")
        .trim_end_matches(".git")
        .replace(':', "/");
    Some(stripped)
}

fn prepare(req: &Request) -> Result<Prepared, String> {
    if req.max_commits == 0 {
        return Err("max-commits must be at least 1".into());
    }
    check_bound("since", req.since.as_deref())?;
    check_bound("until", req.until.as_deref())?;
    let root = std::fs::canonicalize(&req.repo_path).map_err(|_| {
        format!(
            "ingest-git path '{}' is not an available directory",
            req.repo_path.display()
        )
    })?;
    if !root.is_dir() {
        return Err("ingest-git path must be a directory".into());
    }
    let display = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".to_string());
    let identity = repo_identity(&root);
    let run = read_log(&root, req)?;
    if let Some(failure) = run.failure {
        return Err(format!("git log failed: {failure}"));
    }
    let commits = git_records::parse_git_log(&String::from_utf8_lossy(&run.stdout));
    Ok(Prepared {
        display,
        identity,
        commits,
    })
}

fn check_bound(name: &str, value: Option<&str>) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_empty() || value.len() > 200 || value.chars().any(char::is_control) {
        return Err(format!(
            "{name} must be at most 200 characters with no control characters"
        ));
    }
    Ok(())
}

fn read_log(root: &Path, req: &Request) -> Result<GitRun, String> {
    let mut args = vec![
        "-c".to_string(),
        "diff.renames=true".into(),
        "log".into(),
        "-p".into(),
        "--unified=0".into(),
        "--no-color".into(),
        "--no-ext-diff".into(),
        "--no-textconv".into(),
        "--no-show-signature".into(),
        "-n".into(),
        req.max_commits.to_string(),
        format!("--pretty=format:{}", git_records::GIT_LOG_PRETTY),
    ];
    if let Some(since) = &req.since {
        args.push(format!("--since={since}"));
    }
    if let Some(until) = &req.until {
        args.push(format!("--until={until}"));
    }
    repo_ingest::run_git(root, &args)
}

fn write_all(
    store: &mut strata_store::StrataStore,
    prepared: &Prepared,
) -> Result<Vec<CommitOut>, String> {
    let mut known: HashMap<String, String> = HashMap::new();
    for node in store.nodes() {
        if !node.is_live() || node.scope != DEFAULT_MEMORY_SCOPE {
            continue;
        }
        let Some(source) = node.source.as_ref() else {
            continue;
        };
        if source.system == git_records::SOURCE_SYSTEM {
            known.insert(source.id.clone(), node.id.clone());
        }
    }
    let mut out = Vec::with_capacity(prepared.commits.len());
    // git lists newest first. Write oldest first so ids follow history.
    for commit in prepared.commits.iter().rev() {
        out.push(write_one(store, &mut known, &prepared.identity, commit)?);
    }
    Ok(out)
}

fn source_key(identity: &str, sha: &str) -> SourceKey {
    SourceKey {
        system: git_records::SOURCE_SYSTEM.to_string(),
        project: identity.to_string(),
        id: format!("{identity}#{sha}"),
    }
}

fn write_one(
    store: &mut strata_store::StrataStore,
    known: &mut HashMap<String, String>,
    identity: &str,
    commit: &GitCommit,
) -> Result<CommitOut, String> {
    let content = git_records::record_content(commit);
    let tag = format!("commit:{}", commit.sha);
    let at = commit.time.timestamp_millis();
    if !blocking_secrets_in([
        content.as_str(),
        git_records::COMMIT_TAG,
        tag.as_str(),
        identity,
    ])
    .is_empty()
    {
        return Ok(CommitOut {
            sha: commit.sha.clone(),
            id: None,
            creating_seq: None,
            data_seq: None,
            created_node: false,
            unchanged: false,
            skipped_secret: true,
            extra_files: commit.extra_files,
            extra_hunks: commit.extra_hunks,
            edges: Vec::new(),
            anchors: Vec::new(),
        });
    }

    let key = source_key(identity, &commit.sha);
    let (node_id, created) = if let Some(id) = known.get(&key.id).cloned() {
        (id, false)
    } else {
        let (id, _seq) = store
            .ingest_in_scope_with_receipt(
                IngestInput {
                    content,
                    source: Some(key.clone()),
                    source_updated_at_ms: Some(at),
                    node_type: "event".into(),
                    tags: vec![git_records::COMMIT_TAG.to_string(), tag],
                    created_at_ms: Some(at),
                    valid_from_ms: Some(at),
                    valid_until_ms: None,
                },
                DEFAULT_MEMORY_SCOPE,
            )
            .map_err(|err| format!("commit {} was not admitted: {err}", commit.sha))?;
        known.insert(key.id.clone(), id.clone());
        (id, true)
    };

    let creating_seq = store.origin_seq(&node_id);
    let data_seq = creating_seq
        .and_then(|seq| store.effect_by_seq(seq).ok().flatten())
        .map(|proof| proof.data_seq);

    let mut edges = Vec::new();
    let mut edges_written = 0usize;
    for file in &commit.files {
        let target = qualified_file_handle(identity, file);
        let edge = write_touched(store, &node_id, &target, &commit.sha, at)?;
        if edge.written {
            edges_written += 1;
        }
        edges.push(edge);
    }

    let mut anchors = Vec::new();
    let mut fresh: Vec<AnchorRecord> = Vec::new();
    let mut anchors_written = 0usize;
    for hunk in commit.hunks.iter().filter(|hunk| hunk.len > 0) {
        let row = anchor_row(&key, &node_id, hunk, at);
        if store.anchor(&row.id).as_ref() == Some(&row) {
            anchors.push(anchor_out(store, &row, false)?);
            continue;
        }
        if fresh.iter().any(|kept| kept.id == row.id) {
            continue;
        }
        fresh.push(row);
    }
    if !fresh.is_empty() {
        store
            .record_anchors(fresh.clone())
            .map_err(|err| format!("hunk anchors for {} were not admitted: {err}", commit.sha))?;
        anchors_written = fresh.len();
        for row in &fresh {
            anchors.push(anchor_out(store, row, true)?);
        }
    }

    let unchanged = !created && edges_written == 0 && anchors_written == 0;
    Ok(CommitOut {
        sha: commit.sha.clone(),
        id: Some(node_id),
        creating_seq,
        data_seq,
        created_node: created,
        unchanged,
        skipped_secret: false,
        extra_files: commit.extra_files,
        extra_hunks: commit.extra_hunks,
        edges,
        anchors,
    })
}

fn write_touched(
    store: &mut strata_store::StrataStore,
    source: &str,
    target: &str,
    sha: &str,
    at: i64,
) -> Result<EdgeOut, String> {
    let existing = store
        .get_edges_for(source, EdgeDirection::Outgoing, Some(EdgeKind::Touched))
        .into_iter()
        .any(|edge| edge.target_id == target && edge.meta_sha.as_deref() == Some(sha));
    let written = if existing {
        false
    } else {
        store
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength_milli: 1000,
                link_type: EdgeKind::Touched.as_str().to_string(),
                meta_sha: Some(sha.to_string()),
                created_at_ms: at,
                activation_count: 0,
            })
            .map_err(|err| format!("touched edge {source} -> {target} was not admitted: {err}"))?;
        true
    };
    let proof = store
        .edge_proofs(source, target, EdgeKind::Touched.as_str())
        .map_err(|err| err.to_string())?
        .into_iter()
        .next_back();
    Ok(EdgeOut {
        target: target.to_string(),
        meta_sha: sha.to_string(),
        effect_seq: proof.as_ref().map(|row| row.effect_seq),
        data_seq: proof.as_ref().map(|row| row.data_seq),
        written,
    })
}

fn anchor_row(source: &SourceKey, node_id: &str, hunk: &HunkSpan, at: i64) -> AnchorRecord {
    AnchorRecord {
        id: hunk_anchor_id(source, &hunk.file, hunk.start, hunk.len),
        node_id: node_id.to_string(),
        file_path: hunk.file.clone(),
        symbol: hunk.symbol.clone(),
        symbol_kind: None,
        start_line: Some(hunk.start),
        end_line: Some(hunk.start.saturating_add(hunk.len).saturating_sub(1)),
        span_lines: None,
        content_hash: None,
        captured_at_ms: at,
        last_verified_at_ms: None,
        last_status: None,
    }
}

fn anchor_out(
    store: &strata_store::StrataStore,
    row: &AnchorRecord,
    written: bool,
) -> Result<AnchorOut, String> {
    let proof = store
        .latest_effect(&row.id)
        .map_err(|err| err.to_string())?;
    Ok(AnchorOut {
        id: row.id.clone(),
        file: row.file_path.clone(),
        symbol: row.symbol.clone(),
        start_line: row.start_line.unwrap_or(0),
        end_line: row.end_line.unwrap_or(0),
        effect_seq: proof.as_ref().map(|item| item.effect_seq),
        data_seq: proof.as_ref().map(|item| item.data_seq),
        written,
    })
}

fn receipt(seq: Option<u64>) -> Value {
    seq.map(effect_receipt_id)
        .map(Value::String)
        .unwrap_or(Value::Null)
}

fn report(prepared: &Prepared, commits: &[CommitOut]) -> Value {
    let created = commits.iter().filter(|commit| commit.created_node).count();
    let unchanged = commits.iter().filter(|commit| commit.unchanged).count();
    let skipped_secrets = commits
        .iter()
        .filter(|commit| commit.skipped_secret)
        .count();
    let edges_written = commits
        .iter()
        .flat_map(|commit| commit.edges.iter())
        .filter(|edge| edge.written)
        .count();
    let anchors_written = commits
        .iter()
        .flat_map(|commit| commit.anchors.iter())
        .filter(|anchor| anchor.written)
        .count();
    let status = if created + edges_written + anchors_written > 0 {
        "recorded"
    } else if skipped_secrets > 0 {
        "skipped"
    } else {
        "unchanged"
    };
    json!({
        "repo": prepared.display,
        "repo_identity": prepared.identity,
        "commits_seen": prepared.commits.len(),
        "created": created,
        "unchanged": unchanged,
        "skipped_secrets": skipped_secrets,
        "edges_written": edges_written,
        "anchors_written": anchors_written,
        "status": status,
        "commits": commits.iter().map(commit_json).collect::<Vec<_>>(),
    })
}

fn commit_json(commit: &CommitOut) -> Value {
    json!({
        "sha": commit.sha,
        "id": commit.id,
        "creating_seq": commit.creating_seq,
        "data_seq": commit.data_seq,
        "receipt": receipt(commit.creating_seq),
        "unchanged": commit.unchanged,
        "skipped_secret": commit.skipped_secret,
        "extra_files": commit.extra_files,
        "extra_hunks": commit.extra_hunks,
        "edges": commit.edges.iter().map(|edge| json!({
            "target": edge.target,
            "meta_sha": edge.meta_sha,
            "link_type": "touched",
            "effect_seq": edge.effect_seq,
            "data_seq": edge.data_seq,
            "receipt": receipt(edge.effect_seq),
            "written": edge.written,
        })).collect::<Vec<_>>(),
        "anchors": commit.anchors.iter().map(|anchor| json!({
            "id": anchor.id,
            "file": anchor.file,
            "symbol": anchor.symbol,
            "start_line": anchor.start_line,
            "end_line": anchor.end_line,
            "effect_seq": anchor.effect_seq,
            "data_seq": anchor.data_seq,
            "receipt": receipt(anchor.effect_seq),
            "written": anchor.written,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strata_memory::ghostlink::{Lens, ProposeRequest, propose};
    use std::process::Command;
    use std::sync::Arc;

    const D0: &str = "2024-01-01T00:00:00+00:00";
    const D1: &str = "2024-01-02T00:00:00+00:00";

    struct Repo {
        dir: tempfile::TempDir,
    }

    impl Repo {
        fn new() -> Self {
            let repo = Repo {
                dir: tempfile::TempDir::new().unwrap(),
            };
            repo.git(&["init", "-q"], D0);
            repo
        }

        fn path(&self) -> &Path {
            self.dir.path()
        }

        fn git(&self, args: &[&str], date: &str) -> String {
            let out = Command::new("git")
                .arg("-C")
                .arg(self.path())
                .args([
                    "-c",
                    "user.email=t@example.com",
                    "-c",
                    "user.name=t",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                ])
                .args(args)
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_SYSTEM", "/dev/null")
                .env("GIT_AUTHOR_DATE", date)
                .env("GIT_COMMITTER_DATE", date)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        }

        fn write(&self, rel: &str, body: &str) {
            let path = self.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }

        fn commit(&self, message: &str, date: &str) {
            self.git(&["add", "-A"], date);
            self.git(&["commit", "-q", "-m", message], date);
        }

        fn request(&self) -> Request {
            Request {
                repo_path: self.path().to_path_buf(),
                since: None,
                until: None,
                max_commits: 50,
            }
        }
    }

    fn strata() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn logged(repo: &Repo) -> Vec<GitCommit> {
        let run = repo_ingest::run_git(
            repo.path(),
            &[
                "-c".into(),
                "diff.renames=true".into(),
                "log".into(),
                "-p".into(),
                "--unified=0".into(),
                "--no-color".into(),
                "--no-ext-diff".into(),
                "--no-textconv".into(),
                "--no-show-signature".into(),
                "-n".into(),
                "50".into(),
                format!("--pretty=format:{}", git_records::GIT_LOG_PRETTY),
            ],
        )
        .unwrap();
        assert!(run.failure.is_none(), "{:?}", run.failure);
        let mut commits = git_records::parse_git_log(&String::from_utf8_lossy(&run.stdout));
        commits.reverse();
        commits
    }

    fn paths_of(identity: &str, commit: &Value) -> Vec<String> {
        commit["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| {
                assert_eq!(edge["meta_sha"], commit["sha"], "{edge}");
                assert_eq!(edge["link_type"], "touched", "{edge}");
                assert!(
                    edge["receipt"].as_str().unwrap().starts_with("eff-"),
                    "{edge}"
                );
                let (repo, path) =
                    strata_store::parse_qualified_file_handle(edge["target"].as_str().unwrap())
                        .unwrap();
                assert_eq!(repo, identity);
                path
            })
            .collect()
    }

    #[test]
    fn touched_edges_and_hunk_anchors_match_git_and_bridge() {
        let repo = Repo::new();
        repo.write("src/a.rs", "fn parse() {\n    let _ = 1;\n}\n");
        repo.commit("add parse", D0);
        repo.write(
            "src/a.rs",
            "use other::thing::module;\nfn parse() {\n    let _ = 2;\n    let _ = \"BLOCK_TOKEN_ZX91\";\n}\n",
        );
        repo.commit("edit parse", D1);

        let (storage, _dir) = strata();
        let report = execute(storage.as_ref(), repo.request()).unwrap();
        assert_eq!(report["status"], "recorded", "{report}");
        assert_eq!(report["created"], 2, "{report}");
        assert_eq!(report["commits_seen"], 2, "{report}");
        let identity = report["repo_identity"].as_str().unwrap().to_string();
        let git_commits = logged(&repo);
        let rows = report["commits"].as_array().unwrap();
        assert_eq!(rows.len(), git_commits.len());
        for (row, git_commit) in rows.iter().zip(&git_commits) {
            assert_eq!(row["sha"], git_commit.sha, "{row}");
            assert_eq!(row["extra_files"], git_commit.extra_files, "{row}");
            assert_eq!(row["extra_hunks"], git_commit.extra_hunks, "{row}");
            assert!(
                row["receipt"].as_str().unwrap().starts_with("eff-"),
                "{row}"
            );
            let mut paths = paths_of(&identity, row);
            let mut files = git_commit.files.clone();
            paths.sort();
            files.sort();
            assert_eq!(
                paths, files,
                "edges are the diff files, not mentions: {row}"
            );
            assert!(
                row["edges"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|edge| !edge["target"]
                        .as_str()
                        .unwrap()
                        .contains("BLOCK_TOKEN_ZX91")
                        && !edge["target"].as_str().unwrap().contains("other")),
                "{row}"
            );
            let anchors = row["anchors"].as_array().unwrap();
            let wanted: Vec<&HunkSpan> = git_commit
                .hunks
                .iter()
                .filter(|hunk| hunk.len > 0)
                .collect();
            assert_eq!(anchors.len(), wanted.len(), "{row}");
            for hunk in wanted {
                assert!(
                    anchors.iter().any(|anchor| {
                        anchor["file"] == hunk.file
                            && anchor["start_line"] == hunk.start
                            && anchor["end_line"] == hunk.start + hunk.len - 1
                            && anchor["symbol"] == json!(hunk.symbol)
                            && anchor["receipt"].as_str().unwrap().starts_with("eff-")
                    }),
                    "missing {hunk:?} in {anchors:?}"
                );
            }
            let id = row["id"].as_str().unwrap();
            strata_memory::live_memory(storage.as_ref())
                .unwrap()
                .with_store_mut(|store| {
                    for anchor in store.anchors_for(id) {
                        assert!(anchor.content_hash.is_none(), "{anchor:?}");
                        assert!(anchor.span_lines.is_none(), "{anchor:?}");
                        assert!(
                            !anchor.file_path.contains(&identity),
                            "anchor path is repository-relative: {anchor:?}"
                        );
                    }
                    for edge in
                        store.get_edges_for(id, EdgeDirection::Outgoing, Some(EdgeKind::Touched))
                    {
                        assert_eq!(edge.meta_sha.as_deref(), Some(git_commit.sha.as_str()));
                    }
                });
        }

        let answer = propose(
            storage.as_ref(),
            &ProposeRequest {
                lens: Lens::Bridge,
                scope: Some(DEFAULT_MEMORY_SCOPE.to_string()),
                tags: Vec::new(),
                limit: 10,
                cursor: None,
            },
        )
        .unwrap();
        let candidates = answer["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 1, "{answer}");
        assert_eq!(candidates[0]["proof"]["hops"], 2, "{answer}");
        let path = candidates[0]["proof"]["path"].to_string();
        let handle = qualified_file_handle(&identity, "src/a.rs");
        assert!(path.contains(&handle), "{path}");
        assert!(path.contains("touched"), "{path}");

        let again = execute(storage.as_ref(), repo.request()).unwrap();
        assert_eq!(again["status"], "unchanged", "{again}");
        assert_eq!(again["created"], 0, "{again}");
        assert_eq!(again["edges_written"], 0, "{again}");
        assert_eq!(again["anchors_written"], 0, "{again}");
        assert_eq!(again["unchanged"], 2, "{again}");
        assert!(
            render_human(&again).contains("no new frames"),
            "{}",
            render_human(&again)
        );
        let edges_after = storage.get_all_connections().unwrap().len();
        let third = execute(storage.as_ref(), repo.request()).unwrap();
        assert_eq!(third["status"], "unchanged", "{third}");
        assert_eq!(storage.get_all_connections().unwrap().len(), edges_after);
    }

    #[test]
    fn rename_records_the_new_path_and_delete_keeps_a_touched_edge_without_a_range() {
        let repo = Repo::new();
        repo.write("src/old.rs", "fn kept() {\n    let _ = 1;\n}\n");
        repo.commit("add", D0);
        repo.git(&["mv", "src/old.rs", "src/new.rs"], D1);
        repo.commit("rename", D1);
        repo.git(&["rm", "src/new.rs"], D1);
        repo.commit("delete", D1);

        let (storage, _dir) = strata();
        let report = execute(storage.as_ref(), repo.request()).unwrap();
        let identity = report["repo_identity"].as_str().unwrap().to_string();
        let rows = report["commits"].as_array().unwrap();
        assert_eq!(rows.len(), 3, "{report}");
        let rename_paths = paths_of(&identity, &rows[1]);
        assert_eq!(rename_paths, vec!["src/new.rs".to_string()], "{report}");
        assert!(!rename_paths.iter().any(|path| path == "src/old.rs"));
        let delete_paths = paths_of(&identity, &rows[2]);
        assert_eq!(delete_paths, vec!["src/new.rs".to_string()], "{report}");
        assert!(
            rows[2]["anchors"].as_array().unwrap().is_empty(),
            "a pure deletion has no new-side line range: {}",
            rows[2]
        );
    }

    #[test]
    fn file_and_hunk_overflow_are_counted() {
        let repo = Repo::new();
        for index in 0..51 {
            repo.write(&format!("f{index}.txt"), "x\n");
        }
        repo.commit("many files", D0);
        let mut wide = String::new();
        for line in 0..403 {
            wide.push_str(&format!("line {line} stable\n"));
        }
        repo.write("wide.txt", &wide);
        repo.commit("wide base", D1);
        let mut changed = String::new();
        for line in 0..403 {
            if line % 2 == 0 {
                changed.push_str(&format!("line {line} changed\n"));
            } else {
                changed.push_str(&format!("line {line} stable\n"));
            }
        }
        repo.write("wide.txt", &changed);
        repo.commit("wide edit", D1);

        let (storage, _dir) = strata();
        let report = execute(storage.as_ref(), repo.request()).unwrap();
        let git_commits = logged(&repo);
        let rows = report["commits"].as_array().unwrap();
        assert_eq!(rows.len(), git_commits.len());
        for (row, git_commit) in rows.iter().zip(&git_commits) {
            assert_eq!(
                row["extra_files"].as_u64().unwrap() as usize,
                git_commit.extra_files
            );
            assert_eq!(
                row["extra_hunks"].as_u64().unwrap() as usize,
                git_commit.extra_hunks
            );
            assert_eq!(
                row["edges"].as_array().unwrap().len(),
                git_commit.files.len(),
                "overflow files are not silent edges: {row}"
            );
            let ranged = git_commit.hunks.iter().filter(|hunk| hunk.len > 0).count();
            assert_eq!(row["anchors"].as_array().unwrap().len(), ranged, "{row}");
        }
        assert!(
            rows.iter()
                .any(|row| row["extra_files"].as_u64().unwrap() > 0)
        );
        assert!(
            rows.iter()
                .any(|row| row["extra_hunks"].as_u64().unwrap() > 0)
        );
    }

    #[test]
    fn two_repositories_do_not_bridge_through_a_shared_relative_path() {
        let left = Repo::new();
        let right = Repo::new();
        for repo in [&left, &right] {
            repo.write("src/a.rs", "fn parse() {\n    let _ = 1;\n}\n");
            repo.commit("add", D0);
            repo.write("src/a.rs", "fn parse() {\n    let _ = 2;\n}\n");
            repo.commit("edit", D1);
        }
        let (storage, _dir) = strata();
        let left_report = execute(storage.as_ref(), left.request()).unwrap();
        let right_report = execute(storage.as_ref(), right.request()).unwrap();
        let left_id = left_report["repo_identity"].as_str().unwrap().to_string();
        let right_id = right_report["repo_identity"].as_str().unwrap().to_string();
        assert_ne!(left_id, right_id);
        assert_ne!(
            qualified_file_handle(&left_id, "src/a.rs"),
            qualified_file_handle(&right_id, "src/a.rs")
        );
        let answer = propose(
            storage.as_ref(),
            &ProposeRequest {
                lens: Lens::Bridge,
                scope: Some(DEFAULT_MEMORY_SCOPE.to_string()),
                tags: Vec::new(),
                limit: 10,
                cursor: None,
            },
        )
        .unwrap();
        let candidates = answer["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 2, "{answer}");
        let project = |id: &str| -> String {
            strata_memory::live_memory(storage.as_ref())
                .unwrap()
                .with_store_mut(|store| store.get_node(id).unwrap().source.unwrap().project)
        };
        for candidate in candidates {
            let first = candidate["firstId"].as_str().unwrap();
            let second = candidate["secondId"].as_str().unwrap();
            assert_eq!(project(first), project(second), "{candidate}");
            let path = candidate["proof"]["path"].to_string();
            let handle = qualified_file_handle(&project(first), "src/a.rs");
            assert!(path.contains(&handle), "{path}");
            assert_eq!(candidate["proof"]["hops"], 2, "{candidate}");
        }
    }

    #[test]
    fn a_path_that_is_not_a_git_checkout_writes_nothing() {
        let (storage, dir) = strata();
        let before = storage.get_stats().unwrap().total_nodes;
        let err = execute(
            storage.as_ref(),
            Request {
                repo_path: dir.path().to_path_buf(),
                since: None,
                until: None,
                max_commits: 10,
            },
        )
        .unwrap_err();
        assert!(err.contains("git log failed"), "{err}");
        assert!(!err.contains("unavailable_in_4_0"), "{err}");
        assert_eq!(storage.get_stats().unwrap().total_nodes, before);
    }
}
