//! Commit receipts: the facts a commit receipt must carry, read from git.
//!
//! One source for three callers: the Reflex Stop invariant (an agent's
//! receipt must name these facts), `vestige hook post-commit` (writes the
//! machine receipt into Strata on every commit), and `vestige hook
//! pre-commit` (secret scan, open stops, witness for the staged tree).
//! Everything is read from git plumbing. No word lists, no LLM.

use std::path::Path;
use std::process::{Command, Stdio};

/// Tag on every machine-written receipt. An agent's own receipt does not
/// carry it, so the Stop invariant can tell the two apart.
pub const AUTO_TAG: &str = "commit-receipt-auto";
/// Tag on every commit receipt, machine or agent.
pub const RECEIPT_TAG: &str = "commit-receipt";
/// Largest number of files a receipt is required to name.
pub const MAX_FILES: usize = 64;

/// Changed line ranges of one file, from `git diff -U0` (new side).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileRanges {
    pub path: String,
    /// Inclusive `(start, end)` on the new side. A pure deletion is `(n, n-1)`
    /// and prints as `n-0`.
    pub ranges: Vec<(u32, u32)>,
}

/// What git says about one commit.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CommitFacts {
    pub sha: String,
    pub parents: Vec<String>,
    /// `None` on a detached HEAD.
    pub branch: Option<String>,
    pub author: String,
    pub subject: String,
    pub files: Vec<FileRanges>,
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["-c", "core.fsmonitor=false", "-c", "color.ui=false"])
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Parse `git diff -U0` output into per-file new-side ranges.
pub fn parse_unified_zero(diff: &str) -> Vec<FileRanges> {
    let mut files: Vec<FileRanges> = Vec::new();
    let mut minus_path: Option<String> = None;
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            minus_path = rest.strip_prefix("a/").map(str::to_string);
            continue;
        }
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = match rest.strip_prefix("b/") {
                Some(p) => p.to_string(),
                // Deleted file: the new side is /dev/null.
                None => minus_path.clone().unwrap_or_default(),
            };
            if !path.is_empty() {
                files.push(FileRanges {
                    path,
                    ranges: Vec::new(),
                });
            }
            continue;
        }
        let Some(rest) = line.strip_prefix("@@ ") else {
            continue;
        };
        let Some(plus) = rest.split_whitespace().find(|w| w.starts_with('+')) else {
            continue;
        };
        let spec = &plus[1..];
        let (start, count) = match spec.split_once(',') {
            Some((a, b)) => (a.parse::<u32>().ok(), b.parse::<u32>().ok()),
            None => (spec.parse::<u32>().ok(), Some(1)),
        };
        if let (Some(start), Some(count), Some(file)) = (start, count, files.last_mut()) {
            let end = (start + count).saturating_sub(1);
            file.ranges.push((start, end.max(start.saturating_sub(1))));
        }
    }
    files
}

/// Read the facts of `rev` (normally `HEAD`) in `repo`.
pub fn read(repo: &Path, rev: &str) -> Option<CommitFacts> {
    let line = git(repo, &["rev-list", "--parents", "-n", "1", rev, "--"])?;
    let mut ids = line.split_whitespace().map(str::to_string);
    let sha = ids.next().filter(|s| is_hex(s))?;
    let parents: Vec<String> = ids.filter(|s| is_hex(s)).collect();
    let head = git(repo, &["rev-parse", "HEAD"]).unwrap_or_default();
    let branch = if head.trim() == sha {
        git(repo, &["symbolic-ref", "--short", "-q", "HEAD"])
            .map(|b| b.trim().to_string())
            .filter(|b| !b.is_empty())
    } else {
        None
    };
    let meta = git(repo, &["show", "-s", "--format=%an <%ae>%n%s", &sha])?;
    let mut meta = meta.lines();
    let author = meta.next().unwrap_or("").trim().to_string();
    let subject = meta.next().unwrap_or("").trim().to_string();
    let diff = git(
        repo,
        &[
            "show",
            "-U0",
            "--format=",
            "--no-ext-diff",
            "--no-renames",
            "--first-parent",
            &sha,
        ],
    )
    .unwrap_or_default();
    Some(CommitFacts {
        sha,
        parents,
        branch,
        author,
        subject,
        files: parse_unified_zero(&diff),
    })
}

fn contains_ci(hay: &str, needle: &str) -> bool {
    !needle.is_empty()
        && hay
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase())
}

impl CommitFacts {
    pub fn short(&self) -> &str {
        &self.sha[..self.sha.len().min(12)]
    }

    /// `path:12-30,41-41` per file.
    pub fn file_lines(&self) -> Vec<String> {
        self.files
            .iter()
            .map(|f| {
                let ranges: Vec<String> =
                    f.ranges.iter().map(|(a, b)| format!("{a}-{b}")).collect();
                if ranges.is_empty() {
                    f.path.clone()
                } else {
                    format!("{}:{}", f.path, ranges.join(","))
                }
            })
            .collect()
    }

    /// The facts a receipt `text` leaves out. A receipt names the full SHA,
    /// each parent (7+ hex prefix), the branch, the author, every changed
    /// file (up to [`MAX_FILES`]) and every gate command `gates` lists.
    pub fn missing_in(&self, text: &str, gates: &[String]) -> Vec<String> {
        let mut missing = Vec::new();
        if !contains_ci(text, &self.sha) {
            missing.push(format!("SHA {}", self.sha));
        }
        for p in &self.parents {
            if !contains_ci(text, &p[..p.len().min(7)]) {
                missing.push(format!("parent {}", &p[..p.len().min(12)]));
            }
        }
        if let Some(b) = &self.branch
            && !text.contains(b.as_str())
        {
            missing.push(format!("branch {b}"));
        }
        let name = self
            .author
            .split(" <")
            .next()
            .unwrap_or(&self.author)
            .trim();
        if !name.is_empty() && !text.contains(name) {
            missing.push(format!("author {name}"));
        }
        for f in self.files.iter().take(MAX_FILES) {
            if !text.contains(f.path.as_str()) {
                missing.push(format!("file {}", f.path));
            }
        }
        for g in gates {
            if !text.contains(g.as_str()) {
                missing.push(format!("gate `{g}`"));
            }
        }
        missing
    }

    /// Tags for a receipt: `commit:<sha>`, its short form, the branch.
    pub fn tags(&self, auto: bool) -> Vec<String> {
        let mut tags = vec![
            format!("commit:{}", self.sha),
            format!("commit:{}", &self.sha[..self.sha.len().min(7)]),
            RECEIPT_TAG.to_string(),
        ];
        if auto {
            tags.push(AUTO_TAG.to_string());
        }
        if let Some(b) = &self.branch {
            tags.push(format!("branch:{b}"));
        }
        tags
    }

    /// The machine receipt body. `gate_output` is what the caller captured
    /// (a post-commit hook has none unless the owner points it at a file).
    pub fn receipt_text(&self, gate_output: Option<&str>) -> String {
        let parents = if self.parents.is_empty() {
            "none (root commit)".to_string()
        } else {
            self.parents.join(", ")
        };
        let gates = gate_output
            .map(str::trim)
            .filter(|g| !g.is_empty())
            .map(|g| g.chars().take(4000).collect::<String>())
            .unwrap_or_else(|| "not captured by the git hook".to_string());
        format!(
            "COMMIT RECEIPT, written by the Vestige post-commit hook.\n\
             SHA: {}\nParent(s): {}\nBranch: {}\nAuthor: {}\nSubject: {}\n\
             Files (git diff -U0 new-side ranges):\n{}\nGate output: {}",
            self.sha,
            parents,
            self.branch.as_deref().unwrap_or("(detached HEAD)"),
            self.author,
            self.subject,
            self.file_lines()
                .iter()
                .map(|l| format!("- {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
            gates,
        )
    }
}

/// Added lines of the staged change, per file, for the pre-commit secret
/// scan. Lines are returned as written; callers never print them.
pub fn staged_added_lines(repo: &Path) -> Vec<(String, String)> {
    let diff = git(
        repo,
        &["diff", "--cached", "-U0", "--no-ext-diff", "--no-color"],
    )
    .unwrap_or_default();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut current = String::new();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            current = rest.strip_prefix("b/").unwrap_or("").to_string();
            continue;
        }
        if line.starts_with("+") && !current.is_empty() {
            match out.last_mut() {
                Some((p, body)) if *p == current => {
                    body.push_str(&line[1..]);
                    body.push('\n');
                }
                _ => out.push((current.clone(), format!("{}\n", &line[1..]))),
            }
        }
    }
    out
}

/// Where a git hook reads and writes: the server that owns the store, or the
/// log opened in this process when no server holds it.
pub enum ReceiptStore {
    Owner(std::path::PathBuf),
    Local(std::sync::Arc<vestige_core::Storage>),
}

fn collect_ids(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, x) in map {
                if matches!(k.as_str(), "id" | "nodeId")
                    && let Some(s) = x.as_str()
                    && !out.iter().any(|o| o == s)
                {
                    out.push(s.to_string());
                }
                if !matches!(k.as_str(), "neighbors" | "edges" | "links") {
                    collect_ids(x, out);
                }
            }
        }
        serde_json::Value::Array(rows) => rows.iter().for_each(|r| collect_ids(r, out)),
        _ => {}
    }
}

impl ReceiptStore {
    /// Save one memory through the same gate `remember` uses.
    pub fn remember(&self, args: serde_json::Value) -> Result<serde_json::Value, String> {
        match self {
            ReceiptStore::Owner(dir) => {
                crate::attach::call_tool_blocking(dir, "remember", args).map_err(|e| e.to_string())
            }
            ReceiptStore::Local(storage) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
                    crate::cognitive::CognitiveEngine::new(),
                ));
                rt.block_on(crate::tools::smart_ingest::execute(
                    storage,
                    &cognitive,
                    Some(args),
                ))
            }
        }
    }

    /// Ids of the memories carrying an exact tag.
    pub fn tag_ids(&self, tag: &str) -> Vec<String> {
        match self {
            ReceiptStore::Owner(dir) => {
                let args = serde_json::json!({"handle": tag, "detail_level": "brief", "limit": 20});
                let mut out = Vec::new();
                if let Ok(v) = crate::attach::call_tool_blocking(dir, "recall", args) {
                    collect_ids(&v, &mut out);
                }
                out
            }
            ReceiptStore::Local(storage) => storage.resolve_handle(tag).ids,
        }
    }
}

/// `vestige hook post-commit`: write the machine commit receipt into Strata.
pub fn post_commit(
    store: &ReceiptStore,
    repo: &Path,
    gate_output: Option<&str>,
) -> Result<(CommitFacts, String), String> {
    let facts = read(repo, "HEAD").ok_or_else(|| "git could not read HEAD".to_string())?;
    let args = serde_json::json!({
        "content": facts.receipt_text(gate_output),
        "tags": facts.tags(true),
        "node_type": "event",
        "source": format!("vestige hook post-commit {}", facts.short()),
    });
    let value = store.remember(args)?;
    let mut ids = Vec::new();
    collect_ids(&value, &mut ids);
    let id = ids
        .into_iter()
        .next()
        .ok_or_else(|| format!("remember returned no id: {value}"))?;
    Ok((facts, id))
}

/// Open Operator STOPs for this repository: a STOP whose action has no
/// later ALLOW in the receipts window.
pub fn open_stops(repo: &Path, window: &str) -> Vec<String> {
    let cwd = repo.display().to_string();
    let Ok(out) = crate::tools::precedent::execute_wide(
        Some(serde_json::json!({"since": window, "cwd": cwd, "decision": "all", "limit": 500})),
        500,
    ) else {
        return Vec::new();
    };
    let items = out
        .get("items")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    // Items arrive newest first. The newest decision per action wins.
    let mut seen: Vec<String> = Vec::new();
    let mut open = Vec::new();
    for item in items {
        let key = item
            .get("action_digest")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                item.get("action_preview")
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("")
            .to_string();
        if key.is_empty() || seen.contains(&key) {
            continue;
        }
        seen.push(key);
        if item.get("decision").and_then(serde_json::Value::as_str) == Some("STOP") {
            let rule = item
                .get("rule")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("STOP");
            let digest = item
                .get("digest")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            open.push(format!(
                "{rule} (recall handle op-receipt-{})",
                &digest[..digest.len().min(16)]
            ));
        }
    }
    open
}

fn staged_tree(repo: &Path) -> Option<String> {
    git(repo, &["write-tree"])
        .map(|t| t.trim().to_string())
        .filter(|t| is_hex(t))
}

/// `vestige hook pre-commit`: every reason the staged change may not land.
/// Secret findings name the pattern and file only, never the value.
pub fn pre_commit(
    store: Option<&ReceiptStore>,
    data_dir: &Path,
    repo: &Path,
    window: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (path, text) in staged_added_lines(repo) {
        let mut kinds: Vec<&'static str> = Vec::new();
        for f in vestige_core::scan_secrets(&text) {
            if f.blocks_ingestion() && !kinds.contains(&f.kind.as_str()) {
                kinds.push(f.kind.as_str());
            }
        }
        for kind in kinds {
            problems.push(format!("secret scan: a {kind} is staged in {path}"));
        }
    }
    for stop in open_stops(repo, window) {
        problems.push(format!("open Operator stop: {stop}"));
    }
    let cfg = crate::tools::reflex::config_for(data_dir, Some(repo));
    if !cfg.commands.is_empty() {
        let tree = staged_tree(repo);
        let witnessed = match (store, tree.as_deref()) {
            (Some(store), Some(t)) => !store.tag_ids(&format!("tree-{t}")).is_empty(),
            _ => false,
        };
        if !witnessed {
            problems.push(format!(
                "no prove witness for the staged tree {}: run one of [{}] on exactly this change first",
                tree.as_deref().unwrap_or("?"),
                cfg.commands.join(", ")
            ));
        }
    }
    problems
}

const HOOK_MARK: &str = "# vestige-git-hook v1";

fn hook_script(kind: &str, exe: &Path) -> String {
    let exe = exe.display().to_string().replace('\'', "'\\''");
    let tail = match kind {
        // Fail closed: a commit cannot land without Vestige checking it.
        "pre-commit" => {
            "if [ ! -x \"$VESTIGE_BIN\" ]; then\n  echo \"vestige pre-commit: $VESTIGE_BIN not found; commit refused (git commit --no-verify skips it)\" >&2\n  exit 1\nfi\nexec \"$VESTIGE_BIN\" hook pre-commit --repo \"$(git rev-parse --show-toplevel)\"\n"
        }
        // A post-commit hook cannot undo a commit; it records and never fails.
        _ => {
            "[ -x \"$VESTIGE_BIN\" ] || exit 0\n\"$VESTIGE_BIN\" hook post-commit --repo \"$(git rev-parse --show-toplevel)\" || true\n"
        }
    };
    format!(
        "#!/bin/sh\n{HOOK_MARK} ({kind}), installed by `vestige init --git-hooks`.\nVESTIGE_DEFAULT='{exe}'\nVESTIGE_BIN=\"${{VESTIGE_BIN:-$VESTIGE_DEFAULT}}\"\n{tail}"
    )
}

/// Install `pre-commit` and `post-commit` into the repository's hooks
/// directory (`core.hooksPath` respected). A foreign hook is kept unless
/// `force`, which moves it to `<name>.pre-vestige`. Installing twice is a
/// no-op.
pub fn install_git_hooks(repo: &Path, exe: &Path, force: bool) -> Result<Vec<String>, String> {
    let hooks = git(repo, &["rev-parse", "--git-path", "hooks"])
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| format!("{} is not a git repository", repo.display()))?;
    let hooks = if Path::new(&hooks).is_absolute() {
        std::path::PathBuf::from(hooks)
    } else {
        repo.join(hooks)
    };
    std::fs::create_dir_all(&hooks).map_err(|e| e.to_string())?;
    let mut done = Vec::new();
    for kind in ["pre-commit", "post-commit"] {
        let path = hooks.join(kind);
        let body = hook_script(kind, exe);
        if let Ok(existing) = std::fs::read_to_string(&path) {
            if existing == body {
                done.push(format!("{} unchanged", path.display()));
                continue;
            }
            if !existing.contains(HOOK_MARK) {
                if !force {
                    return Err(format!(
                        "{} already exists and is not a Vestige hook; rerun with --force to move it to {kind}.pre-vestige",
                        path.display()
                    ));
                }
                std::fs::rename(&path, hooks.join(format!("{kind}.pre-vestige")))
                    .map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(&path, &body).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| e.to_string())?;
        }
        done.push(format!("{} installed", path.display()));
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_zero_ranges_are_new_side_inclusive() {
        let diff = "diff --git a/x.rs b/x.rs\n--- a/x.rs\n+++ b/x.rs\n@@ -3,0 +4,2 @@\n+a\n+b\n@@ -9 +11 @@\n-c\n+d\ndiff --git a/gone.rs b/gone.rs\n--- a/gone.rs\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n";
        let files = parse_unified_zero(diff);
        assert_eq!(files[0].path, "x.rs");
        assert_eq!(files[0].ranges, vec![(4, 5), (11, 11)]);
        assert_eq!(files[1].path, "gone.rs");
        assert_eq!(files[1].ranges, vec![(0, 0)]);
    }

    #[test]
    fn a_receipt_must_name_every_fact() {
        let facts = CommitFacts {
            sha: "a".repeat(40),
            parents: vec!["b".repeat(40)],
            branch: Some("feat/x".into()),
            author: "Sam Valladares <s@example.com>".into(),
            subject: "s".into(),
            files: vec![FileRanges {
                path: "src/lib.rs".into(),
                ranges: vec![(1, 2)],
            }],
        };
        let gates = vec!["cargo test".to_string()];
        let missing = facts.missing_in(&format!("commit {}", "a".repeat(40)), &gates);
        assert_eq!(missing.len(), 5, "{missing:?}");
        let full = format!(
            "SHA {} parent {} branch feat/x author Sam Valladares files src/lib.rs:1-2 gates: cargo test ok",
            "a".repeat(40),
            "bbbbbbb"
        );
        assert!(facts.missing_in(&full, &gates).is_empty());
        assert!(facts.receipt_text(None).contains("src/lib.rs:1-2"));
        assert!(facts.tags(true).contains(&AUTO_TAG.to_string()));
    }

    #[test]
    fn git_hooks_install_once_and_keep_a_foreign_hook() {
        let dir = tempfile::TempDir::new().unwrap();
        let ok = Command::new("git")
            .args(["init", "-q"])
            .arg(dir.path())
            .status()
            .unwrap();
        assert!(ok.success());
        let exe = Path::new("/usr/local/bin/vestige");
        let first = install_git_hooks(dir.path(), exe, false).unwrap();
        assert!(first.iter().all(|l| l.ends_with("installed")), "{first:?}");
        let again = install_git_hooks(dir.path(), exe, false).unwrap();
        assert!(again.iter().all(|l| l.ends_with("unchanged")), "{again:?}");
        let hooks = dir.path().join(".git/hooks");
        let pre = std::fs::read_to_string(hooks.join("pre-commit")).unwrap();
        assert!(pre.contains("hook pre-commit") && pre.contains("exit 1"));
        std::fs::write(hooks.join("post-commit"), "#!/bin/sh\necho mine\n").unwrap();
        assert!(install_git_hooks(dir.path(), exe, false).is_err());
        install_git_hooks(dir.path(), exe, true).unwrap();
        assert!(hooks.join("post-commit.pre-vestige").exists());
    }
}
