//! Catalog gaps for `causal_walk` and `ingest_repo`.
//!
//! Stores and checkouts live under `~/Downloads`, never a temp directory.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use vestige_core::{ConnectionRecord, IngestInput, Storage};

fn work(label: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join("Downloads/vestige-blind-spots-975d")
        .join(format!("{label}-{n}"));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn open_store(label: &str) -> (Arc<Storage>, PathBuf) {
    let root = work(label);
    let storage = crate::strata_memory::open(root.join("log")).unwrap();
    (storage, root)
}

fn git(root: &Path, args: &[&str], date: &str) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
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

fn init_repo(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    git(root, &["init", "-q", "-b", "main"], "2024-01-01T00:00:00Z");
}

fn write_rel(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

fn commit_file(root: &Path, rel: &str, body: &str, message: &str, date: &str) -> String {
    write_rel(root, rel, body);
    git(root, &["add", "-A"], date);
    git(root, &["commit", "-q", "-m", message], date);
    git(root, &["rev-parse", "HEAD"], date)
}

async fn ingest_repo(storage: &Arc<Storage>, root: &Path, scope: &str, codebase: &str) -> Value {
    crate::tools::repo_ingest::execute(
        storage,
        crate::tools::repo_ingest::Request {
            repo_path: root.to_path_buf(),
            codebase: Some(codebase.into()),
            scope: Some(scope.into()),
            rev: Some("HEAD".into()),
            since: None,
            until: None,
            limit: Some(50),
            dry_run: false,
            budget: None,
        },
    )
    .await
    .unwrap()
}

fn id_for_sha(storage: &Arc<Storage>, scope: &str, sha: &str) -> String {
    let tag = crate::tools::repo_ingest::commit_tag(sha);
    storage
        .current_code_context_nodes("event", Some(&tag), scope, 5)
        .unwrap()
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("missing commit {sha}"))
        .id
}

fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
    let now = Utc::now();
    storage
        .save_connection(&ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .unwrap();
}

fn put_commit(storage: &Arc<Storage>, scope: &str, sha: &str, when: &str) -> String {
    storage
        .ingest_in_scope(
            IngestInput {
                content: format!("commit {sha}"),
                node_type: "event".into(),
                tags: vec![format!("commit:{sha}"), "git-commit".into()],
                valid_from: Some(when.parse().unwrap()),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id
}

fn cause_shas(out: &Value) -> Vec<String> {
    out["causes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["structure"]["sha"].as_str().map(str::to_string))
        .collect()
}

fn cause_by_sha<'a>(out: &'a Value, sha: &str) -> &'a Value {
    out["causes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["structure"]["sha"].as_str() == Some(sha))
        .unwrap_or_else(|| panic!("missing cause {sha} in {out}"))
}

#[tokio::test]
async fn item_12_merge_base_keeps_a_second_parent_descendant() {
    let (storage, root) = open_store("item-12");
    let repo = root.join("repo");
    init_repo(&repo);
    write_rel(&repo, "src/a.rs", "fn a() { let v = 0; }\n");
    git(&repo, &["add", "-A"], "2024-01-01T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "root"],
        "2024-01-01T00:00:00Z",
    );
    let root_sha = git(&repo, &["rev-parse", "HEAD"], "2024-01-01T00:00:00Z");
    write_rel(&repo, "src/a.rs", "fn a() { let v = 1; }\n");
    git(&repo, &["add", "-A"], "2024-02-01T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "good"],
        "2024-02-01T00:00:00Z",
    );
    let good = git(&repo, &["rev-parse", "HEAD"], "2024-02-01T00:00:00Z");
    git(&repo, &["tag", "good"], "2024-02-01T00:00:00Z");
    git(
        &repo,
        &["checkout", "-q", "-b", "side"],
        "2024-03-01T00:00:00Z",
    );
    write_rel(&repo, "src/a.rs", "fn a() { let v = 2; }\n");
    git(&repo, &["add", "-A"], "2024-03-01T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "side"],
        "2024-03-01T00:00:00Z",
    );
    let side = git(&repo, &["rev-parse", "HEAD"], "2024-03-01T00:00:00Z");
    git(&repo, &["checkout", "-q", "main"], "2024-03-02T00:00:00Z");
    git(
        &repo,
        &["merge", "--no-ff", "--no-edit", "side"],
        "2024-03-02T00:00:00Z",
    );
    git(
        &repo,
        &["checkout", "-q", "-b", "ancient", &root_sha],
        "2024-01-15T00:00:00Z",
    );
    write_rel(&repo, "src/off.rs", "fn off() {}\n");
    git(&repo, &["add", "-A"], "2024-01-15T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "before good"],
        "2024-01-15T00:00:00Z",
    );
    let off = git(&repo, &["rev-parse", "HEAD"], "2024-01-15T00:00:00Z");
    git(&repo, &["checkout", "-q", "main"], "2024-04-01T00:00:00Z");
    git(
        &repo,
        &["merge", "--no-ff", "--no-edit", "ancient"],
        "2024-04-01T00:00:00Z",
    );
    write_rel(&repo, "src/a.rs", "fn a() { let v = 3; }\n");
    git(&repo, &["add", "-A"], "2024-05-01T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "bad"],
        "2024-05-01T00:00:00Z",
    );
    let bad = git(&repo, &["rev-parse", "HEAD"], "2024-05-01T00:00:00Z");
    git(&repo, &["tag", "bad"], "2024-05-01T00:00:00Z");

    let scope = "range";
    let good_id = put_commit(&storage, scope, &good, "2024-02-01T00:00:00Z");
    let side_id = put_commit(&storage, scope, &side, "2024-03-01T00:00:00Z");
    let off_id = put_commit(&storage, scope, &off, "2024-01-15T00:00:00Z");
    let bad_id = put_commit(&storage, scope, &bad, "2024-05-01T00:00:00Z");
    for id in [&good_id, &side_id, &off_id, &bad_id] {
        link(&storage, id, "file:src/a.rs", "touched");
    }
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: "failure at src/a.rs".into(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id;
    // The observed revision is not `bad`. The walk drops the failure's own
    // commit, and `broke_in` still has to remain a candidate.
    let observed_id = put_commit(
        &storage,
        scope,
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "2024-06-01T00:00:00Z",
    );
    link(&storage, &failure, &observed_id, "derived_from");

    let out = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [
                {"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": failure},
                {
                    "kind": "version_range",
                    "worked_in": "good",
                    "broke_in": "bad",
                    "repo": repo.display().to_string()
                }
            ]
        })),
    )
    .await
    .unwrap();
    let shas = cause_shas(&out);
    assert!(
        shas.iter().any(|sha| sha == &side),
        "a commit merged from worked_in stays: {out}"
    );
    assert!(shas.iter().any(|sha| sha == &bad), "broke_in stays: {out}");
    assert!(
        !shas.iter().any(|sha| sha == &good),
        "worked_in itself stays out: {out}"
    );
    assert!(
        !shas.iter().any(|sha| sha == &off),
        "a commit whose ancestor is before worked_in stays out: {out}"
    );
    assert!(
        out["range"]["because"]
            .as_str()
            .unwrap()
            .contains("merge-base --is-ancestor"),
        "{out}"
    );
}

#[tokio::test]
async fn item_10_whitespace_or_comment_blame_does_not_win() {
    let (storage, root) = open_store("item-10");
    let repo = root.join("repo");
    init_repo(&repo);
    commit_file(
        &repo,
        "src/ws.rs",
        "fn a() {\n    let v = 1;\n}\n",
        "introduce",
        "2024-01-01T00:00:00Z",
    );
    write_rel(&repo, "src/note.rs", "fn n() {\n    let v = 1;\n}\n");
    git(&repo, &["add", "-A"], "2024-01-01T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "--amend", "--no-edit"],
        "2024-01-01T00:00:00Z",
    );
    let introduced = git(&repo, &["rev-parse", "HEAD"], "2024-01-01T00:00:00Z");
    let whitespace = commit_file(
        &repo,
        "src/ws.rs",
        "fn a() {\n        let v = 1;\n}\n",
        "reindent",
        "2024-02-01T00:00:00Z",
    );
    let comment = commit_file(
        &repo,
        "src/note.rs",
        "fn n() {\n    // note\n    let v = 1;\n}\n",
        "comment",
        "2024-02-02T00:00:00Z",
    );
    let observed = commit_file(
        &repo,
        "src/tail.rs",
        "fn tail() {}\n",
        "observe",
        "2024-03-01T00:00:00Z",
    );
    assert!(crate::tools::repo_ingest::line_change_is_blank_or_comment(
        &repo,
        &whitespace,
        "src/ws.rs",
        2
    ));
    assert!(crate::tools::repo_ingest::line_change_is_blank_or_comment(
        &repo,
        &comment,
        "src/note.rs",
        2
    ));

    let scope = "blame";
    ingest_repo(&storage, &repo, scope, "demo").await;
    let observed_id = id_for_sha(&storage, scope, &observed);
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: "the line moved and nothing else".into(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id;
    link(&storage, &failure, &observed_id, "derived_from");

    let ws = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [{
                "kind": "stack_frame",
                "frame": "src/ws.rs:2",
                "node_id": failure
            }]
        })),
    )
    .await
    .unwrap();
    let ws_cause = cause_by_sha(&ws, &whitespace);
    assert_eq!(ws_cause["structure"]["blameOfLine"], json!(false), "{ws}");
    assert_eq!(
        ws_cause["structure"]["touchedFailingHunk"],
        json!(false),
        "{ws}"
    );
    assert_ne!(
        ws["causes"][0]["structure"]["sha"],
        json!(whitespace),
        "{ws}"
    );
    let intro = cause_by_sha(&ws, &introduced);
    assert_eq!(
        intro["structure"]["touchedFailingHunk"],
        json!(true),
        "{ws}"
    );

    let note = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [{
                "kind": "stack_frame",
                "frame": "src/note.rs:2",
                "node_id": failure
            }]
        })),
    )
    .await
    .unwrap();
    let note_cause = cause_by_sha(&note, &comment);
    assert_eq!(
        note_cause["structure"]["blameOfLine"],
        json!(false),
        "{note}"
    );
    assert_ne!(
        note["causes"][0]["structure"]["sha"],
        json!(comment),
        "{note}"
    );

    let unmarked = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [{
                "kind": "stack_frame",
                "frame": "src/ws.rs",
                "node_id": failure
            }]
        })),
    )
    .await
    .unwrap();
    assert!(
        !unmarked["causes"].as_array().unwrap().is_empty(),
        "{unmarked}"
    );
    assert_eq!(unmarked["needsProve"]["flag"], json!(true), "{unmarked}");
    assert_eq!(
        unmarked["needsProve"]["command"],
        json!("vestige prove"),
        "{unmarked}"
    );
    assert_eq!(
        unmarked["needsProve"]["over"],
        json!("version_range"),
        "{unmarked}"
    );
    assert_eq!(
        unmarked["ranking"]["evidence_status"],
        json!("needs_prove"),
        "{unmarked}"
    );
    assert!(
        unmarked["causes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|cause| cause["rank"].is_null()),
        "{unmarked}"
    );
}

#[tokio::test]
async fn item_22_deletion_records_the_old_side_hunk() {
    let (storage, root) = open_store("item-22");
    let repo = root.join("repo");
    init_repo(&repo);
    commit_file(
        &repo,
        "src/drop.rs",
        "line1\nline2\nline3\nline4\nline5\n",
        "add",
        "2024-01-01T00:00:00Z",
    );
    let deletion = commit_file(
        &repo,
        "src/drop.rs",
        "line1\nline5\n",
        "drop the middle",
        "2024-02-01T00:00:00Z",
    );
    ingest_repo(&storage, &repo, "demo", "demo").await;
    let id = id_for_sha(&storage, "demo", &deletion);
    let edges = storage.get_connections_for_memory(&id).unwrap();
    assert!(
        edges.iter().any(|edge| {
            edge.source_id == id
                && edge.target_id == "hunk:src/drop.rs:2+3"
                && edge.link_type == "touched"
        }),
        "old side 2+3 is the deletion candidate: {edges:?}"
    );
}

#[test]
fn item_20_one_path_prefix_is_a_path_identity() {
    let marked = crate::auto_connect::extract_identities(
        "broke path:packages/quill/src/core/editor.ts",
        &[],
    );
    assert!(
        marked
            .iter()
            .any(|identity| identity.to_string() == "path:packages/quill/src/core/editor.ts"),
        "{marked:?}"
    );
    let doubled = crate::auto_connect::extract_identities("path:path:foo.rs", &[]);
    assert!(
        doubled
            .iter()
            .all(|identity| identity.kind != crate::auto_connect::IdentityKind::Path),
        "a second path: prefix stays: {doubled:?}"
    );
    let lined = crate::auto_connect::extract_identities("path:src/a.rs:12", &[]);
    assert!(
        lined
            .iter()
            .any(|identity| identity.to_string() == "path:src/a.rs"),
        "{lined:?}"
    );
}

#[tokio::test]
async fn item_20_file_anchor_scope_and_walk() {
    let (storage, root) = open_store("item-20");
    let repo = root.join("repo");
    init_repo(&repo);
    let sha = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\n",
        "add",
        "2024-01-01T00:00:00Z",
    );
    ingest_repo(&storage, &repo, "quill-repo", "quill").await;
    let cognitive = Arc::new(Mutex::new(crate::cognitive::CognitiveEngine::new()));

    let refused = crate::tools::smart_ingest::execute(
        &storage,
        &cognitive,
        Some(json!({
            "content": "failure path:src/a.rs in the wrong scope",
            "scope": "other-scope",
            "tags": ["failure", "codebase:quill"],
            "forceCreate": true
        })),
    )
    .await;
    let err = refused.expect_err("a foreign codebase tag is refused");
    assert!(
        err.contains("quill-repo"),
        "the error names the scope that holds the tag: {err}"
    );
    let stored = storage
        .get_all_nodes_in_scope("other-scope", i32::MAX, 0)
        .unwrap();
    assert!(
        stored
            .iter()
            .all(|node| !node.content.contains("wrong scope")),
        "the refused write was not saved: {stored:?}"
    );

    let saved = crate::tools::smart_ingest::execute(
        &storage,
        &cognitive,
        Some(json!({
            "content": "failure path:src/a.rs",
            "scope": "quill-repo",
            "tags": ["failure", "codebase:quill"],
            "forceCreate": true
        })),
    )
    .await
    .unwrap();
    let failure = saved["nodeId"].as_str().unwrap().to_string();
    let edges = storage.get_connections_for_memory(&failure).unwrap();
    assert!(
        edges.iter().any(|edge| {
            edge.source_id == failure
                && edge.target_id == "file:src/a.rs"
                && edge.link_type == "touched"
        }),
        "the failure touches the byte-equal file anchor: {edges:?}"
    );

    let walked = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": "quill-repo",
            "start_points": [{"kind": "logged_write", "node_id": failure}]
        })),
    )
    .await
    .unwrap();
    let shas = cause_shas(&walked);
    assert!(
        shas.iter().any(|found| found == &sha),
        "the walk starts from the file anchor without a stack frame: {walked}"
    );

    let batch = crate::tools::smart_ingest::execute(
        &storage,
        &cognitive,
        Some(json!({
            "items": [{
                "content": "batch failure path:src/a.rs elsewhere",
                "scope": "batch-scope",
                "tags": ["failure", "codebase:quill"],
                "forceCreate": true
            }]
        })),
    )
    .await
    .unwrap();
    assert_eq!(batch["results"][0]["status"], json!("error"), "{batch}");
    assert!(
        batch["results"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("quill-repo"),
        "{batch}"
    );
}

#[ignore]
#[test]
fn catalog_item_4_dependency_bump_without_the_app_file() {
    let diff = "\
diff --git a/pnpm-lock.yaml b/pnpm-lock.yaml
--- a/pnpm-lock.yaml
+++ b/pnpm-lock.yaml
@@ -1,1 +1,1 @@
-  left-pad@1.0.0:
+  left-pad@1.0.1:
";
    let bumps = vestige_core::advanced::git_records::lock_bumps_from_diff(diff);
    assert!(
        bumps.iter().any(|bump| bump.package == "left-pad"),
        "catalog 4: a lockfile already on disk records the bump: {bumps:?}"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_13_squash_merges() {
    let (storage, root) = open_store("item-13");
    let repo = root.join("repo");
    init_repo(&repo);
    commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\n",
        "base",
        "2024-01-01T00:00:00Z",
    );
    git(
        &repo,
        &["checkout", "-q", "-b", "feature"],
        "2024-02-01T00:00:00Z",
    );
    commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 2; }\n",
        "feature",
        "2024-02-01T00:00:00Z",
    );
    let feature = git(&repo, &["rev-parse", "HEAD"], "2024-02-01T00:00:00Z");
    git(&repo, &["checkout", "-q", "main"], "2024-03-01T00:00:00Z");
    git(
        &repo,
        &["merge", "--squash", "feature"],
        "2024-03-01T00:00:00Z",
    );
    git(
        &repo,
        &["commit", "-q", "-m", "squash"],
        "2024-03-01T00:00:00Z",
    );
    ingest_repo(&storage, &repo, "demo", "demo").await;
    let edges = storage.get_all_connections().unwrap();
    assert!(
        edges
            .iter()
            .any(|edge| edge.target_id.contains(&feature) || edge.source_id.contains(&feature)),
        "catalog 13: the squash links to the pre-squash head {feature}"
    );
}

#[ignore]
#[test]
fn catalog_item_28_author_time_after_the_failure() {
    assert!(
        vestige_core::advanced::git_records::GIT_LOG_PRETTY.contains("%cI"),
        "catalog 28: committer time is stored beside author time"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_30_conflicted_cherry_pick() {
    let (storage, root) = open_store("item-30");
    let repo = root.join("repo");
    init_repo(&repo);
    commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\n",
        "base",
        "2024-01-01T00:00:00Z",
    );
    let original = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 2; }\n",
        "change a",
        "2024-02-01T00:00:00Z",
    );
    git(
        &repo,
        &["checkout", "-q", "-b", "other", "HEAD~1"],
        "2024-03-01T00:00:00Z",
    );
    commit_file(
        &repo,
        "src/b.rs",
        "fn b() {}\n",
        "other file",
        "2024-03-01T00:00:00Z",
    );
    write_rel(&repo, "src/a.rs", "fn a() { let v = 2; }\n");
    write_rel(&repo, "src/c.rs", "fn c() {}\n");
    git(&repo, &["add", "-A"], "2024-03-02T00:00:00Z");
    git(
        &repo,
        &["commit", "-q", "-m", "same line, extra file"],
        "2024-03-02T00:00:00Z",
    );
    let cherry = git(&repo, &["rev-parse", "HEAD"], "2024-03-02T00:00:00Z");
    ingest_repo(&storage, &repo, "demo", "demo").await;
    git(&repo, &["checkout", "-q", "main"], "2024-03-03T00:00:00Z");
    ingest_repo(&storage, &repo, "demo", "demo").await;
    let original_id = id_for_sha(&storage, "demo", &original);
    let cherry_id = id_for_sha(&storage, "demo", &cherry);
    let edges = storage.get_all_connections().unwrap();
    assert!(
        edges.iter().any(|edge| {
            (edge.source_id == cherry_id && edge.target_id == original_id)
                || (edge.source_id == original_id && edge.target_id == cherry_id)
        }),
        "catalog 30: a per-file patch match links the cherry even when the whole commit differs"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_32_revert_of_a_revert() {
    let (storage, _root) = open_store("item-32");
    let scope = "revert";
    let original = put_commit(&storage, scope, &"a".repeat(40), "2024-01-01T00:00:00Z");
    let revert = put_commit(&storage, scope, &"b".repeat(40), "2024-03-01T00:00:00Z");
    let unrevert = put_commit(&storage, scope, &"c".repeat(40), "2024-04-01T00:00:00Z");
    let observed = put_commit(&storage, scope, &"d".repeat(40), "2024-02-01T00:00:00Z");
    for id in [&original, &revert, &unrevert, &observed] {
        link(&storage, id, "file:src/a.rs", "touched");
    }
    link(&storage, &revert, &original, "corrects");
    link(&storage, &unrevert, &revert, "corrects");
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: "failure".into(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id;
    link(&storage, &failure, &observed, "derived_from");
    let out = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [{"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": failure}]
        })),
    )
    .await
    .unwrap();
    let row = out["causes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|cause| cause["id"] == original)
        .expect("original commit");
    assert_eq!(
        row["structure"]["revertedAfterFailure"],
        json!(false),
        "catalog 32: a revert that was itself reverted does not restore the blame: {out}"
    );
}

#[ignore]
#[test]
fn catalog_item_43_submodule_gitlink() {
    let parent = "a".repeat(40);
    let before = "b".repeat(40);
    let after = "c".repeat(40);
    let raw = format!(
        "\u{1e}{parent}\u{1f}2024-01-01T00:00:00+00:00\u{1f}\u{1f}bump the submodule\u{1f}\u{1d}\
diff --git a/vendor/lib b/vendor/lib\n\
index {before}..{after} 160000\n\
--- a/vendor/lib\n\
+++ b/vendor/lib\n\
@@ -1 +1 @@\n\
-Subproject commit {before}\n\
+Subproject commit {after}\n"
    );
    let commits = vestige_core::advanced::git_records::parse_git_log(&raw);
    let text = vestige_core::advanced::git_records::record_content(&commits[0]);
    assert!(
        text.contains(&after),
        "catalog 43: the gitlink sha is an anchor, not only the path: {text}"
    );
}

#[ignore]
#[test]
fn catalog_item_44_lfs_pointer() {
    let sha = "d".repeat(40);
    let oid = "e".repeat(64);
    let raw = format!(
        "\u{1e}{sha}\u{1f}2024-01-01T00:00:00+00:00\u{1f}\u{1f}add a pointer\u{1f}\u{1d}\
diff --git a/assets/blob.bin b/assets/blob.bin\n\
--- /dev/null\n\
+++ b/assets/blob.bin\n\
@@ -0,0 +1,3 @@\n\
+version https://git-lfs.github.com/spec/v1\n\
+oid sha256:{oid}\n\
+size 4\n"
    );
    let commits = vestige_core::advanced::git_records::parse_git_log(&raw);
    let text = vestige_core::advanced::git_records::record_content(&commits[0]);
    assert!(
        text.contains(&oid),
        "catalog 44: the LFS oid already in the blob is an anchor: {text}"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_51_shallow_checkout() {
    let (storage, root) = open_store("item-51");
    let repo = root.join("repo");
    init_repo(&repo);
    commit_file(
        &repo,
        "src/a.rs",
        "fn a() {}\n",
        "one",
        "2024-01-01T00:00:00Z",
    );
    commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\n",
        "two",
        "2024-02-01T00:00:00Z",
    );
    let shallow = root.join("shallow");
    git(
        &root,
        &["clone", "--depth", "1", repo.to_str().unwrap(), "shallow"],
        "2024-02-01T00:00:00Z",
    );
    let out = ingest_repo(&storage, &shallow, "demo", "demo").await;
    assert_eq!(
        out["shallow"],
        json!(true),
        "catalog 51: a shallow checkout is recorded from the local git dir: {out}"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_53_one_line_on_every_path() {
    let (storage, root) = open_store("item-53");
    let repo = root.join("repo");
    init_repo(&repo);
    let introduced = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\nfn b() { let v = 1; }\n",
        "both lines",
        "2024-01-01T00:00:00Z",
    );
    let second = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\nfn b() { let v = 2; }\n",
        "second line",
        "2024-02-01T00:00:00Z",
    );
    let observed = commit_file(
        &repo,
        "src/tail.rs",
        "fn tail() {}\n",
        "observe",
        "2024-03-01T00:00:00Z",
    );
    ingest_repo(&storage, &repo, "demo", "demo").await;
    let observed_id = id_for_sha(&storage, "demo", &observed);
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: "two frames".into(),
                ..Default::default()
            },
            "demo",
        )
        .unwrap()
        .id;
    link(&storage, &failure, &observed_id, "derived_from");
    let out = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": "demo",
            "start_points": [
                {"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": failure},
                {"kind": "stack_frame", "frame": "src/a.rs:2", "node_id": failure}
            ]
        })),
    )
    .await
    .unwrap();
    let _ = introduced;
    let row = cause_by_sha(&out, &second);
    assert_eq!(
        row["structure"]["blameOfLine"],
        json!(true),
        "catalog 53: the second frame keeps its own line: {out}"
    );
}

#[ignore]
#[tokio::test]
async fn catalog_item_117_untested_sha_gap() {
    let (storage, root) = open_store("item-117");
    let repo = root.join("repo");
    init_repo(&repo);
    let good = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 1; }\n",
        "good",
        "2024-01-01T00:00:00Z",
    );
    let middle = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 2; }\n",
        "middle",
        "2024-02-01T00:00:00Z",
    );
    let bad = commit_file(
        &repo,
        "src/a.rs",
        "fn a() { let v = 3; }\n",
        "bad",
        "2024-03-01T00:00:00Z",
    );
    git(&repo, &["tag", "good"], "2024-01-01T00:00:00Z");
    git(&repo, &["tag", "bad"], "2024-03-01T00:00:00Z");
    let scope = "gap";
    for (sha, when) in [
        (&good, "2024-01-01T00:00:00Z"),
        (&middle, "2024-02-01T00:00:00Z"),
        (&bad, "2024-03-01T00:00:00Z"),
    ] {
        let id = put_commit(&storage, scope, sha, when);
        link(&storage, &id, "file:src/a.rs", "touched");
    }
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: "untested middle".into(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
        .id;
    link(
        &storage,
        &failure,
        &put_commit(&storage, scope, &bad, "2024-03-01T00:00:00Z"),
        "derived_from",
    );
    let out = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "start_points": [
                {"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": failure},
                {"kind": "version_range", "worked_in": "good", "broke_in": "bad", "repo": repo.display().to_string()}
            ]
        })),
    )
    .await
    .unwrap();
    let untested = out["range"]["untested"]
        .as_array()
        .unwrap_or_else(|| panic!("{out}"));
    assert!(
        untested
            .iter()
            .any(|sha| sha.as_str() == Some(middle.as_str())),
        "catalog 117: the sha with no recorded pass is the gap: {out}"
    );
}

#[ignore]
#[test]
fn catalog_item_63_issue_trailer() {
    let targets = vestige_core::advanced::git_records::fixes_targets("Fixes: slab/quill#4509\n");
    assert_eq!(
        targets,
        vec!["slab/quill#4509".to_string()],
        "catalog 63: an owner/repo#n trailer is the issue identity"
    );
}
