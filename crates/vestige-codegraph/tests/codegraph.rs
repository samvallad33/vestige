use std::path::Path;
use std::process::Command;

use tempfile::TempDir;
use vestige_codegraph::{CodeGraph, CongruenceRule, GitWalkOptions};

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn init_scratch_repo() -> TempDir {
    let tmp = TempDir::new().expect("tempdir");
    git(tmp.path(), &["init", "-q"]);
    git(tmp.path(), &["config", "user.name", "Sam Valladares"]);
    git(
        tmp.path(),
        &["config", "user.email", "samvalladares21@gmail.com"],
    );
    std::fs::create_dir_all(tmp.path().join("src")).expect("create src dir");
    tmp
}

#[test]
fn test_rename_move_shape_divergence_untracked_and_deterministic_digest() {
    let tmp = init_scratch_repo();
    let repo = tmp.path();

    // Commit 1: `fn add` in src/math.rs
    std::fs::write(
        repo.join("src/math.rs"),
        "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
    )
    .expect("write math.rs v1");
    git(repo, &["add", "src/math.rs"]);
    git(repo, &["commit", "-q", "-m", "feat: add fn add"]);

    // Commit 2: Rename `fn add` -> `fn sum` with identical body.
    std::fs::write(
        repo.join("src/math.rs"),
        "pub fn sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
    )
    .expect("write math.rs v2");
    git(repo, &["add", "src/math.rs"]);
    git(repo, &["commit", "-q", "-m", "refactor: rename add to sum"]);

    // Commit 3: Add `fn altered` with a DIFFERENT body (must NOT falsely unify with `sum`).
    std::fs::write(
        repo.join("src/math.rs"),
        "pub fn sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n\npub fn altered(a: i32, b: i32) -> i32 {\n    a + b + 1\n}\n",
    )
    .expect("write math.rs v3");
    git(repo, &["add", "src/math.rs"]);
    git(repo, &["commit", "-q", "-m", "feat: add fn altered"]);

    let graph_before_move = CodeGraph::index_repo(
        repo,
        GitWalkOptions {
            hash_pii: true,
            max_commits: None,
        },
    )
    .expect("index repo before move");

    // Verify PII hashing on author email.
    assert!(graph_before_move
        .commits
        .iter()
        .all(|c| c.author_email.starts_with("blake3:")));

    // `why` on `fn sum` (line 2) traces back across BOTH rename commit and original `fn add` commit.
    let why_sum = graph_before_move.why("src/math.rs", 2);
    assert!(why_sum.unresolved_reason.is_none());
    assert_eq!(
        why_sum.symbol.as_ref().map(|s| s.name.as_str()),
        Some("sum")
    );
    assert_eq!(why_sum.commits.len(), 2);
    assert!(why_sum
        .commits
        .iter()
        .any(|hop| matches!(hop.rule, Some(CongruenceRule::ExactShapeMatch { .. }))));

    // `fn altered` (line 6) has a different body shape, so it must NOT unify with `add`/`sum`.
    let why_altered = graph_before_move.why("src/math.rs", 6);
    assert!(why_altered.unresolved_reason.is_none());
    assert_eq!(
        why_altered.symbol.as_ref().map(|s| s.name.as_str()),
        Some("altered")
    );
    assert_eq!(why_altered.commits.len(), 1);

    // Commit 4: Move `src/math.rs` -> `src/calc.rs`.
    git(repo, &["mv", "src/math.rs", "src/calc.rs"]);
    git(repo, &["commit", "-q", "-m", "refactor: move math to calc"]);

    let graph_a = CodeGraph::index_repo(
        repo,
        GitWalkOptions {
            hash_pii: false,
            max_commits: None,
        },
    )
    .expect("index repo A");
    let graph_b = CodeGraph::index_repo(
        repo,
        GitWalkOptions {
            hash_pii: false,
            max_commits: None,
        },
    )
    .expect("index repo B");

    // Deterministic digest across repeated builds.
    assert_eq!(graph_a.digest(), graph_b.digest());

    // `why` on `src/calc.rs` line 2 still traces back across move + rename + initial add.
    let why_moved = graph_a.why("src/calc.rs", 2);
    assert!(why_moved.unresolved_reason.is_none());
    assert_eq!(
        why_moved.symbol.as_ref().map(|s| s.name.as_str()),
        Some("sum")
    );
    assert!(
        why_moved.commits.len() >= 3,
        "expected >= 3 commits across move + rename + initial add, got {}",
        why_moved.commits.len()
    );

    // Untracked file returns explicit `unresolved_reason` and zero commits.
    std::fs::write(repo.join("src/untracked.rs"), "pub fn ghost() {}\n").expect("write untracked");
    let why_untracked = graph_a.why("src/untracked.rs", 1);
    assert!(why_untracked.commits.is_empty());
    assert!(why_untracked.unresolved_reason.is_some());
}
