//! Every `vestige` command and flag, invoked as a real process.

use std::time::Duration;

use super::support::*;

fn help(args: &[&str]) -> String {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(&owned, home.path(), Duration::from_secs(20));
    format!("{}{}", out.stdout, out.stderr)
}

#[test]
fn cli_help_lists_every_command_and_flag() {
    let root = help(&["--help"]);
    let commands = [
        "stats",
        "health",
        "consolidate",
        "upgrade",
        "update",
        "sandwich",
        "restore",
        "backup",
        "migrate-to-strata",
        "export",
        "portable-export",
        "portable-import",
        "sync",
        "gc",
        "dashboard",
        "ingest",
        "ingest-git",
        "scan-secrets",
        "backfill",
        "causal-walk",
        "recall",
        "compose",
        "project",
        "serve",
        "selftest",
        "forgotten-lesson",
    ];
    let mut missing_bits = Vec::new();
    for cmd in commands {
        if !root.contains(cmd) {
            missing_bits.push(format!("command {cmd}"));
        }
    }
    for flag in ["--data-dir", "--help", "--version"] {
        if !root.contains(flag) {
            missing_bits.push(format!("global {flag}"));
        }
    }
    let per_command: &[(&str, &[&str])] = &[
        ("stats", &["--tagging", "--states"]),
        ("upgrade", &["--dry-run"]),
        (
            "update",
            &[
                "--version",
                "--install-dir",
                "--dry-run",
                "--no-sandwich",
                "--sandwich-companion",
                "--force",
                "--enable-preflight",
                "--enable-sandwich",
                "--enable-sanhedrin",
                "--with-launchd",
                "--include-memory-loader",
                "--sanhedrin-endpoint",
                "--sanhedrin-model",
            ],
        ),
        ("export", &["--format", "--tags", "--since"]),
        ("portable-import", &["--merge"]),
        ("sync", &["--cloud", "--endpoint"]),
        (
            "gc",
            &["--min-retention", "--max-age-days", "--dry-run", "--yes"],
        ),
        ("dashboard", &["--port", "--no-open"]),
        (
            "ingest",
            &[
                "--tags",
                "--node-type",
                "--source",
                "--ago-days",
                "--created-at",
                "--allow-secrets",
            ],
        ),
        (
            "ingest-git",
            &["--since", "--until", "--max-commits", "--json"],
        ),
        (
            "scan-secrets",
            &["--include-suspected", "--json", "--limit"],
        ),
        (
            "backfill",
            &[
                "--failure-id",
                "--manual",
                "--lookback-days",
                "--no-promote",
                "--contrast",
                "--json",
                "--git-repo",
                "--worked-in",
                "--broke-in",
                "--why-not",
            ],
        ),
        (
            "causal-walk",
            &[
                "--failing-test",
                "--stack-frame",
                "--ci-run",
                "--logged-write",
                "--git-repo",
                "--worked-in",
                "--broke-in",
                "--lookback-days",
                "--no-promote",
                "--scope",
                "--json",
            ],
        ),
        ("recall", &["--handle", "--json"]),
        ("compose", &["--limit", "--tags", "--json"]),
        (
            "project",
            &[
                "--out",
                "--format",
                "--scope",
                "--min-retention",
                "--max-items",
                "--write",
                "--json",
            ],
        ),
        ("serve", &["--port", "--dashboard", "--dashboard-port"]),
        ("forgotten-lesson", &["--scope", "--json"]),
        (
            "migrate-to-strata",
            &["--from", "--to", "--dry-run", "--accept-wal-snapshot"],
        ),
    ];
    for (cmd, flags) in per_command {
        let text = help(&[cmd, "--help"]);
        for flag in *flags {
            if !text.contains(flag) {
                missing_bits.push(format!("{cmd} {flag}"));
            }
        }
    }
    let sandwich = help(&["sandwich", "install", "--help"]);
    for flag in [
        "--version",
        "--force",
        "--enable-preflight",
        "--enable-sandwich",
        "--enable-sanhedrin",
        "--with-launchd",
        "--include-memory-loader",
        "--sanhedrin-endpoint",
        "--sanhedrin-model",
    ] {
        if !sandwich.contains(flag) {
            missing_bits.push(format!("sandwich install {flag}"));
        }
    }
    assert!(
        missing_bits.is_empty(),
        "CLI help is missing: {}",
        missing_bits.join(", ")
    );
}

#[test]
fn cli_migrate_to_strata_advertises_dry_run_and_rollback() {
    let text = help(&["migrate-to-strata", "--help"]);
    assert!(
        text.contains("--dry-run"),
        "migrate-to-strata help has no --dry-run: {text}"
    );
    if !text.contains("--rollback") {
        missing(
            "`vestige migrate-to-strata --rollback` is not on this head. \
             The hidden fallback must support --dry-run (present) and --rollback \
             (absent from help). A partial import has to be reversible without \
             appending a second copy.",
        );
    }
}

fn fresh(args: &[&str], timeout: Duration) -> (tempfile::TempDir, super::support::CmdOut) {
    let dir = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let mut owned = vec!["--data-dir".to_string(), dir.path().display().to_string()];
    owned.extend(args.iter().map(|s| (*s).to_string()));
    let out = run_vestige(&owned, home.path(), timeout);
    let _home = home;
    (dir, out)
}

fn require_strata_cli(what: &str, args: &[&str]) {
    let (dir, out) = fresh(args, Duration::from_secs(25));
    let artifacts = sqlite_artifacts(dir.path());
    if !artifacts.is_empty() {
        missing(&format!(
            "`vestige {what}` created a SQLite file: {artifacts:?}"
        ));
    }
    let blob = format!("{}{}", out.stdout, out.stderr);
    refuse_sqlite_creation(&blob, &format!("`vestige {what}`"));
    if out.status != Some(0) {
        missing(&format!(
            "`vestige {what}` did not complete on an empty strata store (exit {:?}). \
             Output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        ));
    }
}

#[test]
fn cli_stats_on_fresh_store() {
    require_strata_cli("stats", &["stats"]);
}
#[test]
fn cli_health_on_fresh_store() {
    require_strata_cli("health", &["health"]);
}
#[test]
fn cli_consolidate_on_fresh_store() {
    require_strata_cli("consolidate", &["consolidate"]);
}
#[test]
fn cli_upgrade_on_fresh_store() {
    require_strata_cli("upgrade --dry-run", &["upgrade", "--dry-run"]);
}
#[test]
fn cli_scan_secrets_on_fresh_store() {
    require_strata_cli("scan-secrets", &["scan-secrets", "--json"]);
}
#[test]
fn cli_compose_on_fresh_store() {
    require_strata_cli("compose", &["compose", "--json"]);
}
#[test]
fn cli_selftest_on_fresh_store() {
    require_strata_cli("selftest", &["selftest"]);
}
#[test]
fn cli_recall_on_fresh_store() {
    // An exact miss must say handle_required, and must not search.
    let (dir, out) = fresh(
        &["recall", "--handle", MISSING_HANDLE, "--json"],
        Duration::from_secs(25),
    );
    assert!(sqlite_artifacts(dir.path()).is_empty());
    let blob = format!("{}{}", out.stdout, out.stderr);
    refuse_sqlite_creation(&blob, "`vestige recall`");
    if !blob.contains("handle_required") {
        missing(&format!(
            "`vestige recall --handle {MISSING_HANDLE}` did not return handle_required. Output: {}",
            blob.chars().take(800).collect::<String>()
        ));
    }
}
#[test]
fn cli_ingest_on_fresh_store() {
    require_strata_cli(
        "ingest",
        &[
            "ingest",
            "release matrix cli fact",
            "--tags",
            "release-matrix",
        ],
    );
}
#[test]
fn cli_causal_walk_on_fresh_store() {
    require_strata_cli(
        "causal-walk",
        &[
            "causal-walk",
            "--failing-test",
            "release_matrix_absent",
            "--json",
        ],
    );
}
#[test]
fn cli_backfill_on_fresh_store() {
    require_strata_cli("backfill", &["backfill", "--json", "--no-promote"]);
}
#[test]
fn cli_project_on_fresh_store() {
    let dir = tempfile::tempdir().unwrap();
    let out_file = dir.path().join("CLAUDE.md");
    require_strata_cli(
        "project",
        &["project", "--out", out_file.to_str().unwrap(), "--json"],
    );
}
#[test]
fn cli_gc_on_fresh_store() {
    require_strata_cli("gc", &["gc", "--dry-run", "--yes"]);
}
#[test]
fn cli_forgotten_lesson_on_fresh_store() {
    let (dir, out) = fresh(
        &["forgotten-lesson", MISSING_HANDLE, "--json"],
        Duration::from_secs(25),
    );
    assert!(sqlite_artifacts(dir.path()).is_empty());
    let blob = format!("{}{}", out.stdout, out.stderr);
    refuse_sqlite_creation(&blob, "`vestige forgotten-lesson`");
    if !blob.contains("handle_required") {
        missing(&format!(
            "`vestige forgotten-lesson` on an unknown id did not return handle_required. Output: {}",
            blob.chars().take(800).collect::<String>()
        ));
    }
}
#[test]
fn cli_export_on_fresh_store() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.json");
    require_strata_cli(
        "export",
        &["export", dest.to_str().unwrap(), "--format", "json"],
    );
}
#[test]
fn cli_portable_export_on_fresh_store() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("archive.json");
    require_strata_cli(
        "portable-export",
        &["portable-export", dest.to_str().unwrap()],
    );
}
#[test]
fn cli_portable_import_on_fresh_store() {
    let (dir, out) = fresh(
        &["portable-import", "/no/such/archive.json"],
        Duration::from_secs(25),
    );
    assert!(sqlite_artifacts(dir.path()).is_empty());
    let blob = format!("{}{}", out.stdout, out.stderr);
    refuse_sqlite_creation(&blob, "`vestige portable-import`");
}
#[test]
fn cli_backup_on_fresh_store() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("backup.db");
    let (_data, out) = fresh(&["backup", dest.to_str().unwrap()], Duration::from_secs(25));
    let blob = format!("{}{}", out.stdout, out.stderr);
    if blob.contains(".sqlite")
        || dest.extension().and_then(|e| e.to_str()) == Some("db") && dest.exists()
    {
        // A 4.0 backup must not be a SQLite snapshot of a store 4.0 does not create.
        if dest.exists() {
            missing(
                "`vestige backup` wrote a SQLite snapshot. 4.0 backup is a strata \
                 segment copy, and a fresh install has a strata log, not vestige.db.",
            );
        }
    }
    refuse_sqlite_creation(&blob, "`vestige backup`");
    if out.status != Some(0) {
        missing(&format!(
            "`vestige backup` did not run against a strata store. Output: {}",
            blob.chars().take(800).collect::<String>()
        ));
    }
}
#[test]
fn cli_restore_on_fresh_store() {
    let (dir, out) = fresh(
        &["restore", "/no/such/backup.json"],
        Duration::from_secs(25),
    );
    assert!(sqlite_artifacts(dir.path()).is_empty());
    let blob = format!("{}{}", out.stdout, out.stderr);
    refuse_sqlite_creation(&blob, "`vestige restore`");
}
#[test]
fn cli_sync_on_fresh_store() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("sync.json");
    require_strata_cli("sync", &["sync", archive.to_str().unwrap()]);
}
#[test]
fn cli_ingest_git_on_fresh_store() {
    let repo = tempfile::tempdir().unwrap();
    let git = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(repo.path())
            .env("GIT_AUTHOR_NAME", "release-matrix")
            .env("GIT_AUTHOR_EMAIL", "release-matrix@example.com")
            .env("GIT_COMMITTER_NAME", "release-matrix")
            .env("GIT_COMMITTER_EMAIL", "release-matrix@example.com")
            .status()
            .expect("git");
        assert!(status.success(), "git {args:?}");
    };
    git(&["init"]);
    std::fs::write(repo.path().join("README.md"), "release matrix\n").unwrap();
    git(&["add", "README.md"]);
    git(&["commit", "-m", "release matrix"]);
    require_strata_cli(
        "ingest-git",
        &[
            "ingest-git",
            repo.path().to_str().unwrap(),
            "--max-commits",
            "5",
            "--json",
        ],
    );
}
#[test]
fn cli_dashboard_on_fresh_store() {
    long_running(&["dashboard", "--port", "0", "--no-open"]);
}
#[test]
fn cli_serve_on_fresh_store() {
    long_running(&["serve", "--port", "0"]);
}

fn long_running(args: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let mut owned = vec!["--data-dir".to_string(), dir.path().display().to_string()];
    owned.extend(args.iter().map(|s| (*s).to_string()));
    let mut child = std::process::Command::new(product_bin("vestige"))
        .args(&owned)
        .env("HOME", home.path())
        .env_remove("VESTIGE_DATA_DIR")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn vestige");
    std::thread::sleep(Duration::from_secs(4));
    let status = child.try_wait().expect("poll");
    let mut stderr = String::new();
    if let Some(mut err) = child.stderr.take() {
        use std::io::Read;
        let _ = err.read_to_string(&mut stderr);
    }
    if status.is_some() {
        let _ = child.wait();
        if !sqlite_artifacts(dir.path()).is_empty() {
            missing(&format!("{args:?} created sqlite"));
        }
        refuse_sqlite_creation(&stderr, &format!("{args:?}"));
        missing(&format!(
            "{args:?} exited before serving. stderr: {}",
            stderr.chars().take(800).collect::<String>()
        ));
    }
    let _ = child.kill();
    let _ = child.wait();
    if !sqlite_artifacts(dir.path()).is_empty() {
        missing(&format!("{args:?} created sqlite while running"));
    }
}

#[test]
fn cli_update_dry_run_creates_no_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let install = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "--data-dir".into(),
            dir.path().display().to_string(),
            "update".into(),
            "--dry-run".into(),
            "--install-dir".into(),
            install.path().display().to_string(),
        ],
        home.path(),
        Duration::from_secs(40),
    );
    assert!(
        sqlite_artifacts(dir.path()).is_empty(),
        "update --dry-run created sqlite: {:?}",
        sqlite_artifacts(dir.path())
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    // Network failure is acceptable. Creating a store is not. A dry run must
    // not claim it wrote binaries into the install dir.
    if install.path().read_dir().unwrap().next().is_some()
        && out.status == Some(0)
        && !blob.contains("dry")
    {
        panic!("update --dry-run wrote into the install dir: {blob}");
    }
}
