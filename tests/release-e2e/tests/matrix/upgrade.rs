//! Fresh install, v3.1.1 auto-upgrade, failed import, SIGKILL, and the
//! hidden migrate-to-strata fallback.
//!
//! `cli_first_launch_upgrades_v3` is the pass criterion for the CLI path.
//! PR #310 hooks automatic upgrade from `vestige-mcp` `serve()` only; the
//! `vestige` binary is a separate row and is expected to fail until that
//! path backups, imports, verifies, and switches.

use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::support::*;

fn source_hashes(dir: &std::path::Path) -> Vec<(String, String)> {
    tree_hashes(dir)
}

#[test]
fn upgrade_fresh_install_started_twice_creates_no_sqlite_and_stays_up() {
    let dir = tempfile::tempdir().unwrap();
    for pass in 1..=2 {
        let mut child = Command::new(product_bin("vestige-mcp"))
            .arg("--data-dir")
            .arg(dir.path())
            .env("VESTIGE_DATA_DIR", dir.path())
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("HOME", dir.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        std::thread::sleep(Duration::from_secs(3));
        let status = child.try_wait().expect("poll");
        let artifacts = sqlite_artifacts(dir.path());
        if !artifacts.is_empty() {
            let _ = child.kill();
            missing(&format!(
                "fresh install pass {pass} created a SQLite file: {artifacts:?}. \
                 The shipped binary must never create one."
            ));
        }
        if let Some(status) = status {
            let mut err = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                use std::io::Read;
                let _ = stderr.read_to_string(&mut err);
            }
            let _ = child.wait();
            missing(&format!(
                "fresh install pass {pass} exited {status} instead of serving a strata log. \
                 stderr: {}",
                err.chars().take(800).collect::<String>()
            ));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[test]
fn upgrade_v311_auto_upgrade_preserves_memories_fsrs_and_source_sha() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let rows = sqlite_text(
        &db,
        "SELECT id, content, stability, difficulty, reps, lapses, next_review FROM knowledge_nodes",
    );
    assert!(!rows.is_empty(), "fixture has no memories");
    let mut child = Command::new(product_bin("vestige-mcp"))
        .arg("--data-dir")
        .arg(&src)
        .env("HOME", dir.path())
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    std::thread::sleep(Duration::from_secs(5));
    let status = child.try_wait().expect("poll");
    let after = sha256_file(&db);
    assert_eq!(before, after, "auto-upgrade modified the v3 file");
    if let Some(code) = status {
        let mut err = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            use std::io::Read;
            let _ = stderr.read_to_string(&mut err);
        }
        let _ = child.wait();
        missing(&format!(
            "auto-upgrade on first launch is not wired. vestige-mcp exited {code} on a real \
             v3.1.1 store (schema 38, {} memories) instead of backup + import + verify + \
             switch. The v3 file sha256 was unchanged. stderr: {}",
            rows.len(),
            err.chars().take(900).collect::<String>()
        ));
    }
    // Still running: kill it and require a strata log whose memories and FSRS
    // columns match the v3 rows. The v3 file hash was already checked.
    let _ = child.kill();
    let _ = child.wait();
    let Some(strata_dir) = find_strata_dir(dir.path()) else {
        missing(
            "auto-upgrade kept vestige-mcp alive but wrote no strata segment. \
             First launch must backup, import, verify, and switch, leaving the \
             v3 file byte-identical.",
        );
    };
    let dump = run_driver(&["dump-migration", strata_dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "auto-upgrade wrote a strata directory that does not reopen: {dump}"
        ));
    }
    let nodes = dump["nodes"].as_array().cloned().unwrap_or_default();
    if nodes.len() != rows.len() {
        missing(&format!(
            "auto-upgrade imported {} memories, the v3 store has {}",
            nodes.len(),
            rows.len()
        ));
    }
    for row in &rows {
        let id = &row[0];
        let node = nodes
            .iter()
            .find(|n| n["legacy_id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("auto-upgrade dropped memory {id}"));
        let legacy = &node["legacy"];
        for (offset, col) in ["stability", "difficulty", "reps", "lapses", "next_review"]
            .into_iter()
            .enumerate()
        {
            let key = format!("knowledge_nodes.{col}");
            let got = legacy[&key].as_str().unwrap_or("");
            if got.is_empty() {
                missing(&format!(
                    "auto-upgrade dropped FSRS column {col} on {id}. legacy={legacy}"
                ));
            }
            let want = &row[offset + 2];
            if col == "stability" || col == "difficulty" {
                let g: f64 = got.parse().unwrap_or(f64::NAN);
                let w: f64 = want.parse().unwrap_or(f64::NAN);
                assert!((g - w).abs() < 1e-9, "{id} {col}: {got} != {want}");
            } else if got != want {
                panic!("{id} {col}: log {got:?} != sqlite {want:?}");
            }
        }
    }
}

/// Pass criterion for the CLI path of first-launch auto-upgrade.
///
/// The first process is `vestige`. Nothing starts `vestige-mcp` before it,
/// and the user does not pass `migrate-to-strata`. The only command is the
/// read `stats` against a real v3.1.1 store. A green row means that read
/// triggered backup, import, verify, and a switch onto the Strata log, the
/// v3 file's sha256 did not change, and memory count plus FSRS review
/// columns match the source. PR #310 does not hook this binary, so the row
/// fails on the current branch.
#[test]
fn cli_first_launch_upgrades_v3() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let rows = sqlite_text(
        &db,
        "SELECT id, content, stability, difficulty, reps, lapses, next_review FROM knowledge_nodes",
    );
    assert!(!rows.is_empty(), "fixture has no memories");
    let home = tempfile::tempdir().unwrap();
    // First process: the CLI, on a harmless read. No migrate command.
    let out = run_vestige(
        &[
            "--data-dir".into(),
            src.display().to_string(),
            "stats".into(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    let after = sha256_file(&db);
    assert_eq!(
        before, after,
        "CLI first-launch upgrade modified the v3 file"
    );
    let record = cli_upgrade_record(&src, &out);
    if v3_refusal(&record) {
        missing(&format!(
            "CLI pass criterion failed: `vestige stats` refused a real v3.1.1 store \
             (schema 38, {} memories) with the v3 refusal. The first process was the \
             CLI, not vestige-mcp, and the user issued no migrate command. Automatic \
             upgrade must run on that read: backup the sqlite family, import, verify, \
             and switch to Strata, leaving the v3 sha256 unchanged. PR #310 hooks this \
             only from vestige-mcp serve(). Output: {}",
            rows.len(),
            record.chars().take(900).collect::<String>()
        ));
    }
    let backups = v3_backup_files(&src);
    let backup_matches = backups.iter().any(|path| sha256_file(path) == before);
    if backups.is_empty() || !backup_matches {
        missing(&format!(
            "CLI pass criterion failed: `vestige stats` did not leave a byte-identical \
             v3 backup (`*.v3-backup-*` matching the source sha256) before switching. \
             Found {backups:?}. Output: {}",
            record.chars().take(700).collect::<String>()
        ));
    }
    let Some(log_dir) = installed_strata_log(&src) else {
        missing(&format!(
            "CLI pass criterion failed: `vestige stats` did not switch the data dir onto \
             a Strata log at log/. Staging left in place is not a switch. Output: {}",
            record.chars().take(700).collect::<String>()
        ));
    };
    let lowered = record.to_lowercase();
    if !lowered.contains("import")
        || !lowered.contains("verif")
        || lowered.contains("verify failed")
    {
        missing(&format!(
            "CLI pass criterion failed: `vestige stats` did not record an import and a \
             passing verify (upgrade.log or the process output). Output: {}",
            record.chars().take(700).collect::<String>()
        ));
    }
    let dump = run_driver(&["dump-migration", log_dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "CLI pass criterion failed: switched log does not reopen after verify: {dump}"
        ));
    }
    let nodes = dump["nodes"].as_array().cloned().unwrap_or_default();
    if nodes.len() != rows.len() {
        missing(&format!(
            "CLI pass criterion failed: import has {} memories, the v3 store has {}",
            nodes.len(),
            rows.len()
        ));
    }
    assert_fsrs_review_columns(&nodes, &rows);
    if out.status != Some(0) {
        missing(&format!(
            "CLI pass criterion failed: upgrade artifacts are present but `vestige stats` \
             exited {:?} instead of reading the switched Strata store. Output: {}",
            out.status,
            record.chars().take(700).collect::<String>()
        ));
    }
}

/// Mirror of [`cli_first_launch_upgrades_v3`]. A corrupted v3.1.1 store run
/// through the same CLI read must leave the v3 file byte-identical and exit
/// with a message that names the v3.1.1 release. The v3 refusal is not that
/// message. This is still the CLI path, not `vestige-mcp` `serve()`.
#[test]
fn cli_first_launch_corrupt_v3_leaves_bytes_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let len = file_len(&db);
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    let before = sha256_file(&db);
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "--data-dir".into(),
            src.display().to_string(),
            "stats".into(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    let after = sha256_file(&db);
    assert_eq!(
        before, after,
        "corrupt CLI upgrade modified the v3 file (len was {len})"
    );
    let record = cli_upgrade_record(&src, &out);
    if out.status == Some(0) {
        panic!("corrupt v3 store through `vestige stats` exited 0: {record}");
    }
    if v3_refusal(&record) || !record.contains("v3.1.1") {
        missing(&format!(
            "CLI pass criterion (failure mirror) failed: a corrupted v3 store run through \
             `vestige stats` must exit non-zero with a message naming v3.1.1 and must leave \
             the v3 file byte-identical. The v3 refusal is not that message. PR #310 emits \
             the v3.1.1 failure only from vestige-mcp serve(). Output: {}",
            record.chars().take(900).collect::<String>()
        ));
    }
}

/// Both binaries start a first launch against one v3.1.1 store.
///
/// `vestige stats` and `vestige-mcp` are spawned before either is waited on.
/// Exactly one import and one verify may run. Both must then open that same
/// verified Strata log with no error, the v3 sha256 stays unchanged, and
/// the staging directory (`log` plus `strata_migrate::STAGING_SUFFIX`) is
/// gone. Expected to fail until that path is on this branch.
#[test]
fn concurrent_first_launch_upgrades_once() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let home_cli = tempfile::tempdir().unwrap();
    let home_mcp = tempfile::tempdir().unwrap();
    // Spawn both before waiting on either.
    let mut cli = spawn_first_launch("vestige", &src, home_cli.path(), &["stats"]);
    let mut mcp = spawn_first_launch("vestige-mcp", &src, home_mcp.path(), &[]);
    let started = Instant::now();
    let limit = Duration::from_secs(60);
    let mut cli_status = None;
    let mut mcp_status = None;
    loop {
        if started.elapsed() > limit {
            let _ = cli.child.kill();
            let _ = mcp.child.kill();
            let _ = cli.child.wait();
            let _ = mcp.child.wait();
            panic!(
                "concurrent first launch hung past 60s. staging left: {}",
                staging_left(&src)
            );
        }
        if cli_status.is_none() {
            cli_status = cli.child.try_wait().expect("poll vestige");
        }
        if mcp_status.is_none() {
            mcp_status = mcp.child.try_wait().expect("poll vestige-mcp");
        }
        let ready = installed_strata_log(&src).is_some() && !staging_left(&src);
        if cli_status.is_some() && mcp_status.is_none() && ready {
            break;
        }
        if cli_status.is_some() && mcp_status.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        before,
        sha256_file(&db),
        "concurrent first launch modified the v3 file"
    );
    let cli_blob = cli.output();
    let mcp_blob = mcp.output();
    let record = format!(
        "{cli_blob}{mcp_blob}{}",
        fs::read_to_string(src.join("upgrade.log")).unwrap_or_default()
    );
    if v3_refusal(&record) {
        missing(&format!(
            "concurrent first launch failed: `vestige stats` and `vestige-mcp` were spawned \
             together on a real v3.1.1 store and refused it. Exactly one import and one verify \
             must run, both processes must open that same verified Strata log with no error, \
             the v3 sha256 must stay unchanged, and no staging folder may remain. Output: {}",
            record.chars().take(900).collect::<String>()
        ));
    }
    let (imports, verifies) = import_verify_counts(&src);
    if imports != 1 || verifies != 1 {
        missing(&format!(
            "concurrent first launch recorded {imports} imports and {verifies} verifies; \
             exactly one of each must happen. upgrade.log: {}",
            fs::read_to_string(src.join("upgrade.log"))
                .unwrap_or_default()
                .chars()
                .take(700)
                .collect::<String>()
        ));
    }
    if staging_left(&src) {
        panic!("concurrent first launch left a staging folder behind");
    }
    let Some(log_dir) = installed_strata_log(&src) else {
        missing("concurrent first launch did not switch onto log/");
    };
    let dump = run_driver(&["dump-migration", log_dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "concurrent first launch log did not reopen verified: {dump}"
        ));
    }
    match cli_status {
        Some(status) if status.success() => {}
        other => missing(&format!(
            "concurrent first launch: `vestige stats` did not open the Strata store (exit {other:?}). \
             Output: {}",
            cli_blob.chars().take(600).collect::<String>()
        )),
    }
    if mcp_status.is_some() {
        missing(&format!(
            "concurrent first launch: `vestige-mcp` exited {:?} instead of serving the same \
             Strata log. Output: {}",
            mcp_status.and_then(|s| s.code()),
            mcp_blob.chars().take(600).collect::<String>()
        ));
    }
    if record.to_lowercase().contains("upgrade failed") {
        missing(&format!(
            "concurrent first launch reported an upgrade error: {}",
            record.chars().take(600).collect::<String>()
        ));
    }
}

/// A first launch killed mid-import leaves the staging directory
/// (`<data-dir>/log` plus `strata_migrate::STAGING_SUFFIX`). The next
/// process must sweep it and finish the upgrade within 60s, not wait on the
/// dead pid. The v3 sha256 stays unchanged. Expected to fail until that
/// path is on this branch.
#[test]
fn stale_staging_after_sigkill_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    // Cloned rows widen the import so a real upgrade still has a segment in
    // staging at the moment of the kill. The bytes under test are this file.
    inflate_nodes(&db, 4000);
    let before = sha256_file(&db);
    let home = tempfile::tempdir().unwrap();
    let mut first = spawn_first_launch("vestige-mcp", &src, home.path(), &[]);
    let started = Instant::now();
    let mut killed_mid_import = false;
    while started.elapsed() < Duration::from_secs(180) {
        if staging_has_segment(&src) && installed_strata_log(&src).is_none() {
            // Child::kill is SIGKILL on Unix.
            let _ = first.child.kill();
            let _ = first.child.wait();
            killed_mid_import = true;
            break;
        }
        if first.child.try_wait().expect("poll first launch").is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        before,
        sha256_file(&db),
        "SIGKILL mid-import modified the v3 file"
    );
    let staging = staging_directory(&src);
    if !killed_mid_import || !staging.exists() {
        missing(&format!(
            "stale-staging row: the first `vestige-mcp` launch did not leave \
             {} behind, so it was not SIGKILLed mid-import. The next \
             process must be able to sweep that folder and finish. Output: {}",
            staging.display(),
            first.output().chars().take(800).collect::<String>()
        ));
    }

    let home_again = tempfile::tempdir().unwrap();
    let mut second = spawn_first_launch("vestige-mcp", &src, home_again.path(), &[]);
    let started = Instant::now();
    let limit = Duration::from_secs(60);
    loop {
        if started.elapsed() > limit {
            let _ = second.child.kill();
            let _ = second.child.wait();
            panic!(
                "stale staging relaunch hung past 60s waiting on the dead process. \
                 staging left: {}",
                staging_left(&src)
            );
        }
        let exited = second.child.try_wait().expect("poll relaunch");
        let ready = installed_strata_log(&src).is_some() && !staging_left(&src);
        if ready && exited.is_none() {
            break;
        }
        if let Some(status) = exited {
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(before, sha256_file(&db), "relaunch modified the v3 file");
            missing(&format!(
                "stale-staging relaunch exited {:?} before it swept staging and served the \
                 verified Strata log. Output: {}",
                status.code(),
                second.output().chars().take(800).collect::<String>()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        before,
        sha256_file(&db),
        "relaunch after SIGKILL modified the v3 file"
    );
    assert!(
        !staging_left(&src),
        "relaunch left staging behind: {}",
        staging_directory(&src).display()
    );
    let log_dir = installed_strata_log(&src).expect("log missing after ready");
    let record = format!(
        "{}{}",
        second.output(),
        fs::read_to_string(src.join("upgrade.log")).unwrap_or_default()
    );
    if v3_refusal(&record) || record.to_lowercase().contains("upgrade failed") {
        missing(&format!(
            "stale-staging relaunch opened with an error: {}",
            record.chars().take(700).collect::<String>()
        ));
    }
    let (imports, verifies) = import_verify_counts(&src);
    if verifies < 1 {
        missing(&format!(
            "stale-staging relaunch did not record a verify after sweeping staging \
             ({imports} imports, {verifies} verifies)"
        ));
    }
    let dump = run_driver(&["dump-migration", log_dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "stale-staging relaunch log did not reopen: {dump}"
        ));
    }
}

#[test]
fn upgrade_failed_import_leaves_v3_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    // Corrupt past the header so the file is still a SQLite database and the
    // import has something to fail on. The bytes we leave are the pre-import
    // image; the process must not change them.
    let len = file_len(&db);
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    let before = sha256_file(&db);
    let out = run_cmd(
        &product_bin("vestige-mcp"),
        &["--data-dir".into(), src.display().to_string()],
        &[("HOME", dir.path().to_str().unwrap())],
        &["VESTIGE_DATA_DIR"],
        Duration::from_secs(20),
    );
    let after = sha256_file(&db);
    assert_eq!(
        before, after,
        "failed import modified the v3 file (len was {len})"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status == Some(0) {
        panic!("corrupt v3 import exited 0: {blob}");
    }
    if !blob.contains("v3.1.1") {
        missing(&format!(
            "failed import did not point at the v3.1.1 release. The v3 file was left \
             untouched, but the process must exit with a message naming v3.1.1 so the \
             operator can go back. Output: {}",
            blob.chars().take(900).collect::<String>()
        ));
    }
}

#[test]
fn upgrade_migrate_dry_run_writes_nothing_and_keeps_source_sha() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let before = source_hashes(&src);
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
            "--dry-run".into(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(before, source_hashes(&src), "dry-run changed the v3 store");
    let wrote = dest.exists() && fs::read_dir(&dest).unwrap().next().is_some();
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) || wrote {
        missing(&format!(
            "migrate-to-strata --dry-run did not report counts without writing \
             (exit {:?}, dest written {wrote}). Output: {}",
            out.status,
            blob.chars().take(900).collect::<String>()
        ));
    }
    assert!(
        blob.contains("node") || blob.contains("Migrated") || blob.contains("Dry run"),
        "dry-run produced no count report: {blob}"
    );
}

#[test]
fn upgrade_real_migrate_keeps_source_sha_and_fsrs() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let expected = sqlite_text(
        &db,
        "SELECT id, stability, difficulty, reps, lapses, next_review, suppression_count, protected \
         FROM knowledge_nodes ORDER BY id",
    );
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    assert_eq!(before, sha256_file(&db), "migrate modified the v3 file");
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) {
        missing(&format!(
            "migrate-to-strata failed on the real v3.1.1 fixture (schema 38). Output: {}",
            blob.chars().take(1000).collect::<String>()
        ));
    }
    let dump = run_driver(&["dump-migration", dest.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "migrated log did not reopen for a frame dump: {dump}"
        ));
    }
    let nodes = dump["nodes"].as_array().cloned().unwrap_or_default();
    assert_eq!(nodes.len(), expected.len(), "node count drifted: {dump}");
    for row in &expected {
        let id = &row[0];
        let node = nodes
            .iter()
            .find(|n| n["legacy_id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("migrated log dropped memory {id}"));
        let legacy = &node["legacy"];
        for (idx, col) in [
            "stability",
            "difficulty",
            "reps",
            "lapses",
            "next_review",
            "suppression_count",
            "protected",
        ]
        .into_iter()
        .enumerate()
        {
            let key = format!("knowledge_nodes.{col}");
            let got = legacy[&key].as_str().unwrap_or("");
            if got.is_empty() {
                missing(&format!(
                    "migrated node {id} dropped {col}. FSRS schedule and flags must be \
                     carried exactly. legacy={legacy}"
                ));
            }
            let want = &row[idx + 1];
            if col == "stability" || col == "difficulty" {
                let g: f64 = got.parse().unwrap_or(f64::NAN);
                let w: f64 = want.parse().unwrap_or(f64::NAN);
                assert!(
                    (g - w).abs() < 1e-9,
                    "{id} {col}: log {got} != sqlite {want}"
                );
            } else if got != want {
                panic!("{id} {col}: log {got:?} != sqlite {want:?}");
            }
        }
    }
}

#[test]
fn upgrade_nonempty_to_is_refused() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("probe-v38-ckpt"), &src);
    let before = sha256_file(&src.join("vestige.db"));
    let dest = work.path().join("strata");
    fs::create_dir_all(&dest).unwrap();
    fs::write(dest.join("already-here"), b"not empty").unwrap();
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(before, sha256_file(&src.join("vestige.db")));
    assert!(
        fs::read(dest.join("already-here")).unwrap() == b"not empty",
        "refused migrate changed the destination"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status == Some(0)
        || !blob.to_lowercase().contains("not empty") && !blob.to_lowercase().contains("empty")
    {
        missing(&format!(
            "non-empty --to was not refused. exit {:?} output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        ));
    }
}

#[test]
fn upgrade_wal_without_flag_refuses_and_source_unchanged() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-wal"), &src);
    let before = source_hashes(&src);
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.join("vestige.db").display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(
        before,
        source_hashes(&src),
        "WAL refusal modified the source"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    assert!(
        out.status != Some(0) && blob.to_lowercase().contains("wal"),
        "non-empty WAL must be refused without --accept-wal-snapshot: {blob}"
    );
}

#[test]
fn upgrade_sigkill_mid_import_then_rerun_does_not_duplicate() {
    let killed = kill_migrate_during_segment_write();
    assert_killed_import_left_no_dest(&killed);
    let home = tempfile::tempdir().unwrap();
    let again = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            killed.src.display().to_string(),
            "--to".into(),
            killed.dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    let blob = format!("{}{}", again.stdout, again.stderr);
    assert_eq!(
        killed.before_sha,
        sha256_file(&killed.db),
        "rerun modified the v3 file"
    );
    assert_eq!(
        again.status,
        Some(0),
        "rerun after a mid-write SIGKILL must finish the import. output: {}",
        blob.chars().take(800).collect::<String>()
    );
    assert!(
        !killed.staging.exists(),
        "staging directory survived a successful rerun: {}",
        killed.staging.display()
    );
    assert!(
        killed.dest.is_dir(),
        "rerun exited 0 without publishing {}",
        killed.dest.display()
    );
    let got = node_count_if_log(&killed.dest);
    assert_eq!(
        got, killed.source_nodes,
        "published node count {got} != source memory count {}. A second copy was written.",
        killed.source_nodes
    );
    let verify = run_cmd(
        &product_bin("strata-verify"),
        &[killed.dest.display().to_string()],
        &[],
        &[],
        Duration::from_secs(60),
    );
    let verify_blob = format!("{}{}", verify.stdout, verify.stderr);
    assert!(
        verify.status == Some(0) && verify_blob.contains("OK"),
        "strata-verify did not accept the republished log (exit {:?}): {}",
        verify.status,
        verify_blob.chars().take(800).collect::<String>()
    );
}

#[test]
fn upgrade_killed_import_leaves_no_dest() {
    let killed = kill_migrate_during_segment_write();
    assert_killed_import_left_no_dest(&killed);
}

#[test]
fn upgrade_legacy_links_are_not_causal_edges() {
    let work = tempfile::tempdir().unwrap();
    let cases = [
        ("fresh-v38-ckpt", "narrative"),
        ("vnc-v36", "semantic"),
        ("backfill-v31", "backfill_candidate"),
    ];
    for (fixture_name, legacy_type) in cases {
        let src = work.path().join(fixture_name);
        copy_tree(&fixture(fixture_name), &src);
        let before = sha256_file(&src.join("vestige.db"));
        let dest = work.path().join(format!("{fixture_name}-strata"));
        let home = tempfile::tempdir().unwrap();
        let out = run_vestige(
            &[
                "migrate-to-strata".into(),
                "--from".into(),
                src.display().to_string(),
                "--to".into(),
                dest.display().to_string(),
            ],
            home.path(),
            Duration::from_secs(180),
        );
        assert_eq!(before, sha256_file(&src.join("vestige.db")));
        let blob = format!("{}{}", out.stdout, out.stderr);
        if out.status != Some(0) {
            missing(&format!(
                "migrate of {fixture_name} failed, so legacy link {legacy_type} could not be checked: {}",
                blob.chars().take(600).collect::<String>()
            ));
        }
        let dump = run_driver(&["dump-migration", dest.to_str().unwrap()]);
        let edges = dump["edges"].as_array().cloned().unwrap_or_default();
        let matched: Vec<&Value> = edges
            .iter()
            .filter(|e| e["legacy_link_type"].as_str() == Some(legacy_type))
            .collect();
        if matched.is_empty() {
            missing(&format!(
                "{fixture_name} link type {legacy_type} did not survive as legacy provenance. edges={edges:?}"
            ));
        }
        for edge in &matched {
            assert_eq!(
                edge["legacy_inferred"], true,
                "{legacy_type} must be legacy data, not a causal edge: {edge}"
            );
            assert_eq!(edge["link_type"], "derived_from");
            let walked = edge["link_type"].as_str() == Some(legacy_type);
            assert!(!walked, "{legacy_type} is stored as a walkable link_type");
        }
        // The product walk must not treat that pair as a causal neighbor.
        // Until causal_walk reads the strata log, this stays a named gap.
        let home = tempfile::tempdir().unwrap();
        let walk = run_vestige(
            &[
                "--data-dir".into(),
                dest.display().to_string(),
                "causal-walk".into(),
                "--logged-write".into(),
                matched[0]["source_legacy_id"].as_str().unwrap_or("").into(),
                "--json".into(),
            ],
            home.path(),
            Duration::from_secs(40),
        );
        let walk_blob = format!("{}{}", walk.stdout, walk.stderr);
        if walk_blob.contains("SQLite store creation is disabled")
            || walk_blob.contains("cannot be opened by 4.0")
            || walk_blob.contains("no store")
        {
            missing(&format!(
                "causal walk is not wired to the strata log, so a migrated {legacy_type} \
                 edge cannot be proven unwalkable through the product. The log flags it \
                 legacy_inferred=true (that part held). CLI output: {}",
                walk_blob.chars().take(500).collect::<String>()
            ));
        }
        assert!(
            !walk_blob.contains(&format!("\"link_type\":\"{legacy_type}\"")),
            "causal walk returned legacy type {legacy_type}: {walk_blob}"
        );
    }
}

fn find_strata_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut found = None;
    fn walk(dir: &std::path::Path, found: &mut Option<std::path::PathBuf>) {
        if found.is_some() {
            return;
        }
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().and_then(|e| e.to_str()) == Some("seg") {
                *found = path.parent().map(|p| p.to_path_buf());
                return;
            }
        }
    }
    walk(root, &mut found);
    found
}

struct KilledImport {
    _work: tempfile::TempDir,
    src: std::path::PathBuf,
    db: std::path::PathBuf,
    dest: std::path::PathBuf,
    staging: std::path::PathBuf,
    before_sha: String,
    source_nodes: usize,
    partial_nodes: usize,
    dest_exists: bool,
    staging_exists: bool,
}

/// `migrate-to-strata --to <dest>` stages at `<dest>` plus
/// [`strata_migrate::STAGING_SUFFIX`].
fn migrate_staging(dest: &std::path::Path) -> std::path::PathBuf {
    let mut name = dest
        .file_name()
        .unwrap_or(OsStr::new("strata"))
        .to_os_string();
    name.push(strata_migrate::STAGING_SUFFIX);
    dest.with_file_name(name)
}

fn segment_bytes(dir: &std::path::Path) -> u64 {
    seg_files(dir)
        .iter()
        .map(|path| fs::metadata(path).map(|meta| meta.len()).unwrap_or(0))
        .sum()
}

/// SIGKILL `vestige migrate-to-strata --to <dest>` while a staging segment is
/// growing. The sleep-window env var is not set: the kill lands on a write
/// whose size increased between two polls.
fn kill_migrate_during_segment_write() -> KilledImport {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    inflate_nodes(&db, 8000);
    let source_nodes = sqlite_text(&db, "SELECT COUNT(*) FROM knowledge_nodes")[0][0]
        .parse::<usize>()
        .unwrap();
    let before_sha = sha256_file(&db);
    let dest = work.path().join("strata");
    let staging = migrate_staging(&dest);
    let home = tempfile::tempdir().unwrap();
    let mut child = Command::new(product_bin("vestige"))
        .args([
            "migrate-to-strata",
            "--from",
            src.to_str().unwrap(),
            "--to",
            dest.to_str().unwrap(),
        ])
        .env("HOME", home.path())
        .env_remove("VESTIGE_DATA_DIR")
        .env_remove("STRATA_MIGRATE_SIGKILL_WINDOW")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn migrate");
    let started = Instant::now();
    let mut previous = 0u64;
    let mut killed = false;
    while started.elapsed() < Duration::from_secs(180) {
        if child.try_wait().expect("poll").is_some() {
            break;
        }
        let bytes = segment_bytes(&staging);
        if previous > 0 && bytes > previous {
            let _ = child.kill();
            killed = true;
            break;
        }
        if bytes > 0 {
            previous = bytes;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let _ = child.wait();
    if !killed {
        missing(
            "could not SIGKILL vestige migrate-to-strata during a staging segment \
             write. The process finished before a segment grew, even after the \
             source was inflated. The gate needs a kill on an actual write.",
        );
    }
    let dest_exists = dest.exists();
    let staging_exists = staging.exists();
    let partial_nodes = node_count_if_log(&dest);
    eprintln!(
        "post-kill partial_nodes={partial_nodes} dest_exists={dest_exists} staging_exists={staging_exists} staging={}",
        staging.display()
    );
    KilledImport {
        _work: work,
        src,
        db,
        dest,
        staging,
        before_sha,
        source_nodes,
        partial_nodes,
        dest_exists,
        staging_exists,
    }
}

fn assert_killed_import_left_no_dest(killed: &KilledImport) {
    assert_eq!(
        killed.before_sha,
        sha256_file(&killed.db),
        "SIGKILL path modified the v3 file"
    );
    assert!(
        !killed.dest_exists,
        "mid-write SIGKILL left <dest> at {} (partial_nodes={}). Only <dest>{} may exist.",
        killed.dest.display(),
        killed.partial_nodes,
        strata_migrate::STAGING_SUFFIX
    );
    if killed.staging_exists {
        assert!(
            killed.staging.is_dir(),
            "staging path is not a directory: {}",
            killed.staging.display()
        );
    }
}

fn inflate_nodes(db: &std::path::Path, extra: usize) {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare("SELECT * FROM knowledge_nodes LIMIT 1")
        .unwrap();
    let cols: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut rows = stmt.query([]).unwrap();
    let row = rows.next().unwrap().unwrap();
    let mut values: Vec<rusqlite::types::Value> = Vec::new();
    for i in 0..cols.len() {
        values.push(row.get(i).unwrap());
    }
    drop(rows);
    drop(stmt);
    let placeholders = vec!["?"; cols.len()].join(",");
    let sql = format!(
        "INSERT INTO knowledge_nodes ({}) VALUES ({placeholders})",
        cols.join(",")
    );
    let id_idx = cols.iter().position(|c| c == "id").unwrap();
    let tx = conn.unchecked_transaction().unwrap();
    for n in 0..extra {
        values[id_idx] =
            rusqlite::types::Value::Text(format!("aaaaaaaa-bbbb-4ccc-8ddd-{:012x}", n as u64));
        tx.execute(&sql, rusqlite::params_from_iter(values.iter()))
            .unwrap();
    }
    tx.commit().unwrap();
}

fn node_count_if_log(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    let dump = run_driver(&["dump-migration", dir.to_str().unwrap()]);
    dump["nodes"].as_array().map(|a| a.len()).unwrap_or(0)
}

fn v3_refusal(blob: &str) -> bool {
    blob.contains("cannot be opened by 4.0")
}

const IMPORT_MARK: &str = "importing into";
const VERIFY_MARK: &str = "verifying strata log";

struct Tracked {
    child: Child,
    stdout: Arc<Mutex<String>>,
    stderr: Arc<Mutex<String>>,
}

impl Tracked {
    fn output(&self) -> String {
        let stdout = self.stdout.lock().expect("stdout lock").clone();
        let stderr = self.stderr.lock().expect("stderr lock").clone();
        format!("{stdout}{stderr}")
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Spawn a first launch. Stdin is left open: dropping it is EOF, and
/// `vestige-mcp` would treat that as the client disconnecting.
fn spawn_first_launch(
    bin: &str,
    data_dir: &std::path::Path,
    home: &std::path::Path,
    extra_args: &[&str],
) -> Tracked {
    let mut command = Command::new(product_bin(bin));
    command
        .arg("--data-dir")
        .arg(data_dir)
        .args(extra_args)
        .env("HOME", home)
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .unwrap_or_else(|e| panic!("HARNESS: failed to spawn {bin}: {e}"));
    if let Some(stdin) = child.stdin.take() {
        std::mem::forget(stdin);
    }
    let stdout = Arc::new(Mutex::new(String::new()));
    let stderr = Arc::new(Mutex::new(String::new()));
    let mut out_pipe = child.stdout.take().expect("stdout");
    let out_buf = Arc::clone(&stdout);
    std::thread::spawn(move || {
        let mut tmp = String::new();
        let _ = out_pipe.read_to_string(&mut tmp);
        *out_buf.lock().expect("stdout lock") = tmp;
    });
    let mut err_pipe = child.stderr.take().expect("stderr");
    let err_buf = Arc::clone(&stderr);
    std::thread::spawn(move || {
        let mut tmp = String::new();
        let _ = err_pipe.read_to_string(&mut tmp);
        *err_buf.lock().expect("stderr lock") = tmp;
    });
    Tracked {
        child,
        stdout,
        stderr,
    }
}

/// First-launch staging: `<data-dir>/log` plus [`strata_migrate::STAGING_SUFFIX`].
/// `migrate-to-strata --to <dest>` uses the same suffix on `<dest>`.
fn staging_directory(data_dir: &std::path::Path) -> std::path::PathBuf {
    let dest = data_dir.join("log");
    let mut name = dest
        .file_name()
        .unwrap_or(std::ffi::OsStr::new("log"))
        .to_os_string();
    name.push(strata_migrate::STAGING_SUFFIX);
    dest.with_file_name(name)
}

fn staging_left(data_dir: &std::path::Path) -> bool {
    staging_directory(data_dir).exists()
}

fn staging_has_segment(data_dir: &std::path::Path) -> bool {
    !seg_files(&staging_directory(data_dir)).is_empty()
}

/// Counts in `upgrade.log` only. Each process also echoes the same line to
/// stderr, so summing stderr would double-count one import.
fn import_verify_counts(data_dir: &std::path::Path) -> (usize, usize) {
    let log = fs::read_to_string(data_dir.join("upgrade.log")).unwrap_or_default();
    (
        log.matches(IMPORT_MARK).count(),
        log.matches(VERIFY_MARK).count(),
    )
}

fn cli_upgrade_record(data_dir: &std::path::Path, out: &CmdOut) -> String {
    let log = fs::read_to_string(data_dir.join("upgrade.log")).unwrap_or_default();
    format!("{}{}{log}", out.stdout, out.stderr)
}

fn v3_backup_files(data_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(data_dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains(".v3-backup-") && !name.ends_with(".partial") {
            out.push(entry.path());
        }
    }
    out.sort();
    out
}

/// The log the upgrade switches into place. A leftover staging directory is
/// not a switch.
fn installed_strata_log(data_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let log = data_dir.join("log");
    if seg_files(&log).is_empty() {
        None
    } else {
        Some(log)
    }
}

fn assert_fsrs_review_columns(nodes: &[Value], rows: &[Vec<String>]) {
    for row in rows {
        let id = &row[0];
        let node = nodes
            .iter()
            .find(|n| n["legacy_id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("CLI upgrade dropped memory {id}"));
        let legacy = &node["legacy"];
        for (offset, col) in ["stability", "difficulty", "reps", "lapses", "next_review"]
            .into_iter()
            .enumerate()
        {
            let key = format!("knowledge_nodes.{col}");
            let got = legacy[&key].as_str().unwrap_or("");
            if got.is_empty() {
                missing(&format!(
                    "CLI pass criterion failed: FSRS review column {col} was dropped on {id}. \
                     legacy={legacy}"
                ));
            }
            let want = &row[offset + 2];
            if col == "stability" || col == "difficulty" {
                let g: f64 = got.parse().unwrap_or(f64::NAN);
                let w: f64 = want.parse().unwrap_or(f64::NAN);
                assert!((g - w).abs() < 1e-9, "{id} {col}: {got} != {want}");
            } else if got != want {
                panic!("{id} {col}: log {got:?} != sqlite {want:?}");
            }
        }
    }
}
