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

use ed25519_dalek::{Signer, Verifier};
use serde_json::Value;
use strata::{
    frame_hash, header_hash, merkle_root, parse_frame, payload_blake3, signature_message, Frame,
    SegmentHeader, SegmentTrailer, HEADER_WIRE_SIZE, TRAILER_WIRE_SIZE,
};

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

/// A real v3.1.1 binary, the official GitHub release asset, must reopen the
/// original `vestige.db` after 4.0 has auto-upgraded that data dir and written
/// a new memory. 4.0 leaves the sqlite family byte-identical. v3.1.1 stats
/// lists the pre-upgrade count, and the 4.0 memory is absent from those rows.
#[test]
fn v311_reopens_untouched_db_after_upgrade() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sqlite_family_hashes(&src);
    let contents_before = sqlite_text(&db, "SELECT id, content FROM knowledge_nodes ORDER BY id");
    let pre_count = contents_before.len();
    assert!(pre_count > 0, "fixture has no memories");
    let marker = "release-matrix-v311-reopen-only-9f3c2a";
    assert!(
        sqlite_text(
            &db,
            "SELECT COUNT(*) FROM knowledge_nodes WHERE content LIKE '%release-matrix-v311-reopen-only-9f3c2a%'",
        )[0][0]
            == "0",
        "fixture already contains the 4.0 marker"
    );

    let home = tempfile::tempdir().unwrap();
    let stats = run_vestige(
        &[
            "--data-dir".into(),
            src.display().to_string(),
            "stats".into(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    assert_eq!(
        before,
        sqlite_family_hashes(&src),
        "4.0 first launch changed the v3 sqlite family"
    );
    let stats_blob = format!("{}{}", stats.stdout, stats.stderr);
    if stats.status != Some(0) || v3_refusal(&stats_blob) {
        panic!(
            "FAIL: 4.0 did not auto-upgrade the v3.1.1 store on first launch, so the reopen check cannot run. exit {:?} output: {}",
            stats.status,
            stats_blob.chars().take(800).collect::<String>()
        );
    }

    let home = tempfile::tempdir().unwrap();
    let ingest = run_vestige(
        &[
            "--data-dir".into(),
            src.display().to_string(),
            "ingest".into(),
            marker.into(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    let ingest_blob = format!("{}{}", ingest.stdout, ingest.stderr);
    assert_eq!(
        before,
        sqlite_family_hashes(&src),
        "4.0 ingest changed the v3 sqlite family"
    );
    if ingest.status != Some(0) || !log_contains_marker(&src, marker) {
        panic!(
            "FAIL: 4.0 did not record the new memory in the strata log, so a v3 miss would be vacuous. exit {:?} output: {}",
            ingest.status,
            ingest_blob.chars().take(800).collect::<String>()
        );
    }

    // The digest gate is here, before the v3.1.1 process. That binary's own
    // `stats` rewrites sqlite bookkeeping on a read of an unmodified file
    // (header change counter and later pages, same file length), so a digest
    // taken after it starts is not the 4.0 contract.
    assert_eq!(
        before,
        sqlite_family_hashes(&src),
        "4.0 left a different vestige.db than the one v3.1.1 is about to open"
    );
    let v311 = fetch_v311_vestige(&work.path().join("v311-release"));
    let home = tempfile::tempdir().unwrap();
    let reopened = run_cmd(
        &v311,
        &[
            "--data-dir".into(),
            src.display().to_string(),
            "stats".into(),
        ],
        &[("HOME", home.path().to_str().unwrap())],
        &["VESTIGE_DATA_DIR"],
        Duration::from_secs(60),
    );
    let blob = format!("{}{}", reopened.stdout, reopened.stderr);
    assert_eq!(
        reopened.status,
        Some(0),
        "v3.1.1 did not start against the untouched vestige.db: {}",
        blob.chars().take(800).collect::<String>()
    );
    let listed = stats_memory_count(&blob).unwrap_or_else(|| {
        panic!(
            "v3.1.1 stats did not list a memory count: {}",
            blob.chars().take(800).collect::<String>()
        )
    });
    assert_eq!(
        listed,
        pre_count,
        "v3.1.1 stats listed {listed} memories, the pre-upgrade store has {pre_count}: {}",
        blob.chars().take(500).collect::<String>()
    );
    assert!(
        !blob.contains(marker),
        "v3.1.1 stats showed the memory written under 4.0"
    );
    let contents_after = sqlite_text(&db, "SELECT id, content FROM knowledge_nodes ORDER BY id");
    assert_eq!(
        contents_before, contents_after,
        "vestige.db rows changed; the memory written under 4.0 must stay absent"
    );
    assert!(
        contents_after.iter().all(|row| row[1] != marker),
        "the memory written under 4.0 is in vestige.db"
    );
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

/// SIGKILL while staging segments are still growing, before `publish`
/// renames staging onto `<dest>`. `<dest>` must be absent. A rerun exits 0,
/// leaves the v3 sha unchanged, and rebuilds a log whose node, edge, FSRS,
/// and receipt frame counts match a clean run of the same source. The
/// migrate stdout line is not the count: an idempotent report prints source
/// table rows.
#[test]
fn upgrade_sigkill_mid_import_then_rerun_does_not_duplicate() {
    let killed = kill_migrate(KillPoint::BeforePublish);
    assert_killed_before_publish(&killed);
    let clean = killed.dest.with_file_name("clean-run");
    let clean_out = migrate_to(&killed.src, &clean);
    assert_eq!(
        clean_out.status,
        Some(0),
        "clean migrate failed, so the pre-rename rerun has no log oracle: {}",
        format!("{}{}", clean_out.stdout, clean_out.stderr)
            .chars()
            .take(600)
            .collect::<String>()
    );
    let again = migrate_to(&killed.src, &killed.dest);
    assert_rerun_keeps_v3(&killed, &again);
    assert!(
        !killed.staging.exists(),
        "staging directory survived a successful rerun: {}",
        killed.staging.display()
    );
    assert!(
        killed.dest.is_dir(),
        "rerun did not publish {}",
        killed.dest.display()
    );
    // Frame counts come from the log. Stdout `Migrated: N nodes, … M fsrs
    // events` is source row counts on the idempotent path.
    assert_same_frames(&frame_counts(&killed.dest), &frame_counts(&clean));
    assert_strata_verify_ok(&killed.dest);
    assert_strata_verify_ok(&clean);
}

#[test]
fn upgrade_killed_import_leaves_no_dest() {
    let killed = kill_migrate(KillPoint::BeforePublish);
    assert_killed_before_publish(&killed);
}

/// SIGKILL after the publish rename. `<dest>` is the finished log. The rerun
/// exits 0 without rewriting it: the log bytes and the v3 sha stay identical.
#[test]
fn upgrade_sigkill_after_publish_rerun_is_idempotent() {
    let killed = kill_migrate(KillPoint::AfterPublish);
    assert_eq!(
        killed.before_sha,
        sha256_file(&killed.db),
        "post-rename SIGKILL modified the v3 file"
    );
    assert!(
        killed.dest_exists && killed.dest.is_dir(),
        "post-rename SIGKILL left no finished log at {}",
        killed.dest.display()
    );
    assert!(
        !killed.staging_exists,
        "post-rename SIGKILL left staging at {}",
        killed.staging.display()
    );
    // `receipt-signing.key` in the parent is written before the rename. Its
    // presence after the kill is the expected layout, not a leaked secret.
    assert_receipt_key_allowed(&killed.dest);
    let log_before = tree_hashes(&killed.dest);
    let key = receipt_key_path(&killed.dest);
    let key_before = key.is_file().then(|| sha256_file(&key));
    let frames_before = frame_counts(&killed.dest);
    assert!(
        frames_before.receipts >= 1,
        "finished log has no receipt frame before the rerun"
    );
    let again = migrate_to(&killed.src, &killed.dest);
    assert_rerun_keeps_v3(&killed, &again);
    assert_eq!(
        log_before,
        tree_hashes(&killed.dest),
        "idempotent rerun rewrote the published log"
    );
    let key_after = key.is_file().then(|| sha256_file(&key));
    assert_eq!(
        key_before,
        key_after,
        "idempotent rerun rewrote {}",
        strata_migrate::records::RECEIPT_KEY_FILE
    );
    assert_same_frames(&frame_counts(&killed.dest), &frames_before);
    assert_strata_verify_ok(&killed.dest);
}

/// An import into a sibling `--to`, including one killed mid-write and rerun,
/// must not replace a receipt key that already signs another store in the
/// same parent. The key is written by `load_or_create_receipt_key`
/// (`crates/strata-migrate/src/records.rs:295`) into the parent chosen by
/// `receipt_key_dir_of` (`crates/strata-migrate/src/lib.rs:983`), called from
/// `finish` (`crates/strata-migrate/src/lib.rs:929`).
///
/// `strata-verify` on a migrated log checks a receipt against the public key
/// embedded in that receipt (`crates/strata-verify/src/migration.rs:89`), not
/// against the on-disk `receipt-signing.key`. Old receipts verifying says
/// nothing about the key file. The sha256 of the file is the byte assertion.
///
/// After the import, the old store signs a new receipt through the binary.
/// `vestige ingest` is the gated write; `vestige backup` seals that segment
/// under the key the log already holds (`strata.key`). The sealed trailer
/// has to verify against the key snapshotted before the write. That is the
/// step that fails when the key file was replaced. The later import's
/// migration receipt is checked the same way against the public key embedded
/// in the old log, because that signature is the one `receipt-signing.key`
/// produces.
#[test]
fn import_never_overwrites_existing_receipt_key() {
    let work = tempfile::tempdir().unwrap();
    let parent = work.path().join("parent");
    fs::create_dir_all(&parent).unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    fs::copy(src.join("vestige.db"), parent.join("vestige.db")).unwrap();
    let existing = parent.join("log");
    let first = migrate_to(&src, &existing);
    assert_eq!(
        first.status,
        Some(0),
        "could not build the pre-existing store: {}",
        cmd_excerpt(&first)
    );
    let key = parent.join(strata_migrate::records::RECEIPT_KEY_FILE);
    assert!(
        key.is_file(),
        "pre-existing store left no {} in {}",
        strata_migrate::records::RECEIPT_KEY_FILE,
        parent.display()
    );
    let key_sha = sha256_file(&key);
    let embedded = embedded_receipt_key(&existing);

    let sibling = parent.join("sibling");
    let second = migrate_to(&src, &sibling);
    assert_eq!(
        second.status,
        Some(0),
        "sibling import failed: {}",
        cmd_excerpt(&second)
    );
    assert_eq!(
        key_sha,
        sha256_file(&key),
        "sibling import rewrote the pre-existing receipt key (written at \
         crates/strata-migrate/src/records.rs:295)"
    );
    assert_new_receipt_matches_old_key(&sibling, &embedded);

    let kill_src = work.path().join("kill-src");
    copy_tree(&src, &kill_src);
    inflate_nodes(&kill_src.join("vestige.db"), 8000);
    let killed_dest = parent.join("killed");
    let killed = kill_into(&kill_src, &killed_dest, KillPoint::BeforePublish);
    assert_killed_before_publish(&killed);
    assert_eq!(
        key_sha,
        sha256_file(&key),
        "mid-import SIGKILL rewrote the pre-existing receipt key (written at \
         crates/strata-migrate/src/records.rs:295)"
    );
    let rerun = migrate_to(&kill_src, &killed.dest);
    assert_eq!(
        rerun.status,
        Some(0),
        "rerun after the mid-import SIGKILL failed: {}",
        cmd_excerpt(&rerun)
    );
    assert_eq!(
        key_sha,
        sha256_file(&key),
        "rerun rewrote the pre-existing receipt key (written at \
         crates/strata-migrate/src/records.rs:295, parent from \
         crates/strata-migrate/src/lib.rs:983, called from finish at \
         crates/strata-migrate/src/lib.rs:929)"
    );
    assert_new_receipt_matches_old_key(&killed.dest, &embedded);

    let log_key_before = fs::read(existing.join("strata.key")).unwrap();
    let marker = "release-matrix-old-store-gated-9c1e";
    let home = tempfile::tempdir().unwrap();
    let ingest = run_vestige(
        &[
            "--data-dir".into(),
            parent.display().to_string(),
            "ingest".into(),
            marker.into(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(
        ingest.status,
        Some(0),
        "gated ingest on the old store failed: {}",
        cmd_excerpt(&ingest)
    );
    assert_eq!(
        key_sha,
        sha256_file(&key),
        "gated ingest rewrote the pre-existing receipt key"
    );
    assert!(
        log_contains_marker(&parent, marker),
        "gated ingest did not land in the old store's log"
    );
    let backup_to = work.path().join("old-store-backup");
    let backup = run_vestige(
        &[
            "--data-dir".into(),
            parent.display().to_string(),
            "backup".into(),
            backup_to.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(
        backup.status,
        Some(0),
        "sealing the gated write failed: {}",
        cmd_excerpt(&backup)
    );
    assert_eq!(
        key_sha,
        sha256_file(&key),
        "sealing the gated write rewrote the pre-existing receipt key"
    );
    assert!(
        marker_in_sealed_segment(&existing, marker),
        "gated write landed, but the segment that holds it was not signed"
    );
    assert!(
        newest_trailer_verifies(&existing, &log_key_before),
        "gated write did not verify under the key the old log already pinned"
    );
}

/// Case (a) of `verify_rejects_resigned_log_with_new_embedded_key`.
/// One node is rewritten and the receipt is sealed under a fresh key. The
/// segment trailer stays under the log's `strata.key`. `receipt-signing.key`
/// beside the log is not touched. Verify must exit non-zero and print that
/// pin's fingerprint. Current binaries trust `strata.key`
/// (`crates/strata-verify/src/readonly.rs:89`) and the embedded receipt key
/// (`crates/strata-verify/src/migration.rs:89`), so this is a real failure
/// until the pin check at `crates/strata-verify/src/pin.rs:48` lands.
#[test]
fn verify_rejects_resigned_log_with_new_embedded_key() {
    let work = tempfile::tempdir().unwrap();
    let (store, pin, original_fp, fresh) = resigned_store_with_untouched_pin(work.path());
    let pin_sha = sha256_file(&pin);
    let runs = verify_both(&store, None);
    assert_eq!(pin_sha, sha256_file(&pin), "verify rewrote the pinned key");
    assert_eq!(
        embedded_receipt_key(&store),
        public_key_of_seed(&fresh),
        "embedded receipt key is not the fresh signer"
    );
    for (label, out) in &runs {
        assert_case_a_rejects(out, label, &original_fp);
    }
}

/// Case (b). The attacker also replaces `receipt-signing.key` with the key
/// that signed the new receipt. `--expect-key` of the original fingerprint
/// must still exit non-zero, and verify must print the fingerprint of the
/// key it actually used. A missing flag is a real failure: the swapped pin
/// would otherwise be trusted. It is not blocked.
#[test]
fn verify_rejects_swapped_pin_with_original_expect_key() {
    let work = tempfile::tempdir().unwrap();
    let (store, pin, original_fp, fresh) = resigned_store_with_untouched_pin(work.path());
    fs::write(&pin, fresh).unwrap();
    let inside = store.join(strata_migrate::records::RECEIPT_KEY_FILE);
    if inside.is_file() {
        fs::write(&inside, fresh).unwrap();
    }
    let fresh_fp = signing_fingerprint(&fresh);
    if !expect_key_supported() {
        let plain = verify_both(&store, None);
        let accepted_swap = plain.iter().all(|(_, out)| accepted(out));
        let printed = plain
            .iter()
            .any(|(_, out)| prints_fingerprint(out, &fresh_fp));
        panic!(
            "FAIL: case (b) cannot pin the original fingerprint. `--expect-key` \
             is not implemented (crates/strata-verify/src/bin/strata_verify.rs:10 \
             takes one positional directory). Replacing receipt-signing.key with \
             the attacker's seed is not checked against {original_fp}. Plain \
             verify accepted the swapped pin: {accepted_swap}. It printed \
             `key fingerprint: {fresh_fp}`: {printed}. The comparison that must \
             fail this is crates/strata-verify/src/pin.rs:48. This is a real \
             failure, not blocked and not pending_strata."
        );
    }
    for (label, out) in &verify_both(&store, Some(&original_fp)) {
        assert_case_b_rejects(out, label, &original_fp, &fresh_fp);
    }
}

/// Case (c). An untampered store with `--expect-key` of its own fingerprint
/// must exit 0 and print that fingerprint. Until the flag exists this case
/// is blocked, not a pin-check failure.
#[test]
fn verify_accepts_untampered_store_with_expect_key() {
    let work = tempfile::tempdir().unwrap();
    let (store, pin) = migrated_store(work.path());
    let fingerprint = signing_fingerprint(&fs::read(&pin).unwrap());
    if !expect_key_supported() {
        panic!(
            "BLOCKED: case (c) needs `--expect-key` and a printed signing-key \
             fingerprint (`key fingerprint: {fingerprint}`). \
             crates/strata-verify/src/bin/strata_verify.rs:10 has no such flag \
             yet. This waits on the strata-verify pin fix stacked on #314. \
             It is not a failure of the untouched-pin or swapped-pin cases."
        );
    }
    for (label, out) in &verify_both(&store, Some(&fingerprint)) {
        assert_case_c_accepts(out, label, &fingerprint);
    }
}

/// Inferred v3 `memory_connections` (any `link_type` outside the 8 causal
/// kinds) must import as edge kind `legacy_inferred`. A declared row
/// (`touched`, `anchored_to`, `derived_from`, `supersedes`, `corrects`,
/// `closed_by`, `projected_to`, `evidence_of`) is not an inferred link.
///
/// The official fixture is `v3.1.1-sample.sqlite`. A store from
/// `/workspace/v3-real-stores.tar.gz` is included when that archive is
/// present. Current `extract_edges` (`crates/strata-migrate/src/lib.rs:1293`)
/// rewrites every other v3 string to `derived_from`, so this row fails.
#[test]
fn imported_inferred_links_are_legacy_inferred() {
    let work = tempfile::tempdir().unwrap();
    let mut sources = vec![official_v311_sample(work.path())];
    match real_store_dbs(work.path()) {
        Ok(found) => sources.extend(found),
        Err(err) => panic!("FAIL: {err}"),
    }
    assert!(
        sources
            .iter()
            .any(|source| source.label == "v3.1.1-sample.sqlite"),
        "the official v3.1.1 fixture was not opened"
    );
    let mut failures = Vec::new();
    for source in &sources {
        let before = sha256_file(&source.db);
        let inferred = inferred_link_counts(&source.db);
        let inferred_total: usize = inferred.values().sum();
        if source.label == "v3.1.1-sample.sqlite" {
            assert!(
                inferred_total > 0,
                "official v3.1.1 fixture has no inferred memory_connections"
            );
        }
        if inferred_total == 0 {
            continue;
        }
        let edges = import_log_edges(work.path(), source);
        let after = sha256_file(&source.db);
        if before != after {
            failures.push(format!(
                "{}: import modified the source sqlite",
                source.label
            ));
            continue;
        }
        if let Some(failure) = inferred_kind_failure(source, &inferred, &edges) {
            failures.push(failure);
        }
    }
    if !failures.is_empty() {
        panic!(
            "FAIL: imported inferred links are not edge kind legacy_inferred. \
             crates/strata-migrate/src/lib.rs:1293 rewrites a v3 link_type \
             outside the 8 causal kinds to derived_from. This is a real \
             failure, not pending_strata.\n{}",
            failures.join("\n")
        );
    }
}

/// v3 `last_review`, `due`, `stability`, and `difficulty` must come back
/// unchanged for every card. The official fixture keeps them on
/// `fsrs_cards`. A reachable real store is included too; when that store
/// has no `fsrs_cards` rows the node columns `last_accessed` and
/// `next_review` are the schedule. 4.0's review frame has no timestamp, so
/// the folded card's last review is `last_seq`.
#[test]
fn imported_review_schedule_preserved() {
    let work = tempfile::tempdir().unwrap();
    let mut sources = vec![official_v311_sample(work.path())];
    match one_scheduled_real_store(work.path()) {
        Ok(Some(store)) => sources.push(store),
        Ok(None) => {}
        Err(err) => panic!("FAIL: {err}"),
    }
    let mut reports = Vec::new();
    let mut any_differ = false;
    for source in &sources {
        let before = sha256_file(&source.db);
        let (columns, cards) = v3_schedule(&source.db);
        if source.label == "v3.1.1-sample.sqlite" {
            assert!(
                !cards.is_empty(),
                "official v3.1.1 fixture has no review schedule"
            );
        }
        if cards.is_empty() {
            reports.push(format!(
                "{}: no review schedule to compare ({columns})",
                source.label
            ));
            continue;
        }
        let _edges = import_log_edges(work.path(), source);
        assert_eq!(
            before,
            sha256_file(&source.db),
            "{} import modified the source",
            source.label
        );
        let dump = run_driver(&[
            "dump-schedule",
            work.path()
                .join(format!("log-{}", source.label))
                .to_str()
                .unwrap(),
        ]);
        if dump.get("ok") != Some(&Value::Bool(true)) {
            panic!("FAIL: schedule read failed for {}: {dump}", source.label);
        }
        let (report, differ) = schedule_report(source, &columns, &cards, &dump);
        any_differ |= differ;
        reports.push(report);
    }
    if any_differ {
        panic!(
            "FAIL: imported review schedule was not preserved. \
             4.0 gets last_review from CardState.last_seq \
             (crates/strata-kernel/src/fsrs.rs:138), written as \
             ReviewEvent.event_seq (crates/strata-kernel/src/event.rs:22, \
             crates/strata-migrate/src/lib.rs:859). The review payload is \
             card_id and rating only (crates/strata-store/src/op.rs:43). \
             A v3 timestamp copied into node.legacy is not that seq.\n{}",
            reports.join("\n")
        );
    }
}

/// Retrievability after an explicit review must follow the review time.
/// `ReviewNode` has no timestamp, and there is no test clock, so elapsed
/// days are `last_acked_seq - last_seq`.
#[test]
fn retrievability_uses_review_time() {
    let work = tempfile::tempdir().unwrap();
    let dir = work.path().join("fresh");
    let probe = run_driver(&["probe-retrievability", dir.to_str().unwrap()]);
    if probe.get("ok") != Some(&Value::Bool(true)) {
        panic!("FAIL: retrievability probe failed: {probe}");
    }
    let hooks = probe["clock_hooks_set"]
        .as_array()
        .map(|items| items.len())
        .unwrap_or(0);
    let clock = if hooks == 0 {
        "no test clock hook (checked VESTIGE_NOW_MS, VESTIGE_CLOCK, STRATA_NOW_MS, STRATA_TEST_CLOCK; none is read)"
    } else {
        "a clock env var is set, and the fold still does not read it"
    };
    let r0 = probe["retrievability_after_review"].as_f64();
    let r1 = probe["retrievability_after_seq_advance"].as_f64();
    let review_time = probe["review_has_timestamp"].as_bool() == Some(true);
    let stable_across_seq = match (r0, r1) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        _ => false,
    };
    if review_time && stable_across_seq {
        return;
    }
    panic!(
        "FAIL: retrievability is not computed from the review time. \
         {clock}. created_at_ms={} is the node created_at. \
         review payload fields={}. last_seq={}. \
         R after review={r0:?}; R after a later ingest that did not review \
         this card={r1:?}. Seq moved and the review time did not, so R \
         moved with seq.\n{}",
        probe["created_at_ms"],
        probe["review_payload_fields"],
        probe["last_seq"],
        elapsed_day_paths()
    );
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
    _work: Option<tempfile::TempDir>,
    src: std::path::PathBuf,
    db: std::path::PathBuf,
    dest: std::path::PathBuf,
    staging: std::path::PathBuf,
    before_sha: String,
    dest_exists: bool,
    staging_exists: bool,
}

#[derive(Clone, Copy)]
enum KillPoint {
    BeforePublish,
    AfterPublish,
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

fn kill_migrate(point: KillPoint) -> KilledImport {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    inflate_nodes(&db, 8000);
    let killed = kill_into(&src, &work.path().join(dest_name(point)), point);
    KilledImport {
        _work: Some(work),
        ..killed
    }
}

fn dest_name(point: KillPoint) -> &'static str {
    match point {
        KillPoint::BeforePublish => "before-publish",
        KillPoint::AfterPublish => "after-publish",
    }
}

/// SIGKILL `vestige migrate-to-strata --to <dest>`. Before the publish rename
/// the kill lands on a staging segment whose size grew between two polls.
/// After the rename it lands once `<dest>` exists and the process is still
/// alive. The sleep-window env var is not set. A parent
/// `receipt-signing.key` is recorded, not treated as a leak.
fn kill_into(src: &std::path::Path, dest: &std::path::Path, point: KillPoint) -> KilledImport {
    let db = src.join("vestige.db");
    let before_sha = sha256_file(&db);
    let attempts = match point {
        KillPoint::BeforePublish => 1,
        KillPoint::AfterPublish => 4,
    };
    for attempt in 0..attempts {
        let dest = if attempt == 0 {
            dest.to_path_buf()
        } else {
            dest.with_file_name(format!(
                "{}-retry-{attempt}",
                dest.file_name()
                    .unwrap_or(OsStr::new("dest"))
                    .to_string_lossy()
            ))
        };
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
        let caught = watch_and_kill(&mut child, &dest, &staging, point);
        let _ = child.wait();
        if !caught {
            continue;
        }
        let dest_exists = dest.exists();
        let staging_exists = staging.exists();
        let key_exists = receipt_key_path(&dest).is_file();
        eprintln!(
            "post-kill point={} dest_exists={dest_exists} staging_exists={staging_exists} receipt_key_exists={key_exists} dest={}",
            dest_name(point),
            dest.display()
        );
        return KilledImport {
            _work: None,
            src: src.to_path_buf(),
            db,
            dest,
            staging,
            before_sha,
            dest_exists,
            staging_exists,
        };
    }
    missing(&format!(
        "could not SIGKILL vestige migrate-to-strata at {}. Before the publish \
         rename the process must be killed while a staging segment is growing. \
         After the rename it must be killed once <dest> exists and the process \
         is still alive.",
        dest_name(point)
    ));
}

fn watch_and_kill(
    child: &mut Child,
    dest: &std::path::Path,
    staging: &std::path::Path,
    point: KillPoint,
) -> bool {
    let started = Instant::now();
    let mut previous = 0u64;
    while started.elapsed() < Duration::from_secs(180) {
        let exited = child.try_wait().expect("poll").is_some();
        match point {
            KillPoint::BeforePublish => {
                if exited || dest.exists() {
                    return false;
                }
                let bytes = segment_bytes(staging);
                if previous > 0 && bytes > previous {
                    let _ = child.kill();
                    return true;
                }
                if bytes > 0 {
                    previous = bytes;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            KillPoint::AfterPublish => {
                if dest.exists() && !exited {
                    let _ = child.kill();
                    return true;
                }
                if exited {
                    return false;
                }
                std::thread::yield_now();
            }
        }
    }
    let _ = child.kill();
    false
}

fn assert_killed_before_publish(killed: &KilledImport) {
    assert_eq!(
        killed.before_sha,
        sha256_file(&killed.db),
        "SIGKILL path modified the v3 file"
    );
    assert!(
        !killed.dest_exists,
        "pre-rename SIGKILL left <dest> at {}. Only <dest>{} may hold the partial log. \
         A parent {} is expected and is not a leak.",
        killed.dest.display(),
        strata_migrate::STAGING_SUFFIX,
        strata_migrate::records::RECEIPT_KEY_FILE
    );
    if killed.staging_exists {
        assert!(
            killed.staging.is_dir(),
            "staging path is not a directory: {}",
            killed.staging.display()
        );
    }
    assert_receipt_key_allowed(&killed.dest);
}

fn receipt_key_path(dest: &std::path::Path) -> std::path::PathBuf {
    dest.parent()
        .unwrap_or(dest)
        .join(strata_migrate::records::RECEIPT_KEY_FILE)
}

fn assert_receipt_key_allowed(dest: &std::path::Path) {
    let key = receipt_key_path(dest);
    if key.exists() {
        assert!(
            key.is_file(),
            "parent {} is not a file",
            strata_migrate::records::RECEIPT_KEY_FILE
        );
    }
}

fn migrate_to(src: &std::path::Path, dest: &std::path::Path) -> CmdOut {
    let home = tempfile::tempdir().unwrap();
    run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(180),
    )
}

fn assert_rerun_keeps_v3(killed: &KilledImport, again: &CmdOut) {
    let blob = format!("{}{}", again.stdout, again.stderr);
    assert_eq!(
        killed.before_sha,
        sha256_file(&killed.db),
        "rerun modified the v3 file"
    );
    assert_eq!(
        again.status,
        Some(0),
        "rerun after SIGKILL must exit 0. output: {}",
        blob.chars().take(800).collect::<String>()
    );
}

struct FrameCounts {
    nodes: usize,
    edges: usize,
    fsrs: usize,
    receipts: usize,
    kinds: Vec<u64>,
    nodes_body: Value,
    edges_body: Value,
}

/// Node, edge, FSRS, and receipt counts from log frames. Not the migrate
/// stdout line.
fn frame_counts(dir: &std::path::Path) -> FrameCounts {
    let dump = run_driver(&["dump-migration", dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&Value::Bool(true)) {
        panic!("frame read failed for {}: {dump}", dir.display());
    }
    let kinds = dump["kinds"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|kind| kind.as_u64().unwrap_or(0))
        .collect::<Vec<_>>();
    let tally = |kind: u8| kinds.iter().filter(|got| **got == u64::from(kind)).count();
    FrameCounts {
        nodes: tally(strata_migrate::records::KIND_NODE),
        edges: tally(strata_migrate::records::KIND_EDGE),
        fsrs: tally(strata_migrate::records::KIND_FSRS_REVIEW),
        receipts: tally(strata_migrate::records::KIND_MIGRATION_RECEIPT),
        kinds,
        nodes_body: dump.get("nodes").cloned().unwrap_or(Value::Null),
        edges_body: dump.get("edges").cloned().unwrap_or(Value::Null),
    }
}

fn assert_same_frames(got: &FrameCounts, want: &FrameCounts) {
    assert_eq!(
        (got.nodes, got.edges, got.fsrs, got.receipts),
        (want.nodes, want.edges, want.fsrs, want.receipts),
        "log frame counts diverged (nodes, edges, fsrs, receipts)"
    );
    assert!(got.receipts >= 1, "published log has no receipt frame");
    assert_eq!(got.kinds, want.kinds, "frame kind sequence diverged");
    assert_eq!(
        got.nodes_body, want.nodes_body,
        "node frame payloads diverged"
    );
    assert_eq!(
        got.edges_body, want.edges_body,
        "edge frame payloads diverged"
    );
}

fn cmd_excerpt(out: &CmdOut) -> String {
    format!("{}{}", out.stdout, out.stderr)
        .chars()
        .take(600)
        .collect()
}

fn embedded_receipt_key(dir: &std::path::Path) -> [u8; 32] {
    migration_receipt(dir).verifying_key
}

fn migration_receipt(dir: &std::path::Path) -> strata_migrate::records::MigrationReceipt {
    for path in segment_paths(dir) {
        let (_header, frames, _trailer) = split_segment(&fs::read(&path).unwrap())
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        for frame in frames {
            if frame.kind == strata_migrate::records::KIND_MIGRATION_RECEIPT {
                return strata_migrate::records::decode_receipt(&frame.payload)
                    .unwrap_or_else(|err| panic!("receipt decode {}: {err}", path.display()));
            }
        }
    }
    panic!("no migration receipt in {}", dir.display());
}

/// The new receipt was signed by the on-disk key. Checking it with
/// `verify_signature` alone uses the key embedded in that same receipt.
/// Swap in the old log's embedded key so a replaced file fails here.
fn assert_new_receipt_matches_old_key(dir: &std::path::Path, old_embedded: &[u8; 32]) {
    let receipt = migration_receipt(dir);
    let mut under_old = receipt.clone();
    under_old.verifying_key = *old_embedded;
    assert!(
        under_old.verify_signature(),
        "new receipt does not verify under the public key embedded in the old log"
    );
    assert_eq!(
        receipt.verifying_key, *old_embedded,
        "new receipt embedded a different key than the old log"
    );
}

fn public_key_of_seed(bytes: &[u8]) -> [u8; 32] {
    let seed: [u8; 32] = bytes
        .try_into()
        .unwrap_or_else(|_| panic!("signing seed is {} bytes", bytes.len()));
    ed25519_dalek::SigningKey::from_bytes(&seed)
        .verifying_key()
        .to_bytes()
}

fn segment_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut paths = seg_files(dir);
    paths.sort();
    paths
}

fn split_segment(
    bytes: &[u8],
) -> Result<(SegmentHeader, Vec<Frame>, Option<SegmentTrailer>), String> {
    if bytes.len() < HEADER_WIRE_SIZE {
        return Err("segment shorter than a header".into());
    }
    let header: SegmentHeader =
        borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).map_err(|err| format!("header: {err}"))?;
    let mut frames = Vec::new();
    let mut off = HEADER_WIRE_SIZE;
    loop {
        let rem = bytes.len().saturating_sub(off);
        if rem == 0 {
            return Ok((header, frames, None));
        }
        if rem == TRAILER_WIRE_SIZE {
            let trailer: SegmentTrailer =
                borsh::from_slice(&bytes[off..]).map_err(|err| format!("trailer: {err}"))?;
            return Ok((header, frames, Some(trailer)));
        }
        let (frame, used) =
            parse_frame(&bytes[off..]).map_err(|err| format!("frame at {off}: {err}"))?;
        off += used;
        frames.push(frame);
    }
}

fn reseal(
    header: &SegmentHeader,
    frames: &[Frame],
    signing: &ed25519_dalek::SigningKey,
) -> Vec<u8> {
    let mut body = borsh::to_vec(header).unwrap();
    let mut prev = header_hash(header);
    let mut leaves = Vec::with_capacity(frames.len());
    for frame in frames {
        let linked = Frame {
            kind: frame.kind,
            payload: frame.payload.clone(),
            payload_blake3: payload_blake3(frame.kind, &frame.payload),
            prev_frame_hash: prev,
        };
        prev = frame_hash(&linked);
        leaves.push(linked.payload_blake3);
        body.extend(borsh::to_vec(&linked).unwrap());
    }
    let root = merkle_root(&leaves);
    let message = signature_message(&header.segment_id, &header.prev_segment_hash, &root);
    let trailer = SegmentTrailer {
        frame_count: frames.len() as u64,
        merkle_root: root,
        signature: signing.sign(&message).to_bytes(),
    };
    body.extend(borsh::to_vec(&trailer).unwrap());
    body
}

/// Rewrite one node and seal the receipt under `fresh`. Segment trailers
/// stay under the on-disk `strata.key`. `receipt-signing.key` is not written.
fn resign_embedded_receipt(dir: &std::path::Path, fresh: &ed25519_dalek::SigningKey) {
    let strata_seed =
        seed_array(&fs::read(dir.join("strata.key")).unwrap()).expect("strata.key is 32 bytes");
    let trailer_key = ed25519_dalek::SigningKey::from_bytes(&strata_seed);
    let paths = segment_paths(dir);
    assert!(!paths.is_empty(), "store has no segment to resign");
    let mut prev_segment = strata::GENESIS_PREV_SEGMENT_HASH;
    let mut changed_node = false;
    let mut changed_receipt = false;
    for path in &paths {
        let bytes = fs::read(path).unwrap();
        let (mut header, mut frames, trailer) =
            split_segment(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        header.prev_segment_hash = prev_segment;
        // `seal` rolls a header-only segment after the signed one. It has no
        // trailer. Keep it unsealed and chain it to the segment we just rewrote.
        if frames.is_empty() && trailer.is_none() {
            fs::write(path, borsh::to_vec(&header).unwrap()).unwrap();
            prev_segment = *blake3::hash(&fs::read(path).unwrap()).as_bytes();
            continue;
        }
        assert!(
            trailer.is_some(),
            "migration segment {} has frames and no trailer",
            path.display()
        );
        for frame in &mut frames {
            if frame.kind == strata_migrate::records::KIND_NODE && !changed_node {
                let mut node = strata_migrate::records::decode_node(&frame.payload)
                    .unwrap_or_else(|err| panic!("node decode: {err}"));
                node.content.push_str(" release-matrix-resigned");
                frame.payload = borsh::to_vec(&node).unwrap();
                changed_node = true;
            }
            if frame.kind == strata_migrate::records::KIND_MIGRATION_RECEIPT {
                let receipt = strata_migrate::records::decode_receipt(&frame.payload)
                    .unwrap_or_else(|err| panic!("receipt decode: {err}"));
                let resealed = strata_migrate::records::MigrationReceipt::seal(receipt.body, fresh);
                frame.payload = borsh::to_vec(&resealed).unwrap();
                changed_receipt = true;
            }
        }
        let encoded = reseal(&header, &frames, &trailer_key);
        prev_segment = *blake3::hash(&encoded).as_bytes();
        fs::write(path, &encoded).unwrap();
    }
    assert!(changed_node, "resign found no node frame");
    assert!(changed_receipt, "resign found no receipt frame");
}

fn newest_trailer_verifies(dir: &std::path::Path, seed: &[u8]) -> bool {
    let Some(seed) = seed_array(seed) else {
        return false;
    };
    log_continues_under(dir, &seed)
}

fn seed_array(bytes: &[u8]) -> Option<[u8; 32]> {
    bytes.try_into().ok()
}

fn log_continues_under(dir: &std::path::Path, seed: &[u8; 32]) -> bool {
    let verifying = ed25519_dalek::SigningKey::from_bytes(seed).verifying_key();
    let mut prev_segment = strata::GENESIS_PREV_SEGMENT_HASH;
    let paths = segment_paths(dir);
    if paths.is_empty() {
        return false;
    }
    for (index, path) in paths.iter().enumerate() {
        let bytes = fs::read(path).unwrap_or_default();
        if bytes.len() < HEADER_WIRE_SIZE {
            return false;
        }
        let Ok((header, frames, trailer)) = split_segment(&bytes) else {
            return false;
        };
        if header.prev_segment_hash != prev_segment {
            return false;
        }
        if frames.is_empty() {
            match trailer {
                // `seal` rolls a header-only segment after the signed one.
                None => return index + 1 == paths.len(),
                Some(trailer) => {
                    if trailer.frame_count != 0 || !sealed_under(&verifying, &header, &trailer, &[])
                    {
                        return false;
                    }
                    prev_segment = *blake3::hash(&bytes).as_bytes();
                    continue;
                }
            }
        }
        let mut prev = header_hash(&header);
        let mut leaves = Vec::new();
        for frame in &frames {
            if frame.payload_blake3 != payload_blake3(frame.kind, &frame.payload)
                || frame.prev_frame_hash != prev
            {
                return false;
            }
            prev = frame_hash(frame);
            leaves.push(frame.payload_blake3);
        }
        match trailer {
            Some(trailer) => {
                if !sealed_under(&verifying, &header, &trailer, &leaves) {
                    return false;
                }
                if trailer.frame_count != frames.len() as u64 {
                    return false;
                }
                prev_segment = *blake3::hash(&bytes).as_bytes();
            }
            None if index + 1 == paths.len() => return true,
            None => return false,
        }
    }
    true
}

fn sealed_under(
    verifying: &ed25519_dalek::VerifyingKey,
    header: &SegmentHeader,
    trailer: &SegmentTrailer,
    leaves: &[[u8; 32]],
) -> bool {
    if trailer.merkle_root != merkle_root(leaves) {
        return false;
    }
    let message = signature_message(
        &header.segment_id,
        &header.prev_segment_hash,
        &trailer.merkle_root,
    );
    let signature = ed25519_dalek::Signature::from_bytes(&trailer.signature);
    verifying.verify(&message, &signature).is_ok()
}

/// The marker's bytes have to sit in a segment that carries a trailer.
/// An unsealed tail proves the write was appended, not that it was signed.
fn marker_in_sealed_segment(dir: &std::path::Path, marker: &str) -> bool {
    let needle = marker.as_bytes();
    segment_paths(dir).into_iter().any(|path| {
        let Ok(bytes) = fs::read(&path) else {
            return false;
        };
        if needle.is_empty() || !bytes.windows(needle.len()).any(|window| window == needle) {
            return false;
        }
        split_segment(&bytes)
            .map(|(_, _, trailer)| trailer.is_some())
            .unwrap_or(false)
    })
}

fn migrated_store(work: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = work.join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let store = work.join("store");
    let migrated = migrate_to(&src, &store);
    assert_eq!(
        migrated.status,
        Some(0),
        "could not build the store: {}",
        cmd_excerpt(&migrated)
    );
    let pin = work.join(strata_migrate::records::RECEIPT_KEY_FILE);
    assert!(pin.is_file(), "store published no receipt key to pin");
    (store, pin)
}

fn resigned_store_with_untouched_pin(
    work: &std::path::Path,
) -> (std::path::PathBuf, std::path::PathBuf, String, [u8; 32]) {
    let (store, pin) = migrated_store(work);
    let original_fp = signing_fingerprint(&fs::read(&pin).unwrap());
    let pin_sha = sha256_file(&pin);
    let strata_sha = sha256_file(&store.join("strata.key"));
    let mut fresh = [0u8; 32];
    fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut fresh)
        .unwrap();
    resign_embedded_receipt(&store, &ed25519_dalek::SigningKey::from_bytes(&fresh));
    assert_eq!(
        pin_sha,
        sha256_file(&pin),
        "resign rewrote receipt-signing.key"
    );
    assert_eq!(
        strata_sha,
        sha256_file(&store.join("strata.key")),
        "resign rewrote strata.key"
    );
    assert_ne!(
        embedded_receipt_key(&store),
        public_key_of_seed(&fs::read(&pin).unwrap()),
        "resigned log still embeds the pinned receipt key"
    );
    (store, pin, original_fp, fresh)
}

/// blake3 hex of the 32-byte ed25519 verifying key. That is the fingerprint
/// `--expect-key` compares, and the line verify must print.
fn signing_fingerprint(seed: &[u8]) -> String {
    let public = public_key_of_seed(seed);
    blake3::hash(&public).to_hex().to_string()
}

fn expect_key_supported() -> bool {
    let bare = run_cmd(
        &product_bin("strata-verify"),
        &["--help".to_string()],
        &[],
        &[],
        Duration::from_secs(20),
    );
    let none = run_cmd(
        &product_bin("strata-verify"),
        &[],
        &[],
        &[],
        Duration::from_secs(20),
    );
    let home = tempfile::tempdir().unwrap();
    let via = run_vestige(
        &["strata-verify".into(), "--help".into()],
        home.path(),
        Duration::from_secs(20),
    );
    format!(
        "{}{}{}{}{}{}",
        bare.stdout, bare.stderr, none.stdout, none.stderr, via.stdout, via.stderr
    )
    .contains("--expect-key")
}

fn verify_both(dir: &std::path::Path, expect: Option<&str>) -> Vec<(&'static str, CmdOut)> {
    let mut args = Vec::new();
    if let Some(fingerprint) = expect {
        args.push("--expect-key".to_string());
        args.push(fingerprint.to_string());
    }
    args.push(dir.display().to_string());
    let plain = run_cmd(
        &product_bin("strata-verify"),
        &args,
        &[],
        &[],
        Duration::from_secs(60),
    );
    let mut vestige_args = vec!["strata-verify".to_string()];
    vestige_args.extend(args);
    let home = tempfile::tempdir().unwrap();
    let via = run_vestige(&vestige_args, home.path(), Duration::from_secs(60));
    vec![("strata-verify", plain), ("vestige strata-verify", via)]
}

fn prints_fingerprint(out: &CmdOut, fingerprint: &str) -> bool {
    format!("{}{}", out.stdout, out.stderr).contains(&format!("key fingerprint: {fingerprint}"))
}

fn accepted(out: &CmdOut) -> bool {
    let blob = format!("{}{}", out.stdout, out.stderr);
    out.status == Some(0) || blob.lines().any(|line| line.trim() == "OK")
}

fn assert_case_a_rejects(out: &CmdOut, label: &str, original_fp: &str) {
    let blob = format!("{}{}", out.stdout, out.stderr);
    if accepted(out) {
        panic!(
            "FAIL: case (a) {label} accepted a re-signed log while receipt-signing.key \
             was untouched. The embedded verifying key does not match that pin. \
             Verify loads strata.key at crates/strata-verify/src/readonly.rs:89 and \
             checks the receipt against its embedded key at \
             crates/strata-verify/src/migration.rs:89. The comparison that must \
             reject this is crates/strata-verify/src/pin.rs:48. This is a real \
             failure, not pending_strata. output: {}",
            blob.chars().take(800).collect::<String>()
        );
    }
    if !prints_fingerprint(out, original_fp) {
        panic!(
            "FAIL: case (a) {label} exited {:?} without printing `key fingerprint: \
             {original_fp}`. The pin file's fingerprint is the signing key verify \
             must name. crates/strata-verify/src/readonly.rs:89 and \
             crates/strata-verify/src/migration.rs:89 still choose the key. This \
             is a real failure, not pending_strata. output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        );
    }
    let lower = blob.to_lowercase();
    if !lower.contains("receipt-signing.key") || !lower.contains("match") {
        panic!(
            "FAIL: case (a) {label} exited {:?} without naming an embedded-key \
             mismatch against receipt-signing.key. A torn trailer is not this \
             check. The decision is crates/strata-verify/src/pin.rs:48. This is \
             a real failure, not pending_strata. output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        );
    }
}

fn assert_case_b_rejects(out: &CmdOut, label: &str, original_fp: &str, fresh_fp: &str) {
    let blob = format!("{}{}", out.stdout, out.stderr);
    if accepted(out) || !blob.to_lowercase().contains("does not match --expect-key") {
        panic!(
            "FAIL: case (b) {label} did not reject `--expect-key {original_fp}` after \
             receipt-signing.key was replaced. Exit {:?}. The comparison is \
             crates/strata-verify/src/pin.rs:48. This is a real failure, not \
             blocked and not pending_strata. output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        );
    }
    if !prints_fingerprint(out, fresh_fp) {
        panic!(
            "FAIL: case (b) {label} exited {:?} without printing `key fingerprint: \
             {fresh_fp}`, the key now in the folder. output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        );
    }
}

fn assert_case_c_accepts(out: &CmdOut, label: &str, fingerprint: &str) {
    let blob = format!("{}{}", out.stdout, out.stderr);
    assert!(
        out.status == Some(0) && blob.lines().any(|line| line.trim() == "OK"),
        "case (c) {label} rejected an untampered store with `--expect-key` of \
         its own fingerprint (exit {:?}): {}",
        out.status,
        blob.chars().take(800).collect::<String>()
    );
    assert!(
        prints_fingerprint(out, fingerprint),
        "case (c) {label} exited 0 without printing `key fingerprint: {fingerprint}`: {}",
        blob.chars().take(800).collect::<String>()
    );
}

const CAUSAL_EDGE_KINDS: [&str; 8] = [
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

struct InferredSource {
    label: String,
    /// Directory or sqlite file passed to `migrate-to-strata --from`.
    from: std::path::PathBuf,
    db: std::path::PathBuf,
    accept_wal: bool,
}

fn official_v311_sample(work: &std::path::Path) -> InferredSource {
    let bundled = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    let sha = sha256_file(&bundled);
    assert_eq!(
        sha, "961f12d1750dbd2f6e6a8fc365c4a4bb42dd1b7e49cbf465985a36b665e10479",
        "v3.1.1-sample.sqlite is not the official fixture"
    );
    let db = work.join("v3.1.1-sample.sqlite");
    fs::copy(&bundled, &db).unwrap();
    InferredSource {
        label: "v3.1.1-sample.sqlite".into(),
        from: db.clone(),
        db,
        accept_wal: false,
    }
}

/// Stores inside `/workspace/v3-real-stores.tar.gz`, or `VESTIGE_V3_REAL_STORES`
/// when that path is a file. Absent archive means the official fixture only.
fn real_store_dbs(work: &std::path::Path) -> Result<Vec<InferredSource>, String> {
    let from_env = std::env::var_os("VESTIGE_V3_REAL_STORES").map(std::path::PathBuf::from);
    let archive = from_env
        .into_iter()
        .chain(std::iter::once(std::path::PathBuf::from(
            "/workspace/v3-real-stores.tar.gz",
        )))
        .find(|path| path.is_file());
    let Some(archive) = archive else {
        return Ok(Vec::new());
    };
    let extracted = work.join("real-stores");
    fs::create_dir_all(&extracted).unwrap();
    let tar = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(&archive)
        .arg("-C")
        .arg(&extracted)
        .output()
        .map_err(|err| format!("tar {}: {err}", archive.display()))?;
    if !tar.status.success() {
        return Err(format!(
            "could not read {}: {}",
            archive.display(),
            String::from_utf8_lossy(&tar.stderr)
        ));
    }
    let mut dbs = Vec::new();
    collect_vestige_dbs(&extracted, &mut dbs);
    dbs.sort();
    let mut sources = Vec::new();
    for db in dbs {
        let label = db
            .parent()
            .and_then(|dir| dir.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("store")
            .to_string();
        if label.contains("shim") {
            continue;
        }
        let wal = db.with_file_name("vestige.db-wal");
        let accept_wal = wal.is_file() && fs::metadata(&wal).map(|m| m.len() > 0).unwrap_or(false);
        let from = db.parent().unwrap_or(db.as_path()).to_path_buf();
        sources.push(InferredSource {
            label,
            from,
            db,
            accept_wal,
        });
    }
    if sources.is_empty() {
        return Err(format!("{} contained no vestige.db", archive.display()));
    }
    Ok(sources)
}

fn collect_vestige_dbs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_vestige_dbs(&path, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some("vestige.db") {
            out.push(path);
        }
    }
}

fn inferred_link_counts(db: &std::path::Path) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for row in sqlite_text(
        db,
        "SELECT link_type, COUNT(*) FROM memory_connections GROUP BY link_type ORDER BY link_type",
    ) {
        let link_type = row.first().cloned().unwrap_or_default();
        if CAUSAL_EDGE_KINDS.contains(&link_type.as_str()) {
            continue;
        }
        let count = row.get(1).and_then(|cell| cell.parse().ok()).unwrap_or(0);
        if count > 0 {
            counts.insert(link_type, count);
        }
    }
    counts
}

fn import_log_edges(work: &std::path::Path, source: &InferredSource) -> Vec<Value> {
    let dest = work.join(format!("log-{}", source.label));
    let mut args = vec![
        "migrate-to-strata".to_string(),
        "--from".to_string(),
        source.from.display().to_string(),
        "--to".to_string(),
        dest.display().to_string(),
    ];
    if source.accept_wal {
        args.push("--accept-wal-snapshot".into());
    }
    let home = tempfile::tempdir().unwrap();
    let migrated = run_vestige(&args, home.path(), Duration::from_secs(180));
    assert_eq!(
        migrated.status,
        Some(0),
        "import of {} failed: {}",
        source.label,
        cmd_excerpt(&migrated)
    );
    let dump = run_driver(&["dump-migration", dest.to_str().unwrap()]);
    if dump.get("ok") != Some(&Value::Bool(true)) {
        panic!("edge read failed for {}: {dump}", source.label);
    }
    dump["edges"].as_array().cloned().unwrap_or_default()
}

struct V3Card {
    id: String,
    last_review: String,
    due: String,
    stability: String,
    difficulty: String,
}

fn one_scheduled_real_store(work: &std::path::Path) -> Result<Option<InferredSource>, String> {
    let sources = real_store_dbs(work)?;
    Ok(sources
        .into_iter()
        .find(|source| !v3_schedule(&source.db).1.is_empty()))
}

fn v3_schedule(db: &std::path::Path) -> (String, Vec<V3Card>) {
    let fsrs = sqlite_text(
        db,
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fsrs_cards'",
    );
    if !fsrs.is_empty() {
        let rows = sqlite_text(
            db,
            "SELECT memory_id, last_review, due_date, stability, difficulty \
             FROM fsrs_cards ORDER BY memory_id",
        );
        if !rows.is_empty() {
            return (
                "fsrs_cards.last_review, fsrs_cards.due_date, fsrs_cards.stability, fsrs_cards.difficulty".into(),
                rows.into_iter().map(card_from_row).collect(),
            );
        }
    }
    let names: std::collections::BTreeSet<String> =
        sqlite_text(db, "PRAGMA table_info(knowledge_nodes)")
            .into_iter()
            .filter_map(|row| row.get(1).cloned())
            .collect();
    if !names.contains("stability") || !names.contains("difficulty") {
        return (
            "knowledge_nodes has no stability/difficulty".into(),
            Vec::new(),
        );
    }
    let last = if names.contains("last_review") {
        "last_review"
    } else if names.contains("last_accessed") {
        "last_accessed"
    } else {
        "''"
    };
    let due = if names.contains("due_date") {
        "due_date"
    } else if names.contains("next_review") {
        "next_review"
    } else if names.contains("due") {
        "due"
    } else {
        "''"
    };
    let rows = sqlite_text(
        db,
        &format!(
            "SELECT id, {last}, {due}, stability, difficulty FROM knowledge_nodes ORDER BY id"
        ),
    );
    (
        format!("knowledge_nodes.{last}, knowledge_nodes.{due}, knowledge_nodes.stability, knowledge_nodes.difficulty"),
        rows.into_iter().map(card_from_row).collect(),
    )
}

fn card_from_row(row: Vec<String>) -> V3Card {
    V3Card {
        id: row.first().cloned().unwrap_or_default(),
        last_review: row.get(1).cloned().unwrap_or_default(),
        due: row.get(2).cloned().unwrap_or_default(),
        stability: row.get(3).cloned().unwrap_or_default(),
        difficulty: row.get(4).cloned().unwrap_or_default(),
    }
}

fn schedule_report(
    source: &InferredSource,
    columns: &str,
    cards: &[V3Card],
    dump: &Value,
) -> (String, bool) {
    let imported = dump["cards"].as_array().cloned().unwrap_or_default();
    let mut match_n = [0usize; 4];
    let mut differ_n = [0usize; 4];
    let mut example = String::new();
    for card in cards {
        let got = imported
            .iter()
            .find(|row| row["legacy_id"].as_str() == Some(card.id.as_str()));
        let last_seq = got
            .and_then(|row| row["last_seq"].as_u64())
            .map(|n| n.to_string());
        let stability = got.and_then(|row| json_number(&row["stability"]));
        let difficulty = got.and_then(|row| json_number(&row["difficulty"]));
        let due = got.and_then(|row| row["due"].as_str()).unwrap_or("");
        let fields = [
            (
                "last_review",
                text_eq(&card.last_review, last_seq.as_deref().unwrap_or("")),
            ),
            ("due", text_eq(&card.due, due)),
            ("stability", number_eq(&card.stability, stability)),
            ("difficulty", number_eq(&card.difficulty, difficulty)),
        ];
        for (index, (_, same)) in fields.iter().enumerate() {
            if *same {
                match_n[index] += 1;
            } else {
                differ_n[index] += 1;
            }
        }
        if example.is_empty() && fields.iter().any(|(_, same)| !same) {
            let legacy = got
                .and_then(|row| row["legacy"].as_object())
                .map(|map| {
                    map.iter()
                        .map(|(key, value)| format!("{key}={value}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            example = format!(
                "  example {}: v3 last_review={} due={} stability={} difficulty={}; \
                 4.0 last_seq={} stability={} difficulty={} due=null; legacy [{legacy}]",
                card.id,
                card.last_review,
                card.due,
                card.stability,
                card.difficulty,
                last_seq.unwrap_or_else(|| "absent".into()),
                got.map(|row| row["stability"].to_string())
                    .unwrap_or_else(|| "absent".into()),
                got.map(|row| row["difficulty"].to_string())
                    .unwrap_or_else(|| "absent".into()),
            );
        }
    }
    let names = ["last_review", "due", "stability", "difficulty"];
    let counts = names
        .iter()
        .enumerate()
        .map(|(index, name)| format!("{name} match={} differ={}", match_n[index], differ_n[index]))
        .collect::<Vec<_>>()
        .join("; ");
    let differ = differ_n.iter().any(|n| *n > 0);
    (
        format!(
            "{}: cards={} via {columns}; {counts}\n{example}",
            source.label,
            cards.len()
        ),
        differ,
    )
}

fn text_eq(source: &str, got: &str) -> bool {
    source.is_empty() && got.is_empty() || source == got
}

fn number_eq(source: &str, got: Option<f64>) -> bool {
    match (source.parse::<f64>(), got) {
        (Ok(want), Some(have)) => (want - have).abs() <= 1e-4,
        (Err(_), None) if source.is_empty() => true,
        _ => false,
    }
}

fn json_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|n| n as f64))
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

fn elapsed_day_paths() -> String {
    "\
elapsed-day paths:
  search: crates/vestige-core/src/search/temporal.rs:69 uses created_at (age_days = now - created_at). \
Strata search_records is pending at crates/vestige-mcp/src/strata_memory.rs:279. \
project_node copies created_at into last_accessed at strata_memory.rs:164 and calls \
store.retrievability at strata_memory.rs:157.
  forgetting: crates/vestige-core/src/storage/sqlite/lifecycle.rs:1030 uses last_accessed \
(days_since = now - last_accessed). mark_reviewed sets last_review from last_accessed at \
lifecycle.rs:24 and elapsed at lifecycle.rs:32. The strata fold uses \
current_seq - last_seq at crates/strata-kernel/src/fsrs.rs:196, and store.retrievability \
passes last_acked_seq at crates/strata-store/src/store.rs:761.
  project preview floor: crates/vestige-core/src/projection.rs:127 compares stored \
retention_strength to min_retention and does not read a review timestamp. \
Strata projection_candidates is pending at crates/vestige-mcp/src/strata_memory.rs:792. \
The retention project_node would supply is the seq retrievability (strata_memory.rs:175).
  forgotten_lesson: crates/vestige-mcp/src/tools/forgotten_lesson.rs:126 uses \
failure_at - node.last_accessed. On a strata node that last_accessed is created_at \
(crates/vestige-mcp/src/strata_memory.rs:164)."
        .into()
}

fn inferred_kind_failure(
    source: &InferredSource,
    inferred: &std::collections::BTreeMap<String, usize>,
    edges: &[Value],
) -> Option<String> {
    let inferred_total: usize = inferred.values().sum();
    let mut mapped: std::collections::BTreeMap<(String, String), usize> =
        std::collections::BTreeMap::new();
    let mut legacy_inferred_edges = 0usize;
    let mut causal_inferred = 0usize;
    for edge in edges {
        let kind = edge["link_type"].as_str().unwrap_or("").to_string();
        if kind == "legacy_inferred" {
            legacy_inferred_edges += 1;
        }
        let v3_type = edge["legacy_link_type"].as_str().unwrap_or("").to_string();
        if !inferred.contains_key(&v3_type) {
            continue;
        }
        if CAUSAL_EDGE_KINDS.contains(&kind.as_str()) {
            causal_inferred += 1;
        }
        let flag = edge["legacy_inferred"].as_bool().unwrap_or(false);
        let shown = format!("{kind} (legacy_inferred={flag})");
        *mapped.entry((v3_type, shown)).or_default() += 1;
    }
    let kinds_ok = edges
        .iter()
        .filter(|edge| {
            let v3_type = edge["legacy_link_type"].as_str().unwrap_or("");
            inferred.contains_key(v3_type)
        })
        .all(|edge| edge["link_type"].as_str() == Some("legacy_inferred"));
    if kinds_ok && causal_inferred == 0 && legacy_inferred_edges == inferred_total {
        return None;
    }
    let histogram = inferred
        .iter()
        .map(|(kind, count)| format!("{kind}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mapping = if mapped.is_empty() {
        "no imported edge carried an inferred v3 link_type".to_string()
    } else {
        mapped
            .iter()
            .map(|((v3_type, kind), count)| format!("  {v3_type} -> {kind} x{count}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    Some(format!(
        "{}: source inferred={inferred_total} [{histogram}]; \
         legacy_inferred edges={legacy_inferred_edges}; \
         inferred links imported as a causal kind={causal_inferred}\n{mapping}",
        source.label
    ))
}

fn assert_strata_verify_ok(dir: &std::path::Path) {
    let verify = run_cmd(
        &product_bin("strata-verify"),
        &[dir.display().to_string()],
        &[],
        &[],
        Duration::from_secs(60),
    );
    let blob = format!("{}{}", verify.stdout, verify.stderr);
    assert!(
        verify.status == Some(0) && blob.contains("OK"),
        "strata-verify did not accept {} (exit {:?}): {}",
        dir.display(),
        verify.status,
        blob.chars().take(800).collect::<String>()
    );
}

struct V311Asset {
    archive: &'static str,
    sha256: &'static str,
}

/// Official `samvallad33/vestige` v3.1.1 release assets. The pin is the
/// GitHub release digest, not an npm tarball.
fn v311_release_asset() -> Option<V311Asset> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some(V311Asset {
            archive: "vestige-mcp-x86_64-unknown-linux-gnu.tar.gz",
            sha256: "6b698908cb33b1827d6a75715c7b7c949dae0dc190fc8719114bac24d9e21991",
        }),
        ("macos", "aarch64") => Some(V311Asset {
            archive: "vestige-mcp-aarch64-apple-darwin.tar.gz",
            sha256: "d63f57c54e1d0f854b5411c7953d203e79118c300e03363c2cb6d69972ee307a",
        }),
        ("macos", "x86_64") => Some(V311Asset {
            archive: "vestige-mcp-x86_64-apple-darwin.tar.gz",
            sha256: "c78ba5f0ec8f142497a09c310e6c9d728f6ccfad213b3d9df60f678245d6f885",
        }),
        ("windows", "x86_64") => Some(V311Asset {
            archive: "vestige-mcp-x86_64-pc-windows-msvc.zip",
            sha256: "6a2c258f318be8a5d1211ec90e3a1208e7f5fd9a652f3ac1486a9822e607d7dc",
        }),
        _ => None,
    }
}

fn skip_release_asset(why: &str) -> ! {
    panic!("SKIP: {why}");
}

fn sqlite_family_hashes(dir: &std::path::Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in ["vestige.db", "vestige.db-wal", "vestige.db-shm"] {
        let path = dir.join(name);
        if path.is_file() {
            out.push((name.to_string(), sha256_file(&path)));
        }
    }
    assert!(
        out.iter().any(|(name, _)| name == "vestige.db"),
        "data dir has no vestige.db"
    );
    out
}

fn log_contains_marker(data_dir: &std::path::Path, marker: &str) -> bool {
    let needle = marker.as_bytes();
    seg_files(&data_dir.join("log")).into_iter().any(|path| {
        fs::read(&path)
            .map(|bytes| bytes.windows(needle.len()).any(|window| window == needle))
            .unwrap_or(false)
    })
}

fn stats_memory_count(blob: &str) -> Option<usize> {
    let plain = strip_ansi(blob);
    for line in plain.lines() {
        let Some(rest) = line.split("Total Memories").nth(1) else {
            continue;
        };
        let digits: String = rest
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    None
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

fn fetch_v311_vestige(dest: &std::path::Path) -> std::path::PathBuf {
    let Some(asset) = v311_release_asset() else {
        skip_release_asset(&format!(
            "no official v3.1.1 GitHub release asset for {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
    };
    fs::create_dir_all(dest).unwrap();
    let archive = dest.join(asset.archive);
    let url = format!(
        "https://github.com/samvallad33/vestige/releases/download/v3.1.1/{}",
        asset.archive
    );
    let fetched = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "--retry-delay", "2", "-o"])
        .arg(&archive)
        .arg(&url)
        .output();
    let fetched = match fetched {
        Ok(output) => output,
        Err(err) => skip_release_asset(&format!("could not run curl to download {url}: {err}")),
    };
    if !fetched.status.success() {
        let err = String::from_utf8_lossy(&fetched.stderr);
        skip_release_asset(&format!(
            "could not download the official v3.1.1 asset {url}: {}",
            err.chars().take(400).collect::<String>()
        ));
    }
    let got = sha256_file(&archive);
    assert_eq!(
        got, asset.sha256,
        "official v3.1.1 asset {} sha256 {got} != pinned {}",
        asset.archive, asset.sha256
    );
    let extracted = dest.join("extract");
    fs::create_dir_all(&extracted).unwrap();
    let mut tar = Command::new("tar");
    if asset.archive.ends_with(".zip") {
        tar.arg("-xf");
    } else {
        tar.arg("-xzf");
    }
    let status = tar
        .arg(&archive)
        .arg("-C")
        .arg(&extracted)
        .status()
        .unwrap_or_else(|err| panic!("HARNESS: tar failed: {err}"));
    assert!(status.success(), "could not extract {}", asset.archive);
    let binary_name = if cfg!(windows) {
        "vestige.exe"
    } else {
        "vestige"
    };
    find_file_named(&extracted, binary_name).unwrap_or_else(|| {
        panic!(
            "HARNESS: {binary_name} is missing from the official v3.1.1 asset {}",
            asset.archive
        )
    })
}

fn find_file_named(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(rd) = fs::read_dir(&current) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) == Some(name) {
                return Some(path);
            }
        }
    }
    None
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
