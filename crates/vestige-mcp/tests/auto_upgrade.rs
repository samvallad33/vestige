//! Real v3 fixture upgrades: happy path, injected import/verify failures, and
//! a SIGKILL mid-upgrade followed by a relaunch.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use vestige_mcp::auto_upgrade::{
    self, LOG_DIR_NAME, UPGRADE_LOG_NAME, UpgradeOptions, UpgradeStatus, V311_RELEASE,
    staging_directory,
};

fn fixture_db() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite")
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn plant(dir: &Path) -> PathBuf {
    let db = dir.join("vestige.db");
    fs::copy(fixture_db(), &db).unwrap();
    db
}

fn assert_failure_message(err: &str, log_path: &Path) {
    assert!(err.contains("upgrade failed"), "{err}");
    assert!(err.contains("untouched"), "{err}");
    assert!(err.contains("v3.1.1"), "{err}");
    assert!(err.contains(V311_RELEASE), "{err}");
    assert!(err.contains(&log_path.display().to_string()), "{err}");
    let logged = fs::read_to_string(log_path).unwrap();
    assert!(logged.contains("upgrade failed"), "{logged}");
    assert!(logged.contains(V311_RELEASE), "{logged}");
}

fn snapshot(log_dir: &Path) -> strata_migrate::snapshot::Snapshot {
    let log = strata::StrataLog::open(log_dir).unwrap();
    strata_migrate::read_snapshot(&log).unwrap()
}

fn knowledge_ids(db: &Path) -> Vec<String> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap();
    let mut stmt = conn
        .prepare("SELECT id FROM knowledge_nodes ORDER BY id")
        .unwrap();
    stmt.query_map([], |row| row.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
}

fn assert_fixture_landed(db: &Path, log_dir: &Path) {
    let verified = strata_verify::migration::verify_migrated_log(log_dir).unwrap();
    assert!(verified.ok, "{verified:?}");
    let snap = snapshot(log_dir);
    let ids = knowledge_ids(db);
    assert!(ids.len() >= 4, "fixture lost its memories");
    for id in &ids {
        let node = snap
            .nodes
            .iter()
            .find(|node| &node.legacy_id == id)
            .unwrap_or_else(|| panic!("dropped memory {id}"));
        if id.starts_with("11111111") {
            let stability = node
                .legacy
                .iter()
                .find(|(key, _)| key == "fsrs_cards.stability")
                .map(|(_, value)| value.as_str())
                .unwrap_or("");
            let parsed: f64 = stability.parse().unwrap_or(f64::NAN);
            assert!((parsed - 12.25).abs() < 1e-9, "stability {stability}");
        }
    }
    let causal = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "causal")
        .expect("causal link dropped");
    assert!(causal.legacy_inferred);
    assert_eq!(causal.link_type, "legacy_inferred");
    let semantic = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "semantic")
        .expect("semantic link dropped");
    assert!(semantic.legacy_inferred);
    assert_eq!(semantic.link_type, "legacy_inferred");
    let touched = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "touched")
        .expect("touched link dropped");
    assert!(!touched.legacy_inferred);
    assert_eq!(touched.link_type, "touched");
    assert_v3_source_kept(db, log_dir);
}

/// The v3.1.1 fixture's `source` text and its `updated_at` survive the upgrade
/// on `source` / `source_updated_at_ms`. Rows with no source (the walk
/// receipt) stay `None`.
fn assert_v3_source_kept(db: &Path, log_dir: &Path) {
    let snap = snapshot(log_dir);
    let expected_ms = strata_migrate::source::timestamp_ms("2026-02-20T11:30:00+00:00").unwrap();
    let ids = knowledge_ids(db);
    assert!(ids.len() >= 4, "fixture lost its memories");
    for id in &ids {
        let node = snap
            .nodes
            .iter()
            .find(|node| &node.legacy_id == id)
            .unwrap_or_else(|| panic!("dropped memory {id}"));
        let source = node
            .source
            .as_ref()
            .unwrap_or_else(|| panic!("{id} dropped its v3 source"));
        assert_eq!(source.system, "fixture", "{id}");
        assert!(source.project.is_empty(), "{id}");
        assert!(source.id.is_empty(), "{id}");
        assert_eq!(node.source_updated_at_ms, Some(expected_ms), "{id}");
        let raw = node
            .legacy
            .iter()
            .find(|(key, _)| key == "knowledge_nodes.source")
            .map(|(_, value)| value.as_str());
        assert_eq!(raw, Some("fixture"), "{id} legacy source column dropped");
    }
    let receipts: Vec<_> = snap
        .nodes
        .iter()
        .filter(|node| node.node_type == "walk_receipt")
        .collect();
    assert!(
        !receipts.is_empty(),
        "fixture walk receipt missing; cannot check the None path"
    );
    for node in receipts {
        assert!(
            node.source.is_none(),
            "walk receipt invented a source: {:?}",
            node.source
        );
        assert!(node.source_updated_at_ms.is_none());
    }
}

#[test]
fn v311_upgrade_keeps_source_and_source_updated_at() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let status = auto_upgrade::upgrade_if_needed(&db).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected an installed strata log");
    };
    assert_v3_source_kept(&db, &log_dir);
}

#[test]
fn happy_path_on_real_v3_fixture_keeps_the_source_hash() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    // A crashed previous attempt must be discarded, not resumed.
    let staging = staging_directory(dir.path());
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("leftover.seg"), b"not a log").unwrap();

    let status = auto_upgrade::upgrade_if_needed(&db).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected an installed strata log");
    };
    assert_eq!(before, sha256_file(&db), "upgrade modified the v3 file");
    assert!(!staging.exists(), "staging survived a successful swap");
    assert!(log_dir.join("strata.key").is_file() || dir_has_seg(&log_dir));
    let backups: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains(".v3-backup-"))
        .map(|entry| entry.path())
        .collect();
    assert_eq!(backups.len(), 1, "{backups:?}");
    assert_eq!(
        before,
        sha256_file(&backups[0]),
        "backup is not a byte copy"
    );
    assert_fixture_landed(&db, &log_dir);

    let again = auto_upgrade::upgrade_if_needed(&db).unwrap();
    assert!(matches!(again, UpgradeStatus::StrataReady { .. }));
    assert_eq!(before, sha256_file(&db));
    let backups_after = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains(".v3-backup-"))
        .count();
    assert_eq!(backups_after, 1, "relaunch took another backup");
}

#[test]
fn import_failure_leaves_the_v3_hash_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    drop(file);
    let before = sha256_file(&db);
    let err = auto_upgrade::upgrade_if_needed(&db).unwrap_err();
    let text = err.to_string();
    assert_failure_message(&text, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(text.contains("import failed"), "{text}");
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}

#[test]
fn verify_failure_leaves_the_v3_hash_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let err = auto_upgrade::upgrade_with(
        &db,
        UpgradeOptions {
            after_import: Some(Box::new(|staging| {
                let seg = fs::read_dir(staging)?
                    .flatten()
                    .map(|entry| entry.path())
                    .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::NotFound, "no segment to tamper")
                    })?;
                let mut bytes = fs::read(&seg)?;
                let index = bytes.len() / 2;
                bytes[index] ^= 0xff;
                fs::write(&seg, bytes)?;
                Ok(())
            })),
        },
    )
    .unwrap_err();
    let text = err.to_string();
    assert_failure_message(&text, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(
        text.contains("strata-verify failed"),
        "verify injection did not fail closed: {text}"
    );
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}

struct Running {
    child: std::process::Child,
    stdout: Arc<Mutex<String>>,
    stderr: Arc<Mutex<String>>,
}

fn spawn_mcp(data_dir: &Path) -> Running {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
        .arg("--data-dir")
        .arg(data_dir)
        .env("HOME", data_dir)
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vestige-mcp");
    let stdout = Arc::new(Mutex::new(String::new()));
    let stderr = Arc::new(Mutex::new(String::new()));
    let mut out_pipe = child.stdout.take().expect("stdout");
    let out_buf = Arc::clone(&stdout);
    std::thread::spawn(move || {
        let mut tmp = String::new();
        let _ = out_pipe.read_to_string(&mut tmp);
        *out_buf.lock().unwrap() = tmp;
    });
    let mut err_pipe = child.stderr.take().expect("stderr");
    let err_buf = Arc::clone(&stderr);
    std::thread::spawn(move || {
        let mut tmp = String::new();
        let _ = err_pipe.read_to_string(&mut tmp);
        *err_buf.lock().unwrap() = tmp;
    });
    Running {
        child,
        stdout,
        stderr,
    }
}

fn dir_has_seg(dir: &Path) -> bool {
    fs::read_dir(dir).ok().is_some_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("seg"))
    })
}

#[test]
fn failed_import_process_exits_on_stderr_and_leaves_stdout_empty() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    drop(file);
    let before = sha256_file(&db);
    let mut running = spawn_mcp(dir.path());
    let status = running
        .child
        .wait_timeout_ext(Duration::from_secs(60))
        .expect("corrupt import should exit");
    assert_ne!(status.code(), Some(0), "corrupt import exited 0");
    std::thread::sleep(Duration::from_millis(50));
    let stdout = running.stdout.lock().unwrap().clone();
    let stderr = running.stderr.lock().unwrap().clone();
    assert!(
        stdout.trim().is_empty(),
        "upgrade wrote to stdout (MCP): {stdout}"
    );
    assert_failure_message(&stderr, &dir.path().join(UPGRADE_LOG_NAME));
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
}

trait WaitTimeout {
    fn wait_timeout_ext(&mut self, limit: Duration) -> Option<std::process::ExitStatus>;
}

impl WaitTimeout for std::process::Child {
    fn wait_timeout_ext(&mut self, limit: Duration) -> Option<std::process::ExitStatus> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.try_wait().unwrap() {
                return Some(status);
            }
            if started.elapsed() > limit {
                let _ = self.kill();
                let _ = self.wait();
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

fn reset_attempt(data: &Path, db: &Path) {
    let _ = fs::remove_dir_all(data.join(LOG_DIR_NAME));
    let _ = fs::remove_dir_all(staging_directory(data));
    for entry in fs::read_dir(data).unwrap().flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.contains(".v3-backup-") || name == UPGRADE_LOG_NAME || name == "receipt-signing.key"
        {
            let _ = fs::remove_file(entry.path());
        }
    }
    if db.exists() {
        let _ = fs::remove_file(db);
    }
    fs::copy(fixture_db(), db).unwrap();
}

fn inflate_nodes(db: &Path, extra: usize) {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare("SELECT * FROM knowledge_nodes LIMIT 1")
        .unwrap();
    let cols: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|c| (*c).to_string())
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
        values[id_idx] = rusqlite::types::Value::Text(format!("aaaaaaaa-bbbb-4ccc-8ddd-{n:012x}"));
        tx.execute(&sql, rusqlite::params_from_iter(values.iter()))
            .unwrap();
    }
    tx.commit().unwrap();
}

#[test]
fn sigkill_mid_upgrade_then_relaunch_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let db = data.join("vestige.db");
    let mut extra = 8_000usize;
    let killed = loop {
        if extra > 64_000 {
            panic!("could not SIGKILL mid-upgrade; import finished before a segment appeared");
        }
        reset_attempt(&data, &db);
        inflate_nodes(&db, extra);
        let before = sha256_file(&db);
        let mut running = spawn_mcp(&data);
        let started = Instant::now();
        let mut saw_segment = false;
        while started.elapsed() < Duration::from_secs(180) {
            if dir_has_seg(&staging_directory(&data)) {
                saw_segment = true;
                break;
            }
            if running.child.try_wait().unwrap().is_some() {
                break;
            }
            if dir_has_seg(&data.join(LOG_DIR_NAME)) && !staging_directory(&data).exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = running.child.kill();
        let _ = running.child.wait();
        assert_eq!(
            before,
            sha256_file(&db),
            "SIGKILL path modified the v3 file"
        );
        if saw_segment && !dir_has_seg(&data.join(LOG_DIR_NAME)) {
            break before;
        }
        extra *= 2;
    };

    let before = killed;
    let mut again = spawn_mcp(&data);
    let started = Instant::now();
    let log_dir = data.join(LOG_DIR_NAME);
    loop {
        if dir_has_seg(&log_dir) && !staging_directory(&data).exists() {
            break;
        }
        if let Some(status) = again.child.try_wait().unwrap() {
            std::thread::sleep(Duration::from_millis(50));
            panic!(
                "relaunch exited {status} before the strata log was installed. stderr: {}",
                again.stderr.lock().unwrap()
            );
        }
        if started.elapsed() > Duration::from_secs(180) {
            let _ = again.child.kill();
            let _ = again.child.wait();
            panic!(
                "relaunch did not finish the upgrade. stderr: {}",
                again.stderr.lock().unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        again.child.try_wait().unwrap().is_none(),
        "relaunch exited after installing the log"
    );
    assert_eq!(before, sha256_file(&db), "relaunch modified the v3 file");
    // The booted server holds `log/strata.lock`. Stop it, then read the log.
    let _ = again.child.kill();
    let _ = again.child.wait();
    std::thread::sleep(Duration::from_millis(50));
    assert_fixture_landed(&db, &log_dir);
    let ids = knowledge_ids(&db);
    let snap = snapshot(&log_dir);
    let imported = snap
        .nodes
        .iter()
        .filter(|node| ids.iter().any(|id| id == &node.legacy_id))
        .count();
    assert_eq!(imported, ids.len(), "relaunch dropped or duplicated nodes");
    let stdout = again.stdout.lock().unwrap().clone();
    assert!(
        stdout.trim().is_empty(),
        "upgrade wrote to stdout (MCP): {stdout}"
    );
}

fn run_cli_stats(data_dir: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args([
            "--data-dir",
            data_dir.to_str().expect("temp path is utf-8"),
            "stats",
        ])
        .output()
        .expect("spawn vestige")
}

fn assert_progress_stayed_on_stderr(stdout: &str) {
    assert!(
        !stdout.contains("vestige:") && !stdout.contains("upgrading to strata"),
        "upgrade progress leaked to stdout: {stdout}"
    );
}

fn assert_no_sqlite_sidecars(db: &Path) {
    for suffix in ["-wal", "-shm"] {
        let mut name = db.as_os_str().to_os_string();
        name.push(suffix);
        let path = PathBuf::from(name);
        assert!(!path.exists(), "sqlite sidecar created: {}", path.display());
    }
}

#[test]
fn cli_first_upgrades_fixture_and_preserves_v3_sha() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let output = run_cli_stats(dir.path());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        before,
        sha256_file(&db),
        "cli upgrade changed the v3 file\n{stderr}"
    );
    assert_no_sqlite_sidecars(&db);
    assert_eq!(
        output.status.code(),
        Some(0),
        "vestige stats exited {:?}\nstdout: {stdout}\nstderr: {stderr}",
        output.status.code()
    );
    assert!(
        stdout.contains("Total Memories"),
        "stats did not read the switched log\n{stdout}"
    );
    assert_progress_stayed_on_stderr(&stdout);
    let record = format!("{stdout}{stderr}");
    let lowered = record.to_lowercase();
    assert!(
        lowered.contains("import")
            && lowered.contains("verif")
            && !lowered.contains("verify failed"),
        "stats did not record import and verify\n{record}"
    );
    assert!(
        stderr.contains("strata log ready"),
        "cli did not report the upgrade\n{stderr}"
    );
    assert!(!stderr.contains("upgrade failed"), "{stderr}");
    assert!(
        !stderr.contains("cannot be opened by 4.0"),
        "cli hit the v3 refusal instead of upgrading\n{stderr}"
    );
    let log_dir = dir.path().join(LOG_DIR_NAME);
    assert!(!staging_directory(dir.path()).exists());
    assert!(!dir.path().join(".strata-upgrade-verify").exists());
    assert_fixture_landed(&db, &log_dir);
    assert_memory_count(&db, &log_dir);
}

#[test]
fn cli_first_corrupt_import_leaves_bytes_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    drop(file);
    let before = sha256_file(&db);
    let output = run_cli_stats(dir.path());
    assert!(
        !output.status.success(),
        "corrupt cli import exited 0: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(before, sha256_file(&db));
    assert_no_sqlite_sidecars(&db);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_progress_stayed_on_stderr(&stdout);
    assert!(stdout.trim().is_empty(), "cli wrote stdout: {stdout}");
    assert_failure_message(&stderr, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(!staging_directory(dir.path()).exists());
    assert!(
        !dir_has_seg(&dir.path().join(LOG_DIR_NAME)),
        "failed cli upgrade installed a log"
    );
}

#[test]
fn two_cli_processes_upgrade_the_fixture_once() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let dir_s = dir.path().to_str().expect("temp path is utf-8");
    let spawn = || {
        Command::new(env!("CARGO_BIN_EXE_vestige"))
            .args(["--data-dir", dir_s, "stats"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige")
    };
    let mut first = spawn();
    let mut second = spawn();
    let first_status = first
        .wait_timeout_ext(Duration::from_secs(120))
        .expect("first cli upgrade timed out");
    let second_status = second
        .wait_timeout_ext(Duration::from_secs(120))
        .expect("second cli upgrade timed out");
    assert!(
        first_status.success(),
        "first cli stats failed: {first_status}"
    );
    assert!(
        second_status.success(),
        "second cli stats failed: {second_status}"
    );
    assert_eq!(
        before,
        sha256_file(&db),
        "concurrent cli upgrade wrote the v3 file"
    );
    assert_no_sqlite_sidecars(&db);
    let log_dir = dir.path().join(LOG_DIR_NAME);
    assert!(!staging_directory(dir.path()).exists());
    assert_fixture_landed(&db, &log_dir);
    assert_memory_count(&db, &log_dir);
}

fn assert_memory_count(db: &Path, log_dir: &Path) {
    let ids = knowledge_ids(db);
    let snap = snapshot(log_dir);
    let imported = snap
        .nodes
        .iter()
        .filter(|node| ids.iter().any(|id| id == &node.legacy_id))
        .count();
    assert_eq!(
        imported,
        ids.len(),
        "memory count diverged (log nodes {}, memories {})",
        snap.nodes.len(),
        ids.len()
    );
}
