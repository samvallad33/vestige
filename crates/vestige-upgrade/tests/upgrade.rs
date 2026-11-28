//! Real v3 fixture upgrades: happy path, injected import/verify failures, and
//! a SIGKILL mid-upgrade followed by a relaunch.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use vestige_upgrade::{
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
    // #318: an inferred v3 link is history, not recorded causal proof, so it
    // migrates as legacy_inferred rather than a Strata edge kind.
    assert_eq!(causal.link_type, strata_migrate::LEGACY_INFERRED_KIND);
    let semantic = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "semantic")
        .expect("semantic link dropped");
    assert!(semantic.legacy_inferred);
    assert_eq!(semantic.link_type, strata_migrate::LEGACY_INFERRED_KIND);
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
    let status = vestige_upgrade::upgrade_if_needed(&db).unwrap();
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

    let status = vestige_upgrade::upgrade_if_needed(&db).unwrap();
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

    let again = vestige_upgrade::upgrade_if_needed(&db).unwrap();
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
    let err = vestige_upgrade::upgrade_if_needed(&db).unwrap_err();
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
    let err = vestige_upgrade::upgrade_with(
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

fn spawn_upgrade(data_dir: &Path) -> Running {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .arg("--data-dir")
        .arg(data_dir)
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vestige-upgrade");
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
    let mut running = spawn_upgrade(dir.path());
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
        .map(str::to_string)
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
        let mut running = spawn_upgrade(&data);
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
    let mut again = spawn_upgrade(&data);
    let started = Instant::now();
    let log_dir = data.join(LOG_DIR_NAME);
    let status = loop {
        if let Some(status) = again.child.try_wait().unwrap() {
            break status;
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
    };
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        status.success(),
        "relaunch exited {status}. stderr: {}",
        again.stderr.lock().unwrap()
    );
    assert!(
        dir_has_seg(&log_dir) && !staging_directory(&data).exists(),
        "relaunch did not publish the strata log"
    );
    assert_eq!(before, sha256_file(&db), "relaunch modified the v3 file");
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

fn run_upgrade_bin(data_dir: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .arg("--data-dir")
        .arg(data_dir)
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .expect("spawn vestige-upgrade")
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
fn upgrade_bin_imports_fixture_and_preserves_v3_sha() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let output = run_upgrade_bin(dir.path());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        before,
        sha256_file(&db),
        "upgrade changed the v3 file\n{stderr}"
    );
    assert_no_sqlite_sidecars(&db);
    assert_eq!(
        output.status.code(),
        Some(0),
        "vestige-upgrade exited {:?}\nstdout: {stdout}\nstderr: {stderr}",
        output.status.code()
    );
    assert_progress_stayed_on_stderr(&stdout);
    let record = format!("{stdout}{stderr}");
    let lowered = record.to_lowercase();
    assert!(
        lowered.contains("import")
            && lowered.contains("verif")
            && !lowered.contains("verify failed"),
        "upgrade did not record import and verify\n{record}"
    );
    assert!(
        stderr.contains("strata log ready"),
        "upgrade did not report the log\n{stderr}"
    );
    assert!(!stderr.contains("upgrade failed"), "{stderr}");
    assert!(
        !stderr.contains("cannot be opened by 4.0"),
        "upgrade hit the v3 refusal instead of importing\n{stderr}"
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
    let output = run_upgrade_bin(dir.path());
    assert!(
        !output.status.success(),
        "corrupt import exited 0: {}",
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
        Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
            .args(["--data-dir", dir_s])
            .env_remove("VESTIGE_DATA_DIR")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige-upgrade")
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
        "first upgrade failed: {first_status}"
    );
    assert!(
        second_status.success(),
        "second upgrade failed: {second_status}"
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

/// v3 `MIGRATION_V3_UP` intentions table plus `MIGRATION_V37_UP` `scope`.
/// The committed fixture predates intentions, so tests plant them.
const INTENTIONS_DDL: &str = r#"
CREATE TABLE intentions (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_data TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    deadline TEXT,
    fulfilled_at TEXT,
    reminder_count INTEGER DEFAULT 0,
    last_reminded_at TEXT,
    notes TEXT,
    tags TEXT DEFAULT '[]',
    related_memories TEXT DEFAULT '[]',
    snoozed_until TEXT,
    source_type TEXT NOT NULL DEFAULT 'api',
    source_data TEXT
);
ALTER TABLE intentions ADD COLUMN scope TEXT;
"#;

/// One row per v3 status, plus the columns each one exercises.
fn plant_intentions(db: &Path) {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.execute_batch(INTENTIONS_DDL).unwrap();
    conn.execute_batch(
        r#"
        INSERT INTO intentions (id, content, trigger_type, trigger_data, priority, status,
            created_at, deadline, tags, related_memories, source_type, scope)
        VALUES ('6fc170c5-67b5-432e-acee-eed12881067b', 'Post the launch thread', 'time',
            '{"type":"time","at":"2026-10-01T09:00:00Z"}', 4, 'active',
            '2026-01-27T04:45:43.353514+00:00', '2026-10-01T09:00:00+00:00',
            '["launch","hn"]', '["11111111-1111-4111-8111-111111111111"]', 'mcp', 'vestige');
        INSERT INTO intentions (id, content, trigger_type, trigger_data, priority, status,
            created_at, fulfilled_at, reminder_count, last_reminded_at, source_type)
        VALUES ('94169f00-004f-4a6e-b7ed-25d99d7dfd21', 'Check PR 654', 'context',
            '{"type":"context","topic":"pr"}', 3, 'fulfilled',
            '2026-02-04T01:15:46.898393+00:00', '2026-02-04T01:18:28.837461+00:00', 2,
            '2026-02-04T01:17:00+00:00', 'mcp');
        INSERT INTO intentions (id, content, trigger_type, trigger_data, priority, status,
            created_at, notes, source_type, source_data, scope)
        VALUES ('ad4f0e90-95e8-4f37-bca3-6f95ecb424e7', 'Buy concert tickets', 'event',
            '{"type":"event","condition":"tickets on sale"}', 1, 'cancelled',
            '2026-01-27T07:59:57.263471+00:00', 'plans changed', 'nlp',
            '{"text":"remind me to buy tickets"}', 'user');
        INSERT INTO intentions (id, content, trigger_type, trigger_data, status,
            created_at, snoozed_until)
        VALUES ('f0000000-0000-4000-8000-000000000004', 'Snoozed legacy row', 'manual', '{}',
            'snoozed', '2026-02-10 23:41:40', '2026-10-01T09:00:00+00:00');
        "#,
    )
    .unwrap();
}

/// What the Strata store must hold for [`plant_intentions`], in id order.
fn planted_intentions() -> Vec<strata_store::IntentionRecord> {
    let base = strata_store::IntentionRecord {
        id: String::new(),
        content: String::new(),
        trigger_type: String::new(),
        trigger_data: String::new(),
        priority: 2,
        status: "active".into(),
        created_at_ms: 0,
        deadline_ms: None,
        fulfilled_at_ms: None,
        reminder_count: 0,
        last_reminded_at_ms: None,
        notes: None,
        tags: Vec::new(),
        related_memories: Vec::new(),
        snoozed_until_ms: None,
        source_type: "api".into(),
        source_data: None,
        scope: None,
    };
    vec![
        strata_store::IntentionRecord {
            id: "6fc170c5-67b5-432e-acee-eed12881067b".into(),
            content: "Post the launch thread".into(),
            trigger_type: "time".into(),
            trigger_data: r#"{"type":"time","at":"2026-10-01T09:00:00Z"}"#.into(),
            priority: 4,
            created_at_ms: 1_769_489_143_353,
            deadline_ms: Some(1_790_845_200_000),
            tags: vec!["launch".into(), "hn".into()],
            related_memories: vec!["11111111-1111-4111-8111-111111111111".into()],
            source_type: "mcp".into(),
            scope: Some("vestige".into()),
            ..base.clone()
        },
        strata_store::IntentionRecord {
            id: "94169f00-004f-4a6e-b7ed-25d99d7dfd21".into(),
            content: "Check PR 654".into(),
            trigger_type: "context".into(),
            trigger_data: r#"{"type":"context","topic":"pr"}"#.into(),
            priority: 3,
            status: "fulfilled".into(),
            created_at_ms: 1_770_167_746_898,
            fulfilled_at_ms: Some(1_770_167_908_837),
            reminder_count: 2,
            last_reminded_at_ms: Some(1_770_167_820_000),
            source_type: "mcp".into(),
            ..base.clone()
        },
        strata_store::IntentionRecord {
            id: "ad4f0e90-95e8-4f37-bca3-6f95ecb424e7".into(),
            content: "Buy concert tickets".into(),
            trigger_type: "event".into(),
            trigger_data: r#"{"type":"event","condition":"tickets on sale"}"#.into(),
            priority: 1,
            status: "cancelled".into(),
            created_at_ms: 1_769_500_797_263,
            notes: Some("plans changed".into()),
            source_type: "nlp".into(),
            source_data: Some(r#"{"text":"remind me to buy tickets"}"#.into()),
            scope: Some("user".into()),
            ..base.clone()
        },
        strata_store::IntentionRecord {
            id: "f0000000-0000-4000-8000-000000000004".into(),
            content: "Snoozed legacy row".into(),
            trigger_type: "manual".into(),
            trigger_data: "{}".into(),
            status: "snoozed".into(),
            created_at_ms: 1_770_766_900_000,
            snoozed_until_ms: Some(1_790_845_200_000),
            ..base
        },
    ]
}

fn intention_receipts(store: &strata_store::StrataStore) -> Vec<String> {
    store
        .prove_effects()
        .unwrap()
        .into_iter()
        .filter(|proof| proof.action == strata_store::EffectAction::Intention)
        .map(|proof| proof.node_id)
        .collect()
}

#[test]
fn v3_intentions_survive_the_upgrade_a_reopen_and_a_relaunch() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    plant_intentions(&db);
    let before = sha256_file(&db);

    let status = vestige_upgrade::upgrade_if_needed(&db).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected an installed strata log");
    };
    assert_eq!(before, sha256_file(&db), "upgrade modified the v3 file");
    assert!(!staging_directory(dir.path()).exists());
    let expected = planted_intentions();
    let mut ids: Vec<String> = expected.iter().map(|row| row.id.clone()).collect();
    ids.sort();

    // The server's open: the same data dir, the default policy.
    let store = strata_store::StrataStore::open(dir.path()).unwrap();
    assert_eq!(store.intentions(), expected);
    let mut proved = intention_receipts(&store);
    proved.sort();
    assert_eq!(proved, ids, "every intention needs its own receipt");
    let digest = store.state_digest();
    let frames = store.log().head().frames_total;
    drop(store);

    let logged = fs::read_to_string(dir.path().join(UPGRADE_LOG_NAME)).unwrap();
    assert!(
        logged.contains("4 intentions imported"),
        "upgrade.log does not report the carry-over\n{logged}"
    );

    // Relaunch: the installed log is complete, so nothing is admitted again.
    let again = vestige_upgrade::upgrade_if_needed(&db).unwrap();
    assert!(matches!(again, UpgradeStatus::StrataReady { .. }));
    let reopened = strata_store::StrataStore::open(dir.path()).unwrap();
    assert_eq!(reopened.intentions(), expected);
    assert_eq!(reopened.state_digest(), digest);
    assert_eq!(reopened.log().head().frames_total, frames);
    assert_eq!(intention_receipts(&reopened).len(), expected.len());
    drop(reopened);

    // The migration itself still verifies with store writes after its receipt.
    assert_fixture_landed(&db, &log_dir);
}

#[test]
fn a_failed_intention_carry_installs_no_log() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    plant_intentions(&db);
    // The store refuses an empty intention id, so the carry-over fails.
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO intentions (id, content, trigger_type, trigger_data, created_at)
             VALUES ('', 'no id', 'manual', '{}', '2026-02-10T00:00:00+00:00')",
            [],
        )
        .unwrap();
    let before = sha256_file(&db);
    let err = vestige_upgrade::upgrade_if_needed(&db).unwrap_err();
    let text = err.to_string();
    assert_failure_message(&text, &dir.path().join(UPGRADE_LOG_NAME));
    assert!(text.contains("intention carry-over failed"), "{text}");
    assert_eq!(before, sha256_file(&db));
    assert!(!staging_directory(dir.path()).exists());
    assert!(
        !dir.path().join(LOG_DIR_NAME).exists(),
        "a failed carry-over left an installed log the next launch would skip"
    );
}

/// v3 code anchors ride the same staged carry-over as intentions: each live
/// memory keeps its anchors, admitted through the gate, after a reopen.
#[test]
fn v3_code_anchors_survive_the_upgrade() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let node = "11111111-1111-4111-8111-111111111111";
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS code_memory_anchors (
                id TEXT PRIMARY KEY, node_id TEXT NOT NULL, file_path TEXT NOT NULL,
                symbol TEXT, symbol_kind TEXT, start_line INTEGER, end_line INTEGER,
                span_lines INTEGER, content_hash TEXT, captured_at TEXT NOT NULL,
                last_verified_at TEXT, last_status TEXT)",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO code_memory_anchors VALUES
             ('anchor-a', ?1, 'src/state.rs', 'load_config', 'fn', 3, 6, 4,
              'v2:0123456789abcdef', '2026-03-01 09:15:00', NULL, NULL)",
            [node],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO code_memory_anchors VALUES
             ('anchor-b', ?1, 'README.md', NULL, NULL, NULL, NULL, NULL, NULL,
              '2026-03-01T09:15:00+00:00', '2026-03-02T09:15:00+00:00', 'verified')",
            [node],
        )
        .unwrap();
    }
    let before = sha256_file(&db);
    let status = vestige_upgrade::upgrade_if_needed(&db).unwrap();
    assert!(matches!(status, UpgradeStatus::StrataReady { .. }));
    assert_eq!(before, sha256_file(&db), "upgrade modified the v3 file");

    let store = strata_store::StrataStore::open(dir.path()).unwrap();
    let anchors = store.anchors_for(node);
    assert_eq!(anchors.len(), 2, "{anchors:?}");
    let symbol = anchors.iter().find(|a| a.id == "anchor-a").unwrap();
    assert_eq!(symbol.symbol.as_deref(), Some("load_config"));
    assert_eq!(
        (symbol.start_line, symbol.end_line, symbol.span_lines),
        (Some(3), Some(6), Some(4))
    );
    assert_eq!(symbol.content_hash.as_deref(), Some("v2:0123456789abcdef"));
    let path = anchors.iter().find(|a| a.id == "anchor-b").unwrap();
    assert_eq!(path.last_status.as_deref(), Some("verified"));
    assert!(path.last_verified_at_ms.is_some());
    let logged = fs::read_to_string(dir.path().join(UPGRADE_LOG_NAME)).unwrap();
    assert!(logged.contains("carried 2 code anchors"), "{logged}");
}

/// A `vestige.db` that is a symlink resolves to the real file: the `-wal`
/// beside the real file is imported and backed up with it.
#[cfg(unix)]
#[test]
fn symlinked_db_imports_and_backs_up_the_wal_beside_the_real_file() {
    let real_dir = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();
    let real = plant(real_dir.path());
    let conn = rusqlite::Connection::open(&real).unwrap();
    conn.pragma_update(None, "journal_mode", "WAL").unwrap();
    conn.execute(
        "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, last_accessed, tags)
         VALUES ('55555555-5555-4555-8555-555555555555', 'WAL_ONLY_ROW_NOT_IN_MAIN', 'fact',
                 '2026-03-03T00:00:00+00:00', '2026-03-03T00:00:00+00:00',
                 '2026-03-03T00:00:00+00:00', '[]')",
        [],
    )
    .unwrap();
    let wal = real_dir.path().join("vestige.db-wal");
    assert!(
        fs::metadata(&wal).unwrap().len() > 0,
        "row must sit in the wal"
    );

    let link = data_dir.path().join("vestige.db");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let status = vestige_upgrade::upgrade_if_needed(&link).unwrap();
    let UpgradeStatus::StrataReady { log_dir } = status else {
        panic!("expected a ready log, got {status:?}");
    };

    let snap = snapshot(&log_dir);
    assert!(
        snap.nodes
            .iter()
            .any(|node| node.content == "WAL_ONLY_ROW_NOT_IN_MAIN"),
        "the row that lives only in the real file's wal was dropped"
    );
    let has_backup = |name: &str| {
        fs::read_dir(real_dir.path()).unwrap().flatten().any(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(name) && n.contains(".v3-backup-") && !n.ends_with(".partial")
        })
    };
    assert!(has_backup("vestige.db-wal"), "backup lacks the wal");
    assert!(has_backup("vestige.db.v3-backup-"), "backup lacks the db");
    drop(conn);
}

/// Take the same exclusive lock a serving `vestige-mcp` holds.
fn hold_serve_lock(data_dir: &Path) -> fs::File {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(data_dir.join(".serve.lock"))
        .unwrap();
    file.lock().unwrap();
    file
}

#[test]
fn a_held_serve_lock_ends_in_a_bounded_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    let _held = hold_serve_lock(dir.path());
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .arg("--data-dir")
        .arg(dir.path())
        .env_remove("VESTIGE_DATA_DIR")
        .env("VESTIGE_ATTACH_WAIT_SECS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vestige-upgrade");
    let mut err_pipe = child.stderr.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = err_pipe.read_to_string(&mut text);
        text
    });
    let status = child
        .wait_timeout_ext(Duration::from_secs(30))
        .expect("a held serve lock must not block the upgrade forever");
    assert_eq!(status.code(), Some(1));
    let stderr = reader.join().unwrap();
    assert!(stderr.contains(".serve.lock"), "{stderr}");
    assert_eq!(before, sha256_file(&db));
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}

#[test]
fn a_held_serve_lock_over_a_published_log_needs_no_upgrade() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let status = vestige_upgrade::upgrade_if_needed(&db).unwrap();
    assert!(matches!(status, UpgradeStatus::StrataReady { .. }));
    let _held = hold_serve_lock(dir.path());
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .arg("--data-dir")
        .arg(dir.path())
        .env_remove("VESTIGE_DATA_DIR")
        .env("VESTIGE_ATTACH_WAIT_SECS", "600")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn vestige-upgrade");
    let status = child
        .wait_timeout_ext(Duration::from_secs(30))
        .expect("a published log must not wait on the serve lock");
    assert_eq!(status.code(), Some(0));
}

#[test]
fn a_closed_stderr_does_not_abort_the_upgrade() {
    let dir = tempfile::tempdir().unwrap();
    let db = plant(dir.path());
    let before = sha256_file(&db);
    // Hold the lock so the process is still waiting when its reader goes.
    let held = hold_serve_lock(dir.path());
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .arg("--data-dir")
        .arg(dir.path())
        .env_remove("VESTIGE_DATA_DIR")
        .env("VESTIGE_ATTACH_WAIT_SECS", "60")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vestige-upgrade");
    drop(child.stderr.take());
    std::thread::sleep(Duration::from_millis(500));
    drop(held);
    let status = child
        .wait_timeout_ext(Duration::from_secs(120))
        .expect("upgrade should finish");
    assert_eq!(status.code(), Some(0), "upgrade died with {status:?}");
    assert!(dir.path().join(LOG_DIR_NAME).exists());
    assert_eq!(before, sha256_file(&db));
}

// APFS refuses file names that are not valid UTF-8, so this runs where the
// filesystem can hold one.
#[cfg(all(unix, not(target_os = "macos")))]
#[test]
fn a_non_utf8_data_dir_is_accepted() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let parent = tempfile::tempdir().unwrap();
    let data = parent.path().join(OsStr::from_bytes(b"data-\xff\xfe"));
    fs::create_dir(&data).unwrap();
    let db = plant(&data);
    let before = sha256_file(&db);
    let out = run_upgrade_bin(&data);
    assert!(
        out.status.success(),
        "{:?} {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(data.join(LOG_DIR_NAME).exists());
    assert_eq!(before, sha256_file(&db));
}

/// A `vestige.db` that is empty, not plain SQLite (for example an encrypted
/// store) or unreadable is not "no v3 store": the upgrade refuses, installs no
/// log, and leaves the file as it was.
#[test]
fn unrecognised_vestige_db_is_refused_and_installs_no_log() {
    let encrypted_like: Vec<u8> = (0u32..4096)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8 | 0x80)
        .collect();
    let cases: [(&str, Vec<u8>); 3] = [
        ("empty", Vec::new()),
        ("short", b"SQLite".to_vec()),
        ("not-plain-sqlite", encrypted_like),
    ];
    for (label, bytes) in cases {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("vestige.db");
        fs::write(&db, &bytes).unwrap();

        let err = vestige_upgrade::upgrade_if_needed(&db)
            .expect_err(&format!("{label}: must not count as no v3 store"));
        let text = err.to_string();
        assert!(text.contains("vestige.db"), "{label}: {text}");
        assert_eq!(fs::read(&db).unwrap(), bytes, "{label}: file changed");
        assert!(
            !dir.path().join(LOG_DIR_NAME).exists(),
            "{label}: a log was created"
        );

        let output = run_upgrade_bin(dir.path());
        assert!(
            !output.status.success(),
            "{label}: upgrade binary exited 0: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(&db).unwrap(), bytes, "{label}: file changed");
        assert!(!dir.path().join(LOG_DIR_NAME).exists(), "{label}: log");
    }
}

#[test]
fn unreadable_vestige_db_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    // A directory opens but cannot be read, like a file the process has no
    // access to, and works whatever user runs the tests.
    let db = dir.path().join("vestige.db");
    fs::create_dir(&db).unwrap();
    let err = vestige_upgrade::upgrade_if_needed(&db)
        .expect_err("an unreadable vestige.db must not count as no v3 store");
    assert!(err.to_string().contains("vestige.db"), "{err}");
    assert!(!dir.path().join(LOG_DIR_NAME).exists());
}

/// Without a `vestige.db` there is nothing to refuse.
#[test]
fn absent_vestige_db_still_means_no_v3() {
    let dir = tempfile::tempdir().unwrap();
    let status = vestige_upgrade::upgrade_if_needed(&dir.path().join("vestige.db")).unwrap();
    assert_eq!(status, UpgradeStatus::NoV3);
}
