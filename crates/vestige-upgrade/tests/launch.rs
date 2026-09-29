//! Process-level first launch. These tests spawn the main binaries, which
//! must already be in the same target directory as `vestige-upgrade`
//! (`cargo build -p vestige-mcp --bins`).

use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use vestige_upgrade::{LOG_DIR_NAME, UPGRADE_LOG_NAME, V311_RELEASE, staging_directory};

const UPGRADE_HINT: &str = "install vestige-upgrade from https://github.com/samvallad33/vestige/releases and place it next to this binary or on PATH";

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
    let verified = strata_verify::verify_path(log_dir);
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
    assert_eq!(causal.link_type, "derived_from");
    let semantic = snap
        .edges
        .iter()
        .find(|edge| edge.legacy_link_type == "semantic")
        .expect("semantic link dropped");
    assert!(semantic.legacy_inferred);
    assert_eq!(semantic.link_type, "derived_from");
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

struct Running {
    child: std::process::Child,
    stdout: Arc<Mutex<String>>,
    stderr: Arc<Mutex<String>>,
}

fn target_bin(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_BIN_EXE_vestige-upgrade"))
        .parent()
        .unwrap()
        .join(name);
    assert!(
        path.is_file(),
        "missing {}; build the main binaries with `cargo build -p vestige-mcp --bins`",
        path.display()
    );
    path
}

fn spawn_mcp(data_dir: &Path) -> Running {
    let bin = target_bin("vestige-mcp");
    let mut child = Command::new(&bin)
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
    let bin = target_bin("vestige");
    Command::new(&bin)
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
    let bin = target_bin("vestige");
    let spawn = || {
        Command::new(&bin)
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

fn copy_exe(src: &Path, dst: &Path) {
    fs::copy(src, dst).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(dst).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(dst, perms).unwrap();
    }
}

fn cargo_bin(name: &str) -> PathBuf {
    if name == "vestige-upgrade" {
        PathBuf::from(env!("CARGO_BIN_EXE_vestige-upgrade"))
    } else {
        target_bin(name)
    }
}

fn isolated_bins(dir: &Path, names: &[&str]) -> PathBuf {
    let bin_dir = dir.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    for name in names {
        let src = cargo_bin(name);
        assert!(
            src.is_file(),
            "missing {}; build it before this test",
            src.display()
        );
        copy_exe(&src, &bin_dir.join(name));
    }
    bin_dir
}

/// Real stdio: vestige-mcp finds the sibling importer, upgrades the v3.1.1
/// fixture, finishes initialize, and `vestige strata-verify` accepts the log.
#[test]
fn stdio_upgrade_keeps_source_and_passes_strata_verify() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let db = plant(&data);
    let before = fs::read(&db).unwrap();
    let bin_dir = isolated_bins(dir.path(), &["vestige-mcp", "vestige-upgrade"]);
    let mcp = bin_dir.join("vestige-mcp");

    let mut child = Command::new(&mcp)
        .arg("--data-dir")
        .arg(&data)
        .env("HOME", &data)
        .env("PATH", "/usr/bin:/bin")
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn copied vestige-mcp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let stderr = Arc::new(Mutex::new(String::new()));
    let err_buf = Arc::clone(&stderr);
    std::thread::spawn(move || {
        let mut tmp = String::new();
        let _ = std::io::Read::read_to_string(&mut err_pipe, &mut tmp);
        *err_buf.lock().unwrap() = tmp;
    });

    let init = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "upgrade-stdio", "version": "1" }
        }
    });
    writeln!(stdin, "{init}").unwrap();
    stdin.flush().unwrap();
    writeln!(
        stdin,
        "{}",
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    let mut reader = std::io::BufReader::new(stdout);
    let response = loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "server closed stdout before initialize");
        let value: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        if value.get("id").is_some() {
            break value;
        }
    };
    assert!(
        response.get("error").is_none(),
        "{response} stderr {}",
        stderr.lock().unwrap()
    );

    let _ = child.kill();
    let _ = child.wait();
    std::thread::sleep(Duration::from_millis(50));

    assert_eq!(
        before,
        fs::read(&db).unwrap(),
        "stdio upgrade changed vestige.db"
    );
    let log_dir = data.join(LOG_DIR_NAME);
    assert_v3_source_kept(&db, &log_dir);

    let verify = Command::new(target_bin("vestige"))
        .arg("strata-verify")
        .arg(&log_dir)
        .output()
        .expect("spawn vestige strata-verify");
    let verify_out = String::from_utf8_lossy(&verify.stdout);
    let verify_err = String::from_utf8_lossy(&verify.stderr);
    assert!(
        verify.status.success(),
        "strata-verify failed\n{verify_out}\n{verify_err}"
    );
    assert!(
        verify_out.contains("OK"),
        "strata-verify did not print OK\n{verify_out}"
    );
}

/// No importer next to the binary and none on PATH: one-line hint, zero writes.
#[test]
fn missing_upgrade_refuses_with_hint_and_leaves_db_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let db = plant(&data);
    let before = fs::read(&db).unwrap();
    let bin_dir = isolated_bins(dir.path(), &["vestige-mcp", "vestige"]);

    let mcp = Command::new(bin_dir.join("vestige-mcp"))
        .arg("--data-dir")
        .arg(&data)
        .env("HOME", &data)
        .env("PATH", "/usr/bin:/bin")
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .expect("spawn vestige-mcp without importer");
    assert!(!mcp.status.success(), "missing importer exited 0");
    let mcp_err = String::from_utf8_lossy(&mcp.stderr);
    assert!(
        mcp_err.contains(UPGRADE_HINT),
        "mcp stderr missing the install hint:\n{mcp_err}"
    );
    assert!(
        mcp_err.contains("was not modified"),
        "mcp stderr did not say the store was untouched:\n{mcp_err}"
    );
    assert_eq!(before, fs::read(&db).unwrap(), "mcp touched vestige.db");
    assert_no_sqlite_sidecars(&db);
    assert!(!data.join(LOG_DIR_NAME).exists());
    assert!(!staging_directory(&data).exists());

    let cli = Command::new(bin_dir.join("vestige"))
        .args([
            "--data-dir",
            data.to_str().expect("temp path is utf-8"),
            "stats",
        ])
        .env("HOME", &data)
        .env("PATH", "/usr/bin:/bin")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .expect("spawn vestige without importer");
    assert!(!cli.status.success(), "missing importer cli exited 0");
    let cli_err = String::from_utf8_lossy(&cli.stderr);
    assert!(
        cli_err.contains(UPGRADE_HINT),
        "cli stderr missing the install hint:\n{cli_err}"
    );
    assert_eq!(
        before,
        fs::read(&db).unwrap(),
        "cli touched vestige.db after the refusal"
    );
    assert_no_sqlite_sidecars(&db);
}

fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_mode(mode);
    fs::set_permissions(path, perms).unwrap();
}

fn mode_of(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

fn euid() -> u32 {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Uid:") {
            return rest.split_whitespace().nth(1).unwrap().parse().unwrap();
        }
    }
    panic!("no Uid in /proc/self/status");
}

fn passwd_ids(name: &str) -> (u32, u32) {
    let text = fs::read_to_string("/etc/passwd").unwrap();
    for line in text.lines() {
        let mut parts = line.split(':');
        if parts.next() == Some(name) {
            let _password = parts.next();
            let uid: u32 = parts.next().unwrap().parse().unwrap();
            let gid: u32 = parts.next().unwrap().parse().unwrap();
            return (uid, gid);
        }
    }
    panic!("passwd has no {name}");
}

/// Root ignores mode 000. Drop to nobody so an open would fail.
fn drop_privs(cmd: &mut Command) {
    if euid() != 0 {
        return;
    }
    let (uid, gid) = passwd_ids("nobody");
    use std::os::unix::process::CommandExt;
    cmd.uid(uid).gid(gid);
}

fn expose_owned(path: &Path, mode: u32) {
    if euid() == 0 {
        set_mode(path, mode);
    }
}

fn finish_within(child: &mut std::process::Child, limit: Duration) -> std::process::ExitStatus {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            return child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn spawn_refusing(bin: &Path, data: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .env("HOME", data)
        .env("PATH", "/usr/bin:/bin")
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("RUST_LOG", "error")
        .env_remove("VESTIGE_DATA_DIR")
        .env_remove("STRATA_MIGRATE_SIGKILL_WINDOW")
        .env_remove("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    drop_privs(&mut cmd);
    let mut child = cmd
        .spawn()
        .unwrap_or_else(|err| panic!("spawn {}: {err}", bin.display()));
    let mut out_pipe = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let stdout = Arc::new(Mutex::new(Vec::new()));
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let out_buf = Arc::clone(&stdout);
    let err_buf = Arc::clone(&stderr);
    let out_thread = std::thread::spawn(move || {
        let _ = std::io::Read::read_to_end(&mut out_pipe, &mut out_buf.lock().unwrap());
    });
    let err_thread = std::thread::spawn(move || {
        let _ = std::io::Read::read_to_end(&mut err_pipe, &mut err_buf.lock().unwrap());
    });
    let status = finish_within(&mut child, Duration::from_secs(30));
    let _ = out_thread.join();
    let _ = err_thread.join();
    std::process::Output {
        status,
        stdout: stdout.lock().unwrap().clone(),
        stderr: stderr.lock().unwrap().clone(),
    }
}

/// chmod 000 and junk bytes are both v3 candidates. With no importer, 4.0
/// refuses. It never opens the file: mode 000 stays 000 and the bytes match.
#[test]
fn unreadable_and_junk_db_are_detected_and_refused() {
    for (label, plant_db) in [
        (
            "unreadable",
            plant_unreadable as fn(&Path) -> (PathBuf, Vec<u8>),
        ),
        ("junk", plant_junk),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        expose_owned(dir.path(), 0o755);
        expose_owned(&data, 0o755);
        let (db, before) = plant_db(&data);
        let bin_dir = isolated_bins(dir.path(), &["vestige-mcp", "vestige"]);
        expose_owned(&bin_dir, 0o755);

        let mcp = spawn_refusing(
            &bin_dir.join("vestige-mcp"),
            &data,
            &["--data-dir", data.to_str().unwrap()],
        );
        assert!(
            !mcp.status.success(),
            "{label} mcp exited 0: {}",
            String::from_utf8_lossy(&mcp.stderr)
        );
        let mcp_err = String::from_utf8_lossy(&mcp.stderr);
        assert!(
            mcp_err.contains(UPGRADE_HINT),
            "{label} mcp did not refuse with the hint:\n{mcp_err}"
        );
        assert!(
            mcp_err.contains("was not modified"),
            "{label} mcp did not say the store was untouched:\n{mcp_err}"
        );
        assert!(
            !mcp_err.contains("cannot be read") && !mcp_err.contains("not a v3 SQLite"),
            "{label} mcp opened the file instead of refusing:\n{mcp_err}"
        );
        assert_untouched(&db, &before, label);
        assert!(
            !data.join(LOG_DIR_NAME).exists(),
            "{label} mcp created a log"
        );
        assert!(
            !staging_directory(&data).exists(),
            "{label} mcp created staging"
        );
        assert!(
            !data.join("upgrade.lock").exists(),
            "{label} mcp took the upgrade lock"
        );

        let cli = spawn_refusing(
            &bin_dir.join("vestige"),
            &data,
            &["--data-dir", data.to_str().unwrap(), "stats"],
        );
        assert!(!cli.status.success(), "{label} cli exited 0");
        let cli_err = String::from_utf8_lossy(&cli.stderr);
        assert!(
            cli_err.contains(UPGRADE_HINT),
            "{label} cli did not refuse with the hint:\n{cli_err}"
        );
        assert_untouched(&db, &before, label);
    }
}

/// The same two files are handed to vestige-upgrade when it sits beside the binary.
#[test]
fn unreadable_and_junk_db_are_handed_to_vestige_upgrade() {
    let cases = [
        ("unreadable", "cannot be read", true),
        ("junk", "not a v3 SQLite", false),
    ];
    for (label, needle, lock_mode) in cases {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(&data).unwrap();
        expose_owned(dir.path(), 0o755);
        expose_owned(&data, if euid() == 0 { 0o777 } else { 0o755 });
        let (db, before) = if lock_mode {
            plant_unreadable(&data)
        } else {
            plant_junk(&data)
        };
        let bin_dir = isolated_bins(dir.path(), &["vestige-mcp", "vestige", "vestige-upgrade"]);
        expose_owned(&bin_dir, 0o755);

        let mcp = spawn_refusing(
            &bin_dir.join("vestige-mcp"),
            &data,
            &["--data-dir", data.to_str().unwrap()],
        );
        assert!(
            !mcp.status.success(),
            "{label} handoff exited 0: {}",
            String::from_utf8_lossy(&mcp.stderr)
        );
        let mcp_err = String::from_utf8_lossy(&mcp.stderr);
        assert!(
            mcp_err.contains(needle),
            "{label} was not handed to vestige-upgrade:\n{mcp_err}"
        );
        assert!(
            !mcp_err.contains(UPGRADE_HINT),
            "{label} refused instead of handing off:\n{mcp_err}"
        );
        assert_untouched(&db, &before, label);
        assert!(
            !dir_has_seg(&data.join(LOG_DIR_NAME)),
            "{label} handoff installed a log"
        );

        let cli = spawn_refusing(
            &bin_dir.join("vestige"),
            &data,
            &["--data-dir", data.to_str().unwrap(), "stats"],
        );
        assert!(!cli.status.success(), "{label} cli handoff exited 0");
        let cli_err = String::from_utf8_lossy(&cli.stderr);
        assert!(
            cli_err.contains(needle),
            "{label} cli was not handed to vestige-upgrade:\n{cli_err}"
        );
        assert_untouched(&db, &before, label);
    }
}

fn plant_unreadable(data: &Path) -> (PathBuf, Vec<u8>) {
    let db = plant(data);
    let before = fs::read(&db).unwrap();
    set_mode(&db, 0);
    assert_eq!(mode_of(&db), 0, "chmod 000 did not stick");
    (db, before)
}

fn plant_junk(data: &Path) -> (PathBuf, Vec<u8>) {
    let db = data.join("vestige.db");
    let before = b"not a sqlite database -- junk bytes\n".to_vec();
    fs::write(&db, &before).unwrap();
    (db, before)
}

fn assert_untouched(db: &Path, before: &[u8], label: &str) {
    let locked = mode_of(db) == 0;
    if locked {
        set_mode(db, 0o644);
    }
    assert_eq!(
        before,
        fs::read(db).unwrap(),
        "{label} changed vestige.db bytes"
    );
    if locked {
        set_mode(db, 0);
        assert_eq!(mode_of(db), 0, "{label} changed the file mode");
    }
    assert_no_sqlite_sidecars(db);
}

fn backup_count(data: &Path) -> usize {
    fs::read_dir(data)
        .unwrap()
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.contains(".v3-backup-") && !name.ends_with(".partial")
        })
        .count()
}

fn kill_upgrade(child: &mut std::process::Child) -> std::process::ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    child.kill().expect("SIGKILL vestige-upgrade");
    let status = child.wait().unwrap();
    assert_eq!(
        status.signal(),
        Some(9),
        "vestige-upgrade was not SIGKILLed: {status}"
    );
    status
}

/// SIGKILL of vestige-upgrade itself, before the staging rename and after it.
/// Both reruns finish cleanly and vestige.db is byte-identical.
#[test]
fn sigkill_vestige_upgrade_before_and_after_rename() {
    let upgrade = PathBuf::from(env!("CARGO_BIN_EXE_vestige-upgrade"));

    let before_dir = tempfile::tempdir().unwrap();
    let before_db = plant(before_dir.path());
    let before_bytes = fs::read(&before_db).unwrap();
    let mut early = Command::new(&upgrade)
        .args(["--db", before_db.to_str().unwrap()])
        .env("STRATA_MIGRATE_SIGKILL_WINDOW", "1")
        .env_remove("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME")
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn vestige-upgrade");
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(30) {
        if dir_has_seg(&staging_directory(before_dir.path()))
            && !dir_has_seg(&before_dir.path().join(LOG_DIR_NAME))
        {
            break;
        }
        if early.try_wait().unwrap().is_some() {
            panic!("vestige-upgrade exited before the pre-rename window");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        dir_has_seg(&staging_directory(before_dir.path())),
        "pre-rename window never opened"
    );
    assert!(!dir_has_seg(&before_dir.path().join(LOG_DIR_NAME)));
    kill_upgrade(&mut early);
    assert!(
        !dir_has_seg(&before_dir.path().join(LOG_DIR_NAME)),
        "SIGKILL before rename published the log"
    );
    assert_eq!(before_bytes, fs::read(&before_db).unwrap());
    assert_no_sqlite_sidecars(&before_db);

    let rerun = Command::new(&upgrade)
        .args(["--db", before_db.to_str().unwrap()])
        .env_remove("STRATA_MIGRATE_SIGKILL_WINDOW")
        .env_remove("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME")
        .output()
        .expect("rerun after pre-rename SIGKILL");
    let rerun_err = String::from_utf8_lossy(&rerun.stderr);
    assert!(
        rerun.status.success(),
        "rerun after pre-rename SIGKILL failed: {rerun_err}"
    );
    assert!(dir_has_seg(&before_dir.path().join(LOG_DIR_NAME)));
    assert!(!staging_directory(before_dir.path()).exists());
    assert_eq!(
        before_bytes,
        fs::read(&before_db).unwrap(),
        "rerun after pre-rename SIGKILL changed vestige.db"
    );
    assert_no_sqlite_sidecars(&before_db);

    let after_dir = tempfile::tempdir().unwrap();
    let after_db = plant(after_dir.path());
    let after_bytes = fs::read(&after_db).unwrap();
    let mut late = Command::new(&upgrade)
        .args(["--db", after_db.to_str().unwrap()])
        .env("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME", "1")
        .env_remove("STRATA_MIGRATE_SIGKILL_WINDOW")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn vestige-upgrade after-rename window");
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(60) {
        if dir_has_seg(&after_dir.path().join(LOG_DIR_NAME))
            && !staging_directory(after_dir.path()).exists()
        {
            break;
        }
        if late.try_wait().unwrap().is_some() {
            panic!("vestige-upgrade exited before the post-rename window");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        dir_has_seg(&after_dir.path().join(LOG_DIR_NAME)),
        "post-rename window never opened"
    );
    assert!(!staging_directory(after_dir.path()).exists());
    assert!(
        late.try_wait().unwrap().is_none(),
        "process left the post-rename window before SIGKILL"
    );
    kill_upgrade(&mut late);
    let backups = backup_count(after_dir.path());
    assert!(backups >= 1, "upgrade did not back up vestige.db");
    assert_eq!(after_bytes, fs::read(&after_db).unwrap());
    assert_no_sqlite_sidecars(&after_db);

    let again = Command::new(&upgrade)
        .args(["--db", after_db.to_str().unwrap()])
        .env_remove("STRATA_MIGRATE_SIGKILL_WINDOW")
        .env_remove("VESTIGE_UPGRADE_SIGKILL_AFTER_RENAME")
        .output()
        .expect("rerun after post-rename SIGKILL");
    let again_err = String::from_utf8_lossy(&again.stderr);
    assert!(
        again.status.success(),
        "rerun after post-rename SIGKILL failed: {again_err}"
    );
    assert!(
        again_err.contains("already present"),
        "rerun imported again instead of keeping the published log:\n{again_err}"
    );
    assert_eq!(
        backups,
        backup_count(after_dir.path()),
        "rerun wrote another v3 backup"
    );
    assert!(!staging_directory(after_dir.path()).exists());
    assert_eq!(
        after_bytes,
        fs::read(&after_db).unwrap(),
        "rerun after post-rename SIGKILL changed vestige.db"
    );
    assert_no_sqlite_sidecars(&after_db);
}
