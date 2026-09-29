//! Real stdio session: `intention set` admits one receipt on a fresh Strata
//! store, and `vestige strata-verify` accepts that store plus a migrated
//! v3.1.1 fixture. The verify half needs `--features migrate-to-strata`.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Session {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .args(["--no-http", "--data-dir"])
            .arg(data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                    if let Ok(mut sink) = sink.lock() {
                        sink.push(line);
                    }
                }
            });
        }
        Self {
            child,
            stdin: Some(stdin),
            lines,
            stderr,
            next_id: 0,
        }
    }

    fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}")
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| {
                panic!(
                    "stdio write failed: {error}; stderr: {:?}",
                    self.stderr.lock().unwrap()
                );
            });
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let request = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.send(&request.to_string());
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let wait = deadline.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                panic!("timed out waiting for {method}");
            }
            let line = self
                .lines
                .recv_timeout(wait)
                .unwrap_or_else(|_| panic!("no response for {method}"));
            let response: Value = serde_json::from_str(&line).expect("json");
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            assert!(response.get("error").is_none(), "{response}");
            return response["result"].clone();
        }
    }

    fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.rpc("tools/call", json!({"name": name, "arguments": arguments}));
        assert_eq!(result["isError"], json!(false), "{result}");
        result.get("structuredContent").cloned().unwrap_or_else(|| {
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
        })
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll") {
                Some(status) => {
                    assert!(status.success(), "{status}");
                    return;
                }
                None if Instant::now() >= deadline => panic!("stdio server did not exit"),
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn handshake(session: &mut Session) {
    session.rpc(
        "initialize",
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "intention-stdio", "version": "1"}
        }),
    );
    session.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
}

fn no_sqlite(dir: &Path) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if name.ends_with(".sqlite")
                || name.ends_with(".sqlite3")
                || name.ends_with(".db")
                || name.ends_with(".db-wal")
                || name.ends_with(".db-shm")
            {
                return false;
            }
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    true
}

fn set_intention(session: &mut Session) -> String {
    let created = session.tool(
        "intention",
        json!({
            "action": "set",
            "description": "Synthetic reminder",
            "trigger": {"type": "time", "at": "2020-01-01T00:00:00Z"}
        }),
    );
    let id = created["intentionId"].as_str().unwrap().to_string();
    let receipt = created["receiptId"].as_str().unwrap();
    assert!(receipt.starts_with("eff-"), "{created}");
    assert!(
        created["receipt"]["retrieved"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == &id)
    );
    id
}

fn graph(session: &mut Session, command: Value) -> Value {
    session.tool(
        "intention",
        json!({
            "action": "graph",
            "scope": "user",
            "at": "2026-10-01T09:00:00Z",
            "command": command,
        }),
    )
}

#[test]
fn intention_set_list_check_update_graph_survives_restart() {
    let dir = tempfile::tempdir().expect("data dir");
    let id = {
        let mut session = Session::spawn(dir.path());
        handshake(&mut session);
        let id = set_intention(&mut session);
        let listed = session.tool("intention", json!({"action": "list"}));
        assert!(
            listed["intentions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == id && row["description"] == "Synthetic reminder"),
            "{listed}"
        );
        let checked = session.tool(
            "intention",
            json!({"action": "check", "context": {"current_time": "2020-01-02T00:00:00Z"}}),
        );
        assert!(
            checked["triggered"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == id),
            "{checked}"
        );
        assert!(
            checked["receiptId"].as_str().unwrap().starts_with("eff-"),
            "{checked}"
        );
        let updated = session.tool(
            "intention",
            json!({"action": "update", "id": id, "status": "complete"}),
        );
        assert_eq!(updated["success"], json!(true), "{updated}");
        assert!(
            updated["receiptId"].as_str().unwrap().starts_with("eff-"),
            "{updated}"
        );
        let fulfilled = session.tool(
            "intention",
            json!({"action": "list", "filter_status": "fulfilled"}),
        );
        assert!(
            fulfilled["intentions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == id),
            "{fulfilled}"
        );
        let planned = graph(
            &mut session,
            json!({
                "action": "plan",
                "id": "fixture-plan",
                "description": "Synthetic graph plan",
                "requirements": [],
                "conflict_keys": []
            }),
        );
        assert!(planned["journal_seq"].as_i64().unwrap() >= 1, "{planned}");
        let replayed = graph(&mut session, json!({"action": "replay"}));
        assert_eq!(replayed["matched"], json!(true), "{replayed}");
        assert_eq!(replayed["commands"], json!(1), "{replayed}");
        let explained = graph(
            &mut session,
            json!({"action": "explain", "id": "fixture-plan"}),
        );
        assert!(
            explained.to_string().contains("Synthetic graph plan"),
            "{explained}"
        );
        let visible = session.tool(
            "intention",
            json!({"action": "list", "filter_status": "all", "limit": 50}),
        );
        let visible_text = visible.to_string();
        assert!(!visible_text.contains("Synthetic graph plan"), "{visible}");
        assert!(!visible_text.contains("igj|"), "{visible}");
        assert!(
            visible["intentions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == id),
            "{visible}"
        );
        session.shutdown();
        id
    };
    assert!(no_sqlite(dir.path()));
    let mut again = Session::spawn(dir.path());
    handshake(&mut again);
    let fulfilled = again.tool(
        "intention",
        json!({"action": "list", "filter_status": "fulfilled"}),
    );
    assert!(
        fulfilled["intentions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["id"] == id && row["description"] == "Synthetic reminder"),
        "{fulfilled}"
    );
    let active = again.tool("intention", json!({"action": "list"}));
    assert!(
        !active["intentions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["id"] == id),
        "{active}"
    );
    let replayed = graph(&mut again, json!({"action": "replay"}));
    assert_eq!(replayed["matched"], json!(true), "{replayed}");
    assert_eq!(replayed["commands"], json!(1), "{replayed}");
    assert_eq!(
        replayed["state_digest"].as_str().unwrap().len(),
        64,
        "{replayed}"
    );
    let explained = graph(
        &mut again,
        json!({"action": "explain", "id": "fixture-plan"}),
    );
    assert!(
        explained.to_string().contains("Synthetic graph plan"),
        "{explained}"
    );
    again.shutdown();
    assert!(no_sqlite(dir.path()));

    // An unsealed store has no store.meta. strata-verify still checks log/*.seg.
    #[cfg(feature = "migrate-to-strata")]
    strata_verify(dir.path());
}

#[cfg(feature = "migrate-to-strata")]
fn strata_verify(dir: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(["strata-verify"])
        .arg(dir)
        .output()
        .expect("spawn vestige strata-verify");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "strata-verify failed: {stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("OK"), "{stdout}");
}

#[cfg(feature = "migrate-to-strata")]
#[test]
fn strata_verify_accepts_a_migrated_v3_1_1_store() {
    let tmp = tempfile::tempdir().expect("temp");
    let source = tmp.path().join("v3.1.1-sample.sqlite");
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    std::fs::copy(&fixture, &source).expect("copy fixture");
    let before = std::fs::read(&source).expect("read fixture");
    let dest = tmp.path().join("migrated");
    let migrated = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(["migrate-to-strata", "--from"])
        .arg(&source)
        .arg("--to")
        .arg(&dest)
        .output()
        .expect("spawn migrate-to-strata");
    assert!(
        migrated.status.success(),
        "migrate failed: {} {}",
        String::from_utf8_lossy(&migrated.stdout),
        String::from_utf8_lossy(&migrated.stderr)
    );
    assert_eq!(
        std::fs::read(&source).unwrap(),
        before,
        "source was modified"
    );
    strata_verify(&dest);
}
