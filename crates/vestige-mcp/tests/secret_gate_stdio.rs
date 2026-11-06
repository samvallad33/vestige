//! Real stdio session on a Strata store: the credential gate covers every
//! stored field of a memory and of an intention, and no response echoes a
//! matched value back.

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

    /// The whole `tools/call` result, refusals included.
    fn call(&mut self, name: &str, arguments: Value) -> Value {
        self.rpc("tools/call", json!({"name": name, "arguments": arguments}))
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
            "clientInfo": {"name": "secret-gate-stdio", "version": "1"}
        }),
    );
    session.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
}

fn token(fill: &str) -> String {
    format!("ghp_{}", fill.repeat(36))
}

fn is_refusal(result: &Value) -> bool {
    result["isError"] == json!(true)
}

#[test]
fn smart_ingest_refuses_a_credential_in_tags_source_or_scope_without_echo() {
    let dir = tempfile::tempdir().expect("data dir");
    let mut session = Session::spawn(dir.path());
    handshake(&mut session);
    let secret = token("A");

    let cases = [
        json!({"content": "Synthetic note", "forceCreate": true, "tags": ["safe", secret]}),
        json!({"content": "Synthetic note", "forceCreate": true, "source": secret}),
        json!({"content": "Synthetic note", "forceCreate": true, "scope": secret}),
        json!({"content": "Synthetic note", "tags": ["safe"], "scope": secret,
               "previewTagSuggestions": true}),
        json!({"items": [{"content": "Synthetic note", "tags": [secret]}]}),
        json!({"items": [{"content": "Synthetic note", "source": secret}]}),
        json!({"items": [{"content": "Synthetic note", "scope": secret}]}),
    ];
    for tool in ["smart_ingest", "remember"] {
        for (index, arguments) in cases.iter().enumerate() {
            let result = session.call(tool, arguments.clone());
            let text = result.to_string();
            assert!(
                !text.contains(&secret),
                "case {index}: the response must not echo the credential: {text}"
            );
            assert!(
                is_refusal(&result) || text.contains("rejected"),
                "case {index}: a credential-bearing write must be refused: {text}"
            );
        }
    }

    session.shutdown();

    let storage = vestige_mcp::strata_memory::open(dir.path()).expect("reopen");
    assert_eq!(
        storage.get_stats().expect("stats").total_nodes,
        0,
        "refused smart_ingest calls left memories in the log"
    );
}

#[test]
fn intention_set_refuses_a_credential_without_echo_or_write() {
    let dir = tempfile::tempdir().expect("data dir");
    let mut session = Session::spawn(dir.path());
    handshake(&mut session);
    let secret = token("B");

    for arguments in [
        json!({"action": "set", "description": format!("rotate {secret} on Friday")}),
        json!({"action": "set", "description": "Synthetic reminder", "scope": secret}),
        json!({"action": "set", "description": "Synthetic reminder",
               "trigger": {"type": "event", "condition": format!("after {secret}")}}),
        json!({"action": "set", "description": "Synthetic reminder",
               "trigger": {"type": "context", "topic": secret}}),
    ] {
        let result = session.call("intention", arguments.clone());
        let text = result.to_string();
        assert!(
            !text.contains(&secret),
            "the response must not echo the credential: {text}"
        );
        assert!(
            is_refusal(&result),
            "a credential-bearing intention must be refused: {arguments} -> {text}"
        );
    }
    let listed = session.call(
        "intention",
        json!({"action": "list", "filter_status": "all"}),
    );
    assert!(
        !listed.to_string().contains(&secret),
        "no intention carries the credential: {listed}"
    );
    assert_eq!(
        listed.to_string().matches("Synthetic reminder").count(),
        0,
        "refused intentions left rows behind: {listed}"
    );
    session.shutdown();
}
