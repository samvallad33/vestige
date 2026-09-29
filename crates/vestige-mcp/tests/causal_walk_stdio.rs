//! Real stdio: `causal_walk` returns the recorded cause, not `pending_strata`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::Duration;

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    transcript: Vec<String>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let binary = server_binary();
        let mut child = Command::new(&binary)
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn {}: {err}", binary.display()));
        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                eprintln!("stderr: {line}");
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            stdout,
            transcript: Vec::new(),
            next_id: 0,
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        let line = message.to_string();
        self.transcript.push(format!(">> {line}"));
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}")
            .and_then(|()| stdin.flush())
            .expect("write");
        let response = self.read_response();
        let value: Value = serde_json::from_str(&response).expect("json");
        assert_eq!(value["id"], json!(id), "{value}");
        assert!(value.get("error").is_none(), "{value}");
        value["result"].clone()
    }

    fn read_response(&mut self) -> String {
        loop {
            let line = match self.stdout.recv_timeout(Duration::from_secs(90)) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("stdio timeout\n{}", self.transcript.join("\n"))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdio closed\n{}", self.transcript.join("\n"))
                }
            };
            if line.is_empty() {
                continue;
            }
            self.transcript.push(format!("<< {line}"));
            let value: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            let notification = value.get("id").is_none() && value.get("method").is_some();
            if !notification {
                return line;
            }
        }
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "causal-walk-stdio", "version": "1" },
            })),
        );
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        let line = note.to_string();
        self.transcript.push(format!(">> {line}"));
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}")
            .and_then(|()| stdin.flush())
            .expect("notify");
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.request(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        assert_ne!(result["isError"], true, "{result}");
        result["structuredContent"].clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn server_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_vestige-mcp"))
}

#[test]
fn causal_walk_stdio_returns_the_recorded_cause() {
    let dir = std::env::temp_dir().join(format!(
        "vestige-causal-walk-stdio-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (effect, cause, decoy) =
        vestige_mcp::tools::causal_walk::seed_recorded_cause(&dir).expect("seed strata log");

    let mut server = Server::spawn(&dir);
    server.handshake();
    let args = json!({
        "scope": "user",
        "start_points": [{ "kind": "logged_write", "node_id": effect }],
    });
    let first = server.call_tool("causal_walk", args.clone());
    let second = server.call_tool("causal_walk", args);
    println!("{}", server.transcript.join("\n"));

    assert_eq!(first, second, "stdio walk must be deterministic");
    let body = serde_json::to_string(&first).unwrap();
    assert!(!body.contains("pending_strata"), "{first}");
    assert_eq!(first["status"], "completed");
    assert_eq!(first["direction"], "backward");
    assert!(first["needs_report"].is_null(), "{first}");
    let cause_row = first["causes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == cause)
        .unwrap_or_else(|| panic!("missing recorded cause: {first}"));
    assert_eq!(cause_row["path"][0]["source_id"], cause);
    assert_eq!(cause_row["path"][0]["target_id"], effect);
    assert_eq!(cause_row["path"][0]["link_type"], "derived_from");
    let ids: Vec<&str> = first["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|node| node["id"].as_str().unwrap())
        .collect();
    assert!(
        !ids.contains(&decoy.as_str()),
        "same text is not a cause: {first}"
    );
    assert_eq!(
        ids.len(),
        2,
        "only the start and its recorded cause: {first}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
