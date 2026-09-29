//! Real-stdio proof for `selftest`.
//!
//! Drives the shipped `vestige-mcp` binary over line-framed JSON-RPC. The
//! caller data dir is a Strata log seeded before the process starts. After
//! two `selftest` calls the log bytes and the seeded memory must match.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");

        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines() {
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
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr_lines(&self) -> Vec<String> {
        self.stderr.lock().expect("stderr").clone()
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| {
                panic!(
                    "stdin write failed ({error}). stderr: {:?}",
                    self.stderr_lines()
                )
            });
    }

    fn read_line(&mut self) -> String {
        loop {
            let line = match self.stdout.recv_timeout(RPC_TIMEOUT) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => panic!(
                    "no response within {RPC_TIMEOUT:?}. stderr: {:?}",
                    self.stderr_lines()
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdout closed. stderr: {:?}", self.stderr_lines())
                }
            };
            let value: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            return line;
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
        let line = self.read_line();
        let response: Value = serde_json::from_str(&line)
            .unwrap_or_else(|error| panic!("non-JSON {line:?}: {error}"));
        assert_eq!(response["id"], json!(id), "{response}");
        response
    }

    fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("error").is_none(),
            "{method} error: {response} stderr: {:?}",
            self.stderr_lines()
        );
        response["result"].clone()
    }

    fn handshake(&mut self) {
        self.result(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "selftest-stdio", "version": "1" },
            })),
        );
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        self.write_line(&note.to_string());
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.result(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        assert_ne!(result["isError"], json!(true), "{name} isError: {result}");
        if let Some(structured) = result.get("structuredContent") {
            return structured.clone();
        }
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} returned no text: {result}"));
        serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll") {
                Some(status) => {
                    assert!(
                        status.success(),
                        "exit {status}. stderr: {:?}",
                        self.stderr_lines()
                    );
                    return;
                }
                None if std::time::Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!(
                        "vestige-mcp did not exit. stderr: {:?}",
                        self.stderr_lines()
                    );
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn tree_bytes(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if rel.ends_with("strata.lock") {
                continue;
            }
            out.insert(rel, std::fs::read(&path).unwrap());
        }
    }
    out
}

#[test]
fn selftest_over_stdio_completes_passes_every_check_and_leaves_the_user_store() {
    let dir = tempfile::tempdir().expect("data dir");
    let seeded = {
        let storage = vestige_mcp::strata_memory::open(dir.path()).expect("seed store");
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "caller memory that selftest must not alter".to_string(),
                ..Default::default()
            })
            .expect("seed");
        node.id
    };

    let mut server = Server::spawn(dir.path());
    server.handshake();
    let before = tree_bytes(dir.path());

    let first = server.call_tool("selftest", json!({}));
    let second = server.call_tool("selftest", json!({}));
    let after = tree_bytes(dir.path());
    server.shutdown();

    assert_eq!(first["status"], json!("completed"), "{first}");
    assert_eq!(first["passed"], json!(true), "{first}");
    assert_eq!(first["live_store_touched"], json!(false), "{first}");
    assert_eq!(first["temp_store_deleted"], json!(true), "{first}");
    assert!(
        !first.to_string().contains("pending_strata"),
        "selftest must not defer: {first}"
    );
    let checks = first["checks"].as_array().expect("checks");
    assert!(!checks.is_empty());
    for item in checks {
        assert_eq!(item["pass"], json!(true), "{item} in {first}");
    }
    assert_eq!(first, second, "stdio selftest must be deterministic");
    assert_eq!(before, after, "caller log bytes changed");

    let storage = vestige_mcp::strata_memory::open(dir.path()).expect("reopen");
    let nodes = storage.get_all_nodes(100, 0).expect("nodes");
    assert_eq!(
        nodes.len(),
        1,
        "planted memories leaked into the caller store"
    );
    assert_eq!(nodes[0].id, seeded);
    assert_eq!(
        nodes[0].content,
        "caller memory that selftest must not alter"
    );
    assert!(
        storage.get_all_connections().expect("edges").is_empty(),
        "planted edges leaked into the caller store"
    );
}
