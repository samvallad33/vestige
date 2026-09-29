//! Real-stdio proof for `selftest`.
//!
//! Spawns the shipped `vestige-mcp` binary, speaks line-framed JSON-RPC, and
//! checks the recorded-edge walk completes with every check passing while the
//! user Strata log is unchanged.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SENTINEL: &str = "Sentinel user memory: the kettle is on the third shelf.";
const RPC_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Receiver<String>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(server_binary())
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE", "0")
            .env("VESTIGE_BACKFILL_AUTOFIRE", "0")
            .env("VESTIGE_FAILURE_FEEDBACK", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");

        let stdin = child.stdin.take().expect("child stdin");
        let raw_stdout = child.stdout.take().expect("child stdout");
        let raw_stderr = child.stderr.take().expect("child stderr");

        let (stdout_tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines() {
                match line {
                    Ok(line) => {
                        if stdout_tx.send(line).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        });
        let (stderr_tx, stderr) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                if stderr_tx.send(line).is_err() {
                    return;
                }
            }
        });

        Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.stderr.try_recv() {
            lines.push(line);
        }
        lines
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| {
                panic!(
                    "writing to vestige-mcp stdin failed ({error}). stderr: {:?}",
                    self.stderr_lines()
                )
            });
    }

    fn read_response(&mut self, id: u64) -> Value {
        let deadline = Instant::now() + RPC_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!(
                    "timed out waiting for response {id}. stderr: {:?}",
                    self.stderr_lines()
                );
            }
            let line = self.stdout.recv_timeout(remaining).unwrap_or_else(|_| {
                panic!(
                    "vestige-mcp closed stdout waiting for {id}. stderr: {:?}",
                    self.stderr_lines()
                )
            });
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("stdout was not JSON ({error}): {line}"));
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            if value.get("id") != Some(&json!(id)) {
                panic!("unexpected response id: {value}");
            }
            return value;
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
        self.read_response(id)
    }

    fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("error").is_none(),
            "{method} returned an error: {response}"
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
        let note = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });
        self.write_line(&note.to_string());
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.result(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        assert_ne!(
            result["isError"],
            json!(true),
            "tool {name} failed: {result}"
        );
        result
            .get("structuredContent")
            .cloned()
            .unwrap_or_else(|| panic!("tool {name} returned no structuredContent: {result}"))
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll vestige-mcp") {
                Some(status) => {
                    assert!(
                        status.success(),
                        "stdin EOF must be a clean shutdown, got {status}. stderr: {:?}",
                        self.stderr_lines()
                    );
                    return;
                }
                None if Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!("vestige-mcp did not exit within 30s of stdin EOF");
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
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

/// Content hash of the Strata log segments and head state. The lock file is
/// the live writer's pid and is not part of the store.
fn log_digest(data_dir: &Path) -> BTreeMap<String, String> {
    let log_dir = data_dir.join("log");
    let mut files = BTreeMap::new();
    let entries = fs::read_dir(&log_dir)
        .unwrap_or_else(|error| panic!("reading {} failed: {error}", log_dir.display()));
    for entry in entries {
        let entry = entry.expect("log dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "strata.lock" {
            continue;
        }
        let meta = entry.metadata().expect("log metadata");
        if !meta.is_file() {
            continue;
        }
        let bytes = fs::read(entry.path()).expect("read log file");
        let digest = Sha256::digest(&bytes);
        files.insert(name, hex_encode(&digest));
    }
    assert!(!files.is_empty(), "the user Strata log must exist");
    files
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

#[test]
fn selftest_over_stdio_completes_with_every_check_and_leaves_the_user_store_unchanged() {
    let dir = tempfile::tempdir().expect("data dir");
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let ingested = server.call_tool(
        "smart_ingest",
        json!({ "content": SENTINEL, "forceCreate": true }),
    );
    assert_eq!(ingested["success"], json!(true), "{ingested}");
    let sentinel_id = ingested["nodeId"]
        .as_str()
        .unwrap_or_else(|| panic!("smart_ingest returned no nodeId: {ingested}"))
        .to_string();

    let before_get = server.call_tool("memory", json!({ "action": "get", "id": sentinel_id }));
    assert_eq!(before_get["found"], json!(true), "{before_get}");
    assert_eq!(before_get["node"]["content"], json!(SENTINEL));

    let before_status = server.call_tool("memory_status", json!({ "view": "retention" }));
    let before_total = before_status["totalMemories"].clone();
    let before_log = log_digest(dir.path());

    let first = server.call_tool("selftest", json!({}));
    let second = server.call_tool("selftest", json!({}));

    assert_eq!(first["status"], json!("completed"), "{first}");
    assert_eq!(first["all_passed"], json!(true), "{first}");
    assert_eq!(first["live_store_touched"], json!(false), "{first}");
    assert_eq!(first["temp_store_deleted"], json!(true), "{first}");
    assert_eq!(first, second, "selftest must be deterministic over stdio");
    let checks = first["checks"].as_array().expect("checks");
    assert!(!checks.is_empty(), "{first}");
    for check in checks {
        assert_eq!(check["pass"], json!(true), "{check}");
    }
    assert!(
        !first.to_string().contains("pending_strata"),
        "selftest must not defer on strata: {first}"
    );
    assert!(
        !first.to_string().contains(SENTINEL),
        "the selftest payload must not echo the user store"
    );

    let after_log = log_digest(dir.path());
    assert_eq!(before_log, after_log, "user Strata log bytes changed");

    let after_get = server.call_tool("memory", json!({ "action": "get", "id": sentinel_id }));
    assert_eq!(after_get, before_get, "sentinel memory changed");
    let after_status = server.call_tool("memory_status", json!({ "view": "retention" }));
    assert_eq!(
        after_status["totalMemories"], before_total,
        "user memory count changed: before {before_status} after {after_status}"
    );

    server.shutdown();

    let storage = vestige_mcp::strata_memory::open(dir.path()).expect("reopen user store");
    let nodes = storage.get_all_nodes(100, 0).expect("list user nodes");
    assert_eq!(
        nodes.len(),
        1,
        "planted selftest memories leaked: {nodes:?}"
    );
    assert_eq!(nodes[0].id, sentinel_id);
    assert_eq!(nodes[0].content, SENTINEL);
    assert!(
        storage
            .get_all_connections()
            .expect("list user edges")
            .is_empty(),
        "selftest edges leaked into the user store"
    );
}
