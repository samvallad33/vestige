//! Real stdio: import a v3 store, boot `vestige-mcp`, and walk the graph.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::Duration;

use serde_json::{Value, json};

const FIXTURE_SHA256: &str = "961f12d1750dbd2f6e6a8fc365c4a4bb42dd1b7e49cbf465985a36b665e10479";
const A: &str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const B: &str = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const C: &str = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
const LEFT: &str = "11111111-1111-4111-8111-111111111111";
const MID: &str = "22222222-2222-4222-8222-222222222222";
const RIGHT: &str = "33333333-3333-4333-8333-333333333333";

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite")
}

fn sha256_file(path: &Path) -> String {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum utf8")
        .split_whitespace()
        .next()
        .expect("sha256 digest")
        .to_string()
}

fn plant_touched_chain(db: &Path) {
    let conn = rusqlite::Connection::open(db).unwrap();
    for (id, content) in [
        (A, "chain node A declared touched"),
        (B, "chain node B declared touched"),
        (C, "chain node C declared touched"),
    ] {
        conn.execute(
            "INSERT INTO knowledge_nodes
             (id, content, node_type, created_at, updated_at, last_accessed, tags, source)
             VALUES (?1, ?2, 'fact', '2026-01-15T10:00:00+00:00', '2026-02-20T11:30:00+00:00',
                     '2026-03-01T09:15:00+00:00', '[]', 'fixture')",
            rusqlite::params![id, content],
        )
        .unwrap();
    }
    for (source, target) in [(A, B), (B, C)] {
        conn.execute(
            "INSERT INTO memory_connections
             (source_id, target_id, strength, link_type, created_at, last_activated, activation_count)
             VALUES (?1, ?2, 1.0, 'touched', '2026-01-16T08:00:00+00:00', '2026-03-02T08:00:00+00:00', 1)",
            rusqlite::params![source, target],
        )
        .unwrap();
    }
}

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Receiver<String>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let raw_stdout = child.stdout.take().unwrap();
        let raw_stderr = child.stderr.take().unwrap();
        let (out_tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if out_tx.send(line).is_err() {
                    return;
                }
            }
        });
        let (err_tx, stderr) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                if err_tx.send(line).is_err() {
                    return;
                }
            }
        });
        Self {
            child,
            stdin,
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr_tail(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.stderr.try_recv() {
            lines.push(line);
        }
        lines
    }

    fn line(&mut self) -> String {
        loop {
            match self.stdout.recv_timeout(Duration::from_secs(60)) {
                Ok(line) => {
                    let value: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
                    if value.get("id").is_none() && value.get("method").is_some() {
                        continue;
                    }
                    return line;
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("stdio timeout. stderr: {:?}", self.stderr_tail())
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdio closed. stderr: {:?}", self.stderr_tail())
                }
            }
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{message}").unwrap();
        stdin.flush().unwrap();
        let raw = self.line();
        let response: Value = serde_json::from_str(&raw).expect("json");
        assert_eq!(response["id"], json!(id), "{response}");
        if let Some(error) = response.get("error") {
            panic!("{method} error: {error}");
        }
        response["result"].clone()
    }

    fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.request(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        if let Some(structured) = result.get("structuredContent") {
            return structured.clone();
        }
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} returned no text: {result}"));
        serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn imported_graph_chain_over_stdio() {
    assert_eq!(sha256_file(&fixture()), FIXTURE_SHA256);
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source.sqlite");
    std::fs::copy(fixture(), &source).unwrap();
    plant_touched_chain(&source);
    let source_sha = sha256_file(&source);
    let data = root.path().join("data");
    let log = data.join("log");
    strata_migrate::migrate(&source, &log).expect("import");
    assert_eq!(sha256_file(&fixture()), FIXTURE_SHA256);
    assert_eq!(sha256_file(&source), source_sha);
    let verified = strata_verify::verify_path(&log);
    assert!(
        verified.ok,
        "strata-verify: {}",
        verified.failures.join("; ")
    );

    let mut server = Server::spawn(&data);
    server.request(
        "initialize",
        Some(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "imported-graph", "version": "1" }
        })),
    );
    let stdin = server.stdin.as_mut().unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
    )
    .unwrap();

    // Query recall is not a scan. The imported node is readable by exact handle.
    let refused = server.tool("recall", json!({ "query": LEFT }));
    assert!(
        refused["error"]
            .as_str()
            .is_some_and(|err| err.contains("similarity_disabled")),
        "query recall must stay off the imported log: {refused}"
    );
    let found = server.tool("recall", json!({ "handle": LEFT }));
    let nodes = found["nodes"]
        .as_array()
        .unwrap_or_else(|| panic!("handle recall nodes: {found}"));
    assert!(
        nodes.iter().any(|row| row["id"] == json!(LEFT)),
        "handle recall missed the imported node: {found}"
    );
    let got = server.tool("memory", json!({ "action": "get", "id": LEFT }));
    assert_eq!(got["found"], json!(true), "{got}");
    assert_eq!(got["node"]["id"], json!(LEFT));

    let declared = server.tool("graph", json!({ "action": "chain", "from": A, "to": C }));
    assert!(
        declared.get("message").and_then(|m| m.as_str())
            != Some("No chain found between these memories"),
        "touched A->B->C must chain: {declared}"
    );
    let steps = declared["steps"].as_array().expect("steps");
    let step_ids: Vec<_> = steps
        .iter()
        .filter_map(|step| step["memory_id"].as_str())
        .collect();
    assert!(
        step_ids.contains(&A) && step_ids.contains(&B) && step_ids.contains(&C),
        "{declared}"
    );

    let direct = server.tool(
        "graph",
        json!({ "action": "chain", "from": LEFT, "to": RIGHT }),
    );
    assert!(
        direct["steps"]
            .as_array()
            .is_some_and(|steps| !steps.is_empty()),
        "recorded touched edge must chain: {direct}"
    );

    let inferred = server.tool(
        "graph",
        json!({ "action": "chain", "from": LEFT, "to": MID }),
    );
    assert_eq!(
        inferred["message"],
        json!("No chain found between these memories"),
        "legacy_inferred-only path must not chain: {inferred}"
    );
    let blast = server.tool("blast_radius", json!({ "root_id": LEFT }));
    let reached: Vec<_> = blast["affected"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .filter(|id| *id != LEFT)
        .collect();
    assert!(
        !reached.contains(&MID),
        "blast walked a legacy_inferred link: {blast}"
    );
    let walk = server.tool(
        "causal_walk",
        json!({ "start_points": [{ "kind": "logged_write", "node_id": LEFT }] }),
    );
    let walk_text = walk.to_string();
    assert!(
        !walk_text.contains(MID),
        "causal_walk walked a legacy_inferred link: {walk}"
    );

    drop(server);
    assert_eq!(sha256_file(&fixture()), FIXTURE_SHA256);
    assert_eq!(sha256_file(&source), source_sha);
    let again = strata_verify::verify_path(&log);
    assert!(
        again.ok,
        "strata-verify after stdio: {}",
        again.failures.join("; ")
    );
}
