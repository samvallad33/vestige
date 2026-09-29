//! `maintain(action=dream)` over the default binary's stdio.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use serde_json::{Value, json};
use vestige_core::{ConnectionRecord, IngestInput, MemoryStore};

struct Server {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl Server {
    fn spawn(dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .args(["--no-http", "--data-dir"])
            .arg(dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stderr = child.stderr.take().expect("stderr");
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                line.clear();
            }
        });
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut server = Self {
            child,
            stdin: Some(stdin),
            stdout,
            next_id: 1,
        };
        server.handshake();
        server
    }

    fn handshake(&mut self) {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "dream-stdio", "version": "0"}
            }
        }));
        let ready = self.read_id(id);
        assert!(ready.get("result").is_some(), "{ready}");
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }));
        let message = self.read_id(id);
        assert!(message.get("error").is_none(), "{message}");
        let result = &message["result"];
        assert_ne!(result["isError"], json!(true), "{result}");
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    fn send(&mut self, value: &Value) {
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{value}").unwrap();
        stdin.flush().unwrap();
    }

    fn read_id(&mut self, id: u64) -> Value {
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).expect("stdout");
            assert!(n > 0, "server closed stdout");
            let value: Value = serde_json::from_str(line.trim()).expect("json");
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return value;
            }
        }
    }

    fn shutdown(mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn seed(dir: &Path) -> (Vec<String>, String, i32) {
    let storage = vestige_mcp::strata_memory::open(dir).expect("seed");
    let mut ids = Vec::new();
    for content in ["aa", "bb", "cc", "dd", "ee", "ff"] {
        ids.push(
            storage
                .ingest(IngestInput {
                    content: content.into(),
                    ..IngestInput::default()
                })
                .unwrap()
                .id,
        );
    }
    let created = storage.get_node(&ids[0]).unwrap().unwrap().created_at;
    let link = |source: &str, target: &str, kind: &str, strength: f64| {
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength,
                link_type: kind.to_string(),
                created_at: created,
                last_activated: created,
                activation_count: 1,
            })
            .unwrap();
    };
    link(&ids[0], &ids[1], "derived_from", 1.0);
    link(&ids[1], &ids[2], "evidence_of", 0.8);
    link(&ids[3], &ids[4], "touched", 0.1);
    let reps = storage.get_node(&ids[0]).unwrap().unwrap().reps;
    let quiet = ids[5].clone();
    drop(storage);
    (ids, quiet, reps)
}

fn assert_no_sqlite(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            assert!(
                !name.ends_with(".sqlite")
                    && !name.ends_with(".sqlite3")
                    && !name.ends_with(".db")
                    && !name.ends_with(".db-wal")
                    && !name.ends_with(".db-shm"),
                "sqlite file {path:?}"
            );
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
}

fn strata_verify_bin() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-verify/Cargo.toml");
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-verify/target");
    let status = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .args(["--bin", "strata-verify", "--target-dir"])
        .arg(&target)
        .env_remove("CARGO_TARGET_DIR")
        .status()
        .expect("build strata-verify");
    assert!(status.success(), "strata-verify failed to build");
    target.join("debug/strata-verify")
}

#[test]
fn dream_over_stdio_completes_from_recorded_edges_and_fsrs() {
    let dir = std::env::temp_dir().join(format!("vestige-dream-stdio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (ids, quiet, reps_before) = seed(&dir);

    let mut server = Server::spawn(&dir);
    let value = server.call(
        "maintain",
        json!({ "action": "dream", "min_similarity": 0.99 }),
    );
    assert_eq!(value["status"], "completed", "{value}");
    assert_eq!(value["basis"], "recorded_edges_fsrs", "{value}");
    assert_eq!(value["edgesConsolidated"], 3, "{value}");
    assert_eq!(value["connectionsPersisted"], 0, "{value}");
    let text = serde_json::to_string(&value).unwrap();
    assert!(!text.contains("pending_strata"), "{value}");
    assert!(!text.contains("not implemented"), "{value}");
    assert!(!text.contains("aa"), "{value}");
    let mut reviewed: Vec<String> = value["reviews"]
        .as_array()
        .unwrap()
        .iter()
        .map(|review| review["id"].as_str().unwrap().to_string())
        .collect();
    reviewed.sort();
    let mut expected = ids[..5].to_vec();
    expected.sort();
    assert_eq!(reviewed, expected, "{value}");

    let strengthened = server.call("memory", json!({ "action": "get", "id": ids[0] }));
    assert_eq!(
        strengthened["node"]["reps"],
        json!(reps_before + 1),
        "{strengthened}"
    );
    let untouched = server.call("memory", json!({ "action": "get", "id": quiet }));
    assert_eq!(untouched["node"]["reps"], json!(reps_before), "{untouched}");
    server.shutdown();

    let output = Command::new(strata_verify_bin())
        .arg(&dir)
        .output()
        .expect("run strata-verify");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "strata-verify failed\n{stdout}\n{stderr}"
    );
    assert!(stdout.contains("OK"), "{stdout}");
    assert_no_sqlite(&dir);
    let _ = std::fs::remove_dir_all(&dir);
}
