//! Real-stdio proof that `forgotten_lesson` completes on a Strata log.
//!
//! The binary is the shipped `vestige-mcp`. Edges are appended to that
//! process's log only after it has exited, then a second process reads them.
//! No SQLite file is created.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(60);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
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
            .env("VESTIGE_BACKFILL_AUTOFIRE", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        // Drain stderr so a chatty child cannot block on a full pipe.
        let raw_stderr = child.stderr.take().expect("stderr");
        std::thread::spawn(move || for _ in BufReader::new(raw_stderr).lines() {});
        Self {
            child,
            stdin: Some(stdin),
            stdout,
            next_id: 0,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{message}").expect("write rpc");
        stdin.flush().expect("flush rpc");
        loop {
            let line = match self.stdout.recv_timeout(RPC_TIMEOUT) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => panic!("{method} hung"),
                Err(RecvTimeoutError::Disconnected) => panic!("{method} crashed"),
            };
            let value: Value = serde_json::from_str(&line).expect("json");
            if value.get("id").is_none() {
                continue;
            }
            assert_eq!(value["id"], json!(id), "{value}");
            assert!(value.get("error").is_none(), "{method} error: {value}");
            return value["result"].clone();
        }
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "forgotten-lesson-stdio", "version": "1" },
            }),
        );
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        .expect("initialized");
        stdin.flush().expect("flush initialized");
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        );
        assert_ne!(result["isError"], json!(true), "{name} failed: {result}");
        result["structuredContent"].clone()
    }

    fn ingest(&mut self, content: &str) -> String {
        let value = self.call_tool(
            "smart_ingest",
            json!({ "content": content, "forceCreate": true }),
        );
        assert_eq!(value["success"], json!(true), "{value}");
        value["nodeId"]
            .as_str()
            .unwrap_or_else(|| panic!("no nodeId: {value}"))
            .to_string()
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll") {
                Some(status) => {
                    assert!(status.success(), "shutdown failed: {status}");
                    return;
                }
                None if Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!("vestige-mcp did not exit");
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

fn edge(source: &str, target: &str, link_type: &str) -> strata_store::ConnectionRecord {
    strata_store::ConnectionRecord {
        source_id: source.to_string(),
        target_id: target.to_string(),
        strength_milli: 1000,
        link_type: link_type.to_string(),
        meta_sha: None,
        created_at_ms: 0,
        activation_count: 0,
    }
}

#[test]
fn forgotten_lesson_completes_over_stdio_from_recorded_causal_edges() {
    let dir = tempfile::tempdir().expect("data dir");
    let (lesson, cause, failure, decoy) = {
        let mut server = Server::spawn(dir.path());
        server.handshake();
        let lesson = server.ingest("Recorded correction for the pool ceiling");
        let cause = server.ingest("Prior adjustment of the pool ceiling");
        let failure = server.ingest("Outage crash loop after the pool ceiling change");
        let decoy = server.ingest("Recorded correction for the pool ceiling, unlinked copy");
        server.shutdown();
        (lesson, cause, failure, decoy)
    };

    {
        let mut store = strata_store::StrataStore::open(dir.path()).expect("reopen log");
        store
            .save_connection(&edge(&cause, &failure, "derived_from"))
            .expect("cause edge");
        store
            .save_connection(&edge(&lesson, &cause, "evidence_of"))
            .expect("lesson edge");
        store
            .save_connection(&edge(&decoy, &failure, "anchored_to"))
            .expect("non-causal edge");
        // Easy reviews raise the older lesson's stability so the untouched
        // cause is the more forgotten card. Rank follows that score.
        for _ in 0..8 {
            store.review(&lesson, 4).expect("strengthen lesson");
        }
    }

    let mut server = Server::spawn(dir.path());
    server.handshake();
    let args = json!({ "failure_id": failure });
    let first = server.call_tool("forgotten_lesson", args.clone());
    let second = server.call_tool("forgotten_lesson", args);
    assert_eq!(first, second, "forgotten_lesson must be deterministic");

    let body = serde_json::to_string(&first).unwrap();
    assert!(
        !body.contains("pending_strata"),
        "completed result, not a strata deferral: {first}"
    );
    assert_eq!(first["tool"], json!("forgotten_lesson"));
    assert_eq!(first["failure_id"], json!(failure));
    assert!(first.get("error").is_none(), "{first}");

    let lessons = first["forgotten_lessons"]
        .as_array()
        .expect("forgotten_lessons");
    let ids: Vec<&str> = lessons
        .iter()
        .map(|entry| entry["lesson_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&lesson.as_str()), "{first}");
    assert!(ids.contains(&cause.as_str()), "{first}");
    assert!(
        !ids.contains(&decoy.as_str()),
        "anchored_to is not walked: {first}"
    );
    assert_eq!(
        ids[0],
        cause.as_str(),
        "lowest FSRS retrievability first: {first}"
    );

    let cause_entry = lessons
        .iter()
        .find(|entry| entry["lesson_id"] == cause)
        .unwrap();
    assert_eq!(cause_entry["edge_path"][0]["link_type"], "derived_from");
    assert_eq!(cause_entry["edge_path"][0]["source_id"], json!(cause));
    assert_eq!(cause_entry["edge_path"][0]["target_id"], json!(failure));

    let lesson_entry = lessons
        .iter()
        .find(|entry| entry["lesson_id"] == lesson)
        .unwrap();
    assert_eq!(
        lesson_entry["edge_path"].as_array().unwrap().len(),
        2,
        "{first}"
    );
    assert_eq!(lesson_entry["edge_path"][0]["link_type"], "evidence_of");
    assert_eq!(lesson_entry["edge_path"][1]["link_type"], "derived_from");

    let mut previous = f64::NEG_INFINITY;
    for entry in lessons {
        let retention = entry["retention_pct"].as_f64().unwrap();
        assert!(
            retention + 1e-9 >= previous,
            "not sorted lowest-first: {first}"
        );
        previous = retention;
        assert!(!entry["edge_path"].as_array().unwrap().is_empty());
    }

    server.shutdown();
}
