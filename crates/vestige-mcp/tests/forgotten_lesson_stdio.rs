//! Real-stdio proof that `forgotten_lesson` completes on a Strata log.
//!
//! The binary is the shipped `vestige-mcp` process. The store is seeded
//! with recorded causal edges before the process starts; the tool is then
//! called over line-framed JSON-RPC. A `pending_strata` error is a failure.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::Duration;

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
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
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines() {
                if line.is_err() {
                    return;
                }
            }
        });
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
        writeln!(stdin, "{message}").expect("write");
        stdin.flush().expect("flush");
        let line = match self.stdout.recv_timeout(Duration::from_secs(90)) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => panic!("vestige-mcp hung on {method}"),
            Err(RecvTimeoutError::Disconnected) => panic!("vestige-mcp closed stdout on {method}"),
        };
        if line.contains("\"method\"") && !line.contains("\"id\"") {
            return self.request_continue(method, id);
        }
        let response: Value = serde_json::from_str(&line)
            .unwrap_or_else(|error| panic!("non-json from vestige-mcp: {line} ({error})"));
        if response.get("id").is_none() && response.get("method").is_some() {
            return self.request_continue(method, id);
        }
        assert_eq!(response["id"], json!(id), "{response}");
        response
    }

    fn request_continue(&mut self, method: &str, id: u64) -> Value {
        loop {
            let line = match self.stdout.recv_timeout(Duration::from_secs(90)) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => panic!("vestige-mcp hung on {method}"),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("vestige-mcp closed stdout on {method}")
                }
            };
            let response: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("non-json from vestige-mcp: {line} ({error})"));
            if response.get("id").is_none() {
                continue;
            }
            assert_eq!(response["id"], json!(id), "{response}");
            return response;
        }
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    assert!(status.success(), "stdio server exited {status}");
                    return;
                }
                Ok(None) if std::time::Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!("vestige-mcp did not exit after stdin closed");
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(error) => panic!("poll vestige-mcp: {error}"),
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

struct Graph {
    failure: String,
    low: String,
    high: String,
    mid: String,
    twin: String,
    bystander: String,
    later: String,
    other: String,
}

fn mem(content: &str, at: i64) -> strata_store::IngestInput {
    strata_store::IngestInput {
        content: content.to_string(),
        created_at_ms: Some(at),
        ..Default::default()
    }
}

fn link(src: &str, dst: &str, kind: strata_store::EdgeKind) -> strata_store::ConnectionRecord {
    strata_store::ConnectionRecord {
        source_id: src.to_string(),
        target_id: dst.to_string(),
        link_type: kind.as_str().to_string(),
        created_at_ms: 1,
        ..Default::default()
    }
}

fn kernel_r(store: &strata_store::StrataStore, id: &str) -> f64 {
    store
        .retrievability(id)
        .expect("retrievability")
        .expect("card")
}

fn seed(dir: &Path) -> Graph {
    let mut store = strata_store::StrataStore::open(dir).expect("open");
    let low = store
        .ingest_in_scope(mem("again-rated cause", 1_000), "user")
        .unwrap();
    let twin = store
        .ingest_in_scope(
            mem("fixed the API_TIMEOUT overflow in the same outage", 1_100),
            "user",
        )
        .unwrap();
    let other = store
        .ingest_in_scope(mem("other-scope cause", 1_200), "other")
        .unwrap();
    let bystander = store
        .ingest_in_scope(mem("file anchor only", 1_300), "user")
        .unwrap();
    let mid = store
        .ingest_in_scope(mem("intermediate cause", 2_000), "user")
        .unwrap();
    let high = store
        .ingest_in_scope(mem("easy-rated cause", 3_000), "user")
        .unwrap();
    let failure = store
        .ingest_in_scope(mem("outage after API_TIMEOUT change", 5_000), "user")
        .unwrap();
    let later = store
        .ingest_in_scope(mem("later note", 9_500), "user")
        .unwrap();
    store
        .save_connection(&link(&mid, &failure, strata_store::EdgeKind::DerivedFrom))
        .unwrap();
    store
        .save_connection(&link(&low, &mid, strata_store::EdgeKind::Corrects))
        .unwrap();
    store
        .save_connection(&link(&high, &failure, strata_store::EdgeKind::EvidenceOf))
        .unwrap();
    store
        .save_connection(&link(&bystander, &failure, strata_store::EdgeKind::Touched))
        .unwrap();
    store
        .save_connection(&link(&later, &failure, strata_store::EdgeKind::DerivedFrom))
        .unwrap();
    store
        .save_connection(&link(&other, &failure, strata_store::EdgeKind::DerivedFrom))
        .unwrap();
    store.review(&high, 4).unwrap();
    for step in 0..8 {
        store
            .ingest_in_scope(mem(&format!("pre-pad {step}"), 8_000 + step), "user")
            .unwrap();
    }
    store.review(&low, 1).unwrap();
    for step in 1..=80 {
        let low_r = kernel_r(&store, &low);
        let high_r = kernel_r(&store, &high);
        if low_r < high_r {
            break;
        }
        assert!(
            step < 80,
            "retrievability did not separate: low={low_r} high={high_r}"
        );
        store
            .ingest_in_scope(mem(&format!("pad {step}"), 9_000 + step), "user")
            .unwrap();
    }
    drop(store);
    Graph {
        failure,
        low,
        high,
        mid,
        twin,
        bystander,
        later,
        other,
    }
}

fn tool_body(response: &Value) -> Value {
    assert!(
        response.get("error").is_none(),
        "json-rpc error: {response}"
    );
    let result = &response["result"];
    assert_ne!(result["isError"], json!(true), "tool error: {result}");
    result["structuredContent"].clone()
}

#[test]
fn forgotten_lesson_completes_over_real_stdio() {
    let dir = tempfile::tempdir().expect("tempdir");
    let graph = seed(dir.path());
    let mut server = Server::spawn(dir.path());
    let hello = server.request(
        "initialize",
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "forgotten-lesson-stdio", "version": "1" },
        }),
    );
    assert!(hello.get("error").is_none(), "{hello}");
    let stdin = server.stdin.as_mut().expect("stdin");
    writeln!(
        stdin,
        "{}",
        json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();
    stdin.flush().unwrap();

    let first = tool_body(&server.request(
        "tools/call",
        json!({
            "name": "forgotten_lesson",
            "arguments": { "failure_id": graph.failure },
        }),
    ));
    let second = tool_body(&server.request(
        "tools/call",
        json!({
            "name": "forgotten_lesson",
            "arguments": { "failure_id": graph.failure },
        }),
    ));
    assert_eq!(first, second, "stdio result is deterministic");
    assert_eq!(first["tool"], "forgotten_lesson");
    assert!(!first.to_string().contains("pending_strata"), "{first}");
    assert!(first.get("error").is_none(), "{first}");
    assert!(first["count"].as_u64().unwrap() >= 2, "{first}");

    let lessons = first["forgotten_lessons"].as_array().expect("lessons");
    let ids: Vec<&str> = lessons
        .iter()
        .map(|entry| entry["lesson_id"].as_str().unwrap())
        .collect();
    for banned in [
        graph.twin.as_str(),
        graph.bystander.as_str(),
        graph.later.as_str(),
        graph.other.as_str(),
        graph.failure.as_str(),
    ] {
        assert!(!ids.contains(&banned), "{banned} leaked into {ids:?}");
    }
    let low_at = ids
        .iter()
        .position(|id| *id == graph.low)
        .expect("low lesson");
    let high_at = ids
        .iter()
        .position(|id| *id == graph.high)
        .expect("high lesson");
    assert!(
        low_at < high_at,
        "lowest FSRS retrievability first: {ids:?}"
    );
    let scores: Vec<f64> = lessons
        .iter()
        .map(|entry| entry["retrievability"].as_f64().unwrap())
        .collect();
    for window in scores.windows(2) {
        assert!(window[0] <= window[1], "{scores:?}");
    }
    let low = lessons
        .iter()
        .find(|entry| entry["lesson_id"] == graph.low)
        .unwrap();
    assert_eq!(low["edge_path"][0]["link_type"], "derived_from");
    assert_eq!(low["edge_path"][0]["source_id"], graph.mid);
    assert_eq!(low["edge_path"][0]["target_id"], graph.failure);
    assert_eq!(low["edge_path"][1]["link_type"], "corrects");
    assert_eq!(low["edge_path"][1]["source_id"], graph.low);
    let high = lessons
        .iter()
        .find(|entry| entry["lesson_id"] == graph.high)
        .unwrap();
    assert_eq!(high["edge_path"][0]["link_type"], "evidence_of");
    assert_eq!(high["edge_path"][0]["source_id"], graph.high);
    assert_eq!(high["edge_path"][0]["target_id"], graph.failure);

    server.shutdown();
}
