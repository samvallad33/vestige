//! Real stdio: a capability this build lacks says so. It never returns an
//! empty-looking success, and a success never carries a field that reads as a
//! failure. Covers `maintain consolidate`, `dedup scan`, `maintain dream`,
//! `maintain dream_compile` and `smart_ingest` on the real binary.

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
        let binary = PathBuf::from(env!("CARGO_BIN_EXE_vestige-mcp"));
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
            if value.get("id") == Some(&json!(id)) {
                assert!(value.get("error").is_none(), "{value}");
                return value["result"].clone();
            }
        }
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "unavailable-stdio", "version": "1" },
            })),
        );
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{note}")
            .and_then(|()| stdin.flush())
            .expect("notify");
    }

    /// The whole tools/call result, errors included.
    fn call_raw(&mut self, name: &str, arguments: Value) -> Value {
        self.request(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        )
    }

    /// The structured result of a call that must succeed.
    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.call_raw(name, arguments);
        assert_ne!(result["isError"], true, "{name}: {result}");
        result["structuredContent"].clone()
    }

    fn list_tools(&mut self) -> Vec<Value> {
        self.request("tools/list", None)["tools"]
            .as_array()
            .expect("tools")
            .clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vestige-unavailable-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn seed(server: &mut Server, count: usize) {
    for i in 0..count {
        server.call_tool(
            "smart_ingest",
            json!({"content": format!("probe memory {i}"), "node_type": "fact", "tags": ["probe"]}),
        );
    }
}

fn error_text(result: &Value) -> String {
    result["structuredContent"]["error"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| result.to_string())
}

#[test]
fn consolidate_is_withheld_not_a_zero_success() {
    let dir = temp_dir("consolidate");
    let mut server = Server::spawn(&dir);
    server.handshake();
    seed(&mut server, 4);

    // not advertised, and the description says so
    let tools = server.list_tools();
    let maintain = tools.iter().find(|t| t["name"] == "maintain").unwrap();
    let actions: Vec<&str> = maintain["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    assert!(!actions.contains(&"consolidate"), "{actions:?}");
    assert!(actions.contains(&"dream"), "{actions:?}");
    let description = maintain["description"].as_str().unwrap();
    assert!(
        description.contains("Withheld on Strata in 4.0:") && description.contains("consolidate"),
        "{description}"
    );

    // called anyway: refused with the stable code, and nothing that looks like a count
    let refused = server.call_raw("maintain", json!({"action": "consolidate"}));
    assert_eq!(refused["isError"], true, "{refused}");
    let text = error_text(&refused);
    assert!(text.starts_with("unavailable_in_4_0:"), "{text}");
    assert!(text.contains("maintain action 'consolidate'"), "{text}");
    assert!(
        text.contains("maintain action='dream'"),
        "names what works: {text}"
    );
    assert!(!refused.to_string().contains("nodesProcessed"), "{refused}");

    // the hidden pre-2.2 alias carries no `action`; it must say the same thing
    let alias = server.call_raw("consolidate", json!({}));
    assert_eq!(alias["isError"], true, "{alias}");
    assert_eq!(error_text(&alias), text, "alias and tool agree");
    assert!(!alias.to_string().contains("nodesProcessed"), "{alias}");

    // a bad phase is still an argument error, not hidden behind the refusal
    let bad_phase = server.call_raw("consolidate", json!({"phase": "embeddings"}));
    assert_eq!(bad_phase["isError"], true);
    assert!(
        error_text(&bad_phase).contains("phase must be"),
        "{bad_phase}"
    );

    println!("{}", server.transcript.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dedup_scan_keeps_the_real_report_and_marks_the_gated_one_unavailable() {
    let dir = temp_dir("dedup");
    let mut server = Server::spawn(&dir);
    server.handshake();
    // two byte-identical memories: the exact-equality cluster is real work
    for _ in 0..2 {
        server.call_tool(
            "smart_ingest",
            json!({"content": "the exact same sentence", "node_type": "fact", "forceCreate": true}),
        );
    }
    seed(&mut server, 2);

    let scan = server.call_tool("dedup", json!({"action": "scan"}));
    let merge = &scan["mergeCandidates"];
    assert_eq!(merge["status"], "unavailable", "{scan}");
    assert_eq!(merge["reason"], "embeddings_unavailable");
    let merge_keys: Vec<&str> = merge
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert!(
        !merge_keys.contains(&"candidates") && !merge_keys.contains(&"error"),
        "an empty candidates list reads as 'looked, found none': {merge}"
    );

    // the part of the scan that works still works
    let clusters = &scan["duplicateClusters"];
    assert!(
        clusters["totalClusters"].as_u64().unwrap() >= 1,
        "exact duplicates are still found: {clusters}"
    );

    // and the hint no longer sends the caller to actions this build refuses
    let next = scan["nextStep"].as_str().unwrap();
    assert!(!next.contains("Use action='plan_merge'"), "{next}");
    assert!(next.contains("duplicateClusters"), "{next}");
    println!("{}", server.transcript.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dream_and_dream_compile_say_what_they_cannot_discover() {
    let dir = temp_dir("dream");
    let mut server = Server::spawn(&dir);
    server.handshake();
    seed(&mut server, 9);

    // no recorded edges: the zeros are explained, and the dashboard's
    // numeric field is still a number
    let dream = server.call_tool("maintain", json!({"action": "dream"}));
    assert_eq!(dream["status"], "completed");
    assert_eq!(dream["discovery"]["status"], "unavailable", "{dream}");
    assert_eq!(dream["discovery"]["reason"], "embeddings_unavailable");
    assert!(dream["connectionsFound"].is_number(), "{dream}");
    assert_eq!(dream["edgesConsidered"], 0);
    assert!(
        dream["emptyBecause"]
            .as_str()
            .unwrap()
            .contains("no recorded edge"),
        "{dream}"
    );

    let compile = server.call_tool("maintain", json!({"action": "dream_compile"}));
    assert_eq!(compile["status"], "compiled");
    assert_eq!(compile["discovery"]["status"], "unavailable", "{compile}");
    assert!(compile["contradictionsFound"].is_number(), "{compile}");
    assert!(
        compile["emptyBecause"]
            .as_str()
            .unwrap()
            .contains("no recorded edge"),
        "{compile}"
    );
    println!("{}", server.transcript.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dream_with_recorded_edges_does_not_claim_emptiness() {
    let dir = temp_dir("dream-edges");
    // 4 memories joined by 3 recorded edges, written before the server starts
    vestige_mcp::tools::causal_walk::seed_recorded_cause(&dir).expect("seed strata log");
    let mut server = Server::spawn(&dir);
    server.handshake();
    seed(&mut server, 3);

    let dream = server.call_tool("maintain", json!({"action": "dream"}));
    assert_eq!(dream["status"], "completed", "{dream}");
    assert!(
        dream["edgesConsidered"].as_u64().unwrap() >= 1,
        "recorded edges were replayed: {dream}"
    );
    assert!(
        dream["emptyBecause"].is_null(),
        "it did real work, so it must not claim there was nothing: {dream}"
    );
    // discovery is still unavailable: replaying recorded edges is not discovering new ones
    assert_eq!(dream["discovery"]["status"], "unavailable");

    let compile = server.call_tool("maintain", json!({"action": "dream_compile"}));
    assert!(compile["emptyBecause"].is_null(), "{compile}");
    assert_eq!(compile["discovery"]["status"], "unavailable");
    println!("{}", server.transcript.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Collects every object key named `error` anywhere in `value`.
fn error_keys(value: &Value, path: &str, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let here = format!("{path}/{key}");
                if key == "error" {
                    out.push(here.clone());
                }
                error_keys(child, &here, out);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                error_keys(child, &format!("{path}/{index}"), out);
            }
        }
        _ => {}
    }
}

#[test]
fn a_successful_save_carries_no_error_shaped_field() {
    let dir = temp_dir("save-parts");
    let mut server = Server::spawn(&dir);
    server.handshake();

    // 4.1.0 answered every save with `success: true` beside
    // `synapticCapture.error: "...not committed..."` and
    // `tagSuggestionStatus.reason: "Initialization error: similarity_disabled..."`.
    let saved = server.call_tool(
        "smart_ingest",
        json!({"content": "The checkout worker timeout is 8 seconds.", "node_type": "fact", "tags": ["checkout"]}),
    );
    assert_eq!(saved["success"], true, "{saved}");
    assert!(saved["nodeId"].as_str().is_some(), "{saved}");

    let mut errors = Vec::new();
    error_keys(&saved, "", &mut errors);
    assert!(errors.is_empty(), "error keys {errors:?} in {saved}");
    let text = saved.to_string();
    assert!(!text.contains("Initialization error"), "{text}");
    assert!(!text.contains("not committed"), "{text}");

    // What the store does not do is still said, as a typed part.
    let tags = &saved["tagSuggestionStatus"];
    assert_eq!(tags["status"], "unavailable", "{saved}");
    assert_eq!(tags["reason"], "similarity_disabled", "{saved}");
    assert_eq!(tags["scope"], "user", "{saved}");
    let capture = &saved["synapticCapture"];
    if !capture.is_null() {
        assert_eq!(capture["status"], "unavailable", "{saved}");
        assert_eq!(capture["reason"], "synaptic_capture_unavailable", "{saved}");
        assert_eq!(capture["durable"], false, "{saved}");
    }

    // The batch path builds its items separately: same rule.
    let batch = server.call_tool(
        "smart_ingest",
        json!({"items": [
            {"content": "Batch item one.", "tags": ["checkout"]},
            {"content": "Batch item two.", "tags": ["checkout"]}
        ]}),
    );
    assert_eq!(batch["success"], true, "{batch}");
    let mut batch_errors = Vec::new();
    error_keys(&batch, "", &mut batch_errors);
    assert!(
        batch_errors.is_empty(),
        "error keys {batch_errors:?} in {batch}"
    );
    assert!(
        !batch.to_string().contains("Initialization error"),
        "{batch}"
    );
}

#[test]
fn a_demote_explains_why_retrievability_reads_one() {
    let dir = temp_dir("review-shape");
    let mut server = Server::spawn(&dir);
    server.handshake();
    let saved = server.call_tool(
        "smart_ingest",
        json!({"content": "A memory that turns out to be wrong.", "tags": ["review-shape"]}),
    );
    let id = saved["nodeId"].as_str().unwrap().to_string();

    // A demote is an FSRS review rated again, and retrievability reads 1.0
    // right after any review. 4.1.0 showed that 1.0 beside a note saying the
    // memory "ranks lower", with nothing to connect the two.
    let demoted = server.call_tool("memory", json!({"action": "demote", "id": id}));
    assert_eq!(demoted["success"], true, "{demoted}");
    let text = demoted.to_string();
    for stale in ["\"-0.30\"", "\"-0.15\"", "\"0.5x\"", "ranks lower"] {
        assert!(
            !text.contains(stale),
            "{stale} describes the v3 engine: {text}"
        );
    }
    assert!(
        demoted["message"]
            .as_str()
            .unwrap()
            .starts_with("Memory demoted under FSRS-6 (Again)."),
        "{demoted}"
    );
    assert_eq!(demoted["changes"]["lapses"]["after"], 1, "{demoted}");
    let before = demoted["changes"]["stability"]["before"].as_f64().unwrap();
    let after = demoted["changes"]["stability"]["after"].as_f64().unwrap();
    assert!(after < before, "a demote lowers stability: {demoted}");
    let note = demoted["note"].as_str().unwrap();
    assert!(note.contains("not deleted"), "{note}");
    assert!(
        note.contains("Retrievability reads 1.0 right after any review"),
        "{note}"
    );
}
