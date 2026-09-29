//! Real-stdio completion for `receipt` replay and `project` on the default
//! Strata binary. The child is `vestige-mcp` with default features. No SQLite
//! file is created. Projection selection does not scan content; the write
//! records `projected_to` edges and nothing else.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct StdioServer {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    next_id: u64,
    transcript: Vec<Value>,
}

impl StdioServer {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .arg("--no-http")
            .arg("--data-dir")
            .arg(data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout");
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
        Self {
            child,
            stdin,
            lines,
            next_id: 0,
            transcript: Vec::new(),
        }
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&request).unwrap();
        line.push('\n');
        self.stdin
            .as_mut()
            .expect("stdin open")
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.as_mut().unwrap().flush())
            .expect("write stdin");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("timed out waiting for {method}");
            }
            let raw = self
                .lines
                .recv_timeout(remaining)
                .unwrap_or_else(|error| panic!("stdio closed during {method}: {error}"));
            let response: Value = serde_json::from_str(&raw).expect("json-rpc line");
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            self.transcript.push(json!({
                "request": request,
                "response": response,
            }));
            assert!(
                response.get("error").is_none(),
                "rpc error for {method}: {response}"
            );
            return response["result"].clone();
        }
    }

    fn tool(&mut self, name: &str, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({ "name": name, "arguments": args }));
        assert_eq!(
            result.get("isError").and_then(Value::as_bool),
            Some(false),
            "{name} failed: {result}"
        );
        let body = result
            .get("structuredContent")
            .cloned()
            .expect("structuredContent");
        let text = serde_json::to_string(&body).unwrap();
        assert!(
            !text.contains("pending_strata"),
            "{name} returned pending_strata: {body}"
        );
        body
    }

    fn tool_err(&mut self, name: &str, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({ "name": name, "arguments": args }));
        assert_eq!(
            result.get("isError").and_then(Value::as_bool),
            Some(true),
            "{name} was expected to fail: {result}"
        );
        result
            .get("structuredContent")
            .cloned()
            .expect("structuredContent")
    }

    fn handshake(&mut self) {
        self.rpc(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "strata-stdio-receipt-project", "version": "1"},
            }),
        );
        let note = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
        self.stdin
            .as_mut()
            .unwrap()
            .write_all(note.as_bytes())
            .and_then(|()| self.stdin.as_mut().unwrap().flush())
            .unwrap();
    }

    fn shutdown(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for StdioServer {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn canonical(value: &Value) -> String {
    serde_json::to_string(value).unwrap()
}

fn assert_same(label: &str, left: &Value, right: &Value) {
    assert_eq!(
        canonical(left),
        canonical(right),
        "{label} is not deterministic"
    );
}

fn assert_no_sqlite(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            let sqlite = name.ends_with(".sqlite")
                || name.ends_with(".sqlite3")
                || name.ends_with(".db")
                || name.ends_with(".db-wal")
                || name.ends_with(".db-shm");
            assert!(!sqlite, "sqlite file appeared at {}", path.display());
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
}

fn ingest(server: &mut StdioServer, content: &str, node_type: &str, tags: &[&str]) -> String {
    let created = server.tool(
        "smart_ingest",
        json!({
            "content": content,
            "node_type": node_type,
            "tags": tags,
            "forceCreate": true,
        }),
    );
    assert_eq!(created["success"], true, "{created}");
    created["nodeId"].as_str().expect("nodeId").to_string()
}

#[test]
fn receipt_replay_and_project_complete_over_real_stdio() {
    let data = tempfile::tempdir().expect("data dir");
    let root = tempfile::tempdir().expect("projection root");
    let mut server = StdioServer::spawn(data.path());
    server.handshake();

    let empty = server.tool("project", json!({}));
    let empty_preview = server.tool("project", json!({"action": "preview"}));
    let empty_again = server.tool("project", json!({}));
    assert_eq!(empty["action"], "preview");
    assert_eq!(empty["scope"], "user");
    assert_eq!(empty["itemCount"], 0);
    assert_same("project {}", &empty, &empty_preview);
    assert_same("project {} repeat", &empty, &empty_again);

    let missing = server.tool_err(
        "receipt",
        json!({"action": "replay", "receipt_id": "no-such-receipt"}),
    );
    let missing_text = canonical(&missing);
    assert!(missing_text.contains("not found"), "{missing}");
    assert!(!missing_text.contains("pending_strata"), "{missing}");

    let decision = ingest(
        &mut server,
        "Ship releases from an integration branch",
        "decision",
        &[],
    );
    let plain = ingest(&mut server, "The office moved in spring", "fact", &[]);
    let keyword = ingest(
        &mut server,
        "A fact that mentions a decision and a preference rule",
        "fact",
        &["notes"],
    );
    let preference = ingest(
        &mut server,
        "Prefer tabs in Svelte files",
        "fact",
        &["preference"],
    );

    let replay_args = json!({
        "action": "replay",
        "receipt_id": decision,
        "withheld_slots": [],
    });
    let replay = server.tool("receipt", replay_args.clone());
    let replay_again = server.tool("receipt", replay_args);
    assert_eq!(replay["kind"], "strata");
    assert_eq!(replay["readOnly"], true);
    assert_eq!(replay["matched"], true);
    assert_eq!(replay["mismatches"], json!([]));
    assert_eq!(replay["nodeId"], decision);
    assert_eq!(replay["stateDigest"], replay["replayedDigest"]);
    assert!(replay["frames"].as_u64().unwrap_or(0) > 0);
    assert_same("receipt replay", &replay, &replay_again);

    let got = server.tool("receipt", json!({"action": "get", "receipt_id": decision}));
    let receipt_id = got["receipt"]["receipt_id"]
        .as_str()
        .expect("receipt_id")
        .to_string();
    assert!(receipt_id.starts_with("eff-"), "{receipt_id}");
    let by_receipt = server.tool(
        "receipt",
        json!({"action": "replay", "receipt_id": receipt_id}),
    );
    assert_eq!(by_receipt["matched"], true);
    assert_eq!(by_receipt["receiptId"], receipt_id);
    assert_eq!(by_receipt["stateDigest"], replay["stateDigest"]);

    let preview = server.tool("project", json!({"action": "preview"}));
    let defaults = server.tool("project", json!({}));
    let preview_again = server.tool("project", json!({"action": "preview"}));
    assert_same("project preview", &preview, &defaults);
    assert_same("project preview repeat", &preview, &preview_again);
    assert_eq!(preview["action"], "preview");
    assert_eq!(preview["itemCount"], 2, "{preview}");
    let ids: Vec<&str> = preview["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec![decision.as_str(), preference.as_str()], "{ids:?}");
    let region = preview["region"].as_str().unwrap();
    assert!(region.contains(&decision), "{region}");
    assert!(region.contains(&preference), "{region}");
    assert!(!region.contains(&plain), "{region}");
    assert!(!region.contains(&keyword), "{region}");
    assert!(!region.contains("office moved"), "{region}");
    assert!(!region.contains("mentions a decision"), "{region}");

    let root_canon = root.path().canonicalize().expect("projection root");
    let target = root_canon.join("CLAUDE.md");
    let write_args = json!({
        "action": "write",
        "path": "CLAUDE.md",
        "root": root_canon,
        "confirm": true,
    });
    let written = server.tool("project", write_args.clone());
    assert_eq!(written["action"], "write");
    assert_eq!(written["written"], true, "{written}");
    assert!(written.get("refused").is_none() || written["refused"].is_null());
    let hash = written["receipt"]["hash"].as_str().unwrap().to_string();
    let write_receipt = written["receipt"]["receiptId"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(write_receipt.starts_with("eff-"), "{write_receipt}");
    let bytes = std::fs::read(&target).unwrap();
    assert_eq!(hash, blake3::hash(&bytes).to_hex().as_str());
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(
        text.contains(&decision) && text.contains(&preference),
        "{text}"
    );
    assert!(!text.contains("office moved"), "{text}");

    let after = server.tool("project", json!({}));
    assert_eq!(after["region"], preview["region"]);
    assert_eq!(after["itemCount"], 2);

    let replay_after = server.tool(
        "receipt",
        json!({"action": "replay", "receipt_id": decision}),
    );
    let replay_after_again = server.tool(
        "receipt",
        json!({"action": "replay", "receipt_id": decision}),
    );
    assert_eq!(replay_after["matched"], true, "{replay_after}");
    assert_same("replay after write", &replay_after, &replay_after_again);

    let second = server.tool("project", write_args);
    assert_eq!(second["written"], false, "{second}");
    assert_eq!(second["receipt"]["hash"], hash);
    assert_eq!(std::fs::read(&target).unwrap(), bytes);

    if let Ok(path) = std::env::var("VESTIGE_STDIO_TRANSCRIPT") {
        let body = serde_json::to_string_pretty(&server.transcript).unwrap();
        std::fs::write(path, body).unwrap();
    }
    server.shutdown();
    drop(server);

    assert_no_sqlite(data.path());
    // The running server leaves the log unsealed. The checkpoint frame is the
    // existing KIND_STORE_CHECKPOINT seal, the same one the handler test uses
    // before strata-verify. It is not a new op kind.
    {
        let mut store = strata_store::StrataStore::open(data.path()).unwrap();
        store.seal_checkpoint().unwrap();
    }
    let report = strata_verify::verify_path(data.path());
    assert!(report.ok, "strata-verify failed: {}", report.json);

    let store = strata_store::StrataStore::open(data.path()).unwrap();
    let edges: Vec<_> = store
        .edges()
        .into_iter()
        .filter(|edge| edge.link_type == "projected_to")
        .collect();
    let sources: Vec<&str> = edges.iter().map(|edge| edge.source_id.as_str()).collect();
    assert!(sources.contains(&decision.as_str()), "{sources:?}");
    assert!(sources.contains(&preference.as_str()), "{sources:?}");
    assert!(!sources.contains(&plain.as_str()), "{sources:?}");
    assert!(!sources.contains(&keyword.as_str()), "{sources:?}");
    assert!(
        edges
            .iter()
            .all(|edge| edge.target_id == target.display().to_string())
    );
    assert!(
        edges
            .iter()
            .all(|edge| edge.meta_sha.as_deref() == Some(hash.as_str()))
    );
    let frames = store.log().read_frames(1).unwrap();
    assert!(!frames.is_empty());
    for frame in &frames {
        let known = (1..=7).contains(&frame.kind)
            || frame.kind == strata_store::KIND_STORE_WRITE
            || frame.kind == strata_store::KIND_STORE_CHECKPOINT;
        assert!(known, "unexpected frame kind {}", frame.kind);
    }
}
