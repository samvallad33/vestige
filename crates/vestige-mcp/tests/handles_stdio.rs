//! Falsifier for #408: file and symbol handles resolve from anchors, not content.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "0")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        }
    }

    fn send(&mut self, message: &Value) {
        let stdin = self.stdin.as_mut().expect("stdin");
        stdin
            .write_all(message.to_string().as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .expect("write stdio");
    }

    fn read(&mut self) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if Instant::now() > deadline {
                panic!("stdio timeout");
            }
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("read stdio");
            if line.is_empty() {
                panic!("vestige-mcp closed stdout");
            }
            let value: Value = serde_json::from_str(&line).expect("json");
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            return value;
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        let response = self.read();
        assert_eq!(response["id"], json!(id));
        assert!(response.get("error").is_none(), "{method}: {response}");
        response["result"].clone()
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "handles-stdio", "version": "1"}
            }),
        );
        self.send(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        assert_ne!(result["isError"], json!(true), "{name}: {result}");
        if let Some(structured) = result.get("structuredContent") {
            return structured.clone();
        }
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `recall` of a path and of `path#symbol` returns the anchored memory, with
/// the anchor's `eff-` receipt, and never the memory that only mentions the path.
#[test]
fn file_and_symbol_handles_resolve_from_anchors_not_content() {
    let data = tempfile::tempdir().unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let file = workspace.path().join("src/a.rs");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "fn parse() {\n    let _ = 1;\n}\n").unwrap();

    let mut server = Server::spawn(data.path());
    server.handshake();
    let saved = server.call(
        "codebase",
        json!({
            "action": "remember_pattern",
            "name": "Parse",
            "description": "A function named parse.",
            "files": ["src/a.rs#parse"],
            "repoPath": workspace.path()
        }),
    );
    let pattern = saved["nodeId"].as_str().unwrap().to_string();
    assert!(
        saved["anchors"]["verifiable"].as_u64().unwrap_or(0) >= 1,
        "the symbol must be captured from the file: {saved}"
    );
    let mention = server.call(
        "smart_ingest",
        json!({
            "content": "notes mention src/a.rs and parse but record no anchor",
            "forceCreate": true
        }),
    );
    let mentioned = mention["nodeId"].as_str().unwrap().to_string();
    assert_ne!(pattern, mentioned);

    let empty = server.call("codebase", json!({"action": "record_runs"}));
    assert_eq!(empty["recorded"], 0, "{empty}");

    let mut anchor_receipt = None;
    for (handle, kind) in [("src/a.rs", "file"), ("src/a.rs#parse", "symbol")] {
        let resolved = server.call("recall", json!({"handle": handle}));
        assert_eq!(resolved["kind"], kind, "{handle}: {resolved}");
        assert_eq!(resolved["exact"], true, "{resolved}");
        let ids: Vec<_> = resolved["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| node["id"].as_str())
            .collect();
        assert_eq!(ids, vec![pattern.as_str()], "{handle}: {resolved}");
        assert!(
            !resolved.to_string().contains(&mentioned),
            "{handle} returned the content-only memory: {resolved}"
        );
        let proof = &resolved["proofs"][0];
        assert_eq!(proof["table"], "anchor", "{proof}");
        assert_eq!(proof["id"], pattern, "{proof}");
        let receipt = proof["receipt"].as_str().unwrap().to_string();
        assert!(receipt.starts_with("eff-"), "{proof}");
        let anchor_id = proof["anchor_id"].as_str().unwrap().to_string();
        anchor_receipt = Some((anchor_id, receipt));
    }
    server.shutdown();
    let (anchor_id, receipt) = anchor_receipt.unwrap();
    let store = strata_store::StrataStore::open(data.path()).unwrap();
    let proof = store.latest_effect(&anchor_id).unwrap().unwrap();
    assert_eq!(strata_store::effect_receipt_id(proof.effect_seq), receipt);
}
