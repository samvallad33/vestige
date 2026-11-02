//! Six-tool wire surface, real inline receipts, stability, and hidden legacy compatibility.
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

struct Server {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    seq: u64,
}
impl Server {
    fn open(path: &Path, autofire: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .args(["--no-http", "--data-dir"])
            .arg(path)
            .env("VESTIGE_BACKFILL_AUTOFIRE", autofire)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE_ENABLED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut server = Self {
            child,
            input,
            output,
            seq: 0,
        };
        server.rpc("initialize",json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"rsb-acceptance","version":"1"}}));
        server
    }
    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.seq += 1;
        writeln!(
            self.input,
            "{}",
            json!({"jsonrpc":"2.0","id":self.seq,"method":method,"params":params})
        )
        .unwrap();
        self.input.flush().unwrap();
        loop {
            let mut line = String::new();
            assert!(
                self.output.read_line(&mut line).unwrap() > 0,
                "server closed"
            );
            let response: Value = serde_json::from_str(&line).unwrap();
            if response["id"] == self.seq {
                assert!(response.get("error").is_none(), "{response}");
                return response["result"].clone();
            }
        }
    }
    fn tool(&mut self, name: &str, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({"name":name,"arguments":args}));
        assert_ne!(result["isError"], true, "{name}: {result}");
        result.get("structuredContent").cloned().unwrap_or_else(|| {
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn six_tool_surface_inline_receipts_stability_and_legacy_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let catalog = s.rpc("tools/list", json!({}));
    let names: Vec<_> = catalog["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "causal_walk",
            "curate",
            "precedent",
            "recall",
            "remember",
            "session"
        ]
    );
    let guide = s.tool(
        "session",
        json!({"action":"status", "view":"tools", "tool":"remember"}),
    );
    assert_eq!(guide["tools"].as_array().unwrap().len(), 1);
    let guidance = guide["guidance"].as_str().unwrap();
    assert!(guidance.contains("session(action='status', view='tools'"));
    assert!(!guidance.contains("memory_status"));
    assert!(guide["tools"][0]["inputSchema"]["properties"]["content"].is_object());
    let first = s.tool("remember", json!({"content":"Tokyo transit uses a Suica card", "tags":["symbol:lme:tokyo"], "outputProfile":"compact"}));
    let id = first["nodeId"].as_str().unwrap();
    let before =
        s.tool("recall", json!({"id":id, "detail_level":"brief"}))["nodes"][0]["stability"]
            .as_f64()
            .unwrap();
    std::thread::sleep(Duration::from_millis(3));
    let second = s.tool("remember", json!({"content":"Tokyo transit directions updated", "tags":["symbol:lme:tokyo"], "outputProfile":"compact"}));
    assert_eq!(second["backfill_errors"], json!([]));
    let rows = second["backfill_receipts"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{second}");
    assert_eq!(rows[0]["memory_id"], id);
    assert_eq!(rows[0]["old_score"].as_f64().unwrap(), before);
    let after = s.tool(
        "recall",
        json!({"handle":"symbol:lme:tokyo", "detail_level":"brief"}),
    );
    let updated = after["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == id)
        .unwrap();
    assert_eq!(updated["stability"], rows[0]["new_score"]);
    assert!(updated["stability"].as_f64().unwrap() > before);
    assert!(updated.get("content").is_none());
    let walk = s.tool("causal_walk", json!({"action":"walk", "start_points":[{"kind":"logged_write","node_id":second["nodeId"]}], "promote":false}));
    assert!(walk.get("causes").is_some());
    let receipt = s.tool(
        "precedent",
        json!({"action":"receipt", "receipt_id":rows[0]["receipt_id"]}),
    );
    assert!(receipt.is_object());
    s.tool("curate", json!({"action":"promote", "id":id}));
    // Legacy names stay hidden, but retain their original argument contracts.
    let legacy = s.tool(
        "smart_ingest",
        json!({"content":"Legacy compatibility fixture", "tags":["compat-fixture"]}),
    );
    let got = s.tool("memory", json!({"action":"get", "id":legacy["nodeId"]}));
    assert_eq!(got["node"]["content"], "Legacy compatibility fixture");
    let receipts = s.tool("backfill", json!({"action":"receipts"}));
    assert_eq!(receipts["receipts"].as_array().unwrap().len(), 1);
    let status = s.tool("session", json!({"action":"status"}));
    assert_eq!(status["runtime"]["pid"], s.child.id());
    assert_eq!(status["runtime"]["legacy_aliases"]["total"], 3);
    for name in ["smart_ingest", "memory", "backfill"] {
        assert_eq!(status["runtime"]["legacy_aliases"]["by_tool"][name], 1);
    }
    let audit = std::fs::read_to_string(dir.path().join("legacy-tool-calls.jsonl")).unwrap();
    assert!(!audit.contains("Legacy compatibility fixture"));
    assert!(!audit.contains("arguments"));
    for line in audit.lines() {
        let receipt: Value = serde_json::from_str(line).unwrap();
        assert!(
            receipt["receipt_id"]
                .as_str()
                .unwrap()
                .starts_with("legacy-")
        );
        assert_eq!(receipt["signed"], false);
    }
}

#[test]
fn alias_tripwire_survives_restart_and_canonical_reflex_is_not_legacy() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut s = Server::open(dir.path(), "off");
        s.tool(
            "session",
            json!({"action":"reflex","params":{"action":"status"}}),
        );
        let status = s.tool("session", json!({"action":"status"}));
        assert_eq!(status["runtime"]["legacy_aliases"]["total"], 0);
        s.tool("reflex", json!({"action":"status"}));
        let status = s.tool("session", json!({"action":"status"}));
        assert_eq!(status["runtime"]["legacy_aliases"]["by_tool"]["reflex"], 1);
    }
    let mut s = Server::open(dir.path(), "off");
    let status = s.tool("session", json!({"action":"status"}));
    assert_eq!(status["runtime"]["legacy_aliases"]["total"], 1);
    assert!(status["runtime"]["legacy_aliases"]["read_error"].is_null());
}

#[test]
fn oversized_trigger_skips_backfill_without_dropping_retrieval_handles() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let mut tags: Vec<String> = (0..65).map(|i| format!("symbol:lme:topic{i}")).collect();
    tags.push("session:opaque".into());
    let saved = s.tool(
        "remember",
        json!({"content":"Whole dialogue round preserved", "tags":tags}),
    );
    assert_eq!(saved["backfill_skipped"], 1);
    assert_eq!(saved["backfill_receipts"], json!([]));
    let got = s.tool(
        "recall",
        json!({"handle":"symbol:lme:topic64", "detail_level":"full"}),
    );
    assert_eq!(got["nodes"][0]["content"], "Whole dialogue round preserved");
    assert_eq!(got["nodes"][0]["tags"].as_array().unwrap().len(), 66);
}
