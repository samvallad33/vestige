//! Real Strata/MCP acceptance for personal Backfill. All data is disposable.
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
    fn save(&mut self, content: &str, entities: Value, attributes: Value) -> String {
        std::thread::sleep(Duration::from_millis(3));
        let saved = self.tool(
            "smart_ingest",
            json!({"content":content,"entities":entities,"attributes":attributes}),
        );
        saved
            .get("nodeId")
            .or_else(|| saved.get("echoNodeId"))
            .unwrap_or(&Value::Null)
            .as_str()
            .unwrap_or_else(|| panic!("missing node: {saved}"))
            .into()
    }
    fn get(&mut self, id: &str) -> Value {
        self.tool("memory", json!({"action":"get","id":id}))["node"].clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn snapshot(path: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &Path, path: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out)
            } else if path.is_file() {
                out.push((
                    path.strip_prefix(root).unwrap().to_string_lossy().into(),
                    std::fs::read(path).unwrap(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(path, path, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn maya_birthday_recall_surfaces_reaction_with_receipt_and_person_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let reaction = s.save(
        "Maya felt off after dinner at Bangkok Garden",
        json!(["person:Maya", "place:Bangkok Garden"]),
        json!([]),
    );
    s.tool("memory", json!({"action":"demote","id":reaction}));
    let before = s.get(&reaction)["stability"].as_f64().unwrap();
    let other = s.save(
        "Liam felt off after dinner at Bangkok Garden",
        json!(["person:Liam", "place:Bangkok Garden"]),
        json!([]),
    );
    let other_before = s.get(&other)["stability"].as_f64().unwrap();
    s.save(
        "The bicycle tire was repaired",
        json!(["service:bicycle"]),
        json!([]),
    );
    let allergy = s.save(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
    );
    let after = s.get(&reaction)["stability"].as_f64().unwrap();
    assert!(after > before && after <= before * 1.5 + 1e-8);
    assert_eq!(s.get(&other)["stability"].as_f64().unwrap(), other_before);
    // The agent resolves the birthday question's explicit subject to this handle.
    let recalled = s.tool("recall", json!({"handle":"person:Maya"}));
    assert_eq!(recalled["nodes"][0]["id"], reaction, "{recalled}");
    let receipts = s.tool("backfill", json!({"action":"receipts"}));
    let receipt = receipts["receipts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["memory_id"] == reaction && r["trigger_id"] == allergy)
        .unwrap();
    assert_eq!(receipt["shared_entity"], "person:Maya");
    assert!(
        receipt["reason"]
            .as_str()
            .unwrap()
            .contains("new health fact about Maya")
    );
    assert!(
        receipt["reason"]
            .as_str()
            .unwrap()
            .contains("bad reaction after Bangkok Garden relevant")
    );
    assert!(receipt["timestamp_ms"].as_i64().unwrap() > 0);
    let id = receipt["receipt_id"].as_str().unwrap().to_string();
    let proof = s.tool("receipt", json!({"action":"get","receipt_id":id}));
    assert_eq!(
        proof["attestation"]["verification"]["chainValid"], true,
        "{proof}"
    );
}

#[test]
fn repeated_content_with_explicit_metadata_keeps_personal_backfill_working() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let reaction = s.save(
        "Maya felt off after dinner at Bangkok Garden",
        json!(["person:Maya", "place:Bangkok Garden"]),
        json!([]),
    );
    s.tool("memory", json!({"action":"demote","id":reaction}));
    let before = s.get(&reaction)["stability"].as_f64().unwrap();
    s.save("Maya has a peanut allergy", json!([]), json!([]));
    s.save(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
    );
    assert!(
        s.get(&reaction)["stability"].as_f64().unwrap() > before,
        "explicit metadata must survive a prior content-only save"
    );
}

#[test]
fn autofire_off_manual_dry_run_exact_undo_idempotency_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "off");
    let target = s.save(
        "Maya felt off after dinner at Bangkok Garden",
        json!(["person:Maya", "place:Bangkok Garden"]),
        json!([]),
    );
    let before = s.get(&target)["stability"].as_f64().unwrap();
    let trigger = s.save(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
    );
    assert_eq!(s.get(&target)["stability"].as_f64().unwrap(), before);
    let frozen = snapshot(dir.path());
    let preview = s.tool(
        "remember",
        json!({"action":"backfill","params":{"trigger_id":trigger,"dry_run":true}}),
    );
    assert_eq!(preview["changed"], 0);
    assert_eq!(preview["would_change"], 1, "{preview}");
    assert_eq!(snapshot(dir.path()), frozen, "dry run wrote to the store");
    let applied = s.tool(
        "remember",
        json!({"action":"backfill","params":{"trigger_id":trigger,"manual":true}}),
    );
    assert_eq!(applied["changed"], 1, "{applied}");
    let receipt = applied["changes"][0].clone();
    assert_eq!(
        s.tool(
            "remember",
            json!({"action":"backfill","params":{"trigger_id":trigger,"manual":true}}),
        )["changed"],
        0
    );
    // Exercise the real CLI while this MCP owner holds the store lock.
    let cli = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(dir.path())
        .arg("backfill")
        .arg("undo")
        .arg(receipt["receipt_id"].as_str().unwrap())
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let undo: Value = serde_json::from_slice(&cli.stdout).unwrap();
    assert_eq!(undo["new_stability_q"], receipt["old_stability_q"]);
    assert_eq!(s.get(&target)["stability"].as_f64().unwrap(), before);
    drop(s);
    let mut reopened = Server::open(dir.path(), "off");
    assert_eq!(reopened.get(&target)["stability"].as_f64().unwrap(), before);
    assert_eq!(
        reopened.tool(
            "remember",
            json!({"action":"backfill","params":{"trigger_id":trigger,"manual":true}}),
        )["changed"],
        0
    );
}

#[test]
fn newer_explicit_preference_lowers_older_one() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let old = s.save(
        "Maya prefers quiet places",
        json!(["person:Maya"]),
        json!(["preference:ambience=quiet"]),
    );
    let before = s.get(&old)["stability"].as_f64().unwrap();
    s.save(
        "Maya now prefers lively places",
        json!(["person:Maya"]),
        json!(["preference:ambience=lively"]),
    );
    let after = s.get(&old)["stability"].as_f64().unwrap();
    assert!(after < before && after + 1e-8 >= before / 1.5);
}

#[test]
fn coding_failure_autofires_without_similarity_and_manual_never_updates_future() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let cause = s.save(
        "Changed the access policy in cache.rs",
        json!(["file:cache.rs"]),
        json!([]),
    );
    let before = s.get(&cause)["stability"].as_f64().unwrap();
    let trigger = s.save(
        "Failure: panic while starting the application",
        json!(["file:cache.rs"]),
        json!([]),
    );
    assert!(s.get(&cause)["stability"].as_f64().unwrap() > before);
    let future = s.save(
        "A later note about cache.rs",
        json!(["file:cache.rs"]),
        json!([]),
    );
    let future_before = s.get(&future)["stability"].as_f64().unwrap();
    s.tool("backfill", json!({"trigger_id":trigger,"manual":true}));
    assert_eq!(s.get(&future)["stability"].as_f64().unwrap(), future_before);
    let benign = s.save(
        "A release planning note",
        json!(["file:cache.rs"]),
        json!([]),
    );
    let now = s.get(&future)["stability"].as_f64().unwrap();
    let manual = s.tool("backfill", json!({"trigger_id":benign,"manual":true}));
    assert!(manual["changed"].as_u64().unwrap() > 0);
    assert!(s.get(&future)["stability"].as_f64().unwrap() > now);
}

#[test]
fn repeated_bounded_passes_finish_remaining_pairs_without_compounding() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "off");
    for index in 0..20 {
        s.save(
            &format!("Maya earlier event {index}"),
            json!(["person:Maya"]),
            json!([]),
        );
    }
    let trigger = s.save(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
    );
    assert_eq!(
        s.tool("backfill", json!({"trigger_id":trigger}))["changed"],
        16
    );
    assert_eq!(
        s.tool("backfill", json!({"trigger_id":trigger}))["changed"],
        4
    );
    assert_eq!(
        s.tool("backfill", json!({"trigger_id":trigger}))["changed"],
        0
    );
    assert_eq!(
        s.tool("backfill", json!({"action":"receipts"}))["receipts"]
            .as_array()
            .unwrap()
            .len(),
        20
    );
}
