//! Real stdio coverage for the six-tool local hook envelopes. All stores are disposable.
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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
            .env("OPERATOR_HOME", path.join("operator"))
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
    fn tool_error(&mut self, name: &str, args: Value) -> String {
        let result = self.rpc("tools/call", json!({"name":name,"arguments":args}));
        assert_eq!(result["isError"], true, "{name}: {result}");
        result["content"][0]["text"].as_str().unwrap().to_string()
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
fn catalog_and_full_schemas_cover_hook_envelopes() {
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
    assert!(
        catalog.to_string().len() < 22 * 1024,
        "catalog must stay bounded"
    );
    for (tool, actions) in [
        ("session", vec!["reflex", "intention", "codebase"]),
        ("remember", vec!["backfill"]),
        (
            "curate",
            vec!["memory", "graph", "ghostlink", "dream", "maintain"],
        ),
    ] {
        let guide = s.tool(
            "session",
            json!({"action":"status", "view":"tools", "tool":tool}),
        );
        let schema = &guide["tools"][0]["inputSchema"];
        assert_eq!(schema["properties"]["params"]["type"], "object");
        for action in actions {
            assert!(
                schema["properties"]["action"]["enum"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(action))
            );
            assert!(schema["allOf"].as_array().unwrap().iter().any(
                |branch| branch["if"]["properties"]["action"]["const"] == action
                    && branch["then"]["properties"]["params"].is_object()
            ));
        }
        if tool == "remember" {
            let props = &schema["properties"];
            assert!(
                props["node_type"].get("enum").is_none(),
                "native event types must not be excluded"
            );
            for field in [
                "entities",
                "attributes",
                "intent_id",
                "links",
                "scope",
                "validFrom",
            ] {
                assert!(props[field].is_object(), "root {field}");
                assert!(
                    props["items"]["items"]["properties"][field].is_object(),
                    "batch {field}"
                );
            }
        }
    }
}

#[test]
fn session_memory_packet_and_reflex_exposure_dedup_are_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let saved = s.tool(
        "remember",
        json!({"content":"Maya takes the train", "node_type":"event", "entities":["person:Maya"]}),
    );
    let hook = json!({"action":"start", "memory_hook":{"prompt":"Maya travel", "limit":3,
        "session":"hook-session", "host":"claude", "event":"prompt", "trigger":{"mode":"light"}}});
    let first = s.tool("session", hook.clone());
    assert_eq!(first["selection"], "exact_recorded_entities", "{first}");
    assert_eq!(first["memories"][0]["id"], saved["nodeId"]);
    let second = s.tool("session", hook);
    assert_eq!(second["memories"], json!([]), "{second}");
    assert_eq!(
        second["reflex"]["skippedAlreadyHeld"],
        json!([saved["nodeId"]])
    );
    let ledger = s.tool(
        "session",
        json!({"action":"reflex", "params":{"action":"ledger"}}),
    );
    assert_eq!(ledger["context"]["injected"], 1, "{ledger}");
    assert_eq!(ledger["context"]["skippedAlreadyHeld"], 1, "{ledger}");
    let status = s.tool(
        "session",
        json!({"action":"reflex", "params":{"action":"status"}}),
    );
    assert!(status["version"].is_string(), "{status}");
    assert_eq!(status["sessions"], 1);
    let boot = s.tool(
        "session",
        json!({"action":"start", "include_intentions":true, "include_status":true}),
    );
    assert!(
        boot.get("resolved_tags").is_none(),
        "boot must use session context: {boot}"
    );
    assert!(
        boot["context"].as_str().unwrap().contains("## Session"),
        "{boot}"
    );
    let rules = s.tool("session", json!({"action":"start", "cwd":dir.path()}));
    assert!(
        rules["resolved_tags"].is_array(),
        "cwd rules route remains intact: {rules}"
    );
}

#[test]
fn backfill_receipts_proof_queries_and_native_links_use_six_names() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let first = s.tool(
        "remember",
        json!({"content":"A logged observation", "tags":["symbol:hook:fixture"]}),
    );
    let second = s.tool(
        "remember",
        json!({"content":"A later logged observation", "tags":["symbol:hook:fixture"]}),
    );
    let receipts = s.tool(
        "remember",
        json!({"action":"backfill", "params":{"action":"receipts"}}),
    );
    assert!(receipts["receipts"].is_array());
    let preview = s.tool(
        "remember",
        json!({"action":"backfill", "params":{
        "action":"run", "trigger_id":second["nodeId"], "dry_run":true, "manual":true}}),
    );
    assert!(preview.is_object());
    for query in ["receipt", "replay"] {
        let proof = s.tool(
            "precedent",
            json!({"query":query, "receipt_id":first["nodeId"]}),
        );
        assert!(proof.is_object(), "{proof}");
    }
    let proof_error = s.tool_error("precedent", json!({"query":"prove", "repoPath":"relative"}));
    assert!(proof_error.contains("absolute path"), "{proof_error}");
    assert!(
        !proof_error.contains("unknown field"),
        "facade query must be removed"
    );
    let link = s.tool(
        "curate",
        json!({"action":"link", "source_id":second["nodeId"],
        "target_id":first["nodeId"], "link_type":"derived_from"}),
    );
    assert_eq!(link["links"][0]["edge"], "derived_from");
    let edge_receipt = s.tool(
        "precedent",
        json!({"query":"receipt", "receipt_id":link["links"][0]["receiptId"]}),
    );
    assert!(edge_receipt.is_object());
    let walk = s.tool("causal_walk", json!({"action":"walk", "start_points":[{"kind":"logged_write", "node_id":second["nodeId"]}], "promote":false}));
    assert!(
        walk.to_string().contains(first["nodeId"].as_str().unwrap()),
        "{walk}"
    );
    s.tool(
        "precedent",
        json!({"query":"forgotten_lesson", "failure_id":second["nodeId"]}),
    );
    let error = s.tool_error(
        "curate",
        json!({"action":"link", "source_id":second["nodeId"],
        "target_id":first["nodeId"], "link_type":"invented_edge"}),
    );
    assert!(error.contains("link_type must be"));
    let error = s.tool_error(
        "curate",
        json!({"action":"link", "source_id":second["nodeId"],
        "target_id":first["nodeId"], "scope":"foreign", "link_type":"derived_from"}),
    );
    assert!(error.contains("scope"));
}

#[test]
fn delegated_hook_operations_keep_native_results_and_withholding() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let saved = s.tool(
        "remember",
        json!({"content":"Hook route fixture", "tags":["hook-route"]}),
    );
    let got = s.tool(
        "curate",
        json!({"action":"memory", "params":{"action":"get", "id":saved["nodeId"]}}),
    );
    assert_eq!(got["node"]["content"], "Hook route fixture");
    let check = s.tool(
        "session",
        json!({"action":"intention", "params":{"action":"check", "context":{}}}),
    );
    assert!(check.is_object());
    let codebase = s.tool("session", json!({"action":"codebase", "params":{"action":"get_context", "codebase":"fixture", "repoPath":dir.path()}}));
    assert!(codebase.is_object());
    let graph = s.tool(
        "curate",
        json!({"action":"graph", "params":{"action":"recent", "limit":1}}),
    );
    assert!(graph.is_object());
    let map = s.tool(
        "curate",
        json!({"action":"ghostlink", "params":{"mode":"map", "limit":1}}),
    );
    assert!(map.is_object());
    let withheld = s.tool_error("curate", json!({"action":"memory", "params":{"action":"purge", "id":saved["nodeId"], "confirm":true}}));
    assert!(withheld.contains("unavailable"), "{withheld}");
    let retained = s.tool("recall", json!({"id":saved["nodeId"]}));
    assert_eq!(retained["nodes"][0]["id"], saved["nodeId"]);
    let withheld = s.tool_error(
        "curate",
        json!({"action":"maintain", "params":{"action":"consolidate"}}),
    );
    assert!(withheld.contains("unavailable"), "{withheld}");
}

#[test]
fn reflex_fold_guidance_unfolds_full_bytes_through_recall() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let body = "Native full body including all evidence. ".repeat(200);
    let saved = s.tool("remember", json!({"content":body, "tags":["fold-fixture"]}));
    s.tool(
        "session",
        json!({"action":"reflex", "params":{
        "action":"hook", "event":"SessionStart", "session":"fold-session", "host":"claude"}}),
    );
    let mut reads = Vec::new();
    for (call, args) in [
        (
            "call1",
            json!({"handle":"fold-fixture", "detail_level":"full"}),
        ),
        (
            "call2",
            json!({"handle":"fold-fixture", "detail_level":"full"}),
        ),
        (
            "call3",
            json!({"id":saved["nodeId"], "detail_level":"full"}),
        ),
    ] {
        s.tool(
            "session",
            json!({"action":"reflex", "params":{
            "action":"hook", "event":"PreToolUse", "session":"fold-session", "host":"claude",
            "tool":"mcp__vestige__recall", "call":call}}),
        );
        let result = s.rpc(
            "tools/call",
            json!({"name":"recall", "arguments":args,
            "_meta":{"claudecode/toolUseId":call}}),
        );
        assert_ne!(result["isError"], true, "{result}");
        reads.push(result["structuredContent"].clone());
    }
    assert_eq!(reads[0]["nodes"][0]["content"], body);
    let folded = reads[1]["nodes"][0]["content"].as_str().unwrap();
    assert!(folded.contains("recall(id="), "{folded}");
    assert!(folded.contains("detail_level='full'"), "{folded}");
    assert_eq!(reads[2]["nodes"][0]["content"], body);
    assert!(reads[2].get("reflexFold").is_none());
}

#[test]
fn unsupported_walk_actions_are_not_advertised_or_silently_executed() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let guide = s.tool(
        "session",
        json!({"action":"status", "view":"tools", "tool":"causal_walk"}),
    );
    assert_eq!(
        guide["tools"][0]["inputSchema"]["properties"]["action"]["enum"],
        json!(["walk", "prove"])
    );
    for action in ["ingest_repo", "verify", "typo"] {
        let error = s.tool_error("causal_walk", json!({"action":action}));
        assert!(error.contains("unknown causal_walk action"), "{error}");
        assert!(error.contains("session action='codebase'"), "{error}");
    }
}

#[test]
fn proof_returns_native_unwitnessed_result_and_rejects_unused_capabilities() {
    let dir = tempfile::tempdir().unwrap();
    let repo = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .current_dir(repo.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    git(&["init", "--quiet"]);
    std::fs::write(repo.path().join("fixture.txt"), "unwitnessed fixture").unwrap();
    git(&["add", "fixture.txt"]);
    git(&[
        "-c",
        "user.name=Native Test",
        "-c",
        "user.email=native@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    ]);
    let proof = s.tool(
        "causal_walk",
        json!({"action":"prove", "repoPath":repo.path(), "commit":"HEAD", "out":"none"}),
    );
    assert_eq!(proof["action"], "prove");
    assert_eq!(proof["summary"]["green"], 0, "{proof}");
    assert_eq!(proof["summary"]["witnesses"], 0, "{proof}");
    assert!(
        proof.get("proof_card").is_none(),
        "no synthetic VERIFIED card: {proof}"
    );
    assert_eq!(proof["written"], json!({}));
    assert!(!repo.path().join(".vestige/proof").exists());
    for field in ["github_issue", "good_ref", "bad_ref", "test_cmd"] {
        let mut args = json!({"action":"walk"});
        args[field] = json!("unused");
        let error = s.tool_error("causal_walk", args);
        assert!(
            error.contains("unsupported causal_walk argument"),
            "{error}"
        );
    }
    let guide = s.tool(
        "session",
        json!({"action":"status", "view":"tools", "tool":"causal_walk"}),
    );
    let props = &guide["tools"][0]["inputSchema"]["properties"];
    for field in ["github_issue", "good_ref", "bad_ref", "test_cmd"] {
        assert!(props.get(field).is_none());
    }
}

#[test]
fn memory_pr_queue_reports_native_strata_unavailability_explicitly() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let status = s.tool("session", json!({"action":"status", "view":"memory_prs"}));
    assert_eq!(status["available"], false, "{status}");
    assert_eq!(status["mode"], "fast");
    assert!(
        status["reason"]
            .as_str()
            .unwrap()
            .contains("not available on a Strata log")
    );
    let guide = s.tool(
        "session",
        json!({"action":"status", "view":"tools", "tool":"session"}),
    );
    assert!(
        guide["tools"][0]["inputSchema"]["properties"]["view"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("memory_prs"))
    );
}

#[test]
fn full_schemas_resolve_every_embedded_local_json_pointer() {
    fn validate(root: &Value, value: &Value) {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                    if let Some(pointer) = reference.strip_prefix('#') {
                        assert!(
                            root.pointer(pointer).is_some(),
                            "unresolved local schema reference {reference}"
                        );
                    }
                }
                for child in object.values() {
                    validate(root, child);
                }
            }
            Value::Array(values) => {
                for child in values {
                    validate(root, child);
                }
            }
            _ => {}
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    for tool in [
        "session",
        "remember",
        "recall",
        "causal_walk",
        "curate",
        "precedent",
    ] {
        let guide = s.tool(
            "session",
            json!({"action":"status", "view":"tools", "tool":tool}),
        );
        let schema = &guide["tools"][0]["inputSchema"];
        validate(schema, schema);
    }
}
