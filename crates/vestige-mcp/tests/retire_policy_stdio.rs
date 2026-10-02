//! Real vestige-mcp stdio: one allowed RETIRE and one held RETIRE, then
//! `vestige strata-verify` on that store and on a v3.1.1 migration.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio as ProcessStdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use strata_store::{AdmissionContext, RULE_SUPPRESS, StrataStore};

struct McpSession {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    seq: u64,
}

fn spawn(bin: &str, data_dir: &Path) -> McpSession {
    let mut cmd = Command::new(bin);
    cmd.arg("--no-http").arg("--data-dir").arg(data_dir);
    cmd.env_clear();
    for (key, value) in std::env::vars() {
        if key.starts_with("VESTIGE_") || key.starts_with("REDMINE_") || key.starts_with("GITHUB_")
        {
            continue;
        }
        cmd.env(key, value);
    }
    cmd.env("VESTIGE_DASHBOARD_ENABLED", "false");
    cmd.env("VESTIGE_HTTP_ENABLED", "false");
    cmd.env("RUST_LOG", "error");
    cmd.stdin(ProcessStdio::piped())
        .stdout(ProcessStdio::piped())
        .stderr(ProcessStdio::piped());
    let mut child = cmd.spawn().expect("spawn vestige-mcp");
    let stdout = child.stdout.take().expect("stdout");
    let stderr = child.stderr.take().expect("stderr");
    thread::spawn(move || {
        let mut sink = String::new();
        let _ = BufReader::new(stderr).read_to_string(&mut sink);
    });
    let (tx, lines) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let stdin = child.stdin.take().expect("stdin");
    McpSession {
        child,
        stdin,
        lines,
        seq: 0,
    }
}

impl McpSession {
    fn rpc(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.seq += 1;
        let id = self.seq;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write rpc");
        self.stdin.flush().expect("flush rpc");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("timed out waiting for {method}");
            }
            let line = self
                .lines
                .recv_timeout(remaining)
                .unwrap_or_else(|_| panic!("stdio closed waiting for {method}"));
            let value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if value.get("id") != Some(&serde_json::json!(id)) {
                continue;
            }
            assert!(value.get("error").is_none(), "{value}");
            return value["result"].clone();
        }
    }

    fn tool(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let result = self.rpc(
            "tools/call",
            serde_json::json!({ "name": name, "arguments": args }),
        );
        assert_eq!(
            result.get("isError"),
            Some(&serde_json::json!(false)),
            "{result}"
        );
        if let Some(body) = result
            .get("structuredContent")
            .filter(|body| body.get("nodeId").is_some())
        {
            return body.clone();
        }
        let text = result["content"][0]["text"].as_str().expect("tool text");
        serde_json::from_str(text).unwrap_or_else(|_| panic!("tool text is not json: {text}"))
    }

    fn close(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait().expect("wait") {
                assert!(status.success(), "vestige-mcp exit {status}");
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!("vestige-mcp did not exit after stdin EOF");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

fn ingest(mcp: &mut McpSession, content: &str) -> String {
    let created = mcp.tool(
        "smart_ingest",
        serde_json::json!({ "content": content, "forceCreate": true }),
    );
    created["nodeId"].as_str().expect("nodeId").to_string()
}

fn vestige_ok(args: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(args)
        .output()
        .expect("spawn vestige");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "vestige {args:?}\n{stdout}\n{stderr}"
    );
    if args.first() == Some(&"strata-verify") {
        assert!(stdout.contains("OK"), "{stdout}\n{stderr}");
    }
}

#[test]
fn stdio_allows_one_retire_and_holds_another_then_strata_verify() {
    let dir = std::env::temp_dir().join(format!("vestige-retire-stdio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("data dir");

    let (allowed_id, held_id, successor_id, edit_source, edit_successor, edit_receipt) = {
        let mut mcp = spawn(env!("CARGO_BIN_EXE_vestige-mcp"), &dir);
        mcp.rpc(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "retire-policy", "version": "1" },
            }),
        );
        writeln!(
            mcp.stdin,
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
        )
        .unwrap();
        mcp.stdin.flush().unwrap();
        let allowed_id = ingest(&mut mcp, "STDIO_RETIRE_ALLOWED");
        let held_id = ingest(&mut mcp, "STDIO_RETIRE_HELD");
        let successor_id = ingest(&mut mcp, "STDIO_RETIRE_SUCCESSOR");
        let review_id = ingest(&mut mcp, "STDIO_REVIEW_TARGET");
        let promoted = mcp.tool(
            "memory",
            serde_json::json!({
                "action": "promote",
                "id": review_id,
                "reason": "stdio"
            }),
        );
        assert_eq!(promoted["success"], true);
        assert_eq!(promoted["action"], "promoted");
        assert_eq!(promoted["nodeId"], review_id);
        let promote_receipt = promoted["receiptId"].as_str().expect("promote receipt");
        assert!(promote_receipt.starts_with("eff-"), "{promoted}");
        assert!(promoted.get("actor").is_some(), "{promoted}");
        let demoted = mcp.tool(
            "memory",
            serde_json::json!({
                "action": "demote",
                "id": review_id,
                "reason": "stdio"
            }),
        );
        assert_eq!(demoted["success"], true);
        assert_eq!(demoted["action"], "demoted");
        assert_eq!(demoted["nodeId"], review_id);
        assert!(
            demoted["note"]
                .as_str()
                .unwrap_or("")
                .contains("not deleted"),
            "{demoted}"
        );
        assert_ne!(demoted["receiptId"], promote_receipt);
        assert!(demoted["receiptId"].as_str().unwrap().starts_with("eff-"));
        assert!(demoted.get("actor").is_some(), "{demoted}");
        let edit_source = ingest(&mut mcp, "STDIO_EDIT_SOURCE");
        let edited = mcp.tool(
            "memory",
            serde_json::json!({
                "action": "edit",
                "id": edit_source,
                "content": "STDIO_EDIT_SUCCESSOR"
            }),
        );
        let edit_successor = edited["nodeId"].as_str().expect("successor").to_string();
        let edit_receipt = edited["receiptId"].as_str().expect("receipt").to_string();
        assert_ne!(edit_successor, edit_source);
        assert_eq!(edited["rule"], "edit");
        assert_eq!(edited["supersedes"], edit_source);
        assert_eq!(edited["success"], true);
        assert!(edit_receipt.starts_with("eff-"));
        let retired = mcp.tool("recall", serde_json::json!({ "handle": edit_source }));
        assert!(
            retired.get("nodes").is_none()
                || retired["nodes"]
                    .as_array()
                    .is_some_and(|nodes| nodes.is_empty()),
            "old node returned by recall: {retired}"
        );
        let live = mcp.tool("recall", serde_json::json!({ "handle": edit_successor }));
        assert_eq!(live["nodes"][0]["content"], "STDIO_EDIT_SUCCESSOR");
        mcp.close();
        (
            allowed_id,
            held_id,
            successor_id,
            edit_source,
            edit_successor,
            edit_receipt,
        )
    };

    // The stdio process writes `log/*.seg` and no `store.meta`. That log is
    // the fresh store.
    let log_dir = dir.join("log");
    vestige_ok(&["strata-verify", log_dir.to_str().expect("utf8 log")]);

    {
        let mut store = StrataStore::open(&dir).expect("open stdio store");
        let edit_seq = u64::from_str_radix(edit_receipt.strip_prefix("eff-").expect("prefix"), 16)
            .expect("receipt seq");
        let edit_again = store.retire_receipt(edit_seq).expect("edit receipt");
        assert_eq!(edit_again.rule_id, Some(strata_store::RULE_EDIT));
        assert_eq!(edit_again.receipt_id, edit_receipt);
        assert_eq!(
            store
                .get_node(&edit_source)
                .expect("edited source")
                .superseded_by
                .as_deref(),
            Some(edit_successor.as_str())
        );
        assert_eq!(
            store.get_node(&edit_source).expect("source").content,
            "STDIO_EDIT_SOURCE"
        );
        assert_eq!(
            store.get_node(&edit_successor).expect("successor").content,
            "STDIO_EDIT_SUCCESSOR"
        );
        let scope = store.get_node(&edit_source).expect("source").scope;
        assert!(
            store
                .get_all_nodes_in_scope(&scope)
                .iter()
                .all(|node| node.id != edit_source)
        );
        let receipt = store
            .retire(
                &allowed_id,
                &successor_id,
                &AdmissionContext {
                    rule_id: Some(RULE_SUPPRESS.to_string()),
                    confirm: false,
                },
            )
            .expect("suppress rule allows the RETIRE");
        assert_eq!(receipt.rule_id, Some(RULE_SUPPRESS));
        assert_eq!(
            receipt.receipt_id,
            strata_store::effect_receipt_id(receipt.effect_seq)
        );
        assert!(receipt.receipt_id.starts_with("eff-"));
        assert_eq!(
            store
                .get_node(&allowed_id)
                .expect("allowed")
                .superseded_by
                .as_deref(),
            Some(successor_id.as_str())
        );
        let err = store
            .retire(&held_id, &successor_id, &AdmissionContext::default())
            .expect_err("RETIRE with no rule id is held");
        assert!(
            matches!(err, strata_store::StoreError::Held { .. }),
            "{err}"
        );
        assert!(
            store
                .get_node(&held_id)
                .expect("held")
                .superseded_by
                .is_none()
        );
        // `store.meta` is what makes the data dir a live store for strata-verify.
        store.seal_checkpoint().expect("seal");
    }

    vestige_ok(&["strata-verify", dir.to_str().expect("utf8 dir")]);

    let migrated = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    let dest = std::env::temp_dir().join(format!("vestige-retire-migrated-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dest);
    vestige_ok(&[
        "migrate-to-strata",
        "--from",
        migrated.to_str().expect("fixture path"),
        "--to",
        dest.to_str().expect("dest path"),
    ]);
    vestige_ok(&["strata-verify", dest.to_str().expect("dest path")]);

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dest).ok();
}

fn handshake(mcp: &mut McpSession) {
    mcp.rpc(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "purge-retire", "version": "1" },
        }),
    );
    writeln!(
        mcp.stdin,
        "{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}}"
    )
    .unwrap();
    mcp.stdin.flush().unwrap();
}

fn tool_body(mcp: &mut McpSession, name: &str, args: serde_json::Value) -> serde_json::Value {
    let result = mcp.rpc(
        "tools/call",
        serde_json::json!({ "name": name, "arguments": args }),
    );
    if let Some(body) = result
        .get("structuredContent")
        .filter(|body| !body.is_null())
    {
        return body.clone();
    }
    let text = result["content"][0]["text"].as_str().unwrap_or("");
    serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({ "text": text }))
}

fn log_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.join("log")];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|name| name.to_str()) == Some("strata.lock") {
                continue;
            }
            if let Ok(bytes) = std::fs::read(&path) {
                let rel = path
                    .strip_prefix(dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, bytes));
            }
        }
    }
    out.sort();
    out
}

/// 4.0 withholds erasure on Strata. Over real stdio: purge, the memory
/// purge/delete actions and the hidden delete_knowledge alias all refuse,
/// purge is not advertised, the log is byte-identical afterwards, and the
/// memory is still readable after a restart. Suppress stays advertised as a
/// destructive hide (it never erased).
#[test]
fn stdio_erasure_is_withheld_and_writes_nothing() {
    let dir = std::env::temp_dir().join(format!("vestige-withheld-stdio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("data dir");
    let marker = "STDIO_WITHHELD_MARKER_9f3a";

    let node_id = {
        let mut mcp = spawn(env!("CARGO_BIN_EXE_vestige-mcp"), &dir);
        handshake(&mut mcp);
        let node_id = ingest(&mut mcp, marker);
        mcp.close();
        node_id
    };
    let before = log_files(&dir);

    {
        let mut mcp = spawn(env!("CARGO_BIN_EXE_vestige-mcp"), &dir);
        handshake(&mut mcp);
        let listed = mcp.rpc("tools/list", serde_json::json!({}));
        let names: Vec<&str> = listed["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();
        assert!(!names.contains(&"purge"), "{names:?}");
        let suppress = listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "suppress")
            .expect("suppress stays advertised: it hides, it never erased");
        assert_eq!(
            suppress["annotations"]["destructiveHint"],
            serde_json::json!(true),
            "{suppress}"
        );
        let memory = listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "memory")
            .expect("memory tool");
        let actions = memory["inputSchema"]["properties"]["action"]["enum"].to_string();
        assert!(
            !actions.contains("purge") && !actions.contains("delete"),
            "{actions}"
        );

        for (name, args) in [
            (
                "purge",
                serde_json::json!({ "id": node_id, "confirm": true }),
            ),
            (
                "memory",
                serde_json::json!({ "action": "purge", "id": node_id, "confirm": true }),
            ),
            (
                "memory",
                serde_json::json!({ "action": "delete", "id": node_id, "confirm": true }),
            ),
            (
                "delete_knowledge",
                serde_json::json!({ "id": node_id, "confirm": true }),
            ),
        ] {
            let result = mcp.rpc(
                "tools/call",
                serde_json::json!({ "name": name, "arguments": args }),
            );
            assert_eq!(
                result["isError"],
                serde_json::json!(true),
                "{name}: {result}"
            );
            assert!(
                result.to_string().contains("unavailable_in_4_0"),
                "{name}: {result}"
            );
        }
        mcp.close();
    }
    assert_eq!(log_files(&dir), before, "a withheld call changed the log");

    {
        let mut mcp = spawn(env!("CARGO_BIN_EXE_vestige-mcp"), &dir);
        handshake(&mut mcp);
        let got = tool_body(
            &mut mcp,
            "memory",
            serde_json::json!({ "action": "get", "id": node_id }),
        );
        assert!(got.to_string().contains(marker), "{got}");
        mcp.close();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
