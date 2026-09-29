//! Every advertised MCP tool, over real stdio: valid input, invalid input,
//! and an exact-handle miss that must return `handle_required`.

use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::support::*;

fn exercise(tool: &str, invalid: Value, valid: Value, miss: Value, write: bool) {
    with_server(|server| {
        let listed = server
            .call("tools/list", None)
            .unwrap_or_else(|e| missing(&format!("tools/list failed: {e}")));
        let names: Vec<String> = listed["result"]["tools"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_string))
            .collect();
        if !names.iter().any(|n| n == tool) {
            missing(&format!(
                "server tool list does not advertise `{tool}`. Live list: {names:?}"
            ));
        }
        let bad = server.tool(tool, invalid).expect("rpc");
        assert!(
            bad.rejected(),
            "{tool} accepted invalid input: {}",
            bad.blob()
        );
        let good = server.tool(tool, valid).expect("rpc");
        if good.rejected() {
            missing(&format!(
                "`{tool}` rejected a schema-valid call. The strata tool dispatch is incomplete \
                 on this head. Got: {}",
                good.blob().chars().take(800).collect::<String>()
            ));
        }
        if write {
            assert_receipt(tool, &good);
        }
        let missed = server.tool(tool, miss).expect("rpc");
        assert_handle_required(tool, &missed);
    });
}

#[test]
fn mcp_catalog_matches_server_tool_list() {
    with_server(|server| {
        let listed = server.call("tools/list", None).expect("tools/list");
        let mut names: Vec<String> = listed["result"]["tools"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_string))
            .collect();
        names.sort();
        let mut expected = expected_tool_names();
        expected.sort();
        if names != expected {
            missing(&format!(
                "tools/list does not match expected_tools_4.0.txt. live={names:?} expected={expected:?}"
            ));
        }
    });
}

#[test]
fn mcp_recall_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server.tool("recall", json!({ "handle": 1 })).expect("rpc");
        assert!(bad.rejected() || bad.handle_required(), "{}", bad.blob());
        let created = server
            .tool(
                "smart_ingest",
                json!({ "content": "release matrix recall anchor", "forceCreate": true }),
            )
            .expect("rpc");
        if created.rejected() {
            missing(&format!(
                "recall's valid path needs a stored memory, and smart_ingest failed: {}",
                created.blob().chars().take(600).collect::<String>()
            ));
        }
        let id = created.body["nodeId"]
            .as_str()
            .or_else(|| created.body["id"].as_str())
            .unwrap_or("");
        if id.is_empty() {
            missing(&format!(
                "smart_ingest returned no node id, so recall has no exact handle to resolve: {}",
                created.blob().chars().take(600).collect::<String>()
            ));
        }
        let handle = if id.starts_with("mem-") || id.contains(':') {
            id.to_string()
        } else {
            format!("mem:{id}")
        };
        let good = server
            .tool("recall", json!({ "handle": handle }))
            .expect("rpc");
        if good.rejected() || good.handle_required() {
            missing(&format!(
                "recall of a handle this process just wrote failed: {}",
                good.blob().chars().take(800).collect::<String>()
            ));
        }
        assert_receipt("recall", &good);
        let missed = server
            .tool("recall", json!({ "handle": MISSING_HANDLE }))
            .expect("rpc");
        assert_handle_required("recall", &missed);
    });
}

#[test]
fn mcp_recall_free_text_is_not_a_search() {
    with_server(|server| {
        let _ = server.tool(
            "smart_ingest",
            json!({
                "content": "payments api timeout is thirty seconds",
                "forceCreate": true
            }),
        );
        let reply = server
            .tool("recall", json!({ "query": "payments api timeout" }))
            .expect("rpc");
        assert_handle_required("recall", &reply);
        let blob = reply.blob();
        assert!(
            !blob.contains("thirty seconds"),
            "free-text recall returned memory content (keyword/semantic match is forbidden): {blob}"
        );
    });
}

#[test]
fn mcp_receipt_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server
            .tool("receipt", json!({ "action": "not-an-action" }))
            .expect("rpc");
        assert!(bad.rejected(), "{}", bad.blob());
        let missed = server
            .tool(
                "receipt",
                json!({ "action": "get", "receipt_id": MISSING_HANDLE }),
            )
            .expect("rpc");
        assert_handle_required("receipt", &missed);
        // A valid get needs a receipt this server just signed. Recall of a
        // freshly written handle is the writer. If that path has no receipt
        // id yet, the test names the gap instead of pretending a fake id worked.
        let created = server
            .tool(
                "smart_ingest",
                json!({ "content": "receipt source memory", "forceCreate": true }),
            )
            .expect("rpc");
        if created.rejected() {
            missing(&format!(
                "receipt get needs a live memory to recall. smart_ingest failed: {}",
                created.blob().chars().take(500).collect::<String>()
            ));
        }
        let id = created.body["nodeId"].as_str().unwrap_or("");
        let handle = if id.starts_with("mem-") || id.contains(':') {
            id.to_string()
        } else {
            format!("mem:{id}")
        };
        let recall = server
            .tool("recall", json!({ "handle": handle }))
            .expect("rpc");
        if !has_gate_receipt(&recall) {
            missing(&format!(
                "recall did not return a signed receipt, so receipt(action=get) has nothing real to fetch: {}",
                recall.blob().chars().take(600).collect::<String>()
            ));
        }
        let receipt_id = recall.body["receiptId"]
            .as_str()
            .or_else(|| recall.body["receipt_id"].as_str())
            .or_else(|| recall.body["receipt"]["id"].as_str())
            .unwrap_or("");
        if receipt_id.is_empty() {
            missing(&format!(
                "recall receipt payload has no id field: {}",
                recall.blob().chars().take(600).collect::<String>()
            ));
        }
        let good = server
            .tool(
                "receipt",
                json!({ "action": "get", "receipt_id": receipt_id }),
            )
            .expect("rpc");
        if good.rejected() {
            missing(&format!(
                "receipt get of the id recall just returned failed: {}",
                good.blob().chars().take(600).collect::<String>()
            ));
        }
    });
}

#[test]
fn mcp_memory_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server.tool("memory", json!({ "action": 1 })).expect("rpc");
        assert!(bad.rejected(), "{}", bad.blob());
        let created = server
            .tool(
                "smart_ingest",
                json!({ "content": "memory get target", "forceCreate": true }),
            )
            .expect("rpc");
        if created.rejected() {
            missing(&format!(
                "memory get needs a stored node. smart_ingest failed: {}",
                created.blob().chars().take(500).collect::<String>()
            ));
        }
        let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
        if id.is_empty() {
            missing("smart_ingest returned no nodeId");
        }
        let good = server
            .tool("memory", json!({ "action": "get", "id": id }))
            .expect("rpc");
        if good.rejected() || good.handle_required() {
            missing(&format!(
                "memory get of a node just written failed: {}",
                good.blob().chars().take(600).collect::<String>()
            ));
        }
        let missed = server
            .tool("memory", json!({ "action": "get", "id": MISSING_HANDLE }))
            .expect("rpc");
        assert_handle_required("memory", &missed);
    });
}

const PURGE_CANARY: &str = "CANARY-purge-4f91c0e2-must-not-be-readable";

fn ingest_canary(server: &mut Server) -> String {
    let created = server
        .tool(
            "smart_ingest",
            json!({ "content": PURGE_CANARY, "forceCreate": true }),
        )
        .expect("rpc");
    if created.rejected() {
        missing(&format!(
            "purge needs a stored node. smart_ingest failed: {}",
            created.blob().chars().take(500).collect::<String>()
        ));
    }
    let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
    if id.is_empty() {
        missing("smart_ingest returned no nodeId for the purge canary");
    }
    id
}

fn memory_handle(id: &str) -> String {
    if id.starts_with("mem-") || id.contains(':') {
        id.to_string()
    } else {
        format!("mem:{id}")
    }
}

/// Read and search surfaces. The confirmed-purge row fails if any of them
/// still returns the canary. Segment bytes are not one of these surfaces.
fn read_surfaces(server: &mut Server, id: &str) -> Vec<(String, String)> {
    let handle = memory_handle(id);
    let calls = [
        ("memory get", "memory", json!({ "action": "get", "id": id })),
        (
            "memory get_batch",
            "memory",
            json!({ "action": "get_batch", "ids": [id] }),
        ),
        ("recall", "recall", json!({ "handle": handle })),
        (
            "graph memory",
            "graph",
            json!({ "action": "memory", "id": id }),
        ),
        (
            "graph recent",
            "graph",
            json!({ "action": "recent", "limit": 20 }),
        ),
        (
            "project preview",
            "project",
            json!({ "action": "preview", "scope": "user" }),
        ),
        ("dedup scan", "dedup", json!({ "action": "scan" })),
        (
            "memory_status",
            "memory_status",
            json!({ "view": "health" }),
        ),
    ];
    calls
        .into_iter()
        .map(|(label, tool, args)| {
            let reply = server
                .tool(tool, args)
                .unwrap_or_else(|e| panic!("HARNESS: {label} rpc failed: {e}"));
            (label.to_string(), reply.blob())
        })
        .collect()
}

fn verify_store_ok(dir: &Path) {
    let log = dir.join("log");
    let target = if log.is_dir() { log } else { dir.to_path_buf() };
    let out = run_cmd(
        &product_bin("strata-verify"),
        &[target.display().to_string()],
        &[],
        &[],
        std::time::Duration::from_secs(60),
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) || !blob.contains("OK") {
        panic!(
            "FAIL: strata-verify did not pass the store after confirmed purge (exit {:?}). {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        );
    }
}

/// Purge is a gated retire. `memory` purge with `confirm:true` returns a
/// gate receipt, and no read or search tool returns the canary afterward.
/// The raw `log/*.seg` bytes are not inspected here.
#[test]
fn memory_purge_confirm_returns_gate_receipt() {
    with_server(|server| {
        let id = ingest_canary(server);
        let purged = server
            .tool(
                "memory",
                json!({ "action": "purge", "id": id, "confirm": true }),
            )
            .expect("rpc");
        if !has_gate_receipt(&purged) {
            panic!(
                "FAIL: memory purge with confirm:true did not return a gate receipt. {}",
                purged.blob().chars().take(800).collect::<String>()
            );
        }
        let leaks: Vec<String> = read_surfaces(server, &id)
            .into_iter()
            .filter(|(_, blob)| blob.contains(PURGE_CANARY))
            .map(|(label, _)| label)
            .collect();
        if !leaks.is_empty() {
            panic!(
                "FAIL: confirmed purge left the canary readable over stdio via {leaks:?}. \
                 Segment bytes were not checked."
            );
        }
        verify_store_ok(&server.data_dir);
    });
}

/// Unconfirmed purge is a refusal, not a hold and not a receipt.
#[test]
fn purge_without_confirm_refused_with_no_receipt() {
    with_server(|server| {
        let id = ingest_canary(server);
        for (tool, args) in [
            ("memory", json!({ "action": "purge", "id": id })),
            ("purge", json!({ "id": id })),
        ] {
            let refused = server.tool(tool, args).expect("rpc");
            if !refused.rejected() || has_gate_receipt(&refused) {
                panic!(
                    "FAIL: {tool} purge without confirm:true must be refused and must not \
                     return a receipt. rejected={} receipt={} body={}",
                    refused.rejected(),
                    has_gate_receipt(&refused),
                    refused.blob().chars().take(800).collect::<String>()
                );
            }
        }
        let still = server
            .tool("memory", json!({ "action": "get", "id": id }))
            .expect("rpc");
        if !still.blob().contains(PURGE_CANARY) {
            panic!(
                "FAIL: purge without confirm:true dropped the memory. {}",
                still.blob().chars().take(500).collect::<String>()
            );
        }
    });
}

/// A retire that does not name a rule is held. The memory stays readable.
#[test]
fn retire_under_no_named_rule_is_held() {
    with_server(|server| {
        let id = ingest_canary(server);
        let held = server
            .tool(
                "memory",
                json!({ "action": "retire", "id": id, "confirm": true }),
            )
            .expect("rpc");
        let blob = held.blob().to_ascii_lowercase();
        let is_hold = blob.contains("\"hold\"")
            || blob.contains("held")
            || blob.contains("outcome\":\"held")
            || blob.contains("verdict\":\"hold");
        if !is_hold {
            panic!(
                "FAIL: retire with no named rule must be held. No `rule` argument was sent. {}",
                held.blob().chars().take(800).collect::<String>()
            );
        }
        let still = server
            .tool("memory", json!({ "action": "get", "id": id }))
            .expect("rpc");
        if !still.blob().contains(PURGE_CANARY) {
            panic!(
                "FAIL: a held retire removed the memory. {}",
                still.blob().chars().take(500).collect::<String>()
            );
        }
    });
}

/// 4.1 erasure check. Not part of the 4.0 run (`cargo test` does not execute
/// `#[ignore]`). After a confirmed purge the canary must be absent from
/// `log/*.seg`.
#[test]
#[ignore = "deferred_4_1"]
fn purge_canary_absent_from_segment_bytes() {
    with_server(|server| {
        let id = ingest_canary(server);
        let purged = server
            .tool(
                "memory",
                json!({ "action": "purge", "id": id, "confirm": true }),
            )
            .expect("rpc");
        assert!(
            has_gate_receipt(&purged),
            "deferred_4_1 purge did not return a receipt: {}",
            purged.blob().chars().take(400).collect::<String>()
        );
        let needle = PURGE_CANARY.as_bytes();
        let mut found = Vec::new();
        let mut stack = vec![server.data_dir.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("seg") {
                    continue;
                }
                if fs::read(&path)
                    .unwrap_or_default()
                    .windows(needle.len())
                    .any(|window| window == needle)
                {
                    found.push(path);
                }
            }
        }
        assert!(
            found.is_empty(),
            "deferred_4_1: canary bytes still in {found:?}"
        );
    });
}

#[test]
fn mcp_codebase_valid_invalid_handle_miss() {
    exercise(
        "codebase",
        json!({ "action": "nope" }),
        json!({
            "action": "remember_decision",
            "decision": "Strata is the only store",
            "rationale": "signed append-only log",
            "codebase": "vestige"
        }),
        json!({ "action": "reanchor", "memoryId": MISSING_HANDLE }),
        true,
    );
}

#[test]
fn mcp_project_valid_invalid_handle_miss() {
    exercise(
        "project",
        json!({ "action": 7 }),
        json!({ "action": "preview", "scope": "user" }),
        json!({ "action": "preview", "handle": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_intention_valid_invalid_handle_miss() {
    exercise(
        "intention",
        json!({ "action": "nope" }),
        json!({ "action": "list" }),
        json!({ "action": "check", "id": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_smart_ingest_valid_invalid_handle_miss() {
    exercise(
        "smart_ingest",
        json!({ "content": 1 }),
        json!({ "content": "release matrix ingest", "forceCreate": true }),
        json!({ "content": "release matrix ingest", "handle": MISSING_HANDLE }),
        true,
    );
}

#[test]
fn mcp_source_sync_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server
            .tool("source_sync", json!({ "source": 1 }))
            .expect("rpc");
        assert!(bad.rejected(), "{}", bad.blob());
        // No token is a completed call: the tool must refuse with a
        // configuration error, not by searching or crashing.
        let good = server
            .tool(
                "source_sync",
                json!({ "source": "github", "repo": "example/none" }),
            )
            .expect("rpc");
        let blob = good.blob();
        if blob.contains("sqlite") {
            missing(&format!("source_sync touched sqlite: {blob}"));
        }
        let missed = server
            .tool(
                "source_sync",
                json!({ "source": "github", "repo": "example/none", "handle": MISSING_HANDLE }),
            )
            .expect("rpc");
        assert_handle_required("source_sync", &missed);
    });
}

#[test]
fn mcp_memory_status_valid_invalid_handle_miss() {
    exercise(
        "memory_status",
        json!({ "view": 1 }),
        json!({ "view": "health" }),
        json!({ "view": "health", "handle": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_maintain_valid_invalid_handle_miss() {
    exercise(
        "maintain",
        json!({ "action": "nope" }),
        json!({ "action": "gc", "dry_run": true }),
        json!({ "action": "gc", "dry_run": true, "handle": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_dedup_valid_invalid_handle_miss() {
    exercise(
        "dedup",
        json!({ "action": "nope" }),
        json!({ "action": "scan" }),
        json!({ "action": "protect", "id": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_graph_valid_invalid_handle_miss() {
    exercise(
        "graph",
        json!({ "action": "nope" }),
        json!({ "action": "recent", "limit": 5 }),
        json!({ "action": "memory", "id": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_session_start_valid_invalid_handle_miss() {
    exercise(
        "session_start",
        json!({ "budget_tokens": "lots" }),
        json!({}),
        json!({ "handle": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_suppress_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server.tool("suppress", json!({ "id": 1 })).expect("rpc");
        assert!(bad.rejected(), "{}", bad.blob());
        let created = server
            .tool(
                "smart_ingest",
                json!({ "content": "suppress target", "forceCreate": true }),
            )
            .expect("rpc");
        if created.rejected() {
            missing(&format!(
                "suppress needs a stored node. smart_ingest failed: {}",
                created.blob().chars().take(500).collect::<String>()
            ));
        }
        let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
        let good = server
            .tool("suppress", json!({ "id": id, "reason": "release-matrix" }))
            .expect("rpc");
        if good.rejected() && !has_gate_receipt(&good) {
            missing(&format!(
                "suppress of a node just written failed without a gate receipt: {}",
                good.blob().chars().take(600).collect::<String>()
            ));
        }
        assert_receipt("suppress", &good);
        let missed = server
            .tool("suppress", json!({ "id": MISSING_HANDLE }))
            .expect("rpc");
        assert_handle_required("suppress", &missed);
    });
}

#[test]
fn mcp_causal_walk_valid_invalid_handle_miss() {
    exercise(
        "causal_walk",
        json!({ "start_points": "nope" }),
        json!({
            "start_points": [{ "kind": "failing_test", "name": "release_matrix_absent" }],
            "promote": false
        }),
        json!({
            "start_points": [{ "kind": "logged_write", "node_id": MISSING_HANDLE }]
        }),
        false,
    );
}

#[test]
fn mcp_selftest_valid_invalid_handle_miss() {
    exercise(
        "selftest",
        json!({ "rounds": "many" }),
        json!({}),
        json!({ "handle": MISSING_HANDLE }),
        false,
    );
}

#[test]
fn mcp_forgotten_lesson_valid_invalid_handle_miss() {
    with_server(|server| {
        let bad = server
            .tool("forgotten_lesson", json!({ "failure_id": 1 }))
            .expect("rpc");
        assert!(bad.rejected(), "{}", bad.blob());
        let created = server
            .tool(
                "smart_ingest",
                json!({
                    "content": "error: release matrix failure fixture",
                    "forceCreate": true,
                    "tags": ["failure"]
                }),
            )
            .expect("rpc");
        if created.rejected() {
            missing(&format!(
                "forgotten_lesson needs a failure memory. smart_ingest failed: {}",
                created.blob().chars().take(500).collect::<String>()
            ));
        }
        let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
        let good = server
            .tool("forgotten_lesson", json!({ "failure_id": id }))
            .expect("rpc");
        if good.rejected() && !good.handle_required() {
            missing(&format!(
                "forgotten_lesson rejected the failure id just written: {}",
                good.blob().chars().take(600).collect::<String>()
            ));
        }
        let missed = server
            .tool("forgotten_lesson", json!({ "failure_id": MISSING_HANDLE }))
            .expect("rpc");
        assert_handle_required("forgotten_lesson", &missed);
    });
}

#[test]
fn mcp_writes_and_destructive_actions_carry_gate_receipts() {
    with_server(|server| {
        let ingest = server
            .tool(
                "smart_ingest",
                json!({ "content": "receipt anchor for purge", "forceCreate": true }),
            )
            .expect("rpc");
        if ingest.rejected() {
            missing(&format!(
                "smart_ingest (write) failed before a receipt could be checked: {}",
                ingest.blob().chars().take(600).collect::<String>()
            ));
        }
        assert_receipt("smart_ingest", &ingest);
        let id = ingest.body["nodeId"].as_str().unwrap_or("").to_string();
        if id.is_empty() {
            missing("smart_ingest succeeded without a node id");
        }
        let purge = server
            .tool(
                "purge",
                json!({ "id": id, "confirm": true, "reason": "release-matrix" }),
            )
            .expect("rpc");
        if purge.rejected() && !purge.handle_required() {
            // A rejection that is itself a hold/deny is a receipt if it
            // carries the verdict. Otherwise the destructive path is missing.
            if !has_gate_receipt(&purge) {
                missing(&format!(
                    "purge did not return a gate verdict/receipt: {}",
                    purge.blob().chars().take(800).collect::<String>()
                ));
            }
        } else if !purge.rejected() {
            assert_receipt("purge", &purge);
        }
    });
}

/// `confirm: true` may change the target only when a gated receipt for a
/// `projected_to` edge records blake3 or sha256 of the bytes now in the file.
/// `confirm: false` leaves the file byte-identical. A write that lands without
/// that receipt is a real failure. It is not pending_strata.
#[test]
fn project_write_requires_matching_receipt() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("CLAUDE.md");
    let preamble = b"# human\n\nKeep this line.\n";
    fs::write(&target, preamble).unwrap();
    let before = fs::read(&target).unwrap();

    with_server(|server| {
        let _ = server.tool(
            "smart_ingest",
            json!({
                "content": "Deploys use blue-green behind the ALB; cutover is a target-group swap.",
                "nodeType": "decision",
                "tags": ["rule"],
                "forceCreate": true
            }),
        );
        let refused = server
            .tool(
                "project",
                json!({
                    "action": "write",
                    "path": "CLAUDE.md",
                    "root": root.path(),
                    "confirm": false
                }),
            )
            .expect("rpc");
        let after_refuse = fs::read(&target).unwrap();
        assert_eq!(
            after_refuse,
            before,
            "confirm:false changed the target file. reply: {}",
            refused.blob().chars().take(800).collect::<String>()
        );

        let wrote = server
            .tool(
                "project",
                json!({
                    "action": "write",
                    "path": "CLAUDE.md",
                    "root": root.path(),
                    "confirm": true
                }),
            )
            .expect("rpc");
        let after = fs::read(&target).unwrap();
        let changed = after != before;
        let matched = projected_receipt_matches(&server.data_dir, &wrote, &after);
        if !(changed && matched) {
            panic!(
                "FAIL: project confirm:true is a real gate failure, not pending_strata. The file must change and a gated receipt for a projected_to edge must record blake3 or sha256 of those bytes. An ungated write is not a pass. file_changed={changed} receipt_match={matched} reply={}",
                wrote.blob().chars().take(1200).collect::<String>()
            );
        }
    });
}

fn projected_receipt_matches(data_dir: &Path, reply: &ToolReply, file: &[u8]) -> bool {
    let blake = *blake3::hash(file).as_bytes();
    let sha = Sha256::digest(file);
    let blake_hex = hex_encode(&blake);
    let sha_hex = hex_encode(&sha);
    if json_has_projected_receipt(&reply.body, &blake_hex, &sha_hex)
        || json_has_projected_receipt(&reply.raw, &blake_hex, &sha_hex)
    {
        return true;
    }
    let mut blobs = Vec::new();
    collect_log_blobs(data_dir, &mut blobs);
    blobs.iter().any(|blob| {
        window_has_projected_receipt(blob, &blake, sha.as_slice(), &blake_hex, &sha_hex)
    })
}

fn json_has_projected_receipt(value: &Value, blake_hex: &str, sha_hex: &str) -> bool {
    match value {
        Value::Object(map) => {
            let rendered = value.to_string();
            let edge = rendered.contains("projected_to");
            let hash = rendered.contains(blake_hex) || rendered.contains(sha_hex);
            let gated = rendered.contains("receipt")
                || rendered.contains("verdict")
                || rendered.contains("admission");
            if edge && hash && gated {
                return true;
            }
            map.values()
                .any(|child| json_has_projected_receipt(child, blake_hex, sha_hex))
        }
        Value::Array(items) => items
            .iter()
            .any(|child| json_has_projected_receipt(child, blake_hex, sha_hex)),
        _ => false,
    }
}

fn collect_log_blobs(dir: &Path, out: &mut Vec<Vec<u8>>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_log_blobs(&path, out);
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.len() > 32 * 1024 * 1024 {
            continue;
        }
        if let Ok(bytes) = fs::read(&path) {
            out.push(bytes);
        }
    }
}

fn window_has_projected_receipt(
    bytes: &[u8],
    blake: &[u8; 32],
    sha: &[u8],
    blake_hex: &str,
    sha_hex: &str,
) -> bool {
    let needle = b"projected_to";
    let mut from = 0;
    while let Some(rel) = find_bytes(&bytes[from..], needle) {
        let at = from + rel;
        let start = at.saturating_sub(8192);
        let end = (at + needle.len() + 8192).min(bytes.len());
        let window = &bytes[start..end];
        let hash = window.windows(32).any(|w| w == blake)
            || window.windows(sha.len()).any(|w| w == sha)
            || String::from_utf8_lossy(window).contains(blake_hex)
            || String::from_utf8_lossy(window).contains(sha_hex);
        let gated = [
            b"receipt".as_slice(),
            b"verdict",
            b"admission",
            b"Allow",
            b"Hold",
        ]
        .iter()
        .any(|token| find_bytes(window, token).is_some());
        if hash && gated {
            return true;
        }
        from = at + needle.len();
    }
    false
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
