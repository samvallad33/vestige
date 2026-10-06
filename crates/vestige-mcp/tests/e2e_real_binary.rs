//! Real-binary end-to-end regression suite.
//!
//! # Why this file exists
//!
//! This file drives the shipped executable directly, speaking line-framed
//! JSON-RPC over its stdin/stdout, exactly as an MCP client does. Every
//! defect worth catching at the seam between "the code is correct" and
//! "the product works" lives in that gap: process startup, SQLite
//! migration against a real file, integrity repair on a damaged store,
//! and JSON-RPC framing over a real pipe.
//!
//! # Running it
//!
//! ```sh
//! cargo test -p vestige-mcp --test e2e_real_binary
//! ```
//!
//! # Keyword-path only
//!
//! The embedding/vector runtime is being removed from the product. Every
//! test here exercises the keyword/FTS/BM25 retrieval path, SQLite
//! integrity, migration, JSON-RPC framing, purge, suppression, and
//! durability — none of them depend on vector machinery. Tests that
//! previously required the real (~670 MB) embedding model were removed
//! with it; the contradiction and correction regressions were converted
//! to the keyword path because their subjects (retrieval-side
//! contradiction protection, the ingest gate's shared contradiction
//! detector) are lexical by construction: the two sides of a
//! contradiction share nearly every token.

// The process harness (Server, store helpers, payload assertions) lives in
// `common/mod.rs`, shared verbatim with `e2e_failure_cases.rs` so both suites
// drive the binary through one set of helpers.
mod common;

use std::collections::HashMap;
use std::time::Duration;

use rusqlite::Connection;
use serde_json::json;

use common::*;

// ============================================================================
// 1. Protocol surface
// ============================================================================

/// `server/discover` must answer with no handshake at all, and must advertise
/// exactly the revisions the server serves — no more, no fewer.
///
/// Catches: gating discovery behind `initialize` (which makes it useless, since
/// its entire purpose is to precede the handshake), and advertising a revision
/// the server does not implement — a false version claim fails conformance for
/// real. Since #241 the server implements BOTH eras, so the modern revision
/// must be named; a client that cannot speak it still negotiates down to
/// `2025-11-25` through the handshake.
// Re-lands with the strata runtime boot: this journey starts from an
// EMPTY data dir, and a guard-armed 4.0 binary creates no SQLite store
// (audit blocker 1) — there is nothing to boot until strata lands.
#[test]
fn discover_answers_before_any_handshake_and_does_not_overclaim() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());

    // Deliberately no initialize.
    let result = server.result("server/discover", None);

    // `DiscoverResult` shape: identity under `_meta`, cache hints, resultType.
    // The first version of this handler invented `protocolVersions` and
    // `serverInfo`, and a conforming client read "no revisions offered" (#175).
    let identity = &result["_meta"]["io.modelcontextprotocol/serverInfo"];
    assert_eq!(identity["name"], json!("vestige"));
    assert!(
        identity["version"].is_string(),
        "discover must report a version: {result}"
    );
    assert_eq!(result["resultType"], json!("complete"));
    assert_eq!(result["cacheScope"], json!("public"));
    assert!(
        result["ttlMs"].as_u64().is_some(),
        "ttlMs must be a non-negative integer: {result}"
    );
    assert_eq!(result["capabilities"]["tools"]["listChanged"], json!(false));

    let versions: Vec<String> = result["supportedVersions"]
        .as_array()
        .expect("supportedVersions array")
        .iter()
        .map(|v| v.as_str().expect("version string").to_string())
        .collect();
    assert!(
        versions.contains(&"2025-11-25".to_string()),
        "discover must advertise the revision legacy clients negotiate down to: {versions:?}"
    );
    assert!(
        versions.contains(&"2026-07-28".to_string()),
        "#241: the server implements the 2026-07-28 stateless core, resultType \
         and the receipt-card app, so it must advertise it: {versions:?}"
    );
    // One writer per log. Release this process before the probes open the
    // same directory.
    server.shutdown();
    // Every LEGACY advertised revision must be one initialize will actually
    // accept. The modern revision is deliberately absent from the handshake:
    // a 2026-07-28 client never shakes hands at all, it sends per-request
    // `_meta` and the server serves it statelessly.
    for version in versions.iter().filter(|v| v.as_str() != "2026-07-28") {
        let mut probe = Server::spawn(dir.path());
        let negotiated = probe.result(
            "initialize",
            Some(json!({
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": { "name": "probe", "version": "1" },
            })),
        );
        assert_eq!(
            negotiated["protocolVersion"],
            json!(version),
            "discover advertised {version} but initialize negotiated something else"
        );
        probe.shutdown();
    }
}

/// Everything except `initialize` and `server/discover` must be refused before
/// the handshake, and refused cleanly rather than by panicking.
#[test]
fn uninitialized_requests_are_refused_but_discover_is_exempt() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());

    for method in [
        "tools/list",
        "resources/list",
        "resources/templates/list",
        "ping",
    ] {
        let error = server.error(method, None);
        assert_eq!(
            error["code"],
            json!(-32003),
            "{method} before initialize must be 'not initialized': {error}"
        );
    }
    assert!(
        server.result("server/discover", None)["supportedVersions"].is_array(),
        "server/discover must remain callable before initialize"
    );

    server.handshake();
    assert!(server.result("tools/list", None)["tools"].is_array());

    server.shutdown();
}

/// `resources/templates/list` belongs to the `resources` capability the server
/// declares, so it must answer rather than method-not-found. Since #241 it
/// advertises the MCP App receipt-card template (`ui://vestige/receipt/{id}`),
/// and an unknown pagination cursor on any list method must be refused with
/// `-32602` instead of being answered with page one.
///
/// Catches: a client that believes it is paginating looping on page one forever,
/// and a conformance suite that cannot verify the templates surface at all while
/// it errors (#175).
#[test]
fn list_methods_reject_unknown_cursors_and_templates_list_is_empty() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let templates = server.result("resources/templates/list", None);
    let template_list = templates["resourceTemplates"].as_array().expect("array");
    assert!(
        template_list
            .iter()
            .any(|t| t["uriTemplate"] == json!("ui://vestige/receipt/{id}")),
        "the receipt-card app template must be advertised: {template_list:?}"
    );
    assert!(
        template_list
            .iter()
            .all(|t| t["mimeType"] == json!("text/html;profile=mcp-app")),
        "every ui:// template carries the SEP-1865 app MIME type: {template_list:?}"
    );

    for method in ["tools/list", "resources/list", "resources/templates/list"] {
        let error = server.error(method, Some(json!({ "cursor": "not-one-we-issued" })));
        assert_eq!(
            error["code"],
            json!(-32602),
            "{method} with an unknown cursor must be invalid params: {error}"
        );
        // Absent-equivalent cursors are not an error.
        assert!(
            server
                .result(method, Some(json!({ "cursor": null })))
                .is_object()
        );
        assert!(
            server
                .result(method, Some(json!({ "cursor": "" })))
                .is_object()
        );
    }

    server.shutdown();
}

/// `tools/list` must be byte-for-byte identical across independent server
/// processes, and must carry the `CacheableResult` freshness hints.
///
/// Catches: a hand-ordered tool vec leaking into the wire order (any reorder
/// silently busts every client's prompt cache and re-sends ~28 KB of schema on
/// every session start), and a dropped `ttlMs`/`cacheScope`, which leaves the
/// client no way to know it could have kept its copy.
#[test]
fn tools_list_is_deterministic_across_restarts_and_carries_cache_hints() {
    let first_dir = data_dir();
    let second_dir = data_dir();

    let mut first = Server::spawn(first_dir.path());
    first.handshake();
    let a = first.result("tools/list", None);
    first.shutdown();

    let mut second = Server::spawn(second_dir.path());
    second.handshake();
    let b = second.result("tools/list", None);
    second.shutdown();

    assert_eq!(
        a, b,
        "tools/list must be identical across processes so clients can cache it"
    );

    assert_eq!(
        a["ttlMs"],
        json!(3_600_000u64),
        "missing ttlMs freshness hint"
    );
    assert_eq!(
        a["cacheScope"],
        json!("private"),
        "the tool list can vary per install, so it must not be shared-cacheable"
    );

    let names: Vec<String> = a["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name").to_string())
        .collect();
    assert!(!names.is_empty(), "tools/list returned nothing");
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        names, sorted,
        "tools must be emitted in a stable sorted order, got {names:?}"
    );
    let mut seen = HashMap::new();
    for name in &names {
        assert!(
            seen.insert(name.clone(), ()).is_none(),
            "duplicate tool advertised: {name}"
        );
    }
    // purge is withheld on Strata in 4.0 (not advertised).
    assert!(
        !names.contains(&"purge".to_string()),
        "withheld tool advertised on Strata: {names:?}"
    );
    for required in ["recall", "smart_ingest", "memory", "suppress"] {
        assert!(
            names.contains(&required.to_string()),
            "advertised surface lost {required}: {names:?}"
        );
    }

    // High-payload tools must keep their truncation override, or large results
    // get silently clipped at the client's 50K default and spilled to disk.
    let recall = a["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == json!("recall"))
        .expect("recall tool");
    assert_eq!(
        recall["_meta"]["anthropic/maxResultSizeChars"],
        json!(300_000),
        "recall lost its result-size annotation: {recall}"
    );

    // The expanded v3 catalog includes projection, the intention graph and
    // complete maintenance actions. Its integrated baseline is 52,988 bytes.
    // Preserve a bounded full catalog and separately bound the common subset
    // used by clients with v3 progressive discovery.
    let bytes = serde_json::to_string(&a).unwrap().len();
    assert!(
        bytes <= 55_000,
        "tools/list is {bytes} bytes, over the 55,000 byte v3 ceiling; a schema or description grew"
    );

    let common: Vec<_> = a["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|tool| {
            ["recall", "smart_ingest", "memory"].contains(&tool["name"].as_str().unwrap())
        })
        .collect();
    assert_eq!(common.len(), 3);
    assert!(
        serde_json::to_vec(&common).unwrap().len() <= 13_000,
        "common progressive tool subset exceeded its 13KB budget"
    );

    // Behaviour hints reach the client in MCP's camelCase shape, and the two
    // hints a client acts on (read-only, destructive) are set for every tool.
    for tool in a["tools"].as_array().unwrap() {
        let name = tool["name"].as_str().unwrap();
        assert!(tool["title"].is_string(), "{name} has no title on the wire");
        let ann = &tool["annotations"];
        assert!(ann["readOnlyHint"].is_boolean(), "{name}: {ann}");
        assert!(ann["destructiveHint"].is_boolean(), "{name}: {ann}");
        assert!(ann["idempotentHint"].is_boolean(), "{name}: {ann}");
        assert!(ann["openWorldHint"].is_boolean(), "{name}: {ann}");
    }
    // Reason mode records composition evidence; hints describe the whole tool.
    assert_eq!(recall["annotations"]["readOnlyHint"], json!(false));
    let memory = a["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == json!("memory"))
        .expect("memory tool");
    assert_eq!(memory["annotations"]["destructiveHint"], json!(true));
}

/// Malformed, unknown, oversized and structurally wrong input must all produce
/// clean JSON-RPC errors, and the server must stay alive and in sync.
///
/// Catches: a panic or a hang on hostile input, and — via the trailing `ping` —
/// a framing desync where one bad message shifts every later response onto the
/// wrong request id.
#[test]
fn hostile_input_produces_clean_errors_without_panicking_or_desyncing() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    // Truncated JSON and outright garbage: parse error, no id.
    for bad in [
        r#"{"jsonrpc":"2.0","id":99,"method":"#,
        "this is not json at all",
        "{",
        "[]",
    ] {
        let response = server.raw_roundtrip(bad);
        assert_eq!(
            response["error"]["code"],
            json!(-32700),
            "expected a parse error for {bad:?}, got {response}"
        );
    }

    assert_eq!(
        server.error("nonexistent/method", None)["code"],
        json!(-32601),
        "unknown method must be -32601"
    );
    assert_eq!(
        server.error("tools/call", None)["code"],
        json!(-32602),
        "tools/call with no params must be -32602"
    );
    assert_eq!(
        server.error(
            "tools/call",
            Some(json!({ "name": "no_such_tool", "arguments": {} }))
        )["code"],
        json!(-32602),
        "unknown tool must be -32602"
    );
    assert_eq!(
        server.error(
            "tools/call",
            Some(json!({ "name": "recall", "arguments": "not-an-object" })),
        )["code"],
        json!(-32602),
        "non-object arguments must be rejected, not coerced"
    );

    // Missing a required tool argument is a tool-level error, not a crash.
    let missing = server.call_tool("recall", json!({}));
    assert!(
        missing["error"].is_string(),
        "recall without `query` must report a tool error: {missing}"
    );

    // Oversized input must be bounded, not swallowed into the store.
    let oversized = server.call_tool("smart_ingest", json!({ "content": "x".repeat(2_000_000) }));
    assert!(
        oversized["error"]
            .as_str()
            .is_some_and(|e| e.contains("too large")),
        "a 2 MB payload must be refused with a size error: {oversized}"
    );

    // Still alive, still in sync.
    assert_eq!(server.result("ping", None), json!({}));
    assert!(
        server.error_lines().is_empty(),
        "hostile input must not log server errors: {:?}",
        server.error_lines()
    );

    server.shutdown();
}

/// Blank lines and notifications produce no output, and must not shift the
/// response stream.
///
/// Catches: a transport that answers a notification (which would leave the
/// client one response ahead forever) or that treats a blank keepalive line as
/// a message.
#[test]
fn blank_lines_and_notifications_produce_no_response() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    server.write_line("");
    server.write_line("   ");
    server.notify("notifications/cancelled", Some(json!({ "requestId": 1 })));
    server.expect_silence(Duration::from_millis(400));

    // The next real request must still get its own id back.
    assert_eq!(server.result("ping", None), json!({}));

    // A notification sent WITH an id is a protocol violation and must be told so.
    let error = server.error("notifications/initialized", None);
    assert_eq!(
        error["code"],
        json!(-32600),
        "expected invalid request: {error}"
    );

    assert_eq!(server.result("ping", None), json!({}));
    server.shutdown();
}

// ============================================================================
// 2. Store integrity: the store must not be brickable
// ============================================================================

// ============================================================================
// 3. Retrieval correctness (embedding-independent paths)
// ============================================================================

// ============================================================================
// 4. Deletion, suppression and the review gate
// ============================================================================

// ============================================================================
// 5. Contradiction and correction handling on the keyword path
//
// These two regressions used to require the real embedding runtime, but their
// subjects are lexical by construction: the two sides of a contradiction
// share nearly every token, so keyword retrieval reaches them without any
// vector machinery. They run in the default suite now.
// ============================================================================

#[test]
fn memory_status_every_view_has_its_shape_and_an_unknown_view_errors() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.ingest_keyword_only("A memory so the store is not empty", &["e2e"]);

    let health = server.call_tool_ok("memory_status", json!({ "view": "health" }));
    assert_keys(
        &health,
        &["cognitiveHealth", "averageRetention"],
        "memory_status health",
    );
    assert_under(&health, 8_000, "memory_status health");

    let stats = server.call_tool_ok("memory_status", json!({ "view": "stats" }));
    assert_keys(
        &stats,
        &["counts", "lifecycle", "retentionDistribution", "population"],
        "stats",
    );
    assert_under(&stats, 24_000, "memory_status stats");

    let timeline = server.call_tool_ok("memory_status", json!({ "view": "timeline" }));
    assert_keys(
        &timeline,
        &["days", "timeline", "totalMemories"],
        "timeline",
    );
    assert_eq!(timeline["totalMemories"], json!(1));
    assert_under(&timeline, 4_000, "memory_status timeline");

    let changelog = server.call_tool_ok("memory_status", json!({ "view": "changelog" }));
    assert_keys(&changelog, &["events", "totalEvents"], "changelog");
    assert_under(&changelog, 4_000, "memory_status changelog");

    let retention = server.call_tool_ok("memory_status", json!({ "view": "retention" }));
    assert_keys(
        &retention,
        &["avgRetention", "distribution", "trend", "totalMemories"],
        "retention",
    );
    assert_under(&retention, 3_000, "memory_status retention");

    let bad = server.call_tool("memory_status", json!({ "view": "weather" }));
    assert!(bad.get("error").is_some(), "unknown view must error: {bad}");
    server.shutdown();
}

#[test]
fn maintain_scores_importance_dry_runs_gc_consolidates_and_restore_needs_a_path() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.ingest_keyword_only("A memory for the maintenance pass to touch", &["e2e"]);

    let score = server.call_tool_ok(
        "maintain",
        json!({ "action": "importance_score", "content": "the cache key rotation broke production" }),
    );
    assert_keys(
        &score,
        &["composite", "channels", "dominantSignal"],
        "maintain importance_score",
    );
    assert_under(&score, 6_000, "maintain importance_score");

    let gc = server.call_tool_ok("maintain", json!({ "action": "gc" }));
    assert_keys(
        &gc,
        &[
            "dryRun",
            "candidateCount",
            "processed",
            "hasMore",
            "nextCursor",
            "atomic",
        ],
        "maintain gc",
    );
    assert_eq!(
        gc["dryRun"],
        json!(true),
        "gc must default to a dry run: {gc}"
    );
    assert_eq!(gc["atomic"], json!(true));
    assert!(gc["processed"].as_u64().unwrap() <= 100);
    assert_under(&gc, 3_000, "maintain gc");

    // Every consolidate phase is a no-op on a Strata log, so the action is
    // withheld rather than answered with zero counts that read as a pass.
    let consolidate = server.call_tool("maintain", json!({ "action": "consolidate" }));
    assert_error_mentions(&consolidate, "unavailable_in_4_0", "maintain consolidate");
    assert!(
        consolidate.get("nodesProcessed").is_none(),
        "a withheld action must not carry counts: {consolidate}"
    );

    let bad = server.call_tool("maintain", json!({ "action": "restore" }));
    assert!(
        bad.get("error").is_some(),
        "restore without a path must error: {bad}"
    );
    server.shutdown();
}

/// `maintain(action=dream)` on a Strata log consolidates recorded edges and
/// FSRS state. Shared wording across memories does not create a review, and
/// an edge under the strength floor does not either.
#[test]
fn dream_over_stdio_completes_from_recorded_edges_and_fsrs() {
    let dir = data_dir();
    let (strong, quiet, reps_before) = {
        let storage = vestige_mcp::strata_memory::open(dir.path()).expect("seed strata log");
        let mut ids = Vec::new();
        for i in 0..6 {
            let node = storage
                .ingest(vestige_core::IngestInput {
                    content: format!("shared token salad {i} alpha beta gamma"),
                    ..vestige_core::IngestInput::default()
                })
                .unwrap();
            ids.push(node.id);
        }
        let now = storage.get_node(&ids[0]).unwrap().unwrap().created_at;
        let link = |source: &str, target: &str, kind: &str, strength: f64| {
            storage
                .save_connection(&vestige_core::ConnectionRecord {
                    source_id: source.to_string(),
                    target_id: target.to_string(),
                    strength,
                    link_type: kind.to_string(),
                    created_at: now,
                    last_activated: now,
                    activation_count: 1,
                })
                .unwrap();
        };
        link(&ids[0], &ids[1], "derived_from", 1.0);
        link(&ids[1], &ids[2], "evidence_of", 0.8);
        link(&ids[3], &ids[4], "touched", 0.1);
        let reps = storage.get_node(&ids[0]).unwrap().unwrap().reps;
        (
            vec![ids[0].clone(), ids[1].clone(), ids[2].clone()],
            ids[5].clone(),
            reps,
        )
    };

    let mut server = Server::spawn(dir.path());
    server.handshake();
    let value = server.call_tool_ok("maintain", json!({ "action": "dream" }));
    assert_eq!(value["status"], json!("completed"), "{value}");
    assert_eq!(value["basis"], json!("recorded_edges_fsrs"), "{value}");
    assert_eq!(value["edgesConsolidated"], json!(2), "{value}");
    assert_eq!(value["connectionsPersisted"], json!(0), "{value}");
    let text = serde_json::to_string(&value).unwrap();
    assert!(!text.contains("pending_strata"), "{value}");
    assert!(!text.contains("not implemented"), "{value}");
    assert!(!text.contains("token salad"), "{value}");
    let mut reviewed: Vec<String> = value["reviews"]
        .as_array()
        .unwrap()
        .iter()
        .map(|review| review["id"].as_str().unwrap().to_string())
        .collect();
    let mut expected = strong.clone();
    expected.sort();
    reviewed.sort();
    assert_eq!(reviewed, expected, "{value}");

    let strengthened = server.call_tool_ok("memory", json!({ "action": "get", "id": strong[0] }));
    assert_eq!(
        strengthened["node"]["reps"],
        json!(reps_before + 1),
        "{strengthened}"
    );
    let untouched = server.call_tool_ok("memory", json!({ "action": "get", "id": quiet }));
    assert_eq!(untouched["node"]["reps"], json!(reps_before), "{untouched}");
    server.shutdown();
}

#[test]
fn codebase_remembers_a_decision_returns_context_verifies_and_needs_its_fields() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let remembered = server.call_tool_ok(
        "codebase",
        json!({
            "action": "remember_decision",
            "codebase": "e2e-probe",
            "decision": "Use content-hashed cache keys for build artifacts",
            "rationale": "stale keys shipped a broken build",
            "files": ["src/cache.rs"]
        }),
    );
    assert!(remembered.get("error").is_none(), "{remembered}");
    assert_under(&remembered, 4_000, "codebase remember_decision");

    let context = server.call_tool_ok(
        "codebase",
        json!({ "action": "get_context", "codebase": "e2e-probe" }),
    );
    assert_keys(
        &context,
        &["decisions", "patterns", "staleMemories"],
        "codebase get_context",
    );
    // `decisions` is `{ count, items }`; the anchor to src/cache.rs is reported
    // missing because that file does not exist here, which is the honest answer.
    assert_eq!(
        context["decisions"]["count"],
        json!(1),
        "the decision must come back in context: {context}"
    );
    assert_under(&context, 6_000, "codebase get_context");

    let checkout = tempfile::tempdir().unwrap();
    let verify = server.call_tool_ok(
        "codebase",
        json!({ "action": "verify", "codebase": "e2e-probe", "repoPath": checkout.path() }),
    );
    assert_keys(
        &verify,
        &["checked", "fresh", "stale", "unverifiable"],
        "codebase verify",
    );
    assert_under(&verify, 6_000, "codebase verify");

    let bad = server.call_tool("codebase", json!({ "action": "remember_decision" }));
    assert_error_mentions(&bad, "decision", "remember_decision without a decision");
    server.shutdown();
}

/// `source_sync` reaches an external system, so it has no offline happy path.
/// Both of its tests are error paths by design; the connector itself is
/// covered by the unit tests in `vestige-core`.
#[cfg(feature = "connectors")]
#[test]
fn source_sync_rejects_an_unknown_source_and_a_missing_repo_without_touching_the_network() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let unknown = server.call_tool("source_sync", json!({ "source": "gitlab", "repo": "a/b" }));
    assert!(
        unknown.get("error").is_some(),
        "unknown source must error: {unknown}"
    );

    let missing = server.call_tool("source_sync", json!({ "source": "github" }));
    assert!(
        missing.get("error").is_some(),
        "github without repo must error: {missing}"
    );
    server.shutdown();
}

#[test]
fn smart_ingest_and_suppress_reject_calls_without_their_subject() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let no_content = server.call_tool("smart_ingest", json!({ "tags": ["orphan"] }));
    assert!(
        no_content.get("error").is_some(),
        "smart_ingest without content must error: {no_content}"
    );

    let no_id = server.call_tool("suppress", json!({ "reason": "no subject" }));
    assert!(
        no_id.get("error").is_some(),
        "suppress without id must error: {no_id}"
    );
    server.shutdown();
}

/// GhostLink over stdio on a fresh 4.0 store: the bridge lens explains an
/// empty answer, and every divergent candidate carries its no-edge proof.
#[test]
fn ghostlink_propose_answers_both_lenses_with_proofs() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    for content in [
        "GhostLink e2e: refunds need multi-row transactions",
        "GhostLink e2e: the March outage began in the retry loop",
        "GhostLink e2e: invoices are rendered by the PDF worker",
    ] {
        server.call_tool_ok("smart_ingest", json!({ "content": content }));
    }
    let bridge = server.call_tool_ok("ghostlink", json!({ "mode": "propose", "limit": 5 }));
    assert_eq!(bridge["lens"], json!("bridge"), "{bridge}");
    assert_eq!(bridge["globalNoveltyVerified"], json!(false), "{bridge}");
    assert!(
        bridge["admission"].is_object(),
        "a proposal says what admits a pair: {bridge}"
    );
    let divergent = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 5 }),
    );
    let candidates = divergent["candidates"].as_array().expect("candidates");
    assert!(!candidates.is_empty(), "{divergent}");
    for candidate in candidates {
        assert_eq!(
            candidate["proof"]["noEdgeVerified"],
            json!(true),
            "{candidate}"
        );
    }
    server.shutdown();
}

/// The guard: every tool the server advertises has at least two calls in this
/// file. Adding a tool without driving it over stdio fails here, not in a
/// user's client.
#[test]
fn every_advertised_tool_is_called_at_least_twice_in_this_suite() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    let list = server.result("tools/list", None);
    server.shutdown();

    // rustfmt wraps long calls onto several lines, so count on a copy with the
    // whitespace removed.
    let source: String = include_str!("e2e_real_binary.rs")
        .split_whitespace()
        .collect();
    let mut thin = Vec::new();
    for tool in list["tools"].as_array().expect("tools array") {
        let name = tool["name"].as_str().expect("tool name");
        let calls = source.matches(&format!("call_tool(\"{name}\"")).count()
            + source.matches(&format!("call_tool_ok(\"{name}\"")).count()
            + source.matches(&format!("\"name\":\"{name}\"")).count();
        if calls < 2 {
            thin.push(format!("{name} ({calls})"));
        }
    }
    assert!(
        thin.is_empty(),
        "advertised tools with fewer than two e2e calls: {thin:?}"
    );
}
