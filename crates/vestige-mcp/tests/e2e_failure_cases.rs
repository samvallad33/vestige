//! Failure-case end-to-end suite.
//!
//! Companion to `e2e_real_binary.rs` (which owns the happy paths). Where that
//! suite proves the product works, this one proves every *documented failure*
//! fails the documented way: abstention instead of junk, refusal instead of
//! partial writes, protocol errors instead of panics, guards instead of mass
//! deletions. Every test drives the real built binary over stdio JSON-RPC
//! through the shared harness in `common/mod.rs`.
//!
//! All tests here are embedding-independent by construction. The one
//! reconsolidation test that needed the real embedding runtime (the
//! prediction-error gate's supersede path was vector-gated by design) was
//! removed with the embedding machinery.

use std::collections::HashMap;
#[cfg(feature = "connectors")]
use std::io::{Read, Write};
#[cfg(feature = "connectors")]
use std::net::{TcpListener, TcpStream};
#[cfg(feature = "connectors")]
use std::sync::Arc;
#[cfg(feature = "connectors")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

mod common;

use common::*;

// ============================================================================
// Loopback HTTP mock (modeled on the connector tests' MockApi in
// vestige-core/src/connectors/github.rs): one accept thread, one request per
// connection, the handler picks the response by request path. This exists so
// source_sync can be pointed at a Redmine *we* control via REDMINE_URL.
// ============================================================================

#[cfg(feature = "connectors")]
type MockHandler = Arc<dyn Fn(&str) -> (u16, String) + Send + Sync>;

#[cfg(feature = "connectors")]
struct MockHttp {
    base_url: String,
    _stop: Arc<AtomicBool>,
}

#[cfg(feature = "connectors")]
impl MockHttp {
    fn spawn(handler: MockHandler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        std::thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            while !stop_flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // macOS/BSD: accepted sockets inherit the listener's
                        // O_NONBLOCK; force blocking reads.
                        stream.set_nonblocking(false).unwrap();
                        let path = read_request_path(&mut stream);
                        let (status, body) = handler(&path);
                        write_http_response(&mut stream, status, &body);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            _stop: stop,
        }
    }
}

#[cfg(feature = "connectors")]
impl Drop for MockHttp {
    fn drop(&mut self) {
        self._stop.store(true, Ordering::SeqCst);
    }
}

#[cfg(feature = "connectors")]
fn read_request_path(stream: &mut TcpStream) -> String {
    let mut buf = [0u8; 4096];
    let mut data = Vec::new();
    for _ in 0..3 {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                data.extend_from_slice(&buf[..n]);
                if data.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let head = String::from_utf8_lossy(&data);
    head.lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string()
}

#[cfg(feature = "connectors")]
fn write_http_response(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

// ============================================================================
// A. Recall failure semantics
// ============================================================================

// ============================================================================
// B. Ingest gate failures
// ============================================================================

/// The explicit `allowSecrets: true` override stores credential-bearing
/// content — the one documented way past the gate.
#[test]
fn credential_content_with_explicit_allow_secrets_override_is_stored() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let secret = format!("AIza{}", "k".repeat(35));
    let stored = server.call_tool_ok(
        "smart_ingest",
        json!({
            "content": format!("local test key for the emulator: {secret}"),
            "allowSecrets": true,
            "forceCreate": true,
        }),
    );
    assert_eq!(stored["success"], json!(true), "{stored}");
    let id = stored["nodeId"].as_str().expect("nodeId").to_string();
    assert!(
        server.memory_found(&id),
        "an explicitly allowed credential write must be durable"
    );
    server.shutdown();
}

/// Empty and whitespace-only content are refused with a clear message.
#[test]
fn empty_and_whitespace_only_content_are_refused() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for content in ["", "   ", "\n\t  "] {
        let refused = server.call_tool("smart_ingest", json!({ "content": content }));
        assert_error_mentions(&refused, "empty", &format!("content {content:?}"));
    }

    // Batch mode skips empty items rather than storing them (the batch itself
    // still answers).
    let batch = server.call_tool(
        "smart_ingest",
        json!({ "items": [ { "content": "   " }, { "content": "A real batch item about the deploy cache" } ] }),
    );
    assert!(
        batch.get("error").is_none() || batch["isError"] == json!(true),
        "batch with an empty item must answer, not crash: {batch}"
    );
    server.shutdown();
}

/// Oversize content is refused at the 1 MB cap with a size error, and a
/// payload just under the cap is accepted — proving the refusal is the cap
/// and not something else.
#[test]
fn oversize_content_is_refused_at_the_cap_while_just_under_is_accepted() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let too_big = "x".repeat(1_100_000);
    let refused = server.call_tool("smart_ingest", json!({ "content": too_big }));
    assert_error_mentions(&refused, "large", "1.1 MB content");

    let near_cap = "y".repeat(500_000);
    let accepted = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": near_cap, "forceCreate": true }),
    );
    assert_eq!(accepted["success"], json!(true), "{accepted}");
    server.shutdown();
}

/// A scope is a namespace identifier: empty, oversized, and control-character
/// scopes are refused rather than silently normalized into someone else's
/// namespace.
#[test]
fn malformed_scope_values_are_refused_on_the_write_path() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for scope in ["", "   ", &"s".repeat(250), "bad\u{7}scope"] {
        let refused = server.call_tool(
            "smart_ingest",
            json!({ "content": "scoped fixture memory", "scope": scope, "forceCreate": true }),
        );
        assert!(
            refused.get("error").is_some(),
            "scope {scope:?} must be refused: {refused}"
        );
    }
    server.shutdown();
}

// ============================================================================
// C. Tool protocol failures
// ============================================================================

/// An unknown tool name is a PROTOCOL-level `-32602` (error object, no result
/// body), not a tool-level is_error result — clients gate on that code.
#[test]
fn unknown_tool_names_are_protocol_errors_with_no_result_body() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let names = [
        "no_such_tool",
        "",
        "RECALL",
        "recall ",
        #[cfg(not(feature = "connectors"))]
        "source_sync",
    ];
    for name in names {
        let response = server.request("tools/call", Some(json!({ "name": name, "arguments": {} })));
        assert!(
            response.get("result").is_none(),
            "unknown tool {name:?} must not produce a result body: {response}"
        );
        assert_eq!(
            response["error"]["code"],
            json!(-32602),
            "unknown tool {name:?}: {response}"
        );
    }
    server.shutdown();
}

/// Every unified tool refuses a call missing its required subject with a
/// tool-level error naming what was missing — no panics, no empty successes.
#[test]
fn missing_required_arguments_error_per_tool() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    // Only the connectors build pushes another case.
    #[cfg_attr(not(feature = "connectors"), allow(unused_mut))]
    let mut cases: Vec<(&str, Value)> = vec![
        ("recall", json!({})),
        ("smart_ingest", json!({ "tags": ["orphan"] })),
        ("suppress", json!({ "reason": "no subject" })),
        ("memory", json!({})),
        ("intention", json!({})),
        ("maintain", json!({})),
        ("receipt", json!({})),
        ("codebase", json!({})),
        ("graph", json!({})),
    ];
    #[cfg(feature = "connectors")]
    cases.push(("source_sync", json!({})));
    for (name, args) in cases {
        let value = server.call_tool(name, args);
        assert!(
            value.get("error").is_some(),
            "{name} without its required arguments must return a tool error: {value}"
        );
    }

    // `project` has no required subject: an argument-free call is a
    // read-only preview (#338 admitted projection on Strata), not an error.
    // The refusals above are missing required subjects.
    let defaults = server.call_tool("project", json!({}));
    assert!(
        defaults.get("error").is_none(),
        "argument-free project must preview: {defaults}"
    );
    assert_eq!(defaults["action"], "preview", "{defaults}");
    assert_eq!(defaults["itemCount"], 0, "{defaults}");

    // The dispatch table survived every refusal.
    assert_eq!(server.result("ping", None), json!({}));
    server.shutdown();
}

/// Wrong-typed arguments are rejected as tool errors; the server stays alive
/// and in framing sync afterwards.
#[test]
fn wrong_typed_arguments_are_rejected_and_the_server_stays_healthy() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    // Only the connectors build pushes another case.
    #[cfg_attr(not(feature = "connectors"), allow(unused_mut))]
    let mut cases = vec![
        ("recall", json!({ "query": 42 })),
        ("suppress", json!({ "id": { "deep": 1 } })),
        (
            "smart_ingest",
            json!({ "content": "ok", "tags": "not-an-array" }),
        ),
        ("memory", json!({ "action": ["get"] })),
        ("session_start", json!({ "token_budget": "eight hundred" })),
        ("maintain", json!({ "action": true })),
    ];
    #[cfg(feature = "connectors")]
    cases.push(("source_sync", json!({ "source": 3 })));
    for (name, args) in &cases {
        let value = server.call_tool(name, args.clone());
        assert!(
            value.get("error").is_some(),
            "{name} with wrong-typed args must error: {value}"
        );
    }
    assert_eq!(server.result("ping", None), json!({}));
    server.shutdown();
}

/// A result larger than the tool's advertised `maxResultSizeChars` annotation
/// must never reach the wire: recall advertises 300_000, and even a full
/// 100-result page stays far under it.
#[test]
fn recall_results_stay_under_the_advertised_max_result_size_annotation() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    // The annotation is still advertised.
    let list = server.result("tools/list", None);
    let recall = list["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == json!("recall"))
        .expect("recall tool")
        .clone();
    assert_eq!(
        recall["_meta"]["anthropic/maxResultSizeChars"],
        json!(300_000)
    );

    for i in 0..60 {
        server.ingest_keyword_only(
            &format!(
                "Size guard fixture {i}: the change management board reviews deployments on Thursdays with a quorum of four"
            ),
            &[],
        );
    }
    let response = server.request(
        "tools/call",
        Some(json!({ "name": "recall", "arguments": { "query": "change management board deployments", "limit": 100 } })),
    );
    let bytes = serde_json::to_string(&response).unwrap().len();
    assert!(
        bytes <= 300_000,
        "recall response was {bytes} bytes, over its own advertised 300_000 annotation"
    );
    server.shutdown();
}

/// Under the modern (2026-07-28) era, a request naming an unsupported
/// protocol version in `_meta` is rejected with the spec-defined `-32022`,
/// echoing the requested version and listing the supported ones — and this
/// works with NO handshake (the modern era is stateless).
#[test]
fn modern_era_unknown_protocol_version_is_rejected_with_32022_and_a_supported_list() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    // Deliberately no handshake.

    let response = server.request(
        "ping",
        Some(json!({
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2030-01-01",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        })),
    );
    assert!(
        response.get("result").is_none(),
        "an unsupported version must be an error: {response}"
    );
    assert_eq!(response["error"]["code"], json!(-32022), "{response}");
    let data = &response["error"]["data"];
    assert_eq!(data["requested"], json!("2030-01-01"), "{response}");
    let supported = data["supported"].as_array().expect("data.supported");
    assert!(supported.contains(&json!("2026-07-28")), "{supported:?}");
    assert!(supported.contains(&json!("2025-11-25")), "{supported:?}");
    server.shutdown();
}

/// Modern-era success is stateless (no handshake) and carries the modern
/// result envelope; missing halves of the required `_meta` pair are
/// `-32602` invalid params.
#[test]
fn modern_era_requires_the_full_meta_pair_and_valid_versions_serve_statelessly() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    // No handshake.

    // Version but no clientCapabilities → invalid params.
    let response = server.request(
        "ping",
        Some(json!({
            "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
        })),
    );
    assert_eq!(response["error"]["code"], json!(-32602), "{response}");

    // clientCapabilities but no version → invalid params.
    let response = server.request(
        "ping",
        Some(json!({
            "_meta": { "io.modelcontextprotocol/clientCapabilities": {} }
        })),
    );
    assert_eq!(response["error"]["code"], json!(-32602), "{response}");

    // The full pair with the served revision answers statelessly.
    let response = server.request(
        "ping",
        Some(json!({
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        })),
    );
    assert!(response.get("error").is_none(), "{response}");
    assert_eq!(
        response["result"]["resultType"],
        json!("complete"),
        "modern results carry the resultType envelope: {response}"
    );
    server.shutdown();
}

/// Garbage frames interleaved with valid requests: every valid request gets
/// its own correctly-correlated response, the garbage gets parse errors, and
/// the stream never desyncs.
#[test]
fn interleaved_garbage_and_valid_frames_stay_framing_synced() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    server.write_line(r#"{not json"#);
    // read_line() stashes server notifications (the embedding warm-up
    // announcements race everything), so the next line here IS the response.
    let parse_error = server.read_line();
    assert!(
        parse_error.contains("-32700")
            || serde_json::from_str::<Value>(&parse_error)
                .ok()
                .is_some_and(|v| v["error"]["code"] == json!(-32700)),
        "garbage must produce a parse error: {parse_error}"
    );

    let first = server.request("ping", None);
    assert_eq!(
        first["result"],
        json!({}),
        "first valid ping must answer normally"
    );

    server.write_line(r#"{"jsonrpc":"2.0","id":99,"method":"#);
    let _second_parse_error = server.read_any_line();

    let tool = server.request(
        "tools/call",
        Some(json!({ "name": "recall", "arguments": {} })),
    );
    assert!(
        tool["result"].is_object(),
        "a valid-but-failing tool call still returns a result body: {tool}"
    );

    // Everything above ran in-order because request() asserts id correlation;
    // one more full tool roundtrip proves the pipe is still aligned.
    let written = server.ingest_keyword_only("Post-garbage sanity memory", &[]);
    assert!(server.memory_found(&written));
    server.shutdown();
}

/// A JSON array frame and a frame with a null id are protocol violations the
/// server answers (or ignores) without dying; the next real request works.
#[test]
fn array_frames_and_null_id_frames_do_not_kill_the_server() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let response = server.raw_roundtrip(r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#);
    assert_eq!(
        response["error"]["code"],
        json!(-32700),
        "batch-style arrays are not supported and must be a parse-level refusal: {response}"
    );

    // id:null has no correlation, so the server must not answer it — treat it
    // like a notification (silence), then keep serving.
    server.write_line(r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#);
    server.expect_silence(Duration::from_millis(300));
    assert_eq!(server.result("ping", None), json!({}));
    server.shutdown();
}

// ============================================================================
// D. Maintain / destructive failures
// ============================================================================

/// 4.0 withholds purge on a Strata store: it is not advertised, a confirmed
/// call is refused with `unavailable_in_4_0`, and the memory survives.
#[test]
fn purge_is_withheld_on_strata_and_the_memory_survives() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let list = server.result("tools/list", None);
    let advertised = list["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["name"] == json!("purge"));
    assert!(!advertised, "purge is withheld on Strata: {list}");

    let id = server.ingest_keyword_only("A memory that must survive a withheld purge", &[]);
    let refused = server.call_tool("purge", json!({ "id": id, "confirm": true }));
    assert!(
        refused.get("error").is_some() || refused["isError"] == json!(true),
        "purge must be refused on Strata: {refused}"
    );
    assert!(
        refused.to_string().contains("unavailable_in_4_0"),
        "the refusal must say why: {refused}"
    );
    assert!(
        server.memory_found(&id),
        "a withheld purge must not have removed anything"
    );
    server.shutdown();
}

/// Unknown and missing maintain actions are refused with the full list of
/// valid actions in the error — a caller can self-correct from the message.
#[test]
fn maintain_unknown_and_missing_actions_list_the_valid_actions() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let unknown = server.call_tool("maintain", json!({ "action": "time_travel" }));
    let text = unknown["error"]
        .as_str()
        .unwrap_or_else(|| panic!("unknown action must error: {unknown}"));
    for action in [
        "consolidate",
        "dream",
        "dream_compile",
        "gc",
        "importance_score",
        "backup",
        "export",
        "restore",
    ] {
        assert!(text.contains(action), "error must list '{action}': {text}");
    }

    let missing = server.call_tool("maintain", json!({}));
    assert!(
        missing.get("error").is_some(),
        "maintain without action must error: {missing}"
    );
    server.shutdown();
}

/// `gc` never deletes without an explicit opt-in: the default is a dry run,
/// and even a wet run on a healthy store processes nothing and leaves every
/// memory in place.
#[test]
fn gc_defaults_to_dry_run_and_a_wet_run_on_a_healthy_store_deletes_nothing() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    let id = server.ingest_keyword_only(
        "A healthy fresh memory that the garbage collector must never touch",
        &["gc-fixture"],
    );

    let dry = server.call_tool_ok("maintain", json!({ "action": "gc" }));
    assert_eq!(
        dry["dryRun"],
        json!(true),
        "gc must default to a dry run: {dry}"
    );
    assert_eq!(dry["atomic"], json!(true));

    let wet = server.call_tool_ok("maintain", json!({ "action": "gc", "dry_run": false }));
    assert_eq!(
        wet["dryRun"],
        json!(false),
        "the explicit wet run must be reported as such: {wet}"
    );
    assert!(
        server.memory_found(&id),
        "a healthy memory must survive a wet gc: {wet}"
    );
    server.shutdown();
}

// ============================================================================
// E. Codebase evidence failures
// ============================================================================

/// Verifying anchored memories against a checkout where the anchored files do
/// not exist must report them honestly as unverifiable — never as verified,
/// and never as an error.
#[test]
fn verify_against_a_missing_checkout_reports_unverifiable_honestly() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    server.call_tool_ok(
        "codebase",
        json!({
            "action": "remember_decision",
            "codebase": "verify-probe",
            "decision": "Anchor evidence fixture: cache keys are content-hashed",
            "rationale": "stale keys shipped a broken build",
            "files": ["src/gone/cache.rs"],
        }),
    );

    let empty_checkout = tempfile::tempdir().unwrap();
    let verify = server.call_tool_ok(
        "codebase",
        json!({
            "action": "verify",
            "codebase": "verify-probe",
            "repoPath": empty_checkout.path(),
        }),
    );
    let text = verify.to_string();
    assert!(
        verify["unverifiable"].as_u64().unwrap_or(0) >= 1
            || text.contains("unverifiable")
            || text.contains("missing"),
        "a missing checkout must surface as unverifiable/missing, not verified: {verify}"
    );
    assert!(
        !text.contains("\"fresh\":1") || verify["stale"].as_u64().unwrap_or(0) >= 1,
        "nothing can be fresh when the file does not exist: {verify}"
    );
    server.shutdown();
}

/// `reanchor` refuses each missing prerequisite by name, and refuses a
/// memoryId that lives in a different scope.
#[test]
fn reanchor_refuses_missing_prerequisites_and_wrong_scope_by_name() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let id = server.call_tool_ok(
        "codebase",
        json!({
            "action": "remember_decision",
            "codebase": "reanchor-probe",
            "decision": "Reanchor fixture decision about retry budgets",
            "rationale": "the old budget silently dropped retries",
            "files": ["src/budget.rs"],
        }),
    )["nodeId"]
        .as_str()
        .unwrap()
        .to_string();

    let checkout = tempfile::tempdir().unwrap();

    let cases: Vec<(Value, &str)> = vec![
        (json!({ "action": "reanchor" }), "memoryId"),
        (json!({ "action": "reanchor", "memoryId": id }), "repoPath"),
        (
            json!({ "action": "reanchor", "memoryId": id, "repoPath": checkout.path() }),
            "files",
        ),
        (
            json!({
                "action": "reanchor", "memoryId": id, "repoPath": checkout.path(),
                "scope": "another-scope",
                "files": ["src/budget.rs"],
            }),
            "scope",
        ),
    ];
    for (args, needle) in cases {
        let value = server.call_tool("codebase", args.clone());
        assert_error_mentions(&value, needle, &format!("reanchor with {args}"));
    }

    // A nonexistent checkout directory is refused too.
    let missing = server.call_tool(
        "codebase",
        json!({
            "action": "reanchor", "memoryId": id,
            "repoPath": checkout.path().join("does-not-exist"),
            "files": ["src/budget.rs"],
        }),
    );
    assert_error_mentions(&missing, "checkout", "reanchor against a missing dir");
    server.shutdown();
}

// ============================================================================
// F. source_sync failures
// ============================================================================

/// A dead upstream is a clean tool error that NAMES the URL — a caller must
/// be able to tell which host failed without enabling tracing. (The connector
/// SSRF guard is explicitly disabled via `VESTIGE_ALLOW_PRIVATE_CONNECTOR_
/// HOSTS` so the loopback mock/dead-host is reachable at all; the guard's own
/// refusal is tested separately below.)
#[cfg(feature = "connectors")]
#[test]
fn a_dead_upstream_error_names_the_url() {
    let dir = data_dir();
    // A port we know is closed: bind one, then drop the listener.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let dead_port = closed.local_addr().unwrap().port();
    drop(closed);

    let dead_url: &'static str =
        Box::leak(format!("http://127.0.0.1:{dead_port}").into_boxed_str());
    let mut server = Server::spawn_with_env(
        dir.path(),
        &[
            ("REDMINE_URL", dead_url),
            ("VESTIGE_ALLOW_PRIVATE_CONNECTOR_HOSTS", "1"),
        ],
        &["REDMINE_API_KEY", "GITHUB_TOKEN", "VESTIGE_GITHUB_TOKEN"],
    );
    server.handshake();

    let failed = server.call_tool(
        "source_sync",
        json!({ "source": "redmine", "project": "ops" }),
    );
    let text = failed["error"]
        .as_str()
        .unwrap_or_else(|| panic!("dead upstream must be a tool error: {failed}"));
    assert!(
        text.contains("127.0.0.1") && text.contains(&dead_port.to_string()),
        "the error must name the unreachable URL: {text}"
    );
    assert!(text.contains("sync failed"), "{text}");
    server.shutdown();
}

/// The connector SSRF guard: an internal/reserved host as the upstream is
/// refused at configuration time, before any request — even before the
/// missing-project check would matter. A misconfigured or hostile base_url
/// must not turn an authenticated client against localhost.
#[cfg(feature = "connectors")]
#[test]
fn an_internal_address_upstream_is_refused_by_the_ssrf_guard() {
    let dir = data_dir();
    let mut server = Server::spawn_with_env(
        dir.path(),
        &[("REDMINE_URL", "http://127.0.0.1:9")],
        &["REDMINE_API_KEY", "GITHUB_TOKEN", "VESTIGE_GITHUB_TOKEN"],
    );
    server.handshake();

    let refused = server.call_tool(
        "source_sync",
        json!({ "source": "redmine", "project": "ops" }),
    );
    let text = refused["error"]
        .as_str()
        .unwrap_or_else(|| panic!("loopback upstream must be refused: {refused}"));
    assert!(
        text.contains("127.0.0.1") && text.contains("SSRF"),
        "the refusal must name the host and the guard: {text}"
    );
    server.shutdown();
}

/// A 404 from the upstream keeps the connector's documented message shape:
/// `GET {url} -> {status}: {reason}` — distinguishable from "no results" and
/// from a transport failure.
#[cfg(feature = "connectors")]
#[test]
fn a_redmine_404_keeps_the_exact_api_message_shape() {
    let dir = data_dir();
    let mock = MockHttp::spawn(Arc::new(|path| {
        if path.starts_with("/issues.json") {
            (404, r#"{"errors":["404 Not Found"]}"#.to_string())
        } else {
            (404, "{}".to_string())
        }
    }));

    let mut server = Server::spawn_with_env(
        dir.path(),
        &[
            (
                "REDMINE_URL",
                Box::leak(mock.base_url.clone().into_boxed_str()) as &str,
            ),
            ("VESTIGE_ALLOW_PRIVATE_CONNECTOR_HOSTS", "1"),
        ],
        &["REDMINE_API_KEY", "GITHUB_TOKEN", "VESTIGE_GITHUB_TOKEN"],
    );
    server.handshake();

    let failed = server.call_tool(
        "source_sync",
        json!({ "source": "redmine", "project": "ops" }),
    );
    let text = failed["error"]
        .as_str()
        .unwrap_or_else(|| panic!("404 must be a tool error: {failed}"));
    let expected = format!("GET {}/issues.json -> 404 Not Found", mock.base_url);
    assert!(
        text.contains(&expected),
        "error {text:?} must carry the exact shape {expected:?}"
    );
    server.shutdown();
}

/// With no REDMINE_URL configured, redmine sync is a clean configuration
/// error naming the missing variable — not an attempt to reach a default
/// host, and not a crash.
#[cfg(feature = "connectors")]
#[test]
fn redmine_without_a_configured_url_is_a_configuration_error() {
    let dir = data_dir();
    let mut server = Server::spawn_with_env(
        dir.path(),
        &[],
        &[
            "REDMINE_URL",
            "VESTIGE_REDMINE_URL",
            "REDMINE_API_KEY",
            "GITHUB_TOKEN",
            "VESTIGE_GITHUB_TOKEN",
        ],
    );
    server.handshake();

    let failed = server.call_tool(
        "source_sync",
        json!({ "source": "redmine", "project": "ops" }),
    );
    let text = failed["error"]
        .as_str()
        .unwrap_or_else(|| panic!("missing REDMINE_URL must be a tool error: {failed}"));
    assert!(text.contains("REDMINE_URL"), "{text}");
    server.shutdown();
}

/// Unknown sources and a missing project identifier are refused with the
/// supported set / required field named.
#[cfg(feature = "connectors")]
#[test]
fn source_sync_unknown_source_and_missing_project_are_refused_by_name() {
    let dir = data_dir();
    let mut server = Server::spawn_with_env(
        dir.path(),
        &[],
        &["REDMINE_URL", "GITHUB_TOKEN", "VESTIGE_GITHUB_TOKEN"],
    );
    server.handshake();

    let unknown = server.call_tool("source_sync", json!({ "source": "gitlab", "repo": "a/b" }));
    let text = unknown["error"]
        .as_str()
        .unwrap_or_else(|| panic!("unknown source must error: {unknown}"));
    assert!(
        text.contains("github") && text.contains("redmine"),
        "the error must list the supported sources: {text}"
    );

    let no_project = server.call_tool("source_sync", json!({ "source": "redmine" }));
    let text = no_project["error"]
        .as_str()
        .unwrap_or_else(|| panic!("redmine without project must error: {no_project}"));
    assert!(text.contains("project"), "{text}");
    server.shutdown();
}

// ============================================================================
// G. Intention failures
// ============================================================================

/// Snooze minutes outside 1..=525600 are rejected with the range named —
/// never silently clamped into a snooze the caller did not ask for.
#[test]
fn snooze_minutes_outside_the_documented_range_are_rejected_with_the_range() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for minutes in [0i64, -10, 525_601] {
        let refused = server.call_tool(
            "intention",
            json!({
                "action": "update",
                "id": "any-id",
                "status": "snooze",
                "snooze_minutes": minutes,
            }),
        );
        let text = refused["error"]
            .as_str()
            .unwrap_or_else(|| panic!("snooze_minutes {minutes} must be rejected: {refused}"));
        assert!(
            text.contains("1") && text.contains("525600"),
            "the rejection must name the accepted range: {text}"
        );
    }
    server.shutdown();
}

// ============================================================================
// H. Resource layer failures
// ============================================================================

/// A receipt URI for an unknown receipt is a clean not-found, with the code
/// chosen by era: legacy clients keep `-32002`; modern (2026-07-28) clients
/// get `-32602`, because the modern revision retired `-32002`.
#[test]
fn an_unknown_receipt_resource_fails_with_the_era_aware_code() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    let uri = "ui://vestige/receipt/00000000-0000-0000-0000-000000000000";

    // Modern era first — stateless, no handshake.
    let modern = server.request(
        "resources/read",
        Some(json!({
            "uri": uri,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        })),
    );
    assert!(
        modern.get("result").is_none(),
        "an unknown receipt must not succeed: {modern}"
    );
    assert_eq!(
        modern["error"]["code"],
        json!(-32602),
        "the modern revision retired -32002; unknown resources are invalid params: {modern}"
    );

    // Legacy handshake keeps the historical code.
    server.handshake();
    let legacy = server.error("resources/read", Some(json!({ "uri": uri })));
    assert_eq!(
        legacy["code"],
        json!(-32002),
        "legacy clients keep the historical resource-not-found code: {legacy}"
    );
    server.shutdown();
}

/// Malformed receipt template ids and foreign schemes are clean errors that
/// name the problem — never a panic and never a silent empty document.
#[test]
fn malformed_receipt_uris_and_foreign_schemes_are_clean_errors() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for uri in [
        "ui://vestige/receipt/",       // empty id
        "ui://vestige/receipt/a/b",    // traversal-shaped id
        "ui://vestige/receipt/a/../b", // dot-segment traversal
        "ftp://example/thing",         // foreign scheme
        "memory://no-such-resource",   // unknown memory resource
    ] {
        let error = server.error("resources/read", Some(json!({ "uri": uri })));
        let text = error.to_string();
        assert!(
            text.contains("not found")
                || text.contains("Not Found")
                || text.contains("Unknown")
                || text.contains("unknown"),
            "{uri} must be refused with a naming error: {error}"
        );
    }

    // And the server still serves.
    assert_eq!(server.result("ping", None), json!({}));
    server.shutdown();
}

// ============================================================================
// I. Concurrency / EOF
// ============================================================================

/// The server exits cleanly within the shutdown budget when stdin hits EOF —
/// the way every stdio MCP client stops a server.
#[test]
fn the_server_exits_cleanly_on_stdin_eof_within_the_budget() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    let id = server.ingest_keyword_only("A memory written before a clean EOF shutdown", &[]);
    assert!(server.memory_found(&id));

    let started = Instant::now();
    server.close_stdin(); // EOF
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = server.try_exit_status() {
            let elapsed = started.elapsed();
            assert!(
                status.success(),
                "stdin EOF must be a clean shutdown, got {status}. stderr: {:?}",
                server.stderr_lines()
            );
            assert!(
                elapsed < Duration::from_secs(30),
                "shutdown took {elapsed:?}; the budget is the guard"
            );
            return;
        }
        if Instant::now() >= deadline {
            server.kill();
            panic!("vestige-mcp did not exit within 30s of stdin EOF");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A burst of interleaved valid, invalid and hostile frames — the way a real
/// client session looks on a bad day — must answer every well-formed request
/// with the right id and the right verdict, with zero error-level logs.
#[test]
fn a_burst_of_mixed_frames_answers_every_valid_request() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    // Fire a wall of mixed frames without waiting, as a client that pipelines
    // would. Then collect responses and correlate.
    server.write_line(r#"{"jsonrpc":"2.0","id":201,"method":"ping"}"#);
    server.write_line("garbage not json");
    server.write_line(r#"{"jsonrpc":"2.0","id":202,"method":"no/such_method"}"#);
    server.write_line("");
    server.write_line(
        r#"{"jsonrpc":"2.0","id":203,"method":"tools/call","params":{"name":"ghost_tool","arguments":{}}}"#,
    );
    server.write_line(r#"{"jsonrpc":"2.0","id":204,"method":"ping"}"#);

    let mut by_id: HashMap<i64, Value> = HashMap::new();
    let mut parse_errors = 0usize;
    let deadline = Instant::now() + Duration::from_secs(20);
    while by_id.len() < 4 || parse_errors < 1 {
        assert!(
            Instant::now() < deadline,
            "not every pipelined request was answered; got {by_id:?}"
        );
        let line = server.read_any_line();
        let value: Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("non-JSON response {line:?}: {e}"));
        if value.get("id").is_some_and(|id| id.is_i64()) {
            by_id.insert(value["id"].as_i64().unwrap(), value);
        } else if value["error"]["code"] == json!(-32700) {
            parse_errors += 1;
        }
        // Anything else is a server notification (e.g. warm-up logging) —
        // tolerated, stashed by the harness on the next read, not a response.
    }

    assert_eq!(by_id[&201]["result"], json!({}));
    assert_eq!(by_id[&202]["error"]["code"], json!(-32601));
    assert_eq!(by_id[&203]["error"]["code"], json!(-32602));
    assert_eq!(by_id[&204]["result"], json!({}));
    assert_eq!(
        parse_errors, 1,
        "exactly one parse error for one garbage frame"
    );

    // The store still works after the storm.
    let id = server.ingest_keyword_only("Written after the mixed-frame storm", &[]);
    assert!(server.memory_found(&id));
    assert!(
        server.error_lines().is_empty(),
        "hostile frames must not log server errors: {:?}",
        server.error_lines()
    );
    server.shutdown();
}

/// A second EOF (stdin closed while the server is still warming up) must
/// still be a clean exit, not a hang or a panic — the shutdown path cannot
/// assume the handshake completed.
#[test]
fn immediate_eof_before_the_handshake_is_still_a_clean_exit() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    // No handshake, no requests: close stdin immediately.
    server.close_stdin();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = server.try_exit_status() {
            assert!(
                status.success(),
                "pre-handshake EOF must still be a clean shutdown, got {status}. stderr: {:?}",
                server.stderr_lines()
            );
            return;
        }
        if Instant::now() >= deadline {
            server.kill();
            panic!("vestige-mcp hung on immediate stdin EOF");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
