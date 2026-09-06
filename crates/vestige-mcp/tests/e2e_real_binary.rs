//! Real-binary end-to-end regression suite.
//!
//! # Why this file exists
//!
//! The repository has ~281 e2e tests, and they run against mocks
//! (`tests/e2e/src/mocks/mock_embedding.rs`) inside the process. Before this
//! file, exactly one test spawned the shipped executable
//! (`crates/vestige-mcp/tests/stdio_shutdown.rs`). Every defect worth catching
//! at the seam between "the code is correct" and "the product works" lives in
//! that gap: process startup, SQLite migration against a real file, integrity
//! repair on a damaged store, JSON-RPC framing over a real pipe, and the actual
//! embedding runtime being loaded rather than stubbed.
//!
//! Every test here drives `target/<profile>/vestige-mcp` as a child process and
//! speaks line-framed JSON-RPC over its stdin/stdout, exactly as an MCP client
//! does.
//!
//! # Running it
//!
//! ```sh
//! # Fast suite. No model load, no network. This is what CI runs.
//! cargo test -p vestige-mcp --test e2e_real_binary
//!
//! # Full suite, including tests that need the real embedding runtime.
//! cargo test -p vestige-mcp --test e2e_real_binary -- --ignored
//! ```
//!
//! # The embedding trap
//!
//! `smart_ingest` and `recall` behave differently before the ONNX embedding
//! model finishes loading: ingest returns `hasEmbedding: false` and retrieval
//! silently degrades to a keyword-only path. A test that ingests immediately
//! after `initialize` and then asserts on semantic behaviour is measuring the
//! fallback and proves nothing.
//!
//! Rather than sleeping blindly, [`Server::wait_for_embeddings`] blocks on the
//! server's own readiness line on stderr, and [`Server::ingest_embedded`]
//! additionally asserts `hasEmbedding == true` on every write. A test cannot
//! silently drift onto the degraded path: it fails instead.
//!
//! Tests that genuinely need that runtime are `#[ignore]`d, because on a cold
//! machine the first run downloads a ~670 MB model. Tests whose subject is
//! embedding-independent by construction (FTS/BM25 keyword retrieval, SQLite
//! integrity, migration, JSON-RPC framing, purge, suppression, durability) run
//! by default and use [`Server::ingest_keyword_only`], which documents that
//! choice at the call site.
//!
//! # Known product defect documented here
//!
//! [`correction_must_not_be_swallowed_by_the_ingest_gate`] is `#[ignore]`d
//! because it FAILS against the current build. See its doc comment; the defect
//! was not fixed here on purpose.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rusqlite::Connection;
use serde_json::{Value, json};

// ============================================================================
// Harness
// ============================================================================

/// How long to wait for a single JSON-RPC response before declaring a hang.
/// Deliberately finite: "the server hung" must be a test failure, never a
/// wedged CI job.
const RPC_TIMEOUT: Duration = Duration::from_secs(120);

/// How long to wait for the embedding runtime. Generous because the first run
/// on a cold machine downloads the model; warm it is well under a second.
const EMBEDDING_TIMEOUT: Duration = Duration::from_secs(300);

/// The server's own readiness line. Waiting for this is strictly better than
/// sleeping: it is the exact event we care about, and it fails loudly if the
/// runtime never comes up instead of quietly proceeding on the fallback path.
const EMBEDDINGS_READY: &str = "Legacy Nomic embedding service initialized successfully";

/// A running `vestige-mcp` child process plus its stdio plumbing.
///
/// Reader threads drain stdout and stderr so the child can never block on a
/// full pipe, and every read is bounded by a timeout. [`Drop`] kills the child,
/// so a panicking test cannot leak a process.
struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Server {
    /// Spawn the shipped binary against `data_dir`.
    ///
    /// The environment is pinned so a developer's own Vestige configuration can
    /// never leak into a test: no dashboard, no HTTP transport, no inherited
    /// data directory.
    ///
    /// `VESTIGE_AUTOPILOT_ENABLED=0` matters more than it looks. The autopilot
    /// subscribes to `MemoryCreated` and takes a *blocking* `cognitive.lock()`
    /// per event, while `recall`'s retrieval-competition stage takes a
    /// *non-blocking* `try_lock()`. Leaving it on means a burst of ingests
    /// followed immediately by a recall can silently skip that whole stage,
    /// which makes any test of it load-dependent. See the note on
    /// [`contradictions_are_returned_intact_and_flagged_as_protected`].
    fn spawn(data_dir: &Path) -> Self {
        let mut command = Command::new(server_binary());
        command
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn().expect("spawn vestige-mcp");

        let stdin = child.stdin.take().expect("child stdin");
        let raw_stdout = child.stdout.take().expect("child stdout");
        let raw_stderr = child.stderr.take().expect("child stderr");

        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines() {
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

        let stderr = Arc::new(Mutex::new(Vec::new()));
        {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                    if let Ok(mut sink) = sink.lock() {
                        sink.push(line);
                    }
                }
            });
        }

        Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        }
    }

    /// Everything the server has written to stderr so far.
    fn stderr_lines(&self) -> Vec<String> {
        self.stderr.lock().expect("stderr sink").clone()
    }

    /// Every stderr line that the tracing subscriber marked as an error.
    fn error_lines(&self) -> Vec<String> {
        self.stderr_lines()
            .into_iter()
            .filter(|line| line.contains("ERROR"))
            .collect()
    }

    fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| {
                panic!(
                    "writing to vestige-mcp stdin failed ({error}); the server probably died. \
                     stderr: {:?}",
                    self.stderr_lines()
                )
            });
    }

    /// Read one line of output, failing the test rather than blocking forever.
    fn read_line(&mut self) -> String {
        match self.stdout.recv_timeout(RPC_TIMEOUT) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => panic!(
                "vestige-mcp produced no response within {RPC_TIMEOUT:?} (hang). stderr: {:?}",
                self.stderr_lines()
            ),
            Err(RecvTimeoutError::Disconnected) => panic!(
                "vestige-mcp closed stdout without responding (crash?). stderr: {:?}",
                self.stderr_lines()
            ),
        }
    }

    /// Assert that the server sends nothing at all within `window`. Used to
    /// prove that notifications and blank lines produce no response.
    fn expect_silence(&mut self, window: Duration) {
        if let Ok(unexpected) = self.stdout.recv_timeout(window) {
            panic!("expected no response, got: {unexpected}");
        }
    }

    /// Send a raw line and read one response line. For malformed-input tests.
    fn raw_roundtrip(&mut self, line: &str) -> Value {
        self.write_line(line);
        let response = self.read_line();
        serde_json::from_str(&response)
            .unwrap_or_else(|error| panic!("server emitted non-JSON {response:?}: {error}"))
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        let response = self.raw_roundtrip(&message.to_string());
        assert_eq!(
            response["id"],
            json!(id),
            "response id must match the request id (framing desync): {response}"
        );
        assert_eq!(
            response["jsonrpc"],
            json!("2.0"),
            "bad envelope: {response}"
        );
        response
    }

    fn notify(&mut self, method: &str, params: Option<Value>) {
        let mut message = json!({ "jsonrpc": "2.0", "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
    }

    fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("error").is_none(),
            "{method} returned an error: {response}"
        );
        response["result"].clone()
    }

    fn error(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("result").is_none(),
            "{method} unexpectedly succeeded: {response}"
        );
        response["error"].clone()
    }

    fn handshake(&mut self) -> Value {
        let result = self.result(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "e2e-real-binary", "version": "1" },
            })),
        );
        self.notify("notifications/initialized", None);
        result
    }

    /// Block until the embedding runtime reports ready.
    ///
    /// This is the honest replacement for "sleep 45 seconds and hope": it waits
    /// for the exact event, returns as soon as it happens, and panics with the
    /// server's own log if it never does.
    fn wait_for_embeddings(&mut self) {
        let deadline = Instant::now() + EMBEDDING_TIMEOUT;
        loop {
            if self
                .stderr_lines()
                .iter()
                .any(|line| line.contains(EMBEDDINGS_READY))
            {
                return;
            }
            assert!(
                self.is_running(),
                "vestige-mcp exited before the embedding runtime came up. stderr: {:?}",
                self.stderr_lines()
            );
            if Instant::now() >= deadline {
                panic!(
                    "embedding runtime never became ready within {EMBEDDING_TIMEOUT:?}. \
                     stderr: {:?}",
                    self.stderr_lines()
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Call a tool and return its structured payload.
    ///
    /// Vestige reports tool-level failures as `isError: true` with a JSON body,
    /// not as a JSON-RPC error, so this returns the body either way and lets
    /// the caller decide.
    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.result(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        if let Some(structured) = result.get("structuredContent") {
            return structured.clone();
        }
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("tool {name} returned no text content: {result}"));
        serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
    }

    fn call_tool_ok(&mut self, name: &str, arguments: Value) -> Value {
        let value = self.call_tool(name, arguments);
        assert!(value.get("error").is_none(), "tool {name} failed: {value}");
        value
    }

    /// Ingest a memory whose retrieval is exercised only through keyword/FTS
    /// paths, so the embedding runtime is irrelevant to the assertion.
    fn ingest_keyword_only(&mut self, content: &str, tags: &[&str]) -> String {
        self.ingest_inner(content, tags, false)
    }

    /// Ingest a memory and assert the real embedding runtime produced a vector.
    ///
    /// This is the guard against silently testing the degraded no-embedding
    /// path: if the model is not loaded, `hasEmbedding` is `false` and the test
    /// fails here rather than producing a meaningless green.
    fn ingest_embedded(&mut self, content: &str, tags: &[&str]) -> String {
        self.ingest_inner(content, tags, true)
    }

    fn ingest_inner(&mut self, content: &str, tags: &[&str], require_embedding: bool) -> String {
        let value = self.call_tool_ok(
            "smart_ingest",
            json!({ "content": content, "tags": tags, "forceCreate": true }),
        );
        assert_eq!(
            value["success"],
            json!(true),
            "smart_ingest failed for {content:?}: {value}"
        );
        if require_embedding {
            assert_eq!(
                value["hasEmbedding"],
                json!(true),
                "the embedding runtime was not actually used for {content:?}; this test would \
                 have measured the degraded keyword-only fallback. Response: {value}"
            );
        }
        value["nodeId"]
            .as_str()
            .unwrap_or_else(|| panic!("smart_ingest returned no nodeId: {value}"))
            .to_string()
    }

    /// Run `recall` and return the result ids in rank order.
    fn recall_ids(&mut self, arguments: Value) -> Vec<String> {
        let value = self.call_tool_ok("recall", arguments);
        value["results"]
            .as_array()
            .map(|results| {
                results
                    .iter()
                    .filter_map(|r| r["id"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn memory_found(&mut self, id: &str) -> bool {
        self.call_tool("memory", json!({ "action": "get", "id": id }))["found"] == json!(true)
    }

    /// Close stdin and wait for a clean exit, the way an MCP client shuts a
    /// stdio server down.
    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll vestige-mcp") {
                Some(status) => {
                    assert!(
                        status.success(),
                        "stdin EOF must be a clean shutdown, got {status}. stderr: {:?}",
                        self.stderr_lines()
                    );
                    return;
                }
                None if Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!("vestige-mcp did not exit within 30s of stdin EOF");
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    /// Give up on a server we expect to be dead or that we no longer need.
    fn abandon(mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Guaranteed cleanup even when a test panics mid-conversation.
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A temporary Vestige data directory.
/// A private copy of the server binary, made once per test process.
///
/// `CARGO_BIN_EXE_vestige-mcp` is the correct path and Cargo guarantees the
/// binary is built before this test runs. It does NOT guarantee the file stays
/// in place: under `cargo test --workspace` the binary at `target/<profile>/`
/// can be relinked while these tests are already running, and a spawn landing
/// in that window fails with a bare `NotFound` that reads like a missing
/// build. Observed once in a full workspace run, never when this suite runs
/// alone, which is exactly the signature of a race against the build directory
/// rather than a defect in the product.
///
/// Copying once into this process's own temp directory removes the race
/// instead of retrying around it, so a `NotFound` from here again would mean
/// something genuinely wrong rather than a known flake.
fn server_binary() -> &'static Path {
    static BINARY: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    BINARY.get_or_init(|| {
        let source = Path::new(env!("CARGO_BIN_EXE_vestige-mcp"));
        // Leaked deliberately: this must outlive every test in the process, and
        // the OS reclaims it when the run ends.
        let dir = Box::leak(Box::new(
            tempfile::tempdir().expect("temporary directory for the server binary"),
        ));
        let destination = dir.path().join("vestige-mcp");
        std::fs::copy(source, &destination)
            .unwrap_or_else(|error| panic!("copy {} for the test run: {error}", source.display()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o755))
                .expect("make the copied server executable");
        }
        destination
    })
}

fn data_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary Vestige data directory")
}

fn db_path(dir: &Path) -> PathBuf {
    dir.join("vestige.db")
}

/// Open the store directly. Only ever called while no server is running.
fn open_db(dir: &Path) -> Connection {
    Connection::open(db_path(dir)).expect("open vestige.db directly")
}

fn quick_check(conn: &Connection) -> Vec<String> {
    let mut statement = conn.prepare("PRAGMA quick_check").expect("prepare");
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("quick_check");
    rows.map(|row| row.expect("quick_check row")).collect()
}

fn foreign_key_violations(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
        row.get(0)
    })
    .expect("foreign_key_check")
}

fn schema_version(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .expect("schema_version")
}

/// Assert the store on disk is healthy: no corruption, no dangling children.
fn assert_store_is_healthy(dir: &Path) {
    let conn = open_db(dir);
    assert_eq!(
        quick_check(&conn),
        vec!["ok".to_string()],
        "store failed PRAGMA quick_check"
    );
    assert_eq!(
        foreign_key_violations(&conn),
        0,
        "store has dangling foreign keys"
    );
}

/// Put the review gate into `fast` mode.
///
/// In the default `risk_gated` mode a destructive or suppressive mutation is
/// intercepted and turned into a pending Memory PR rather than applied, so a
/// test that wants to observe the mutation itself has to opt out of review
/// first. See [`purge_with_confirm_is_review_gated_by_default`], which pins the
/// default behaviour.
fn disable_review_gate(dir: &Path) {
    std::fs::write(dir.join("review_mode.json"), r#"{"mode":"fast"}"#)
        .expect("write review_mode.json");
}

// ============================================================================
// 1. Protocol surface
// ============================================================================

/// `server/discover` must answer with no handshake at all, and must not claim a
/// protocol revision the server does not implement.
///
/// Catches: gating discovery behind `initialize` (which makes it useless, since
/// its entire purpose is to precede the handshake), and advertising
/// `2026-07-28` — a revision whose stateless core, `resultType` and MRTR are
/// not implemented here. A false version claim fails conformance for real.
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
        "discover must advertise the revision the server actually speaks: {versions:?}"
    );
    assert!(
        !versions.contains(&"2026-07-28".to_string()),
        "the server does not implement 2026-07-28 (stateless core / resultType / MRTR); \
         advertising it would be a false conformance claim: {versions:?}"
    );
    // Every advertised revision must be one initialize will actually accept.
    for version in &versions {
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
        probe.abandon();
    }

    server.shutdown();
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
/// declares, so it must answer (with nothing to advertise) rather than
/// method-not-found, and an unknown pagination cursor on any list method must be
/// refused with `-32602` instead of being answered with page one.
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
    assert_eq!(templates["resourceTemplates"], json!([]));

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
    assert_eq!(recall["annotations"]["readOnlyHint"], json!(true));
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

/// A corrupt FTS5 index must not strand the user's memories.
///
/// `knowledge_fts` is declared `content='knowledge_nodes'`, so it is derived
/// state and always reconstructible. Catches the field failure where a store
/// with thousands of intact memories became unopenable because one fts5 blob
/// was damaged: the server must start, rebuild the index, keep every memory,
/// and serve keyword search again.
#[test]
fn corrupt_fts_index_does_not_brick_the_store() {
    let dir = data_dir();

    let mut server = Server::spawn(dir.path());
    server.handshake();
    let mut ids = Vec::new();
    for i in 0..5 {
        ids.push(server.ingest_keyword_only(
            &format!("Memory number {i} about the deployment rollout checklist"),
            &[],
        ));
    }
    server.shutdown();

    // Corrupt the index the way an interrupted rebuild does.
    {
        let conn = open_db(dir.path());
        conn.execute_batch(
            // Fixed byte pattern, not randomblob(): an unseeded random block
            // sometimes damages the segment so badly that quick_check itself
            // fails with SQLITE_NOMEM, and the test flakes (Aug 30, Sep 1).
            &format!(
                "UPDATE knowledge_fts_data SET block = x'{}' \
                 WHERE id = (SELECT id FROM knowledge_fts_data WHERE id > 1 LIMIT 1);",
                "A5".repeat(200)
            ),
        )
        .expect("corrupt the fts index");
        assert!(
            conn.execute_batch(
                "INSERT INTO knowledge_fts(knowledge_fts) VALUES('integrity-check');"
            )
            .is_err(),
            "the fixture must actually corrupt the index, otherwise this test proves nothing"
        );
        assert_ne!(
            quick_check(&conn),
            vec!["ok".to_string()],
            "the corrupted store must fail quick_check before reopening"
        );
    }

    let mut reopened = Server::spawn(dir.path());
    reopened.handshake();

    for id in &ids {
        assert!(
            reopened.memory_found(id),
            "memory {id} was lost when the derived FTS index was rebuilt"
        );
    }
    let hits = reopened.recall_ids(json!({
        "query": "deployment rollout checklist",
        "limit": 10,
        "concrete": true,
    }));
    assert_eq!(
        hits.len(),
        ids.len(),
        "the rebuilt index must find every seeded memory again, got {hits:?}"
    );

    reopened.shutdown();
    assert_store_is_healthy(dir.path());
}

/// A CASCADE-declared orphan row must be repaired on open, not treated as fatal.
///
/// Catches the regression where any store carrying deletion residue from a
/// build that ran without `PRAGMA foreign_keys = ON` became unopenable, with no
/// recovery short of manual SQLite surgery.
#[test]
fn foreign_key_orphans_are_repaired_instead_of_being_fatal() {
    let dir = data_dir();

    let mut server = Server::spawn(dir.path());
    server.handshake();
    let survivor =
        server.ingest_keyword_only("A memory that must survive an orphan repair", &["repair"]);
    server.shutdown();

    // A child row whose knowledge_nodes parent is gone. Its own schema says
    // ON DELETE CASCADE, so it is unreachable by construction.
    {
        let conn = open_db(dir.path());
        conn.execute_batch("PRAGMA foreign_keys=OFF;")
            .expect("disable fk enforcement for the fixture");
        conn.execute(
            "INSERT INTO node_embeddings(node_id, embedding, dimensions, model, created_at) \
             VALUES ('ghost-parent-0001', X'00010203', 4, 'fixture', datetime('now'))",
            [],
        )
        .expect("insert orphan child row");
        assert_eq!(
            foreign_key_violations(&conn),
            1,
            "the fixture must actually create a violation"
        );
    }

    let mut reopened = Server::spawn(dir.path());
    reopened.handshake();
    assert!(
        reopened.memory_found(&survivor),
        "the orphan repair must not take live memories with it"
    );
    assert!(
        reopened
            .stderr_lines()
            .iter()
            .any(|line| line.contains("repaired orphaned child rows")),
        "the repair must be logged, not silent. stderr: {:?}",
        reopened.stderr_lines()
    );
    reopened.shutdown();

    let conn = open_db(dir.path());
    assert_eq!(
        foreign_key_violations(&conn),
        0,
        "the orphan must be gone after the repair"
    );
    let ghosts: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM node_embeddings WHERE node_id = 'ghost-parent-0001'",
            [],
            |row| row.get(0),
        )
        .expect("count ghosts");
    assert_eq!(
        ghosts, 0,
        "the unreachable child row should have been deleted"
    );
}

/// Migration must survive another process holding a write lock on the database.
///
/// This is the scenario that damaged a real store: a second SQLite writer is
/// live while the server starts and runs its migrations. Nothing else in the
/// repository covers it, because every other test opens the store from inside
/// one process.
///
/// The fixture takes a genuine `BEGIN IMMEDIATE` write lock without leaving any
/// user table in the committed snapshot, so the server still sees a fresh store
/// and runs the full migration chain — the maximum amount of migration work
/// possible — with a competing writer on the file.
#[test]
fn migration_survives_a_concurrent_sqlite_writer() {
    let dir = data_dir();

    let squatter = Connection::open(db_path(dir.path())).expect("create the db file");
    squatter
        .execute_batch("PRAGMA journal_mode=WAL;")
        .expect("WAL");
    squatter
        .execute_batch("BEGIN IMMEDIATE; CREATE TABLE squatter_uncommitted(x);")
        .expect("take a write lock");

    // Sanity: the committed snapshot the server will read is still empty, so it
    // takes the fresh-store path rather than the "damaged non-empty db" refusal.
    {
        let observer = Connection::open(db_path(dir.path())).expect("second reader");
        let tables: i64 = observer
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .expect("count tables");
        assert_eq!(tables, 0, "the fixture must not publish a table");
    }

    let mut server = Server::spawn(dir.path());
    // Let the server get well into its migration chain while the lock is held.
    std::thread::sleep(Duration::from_secs(2));
    squatter
        .execute_batch("ROLLBACK;")
        .expect("release the lock");
    drop(squatter);

    server.handshake();
    assert!(
        server.result("tools/list", None)["tools"].is_array(),
        "the server must be fully functional after migrating under contention"
    );
    let written = server.ingest_keyword_only("Wrote after a contended migration", &["contention"]);
    assert!(
        server.memory_found(&written),
        "post-migration write was lost"
    );
    server.shutdown();

    let conn = open_db(dir.path());
    assert_eq!(
        quick_check(&conn),
        vec!["ok".to_string()],
        "a contended migration must not corrupt the store"
    );
    assert_eq!(foreign_key_violations(&conn), 0);
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
        .expect("count schema_version rows");
    assert_eq!(
        rows, 1,
        "a partially replayed migration chain would leave more than one version row"
    );
    assert!(
        schema_version(&conn) > 0,
        "the migration chain must have completed, not stalled at version 0"
    );
    assert!(
        !conn
            .prepare("SELECT 1 FROM sqlite_master WHERE name='squatter_uncommitted'")
            .and_then(|mut s| s.exists([]))
            .expect("look for the rolled-back table"),
        "the fixture's uncommitted table must never have landed"
    );
}

/// Two servers racing to migrate the same data directory must not corrupt it.
///
/// A loser is allowed to fail — SQLite has one writer — but it must fail loudly
/// and leave a complete, single-versioned, quick_check-clean store behind.
/// Catches: a half-applied migration chain, duplicate `schema_version` rows, or
/// a panic instead of a diagnosable "database is locked".
#[test]
fn concurrent_server_startups_leave_an_intact_store() {
    let dir = data_dir();

    let mut first = Server::spawn(dir.path());
    let mut second = Server::spawn(dir.path());
    std::thread::sleep(Duration::from_secs(6));

    let mut healthy = 0usize;
    for server in [&mut first, &mut second] {
        if server.is_running() {
            healthy += 1;
        } else {
            let errors = server.error_lines();
            assert!(
                !errors.is_empty(),
                "a server that lost the migration race must say why it exited. stderr: {:?}",
                server.stderr_lines()
            );
            assert!(
                errors
                    .iter()
                    .any(|line| line.contains("Failed to initialize storage")),
                "the loser must report a storage-init failure, not an opaque crash: {errors:?}"
            );
            assert!(
                !server
                    .stderr_lines()
                    .iter()
                    .any(|line| line.contains("panicked at")),
                "losing the race must not panic: {:?}",
                server.stderr_lines()
            );
        }
    }
    assert!(
        healthy >= 1,
        "at least one racing server must come up; otherwise a concurrent start is a total outage"
    );

    for mut server in [first, second] {
        if server.is_running() {
            server.handshake();
            let id = server.ingest_keyword_only("Survived a startup race", &["race"]);
            assert!(server.memory_found(&id));
        }
        server.abandon();
    }

    let conn = open_db(dir.path());
    assert_eq!(
        quick_check(&conn),
        vec!["ok".to_string()],
        "a startup race must not corrupt the store"
    );
    assert_eq!(foreign_key_violations(&conn), 0);
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))
        .expect("count schema_version rows");
    assert_eq!(
        rows, 1,
        "the migration chain must have been applied exactly once"
    );
    assert!(schema_version(&conn) > 0);
}

/// Everything ingested must still be there after a clean stop and restart.
///
/// Catches: writes that live only in an in-process cache, a WAL that is never
/// checkpointed, and tags or content mangled on reload.
#[test]
fn a_clean_restart_preserves_every_memory() {
    let dir = data_dir();

    let mut first = Server::spawn(dir.path());
    first.handshake();
    let alpha = first.ingest_keyword_only(
        "The invoicing reconciliation job runs at midnight in the Frankfurt region",
        &["billing", "Ops:Nightly"],
    );
    let beta = first.ingest_keyword_only(
        "The vendor onboarding checklist requires a signed data processing addendum",
        &["legal"],
    );
    first.shutdown();

    let mut second = Server::spawn(dir.path());
    second.handshake();

    for id in [&alpha, &beta] {
        assert!(
            second.memory_found(id),
            "memory {id} did not survive a restart"
        );
    }

    let node =
        second.call_tool_ok("memory", json!({ "action": "get", "id": &alpha }))["node"].clone();
    assert!(
        node["content"]
            .as_str()
            .expect("content")
            .contains("Frankfurt"),
        "content was mangled across the restart: {node}"
    );
    assert_eq!(
        node["tags"],
        json!(["billing", "Ops:Nightly"]),
        "tags must round-trip verbatim, including case: {node}"
    );

    let hits = second.recall_ids(json!({
        "query": "reconciliation Frankfurt",
        "limit": 10,
        "concrete": true,
    }));
    assert_eq!(
        hits,
        vec![alpha.clone()],
        "keyword index did not survive the restart"
    );

    second.shutdown();
    assert_store_is_healthy(dir.path());
}

// ============================================================================
// 3. Retrieval correctness (embedding-independent paths)
// ============================================================================

/// A save costs the agent context on every call, so the create response has a
/// byte ceiling and must not carry a tag-status block that says nothing.
#[test]
fn smart_ingest_create_response_is_lean() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let value = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": "A plain engineering note about the deploy cache", "tags": ["deploy"], "forceCreate": true }),
    );
    assert_eq!(value["success"], json!(true), "{value}");
    let bytes = serde_json::to_string(&value).unwrap().len();
    assert!(bytes <= 1_900, "create response is {bytes} bytes: {value}");
    assert!(
        value.get("tagSuggestionStatus").is_none(),
        "a create with nothing to report about tags must not carry the status block: {value}"
    );
    for key in ["similarity", "supersededId", "previousContent", "mergePreview", "mergedFrom"] {
        assert!(value.get(key).is_none(), "{key} is null on a create and must be absent: {value}");
    }
    server.shutdown();
}

/// A capitalised tag must be findable by a lower-case prefix, and vice versa.
///
/// A silent zero here is the worst failure shape a memory system has: the
/// caller asks for their `Infra:` memories, gets an empty list, and concludes
/// nothing was ever saved. The tool schema still describes this filter as
/// "case-sensitive", so the documented contract and the implemented one
/// disagree; the implementation is the one users depend on.
#[test]
fn tag_prefix_filtering_is_case_insensitive_on_the_keyword_path() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let deploy = server.ingest_keyword_only(
        "Rollout gate alpha for the payments service",
        &["Infra:Deploy"],
    );
    let staging = server.ingest_keyword_only(
        "Rollout gate beta for the payments service",
        &["Infra:Staging"],
    );
    let office = server.ingest_keyword_only(
        "Rollout gate gamma for the office kitchen",
        &["Office:Kitchen"],
    );

    let unfiltered = server.recall_ids(json!({
        "query": "Rollout gate",
        "limit": 10,
        "concrete": true,
    }));
    assert_eq!(
        unfiltered.len(),
        3,
        "baseline query must see all three memories, got {unfiltered:?}"
    );

    // Every casing of the same prefix must select the same two memories.
    for prefix in ["Infra:", "infra:", "INFRA:", "InFrA:"] {
        let mut filtered = server.recall_ids(json!({
            "query": "Rollout gate",
            "limit": 10,
            "concrete": true,
            "tag_prefix": prefix,
        }));
        filtered.sort();
        let mut expected = vec![deploy.clone(), staging.clone()];
        expected.sort();
        assert_eq!(
            filtered, expected,
            "tag_prefix {prefix:?} must match 'Infra:Deploy'/'Infra:Staging' regardless of case"
        );
        assert!(
            !filtered.contains(&office),
            "tag_prefix {prefix:?} must still exclude non-matching tags"
        );
    }

    // And the filter must genuinely filter, not just pass everything through.
    let none = server.recall_ids(json!({
        "query": "Rollout gate",
        "limit": 10,
        "concrete": true,
        "tag_prefix": "nonexistent:",
    }));
    assert!(
        none.is_empty(),
        "a prefix matching nothing must return nothing: {none:?}"
    );

    server.shutdown();
}

/// The memory that IS an identifier must outrank the memory that merely cites it.
///
/// Raw BM25 magnitude is unbounded while the literal-match bonus is capped, so
/// a note repeating a UUID three times can outscore the exact match and invert
/// the documented exact-lookup guarantee. Filler documents are required: with a
/// tiny corpus BM25's IDF term is degenerate and the ranking proves nothing.
#[test]
fn exact_identifier_lookup_beats_a_memory_that_only_cites_it() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for i in 0..40 {
        server.ingest_keyword_only(
            &format!("Routine note {i} about deployment pipelines and review cadence"),
            &[],
        );
    }

    let needle = "PAYMENTS_REDIS_URL";
    let exact = server.ingest_keyword_only(needle, &[]);
    let citer = server.ingest_keyword_only(
        &format!(
            "See {needle} for the rollout; {needle} was rotated in review, and \
             {needle} supersedes the older connection note entirely"
        ),
        &[],
    );

    let ranked = server.recall_ids(json!({ "query": needle, "limit": 5 }));
    assert!(
        ranked.len() >= 2,
        "both the exact match and the citing note should surface: {ranked:?}"
    );
    assert_eq!(
        ranked.first(),
        Some(&exact),
        "the memory that IS {needle} must rank above the one that merely cites it \
         three times; got {ranked:?}"
    );
    assert!(
        ranked.contains(&citer),
        "the citing memory must still be retrievable, just not first"
    );

    server.shutdown();
}

/// Typographic punctuation must not swallow the words next to it.
///
/// An em dash, a curly apostrophe and an accented word all sit inside one
/// memory alongside 25 unrelated ones. Catches the tokenizer regression where
/// `window — carefully` indexed as a single unsearchable token and `naïve`
/// could not be reached from `naive`. The filler corpus makes a hit meaningful:
/// with one memory in the store every query "succeeds".
#[test]
fn unicode_and_typographic_content_stays_findable_by_keyword() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    for i in 0..25 {
        server.ingest_keyword_only(
            &format!(
                "Unrelated filler memory {i} covering invoicing, payroll and vendor onboarding"
            ),
            &[],
        );
    }
    let target = server.ingest_keyword_only(
        "The rollout window — carefully negotiated — is Tuesday; the team’s naïve estimate slipped",
        &[],
    );

    // Words adjacent to the em dash, adjacent to the curly apostrophe, the
    // accented word itself, and its unaccented spelling.
    for term in [
        "window",     // immediately before an em dash
        "carefully",  // immediately after an em dash
        "negotiated", // immediately before an em dash
        "team",       // immediately before a curly apostrophe
        "naïve",      // the accented word itself
        "naive",      // the same word without the accent
        "estimate",   // immediately after the accented word
    ] {
        let hits = server.recall_ids(json!({
            "query": term,
            "limit": 10,
            "concrete": true,
        }));
        assert_eq!(
            hits,
            vec![target.clone()],
            "keyword search for {term:?} must return exactly the typographic memory, got {hits:?}"
        );
    }

    server.shutdown();
}

// ============================================================================
// 4. Deletion, suppression and the review gate
// ============================================================================

/// In the default review mode a confirmed purge is held for review, not applied.
///
/// This is load-bearing and surprising: `memory(action='purge', confirm=true)`
/// answers `purge_pending_review`, the memory stays fully retrievable, and
/// nothing is erased until the Memory PR is decided. A caller that reads
/// `confirm=true` as "erased" would be wrong. Catches a regression in either
/// direction: silently erasing without review, or dropping the review record.
#[test]
fn purge_with_confirm_is_review_gated_by_default() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let subject = server.ingest_keyword_only(
        "The quarterly revenue figure for the Helsinki office was 4.2 million euros",
        &["finance"],
    );

    let unconfirmed = server.call_tool("memory", json!({ "action": "purge", "id": &subject }));
    assert!(
        unconfirmed["error"]
            .as_str()
            .is_some_and(|e| e.contains("confirm=true")),
        "purge without confirm must refuse: {unconfirmed}"
    );

    let gated = server.call_tool(
        "memory",
        json!({ "action": "purge", "id": &subject, "confirm": true }),
    );
    assert_eq!(
        gated["action"],
        json!("purge_pending_review"),
        "default review mode must hold a destructive mutation: {gated}"
    );
    assert_eq!(gated["success"], json!(false));
    assert_eq!(gated["pendingReview"], json!(true));
    assert!(
        gated["memoryPrsOpened"][0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("pr_")),
        "a Memory PR must be opened as the audit record: {gated}"
    );

    assert!(
        server.memory_found(&subject),
        "a pending purge must NOT have removed the memory"
    );

    server.shutdown();
}

/// An approved purge must remove the content and leave only a content-free
/// tombstone.
///
/// Catches: a "delete" that only hides the row (the text stays greppable in the
/// database), and a tombstone that leaks what it was told to forget — the audit
/// record must prove a removal happened without retaining the removed content,
/// its tags, or its reason.
#[test]
fn approved_purge_removes_content_and_leaves_a_content_free_tombstone() {
    let dir = data_dir();
    disable_review_gate(dir.path());
    let mut server = Server::spawn(dir.path());
    server.handshake();

    let subject = server.ingest_keyword_only(
        "The quarterly revenue figure for the Helsinki office was 4.2 million euros",
        &["finance", "confidential"],
    );
    let survivor = server.ingest_keyword_only(
        "The Helsinki office moved to a new building last spring",
        &["office"],
    );

    let purged = server.call_tool_ok(
        "memory",
        json!({
            "action": "purge",
            "id": &subject,
            "confirm": true,
            "reason": "regulator erasure request",
        }),
    );
    assert_eq!(purged["action"], json!("purge"));
    assert_eq!(
        purged["success"],
        json!(true),
        "purge did not apply: {purged}"
    );

    assert!(
        !server.memory_found(&subject),
        "purged memory is still readable"
    );
    assert!(
        server.memory_found(&survivor),
        "purge took an unrelated memory with it"
    );

    let remaining = server.recall_ids(json!({
        "query": "Helsinki quarterly revenue",
        "limit": 10,
        "concrete": true,
    }));
    assert!(
        !remaining.contains(&subject),
        "purged memory still surfaces in retrieval: {remaining:?}"
    );

    server.shutdown();

    let conn = open_db(dir.path());
    let nodes: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE id = ?1",
            [&subject],
            |row| row.get(0),
        )
        .expect("count nodes");
    assert_eq!(nodes, 0, "the row itself must be gone");
    let embeddings: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM node_embeddings WHERE node_id = ?1",
            [&subject],
            |row| row.get(0),
        )
        .expect("count embeddings");
    assert_eq!(embeddings, 0, "the embedding must be gone too");
    let leaked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE content LIKE '%4.2 million%'",
            [],
            |row| row.get(0),
        )
        .expect("scan for leaked content");
    assert_eq!(leaked, 0, "the purged text is still stored somewhere");

    // Exactly one content-free tombstone: proof of removal, no payload.
    let (marker, node_type, tags, reason): (String, String, String, Option<String>) = conn
        .query_row(
            "SELECT memory_id, node_type, tags, reason FROM deletion_tombstones",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("exactly one tombstone");
    assert!(
        marker.starts_with("opaque:"),
        "the tombstone must not retain the raw memory id: {marker}"
    );
    assert!(
        !marker.contains(&subject),
        "the tombstone marker leaks the purged id: {marker}"
    );
    assert_eq!(
        node_type, "fact",
        "the tombstone should keep only content-free metadata"
    );
    assert_eq!(tags, "[]", "the tombstone must not retain tags: {tags}");
    assert_eq!(
        reason, None,
        "the purge reason is user-supplied prose and must not be retained: {reason:?}"
    );

    assert_store_is_healthy(dir.path());
}

/// Suppression must persist and compound across a restart.
///
/// Active forgetting is not deletion: the memory stays, inhibited. Catches the
/// failure where a restart rehydrates FSRS state from defaults and quietly
/// restores a memory the user deliberately suppressed — the memory system
/// undoing the user's decision behind their back.
#[test]
fn suppression_survives_a_restart_and_keeps_compounding() {
    let dir = data_dir();
    disable_review_gate(dir.path());

    let mut first = Server::spawn(dir.path());
    first.handshake();
    let stale = first.ingest_keyword_only(
        "The legacy billing exporter should be run manually every Friday afternoon",
        &["ops"],
    );

    let before = first.call_tool_ok("memory", json!({ "action": "state", "id": &stale }));
    let baseline_retrieval = before["components"]["retrievalStrength"]
        .as_f64()
        .expect("retrievalStrength");

    let suppressed = first.call_tool_ok(
        "suppress",
        json!({ "id": &stale, "reason": "superseded by the scheduler" }),
    );
    assert_eq!(
        suppressed["success"],
        json!(true),
        "suppress failed: {suppressed}"
    );
    assert_eq!(suppressed["suppressionCount"], json!(1));

    let after = first.call_tool_ok("memory", json!({ "action": "state", "id": &stale }));
    let suppressed_retrieval = after["components"]["retrievalStrength"]
        .as_f64()
        .expect("retrievalStrength");
    assert!(
        suppressed_retrieval < baseline_retrieval,
        "suppression must actually inhibit retrieval ({suppressed_retrieval} vs {baseline_retrieval})"
    );
    first.shutdown();

    let mut second = Server::spawn(dir.path());
    second.handshake();

    let restarted = second.call_tool_ok("memory", json!({ "action": "state", "id": &stale }));
    assert_eq!(
        restarted["components"]["retrievalStrength"]
            .as_f64()
            .expect("retrievalStrength"),
        suppressed_retrieval,
        "a restart restored a suppressed memory's retrieval strength: {restarted}"
    );
    assert!(
        restarted["content"].is_string(),
        "suppression is not deletion; the memory must still exist: {restarted}"
    );

    // The suppression ledger must have survived too, so a second call compounds
    // rather than starting over.
    let again = second.call_tool_ok("suppress", json!({ "id": &stale }));
    assert_eq!(
        again["priorCount"],
        json!(1),
        "the pre-restart suppression was forgotten: {again}"
    );
    assert_eq!(again["suppressionCount"], json!(2));

    second.shutdown();
    assert_store_is_healthy(dir.path());
}

// ============================================================================
// 5. Tests that require the real embedding runtime
//
// These load a ~670 MB ONNX model (downloading it on a cold machine), so they
// are ignored by default. Run them with:
//     cargo test -p vestige-mcp --test e2e_real_binary -- --ignored
// ============================================================================

/// Baseline for every ignored test below: the real runtime must actually be in
/// play, and hybrid retrieval must be the path taken.
///
/// If this fails, the rest of this section is measuring the keyword fallback
/// and its results are meaningless.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn the_real_embedding_runtime_produces_vectors_and_hybrid_retrieval() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();

    let id = server.ingest_embedded(
        "The deployment pipeline uses blue-green rollout on the Kubernetes cluster",
        &["infra"],
    );

    let node = server.call_tool_ok("memory", json!({ "action": "get", "id": &id }))["node"].clone();
    assert_eq!(
        node["hasEmbedding"],
        json!(true),
        "no vector was stored: {node}"
    );
    assert!(
        node["embeddingModel"]
            .as_str()
            .is_some_and(|m| !m.is_empty()),
        "the stored vector must record which model produced it: {node}"
    );

    // A paraphrase sharing NO content word with the memory. If the vector side
    // is dead, BM25 alone cannot bridge this and the result set is empty.
    let value = server.call_tool_ok(
        "recall",
        json!({
            "query": "zero downtime release strategy",
            "limit": 5,
            "min_similarity": 0.3,
        }),
    );
    assert_eq!(
        value["method"],
        json!("hybrid+cognitive"),
        "expected the hybrid path, got {}",
        value["method"]
    );
    let hit = value["results"]
        .as_array()
        .expect("results")
        .iter()
        .find(|r| r["id"] == json!(&id))
        .unwrap_or_else(|| {
            panic!("semantic retrieval failed to find a paraphrased match: {value}")
        });
    assert!(
        hit["keywordScore"].is_null(),
        "the paraphrase must have matched semantically, not lexically: {hit}"
    );
    assert!(
        hit["semanticScore"].as_f64().is_some_and(|s| s > 0.3),
        "expected a real semantic score on the paraphrase: {hit}"
    );

    server.shutdown();
}

/// Both sides of a contradiction must survive retrieval, and the dissenting
/// side must be flagged rather than quietly demoted.
///
/// Retrieval-induced forgetting suppresses the loser of a competition between
/// SIMILAR memories, and a contradiction is near-identical text with opposite
/// meaning — the most suppressible class of memory there is. Without the
/// exemption, every time an agent retrieves one side, the evidence that would
/// correct it gets demoted and can fall out of the returned window. That is
/// precisely how a memory system buries its own correction.
///
/// Covers both detectable shapes: an explicit negation pair ("Never X" /
/// "Always X") and an antonym pair with no negation in either side
/// ("X hurts accuracy" / "X improves accuracy").
///
/// # Note on the retry, and a product observation
///
/// The retrieval-competition stage that produces `contradictionProtected` is
/// guarded by a NON-BLOCKING `cognitive.try_lock()`. If that lock is held when
/// the recall runs, the entire stage — competition AND the contradiction
/// exemption — is skipped, and the response says nothing about it: the caller
/// gets a normal-looking result with the safeguard silently switched off.
///
/// This was observed here, not theorised. With the autopilot enabled (the
/// shipped default) it subscribes to `MemoryCreated` and takes a *blocking*
/// `cognitive.lock()` per event, so ingesting a burst of memories and recalling
/// immediately afterwards reliably loses the flag under machine load: this test
/// passed 3/3 in isolation and failed when run alongside the rest of the
/// suite. The harness pins `VESTIGE_AUTOPILOT_ENABLED=0` to remove that
/// contention; the bounded retry below covers the remaining best-effort window.
///
/// The invariant that must hold unconditionally — both sides returned — is
/// asserted without any retry.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn contradictions_are_returned_intact_and_flagged_as_protected() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();

    let never = server.ingest_embedded(
        "Never use prompt diversity when the sampling temperature exceeds zero point six",
        &[],
    );
    let always = server.ingest_embedded(
        "Always use prompt diversity when the sampling temperature exceeds zero point six",
        &[],
    );
    let hurts = server.ingest_embedded(
        "Prompt diversity hurts accuracy on the competition benchmark evaluation",
        &[],
    );
    let improves = server.ingest_embedded(
        "Prompt diversity improves accuracy on the competition benchmark evaluation",
        &[],
    );

    let query = json!({ "query": "prompt diversity sampling temperature", "limit": 10 });

    // Unconditional invariant: nothing may be suppressed out of the window.
    let value = server.call_tool_ok("recall", query.clone());
    let ids: Vec<String> = value["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|r| r["id"].as_str().map(str::to_string))
        .collect();
    for (label, id) in [
        ("negation:never", &never),
        ("negation:always", &always),
        ("antonym:hurts", &hurts),
        ("antonym:improves", &improves),
    ] {
        assert!(
            ids.contains(id),
            "{label} was suppressed out of the result window; the caller would never see \
             the other side. Returned: {ids:?}"
        );
    }
    assert!(
        value["receipt"]["suppressed"]
            .as_array()
            .is_none_or(|s| s.is_empty()),
        "no side of a live contradiction may be recorded as suppressed: {}",
        value["receipt"]
    );

    // The explicit flag. Retried because the stage behind it is best-effort;
    // see this test's doc comment.
    let mut protected = value["contradictionProtected"].clone();
    for _ in 0..10 {
        if protected.is_object() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
        protected = server.call_tool_ok("recall", query.clone())["contradictionProtected"].clone();
    }
    assert!(
        protected.is_object(),
        "the dissenting side must be reported, not silently spared. The retrieval-competition \
         stage never ran across 11 attempts, which means the contradiction safeguard is off \
         and nothing in the response says so."
    );
    let protected_ids: Vec<String> = protected["memoryIds"]
        .as_array()
        .expect("memoryIds")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    assert!(
        !protected_ids.is_empty(),
        "contradictionProtected must name the memories it spared: {protected}"
    );
    assert!(
        protected_ids
            .iter()
            .all(|id| id == &never || id == &always || id == &hurts || id == &improves),
        "contradictionProtected named a memory that is not part of either pair: {protected_ids:?}"
    );
    assert!(
        protected["notice"]
            .as_str()
            .is_some_and(|n| n.contains("contradict")),
        "the protection must come with a notice telling the caller to read both sides: {protected}"
    );

    server.shutdown();
}

/// Tag filtering must be case-insensitive on the full hybrid path too.
///
/// The keyword path and the hybrid path apply this filter in two different
/// places, so covering only one leaves half the surface untested.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn tag_prefix_filtering_is_case_insensitive_on_the_hybrid_path() {
    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();

    let deploy = server.ingest_embedded(
        "Rollout gate alpha guards the payments service release",
        &["Infra:Deploy"],
    );
    let staging = server.ingest_embedded(
        "Rollout gate beta guards the payments service release",
        &["Infra:Staging"],
    );
    let office = server.ingest_embedded(
        "Rollout gate gamma guards the office kitchen refit",
        &["Office:Kitchen"],
    );

    let unfiltered = server.recall_ids(json!({ "query": "rollout gate guards", "limit": 10 }));
    assert!(
        unfiltered.contains(&office),
        "baseline hybrid query must include the memory the filter should later drop: {unfiltered:?}"
    );

    for prefix in ["Infra:", "infra:", "INFRA:"] {
        let mut filtered = server.recall_ids(
            json!({ "query": "rollout gate guards", "limit": 10, "tag_prefix": prefix }),
        );
        filtered.sort();
        let mut expected = vec![deploy.clone(), staging.clone()];
        expected.sort();
        assert_eq!(
            filtered, expected,
            "hybrid tag_prefix {prefix:?} must be case-insensitive and must exclude {office}"
        );
    }

    server.shutdown();
}

/// A correction must not be swallowed by the ingest gate.
///
/// # THIS TEST FAILS AGAINST THE CURRENT BUILD. It documents a real defect and
/// is deliberately NOT fixed here.
///
/// Reproduction, end to end over the real binary, with the embedding runtime
/// loaded:
///
/// 1. ingest `"Never use prompt diversity when the sampling temperature exceeds
///    zero point six"`
/// 2. ingest `"Always use prompt diversity when the sampling temperature
///    exceeds zero point six"`
///
/// Observed: the second ingest returns `decision: "reinforce"` at similarity
/// 0.965. The correction is DISCARDED and the memory it contradicts is
/// STRENGTHENED. `recall` afterwards returns only the stale "Never" memory.
///
/// Reversing the order changes the outcome: ingesting "Always" first and
/// "Never" second returns `decision: "create"` and keeps both. Same two
/// memories, same similarity — only the order differs.
///
/// Root cause: `crates/vestige-core/src/advanced/prediction_error.rs`. The
/// near-identical branch is correctly guarded by `!best.appears_contradictory`,
/// but the flag comes from `detect_contradiction` in that same file, which
/// tests `new.contains(neg) && old.contains(pos)` — it only fires when the NEW
/// content is the negative one. It also has no antonym branch and no
/// mutually-exclusive-value branch, so these are swallowed identically
/// (all measured over the real binary):
///
/// | first ingest | second ingest | decision | similarity |
/// |---|---|---|---|
/// | Never use prompt diversity … | Always use prompt diversity … | reinforce | 0.965 |
/// | … hurts accuracy … | … improves accuracy … | reinforce | 0.962 |
/// | … PostgreSQL 14 … | … PostgreSQL 16 … | reinforce | 0.940 |
/// | Priya holds a Bachelor degree … | Priya holds a Master of Science degree … | reinforce | 0.976 |
///
/// That was the pre-fix state: the richer retrieval-side detector could see
/// every one of these shapes, but the write path had its own blind copy, and
/// retrieval-side protection cannot protect a memory destroyed at ingest one
/// stage earlier. Both paths now consult the shared detector in
/// `vestige-core/src/advanced/contradiction.rs`; this test locks the write
/// path's behaviour over the real binary.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored. \
            (Documented the ingest-gate defect before the shared-detector fix; \
            it now passes and guards the regression.)"]
fn correction_must_not_be_swallowed_by_the_ingest_gate() {
    const NEGATIVE: &str =
        "Never use prompt diversity when the sampling temperature exceeds zero point six";
    const POSITIVE: &str =
        "Always use prompt diversity when the sampling temperature exceeds zero point six";

    // Control: the SAME two memories in the OPPOSITE order. This must pass, and
    // it proves the failure below is an ordering defect in the detector rather
    // than the gate simply never creating on near-identical text.
    {
        let control_dir = data_dir();
        let mut control = Server::spawn(control_dir.path());
        control.handshake();
        control.wait_for_embeddings();
        control.call_tool_ok("smart_ingest", json!({ "content": POSITIVE }));
        let flipped = control.call_tool_ok("smart_ingest", json!({ "content": NEGATIVE }));
        assert_eq!(
            flipped["decision"],
            json!("create"),
            "control: ingesting the negative claim over the positive one is correctly \
             detected as a contradiction and kept: {flipped}"
        );
        assert_eq!(
            control
                .recall_ids(json!({
                    "query": "prompt diversity sampling temperature",
                    "limit": 10,
                }))
                .len(),
            2,
            "control: both positions must be retrievable in this direction"
        );
        control.shutdown();
    }

    let dir = data_dir();
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();

    // Deliberately NOT forceCreate: this is the Prediction Error Gate under test.
    let original = server.call_tool_ok("smart_ingest", json!({ "content": NEGATIVE }));
    assert_eq!(original["decision"], json!("create"));
    assert_eq!(original["hasEmbedding"], json!(true));
    let original_id = original["nodeId"].as_str().expect("nodeId").to_string();

    let correction = server.call_tool_ok("smart_ingest", json!({ "content": POSITIVE }));

    assert_ne!(
        correction["decision"],
        json!("reinforce"),
        "the gate reinforced the memory the user just contradicted, and discarded the \
         correction. decision={} similarity={}. The control above proves the very same \
         pair IS handled correctly in the opposite order, so this is a direction bug in \
         detect_contradiction, not a threshold choice. This is the worst outcome the gate \
         can produce: the stale claim gets stronger and the correction ceases to exist.",
        correction["decision"],
        correction["similarity"]
    );

    let ids = server.recall_ids(json!({
        "query": "prompt diversity sampling temperature",
        "limit": 10,
    }));
    assert!(
        ids.len() >= 2,
        "after a correction both positions must be retrievable, got {ids:?}"
    );
    assert!(
        ids.contains(&original_id),
        "the original claim vanished: {ids:?}"
    );
    let correction_id = correction["nodeId"].as_str().expect("nodeId").to_string();
    assert!(
        ids.contains(&correction_id),
        "the correction is not retrievable: {ids:?}"
    );

    server.shutdown();
}

/// An approved purge must also drop the vector, not just the row.
///
/// The keyword-path purge test cannot prove this: with no embedding runtime
/// there was never a vector to remove. This one seeds a real vector first.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn approved_purge_removes_the_stored_embedding() {
    let dir = data_dir();
    disable_review_gate(dir.path());
    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();

    let subject = server.ingest_embedded(
        "The quarterly revenue figure for the Helsinki office was 4.2 million euros",
        &["finance"],
    );

    {
        // The vector must exist before we can claim the purge removed it.
        let conn = open_db(dir.path());
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM node_embeddings WHERE node_id = ?1",
                [&subject],
                |row| row.get(0),
            )
            .expect("count embeddings");
        assert_eq!(count, 1, "fixture must have stored a real vector first");
    }

    let purged = server.call_tool_ok(
        "memory",
        json!({ "action": "purge", "id": &subject, "confirm": true }),
    );
    assert_eq!(
        purged["success"],
        json!(true),
        "purge did not apply: {purged}"
    );
    server.shutdown();

    let conn = open_db(dir.path());
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM node_embeddings WHERE node_id = ?1",
            [&subject],
            |row| row.get(0),
        )
        .expect("count embeddings");
    assert_eq!(count, 0, "the vector outlived the purged memory");
}

/// Everything, including vectors, must survive a restart.
///
/// Catches a rebuilt-on-boot vector index that silently drops rows: the memory
/// is still listed but no longer semantically reachable, which looks like a
/// ranking regression rather than data loss.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn embeddings_and_semantic_retrieval_survive_a_restart() {
    let dir = data_dir();

    let mut first = Server::spawn(dir.path());
    first.handshake();
    first.wait_for_embeddings();
    let id = first.ingest_embedded(
        "The deployment pipeline uses blue-green rollout on the Kubernetes cluster",
        &["infra"],
    );
    first.shutdown();

    let mut second = Server::spawn(dir.path());
    second.handshake();
    second.wait_for_embeddings();

    let node = second.call_tool_ok("memory", json!({ "action": "get", "id": &id }))["node"].clone();
    assert_eq!(
        node["hasEmbedding"],
        json!(true),
        "the stored vector did not survive the restart: {node}"
    );

    let ids = second.recall_ids(json!({
        "query": "zero downtime release strategy",
        "limit": 5,
        "min_similarity": 0.3,
    }));
    assert!(
        ids.contains(&id),
        "semantic retrieval broke across the restart: {ids:?}"
    );

    second.shutdown();
    assert_store_is_healthy(dir.path());
}

/// A corrupt FTS index must be rebuilt without losing vectors either.
///
/// The default-suite version of this test proves memories and keyword search
/// survive. This one additionally proves the rebuild does not disturb the
/// vector side of the store.
#[test]
#[ignore = "loads the real embedding runtime (~670 MB model); run with --ignored"]
fn corrupt_fts_rebuild_preserves_embeddings() {
    let dir = data_dir();

    let mut server = Server::spawn(dir.path());
    server.handshake();
    server.wait_for_embeddings();
    let mut ids = Vec::new();
    for i in 0..5 {
        ids.push(server.ingest_embedded(
            &format!("Memory number {i} about the deployment rollout checklist"),
            &[],
        ));
    }
    server.shutdown();

    {
        let conn = open_db(dir.path());
        conn.execute_batch(
            // Fixed byte pattern, not randomblob(): an unseeded random block
            // sometimes damages the segment so badly that quick_check itself
            // fails with SQLITE_NOMEM, and the test flakes (Aug 30, Sep 1).
            &format!(
                "UPDATE knowledge_fts_data SET block = x'{}' \
                 WHERE id = (SELECT id FROM knowledge_fts_data WHERE id > 1 LIMIT 1);",
                "A5".repeat(200)
            ),
        )
        .expect("corrupt the fts index");
        assert!(
            conn.execute_batch(
                "INSERT INTO knowledge_fts(knowledge_fts) VALUES('integrity-check');"
            )
            .is_err(),
            "the fixture must actually corrupt the index"
        );
    }

    let mut reopened = Server::spawn(dir.path());
    reopened.handshake();
    reopened.wait_for_embeddings();

    for id in &ids {
        let node =
            reopened.call_tool_ok("memory", json!({ "action": "get", "id": id }))["node"].clone();
        assert_eq!(
            node["hasEmbedding"],
            json!(true),
            "memory {id} lost its vector during the FTS rebuild: {node}"
        );
    }
    let semantic = reopened.recall_ids(json!({
        "query": "what do we check before shipping a release",
        "limit": 10,
    }));
    assert!(
        !semantic.is_empty(),
        "semantic retrieval must work again after the rebuild"
    );

    reopened.shutdown();
    assert_store_is_healthy(dir.path());
}
