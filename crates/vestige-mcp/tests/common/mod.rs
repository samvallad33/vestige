//! Shared real-binary test harness for the `e2e_*` suites.
//!
//! Extracted verbatim from `e2e_real_binary.rs` so the failure-case suite
//! (`e2e_failure_cases.rs`) drives the same shipped binary through the same
//! helpers instead of growing a divergent copy. See that file's header for
//! the full rationale: every helper here spawns
//! `target/<profile>/vestige-mcp` as a child process and speaks line-framed
//! JSON-RPC over its stdin/stdout, exactly as an MCP client does.
//!
//! Each `cargo test` invocation compiles this module per test target, and
//! each suite uses a different subset of the helpers — hence the
//! module-level dead-code allowance.

#![allow(dead_code)]

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
pub(crate) const RPC_TIMEOUT: Duration = Duration::from_secs(120);

/// A running `vestige-mcp` child process plus its stdio plumbing.
///
/// Reader threads drain stdout and stderr so the child can never block on a
/// full pipe, and every read is bounded by a timeout. [`Drop`] kills the child,
/// so a panicking test cannot leak a process.
pub(crate) struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
    /// Server-initiated notifications (`notifications/message` and friends)
    /// seen while waiting for responses. A real MCP client must tolerate them
    /// between any two lines; the harness stashes them here for assertions.
    pub(crate) notifications: Vec<Value>,
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
    pub(crate) fn spawn(data_dir: &Path) -> Self {
        Self::spawn_with_env(data_dir, &[], &[])
    }

    /// Spawn with extra environment: `set` is applied after the standard
    /// pinning (so a test can point the server at a loopback mock, or widen
    /// the reconsolidation labile window), and `remove` deletes variables a
    /// developer's shell may have exported (so "no REDMINE_URL configured" is
    /// actually testable on a machine that has one).
    pub(crate) fn spawn_with_env(
        data_dir: &Path,
        set: &[(&str, &str)],
        remove: &[&str],
    ) -> Self {
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
        for (key, value) in set {
            command.env(key, value);
        }
        for key in remove {
            command.env_remove(key);
        }

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
            notifications: Vec::new(),
        }
    }

    /// Everything the server has written to stderr so far.
    pub(crate) fn stderr_lines(&self) -> Vec<String> {        self.stderr.lock().expect("stderr sink").clone()
    }

    /// Every stderr line that the tracing subscriber marked as an error.
    pub(crate) fn error_lines(&self) -> Vec<String> {
        self.stderr_lines()
            .into_iter()
            .filter(|line| line.contains("ERROR"))
            .collect()
    }

    pub(crate) fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Close stdin without waiting: the client-side half of an EOF shutdown.
    pub(crate) fn close_stdin(&mut self) {
        self.stdin.take();
    }

    /// Non-blocking exit poll for shutdown-budget assertions.
    pub(crate) fn try_exit_status(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().expect("poll vestige-mcp")
    }

    /// Kill and reap. For tests that have given up on a graceful exit.
    pub(crate) fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub(crate) fn write_line(&mut self, line: &str) {
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
    /// Is this line a server-initiated notification (a method, no id)?
    pub(crate) fn is_server_notification(line: &str) -> Option<Value> {
        let value: Value = serde_json::from_str(line).ok()?;
        (value.get("id").is_none() && value.get("method").is_some()).then_some(value)
    }

    /// Next response line. Server notifications are stashed, not returned:
    /// the protocol allows them between any two lines once the handshake is
    /// done, and a client reading "the next line" must skip them.
    pub(crate) fn read_line(&mut self) -> String {
        loop {
            let line = self.read_any_line();
            match Self::is_server_notification(&line) {
                Some(notification) => self.notifications.push(notification),
                None => return line,
            }
        }
    }

    pub(crate) fn read_any_line(&mut self) -> String {
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
    pub(crate) fn expect_silence(&mut self, window: Duration) {
        let deadline = Instant::now() + window;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            match self.stdout.recv_timeout(remaining) {
                Ok(line) => match Self::is_server_notification(&line) {
                    // Logging from the warm-up tasks is not a response.
                    Some(notification) => self.notifications.push(notification),
                    None => panic!("expected no response, got: {line}"),
                },
                Err(_) => return,
            }
        }
    }

    /// Send a raw line and read one response line. For malformed-input tests.
    pub(crate) fn raw_roundtrip(&mut self, line: &str) -> Value {
        self.write_line(line);
        let response = self.read_line();
        serde_json::from_str(&response)
            .unwrap_or_else(|error| panic!("server emitted non-JSON {response:?}: {error}"))
    }

    pub(crate) fn request(&mut self, method: &str, params: Option<Value>) -> Value {
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

    pub(crate) fn notify(&mut self, method: &str, params: Option<Value>) {
        let mut message = json!({ "jsonrpc": "2.0", "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
    }

    pub(crate) fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("error").is_none(),
            "{method} returned an error: {response}"
        );
        response["result"].clone()
    }

    pub(crate) fn error(&mut self, method: &str, params: Option<Value>) -> Value {
        let response = self.request(method, params);
        assert!(
            response.get("result").is_none(),
            "{method} unexpectedly succeeded: {response}"
        );
        response["error"].clone()
    }

    pub(crate) fn handshake(&mut self) -> Value {
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

    /// Call a tool and return its structured payload.
    ///
    /// Vestige reports tool-level failures as `isError: true` with a JSON body,
    /// not as a JSON-RPC error, so this returns the body either way and lets
    /// the caller decide.
    pub(crate) fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
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

    pub(crate) fn call_tool_ok(&mut self, name: &str, arguments: Value) -> Value {
        let value = self.call_tool(name, arguments);
        assert!(value.get("error").is_none(), "tool {name} failed: {value}");
        value
    }

    /// Ingest a memory whose retrieval is exercised only through keyword/FTS
    /// paths, so the (removed) embedding runtime is irrelevant to the
    /// assertion. This is the only ingest helper now.
    pub(crate) fn ingest_keyword_only(&mut self, content: &str, tags: &[&str]) -> String {
        let value = self.call_tool_ok(
            "smart_ingest",
            json!({ "content": content, "tags": tags, "forceCreate": true }),
        );
        assert_eq!(
            value["success"],
            json!(true),
            "smart_ingest failed for {content:?}: {value}"
        );
        value["nodeId"]
            .as_str()
            .unwrap_or_else(|| panic!("smart_ingest returned no nodeId: {value}"))
            .to_string()
    }

    /// Run `recall` and return the result ids in rank order.
    pub(crate) fn recall_ids(&mut self, arguments: Value) -> Vec<String> {
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

    pub(crate) fn memory_found(&mut self, id: &str) -> bool {
        self.call_tool("memory", json!({ "action": "get", "id": id }))["found"] == json!(true)
    }

    /// Close stdin and wait for a clean exit, the way an MCP client shuts a
    /// stdio server down.
    pub(crate) fn shutdown(mut self) {
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
    pub(crate) fn abandon(mut self) {
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
pub(crate) fn server_binary() -> &'static Path {
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
pub(crate) fn data_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temporary Vestige data directory")
}

pub(crate) fn db_path(dir: &Path) -> PathBuf {
    dir.join("vestige.db")
}

/// Open the store directly. Only ever called while no server is running.
pub(crate) fn open_db(dir: &Path) -> Connection {
    Connection::open(db_path(dir)).expect("open vestige.db directly")
}

pub(crate) fn quick_check(conn: &Connection) -> Vec<String> {
    let mut statement = conn.prepare("PRAGMA quick_check").expect("prepare");
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("quick_check");
    rows.map(|row| row.expect("quick_check row")).collect()
}

pub(crate) fn foreign_key_violations(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
        row.get(0)
    })
    .expect("foreign_key_check")
}

pub(crate) fn schema_version(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .expect("schema_version")
}

/// Assert the store on disk is healthy: no corruption, no dangling children.
pub(crate) fn assert_store_is_healthy(dir: &Path) {
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
/// In the opt-in `risk_gated` mode a destructive or suppressive mutation is
/// intercepted and turned into a pending Memory PR rather than applied, so a
/// test that wants to observe the mutation itself has to opt out of review
/// first. See [`purge_with_confirm_is_review_gated_when_opted_in`], which pins the
/// default behaviour.
pub(crate) fn disable_review_gate(dir: &Path) {
    std::fs::write(dir.join("review_mode.json"), r#"{"mode":"fast"}"#)
        .expect("write review_mode.json");
}
pub(crate) fn payload_bytes(value: &Value) -> usize {
    serde_json::to_string(value)
        .map(|s| s.len())
        .unwrap_or(usize::MAX)
}

pub(crate) fn assert_keys(value: &Value, keys: &[&str], context: &str) {
    for key in keys {
        assert!(
            value.get(*key).is_some(),
            "{context}: response is missing `{key}`: {value}"
        );
    }
}

pub(crate) fn assert_under(value: &Value, ceiling: usize, context: &str) {
    let bytes = payload_bytes(value);
    assert!(
        bytes <= ceiling,
        "{context}: {bytes} bytes exceeds the {ceiling} byte ceiling; the response grew: {value}"
    );
}

pub(crate) fn assert_error_mentions(value: &Value, needle: &str, context: &str) {
    let error = value["error"]
        .as_str()
        .unwrap_or_else(|| panic!("{context}: expected an error, got {value}"));
    assert!(
        error.to_lowercase().contains(&needle.to_lowercase()),
        "{context}: error {error:?} does not mention {needle:?}"
    );
}
