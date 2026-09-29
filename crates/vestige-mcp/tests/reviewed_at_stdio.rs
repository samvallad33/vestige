//! Real vestige-mcp stdio: an explicit review lands `reviewed_at_ms` on the
//! signed `ReviewNode` payload, and `vestige strata-verify` accepts the store.
//!
//! `VESTIGE_MCP_BIN` defaults to the package binary. `VESTIGE_BIN` is the
//! `vestige` binary built with `--features migrate-to-strata`.

#![cfg(feature = "migrate-to-strata")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use strata_store::StrataStore;

fn bin(env_var: &str, cargo_exe: Option<&str>, name: &str) -> PathBuf {
    if let Some(path) = std::env::var_os(env_var) {
        return PathBuf::from(path);
    }
    if let Some(path) = cargo_exe {
        return PathBuf::from(path);
    }
    // Clippy does not define `CARGO_BIN_EXE_*`. The workspace target dir is
    // two levels above this crate.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/debug")
        .join(name)
}

fn mcp_bin() -> PathBuf {
    bin(
        "VESTIGE_MCP_BIN",
        option_env!("CARGO_BIN_EXE_vestige-mcp"),
        "vestige-mcp",
    )
}

fn vestige_bin() -> PathBuf {
    bin(
        "VESTIGE_BIN",
        option_env!("CARGO_BIN_EXE_vestige"),
        "vestige",
    )
}

struct StdioServer {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    seq: u64,
}

impl StdioServer {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(mcp_bin())
            .args(["--no-http", "--data-dir"])
            .arg(data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "false")
            .env("RUST_LOG", "error")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn {}: {err}", mcp_bin().display()));
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            seq: 0,
        }
    }

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
        self.stdin.flush().unwrap();
        let deadline = SystemTime::now() + Duration::from_secs(60);
        loop {
            if SystemTime::now() > deadline {
                panic!("timeout waiting for {method}");
            }
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).expect("read rpc");
            assert!(n > 0, "vestige-mcp closed stdout");
            let response: serde_json::Value = serde_json::from_str(line.trim()).expect("json");
            if response.get("id").and_then(|v| v.as_u64()) == Some(id) {
                assert!(response.get("error").is_none(), "{response}");
                return response["result"].clone();
            }
        }
    }

    fn notify(&mut self, method: &str) {
        let note = serde_json::json!({"jsonrpc": "2.0", "method": method});
        writeln!(self.stdin, "{note}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn tool(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let result = self.rpc(
            "tools/call",
            serde_json::json!({"name": name, "arguments": args}),
        );
        assert_ne!(result["isError"], serde_json::json!(true), "{result}");
        result
            .get("structuredContent")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or_else(|| {
                let text = result["content"][0]["text"].as_str().unwrap_or("{}");
                serde_json::from_str(text).unwrap_or(result)
            })
    }
}

impl Drop for StdioServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

#[test]
fn stdio_review_stamps_reviewed_at_ms_and_verifies() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let started = now_ms();
    let node_id = {
        let mut server = StdioServer::spawn(data);
        server.rpc(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "reviewed-at", "version": "1"},
            }),
        );
        server.notify("notifications/initialized");
        let created = server.tool(
            "smart_ingest",
            serde_json::json!({
                "content": "stdio review clock fixture",
                "forceCreate": true,
            }),
        );
        let node_id = created["nodeId"].as_str().expect("nodeId").to_string();
        assert!(node_id.starts_with("mem-"), "{node_id}");
        let reviewed = server.tool(
            "mark_reviewed",
            serde_json::json!({"id": node_id, "rating": 4}),
        );
        assert_eq!(reviewed["success"], serde_json::json!(true), "{reviewed}");
        node_id
    };

    let store = StrataStore::open(data).expect("reopen strata store");
    let stamped = store.reviewed_at_ms(&node_id).expect("reviewed_at_ms");
    let finished = now_ms();
    assert!(
        (started..=finished + 5_000).contains(&stamped),
        "admission clock {stamped} not in [{started}, {finished}]"
    );

    let verify = Command::new(vestige_bin())
        .args(["strata-verify"])
        .arg(data)
        .output()
        .expect("run vestige strata-verify");
    let stdout = String::from_utf8_lossy(&verify.stdout);
    let stderr = String::from_utf8_lossy(&verify.stderr);
    assert!(
        verify.status.success(),
        "strata-verify failed: {stdout}{stderr}"
    );
    assert!(stdout.contains("OK"), "{stdout}");
}

#[test]
fn strata_verify_accepts_migrated_v3_1_1_store() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    let dir = tempfile::tempdir().expect("tempdir");
    let dest = dir.path().join("strata");
    let migrate = Command::new(vestige_bin())
        .args(["migrate-to-strata", "--from"])
        .arg(&fixture)
        .arg("--to")
        .arg(&dest)
        .output()
        .expect("run vestige migrate-to-strata");
    let migrate_out = String::from_utf8_lossy(&migrate.stdout);
    let migrate_err = String::from_utf8_lossy(&migrate.stderr);
    assert!(
        migrate.status.success(),
        "migrate failed: {migrate_out}{migrate_err}"
    );
    let verify = Command::new(vestige_bin())
        .args(["strata-verify"])
        .arg(&dest)
        .output()
        .expect("run vestige strata-verify");
    let stdout = String::from_utf8_lossy(&verify.stdout);
    let stderr = String::from_utf8_lossy(&verify.stderr);
    assert!(
        verify.status.success(),
        "strata-verify failed: {stdout}{stderr}"
    );
    assert!(stdout.contains("OK"), "{stdout}");
}
