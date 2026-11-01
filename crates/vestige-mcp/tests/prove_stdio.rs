//! Real-stdio proof for proof-carrying pull requests.
//!
//! Spawns the shipped `vestige-mcp` binary on a fresh data directory with an
//! allowlisted test command, fires the Reflex hooks a Claude Code session
//! fires around that command, asks `receipt(action='prove')` for the bundle,
//! and checks it two ways: with the library verifier, and with
//! `vestige verify-proof` while the server still owns the store, which works
//! only because the verifier never opens the store.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Receiver<String>,
    next_id: u64,
}

impl Server {
    /// A server on `data_dir` whose HOME is `home`, so nothing of the
    /// developer's own setup can leak into a test.
    fn spawn(data_dir: &Path, home: &Path) -> Self {
        Self::spawn_with_trace(data_dir, home, false)
    }

    fn spawn_with_trace(data_dir: &Path, home: &Path, tracing: bool) -> Self {
        let mut child = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_vestige-mcp")))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("HOME", home)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE", if tracing { "1" } else { "0" })
            .env("VESTIGE_BACKFILL_AUTOFIRE", "0")
            .env("VESTIGE_FAILURE_FEEDBACK", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("child stdin");
        let raw_stdout = child.stdout.take().expect("child stdout");
        let raw_stderr = child.stderr.take().expect("child stderr");
        let (stdout_tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if stdout_tx.send(line).is_err() {
                    return;
                }
            }
        });
        let (stderr_tx, stderr) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                if stderr_tx.send(line).is_err() {
                    return;
                }
            }
        });
        let mut server = Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        };
        server.result(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "prove-stdio", "version": "1" },
            })),
        );
        server.write_line(
            &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }).to_string(),
        );
        server
    }

    fn stderr_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.stderr.try_recv() {
            lines.push(line);
        }
        lines
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| panic!("writing to vestige-mcp failed: {error}"));
    }

    fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
        let deadline = Instant::now() + RPC_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = self.stdout.recv_timeout(remaining).unwrap_or_else(|_| {
                panic!("no response to {method}. stderr: {:?}", self.stderr_lines())
            });
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("stdout was not JSON ({error}): {line}"));
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            assert_eq!(value["id"], json!(id), "unexpected response: {value}");
            assert!(value.get("error").is_none(), "{method} failed: {value}");
            return value["result"].clone();
        }
    }

    /// A tool's payload; a tool-level failure comes back as `{isError, text}`.
    fn call_tool(&mut self, name: &str, mut arguments: Value) -> Value {
        // These checks read the full save report, which smart_ingest returns in its standard profile.
        if name == "smart_ingest" && arguments.get("outputProfile").is_none() {
            arguments["outputProfile"] = json!("standard");
        }
        let result = self.result(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        );
        if result["isError"] == json!(true) {
            return json!({ "isError": true, "text": result["content"][0]["text"] });
        }
        result
            .get("structuredContent")
            .cloned()
            .or_else(|| {
                result["content"][0]["text"]
                    .as_str()
                    .and_then(|text| serde_json::from_str(text).ok())
            })
            .unwrap_or_else(|| panic!("tool {name} returned no JSON payload: {result}"))
    }

    fn call_tool_ok(&mut self, name: &str, arguments: Value) -> Value {
        let value = self.call_tool(name, arguments);
        assert!(
            value.get("isError").is_none() && value.get("error").is_none(),
            "tool {name} failed: {value}"
        );
        value
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        while self.child.try_wait().expect("poll vestige-mcp").is_none() {
            assert!(
                Instant::now() < deadline,
                "vestige-mcp did not exit on stdin EOF"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn hook(server: &mut Server, event: &str, repo: &Path, extra: Value) -> Value {
    let mut args = json!({
        "event": event, "session": "s-stdio", "tool": "Bash", "call": "toolu_stdio",
        "command": "cargo test --workspace", "cwd": repo.to_string_lossy(), "host": "claude",
    });
    for (k, v) in extra.as_object().unwrap() {
        args[k] = v.clone();
    }
    server.call_tool("reflex", args)
}

#[test]
fn an_agent_test_run_becomes_a_bundle_the_offline_verifier_accepts() {
    let data = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    std::fs::write(
        data.path().join("vestige.toml"),
        "[witness]\ncommands = [\"cargo test --workspace\"]\n",
    )
    .unwrap();
    let repo = tempfile::tempdir().unwrap();
    for args in [
        vec!["init", "-q", "-b", "main"],
        vec!["config", "user.email", "test@example.com"],
        vec!["config", "user.name", "Test"],
        vec!["config", "commit.gpgsign", "false"],
    ] {
        git(repo.path(), &args);
    }
    std::fs::write(repo.path().join("lib.rs"), "fn add() {}\n").unwrap();
    git(repo.path(), &["add", "-A"]);
    git(repo.path(), &["commit", "-q", "-m", "first"]);
    let tree = git(repo.path(), &["rev-parse", "HEAD^{tree}"]);

    let mut server = Server::spawn(data.path(), home.path());
    hook(&mut server, "PreToolUse", repo.path(), json!({}));
    let told = hook(
        &mut server,
        "PostToolUse",
        repo.path(),
        json!({"stdout": "test result: ok. 1 passed\n", "stderr": ""}),
    );
    let text = told.to_string();
    assert!(
        text.contains("witnessed `cargo test --workspace` green"),
        "{told}"
    );

    let out = server.call_tool_ok(
        "receipt",
        json!({"action": "prove", "repoPath": repo.path().to_string_lossy(), "commit": "HEAD"}),
    );
    assert_eq!(out["tree"], tree, "{out}");
    assert_eq!(out["summary"]["green"], 1, "{out}");
    let signer = out["signer"].as_str().unwrap().to_string();
    let actor = vestige_core::actor::ProcessActor::load(&data.path().join("actor.key")).unwrap();
    assert_eq!(
        signer,
        actor.did(),
        "bundles are signed by the store's own actor key"
    );

    let file = repo
        .path()
        .join(vestige_proof::PROOF_DIR)
        .join(vestige_proof::bundle_file_name(&tree));
    let bytes = std::fs::read(&file).unwrap();
    let key = vestige_proof::didkey::TrustedKey::parse(&signer).unwrap();
    let report = vestige_proof::verify::verify_bundle(
        &bytes,
        &[key],
        Some(&tree),
        &vestige_proof::verify::Policy::default(),
    );
    assert!(report.ok, "{:?}", report.failures);

    // The CLI verifier while this server still holds the store.
    let cli = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_vestige")))
        .args(["verify-proof", "--repo"])
        .arg(repo.path())
        .args(["--trust", &signer])
        .env("VESTIGE_DATA_DIR", data.path())
        .env("HOME", home.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&cli.stdout);
    assert_eq!(cli.status.code(), Some(0), "{cli:?}");
    assert!(stdout.contains("PASS"), "{stdout}");

    std::fs::write(&file, {
        let mut bad = bytes.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 0x01;
        bad
    })
    .unwrap();
    let flipped = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_vestige")))
        .args(["verify-proof", "--repo"])
        .arg(repo.path())
        .args(["--trust", &signer])
        .output()
        .unwrap();
    assert_eq!(flipped.status.code(), Some(1), "{flipped:?}");
    server.shutdown();
}
