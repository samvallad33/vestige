//! Real stdio: `dedup` action `policy` returns the saved Fellegi-Sunter policy.
//!
//! The shipped binary is the client. A modern `tools/call` must come back
//! `resultType: "complete"` with `isError: false` and the three policy fields,
//! including after the process is restarted against the same data directory.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let binary = server_binary();
        let mut command = Command::new(binary);
        command
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .env_remove("VESTIGE_MERGE_MATCH_THRESHOLD")
            .env_remove("VESTIGE_MERGE_POSSIBLE_THRESHOLD")
            .env_remove("VESTIGE_MERGE_AUTO_APPLY")
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
                    sink.lock().expect("stderr lock").push(line);
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

    fn stderr_text(&self) -> String {
        self.stderr
            .lock()
            .expect("stderr lock")
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{message}").expect("write request");
        stdin.flush().expect("flush request");

        loop {
            let line = match self.stdout.recv_timeout(RPC_TIMEOUT) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => panic!(
                    "no response within {RPC_TIMEOUT:?}. stderr: {}",
                    self.stderr_text()
                ),
                Err(RecvTimeoutError::Disconnected) => panic!(
                    "stdout closed without a response. stderr: {}",
                    self.stderr_text()
                ),
            };
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("non-JSON {line:?}: {error}"));
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            assert_eq!(value["id"], json!(id), "framing desync: {value}");
            return value;
        }
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            match self.child.try_wait().expect("poll vestige-mcp") {
                Some(status) => {
                    assert!(
                        status.success(),
                        "stdin EOF must exit 0, got {status}. stderr: {}",
                        self.stderr_text()
                    );
                    return;
                }
                None if std::time::Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!("vestige-mcp did not exit within 30s of stdin EOF");
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
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

fn server_binary() -> PathBuf {
    let source = Path::new(env!("CARGO_BIN_EXE_vestige-mcp"));
    let destination =
        std::env::temp_dir().join(format!("vestige-mcp-dedup-policy-{}", std::process::id()));
    std::fs::copy(source, &destination)
        .unwrap_or_else(|error| panic!("copy {} : {error}", source.display()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&destination).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&destination, permissions).unwrap();
    }
    destination
}

fn modern(mut params: Value) -> Value {
    params["_meta"] = json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
    });
    params
}

fn policy_call(server: &mut Server, arguments: Value) -> Value {
    let response = server.request(
        "tools/call",
        modern(json!({
            "name": "dedup",
            "arguments": arguments,
        })),
    );
    assert!(
        response.get("error").is_none(),
        "tools/call failed: {response}"
    );
    let result = &response["result"];
    assert_eq!(result["resultType"], json!("complete"), "{result}");
    assert_eq!(result["isError"], json!(false), "{result}");
    let body = result
        .get("structuredContent")
        .cloned()
        .unwrap_or_else(|| panic!("missing structuredContent: {result}"));
    let rendered = body.to_string();
    assert!(
        !rendered.contains("pending_strata") && !rendered.contains("not implemented"),
        "{body}"
    );
    body
}

fn assert_f32(body: &Value, key: &str, expected: f32) {
    let got = body[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} missing in {body}"));
    let expected = f64::from(expected);
    assert!(
        (got - expected).abs() <= 1e-6,
        "{key}: {got} vs {expected} in {body}"
    );
}

#[test]
fn dedup_policy_over_real_stdio_completes_and_survives_restart() {
    let dir = tempfile::tempdir().expect("data dir");
    let mut server = Server::spawn(dir.path());

    let listed = server.request("tools/list", modern(json!({})));
    assert!(listed.get("error").is_none(), "{listed}");
    assert_eq!(listed["result"]["resultType"], json!("complete"));
    let tools = listed["result"]["tools"].as_array().expect("tools");
    let dedup = tools
        .iter()
        .find(|tool| tool["name"] == "dedup")
        .expect("dedup is advertised");
    let actions = dedup["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .expect("action enum");
    assert!(
        actions.iter().any(|action| action == "policy"),
        "dedup must advertise policy: {actions:?}"
    );

    let defaults = policy_call(&mut server, json!({ "action": "policy" }));
    assert_f32(&defaults, "matchThreshold", 0.86);
    assert_f32(&defaults, "possibleThreshold", 0.72);
    assert_eq!(defaults["autoApply"], json!(false));
    assert!(defaults.get("updated").is_none(), "{defaults}");
    assert!(
        defaults["note"]
            .as_str()
            .is_some_and(|note| note.contains("Two-threshold")),
        "{defaults}"
    );

    let saved = policy_call(
        &mut server,
        json!({
            "action": "policy",
            "match_threshold": 0.91,
            "possible_threshold": 0.4,
            "auto_apply": true
        }),
    );
    assert_eq!(saved["updated"], json!(true));
    assert_f32(&saved, "matchThreshold", 0.91);
    assert_f32(&saved, "possibleThreshold", 0.4);
    assert_eq!(saved["autoApply"], json!(true));
    assert!(
        saved["note"]
            .as_str()
            .is_some_and(|note| note.contains("Fellegi-Sunter")),
        "{saved}"
    );

    // possible_threshold above match_threshold clamps down; the rule is unchanged.
    let clamped = policy_call(
        &mut server,
        json!({
            "action": "policy",
            "match_threshold": 0.5,
            "possible_threshold": 0.95,
            "auto_apply": false
        }),
    );
    assert_eq!(clamped["updated"], json!(true));
    assert_f32(&clamped, "matchThreshold", 0.5);
    assert_f32(&clamped, "possibleThreshold", 0.5);
    assert_eq!(clamped["autoApply"], json!(false));

    server.shutdown();

    let mut restarted = Server::spawn(dir.path());
    let again = policy_call(&mut restarted, json!({ "action": "policy" }));
    assert_f32(&again, "matchThreshold", 0.5);
    assert_f32(&again, "possibleThreshold", 0.5);
    assert_eq!(again["autoApply"], json!(false));
    assert!(again.get("updated").is_none(), "{again}");
    restarted.shutdown();
}
