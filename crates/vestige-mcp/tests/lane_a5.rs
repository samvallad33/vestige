//! Lane A5: capsule routing, the proof header, and the capsule installer.
//! These run the built binaries, so they compile on the default build.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(90);

struct Client {
    name: &'static str,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Client {
    fn spawn(name: &'static str, data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                    sink.lock().unwrap().push(line);
                }
            });
        }
        Self {
            name,
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr(&self) -> String {
        self.stderr.lock().unwrap().join("\n")
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
        )
        .unwrap();
        stdin.flush().unwrap();
        let deadline = Instant::now() + RPC_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.stdout.recv_timeout(left) {
                Ok(line) => {
                    let message: Value = serde_json::from_str(&line)
                        .unwrap_or_else(|err| panic!("{}: not JSON ({err}): {line}", self.name));
                    if message.get("method").is_none() && message["id"] == json!(id) {
                        return message;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("{}: no answer to {method}\n{}", self.name, self.stderr())
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!(
                        "{}: stdout closed on {method}\n{}",
                        self.name,
                        self.stderr()
                    )
                }
            }
        }
    }

    fn initialize(&mut self) -> String {
        let answer = self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": self.name, "version": "1"}
            }),
        );
        self.request_notify();
        answer["result"]["instructions"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: initialize: {answer}", self.name))
            .to_string()
    }

    fn request_notify(&mut self) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        .unwrap();
        stdin.flush().unwrap();
    }

    fn call(&mut self, tool: &str, arguments: Value) -> Value {
        let answer = self.request("tools/call", json!({"name": tool, "arguments": arguments}));
        assert!(
            answer.get("error").is_none() && answer["result"]["isError"] != json!(true),
            "{}: {tool} failed: {answer}\n{}",
            self.name,
            self.stderr()
        );
        if !answer["result"]["structuredContent"].is_null() {
            return answer["result"]["structuredContent"].clone();
        }
        let text = answer["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or("null");
        serde_json::from_str(text).unwrap_or(Value::String(text.to_string()))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "Capsule")
        .env("GIT_AUTHOR_EMAIL", "capsule@example.com")
        .env("GIT_COMMITTER_NAME", "Capsule")
        .env("GIT_COMMITTER_EMAIL", "capsule@example.com")
        .status()
        .unwrap();
    assert!(status.success(), "{args:?}");
}

fn capsule(data: &Path, repo: &Path) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(data)
        .args([
            "capsule",
            "--harness",
            "claude-code",
            "--event",
            "session-start",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        write!(
            stdin,
            "{}",
            json!({"source": "startup", "cwd": repo, "hook_event_name": "SessionStart"})
        )
        .unwrap();
    }
    child.wait_with_output().unwrap()
}

#[test]
fn capsule_routes_through_the_owner_and_is_byte_stable() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    git(&repo, &["init"]);
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn parse(raw: &str) -> &str { raw }\n",
    )
    .unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "parse"]);
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();

    let mut owner = Client::spawn("owner", &data);
    owner.initialize();
    let saved = owner.call(
        "codebase",
        json!({
            "action": "remember_decision",
            "codebase": "fixture",
            "decision": "parse stays borrowed",
            "rationale": "the caller owns the string",
            "files": ["src/lib.rs#parse"],
            "repoPath": repo,
        }),
    );
    let id = saved["nodeId"].as_str().unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn parse(raw: &str) -> &str { raw.trim() }\n",
    )
    .unwrap();

    let first = capsule(&data, &repo);
    let second = capsule(&data, &repo);
    assert!(
        first.status.success(),
        "stderr {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(second.status.success());
    assert_eq!(first.stdout, second.stdout);
    let parsed: Value = serde_json::from_slice(&first.stdout).expect("hook json");
    let body = parsed["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert_eq!(
        parsed["hookSpecificOutput"]["hookEventName"],
        "SessionStart"
    );
    assert!(body.contains(&format!("`vst:{id}`")), "{body}");
    assert!(body.contains("eff-"), "{body}");
    assert!(body.contains("src/lib.rs"), "{body}");
    assert!(body.contains("vst:head:"), "{body}");

    let stats = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(&data)
        .arg("stats")
        .output()
        .unwrap();
    assert!(
        !stats.status.success(),
        "stats must refuse while the owner holds the log"
    );
    let stats_err = String::from_utf8_lossy(&stats.stderr);
    assert!(
        stats_err.contains("serving")
            || stats_err.contains("writer")
            || stats_err.contains("server"),
        "{stats_err}"
    );

    drop(owner);
    let verify = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("strata-verify")
        .arg(&data)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "stderr {}",
        String::from_utf8_lossy(&verify.stderr)
    );
}

#[test]
fn initialize_instructions_name_the_current_head() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut owner = Client::spawn("A", &data);
    let a = owner.initialize();
    assert!(a.contains("Causal Proof Engine"), "{a}");
    assert!(a.contains("PROOF `vst:head:"), "{a}");
    assert!(a.contains("keep every `vst:` handle verbatim"));
    let discover_a = owner.request("server/discover", json!({}));
    assert_eq!(discover_a["result"]["cacheScope"], "public");
    let instructions_a = discover_a["result"]["instructions"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        !instructions_a.contains("PROOF `vst:head:"),
        "{instructions_a}"
    );

    owner.call(
        "smart_ingest",
        json!({"content": "head moves", "forceCreate": true}),
    );
    let mut b = Client::spawn("B", &data);
    let b_text = b.initialize();
    let mut c = Client::spawn("C", &data);
    let c_text = c.initialize();
    assert_ne!(a, b_text);
    let seq = |text: &str| {
        let start = text.find("vst:head:").unwrap() + "vst:head:".len();
        u64::from_str_radix(&text[start..start + 16], 16).unwrap()
    };
    assert!(seq(&b_text) > seq(&a), "B {b_text}\nA {a}");
    let b_header = b_text.rsplit_once('\n').unwrap().1;
    let c_header = c_text.rsplit_once('\n').unwrap().1;
    assert_eq!(b_header, c_header);
    let discover_b = b.request("server/discover", json!({}));
    assert_eq!(
        discover_b["result"]["instructions"].as_str().unwrap(),
        instructions_a
    );
    let resource = b.request("resources/read", json!({"uri": "memory://proof-header"}));
    let resource_text = resource["result"]["contents"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("{resource}"));
    assert_eq!(resource_text, b_header);
}

#[test]
fn enable_capsule_wires_a_sessionstart_command_hook() {
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".claude");
    std::fs::create_dir_all(&claude).unwrap();
    let settings = claude.join("settings.json");
    let user = json!({
        "hooks": {
            "SessionStart": [{
                "matcher": "startup",
                "hooks": [{"type": "command", "command": "echo mine", "timeout": 1}]
            }]
        }
    });
    std::fs::write(&settings, serde_json::to_vec_pretty(&user).unwrap()).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_vestige"))
            .env("HOME", home.path())
            .args(args)
            .output()
            .unwrap()
    };
    let installed = run(&[
        "sandwich",
        "install",
        "--src",
        root.to_str().unwrap(),
        "--enable-capsule",
    ]);
    assert!(
        installed.status.success(),
        "stdout {}\nstderr {}",
        String::from_utf8_lossy(&installed.stdout),
        String::from_utf8_lossy(&installed.stderr)
    );
    let text = std::fs::read_to_string(&settings).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let groups = value["hooks"]["SessionStart"].as_array().unwrap();
    assert_eq!(groups[0]["hooks"][0]["command"], "echo mine");
    let capsule = groups.iter().find(|group| {
        group["hooks"][0]["command"]
            == "vestige capsule --harness claude-code --event session-start"
    });
    let capsule = capsule.expect(&text);
    assert_eq!(capsule["matcher"], "startup|resume|clear|compact|fork");
    assert_eq!(capsule["hooks"][0]["type"], "command");
    let again = run(&[
        "sandwich",
        "install",
        "--src",
        root.to_str().unwrap(),
        "--enable-capsule",
    ]);
    assert!(
        again.status.success(),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert_eq!(std::fs::read(&settings).unwrap(), text.into_bytes());
    let removed = run(&["sandwich", "install", "--src", root.to_str().unwrap()]);
    assert!(
        removed.status.success(),
        "{}",
        String::from_utf8_lossy(&removed.stderr)
    );
    let after: Value = serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
    let left = after["hooks"]["SessionStart"].as_array().unwrap();
    assert!(left.iter().all(|group| group["hooks"][0]["command"]
        != "vestige capsule --harness claude-code --event session-start"));
    assert_eq!(left[0]["hooks"][0]["command"], "echo mine");

    std::fs::write(&settings, "{not json").unwrap();
    let refused = run(&[
        "sandwich",
        "install",
        "--src",
        root.to_str().unwrap(),
        "--enable-capsule",
    ]);
    assert!(!refused.status.success());
    assert_eq!(std::fs::read_to_string(&settings).unwrap(), "{not json");
}
