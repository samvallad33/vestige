//! Real stdio: `codebase action=ingest_repo` turns a git checkout into
//! anchored change records, previews by default, and is idempotent.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::Duration;

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    transcript: Vec<String>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let binary = PathBuf::from(env!("CARGO_BIN_EXE_vestige-mcp"));
        let mut child = Command::new(&binary)
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn {}: {err}", binary.display()));
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
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                eprintln!("stderr: {line}");
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            stdout,
            transcript: Vec::new(),
            next_id: 0,
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        let line = message.to_string();
        self.transcript.push(format!(">> {line}"));
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{line}")
            .and_then(|()| stdin.flush())
            .expect("write");
        loop {
            let line = match self.stdout.recv_timeout(Duration::from_secs(90)) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("stdio timeout\n{}", self.transcript.join("\n"))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdio closed\n{}", self.transcript.join("\n"))
                }
            };
            if line.is_empty() {
                continue;
            }
            self.transcript.push(format!("<< {line}"));
            let value: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            if value.get("id") == Some(&json!(id)) {
                assert!(value.get("error").is_none(), "{value}");
                return value["result"].clone();
            }
        }
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "unavailable-stdio", "version": "1" },
            })),
        );
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        let stdin = self.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{note}")
            .and_then(|()| stdin.flush())
            .expect("notify");
    }

    /// The whole tools/call result, errors included.
    fn call_raw(&mut self, name: &str, arguments: Value) -> Value {
        self.request(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        )
    }

    /// The structured result of a call that must succeed.
    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.call_raw(name, arguments);
        assert_ne!(result["isError"], true, "{name}: {result}");
        result["structuredContent"].clone()
    }

    fn list_tools(&mut self) -> Vec<Value> {
        self.request("tools/list", None)["tools"]
            .as_array()
            .expect("tools")
            .clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vestige-ingest-repo-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn git(repo: &Path, args: &[&str], date: &str) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_AUTHOR_DATE", date)
        .env("GIT_COMMITTER_DATE", date)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

const AUTH: &str =
    "pub fn load_config(path: &str) -> String {\n    let raw = path.to_string();\n    raw\n}\n";

fn fixture_repo(dir: &Path) -> Vec<String> {
    git(dir, &["init", "-q"], "2026-02-01T09:00:00+00:00");
    let mut shas = Vec::new();
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/auth.rs"), AUTH).unwrap();
    git(dir, &["add", "-A"], "2026-02-01T09:00:00+00:00");
    git(
        dir,
        &["commit", "-q", "-m", "add auth"],
        "2026-02-01T09:00:00+00:00",
    );
    shas.push(git(
        dir,
        &["rev-parse", "HEAD"],
        "2026-02-01T09:00:00+00:00",
    ));
    std::fs::write(
        dir.join("src/auth.rs"),
        AUTH.replace("path.to_string()", "path.trim().to_string()"),
    )
    .unwrap();
    git(dir, &["add", "-A"], "2026-02-02T09:00:00+00:00");
    git(
        dir,
        &["commit", "-q", "-m", "trim the path in load_config"],
        "2026-02-02T09:00:00+00:00",
    );
    shas.push(git(
        dir,
        &["rev-parse", "HEAD"],
        "2026-02-02T09:00:00+00:00",
    ));
    shas
}

#[test]
fn ingest_repo_end_to_end_over_the_real_binary() {
    let store = temp_dir("store");
    let repo = temp_dir("repo");
    let shas = fixture_repo(&repo);
    let repo_path = repo.to_str().unwrap();

    let mut server = Server::spawn(&store);
    server.handshake();

    // advertised: the wire schema keeps the selector enum and the tool
    // description; the full schema (with the preview default) is one call away
    let tools = server.list_tools();
    let codebase = tools.iter().find(|t| t["name"] == "codebase").unwrap();
    let actions = codebase["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    assert!(actions.contains(&json!("ingest_repo")), "{actions:?}");
    let description = codebase["description"].as_str().unwrap();
    assert!(
        description.contains("ingest_repo") && description.contains("previews unless dryRun=false"),
        "{description}"
    );
    let full = server.call_tool(
        "memory_status",
        json!({"view": "tools", "tool": "codebase"}),
    );
    assert!(
        full.to_string().contains("\"dryRun\""),
        "the full schema documents dryRun: {full}"
    );

    // no dryRun given: a preview, and nothing is written
    let preview = server.call_tool(
        "codebase",
        json!({"action": "ingest_repo", "repoPath": repo_path, "codebase": "demo"}),
    );
    assert_eq!(preview["dryRun"], true, "{preview}");
    assert_eq!(preview["commits"]["new"], 2);
    assert_eq!(preview["commits"]["created"], 0);
    let missing = server.call_raw("recall", json!({"handle": format!("commit:{}", shas[0])}));
    assert!(
        missing.to_string().contains("handle_required"),
        "a preview wrote a record: {missing}"
    );

    // write
    let written = server.call_tool(
        "codebase",
        json!({"action": "ingest_repo", "repoPath": repo_path, "codebase": "demo", "dryRun": false}),
    );
    assert_eq!(written["commits"]["created"], 2, "{written}");
    assert_eq!(written["scope"], "demo");
    assert_eq!(written["partial"], false);
    assert_eq!(
        written["anchors"]["recorded"], preview["anchors"]["wouldRecord"],
        "the preview predicted the write"
    );

    // found by exact handle, never by text
    let found = server.call_tool("recall", json!({"handle": format!("commit:{}", shas[1])}));
    let node = &found["nodes"][0];
    assert!(
        node["content"]
            .as_str()
            .unwrap()
            .starts_with(&format!("commit {}", shas[1])),
        "{found}"
    );
    let tags: Vec<&str> = node["tags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert!(
        tags.contains(&"git-commit") && tags.contains(&"codebase:demo"),
        "{tags:?}"
    );

    // provenance names where the record came from
    let id = node["id"].as_str().unwrap();
    let provenance = server.call_tool(
        "memory_status",
        json!({"view": "provenance", "memoryId": id}),
    );
    let text = provenance.to_string();
    assert!(
        text.contains("git") && text.contains(&shas[1]),
        "{provenance}"
    );

    // the shared user scope was not touched
    let user = server.call_tool(
        "codebase",
        json!({"action": "get_context", "codebase": "demo"}),
    );
    assert_eq!(user["scope"], "user");

    // verify watches the change records; the located symbol is fresh
    let verify = server.call_tool(
        "codebase",
        json!({"action": "verify", "codebase": "demo", "scope": "demo", "repoPath": repo_path}),
    );
    assert_eq!(verify["checked"], 2, "{verify}");
    assert_eq!(verify["stale"], 0, "{verify}");
    assert_eq!(verify["fresh"], 1, "{verify}");

    // the code a commit touched changes: the record drifts
    std::fs::write(
        repo.join("src/auth.rs"),
        "pub fn load_config(path: &str) -> String {\n    String::new()\n}\n",
    )
    .unwrap();
    let drifted = server.call_tool(
        "codebase",
        json!({"action": "verify", "codebase": "demo", "scope": "demo", "repoPath": repo_path}),
    );
    assert_eq!(drifted["stale"], 1, "{drifted}");
    assert_eq!(drifted["staleMemories"][0]["nodeType"], "event");

    // a rerun is idempotent
    let again = server.call_tool(
        "codebase",
        json!({"action": "ingest_repo", "repoPath": repo_path, "codebase": "demo", "dryRun": false}),
    );
    assert_eq!(again["commits"]["created"], 0, "{again}");
    assert_eq!(again["commits"]["alreadyIngested"], 2);

    // a bad ref is refused, not handed to git
    let refused = server.call_raw(
        "codebase",
        json!({"action": "ingest_repo", "repoPath": repo_path, "rev": "--output=x"}),
    );
    assert_eq!(refused["isError"], true, "{refused}");

    println!("{}", server.transcript.join("\n"));
    let _ = std::fs::remove_dir_all(&store);
    let _ = std::fs::remove_dir_all(&repo);
}
