//! `dedup` action `policy` over the default binary's stdio.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

#[test]
fn dedup_policy_action_completes_over_stdio() {
    let defaults = tempfile::tempdir().expect("defaults dir");
    let mut first = Session::spawn(defaults.path(), &[]);
    let got = first.policy(json!({ "action": "policy" }));
    assert!(got.get("error").is_none(), "{got}");
    assert_eq!(got["autoApply"], json!(false));
    assert_close(got["matchThreshold"].as_f64().unwrap(), 0.86);
    assert_close(got["possibleThreshold"].as_f64().unwrap(), 0.72);
    assert_eq!(
        got["note"],
        "Two-threshold merge policy. Pass match_threshold / possible_threshold / auto_apply to change it."
    );
    assert!(got.get("updated").is_none(), "{got}");
    drop(first);
    assert!(!defaults.path().join("merge-policy.json").exists());

    let saved = tempfile::tempdir().expect("saved dir");
    let env = [
        ("VESTIGE_MERGE_MATCH_THRESHOLD", "0.50"),
        ("VESTIGE_MERGE_POSSIBLE_THRESHOLD", "0.20"),
        ("VESTIGE_MERGE_AUTO_APPLY", "true"),
    ];
    let mut from_env = Session::spawn(saved.path(), &env);
    let env_policy = from_env.policy(json!({ "action": "policy" }));
    assert_close(env_policy["matchThreshold"].as_f64().unwrap(), 0.50);
    assert_close(env_policy["possibleThreshold"].as_f64().unwrap(), 0.20);
    assert_eq!(env_policy["autoApply"], json!(true));
    drop(from_env);

    let mut writer = Session::spawn(saved.path(), &env);
    let updated = writer.policy(json!({
        "action": "policy",
        "match_threshold": 0.4,
        "possible_threshold": 0.99,
        "auto_apply": false
    }));
    assert_eq!(updated["updated"], json!(true));
    assert_eq!(updated["autoApply"], json!(false));
    assert_close(updated["matchThreshold"].as_f64().unwrap(), 0.4);
    assert_close(updated["possibleThreshold"].as_f64().unwrap(), 0.4);
    assert!(
        updated["note"]
            .as_str()
            .unwrap()
            .starts_with("Policy saved.")
    );
    drop(writer);

    let mut reread = Session::spawn(saved.path(), &env);
    let persisted = reread.policy(json!({ "action": "policy" }));
    assert_eq!(persisted["autoApply"], json!(false));
    assert_close(persisted["matchThreshold"].as_f64().unwrap(), 0.4);
    assert_close(persisted["possibleThreshold"].as_f64().unwrap(), 0.4);
    assert!(persisted.get("updated").is_none(), "{persisted}");
    drop(reread);

    assert_no_sqlite(defaults.path());
    assert_no_sqlite(saved.path());
}

fn assert_close(got: f64, expect: f64) {
    assert!((got - expect).abs() < 1e-5, "{got} != {expect}");
}

fn assert_no_sqlite(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path).expect("read dir").flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            assert!(
                !name.ends_with(".sqlite")
                    && !name.ends_with(".sqlite3")
                    && !name.ends_with(".db")
                    && !name.ends_with(".db-wal")
                    && !name.ends_with(".db-shm"),
                "sqlite file: {path:?}"
            );
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
}

struct Session {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl Session {
    fn spawn(data_dir: &Path, extra: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"));
        command
            .arg("--no-http")
            .arg("--data-dir")
            .arg(data_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (key, _) in std::env::vars() {
            if key.starts_with("VESTIGE_")
                || key.starts_with("REDMINE_")
                || key.starts_with("GITHUB_")
            {
                command.env_remove(key);
            }
        }
        command
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "false")
            .env("RUST_LOG", "error");
        for (key, value) in extra {
            command.env(key, value);
        }
        let mut child = command.spawn().expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut session = Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        };
        session.rpc(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "dedup-policy-stdio", "version": "1"}
            }),
        );
        session
            .stdin
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .expect("initialized");
        session.stdin.flush().expect("flush");
        session
    }

    fn policy(&mut self, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({ "name": "dedup", "arguments": args }));
        assert_eq!(result["isError"], json!(false), "{result}");
        result["structuredContent"].clone()
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let request = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        writeln!(self.stdin, "{request}").expect("write rpc");
        self.stdin.flush().expect("flush rpc");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if Instant::now() >= deadline {
                panic!("timed out waiting for {method}");
            }
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("read stdout");
            if line.is_empty() {
                panic!("vestige-mcp exited during {method}");
            }
            let response: Value = serde_json::from_str(&line).expect("json line");
            if response.get("id").and_then(Value::as_u64) == Some(id) {
                assert!(response.get("error").is_none(), "{response}");
                return response["result"].clone();
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
