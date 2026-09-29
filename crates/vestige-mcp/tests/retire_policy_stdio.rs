//! Real vestige-mcp stdio: one allowed RETIRE and one held RETIRE, then
//! `vestige strata-verify` on that store and on a v3.1.1 migration.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio as ProcessStdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use strata_store::{AdmissionContext, RULE_SUPPRESS, StrataStore};

struct Stdio {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    seq: u64,
}

fn spawn(bin: &str, data_dir: &Path) -> Stdio {
    let mut cmd = Command::new(bin);
    cmd.arg("--no-http").arg("--data-dir").arg(data_dir);
    cmd.env_clear();
    for (key, value) in std::env::vars() {
        if key.starts_with("VESTIGE_") || key.starts_with("REDMINE_") || key.starts_with("GITHUB_")
        {
            continue;
        }
        cmd.env(key, value);
    }
    cmd.env("VESTIGE_DASHBOARD_ENABLED", "false");
    cmd.env("VESTIGE_HTTP_ENABLED", "false");
    cmd.env("RUST_LOG", "error");
    cmd.stdin(ProcessStdio::piped())
        .stdout(ProcessStdio::piped())
        .stderr(ProcessStdio::piped());
    let mut child = cmd.spawn().expect("spawn vestige-mcp");
    let stdout = child.stdout.take().expect("stdout");
    let stderr = child.stderr.take().expect("stderr");
    thread::spawn(move || {
        let mut sink = String::new();
        let _ = BufReader::new(stderr).read_to_string(&mut sink);
    });
    let (tx, lines) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let stdin = child.stdin.take().expect("stdin");
    Stdio {
        child,
        stdin,
        lines,
        seq: 0,
    }
}

impl Stdio {
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
        self.stdin.flush().expect("flush rpc");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("timed out waiting for {method}");
            }
            let line = self
                .lines
                .recv_timeout(remaining)
                .unwrap_or_else(|_| panic!("stdio closed waiting for {method}"));
            let value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if value.get("id") != Some(&serde_json::json!(id)) {
                continue;
            }
            assert!(value.get("error").is_none(), "{value}");
            return value["result"].clone();
        }
    }

    fn tool(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let result = self.rpc(
            "tools/call",
            serde_json::json!({ "name": name, "arguments": args }),
        );
        assert_eq!(
            result.get("isError"),
            Some(&serde_json::json!(false)),
            "{result}"
        );
        if let Some(body) = result
            .get("structuredContent")
            .filter(|body| body.get("nodeId").is_some())
        {
            return body.clone();
        }
        let text = result["content"][0]["text"].as_str().expect("tool text");
        serde_json::from_str(text).unwrap_or_else(|_| panic!("tool text is not json: {text}"))
    }

    fn close(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait().expect("wait") {
                assert!(status.success(), "vestige-mcp exit {status}");
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!("vestige-mcp did not exit after stdin EOF");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

fn ingest(mcp: &mut Stdio, content: &str) -> String {
    let created = mcp.tool(
        "smart_ingest",
        serde_json::json!({ "content": content, "forceCreate": true }),
    );
    created["nodeId"].as_str().expect("nodeId").to_string()
}

fn vestige_ok(args: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(args)
        .output()
        .expect("spawn vestige");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "vestige {args:?}\n{stdout}\n{stderr}"
    );
    if args.first() == Some(&"strata-verify") {
        assert!(stdout.contains("OK"), "{stdout}\n{stderr}");
    }
}

#[test]
fn stdio_allows_one_retire_and_holds_another_then_strata_verify() {
    let dir = std::env::temp_dir().join(format!("vestige-retire-stdio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("data dir");

    let (allowed_id, held_id, successor_id) = {
        let mut mcp = spawn(env!("CARGO_BIN_EXE_vestige-mcp"), &dir);
        mcp.rpc(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "retire-policy", "version": "1" },
            }),
        );
        writeln!(
            mcp.stdin,
            "{}",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#
        )
        .unwrap();
        mcp.stdin.flush().unwrap();
        let allowed_id = ingest(&mut mcp, "STDIO_RETIRE_ALLOWED");
        let held_id = ingest(&mut mcp, "STDIO_RETIRE_HELD");
        let successor_id = ingest(&mut mcp, "STDIO_RETIRE_SUCCESSOR");
        mcp.close();
        (allowed_id, held_id, successor_id)
    };

    {
        let mut store = StrataStore::open(&dir).expect("open stdio store");
        let receipt = store
            .retire(
                &allowed_id,
                &successor_id,
                &AdmissionContext {
                    rule_id: Some(RULE_SUPPRESS.to_string()),
                    confirm: false,
                },
            )
            .expect("suppress rule allows the RETIRE");
        assert_eq!(receipt.rule_id, Some(RULE_SUPPRESS));
        assert_eq!(
            receipt.receipt_id,
            strata_store::effect_receipt_id(receipt.effect_seq)
        );
        assert!(receipt.receipt_id.starts_with("eff-"));
        assert_eq!(
            store
                .get_node(&allowed_id)
                .expect("allowed")
                .superseded_by
                .as_deref(),
            Some(successor_id.as_str())
        );
        let err = store
            .retire(&held_id, &successor_id, &AdmissionContext::default())
            .expect_err("RETIRE with no rule id is held");
        assert!(
            matches!(err, strata_store::StoreError::Held { .. }),
            "{err}"
        );
        assert!(
            store
                .get_node(&held_id)
                .expect("held")
                .superseded_by
                .is_none()
        );
    }

    vestige_ok(&["strata-verify", dir.to_str().expect("utf8 dir")]);

    let migrated = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite");
    let dest = std::env::temp_dir().join(format!("vestige-retire-migrated-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dest);
    vestige_ok(&[
        "migrate-to-strata",
        "--from",
        migrated.to_str().expect("fixture path"),
        "--to",
        dest.to_str().expect("dest path"),
    ]);
    vestige_ok(&["strata-verify", dest.to_str().expect("dest path")]);

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dest).ok();
}
