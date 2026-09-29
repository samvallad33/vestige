//! Real stdio: `maintain` action `dream_compile` completes from recorded edges.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use vestige_core::ConnectionRecord;

struct Server {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl Server {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "0")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        }
    }

    fn send(&mut self, message: &Value) {
        let stdin = self.stdin.as_mut().expect("stdin");
        stdin
            .write_all(message.to_string().as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .expect("write stdio");
    }

    fn read(&mut self) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if Instant::now() > deadline {
                panic!("stdio timeout");
            }
            let mut line = String::new();
            self.stdout.read_line(&mut line).expect("read stdio");
            if line.is_empty() {
                panic!("vestige-mcp closed stdout");
            }
            let value: Value = serde_json::from_str(&line).expect("json");
            if value.get("id").is_none() && value.get("method").is_some() {
                continue;
            }
            return value;
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        let response = self.read();
        assert_eq!(response["id"], json!(id));
        assert!(response.get("error").is_none(), "{method}: {response}");
        response["result"].clone()
    }

    fn handshake(&mut self) {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "dream-compile-stdio", "version": "1"}
            }),
        );
        self.send(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        if let Some(structured) = result.get("structuredContent") {
            return structured.clone();
        }
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if self.child.try_wait().expect("poll").is_some() {
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                panic!("vestige-mcp did not exit");
            }
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

fn link(storage: &vestige_core::Storage, source: &str, target: &str, link_type: &str, milli: i64) {
    let at = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(1_700_000_000_000).unwrap();
    storage
        .save_connection(&ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: milli as f64 / 1000.0,
            link_type: link_type.to_string(),
            created_at: at,
            last_activated: at,
            activation_count: 1,
        })
        .unwrap();
}

fn verify_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-verify/target/debug/strata-verify")
}

#[test]
fn dream_compile_stdio_completes_from_recorded_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut ids = Vec::new();
    {
        let mut server = Server::spawn(dir.path());
        server.handshake();
        let listed = server.request("tools/list", json!({}));
        let maintain = listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "maintain")
            .expect("maintain");
        let advertised = maintain.to_string();
        assert!(
            advertised.contains("dream_compile"),
            "maintain must advertise dream_compile"
        );
        for index in 0..6 {
            let created = server.call(
                "smart_ingest",
                json!({
                    "content": format!("dream compile stdio memory {index}"),
                    "forceCreate": true
                }),
            );
            assert_eq!(created["success"], json!(true), "{created}");
            ids.push(created["nodeId"].as_str().unwrap().to_string());
        }
        server.shutdown();
    }

    {
        let storage = vestige_mcp::strata_memory::open(dir.path()).unwrap();
        // First ingest has the lowest retrievability. The other five are the replay set.
        link(storage.as_ref(), &ids[4], &ids[5], "derived_from", 600);
        link(storage.as_ref(), &ids[1], &ids[0], "derived_from", 400);
        link(storage.as_ref(), &ids[2], &ids[3], "corrects", 800);
    }

    let report = {
        let mut server = Server::spawn(dir.path());
        server.handshake();
        let report = server.call(
            "maintain",
            json!({"action": "dream_compile", "memory_count": 5}),
        );
        server.shutdown();
        report
    };

    assert_eq!(report["status"], "compiled", "{report}");
    assert_eq!(report["memoriesSelected"], 5);
    assert_eq!(report["edgesStrengthened"], 2);
    assert_eq!(report["edgesDownscaled"], 1);
    assert_eq!(report["contradictionsFound"], 1);
    assert_eq!(report["insightsGenerated"], 1);
    assert_eq!(report["prsFiled"].as_array().unwrap().len(), 0);
    assert_eq!(report["dreamHistoryRecorded"], false);
    assert_eq!(report["phases"].as_array().unwrap().len(), 4);
    assert!(!report.to_string().contains("pending_strata"), "{report}");
    assert!(!report.to_string().contains("not implemented"), "{report}");

    // Live-store verify keys off store.meta, which a checkpoint seal writes.
    {
        let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
        store.seal_checkpoint().unwrap();
    }
    let bin = verify_bin();
    assert!(
        bin.is_file(),
        "build strata-verify before this test: {}",
        bin.display()
    );
    let verified = Command::new(&bin)
        .arg(dir.path())
        .output()
        .expect("run strata-verify");
    assert!(
        verified.status.success(),
        "strata-verify failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let stdout = String::from_utf8_lossy(&verified.stdout);
    assert!(stdout.contains("OK"), "{stdout}");
}
