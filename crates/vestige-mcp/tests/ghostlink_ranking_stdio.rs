//! Real-stdio proof for GhostLink ranking on a Strata log: the per-memory
//! page cap, proof strength putting a causal path above auto-connect
//! associations, the divergent rotating window's fields, `max_per_memory`
//! validation, and `explore` accepting `center_id` as an alias for `from`.

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
    /// A server on `data_dir` whose HOME is `home`, so a developer's own
    /// `~/.vestige/ghostlink-laws.json` can never leak into a test.
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
                "clientInfo": { "name": "ghostlink-ranking-stdio", "version": "1" },
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
    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
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

fn data_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

/// Save one memory under its own exact tag (no auto-connect), with optional
/// declared links.
fn save(server: &mut Server, content: &str, tags: Value, links: Value) -> String {
    let saved = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": content, "node_type": "fact", "tags": tags, "links": links }),
    );
    saved["nodeId"]
        .as_str()
        .unwrap_or_else(|| panic!("smart_ingest returned no nodeId: {saved}"))
        .to_string()
}

fn pairs(proposal: &Value) -> Vec<(String, String)> {
    proposal["candidates"]
        .as_array()
        .unwrap_or_else(|| panic!("no candidates array: {proposal}"))
        .iter()
        .map(|c| {
            (
                c["firstId"].as_str().unwrap().to_string(),
                c["secondId"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Five memories auto-connected on one shared exact tag, plus a causal
/// chain `a -derived_from-> x <-derived_from- b`.
struct Fixture {
    mates: Vec<String>,
    x: String,
    a: String,
    b: String,
}

fn fixture(server: &mut Server) -> Fixture {
    let mates: Vec<String> = (0..5)
        .map(|at| {
            save(
                server,
                &format!("tag mate {at}"),
                json!(["rank-star"]),
                json!([]),
            )
        })
        .collect();
    let x = save(server, "the shared cause", json!(["rank-x"]), json!([]));
    let a = save(
        server,
        "first effect",
        json!(["rank-a"]),
        json!([{ "kind": "derived_from", "to": x }]),
    );
    let b = save(
        server,
        "second effect",
        json!(["rank-b"]),
        json!([{ "kind": "derived_from", "to": x }]),
    );
    Fixture { mates, x, a, b }
}

#[test]
fn a_bridge_page_holds_each_memory_once_and_ranks_causal_over_auto_connect() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = Server::spawn(dir.path(), home.path());
    let fx = fixture(&mut server);

    // Default cap: one pair per memory on the page.
    let capped = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "limit": 3 }),
    );
    let got = pairs(&capped);
    assert_eq!(got.len(), 3, "{capped}");
    let mut members: Vec<&String> = got.iter().flat_map(|(a, b)| [a, b]).collect();
    members.sort();
    let before = members.len();
    members.dedup();
    assert_eq!(members.len(), before, "each memory at most once: {capped}");
    assert_eq!(
        capped["admission"]["perMemoryCap"]["max"],
        json!(1),
        "{capped}"
    );

    // Uncapped, the full ranking: the 2-hop causal pair a-b outranks every
    // 1-hop auto-connect pair between tag mates.
    let full = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "limit": 50, "max_per_memory": 50 }),
    );
    let candidates = full["candidates"].as_array().expect("candidates");
    let score_of = |a: &str, b: &str| -> f64 {
        candidates
            .iter()
            .find(|c| {
                key(
                    c["firstId"].as_str().unwrap(),
                    c["secondId"].as_str().unwrap(),
                ) == key(a, b)
            })
            .unwrap_or_else(|| panic!("{a},{b} not proposed: {full}"))["score"]
            .as_f64()
            .unwrap()
    };
    let causal = score_of(&fx.a, &fx.b);
    for (at, m) in fx.mates.iter().enumerate() {
        for n in &fx.mates[at + 1..] {
            let association = score_of(m, n);
            assert!(
                association < causal,
                "auto-connect {m},{n} ({association}) must rank below causal ({causal})"
            );
            assert!(association < 7.0, "no tie at the ceiling: {association}");
        }
    }
    let ab = candidates
        .iter()
        .find(|c| {
            key(
                c["firstId"].as_str().unwrap(),
                c["secondId"].as_str().unwrap(),
            ) == key(&fx.a, &fx.b)
        })
        .unwrap();
    assert_eq!(
        ab["proof"]["path"][0]["kind"],
        json!("derived_from"),
        "{ab}"
    );
    assert!(ab["proofStrength"].as_f64().unwrap() > 0.0, "{ab}");
    let mut scores: Vec<f64> = candidates
        .iter()
        .map(|c| c["score"].as_f64().unwrap())
        .collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "best first: {scores:?}"
    );
    scores.dedup();
    assert!(scores.len() >= 3, "scores spread: {scores:?}");
    assert!(pairs(&full).contains(&key(&fx.x, &fx.a)));
    server.shutdown();
}

#[test]
fn explore_accepts_center_id_for_from() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = Server::spawn(dir.path(), home.path());
    let fx = fixture(&mut server);
    let bridges = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "explore", "kind": "bridges", "center_id": fx.a, "to": fx.b }),
    );
    assert_eq!(bridges["bridges"], json!([fx.x]), "{bridges}");
    let associations = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "explore", "kind": "associations", "center_id": fx.x }),
    );
    assert_eq!(associations["from"], json!(fx.x), "{associations}");
    assert_eq!(associations["count"], json!(2), "{associations}");
    let missing = server.call_tool(
        "ghostlink",
        json!({ "mode": "explore", "kind": "bridges", "to": fx.b }),
    );
    assert!(
        missing["text"]
            .as_str()
            .is_some_and(|text| text.contains("center_id")),
        "the error names the alias: {missing}"
    );
    server.shutdown();
}

#[test]
fn divergent_pages_report_the_window_and_cap_and_validate_max_per_memory() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = Server::spawn(dir.path(), home.path());
    for at in 0..5 {
        save(
            &mut server,
            &format!("cold {at}"),
            json!([format!("cold-{at}")]),
            json!([]),
        );
    }
    let page = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 2 }),
    );
    assert_eq!(page["summary"]["measuredWindows"], json!(1), "{page}");
    assert_eq!(
        page["summary"]["measuredMembersBeyondCap"],
        json!(0),
        "{page}"
    );
    assert_eq!(page["admission"]["perMemoryCap"]["max"], json!(1), "{page}");
    assert!(
        page["admission"]["measuredWindow"].as_str().is_some(),
        "{page}"
    );
    let cursor = page["nextCursor"].as_str().expect("more pages");
    assert!(cursor.starts_with("gl2."), "{cursor}");
    let next = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 2, "cursor": cursor }),
    );
    assert!(
        pairs(&next).iter().all(|pair| !pairs(&page).contains(pair)),
        "no pair repeats across pages"
    );

    let zero = server.call_tool(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "max_per_memory": 0 }),
    );
    assert!(
        zero["text"]
            .as_str()
            .is_some_and(|text| text.contains("max_per_memory")),
        "{zero}"
    );
    let wrong_mode = server.call_tool(
        "ghostlink",
        json!({ "mode": "inspect", "view": "recent", "max_per_memory": 2 }),
    );
    assert!(
        wrong_mode["text"]
            .as_str()
            .is_some_and(|text| text.contains("max_per_memory")),
        "{wrong_mode}"
    );
    server.shutdown();
}
