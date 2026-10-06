//! Real-stdio proof for `ghostlink` on a Strata log.
//!
//! Spawns the shipped `vestige-mcp` binary on a fresh data directory and
//! checks, over line-framed JSON-RPC: the advertised surface (ghostlink in,
//! graph out, 16 tools), both propose lenses with their proofs, weave writing
//! through the gate with a receipt per write and removing the pair, inspect,
//! harden seeding once and then reporting every law as already present, a
//! laws file taking precedence, a malformed laws file naming itself, the
//! hidden graph alias still answering, and explore chain keeping the legacy
//! contract (every memory on the chain, origin first; the stable no-chain
//! message).

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
        let mut child = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_vestige-mcp")))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("HOME", home)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE", "0")
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
                "clientInfo": { "name": "ghostlink-stdio", "version": "1" },
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

fn spawn(dir: &Path, home: &Path) -> Server {
    Server::spawn(dir, home)
}

/// Seed one memory. Each seed carries its own tag: an exact shared tag would
/// auto-connect the pairs at save time, and these tests assert a cold log
/// until they write their own edges. Words the seeds share never join them
/// (see `smart_ingest_auto_connects_on_exact_identities_only`).
fn save(server: &mut Server, content: &str, node_type: &str) -> String {
    static SEED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let tag = format!(
        "ghostlink-seed-{}",
        SEED.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let saved = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": content, "node_type": node_type, "tags": [tag] }),
    );
    saved["nodeId"]
        .as_str()
        .unwrap_or_else(|| panic!("smart_ingest returned no nodeId: {saved}"))
        .to_string()
}

/// smart_ingest's ingest-time auto-connect joins on exact identities only
/// and explains each edge: memories sharing most of their words are not
/// joined, an exact shared tag is, and the response names the pair and the
/// identity that joined it.
#[test]
fn smart_ingest_auto_connects_on_exact_identities_only() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());

    let ingest = |server: &mut Server, content: &str, tags: Value| {
        server.call_tool_ok(
            "smart_ingest",
            json!({ "content": content, "node_type": "fact", "tags": tags }),
        )
    };

    let first = ingest(
        &mut server,
        "streaming usage parser drops cached_tokens for the gateway",
        json!(["exact-a"]),
    );
    assert!(first.get("autoConnect").is_none(), "{first}");

    // Nearly the same words, a different tag: no identity is shared.
    let words = ingest(
        &mut server,
        "streaming usage parser drops cached_tokens for the gateway again",
        json!(["exact-b"]),
    );
    assert!(words.get("autoConnect").is_none(), "{words}");
    assert!(words.get("autoConnectError").is_none(), "{words}");

    // No word in common with the first memory, the same exact tag.
    let tagged = ingest(
        &mut server,
        "unrelated wording entirely",
        json!(["exact-a"]),
    );
    let report = &tagged["autoConnect"];
    assert_eq!(report["edges"], json!(1), "{tagged}");
    assert_eq!(
        report["sharedIdentities"],
        json!(["tag:exact-a"]),
        "{tagged}"
    );
    assert_eq!(
        report["pairs"],
        json!([{
            "source": first["nodeId"],
            "target": tagged["nodeId"],
            "joinedOn": ["tag:exact-a"],
        }]),
        "{tagged}"
    );

    // A third carrier of the tag joins both earlier ones in ONE write, and
    // the response names the receipt that lists both edges.
    let third = ingest(&mut server, "a third wording", json!(["exact-a"]));
    let report = &third["autoConnect"];
    assert_eq!(report["edges"], json!(2), "{third}");
    let receipt_id = report["receiptId"]
        .as_str()
        .unwrap_or_else(|| panic!("autoConnect names no receipt: {third}"));
    assert!(receipt_id.starts_with("eff-"), "{third}");
    let receipt = server.call_tool_ok(
        "receipt",
        json!({ "action": "get", "receipt_id": receipt_id }),
    );
    let mutations = receipt["receipt"]["mutations"]
        .as_array()
        .unwrap_or_else(|| panic!("no mutations: {receipt}"));
    let sources: Vec<&str> = mutations
        .iter()
        .filter_map(|mutation| mutation["id"].as_str())
        .collect();
    assert_eq!(
        sources,
        vec![
            first["nodeId"].as_str().unwrap(),
            tagged["nodeId"].as_str().unwrap()
        ],
        "one receipt lists both edges: {receipt}"
    );
    assert!(
        mutations
            .iter()
            .all(|mutation| mutation["kind"] == "edge_recorded"),
        "{receipt}"
    );
    server.shutdown();
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

#[test]
fn ghostlink_is_the_advertised_composition_tool() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let listed = server.result("tools/list", None);
    let names: Vec<&str> = listed["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 16, "{names:?}");
    assert!(names.contains(&"ghostlink"), "{names:?}");
    assert!(
        !names.contains(&"graph"),
        "graph must be a hidden alias: {names:?}"
    );
    let ghostlink = listed["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "ghostlink")
        .unwrap();
    assert_eq!(ghostlink["annotations"]["readOnlyHint"], json!(false));
    let modes = &ghostlink["inputSchema"]["properties"]["mode"]["enum"];
    for mode in [
        "propose", "bounty", "weave", "map", "inspect", "explore", "predict", "harden",
    ] {
        assert!(
            modes.as_array().unwrap().iter().any(|m| m == mode),
            "mode {mode} missing: {modes}"
        );
    }
    server.shutdown();
}

#[test]
fn propose_weave_and_inspect_carry_proofs_and_receipts() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let ids: Vec<String> = [
        ("Refunds need multi-row transactions.", "decision"),
        ("The March outage started in the retry loop.", "event"),
        ("Kettle descaling is due every six weeks.", "fact"),
        ("Invoices are rendered by the PDF worker.", "fact"),
    ]
    .iter()
    .map(|(content, kind)| save(&mut server, content, kind))
    .collect();

    // A cold log has no typed edges: the bridge lens is empty and says why.
    let bridge = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "limit": 5 }),
    );
    assert!(pairs(&bridge).is_empty(), "{bridge}");
    assert!(
        bridge["admission"]["emptyBecause"].as_str().is_some(),
        "an empty bridge must explain itself: {bridge}"
    );

    // The divergent lens forces juxtapositions, each with its proof and no
    // claimed score.
    let divergent = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 2 }),
    );
    let first = pairs(&divergent);
    assert!(!first.is_empty(), "{divergent}");
    for candidate in divergent["candidates"].as_array().unwrap() {
        assert_eq!(
            candidate["proof"]["noEdgeVerified"],
            json!(true),
            "{candidate}"
        );
        assert_eq!(candidate["proof"]["neverWoven"], json!(true), "{candidate}");
        assert_eq!(candidate["lane"], json!("juxtaposition"), "{candidate}");
        assert!(
            candidate["score"].is_null(),
            "unmeasured pairs claim no score: {candidate}"
        );
        assert!(
            candidate["compositionQuestion"]
                .as_str()
                .is_some_and(|q| q.contains("Force a composition")),
            "{candidate}"
        );
    }
    let members: Vec<&String> = first.iter().flat_map(|(a, b)| [a, b]).collect();
    let mut unique = members.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        members.len(),
        "each memory at most once per page"
    );
    let again = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 2 }),
    );
    assert_eq!(pairs(&again), first, "same log head, same page");

    // Weave writes a composition record and two derived_from edges, each
    // with a receipt, and the pair leaves both lenses.
    let (a, b) = first[0].clone();
    let woven = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "weave", "first_id": a, "second_id": b,
                "outcome_type": "helpful", "lens": "divergent" }),
    );
    let receipts = woven["receipts"]
        .as_array()
        .unwrap_or_else(|| panic!("{woven}"));
    assert_eq!(receipts.len(), 3, "{woven}");
    for receipt in receipts {
        assert!(
            receipt["receiptId"]
                .as_str()
                .is_some_and(|r| r.starts_with("eff-")),
            "{receipt}"
        );
    }
    let record = woven["recordId"].as_str().unwrap().to_string();
    // Every write the weave reports resolves: the record and both edges.
    for receipt in receipts {
        let got = server.call_tool_ok(
            "receipt",
            json!({ "action": "get", "receipt_id": receipt["receiptId"] }),
        );
        assert!(got.get("error").is_none(), "{receipt}: {got}");
        let proof = server.call_tool_ok(
            "receipt",
            json!({ "action": "replay", "receipt_id": receipt["receiptId"] }),
        );
        assert_eq!(proof["matched"], json!(true), "{receipt}: {proof}");
    }

    let after = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "divergent", "limit": 50 }),
    );
    assert!(
        !pairs(&after)
            .iter()
            .any(|(x, y)| (x == &a && y == &b) || (x == &b && y == &a)),
        "a woven pair must not be proposed again: {after}"
    );
    assert!(
        !pairs(&after)
            .iter()
            .any(|(x, y)| x == &record || y == &record),
        "composition records are bridges, not ideas"
    );

    let recent = server.call_tool_ok("ghostlink", json!({ "mode": "inspect", "view": "recent" }));
    assert_eq!(recent["events"][0]["id"], json!(record), "{recent}");
    let neighbors = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "inspect", "view": "neighbors", "memory_id": a }),
    );
    assert_eq!(
        neighbors["neighbors"][0]["memoryId"],
        json!(b),
        "{neighbors}"
    );
    assert!(ids.contains(&a) && ids.contains(&b));

    // The hidden graph alias still dispatches, to the same engine.
    let alias = server.call_tool_ok("graph", json!({ "action": "never_composed", "limit": 3 }));
    assert_eq!(alias["lens"], json!("bridge"), "{alias}");
    server.shutdown();
}

#[test]
fn harden_seeds_once_then_reports_every_law_present() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let first = server.call_tool_ok("ghostlink", json!({ "mode": "harden" }));
    assert_eq!(first["lawsSource"]["kind"], json!("embedded"), "{first}");
    assert_eq!(first["laws"], json!(6), "{first}");
    assert_eq!(first["seeded"], json!(6), "{first}");
    assert_eq!(first["failed"], json!(0), "{first}");
    for result in first["results"].as_array().unwrap() {
        assert!(
            result["receiptId"]
                .as_str()
                .is_some_and(|r| r.starts_with("eff-")),
            "a seeded law carries its receipt: {result}"
        );
    }
    let second = server.call_tool_ok("ghostlink", json!({ "mode": "harden" }));
    assert_eq!(second["seeded"], json!(0), "{second}");
    assert_eq!(second["already_present"], json!(6), "{second}");
    let laws = server.call_tool_ok("recall", json!({ "handle": "invariant-law" }));
    assert_eq!(
        laws["nodes"].as_array().map(Vec::len),
        Some(6),
        "no duplicate law memories: {laws}"
    );
    server.shutdown();
}

#[test]
fn a_laws_file_wins_and_a_malformed_one_names_itself() {
    let dir = data_dir();
    let home = data_dir();
    std::fs::write(
        dir.path().join("ghostlink-laws.json"),
        json!({ "ghostlink_invariant_laws": [{
            "id": "LAW-TEST", "name": "Test law", "law": "Every retry carries its attempt id.",
            "signals": ["retry"], "severity_if_absent": "High"
        }]})
        .to_string(),
    )
    .unwrap();
    let mut server = spawn(dir.path(), home.path());
    let seeded = server.call_tool_ok("ghostlink", json!({ "mode": "harden" }));
    assert_eq!(seeded["lawsSource"]["kind"], json!("file"), "{seeded}");
    assert_eq!(seeded["laws"], json!(1), "{seeded}");
    assert_eq!(seeded["results"][0]["lawId"], json!("LAW-TEST"), "{seeded}");
    server.shutdown();

    let broken = data_dir();
    std::fs::write(broken.path().join("ghostlink-laws.json"), "{ not json").unwrap();
    let mut server = spawn(broken.path(), home.path());
    let refused = server.call_tool("ghostlink", json!({ "mode": "harden" }));
    let text = refused.to_string();
    assert!(
        text.contains("ghostlink-laws.json") && text.contains("malformed"),
        "a malformed laws file must be named, not skipped: {refused}"
    );
    server.shutdown();
}

#[test]
fn chain_lists_every_memory_origin_first_and_keeps_the_no_chain_contract() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let a = save(&mut server, "The billing export runs nightly.", "fact");
    let b = save(
        &mut server,
        "Ledger rows are immutable once posted.",
        "decision",
    );
    let c = save(&mut server, "The office plant needs light.", "fact");
    let woven = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "weave", "first_id": a, "second_id": b, "outcome_type": "helpful" }),
    );
    let record = woven["recordId"].as_str().unwrap().to_string();

    // a <-derived_from- record -derived_from-> b: steps name every memory on
    // the chain, origin first, and each later step the edge it arrived by.
    let chain = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "explore", "kind": "chain", "from": a, "to": b }),
    );
    let steps = chain["steps"]
        .as_array()
        .unwrap_or_else(|| panic!("{chain}"));
    let ids: Vec<&str> = steps
        .iter()
        .filter_map(|step| step["memory_id"].as_str())
        .collect();
    assert_eq!(
        ids,
        vec![a.as_str(), record.as_str(), b.as_str()],
        "{chain}"
    );
    assert_eq!(steps[0]["connection_type"], json!("origin"), "{chain}");
    assert_eq!(
        steps[1]["connection_type"],
        json!("derived_from"),
        "{chain}"
    );
    assert_eq!(chain["total_hops"], json!(2), "{chain}");
    assert_eq!(chain["path"].as_array().map(Vec::len), Some(2), "{chain}");

    // No recorded path keeps the stable message and says why.
    let none = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "explore", "kind": "chain", "from": a, "to": c }),
    );
    assert_eq!(none["steps"], json!([]), "{none}");
    assert_eq!(
        none["message"],
        json!("No chain found between these memories"),
        "{none}"
    );
    assert!(
        none["reason"]
            .as_str()
            .is_some_and(|r| r.contains("no recorded typed path")),
        "{none}"
    );

    // The legacy fields a chain answer carried stay, read from the
    // recorded edges' own strengths: a woven edge is written at full
    // strength.
    assert_eq!(chain["confidence"], json!(1.0), "{chain}");
    assert_eq!(steps[0]["connection_strength"], json!(1.0), "{chain}");
    assert_eq!(steps[1]["connection_strength"], json!(1.0), "{chain}");
    assert!(
        steps[1]["reasoning"]
            .as_str()
            .is_some_and(|r| r.contains("derived_from")),
        "{chain}"
    );

    // The hidden graph alias answers with the same contract.
    let alias = server.call_tool_ok("graph", json!({ "action": "chain", "from": a, "to": b }));
    assert_eq!(alias["steps"], chain["steps"], "{alias}");

    // bridges stays a list of memory ids; the described bridges ride along.
    let bridges = server.call_tool_ok("graph", json!({ "action": "bridges", "from": a, "to": b }));
    assert_eq!(bridges["bridges"], json!([record]), "{bridges}");
    assert_eq!(
        bridges["bridgeDetails"][0]["memory_id"],
        json!(record),
        "{bridges}"
    );
    assert_eq!(bridges["count"], json!(1), "{bridges}");

    // associations keep a strength per neighbor.
    let assoc = server.call_tool_ok("graph", json!({ "action": "associations", "from": a }));
    let rows = assoc["associations"]
        .as_array()
        .unwrap_or_else(|| panic!("{assoc}"));
    let row = rows
        .iter()
        .find(|row| row["memory_id"] == json!(record))
        .unwrap_or_else(|| panic!("{assoc}"));
    assert_eq!(row["strength"], json!(1.0), "{assoc}");
    assert_eq!(row["link_type"], json!("derived_from"), "{assoc}");
    server.shutdown();
}

/// Links declared at save time are typed edges with receipts, and they are
/// what the bridge lens walks: two memories that each derive from the same
/// memory become a never-composed bridge pair through it.
#[test]
fn links_declared_at_save_time_feed_the_bridge_lens() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let hub = save(&mut server, "The auth timeout is 30 seconds.", "fact");
    let lone = save(&mut server, "The deploy runs at noon.", "fact");

    let before = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "limit": 10 }),
    );
    assert!(
        pairs(&before).is_empty(),
        "no typed edge exists yet: {before}"
    );

    let first = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": "Login retries fail after 30s.", "node_type": "event",
                "links": [{ "kind": "derived_from", "to": hub }] }),
    );
    let first_id = first["nodeId"].as_str().unwrap().to_string();
    let link = &first["links"][0];
    assert_eq!(link["edge"], json!("derived_from"), "{first}");
    assert_eq!(link["source"], json!(first_id), "{first}");
    assert_eq!(link["target"], json!(hub), "{first}");
    let receipt = server.call_tool_ok(
        "receipt",
        json!({ "action": "get", "receipt_id": link["receiptId"] }),
    );
    assert_eq!(
        receipt["attestation"]["verification"]["locallyVerified"],
        json!(true),
        "{receipt}"
    );

    // The batch form carries links per item.
    let batch = server.call_tool_ok(
        "smart_ingest",
        json!({ "items": [{ "content": "Session refresh also waits 30s.", "node_type": "event",
                            "links": [{ "kind": "derived_from", "to": hub }] }] }),
    );
    let second_id = batch["results"][0]["nodeId"].as_str().unwrap().to_string();
    assert!(
        batch["results"][0]["links"][0]["receiptId"]
            .as_str()
            .is_some(),
        "{batch}"
    );

    let after = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "propose", "lens": "bridge", "limit": 10 }),
    );
    let found = pairs(&after);
    assert!(
        found.contains(&(first_id.clone(), second_id.clone()))
            || found.contains(&(second_id.clone(), first_id.clone())),
        "the two memories that derive from {hub} must bridge: {after}"
    );
    assert!(
        !found.iter().any(|(a, b)| a == &lone || b == &lone),
        "a memory with no link must not bridge: {after}"
    );
    server.shutdown();
}

/// `closes` is recorded from the closed memory's side, and bad links are
/// refused before anything is written.
#[test]
fn links_are_checked_before_anything_is_written() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let issue = save(&mut server, "Uploads over 2 GB fail.", "event");

    let fix = server.call_tool_ok(
        "smart_ingest",
        json!({ "content": "Chunked uploads fix the 2 GB failure.", "node_type": "decision",
                "links": [{ "kind": "closes", "to": issue }] }),
    );
    let fix_id = fix["nodeId"].as_str().unwrap();
    assert_eq!(fix["links"][0]["edge"], json!("closed_by"), "{fix}");
    assert_eq!(fix["links"][0]["source"], json!(issue), "{fix}");
    assert_eq!(fix["links"][0]["target"], json!(fix_id), "{fix}");

    for (links, needle) in [
        (
            json!([{ "kind": "touched", "to": issue }]),
            "not declarable",
        ),
        (
            json!([{ "kind": "derived_from", "to": "mem-ffffffffffffffff" }]),
            "not a memory",
        ),
        (
            json!([{ "kind": "derived_from", "to": issue }, { "kind": "derived_from", "to": issue }]),
            "twice",
        ),
        (json!([{ "kind": "derived_from" }]), "needs `to`"),
    ] {
        let refused = server.call_tool(
            "smart_ingest",
            json!({ "content": "This must not be saved.", "tags": ["never-written"], "links": links }),
        );
        assert_eq!(refused["isError"], json!(true), "{refused}");
        assert!(
            refused["text"].as_str().unwrap().contains(needle),
            "{refused}"
        );
    }
    let other_scope = server.call_tool(
        "smart_ingest",
        json!({ "content": "This must not be saved either.", "tags": ["never-written"],
                "scope": "elsewhere", "links": [{ "kind": "derived_from", "to": issue }] }),
    );
    assert!(
        other_scope["text"]
            .as_str()
            .unwrap_or("")
            .contains("links stay within one scope"),
        "{other_scope}"
    );
    let nothing = server.call_tool("recall", json!({ "handle": "never-written" }));
    assert!(
        nothing.get("nodes").is_none() || nothing["nodes"].as_array().unwrap().is_empty(),
        "a refused save wrote a memory: {nothing}"
    );
    server.shutdown();
}

/// A weave can carry external findings for its pair: each is recorded on the
/// composition record, tagged `evidence:<sha256>`, and found again by that tag.
#[test]
fn weave_records_external_evidence_found_again_by_its_hash() {
    let dir = data_dir();
    let home = data_dir();
    let mut server = spawn(dir.path(), home.path());
    let a = save(
        &mut server,
        "The importer drops rows with a null owner.",
        "fact",
    );
    let b = save(
        &mut server,
        "Ownership moved to a join table in v2.",
        "decision",
    );
    let sha = "a".repeat(63) + "b";
    let woven = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "weave", "first_id": a, "second_id": b, "outcome_type": "helpful",
                "evidence": [{ "url": "https://example.com/postmortem", "sha256": sha.to_uppercase(),
                               "retrievedAt": "2026-09-30T12:00:00Z", "note": "matches the v2 schema change" }] }),
    );
    let record = woven["recordId"].as_str().unwrap().to_string();
    assert_eq!(
        woven["evidence"][0]["sha256"],
        json!(sha),
        "hashes are stored lowercase: {woven}"
    );
    assert_eq!(
        woven["evidence"][0]["tag"],
        json!(format!("evidence:{sha}")),
        "{woven}"
    );

    let found = server.call_tool_ok("recall", json!({ "handle": format!("evidence:{sha}") }));
    let text = found.to_string();
    assert!(
        text.contains(&record),
        "recall by the evidence tag finds the record: {found}"
    );
    assert!(text.contains("https://example.com/postmortem"), "{found}");

    for (evidence, needle) in [
        (
            json!([{ "url": "ftp://example.com/x", "sha256": sha, "retrievedAt": "2026-09-30T12:00:00Z" }]),
            "http(s)",
        ),
        (
            json!([{ "url": "https://example.com/x", "sha256": "abc", "retrievedAt": "2026-09-30T12:00:00Z" }]),
            "64 hex",
        ),
        (
            json!([{ "url": "https://example.com/x", "sha256": sha, "retrievedAt": "2999-01-01T00:00:00Z" }]),
            "future",
        ),
        (
            json!([{ "url": "https://example.com/x", "sha256": sha, "retrievedAt": "yesterday" }]),
            "RFC 3339",
        ),
    ] {
        let refused = server.call_tool(
            "ghostlink",
            json!({ "mode": "weave", "first_id": a, "second_id": b, "outcome_type": "dead_end", "evidence": evidence }),
        );
        assert_eq!(refused["isError"], json!(true), "{refused}");
        assert!(
            refused["text"].as_str().unwrap().contains(needle),
            "{refused}"
        );
    }
    let records = server.call_tool_ok(
        "ghostlink",
        json!({ "mode": "inspect", "view": "memory", "memory_id": a }),
    );
    assert!(
        !records.to_string().contains("dead_end"),
        "a refused weave recorded an outcome: {records}"
    );
    server.shutdown();
}
