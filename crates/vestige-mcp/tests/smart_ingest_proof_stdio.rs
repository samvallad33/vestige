//! Real-stdio proof for the INGEST-V5 write path (feat/ingest-proof, spec
//! B5). One sequential story against ONE default-featured Strata store:
//!
//! 1. a forceCreate create carries `receiptId`/`canonicalHash`/`entities`/
//!    `importance`;
//! 2. resubmitting the same bytes reinforces via an echo node, and the echo
//!    never becomes the dedup target;
//! 3. canonically identical but byte-different content (zero-width, case,
//!    whitespace; and e+combining vs precomposed NFC) still reinforces the
//!    same original;
//! 4. an `intent_id` replays with ZERO writes (node count proved unchanged);
//! 5. a replay with different bytes still replays, and the divergence is
//!    visible via `requestCanonicalHash`;
//! 6. a `supersedes` declared link persists B -supersedes-> A;
//! 7. a batch of four (an identical pair, a new item, a duplicate of A)
//!    counts created/reinforced correctly, returns caller order, and a
//!    second run turns the intent-keyed creates into replays.
//!
//! The child is `vestige-mcp` with default features, spawned exactly the way
//! `strata_stdio_receipt_project.rs` spawns it (same flags, same pinned
//! environment). No SQLite file is created; after shutdown the log is
//! checkpoint-sealed and strata-verified, and the edges are inspected
//! directly.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct StdioServer {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    next_id: u64,
    transcript: Vec<Value>,
}

impl StdioServer {
    fn spawn(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .arg("--no-http")
            .arg("--data-dir")
            .arg(data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout");
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
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
        Self {
            child,
            stdin,
            lines,
            next_id: 0,
            transcript: Vec::new(),
        }
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&request).unwrap();
        line.push('\n');
        self.stdin
            .as_mut()
            .expect("stdin open")
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.as_mut().unwrap().flush())
            .expect("write stdin");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!("timed out waiting for {method}");
            }
            let raw = self
                .lines
                .recv_timeout(remaining)
                .unwrap_or_else(|error| panic!("stdio closed during {method}: {error}"));
            let response: Value = serde_json::from_str(&raw).expect("json-rpc line");
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            self.transcript.push(json!({
                "request": request,
                "response": response,
            }));
            assert!(
                response.get("error").is_none(),
                "rpc error for {method}: {response}"
            );
            return response["result"].clone();
        }
    }

    fn tool(&mut self, name: &str, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({ "name": name, "arguments": args }));
        assert_eq!(
            result.get("isError").and_then(Value::as_bool),
            Some(false),
            "{name} failed: {result}"
        );
        let body = result
            .get("structuredContent")
            .cloned()
            .expect("structuredContent");
        let text = serde_json::to_string(&body).unwrap();
        assert!(
            !text.contains("pending_strata"),
            "{name} returned pending_strata: {body}"
        );
        body
    }

    fn handshake(&mut self) {
        self.rpc(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "smart-ingest-proof-stdio", "version": "1"},
            }),
        );
        let note = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
        self.stdin
            .as_mut()
            .unwrap()
            .write_all(note.as_bytes())
            .and_then(|()| self.stdin.as_mut().unwrap().flush())
            .unwrap();
    }

    fn shutdown(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for StdioServer {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn canonical(value: &Value) -> String {
    serde_json::to_string(value).unwrap()
}

fn assert_same(label: &str, left: &Value, right: &Value) {
    assert_eq!(
        canonical(left),
        canonical(right),
        "{label} is not deterministic"
    );
}

/// The live node count, read through the store's own status surface so the
/// "a replay writes nothing" proof never has to open the log underneath the
/// running server.
fn total_memories(server: &mut StdioServer) -> u64 {
    let status = server.tool("memory_status", json!({ "view": "health" }));
    status["totalMemories"]
        .as_u64()
        .unwrap_or_else(|| panic!("memory_status health returned no totalMemories: {status}"))
}

/// Find the span with the given surface whose kind names `needle`
/// (case-insensitive): the extractor's exact kind strings are Lane C's, the
/// B5 contract only pins FilePath and CommitSha semantics plus surfaces and
/// byte offsets.
fn span_with<'a>(entities: &'a [Value], needle: &str, surface: &str) -> &'a Value {
    entities
        .iter()
        .find(|span| {
            span["surface"] == json!(surface)
                && span["kind"]
                    .as_str()
                    .is_some_and(|kind| kind.to_ascii_lowercase().contains(needle))
        })
        .unwrap_or_else(|| panic!("no {needle} span for {surface:?}: {entities:?}"))
}

#[test]
fn proof_carrying_write_path_over_real_stdio() {
    let data = tempfile::tempdir().expect("data dir");
    let mut server = StdioServer::spawn(data.path());
    server.handshake();

    // --- 1. forceCreate create carries the proof fields (B5.1) ---
    let alpha = "alpha content with src/store.py and commit a1b2c3d";
    let created = server.tool(
        "smart_ingest",
        json!({ "content": alpha, "forceCreate": true }),
    );
    assert_eq!(created["success"], json!(true), "{created}");
    assert_eq!(created["decision"], "create");
    let a_node = created["nodeId"].as_str().expect("nodeId").to_string();
    let a_receipt = created["receiptId"].as_str().expect("receiptId").to_string();
    assert!(a_receipt.starts_with("eff-"), "{a_receipt}");
    let a_hash = created["canonicalHash"]
        .as_str()
        .expect("canonicalHash")
        .to_string();
    assert_eq!(a_hash.len(), 64, "{a_hash}");
    assert!(
        a_hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{a_hash} is not hex"
    );
    let entities = created["entities"].as_array().expect("entities");
    assert!(!entities.is_empty(), "{entities:?}");
    assert!(entities.len() <= 32, "{entities:?}");
    let file_span = span_with(entities, "file", "src/store.py");
    let file_start = alpha.find("src/store.py").expect("byte offset");
    assert_eq!(file_span["byteStart"], json!(file_start));
    assert_eq!(
        file_span["byteEnd"],
        json!(file_start + "src/store.py".len())
    );
    let commit_span = span_with(entities, "commit", "a1b2c3d");
    let commit_start = alpha.find("a1b2c3d").expect("byte offset");
    assert_eq!(commit_span["byteStart"], json!(commit_start));
    assert_eq!(
        commit_span["byteEnd"],
        json!(commit_start + "a1b2c3d".len())
    );
    let score = created["importance"]["score"].as_f64().expect("score");
    assert!(score > 0.0 && score <= 1.0, "{score}");
    assert!(
        created["importance"]["weightsVersion"].is_string(),
        "{created}"
    );

    // --- 2. exact resubmission reinforces; the echo never becomes the
    //        dedup target (B5.2) ---
    let reinforced = server.tool("smart_ingest", json!({ "content": alpha }));
    assert_eq!(reinforced["decision"], "reinforce", "{reinforced}");
    assert_eq!(reinforced["duplicateOf"], json!(a_node));
    assert_eq!(reinforced["canonicalHash"], json!(a_hash));
    assert_eq!(reinforced["pipeline"], "nfc-lower-zwstrip-wscollapse-v1");
    assert_eq!(reinforced["rawBytes"], json!(alpha.len()));
    let echo = reinforced["echoNodeId"].as_str().expect("echoNodeId").to_string();
    assert_ne!(echo, a_node);
    assert!(
        reinforced["receiptId"]
            .as_str()
            .is_some_and(|receipt| receipt.starts_with("eff-")),
        "{reinforced}"
    );

    let reinforced_again = server.tool("smart_ingest", json!({ "content": alpha }));
    assert_eq!(reinforced_again["decision"], "reinforce");
    assert_eq!(
        reinforced_again["duplicateOf"],
        json!(a_node),
        "the echo node must never become the dedup target: {reinforced_again}"
    );

    // --- 3. canonically identical but byte-different bytes (B5.3) ---
    // Zero-width characters, different case, extra whitespace: the pipeline
    // folds all of it onto A. (A itself is ASCII-only — the pinned B5.1
    // fixture — so the NFC half of the property is exercised against its own
    // original below.)
    let variant = "\u{200b}ALPHA   content with \u{200b}src/store.py and commit a1b2c3d   ";
    assert_ne!(variant.as_bytes(), alpha.as_bytes());
    let folded = server.tool("smart_ingest", json!({ "content": variant }));
    assert_eq!(folded["decision"], "reinforce", "{folded}");
    assert_eq!(folded["duplicateOf"], json!(a_node));
    assert_eq!(folded["canonicalHash"], json!(a_hash));

    // NFC: e + combining acute vs precomposed, different case, extra
    // whitespace — canonically identical, reinforces the precomposed
    // original.
    let precomposed = "the café canon fixture node";
    let cafe = server.tool("smart_ingest", json!({ "content": precomposed }));
    assert_eq!(cafe["decision"], "create", "{cafe}");
    let cafe_node = cafe["nodeId"].as_str().expect("nodeId").to_string();
    let decomposed = "THE CAFE\u{301} CANON FIXTURE   NODE";
    let folded_nfc = server.tool("smart_ingest", json!({ "content": decomposed }));
    assert_eq!(folded_nfc["decision"], "reinforce", "{folded_nfc}");
    assert_eq!(folded_nfc["duplicateOf"], json!(cafe_node));
    assert_eq!(
        folded_nfc["canonicalHash"],
        cafe["canonicalHash"],
        "NFC-folded variant must hash identically to the precomposed original"
    );

    // --- 4. intent create + replay writes NOTHING (B5.4) ---
    let before = total_memories(&mut server);
    let intent_create = server.tool(
        "smart_ingest",
        json!({ "content": "intent-guarded memory for run-42", "intent_id": "run-42" }),
    );
    assert_eq!(intent_create["decision"], "create", "{intent_create}");
    assert_eq!(intent_create["intentId"], json!("run-42"));
    assert!(
        intent_create["receiptId"]
            .as_str()
            .is_some_and(|receipt| receipt.starts_with("eff-")),
        "{intent_create}"
    );
    let run42_node = intent_create["nodeId"].as_str().expect("nodeId").to_string();
    let after_create = total_memories(&mut server);
    assert_eq!(after_create, before + 1);

    let replay = server.tool(
        "smart_ingest",
        json!({ "content": "intent-guarded memory for run-42", "intent_id": "run-42" }),
    );
    assert_eq!(replay["decision"], "replay", "{replay}");
    assert_eq!(replay["replayed"], json!(true));
    assert_eq!(replay["replayOf"], json!(run42_node));
    assert_eq!(replay["intentId"], json!("run-42"));
    assert!(replay["effectSeq"].as_u64().is_some(), "{replay}");
    assert!(
        replay["intentDigest"]
            .as_str()
            .is_some_and(|digest| digest.len() == 64),
        "{replay}"
    );
    assert_eq!(
        replay["requestCanonicalHash"],
        intent_create["canonicalHash"],
        "same bytes: the request hash must equal the create's canonicalHash"
    );
    assert_eq!(
        total_memories(&mut server),
        after_create,
        "a replay must not write a node"
    );
    let replay_again = server.tool(
        "smart_ingest",
        json!({ "content": "intent-guarded memory for run-42", "intent_id": "run-42" }),
    );
    assert_same("intent replay", &replay, &replay_again);
    assert_eq!(total_memories(&mut server), after_create);

    // --- 5. a divergent replay still replays, visibly (B5.5) ---
    let divergent = server.tool(
        "smart_ingest",
        json!({ "content": "COMPLETELY different bytes under run-42", "intent_id": "run-42" }),
    );
    assert_eq!(divergent["decision"], "replay", "{divergent}");
    assert_eq!(divergent["replayOf"], json!(run42_node));
    assert_eq!(divergent["intentDigest"], replay["intentDigest"]);
    assert_ne!(
        divergent["requestCanonicalHash"],
        intent_create["canonicalHash"],
        "the divergence must be visible in requestCanonicalHash"
    );
    assert_eq!(
        total_memories(&mut server),
        after_create,
        "a divergent replay still writes nothing"
    );

    // --- 6. supersedes declared link, direction B -supersedes-> A (B5.6) ---
    let superseder = server.tool(
        "smart_ingest",
        json!({
            "content": "full replacement note standing in for the alpha memory",
            "links": [{ "kind": "supersedes", "to": a_node }],
        }),
    );
    assert_eq!(superseder["decision"], "create", "{superseder}");
    let b_node = superseder["nodeId"].as_str().expect("nodeId").to_string();
    let written_links = superseder["links"].as_array().expect("links written");
    assert_eq!(written_links.len(), 1, "{written_links:?}");
    assert_eq!(written_links[0]["kind"], "supersedes");
    assert_eq!(written_links[0]["source"], json!(b_node));
    assert_eq!(written_links[0]["target"], json!(a_node));
    assert!(
        written_links[0]["receiptId"]
            .as_str()
            .is_some_and(|receipt| receipt.starts_with("eff-")),
        "{written_links:?}"
    );

    // --- 7. batch: identical pair + new + duplicate of A (B5.7) ---
    let batch_args = json!({
        "forceCreate": false,
        "items": [
            { "content": "batch pair one bravo", "intent_id": "batch-pair-1" },
            { "content": "batch new charlie", "intent_id": "batch-new-1" },
            { "content": "BATCH   PAIR ONE  BRAVO" },
            { "content": alpha },
        ]
    });
    let batch = server.tool("smart_ingest", batch_args.clone());
    assert_eq!(batch["summary"]["created"], json!(2), "{batch}");
    assert_eq!(batch["summary"]["reinforced"], json!(2));
    assert_eq!(batch["summary"]["replayed"], json!(0));
    assert_eq!(batch["summary"]["errors"], json!(0));
    assert_eq!(batch["summary"]["total"], json!(4));
    let results = batch["results"].as_array().expect("results");
    assert_eq!(results.len(), 4);
    for (position, expected) in [(0usize, 0u64), (1, 1), (2, 2), (3, 3)] {
        assert_eq!(
            results[position]["index"],
            json!(expected),
            "results must come back in caller order: {batch}"
        );
    }
    assert_eq!(results[0]["decision"], "create");
    let pair_node = results[0]["nodeId"].as_str().expect("nodeId").to_string();
    assert_eq!(results[1]["decision"], "create");
    assert_eq!(results[2]["decision"], "reinforce");
    assert_eq!(
        results[2]["duplicateOf"],
        json!(pair_node),
        "intra-batch duplicates resolve to the first identical item"
    );
    assert_eq!(results[3]["decision"], "reinforce");
    assert_eq!(results[3]["duplicateOf"], json!(a_node));

    // Order-independence: the same batch again replays the intent-keyed
    // creates and reinforces the rest.
    let replayed_batch = server.tool("smart_ingest", batch_args);
    assert_eq!(replayed_batch["summary"]["created"], json!(0), "{replayed_batch}");
    assert_eq!(replayed_batch["summary"]["replayed"], json!(2));
    assert_eq!(replayed_batch["summary"]["reinforced"], json!(2));
    let replayed_results = replayed_batch["results"].as_array().expect("results");
    assert_eq!(replayed_results[0]["decision"], "replay");
    assert_eq!(replayed_results[0]["replayOf"], json!(pair_node));
    assert_eq!(replayed_results[1]["decision"], "replay");
    assert_eq!(replayed_results[2]["decision"], "reinforce");
    assert_eq!(replayed_results[3]["decision"], "reinforce");

    if let Ok(path) = std::env::var("VESTIGE_STDIO_TRANSCRIPT") {
        let body = serde_json::to_string_pretty(&server.transcript).unwrap();
        std::fs::write(path, body).unwrap();
    }
    server.shutdown();
    drop(server);

    // --- on-disk proofs: seal, verify, inspect edges and echo records ---
    {
        let mut store = strata_store::StrataStore::open(data.path()).unwrap();
        store.seal_checkpoint().unwrap();
    }
    let report = strata_verify::verify_path(data.path());
    assert!(report.ok, "strata-verify failed: {}", report.json);

    let store = strata_store::StrataStore::open(data.path()).unwrap();
    let echo_record = store
        .get_node(&echo)
        .unwrap_or_else(|| panic!("echo node {echo} must exist"));
    assert!(
        echo_record
            .content
            .starts_with(&format!("duplicate of {a_node}\n")),
        "{}",
        echo_record.content
    );
    assert!(
        echo_record.tags.iter().any(|tag| tag == "duplicate"),
        "{:?}",
        echo_record.tags
    );
    let edges = store.edges();
    assert!(
        edges.iter().any(|edge| edge.link_type == "evidence_of"
            && edge.source_id == echo
            && edge.target_id == a_node),
        "no evidence_of edge {echo} -> {a_node}: {edges:?}"
    );
    assert!(
        edges
            .iter()
            .any(|edge| edge.link_type == "supersedes"
                && edge.source_id == b_node
                && edge.target_id == a_node),
        "no supersedes edge {b_node} -> {a_node}: {edges:?}"
    );
}
