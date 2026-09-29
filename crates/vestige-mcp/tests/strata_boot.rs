//! build/wire-strata acceptance: a legacy-sqlite-free (wiped) build boots
//! the MCP server on the STRATA Causal Proof Engine behind vestige-core's
//! storage trait — ingest / get / edge round-trips run over the trait, and
//! the real JSON-RPC handler answers on top of the STRATA-backed store.
//!
//! This target intentionally has NO `required-features`: it must run in the
//! default (wiped, no-legacy-sqlite) build, which is the mode whose runtime
//! store is STRATA. Feature unification still compiles vestige-core's
//! SQLite seam inside the workspace, so the test constructs the STRATA
//! backend explicitly (the same constructor `main.rs` installs in wiped
//! builds) instead of routing through `open_storage`.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::Mutex;
use vestige_core::storage::Storage;
use vestige_core::{ConnectionRecord, IngestInput};
use vestige_mcp::cognitive::CognitiveEngine;
use vestige_mcp::protocol::types::JsonRpcRequest;
use vestige_mcp::server::McpServer;

fn make_request(method: &str, params: Option<serde_json::Value>) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: method.to_string(),
        params,
    }
}

/// Boot the server handler over a STRATA-backed store; ingest one memory,
/// read it back, save one edge, read the edge — all through the trait.
#[tokio::test]
async fn wiped_mode_server_boots_on_strata_and_round_trips_over_the_trait() {
    let dir = tempfile::tempdir().unwrap();
    let strata_dir = dir.path().join("strata");

    // The exact backend constructor wiped-mode main.rs installs.
    let storage: Arc<Storage> =
        strata_bridge::StrataBackend::open(&strata_dir).expect("STRATA store opens");
    assert_eq!(
        storage.data_dir(),
        strata_dir.as_path(),
        "data_dir is the strata dir"
    );

    // Boot the real handler over it. McpServer::new exercises the
    // boot-critical seam: data_dir, set_process_actor (warn-only path),
    // sidecar/output-config load.
    let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
    let server = McpServer::new(storage.clone(), cognitive);

    // The handler answers a full handshake + tool catalog on the
    // STRATA-backed server.
    let init = server
        .handle_request(make_request(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "strata-boot-test", "version": "0.0.0" }
            })),
        ))
        .await
        .expect("initialize returns a response");
    assert!(
        init.result.is_some(),
        "initialize succeeded: {:?}",
        init.error
    );
    let tools = server
        .handle_request(make_request("tools/list", None))
        .await
        .expect("tools/list returns a response");
    assert!(
        tools.result.is_some(),
        "tools/list succeeded: {:?}",
        tools.error
    );
    assert!(
        tools.result.as_ref().unwrap()["tools"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "tool catalog is populated"
    );

    // Ingest one memory over the trait.
    let node = storage
        .ingest(IngestInput {
            content: "STRATA is the runtime store behind the trait wall".to_string(),
            node_type: "fact".to_string(),
            ..IngestInput::default()
        })
        .expect("ingest over the trait");
    assert!(
        node.id.starts_with("mem-"),
        "log-derived strata id: {}",
        node.id
    );

    // Get it back over the trait.
    let fetched = storage
        .get_node(&node.id)
        .expect("get_node over the trait")
        .expect("ingested node reads back");
    assert_eq!(
        fetched.content,
        "STRATA is the runtime store behind the trait wall"
    );
    assert_eq!(fetched.node_type, "fact");
    assert!(
        fetched.reps >= 1,
        "ingest folded one FSRS review (rating Good)"
    );

    // Save one edge over the trait, then read it back.
    let target = storage
        .ingest(IngestInput {
            content: "the bridge maps the typed-edge vocabulary".to_string(),
            ..IngestInput::default()
        })
        .expect("second ingest");
    storage
        .save_connection(&ConnectionRecord {
            source_id: node.id.clone(),
            target_id: target.id.clone(),
            strength: 0.75,
            link_type: "derived_from".to_string(),
            created_at: chrono::Utc::now(),
            last_activated: chrono::Utc::now(),
            activation_count: 0,
        })
        .expect("save_connection over the trait");
    let edges = storage
        .get_connections_for_memory(&node.id)
        .expect("edge read over the trait");
    assert_eq!(edges.len(), 1, "exactly one edge");
    assert_eq!(edges[0].target_id, target.id);
    assert_eq!(edges[0].link_type, "derived_from");
    assert!(
        (edges[0].strength - 0.75).abs() < 1e-9,
        "milli-round-trip strength"
    );

    // Once linked, the pair leaves the never-composed set (strata derived
    // index behind the graph tool's fusion candidates).
    let candidates = storage
        .get_never_composed_candidates_in_scope(50, None, None)
        .expect("never-composed query over the trait");
    assert!(
        candidates
            .iter()
            .all(|c| !(c.first_id == node.id && c.second_id == target.id)),
        "linked pair is not never-composed"
    );

    // The gate holds destructive writes loudly: RETIRE (supersession) is
    // held by the default policy — the loud-fail contract, not a stub.
    // (Exercised via the store's own API through the trait's pair read:
    // nothing is superseded here.)
    assert!(
        storage
            .supersession_pairs()
            .expect("supersession pairs")
            .is_empty()
    );

    // The store's own vocabulary gate still holds underneath the bridge:
    // legacy free-form link types are rejected.
    let bad_edge = ConnectionRecord {
        source_id: node.id.clone(),
        target_id: target.id.clone(),
        strength: 1.0,
        link_type: "semantic".to_string(),
        created_at: chrono::Utc::now(),
        last_activated: chrono::Utc::now(),
        activation_count: 0,
    };
    assert!(
        storage.save_connection(&bad_edge).is_err(),
        "free-form link_type rejected"
    );

    // A method the strata v1 surface lacks keeps the trait's loud-fail
    // default (never a silent success).
    let loud = storage.search("strata", 10).expect_err("search is loud");
    assert!(
        loud.to_string().contains("not implemented by this backend"),
        "loud default failure, got: {loud}"
    );

    // Open failures are loud too: a file where the store dir should be
    // errors out instead of panicking or silently succeeding.
    let blocker = tempfile::tempdir().unwrap();
    let blocked = blocker.path().join("strata");
    std::fs::write(&blocked, b"not a directory").unwrap();
    assert!(strata_bridge::StrataBackend::open(&blocked).is_err());

    // Persistence: drop every handle (the store thread exits with the last
    // sender, releasing the log's directory lock), reopen through the same
    // constructor, and read the memory back from the replayed log.
    drop(server);
    drop(storage);
    let mut reopened: Option<Arc<Storage>> = None;
    for _ in 0..100 {
        if let Ok(backend) = strata_bridge::StrataBackend::open(&strata_dir) {
            reopened = Some(backend);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let reopened = reopened.expect("STRATA store reopens after the lock releases");
    let reread = reopened
        .get_node(&node.id)
        .expect("get_node after reopen")
        .expect("the memory survived close/reopen via log replay");
    assert_eq!(
        reread.content,
        "STRATA is the runtime store behind the trait wall"
    );
    assert_eq!(
        reopened
            .get_connections_for_memory(&node.id)
            .expect("edges after reopen")
            .len(),
        1,
        "the edge survived close/reopen"
    );
}

/// The constructor hook wiring: the fn pointer installs into vestige-core's
/// `STORE_CONSTRUCTOR` (unconditional in every build), and the path
/// resolution maps SQLite-shaped DB paths to `<data dir>/strata`.
#[test]
fn constructor_hook_installs_and_resolves_directories() {
    vestige_core::storage::install_store_constructor(strata_bridge::strata_constructor);
    assert!(
        vestige_core::storage::STORE_CONSTRUCTOR
            .lock()
            .map(|slot| slot.is_some())
            .unwrap_or(false),
        "the strata constructor is installed in the hook"
    );

    // Path mapping (skipped when $VESTIGE_STRATA_DIR redirects the world).
    if std::env::var_os("VESTIGE_STRATA_DIR").is_none() {
        let db_shaped = std::path::Path::new("/tmp/vestige-data/vestige.db");
        assert_eq!(
            strata_bridge::strata_dir_for(db_shaped),
            std::path::Path::new("/tmp/vestige-data/strata")
        );
        let dir_shaped = std::path::Path::new("/tmp/vestige-data");
        assert_eq!(
            strata_bridge::strata_dir_for(dir_shaped),
            std::path::Path::new("/tmp/vestige-data/strata")
        );
    }
}
