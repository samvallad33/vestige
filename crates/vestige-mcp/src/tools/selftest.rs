//! `selftest` MCP tool.
//!
//! On a Strata log this plants a synthetic chain in a temp store, walks
//! recorded `derived_from` edges backward from the symptom, and reports
//! pass/fail per check. The user store is only read. The legacy SQLite path
//! below still scores planted causes through backfill.

use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::{IngestInput, Storage};

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Plant a synthetic causal chain in a temp store and score a bounded backward walk over recorded edges. Does not write the user store.",
        "properties": {}
    })
}

/// One scored round: cause planted, failure referencing exactly one anchor.
struct RoundOutcome {
    round: u32,
    cause_id: String,
    hit_at_1: bool,
    hit_at_3: bool,
    rank: Option<usize>,
}

/// Plant a quiet, non-failure-looking cause carrying exactly two anchors: the
/// env-shaped `PLANTED_CAUSE_<k>` tag and a file path in the content.
fn plant_cause(storage: &Arc<Storage>, scope: &str, k: u32) -> Result<String, String> {
    let cause = storage
        .ingest_in_scope(
            IngestInput {
                content: format!("Tuned PLANTED_KNOB_{k} inside src/selftest/planted_{k}.rs"),
                node_type: "decision".to_string(),
                tags: vec![format!("PLANTED_CAUSE_{k}")],
                ..Default::default()
            },
            scope,
        )
        .map_err(|e| format!("round {k}: planting cause failed: {e}"))?;
    storage
        .set_created_at(&cause.id, chrono::Utc::now() - chrono::Duration::days(5))
        .map_err(|e| format!("round {k}: backdating cause failed: {e}"))?;
    Ok(cause.id)
}

/// Ingest a synthetic failure that shares EXACTLY the `PLANTED_CAUSE_<k>`
/// anchor with round k's cause (share_anchor=true) or none at all (round 6).
fn plant_failure(
    storage: &Arc<Storage>,
    scope: &str,
    k: u32,
    share_anchor: bool,
) -> Result<String, String> {
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: if share_anchor {
                    format!("Outage on shard {k}: crash followed PLANTED_CAUSE_{k} change")
                } else {
                    "Outage with no recorded trigger: crash after PLANTED_CAUSE_6 flip".to_string()
                },
                node_type: "event".to_string(),
                tags: vec![format!("PLANTED_CAUSE_{k}")],
                ..Default::default()
            },
            scope,
        )
        .map_err(|e| format!("round {k}: planting failure failed: {e}"))?;
    Ok(failure.id)
}

/// Run the REAL backfill tool (manual=true, preview) against the copy.
async fn run_backfill(
    storage: &Arc<Storage>,
    scope: &str,
    failure_id: &str,
) -> Result<Value, String> {
    super::backfill::execute(
        storage,
        Some(json!({
            "failure_id": failure_id,
            "scope": scope,
            "manual": true,
            "promote": false,
            "lookback_days": 30,
            "scan_limit": 100,
        })),
    )
    .await
}

/// Recorded cause→effect edge. Backward walks follow `target → source`.
const CAUSAL_LINK: &str = "derived_from";
/// Hop cap. The planted chain is ancestor → cause → intermediate → symptom;
/// a bound of 2 stops on the cause and leaves the ancestor out.
const WALK_BOUND: usize = 2;

struct Planted {
    ancestor: String,
    cause: String,
    intermediate: String,
    symptom: String,
    noise_a: String,
    noise_b: String,
    noise_c: String,
}

fn plant_node(storage: &Storage, content: &str) -> Result<String, String> {
    let node = storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            },
            "selftest",
        )
        .map_err(|e| format!("planting {content} failed: {e}"))?;
    Ok(node.id)
}

fn record_edge(
    storage: &Storage,
    source: &str,
    target: &str,
    link_type: &str,
) -> Result<(), String> {
    let at = chrono::DateTime::<chrono::Utc>::UNIX_EPOCH;
    storage
        .save_connection(&vestige_core::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: at,
            last_activated: at,
            activation_count: 0,
        })
        .map_err(|e| format!("recording {link_type} {source}->{target} failed: {e}"))
}

fn edge_recorded(
    storage: &Storage,
    source: &str,
    target: &str,
    link_type: &str,
) -> Result<bool, String> {
    let edges = storage
        .get_connections_for_memory(target)
        .map_err(|e| format!("reading recorded edges for {target}: {e}"))?;
    Ok(edges.iter().any(|edge| {
        edge.source_id == source && edge.target_id == target && edge.link_type == link_type
    }))
}

/// Cause → intermediate → symptom, plus an ancestor past the hop bound and
/// distractors that share no recorded causal incoming edge with the symptom.
fn plant_chain(storage: &Storage) -> Result<Planted, String> {
    let ancestor = plant_node(storage, "selftest ancestor")?;
    let cause = plant_node(storage, "selftest cause")?;
    let intermediate = plant_node(storage, "selftest intermediate")?;
    let symptom = plant_node(storage, "selftest symptom")?;
    let noise_a = plant_node(storage, "selftest noise a")?;
    let noise_b = plant_node(storage, "selftest noise b")?;
    let noise_c = plant_node(storage, "selftest noise c")?;
    record_edge(storage, &ancestor, &cause, CAUSAL_LINK)?;
    record_edge(storage, &cause, &intermediate, CAUSAL_LINK)?;
    record_edge(storage, &intermediate, &symptom, CAUSAL_LINK)?;
    // Forward edge and a non-causal incoming edge must not enter the walk.
    record_edge(storage, &symptom, &noise_a, CAUSAL_LINK)?;
    record_edge(storage, &noise_b, &symptom, "evidence_of")?;
    record_edge(storage, &noise_b, &noise_c, CAUSAL_LINK)?;
    Ok(Planted {
        ancestor,
        cause,
        intermediate,
        symptom,
        noise_a,
        noise_b,
        noise_c,
    })
}

/// Backward BFS over recorded `derived_from` edges. Ids within a hop are sorted.
fn backward_walk(storage: &Storage, start: &str, bound: usize) -> Result<Vec<String>, String> {
    let mut seen = std::collections::BTreeSet::from([start.to_string()]);
    let mut frontier = vec![start.to_string()];
    let mut path = Vec::new();
    for _hop in 0..bound {
        let mut next = Vec::new();
        frontier.sort();
        for node in &frontier {
            let edges = storage
                .get_connections_for_memory(node)
                .map_err(|e| format!("reading recorded edges for {node}: {e}"))?;
            let mut preds: Vec<String> = edges
                .into_iter()
                .filter(|edge| edge.link_type == CAUSAL_LINK && edge.target_id == *node)
                .map(|edge| edge.source_id)
                .filter(|id| seen.insert(id.clone()))
                .collect();
            preds.sort();
            preds.dedup();
            next.extend(preds);
        }
        if next.is_empty() {
            break;
        }
        path.extend(next.iter().cloned());
        frontier = next;
    }
    Ok(path)
}

fn snapshot(storage: &Storage) -> Result<(Vec<String>, Vec<String>), String> {
    let mut nodes = Vec::new();
    let mut offset = 0i32;
    loop {
        let page = storage
            .get_all_nodes(500, offset)
            .map_err(|e| format!("reading the user store failed: {e}"))?;
        if page.is_empty() {
            break;
        }
        let page_len = page.len();
        offset += i32::try_from(page_len).unwrap_or(i32::MAX);
        nodes.extend(
            page.into_iter()
                .map(|node| format!("{}\t{}", node.id, node.content)),
        );
        if page_len < 500 {
            break;
        }
    }
    nodes.sort();
    let mut edges = storage
        .get_all_connections()
        .map_err(|e| format!("reading user-store edges failed: {e}"))?
        .into_iter()
        .map(|edge| format!("{}\t{}\t{}", edge.source_id, edge.target_id, edge.link_type))
        .collect::<Vec<_>>();
    edges.sort();
    Ok((nodes, edges))
}

fn check(name: &str, pass: bool) -> Value {
    json!({"name": name, "pass": pass})
}

fn strata_selftest(user: &Storage) -> Result<Value, String> {
    let before = snapshot(user)?;
    let dir = tempfile::TempDir::new().map_err(|e| format!("tempdir failed: {e}"))?;
    let (walk, mut checks) = {
        let temp = crate::strata_memory::open(dir.path())
            .map_err(|e| format!("opening the selftest temp store failed: {e}"))?;
        let planted = plant_chain(temp.as_ref())?;
        let walk = backward_walk(temp.as_ref(), &planted.symptom, WALK_BOUND)?;
        let chain_ids = [&planted.cause, &planted.intermediate, &planted.symptom];
        let mut chain_recorded = chain_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == 3;
        for id in chain_ids {
            if temp
                .get_node(id)
                .map_err(|e| format!("reading planted {id}: {e}"))?
                .is_none()
            {
                chain_recorded = false;
            }
        }
        let causal_edges_recorded = edge_recorded(
            temp.as_ref(),
            &planted.ancestor,
            &planted.cause,
            CAUSAL_LINK,
        )? && edge_recorded(
            temp.as_ref(),
            &planted.cause,
            &planted.intermediate,
            CAUSAL_LINK,
        )? && edge_recorded(
            temp.as_ref(),
            &planted.intermediate,
            &planted.symptom,
            CAUSAL_LINK,
        )?;
        let distractor_edges_recorded = edge_recorded(
            temp.as_ref(),
            &planted.symptom,
            &planted.noise_a,
            CAUSAL_LINK,
        )? && edge_recorded(
            temp.as_ref(),
            &planted.noise_b,
            &planted.symptom,
            "evidence_of",
        )? && edge_recorded(
            temp.as_ref(),
            &planted.noise_b,
            &planted.noise_c,
            CAUSAL_LINK,
        )?;
        let walk_order =
            walk.len() == 2 && walk[0] == planted.intermediate && walk[1] == planted.cause;
        let distractors = [&planted.noise_a, &planted.noise_b, &planted.noise_c];
        let distractors_excluded = distractors
            .iter()
            .all(|id| !walk.iter().any(|step| step == *id));
        let bound_held = walk.len() == WALK_BOUND
            && !walk.iter().any(|id| id == &planted.ancestor)
            && edge_recorded(
                temp.as_ref(),
                &planted.ancestor,
                &planted.cause,
                CAUSAL_LINK,
            )?;
        let checks = vec![
            check("chain_recorded", chain_recorded),
            check("causal_edges_recorded", causal_edges_recorded),
            check("distractor_edges_recorded", distractor_edges_recorded),
            check("walk_order", walk_order),
            check("distractors_excluded", distractors_excluded),
            check("bound_held", bound_held),
        ];
        (walk, checks)
    };
    // Drop the temp store before the user-store comparison.
    let temp_store_deleted = dir.close().is_ok();
    let user_store_unchanged = snapshot(user)? == before;
    checks.push(check("user_store_unchanged", user_store_unchanged));
    checks.push(check("temp_store_deleted", temp_store_deleted));
    let passed = checks.iter().all(|item| item["pass"] == json!(true));
    Ok(json!({
        "tool": "selftest",
        "kind": "recorded_edge_walk",
        "passed": passed,
        "deterministic": true,
        "live_store_touched": !user_store_unchanged,
        "temp_store_deleted": temp_store_deleted,
        "bound": WALK_BOUND,
        "walk": walk,
        "checks": checks,
        "note": "Bounded backward walk over recorded derived_from edges in a temp log. The user store is not written.",
    }))
}

pub async fn execute(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return strata_selftest(storage.as_ref());
    }
    // 1. Consistent snapshot of the live store into a throwaway tempdir.
    let dir = tempfile::TempDir::new().map_err(|e| format!("tempdir failed: {e}"))?;
    let copy_path = dir.path().join("selftest-copy.db");
    storage
        .backup_to(&copy_path)
        .map_err(|e| format!("backing the store up for the selftest failed: {e}"))?;

    // 2. Open the copy as its own storage. Everything below mutates only this.
    let copy = Arc::new(
        vestige_core::open_storage(Some(copy_path))
            .map_err(|e| format!("opening the selftest copy failed: {e}"))?,
    );

    // 3. Five planted-cause rounds, each isolated in its own scope.
    let mut outcomes: Vec<RoundOutcome> = Vec::new();
    for k in 1..=5u32 {
        let scope = format!("selftest-round-{k}");
        let cause_id = plant_cause(&copy, &scope, k)?;
        let failure_id = plant_failure(&copy, &scope, k, true)?;
        let out = run_backfill(&copy, &scope, &failure_id).await?;
        let causes = out["causes"].as_array().cloned().unwrap_or_default();
        let rank = causes
            .iter()
            .position(|c| c["memory_id"].as_str() == Some(cause_id.as_str()));
        outcomes.push(RoundOutcome {
            round: k,
            cause_id,
            hit_at_1: rank == Some(0),
            hit_at_3: rank.is_some_and(|r| r < 3),
            rank,
        });
    }

    let hits = outcomes.iter().filter(|o| o.hit_at_1).count();
    let hits_at_3 = outcomes.iter().filter(|o| o.hit_at_3).count();
    let misses = outcomes.len() - hits;

    // 4. Gap-calibration round: NO shared anchor. The backfill must fire its
    //    gap report and name the failure's env-shaped anchor as the missing
    //    record class, instead of surfacing an unrelated cause. The only
    //    in-window record is an unrelated change that shares nothing.
    let gap_scope = "selftest-round-6".to_string();
    let unrelated = copy
        .ingest_in_scope(
            IngestInput {
                content: "Rotated an unrelated credential in src/selftest/other.rs".to_string(),
                node_type: "decision".to_string(),
                tags: vec!["SELFTEST_UNRELATED".to_string()],
                ..Default::default()
            },
            &gap_scope,
        )
        .map_err(|e| format!("round 6: planting unrelated record failed: {e}"))?;
    copy.set_created_at(
        &unrelated.id,
        chrono::Utc::now() - chrono::Duration::days(5),
    )
    .map_err(|e| format!("round 6: backdating failed: {e}"))?;
    let gap_failure_id = plant_failure(&copy, &gap_scope, 6, false)?;
    let gap_out = run_backfill(&copy, &gap_scope, &gap_failure_id).await?;

    let gap_fired = gap_out["triggered"] == json!(true)
        && gap_out["causes"].as_array().is_some_and(Vec::is_empty)
        && gap_out["gap"].is_object();
    let missing: Vec<String> = gap_out["gap"]["missing_entities"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut missing = missing;
    missing.sort();
    let named_missing_anchor = missing.iter().find(|e| *e == "planted_cause_6").cloned();
    let gap_calibration = gap_fired && named_missing_anchor.is_some();

    // 5. Drop the copy (closing its connections) and remove the temp store.
    drop(copy);
    let temp_store_deleted = dir.close().is_ok();

    Ok(json!({
        "tool": "selftest",
        "kind": "planted_cause_selftest",
        "rounds": outcomes.len(),
        "hits": hits,
        "misses": misses,
        "hit_at_3": hits_at_3,
        "hit_rate_1": (hits as f64 / outcomes.len() as f64 * 1000.0).round() / 1000.0,
        "hit_rate_3": (hits_at_3 as f64 / outcomes.len() as f64 * 1000.0).round() / 1000.0,
        "rounds_detail": outcomes.iter().map(|o| json!({
            "round": o.round,
            "cause_id": o.cause_id,
            "rank": o.rank.map(|r| r + 1),
            "hit_at_1": o.hit_at_1,
            "hit_at_3": o.hit_at_3,
        })).collect::<Vec<_>>(),
        "gap_calibration": gap_calibration,
        "gap": {
            "fired": gap_fired,
            "named_missing_anchor": named_missing_anchor,
            "missing_entities": missing,
        },
        "deterministic": true,
        "live_store_touched": false,
        "temp_store_deleted": temp_store_deleted,
        "note": "Planted 5 quiet causes (env-shaped anchor + file path, backdated 5d) plus failures sharing exactly one anchor, ran the real backfill against a temp copy of the store, and scored hit@1/hit@3. The 6th round shares no anchor and must fire the gap report naming the missing anchor. The live store is only read (backup_to snapshot).",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    fn live_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("live.db"))).unwrap();
        (storage, dir)
    }

    /// The full planted-cause flow against a seeded (non-empty) store: every
    /// round must hit@1, the gap round must fire and name the missing
    /// anchor, the temp store must be deleted, and the LIVE store must be
    /// untouched (same nodes, no selftest scopes).
    #[tokio::test]
    async fn planted_cause_selftest_scores_perfectly_and_never_touches_the_live_store() {
        let (storage, _dir) = live_store();
        // Seed one ordinary memory so the backup copies real content.
        storage
            .ingest(IngestInput {
                content: "Ordinary note: the coffee machine is on the third floor".to_string(),
                ..Default::default()
            })
            .unwrap();

        let out = execute(&storage, None).await.expect("selftest must run");

        assert_eq!(out["rounds"], json!(5));
        assert_eq!(
            out["hits"],
            json!(5),
            "every planted cause must be rank 1: {out}"
        );
        assert_eq!(out["misses"], json!(0));
        assert_eq!(out["hit_at_3"], json!(5));
        assert_eq!(out["hit_rate_1"], json!(1.0));
        assert_eq!(out["deterministic"], json!(true));
        assert_eq!(out["temp_store_deleted"], json!(true));
        assert_eq!(out["live_store_touched"], json!(false));

        // Calibration metric: the no-shared-anchor round fires the gap report
        // and names the missing env-shaped anchor class.
        assert_eq!(out["gap_calibration"], json!(true), "{out}");
        assert_eq!(out["gap"]["fired"], json!(true));
        assert_eq!(out["gap"]["named_missing_anchor"], json!("planted_cause_6"));

        // Live store untouched: same single node, nothing in the selftest scopes.
        let nodes = storage.get_all_nodes(100, 0).unwrap();
        assert_eq!(
            nodes.len(),
            1,
            "no planted record may leak into the live store"
        );
        assert!(
            storage
                .get_all_nodes_in_scope("selftest-round-1", 100, 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            storage
                .get_all_nodes_in_scope("selftest-round-6", 100, 0)
                .unwrap()
                .is_empty()
        );
    }

    /// The selftest works on an empty store too (a fresh install's baseline).
    #[tokio::test]
    async fn planted_cause_selftest_runs_on_an_empty_store() {
        let (storage, _dir) = live_store();
        let out = execute(&storage, None).await.expect("selftest must run");
        assert_eq!(out["hits"], json!(5), "{out}");
        assert_eq!(out["gap_calibration"], json!(true));
    }
}

#[cfg(test)]
mod strata_stdio_tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use crate::protocol::stdio::run_io;
    use crate::server::McpServer;
    use std::path::Path;
    use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};

    fn fingerprint(storage: &Storage) -> (Vec<String>, Vec<String>) {
        snapshot(storage).expect("user store snapshot")
    }

    fn sqlite_files(dir: &Path) -> Vec<String> {
        let mut found = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let entries = match std::fs::read_dir(&path) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with(".sqlite")
                    || name.ends_with(".sqlite3")
                    || name.ends_with(".db")
                    || name.ends_with(".db-wal")
                    || name.ends_with(".db-shm")
                {
                    found.push(path.display().to_string());
                }
            }
        }
        found.sort();
        found
    }

    #[tokio::test]
    async fn selftest_over_stdio_passes_every_check_and_leaves_the_user_store_unchanged() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        storage
            .ingest(IngestInput {
                content: "user memory that selftest must not touch".to_string(),
                ..Default::default()
            })
            .unwrap();
        let server = McpServer::new(
            Arc::clone(&storage),
            Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new())),
        );
        let before = fingerprint(storage.as_ref());

        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, mut client_r) = tokio::io::duplex(1 << 20);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );

        let mut input = serde_json::to_string(&json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "selftest", "version": "1"}
            }
        }))
        .unwrap();
        input.push('\n');
        input.push_str(
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))
            .unwrap(),
        );
        input.push('\n');
        for id in [1, 2] {
            input.push_str(
                &serde_json::to_string(&json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": "tools/call",
                    "params": {"name": "selftest", "arguments": {}}
                }))
                .unwrap(),
            );
            input.push('\n');
        }
        client_w.write_all(input.as_bytes()).await.unwrap();
        drop(client_w);
        let mut buf = String::new();
        client_r.read_to_string(&mut buf).await.unwrap();
        handle.await.unwrap().unwrap();

        let lines: Vec<Value> = buf
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("stdio line is JSON"))
            .collect();
        let body = |id: i64| {
            let msg = lines
                .iter()
                .find(|value| value["id"] == json!(id))
                .unwrap_or_else(|| panic!("missing id {id}: {lines:?}"));
            assert!(msg.get("error").is_none(), "{msg}");
            let result = &msg["result"];
            assert_eq!(result["isError"], json!(false), "{result}");
            assert!(
                !result.to_string().contains("pending_strata"),
                "selftest must complete: {result}"
            );
            result["structuredContent"].clone()
        };
        let first = body(1);
        let second = body(2);
        assert_eq!(first, second, "selftest output must be deterministic");
        assert_eq!(first["tool"], json!("selftest"));
        assert_eq!(first["passed"], json!(true), "{first}");
        assert_eq!(first["deterministic"], json!(true));
        assert_eq!(first["live_store_touched"], json!(false));
        assert_eq!(first["temp_store_deleted"], json!(true));
        assert_eq!(first["walk"].as_array().map(Vec::len), Some(2), "{first}");
        let checks = first["checks"].as_array().expect("checks");
        assert!(!checks.is_empty());
        for item in checks {
            assert_eq!(item["pass"], json!(true), "{item} in {first}");
        }
        assert_eq!(fingerprint(storage.as_ref()), before);
        assert!(
            sqlite_files(dir.path()).is_empty(),
            "selftest must not create a sqlite file: {:?}",
            sqlite_files(dir.path())
        );
    }
}
