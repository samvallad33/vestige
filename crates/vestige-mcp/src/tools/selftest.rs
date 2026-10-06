//! `selftest` MCP tool — recorded-edge walk on a throwaway store.
//!
//! The live store is read twice (a fingerprint before and after) and never
//! written. Everything planted lives in a fresh Strata log under a temp
//! directory that is deleted before the tool returns:
//!
//! cause → intermediate → symptom, as memories plus recorded `derived_from`
//! edges (each stored as `later derived_from earlier`), with distractors that a keyword, entity, or non-causal edge walk
//! would pick up. A local backward walk follows recorded causal edges only,
//! bounded, from the symptom. Each check is a deterministic pass/fail.

use std::collections::{HashSet, VecDeque};

use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::{ConnectionRecord, IngestInput, Storage};

/// Declared record → record lineage. The only edge kind the walk follows.
const CAUSAL_LINK: &str = "derived_from";
/// Non-lineage edge planted as a distractor. The walk must ignore it.
const NON_CAUSAL_LINK: &str = "touched";
/// Expand predecessors only while `depth < HOP_BOUND`.
/// The planted chain is symptom ← intermediate ← cause (depths 1 and 2).
/// The cycle edge sits on the cause, one hop past the bound, so a correct
/// walk records the chain and cuts the cycle.
const HOP_BOUND: usize = 2;
const PLANT_SCOPE: &str = "selftest";

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Planted-cause selftest; read-only.",
        "properties": {}
    })
}

struct Planted {
    cause: String,
    intermediate: String,
    symptom: String,
    distractor: String,
    side_branch: String,
    decoy: String,
    loopback: String,
}

struct Hop {
    id: String,
    role: &'static str,
    depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeFingerprint {
    id: String,
    content: String,
    node_type: String,
    tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeFingerprint {
    source_id: String,
    target_id: String,
    link_type: String,
    strength_milli: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StoreFingerprint {
    nodes: Vec<NodeFingerprint>,
    edges: Vec<EdgeFingerprint>,
}

fn fingerprint(storage: &Storage) -> Result<StoreFingerprint, String> {
    let mut nodes = Vec::new();
    let mut offset = 0i32;
    loop {
        let page = storage
            .get_all_nodes(500, offset)
            .map_err(|e| format!("reading the user store failed: {e}"))?;
        if page.is_empty() {
            break;
        }
        let count = i32::try_from(page.len()).unwrap_or(i32::MAX);
        for node in page {
            let mut tags = node.tags;
            tags.sort();
            nodes.push(NodeFingerprint {
                id: node.id,
                content: node.content,
                node_type: node.node_type,
                tags,
            });
        }
        offset = offset.saturating_add(count);
        if count < 500 {
            break;
        }
    }
    nodes.sort();

    let mut edges = Vec::new();
    for edge in storage
        .get_all_connections()
        .map_err(|e| format!("reading user-store edges failed: {e}"))?
    {
        edges.push(EdgeFingerprint {
            source_id: edge.source_id,
            target_id: edge.target_id,
            link_type: edge.link_type,
            strength_milli: (edge.strength * 1000.0).round() as i64,
        });
    }
    edges.sort();
    Ok(StoreFingerprint { nodes, edges })
}

fn plant_memory(storage: &Storage, content: &str) -> Result<String, String> {
    let node = storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            },
            PLANT_SCOPE,
        )
        .map_err(|e| format!("planting a selftest memory failed: {e}"))?;
    Ok(node.id)
}

/// The stored endpoints of an edge from an `earlier` record to a `later` one.
/// `later derived_from earlier` is stored from the derivative to its origin,
/// the way a link declared at save and a GhostLink weave record it. Other
/// kinds are stored from the earlier record to the later one.
fn stored_ends<'a>(earlier: &'a str, later: &'a str, link_type: &str) -> (&'a str, &'a str) {
    if link_type == CAUSAL_LINK {
        (later, earlier)
    } else {
        (earlier, later)
    }
}

fn record_edge(
    storage: &Storage,
    earlier: &str,
    later: &str,
    link_type: &str,
) -> Result<(), String> {
    let at = chrono::DateTime::UNIX_EPOCH;
    let (source, target) = stored_ends(earlier, later, link_type);
    storage
        .save_connection(&ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: at,
            last_activated: at,
            activation_count: 0,
        })
        .map_err(|e| format!("recording a selftest edge failed: {e}"))
}

fn edge_present(
    storage: &Storage,
    earlier: &str,
    later: &str,
    link_type: &str,
) -> Result<bool, String> {
    let (source, target) = stored_ends(earlier, later, link_type);
    let edges = storage
        .get_connections_for_memory(target)
        .map_err(|e| format!("reading recorded edges failed: {e}"))?;
    Ok(edges.iter().any(|edge| {
        edge.source_id == source && edge.target_id == target && edge.link_type == link_type
    }))
}

/// Plant cause → intermediate → symptom plus distractors.
///
/// Each chain edge is recorded as `later derived_from earlier`, the direction
/// a declared link has. The backward walk goes from a record to what it is
/// derived from.
fn plant(storage: &Storage) -> Result<Planted, String> {
    let symptom_text =
        "Selftest symptom record: observed failure at the end of the recorded chain.";
    let planted = Planted {
        cause: plant_memory(
            storage,
            "Selftest cause record: upstream change recorded before the intermediate hop.",
        )?,
        intermediate: plant_memory(
            storage,
            "Selftest intermediate record: the recorded hop between cause and symptom.",
        )?,
        symptom: plant_memory(storage, symptom_text)?,
        // Same sentence as the symptom, plus an unlinked tail. A keyword or
        // entity walk would surface this; a recorded-edge walk must not.
        distractor: plant_memory(
            storage,
            &format!("{symptom_text} Unlinked distractor sharing the symptom wording."),
        )?,
        side_branch: plant_memory(
            storage,
            "Selftest side branch: recorded downstream of the cause, not an ancestor of the symptom.",
        )?,
        decoy: plant_memory(
            storage,
            "Selftest decoy: touches the symptom through a non-causal edge and shares cause symptom words.",
        )?,
        loopback: plant_memory(
            storage,
            "Selftest loopback: causal cycle sitting one hop past the walk bound.",
        )?,
    };

    record_edge(storage, &planted.cause, &planted.intermediate, CAUSAL_LINK)?;
    record_edge(
        storage,
        &planted.intermediate,
        &planted.symptom,
        CAUSAL_LINK,
    )?;
    // Downstream of the cause, and also a forward edge out of the symptom.
    // Neither direction is a backward predecessor of the symptom except
    // through the cause, which the hop bound does not expand.
    record_edge(storage, &planted.cause, &planted.side_branch, CAUSAL_LINK)?;
    record_edge(storage, &planted.symptom, &planted.side_branch, CAUSAL_LINK)?;
    record_edge(storage, &planted.decoy, &planted.symptom, NON_CAUSAL_LINK)?;
    record_edge(storage, &planted.loopback, &planted.cause, CAUSAL_LINK)?;
    record_edge(storage, &planted.cause, &planted.loopback, CAUSAL_LINK)?;
    Ok(planted)
}

fn role_of(planted: &Planted, id: &str) -> &'static str {
    if id == planted.cause {
        "cause"
    } else if id == planted.intermediate {
        "intermediate"
    } else if id == planted.symptom {
        "symptom"
    } else if id == planted.distractor {
        "distractor"
    } else if id == planted.side_branch {
        "side_branch"
    } else if id == planted.decoy {
        "decoy"
    } else if id == planted.loopback {
        "loopback"
    } else {
        "unknown"
    }
}

/// Bounded backward walk over recorded `derived_from` edges.
///
/// From the current node, a predecessor is the target of a recorded
/// `derived_from` edge whose source is the current node: what this record is
/// derived from. Other link types, edges that point at the current node (the
/// records derived from it), and nodes past `hop_bound` are not followed. Predecessor ids are sorted
/// before enqueue so the BFS order is deterministic. Visited nodes are not
/// expanded twice, so a cycle ends.
fn backward_walk(
    storage: &Storage,
    planted: &Planted,
    hop_bound: usize,
) -> Result<Vec<Hop>, String> {
    let mut visited = HashSet::new();
    visited.insert(planted.symptom.clone());
    let mut queue = VecDeque::from([(planted.symptom.clone(), 0usize)]);
    let mut reached = Vec::new();

    while let Some((id, depth)) = queue.pop_front() {
        if depth >= hop_bound {
            continue;
        }
        let edges = storage
            .get_connections_for_memory(&id)
            .map_err(|e| format!("walking recorded edges failed: {e}"))?;
        let mut predecessors = Vec::new();
        for edge in edges {
            if edge.link_type != CAUSAL_LINK || edge.source_id != id || edge.target_id == id {
                continue;
            }
            predecessors.push(edge.target_id);
        }
        predecessors.sort();
        predecessors.dedup();
        for predecessor in predecessors {
            if !visited.insert(predecessor.clone()) {
                continue;
            }
            reached.push(Hop {
                role: role_of(planted, &predecessor),
                depth: depth + 1,
                id: predecessor.clone(),
            });
            queue.push_back((predecessor, depth + 1));
        }
    }
    Ok(reached)
}

fn reached_has(reached: &[Hop], role: &str, depth: usize) -> bool {
    reached
        .iter()
        .any(|hop| hop.role == role && hop.depth == depth)
}

fn reached_lacks(reached: &[Hop], role: &str) -> bool {
    reached.iter().all(|hop| hop.role != role)
}

pub async fn execute(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    let before = fingerprint(storage.as_ref())?;

    let dir = tempfile::TempDir::new().map_err(|e| format!("tempdir failed: {e}"))?;
    let temp = crate::strata_memory::open(dir.path())
        .map_err(|e| format!("opening the selftest store failed: {e}"))?;

    let planted = plant(temp.as_ref())?;
    let reached = backward_walk(temp.as_ref(), &planted, HOP_BOUND)?;

    let chain_edges_recorded = edge_present(
        temp.as_ref(),
        &planted.cause,
        &planted.intermediate,
        CAUSAL_LINK,
    )? && edge_present(
        temp.as_ref(),
        &planted.intermediate,
        &planted.symptom,
        CAUSAL_LINK,
    )?;
    let decoy_edge_recorded = edge_present(
        temp.as_ref(),
        &planted.decoy,
        &planted.symptom,
        NON_CAUSAL_LINK,
    )?;
    let forward_edge_recorded = edge_present(
        temp.as_ref(),
        &planted.symptom,
        &planted.side_branch,
        CAUSAL_LINK,
    )?;
    let cycle_edge_recorded = edge_present(
        temp.as_ref(),
        &planted.loopback,
        &planted.cause,
        CAUSAL_LINK,
    )?;

    // Strata ids are `mem-<seq>` inside each log, so a user memory and a
    // temp-store memory can share an id. Belonging is the temp chain, not
    // id inequality with the user store.
    let reached_ids_are_temp_chain = reached.iter().all(|hop| match hop.role {
        "cause" => hop.id == planted.cause,
        "intermediate" => hop.id == planted.intermediate,
        _ => false,
    });
    let predecessors_are_exact = reached.len() == 2
        && reached_has(&reached, "intermediate", 1)
        && reached_has(&reached, "cause", 2)
        && reached.iter().all(|hop| hop.depth <= HOP_BOUND);

    drop(temp);
    let temp_store_deleted = dir.close().is_ok();
    let after = fingerprint(storage.as_ref())?;
    let user_store_unchanged = before == after;

    let mut checks = vec![
        (
            "cause_reached_at_bound",
            reached_has(&reached, "cause", HOP_BOUND),
        ),
        ("chain_edges_recorded", chain_edges_recorded),
        (
            "cycle_cut_by_bound",
            cycle_edge_recorded && reached_lacks(&reached, "loopback"),
        ),
        (
            "decoy_edge_ignored",
            decoy_edge_recorded && reached_lacks(&reached, "decoy"),
        ),
        (
            "forward_edge_ignored",
            forward_edge_recorded && reached_lacks(&reached, "side_branch"),
        ),
        (
            "intermediate_reached_first",
            reached
                .first()
                .is_some_and(|hop| hop.role == "intermediate" && hop.depth == 1),
        ),
        (
            "keyword_distractor_excluded",
            reached_lacks(&reached, "distractor"),
        ),
        ("predecessors_are_exact", predecessors_are_exact),
        ("temp_store_deleted", temp_store_deleted),
        ("reached_ids_are_temp_chain", reached_ids_are_temp_chain),
        ("user_store_unchanged", user_store_unchanged),
    ];
    checks.sort_by_key(|check| check.0);

    let checks_passed = checks.iter().filter(|check| check.1).count();
    let all_passed = checks_passed == checks.len();
    let checks_json: Vec<Value> = checks
        .iter()
        .map(|(name, pass)| json!({ "name": name, "pass": pass }))
        .collect();
    let walk: Vec<Value> = reached
        .iter()
        .map(|hop| json!({ "role": hop.role, "depth": hop.depth }))
        .collect();

    Ok(json!({
        "tool": "selftest",
        "status": "completed",
        "kind": "recorded_edge_walk",
        "deterministic": true,
        "live_store_touched": !user_store_unchanged,
        "temp_store_deleted": temp_store_deleted,
        "hop_bound": HOP_BOUND,
        "walk": walk,
        "checks": checks_json,
        "checks_passed": checks_passed,
        "checks_total": checks.len(),
        "all_passed": all_passed,
        "note": "Planted cause, intermediate, and symptom with recorded derived_from edges plus distractors in a temp Strata log, walked backward over recorded causal edges only, and deleted the temp store. The user store was only fingerprinted.",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn assert_completed(out: &Value) {
        assert_eq!(out["tool"], json!("selftest"));
        assert_eq!(out["status"], json!("completed"));
        assert_eq!(out["kind"], json!("recorded_edge_walk"));
        assert_eq!(out["deterministic"], json!(true));
        assert_eq!(out["all_passed"], json!(true), "{out}");
        assert_eq!(out["live_store_touched"], json!(false));
        assert_eq!(out["temp_store_deleted"], json!(true));
        assert_eq!(out["hop_bound"], json!(HOP_BOUND));
        assert_eq!(out["checks_passed"], out["checks_total"]);
        assert_eq!(
            out["walk"],
            json!([
                {"role": "intermediate", "depth": 1},
                {"role": "cause", "depth": 2}
            ])
        );
        let checks = out["checks"].as_array().expect("checks");
        assert!(!checks.is_empty());
        for check in checks {
            assert_eq!(check["pass"], json!(true), "{check} in {out}");
        }
        let names: Vec<&str> = checks
            .iter()
            .filter_map(|check| check["name"].as_str())
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "checks must be reported in name order");
    }

    #[tokio::test]
    async fn recorded_edge_selftest_passes_and_leaves_a_seeded_store_unchanged() {
        let (storage, _dir) = user_store();
        let sentinel = storage
            .ingest(IngestInput {
                content: "Sentinel user memory: the kettle is on the third shelf.".to_string(),
                ..Default::default()
            })
            .unwrap();
        let before = fingerprint(storage.as_ref()).unwrap();

        let first = execute(&storage, None).await.expect("selftest must run");
        let second = execute(&storage, None)
            .await
            .expect("selftest must be repeatable");
        assert_completed(&first);
        assert_eq!(first, second, "two runs must be identical");

        let after = fingerprint(storage.as_ref()).unwrap();
        assert_eq!(before, after);
        let nodes = storage.get_all_nodes(100, 0).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id, sentinel.id);
        assert_eq!(nodes[0].content, sentinel.content);
        assert!(storage.get_all_connections().unwrap().is_empty());
        assert!(
            storage
                .get_all_nodes_in_scope(PLANT_SCOPE, 100, 0)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn recorded_edge_selftest_passes_on_an_empty_store() {
        let (storage, _dir) = user_store();
        let out = execute(&storage, None).await.expect("selftest must run");
        assert_completed(&out);
        assert!(storage.get_all_nodes(100, 0).unwrap().is_empty());
    }
}
