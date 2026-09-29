//! `selftest` MCP tool — recorded-edge walk on a throwaway Strata log.
//!
//! The live store is never mutated. The tool opens a fresh temp log (it does
//! not copy, vacuum, or seal the caller's store), plants a cause → intermediate
//! → symptom chain as recorded memories plus recorded `derived_from` edges,
//! plants distractors (shared wording, a non-causal incoming edge, a forward
//! causal edge, a disconnected chain, an ancestor past the bound, and a
//! cycle), then runs a bounded backward walk from the symptom.
//!
//! The walk reads recorded edges only. It does not read content, tags,
//! entities, keywords, FTS, embeddings, or SQLite, and it does not call a
//! model. Pass/fail is one deterministic check per property. The temp log
//! is deleted before the tool returns.

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vestige_core::{ConnectionRecord, IngestInput, Storage};

const CAUSAL_LINK: &str = "derived_from";
/// Hop cap. Depth 0 is the symptom, so 2 reaches the cause and stops before
/// the ancestor planted one hop further back.
const WALK_BOUND: usize = 2;
const PLANTED_NODES: usize = 9;
const PLANTED_EDGES: usize = 8;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Plant cause, intermediate, and symptom plus distractors in a throwaway Strata log, walk backward over recorded derived_from edges only, report pass/fail per check, and delete the temp log. The caller store is not mutated.",
        "properties": {}
    })
}

struct Planted {
    cause: String,
    intermediate: String,
    symptom: String,
    ancestor: String,
    overlap: String,
    noncausal: String,
    forward: String,
    side_a: String,
    side_b: String,
}

#[derive(Clone, PartialEq, Eq)]
struct Hop {
    depth: usize,
    memory_id: String,
    via: Option<String>,
}

struct Check {
    name: &'static str,
    pass: bool,
    detail: String,
}

pub async fn execute(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    let before = snapshot(storage.as_ref())?;
    let user_dir = storage.data_dir().to_path_buf();

    let dir = tempfile::TempDir::new().map_err(|e| format!("tempdir failed: {e}"))?;
    let temp_dir = dir.path().to_path_buf();
    let isolated = paths_isolated(&user_dir, &temp_dir);

    let temp = crate::strata_memory::open(&temp_dir)
        .map_err(|e| format!("opening the temp store failed: {e}"))?;
    let evaluated = evaluate(&temp);
    drop(temp);
    let temp_store_deleted = dir.close().is_ok() && !temp_dir.exists();

    let after = snapshot(storage.as_ref())?;
    let user_store_unchanged = before == after;

    let (mut checks, walk, walk_twice_equal) = evaluated?;
    checks.push(check(
        "temp_store_deleted",
        temp_store_deleted,
        if temp_store_deleted {
            "throwaway log removed"
        } else {
            "throwaway log still present"
        },
    ));
    checks.push(check(
        "temp_store_isolated",
        isolated,
        if isolated {
            "temp log is not the caller data dir"
        } else {
            "temp log overlaps the caller data dir"
        },
    ));
    checks.push(check(
        "user_store_unchanged",
        user_store_unchanged,
        if user_store_unchanged {
            "caller node and edge snapshot matched"
        } else {
            "caller node or edge snapshot changed"
        },
    ));
    checks.sort_by(|a, b| a.name.cmp(b.name));

    let passed = checks.iter().all(|c| c.pass) && walk_twice_equal;
    let walk_json: Vec<Value> = walk
        .iter()
        .map(|hop| {
            json!({
                "depth": hop.depth,
                "role": hop.role,
                "memory_id": hop.memory_id,
                "via": hop.via,
            })
        })
        .collect();

    Ok(json!({
        "tool": "selftest",
        "kind": "recorded_edge_walk",
        "status": "completed",
        "passed": passed,
        "deterministic": walk_twice_equal,
        "bound": WALK_BOUND,
        "link_type": CAUSAL_LINK,
        "live_store_touched": !user_store_unchanged,
        "temp_store_deleted": temp_store_deleted,
        "checks": checks.into_iter().map(|c| json!({
            "name": c.name,
            "pass": c.pass,
            "detail": c.detail,
        })).collect::<Vec<_>>(),
        "walk": walk_json,
    }))
}

struct LabeledHop {
    depth: usize,
    role: &'static str,
    memory_id: String,
    via: Option<String>,
}

fn evaluate(temp: &Arc<Storage>) -> Result<(Vec<Check>, Vec<LabeledHop>, bool), String> {
    let planted = plant(temp)?;
    let walk = backward_walk(temp.as_ref(), &planted.symptom, WALK_BOUND)?;
    let again = backward_walk(temp.as_ref(), &planted.symptom, WALK_BOUND)?;
    let walk_twice_equal = walk == again;
    let checks = fixture_checks(temp.as_ref(), &planted, &walk, walk_twice_equal)?;
    let labeled = label_walk(&planted, &walk);
    Ok((checks, labeled, walk_twice_equal))
}

fn plant(storage: &Arc<Storage>) -> Result<Planted, String> {
    // Wording overlaps on purpose. The walk must not use it.
    let cause = ingest(storage, "cause raised PLANTED_CHAIN_TOKEN pool limit")?;
    let intermediate = ingest(
        storage,
        "intermediate relay of PLANTED_CHAIN_TOKEN pressure",
    )?;
    let symptom = ingest(
        storage,
        "symptom outage after PLANTED_CHAIN_TOKEN cache stampede",
    )?;
    let ancestor = ingest(storage, "ancestor deploy of PLANTED_CHAIN_TOKEN binary")?;
    let overlap = ingest(
        storage,
        "distractor note mentioning symptom outage PLANTED_CHAIN_TOKEN but no edge",
    )?;
    let noncausal = ingest(storage, "distractor file touch beside the symptom")?;
    let forward = ingest(storage, "distractor downstream alert of the symptom")?;
    let side_a = ingest(storage, "distractor billing job PLANTED_CHAIN_TOKEN")?;
    let side_b = ingest(storage, "distractor invoice delay PLANTED_CHAIN_TOKEN")?;

    // Upstream is source, downstream is target. Backward walk follows target → source.
    record_edge(storage, &ancestor, &cause, CAUSAL_LINK)?;
    record_edge(storage, &cause, &intermediate, CAUSAL_LINK)?;
    record_edge(storage, &intermediate, &symptom, CAUSAL_LINK)?;
    // Cycle inside the bound: symptom also points at intermediate.
    record_edge(storage, &symptom, &intermediate, CAUSAL_LINK)?;
    record_edge(storage, &symptom, &noncausal, "touched")?;
    record_edge(storage, &noncausal, &symptom, "anchored_to")?;
    record_edge(storage, &symptom, &forward, CAUSAL_LINK)?;
    record_edge(storage, &side_a, &side_b, CAUSAL_LINK)?;

    Ok(Planted {
        cause,
        intermediate,
        symptom,
        ancestor,
        overlap,
        noncausal,
        forward,
        side_a,
        side_b,
    })
}

fn ingest(storage: &Arc<Storage>, content: &str) -> Result<String, String> {
    let node = storage
        .ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .map_err(|e| format!("planting memory failed: {e}"))?;
    Ok(node.id)
}

fn record_edge(
    storage: &Arc<Storage>,
    source: &str,
    target: &str,
    link_type: &str,
) -> Result<(), String> {
    let stamp = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(0)
        .ok_or_else(|| "epoch timestamp missing".to_string())?;
    storage
        .save_connection(&ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: stamp,
            last_activated: stamp,
            activation_count: 0,
        })
        .map_err(|e| format!("recording {link_type} edge failed: {e}"))
}

/// Bounded backward walk over recorded causal edges.
///
/// An edge counts only when it is stored, `link_type` is [`CAUSAL_LINK`], and
/// the current node is its target (the effect). The predecessor is the source.
/// Content, tags, entities, keywords, FTS, embeddings, and models are not read.
fn backward_walk(storage: &Storage, start: &str, bound: usize) -> Result<Vec<Hop>, String> {
    let mut visited = BTreeSet::from([start.to_string()]);
    let mut hops = vec![Hop {
        depth: 0,
        memory_id: start.to_string(),
        via: None,
    }];
    let mut frontier = vec![start.to_string()];
    for depth in 1..=bound {
        let mut next: Vec<(String, String)> = Vec::new();
        for current in &frontier {
            let mut incoming = incoming_causal(storage, current)?;
            incoming.sort_by(|a, b| {
                a.source_id
                    .cmp(&b.source_id)
                    .then(a.link_type.cmp(&b.link_type))
            });
            for edge in incoming {
                if visited.insert(edge.source_id.clone()) {
                    next.push((edge.source_id, edge.link_type));
                }
            }
        }
        next.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if next.is_empty() {
            break;
        }
        let mut frontier_next = Vec::with_capacity(next.len());
        for (memory_id, via) in next {
            hops.push(Hop {
                depth,
                memory_id: memory_id.clone(),
                via: Some(via),
            });
            frontier_next.push(memory_id);
        }
        frontier = frontier_next;
    }
    Ok(hops)
}

fn incoming_causal(storage: &Storage, current: &str) -> Result<Vec<ConnectionRecord>, String> {
    let edges = storage
        .get_connections_for_memory(current)
        .map_err(|e| format!("reading recorded edges failed: {e}"))?;
    Ok(edges
        .into_iter()
        .filter(|edge| {
            edge.target_id == current && edge.source_id != current && edge.link_type == CAUSAL_LINK
        })
        .collect())
}

fn fixture_checks(
    storage: &Storage,
    planted: &Planted,
    walk: &[Hop],
    walk_twice_equal: bool,
) -> Result<Vec<Check>, String> {
    let nodes = all_nodes(storage)?;
    let edges = storage
        .get_all_connections()
        .map_err(|e| format!("reading temp edges failed: {e}"))?;
    let ids: BTreeSet<&str> = walk.iter().map(|hop| hop.memory_id.as_str()).collect();
    let at = |depth: usize| -> Option<&Hop> { walk.iter().find(|hop| hop.depth == depth) };

    let mut checks = vec![
        check(
            "planted_cause",
            nodes.iter().any(|node| node.id == planted.cause),
            "cause memory is on the temp log",
        ),
        check(
            "planted_intermediate",
            nodes.iter().any(|node| node.id == planted.intermediate),
            "intermediate memory is on the temp log",
        ),
        check(
            "planted_symptom",
            nodes.iter().any(|node| node.id == planted.symptom),
            "symptom memory is on the temp log",
        ),
        check(
            "causal_edge_cause_intermediate",
            has_recorded_edge(&edges, &planted.cause, &planted.intermediate, CAUSAL_LINK),
            "derived_from cause to intermediate is recorded",
        ),
        check(
            "causal_edge_intermediate_symptom",
            has_recorded_edge(&edges, &planted.intermediate, &planted.symptom, CAUSAL_LINK),
            "derived_from intermediate to symptom is recorded",
        ),
        check(
            "temp_store_is_fresh",
            nodes.len() == PLANTED_NODES,
            "temp log holds only the planted memories",
        ),
        check(
            "temp_store_edge_count",
            edges.len() == PLANTED_EDGES,
            "temp log holds only the planted edges",
        ),
        check(
            "overlap_has_no_edges",
            edge_touch_count(&edges, &planted.overlap) == 0,
            "keyword-overlapping distractor has no recorded edge",
        ),
        check(
            "noncausal_edge_recorded",
            has_recorded_edge(&edges, &planted.noncausal, &planted.symptom, "anchored_to"),
            "non-causal incoming edge is recorded and must be ignored",
        ),
        check(
            "forward_edge_recorded",
            has_recorded_edge(&edges, &planted.symptom, &planted.forward, CAUSAL_LINK),
            "forward derived_from edge is recorded and must not be walked backward",
        ),
        check(
            "disconnected_edge_recorded",
            has_recorded_edge(&edges, &planted.side_a, &planted.side_b, CAUSAL_LINK),
            "disconnected causal chain is recorded",
        ),
        check(
            "ancestor_edge_recorded",
            has_recorded_edge(&edges, &planted.ancestor, &planted.cause, CAUSAL_LINK),
            "ancestor beyond the bound is recorded",
        ),
        check(
            "walk_reaches_intermediate",
            at(1).is_some_and(|hop| {
                hop.memory_id == planted.intermediate && hop.via.as_deref() == Some(CAUSAL_LINK)
            }),
            "depth 1 is the intermediate over derived_from",
        ),
        check(
            "walk_reaches_cause",
            at(2).is_some_and(|hop| {
                hop.memory_id == planted.cause && hop.via.as_deref() == Some(CAUSAL_LINK)
            }),
            "depth 2 is the cause over derived_from",
        ),
        check(
            "walk_node_count",
            walk.len() == 3 && ids.len() == 3,
            "walk is symptom, intermediate, cause",
        ),
        check(
            "walk_recorded_edges_only",
            walk.iter()
                .all(|hop| hop.depth == 0 || hop.via.as_deref() == Some(CAUSAL_LINK)),
            "every hop is a recorded derived_from edge",
        ),
        check(
            "walk_respects_bound",
            walk.iter().all(|hop| hop.depth <= WALK_BOUND)
                && !ids.contains(planted.ancestor.as_str()),
            "ancestor past the bound is not reached",
        ),
        check(
            "cycle_not_revisited",
            walk.iter()
                .filter(|hop| hop.memory_id == planted.symptom)
                .count()
                == 1,
            "cycle edge back to the symptom is not re-entered",
        ),
        check(
            "overlap_distractor_excluded",
            !ids.contains(planted.overlap.as_str()),
            "shared wording is not a walk admission",
        ),
        check(
            "noncausal_neighbor_excluded",
            !ids.contains(planted.noncausal.as_str()),
            "anchored_to incoming edge is not followed",
        ),
        check(
            "forward_edge_excluded",
            !ids.contains(planted.forward.as_str()),
            "outgoing derived_from edge is not a backward hop",
        ),
        check(
            "disconnected_chain_excluded",
            !ids.contains(planted.side_a.as_str()) && !ids.contains(planted.side_b.as_str()),
            "unconnected causal chain is not reached",
        ),
        check(
            "walk_deterministic",
            walk_twice_equal,
            "second walk matched the first",
        ),
    ];
    checks.sort_by(|a, b| a.name.cmp(b.name));
    Ok(checks)
}

fn label_walk(planted: &Planted, walk: &[Hop]) -> Vec<LabeledHop> {
    walk.iter()
        .map(|hop| LabeledHop {
            depth: hop.depth,
            role: role_of(planted, &hop.memory_id),
            memory_id: hop.memory_id.clone(),
            via: hop.via.clone(),
        })
        .collect()
}

fn role_of(planted: &Planted, id: &str) -> &'static str {
    if id == planted.symptom {
        "symptom"
    } else if id == planted.intermediate {
        "intermediate"
    } else if id == planted.cause {
        "cause"
    } else if id == planted.ancestor {
        "ancestor"
    } else if id == planted.overlap {
        "overlap"
    } else if id == planted.noncausal {
        "noncausal"
    } else if id == planted.forward {
        "forward"
    } else if id == planted.side_a {
        "side_a"
    } else if id == planted.side_b {
        "side_b"
    } else {
        "unlabeled"
    }
}

fn has_recorded_edge(
    edges: &[ConnectionRecord],
    source: &str,
    target: &str,
    link_type: &str,
) -> bool {
    edges.iter().any(|edge| {
        edge.source_id == source && edge.target_id == target && edge.link_type == link_type
    })
}

fn edge_touch_count(edges: &[ConnectionRecord], id: &str) -> usize {
    edges
        .iter()
        .filter(|edge| edge.source_id == id || edge.target_id == id)
        .count()
}

fn all_nodes(storage: &Storage) -> Result<Vec<vestige_core::KnowledgeNode>, String> {
    let mut nodes = Vec::new();
    let mut offset = 0i32;
    loop {
        let page = storage
            .get_all_nodes(1000, offset)
            .map_err(|e| format!("reading memories failed: {e}"))?;
        if page.is_empty() {
            break;
        }
        let len = i32::try_from(page.len()).unwrap_or(i32::MAX);
        offset = offset.saturating_add(len);
        let done = page.len() < 1000;
        nodes.extend(page);
        if done {
            break;
        }
    }
    Ok(nodes)
}

fn check(name: &'static str, pass: bool, detail: &str) -> Check {
    Check {
        name,
        pass,
        detail: detail.to_string(),
    }
}

fn paths_isolated(user: &Path, temp: &Path) -> bool {
    let user = canonicalize_or_buf(user);
    let temp = canonicalize_or_buf(temp);
    user != temp && !temp.starts_with(&user) && !user.starts_with(&temp)
}

fn canonicalize_or_buf(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Content-bearing snapshot of the caller store. Read-only: ids, content,
/// type, tags, and recorded edge endpoints. No scores, no clock.
fn snapshot(storage: &Storage) -> Result<String, String> {
    let mut nodes = all_nodes(storage)?;
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    let mut edges = storage
        .get_all_connections()
        .map_err(|e| format!("reading caller edges failed: {e}"))?;
    edges.sort_by(|a, b| {
        (&a.source_id, &a.target_id, &a.link_type).cmp(&(&b.source_id, &b.target_id, &b.link_type))
    });
    let mut out = String::new();
    for node in &nodes {
        let mut tags = node.tags.clone();
        tags.sort();
        out.push_str(&node.id);
        out.push('\t');
        out.push_str(&node.node_type);
        out.push('\t');
        out.push_str(&tags.join(","));
        out.push('\t');
        out.push_str(&node.content);
        out.push('\n');
    }
    out.push_str("---\n");
    for edge in &edges {
        out.push_str(&edge.source_id);
        out.push('\t');
        out.push_str(&edge.target_id);
        out.push('\t');
        out.push_str(&edge.link_type);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    #[tokio::test]
    async fn recorded_edge_selftest_passes_and_leaves_the_caller_store_unchanged() {
        let (storage, dir) = user_store();
        storage
            .ingest(IngestInput {
                content: "caller memory that selftest must not alter".to_string(),
                ..Default::default()
            })
            .unwrap();
        let before_nodes = storage.get_all_nodes(100, 0).unwrap();
        let before_edges = storage.get_all_connections().unwrap();
        let before_files = tree_bytes(dir.path());

        let first = execute(&storage, None).await.expect("selftest");
        let second = execute(&storage, None).await.expect("selftest");

        assert_eq!(first, second, "two runs must be identical");
        assert_eq!(first["status"], json!("completed"));
        assert_eq!(first["passed"], json!(true), "{first}");
        assert_eq!(first["deterministic"], json!(true));
        assert_eq!(first["live_store_touched"], json!(false));
        assert_eq!(first["temp_store_deleted"], json!(true));
        assert_eq!(first["link_type"], json!(CAUSAL_LINK));
        assert_eq!(first["bound"], json!(WALK_BOUND));

        let checks = first["checks"].as_array().expect("checks");
        assert!(checks.len() >= 20, "expected a check per property: {first}");
        for item in checks {
            assert_eq!(item["pass"], json!(true), "check failed: {item}");
        }
        let names: Vec<&str> = checks.iter().filter_map(|c| c["name"].as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "checks must be emitted in name order");

        let walk = first["walk"].as_array().expect("walk");
        assert_eq!(walk.len(), 3);
        assert_eq!(walk[0]["role"], json!("symptom"));
        assert_eq!(walk[0]["depth"], json!(0));
        assert_eq!(walk[1]["role"], json!("intermediate"));
        assert_eq!(walk[1]["via"], json!(CAUSAL_LINK));
        assert_eq!(walk[2]["role"], json!("cause"));
        assert_eq!(walk[2]["via"], json!(CAUSAL_LINK));

        let after_nodes = storage.get_all_nodes(100, 0).unwrap();
        let after_edges = storage.get_all_connections().unwrap();
        assert_eq!(after_nodes.len(), 1);
        assert_eq!(after_nodes[0].id, before_nodes[0].id);
        assert_eq!(after_nodes[0].content, before_nodes[0].content);
        assert_eq!(before_edges.len(), 0);
        assert_eq!(after_edges.len(), 0);
        assert_eq!(tree_bytes(dir.path()), before_files);
    }

    #[tokio::test]
    async fn recorded_edge_selftest_passes_on_an_empty_caller_store() {
        let (storage, _dir) = user_store();
        let out = execute(&storage, None).await.expect("selftest");
        assert_eq!(out["status"], json!("completed"));
        assert_eq!(out["passed"], json!(true), "{out}");
        assert!(storage.get_all_nodes(100, 0).unwrap().is_empty());
    }

    fn tree_bytes(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).unwrap();
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    // The directory lock names this process and is removed on drop.
                    if rel.ends_with("strata.lock") {
                        continue;
                    }
                    out.push((rel, std::fs::read(&path).unwrap()));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}
