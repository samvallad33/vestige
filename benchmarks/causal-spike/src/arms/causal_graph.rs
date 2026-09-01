//! Arm C — persistent causal graph (the moonshot primitive, v1).
//!
//! Not transfer entropy / PCMCI. Accumulated `backfill_candidate` edges plus
//! lagged temporal-precedence traversal to depth 2.
//!
//! Pass 1: for every `looks_like_failure` memory, run Arm B and persist edges.
//! Pass 2: for every persisted cause (quiet intermediates), run a *manual*
//! backfill so root→intermediate edges exist. Without pass 2, depth-2 walk
//! from a failure cannot recover a root that shares no entity with the failure
//! — the intermediate never looks like a failure, so it would never be a
//! backfill source. This is compounding evidence accumulation, not a new
//! causal estimator.

use super::backfill::{engine, persist_edge, rank_backfill};
use super::{ArmRun, load_all_nodes, take_top_k};
use crate::types::{
    EVIDENCE_FLOOR, LAG_DECAY_TAU_DAYS, LOOKBACK_DAYS, QueryResult, squash_strength,
};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use vestige_core::advanced::retroactive_backfill::{extract_entities, looks_like_failure};
use vestige_core::{KnowledgeNode, Storage};

pub fn lag_decay(days: f64) -> f64 {
    if days < 0.0 {
        return 0.0;
    }
    (-days / LAG_DECAY_TAU_DAYS).exp()
}

pub fn run(storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    let all = load_all_nodes(storage)?;
    let acc_t0 = Instant::now();
    accumulate(storage, &all)?;
    let accumulation_ms = acc_t0.elapsed().as_secs_f64() * 1000.0;

    let nodes: HashMap<String, KnowledgeNode> =
        all.iter().cloned().map(|n| (n.id.clone(), n)).collect();
    let edges = load_evidence_edges(storage, &nodes)?;
    let mut queries = Vec::with_capacity(failure_ids.len());

    for failure_id in failure_ids {
        let t0 = Instant::now();
        let ranked = traverse(&nodes, &edges, failure_id);
        let (ranked_ids, scores) = take_top_k(ranked);
        queries.push(QueryResult {
            failure_id: failure_id.clone(),
            ranked_ids,
            scores,
            wall_clock_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    Ok(ArmRun {
        queries,
        search_mode: "causal_graph_v1".into(),
        accumulation_ms: Some(accumulation_ms),
    })
}

fn accumulate(storage: &Storage, all: &[KnowledgeNode]) -> Result<()> {
    let backfill = engine();
    let mut pass1_sources: HashSet<String> = HashSet::new();

    for node in all {
        if !looks_like_failure(&node.content, &node.tags) {
            continue;
        }
        let ranked = rank_backfill(storage, &backfill, node, all, false)?;
        for (cause_id, score) in ranked {
            if squash_strength(score) < EVIDENCE_FLOOR {
                continue;
            }
            persist_edge(storage, &cause_id, &node.id, score)?;
            pass1_sources.insert(cause_id);
        }
    }

    let mut pass1_sources: Vec<String> = pass1_sources.into_iter().collect();
    pass1_sources.sort();
    for cause_id in pass1_sources {
        let Some(cause_node) = all.iter().find(|n| n.id == cause_id) else {
            continue;
        };
        if looks_like_failure(&cause_node.content, &cause_node.tags) {
            continue;
        }
        let ranked = rank_backfill(storage, &backfill, cause_node, all, true)?;
        for (root_id, score) in ranked {
            if squash_strength(score) < EVIDENCE_FLOOR {
                continue;
            }
            persist_edge(storage, &root_id, &cause_id, score)?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct EvidenceEdge {
    source_id: String,
    target_id: String,
    strength: f64,
}

fn load_evidence_edges(
    storage: &Storage,
    nodes: &HashMap<String, KnowledgeNode>,
) -> Result<Vec<EvidenceEdge>> {
    let mut out = Vec::new();
    for conn in storage.get_all_connections()? {
        if conn.link_type != "backfill_candidate" {
            continue;
        }
        if conn.strength < EVIDENCE_FLOOR {
            continue;
        }
        let Some(src) = nodes.get(&conn.source_id) else {
            continue;
        };
        let Some(tgt) = nodes.get(&conn.target_id) else {
            continue;
        };
        if src.created_at >= tgt.created_at {
            continue;
        }
        let src_ents: HashSet<String> = extract_entities(&src.content, &src.tags)
            .into_iter()
            .collect();
        let tgt_ents = extract_entities(&tgt.content, &tgt.tags);
        if !tgt_ents.iter().any(|e| src_ents.contains(e)) {
            continue;
        }
        out.push(EvidenceEdge {
            source_id: conn.source_id,
            target_id: conn.target_id,
            strength: conn.strength,
        });
    }
    Ok(out)
}

fn traverse(
    nodes: &HashMap<String, KnowledgeNode>,
    edges: &[EvidenceEdge],
    failure_id: &str,
) -> Vec<(String, f64)> {
    let Some(failure) = nodes.get(failure_id) else {
        return Vec::new();
    };
    let mut incoming: HashMap<&str, Vec<&EvidenceEdge>> = HashMap::new();
    for e in edges {
        incoming.entry(e.target_id.as_str()).or_default().push(e);
    }

    let mut scores: HashMap<String, f64> = HashMap::new();

    // Depth 1: cause → failure
    let d1 = incoming.get(failure_id).cloned().unwrap_or_default();
    for e in &d1 {
        let Some(cause) = nodes.get(&e.source_id) else {
            continue;
        };
        if cause.created_at >= failure.created_at {
            continue;
        }
        let days = (failure.created_at - cause.created_at).num_seconds() as f64 / 86_400.0;
        if days <= 0.0 || days > LOOKBACK_DAYS as f64 {
            continue;
        }
        let s = e.strength * lag_decay(days);
        scores
            .entry(e.source_id.clone())
            .and_modify(|g| *g = g.max(s))
            .or_insert(s);

        // Depth 2: root → cause → failure
        if let Some(up) = incoming.get(e.source_id.as_str()) {
            for e2 in up {
                let Some(root) = nodes.get(&e2.source_id) else {
                    continue;
                };
                if root.created_at >= cause.created_at || cause.created_at >= failure.created_at {
                    continue;
                }
                let d_root = (cause.created_at - root.created_at).num_seconds() as f64 / 86_400.0;
                if d_root <= 0.0 || d_root > LOOKBACK_DAYS as f64 {
                    continue;
                }
                let hop = e2.strength * lag_decay(d_root);
                let path = hop * s;
                scores
                    .entry(e2.source_id.clone())
                    .and_modify(|g| *g = g.max(path))
                    .or_insert(path);
            }
        }
    }

    scores.remove(failure_id);
    scores.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_decay_is_one_at_zero_and_shrinks() {
        assert!((lag_decay(0.0) - 1.0).abs() < 1e-12);
        assert!(lag_decay(30.0) < lag_decay(1.0));
        assert!(lag_decay(-1.0) == 0.0);
    }

    #[test]
    fn evidence_floor_matches_protocol() {
        assert_eq!(EVIDENCE_FLOOR, 0.2);
    }

    #[test]
    fn live_graph_recovers_root_via_compounding_hop() {
        use crate::seed::failure_text;
        use crate::types::{bridge_name, entity_name, t0};
        use vestige_core::IngestInput;

        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let epoch = t0();
        let entity = entity_name(0, 0);
        let bridge = bridge_name(0, 0);

        let root = storage
            .ingest(IngestInput {
                content: format!("Recorded {entity} in the toolchain file during the weekly pass"),
                node_type: "decision".into(),
                tags: vec![entity.clone()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&root.id, epoch - chrono::Duration::days(20))
            .unwrap();

        let cause = storage
            .ingest(IngestInput {
                content: format!("Copied {entity} into {bridge} for the deploy env"),
                node_type: "decision".into(),
                tags: vec![entity, bridge.clone()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&cause.id, epoch - chrono::Duration::days(12))
            .unwrap();

        let failure = storage
            .ingest(IngestInput {
                content: failure_text(0),
                node_type: "event".into(),
                tags: vec![bridge, "crash".into()],
                ..Default::default()
            })
            .unwrap();
        storage.set_created_at(&failure.id, epoch).unwrap();

        let ArmRun { queries, .. } = run(&storage, std::slice::from_ref(&failure.id)).unwrap();
        let ranked = &queries[0].ranked_ids;
        assert!(
            ranked.iter().any(|id| id == &root.id),
            "depth-2 graph must recover the root cause, ranked={ranked:?} root={} cause={}",
            root.id,
            cause.id
        );
        assert!(
            ranked.iter().any(|id| id == &cause.id),
            "depth-1 must still recover the intermediate, ranked={ranked:?}"
        );
    }
}
