//! Arm B — current per-query Retroactive Salience Backfill (the shipping control).

use super::{ArmRun, load_all_nodes, take_top_k};
use crate::types::{EVIDENCE_FLOOR, LOOKBACK_DAYS, QueryResult, TOP_K, squash_strength};
use anyhow::Result;
use chrono::Utc;
use std::time::Instant;
use vestige_core::advanced::prediction_error::cosine_similarity;
use vestige_core::advanced::retroactive_backfill::{
    BackfillCandidate, FailureEvent, RetroactiveBackfill, extract_entities, looks_like_failure,
};
use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

pub fn engine() -> RetroactiveBackfill {
    RetroactiveBackfill {
        lookback_days: LOOKBACK_DAYS,
        max_causes: TOP_K,
        ..RetroactiveBackfill::new()
    }
}

pub fn run(storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    let all = load_all_nodes(storage)?;
    let mut queries = Vec::with_capacity(failure_ids.len());
    let backfill = engine();

    for failure_id in failure_ids {
        let t0 = Instant::now();
        let failure_node = all
            .iter()
            .find(|n| n.id == *failure_id)
            .ok_or_else(|| anyhow::anyhow!("failure {failure_id} not in store"))?;
        let ranked = rank_backfill(storage, &backfill, failure_node, &all, false)?;
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
        search_mode: "retroactive_backfill".into(),
        accumulation_ms: None,
    })
}

/// Rank candidates for one event. `force_manual` bypasses the salience gate so
/// Arm C's compounding hop can backfill from quiet intermediate causes.
pub fn rank_backfill(
    storage: &Storage,
    backfill: &RetroactiveBackfill,
    event: &KnowledgeNode,
    all: &[KnowledgeNode],
    force_manual: bool,
) -> Result<Vec<(String, f64)>> {
    let failure_embedding = storage.get_node_embedding(&event.id).ok().flatten();
    let pe = if looks_like_failure(&event.content, &event.tags) || force_manual {
        0.9_f32
    } else {
        0.3_f32
    };
    let failure = FailureEvent {
        id: event.id.clone(),
        content: event.content.clone(),
        entities: extract_entities(&event.content, &event.tags),
        tags: event.tags.clone(),
        prediction_error: pe,
        manual: force_manual,
    };
    let mut candidates = Vec::new();
    for node in all {
        if node.id == event.id {
            continue;
        }
        let age = (event.created_at - node.created_at).num_seconds() as f64 / 86_400.0;
        if age <= 0.0 {
            continue;
        }
        let sim = match (
            &failure_embedding,
            storage.get_node_embedding(&node.id).ok().flatten(),
        ) {
            (Some(f), Some(c)) if f.len() == c.len() => Some(cosine_similarity(f, &c)),
            _ => None,
        };
        candidates.push(BackfillCandidate {
            id: node.id.clone(),
            content: node.content.clone(),
            entities: extract_entities(&node.content, &node.tags),
            age_days_before_failure: age,
            stability: node.stability,
            similarity_to_failure: sim,
        });
    }
    let result = backfill.run(&failure, &candidates);
    Ok(result
        .causes
        .into_iter()
        .map(|c| (c.memory_id, c.score))
        .collect())
}

pub fn persist_edge(storage: &Storage, cause_id: &str, failure_id: &str, score: f64) -> Result<()> {
    let strength = squash_strength(score);
    if strength < EVIDENCE_FLOOR {
        return Ok(());
    }
    let now = Utc::now();
    // INSERT OR REPLACE on (source_id, target_id) so a rerun can rewrite
    // strength after the squash erratum; skip only if an identical live edge exists.
    let already = storage
        .get_connections_for_memory(cause_id)?
        .iter()
        .any(|c| {
            c.source_id == cause_id
                && c.target_id == failure_id
                && c.link_type == "backfill_candidate"
                && (c.strength - strength).abs() < 1e-12
        });
    if already {
        return Ok(());
    }
    storage.save_connection(&ConnectionRecord {
        source_id: cause_id.to_string(),
        target_id: failure_id.to_string(),
        strength,
        link_type: "backfill_candidate".into(),
        created_at: now,
        last_activated: now,
        activation_count: 0,
    })?;
    Ok(())
}
