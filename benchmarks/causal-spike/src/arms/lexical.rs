use super::{ArmRun, take_top_k};
use crate::types::{QueryResult, TOP_K};
use anyhow::Result;
use std::time::Instant;
use vestige_core::{MatchType, Storage};

pub fn run(storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    run_inner(storage, failure_ids, SearchKind::Hybrid)
}

pub fn run_or(storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    run_inner(storage, failure_ids, SearchKind::OrBm25)
}

enum SearchKind {
    Hybrid,
    OrBm25,
}

fn run_inner(storage: &Storage, failure_ids: &[String], kind: SearchKind) -> Result<ArmRun> {
    let mut queries = Vec::with_capacity(failure_ids.len());
    let mut saw_semantic = false;

    for failure_id in failure_ids {
        let t0 = Instant::now();
        let node = storage
            .get_node(failure_id)?
            .ok_or_else(|| anyhow::anyhow!("failure {failure_id} not in store"))?;
        let ranked = match kind {
            SearchKind::Hybrid => {
                let hits = storage.hybrid_search(&node.content, (TOP_K + 1) as i32, 0.5, 0.5)?;
                for hit in &hits {
                    if hit.semantic_score.is_some()
                        || matches!(hit.match_type, MatchType::Semantic | MatchType::Both)
                    {
                        saw_semantic = true;
                    }
                }
                hits.into_iter()
                    .filter(|h| h.node.id != *failure_id)
                    .map(|h| (h.node.id, h.combined_score as f64))
                    .collect()
            }
            SearchKind::OrBm25 => storage
                .search(&node.content, (TOP_K + 1) as i32)?
                .into_iter()
                .filter(|n| n.id != *failure_id)
                .enumerate()
                .map(|(i, n)| (n.id, 1.0 / (i as f64 + 1.0)))
                .collect(),
        };
        let (ranked_ids, scores) = take_top_k(ranked);
        queries.push(QueryResult {
            failure_id: failure_id.clone(),
            ranked_ids,
            scores,
            wall_clock_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let search_mode = match kind {
        SearchKind::OrBm25 => "fts_or_bm25".into(),
        SearchKind::Hybrid if saw_semantic => "hybrid_embeddings".into(),
        SearchKind::Hybrid => "fts_keyword_and".into(),
    };

    Ok(ArmRun {
        queries,
        search_mode,
        accumulation_ms: None,
    })
}
