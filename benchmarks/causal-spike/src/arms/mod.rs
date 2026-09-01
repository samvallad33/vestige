pub mod backfill;
pub mod causal_graph;
pub mod lexical;

use crate::types::{Arm, QueryResult, TOP_K};
use anyhow::Result;
use vestige_core::Storage;

pub struct ArmRun {
    pub queries: Vec<QueryResult>,
    pub search_mode: String,
    pub accumulation_ms: Option<f64>,
}

pub fn run_arm(arm: Arm, storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    match arm {
        Arm::Lexical => lexical::run(storage, failure_ids),
        Arm::LexicalAnd => lexical::run_and(storage, failure_ids),
        Arm::LexicalOr => lexical::run_or(storage, failure_ids),
        Arm::LexicalEmbed => lexical::run_embed(storage, failure_ids),
        Arm::Backfill => backfill::run(storage, failure_ids),
        Arm::CausalGraph => causal_graph::run(storage, failure_ids),
    }
}

pub fn load_all_nodes(storage: &Storage) -> Result<Vec<vestige_core::KnowledgeNode>> {
    let mut all = Vec::new();
    let page = 500i32;
    let mut offset = 0i32;
    loop {
        let batch = storage.get_all_nodes(page, offset)?;
        if batch.is_empty() {
            break;
        }
        let n = batch.len() as i32;
        all.extend(batch);
        offset += n;
        if n < page {
            break;
        }
    }
    Ok(all)
}

pub fn take_top_k(mut ranked: Vec<(String, f64)>) -> (Vec<String>, Vec<f64>) {
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(TOP_K);
    let ids = ranked.iter().map(|(id, _)| id.clone()).collect();
    let scores = ranked.iter().map(|(_, s)| *s).collect();
    (ids, scores)
}
