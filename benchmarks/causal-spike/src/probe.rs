//! S3 live-store copy probe. Caller copies the live DB first; this command
//! only opens the path it is given.

use crate::arms;
use crate::arms::backfill::{engine, rank_backfill};
use crate::types::TOP_K;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;
use vestige_core::Storage;

#[derive(Serialize)]
pub(crate) struct ProbeOut {
    store: String,
    failure_id: String,
    looks_like_failure: bool,
    embedding_ready: bool,
    backfill_top: Vec<(String, f64)>,
    causal_graph_top: Vec<(String, f64)>,
    graph_note: String,
}

pub fn probe(store: &Path, failure_id: &str) -> Result<ProbeOut> {
    let storage = Storage::new(Some(store.to_path_buf()))
        .with_context(|| format!("open probe store {store:?} (must be a copy)"))?;
    let node = storage
        .get_node(failure_id)?
        .ok_or_else(|| anyhow::anyhow!("id {failure_id} not in store"))?;
    let looks =
        vestige_core::advanced::retroactive_backfill::looks_like_failure(&node.content, &node.tags);
    let all = arms::load_all_nodes(&storage)?;
    let bf = rank_backfill(&storage, &engine(), &node, &all, false)?;
    Ok(ProbeOut {
        store: store.display().to_string(),
        failure_id: failure_id.to_string(),
        looks_like_failure: looks,
        embedding_ready: storage.is_embedding_ready(),
        backfill_top: bf.into_iter().take(TOP_K).collect(),
        causal_graph_top: Vec::new(),
        graph_note: "S3 probe ranks per-query backfill only; Arm C accumulate on a 2.8k-node live copy is not this measurement".into(),
    })
}
