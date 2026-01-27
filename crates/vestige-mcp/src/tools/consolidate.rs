#![allow(dead_code)]
//! Consolidation Tool (Deprecated)
//!
//! Run memory consolidation cycle with FSRS decay and embedding generation.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

/// Input schema for run_consolidation tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

pub async fn execute(storage: &Arc<Mutex<Storage>>) -> Result<Value, String> {
    let mut storage = storage.lock().await;
    let result = storage.run_consolidation().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": true,
        "nodesProcessed": result.nodes_processed,
        "nodesPromoted": result.nodes_promoted,
        "nodesPruned": result.nodes_pruned,
        "decayApplied": result.decay_applied,
        "embeddingsGenerated": result.embeddings_generated,
        "durationMs": result.duration_ms,
        "message": format!(
            "Consolidation complete: {} nodes processed, {} embeddings generated, {}ms",
            result.nodes_processed,
            result.embeddings_generated,
            result.duration_ms
        ),
    }))
}
