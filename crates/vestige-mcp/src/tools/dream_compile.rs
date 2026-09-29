//! Dream compile — the review-gated DreamEngine action.
//!
//! Runs the 4-phase DreamEngine (NREM1 triage → NREM3 consolidation → REM
//! creative → Integration) over the top memories by retention strength in a
//! scope, then lands EVERYTHING it found as reviewable Memory PRs
//! (`kind=dream_consolidation`). The only direct writes are
//! `memory_connections` strength deltas: co-hit edges strengthen, weak
//! unreplayed edges take the 0.95 synaptic-downscaling factor. No memory row
//! is ever mutated — sleep proposes, the reviewer disposes.
//!
//! Integrated, not duplicated: the existing `maintain(action="dream")`
//! (MemoryDreamer) stays as-is; this action is the 4-phase engine's surface.

use serde_json::Value;
use std::sync::Arc;

use vestige_core::{DreamCompileConfig, Storage, run_dream_compile};

pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "scope":{"type":"string","default":"user","description":"Namespace to dream over."},
            "memory_count":{"type":"integer","minimum":5,"maximum":500,"default":50,"description":"Top memories by retention strength entering the dream."},
            "max_prs":{"type":"integer","minimum":0,"maximum":50,"default":20,"description":"Review-PR budget; contradictions file first, then REM insights. No memory is written autonomously either way."}
        }
    })
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[derive(serde::Deserialize, Default)]
    struct Args {
        scope: Option<String>,
        memory_count: Option<usize>,
        max_prs: Option<usize>,
    }
    let parsed: Args = serde_json::from_value(args.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|e| e.to_string())?;
    let memory_count = parsed.memory_count.unwrap_or(50);
    let max_prs = parsed.max_prs.unwrap_or(20);
    if !(5..=500).contains(&memory_count) || max_prs > 50 {
        return Err("invalid memory_count (5..=500) or max_prs (0..=50)".into());
    }
    let config = DreamCompileConfig {
        scope: parsed.scope.unwrap_or_else(|| "user".into()),
        memory_count,
        max_prs,
    };

    // Synchronous DB work goes to the blocking pool, matching consolidate.
    let storage = Arc::clone(storage);
    let report = tokio::task::spawn_blocking(move || run_dream_compile(&*storage, &config))
        .await
        .map_err(|e| format!("dream compile task failed: {e}"))??;

    serde_json::to_value(report).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    async fn ingest_n(storage: &Arc<Storage>, n: usize) {
        for i in 0..n {
            storage
                .ingest(vestige_core::IngestInput {
                    content: format!(
                        "Dream compile wire test memory {i} about the deploy pipeline"
                    ),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["dream-compile-wire".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }
    }

    #[test]
    fn schema_is_compact_and_bounded() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert_eq!(s["properties"]["memory_count"]["default"], 50);
        assert_eq!(s["properties"]["memory_count"]["maximum"], 500);
        assert_eq!(s["properties"]["max_prs"]["default"], 20);
    }

    #[tokio::test]
    async fn insufficient_memories_is_a_clean_response() {
        let (storage, _dir) = test_storage().await;
        let value = execute(&storage, None).await.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["memoriesSelected"], 0);
        assert_eq!(value["prsFiled"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn compiles_a_small_store_end_to_end() {
        let (storage, _dir) = test_storage().await;
        ingest_n(&storage, 10).await;

        let value = execute(&storage, None).await.unwrap();
        assert_eq!(value["status"], "compiled");
        assert_eq!(value["memoriesSelected"], 10);
        assert_eq!(value["dreamHistoryRecorded"], true);
        let phases = value["phases"].as_array().unwrap();
        assert_eq!(phases.len(), 4);
        assert_eq!(phases[0]["phase"], "NREM1_Triage");
        assert_eq!(phases[3]["phase"], "Integration");
        assert!(value["edgesStrengthened"].is_number());
        assert!(value["edgesDownscaled"].is_number());
        assert!(value["contradictionsFound"].is_number());
        assert!(value["prsFiled"].is_array());
        // The history row resets the needsDream automation trigger.
        assert!(storage.get_last_dream().unwrap().is_some());
    }

    #[tokio::test]
    async fn rejects_out_of_range_arguments() {
        let (storage, _dir) = test_storage().await;
        for args in [
            serde_json::json!({"memory_count": 4}),
            serde_json::json!({"memory_count": 501}),
            serde_json::json!({"max_prs": 51}),
            serde_json::json!({"scope": " "}),
        ] {
            assert!(execute(&storage, Some(args)).await.is_err());
        }
    }
}
