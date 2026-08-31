//! `suppress` MCP Tool (v2.0.5) — Top-Down Active Forgetting
//!
//! Actively suppress a memory via top-down inhibitory control. Distinct from
//! `memory.delete` (which removes the row) and `memory.demote` (which is a
//! one-shot thumb-down). Each call compounds: suppression_count increments,
//! FSRS state is dealt a strong blow, and a background Rac1 cascade worker
//! (in the existing consolidation loop) will fade co-activated neighbors.
//!
//! Reversible within a 24-hour labile window via `reverse: true`.
//!
//! References:
//! - Anderson et al. (2025). Brain mechanisms underlying the inhibitory
//!   control of thought. Nat Rev Neurosci. DOI 10.1038/s41583-025-00929-y
//! - Cervantes-Sandoval & Davis (2020). Rac1 Impairs Forgetting-Induced
//!   Cellular Plasticity. Front Cell Neurosci. PMC7477079

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::Storage;
use vestige_core::neuroscience::active_forgetting::{ActiveForgettingSystem, DEFAULT_LABILE_HOURS};

/// Input schema for the `suppress` tool.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Actively suppress a memory via top-down inhibitory control, following the compounding Suppression-Induced Forgetting effect in Anderson et al. 2025. Distinct from delete: the memory persists but is inhibited from retrieval and actively decays. Each call compounds suppression strength. A background worker also spreads accelerated decay to co-activated neighbors over the next 72 hours (our own design, named after Rac1 by analogy). Reversible within 24 hours via reverse=true.",
        "properties": {
            "id": {
                "type": "string",
                "description": "Memory UUID to suppress (or reverse-suppress)"
            },
            "reason": {
                "type": "string",
                "description": "Optional free-form note explaining why this memory is being suppressed. Logged for audit."
            },
            "reverse": {
                "type": "boolean",
                "default": false,
                "description": "If true, reverse a previous suppression. Only works within the 24-hour labile window."
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SuppressArgs {
    id: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    reverse: bool,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: SuppressArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.id.trim().is_empty() {
        return Err("'id' must not be empty".to_string());
    }
    // Basic UUID sanity check — don't reject if missing, but warn
    if uuid::Uuid::parse_str(&args.id).is_err() {
        return Err(format!("Invalid memory ID format: {}", args.id));
    }

    let sys = ActiveForgettingSystem::new();

    if args.reverse {
        // Reverse path — only allowed within labile window.
        match storage.reverse_suppression(&args.id, sys.labile_hours) {
            Ok(node) => {
                let still_suppressed = node.suppression_count > 0;
                Ok(json!({
                    "success": true,
                    "action": "reverse",
                    "id": args.id,
                    "suppressionCount": node.suppression_count,
                    "stillSuppressed": still_suppressed,
                    "retentionStrength": node.retention_strength,
                    "retrievalStrength": node.retrieval_strength,
                    "stability": node.stability,
                    "message": if still_suppressed {
                        format!(
                            "Reversal applied. {} suppression(s) remain on this memory.",
                            node.suppression_count
                        )
                    } else {
                        "Suppression fully reversed. Memory is no longer inhibited.".to_string()
                    },
                }))
            }
            Err(e) => Err(format!("Reverse failed: {}", e)),
        }
    } else {
        // Forward path — suppress + log reason + tell the user what will happen.
        let before_count = storage
            .get_node(&args.id)
            .map_err(|e| format!("Failed to load memory: {}", e))?
            .map(|n| n.suppression_count)
            .unwrap_or(0);

        let node = storage
            .suppress_memory(&args.id)
            .map_err(|e| format!("Suppress failed: {}", e))?;

        // Count how many neighbors will be cascaded over the coming 72h.
        // We don't run the cascade synchronously — it happens in the
        // background consolidation loop via `run_rac1_cascade_sweep`. But we
        // can give the user an estimate.
        let edges = storage
            .get_connections_for_memory(&args.id)
            .unwrap_or_default();
        let estimated_cascade = edges.len().min(100);

        let reversible_until = node
            .suppressed_at
            .map(|t| sys.reversible_until(t))
            .unwrap_or_else(chrono::Utc::now);
        let retrieval_penalty = sys.retrieval_penalty(node.suppression_count);

        tracing::info!(
            id = %args.id,
            count = node.suppression_count,
            reason = args.reason.as_deref().unwrap_or(""),
            "Memory suppressed"
        );

        Ok(json!({
            "success": true,
            "action": "suppress",
            "id": args.id,
            "suppressionCount": node.suppression_count,
            "priorCount": before_count,
            "retrievalPenalty": retrieval_penalty,
            "retentionStrength": node.retention_strength,
            "retrievalStrength": node.retrieval_strength,
            "stability": node.stability,
            "estimatedCascadeNeighbors": estimated_cascade,
            "reversibleUntil": reversible_until.to_rfc3339(),
            "labileWindowHours": DEFAULT_LABILE_HOURS,
            "reason": args.reason,
            "message": format!(
                "Actively forgetting. Suppression #{} applied. ~{} co-activated neighbors will fade over the next 72h via Rac1 cascade. Reversible for {}h.",
                node.suppression_count, estimated_cascade, DEFAULT_LABILE_HOURS
            ),
            "citation": "Anderson et al. 2025, Nat Rev Neurosci, DOI: 10.1038/s41583-025-00929-y"
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    fn ingest(storage: &Storage, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["test".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    #[test]
    fn test_schema_is_valid() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["id"].is_object());
        assert!(s["properties"]["reverse"].is_object());
        assert_eq!(s["required"][0], "id");
    }

    #[tokio::test]
    async fn test_suppress_missing_args() {
        let (storage, _dir) = test_storage();
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_suppress_invalid_uuid() {
        let (storage, _dir) = test_storage();
        let args = json!({"id": "not-a-uuid"});
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory ID"));
    }

    #[tokio::test]
    async fn test_suppress_increments_count() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake is my roommate");

        // First call
        let r1 = execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        assert_eq!(r1["suppressionCount"], 1);
        assert_eq!(r1["priorCount"], 0);

        // Second call — compounds
        let r2 = execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        assert_eq!(r2["suppressionCount"], 2);
        assert_eq!(r2["priorCount"], 1);
    }

    #[tokio::test]
    async fn test_suppress_applies_fsrs_penalty() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        let before = storage.get_node(&id).unwrap().unwrap();
        let result = execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Stability should be heavily reduced
        let after_stability = result["stability"].as_f64().unwrap();
        assert!(after_stability < before.stability);
        // Retention should be reduced
        let after_retention = result["retentionStrength"].as_f64().unwrap();
        assert!(after_retention < before.retention_strength);
    }

    #[tokio::test]
    async fn test_suppress_is_not_delete() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Memory must still be retrievable via get_node
        let node = storage.get_node(&id).unwrap();
        assert!(node.is_some(), "Suppressed memory must still exist");
        assert_eq!(node.unwrap().suppression_count, 1);
    }

    #[tokio::test]
    async fn test_reverse_within_window_decrements() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Now reverse — count should drop from 2 to 1
        let r = execute(&storage, Some(json!({"id": id.clone(), "reverse": true})))
            .await
            .unwrap();
        assert_eq!(r["suppressionCount"], 1);
        assert_eq!(r["stillSuppressed"], true);

        // Reverse again — should go to 0
        let r = execute(&storage, Some(json!({"id": id.clone(), "reverse": true})))
            .await
            .unwrap();
        assert_eq!(r["suppressionCount"], 0);
        assert_eq!(r["stillSuppressed"], false);
    }

    #[tokio::test]
    async fn test_reverse_without_prior_suppression_fails() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Fresh memory");

        let result = execute(&storage, Some(json!({"id": id.clone(), "reverse": true}))).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no active suppression"));
    }

    #[tokio::test]
    async fn test_suppress_records_timestamp() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        execute(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        let node = storage.get_node(&id).unwrap().unwrap();
        assert!(node.suppressed_at.is_some(), "suppressed_at must be set");
    }
}
