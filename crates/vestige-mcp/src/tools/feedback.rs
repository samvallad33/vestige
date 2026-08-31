//! Feedback Tools
//!
//! Promote and demote memories based on outcome quality.
//! Implements preference learning for Vestige.
//!
//! v1.5.0: Enhanced with cognitive pipeline:
//!   - Reward signal recording (4-channel importance)
//!   - Importance tracking (retrieval outcome)
//!   - Reconsolidation modification (labile window boost)
//!   - Activation network reinforcement

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{Modification, OutcomeType, Storage};

/// Input schema for promote_memory tool
pub fn promote_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the memory to promote"
            },
            "reason": {
                "type": "string",
                "description": "Why this memory was helpful (optional, for logging)"
            }
        },
        "required": ["id"]
    })
}

/// Input schema for demote_memory tool
pub fn demote_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the memory to demote"
            },
            "reason": {
                "type": "string",
                "description": "Why this memory was unhelpful or wrong (optional, for logging)"
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize)]
struct FeedbackArgs {
    id: String,
    reason: Option<String>,
}

/// Promote a memory (thumbs up) - it led to a good outcome
pub async fn execute_promote(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: FeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    // Get node before for comparison
    let before = storage
        .get_node(&args.id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    let node = storage
        .promote_memory(&args.id)
        .map_err(|e| e.to_string())?;

    // ====================================================================
    // COGNITIVE FEEDBACK PIPELINE (promote)
    // ====================================================================
    if let Ok(mut cog) = cognitive.try_lock() {
        // 5A. Reward signal — record positive outcome
        cog.reward_signal
            .record_outcome(&args.id, OutcomeType::Helpful);

        // 5B. Importance tracking — mark as helpful retrieval
        cog.importance_tracker.on_retrieved(&args.id, true);

        // 5C. Reconsolidation — boost retrieval if memory is labile
        if cog.reconsolidation.is_labile(&args.id) {
            cog.reconsolidation.apply_modification(
                &args.id,
                Modification::StrengthenConnection {
                    target_memory_id: args.id.clone(),
                    boost: 0.2,
                },
            );
        }
    }

    Ok(serde_json::json!({
        "success": true,
        "action": "promoted",
        "nodeId": node.id,
        "reason": args.reason,
        "changes": {
            "retrievalStrength": {
                "before": before.retrieval_strength,
                "after": node.retrieval_strength,
                "delta": "+0.20"
            },
            "retentionStrength": {
                "before": before.retention_strength,
                "after": node.retention_strength,
                "delta": "+0.10"
            },
            "stability": {
                "before": before.stability,
                "after": node.stability,
                "multiplier": "1.5x"
            }
        },
        "message": format!("Memory promoted. It will now surface more often in searches. Retrieval: {:.2} -> {:.2}",
            before.retrieval_strength, node.retrieval_strength),
    }))
}

/// Demote a memory (thumbs down) - it led to a bad outcome
pub async fn execute_demote(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: FeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    // Get node before for comparison
    let before = storage
        .get_node(&args.id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    let node = storage.demote_memory(&args.id).map_err(|e| e.to_string())?;

    // ====================================================================
    // COGNITIVE FEEDBACK PIPELINE (demote)
    // ====================================================================
    if let Ok(mut cog) = cognitive.try_lock() {
        // 5A. Reward signal — record negative outcome
        cog.reward_signal
            .record_outcome(&args.id, OutcomeType::NotHelpful);

        // 5B. Importance tracking — mark as unhelpful retrieval
        cog.importance_tracker.on_retrieved(&args.id, false);

        // 5C. Reconsolidation — weaken if memory is labile
        if cog.reconsolidation.is_labile(&args.id) {
            cog.reconsolidation.apply_modification(
                &args.id,
                Modification::AddContext {
                    context: "User reported this memory was wrong/unhelpful".to_string(),
                },
            );
        }
    }

    Ok(serde_json::json!({
        "success": true,
        "action": "demoted",
        "nodeId": node.id,
        "reason": args.reason,
        "changes": {
            "retrievalStrength": {
                "before": before.retrieval_strength,
                "after": node.retrieval_strength,
                "delta": "-0.30"
            },
            "retentionStrength": {
                "before": before.retention_strength,
                "after": node.retention_strength,
                "delta": "-0.15"
            },
            "stability": {
                "before": before.stability,
                "after": node.stability,
                "multiplier": "0.5x"
            }
        },
        "message": format!("Memory demoted. Better alternatives will now surface instead. Retrieval: {:.2} -> {:.2}",
            before.retrieval_strength, node.retrieval_strength),
        "note": "Memory is NOT deleted - it remains searchable but ranks lower."
    }))
}

/// Input schema for request_feedback tool
pub fn request_feedback_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the memory to request feedback on"
            },
            "context": {
                "type": "string",
                "description": "What the memory was used for (e.g., 'error handling advice')"
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize)]
struct RequestFeedbackArgs {
    id: String,
    context: Option<String>,
}

/// Request feedback from the user about a memory's usefulness
/// Returns a structured prompt for Claude to ask the user
pub async fn execute_request_feedback(
    storage: &Arc<Storage>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: RequestFeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let node = storage
        .get_node(&args.id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    // Truncate content for display
    let preview: String = node.content.chars().take(100).collect();
    let preview = if node.content.len() > 100 {
        format!("{}...", preview)
    } else {
        preview
    };

    Ok(serde_json::json!({
        "action": "request_feedback",
        "nodeId": node.id,
        "memoryPreview": preview,
        "context": args.context,
        "prompt": "Was this memory helpful?",
        "options": [
            {
                "key": "A",
                "label": "Yes, helpful",
                "action": "promote",
                "description": "Memory will surface more often"
            },
            {
                "key": "B",
                "label": "No, wrong/outdated",
                "action": "demote",
                "description": "Better alternatives will surface instead"
            },
            {
                "key": "C",
                "label": "Ask Claude...",
                "action": "custom",
                "description": "Give Claude a custom instruction (e.g., 'update this memory', 'merge with X', 'add tag Y')"
            }
        ],
        "instruction": "PRESENT THESE OPTIONS TO THE USER. If they choose A, call memory(action='promote'). If B, call memory(action='demote'). If C, they will provide a custom instruction - execute it (could be: update the memory content, delete it, merge it, add tags, research something, etc.)."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_test_memory(storage: &Arc<Storage>) -> String {
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Test memory for feedback".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        node.id
    }

    // === PROMOTE SCHEMA ===

    #[test]
    fn test_promote_schema_has_required_fields() {
        let schema = promote_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["id"].is_object());
        assert!(schema["properties"]["reason"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("id"))
        );
    }

    #[test]
    fn test_demote_schema_has_required_fields() {
        let schema = demote_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["id"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("id"))
        );
    }

    #[test]
    fn test_request_feedback_schema_has_required_fields() {
        let schema = request_feedback_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["id"].is_object());
        assert!(schema["properties"]["context"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("id"))
        );
    }

    // === PROMOTE TESTS ===

    #[tokio::test]
    async fn test_promote_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_promote(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_promote_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "not-a-uuid" });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid node ID format"));
    }

    #[tokio::test]
    async fn test_promote_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Node not found"));
    }

    #[tokio::test]
    async fn test_promote_missing_id_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "reason": "test" });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    #[tokio::test]
    async fn test_promote_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id, "reason": "It was helpful" });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "promoted");
        assert_eq!(value["nodeId"], id);
        assert_eq!(value["reason"], "It was helpful");
        assert!(value["changes"]["retrievalStrength"].is_object());
    }

    #[tokio::test]
    async fn test_promote_without_reason_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["reason"].is_null());
    }

    #[tokio::test]
    async fn test_promote_changes_contain_expected_fields() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id });
        let result = execute_promote(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        // Verify response structure includes before/after/delta for all 3 metrics
        assert!(value["changes"]["retrievalStrength"]["before"].is_number());
        assert!(value["changes"]["retrievalStrength"]["after"].is_number());
        assert_eq!(value["changes"]["retrievalStrength"]["delta"], "+0.20");
        assert!(value["changes"]["retentionStrength"]["before"].is_number());
        assert!(value["changes"]["retentionStrength"]["after"].is_number());
        assert_eq!(value["changes"]["retentionStrength"]["delta"], "+0.10");
        assert!(value["changes"]["stability"]["before"].is_number());
        assert!(value["changes"]["stability"]["after"].is_number());
        assert_eq!(value["changes"]["stability"]["multiplier"], "1.5x");
    }

    // === DEMOTE TESTS ===

    #[tokio::test]
    async fn test_demote_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_demote(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_demote_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "bad-id" });
        let result = execute_demote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid node ID format"));
    }

    #[tokio::test]
    async fn test_demote_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute_demote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Node not found"));
    }

    #[tokio::test]
    async fn test_demote_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id, "reason": "It was wrong" });
        let result = execute_demote(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "demoted");
        assert_eq!(value["nodeId"], id);
        assert_eq!(value["reason"], "It was wrong");
        assert!(value["note"].as_str().unwrap().contains("NOT deleted"));
    }

    #[tokio::test]
    async fn test_demote_changes_contain_expected_fields() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id });
        let result = execute_demote(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        assert!(value["changes"]["retrievalStrength"]["before"].is_number());
        assert!(value["changes"]["retrievalStrength"]["after"].is_number());
        assert_eq!(value["changes"]["retrievalStrength"]["delta"], "-0.30");
        assert_eq!(value["changes"]["retentionStrength"]["delta"], "-0.15");
        assert_eq!(value["changes"]["stability"]["multiplier"], "0.5x");
    }

    // === REQUEST FEEDBACK TESTS ===

    #[tokio::test]
    async fn test_request_feedback_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_request_feedback(&storage, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_feedback_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "not-valid" });
        let result = execute_request_feedback(&storage, Some(args)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_feedback_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute_request_feedback(&storage, Some(args)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_feedback_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "id": id, "context": "debugging" });
        let result = execute_request_feedback(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "request_feedback");
        assert_eq!(value["nodeId"], id);
        assert!(value["memoryPreview"].is_string());
        assert!(value["options"].is_array());
        assert_eq!(value["options"].as_array().unwrap().len(), 3);
        assert_eq!(value["context"], "debugging");
    }

    #[tokio::test]
    async fn test_request_feedback_truncates_long_content() {
        let (storage, _dir) = test_storage().await;
        let long_content = "A".repeat(200);
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: long_content,
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        let node_id = node.id.clone();

        let args = serde_json::json!({ "id": node_id });
        let result = execute_request_feedback(&storage, Some(args)).await;
        let value = result.unwrap();
        let preview = value["memoryPreview"].as_str().unwrap();
        assert!(preview.ends_with("..."));
        assert!(preview.len() <= 103);
    }
}
