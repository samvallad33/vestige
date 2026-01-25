//! Smart Ingest Tool
//!
//! Intelligent memory ingestion with Prediction Error Gating.
//! Automatically decides whether to create, update, or supersede memories
//! based on semantic similarity to existing content.
//!
//! This solves the "bad vs good similar memory" problem by:
//! - Detecting when new content is similar to existing memories
//! - Updating existing memories when appropriate (low prediction error)
//! - Creating new memories when content is substantially different (high PE)
//! - Superseding demoted/outdated memories with better alternatives

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{IngestInput, Storage};

/// Input schema for smart_ingest tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember. Will be compared against existing memories."
            },
            "node_type": {
                "type": "string",
                "description": "Type of knowledge: fact, concept, event, person, place, note, pattern, decision",
                "default": "fact"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tags for categorization"
            },
            "source": {
                "type": "string",
                "description": "Source or reference for this knowledge"
            },
            "forceCreate": {
                "type": "boolean",
                "description": "Force creation of a new memory even if similar content exists",
                "default": false
            }
        },
        "required": ["content"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SmartIngestArgs {
    content: String,
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    source: Option<String>,
    force_create: Option<bool>,
}

pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SmartIngestArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate content
    if args.content.trim().is_empty() {
        return Err("Content cannot be empty".to_string());
    }

    if args.content.len() > 1_000_000 {
        return Err("Content too large (max 1MB)".to_string());
    }

    let input = IngestInput {
        content: args.content,
        node_type: args.node_type.unwrap_or_else(|| "fact".to_string()),
        source: args.source,
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: args.tags.unwrap_or_default(),
        valid_from: None,
        valid_until: None,
    };

    let mut storage = storage.lock().await;

    // Check if force_create is enabled
    if args.force_create.unwrap_or(false) {
        // Use regular ingest
        let node = storage.ingest(input).map_err(|e| e.to_string())?;
        return Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node.id,
            "message": "Memory created (force_create=true)",
            "hasEmbedding": node.has_embedding.unwrap_or(false),
            "predictionError": 1.0,
            "reason": "Forced creation - skipped similarity check"
        }));
    }

    // Use smart ingest with prediction error gating
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let result = storage.smart_ingest(input).map_err(|e| e.to_string())?;

        Ok(serde_json::json!({
            "success": true,
            "decision": result.decision,
            "nodeId": result.node.id,
            "message": format!("Smart ingest complete: {}", result.reason),
            "hasEmbedding": result.node.has_embedding.unwrap_or(false),
            "similarity": result.similarity,
            "predictionError": result.prediction_error,
            "supersededId": result.superseded_id,
            "reason": result.reason,
            "explanation": match result.decision.as_str() {
                "create" => "Created new memory - content was different enough from existing memories",
                "update" => "Updated existing memory - content was similar to an existing memory",
                "reinforce" => "Reinforced existing memory - content was nearly identical",
                "supersede" => "Superseded old memory - new content is an improvement/correction",
                "merge" => "Merged with related memories - content connects multiple topics",
                "replace" => "Replaced existing memory content entirely",
                "add_context" => "Added new content as context to existing memory",
                _ => "Memory processed successfully"
            }
        }))
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        // Fall back to regular ingest if features not available
        let node = storage.ingest(input).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node.id,
            "message": "Memory created (smart ingest requires embeddings feature)",
            "hasEmbedding": false,
            "predictionError": 1.0,
            "reason": "Embeddings not available - used regular ingest"
        }))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Mutex<Storage>>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(Mutex::new(storage)), dir)
    }

    #[tokio::test]
    async fn test_smart_ingest_empty_content_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_smart_ingest_basic_content_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "This is a test fact to remember."
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert!(value["decision"].is_string());
    }

    #[tokio::test]
    async fn test_smart_ingest_force_create() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Force create test content.",
            "forceCreate": true
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["decision"], "create");
        assert!(value["reason"].as_str().unwrap().contains("Forced") ||
                value["reason"].as_str().unwrap().contains("Embeddings not available"));
    }

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["content"].is_object());
        assert!(schema_value["properties"]["forceCreate"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("content")));
    }
}
