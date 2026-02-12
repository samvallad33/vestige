//! Ingest Tool
//!
//! Add new knowledge to memory.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{IngestInput, Storage};

/// Input schema for ingest tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember"
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
            }
        },
        "required": ["content"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IngestArgs {
    content: String,
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    source: Option<String>,
}

pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: IngestArgs = match args {
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

    // Route through smart_ingest when embeddings are available to prevent duplicates.
    // Falls back to raw ingest only when embeddings aren't ready.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let fallback_input = input.clone();
        match storage.smart_ingest(input) {
            Ok(result) => {
                return Ok(serde_json::json!({
                    "success": true,
                    "nodeId": result.node.id,
                    "decision": result.decision,
                    "message": format!("Knowledge ingested successfully. Node ID: {} ({})", result.node.id, result.decision),
                    "hasEmbedding": result.node.has_embedding.unwrap_or(false),
                    "similarity": result.similarity,
                    "reason": result.reason,
                }));
            }
            Err(_) => {
                // smart_ingest failed — fall through to raw ingest with cloned input
                let node = storage.ingest(fallback_input).map_err(|e| e.to_string())?;
                return Ok(serde_json::json!({
                    "success": true,
                    "nodeId": node.id,
                    "decision": "create",
                    "message": format!("Knowledge ingested successfully. Node ID: {}", node.id),
                    "hasEmbedding": node.has_embedding.unwrap_or(false),
                }));
            }
        }
    }

    // Fallback for builds without embedding features
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let node = storage.ingest(input).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "success": true,
            "nodeId": node.id,
            "decision": "create",
            "message": format!("Knowledge ingested successfully. Node ID: {}", node.id),
            "hasEmbedding": node.has_embedding.unwrap_or(false),
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

    // ========================================================================
    // INPUT VALIDATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_ingest_empty_content_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_ingest_whitespace_only_content_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "   \n\t  " });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_ingest_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_ingest_missing_content_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "node_type": "fact" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    // ========================================================================
    // LARGE CONTENT TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_ingest_large_content_fails() {
        let (storage, _dir) = test_storage().await;
        // Create content larger than 1MB
        let large_content = "x".repeat(1_000_001);
        let args = serde_json::json!({ "content": large_content });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[tokio::test]
    async fn test_ingest_exactly_1mb_succeeds() {
        let (storage, _dir) = test_storage().await;
        // Create content exactly 1MB
        let exact_content = "x".repeat(1_000_000);
        let args = serde_json::json!({ "content": exact_content });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    // ========================================================================
    // SUCCESSFUL INGEST TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_ingest_basic_content_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "This is a test fact to remember."
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert!(value["message"].as_str().unwrap().contains("successfully"));
    }

    #[tokio::test]
    async fn test_ingest_with_node_type() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Error handling should use Result<T, E> pattern.",
            "node_type": "pattern"
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_ingest_with_tags() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "The Rust programming language emphasizes safety.",
            "tags": ["rust", "programming", "safety"]
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_ingest_with_source() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "MCP protocol version 2024-11-05 is the current standard.",
            "source": "https://modelcontextprotocol.io/spec"
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_ingest_with_all_optional_fields() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Complex memory with all metadata.",
            "node_type": "decision",
            "tags": ["architecture", "design"],
            "source": "team meeting notes"
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
    }

    // ========================================================================
    // NODE TYPE DEFAULTS
    // ========================================================================

    #[tokio::test]
    async fn test_ingest_default_node_type_is_fact() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Default type test content."
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        // Verify node was created - the default type is "fact"
        let node_id = result.unwrap()["nodeId"].as_str().unwrap().to_string();
        let storage_lock = storage.lock().await;
        let node = storage_lock.get_node(&node_id).unwrap().unwrap();
        assert_eq!(node.node_type, "fact");
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["content"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("content")));
    }

    #[test]
    fn test_schema_has_optional_fields() {
        let schema_value = schema();
        assert!(schema_value["properties"]["node_type"].is_object());
        assert!(schema_value["properties"]["tags"].is_object());
        assert!(schema_value["properties"]["source"].is_object());
    }
}
