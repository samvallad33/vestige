#![allow(dead_code)]
//! Recall Tool (Deprecated - use search_unified instead)
//!
//! Search and retrieve knowledge from memory.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{RecallInput, SearchMode, Storage};

/// Input schema for recall tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "min_retention": {
                "type": "number",
                "description": "Minimum retention strength (0.0-1.0, default: 0.0)",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecallArgs {
    query: String,
    limit: Option<i32>,
    min_retention: Option<f64>,
}

pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: RecallArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let input = RecallInput {
        query: args.query.clone(),
        limit: args.limit.unwrap_or(10).clamp(1, 100),
        min_retention: args.min_retention.unwrap_or(0.0).clamp(0.0, 1.0),
        search_mode: SearchMode::Hybrid,
        valid_at: None,
    };

    let storage = storage.lock().await;
    let nodes = storage.recall(input).map_err(|e| e.to_string())?;

    let results: Vec<Value> = nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content,
                "nodeType": n.node_type,
                "retentionStrength": n.retention_strength,
                "stability": n.stability,
                "difficulty": n.difficulty,
                "reps": n.reps,
                "tags": n.tags,
                "source": n.source,
                "createdAt": n.created_at.to_rfc3339(),
                "lastAccessed": n.last_accessed.to_rfc3339(),
                "nextReview": n.next_review.map(|d| d.to_rfc3339()),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "query": args.query,
        "total": results.len(),
        "results": results,
    }))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;
    use tempfile::TempDir;

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Mutex<Storage>>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(Mutex::new(storage)), dir)
    }

    /// Helper to ingest test content
    async fn ingest_test_content(storage: &Arc<Mutex<Storage>>, content: &str) -> String {
        let input = IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec![],
            valid_from: None,
            valid_until: None,
        };
        let mut storage_lock = storage.lock().await;
        let node = storage_lock.ingest(input).unwrap();
        node.id
    }

    // ========================================================================
    // QUERY VALIDATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_recall_empty_query_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "query": "" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_recall_whitespace_only_query_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "query": "   \t\n  " });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_recall_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_recall_missing_query_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "limit": 10 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    // ========================================================================
    // LIMIT CLAMPING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_recall_limit_clamped_to_minimum() {
        let (storage, _dir) = test_storage().await;
        // Ingest some content first
        ingest_test_content(&storage, "Test content for limit clamping").await;

        // Try with limit 0 - should clamp to 1
        let args = serde_json::json!({
            "query": "test",
            "limit": 0
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recall_limit_clamped_to_maximum() {
        let (storage, _dir) = test_storage().await;
        // Ingest some content first
        ingest_test_content(&storage, "Test content for max limit").await;

        // Try with limit 1000 - should clamp to 100
        let args = serde_json::json!({
            "query": "test",
            "limit": 1000
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recall_negative_limit_clamped() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for negative limit").await;

        let args = serde_json::json!({
            "query": "test",
            "limit": -5
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    // ========================================================================
    // MIN_RETENTION CLAMPING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_recall_min_retention_clamped_to_zero() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for retention clamping").await;

        let args = serde_json::json!({
            "query": "test",
            "min_retention": -0.5
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recall_min_retention_clamped_to_one() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for max retention").await;

        let args = serde_json::json!({
            "query": "test",
            "min_retention": 1.5
        });
        let result = execute(&storage, Some(args)).await;
        // Should succeed but return no results (retention > 1.0 clamped to 1.0)
        assert!(result.is_ok());
    }

    // ========================================================================
    // SUCCESSFUL RECALL TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_recall_basic_query_succeeds() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "The Rust programming language is memory safe.").await;

        let args = serde_json::json!({ "query": "rust" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["query"], "rust");
        assert!(value["total"].is_number());
        assert!(value["results"].is_array());
    }

    #[tokio::test]
    async fn test_recall_returns_matching_content() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Python is a dynamic programming language.").await;

        let args = serde_json::json!({ "query": "python" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["id"], node_id);
    }

    #[tokio::test]
    async fn test_recall_with_limit() {
        let (storage, _dir) = test_storage().await;
        // Ingest multiple items
        ingest_test_content(&storage, "Testing content one").await;
        ingest_test_content(&storage, "Testing content two").await;
        ingest_test_content(&storage, "Testing content three").await;

        let args = serde_json::json!({
            "query": "testing",
            "limit": 2
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_recall_empty_database_returns_empty_array() {
        // With hybrid search (keyword + semantic), any query against content
        // may return low-similarity matches. The true "no matches" case
        // is an empty database.
        let (storage, _dir) = test_storage().await;
        // Don't ingest anything - database is empty

        let args = serde_json::json!({ "query": "anything" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["total"], 0);
        assert!(value["results"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_recall_result_contains_expected_fields() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Testing field presence in recall results.").await;

        let args = serde_json::json!({ "query": "testing" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        if !results.is_empty() {
            let first = &results[0];
            assert!(first["id"].is_string());
            assert!(first["content"].is_string());
            assert!(first["nodeType"].is_string());
            assert!(first["retentionStrength"].is_number());
            assert!(first["stability"].is_number());
            assert!(first["difficulty"].is_number());
            assert!(first["reps"].is_number());
            assert!(first["createdAt"].is_string());
            assert!(first["lastAccessed"].is_string());
        }
    }

    // ========================================================================
    // DEFAULT VALUES TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_recall_default_limit_is_10() {
        let (storage, _dir) = test_storage().await;
        // Ingest more than 10 items
        for i in 0..15 {
            ingest_test_content(&storage, &format!("Item number {}", i)).await;
        }

        let args = serde_json::json!({ "query": "item" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results.len() <= 10);
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["query"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("query")));
    }

    #[test]
    fn test_schema_has_optional_fields() {
        let schema_value = schema();
        assert!(schema_value["properties"]["limit"].is_object());
        assert!(schema_value["properties"]["min_retention"].is_object());
    }

    #[test]
    fn test_schema_limit_has_bounds() {
        let schema_value = schema();
        let limit_schema = &schema_value["properties"]["limit"];
        assert_eq!(limit_schema["minimum"], 1);
        assert_eq!(limit_schema["maximum"], 100);
        assert_eq!(limit_schema["default"], 10);
    }

    #[test]
    fn test_schema_min_retention_has_bounds() {
        let schema_value = schema();
        let retention_schema = &schema_value["properties"]["min_retention"];
        assert_eq!(retention_schema["minimum"], 0.0);
        assert_eq!(retention_schema["maximum"], 1.0);
        assert_eq!(retention_schema["default"], 0.0);
    }
}
