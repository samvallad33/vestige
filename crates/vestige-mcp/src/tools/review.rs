#![allow(dead_code)]
//! Review Tool (Deprecated)
//!
//! Mark memories as reviewed using FSRS-6 algorithm.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{Rating, Storage};

/// Input schema for mark_reviewed tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the memory to review"
            },
            "rating": {
                "type": "integer",
                "description": "Review rating: 1=Again (forgot), 2=Hard, 3=Good, 4=Easy",
                "minimum": 1,
                "maximum": 4,
                "default": 3
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReviewArgs {
    id: String,
    rating: Option<i32>,
}

pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: ReviewArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let rating_value = args.rating.unwrap_or(3);
    if !(1..=4).contains(&rating_value) {
        return Err("Rating must be between 1 and 4".to_string());
    }

    let rating = Rating::from_i32(rating_value)
        .ok_or_else(|| "Invalid rating value".to_string())?;

    let mut storage = storage.lock().await;

    // Get node before review for comparison
    let before = storage.get_node(&args.id).map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    let node = storage.mark_reviewed(&args.id, rating).map_err(|e| e.to_string())?;

    let rating_name = match rating {
        Rating::Again => "Again",
        Rating::Hard => "Hard",
        Rating::Good => "Good",
        Rating::Easy => "Easy",
    };

    Ok(serde_json::json!({
        "success": true,
        "nodeId": node.id,
        "rating": rating_name,
        "fsrs": {
            "previousRetention": before.retention_strength,
            "newRetention": node.retention_strength,
            "previousStability": before.stability,
            "newStability": node.stability,
            "difficulty": node.difficulty,
            "reps": node.reps,
            "lapses": node.lapses,
        },
        "nextReview": node.next_review.map(|d| d.to_rfc3339()),
        "message": format!("Memory reviewed with rating '{}'. Retention: {:.2} -> {:.2}",
            rating_name, before.retention_strength, node.retention_strength),
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

    /// Helper to ingest test content and return node ID
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
    // RATING VALIDATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_rating_zero_fails() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for rating validation").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 0
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("between 1 and 4"));
    }

    #[tokio::test]
    async fn test_review_rating_five_fails() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for high rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 5
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("between 1 and 4"));
    }

    #[tokio::test]
    async fn test_review_rating_negative_fails() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for negative rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": -1
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("between 1 and 4"));
    }

    #[tokio::test]
    async fn test_review_rating_very_high_fails() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for very high rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 100
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("between 1 and 4"));
    }

    // ========================================================================
    // VALID RATINGS TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_rating_again_succeeds() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for Again rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 1
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["rating"], "Again");
    }

    #[tokio::test]
    async fn test_review_rating_hard_succeeds() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for Hard rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 2
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["rating"], "Hard");
    }

    #[tokio::test]
    async fn test_review_rating_good_succeeds() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for Good rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["rating"], "Good");
    }

    #[tokio::test]
    async fn test_review_rating_easy_succeeds() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for Easy rating").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 4
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["rating"], "Easy");
    }

    // ========================================================================
    // NODE ID VALIDATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;

        let args = serde_json::json!({
            "id": "not-a-valid-uuid",
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid node ID"));
    }

    #[tokio::test]
    async fn test_review_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let fake_uuid = uuid::Uuid::new_v4().to_string();

        let args = serde_json::json!({
            "id": fake_uuid,
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_review_missing_id_fails() {
        let (storage, _dir) = test_storage().await;

        let args = serde_json::json!({
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    #[tokio::test]
    async fn test_review_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    // ========================================================================
    // FSRS UPDATE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_updates_reps_counter() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for reps counter").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["fsrs"]["reps"], 1);
    }

    #[tokio::test]
    async fn test_review_multiple_times_increases_reps() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for multiple reviews").await;

        // Review first time
        let args = serde_json::json!({ "id": node_id, "rating": 3 });
        execute(&storage, Some(args)).await.unwrap();

        // Review second time
        let args = serde_json::json!({ "id": node_id, "rating": 3 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["fsrs"]["reps"], 2);
    }

    #[tokio::test]
    async fn test_same_day_again_does_not_count_as_lapse() {
        // FSRS-6 treats same-day reviews differently - they don't increment lapses.
        // This is by design: same-day reviews indicate the user is still learning,
        // not that they've forgotten and need to re-learn (which is what lapses track).
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for lapses").await;

        // First review to get out of new state
        let args = serde_json::json!({ "id": node_id, "rating": 3 });
        execute(&storage, Some(args)).await.unwrap();

        // Immediate "Again" rating (same-day) should NOT count as a lapse
        let args = serde_json::json!({ "id": node_id, "rating": 1 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        // Same-day reviews preserve lapse count per FSRS-6 algorithm
        assert_eq!(value["fsrs"]["lapses"].as_i64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_review_returns_next_review_date() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for next review").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["nextReview"].is_string());
    }

    // ========================================================================
    // DEFAULT RATING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_default_rating_is_good() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for default rating").await;

        // Omit rating, should default to 3 (Good)
        let args = serde_json::json!({
            "id": node_id
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["rating"], "Good");
    }

    // ========================================================================
    // RESPONSE FORMAT TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_review_response_contains_expected_fields() {
        let (storage, _dir) = test_storage().await;
        let node_id = ingest_test_content(&storage, "Test content for response format").await;

        let args = serde_json::json!({
            "id": node_id,
            "rating": 3
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert!(value["rating"].is_string());
        assert!(value["fsrs"].is_object());
        assert!(value["fsrs"]["previousRetention"].is_number());
        assert!(value["fsrs"]["newRetention"].is_number());
        assert!(value["fsrs"]["previousStability"].is_number());
        assert!(value["fsrs"]["newStability"].is_number());
        assert!(value["fsrs"]["difficulty"].is_number());
        assert!(value["fsrs"]["reps"].is_number());
        assert!(value["fsrs"]["lapses"].is_number());
        assert!(value["message"].is_string());
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["id"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("id")));
    }

    #[test]
    fn test_schema_rating_has_bounds() {
        let schema_value = schema();
        let rating_schema = &schema_value["properties"]["rating"];
        assert_eq!(rating_schema["minimum"], 1);
        assert_eq!(rating_schema["maximum"], 4);
        assert_eq!(rating_schema["default"], 3);
    }
}
