//! Unified Memory Tool
//!
//! Merges get_knowledge, delete_knowledge, and get_memory_state into a single
//! `memory` tool with action-based dispatch.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{MemoryState, Storage};

// Accessibility thresholds based on retention strength
const ACCESSIBILITY_ACTIVE: f64 = 0.7;
const ACCESSIBILITY_DORMANT: f64 = 0.4;
const ACCESSIBILITY_SILENT: f64 = 0.1;

/// Compute accessibility score from memory strengths
/// Combines retention, retrieval, and storage strengths
fn compute_accessibility(retention: f64, retrieval: f64, storage: f64) -> f64 {
    // Weighted combination: retention is most important for accessibility
    retention * 0.5 + retrieval * 0.3 + storage * 0.2
}

/// Determine memory state from accessibility score
fn state_from_accessibility(accessibility: f64) -> MemoryState {
    if accessibility >= ACCESSIBILITY_ACTIVE {
        MemoryState::Active
    } else if accessibility >= ACCESSIBILITY_DORMANT {
        MemoryState::Dormant
    } else if accessibility >= ACCESSIBILITY_SILENT {
        MemoryState::Silent
    } else {
        MemoryState::Unavailable
    }
}

/// Input schema for the unified memory tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "delete", "state"],
                "description": "Action to perform: 'get' retrieves full memory node, 'delete' removes memory, 'state' returns accessibility state"
            },
            "id": {
                "type": "string",
                "description": "The ID of the memory node"
            }
        },
        "required": ["action", "id"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MemoryArgs {
    action: String,
    id: String,
}

/// Execute the unified memory tool
pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: MemoryArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID format
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid memory ID format".to_string())?;

    match args.action.as_str() {
        "get" => execute_get(storage, &args.id).await,
        "delete" => execute_delete(storage, &args.id).await,
        "state" => execute_state(storage, &args.id).await,
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: get, delete, state",
            args.action
        )),
    }
}

/// Get full memory node with all metadata
async fn execute_get(storage: &Arc<Mutex<Storage>>, id: &str) -> Result<Value, String> {
    let storage = storage.lock().await;
    let node = storage.get_node(id).map_err(|e| e.to_string())?;

    match node {
        Some(n) => Ok(serde_json::json!({
            "action": "get",
            "found": true,
            "node": {
                "id": n.id,
                "content": n.content,
                "nodeType": n.node_type,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "lastAccessed": n.last_accessed.to_rfc3339(),
                "stability": n.stability,
                "difficulty": n.difficulty,
                "reps": n.reps,
                "lapses": n.lapses,
                "storageStrength": n.storage_strength,
                "retrievalStrength": n.retrieval_strength,
                "retentionStrength": n.retention_strength,
                "sentimentScore": n.sentiment_score,
                "sentimentMagnitude": n.sentiment_magnitude,
                "nextReview": n.next_review.map(|d| d.to_rfc3339()),
                "source": n.source,
                "tags": n.tags,
                "hasEmbedding": n.has_embedding,
                "embeddingModel": n.embedding_model,
            }
        })),
        None => Ok(serde_json::json!({
            "action": "get",
            "found": false,
            "nodeId": id,
            "message": "Memory not found",
        })),
    }
}

/// Delete a memory and return success status
async fn execute_delete(storage: &Arc<Mutex<Storage>>, id: &str) -> Result<Value, String> {
    let mut storage = storage.lock().await;
    let deleted = storage.delete_node(id).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "action": "delete",
        "success": deleted,
        "nodeId": id,
        "message": if deleted { "Memory deleted successfully" } else { "Memory not found" },
    }))
}

/// Get accessibility state of a memory (Active/Dormant/Silent/Unavailable)
async fn execute_state(storage: &Arc<Mutex<Storage>>, id: &str) -> Result<Value, String> {
    let storage = storage.lock().await;

    // Get the memory
    let memory = storage
        .get_node(id)
        .map_err(|e| format!("Error: {}", e))?
        .ok_or("Memory not found")?;

    // Calculate accessibility score
    let accessibility = compute_accessibility(
        memory.retention_strength,
        memory.retrieval_strength,
        memory.storage_strength,
    );

    // Determine state
    let state = state_from_accessibility(accessibility);

    let state_description = match state {
        MemoryState::Active => "Easily retrievable - this memory is fresh and accessible",
        MemoryState::Dormant => "Retrievable with effort - may need cues to recall",
        MemoryState::Silent => "Difficult to retrieve - exists but hard to access",
        MemoryState::Unavailable => "Cannot be retrieved - needs significant reinforcement",
    };

    Ok(serde_json::json!({
        "action": "state",
        "memoryId": id,
        "content": memory.content,
        "state": format!("{:?}", state),
        "accessibility": accessibility,
        "description": state_description,
        "components": {
            "retentionStrength": memory.retention_strength,
            "retrievalStrength": memory.retrieval_strength,
            "storageStrength": memory.storage_strength
        },
        "thresholds": {
            "active": ACCESSIBILITY_ACTIVE,
            "dormant": ACCESSIBILITY_DORMANT,
            "silent": ACCESSIBILITY_SILENT
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessibility_thresholds() {
        // Test Active state
        let accessibility = compute_accessibility(0.9, 0.8, 0.7);
        assert!(accessibility >= ACCESSIBILITY_ACTIVE);
        assert!(matches!(state_from_accessibility(accessibility), MemoryState::Active));

        // Test Dormant state
        let accessibility = compute_accessibility(0.5, 0.5, 0.5);
        assert!(accessibility >= ACCESSIBILITY_DORMANT && accessibility < ACCESSIBILITY_ACTIVE);
        assert!(matches!(state_from_accessibility(accessibility), MemoryState::Dormant));

        // Test Silent state
        let accessibility = compute_accessibility(0.2, 0.2, 0.2);
        assert!(accessibility >= ACCESSIBILITY_SILENT && accessibility < ACCESSIBILITY_DORMANT);
        assert!(matches!(state_from_accessibility(accessibility), MemoryState::Silent));

        // Test Unavailable state
        let accessibility = compute_accessibility(0.05, 0.05, 0.05);
        assert!(accessibility < ACCESSIBILITY_SILENT);
        assert!(matches!(state_from_accessibility(accessibility), MemoryState::Unavailable));
    }

    #[test]
    fn test_schema_structure() {
        let schema = schema();
        assert!(schema["properties"]["action"].is_object());
        assert!(schema["properties"]["id"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["action", "id"]));
    }
}
