#![allow(dead_code)]
//! Knowledge Tools (Deprecated - use memory_unified instead)
//!
//! Get and delete specific knowledge nodes.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

/// Input schema for get_knowledge tool
pub fn get_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the knowledge node to retrieve"
            }
        },
        "required": ["id"]
    })
}

/// Input schema for delete_knowledge tool
pub fn delete_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the knowledge node to delete"
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KnowledgeArgs {
    id: String,
}

pub async fn execute_get(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: KnowledgeArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let storage = storage.lock().await;
    let node = storage.get_node(&args.id).map_err(|e| e.to_string())?;

    match node {
        Some(n) => Ok(serde_json::json!({
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
            "found": false,
            "nodeId": args.id,
            "message": "Node not found",
        })),
    }
}

pub async fn execute_delete(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: KnowledgeArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let mut storage = storage.lock().await;
    let deleted = storage.delete_node(&args.id).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": deleted,
        "nodeId": args.id,
        "message": if deleted { "Node deleted successfully" } else { "Node not found" },
    }))
}
