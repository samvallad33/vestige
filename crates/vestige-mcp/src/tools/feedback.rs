#![allow(dead_code)]
//! Feedback Tools (Deprecated - use promote_memory/demote_memory instead)
//!
//! Promote and demote memories based on outcome quality.
//! Implements preference learning for Vestige.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

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
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: FeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let storage = storage.lock().await;

    // Get node before for comparison
    let before = storage.get_node(&args.id).map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    let node = storage.promote_memory(&args.id).map_err(|e| e.to_string())?;

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
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: FeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let storage = storage.lock().await;

    // Get node before for comparison
    let before = storage.get_node(&args.id).map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", args.id))?;

    let node = storage.demote_memory(&args.id).map_err(|e| e.to_string())?;

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
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: RequestFeedbackArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Validate UUID
    uuid::Uuid::parse_str(&args.id).map_err(|_| "Invalid node ID format".to_string())?;

    let storage = storage.lock().await;

    let node = storage.get_node(&args.id).map_err(|e| e.to_string())?
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
        "instruction": "PRESENT THESE OPTIONS TO THE USER. If they choose A, call promote_memory. If B, call demote_memory. If C, they will provide a custom instruction - execute it (could be: update the memory content, delete it, merge it, add tags, research something, etc.)."
    }))
}
