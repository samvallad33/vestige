#![allow(dead_code)]
//! Memory States Tool (Deprecated - use memory_unified instead)
//!
//! Query and manage memory states (Active, Dormant, Silent, Unavailable).
//! Based on accessibility continuum theory.

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

/// Input schema for get_memory_state tool
pub fn get_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to check state for"
            }
        },
        "required": ["memory_id"]
    })
}

/// Input schema for list_by_state tool
pub fn list_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["active", "dormant", "silent", "unavailable"],
                "description": "Filter memories by state"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 20)"
            }
        },
        "required": []
    })
}

/// Input schema for state_stats tool
pub fn stats_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

/// Get the cognitive state of a specific memory
pub async fn execute_get(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.ok_or("Missing arguments")?;

    let memory_id = args["memory_id"]
        .as_str()
        .ok_or("memory_id is required")?;

    let storage = storage.lock().await;

    // Get the memory
    let memory = storage.get_node(memory_id)
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
        "memoryId": memory_id,
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

/// List memories by state
pub async fn execute_list(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.unwrap_or(serde_json::json!({}));

    let state_filter = args["state"].as_str();
    let limit = args["limit"].as_i64().unwrap_or(20) as usize;

    let storage = storage.lock().await;

    // Get all memories
    let memories = storage.get_all_nodes(500, 0)
        .map_err(|e| e.to_string())?;

    // Categorize by state
    let mut active = Vec::new();
    let mut dormant = Vec::new();
    let mut silent = Vec::new();
    let mut unavailable = Vec::new();

    for memory in memories {
        let accessibility = compute_accessibility(
            memory.retention_strength,
            memory.retrieval_strength,
            memory.storage_strength,
        );

        let entry = serde_json::json!({
            "id": memory.id,
            "content": memory.content,
            "accessibility": accessibility,
            "retentionStrength": memory.retention_strength
        });

        let state = state_from_accessibility(accessibility);
        match state {
            MemoryState::Active => active.push(entry),
            MemoryState::Dormant => dormant.push(entry),
            MemoryState::Silent => silent.push(entry),
            MemoryState::Unavailable => unavailable.push(entry),
        }
    }

    // Apply filter and limit
    let result = match state_filter {
        Some("active") => serde_json::json!({
            "state": "active",
            "count": active.len(),
            "memories": active.into_iter().take(limit).collect::<Vec<_>>()
        }),
        Some("dormant") => serde_json::json!({
            "state": "dormant",
            "count": dormant.len(),
            "memories": dormant.into_iter().take(limit).collect::<Vec<_>>()
        }),
        Some("silent") => serde_json::json!({
            "state": "silent",
            "count": silent.len(),
            "memories": silent.into_iter().take(limit).collect::<Vec<_>>()
        }),
        Some("unavailable") => serde_json::json!({
            "state": "unavailable",
            "count": unavailable.len(),
            "memories": unavailable.into_iter().take(limit).collect::<Vec<_>>()
        }),
        _ => serde_json::json!({
            "all": true,
            "active": { "count": active.len(), "memories": active.into_iter().take(limit).collect::<Vec<_>>() },
            "dormant": { "count": dormant.len(), "memories": dormant.into_iter().take(limit).collect::<Vec<_>>() },
            "silent": { "count": silent.len(), "memories": silent.into_iter().take(limit).collect::<Vec<_>>() },
            "unavailable": { "count": unavailable.len(), "memories": unavailable.into_iter().take(limit).collect::<Vec<_>>() }
        })
    };

    Ok(result)
}

/// Get memory state statistics
pub async fn execute_stats(
    storage: &Arc<Mutex<Storage>>,
) -> Result<Value, String> {
    let storage = storage.lock().await;

    let memories = storage.get_all_nodes(1000, 0)
        .map_err(|e| e.to_string())?;

    let total = memories.len();
    let mut active_count = 0;
    let mut dormant_count = 0;
    let mut silent_count = 0;
    let mut unavailable_count = 0;
    let mut total_accessibility = 0.0;

    for memory in &memories {
        let accessibility = compute_accessibility(
            memory.retention_strength,
            memory.retrieval_strength,
            memory.storage_strength,
        );
        total_accessibility += accessibility;

        let state = state_from_accessibility(accessibility);
        match state {
            MemoryState::Active => active_count += 1,
            MemoryState::Dormant => dormant_count += 1,
            MemoryState::Silent => silent_count += 1,
            MemoryState::Unavailable => unavailable_count += 1,
        }
    }

    let avg_accessibility = if total > 0 { total_accessibility / total as f64 } else { 0.0 };

    Ok(serde_json::json!({
        "totalMemories": total,
        "averageAccessibility": avg_accessibility,
        "stateDistribution": {
            "active": {
                "count": active_count,
                "percentage": if total > 0 { (active_count as f64 / total as f64) * 100.0 } else { 0.0 }
            },
            "dormant": {
                "count": dormant_count,
                "percentage": if total > 0 { (dormant_count as f64 / total as f64) * 100.0 } else { 0.0 }
            },
            "silent": {
                "count": silent_count,
                "percentage": if total > 0 { (silent_count as f64 / total as f64) * 100.0 } else { 0.0 }
            },
            "unavailable": {
                "count": unavailable_count,
                "percentage": if total > 0 { (unavailable_count as f64 / total as f64) * 100.0 } else { 0.0 }
            }
        },
        "thresholds": {
            "active": ACCESSIBILITY_ACTIVE,
            "dormant": ACCESSIBILITY_DORMANT,
            "silent": ACCESSIBILITY_SILENT
        },
        "science": {
            "theory": "Accessibility Continuum (Tulving, 1983)",
            "principle": "Memories exist on a continuum from highly accessible to completely inaccessible"
        }
    }))
}
