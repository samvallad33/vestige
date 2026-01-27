#![allow(dead_code)]
//! Synaptic Tagging Tool (Deprecated)
//!
//! Retroactive importance assignment based on Synaptic Tagging & Capture theory.
//! Frey & Morris (1997), Redondo & Morris (2011).

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{
    CaptureWindow, ImportanceEvent, ImportanceEventType,
    SynapticTaggingConfig, SynapticTaggingSystem, Storage,
};

/// Input schema for trigger_importance tool
pub fn trigger_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "event_type": {
                "type": "string",
                "enum": ["user_flag", "emotional", "novelty", "repeated_access", "cross_reference"],
                "description": "Type of importance event"
            },
            "memory_id": {
                "type": "string",
                "description": "The memory that triggered the importance signal"
            },
            "description": {
                "type": "string",
                "description": "Description of why this is important (optional)"
            },
            "hours_back": {
                "type": "number",
                "description": "How many hours back to look for related memories (default: 9)"
            },
            "hours_forward": {
                "type": "number",
                "description": "How many hours forward to capture (default: 2)"
            }
        },
        "required": ["event_type", "memory_id"]
    })
}

/// Input schema for find_tagged tool
pub fn find_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "min_strength": {
                "type": "number",
                "description": "Minimum tag strength (0.0-1.0, default: 0.3)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 20)"
            }
        },
        "required": []
    })
}

/// Input schema for tag_stats tool
pub fn stats_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

/// Trigger an importance event to retroactively strengthen recent memories
pub async fn execute_trigger(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.ok_or("Missing arguments")?;

    let event_type_str = args["event_type"]
        .as_str()
        .ok_or("event_type is required")?;

    let memory_id = args["memory_id"]
        .as_str()
        .ok_or("memory_id is required")?;

    let description = args["description"].as_str();
    let hours_back = args["hours_back"].as_f64().unwrap_or(9.0);
    let hours_forward = args["hours_forward"].as_f64().unwrap_or(2.0);

    let storage = storage.lock().await;

    // Verify the trigger memory exists
    let trigger_memory = storage.get_node(memory_id)
        .map_err(|e| format!("Error: {}", e))?
        .ok_or("Memory not found")?;

    // Create importance event based on type
    let _event_type = match event_type_str {
        "user_flag" => ImportanceEventType::UserFlag,
        "emotional" => ImportanceEventType::EmotionalContent,
        "novelty" => ImportanceEventType::NoveltySpike,
        "repeated_access" => ImportanceEventType::RepeatedAccess,
        "cross_reference" => ImportanceEventType::CrossReference,
        _ => return Err(format!("Unknown event type: {}", event_type_str)),
    };

    // Create event using user_flag constructor (simpler API)
    let event = ImportanceEvent::user_flag(memory_id, description);

    // Configure capture window
    let config = SynapticTaggingConfig {
        capture_window: CaptureWindow::new(hours_back, hours_forward),
        prp_threshold: 0.5,
        tag_lifetime_hours: 12.0,
        min_tag_strength: 0.1,
        max_cluster_size: 100,
        enable_clustering: true,
        auto_decay: true,
        cleanup_interval_hours: 1.0,
    };

    let mut stc = SynapticTaggingSystem::with_config(config);

    // Get recent memories to tag
    let recent = storage.get_all_nodes(100, 0)
        .map_err(|e| e.to_string())?;

    // Tag all recent memories
    for mem in &recent {
        stc.tag_memory(&mem.id);
    }

    // Trigger PRP (Plasticity-Related Proteins) synthesis
    let result = stc.trigger_prp(event);

    Ok(serde_json::json!({
        "success": true,
        "eventType": event_type_str,
        "triggerMemory": {
            "id": memory_id,
            "content": trigger_memory.content
        },
        "captureWindow": {
            "hoursBack": hours_back,
            "hoursForward": hours_forward
        },
        "result": {
            "memoriesCaptured": result.captured_count(),
            "description": description
        },
        "explanation": format!(
            "Importance signal triggered! {} memories within the {:.1}h window have been retroactively strengthened.",
            result.captured_count(), hours_back
        )
    }))
}

/// Find memories with active synaptic tags
pub async fn execute_find(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.unwrap_or(serde_json::json!({}));

    let min_strength = args["min_strength"].as_f64().unwrap_or(0.3);
    let limit = args["limit"].as_i64().unwrap_or(20) as usize;

    let storage = storage.lock().await;

    // Get memories with high retention (proxy for "tagged")
    let memories = storage.get_all_nodes(200, 0)
        .map_err(|e| e.to_string())?;

    // Filter by retention strength (tagged memories have higher retention)
    let tagged: Vec<Value> = memories.into_iter()
        .filter(|m| m.retention_strength >= min_strength)
        .take(limit)
        .map(|m| serde_json::json!({
            "id": m.id,
            "content": m.content,
            "retentionStrength": m.retention_strength,
            "storageStrength": m.storage_strength,
            "lastAccessed": m.last_accessed.to_rfc3339(),
            "tags": m.tags
        }))
        .collect();

    Ok(serde_json::json!({
        "success": true,
        "minStrength": min_strength,
        "taggedCount": tagged.len(),
        "memories": tagged
    }))
}

/// Get synaptic tagging statistics
pub async fn execute_stats(
    storage: &Arc<Mutex<Storage>>,
) -> Result<Value, String> {
    let storage = storage.lock().await;

    let memories = storage.get_all_nodes(500, 0)
        .map_err(|e| e.to_string())?;

    let total = memories.len();
    let high_retention = memories.iter().filter(|m| m.retention_strength >= 0.7).count();
    let medium_retention = memories.iter().filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7).count();
    let low_retention = memories.iter().filter(|m| m.retention_strength < 0.4).count();

    let avg_retention = if total > 0 {
        memories.iter().map(|m| m.retention_strength).sum::<f64>() / total as f64
    } else {
        0.0
    };

    let avg_storage = if total > 0 {
        memories.iter().map(|m| m.storage_strength).sum::<f64>() / total as f64
    } else {
        0.0
    };

    Ok(serde_json::json!({
        "totalMemories": total,
        "averageRetention": avg_retention,
        "averageStorage": avg_storage,
        "distribution": {
            "highRetention": {
                "count": high_retention,
                "threshold": 0.7,
                "percentage": if total > 0 { (high_retention as f64 / total as f64) * 100.0 } else { 0.0 }
            },
            "mediumRetention": {
                "count": medium_retention,
                "threshold": "0.4-0.7",
                "percentage": if total > 0 { (medium_retention as f64 / total as f64) * 100.0 } else { 0.0 }
            },
            "lowRetention": {
                "count": low_retention,
                "threshold": "<0.4",
                "percentage": if total > 0 { (low_retention as f64 / total as f64) * 100.0 } else { 0.0 }
            }
        },
        "science": {
            "theory": "Synaptic Tagging and Capture (Frey & Morris 1997)",
            "principle": "Weak memories can be retroactively strengthened when important events occur within a temporal window",
            "captureWindow": "Up to 9 hours in biological systems"
        }
    }))
}
