//! Synaptic Tagging Tool (Deprecated)
//!
//! Retroactive importance assignment based on Synaptic Tagging & Capture theory.
//! Frey & Morris (1997), Redondo & Morris (2011).

use serde_json::Value;
use std::sync::Arc;

use vestige_core::Storage;

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

/// Legacy trigger endpoint deliberately disabled.
///
/// It previously built a temporary in-memory tagging system, reported captures
/// that never reached SQLite, and then dropped that system. V22 capture is a
/// storage transaction that records the event, candidates, mutation, and
/// receipt together. Until this legacy endpoint can provide that same durable
/// contract, failing closed is safer than returning an unaudited capture claim.
pub async fn execute_trigger(
    _storage: &Arc<Storage>,
    _args: Option<Value>,
) -> Result<Value, String> {
    Err(
        "trigger_importance is disabled because its legacy in-memory capture path cannot produce a durable V22 decision receipt. Use smart_ingest; qualifying high-salience ingests run the auditable capture transaction."
            .to_string(),
    )
}

/// Find memories with active synaptic tags
pub async fn execute_find(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args = args.unwrap_or(serde_json::json!({}));

    let min_strength = args["min_strength"].as_f64().unwrap_or(0.3);
    let limit = args["limit"].as_i64().unwrap_or(20) as usize;

    // Get memories with high retention (proxy for "tagged")
    let memories = storage.get_all_nodes(200, 0).map_err(|e| e.to_string())?;

    // Filter by retention strength (tagged memories have higher retention)
    let tagged: Vec<Value> = memories
        .into_iter()
        .filter(|m| m.retention_strength >= min_strength)
        .take(limit)
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "content": m.content,
                "retentionStrength": m.retention_strength,
                "storageStrength": m.storage_strength,
                "lastAccessed": m.last_accessed.to_rfc3339(),
                "tags": m.tags
            })
        })
        .collect();

    Ok(serde_json::json!({
        "success": true,
        "minStrength": min_strength,
        "taggedCount": tagged.len(),
        "memories": tagged
    }))
}

/// Get synaptic tagging statistics
pub async fn execute_stats(storage: &Arc<Storage>) -> Result<Value, String> {
    let memories = storage.get_all_nodes(500, 0).map_err(|e| e.to_string())?;

    let total = memories.len();
    let high_retention = memories
        .iter()
        .filter(|m| m.retention_strength >= 0.7)
        .count();
    let medium_retention = memories
        .iter()
        .filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7)
        .count();
    let low_retention = memories
        .iter()
        .filter(|m| m.retention_strength < 0.4)
        .count();

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

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    #[tokio::test]
    async fn legacy_trigger_fails_closed_without_mutating_memory_state() {
        let directory = tempfile::tempdir().expect("temporary database directory");
        let storage = Arc::new(
            Storage::new(Some(directory.path().join("tagging.db"))).expect("test storage"),
        );
        let node = storage
            .ingest(IngestInput {
                content: "must not be captured by a temporary in-memory system".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .expect("seed memory");
        let before = storage
            .get_node(&node.id)
            .expect("load before")
            .expect("node");

        let error = execute_trigger(
            &storage,
            Some(serde_json::json!({
                "event_type": "user_flag",
                "memory_id": node.id,
            })),
        )
        .await
        .expect_err("legacy trigger must fail closed");

        assert!(error.contains("durable V22 decision receipt"));
        let after = storage
            .get_node(&node.id)
            .expect("load after")
            .expect("node");
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
        assert_eq!(after.retention_strength, before.retention_strength);
        assert_eq!(after.stability, before.stability);
    }
}
