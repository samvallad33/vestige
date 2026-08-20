//! memory_health tool — Retention dashboard for memory quality monitoring.
//! v1.9.0: Lightweight alternative to full system_status focused on memory health.

use std::sync::Arc;
use vestige_core::Storage;

pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

pub async fn execute(
    storage: &Arc<Storage>,
    _args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    // Average retention
    let avg_retention = storage
        .get_avg_retention()
        .map_err(|e| format!("Failed to get avg retention: {}", e))?;

    // Retention distribution
    let distribution = storage
        .get_retention_distribution()
        .map_err(|e| format!("Failed to get retention distribution: {}", e))?;

    let distribution_json: serde_json::Value = distribution
        .iter()
        .map(|(bucket, count)| serde_json::json!({ "bucket": bucket, "count": count }))
        .collect();

    // Retention trend
    let trend = storage
        .get_retention_trend()
        .unwrap_or_else(|_| "unknown".to_string());

    // Total memories and those below key thresholds
    let stats = storage
        .get_stats()
        .map_err(|e| format!("Failed to get stats: {}", e))?;

    let below_30 = storage.count_memories_below_retention(0.3).unwrap_or(0);
    let below_50 = storage.count_memories_below_retention(0.5).unwrap_or(0);

    // Retention target
    let retention_target: f64 = std::env::var("VESTIGE_RETENTION_TARGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);

    let meets_target = avg_retention >= retention_target;

    // Generate recommendation
    let recommendation = if avg_retention >= 0.8 {
        "Excellent memory health. Retention is strong across the board."
    } else if avg_retention >= 0.6 {
        "Good memory health. Consider reviewing memories in the 0-40% range."
    } else if avg_retention >= 0.4 {
        "Fair memory health. Many memories are decaying. Run consolidation and consider GC."
    } else {
        "Poor memory health. Urgent: run consolidation, then GC stale memories below 0.3."
    };

    Ok(serde_json::json!({
        "avgRetention": format!("{:.1}%", avg_retention * 100.0),
        "avgRetentionRaw": avg_retention,
        "retentionTarget": retention_target,
        "meetsTarget": meets_target,
        "totalMemories": stats.total_nodes,
        "distribution": distribution_json,
        "trend": trend,
        "memoriesBelow30pct": below_30,
        "memoriesBelow50pct": below_50,
        "recommendation": recommendation,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[test]
    fn test_schema_is_valid() {
        let s = schema();
        assert_eq!(s["type"], "object");
    }

    #[tokio::test]
    async fn test_health_empty_database() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 0);
        assert!(value["avgRetention"].is_string());
        assert!(value["recommendation"].is_string());
    }

    #[tokio::test]
    async fn test_health_with_memories() {
        let (storage, _dir) = test_storage().await;
        // Ingest some test memories
        for i in 0..5 {
            storage
                .ingest(vestige_core::IngestInput {
                    content: format!("Health test memory {}", i),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["test".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }

        let result = execute(&storage, None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 5);
        assert!(value["distribution"].is_array());
        assert!(value["meetsTarget"].is_boolean());
    }

    #[tokio::test]
    async fn test_health_distribution_buckets() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(vestige_core::IngestInput {
                content: "Test memory for distribution".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();

        let result = execute(&storage, None).await.unwrap();
        let dist = result["distribution"].as_array().unwrap();
        // Should have at least one bucket with data
        assert!(!dist.is_empty());
        let total: i64 = dist.iter().map(|b| b["count"].as_i64().unwrap_or(0)).sum();
        assert_eq!(total, 1);
    }
}
