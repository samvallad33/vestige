#![allow(dead_code)]
//! Stats Tools (Deprecated - use memory_unified instead)
//!
//! Memory statistics and health check.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{MemoryStats, Storage};

/// Input schema for get_stats tool
pub fn stats_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

/// Input schema for health_check tool
pub fn health_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

pub async fn execute_stats(storage: &Arc<Mutex<Storage>>) -> Result<Value, String> {
    let storage = storage.lock().await;
    let stats = storage.get_stats().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "totalNodes": stats.total_nodes,
        "nodesDueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "averageStorageStrength": stats.average_storage_strength,
        "averageRetrievalStrength": stats.average_retrieval_strength,
        "oldestMemory": stats.oldest_memory.map(|d| d.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|d| d.to_rfc3339()),
        "nodesWithEmbeddings": stats.nodes_with_embeddings,
        "embeddingModel": stats.embedding_model,
        "embeddingServiceReady": storage.is_embedding_ready(),
    }))
}

pub async fn execute_health(storage: &Arc<Mutex<Storage>>) -> Result<Value, String> {
    let storage = storage.lock().await;
    let stats = storage.get_stats().map_err(|e| e.to_string())?;

    // Determine health status
    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    let mut warnings = Vec::new();

    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings.push("Low average retention - consider running consolidation or reviewing memories".to_string());
    }

    if stats.nodes_due_for_review > 10 {
        warnings.push(format!("{} memories are due for review", stats.nodes_due_for_review));
    }

    if stats.total_nodes > 0 && stats.nodes_with_embeddings == 0 {
        warnings.push("No embeddings generated - semantic search unavailable. Run consolidation.".to_string());
    }

    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };

    if embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push(format!("Only {:.1}% of memories have embeddings", embedding_coverage));
    }

    Ok(serde_json::json!({
        "status": status,
        "totalNodes": stats.total_nodes,
        "nodesDueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "embeddingCoverage": format!("{:.1}%", embedding_coverage),
        "embeddingServiceReady": storage.is_embedding_ready(),
        "warnings": warnings,
        "recommendations": get_recommendations(&stats, status),
    }))
}

fn get_recommendations(
    stats: &MemoryStats,
    status: &str,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    if status == "critical" {
        recommendations.push("CRITICAL: Many memories have very low retention. Review important memories with 'mark_reviewed'.".to_string());
    }

    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.".to_string());
    }

    if stats.nodes_with_embeddings < stats.total_nodes {
        recommendations.push("Run 'run_consolidation' to generate embeddings for better semantic search.".to_string());
    }

    if stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider running periodic consolidation to maintain memory health.".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Memory system is healthy!".to_string());
    }

    recommendations
}
