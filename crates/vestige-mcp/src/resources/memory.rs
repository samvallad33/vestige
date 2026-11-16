//! Memory Resources
//!
//! memory:// URI scheme resources for the MCP server.

use std::sync::Arc;

use vestige_core::Storage;

/// Read a memory:// resource
pub async fn read(storage: &Arc<Storage>, uri: &str) -> Result<String, String> {
    let path = uri.strip_prefix("memory://").unwrap_or("");

    // Parse query parameters if present
    let (path, query) = match path.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (path, None),
    };

    match path {
        "stats" => read_stats(storage).await,
        "recent" => {
            let n = parse_query_param(query, "n", 10);
            read_recent(storage, n).await
        }
        "decaying" => read_decaying(storage).await,
        "due" => read_due(storage).await,
        "intentions" => read_intentions(storage).await,
        "intentions/due" => read_triggered_intentions(storage).await,
        "proof-header" => crate::resources::proof_header::read(storage.as_ref()),
        "insights" => read_insights(storage).await,
        "consolidation-log" => read_consolidation_log(storage).await,
        _ => Err(format!("Unknown memory resource: {}", path)),
    }
}

fn parse_query_param(query: Option<&str>, key: &str, default: i32) -> i32 {
    query
        .and_then(|q| {
            q.split('&').find_map(|pair| {
                let (k, v) = pair.split_once('=')?;
                if k == key { v.parse().ok() } else { None }
            })
        })
        .unwrap_or(default)
        .clamp(1, 100)
}

async fn read_stats(storage: &Arc<Storage>) -> Result<String, String> {
    let stats = storage.get_stats().map_err(|e| e.to_string())?;

    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };

    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    let result = serde_json::json!({
        "status": status,
        "totalNodes": stats.total_nodes,
        "nodesDueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "averageStorageStrength": stats.average_storage_strength,
        "averageRetrievalStrength": stats.average_retrieval_strength,
        "oldestMemory": stats.oldest_memory.map(|d| d.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|d| d.to_rfc3339()),
        "nodesWithEmbeddings": stats.nodes_with_embeddings,
        "embeddingCoverage": format!("{:.1}%", embedding_coverage),
        "embeddingModel": stats.embedding_model,
        // w1b: vector code removed; there is no embedding service to be ready.
        "embeddingServiceReady": false,
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_recent(storage: &Arc<Storage>, limit: i32) -> Result<String, String> {
    let nodes = storage.get_all_nodes(limit, 0).map_err(|e| e.to_string())?;

    let items: Vec<serde_json::Value> = nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "summary": if n.content.len() > 200 {
                    format!("{}...", &n.content[..n.content.floor_char_boundary(200)])
                } else {
                    n.content.clone()
                },
                "nodeType": n.node_type,
                "tags": n.tags,
                "createdAt": n.created_at.to_rfc3339(),
                "retentionStrength": n.retention_strength,
            })
        })
        .collect();

    let result = serde_json::json!({
        "total": nodes.len(),
        "items": items,
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_decaying(storage: &Arc<Storage>) -> Result<String, String> {
    // Get nodes with low retention (below 0.5)
    let all_nodes = storage.get_all_nodes(100, 0).map_err(|e| e.to_string())?;

    let mut decaying: Vec<_> = all_nodes
        .into_iter()
        .filter(|n| n.retention_strength < 0.5)
        .collect();

    // Sort by retention strength (lowest first)
    decaying.sort_by(|a, b| {
        a.retention_strength
            .partial_cmp(&b.retention_strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let items: Vec<serde_json::Value> = decaying
        .iter()
        .take(20)
        .map(|n| {
            let days_since_access = (chrono::Utc::now() - n.last_accessed).num_days();
            serde_json::json!({
                "id": n.id,
                "summary": if n.content.len() > 200 {
                    format!("{}...", &n.content[..n.content.floor_char_boundary(200)])
                } else {
                    n.content.clone()
                },
                "retentionStrength": n.retention_strength,
                "daysSinceAccess": days_since_access,
                "lastAccessed": n.last_accessed.to_rfc3339(),
                "hint": if n.retention_strength < 0.2 {
                    "Critical - review immediately!"
                } else {
                    "Should be reviewed soon"
                },
            })
        })
        .collect();

    let result = serde_json::json!({
        "total": decaying.len(),
        "showing": items.len(),
        "items": items,
        "recommendation": if decaying.is_empty() {
            "All memories are healthy!"
        } else if decaying.len() > 10 {
            "Many memories are decaying. Consider reviewing the most important ones."
        } else {
            "Some memories need attention. Review to strengthen retention."
        },
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_due(storage: &Arc<Storage>) -> Result<String, String> {
    let nodes = storage.get_review_queue(20).map_err(|e| e.to_string())?;

    let items: Vec<serde_json::Value> = nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "summary": if n.content.len() > 200 {
                    format!("{}...", &n.content[..n.content.floor_char_boundary(200)])
                } else {
                    n.content.clone()
                },
                "nodeType": n.node_type,
                "retentionStrength": n.retention_strength,
                "difficulty": n.difficulty,
                "reps": n.reps,
                "nextReview": n.next_review.map(|d| d.to_rfc3339()),
            })
        })
        .collect();

    let result = serde_json::json!({
        "total": nodes.len(),
        "items": items,
        "instruction": "Use mark_reviewed with rating 1-4 to complete review",
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_intentions(storage: &Arc<Storage>) -> Result<String, String> {
    let intentions = storage.get_active_intentions().map_err(|e| e.to_string())?;
    let now = chrono::Utc::now();

    let items: Vec<serde_json::Value> = intentions
        .iter()
        .map(|i| {
            let is_overdue = i.deadline.map(|d| d < now).unwrap_or(false);
            serde_json::json!({
                "id": i.id,
                "description": i.content,
                "status": i.status,
                "priority": match i.priority {
                    1 => "low",
                    3 => "high",
                    4 => "critical",
                    _ => "normal",
                },
                "createdAt": i.created_at.to_rfc3339(),
                "deadline": i.deadline.map(|d| d.to_rfc3339()),
                "isOverdue": is_overdue,
                "snoozedUntil": i.snoozed_until.map(|d| d.to_rfc3339()),
            })
        })
        .collect();

    let overdue_count = items
        .iter()
        .filter(|i| i["isOverdue"].as_bool().unwrap_or(false))
        .count();

    let result = serde_json::json!({
        "total": intentions.len(),
        "overdueCount": overdue_count,
        "items": items,
        "tip": "Use set_intention to add new intentions, complete_intention to mark done",
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_triggered_intentions(storage: &Arc<Storage>) -> Result<String, String> {
    let overdue = storage
        .get_overdue_intentions()
        .map_err(|e| e.to_string())?;
    let now = chrono::Utc::now();

    let items: Vec<serde_json::Value> = overdue
        .iter()
        .map(|i| {
            let overdue_by = i.deadline.map(|d| {
                let duration = now - d;
                if duration.num_days() > 0 {
                    format!("{} days", duration.num_days())
                } else if duration.num_hours() > 0 {
                    format!("{} hours", duration.num_hours())
                } else {
                    format!("{} minutes", duration.num_minutes())
                }
            });
            serde_json::json!({
                "id": i.id,
                "description": i.content,
                "priority": match i.priority {
                    1 => "low",
                    3 => "high",
                    4 => "critical",
                    _ => "normal",
                },
                "deadline": i.deadline.map(|d| d.to_rfc3339()),
                "overdueBy": overdue_by,
            })
        })
        .collect();

    let result = serde_json::json!({
        "triggered": items.len(),
        "items": items,
        "message": if items.is_empty() {
            "No overdue intentions!"
        } else {
            "These intentions need attention"
        },
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_insights(storage: &Arc<Storage>) -> Result<String, String> {
    let insights = storage.get_insights(50).map_err(|e| e.to_string())?;

    let pending: Vec<_> = insights.iter().filter(|i| i.feedback.is_none()).collect();
    let accepted: Vec<_> = insights
        .iter()
        .filter(|i| i.feedback.as_deref() == Some("accepted"))
        .collect();

    let items: Vec<serde_json::Value> = insights
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "insight": i.insight,
                "type": i.insight_type,
                "confidence": i.confidence,
                "noveltyScore": i.novelty_score,
                "sourceMemories": i.source_memories,
                "generatedAt": i.generated_at.to_rfc3339(),
                "feedback": i.feedback,
            })
        })
        .collect();

    let result = serde_json::json!({
        "total": insights.len(),
        "pendingReview": pending.len(),
        "accepted": accepted.len(),
        "items": items,
        "tip": "These insights were discovered during memory consolidation",
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}

async fn read_consolidation_log(storage: &Arc<Storage>) -> Result<String, String> {
    let history = storage
        .get_consolidation_history(20)
        .map_err(|e| e.to_string())?;
    let last_run = storage
        .get_last_consolidation()
        .map_err(|e| e.to_string())?;

    let items: Vec<serde_json::Value> = history
        .iter()
        .map(|h| {
            serde_json::json!({
                "id": h.id,
                "completedAt": h.completed_at.to_rfc3339(),
                "durationMs": h.duration_ms,
                "memoriesReplayed": h.memories_replayed,
                "connectionsFound": h.connections_found,
                "connectionsStrengthened": h.connections_strengthened,
                "connectionsPruned": h.connections_pruned,
                "insightsGenerated": h.insights_generated,
            })
        })
        .collect();

    let result = serde_json::json!({
        "lastRun": last_run.map(|d| d.to_rfc3339()),
        "totalRuns": history.len(),
        "history": items,
    });

    serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
}
