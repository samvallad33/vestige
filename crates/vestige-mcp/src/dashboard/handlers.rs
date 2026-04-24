//! Dashboard API endpoint handlers
//!
//! v2.0: Adds cognitive operation endpoints (dream, explore, predict, importance, consolidation)

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Json, Redirect};
use chrono::{Duration, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::events::VestigeEvent;
use super::state::AppState;

/// Redirect root to the SvelteKit dashboard
pub async fn serve_dashboard() -> Redirect {
    Redirect::permanent("/dashboard")
}

#[derive(Debug, Deserialize)]
pub struct MemoryListParams {
    pub q: Option<String>,
    pub node_type: Option<String>,
    pub tag: Option<String>,
    pub min_retention: Option<f64>,
    pub sort: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

/// List memories with optional search
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<MemoryListParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(50).clamp(1, 200);
    let offset = params.offset.unwrap_or(0).max(0);

    if let Some(query) = params.q.as_ref().filter(|q| !q.trim().is_empty()) {
        // Use hybrid search
        let results = state
            .storage
            .hybrid_search(query, limit, 0.3, 0.7)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let formatted: Vec<Value> = results
            .into_iter()
            .filter(|r| {
                if let Some(min_ret) = params.min_retention {
                    r.node.retention_strength >= min_ret
                } else {
                    true
                }
            })
            .map(|r| {
                serde_json::json!({
                    "id": r.node.id,
                    "content": r.node.content,
                    "nodeType": r.node.node_type,
                    "tags": r.node.tags,
                    "retentionStrength": r.node.retention_strength,
                    "storageStrength": r.node.storage_strength,
                    "retrievalStrength": r.node.retrieval_strength,
                    "createdAt": r.node.created_at.to_rfc3339(),
                    "updatedAt": r.node.updated_at.to_rfc3339(),
                    "combinedScore": r.combined_score,
                    "source": r.node.source,
                    "reviewCount": r.node.reps,
                })
            })
            .collect();

        return Ok(Json(serde_json::json!({
            "total": formatted.len(),
            "memories": formatted,
        })));
    }

    // No search query — list all memories
    let mut nodes = state
        .storage
        .get_all_nodes(limit, offset)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Apply filters
    if let Some(ref node_type) = params.node_type {
        nodes.retain(|n| n.node_type == *node_type);
    }
    if let Some(ref tag) = params.tag {
        nodes.retain(|n| n.tags.iter().any(|t| t == tag));
    }
    if let Some(min_ret) = params.min_retention {
        nodes.retain(|n| n.retention_strength >= min_ret);
    }

    let formatted: Vec<Value> = nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content,
                "nodeType": n.node_type,
                "tags": n.tags,
                "retentionStrength": n.retention_strength,
                "storageStrength": n.storage_strength,
                "retrievalStrength": n.retrieval_strength,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "source": n.source,
                "reviewCount": n.reps,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "total": formatted.len(),
        "memories": formatted,
    })))
}

/// Get a single memory by ID
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .get_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(serde_json::json!({
        "id": node.id,
        "content": node.content,
        "nodeType": node.node_type,
        "tags": node.tags,
        "retentionStrength": node.retention_strength,
        "storageStrength": node.storage_strength,
        "retrievalStrength": node.retrieval_strength,
        "sentimentScore": node.sentiment_score,
        "sentimentMagnitude": node.sentiment_magnitude,
        "source": node.source,
        "createdAt": node.created_at.to_rfc3339(),
        "updatedAt": node.updated_at.to_rfc3339(),
        "lastAccessedAt": node.last_accessed.to_rfc3339(),
        "nextReviewAt": node.next_review.map(|dt| dt.to_rfc3339()),
        "reviewCount": node.reps,
        "validFrom": node.valid_from.map(|dt| dt.to_rfc3339()),
        "validUntil": node.valid_until.map(|dt| dt.to_rfc3339()),
    })))
}

/// Delete a memory by ID
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let deleted = state
        .storage
        .delete_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if deleted {
        state.emit(VestigeEvent::MemoryDeleted {
            id: id.clone(),
            timestamp: chrono::Utc::now(),
        });
        Ok(Json(serde_json::json!({ "deleted": true, "id": id })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Promote a memory
pub async fn promote_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .promote_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state.emit(VestigeEvent::MemoryPromoted {
        id: node.id.clone(),
        new_retention: node.retention_strength,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "promoted": true,
        "id": node.id,
        "retentionStrength": node.retention_strength,
    })))
}

/// Demote a memory
pub async fn demote_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .demote_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state.emit(VestigeEvent::MemoryDemoted {
        id: node.id.clone(),
        new_retention: node.retention_strength,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "demoted": true,
        "id": node.id,
        "retentionStrength": node.retention_strength,
    })))
}

/// Actively suppress a memory via top-down inhibitory control.
///
/// Optional JSON body: `{"reason": "..."}`. Each call compounds — the
/// `suppression_count` field on the memory increments, FSRS takes a hit,
/// and the background Rac1 worker fades co-activated neighbors over the
/// next 72 hours. Emits a `MemorySuppressed` event so the 3D graph plays
/// the violet implosion animation.
///
/// Reversible within the 24-hour labile window via `unsuppress_memory`.
///
/// Fixes the v2.0.5 UI gap: `suppress` had full graph event handlers and
/// MCP tool exposure, but zero HTTP endpoint and no dashboard trigger.
pub async fn suppress_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<Value>>,
) -> Result<Json<Value>, StatusCode> {
    use vestige_core::neuroscience::active_forgetting::{
        ActiveForgettingSystem, DEFAULT_LABILE_HOURS,
    };

    let reason = body
        .as_ref()
        .and_then(|Json(v)| v.get("reason"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let sys = ActiveForgettingSystem::new();

    // Pre-count to surface in the response + for the event payload.
    let before_count = state
        .storage
        .get_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(|n| n.suppression_count)
        .unwrap_or(0);

    let node = state
        .storage
        .suppress_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Estimate cascade size for the UX; capped at 100 so the number is
    // stable even on highly-connected nodes.
    let estimated_cascade = state
        .storage
        .get_connections_for_memory(&id)
        .map(|edges| edges.len().min(100))
        .unwrap_or(0);

    let reversible_until = node
        .suppressed_at
        .map(|t| sys.reversible_until(t))
        .unwrap_or_else(chrono::Utc::now);
    let retrieval_penalty = sys.retrieval_penalty(node.suppression_count);

    tracing::info!(
        id = %id,
        count = node.suppression_count,
        reason = reason.as_deref().unwrap_or(""),
        "Memory suppressed via dashboard"
    );

    state.emit(VestigeEvent::MemorySuppressed {
        id: node.id.clone(),
        suppression_count: node.suppression_count,
        estimated_cascade,
        reversible_until,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "suppressed": true,
        "id": node.id,
        "suppressionCount": node.suppression_count,
        "priorCount": before_count,
        "retrievalPenalty": retrieval_penalty,
        "retentionStrength": node.retention_strength,
        "retrievalStrength": node.retrieval_strength,
        "stability": node.stability,
        "estimatedCascadeNeighbors": estimated_cascade,
        "reversibleUntil": reversible_until.to_rfc3339(),
        "labileWindowHours": DEFAULT_LABILE_HOURS,
        "reason": reason,
    })))
}

/// Reverse a prior suppression, if still inside the 24-hour labile
/// window. Emits `MemoryUnsuppressed` so the graph plays the rainbow
/// reversal burst. Returns the current suppression state so the UI
/// knows whether a single click fully cleared the suppression or whether
/// the memory still has compounded suppressions remaining.
pub async fn unsuppress_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    use vestige_core::neuroscience::active_forgetting::ActiveForgettingSystem;

    let sys = ActiveForgettingSystem::new();
    let node = state
        .storage
        .reverse_suppression(&id, sys.labile_hours)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let still_suppressed = node.suppression_count > 0;

    state.emit(VestigeEvent::MemoryUnsuppressed {
        id: node.id.clone(),
        remaining_count: node.suppression_count,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "unsuppressed": true,
        "id": node.id,
        "suppressionCount": node.suppression_count,
        "stillSuppressed": still_suppressed,
        "retentionStrength": node.retention_strength,
        "retrievalStrength": node.retrieval_strength,
        "stability": node.stability,
    })))
}

/// Get system stats
pub async fn get_stats(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let stats = state
        .storage
        .get_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };

    Ok(Json(serde_json::json!({
        "totalMemories": stats.total_nodes,
        "dueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "averageStorageStrength": stats.average_storage_strength,
        "averageRetrievalStrength": stats.average_retrieval_strength,
        "withEmbeddings": stats.nodes_with_embeddings,
        "embeddingCoverage": embedding_coverage,
        "embeddingModel": stats.embedding_model,
        "oldestMemory": stats.oldest_memory.map(|dt| dt.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|dt| dt.to_rfc3339()),
    })))
}

#[derive(Debug, Deserialize)]
pub struct TimelineParams {
    pub days: Option<i64>,
    pub limit: Option<i32>,
}

/// Get timeline data
pub async fn get_timeline(
    State(state): State<AppState>,
    Query(params): Query<TimelineParams>,
) -> Result<Json<Value>, StatusCode> {
    let days = params.days.unwrap_or(7).clamp(1, 90);
    let limit = params.limit.unwrap_or(200).clamp(1, 500);

    let start = Utc::now() - Duration::days(days);
    let nodes = state
        .storage
        .query_time_range(Some(start), Some(Utc::now()), limit, None, None)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Group by day
    let mut by_day: std::collections::BTreeMap<String, Vec<Value>> =
        std::collections::BTreeMap::new();
    for node in &nodes {
        let date = node.created_at.format("%Y-%m-%d").to_string();
        let content_preview: String = {
            let preview: String = node.content.chars().take(100).collect();
            if preview.len() < node.content.len() {
                format!("{}...", preview)
            } else {
                preview
            }
        };
        by_day.entry(date).or_default().push(serde_json::json!({
            "id": node.id,
            "content": content_preview,
            "nodeType": node.node_type,
            "retentionStrength": node.retention_strength,
            "createdAt": node.created_at.to_rfc3339(),
        }));
    }

    let timeline: Vec<Value> = by_day
        .into_iter()
        .rev()
        .map(|(date, memories)| {
            serde_json::json!({
                "date": date,
                "count": memories.len(),
                "memories": memories,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "days": days,
        "totalMemories": nodes.len(),
        "timeline": timeline,
    })))
}

/// Health check
pub async fn health_check(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let stats = state
        .storage
        .get_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    Ok(Json(serde_json::json!({
        "status": status,
        "totalMemories": stats.total_nodes,
        "averageRetention": stats.average_retention,
        "version": env!("CARGO_PKG_VERSION"),
    })))
}

// ============================================================================
// MEMORY GRAPH
// ============================================================================

/// Redirect legacy graph to SvelteKit dashboard graph page
pub async fn serve_graph() -> Redirect {
    Redirect::permanent("/dashboard/graph")
}

#[derive(Debug, Deserialize)]
pub struct GraphParams {
    pub query: Option<String>,
    pub center_id: Option<String>,
    pub depth: Option<u32>,
    pub max_nodes: Option<usize>,
    /// How to choose the default center when neither `query` nor `center_id`
    /// is provided. "recent" (default) uses the newest memory — matches
    /// what users actually expect ("show me my recent stuff"). "connected"
    /// uses the most-connected memory for a richer initial subgraph; used
    /// to be the default but ended up clustering on historical hotspots
    /// and hiding fresh memories that hadn't accumulated edges yet.
    /// Unknown values fall back to "recent".
    pub sort: Option<String>,
}

/// Which memory to center the default subgraph on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphSort {
    Recent,
    Connected,
}

impl GraphSort {
    fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::to_ascii_lowercase).as_deref() {
            Some("connected") => Self::Connected,
            _ => Self::Recent,
        }
    }
}

/// Get memory graph data (nodes + edges with layout positions)
pub async fn get_graph(
    State(state): State<AppState>,
    Query(params): Query<GraphParams>,
) -> Result<Json<Value>, StatusCode> {
    let depth = params.depth.unwrap_or(2).clamp(1, 3);
    let max_nodes = params.max_nodes.unwrap_or(50).clamp(1, 200);
    let sort = GraphSort::parse(params.sort.as_deref());

    // Determine center node
    let explicit_center = params.center_id.is_some() || params.query.is_some();
    let center_id = if let Some(ref id) = params.center_id {
        id.clone()
    } else if let Some(ref query) = params.query {
        let results = state
            .storage
            .search(query, 1)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        results
            .first()
            .map(|n| n.id.clone())
            .ok_or(StatusCode::NOT_FOUND)?
    } else {
        default_center_id(&state.storage, sort)?
    };

    // Get subgraph
    let (mut nodes, mut edges) = state
        .storage
        .get_memory_subgraph(&center_id, depth, max_nodes)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Default-load fallback: if the newest memory is isolated (1 node, 0 edges),
    // silently re-resolve via Connected so the user sees the densest cluster
    // instead of a lonely orb. Explicit query/center_id requests are honored
    // as-is — the user asked for that specific subgraph.
    let mut center_id = center_id;
    if !explicit_center
        && sort == GraphSort::Recent
        && nodes.len() <= 1
        && edges.is_empty()
        && let Ok(fallback) = default_center_id(&state.storage, GraphSort::Connected)
        && fallback != center_id
        && let Ok((n2, e2)) = state.storage.get_memory_subgraph(&fallback, depth, max_nodes)
        && n2.len() > nodes.len()
    {
        center_id = fallback;
        nodes = n2;
        edges = e2;
    }

    if nodes.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Build nodes JSON with timestamps for recency calculation
    let nodes_json: Vec<Value> = nodes
        .iter()
        .map(|n| {
            let label = if n.content.chars().count() > 80 {
                format!("{}...", n.content.chars().take(77).collect::<String>())
            } else {
                n.content.clone()
            };
            serde_json::json!({
                "id": n.id,
                "label": label,
                "type": n.node_type,
                "retention": n.retention_strength,
                "tags": n.tags,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "isCenter": n.id == center_id,
            })
        })
        .collect();

    let edges_json: Vec<Value> = edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "source": e.source_id,
                "target": e.target_id,
                "weight": e.strength,
                "type": e.link_type,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
        "center_id": center_id,
        "depth": depth,
        "nodeCount": nodes.len(),
        "edgeCount": edges.len(),
    })))
}

/// Pick the default subgraph center when neither `query` nor `center_id`
/// was provided. Factored out so both the route handler and unit tests can
/// exercise the same branching (recent vs connected + empty-db fallback)
/// without spinning up a full axum server.
fn default_center_id(
    storage: &std::sync::Arc<vestige_core::Storage>,
    sort: GraphSort,
) -> Result<String, StatusCode> {
    match sort {
        GraphSort::Recent => {
            let recent = storage
                .get_all_nodes(1, 0)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            recent
                .first()
                .map(|n| n.id.clone())
                .ok_or(StatusCode::NOT_FOUND)
        }
        GraphSort::Connected => {
            let most_connected = storage
                .get_most_connected_memory()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            if let Some(id) = most_connected {
                Ok(id)
            } else {
                // Nothing connected yet (fresh DB, or every node is isolated) —
                // fall through to the newest memory so the user still sees
                // SOMETHING rather than a 404.
                let recent = storage
                    .get_all_nodes(1, 0)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                recent
                    .first()
                    .map(|n| n.id.clone())
                    .ok_or(StatusCode::NOT_FOUND)
            }
        }
    }
}

// ============================================================================
// SEARCH (dedicated endpoint)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<i32>,
    pub min_retention: Option<f64>,
}

/// Search memories with hybrid search
pub async fn search_memories(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(20).clamp(1, 100);
    let start = std::time::Instant::now();

    let results = state
        .storage
        .hybrid_search(&params.q, limit, 0.3, 0.7)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let result_ids: Vec<String> = results.iter().map(|r| r.node.id.clone()).collect();

    // Emit search event
    state.emit(VestigeEvent::SearchPerformed {
        query: params.q.clone(),
        result_count: results.len(),
        result_ids: result_ids.clone(),
        duration_ms,
        timestamp: Utc::now(),
    });

    let formatted: Vec<Value> = results
        .into_iter()
        .filter(|r| {
            params
                .min_retention
                .is_none_or(|min| r.node.retention_strength >= min)
        })
        .map(|r| {
            serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
                "combinedScore": r.combined_score,
                "createdAt": r.node.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "query": params.q,
        "total": formatted.len(),
        "durationMs": duration_ms,
        "results": formatted,
    })))
}

// ============================================================================
// COGNITIVE OPERATIONS (v2.0)
// ============================================================================

/// Trigger a dream cycle via CognitiveEngine
pub async fn trigger_dream(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let cognitive = state
        .cognitive
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let start = std::time::Instant::now();
    let memory_count: usize = 50;

    // Load memories for dreaming
    let all_nodes = state
        .storage
        .get_all_nodes(memory_count as i32, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if all_nodes.len() < 5 {
        return Ok(Json(serde_json::json!({
            "status": "insufficient_memories",
            "message": format!("Need at least 5 memories. Current: {}", all_nodes.len()),
        })));
    }

    // Emit start event
    state.emit(VestigeEvent::DreamStarted {
        memory_count: all_nodes.len(),
        timestamp: Utc::now(),
    });

    // Build dream memories
    let dream_memories: Vec<vestige_core::DreamMemory> = all_nodes
        .iter()
        .map(|n| vestige_core::DreamMemory {
            id: n.id.clone(),
            content: n.content.clone(),
            embedding: state.storage.get_node_embedding(&n.id).ok().flatten(),
            tags: n.tags.clone(),
            created_at: n.created_at,
            access_count: n.reps as u32,
        })
        .collect();

    // Run dream through CognitiveEngine
    let cog = cognitive.lock().await;
    // Capture start time before the dream — composite-score eviction in store_connections
    // reorders the buffer, making positional slicing (pre_dream_count..) unreliable.
    let dream_start = Utc::now();
    let dream_result = cog.dreamer.dream(&dream_memories).await;
    let insights = cog.dreamer.synthesize_insights(&dream_memories);
    let all_connections = cog.dreamer.get_connections();
    drop(cog);

    // Persist new connections
    // Filter by timestamp — same approach as dream.rs to avoid positional index issues.
    let new_connections: Vec<&vestige_core::DiscoveredConnection> = all_connections
        .iter()
        .filter(|c| c.discovered_at >= dream_start)
        .collect();
    let mut connections_persisted = 0u64;
    let now = Utc::now();
    for conn in new_connections.iter() {
        let link_type = match conn.connection_type {
            vestige_core::DiscoveredConnectionType::Semantic => "semantic",
            vestige_core::DiscoveredConnectionType::SharedConcept => "shared_concepts",
            vestige_core::DiscoveredConnectionType::Temporal => "temporal",
            vestige_core::DiscoveredConnectionType::Complementary => "complementary",
            vestige_core::DiscoveredConnectionType::CausalChain => "causal",
        };
        let record = vestige_core::ConnectionRecord {
            source_id: conn.from_id.clone(),
            target_id: conn.to_id.clone(),
            strength: conn.similarity,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 1,
        };
        if state.storage.save_connection(&record).is_ok() {
            connections_persisted += 1;
        }

        // Emit connection events
        state.emit(VestigeEvent::ConnectionDiscovered {
            source_id: conn.from_id.clone(),
            target_id: conn.to_id.clone(),
            connection_type: link_type.to_string(),
            weight: conn.similarity,
            timestamp: now,
        });
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    // Emit completion event
    state.emit(VestigeEvent::DreamCompleted {
        memories_replayed: dream_memories.len(),
        connections_found: connections_persisted as usize,
        insights_generated: insights.len(),
        duration_ms,
        timestamp: Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "status": "dreamed",
        "memoriesReplayed": dream_memories.len(),
        "connectionsPersisted": connections_persisted,
        "insights": insights.iter().map(|i| serde_json::json!({
            "type": format!("{:?}", i.insight_type),
            "insight": i.insight,
            "sourceMemories": i.source_memories,
            "confidence": i.confidence,
            "noveltyScore": i.novelty_score,
        })).collect::<Vec<Value>>(),
        "stats": {
            "newConnectionsFound": dream_result.new_connections_found,
            "connectionsPersisted": connections_persisted,
            "memoriesStrengthened": dream_result.memories_strengthened,
            "memoriesCompressed": dream_result.memories_compressed,
            "insightsGenerated": dream_result.insights_generated.len(),
            "durationMs": duration_ms,
        }
    })))
}

#[derive(Debug, Deserialize)]
pub struct ExploreRequest {
    pub from_id: String,
    pub to_id: Option<String>,
    pub action: Option<String>, // "associations", "chains", "bridges"
    pub limit: Option<usize>,
}

/// Explore connections between memories
pub async fn explore_connections(
    State(state): State<AppState>,
    Json(req): Json<ExploreRequest>,
) -> Result<Json<Value>, StatusCode> {
    let action = req.action.as_deref().unwrap_or("associations");
    let limit = req.limit.unwrap_or(10).clamp(1, 50);

    match action {
        "associations" => {
            // Get the source memory content for similarity search
            let source_node = state
                .storage
                .get_node(&req.from_id)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .ok_or(StatusCode::NOT_FOUND)?;

            // Use hybrid search with source content to find associated memories
            let results = state
                .storage
                .hybrid_search(&source_node.content, limit as i32, 0.3, 0.7)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let formatted: Vec<Value> = results
                .iter()
                .filter(|r| r.node.id != req.from_id) // Exclude self
                .map(|r| {
                    serde_json::json!({
                        "id": r.node.id,
                        "content": r.node.content,
                        "nodeType": r.node.node_type,
                        "score": r.combined_score,
                        "retention": r.node.retention_strength,
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({
                "action": "associations",
                "fromId": req.from_id,
                "results": formatted,
            })))
        }
        "chains" | "bridges" => {
            let to_id = req.to_id.as_deref().ok_or(StatusCode::BAD_REQUEST)?;

            let (nodes, edges) = state
                .storage
                .get_memory_subgraph(&req.from_id, 2, limit)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let nodes_json: Vec<Value> = nodes
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "id": n.id,
                        "content": n.content.chars().take(100).collect::<String>(),
                        "nodeType": n.node_type,
                        "retention": n.retention_strength,
                    })
                })
                .collect();

            let edges_json: Vec<Value> = edges
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "source": e.source_id,
                        "target": e.target_id,
                        "weight": e.strength,
                        "type": e.link_type,
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({
                "action": action,
                "fromId": req.from_id,
                "toId": to_id,
                "nodes": nodes_json,
                "edges": edges_json,
            })))
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

/// Predict which memories will be needed
pub async fn predict_memories(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    // Get recent memories as predictions based on activity
    let recent = state
        .storage
        .get_all_nodes(10, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let predictions: Vec<Value> = recent
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content.chars().take(100).collect::<String>(),
                "nodeType": n.node_type,
                "retention": n.retention_strength,
                "predictedNeed": "high",
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "predictions": predictions,
        "basedOn": "recent_activity",
    })))
}

#[derive(Debug, Deserialize)]
pub struct ImportanceRequest {
    pub content: String,
}

/// Score content importance using 4-channel model
pub async fn score_importance(
    State(state): State<AppState>,
    Json(req): Json<ImportanceRequest>,
) -> Result<Json<Value>, StatusCode> {
    if let Some(ref cognitive) = state.cognitive {
        let context = vestige_core::ImportanceContext::current();
        let cog = cognitive.lock().await;
        let score = cog
            .importance_signals
            .compute_importance(&req.content, &context);
        drop(cog);

        let composite = score.composite;
        let novelty = score.novelty;
        let arousal = score.arousal;
        let reward = score.reward;
        let attention = score.attention;

        state.emit(VestigeEvent::ImportanceScored {
            memory_id: None, // /api/importance scores arbitrary content, not a stored memory
            content_preview: req.content.chars().take(80).collect(),
            composite_score: composite,
            novelty,
            arousal,
            reward,
            attention,
            timestamp: Utc::now(),
        });

        Ok(Json(serde_json::json!({
            "composite": composite,
            "channels": {
                "novelty": novelty,
                "arousal": arousal,
                "reward": reward,
                "attention": attention,
            },
            "recommendation": if composite > 0.6 { "save" } else { "skip" },
        })))
    } else {
        // Fallback: basic heuristic scoring
        let word_count = req.content.split_whitespace().count();
        let has_code = req.content.contains("```") || req.content.contains("fn ");
        let composite = if has_code {
            0.7
        } else {
            (word_count as f64 / 100.0).min(0.8)
        };

        Ok(Json(serde_json::json!({
            "composite": composite,
            "channels": {
                "novelty": composite,
                "arousal": 0.5,
                "reward": 0.5,
                "attention": composite,
            },
            "recommendation": if composite > 0.6 { "save" } else { "skip" },
        })))
    }
}

/// Trigger consolidation
pub async fn trigger_consolidation(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    state.emit(VestigeEvent::ConsolidationStarted {
        timestamp: Utc::now(),
    });

    let start = std::time::Instant::now();

    let result = state
        .storage
        .run_consolidation()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let duration_ms = start.elapsed().as_millis() as u64;

    state.emit(VestigeEvent::ConsolidationCompleted {
        nodes_processed: result.nodes_processed as usize,
        decay_applied: result.decay_applied as usize,
        embeddings_generated: result.embeddings_generated as usize,
        duration_ms,
        timestamp: Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "nodesProcessed": result.nodes_processed,
        "decayApplied": result.decay_applied,
        "embeddingsGenerated": result.embeddings_generated,
        "duplicatesMerged": result.duplicates_merged,
        "activationsComputed": result.activations_computed,
        "durationMs": duration_ms,
    })))
}

/// Get retention distribution (for histogram visualization)
pub async fn retention_distribution(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    // Cap at 1000 to prevent excessive memory usage on large databases
    let nodes = state
        .storage
        .get_all_nodes(1000, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Build distribution buckets
    let mut buckets = [0u32; 10]; // 0-10%, 10-20%, ..., 90-100%
    let mut by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut endangered = Vec::new();

    for node in &nodes {
        let bucket = ((node.retention_strength * 10.0).floor() as usize).min(9);
        buckets[bucket] += 1;
        *by_type.entry(node.node_type.clone()).or_default() += 1;

        // Endangered: retention below 30%
        if node.retention_strength < 0.3 {
            endangered.push(serde_json::json!({
                "id": node.id,
                "content": node.content.chars().take(60).collect::<String>(),
                "retention": node.retention_strength,
                "nodeType": node.node_type,
            }));
        }
    }

    let distribution: Vec<Value> = buckets
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            serde_json::json!({
                "range": format!("{}-{}%", i * 10, (i + 1) * 10),
                "count": count,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "distribution": distribution,
        "byType": by_type,
        "endangered": endangered,
        "total": nodes.len(),
    })))
}

// ============================================================================
// INTENTIONS (v2.0)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IntentionListParams {
    pub status: Option<String>,
}

/// List intentions
pub async fn list_intentions(
    State(state): State<AppState>,
    Query(params): Query<IntentionListParams>,
) -> Result<Json<Value>, StatusCode> {
    let status_filter = params.status.unwrap_or_else(|| "active".to_string());

    let intentions = if status_filter == "all" {
        // Get all statuses
        let mut all = state.storage.get_active_intentions().unwrap_or_default();
        all.extend(
            state
                .storage
                .get_intentions_by_status("fulfilled")
                .unwrap_or_default(),
        );
        all.extend(
            state
                .storage
                .get_intentions_by_status("cancelled")
                .unwrap_or_default(),
        );
        all.extend(
            state
                .storage
                .get_intentions_by_status("snoozed")
                .unwrap_or_default(),
        );
        all
    } else if status_filter == "active" {
        state
            .storage
            .get_active_intentions()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        state
            .storage
            .get_intentions_by_status(&status_filter)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let count = intentions.len();
    Ok(Json(serde_json::json!({
        "intentions": intentions,
        "total": count,
        "filter": status_filter,
    })))
}

// ============================================================================
// DEEP REFERENCE (Reasoning Theater, v2.0.8)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DeepReferenceBody {
    pub query: String,
    pub depth: Option<i32>,
}

/// Run the 8-stage deep_reference cognitive pipeline over HTTP.
///
/// Wraps the existing `crate::tools::cross_reference::execute` tool so the
/// dashboard can surface the same reasoning chain the MCP clients see. Emits
/// a `DeepReferenceCompleted` event with primary + supporting + contradicting
/// memory IDs so Graph3D can camera-glide, pulse evidence nodes, and draw
/// contradiction arcs in the 3D scene.
pub async fn deep_reference_query(
    State(state): State<AppState>,
    Json(body): Json<DeepReferenceBody>,
) -> Result<Json<Value>, StatusCode> {
    let cognitive = state
        .cognitive
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    if body.query.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let args = serde_json::json!({
        "query": body.query.clone(),
        "depth": body.depth.unwrap_or(20).clamp(5, 50),
    });

    let start = std::time::Instant::now();
    let response = crate::tools::cross_reference::execute(&state.storage, cognitive, Some(args))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let duration_ms = start.elapsed().as_millis() as u64;

    // Pull evidence IDs out for the WebSocket event so Graph3D can glide,
    // pulse, and arc. We intentionally read from the serialized JSON rather
    // than re-running the pipeline — whatever the tool decided is what the
    // Theater visualizes.
    let primary_id = response
        .get("recommended")
        .and_then(|r| r.get("memory_id"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let supporting_ids: Vec<String> = response
        .get("evidence")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let role = e.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    if role == "supporting" || role == "primary" {
                        e.get("id").and_then(|v| v.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let contradicting_ids: Vec<String> = response
        .get("evidence")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let role = e.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    if role == "contradicting" {
                        e.get("id").and_then(|v| v.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let contradiction_pairs: Vec<(String, String)> = response
        .get("contradictions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let a = c.get("a_id").and_then(|v| v.as_str())?.to_string();
                    let b = c.get("b_id").and_then(|v| v.as_str())?.to_string();
                    Some((a, b))
                })
                .collect()
        })
        .unwrap_or_default();

    let memories_analyzed = response
        .get("memoriesAnalyzed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let confidence = response
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let intent = response
        .get("intent")
        .and_then(|v| v.as_str())
        .unwrap_or("Synthesis")
        .to_string();

    let status = response
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    state.emit(VestigeEvent::DeepReferenceCompleted {
        query: body.query,
        intent,
        status,
        confidence,
        primary_id,
        supporting_ids,
        contradicting_ids,
        contradiction_pairs,
        memories_analyzed,
        duration_ms,
        timestamp: Utc::now(),
    });

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::Arc;
    use tempfile::tempdir;
    use vestige_core::memory::IngestInput;
    use vestige_core::{ConnectionRecord, Storage};

    #[test]
    fn graph_sort_parse_defaults_to_recent() {
        assert_eq!(GraphSort::parse(None), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("recent")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("RECENT")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("Recent")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("garbage")), GraphSort::Recent);
    }

    #[test]
    fn graph_sort_parse_accepts_connected_case_insensitive() {
        assert_eq!(GraphSort::parse(Some("connected")), GraphSort::Connected);
        assert_eq!(GraphSort::parse(Some("CONNECTED")), GraphSort::Connected);
        assert_eq!(GraphSort::parse(Some("Connected")), GraphSort::Connected);
    }

    fn seed_storage() -> (tempfile::TempDir, Arc<Storage>) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = Arc::new(Storage::new(Some(db_path)).unwrap());
        (dir, storage)
    }

    fn ingest(storage: &Storage, content: &str) -> String {
        let node = storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        node.id
    }

    #[test]
    fn default_center_id_recent_returns_newest_node() {
        let (_dir, storage) = seed_storage();
        ingest(&storage, "first");
        ingest(&storage, "second");
        let newest = ingest(&storage, "third");

        let center = default_center_id(&storage, GraphSort::Recent).unwrap();
        assert_eq!(
            center, newest,
            "Recent mode should pick the newest ingested memory"
        );
    }

    fn link(storage: &Storage, source: &str, target: &str) {
        let now = Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 0.9,
                link_type: "semantic".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    #[test]
    fn default_center_id_connected_prefers_hub_over_newest() {
        let (_dir, storage) = seed_storage();
        let hub = ingest(&storage, "hub node");
        let spoke_a = ingest(&storage, "spoke A");
        let spoke_b = ingest(&storage, "spoke B");
        let spoke_c = ingest(&storage, "spoke C");
        // Wire the spokes into `hub` so it has the most connections. Leave
        // the final `lonely` node unconnected — it's the newest by
        // insertion order and would win in Recent mode.
        for spoke in [&spoke_a, &spoke_b, &spoke_c] {
            link(&storage, &hub, spoke);
        }
        let _lonely = ingest(&storage, "lonely newcomer");

        let center = default_center_id(&storage, GraphSort::Connected).unwrap();
        assert_eq!(
            center, hub,
            "Connected mode should pick the densest node, not the newest"
        );
    }

    #[test]
    fn default_center_id_connected_falls_back_to_recent_when_no_edges() {
        let (_dir, storage) = seed_storage();
        ingest(&storage, "alpha");
        let newest = ingest(&storage, "beta");

        // No connections exist — Connected mode should degrade to Recent
        // rather than returning 404.
        let center = default_center_id(&storage, GraphSort::Connected).unwrap();
        assert_eq!(center, newest);
    }

    #[test]
    fn default_center_id_returns_not_found_on_empty_db() {
        let (_dir, storage) = seed_storage();

        let recent_err = default_center_id(&storage, GraphSort::Recent).unwrap_err();
        assert_eq!(recent_err, StatusCode::NOT_FOUND);

        let connected_err = default_center_id(&storage, GraphSort::Connected).unwrap_err();
        assert_eq!(connected_err, StatusCode::NOT_FOUND);
    }
}
