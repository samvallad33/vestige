#![allow(dead_code)]
//! Search Tools (Deprecated - use search_unified instead)
//!
//! Semantic and hybrid search implementations.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

/// Input schema for semantic_search tool
pub fn semantic_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for semantic similarity"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 50
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity threshold (0.0-1.0, default: 0.5)",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["query"]
    })
}

/// Input schema for hybrid_search tool
pub fn hybrid_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 50
            },
            "keyword_weight": {
                "type": "number",
                "description": "Weight for keyword search (0.0-1.0, default: 0.5)",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "semantic_weight": {
                "type": "number",
                "description": "Weight for semantic search (0.0-1.0, default: 0.5)",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticSearchArgs {
    query: String,
    limit: Option<i32>,
    min_similarity: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HybridSearchArgs {
    query: String,
    limit: Option<i32>,
    keyword_weight: Option<f32>,
    semantic_weight: Option<f32>,
}

pub async fn execute_semantic(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SemanticSearchArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let storage = storage.lock().await;

    // Check if embeddings are ready
    if !storage.is_embedding_ready() {
        return Ok(serde_json::json!({
            "error": "Embedding service not ready",
            "hint": "Run consolidation first to initialize embeddings, or the model may still be loading.",
        }));
    }

    let results = storage
        .semantic_search(
            &args.query,
            args.limit.unwrap_or(10).clamp(1, 50),
            args.min_similarity.unwrap_or(0.5).clamp(0.0, 1.0),
        )
        .map_err(|e| e.to_string())?;

    let formatted: Vec<Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "similarity": r.similarity,
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "query": args.query,
        "method": "semantic",
        "total": formatted.len(),
        "results": formatted,
    }))
}

pub async fn execute_hybrid(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: HybridSearchArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let storage = storage.lock().await;

    let results = storage
        .hybrid_search(
            &args.query,
            args.limit.unwrap_or(10).clamp(1, 50),
            args.keyword_weight.unwrap_or(0.5).clamp(0.0, 1.0),
            args.semantic_weight.unwrap_or(0.5).clamp(0.0, 1.0),
        )
        .map_err(|e| e.to_string())?;

    let formatted: Vec<Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "combinedScore": r.combined_score,
                "keywordScore": r.keyword_score,
                "semanticScore": r.semantic_score,
                "matchType": format!("{:?}", r.match_type),
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "query": args.query,
        "method": "hybrid",
        "total": formatted.len(),
        "results": formatted,
    }))
}
