#![allow(dead_code)]
//! Context-Dependent Memory Tool (Deprecated)
//!
//! Retrieval based on encoding context match.
//! Based on Tulving & Thomson's Encoding Specificity Principle (1973).

use chrono::Utc;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{RecallInput, SearchMode, Storage};

/// Input schema for match_context tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for content matching"
            },
            "topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Active topics in current context"
            },
            "project": {
                "type": "string",
                "description": "Current project name"
            },
            "mood": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Current emotional state"
            },
            "time_weight": {
                "type": "number",
                "description": "Weight for temporal context (0.0-1.0, default: 0.3)"
            },
            "topic_weight": {
                "type": "number",
                "description": "Weight for topical context (0.0-1.0, default: 0.4)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 10)"
            }
        },
        "required": ["query"]
    })
}

pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args = args.ok_or("Missing arguments")?;

    let query = args["query"]
        .as_str()
        .ok_or("query is required")?;

    let topics: Vec<String> = args["topics"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let project = args["project"].as_str().map(String::from);
    let mood = args["mood"].as_str().unwrap_or("neutral");

    let time_weight = args["time_weight"].as_f64().unwrap_or(0.3);
    let topic_weight = args["topic_weight"].as_f64().unwrap_or(0.4);

    let limit = args["limit"].as_i64().unwrap_or(10) as i32;

    let storage = storage.lock().await;
    let now = Utc::now();

    // Get candidate memories
    let recall_input = RecallInput {
        query: query.to_string(),
        limit: limit * 2, // Get more, then filter
        min_retention: 0.0,
        search_mode: SearchMode::Hybrid,
        valid_at: None,
    };
    let candidates = storage.recall(recall_input)
        .map_err(|e| e.to_string())?;

    // Score by context match (simplified implementation)
    let mut scored_results: Vec<_> = candidates.into_iter()
        .map(|mem| {
            // Calculate context score based on:
            // 1. Temporal proximity (how recent)
            let hours_ago = (now - mem.created_at).num_hours() as f64;
            let temporal_score = 1.0 / (1.0 + hours_ago / 24.0); // Decay over days

            // 2. Tag overlap with topics
            let tag_overlap = if topics.is_empty() {
                0.5 // Neutral if no topics specified
            } else {
                let matching = mem.tags.iter()
                    .filter(|t| topics.iter().any(|topic| topic.to_lowercase().contains(&t.to_lowercase())))
                    .count();
                matching as f64 / topics.len().max(1) as f64
            };

            // 3. Project match
            let project_score = match (&project, &mem.source) {
                (Some(p), Some(s)) if s.to_lowercase().contains(&p.to_lowercase()) => 1.0,
                (Some(_), None) => 0.0,
                (None, _) => 0.5,
                _ => 0.3,
            };

            // 4. Emotional match (simplified)
            let mood_score = match mood {
                "positive" if mem.sentiment_score > 0.0 => 0.8,
                "negative" if mem.sentiment_score < 0.0 => 0.8,
                "neutral" if mem.sentiment_score.abs() < 0.3 => 0.8,
                _ => 0.5,
            };

            // Combine scores
            let context_score = temporal_score * time_weight
                + tag_overlap * topic_weight
                + project_score * 0.2
                + mood_score * 0.1;

            let combined_score = mem.retention_strength * 0.5 + context_score * 0.5;

            (mem, context_score, combined_score)
        })
        .collect();

    // Sort by combined score (handle NaN safely)
    scored_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored_results.truncate(limit as usize);

    let results: Vec<Value> = scored_results.into_iter()
        .map(|(mem, ctx_score, combined)| {
            serde_json::json!({
                "id": mem.id,
                "content": mem.content,
                "retentionStrength": mem.retention_strength,
                "contextScore": ctx_score,
                "combinedScore": combined,
                "tags": mem.tags,
                "createdAt": mem.created_at.to_rfc3339()
            })
        })
        .collect();

    Ok(serde_json::json!({
        "success": true,
        "query": query,
        "currentContext": {
            "topics": topics,
            "project": project,
            "mood": mood
        },
        "weights": {
            "temporal": time_weight,
            "topical": topic_weight
        },
        "resultCount": results.len(),
        "results": results,
        "science": {
            "theory": "Encoding Specificity Principle (Tulving & Thomson, 1973)",
            "principle": "Memory retrieval is most effective when retrieval context matches encoding context"
        }
    }))
}
