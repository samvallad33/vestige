#![allow(dead_code)]
//! Codebase Tools (Deprecated - use codebase_unified instead)
//!
//! Remember patterns, decisions, and context about codebases.
//! This is a differentiating feature for AI-assisted development.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{IngestInput, Storage};

/// Input schema for remember_pattern tool
pub fn pattern_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name/title for this pattern"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the pattern"
            },
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files where this pattern is used"
            },
            "codebase": {
                "type": "string",
                "description": "Codebase/project identifier (e.g., 'vestige-tauri')"
            }
        },
        "required": ["name", "description"]
    })
}

/// Input schema for remember_decision tool
pub fn decision_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "description": "The architectural or design decision made"
            },
            "rationale": {
                "type": "string",
                "description": "Why this decision was made"
            },
            "alternatives": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alternatives that were considered"
            },
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files affected by this decision"
            },
            "codebase": {
                "type": "string",
                "description": "Codebase/project identifier"
            }
        },
        "required": ["decision", "rationale"]
    })
}

/// Input schema for get_codebase_context tool
pub fn context_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "codebase": {
                "type": "string",
                "description": "Codebase/project identifier to get context for"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum items per category (default: 10)",
                "default": 10
            }
        },
        "required": []
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PatternArgs {
    name: String,
    description: String,
    files: Option<Vec<String>>,
    codebase: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DecisionArgs {
    decision: String,
    rationale: String,
    alternatives: Option<Vec<String>>,
    files: Option<Vec<String>>,
    codebase: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContextArgs {
    codebase: Option<String>,
    limit: Option<i32>,
}

pub async fn execute_pattern(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: PatternArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.name.trim().is_empty() {
        return Err("Pattern name cannot be empty".to_string());
    }

    // Build content with structured format
    let mut content = format!("# Code Pattern: {}\n\n{}", args.name, args.description);

    if let Some(ref files) = args.files {
        if !files.is_empty() {
            content.push_str("\n\n## Files:\n");
            for f in files {
                content.push_str(&format!("- {}\n", f));
            }
        }
    }

    // Build tags
    let mut tags = vec!["pattern".to_string(), "codebase".to_string()];
    if let Some(ref codebase) = args.codebase {
        tags.push(format!("codebase:{}", codebase));
    }

    let input = IngestInput {
        content,
        node_type: "pattern".to_string(),
        source: args.codebase.clone(),
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags,
        valid_from: None,
        valid_until: None,
    };

    let mut storage = storage.lock().await;
    let node = storage.ingest(input).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": true,
        "nodeId": node.id,
        "patternName": args.name,
        "message": format!("Pattern '{}' remembered successfully", args.name),
    }))
}

pub async fn execute_decision(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: DecisionArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.decision.trim().is_empty() {
        return Err("Decision cannot be empty".to_string());
    }

    // Build content with structured format (ADR-like)
    let mut content = format!(
        "# Decision: {}\n\n## Context\n\n{}\n\n## Decision\n\n{}",
        &args.decision[..args.decision.len().min(50)],
        args.rationale,
        args.decision
    );

    if let Some(ref alternatives) = args.alternatives {
        if !alternatives.is_empty() {
            content.push_str("\n\n## Alternatives Considered:\n");
            for alt in alternatives {
                content.push_str(&format!("- {}\n", alt));
            }
        }
    }

    if let Some(ref files) = args.files {
        if !files.is_empty() {
            content.push_str("\n\n## Affected Files:\n");
            for f in files {
                content.push_str(&format!("- {}\n", f));
            }
        }
    }

    // Build tags
    let mut tags = vec!["decision".to_string(), "architecture".to_string(), "codebase".to_string()];
    if let Some(ref codebase) = args.codebase {
        tags.push(format!("codebase:{}", codebase));
    }

    let input = IngestInput {
        content,
        node_type: "decision".to_string(),
        source: args.codebase.clone(),
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags,
        valid_from: None,
        valid_until: None,
    };

    let mut storage = storage.lock().await;
    let node = storage.ingest(input).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": true,
        "nodeId": node.id,
        "message": "Architectural decision remembered successfully",
    }))
}

pub async fn execute_context(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: ContextArgs = args
        .map(|v| serde_json::from_value(v))
        .transpose()
        .map_err(|e| format!("Invalid arguments: {}", e))?
        .unwrap_or(ContextArgs {
            codebase: None,
            limit: Some(10),
        });

    let limit = args.limit.unwrap_or(10).clamp(1, 50);
    let storage = storage.lock().await;

    // Build tag filter for codebase
    // Tags are stored as: ["pattern", "codebase", "codebase:vestige"]
    // We search for the "codebase:{name}" tag
    let tag_filter = args.codebase.as_ref().map(|cb| format!("codebase:{}", cb));

    // Query patterns by node_type and tag
    let patterns = storage
        .get_nodes_by_type_and_tag("pattern", tag_filter.as_deref(), limit)
        .unwrap_or_default();

    // Query decisions by node_type and tag
    let decisions = storage
        .get_nodes_by_type_and_tag("decision", tag_filter.as_deref(), limit)
        .unwrap_or_default();

    let formatted_patterns: Vec<Value> = patterns
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content,
                "tags": n.tags,
                "retentionStrength": n.retention_strength,
                "createdAt": n.created_at.to_rfc3339(),
            })
        })
        .collect();

    let formatted_decisions: Vec<Value> = decisions
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content,
                "tags": n.tags,
                "retentionStrength": n.retention_strength,
                "createdAt": n.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "codebase": args.codebase,
        "patterns": {
            "count": formatted_patterns.len(),
            "items": formatted_patterns,
        },
        "decisions": {
            "count": formatted_decisions.len(),
            "items": formatted_decisions,
        },
    }))
}
