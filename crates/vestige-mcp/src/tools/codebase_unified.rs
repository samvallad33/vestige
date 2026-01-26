//! Unified Codebase Tool
//!
//! Merges remember_pattern, remember_decision, and get_codebase_context into a single
//! `codebase` tool with action-based dispatch.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{IngestInput, Storage};

/// Input schema for the unified codebase tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember_pattern", "remember_decision", "get_context"],
                "description": "Action to perform: 'remember_pattern' stores a code pattern, 'remember_decision' stores an architectural decision, 'get_context' retrieves patterns and decisions for a codebase"
            },
            // remember_pattern fields
            "name": {
                "type": "string",
                "description": "Name/title for the pattern (required for remember_pattern)"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the pattern (required for remember_pattern)"
            },
            // remember_decision fields
            "decision": {
                "type": "string",
                "description": "The architectural or design decision made (required for remember_decision)"
            },
            "rationale": {
                "type": "string",
                "description": "Why this decision was made (required for remember_decision)"
            },
            "alternatives": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alternatives that were considered (optional for remember_decision)"
            },
            // Shared fields
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files where this pattern is used or affected by this decision"
            },
            "codebase": {
                "type": "string",
                "description": "Codebase/project identifier (e.g., 'vestige-tauri')"
            },
            // get_context fields
            "limit": {
                "type": "integer",
                "description": "Maximum items per category (default: 10, for get_context)",
                "default": 10
            }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodebaseArgs {
    action: String,
    // Pattern fields
    name: Option<String>,
    description: Option<String>,
    // Decision fields
    decision: Option<String>,
    rationale: Option<String>,
    alternatives: Option<Vec<String>>,
    // Shared fields
    files: Option<Vec<String>>,
    codebase: Option<String>,
    // Context fields
    limit: Option<i32>,
}

/// Execute the unified codebase tool
pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: CodebaseArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    match args.action.as_str() {
        "remember_pattern" => execute_remember_pattern(storage, &args).await,
        "remember_decision" => execute_remember_decision(storage, &args).await,
        "get_context" => execute_get_context(storage, &args).await,
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: remember_pattern, remember_decision, get_context",
            args.action
        )),
    }
}

/// Remember a code pattern
async fn execute_remember_pattern(
    storage: &Arc<Mutex<Storage>>,
    args: &CodebaseArgs,
) -> Result<Value, String> {
    let name = args
        .name
        .as_ref()
        .ok_or("'name' is required for remember_pattern action")?;
    let description = args
        .description
        .as_ref()
        .ok_or("'description' is required for remember_pattern action")?;

    if name.trim().is_empty() {
        return Err("Pattern name cannot be empty".to_string());
    }

    // Build content with structured format
    let mut content = format!("# Code Pattern: {}\n\n{}", name, description);

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
        "action": "remember_pattern",
        "success": true,
        "nodeId": node.id,
        "patternName": name,
        "message": format!("Pattern '{}' remembered successfully", name),
    }))
}

/// Remember an architectural decision
async fn execute_remember_decision(
    storage: &Arc<Mutex<Storage>>,
    args: &CodebaseArgs,
) -> Result<Value, String> {
    let decision = args
        .decision
        .as_ref()
        .ok_or("'decision' is required for remember_decision action")?;
    let rationale = args
        .rationale
        .as_ref()
        .ok_or("'rationale' is required for remember_decision action")?;

    if decision.trim().is_empty() {
        return Err("Decision cannot be empty".to_string());
    }

    // Build content with structured format (ADR-like)
    let mut content = format!(
        "# Decision: {}\n\n## Context\n\n{}\n\n## Decision\n\n{}",
        &decision[..decision.len().min(50)],
        rationale,
        decision
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
    let mut tags = vec![
        "decision".to_string(),
        "architecture".to_string(),
        "codebase".to_string(),
    ];
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
        "action": "remember_decision",
        "success": true,
        "nodeId": node.id,
        "message": "Architectural decision remembered successfully",
    }))
}

/// Get codebase context (patterns and decisions)
async fn execute_get_context(
    storage: &Arc<Mutex<Storage>>,
    args: &CodebaseArgs,
) -> Result<Value, String> {
    let limit = args.limit.unwrap_or(10).clamp(1, 50);
    let storage = storage.lock().await;

    // Build tag filter for codebase
    // Tags are stored as: ["pattern", "codebase", "codebase:vestige"]
    // We search for the "codebase:{name}" tag
    let tag_filter = args
        .codebase
        .as_ref()
        .map(|cb| format!("codebase:{}", cb));

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
        "action": "get_context",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_structure() {
        let schema = schema();
        assert!(schema["properties"]["action"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["action"]));

        // Check action enum values
        let action_enum = &schema["properties"]["action"]["enum"];
        assert!(action_enum
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("remember_pattern")));
        assert!(action_enum
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("remember_decision")));
        assert!(action_enum
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("get_context")));
    }
}
