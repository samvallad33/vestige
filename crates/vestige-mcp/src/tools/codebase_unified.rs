//! Unified Codebase Tool
//!
//! Merges remember_pattern, remember_decision, and get_codebase_context into a single
//! `codebase` tool with action-based dispatch.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{IngestInput, OutputConfig, Storage};

use super::search_unified::apply_output_masks;

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
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: CodebaseArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    match args.action.as_str() {
        "remember_pattern" => execute_remember_pattern(storage, cognitive, &args).await,
        "remember_decision" => execute_remember_decision(storage, cognitive, &args).await,
        "get_context" => execute_get_context(storage, cognitive, output_config, &args).await,
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: remember_pattern, remember_decision, get_context",
            args.action
        )),
    }
}

/// Remember a code pattern
async fn execute_remember_pattern(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
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

    if let Some(ref files) = args.files
        && !files.is_empty()
    {
        content.push_str("\n\n## Files:\n");
        for f in files {
            content.push_str(&format!("- {}\n", f));
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

    let node = storage.ingest(input).map_err(|e| e.to_string())?;
    let node_id = node.id.clone();

    // ====================================================================
    // COGNITIVE: Cross-project pattern recording
    // ====================================================================
    if let Ok(cog) = cognitive.try_lock() {
        let codebase_name = args.codebase.as_deref().unwrap_or("default");
        cog.cross_project
            .record_project_memory(&node_id, codebase_name, None);

        // Also index in hippocampal index for fast retrieval
        let _ = cog.hippocampal_index.index_memory(
            &node_id,
            &format!("{}: {}", name, description),
            "pattern",
            chrono::Utc::now(),
            None,
        );
    }

    Ok(serde_json::json!({
        "action": "remember_pattern",
        "success": true,
        "nodeId": node_id,
        "patternName": name,
        "message": format!("Pattern '{}' remembered successfully", name),
    }))
}

/// Remember an architectural decision
async fn execute_remember_decision(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
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
        &decision[..decision.floor_char_boundary(50)],
        rationale,
        decision
    );

    if let Some(ref alternatives) = args.alternatives
        && !alternatives.is_empty()
    {
        content.push_str("\n\n## Alternatives Considered:\n");
        for alt in alternatives {
            content.push_str(&format!("- {}\n", alt));
        }
    }

    if let Some(ref files) = args.files
        && !files.is_empty()
    {
        content.push_str("\n\n## Affected Files:\n");
        for f in files {
            content.push_str(&format!("- {}\n", f));
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

    let node = storage.ingest(input).map_err(|e| e.to_string())?;
    let node_id = node.id.clone();

    // ====================================================================
    // COGNITIVE: Cross-project decision recording
    // ====================================================================
    if let Ok(cog) = cognitive.try_lock() {
        let codebase_name = args.codebase.as_deref().unwrap_or("default");
        cog.cross_project
            .record_project_memory(&node_id, codebase_name, None);

        // Index in hippocampal index
        let _ = cog.hippocampal_index.index_memory(
            &node_id,
            &format!("Decision: {}", decision),
            "decision",
            chrono::Utc::now(),
            None,
        );
    }

    Ok(serde_json::json!({
        "action": "remember_decision",
        "success": true,
        "nodeId": node_id,
        "message": "Architectural decision remembered successfully",
    }))
}

/// Get codebase context (patterns and decisions)
async fn execute_get_context(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: &CodebaseArgs,
) -> Result<Value, String> {
    // Precedence: explicit MCP param > config limit > built-in default (10).
    let limit = output_config.resolve_limit(args.limit, 10).clamp(1, 50);

    // Build tag filter for codebase
    let tag_filter = args.codebase.as_ref().map(|cb| format!("codebase:{}", cb));

    // Query patterns by node_type and tag
    let patterns = storage
        .get_nodes_by_type_and_tag("pattern", tag_filter.as_deref(), limit)
        .unwrap_or_default();

    // Query decisions by node_type and tag
    let decisions = storage
        .get_nodes_by_type_and_tag("decision", tag_filter.as_deref(), limit)
        .unwrap_or_default();

    let mut formatted_patterns: Vec<Value> = patterns
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
    apply_output_masks(&mut formatted_patterns, output_config);

    let mut formatted_decisions: Vec<Value> = decisions
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
    apply_output_masks(&mut formatted_decisions, output_config);

    // ====================================================================
    // COGNITIVE: Cross-project knowledge discovery
    // ====================================================================
    let mut universal_patterns = Vec::new();
    if let Some(codebase_name) = &args.codebase
        && let Ok(cog) = cognitive.try_lock()
    {
        let context = vestige_core::advanced::cross_project::ProjectContext {
            path: None,
            name: Some(codebase_name.clone()),
            languages: Vec::new(),
            frameworks: Vec::new(),
            file_types: std::collections::HashSet::new(),
            dependencies: Vec::new(),
            structure: Vec::new(),
        };
        let applicable = cog.cross_project.detect_applicable(&context);
        for knowledge in applicable {
            universal_patterns.push(serde_json::json!({
                "pattern": format!("{:?}", knowledge),
            }));
        }
    }

    Ok(serde_json::json!({
        "action": "get_context",
        "codebase": args.codebase,
        "profile": output_config.profile.as_str(),
        "patterns": {
            "count": formatted_patterns.len(),
            "items": formatted_patterns,
        },
        "decisions": {
            "count": formatted_decisions.len(),
            "items": formatted_decisions,
        },
        "crossProjectInsights": universal_patterns,
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
        assert!(
            action_enum
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("remember_pattern"))
        );
        assert!(
            action_enum
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("remember_decision"))
        );
        assert!(
            action_enum
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("get_context"))
        );
    }

    // === INTEGRATION TESTS ===

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[tokio::test]
    async fn test_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_invalid_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "invalid" });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid action"));
    }

    #[tokio::test]
    async fn test_remember_pattern_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_pattern",
            "name": "Error Handling Pattern",
            "description": "Use Result<T, E> with custom error types",
            "files": ["src/lib.rs"],
            "codebase": "vestige"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "remember_pattern");
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert_eq!(value["patternName"], "Error Handling Pattern");
    }

    #[tokio::test]
    async fn test_remember_pattern_missing_name_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_pattern",
            "description": "Some description"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'name' is required"));
    }

    #[tokio::test]
    async fn test_remember_pattern_missing_description_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_pattern",
            "name": "Test Pattern"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'description' is required"));
    }

    #[tokio::test]
    async fn test_remember_pattern_empty_name_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_pattern",
            "name": "   ",
            "description": "Some description"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_remember_decision_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_decision",
            "decision": "Use SQLite for storage",
            "rationale": "Embedded, no separate server needed",
            "alternatives": ["PostgreSQL", "Redis"],
            "files": ["src/storage.rs"],
            "codebase": "vestige"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "remember_decision");
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
    }

    #[tokio::test]
    async fn test_remember_decision_missing_decision_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_decision",
            "rationale": "Some rationale"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'decision' is required"));
    }

    #[tokio::test]
    async fn test_remember_decision_missing_rationale_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_decision",
            "decision": "Use SQLite"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'rationale' is required"));
    }

    #[tokio::test]
    async fn test_remember_decision_empty_decision_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "remember_decision",
            "decision": "  ",
            "rationale": "Something"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_get_context_empty() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "get_context",
            "codebase": "nonexistent"
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "get_context");
        assert_eq!(value["patterns"]["count"], 0);
        assert_eq!(value["decisions"]["count"], 0);
    }

    #[tokio::test]
    async fn test_get_context_retrieves_saved_patterns() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        // Save a pattern first
        let save_args = serde_json::json!({
            "action": "remember_pattern",
            "name": "Test Pattern",
            "description": "A test pattern",
            "codebase": "myproject"
        });
        execute(&storage, &cog, &OutputConfig::default(), Some(save_args)).await.unwrap();

        // Now retrieve
        let get_args = serde_json::json!({
            "action": "get_context",
            "codebase": "myproject"
        });
        let result = execute(&storage, &cog, &OutputConfig::default(), Some(get_args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value["patterns"]["count"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_get_context_no_codebase() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "get_context" });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "get_context");
        assert!(value["codebase"].is_null());
    }

    /// Phase 2: the `lean` profile masks the `createdAt` timestamp from
    /// get_context items, and the response echoes the active profile.
    #[tokio::test]
    async fn test_get_context_lean_profile_masks_timestamps() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let save_args = serde_json::json!({
            "action": "remember_pattern",
            "name": "Lean Pattern",
            "description": "A pattern for lean masking",
            "codebase": "leanproj"
        });
        execute(&storage, &cog, &OutputConfig::default(), Some(save_args))
            .await
            .unwrap();

        let cfg = vestige_core::VestigeConfig::parse("[defaults]\nprofile=lean").output();
        let get_args = serde_json::json!({ "action": "get_context", "codebase": "leanproj" });
        let value = execute(&storage, &cog, &cfg, Some(get_args)).await.unwrap();
        assert_eq!(value["profile"], "lean");
        let item = &value["patterns"]["items"][0];
        assert!(item.get("createdAt").is_none(), "lean must drop createdAt");
        assert!(item.get("content").is_some(), "content still present");
    }
}
