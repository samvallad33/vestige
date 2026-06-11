//! Session Context Tool — One-call session initialization (v1.8.0)
//!
//! Combines search, intentions, status, predictions, and codebase context
//! into a single token-budgeted response. Replaces 5 separate calls at
//! session start (~15K tokens → ~500-1000 tokens).

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::cognitive::CognitiveEngine;
use vestige_core::{OutputConfig, Storage};

/// Input schema for session_context tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Search queries to run (default: [\"user preferences\"])"
            },
            "token_budget": {
                "type": "integer",
                "description": "Max tokens for response (default: 1000). Server truncates content to fit budget. With 1M context models, budgets up to 100K are practical.",
                "default": 1000,
                "minimum": 100,
                "maximum": 100000
            },
            "context": {
                "type": "object",
                "description": "Current context for intention matching and predictions",
                "properties": {
                    "codebase": { "type": "string" },
                    "topics": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "file": { "type": "string" }
                }
            },
            "include_status": {
                "type": "boolean",
                "description": "Include system health info (default: true)",
                "default": true
            },
            "include_intentions": {
                "type": "boolean",
                "description": "Include triggered intentions (default: true)",
                "default": true
            },
            "include_predictions": {
                "type": "boolean",
                "description": "Include memory predictions (default: true)",
                "default": true
            }
        }
    })
}

#[derive(Debug, Deserialize, Default)]
struct SessionContextArgs {
    queries: Option<Vec<String>>,
    token_budget: Option<i32>,
    context: Option<ContextSpec>,
    include_status: Option<bool>,
    include_intentions: Option<bool>,
    include_predictions: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ContextSpec {
    codebase: Option<String>,
    topics: Option<Vec<String>>,
    file: Option<String>,
}

/// Extract the first sentence or first line from content, capped at 150 chars.
fn first_sentence(content: &str) -> String {
    let content = content.trim();
    let end = content
        .find(". ")
        .map(|i| i + 1)
        .or_else(|| content.find('\n'))
        .unwrap_or(content.len())
        .min(150);
    // UTF-8 safe boundary
    let end = content.floor_char_boundary(end);
    content[..end].to_string()
}

/// Execute session_context tool — one-call session initialization.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SessionContextArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => SessionContextArgs::default(),
    };

    // Per-query search width honors the active profile (e.g. `research` widens
    // it, `lean` narrows it). No explicit MCP param exists here, so the config
    // limit (or built-in default of 5) applies. Capped to keep the budgeted
    // session response compact.
    let per_query_limit = output_config.resolve_limit(None, 5).clamp(1, 25);
    // The `lean` profile suppresses the inline memory date to save tokens.
    let show_dates = output_config.show_timestamps;

    let token_budget = args.token_budget.unwrap_or(1000).clamp(100, 100000) as usize;
    let budget_chars = token_budget * 4;
    let include_status = args.include_status.unwrap_or(true);
    let include_intentions = args.include_intentions.unwrap_or(true);
    let include_predictions = args.include_predictions.unwrap_or(true);
    let queries = args
        .queries
        .unwrap_or_else(|| vec!["user preferences".to_string()]);

    let mut context_parts: Vec<String> = Vec::new();
    let mut expandable_ids: Vec<String> = Vec::new();
    let mut char_count = 0;

    // ====================================================================
    // 1. Search queries — extract first sentence per result, dedup by ID
    // ====================================================================
    let mut seen_ids = HashSet::new();
    let mut memory_lines: Vec<String> = Vec::new();

    for query in &queries {
        let results = storage
            .hybrid_search(query, per_query_limit, 0.3, 0.7)
            .map_err(|e| e.to_string())?;

        for r in results {
            if seen_ids.contains(&r.node.id) {
                continue;
            }
            let summary = first_sentence(&r.node.content);
            let line = if show_dates {
                let date_str = r.node.updated_at.format("%b %d, %Y").to_string();
                format!("- ({}) {}", date_str, summary)
            } else {
                format!("- {}", summary)
            };
            let line_len = line.len() + 1; // +1 for newline

            if char_count + line_len > budget_chars {
                expandable_ids.push(r.node.id.clone());
            } else {
                memory_lines.push(line);
                char_count += line_len;
            }
            seen_ids.insert(r.node.id.clone());
        }
    }

    // Auto-strengthen accessed memories (Testing Effect)
    let accessed_ids: Vec<&str> = seen_ids.iter().map(|s| s.as_str()).collect();
    let _ = storage.strengthen_batch_on_access(&accessed_ids);

    if !memory_lines.is_empty() {
        context_parts.push(format!("**Memories:**\n{}", memory_lines.join("\n")));
    }

    // ====================================================================
    // 2. Intentions — find triggered + pending high-priority
    // ====================================================================
    if include_intentions {
        let intentions = storage.get_active_intentions().map_err(|e| e.to_string())?;
        let now = Utc::now();
        let mut triggered_lines: Vec<String> = Vec::new();

        for intention in &intentions {
            let is_overdue = intention.deadline.map(|d| d < now).unwrap_or(false);

            // Check context-based triggers
            let is_context_triggered = if let Some(ctx) = &args.context {
                check_intention_triggered(intention, ctx, now)
            } else {
                false
            };

            if is_overdue || is_context_triggered || intention.priority >= 3 {
                let priority_str = match intention.priority {
                    4 => " (critical)",
                    3 => " (high)",
                    _ => "",
                };
                let deadline_str = intention
                    .deadline
                    .map(|d| format!(" [due {}]", d.format("%b %d")))
                    .unwrap_or_default();
                let line = format!(
                    "- {}{}{}",
                    first_sentence(&intention.content),
                    priority_str,
                    deadline_str
                );
                let line_len = line.len() + 1;
                if char_count + line_len <= budget_chars {
                    triggered_lines.push(line);
                    char_count += line_len;
                }
            }
        }

        if !triggered_lines.is_empty() {
            context_parts.push(format!("**Triggered:**\n{}", triggered_lines.join("\n")));
        }
    }

    // ====================================================================
    // 3. System status — compact one-liner
    // ====================================================================
    let stats = storage.get_stats().map_err(|e| e.to_string())?;
    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    // Automation triggers
    let last_dream = storage.get_last_dream().ok().flatten();
    let saves_since_last_dream = match &last_dream {
        Some(dt) => storage.count_memories_since(*dt).unwrap_or(0),
        None => stats.total_nodes,
    };
    let last_backup = storage.last_backup_timestamp();
    let now = Utc::now();

    let needs_dream = last_dream
        .map(|dt| now - dt > Duration::hours(24) || saves_since_last_dream > 50)
        .unwrap_or(true);
    let needs_backup = last_backup
        .map(|dt| now - dt > Duration::days(7))
        .unwrap_or(true);
    let needs_gc = status == "degraded" || status == "critical";

    if include_status {
        let embedding_pct = if stats.total_nodes > 0 {
            (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
        } else {
            0.0
        };
        let status_line = format!(
            "**Status:** {} memories | {} | {:.0}% embeddings",
            stats.total_nodes, status, embedding_pct
        );
        let status_len = status_line.len() + 1;
        if char_count + status_len <= budget_chars {
            context_parts.push(status_line);
            char_count += status_len;
        }

        // Needs line (only if any automation needed)
        let mut needs: Vec<&str> = Vec::new();
        if needs_dream {
            needs.push("dream");
        }
        if needs_backup {
            needs.push("backup");
        }
        if needs_gc {
            needs.push("gc");
        }
        if !needs.is_empty() {
            let needs_line = format!("**Needs:** {}", needs.join(", "));
            let needs_len = needs_line.len() + 1;
            if char_count + needs_len <= budget_chars {
                context_parts.push(needs_line);
                char_count += needs_len;
            }
        }
    }

    // ====================================================================
    // 4. Predictions — top 3 with content preview
    // ====================================================================
    if include_predictions {
        let cog = cognitive.lock().await;

        let session_ctx =
            vestige_core::neuroscience::predictive_retrieval::SessionContext {
                started_at: Utc::now(),
                current_focus: args
                    .context
                    .as_ref()
                    .and_then(|c| c.topics.as_ref())
                    .and_then(|t| t.first())
                    .cloned(),
                active_files: args
                    .context
                    .as_ref()
                    .and_then(|c| c.file.as_ref())
                    .map(|f| vec![f.clone()])
                    .unwrap_or_default(),
                accessed_memories: Vec::new(),
                recent_queries: Vec::new(),
                detected_intent: None,
                project_context: args.context.as_ref().and_then(|c| c.codebase.as_ref()).map(
                    |name| vestige_core::neuroscience::predictive_retrieval::ProjectContext {
                        name: name.to_string(),
                        path: String::new(),
                        technologies: Vec::new(),
                        primary_language: None,
                    },
                ),
            };

        let predictions = cog
            .predictive_memory
            .predict_needed_memories(&session_ctx)
            .unwrap_or_default();

        if !predictions.is_empty() {
            let pred_lines: Vec<String> = predictions
                .iter()
                .take(3)
                .map(|p| {
                    format!(
                        "- {} ({:.0}%)",
                        first_sentence(&p.content_preview),
                        p.confidence * 100.0
                    )
                })
                .collect();

            let pred_section = format!("**Predicted:**\n{}", pred_lines.join("\n"));
            let pred_len = pred_section.len() + 1;
            if char_count + pred_len <= budget_chars {
                context_parts.push(pred_section);
                char_count += pred_len;
            }
        }
    }

    // ====================================================================
    // 5. Codebase patterns/decisions (if codebase specified)
    // ====================================================================
    if let Some(ref ctx) = args.context
        && let Some(ref codebase) = ctx.codebase
    {
        let codebase_tag = format!("codebase:{}", codebase);
        let mut cb_lines: Vec<String> = Vec::new();

        // Get patterns
        if let Ok(patterns) = storage.get_nodes_by_type_and_tag("pattern", Some(&codebase_tag), 3) {
            for p in &patterns {
                let line = format!("- [pattern] {}", first_sentence(&p.content));
                let line_len = line.len() + 1;
                if char_count + line_len <= budget_chars {
                    cb_lines.push(line);
                    char_count += line_len;
                }
            }
        }

        // Get decisions
        if let Ok(decisions) = storage.get_nodes_by_type_and_tag("decision", Some(&codebase_tag), 3)
        {
            for d in &decisions {
                let line = format!("- [decision] {}", first_sentence(&d.content));
                let line_len = line.len() + 1;
                if char_count + line_len <= budget_chars {
                    cb_lines.push(line);
                    char_count += line_len;
                }
            }
        }

        if !cb_lines.is_empty() {
            context_parts.push(format!(
                "**Codebase ({}):**\n{}",
                codebase,
                cb_lines.join("\n")
            ));
        }
    }

    // ====================================================================
    // 6. Assemble final response
    // ====================================================================
    let header = format!("## Session ({} memories, {})\n", stats.total_nodes, status);
    let context_text = format!("{}{}", header, context_parts.join("\n\n"));
    let tokens_used = context_text.len() / 4;

    Ok(serde_json::json!({
        "context": context_text,
        "profile": output_config.profile.as_str(),
        "tokensUsed": tokens_used,
        "tokenBudget": token_budget,
        "expandable": expandable_ids,
        "automationTriggers": {
            "needsDream": needs_dream,
            "needsBackup": needs_backup,
            "needsGc": needs_gc,
        },
    }))
}

/// Check if an intention should be triggered based on the current context.
fn check_intention_triggered(
    intention: &vestige_core::IntentionRecord,
    ctx: &ContextSpec,
    now: DateTime<Utc>,
) -> bool {
    // Parse trigger data
    let trigger: Option<TriggerData> = serde_json::from_str(&intention.trigger_data).ok();
    let Some(trigger) = trigger else {
        return false;
    };

    match trigger.trigger_type.as_deref() {
        Some("time") => {
            if let Some(ref at) = trigger.at
                && let Ok(trigger_time) = DateTime::parse_from_rfc3339(at)
            {
                return trigger_time.with_timezone(&Utc) <= now;
            }
            if let Some(mins) = trigger.in_minutes {
                let trigger_time = intention.created_at + Duration::minutes(mins);
                return trigger_time <= now;
            }
            false
        }
        Some("context") => {
            // Check codebase match
            if let (Some(trigger_cb), Some(current_cb)) = (&trigger.codebase, &ctx.codebase)
                && current_cb
                    .to_lowercase()
                    .contains(&trigger_cb.to_lowercase())
            {
                return true;
            }
            // Check file pattern match
            if let (Some(pattern), Some(file)) = (&trigger.file_pattern, &ctx.file)
                && file.contains(pattern.as_str())
            {
                return true;
            }
            // Check topic match
            if let (Some(topic), Some(topics)) = (&trigger.topic, &ctx.topics)
                && topics
                    .iter()
                    .any(|t| t.to_lowercase().contains(&topic.to_lowercase()))
            {
                return true;
            }
            false
        }
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TriggerData {
    #[serde(rename = "type")]
    trigger_type: Option<String>,
    at: Option<String>,
    in_minutes: Option<i64>,
    codebase: Option<String>,
    file_pattern: Option<String>,
    topic: Option<String>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_test_content(storage: &Arc<Storage>, content: &str, tags: Vec<&str>) -> String {
        let input = IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: tags.into_iter().map(|s| s.to_string()).collect(),
            valid_from: None,
            valid_until: None,
        };
        let node = storage.ingest(input).unwrap();
        node.id
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_properties() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["queries"].is_object());
        assert!(s["properties"]["token_budget"].is_object());
        assert!(s["properties"]["context"].is_object());
        assert!(s["properties"]["include_status"].is_object());
        assert!(s["properties"]["include_intentions"].is_object());
        assert!(s["properties"]["include_predictions"].is_object());
    }

    #[test]
    fn test_schema_token_budget_bounds() {
        let s = schema();
        let tb = &s["properties"]["token_budget"];
        assert_eq!(tb["minimum"], 100);
        assert_eq!(tb["maximum"], 100000);
        assert_eq!(tb["default"], 1000);
    }

    // ========================================================================
    // EXECUTE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_default_no_args() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["context"].is_string());
        assert!(value["tokensUsed"].is_number());
        assert!(value["tokenBudget"].is_number());
        assert_eq!(value["tokenBudget"], 1000);
        assert!(value["expandable"].is_array());
        assert!(value["automationTriggers"].is_object());
    }

    #[tokio::test]
    async fn test_with_queries() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(
            &storage,
            "The user prefers Rust and TypeScript for all projects.",
            vec![],
        )
        .await;

        let args = serde_json::json!({
            "queries": ["user preferences", "project context"]
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(ctx.contains("Session"));
    }

    #[tokio::test]
    async fn test_token_budget_respected() {
        let (storage, _dir) = test_storage().await;
        // Ingest several memories to generate content
        for i in 0..20 {
            ingest_test_content(
                &storage,
                &format!(
                    "Memory number {} contains detailed information about topic {} that is quite long and verbose to fill up the token budget.",
                    i, i
                ),
                vec![],
            )
            .await;
        }

        let args = serde_json::json!({
            "queries": ["memory"],
            "token_budget": 200
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["context"].is_string());
        // Context should be within budget (200 tokens * 4 = 800 chars + header overhead)
        // The actual char count of context should be reasonable
        let tokens_used = value["tokensUsed"].as_u64().unwrap();
        // Allow some overhead for the header
        assert!(
            tokens_used <= 300,
            "tokens_used {} should be near budget 200",
            tokens_used
        );
    }

    #[tokio::test]
    async fn test_expandable_ids() {
        let (storage, _dir) = test_storage().await;
        // Ingest many memories
        for i in 0..20 {
            ingest_test_content(
                &storage,
                &format!(
                    "Expandable test memory {} with enough content to take up space in the token budget allocation.",
                    i
                ),
                vec![],
            )
            .await;
        }

        let args = serde_json::json!({
            "queries": ["expandable test memory"],
            "token_budget": 150
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        // expandable should be a valid array (may be empty if all fit within budget)
        assert!(value["expandable"].is_array());
    }

    #[tokio::test]
    async fn test_automation_triggers_booleans() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let triggers = &value["automationTriggers"];
        assert!(triggers["needsDream"].is_boolean());
        assert!(triggers["needsBackup"].is_boolean());
        assert!(triggers["needsGc"].is_boolean());
    }

    #[tokio::test]
    async fn test_disable_sections() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test memory for disable sections.", vec![]).await;

        let args = serde_json::json!({
            "include_status": false,
            "include_intentions": false,
            "include_predictions": false
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let context_str = value["context"].as_str().unwrap();
        // Should NOT contain status line when disabled
        assert!(!context_str.contains("**Status:**"));
        // automationTriggers should still be present (always computed)
        assert!(value["automationTriggers"].is_object());
    }

    #[tokio::test]
    async fn test_with_codebase_context() {
        let (storage, _dir) = test_storage().await;
        // Ingest a pattern with codebase tag
        let input = IngestInput {
            content: "Code pattern: Use Arc<Mutex<>> for shared state in async contexts."
                .to_string(),
            node_type: "pattern".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec!["pattern".to_string(), "codebase:vestige".to_string()],
            valid_from: None,
            valid_until: None,
        };
        storage.ingest(input).unwrap();

        let args = serde_json::json!({
            "context": {
                "codebase": "vestige",
                "topics": ["performance"]
            }
        });
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let ctx = value["context"].as_str().unwrap();
        // Should contain codebase section
        assert!(ctx.contains("vestige"));
    }

    /// Phase 2: the response echoes the active profile, and the `lean` profile
    /// suppresses inline memory dates to save tokens.
    #[tokio::test]
    async fn test_session_context_profile_echo_and_lean_dates() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Session profile content sentence.", vec![]).await;

        // Default profile -> profile echoed, dates present.
        let args = serde_json::json!({ "queries": ["profile content"] });
        let value = execute(&storage, &test_cognitive(), &OutputConfig::default(), Some(args))
            .await
            .unwrap();
        assert_eq!(value["profile"], "default");

        // Lean profile -> profile echoed as lean. The memory line must not carry
        // the "(Mon DD, YYYY)" inline date prefix.
        let cfg = vestige_core::VestigeConfig::parse("[defaults]\nprofile=lean").output();
        let args = serde_json::json!({ "queries": ["profile content"] });
        let value = execute(&storage, &test_cognitive(), &cfg, Some(args))
            .await
            .unwrap();
        assert_eq!(value["profile"], "lean");
        let ctx = value["context"].as_str().unwrap();
        if ctx.contains("**Memories:**") {
            assert!(
                !ctx.contains(", 20"),
                "lean profile should omit the inline year in memory dates"
            );
        }
    }

    // ========================================================================
    // HELPER TESTS
    // ========================================================================

    #[test]
    fn test_first_sentence_period() {
        assert_eq!(
            first_sentence("Hello world. More text here."),
            "Hello world."
        );
    }

    #[test]
    fn test_first_sentence_newline() {
        assert_eq!(first_sentence("First line\nSecond line"), "First line");
    }

    #[test]
    fn test_first_sentence_short() {
        assert_eq!(first_sentence("Short"), "Short");
    }

    #[test]
    fn test_first_sentence_long_truncated() {
        let long = "A".repeat(200);
        let result = first_sentence(&long);
        assert!(result.len() <= 150);
    }

    #[test]
    fn test_first_sentence_empty() {
        assert_eq!(first_sentence(""), "");
    }

    #[test]
    fn test_first_sentence_whitespace() {
        assert_eq!(first_sentence("  Hello world.  "), "Hello world.");
    }
}
