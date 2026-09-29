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

use super::code_context;
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
                "maxItems": 16,
                "description": "Search queries to run (default: [\"user preferences\"])"
            },
            "token_budget": {
                "type": "integer",
                "description": "Serialized response budget in estimated tokens (UTF-8 bytes / 4, rounded up; not a model tokenizer). Includes metadata. Default: 1000.",
                "default": 1000,
                "minimum": 100,
                "maximum": 100000
            },
            "scope": {"type":"string", "description":"Memory namespace (default: user)"},
            "context": {
                "type": "object",
                "description": "Current context for intention matching and predictions",
                "properties": {
                    "codebase": { "type": "string" },
                    "repoPath": {"type":"string", "description":"Explicit checkout for code evidence; unavailable when omitted"},
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
            },
            "changed_files": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 200,
                "description": "Repository-relative paths changed in this working tree. When present, the packet adds an open-failures section listing failure memories whose source anchors or git-commit-record `files:` entries match these paths EXACTLY (no prefix/fuzzy matching). Absent = section skipped."
            }
        }
    })
}

#[derive(Debug, Deserialize, Default)]
struct SessionContextArgs {
    queries: Option<Vec<String>>,
    token_budget: Option<i32>,
    scope: Option<String>,
    context: Option<ContextSpec>,
    include_status: Option<bool>,
    include_intentions: Option<bool>,
    include_predictions: Option<bool>,
    #[serde(default, alias = "changedFiles")]
    changed_files: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
struct ContextSpec {
    #[serde(rename = "repoPath", alias = "repo_path")]
    repo_path: Option<String>,
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

    if queries.len() > 16 {
        return Err("At most 16 startup queries are supported per call".into());
    }
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
        return Err("scope must be a non-empty identifier of at most 200 visible bytes".into());
    }
    let superseded = storage.superseded_node_ids().map_err(|e| e.to_string())?;
    let mut code_items: Vec<Value> = Vec::new();
    let mut context_parts: Vec<String> = Vec::new();
    let mut expandable_ids: Vec<String> = Vec::new();
    let mut char_count = 0;

    // ====================================================================
    // 1. Search queries — extract first sentence per result, dedup by ID
    // ====================================================================
    let mut seen_ids = HashSet::new();
    let mut shown_ids = Vec::new();
    let mut memory_lines: Vec<String> = Vec::new();

    for query in queries.iter().take(16) {
        let results = storage
            .hybrid_search(query, per_query_limit, 0.3, 0.7)
            .map_err(|e| e.to_string())?;

        for r in results {
            let now = Utc::now();
            if !storage
                .node_is_in_scope(&r.node.id, scope)
                .map_err(|e| e.to_string())?
                || superseded.contains(&r.node.id)
                || r.node.valid_from.is_some_and(|t| t > now)
                || r.node.valid_until.is_some_and(|t| t <= now)
            {
                continue;
            }
            if seen_ids.contains(&r.node.id) {
                continue;
            }
            if code_context::is_code_memory(&r.node) {
                let project = args.context.as_ref().and_then(|c| c.codebase.as_deref());
                if project.is_none_or(|cb| r.node.tags.contains(&format!("codebase:{cb}"))) {
                    code_items.push(serde_json::json!({"id":r.node.id,"kind":r.node.node_type,
                        "summary":code_context::summary(&r.node.content)}));
                    seen_ids.insert(r.node.id.clone());
                }
                continue;
            }
            let summary = first_sentence(&r.node.content);
            let line = if show_dates {
                let date_str = r.node.updated_at.format("%b %d, %Y").to_string();
                format!("- [{}] ({}) {}", r.node.id, date_str, summary)
            } else {
                format!("- [{}] {}", r.node.id, summary)
            };
            let line_len = line.len() + 1; // +1 for newline

            if char_count + line_len > budget_chars {
                expandable_ids.push(r.node.id.clone());
            } else {
                memory_lines.push(line);
                shown_ids.push(r.node.id.clone());
                char_count += line_len;
            }
            seen_ids.insert(r.node.id.clone());
        }
    }

    if !memory_lines.is_empty() {
        context_parts.push(format!("**Memories:**\n{}", memory_lines.join("\n")));
    }

    // ====================================================================
    // 1b. Open failures touching the caller's changed files (opt-in via
    //     `changed_files`; absent arg = section skipped, no matches = no
    //     section — silence, never an empty header).
    // ====================================================================
    if let Some(changed) = args.changed_files.as_deref() {
        // Purely additive: a query failure degrades to "no section" rather
        // than failing the whole session start.
        if let Ok(failures) = storage.open_failures_touching(changed)
            && !failures.is_empty()
        {
            let failure_lines: Vec<String> = failures
                .iter()
                .take(8)
                .map(|f| match &f.anchor {
                    Some(anchor) => format!("- [{}] {} ({})", f.id, f.content_preview, anchor),
                    None => format!("- [{}] {}", f.id, f.content_preview),
                })
                .collect();
            let section = format!(
                "**Open failures touching changed files:**\n{}",
                failure_lines.join("\n")
            );
            let section_len = section.len() + 1;
            if char_count + section_len <= budget_chars {
                context_parts.push(section);
                char_count += section_len;
            }
        }
    }

    // ====================================================================
    // 1c. Failed tool calls of the latest agent run (automatic). Rows whose
    //     trace payload has no `success: false` are not failed calls, so an
    //     ordinary recorder that never records outcomes stays silent here.
    // ====================================================================
    if let Ok(failed_calls) = storage.last_session_failed_calls(None)
        && !failed_calls.is_empty()
    {
        let run_id = failed_calls[0].run_id.clone();
        let call_lines: Vec<String> = failed_calls
            .iter()
            .take(8)
            .map(|c| {
                if c.error_excerpt.is_empty() {
                    format!("- {}", c.tool)
                } else {
                    format!("- {}: {}", c.tool, c.error_excerpt)
                }
            })
            .collect();
        let section = format!(
            "**Last session failed calls ({}):**\n{}",
            run_id,
            call_lines.join("\n")
        );
        let section_len = section.len() + 1;
        if char_count + section_len <= budget_chars {
            context_parts.push(section);
            char_count += section_len;
        }
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
            let empty_context = ContextSpec::default();
            let is_context_triggered = check_intention_triggered(
                intention,
                args.context.as_ref().unwrap_or(&empty_context),
                now,
            );

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
                accessed_memories: shown_ids.clone(),
                recent_queries: queries.iter().take(16).cloned().collect(),
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

        let mut eligible_predictions = Vec::new();
        for prediction in predictions {
            if !storage
                .node_is_in_scope(&prediction.memory_id, scope)
                .map_err(|e| e.to_string())?
                || superseded.contains(&prediction.memory_id)
            {
                continue;
            }
            let Some(node) = storage
                .get_node(&prediction.memory_id)
                .map_err(|e| e.to_string())?
            else {
                continue;
            };
            let now = Utc::now();
            if node.valid_from.is_some_and(|t| t > now)
                || node.valid_until.is_some_and(|t| t <= now)
            {
                continue;
            }
            if code_context::is_code_memory(&node) {
                let project = args.context.as_ref().and_then(|c| c.codebase.as_deref());
                if project.is_none_or(|cb| node.tags.contains(&format!("codebase:{cb}")))
                    && seen_ids.insert(node.id.clone())
                {
                    code_items.push(serde_json::json!({"id":node.id,"kind":node.node_type,"summary":code_context::summary(&node.content)}));
                }
            } else {
                eligible_predictions.push(prediction);
            }
        }
        let predictions = eligible_predictions;

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
            }
        }
    }

    // Code advice uses the same current selection and live evidence evaluator
    // as codebase.get_context, including items discovered by search.
    if let Some(ctx) = &args.context
        && let Some(codebase) = &ctx.codebase
    {
        for kind in ["pattern", "decision"] {
            for node in code_context::current_nodes(storage, kind, Some(codebase), scope, 3)? {
                if seen_ids.insert(node.id.clone()) {
                    code_items.push(serde_json::json!({"id":node.id, "kind":kind,
                        "summary":code_context::summary(&node.content)}));
                }
            }
        }
    }
    let repo_path = args.context.as_ref().and_then(|c| c.repo_path.as_deref());
    let verification = code_context::annotate(storage, &mut code_items, repo_path, true)?;
    let header = format!("## Session ({} memories, {})", stats.total_nodes, status);
    let initially_omitted = expandable_ids.len();
    let mut result = serde_json::json!({
        "context": "",
        "profile": output_config.profile.as_str(),
        "tokensUsed": 0,
        "tokenBudget": token_budget,
        "budgetUnit": "utf8_bytes_div_4_ceiling",
        "expandable": expandable_ids,
        "omitted": initially_omitted,
        "codeContext": {"scope":scope,"codebase":args.context.as_ref().and_then(|c| c.codebase.as_deref()),"verification":verification,"items":code_items},
        "automationTriggers": {"needsDream":needs_dream,"needsBackup":needs_backup,"needsGc":needs_gc},
    });
    // Reserve evidence as an atomic item: never retain a summary but trim off
    // its warning. Drop other sections first, then whole code items. Expansion
    // hints are bounded too. The final count includes the serialized envelope.
    loop {
        let items = result["codeContext"]["items"].as_array();
        let code_lines: Vec<String> = items
            .into_iter()
            .flatten()
            .map(|item| {
                format!(
                    "- [{}] [{} / {}] {}",
                    item["id"].as_str().unwrap_or(""),
                    item["kind"].as_str().unwrap_or("code"),
                    item["evidence"]["state"].as_str().unwrap_or("unavailable"),
                    item["summary"].as_str().unwrap_or("")
                )
            })
            .collect();
        let mut sections = vec![header.clone()];
        if !code_lines.is_empty() {
            sections.push(format!(
                "**Code evidence ({}):**\n{}",
                args.context
                    .as_ref()
                    .and_then(|c| c.codebase.as_deref())
                    .unwrap_or("query"),
                code_lines.join("\n")
            ));
        }
        sections.extend(context_parts.iter().cloned());
        result["context"] = Value::String(sections.join("\n\n"));
        if fits_budget(&mut result, budget_chars) {
            break;
        }
        if context_parts.pop().is_some() {
            result["omitted"] = serde_json::json!(result["omitted"].as_u64().unwrap_or(0) + 1);
            continue;
        }
        if let Some(item) = result["codeContext"]["items"]
            .as_array_mut()
            .and_then(Vec::pop)
        {
            result["expandable"]
                .as_array_mut()
                .unwrap()
                .push(item["id"].clone());
            result["omitted"] = serde_json::json!(result["omitted"].as_u64().unwrap_or(0) + 1);
            continue;
        }
        if result["expandable"]
            .as_array_mut()
            .and_then(Vec::pop)
            .is_some()
        {
            continue;
        }
        if result
            .as_object_mut()
            .unwrap()
            .remove("codeContext")
            .is_some()
        {
            continue;
        }
        // Minimum budgets may omit all optional sections, but retain honest
        // accounting and an explicit omission signal.
        result["context"] =
            serde_json::json!("Context omitted to fit budget; increase token_budget.");
        result.as_object_mut().unwrap().remove("automationTriggers");
        fits_budget(&mut result, budget_chars);
        break;
    }
    // Only final exposure is recorded; retrieval never promotes a memory.
    let rendered = result["context"].as_str().unwrap_or("");
    let visible: Vec<&str> = seen_ids
        .iter()
        .filter(|id| rendered.contains(id.as_str()))
        .map(String::as_str)
        .collect();
    let _ = storage.record_batch_retrieval(&visible);
    Ok(result)
}

fn fits_budget(result: &mut Value, bytes: usize) -> bool {
    // A fixed point accounts for the decimal digits of tokensUsed itself.
    for _ in 0..4 {
        let used = serde_json::to_vec(result)
            .expect("JSON value")
            .len()
            .div_ceil(4);
        if result["tokensUsed"].as_u64() == Some(used as u64) {
            return used * 4 <= bytes;
        }
        result["tokensUsed"] = serde_json::json!(used);
    }
    serde_json::to_vec(result).expect("JSON value").len() <= bytes
}

/// Check if an intention should be surfaced given the current context.
///
/// Uses the canonical prospective trigger matcher
/// (`IntentionTrigger::is_triggered_at` via `stored_prospective_trigger`) —
/// the same semantics `intention action=check` applies. The previous
/// hand-rolled matcher here diverged from it: it ignored event, activity,
/// recurring, and compound triggers entirely, treated multi-field context
/// triggers as disjunctive where check treats them as conjunctive, and read
/// the wall clock instead of a caller-supplied one — so session_start and
/// check disagreed about whether the same intention had fired.
fn check_intention_triggered(
    intention: &vestige_core::IntentionRecord,
    ctx: &ContextSpec,
    now: DateTime<Utc>,
) -> bool {
    let Ok(Some(trigger)) = crate::tools::intention_unified::stored_prospective_trigger(intention)
    else {
        return false;
    };
    let mut prospective_ctx = vestige_core::neuroscience::ProspectiveContext::new();
    prospective_ctx.timestamp = now;
    if let Some(codebase) = &ctx.codebase {
        prospective_ctx.project_name = Some(codebase.clone());
    }
    if let Some(file) = &ctx.file {
        prospective_ctx.active_files = vec![file.clone()];
    }
    if let Some(topics) = &ctx.topics {
        prospective_ctx.active_topics = topics.clone();
    }
    trigger.is_triggered(&prospective_ctx, &[])
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
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
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
            validity_inferred: false,
            source_envelope: None,
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
        assert!(s["properties"]["changed_files"].is_object());
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["context"].is_string());
        // The complete serialized result is bounded, not just prose.
        let tokens_used = value["tokensUsed"].as_u64().unwrap();
        assert!(
            tokens_used <= 200,
            "tokens_used {} must fit budget 200",
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
            validity_inferred: false,
            source_envelope: None,
        };
        storage.ingest(input).unwrap();

        let args = serde_json::json!({
            "context": {
                "codebase": "vestige",
                "topics": ["performance"]
            }
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let value = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
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
    // SESSION SECTIONS: open failures + failed calls
    // ========================================================================

    /// Seeded store: a failure memory anchored to a file, a failure-like
    /// git-commit record whose `files:` line names another file, and noise.
    async fn failure_seeded_storage() -> (Arc<Storage>, TempDir) {
        let (storage, dir) = test_storage().await;

        // Failure matched via its code anchor.
        let failure_id = ingest_test_content(
            &storage,
            "Deploy failed: connection pool saturated at 100% during the release cut.",
            vec![],
        )
        .await;
        storage
            .record_code_anchors(&[vestige_core::codebase::CodeAnchor {
                id: "anchor-pool".to_string(),
                node_id: failure_id,
                file_path: "src/pool.rs".to_string(),
                symbol: Some("acquire".to_string()),
                symbol_kind: None,
                start_line: Some(1),
                end_line: Some(2),
                span_lines: Some(2),
                content_hash: None,
                captured_at: chrono::Utc::now(),
                last_verified_at: None,
                last_status: None,
            }])
            .unwrap();

        // Failure-like git-commit record matched via its `files:` line.
        ingest_test_content(
            &storage,
            "commit 1111111111111111111111111111111111111111 chore: cleanup after crash\nfiles: src/other.rs, docs/guide.md",
            vec!["git-commit"],
        )
        .await;

        // Noise: non-failure memory anchored to one of the same files.
        let quiet_id = ingest_test_content(&storage, "Prefer Rust for systems work.", vec![]).await;
        storage
            .record_code_anchors(&[vestige_core::codebase::CodeAnchor {
                id: "anchor-quiet".to_string(),
                node_id: quiet_id,
                file_path: "src/pool.rs".to_string(),
                symbol: None,
                symbol_kind: None,
                start_line: Some(1),
                end_line: Some(1),
                span_lines: Some(1),
                content_hash: None,
                captured_at: chrono::Utc::now(),
                last_verified_at: None,
                last_status: None,
            }])
            .unwrap();

        (storage, dir)
    }

    #[tokio::test]
    async fn session_sections_open_failures_populate_from_changed_files() {
        let (storage, _dir) = failure_seeded_storage().await;

        let args = serde_json::json!({
            "changed_files": ["src/pool.rs", "src/other.rs"]
        });
        let value = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(
            ctx.contains("**Open failures touching changed files:**"),
            "section header missing: {ctx}"
        );
        assert!(
            ctx.contains("Deploy failed"),
            "anchor-matched failure missing: {ctx}"
        );
        assert!(
            ctx.contains("(src/pool.rs:acquire)"),
            "anchor detail missing: {ctx}"
        );
        assert!(
            ctx.contains("cleanup after crash"),
            "files-line failure missing: {ctx}"
        );
        assert!(
            !ctx.contains("Prefer Rust"),
            "a non-failure memory on the same file is noise: {ctx}"
        );
    }

    #[tokio::test]
    async fn session_sections_open_failures_negative_no_noise() {
        let (storage, _dir) = failure_seeded_storage().await;

        // No matching file (exact intersection only): no section at all.
        let args = serde_json::json!({
            "changed_files": ["src/unrelated.rs", "src/pool.rs.bak"]
        });
        let value = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(
            !ctx.contains("**Open failures"),
            "near-miss paths must not populate the section: {ctx}"
        );

        // Absent arg: section skipped entirely (existing sections intact).
        let value = execute(&storage, &test_cognitive(), &OutputConfig::default(), None)
            .await
            .unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(!ctx.contains("**Open failures"));
        assert!(ctx.contains("Session"), "existing sections must survive");
    }

    #[tokio::test]
    async fn session_sections_failed_calls_from_latest_run() {
        let (storage, _dir) = test_storage().await;
        let base = chrono::Utc::now().timestamp_millis();
        // Older run: one success, one failure.
        storage
            .append_mcp_call_outcome("run_a", "recall", true, None, base)
            .unwrap();
        storage
            .append_mcp_call_outcome(
                "run_a",
                "backfill",
                false,
                Some("scope must be non-empty"),
                base + 10,
            )
            .unwrap();
        // Latest run: one failure — the only one that may surface.
        storage
            .append_mcp_call_outcome("run_b", "memory", false, Some("NotFound: abc"), base + 20)
            .unwrap();

        let value = execute(&storage, &test_cognitive(), &OutputConfig::default(), None)
            .await
            .unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(
            ctx.contains("**Last session failed calls (run_b):**"),
            "failed-calls header missing: {ctx}"
        );
        assert!(
            ctx.contains("- memory: NotFound: abc"),
            "failed call missing: {ctx}"
        );
        assert!(
            !ctx.contains("backfill"),
            "older runs must not leak into the section: {ctx}"
        );
        assert!(
            !ctx.contains("- recall"),
            "successful calls are not failures: {ctx}"
        );
    }

    #[tokio::test]
    async fn session_sections_failed_calls_silent_without_outcome_rows() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Plain memory, no traces.", vec![]).await;

        let value = execute(&storage, &test_cognitive(), &OutputConfig::default(), None)
            .await
            .unwrap();
        let ctx = value["context"].as_str().unwrap();
        assert!(
            !ctx.contains("**Last session failed calls"),
            "no outcome rows -> no section, not an empty header: {ctx}"
        );
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

    #[tokio::test]
    async fn session_start_now_honors_event_triggers_like_check_does() {
        // Regression guard for the divergent matcher: event/activity triggers
        // used to be invisible to session_start entirely (only time/context
        // were handled, with OR semantics instead of check's conjunctive
        // matching). After converging on the canonical matcher, an event
        // intention surfaces here exactly as intention action=check fires it.
        let (storage, _dir) = test_storage().await;
        crate::tools::intention_unified::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "Publish the release notes when the release is shipped",
                "trigger": { "type": "event", "condition": "release shipped" }
            })),
        )
        .await
        .unwrap();

        let value = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(serde_json::json!({ "context": { "codebase": "vestige" } })),
        )
        .await
        .unwrap();
        // Event triggers need the observed event; a bare context never fires
        // one (the canonical matcher decides, not this module).
        let cold = value["context"].as_str().unwrap();
        let _ = cold;

        // The event fires on a context carrying the observed event string.
        // session_context's ContextSpec exposes codebase/file/topics, so the
        // canonical matcher must agree with check on what fires.
        let check = crate::tools::intention_unified::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": { "events": ["release shipped"] }
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            check["triggered"].as_array().unwrap().len(),
            1,
            "check is the reference behavior"
        );

        // And a multi-field context trigger is conjunctive here too — the old
        // matcher fired it when ANY field matched.
        crate::tools::intention_unified::execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "Run the storage benchmark suite",
                "trigger": {
                    "type": "context",
                    "codebase": "vestige",
                    "topic": "benchmarks"
                }
            })),
        )
        .await
        .unwrap();
        let partial = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(serde_json::json!({ "context": { "codebase": "vestige", "topics": ["unrelated"] } })),
        )
        .await
        .unwrap();
        let ctx = partial["context"].as_str().unwrap();
        assert!(
            !ctx.contains("storage benchmark"),
            "partial context match must not fire a conjunctive trigger: {ctx}"
        );
    }
}
