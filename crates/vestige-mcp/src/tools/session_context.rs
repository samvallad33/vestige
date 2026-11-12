//! Session Context Tool — One-call session initialization (v1.8.0)
//!
//! Combines search, intentions, status, predictions, and codebase context
//! into a single token-budgeted response. Replaces 5 separate calls at
//! session start (~15K tokens → ~500-1000 tokens).

use std::collections::{HashMap, HashSet};
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
            "memory_hook": {"type":"object", "description":"Read-only hook packet: exact-token BM25 full records, about 3000 tokens.",
                "properties":{"prompt":{"type":"string","maxLength":65536},"limit":{"type":"integer","minimum":1,"maximum":24},
                    "trigger":{"type":"object"}}},
            "queries": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 16,
                "description": "Legacy engine only: search queries to run (default [\"user preferences\"]). Strata ignores them with a notice; use recall with an exact handle."
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
                "description": "Current context for intention matching and code evidence",
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
                "description": "Include store status (default: true)",
                "default": true
            },
            "include_intentions": {
                "type": "boolean",
                "description": "Include triggered intentions (default: true)",
                "default": true
            },
            "include_predictions": {
                "type": "boolean",
                "description": "Include predicted memories (default: true). On Strata, ghostlink predict ranks exact code anchors for context.file and changed_files. An explicit true with no anchor returns a notice.",
                "default": true
            },
            "changed_files": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 200,
                "description": "Repository-relative paths changed in this working tree. On Strata, memories with an exact file:<path> handle, an exact path tag, a code anchor on that path, or a git-commit files: entry equal to the path are listed (failure, bug, and decision memories first). When none match, the response carries a notice. Absent = section skipped."
            }
        }
    })
}

#[derive(Debug, Deserialize, Default)]
struct SessionContextArgs {
    memory_hook: Option<Value>,
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
    if let Some(hook) = &args.memory_hook {
        let scope = args.scope.as_deref().unwrap_or("user").trim();
        if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
            return Err("invalid hook scope".into());
        }
        return super::memory_packet::packet(storage.as_ref(), scope, hook);
    }

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
    // Strata finds memories by exact handle only. A bare session_start opens
    // without the default "user preferences" search, and explicit queries are
    // dropped with a notice instead of failing the whole call.
    let is_strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let mut notices: Vec<String> = Vec::new();
    let queries = args.queries.clone().unwrap_or_else(|| {
        if is_strata {
            Vec::new()
        } else {
            vec!["user preferences".to_string()]
        }
    });

    if queries.len() > 16 {
        return Err("At most 16 startup queries are supported per call".into());
    }
    let queries = if is_strata && !queries.is_empty() {
        notices.push(format!(
            "queries ignored ({}): Strata finds memories by exact handle only. Call recall with a full memory id or an exact tag instead.",
            queries.len()
        ));
        Vec::new()
    } else {
        queries
    };
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
        return Err("scope must be a non-empty identifier of at most 200 visible bytes".into());
    }
    let superseded = storage.superseded_node_ids().map_err(|e| e.to_string())?;
    // `changed_files_and_requested_predictions_are_noted_as_unused` pins both
    // notices on an empty log. Keep them when nothing resolves; replace them
    // when an exact file handle or code anchor does.
    let cwd = context_cwd(args.context.as_ref());
    let changed_hits = if is_strata && args.changed_files.is_some() {
        let files = args.changed_files.as_deref().unwrap_or(&[]);
        let lines = strata_changed_file_lines(storage.as_ref(), scope, files, cwd, &superseded);
        if lines.is_empty() {
            notices.push(
                "changed_files ignored: open-failure matching is not available on Strata."
                    .to_string(),
            );
            Vec::new()
        } else {
            lines
        }
    } else {
        Vec::new()
    };
    let prediction_hits = if is_strata && include_predictions {
        let lines = strata_prediction_lines(storage.as_ref(), &args, scope, cwd, &superseded);
        if lines.is_empty() {
            // Only an explicit request is told that nothing was anchored. The
            // default `true` stays quiet so a bare session_start has no notice.
            if args.include_predictions == Some(true) {
                notices.push("predictions are not available on Strata.".to_string());
            }
            Vec::new()
        } else {
            lines
        }
    } else {
        Vec::new()
    };
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
    let changed_kept = budget_id_lines(
        changed_hits,
        &mut char_count,
        budget_chars,
        &mut expandable_ids,
        &mut seen_ids,
    );
    if !changed_kept.is_empty() {
        context_parts.push(format!(
            "**Changed files:**\n{}",
            changed_kept.join("\n")
        ));
    }
    let predicted_kept = budget_id_lines(
        prediction_hits,
        &mut char_count,
        budget_chars,
        &mut expandable_ids,
        &mut seen_ids,
    );
    if !predicted_kept.is_empty() {
        context_parts.push(format!("**Predicted:**\n{}", predicted_kept.join("\n")));
    }

    // ====================================================================
    // 2. Intentions — find triggered + pending high-priority
    // ====================================================================
    if include_intentions {
        let intentions = storage
            .get_active_intentions_in_scope(scope)
            .map_err(|e| e.to_string())?;
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
                    .map(|d| format!(" [due {}]", d.format("%b %d, %Y")))
                    .unwrap_or_default();
                let overdue_str = if is_overdue { " OVERDUE" } else { "" };
                let line = format!(
                    "- [{}] {}{}{}{}",
                    intention.id,
                    first_sentence(&intention.content),
                    priority_str,
                    deadline_str,
                    overdue_str
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
        let status_line = format!("**Status:** {} memories | {}", stats.total_nodes, status);
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
            }
        }
    }
    let _ = cognitive;

    // Code advice uses the same current selection and live evidence evaluator
    // as codebase.get_context, including items discovered by search.
    let mut code_elsewhere: Option<Value> = None;
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
        // Nothing for this codebase here is not nothing anywhere: name the
        // other scopes that hold its code memories instead of staying silent.
        // A backend that cannot list scopes leaves the section as it was.
        if code_items.is_empty()
            && let Ok(rows) = code_context::code_scopes(storage, Some(codebase))
        {
            let others: Vec<&code_context::ScopeCounts> =
                rows.iter().filter(|row| row.scope != scope).collect();
            if !others.is_empty() {
                notices.push(code_elsewhere_notice(codebase, scope, &others));
                code_elsewhere = Some(Value::Array(others.iter().map(|row| row.json()).collect()));
            }
        }
    }
    let repo_path = args.context.as_ref().and_then(|c| c.repo_path.as_deref());
    // session_start is annotated read-only: report live verdicts, write none.
    let verification =
        code_context::annotate_with(storage, &mut code_items, repo_path, true, false)?;
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
    if let Some(elsewhere) = code_elsewhere {
        result["codeContext"]["elsewhere"] = elsewhere;
    }
    if !notices.is_empty() {
        result["notices"] = serde_json::json!(notices);
    }
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
    // `seen_ids` is a hash set: put the ids in id order before they reach the
    // store, so the same response records its exposures in the same order.
    let mut visible: Vec<&str> = seen_ids
        .iter()
        .filter(|id| rendered.contains(id.as_str()))
        .map(String::as_str)
        .collect();
    visible.sort_unstable();
    let _ = storage.record_batch_retrieval(&visible);
    Ok(result)
}

/// Scopes a notice names before it summarizes the rest; `codeContext.elsewhere`
/// lists them all.
const ELSEWHERE_NAMED: usize = 5;

/// The notice for a codebase whose code memories are all in other scopes.
/// Bounded, since notices stay in the response whatever the token budget.
fn code_elsewhere_notice(
    codebase: &str,
    scope: &str,
    others: &[&code_context::ScopeCounts],
) -> String {
    let mut named: Vec<String> = others
        .iter()
        .take(ELSEWHERE_NAMED)
        .map(|row| {
            format!(
                "'{}' ({} patterns, {} decisions, {} events)",
                row.scope, row.patterns, row.decisions, row.events
            )
        })
        .collect();
    if others.len() > ELSEWHERE_NAMED {
        named.push(format!("{} more", others.len() - ELSEWHERE_NAMED));
    }
    format!(
        "codeContext: nothing for codebase '{codebase}' in scope '{scope}', but other scopes hold its code memories: {}. Read patterns and decisions with session action='codebase' params={{action:'get_context',codebase:'{codebase}',scope:<name>}}; check change records (events) with session action='codebase' params={{action:'verify',codebase:'{codebase}',scope:<name>,repoPath:<checkout>}}.",
        named.join(", ")
    )
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

fn context_cwd(spec: Option<&ContextSpec>) -> Option<&str> {
    spec.and_then(|spec| spec.repo_path.as_deref())
        .map(str::trim)
        .filter(|cwd| !cwd.is_empty())
}

fn normalize_repo_path(raw: &str) -> String {
    raw.trim().replace('\\', "/")
}

/// Exact path spellings to try: the caller's text, slash-normalized, `./`
/// stripped, and joined with `context.repoPath` when that checkout is known.
fn push_unique(out: &mut Vec<String>, value: String) {
    if !value.is_empty() && !out.iter().any(|kept| kept == &value) {
        out.push(value);
    }
}

fn path_variants(raw: &str, cwd: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return out;
    }
    push_unique(&mut out, trimmed.to_string());
    let slash = normalize_repo_path(trimmed);
    push_unique(&mut out, slash.clone());
    if let Some(relative) = slash.strip_prefix("./") {
        push_unique(&mut out, relative.to_string());
    }
    if let Some(cwd) = cwd {
        let cwd = normalize_repo_path(cwd).trim_end_matches('/').to_string();
        if !cwd.is_empty() {
            let prefix = format!("{cwd}/");
            push_unique(&mut out, format!("{prefix}{slash}"));
            if let Some(relative) = slash.strip_prefix(&prefix) {
                push_unique(&mut out, relative.to_string());
            }
        }
    }
    out
}

fn exact_handle_ids(storage: &Storage, query: &str) -> Vec<String> {
    let resolved = storage.resolve_handle(query);
    if resolved.exact {
        resolved.ids
    } else {
        Vec::new()
    }
}

fn tags_name_file(tags: &[String], variants: &[String]) -> bool {
    tags.iter().any(|tag| {
        variants
            .iter()
            .any(|variant| tag == variant || tag == &format!("file:{variant}"))
    })
}

fn content_names_file(content: &str, variants: &[String]) -> bool {
    for line in content.lines() {
        let Some(rest) = line.trim().strip_prefix("files:") else {
            continue;
        };
        let rest = rest.split("(+").next().unwrap_or(rest);
        for token in rest.split(',') {
            let token = normalize_repo_path(token);
            if token.is_empty() {
                continue;
            }
            if variants.iter().any(|variant| {
                variant == &token || normalize_repo_path(variant) == token
            }) {
                return true;
            }
        }
    }
    false
}

fn anchor_names_file(file_path: &str, variants: &[String]) -> bool {
    let slash = normalize_repo_path(file_path);
    variants.iter().any(|variant| {
        file_path == variant || slash == *variant || slash == normalize_repo_path(variant)
    })
}

fn strata_node_selectable(
    storage: &Storage,
    node: &vestige_core::KnowledgeNode,
    scope: &str,
    superseded: &HashSet<String>,
    now: DateTime<Utc>,
) -> bool {
    if superseded.contains(&node.id) {
        return false;
    }
    if node.valid_from.is_some_and(|t| t > now) || node.valid_until.is_some_and(|t| t <= now) {
        return false;
    }
    storage
        .node_is_in_scope(&node.id, scope)
        .unwrap_or(false)
}

fn file_hit_rank(node: &vestige_core::KnowledgeNode) -> u8 {
    if node.node_type == "bug"
        || node.node_type == "failure"
        || super::backfill::looks_like_failure(node)
    {
        0
    } else if node.node_type == "decision" {
        1
    } else {
        2
    }
}

fn file_hit_label(node: &vestige_core::KnowledgeNode) -> String {
    if node.node_type == "bug" {
        "bug".to_string()
    } else if node.node_type == "failure" || super::backfill::looks_like_failure(node) {
        "failure".to_string()
    } else if node.node_type == "decision" {
        "decision".to_string()
    } else {
        node.node_type.clone()
    }
}

/// Live in-scope memories tied to `files` by an exact handle or anchor.
///
/// `Storage` has no `get_anchors_for_file`. Anchors are read with
/// `code_anchors_for_nodes` and matched on `file_path`. Handles are
/// `file:<path>` and the path itself (`resolve_handle`).
fn strata_changed_file_lines(
    storage: &Storage,
    scope: &str,
    files: &[String],
    cwd: Option<&str>,
    superseded: &HashSet<String>,
) -> Vec<(String, String)> {
    let variants: Vec<String> = files
        .iter()
        .flat_map(|file| path_variants(file, cwd))
        .fold(Vec::new(), |mut acc, variant| {
            if !acc.iter().any(|kept: &String| kept == &variant) {
                acc.push(variant);
            }
            acc
        });
    if variants.is_empty() {
        return Vec::new();
    }

    let mut nodes: HashMap<String, vestige_core::KnowledgeNode> = HashMap::new();
    for variant in &variants {
        for query in [format!("file:{variant}"), variant.clone()] {
            for id in exact_handle_ids(storage, &query) {
                if nodes.contains_key(&id) {
                    continue;
                }
                if let Ok(Some(node)) = storage.get_node(&id) {
                    nodes.insert(id, node);
                }
            }
        }
    }

    let mut offset = 0i32;
    loop {
        let Ok(batch) = storage.get_all_nodes_in_scope(scope, 500, offset) else {
            break;
        };
        let count = batch.len();
        if count == 0 {
            break;
        }
        let ids: Vec<String> = batch.iter().map(|node| node.id.clone()).collect();
        // No `get_anchors_for_file` on Storage. Anchors are per memory.
        let anchors = storage.code_anchors_for_nodes(&ids).unwrap_or_default();
        for node in batch {
            let rows = anchors.get(&node.id);
            let touches = tags_name_file(&node.tags, &variants)
                || content_names_file(&node.content, &variants)
                || rows.is_some_and(|rows| {
                    rows.iter()
                        .any(|anchor| anchor_names_file(&anchor.file_path, &variants))
                });
            if touches {
                nodes.insert(node.id.clone(), node);
            }
        }
        if count < 500 {
            break;
        }
        let next = offset.saturating_add(500);
        if next == offset {
            break;
        }
        offset = next;
    }

    let now = Utc::now();
    let mut hits: Vec<vestige_core::KnowledgeNode> = nodes
        .into_values()
        .filter(|node| strata_node_selectable(storage, node, scope, superseded, now))
        .collect();
    hits.sort_by(|a, b| {
        file_hit_rank(a)
            .cmp(&file_hit_rank(b))
            .then_with(|| a.id.cmp(&b.id))
    });
    hits.into_iter()
        .map(|node| {
            let label = file_hit_label(&node);
            let summary = first_sentence(&node.content);
            let line = format!("- [{}] ({label}) {summary}", node.id);
            (node.id, line)
        })
        .collect()
}

/// `ghostlink::predict` context: `current_file` from `context.file` and each
/// changed path, plus `codebase` and `cwd` (`context.repoPath`). `current_topics`
/// is omitted — a non-empty topic list returns `similarity_disabled` and would
/// drop the file anchors.
fn strata_prediction_lines(
    storage: &Storage,
    args: &SessionContextArgs,
    scope: &str,
    cwd: Option<&str>,
    superseded: &HashSet<String>,
) -> Vec<(String, String)> {
    let mut targets = Vec::new();
    if let Some(spec) = &args.context
        && let Some(file) = &spec.file
    {
        targets.extend(path_variants(file, cwd));
    }
    if let Some(files) = &args.changed_files {
        for file in files {
            for variant in path_variants(file, cwd) {
                if !targets.iter().any(|kept: &String| kept == &variant) {
                    targets.push(variant);
                }
            }
        }
    }
    if targets.is_empty() {
        return Vec::new();
    }

    let now = Utc::now();
    let mut seen = HashSet::new();
    let mut ranked: Vec<(f64, String, String)> = Vec::new();
    for file in targets.into_iter().take(16) {
        let mut context = serde_json::json!({ "current_file": file });
        if let Some(spec) = &args.context {
            if let Some(codebase) = &spec.codebase {
                context["codebase"] = serde_json::Value::String(codebase.clone());
            }
            if let Some(cwd) = cwd {
                context["cwd"] = serde_json::Value::String(cwd.to_string());
            }
        }
        let predicted = match crate::strata_memory::ghostlink::predict(storage, Some(&context)) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(rows) = predicted.get("predictions").and_then(|value| value.as_array()) else {
            continue;
        };
        for row in rows {
            let Some(id) = row.get("memory_id").and_then(|value| value.as_str()) else {
                continue;
            };
            if !seen.insert(id.to_string()) {
                continue;
            }
            let Ok(Some(node)) = storage.get_node(id) else {
                continue;
            };
            if !strata_node_selectable(storage, &node, scope, superseded, now) {
                continue;
            }
            let preview = row
                .get("content_preview")
                .and_then(|value| value.as_str())
                .filter(|text| !text.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| first_sentence(&node.content));
            let confidence = row
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let line = format!("- [{id}] {preview}");
            ranked.push((confidence, id.to_string(), line));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    ranked
        .into_iter()
        .map(|(_, id, line)| (id, line))
        .collect()
}

fn budget_id_lines(
    lines: Vec<(String, String)>,
    char_count: &mut usize,
    budget_chars: usize,
    expandable_ids: &mut Vec<String>,
    seen_ids: &mut HashSet<String>,
) -> Vec<String> {
    let mut kept = Vec::new();
    for (id, line) in lines {
        let line_len = line.len() + 1;
        if *char_count + line_len > budget_chars {
            expandable_ids.push(id);
        } else {
            *char_count += line_len;
            seen_ids.insert(id);
            kept.push(line);
        }
    }
    kept
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod strata_tests {
    use super::*;
    use serde_json::json;

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    fn strata() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    async fn start(storage: &Arc<Storage>, args: Value) -> Value {
        execute(storage, &cognitive(), &OutputConfig::default(), Some(args))
            .await
            .expect("session_start must answer on Strata")
    }

    async fn set_intention(storage: &Arc<Storage>, args: Value) -> String {
        let set = crate::tools::intention_unified::execute(storage, &cognitive(), Some(args))
            .await
            .unwrap();
        set["intentionId"]
            .as_str()
            .expect("intention id")
            .to_string()
    }

    #[tokio::test]
    async fn a_bare_start_carries_no_notice() {
        let (storage, _dir) = strata();
        let out = start(&storage, json!({})).await;
        assert!(out.get("notices").is_none(), "{out}");
        assert!(out["context"].as_str().unwrap().starts_with("## Session"));
    }

    #[tokio::test]
    async fn explicit_queries_are_dropped_with_a_notice_not_an_error() {
        let (storage, _dir) = strata();
        let out = start(
            &storage,
            json!({"queries": ["user preferences", "release"]}),
        )
        .await;
        let notices = out["notices"]
            .as_array()
            .expect("notice for dropped queries");
        assert!(
            notices[0]
                .as_str()
                .unwrap()
                .starts_with("queries ignored (2)"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn changed_files_and_requested_predictions_are_noted_as_unused() {
        let (storage, _dir) = strata();
        let out = start(
            &storage,
            json!({"changed_files": ["src/lib.rs"], "include_predictions": true}),
        )
        .await;
        let notices = out["notices"].to_string();
        assert!(notices.contains("changed_files ignored"), "{out}");
        assert!(notices.contains("predictions are not available"), "{out}");
        let context = out["context"].as_str().unwrap();
        assert!(!context.contains("embeddings"), "{context}");
        assert!(!context.contains("**Predicted:**"), "{context}");
    }

    #[tokio::test]
    async fn intentions_follow_scope_carry_ids_and_date_with_the_year() {
        let (storage, _dir) = strata();
        let mine = set_intention(
            &storage,
            json!({"action": "set", "description": "Rotate the signing key",
                   "deadline": "2020-01-02T00:00:00Z", "priority": "high", "scope": "user"}),
        )
        .await;
        let theirs = set_intention(
            &storage,
            json!({"action": "set", "description": "Other project chore",
                   "priority": "high", "scope": "other-project"}),
        )
        .await;
        let out = start(&storage, json!({"scope": "user"})).await;
        let context = out["context"].as_str().unwrap();
        assert!(context.contains(&format!("[{mine}]")), "{context}");
        assert!(
            context.contains("2020"),
            "the due date carries its year: {context}"
        );
        assert!(context.contains("OVERDUE"), "{context}");
        assert!(
            !context.contains(&theirs),
            "another scope's intention leaked: {context}"
        );
        assert!(!context.contains("Other project chore"), "{context}");
    }

    #[tokio::test]
    async fn needs_backup_and_needs_dream_follow_the_real_stamps() {
        let (storage, dir) = strata();
        let before = start(&storage, json!({})).await;
        assert_eq!(before["automationTriggers"]["needsBackup"], json!(true));
        assert_eq!(before["automationTriggers"]["needsDream"], json!(true));

        let backup = tempfile::tempdir().unwrap();
        storage.backup_to(&backup.path().join("snap")).unwrap();
        let record = vestige_core::DreamHistoryRecord {
            dreamed_at: Utc::now(),
            duration_ms: 1,
            memories_replayed: 0,
            connections_found: 0,
            insights_generated: 0,
            memories_strengthened: 0,
            memories_compressed: 0,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        };
        storage.save_dream_history(&record).unwrap();

        let after = start(&storage, json!({})).await;
        assert_eq!(
            after["automationTriggers"]["needsBackup"],
            json!(false),
            "{after}"
        );
        assert_eq!(
            after["automationTriggers"]["needsDream"],
            json!(false),
            "{after}"
        );
        let stamps = std::fs::read_to_string(dir.path().join("maintenance-stamps.json")).unwrap();
        assert!(
            stamps.contains("lastBackupMs") && stamps.contains("lastDreamMs"),
            "{stamps}"
        );
    }

    #[tokio::test]
    async fn maintenance_stamps_keep_each_other_and_never_move_back() {
        let (storage, _dir) = strata();
        let at = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc);
        let dream = |when| vestige_core::DreamHistoryRecord {
            dreamed_at: when,
            duration_ms: 1,
            memories_replayed: 0,
            connections_found: 0,
            insights_generated: 0,
            memories_strengthened: 0,
            memories_compressed: 0,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        };
        storage
            .save_dream_history(&dream(at("2026-09-02T00:00:00Z")))
            .unwrap();
        let backup = tempfile::tempdir().unwrap();
        storage.backup_to(&backup.path().join("snap")).unwrap();
        storage
            .save_dream_history(&dream(at("2026-09-01T00:00:00Z")))
            .unwrap();
        assert_eq!(
            storage.get_last_dream().unwrap(),
            Some(at("2026-09-02T00:00:00Z")),
            "an older dream must not move the stamp back"
        );
        assert!(
            storage.last_backup_timestamp().is_some(),
            "the dream write dropped the backup stamp"
        );
    }

    #[tokio::test]
    async fn session_start_reports_live_verdicts_and_writes_none() {
        let (storage, _dir) = strata();
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        let source = repo.path().join("src/state.rs");
        std::fs::write(
            &source,
            "pub fn load_config(path: &str) -> String {\n    std::fs::read_to_string(path).unwrap()\n}\n",
        )
        .unwrap();
        let saved = crate::tools::codebase_unified::execute(
            &storage,
            &cognitive(),
            &OutputConfig::default(),
            Some(json!({"action": "remember_pattern", "codebase": "anchored",
                        "repoPath": repo.path(), "name": "Eager config read",
                        "description": "load_config reads the whole file eagerly",
                        "files": ["src/state.rs#load_config"]})),
        )
        .await
        .unwrap();
        let id = saved["nodeId"].as_str().unwrap().to_string();
        let recorded = |storage: &Arc<Storage>| {
            storage
                .code_anchors_for_nodes(std::slice::from_ref(&id))
                .unwrap()[&id][0]
                .last_status
        };
        let before = recorded(&storage);

        // The symbol disappears, so a live check now reports it missing.
        std::fs::write(&source, "pub fn other() {}\n").unwrap();
        let out = start(
            &storage,
            json!({"context": {"codebase": "anchored", "repoPath": repo.path()}}),
        )
        .await;
        assert_eq!(
            out["codeContext"]["verification"]["enabled"],
            json!(true),
            "{out}"
        );
        assert_eq!(
            out["codeContext"]["verification"]["stale"],
            json!(1),
            "{out}"
        );
        assert_eq!(
            recorded(&storage),
            before,
            "a read-only session_start persisted a verdict"
        );
    }

    /// A codebase whose code memories all live in other scopes is named, not
    /// met with an empty, silent codeContext.
    #[tokio::test]
    async fn code_context_names_the_other_scopes_that_hold_the_codebase() {
        let (storage, _dir) = strata();
        crate::tools::codebase_unified::execute(
            &storage,
            &cognitive(),
            &OutputConfig::default(),
            Some(
                json!({"action": "remember_decision", "codebase": "demo", "scope": "demo-repo",
                        "decision": "Walk only recorded edges", "rationale": "elsewhere test"}),
            ),
        )
        .await
        .unwrap();
        storage
            .ingest_in_scope(
                vestige_core::IngestInput {
                    content: "commit 0123 touched src/lib.rs".to_string(),
                    node_type: "event".to_string(),
                    tags: vec!["git-commit".to_string(), "codebase:demo".to_string()],
                    ..Default::default()
                },
                "demo",
            )
            .unwrap();

        let out = start(&storage, json!({"context": {"codebase": "demo"}})).await;
        assert_eq!(out["codeContext"]["items"], json!([]), "{out}");
        assert_eq!(
            out["codeContext"]["elsewhere"],
            json!([
                {"scope": "demo", "patterns": 0, "decisions": 0, "events": 1},
                {"scope": "demo-repo", "patterns": 0, "decisions": 1, "events": 0},
            ]),
            "{out}"
        );
        let notice = out["notices"]
            .as_array()
            .expect("a notice names the scopes")
            .iter()
            .filter_map(Value::as_str)
            .find(|notice| notice.starts_with("codeContext:"))
            .unwrap_or_else(|| panic!("no codeContext notice: {out}"))
            .to_string();
        assert!(notice.contains("in scope 'user'"), "{notice}");
        assert!(
            notice.contains("'demo-repo' (0 patterns, 1 decisions, 0 events)"),
            "{notice}"
        );
        assert!(
            notice.contains("'demo' (0 patterns, 0 decisions, 1 events)"),
            "{notice}"
        );
        assert!(
            notice
                .contains("session action='codebase' params={action:'get_context',codebase:'demo'"),
            "{notice}"
        );

        // the scope that holds the advice: its items, and no pointer elsewhere
        let out = start(
            &storage,
            json!({"scope": "demo-repo", "context": {"codebase": "demo"}}),
        )
        .await;
        assert_eq!(
            out["codeContext"]["items"].as_array().unwrap().len(),
            1,
            "{out}"
        );
        assert!(out["codeContext"].get("elsewhere").is_none(), "{out}");
        assert!(out.get("notices").is_none(), "{out}");

        // a codebase no scope holds: nothing to point at, so nothing is said
        let out = start(&storage, json!({"context": {"codebase": "nowhere"}})).await;
        assert!(out["codeContext"].get("elsewhere").is_none(), "{out}");
        assert!(out.get("notices").is_none(), "{out}");
    }
}
