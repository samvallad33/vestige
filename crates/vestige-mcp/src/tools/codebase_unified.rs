//! Unified Codebase Tool
//!
//! Merges remember_pattern, remember_decision, and get_codebase_context into a single
//! `codebase` tool with action-based dispatch.

use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::codebase::{
    AnchorDraft, AnchorStatus, AnchorVerification, CodeAnchor, capture_anchor, verify_anchor,
};
use vestige_core::{IngestInput, OutputConfig, Storage};

use super::search_unified::apply_output_masks;

/// Input schema for the unified codebase tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember_pattern", "remember_decision", "get_context", "verify"],
                "description": "'remember_pattern' stores a code pattern, 'remember_decision' an architectural decision, 'get_context' returns both with a current-or-stale mark, 'verify' re-checks every anchored code memory against the working tree"
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
                "description": "Files this pattern or decision touches: a path ('src/state.py'), path plus symbol ('src/state.py#load_config'), or path plus lines ('src/state.py:552-580'). Anything beyond a plain path is content-hashed so the memory can report when the code changed."
            },
            "anchors": {
                "type": "array",
                "description": "Structured form of 'files' for callers that know the symbol. Each anchor is content-hashed at save time so staleness is detectable.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Repository-relative path to the anchored file" },
                        "symbol": { "type": "string", "description": "Function/type/class this memory is about. Strongly preferred over a line number: line numbers rot within a week." },
                        "symbolKind": { "type": "string", "description": "Optional display kind, e.g. 'fn', 'class'" },
                        "startLine": { "type": "integer", "description": "1-based start line, when the exact span is known" },
                        "endLine": { "type": "integer", "description": "1-based inclusive end line" }
                    },
                    "required": ["path"]
                }
            },
            "repoPath": {
                "type": "string",
                "description": "Repository root used to resolve anchor paths (default: the server's working directory)"
            },
            "verify": {
                "type": "boolean",
                "description": "Check returned code memories against the current source and flag the ones that no longer match (default: true)",
                "default": true
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
    anchors: Option<Vec<AnchorArg>>,
    repo_path: Option<String>,
    codebase: Option<String>,
    // Context fields
    limit: Option<i32>,
    verify: Option<bool>,
}

/// Structured anchor as it arrives over MCP.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnchorArg {
    path: String,
    symbol: Option<String>,
    symbol_kind: Option<String>,
    start_line: Option<u32>,
    end_line: Option<u32>,
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
        "verify" => execute_verify(storage, &args).await,
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: remember_pattern, remember_decision, get_context, verify",
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
        validity_inferred: false,
        source_envelope: None,
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

    // Anchor the memory to the code it describes so it can later be checked
    // instead of being served with unearned confidence.
    let anchors = capture_and_record(storage, &node_id, args);

    Ok(serde_json::json!({
        "action": "remember_pattern",
        "success": true,
        "nodeId": node_id,
        "patternName": name,
        "anchors": anchors,
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
        validity_inferred: false,
        source_envelope: None,
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

    let anchors = capture_and_record(storage, &node_id, args);

    Ok(serde_json::json!({
        "action": "remember_decision",
        "success": true,
        "nodeId": node_id,
        "anchors": anchors,
        "message": "Architectural decision remembered successfully",
    }))
}

// ============================================================================
// SOURCE ANCHORING
// ============================================================================
//
// A code memory used to anchor to source by printing the caller's `files`
// array into its markdown body. Nothing about that shape could answer the only
// question that matters when the memory is served back months later: does the
// code it describes still exist, and does it still say the same thing?
//
// Because nothing could answer it, a memory that had rotted came back with
// exactly the same confidence as one that was still true. The fix is not to
// delete rotted memories - the user values that memories are preserved - it is
// to make rot *visible* at retrieval time.

/// Resolve the repository root used to interpret anchor paths.
fn resolve_repo_root(args: &CodebaseArgs) -> Option<PathBuf> {
    match args.repo_path.as_deref() {
        Some(raw) if !raw.trim().is_empty() => Some(PathBuf::from(raw.trim())),
        _ => std::env::current_dir().ok(),
    }
}

/// Collect anchor drafts from both input shapes: the structured `anchors`
/// array and the existing `files` array (which now also understands the
/// compact `path#symbol` and `path:start-end` forms).
fn collect_drafts(args: &CodebaseArgs) -> Vec<AnchorDraft> {
    let mut drafts: Vec<AnchorDraft> = Vec::new();

    if let Some(anchors) = args.anchors.as_ref() {
        for a in anchors {
            if a.path.trim().is_empty() {
                continue;
            }
            drafts.push(AnchorDraft {
                file_path: a.path.trim().to_string(),
                symbol: a.symbol.clone().filter(|s| !s.trim().is_empty()),
                symbol_kind: a.symbol_kind.clone(),
                start_line: a.start_line,
                end_line: a.end_line,
            });
        }
    }

    if let Some(files) = args.files.as_ref() {
        for f in files {
            if f.trim().is_empty() {
                continue;
            }
            drafts.push(AnchorDraft::parse(f));
        }
    }

    drafts.dedup_by(|a, b| {
        a.file_path == b.file_path && a.symbol == b.symbol && a.start_line == b.start_line
    });
    drafts
}

/// Capture and persist anchors for a freshly stored memory, and describe what
/// was captured. Never fails the write: an anchor that could not be hashed is
/// still recorded (and reported) as unverifiable, which is strictly more
/// honest than recording nothing.
fn capture_and_record(storage: &Arc<Storage>, node_id: &str, args: &CodebaseArgs) -> Value {
    let drafts = collect_drafts(args);
    if drafts.is_empty() {
        return serde_json::json!({
            "count": 0,
            "verifiable": 0,
            "items": [],
            "note": "No files were anchored, so this memory cannot self-check. Pass `files: [\"src/x.rs#my_symbol\"]` or `anchors` to make it verifiable.",
        });
    }

    let Some(repo_root) = resolve_repo_root(args) else {
        return serde_json::json!({
            "count": 0,
            "verifiable": 0,
            "items": [],
            "note": "Could not determine a repository root, so no anchors were captured. Pass `repoPath` to enable staleness detection.",
        });
    };

    let anchors: Vec<CodeAnchor> = drafts
        .iter()
        .map(|d| capture_anchor(node_id, &repo_root, d))
        .collect();

    let items: Vec<Value> = anchors
        .iter()
        .map(|a| {
            serde_json::json!({
                "path": a.file_path,
                "symbol": a.symbol,
                "startLine": a.start_line,
                "endLine": a.end_line,
                "verifiable": a.is_verifiable(),
                "reason": if a.is_verifiable() {
                    Value::Null
                } else if a.symbol.is_some() {
                    Value::String("the symbol could not be located in that file, so no content hash was stored".into())
                } else {
                    Value::String("path-only anchor: no symbol or line span, so only file existence can be checked later".into())
                },
            })
        })
        .collect();

    let verifiable = anchors.iter().filter(|a| a.is_verifiable()).count();
    let recorded = storage.record_code_anchors(&anchors);

    serde_json::json!({
        "count": anchors.len(),
        "verifiable": verifiable,
        "items": items,
        "recorded": recorded.as_ref().map(|n| *n as i64).unwrap_or(0),
        "error": recorded.err().map(|e| e.to_string()),
    })
}

/// One anchor verdict, rendered for the tool response.
fn verification_json(v: &AnchorVerification) -> Value {
    serde_json::json!({
        "path": v.file_path,
        "symbol": v.symbol,
        "status": v.status.as_str(),
        "detail": v.detail,
        "recordedLine": v.recorded_line,
        "currentLine": v.current_line,
    })
}

/// Roll several anchor verdicts for one memory into a single status.
///
/// Staleness wins over freshness: if any anchor of a memory no longer matches,
/// the memory is flagged. A memory whose anchors are all unverifiable is
/// "unverifiable", never "stale" - this is the legacy path, and accusing a
/// correct memory of being wrong would be worse than the bug being fixed.
fn worst_status(verifications: &[AnchorVerification]) -> AnchorStatus {
    if verifications
        .iter()
        .any(|v| v.status == AnchorStatus::Missing)
    {
        AnchorStatus::Missing
    } else if verifications
        .iter()
        .any(|v| v.status == AnchorStatus::Drifted)
    {
        AnchorStatus::Drifted
    } else if verifications.iter().any(|v| v.status.is_fresh()) {
        // Some anchor positively matched and none contradicted it.
        if verifications
            .iter()
            .any(|v| v.status == AnchorStatus::Moved)
        {
            AnchorStatus::Moved
        } else {
            AnchorStatus::Verified
        }
    } else {
        AnchorStatus::Unverifiable
    }
}

/// Verify every anchor belonging to `node_ids` and return, per node, the rolled
/// up status plus the individual verdicts.
fn verify_nodes(
    storage: &Arc<Storage>,
    repo_root: &std::path::Path,
    node_ids: &[String],
) -> std::collections::HashMap<String, (AnchorStatus, Vec<AnchorVerification>)> {
    let mut out = std::collections::HashMap::new();
    let Ok(by_node) = storage.code_anchors_for_nodes(node_ids) else {
        return out;
    };
    for (node_id, anchors) in by_node {
        let verdicts: Vec<AnchorVerification> = anchors
            .iter()
            .map(|a| verify_anchor(a, repo_root))
            .collect();
        // Cache the verdict for reporting. The retrieval path always
        // re-verifies against the live tree, so this is never load-bearing.
        for (anchor, verdict) in anchors.iter().zip(verdicts.iter()) {
            let _ =
                storage.record_anchor_verification(&anchor.id, verdict.status, verdict.checked_at);
        }
        out.insert(node_id, (worst_status(&verdicts), verdicts));
    }
    out
}

/// Annotate already-formatted memory items with their verification verdict.
/// Returns the ids of the memories that are visibly stale.
fn annotate_items(
    items: &mut [Value],
    verified: &std::collections::HashMap<String, (AnchorStatus, Vec<AnchorVerification>)>,
) -> Vec<String> {
    let mut stale_ids = Vec::new();
    for item in items.iter_mut() {
        let Some(id) = item.get("id").and_then(|v| v.as_str()).map(str::to_string) else {
            continue;
        };
        let Some(obj) = item.as_object_mut() else {
            continue;
        };

        match verified.get(&id) {
            Some((status, verdicts)) => {
                obj.insert(
                    "anchorStatus".to_string(),
                    Value::String(status.as_str().to_string()),
                );
                obj.insert(
                    "anchors".to_string(),
                    Value::Array(verdicts.iter().map(verification_json).collect()),
                );
                if status.is_stale() {
                    stale_ids.push(id);
                    let reason = verdicts
                        .iter()
                        .filter(|v| v.is_stale())
                        .map(|v| v.detail.clone())
                        .collect::<Vec<_>>()
                        .join(" ");
                    obj.insert("stale".to_string(), Value::Bool(true));
                    obj.insert("staleReason".to_string(), Value::String(reason));
                }
            }
            None => {
                // No anchor row at all: every memory written before anchoring
                // existed lands here. Unverifiable, explicitly not stale.
                obj.insert(
                    "anchorStatus".to_string(),
                    Value::String("unanchored".to_string()),
                );
                obj.insert(
                    "anchorNote".to_string(),
                    Value::String(
                        "This memory has no source anchor, so Vestige cannot check it against the code. It may be perfectly correct - it just cannot prove it.".to_string(),
                    ),
                );
            }
        }
    }
    stale_ids
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

    // ====================================================================
    // STALENESS: a code memory that no longer matches its source must be
    // VISIBLY wrong here, not silently wrong. Nothing is deleted or rewritten
    // - the memory is returned exactly as the user saved it, with the verdict
    // attached so it cannot be read without being seen.
    // ====================================================================
    let verify_enabled = args.verify.unwrap_or(true);
    let repo_root = if verify_enabled {
        resolve_repo_root(args)
    } else {
        None
    };

    let mut verification = serde_json::json!({
        "enabled": false,
        "reason": if !verify_enabled {
            "verification disabled by the caller (verify=false)"
        } else {
            "no repository root could be resolved; pass repoPath to enable staleness detection"
        },
    });
    let mut stale_ids: Vec<String> = Vec::new();

    if let Some(root) = repo_root.as_deref() {
        let node_ids: Vec<String> = patterns
            .iter()
            .chain(decisions.iter())
            .map(|n| n.id.clone())
            .collect();
        let verified = verify_nodes(storage, root, &node_ids);

        stale_ids.extend(annotate_items(&mut formatted_patterns, &verified));
        stale_ids.extend(annotate_items(&mut formatted_decisions, &verified));

        let all: Vec<&AnchorStatus> = verified.values().map(|(s, _)| s).collect();
        let fresh = all.iter().filter(|s| s.is_fresh()).count();
        let stale = all.iter().filter(|s| s.is_stale()).count();
        let unverifiable = node_ids.len() - fresh - stale;

        verification = serde_json::json!({
            "enabled": true,
            "repoPath": root.display().to_string(),
            "checked": node_ids.len(),
            "fresh": fresh,
            "stale": stale,
            "unverifiable": unverifiable,
            "warning": if stale > 0 {
                Value::String(format!(
                    "{stale} of {} returned code memories no longer match the code they describe (`stale: true`, see `staleReason`). They are preserved, not deleted - re-read the source before acting on them.",
                    node_ids.len()
                ))
            } else {
                Value::Null
            },
            "note": if unverifiable > 0 {
                Value::String(format!(
                    "{unverifiable} memory/memories have no verifiable anchor. That means Vestige cannot check them, NOT that they are wrong. Re-save them with `files: [\"path#symbol\"]` to make them self-checking."
                ))
            } else {
                Value::Null
            },
        });
    }

    Ok(serde_json::json!({
        "action": "get_context",
        "codebase": args.codebase,
        "profile": output_config.profile.as_str(),
        "verification": verification,
        "staleMemories": stale_ids,
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

/// Re-check every anchored code memory against the working tree.
///
/// The user found their one dangerous code memory "by looking". This action is
/// that same audit, done mechanically: it reports which memories still match
/// their source, which have drifted, and which cannot be checked at all -
/// without changing or removing any of them.
async fn execute_verify(storage: &Arc<Storage>, args: &CodebaseArgs) -> Result<Value, String> {
    let repo_root = resolve_repo_root(args).ok_or(
        "Could not resolve a repository root. Pass `repoPath` pointing at the checkout to verify against.",
    )?;

    let tag_filter = args.codebase.as_ref().map(|cb| format!("codebase:{}", cb));
    let limit = args.limit.unwrap_or(200).clamp(1, 1000);

    let mut nodes = storage
        .get_nodes_by_type_and_tag("pattern", tag_filter.as_deref(), limit)
        .unwrap_or_default();
    nodes.extend(
        storage
            .get_nodes_by_type_and_tag("decision", tag_filter.as_deref(), limit)
            .unwrap_or_default(),
    );

    let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let verified = verify_nodes(storage, &repo_root, &node_ids);

    let mut stale = Vec::new();
    let mut fresh = 0usize;
    let mut unanchored: Vec<(String, f64)> = Vec::new();
    // Every anchored verdict in this sweep is one observation about the
    // capture-to-rot lag distribution (Brookmeyer & Gail backcalculation:
    // rot is only ever OBSERVED at verification time). Fresh verdicts are
    // right-censored; stale verdicts are events.
    let mut staleness_observations = Vec::new();
    let now = chrono::Utc::now();

    for node in &nodes {
        let age_days = (now - node.created_at).num_seconds() as f64 / 86400.0;
        match verified.get(&node.id) {
            Some((status, verdicts)) if status.is_stale() => {
                staleness_observations.push(
                    vestige_core::codebase::staleness::StalenessObservation {
                        age_days,
                        drifted: true,
                    },
                );
                stale.push(serde_json::json!({
                    "id": node.id,
                    "status": status.as_str(),
                    "content": node.content,
                    "anchors": verdicts.iter().map(verification_json).collect::<Vec<_>>(),
                }))
            }
            Some((status, _)) if status.is_fresh() => {
                staleness_observations.push(
                    vestige_core::codebase::staleness::StalenessObservation {
                        age_days,
                        drifted: false,
                    },
                );
                fresh += 1;
            }
            _ => unanchored.push((node.id.clone(), age_days)),
        }
    }

    // Predict for the memories verification cannot reach. The predictor
    // refuses to fit without enough evidence, and a prediction is a
    // probability shown next to the memory, never an action taken on it.
    let predictor =
        vestige_core::codebase::staleness::StalenessPredictor::fit(&staleness_observations);
    let unverifiable_memories: Vec<Value> = unanchored
        .iter()
        .map(|(id, age_days)| match &predictor {
            Some(fitted) => serde_json::json!({
                "id": id,
                "predictedStaleProbability":
                    (fitted.predict_stale_probability(*age_days) * 1000.0).round() / 1000.0,
            }),
            None => serde_json::json!({ "id": id }),
        })
        .collect();
    let staleness_prediction = match &predictor {
        Some(fitted) => serde_json::json!({
            "fitted": true,
            "driftEventsObserved": fitted.events(),
            "verificationsObserved": fitted.observations(),
            "basis": "Kaplan-Meier over this sweep's anchored verdicts (stale = event, fresh = censored)",
        }),
        None => serde_json::json!({
            "fitted": false,
            "reason": "insufficient verification history: predictions need at least 12 anchored verdicts including 4 observed drift events",
        }),
    };

    Ok(serde_json::json!({
        "action": "verify",
        "codebase": args.codebase,
        "repoPath": repo_root.display().to_string(),
        "checked": nodes.len(),
        "fresh": fresh,
        "stale": stale.len(),
        "unverifiable": unanchored.len(),
        "staleMemories": stale,
        "unverifiableMemories": unverifiable_memories,
        "stalenessPrediction": staleness_prediction,
        "message": if stale.is_empty() {
            format!("{fresh} of {} code memories still match their source. Nothing was modified or deleted.", nodes.len())
        } else {
            format!(
                "{} of {} code memories no longer match the code they describe. They are listed in full and left untouched - review them, do not assume they are correct.",
                stale.len(),
                nodes.len()
            )
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
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
        execute(&storage, &cog, &OutputConfig::default(), Some(save_args))
            .await
            .unwrap();

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
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "get_context");
        assert!(value["codebase"].is_null());
    }

    // =====================================================================
    // STALENESS DETECTION
    //
    // The production report: "4 code memories in 6 months, 1 of them actively
    // dangerous, found by looking." These tests are that audit, mechanized.
    // Each one asserts a property the pre-change code could not express,
    // because a code memory anchored to nothing but a path printed into
    // markdown has no way to be checked at all.
    // =====================================================================

    const SOURCE: &str = "\
use std::fs;

pub fn load_config(path: &str) -> Config {
    let raw = fs::read_to_string(path).unwrap();
    parse(&raw)
}
";

    fn repo_with_source(body: &str) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/state.rs"), body).unwrap();
        dir
    }

    fn rewrite_source(repo: &tempfile::TempDir, body: &str) {
        std::fs::write(repo.path().join("src/state.rs"), body).unwrap();
    }

    async fn save_anchored(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
        name: &str,
    ) -> Value {
        let args = serde_json::json!({
            "action": "remember_pattern",
            "name": name,
            "description": "load_config reads the whole file eagerly; do not call it in a loop",
            "files": ["src/state.rs#load_config"],
            "repoPath": repo.path().to_str().unwrap(),
            "codebase": "anchored"
        });
        execute(storage, cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap()
    }

    async fn get_context(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
    ) -> Value {
        let args = serde_json::json!({
            "action": "get_context",
            "codebase": "anchored",
            "repoPath": repo.path().to_str().unwrap()
        });
        execute(storage, cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn saving_with_a_symbol_anchor_records_a_verifiable_hash() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);

        let saved = save_anchored(&storage, &cog, &repo, "Eager config read").await;
        assert_eq!(saved["anchors"]["count"], 1);
        assert_eq!(
            saved["anchors"]["verifiable"], 1,
            "a `path#symbol` anchor must be content-hashed at save time"
        );
        assert_eq!(saved["anchors"]["recorded"], 1);
        assert_eq!(saved["anchors"]["items"][0]["symbol"], "load_config");
    }

    /// The actively dangerous case. The symbol still resolves, so every
    /// symbol-only or path-only scheme reports "fine", but the behavior the
    /// memory describes is gone. It must come back flagged.
    #[tokio::test]
    async fn a_code_memory_whose_source_changed_is_visibly_stale_on_retrieval() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        // Same symbol, completely different body: the memory is now wrong.
        rewrite_source(
            &repo,
            "pub fn load_config(path: &str) -> Config {\n    Config::from_env()\n}\n",
        );

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["patterns"]["items"][0];

        assert_eq!(item["anchorStatus"], "drifted", "response: {ctx}");
        assert_eq!(item["stale"], true, "a rotted memory must be flagged stale");
        assert!(
            item["staleReason"]
                .as_str()
                .unwrap()
                .contains("load_config"),
            "the reason must name what changed"
        );
        assert_eq!(ctx["verification"]["stale"], 1);
        assert!(ctx["verification"]["warning"].is_string());
        assert_eq!(ctx["staleMemories"].as_array().unwrap().len(), 1);

        // Preserved, not deleted or rewritten.
        assert_eq!(ctx["patterns"]["count"], 1);
        assert!(
            item["content"]
                .as_str()
                .unwrap()
                .contains("do not call it in a loop"),
            "the memory itself must be returned untouched"
        );
    }

    #[tokio::test]
    async fn a_deleted_source_file_makes_the_memory_visibly_stale() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        std::fs::remove_file(repo.path().join("src/state.rs")).unwrap();

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["patterns"]["items"][0];
        assert_eq!(item["anchorStatus"], "missing", "response: {ctx}");
        assert_eq!(item["stale"], true);
    }

    /// The false alarm a line-number anchor produces on every insertion above
    /// it. `state.py:552` is wrong within a week; the content hash is not.
    #[tokio::test]
    async fn code_that_only_moved_is_not_reported_as_stale() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        rewrite_source(&repo, &format!("// new header\n// more header\n\n{SOURCE}"));

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["patterns"]["items"][0];
        assert_eq!(item["anchorStatus"], "moved", "response: {ctx}");
        assert!(
            item.get("stale").is_none(),
            "a pure relocation must never be flagged stale"
        );
        assert_eq!(ctx["verification"]["stale"], 0);
        assert_eq!(ctx["verification"]["fresh"], 1);
        assert!(
            item["anchors"][0]["currentLine"].as_u64().unwrap()
                > item["anchors"][0]["recordedLine"].as_u64().unwrap(),
            "the new location should be reported"
        );
    }

    #[tokio::test]
    async fn unchanged_code_is_reported_fresh() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        let ctx = get_context(&storage, &cog, &repo).await;
        assert_eq!(ctx["patterns"]["items"][0]["anchorStatus"], "verified");
        assert_eq!(ctx["verification"]["fresh"], 1);
        assert_eq!(ctx["verification"]["stale"], 0);
    }

    /// MIGRATION SAFETY. Every code memory that already exists was written
    /// without anchors. Those must degrade to "unverifiable" - telling a user
    /// their correct memory is wrong is worse than the bug being fixed.
    #[tokio::test]
    async fn a_pre_existing_unanchored_memory_is_unverifiable_never_stale() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);

        // Exactly the old shape: files printed into markdown, no anchors.
        let args = serde_json::json!({
            "action": "remember_pattern",
            "name": "Legacy memory",
            "description": "Something true about state.rs",
            "codebase": "anchored"
        });
        execute(&storage, &cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap();

        // Rewrite the world underneath it.
        rewrite_source(&repo, "nothing like the original at all\n");

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["patterns"]["items"][0];
        assert_eq!(item["anchorStatus"], "unanchored", "response: {ctx}");
        assert!(
            item.get("stale").is_none(),
            "an unanchored memory must never be accused of being stale"
        );
        assert!(item["anchorNote"].as_str().unwrap().contains("cannot"));
        assert_eq!(ctx["verification"]["stale"], 0);
        assert_eq!(ctx["verification"]["unverifiable"], 1);
        assert!(ctx["staleMemories"].as_array().unwrap().is_empty());
    }

    /// A path-only anchor cannot be content-checked while the file exists, and
    /// must say so rather than claim verification it did not perform.
    #[tokio::test]
    async fn a_path_only_anchor_reports_itself_unverifiable() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);

        let args = serde_json::json!({
            "action": "remember_decision",
            "decision": "Config loading lives in state.rs",
            "rationale": "Keeps IO in one place",
            "files": ["src/state.rs"],
            "repoPath": repo.path().to_str().unwrap(),
            "codebase": "anchored"
        });
        let saved = execute(&storage, &cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap();
        assert_eq!(saved["anchors"]["count"], 1);
        assert_eq!(saved["anchors"]["verifiable"], 0);
        assert!(
            saved["anchors"]["items"][0]["reason"]
                .as_str()
                .unwrap()
                .contains("path-only")
        );

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["decisions"]["items"][0];
        assert_eq!(item["anchorStatus"], "unverifiable");
        assert!(item.get("stale").is_none());
    }

    #[tokio::test]
    async fn verify_action_audits_every_anchored_memory_without_changing_them() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        rewrite_source(
            &repo,
            "pub fn load_config(path: &str) -> Config {\n    Config::from_env()\n}\n",
        );

        let args = serde_json::json!({
            "action": "verify",
            "codebase": "anchored",
            "repoPath": repo.path().to_str().unwrap()
        });
        let report = execute(&storage, &cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap();

        assert_eq!(report["action"], "verify");
        assert_eq!(report["stale"], 1, "report: {report}");
        assert_eq!(report["fresh"], 0);
        assert_eq!(report["staleMemories"][0]["status"], "drifted");
        assert!(
            report["message"]
                .as_str()
                .unwrap()
                .contains("left untouched")
        );

        // The memory itself survives the audit.
        let ctx = get_context(&storage, &cog, &repo).await;
        assert_eq!(ctx["patterns"]["count"], 1);
    }

    #[tokio::test]
    async fn verification_can_be_turned_off() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        save_anchored(&storage, &cog, &repo, "Eager config read").await;

        let args = serde_json::json!({
            "action": "get_context",
            "codebase": "anchored",
            "repoPath": repo.path().to_str().unwrap(),
            "verify": false
        });
        let ctx = execute(&storage, &cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap();
        assert_eq!(ctx["verification"]["enabled"], false);
        assert!(ctx["patterns"]["items"][0].get("anchorStatus").is_none());
    }

    #[tokio::test]
    async fn schema_advertises_the_verify_action_and_anchor_inputs() {
        let schema = schema();
        let actions = schema["properties"]["action"]["enum"].as_array().unwrap();
        assert!(actions.contains(&serde_json::json!("verify")));
        assert!(schema["properties"]["anchors"].is_object());
        assert!(schema["properties"]["repoPath"].is_object());
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
