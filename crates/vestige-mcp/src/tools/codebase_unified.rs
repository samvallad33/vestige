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
use vestige_core::codebase::{AnchorDraft, CodeAnchor, capture_anchor};
use vestige_core::{IngestInput, OutputConfig, Storage};

use super::code_context::{self, ScopeCounts, code_scopes, verification_json, verify_nodes};
use super::search_unified::apply_output_masks;

/// Memories `verify` checks per node type unless the caller says otherwise.
const DEFAULT_VERIFY_LIMIT: i32 = 200;
/// Most memories `verify` checks per node type in one call.
const MAX_VERIFY_LIMIT: i32 = 1000;

/// Input schema for the unified codebase tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember_pattern", "remember_decision", "get_context", "verify", "reanchor", "ingest_repo", "record_runs"],
                "description": "Save, list, and re-check code knowledge. 'remember_pattern' stores a code pattern, 'remember_decision' an architectural decision, 'get_context' returns both with a current-or-stale mark, 'verify' checks a bounded set of anchored memories (with a codebase, its change records too), 'reanchor' explicitly replaces reviewed source anchors for memoryId, 'ingest_repo' turns the commits of a local checkout (repoPath) into anchored change records in their own scope; it previews unless dryRun=false. 'record_runs' admits run records (or a JUnit document) onto the Strata log; an empty call records nothing."
            },
            // remember_pattern fields
            "name": {
                "type": "string",
                "description": "Pattern name (remember_pattern)."
            },
            "description": {
                "type": "string",
                "description": "Pattern description (remember_pattern)."
            },
            // remember_decision fields
            "decision": {
                "type": "string",
                "description": "The decision made (remember_decision)."
            },
            "rationale": {
                "type": "string",
                "description": "Why it was made (remember_decision)."
            },
            "alternatives": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alternatives considered (optional)."
            },
            // Shared fields
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files touched: 'src/a.py', 'src/a.py#symbol', or 'src/a.py:10-20'. Anything beyond a bare path is content-hashed so staleness is detectable."
            },
            "anchors": {
                "type": "array",
                "description": "Structured form of 'files'; each anchor is content-hashed at save time.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Repository-relative file path." },
                        "symbol": { "type": "string", "description": "Function, type or class this is about. Prefer over line numbers, which rot." },
                        "symbolKind": { "type": "string", "description": "Optional display kind, e.g. 'fn', 'class'" },
                        "startLine": { "type": "integer", "description": "1-based start line, when known." },
                        "endLine": { "type": "integer", "description": "1-based inclusive end line" }
                    },
                    "required": ["path"]
                }
            },
            "memoryId": {"type":"string", "description":"Existing code memory to reanchor after reviewing its advice against source"},
            "scope": {"type":"string", "description":"Memory namespace (default: user; for ingest_repo, the codebase name). get_context reads exactly this scope; its response lists every other scope that holds matching code memories."},
            "allScopes": {
                "type": "boolean",
                "default": false,
                "description": "get_context only: read code memories from every scope, each item tagged with its scope. Pass this or scope, not both."
            },
            "repoPath": {
                "type": "string",
                "description": "Checkout for code evidence. Required for verify and reanchor; get_context uses it for evidence; remember actions fall back to the server working directory when it is omitted."
            },
            "verify": {
                "type": "boolean",
                "description": "Re-check returned code memories against the source and flag stale ones (default true).",
                "default": true
            },
            "codebase": {
                "type": "string",
                "description": "Codebase or project identifier (ingest_repo default: the checkout's directory name)."
            },
            // get_context, verify and ingest_repo fields
            "limit": {
                "type": "integer",
                "description": "get_context: max items per category (default 10). verify: max memories checked per type (default 200, max 1000). ingest_repo: max commits read (default 100, max 500).",
                "default": 10
            },
            // ingest_repo fields
            "dryRun": {
                "type": "boolean",
                "default": true,
                "description": "ingest_repo: preview only (default true). The log is append-only, so a bulk write cannot be undone; pass false to write."
            },
            "rev": {
                "type": "string",
                "description": "ingest_repo: git revision or range to read (default HEAD), e.g. 'v1.2..v1.3' or '<sha>~1' to page further back."
            },
            "since": {
                "type": "string",
                "description": "ingest_repo: only commits after this date (git date syntax)."
            },
            "until": {
                "type": "string",
                "description": "ingest_repo: only commits before this date (git date syntax)."
            },
            "runs": {
                "type": "array",
                "description": "record_runs: run records to admit. Each item has runId, kind (test|ci|agent), subject, commit, status (passed|failed|skipped|errored), startedMs, finishedMs.",
                "items": { "type": "object" }
            },
            "junitXml": {
                "type": "string",
                "description": "record_runs: JUnit XML. One run per testcase, id {suite}::{classname}::{name}, subject {classname}::{name}."
            },
            "commit": {
                "type": "string",
                "description": "record_runs: commit copied onto each JUnit run. Exact bytes."
            }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodebaseArgs {
    action: String,
    memory_id: Option<String>,
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
    scope: Option<String>,
    codebase: Option<String>,
    // Context fields
    limit: Option<i32>,
    verify: Option<bool>,
    /// get_context: read every scope instead of one. Rejected together with
    /// an explicit `scope`.
    #[serde(alias = "all_scopes")]
    all_scopes: Option<bool>,
    // ingest_repo fields
    #[serde(alias = "dry_run")]
    dry_run: Option<bool>,
    rev: Option<String>,
    since: Option<String>,
    until: Option<String>,
    // record_runs
    runs: Option<Vec<RunArg>>,
    junit_xml: Option<String>,
    commit: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunArg {
    run_id: String,
    kind: String,
    subject: String,
    commit: Option<String>,
    status: String,
    started_ms: Option<i64>,
    finished_ms: Option<i64>,
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
        "reanchor" => execute_reanchor(storage, &args),
        "ingest_repo" => execute_ingest_repo(storage, &args).await,
        "record_runs" => execute_record_runs(storage, &args),
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: remember_pattern, remember_decision, get_context, verify, reanchor, ingest_repo, record_runs",
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

    let node = storage
        .ingest_in_scope(input, args.scope.as_deref().unwrap_or("user"))
        .map_err(|e| e.to_string())?;
    let node_id = node.id.clone();

    // ====================================================================
    // COGNITIVE: Cross-project pattern recording
    // ====================================================================
    if args.scope.as_deref().unwrap_or("user").trim() == "user"
        && let Ok(cog) = cognitive.try_lock()
    {
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

    let node = storage
        .ingest_in_scope(input, args.scope.as_deref().unwrap_or("user"))
        .map_err(|e| e.to_string())?;
    let node_id = node.id.clone();

    // ====================================================================
    // COGNITIVE: Cross-project decision recording
    // ====================================================================
    if args.scope.as_deref().unwrap_or("user").trim() == "user"
        && let Ok(cog) = cognitive.try_lock()
    {
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

/// Admit run records, or a JUnit document, onto the Strata log.
///
/// An empty call records nothing. The same run id with equal fields appends
/// nothing.
fn execute_record_runs(storage: &Arc<Storage>, args: &CodebaseArgs) -> Result<Value, String> {
    let runs = args
        .runs
        .iter()
        .flatten()
        .map(|run| super::record_runs::RunInput {
            run_id: run.run_id.clone(),
            kind: run.kind.clone(),
            subject: run.subject.clone(),
            commit: run.commit.clone().unwrap_or_default(),
            status: run.status.clone(),
            started_ms: run.started_ms.unwrap_or(0),
            finished_ms: run.finished_ms.unwrap_or(0),
        })
        .collect();
    super::record_runs::execute(
        storage.as_ref(),
        super::record_runs::Request {
            runs,
            junit_xml: args.junit_xml.clone(),
            commit: args.commit.clone().unwrap_or_default(),
        },
    )
}

/// Turn the commits of a local checkout into anchored change records. Previews
/// unless `dryRun=false`: the log is append-only, so a bulk write is opt-in.
async fn execute_ingest_repo(storage: &Arc<Storage>, args: &CodebaseArgs) -> Result<Value, String> {
    let repo_path = explicit_repo_root(args).ok_or(
        "ingest_repo requires an explicit `repoPath` pointing at the checkout to read; it does not fall back to the server working directory.",
    )?;
    let limit = match args.limit {
        None => None,
        Some(limit) if limit >= 1 => Some(limit as usize),
        Some(_) => return Err("limit must be at least 1".into()),
    };
    super::repo_ingest::execute(
        storage,
        super::repo_ingest::Request {
            repo_path,
            codebase: args.codebase.clone(),
            scope: args.scope.clone(),
            rev: args.rev.clone(),
            since: args.since.clone(),
            until: args.until.clone(),
            limit,
            dry_run: args.dry_run.unwrap_or(true),
            budget: None,
        },
    )
    .await
}

/// Explicit opt-in only: changed source is not automatically accepted as evidence.
fn execute_reanchor(storage: &Arc<Storage>, args: &CodebaseArgs) -> Result<Value, String> {
    let id = args
        .memory_id
        .as_deref()
        .ok_or("reanchor requires memoryId")?;
    let scope = args.scope.as_deref().unwrap_or("user");
    if !storage
        .node_is_in_scope(id, scope)
        .map_err(|e| e.to_string())?
    {
        return Err("Code memory not found in requested scope".into());
    }
    let raw_root = args
        .repo_path
        .as_deref()
        .filter(|p| !p.trim().is_empty())
        .ok_or("reanchor requires explicit repoPath")?;
    let root =
        std::fs::canonicalize(raw_root).map_err(|_| "reanchor requires an available checkout")?;
    if !root.is_dir() {
        return Err("repoPath must be a directory".into());
    }
    let drafts = collect_drafts(args);
    if drafts.is_empty() {
        return Err("reanchor requires explicit files or anchors reviewed by the caller".into());
    }
    let anchors: Vec<_> = drafts
        .iter()
        .map(|d| capture_anchor(id, &root, d))
        .collect();
    if anchors.iter().any(|a| !a.is_verifiable()) {
        return Err(
            "Could not capture every requested anchor; existing evidence was preserved".into(),
        );
    }
    let count = storage
        .replace_code_anchors(id, scope, &anchors)
        .map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"action":"reanchor","nodeId":id,"scope":scope,"anchorsReplaced":count,
        "hashVersion":"v2","claimVerified":false,"memoryContentChanged":false}),
    )
}

/// Resolve the repository root used to interpret anchor paths.
fn resolve_repo_root(args: &CodebaseArgs) -> Option<PathBuf> {
    match args.repo_path.as_deref() {
        Some(raw) if !raw.trim().is_empty() => Some(PathBuf::from(raw.trim())),
        _ => std::env::current_dir().ok(),
    }
}

/// The checkout named by the caller, if any. Blank values count as absent.
fn explicit_repo_root(args: &CodebaseArgs) -> Option<PathBuf> {
    args.repo_path
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
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

    let mut result = serde_json::json!({
        "count": anchors.len(),
        "verifiable": verifiable,
        "items": items,
        "recorded": recorded.as_ref().map(|n| *n as i64).unwrap_or(0),
        "error": recorded.err().map(|e| e.to_string()),
    });
    // Teach the convention at exactly the failure point: a bare `files` entry
    // anchors the path but hashes nothing, so the memory can never be checked.
    // The compact wire schema drops this tool's field prose, making this
    // response the one place a first-time caller reliably learns `path#symbol`.
    if verifiable < anchors.len() {
        result["note"] = serde_json::json!(format!(
            "{} of {} anchors are not content-verifiable (path-only, symbol not found, or unreadable file). Re-save with `files: [\"src/x.py#symbol\"]` or an explicit `path:start-end` span so this memory can be checked against the code later.",
            anchors.len() - verifiable,
            anchors.len()
        ));
    }
    result
}

/// Up to `limit` current items of `kind`, read scope by scope in the order
/// given, each paired with the scope it came from.
fn fetch_code_nodes(
    storage: &Arc<Storage>,
    kind: &str,
    codebase: Option<&str>,
    scopes: &[String],
    limit: i32,
) -> Result<Vec<(String, vestige_core::KnowledgeNode)>, String> {
    let cap = usize::try_from(limit).unwrap_or(0);
    let mut out = Vec::new();
    for scope in scopes {
        for node in code_context::current_nodes(storage, kind, codebase, scope, limit)? {
            if out.len() >= cap {
                return Ok(out);
            }
            out.push((scope.clone(), node));
        }
    }
    Ok(out)
}

/// The note on an empty `get_context` answer, built from the scope listing.
/// It says either that no scope holds any code memory for the codebase, or
/// where its patterns and decisions are instead, and where its change records
/// are: those are `event` memories, which `get_context` does not list, so
/// without this an ingested repository reads as an empty one. `None` when
/// there is nothing to point at.
fn empty_answer_note(
    rows: &[ScopeCounts],
    scope: &str,
    all_scopes: bool,
    codebase: Option<&str>,
) -> Option<String> {
    let of_codebase = codebase
        .map(|c| format!(" for codebase '{c}'"))
        .unwrap_or_default();
    if rows.is_empty() {
        return Some(format!("No code memories{of_codebase} in any scope."));
    }
    let mut note: Vec<String> = Vec::new();
    let advice_elsewhere: Vec<String> = rows
        .iter()
        .filter(|row| !all_scopes && row.scope != scope && row.has_advice())
        .map(|row| {
            format!(
                "'{}' ({} patterns, {} decisions)",
                row.scope, row.patterns, row.decisions
            )
        })
        .collect();
    if !advice_elsewhere.is_empty() {
        note.push(format!(
            "No code memories{of_codebase} in scope '{scope}', but other scopes hold some: {}. Pass scope=<name> to read one, or allScopes=true to read them all.",
            advice_elsewhere.join(", ")
        ));
    }
    // Events are only counted for a named codebase, so `codebase` is set here.
    let with_events: Vec<&ScopeCounts> = rows.iter().filter(|row| row.events > 0).collect();
    if let Some(codebase) = codebase
        && !with_events.is_empty()
    {
        if advice_elsewhere.is_empty() {
            note.push(format!(
                "No patterns or decisions{of_codebase} are recorded."
            ));
        }
        let listed = with_events
            .iter()
            .map(|row| format!("'{}' ({})", row.scope, row.events))
            .collect::<Vec<_>>()
            .join(", ");
        let (scopes_word, verify_scope) = match with_events.as_slice() {
            [only] => ("scope", only.scope.as_str()),
            _ => ("scopes", "<scope>"),
        };
        note.push(format!(
            "get_context lists patterns and decisions only; the change records (event memories) for codebase '{codebase}' are in {scopes_word} {listed}. Check them with codebase action='verify' codebase='{codebase}' scope='{verify_scope}' repoPath=<checkout>, or find one with recall handle='commit:<sha>'."
        ));
    }
    (!note.is_empty()).then(|| note.join(" "))
}

/// Get codebase context (patterns and decisions)
///
/// Reads one scope (`scope`, default `user`) or, with `allScopes`, every scope.
/// It never goes quietly empty: the response always lists the scopes that hold
/// matching code memories with their exact totals, and an empty answer for the
/// requested scope says where the memories are instead.
async fn execute_get_context(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: &CodebaseArgs,
) -> Result<Value, String> {
    // Precedence: explicit MCP param > config limit > built-in default (10).
    let limit = output_config.resolve_limit(args.limit, 10).clamp(1, 50);

    let all_scopes = args.all_scopes.unwrap_or(false);
    let requested = args
        .scope
        .as_deref()
        .map(str::trim)
        .filter(|scope| !scope.is_empty());
    if all_scopes && requested.is_some() {
        return Err("pass either scope or allScopes=true, not both".into());
    }
    let scope = requested.unwrap_or("user");

    let known = code_scopes(storage, args.codebase.as_deref());
    let read_scopes: Vec<String> = if all_scopes {
        known
            .as_ref()
            .map_err(|e| format!("allScopes needs a scope listing: {e}"))?
            .iter()
            .map(|row| row.scope.clone())
            .collect()
    } else {
        vec![scope.to_string()]
    };

    let patterns = fetch_code_nodes(
        storage,
        "pattern",
        args.codebase.as_deref(),
        &read_scopes,
        limit,
    )?;
    let decisions = fetch_code_nodes(
        storage,
        "decision",
        args.codebase.as_deref(),
        &read_scopes,
        limit,
    )?;

    let format_items = |rows: &[(String, vestige_core::KnowledgeNode)]| -> Vec<Value> {
        rows.iter()
            .map(|(item_scope, n)| {
                serde_json::json!({
                    "id": n.id,
                    "scope": item_scope,
                    "content": n.content,
                    "tags": n.tags,
                    "retentionStrength": n.retention_strength,
                    "createdAt": n.created_at.to_rfc3339(),
                })
            })
            .collect()
    };
    let mut formatted_patterns = format_items(&patterns);
    apply_output_masks(&mut formatted_patterns, output_config);
    let mut formatted_decisions = format_items(&decisions);
    apply_output_masks(&mut formatted_decisions, output_config);

    // ====================================================================
    // COGNITIVE: Cross-project knowledge discovery
    // ====================================================================
    let mut universal_patterns = Vec::new();
    if let Some(codebase_name) = &args.codebase
        && (all_scopes || scope == "user")
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
    let mut items = formatted_patterns;
    let pattern_count = items.len();
    items.extend(formatted_decisions);
    let verification = code_context::annotate(
        storage,
        &mut items,
        args.repo_path.as_deref(),
        args.verify.unwrap_or(true),
    )?;
    let stale_ids: Vec<String> = items
        .iter()
        .filter(|v| v["stale"] == true)
        .filter_map(|v| v["id"].as_str().map(str::to_owned))
        .collect();
    let formatted_decisions = items.split_off(pattern_count);
    let formatted_patterns = items;

    // Where code memories live, and an honest account of an empty answer.
    let in_view = |row: &ScopeCounts| all_scopes || row.scope == scope;
    let (scopes_json, total_patterns, total_decisions, note) = match &known {
        Ok(rows) => {
            let listed: Vec<Value> = rows
                .iter()
                .map(|row| {
                    let mut listed = row.json();
                    listed["requested"] = serde_json::json!(!all_scopes && row.scope == scope);
                    listed
                })
                .collect();
            let total_patterns: usize = rows.iter().filter(|r| in_view(r)).map(|r| r.patterns).sum();
            let total_decisions: usize =
                rows.iter().filter(|r| in_view(r)).map(|r| r.decisions).sum();
            let note = if formatted_patterns.is_empty() && formatted_decisions.is_empty() {
                empty_answer_note(rows, scope, all_scopes, args.codebase.as_deref())
            } else {
                None
            };
            (
                Value::Array(listed),
                Some(total_patterns),
                Some(total_decisions),
                note,
            )
        }
        Err(reason) => (
            serde_json::json!({"status": "unavailable", "reason": reason}),
            None,
            None,
            (formatted_patterns.is_empty() && formatted_decisions.is_empty()).then(|| {
                format!(
                    "No code memories found in scope '{scope}'; other scopes could not be listed on this backend."
                )
            }),
        ),
    };

    Ok(serde_json::json!({
        "action": "get_context",
        "scope": if all_scopes { Value::Null } else { serde_json::json!(scope) },
        "allScopes": all_scopes,
        "codebase": args.codebase,
        "profile": output_config.profile.as_str(),
        "scopes": scopes_json,
        "note": note,
        "verification": verification,
        "staleMemories": stale_ids,
        "patterns": {
            "count": formatted_patterns.len(),
            "total": total_patterns,
            "items": formatted_patterns,
        },
        "decisions": {
            "count": formatted_decisions.len(),
            "total": total_decisions,
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
    let repo_root = explicit_repo_root(args).ok_or(
        "verify requires an explicit `repoPath` pointing at the checkout to verify against; it does not fall back to the server working directory.",
    )?;

    let limit = args
        .limit
        .unwrap_or(DEFAULT_VERIFY_LIMIT)
        .clamp(1, MAX_VERIFY_LIMIT);
    // Scopes are stored trimmed, so the reads and the totals below agree on
    // every backend.
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    let codebase = args.codebase.as_deref();
    // Naming a codebase also checks its change records (ingest_repo) and any
    // other event recorded for it. Without one, `event` would mean every event
    // in the scope, so it is not read.
    let kinds: &[&str] = if codebase.is_some() {
        &["pattern", "decision", "event"]
    } else {
        &["pattern", "decision"]
    };
    let mut nodes: Vec<vestige_core::KnowledgeNode> = Vec::new();
    for kind in kinds {
        nodes.extend(code_context::current_nodes(
            storage, kind, codebase, scope, limit,
        )?);
    }

    let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let verified = verify_nodes(storage, &repo_root, &node_ids)?;

    let mut checked_by_type: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for node in &nodes {
        *checked_by_type.entry(node.node_type.clone()).or_default() += 1;
    }

    // `limit` applies per type, so a sweep that stopped short must say how
    // many it left out instead of reading as the whole scope.
    let held = code_scopes(storage, codebase)?;
    let in_scope = held.iter().find(|row| row.scope == scope);
    let mut total_by_type: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut unchecked_by_type: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for kind in kinds {
        let total = in_scope.map_or(0, |row| match *kind {
            "pattern" => row.patterns,
            "decision" => row.decisions,
            _ => row.events,
        });
        let checked = checked_by_type.get(*kind).copied().unwrap_or(0);
        if total > checked {
            unchecked_by_type.insert(kind.to_string(), total - checked);
        }
        total_by_type.insert(kind.to_string(), total);
    }
    let unchecked: usize = unchecked_by_type.values().sum();

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
                    "nodeType": node.node_type,
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

    let mut message = if stale.is_empty() {
        format!(
            "{fresh} of {} code memories still match their source. Nothing was modified or deleted.",
            nodes.len()
        )
    } else {
        format!(
            "{} of {} code memories no longer match the code they describe. They are listed in full and left untouched - review them, do not assume they are correct.",
            stale.len(),
            nodes.len()
        )
    };
    if unchecked > 0 {
        message.push_str(&format!(
            " {unchecked} more were not checked; raise limit (max {MAX_VERIFY_LIMIT})."
        ));
    }

    Ok(serde_json::json!({
        "action": "verify",
        "codebase": args.codebase,
        "repoPath": repo_root.display().to_string(),
        "checked": nodes.len(),
        "checkedByType": checked_by_type,
        "totalByType": total_by_type,
        "uncheckedByType": unchecked_by_type,
        "truncated": unchecked > 0,
        "fresh": fresh,
        "stale": stale.len(),
        "unverifiable": unanchored.len(),
        "staleMemories": stale,
        "unverifiableMemories": unverifiable_memories,
        "stalenessPrediction": staleness_prediction,
        "message": message,
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
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
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
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
        assert_eq!(
            ctx["patterns"]["items"][0]["evidence"]["state"],
            "unavailable"
        );
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

    // =====================================================================
    // THE WHOLE LOOP, END TO END
    //
    // remember (anchored to a real file) -> recall surfaces it -> the file
    // changes -> verify flags it stale AND persists that verdict -> reanchor
    // restores the evidence WITHOUT touching the memory or its FSRS state.
    // =====================================================================

    fn node_state(storage: &Arc<Storage>, id: &str) -> (String, String, f64, i32, i32) {
        let node = storage.get_node(id).unwrap().expect("node exists");
        (
            node.id,
            node.content,
            node.retention_strength,
            node.reps,
            node.lapses,
        )
    }

    /// Reanchor replaces reviewed source evidence and nothing else. The memory
    /// keeps its content and FSRS state - evidence was wrong, not the memory.
    #[tokio::test]
    async fn reanchor_preserves_the_memory_and_its_fsrs_state() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = save_anchored(&storage, &cog, &repo, "Eager config read").await;
        let id = saved["nodeId"].as_str().unwrap().to_string();

        let before = node_state(&storage, &id);
        rewrite_source(
            &repo,
            "pub fn load_config(path: &str) -> Config {\n    Config::from_env()\n}\n",
        );

        let rea = execute(
            &storage,
            &cog,
            &OutputConfig::default(),
            Some(serde_json::json!({
                "action": "reanchor", "memoryId": id,
                "repoPath": repo.path().to_str().unwrap(),
                "files": ["src/state.rs#load_config"]
            })),
        )
        .await
        .unwrap();
        assert_eq!(rea["anchorsReplaced"], 1, "response: {rea}");
        assert_eq!(rea["memoryContentChanged"], false);

        let after = node_state(&storage, &id);
        assert_eq!(before, after, "reanchor must not touch the memory row");

        let ctx = get_context(&storage, &cog, &repo).await;
        assert_eq!(ctx["patterns"]["items"][0]["anchorStatus"], "verified");
    }

    /// A stale verdict must outlive the sweep that produced it. The persisted
    /// `last_status` is what recall's codeEvidence block reads; if verify never
    /// writes it, the loop has no memory between sessions.
    #[tokio::test]
    async fn verify_action_persists_the_last_known_verdict() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = save_anchored(&storage, &cog, &repo, "Eager config read").await;
        let id = saved["nodeId"].as_str().unwrap().to_string();

        // Fresh check persists a fresh verdict.
        super::code_context::verify_nodes(&storage, repo.path(), std::slice::from_ref(&id))
            .unwrap();
        let anchors = storage.code_anchors_for_node(&id).unwrap();
        assert_eq!(
            anchors[0].last_status,
            Some(vestige_core::codebase::AnchorStatus::Verified)
        );
        assert!(anchors[0].last_verified_at.is_some());

        // The code changes; the next check persists the accusation.
        rewrite_source(
            &repo,
            "pub fn load_config(path: &str) -> Config {\n    Config::from_env()\n}\n",
        );
        super::code_context::verify_nodes(&storage, repo.path(), std::slice::from_ref(&id))
            .unwrap();
        let anchors = storage.code_anchors_for_node(&id).unwrap();
        assert_eq!(
            anchors[0].last_status,
            Some(vestige_core::codebase::AnchorStatus::Drifted)
        );

        // Reanchor resets the evidence: a fresh capture has not been checked yet.
        execute(
            &storage,
            &cog,
            &OutputConfig::default(),
            Some(serde_json::json!({
                "action": "reanchor", "memoryId": id,
                "repoPath": repo.path().to_str().unwrap(),
                "files": ["src/state.rs#load_config"]
            })),
        )
        .await
        .unwrap();
        let anchors = storage.code_anchors_for_node(&id).unwrap();
        assert_eq!(anchors[0].last_status, None);
        assert!(anchors[0].last_verified_at.is_none());
    }

    /// A bare `files` entry anchors the path but hashes nothing. The remember
    /// response must teach `path#symbol` at exactly that failure point,
    /// because the compact wire schema drops this tool's field prose.
    #[tokio::test]
    async fn remember_response_teaches_the_symbol_convention_when_unverifiable() {
        let (storage, _dir) = test_storage().await;
        let cog = test_cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = execute(
            &storage,
            &cog,
            &OutputConfig::default(),
            Some(serde_json::json!({
                "action": "remember_pattern",
                "name": "Path only",
                "description": "Anchored by bare path",
                "files": ["src/state.rs"],
                "repoPath": repo.path().to_str().unwrap(),
                "codebase": "anchored"
            })),
        )
        .await
        .unwrap();
        assert_eq!(saved["anchors"]["verifiable"], 0);
        let note = saved["anchors"]["note"].as_str().unwrap();
        assert!(note.contains("#symbol"), "note: {note}");
        assert!(note.contains("not content-verifiable"));
    }

    /// The compact wire schema caps the action description at 50 chars; the
    /// text must be shaped so truncation lands on the end of the first
    /// sentence instead of mid-list (the enum alone used to carry the rest).
    #[test]
    fn compact_schema_truncates_the_action_description_on_a_sentence_boundary() {
        let compact = crate::tools::compact::of(&schema());
        let desc = compact["properties"]["action"]["description"]
            .as_str()
            .unwrap();
        assert!(
            desc.ends_with("code knowledge."),
            "compact truncation must keep the whole first sentence, got: {desc}"
        );
        // Every action itself survives as the enum, ingest_repo included.
        let actions = compact["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 6);
        assert!(actions.contains(&serde_json::json!("ingest_repo")));
    }

    /// A NULL or blank `scope` on a legacy row can never be read (reads match
    /// `scope = ?`), so it must not be listed either, and above all it must
    /// not turn the whole scope listing into "unavailable".
    #[tokio::test]
    async fn a_null_or_blank_legacy_scope_does_not_break_the_scope_listing() {
        let (storage, dir) = test_storage().await;
        let cog = test_cognitive();
        let mut ids = Vec::new();
        for (scope, decision) in [
            ("user", "kept in user"),
            ("user", "nulled out"),
            ("proj", "kept in proj"),
            ("proj", "blanked out"),
        ] {
            let saved = execute(
                &storage,
                &cog,
                &OutputConfig::default(),
                Some(serde_json::json!({
                    "action": "remember_decision", "codebase": "legacy",
                    "decision": decision, "rationale": "scope listing test",
                    "scope": scope,
                })),
            )
            .await
            .unwrap();
            ids.push(saved["nodeId"].as_str().unwrap().to_string());
        }
        let db = rusqlite::Connection::open(dir.path().join("test.db")).unwrap();
        db.execute(
            "UPDATE knowledge_nodes SET scope = NULL WHERE id = ?1",
            rusqlite::params![ids[1]],
        )
        .unwrap();
        db.execute(
            "UPDATE knowledge_nodes SET scope = '  ' WHERE id = ?1",
            rusqlite::params![ids[3]],
        )
        .unwrap();
        drop(db);

        let ctx = execute(
            &storage,
            &cog,
            &OutputConfig::default(),
            Some(serde_json::json!({"action": "get_context", "codebase": "legacy"})),
        )
        .await
        .unwrap();
        assert_eq!(
            ctx["scopes"],
            serde_json::json!([
                {"scope": "proj", "patterns": 0, "decisions": 1, "events": 0, "requested": false},
                {"scope": "user", "patterns": 0, "decisions": 1, "events": 0, "requested": true},
            ]),
            "{ctx}"
        );
        assert_eq!(ctx["decisions"]["count"], 1, "{ctx}");
        assert_eq!(ctx["decisions"]["total"], 1, "{ctx}");
    }
}

/// The anchor contract on a Strata log, the backend the shipped binary boots.
/// Anchors are admitted through the gate and replayed from the log, so every
/// verdict here must also survive a reopen.
#[cfg(all(test, not(feature = "legacy-sqlite")))]
mod strata_tests {
    use super::*;
    use vestige_core::codebase::AnchorStatus;

    const SOURCE: &str = "\
use std::fs;

pub fn load_config(path: &str) -> Config {
    let raw = fs::read_to_string(path).unwrap();
    parse(&raw)
}
";

    const DRIFTED: &str = "pub fn load_config(path: &str) -> Config {\n    Config::from_env()\n}\n";

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    fn strata(dir: &std::path::Path) -> Arc<Storage> {
        let storage = crate::strata_memory::open(dir).unwrap();
        assert!(crate::strata_memory::is_strata_backend(storage.as_ref()));
        storage
    }

    fn repo_with_source(body: &str) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/state.rs"), body).unwrap();
        dir
    }

    fn rewrite_source(repo: &tempfile::TempDir, body: &str) {
        std::fs::write(repo.path().join("src/state.rs"), body).unwrap();
    }

    fn repo_path(repo: &tempfile::TempDir) -> &str {
        repo.path().to_str().unwrap()
    }

    async fn call(storage: &Arc<Storage>, cog: &Arc<Mutex<CognitiveEngine>>, args: Value) -> Value {
        execute(storage, cog, &OutputConfig::default(), Some(args))
            .await
            .unwrap()
    }

    async fn save_anchored(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
    ) -> Value {
        call(
            storage,
            cog,
            serde_json::json!({
                "action": "remember_pattern",
                "name": "Eager config read",
                "description": "load_config reads the whole file eagerly; do not call it in a loop",
                "files": ["src/state.rs#load_config"],
                "repoPath": repo_path(repo),
                "codebase": "anchored"
            }),
        )
        .await
    }

    async fn verify(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
    ) -> Value {
        call(
            storage,
            cog,
            serde_json::json!({"action": "verify", "codebase": "anchored", "repoPath": repo_path(repo)}),
        )
        .await
    }

    async fn get_context(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
    ) -> Value {
        call(
            storage,
            cog,
            serde_json::json!({"action": "get_context", "codebase": "anchored", "repoPath": repo_path(repo)}),
        )
        .await
    }

    async fn reanchor(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        repo: &tempfile::TempDir,
        id: &str,
    ) -> Result<Value, String> {
        execute(
            storage,
            cog,
            &OutputConfig::default(),
            Some(serde_json::json!({
                "action": "reanchor", "memoryId": id,
                "repoPath": repo_path(repo),
                "files": ["src/state.rs#load_config"]
            })),
        )
        .await
    }

    fn status_of(storage: &Arc<Storage>, id: &str) -> Option<AnchorStatus> {
        let anchors = storage.code_anchors_for_node(id).unwrap();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        anchors[0].last_status
    }

    /// remember -> verify (fresh) -> the code changes -> verify and
    /// get_context (stale) -> reanchor -> verify (fresh), then the whole
    /// state replays from the log after a reopen.
    #[tokio::test]
    async fn remember_verify_drift_reanchor_on_a_strata_log() {
        let data = tempfile::TempDir::new().unwrap();
        let storage = strata(data.path());
        let cog = cognitive();
        let repo = repo_with_source(SOURCE);

        let saved = save_anchored(&storage, &cog, &repo).await;
        let anchors = &saved["anchors"];
        assert_eq!(anchors["count"], 1, "{saved}");
        assert_eq!(anchors["verifiable"], 1, "{saved}");
        assert_eq!(anchors["recorded"], 1, "{saved}");
        assert!(anchors["error"].is_null(), "{saved}");
        let id = saved["nodeId"].as_str().unwrap().to_string();
        let captured = storage.code_anchors_for_node(&id).unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].node_id, id);
        assert_eq!(captured[0].symbol.as_deref(), Some("load_config"));
        assert!(
            captured[0]
                .content_hash
                .as_deref()
                .unwrap()
                .starts_with("v2:")
        );
        assert!(captured[0].last_status.is_none());

        let report = verify(&storage, &cog, &repo).await;
        assert_eq!(report["checked"], 1, "{report}");
        assert_eq!(report["fresh"], 1, "{report}");
        assert_eq!(report["stale"], 0, "{report}");
        assert_eq!(report["unverifiable"], 0, "{report}");
        assert!(
            report["message"]
                .as_str()
                .unwrap()
                .starts_with("1 of 1 code memories still match"),
            "{report}"
        );
        assert_eq!(status_of(&storage, &id), Some(AnchorStatus::Verified));

        let ctx = get_context(&storage, &cog, &repo).await;
        assert_eq!(ctx["patterns"]["items"][0]["id"], id.as_str());
        assert_eq!(
            ctx["patterns"]["items"][0]["anchorStatus"], "verified",
            "{ctx}"
        );
        assert_eq!(ctx["verification"]["fresh"], 1);
        assert!(ctx["staleMemories"].as_array().unwrap().is_empty());

        rewrite_source(&repo, DRIFTED);
        let report = verify(&storage, &cog, &repo).await;
        assert_eq!(report["stale"], 1, "{report}");
        assert_eq!(report["fresh"], 0, "{report}");
        assert_eq!(report["staleMemories"][0]["id"], id.as_str());
        assert_eq!(report["staleMemories"][0]["status"], "drifted");
        assert_eq!(status_of(&storage, &id), Some(AnchorStatus::Drifted));
        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["patterns"]["items"][0];
        assert_eq!(item["anchorStatus"], "drifted", "{ctx}");
        assert_eq!(item["stale"], true, "{ctx}");
        assert_eq!(ctx["staleMemories"], serde_json::json!([id.clone()]));
        assert!(
            item["content"]
                .as_str()
                .unwrap()
                .contains("do not call it in a loop"),
            "the memory is returned untouched"
        );

        let before = storage.get_node(&id).unwrap().unwrap();
        let rea = reanchor(&storage, &cog, &repo, &id).await.unwrap();
        assert_eq!(rea["anchorsReplaced"], 1, "{rea}");
        assert_eq!(rea["memoryContentChanged"], false);
        let replaced = storage.code_anchors_for_node(&id).unwrap();
        assert_eq!(replaced.len(), 1);
        assert_ne!(replaced[0].id, captured[0].id, "old evidence was replaced");
        assert_ne!(replaced[0].content_hash, captured[0].content_hash);
        assert!(replaced[0].last_status.is_none() && replaced[0].last_verified_at.is_none());
        let after = storage.get_node(&id).unwrap().unwrap();
        assert_eq!(
            (&before.content, before.reps, before.lapses),
            (&after.content, after.reps, after.lapses),
            "reanchor must not touch the memory"
        );
        let report = verify(&storage, &cog, &repo).await;
        assert_eq!(report["fresh"], 1, "{report}");
        assert_eq!(report["stale"], 0, "{report}");

        // The memory's own receipt still proves, and replay still matches.
        let receipt = storage.get_receipt(&id).unwrap().expect("write receipt");
        assert_eq!(receipt.mutations[0].kind, "created");
        let replay = storage.replay_receipt(&id).unwrap();
        assert_eq!(replay["matched"], true, "{replay}");
        drop(storage);

        let reopened = strata(data.path());
        let replayed = reopened.code_anchors_for_node(&id).unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, replaced[0].id);
        assert_eq!(replayed[0].content_hash, replaced[0].content_hash);
        assert_eq!(replayed[0].last_status, Some(AnchorStatus::Verified));
        let report = verify(&reopened, &cog, &repo).await;
        assert_eq!(report["fresh"], 1, "{report}");
        rewrite_source(&repo, SOURCE);
        let report = verify(&reopened, &cog, &repo).await;
        assert_eq!(
            report["stale"], 1,
            "the reanchored hash is the drifted body: {report}"
        );
    }

    #[tokio::test]
    async fn verify_without_an_explicit_repo_refuses_and_persists_nothing() {
        let data = tempfile::TempDir::new().unwrap();
        let storage = strata(data.path());
        let cog = cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = save_anchored(&storage, &cog, &repo).await;
        let id = saved["nodeId"].as_str().unwrap().to_string();

        for repo_path in [None, Some(""), Some("   ")] {
            let mut args = serde_json::json!({"action": "verify", "codebase": "anchored"});
            if let Some(raw) = repo_path {
                args["repoPath"] = serde_json::json!(raw);
            }
            let err = execute(&storage, &cog, &OutputConfig::default(), Some(args))
                .await
                .expect_err("verify must not fall back to the server working directory");
            assert!(err.contains("repoPath"), "error: {err}");
        }
        assert_eq!(
            status_of(&storage, &id),
            None,
            "a refused verify must not persist any verdict"
        );

        // The same memory still verifies against its real checkout.
        let report = verify(&storage, &cog, &repo).await;
        assert_eq!(report["fresh"], 1, "report: {report}");
        assert_eq!(status_of(&storage, &id), Some(AnchorStatus::Verified));
    }

    #[tokio::test]
    async fn remember_decision_records_anchors_and_an_unresolved_symbol_is_unverifiable() {
        let data = tempfile::TempDir::new().unwrap();
        let storage = strata(data.path());
        let cog = cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = call(
            &storage,
            &cog,
            serde_json::json!({
                "action": "remember_decision",
                "decision": "Config loading lives in state.rs",
                "rationale": "Keeps IO in one place",
                "files": ["src/state.rs#load_config", "src/state.rs#no_such_symbol"],
                "repoPath": repo_path(&repo),
                "codebase": "anchored"
            }),
        )
        .await;
        assert_eq!(saved["anchors"]["count"], 2, "{saved}");
        assert_eq!(saved["anchors"]["verifiable"], 1, "{saved}");
        assert_eq!(saved["anchors"]["recorded"], 2, "{saved}");
        assert!(saved["anchors"]["error"].is_null(), "{saved}");
        let id = saved["nodeId"].as_str().unwrap();
        assert_eq!(storage.code_anchors_for_node(id).unwrap().len(), 2);

        let ctx = get_context(&storage, &cog, &repo).await;
        let item = &ctx["decisions"]["items"][0];
        assert_eq!(item["anchorStatus"], "unverifiable", "{ctx}");
        assert!(item.get("stale").is_none(), "{ctx}");

        rewrite_source(&repo, DRIFTED);
        let ctx = get_context(&storage, &cog, &repo).await;
        assert_eq!(
            ctx["decisions"]["items"][0]["anchorStatus"], "drifted",
            "{ctx}"
        );
    }

    /// An edit retires the memory on Strata. The retired node's anchors stop
    /// being returned, and it can no longer be reanchored.
    #[tokio::test]
    async fn anchors_of_a_retired_memory_are_not_returned() {
        let data = tempfile::TempDir::new().unwrap();
        let storage = strata(data.path());
        let cog = cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = save_anchored(&storage, &cog, &repo).await;
        let id = saved["nodeId"].as_str().unwrap().to_string();
        let anchor_id = storage.code_anchors_for_node(&id).unwrap()[0].id.clone();

        storage
            .update_node_content(&id, "# Code Pattern: edited\n\nrewritten advice")
            .unwrap();
        assert!(storage.code_anchors_for_node(&id).unwrap().is_empty());
        assert!(
            storage
                .code_anchors_for_nodes(std::slice::from_ref(&id))
                .unwrap()
                .is_empty()
        );
        // The edit moved the anchor to the successor, so a verdict for that
        // anchor id lands there, never on the retired memory.
        storage
            .record_anchor_verification(&anchor_id, AnchorStatus::Drifted, chrono::Utc::now())
            .unwrap();
        assert!(storage.code_anchors_for_node(&id).unwrap().is_empty());
        let refused = reanchor(&storage, &cog, &repo, &id).await.unwrap_err();
        assert!(refused.contains("not found"), "{refused}");
        // The live successor still matches its source.
        let report = verify(&storage, &cog, &repo).await;
        assert_eq!(report["fresh"], 1, "{report}");
        assert_eq!(report["stale"], 0, "{report}");
    }

    #[test]
    fn store_methods_mirror_the_sqlite_contract() {
        let data = tempfile::TempDir::new().unwrap();
        let storage = strata(data.path());
        let repo = repo_with_source(SOURCE);
        let fact = storage
            .ingest_in_scope(
                IngestInput {
                    content: "a plain fact".into(),
                    node_type: "fact".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        let pattern = storage
            .ingest_in_scope(
                IngestInput {
                    content: "# Code Pattern: x".into(),
                    node_type: "pattern".into(),
                    tags: vec!["codebase".into()],
                    ..IngestInput::default()
                },
                "proj",
            )
            .unwrap();
        let draft = AnchorDraft::parse("src/state.rs#load_config");
        let first = capture_anchor(&pattern.id, repo.path(), &draft);
        let mut again = capture_anchor(&pattern.id, repo.path(), &draft);
        again.id = first.id.clone();
        again.symbol_kind = Some("fn".into());
        // Same capture instant: two captures can straddle a millisecond, and
        // this test is about replacement by id, not about the clock.
        again.captured_at = first.captured_at;

        assert_eq!(storage.record_code_anchors(&[]).unwrap(), 0);
        // A repeated id keeps the last row, as SQLite's INSERT OR REPLACE does.
        assert_eq!(
            storage
                .record_code_anchors(&[first.clone(), again.clone()])
                .unwrap(),
            2
        );
        let rows = storage.code_anchors_for_node(&pattern.id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].symbol_kind.as_deref(), Some("fn"));
        assert_eq!(
            rows[0].captured_at.timestamp_millis(),
            first.captured_at.timestamp_millis()
        );
        let unknown = capture_anchor("mem-ffffffffffffffff", repo.path(), &draft);
        assert!(storage.record_code_anchors(&[unknown]).is_err());

        let unverifiable = capture_anchor(
            &pattern.id,
            repo.path(),
            &AnchorDraft::parse("src/state.rs#missing_symbol"),
        );
        let err = storage
            .replace_code_anchors(&pattern.id, "proj", &[unverifiable])
            .unwrap_err()
            .to_string();
        assert!(err.contains("complete verifiable anchors"), "{err}");
        let fresh = capture_anchor(&pattern.id, repo.path(), &draft);
        let err = storage
            .replace_code_anchors(&pattern.id, "user", std::slice::from_ref(&fresh))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found in requested scope"), "{err}");
        let fact_anchor = capture_anchor(&fact.id, repo.path(), &draft);
        let err = storage
            .replace_code_anchors(&fact.id, "user", &[fact_anchor])
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found in requested scope"), "{err}");
        assert_eq!(
            storage
                .replace_code_anchors(&pattern.id, " proj ", std::slice::from_ref(&fresh))
                .unwrap(),
            1
        );
        let rows = storage.code_anchors_for_node(&pattern.id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, fresh.id);

        // An unknown anchor id is a no-op, as an UPDATE of zero rows is.
        storage
            .record_anchor_verification(
                "anchor-unknown",
                AnchorStatus::Verified,
                chrono::Utc::now(),
            )
            .unwrap();
        let checked_at = chrono::Utc::now();
        storage
            .record_anchor_verification(&fresh.id, AnchorStatus::Moved, checked_at)
            .unwrap();
        let rows = storage.code_anchors_for_node(&pattern.id).unwrap();
        assert_eq!(rows[0].last_status, Some(AnchorStatus::Moved));
        assert_eq!(
            rows[0].last_verified_at.map(|at| at.timestamp_millis()),
            Some(checked_at.timestamp_millis())
        );
        let by_node = storage
            .code_anchors_for_nodes(&[pattern.id.clone(), fact.id.clone()])
            .unwrap();
        assert_eq!(by_node.len(), 1);
        assert_eq!(by_node[&pattern.id].len(), 1);
    }

    /// Editing a code memory on Strata admits a successor and retires the
    /// old node. The anchors must move with it, or the edited memory would
    /// lose its source evidence.
    #[tokio::test]
    async fn editing_a_code_memory_keeps_its_anchors() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let cog = cognitive();
        let repo = repo_with_source(SOURCE);
        let saved = save_anchored(&storage, &cog, &repo).await;
        let old_id = saved["nodeId"]
            .as_str()
            .or_else(|| saved["id"].as_str())
            .expect("saved memory id")
            .to_string();
        let before = storage.code_anchors_for_node(&old_id).unwrap();
        assert_eq!(before.len(), 1, "{saved}");

        let edited = crate::tools::memory_unified::execute(
            &storage,
            &cog,
            Some(serde_json::json!({
                "action": "edit",
                "id": old_id,
                "content": "load_config still reads eagerly; cache the result"
            })),
        )
        .await
        .unwrap();
        let successor = edited["nodeId"].as_str().expect("successor id").to_string();
        assert_ne!(successor, old_id);
        let moved = storage.code_anchors_for_node(&successor).unwrap();
        assert_eq!(moved.len(), 1, "anchors must follow the edit: {edited}");
        assert_eq!(moved[0].id, before[0].id);
        assert_eq!(moved[0].content_hash, before[0].content_hash);
        assert!(storage.code_anchors_for_node(&old_id).unwrap().is_empty());
    }
    // =====================================================================
    // SCOPES: get_context must never go quietly empty
    // =====================================================================

    async fn remember_decision_in(
        storage: &Arc<Storage>,
        cog: &Arc<Mutex<CognitiveEngine>>,
        scope: Option<&str>,
        decision: &str,
    ) {
        let mut args = serde_json::json!({
            "action": "remember_decision",
            "codebase": "demo",
            "decision": decision,
            "rationale": "scope visibility test",
        });
        if let Some(scope) = scope {
            args["scope"] = serde_json::json!(scope);
        }
        call(storage, cog, args).await;
    }

    #[tokio::test]
    async fn get_context_in_the_default_scope_says_where_the_memories_are() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let cog = cognitive();
        remember_decision_in(
            &storage,
            &cog,
            Some("demo-repo"),
            "Walk only recorded edges",
        )
        .await;

        // the failing scenario: no scope asked, so `user` is read and is empty
        let ctx = call(
            &storage,
            &cog,
            serde_json::json!({"action": "get_context", "codebase": "demo"}),
        )
        .await;
        assert_eq!(ctx["scope"], "user");
        assert_eq!(ctx["decisions"]["count"], 0);
        // ...but it is no longer silent: it names the scope that holds the memory
        assert_eq!(
            ctx["scopes"],
            serde_json::json!([
                {"scope": "demo-repo", "patterns": 0, "decisions": 1, "events": 0, "requested": false}
            ]),
            "{ctx}"
        );
        let note = ctx["note"]
            .as_str()
            .expect("an empty answer carries a note");
        assert!(note.contains("scope 'user'"), "{note}");
        assert!(
            note.contains("'demo-repo' (0 patterns, 1 decisions)"),
            "{note}"
        );
        assert!(note.contains("allScopes=true"), "{note}");

        // asking for that scope returns it, and no longer calls it empty
        let ctx = call(
            &storage,
            &cog,
            serde_json::json!({"action": "get_context", "codebase": "demo", "scope": "demo-repo"}),
        )
        .await;
        assert_eq!(ctx["scope"], "demo-repo");
        assert_eq!(ctx["decisions"]["count"], 1);
        assert_eq!(ctx["decisions"]["total"], 1);
        assert_eq!(ctx["decisions"]["items"][0]["scope"], "demo-repo");
        assert!(ctx["note"].is_null(), "{ctx}");
        assert_eq!(ctx["scopes"][0]["requested"], true);
    }

    #[tokio::test]
    async fn all_scopes_reads_every_scope_and_tags_each_item() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let cog = cognitive();
        remember_decision_in(&storage, &cog, None, "decision in user").await;
        remember_decision_in(&storage, &cog, Some("demo-repo"), "decision in demo-repo").await;
        remember_decision_in(&storage, &cog, Some("other"), "decision in other").await;

        for key in ["allScopes", "all_scopes"] {
            let mut args = serde_json::json!({"action": "get_context", "codebase": "demo"});
            args[key] = serde_json::json!(true);
            let ctx = call(&storage, &cog, args).await;
            assert_eq!(ctx["allScopes"], true, "{key}");
            assert!(ctx["scope"].is_null(), "{key}: no single scope was read");
            assert_eq!(ctx["decisions"]["count"], 3, "{key}: {ctx}");
            assert_eq!(ctx["decisions"]["total"], 3, "{key}");
            let mut seen: Vec<&str> = ctx["decisions"]["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["scope"].as_str().unwrap())
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, vec!["demo-repo", "other", "user"], "{key}");
            assert!(ctx["note"].is_null(), "{key}: {ctx}");
            let listed: Vec<&str> = ctx["scopes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["scope"].as_str().unwrap())
                .collect();
            assert_eq!(
                listed,
                vec!["demo-repo", "other", "user"],
                "{key}: ordered by name"
            );
        }

        // one scope, asked for exactly: the others are still listed, not read
        let ctx = call(
            &storage,
            &cog,
            serde_json::json!({"action": "get_context", "codebase": "demo"}),
        )
        .await;
        assert_eq!(ctx["decisions"]["count"], 1);
        assert_eq!(ctx["scopes"].as_array().unwrap().len(), 3);
        assert!(ctx["note"].is_null());
    }

    #[tokio::test]
    async fn scope_and_all_scopes_together_are_refused_not_guessed() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let err = execute(
            &storage,
            &cognitive(),
            &OutputConfig::default(),
            Some(serde_json::json!({
                "action": "get_context", "codebase": "demo",
                "scope": "demo-repo", "allScopes": true
            })),
        )
        .await
        .unwrap_err();
        assert!(err.contains("not both"), "{err}");
    }

    #[tokio::test]
    async fn an_empty_store_says_no_scope_holds_anything() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let ctx = call(
            &storage,
            &cognitive(),
            serde_json::json!({"action": "get_context", "codebase": "nowhere"}),
        )
        .await;
        assert_eq!(ctx["scopes"], serde_json::json!([]));
        assert_eq!(
            ctx["note"],
            "No code memories for codebase 'nowhere' in any scope."
        );
        assert_eq!(ctx["patterns"]["total"], 0);
    }

    #[tokio::test]
    async fn totals_show_what_the_limit_cut() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = strata(dir.path());
        let cog = cognitive();
        for i in 0..3 {
            call(
                &storage,
                &cog,
                serde_json::json!({
                    "action": "remember_pattern", "codebase": "demo",
                    "name": format!("pattern {i}"), "description": format!("description {i}"),
                }),
            )
            .await;
        }
        let ctx = call(
            &storage,
            &cog,
            serde_json::json!({"action": "get_context", "codebase": "demo", "limit": 2}),
        )
        .await;
        assert_eq!(ctx["patterns"]["count"], 2, "{ctx}");
        assert_eq!(
            ctx["patterns"]["total"], 3,
            "the cut is visible, not silent"
        );
        assert_eq!(ctx["scopes"][0]["patterns"], 3);
    }
}

/// The schema states each action's defaults and caps. Checked against the
/// constants that enforce them, on every feature set.
#[cfg(test)]
mod schema_text {
    use super::*;
    use crate::tools::repo_ingest;

    fn description(field: &str) -> String {
        schema()["properties"][field]["description"]
            .as_str()
            .unwrap_or_else(|| panic!("{field} has a description"))
            .to_string()
    }

    #[test]
    fn limit_scope_and_codebase_state_the_defaults_each_action_applies() {
        let limit = description("limit");
        assert!(
            limit.contains(&format!(
                "ingest_repo: max commits read (default {}, max {})",
                repo_ingest::DEFAULT_LIMIT,
                repo_ingest::MAX_LIMIT
            )),
            "{limit}"
        );
        assert!(
            limit.contains(&format!(
                "verify: max memories checked per type (default {DEFAULT_VERIFY_LIMIT}, max {MAX_VERIFY_LIMIT})"
            )),
            "{limit}"
        );
        let scope = description("scope");
        assert!(
            scope.contains("for ingest_repo, the codebase name"),
            "{scope}"
        );
        let codebase = description("codebase");
        assert!(
            codebase.contains("ingest_repo default: the checkout's directory name"),
            "{codebase}"
        );
    }
}
