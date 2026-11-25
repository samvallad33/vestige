//! `remember` MCP tool — The Single Gated Write & Inline Backfill.
//!
//! Absorbs: `smart_ingest` + `codebase(remember_pattern | remember_decision)` + `backfill` + inline dedup.
//! Upgrades:
//! - If `repo_path` or `files` are passed, automatically anchors as a codebase record.
//! - Runs Retroactive Salience Backfill inline on every save and returns `backfilled_handles: [...]` receipt in the response.
//! - If tagged "pinned", automatically re-compiles the CLAUDE.md fence on save.

use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::intention_graph::{Command, Priority};
use vestige_core::retrieval::{standing_policy_id, utterance_is_standing_policy};
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

pub fn schema() -> Value {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "action": {"type":"string", "enum":["save", "backfill"], "default":"save"},
            "params": {"type":"object", "description":"For action=backfill: native backfill arguments, including action=run or receipts."},
            "content": { "type": "string", "description": "Text of the memory to save." },
            "source": { "type": "string" },
            "outputProfile": { "type": "string", "enum": ["compact", "standard", "full"] },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "node_type": { "type": "string" },
                        "tags": { "type": "array", "items": { "type": "string" } },
                        "source": { "type": "string" }
                    },
                    "required": ["content"]
                },
                "description": "Batch save of up to 20 memories."
            },
            "node_type": {
                "type": "string",
                "enum": ["fact", "pattern", "decision", "note", "preference"],
                "default": "note",
                "description": "Kind of knowledge record."
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Exact tag handles for retrieval (e.g. ['pinned', 'auth'])."
            },
            "repo_path": { "type": "string", "description": "Git repository path to anchor codebase memory." },
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Source file anchors (e.g. ['src/auth.rs#verify_token'])."
            },
            "allow_secrets": { "type": "boolean", "default": false, "description": "Override refusal for secret-shaped content." },
            "scope": { "type": "string", "default": "user", "description": "Project or user namespace." }
        }
    });
    // Admission fields (entities, declared links, attributes, provenance, etc.)
    // remain available through the single write tool, not just through an alias.
    if let Some(properties) = tools::smart_ingest::schema()
        .get("properties")
        .and_then(Value::as_object)
    {
        for (name, value) in properties {
            schema["properties"][name] = value.clone();
        }
    }
    schema["allOf"] = json!([{
        "if":{"properties":{"action":{"const":"backfill"}},"required":["action"]},
        "then":{"properties":{"params":tools::session_unified::embedded_schema(tools::backfill::schema(), "/allOf/0/then/properties/params")},"required":["params"]}
    }]);
    schema
}

#[derive(Debug, Deserialize)]
struct RememberArgs {
    tags: Option<Vec<String>>,
    repo_path: Option<String>,
    files: Option<Vec<String>>,
    #[serde(default)]
    allow_secrets: bool,
    scope: Option<String>,
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let raw_args = args.unwrap_or_else(|| json!({}));
    let parsed: RememberArgs = serde_json::from_value(raw_args.clone())
        .map_err(|e| format!("Invalid remember arguments: {e}"))?;

    // Invisible Unicode Concealment & Trojan Source Firewall: reject hidden TAG-block
    // (`U+E0000..=U+E007F`) prompt smuggling or Bidi overrides before writing any bytes.
    if let Some(content) = raw_args.get("content").and_then(Value::as_str) {
        let hits = crate::tools::bot_fleet::scan_concealed_unicode(content);
        if let Some(first) = hits.first() {
            return Err(format!(
                "UNICODE_CONCEALMENT_REJECTED: refused invisible {} ({}) at char offset {} — nothing was written to Strata.",
                first.kind, first.codepoint, first.char_offset
            ));
        }
    }
    if let Some(items) = raw_args.get("items").and_then(Value::as_array) {
        for (idx, item) in items.iter().enumerate() {
            if let Some(content) = item.get("content").and_then(Value::as_str) {
                let hits = crate::tools::bot_fleet::scan_concealed_unicode(content);
                if let Some(first) = hits.first() {
                    return Err(format!(
                        "UNICODE_CONCEALMENT_REJECTED: refused invisible {} ({}) in items[{idx}] at char offset {} — nothing was written to Strata.",
                        first.kind, first.codepoint, first.char_offset
                    ));
                }
            }
        }
    }

    let is_codebase =
        parsed.repo_path.is_some() || parsed.files.as_ref().is_some_and(|f| !f.is_empty());
    let machine_receipt = is_machine_receipt(&raw_args, &parsed);
    let mut extra_tags = Vec::new();
    let mut extracted_file_anchors = Vec::new();

    if is_codebase && let Some(files) = &parsed.files {
        for f in files {
            let norm = normalize_file_anchor(f);
            if !norm.is_empty() {
                let tag = format!("file:{norm}");
                if !extra_tags.contains(&tag) {
                    extra_tags.push(tag);
                }
            }
        }
    }
    if let Some(content) = raw_args.get("content").and_then(Value::as_str) {
        for path in extract_inline_file_anchors(content) {
            let tag = format!("file:{path}");
            if !extra_tags.contains(&tag) {
                extra_tags.push(tag);
            }
            if !extracted_file_anchors.contains(&path) {
                extracted_file_anchors.push(path);
            }
        }
    }

    // Auto-bind HEAD commit receipt lineage (`derived_from` + `commit:<short7>`) whenever a git
    // repo path is supplied so every decision/failure/fix enters `causal_walk`'s full-depth
    // lineage graph automatically instead of stopping behind the 1-`touched`-edge cap.
    let head_lineage = if machine_receipt {
        None
    } else {
        fence_base_dir(&raw_args, &parsed)
            .and_then(|repo| resolve_head_commit_lineage(storage.as_ref(), repo))
    };
    let mut auto_linked_head_receipt: Option<String> = None;

    let mut ingest_payload = raw_args.clone();
    if let Some(obj) = ingest_payload.as_object_mut() {
        if parsed.allow_secrets {
            obj.insert("allowSecrets".into(), json!(true));
        }
        if let Some((ref sha, ref receipt_id)) = head_lineage {
            let short_tag = format!("commit:{}", &sha[..sha.len().min(7)]);
            let caller_has_commit_tag = obj
                .get("tags")
                .and_then(Value::as_array)
                .is_some_and(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .any(|t| t.starts_with("commit:"))
                });
            if !caller_has_commit_tag && !extra_tags.contains(&short_tag) {
                extra_tags.push(short_tag);
            }
            if let Some(rid) = receipt_id {
                let links_val = obj.entry("links".to_string()).or_insert_with(|| json!([]));
                if let Some(arr) = links_val.as_array_mut() {
                    let already_linked = arr.iter().any(|item| {
                        item.as_str() == Some(rid.as_str())
                            || item.get("to").and_then(Value::as_str) == Some(rid.as_str())
                    });
                    if !already_linked && arr.len() < 16 {
                        arr.push(json!({ "kind": "derived_from", "to": rid }));
                        auto_linked_head_receipt = Some(rid.clone());
                    }
                }
            }
        }
        if !extra_tags.is_empty() {
            let existing_tags = obj.entry("tags".to_string()).or_insert_with(|| json!([]));
            if let Some(arr) = existing_tags.as_array_mut() {
                for t in extra_tags {
                    if !arr.iter().any(|v| v.as_str() == Some(&t)) {
                        arr.push(json!(t));
                    }
                }
            }
        }
    }

    // Hard-Fail Causal Completeness Gate: reject incomplete engineering/codebase saves BEFORE
    // writing to Strata so agents are forced to provide repo_path, verified files, path:line
    // frames, tags, and upstream causal lineage (`links` or a Strata HEAD commit receipt).
    if !machine_receipt && requires_causal_completeness(&raw_args, &parsed) {
        enforce_causal_completeness(&CausalCompletenessCtx {
            raw: &raw_args,
            parsed: &parsed,
            repo_dir: fence_base_dir(&raw_args, &parsed),
            extracted_file_anchors: &extracted_file_anchors,
            head_lineage: head_lineage.as_ref(),
            auto_linked_head_receipt: auto_linked_head_receipt.as_deref(),
        })?;
    }

    let explicit_links: Vec<String> = ingest_payload
        .get("links")
        .or_else(|| raw_args.get("links"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_str()
                        .or_else(|| v.get("to").and_then(Value::as_str))
                        .map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();

    // Step 1: Execute Smart Ingest
    let ingest_res = tools::smart_ingest::execute(storage, cognitive, Some(ingest_payload)).await?;

    // Step 2: Extract created/reinforced node IDs
    let mut new_node_ids = Vec::new();
    if let Some(node_id) = ingest_res.get("nodeId").and_then(Value::as_str) {
        new_node_ids.push(node_id.to_string());
    } else if let Some(results) = ingest_res.get("results").and_then(Value::as_array) {
        for item in results {
            if let Some(nid) = item.get("nodeId").and_then(Value::as_str) {
                new_node_ids.push(nid.to_string());
            }
        }
    }

    // Step 2B: Strata Intake Compiler — compile exclusive slots, chiral bindings,
    // verified auto-supersession, BLAKE3 file Anchor Watch, native 9h Synaptic Capture,
    // and incremental ZSet delta updates automatically on every save.
    let scope_for_intake = parsed.scope.as_deref().unwrap_or("user");
    let explicit_files_for_intake = parsed.files.clone().unwrap_or_default();
    let repo_for_intake = fence_base_dir(&raw_args, &parsed);
    let mut intake_reports = Vec::new();
    let mut auto_superseded_links = Vec::new();
    for nid in &new_node_ids {
        if let Ok(Some(node)) = storage.get_node(nid) {
            let report = crate::cognitive_kernel::compile_ingest_write(
                storage.as_ref(),
                &crate::cognitive_kernel::IntakeWriteRequest {
                    node_id: nid,
                    content: &node.content,
                    node_type: &node.node_type,
                    tags: &node.tags,
                    scope: scope_for_intake,
                    repo_path: repo_for_intake,
                    files: &explicit_files_for_intake,
                },
            );
            for old_id in &report.auto_superseded_ids {
                auto_superseded_links.push(json!({
                    "kind": "supersedes",
                    "source": nid,
                    "target": old_id,
                }));
            }
            intake_reports.push(json!({
                "nodeId": nid,
                "report": report,
            }));
        }
    }

    // The native planner is bounded. Keep all retrieval tags, but skip scans
    // whose trigger cannot produce a change; never trim identities to fit.
    let scope = parsed.scope.as_deref().unwrap_or("user");
    let mut backfill_skipped = 0;
    let mut backfill_errors = Vec::new();
    for nid in &new_node_ids {
        let node = storage.get_node(nid).map_err(|e| e.to_string())?;
        if !node
            .as_ref()
            .is_some_and(tools::backfill::salience_eligible)
        {
            backfill_skipped += 1;
            continue;
        }
        // Automatic receipts consume trigger/target pairs, so this manual
        // fallback cannot apply the same pair twice.
        if let Err(error) = tools::backfill::execute(
            storage,
            Some(json!({
                "action": "run", "trigger_id": nid, "scope": scope,
                "manual": true, "scan_limit": 500,
            })),
        )
        .await
        {
            backfill_errors.push(json!({"trigger_id": nid, "error": error}));
        }
    }
    // Include automatic and manual deltas for this admission, with native
    // receipt IDs so clients can deduplicate without inferring scores.
    let backfill_receipts: Vec<_> = storage
        .salience_receipts()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|row| {
            row["trigger_id"]
                .as_str()
                .is_some_and(|id| new_node_ids.iter().any(|n| n == id))
        })
        .collect();
    let mut backfilled_handles: Vec<_> = backfill_receipts
        .iter()
        .filter_map(|row| row["memory_id"].as_str().map(str::to_owned))
        .collect();
    backfilled_handles.sort();
    backfilled_handles.dedup();

    // Step 4: If tagged "pinned", re-compile the rule fence. A relative path
    // without `root` is refused by `project`, so the directory is resolved here.
    let has_pinned = parsed
        .tags
        .as_ref()
        .is_some_and(|tags| tags.iter().any(|t| t == "pinned"));
    let mut recompiled_fence = false;
    let mut recompiled_fence_error: Option<String> = None;
    if has_pinned {
        let (path, root) = fence_projection_target(&raw_args, &parsed);
        let mut request = json!({
            "action": "write",
            "path": path,
            "confirm": true,
        });
        if let Some(root) = root {
            request["root"] = json!(root);
        }
        if let Some(scope) = parsed
            .scope
            .as_deref()
            .map(str::trim)
            .filter(|scope| !scope.is_empty())
        {
            request["scope"] = json!(scope);
        }
        match tools::project::execute(storage, Some(request)).await {
            Ok(_) => recompiled_fence = true,
            Err(error) => recompiled_fence_error = Some(error),
        }
    }

    // Standing policies become active intention-graph nodes. The memory
    // write already committed; a graph refusal is reported, not fatal.
    let standing_policies = register_standing_policies(storage, scope, &raw_args);

    // Wave 14M (#774): Mandatory Birth Certificate & causal-anchor auto-bind.
    let mut explicit_files: Vec<String> = parsed.files.clone().unwrap_or_default();
    for path in &extracted_file_anchors {
        if !explicit_files.contains(path) {
            explicit_files.push(path.clone());
        }
    }
    let active_receipt_hash = ingest_res
        .get("receiptId")
        .and_then(Value::as_str)
        .map(|s| *blake3::hash(s.as_bytes()).as_bytes());
    let prov_outcome = strata_gate::verify_or_autobind(strata_gate::ProvenanceInput {
        writer_id: "vestige-mcp".to_string(),
        passport_digest: None,
        session_id: scope.to_string(),
        prev_session_ts: None,
        session_ts: Utc::now().timestamp_millis().max(0) as u64,
        explicit_links,
        explicit_files,
        active_tool_receipt: active_receipt_hash,
    });
    let birth_cert_json = match prov_outcome {
        strata_gate::ProvenanceOutcome::Admitted(cert) => json!({
            "status": "admitted",
            "writer_id": cert.writer_id,
            "session_id": cert.session_id,
            "anchors_count": cert.anchors.len(),
            "auto_bound": cert.auto_bound,
        }),
        strata_gate::ProvenanceOutcome::Quarantined(q) => json!({
            "status": "quarantined",
            "code": q.code,
            "reason": q.reason,
        }),
        strata_gate::ProvenanceOutcome::ClockSpoofRejected(s) => json!({
            "status": "clock_spoof_rejected",
            "prev_ts": s.prev_ts,
            "attempted_ts": s.attempted_ts,
        }),
    };

    // Step 6: Format upgraded response
    let mut response = ingest_res;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("birth_certificate".to_string(), birth_cert_json);
        obj.insert("intake_compile".to_string(), json!(intake_reports));
        if !auto_superseded_links.is_empty() {
            let links = obj.entry("links".to_string()).or_insert_with(|| json!([]));
            if let Some(arr) = links.as_array_mut() {
                arr.extend(auto_superseded_links);
            }
        }
        obj.insert("backfilled_handles".to_string(), json!(backfilled_handles));
        obj.insert("backfill_receipts".to_string(), json!(backfill_receipts));
        obj.insert("backfill_skipped".to_string(), json!(backfill_skipped));
        obj.insert("backfill_errors".to_string(), json!(backfill_errors));
        if has_pinned {
            obj.insert("recompiled_fence".to_string(), json!(recompiled_fence));
            if let Some(error) = recompiled_fence_error {
                obj.insert("recompiled_fence_error".to_string(), json!(error));
            }
        }
        if !standing_policies.is_empty() {
            obj.insert("standing_policies".to_string(), json!(standing_policies));
        }
        if head_lineage.is_some() || !extracted_file_anchors.is_empty() {
            let (head_commit, head_receipt_id) = match &head_lineage {
                Some((sha, rid)) => (Some(sha.as_str()), rid.as_deref()),
                None => (None, None),
            };
            obj.insert(
                "causal_lineage".to_string(),
                json!({
                    "head_commit": head_commit,
                    "head_receipt_id": head_receipt_id,
                    "auto_linked": auto_linked_head_receipt.is_some(),
                    "extracted_file_anchors": extracted_file_anchors,
                }),
            );
        }
    }
    Ok(response)
}

// hardcode-scan: protocol (file extensions for source/config path:line anchor extraction)
const CODE_EXTS: &[&str] = &[
    ".rs", ".ts", ".tsx", ".js", ".mjs", ".jsx", ".svelte", ".py", ".go", ".c", ".h", ".cpp",
    ".java", ".toml", ".json", ".sql", ".sh", ".md", ".yaml", ".yml",
];

fn normalize_file_anchor(raw: &str) -> String {
    let trimmed = raw.trim().replace('\\', "/");
    let without_dot = trimmed.strip_prefix("./").unwrap_or(&trimmed);
    let path_only = without_dot.split('#').next().unwrap_or(without_dot);
    path_only.trim().to_string()
}

fn is_machine_receipt(raw: &Value, parsed: &RememberArgs) -> bool {
    if parsed.tags.as_ref().is_some_and(|tags| {
        tags.iter().any(|t| {
            t == crate::commit_receipt::AUTO_TAG
                || t == crate::commit_receipt::RECEIPT_TAG
                || t == "operator-stop"
                || t == "operator-rule"
        })
    }) {
        return true;
    }
    raw.get("source")
        .and_then(Value::as_str)
        .is_some_and(|s| s.starts_with("vestige hook ") || s.starts_with("operator-gate"))
}

fn token_is_source_file(raw: &str) -> bool {
    let cleaned = raw.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '<' | '>' | '.'
        )
    });
    if cleaned.is_empty() || cleaned.contains("://") {
        return false;
    }
    let candidate = cleaned
        .rsplit_once(':')
        .map(|(left, _)| left)
        .unwrap_or(cleaned)
        .split('#')
        .next()
        .unwrap_or(cleaned);
    let lower = candidate.to_ascii_lowercase();
    CODE_EXTS.iter().any(|ext| {
        lower
            .strip_suffix(ext)
            .is_some_and(|stem| stem.bytes().any(|b| b.is_ascii_alphanumeric()))
    })
}

fn content_mentions_source_file(content: &str) -> bool {
    content.split_whitespace().any(token_is_source_file)
}

fn requires_causal_completeness(raw: &Value, parsed: &RememberArgs) -> bool {
    if parsed.scope.as_deref() == Some("personal") {
        return false;
    }
    if parsed.repo_path.is_some()
        || raw.get("repoPath").is_some()
        || raw.get("cwd").is_some()
        || parsed.files.as_ref().is_some_and(|f| !f.is_empty())
    {
        return true;
    }
    if let Some(nt) = raw.get("node_type").and_then(Value::as_str)
        && matches!(nt, "decision" | "pattern")
    {
        return true;
    }
    if parsed.tags.as_ref().is_some_and(|tags| {
        tags.iter().any(|t| {
            t.starts_with("file:")
                || t.starts_with("commit:")
                || t.starts_with("pr:")
                || t.starts_with("issue:")
                || t == "git-commit"
        })
    }) {
        return true;
    }
    if raw
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(content_mentions_source_file)
    {
        return true;
    }
    raw.get("items")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.get("node_type")
                    .and_then(Value::as_str)
                    .is_some_and(|nt| matches!(nt, "decision" | "pattern"))
                    || item
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(content_mentions_source_file)
            })
        })
}

struct CausalCompletenessCtx<'a> {
    raw: &'a Value,
    parsed: &'a RememberArgs,
    repo_dir: Option<&'a str>,
    extracted_file_anchors: &'a [String],
    head_lineage: Option<&'a (String, Option<String>)>,
    auto_linked_head_receipt: Option<&'a str>,
}

fn file_exists_in_repo(repo_dir: &str, rel_path: &str) -> bool {
    let base = Path::new(repo_dir);
    if base.join(rel_path).exists() {
        return true;
    }
    // Deleted files that still exist in HEAD remain valid provenance anchors.
    std::process::Command::new("git")
        .args(["-C", repo_dir, "cat-file", "-e", &format!("HEAD:{rel_path}")])
        .output()
        .is_ok_and(|out| out.status.success())
}

fn has_explicit_causal_link(raw: &Value) -> bool {
    raw.get("links")
        .and_then(Value::as_array)
        .is_some_and(|arr| {
            arr.iter().any(|item| {
                let kind_ok = item.get("kind").and_then(Value::as_str).is_some_and(|k| {
                    matches!(
                        k,
                        "derived_from" | "evidence_of" | "closes" | "supersedes" | "reopened"
                    )
                });
                let to_ok = item
                    .get("to")
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.trim().is_empty());
                kind_ok && to_ok
            })
        })
}

fn enforce_causal_completeness(ctx: &CausalCompletenessCtx<'_>) -> Result<(), String> {
    let mut violations: Vec<String> = Vec::new();

    // 1. Existing repository root directory (`repo_path`)
    let valid_repo = match ctx.repo_dir {
        Some(dir) if Path::new(dir).is_dir() => Some(dir),
        Some(dir) => {
            violations.push(format!(
                "INVALID_REPO_PATH: repo_path `{dir}` does not exist on disk — pass the real absolute git worktree root."
            ));
            None
        }
        None => {
            violations.push(
                "MISSING_REPO_PATH: pass `repo_path` pointing to the absolute git repository root."
                    .to_string(),
            );
            None
        }
    };

    // 2. Explicit, non-empty, verified repo-relative `files=[...]`
    let normalized_files: Vec<String> = ctx
        .parsed
        .files
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|f| normalize_file_anchor(f))
        .filter(|f| !f.is_empty())
        .collect();
    if normalized_files.is_empty() {
        violations.push(
            "MISSING_FILES: pass non-empty `files: [\"path/to/file.ext\", ...]` listing every touched/relevant repo-relative file."
                .to_string(),
        );
    } else {
        let invalid_format: Vec<&str> = normalized_files
            .iter()
            .filter(|f| {
                let is_abs = Path::new(f).is_absolute()
                    || (f.len() >= 2 && f.as_bytes()[1] == b':');
                let has_structure = f.contains('/') || token_is_source_file(f);
                is_abs || !has_structure
            })
            .map(String::as_str)
            .collect();
        if !invalid_format.is_empty() {
            violations.push(format!(
                "INVALID_FILES_FORMAT: `files` must contain repo-relative paths (not absolute paths or bare words): {invalid_format:?}."
            ));
        }
        if let Some(repo) = valid_repo {
            let unverified: Vec<&str> = normalized_files
                .iter()
                .filter(|f| !file_exists_in_repo(repo, f))
                .map(String::as_str)
                .collect();
            if !unverified.is_empty() {
                violations.push(format!(
                    "UNVERIFIED_FILES: files not found under repo_path `{repo}` (or HEAD): {unverified:?}."
                ));
            }
        }
    }

    // 3. At least one exact `path/to/file.ext:line` frame inside `content`, and every cited inline file listed in `files`
    if ctx.extracted_file_anchors.is_empty() {
        violations.push(
            "MISSING_LINE_ANCHORS: include exact `path/to/file.ext:line` (or `path/to/file.ext:start-end`) code frames inside `content` so causal_walk stack_frame & git blame resolve."
                .to_string(),
        );
    } else if !normalized_files.is_empty() {
        let missing_from_files: Vec<&str> = ctx
            .extracted_file_anchors
            .iter()
            .filter(|anchor| !normalized_files.iter().any(|f| f == *anchor))
            .map(String::as_str)
            .collect();
        if !missing_from_files.is_empty() {
            violations.push(format!(
                "MISSING_INLINE_FILES_IN_FILES_PARAM: content cites line frames in {missing_from_files:?} that are absent from `files=[...]` — list every cited file in `files`."
            ));
        }
    }

    // 4. Explicit domain/handle tags (`tags=[...]`)
    let caller_tags = ctx
        .parsed
        .tags
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter(|t| !t.trim().is_empty())
        .count();
    if caller_tags == 0 {
        violations.push(
            "MISSING_TAGS: pass non-empty `tags: [\"...\"]` with exact handles for deterministic recall."
                .to_string(),
        );
    }

    // 5. Traversable upstream causal lineage (`links` OR Strata HEAD commit receipt)
    if !has_explicit_causal_link(ctx.raw) && ctx.auto_linked_head_receipt.is_none() {
        let head_hint = match ctx.head_lineage {
            Some((sha, None)) => format!(
                " (HEAD `{}` has no Strata commit receipt yet — run `session(action='codebase', params={{'action':'ingest_repo','repoPath':'<repo>','dryRun':false}})` or `vestige init --git-hooks`)",
                &sha[..sha.len().min(7)]
            ),
            _ => String::new(),
        };
        violations.push(format!(
            "MISSING_CAUSAL_LINEAGE: pass explicit `links: [{{\"kind\":\"derived_from\"|\"evidence_of\"|\"closes\"|\"supersedes\",\"to\":\"<memory-id>\"}}]` OR ensure HEAD has an ingested Strata commit receipt{head_hint}."
        ));
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "CAUSAL_COMPLETENESS_REJECTED ({} missing/invalid requirements — nothing was written to Strata; fix ALL items below and retry remember):\n- {}",
            violations.len(),
            violations.join("\n- ")
        ))
    }
}

/// Extract exact repo-relative `path:line` / `path:start-end` code frames written in `content`
/// so `causal_walk(start_points=[{kind: "stack_frame", ...}])` has deterministic `file:` anchors.
fn extract_inline_file_anchors(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in content.split_whitespace() {
        let cleaned = raw.trim_end_matches('.').trim_matches(|c: char| {
            matches!(
                c,
                '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '<' | '>'
            )
        });
        let Some((path_raw, line_part)) = cleaned.rsplit_once(':') else {
            continue;
        };
        let path_part = path_raw.strip_prefix("./").unwrap_or(path_raw);
        let line_clean = line_part.trim_end_matches('.');
        let valid_line = !line_clean.is_empty()
            && line_clean
                .split('-')
                .all(|seg| !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()));
        if !valid_line || path_part.is_empty() || path_part.contains("://") {
            continue;
        }
        let lower = path_part.to_ascii_lowercase();
        if !CODE_EXTS.iter().any(|ext| {
            lower
                .strip_suffix(ext)
                .is_some_and(|stem| stem.bytes().any(|b| b.is_ascii_alphanumeric()))
        }) {
            continue;
        }
        let normalized = path_part.replace('\\', "/");
        if !out.contains(&normalized) {
            out.push(normalized);
            if out.len() >= 16 {
                break;
            }
        }
    }
    out
}

/// Resolve `HEAD` SHA for `repo_dir` and look up its exact Strata commit receipt memory
/// (`commit:<sha>` / `commit:<short7>`) for automatic `derived_from` lineage chaining.
fn resolve_head_commit_lineage(storage: &Storage, repo_dir: &str) -> Option<(String, Option<String>)> {
    let dir = Path::new(repo_dir);
    if !dir.exists() {
        return None;
    }
    let output = std::process::Command::new("git")
        .args(["-C", repo_dir, "rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?.trim().to_ascii_lowercase();
    if sha.len() < 7 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let short = &sha[..7];
    let mut ids = storage.resolve_handle(&format!("commit:{sha}")).ids;
    if ids.is_empty() {
        ids = storage.resolve_handle(&format!("commit:{short}")).ids;
    }
    // Prefer the machine commit receipt node over secondary memories tagged with the same commit.
    let receipt_id = ids
        .iter()
        .find(|id| {
            storage.get_node(id).ok().flatten().is_some_and(|n| {
                n.tags
                    .iter()
                    .any(|t| t == crate::commit_receipt::RECEIPT_TAG || t == crate::commit_receipt::AUTO_TAG)
            })
        })
        .cloned()
        .or_else(|| ids.into_iter().next());
    Some((sha, receipt_id))
}

/// Directory a pinned save should project into: an explicit `cwd` or
/// `repo_path` when the caller gave one, otherwise the process working directory.
fn fence_base_dir<'a>(raw: &'a Value, parsed: &'a RememberArgs) -> Option<&'a str> {
    let from_raw = |key: &str| {
        raw.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
    };
    from_raw("cwd")
        .or_else(|| {
            parsed
                .repo_path
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .or_else(|| from_raw("repo_path"))
        .or_else(|| from_raw("repoPath"))
}

/// `(file name, root)` for the fence write.
///
/// An explicit base prefers `AGENTS.md` when that file is already there, and
/// otherwise `CLAUDE.md` in the same directory. With no base, an existing
/// `CLAUDE.md` in the process cwd wins, then an existing `AGENTS.md`, then
/// `CLAUDE.md`.
fn fence_projection_target(raw: &Value, parsed: &RememberArgs) -> (String, Option<String>) {
    if let Some(base) = fence_base_dir(raw, parsed) {
        let name = if Path::new(base).join("AGENTS.md").is_file() {
            "AGENTS.md"
        } else {
            "CLAUDE.md"
        };
        return (name.to_string(), Some(base.to_string()));
    }
    match std::env::current_dir() {
        Ok(dir) => {
            let name = if dir.join("CLAUDE.md").is_file() {
                "CLAUDE.md"
            } else if dir.join("AGENTS.md").is_file() {
                "AGENTS.md"
            } else {
                "CLAUDE.md"
            };
            (name.to_string(), Some(dir.display().to_string()))
        }
        Err(_) => ("CLAUDE.md".to_string(), None),
    }
}

fn policy_utterances(raw: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |text: &str| {
        if utterance_is_standing_policy(text) && !out.iter().any(|seen| seen == text) {
            out.push(text.to_string());
        }
    };
    if let Some(content) = raw.get("content").and_then(Value::as_str) {
        push(content);
    }
    if let Some(items) = raw.get("items").and_then(Value::as_array) {
        for item in items {
            if let Some(content) = item.get("content").and_then(Value::as_str) {
                push(content);
            }
        }
    }
    out
}

fn policy_description(text: &str) -> String {
    let mut desc: String = text
        .trim()
        .chars()
        .filter(|c| !c.is_control())
        .collect();
    const MAX_BYTES: usize = 8_192;
    if desc.len() > MAX_BYTES {
        desc.truncate(MAX_BYTES);
        while !desc.is_char_boundary(desc.len()) {
            desc.pop();
        }
    }
    desc
}

fn register_standing_policies(storage: &Arc<Storage>, scope: &str, raw: &Value) -> Vec<Value> {
    let mut registered = Vec::new();
    for text in policy_utterances(raw) {
        let description = policy_description(&text);
        if description.is_empty() {
            continue;
        }
        let id = standing_policy_id(&text);
        let command = Command::Plan {
            id: id.clone(),
            description,
            priority: Priority::High,
            requirements: Vec::new(),
            min_attention_interval_seconds: 7 * 24 * 60 * 60,
            conflict_keys: Vec::new(),
        };
        match storage.apply_intention_graph(scope, command, Utc::now()) {
            Ok(_) => registered.push(json!({"id": id, "lifecycle": "active"})),
            Err(err) if err.contains("already exists") => {
                registered.push(json!({"id": id, "lifecycle": "active"}));
            }
            Err(err) => registered.push(json!({"id": id, "error": err})),
        }
    }
    registered
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::intention_graph::Command;

    fn open() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    async fn remember(storage: &Arc<Storage>, content: &str) -> Value {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        execute(storage, &cognitive, Some(json!({ "content": content })))
            .await
            .unwrap_or_else(|err| panic!("remember failed: {err}"))
    }

    #[tokio::test]
    async fn remember_registers_a_standing_policy_on_the_intention_graph() {
        let (storage, _dir) = open();
        let text = "Always include the receipt hash.";
        let id = standing_policy_id(text);

        let saved = remember(&storage, text).await;
        let policies = saved["standing_policies"].as_array().expect("standing_policies");
        assert_eq!(policies.len(), 1, "{saved}");
        assert_eq!(policies[0]["id"], id);
        assert_eq!(policies[0]["lifecycle"], "active");
        assert!(policies[0].get("error").is_none(), "{saved}");

        let explained = storage
            .apply_intention_graph("user", Command::Explain { id: id.clone() }, Utc::now())
            .unwrap_or_else(|err| panic!("explain failed: {err}"));
        assert_eq!(explained["result"]["plan"]["lifecycle"], "active", "{explained}");
        assert_eq!(explained["result"]["plan"]["id"], id);
        assert_eq!(
            explained["result"]["current"]["description"],
            text,
            "{explained}"
        );
        assert_eq!(explained["result"]["current"]["priority"], "high");

        let again = remember(&storage, text).await;
        assert_eq!(again["standing_policies"][0]["id"], id);
        assert_eq!(again["standing_policies"][0]["lifecycle"], "active");
        assert!(again["standing_policies"][0].get("error").is_none(), "{again}");
        let replay = storage.replay_intention_graph("user").unwrap();
        assert_eq!(replay["matched"], true, "{replay}");
        assert_eq!(replay["commands"], 1, "{replay}");

        let ordinary = remember(&storage, "The cafe serves oat milk.").await;
        assert!(ordinary.get("standing_policies").is_none(), "{ordinary}");
    }

    #[test]
    fn extract_inline_file_anchors_captures_exact_path_line_frames() {
        let content = "Fixed crash in `crates/vestige-core/src/retrieval/plan.rs:142-155` and (src/auth.rs:19). Ignore https://example.com:443 and plain words.";
        let anchors = extract_inline_file_anchors(content);
        assert_eq!(
            anchors,
            vec![
                "crates/vestige-core/src/retrieval/plan.rs".to_string(),
                "src/auth.rs".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn remember_auto_links_derived_from_head_commit_receipt_for_causal_walk() {
        let (storage, _store_dir) = open();
        let repo = tempfile::tempdir().unwrap();
        let git_run = |args: &[&str]| {
            let st = std::process::Command::new("git")
                .current_dir(repo.path())
                .args(args)
                .status()
                .unwrap();
            assert!(st.success(), "git {args:?} failed");
        };
        git_run(&["init", "-q"]);
        git_run(&["config", "user.email", "test@example.com"]);
        git_run(&["config", "user.name", "Test User"]);
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/auth.rs"), "pub fn check() {}\n").unwrap();
        git_run(&["add", "src/auth.rs"]);
        git_run(&["commit", "-q", "-m", "add auth check"]);

        // Write the machine post-commit receipt first, matching real git-hook behavior.
        // `post_commit` runs its own sync tokio runtime internally, so invoke it off the async worker.
        let receipt_store = crate::commit_receipt::ReceiptStore::Local(storage.clone());
        let repo_path_owned = repo.path().to_path_buf();
        let (facts, commit_receipt_id) = tokio::task::spawn_blocking(move || {
            crate::commit_receipt::post_commit(
                &receipt_store,
                &repo_path_owned,
                Some("cargo test ok"),
            )
        })
        .await
        .expect("spawn_blocking join")
        .expect("post_commit receipt");

        // Now call remember(action='save') with complete causal anchors — ZERO manual links needed
        // because HEAD has a Strata commit receipt that auto-links via derived_from!
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let saved = execute(
            &storage,
            &cognitive,
            Some(json!({
                "content": "Auth token check updated in `src/auth.rs:1` after review.",
                "node_type": "decision",
                "repo_path": repo.path().display().to_string(),
                "files": ["src/auth.rs"],
                "tags": ["auth"],
            })),
        )
        .await
        .expect("remember save");

        assert_eq!(saved["causal_lineage"]["auto_linked"], json!(true), "{saved}");
        assert_eq!(
            saved["causal_lineage"]["head_commit"],
            json!(facts.sha),
            "{saved}"
        );
        assert_eq!(
            saved["causal_lineage"]["head_receipt_id"],
            json!(commit_receipt_id),
            "{saved}"
        );
        assert_eq!(
            saved["causal_lineage"]["extracted_file_anchors"],
            json!(["src/auth.rs"]),
            "{saved}"
        );

        let saved_id = saved["nodeId"].as_str().unwrap().to_string();
        let node = storage.get_node(&saved_id).unwrap().unwrap();
        assert!(node.tags.contains(&"file:src/auth.rs".to_string()));
        assert!(
            node.tags
                .contains(&format!("commit:{}", &facts.sha[..7]))
        );

        // Verify causal_walk follows the auto-attached derived_from edge straight to the commit receipt!
        let walk = crate::tools::causal_walk::execute(
            &storage,
            Some(json!({
                "action": "walk",
                "from_handle": saved_id,
            })),
        )
        .await
        .expect("causal_walk");
        let cause_ids: Vec<&str> = walk["causes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["id"].as_str())
            .collect();
        assert!(
            cause_ids.contains(&commit_receipt_id.as_str()),
            "causal_walk must reach HEAD commit receipt via auto-bound derived_from edge: {walk}"
        );
    }

    #[tokio::test]
    async fn completeness_gate_rejects_incomplete_codebase_saves_with_zero_writes() {
        let (storage, _store_dir) = open();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));

        // A lazy decision OR a sneaky "note" mentioning a source file must HARD-FAIL with all missing fields.
        for payload in [
            json!({
                "content": "Refactored token verification logic.",
                "node_type": "decision",
            }),
            json!({
                "content": "Fixed regression in src/auth.rs.",
                "node_type": "note",
            }),
        ] {
            let err = execute(&storage, &cognitive, Some(payload))
                .await
                .expect_err("incomplete codebase save must hard-fail");
            assert!(err.contains("CAUSAL_COMPLETENESS_REJECTED"), "{err}");
            assert!(err.contains("MISSING_REPO_PATH"), "{err}");
            assert!(err.contains("MISSING_FILES"), "{err}");
            assert!(err.contains("MISSING_LINE_ANCHORS"), "{err}");
            assert!(err.contains("MISSING_TAGS"), "{err}");
            assert!(err.contains("MISSING_CAUSAL_LINEAGE"), "{err}");
        }

        assert_eq!(
            storage.get_stats().unwrap().total_nodes,
            0,
            "rejected writes must leave zero nodes in Strata"
        );
    }

    #[tokio::test]
    async fn completeness_gate_rejects_hallucinated_files_and_unlinked_commits() {
        let (storage, _store_dir) = open();
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/real.rs"), "pub fn real() {}\n").unwrap();

        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));

        // Hallucinated file path AND missing causal lineage (no commit receipt & no links).
        let err = execute(
            &storage,
            &cognitive,
            Some(json!({
                "content": "Updated `src/fake_hallucinated.rs:10` to fix crash.",
                "node_type": "decision",
                "repo_path": repo.path().display().to_string(),
                "files": ["src/fake_hallucinated.rs"],
                "tags": ["bugfix"],
            })),
        )
        .await
        .expect_err("hallucinated file and missing lineage must hard-fail");
        assert!(err.contains("UNVERIFIED_FILES"), "{err}");
        assert!(err.contains("MISSING_CAUSAL_LINEAGE"), "{err}");

        // Real file, real path:line, real tag, real explicit link -> SUCCEEDS!
        let seed = remember(&storage, "Root failure observed on CI.").await;
        let seed_id = seed["nodeId"].as_str().unwrap();
        let ok = execute(
            &storage,
            &cognitive,
            Some(json!({
                "content": "Fixed `src/real.rs:1` root cause.",
                "node_type": "decision",
                "repo_path": repo.path().display().to_string(),
                "files": ["src/real.rs"],
                "tags": ["bugfix"],
                "links": [{ "kind": "closes", "to": seed_id }],
            })),
        )
        .await
        .expect("complete save with verified file and explicit link must pass");
        assert!(ok["nodeId"].as_str().is_some(), "{ok}");
    }

    #[tokio::test]
    async fn unicode_concealment_firewall_rejects_tag_smuggling_with_zero_writes() {
        let (storage, _store_dir) = open();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let smuggled = "Normal note\u{E0041}hidden instruction";
        let err = execute(&storage, &cognitive, Some(json!({ "content": smuggled })))
            .await
            .expect_err("Unicode TAG smuggling must be rejected");
        assert!(err.contains("UNICODE_CONCEALMENT_REJECTED"), "{err}");
        assert!(err.contains("UNICODE_TAG_SMUGGLING"), "{err}");
        assert_eq!(storage.get_stats().unwrap().total_nodes, 0);
    }
}
