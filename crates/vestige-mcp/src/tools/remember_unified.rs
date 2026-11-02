//! `remember` MCP tool — The Single Gated Write & Inline Backfill.
//!
//! Absorbs: `smart_ingest` + `codebase(remember_pattern | remember_decision)` + `backfill` + inline dedup.
//! Upgrades:
//! - If `repo_path` or `files` are passed, automatically anchors as a codebase record.
//! - Runs Retroactive Salience Backfill inline on every save and returns `backfilled_handles: [...]` receipt in the response.
//! - If tagged "pinned", automatically re-compiles the CLAUDE.md fence on save.

use std::sync::Arc;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "content": { "type": "string", "description": "Text of the memory to save." },
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
    })
}

#[derive(Debug, Deserialize)]
struct RememberArgs {
    content: Option<String>,
    items: Option<Vec<Value>>,
    node_type: Option<String>,
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
    let mut parsed: RememberArgs = serde_json::from_value(raw_args.clone())
        .map_err(|e| format!("Invalid remember arguments: {e}"))?;

    let is_codebase = parsed.repo_path.is_some() || (parsed.files.as_ref().map_or(false, |f| !f.is_empty()));
    let mut extra_tags = Vec::new();

    if is_codebase {
        if let Some(files) = &parsed.files {
            for f in files {
                extra_tags.push(format!("file:{f}"));
            }
        }
    }

    let mut ingest_payload = raw_args.clone();
    if let Some(obj) = ingest_payload.as_object_mut() {
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

    // Step 3: Run Retroactive Salience Backfill inline on every saved node
    let scope = parsed.scope.as_deref().unwrap_or("user");
    let mut backfilled_handles = Vec::new();
    for nid in &new_node_ids {
        if let Ok(bf_res) = tools::backfill::execute(
            storage,
            Some(json!({
                "action": "run",
                "trigger_id": nid,
                "scope": scope,
                "manual": true,
            })),
        ).await {
            if let Some(changes) = bf_res.get("changes").and_then(Value::as_array) {
                for ch in changes {
                    if let Some(mid) = ch.get("memory_id").and_then(Value::as_str) {
                        if !backfilled_handles.contains(&mid.to_string()) {
                            backfilled_handles.push(mid.to_string());
                        }
                    }
                }
            }
        }
    }

    // Step 4: If tagged "pinned", automatically re-compile CLAUDE.md fence lock-free
    let has_pinned = parsed.tags.as_ref().map_or(false, |tags| tags.iter().any(|t| t == "pinned"));
    let mut recompiled_fence = false;
    if has_pinned {
        if let Ok(_) = tools::project::execute(
            storage,
            Some(json!({
                "action": "write",
                "path": "CLAUDE.md",
                "confirm": true,
            })),
        ).await {
            recompiled_fence = true;
        }
    }

    // Step 5: Format upgraded response
    let mut response = ingest_res;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("backfilled_handles".to_string(), json!(backfilled_handles));
        if has_pinned {
            obj.insert("recompiled_fence".to_string(), json!(recompiled_fence));
        }
    }
    Ok(response)
}
