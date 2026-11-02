//! `remember` MCP tool — The Single Gated Write & Inline Backfill.
//!
//! Absorbs: `smart_ingest` + `codebase(remember_pattern | remember_decision)` + `backfill` + inline dedup.
//! Upgrades:
//! - If `repo_path` or `files` are passed, automatically anchors as a codebase record.
//! - Runs Retroactive Salience Backfill inline on every save and returns `backfilled_handles: [...]` receipt in the response.
//! - If tagged "pinned", automatically re-compiles the CLAUDE.md fence on save.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
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

    let is_codebase =
        parsed.repo_path.is_some() || (parsed.files.as_ref().map_or(false, |f| !f.is_empty()));
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
        if parsed.allow_secrets {
            obj.insert("allowSecrets".into(), json!(true));
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

    // Step 4: If tagged "pinned", automatically re-compile CLAUDE.md fence lock-free
    let has_pinned = parsed
        .tags
        .as_ref()
        .map_or(false, |tags| tags.iter().any(|t| t == "pinned"));
    let mut recompiled_fence = false;
    if has_pinned {
        if let Ok(_) = tools::project::execute(
            storage,
            Some(json!({
                "action": "write",
                "path": "CLAUDE.md",
                "confirm": true,
            })),
        )
        .await
        {
            recompiled_fence = true;
        }
    }

    // Step 5: Format upgraded response
    let mut response = ingest_res;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("backfilled_handles".to_string(), json!(backfilled_handles));
        obj.insert("backfill_receipts".to_string(), json!(backfill_receipts));
        obj.insert("backfill_skipped".to_string(), json!(backfill_skipped));
        obj.insert("backfill_errors".to_string(), json!(backfill_errors));
        if has_pinned {
            obj.insert("recompiled_fence".to_string(), json!(recompiled_fence));
        }
    }
    Ok(response)
}
