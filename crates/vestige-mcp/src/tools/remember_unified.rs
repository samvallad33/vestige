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

    let is_codebase =
        parsed.repo_path.is_some() || parsed.files.as_ref().is_some_and(|f| !f.is_empty());
    let mut extra_tags = Vec::new();

    if is_codebase && let Some(files) = &parsed.files {
        for f in files {
            extra_tags.push(format!("file:{f}"));
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
        .is_some_and(|tags| tags.iter().any(|t| t == "pinned"));
    let mut recompiled_fence = false;
    if has_pinned
        && tools::project::execute(
            storage,
            Some(json!({
                "action": "write",
                "path": "CLAUDE.md",
                "confirm": true,
            })),
        )
        .await
        .is_ok()
    {
        recompiled_fence = true;
    }

    // Standing policies become active intention-graph nodes. The memory
    // write already committed; a graph refusal is reported, not fatal.
    let standing_policies = register_standing_policies(storage, scope, &raw_args);

    // Step 6: Format upgraded response
    let mut response = ingest_res;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("backfilled_handles".to_string(), json!(backfilled_handles));
        obj.insert("backfill_receipts".to_string(), json!(backfill_receipts));
        obj.insert("backfill_skipped".to_string(), json!(backfill_skipped));
        obj.insert("backfill_errors".to_string(), json!(backfill_errors));
        if has_pinned {
            obj.insert("recompiled_fence".to_string(), json!(recompiled_fence));
        }
        if !standing_policies.is_empty() {
            obj.insert("standing_policies".to_string(), json!(standing_policies));
        }
    }
    Ok(response)
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
}
