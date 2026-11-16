//! Unified `maintain` Tool (v2.2 — Tool Consolidation)
//!
//! Folds the maintenance/lifecycle tools into one action-dispatched
//! surface:
//!
//!   action = consolidate | dream | dream_compile | gc | importance_score | backup | export | restore
//!
//! This is a thin facade: each action forwards the *same* args envelope to the
//! existing handler. None of the underlying arg structs use
//! `deny_unknown_fields`, so the `action` discriminator is ignored by each
//! handler and per-action params validate as before. Safety defaults are
//! preserved because they live inside the callees:
//!   - `gc` defaults `dry_run=true` (handler-internal),
//!   - `restore` keeps path-confinement (handler-internal),
//!   - `export` keeps its traversal guard (handler-internal),
//!   - `dream_compile` files everything as review-gated Memory PRs — it never
//!     mutates a memory row (handler-internal).
//!
//! The `consolidate`/`dream` *Started* events and the
//! `consolidate`/`dream`/`importance_score` *Completed* events are emitted by
//! the server dispatch + `emit_tool_event` (which normalizes the `maintain`
//! name to its effective action) — not here.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;

/// Discriminated-union schema for the unified `maintain` tool.
const ACTIONS: [&str; 10] = [
    "consolidate",
    "dream",
    "dream_compile",
    "gc",
    "importance_score",
    "backup",
    "export",
    "restore",
    "mesh_status",
    "mesh_sync",
];

fn action_schema(action: &str) -> Option<Value> {
    Some(match action {
        "consolidate" => super::maintenance::consolidate_schema(),
        "dream" => super::dream::schema(),
        "dream_compile" => super::dream_compile::schema(),
        "gc" => super::maintenance::gc_schema(),
        "importance_score" => super::importance::schema(),
        "backup" => super::maintenance::backup_schema(),
        "export" => super::maintenance::export_schema(),
        "restore" => super::restore::schema(),
        "mesh_status" => serde_json::json!({"type":"object","properties":{}}),
        "mesh_sync" => serde_json::json!({
            "type":"object",
            "properties":{
                "listen":{"type":"boolean","description":"Accept a paired peer instead of dialing."},
                "once":{"type":"boolean","description":"Accept one connection, then return."},
                "handle":{"type":"string","description":"Qualified handle to count as demand. Does not change what is admitted."}
            }
        }),
        _ => return None,
    })
}

/// Derive each action's arguments from the handler's own schema.
pub fn schema() -> Value {
    let mut properties = serde_json::Map::new();
    let mut branches = Vec::new();
    for action in ACTIONS {
        let mut branch = action_schema(action).expect("known maintenance action");
        if let Some(fields) = branch["properties"].as_object() {
            for (name, field) in fields {
                let mut field = field.clone();
                if let Some(description) = field["description"].as_str() {
                    // consolidate and restore are withheld on Strata.
                    let label = if matches!(action, "consolidate" | "restore") {
                        format!("{action}; legacy engine only")
                    } else {
                        action.to_string()
                    };
                    field["description"] = format!("[{label}] {description}").into();
                }
                properties.insert(name.clone(), field);
            }
        }
        branch["properties"]["action"] = serde_json::json!({"const":action});
        let mut required = branch["required"].as_array().cloned().unwrap_or_default();
        required.push("action".into());
        branch["required"] = required.into();
        branches.push(branch);
    }
    properties.insert("action".into(), serde_json::json!({"type":"string", "enum":ACTIONS,
        "description":"Store-wide maintenance: dream, dream_compile (replays recorded edges; the legacy engine also files reviewable PRs), gc (preview by default), importance_score, backup, export, mesh_status, mesh_sync; legacy engine only: consolidate, restore. mesh_sync reaches already-paired peers and does not pair. Inspect the selected action's schema. Export uses since; start/end are unsupported."}));
    // path has different meanings in export and restore; do not hide either.
    properties.get_mut("path").unwrap()["description"] = "[export] Confined filename inside exports/. [restore; legacy engine only] JSON archive path, confined unless allowAnyPath=true for a trusted file.".into();
    serde_json::json!({"type":"object", "properties":properties, "required":["action"], "oneOf":branches})
}

/// Unified dispatcher for `maintain`. Routes on `action` (required).
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    // Clone the discriminator out before the args envelope is moved into a callee.
    let action = args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(|v| v.as_str())
        .ok_or(
            "Missing 'action'. Use consolidate|dream|dream_compile|gc|importance_score|backup|export|restore|mesh_status|mesh_sync.",
        )?
        .to_string();

    if let Some(schema) = action_schema(&action)
        && let Some(fields) = args.as_ref().and_then(Value::as_object)
    {
        for key in fields.keys() {
            let canonical = match key.as_str() {
                "minRetention" => "min_retention",
                "maxAgeDays" => "max_age_days",
                "dryRun" => "dry_run",
                "contextTopics" => "context_topics",
                other => other,
            };
            if canonical != "action"
                && canonical != "runId"
                && schema["properties"].get(canonical).is_none()
            {
                return Err(format!(
                    "maintain action '{action}' does not support '{key}'; inspect memory_status(view='tools', tool='maintain')"
                ));
            }
        }
    }

    match action.as_str() {
        "consolidate" => super::maintenance::execute_consolidate(storage, args).await,
        "dream" => super::dream::execute(storage, cognitive, args).await,
        "dream_compile" => super::dream_compile::execute(storage, args).await,
        "gc" => super::maintenance::execute_gc(storage, args).await,
        "importance_score" => super::importance::execute(storage, cognitive, args).await,
        "backup" => super::maintenance::execute_backup(storage, args).await,
        "export" => super::maintenance::execute_export(storage, args).await,
        "restore" => super::restore::execute(storage, args).await,
        "mesh_status" => crate::mesh::status(storage).await,
        "mesh_sync" => {
            let listen = args
                .as_ref()
                .and_then(|value| value.get("listen"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let once = args
                .as_ref()
                .and_then(|value| value.get("once"))
                .and_then(|value| value.as_bool())
                .unwrap_or(listen);
            let handle = args
                .as_ref()
                .and_then(|value| value.get("handle"))
                .and_then(|value| value.as_str());
            crate::mesh::sync(storage, listen, once, handle).await
        }
        other => Err(format!(
            "Unknown maintain action '{other}'. Use consolidate|dream|dream_compile|gc|importance_score|backup|export|restore|mesh_status|mesh_sync."
        )),
    }
}
