//! Unified `maintain` Tool (v2.2 — Tool Consolidation)
//!
//! Folds the seven maintenance/lifecycle tools into one action-dispatched
//! surface:
//!
//!   action = consolidate | dream | gc | importance_score | backup | export | restore
//!
//! This is a thin facade: each action forwards the *same* args envelope to the
//! existing handler. None of the underlying arg structs use
//! `deny_unknown_fields`, so the `action` discriminator is ignored by each
//! handler and per-action params validate as before. Safety defaults are
//! preserved because they live inside the callees:
//!   - `gc` defaults `dry_run=true` (handler-internal),
//!   - `restore` keeps path-confinement (handler-internal),
//!   - `export` keeps its traversal guard (handler-internal).
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
const ACTIONS: [&str; 7] = [
    "consolidate",
    "dream",
    "gc",
    "importance_score",
    "backup",
    "export",
    "restore",
];

fn action_schema(action: &str) -> Option<Value> {
    Some(match action {
        "consolidate" => super::maintenance::consolidate_schema(),
        "dream" => super::dream::schema(),
        "gc" => super::maintenance::gc_schema(),
        "importance_score" => super::importance::schema(),
        "backup" => super::maintenance::backup_schema(),
        "export" => super::maintenance::export_schema(),
        "restore" => super::restore::schema(),
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
                    field["description"] = format!("[{action}] {description}").into();
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
        "description":"Store-wide maintenance: consolidate, dream, gc (preview by default), importance_score, backup, export, restore. Inspect the selected action's schema. Export uses since; start/end are unsupported."}));
    // path has different meanings in export and restore; do not hide either.
    properties.get_mut("path").unwrap()["description"] = "[export] Confined filename inside exports/. [restore] JSON archive path, confined unless allowAnyPath=true for a trusted file.".into();
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
            "Missing 'action'. Use consolidate|dream|gc|importance_score|backup|export|restore.",
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
        "gc" => super::maintenance::execute_gc(storage, args).await,
        "importance_score" => super::importance::execute(storage, cognitive, args).await,
        "backup" => super::maintenance::execute_backup(storage, args).await,
        "export" => super::maintenance::execute_export(storage, args).await,
        "restore" => super::restore::execute(storage, args).await,
        other => Err(format!(
            "Unknown maintain action '{other}'. Use consolidate|dream|gc|importance_score|backup|export|restore."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_storage() -> Arc<Storage> {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        std::mem::forget(dir);
        Arc::new(storage)
    }

    #[test]
    fn test_schema_actions() {
        let s = schema();
        let actions = s["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 7);
        assert_eq!(s["required"][0], "action");
        assert_eq!(
            s["properties"]["format"]["enum"],
            serde_json::json!(["json", "jsonl", "portable"])
        );
        for name in [
            "since",
            "max_age_days",
            "memory_count",
            "allowAnyPath",
            "merge",
            "context_topics",
        ] {
            assert!(s["properties"].get(name).is_some(), "{name}");
        }
        assert!(s["properties"].get("start").is_none());
        assert_eq!(s["oneOf"].as_array().unwrap().len(), ACTIONS.len());
    }

    #[tokio::test]
    async fn maintenance_fields_are_effective_or_explicitly_rejected() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let result = execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({
                "action":"gc", "dry_run":false, "min_retention":0.0, "max_age_days":1
            })),
        )
        .await
        .unwrap();
        assert_eq!(result["dryRun"], false);
        for args in [
            serde_json::json!({"action":"export", "start":"2026-01-01"}),
            serde_json::json!({"action":"gc", "scope":"project"}),
        ] {
            assert!(
                execute(&storage, &cognitive, Some(args))
                    .await
                    .unwrap_err()
                    .contains("does not support")
            );
        }
    }

    #[tokio::test]
    async fn test_missing_action_errors() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let r = execute(&storage, &cognitive, None).await;
        assert!(r.is_err(), "missing action must error");
    }

    #[tokio::test]
    async fn test_gc_defaults_dry_run() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        // No dry_run passed → handler default true → nothing is actually deleted.
        let args = Some(serde_json::json!({ "action": "gc" }));
        let r = execute(&storage, &cognitive, args).await.unwrap();
        // gc's envelope reports dry_run; assert it stayed true.
        let dry = r
            .get("dryRun")
            .or(r.get("dry_run"))
            .and_then(|v| v.as_bool());
        assert_eq!(
            dry,
            Some(true),
            "gc must default to dry_run=true via maintain"
        );
    }

    #[tokio::test]
    async fn test_consolidate_resolves() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let args = Some(serde_json::json!({ "action": "consolidate" }));
        assert!(execute(&storage, &cognitive, args).await.is_ok());
    }
}
