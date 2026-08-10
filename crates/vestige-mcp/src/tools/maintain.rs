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
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["consolidate", "dream", "gc", "importance_score", "backup", "export", "restore"],
                "description": "Maintenance op. 'consolidate' (run FSRS-6 decay/embedding cycle), 'dream' (replay memories → insights/connections), 'gc' (garbage-collect stale memories; dry_run=true by default), 'importance_score' (4-channel neuroscience score for 'content'), 'backup' (SQLite DB backup), 'export' (memories as JSON/JSONL with filters), 'restore' (restore from a JSON backup at 'path')."
            },
            // --- gc ---
            "min_retention": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "[gc] Collect memories below this retention (default 0.1)." },
            "dry_run": { "type": "boolean", "description": "[gc] Preview only. Defaults to TRUE for safety." },
            // --- importance_score ---
            "content": { "type": "string", "description": "[importance_score] Content to score." },
            // --- export ---
            "format": { "type": "string", "enum": ["json", "jsonl"], "description": "[export] Output format." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[export] Tag filter." },
            "start": { "type": "string", "description": "[export] Start date filter (ISO 8601)." },
            "end": { "type": "string", "description": "[export] End date filter (ISO 8601)." },
            // --- backup / restore ---
            "path": { "type": "string", "description": "[restore] Path to a JSON backup file (path-confined)." }
        },
        "required": ["action"]
    })
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
