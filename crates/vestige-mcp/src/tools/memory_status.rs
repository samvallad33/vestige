//! Unified `memory_status` Tool (v2.2 — Tool Consolidation)
//!
//! Folds four read-only status/health/temporal tools into one
//! view-dispatched surface:
//!
//!   view = health (default) | retention | timeline | changelog | stats
//!
//! - `health` → full system health + statistics (the former `system_status`).
//!   Returns the byte-for-byte `system_status` shape (audit scripts parse it),
//!   including `schema_introspection` passthrough.
//! - `retention` → the lightweight retention dashboard (former `memory_health`).
//! - `timeline` → chronological browse (former `memory_timeline`).
//! - `changelog` → audit trail of memory changes (former `memory_changelog`).
//! - `stats` → full-store hygiene aggregates plus bounded diagnostic lists.
//!
//! This is a thin facade: each view forwards the *same* args envelope to the
//! existing handler. None of the underlying arg structs use
//! `deny_unknown_fields`, so the discriminator `view` is simply ignored by each
//! handler — no lossy re-scoping required, and per-view fields validate as
//! before. The `cognitive` lock is never held across a forwarded call.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

/// Discriminated-union schema for the unified `memory_status` tool.
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["health", "retention", "timeline", "changelog", "stats"],
                "default": "health",
                "description": "'health' (default): system health, stats, decay preview, warnings, recommendations. 'retention': average, distribution, trend. 'timeline': memories by date. 'changelog': state-change audit trail. 'stats': hygiene counts by type, tag, age, retention, and lifecycle, with bounded detail lists and recent tag operations."
            },
            // --- [health view] ---
            "schema_introspection": {
                "type": "boolean",
                "description": "[health view] Include the response-schema description in the output."
            },
            // --- [timeline view] ---
            "start": { "type": "string", "description": "[timeline/changelog view] Start of range (ISO 8601 date or datetime)." },
            "end": { "type": "string", "description": "[timeline/changelog view] End of range (ISO 8601 date or datetime)." },
            "node_type": { "type": "string", "description": "[timeline view] Filter by node type (e.g. 'fact', 'decision')." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[timeline view] Filter by tags (ANY match)." },
            "detail_level": {
                "type": "string", "enum": ["brief", "summary", "full"],
                "description": "[timeline view] Level of detail (default 'summary')."
            },
            // --- [changelog view] ---
            "memory_id": { "type": "string", "description": "[changelog view] Per-memory mode: state transitions for this memory id." },
            // --- [stats view] ---
            "scope": {
                "type": "string",
                "default": "user",
                "description": "[stats view] Exact normalized memory scope to aggregate (default 'user'). Ignored when all_scopes=true."
            },
            "all_scopes": {
                "type": "boolean",
                "default": false,
                "description": "[stats view] Explicitly aggregate every stored scope. Default false preserves project isolation."
            },
            // --- shared: limit (per-view ranges differ; clamped internally) ---
            "limit": {
                "type": "integer",
                "description": "Max results. timeline: default 50, max 200. changelog: default 20, max 100. stats detail lists: default 50, max 200 (aggregate counts always cover the whole store). Ignored by health and retention.",
                "minimum": 1, "maximum": 200
            }
        }
    })
}

/// Unified dispatcher for `memory_status`. Routes on `view` (default `health`).
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let view = args
        .as_ref()
        .and_then(|a| a.get("view"))
        .and_then(|v| v.as_str())
        .unwrap_or("health")
        .to_string();

    match view.as_str() {
        // Byte-for-byte system_status shape (incl. schema_introspection passthrough).
        "health" => super::maintenance::execute_system_status(storage, cognitive, args).await,
        "retention" => super::health::execute(storage, args).await,
        "timeline" => super::timeline::execute(storage, output_config, args).await,
        "changelog" => super::changelog::execute(storage, args).await,
        "stats" => super::hygiene_stats::execute(storage, args).await,
        other => Err(format!(
            "Unknown memory_status view '{other}'. Use health|retention|timeline|changelog|stats."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;

    fn test_storage() -> Arc<Storage> {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        // Keep the tempdir alive for the duration of the process by leaking it;
        // these are short-lived unit tests.
        std::mem::forget(dir);
        Arc::new(storage)
    }

    #[test]
    fn test_schema_views() {
        let s = schema();
        let views = s["properties"]["view"]["enum"].as_array().unwrap();
        assert_eq!(views.len(), 5);
        assert_eq!(s["properties"]["view"]["default"], "health");
        assert_eq!(s["properties"]["scope"]["default"], "user");
        assert_eq!(s["properties"]["all_scopes"]["default"], false);
    }

    #[tokio::test]
    async fn test_default_view_is_health() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // No args → health view → must match system_status output exactly.
        let unified = execute(&storage, &cognitive, &oc, None).await.unwrap();
        let direct = super::super::maintenance::execute_system_status(&storage, &cognitive, None)
            .await
            .unwrap();
        assert_eq!(
            unified, direct,
            "memory_status view=health must equal system_status byte-for-byte"
        );
    }

    #[tokio::test]
    async fn test_all_views_resolve() {
        let storage = test_storage();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        for view in ["health", "retention", "timeline", "changelog", "stats"] {
            let args = Some(serde_json::json!({ "view": view }));
            let r = execute(&storage, &cognitive, &oc, args).await;
            assert!(r.is_ok(), "view={view} should resolve, got {r:?}");
        }
    }
}
