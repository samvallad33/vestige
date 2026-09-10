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
//! - `tools` → server-resolved progressive discovery from the actual tools/list.
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

/// Progressive discovery from the exact `tools/list` result, with no second
/// registry of action names or schemas to become stale. Tool-level annotations
/// remain tool-level hints; they are not permission for individual actions.
pub fn tool_guide(catalog: &Value, args: &Value) -> Result<Value, String> {
    let selected = match args.get("tool") {
        None => None,
        Some(Value::String(name)) if !name.trim().is_empty() => Some(name.as_str()),
        Some(_) => return Err("tool must be a non-empty advertised tool name".into()),
    };
    let definitions = catalog["tools"]
        .as_array()
        .ok_or("Tool catalog unavailable")?;
    let mut entries = Vec::new();
    for definition in definitions {
        if selected.is_some_and(|name| definition["name"] != name) {
            continue;
        }
        let schema = &definition["inputSchema"];
        let mut selectors = serde_json::Map::new();
        for field in ["action", "mode", "view"] {
            if let Some(values) = schema["properties"][field]["enum"].as_array() {
                selectors.insert(
                    field.into(),
                    serde_json::json!({
                        "values": values,
                        "default": schema["properties"][field]["default"],
                    }),
                );
            }
        }
        let mut entry = serde_json::json!({
            "name": definition["name"],
            "description": definition["description"],
            "selectors": selectors,
            "toolAnnotations": definition["annotations"],
        });
        if selected.is_some() {
            entry["inputSchema"] = schema.clone();
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err(format!(
            "Unknown advertised tool '{}'; omit tool to list available names",
            selected.unwrap_or_default()
        ));
    }
    Ok(serde_json::json!({
        "view": "tools",
        "catalogVersion": env!("CARGO_PKG_VERSION"),
        "source": "tools/list",
        "compiledFeatures": {
            "embeddings": cfg!(feature = "embeddings"),
            "vectorSearch": cfg!(feature = "vector-search"),
            "connectors": cfg!(feature = "connectors"),
            "cloudSync": cfg!(feature = "cloud-sync"),
        },
        "availabilityNote": "Compiled features are not runtime readiness. Embedding-backed dedup scan/plan/apply require embeddings and vector-search; tag maintenance and its undo remain available without them. Inspect health for runtime state. Connector calls additionally require configured upstream access.",
        "tools": entries,
        "guidance": "Choose tools for the task; there is no call quota. To inspect arguments, call memory_status with view='tools' and tool='<name>'. Descriptions and schemas describe capabilities, not authorization. Mixed tools include both reads and writes; inspect the selected action and confirmation requirements. Memory matches, graph links, retention scores and receipt replay do not establish truth or causality."
    }))
}

/// Discriminated-union schema for the unified `memory_status` tool.
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["health", "retention", "timeline", "changelog", "stats", "tools"],
                "default": "health",
                "description": "'tools': current tool/action inventory, or one full schema with tool. 'health' (default): system health, stats, decay preview, warnings, recommendations. 'retention': average, distribution, trend. 'timeline': memories by date. 'changelog': state-change audit trail. 'stats': hygiene counts by type, tag, age, retention, and lifecycle, with bounded detail lists and recent tag operations."
            },
            "tool": {
                "type": "string",
                "description": "[tools view] Exact advertised tool name. Omit for a compact inventory of every tool and action; specify a name for its complete input schema. This is discovery, not tool execution."
            },
            // --- [health view] ---
            "schema_introspection": {
                "type": "boolean",
                "description": "[health] Include the response-schema description."
            },
            // --- [timeline view] ---
            "start": { "type": "string", "description": "[timeline, changelog] Range start (ISO 8601)." },
            "end": { "type": "string", "description": "[timeline, changelog] Range end (ISO 8601)." },
            "node_type": { "type": "string", "description": "[timeline] Filter by node type." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[timeline view] Filter by tags (ANY match)." },
            "detail_level": {
                "type": "string", "enum": ["brief", "summary", "full"],
                "description": "[timeline] Detail level (default 'summary')."
            },
            // --- [changelog view] ---
            "memory_id": { "type": "string", "description": "[changelog] State transitions for this memory only." },
            // --- [stats view] ---
            "scope": {
                "type": "string",
                "default": "user",
                "description": "[stats] Scope to aggregate (default 'user'); ignored when all_scopes."
            },
            "all_scopes": {
                "type": "boolean",
                "default": false,
                "description": "[stats] Aggregate every scope. Default false."
            },
            // --- shared: limit (per-view ranges differ; clamped internally) ---
            "limit": {
                "type": "integer",
                "description": "Max results: timeline 50 (max 200), changelog 20 (max 100), stats lists 50 (max 200). Ignored by health and retention.",
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
        assert_eq!(views.len(), 6);
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
