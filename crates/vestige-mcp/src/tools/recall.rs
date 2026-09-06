//! Unified `recall` Tool (v2.2 — Tool Consolidation, HOT PATH)
//!
//! Folds the four retrieval/reasoning tools into one mode-dispatched surface:
//!
//!   mode = lookup (DEFAULT) | reason | contradictions
//!
//! - `lookup` (default) → hybrid search (the former `search`). This is the hot
//!   path: with no `mode` set, `recall` is a ZERO-overhead pass-through to
//!   `search_unified::execute` — it must never pay the cost of the reasoning
//!   path. (`deep_reference`/`reason` runs spreading activation + FSRS trust
//!   scoring + contradiction analysis and is 5–10× slower.)
//! - `reason` → deep cognitive reasoning across memories (former
//!   `deep_reference` / `cross_reference`).
//! - `contradictions` → trust-weighted disagreement pairs (former
//!   `contradictions`).
//!
//! The schema is derived from `search_unified::schema()` (so every lookup
//! parameter stays available and documented) plus the `mode` discriminator and
//! the reason/contradictions fields. `query` is NOT globally required because
//! the contradictions mode is scoped by `topic`; per-mode requirements are
//! validated at runtime.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

/// Discriminated-union schema for the unified `recall` tool.
///
/// Built on top of `search_unified::schema()` so all lookup parameters carry
/// through verbatim; the `required: ["query"]` constraint is dropped (validated
/// per-mode at runtime) and the mode/reason/contradictions fields are added.
pub fn schema() -> Value {
    let mut schema = super::search_unified::schema();

    if let Some(obj) = schema.as_object_mut() {
        // Drop the global `query` requirement — contradictions uses `topic`.
        obj.remove("required");

        if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
            props.insert(
                "mode".to_string(),
                serde_json::json!({
                    "type": "string",
                    "enum": ["lookup", "reason", "contradictions"],
                    "default": "lookup",
                    "description": "'lookup' (default): fast hybrid search. 'reason': deep pass with trust scoring, spreading activation, supersession, and contradiction analysis; needs 'query'. Its reasoning text is assembled by the server from those computed values (confidenceBreakdown shows the arithmetic); no model writes it. 'contradictions': trust-weighted disagreement pairs for a 'topic', or recent memories."
                }),
            );
            // reason (deep_reference) extra field.
            props.insert(
                "depth".to_string(),
                serde_json::json!({
                    "type": "integer",
                    "description": "[reason mode] How many memories to analyze (default 20, max 50).",
                    "minimum": 5, "maximum": 50
                }),
            );
            // contradictions extra fields.
            props.insert(
                "topic".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "[contradictions mode] Topic to scope contradiction detection. If omitted, scans recent memories."
                }),
            );
            props.insert(
                "since".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "[contradictions mode] RFC3339 timestamp; only memories updated after this are considered."
                }),
            );
            props.insert(
                "min_trust".to_string(),
                serde_json::json!({
                    "type": "number",
                    "minimum": 0.0, "maximum": 1.0,
                    "description": "[contradictions mode] Minimum trust for both sides of a contradiction (default 0.3)."
                }),
            );
        }
    }

    schema
}

/// Unified dispatcher for `recall`. Routes on `mode` (default `lookup`).
///
/// HOT-PATH INVARIANT: `mode` absent ⇒ `lookup` ⇒ direct pass-through to
/// `search_unified::execute`, no extra work.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let mode = args
        .as_ref()
        .and_then(|a| a.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("lookup");

    match mode {
        // Zero-overhead default: straight to hybrid search.
        "lookup" => super::search_unified::execute(storage, cognitive, output_config, args).await,
        // Deep reasoning (deep_reference / cross_reference share this handler).
        "reason" => super::cross_reference::execute(storage, cognitive, args).await,
        // Trust-weighted contradiction pairs (storage-only).
        "contradictions" => super::contradictions::execute(storage, args).await,
        other => Err(format!(
            "Unknown recall mode '{other}'. Use lookup|reason|contradictions."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_mode_and_no_required() {
        let s = schema();
        let modes = s["properties"]["mode"]["enum"].as_array().unwrap();
        assert_eq!(modes.len(), 3);
        assert_eq!(s["properties"]["mode"]["default"], "lookup");
        // query must NOT be globally required (contradictions uses topic).
        assert!(
            s.get("required").is_none(),
            "recall must not globally require 'query'"
        );
        // lookup params carried over from search schema.
        assert!(s["properties"]["limit"].is_object());
        assert!(s["properties"]["detail_level"].is_object());
    }

    #[tokio::test]
    async fn test_lookup_is_default_and_resolves() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // No mode → lookup → behaves like search (query required by search).
        let args = Some(serde_json::json!({ "query": "anything" }));
        let r = execute(&storage, &cognitive, &oc, args).await;
        assert!(r.is_ok(), "default lookup should resolve: {r:?}");
    }

    #[tokio::test]
    async fn test_contradictions_mode_resolves_without_query() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // contradictions uses topic, not query — must resolve with no query.
        let args = Some(serde_json::json!({ "mode": "contradictions" }));
        let r = execute(&storage, &cognitive, &oc, args).await;
        assert!(r.is_ok(), "contradictions mode should resolve: {r:?}");
    }
}
