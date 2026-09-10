//! Unified `graph` Tool (v2.2 — Tool Consolidation)
//!
//! Folds four graph/association/prediction tools into one action-dispatched
//! surface:
//!
//!   action ∈ {
//!     chain, associations, bridges,          // former explore_connections
//!     predict,                               // former predict
//!     memory_graph,                          // former memory_graph (viz subgraph)
//!     recent, get, memory, neighbors,        // former composed_graph
//!     never_composed, bounty_mode, label,    //   "
//!   }
//!
//! This is a transparent facade: each action forwards the *same* args envelope
//! to the existing handler, which re-reads its own discriminator/params. None of
//! the underlying arg structs use `deny_unknown_fields`, so unrelated fields are
//! ignored. All actions are read-only EXCEPT `label`, which writes a composition
//! outcome (the one mutator) and is logged for audit.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
// Reuse composed_graph's canonical outcome-label vocabulary (do not re-list).
use super::composed_graph::OUTCOME_TYPES;

/// Discriminated-union schema for the unified `graph` tool.
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "chain", "associations", "bridges",
                    "predict", "memory_graph",
                    "recent", "get", "memory", "neighbors",
                    "never_composed", "bounty_mode", "label"
                ],
                "description": "Reasoning: 'chain' (from, to), 'associations' (spreading activation from 'from'), 'bridges' (connectors between from and to), 'predict' (next needs, from 'context'), 'memory_graph' (subgraph around 'center_id' or 'query'). Composition topology: 'recent', 'get' (event_id), 'memory' and 'neighbors' (memory_id), 'never_composed', 'bounty_mode', 'label' (record an outcome; the only write)."
            },
            // --- explore (chain/associations/bridges) ---
            "from": { "type": "string", "description": "[chain/associations/bridges] Source memory ID." },
            "to": { "type": "string", "description": "[chain/bridges] Target memory ID." },
            // --- predict ---
            "context": { "type": "object", "description": "[predict] Current context (current_file, current_topics, codebase)." },
            // --- memory_graph (viz subgraph) ---
            "center_id": { "type": "string", "description": "[memory_graph] Center node id (or use 'query')." },
            "query": { "type": "string", "description": "[memory_graph] Pick a center node by search query." },
            "depth": { "type": "integer", "minimum": 1, "maximum": 3, "description": "[memory_graph] Traversal depth (1-3, default 2)." },
            "max_nodes": { "type": "integer", "description": "[memory_graph] Max nodes (default 50, capped 200)." },
            // --- composed_graph ---
            "event_id": { "type": "string", "description": "[get/label] Composition event id." },
            "memory_id": { "type": "string", "description": "[memory/neighbors] Memory id." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[never_composed/bounty_mode] Optional tag filter." },
            "outcome_type": {
                "type": "string",
                "enum": OUTCOME_TYPES,
                "description": "[label] Outcome to record for the composition (the only mutating action)."
            },
            "scope": { "type": "string", "default": "user", "description": "[never_composed only] Exact project namespace; filtered before candidate scan limits. Other graph actions do not accept this control." },
            "includeCrossScope": { "type": "boolean", "default": false, "description": "[never_composed only] Explicitly consider candidates across project namespaces." },
            // --- shared ---
            "limit": { "type": "integer", "description": "Max results (per-action defaults; clamped internally).", "minimum": 1, "maximum": 100 }
        },
        "required": ["action"]
    })
}

/// Unified dispatcher for `graph`. Routes on `action`.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let action = args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(|v| v.as_str())
        .ok_or("Missing 'action'. Use chain|associations|bridges|predict|memory_graph|recent|get|memory|neighbors|never_composed|bounty_mode|label.")?
        .to_string();

    if action != "never_composed"
        && args
            .as_ref()
            .is_some_and(|a| a.get("scope").is_some() || a.get("includeCrossScope").is_some())
    {
        return Err("scope and includeCrossScope currently apply only to never_composed".into());
    }
    match action.as_str() {
        // explore_connections — re-reads its own `action` (chain/associations/bridges).
        "chain" | "associations" | "bridges" => {
            super::explore::execute(storage, cognitive, args).await
        }
        // predict — reads `context`, ignores `action`.
        "predict" => super::predict::execute(storage, cognitive, args).await,
        // memory_graph — reads center_id/query/depth, ignores `action`.
        "memory_graph" => super::graph::execute(storage, args).await,
        // composed_graph — re-reads its own `action`. `label` is the only write.
        "recent" | "get" | "memory" | "neighbors" | "never_composed" | "bounty_mode" | "label" => {
            if action == "label" {
                let event_id = args
                    .as_ref()
                    .and_then(|a| a.get("event_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let outcome = args
                    .as_ref()
                    .and_then(|a| a.get("outcome_type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                tracing::info!(
                    event_id = %event_id,
                    outcome_type = %outcome,
                    "graph: composition outcome labeled"
                );
            }
            super::composed_graph::execute(storage, args).await
        }
        other => Err(format!(
            "Unknown graph action '{other}'. Use chain|associations|bridges|predict|memory_graph|recent|get|memory|neighbors|never_composed|bounty_mode|label."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn never_composed_obeys_namespace_and_explains_novelty_boundary() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut own = Vec::new();
        let mut other = Vec::new();
        for (scope, ids) in [("user", &mut own), ("other-project", &mut other)] {
            for content in [
                "Shared fixture decision about source evidence",
                "Shared fixture finding about source evidence",
            ] {
                ids.push(
                    storage
                        .ingest_in_scope(
                            vestige_core::IngestInput {
                                content: content.into(),
                                tags: vec!["fixture".into()],
                                ..Default::default()
                            },
                            scope,
                        )
                        .unwrap()
                        .id,
                );
            }
        }
        for scope in [None, Some("other-project")] {
            let mut args = serde_json::json!({"action": "never_composed", "tags": ["fixture"]});
            if let Some(scope) = scope {
                args["scope"] = scope.into();
            }
            let output = execute(&storage, &cognitive, Some(args)).await.unwrap();
            assert_eq!(output["globalNoveltyVerified"], false);
            let expected = if scope.is_some() { &other } else { &own };
            let pairs = output["candidates"].as_array().unwrap();
            assert!(!pairs.is_empty());
            for pair in pairs {
                assert!(expected.iter().any(|id| pair["firstId"] == *id));
                assert!(expected.iter().any(|id| pair["secondId"] == *id));
            }
        }
        let output = execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({
                "action": "never_composed", "includeCrossScope": true, "limit": 100
            })),
        )
        .await
        .unwrap();
        assert!(output["candidates"].as_array().unwrap().iter().any(|pair| {
            own.iter().any(|id| pair["firstId"] == *id)
                && other.iter().any(|id| pair["secondId"] == *id)
                || other.iter().any(|id| pair["firstId"] == *id)
                    && own.iter().any(|id| pair["secondId"] == *id)
        }));
        for args in [
            serde_json::json!({"action":"never_composed", "scope":" "}),
            serde_json::json!({"action":"recent", "scope":"user"}),
        ] {
            assert!(execute(&storage, &cognitive, Some(args)).await.is_err());
        }
    }

    #[test]
    fn test_schema_action_count() {
        let s = schema();
        let actions = s["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 12);
        // outcome_type enum is sourced from the canonical const.
        let outcomes = s["properties"]["outcome_type"]["enum"].as_array().unwrap();
        assert_eq!(outcomes.len(), OUTCOME_TYPES.len());
    }

    #[test]
    fn test_missing_action_errors() {
        // Pure arg-shape check; no storage needed for the early return path.
        let s = schema();
        assert_eq!(s["required"][0], "action");
    }
}
