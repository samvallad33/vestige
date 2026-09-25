//! GhostLink — the renamed, collapsed graph surface (v3.1).
//!
//! The ghost is the never-composed: a pairing that already exists in the memory
//! graph but has never been summoned. GhostLink proposes the summons.
//!
//! This module is a facade over [`super::graph_unified`]: every mode maps 1:1 to
//! an existing graph action, so write gating, receipts, trace events, and the
//! composed-graph storage contract are inherited unchanged. `weave` (the `label`
//! action) is the only write.

use super::composed_graph::OUTCOME_TYPES;
use crate::cognitive::CognitiveEngine;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::Storage;

/// GhostLink tool schema. Modes are verbs; one selection decision replaces the
/// old 12-action graph enum (RAG-MCP 2505.03275: selection accuracy collapses
/// as the visible action surface grows).
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["propose", "bounty", "weave", "map", "inspect", "explore", "predict"],
                "description": "propose: surface never-composed memory pairings (the ghosts: combinations that exist in your graph but have never fired together). bounty: gamified ghost-hunting lanes. weave: record a composition outcome for an event (the only write). map: force-directed subgraph for visualization. inspect: recent/get/memory/neighbors over composition events. explore: chain/associations/bridges between two memories. predict: context-ahead memory predictions."
            },
            "view": {
                "type": "string",
                "enum": ["recent", "get", "memory", "neighbors"],
                "description": "[inspect] Which composition-event view to read."
            },
            "kind": {
                "type": "string",
                "enum": ["chain", "associations", "bridges"],
                "description": "[explore] Which reasoning path to run."
            },
            "from": { "type": "string", "description": "[explore] Source memory ID." },
            "to": { "type": "string", "description": "[explore:chain/bridges] Target memory ID." },
            "context": { "type": "object", "description": "[predict] Context: current_file, current_topics, codebase." },
            "center_id": { "type": "string", "description": "[map] Center node id (or use query)." },
            "query": { "type": "string", "description": "[map] Pick the center node by search query." },
            "depth": { "type": "integer", "minimum": 1, "maximum": 3, "description": "[map] Traversal depth (1-3, default 2)." },
            "max_nodes": { "type": "integer", "description": "[map] Max nodes (default 50, capped 200)." },
            "event_id": { "type": "string", "description": "[weave/inspect:get] Composition event id." },
            "memory_id": { "type": "string", "description": "[inspect:memory/neighbors] Memory id." },
            "tags": { "type": "array", "items": { "type": "string" }, "description": "[propose/bounty] Optional tag filter." },
            "outcome_type": {
                "type": "string",
                "enum": ["helpful", "dead_end", "submitted", "accepted", "rejected", "duplicate_risk", "needs_poc", "bad_severity", "user_promoted", "user_demoted", "closed_by_scope", "closed_by_duplicate", "closed_by_false_assumption", "closed_by_user", "expired_lane"],
                "description": "[weave] Outcome to record for the event."
            },
            "scope": { "type": "string", "default": "user", "description": "[propose] Exact project namespace; filtered before candidate scan limits." },
            "includeCrossScope": { "type": "boolean", "default": false, "description": "[propose] Explicitly consider candidates across project namespaces." },
            "limit": { "type": "integer", "description": "Max results (per-mode defaults, clamped).", "minimum": 1, "maximum": 100 }
        },
        "required": ["mode"]
    })
}

/// Map a GhostLink mode (+ view/kind) to the underlying graph action, then
/// delegate to the unified dispatcher so gating, receipts, and traces carry.
pub async fn execute(
    storage: &std::sync::Arc<Storage>,
    cognitive: &std::sync::Arc<tokio::sync::Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let mut args = args.unwrap_or_else(|| serde_json::json!({}));
    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'mode'. Use propose|bounty|weave|map|inspect|explore|predict.")?
        .to_string();

    let action: String = match mode.as_str() {
        "propose" => "never_composed".to_string(),
        "bounty" => "bounty_mode".to_string(),
        "weave" => "label".to_string(),
        "map" => "memory_graph".to_string(),
        "predict" => "predict".to_string(),
        "inspect" | "explore" => {
            // inspect/explore carry a second discriminator (view/kind) that IS
            // the underlying action name; validate it against the known set.
            let key = if mode == "inspect" { "view" } else { "kind" };
            let sub = args
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Missing '{key}' for mode '{mode}'."))?
                .to_string();
            let allowed: &[&str] = if mode == "inspect" {
                &["recent", "get", "memory", "neighbors"]
            } else {
                &["chain", "associations", "bridges"]
            };
            if !allowed.contains(&sub.as_str()) {
                let allowed_str = allowed.join(", ");
                return Err(format!(
                    "Invalid {key} '{sub}' for mode '{mode}'. Allowed: {allowed_str}."
                ));
            }
            sub
        }
        other => return Err(format!("Unknown mode '{other}'.")),
    };

    // Inject the underlying action; strip GhostLink-only discriminators the
    // graph dispatcher does not know.
    if let Some(obj) = args.as_object_mut() {
        obj.remove("mode");
        obj.remove("view");
        obj.remove("kind");
        obj.insert("action".into(), Value::String(action));
    }


    super::graph_unified::execute(storage, cognitive, Some(args)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_four_core_modes_and_write_flagged() {
        let s = schema();
        let modes: Vec<&str> = s["properties"]["mode"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            modes,
            ["propose", "bounty", "weave", "map", "inspect", "explore", "predict"]
        );
        let mode_desc = s["properties"]["mode"]["description"].as_str().unwrap_or("");
        assert!(
            mode_desc.contains("the only write"),
            "mode description must flag weave as the only write"
        );
    }

    #[test]
    fn mode_mapping_covers_every_graph_action() {
        // Every legacy graph action must remain reachable through GhostLink.
        let legacy: &[&str] = &[
            "chain",
            "associations",
            "bridges",
            "predict",
            "memory_graph",
            "recent",
            "get",
            "memory",
            "neighbors",
            "never_composed",
            "bounty_mode",
            "label",
        ];
        let via: &[(&str, Option<&str>)] = &[
            ("propose", None),
            ("bounty", None),
            ("weave", None),
            ("map", None),
            ("predict", None),
            ("inspect", Some("view")),
            ("explore", Some("kind")),
        ];
        for action in legacy {
            let reachable = via.iter().any(|(mode, key)| match key {
                None => match *mode {
                    "propose" => *action == "never_composed",
                    "bounty" => *action == "bounty_mode",
                    "weave" => *action == "label",
                    "map" => *action == "memory_graph",
                    "predict" => *action == "predict",
                    _ => false,
                },
                Some(key) => match *action {
                    "recent" | "get" | "memory" | "neighbors" => *key == "view",
                    "chain" | "associations" | "bridges" => *key == "kind",
                    _ => false,
                },
            });
            assert!(
                reachable,
                "legacy action {action} unreachable via GhostLink"
            );
        }
    }
}
