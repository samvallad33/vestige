//! Public adapter for `intention action=graph`. All evidence is local or supplied
//! by the caller. This module does not fetch URLs, execute tools, or send alerts.

use crate::cognitive::CognitiveEngine;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::{Storage, intention_graph::Command};

/// Extend the compatible intention interface without adding another MCP tool.
pub fn schema() -> Value {
    let mut schema = super::intention_unified::schema();
    schema["properties"]["action"]["enum"]
        .as_array_mut()
        .expect("action enum")
        .push(json!("graph"));
    schema["properties"]["scope"] = json!({"type":"string","default":"user","maxLength":128,
        "description":"[graph] Local intention namespace, not an authorization boundary."});
    schema["properties"]["at"] = json!({"type":"string","format":"date-time",
        "description":"[graph] Explicit evaluation clock for reproducible fixtures; defaults to now."});
    schema["properties"]["command"] = vestige_core::intention_graph::schema();
    schema["properties"]["command"]["description"] = json!(
        "[graph] Evidence-aware plan/revise/observe/evaluate/explain/portfolio/complete/cancel/acknowledge. replay checks committed history. memory_snapshot reads a scoped content digest; refresh_memory observes it using memory_id,event_id,source_revision. No external actions or fact verification."
    );
    // Adapter commands have disjoint action tags and never accept caller values
    // for the reserved local-memory source namespace.
    if let Some(variants) = schema["properties"]["command"]["oneOf"].as_array_mut() {
        variants.extend([
            json!({"type":"object","properties":{"action":{"const":"replay"}},"required":["action"],"additionalProperties":false}),
            json!({"type":"object","properties":{"action":{"const":"memory_snapshot"},"memory_id":{"type":"string"}},"required":["action","memory_id"],"additionalProperties":false}),
            json!({"type":"object","properties":{"action":{"const":"refresh_memory"},"memory_id":{"type":"string"},"event_id":{"type":"string"},"source_revision":{"type":"integer","minimum":1}},"required":["action","memory_id","event_id","source_revision"],"additionalProperties":false}),
        ]);
    }
    schema
}

/// Route new graph commands while retaining all existing intention actions.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    if args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(Value::as_str)
        != Some("graph")
    {
        return super::intention_unified::execute(storage, cognitive, args).await;
    }
    execute_graph(storage, args.as_ref().ok_or("Missing arguments")?)
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Missing string field '{key}'"))
}

fn execute_graph(storage: &Storage, args: &Value) -> Result<Value, String> {
    if args.to_string().len() > 128 * 1024 {
        return Err("graph arguments exceed 128 KiB".into());
    }
    let scope = match args.get("scope") {
        None => "user",
        Some(v) => v.as_str().ok_or("scope must be a string")?,
    };
    let now = match args.get("at") {
        None => Utc::now(),
        Some(v) => DateTime::parse_from_rfc3339(v.as_str().ok_or("at must be an RFC3339 string")?)
            .map_err(|_| "at must be an RFC3339 timestamp")?
            .with_timezone(&Utc),
    };
    let raw = args.get("command").ok_or("Missing graph command")?;
    let action = field(raw, "action")?;
    let adapter_fields: Option<&[&str]> = match action {
        "replay" => Some(&["action"]),
        "memory_snapshot" => Some(&["action", "memory_id"]),
        "refresh_memory" => Some(&["action", "memory_id", "event_id", "source_revision"]),
        _ => None,
    };
    if let Some(allowed) = adapter_fields
        && raw
            .as_object()
            .is_some_and(|object| object.keys().any(|key| !allowed.contains(&key.as_str())))
    {
        return Err("Unexpected field in intention graph adapter command".into());
    }
    match action {
        "replay" => storage.replay_intention_graph(scope),
        "memory_snapshot" => {
            storage.intention_memory_snapshot(scope, field(raw, "memory_id")?, now)
        }
        "refresh_memory" => {
            let snapshot =
                storage.intention_memory_snapshot(scope, field(raw, "memory_id")?, now)?;
            let revision = raw
                .get("source_revision")
                .and_then(Value::as_u64)
                .filter(|r| *r > 0)
                .ok_or("source_revision must be a positive integer")?;
            let command: Command = serde_json::from_value(json!({"action":"observe",
                "event_id":field(raw,"event_id")?,"source_key":snapshot["source_key"],
                "source_revision":revision,"value":snapshot["value"],
                "observed_events":[],"covered_events":[]}))
            .map_err(|e| format!("Invalid memory observation: {e}"))?;
            storage.apply_intention_graph(scope, command, now)
        }
        _ => {
            if action == "observe" && field(raw, "source_key")?.starts_with("memory:") {
                return Err("memory: sources are reserved; use refresh_memory to read the local scoped snapshot".into());
            }
            let command: Command = serde_json::from_value(raw.clone())
                .map_err(|e| format!("Invalid graph command: {e}"))?;
            storage.apply_intention_graph(scope, command, now)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_schema_advertises_engine_and_adapter_commands() {
        let schema = schema();
        let variants = schema["properties"]["command"]["oneOf"].as_array().unwrap();
        let actions: std::collections::BTreeSet<_> = variants
            .iter()
            .filter_map(|variant| variant["properties"]["action"]["const"].as_str())
            .collect();
        for action in [
            "plan",
            "revise",
            "observe",
            "evaluate",
            "explain",
            "portfolio",
            "complete",
            "cancel",
            "acknowledge",
            "replay",
            "memory_snapshot",
            "refresh_memory",
        ] {
            assert!(actions.contains(action), "missing graph command {action}");
        }
        assert_eq!(actions.len(), 12);
    }

    #[test]
    fn graph_memory_bridge_derives_values_and_replays_recorded_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let node = storage
            .ingest_in_scope(
                vestige_core::IngestInput {
                    content: "Synthetic memory premise".into(),
                    ..Default::default()
                },
                "fixture",
            )
            .unwrap();
        let at = "2026-10-01T09:00:00Z";
        let snapshot = execute_graph(
            &storage,
            &json!({"action":"graph","scope":"fixture","at":at,
            "command":{"action":"memory_snapshot","memory_id":node.id}}),
        )
        .unwrap();
        execute_graph(&storage, &json!({"action":"graph","scope":"fixture","at":at,
            "command":{"action":"plan","id":"p","description":"Retain the premise","requirements":[{
                "type":"evidence","id":"premise","source_key":snapshot["source_key"],"memory_id":node.id,
                "condition":{"op":"equals","value":snapshot["value"]}}]}})).unwrap();
        let observed = execute_graph(&storage, &json!({"action":"graph","scope":"fixture","at":at,
            "command":{"action":"refresh_memory","memory_id":node.id,"event_id":"local-1","source_revision":1}})).unwrap();
        assert!(observed.to_string().contains("satisfied"));
        assert!(!observed.to_string().contains("Synthetic memory premise"));
        assert!(execute_graph(&storage, &json!({"action":"graph","scope":"fixture","at":at,
            "command":{"action":"refresh_memory","memory_id":node.id,"event_id":"forged","source_revision":2,"value":"forged"}})).is_err());
        let replay = execute_graph(
            &storage,
            &json!({"action":"graph","scope":"fixture",
            "command":{"action":"replay"}}),
        )
        .unwrap();
        assert_eq!(replay["matched"], true);
        assert_eq!(replay["commands"], 2);
    }

    #[test]
    fn graph_rejects_invalid_clock_and_reserved_memory_assertions() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        assert!(
            execute_graph(
                &storage,
                &json!({"action":"graph","at":"tomorrow","command":{"action":"portfolio"}})
            )
            .is_err()
        );
        assert!(execute_graph(&storage,&json!({"action":"graph","command":{"action":"observe","source_key":"memory:forged"}})).is_err());
        assert!(
            execute_graph(
                &storage,
                &json!({"action":"graph","scope":12,"command":{"action":"portfolio"}})
            )
            .is_err()
        );
    }

    #[test]
    fn graph_plan_round_trips_through_public_adapter_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let storage = Storage::new(Some(path.clone())).unwrap();
        let result=execute_graph(&storage,&json!({"action":"graph","at":"2026-10-01T09:00:00Z",
            "command":{"action":"plan","id":"projector","description":"Buy the selected projector", "requirements":[],"conflict_keys":[]}})).unwrap();
        assert!(result["journal_seq"].as_i64().is_some());
        drop(storage);
        let reopened = Storage::new(Some(path)).unwrap();
        let result = execute_graph(
            &reopened,
            &json!({"action":"graph","command":{"action":"replay"}}),
        )
        .unwrap();
        assert_eq!(result["matched"], true);
        let result = execute_graph(
            &reopened,
            &json!({"action":"graph","at":"2026-10-01T09:00:00Z","command":{"action":"explain","id":"projector"}}),
        )
        .unwrap();
        assert!(result.to_string().contains("Buy the selected projector"));
    }
}
