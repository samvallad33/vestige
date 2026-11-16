//! Public adapter for `intention action=graph`. All evidence is local or supplied
//! by the caller. This module does not fetch URLs, execute tools, or send alerts.

use crate::cognitive::CognitiveEngine;
use chrono::{DateTime, Duration, Utc};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::{Storage, intention_graph::Command};

/// How far ahead of the wall clock a caller-supplied evaluation clock may be.
/// Committed graph state never accepts an earlier clock afterwards, so an
/// unbounded value would freeze the scope against every later command.
const MAX_CLOCK_AHEAD_HOURS: i64 = 24;

/// Extend the compatible intention interface without adding another MCP tool.
pub fn schema() -> Value {
    let mut schema = super::intention_unified::schema();
    schema["properties"]["action"]["enum"]
        .as_array_mut()
        .expect("action enum")
        .push(json!("graph"));
    schema["properties"]["scope"] = json!({"type":"string","default":"user","maxLength":128,
        "description":"[set] Project namespace for the intention; recall resurfacing never crosses scopes. [list/check] Restrict to one namespace; omit for all namespaces. [graph] Local intention namespace, not an authorization boundary."});
    schema["properties"]["at"] = json!({"type":"string","format":"date-time",
        "description":"[graph] Explicit evaluation clock for reproducible fixtures; defaults to now. Must not be more than 24 hours ahead of the current time."});
    schema["properties"]["command"] = vestige_core::intention_graph::schema();
    schema["properties"]["command"]["description"] = json!(
        "[graph] Evidence-aware plan/revise/observe/evaluate/explain/portfolio/complete/cancel/acknowledge. replay checks committed history. memory_snapshot reads a scoped content digest; refresh_memory observes it using memory_id,event_id,source_revision. No external actions or fact verification."
    );
    // Adapter commands have disjoint action tags and never accept caller values
    // for the reserved local-memory source namespace.
    if let Some(variants) = schema["properties"]["command"]["oneOf"].as_array_mut() {
        // hardcode-scan: protocol (JSON-RPC / JSON-schema keys)
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

fn execute_graph(storage: &Arc<Storage>, args: &Value) -> Result<Value, String> {
    if args.to_string().len() > 128 * 1024 {
        return Err("graph arguments exceed 128 KiB".into());
    }
    let scope = match args.get("scope") {
        None => "user",
        Some(v) => v.as_str().ok_or("scope must be a string")?,
    };
    let now = match args.get("at") {
        None => Utc::now(),
        Some(v) => {
            let at =
                DateTime::parse_from_rfc3339(v.as_str().ok_or("at must be an RFC3339 string")?)
                    .map_err(|_| "at must be an RFC3339 timestamp")?
                    .with_timezone(&Utc);
            if at > Utc::now() + Duration::hours(MAX_CLOCK_AHEAD_HOURS) {
                return Err(format!(
                    "at must not be more than {MAX_CLOCK_AHEAD_HOURS} hours ahead of the current time"
                ));
            }
            at
        }
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
mod clock_bound_tests {
    use super::*;

    fn plan(id: &str) -> Value {
        json!({"id": id, "description": "Synthetic plan", "requirements": [], "conflict_keys": []})
    }

    #[test]
    fn graph_rejects_an_evaluation_clock_far_ahead_and_stays_usable() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).expect("strata log");
        let mut far = plan("far");
        far["action"] = json!("plan");
        let rejected = execute_graph(
            &storage,
            &json!({"action":"graph","at":"2999-01-01T00:00:00Z","command":far}),
        );
        assert!(rejected.is_err(), "far-future clock must be refused");

        let mut near = plan("near");
        near["action"] = json!("plan");
        let ok = execute_graph(&storage, &json!({"action":"graph","command":near}));
        assert!(ok.is_ok(), "later calls must keep working: {ok:?}");
        let replay = execute_graph(
            &storage,
            &json!({"action":"graph","command":{"action":"replay"}}),
        )
        .unwrap();
        assert_eq!(replay["matched"], true);

        drop(storage);
        let reopened = crate::strata_memory::open(dir.path()).expect("reopen");
        let mut after = plan("after");
        after["action"] = json!("plan");
        let ok = execute_graph(&reopened, &json!({"action":"graph","command":after}));
        assert!(ok.is_ok(), "state must stay usable after restart: {ok:?}");
    }
}
