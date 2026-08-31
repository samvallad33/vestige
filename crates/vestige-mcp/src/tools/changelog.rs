//! Memory Changelog Tool
//!
//! View audit trail of memory changes.
//! Per-memory mode: state transitions for a single memory.
//! System-wide mode: consolidations + recent state transitions.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use uuid::Uuid;

use vestige_core::Storage;

/// Input schema for memory_changelog tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Scope to a single memory's audit trail. If omitted, returns system-wide changelog."
            },
            "start": {
                "type": "string",
                "description": "Start of time range (ISO 8601). Only used in system-wide mode."
            },
            "end": {
                "type": "string",
                "description": "End of time range (ISO 8601). Only used in system-wide mode."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries (default: 20, max: 100)",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChangelogArgs {
    #[serde(alias = "memory_id")]
    memory_id: Option<String>,
    start: Option<String>,
    end: Option<String>,
    limit: Option<i32>,
}

/// Execute memory_changelog tool
pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ChangelogArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => ChangelogArgs {
            memory_id: None,
            start: None,
            end: None,
            limit: None,
        },
    };

    let limit = args.limit.unwrap_or(20).clamp(1, 100);

    // Parse the optional ISO-8601 time bounds. These were advertised in the
    // schema since v1.7 but silently ignored until v2.0.7 — any caller that
    // set them was getting unfiltered results with no warning.
    let start_ts = parse_iso_bound(args.start.as_deref(), "start")?;
    let end_ts = parse_iso_bound(args.end.as_deref(), "end")?;

    if let Some(ref memory_id) = args.memory_id {
        // Per-memory mode: state transitions for a specific memory.
        // start/end are only meaningful in system-wide mode; ignored here.
        execute_per_memory(storage, memory_id, limit)
    } else {
        // System-wide mode: consolidations + recent transitions, optionally
        // time-bounded by start/end.
        execute_system_wide(storage, limit, start_ts, end_ts)
    }
}

/// Parse an ISO-8601 timestamp bound, returning a helpful error on bad input
/// instead of silently dropping the filter.
fn parse_iso_bound(raw: Option<&str>, field: &str) -> Result<Option<DateTime<Utc>>, String> {
    match raw {
        Some(s) if !s.is_empty() => DateTime::parse_from_rfc3339(s)
            .map(|dt| Some(dt.with_timezone(&Utc)))
            .map_err(|e| {
                format!(
                    "Invalid {} timestamp {:?}: {}. Use ISO-8601 / RFC-3339 (e.g. 2026-04-19T12:00:00Z).",
                    field, s, e
                )
            }),
        _ => Ok(None),
    }
}

/// Per-memory changelog: state transition audit trail
fn execute_per_memory(storage: &Storage, memory_id: &str, limit: i32) -> Result<Value, String> {
    // Validate UUID format
    Uuid::parse_str(memory_id)
        .map_err(|_| format!("Invalid memory_id '{}'. Must be a valid UUID.", memory_id))?;

    // Get the memory for context
    let node = storage
        .get_node(memory_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Memory '{}' not found.", memory_id))?;

    // Get state transitions
    let transitions = storage
        .get_state_transitions(memory_id, limit)
        .map_err(|e| e.to_string())?;

    let formatted_transitions: Vec<Value> = transitions
        .iter()
        .map(|t| {
            serde_json::json!({
                "fromState": t.from_state,
                "toState": t.to_state,
                "reasonType": t.reason_type,
                "reasonData": t.reason_data,
                "timestamp": t.timestamp.to_rfc3339(),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "tool": "memory_changelog",
        "mode": "per_memory",
        "memoryId": memory_id,
        "memoryContent": node.content,
        "memoryType": node.node_type,
        "currentRetention": node.retention_strength,
        "totalTransitions": formatted_transitions.len(),
        "transitions": formatted_transitions,
    }))
}

/// System-wide changelog: consolidations + recent state transitions.
///
/// `start`/`end` optionally bound the returned events to an inclusive
/// time window. Filtering happens in Rust after the DB reads because
/// the underlying storage helpers don't yet take time parameters —
/// moving the filter into SQL is a v2.1+ optimisation (tracked in the
/// v2.0.7 scope notes). For now we over-fetch (up to 4× `limit`) when
/// a window is supplied so filtering doesn't starve the result set.
fn execute_system_wide(
    storage: &Storage,
    limit: i32,
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
) -> Result<Value, String> {
    // When a window is supplied we can't predict how many events fall inside,
    // so over-fetch to reduce the chance of returning an empty page on a
    // long-tail time range. 4× is arbitrary; revisit if it becomes a cost
    // on very large changelog tables.
    let fetch_limit = if start.is_some() || end.is_some() {
        limit.saturating_mul(4)
    } else {
        limit
    };

    // Get consolidation history
    let consolidations = storage
        .get_consolidation_history(fetch_limit)
        .map_err(|e| e.to_string())?;

    // Get recent state transitions across all memories
    let transitions = storage
        .get_recent_state_transitions(fetch_limit)
        .map_err(|e| e.to_string())?;

    // Get dream history (Bug #9 fix — dreams were invisible to audit trail)
    let dreams = storage.get_dream_history(fetch_limit).unwrap_or_default();

    // Build unified event list
    let mut events: Vec<(DateTime<Utc>, Value)> = Vec::new();

    for c in &consolidations {
        events.push((
            c.completed_at,
            serde_json::json!({
                "type": "consolidation",
                "timestamp": c.completed_at.to_rfc3339(),
                "durationMs": c.duration_ms,
                "memoriesReplayed": c.memories_replayed,
                "connectionFound": c.connections_found,
                "connectionsStrengthened": c.connections_strengthened,
                "connectionsPruned": c.connections_pruned,
                "insightsGenerated": c.insights_generated,
            }),
        ));
    }

    for t in &transitions {
        events.push((
            t.timestamp,
            serde_json::json!({
                "type": "state_transition",
                "timestamp": t.timestamp.to_rfc3339(),
                "memoryId": t.memory_id,
                "fromState": t.from_state,
                "toState": t.to_state,
                "reasonType": t.reason_type,
                "reasonData": t.reason_data,
            }),
        ));
    }

    for d in &dreams {
        events.push((
            d.dreamed_at,
            serde_json::json!({
                "type": "dream",
                "timestamp": d.dreamed_at.to_rfc3339(),
                "durationMs": d.duration_ms,
                "memoriesReplayed": d.memories_replayed,
                "connectionFound": d.connections_found,
                "insightsGenerated": d.insights_generated,
            }),
        ));
    }

    // Apply the optional [start, end] window. `start` is inclusive, `end`
    // is inclusive — matches "show me events between these two wall-clock
    // instants" user expectation.
    if start.is_some() || end.is_some() {
        events.retain(|(ts, _)| {
            let after_start = start.is_none_or(|s| *ts >= s);
            let before_end = end.is_none_or(|e| *ts <= e);
            after_start && before_end
        });
    }

    // Sort by timestamp descending
    events.sort_by_key(|b| std::cmp::Reverse(b.0));

    // Truncate to limit
    events.truncate(limit as usize);

    let formatted_events: Vec<Value> = events.into_iter().map(|(_, v)| v).collect();

    Ok(serde_json::json!({
        "tool": "memory_changelog",
        "mode": "system_wide",
        "totalEvents": formatted_events.len(),
        "events": formatted_events,
        "filter": serde_json::json!({
            "start": start.map(|s| s.to_rfc3339()),
            "end": end.map(|e| e.to_rfc3339()),
        }),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_test_memory(storage: &Arc<Storage>) -> String {
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Changelog test memory".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        node.id
    }

    #[test]
    fn test_schema_has_properties() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["memory_id"].is_object());
        assert!(s["properties"]["start"].is_object());
        assert!(s["properties"]["end"].is_object());
        assert!(s["properties"]["limit"].is_object());
        assert_eq!(s["properties"]["limit"]["default"], 20);
        assert_eq!(s["properties"]["limit"]["minimum"], 1);
        assert_eq!(s["properties"]["limit"]["maximum"], 100);
    }

    #[tokio::test]
    async fn test_changelog_no_args_system_wide() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["tool"], "memory_changelog");
        assert_eq!(value["mode"], "system_wide");
        assert!(value["events"].is_array());
    }

    #[tokio::test]
    async fn test_changelog_system_wide_empty() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        let value = result.unwrap();
        assert_eq!(value["totalEvents"], 0);
        assert!(value["events"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_changelog_per_memory_valid_id() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "memory_id": id });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["tool"], "memory_changelog");
        assert_eq!(value["mode"], "per_memory");
        assert_eq!(value["memoryId"], id);
        assert!(value["memoryContent"].is_string());
        assert!(value["transitions"].is_array());
    }

    #[tokio::test]
    async fn test_changelog_per_memory_invalid_uuid() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "memory_id": "not-a-uuid" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory_id"));
    }

    #[tokio::test]
    async fn test_changelog_per_memory_nonexistent() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "memory_id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_changelog_limit_clamped() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "limit": 0 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok()); // clamped to 1
    }

    #[tokio::test]
    async fn test_changelog_limit_high_clamped() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "limit": 999 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok()); // clamped to 100
    }

    #[tokio::test]
    async fn test_changelog_per_memory_no_transitions() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_test_memory(&storage).await;
        let args = serde_json::json!({ "memory_id": id });
        let result = execute(&storage, Some(args)).await;
        let value = result.unwrap();
        assert_eq!(value["totalTransitions"], 0);
        assert!(value["transitions"].as_array().unwrap().is_empty());
    }

    /// v2.0.7 hygiene: malformed `start` must return a helpful error instead
    /// of silently dropping the filter (the pre-v2.0.7 behavior was to
    /// `#[allow(dead_code)]` the field entirely). Guards against a regression
    /// where someone unwraps the parse and triggers a panic on bad input.
    #[tokio::test]
    async fn test_changelog_malformed_start_returns_error() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "start": "not-a-date" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err(), "malformed start should error");
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid start"),
            "error should name the offending field; got: {}",
            err
        );
        assert!(
            err.contains("ISO-8601") || err.contains("RFC-3339"),
            "error should hint at the expected format; got: {}",
            err
        );
    }

    /// The response must echo the applied `start` bound so callers can confirm
    /// the window was honored. Empty store so filter narrows to 0 events.
    #[tokio::test]
    async fn test_changelog_filter_field_echoes_start() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "start": "2026-04-19T00:00:00Z" });
        let result = execute(&storage, Some(args)).await;
        let value = result.unwrap();
        assert_eq!(value["filter"]["start"], "2026-04-19T00:00:00+00:00");
        assert!(value["filter"]["end"].is_null());
    }
}
