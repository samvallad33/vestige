//! Agent Black Box Resources
//!
//! `trace://` URI scheme — exposes replayable agent-run traces as MCP resources
//! so a coding agent can read its *own* black box back. This closes the trace
//! correlation spine on the MCP side: the same `runId` an agent received in a
//! tool result's `traceUri` resolves here to the full event timeline.
//!
//! - `trace://{runId}` — the full ordered event log for a run.
//! - `trace://{runId}/summary` — just the roll-up counts.
//! - `trace://runs` — recent runs (the run picker).
//! - `trace://latest` — the most recently active run's full trace.

use std::sync::Arc;

use vestige_core::Storage;

/// Read a `trace://` resource.
pub async fn read(storage: &Arc<Storage>, uri: &str) -> Result<String, String> {
    let path = uri.strip_prefix("trace://").unwrap_or("");
    let (path, _query) = match path.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (path, None),
    };

    match path {
        "" | "runs" => read_runs(storage).await,
        "latest" => read_latest(storage).await,
        other => {
            if let Some(run_id) = other.strip_suffix("/summary") {
                read_summary(storage, run_id).await
            } else {
                read_run(storage, other).await
            }
        }
    }
}

async fn read_runs(storage: &Arc<Storage>) -> Result<String, String> {
    let runs = storage.list_agent_runs(50).map_err(|e| e.to_string())?;
    let json: Vec<_> = runs
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "runId": r.run_id,
                "firstTool": r.first_tool,
                "eventCount": r.event_count,
                "retrievedCount": r.retrieved_count,
                "suppressedCount": r.suppressed_count,
                "writeCount": r.write_count,
                "vetoCount": r.veto_count,
                "startedAt": r.started_at,
                "lastAt": r.last_at,
            })
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::json!({ "runs": json }))
        .map_err(|e| e.to_string())
}

async fn read_latest(storage: &Arc<Storage>) -> Result<String, String> {
    let runs = storage.list_agent_runs(1).map_err(|e| e.to_string())?;
    let run = runs
        .into_iter()
        .next()
        .ok_or_else(|| "No agent runs recorded yet".to_string())?;
    read_run(storage, &run.run_id).await
}

async fn read_run(storage: &Arc<Storage>, run_id: &str) -> Result<String, String> {
    let events = storage.get_trace(run_id).map_err(|e| e.to_string())?;
    if events.is_empty() {
        return Err(format!("No trace found for run: {run_id}"));
    }
    let summary = storage.get_agent_run(run_id).ok().flatten();
    let body = serde_json::json!({
        "runId": run_id,
        "summary": summary.map(summary_json),
        "events": events,
    });
    serde_json::to_string_pretty(&body).map_err(|e| e.to_string())
}

async fn read_summary(storage: &Arc<Storage>, run_id: &str) -> Result<String, String> {
    let summary = storage
        .get_agent_run(run_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("No run: {run_id}"))?;
    serde_json::to_string_pretty(&summary_json(summary)).map_err(|e| e.to_string())
}

fn summary_json(s: vestige_core::AgentRunSummary) -> serde_json::Value {
    serde_json::json!({
        "runId": s.run_id,
        "firstTool": s.first_tool,
        "eventCount": s.event_count,
        "retrievedCount": s.retrieved_count,
        "suppressedCount": s.suppressed_count,
        "writeCount": s.write_count,
        "vetoCount": s.veto_count,
        "startedAt": s.started_at,
        "lastAt": s.last_at,
    })
}
