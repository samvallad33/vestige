//! Admit run records onto the Strata log.
//!
//! A caller passes records directly or a JUnit document. The same `run_id`
//! with equal fields appends nothing. Node content is not involved.

use serde_json::{Value, json};
use strata_store::{RunKind, RunRecord, RunStatus, parse_junit};
use vestige_core::Storage;

use crate::strata_memory;

/// One run the caller already shaped.
pub struct RunInput {
    pub run_id: String,
    pub kind: String,
    pub subject: String,
    pub commit: String,
    pub status: String,
    pub started_ms: i64,
    pub finished_ms: i64,
}

/// What to admit.
pub struct Request {
    pub runs: Vec<RunInput>,
    pub junit_xml: Option<String>,
    pub commit: String,
}

/// Admit `req`. An empty request records nothing and still succeeds.
pub fn execute(storage: &Storage, req: Request) -> Result<Value, String> {
    let memory = strata_memory::live_memory(storage)
        .ok_or_else(|| "record_runs writes run records on the Strata log".to_string())?;
    let mut records = Vec::new();
    for run in req.runs {
        records.push(RunRecord {
            run_id: run.run_id,
            kind: RunKind::parse(&run.kind)
                .ok_or_else(|| format!("run kind '{}' is not test, ci, or agent", run.kind))?,
            subject: run.subject,
            commit: run.commit,
            status: RunStatus::parse(&run.status).ok_or_else(|| {
                format!(
                    "run status '{}' is not passed, failed, skipped, or errored",
                    run.status
                )
            })?,
            started_ms: run.started_ms,
            finished_ms: run.finished_ms,
        });
    }
    if let Some(xml) = req.junit_xml.as_deref() {
        records.extend(parse_junit(xml, &req.commit)?);
    }
    let admission = memory
        .with_store_mut(|store| store.record_runs(records))
        .map_err(|err| err.to_string())?;
    let receipts: Vec<Value> = admission
        .written
        .iter()
        .map(|id| {
            json!({
                "runId": id,
                "written": true,
                "receipt": admission.effect_seq.map(strata_store::effect_receipt_id),
            })
        })
        .chain(admission.unchanged.iter().map(|(id, seq)| {
            json!({
                "runId": id,
                "written": false,
                "receipt": seq.map(strata_store::effect_receipt_id),
            })
        }))
        .collect();
    Ok(json!({
        "action": "record_runs",
        "recorded": admission.written.len(),
        "unchanged": admission.unchanged.len(),
        "receipts": receipts,
    }))
}

/// Plain-text form of [`execute`].
pub fn render_human(report: &Value) -> String {
    format!(
        "recorded {}  unchanged {}  receipts {}",
        report["recorded"].as_u64().unwrap_or(0),
        report["unchanged"].as_u64().unwrap_or(0),
        report["receipts"]
            .as_array()
            .map(|rows| rows.len())
            .unwrap_or(0)
    )
}
