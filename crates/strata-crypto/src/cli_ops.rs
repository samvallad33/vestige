//! Enterprise compliance exporters (EU AI Act Art. 12, FINRA Rule 4511,
//! Change-Evidence Audit, OpenTelemetry GenAI spans) and CLI superpowers
//! (`vestige why`, `vestige edit-bisect`, `vestige top`) (#403, #422, #541,
//! #567, #571, #573, #574).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ledger::HunkTouchedEdge;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEventRecord {
    pub event_id: String,
    pub timestamp_ms: u64,
    pub actor_id: String,
    pub decision: String,
    pub tool_name: String,
    pub receipt_hash: String,
}

/// EU AI Act Article 12 automatic record-keeping export (#403).
pub fn export_eu_ai_act_article12(system_id: &str, events: &[AuditEventRecord]) -> Value {
    let start_ms = events.iter().map(|e| e.timestamp_ms).min().unwrap_or(0);
    let end_ms = events.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);
    let digest = blake3::hash(serde_json::to_vec(events).unwrap().as_slice())
        .to_hex()
        .to_string();
    json!({
        "standard": "EU-AI-Act-Article-12",
        "system_id": system_id,
        "operating_period": { "start_ms": start_ms, "end_ms": end_ms },
        "event_count": events.len(),
        "human_oversight_stops": events.iter().filter(|e| e.decision == "STOP").count(),
        "events": events,
        "ledger_blake3": digest,
    })
}

/// FINRA Rule 4511 / SEC 17a-4 WORM books-and-records retention export (#422).
pub fn export_finra_4511_worm(firm_id: &str, retention_years: u32, events: &[AuditEventRecord]) -> Value {
    let digest = blake3::hash(serde_json::to_vec(events).unwrap().as_slice())
        .to_hex()
        .to_string();
    json!({
        "standard": "FINRA-4511-SEC-17a-4-WORM",
        "firm_id": firm_id,
        "worm_immutable": true,
        "retention_years": retention_years,
        "serial_index_count": events.len(),
        "records": events,
        "worm_seal_blake3": digest,
    })
}

/// Change-evidence bundle for third-party code/agent change audits (#573).
pub fn export_change_evidence_bundle(
    commit_sha: &str,
    hunks: &[HunkTouchedEdge],
    events: &[AuditEventRecord],
) -> Value {
    json!({
        "bundle_kind": "vestige.change_evidence.v1",
        "commit_sha": commit_sha,
        "touched_hunks": hunks,
        "governing_receipts": events,
    })
}

/// OpenTelemetry GenAI semantic conventions span exporter (#567).
pub fn export_otel_genai_spans(service_name: &str, events: &[AuditEventRecord]) -> Value {
    let spans: Vec<Value> = events
        .iter()
        .map(|ev| {
            json!({
                "name": format!("vestige.gate.{}", ev.tool_name),
                "startTimeUnixNano": ev.timestamp_ms.saturating_mul(1_000_000),
                "attributes": {
                    "gen_ai.system": "vestige.strata",
                    "vestige.actor_id": ev.actor_id,
                    "vestige.decision": ev.decision,
                    "vestige.receipt_hash": ev.receipt_hash,
                }
            })
        })
        .collect();
    json!({
        "resource": { "service.name": service_name },
        "scopeSpans": [{ "scope": { "name": "vestige-mcp" }, "spans": spans }]
    })
}

/// `vestige why <file>:<line>` — walk hunk -> commit -> recorded decision & receipt (#541).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhyExplanation {
    pub file_path: String,
    pub line: u32,
    pub commit_sha: String,
    pub decision_id: String,
    pub reason_summary: String,
    pub receipt_hash: String,
}

pub fn vestige_why_line(
    file_path: &str,
    line: u32,
    hunks: &[HunkTouchedEdge],
    commit_decisions: &[(String, String, String, String)], // (commit_sha, decision_id, summary, receipt)
) -> Vec<WhyExplanation> {
    let norm = file_path.replace('\\', "/");
    let mut out = Vec::new();
    for h in hunks {
        let end_line = h.new_start.saturating_add(h.new_count.max(1) - 1);
        if h.file_path.replace('\\', "/") == norm && line >= h.new_start && line <= end_line {
            for (sha, dec_id, summary, receipt) in commit_decisions {
                if sha == &h.commit_sha {
                    out.push(WhyExplanation {
                        file_path: norm.clone(),
                        line,
                        commit_sha: sha.clone(),
                        decision_id: dec_id.clone(),
                        reason_summary: summary.clone(),
                        receipt_hash: receipt.clone(),
                    });
                }
            }
        }
    }
    out
}

/// `Edit bisect` (#571): deterministic O(log n) binary search over recorded edits
/// identifying the exact first edit that broke the build/test oracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BisectWitnessReceipt {
    pub culprit_index: usize,
    pub culprit_edit_id: String,
    pub probes_executed: usize,
    pub witness_blake3: String,
}

pub fn edit_bisect<F>(edit_ids: &[String], mut build_passes_at: F) -> Option<BisectWitnessReceipt>
where
    F: FnMut(usize) -> bool,
{
    if edit_ids.is_empty() || build_passes_at(edit_ids.len() - 1) {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = edit_ids.len() - 1;
    let mut probes = 1usize; // already checked `len - 1`
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        probes += 1;
        if build_passes_at(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let culprit_id = edit_ids[lo].clone();
    let witness_blake3 = blake3::hash(format!("bisect:{lo}:{culprit_id}:{probes}").as_bytes())
        .to_hex()
        .to_string();
    Some(BisectWitnessReceipt {
        culprit_index: lo,
        culprit_edit_id: culprit_id,
        probes_executed: probes,
        witness_blake3,
    })
}

/// `vestige top` (#574): live terminal summary of stops, failures, fixes, and receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopSnapshot {
    pub head_seq: u64,
    pub stops_count: u64,
    pub failures_count: u64,
    pub fixes_anchored: u64,
    pub recent_receipts: Vec<String>,
}

pub fn render_vestige_top(snap: &TopSnapshot) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "VESTIGE TOP | head_seq={} | stops={} | failures={} | fixes={}\n",
        snap.head_seq, snap.stops_count, snap.failures_count, snap.fixes_anchored
    ));
    out.push_str("-------------------------------------------------------------\n");
    for r in &snap.recent_receipts {
        out.push_str(&format!("  receipt: {r}\n"));
    }
    out
}
