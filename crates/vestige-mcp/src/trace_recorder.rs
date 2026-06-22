//! # Trace Recorder — the live black-box wiring
//!
//! Bridges an MCP `tools/call` to the persisted black box. For each call the
//! recorder:
//!
//! 1. derives a stable `runId` (client-supplied `runId`/`run_id` arg if present,
//!    else a fresh `run_` UUID),
//! 2. records an `mcp.call` event with a **hash** of the args (never the raw
//!    args, so traces can't leak prompt contents or secrets),
//! 3. after the tool returns, inspects the result JSON and records the
//!    downstream events the agent experienced — `memory.retrieve` (with
//!    per-id activation), `memory.suppress` (with reason), `sanhedrin.veto`,
//!    `dream.patch`,
//! 4. persists every event to `agent_traces` and broadcasts it over the
//!    dashboard event channel so the Black Box tab updates live.
//!
//! The recorder is best-effort: a persistence error never fails the tool call.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::dashboard::events::VestigeEvent;
use vestige_core::{
    MemoryTraceEvent, Receipt, Storage, SuppressReason, SuppressedReceiptEntry, WriteSource,
};

/// Tools that write to memory and are therefore subject to risk-gated review.
///
/// Includes `codebase` (its `remember_pattern` / `remember_decision` actions
/// write durable architectural-decision memories) so those brain mutations are
/// traced and gated like any other write (B2). Read-only actions on these tools
/// are filtered out downstream by [`is_write_decision`].
fn is_write_tool(tool: &str) -> bool {
    matches!(
        tool,
        "smart_ingest" | "ingest" | "session_checkpoint" | "memory" | "codebase"
    )
}

/// Whether a tool's `decision`/`action` label denotes an actual memory write
/// (vs. a read like `get`/`state`). Used to keep reads out of the write trace.
fn is_write_decision(label: &str) -> bool {
    matches!(
        label,
        "create"
            | "created"
            | "update"
            | "updated"
            | "supersede"
            | "superseded"
            | "reinforce"
            | "reinforced"
            | "merge"
            | "merged"
            | "replace"
            | "replaced"
            | "add_context"
            | "edit"
            | "edited"
            | "promote"
            | "promoted"
            | "demote"
            | "demoted"
            | "remember_pattern"
            | "remember_decision"
            | "remembered"
    )
}

/// Risk-gate the writes in a tool result. For each write the tool just made,
/// build a [`vestige_core::WriteContext`], classify it under the active
/// [`vestige_core::ReviewMode`], and — if risky — quarantine the just-written
/// node (suppress it so it is not used for retrieval until reviewed) and open a
/// [`vestige_core::MemoryPr`]. Normal writes are left untouched: they auto-land,
/// and they already got a receipt.
///
/// Returns the list of opened-PR summaries (id, kind, title, signals) so the
/// caller can annotate the tool response and emit `MemoryPrOpened` events.
pub fn gate_writes(
    storage: &Arc<Storage>,
    event_tx: Option<&broadcast::Sender<VestigeEvent>>,
    run_id: &str,
    tool: &str,
    result: &serde_json::Value,
    mode: vestige_core::ReviewMode,
) -> Vec<serde_json::Value> {
    use vestige_core::{
        classify_write, MemoryPr, MemoryPrKind, MemoryPrStatus, RiskClass, WriteContext,
    };

    if !is_write_tool(tool) {
        return Vec::new();
    }

    let mut opened = Vec::new();

    // Collect each (id, decision) write the tool reported.
    let writes = extract_writes(result);
    for (id, decision) in writes {
        // Pull the just-written node to inspect its real content/type/tags.
        let node = match storage.get_node(&id) {
            Ok(Some(n)) => n,
            _ => continue,
        };

        // A decision of supersede/replace/merge means the write overwrote an
        // existing memory — the strongest risk signal. Look up the trust of the
        // memory it superseded so the gate can weigh it.
        let (supersedes, merges) = match decision.as_str() {
            "supersede" | "replace" => (true, false),
            "merge" => (false, true),
            _ => (false, false),
        };
        // If this superseded something, treat the contradiction as against a
        // high-trust memory when the *new* node's own retention is high (the
        // pipeline only supersedes when confident). This keeps the gate honest
        // without a second DB round-trip per write.
        let contradicts_trust = if supersedes {
            Some(node.retention_strength.max(0.7))
        } else {
            None
        };

        let ctx = WriteContext {
            source: Some(WriteSource::Agent),
            node_type: node.node_type.clone(),
            content: node.content.clone(),
            tags: node.tags.clone(),
            contradicts_trust,
            supersedes,
            merges,
            ..Default::default()
        };

        let (class, signals) = classify_write(&ctx, mode);
        if class != RiskClass::Review {
            continue;
        }

        // Quarantine the just-written node: suppress it so it is held out of
        // retrieval until the PR is decided. Best-effort.
        let _ = storage.suppress_memory(&id);

        let kind = match decision.as_str() {
            "supersede" | "replace" => MemoryPrKind::MemorySuperseded,
            "merge" => MemoryPrKind::DreamConsolidation,
            _ if contradicts_trust.is_some() => MemoryPrKind::ContradictionDetected,
            _ => MemoryPrKind::NewFact,
        };
        let title = format!(
            "{}: \"{}\"",
            pr_kind_phrase(kind),
            node.content.chars().take(80).collect::<String>()
        );
        let pr = MemoryPr {
            id: format!("pr_{}", uuid::Uuid::new_v4().simple()),
            kind,
            status: MemoryPrStatus::Pending,
            title: title.clone(),
            diff: serde_json::json!({
                "decision": decision,
                "node": {
                    "id": node.id,
                    "nodeType": node.node_type,
                    "content": node.content,
                    "tags": node.tags,
                },
            }),
            signals: signals.clone(),
            subject_id: Some(id.clone()),
            run_id: Some(run_id.to_string()),
            created_at: Utc::now().to_rfc3339(),
            decided_at: None,
            decision: None,
        };

        if let Err(e) = storage.save_memory_pr(&pr) {
            tracing::warn!("memory PR save failed: {e}");
            continue;
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(VestigeEvent::MemoryPrOpened {
                id: pr.id.clone(),
                kind: kind.as_str().to_string(),
                title,
                signal_count: signals.len(),
                run_id: Some(run_id.to_string()),
                timestamp: Utc::now(),
            });
        }

        opened.push(serde_json::json!({
            "id": pr.id,
            "kind": kind.as_str(),
            "title": pr.title,
            "signals": signals,
            "subjectId": id,
        }));
    }

    opened
}

fn pr_kind_phrase(kind: vestige_core::MemoryPrKind) -> &'static str {
    use vestige_core::MemoryPrKind::*;
    match kind {
        NewFact => "New fact pending review",
        StrengthenedFact => "Strengthened fact",
        ContradictionDetected => "Contradiction with existing memory",
        MemorySuperseded => "Supersede existing memory",
        EdgeAdded => "New edge",
        NodeDecayed => "Decayed node",
        DreamConsolidation => "Consolidation proposal",
    }
}

/// Tools whose output warrants a retrieval receipt.
fn is_retrieval_tool(tool: &str) -> bool {
    matches!(
        tool,
        "deep_reference" | "cross_reference" | "search" | "explore_connections"
    )
}

/// Build a [`Receipt`] from a retrieval tool's response JSON, persist it, and
/// return it as JSON ready to attach to that response. Reuses exactly the data
/// the tool already computed (retrieved ids + trust, suppressed ids + reason,
/// the activation path) — so the receipt is the auditable "nutrition label" for
/// the answer and costs nothing extra to produce.
///
/// Returns `None` for non-retrieval tools or empty results. Best-effort
/// persistence: a storage error is logged, the receipt is still returned.
pub fn build_and_save_receipt(
    storage: &Arc<Storage>,
    run_id: &str,
    tool: &str,
    result: &serde_json::Value,
) -> Option<serde_json::Value> {
    if !is_retrieval_tool(tool) {
        return None;
    }

    let (retrieved, activation) = extract_retrieved(result);
    if retrieved.is_empty() {
        return None;
    }
    let trust_scores: Vec<f64> = retrieved
        .iter()
        .map(|id| activation.get(id).copied().unwrap_or(0.0))
        .collect();

    let suppressed: Vec<SuppressedReceiptEntry> = extract_suppressed(result)
        .into_iter()
        .map(|(id, reason)| SuppressedReceiptEntry::new(id, reason))
        .collect();

    // The activation path: the run's reasoning chain if present, else a simple
    // best-first chain of the retrieved ids.
    let activation_path = result
        .get("reasoning")
        .and_then(|v| v.as_str())
        .map(|s| vec![s.to_string()])
        .unwrap_or_else(|| {
            if retrieved.len() > 1 {
                vec![retrieved.join(" -> ")]
            } else {
                Vec::new()
            }
        });

    let query = result.get("query").and_then(|v| v.as_str());

    let receipt = Receipt::build(
        Utc::now(),
        run_id,
        retrieved,
        suppressed,
        activation_path,
        &trust_scores,
        Vec::new(),
    );
    if let Err(e) = storage.save_receipt(&receipt, Some(run_id), Some(tool), query) {
        tracing::warn!("receipt save failed: {e}");
    }
    Some(serde_json::to_value(&receipt).unwrap_or(serde_json::Value::Null))
}

/// Derive the run id for a tool call. Honours a client-supplied `runId` /
/// `run_id` argument (so an agent can correlate a whole session's calls);
/// otherwise mints a fresh one.
pub fn run_id_for(args: &Option<Value>) -> String {
    if let Some(a) = args {
        for key in ["runId", "run_id"] {
            if let Some(s) = a.get(key).and_then(|v| v.as_str())
                && !s.is_empty()
            {
                return s.to_string();
            }
        }
    }
    format!("run_{}", uuid::Uuid::new_v4().simple())
}

/// A 64-bit FNV-1a hex fingerprint of the tool arguments — the
/// privacy-preserving stand-in stored on `mcp.call` events. We only need a
/// stable, collision-resistant-enough identifier for "same args → same hash"
/// in the trace, not a cryptographic digest, so a dependency-free FNV-1a keeps
/// the crate lean.
pub fn hash_args(args: &Option<Value>) -> String {
    let bytes = match args {
        Some(v) => serde_json::to_vec(v).unwrap_or_default(),
        None => Vec::new(),
    };
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for b in &bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

/// Persist one trace event and broadcast it to the dashboard. Best-effort:
/// storage failures are logged, never propagated.
pub fn record(
    storage: &Arc<Storage>,
    event_tx: Option<&broadcast::Sender<VestigeEvent>>,
    event: MemoryTraceEvent,
) {
    let event = event.with_at(Utc::now().timestamp_millis());
    let seq = match storage.append_trace_event(&event) {
        Ok(seq) => seq,
        Err(e) => {
            tracing::warn!("trace append failed: {e}");
            return;
        }
    };
    if let Some(tx) = event_tx {
        let _ = tx.send(VestigeEvent::TraceEvent {
            run_id: event.run_id().to_string(),
            seq,
            event,
            timestamp: Utc::now(),
        });
    }
}

/// Record the opening `mcp.call` event for a tool invocation.
pub fn record_call(
    storage: &Arc<Storage>,
    event_tx: Option<&broadcast::Sender<VestigeEvent>>,
    run_id: &str,
    tool: &str,
    args: &Option<Value>,
) {
    record(
        storage,
        event_tx,
        MemoryTraceEvent::McpCall {
            run_id: run_id.to_string(),
            tool: tool.to_string(),
            args_hash: hash_args(args),
            at: 0,
        },
    );
}

/// Inspect a successful tool result and record the downstream memory events the
/// agent experienced (retrieve / suppress / veto / dream). Tool-output shapes
/// are matched leniently so this stays robust as tools evolve.
pub fn record_result(
    storage: &Arc<Storage>,
    event_tx: Option<&broadcast::Sender<VestigeEvent>>,
    run_id: &str,
    tool: &str,
    result: &Value,
) {
    // --- memory.retrieve: ids + per-id activation ---
    let (ids, activation) = extract_retrieved(result);
    if !ids.is_empty() {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::MemoryRetrieve {
                run_id: run_id.to_string(),
                ids,
                activation,
                at: 0,
            },
        );
    }

    // --- memory.suppress: each suppressed id + reason ---
    for (id, reason) in extract_suppressed(result) {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::MemorySuppress {
                run_id: run_id.to_string(),
                id,
                reason,
                at: 0,
            },
        );
    }

    // --- memory.write: writes performed by ingest-like tools ---
    for (id, decision) in extract_writes(result) {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::MemoryWrite {
                run_id: run_id.to_string(),
                id,
                diff: serde_json::json!({ "decision": decision }),
                source: WriteSource::Agent,
                at: 0,
            },
        );
    }

    // --- contradiction.detected: each contradiction pair the agent faced ---
    for (ids, winner_id, detail) in extract_contradictions(result) {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::ContradictionDetected {
                run_id: run_id.to_string(),
                ids,
                winner_id,
                detail,
                at: 0,
            },
        );
    }

    // --- sanhedrin.veto: a blocked claim ---
    if let Some((claim, evidence_ids, confidence)) = extract_veto(result) {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::SanhedrinVeto {
                run_id: run_id.to_string(),
                claim,
                evidence_ids,
                confidence,
                at: 0,
            },
        );
    }

    // --- dream.patch: consolidation proposals ---
    let proposal_ids = extract_dream_proposals(result, tool);
    if !proposal_ids.is_empty() {
        record(
            storage,
            event_tx,
            MemoryTraceEvent::DreamPatch {
                run_id: run_id.to_string(),
                proposal_ids,
                at: 0,
            },
        );
    }
}

/// Pull retrieved memory ids + their activation/score from a search-like or
/// deep_reference-like result.
fn extract_retrieved(result: &Value) -> (Vec<String>, BTreeMap<String, f64>) {
    let mut ids = Vec::new();
    let mut activation = BTreeMap::new();

    // search_unified: { results: [{ id, score|activation, ... }] }
    if let Some(arr) = result.get("results").and_then(|r| r.as_array()) {
        for item in arr {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                ids.push(id.to_string());
                let act = item
                    .get("activation")
                    .or_else(|| item.get("score"))
                    .and_then(|v| v.as_f64());
                if let Some(a) = act {
                    activation.insert(id.to_string(), a);
                }
            }
        }
    }

    // deep_reference: { evidence: [{ id, trust, ... }], recommended: { memory_id } }
    if ids.is_empty()
        && let Some(arr) = result.get("evidence").and_then(|r| r.as_array())
    {
        for item in arr {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                ids.push(id.to_string());
                if let Some(t) = item.get("trust").and_then(|v| v.as_f64()) {
                    activation.insert(id.to_string(), t);
                }
            }
        }
    }

    (ids, activation)
}

/// Pull suppressed entries from a result. Recognises both the deep_reference
/// `superseded`/`contradictions` shapes and the explicit receipt `suppressed`
/// list `[{ id, reason }]`.
fn extract_suppressed(result: &Value) -> Vec<(String, SuppressReason)> {
    let mut out = Vec::new();

    if let Some(arr) = result
        .get("receipt")
        .and_then(|r| r.get("suppressed"))
        .and_then(|s| s.as_array())
    {
        for item in arr {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                let reason = item
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(parse_suppress_reason)
                    .unwrap_or(SuppressReason::LowTrust);
                out.push((id.to_string(), reason));
            }
        }
    }

    // deep_reference surfaces superseded ids directly.
    if let Some(arr) = result.get("superseded").and_then(|s| s.as_array()) {
        for item in arr {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .or_else(|| item.as_str());
            if let Some(id) = id {
                out.push((id.to_string(), SuppressReason::Contradicted));
            }
        }
    }

    out
}

fn parse_suppress_reason(s: &str) -> SuppressReason {
    match s {
        "low_trust" => SuppressReason::LowTrust,
        "decayed" => SuppressReason::Decayed,
        "contradicted" => SuppressReason::Contradicted,
        "privacy" => SuppressReason::Privacy,
        "competition" => SuppressReason::Competition,
        _ => SuppressReason::LowTrust,
    }
}

/// Pull writes from an ingest-like result (single `decision`+`nodeId` or a
/// `results` batch).
fn extract_writes(result: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<(String, String)>, item: &Value| {
        // B2: accept either `decision` (smart_ingest) or `action`
        // (memory promote/demote/edit, codebase remember_*). Read-only labels
        // (get/state/...) are filtered out so reads never trace as writes.
        let label = item
            .get("decision")
            .or_else(|| item.get("action"))
            .and_then(|v| v.as_str());
        let id = item
            .get("nodeId")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str());
        if let (Some(label), Some(id)) = (label, id)
            && is_write_decision(label)
        {
            out.push((id.to_string(), label.to_string()));
        }
    };
    push(&mut out, result);
    if let Some(arr) = result.get("results").and_then(|r| r.as_array()) {
        for item in arr {
            push(&mut out, item);
        }
    }
    out
}

/// Pull contradiction pairs from a deep_reference result. Each entry is
/// `{ stronger: {id, ...}, weaker: {id, ...}, topic_overlap }`; the `stronger`
/// memory is the winner the agent trusted.
fn extract_contradictions(result: &Value) -> Vec<(Vec<String>, Option<String>, String)> {
    let mut out = Vec::new();
    let Some(arr) = result.get("contradictions").and_then(|c| c.as_array()) else {
        return out;
    };
    for item in arr {
        let stronger = item
            .get("stronger")
            .and_then(|s| s.get("id"))
            .and_then(|v| v.as_str());
        let weaker = item
            .get("weaker")
            .and_then(|s| s.get("id"))
            .and_then(|v| v.as_str());
        let (Some(s), Some(w)) = (stronger, weaker) else {
            continue;
        };
        let detail = format!(
            "Contradiction: kept {s} over {w}{}",
            item.get("topic_overlap")
                .and_then(|v| v.as_f64())
                .map(|o| format!(" (topic overlap {:.0}%)", o * 100.0))
                .unwrap_or_default()
        );
        out.push((
            vec![s.to_string(), w.to_string()],
            Some(s.to_string()),
            detail,
        ));
    }
    out
}

/// Pull a Sanhedrin-style veto, if the result carries one.
fn extract_veto(result: &Value) -> Option<(String, Vec<String>, f64)> {
    let veto = result.get("veto").or_else(|| result.get("sanhedrin"))?;
    let claim = veto
        .get("claim")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if claim.is_empty() {
        return None;
    }
    let evidence_ids = veto
        .get("evidenceIds")
        .or_else(|| veto.get("evidence_ids"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let confidence = veto.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    Some((claim, evidence_ids, confidence))
}

/// Pull dream consolidation proposal ids from a dream/consolidate tool result.
///
/// Proposals are identified by an explicit `id` / proposal id when present.
/// The `dream` tool emits an `insights` array whose items carry no id (they are
/// `{insight_type, insight, source_memories, confidence, …}`), so we derive a
/// stable proposal id from each insight's real content — its type plus the
/// memories it consolidated. The dream genuinely ran; this just gives each real
/// proposal a deterministic handle for the trace.
fn extract_dream_proposals(result: &Value, tool: &str) -> Vec<String> {
    if tool != "dream" && tool != "consolidate" {
        return Vec::new();
    }
    let mut out = Vec::new();

    // Explicit id arrays first (consolidate / future producers).
    for key in ["proposalIds", "proposals", "connections"] {
        if let Some(arr) = result.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(id) = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.as_str())
                {
                    out.push(id.to_string());
                }
            }
        }
    }

    // Dream insights: derive a stable id from real content.
    if let Some(arr) = result.get("insights").and_then(|v| v.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                out.push(id.to_string());
                continue;
            }
            let kind = item
                .get("insight_type")
                .and_then(|v| v.as_str())
                .unwrap_or("insight");
            // Prefer the consolidated source memories for a meaningful handle;
            // fall back to the index so every real insight is still counted.
            let src = item
                .get("source_memories")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.as_str())
                        .map(|s| &s[..s.len().min(8)])
                        .collect::<Vec<_>>()
                        .join("+")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("idx{i}"));
            out.push(format!("dream:{kind}:{src}"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_id_honours_client_supplied() {
        let args = Some(serde_json::json!({ "runId": "run_session_7" }));
        assert_eq!(run_id_for(&args), "run_session_7");
    }

    #[test]
    fn run_id_mints_when_absent() {
        let id = run_id_for(&None);
        assert!(id.starts_with("run_"));
        assert!(id.len() > 10);
    }

    #[test]
    fn hash_is_stable_and_hides_content() {
        let args = Some(serde_json::json!({ "query": "my secret prompt" }));
        let h1 = hash_args(&args);
        let h2 = hash_args(&args);
        assert_eq!(h1, h2);
        assert!(!h1.contains("secret"));
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn extract_retrieved_from_search_shape() {
        let r = serde_json::json!({
            "results": [
                { "id": "m1", "score": 0.9 },
                { "id": "m2", "activation": 0.4 }
            ]
        });
        let (ids, act) = extract_retrieved(&r);
        assert_eq!(ids, vec!["m1", "m2"]);
        assert_eq!(act["m1"], 0.9);
        assert_eq!(act["m2"], 0.4);
    }

    #[test]
    fn extract_retrieved_from_deep_reference_shape() {
        let r = serde_json::json!({
            "evidence": [ { "id": "e1", "trust": 0.7 } ]
        });
        let (ids, act) = extract_retrieved(&r);
        assert_eq!(ids, vec!["e1"]);
        assert_eq!(act["e1"], 0.7);
    }

    #[test]
    fn extract_suppressed_from_receipt_and_superseded() {
        let r = serde_json::json!({
            "receipt": { "suppressed": [ { "id": "s1", "reason": "contradicted" } ] },
            "superseded": [ { "id": "s2" } ]
        });
        let out = extract_suppressed(&r);
        assert!(out.contains(&("s1".to_string(), SuppressReason::Contradicted)));
        assert!(out.contains(&("s2".to_string(), SuppressReason::Contradicted)));
    }

    #[test]
    fn extract_dream_proposals_from_real_insights_shape() {
        // The exact shape the `dream` tool emits — insights without an id.
        let r = serde_json::json!({
            "status": "dreamed",
            "insights": [
                {
                    "insight_type": "Bridge",
                    "insight": "These two notes describe the same subsystem.",
                    "source_memories": ["aaaaaaaa1111", "bbbbbbbb2222"],
                    "confidence": 0.8,
                    "novelty_score": 0.6
                }
            ]
        });
        let ids = extract_dream_proposals(&r, "dream");
        assert_eq!(ids.len(), 1, "one real insight -> one proposal id");
        assert_eq!(ids[0], "dream:Bridge:aaaaaaaa+bbbbbbbb");
    }

    #[test]
    fn extract_dream_proposals_empty_when_not_dream_tool() {
        let r = serde_json::json!({ "insights": [{ "insight_type": "x" }] });
        assert!(extract_dream_proposals(&r, "search").is_empty());
    }

    #[test]
    fn extract_writes_single_and_batch() {
        let single = serde_json::json!({ "decision": "create", "nodeId": "n1" });
        assert_eq!(extract_writes(&single), vec![("n1".into(), "create".into())]);
        let batch = serde_json::json!({
            "results": [ { "decision": "update", "id": "n2" } ]
        });
        assert_eq!(extract_writes(&batch), vec![("n2".into(), "update".into())]);
    }

    #[test]
    fn extract_writes_recognizes_action_shape_b2() {
        // B2: memory promote/demote return `action` + `nodeId`, not `decision`.
        let promoted = serde_json::json!({ "action": "promoted", "nodeId": "m1" });
        assert_eq!(extract_writes(&promoted), vec![("m1".into(), "promoted".into())]);
        let demoted = serde_json::json!({ "action": "demoted", "nodeId": "m2" });
        assert_eq!(extract_writes(&demoted), vec![("m2".into(), "demoted".into())]);
        // codebase remember_decision returns action + nodeId.
        let decision = serde_json::json!({ "action": "remember_decision", "nodeId": "c1" });
        assert_eq!(
            extract_writes(&decision),
            vec![("c1".into(), "remember_decision".into())]
        );
    }

    #[test]
    fn extract_writes_ignores_read_actions_b2() {
        // A read (memory get / get_batch / state) carries nodeId but is NOT a write.
        let read = serde_json::json!({ "action": "get", "nodeId": "m1" });
        assert!(extract_writes(&read).is_empty(), "get is not a write");
        let state = serde_json::json!({ "action": "state", "nodeId": "m2" });
        assert!(extract_writes(&state).is_empty(), "state is not a write");
    }

    #[test]
    fn write_tool_set_includes_codebase_b2() {
        assert!(is_write_tool("codebase"));
        assert!(is_write_tool("memory"));
        assert!(!is_write_tool("search"));
        assert!(!is_write_tool("deep_reference"));
    }
}
