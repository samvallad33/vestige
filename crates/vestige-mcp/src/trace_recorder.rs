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

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::sync::{Arc, OnceLock};

use chrono::Utc;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::dashboard::events::VestigeEvent;
use vestige_core::storage::{
    SignedReceiptWrite, load_receipt_signing_seed,
    receipt_attestation::{
        AttestationChainPosition, DisclosureMapping, ProducerIdentity, ReceiptAttestationV1,
        RedactionSafeDecisionProjectionV1, SignedReceiptAttestation, SigningKeyStatus,
        sign_attestation,
    },
};
use vestige_core::{
    BACKFILL_RECEIPT_CLAIM_BOUNDARY, BACKFILL_RECEIPT_SCHEMA_V1, BackfillCandidateEvidence,
    MemoryTraceEvent, REPLAY_SELECTION_BOUNDARY, Receipt, ReceiptEvidence, ReplayDecayRisk,
    RetrievalReplayCapsuleDraft, RetrievalReplayItemDraft, Storage, SuppressReason,
    SuppressedReceiptEntry, WriteSource, private_evidence_digest, replay_evidence_slot,
    replay_policy_digest,
};

/// Opt-in environment configuration for the live receipt signer. The process
/// never provisions a seed or registers a public key: operators must do those
/// actions first, then explicitly configure both values below.
const RECEIPT_SIGNING_KEY_ID_ENV: &str = "VESTIGE_RECEIPT_SIGNING_KEY_ID";
const RECEIPT_SIGNING_SEED_PATH_ENV: &str = "VESTIGE_RECEIPT_SIGNING_SEED_PATH";
const RECEIPT_ATTESTATION_ALGORITHM_V1: &str = "mcp-retrieval-receipt-v1";

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
            // C2: destructive removals are brain mutations too — they must
            // trace as memory.write and be gateable, not bypass review.
            | "purge"
            | "purged"
            | "delete"
            | "deleted"
            | "forget"
            | "forgotten"
    )
}

/// Read the persisted [`vestige_core::ReviewMode`] for this brain.
///
/// The mode lives in `<data_dir>/review_mode.json` and is written by the
/// dashboard (`POST /api/memory-prs/mode`). Anything missing, unreadable, or
/// unrecognised falls back to the default [`vestige_core::ReviewMode::RiskGated`],
/// so a corrupt file can never silently disable gating.
///
/// This is the single source of truth: the dashboard handler delegates here so
/// the MCP write path and the dashboard can never disagree about the mode.
pub fn read_review_mode(storage: &Storage) -> vestige_core::ReviewMode {
    let path = storage.data_dir().join("review_mode.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        // No file is the normal first-run state: default, quietly.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return vestige_core::ReviewMode::default();
        }
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "review_mode.json is unreadable; using the default risk_gated review mode"
            );
            return vestige_core::ReviewMode::default();
        }
    };
    let label = serde_json::from_str::<Value>(&raw)
        .ok()
        .and_then(|v| v.get("mode").and_then(|m| m.as_str()).map(str::to_owned));
    // Never fall back silently: an operator who wrote `fast` with a typo must
    // learn they are still risk-gated from the log, not from a surprise later.
    match label.as_deref().map(vestige_core::ReviewMode::try_from_label) {
        Some(Some(mode)) => mode,
        Some(None) => {
            tracing::warn!(
                path = %path.display(),
                label = label.as_deref().unwrap_or_default(),
                "review_mode.json names an unknown review mode (known: fast, risk_gated, paranoid); using the default risk_gated review mode"
            );
            vestige_core::ReviewMode::default()
        }
        None => {
            tracing::warn!(
                path = %path.display(),
                "review_mode.json has no string \"mode\" field; using the default risk_gated review mode"
            );
            vestige_core::ReviewMode::default()
        }
    }
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
        MemoryPr, MemoryPrKind, MemoryPrStatus, RiskClass, WriteContext, classify_write,
    };

    if !is_write_tool(tool) {
        return Vec::new();
    }

    let mut opened = Vec::new();

    // Collect each (id, decision) write the tool reported.
    let writes = extract_writes(result);
    for (id, decision) in writes {
        let destructive = is_destructive_decision(&decision);

        // Pull the just-written node to inspect its real content/type/tags.
        // C2: a destructive write (purge/delete/forget) has ALREADY removed the
        // row, so get_node returns None — we must NOT skip it (that's how
        // destructive removals were bypassing review). For those, build the
        // context from the decision alone; for normal writes a missing node
        // genuinely means nothing to gate, so skip.
        let node = match storage.get_node(&id) {
            Ok(Some(n)) => Some(n),
            _ if destructive => None,
            _ => continue,
        };

        let (supersedes, merges) = match decision.as_str() {
            "supersede" | "replace" | "superseded" => (true, false),
            "merge" | "merged" => (false, true),
            _ => (false, false),
        };
        // If this superseded something, treat the contradiction as against a
        // high-trust memory when the *new* node's own retention is high (the
        // pipeline only supersedes when confident). This keeps the gate honest
        // without a second DB round-trip per write.
        let contradicts_trust = match (&node, supersedes) {
            (Some(n), true) => Some(n.retention_strength.max(0.7)),
            _ => None,
        };

        let ctx = WriteContext {
            source: Some(WriteSource::Agent),
            node_type: node
                .as_ref()
                .map(|n| n.node_type.clone())
                .unwrap_or_default(),
            content: node.as_ref().map(|n| n.content.clone()).unwrap_or_default(),
            tags: node.as_ref().map(|n| n.tags.clone()).unwrap_or_default(),
            contradicts_trust,
            supersedes,
            merges,
            forgets: destructive,
            ..Default::default()
        };

        // A destructive write ALWAYS warrants review (erasing brain state) even
        // in Fast mode is debatable, but we respect the mode: the `forgets`
        // signal in WriteContext makes classify_write gate it in Risk-Gated.
        let (class, signals) = classify_write(&ctx, mode);
        if class != RiskClass::Review {
            continue;
        }

        // Quarantine the just-written node so it's held out of retrieval until
        // the PR is decided. For a destructive write there's no live node to
        // suppress — the PR records the action for review/audit instead.
        // `held` is reported truthfully to the caller: a destructive write is
        // NOT held (it already happened), and a failed suppression must not be
        // announced as a quarantine.
        let held = if node.is_some() {
            match storage.suppress_memory(&id) {
                Ok(_) => true,
                Err(e) => {
                    tracing::warn!("memory PR gate: quarantine of {id} failed: {e}");
                    false
                }
            }
        } else {
            false
        };

        let kind = match decision.as_str() {
            "supersede" | "replace" | "superseded" => MemoryPrKind::MemorySuperseded,
            "merge" | "merged" => MemoryPrKind::DreamConsolidation,
            _ if destructive => MemoryPrKind::NodeDecayed,
            _ if contradicts_trust.is_some() => MemoryPrKind::ContradictionDetected,
            _ => MemoryPrKind::NewFact,
        };

        // PRIV: never copy full memory content into the PR (it can hold a
        // secret, and the PR row is read by the dashboard and may be exported).
        // Store a short, redacted preview + a content hash instead. The preview
        // is dropped entirely when the write was gated for a sensitive topic.
        let raw_content = node.as_ref().map(|n| n.content.as_str()).unwrap_or("");
        let sensitive = is_sensitive_pr_content(raw_content, &signals);
        let preview = content_preview(raw_content, sensitive);
        let content_hash = hash_content(raw_content);

        let title = format!("{}: {}", pr_kind_phrase(kind), preview);
        let pr = MemoryPr {
            id: format!("pr_{}", uuid::Uuid::new_v4().simple()),
            kind,
            status: MemoryPrStatus::Pending,
            title: title.clone(),
            diff: serde_json::json!({
                "decision": decision,
                "node": {
                    "id": id,
                    "nodeType": node.as_ref().map(|n| n.node_type.clone()).unwrap_or_default(),
                    // Redacted: preview (or "[redacted — sensitive]") + hash,
                    // never the full content.
                    "contentPreview": preview,
                    "contentHash": content_hash,
                    "tags": node.as_ref().map(|n| n.tags.clone()).unwrap_or_default(),
                    "deleted": node.is_none(),
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
            // true: node suppressed until the PR is decided. false: nothing is
            // held — either the write was destructive (already applied, PR is
            // an audit record) or suppression failed.
            "held": held,
        }));
    }

    opened
}

struct PendingMemoryMutation {
    action: String,
    id: String,
    reason: Option<String>,
}

/// Pre-gate memory mutations that would otherwise be irreversible or directly
/// inhibitory before the reviewer sees them.
///
/// Normal risky writes are still handled post-commit by [`gate_writes`]. Purge,
/// delete, and suppress are different: executing the tool first either removes
/// the row or mutates retrieval influence before review. Under Risk-Gated and
/// Paranoid modes this function opens a pending Memory PR and returns a tool
/// response without performing the mutation. Fast mode keeps the historical
/// direct-execution behavior.
pub fn gate_pending_memory_mutation(
    storage: &Arc<Storage>,
    event_tx: Option<&broadcast::Sender<VestigeEvent>>,
    run_id: &str,
    tool: &str,
    args: &Option<serde_json::Value>,
    mode: vestige_core::ReviewMode,
) -> Result<Option<serde_json::Value>, String> {
    use vestige_core::{
        MemoryPr, MemoryPrKind, MemoryPrStatus, RiskClass, WriteContext, classify_write,
    };

    if matches!(mode, vestige_core::ReviewMode::Fast) {
        return Ok(None);
    }

    let Some(pending) = pending_memory_mutation(tool, args) else {
        return Ok(None);
    };
    let node = match storage.get_node(&pending.id) {
        Ok(Some(node)) => node,
        Ok(None) => return Ok(None),
        Err(e) => return Err(format!("failed to inspect memory before review gate: {e}")),
    };

    let ctx = WriteContext {
        source: Some(WriteSource::Agent),
        node_type: node.node_type.clone(),
        content: node.content.clone(),
        tags: node.tags.clone(),
        forgets: true,
        ..Default::default()
    };
    let (class, signals) = classify_write(&ctx, mode);
    if class != RiskClass::Review {
        return Ok(None);
    }

    let sensitive = is_sensitive_pr_content(&node.content, &signals);
    let preview = content_preview(&node.content, sensitive);
    let content_hash = hash_content(&node.content);
    let kind = MemoryPrKind::NodeDecayed;
    let title = format!("{}: {}", pr_kind_phrase(kind), preview);
    let pr = MemoryPr {
        id: format!("pr_{}", uuid::Uuid::new_v4().simple()),
        kind,
        status: MemoryPrStatus::Pending,
        title: title.clone(),
        diff: serde_json::json!({
            "decision": pending.action,
            "pendingAction": pending.action,
            "requiresApproval": true,
            "reason": pending.reason,
            "node": {
                "id": pending.id,
                "nodeType": node.node_type,
                "contentPreview": preview,
                "contentHash": content_hash,
                "tags": node.tags,
                "deleted": false,
            },
        }),
        signals: signals.clone(),
        subject_id: Some(pending.id.clone()),
        run_id: Some(run_id.to_string()),
        created_at: Utc::now().to_rfc3339(),
        decided_at: None,
        decision: None,
    };

    if let Err(e) = storage.save_memory_pr(&pr) {
        tracing::warn!("pending destructive Memory PR save failed: {e}");
        return Err(format!(
            "review gate failed closed: could not open Memory PR for pending mutation: {e}"
        ));
    }

    if let Some(tx) = event_tx {
        let _ = tx.send(VestigeEvent::MemoryPrOpened {
            id: pr.id.clone(),
            kind: kind.as_str().to_string(),
            title: title.clone(),
            signal_count: signals.len(),
            run_id: Some(run_id.to_string()),
            timestamp: Utc::now(),
        });
    }

    let opened = serde_json::json!({
        "id": pr.id,
        "kind": kind.as_str(),
        "title": pr.title,
        "signals": signals,
        "subjectId": pending.id,
    });

    Ok(Some(serde_json::json!({
        "action": format!("{}_pending_review", pending.action),
        "success": false,
        "pendingReview": true,
        "nodeId": pending.id,
        "message": "Mutation was not executed. Vestige opened a Memory PR and is waiting for review.",
        "memoryPrsOpened": [opened],
        "memoryPrNotice": "Vestige opened a Memory PR before applying this destructive or suppressive memory mutation. Approve with `forget`; keep the memory with `promote`; hold it suppressed with `quarantine`.",
    })))
}

fn pending_memory_mutation(
    tool: &str,
    args: &Option<serde_json::Value>,
) -> Option<PendingMemoryMutation> {
    let args = args.as_ref()?;
    match tool {
        "memory" => {
            let action = args.get("action")?.as_str()?.to_ascii_lowercase();
            if !matches!(action.as_str(), "purge" | "delete") {
                return None;
            }
            if !args
                .get("confirm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return None;
            }
            Some(PendingMemoryMutation {
                action,
                id: args.get("id")?.as_str()?.to_string(),
                reason: args
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            })
        }
        "delete_knowledge" => {
            if !args
                .get("confirm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return None;
            }
            Some(PendingMemoryMutation {
                action: "delete".to_string(),
                id: args.get("id")?.as_str()?.to_string(),
                reason: args
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            })
        }
        "suppress" => {
            if args
                .get("reverse")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return None;
            }
            Some(PendingMemoryMutation {
                action: "suppress".to_string(),
                id: args.get("id")?.as_str()?.to_string(),
                reason: args
                    .get("reason")
                    .or_else(|| args.get("note"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            })
        }
        _ => None,
    }
}

/// Whether a write decision permanently removes / forgets memory (so the live
/// row may already be gone when the gate runs).
fn is_destructive_decision(label: &str) -> bool {
    matches!(
        label,
        "purge" | "purged" | "delete" | "deleted" | "forget" | "forgotten"
    )
}

/// Whether content must be fully redacted from a Memory PR.
///
/// Risk signals cover the broad policy categories (auth, money, identity,
/// etc.), but credential detection is intentionally stricter: a legacy or
/// explicitly allowed node can still hold a provider-shaped secret even when
/// its surrounding prose did not trigger a sensitive-topic signal.  Both PR
/// creation paths use this helper so neither can turn that secret into a
/// title, preview, or diff field.
fn is_sensitive_pr_content(content: &str, signals: &[vestige_core::RiskSignal]) -> bool {
    signals
        .iter()
        .any(|s| s.code == "sensitive_topic" || s.code == "sensitive_node_type")
        || has_blocking_secret(content)
}

/// Match the storage boundary's blocking credential policy without retaining
/// the detector output or the matched value.
fn has_blocking_secret(content: &str) -> bool {
    vestige_core::scan_secrets(content)
        .iter()
        .any(vestige_core::SecretFinding::blocks_ingestion)
}

/// A short, privacy-preserving preview of memory content for a Memory PR.
/// When the write was flagged for a sensitive topic, the content is redacted
/// entirely — the reviewer sees the risk signals + hash, never the secret.
fn content_preview(content: &str, sensitive: bool) -> String {
    if content.is_empty() {
        return "(no content)".to_string();
    }
    if sensitive {
        return "[redacted — sensitive content; review via risk signals]".to_string();
    }
    let trimmed: String = content.chars().take(80).collect();
    if content.chars().count() > 80 {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

/// FNV-1a hex fingerprint of memory content — lets a reviewer correlate /
/// dedupe without the PR row carrying the raw (possibly secret) text.
fn hash_content(content: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for b in content.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
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
        "recall" | "deep_reference" | "cross_reference" | "search" | "explore_connections"
    )
}

/// Process-private key used to prevent replay item digests from becoming a
/// public content-equality oracle. Operators that need to verify current
/// materializations after a restart can provide a stable 32-byte key as
/// exactly 64 hexadecimal characters in `VESTIGE_REPLAY_DIGEST_KEY`.
///
/// Without that explicit key, Vestige deliberately chooses a fresh random key
/// for each process. Persisted structural replay remains deterministic and
/// restart-safe; content materialization verification fails closed after a
/// restart until a durable local keystore is available.
static REPLAY_DIGEST_KEY: OnceLock<[u8; 32]> = OnceLock::new();

fn replay_digest_key() -> &'static [u8; 32] {
    REPLAY_DIGEST_KEY.get_or_init(|| {
        if let Ok(encoded) = std::env::var("VESTIGE_REPLAY_DIGEST_KEY") {
            if let Some(key) = parse_replay_digest_key(&encoded) {
                return key;
            }
            tracing::warn!(
                "VESTIGE_REPLAY_DIGEST_KEY must be exactly 64 hexadecimal characters; using a process-private replay digest key"
            );
        }

        let mut key = [0_u8; 32];
        key[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        key[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        key
    })
}

fn parse_replay_digest_key(encoded: &str) -> Option<[u8; 32]> {
    if encoded.len() != 64 || !encoded.is_ascii() {
        return None;
    }
    let mut key = [0_u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        let offset = index * 2;
        let high = decode_hex_nibble(encoded.as_bytes()[offset])?;
        let low = decode_hex_nibble(encoded.as_bytes()[offset + 1])?;
        *byte = (high << 4) | low;
    }
    Some(key)
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

struct ReturnedReplayEvidence<'a> {
    memory_id: &'a str,
    value: &'a Value,
    trust_score: f64,
}

/// Select only the evidence collection that crossed the MCP boundary after
/// all ranking, masking, limiting, and token-budget enforcement completed.
/// Candidate-only fields such as `expandable` are never considered.
fn final_returned_replay_evidence<'a>(
    tool: &str,
    result: &'a Value,
) -> Vec<ReturnedReplayEvidence<'a>> {
    let collection = match tool {
        "recall" => result
            .get("results")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty())
            .map(|items| ("results", items))
            .or_else(|| {
                result
                    .get("evidence")
                    .and_then(Value::as_array)
                    .filter(|items| !items.is_empty())
                    .map(|items| ("evidence", items))
            }),
        "search" => result
            .get("results")
            .and_then(Value::as_array)
            .map(|items| ("results", items)),
        "deep_reference" | "cross_reference" => result
            .get("evidence")
            .and_then(Value::as_array)
            .map(|items| ("evidence", items)),
        "explore_connections" => match result.get("action").and_then(Value::as_str) {
            Some("chain") => result
                .get("steps")
                .and_then(Value::as_array)
                .map(|items| ("steps", items)),
            Some("associations") => result
                .get("associations")
                .and_then(Value::as_array)
                .map(|items| ("associations", items)),
            Some("bridges") => result
                .get("bridges")
                .and_then(Value::as_array)
                .map(|items| ("bridges", items)),
            _ => None,
        },
        _ => None,
    };

    collection
        .into_iter()
        .flat_map(|(_, items)| items)
        .filter_map(|value| {
            let memory_id = replay_memory_id(value)?;
            Some(ReturnedReplayEvidence {
                memory_id,
                value,
                trust_score: replay_trust_score(value),
            })
        })
        .collect()
}

fn replay_memory_id(value: &Value) -> Option<&str> {
    value
        .get("id")
        .or_else(|| value.get("memory_id"))
        .or_else(|| value.get("memoryId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
}

fn replay_trust_score(value: &Value) -> f64 {
    [
        "trust",
        "trustScore",
        "trust_score",
        "retentionStrength",
        "retention_strength",
        "activation",
        "score",
        "strength",
        "connectionStrength",
        "connection_strength",
        "combinedScore",
        "combined_score",
    ]
    .into_iter()
    .find_map(|field| value.get(field).and_then(Value::as_f64))
    .filter(|score| score.is_finite())
    .unwrap_or(0.0)
    .clamp(0.0, 1.0)
}

fn replay_decay_risk(trust_score: f64) -> ReplayDecayRisk {
    if trust_score >= 0.7 {
        ReplayDecayRisk::Low
    } else if trust_score >= 0.4 {
        ReplayDecayRisk::Medium
    } else {
        ReplayDecayRisk::High
    }
}

fn replay_policy_bytes(tool: &str, result: &Value, collection: &str) -> Vec<u8> {
    let mut policy = BTreeMap::new();
    policy.insert(
        "selectionBoundary".to_string(),
        Value::String(REPLAY_SELECTION_BOUNDARY.to_string()),
    );
    policy.insert("tool".to_string(), Value::String(tool.to_string()));
    policy.insert(
        "evidenceCollection".to_string(),
        Value::String(collection.to_string()),
    );
    for field in [
        "action",
        "method",
        "retrievalMode",
        "concrete",
        "detailLevel",
        "profile",
        "tokenBudgetLimit",
    ] {
        if let Some(value) = result.get(field)
            && (value.is_string() || value.is_boolean() || value.is_number())
        {
            policy.insert(field.to_string(), value.clone());
        }
    }
    serde_json::to_vec(&policy).unwrap_or_default()
}

fn replay_evidence_collection(tool: &str, result: &Value) -> Option<&'static str> {
    match tool {
        "recall"
            if result
                .get("results")
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty()) =>
        {
            Some("results")
        }
        "recall"
            if result
                .get("evidence")
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty()) =>
        {
            Some("evidence")
        }
        "search" => Some("results"),
        "deep_reference" | "cross_reference" => Some("evidence"),
        "explore_connections" => match result.get("action").and_then(Value::as_str) {
            Some("chain") => Some("steps"),
            Some("associations") => Some("associations"),
            Some("bridges") => Some("bridges"),
            _ => None,
        },
        _ => None,
    }
}

fn build_replay_capsule_draft(
    receipt_id: &str,
    tool: &str,
    result: &Value,
) -> Option<RetrievalReplayCapsuleDraft> {
    let evidence = final_returned_replay_evidence(tool, result);
    if evidence.is_empty() {
        return None;
    }
    let collection = replay_evidence_collection(tool, result)?;
    let items = evidence
        .into_iter()
        .enumerate()
        .filter_map(|(index, evidence)| {
            let evidence_slot = replay_evidence_slot(index + 1);
            let evidence_bytes = serde_json::to_vec(evidence.value).ok()?;
            let token_estimate = u64::try_from(evidence_bytes.len()).ok()?.div_ceil(4);
            Some(RetrievalReplayItemDraft {
                evidence_slot: evidence_slot.clone(),
                memory_id: evidence.memory_id.to_string(),
                private_digest: private_evidence_digest(
                    replay_digest_key(),
                    &evidence_slot,
                    &evidence_bytes,
                ),
                token_estimate,
                trust_score: evidence.trust_score,
                decay_risk: replay_decay_risk(evidence.trust_score),
            })
        })
        .collect::<Vec<_>>();
    if items.is_empty() {
        return None;
    }

    Some(RetrievalReplayCapsuleDraft::new(
        receipt_id,
        replay_policy_digest(&replay_policy_bytes(tool, result, collection)),
        items,
    ))
}

#[derive(Debug)]
struct ReceiptSigner {
    key_id: String,
    seed: [u8; 32],
}

/// Load the live signer only when an operator supplied a complete explicit
/// configuration. A partial configuration, unreadable sidecar, unregistered
/// key, seed/public-key mismatch, or non-active key is an error: retrieval
/// receipt persistence then fails closed instead of quietly emitting unsigned
/// evidence under a signing-enabled deployment.
fn configured_receipt_signer(storage: &Storage) -> Result<Option<ReceiptSigner>, String> {
    let key_id = env::var(RECEIPT_SIGNING_KEY_ID_ENV).ok();
    let seed_path = env::var_os(RECEIPT_SIGNING_SEED_PATH_ENV);
    match (key_id, seed_path) {
        (None, None) => Ok(None),
        (Some(key_id), Some(seed_path)) if !key_id.trim().is_empty() => {
            let seed = load_receipt_signing_seed(std::path::Path::new(&seed_path))
                .map_err(|error| format!("receipt signing seed is unavailable: {error}"))?;
            let registered = storage
                .registered_receipt_signing_key(&key_id)
                .map_err(|error| format!("receipt signing key lookup failed: {error}"))?
                .ok_or_else(|| {
                    format!(
                        "receipt signing key '{key_id}' is not registered; register its public key before enabling signing"
                    )
                })?;
            if registered.status != SigningKeyStatus::Active {
                return Err(format!(
                    "receipt signing key '{key_id}' is not active for new receipts"
                ));
            }
            let public_key = ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            if public_key != registered.public_key {
                return Err(format!(
                    "receipt signing seed does not match registered public key '{key_id}'"
                ));
            }
            Ok(Some(ReceiptSigner { key_id, seed }))
        }
        _ => Err(format!(
            "receipt signing requires both {RECEIPT_SIGNING_KEY_ID_ENV} and {RECEIPT_SIGNING_SEED_PATH_ENV}, or neither"
        )),
    }
}

fn retrieval_projection(receipt: &Receipt) -> Result<RedactionSafeDecisionProjectionV1, String> {
    let count = |name: &str, value: usize| {
        u32::try_from(value).map_err(|_| format!("receipt {name} exceeds the attestation bound"))
    };
    let trust_floor_basis_points = if receipt.trust_floor.is_finite() {
        (receipt.trust_floor.clamp(0.0, 1.0) * 10_000.0).round() as u16
    } else {
        return Err("receipt trust floor is not finite".into());
    };
    Ok(RedactionSafeDecisionProjectionV1::RetrievalSelection {
        returned_count: count("retrieved count", receipt.retrieved.len())?,
        suppressed_count: count("suppressed count", receipt.suppressed.len())?,
        mutation_count: count("mutation count", receipt.mutations.len())?,
        trust_floor_basis_points,
    })
}

fn receipt_memory_ids(receipt: &Receipt) -> Vec<String> {
    let mut ids = BTreeSet::new();
    ids.extend(receipt.retrieved.iter().cloned());
    ids.extend(receipt.suppressed.iter().map(|entry| entry.id.clone()));
    ids.extend(receipt.mutations.iter().map(|mutation| mutation.id.clone()));
    ids.into_iter().collect()
}

fn sign_retrieval_receipt(
    storage: &Storage,
    signer: &ReceiptSigner,
    receipt: &mut Receipt,
) -> Result<
    (
        ReceiptAttestationV1,
        Vec<DisclosureMapping>,
        SignedReceiptAttestation,
    ),
    String,
> {
    let predecessor = storage
        .latest_receipt_chain_entry()
        .map_err(|error| format!("receipt chain lookup failed: {error}"))?;
    let chain_position = predecessor
        .as_ref()
        .map(AttestationChainPosition::Successor)
        .unwrap_or(AttestationChainPosition::Genesis);
    let prepared = ReceiptAttestationV1::build(
        Utc::now(),
        ProducerIdentity::new(
            "vestige-mcp",
            env!("CARGO_PKG_VERSION"),
            "receipt-runtime-v1",
        )
        .map_err(|error| format!("receipt producer identity is invalid: {error}"))?,
        chain_position,
        RECEIPT_ATTESTATION_ALGORITHM_V1,
        retrieval_projection(receipt)?,
        receipt_memory_ids(receipt),
    )
    .map_err(|error| format!("receipt attestation build failed: {error}"))?;
    let (attestation, disclosures) = prepared
        .bind_receipt(receipt)
        .map_err(|error| format!("receipt attestation binding failed: {error}"))?
        .into_parts();
    let signed = sign_attestation(&attestation, &signer.key_id, &signer.seed)
        .map_err(|error| format!("receipt attestation signing failed: {error}"))?;
    Ok((attestation, disclosures, signed))
}

/// Build a [`Receipt`] from a retrieval tool's response JSON, persist it, and
/// return it as JSON ready to attach to that response. Reuses exactly the data
/// the tool already computed (retrieved ids + trust, suppressed ids + reason,
/// the activation path) — so the receipt is the auditable "nutrition label" for
/// the answer and costs nothing extra to produce.
///
/// Returns `None` for non-retrieval tools or empty results. When persistence
/// fails, returns an explicit status object with no receipt id or payload, so a
/// successful retrieval never implies it has durable receipt evidence.
pub fn build_and_save_receipt(
    storage: &Arc<Storage>,
    run_id: &str,
    tool: &str,
    result: &serde_json::Value,
) -> Option<serde_json::Value> {
    if tool == "backfill" {
        return build_and_save_backfill_receipt(storage, run_id, result);
    }
    if !is_retrieval_tool(tool) {
        return None;
    }

    let returned_evidence = final_returned_replay_evidence(tool, result);
    if returned_evidence.is_empty() {
        return None;
    }
    let retrieved: Vec<String> = returned_evidence
        .iter()
        .map(|evidence| evidence.memory_id.to_string())
        .collect();
    let trust_scores: Vec<f64> = returned_evidence
        .iter()
        .map(|evidence| evidence.trust_score)
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

    let mut receipt = Receipt::build(
        Utc::now(),
        run_id,
        retrieved,
        suppressed,
        activation_path,
        &trust_scores,
        Vec::new(),
    );
    let signing_result = configured_receipt_signer(storage).and_then(|signer| {
        let capsule_and_save = |receipt: &Receipt,
                                signed: Option<(
            &ReceiptAttestationV1,
            &[DisclosureMapping],
            &SignedReceiptAttestation,
        )>|
         -> Result<(), String> {
            let capsule = build_replay_capsule_draft(&receipt.receipt_id, tool, result)
                .ok_or_else(|| "retrieval receipt has no replayable final evidence".to_string())?;
            match signed {
                Some((attestation, disclosures, signed)) => storage
                    .save_signed_retrieval_receipt_with_replay_capsule_atomic(
                        SignedReceiptWrite {
                            receipt,
                            attestation,
                            signed,
                            disclosures,
                            run_id: Some(run_id),
                            tool: Some(tool),
                            query: None,
                        },
                        &capsule,
                    )
                    .map(|_| ())
                    .map_err(|error| format!("atomic signed receipt save failed: {error}")),
                None => storage
                    .save_retrieval_receipt_with_replay_capsule(
                        receipt,
                        Some(run_id),
                        Some(tool),
                        &capsule,
                    )
                    .map(|_| ())
                    .map_err(|error| format!("atomic receipt save failed: {error}")),
            }
        };
        match signer {
            Some(signer) => {
                let (attestation, disclosures, signed) =
                    sign_retrieval_receipt(storage, &signer, &mut receipt)?;
                capsule_and_save(&receipt, Some((&attestation, &disclosures, &signed)))
            }
            None => capsule_and_save(&receipt, None),
        }
    });
    if let Err(error) = signing_result {
        tracing::warn!("atomic receipt persistence failed: {error}");
        return Some(receipt_persistence_unavailable());
    }
    Some(serde_json::to_value(&receipt).unwrap_or(serde_json::Value::Null))
}

fn receipt_persistence_unavailable() -> serde_json::Value {
    serde_json::json!({
        "persistence": "unavailable",
        "claimBoundary": "Receipt evidence was not persisted; no durable receipt is claimed.",
        "message": "Receipt persistence is temporarily unavailable. Retry the retrieval to obtain durable evidence."
    })
}

/// Persist the exact output of a Backfill run as typed receipt evidence.
/// Records candidates, not an asserted root cause — the UI must keep that
/// epistemic boundary. Incomplete `path_ids` still save the receipt but
/// fail closed at render time via [`Receipt::backfill_path_ids`].
fn build_and_save_backfill_receipt(
    storage: &Arc<Storage>,
    run_id: &str,
    result: &Value,
) -> Option<Value> {
    if !result.get("triggered")?.as_bool()? {
        return None;
    }
    let failure = result.get("failure")?;
    let failure_id = failure.get("id")?.as_str()?.to_string();
    let failure_preview = failure
        .get("content_preview")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let candidates: Vec<BackfillCandidateEvidence> = result
        .get("causes")
        .and_then(|v| v.as_array())?
        .iter()
        .filter_map(|cause| {
            Some(BackfillCandidateEvidence {
                memory_id: cause.get("memory_id")?.as_str()?.to_string(),
                content_preview: cause
                    .get("content_preview")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                shared_entities: cause
                    .get("shared_entities")
                    .and_then(|v| v.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(ToString::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                age_days_before_failure: cause
                    .get("age_days_before_failure")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                similarity_rank: cause
                    .get("similarity_rank")
                    .and_then(|v| v.as_u64())
                    .map(|rank| rank as usize),
                backfill_score: cause
                    .get("backfill_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                promoted: cause
                    .get("promoted")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                candidate_edge_persisted: cause
                    .get("candidate_edge_persisted")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            })
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let retrieved: Vec<String> = candidates
        .iter()
        .map(|candidate| candidate.memory_id.clone())
        .collect();
    let activation_path: Vec<String> = candidates
        .iter()
        .map(|candidate| format!("{} -> {}", candidate.memory_id, failure_id))
        .collect();
    let mutations = candidates
        .iter()
        .filter(|candidate| candidate.promoted)
        .map(|candidate| vestige_core::ReceiptMutation {
            id: candidate.memory_id.clone(),
            kind: "backfill_candidate_promoted".to_string(),
            note: Some("Promoted after an explicit-entity backward candidate match".to_string()),
        })
        .collect();
    let path_ids: Vec<String> = result
        .get("path_ids")
        .and_then(|value| value.as_array())
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();
    let receipt = Receipt::build(
        Utc::now(),
        run_id,
        retrieved,
        Vec::new(),
        activation_path,
        &[],
        mutations,
    )
    .with_evidence(ReceiptEvidence::Backfill {
        schema: BACKFILL_RECEIPT_SCHEMA_V1.to_string(),
        schema_version: 1,
        failure_id,
        failure_preview,
        scanned: result.get("scanned").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        lookback_days: result
            .get("lookback_days")
            .and_then(|v| v.as_i64())
            .unwrap_or(30),
        baseline: "embedding cosine rank within the scanned candidate set".to_string(),
        path_ids,
        candidates,
        claim_boundary: BACKFILL_RECEIPT_CLAIM_BOUNDARY.to_string(),
    });
    if let Err(error) = storage.save_receipt(&receipt, Some(run_id), Some("backfill"), None) {
        tracing::warn!(%error, "backfill receipt save failed");
    }
    Some(serde_json::to_value(receipt).unwrap_or(Value::Null))
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

/// Fixed trace marker used when `smart_ingest` rejects credential-shaped
/// content.  Hashing those arguments, even without storing the raw JSON,
/// leaves a stable value derived from a rejected secret that can be brute
/// forced for low-entropy values.  The marker records only that redaction
/// happened; it is intentionally independent of every argument byte.
const REDACTED_SMART_INGEST_SECRET_ARGS_HASH: &str = "redacted_secret_input";

fn trace_args_hash(tool: &str, args: &Option<Value>) -> String {
    if tool == "smart_ingest" && smart_ingest_has_blocking_secret(args) {
        return REDACTED_SMART_INGEST_SECRET_ARGS_HASH.to_string();
    }
    hash_args(args)
}

/// Detect the fields `smart_ingest` passes to the credential-aware storage
/// boundary. This mirrors its single and batch request shapes, so a rejected
/// item cannot leave a raw-derived fingerprint behind in the opening trace.
fn smart_ingest_has_blocking_secret(args: &Option<Value>) -> bool {
    let Some(args) = args.as_ref() else {
        return false;
    };

    smart_ingest_input_has_blocking_secret(args)
        || args
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flat_map(|items| items.iter())
            .any(smart_ingest_input_has_blocking_secret)
}

fn smart_ingest_input_has_blocking_secret(input: &Value) -> bool {
    let scalar_fields = ["content", "source"]
        .into_iter()
        .filter_map(|field| input.get(field).and_then(Value::as_str));
    let tags = input
        .get("tags")
        .and_then(Value::as_array)
        .into_iter()
        .flat_map(|tags| tags.iter())
        .filter_map(Value::as_str);

    scalar_fields.chain(tags).any(has_blocking_secret)
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
            args_hash: trace_args_hash(tool, args),
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
    // Receipts committed atomically inside write tools do not yet know the
    // enclosing MCP run id. Link their denormalized Black Box column once the
    // tool returns, without rewriting the signed/typed payload surface.
    for receipt_id in extract_embedded_receipt_ids(result) {
        let _ = storage.link_receipt_to_run(&receipt_id, run_id);
    }

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

fn extract_embedded_receipt_ids(result: &Value) -> Vec<String> {
    fn push_from_item(item: &Value, out: &mut Vec<String>) {
        if let Some(capture) = item.get("synapticCapture") {
            if let Some(id) = capture.get("receiptId").and_then(Value::as_str)
                && !out.iter().any(|existing| existing == id)
            {
                out.push(id.to_string());
            }
            if let Some(forward) = capture.get("forwardReceipts").and_then(Value::as_array) {
                for pair in forward {
                    if let Some(id) = pair.get("receiptId").and_then(Value::as_str)
                        && !out.iter().any(|existing| existing == id)
                    {
                        out.push(id.to_string());
                    }
                }
            }
        }
    }

    let mut out = Vec::new();
    push_from_item(result, &mut out);
    if let Some(items) = result.get("results").and_then(Value::as_array) {
        for item in items {
            push_from_item(item, &mut out);
        }
    }
    out
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
    let confidence = veto
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
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
                        // char-boundary-safe: byte-slicing &s[..8] panics when a
                        // multi-byte UTF-8 char straddles byte 8.
                        .map(|s| s.chars().take(8).collect::<String>())
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

    fn receipt_signing_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    struct ReceiptSigningEnvReset {
        key_id: Option<std::ffi::OsString>,
        seed_path: Option<std::ffi::OsString>,
    }

    impl ReceiptSigningEnvReset {
        fn capture() -> Self {
            Self {
                key_id: env::var_os(RECEIPT_SIGNING_KEY_ID_ENV),
                seed_path: env::var_os(RECEIPT_SIGNING_SEED_PATH_ENV),
            }
        }
    }

    impl Drop for ReceiptSigningEnvReset {
        fn drop(&mut self) {
            for (name, previous) in [
                (RECEIPT_SIGNING_KEY_ID_ENV, self.key_id.take()),
                (RECEIPT_SIGNING_SEED_PATH_ENV, self.seed_path.take()),
            ] {
                match previous {
                    Some(value) => unsafe { env::set_var(name, value) },
                    None => unsafe { env::remove_var(name) },
                }
            }
        }
    }

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
    fn rejected_smart_ingest_secret_uses_fixed_non_derived_trace_marker() {
        let s = store();
        let secret = format!("ghp_{}", "A".repeat(36));
        let args = Some(serde_json::json!({
            "content": format!("{secret}"),
            "runId": "run_secret_rejection"
        }));

        // `record_call` runs before smart_ingest dispatches and rejects the
        // credential, so this is the persistence boundary that must redact.
        record_call(&s, None, "run_secret_rejection", "smart_ingest", &args);

        let trace = s.get_trace("run_secret_rejection").unwrap();
        assert_eq!(trace.len(), 1);
        let MemoryTraceEvent::McpCall { args_hash, .. } = &trace[0] else {
            panic!("opening trace event must be mcp.call");
        };
        assert_eq!(args_hash, REDACTED_SMART_INGEST_SECRET_ARGS_HASH);
        assert_ne!(args_hash, &hash_args(&args));

        let persisted = serde_json::to_string(&trace).unwrap();
        assert!(
            !persisted.contains(&secret),
            "rejected credential must not be persisted in the trace"
        );
    }

    #[test]
    fn rejected_smart_ingest_batch_secret_uses_fixed_trace_marker() {
        let secret = format!("ghp_{}", "B".repeat(36));
        let args = Some(serde_json::json!({
            "items": [
                { "content": "safe item" },
                { "content": format!("{secret}") }
            ]
        }));

        assert_eq!(
            trace_args_hash("smart_ingest", &args),
            REDACTED_SMART_INGEST_SECRET_ARGS_HASH,
            "one rejected batch item must redact the whole call fingerprint"
        );
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
    fn replay_digest_key_parser_requires_exact_32_byte_hex() {
        let lower = "00ff".repeat(16);
        let upper = "A1".repeat(32);
        assert_eq!(
            parse_replay_digest_key(&lower).unwrap()[..4],
            [0, 255, 0, 255]
        );
        assert_eq!(parse_replay_digest_key(&upper).unwrap(), [0xA1; 32]);
        assert!(parse_replay_digest_key(&"0".repeat(63)).is_none());
        assert!(parse_replay_digest_key(&"z0".repeat(32)).is_none());
    }

    #[test]
    fn replay_capsule_draft_freezes_only_final_returned_evidence() {
        let result = serde_json::json!({
            "query": "raw query must not enter replay policy",
            "method": "hybrid+cognitive",
            "retrievalMode": "balanced",
            "detailLevel": "summary",
            "profile": "standard",
            "tokenBudgetLimit": 200,
            "tokenBudgetUsed": 87,
            "results": [
                {
                    "id": "memory_selected_1",
                    "content": "first private memory fragment",
                    "retentionStrength": 0.82,
                    "combinedScore": 0.91
                },
                {
                    "id": "memory_selected_2",
                    "content": "second private memory fragment",
                    "retentionStrength": 0.51,
                    "combinedScore": 0.74
                }
            ],
            "expandable": ["memory_candidate_not_returned"]
        });

        let draft = build_replay_capsule_draft("r_2026_08_10_run_abcdef", "recall", &result)
            .expect("final selected evidence should create a capsule draft");
        assert_eq!(draft.items.len(), 2);
        assert_eq!(draft.items[0].evidence_slot, "evidence_1");
        assert_eq!(draft.items[1].evidence_slot, "evidence_2");
        assert_eq!(draft.items[0].memory_id, "memory_selected_1");
        assert_eq!(draft.items[1].memory_id, "memory_selected_2");
        assert_eq!(draft.items[0].trust_score, 0.82);
        assert_eq!(draft.items[1].trust_score, 0.51);
        assert_eq!(draft.items[0].decay_risk, ReplayDecayRisk::Low);
        assert_eq!(draft.items[1].decay_risk, ReplayDecayRisk::Medium);

        let first_bytes = serde_json::to_vec(&result["results"][0]).unwrap();
        assert_eq!(
            draft.items[0].private_digest,
            private_evidence_digest(replay_digest_key(), "evidence_1", &first_bytes)
        );
        assert_eq!(
            draft.items[0].token_estimate,
            u64::try_from(first_bytes.len()).unwrap().div_ceil(4)
        );
        assert!(
            draft
                .items
                .iter()
                .all(|item| item.memory_id != "memory_candidate_not_returned")
        );

        let mut changed_query_and_candidates = result.clone();
        changed_query_and_candidates["query"] = serde_json::json!("different secret query");
        changed_query_and_candidates["expandable"] =
            serde_json::json!(["different_unreturned_candidate"]);
        let retry = build_replay_capsule_draft(
            "r_2026_08_10_run_abcdef",
            "recall",
            &changed_query_and_candidates,
        )
        .unwrap();
        assert_eq!(draft.policy_digest, retry.policy_digest);
        assert_eq!(draft.items, retry.items);
    }

    #[test]
    fn replay_capsule_uses_reason_evidence_not_recommended_duplicate() {
        let result = serde_json::json!({
            "query": "private reasoning query",
            "recommended": {
                "memory_id": "memory_primary",
                "answer_preview": "duplicate answer material",
                "trust_score": 0.9
            },
            "evidence": [
                { "id": "memory_primary", "preview": "primary", "trust": 0.9 },
                { "id": "memory_support", "preview": "support", "trust": 0.6 }
            ],
            "reasoning": "raw synthesized output"
        });

        let draft = build_replay_capsule_draft("r_reason", "recall", &result).unwrap();
        assert_eq!(draft.items.len(), 2);
        assert_eq!(draft.items[0].memory_id, "memory_primary");
        assert_eq!(draft.items[1].memory_id, "memory_support");
    }

    #[test]
    fn receipt_and_final_capsule_persist_atomically_without_raw_replay_material() {
        // configured_receipt_signer reads process env. Serialize with the
        // signing-env tests so a parallel missing-key fixture cannot turn this
        // unsigned persist into "temporarily unavailable".
        let _lock = receipt_signing_env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("retrieval-capsule.db");
        let storage = Arc::new(vestige_core::Storage::new(Some(db_path.clone())).unwrap());
        // Replay-item rows deliberately carry an FK to the canonical memory so
        // this regression exercises the same path as a live retrieval rather
        // than relying on impossible fixture identifiers.
        let first = storage
            .ingest(vestige_core::IngestInput {
                content: "raw memory sentinel alpha".into(),
                ..Default::default()
            })
            .unwrap();
        let second = storage
            .ingest(vestige_core::IngestInput {
                content: "raw memory sentinel beta".into(),
                ..Default::default()
            })
            .unwrap();
        let result = serde_json::json!({
            "query": "private query sentinel",
            "method": "concrete",
            "retrievalMode": "precise",
            "detailLevel": "summary",
            "profile": "standard",
            "tokenBudgetLimit": 120,
            "results": [
                {
                    "id": first.id,
                    "content": "raw memory sentinel alpha",
                    "retentionStrength": 0.75
                },
                {
                    "id": second.id,
                    "content": "raw memory sentinel beta",
                    "retentionStrength": 0.35
                }
            ],
            "expandable": ["memory_expandable_sentinel"]
        });

        let receipt_json =
            build_and_save_receipt(&storage, "run_product_replay", "recall", &result)
                .expect("retrieval should return its receipt");
        let receipt_id = receipt_json["receipt_id"]
            .as_str()
            .unwrap_or_else(|| panic!("expected signed receipt, got {receipt_json}"));
        let persisted_receipt = storage.get_receipt(receipt_id).unwrap().unwrap();
        assert_eq!(
            persisted_receipt.retrieved,
            vec![first.id.clone(), second.id.clone()],
            "live retrieval evidence must retain its canonical active ids"
        );
        assert_eq!(persisted_receipt.trust_floor, 0.35);

        let capsule = storage
            .get_retrieval_replay_capsule(receipt_id)
            .unwrap()
            .expect("receipt and capsule must commit together");
        assert_eq!(capsule.item_count, Some(2));
        assert_eq!(
            capsule
                .items
                .iter()
                .map(|item| item.evidence_slot.as_str())
                .collect::<Vec<_>>(),
            vec!["evidence_1", "evidence_2"]
        );
        let public_capsule_json = serde_json::to_string(&capsule).unwrap();
        for forbidden in [
            "private query sentinel",
            "raw memory sentinel alpha",
            "raw memory sentinel beta",
            "memory_expandable_sentinel",
            "b3k:",
        ] {
            assert!(
                !public_capsule_json.contains(forbidden),
                "public capsule leaked {forbidden}"
            );
        }

        let reader = rusqlite::Connection::open(db_path).unwrap();
        let (query, payload): (Option<String>, String) = reader
            .query_row(
                "SELECT query, payload FROM memory_receipts WHERE receipt_id = ?1",
                [receipt_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert!(
            query.is_none(),
            "atomic replay receipts must not duplicate queries"
        );
        for forbidden in [
            "private query sentinel",
            "raw memory sentinel alpha",
            "raw memory sentinel beta",
            "memory_expandable_sentinel",
        ] {
            assert!(
                !payload.contains(forbidden),
                "receipt payload leaked {forbidden}"
            );
        }
        let expandable_rows: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM retrieval_replay_items WHERE memory_id = ?1",
                ["memory_expandable_sentinel"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(expandable_rows, 0);
    }

    #[test]
    fn receipt_persistence_failure_is_explicit_and_never_claims_a_receipt() {
        let status = receipt_persistence_unavailable();
        assert_eq!(status["persistence"], "unavailable");
        assert!(status.get("receipt_id").is_none());
        assert!(
            status["claimBoundary"]
                .as_str()
                .unwrap()
                .contains("not persisted")
        );
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
        assert_eq!(
            extract_writes(&single),
            vec![("n1".into(), "create".into())]
        );
        let batch = serde_json::json!({
            "results": [ { "decision": "update", "id": "n2" } ]
        });
        assert_eq!(extract_writes(&batch), vec![("n2".into(), "update".into())]);
    }

    #[test]
    fn extract_writes_recognizes_action_shape_b2() {
        // B2: memory promote/demote return `action` + `nodeId`, not `decision`.
        let promoted = serde_json::json!({ "action": "promoted", "nodeId": "m1" });
        assert_eq!(
            extract_writes(&promoted),
            vec![("m1".into(), "promoted".into())]
        );
        let demoted = serde_json::json!({ "action": "demoted", "nodeId": "m2" });
        assert_eq!(
            extract_writes(&demoted),
            vec![("m2".into(), "demoted".into())]
        );
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
    fn destructive_decision_classification_c2() {
        for d in [
            "purge",
            "delete",
            "forget",
            "purged",
            "deleted",
            "forgotten",
        ] {
            assert!(is_destructive_decision(d), "{d} is destructive");
        }
        for d in ["create", "update", "promote", "reinforce"] {
            assert!(!is_destructive_decision(d), "{d} is not destructive");
        }
    }

    #[test]
    fn content_preview_redacts_sensitive_and_truncates() {
        // PRIV: sensitive content is fully redacted, never previewed.
        assert_eq!(
            content_preview("the production auth token is sk-abc123", true),
            "[redacted — sensitive content; review via risk signals]"
        );
        // Ordinary content is truncated, not redacted.
        let long = "a".repeat(200);
        let prev = content_preview(&long, false);
        assert!(prev.ends_with('…'));
        assert!(prev.chars().count() <= 81);
        // Empty content.
        assert_eq!(content_preview("", false), "(no content)");
    }

    #[test]
    fn content_hash_is_stable_and_hides_text() {
        let h = hash_content("my secret memory");
        assert_eq!(h, hash_content("my secret memory"), "stable");
        assert!(!h.contains("secret"));
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn embedded_synaptic_receipts_are_extracted_for_run_linkage() {
        let result = serde_json::json!({
            "synapticCapture": {
                "receiptId": "r_one",
                "forwardReceipts": [
                    { "receiptId": "r_pair_one" },
                    { "receiptId": "r_one" }
                ]
            },
            "results": [
                { "synapticCapture": {
                    "receiptId": "r_two",
                    "forwardReceipts": [{ "receiptId": "r_pair_two" }]
                } },
                { "synapticCapture": { "receiptId": "r_one" } }
            ]
        });
        assert_eq!(
            extract_embedded_receipt_ids(&result),
            vec![
                "r_one".to_string(),
                "r_pair_one".to_string(),
                "r_two".to_string(),
                "r_pair_two".to_string()
            ]
        );
    }

    #[test]
    fn extract_writes_recognizes_destructive_actions_c2() {
        // C2: purge/delete are brain mutations and must trace + be gateable.
        for act in ["purge", "delete"] {
            let r = serde_json::json!({ "action": act, "nodeId": "m1", "success": true });
            assert_eq!(
                extract_writes(&r),
                vec![("m1".into(), act.to_string())],
                "{act} must be traced as a write"
            );
        }
    }

    fn store() -> std::sync::Arc<vestige_core::Storage> {
        let dir = tempfile::tempdir().unwrap();
        std::sync::Arc::new(
            vestige_core::Storage::new(Some(dir.path().join("gate_test.db"))).unwrap(),
        )
    }

    #[test]
    fn configured_signing_commits_receipt_envelope_and_replay_capsule_together() {
        let _lock = receipt_signing_env_lock().lock().unwrap();
        let _reset = ReceiptSigningEnvReset::capture();
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("signed-receipt.db"))).unwrap());
        let provisioned = vestige_core::storage::provision_receipt_signing_key_sidecar(
            &dir.path().join("receipt-keys"),
            "test-receipt-key",
            chrono::DateTime::<Utc>::UNIX_EPOCH,
        )
        .unwrap();
        storage
            .register_receipt_signing_key(&provisioned.trusted_key)
            .unwrap();
        unsafe {
            env::set_var(RECEIPT_SIGNING_KEY_ID_ENV, "test-receipt-key");
            env::set_var(RECEIPT_SIGNING_SEED_PATH_ENV, &provisioned.seed_path);
        }
        let memory = storage
            .ingest(vestige_core::IngestInput {
                content: "private retrieval content".to_string(),
                ..Default::default()
            })
            .unwrap();

        let result = serde_json::json!({
            "results": [{
                "id": memory.id,
                "content": "private retrieval content",
                "retentionStrength": 0.91
            }]
        });
        let mut receipt = Receipt::build(
            Utc::now(),
            "run_signed",
            vec![memory.id],
            Vec::new(),
            Vec::new(),
            &[0.91],
            Vec::new(),
        );
        let signer = configured_receipt_signer(&storage).unwrap().unwrap();
        let (attestation, disclosures, signed) =
            sign_retrieval_receipt(&storage, &signer, &mut receipt).unwrap();
        let capsule = build_replay_capsule_draft(&receipt.receipt_id, "recall", &result).unwrap();
        storage
            .save_signed_retrieval_receipt_with_replay_capsule_atomic(
                SignedReceiptWrite {
                    receipt: &receipt,
                    attestation: &attestation,
                    signed: &signed,
                    disclosures: &disclosures,
                    run_id: Some("run_signed"),
                    tool: Some("recall"),
                    query: None,
                },
                &capsule,
            )
            .unwrap();
        let receipt_id = receipt.receipt_id.as_str();
        assert!(receipt_id.starts_with("ratt_"));
        assert_eq!(
            storage.receipt_attestation_status(receipt_id).unwrap(),
            Some(vestige_core::storage::ReceiptAttestationStatus::SignedV1)
        );
        assert!(
            storage
                .get_retrieval_replay_capsule(receipt_id)
                .unwrap()
                .is_some()
        );
        let verification = storage
            .verify_stored_receipt_attestation(receipt_id)
            .unwrap()
            .expect("stored verification");
        assert!(
            verification.is_valid(),
            "{:?}",
            verification.report.failures
        );
        assert!(
            storage
                .get_receipt_attestation_envelope(receipt_id)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn partial_signing_configuration_fails_closed() {
        let _lock = receipt_signing_env_lock().lock().unwrap();
        let _reset = ReceiptSigningEnvReset::capture();
        unsafe {
            env::set_var(RECEIPT_SIGNING_KEY_ID_ENV, "missing-key");
            env::remove_var(RECEIPT_SIGNING_SEED_PATH_ENV);
        }
        let storage = store();
        assert!(configured_receipt_signer(&storage).is_err());
    }

    #[test]
    fn gate_opens_pr_for_destructive_write_after_node_deleted_c2() {
        // C2-deep: the row is GONE by the time the gate runs (purge deleted it),
        // but a destructive write must STILL open a Memory PR — not be skipped.
        let s = store();
        let node = s
            .ingest(vestige_core::IngestInput {
                content: "A memory the agent is about to purge.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        // Actually delete the row, like purge does.
        let _ = s.delete_node(&node.id);
        assert!(s.get_node(&node.id).unwrap().is_none(), "row is gone");

        // The tool result the recorder sees for the purge.
        let result = serde_json::json!({ "action": "purge", "nodeId": node.id, "success": true });
        let opened = gate_writes(
            &s,
            None,
            "run_c2",
            "memory",
            &result,
            vestige_core::ReviewMode::RiskGated,
        );

        assert_eq!(
            opened.len(),
            1,
            "destructive write must open a PR even with the node gone"
        );
        let pr = s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(pr.len(), 1);
        assert_eq!(pr[0].subject_id.as_deref(), Some(node.id.as_str()));
        // The diff marks the node as deleted and carries no resurrected content.
        assert_eq!(pr[0].diff["node"]["deleted"], serde_json::json!(true));
    }

    #[test]
    fn gate_redacts_sensitive_content_in_pr_priv() {
        // PRIV: a write gated for a sensitive topic must NOT carry the raw
        // content into the PR diff/title — only a redaction + hash.
        let s = store();
        let secret = "the production auth token is sk-live-SECRET-XYZ";
        let node = s
            .ingest(vestige_core::IngestInput {
                content: secret.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let result = serde_json::json!({ "decision": "create", "nodeId": node.id });
        let opened = gate_writes(
            &s,
            None,
            "run_priv",
            "smart_ingest",
            &result,
            vestige_core::ReviewMode::RiskGated,
        );
        assert_eq!(opened.len(), 1, "sensitive write opens a PR");

        let pr = &s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap()[0];
        let serialized = serde_json::to_string(pr).unwrap();
        assert!(
            !serialized.contains("SECRET-XYZ") && !serialized.contains("sk-live"),
            "PR must not contain the raw secret content; got: {serialized}"
        );
        assert!(
            serialized.contains("redacted"),
            "PR must mark the content redacted"
        );
        // A content hash is present so reviewers can still correlate.
        assert!(pr.diff["node"]["contentHash"].as_str().is_some());
    }

    #[test]
    fn gate_redacts_detected_secret_without_sensitive_topic_signal() {
        let s = store();
        let secret = format!("ghp_{}", "C".repeat(36));
        let node = s
            .ingest_with_secret_policy(
                vestige_core::IngestInput {
                    content: secret.clone(),
                    node_type: "fact".to_string(),
                    ..Default::default()
                },
                vestige_core::SecretPolicy::AllowExplicitly,
            )
            .unwrap();
        let result = serde_json::json!({ "decision": "create", "nodeId": node.id });

        // Paranoid mode opens a PR even though a bare token has no topic signal;
        // credential detection itself must still force full redaction.
        let opened = gate_writes(
            &s,
            None,
            "run_detected_secret",
            "smart_ingest",
            &result,
            vestige_core::ReviewMode::Paranoid,
        );
        assert_eq!(opened.len(), 1);

        let pr = &s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap()[0];
        let serialized = serde_json::to_string(pr).unwrap();
        assert!(!serialized.contains(&secret));
        assert!(serialized.contains("redacted"));
    }

    #[test]
    fn pre_gate_redacts_detected_secret_without_sensitive_topic_signal() {
        let s = store();
        let secret = format!("ghp_{}", "D".repeat(36));
        let node = s
            .ingest_with_secret_policy(
                vestige_core::IngestInput {
                    content: secret.clone(),
                    node_type: "fact".to_string(),
                    ..Default::default()
                },
                vestige_core::SecretPolicy::AllowExplicitly,
            )
            .unwrap();
        let args = Some(serde_json::json!({
            "action": "purge",
            "id": node.id,
            "confirm": true
        }));

        let response = gate_pending_memory_mutation(
            &s,
            None,
            "run_pending_detected_secret",
            "memory",
            &args,
            vestige_core::ReviewMode::RiskGated,
        )
        .unwrap();
        assert!(response.is_some());

        let pr = &s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap()[0];
        let serialized = serde_json::to_string(pr).unwrap();
        assert!(!serialized.contains(&secret));
        assert!(serialized.contains("redacted"));
    }

    #[test]
    fn pre_gate_blocks_purge_before_deleting_c2() {
        let s = store();
        let node = s
            .ingest(vestige_core::IngestInput {
                content: "A memory awaiting destructive review.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let args = Some(serde_json::json!({
            "action": "purge",
            "id": node.id,
            "confirm": true,
            "reason": "test purge"
        }));

        let response = gate_pending_memory_mutation(
            &s,
            None,
            "run_pre_gate",
            "memory",
            &args,
            vestige_core::ReviewMode::RiskGated,
        )
        .unwrap()
        .expect("purge should be pre-gated");

        assert_eq!(response["pendingReview"], serde_json::json!(true));
        assert!(
            s.get_node(&node.id).unwrap().is_some(),
            "pre-gating must not delete before review"
        );
        let pr = s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(pr.len(), 1);
        assert_eq!(pr[0].diff["pendingAction"], serde_json::json!("purge"));
        assert_eq!(pr[0].diff["node"]["deleted"], serde_json::json!(false));
    }

    #[test]
    fn pre_gate_leaves_fast_mode_direct() {
        let s = store();
        let node = s
            .ingest(vestige_core::IngestInput {
                content: "Fast mode purge target.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let args = Some(serde_json::json!({
            "action": "purge",
            "id": node.id,
            "confirm": true
        }));

        assert!(
            gate_pending_memory_mutation(
                &s,
                None,
                "run_fast",
                "memory",
                &args,
                vestige_core::ReviewMode::Fast,
            )
            .unwrap()
            .is_none(),
            "Fast mode should preserve direct execution"
        );
    }

    #[test]
    fn pre_gate_blocks_direct_suppress_before_mutating() {
        let s = store();
        let node = s
            .ingest(vestige_core::IngestInput {
                content: "A memory awaiting suppress review.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let args = Some(serde_json::json!({ "id": node.id, "reason": "test suppress" }));

        let response = gate_pending_memory_mutation(
            &s,
            None,
            "run_suppress",
            "suppress",
            &args,
            vestige_core::ReviewMode::RiskGated,
        )
        .unwrap()
        .expect("suppress should be pre-gated");

        assert_eq!(response["pendingReview"], serde_json::json!(true));
        let kept = s.get_node(&node.id).unwrap().unwrap();
        assert_eq!(kept.suppression_count, 0, "pre-gate must not suppress yet");
        let pr = s
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(pr[0].diff["pendingAction"], serde_json::json!("suppress"));
    }

    #[test]
    fn write_tool_set_includes_codebase_b2() {
        assert!(is_write_tool("codebase"));
        assert!(is_write_tool("memory"));
        assert!(!is_write_tool("search"));
        assert!(!is_write_tool("deep_reference"));
    }
}
