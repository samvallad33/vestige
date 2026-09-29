//! Unified receipt inspection and controlled context-ablation replay.

use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use vestige_core::advanced::git_records;
use vestige_core::advanced::retroactive_backfill::{
    BackfillCandidate, FailureEvent, RetroactiveBackfill,
};
use vestige_core::storage::ReceiptAttestationStatus;
use vestige_core::storage::canonical_walk_json;
use vestige_core::{
    BACKFILL_RECEIPT_CLAIM_BOUNDARY, REPLAY_CLAIM_BOUNDARY, Receipt, ReceiptEvidence,
    ReplayPrivacyState, SYNAPTIC_CAPTURE_CLAIM_BOUNDARY, Storage, WALK_RECEIPT_CLAIM_BOUNDARY,
    WALK_RECEIPT_SCHEMA_V1,
};

const COUNTERFACTUAL_REPLAY_SCHEMA: &str =
    "https://vestige.dev/schemas/receipt/counterfactual-replay/v1";

const LEGACY_RECEIPT_CLAIM_BOUNDARY: &str = concat!(
    "This legacy receipt records retrieval metadata only; ",
    "it carries no controlled-replay or synaptic-capture claim."
);

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "replay", "save_walk"],
                "description": "'get': one receipt with its replay-capsule summary. 'replay': on a Strata log, re-derive state from the log and the receipt and report any mismatch (read-only). Otherwise withhold named evidence slots from a frozen capsule, or re-execute a saved walk receipt (id 'wr_…'). 'save_walk': freeze a backfill parameter envelope as a digest-addressed walk receipt."
            },
            "receipt_id": {
                "type": "string",
                "description": "Receipt id. 'get' and counterfactual 'replay' need a retrieval receipt (with a frozen capsule for replay); walk 'replay' needs a walk receipt id ('wr_…'). Not used by 'save_walk'."
            },
            "withheld_slots": {
                "type": "array",
                "items": { "type": "string", "pattern": "^evidence_[1-9][0-9]*$" },
                "uniqueItems": true,
                "description": "[counterfactual replay] Slots to remove from the frozen context; search is never rerun."
            },
            "remove_edge": {
                "type": "string",
                "description": "[walk replay] Candidate memory id or 'source->target' edge to filter out of the candidate pool before the re-run; the verdict delta between the unfiltered and filtered runs is reported. Never combined with withheld_slots."
            },
            "params": {
                "type": "object",
                "description": "[save_walk] The backfill parameter envelope to freeze (scope, failure_id, lookback_days, scan_limit, manual, …). Canonicalized (RFC 8785) and digested (blake3); the same envelope always yields the same receipt id."
            }
        },
        "required": ["action"],
        "additionalProperties": false,
        "oneOf": [
            {
                "properties": { "action": { "const": "get" } },
                "required": ["receipt_id"],
                "not": { "anyOf": [
                    { "required": ["withheld_slots"] },
                    { "required": ["remove_edge"] },
                    { "required": ["params"] }
                ] }
            },
            {
                "properties": { "action": { "const": "replay" } },
                "required": ["receipt_id"],
                "not": { "anyOf": [
                    { "required": ["params"] },
                    { "allOf": [
                        { "required": ["withheld_slots"] },
                        { "required": ["remove_edge"] }
                    ] }
                ] }
            },
            {
                "properties": { "action": { "const": "save_walk" } },
                "required": ["params"],
                "not": { "anyOf": [
                    { "required": ["receipt_id"] },
                    { "required": ["withheld_slots"] },
                    { "required": ["remove_edge"] }
                ] }
            }
        ]
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptArgs {
    action: String,
    #[serde(alias = "receiptId")]
    receipt_id: Option<String>,
    #[serde(alias = "withheldSlots")]
    withheld_slots: Option<Vec<String>>,
    #[serde(alias = "removeEdge")]
    remove_edge: Option<String>,
    params: Option<Value>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ReceiptArgs =
        serde_json::from_value(args.ok_or_else(|| "receipt requires arguments".to_string())?)
            .map_err(|error| format!("Invalid receipt arguments: {error}"))?;
    validate_args(&args)?;
    match args.action.as_str() {
        "get" => execute_get(storage, args.receipt_id.as_deref().unwrap_or_default()),
        "replay" => {
            let receipt_id = args.receipt_id.as_deref().unwrap_or_default();
            if crate::strata_memory::is_strata_backend(storage.as_ref())
                && !receipt_id.starts_with("wr_")
                && args.remove_edge.is_none()
            {
                if args
                    .withheld_slots
                    .as_ref()
                    .is_some_and(|slots| !slots.is_empty())
                {
                    return Err("withheld_slots do not apply to a Strata log replay".into());
                }
                return execute_strata_replay(storage, receipt_id);
            }
            if storage
                .get_walk_receipt(receipt_id)
                .map_err(|error| safe_storage_error("walk receipt lookup", &error))?
                .is_some()
            {
                execute_walk_replay(storage, receipt_id, args.remove_edge.as_deref())
            } else if receipt_id.starts_with("wr_") {
                Err(format!("Walk receipt '{receipt_id}' was not found"))
            } else if args.remove_edge.is_some() {
                Err("remove_edge is only valid when replaying a walk receipt (id 'wr_…')".into())
            } else {
                execute_replay(
                    storage,
                    receipt_id,
                    args.withheld_slots.as_deref().unwrap_or(&[]),
                )
            }
        }
        "save_walk" => execute_save_walk(storage, args.params.as_ref().unwrap_or(&Value::Null)),
        other => Err(format!(
            "Unknown receipt action '{other}'. Use get|replay|save_walk."
        )),
    }
}

fn validate_args(args: &ReceiptArgs) -> Result<(), String> {
    let receipt_id = args.receipt_id.as_deref().unwrap_or_default();
    match args.action.as_str() {
        "get" => {
            if receipt_id.trim().is_empty() {
                return Err("receipt_id must not be empty".into());
            }
            if args.withheld_slots.is_some() {
                return Err("withheld_slots is only valid for action='replay'".into());
            }
            if args.remove_edge.is_some() {
                return Err(
                    "remove_edge is only valid for action='replay' on a walk receipt".into(),
                );
            }
            if args.params.is_some() {
                return Err("params is only valid for action='save_walk'".into());
            }
        }
        "replay" => {
            if receipt_id.trim().is_empty() {
                return Err("receipt_id must not be empty".into());
            }
            if args.params.is_some() {
                return Err("params is only valid for action='save_walk'".into());
            }
            if args.withheld_slots.is_some() && args.remove_edge.is_some() {
                return Err(
                    "withheld_slots (counterfactual replay) and remove_edge (walk replay) cannot be combined"
                        .into(),
                );
            }
        }
        "save_walk" => {
            if args.receipt_id.is_some() {
                return Err("receipt_id is not valid for action='save_walk'".into());
            }
            if args.withheld_slots.is_some() || args.remove_edge.is_some() {
                return Err(
                    "withheld_slots and remove_edge are not valid for action='save_walk'".into(),
                );
            }
            let Some(params) = &args.params else {
                return Err("save_walk requires a params object".into());
            };
            if !params.is_object() {
                return Err("params must be a JSON object of backfill arguments".into());
            }
        }
        _ => {}
    }
    Ok(())
}

fn claim_boundary_for_receipt(receipt: &Receipt) -> &'static str {
    match &receipt.evidence {
        Some(ReceiptEvidence::CounterfactualReplay { .. }) => REPLAY_CLAIM_BOUNDARY,
        Some(ReceiptEvidence::SynapticCapture(_)) => SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
        Some(ReceiptEvidence::Backfill { .. }) => BACKFILL_RECEIPT_CLAIM_BOUNDARY,
        None => LEGACY_RECEIPT_CLAIM_BOUNDARY,
    }
}

fn safe_storage_error(operation: &str, error: &impl std::fmt::Display) -> String {
    let text = error.to_string();
    if text.contains("pending_strata")
        || text.contains("similarity_disabled")
        || text.contains("verification failed")
        || text.contains("gate_denied")
        || text.contains("gate_held")
    {
        return text;
    }
    tracing::warn!(%error, "receipt storage operation failed: {operation}");
    format!("Receipt {operation} is temporarily unavailable")
}

fn execute_get(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    let receipt = storage
        .get_receipt(receipt_id)
        .map_err(|error| safe_storage_error("lookup", &error))?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    let capsule = storage
        .get_retrieval_replay_capsule(receipt_id)
        .map_err(|error| safe_storage_error("lookup", &error))?;
    let claim_boundary = claim_boundary_for_receipt(&receipt);
    let attestation = receipt_attestation_view(storage, receipt_id)?;
    Ok(json!({
        "action": "get",
        "receipt": receipt,
        "replayCapsule": capsule,
        "claimBoundary": claim_boundary,
        "attestation": attestation,
    }))
}

/// Strata effect receipt: recomputed from the log by `get_receipt` (Allow gate,
/// payload digest, hash chain). Not a DSSE envelope and not an external timestamp.
fn strata_effect_attestation(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    let receipt = storage
        .get_receipt(receipt_id)
        .map_err(|error| safe_storage_error("attestation lookup", &error))?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    let note = receipt
        .mutations
        .first()
        .and_then(|mutation| mutation.note.as_deref())
        .unwrap_or("");
    let digest = note
        .split_whitespace()
        .find_map(|part| part.strip_prefix("digest="))
        .unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "Receipt '{receipt_id}' has no proved payload digest"
        ));
    }
    Ok(json!({
        "status": "strata_effect",
        "verification": {
            "locallyVerified": true,
            "chainValid": true,
            "gateAllowed": true,
            "payloadDigest": digest,
            "claimBoundary": "Recomputed from the hash-chained log: the effect cites an Allow gate and its payload digest matches the admitted frame. A sealed segment trailer signature is checked when one is present. This is not an external timestamp or a truth claim."
        }
    }))
}

/// Present cryptographic receipt state without treating local database row
/// checks as an external timestamp or non-equivocation proof.
fn receipt_attestation_view(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return strata_effect_attestation(storage, receipt_id);
    }
    let status = storage
        .receipt_attestation_status(receipt_id)
        .map_err(|error| safe_storage_error("attestation lookup", &error))?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    if status == ReceiptAttestationStatus::LegacyUnsigned {
        return Ok(json!({
            "status": "legacy_unsigned",
            "verification": {
                "locallyVerified": false,
                "claimBoundary": "No DSSE envelope exists for this legacy receipt."
            }
        }));
    }
    let envelope = storage
        .get_receipt_attestation_envelope(receipt_id)
        .map_err(|error| safe_storage_error("attestation envelope lookup", &error))?
        .ok_or_else(|| "Signed receipt is missing its immutable DSSE envelope".to_string())?;
    let verification = storage
        .verify_stored_receipt_attestation(receipt_id)
        .map_err(|error| safe_storage_error("attestation verification", &error))?
        .ok_or_else(|| "Signed receipt is missing its verifiable DSSE state".to_string())?;
    let locally_verified = verification.is_valid();
    let report = verification.report;
    Ok(json!({
        "status": "signed_v1",
        "envelope": envelope,
        "verification": {
            "locallyVerified": locally_verified,
            "signatureValid": report.signature_valid,
            "canonicalPayload": report.canonical_payload,
            "receiptBindingValid": verification.receipt_binding_valid,
            "localChainRowMatched": report.anchored,
            "predecessorRowMatched": report.predecessor_anchored,
            "terminalHeadRowMatched": report.terminal_head_matched,
            "verifiedKeyId": report.verified_key_id,
            "verifiedPublicKeyFingerprint": report.verified_public_key_fingerprint,
            "keyValidity": report.key_validity.map(|value| format!("{value:?}").to_lowercase()),
            "failures": report.failures.iter().map(|failure| format!("{failure:?}")).collect::<Vec<_>>(),
            "warnings": report.warnings.iter().map(|warning| format!("{warning:?}")).collect::<Vec<_>>(),
            "claimBoundary": "Local verification checks immutable rows and registered keys; it does not establish external anchoring, trusted time, truth, completeness, or non-equivocation."
        }
    }))
}

fn linked_replay_receipt(
    storage: &Arc<Storage>,
    replay_id: &str,
) -> Result<Option<Receipt>, String> {
    let replay = storage
        .get_context_ablation_replay(replay_id)
        .map_err(|error| safe_storage_error("recovery", &error))?;
    let Some(replay) = replay else {
        return Ok(None);
    };
    let Some(receipt_id) = replay.receipt_id else {
        return Ok(None);
    };
    storage
        .get_receipt(&receipt_id)
        .map_err(|error| safe_storage_error("recovery", &error))?
        .ok_or_else(|| "Replay receipt link is incomplete; retry the replay".to_string())
        .map(Some)
}

fn execute_strata_replay(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    match storage.replay_receipt(receipt_id) {
        Ok(report) => Ok(report),
        Err(vestige_core::StorageError::NotFound(message)) => Err(message),
        Err(error) => {
            let owned = error.to_string();
            let text = owned
                .strip_prefix("Database error: ")
                .unwrap_or(owned.as_str());
            if text.contains("mismatch")
                || text.contains("blake3")
                || text.contains("verification failed")
                || text.contains("strata halt")
                || text.contains("frame parse")
                || text.contains("trailer")
                || text.contains("segment header")
            {
                Err(text.to_string())
            } else {
                Err(safe_storage_error("replay", &error))
            }
        }
    }
}

fn execute_replay(
    storage: &Arc<Storage>,
    source_receipt_id: &str,
    withheld_slots: &[String],
) -> Result<Value, String> {
    let durable = storage
        .create_context_ablation_replay(source_receipt_id, withheld_slots)
        .map_err(|error| safe_storage_error("creation", &error))?;
    if durable.replay.privacy_state != ReplayPrivacyState::Active {
        return Err("Replay evidence is no longer available under current privacy state".into());
    }
    let result = durable
        .replay
        .result
        .clone()
        .ok_or_else(|| "Replay evidence is unavailable".to_string())?;

    let (receipt, reused_existing_receipt) = if let Some(receipt_id) = &durable.replay.receipt_id {
        storage
            .get_receipt(receipt_id)
            .map_err(|error| safe_storage_error("lookup", &error))?
            .ok_or_else(|| "Replay receipt link is incomplete; retry the replay".to_string())
            .map(|receipt| (receipt, true))?
    } else {
        let receipt = Receipt::build(
            Utc::now(),
            "replay",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[result.counterfactual.trust_floor],
            Vec::new(),
        )
        .with_evidence(ReceiptEvidence::CounterfactualReplay {
            schema: COUNTERFACTUAL_REPLAY_SCHEMA.to_string(),
            schema_version: 1,
            replay_id: durable.replay.replay_id.clone(),
            capsule_id: durable.replay.capsule_id.clone(),
            result: result.clone(),
        });
        match storage.save_counterfactual_replay_receipt(
            &durable.replay.replay_id,
            &receipt,
            None,
            Some("receipt"),
        ) {
            Ok(()) => (receipt, false),
            Err(error) => {
                tracing::warn!(
                    %error,
                    replay_id = %durable.replay.replay_id,
                    "replay receipt persistence failed; checking for a concurrently linked receipt"
                );
                linked_replay_receipt(storage, &durable.replay.replay_id)?
                    .ok_or_else(|| {
                        "Replay receipt could not be persisted or recovered; retry the replay"
                            .to_string()
                    })
                    .map(|receipt| (receipt, true))?
            }
        }
    };

    Ok(json!({
        "action": "replay",
        "sourceReceiptId": source_receipt_id,
        "replayId": durable.replay.replay_id,
        "receiptId": receipt.receipt_id,
        "reusedExisting": durable.reused_existing || reused_existing_receipt,
        "receipt": receipt,
        "result": result,
        "claimBoundary": REPLAY_CLAIM_BOUNDARY,
    }))
}

// ============================================================================
// WALK RECEIPTS — freeze backfill parameters, re-execute them later
// ============================================================================

/// Freeze a backfill parameter envelope as a digest-addressed walk receipt.
/// Output is byte-stable: every field is derived from the canonical envelope
/// or constants, never from the clock.
fn execute_save_walk(storage: &Arc<Storage>, params: &Value) -> Result<Value, String> {
    let canonical = canonical_walk_json(params)
        .map_err(|error| format!("Walk params could not be canonicalized: {error}"))?;
    let handle = storage
        .save_walk_receipt(&canonical, params)
        .map_err(|error| safe_storage_error("walk receipt save", &error))?;
    let stored_engine_version = storage
        .get_walk_receipt(&handle.receipt_id)
        .map_err(|error| safe_storage_error("walk receipt lookup", &error))?
        .map(|row| row.engine_version)
        .unwrap_or_default();
    Ok(json!({
        "action": "save_walk",
        "schema": WALK_RECEIPT_SCHEMA_V1,
        "receiptId": handle.receipt_id,
        "digest": handle.digest,
        "canonicalParams": canonical,
        "engineVersion": stored_engine_version,
        "reusedExisting": handle.reused_existing,
        "claimBoundary": WALK_RECEIPT_CLAIM_BOUNDARY,
    }))
}

/// The subset of the recorded envelope a walk replay honors. `promote` is
/// deliberately NOT honored: a replay re-derives a verdict for comparison and
/// must never strengthen memories or write edges.
#[derive(Debug)]
struct WalkReplayParams {
    scope: String,
    failure_id: Option<String>,
    manual: bool,
    lookback_days: i64,
    scan_limit: i32,
    promote_recorded: bool,
}

fn walk_replay_params(params: &Value) -> WalkReplayParams {
    WalkReplayParams {
        scope: params
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .trim()
            .to_string(),
        failure_id: params
            .get("failure_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        manual: params
            .get("manual")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        lookback_days: params
            .get("lookback_days")
            .and_then(Value::as_i64)
            .unwrap_or(30)
            .clamp(1, 365),
        scan_limit: params
            .get("scan_limit")
            .and_then(Value::as_i64)
            .unwrap_or(500)
            .clamp(10, 5000) as i32,
        promote_recorded: params
            .get("promote")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

/// Candidate assembly for replay, mirroring `tools::backfill::build_candidates`
/// minus the caller-side git version-range exclusion (a replay re-derives from
/// the live store; recorded git ranges are reported as not re-applied).
/// Supersession is followed exactly like the live tool: a superseded record
/// passes its candidacy to its replacement, dated by the superseded record.
fn assemble_walk_candidates(
    failure_id: &str,
    failure_created: chrono::DateTime<Utc>,
    all: &[vestige_core::KnowledgeNode],
    supersession: &std::collections::HashMap<String, String>,
) -> Vec<BackfillCandidate> {
    let mut candidates: Vec<BackfillCandidate> = Vec::new();
    for node in all {
        if node.id == failure_id || node.suppression_count > 0 {
            continue;
        }
        let mut terminal = node.id.clone();
        for _ in 0..5 {
            match supersession.get(&terminal) {
                Some(next) if next != &terminal => terminal = next.clone(),
                _ => break,
            }
        }
        let (origin, current): (&vestige_core::KnowledgeNode, &vestige_core::KnowledgeNode) =
            if terminal == node.id {
                (node, node)
            } else {
                match all.iter().find(|n| n.id == terminal) {
                    Some(replacement) => (node, replacement),
                    None => continue,
                }
            };
        if current.id == failure_id {
            continue;
        }
        let age = (failure_created - origin.created_at).num_seconds() as f64 / 86_400.0;
        let mut entities = vestige_core::advanced::retroactive_backfill::extract_entities(
            &current.content,
            &current.tags,
        );
        if current.id != origin.id {
            for entity in vestige_core::advanced::retroactive_backfill::extract_entities(
                &origin.content,
                &origin.tags,
            ) {
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
        }
        let is_commit = current
            .tags
            .iter()
            .chain(origin.tags.iter())
            .any(|tag| tag == git_records::COMMIT_TAG);
        let candidate = BackfillCandidate {
            id: current.id.clone(),
            content: current.content.clone(),
            entities,
            age_days_before_failure: age,
            stability: current.stability,
            via_supersession_of: if current.id != origin.id {
                Some(origin.id.clone())
            } else {
                None
            },
            is_change_record: is_commit,
        };
        match candidates.iter_mut().find(|c| c.id == candidate.id) {
            Some(existing) if candidate.via_supersession_of.is_some() => *existing = candidate,
            Some(_) => {}
            None => candidates.push(candidate),
        }
    }
    candidates
}

/// Deterministic verdict summary: ids and rounded scores only, so two runs on
/// an unchanged store serialize to identical bytes.
fn walk_verdict_summary(
    result: &vestige_core::advanced::retroactive_backfill::BackfillResult,
) -> Value {
    json!({
        "triggered": result.triggered,
        "failureId": result.failure_id,
        "causeCount": result.causes.len(),
        "causeIds": result.causes.iter().map(|c| c.memory_id.clone()).collect::<Vec<_>>(),
        "topCauseId": result.causes.first().map(|c| c.memory_id.clone()),
        "topCauseScore": result
            .causes
            .first()
            .map(|c| (c.score * 100.0).round() / 100.0),
        "scanned": result.scanned,
    })
}

/// Re-execute a saved walk receipt's parameters against the current store.
/// Pure preview: no edges are written, no memory is reinforced (a recorded
/// `promote=true` is reported, never applied). When `remove_edge` names a
/// candidate id or a `source->target` edge, that candidate is filtered from
/// the pool before the second run and the verdict delta between the two runs
/// is reported.
fn execute_walk_replay(
    storage: &Arc<Storage>,
    receipt_id: &str,
    remove_edge: Option<&str>,
) -> Result<Value, String> {
    let walk = storage
        .get_walk_receipt(receipt_id)
        .map_err(|error| safe_storage_error("walk receipt lookup", &error))?
        .ok_or_else(|| format!("Walk receipt '{receipt_id}' was not found"))?;
    let params = walk_replay_params(&walk.params);
    if params.scope.is_empty() {
        return Err("recorded walk scope must not be empty".into());
    }

    // 1. Resolve the failure event with the same rules as the live tool.
    let failure_node = match params.failure_id.as_deref() {
        Some(id) => {
            if !storage
                .node_is_in_scope(id, &params.scope)
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "recorded failure memory '{id}' is no longer in scope '{}'",
                    params.scope
                ));
            }
            storage
                .get_node(id)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "recorded failure memory no longer exists".to_string())?
        }
        None => {
            let recent = storage
                .get_all_nodes_in_scope(&params.scope, params.scan_limit, 0)
                .map_err(|e| e.to_string())?;
            recent
                .iter()
                .find(|node| super::backfill::looks_like_failure(node))
                .cloned()
                .ok_or_else(|| {
                    "no failure-like memory remains in the recorded scope; the walk cannot be replayed"
                        .to_string()
                })?
        }
    };

    let mut failure_entities = vestige_core::advanced::retroactive_backfill::extract_entities(
        &failure_node.content,
        &failure_node.tags,
    );
    // Entity extraction iterates a hash set, so its order varies call to
    // call. Sorting here keeps replay output byte-stable; matching is by
    // membership, never by order.
    failure_entities.sort();
    failure_entities.dedup();
    let failure = FailureEvent {
        id: failure_node.id.clone(),
        content: failure_node.content.clone(),
        entities: failure_entities.clone(),
        tags: failure_node.tags.clone(),
        prediction_error: if super::backfill::looks_like_failure(&failure_node) {
            0.9_f32
        } else {
            0.3_f32
        },
        manual: params.manual,
    };

    // 2. Rebuild the candidate pool from the CURRENT store.
    let all = storage
        .get_all_nodes_in_scope(&params.scope, params.scan_limit, 0)
        .map_err(|e| e.to_string())?;
    let supersession: std::collections::HashMap<String, String> = storage
        .supersession_pairs()
        .map_err(|e| e.to_string())?
        .into_iter()
        .collect();
    let candidates = assemble_walk_candidates(
        &failure_node.id,
        failure_node.created_at,
        &all,
        &supersession,
    );

    // 3. Apply the remove_edge filter for the counterfactual run.
    let (edge_source, edge_target) = match remove_edge.map(|edge| edge.split_once("->")) {
        Some(Some((source, target))) => (
            Some(source.trim().to_string()),
            Some(target.trim().to_string()),
        ),
        _ => (None, None),
    };
    let filter_matches = |candidate: &BackfillCandidate| -> bool {
        match (&edge_source, &edge_target) {
            (Some(source), Some(target)) => &candidate.id == source || &candidate.id == target,
            _ => candidate.id == remove_edge.unwrap_or_default().trim(),
        }
    };
    let removed_candidate_ids: Vec<String> = candidates
        .iter()
        .filter(|candidate| filter_matches(candidate))
        .map(|candidate| candidate.id.clone())
        .collect();
    let ablated_candidates: Vec<BackfillCandidate> = candidates
        .iter()
        .filter(|candidate| !filter_matches(candidate))
        .cloned()
        .collect();

    // 4. Run the backward reach twice: unfiltered baseline, then filtered.
    let backfill = RetroactiveBackfill {
        lookback_days: params.lookback_days,
        ..RetroactiveBackfill::new()
    };
    let baseline = backfill.run_trail(&failure, &candidates, &[]);
    let ablated = backfill.run_trail(&failure, &ablated_candidates, &[]);
    // Invariant both runs must uphold: the same failure event was evaluated.
    if baseline.failure_id != ablated.failure_id || baseline.failure_id != failure_node.id {
        return Err("walk replay invariant violated: runs evaluated different failures".into());
    }

    let baseline_summary = walk_verdict_summary(&baseline);
    let ablated_summary = walk_verdict_summary(&ablated);
    let verdict_changed = baseline_summary["triggered"] != ablated_summary["triggered"]
        || baseline_summary["causeIds"] != ablated_summary["causeIds"];

    let filter_applied = remove_edge.is_some_and(|edge| !edge.trim().is_empty());
    Ok(json!({
        "action": "replay",
        "kind": "walk",
        "schema": WALK_RECEIPT_SCHEMA_V1,
        "receiptId": walk.receipt_id,
        "digest": walk.digest,
        "params": walk.params,
        "recordedEngineVersion": walk.engine_version,
        "replayMode": "preview",
        "promoteRecordedButNotApplied": params.promote_recorded,
        "ignoredRecordedFields": ["promote", "git_repo", "worked_in", "broke_in", "why_not"],
        "failure": {
            "id": failure_node.id,
            "contentPreview": failure_node.content.chars().take(160).collect::<String>(),
            "entities": failure_entities,
        },
        "lookbackDays": params.lookback_days,
        "scanLimit": params.scan_limit,
        "candidateCount": candidates.len(),
        "filter": {
            "removeEdge": remove_edge,
            "removedCandidateIds": removed_candidate_ids,
            "applied": filter_applied,
        },
        "verdict": baseline_summary,
        "verdictDelta": {
            "filterApplied": filter_applied,
            "verdictChanged": verdict_changed,
            "baseline": baseline_summary,
            "ablated": ablated_summary,
            "note": "Verdict delta compares candidate rankings between the unfiltered and filtered re-runs on the current store. It is evidence about the ranking, not a causal claim.",
        },
        "claimBoundary": WALK_RECEIPT_CLAIM_BOUNDARY,
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    fn test_storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("receipt-tool.db"))).unwrap();
        (storage, dir)
    }

    #[tokio::test]
    async fn legacy_receipt_gets_but_does_not_replay() {
        let (storage, _dir) = test_storage();
        let receipt = Receipt::build(
            Utc::now(),
            "legacy",
            vec!["memory_legacy".into()],
            Vec::new(),
            Vec::new(),
            &[0.8],
            Vec::new(),
        );
        storage
            .save_receipt(&receipt, None, Some("recall"), None)
            .unwrap();

        let get = execute(
            &storage,
            Some(json!({"action": "get", "receipt_id": receipt.receipt_id})),
        )
        .await
        .unwrap();
        assert!(get["replayCapsule"].is_null());
        assert_eq!(
            get["claimBoundary"], LEGACY_RECEIPT_CLAIM_BOUNDARY,
            "legacy receipts must not inherit the replay claim"
        );
        assert!(
            execute(
                &storage,
                Some(json!({
                    "action": "replay",
                    "receipt_id": receipt.receipt_id,
                    "withheld_slots": ["evidence_1"]
                })),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn get_rejects_replay_only_arguments() {
        let (storage, _dir) = test_storage();
        let error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "withheld_slots": ["evidence_1"]
            })),
        )
        .await
        .unwrap_err();
        assert_eq!(error, "withheld_slots is only valid for action='replay'");

        let empty_error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "withheld_slots": []
            })),
        )
        .await
        .unwrap_err();
        assert_eq!(
            empty_error,
            "withheld_slots is only valid for action='replay'"
        );
    }

    #[tokio::test]
    async fn receipt_arguments_reject_unknown_fields() {
        let (storage, _dir) = test_storage();
        let error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "unexpected": true
            })),
        )
        .await
        .unwrap_err();
        assert!(error.contains("unknown field `unexpected`"));
    }

    #[test]
    fn claim_boundary_tracks_typed_receipt_evidence() {
        let legacy = Receipt::build(
            Utc::now(),
            "legacy",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            Vec::new(),
        );
        assert_eq!(
            claim_boundary_for_receipt(&legacy),
            LEGACY_RECEIPT_CLAIM_BOUNDARY
        );

        let synaptic = legacy.with_evidence(ReceiptEvidence::SynapticCapture(
            vestige_core::SynapticCaptureEvidence {
                schema: "https://vestige.dev/schemas/receipt/synaptic-capture/v1".into(),
                schema_version: 1,
                algorithm_version: "test".into(),
                receipt_role: None,
                parent_receipt_id: None,
                evaluation_direction: None,
                trigger: vestige_core::SynapticCaptureTrigger {
                    event_id: "event_test".into(),
                    memory_id: "memory_test".into(),
                    event_type: "test".into(),
                    occurred_at: Utc::now(),
                    importance_score: 0.9,
                },
                capture_window: vestige_core::SynapticCaptureWindow {
                    evaluation_direction: "backward".into(),
                    backward_hours: 1.0,
                    forward_hours: 0.0,
                    tag_lifetime_hours: 1.0,
                    minimum_tag_strength: 0.0,
                    minimum_association_score: None,
                    maximum_captures: 1,
                    decay_function: "exponential".into(),
                    context_threshold: None,
                    context_algorithm_version: None,
                },
                candidates: Vec::new(),
                claim_boundary: "untrusted caller text is ignored".into(),
            },
        ));
        assert_eq!(
            claim_boundary_for_receipt(&synaptic),
            SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
            "the public boundary must be the canonical typed-evidence boundary"
        );
    }

    #[tokio::test]
    async fn recorded_recall_replays_idempotently_with_typed_noncausal_receipt() {
        let (storage, dir) = test_storage();
        let recall_result = json!({
            "method": "hybrid+cognitive",
            "retrievalMode": "balanced",
            "tokenBudgetLimit": 256,
            "results": [
                {
                    "id": "memory_product_a",
                    "content": "private replay sentinel alpha",
                    "retentionStrength": 0.81
                },
                {
                    "id": "memory_product_b",
                    "content": "private replay sentinel beta",
                    "retentionStrength": 0.55
                }
            ],
            "expandable": ["memory_not_returned"]
        });
        let source = crate::trace_recorder::build_and_save_receipt(
            &storage,
            "run_receipt_product",
            "recall",
            &recall_result,
            None,
        )
        .expect("recall should atomically persist receipt and final capsule");
        let source_receipt_id = source["receipt_id"].as_str().unwrap();

        let args = Some(json!({
            "action": "replay",
            "receipt_id": source_receipt_id,
            "withheld_slots": ["evidence_2"]
        }));
        let first = execute(&storage, args.clone()).await.unwrap();
        assert_eq!(first["result"]["baseline"]["itemCount"], 2);
        assert_eq!(first["result"]["counterfactual"]["itemCount"], 1);
        assert_eq!(
            first["result"]["counterfactual"]["orderedSlots"],
            json!(["evidence_1"])
        );
        assert_eq!(first["claimBoundary"], REPLAY_CLAIM_BOUNDARY);
        assert_eq!(
            first["receipt"]["evidence"]["kind"],
            "counterfactual_replay"
        );
        assert_eq!(
            first["receipt"]["evidence"]["predicate"]["result"]["claimBoundary"],
            REPLAY_CLAIM_BOUNDARY
        );

        let public_json = serde_json::to_string(&first).unwrap();
        for forbidden in [
            "private replay sentinel alpha",
            "private replay sentinel beta",
            "memory_product_a",
            "memory_product_b",
            "memory_not_returned",
            "b3k:",
        ] {
            assert!(
                !public_json.contains(forbidden),
                "replay leaked {forbidden}"
            );
        }

        let second = execute(&storage, args).await.unwrap();
        assert_eq!(second["reusedExisting"], true);
        assert_eq!(second["replayId"], first["replayId"]);
        assert_eq!(second["receiptId"], first["receiptId"]);
        let reader = rusqlite::Connection::open(dir.path().join("receipt-tool.db")).unwrap();
        let replay_rows: i64 = reader
            .query_row("SELECT COUNT(*) FROM counterfactual_replays", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(replay_rows, 1);

        let replay_receipt = execute(
            &storage,
            Some(json!({"action": "get", "receipt_id": first["receiptId"]})),
        )
        .await
        .unwrap();
        assert_eq!(replay_receipt["claimBoundary"], REPLAY_CLAIM_BOUNDARY);
    }

    #[test]
    fn schema_exposes_only_get_and_controlled_replay() {
        let actions = schema()["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(
            actions,
            vec![json!("get"), json!("replay"), json!("save_walk")]
        );
        assert!(
            schema()["properties"]["withheld_slots"]["description"]
                .as_str()
                .unwrap()
                .contains("never rerun")
        );
        assert_eq!(schema()["additionalProperties"], false);
    }

    // ---- walk receipts: save, determinism, ablation ----

    /// Seed the exact three-memory scenario from the backfill tool's live
    /// test: a quiet env-var cause, a semantic distractor, a failure.
    fn seeded_walk_store() -> (
        Arc<Storage>,
        vestige_core::KnowledgeNode,
        vestige_core::KnowledgeNode,
    ) {
        let (storage, _dir) = test_storage();
        let cause = storage
            .ingest(vestige_core::IngestInput {
                content: "Set API_TIMEOUT=2 in the deploy env to speed up cold starts".to_string(),
                node_type: "decision".to_string(),
                tags: vec!["API_TIMEOUT".to_string(), "deploy-env".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&cause.id, chrono::Utc::now() - chrono::Duration::days(3))
            .unwrap();
        let distractor = storage
            .ingest(vestige_core::IngestInput {
                content: "A 500 Internal Server Error happened in the billing service last month"
                    .to_string(),
                node_type: "event".to_string(),
                tags: vec!["billing-service".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(
                &distractor.id,
                chrono::Utc::now() - chrono::Duration::days(20),
            )
            .unwrap();
        let failure = storage
            .ingest(vestige_core::IngestInput {
                content: "Service crashed: 500 Internal Server Error on the auth endpoint"
                    .to_string(),
                node_type: "event".to_string(),
                tags: vec![
                    "auth-service".to_string(),
                    "API_TIMEOUT".to_string(),
                    "crash".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();
        (storage, cause, failure)
    }

    #[tokio::test]
    async fn save_walk_is_digest_stable_across_key_order_and_idempotent() {
        let (storage, _cause, _failure) = seeded_walk_store();
        let params = json!({
            "failure_id": "explicit",
            "lookback_days": 30,
            "scope": "user",
            "manual": false
        });
        let first = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": params})),
        )
        .await
        .unwrap();
        assert_eq!(first["action"], "save_walk");
        assert_eq!(first["reusedExisting"], false);
        assert!(first["receiptId"].as_str().unwrap().starts_with("wr_"));
        // Canonical echo: sorted keys, no insignificant whitespace.
        assert!(
            first["canonicalParams"]
                .as_str()
                .unwrap()
                .starts_with(r#"{"failure_id":"explicit","lookback_days":30,"manual":"#)
        );

        // Same envelope, shuffled keys: same receipt, same digest, reused.
        let shuffled = json!({
            "manual": false,
            "scope": "user",
            "lookback_days": 30,
            "failure_id": "explicit"
        });
        let second = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": shuffled})),
        )
        .await
        .unwrap();
        assert_eq!(second["receiptId"], first["receiptId"]);
        assert_eq!(second["digest"], first["digest"]);
        assert_eq!(second["canonicalParams"], first["canonicalParams"]);
        assert_eq!(second["reusedExisting"], true);

        // A different envelope must not collapse onto the same row.
        let different = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": {"failure_id": "other", "lookback_days": 30, "manual": false, "scope": "user"}})),
        )
        .await
        .unwrap();
        assert_ne!(different["receiptId"], first["receiptId"]);

        // Argument policing.
        assert_eq!(
            execute(&storage, Some(json!({"action": "save_walk"})))
                .await
                .unwrap_err(),
            "save_walk requires a params object"
        );
        assert_eq!(
            execute(
                &storage,
                Some(json!({"action": "save_walk", "params": json!({"scope": "user"}), "receipt_id": "r_x"})),
            )
            .await
            .unwrap_err(),
            "receipt_id is not valid for action='save_walk'"
        );
    }

    #[tokio::test]
    async fn walk_replay_is_deterministic_on_a_seeded_store() {
        let (storage, cause, failure) = seeded_walk_store();
        let saved = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": {
                "failure_id": failure.id, "lookback_days": 30, "scan_limit": 500
            }})),
        )
        .await
        .unwrap();
        let receipt_id = saved["receiptId"].as_str().unwrap().to_string();

        let args = Some(json!({"action": "replay", "receipt_id": receipt_id}));
        let first = execute(&storage, args.clone()).await.unwrap();
        let second = execute(&storage, args).await.unwrap();
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap(),
            "walk replay on an unchanged store must be byte-identical"
        );

        assert_eq!(first["kind"], "walk");
        assert_eq!(first["verdict"]["triggered"], true);
        assert_eq!(first["verdict"]["failureId"], json!(failure.id));
        assert!(
            !first["verdict"]["causeIds"].as_array().unwrap().is_empty(),
            "the seeded failure must surface candidates: {}",
            first["verdict"]["causeIds"]
        );
        assert_eq!(first["verdictDelta"]["filterApplied"], false);
        assert_eq!(first["verdictDelta"]["verdictChanged"], false);
        assert_eq!(first["replayMode"], "preview");

        // The replay is a pure preview: no edges, no promotion.
        assert!(
            storage
                .get_connections_for_memory(&cause.id)
                .unwrap()
                .is_empty()
        );
        let node = storage.get_node(&cause.id).unwrap().unwrap();
        assert_eq!(node.reps, 0);

        // Promote recorded in the envelope is reported, never applied.
        let saved_promote = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": {
                "failure_id": failure.id, "promote": true
            }})),
        )
        .await
        .unwrap();
        let replay = execute(
            &storage,
            Some(json!({
                "action": "replay",
                "receipt_id": saved_promote["receiptId"]
            })),
        )
        .await
        .unwrap();
        assert_eq!(replay["promoteRecordedButNotApplied"], true);
        assert!(
            storage
                .get_connections_for_memory(&cause.id)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn walk_replay_remove_edge_reports_verdict_delta() {
        let (storage, _cause, failure) = seeded_walk_store();
        let saved = execute(
            &storage,
            Some(json!({"action": "save_walk", "params": {
                "failure_id": failure.id, "lookback_days": 30
            }})),
        )
        .await
        .unwrap();
        let receipt_id = saved["receiptId"].as_str().unwrap().to_string();

        // Rank-agnostic: ablate whichever candidate the CURRENT scorer ranks
        // first (the live tool's own live test pins the exact ordering; a
        // replay test must not duplicate that pin and drift with it).
        let baseline_run = execute(
            &storage,
            Some(json!({"action": "replay", "receipt_id": receipt_id})),
        )
        .await
        .unwrap();
        let top = baseline_run["verdict"]["topCauseId"]
            .as_str()
            .expect("a top cause must surface")
            .to_string();

        // Removing the top candidate by memory id changes the verdict.
        let ablated = execute(
            &storage,
            Some(json!({
                "action": "replay",
                "receipt_id": receipt_id,
                "remove_edge": top
            })),
        )
        .await
        .unwrap();
        assert_eq!(ablated["filter"]["applied"], true);
        assert_eq!(ablated["filter"]["removedCandidateIds"], json!([top]));
        assert_eq!(ablated["verdictDelta"]["filterApplied"], true);
        assert_eq!(ablated["verdictDelta"]["verdictChanged"], true);
        assert_eq!(
            ablated["verdictDelta"]["baseline"]["topCauseId"],
            json!(top)
        );
        assert!(
            !ablated["verdictDelta"]["ablated"]["causeIds"]
                .as_array()
                .unwrap()
                .contains(&json!(top)),
            "the removed candidate must not surface in the ablated run"
        );
        assert_eq!(ablated["claimBoundary"], WALK_RECEIPT_CLAIM_BOUNDARY);

        // Edge form 'source->target' filters the same candidate (either side).
        let edge_form = format!("{}->{}", top, failure.id);
        let ablated_edge = execute(
            &storage,
            Some(json!({
                "action": "replay",
                "receipt_id": receipt_id,
                "remove_edge": edge_form
            })),
        )
        .await
        .unwrap();
        assert_eq!(ablated_edge["filter"]["removedCandidateIds"], json!([top]));
        assert_eq!(ablated_edge["verdictDelta"]["verdictChanged"], true);

        // An edge naming unknown ids filters nothing: no verdict change, and
        // the delta says so rather than inventing an ablation.
        let unchanged = execute(
            &storage,
            Some(json!({
                "action": "replay",
                "receipt_id": receipt_id,
                "remove_edge": "mem_unknown->mem_also_unknown"
            })),
        )
        .await
        .unwrap();
        assert_eq!(unchanged["filter"]["removedCandidateIds"], json!([]));
        assert_eq!(unchanged["verdictDelta"]["verdictChanged"], false);

        // Mixed counterfactual/walk arguments are rejected up front.
        assert_eq!(
            execute(
                &storage,
                Some(json!({
                    "action": "replay",
                    "receipt_id": receipt_id,
                    "withheld_slots": ["evidence_1"],
                    "remove_edge": top
                })),
            )
            .await
            .unwrap_err(),
            "withheld_slots (counterfactual replay) and remove_edge (walk replay) cannot be combined"
        );

        // remove_edge against a NON-walk receipt id is rejected.
        assert_eq!(
            execute(
                &storage,
                Some(json!({"action": "replay", "receipt_id": "r_missing", "remove_edge": "x"})),
            )
            .await
            .unwrap_err(),
            "remove_edge is only valid when replaying a walk receipt (id 'wr_…')"
        );
    }
}

#[cfg(test)]
mod strata_replay {
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::json;
    use vestige_core::{IngestInput, Storage};

    use super::execute;

    fn open() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<Storage> =
            Arc::new(crate::strata_memory::StrataMemory::open(dir.path()).unwrap());
        (storage, dir)
    }

    fn ingest(storage: &Arc<Storage>, content: &str) -> vestige_core::KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.into(),
                ..IngestInput::default()
            })
            .unwrap()
    }

    /// blake3 over the log directory's file bytes, in path order.
    fn log_digest(dir: &Path) -> String {
        let log = dir.join("log");
        let mut files = Vec::new();
        let mut stack = vec![log.clone()];
        while let Some(path) = stack.pop() {
            let mut entries: Vec<_> = std::fs::read_dir(&path)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let rel = path
                        .strip_prefix(&log)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    files.push((rel, std::fs::read(&path).unwrap()));
                }
            }
        }
        files.sort();
        let mut hasher = blake3::Hasher::new();
        for (name, bytes) in files {
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        hasher.finalize().to_hex().to_string()
    }

    fn flip_segment(dir: &Path) {
        let log = dir.join("log");
        let mut segs: Vec<_> = std::fs::read_dir(&log)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
            .collect();
        segs.sort();
        let path = segs.first().expect("segment");
        let mut bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() > 8, "segment too short to tamper");
        let at = bytes.len() / 2;
        bytes[at] ^= 0xff;
        std::fs::write(path, bytes).unwrap();
    }

    #[tokio::test]
    async fn valid_replay_matches_and_does_not_write_the_log() {
        let (storage, dir) = open();
        let node = ingest(&storage, "strata replay fixture");
        let receipt = storage.get_receipt(&node.id).unwrap().unwrap();
        let before = log_digest(dir.path());

        let by_node = execute(
            &storage,
            Some(json!({
                "action": "replay",
                "receipt_id": node.id,
                "withheld_slots": []
            })),
        )
        .await
        .unwrap();
        let by_receipt = execute(
            &storage,
            Some(json!({"action": "replay", "receipt_id": receipt.receipt_id})),
        )
        .await
        .unwrap();
        let again = execute(
            &storage,
            Some(json!({"action": "replay", "receipt_id": node.id})),
        )
        .await
        .unwrap();

        assert_eq!(before, log_digest(dir.path()), "replay wrote the log");
        assert_eq!(by_node, again, "replay is not deterministic");
        assert_eq!(by_node["action"], "replay");
        assert_eq!(by_node["kind"], "strata");
        assert_eq!(by_node["readOnly"], true);
        assert_eq!(by_node["matched"], true);
        assert_eq!(by_node["mismatches"], json!([]));
        assert_eq!(by_node["nodeId"], node.id);
        assert_eq!(by_node["receiptId"], receipt.receipt_id);
        assert_eq!(by_node["stateDigest"], by_node["replayedDigest"]);
        assert!(by_node["frames"].as_u64().unwrap() > 0);
        assert_eq!(by_receipt["matched"], true);
        assert_eq!(by_receipt["stateDigest"], by_node["stateDigest"]);
        assert_eq!(by_receipt["receiptId"], receipt.receipt_id);
    }

    #[tokio::test]
    async fn tampered_log_fails_and_replay_does_not_repair_it() {
        let (storage, dir) = open();
        let node = ingest(&storage, "strata tamper fixture");
        flip_segment(dir.path());
        let before = log_digest(dir.path());
        let err = execute(
            &storage,
            Some(json!({"action": "replay", "receipt_id": node.id})),
        )
        .await
        .unwrap_err();
        assert_eq!(
            before,
            log_digest(dir.path()),
            "failed replay mutated the log"
        );
        assert!(
            err.contains("mismatch")
                || err.contains("blake3")
                || err.contains("frame parse")
                || err.contains("trailer")
                || err.contains("segment header")
                || err.contains("strata halt"),
            "{err}"
        );
        assert!(!err.starts_with("Database error"), "{err}");
        assert!(!err.contains("pending_strata"), "{err}");
        assert!(!err.contains("temporarily unavailable"), "{err}");
    }

    #[tokio::test]
    async fn unknown_receipt_is_rejected() {
        let (storage, dir) = open();
        let _node = ingest(&storage, "strata unknown fixture");
        let before = log_digest(dir.path());
        let err = execute(
            &storage,
            Some(json!({"action": "replay", "receipt_id": "no-such-receipt"})),
        )
        .await
        .unwrap_err();
        assert_eq!(before, log_digest(dir.path()));
        assert!(err.contains("no-such-receipt"), "{err}");
        assert!(err.contains("not found"), "{err}");
    }
}
