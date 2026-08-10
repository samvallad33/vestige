//! Durable synaptic tag-and-capture persistence.
//!
//! The in-process neuroscience engine remains a fast projection, but this
//! module is the source of truth for capture eligibility and mutation. One
//! SQLite writer transaction records the event, evaluates every candidate,
//! applies guarded promotions, consumes captured tags, writes candidate rows,
//! and inserts the complete typed receipt payload.

use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use uuid::Uuid;

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};
use crate::neuroscience::{CaptureWindow, DecayFunction, SynapticTag};
use crate::trace::{
    Receipt, ReceiptEvidence, ReceiptMutation, StrengthDelta, SuppressReason,
    SuppressedReceiptEntry, SynapticCaptureCandidate, SynapticCaptureDisposition,
    SynapticCaptureEvidence, SynapticCaptureTrigger, SynapticCaptureWindow, SynapticStrengthChange,
};

pub const SYNAPTIC_CAPTURE_ALGORITHM_V1: &str = "vestige.synaptic_capture.v1";
pub const SYNAPTIC_CAPTURE_SCHEMA_V1: &str =
    "https://vestige.dev/schemas/receipt/synaptic-capture/v1";
pub const SYNAPTIC_CAPTURE_CLAIM_BOUNDARY: &str = "Evidence-backed temporal association with a measured memory-state change; not proof that the trigger caused the earlier memory or a downstream outcome.";

/// Frozen scoring policy supplied by the cognitive engine.
#[derive(Debug, Clone)]
pub struct SynapticCapturePolicy {
    pub backward_hours: f64,
    pub forward_hours: f64,
    pub tag_lifetime_hours: f64,
    pub minimum_tag_strength: f64,
    pub maximum_captures: usize,
    pub decay_function: DecayFunction,
}

/// One durable importance-event evaluation request.
#[derive(Debug, Clone)]
pub struct SynapticCaptureRequest {
    pub trigger_memory_id: String,
    pub event_type: String,
    pub occurred_at: DateTime<Utc>,
    pub strength: f64,
    pub policy: SynapticCapturePolicy,
}

/// Observable result of a committed capture transaction.
#[derive(Debug, Clone)]
pub struct DurableSynapticCapture {
    pub event_id: String,
    pub receipt: Receipt,
    pub captured_count: usize,
    pub reused_existing: bool,
}

#[derive(Debug)]
struct CandidateRow {
    tag_id: String,
    memory_id: String,
    encoded_at_ms: i64,
    retrieval_before: f64,
    retention_before: f64,
    stability_before: f64,
    suppression_count: i64,
    currently_valid: bool,
    temporal_distance_hours: f64,
    capture_probability: f64,
    tag_strength: f64,
    capture_score: f64,
}

impl SqliteMemoryStore {
    /// Persist a tag without resetting an already-recorded tag episode.
    /// Returns the deterministic tag id.
    pub fn save_synaptic_tag(&self, tag: &SynapticTag) -> Result<String> {
        let created_at_ms = tag.created_at.timestamp_millis();
        let tag_id = deterministic_id("stag", &format!("{}|{}", tag.memory_id, created_at_ms));
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let transaction_at_ms = Utc::now().timestamp_millis();
        let eligible: Option<i64> = tx
            .query_row(
                "SELECT CASE
                    WHEN suppression_count = 0
                     AND superseded_by IS NULL
                     AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?2)
                     AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?2)
                     AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?3)
                    THEN 1 ELSE 0 END
                 FROM knowledge_nodes WHERE id = ?1",
                params![tag.memory_id, created_at_ms, transaction_at_ms],
                |row| row.get(0),
            )
            .optional()?;
        if eligible != Some(1) {
            return Err(StorageError::Init(
                "synaptic tag memory is missing, suppressed, superseded, or not currently valid"
                    .into(),
            ));
        }
        // A memory can have many historical tag episodes but only one active
        // episode. Excluding this exact deterministic tag keeps retries
        // idempotent instead of expiring the row they are retrying.
        tx.execute(
            "UPDATE synaptic_tags SET state = 'expired'
             WHERE memory_id = ?1 AND state = 'active' AND tag_id <> ?2",
            params![tag.memory_id, tag_id],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO synaptic_tags
                 (tag_id, memory_id, created_at_ms, initial_strength,
                  encoding_context, algorithm_version, state, capture_event_id,
                  captured_at_ms, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', NULL, NULL, ?7)",
            params![
                tag_id,
                tag.memory_id,
                created_at_ms,
                tag.initial_strength,
                tag.encoding_context,
                SYNAPTIC_CAPTURE_ALGORITHM_V1,
                tag.created_at.to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        Ok(tag_id)
    }

    /// Load the current, unsuppressed, bitemporally-valid tag projection.
    pub fn load_active_synaptic_tags(&self) -> Result<Vec<SynapticTag>> {
        let now_ms = Utc::now().timestamp_millis();
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT t.memory_id, t.created_at_ms, t.initial_strength,
                    t.encoding_context, t.capture_event_id, t.captured_at_ms
             FROM synaptic_tags t
             JOIN knowledge_nodes n ON n.id = t.memory_id
             WHERE t.state = 'active'
               AND n.suppression_count = 0
               AND n.superseded_by IS NULL
               AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?1)
             ORDER BY t.created_at_ms ASC, t.tag_id ASC",
        )?;
        let rows = stmt.query_map(params![now_ms], |row| {
            let created_at_ms: i64 = row.get(1)?;
            let captured_at_ms: Option<i64> = row.get(5)?;
            Ok(SynapticTag {
                memory_id: row.get(0)?,
                created_at: millis_to_datetime(created_at_ms),
                tag_strength: row.get(2)?,
                initial_strength: row.get(2)?,
                captured: false,
                capture_event: row.get(4)?,
                captured_at: captured_at_ms.map(millis_to_datetime),
                encoding_context: row.get(3)?,
            })
        })?;
        let mut tags = Vec::new();
        for row in rows {
            tags.push(row?);
        }
        Ok(tags)
    }

    /// Evaluate an importance event and commit its complete decision atomically.
    /// A retry with the same trigger/timestamp/policy returns the existing
    /// receipt and never promotes a memory twice.
    pub fn capture_synaptic_event(
        &self,
        request: &SynapticCaptureRequest,
    ) -> Result<DurableSynapticCapture> {
        let occurred_at_ms = request.occurred_at.timestamp_millis();
        let window_from_ms = occurred_at_ms
            .checked_sub(hours_to_millis(request.policy.backward_hours).unwrap_or(i64::MAX))
            .unwrap_or(i64::MIN);
        // V1 is deliberately backward-only. Persisting a non-zero forward
        // window before tag-arrival matching exists would make the receipt
        // claim an eligibility path the implementation did not evaluate.
        let evaluated_forward_hours = 0.0_f64;
        let window_to_ms = occurred_at_ms;
        let event_strength = if request.strength.is_finite() {
            request.strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let event_key = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}",
            SYNAPTIC_CAPTURE_ALGORITHM_V1,
            request.trigger_memory_id,
            request.event_type,
            occurred_at_ms,
            event_strength.to_bits(),
            request.policy.backward_hours.to_bits(),
            evaluated_forward_hours.to_bits(),
            request.policy.tag_lifetime_hours.to_bits(),
            request.policy.minimum_tag_strength.to_bits(),
            request.policy.maximum_captures,
            request.policy.decay_function,
        );
        // This deterministic fingerprint is private idempotency state. It is
        // deliberately never returned or embedded in a public receipt because
        // it can be recomputed by anyone who knows the trigger memory id.
        let internal_event_id = deterministic_id("sevt_fp", &event_key);
        // Semantic mutation timestamps come from the frozen event, so replaying
        // the same request from the same snapshot yields the same state.
        let recorded_at = request.occurred_at;

        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Sample current validity only after this writer owns the SQLite
        // transaction. A capture queued behind another writer must observe any
        // suppression/tombstone that writer committed while it was waiting.
        let transaction_at_ms = Utc::now().timestamp_millis();

        let existing_payload: Option<String> = tx
            .query_row(
                "SELECT r.payload
                 FROM synaptic_events e
                 JOIN memory_receipts r ON r.receipt_id = e.receipt_id
                 WHERE e.event_id = ?1",
                params![internal_event_id],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(payload) = existing_payload {
            let stored_receipt: Receipt = serde_json::from_str(&payload)
                .map_err(|e| StorageError::Init(format!("receipt deserialize: {e}")))?;
            let captured_count = stored_receipt.mutations.len();
            let public_event_id = match &stored_receipt.evidence {
                Some(ReceiptEvidence::SynapticCapture(evidence)) => {
                    evidence.trigger.event_id.clone()
                }
                _ => {
                    return Err(StorageError::Init(
                        "synaptic event references a receipt without capture evidence".into(),
                    ));
                }
            };
            let public_receipt_id = stored_receipt.receipt_id.clone();
            tx.commit()?;
            drop(writer);
            let receipt = self.get_receipt(&public_receipt_id)?.ok_or_else(|| {
                StorageError::Init("synaptic event references a missing receipt".into())
            })?;
            return Ok(DurableSynapticCapture {
                event_id: public_event_id,
                receipt,
                captured_count,
                reused_existing: true,
            });
        }

        let trigger_eligible: Option<i64> = tx
            .query_row(
                "SELECT CASE
                    WHEN suppression_count = 0
                     AND superseded_by IS NULL
                     AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?2)
                     AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?2)
                     AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?3)
                    THEN 1 ELSE 0 END
                 FROM knowledge_nodes WHERE id = ?1",
                params![request.trigger_memory_id, occurred_at_ms, transaction_at_ms],
                |row| row.get(0),
            )
            .optional()?;
        if trigger_eligible != Some(1) {
            return Err(StorageError::Init(
                "synaptic capture trigger is missing, suppressed, superseded, or not currently valid"
                    .into(),
            ));
        }

        // Public identifiers are minted only for a new decision and carry no
        // trigger-derived material. Retries recover them from the committed
        // receipt above; the private fingerprint remains an internal join key.
        let public_event_id = random_public_id("sevt");
        let public_receipt_id = random_public_id("r_syn");

        tx.execute(
            "INSERT INTO synaptic_events
                 (event_id, trigger_memory_id, event_type, occurred_at_ms,
                  window_from_ms, window_to_ms, strength, algorithm_version,
                  receipt_id, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                internal_event_id,
                request.trigger_memory_id,
                request.event_type,
                occurred_at_ms,
                window_from_ms,
                window_to_ms,
                event_strength,
                SYNAPTIC_CAPTURE_ALGORITHM_V1,
                public_receipt_id,
                recorded_at.to_rfc3339(),
            ],
        )?;

        let effective_window =
            CaptureWindow::new(request.policy.backward_hours, evaluated_forward_hours);
        let mut candidates = {
            let mut stmt = tx.prepare(
                "SELECT t.tag_id, t.memory_id, t.created_at_ms, t.initial_strength,
                        n.retrieval_strength, n.retention_strength, n.stability,
                        n.suppression_count,
                        CASE
                            WHEN n.superseded_by IS NULL
                             AND (n.valid_from IS NULL OR unixepoch(n.valid_from) * 1000 <= ?1)
                             AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?1)
                             AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?4)
                            THEN 1 ELSE 0
                        END AS currently_valid
                 FROM synaptic_tags t
                 JOIN knowledge_nodes n ON n.id = t.memory_id
                 WHERE t.state = 'active'
                   AND t.created_at_ms BETWEEN ?2 AND ?1
                   AND t.memory_id <> ?3
                 ORDER BY t.created_at_ms ASC, t.tag_id ASC",
            )?;
            let rows = stmt.query_map(
                params![
                    occurred_at_ms,
                    window_from_ms,
                    request.trigger_memory_id,
                    transaction_at_ms,
                ],
                |row| {
                    let encoded_at_ms: i64 = row.get(2)?;
                    let initial_strength: f64 = row.get(3)?;
                    let encoded_at = millis_to_datetime(encoded_at_ms);
                    let elapsed_hours = (occurred_at_ms - encoded_at_ms) as f64 / 3_600_000.0;
                    let tag_strength = request.policy.decay_function.apply(
                        initial_strength,
                        elapsed_hours,
                        request.policy.tag_lifetime_hours,
                    );
                    let capture_probability = effective_window
                        .capture_probability(encoded_at, request.occurred_at)
                        .unwrap_or(0.0);
                    let capture_score = tag_strength * capture_probability * event_strength;
                    Ok(CandidateRow {
                        tag_id: row.get(0)?,
                        memory_id: row.get(1)?,
                        encoded_at_ms,
                        retrieval_before: row.get(4)?,
                        retention_before: row.get(5)?,
                        stability_before: row.get(6)?,
                        suppression_count: row.get(7)?,
                        currently_valid: row.get::<_, i64>(8)? != 0,
                        temporal_distance_hours: elapsed_hours,
                        capture_probability,
                        tag_strength,
                        capture_score,
                    })
                },
            )?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row?);
            }
            values
        };

        // Stable ranking makes max-capture competition replayable across runs.
        candidates.sort_by(|a, b| {
            b.capture_score
                .total_cmp(&a.capture_score)
                .then_with(|| a.encoded_at_ms.cmp(&b.encoded_at_ms))
                .then_with(|| a.tag_id.cmp(&b.tag_id))
        });

        let mut evidence_candidates = Vec::with_capacity(candidates.len());
        let mut retrieved = Vec::new();
        let mut suppressed = Vec::new();
        let mut activation_path = Vec::new();
        let mut trust_scores = Vec::new();
        let mut mutations = Vec::new();
        let mut captured_count = 0usize;

        for (index, candidate) in candidates.into_iter().enumerate() {
            let evidence_slot = format!("candidate_{}", index + 1);
            let encoded_at = millis_to_datetime(candidate.encoded_at_ms);
            let mut disposition = SynapticCaptureDisposition::BelowThreshold;
            let mut reason = Some("capture score below the configured threshold".to_string());
            let mut public_memory_id = None;
            let mut strength_change = None;

            if candidate.suppression_count > 0 {
                disposition = SynapticCaptureDisposition::WithheldSuppressed;
                reason =
                    Some("active suppression forbids promotion and stable-id disclosure".into());
                public_memory_id = None;
                suppressed.push(SuppressedReceiptEntry::new(
                    evidence_slot.clone(),
                    SuppressReason::Privacy,
                ));
            } else if !candidate.currently_valid {
                disposition = SynapticCaptureDisposition::WithheldInvalid;
                reason = Some("memory is superseded or outside its valid-time interval".into());
                public_memory_id = None;
                suppressed.push(SuppressedReceiptEntry::new(
                    evidence_slot.clone(),
                    SuppressReason::Contradicted,
                ));
            } else if candidate.tag_strength < request.policy.minimum_tag_strength
                || candidate.capture_score < request.policy.minimum_tag_strength
            {
                if candidate.tag_strength <= f64::EPSILON {
                    tx.execute(
                        "UPDATE synaptic_tags SET state = 'expired' WHERE tag_id = ?1",
                        params![candidate.tag_id],
                    )?;
                }
            } else if captured_count >= request.policy.maximum_captures {
                disposition = SynapticCaptureDisposition::LostCompetition;
                reason = Some("eligible but outside the deterministic capture limit".into());
            } else {
                let changed = tx.execute(
                    "UPDATE knowledge_nodes SET
                        last_accessed = ?1,
                        retrieval_strength = MIN(1.0, retrieval_strength + 0.20),
                        retention_strength = MIN(1.0, retention_strength + 0.10),
                        stability = MIN(stability * 1.5, stability + 365.0),
                        waking_tag = TRUE,
                        waking_tag_at = ?1
                     WHERE id = ?2
                       AND suppression_count = 0
                       AND superseded_by IS NULL
                       AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?3)
                       AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?3)
                       AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?4)",
                    params![
                        recorded_at.to_rfc3339(),
                        candidate.memory_id,
                        occurred_at_ms,
                        transaction_at_ms,
                    ],
                )?;
                if changed == 1 {
                    let (retrieval_after, retention_after, stability_after): (f64, f64, f64) = tx
                        .query_row(
                        "SELECT retrieval_strength, retention_strength, stability
                             FROM knowledge_nodes WHERE id = ?1",
                        params![candidate.memory_id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )?;
                    tx.execute(
                        "UPDATE synaptic_tags SET
                            state = 'captured', capture_event_id = ?1, captured_at_ms = ?2
                         WHERE tag_id = ?3 AND state = 'active'",
                        params![
                            internal_event_id,
                            recorded_at.timestamp_millis(),
                            candidate.tag_id
                        ],
                    )?;
                    tx.execute(
                        "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
                         VALUES (?1, 'synaptic_capture', ?2)",
                        params![candidate.memory_id, recorded_at.to_rfc3339()],
                    )?;
                    disposition = SynapticCaptureDisposition::Captured;
                    public_memory_id = Some(candidate.memory_id.clone());
                    reason = Some(format!(
                        "temporal eligibility score {:.4} met the configured threshold",
                        candidate.capture_score
                    ));
                    strength_change = Some(SynapticStrengthChange {
                        retrieval_strength: StrengthDelta {
                            before: candidate.retrieval_before,
                            after: retrieval_after,
                        },
                        retention_strength: StrengthDelta {
                            before: candidate.retention_before,
                            after: retention_after,
                        },
                        stability: StrengthDelta {
                            before: candidate.stability_before,
                            after: stability_after,
                        },
                    });
                    retrieved.push(candidate.memory_id.clone());
                    trust_scores.push(candidate.retention_before);
                    activation_path.push(format!(
                        "{} --[tagged; {:.2}h before event]--> {}",
                        candidate.memory_id,
                        candidate.temporal_distance_hours,
                        request.trigger_memory_id
                    ));
                    mutations.push(ReceiptMutation {
                        id: candidate.memory_id.clone(),
                        kind: "synaptic_capture".into(),
                        note: Some(format!(
                            "Evidence-backed temporal association; capture score {:.4}",
                            candidate.capture_score
                        )),
                    });
                    captured_count += 1;
                } else {
                    disposition = SynapticCaptureDisposition::WithheldInvalid;
                    reason = Some("guarded promotion rejected a concurrent state change".into());
                    public_memory_id = None;
                    suppressed.push(SuppressedReceiptEntry::new(
                        evidence_slot.clone(),
                        SuppressReason::Privacy,
                    ));
                }
            }

            tx.execute(
                "INSERT INTO synaptic_capture_items
                     (event_id, tag_id, memory_id, evidence_slot, receipt_id,
                      encoded_at_ms, temporal_distance_hours, capture_probability,
                      tag_strength_at_evaluation, capture_score, disposition, reason,
                      retrieval_before, retrieval_after, retention_before,
                      retention_after, stability_before, stability_after, recorded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                         ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
                params![
                    internal_event_id,
                    candidate.tag_id,
                    candidate.memory_id,
                    evidence_slot,
                    public_receipt_id,
                    candidate.encoded_at_ms,
                    candidate.temporal_distance_hours,
                    candidate.capture_probability,
                    candidate.tag_strength,
                    candidate.capture_score,
                    disposition_label(disposition),
                    reason,
                    candidate.retrieval_before,
                    strength_change.as_ref().map(|s| s.retrieval_strength.after),
                    candidate.retention_before,
                    strength_change.as_ref().map(|s| s.retention_strength.after),
                    candidate.stability_before,
                    strength_change.as_ref().map(|s| s.stability.after),
                    recorded_at.to_rfc3339(),
                ],
            )?;

            evidence_candidates.push(SynapticCaptureCandidate {
                memory_id: public_memory_id,
                evidence_slot,
                encoded_at,
                temporal_distance_hours: candidate.temporal_distance_hours,
                capture_probability: candidate.capture_probability,
                tag_strength_at_evaluation: candidate.tag_strength,
                capture_score: candidate.capture_score,
                disposition,
                reason,
                strength_change,
            });
        }

        let evidence = SynapticCaptureEvidence {
            schema: SYNAPTIC_CAPTURE_SCHEMA_V1.into(),
            schema_version: 1,
            algorithm_version: SYNAPTIC_CAPTURE_ALGORITHM_V1.into(),
            trigger: SynapticCaptureTrigger {
                event_id: public_event_id.clone(),
                memory_id: request.trigger_memory_id.clone(),
                event_type: request.event_type.clone(),
                occurred_at: request.occurred_at,
                importance_score: event_strength,
            },
            capture_window: SynapticCaptureWindow {
                evaluation_direction: "backward_only".into(),
                backward_hours: request.policy.backward_hours,
                forward_hours: evaluated_forward_hours,
                tag_lifetime_hours: request.policy.tag_lifetime_hours,
                minimum_tag_strength: request.policy.minimum_tag_strength,
                maximum_captures: request.policy.maximum_captures,
                decay_function: format!("{:?}", request.policy.decay_function).to_lowercase(),
            },
            candidates: evidence_candidates,
            claim_boundary: SYNAPTIC_CAPTURE_CLAIM_BOUNDARY.into(),
        };
        let mut receipt = Receipt::build_with_unique(
            recorded_at,
            &public_event_id,
            &public_receipt_id,
            retrieved,
            suppressed,
            activation_path,
            &trust_scores,
            mutations,
        )
        .with_evidence(ReceiptEvidence::SynapticCapture(evidence));
        receipt.receipt_id = public_receipt_id.clone();
        let payload = serde_json::to_string(&receipt)
            .map_err(|e| StorageError::Init(format!("receipt serialize: {e}")))?;
        tx.execute(
            "INSERT INTO memory_receipts
                 (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
                  trust_floor, decay_risk, payload, created_at)
             VALUES (?1, NULL, 'smart_ingest', NULL, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                public_receipt_id,
                receipt.retrieved.len() as i64,
                receipt.suppressed.len() as i64,
                receipt.trust_floor,
                receipt.decay_risk.as_str(),
                payload,
                recorded_at.to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        drop(writer);
        let receipt = self.get_receipt(&public_receipt_id)?.ok_or_else(|| {
            StorageError::Init("committed synaptic receipt could not be read back".into())
        })?;

        Ok(DurableSynapticCapture {
            event_id: public_event_id,
            receipt,
            captured_count,
            reused_existing: false,
        })
    }
}

fn deterministic_id(prefix: &str, input: &str) -> String {
    let digest = blake3::hash(input.as_bytes()).to_hex();
    format!("{prefix}_{}", &digest.as_str()[..24])
}

fn random_public_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

fn hours_to_millis(hours: f64) -> Option<i64> {
    if !hours.is_finite() || hours < 0.0 {
        return None;
    }
    let millis = hours * 3_600_000.0;
    if millis > i64::MAX as f64 {
        None
    } else {
        Some(millis.round() as i64)
    }
}

fn millis_to_datetime(value: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(value)
        .single()
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
}

fn disposition_label(value: SynapticCaptureDisposition) -> &'static str {
    match value {
        SynapticCaptureDisposition::Captured => "captured",
        SynapticCaptureDisposition::BelowThreshold => "below_threshold",
        SynapticCaptureDisposition::WithheldSuppressed => "withheld_suppressed",
        SynapticCaptureDisposition::WithheldInvalid => "withheld_invalid",
        SynapticCaptureDisposition::LostCompetition => "lost_competition",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IngestInput, Storage};
    use tempfile::TempDir;

    fn policy() -> SynapticCapturePolicy {
        SynapticCapturePolicy {
            backward_hours: 9.0,
            forward_hours: 2.0,
            tag_lifetime_hours: 12.0,
            minimum_tag_strength: 0.3,
            maximum_captures: 50,
            decay_function: DecayFunction::Exponential,
        }
    }

    fn ingest(storage: &Storage, content: &str) -> crate::KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.into(),
                node_type: "decision".into(),
                ..Default::default()
            })
            .unwrap()
    }

    #[test]
    fn active_tags_survive_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("restart.db");
        let memory_id;
        {
            let storage = Storage::new(Some(path.clone())).unwrap();
            let node = ingest(&storage, "persist this tag");
            memory_id = node.id.clone();
            storage
                .save_synaptic_tag(&SynapticTag::new(&node.id))
                .unwrap();
        }
        let reopened = Storage::new(Some(path)).unwrap();
        let tags = reopened.load_active_synaptic_tags().unwrap();
        assert!(tags.iter().any(|tag| tag.memory_id == memory_id));
    }

    #[test]
    fn capture_is_atomic_persisted_and_idempotent() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("capture.db"))).unwrap();
        let earlier = ingest(&storage, "retry policy chosen");
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = Utc::now() - chrono::Duration::minutes(5);
        storage.save_synaptic_tag(&tag).unwrap();
        let before = storage.demote_memory(&earlier.id).unwrap();
        let trigger = ingest(&storage, "retry policy caused a production incident");
        let request = SynapticCaptureRequest {
            trigger_memory_id: trigger.id,
            event_type: "novelty_spike".into(),
            occurred_at: Utc::now(),
            strength: 0.9,
            policy: policy(),
        };

        let first = storage.capture_synaptic_event(&request).unwrap();
        assert_eq!(first.captured_count, 1);
        assert!(!first.reused_existing);
        let (internal_event_id, denormalized_query): (String, Option<String>) = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT e.event_id, r.query
                 FROM synaptic_events e
                 JOIN memory_receipts r ON r.receipt_id = e.receipt_id",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_ne!(
            first.event_id, internal_event_id,
            "the recomputable idempotency fingerprint must never be public"
        );
        assert_ne!(
            first.receipt.receipt_id,
            deterministic_id("r_syn", &internal_event_id),
            "the public receipt id must not derive from the internal event fingerprint"
        );
        assert_eq!(
            denormalized_query, None,
            "receipt metadata must not expose the internal event fingerprint"
        );
        let after_first = storage.get_node(&earlier.id).unwrap().unwrap();
        assert!(after_first.retrieval_strength > before.retrieval_strength);
        let persisted = storage
            .get_receipt(&first.receipt.receipt_id)
            .unwrap()
            .expect("typed receipt persists");
        assert!(matches!(
            persisted.evidence,
            Some(ReceiptEvidence::SynapticCapture(_))
        ));

        let second = storage.capture_synaptic_event(&request).unwrap();
        assert!(second.reused_existing);
        assert_eq!(second.event_id, first.event_id);
        assert_eq!(second.receipt.receipt_id, first.receipt.receipt_id);
        let after_second = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(
            after_second.retrieval_strength, after_first.retrieval_strength,
            "event retry must not promote twice"
        );
    }

    #[test]
    fn retagged_memory_is_promoted_only_once_per_event() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("retag.db"))).unwrap();
        let earlier = ingest(&storage, "same memory receives a newer tag episode");
        let mut first_tag = SynapticTag::new(&earlier.id);
        first_tag.created_at = Utc::now() - chrono::Duration::minutes(6);
        storage.save_synaptic_tag(&first_tag).unwrap();
        let mut second_tag = SynapticTag::new(&earlier.id);
        second_tag.created_at = Utc::now() - chrono::Duration::minutes(3);
        storage.save_synaptic_tag(&second_tag).unwrap();
        let before = storage.demote_memory(&earlier.id).unwrap();
        let trigger = ingest(&storage, "one event evaluates the latest active tag");
        let capture = storage
            .capture_synaptic_event(&SynapticCaptureRequest {
                trigger_memory_id: trigger.id,
                event_type: "novelty_spike".into(),
                occurred_at: Utc::now(),
                strength: 0.9,
                policy: policy(),
            })
            .unwrap();

        assert_eq!(capture.captured_count, 1);
        assert_eq!(capture.receipt.mutations.len(), 1);
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert!((after.retrieval_strength - before.retrieval_strength - 0.20).abs() < 1e-9);
    }

    #[test]
    fn suppressed_candidate_is_audited_without_stable_id_or_promotion() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("suppressed.db"))).unwrap();
        let earlier = ingest(&storage, "approval before incident");
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = Utc::now() - chrono::Duration::minutes(2);
        storage.save_synaptic_tag(&tag).unwrap();
        let suppressed = storage.suppress_memory(&earlier.id).unwrap();
        let trigger = ingest(&storage, "incident after approval");
        let result = storage
            .capture_synaptic_event(&SynapticCaptureRequest {
                trigger_memory_id: trigger.id,
                event_type: "novelty_spike".into(),
                occurred_at: Utc::now(),
                strength: 0.9,
                policy: policy(),
            })
            .unwrap();
        assert_eq!(result.captured_count, 0);
        let serialized = serde_json::to_string(&result.receipt).unwrap();
        assert!(!serialized.contains(&earlier.id));
        assert!(serialized.contains("withheld_suppressed"));
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, suppressed.retrieval_strength);
    }

    #[test]
    fn suppressed_trigger_is_rejected_inside_capture_transaction() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("trigger_guard.db"))).unwrap();
        let earlier = ingest(&storage, "candidate must not move for invalid trigger");
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = Utc::now() - chrono::Duration::minutes(2);
        storage.save_synaptic_tag(&tag).unwrap();
        let before = storage.get_node(&earlier.id).unwrap().unwrap();
        let trigger = ingest(&storage, "trigger suppressed before transaction");
        storage.suppress_memory(&trigger.id).unwrap();

        let result = storage.capture_synaptic_event(&SynapticCaptureRequest {
            trigger_memory_id: trigger.id,
            event_type: "novelty_spike".into(),
            occurred_at: Utc::now(),
            strength: 0.9,
            policy: policy(),
        });
        assert!(result.is_err());
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
        assert!(storage.list_receipts(10).unwrap().is_empty());
    }

    #[test]
    fn expired_or_superseded_candidate_is_withheld_by_guarded_transaction() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("invalid.db"))).unwrap();
        let earlier = ingest(&storage, "decision later superseded");
        let event_time = Utc::now() - chrono::Duration::seconds(2);
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = event_time - chrono::Duration::minutes(2);
        storage.save_synaptic_tag(&tag).unwrap();
        let before = storage.get_node(&earlier.id).unwrap().unwrap();
        {
            let writer = storage.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET valid_until = ?1, superseded_by = 'replacement'
                     WHERE id = ?2",
                    params![
                        (event_time + chrono::Duration::seconds(1)).to_rfc3339(),
                        earlier.id,
                    ],
                )
                .unwrap();
        }
        let trigger = ingest(&storage, "incident after the superseded decision");
        let result = storage
            .capture_synaptic_event(&SynapticCaptureRequest {
                trigger_memory_id: trigger.id,
                event_type: "novelty_spike".into(),
                occurred_at: event_time,
                strength: 0.9,
                policy: policy(),
            })
            .unwrap();

        assert_eq!(result.captured_count, 0);
        let serialized = serde_json::to_string(&result.receipt).unwrap();
        assert!(!serialized.contains(&earlier.id));
        assert!(serialized.contains("withheld_invalid"));
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
    }

    #[test]
    fn capture_samples_current_validity_after_waiting_for_another_writer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("writer_barrier.db");
        let setup = Storage::new(Some(path.clone())).unwrap();
        let event_time = Utc::now() - chrono::Duration::seconds(10);
        let earlier = ingest(&setup, "candidate tombstoned while capture waits");
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = event_time - chrono::Duration::minutes(2);
        setup.save_synaptic_tag(&tag).unwrap();
        let before = setup.get_node(&earlier.id).unwrap().unwrap();
        let trigger = ingest(&setup, "trigger queued behind a tombstone writer");
        drop(setup);

        let blocker = Storage::new(Some(path.clone())).unwrap();
        let capture_store = std::sync::Arc::new(Storage::new(Some(path)).unwrap());
        let request = SynapticCaptureRequest {
            trigger_memory_id: trigger.id,
            event_type: "novelty_spike".into(),
            occurred_at: event_time,
            strength: 0.9,
            policy: policy(),
        };

        // Hold SQLite's writer slot across the validity cutoff. A buggy capture
        // that samples `now` before BEGIN IMMEDIATE sees the candidate as live,
        // then waits and promotes it after this tombstone commits. Sampling
        // after BEGIN observes the committed cutoff and withholds it.
        let cutoff = Utc::now() + chrono::Duration::seconds(2);
        let mut blocker_writer = blocker.writer.lock().unwrap();
        let blocker_tx = blocker_writer
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        blocker_tx
            .execute(
                "UPDATE knowledge_nodes SET valid_until = ?1 WHERE id = ?2",
                params![cutoff.to_rfc3339(), earlier.id],
            )
            .unwrap();

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let capture_barrier = std::sync::Arc::clone(&barrier);
        let capture_clone = std::sync::Arc::clone(&capture_store);
        let capture_thread = std::thread::spawn(move || {
            capture_barrier.wait();
            capture_clone.capture_synaptic_event(&request)
        });
        barrier.wait();
        let sleep_for = (cutoff - Utc::now())
            .to_std()
            .unwrap_or_default()
            .saturating_add(std::time::Duration::from_millis(100));
        std::thread::sleep(sleep_for);
        blocker_tx.commit().unwrap();
        drop(blocker_writer);

        let result = capture_thread.join().unwrap().unwrap();
        assert_eq!(result.captured_count, 0);
        let serialized = serde_json::to_string(&result.receipt).unwrap();
        assert!(serialized.contains("withheld_invalid"));
        assert!(!serialized.contains(&earlier.id));
        let after = capture_store.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
    }

    #[test]
    fn later_suppression_and_purge_cannot_resurrect_ids_through_receipt_reads() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("redaction.db"))).unwrap();
        let earlier = ingest(&storage, "captured memory later suppressed");
        let mut tag = SynapticTag::new(&earlier.id);
        tag.created_at = Utc::now() - chrono::Duration::minutes(3);
        storage.save_synaptic_tag(&tag).unwrap();
        let trigger = ingest(&storage, "trigger later purged");
        let capture = storage
            .capture_synaptic_event(&SynapticCaptureRequest {
                trigger_memory_id: trigger.id.clone(),
                event_type: "novelty_spike".into(),
                occurred_at: Utc::now(),
                strength: 0.9,
                policy: policy(),
            })
            .unwrap();
        let public_event_id = capture.event_id.clone();
        let receipt_id = capture.receipt.receipt_id;
        let internal_event_id: String = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT event_id FROM synaptic_events WHERE trigger_memory_id = ?1",
                params![trigger.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_ne!(public_event_id, internal_event_id);
        assert_ne!(
            receipt_id,
            deterministic_id("r_syn", &internal_event_id),
            "receipt id must remain unlinkable even if the private fingerprint is known"
        );

        storage.suppress_memory(&earlier.id).unwrap();
        let after_suppress = storage.get_receipt(&receipt_id).unwrap().unwrap();
        let public_json = serde_json::to_string(&after_suppress).unwrap();
        assert!(!public_json.contains(&earlier.id));
        assert!(public_json.contains("redacted_"));

        storage
            .purge_node(&trigger.id, Some("privacy test"))
            .unwrap();
        let after_purge = storage.get_receipt(&receipt_id).unwrap().unwrap();
        let public_json = serde_json::to_string(&after_purge).unwrap();
        assert!(!public_json.contains(&trigger.id));
        assert!(public_json.contains("purged_1"));

        let raw_payload: String = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT payload FROM memory_receipts WHERE receipt_id = ?1",
                params![receipt_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            !raw_payload.contains(&trigger.id),
            "purge must scrub the durable receipt payload, not only its public projection"
        );
        let event_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM synaptic_events WHERE trigger_memory_id = ?1",
                params![trigger.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(event_count, 0);
        let capture_event_id: Option<String> = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT capture_event_id FROM synaptic_tags WHERE memory_id = ?1",
                params![earlier.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            capture_event_id, None,
            "purge must clear tag linkage to the private event fingerprint"
        );
    }

    #[test]
    fn frozen_snapshot_replay_matches_receipt_and_semantic_state_after_reopen() {
        let dir = TempDir::new().unwrap();
        let seed_path = dir.path().join("seed.db");
        let arm_a_path = dir.path().join("arm_a.db");
        let arm_b_path = dir.path().join("arm_b.db");
        let event_time = Utc::now();
        let (earlier_id, request) = {
            let seed = Storage::new(Some(seed_path.clone())).unwrap();
            let earlier = ingest(&seed, "frozen replay candidate");
            let mut tag = SynapticTag::new(&earlier.id);
            tag.created_at = event_time - chrono::Duration::minutes(4);
            seed.save_synaptic_tag(&tag).unwrap();
            seed.demote_memory(&earlier.id).unwrap();
            let trigger = ingest(&seed, "frozen replay trigger");
            (
                earlier.id,
                SynapticCaptureRequest {
                    trigger_memory_id: trigger.id,
                    event_type: "novelty_spike".into(),
                    occurred_at: event_time,
                    strength: 0.9,
                    policy: policy(),
                },
            )
        };
        std::fs::copy(&seed_path, &arm_a_path).unwrap();
        std::fs::copy(&seed_path, &arm_b_path).unwrap();

        let run_arm = |path: &std::path::Path| {
            let storage = Storage::new(Some(path.to_path_buf())).unwrap();
            let capture = storage.capture_synaptic_event(&request).unwrap();
            let mut semantic_receipt = capture.receipt;
            semantic_receipt.receipt_id = "<public-receipt-id>".into();
            if let Some(ReceiptEvidence::SynapticCapture(evidence)) = &mut semantic_receipt.evidence
            {
                evidence.trigger.event_id = "<public-event-id>".into();
            }
            let node = storage.get_node(&earlier_id).unwrap().unwrap();
            let tag_state: (String, Option<String>, Option<i64>) = storage
                .reader
                .lock()
                .unwrap()
                .query_row(
                    "SELECT state, capture_event_id, captured_at_ms
                     FROM synaptic_tags WHERE memory_id = ?1 ORDER BY created_at_ms DESC LIMIT 1",
                    params![&earlier_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            (
                serde_json::to_string(&semantic_receipt).unwrap(),
                (
                    node.retrieval_strength,
                    node.retention_strength,
                    node.stability,
                    node.last_accessed,
                ),
                tag_state,
            )
        };

        let arm_a = run_arm(&arm_a_path);
        let arm_b = run_arm(&arm_b_path);
        assert_eq!(arm_a, arm_b);
    }
}
