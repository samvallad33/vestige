//! Durable synaptic tag-and-capture persistence.
//!
//! The in-process neuroscience engine remains a fast projection, but this
//! module is the source of truth for capture eligibility and mutation. One
//! SQLite writer transaction records the event, evaluates every candidate,
//! applies guarded promotions, consumes captured tags, writes candidate rows,
//! and inserts the complete typed receipt payload.

use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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
pub const SYNAPTIC_CAPTURE_ALGORITHM_V2: &str = "vestige.synaptic_capture.v2";
pub const SYNAPTIC_CAPTURE_SCHEMA_V2: &str =
    "https://vestige.dev/schemas/receipt/synaptic-capture/v2";
pub const SYNAPTIC_CONTEXT_ALGORITHM_V1: &str = "vestige.synaptic_context.v1";
pub const SYNAPTIC_CONTEXT_THRESHOLD_V1: f64 = 0.25;
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

/// Privacy-safe numeric snapshot of the importance signal that opened an
/// event. Explanation strings are intentionally excluded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynapticSignalSnapshot {
    pub novelty: f64,
    pub arousal: f64,
    pub reward: f64,
    pub attention: f64,
    pub composite: f64,
}

impl SynapticSignalSnapshot {
    fn normalized(&self) -> Self {
        fn scalar(value: f64) -> f64 {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        Self {
            novelty: scalar(self.novelty),
            arousal: scalar(self.arousal),
            reward: scalar(self.reward),
            attention: scalar(self.attention),
            composite: scalar(self.composite),
        }
    }
}

/// A frozen V2 importance event supplied with the tag produced by one ingest.
#[derive(Debug, Clone)]
pub struct SynapticImportanceEvent {
    pub event_type: String,
    pub occurred_at: DateTime<Utc>,
    pub strength: f64,
    pub policy: SynapticCapturePolicy,
    pub signal_snapshot: SynapticSignalSnapshot,
}

/// Atomic event + tag request used by smart_ingest V2.
#[derive(Debug, Clone)]
pub struct SynapticIngestRequest {
    pub memory_id: String,
    pub tag: Option<SynapticTag>,
    pub event: Option<SynapticImportanceEvent>,
}

/// One immutable V2 forward pair receipt.
#[derive(Debug, Clone)]
pub struct DurableSynapticPairReceipt {
    pub event_id: String,
    pub receipt: Receipt,
    pub disposition: SynapticCaptureDisposition,
    pub reused_existing: bool,
}

/// Observable result of one committed V2 ingest transaction.
#[derive(Debug, Clone)]
pub struct SynapticIngestOutcome {
    pub event: Option<DurableSynapticCapture>,
    pub tag_id: Option<String>,
    pub tag_persisted: bool,
    /// Whether the request tag remains active after the committed transaction.
    /// A tag can be persisted then immediately consumed by an already-open
    /// forward event, so callers must not restore it into a live projection
    /// solely because `tag_persisted` is true.
    pub tag_active: bool,
    pub forward_receipts: Vec<DurableSynapticPairReceipt>,
}

#[derive(Debug, Clone)]
struct ContextEvidence {
    score: f64,
    method: String,
}

#[derive(Debug, Clone)]
struct V2EventRow {
    internal_event_id: String,
    public_event_id: String,
    trigger_memory_id: String,
    event_type: String,
    occurred_at_ms: i64,
    window_from_ms: i64,
    window_to_ms: i64,
    strength: f64,
    parent_receipt_id: String,
    tag_lifetime_hours: f64,
    minimum_tag_strength: f64,
    minimum_association_score: f64,
    maximum_captures: usize,
    decay_function: DecayFunction,
    context_threshold: f64,
    trigger_suppression_count: i64,
    trigger_currently_valid: bool,
}

#[derive(Debug, Clone)]
struct V2TagRow {
    tag_id: String,
    memory_id: String,
    created_at_ms: i64,
    initial_strength: f64,
    suppression_count: i64,
    currently_valid: bool,
}

#[derive(Debug, Clone)]
struct PairReceiptRecord {
    event_id: String,
    receipt_id: String,
    disposition: SynapticCaptureDisposition,
    reused_existing: bool,
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
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_synaptic_tag")?;
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

    /// Atomically persist one ingest's importance event and tag, evaluate the
    /// event's backward window, and reconcile every still-open forward pair.
    ///
    /// The transaction is the only mutation authority. The caller may update
    /// its in-process cognitive projection after this method commits, but a
    /// projection failure cannot make an uncommitted capture appear durable.
    pub fn process_synaptic_ingest(
        &self,
        request: &SynapticIngestRequest,
    ) -> Result<SynapticIngestOutcome> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "process_synaptic_ingest")?;
        let transaction_at_ms = Utc::now().timestamp_millis();

        tx.execute(
            "UPDATE synaptic_events SET event_state = 'closed'
             WHERE algorithm_version = ?1 AND event_state = 'open'
               AND window_to_ms < ?2",
            params![SYNAPTIC_CAPTURE_ALGORITHM_V2, transaction_at_ms],
        )?;

        if !memory_is_eligible_tx(
            &tx,
            &request.memory_id,
            request
                .tag
                .as_ref()
                .map(|tag| tag.created_at.timestamp_millis())
                .or_else(|| {
                    request
                        .event
                        .as_ref()
                        .map(|event| event.occurred_at.timestamp_millis())
                })
                .unwrap_or(transaction_at_ms),
            transaction_at_ms,
        )? {
            return Err(StorageError::Init(
                "synaptic ingest memory is missing, suppressed, superseded, or not currently valid"
                    .into(),
            ));
        }

        let mut event_result = None;
        let mut newly_inserted_event: Option<V2EventRow> = None;
        if let Some(event) = &request.event {
            let (durable, inserted, row) =
                upsert_v2_event_and_backward_tx(&tx, &request.memory_id, event, transaction_at_ms)?;
            event_result = Some(durable);
            // A historical event whose forward window elapsed before this
            // transaction is recorded closed. Its backward decision remains
            // auditable, but it cannot be backfilled into a new forward pair.
            if inserted && row.window_to_ms >= transaction_at_ms {
                newly_inserted_event = Some(row);
            }
        }

        let mut tag_id = None;
        if let Some(tag) = &request.tag {
            if tag.memory_id != request.memory_id {
                return Err(StorageError::Init(
                    "synaptic ingest tag memory does not match request memory".into(),
                ));
            }
            tag_id = Some(upsert_synaptic_tag_tx(&tx, tag, transaction_at_ms)?);
        }

        let mut tags_to_reconcile = BTreeSet::new();
        if let Some(id) = &tag_id {
            tags_to_reconcile.insert(id.clone());
        }
        if let Some(event) = &newly_inserted_event {
            let mut stmt = tx.prepare(
                "SELECT tag_id FROM synaptic_tags
                 WHERE state = 'active'
                   AND created_at_ms > ?1 AND created_at_ms <= ?2
                   AND memory_id <> ?3
                 ORDER BY created_at_ms ASC, tag_id ASC",
            )?;
            let rows = stmt.query_map(
                params![
                    event.occurred_at_ms,
                    event.window_to_ms,
                    event.trigger_memory_id
                ],
                |row| row.get::<_, String>(0),
            )?;
            for row in rows {
                tags_to_reconcile.insert(row?);
            }
        }

        let mut pair_records = Vec::new();
        for id in tags_to_reconcile {
            pair_records.extend(reconcile_forward_tag_tx(&tx, &id, transaction_at_ms)?);
        }

        // `tag_persisted` only means the episode row exists. An existing open
        // V22 event may have consumed it in the reconciliation immediately
        // above. Capture this state while the writer transaction is still the
        // authority so a caller cannot reintroduce a captured tag to its live
        // in-process projection after commit.
        let tag_active = match &tag_id {
            Some(id) => {
                tx.query_row(
                    "SELECT CASE WHEN state = 'active' THEN 1 ELSE 0 END
                 FROM synaptic_tags WHERE tag_id = ?1",
                    params![id],
                    |row| row.get::<_, i64>(0),
                )? != 0
            }
            None => false,
        };

        tx.commit()?;
        drop(writer);

        // Re-read through the state-aware receipt projection so the API never
        // bypasses suppression/redaction rules that changed at commit time.
        if let Some(durable) = &mut event_result {
            durable.receipt = self
                .get_receipt(&durable.receipt.receipt_id)?
                .ok_or_else(|| StorageError::Init("missing committed root receipt".into()))?;
        }
        let mut forward_receipts = Vec::with_capacity(pair_records.len());
        for record in pair_records {
            let receipt = self
                .get_receipt(&record.receipt_id)?
                .ok_or_else(|| StorageError::Init("missing committed pair receipt".into()))?;
            forward_receipts.push(DurableSynapticPairReceipt {
                event_id: record.event_id,
                receipt,
                disposition: record.disposition,
                reused_existing: record.reused_existing,
            });
        }

        Ok(SynapticIngestOutcome {
            event: event_result,
            tag_persisted: tag_id.is_some(),
            tag_id,
            tag_active,
            forward_receipts,
        })
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

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "capture_synaptic_event")?;
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
                  receipt_id, recorded_at, public_event_id, event_state)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'closed')",
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
                public_event_id,
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
                      retention_after, stability_before, stability_after, recorded_at,
                      evaluation_direction, temporal_score, context_score,
                      context_method, association_score, competition_rank,
                      algorithm_version, reason_code)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                         ?13, ?14, ?15, ?16, ?17, ?18, ?19,
                         'backward', ?8, 1.0, 'v1_ungated', ?10, ?20,
                         ?21, ?11)",
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
                    (index + 1) as i64,
                    SYNAPTIC_CAPTURE_ALGORITHM_V1,
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
                evaluation_direction: None,
                temporal_score: None,
                context_score: None,
                context_method: None,
                association_score: None,
                competition_rank: None,
                algorithm_version: None,
                reason_code: None,
                disposition,
                reason,
                strength_change,
            });
        }

        let evidence = SynapticCaptureEvidence {
            schema: SYNAPTIC_CAPTURE_SCHEMA_V1.into(),
            schema_version: 1,
            algorithm_version: SYNAPTIC_CAPTURE_ALGORITHM_V1.into(),
            receipt_role: None,
            parent_receipt_id: None,
            evaluation_direction: None,
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
                minimum_association_score: None,
                maximum_captures: request.policy.maximum_captures,
                decay_function: format!("{:?}", request.policy.decay_function).to_lowercase(),
                context_threshold: None,
                context_algorithm_version: None,
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

fn memory_is_eligible_tx(
    tx: &Transaction<'_>,
    memory_id: &str,
    semantic_at_ms: i64,
    transaction_at_ms: i64,
) -> Result<bool> {
    let eligible = tx
        .query_row(
            "SELECT CASE
                WHEN suppression_count = 0
                 AND superseded_by IS NULL
                 AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?2)
                 AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?2)
                 AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?3)
                THEN 1 ELSE 0 END
             FROM knowledge_nodes WHERE id = ?1",
            params![memory_id, semantic_at_ms, transaction_at_ms],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    Ok(eligible == Some(1))
}

fn upsert_synaptic_tag_tx(
    tx: &Transaction<'_>,
    tag: &SynapticTag,
    transaction_at_ms: i64,
) -> Result<String> {
    let created_at_ms = tag.created_at.timestamp_millis();
    if created_at_ms > transaction_at_ms {
        return Err(StorageError::Init(
            "synaptic tag timestamp is in the future; V22 does not schedule future capture eligibility"
                .into(),
        ));
    }
    if !memory_is_eligible_tx(tx, &tag.memory_id, created_at_ms, transaction_at_ms)? {
        return Err(StorageError::Init(
            "synaptic tag memory is missing, suppressed, superseded, or not currently valid".into(),
        ));
    }
    let tag_id = deterministic_id("stag", &format!("{}|{}", tag.memory_id, created_at_ms));
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
            finite_unit(tag.initial_strength),
            tag.encoding_context,
            SYNAPTIC_CAPTURE_ALGORITHM_V2,
            tag.created_at.to_rfc3339(),
        ],
    )?;
    Ok(tag_id)
}

fn upsert_v2_event_and_backward_tx(
    tx: &Transaction<'_>,
    trigger_memory_id: &str,
    event: &SynapticImportanceEvent,
    transaction_at_ms: i64,
) -> Result<(DurableSynapticCapture, bool, V2EventRow)> {
    let occurred_at_ms = event.occurred_at.timestamp_millis();
    if occurred_at_ms > transaction_at_ms {
        return Err(StorageError::Init(
            "synaptic event timestamp is in the future; V22 does not evaluate future importance events"
                .into(),
        ));
    }
    if !memory_is_eligible_tx(tx, trigger_memory_id, occurred_at_ms, transaction_at_ms)? {
        return Err(StorageError::Init(
            "synaptic event trigger is missing, suppressed, superseded, or not currently valid"
                .into(),
        ));
    }

    let backward_ms = hours_to_millis(event.policy.backward_hours)
        .ok_or_else(|| StorageError::Init("invalid synaptic backward window".into()))?;
    let forward_ms = hours_to_millis(event.policy.forward_hours)
        .ok_or_else(|| StorageError::Init("invalid synaptic forward window".into()))?;
    let window_from_ms = occurred_at_ms.checked_sub(backward_ms).unwrap_or(i64::MIN);
    let window_to_ms = occurred_at_ms.checked_add(forward_ms).unwrap_or(i64::MAX);
    let strength = finite_unit(event.strength);
    let minimum_association_score =
        finite_unit(event.policy.minimum_tag_strength) * SYNAPTIC_CONTEXT_THRESHOLD_V1;
    let snapshot = event.signal_snapshot.normalized();
    let snapshot_json = serde_json::to_string(&snapshot)
        .map_err(|error| StorageError::Init(format!("signal snapshot serialize: {error}")))?;
    let decay_label = decay_function_label(event.policy.decay_function);
    let event_key = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        SYNAPTIC_CAPTURE_ALGORITHM_V2,
        trigger_memory_id,
        event.event_type,
        occurred_at_ms,
        strength.to_bits(),
        event.policy.backward_hours.to_bits(),
        event.policy.forward_hours.to_bits(),
        event.policy.tag_lifetime_hours.to_bits(),
        minimum_association_score.to_bits(),
        event.policy.minimum_tag_strength.to_bits(),
        event.policy.maximum_captures,
        decay_label,
        SYNAPTIC_CONTEXT_THRESHOLD_V1.to_bits(),
        SYNAPTIC_CONTEXT_ALGORITHM_V1,
        snapshot_json,
    );
    let internal_event_id = deterministic_id("sevt_fp", &event_key);

    if let Some((public_event_id, receipt_id, payload)) = tx
        .query_row(
            "SELECT e.public_event_id, e.receipt_id, r.payload
             FROM synaptic_events e
             JOIN memory_receipts r ON r.receipt_id = e.receipt_id
             WHERE e.event_id = ?1",
            params![internal_event_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?
    {
        let receipt: Receipt = serde_json::from_str(&payload)
            .map_err(|error| StorageError::Init(format!("receipt deserialize: {error}")))?;
        let row = load_v2_event_tx(tx, &internal_event_id, transaction_at_ms)?;
        let captured_count = receipt.mutations.len();
        return Ok((
            DurableSynapticCapture {
                event_id: public_event_id,
                receipt: Receipt {
                    receipt_id,
                    ..receipt
                },
                captured_count,
                reused_existing: true,
            },
            false,
            row,
        ));
    }

    let public_event_id = random_public_id("sevt");
    let public_receipt_id = random_public_id("r_syn");
    // The root still records and evaluates its backward evidence. Only the
    // forward path is unavailable once its window has elapsed.
    let event_state = if window_to_ms < transaction_at_ms {
        "closed"
    } else {
        "open"
    };
    tx.execute(
        "INSERT INTO synaptic_events
             (event_id, trigger_memory_id, event_type, occurred_at_ms,
              window_from_ms, window_to_ms, strength, algorithm_version,
              receipt_id, recorded_at, public_event_id, event_state,
              tag_lifetime_hours, minimum_tag_strength,
              minimum_association_score, maximum_captures, decay_function,
              context_threshold, context_algorithm_version,
              signal_snapshot_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                 ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
        params![
            internal_event_id,
            trigger_memory_id,
            event.event_type,
            occurred_at_ms,
            window_from_ms,
            window_to_ms,
            strength,
            SYNAPTIC_CAPTURE_ALGORITHM_V2,
            public_receipt_id,
            event.occurred_at.to_rfc3339(),
            public_event_id,
            event_state,
            event.policy.tag_lifetime_hours,
            event.policy.minimum_tag_strength,
            minimum_association_score,
            event.policy.maximum_captures as i64,
            decay_label,
            SYNAPTIC_CONTEXT_THRESHOLD_V1,
            SYNAPTIC_CONTEXT_ALGORITHM_V1,
            snapshot_json,
        ],
    )?;

    let event_row = load_v2_event_tx(tx, &internal_event_id, transaction_at_ms)?;
    let (receipt, captured_count) = evaluate_backward_v2_tx(tx, &event_row, transaction_at_ms)?;
    insert_receipt_tx(tx, &receipt, event.occurred_at)?;

    Ok((
        DurableSynapticCapture {
            event_id: public_event_id,
            receipt,
            captured_count,
            reused_existing: false,
        },
        true,
        event_row,
    ))
}

fn load_v2_event_tx(
    tx: &Transaction<'_>,
    internal_event_id: &str,
    transaction_at_ms: i64,
) -> Result<V2EventRow> {
    tx.query_row(
        "SELECT e.event_id, e.public_event_id, e.trigger_memory_id,
                e.event_type, e.occurred_at_ms, e.window_from_ms,
                e.window_to_ms, e.strength, e.receipt_id,
                e.tag_lifetime_hours, e.minimum_tag_strength,
                e.minimum_association_score, e.maximum_captures,
                e.decay_function, e.context_threshold,
                n.suppression_count,
                CASE WHEN n.superseded_by IS NULL
                       AND (n.valid_from IS NULL OR unixepoch(n.valid_from) * 1000 <= e.occurred_at_ms)
                       AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > e.occurred_at_ms)
                       AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?2)
                     THEN 1 ELSE 0 END
         FROM synaptic_events e
         JOIN knowledge_nodes n ON n.id = e.trigger_memory_id
         WHERE e.event_id = ?1",
        params![internal_event_id, transaction_at_ms],
        |row| {
            let maximum_captures: i64 = row.get(12)?;
            Ok(V2EventRow {
                internal_event_id: row.get(0)?,
                public_event_id: row.get(1)?,
                trigger_memory_id: row.get(2)?,
                event_type: row.get(3)?,
                occurred_at_ms: row.get(4)?,
                window_from_ms: row.get(5)?,
                window_to_ms: row.get(6)?,
                strength: row.get(7)?,
                parent_receipt_id: row.get(8)?,
                tag_lifetime_hours: row.get(9)?,
                minimum_tag_strength: row.get(10)?,
                minimum_association_score: row.get(11)?,
                maximum_captures: maximum_captures.max(0) as usize,
                decay_function: parse_decay_function(&row.get::<_, String>(13)?),
                context_threshold: row.get(14)?,
                trigger_suppression_count: row.get(15)?,
                trigger_currently_valid: row.get::<_, i64>(16)? != 0,
            })
        },
    )
    .map_err(Into::into)
}

fn evaluate_backward_v2_tx(
    tx: &Transaction<'_>,
    event: &V2EventRow,
    transaction_at_ms: i64,
) -> Result<(Receipt, usize)> {
    let mut stmt = tx.prepare(
        "SELECT t.tag_id, t.memory_id, t.created_at_ms, t.initial_strength,
                n.suppression_count,
                CASE WHEN n.superseded_by IS NULL
                       AND (n.valid_from IS NULL OR unixepoch(n.valid_from) * 1000 <= ?1)
                       AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?1)
                       AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?4)
                     THEN 1 ELSE 0 END
         FROM synaptic_tags t
         JOIN knowledge_nodes n ON n.id = t.memory_id
         WHERE t.state = 'active'
           AND t.created_at_ms BETWEEN ?2 AND ?1
           AND t.memory_id <> ?3
         ORDER BY t.created_at_ms ASC, t.tag_id ASC",
    )?;
    let rows = stmt.query_map(
        params![
            event.occurred_at_ms,
            event.window_from_ms,
            event.trigger_memory_id,
            transaction_at_ms,
        ],
        |row| {
            Ok(V2TagRow {
                tag_id: row.get(0)?,
                memory_id: row.get(1)?,
                created_at_ms: row.get(2)?,
                initial_strength: row.get(3)?,
                suppression_count: row.get(4)?,
                currently_valid: row.get::<_, i64>(5)? != 0,
            })
        },
    )?;
    let mut candidates = Vec::new();
    for row in rows {
        let tag = row?;
        let distance_hours = (event.occurred_at_ms - tag.created_at_ms) as f64 / 3_600_000.0;
        let temporal_score = event.decay_function.apply(
            1.0,
            distance_hours.max(0.0),
            ((event.occurred_at_ms - event.window_from_ms) as f64 / 3_600_000.0).max(f64::EPSILON),
        );
        let tag_strength = event.decay_function.apply(
            finite_unit(tag.initial_strength),
            distance_hours.max(0.0),
            event.tag_lifetime_hours.max(f64::EPSILON),
        );
        let context = compute_synaptic_context_tx(tx, &event.trigger_memory_id, &tag.memory_id)?;
        let association_score = temporal_score * tag_strength * event.strength * context.score;
        candidates.push((
            tag,
            temporal_score,
            tag_strength,
            context,
            association_score,
        ));
    }
    drop(stmt);

    candidates.sort_by(|a, b| {
        b.4.total_cmp(&a.4)
            .then_with(|| a.0.created_at_ms.cmp(&b.0.created_at_ms))
            .then_with(|| a.0.tag_id.cmp(&b.0.tag_id))
    });

    let mut evidence_candidates = Vec::with_capacity(candidates.len());
    let mut retrieved = Vec::new();
    let mut suppressed = Vec::new();
    let mut activation_path = Vec::new();
    let mut trust_scores = Vec::new();
    let mut mutations = Vec::new();
    let mut captured_count = 0usize;

    for (rank, (tag, temporal_score, tag_strength, context, association_score)) in
        candidates.into_iter().enumerate()
    {
        let evidence_slot = format!("candidate_{}", rank + 1);
        let (disposition, reason_code, public_memory_id, strength_change) =
            if tag.suppression_count > 0 {
                suppressed.push(SuppressedReceiptEntry::new(
                    evidence_slot.clone(),
                    SuppressReason::Privacy,
                ));
                (
                    SynapticCaptureDisposition::WithheldSuppressed,
                    "withheld_suppressed",
                    None,
                    None,
                )
            } else if !tag.currently_valid {
                suppressed.push(SuppressedReceiptEntry::new(
                    evidence_slot.clone(),
                    SuppressReason::Contradicted,
                ));
                (
                    SynapticCaptureDisposition::WithheldInvalid,
                    "withheld_invalid",
                    None,
                    None,
                )
            } else if context.score < event.context_threshold {
                (
                    SynapticCaptureDisposition::ContextMismatch,
                    "context_mismatch",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if tag_strength < event.minimum_tag_strength
                || association_score < event.minimum_association_score
            {
                (
                    SynapticCaptureDisposition::BelowThreshold,
                    "below_threshold",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if captured_count >= event.maximum_captures {
                (
                    SynapticCaptureDisposition::LostCompetition,
                    "lost_competition",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if let Some(change) = promote_memory_tx(
                tx,
                &tag.memory_id,
                event.occurred_at_ms,
                event.occurred_at_ms,
                transaction_at_ms,
            )? {
                tx.execute(
                    "UPDATE synaptic_tags SET state = 'captured',
                         capture_event_id = ?1, captured_at_ms = ?2
                     WHERE tag_id = ?3 AND state = 'active'",
                    params![event.internal_event_id, event.occurred_at_ms, tag.tag_id],
                )?;
                captured_count += 1;
                retrieved.push(tag.memory_id.clone());
                trust_scores.push(change.retention_strength.before);
                activation_path.push(format!(
                    "{} --[synaptic_capture]--> {}",
                    tag.memory_id, event.trigger_memory_id
                ));
                mutations.push(ReceiptMutation {
                    id: tag.memory_id.clone(),
                    kind: "synaptic_capture".into(),
                    note: Some("Evidence-backed temporal/context association".into()),
                });
                (
                    SynapticCaptureDisposition::Captured,
                    "captured",
                    Some(tag.memory_id.clone()),
                    Some(change),
                )
            } else {
                suppressed.push(SuppressedReceiptEntry::new(
                    evidence_slot.clone(),
                    SuppressReason::Privacy,
                ));
                (
                    SynapticCaptureDisposition::WithheldInvalid,
                    "guarded_mutation_rejected",
                    None,
                    None,
                )
            };

        insert_capture_item_tx(
            tx,
            event,
            &tag,
            &event.parent_receipt_id,
            "backward",
            temporal_score,
            &context,
            association_score,
            rank + 1,
            disposition,
            reason_code,
            strength_change.as_ref(),
            event.occurred_at_ms,
        )?;

        evidence_candidates.push(SynapticCaptureCandidate {
            memory_id: public_memory_id,
            evidence_slot,
            encoded_at: millis_to_datetime(tag.created_at_ms),
            temporal_distance_hours: (event.occurred_at_ms - tag.created_at_ms) as f64
                / 3_600_000.0,
            capture_probability: temporal_score,
            tag_strength_at_evaluation: tag_strength,
            capture_score: association_score,
            evaluation_direction: Some("backward".into()),
            temporal_score: Some(temporal_score),
            context_score: Some(context.score),
            context_method: Some(context.method),
            association_score: Some(association_score),
            competition_rank: Some(rank + 1),
            algorithm_version: Some(SYNAPTIC_CAPTURE_ALGORITHM_V2.into()),
            reason_code: Some(reason_code.into()),
            disposition,
            reason: Some(reason_message(reason_code).into()),
            strength_change,
        });
    }

    let evidence = SynapticCaptureEvidence {
        schema: SYNAPTIC_CAPTURE_SCHEMA_V2.into(),
        schema_version: 2,
        algorithm_version: SYNAPTIC_CAPTURE_ALGORITHM_V2.into(),
        receipt_role: Some("root".into()),
        parent_receipt_id: None,
        evaluation_direction: Some("backward".into()),
        trigger: SynapticCaptureTrigger {
            event_id: event.public_event_id.clone(),
            memory_id: event.trigger_memory_id.clone(),
            event_type: event.event_type.clone(),
            occurred_at: millis_to_datetime(event.occurred_at_ms),
            importance_score: event.strength,
        },
        capture_window: capture_window_evidence(event, "backward"),
        candidates: evidence_candidates,
        claim_boundary: SYNAPTIC_CAPTURE_CLAIM_BOUNDARY.into(),
    };
    let recorded_at = millis_to_datetime(event.occurred_at_ms);
    let mut receipt = Receipt::build_with_unique(
        recorded_at,
        &event.public_event_id,
        &event.parent_receipt_id,
        retrieved,
        suppressed,
        activation_path,
        &trust_scores,
        mutations,
    )
    .with_evidence(ReceiptEvidence::SynapticCapture(evidence));
    receipt.receipt_id = event.parent_receipt_id.clone();
    Ok((receipt, captured_count))
}

fn reconcile_forward_tag_tx(
    tx: &Transaction<'_>,
    tag_id: &str,
    transaction_at_ms: i64,
) -> Result<Vec<PairReceiptRecord>> {
    let mut records = Vec::new();
    {
        let mut stmt = tx.prepare(
            "SELECT e.public_event_id, i.receipt_id, i.disposition
             FROM synaptic_capture_items i
             JOIN synaptic_events e ON e.event_id = i.event_id
             WHERE i.tag_id = ?1 AND i.evaluation_direction = 'forward'
             ORDER BY i.recorded_at ASC, e.event_id ASC",
        )?;
        let existing = stmt.query_map(params![tag_id], |row| {
            Ok(PairReceiptRecord {
                event_id: row.get(0)?,
                receipt_id: row.get(1)?,
                disposition: parse_disposition(&row.get::<_, String>(2)?),
                reused_existing: true,
            })
        })?;
        for row in existing {
            records.push(row?);
        }
    }

    let tag = tx
        .query_row(
            "SELECT t.tag_id, t.memory_id, t.created_at_ms, t.initial_strength,
                    n.suppression_count,
                    CASE WHEN n.superseded_by IS NULL
                           AND (n.valid_from IS NULL OR unixepoch(n.valid_from) * 1000 <= t.created_at_ms)
                           AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > t.created_at_ms)
                           AND (n.valid_until IS NULL OR unixepoch(n.valid_until) * 1000 > ?2)
                         THEN 1 ELSE 0 END
             FROM synaptic_tags t
             JOIN knowledge_nodes n ON n.id = t.memory_id
             WHERE t.tag_id = ?1 AND t.state = 'active'",
            params![tag_id, transaction_at_ms],
            |row| {
                Ok(V2TagRow {
                    tag_id: row.get(0)?,
                    memory_id: row.get(1)?,
                    created_at_ms: row.get(2)?,
                    initial_strength: row.get(3)?,
                    suppression_count: row.get(4)?,
                    currently_valid: row.get::<_, i64>(5)? != 0,
                })
            },
        )
        .optional()?;
    let Some(tag) = tag else {
        return Ok(records);
    };

    let mut events = Vec::new();
    {
        let mut stmt = tx.prepare(
            "SELECT e.event_id
             FROM synaptic_events e
             WHERE e.algorithm_version = ?1
               AND e.event_state = 'open'
               AND e.occurred_at_ms < ?2 AND ?2 <= e.window_to_ms
               AND e.trigger_memory_id <> ?3
               AND NOT EXISTS (
                   SELECT 1 FROM synaptic_capture_items i
                   WHERE i.event_id = e.event_id AND i.tag_id = ?4
               )
             ORDER BY e.occurred_at_ms DESC, e.event_id ASC",
        )?;
        let rows = stmt.query_map(
            params![
                SYNAPTIC_CAPTURE_ALGORITHM_V2,
                tag.created_at_ms,
                tag.memory_id,
                tag.tag_id,
            ],
            |row| row.get::<_, String>(0),
        )?;
        for row in rows {
            events.push(load_v2_event_tx(tx, &row?, transaction_at_ms)?);
        }
    }

    let mut scored = Vec::with_capacity(events.len());
    for event in events {
        let distance_hours = (tag.created_at_ms - event.occurred_at_ms) as f64 / 3_600_000.0;
        let forward_hours = (event.window_to_ms - event.occurred_at_ms) as f64 / 3_600_000.0;
        let temporal_score = event.decay_function.apply(
            1.0,
            distance_hours.max(0.0),
            forward_hours.max(f64::EPSILON),
        );
        let tag_elapsed_hours = (transaction_at_ms - tag.created_at_ms).max(0) as f64 / 3_600_000.0;
        let tag_strength = event.decay_function.apply(
            finite_unit(tag.initial_strength),
            tag_elapsed_hours,
            event.tag_lifetime_hours.max(f64::EPSILON),
        );
        let context = compute_synaptic_context_tx(tx, &event.trigger_memory_id, &tag.memory_id)?;
        let association_score = temporal_score * tag_strength * event.strength * context.score;
        scored.push((
            event,
            temporal_score,
            tag_strength,
            context,
            association_score,
        ));
    }
    scored.sort_by(|a, b| {
        b.4.total_cmp(&a.4)
            .then_with(|| b.0.occurred_at_ms.cmp(&a.0.occurred_at_ms))
            .then_with(|| a.0.internal_event_id.cmp(&b.0.internal_event_id))
    });

    let mut winner_selected = false;
    for (rank, (event, temporal_score, tag_strength, context, association_score)) in
        scored.into_iter().enumerate()
    {
        let prior_captures: i64 = tx.query_row(
            "SELECT COUNT(*) FROM synaptic_capture_items
             WHERE event_id = ?1 AND disposition = 'captured'",
            params![event.internal_event_id],
            |row| row.get(0),
        )?;
        let evidence_slot = format!("pair_{}", &tag.tag_id[tag.tag_id.len().saturating_sub(8)..]);
        let (disposition, reason_code, public_memory_id, strength_change) =
            if event.trigger_suppression_count > 0 {
                (
                    SynapticCaptureDisposition::WithheldSuppressed,
                    "withheld_suppressed",
                    None,
                    None,
                )
            } else if !event.trigger_currently_valid
                || tag.suppression_count > 0
                || !tag.currently_valid
            {
                (
                    SynapticCaptureDisposition::WithheldInvalid,
                    "withheld_invalid",
                    None,
                    None,
                )
            } else if context.score < event.context_threshold {
                (
                    SynapticCaptureDisposition::ContextMismatch,
                    "context_mismatch",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if tag_strength < event.minimum_tag_strength
                || association_score < event.minimum_association_score
            {
                (
                    SynapticCaptureDisposition::BelowThreshold,
                    "below_threshold",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if winner_selected || prior_captures as usize >= event.maximum_captures {
                (
                    SynapticCaptureDisposition::LostCompetition,
                    "lost_competition",
                    Some(tag.memory_id.clone()),
                    None,
                )
            } else if let Some(change) = promote_memory_tx(
                tx,
                &tag.memory_id,
                tag.created_at_ms,
                tag.created_at_ms,
                transaction_at_ms,
            )? {
                let changed = tx.execute(
                    "UPDATE synaptic_tags SET state = 'captured',
                         capture_event_id = ?1, captured_at_ms = ?2
                     WHERE tag_id = ?3 AND state = 'active'",
                    params![event.internal_event_id, tag.created_at_ms, tag.tag_id],
                )?;
                if changed != 1 {
                    return Err(StorageError::Init(
                        "forward capture lost its active-tag guard".into(),
                    ));
                }
                winner_selected = true;
                if prior_captures as usize + 1 >= event.maximum_captures {
                    tx.execute(
                        "UPDATE synaptic_events SET event_state = 'closed'
                         WHERE event_id = ?1",
                        params![event.internal_event_id],
                    )?;
                }
                (
                    SynapticCaptureDisposition::Captured,
                    "captured",
                    Some(tag.memory_id.clone()),
                    Some(change),
                )
            } else {
                (
                    SynapticCaptureDisposition::WithheldInvalid,
                    "guarded_mutation_rejected",
                    None,
                    None,
                )
            };

        let child_receipt_id = random_public_id("r_syn_pair");
        insert_capture_item_tx(
            tx,
            &event,
            &tag,
            &child_receipt_id,
            "forward",
            temporal_score,
            &context,
            association_score,
            rank + 1,
            disposition,
            reason_code,
            strength_change.as_ref(),
            tag.created_at_ms,
        )?;
        let receipt = build_pair_receipt(
            &event,
            &tag,
            &child_receipt_id,
            &evidence_slot,
            temporal_score,
            tag_strength,
            &context,
            association_score,
            rank + 1,
            disposition,
            reason_code,
            public_memory_id,
            strength_change,
        );
        insert_receipt_tx(tx, &receipt, millis_to_datetime(tag.created_at_ms))?;
        records.push(PairReceiptRecord {
            event_id: event.public_event_id,
            receipt_id: child_receipt_id,
            disposition,
            reused_existing: false,
        });
    }
    Ok(records)
}

fn promote_memory_tx(
    tx: &Transaction<'_>,
    memory_id: &str,
    semantic_at_ms: i64,
    recorded_at_ms: i64,
    transaction_at_ms: i64,
) -> Result<Option<SynapticStrengthChange>> {
    let before = tx
        .query_row(
            "SELECT retrieval_strength, retention_strength, stability
             FROM knowledge_nodes WHERE id = ?1",
            params![memory_id],
            |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((retrieval_before, retention_before, stability_before)) = before else {
        return Ok(None);
    };
    let recorded_at = millis_to_datetime(recorded_at_ms).to_rfc3339();
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
        params![recorded_at, memory_id, semantic_at_ms, transaction_at_ms],
    )?;
    if changed != 1 {
        return Ok(None);
    }
    let (retrieval_after, retention_after, stability_after) = tx.query_row(
        "SELECT retrieval_strength, retention_strength, stability
         FROM knowledge_nodes WHERE id = ?1",
        params![memory_id],
        |row| {
            Ok((
                row.get::<_, f64>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
            ))
        },
    )?;
    tx.execute(
        "INSERT INTO memory_access_log (node_id, access_type, accessed_at)
         VALUES (?1, 'synaptic_capture', ?2)",
        params![memory_id, recorded_at],
    )?;
    Ok(Some(SynapticStrengthChange {
        retrieval_strength: StrengthDelta {
            before: retrieval_before,
            after: retrieval_after,
        },
        retention_strength: StrengthDelta {
            before: retention_before,
            after: retention_after,
        },
        stability: StrengthDelta {
            before: stability_before,
            after: stability_after,
        },
    }))
}

#[allow(clippy::too_many_arguments)]
fn insert_capture_item_tx(
    tx: &Transaction<'_>,
    event: &V2EventRow,
    tag: &V2TagRow,
    receipt_id: &str,
    direction: &str,
    temporal_score: f64,
    context: &ContextEvidence,
    association_score: f64,
    competition_rank: usize,
    disposition: SynapticCaptureDisposition,
    reason_code: &str,
    strength_change: Option<&SynapticStrengthChange>,
    recorded_at_ms: i64,
) -> Result<()> {
    let before: Option<(f64, f64, f64)> = tx
        .query_row(
            "SELECT retrieval_strength, retention_strength, stability
             FROM knowledge_nodes WHERE id = ?1",
            params![tag.memory_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let (retrieval_before, retention_before, stability_before) = strength_change
        .map(|change| {
            (
                Some(change.retrieval_strength.before),
                Some(change.retention_strength.before),
                Some(change.stability.before),
            )
        })
        .unwrap_or_else(|| {
            before
                .map(|(r, retention, stability)| (Some(r), Some(retention), Some(stability)))
                .unwrap_or((None, None, None))
        });
    tx.execute(
        "INSERT INTO synaptic_capture_items
             (event_id, tag_id, memory_id, evidence_slot, receipt_id,
              encoded_at_ms, temporal_distance_hours, capture_probability,
              tag_strength_at_evaluation, capture_score, disposition, reason,
              retrieval_before, retrieval_after, retention_before,
              retention_after, stability_before, stability_after, recorded_at,
              evaluation_direction, temporal_score, context_score,
              context_method, association_score, competition_rank,
              algorithm_version, reason_code)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                 ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23,
                 ?24, ?25, ?26, ?27)",
        params![
            event.internal_event_id,
            tag.tag_id,
            tag.memory_id,
            format!("pair_{}", &tag.tag_id[tag.tag_id.len().saturating_sub(8)..]),
            receipt_id,
            tag.created_at_ms,
            (tag.created_at_ms - event.occurred_at_ms).unsigned_abs() as f64 / 3_600_000.0,
            temporal_score,
            finite_unit(tag.initial_strength),
            association_score,
            disposition_label(disposition),
            reason_message(reason_code),
            retrieval_before,
            strength_change.map(|change| change.retrieval_strength.after),
            retention_before,
            strength_change.map(|change| change.retention_strength.after),
            stability_before,
            strength_change.map(|change| change.stability.after),
            millis_to_datetime(recorded_at_ms).to_rfc3339(),
            direction,
            temporal_score,
            context.score,
            context.method,
            association_score,
            competition_rank as i64,
            SYNAPTIC_CAPTURE_ALGORITHM_V2,
            reason_code,
        ],
    )?;
    Ok(())
}

fn insert_receipt_tx(
    tx: &Transaction<'_>,
    receipt: &Receipt,
    created_at: DateTime<Utc>,
) -> Result<()> {
    let payload = serde_json::to_string(receipt)
        .map_err(|error| StorageError::Init(format!("receipt serialize: {error}")))?;
    tx.execute(
        "INSERT INTO memory_receipts
             (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
              trust_floor, decay_risk, payload, created_at)
         VALUES (?1, NULL, 'smart_ingest', NULL, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            receipt.receipt_id,
            receipt.retrieved.len() as i64,
            receipt.suppressed.len() as i64,
            receipt.trust_floor,
            receipt.decay_risk.as_str(),
            payload,
            created_at.to_rfc3339(),
        ],
    )?;
    Ok(())
}

fn capture_window_evidence(event: &V2EventRow, direction: &str) -> SynapticCaptureWindow {
    SynapticCaptureWindow {
        evaluation_direction: direction.into(),
        backward_hours: (event.occurred_at_ms - event.window_from_ms) as f64 / 3_600_000.0,
        forward_hours: (event.window_to_ms - event.occurred_at_ms) as f64 / 3_600_000.0,
        tag_lifetime_hours: event.tag_lifetime_hours,
        minimum_tag_strength: event.minimum_tag_strength,
        minimum_association_score: Some(event.minimum_association_score),
        maximum_captures: event.maximum_captures,
        decay_function: decay_function_label(event.decay_function).into(),
        context_threshold: Some(event.context_threshold),
        context_algorithm_version: Some(SYNAPTIC_CONTEXT_ALGORITHM_V1.into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pair_receipt(
    event: &V2EventRow,
    tag: &V2TagRow,
    receipt_id: &str,
    evidence_slot: &str,
    temporal_score: f64,
    tag_strength: f64,
    context: &ContextEvidence,
    association_score: f64,
    competition_rank: usize,
    disposition: SynapticCaptureDisposition,
    reason_code: &str,
    public_memory_id: Option<String>,
    strength_change: Option<SynapticStrengthChange>,
) -> Receipt {
    let mut retrieved = Vec::new();
    let mut suppressed = Vec::new();
    let mut mutations = Vec::new();
    let mut trust_scores = Vec::new();
    if disposition == SynapticCaptureDisposition::Captured {
        retrieved.push(tag.memory_id.clone());
        if let Some(change) = &strength_change {
            trust_scores.push(change.retention_strength.before);
        }
        mutations.push(ReceiptMutation {
            id: tag.memory_id.clone(),
            kind: "synaptic_capture".into(),
            note: Some("Evidence-backed temporal/context association".into()),
        });
    } else if matches!(
        disposition,
        SynapticCaptureDisposition::WithheldSuppressed
            | SynapticCaptureDisposition::WithheldInvalid
    ) {
        suppressed.push(SuppressedReceiptEntry::new(
            evidence_slot.to_string(),
            SuppressReason::Privacy,
        ));
    }
    let trigger_memory_id = if event.trigger_suppression_count == 0 && event.trigger_currently_valid
    {
        event.trigger_memory_id.clone()
    } else {
        "trigger_1".into()
    };
    let evidence = SynapticCaptureEvidence {
        schema: SYNAPTIC_CAPTURE_SCHEMA_V2.into(),
        schema_version: 2,
        algorithm_version: SYNAPTIC_CAPTURE_ALGORITHM_V2.into(),
        receipt_role: Some("pair".into()),
        parent_receipt_id: Some(event.parent_receipt_id.clone()),
        evaluation_direction: Some("forward".into()),
        trigger: SynapticCaptureTrigger {
            event_id: event.public_event_id.clone(),
            memory_id: trigger_memory_id,
            event_type: event.event_type.clone(),
            occurred_at: millis_to_datetime(event.occurred_at_ms),
            importance_score: event.strength,
        },
        capture_window: capture_window_evidence(event, "forward"),
        candidates: vec![SynapticCaptureCandidate {
            memory_id: public_memory_id,
            evidence_slot: evidence_slot.into(),
            encoded_at: millis_to_datetime(tag.created_at_ms),
            temporal_distance_hours: (tag.created_at_ms - event.occurred_at_ms) as f64
                / 3_600_000.0,
            capture_probability: temporal_score,
            tag_strength_at_evaluation: tag_strength,
            capture_score: association_score,
            evaluation_direction: Some("forward".into()),
            temporal_score: Some(temporal_score),
            context_score: Some(context.score),
            context_method: Some(context.method.clone()),
            association_score: Some(association_score),
            competition_rank: Some(competition_rank),
            algorithm_version: Some(SYNAPTIC_CAPTURE_ALGORITHM_V2.into()),
            reason_code: Some(reason_code.into()),
            disposition,
            reason: Some(reason_message(reason_code).into()),
            strength_change,
        }],
        claim_boundary: SYNAPTIC_CAPTURE_CLAIM_BOUNDARY.into(),
    };
    let mut receipt = Receipt::build_with_unique(
        millis_to_datetime(tag.created_at_ms),
        &event.public_event_id,
        receipt_id,
        retrieved,
        suppressed,
        Vec::new(),
        &trust_scores,
        mutations,
    )
    .with_evidence(ReceiptEvidence::SynapticCapture(evidence));
    receipt.receipt_id = receipt_id.into();
    receipt
}

#[derive(Debug)]
struct NodeContextRow {
    content: String,
    tags: Vec<String>,
    source_project: Option<String>,
    embedding: Option<(Vec<f32>, String, i64)>,
}

fn compute_synaptic_context_tx(
    tx: &Transaction<'_>,
    first_memory_id: &str,
    second_memory_id: &str,
) -> Result<ContextEvidence> {
    let first = load_node_context_tx(tx, first_memory_id)?;
    let second = load_node_context_tx(tx, second_memory_id)?;
    let mut channels: Vec<(&str, f64)> = Vec::new();

    if let (Some((left, left_model, left_dim)), Some((right, right_model, right_dim))) =
        (&first.embedding, &second.embedding)
        && left_model == right_model
        && left_dim == right_dim
        && !left.is_empty()
        && left.len() == right.len()
    {
        channels.push(("semantic_cosine", cosine_unit(left, right)));
    }

    let graph_strength = tx
        .query_row(
            "SELECT MAX(strength) FROM memory_connections
             WHERE ((source_id = ?1 AND target_id = ?2)
                 OR (source_id = ?2 AND target_id = ?1))
               AND link_type NOT IN ('temporal', 'sequential')",
            params![first_memory_id, second_memory_id],
            |row| row.get::<_, Option<f64>>(0),
        )?
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    if graph_strength > 0.0 {
        channels.push(("associative_graph", graph_strength));
    }

    let tag_overlap = overlap_coefficient(
        first.tags.iter().map(String::as_str),
        second.tags.iter().map(String::as_str),
    );
    if tag_overlap > 0.0 {
        channels.push(("tag_overlap", tag_overlap));
    }

    let first_terms = lexical_terms(&first.content);
    let second_terms = lexical_terms(&second.content);
    let lexical_overlap = overlap_coefficient(
        first_terms.iter().map(String::as_str),
        second_terms.iter().map(String::as_str),
    );
    if lexical_overlap > 0.0 {
        channels.push(("lexical_overlap", lexical_overlap));
    }

    if first
        .source_project
        .as_deref()
        .filter(|value| !value.is_empty())
        .is_some_and(|value| second.source_project.as_deref() == Some(value))
    {
        channels.push(("scoped_project_match", 1.0));
    }

    let (method, score) = channels
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or(("none", 0.0));
    Ok(ContextEvidence {
        score: score.clamp(0.0, 1.0),
        method: method.into(),
    })
}

fn load_node_context_tx(tx: &Transaction<'_>, memory_id: &str) -> Result<NodeContextRow> {
    tx.query_row(
        "SELECT n.content, n.tags, n.source_project,
                e.embedding, e.model, e.dimensions
         FROM knowledge_nodes n
         LEFT JOIN node_embeddings e ON e.node_id = n.id
         WHERE n.id = ?1",
        params![memory_id],
        |row| {
            let tags_json: String = row.get(1)?;
            let blob: Option<Vec<u8>> = row.get(3)?;
            let model: Option<String> = row.get(4)?;
            let dimensions: Option<i64> = row.get(5)?;
            let embedding = match (blob, model, dimensions) {
                (Some(blob), Some(model), Some(dimensions)) if blob.len() % 4 == 0 => Some((
                    blob.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk))
                        .collect(),
                    model,
                    dimensions,
                )),
                _ => None,
            };
            Ok(NodeContextRow {
                content: row.get(0)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                source_project: row.get(2)?,
                embedding,
            })
        },
    )
    .map_err(Into::into)
}

fn lexical_terms(content: &str) -> BTreeSet<String> {
    content
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter_map(|term| {
            let normalized = term.to_lowercase();
            (normalized.len() >= 3).then_some(normalized)
        })
        .collect()
}

fn overlap_coefficient<'a>(
    left: impl Iterator<Item = &'a str>,
    right: impl Iterator<Item = &'a str>,
) -> f64 {
    let left: BTreeSet<String> = left.map(str::to_lowercase).collect();
    let right: BTreeSet<String> = right.map(str::to_lowercase).collect();
    let denominator = left.len().min(right.len());
    if denominator == 0 {
        return 0.0;
    }
    left.intersection(&right).count() as f64 / denominator as f64
}

fn cosine_unit(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        return 0.0;
    }
    // The capture threshold is expressed on a non-negative cosine scale:
    // orthogonal vectors are zero context, and only positively aligned vectors
    // can qualify. Mapping `[-1, 1]` into `[0, 1]` would make an unrelated
    // orthogonal pair score 0.5, silently bypassing the 0.25 context gate.
    (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 1.0)
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn decay_function_label(value: DecayFunction) -> &'static str {
    match value {
        DecayFunction::Exponential => "exponential",
        DecayFunction::Linear => "linear",
        DecayFunction::Power => "power",
        DecayFunction::Logarithmic => "logarithmic",
    }
}

fn parse_decay_function(value: &str) -> DecayFunction {
    match value {
        "linear" => DecayFunction::Linear,
        "power" => DecayFunction::Power,
        "logarithmic" => DecayFunction::Logarithmic,
        _ => DecayFunction::Exponential,
    }
}

fn reason_message(code: &str) -> &'static str {
    match code {
        "captured" => "eligible pair won deterministic competition",
        "below_threshold" => "association score did not meet the frozen threshold",
        "context_mismatch" => "context evidence did not meet the frozen gate",
        "lost_competition" => "eligible pair lost deterministic competition",
        "withheld_suppressed" => "active suppression forbids promotion and stable-id disclosure",
        "withheld_invalid" => "memory is superseded or outside its valid-time interval",
        "guarded_mutation_rejected" => "guarded promotion rejected the current memory state",
        _ => "synaptic pair evaluation completed",
    }
}

fn parse_disposition(value: &str) -> SynapticCaptureDisposition {
    match value {
        "captured" => SynapticCaptureDisposition::Captured,
        "context_mismatch" => SynapticCaptureDisposition::ContextMismatch,
        "lost_competition" => SynapticCaptureDisposition::LostCompetition,
        "withheld_suppressed" => SynapticCaptureDisposition::WithheldSuppressed,
        "withheld_invalid" => SynapticCaptureDisposition::WithheldInvalid,
        _ => SynapticCaptureDisposition::BelowThreshold,
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
        SynapticCaptureDisposition::ContextMismatch => "context_mismatch",
        SynapticCaptureDisposition::WithheldSuppressed => "withheld_suppressed",
        SynapticCaptureDisposition::WithheldInvalid => "withheld_invalid",
        SynapticCaptureDisposition::LostCompetition => "lost_competition",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IngestInput, Storage};
    // Only the blocker fixture drives a transaction by hand; production
    // writers go through SqliteMemoryStore::begin_write_transaction.
    use rusqlite::TransactionBehavior;
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

    fn v2_event(at: DateTime<Utc>) -> SynapticImportanceEvent {
        SynapticImportanceEvent {
            event_type: "novelty_spike".into(),
            occurred_at: at,
            strength: 0.95,
            policy: policy(),
            signal_snapshot: SynapticSignalSnapshot {
                novelty: 0.95,
                arousal: 0.2,
                reward: 0.1,
                attention: 0.8,
                composite: 0.95,
            },
        }
    }

    #[test]
    fn v2_event_restart_then_related_later_tag_captures_once() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("forward_restart.db");
        let event_time = Utc::now() - chrono::Duration::minutes(5);
        let event_id;
        {
            let storage = Storage::new(Some(path.clone())).unwrap();
            let trigger = ingest(&storage, "database retry policy caused incident");
            event_id = trigger.id.clone();
            let root = storage
                .process_synaptic_ingest(&SynapticIngestRequest {
                    memory_id: trigger.id,
                    tag: None,
                    event: Some(v2_event(event_time)),
                })
                .unwrap();
            assert!(root.event.is_some());
            assert!(root.forward_receipts.is_empty());
        }

        let storage = Storage::new(Some(path)).unwrap();
        let later = ingest(&storage, "database retry policy decision");
        let before = storage.demote_memory(&later.id).unwrap();
        let mut tag = SynapticTag::new(&later.id);
        tag.created_at = event_time + chrono::Duration::minutes(1);
        let request = SynapticIngestRequest {
            memory_id: later.id.clone(),
            tag: Some(tag),
            event: None,
        };
        let first = storage.process_synaptic_ingest(&request).unwrap();
        assert_eq!(first.forward_receipts.len(), 1);
        assert_eq!(
            first.forward_receipts[0].disposition,
            SynapticCaptureDisposition::Captured
        );
        assert!(!first.forward_receipts[0].reused_existing);
        let first_receipt_id = first.forward_receipts[0].receipt.receipt_id.clone();
        let after_first = storage.get_node(&later.id).unwrap().unwrap();
        assert!(after_first.retrieval_strength > before.retrieval_strength);

        let retry = storage.process_synaptic_ingest(&request).unwrap();
        assert_eq!(retry.forward_receipts.len(), 1);
        assert!(retry.forward_receipts[0].reused_existing);
        assert_eq!(
            retry.forward_receipts[0].receipt.receipt_id,
            first_receipt_id
        );
        let after_retry = storage.get_node(&later.id).unwrap().unwrap();
        assert_eq!(
            after_retry.retrieval_strength,
            after_first.retrieval_strength
        );

        let trigger_rows: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM synaptic_events WHERE trigger_memory_id = ?1",
                params![event_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(trigger_rows, 1);
    }

    #[test]
    fn v2_unrelated_later_tag_is_context_mismatch_without_mutation() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("context_gate.db"))).unwrap();
        let event_time = Utc::now() - chrono::Duration::minutes(5);
        let trigger = ingest(&storage, "database retry policy incident");
        let trigger_id = trigger.id.clone();
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger_id.clone(),
                tag: None,
                event: Some(v2_event(event_time)),
            })
            .unwrap();

        let unrelated = ingest(&storage, "orchid watercolor palette selection");
        // This is a context-gate unit test, not an embedding-model evaluation.
        // Real model versions and architectures can legitimately assign these
        // phrases different cosine scores across platforms. Remove that
        // environment-dependent channel so the fixture deterministically
        // exercises the no-context fallback on every runner.
        storage
            .writer
            .lock()
            .unwrap()
            .execute(
                "DELETE FROM node_embeddings WHERE node_id IN (?1, ?2)",
                params![trigger_id, &unrelated.id],
            )
            .unwrap();
        let before = storage.get_node(&unrelated.id).unwrap().unwrap();
        let mut tag = SynapticTag::new(&unrelated.id);
        tag.created_at = event_time + chrono::Duration::minutes(1);
        let outcome = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: unrelated.id.clone(),
                tag: Some(tag),
                event: None,
            })
            .unwrap();
        assert_eq!(outcome.forward_receipts.len(), 1);
        assert_eq!(
            outcome.forward_receipts[0].disposition,
            SynapticCaptureDisposition::ContextMismatch
        );
        let after = storage.get_node(&unrelated.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
        assert!(
            serde_json::to_string(&outcome.forward_receipts[0].receipt)
                .unwrap()
                .contains("context_mismatch")
        );
    }

    #[test]
    fn v2_two_open_events_compete_for_one_later_tag() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("competition.db"))).unwrap();
        let base = Utc::now() - chrono::Duration::minutes(10);
        let older = ingest(&storage, "database retry policy incident older");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: older.id,
                tag: None,
                event: Some(v2_event(base)),
            })
            .unwrap();
        let newer = ingest(&storage, "database retry policy incident newer");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: newer.id,
                tag: None,
                event: Some(v2_event(base + chrono::Duration::minutes(4))),
            })
            .unwrap();

        let candidate = ingest(&storage, "database retry policy decision");
        let mut tag = SynapticTag::new(&candidate.id);
        tag.created_at = base + chrono::Duration::minutes(5);
        let outcome = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id,
                tag: Some(tag),
                event: None,
            })
            .unwrap();
        assert_eq!(outcome.forward_receipts.len(), 2);
        assert_eq!(
            outcome
                .forward_receipts
                .iter()
                .filter(|receipt| receipt.disposition == SynapticCaptureDisposition::Captured)
                .count(),
            1
        );
        assert_eq!(
            outcome
                .forward_receipts
                .iter()
                .filter(|receipt| {
                    receipt.disposition == SynapticCaptureDisposition::LostCompetition
                })
                .count(),
            1
        );
    }

    #[test]
    fn v2_tag_first_then_backdated_event_reconciles_uncaptured_tag() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("tag_first.db"))).unwrap();
        let base = Utc::now() - chrono::Duration::minutes(5);
        let candidate = ingest(&storage, "database retry policy decision");
        let before = storage.demote_memory(&candidate.id).unwrap();
        let mut tag = SynapticTag::new(&candidate.id);
        tag.created_at = base + chrono::Duration::minutes(1);
        let tag_only = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id.clone(),
                tag: Some(tag),
                event: None,
            })
            .unwrap();
        assert!(tag_only.forward_receipts.is_empty());

        let trigger = ingest(&storage, "database retry policy caused incident");
        let event_later = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id,
                tag: None,
                event: Some(v2_event(base)),
            })
            .unwrap();
        assert_eq!(event_later.forward_receipts.len(), 1);
        assert_eq!(
            event_later.forward_receipts[0].disposition,
            SynapticCaptureDisposition::Captured
        );
        let after = storage.get_node(&candidate.id).unwrap().unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
    }

    #[test]
    fn v2_future_window_endpoints_are_rejected_before_forward_materialization() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("endpoint.db"))).unwrap();
        let base = Utc::now() - chrono::Duration::hours(1);
        let trigger = ingest(&storage, "database retry endpoint incident");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id,
                tag: None,
                event: Some(v2_event(base)),
            })
            .unwrap();

        let at_endpoint = ingest(&storage, "database retry endpoint decision");
        let mut endpoint_tag = SynapticTag::new(&at_endpoint.id);
        endpoint_tag.created_at = base + chrono::Duration::hours(2);
        let endpoint = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: at_endpoint.id,
                tag: Some(endpoint_tag),
                event: None,
            })
            .unwrap_err();
        assert!(
            endpoint.to_string().contains("timestamp is in the future"),
            "the forward endpoint is in the future and must not be scheduled"
        );

        let after_endpoint = ingest(&storage, "database retry endpoint followup");
        let mut after_tag = SynapticTag::new(&after_endpoint.id);
        after_tag.created_at =
            base + chrono::Duration::hours(2) + chrono::Duration::milliseconds(1);
        let outside = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: after_endpoint.id,
                tag: Some(after_tag),
                event: None,
            })
            .unwrap_err();
        assert!(
            outside.to_string().contains("timestamp is in the future"),
            "one millisecond after a future endpoint must also be rejected"
        );
    }

    #[test]
    fn v2_stale_new_event_is_closed_and_evaluates_backward_only() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("stale_event.db"))).unwrap();
        let event_time = Utc::now() - chrono::Duration::hours(3);
        let candidate = ingest(&storage, "database retry policy decision");
        let mut tag = SynapticTag::new(&candidate.id);
        tag.created_at = event_time - chrono::Duration::minutes(1);
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id.clone(),
                tag: Some(tag),
                event: None,
            })
            .unwrap();
        let before = storage.demote_memory(&candidate.id).unwrap();

        let trigger = ingest(&storage, "database retry policy caused an incident");
        let stale = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id.clone(),
                tag: None,
                event: Some(v2_event(event_time)),
            })
            .unwrap();
        let root = stale.event.expect("stale event still has a root receipt");
        assert_eq!(
            root.captured_count, 1,
            "the valid backward path remains available"
        );
        assert!(
            stale.forward_receipts.is_empty(),
            "a stale event must never create a forward pair receipt"
        );

        let event_state: String = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT event_state FROM synaptic_events
                 WHERE trigger_memory_id = ?1 AND algorithm_version = ?2",
                params![trigger.id, SYNAPTIC_CAPTURE_ALGORITHM_V2],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(event_state, "closed");
        let pair_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM synaptic_capture_items", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(pair_count, 1, "only the backward evidence row is retained");
        let forward_pair_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM synaptic_capture_items
                 WHERE evaluation_direction = 'forward'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(forward_pair_count, 0);
        let after = storage.get_node(&candidate.id).unwrap().unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
    }

    #[test]
    fn v2_rejects_future_dated_tags_before_forward_reconciliation() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("future_tag.db"))).unwrap();
        let event_time = Utc::now() - chrono::Duration::minutes(1);
        let trigger = ingest(&storage, "database retry policy caused an incident");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id,
                tag: None,
                event: Some(v2_event(event_time)),
            })
            .unwrap();

        let candidate = ingest(&storage, "database retry policy decision");
        let before = storage.get_node(&candidate.id).unwrap().unwrap();
        let mut future_tag = SynapticTag::new(&candidate.id);
        future_tag.created_at = Utc::now() + chrono::Duration::minutes(5);
        let error = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id.clone(),
                tag: Some(future_tag),
                event: None,
            })
            .unwrap_err();
        assert!(
            error.to_string().contains("timestamp is in the future"),
            "unexpected error: {error}"
        );

        let tag_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM synaptic_tags WHERE memory_id = ?1",
                params![candidate.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tag_count, 0, "future tag must not be persisted");
        let pair_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM synaptic_capture_items", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(pair_count, 0, "future tag must not produce a pair receipt");
        let after = storage.get_node(&candidate.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
    }

    #[test]
    fn v2_rejects_future_dated_events_before_backward_evaluation() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("future_event.db"))).unwrap();
        let candidate = ingest(&storage, "database retry policy decision");
        let before = storage.get_node(&candidate.id).unwrap().unwrap();
        let mut tag = SynapticTag::new(&candidate.id);
        tag.created_at = Utc::now() - chrono::Duration::minutes(1);
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id.clone(),
                tag: Some(tag),
                event: None,
            })
            .unwrap();

        let trigger = ingest(&storage, "database retry policy caused an incident");
        let error = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id.clone(),
                tag: None,
                event: Some(v2_event(Utc::now() + chrono::Duration::minutes(5))),
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("event timestamp is in the future"),
            "unexpected error: {error}"
        );
        let event_count: i64 = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM synaptic_events
                 WHERE trigger_memory_id = ?1 AND algorithm_version = ?2",
                params![trigger.id, SYNAPTIC_CAPTURE_ALGORITHM_V2],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(event_count, 0, "future event must not be persisted");
        let after = storage.get_node(&candidate.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);
    }

    #[test]
    fn v2_suppressed_trigger_withholds_and_child_receipt_redacts_after_purge() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("v2_privacy.db"))).unwrap();
        let base = Utc::now() - chrono::Duration::minutes(5);
        let trigger = ingest(&storage, "database retry privacy incident");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger.id.clone(),
                tag: None,
                event: Some(v2_event(base)),
            })
            .unwrap();
        storage.suppress_memory(&trigger.id).unwrap();

        let candidate = ingest(&storage, "database retry privacy decision");
        let before = storage.get_node(&candidate.id).unwrap().unwrap();
        let mut tag = SynapticTag::new(&candidate.id);
        tag.created_at = base + chrono::Duration::minutes(1);
        let withheld = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate.id.clone(),
                tag: Some(tag),
                event: None,
            })
            .unwrap();
        assert_eq!(withheld.forward_receipts.len(), 1);
        assert_eq!(
            withheld.forward_receipts[0].disposition,
            SynapticCaptureDisposition::WithheldSuppressed
        );
        let withheld_json = serde_json::to_string(&withheld.forward_receipts[0].receipt).unwrap();
        assert!(!withheld_json.contains(&trigger.id));
        assert!(!withheld_json.contains(&candidate.id));
        let after = storage.get_node(&candidate.id).unwrap().unwrap();
        assert_eq!(after.retrieval_strength, before.retrieval_strength);

        // A separate captured pair proves the generic immutable child receipt
        // is scrubbed when either stable memory id is later purged.
        let trigger2 = ingest(&storage, "database retry purge incident");
        storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: trigger2.id.clone(),
                tag: None,
                event: Some(v2_event(base)),
            })
            .unwrap();
        let candidate2 = ingest(&storage, "database retry purge decision");
        let mut tag2 = SynapticTag::new(&candidate2.id);
        tag2.created_at = base + chrono::Duration::minutes(2);
        let captured = storage
            .process_synaptic_ingest(&SynapticIngestRequest {
                memory_id: candidate2.id.clone(),
                tag: Some(tag2),
                event: None,
            })
            .unwrap();
        let child = captured
            .forward_receipts
            .iter()
            .find(|receipt| receipt.disposition == SynapticCaptureDisposition::Captured)
            .expect("one captured child");
        let receipt_id = child.receipt.receipt_id.clone();
        storage
            .purge_node(&trigger2.id, Some("privacy test"))
            .unwrap();
        storage
            .purge_node(&candidate2.id, Some("privacy test"))
            .unwrap();
        let redacted = storage.get_receipt(&receipt_id).unwrap().unwrap();
        let redacted_json = serde_json::to_string(&redacted).unwrap();
        assert!(!redacted_json.contains(&trigger2.id));
        assert!(!redacted_json.contains(&candidate2.id));
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
        let path = dir.path().join("redaction.db");
        let storage = Storage::new(Some(path.clone())).unwrap();
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
        let (capture_event_id, tag_state): (Option<String>, String) = storage
            .reader
            .lock()
            .unwrap()
            .query_row(
                "SELECT capture_event_id, state FROM synaptic_tags WHERE memory_id = ?1",
                params![earlier.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            capture_event_id, None,
            "purge must clear tag linkage to the private event fingerprint"
        );
        assert_eq!(
            tag_state, "expired",
            "a captured tag without its purged proof must become terminal, not remain captured"
        );
        drop(storage);
        Storage::new(Some(path))
            .expect("purging a trigger must leave a restart-valid synaptic state");
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
