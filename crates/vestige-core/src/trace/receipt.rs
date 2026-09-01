//! # Memory Receipts
//!
//! Every important retrieval returns a [`Receipt`] — a structured record of what
//! the agent's memory actually did to answer a query. It is built entirely from
//! data the retrieval pipeline *already computes* (scored memories, suppression
//! decisions, spreading-activation path, FSRS trust), so attaching one is nearly
//! free and never changes the answer.
//!
//! The canonical shape (matching the product spec):
//!
//! ```json
//! {
//!   "receipt_id": "r_2026_06_22_abc",
//!   "retrieved": ["mem_1", "mem_7", "mem_9"],
//!   "suppressed": [{"id": "mem_4", "reason": "contradicted"}],
//!   "activation_path": ["project_goal -> design_decision -> current_file"],
//!   "trust_floor": 0.62,
//!   "decay_risk": "medium",
//!   "mutations": []
//! }
//! ```

use serde::{Deserialize, Serialize};

use super::SuppressReason;

/// A structured receipt attached to a retrieval's output.
///
/// Field names are snake_case to match the published product spec and the
/// dashboard receipt card exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Receipt {
    /// Stable, human-legible id: `r_<yyyy>_<mm>_<dd>_<short>`.
    pub receipt_id: String,

    /// Ids of the memories that actually informed the answer, best-first.
    pub retrieved: Vec<String>,

    /// Memories that were withheld, each with the reason — the "what the agent
    /// chose NOT to use" channel that makes retrieval auditable.
    pub suppressed: Vec<SuppressedReceiptEntry>,

    /// Human-readable spreading-activation path(s) that surfaced the result,
    /// e.g. `"project_goal -> design_decision -> current_file"`.
    pub activation_path: Vec<String>,

    /// The minimum trust score among the retrieved memories — the weakest link
    /// the answer rests on.
    pub trust_floor: f64,

    /// Coarse decay risk for the retrieved set (how stale the evidence is).
    pub decay_risk: DecayRisk,

    /// Any memory mutations this retrieval triggered (testing-effect
    /// strengthening, reconsolidation, supersession). Empty for a pure read.
    pub mutations: Vec<ReceiptMutation>,

    /// Optional schema-versioned evidence for non-retrieval receipts. This is
    /// kept inside the persisted receipt payload so fetching a receipt after a
    /// restart returns the same predicate that justified the mutation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<ReceiptEvidence>,
}

impl Receipt {
    /// Build a receipt from already-computed retrieval signals.
    ///
    /// `receipt_id` is `r_<date>_<discriminator8>_<unique6>` — human-legible
    /// and dated, with a short random suffix so that **multiple retrievals in
    /// the same run never collide** (B3). The discriminator (usually the runId)
    /// keeps receipts from one run visually grouped; the suffix guarantees
    /// uniqueness so `INSERT OR REPLACE` can't overwrite an earlier receipt.
    /// `trust_scores` is the per-id FSRS retrievability/trust the pipeline
    /// already produced.
    pub fn build(
        now: chrono::DateTime<chrono::Utc>,
        discriminator: &str,
        retrieved: Vec<String>,
        suppressed: Vec<SuppressedReceiptEntry>,
        activation_path: Vec<String>,
        trust_scores: &[f64],
        mutations: Vec<ReceiptMutation>,
    ) -> Self {
        Self::build_with_unique(
            now,
            discriminator,
            &uuid::Uuid::new_v4().simple().to_string()[..6],
            retrieved,
            suppressed,
            activation_path,
            trust_scores,
            mutations,
        )
    }

    /// Like [`Receipt::build`] but with a caller-supplied uniqueness token,
    /// so the id is fully deterministic for tests. Production uses
    /// [`Receipt::build`] which mints a random token.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_unique(
        now: chrono::DateTime<chrono::Utc>,
        discriminator: &str,
        unique: &str,
        retrieved: Vec<String>,
        suppressed: Vec<SuppressedReceiptEntry>,
        activation_path: Vec<String>,
        trust_scores: &[f64],
        mutations: Vec<ReceiptMutation>,
    ) -> Self {
        let trust_floor = trust_scores.iter().copied().fold(f64::INFINITY, f64::min);
        let trust_floor = if trust_floor.is_finite() {
            (trust_floor * 100.0).round() / 100.0
        } else {
            0.0
        };
        let decay_risk = DecayRisk::from_trust_floor(trust_floor);

        let short: String = discriminator
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(8)
            .collect();
        let unique_clean: String = unique
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(6)
            .collect();
        let receipt_id = format!("r_{}_{}_{}", now.format("%Y_%m_%d"), short, unique_clean);

        Self {
            receipt_id,
            retrieved,
            suppressed,
            activation_path,
            trust_floor,
            decay_risk,
            mutations,
            evidence: None,
        }
    }

    /// Attach a typed evidence predicate to this receipt.
    pub fn with_evidence(mut self, evidence: ReceiptEvidence) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Replace one memory id everywhere it can appear in a public receipt.
    /// Purge and state-aware read paths use this to preserve the audit shape
    /// without keeping a correlatable stable identifier.
    pub fn redact_memory_id(&mut self, memory_id: &str, replacement: &str) {
        for id in &mut self.retrieved {
            if id == memory_id {
                *id = replacement.to_string();
            }
        }
        for entry in &mut self.suppressed {
            if entry.id == memory_id {
                entry.id = replacement.to_string();
            }
        }
        for path in &mut self.activation_path {
            if path.contains(memory_id) {
                *path = path.replace(memory_id, replacement);
            }
        }
        for mutation in &mut self.mutations {
            if mutation.id == memory_id {
                mutation.id = replacement.to_string();
            }
        }
        match &mut self.evidence {
            Some(ReceiptEvidence::SynapticCapture(evidence)) => {
                if evidence.trigger.memory_id == memory_id {
                    evidence.trigger.memory_id = replacement.to_string();
                }
                for candidate in &mut evidence.candidates {
                    if candidate.memory_id.as_deref() == Some(memory_id) {
                        candidate.memory_id = None;
                    }
                }
            }
            Some(ReceiptEvidence::Backfill {
                failure_id,
                path_ids,
                candidates,
                ..
            }) => {
                if failure_id == memory_id {
                    *failure_id = replacement.to_string();
                }
                for id in path_ids.iter_mut() {
                    if id == memory_id {
                        *id = replacement.to_string();
                    }
                }
                for candidate in candidates.iter_mut() {
                    if candidate.memory_id == memory_id {
                        candidate.memory_id = replacement.to_string();
                    }
                }
            }
            Some(ReceiptEvidence::CounterfactualReplay { .. }) | None => {}
        }
    }

    /// Receipt-authored backfill route when the persisted evidence carries a
    /// complete ordered path. Callers must not invent topology when this is
    /// `None`.
    pub fn backfill_path_ids(&self) -> Option<&[String]> {
        match &self.evidence {
            Some(ReceiptEvidence::Backfill { path_ids, .. }) if path_ids.len() >= 2 => {
                Some(path_ids.as_slice())
            }
            _ => None,
        }
    }
}

/// Stable schema URI for Retroactive Salience Backfill receipt evidence.
pub const BACKFILL_RECEIPT_SCHEMA_V1: &str = "https://vestige.dev/schemas/receipt/backfill/v1";

/// Explicit epistemic boundary for Backfill receipts: candidate evidence only.
pub const BACKFILL_RECEIPT_CLAIM_BOUNDARY: &str = "Explicit-entity backward candidate evidence from a salient failure; not an asserted root cause or a universal claim about vector search.";

/// One earlier memory surfaced by a Backfill run. Records *candidate* evidence
/// rather than asserting an unverified root cause.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackfillCandidateEvidence {
    pub memory_id: String,
    pub content_preview: String,
    pub shared_entities: Vec<String>,
    pub age_days_before_failure: f64,
    pub similarity_rank: Option<usize>,
    pub backfill_score: f64,
    pub promoted: bool,
    /// Whether the explicit candidate→failure evidence edge was durably
    /// written before this receipt was saved. Separate from promotion: a
    /// preview can record evidence without strengthening a memory.
    #[serde(default)]
    pub candidate_edge_persisted: bool,
}

/// Typed predicate carried by a persisted receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "predicate", rename_all = "snake_case")]
pub enum ReceiptEvidence {
    /// A synaptic-tag capture decision and its measured state transition.
    SynapticCapture(SynapticCaptureEvidence),
    /// One controlled post-retrieval context-ablation replay. The nested
    /// result is identity-free and carries the exact non-causal claim boundary.
    CounterfactualReplay {
        schema: String,
        schema_version: u32,
        replay_id: String,
        capsule_id: String,
        result: crate::storage::CounterfactualReplayResult,
    },
    /// Retroactive Salience Backfill proof. `path_ids` is the only route a
    /// live scene may animate; incomplete paths fail closed (no render).
    Backfill {
        schema: String,
        schema_version: u32,
        failure_id: String,
        failure_preview: String,
        scanned: usize,
        lookback_days: i64,
        baseline: String,
        /// Ordered, receipt-authoritative route: `candidate_id -> failure_id`.
        path_ids: Vec<String>,
        candidates: Vec<BackfillCandidateEvidence>,
        claim_boundary: String,
    },
}

/// Versioned evidence for one durable synaptic-capture evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapticCaptureEvidence {
    /// Stable schema URI for downstream validators.
    pub schema: String,
    /// Predicate schema version.
    pub schema_version: u32,
    /// Version of the scoring/mutation policy used for this decision.
    pub algorithm_version: String,
    /// V2 receipt role: `root` for the event's initial backward decision or
    /// `pair` for an immutable forward event/tag evaluation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_role: Option<String>,
    /// Root event receipt referenced by a V2 pair receipt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_receipt_id: Option<String>,
    /// Explicit direction copied to the top level for receipt validators.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_direction: Option<String>,
    /// The importance event that opened the capture window.
    pub trigger: SynapticCaptureTrigger,
    /// Frozen policy snapshot used for eligibility and scoring.
    pub capture_window: SynapticCaptureWindow,
    /// Every candidate evaluated, including rejected and withheld candidates.
    pub candidates: Vec<SynapticCaptureCandidate>,
    /// Explicit epistemic boundary for product and audit surfaces.
    pub claim_boundary: String,
}

/// Trigger evidence for a synaptic-capture receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapticCaptureTrigger {
    pub event_id: String,
    pub memory_id: String,
    pub event_type: String,
    pub occurred_at: chrono::DateTime<chrono::Utc>,
    pub importance_score: f64,
}

/// Frozen temporal/scoring policy used by a capture decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapticCaptureWindow {
    pub evaluation_direction: String,
    pub backward_hours: f64,
    pub forward_hours: f64,
    pub tag_lifetime_hours: f64,
    pub minimum_tag_strength: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_association_score: Option<f64>,
    pub maximum_captures: usize,
    pub decay_function: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_algorithm_version: Option<String>,
}

/// One evaluated capture candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapticCaptureCandidate {
    /// Stable id only for evidence that was actually allowed to participate.
    /// Withheld candidates use a receipt-local opaque `evidence_slot` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<String>,
    pub evidence_slot: String,
    pub encoded_at: chrono::DateTime<chrono::Utc>,
    pub temporal_distance_hours: f64,
    pub capture_probability: f64,
    pub tag_strength_at_evaluation: f64,
    pub capture_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_direction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub association_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub competition_rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub disposition: SynapticCaptureDisposition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength_change: Option<SynapticStrengthChange>,
}

/// Auditable outcome for a candidate in a capture window.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SynapticCaptureDisposition {
    Captured,
    BelowThreshold,
    ContextMismatch,
    WithheldSuppressed,
    WithheldInvalid,
    LostCompetition,
}

/// Before/after strengths measured in the same transaction as the receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SynapticStrengthChange {
    pub retrieval_strength: StrengthDelta,
    pub retention_strength: StrengthDelta,
    pub stability: StrengthDelta,
}

/// One scalar before/after mutation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StrengthDelta {
    pub before: f64,
    pub after: f64,
}

/// One suppressed-memory entry in a [`Receipt`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuppressedReceiptEntry {
    /// The id of the suppressed memory.
    pub id: String,
    /// Why it was withheld.
    pub reason: SuppressReason,
}

impl SuppressedReceiptEntry {
    /// Convenience constructor.
    pub fn new(id: impl Into<String>, reason: SuppressReason) -> Self {
        Self {
            id: id.into(),
            reason,
        }
    }
}

/// Coarse staleness signal for a retrieved set, derived from the trust floor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DecayRisk {
    /// Trust floor is healthy; the evidence is fresh.
    Low,
    /// Some of the evidence is weakening.
    Medium,
    /// The answer rests on memory that is decaying out.
    High,
}

impl DecayRisk {
    /// Map the weakest retrieved-trust score to a decay-risk band.
    ///
    /// Thresholds align with the FSRS "due for review" intuition: above 0.7 the
    /// memory is comfortably retrievable, 0.4–0.7 is getting weak, below 0.4 is
    /// at risk of being forgotten.
    pub fn from_trust_floor(trust_floor: f64) -> Self {
        if trust_floor >= 0.7 {
            DecayRisk::Low
        } else if trust_floor >= 0.4 {
            DecayRisk::Medium
        } else {
            DecayRisk::High
        }
    }

    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            DecayRisk::Low => "low",
            DecayRisk::Medium => "medium",
            DecayRisk::High => "high",
        }
    }
}

/// A memory mutation that a retrieval triggered, recorded on the receipt so the
/// side effects of "just reading" are never invisible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReceiptMutation {
    /// The mutated memory id.
    pub id: String,
    /// What changed: `"strengthened"`, `"reconsolidated"`, `"superseded"`, …
    pub kind: String,
    /// Optional human note about the change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> chrono::DateTime<chrono::Utc> {
        chrono::Utc.with_ymd_and_hms(2026, 6, 22, 15, 0, 0).unwrap()
    }

    #[test]
    fn receipt_id_is_human_legible_and_dated() {
        let r = Receipt::build_with_unique(
            fixed_now(),
            "abc123!!",
            "u1u2u3",
            vec!["mem_1".into()],
            vec![],
            vec![],
            &[0.9],
            vec![],
        );
        assert_eq!(r.receipt_id, "r_2026_06_22_abc123_u1u2u3");
    }

    #[test]
    fn receipt_ids_unique_within_a_run_b3() {
        // B3: two retrievals in the SAME run (same date + discriminator) must
        // get DISTINCT ids so INSERT OR REPLACE can't overwrite the first.
        let a = Receipt::build(fixed_now(), "run_x", vec![], vec![], vec![], &[], vec![]);
        let b = Receipt::build(fixed_now(), "run_x", vec![], vec![], vec![], &[], vec![]);
        assert_ne!(
            a.receipt_id, b.receipt_id,
            "same-run receipts must not collide"
        );
        assert!(a.receipt_id.starts_with("r_2026_06_22_runx_"));
        assert!(b.receipt_id.starts_with("r_2026_06_22_runx_"));
    }

    #[test]
    fn trust_floor_is_the_weakest_link() {
        let r = Receipt::build(
            fixed_now(),
            "x",
            vec!["a".into(), "b".into(), "c".into()],
            vec![],
            vec![],
            &[0.91, 0.62, 0.78],
            vec![],
        );
        assert_eq!(r.trust_floor, 0.62);
        assert_eq!(r.decay_risk, DecayRisk::Medium);
    }

    #[test]
    fn empty_trust_scores_floor_to_zero_high_risk() {
        let r = Receipt::build(fixed_now(), "x", vec![], vec![], vec![], &[], vec![]);
        assert_eq!(r.trust_floor, 0.0);
        assert_eq!(r.decay_risk, DecayRisk::High);
    }

    #[test]
    fn decay_bands() {
        assert_eq!(DecayRisk::from_trust_floor(0.95), DecayRisk::Low);
        assert_eq!(DecayRisk::from_trust_floor(0.55), DecayRisk::Medium);
        assert_eq!(DecayRisk::from_trust_floor(0.20), DecayRisk::High);
    }

    #[test]
    fn matches_published_spec_shape() {
        let r = Receipt {
            receipt_id: "r_2026_06_22_abc".into(),
            retrieved: vec!["mem_1".into(), "mem_7".into(), "mem_9".into()],
            suppressed: vec![SuppressedReceiptEntry::new(
                "mem_4",
                SuppressReason::Contradicted,
            )],
            activation_path: vec!["project_goal -> design_decision -> current_file".into()],
            trust_floor: 0.62,
            decay_risk: DecayRisk::Medium,
            mutations: vec![],
            evidence: None,
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["receipt_id"], "r_2026_06_22_abc");
        assert_eq!(json["suppressed"][0]["reason"], "contradicted");
        assert_eq!(json["decay_risk"], "medium");
        assert_eq!(json["trust_floor"], 0.62);
        assert!(json["mutations"].as_array().unwrap().is_empty());
        assert!(json.get("evidence").is_none());
    }

    #[test]
    fn v1_synaptic_receipt_deserializes_without_v2_optional_fields() {
        let json = serde_json::json!({
            "receipt_id": "r_syn_v1",
            "retrieved": [],
            "suppressed": [],
            "activation_path": [],
            "trust_floor": 0.0,
            "decay_risk": "high",
            "mutations": [],
            "evidence": {
                "kind": "synaptic_capture",
                "predicate": {
                    "schema": "https://vestige.dev/schemas/receipt/synaptic-capture/v1",
                    "schemaVersion": 1,
                    "algorithmVersion": "vestige.synaptic_capture.v1",
                    "trigger": {
                        "eventId": "sevt_public",
                        "memoryId": "memory_trigger",
                        "eventType": "novelty_spike",
                        "occurredAt": "2026-06-22T15:00:00Z",
                        "importanceScore": 0.9
                    },
                    "captureWindow": {
                        "evaluationDirection": "backward_only",
                        "backwardHours": 9.0,
                        "forwardHours": 0.0,
                        "tagLifetimeHours": 12.0,
                        "minimumTagStrength": 0.3,
                        "maximumCaptures": 50,
                        "decayFunction": "exponential"
                    },
                    "candidates": [],
                    "claimBoundary": "temporal association only"
                }
            }
        });
        let receipt: Receipt = serde_json::from_value(json).expect("V1 receipt remains readable");
        let Some(ReceiptEvidence::SynapticCapture(evidence)) = receipt.evidence else {
            panic!("typed synaptic evidence");
        };
        assert_eq!(evidence.schema_version, 1);
        assert!(evidence.receipt_role.is_none());
        assert!(evidence.capture_window.context_threshold.is_none());
    }

    #[test]
    fn legacy_receipt_without_backfill_evidence_deserializes_untouched() {
        let json = serde_json::json!({
            "receipt_id": "r_2026_06_22_legacy",
            "retrieved": ["mem_1"],
            "suppressed": [],
            "activation_path": ["mem_1"],
            "trust_floor": 0.7,
            "decay_risk": "low",
            "mutations": []
        });
        let receipt: Receipt = serde_json::from_value(json).expect("pre-backfill receipt row");
        assert!(receipt.evidence.is_none());
        assert!(receipt.backfill_path_ids().is_none());
    }

    #[test]
    fn backfill_evidence_round_trips_and_requires_complete_path() {
        let receipt = Receipt::build(
            chrono::Utc::now(),
            "run_bf",
            vec!["cause_1".into()],
            vec![],
            vec!["cause_1 -> fail_1".into()],
            &[0.8],
            vec![],
        )
        .with_evidence(ReceiptEvidence::Backfill {
            schema: BACKFILL_RECEIPT_SCHEMA_V1.into(),
            schema_version: 1,
            failure_id: "fail_1".into(),
            failure_preview: "crashed".into(),
            scanned: 12,
            lookback_days: 30,
            baseline: "embedding cosine rank within the scanned candidate set".into(),
            path_ids: vec!["cause_1".into(), "fail_1".into()],
            candidates: vec![BackfillCandidateEvidence {
                memory_id: "cause_1".into(),
                content_preview: "Set API_TIMEOUT=2".into(),
                shared_entities: vec!["API_TIMEOUT".into()],
                age_days_before_failure: 3.0,
                similarity_rank: Some(9),
                backfill_score: 0.91,
                promoted: false,
                candidate_edge_persisted: true,
            }],
            claim_boundary: BACKFILL_RECEIPT_CLAIM_BOUNDARY.into(),
        });
        let encoded = serde_json::to_value(&receipt).expect("serialize");
        assert_eq!(encoded["evidence"]["kind"], "backfill");
        assert_eq!(encoded["evidence"]["predicate"]["schema_version"], 1);
        assert_eq!(
            encoded["evidence"]["predicate"]["path_ids"],
            serde_json::json!(["cause_1", "fail_1"])
        );
        let decoded: Receipt = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(
            decoded.backfill_path_ids(),
            Some(["cause_1".to_string(), "fail_1".to_string()].as_slice())
        );
        let incomplete = decoded.with_evidence(ReceiptEvidence::Backfill {
            schema: BACKFILL_RECEIPT_SCHEMA_V1.into(),
            schema_version: 1,
            failure_id: "fail_1".into(),
            failure_preview: "crashed".into(),
            scanned: 1,
            lookback_days: 30,
            baseline: "baseline".into(),
            path_ids: vec!["only_one".into()],
            candidates: vec![],
            claim_boundary: BACKFILL_RECEIPT_CLAIM_BOUNDARY.into(),
        });
        assert!(incomplete.backfill_path_ids().is_none());
    }
}
