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
        let trust_floor = trust_scores
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
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
        let receipt_id = format!(
            "r_{}_{}_{}",
            now.format("%Y_%m_%d"),
            short,
            unique_clean
        );

        Self {
            receipt_id,
            retrieved,
            suppressed,
            activation_path,
            trust_floor,
            decay_risk,
            mutations,
        }
    }
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
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["receipt_id"], "r_2026_06_22_abc");
        assert_eq!(json["suppressed"][0]["reason"], "contradicted");
        assert_eq!(json["decay_risk"], "medium");
        assert_eq!(json["trust_floor"], 0.62);
        assert!(json["mutations"].as_array().unwrap().is_empty());
    }
}
