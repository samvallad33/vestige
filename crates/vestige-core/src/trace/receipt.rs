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

use std::collections::BTreeMap;

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

    /// Memories the **Microglial Firewall** caught and quarantined during this
    /// retrieval — each a write that *would* have surfaced but was screened out
    /// as a threat (prompt injection, exfiltration, poisoning). The headline
    /// proof line: these were caught, so they never reached the answer.
    ///
    /// `#[serde(default)]` so legacy receipts (no firewall) still deserialize;
    /// `skip_serializing_if` keeps the common empty case off the wire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quarantined: Vec<QuarantinedReceiptEntry>,

    /// Did anything the firewall caught reach the answer? `true` normally;
    /// `false` the moment **any** memory in this retrieval was quarantined.
    /// This is the single boolean the Black Box and receipt card render as the
    /// proof headline: "no quarantined memory influenced this answer".
    ///
    /// `#[serde(default = "default_true")]` so legacy receipts (which had no
    /// firewall and therefore allowed influence) deserialize as `true`.
    #[serde(default = "default_true")]
    pub influence_allowed: bool,

    /// Descriptive engram-lifecycle label per memory id (e.g. `"stable"`,
    /// `"reactivated"`, `"quarantined"`), surfaced for receipts and Cinema.
    /// This is purely additive colour derived from existing FSRS / suppression /
    /// reconsolidation signals — it never gates retrieval.
    ///
    /// `#[serde(default)]` + `skip_serializing_if` so legacy receipts and the
    /// common empty case stay off the wire.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub engram_phases: BTreeMap<String, String>,
}

/// Default for [`Receipt::influence_allowed`]: a receipt with no firewall data
/// allowed every retrieved memory to influence the answer.
fn default_true() -> bool {
    true
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
            quarantined: Vec::new(),
            influence_allowed: true,
            engram_phases: BTreeMap::new(),
        }
    }

    /// Attach the Microglial Firewall's quarantine verdict to this receipt.
    ///
    /// Records the caught entries and, if any were caught, flips
    /// [`Receipt::influence_allowed`] to `false` — the headline proof that the
    /// firewall's catches never reached the answer. Passing an empty slice is a
    /// no-op (`influence_allowed` stays `true`), so a clean retrieval keeps the
    /// healthy default.
    pub fn with_quarantine(mut self, entries: Vec<QuarantinedReceiptEntry>) -> Self {
        if !entries.is_empty() {
            self.influence_allowed = false;
        }
        self.quarantined = entries;
        self
    }

    /// Attach descriptive engram-lifecycle labels (memory id -> phase) to this
    /// receipt. Purely additive colour; never gates retrieval.
    pub fn with_engram_phases(mut self, phases: BTreeMap<String, String>) -> Self {
        self.engram_phases = phases;
        self
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

/// One memory the **Microglial Firewall** caught and quarantined during a
/// retrieval — the security-screen analogue of [`SuppressedReceiptEntry`].
///
/// Unlike a suppression (a normal retrieval decision: low trust, decay,
/// competition), a quarantine means the write was screened out as a *threat*,
/// so it carries both a machine `reason` code and human `threat` prose.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuarantinedReceiptEntry {
    /// The id of the quarantined memory.
    pub id: String,
    /// Machine code, e.g. `"prompt_injection"`, `"contradicts_high_trust"`,
    /// `"exfiltration"`, `"manual"`.
    pub reason: String,
    /// Human-readable description of why it was caught, e.g. `"Detected an
    /// instruction-injection payload masquerading as a memory."`.
    pub threat: String,
}

impl QuarantinedReceiptEntry {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        reason: impl Into<String>,
        threat: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            reason: reason.into(),
            threat: threat.into(),
        }
    }
}

/// A descriptive engram-lifecycle phase, surfaced in receipts and Cinema.
///
/// This is a *label*, not a state machine — it sits alongside (and never
/// replaces) the load-bearing [`crate::MemoryState`] (Active/Dormant/Silent/
/// Unavailable). It is derived from existing FSRS retrievability, suppression,
/// and reconsolidation signals via [`EngramPhase::derive`] purely to give the
/// dashboard a readable "where is this memory in its life" word.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EngramPhase {
    /// Just written; not yet consolidated.
    Nascent,
    /// Consolidated and comfortably retrievable.
    Stable,
    /// Recently retrieved again, re-strengthening it.
    Reactivated,
    /// Being rewritten/updated — labile while it reconsolidates.
    Reconsolidating,
    /// Caught by the firewall; barred from influencing answers.
    Quarantined,
    /// Decayed below usable retrievability.
    Forgotten,
}

impl EngramPhase {
    /// Stable string label (matches the serde wire form).
    pub fn as_str(&self) -> &'static str {
        match self {
            EngramPhase::Nascent => "nascent",
            EngramPhase::Stable => "stable",
            EngramPhase::Reactivated => "reactivated",
            EngramPhase::Reconsolidating => "reconsolidating",
            EngramPhase::Quarantined => "quarantined",
            EngramPhase::Forgotten => "forgotten",
        }
    }

    /// Derive a descriptive phase from already-computed memory signals.
    ///
    /// Precedence is deliberate, strongest signal first:
    /// 1. `quarantined` — a firewall catch dominates everything.
    /// 2. `reconsolidating` — an in-flight rewrite makes the engram labile.
    /// 3. `forgotten` — FSRS retrievability decayed below the usable floor.
    /// 4. `nascent` — brand-new and not yet consolidated.
    /// 5. `reactivated` — retrieved again recently, re-strengthening it.
    /// 6. `stable` — the healthy default.
    ///
    /// `retrievability` is the FSRS 0..=1 trust score; `recently_retrieved`
    /// marks a fresh reactivation; `is_new` marks an unconsolidated write.
    pub fn derive(
        retrievability: f64,
        is_new: bool,
        recently_retrieved: bool,
        reconsolidating: bool,
        quarantined: bool,
    ) -> Self {
        if quarantined {
            EngramPhase::Quarantined
        } else if reconsolidating {
            EngramPhase::Reconsolidating
        } else if retrievability < 0.4 {
            EngramPhase::Forgotten
        } else if is_new {
            EngramPhase::Nascent
        } else if recently_retrieved {
            EngramPhase::Reactivated
        } else {
            EngramPhase::Stable
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
            quarantined: vec![],
            influence_allowed: true,
            engram_phases: BTreeMap::new(),
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["receipt_id"], "r_2026_06_22_abc");
        assert_eq!(json["suppressed"][0]["reason"], "contradicted");
        assert_eq!(json["decay_risk"], "medium");
        assert_eq!(json["trust_floor"], 0.62);
        assert!(json["mutations"].as_array().unwrap().is_empty());
    }

    #[test]
    fn fresh_receipt_allows_influence_and_omits_empty_firewall_fields() {
        // A clean retrieval defaults to influence_allowed=true, empty quarantine
        // and empty engram_phases, and those empties stay OFF the wire so old
        // dashboards see exactly the legacy shape.
        let r = Receipt::build(fixed_now(), "x", vec!["a".into()], vec![], vec![], &[0.9], vec![]);
        assert!(r.influence_allowed);
        assert!(r.quarantined.is_empty());
        assert!(r.engram_phases.is_empty());

        let json = serde_json::to_value(&r).unwrap();
        assert!(json.get("quarantined").is_none(), "empty quarantine off the wire");
        assert!(json.get("engram_phases").is_none(), "empty phases off the wire");
        // influence_allowed has no skip, so it is always present and true.
        assert_eq!(json["influence_allowed"], true);
    }

    #[test]
    fn with_quarantine_flips_influence_allowed_false() {
        let entry = QuarantinedReceiptEntry::new(
            "mem_evil",
            "prompt_injection",
            "Detected an instruction-injection payload masquerading as a memory.",
        );
        let r = Receipt::build(fixed_now(), "x", vec!["a".into()], vec![], vec![], &[0.9], vec![])
            .with_quarantine(vec![entry.clone()]);

        assert!(!r.influence_allowed, "a quarantine means zero influence");
        assert_eq!(r.quarantined, vec![entry]);

        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["influence_allowed"], false);
        assert_eq!(json["quarantined"][0]["id"], "mem_evil");
        assert_eq!(json["quarantined"][0]["reason"], "prompt_injection");
        assert_eq!(
            json["quarantined"][0]["threat"],
            "Detected an instruction-injection payload masquerading as a memory."
        );
    }

    #[test]
    fn with_quarantine_empty_is_a_noop() {
        let r = Receipt::build(fixed_now(), "x", vec![], vec![], vec![], &[0.9], vec![])
            .with_quarantine(vec![]);
        assert!(r.influence_allowed, "empty quarantine keeps the healthy default");
        assert!(r.quarantined.is_empty());
    }

    #[test]
    fn with_engram_phases_surfaces_labels() {
        let mut phases = BTreeMap::new();
        phases.insert("mem_1".to_string(), EngramPhase::Stable.as_str().to_string());
        phases.insert("mem_2".to_string(), EngramPhase::Reactivated.as_str().to_string());
        let r = Receipt::build(fixed_now(), "x", vec![], vec![], vec![], &[0.9], vec![])
            .with_engram_phases(phases);

        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["engram_phases"]["mem_1"], "stable");
        assert_eq!(json["engram_phases"]["mem_2"], "reactivated");
    }

    #[test]
    fn legacy_receipt_json_without_new_fields_still_deserializes() {
        // A receipt persisted before the firewall existed: no quarantined,
        // influence_allowed, or engram_phases keys at all. Must deserialize with
        // the safe defaults (influence allowed, nothing quarantined).
        let legacy = serde_json::json!({
            "receipt_id": "r_2026_01_01_legacy_aaaaaa",
            "retrieved": ["mem_1", "mem_7"],
            "suppressed": [{"id": "mem_4", "reason": "contradicted"}],
            "activation_path": ["a -> b"],
            "trust_floor": 0.62,
            "decay_risk": "medium",
            "mutations": []
        });
        let r: Receipt = serde_json::from_value(legacy).unwrap();
        assert_eq!(r.receipt_id, "r_2026_01_01_legacy_aaaaaa");
        assert!(r.influence_allowed, "legacy receipts allowed influence");
        assert!(r.quarantined.is_empty());
        assert!(r.engram_phases.is_empty());
    }

    #[test]
    fn quarantined_entry_roundtrips() {
        let entry = QuarantinedReceiptEntry::new("m", "exfiltration", "tried to POST memory to a URL");
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["id"], "m");
        assert_eq!(json["reason"], "exfiltration");
        assert_eq!(json["threat"], "tried to POST memory to a URL");
        let back: QuarantinedReceiptEntry = serde_json::from_value(json).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn engram_phase_labels_match_serde() {
        assert_eq!(EngramPhase::Nascent.as_str(), "nascent");
        assert_eq!(EngramPhase::Reconsolidating.as_str(), "reconsolidating");
        assert_eq!(EngramPhase::Quarantined.as_str(), "quarantined");
        // serde uses the same lowercase form on the wire.
        assert_eq!(
            serde_json::to_value(EngramPhase::Reactivated).unwrap(),
            serde_json::json!("reactivated")
        );
        let back: EngramPhase = serde_json::from_value(serde_json::json!("forgotten")).unwrap();
        assert_eq!(back, EngramPhase::Forgotten);
    }

    #[test]
    fn engram_phase_derive_precedence() {
        // Quarantine dominates every other signal.
        assert_eq!(
            EngramPhase::derive(0.95, true, true, true, true),
            EngramPhase::Quarantined
        );
        // Reconsolidation beats decay/new/reactivation.
        assert_eq!(
            EngramPhase::derive(0.1, true, true, true, false),
            EngramPhase::Reconsolidating
        );
        // Low retrievability => forgotten.
        assert_eq!(
            EngramPhase::derive(0.2, false, false, false, false),
            EngramPhase::Forgotten
        );
        // New (and retrievable) => nascent.
        assert_eq!(
            EngramPhase::derive(0.9, true, false, false, false),
            EngramPhase::Nascent
        );
        // Retrieved again recently => reactivated.
        assert_eq!(
            EngramPhase::derive(0.9, false, true, false, false),
            EngramPhase::Reactivated
        );
        // Healthy default.
        assert_eq!(
            EngramPhase::derive(0.9, false, false, false, false),
            EngramPhase::Stable
        );
    }
}
