//! JSON contracts for seed / run / score. `run` never reads the manifest.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_PATH: &str = "docs/v3/CAUSAL-SPIKE-PROTOCOL.md";
pub const PREREGISTERED_V1: &str = "2026-08-31";
pub const PREREGISTERED_V2: &str = "2026-09-01";
pub const V1_SEED: u64 = 20260831;
pub const V2_SEED: u64 = 20260901;
pub const DEFAULT_SEED: u64 = V2_SEED;
pub const A_FAIR_MOD: usize = 10;
pub const A_FAIR_LT: usize = 3;
pub const CHATTER_BEFORE: usize = 3;
pub const CHATTER_AFTER: usize = 2;
pub const N_STORES: usize = 3;
pub const PAIRS_PER_STORE: usize = 30;
pub const DISTRACTORS_PER_KIND: usize = 5;
pub const LOOKBACK_DAYS: i64 = 60;
pub const TOP_K: usize = 10;
pub const EVIDENCE_FLOOR: f64 = 0.2;
pub const LAG_DECAY_TAU_DAYS: f64 = 30.0;
pub const CAUSE_LAG_MIN_DAYS: u64 = 7;
pub const CAUSE_LAG_MAX_DAYS: u64 = 45;
pub const MULTIHOP_STRIDE: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// v2 operative Arm A: `Storage::search` (FTS OR+BM25).
    Lexical,
    /// Disclosure: `hybrid_search` FTS implicit-AND (v1 Arm A).
    #[value(name = "lexical-and")]
    LexicalAnd,
    /// v1 erratum alias of OR+BM25. Not required in the v2 gate.
    #[value(name = "lexical-or")]
    LexicalOr,
    /// Disclosure: `hybrid_search` after `init_embeddings`.
    #[value(name = "lexical-embed")]
    LexicalEmbed,
    Backfill,
    #[value(name = "causal-graph")]
    CausalGraph,
}

impl Arm {
    pub fn as_str(self) -> &'static str {
        match self {
            Arm::Lexical => "lexical",
            Arm::LexicalAnd => "lexical-and",
            Arm::LexicalOr => "lexical-or",
            Arm::LexicalEmbed => "lexical-embed",
            Arm::Backfill => "backfill",
            Arm::CausalGraph => "causal-graph",
        }
    }

    pub fn is_lexical(self) -> bool {
        matches!(
            self,
            Arm::Lexical | Arm::LexicalAnd | Arm::LexicalOr | Arm::LexicalEmbed
        )
    }
}

impl std::fmt::Display for Arm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub protocol: String,
    pub preregistered: String,
    pub seed: u64,
    pub t0: DateTime<Utc>,
    pub claim_boundary: String,
    #[serde(default)]
    pub dataset_id: String,
    #[serde(default)]
    pub generation: String,
    pub stores: Vec<StoreManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreManifest {
    pub id: String,
    pub db: String,
    pub failures_file: String,
    pub pairs: Vec<PairManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairManifest {
    pub pair_id: String,
    pub cause_id: String,
    pub failure_id: String,
    pub root_id: Option<String>,
    pub entity: String,
    pub bridge_entity: Option<String>,
    pub cause_lag_days: i64,
    pub multihop: bool,
    #[serde(default)]
    pub identifier_in_failure_content: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailuresFile {
    pub store_id: String,
    pub failure_ids: Vec<String>,
    #[serde(default)]
    pub dataset_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunOutput {
    pub arm: Arm,
    pub store: String,
    pub store_id: String,
    pub commit: String,
    pub search_mode: String,
    #[serde(default)]
    pub lookback_days: Option<i64>,
    pub top_k: usize,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub queries: Vec<QueryResult>,
    #[serde(default)]
    pub accumulation_ms: Option<f64>,
    #[serde(default)]
    pub dataset_id: String,
    #[serde(default)]
    pub scratch_store: Option<String>,
    #[serde(default)]
    pub embedding_ready: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub failure_id: String,
    pub ranked_ids: Vec<String>,
    pub scores: Vec<f64>,
    pub wall_clock_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmMetrics {
    pub arm: Arm,
    pub commit: String,
    pub seed: u64,
    pub n_pairs: usize,
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub mrr: f64,
    pub mean_wall_clock_ms: f64,
    pub search_mode: String,
    pub n_multihop: usize,
    pub multihop_recall_at_3: Option<f64>,
    pub separation_rate_vs_lexical: Option<f64>,
    #[serde(default)]
    pub separation_rate_vs_lexical_or: Option<f64>,
    #[serde(default)]
    pub accumulation_ms: Option<f64>,
    #[serde(default)]
    pub amortized_ms_per_query: Option<f64>,
    #[serde(default)]
    pub mean_list_len: f64,
    #[serde(default)]
    pub empty_list_rate: f64,
    #[serde(default)]
    pub n_answered_pairs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateVerdict {
    pub commit: String,
    pub seed: u64,
    pub lexical: ArmMetrics,
    pub backfill: ArmMetrics,
    pub causal_graph: ArmMetrics,
    #[serde(default)]
    pub lexical_or: Option<ArmMetrics>,
    #[serde(default)]
    pub lexical_and: Option<ArmMetrics>,
    #[serde(default)]
    pub lexical_embed: Option<ArmMetrics>,
    pub gate: GateChecks,
    pub outcome: String,
    pub claim_licensed_if_pass: String,
    #[serde(default)]
    pub claim_never_licensed: String,
    #[serde(default)]
    pub protocol: String,
    #[serde(default)]
    pub dataset_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateChecks {
    pub c_recall_at_1_ge_0_60: bool,
    pub c_recall_at_3_ge_0_80: bool,
    pub c_separation_vs_a_ge_0_40: bool,
    pub c_recall_at_3_ge_b: bool,
    pub c_multihop_recall_at_3_ge_0_50: bool,
}

pub fn t0() -> DateTime<Utc> {
    use chrono::TimeZone;
    Utc.with_ymd_and_hms(2026, 8, 31, 0, 0, 0)
        .single()
        .expect("valid T0")
}

pub fn is_multihop_pair(store_idx: usize, pair_idx: usize) -> bool {
    (store_idx * PAIRS_PER_STORE + pair_idx).is_multiple_of(MULTIHOP_STRIDE)
}

/// A-fairness: identifier also appears in failure content. Locked formula:
/// `global_idx % 10 < 3` → 27/90 = 30%.
pub fn is_a_fair_pair(store_idx: usize, pair_idx: usize) -> bool {
    (store_idx * PAIRS_PER_STORE + pair_idx) % A_FAIR_MOD < A_FAIR_LT
}

pub fn entity_name(store_idx: usize, pair_idx: usize) -> String {
    format!("SPIKE{store_idx}_CFG_{pair_idx:02}")
}

pub fn bridge_name(store_idx: usize, pair_idx: usize) -> String {
    format!("SPIKE{store_idx}_BRG_{pair_idx:02}")
}

/// Stable fingerprint of a failure-id list. Binds a run to a generation without
/// opening the manifest.
pub fn dataset_id_for(failure_ids: &[String]) -> String {
    let mut ids = failure_ids.to_vec();
    ids.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        for b in id.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Map raw backfill scores (typically ≥ 1.0) into (0, 1) so persisted
/// `ConnectionRecord.strength` and `EVIDENCE_FLOOR` are live.
pub fn squash_strength(score: f64) -> f64 {
    if !score.is_finite() || score <= 0.0 {
        return 0.0;
    }
    score / (1.0 + score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_maps_unit_score_to_half_and_stays_below_one() {
        assert!((squash_strength(1.0) - 0.5).abs() < 1e-12);
        assert!(squash_strength(10.0) < 1.0);
        assert!(squash_strength(10.0) > squash_strength(1.0));
        assert_eq!(squash_strength(0.0), 0.0);
    }

    #[test]
    fn dataset_id_is_order_invariant() {
        let a = dataset_id_for(&["b".into(), "a".into()]);
        let b = dataset_id_for(&["a".into(), "b".into()]);
        assert_eq!(a, b);
        assert_ne!(a, dataset_id_for(&["a".into()]));
    }

    #[test]
    fn a_fairness_is_exactly_thirty_percent() {
        let n = (0..N_STORES)
            .flat_map(|s| (0..PAIRS_PER_STORE).map(move |p| (s, p)))
            .filter(|(s, p)| is_a_fair_pair(*s, *p))
            .count();
        assert_eq!(n, 27);
        assert_eq!(CHATTER_BEFORE + CHATTER_AFTER, DISTRACTORS_PER_KIND);
    }
}
