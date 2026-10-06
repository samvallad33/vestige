//! Pinned table `mattar-evb-v1`.
//!
//! These values are the preregistration. Later stages consume them and do
//! not retune them after a result exists.

/// Table id written into every manifest and results file.
pub const TABLE_ID: &str = "mattar-evb-v1";

/// Discount per backup and per eval step.
pub const GAMMA: f64 = 0.9;

/// Policy softmax inverse temperature.
pub const BETA: f64 = 5.0;

/// Successor-representation delta-rule step.
pub const ALPHA_T: f64 = 0.9;

/// Neumann degree. Terms are `i = 0..=K`.
pub const K_NEUMANN: usize = 64;

/// Stationary power-iteration steps.
pub const STATIONARY_ITERS: usize = 256;

/// Floor applied to Gain before multiplying by Need. Not a drop filter.
pub const MIN_GAIN: f64 = 1e-10;

/// Planning steps per cut.
pub const BACKUPS: usize = 20;

/// Fig. 1d budget grid, including the no-backup point.
pub const BUDGETS: [usize; 7] = [0, 1, 2, 4, 8, 12, 20];

/// Budgets inside the primary mean. Budget 0 is reported and not averaged.
pub const MEAN_BUDGETS: [usize; 6] = [1, 2, 4, 8, 12, 20];

/// Suffix horizon in frame seq distance.
pub const HORIZON: u64 = 512;

/// Minimum typed edges on an eligible prefix.
pub const MIN_EDGES: usize = 4;

/// Minimum subjects on an eligible prefix.
pub const MIN_STATES: usize = 4;

/// Minimum rating-4 reviews on an eligible prefix.
pub const MIN_PROMOTES: usize = 1;

/// Maximum cuts kept after integer thinning.
pub const MAX_CUTS: usize = 32;

/// NDCG cutoff.
pub const NDCG_K: usize = 5;

/// Log-likelihood temperature on rank scores.
pub const BETA_LL: f64 = 1.0;

/// Master bootstrap seed (`SplitMix64` streams are subseeded from this).
pub const BOOTSTRAP_SEED: u64 = 0x2018_4D41_5454_4152;

/// Bootstrap draws.
pub const BOOTSTRAP_B: usize = 2000;

/// Absolute gap required before a separated claim.
pub const MIN_DELTA: f64 = 0.02;

/// Corpus clock origin, unix milliseconds.
pub const CLOCK_ORIGIN_MS: i64 = 1_700_000_000_000;

/// Corpus clock step, milliseconds per recorded operation.
pub const CLOCK_STEP_MS: i64 = 3_600_000;

/// Synthetic corpus id.
pub const SYNTH_CORPUS: &str = "synth-track-v1";

/// In-process recorded-operations corpus id.
pub const RECORDED_CORPUS: &str = "recorded-ops-v1";

/// Scope string for [`SYNTH_CORPUS`].
pub const SYNTH_SCOPE: &str = "synth";

/// Scope string for [`RECORDED_CORPUS`].
pub const RECORDED_SCOPE: &str = "agent";

/// Need arms, report order.
pub const NEED_ARMS: [&str; 7] = [
    "sr_need",
    "stationary_need",
    "fsrs_r",
    "recency",
    "degree",
    "uniform",
    "random",
];

/// Scheduler arms, report order.
pub const EVB_ARMS: [&str; 9] = [
    "evb",
    "gain_only",
    "need_only",
    "fsrs_r",
    "recency",
    "random",
    "uniform",
    "no_replay",
    "indicator_need",
];

/// Edge kinds that are backup experiences.
pub const EXPERIENCE_LINKS: [&str; 4] = ["derived_from", "closed_by", "evidence_of", "touched"];

/// Edge kinds whose suffix endpoints are occupancy labels.
pub const SUFFIX_EDGE_LINKS: [&str; 5] = [
    "derived_from",
    "evidence_of",
    "closed_by",
    "corrects",
    "touched",
];

/// Outcome types, pinned here so the backtest does not import another crate's table.
pub const OUTCOME_TYPES: [&str; 15] = [
    "helpful",
    "dead_end",
    "submitted",
    "accepted",
    "rejected",
    "duplicate_risk",
    "needs_poc",
    "bad_severity",
    "user_promoted",
    "user_demoted",
    "closed_by_scope",
    "closed_by_duplicate",
    "closed_by_false_assumption",
    "closed_by_user",
    "expired_lane",
];

/// Outcome types that contribute `+1`. Every other [`OUTCOME_TYPES`] entry is `-1`.
pub const OUTCOME_POSITIVE: [&str; 4] = ["helpful", "accepted", "submitted", "user_promoted"];

/// Deviation ids, results-file order.
pub const DEVIATIONS: [&str; 7] = [
    "one_step_backups_pr3_not_run",
    "backup_is_bellman_max_not_policy_expectation",
    "corpus_b_in_process_recorded_ops_corpus_c_not_run",
    "dream_windows_empty",
    "anchor_path_transitions_without_edge_join",
    "write_occupancy_not_decision_occupancy",
    "q_is_the_model_under_test",
];

/// `composition` node type a weave upsert carries.
pub const COMPOSITION_NODE_TYPE: &str = "composition";

/// Tag every weave upsert carries.
pub const GHOSTLINK_TAG: &str = "ghostlink";

/// Exact weave tag.
pub const WEAVE_TAG: &str = "ghostlink-weave";

/// Prefix of an outcome tag (`outcome:{type}`).
pub const OUTCOME_TAG_PREFIX: &str = "outcome:";

/// Fig. 6c bin ids, report order.
pub const FIG6C_BINS: [&str; 3] = ["lt_0.4", "mid_0.4_0.7", "ge_0.7"];

/// Sign of a known outcome type. `None` when the string is not in the table.
pub fn outcome_sign(outcome: &str) -> Option<i32> {
    if !OUTCOME_TYPES.contains(&outcome) {
        return None;
    }
    if OUTCOME_POSITIVE.contains(&outcome) {
        Some(1)
    } else {
        Some(-1)
    }
}

/// Promote / demote accumulator.
///
/// Rating 4 sets `+1`, rating 1 sets `-1`, and ratings 2 and 3 leave the
/// accumulator unchanged.
pub fn review_sign(ratings_in_order: impl IntoIterator<Item = u8>) -> i32 {
    let mut sign = 0;
    for rating in ratings_in_order {
        match rating {
            4 => sign = 1,
            1 => sign = -1,
            _ => {}
        }
    }
    sign
}

/// Bootstrap CI indices for a sample of length `b` (`b >= 2`).
pub fn ci_indices(b: usize) -> (usize, usize) {
    let bm1 = (b - 1) as u128;
    let lo = (25 * bm1) / 1000;
    // `(975 * (B-1) + 999) / 1000`, the preregistered ceil.
    let hi = (975 * bm1).div_ceil(1000);
    (lo as usize, hi as usize)
}

/// True when `link` is one of the eight typed-edge vocabulary strings.
pub fn is_typed_link(link: &str) -> bool {
    strata_store::EdgeKind::parse(link).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_sign_ignores_two_and_three() {
        assert_eq!(review_sign([4, 3, 2]), 1);
        assert_eq!(review_sign([4, 1]), -1);
        assert_eq!(review_sign([1, 2, 3]), -1);
        assert_eq!(review_sign([3, 4]), 1);
        assert_eq!(review_sign([2, 3]), 0);
    }

    #[test]
    fn outcome_table_collapses_to_a_sign() {
        assert_eq!(outcome_sign("helpful"), Some(1));
        assert_eq!(outcome_sign("accepted"), Some(1));
        assert_eq!(outcome_sign("submitted"), Some(1));
        assert_eq!(outcome_sign("user_promoted"), Some(1));
        assert_eq!(outcome_sign("needs_poc"), Some(-1));
        assert_eq!(outcome_sign("dead_end"), Some(-1));
        assert_eq!(outcome_sign("not-an-outcome"), None);
        assert_eq!(OUTCOME_TYPES.len(), 15);
    }

    #[test]
    fn ci_indices_match_the_prereg_formula() {
        assert_eq!(ci_indices(BOOTSTRAP_B), (49, 1950));
    }

    #[test]
    fn prereg_file_pins_the_table() {
        let text = include_str!("../../../docs/benchmarks/MATTAR-EVB-PREREGISTRATION.md");
        for needle in [
            TABLE_ID,
            "0.02",
            "0x2018_4D41_5454_4152",
            "tokio #6714",
            "cargo #10682",
            "uv #10186",
            "prometheus",
            "kubernetes",
            "grafana",
            "scrap",
            "one_step_backups_pr3_not_run",
        ] {
            assert!(text.contains(needle), "prereg missing {needle}");
        }
    }
}
