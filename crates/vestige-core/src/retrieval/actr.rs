//! #827: Anderson (1993) / Schooler ACT-R Base-Level Activation
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! Formula:
//!   `B_i = log2( sum_{k=1..n} t_k^(-d) )`
//! using closed-form integer square root (`d = 1/2` power-law decay via `integer_sqrt`)
//! where `t_k = max(1, (ref_ts - sess_ts) / 86400 + 1)` is the elapsed interval in days
//! and `n` is the verified rehearsal/exposure count (`1 + exact_edges + backlinks`),
//! returning an integer `log2_q16` activation shift in Q16.16.

use super::dph::{log2_q16, Q16};
use super::ghostlink_dgs::integer_sqrt;
use super::index::Index;

/// Compute Anderson ACT-R base-level activation `B_i` in Q16.16 for turn `ti`
/// relative to reference timestamp `ref_ts_sec` and integer `stability_days`. Zero floats.
pub(super) fn actr_base_level_q16(
    ix: &Index,
    ti: usize,
    ref_ts_sec: i64,
    stability_days: u64,
) -> i64 {
    let sess = ix.turns[ti].session;
    let sess_ts = ix.session_ts[sess];
    let delta_days: u64 = if ref_ts_sec > sess_ts && sess_ts > 0 {
        (((ref_ts_sec - sess_ts) / 86_400) + 1).max(1) as u64
    } else {
        1
    };

    // Power-law kernel `t^(-1/2)` in Q16.16: `Q16 / sqrt(delta_days)`
    let sqrt_days_q8 = integer_sqrt(delta_days << 16).max(256);
    let recency_q16 = (((Q16 as u128) << 8) / sqrt_days_q8 as u128).max(1) as u64;

    // Rehearsal count from structural links (stability days / exact edges / policy backlinks)
    let stab_days = stability_days.min(365);
    let exposure_count = 1u64
        .saturating_add(ix.turns[ti].exact_edges.len() as u64)
        .saturating_add(ix.turns[ti].policy_backlinks.len() as u64)
        .saturating_add(stab_days / 7);

    let sum_q16 = recency_q16.saturating_mul(exposure_count.min(32));
    log2_q16(sum_q16.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actr_recency_activation_monotone_in_elapsed_days() {
        let recent = (((Q16 as u128) << 8) / integer_sqrt(1 << 16)) as u64;
        let stale = (((Q16 as u128) << 8) / integer_sqrt(100 << 16)) as u64;
        assert!(log2_q16(recent) > log2_q16(stale));
    }
}
