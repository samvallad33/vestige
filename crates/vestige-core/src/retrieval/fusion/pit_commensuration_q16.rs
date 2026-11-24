//! Engine #2: Empirical-CDF (PIT) Survival Surprisal + NQC Commensuration (`Q16.16`).
//!
//! Replaces min-max (`raw / max_raw`) normalization across multi-channel score fusion:
//! 1. **Empirical CDF / Probability Integral Transform (PIT)**:
//!    Ranks active positive candidates `d` into Hazen survival probability
//!    $1 - U_m(d) = \frac{N - r_m(d) + 1}{N + 1}$ with exact average-rank tie handling.
//! 2. **Upper-Tail Survival Surprisal**:
//!    $S_m(d) = -\log_2(1 - U_m(d)) = \log_2(N + 1) - \log_2(N - r_m(d) + 1)$
//!    computed in `Q16.16` via `kemeny_mrf::log2_u64_q16`.
//! 3. **NQC Channel Commensuration**:
//!    Weights each channel by $\text{NQC}_m = \frac{\sigma_{\text{top-}K}}{\mu_{\text{corpus}} + \epsilon}$,
//!    amplifying sharp, committed retrieval channels and damping flat noise channels.

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;
use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::graph::ghostlink_dgs::integer_sqrt;

/// Minimum NQC channel weight (`0.25` in `Q16.16`) so valid weak channels never vanish completely.
const NQC_MIN_W_Q16: u64 = Q16_ONE >> 2;
/// Maximum NQC channel weight (`2.0` in `Q16.16`) preventing any single channel from monopolizing.
const NQC_MAX_W_Q16: u64 = Q16_ONE * 2;

/// Transform raw non-negative `Q16.16` channel scores into Empirical-CDF upper-tail
/// survival surprisal $-\log_2(1 - U_m(d))$ in `Q16.16`. Zero entries remain `0`.
pub(in crate::retrieval) fn empirical_pit_surprisal_q16(raw_q16: &[u64]) -> Vec<u64> {
    let total = raw_q16.len();
    let mut out = vec![0u64; total];
    let mut active: Vec<(u64, usize)> = raw_q16
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v > 0).then_some((v, i)))
        .collect();
    let n = active.len() as u64;
    if n == 0 {
        return out;
    }

    // Sort ascending by raw score so highest score gets highest rank `r = n` (`tail = 1`).
    active.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let log_n_plus_1 = log2_u64_q16(n.saturating_add(1));

    let mut i = 0usize;
    while i < active.len() {
        let mut j = i + 1;
        while j < active.len() && active[j].0 == active[i].0 {
            j += 1;
        }
        // Average 1-indexed rank across tie group `[i..j]` (`(i + 1 + j) / 2`).
        let avg_rank = ((i + 1 + j) as u64) >> 1;
        let tail_num = n.saturating_add(1).saturating_sub(avg_rank).max(1);
        let surprisal = log_n_plus_1.saturating_sub(log2_u64_q16(tail_num)).max(Q16_ONE >> 4);
        for &(_, doc_idx) in &active[i..j] {
            out[doc_idx] = surprisal;
        }
        i = j;
    }
    out
}

/// Compute Shtok et al. (2012) Normalized Query Commitment ($\text{NQC}_m$) weight in `Q16.16`:
/// `sigma(top_k) / max(1, mean_all_positive)`, clamped to `[0.25, 2.0]` in `Q16.16`.
pub(in crate::retrieval) fn channel_nqc_weight_q16(raw_q16: &[u64], top_k: usize) -> u64 {
    let mut active: Vec<u64> = raw_q16.iter().copied().filter(|&v| v > 0).collect();
    if active.len() < 2 {
        return if active.is_empty() { 0 } else { Q16_ONE };
    }
    let sum_all: u128 = active.iter().map(|&v| v as u128).sum();
    let mean_all = (sum_all / (active.len() as u128)).max(1);

    active.sort_unstable_by(|a, b| b.cmp(a));
    let k = top_k.clamp(2, 32).min(active.len());
    let top = &active[..k];
    let sum_k: u128 = top.iter().map(|&v| v as u128).sum();
    let mean_k = sum_k / (k as u128);

    // Variance scaled carefully to prevent overflow on 1M-scale raw Q16 scores.
    let mut var_acc: u128 = 0;
    for &v in top {
        let diff = (v as u128).abs_diff(mean_k);
        // Normalize `diff / mean_all` directly in Q16 before squaring -> exact dimensionless CV!
        let rel_q16 = ((diff << 16) / mean_all).min((Q16_ONE as u128) * 64);
        var_acc = var_acc.saturating_add(rel_q16.saturating_mul(rel_q16));
    }
    let mean_var_q32 = (var_acc / (k as u128)).min(u64::MAX as u128) as u64;
    let nqc_q16 = integer_sqrt(mean_var_q32);
    nqc_q16.clamp(NQC_MIN_W_Q16, NQC_MAX_W_Q16)
}

/// Commensurate and fuse multiple scoring channels (`(channel_scores_q16, prior_weight_q16)`)
/// using PIT upper-tail survival surprisal weighted by each channel's live NQC commitment,
/// scaled cleanly into `[0, cap_q16]`.
pub(in crate::retrieval) fn fuse_channels_pit_nqc_q16(
    len: usize,
    channels: &[(&[u64], u64)],
    nqc_top_k: usize,
    cap_q16: u64,
) -> Vec<u64> {
    let mut fused = vec![0u64; len];
    if len == 0 || channels.is_empty() || cap_q16 == 0 {
        return fused;
    }

    let mut total_w: u64 = 0;
    let mut acc = vec![0u128; len];

    for &(raw, prior_w) in channels {
        if raw.len() != len || prior_w == 0 {
            continue;
        }
        let nqc_w = channel_nqc_weight_q16(raw, nqc_top_k);
        if nqc_w == 0 {
            continue;
        }
        let eff_w = ((prior_w as u128).saturating_mul(nqc_w as u128) >> 16).max(1) as u64;
        let surp = empirical_pit_surprisal_q16(raw);
        let max_s = surp.iter().copied().max().unwrap_or(0);
        if max_s == 0 {
            continue;
        }
        total_w = total_w.saturating_add(eff_w);
        for i in 0..len {
            if surp[i] > 0 {
                let norm_s = ((surp[i] as u128) << 16) / (max_s as u128);
                acc[i] = acc[i].saturating_add((norm_s.saturating_mul(eff_w as u128)) >> 16);
            }
        }
    }

    if total_w == 0 {
        return fused;
    }
    let max_acc = acc.iter().copied().max().unwrap_or(0);
    if max_acc == 0 {
        return fused;
    }
    for i in 0..len {
        if acc[i] > 0 {
            fused[i] = ((acc[i].saturating_mul(cap_q16 as u128)) / max_acc).min(cap_q16 as u128) as u64;
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pit_surprisal_resists_extreme_single_outlier_collapse() {
        // Under naive min-max (`x / max`), doc 1 (`100`) gets `0.001` vs doc 0 (`100_000`).
        // Under PIT survival surprisal, doc 1 keeps strong tail surprisal above background docs.
        let mut raw = vec![10 * Q16_ONE; 64];
        raw[0] = 100_000 * Q16_ONE;
        raw[1] = 100 * Q16_ONE;
        let surp = empirical_pit_surprisal_q16(&raw);
        assert!(surp[0] > surp[1]);
        assert!(surp[1] > surp[2] * 3, "surp[1]={} vs surp[2]={}", surp[1], surp[2]);
    }

    #[test]
    fn nqc_weights_peaked_channel_higher_than_flat_noise_channel() {
        let peaked = [100 * Q16_ONE, 20 * Q16_ONE, 10 * Q16_ONE, 5 * Q16_ONE, 5 * Q16_ONE];
        let flat = [11 * Q16_ONE, 10 * Q16_ONE, 10 * Q16_ONE, 10 * Q16_ONE, 9 * Q16_ONE];
        assert!(channel_nqc_weight_q16(&peaked, 5) > channel_nqc_weight_q16(&flat, 5) * 2);
    }
}
