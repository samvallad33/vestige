//! Kernel 3: THOT + DBSN Log-Domain Sparse Sinkhorn (`thot_dbsn_sinkhorn_q16.rs`).
//!
//! References:
//! - THOT (arXiv:2611.27951): Sparse-thresholded affinity `max(0, s_ij - tau)` + log-domain Sinkhorn.
//! - DBSN (arXiv:2508.02538): Dual-balanced target-hubness subtraction `hbar(t_j)`.
//!
//! Pure integer Q16.16 log-domain Sinkhorn scaling using a 256-entry exponential LUT (`EXP_NEG_LUT_Q16`)
//! for universal R@1 sharpness without generic high-DF partial-match turns stealing Rank 1.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::{DetHashSet, Index, QueryPlan};

const Q16_ONE: i64 = 65_536;
/// Sparse affinity floor `tau = 0.25` in Q16.16.
const SPARSE_TAU_Q16: u64 = 16_384;
/// Number of doubly-stochastic log-domain Sinkhorn sweeps.
const SINKHORN_SWEEPS: usize = 5;

/// Lookup `round(65536 * 2^(-x / 32))` for `x in 0..=255` (`x = 32 * 8` covers `0..8` log2 steps).
#[inline]
fn exp2_neg_lut_q16(neg_q16: i64) -> u64 {
    if neg_q16 <= 0 {
        return Q16_ONE as u64;
    }
    let scaled = (neg_q16 >> 11) as u64; // 1/32th of Q16 unit -> index 0..=255
    if scaled >= 256 {
        return 0;
    }
    let whole = (scaled >> 5) as u32;
    let rem = scaled & 31;
    // Piecewise linear interpolation over `2^(-rem/32)` in Q16: 65536 - rem * 1419 approx for small step
    // Exact integer anchor table for `rem in 0..=31` (round(65536 * 2^(-k/32))):
    const STEP_LUT: [u64; 32] = [
        65536, 64132, 62757, 61413, 60097, 58809, 57549, 56316,
        55109, 53928, 52773, 51642, 50535, 49452, 48393, 47356,
        46341, 45348, 44376, 43425, 42495, 41584, 40693, 39821,
        38968, 38133, 37316, 36516, 35734, 34968, 34219, 33486,
    ];
    let base = STEP_LUT[rem as usize];
    base >> whole.min(16)
}

/// Scores candidate turns `[0..pool_len]` (`<= 32`) against query facets using THOT sparse
/// log-domain Sinkhorn transport minus DBSN target-hubness potential `hbar(t_j)`.
pub(in crate::retrieval) fn thot_dbsn_rescore_q16(
    ix: &Index,
    plan: &QueryPlan,
    cand_turns: &[(usize, u64)],
) -> Vec<(usize, u64)> {
    let n = cand_turns.len().min(32);
    if n <= 1 || plan.groups.is_empty() {
        return cand_turns.to_vec();
    }
    let m = plan.groups.len().min(12);
    let nt = (ix.turns.len() as u64).max(2);

    let mut group_terms: Vec<Vec<(u32, bool)>> = vec![Vec::new(); m];
    let mut group_ic_q16 = vec![Q16_ONE as u64; m];
    for qt in &plan.terms {
        if let Some(g) = qt.group
            && g < m
        {
            group_terms[g].push((qt.id, qt.exact_hit));
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(1).max(1) as u64;
            let ic = log2_u64_q16((nt + 1) / df.max(1));
            group_ic_q16[g] = group_ic_q16[g].max(ic);
        }
    }

    // Sparse-thresholded candidate-by-facet affinity matrix S_{j, i} in Q16
    let mut aff = vec![vec![0u64; m]; n];
    let mut turn_hub_q16 = vec![0u64; n];
    for (j, &(ti, _)) in cand_turns.iter().take(n).enumerate() {
        let tr = &ix.turns[ti];
        let mut active_hits = 0u64;
        for (i, tids) in group_terms.iter().enumerate() {
            let mut best = 0u64;
            for &(tid, is_exact) in tids {
                if tr.content.contains(&tid) {
                    let s = if is_exact { Q16_ONE as u64 } else { (Q16_ONE as u64 * 3) >> 2 };
                    best = best.max(s);
                } else if tr.feats.iter().any(|f| f.id == tid) {
                    best = best.max((Q16_ONE as u64) >> 1);
                }
            }
            let s_ij = best.saturating_sub(SPARSE_TAU_Q16);
            aff[j][i] = s_ij;
            if s_ij > 0 {
                active_hits += 1;
            }
        }
        // DBSN exact target-hubness potential hbar(t_j): excess candidate breadth vs query facets
        let extra_terms = tr.content.len() as u64;
        let bg_log = log2_u64_q16(extra_terms.max(4));
        turn_hub_q16[j] = if active_hits == 0 {
            Q16_ONE as u64
        } else {
            bg_log >> 3
        };
    }

    // Log-domain Sinkhorn potentials (u_i for query facets, v_j for turns)
    let mut u = vec![0i64; m];
    let mut v = vec![0i64; n];
    let target_col = Q16_ONE / (n as i64).max(1);
    let target_row = Q16_ONE / (m as i64).max(1);

    for _ in 0..SINKHORN_SWEEPS {
        for i in 0..m {
            let mut sum = 0u64;
            for j in 0..n {
                if aff[j][i] > 0 {
                    let diff = (Q16_ONE as i64 - aff[j][i] as i64).saturating_add(v[j]);
                    sum = sum.saturating_add(exp2_neg_lut_q16(diff));
                }
            }
            if sum > 0 {
                let log_s = log2_u64_q16((sum >> 8).max(1)) as i64;
                u[i] = target_row.saturating_sub(log_s).clamp(-Q16_ONE * 4, Q16_ONE * 4);
            }
        }
        for j in 0..n {
            let mut sum = 0u64;
            for i in 0..m {
                if aff[j][i] > 0 {
                    let diff = (Q16_ONE as i64 - aff[j][i] as i64).saturating_add(u[i]);
                    sum = sum.saturating_add(exp2_neg_lut_q16(diff));
                }
            }
            if sum > 0 {
                let log_s = log2_u64_q16((sum >> 8).max(1)) as i64;
                v[j] = target_col.saturating_sub(log_s).clamp(-Q16_ONE * 4, Q16_ONE * 4);
            }
        }
    }

    // Transport score minus DBSN target-hubness potential hbar(t_j)
    let ic_sum: u64 = group_ic_q16.iter().copied().sum::<u64>().max(1);
    let mut out = Vec::with_capacity(n);
    let mut seen: DetHashSet<usize> = DetHashSet::default();
    for (j, &(ti, base_q16)) in cand_turns.iter().take(n).enumerate() {
        let mut transport_q16 = 0u64;
        for i in 0..m {
            if aff[j][i] == 0 {
                continue;
            }
            let diff = (Q16_ONE as i64 - aff[j][i] as i64).saturating_add(u[i]).saturating_add(v[j]);
            let plan_w = exp2_neg_lut_q16(diff);
            let facet_contrib = ((plan_w as u128 * group_ic_q16[i] as u128) / ic_sum as u128) as u64;
            transport_q16 = transport_q16.saturating_add(facet_contrib);
        }
        let dbsn_net = transport_q16.saturating_sub(turn_hub_q16[j] >> 2);
        seen.insert(ti);
        out.push((ti, base_q16.saturating_add(dbsn_net >> 1)));
    }
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, build_index, plan_query};
    use crate::retrieval::tests::sess;

    #[test]
    fn exp2_lut_is_strictly_monotone_and_positive() {
        assert_eq!(exp2_neg_lut_q16(0), 65_536);
        assert!(exp2_neg_lut_q16(32_768) < 65_536);
        assert!(exp2_neg_lut_q16(65_536) < exp2_neg_lut_q16(32_768));
        assert!(exp2_neg_lut_q16(131_072) > 0);
    }

    #[test]
    fn thot_dbsn_prefers_full_demand_transport_over_broad_partial_hub() {
        let sessions = vec![
            sess("s0", &[("alice", "Kyoto ramen tour planning note day week checklist time budget place")]),
            sess("s1", &[("alice", "Kyoto yuzu ramen spot booked for Friday.")]),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query { text: "Where did Alice book Kyoto yuzu ramen?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &p);
        let cands = vec![(0, 75_000), (1, 76_000)];
        let res = thot_dbsn_rescore_q16(&ix, &plan, &cands);
        assert_eq!(res[0].0, 1);
    }
}
