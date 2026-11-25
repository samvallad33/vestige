//! Pre-Cut Session `BM25TP` IDF-Weighted Term Proximity (`Q16.16`, Rasolofo & Savoy ECIR 2003).
//!
//! Scores tight turn-internal co-occurrence of distinct query groups (`min(IC_a, IC_b) / dist^2`
//! for `1 <= dist <= 12`) with integer BM25 saturation prior to the Stage 05 top-10 cut.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::BTreeMap;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const MAX_PROX_DIST: usize = 12;
const BM25TP_K1: u64 = 2;
const BM25TP_COEFF_Q16: u64 = 19_661; // 0.30

pub(in crate::retrieval) fn turn_tpi_q16(hits: &[(usize, usize)], group_ic_q16: &[u64]) -> u64 {
    let mut tpi = 0u64;
    for (i, &(pa, ga)) in hits.iter().enumerate() {
        for &(pb, gb) in &hits[(i + 1)..] {
            let dist = pb.saturating_sub(pa);
            if dist > MAX_PROX_DIST {
                break;
            }
            if dist == 0 || ga == gb {
                continue;
            }
            let d2 = (dist as u64).saturating_mul(dist as u64);
            tpi = tpi.saturating_add(group_ic_q16[ga].min(group_ic_q16[gb]) / d2);
        }
    }
    tpi
}

pub(in crate::retrieval) fn compute_session_bm25tp_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let n_groups = plan.groups.len().min(64);
    if ns < 2 || top0 >= ns || n_groups < 2 || smax_q16 == 0 {
        return out;
    }

    let q_mask = if n_groups == 64 {
        u64::MAX
    } else {
        (1u64 << n_groups) - 1
    };
    let mut tid_to_group: BTreeMap<u32, usize> = BTreeMap::new();
    let mut group_min_df = vec![ns as u64; n_groups];
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            if g < n_groups {
                tid_to_group.insert(qt.id, g);
                let df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(1).max(1) as u64;
                group_min_df[g] = group_min_df[g].min(df);
            }
        }
    }
    let group_ic_q16: Vec<u64> = group_min_df
        .iter()
        .map(|&df| log2_u64_q16(((ns as u64) + 1) / df.max(1)).max(Q16_ONE >> 1))
        .collect();

    let mut sess_tp = vec![0u64; ns];
    let mut hits: Vec<(usize, usize)> = Vec::new();
    for (si, members) in ix.session_turns.iter().enumerate() {
        if si == top0 || (s_exact_mask[si] & q_mask).count_ones() < 2 {
            continue;
        }
        let mut best_turn_tpi = 0u64;
        for &ti in members {
            let t = &ix.turns[ti];
            if t.is_backchannel || t.seq.len() < 2 {
                continue;
            }
            hits.clear();
            for (pos, &(ex_id, st_id)) in t.seq.iter().enumerate() {
                if let Some(&g) = tid_to_group.get(&ex_id).or_else(|| tid_to_group.get(&st_id)) {
                    hits.push((pos, g));
                }
            }
            if hits.len() >= 2 {
                best_turn_tpi = best_turn_tpi.max(turn_tpi_q16(&hits, &group_ic_q16));
            }
        }
        if best_turn_tpi > 0 {
            let num = best_turn_tpi
                .saturating_mul(BM25TP_K1 + 1)
                .saturating_mul(Q16_ONE);
            let den = best_turn_tpi.saturating_add(BM25TP_K1.saturating_mul(Q16_ONE));
            sess_tp[si] = num / den.max(1);
        }
    }

    let max_tp = sess_tp.iter().copied().max().unwrap_or(0);
    if max_tp == 0 {
        return out;
    }
    for (si, &tp) in sess_tp.iter().enumerate() {
        if si != top0 && tp > 0 {
            let norm_q16 = (tp.saturating_mul(Q16_ONE)) / max_tp;
            out[si] = (((smax_q16.saturating_mul(BM25TP_COEFF_Q16)) >> Q16_SHIFT)
                .saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25tp_rewards_adjacent_query_groups() {
        let ic = vec![Q16_ONE * 3, Q16_ONE * 3];
        assert!(turn_tpi_q16(&[(1, 0), (2, 1)], &ic) > turn_tpi_q16(&[(1, 0), (10, 1)], &ic) * 10);
    }
}
