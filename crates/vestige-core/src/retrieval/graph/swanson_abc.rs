//! Swanson's ABC / `AnC` Literature-Based Discovery (`Q16.16`, Swanson 1986).
//!
//! Discovers implicit `A -> B_1..B_n -> C` connections where candidate session `C` shares
//! `>= 2` independent rare intermediate linking terms `B` (`2 <= df(B) <= 8`, `LTC >= 2`)
//! with anchor literature `A`, scored by Arrowsmith minimum-link weight `min(w(A,b), w(b,C))`.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::BTreeMap;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const MAX_A_ANCHORS: usize = 3;
const MIN_LINKING_TERM_COUNT: u32 = 2;
const B_MIN_DF: u32 = 2;
const B_MAX_DF: u32 = 8;
const SWANSON_COEFF_MULTI_Q16: u64 = 22_938; // 0.35
const SWANSON_COEFF_BASE_Q16: u64 = 11_796;  // 0.18

pub(in crate::retrieval) fn score_swanson_anc_q16(
    ns: usize,
    a_set: &[usize],
    b_weights_from_a: &BTreeMap<u32, u64>,
    b_postings: &BTreeMap<u32, Vec<usize>>,
) -> Vec<u64> {
    let mut ltc = vec![0u32; ns];
    let mut arrow_sum = vec![0u64; ns];
    for (&b_id, &w_ab) in b_weights_from_a {
        let Some(sessions) = b_postings.get(&b_id) else {
            continue;
        };
        let df = sessions.len() as u64;
        if df == 0 {
            continue;
        }
        let min_link = w_ab.min(Q16_ONE / df);
        for &si in sessions {
            if si < ns && !a_set.contains(&si) {
                ltc[si] = ltc[si].saturating_add(1);
                arrow_sum[si] = arrow_sum[si].saturating_add(min_link);
            }
        }
    }
    let mut out = vec![0u64; ns];
    for (si, slot) in out.iter_mut().enumerate() {
        if ltc[si] >= MIN_LINKING_TERM_COUNT {
            *slot = arrow_sum[si];
        }
    }
    out
}

pub(in crate::retrieval) fn compute_swanson_abc_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    multi_or_partial: bool,
    base_sight_q16: &[u64],
    s_ord_base: &[usize],
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let Some(&top0) = s_ord_base.first() else {
        return out;
    };
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let top0_sight = base_sight_q16[top0];
    if top0_sight == 0 {
        return out;
    }

    let a_set: Vec<usize> = s_ord_base
        .iter()
        .copied()
        .filter(|&si| si < ns && base_sight_q16[si].saturating_mul(5) >= top0_sight.saturating_mul(2))
        .take(MAX_A_ANCHORS)
        .collect();
    if a_set.is_empty() {
        return out;
    }

    let mut b_postings: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (si, ents) in ix.session_entities.iter().enumerate() {
        for &eid in ents {
            if !plan.group_of.contains_key(&eid) {
                b_postings.entry(eid).or_default().push(si);
            }
        }
    }
    for (si, rares) in ix.session_rare_nt.iter().enumerate() {
        for &tid in rares {
            if !plan.group_of.contains_key(&tid) {
                let list = b_postings.entry(tid).or_default();
                if list.last().copied() != Some(si) {
                    list.push(si);
                }
            }
        }
    }
    b_postings.retain(|_, list| {
        let df = list.len() as u32;
        (B_MIN_DF..=B_MAX_DF).contains(&df)
    });
    if b_postings.is_empty() {
        return out;
    }

    let mut b_weights_from_a: BTreeMap<u32, u64> = BTreeMap::new();
    for (&b_id, sessions) in &b_postings {
        let a_hits = sessions.iter().filter(|s| a_set.contains(s)).count() as u64;
        if a_hits > 0 {
            let df = sessions.len() as u64;
            let ic = log2_u64_q16(((ns as u64) + 1) / df.max(1)).max(Q16_ONE >> 1);
            b_weights_from_a.insert(b_id, (ic.saturating_mul(a_hits)) / (a_set.len() as u64));
        }
    }
    if b_weights_from_a.len() < MIN_LINKING_TERM_COUNT as usize {
        return out;
    }

    let anc = score_swanson_anc_q16(ns, &a_set, &b_weights_from_a, &b_postings);
    let max_anc = anc.iter().copied().max().unwrap_or(0);
    if max_anc == 0 {
        return out;
    }

    let coeff = if multi_or_partial {
        SWANSON_COEFF_MULTI_Q16
    } else {
        SWANSON_COEFF_BASE_Q16
    };
    for (si, &sc) in anc.iter().enumerate() {
        if si != top0 && sc > 0 {
            let norm_q16 = (sc.saturating_mul(Q16_ONE)) / max_anc;
            out[si] = (((smax_q16.saturating_mul(coeff)) >> Q16_SHIFT).saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swanson_anc_enforces_ltc_at_least_two() {
        let a_set = vec![0usize];
        let mut b_a = BTreeMap::new();
        b_a.insert(10, Q16_ONE);
        b_a.insert(20, Q16_ONE);
        let mut bp = BTreeMap::new();
        bp.insert(10, vec![0, 2, 3]);
        bp.insert(20, vec![0, 2]);
        let sc = score_swanson_anc_q16(4, &a_set, &b_a, &bp);
        assert!(sc[2] > 0 && sc[3] == 0 && sc[0] == 0);
    }
}
