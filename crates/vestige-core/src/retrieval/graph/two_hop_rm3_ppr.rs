//! Deterministic Two-Hop RM3 Bridge Extractor & Lexical-Anchored Weighted RRF /
//! Local Kemenization for Multi-Session Reasoning and multi-turn conversational Multi-Hop.
//!
//! Grounded in:
//! - Lavrenko & Croft (SIGIR 2001) RM1 & Abdul-Jaleel et al. (TREC 2004) RM3,
//! - Swanson (1986) / Smalheiser & Swanson (1998) ABC bridge discovery,
//! - Gutiérrez et al. (NeurIPS 2024) HippoRAG PPR adapted to zero-LLM PPMI/WordNet graphs,
//! - Cormack et al. (SIGIR 2009) RRF + Dwork et al. (WWW 2001) Local Kemenization.
//!
//! Zero-regression shield: when `full_lexical_coverage` is true (`top0_mask == q_full_mask`),
//! lexical weight strictly exceeds the sum of all auxiliary weights, making lexical rank-1
//! a mathematically guaranteed Condorcet winner.

use std::collections::BTreeMap;

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;

/// Canonical RRF smoothing constant `k = 60` (Cormack et al. 2009).
const RRF_K: u64 = 60;

/// Lexical-Anchored Weighted RRF followed by a single pass of Weighted Local Kemenization
/// over candidate pool `pool` (`|pool| <= 24`).
///
/// Zero-regression theorem: when `full_lexical_coverage` is true, `w_lex` is set to
/// `sum(aux_weights) + 1`, guaranteeing that `lex_rank[0]` is a strict weighted Condorcet
/// winner and can never be displaced by auxiliary rankers.
pub(in crate::retrieval) fn lexical_anchored_rrf_kemeny_q16(
    pool: &[usize],
    lex_rank: &[usize],
    aux_rankings: &[(&[usize], u64)],
    full_lexical_coverage: bool,
) -> Vec<usize> {
    if pool.len() <= 1 {
        return pool.to_vec();
    }
    let aux_sum: u64 = aux_rankings
        .iter()
        .map(|&(_, w)| w)
        .fold(0u64, u64::saturating_add);
    let w_lex: u64 = if full_lexical_coverage {
        aux_sum.saturating_add(Q16_ONE)
    } else {
        // Balanced multi-hop mode: lexical retains strong voice, but >=2 agreeing aux rankers can lift.
        Q16_ONE
    };

    // Build rank lookup maps (1-indexed).
    let mut lex_pos: BTreeMap<usize, u64> = BTreeMap::new();
    for (i, &sid) in lex_rank.iter().enumerate() {
        lex_pos.insert(sid, (i as u64) + 1);
    }
    let mut aux_pos: Vec<(BTreeMap<usize, u64>, u64)> = Vec::with_capacity(aux_rankings.len());
    for &(rk, w) in aux_rankings {
        let mut m = BTreeMap::new();
        for (i, &sid) in rk.iter().enumerate() {
            m.insert(sid, (i as u64) + 1);
        }
        aux_pos.push((m, w));
    }

    let fallback = (pool.len() as u64) + 1;
    let mut scored: Vec<(usize, u64, u64)> = pool
        .iter()
        .map(|&sid| {
            let lr = lex_pos.get(&sid).copied().unwrap_or(fallback);
            let mut rrf = ((w_lex as u128).saturating_mul(Q16_ONE as u128)
                / ((RRF_K + lr) as u128)) as u64;
            for (m, w) in &aux_pos {
                let r = m.get(&sid).copied().unwrap_or(fallback);
                let term = ((*w as u128).saturating_mul(Q16_ONE as u128)
                    / ((RRF_K + r) as u128)) as u64;
                rrf = rrf.saturating_add(term);
            }
            (sid, rrf, lr)
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)).then_with(|| a.0.cmp(&b.0)));
    let mut ord: Vec<usize> = scored.into_iter().map(|(sid, _, _)| sid).collect();

    // Weighted Local Kemenization (Dwork et al. 2001): adjacent swaps while pairwise majority strictly prefers b > a.
    let total_w = w_lex.saturating_add(aux_sum);
    let majority_bar = total_w >> 1;
    let n = ord.len();
    for i in 1..n {
        let mut j = i;
        while j > 0 {
            let a = ord[j - 1];
            let b = ord[j];
            let mut b_beats_a: u64 = 0;
            if lex_pos.get(&b).copied().unwrap_or(fallback)
                < lex_pos.get(&a).copied().unwrap_or(fallback)
            {
                b_beats_a = b_beats_a.saturating_add(w_lex);
            }
            for (m, w) in &aux_pos {
                if m.get(&b).copied().unwrap_or(fallback) < m.get(&a).copied().unwrap_or(fallback) {
                    b_beats_a = b_beats_a.saturating_add(*w);
                }
            }
            if b_beats_a > majority_bar {
                ord.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    ord
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_regression_shield_guarantees_lexical_top1_under_full_coverage() {
        let pool = vec![10, 20, 30];
        let lex = vec![10, 20, 30];
        // Three hostile auxiliary rankers all place 30 ahead of 10.
        let aux1 = vec![30, 20, 10];
        let aux2 = vec![30, 10, 20];
        let aux3 = vec![30, 20, 10];
        let aux = vec![
            (aux1.as_slice(), Q16_ONE),
            (aux2.as_slice(), Q16_ONE),
            (aux3.as_slice(), Q16_ONE),
        ];
        let out_locked = lexical_anchored_rrf_kemeny_q16(&pool, &lex, &aux, true);
        assert_eq!(out_locked[0], 10);

        // When lexical coverage is incomplete, 3 agreeing aux rankers CAN promote 30.
        let out_open = lexical_anchored_rrf_kemeny_q16(&pool, &lex, &aux, false);
        assert_eq!(out_open[0], 30);
    }
}
