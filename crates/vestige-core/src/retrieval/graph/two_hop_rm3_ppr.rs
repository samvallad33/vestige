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

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;

/// Canonical RRF smoothing constant `k = 60` (Cormack et al. 2009).
const RRF_K: u64 = 60;

/// Extract top-`max_bridges` high-IC RM1/RM3 bridge entities from Hop-1 seed sessions.
/// Only entities occurring in `>= 2` corpus sessions (`df >= 2`) and absent from the
/// original query `query_terms` qualify as bridges (zero single-session noise).
pub(in crate::retrieval) fn extract_rm3_bridge_entities_q16(
    hop1_sessions: &[(usize, u64, &[(u32, u64)])],
    query_terms: &BTreeSet<u32>,
    term_df: &BTreeMap<u32, u32>,
    term_ic_q16: &BTreeMap<u32, u64>,
    max_bridges: usize,
) -> BTreeMap<u32, u64> {
    let mut out: BTreeMap<u32, u64> = BTreeMap::new();
    if hop1_sessions.is_empty() || max_bridges == 0 {
        return out;
    }
    let total_prior: u64 = hop1_sessions
        .iter()
        .map(|&(_, sc, _)| sc)
        .fold(0u64, u64::saturating_add)
        .max(1);

    let mut rm1_acc: BTreeMap<u32, u64> = BTreeMap::new();
    for &(_sid, sc_q16, terms) in hop1_sessions {
        if sc_q16 == 0 {
            continue;
        }
        let doc_len: u64 = terms
            .iter()
            .map(|&(_, tf)| tf)
            .fold(0u64, u64::saturating_add)
            .max(1);
        let p_doc = (((sc_q16 as u128) << 16) / (total_prior as u128)) as u64;
        for &(tid, tf) in terms {
            if query_terms.contains(&tid) {
                continue;
            }
            if term_df.get(&tid).copied().unwrap_or(0) < 2 {
                continue;
            }
            let p_w_d = (((tf as u128) << 16) / (doc_len as u128)) as u64;
            let ic = term_ic_q16.get(&tid).copied().unwrap_or(0);
            let contrib = (((p_doc as u128).saturating_mul(p_w_d as u128) >> 16)
                .saturating_mul(ic as u128)
                >> 16) as u64;
            if contrib > 0 {
                let e = rm1_acc.entry(tid).or_insert(0);
                *e = e.saturating_add(contrib);
            }
        }
    }

    let mut ranked: Vec<(u32, u64)> = rm1_acc.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(max_bridges);
    for (tid, w) in ranked {
        out.insert(tid, w);
    }
    out
}

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

    #[test]
    fn rm3_extracts_only_cross_session_high_ic_bridges() {
        let s0_terms = [(1u32, 2u64), (2u32, 3u64), (3u32, 5u64)];
        let mut q = BTreeSet::new();
        q.insert(1); // already in query
        let mut df = BTreeMap::new();
        df.insert(1, 3);
        df.insert(2, 2); // valid bridge (df >= 2)
        df.insert(3, 1); // singleton noise (df < 2)
        let mut ic = BTreeMap::new();
        ic.insert(1, Q16_ONE * 3);
        ic.insert(2, Q16_ONE * 4);
        ic.insert(3, Q16_ONE * 5);
        let bridges = extract_rm3_bridge_entities_q16(&[(0, Q16_ONE, &s0_terms)], &q, &df, &ic, 4);
        assert!(bridges.contains_key(&2));
        assert!(!bridges.contains_key(&1));
        assert!(!bridges.contains_key(&3));
    }
}
