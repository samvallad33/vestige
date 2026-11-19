//! Statistical Translation Language Model + 2-Hop Lesk/WordNet Bridge (`Q16.16`).
//!
//! Implements Berger & Lafferty (SIGIR 1999) statistical translation `P_trans(q_g | d)`
//! combined with Lesk (SIGDOC 1986) / Banerjee & Pedersen (2002) 2-hop WordNet expansion.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::fusion::kemeny_mrf::{ieee754_bits_to_q16, log2_u64_q16};
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::{BTreeMap, BTreeSet};

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const MAX_PPMI_PER_TERM: usize = 16;
const MAX_WN_HOP1: usize = 16;
const MAX_WN_HOP2: usize = 8;
const WN_HOP1_WEIGHT_Q16: u64 = 32_768; // 0.50
const WN_HOP2_WEIGHT_Q16: u64 = 16_384; // 0.25
const TLM_COEFF_MULTI_Q16: u64 = 23_593; // 0.36
const TLM_COEFF_BASE_Q16: u64 = 11_796;  // 0.18

/// Conjunctive `Q16.16` log-space translation likelihood across query groups.
pub(in crate::retrieval) fn conjunctive_translation_log_score_q16(
    ns: usize,
    group_masses: &[Vec<u64>],
    group_ic_q16: &[u64],
) -> Vec<u64> {
    let mut scores = vec![0u64; ns];
    for (g, masses) in group_masses.iter().enumerate() {
        let ic = group_ic_q16[g];
        for (si, &m) in masses.iter().enumerate() {
            if m > 0 {
                let log_term = log2_u64_q16(1 + (m >> 8)).max(Q16_ONE >> 3);
                let contrib = (ic.saturating_mul(log_term)) >> Q16_SHIFT;
                scores[si] = scores[si].saturating_add(contrib);
            }
        }
    }
    scores
}

pub(in crate::retrieval) fn compute_translation_lm_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    multi_or_partial: bool,
    s_exact_mask: &[u64],
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let n_groups = plan.groups.len().min(64);
    if ns <= 5 || top0 >= ns || n_groups == 0 || smax_q16 == 0 {
        return out;
    }

    let max_df = ((ns / 3).max(2)) as u32;
    let mut group_masses: Vec<Vec<u64>> = vec![vec![0u64; ns]; n_groups];
    let mut group_ic_q16: Vec<u64> = vec![Q16_ONE; n_groups];
    let mut group_qids: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n_groups];
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            if g < n_groups {
                group_qids[g].insert(qt.id);
            }
        }
    }

    for g in 0..n_groups {
        let mut min_df = ns as u64;
        let mut trans_vocab: BTreeMap<u32, u64> = BTreeMap::new();
        for &qid in &group_qids[g] {
            let df = ix.s_stats.df.get(&qid).copied().unwrap_or(1).max(1) as u64;
            min_df = min_df.min(df);
            if let Some(nbrs) = ix.ppmi_neighbors.get(&qid) {
                for &(pid, pw) in nbrs.iter().take(MAX_PPMI_PER_TERM) {
                    let pdf = ix.s_stats.df.get(&pid).copied().unwrap_or(0);
                    if pdf == 0 || pdf > max_df {
                        continue;
                    }
                    let pw_q16 = ieee754_bits_to_q16(pw.to_bits()).min(Q16_ONE >> 1);
                    if pw_q16 > 0 {
                        let e = trans_vocab.entry(pid).or_insert(0);
                        *e = (*e).max(pw_q16);
                    }
                }
            }
        }
        group_ic_q16[g] = log2_u64_q16(((ns as u64) + 1) / min_df.max(1)).max(Q16_ONE >> 1);

        let g_word = &plan.groups[g];
        let hop1: Vec<&'static str> =
            crate::retrieval::graph::bridge::wordnet_lemmas_with_morph(g_word)
                .into_iter()
                .take(MAX_WN_HOP1)
                .collect();
        for &l1 in &hop1 {
            if let Some(vid) = ix.vocab.get(l1) {
                let pdf = ix.s_stats.df.get(&vid).copied().unwrap_or(0);
                if pdf > 0 && pdf <= max_df {
                    let e = trans_vocab.entry(vid).or_insert(0);
                    *e = (*e).max(WN_HOP1_WEIGHT_Q16);
                }
            }
            for l2 in crate::wordnet::upward_hypernyms(l1, 2)
                .iter()
                .copied()
                .take(MAX_WN_HOP2)
            {
                if let Some(vid2) = ix.vocab.get(l2) {
                    let pdf = ix.s_stats.df.get(&vid2).copied().unwrap_or(0);
                    if pdf > 0 && pdf <= max_df {
                        let e = trans_vocab.entry(vid2).or_insert(0);
                        *e = (*e).max(WN_HOP2_WEIGHT_Q16);
                    }
                }
            }
        }

        for (wid, tw) in trans_vocab {
            if let Some(postings) = ix.s_postings.get(wid as usize) {
                for &si_u32 in postings {
                    let si = si_u32 as usize;
                    if si < ns && si != top0 {
                        let mult = if (s_exact_mask[si] & (1u64 << g)) == 0 { 2u64 } else { 1u64 };
                        group_masses[g][si] =
                            group_masses[g][si].saturating_add(tw.saturating_mul(mult));
                    }
                }
            }
        }
    }

    let tlm = conjunctive_translation_log_score_q16(ns, &group_masses, &group_ic_q16);
    let max_tlm = tlm
        .iter()
        .enumerate()
        .filter(|&(si, _)| si != top0)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_tlm == 0 {
        return out;
    }

    let coeff = if multi_or_partial {
        TLM_COEFF_MULTI_Q16
    } else {
        TLM_COEFF_BASE_Q16
    };
    for (si, &sc) in tlm.iter().enumerate() {
        if si != top0 && sc > 0 {
            let norm_q16 = (sc.saturating_mul(Q16_ONE)) / max_tlm;
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
    fn test_conjunctive_translation_prefers_multi_group_support() {
        let masses = vec![vec![0, Q16_ONE, Q16_ONE * 8], vec![0, Q16_ONE, 0]];
        let ic = vec![Q16_ONE * 2, Q16_ONE * 2];
        let sc = conjunctive_translation_log_score_q16(3, &masses, &ic);
        assert!(sc[1] > sc[2] && sc[0] == 0);
    }
}
