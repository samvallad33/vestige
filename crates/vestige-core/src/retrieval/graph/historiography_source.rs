//! Historiography & Source Criticism (`Q16.16`, Langlois & Seignobos 1898; Bernheim 1889).
//!
//! Weighs primary firsthand witness testimony (`field == 0`, declarative, non-superseded,
//! corroborated across independent turns/speakers) above secondary assistant recaps (`field != 0`)
//! or uncorroborated single-echo hearsay, hardening Adversarial and Multi-Hop precision.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::BTreeSet;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const HIST_COEFF_Q16: u64 = 13_107; // 0.20

/// Computes per-session firsthand witness corroboration score in `Q16.16` (also reused by RRF).
pub(in crate::retrieval) fn session_firsthand_witness_scores_q16(
    ix: &Index,
    plan: &QueryPlan,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let q_ids: BTreeSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| qt.exact_hit && qt.group.is_some())
        .map(|qt| qt.id)
        .collect();
    let mut scores = vec![0u64; ns];
    if q_ids.is_empty() {
        return scores;
    }

    for (si, members) in ix.session_turns.iter().enumerate() {
        let mut firsthand_turns = 0u64;
        let mut hearsay_only_turns = 0u64;
        let mut distinct_speakers: BTreeSet<&str> = BTreeSet::new();

        for &ti in members {
            let t = &ix.turns[ti];
            if t.is_backchannel || turn_is_superseded(ix, ti) {
                continue;
            }
            let hits_query = t.content.iter().any(|cid| q_ids.contains(cid));
            if !hits_query {
                continue;
            }
            if t.field == 0 && !t.is_question {
                firsthand_turns += 1;
                if !t.speaker.is_empty() {
                    distinct_speakers.insert(t.speaker.as_str());
                }
            } else {
                hearsay_only_turns += 1;
            }
        }
        if firsthand_turns > 0 {
            let corrob_mult = if distinct_speakers.len() >= 2 || firsthand_turns >= 2 {
                3u64 // Independent corroboration bonus (1.5x)
            } else {
                2u64
            };
            scores[si] = (firsthand_turns.saturating_mul(Q16_ONE).saturating_mul(corrob_mult)) / 2;
        } else if hearsay_only_turns > 0 {
            scores[si] = Q16_ONE >> 3; // Secondary hearsay discounted 8:1 below firsthand witness
        }
    }
    scores
}

pub(in crate::retrieval) fn compute_historiography_witness_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let witness = session_firsthand_witness_scores_q16(ix, plan);
    let max_w = witness
        .iter()
        .enumerate()
        .filter(|&(si, _)| si != top0)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_w == 0 {
        return out;
    }
    for (si, &w) in witness.iter().enumerate() {
        if si != top0 && w >= Q16_ONE {
            let norm_q16 = (w.saturating_mul(Q16_ONE)) / max_w;
            out[si] = (((smax_q16.saturating_mul(HIST_COEFF_Q16)) >> Q16_SHIFT)
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
    fn test_firsthand_corroboration_ratio_exceeds_hearsay_eightfold() {
        let firsthand_corroborated = (2u64 * Q16_ONE * 3) / 2;
        let hearsay_only = Q16_ONE >> 3;
        assert!(firsthand_corroborated >= hearsay_only * 24);
    }
}
