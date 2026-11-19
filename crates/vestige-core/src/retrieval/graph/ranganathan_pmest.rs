//! Ranganathan's Colon Classification 5-Facet `PMEST` Session Lift (`Q16.16`, Ranganathan 1933).
//!
//! Aggregates turn-level `PmestSignature` (`[Personality, Matter, Energy, Space, Time]`) across
//! declarative turns in each session, scoring both direct query-facet alignment and complementary
//! facet coverage on facets weak or absent in `top0` (crucial for Open-Domain synthesis).
//! Zero floats, zero open-class wordlists.

use crate::retrieval::graph::taxa::PmestSignature;
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const PMEST_COEFF_Q16: u64 = 13_107; // 0.20

#[inline]
pub(in crate::retrieval) fn union_pmest(a: PmestSignature, b: PmestSignature) -> PmestSignature {
    PmestSignature {
        personality: a.personality | b.personality,
        matter: a.matter | b.matter,
        energy: a.energy | b.energy,
        space: a.space | b.space,
        time: a.time | b.time,
    }
}

#[inline]
pub(in crate::retrieval) fn complementary_pmest_gain_q16(
    q_sig: PmestSignature,
    top0_sig: PmestSignature,
    cand_sig: PmestSignature,
) -> u64 {
    let base_align = q_sig.align_q16(cand_sig);
    let mut novel_facets = 0u64;
    for (t0, c) in [
        (top0_sig.personality, cand_sig.personality),
        (top0_sig.matter, cand_sig.matter),
        (top0_sig.energy, cand_sig.energy),
        (top0_sig.space, cand_sig.space),
        (top0_sig.time, cand_sig.time),
    ] {
        if (c & !t0) != 0 {
            novel_facets += 1;
        }
    }
    base_align.saturating_add((novel_facets.saturating_mul(Q16_ONE)) / 5)
}

pub(in crate::retrieval) fn compute_ranganathan_pmest_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }

    let q_sig = PmestSignature::for_query(plan);
    let mut sess_sigs = vec![PmestSignature::default(); ns];
    for (si, members) in ix.session_turns.iter().enumerate() {
        let mut acc = PmestSignature::default();
        for &ti in members {
            let t = &ix.turns[ti];
            if !t.is_question && !t.is_backchannel {
                acc = union_pmest(acc, PmestSignature::for_turn(ix, ti));
            }
        }
        sess_sigs[si] = acc;
    }

    let top0_sig = sess_sigs[top0];
    let mut raw = vec![0u64; ns];
    for (si, &sig) in sess_sigs.iter().enumerate() {
        if si != top0 && s_exact_mask[si] != 0 {
            raw[si] = complementary_pmest_gain_q16(q_sig, top0_sig, sig);
        }
    }
    let max_r = raw.iter().copied().max().unwrap_or(0);
    if max_r == 0 {
        return out;
    }
    for (si, &r) in raw.iter().enumerate() {
        if si != top0 && r > 0 {
            let norm_q16 = (r.saturating_mul(Q16_ONE)) / max_r;
            out[si] = (((smax_q16.saturating_mul(PMEST_COEFF_Q16)) >> Q16_SHIFT)
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
    fn test_pmest_rewards_complementary_space_and_time_facets() {
        let q = PmestSignature { personality: 1, matter: 1, energy: 0, space: 1, time: 1 };
        let top0 = PmestSignature { personality: 1, matter: 1, energy: 0, space: 0, time: 0 };
        let comp = PmestSignature { personality: 1, matter: 1, energy: 0, space: 1, time: 1 };
        let dup = PmestSignature { personality: 1, matter: 1, energy: 0, space: 0, time: 0 };
        assert!(
            complementary_pmest_gain_q16(q, top0, comp)
                > complementary_pmest_gain_q16(q, top0, dup)
        );
    }
}
