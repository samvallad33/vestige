//! #829: Moffat & Zobel (2008) Rank-Biased Precision (RBP) Slot Allocation
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! RBP models a user scanning ranked slots `r = 0, 1, 2, ...` with persistence
//! parameter `p = 13/16` (`53248` in Q16.16).
//!
//! In conversational multi-session retrieval, redundant short-question, backchannel,
//! or duplicate assistant turns sitting in tail slots `5..k` have lower marginal RBP
//! utility than an unseated declarative exchange partner (`pos - 1` or `pos + 1`)
//! that carries positive lexical grounding. This pass reallocates redundant tail slots
//! (`5..k`, keeping `0..5` 100% sealed!) without ever evicting substantive standalone
//! question turns that carry higher own-turn lexical score than the incoming candidate.

use super::dph::Q16;
use super::index::Index;
use super::rocchio::score_turn_rocchio_q16;
use super::supersession::turn_is_superseded;
use std::collections::BTreeMap;

/// Persistence parameter `p = 13 / 16` in Q16.16 (`53_248`).
pub(super) const RBP_P_Q16: u64 = (Q16 as u64 * 13) >> 4;

/// Compute `p^exp` in Q16.16 for `exp in 0..16`.
#[inline]
pub(super) fn rbp_pow_q16(exp: usize) -> u64 {
    let mut acc = Q16 as u64;
    let steps = exp.min(16);
    for _ in 0..steps {
        acc = ((acc as u128 * RBP_P_Q16 as u128) >> 16) as u64;
    }
    acc
}

/// Compute the RBP-weighted marginal gain in Q16.16 of seating turn `ti`.
#[inline]
pub(super) fn rbp_marginal_gain_q16(
    score_q16: u64,
    already_seated_in_session: usize,
    is_question: bool,
) -> u64 {
    let session_discount_q16 = rbp_pow_q16(already_seated_in_session.saturating_mul(2));
    let role_q16 = if is_question {
        (Q16 as u64 * 10) >> 4
    } else {
        Q16 as u64
    };
    let combined_q16 = ((session_discount_q16 as u128 * role_q16 as u128) >> 16) as u64;
    ((score_q16 as u128 * combined_q16 as u128) >> 16) as u64
}

/// Apply Moffat RBP tail slot reallocation across `final_turn_indices[5..k]`
/// (keeping `final_turn_indices[0..5]` 100% sealed so `@1` and `@5` never regress).
pub(super) fn apply_rbp_tail_allocation(
    ix: &Index,
    prf_centroid: &BTreeMap<u32, u64>,
    turn_score_q16: &[u64],
    t_own_q16: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
) {
    let len = final_turn_indices.len().min(k);
    if len <= 5 {
        return;
    }

    let has_redundant_tail = (5..len).any(|slot| {
        let vti = final_turn_indices[slot];
        let vt = &ix.turns[vti];
        let prior_in_sess = final_turn_indices[..slot]
            .iter()
            .filter(|&&e| ix.turns[e].session == vt.session)
            .count();
        vt.is_short_question
            || vt.is_backchannel
            || (prior_in_sess >= 1 && (vt.is_question || vt.field == 1))
    });
    if !has_redundant_tail {
        return;
    }

    let mut candidates: Vec<(usize, u64)> = Vec::new();
    for idx in 0..len {
        let seated_ti = final_turn_indices[idx];
        let seated_t = &ix.turns[seated_ti];
        if !seated_t.is_question && idx >= 3 {
            continue;
        }
        let sess = seated_t.session;
        let members = &ix.session_turns[sess];
        let Some(pos) = members.iter().position(|&m| m == seated_ti) else {
            continue;
        };

        let mut try_push = |cand_ti: usize, mult_num: u64| {
            let ct = &ix.turns[cand_ti];
            if final_turn_indices.contains(&cand_ti)
                || ct.is_question
                || ct.is_backchannel
                || turn_is_superseded(ix, cand_ti)
            {
                return;
            }
            // Require genuine lexical grounding OR direct response/opener pair to a question reaction
            if t_own_q16[cand_ti] == 0 && !seated_t.is_question {
                return;
            }
            let base = turn_score_q16[cand_ti].max((turn_score_q16[seated_ti] * 3) >> 2);
            if base == 0 {
                return;
            }
            let rocchio = score_turn_rocchio_q16(ix, cand_ti, prf_centroid) >> 4;
            let u_q16 = ((base.saturating_add(rocchio)) * mult_num) >> 2;
            candidates.push((cand_ti, u_q16));
        };

        if seated_t.is_question {
            if pos >= 1 {
                try_push(members[pos - 1], 6);
            }
            if pos + 1 < members.len() {
                try_push(members[pos + 1], 6);
            }
        } else {
            if pos >= 1 {
                try_push(members[pos - 1], 5);
            }
            if pos + 1 < members.len() {
                try_push(members[pos + 1], 5);
            }
        }
    }

    if candidates.is_empty() {
        return;
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut unique_cands: Vec<(usize, u64)> = Vec::new();
    for (ti, sc) in candidates {
        if !unique_cands.iter().any(|&(existing, _)| existing == ti) {
            unique_cands.push((ti, sc));
        }
    }

    let mut swaps_done = 0usize;
    for (cand_ti, cand_raw_q16) in unique_cands {
        if swaps_done >= 2 || final_turn_indices.contains(&cand_ti) {
            continue;
        }
        let cand_sess = ix.turns[cand_ti].session;
        let cand_sess_count = final_turn_indices[..len]
            .iter()
            .filter(|&&t| ix.turns[t].session == cand_sess)
            .count();
        let cand_gain_q16 = rbp_marginal_gain_q16(cand_raw_q16, cand_sess_count, false);

        let mut worst_slot: Option<(usize, u64)> = None;
        for slot in (5..len).rev() {
            let vti = final_turn_indices[slot];
            let vt = &ix.turns[vti];
            let prior_in_sess = final_turn_indices[..slot]
                .iter()
                .filter(|&&t| ix.turns[t].session == vt.session)
                .count();
            let is_redundant = vt.is_short_question
                || vt.is_backchannel
                || (prior_in_sess >= 1 && (vt.is_question || vt.field == 1));
            if !is_redundant {
                continue;
            }
            if !vt.is_short_question && t_own_q16[vti] >= t_own_q16[cand_ti] && t_own_q16[vti] > 0 {
                continue;
            }
            let v_gain_q16 = rbp_marginal_gain_q16(
                turn_score_q16[vti],
                prior_in_sess,
                vt.is_question,
            );
            if cand_gain_q16 > v_gain_q16 {
                match worst_slot {
                    None => worst_slot = Some((slot, v_gain_q16)),
                    Some((_, best_v)) if v_gain_q16 < best_v => worst_slot = Some((slot, v_gain_q16)),
                    _ => {}
                }
            }
        }

        if let Some((slot, _)) = worst_slot {
            final_turn_indices[slot] = cand_ti;
            swaps_done += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbp_pow_decays_monotonically() {
        let p0 = rbp_pow_q16(0);
        let p1 = rbp_pow_q16(1);
        let p2 = rbp_pow_q16(2);
        assert_eq!(p0, Q16 as u64);
        assert!(p1 < p0 && p2 < p1);
    }
}
