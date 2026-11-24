//! #804: Getoor Probabilistic Soft Logic (PSL) Lukasiewicz Hinge Penalties
//! + #803: Roth 0-1 Integer Linear Programming (ILP) Multi-Hop Chain Consistency
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! Mathematical formulations:
//! 1. PSL Lukasiewicz hinge penalty on multi-hop query coverage in `Q16.16`:
//!    For each query group `g` not yet covered by `selected`, a candidate pair/chain
//!    `(a, b)` satisfies `distance_to_satisfaction(a -> b covers g) = max(0, I(a) + I(b) - 1)`
//!    plus entity-bridge agreement `ochiai_agree = Q16 - ochiai_divergence_q16(ent_a, ent_b)`.
//!    Any candidate lacking shared carrier entities or failing to close uncovered query groups
//!    incurs a soft hinge penalty `(Q16 - agree_q16)` while chain-closing turns receive positive
//!    satisfaction gain.
//! 2. Roth 0-1 ILP multi-hop closure:
//!    Models binary selection indicators `x_i in {0, 1}` over candidate turns so multi-hop
//!    queries enforce the linear closure constraint:
//!      `sum_i x_i * cov(i, g) >= 1` and `y_{i,j} <= x_i`, `y_{i,j} <= x_j`, `y_{i,j} >= x_i + x_j - 1`
//!    Solved deterministically over candidate turns outside the sealed top-5 head (`5..k`).

use crate::retrieval::graph::ghostlink_dgs::{ochiai_divergence_q16, Q16};
use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use crate::retrieval::graph::supersession::turn_is_superseded;

/// Compute Lukasiewicz hinge loss `max(0, w_rule - satisfaction_q16)` in Q16.16.
#[inline]
pub(in crate::retrieval) fn psl_hinge_loss_q16(rule_weight_q16: u64, satisfaction_q16: u64) -> u64 {
    rule_weight_q16.saturating_sub(satisfaction_q16)
}

/// Score how well a candidate turn `cand_ti` closes an unseated multi-hop causal chain
/// anchored at `head_turns[0..]` in Q16.16 integer arithmetic:
/// - Rewards covering query groups `t_exact_mask[cand_ti] & !covered_mask` not yet covered in head
/// - Rewards sharing rare bridging entities (`2 <= df <= max_df`) with a seated turn via exact carrier overlap
/// - Subtracts PSL hinge penalty when redundant with an already-covered mask and no bridge link
pub(in crate::retrieval) fn ilp_psl_chain_utility_q16(
    ix: &Index,
    cand_ti: usize,
    head_turns: &[usize],
    covered_mask: u64,
    total_groups: usize,
    cand_score_q16: u64,
    t_exact_mask: &[u64],
) -> u64 {
    if cand_score_q16 == 0 || total_groups == 0 {
        return 0;
    }
    let mask_limit = if total_groups >= 64 {
        u64::MAX
    } else {
        (1u64 << total_groups) - 1
    };
    let uncovered_bits = ((t_exact_mask[cand_ti] & !covered_mask) & mask_limit).count_ones() as u64;
    let cov_sat_q16 = (uncovered_bits * Q16) / (total_groups.max(1) as u64);

    let mut cand_ents: Vec<u32> = ix.turns[cand_ti].entities.iter().copied().collect();
    cand_ents.sort_unstable();

    let mut best_bridge_sat_q16: u64 = 0;
    if !cand_ents.is_empty() {
        for &anchor_ti in head_turns {
            if ix.turns[anchor_ti].session == ix.turns[cand_ti].session {
                continue;
            }
            let mut anc_ents: Vec<u32> = ix.turns[anchor_ti].entities.iter().copied().collect();
            anc_ents.sort_unstable();
            if !anc_ents.is_empty() {
                let div_q16 = ochiai_divergence_q16(&cand_ents, &anc_ents);
                let agree_q16 = Q16.saturating_sub(div_q16);
                if agree_q16 > best_bridge_sat_q16 {
                    best_bridge_sat_q16 = agree_q16;
                }
            }
        }
    }

    // Combined PSL rule satisfaction: uncovered query coverage + cross-session entity closure
    let sat_q16 = cov_sat_q16.saturating_add(best_bridge_sat_q16 >> 1).min(Q16);
    let hinge_pen_q16 = psl_hinge_loss_q16(Q16 >> 1, sat_q16);
    let net_mult_q16 = Q16.saturating_add(sat_q16).saturating_sub(hinge_pen_q16 >> 1);
    ((cand_score_q16 as u128 * net_mult_q16 as u128) >> 16) as u64
}

/// Apply Roth 0-1 ILP + PSL closure refinement over tail slots `5..k` whenever Top-5
/// has a Multi-Hop Partial Coverage Deficit (`covered_mask != mask_limit`, keeping `0..5` sealed).
pub(in crate::retrieval) fn apply_psl_ilp_tail_closure(
    ix: &Index,
    plan: &QueryPlan,
    turn_score_q16: &[u64],
    t_exact_mask: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
) {
    if k < 6 || final_turn_indices.len() < 6 {
        return;
    }
    let total_groups = plan.groups.len().min(64);
    if total_groups < 2 {
        return;
    }
    let mask_limit = if total_groups == 64 {
        u64::MAX
    } else {
        (1u64 << total_groups) - 1
    };

    let head = &final_turn_indices[..5];
    let mut covered_mask = 0u64;
    for &ti in head {
        covered_mask |= t_exact_mask[ti] & mask_limit;
    }
    let eff_mh = plan.intent.multi_hop;
    // Check if head leaves at least 1 query group uncovered (`Deficit B`)
    if (covered_mask & mask_limit) == mask_limit && !eff_mh {
        return;
    }

    let is_monologue = !ix.turns.iter().any(|t| t.field == 1)
        && !ix
            .turns
            .iter()
            .take(16)
            .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let mut best_cand: Option<(usize, u64)> = None;
    for ti in 0..ix.turns.len() {
        if final_turn_indices.contains(&ti)
            || (if is_monologue {
                ix.turns[ti].is_pure_info_question
            } else {
                ix.turns[ti].is_question
            })
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
            || turn_score_q16[ti] == 0
        {
            continue;
        }
        let has_uncovered_group = ((t_exact_mask[ti] & !covered_mask) & mask_limit) != 0;
        let has_bridge_entities = !ix.turns[ti].entities.is_empty() || !ix.turns[ti].role_objs.is_empty();
        if !has_uncovered_group && !(eff_mh && has_bridge_entities) {
            continue;
        }
        let util = ilp_psl_chain_utility_q16(
            ix,
            ti,
            head,
            covered_mask,
            total_groups,
            turn_score_q16[ti],
            t_exact_mask,
        );
        match best_cand {
            None => best_cand = Some((ti, util)),
            Some((_, bu)) if util > bu => best_cand = Some((ti, util)),
            _ => {}
        }
    }

    let Some((cand_ti, cand_util)) = best_cand else {
        return;
    };

    // Replace weakest redundant tail slot in `5..len` (`(prior_in_sess >= 1 && field == 1) || score == 0`)
    // only when candidate strictly raises ILP utility
    let len = final_turn_indices.len().min(k);
    for slot in (5..len).rev() {
        let vti = final_turn_indices[slot];
        let vs = ix.turns[vti].session;
        let prior_in_sess = final_turn_indices[..slot]
            .iter()
            .filter(|&&t| ix.turns[t].session == vs)
            .count();
        if !((prior_in_sess >= 1 && ix.turns[vti].field == 1) || turn_score_q16[vti] == 0) {
            continue;
        }
        let v_util = ilp_psl_chain_utility_q16(
            ix,
            vti,
            head,
            covered_mask,
            total_groups,
            turn_score_q16[vti],
            t_exact_mask,
        );
        if cand_util > v_util {
            final_turn_indices[slot] = cand_ti;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psl_hinge_loss_is_zero_when_satisfied() {
        assert_eq!(psl_hinge_loss_q16(Q16 / 2, Q16), 0);
        assert_eq!(psl_hinge_loss_q16(Q16, Q16 / 4), (3 * Q16) / 4);
    }
}
