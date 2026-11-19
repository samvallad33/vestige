//! Stage 10: Moffat RBP, Rocchio PRF, QLD/BM25L+DFR, PSL/ILP tail closure, and protected bridge tail.

use crate::retrieval::*;
use crate::retrieval::context::{bag_to_q8_into, RankCtx};

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let p = ctx.p;
    let k = ctx.k;
    let nt = ctx.nt;
    let ns = ctx.ns;
    let plan = &ctx.plan;
    let eff_it = &ctx.eff_it;
    let pref_on = ctx.pref_on;
    let is_monologue_corpus = ctx.is_monologue_corpus;
    let t_own = &ctx.t_own;
    let t_scene = &ctx.t_scene;
    let t_exact_mask = &ctx.t_exact_mask;
    let t_bridge = &ctx.t_bridge;
    let t_bridge_hits = &ctx.t_bridge_hits;
    let mut final_turn_indices = std::mem::take(&mut ctx.final_turn_indices);

    // #829 Moffat Rank-Biased Precision (p = 13/16 in Q16.16) tail slot reallocation (`5..k`, keeping `0..5` sealed),
    // powered by #821 QLD, #823/#825 BM25L+DFR, and #828 Rocchio PRF turn & session signals.
    if k >= 6 && !eff_it.is_event_sequence && !pref_on && !eff_it.task_under_rules {
        let prf_seeds: Vec<usize> = final_turn_indices
            .iter()
            .take(4)
            .copied()
            .filter(|&ti| !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel)
            .collect();
        let prf_negs: Vec<usize> = final_turn_indices.iter().skip(5).copied().collect();
        let prf_centroid = crate::retrieval::lexical::rocchio::build_rocchio_centroid_q16(ix, &plan, &prf_seeds, &prf_negs);
        let mut qld_terms_buf: Vec<crate::retrieval::lexical::qld::QldTermInput> = Vec::with_capacity(plan.terms.len());
        let s_rocchio_q16: Vec<u64> = (0..ns)
            .map(|si| crate::retrieval::lexical::rocchio::score_session_rocchio_q16(ix, si, &prf_centroid))
            .collect();
        let t_scene_q16: Vec<u64> = (0..nt)
            .map(|ti| {
                let v = t_scene[ti];
                if v <= 0.0 {
                    return 0;
                }
                let base_q16 = (v * 65536.0).min(u64::MAX as f32) as u64;
                let s_r = s_rocchio_q16[ix.turns[ti].session] >> 5;
                base_q16.saturating_add(s_r)
            })
            .collect();
        let avg_own_dl = (ix.t_own_stats.avg[0] + ix.t_own_stats.avg[1]).max(16.0);
        let mut best_long_bm25l: Option<(usize, u64)> = None;
        let mut bm25l_raw_vec: Vec<u64> = vec![0; nt];
        let t_own_q16: Vec<u64> = (0..nt)
            .map(|ti| {
                let v = t_own[ti];
                if v <= 0.0 {
                    return 0;
                }
                let base_q16 = (v * 65536.0).min(u64::MAX as f32) as u64;
                let q_bag = bag_to_q8_into(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats, &mut qld_terms_buf);
                let bm25l_q16 = crate::retrieval::lexical::bm25l_dfr::bm25l_dfr_score_q16(&q_bag, &qld_terms_buf);
                let qld_q16 = crate::retrieval::lexical::qld::qld_score_q16(&q_bag, &qld_terms_buf, crate::retrieval::lexical::qld::MU_TURN_Q8);
                let combined_l = bm25l_q16.saturating_add(qld_q16);
                bm25l_raw_vec[ti] = combined_l;
                let dl_ti = ix.turn_own_bags[ti].len[0] + ix.turn_own_bags[ti].len[1];
                if is_monologue_corpus
                    && dl_ti > 3.0 * avg_own_dl
                    && !turn_is_superseded(ix, ti)
                    && !ix.turns[ti].is_pure_info_question
                    && t_exact_mask[ti].count_ones() >= 2
                    && !final_turn_indices[..5.min(final_turn_indices.len())].contains(&ti)
                    && best_long_bm25l.map_or(true, |(_, bs)| combined_l > bs)
                {
                    best_long_bm25l = Some((ti, combined_l));
                }
                base_q16.saturating_add(combined_l >> 6)
            })
            .collect();
        // #823/#825 Unleash BM25L+DFR on long multi-paragraph documents (> 3 * avgdl) penalized by BM25 length norm:
        if let Some((long_ti, long_sc)) = best_long_bm25l {
            if final_turn_indices.len() >= 5 && long_sc > 0 {
                let h0 = final_turn_indices[0];
                let h0_sc = bm25l_raw_vec[h0];
                let inc4 = final_turn_indices[4];
                let inc4_sc = bm25l_raw_vec[inc4];
                let long_bits = t_exact_mask[long_ti].count_ones();
                let h0_bits = t_exact_mask[h0].count_ones();
                let target_at = if long_sc.saturating_mul(2) >= h0_sc.saturating_mul(3)
                    && long_bits > h0_bits
                {
                    Some(0)
                } else if long_sc.saturating_mul(2) >= h0_sc.saturating_mul(3)
                    && long_bits == h0_bits
                {
                    Some(1)
                } else if long_sc.saturating_mul(4) >= inc4_sc.saturating_mul(5)
                    && long_bits >= t_exact_mask[inc4].count_ones()
                {
                    Some(4)
                } else {
                    None
                };
                if let Some(at) = target_at {
                    if let Some(pos) = final_turn_indices.iter().position(|&x| x == long_ti) {
                        final_turn_indices.remove(pos);
                    }
                    final_turn_indices.insert(at.min(final_turn_indices.len()), long_ti);
                    if final_turn_indices.len() > k {
                        final_turn_indices.truncate(k);
                    }
                }
            }
        }
        crate::retrieval::lexical::rbp::apply_rbp_tail_allocation(
            ix,
            &prf_centroid,
            &t_scene_q16,
            &t_own_q16,
            &mut final_turn_indices,
            k,
        );
        // #804 Getoor PSL Hinge + #803 Roth 0-1 ILP multi-hop tail closure (`5..k`, keeping `0..5` sealed)
        crate::retrieval::fusion::psl_ilp::apply_psl_ilp_tail_closure(
            ix,
            &plan,
            &t_scene_q16,
            &t_exact_mask,
            &mut final_turn_indices,
            k,
        );
    }

    // Post-Gatekeeper Sealed-Head Bridge Tail Admission:
    // Slots `0..5` (`@1..@5`) stay 100% SEALED whenever exact matches fill them.
    // Slot `k - 1` admits the top zero-exact-overlap `t_bridge` turn only when slot `k - 1`
    // is a duplicate-session assistant echo or unfilled tail.
    admit_protected_tail(ix, p, k, nt, &t_bridge, &t_bridge_hits, &t_exact_mask, &t_scene, &mut final_turn_indices);

    let base_q16: Vec<u64> = t_scene
        .iter()
        .map(|&v| if v > 0.0 { (v * 65536.0).min(u64::MAX as f32) as u64 } else { 0 })
        .collect();

    ctx.final_turn_indices = final_turn_indices;
    ctx.base_q16 = base_q16;
}
