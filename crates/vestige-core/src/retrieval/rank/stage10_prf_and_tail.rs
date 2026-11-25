//! Stage 10: QLD/BM25L+DFR long-document promotion and protected bridge tail admission.

use crate::retrieval::*;
use crate::retrieval::context::{bag_to_q8_into, RankCtx};

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let p = ctx.p;
    let k = ctx.k;
    let nt = ctx.nt;
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

    // #823/#825 Unleash BM25L+DFR + QLD on long multi-paragraph monologue documents (> 3 * avgdl)
    // penalized by BM25 length normalization.
    if k >= 6 && is_monologue_corpus && !eff_it.is_event_sequence && !pref_on && !eff_it.task_under_rules {
        let mut qld_terms_buf: Vec<crate::retrieval::lexical::qld::QldTermInput> = Vec::with_capacity(plan.terms.len());
        let avg_own_dl = (ix.t_own_stats.avg[0] + ix.t_own_stats.avg[1]).max(16.0);
        let eval_bm25l_turn: Vec<bool> = if nt <= 128 {
            vec![true; nt]
        } else {
            let mut flags = vec![false; nt];
            for &ti in &final_turn_indices {
                if ti < nt {
                    flags[ti] = true;
                }
            }
            let mut ranked_own: Vec<(f32, usize)> = (0..nt)
                .filter_map(|ti| (t_own[ti] > 0.0).then_some((t_own[ti], ti)))
                .collect();
            ranked_own.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            for &(_, ti) in ranked_own.iter().take(128) {
                flags[ti] = true;
            }
            for ti in 0..nt {
                if !flags[ti] && t_own[ti] > 0.0 && t_exact_mask[ti].count_ones() >= 2 {
                    let dl_ti = ix.turn_own_bags[ti].len[0] + ix.turn_own_bags[ti].len[1];
                    if dl_ti > 3.0 * avg_own_dl {
                        flags[ti] = true;
                    }
                }
            }
            flags
        };
        let mut best_long_bm25l: Option<(usize, u64)> = None;
        let mut bm25l_raw_vec: Vec<u64> = vec![0; nt];
        for ti in 0..nt {
            if t_own[ti] <= 0.0 || !eval_bm25l_turn[ti] {
                continue;
            }
            let q_bag = bag_to_q8_into(&ix.turn_own_bags[ti], plan, &ix.t_own_stats, &mut qld_terms_buf);
            let bm25l_q16 = crate::retrieval::lexical::bm25l_dfr::bm25l_dfr_score_q16(&q_bag, &qld_terms_buf);
            let qld_q16 = crate::retrieval::lexical::qld::qld_score_q16(&q_bag, &qld_terms_buf, crate::retrieval::lexical::qld::MU_TURN_Q8);
            let combined_l = bm25l_q16.saturating_add(qld_q16);
            bm25l_raw_vec[ti] = combined_l;
            let dl_ti = ix.turn_own_bags[ti].len[0] + ix.turn_own_bags[ti].len[1];
            if dl_ti > 3.0 * avg_own_dl
                && !turn_is_superseded(ix, ti)
                && !ix.turns[ti].is_pure_info_question
                && t_exact_mask[ti].count_ones() >= 2
                && !final_turn_indices[..5.min(final_turn_indices.len())].contains(&ti)
                && best_long_bm25l.map_or(true, |(_, bs)| combined_l > bs)
            {
                best_long_bm25l = Some((ti, combined_l));
            }
        }
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
