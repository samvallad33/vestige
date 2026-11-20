//! Stage 05: Submodular session set selection (`lin_bilmes`) and `S8-006` cadastral disjoint admission.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let k = ctx.k;
    let ns = ctx.ns;
    let plan = &ctx.plan;
    let s_score = &ctx.s_score;
    let s_mask = &ctx.s_mask;
    let s_policy_floored = &ctx.s_policy_floored;
    let s_pref_floored = &ctx.s_pref_floored;
    let bridge_boost = &ctx.bridge_boost;
    let s_ord_base = &ctx.s_ord_base;
    let smax = ctx.smax;
    let total_q_groups = plan.groups.len().min(64);
    let mask_limit = if total_q_groups == 64 { u64::MAX } else { (1u64 << total_q_groups) - 1 };
    let mut final_session_indices = crate::retrieval::fusion::lin_bilmes::select_submodular_sessions(ctx, total_q_groups, mask_limit);

    // `S8-006` Cadastral disjoint session admission:
    // Admit at most 1 scored session outside Top-10 whose entity-id set is disjoint from Top-10
    // and whose query mask covers a content group none of Top-10 covers.
    if ns > 10 && final_session_indices.len() >= 10 {
        let top10_len = final_session_indices.len().min(10);
        let mut top10_mask = 0u64;
        let mut top10_ents: DetHashSet<u32> = DetHashSet::default();
        for &si in &final_session_indices[..top10_len] {
            top10_mask |= s_mask[si] & mask_limit;
            top10_ents.extend(ix.session_entities[si].iter().copied());
        }
        if let Some(&cad_si) = s_ord_base.iter().find(|&&si| {
            !final_session_indices[..top10_len].contains(&si)
                && s_score[si] > 0.0
                && ((s_mask[si] & mask_limit) & !top10_mask) != 0
                && !ix.session_entities[si].iter().any(|e| top10_ents.contains(e))
        }) {
            let uncov_bits = (s_mask[cad_si] & mask_limit) & !top10_mask;
            let ins_rank = (1..top10_len)
                .rev()
                .find(|&r| {
                    let s = final_session_indices[r];
                    let prev_union = final_session_indices[..r]
                        .iter()
                        .fold(0u64, |acc, &ps| acc | (s_mask[ps] & mask_limit));
                    ((s_mask[s] & mask_limit) & !prev_union) == 0 && (s_mask[s] & uncov_bits) == 0
                })
                .unwrap_or(top10_len - 1);
            if let Some(pos) = final_session_indices.iter().position(|&x| x == cad_si) {
                final_session_indices.remove(pos);
            }
            final_session_indices.insert(ins_rank, cad_si);
            let pool_cap = if crate::retrieval::fusion::lin_bilmes::deep_session_pool_enabled(
                &ctx.eff_it,
                &ctx.q.text,
            ) {
                ns.max(k)
            } else {
                k.max(30).min(ns)
            };
            if final_session_indices.len() > pool_cap {
                final_session_indices.truncate(pool_cap);
            }
        }
    }

    let strong_top_sessions = final_session_indices
        .iter()
        .take(5)
        .filter(|&&si| smax > 0.0 && !s_policy_floored[si] && !s_pref_floored[si] && (s_score[si] + bridge_boost[si]) >= 0.62 * smax)
        .count();

    ctx.total_q_groups = total_q_groups;
    ctx.mask_limit = mask_limit;
    ctx.final_session_indices = final_session_indices;
    ctx.strong_top_sessions = strong_top_sessions;
}
