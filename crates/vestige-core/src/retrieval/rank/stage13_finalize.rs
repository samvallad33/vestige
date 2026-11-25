//! Stage 13: Optional replay dump, premise-coverage computation, and final `RankResult` construction.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: RankCtx<'_>) -> RankResult {
    let ix = ctx.ix;
    let q = ctx.q;
    let k = ctx.k;
    let plan = ctx.plan;
    let it = plan.intent.clone();
    let s_score = ctx.s_score;
    let s_mask = ctx.s_mask;
    let bridge_boost = ctx.bridge_boost;
    let t_own = ctx.t_own;
    let t_turn_raw = ctx.t_turn_raw;
    let t_scene = ctx.t_scene;
    let t_scene_sess = ctx.t_scene_sess;
    let t_bridge = ctx.t_bridge;
    let t_mask = ctx.t_mask;
    let t_exact_mask = ctx.t_exact_mask;
    let standing_rules = ctx.standing_rules;
    let final_session_indices = ctx.final_session_indices;
    let final_turn_indices = ctx.final_turn_indices;
    let q_group_cnt = ctx.q_group_cnt;
    let q_group_mask = ctx.q_group_mask;
    let grounded_group_mask = ctx.grounded_group_mask;

    maybe_dump_replay(
        ix,
        q,
        k,
        &it,
        &t_scene,
        &final_turn_indices,
        &final_session_indices,
        &t_own,
        &t_turn_raw,
        &t_scene_sess,
        &s_score,
        &bridge_boost,
    );
    let mut max_ep_groups = 0usize;
    let mut union_mask_top5 = 0u64;
    let mut exact_union_top5 = 0u64;
    let mut grounded_union = 0u64;
    let mut any_pos_support = !standing_rules.is_empty();
    for &ti in final_turn_indices.iter().take(5) {
        let m = t_mask[ti] & q_group_mask;
        let c = m.count_ones() as usize;
        if c > max_ep_groups {
            max_ep_groups = c;
        }
        union_mask_top5 |= m;
        exact_union_top5 |= t_exact_mask[ti] & q_group_mask;
        grounded_union |= t_exact_mask[ti] & grounded_group_mask;
        if t_scene[ti] > 0.0 || t_bridge[ti] > 0.0 {
            any_pos_support = true;
        }
    }
    let union_groups = union_mask_top5.count_ones() as usize;
    let grounded_groups = grounded_group_mask.count_ones() as usize;
    let grounded_covered = grounded_union.count_ones() as usize;
    let zero_exact_unbridged = exact_union_top5 == 0
        && standing_rules.is_empty()
        && final_turn_indices
            .first()
            .map_or(true, |&h0| t_bridge[h0] <= 0.0)
        && union_groups < q_group_cnt;
    let cov_status = if q_group_cnt > 0
        && ((union_groups == 0 && (!any_pos_support || (q_group_cnt >= 2 && standing_rules.is_empty())))
            || zero_exact_unbridged)
    {
        "ABSTAIN"
    } else if q_group_cnt >= 2 && union_groups > 0 && union_groups < q_group_cnt {
        "PARTIAL_PREMISE"
    } else {
        "COVERED"
    };
    let gc = ctx.ghostcurrent;
    let open_circuit = gc.open_circuit || cov_status == "ABSTAIN";
    let frontier_turn_id = if open_circuit {
        gc.frontier_turn_idx
            .and_then(|ti| ix.turns.get(ti).map(|t| t.id.clone()))
            .or_else(|| {
                final_turn_indices
                    .first()
                    .and_then(|&ti| ix.turns.get(ti).map(|t| t.id.clone()))
            })
    } else {
        None
    };
    let missing_query_groups = if open_circuit {
        if !gc.missing_query_groups.is_empty() {
            gc.missing_query_groups.clone()
        } else {
            plan.groups
                .iter()
                .enumerate()
                .take(64)
                .filter(|&(gi, gw)| {
                    (q_group_mask & (1u64 << gi)) != 0
                        && (union_mask_top5 & (1u64 << gi)) == 0
                        && !gw.starts_with('\u{1}')
                })
                .map(|(_, gw)| gw.clone())
                .collect()
        }
    } else {
        Vec::new()
    };
    let unresolved_slot_type = if open_circuit {
        gc.unresolved_slot_type
            .or_else(|| crate::retrieval::graph::ghostcurrent_mdl_q16::wh_slot_type_label(&q.text))
    } else {
        None
    };
    let premise_coverage = PremiseCoverage {
        status: cov_status,
        total_groups: q_group_cnt,
        max_episode_groups: max_ep_groups,
        union_groups,
        grounded_groups,
        grounded_covered,
        open_circuit,
        frontier_turn_id,
        missing_query_groups,
        unresolved_slot_type,
    };

    let wf_g1_s10 = ctx.wf_g1_s10;
    let wf_g3_t10 = ctx.wf_g3_t10;
    let wf_g4_s5 = ctx.wf_g4_s5;
    let wf_g4_flags = ctx.wf_g4_flags;

    let sessions: Vec<RankedItem> = final_session_indices
        .iter()
        .copied()
        .map(|i| RankedItem {
            id: ix.session_ids[i].clone(),
            session_id: ix.session_ids[i].clone(),
            score: s_score[i] + bridge_boost[i],
            matched: matched_words(&plan, s_mask[i]),
        })
        .collect();

    let turns: Vec<RankedItem> = final_turn_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.turns[i].id.clone(),
            session_id: ix.session_ids[ix.turns[i].session].clone(),
            score: if t_scene[i] > 0.0 { t_scene[i] } else { t_bridge[i] },
            matched: matched_words(&plan, t_mask[i]),
        })
        .collect();

    if let Some(wf_path) = crate::retrieval::rank::waterfall_log_path()
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(wf_path)
    {
        use std::io::Write;
        let g5_s5: Vec<&str> = sessions.iter().take(5).map(|s| s.id.as_str()).collect();
        let g5_t5: Vec<&str> = turns.iter().take(5).map(|t| t.id.as_str()).collect();
        let rec = serde_json::json!({
            "q": q.text,
            "g1_s10": wf_g1_s10,
            "g3_t10": wf_g3_t10,
            "g4_s5": wf_g4_s5,
            "g4_flags": wf_g4_flags,
            "g5_s5": g5_s5,
            "g5_t5": g5_t5,
            "gc_best_absorbed_q16": gc.best_absorbed_q16,
            "gc_best_mdl_energy_q16": gc.best_mdl_energy_q16,
            "gc_open_circuit": open_circuit,
        });
        let _ = writeln!(f, "{rec}");
    }

    RankResult {
        intent: plan.intent,
        premise_coverage,
        sessions,
        turns,
        standing_rules,
    }
}
