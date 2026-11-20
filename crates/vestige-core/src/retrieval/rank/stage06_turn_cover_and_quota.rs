//! Stage 06: Weapon 4 submodular turn cover (`lin_bilmes`), Q4 quota, Jefferson/Rhodia, Step 10a, and Step 15 dual-slot.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let k = ctx.k;
    let ns = ctx.ns;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let eff_it = &ctx.eff_it;
    let pref_on = ctx.pref_on;
    let strong_top_sessions = ctx.strong_top_sessions;
    let smax = ctx.smax;
    let s_score = &ctx.s_score;
    let bridge_boost = &ctx.bridge_boost;
    let t_own = &ctx.t_own;
    let t_exact_mask = &ctx.t_exact_mask;
    let t_policy_floored = &ctx.t_policy_floored;
    let t_pref_floored = &ctx.t_pref_floored;
    let final_session_indices = &ctx.final_session_indices;

    let has_named_speakers_top = ix
        .turns
        .iter()
        .take(64)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let s0_dominant = !eff_it.multi_hop
        && final_session_indices.len() >= 2
        && (s_score[final_session_indices[0]] + bridge_boost[final_session_indices[0]])
            >= 1.35 * (s_score[final_session_indices[1]] + bridge_boost[final_session_indices[1]]);
    // Weapon 4: Greedy Submodular Set Cover for Top-K Turn Selection
    let max_per_session_top5 = if eff_it.is_event_sequence {
        1
    } else if s0_dominant {
        3
    } else if eff_it.multi_hop
        || (!pref_on && !eff_it.task_under_rules && (strong_top_sessions >= 2 || (has_named_speakers_top && ns >= 5)))
    {
        2
    } else {
        3
    };
    let (q4_sess_floor, q4_turn_floor, allow_short_reply_bypass) =
        step10_florilegium_floors(strong_top_sessions);

    let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
    let mut final_turn_indices = crate::retrieval::fusion::lin_bilmes::select_submodular_turns(
        ctx,
        &turn_ts,
        has_named_speakers_top,
        max_per_session_top5,
        allow_short_reply_bypass,
    );
    let mut t_scene = std::mem::take(&mut ctx.t_scene);

    // Q4: Multi-session turn quota guarantee inside top-5 so Session #1 redundant
    // turns cannot crowd out strong Sessions #2..#5 before exact-edge/RSB stages.
    if !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && final_turn_indices.len() >= 5
        && smax > 0.0
    {
        let eff_q4_sess = if eff_it.multi_hop { q4_sess_floor.min(0.42) } else { q4_sess_floor };
        let eff_q4_turn = if eff_it.multi_hop { q4_turn_floor.min(0.38) } else { q4_turn_floor };
        let sess_limit = final_session_indices.len().min(5);
        for &si in &final_session_indices[1..sess_limit] {
            if s_score[si] + bridge_boost[si] < eff_q4_sess * smax {
                continue;
            }
            if final_turn_indices[..5].iter().any(|&t| ix.turns[t].session == si) {
                continue;
            }
            let Some(best_ti) = ix.session_turns[si]
                .iter()
                .copied()
                .filter(|&t| t_scene[t] > 0.0)
                .max_by(|&a, &b| {
                    t_scene[a]
                        .partial_cmp(&t_scene[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.cmp(&a))
                })
            else {
                continue;
            };
            let victim_slot = (1..5).rev().find(|&slot| {
                let vti = final_turn_indices[slot];
                if t_policy_floored[vti] || t_pref_floored[vti] {
                    return false;
                }
                let vs = ix.turns[vti].session;
                let cnt = final_turn_indices[..5]
                    .iter()
                    .filter(|&&t| ix.turns[t].session == vs)
                    .count();
                let min_cnt = if s0_dominant
                    && Some(vs) == final_session_indices.first().copied()
                    && (s_score[si] + bridge_boost[si]) < 0.62 * smax
                {
                    3
                } else {
                    2
                };
                cnt >= min_cnt && t_scene[best_ti] >= eff_q4_turn * t_scene[vti]
            });
            if let Some(slot) = victim_slot {
                if let Some(existing) = final_turn_indices.iter().position(|&t| t == best_ti) {
                    if existing >= 5 {
                        final_turn_indices.remove(existing);
                        final_turn_indices.insert(slot, best_ti);
                    }
                } else {
                    final_turn_indices.insert(slot, best_ti);
                    final_turn_indices.truncate(k);
                }
            }
        }
    }

    // `jefferson_apportionment` + `lex_rhodia_shedding` (`BEAM MSR`, `LME_M MS`, `MAB MS`):
    // On multi-session / count queries with `strong_top_sessions >= 2`, allocate Top-5 seats
    // by D'Hondt greatest divisors `s_sight / (seats + 1)` and evict only from the session
    // holding the most seats (protecting Slot 0).
    if (eff_it.multi_hop || eff_it.sniper_how_many)
        && !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && strong_top_sessions >= 2
        && final_turn_indices.len() >= 5
        && smax > 0.0
    {
        for &cand_si in final_session_indices.iter().take(5) {
            let cand_sight = s_score[cand_si] + bridge_boost[cand_si];
            if cand_sight < 0.55 * smax {
                continue;
            }
            if final_turn_indices[..5].iter().any(|&ti| ix.turns[ti].session == cand_si) {
                continue;
            }
            let Some(best_cand_ti) = ix.session_turns[cand_si]
                .iter()
                .copied()
                .filter(|&ti| {
                    !ix.turns[ti].is_pure_info_question
                        && !ix.turns[ti].is_backchannel
                        && !turn_is_superseded(ix, ti)
                        && t_scene[ti] > 0.0
                })
                .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
            else {
                continue;
            };
            // `lex_rhodia_shedding`: find the lowest-ranked turn in `1..5` belonging to the session
            // with the largest seat count (`>= 2`), when D'Hondt quotient `cand_sight / 1 > over_sight / over_cnt`.
            let mut shed_target: Option<(usize, usize, f32)> = None;
            for slot in (1..5usize).rev() {
                let vti = final_turn_indices[slot];
                if t_policy_floored[vti] || t_pref_floored[vti] {
                    continue;
                }
                let vs = ix.turns[vti].session;
                let s_cnt = final_turn_indices[..5].iter().filter(|&&t| ix.turns[t].session == vs).count();
                if s_cnt < 2 {
                    continue;
                }
                let over_quotient = (s_score[vs] + bridge_boost[vs]) / (s_cnt as f32);
                if cand_sight >= over_quotient {
                    let vsc = t_scene[vti];
                    if shed_target.is_none_or(|(_, bcnt, bsc)| s_cnt > bcnt || (s_cnt == bcnt && vsc < bsc)) {
                        shed_target = Some((slot, s_cnt, vsc));
                    }
                }
            }
            if let Some((slot, _, _)) = shed_target {
                if let Some(pos) = final_turn_indices.iter().position(|&t| t == best_cand_ti) {
                    if pos >= 5 {
                        final_turn_indices.remove(pos);
                        final_turn_indices.insert(slot, best_cand_ti);
                    }
                } else {
                    final_turn_indices.insert(slot, best_cand_ti);
                    final_turn_indices.truncate(k);
                }
            }
        }
    }

    if !eff_it.is_event_sequence && !pref_on && !eff_it.task_under_rules {
        let gl = {
            let g = plan.groups.len().min(64);
            if g == 64 { u64::MAX } else { (1u64 << g) - 1 }
        };
        let floored = |t: usize| t_policy_floored[t] || t_pref_floored[t];
        apply_step10a_evidence_seat(
            ix,
            &final_session_indices,
            &t_scene,
            &t_exact_mask,
            gl,
            &floored,
            k,
            &mut final_turn_indices,
        );
        apply_step15_comparative_dual_slot_seat(
            ix,
            &q.text,
            &plan,
            &it,
            &t_own,
            &t_exact_mask,
            &floored,
            k,
            &mut t_scene,
            &mut final_turn_indices,
        );
    }

    // Phase 4 Step 10C (T63 Question->Answer Adjacency Pairing on named-speaker dialogues):
    // When a question turn occupies Top-4 (`0..4`) and its immediate declarative reply by the partner
    // in the same session is missing from Top-5, seat that reply inside Top-5 (never displacing Slot 0).
    if has_named_speakers_top
        && !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && final_turn_indices.len() >= 5
    {
        let mut pairs_to_seat: Vec<(usize, usize)> = Vec::new();
        for slot in 0..4usize {
            let q_ti = final_turn_indices[slot];
            if !ix.turns[q_ti].is_question {
                continue;
            }
            let sess = ix.turns[q_ti].session;
            let members = &ix.session_turns[sess];
            if let Some(pos) = members.iter().position(|&m| m == q_ti) {
                if let Some(&ans_ti) = members.get(pos + 1) {
                    let ans = &ix.turns[ans_ti];
                    if !ans.is_question
                        && !ans.is_backchannel
                        && ans.speaker != ix.turns[q_ti].speaker
                        && !final_turn_indices[..5].contains(&ans_ti)
                    {
                        pairs_to_seat.push((slot, ans_ti));
                    }
                }
            }
        }
        for (q_slot, ans_ti) in pairs_to_seat.into_iter().take(2) {
            if final_turn_indices[..5].contains(&ans_ti) {
                continue;
            }
            if let Some(existing) = final_turn_indices.iter().position(|&t| t == ans_ti) {
                final_turn_indices.remove(existing);
            }
            let ins_slot = if q_slot == 0 { 1 } else { q_slot };
            final_turn_indices.insert(ins_slot, ans_ti);
            final_turn_indices.truncate(k.max(5));
        }
    }

    ctx.t_scene = t_scene;
    ctx.max_per_session_top5 = max_per_session_top5;
    ctx.turn_ts = turn_ts;
    ctx.final_turn_indices = final_turn_indices;
}
