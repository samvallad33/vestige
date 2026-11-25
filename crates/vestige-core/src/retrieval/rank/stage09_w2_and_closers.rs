//! Stage 09: W2 milestone/trajectory/standing-register, live-task standing rules, slot promotions, vault closers, and multi-hop diversity.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let k = ctx.k;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let eff_it = &ctx.eff_it;
    let seq_mode_eff = ctx.seq_mode_eff;
    let has_named_speakers_eff = ctx.has_named_speakers_eff;
    let is_monologue_corpus = ctx.is_monologue_corpus;
    let cond_rule_turns = &ctx.cond_rule_turns;
    let standing_rules = &ctx.standing_rules;
    let t_own = &ctx.t_own;
    let t_scene = &ctx.t_scene;
    let t_mask = &ctx.t_mask;
    let t_exact_mask = &ctx.t_exact_mask;
    let final_session_indices = &ctx.final_session_indices;
    let mut final_turn_indices = std::mem::take(&mut ctx.final_turn_indices);

    // [W2-033] Milestone Admission: for long turn histories (`turns >= 64`) or monologues,
    // use derived `eff_it` so multi-batch conversations (`ns = 3..63`) still activate trajectory seating.
    let seq_intent = if ix.turns.len() >= 64 || is_monologue_corpus {
        &eff_it
    } else {
        &it
    };
    crate::retrieval::seating::apply_w2_milestone_admission(ix, seq_intent, &mut final_turn_indices, &t_scene);

    // [W2-028 / W2-031] StandingRegister constraint posting & assistant-echo swap
    crate::retrieval::seating::apply_w2_standing_register_seat(ix, &q.text, &plan, &mut final_turn_indices, &t_exact_mask);

    // Scope-matched / live-task standing_rules head merge (keyed on turn count `turns >= 64`,
    // never `session_turns.len()`, so coarse 3..5-batch long chats qualify too):
    let allow_chat_task_rule = !has_named_speakers_eff && ix.turns.len() >= 64 && eff_it.task_under_rules;
    if (is_monologue_corpus || allow_chat_task_rule)
        && !standing_rules.is_empty()
        && !seq_mode_eff
        && !it.temporal
        && !it.sniper_when
        && !it.sniper_how_many
        && final_turn_indices.len() >= 3
    {
        // Structural inverse: when `cond_ok` holds, default to `(r_ti, pti)` and only flip
        // to `(pti, r_ti)` when the query is an explicit preference or recommendation request.
        let prefer_pref_over_rule =
            eff_it.preference || crate::retrieval::step14_is_rec_request(&q.text);
        for sr in standing_rules.iter().take(2) {
            if sr.score <= 0.0 {
                continue;
            }
            let Some(r_ti) = ix.turns.iter().position(|t| t.id == sr.id) else {
                continue;
            };
            let head_already_rule = ix.turns[final_turn_indices[0]].is_standing_policy;
            if turn_is_superseded(ix, r_ti)
                || (final_turn_indices[0] == r_ti)
                || (!allow_chat_task_rule && final_turn_indices[..2].contains(&r_ti))
            {
                continue;
            }
            let has_scope_overlap = t_mask[r_ti] != 0
                || crate::retrieval::seating::c2_policy_inflects_query(ix, r_ti, &plan);
            // Rule-condition gate: a rule whose own condition does not match the
            // query and that shares no query term stays on the standing_rules
            // channel only; it never takes an episodic slot.
            let cond_ok = cond_rule_turns.contains(&r_ti);
            if !cond_ok {
                continue;
            }
            // Prevent standing-rule hijacks on topical user preferences: if an active non-superseded
            // user preference turn shares direct query-topic overlap, seat the preference at Slot 0
            // and place the standing rule at Slot 1 instead of displacing the preference out of Top-5.
            let topical_pref_ti = ix
                .preference_directive_turns
                .iter()
                .copied()
                .filter(|&pti| {
                    let pt = &ix.turns[pti];
                    pt.field == 0
                        && !pt.is_standing_policy
                        && !pt.is_backchannel
                        && !turn_is_superseded(ix, pti)
                        && t_exact_mask.get(pti).copied().unwrap_or(0) != 0
                        && crate::retrieval::seating::preference_lattice::is_explicit_stance_directive(
                            &pt.body_for_headers.to_ascii_lowercase(),
                            pt.is_personal_constraint,
                        )
                })
                .max_by(|&a, &b| {
                    t_exact_mask[a]
                        .count_ones()
                        .cmp(&t_exact_mask[b].count_ones())
                        .then_with(|| {
                            t_scene[a]
                                .partial_cmp(&t_scene[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| b.cmp(&a))
                });
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == r_ti) {
                final_turn_indices.remove(pos);
            }
            let wants_slot0 = (allow_chat_task_rule && !head_already_rule && cond_ok)
                || (has_scope_overlap
                    && t_exact_mask[r_ti].count_ones()
                        >= t_exact_mask[final_turn_indices[0]].count_ones()
                    && !q.text.trim_end().ends_with('?'));
            if wants_slot0 && let Some(pti) = topical_pref_ti {
                let (slot0_ti, slot1_ti) = if prefer_pref_over_rule {
                    (pti, r_ti)
                } else {
                    (r_ti, pti)
                };
                if let Some(ppos) = final_turn_indices.iter().position(|&x| x == pti) {
                    final_turn_indices.remove(ppos);
                }
                final_turn_indices.insert(0, slot0_ti);
                final_turn_indices.insert(1.min(final_turn_indices.len()), slot1_ti);
            } else {
                let ins_at = if wants_slot0 {
                    0
                } else {
                    1.min(final_turn_indices.len())
                };
                final_turn_indices.insert(ins_at, r_ti);
            }
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
            }
            break;
        }
    }

    // Slot-1 Assertion & Exact-Superset Promotion (handles both user declaratives & assistant attribution)
    crate::retrieval::seating::apply_assertion_and_superset_slot1_promotion(
        ix,
        &plan,
        &eff_it,
        &mut final_turn_indices,
        &t_own,
        &t_scene,
        &t_exact_mask,
    );

    // Named-Dialogue Speaker-Subject Alignment at Slot 0
    crate::retrieval::seating::apply_speaker_subject_slot0_promotion(
        ix,
        &plan,
        &q.text,
        &mut final_turn_indices,
        &t_scene,
        &t_own,
        &t_exact_mask,
    );

    // Wave 7 S1-071 Informative Turn Priority: short confirmations yield to substantive declaratives
    crate::retrieval::seating::preference_lattice::apply_informative_turn_priority(
        ix,
        &plan,
        &mut final_turn_indices,
        &t_scene,
        &t_own,
        &t_exact_mask,
    );

    // Multi-Hop Session Diversity on Named-Speaker Dialogues:
    // Cap any single session to at most 2 turns inside `final_turn_indices[0..5]` (leaving Slots 0..2 untouched),
    // swapping any 3rd+ duplicate-session turn in Slots 2..5 with the highest-scoring declarative turn
    // from the next uncovered top session (`final_session_indices[..4]`).
    if eff_it.multi_hop && has_named_speakers_eff && final_turn_indices.len() >= 5 {
        for slot in (2..5usize).rev() {
            let s_cur = ix.turns[final_turn_indices[slot]].session;
            let s_cnt = final_turn_indices[..5]
                .iter()
                .filter(|&&ti| ix.turns[ti].session == s_cur)
                .count();
            if s_cnt < 3 {
                continue;
            }
            let mut replacement: Option<usize> = None;
            for &cand_s in final_session_indices.iter().take(4) {
                if final_turn_indices[..5]
                    .iter()
                    .any(|&ti| ix.turns[ti].session == cand_s)
                {
                    continue;
                }
                let mut best_in_s: Option<(usize, f32)> = None;
                for &cti in &ix.session_turns[cand_s] {
                    let ct = &ix.turns[cti];
                    if ct.is_question || ct.is_backchannel || turn_is_superseded(ix, cti) {
                        continue;
                    }
                    let sc = t_scene[cti];
                    if sc > 0.0 && best_in_s.map_or(true, |(_, bs)| sc > bs) {
                        best_in_s = Some((cti, sc));
                    }
                }
                if let Some((cti, _)) = best_in_s {
                    replacement = Some(cti);
                    break;
                }
            }
            if let Some(cti) = replacement {
                if let Some(existing) = final_turn_indices.iter().position(|&x| x == cti) {
                    final_turn_indices.swap(slot, existing);
                } else {
                    final_turn_indices[slot] = cti;
                }
            }
        }
    }

    ctx.final_turn_indices = final_turn_indices;
}
