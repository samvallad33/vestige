//! Stage 07: Exact-edge credit propagation, ReplySource, BoundaryPrimacy, GivenNew, OpenerLift, Hearsay, and Step 11 WH voter.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let k = ctx.k;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let eff_it = &ctx.eff_it;
    let pref_on = ctx.pref_on;
    let is_monologue_corpus = ctx.is_monologue_corpus;
    let max_per_session_top5 = ctx.max_per_session_top5;
    let t_raw = &ctx.t_raw;
    let mut t_scene = std::mem::take(&mut ctx.t_scene);
    let mut final_turn_indices = std::mem::take(&mut ctx.final_turn_indices);

    // Exact-token selection is finished. A recorded backward reply or exact edge
    // promotes its earlier target to the rank position earned by `credit`, including
    // when `target` was previously sitting in the tail (`5..k`) at a lower score.
    if !final_turn_indices.is_empty() {
        let pair = 1.0 / 3.0f32.log2();
        let mut slot_score: Vec<f32> = final_turn_indices.iter().map(|&i| t_scene[i]).collect();
        let mut best_incoming: std::collections::BTreeMap<usize, f32> = std::collections::BTreeMap::new();
        for &ti in &final_turn_indices {
            let src = t_scene[ti];
            if src <= 0.0 {
                continue;
            }
            if let Some(ai) = ix.turns[ti].answers {
                let c = src * pair;
                let cur = best_incoming.entry(ai).or_insert(0.0);
                if c > *cur {
                    *cur = c;
                }
            }
            for &(degree, target) in &ix.turns[ti].exact_edges {
                let damp = 1.0 / (2.0 + degree.saturating_sub(1) as f32).log2();
                let c = src * damp;
                let cur = best_incoming.entry(target).or_insert(0.0);
                if c > *cur {
                    *cur = c;
                }
            }
            if !(if is_monologue_corpus {
                eff_it.current_state
            } else {
                it.current_state && ctx.has_named_speakers_eff
            }) {
                for &target in &ix.turns[ti].supersedes_turns {
                    let c = src * 0.66;
                    let cur = best_incoming.entry(target).or_insert(0.0);
                    if c > *cur {
                        *cur = c;
                    }
                }
            }
        }
        let mut incoming: Vec<(usize, f32)> = best_incoming.into_iter().collect();
        incoming.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (target, credit) in incoming {
            if credit <= 0.0 {
                continue;
            }
            if let Some(existing_pos) = final_turn_indices.iter().position(|&t| t == target) {
                if credit <= slot_score[existing_pos] {
                    continue;
                }
                let mut insert_at = existing_pos;
                for (i, &sc) in slot_score[..existing_pos].iter().enumerate() {
                    if credit > sc {
                        insert_at = i;
                        break;
                    }
                }
                if insert_at >= existing_pos {
                    slot_score[existing_pos] = credit;
                    if credit > t_scene[target] {
                        t_scene[target] = credit;
                    }
                    continue;
                }
                let sess = ix.turns[target].session;
                if insert_at < 5 && existing_pos >= 5 {
                    let in_top5 = final_turn_indices
                        .iter()
                        .take(5)
                        .filter(|&&t| ix.turns[t].session == sess)
                        .count();
                    if in_top5 >= max_per_session_top5 {
                        continue;
                    }
                }
                final_turn_indices.remove(existing_pos);
                slot_score.remove(existing_pos);
                final_turn_indices.insert(insert_at, target);
                slot_score.insert(insert_at, credit);
                if credit > t_scene[target] {
                    t_scene[target] = credit;
                }
                continue;
            }
            // An event list already spread one turn per date. A reply link must
            // not pull a second turn from a session that is already in the list.
            if eff_it.is_event_sequence
                && final_turn_indices.iter().any(|&sel| ix.turns[sel].session == ix.turns[target].session)
            {
                continue;
            }
            let mut insert_at = slot_score.len();
            for (i, &sc) in slot_score.iter().enumerate() {
                if credit > sc {
                    insert_at = i;
                    break;
                }
            }
            if insert_at >= k {
                continue;
            }
            let sess = ix.turns[target].session;
            if insert_at < 5 {
                let in_top5 = final_turn_indices
                    .iter()
                    .take(5)
                    .filter(|&&t| ix.turns[t].session == sess)
                    .count();
                if in_top5 >= max_per_session_top5 {
                    continue;
                }
            }
            if credit > t_scene[target] {
                t_scene[target] = credit;
            }
            final_turn_indices.insert(insert_at, target);
            slot_score.insert(insert_at, credit);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
                slot_score.truncate(k);
            }
        }
    }

    // ReplySource: an assistant reply at rank 1 is derived from the user turn it answers
    // (the turn immediately before it in the same session). When that prompt was retrieved
    // inside the top 5 already selected, the source takes slot 1 and the reply follows; top-5
    // membership never changes. Position and role
    // only. Skipped when the query asks what the assistant itself said.
    if !eff_it.assistant_attribution && final_turn_indices.len() > 1 {
        let top = final_turn_indices[0];
        if ix.turns[top].speaker == "assistant" {
            let members = &ix.session_turns[ix.turns[top].session];
            if let Some(pos) = members.iter().position(|&m| m == top) {
                if pos > 0 {
                    let prev = members[pos - 1];
                    if ix.turns[prev].speaker == "user" && t_raw[prev] > 0.0 {
                        if let Some(at) = final_turn_indices.iter().position(|&t| t == prev) {
                            if at < 5 {
                                final_turn_indices.remove(at);
                                final_turn_indices.insert(0, prev);
                            } else {
                                final_turn_indices.remove(at);
                                final_turn_indices[0] = prev;
                                final_turn_indices.insert(5.min(final_turn_indices.len()), top);
                            }
                        } else {
                            final_turn_indices[0] = prev;
                            if !eff_it.is_event_sequence || final_turn_indices.len() >= 5 {
                                final_turn_indices.insert(5.min(final_turn_indices.len()), top);
                            }
                        }
                    }
                }
            }
        }
    }
    // BoundaryPrimacy: runs after ReplySource, on the final list, so it shares one path with live recall.
    let q_cues = query_cues(ix, &q.text);
    let cue_guard = RecallCueGuard {
        plan: &plan,
        it: &it,
        t_scene: &t_scene,
    };
    boundary_primacy_with_plan(ix, &q_cues, &mut final_turn_indices, Some(&cue_guard));
    // GivenNew runs after BoundaryPrimacy on the same final list (one path for live and bench).
    given_new_with_plan(ix, &q_cues, &mut final_turn_indices, Some(&cue_guard));
    // SessionOpenerLift runs after GivenNew, on the same final list (one path for live and bench).
    session_opener_lift_with_plan(ix, &q_cues, &mut final_turn_indices, Some(&cue_guard));
    // Hearsay runs after SessionOpenerLift, on the same final list (one path for live and bench).
    hearsay_with_plan(ix, &q_cues, &mut final_turn_indices, Some(&cue_guard));
    // Step 11: Kemeny-Young Condorcet QA-Slot Agreement Voter runs before RSB so
    // `rsb_seat_behind` and `rank1_gatekeeper_keep_rsb_head` stay 100% untouched.
    if !eff_it.is_event_sequence && !pref_on {
        apply_step11_kemeny_wh_slot_voter(ix, &q.text, &eff_it, &t_scene, &mut final_turn_indices);
    }
    ctx.t_scene = t_scene;
    ctx.final_turn_indices = final_turn_indices;
}
