//! Greedy submodular session and turn set selection.

use crate::retrieval::*;
use crate::retrieval::rank::context::RankCtx;

/// Returns `true` when the query requires a deep scored session pool prior to
/// the final `top_k` cut (`event_ordering` and `summarization` only).
/// All other query types stay strictly on the existing `k` cut (byte-identical).
#[inline]
pub(in crate::retrieval) fn deep_session_pool_enabled(
    eff_it: &QueryIntent,
    query: &str,
) -> bool {
    eff_it.is_event_sequence || eff_summary_request(query)
}

pub(in crate::retrieval) fn select_submodular_sessions(
    ctx: &RankCtx<'_>,
    total_q_groups: usize,
    mask_limit: u64,
) -> Vec<usize> {
    let ix = ctx.ix;
    let p = ctx.p;
    let ns = ctx.ns;
    let k = ctx.k;
    let it = &ctx.plan.intent;
    let eff_it = &ctx.eff_it;
    let s_score = &ctx.s_score;
    let s_mask = &ctx.s_mask;
    let s_policy_floored = &ctx.s_policy_floored;
    let s_pref_floored = &ctx.s_pref_floored;
    let bridge_boost = &ctx.bridge_boost;
    let s_ord_base = &ctx.s_ord_base;
    let smax = ctx.smax;
    let initiative_content = &ctx.initiative_content;
    let pool_cap = if deep_session_pool_enabled(eff_it, &ctx.q.text) {
        ns.max(k)
    } else {
        k
    };

if p.enable_submodular && ns > 1 {
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        if let Some(&top0) = s_ord_base.first() {
            if s_score[top0] > 0.0 || bridge_boost[top0] > 0.0 {
                selected.push(top0);
                let mut covered_mask = s_mask[top0];
                let mut covered_ents: DetHashSet<u32> = ix.session_entities[top0].clone();
                let mut mask_counts: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
                *mask_counts.entry(s_mask[top0] & mask_limit).or_insert(0) += 1;
                let mut remaining: Vec<usize> = (0..ns)
                    .filter(|&si| si != top0 && (s_score[si] > 0.0 || bridge_boost[si] > 0.0))
                    .collect();

                while selected.len() < k && !remaining.is_empty() {
                    let mut best_pos = 0usize;
                    let mut best_gain = -1.0f32;
                    let mut found = false;
                    for (idx, &si) in remaining.iter().enumerate() {
                        let m_eff = s_mask[si] & mask_limit;
                        let uncovered = ((s_mask[si] & !covered_mask) & mask_limit).count_ones() as f32;
                        let uncovered_ratio = if total_q_groups > 0 {
                            uncovered / (total_q_groups as f32)
                        } else {
                            0.0
                        };
                        let same_mask_seen = mask_counts.get(&m_eff).copied().unwrap_or(0);
                        let policy_prior = s_policy_floored[si];
                        let pref_prior = s_pref_floored[si] && !policy_prior;
                        let zero_prior = (policy_prior || pref_prior) && m_eff == 0;
                        let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
                        let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
                        let zero_slots = selected.iter().filter(|&&s| {
                            (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
                        }).count();
                        if (policy_prior && policy_slots >= 1)
                            || (pref_prior && pref_slots >= 1)
                            || (zero_prior && zero_slots >= 1)
                        {
                            continue;
                        }
                        let shared_ent = ix.session_entities[si].iter().any(|e| covered_ents.contains(e));
                        let same_date_seen = eff_it.is_event_sequence && selected.iter().any(|&s| ix.session_ts[s] == ix.session_ts[si] && ix.session_ts[si] > 0);
                        let topic_seen = eff_it.is_event_sequence
                            && !initiative_content[si].is_empty()
                            && selected.iter().any(|&s| jaccard(&initiative_content[s], &initiative_content[si]) >= 0.45);
                        let new_event = eff_it.is_event_sequence && !initiative_content[si].is_empty() && !topic_seen;
                        let min_ochiai = selected
                            .iter()
                            .filter_map(|&s| ochiai_divergence(&ix.session_rare_nt[s], &ix.session_rare_nt[si]))
                            .fold(None::<f32>, |acc, d| Some(acc.map_or(d, |a| a.min(d))));
                        let distinct_date = ix.session_ts[si] > 0
                            && selected.iter().all(|&s| ix.session_ts[s] != ix.session_ts[si]);
                        let distinct_episode = min_ochiai.is_some_and(|d| d >= 0.58)
                            && (ix.session_ts[si] == 0 || distinct_date);
                        let redundancy = if same_date_seen {
                            0.25
                        } else if topic_seen {
                            0.35
                        } else if !eff_it.is_event_sequence
                            && !it.sniper_how_many
                            && !distinct_episode
                            && !distinct_date
                            && uncovered == 0.0
                            && bridge_boost[si] < 0.12 * smax
                            && shared_ent
                            && same_mask_seen >= 3
                        {
                            0.62
                        } else {
                            1.0
                        };
                        let event_cover = if new_event { 0.65 } else { 0.0 };
                        let gain = (s_score[si] + bridge_boost[si]) * (1.0 + 0.90 * uncovered_ratio + event_cover) * redundancy;
                        found = true;
                        if gain > best_gain {
                            best_gain = gain;
                            best_pos = idx;
                        }
                    }
                    if !found {
                        break;
                    }
                    let chosen = remaining.remove(best_pos);
                    selected.push(chosen);
                    covered_mask |= s_mask[chosen];
                    covered_ents.extend(ix.session_entities[chosen].iter().copied());
                    *mask_counts.entry(s_mask[chosen] & mask_limit).or_insert(0) += 1;
                }
            }
        }
        for si in s_ord_base {
            if selected.len() >= pool_cap {
                break;
            }
            if selected.contains(si) {
                continue;
            }
            if s_score[*si] <= 0.0 && bridge_boost[*si] <= 0.0 {
                continue;
            }
            let policy_prior = s_policy_floored[*si];
            let pref_prior = s_pref_floored[*si] && !policy_prior;
            let zero_prior = (policy_prior || pref_prior) && (s_mask[*si] & mask_limit) == 0;
            let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
            let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
            let zero_slots = selected.iter().filter(|&&s| {
                (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
            }).count();
            if (policy_prior && policy_slots >= 1)
                || (pref_prior && pref_slots >= 1)
                || (zero_prior && zero_slots >= 1)
            {
                continue;
            }
            selected.push(*si);
        }
        selected
    } else {
        s_ord_base
            .iter()
            .copied()
            .filter(|&si| s_score[si] > 0.0 || bridge_boost[si] > 0.0)
            .take(pool_cap)
            .collect()
    }
}

pub(in crate::retrieval) fn select_submodular_turns(
    ctx: &RankCtx<'_>,
    turn_ts: &[i64],
    has_named_speakers_top: bool,
    max_per_session_top5: usize,
    allow_short_reply_bypass: bool,
) -> Vec<usize> {
    let ix = ctx.ix;
    let p = ctx.p;
    let nt = ctx.nt;
    let k = ctx.k;
    let plan = &ctx.plan;
    let it = &ctx.plan.intent;
    let eff_it = &ctx.eff_it;
    let lexical_tmax = ctx.lexical_tmax;
    let t_scene = &ctx.t_scene;
    let t_mask = &ctx.t_mask;
    let t_exact_mask = &ctx.t_exact_mask;
    let t_policy_floored = &ctx.t_policy_floored;
    let t_pref_floored = &ctx.t_pref_floored;
    let final_session_indices = &ctx.final_session_indices;
    let is_monologue_corpus = ctx.is_monologue_corpus;

if p.enable_submodular {
        let mut selected_turns: Vec<usize> = Vec::with_capacity(k);
        let mut covered_query_terms: u64 = 0;
        let mut session_counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        // Ingest order. Equal gains keep the earlier turn, then the higher recorded stability.
        let mut remaining: Vec<usize> = (0..nt).filter(|&ti| t_scene[ti] > 0.0).collect();

        let total_query_groups = plan.groups.len().min(64);
        let turn_mask_limit = if total_query_groups == 64 { u64::MAX } else { (1u64 << total_query_groups) - 1 };

        while selected_turns.len() < k && !remaining.is_empty() {
            let mut best_turn: Option<usize> = None;
            let mut best_pos = 0usize;
            let mut best_gain = -1.0f32;

            for (pos, &ti) in remaining.iter().enumerate() {
                let sc = t_scene[ti];
                let t_sess = ix.turns[ti].session;
                let count = session_counts.get(&t_sess).copied().unwrap_or(0);
                let topic_repeat = eff_it.is_event_sequence
                    && ix.turns[ti].is_initiative
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].is_initiative && jaccard(&ix.turns[prev].content, &ix.turns[ti].content) >= 0.45
                    });
                let t_m = t_mask[ti];
                let uncovered_mask = t_exact_mask[ti] & !covered_query_terms;
                let uncovered_count = (uncovered_mask & turn_mask_limit).count_ones() as f32;
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() == 1
                    });
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_m & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                // One chat may not fill the top 5. A short reply beside a hit is the same evidence, not a new slot.
                // On named-speaker dialogues, an adjacent exchange partner (e.g. Q->A or caption->reply)
                // may take a 2nd seat when the session currently holds only 1 declarative seat.
                let adj_exchange = has_named_speakers_top
                    && !eff_it.is_event_sequence
                    && !ix.turns[ti].is_question
                    && !ix.turns[ti].is_backchannel
                    && count < 2
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() <= 2
                    });
                let bypass_cap = (short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop && allow_short_reply_bypass)
                    || adj_exchange;
                let count_for_cap = if has_named_speakers_top && !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel {
                    selected_turns
                        .iter()
                        .filter(|&&prev| {
                            ix.turns[prev].session == t_sess
                                && !ix.turns[prev].is_question
                                && !ix.turns[prev].is_backchannel
                        })
                        .count()
                } else {
                    count
                };
                let session_capped = selected_turns.len() < 5
                    && count_for_cap >= max_per_session_top5
                    && !bypass_cap;
                if session_capped {
                    continue;
                }
                let diversity_factor = if short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop {
                    1.0
                } else if count == 0 && (!topic_repeat || (!eff_it.is_event_sequence && sc >= 0.45 * lexical_tmax)) {
                    1.0
                } else if eff_it.is_event_sequence || topic_repeat {
                    0.15
                } else if eff_it.multi_hop && count >= 1 && uncovered_count == 0.0 {
                    0.15
                } else if count == 1 {
                    0.70
                } else {
                    0.40
                } * adj_partner_diversity_factor(
                    ix,
                    ti,
                    t_sess,
                    &selected_turns,
                    it.assistant_attribution,
                    short_reply,
                    max_per_session_top5,
                );
                let uncovered_ratio = if total_query_groups > 0 {
                    uncovered_count / (total_query_groups as f32)
                } else {
                    0.0
                };

                let s_rank = final_session_indices.iter().take(5).position(|&s| s == t_sess);
                let couple = s_rank.map(|r| 1.0 + 0.50 / (1.0 + r as f32)).unwrap_or(1.0);
                let uncov_w = if is_monologue_corpus && !selected_turns.is_empty() {
                    1.35
                } else {
                    0.60
                };
                let gain = sc * (1.0 + uncov_w * uncovered_ratio) * diversity_factor * couple;
                let stab = ix.turns[ti].stability;
                let take = match best_turn {
                    None => true,
                    Some(cur) => {
                        gain > best_gain
                            || (gain == best_gain && stab > ix.turns[cur].stability)
                    }
                };
                if take {
                    best_gain = gain;
                    best_turn = Some(ti);
                    best_pos = pos;
                }
            }

            if let Some(chosen) = best_turn {
                selected_turns.push(chosen);
                remaining.remove(best_pos);
                covered_query_terms |= t_exact_mask[chosen];
                *session_counts.entry(ix.turns[chosen].session).or_insert(0) += 1;
            } else {
                break;
            }
        }

        if selected_turns.len() < k {
            let ord = order(&t_scene, &turn_ts);
            for ti in ord {
                if selected_turns.len() >= k {
                    break;
                }
                if selected_turns.contains(&ti) {
                    continue;
                }
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_mask[ti] & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                let count = selected_turns.iter().filter(|&&p| ix.turns[p].session == ix.turns[ti].session).count();
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == ix.turns[ti].session && (prev as isize - ti as isize).abs() == 1
                    });
                let bypass_cap = short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop && allow_short_reply_bypass;
                if selected_turns.len() < 5 && count >= max_per_session_top5 && !bypass_cap {
                    continue;
                }
                selected_turns.push(ti);
            }
        }
        selected_turns
    } else {
        order(&t_scene, &turn_ts)
            .into_iter()
            .filter(|&ti| t_scene[ti] > 0.0)
            .take(k)
            .collect()
    }
}
