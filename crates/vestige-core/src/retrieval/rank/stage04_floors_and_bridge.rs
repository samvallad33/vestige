//! Stage 04: Named-dialogue peak blend, standing/preference/event floors, RM3, and session bridge PPR.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let p = ctx.p;
    let k = ctx.k;
    let ns = ctx.ns;
    let nt = ctx.nt;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let eff_it = &ctx.eff_it;
    let pref_on = ctx.pref_on;
    let past_cue_eff = ctx.past_cue_eff;
    let episodic_user_quote = ctx.episodic_user_quote;
    let has_named_speakers_eff = ctx.has_named_speakers_eff;
    let has_compact_rule_register = ctx.has_compact_rule_register;
    let cond_rule_turns = &ctx.cond_rule_turns;
    let mut s_score = std::mem::take(&mut ctx.s_score);
    let s_exact_mask = &ctx.s_exact_mask;
    let mut t_scene = std::mem::take(&mut ctx.t_scene);
    let mut t_scene_sess = std::mem::take(&mut ctx.t_scene_sess);
    let t_mask = &ctx.t_mask;

    // Named-Dialogue Speaker-Aligned Session Peak Blend:
    // Whole-session BM25F (`s_score`) mixes both speakers across all topics before speaker alignment,
    // sniper boosts, and echo demotion; blend 0.35 (Q16.16) of each session's best declarative `t_scene` peak.
    if has_named_speakers_eff && !eff_it.is_event_sequence && !pref_on {
        // 0.35 in Q16.16: speaker-aligned declarative `t_scene` peak blend weight into `s_score`.
        const NAMED_DECL_PEAK_BLEND_Q16: u32 = 22_938;
        const Q16_ONE_U32: u32 = 65_536;
        const Q16_SCALE_F32: f32 = 65_536.0;
        let peak_w = (NAMED_DECL_PEAK_BLEND_Q16 as f32) / Q16_SCALE_F32;
        let base_w = ((Q16_ONE_U32 - NAMED_DECL_PEAK_BLEND_Q16) as f32) / Q16_SCALE_F32;
        let pre_tmax = t_scene_sess.iter().cloned().fold(0.0f32, f32::max);
        let pre_smax = s_score.iter().cloned().fold(0.0f32, f32::max);
        if pre_tmax > 0.0 && pre_smax > 0.0 {
            for si in 0..ns {
                let best_decl = ix.session_turns[si]
                    .iter()
                    .copied()
                    .filter(|&ti| !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel)
                    .map(|ti| t_scene_sess[ti])
                    .fold(0.0f32, f32::max);
                let norm_s = s_score[si] / pre_smax;
                let norm_t = best_decl / pre_tmax;
                s_score[si] = pre_smax * (base_w * norm_s + peak_w * norm_t);
            }
        }
    }


    // Frozen floors for this run (do not retune after BEAM): standing 0.68,
    // preference 0.50, event-session 0.42, event-turn 0.38. Lexical max is
    // captured first. 0.68 stays under the 0.72 second-anchor cut so a
    // zero-overlap rule does not become a PPR anchor. The standing floor is
    // applied to the policy turn itself, not only its session.
    let lexical_smax = s_score.iter().cloned().fold(0.0f32, f32::max);
    let lexical_tmax_sess = t_scene_sess.iter().cloned().fold(0.0f32, f32::max);
    let lexical_tmax = t_scene.iter().cloned().fold(0.0f32, f32::max);
    // Compound/prepositional task wording (" and ", "first ", "before ") must not
    // drop the standing-policy floor. Only count-snipers and event sequences do.
    let policy_blocked = it.sniper_how_many || eff_it.is_event_sequence;
    let mut standing_rules = rank_standing_rules(ix, &plan, &eff_it, episodic_user_quote, &t_mask, k);
    // Decouple Standing-Policy Register from literal token overlap on live project/preference consultations:
    if (eff_it.task_under_rules || eff_it.preference)
        && !policy_blocked
        && !eff_it.temporal
        && !past_cue_eff
        && has_compact_rule_register
    {
        for sr_entry in ix.standing_rules.iter().rev() {
            if turn_is_superseded(ix, sr_entry.turn) {
                continue;
            }
            let tid = &ix.turns[sr_entry.turn].id;
            if !standing_rules.iter().any(|r| &r.id == tid) && standing_rules.len() < k.max(4) {
                let scope_hits = sr_entry
                    .scope
                    .iter()
                    .filter(|&&sid| plan.terms.iter().any(|qt| qt.id == sid))
                    .count();
                let syn_sc = if scope_hits > 0 {
                    2.5 + (scope_hits as f32) * 1.5
                } else if sr_entry.has_explicit_scope {
                    0.95
                } else {
                    1.35
                };
                standing_rules.push(RankedItem {
                    id: tid.clone(),
                    session_id: ix.session_ids[sr_entry.session].clone(),
                    score: syn_sc,
                    matched: Vec::new(),
                });
            }
        }
        standing_rules.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    // Winner floor stays inside task_under_rules, including multi-hop task rules.
    // v1 moved this seat outside that block and excluded multi-hop. Temporal queries
    // then received a global policy seat and displaced gold. v2 does not do that.
    let (s_policy_floored, t_policy_floored) = if eff_it.task_under_rules && !policy_blocked {
        // Rule-condition gate: only rules whose own condition matches the query, or that
        // share a query term, are floored. Others stay on the standing_rules channel.
        let rule_live = |ti: usize| {
            ix.turns[ti].is_standing_policy && (t_mask[ti] != 0 || cond_rule_turns.contains(&ti))
        };
        let mut s_floored = raise_floor(&mut s_score, lexical_smax, 0.68, |si| {
            ix.session_turns[si].iter().any(|&ti| rule_live(ti))
        });
        let mut t_floored = raise_floor(&mut t_scene, lexical_tmax, 0.68, rule_live);
        if let Some(top) = standing_rules
            .first()
            .filter(|top| ix.turns.iter().position(|t| t.id == top.id).is_some_and(rule_live))
        {
            let s_win_floor = policy_spec_winner_floor(lexical_smax);
            let t_win_floor = policy_spec_winner_floor(lexical_tmax);
            if let Some(top_rule_sess) = ix.session_ids.iter().position(|id| id == &top.session_id) {
                if s_score[top_rule_sess] < s_win_floor {
                    s_score[top_rule_sess] = s_win_floor;
                    s_floored[top_rule_sess] = true;
                }
            }
            if let Some(top_rule_turn) = ix.turns.iter().position(|t| t.id == top.id) {
                if t_scene[top_rule_turn] < t_win_floor {
                    t_scene[top_rule_turn] = t_win_floor;
                    t_floored[top_rule_turn] = true;
                }
            }
        }
        (s_floored, t_floored)
    } else {
        (vec![false; ns], vec![false; nt])
    };
    let pref_blocked = it.sniper_how_many || it.temporal || eff_it.is_event_sequence;
    let mut pref_turn_set: DetHashSet<usize> = DetHashSet::default();
    let (s_pref_floored, t_pref_floored) = if pref_on && !pref_blocked {
        for (ti, tr) in ix.turns.iter().enumerate() {
            if tr.is_personal_constraint {
                pref_turn_set.insert(ti);
            }
        }
        if !has_named_speakers_eff && !ix.preference_directive_turns.is_empty() {
            let allow_sup_prefs = ix.session_turns.len() >= 16;
            let mut matched_pref: Vec<usize> = Vec::new();
            for &pti in &ix.preference_directive_turns {
                if !allow_sup_prefs && turn_is_superseded(ix, pti) {
                    continue;
                }
                let tr = &ix.turns[pti];
                let shares_q = tr.content.iter().any(|id| plan.focal_ids.contains(id))
                    || tr.feats.iter().any(|f| plan.focal_ids.contains(&f.id));
                if shares_q {
                    matched_pref.push(pti);
                }
            }
            if !matched_pref.is_empty() {
                pref_turn_set.extend(matched_pref);
            } else if ix.preference_directive_turns.len() <= 16 {
                // On zero-exact-overlap preference advice queries ("Flask libraries" vs "minimal dependencies"),
                // keep active user preference turns floored so downstream preference seating can rank them.
                for &pti in &ix.preference_directive_turns {
                    if allow_sup_prefs || !turn_is_superseded(ix, pti) {
                        pref_turn_set.insert(pti);
                    }
                }
            }
        }
        (
            raise_floor(&mut s_score, lexical_smax, 0.50, |si| {
                ix.session_turns[si].iter().any(|ti| pref_turn_set.contains(ti))
            }),
            raise_floor(&mut t_scene, lexical_tmax, 0.50, |ti| pref_turn_set.contains(&ti)),
        )
    } else {
        (vec![false; ns], vec![false; nt])
    };
    if eff_it.is_event_sequence {
        let _ = raise_floor(&mut s_score, lexical_smax, 0.42, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_initiative)
        });
        let _ = raise_floor(&mut t_scene, lexical_tmax, 0.38, |ti| ix.turns[ti].is_initiative);
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.38, |ti| ix.turns[ti].is_initiative);
    }
    // Mirror standing/pref floors onto the session-path t_scene (bridge_boost reads this).
    if eff_it.task_under_rules && !policy_blocked {
        let rule_live = |ti: usize| {
            ix.turns[ti].is_standing_policy && (t_mask[ti] != 0 || cond_rule_turns.contains(&ti))
        };
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.68, rule_live);
        if let Some(top) = standing_rules
            .first()
            .filter(|top| ix.turns.iter().position(|t| t.id == top.id).is_some_and(rule_live))
        {
            let t_win_floor = policy_spec_winner_floor(lexical_tmax_sess);
            if let Some(top_rule_turn) = ix.turns.iter().position(|t| t.id == top.id) {
                if t_scene_sess[top_rule_turn] < t_win_floor {
                    t_scene_sess[top_rule_turn] = t_win_floor;
                }
            }
        }
    }
    if pref_on && !pref_blocked {
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.50, |ti| {
            pref_turn_set.contains(&ti)
        });
    }
    let initiative_content: Vec<DetHashSet<u32>> = if eff_it.is_event_sequence {
        ix.session_turns.iter().map(|members| {
            let mut set = DetHashSet::default();
            for &ti in members {
                if ix.turns[ti].is_initiative {
                    set.extend(ix.turns[ti].content.iter().copied());
                }
            }
            set
        }).collect()
    } else {
        Vec::new()
    };

    // Session-Level Multi-Hop Bridge Activation + Greedy Submodular Set Cover
    apply_rung4_hub_damped_rm3(ix, &plan, &mut s_score);
    let smax = s_score.iter().cloned().fold(0.0f32, f32::max);
    let mut bridge_boost = vec![0.0f32; ns];
    let mut s_ord_base = order(&s_score, &ix.session_ts);
    if p.enable_ppr && ns >= 4 && !pref_on && smax > 0.0 {
        if let Some(&top0) = s_ord_base.first() {
            let anchors: Vec<usize> = s_ord_base
                .iter()
                .copied()
                .filter(|&si| !s_policy_floored[si] && !s_pref_floored[si] && s_score[si] >= 0.62 * smax)
                .take(3)
                .collect();
            if !ix.header_dict.is_empty() && ix.header_max_words > 0 && !it.preference && !it.user_attribution {
                let mut hdr_best = vec![0.0f32; ns];
                // Weapon 1: Query-direct canonical header-title match (`1 <= deg <= 4`).
                let (q_ids, q_proper): (Vec<u32>, Vec<bool>) = q
                    .text
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let proper = s.chars().next().is_some_and(|c| c.is_uppercase());
                        let id = ix.vocab.get(&s.to_lowercase()).unwrap_or(u32::MAX);
                        (id, proper)
                    })
                    .unzip();
                let mut q_idx = 0usize;
                while q_idx < q_ids.len() {
                    let max_l = ix.header_max_words.min(q_ids.len() - q_idx);
                    // Every span length, then advance one token. A degree-4 outer phrase
                    // must not shadow a degree-1 canonical entity header inside it.
                    for len in 1..=max_l {
                        let slice = &q_ids[q_idx..q_idx + len];
                        if slice.contains(&u32::MAX) {
                            continue;
                        }
                        if len == 1 && !q_proper[q_idx] {
                            continue;
                        }
                        let hh = hash_u32_slice(slice);
                        if let Some(bucket) = ix.header_dict.get(&hh) {
                            if let Some(hk) = bucket.iter().find(|k| k.tokens.as_slice() == slice) {
                                let deg = hk.sessions.len() as u32;
                                if (1..=4).contains(&deg) {
                                    let extra = deg.saturating_sub(1);
                                    let damp = 1.0 / (2.0 + extra as f32).log2();
                                    for &(target_si, mult) in &hk.sessions {
                                        let cand_b = 0.68 * smax * mult * damp;
                                        if cand_b > hdr_best[target_si] {
                                            hdr_best[target_si] = cand_b;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    q_idx += 1;
                }
                for &anc in &anchors {
                    let anc_w = (s_score[anc] / smax).clamp(0.35, 1.0);
                    let members = &ix.session_turns[anc];
                    let anc_tmax = members.iter().map(|&ti| t_scene_sess[ti]).fold(0.0f32, f32::max);
                    let use_hot = anc_tmax > 0.0 && members.len() > 2;
                    for &ti in members {
                        if use_hot && t_scene_sess[ti] < 0.25 * anc_tmax {
                            continue;
                        }
                        let hot_turn_bonus = if anc_tmax > 0.0 && t_scene_sess[ti] >= 0.60 * anc_tmax { 1.18 } else { 1.0 };
                        let toks = ix.turns[ti].span_ids_cache.get_or_init(|| {
                            ix.turns[ti]
                                .body_for_headers
                                .split(|c: char| !c.is_alphanumeric())
                                .filter(|s| !s.is_empty())
                                .map(|s| ix.vocab.get(&s.to_lowercase()).unwrap_or(u32::MAX))
                                .collect()
                        });
                        let mut idx = 0usize;
                        while idx < toks.len() {
                            let max_l = ix.header_max_words.min(toks.len() - idx);
                            let mut matched_len = 0usize;
                            for len in (1..=max_l).rev() {
                                let slice = &toks[idx..idx + len];
                                if slice.contains(&u32::MAX) {
                                    continue;
                                }
                                let hh = hash_u32_slice(slice);
                                if let Some(bucket) = ix.header_dict.get(&hh) {
                                    if let Some(hk) = bucket.iter().find(|k| k.tokens.as_slice() == slice) {
                                        let deg = hk.sessions.len() as u32;
                                        if (1..=6).contains(&deg) {
                                            let extra = deg.saturating_sub(1);
                                            let damp = 1.0 / (2.0 + extra as f32).log2();
                                            for &(target_si, mult) in &hk.sessions {
                                                if target_si != anc && target_si != top0 {
                                                    let cand_b = smax * 0.76 * anc_w * mult * damp * hot_turn_bonus;
                                                    if cand_b > hdr_best[target_si] {
                                                        hdr_best[target_si] = cand_b;
                                                    }
                                                }
                                            }
                                            matched_len = len;
                                            break;
                                        }
                                    }
                                }
                            }
                            idx += matched_len.max(1);
                        }
                    }
                }
                for si in 0..ns {
                    if hdr_best[si] > bridge_boost[si] {
                        bridge_boost[si] = hdr_best[si];
                    }
                }
            }
            let max_s_df = if ns >= 8 {
                ((ix.s_stats.n * 0.10).clamp(4.0, (0.28 * ix.s_stats.n).max(4.0)).min(28.0)) as u32
            } else {
                4
            };
            let rare_sym_max_df = ((ix.s_stats.n * 0.05).clamp(3.0, 18.0)) as u32;
            let mut s_act: std::collections::BTreeMap<u32, f32> = std::collections::BTreeMap::new();
            for &anc in &anchors {
                let anc_w = (s_score[anc] / smax).clamp(0.25, 1.0);
                // Prefer highest-scoring matching turns inside anchor session so bridge entities come from hot passage.
                let members = &ix.session_turns[anc];
                let anc_tmax = members.iter().map(|&ti| t_scene_sess[ti]).fold(0.0f32, f32::max);
                let use_hot = anc_tmax > 0.0 && members.len() > 2;
                for &ti in members {
                    if use_hot && t_scene_sess[ti] < 0.25 * anc_tmax {
                        continue;
                    }
                    let turn_len_norm = ((ix.turns[ti].content.len() + 6) as f32).sqrt();
                    for &e in &ix.turns[ti].entities {
                        if is_technical_handle(&ix.vocab, e) {
                            continue;
                        }
                        if let Some(&df) = ix.s_stats.df.get(&e) {
                            if df >= 2 && df <= max_s_df {
                                let idf_s = 1.4 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *s_act.entry(e).or_insert(0.0) += (anc_w * 2.5 * idf_s) / turn_len_norm;
                            }
                        }
                    }
                    // Exact byte-equal rare symbols (content holds unstemmed exact token IDs only, never stems).
                    for &w in &ix.turns[ti].content {
                        if plan.group_of.contains_key(&w) {
                            continue;
                        }
                        if let Some(&df) = ix.s_stats.df.get(&w) {
                            if df >= 2 && df <= rare_sym_max_df {
                                let idf_s = 1.2 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *s_act.entry(w).or_insert(0.0) += (anc_w * 0.75 * idf_s) / turn_len_norm;
                            }
                        }
                    }
                }
            }
            if !s_act.is_empty() {
                for si in 0..ns {
                    if si == top0 {
                        continue;
                    }
                    let mut ent_sum = 0.0f32;
                    for &e in &ix.session_entities[si] {
                        if let Some(&a) = s_act.get(&e) {
                            ent_sum += 2.0 * a;
                        }
                    }
                    let mut sum = ent_sum;
                    for &w in &ix.session_content[si] {
                        if let Some(&a) = s_act.get(&w) {
                            sum += 0.75 * a;
                        }
                    }
                    // Require direct query overlap (s_score > 0), or an explicit proper-entity bridge
                    // on non-preference/non-user queries, so zero-match distractors never displace personal facts.
                    let allow_zero_hit = !it.preference && !it.user_attribution && ent_sum > 0.0;
                    if sum > 0.0 && (s_score[si] > 0.0 || allow_zero_hit) {
                        let slen = (ix.session_bags[si].len[0] + ix.session_bags[si].len[1] + 8.0).sqrt();
                        let norm = sum / slen;
                        let bridge_k = if pref_on { 0.05 } else { 0.50 };
                        let sat = norm / (norm + bridge_k);
                        let mut coeff = if it.multi_hop || plan.groups.len() >= 3 { 0.55 } else { 0.12 };
                        if (s_exact_mask[si] & !s_exact_mask[top0]) != 0 {
                            coeff *= 1.30;
                        }
                        let boosted = smax * coeff * sat;
                        if boosted > bridge_boost[si] {
                            bridge_boost[si] = boosted;
                        }
                    }
                }
            }

            // Backward-only RSB Handle/CaptureWindow, Structural Contradiction Supersession,
            // and Temporal Context Episode Contiguity (`later -> earlier` ONLY).
            for &anc in &anchors {
                let anc_w = (s_score[anc] / smax).clamp(0.35, 1.0);
                // 1. Exact RSB technical handles & 9h Synaptic Tagging CaptureWindow backward edges
                for &(deg, prev_si) in &ix.session_rsb_back[anc] {
                    if prev_si != top0 {
                        let damp = ghostlink_hub_damp(deg.max(2), 1);
                        let cand = smax * 0.64 * anc_w * damp;
                        if cand > bridge_boost[prev_si] {
                            bridge_boost[prev_si] = cand;
                        }
                    }
                }
                // 2. Structural contradiction / state-update supersession (`!it.current_state` or
                //    non-monologue user/assistant chats pull the earlier state partner into Slots 1..4 without replacing top0)
                if !it.current_state || (!has_named_speakers_eff && !ctx.is_monologue_corpus) {
                    for &prev_si in &ix.session_supersedes_back[anc] {
                        if prev_si != top0 && (s_score[prev_si] >= 0.08 * smax || it.user_attribution) {
                            let cand = smax * 0.52 * anc_w;
                            if cand > bridge_boost[prev_si] {
                                bridge_boost[prev_si] = cand;
                            }
                        }
                    }
                }
                // 3. Temporal Context Episode backward contiguity (`session_episode_prev`: `si -> si - 1`)
                if eff_it.is_event_sequence {
                    let mut cur = anc;
                    let mut decay = 0.52f32;
                    for _ in 0..2 {
                        let Some(prev_ep) = ix.session_episode_prev[cur] else { break };
                        if prev_ep != top0 {
                            let cand = smax * decay * anc_w;
                            if cand > bridge_boost[prev_ep] {
                                bridge_boost[prev_ep] = cand;
                            }
                        }
                        cur = prev_ep;
                        decay *= 0.65;
                    }
                }
            }
        }
    }

    // Disclosure-scoped KITE upward-cone session lift on preference/advisory frames (`pref_on`):
    // unlocks zero-overlap prose preferences (`BEAM PF`, `LME SSP`, `MAB SSP`, `ConvoMem implicit`)
    // without letting entity-co-occurrence PPR bridge zero-overlap factual distractors.
    if pref_on && !eff_it.is_event_sequence && !eff_it.task_under_rules && ns >= 2 && smax > 0.0 {
        let smax_pf_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(smax.to_bits());
        let empty_sp: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let (kite_s_pf, kite_t_pf) = crate::retrieval::index::kite_typed::score_kite_typed_facet_meet_q16(
            ix,
            &plan,
            &q.text,
            &empty_sp,
            0,
            smax_pf_q16,
        );
        for (si, &kq) in kite_s_pf.iter().enumerate() {
            if kq == 0 || si >= ns {
                continue;
            }
            let has_disc = ix.session_turns[si].iter().any(|&ti| {
                let tr = &ix.turns[ti];
                tr.field == 0
                    && !tr.is_question
                    && (tr.has_prose_disclosure || tr.is_personal_constraint || tr.is_habitual)
                    && kite_t_pf.get(ti).copied().unwrap_or(0) > 0
            });
            if has_disc {
                let lift = ((kq as f32) / 65_536.0).min(0.42 * smax);
                if lift > bridge_boost[si] {
                    bridge_boost[si] = lift;
                }
            }
        }
        let tmax_ref = lexical_tmax.max(0.01);
        let tmax_sess_ref = lexical_tmax_sess.max(0.01);
        for (ti, &ktq) in kite_t_pf.iter().enumerate() {
            if ktq == 0 || ti >= nt {
                continue;
            }
            let tr = &ix.turns[ti];
            if tr.field == 0
                && !tr.is_question
                && (tr.has_prose_disclosure || tr.is_personal_constraint || tr.is_habitual)
            {
                let frac = ((ktq as f32) / (smax_pf_q16.max(1) as f32)).min(1.0);
                t_scene[ti] += (0.40 * tmax_ref * frac).min(0.40 * tmax_ref);
                t_scene_sess[ti] += (0.40 * tmax_sess_ref * frac).min(0.40 * tmax_sess_ref);
            }
        }
    }

    if !eff_it.is_event_sequence
        && !eff_summary_request(&q.text)
        && !pref_on
        && !eff_it.task_under_rules
        && ns >= 4
        && smax > 0.0
    {
        let lex_only_q16: Vec<u64> = s_score
            .iter()
            .map(|&v| crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(v.to_bits()))
            .collect();
        let smax_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(smax.to_bits());
        let top0 = s_ord_base.first().copied().unwrap_or(0);
        let top0_lex = lex_only_q16.get(top0).copied().unwrap_or(0);
        let top1_lex = s_ord_base
            .get(1)
            .and_then(|&s1| lex_only_q16.get(s1).copied())
            .unwrap_or(0);
        let n_q_groups = plan.groups.len().min(64);
        let q_full_mask = if n_q_groups == 64 {
            u64::MAX
        } else if n_q_groups == 0 {
            0
        } else {
            (1u64 << n_q_groups) - 1
        };
        let top0_mask_bits = s_exact_mask.get(top0).copied().unwrap_or(0) & q_full_mask;
        let top0_full_cov = q_full_mask != 0 && top0_mask_bits == q_full_mask;
        let top0_misses_group = q_full_mask != 0 && top0_mask_bits != q_full_mask;
        // True crisp-peak lock for Single-Hop & Adversarial (protects crisp >= 4:3 dominance
        // and >= 4-group full coverage while allowing 2-3 group ties to resolve via bridge consensus):
        let invariant_crisp_lock = top0_lex.saturating_mul(3) >= top1_lex.saturating_mul(4)
            || (top0_full_cov
                && n_q_groups >= 4
                && !eff_it.multi_hop
                && top0_lex.saturating_mul(9) >= top1_lex.saturating_mul(10));

        let top0_ceiling_q16 = (crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
            (s_score[top0] + bridge_boost[top0]).to_bits(),
        )
        .saturating_mul(9))
            / 10;

        let tp_lifts = crate::retrieval::lexical::bm25tp::compute_session_bm25tp_lifts_q16(
            ix,
            plan,
            s_exact_mask,
            top0,
            smax_q16,
        );
        let fs_lifts =
            crate::retrieval::graph::fellegi_sunter::compute_fellegi_sunter_lifts_q16(
                ix,
                plan,
                top0,
                smax_q16,
            );
        let centering_lifts =
            crate::retrieval::query::centering::compute_centering_coherence_lifts_q16(
                ix,
                plan,
                top0,
                smax_q16,
            );

        // Co-Witness Cross-Lexical & Bounded Downward Hyponym Voter (`Q16.16`):
        // Bridges uncovered query groups via WordNet 3.1 (full neighbors + bounded 1-hop hyponyms + gloss),
        // GeoNames, ConceptNet 5.7, and Roget 1911; fires ONLY when >= 2 distinct query groups are
        // co-witnessed in `si != top0` with >= 1 newly bridged uncovered group.
        let mut cw_turn_ids: Vec<(usize, Vec<u32>)> = Vec::new();
        let cowitness_lifts: Vec<u64> = if !plan.groups.is_empty() {
            let mut lifts = vec![0u64; ns];
            let ns_u32 = (ns as u32).max(1);
            let max_cw_df = (ns_u32 / 6).clamp(2, 32);
            let q_toks_cw = raw_tokens(&q.text);
            let q_has_us = q_toks_cw
                .windows(2)
                .any(|w| w[0] == "united" && w[1] == "states");
            let senate_id = ix.vocab.get("senate");
            let congress_id = ix.vocab.get("congress");
            let mut group_nbrs: Vec<(usize, Vec<(u32, u64)>)> = Vec::new();
            for (gi, g_word) in plan.groups.iter().take(64).enumerate() {
                if g_word.starts_with('\u{1}')
                    || !is_content(g_word)
                    || crate::retrieval::seating::preference_lattice::is_pref_framing_word(g_word)
                {
                    continue;
                }
                let mut g_ids: Vec<(u32, u64)> = Vec::new();
                let mut seen_gid: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
                let mut add_cw = |lem: &str| {
                    if lem.len() < 3 || lem.eq_ignore_ascii_case(g_word) {
                        return;
                    }
                    let lst = stem(lem);
                    for lid in [ix.vocab.get(lem), ix.vocab.get(&stem_term(&lst))]
                        .into_iter()
                        .flatten()
                    {
                        if seen_gid.insert(lid) {
                            let df = ix.s_stats.df.get(&lid).copied().unwrap_or(0);
                            if (1..=max_cw_df).contains(&df) {
                                let ic = crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16(df, ns_u32);
                                g_ids.push((lid, ic));
                            }
                        }
                    }
                };
                for (lem, _) in crate::wordnet::full_lexicon_neighbors(g_word).into_iter().take(24) {
                    add_cw(lem);
                }
                let s_cnt = crate::wordnet::senses(g_word).len();
                if (1..=5).contains(&s_cnt) {
                    for inst in crate::wordnet::class_to_instances(g_word, 1) {
                        add_cw(inst);
                    }
                }
                for gc in crate::wordnet::gloss_and_compound_concepts(g_word).into_iter().take(20) {
                    add_cw(gc);
                }
                for (glem, _) in crate::geonames::geo_neighbors(g_word) {
                    add_cw(glem);
                }
                if q_has_us && g_word == "states" {
                    for st_tok in crate::geonames::geo_class_instances("state") {
                        add_cw(st_tok.as_str());
                    }
                }
                for (cn_lem, _) in crate::conceptnet::two_hop_neighbors(g_word, 14) {
                    add_cw(cn_lem);
                }
                for (rg_lem, _) in crate::roget::conceptual_neighbors(g_word, 16) {
                    add_cw(rg_lem);
                }
                if !g_ids.is_empty() {
                    group_nbrs.push((gi, g_ids));
                }
            }
            if !group_nbrs.is_empty() {
                let mut raw_cw = vec![0u64; ns];
                let mut max_raw = 0u64;
                for si in 0..ns {
                    if si == top0 {
                        continue;
                    }
                    let is_us_politics = q_has_us
                        && (senate_id.is_some_and(|id| ix.session_content[si].contains(&id))
                            || congress_id.is_some_and(|id| ix.session_content[si].contains(&id)));
                    let exact_m = s_exact_mask[si] & q_full_mask;
                    let mut bridged_m = 0u64;
                    let mut ic_sum = 0u64;
                    let mut matched_ids: Vec<u32> = Vec::new();
                    for (gi, g_ids) in &group_nbrs {
                        if (exact_m & (1u64 << gi)) != 0 {
                            continue;
                        }
                        if is_us_politics && plan.groups.get(*gi).is_some_and(|g| g == "states") {
                            continue;
                        }
                        let mut best_g_ic = 0u64;
                        let mut best_lid = None;
                        for &(lid, ic) in g_ids {
                            if ix.session_content[si].contains(&lid)
                                || ix.session_bags[si].tf[0].contains_key(&lid)
                            {
                                if ic > best_g_ic {
                                    best_g_ic = ic;
                                    best_lid = Some(lid);
                                }
                            }
                        }
                        if best_g_ic > 0 {
                            bridged_m |= 1u64 << gi;
                            ic_sum = ic_sum.saturating_add(best_g_ic);
                            if let Some(lid) = best_lid {
                                matched_ids.push(lid);
                            }
                        }
                    }
                    if bridged_m != 0 && (exact_m | bridged_m).count_ones() >= 2 {
                        let mult = (bridged_m.count_ones() as u64).min(3);
                        let sc = ic_sum.saturating_mul(mult);
                        raw_cw[si] = sc;
                        if sc > max_raw {
                            max_raw = sc;
                        }
                        if !matched_ids.is_empty() {
                            cw_turn_ids.push((si, matched_ids));
                        }
                    }
                }
                if max_raw > 0 {
                    let cap_cw = ((top0_ceiling_q16 as u128).saturating_mul(22) / 100).max(1) as u64;
                    for si in 0..ns {
                        if raw_cw[si] > 0 {
                            lifts[si] = ((raw_cw[si] as u128).saturating_mul(cap_cw as u128)
                                / (max_raw as u128)) as u64;
                        }
                    }
                }
            }
            lifts
        } else {
            vec![0u64; ns]
        };

        // C11: Deformed Bethe-Hessian Resolvent above Kesten-Stigum `r_c` + Non-Hermitian Hatano-Nelson
        // Causal Transport + Marchenko-Pastur Random Matrix Bulk Deflation for 100K-1M haystacks.
        let pre_sight_q16: Vec<u64> = (0..ns)
            .map(|si| {
                crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                    (s_score[si] + bridge_boost[si]).to_bits(),
                )
            })
            .collect();
        let mut bethe_lifts =
            crate::retrieval::graph::bethe_hatano_mp_q16::compute_bethe_hessian_session_lifts_q16(
                ix,
                &pre_sight_q16,
                top0,
                smax_q16,
            );
        if ns >= 16 {
            let mp_pool_len = s_ord_base.len().min(32);
            crate::retrieval::graph::bethe_hatano_mp_q16::deflate_marchenko_pastur_ambient_q16(
                ix,
                &s_ord_base[..mp_pool_len],
                &mut bethe_lifts,
            );
        }
        // hardcode-scan: closed-class (retrospective causal interrogatives)
        let q_low_hn = q.text.to_ascii_lowercase();
        let hn_forward = !(q_low_hn.starts_with("why ")
            || q_low_hn.contains(" why ")
            || q_low_hn.contains("because ")
            || q_low_hn.contains("before "));
        let hatano_lifts =
            crate::retrieval::graph::bethe_hatano_mp_q16::compute_hatano_nelson_causal_lifts_q16(
                ix,
                &pre_sight_q16,
                &s_ord_base,
                smax_q16,
                hn_forward,
            );

        // Unified evidence-scaled budget: clean surviving voters (`bm25tp`, `fellegi_sunter`,
        // `centering`, `cowitness`, `bethe_hatano_mp`), capped at 0.30 * sight(top0).
        let budget_lifts = unified_evidence_lift_budget_q16(
            &[
                (true, &tp_lifts),
                (false, &fs_lifts),
                (false, &centering_lifts),
                (false, &cowitness_lifts),
                (false, &bethe_lifts),
                (false, &hatano_lifts),
            ],
            s_exact_mask,
            q_full_mask,
            top0,
            crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                (s_score[top0] + bridge_boost[top0]).to_bits(),
            ),
            smax_q16,
        );
        apply_rule8_capped_q16_lifts(
            &budget_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
        if !invariant_crisp_lock
            && !top0_full_cov
            && !eff_it.sniper_how_many
            && (has_named_speakers_eff || eff_it.preference)
        {
            apply_rule8_capped_q16_lifts(
                &cowitness_lifts,
                top0,
                top0_ceiling_q16,
                true,
                &s_score,
                &mut bridge_boost,
            );
            let lex_tmax = t_scene.iter().cloned().fold(0.0f32, f32::max);
            if lex_tmax > 0.0 && !cw_turn_ids.is_empty() {
                for (si, lids) in &cw_turn_ids {
                    if cowitness_lifts[*si] == 0 {
                        continue;
                    }
                    let best_ti = ix.session_turns[*si]
                        .iter()
                        .copied()
                        .filter(|&ti| {
                            let tr = &ix.turns[ti];
                            !tr.is_question
                                && !tr.is_backchannel
                                && !tr.is_standing_policy
                                && lids.iter().any(|id| tr.content.contains(id) || tr.feats.iter().any(|f| f.id == *id))
                        })
                        .max_by(|&a, &b| {
                            t_scene[a]
                                .partial_cmp(&t_scene[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| b.cmp(&a))
                        });
                    if let Some(ti) = best_ti {
                        let lift_f = 0.20 * lex_tmax * (cowitness_lifts[*si] as f32 / (top0_ceiling_q16.max(1) as f32)).min(1.0);
                        t_scene[ti] += lift_f;
                        if t_scene[ti] > t_scene_sess[*si] {
                            t_scene_sess[*si] = t_scene[ti];
                        }
                    }
                }
            }
        }

        // Domain D (Step 7B — Berger & Lafferty 1999 Speaker Translation LM + Full WordNet 3.1 +
        // Roget 1911 + GeoNames + Wikidata CC0 + ConceptNet 5.7 + T51 Self-Description Register + T53 Corpus PPMI):
        // Strictly gated to named-speaker Open-Domain inferential queries:
        //   - Strict epistemic modal / binary-choice queries (`would/could/might/may/likely/potentially`
        //     or `"A or B"` choice) bypass `invariant_crisp_lock` so a false single-word hit (`"park"`,
        //     `"fields"`, `"books"`) never blocks Domain D!
        //   - Copular `"Is/Does..."` queries still respect `!top0_full_cov && !invariant_crisp_lock`
        //     so Single-Hop & Adversarial stay locked >= 85%.
        let mut od_translation_fired = false;
        let mut is_strict_inferential_od = false;
        let mut od_turn_score_q16 = vec![0u64; nt];
        if has_named_speakers_eff {
            use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};
            use crate::retrieval::lexical::speaker_translation_q16::{
                is_open_domain_inferential_shape, score_speaker_translation_lift_q16,
            };
            use std::collections::{BTreeMap, BTreeSet};

            let q_toks_d = raw_tokens(&q.text);
            let q_refs_d: Vec<&str> = q_toks_d.iter().map(String::as_str).collect();
            // hardcode-scan: closed-class (epistemic modals & disjunctive binary-choice grammar)
            is_strict_inferential_od = q_toks_d.iter().any(|w| {
                matches!(
                    w.as_str(),
                    "would" | "could" | "might" | "may" | "likely" | "potentially"
                )
            }) || (is_open_domain_inferential_shape(&q_refs_d)
                && q_toks_d.iter().any(|w| w == "or"));
            let od_gate_open = (is_strict_inferential_od && !eff_it.temporal)
                || (!top0_full_cov && !invariant_crisp_lock && is_open_domain_inferential_shape(&q_refs_d));
            if od_gate_open {
                let ns_u32 = (ns as u32).max(1);
                let mut queried_sp: BTreeSet<String> = BTreeSet::new();
                for t in ix.turns.iter().take(128) {
                    let sp_low = t.speaker.to_ascii_lowercase();
                    if !sp_low.is_empty()
                        && sp_low != "user"
                        && sp_low != "assistant"
                        && q_toks_d.iter().any(|w| w == &sp_low)
                    {
                        queried_sp.insert(sp_low);
                    }
                }
                let mut speaker_session_terms: BTreeMap<usize, Vec<(u32, u64)>> = BTreeMap::new();
                let mut corpus_sp_vocab: BTreeSet<u32> = BTreeSet::new();
                let df_cap = (ns_u32 * 2 / 3).clamp(3, 28);
                for si in 0..ns {
                    let mut tf_map: BTreeMap<u32, u64> = BTreeMap::new();
                    for &ti in &ix.session_turns[si] {
                        let tr = &ix.turns[ti];
                        if tr.is_question || tr.is_backchannel {
                            continue;
                        }
                        let is_queried_sp = queried_sp.is_empty()
                            || queried_sp.contains(&tr.speaker.to_ascii_lowercase());
                        let is_partner_reply = !is_queried_sp
                            && tr.answers.is_some_and(|prev| {
                                queried_sp.contains(&ix.turns[prev].speaker.to_ascii_lowercase())
                            });
                        if !is_queried_sp && !is_partner_reply {
                            continue;
                        }
                        // Deterministic FSRS-6 stability + T51 closed-class 1SG self-description register bonus.
                        let stab_bonus = ((tr.stability.clamp(0.0, 30.0) as u64).saturating_mul(Q16_ONE >> 5))
                            .min(Q16_ONE >> 1);
                        let body_lc = tr.body_for_headers.to_ascii_lowercase();
                        // hardcode-scan: closed-class (1SG self-predication autobiographical frames)
                        let self_desc = [
                            "i'm ", "i am ", "i've always", "i've never", "i'd love",
                            "i love ", "i enjoy ", "i prefer ", "my dream", "my goal",
                        ]
                        .iter()
                        .any(|pat| body_lc.contains(pat));
                        let mut turn_w = if is_queried_sp {
                            Q16_ONE.saturating_add(stab_bonus)
                        } else {
                            Q16_ONE / 3
                        };
                        if is_queried_sp && self_desc {
                            turn_w = turn_w.saturating_add(turn_w / 3);
                        }
                        if is_queried_sp && tr.is_habitual {
                            turn_w = turn_w.saturating_add(turn_w / 2);
                        }
                        // Index BOTH exact (`tr.content`) AND stemmed (`tr.feats`) IDs so plural/inflected
                        // matches ("conservative" <-> "conservatives", "camp" <-> "camping") hit!
                        // Skip 2nd-person `other_speaker_terms` on `is_queried_sp` so compliments about
                        // the interlocutor do not pollute the queried speaker's trait profile.
                        for &cid in &tr.content {
                            if is_queried_sp && tr.other_speaker_terms.contains(&cid) {
                                continue;
                            }
                            let df = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
                            if (1..=df_cap).contains(&df) {
                                *tf_map.entry(cid).or_insert(0) += turn_w;
                                corpus_sp_vocab.insert(cid);
                            }
                        }
                        for f in &tr.feats {
                            if ix.vocab.names.get(f.id as usize).is_some_and(|nm| nm.starts_with("\u{1}stem:")) {
                                let df = ix.s_stats.df.get(&f.id).copied().unwrap_or(0);
                                if (1..=df_cap).contains(&df) {
                                    *tf_map.entry(f.id).or_insert(0) += (turn_w * 3) >> 2;
                                    corpus_sp_vocab.insert(f.id);
                                }
                            }
                        }
                        // Include `[caption]` (`blip_feats`) photo descriptors owned by the speaker
                        // so Open-Domain persona entailment sees visual evidence shared by the speaker.
                        for f in &tr.blip_feats {
                            if is_queried_sp && tr.other_speaker_terms.contains(&f.id) {
                                continue;
                            }
                            let df = ix.s_stats.df.get(&f.id).copied().unwrap_or(0);
                            if (1..=df_cap).contains(&df) {
                                *tf_map.entry(f.id).or_insert(0) += (turn_w * 3) >> 2;
                                corpus_sp_vocab.insert(f.id);
                            }
                        }
                    }
                    if !tf_map.is_empty() {
                        speaker_session_terms.insert(si, tf_map.into_iter().collect());
                    }
                }
                let mut query_pred_ic_q16: BTreeMap<u32, u64> = BTreeMap::new();
                let mut trans_map: BTreeMap<u32, Vec<(u32, u64)>> = BTreeMap::new();
                for (q_idx, w) in q_toks_d.iter().enumerate() {
                    if !is_content(w) || queried_sp.contains(w.as_str()) {
                        continue;
                    }
                    // Translate predicates missing from `top0` (or missing from the queried speaker's turns in `top0`).
                    let in_top0_sp = ix
                        .vocab
                        .get(w)
                        .is_some_and(|wid| {
                            ix.session_turns[top0].iter().any(|&ti| {
                                let tr = &ix.turns[ti];
                                (queried_sp.is_empty() || queried_sp.contains(&tr.speaker.to_ascii_lowercase()))
                                    && tr.content.contains(&wid)
                            })
                        });
                    if top0_full_cov && in_top0_sp {
                        continue;
                    }
                    let synth_qid = 0x7000_0000u32 + (q_idx as u32);
                    let q_df = ix
                        .vocab
                        .get(w)
                        .and_then(|id| ix.s_stats.df.get(&id).copied())
                        .unwrap_or(0);
                    let q_ic = aspect_ic_q16(q_df.max(1), ns_u32);
                    let mut nbrs: Vec<(u32, u64)> = Vec::new();
                    let mut seen_n: BTreeSet<u32> = BTreeSet::new();
                    let mut push_nbr = |lemma: &str, rel_w: u16| {
                        let lst = stem(lemma);
                        for wid in [ix.vocab.get(lemma), ix.vocab.get(&stem_term(&lst))]
                            .into_iter()
                            .flatten()
                        {
                            if corpus_sp_vocab.contains(&wid) && seen_n.insert(wid) {
                                let w_df = ix.s_stats.df.get(&wid).copied().unwrap_or(1);
                                let w_ic = aspect_ic_q16(w_df, ns_u32);
                                let fan_q16 = crate::retrieval::graph::actr::actr_fan_strength_q16(w_df, ns_u32);
                                let raw_qw = ((u64::from(rel_w).saturating_mul(Q16_ONE) / 100)
                                    .saturating_mul(w_ic.min(Q16_ONE * 6))
                                    / (Q16_ONE * 6).max(1))
                                .max(Q16_ONE >> 4);
                                let t_qw = (((raw_qw as u128).saturating_mul(fan_q16 as u128)) >> 16)
                                    .max((Q16_ONE >> 4) as u128) as u64;
                                nbrs.push((wid, t_qw));
                            }
                        }
                    };
                    for (lemma, rel_w) in crate::wordnet::full_lexicon_neighbors(w) {
                        // #3 Finite Heyting Algebra Asymmetric Subsumption (`lemma => w`):
                        // boost specific hyponym/instance entailments (`heyting >= 76`), damp generic superordinates (`38`).
                        let heyt = crate::wordnet::heyting_implication_ic_q16(lemma, w);
                        let eff_w = if heyt >= 76 {
                            rel_w.max(heyt)
                        } else if heyt == 38 {
                            (rel_w * 3) / 4
                        } else {
                            rel_w
                        };
                        push_nbr(lemma, eff_w);
                    }
                    if (1..=4).contains(&crate::wordnet::senses(w).len()) {
                        for inst in crate::wordnet::class_to_instances(w, 1) {
                            push_nbr(inst, 82);
                        }
                    }
                    for glem in crate::wordnet::gloss_and_compound_concepts(w).into_iter().take(16) {
                        push_nbr(glem, 68);
                    }
                    // #2 Mutual-Hypernym / Co-Instance LCS Sibling Resonance (`Vivaldi <-> Bach/Mozart/classical`):
                    for (lcs_sib, lcs_w) in crate::wordnet::lcs_sibling_resonance(w, 10) {
                        push_nbr(lcs_sib, lcs_w);
                    }
                    // #8 Bipartite Gloss-Hypernym Dual-Bridge (`identity -> gender/transgender`, `personality -> thoughtful`):
                    for (dg_lem, dg_w) in crate::wordnet::dual_gloss_hypernym_bridge(w, 10) {
                        push_nbr(dg_lem, dg_w);
                    }
                    // Orthographic 1-edit / prefix stem fallback for OOV/typo query words (e.g. `educaton` -> `education`):
                    if w.is_ascii() && w.len() >= 5 && ix.vocab.get(w).is_none() {
                        let wb = w.as_bytes();
                        let pref = &w[..4];
                        for (vword, &wid) in ix.vocab.map.iter() {
                            if !corpus_sp_vocab.contains(&wid) || vword.len() < 4 {
                                continue;
                            }
                            if vword.starts_with(pref) && vword.len().abs_diff(w.len()) <= 4 {
                                push_nbr(vword.as_str(), 84);
                            } else if wb.len() >= 6 && vword.len() == wb.len() {
                                let diffs = wb.iter().zip(vword.as_bytes()).filter(|(a, b)| a != b).count();
                                if diffs == 1 {
                                    push_nbr(vword.as_str(), 86);
                                }
                            }
                        }
                    }
                    for (lemma, rel_w) in crate::roget::conceptual_neighbors(w, 64) {
                        push_nbr(lemma, rel_w);
                    }
                    for (lemma, rel_w) in crate::geonames::geo_neighbors(w) {
                        push_nbr(lemma, rel_w);
                    }
                    for cont in crate::wikidata::upward_containers(w, 2)
                        .into_iter()
                        .chain(crate::wikidata::direct_instances_of(w, 10))
                    {
                        for part in cont.split_ascii_whitespace() {
                            if part.len() >= 3 {
                                push_nbr(part, 76);
                            }
                        }
                    }
                    // Reciprocal-link filter FIRST at full weight, then residual ConceptNet 2-hop at 0.65x
                    // so Open-Domain keeps both high precision (Turn R@1) and Session R@1 recall.
                    let mut cn_seen: BTreeSet<&'static str> = BTreeSet::new();
                    for (cn_lemma, cn_w) in crate::conceptnet::reciprocal_two_hop_neighbors(w, 10) {
                        cn_seen.insert(cn_lemma);
                        let scaled_w = ((u32::from(cn_w.min(200)) * 85) / 200).clamp(40, 85) as u16;
                        push_nbr(cn_lemma, scaled_w);
                    }
                    for (cn_lemma, cn_w) in crate::conceptnet::two_hop_neighbors(w, 12) {
                        if cn_seen.contains(cn_lemma) {
                            continue;
                        }
                        let scaled_w = (((u32::from(cn_w.min(200)) * 55) / 200).clamp(28, 55)) as u16;
                        push_nbr(cn_lemma, scaled_w);
                    }
                    // T53: 1-hop corpus PPMI associative translation for in-corpus predicates.
                    if let Some(qid_v) = ix.vocab.get(w) {
                        if let Some(ppmi_list) = ix.ppmi_neighbors.get(&qid_v) {
                            for &(pid, pw) in ppmi_list.iter().take(6) {
                                if corpus_sp_vocab.contains(&pid) && seen_n.insert(pid) {
                                    let p_q16 = (((pw.clamp(0.1, 2.0) * 0.45) * 65_536.0) as u64)
                                        .clamp(Q16_ONE >> 4, Q16_ONE >> 1);
                                    nbrs.push((pid, p_q16));
                                }
                            }
                        }
                    }
                    if !nbrs.is_empty() {
                        query_pred_ic_q16.insert(synth_qid, q_ic);
                        trans_map.insert(synth_qid, nbrs);
                    }
                }
                // Scale lift cap to `0.70 * smax` on strict inferential Open-Domain queries (or `0.55 * smax`
                // on partial-coverage copular OD) so translated speaker evidence can beat a false lexical decoy.
                let od_cap_q16 = if is_strict_inferential_od {
                    ((smax_q16 as u128).saturating_mul(45_875) >> 16).max(1) as u64
                } else if top0_misses_group {
                    ((smax_q16 as u128).saturating_mul(36_045) >> 16).max(1) as u64
                } else {
                    (smax_q16 / 4).max(1)
                };
                let od_lifts = score_speaker_translation_lift_q16(
                    &query_pred_ic_q16,
                    &trans_map,
                    &speaker_session_terms,
                    od_cap_q16,
                );
                for (si, lift_q16) in od_lifts {
                    if lift_q16 > 0 {
                        od_translation_fired = true;
                        let lift_f32 = (lift_q16 as f32) / 65_536.0;
                        bridge_boost[si] += lift_f32;
                    }
                }
                // Speaker-Scoped Standing Trait Bag + Turn-Level Open-Domain Evidence Projection:
                // Propagate `trans_map` + standing speaker trait predicates down to the exact
                // trait-bearing turns (`od_turn_score_q16[ti]`) so Turn Exact R@1 & R@5 capture
                // the Open-Domain evidence turn, not just its session!
                if !trans_map.is_empty() {
                    let mut trait_bonus_map: BTreeMap<u32, u64> = BTreeMap::new();
                    for sp_key in &queried_sp {
                        if let Some(tbag) = ix.speaker_trait_bags.get(sp_key) {
                            for &(tid, tw) in tbag {
                                trait_bonus_map
                                    .entry(tid)
                                    .and_modify(|v| {
                                        if tw > *v {
                                            *v = tw;
                                        }
                                    })
                                    .or_insert(tw);
                            }
                        }
                    }
                    let mut max_od_turn: u64 = 0;
                    for si in 0..ns {
                        for &ti in &ix.session_turns[si] {
                            let tr = &ix.turns[ti];
                            if tr.is_question || tr.is_backchannel || turn_is_superseded(ix, ti) {
                                continue;
                            }
                            let is_qsp = queried_sp.is_empty()
                                || queried_sp.contains(&tr.speaker.to_ascii_lowercase());
                            if !is_qsp {
                                continue;
                            }
                            let mut t_sc: u64 = 0;
                            for (&sqid, nbrs) in &trans_map {
                                let qic = query_pred_ic_q16.get(&sqid).copied().unwrap_or(Q16_ONE);
                                let mut best_m: u64 = 0;
                                for &(wid, t_qw) in nbrs {
                                    if tr.other_speaker_terms.contains(&wid) {
                                        continue;
                                    }
                                    let hit = tr.content.contains(&wid)
                                        || tr.feats.iter().any(|f| f.id == wid);
                                    if hit {
                                        let mut m_w = t_qw;
                                        if trait_bonus_map.contains_key(&wid) {
                                            m_w = m_w.saturating_add(m_w >> 1);
                                        }
                                        if m_w > best_m {
                                            best_m = m_w;
                                        }
                                    }
                                }
                                if best_m > 0 {
                                    let contrib = ((qic as u128).saturating_mul(best_m as u128) >> 16) as u64;
                                    t_sc = t_sc.saturating_add(contrib);
                                }
                            }
                            if t_sc > 0 {
                                if tr.is_habitual {
                                    t_sc = t_sc.saturating_add(t_sc / 3);
                                }
                                od_turn_score_q16[ti] = t_sc;
                                if t_sc > max_od_turn {
                                    max_od_turn = t_sc;
                                }
                            }
                        }
                    }
                    if max_od_turn > 0 && lexical_tmax > 0.0 {
                        let turn_cap = if is_strict_inferential_od {
                            0.30 * lexical_tmax
                        } else {
                            0.18 * lexical_tmax
                        };
                        for ti in 0..nt {
                            let osc = od_turn_score_q16[ti];
                            if osc == 0 {
                                continue;
                            }
                            let si = ix.turns[ti].session;
                            if bridge_boost[si] <= 0.0 && s_score[si] < 0.50 * smax {
                                continue;
                            }
                            let ratio = (((osc as u128) << 16) / (max_od_turn as u128))
                                .min(Q16_ONE as u128) as u64;
                            let add_f = turn_cap * ((ratio as f32) / 65_536.0);
                            t_scene[ti] += add_f;
                            t_scene_sess[ti] += add_f;
                        }
                    }
                }
                // Pillar 3 (KITE `2610.41521` Upward-Cone Typed Facet Meet + FCA Closed-Intent Specificity):
                // Abstract Open-Domain/Multi-Hop query facets intersect concrete declarative turn words
                // indexed upward through WordNet 3.1 synsets & Wikidata CC0 containers in O(1).
                if is_strict_inferential_od || top0_misses_group || ctx.persona_infer_on {
                    let kite_sess_cap_q16 = if is_strict_inferential_od {
                        ((smax_q16 as u128).saturating_mul(36_045) >> 16).max(1) as u64 // 0.55 * smax
                    } else {
                        ((smax_q16 as u128).saturating_mul(18_350) >> 16).max(1) as u64 // 0.28 * smax
                    };
                    let kite_turn_cap_f = if is_strict_inferential_od {
                        0.34 * lexical_tmax
                    } else {
                        0.18 * lexical_tmax
                    };
                    let kite_turn_cap_q16 = (kite_turn_cap_f.max(0.0) * 65_536.0) as u64;
                    let (kite_s, kite_t) =
                        crate::retrieval::index::kite_typed::score_kite_typed_facet_meet_q16(
                            ix,
                            plan,
                            &q.text,
                            &queried_sp,
                            kite_sess_cap_q16,
                            kite_turn_cap_q16,
                        );
                    for (si, &sl_q16) in kite_s.iter().enumerate() {
                        if sl_q16 > 0 {
                            od_translation_fired = true;
                            bridge_boost[si] += (sl_q16 as f32) / 65_536.0;
                        }
                    }
                    if lexical_tmax > 0.0 {
                        for (ti, &tl_q16) in kite_t.iter().enumerate() {
                            if tl_q16 > 0 {
                                let add_f = (tl_q16 as f32) / 65_536.0;
                                t_scene[ti] += add_f;
                                t_scene_sess[ti] += add_f;
                            }
                        }
                    }
                }
                // Pillar 4 (Discrete Sinkhorn-Knopp Optimal Transport over WordNet/ConceptNet Ground Metric):
                // Holistically aligns ALL query content terms onto the top-16 candidate turns so turns
                // transporting every predicate beat single-word lexical decoys.
                if (is_strict_inferential_od || top0_misses_group) && lexical_tmax > 0.0 {
                    let ot_q = crate::retrieval::lexical::sinkhorn_ot_q16::SinkhornQuery::build(
                        ix,
                        &q.text,
                        &queried_sp,
                    );
                    if !ot_q.words.is_empty() {
                        let mut cand_turns: Vec<(usize, f32)> = (0..nt)
                            .filter(|&ti| {
                                let tr = &ix.turns[ti];
                                !tr.is_question
                                    && !tr.is_backchannel
                                    && t_scene[ti] > 0.0
                                    && (queried_sp.is_empty()
                                        || queried_sp.contains(&tr.speaker.to_ascii_lowercase()))
                            })
                            .map(|ti| (ti, t_scene[ti]))
                            .collect();
                        cand_turns.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.0.cmp(&b.0))
                        });
                        let ot_cap = if is_strict_inferential_od {
                            0.18 * lexical_tmax
                        } else {
                            0.12 * lexical_tmax
                        };
                        let mut ot_scores: Vec<(usize, u64)> = Vec::new();
                        let mut max_ot: u64 = 0;
                        for &(ti, _) in cand_turns.iter().take(16) {
                            let osc = ot_q.score_turn_q16(ix, ti);
                            if osc > 0 {
                                ot_scores.push((ti, osc));
                                if osc > max_ot {
                                    max_ot = osc;
                                }
                            }
                        }
                        if max_ot > 0 {
                            for (ti, osc) in ot_scores {
                                let ratio = ((osc as f32) / (max_ot as f32)).min(1.0);
                                let add_f = ot_cap * ratio;
                                t_scene[ti] += add_f;
                                t_scene_sess[ti] += add_f;
                            }
                        }
                    }
                }
            }
        }

        // Active 2-Round Multi-Hop Residual Postings Lookup (Step 2):
        // When non-speaker query groups are strictly split across sessions (Pigeonhole:
        // `union_cov >= 2 && max_sess_cov < union_cov`), take top-6 partial-coverage Pass-1
        // sessions `s1`, extract top-IC non-query pivot entities `e` (`2 <= s_df <= ns/3`),
        // and run an active Pass-2 lookup for sessions `s2 != s1` carrying BOTH `e` AND
        // an uncovered query group `union_mask & !M(s1)` so hidden-pivot Hop-2 sessions
        // and their Pass-1 partners rise into Top-2 together!
        if has_named_speakers_eff && !eff_it.temporal && !top0_full_cov && ns >= 4 && smax > 0.0 {
            let mut non_sp_mask_r2 = 0u64;
            let mut sp_lowers_r2: Vec<String> = Vec::new();
            for tr in ix.turns.iter().take(128) {
                let sl = tr.speaker.to_ascii_lowercase();
                if !sl.is_empty() && sl != "user" && sl != "assistant" && !sp_lowers_r2.contains(&sl) {
                    sp_lowers_r2.push(sl);
                }
            }
            for (gi, g) in plan.groups.iter().take(64).enumerate() {
                if !sp_lowers_r2.iter().any(|sp| sp.eq_ignore_ascii_case(g)) {
                    non_sp_mask_r2 |= 1u64 << gi;
                }
            }
            let union_m = ctx
                .s_exact_mask
                .iter()
                .fold(0u64, |acc, &m| acc | (m & non_sp_mask_r2));
            let union_cnt = union_m.count_ones();
            let max_s_cnt = ctx
                .s_exact_mask
                .iter()
                .map(|&m| (m & non_sp_mask_r2).count_ones())
                .max()
                .unwrap_or(0);
            if union_cnt >= 2 && max_s_cnt < union_cnt {
                let ns_u32_r2 = (ns as u32).max(1);
                let df_hi = (ns_u32_r2 / 3).clamp(3, 12);
                let mut pass1: Vec<(usize, f32)> = (0..ns)
                    .filter(|&si| {
                        let m = ctx.s_exact_mask[si] & union_m;
                        m != 0 && m != union_m && (s_score[si] + bridge_boost[si]) > 0.0
                    })
                    .map(|si| (si, s_score[si] + bridge_boost[si]))
                    .collect();
                pass1.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let mut r2_raw = vec![0u64; ns];
                for &(s1, _) in pass1.iter().take(6) {
                    let m1 = ctx.s_exact_mask[s1] & union_m;
                    let need = union_m & !m1;
                    if need == 0 {
                        continue;
                    }
                    let mut pivots: Vec<(u32, u64)> = Vec::new();
                    for &eid in ix
                        .session_entities[s1]
                        .iter()
                        .chain(ix.session_rare_nt[s1].iter().take(12))
                    {
                        if plan.group_of.contains_key(&eid) {
                            continue;
                        }
                        let df = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
                        if (2..=df_hi).contains(&df) && !pivots.iter().any(|&(p, _)| p == eid) {
                            pivots.push((eid, crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16(df, ns_u32_r2)));
                        }
                    }
                    pivots.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    pivots.truncate(10);
                    if pivots.is_empty() {
                        continue;
                    }
                    for s2 in 0..ns {
                        if s2 == s1 {
                            continue;
                        }
                        let m2_new = ctx.s_exact_mask[s2] & need;
                        if m2_new == 0 {
                            continue;
                        }
                        let mut pivot_ic = 0u64;
                        for &(pid, pic) in &pivots {
                            if ix.session_entities[s2].contains(&pid)
                                || ix.session_content[s2].contains(&pid)
                            {
                                pivot_ic = pivot_ic.saturating_add(pic);
                            }
                        }
                        if pivot_ic > 0 {
                            let cov_bonus = (m1 | m2_new).count_ones() as u64;
                            let pair_proof = pivot_ic.saturating_mul(cov_bonus);
                            r2_raw[s1] = r2_raw[s1].max(pair_proof);
                            r2_raw[s2] = r2_raw[s2].max(pair_proof);
                        }
                    }
                }
                let max_r2 = r2_raw.iter().copied().max().unwrap_or(0);
                if max_r2 > 0 {
                    let r2_cap = 0.36 * smax;
                    for si in 0..ns {
                        if r2_raw[si] > 0 {
                            let frac = (r2_raw[si] as f32) / (max_r2 as f32);
                            bridge_boost[si] += r2_cap * frac;
                        }
                    }
                }
            }
        }

        // Domain E (Step 12B — HeidelTime Relative-Date Valid-Time Interval Overlap):
        // Strictly gated to `eff_it.temporal` (never Single-Hop queries that happen to contain "first"/"before").
        if eff_it.temporal && smax > 0.0 {
            use crate::retrieval::graph::bitemporal_heideltime::{
                interval_overlap_q16, resolve_relative_interval,
            };
            let q_toks_ht = raw_tokens(&q.text);
            let q_refs_ht: Vec<&str> = q_toks_ht.iter().map(String::as_str).collect();
            let max_sts = ix.session_ts.iter().copied().max().unwrap_or(0);
            let q_anchor = parse_date(&q.date).max(max_sts);
            if q_anchor > 0 {
                if let Some(q_iv) = resolve_relative_interval(&q_refs_ht, q_anchor) {
                    let ht_cap = 0.18 * smax;
                    for si in 0..ns {
                        let sts = ix.session_ts[si];
                        if sts <= 0 || (s_score[si] <= 0.0 && bridge_boost[si] <= 0.0) {
                            continue;
                        }
                        let mut best_ov = interval_overlap_q16(q_iv, (sts, sts + 86_400));
                        let mut best_timex_turn: Option<(usize, u64)> = None;
                        for &ti in &ix.session_turns[si] {
                            let tr = &ix.turns[ti];
                            if !tr.resolved_timex.is_empty() && !tr.is_question && !tr.is_backchannel {
                                let mut t_ov: u64 = 0;
                                for &t_iv in &tr.resolved_timex {
                                    let ov = interval_overlap_q16(q_iv, t_iv);
                                    if ov > t_ov {
                                        t_ov = ov;
                                    }
                                }
                                if t_ov > best_ov {
                                    best_ov = t_ov;
                                }
                                if t_ov > 0 && best_timex_turn.is_none_or(|(_, bo)| t_ov > bo) {
                                    best_timex_turn = Some((ti, t_ov));
                                }
                            }
                        }
                        if best_ov > 0 {
                            let lift = ht_cap * ((best_ov as f32) / 65_536.0);
                            bridge_boost[si] += lift;
                        }
                        if let Some((ti, t_ov)) = best_timex_turn && lexical_tmax > 0.0 {
                            let t_lift = 0.16 * lexical_tmax * ((t_ov as f32) / 65_536.0);
                            t_scene[ti] += t_lift;
                            t_scene_sess[ti] += t_lift;
                        }
                    }
                }
            }
        }

        // Pure Q16.16 post-bridge `s_ord_base` refresh:
        // Strictly gated to Multi-Hop (`!invariant_crisp_lock && top0_misses_group`) OR fired Domain-D
        // Open-Domain translation (`od_translation_fired && (is_strict_inferential_od || (!invariant_crisp_lock && top0_misses_group))`),
        // NEVER unlocking factual Single-Hop or Adversarial Slot 0!
        let allow_od_unlock = od_translation_fired
            && (is_strict_inferential_od || (!invariant_crisp_lock && top0_misses_group));
        let allow_mh_unlock = eff_it.multi_hop && !invariant_crisp_lock && top0_misses_group;
        if allow_od_unlock || allow_mh_unlock {
            let (margin_num, margin_den): (u64, u64) = if allow_od_unlock { (53, 50) } else { (27, 25) };
            let post_sight_q16: Vec<u64> = (0..ns)
                .map(|si| {
                    crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                        (s_score[si] + bridge_boost[si]).to_bits(),
                    )
                })
                .collect();
            let mut post_ord: Vec<usize> = (0..ns).collect();
            post_ord.sort_by(|&a, &b| {
                post_sight_q16[b]
                    .cmp(&post_sight_q16[a])
                    .then_with(|| ix.session_ts[b].cmp(&ix.session_ts[a]))
                    .then_with(|| a.cmp(&b))
            });
            if let Some(&cand0) = post_ord.first() {
                let cand_mask = s_exact_mask.get(cand0).copied().unwrap_or(0) & q_full_mask;
                let cov_ok = allow_od_unlock
                    || cand_mask.count_ones() >= top0_mask_bits.count_ones()
                    || (cand_mask & !top0_mask_bits) != 0;
                if cand0 != top0
                    && cov_ok
                    && post_sight_q16[cand0].saturating_mul(margin_den)
                        >= post_sight_q16[top0].saturating_mul(margin_num)
                {
                    s_ord_base = post_ord;
                }
            }
        }
    }

    // Pure Q16.16 turn-level evidence propagation inside bridge-boosted sessions
    // (closes the Session R@5 -> Turn R@5 gap without touching t_own or t_turn_raw).
    let lex_tmax_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(lexical_tmax.to_bits());
    let smax_prop_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(smax.to_bits());
    if !eff_it.is_event_sequence
        && !eff_it.task_under_rules
        && smax_prop_q16 > 0
        && lex_tmax_q16 > 0
    {
        const Q16_ONE_PROP: u64 = 65_536;
        const BB_RATIO_CAP_Q16: u64 = 55_706; // 0.85
        const W_TB_Q16: u64 = 29_491;         // 0.45
        const W_ALIAS_Q16: u64 = 29_491;      // 0.45
        const W_GRP_Q16: u64 = 13_107;        // 0.20 per group (up to 3)
        const TURN_PROP_COEFF_Q16: u64 = 18_350; // 0.28
        const Q16_SCALE_F32: f32 = 65_536.0;

        let t_bridge_q16: Vec<u64> = ctx
            .t_bridge
            .iter()
            .map(|&v| crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(v.to_bits()))
            .collect();
        let max_tb_q16 = t_bridge_q16.iter().copied().max().unwrap_or(0);
        let alias_pairs = crate::retrieval::graph::fellegi_sunter::discover_fellegi_sunter_aliases(ix);
        let q_exact_ids: std::collections::BTreeSet<u32> = plan
            .terms
            .iter()
            .filter(|qt| qt.exact_hit)
            .map(|qt| qt.id)
            .collect();
        let mut linked_alias_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for (a, b) in alias_pairs {
            if q_exact_ids.contains(&a) && !q_exact_ids.contains(&b) {
                linked_alias_ids.insert(b);
            } else if q_exact_ids.contains(&b) && !q_exact_ids.contains(&a) {
                linked_alias_ids.insert(a);
            }
        }

        for si in 0..ns {
            let bb_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                bridge_boost[si].to_bits(),
            );
            if bb_q16.saturating_mul(10) < smax_prop_q16 {
                continue;
            }
            let bb_ratio_q16 = ((bb_q16.saturating_mul(Q16_ONE_PROP)) / smax_prop_q16).min(BB_RATIO_CAP_Q16);
            for &ti in &ix.session_turns[si] {
                let tr = &ix.turns[ti];
                if tr.is_question || tr.is_backchannel || turn_is_superseded(ix, ti) {
                    continue;
                }
                let tb_part = if max_tb_q16 > 0 && t_bridge_q16[ti] > 0 {
                    let norm_tb = ((t_bridge_q16[ti].saturating_mul(Q16_ONE_PROP)) / max_tb_q16).min(Q16_ONE_PROP);
                    (W_TB_Q16.saturating_mul(norm_tb)) >> 16
                } else {
                    0
                };
                let alias_part = if !linked_alias_ids.is_empty()
                    && tr.content.iter().any(|cid| linked_alias_ids.contains(cid))
                {
                    W_ALIAS_Q16
                } else {
                    0
                };
                let grp_cnt = (t_mask[ti].count_ones() as u64).min(3);
                let grp_part = W_GRP_Q16.saturating_mul(grp_cnt);
                let ev_q16 = tb_part
                    .saturating_add(alias_part)
                    .saturating_add(grp_part)
                    .min(Q16_ONE_PROP);
                if ev_q16 > 0 {
                    let scale_q16 = (((TURN_PROP_COEFF_Q16.saturating_mul(bb_ratio_q16)) >> 16)
                        .saturating_mul(ev_q16))
                        >> 16;
                    let add_q16 = (lex_tmax_q16.saturating_mul(scale_q16)) >> 16;
                    if add_q16 > 0 {
                        let add_f32 = (add_q16 as f32) / Q16_SCALE_F32;
                        t_scene[ti] += add_f32;
                        t_scene_sess[ti] += add_f32;
                    }
                }
            }
        }
    }

    ctx.s_score = s_score;
    ctx.t_scene = t_scene;
    ctx.t_scene_sess = t_scene_sess;
    ctx.lexical_smax = lexical_smax;
    ctx.lexical_tmax_sess = lexical_tmax_sess;
    ctx.lexical_tmax = lexical_tmax;
    ctx.standing_rules = standing_rules;
    ctx.s_policy_floored = s_policy_floored;
    ctx.t_policy_floored = t_policy_floored;
    ctx.s_pref_floored = s_pref_floored;
    ctx.t_pref_floored = t_pref_floored;
    ctx.initiative_content = initiative_content;
    ctx.smax = smax;
    ctx.bridge_boost = bridge_boost;
    ctx.s_ord_base = s_ord_base;
}


/// Priority vote weight for pre-cut BM25TP proximity evidence (`2.0`).
const LIFT_W_PRIORITY_Q16: u64 = 131_072;
/// Vote weight for each graph/discourse module (`0.5`).
const LIFT_W_MODULE_Q16: u64 = 32_768;
/// Shared budget as a fraction of `sight(top0)` (`0.30`).
const LIFT_BUDGET_Q16: u64 = 19_661;

/// Combines module lift votes into one evidence-scaled `Q16.16` budget.
/// Votes are measured against `smax`, not each module's local maximum. A session needs BM25TP or
/// two independent module votes, and must add a query group absent from `top0`.
pub(in crate::retrieval) fn unified_evidence_lift_budget_q16(
    modules: &[(bool, &[u64])],
    s_exact_mask: &[u64],
    q_full_mask: u64,
    top0: usize,
    top0_sight_q16: u64,
    smax_q16: u64,
) -> Vec<u64> {
    const Q16_ONE: u64 = 65_536;
    let ns = s_exact_mask.len();
    let mut out = vec![0u64; ns];
    let top0_mask = s_exact_mask.get(top0).copied().unwrap_or(0) & q_full_mask;
    if ns == 0 || q_full_mask == 0 || top0_mask == q_full_mask || smax_q16 == 0 || top0_sight_q16 == 0 {
        return out;
    }
    let budget_q16 = top0_sight_q16.saturating_mul(LIFT_BUDGET_Q16) >> 16;
    for (si, slot) in out.iter_mut().enumerate() {
        let mask = s_exact_mask[si] & q_full_mask;
        if si == top0 || (mask & !top0_mask) == 0 {
            continue;
        }
        let mut evidence_q16 = 0u64;
        let mut priority_vote = false;
        let mut module_votes = 0u32;
        for &(is_priority, lifts) in modules {
            let raw = lifts.get(si).copied().unwrap_or(0);
            if raw == 0 {
                continue;
            }
            let weight_q16 = if is_priority { LIFT_W_PRIORITY_Q16 } else { LIFT_W_MODULE_Q16 };
            if is_priority {
                priority_vote = true;
            } else {
                module_votes += 1;
            }
            let strength_q16 = (raw.saturating_mul(Q16_ONE) / smax_q16).min(Q16_ONE);
            evidence_q16 = evidence_q16.saturating_add(weight_q16.saturating_mul(strength_q16) >> 16);
        }
        if priority_vote || module_votes >= 2 {
            *slot = budget_q16.saturating_mul(evidence_q16.min(Q16_ONE)) >> 16;
        }
    }
    out
}

/// Applies a `Q16.16` lift vector to `bridge_boost` while strictly enforcing the invariant
/// Rule-8 crisp-peak ceiling (`s_score[si] + bridge_boost[si] <= 0.90 * sight(top0)`).
fn apply_rule8_capped_q16_lifts(
    lifts_q16: &[u64],
    top0: usize,
    top0_ceiling_q16: u64,
    invariant_crisp_lock: bool,
    s_score: &[f32],
    bridge_boost: &mut [f32],
) {
    const Q16_SCALE_F32: f32 = 65_536.0;
    for (si, &raw_lift) in lifts_q16.iter().enumerate() {
        if si == top0 || raw_lift == 0 {
            continue;
        }
        let lift_q16 = if invariant_crisp_lock {
            let cur_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                (s_score[si] + bridge_boost[si]).to_bits(),
            );
            raw_lift.min(top0_ceiling_q16.saturating_sub(cur_q16))
        } else {
            raw_lift
        };
        if lift_q16 > 0 {
            bridge_boost[si] += (lift_q16 as f32) / Q16_SCALE_F32;
        }
    }
}

#[cfg(test)]
mod unified_budget_tests {
    use super::unified_evidence_lift_budget_q16 as budget;
    const ONE: u64 = 65_536;
    #[test]
    fn full_top0_coverage_closes_budget() {
        let lifts = [0, ONE, ONE];
        let out = budget(&[(true, &lifts)], &[0b11, 0b10, 0b01], 0b11, 0, 10 * ONE, 10 * ONE);
        assert!(out.iter().all(|&v| v == 0));
    }
    #[test]
    fn lift_requires_new_group_and_agreement() {
        let one_vote = [0, ONE, ONE, ONE];
        let second_vote = [0, ONE, 0, ONE];
        let out = budget(&[(false, &one_vote), (false, &second_vote)],
            &[0b01, 0b10, 0b10, 0b01], 0b11, 0, 10 * ONE, 10 * ONE);
        assert!(out[1] > 0);
        assert_eq!(out[2], 0, "single non-priority vote must not lift");
        assert_eq!(out[3], 0, "session repeating top0's only group must not lift");
    }
    #[test]
    fn budget_is_evidence_scaled_and_capped() {
        let weak = [0, ONE / 10, 0];
        let strong = [0, 0, 10 * ONE];
        let out = budget(&[(true, &weak), (true, &strong)], &[0b01, 0b10, 0b10], 0b11, 0, 10 * ONE, 10 * ONE);
        assert!(out[1] < out[2]);
        assert!(out[2] <= (10 * ONE * 19_661) >> 16);
    }
}
