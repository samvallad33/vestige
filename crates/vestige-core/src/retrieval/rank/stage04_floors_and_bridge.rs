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
        let pre_tmax = t_scene.iter().cloned().fold(0.0f32, f32::max);
        let pre_smax = s_score.iter().cloned().fold(0.0f32, f32::max);
        if pre_tmax > 0.0 && pre_smax > 0.0 {
            for si in 0..ns {
                let best_decl = ix.session_turns[si]
                    .iter()
                    .copied()
                    .filter(|&ti| !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel)
                    .map(|ti| t_scene[ti])
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
    let multi_evidence_intent = it.multi_hop
        || it.sniper_how_many
        || it.temporal
        || eff_it.is_event_sequence;
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
    let (s_pref_floored, t_pref_floored) = if pref_on && !multi_evidence_intent {
        (
            raise_floor(&mut s_score, lexical_smax, 0.50, |si| {
                ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_personal_constraint)
            }),
            raise_floor(&mut t_scene, lexical_tmax, 0.50, |ti| ix.turns[ti].is_personal_constraint),
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
    if pref_on && !multi_evidence_intent {
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.50, |ti| ix.turns[ti].is_personal_constraint);
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
    let s_ord_base = order(&s_score, &ix.session_ts);
    if p.enable_ppr && ns > 5 && smax > 0.0 {
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
                // 2. Structural contradiction / state-update supersession (`!it.current_state` pulls the earlier state
                //    when the earlier session also has lexical grounding or user attribution, without displacing QA2 header bridges)
                if !it.current_state {
                    for &prev_si in &ix.session_supersedes_back[anc] {
                        if prev_si != top0 && (s_score[prev_si] >= 0.12 * smax || it.user_attribution) {
                            let cand = smax * 0.44 * anc_w;
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

    if has_named_speakers_eff
        && !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && ns > 5
        && smax > 0.0
    {
        let pre_sight_q16: Vec<u64> = (0..ns)
            .map(|si| {
                crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                    (s_score[si] + bridge_boost[si]).to_bits(),
                )
            })
            .collect();
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
        let top0_full_cov = q_full_mask != 0
            && (s_exact_mask.get(top0).copied().unwrap_or(0) & q_full_mask) == q_full_mask;
        let invariant_crisp_lock =
            top0_full_cov || top0_lex.saturating_mul(3) >= top1_lex.saturating_mul(4);
        let multi_or_partial = eff_it.multi_hop || !top0_full_cov;

        associative_multihop_bridge_q16::apply_multihop_open_domain_session_lift(
            ix,
            plan,
            it,
            eff_it,
            &s_score,
            s_exact_mask,
            &s_ord_base,
            smax,
            &mut bridge_boost,
        );
        let top0_ceiling_q16 = (crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
            (s_score[top0] + bridge_boost[top0]).to_bits(),
        )
        .saturating_mul(9))
            / 10;

        let ppr_lifts = crate::retrieval::graph::ppr_rwr::compute_bipartite_ppr_lifts_q16(
            ix,
            plan,
            multi_or_partial,
            &pre_sight_q16,
            &s_ord_base,
            smax_q16,
        );
        apply_rule8_capped_q16_lifts(
            &ppr_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
        let tlm_lifts =
            crate::retrieval::lexical::translation_lm::compute_translation_lm_lifts_q16(
                ix,
                plan,
                multi_or_partial,
                s_exact_mask,
                top0,
                smax_q16,
            );
        apply_rule8_capped_q16_lifts(
            &tlm_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
        let tp_lifts = crate::retrieval::lexical::bm25tp::compute_session_bm25tp_lifts_q16(
            ix,
            plan,
            s_exact_mask,
            top0,
            smax_q16,
        );
        apply_rule8_capped_q16_lifts(
            &tp_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
        let swanson_lifts = crate::retrieval::graph::swanson_abc::compute_swanson_abc_lifts_q16(
            ix,
            plan,
            multi_or_partial,
            &pre_sight_q16,
            &s_ord_base,
            smax_q16,
        );
        apply_rule8_capped_q16_lifts(
            &swanson_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
        let fs_lifts =
            crate::retrieval::graph::fellegi_sunter::compute_fellegi_sunter_lifts_q16(
                ix,
                plan,
                top0,
                smax_q16,
            );
        apply_rule8_capped_q16_lifts(
            &fs_lifts,
            top0,
            top0_ceiling_q16,
            invariant_crisp_lock,
            &s_score,
            &mut bridge_boost,
        );
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

/// Domain-agnostic `Q16.16` associative session lift for Multi-Hop and Open-Domain recall.
///
/// Combines four strictly domain-agnostic signals from the index:
/// 1. Corpus-IC weighted `ix.ppmi_neighbors` co-occurrence bridge (2x weight on uncovered query groups),
/// 2. 1-hop WordNet conceptual bridge via `ix.bridge_vocab` + `ix.turn_bridge_postings`,
/// 3. 2-hop damped backward walk over `session_rsb_back`, `session_supersedes_back`, and `session_episode_prev`,
/// 4. Rule-8 crisp-peak ceiling protecting Single-Hop and Adversarial Rank-1 (`top0`) from dethronement.
mod associative_multihop_bridge_q16 {
    use super::*;
    use crate::retrieval::fusion::kemeny_mrf::{ieee754_bits_to_q16, log2_u64_q16};
    use std::collections::BTreeSet;

    /// Unit `1.0` in unsigned `Q16.16` fixed-point math (`2^16`).
    const Q16_ONE: u64 = 65_536;
    /// Fractional shift width (`16`).
    const Q16_SHIFT: u32 = 16;
    /// Maximum anchor sessions seeded for 2-hop RSB/supersession/episode pull (`3`).
    const MAX_ANCHOR_SESSIONS: usize = 3;
    /// Maximum PPMI neighbors inspected per query term (`12`).
    const MAX_PPMI_NEIGHBORS_PER_TERM: usize = 12;
    /// Maximum WordNet lemmas inspected per query group (`16`).
    const MAX_WN_LEMMAS_PER_GROUP: usize = 16;
    /// Fusion weight for Corpus-IC PPMI bridge (`3/8` = `24_576` in `Q16.16`).
    const WEIGHT_PPMI_Q16: u64 = 24_576;
    /// Fusion weight for 1-hop WordNet conceptual bridge (`3/8` = `24_576` in `Q16.16`).
    const WEIGHT_WORDNET_Q16: u64 = 24_576;
    /// Fusion weight for 2-hop RSB / supersession / episode bridge (`2/8` = `16_384` in `Q16.16`).
    const WEIGHT_RSB_2HOP_Q16: u64 = 16_384;
    /// Lift coefficient on multi-hop or partial-coverage queries (`0.52` in `Q16.16`).
    const LIFT_COEFF_MULTI_Q16: u64 = 34_079;
    /// Lift coefficient on full-coverage queries (`0.28` in `Q16.16`).
    const LIFT_COEFF_BASE_Q16: u64 = 18_350;
    /// Rule-8 crisp-peak protection ceiling (`9/10` of incumbent `top0` total sight).
    const RULE8_CEILING_NUM: u64 = 9;
    const RULE8_CEILING_DEN: u64 = 10;
    /// Crisp-peak Top-1 over Top-2 dominance ratio (`4:3`, matching NQC guard).
    const CRISP_PEAK_NUM: u64 = 4;
    const CRISP_PEAK_DEN: u64 = 3;

    /// Pure integer `Q16.16` fusion of normalized `(ppmi, wordnet, rsb_2hop)` scores with Rule-8 cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fuse_associative_lift_q16(
        ppmi_raw: &[u64],
        wn_raw: &[u64],
        rsb_raw: &[u64],
        smax_q16: u64,
        coeff_q16: u64,
        top0: usize,
        top0_sight_q16: u64,
        crisp_peak_lock: bool,
        base_sight_q16: &[u64],
    ) -> Vec<u64> {
        let ns = ppmi_raw.len();
        let mut out = vec![0u64; ns];
        if ns == 0 || smax_q16 == 0 {
            return out;
        }
        let max_ppmi = ppmi_raw.iter().copied().max().unwrap_or(0);
        let max_wn = wn_raw.iter().copied().max().unwrap_or(0);
        let max_rsb = rsb_raw.iter().copied().max().unwrap_or(0);
        if max_ppmi == 0 && max_wn == 0 && max_rsb == 0 {
            return out;
        }
        let ceiling_q16 = (top0_sight_q16.saturating_mul(RULE8_CEILING_NUM)) / RULE8_CEILING_DEN;

        for si in 0..ns {
            if si == top0 {
                continue;
            }
            let n_ppmi = if max_ppmi > 0 {
                (ppmi_raw[si].saturating_mul(Q16_ONE)) / max_ppmi
            } else {
                0
            };
            let n_wn = if max_wn > 0 {
                (wn_raw[si].saturating_mul(Q16_ONE)) / max_wn
            } else {
                0
            };
            let n_rsb = if max_rsb > 0 {
                (rsb_raw[si].saturating_mul(Q16_ONE)) / max_rsb
            } else {
                0
            };
            let fused_norm_q16 = ((n_ppmi.saturating_mul(WEIGHT_PPMI_Q16)) >> Q16_SHIFT)
                .saturating_add((n_wn.saturating_mul(WEIGHT_WORDNET_Q16)) >> Q16_SHIFT)
                .saturating_add((n_rsb.saturating_mul(WEIGHT_RSB_2HOP_Q16)) >> Q16_SHIFT);
            if fused_norm_q16 == 0 {
                continue;
            }
            let mut lift_q16 = (((smax_q16.saturating_mul(coeff_q16)) >> Q16_SHIFT)
                .saturating_mul(fused_norm_q16))
                >> Q16_SHIFT;

            // Rule-8 crisp-peak ceiling: never allow associative lift to push `si` past `0.9 * sight(top0)`.
            if crisp_peak_lock {
                let allowed = ceiling_q16.saturating_sub(base_sight_q16[si]);
                if lift_q16 > allowed {
                    lift_q16 = allowed;
                }
            }
            out[si] = lift_q16;
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_multihop_open_domain_session_lift(
        ix: &Index,
        plan: &QueryPlan,
        it: &QueryIntent,
        eff_it: &QueryIntent,
        s_score: &[f32],
        s_exact_mask: &[u64],
        s_ord_base: &[usize],
        smax: f32,
        bridge_boost: &mut [f32],
    ) {
        let ns = ix.session_turns.len();
        let Some(&top0) = s_ord_base.first() else {
            return;
        };
        if ns <= 5 || top0 >= ns {
            return;
        }

        let total_groups = plan.groups.len().min(64);
        let q_mask = if total_groups == 64 {
            u64::MAX
        } else if total_groups == 0 {
            0
        } else {
            (1u64 << total_groups) - 1
        };
        let top0_mask = s_exact_mask.get(top0).copied().unwrap_or(0) & q_mask;
        let top0_full_cov = q_mask != 0 && top0_mask == q_mask;

        let base_sight_q16: Vec<u64> = (0..ns)
            .map(|si| ieee754_bits_to_q16((s_score[si] + bridge_boost[si]).to_bits()))
            .collect();
        let top0_sight_q16 = base_sight_q16[top0];
        let top1_sight_q16 = s_ord_base
            .get(1)
            .and_then(|&s1| base_sight_q16.get(s1).copied())
            .unwrap_or(0);
        let crisp_peak_lock = top0_full_cov
            || top0_sight_q16.saturating_mul(CRISP_PEAK_DEN)
                >= top1_sight_q16.saturating_mul(CRISP_PEAK_NUM);

        let max_partner_df = ((ns / 3).max(2)) as u32;
        let mut ppmi_raw = vec![0u64; ns];
        let mut wn_raw = vec![0u64; ns];
        let mut rsb_raw = vec![0u64; ns];

        // 1. Corpus-IC weighted PPMI neighbor bridge (2x weight on query groups missing from `top0`).
        let mut seen_qids: BTreeSet<u32> = BTreeSet::new();
        for qt in &plan.terms {
            let Some(g) = qt.group else {
                continue;
            };
            if g >= 64 || !seen_qids.insert(qt.id) {
                continue;
            }
            let Some(neighbors) = ix.ppmi_neighbors.get(&qt.id) else {
                continue;
            };
            let q_df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(1).max(1) as u64;
            let q_ic_q16 = log2_u64_q16(((ns as u64) + 1) / q_df.min(ns as u64)).max(Q16_ONE >> 2);
            let uncov_mult = if (top0_mask & (1u64 << g)) == 0 { 2u64 } else { 1u64 };

            for &(partner_id, ppmi_w) in neighbors.iter().take(MAX_PPMI_NEIGHBORS_PER_TERM) {
                let p_df = ix.s_stats.df.get(&partner_id).copied().unwrap_or(0);
                if p_df == 0 || p_df > max_partner_df {
                    continue;
                }
                let pw_q16 = ieee754_bits_to_q16(ppmi_w.to_bits());
                if pw_q16 == 0 {
                    continue;
                }
                let contrib_q16 = ((pw_q16.saturating_mul(q_ic_q16)) >> Q16_SHIFT)
                    .saturating_mul(uncov_mult);
                if let Some(postings) = ix.s_postings.get(partner_id as usize) {
                    for &si_u32 in postings {
                        let si = si_u32 as usize;
                        if si < ns && si != top0 {
                            ppmi_raw[si] = ppmi_raw[si].saturating_add(contrib_q16);
                        }
                    }
                }
            }
        }

        // 2. 1-hop WordNet conceptual bridge via `ix.bridge_vocab` + `ix.turn_bridge_postings`.
        for (g, g_word) in plan.groups.iter().take(64).enumerate() {
            let uncov_mult = if (top0_mask & (1u64 << g)) == 0 { 2u64 } else { 1u64 };
            let mut hit_sessions: BTreeSet<usize> = BTreeSet::new();
            for lemma in crate::retrieval::graph::bridge::wordnet_lemmas_with_morph(g_word)
                .into_iter()
                .take(MAX_WN_LEMMAS_PER_GROUP)
            {
                let mut cand_bids = [None; 2];
                cand_bids[0] = ix.bridge_vocab.get(lemma);
                let st_l = stem_term(&stem(lemma));
                cand_bids[1] = ix.bridge_vocab.get(&st_l);
                for bid in cand_bids.into_iter().flatten() {
                    if let Some(tlist) = ix.turn_bridge_postings.get(bid as usize) {
                        for &ti_u32 in tlist {
                            let ti = ti_u32 as usize;
                            if let Some(t) = ix.turns.get(ti) {
                                let si = t.session;
                                if si < ns
                                    && si != top0
                                    && (s_exact_mask[si] & (1u64 << g)) == 0
                                {
                                    hit_sessions.insert(si);
                                }
                            }
                        }
                    }
                }
            }
            let hit_df = hit_sessions.len() as u32;
            if hit_df > 0 && hit_df <= max_partner_df {
                let wn_ic_q16 = log2_u64_q16(((ns as u64) + 1) / (hit_df as u64))
                    .max(Q16_ONE >> 2)
                    .saturating_mul(uncov_mult);
                for si in hit_sessions {
                    wn_raw[si] = wn_raw[si].saturating_add(wn_ic_q16);
                }
            }
        }

        // 3. 2-hop damped backward RSB + supersession + episode pull from top anchors.
        let anchors: Vec<usize> = s_ord_base
            .iter()
            .copied()
            .filter(|&si| si < ns && base_sight_q16[si].saturating_mul(5) >= top0_sight_q16.saturating_mul(2))
            .take(MAX_ANCHOR_SESSIONS)
            .collect();
        for &anc in &anchors {
            let mut hop1: BTreeSet<usize> = BTreeSet::new();
            for &(deg, prev_si) in &ix.session_rsb_back[anc] {
                if prev_si < ns && prev_si != top0 {
                    let w = Q16_ONE / (deg.max(2) as u64);
                    rsb_raw[prev_si] = rsb_raw[prev_si].saturating_add(w);
                    hop1.insert(prev_si);
                }
            }
            if !it.current_state {
                for &prev_si in &ix.session_supersedes_back[anc] {
                    if prev_si < ns && prev_si != top0 {
                        rsb_raw[prev_si] = rsb_raw[prev_si].saturating_add(Q16_ONE >> 1);
                        hop1.insert(prev_si);
                    }
                }
            }
            if let Some(prev_ep) = ix.session_episode_prev[anc] {
                if prev_ep < ns && prev_ep != top0 {
                    rsb_raw[prev_ep] = rsb_raw[prev_ep].saturating_add(Q16_ONE >> 1);
                    hop1.insert(prev_ep);
                }
            }
            // Hop 2 (damped 0.5x in Q16.16)
            for mid in hop1 {
                for &(deg, prev2) in &ix.session_rsb_back[mid] {
                    if prev2 < ns && prev2 != top0 && prev2 != anc {
                        let w = (Q16_ONE / (deg.max(2) as u64)) >> 1;
                        rsb_raw[prev2] = rsb_raw[prev2].saturating_add(w);
                    }
                }
                if !it.current_state {
                    for &prev2 in &ix.session_supersedes_back[mid] {
                        if prev2 < ns && prev2 != top0 && prev2 != anc {
                            rsb_raw[prev2] = rsb_raw[prev2].saturating_add(Q16_ONE >> 2);
                        }
                    }
                }
                if let Some(prev2) = ix.session_episode_prev[mid] {
                    if prev2 < ns && prev2 != top0 && prev2 != anc {
                        rsb_raw[prev2] = rsb_raw[prev2].saturating_add(Q16_ONE >> 2);
                    }
                }
            }
        }

        let smax_q16 = ieee754_bits_to_q16(smax.to_bits());
        let coeff_q16 = if eff_it.multi_hop || !top0_full_cov {
            LIFT_COEFF_MULTI_Q16
        } else {
            LIFT_COEFF_BASE_Q16
        };
        let lifts_q16 = fuse_associative_lift_q16(
            &ppmi_raw,
            &wn_raw,
            &rsb_raw,
            smax_q16,
            coeff_q16,
            top0,
            top0_sight_q16,
            crisp_peak_lock,
            &base_sight_q16,
        );
        const Q16_SCALE_F32: f32 = 65_536.0;
        for (si, &lift_q16) in lifts_q16.iter().enumerate() {
            if lift_q16 > 0 {
                let lift_f32 = (lift_q16 as f32) / Q16_SCALE_F32;
                bridge_boost[si] += lift_f32;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_associative_lift_promotes_multihop_and_respects_rule8_ceiling() {
            // top0 = 0 has sight 10.0; session 1 is a crisp runner-up at 6.0;
            // session 2 sits low at 2.0 but carries strong PPMI + WordNet + RSB support.
            let smax_q16 = 10 * Q16_ONE;
            let base = vec![10 * Q16_ONE, 6 * Q16_ONE, 2 * Q16_ONE, 2 * Q16_ONE];
            let ppmi = vec![0, 0, 2 * Q16_ONE, 0];
            let wn = vec![0, 0, 2 * Q16_ONE, 0];
            let rsb = vec![0, 0, Q16_ONE, 0];

            let lifts = fuse_associative_lift_q16(
                &ppmi,
                &wn,
                &rsb,
                smax_q16,
                LIFT_COEFF_MULTI_Q16,
                0,
                base[0],
                true, // Rule 8 crisp-peak lock active
                &base,
            );
            // Session 2 rises cleanly above session 1 (2.0 + 5.2 = 7.2 > 6.0)
            assert!(base[2] + lifts[2] > base[1]);
            // Rule 8 lock: session 2 stays strictly below 0.90 * sight(top0) = 9.0
            assert!(base[2] + lifts[2] <= (base[0] * RULE8_CEILING_NUM) / RULE8_CEILING_DEN);
            // Unrelated session 3 receives zero lift
            assert_eq!(lifts[3], 0);
        }
    }
}
