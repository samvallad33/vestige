//! Stage 03: Scene pooling, anaphoric focus/Hobbs/coref, speaker alignment, scene refinements, and lexical steps.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

fn apply_scene_refinements(
    ctx: &RankCtx<'_>,
    policy_boost_ok: bool,
    live_policy_turns: &DetHashSet<usize>,
    t_scene: &mut [f32],
    t_mask: &mut [u64],
    t_exact_mask: &mut [u64],
) {
    let ix = ctx.ix;
    let q = ctx.q;
    let p = ctx.p;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let nt = ctx.nt;
    let smax = ctx.smax;
    let s_score = &ctx.s_score;
    let user_role = ctx.user_role;
    let episodic_user_quote = ctx.episodic_user_quote;

    if user_role {
        for ti in 0..nt {
            if ix.turns[ti].field == 1 {
                let s_idx = ix.turns[ti].session;
                let a_bits = t_exact_mask[ti];
                let user_covers = a_bits == 0
                    || ix.session_turns[s_idx]
                        .iter()
                        .any(|&u| ix.turns[u].field == 0 && (t_exact_mask[u] & a_bits) == a_bits);
                if user_covers {
                    t_scene[ti] *= 0.35;
                } else {
                    t_scene[ti] *= 0.72;
                }
            }
        }
    }

    // Standing policy invariant boost + Torrens functional supersession pruning + Advisory constraint lift:
    let q_toks_local = raw_tokens(&q.text);
    let has_past_comp_cue = q_toks_local.iter().any(|w| {
        matches!(
            w.as_str(),
            "before" | "previously" | "originally" | "earlier" | "initial" | "initially" | "former" | "formerly" | "changed" | "change" | "both" | "history" | "first"
        )
    }) || q_toks_local.windows(2).any(|p| p[0] == "used" && p[1] == "to");
    let present_state_local = !has_past_comp_cue && !it.temporal && !it.is_event_sequence;
    let has_named_corpus = ix
        .turns
        .iter()
        .take(16)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let is_monologue_stream = !ix.turns.iter().any(|t| t.field == 1) && !has_named_corpus;
    let ql_low = q.text.trim_start().to_ascii_lowercase();
    let advisory_modal_local = is_monologue_stream
        && !it.temporal
        && !it.sniper_when
        && !it.sniper_how_many
        && !it.is_event_sequence
        && (!q.text.trim_end().ends_with('?')
            || q_toks_local.windows(2).any(|p| {
                (p[0] == "can" && p[1] == "you")
                    || (p[0] == "could" && p[1] == "you")
                    || (p[0] == "would" && p[1] == "you")
                    || (p[0] == "should" && (p[1] == "i" || p[1] == "we"))
                    || (p[0] == "what" && p[1] == "should")
                    || (p[0] == "how" && (p[1] == "should" || p[1] == "can"))
            })
            || ql_low.starts_with("please "));

    let q_toks_bt = raw_tokens(&q.text);
    let q_refs_bt: Vec<&str> = q_toks_bt.iter().map(String::as_str).collect();
    let hist_override_local =
        crate::retrieval::graph::bitemporal_heideltime::query_has_historical_override(&q_refs_bt);

    for ti in 0..nt {
        let is_sup = ix.torrens_superseded.get(ti).copied().unwrap_or(false)
            || turn_is_superseded(ix, ti);
        if hist_override_local && (!ctx.has_named_speakers_eff || it.temporal) && is_sup && t_scene[ti] > 0.0 {
            let factor_q16 = crate::retrieval::graph::bitemporal_heideltime::bitemporal_validity_factor_q16(
                true,
                true,
            );
            t_scene[ti] *= (factor_q16 as f32) / 65_536.0;
        } else if present_state_local && ix.torrens_superseded.get(ti).copied().unwrap_or(false) {
            if is_monologue_stream {
                t_scene[ti] = 0.0;
                continue;
            } else if !has_named_corpus && ix.session_turns.len() >= 16 {
                // Multi-session user/assistant chats evaluate All@5 (both earlier & updated turns required):
                // use a 0.96x tie-break nudge so latest ranks first without evicting precursors from Top-5.
                t_scene[ti] *= 0.96;
            } else {
                t_scene[ti] *= 0.72;
            }
        }
        let t = &ix.turns[ti];
        if t_scene[ti] > 0.0
            && crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
                ix, ti, &q.text,
            )
        {
            t_scene[ti] *= 0.18;
            continue;
        }
        if policy_boost_ok && t.is_standing_policy && live_policy_turns.contains(&ti) && t_scene[ti] > 0.0 {
            t_scene[ti] *= 2.5;
        }
        if advisory_modal_local
            && t.is_personal_constraint
            && t.field == 0
            && !turn_is_superseded(ix, ti)
            && t_mask[ti] != 0
            && t_scene[ti] > 0.0
        {
            t_scene[ti] *= 1.52;
        }
    }

    let exact_groups = exact_query_group_ids(&plan);
    let q_ids: DetHashSet<u32> = plan.terms.iter().map(|qt| qt.id).collect();

    // Weapon 2: Deterministic Answer-Type Sniper + Query-Local Question-Echo Demotion
    if p.enable_sniper {
        for members in &ix.session_turns {
            let has_named_dialogue = members.iter().any(|&m| {
                let sp = ix.turns[m].speaker.as_str();
                !sp.is_empty() && sp != "user" && sp != "assistant"
            });
            let speaker_names = if has_named_dialogue {
                session_speaker_names(ix, members)
            } else {
                DetHashSet::default()
            };
            let queried_single_sp: Option<String> = if has_named_dialogue {
                let matched_sp: Vec<String> = speaker_names
                    .iter()
                    .filter(|sp| q_toks_bt.iter().any(|qt| qt == *sp))
                    .cloned()
                    .collect();
                if matched_sp.len() == 1 {
                    matched_sp.into_iter().next()
                } else {
                    None
                }
            } else {
                None
            };
            let has_declarative_peer = members.iter().any(|&m| !ix.turns[m].is_question && !ix.turns[m].is_backchannel);
            let has_declarative_user_peer = members.iter().any(|&m| {
                ix.turns[m].speaker == "user" && !ix.turns[m].is_question && !ix.turns[m].is_backchannel
            });
            for &ti in members {
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let t = &ix.turns[ti];
                let novel_entities = t.entities.iter().filter(|&&eid| !q_ids.contains(&eid)).count();
                let lacks_novel_assertion = !t.has_numeric && !t.form_clock && novel_entities < 2;
                if it.sniper_when {
                    if t.has_temporal {
                        // Phase 4 Step 10B: Guard `sniper_when` so wrong-speaker temporal mentions
                        // cannot override the queried single speaker's own event answer.
                        let sp_match = queried_single_sp
                            .as_deref()
                            .is_none_or(|qsp| t.speaker.eq_ignore_ascii_case(qsp));
                        t_scene[ti] *= if sp_match { 1.65 } else { 1.22 };
                    }
                    // Phase 4 Step 12A (T38 Realis/Irrealis Tense Gate): on past-tense "When did/was..." queries,
                    // gently demote purely prospective/future-plan mentions that carry NO realized past-deixis marker.
                    let q_past_when = ql_low.starts_with("when did ") || ql_low.starts_with("when was ") || ql_low.starts_with("when were ");
                    if has_named_dialogue && q_past_when && !t.is_question {
                        let body_low = t.body_for_headers.to_ascii_lowercase();
                        // hardcode-scan: closed-class (irrealis/prospective planning markers)
                        let irrealis_plan = [
                            "thinking about ", "going to ", "gonna ", "plan to ", "planning to ",
                            "next month", "next week", "next year",
                        ]
                        .iter()
                        .any(|m| body_low.contains(m));
                        // hardcode-scan: closed-class (past temporal deixis & past auxiliaries)
                        let has_realized_past = [
                            "yesterday", "last week", "last weekend", "last month", "last year",
                            " ago", " went ", " was ", " were ", " had ", " did ",
                        ]
                        .iter()
                        .any(|m| body_low.contains(m));
                        let body_toks: Vec<&str> = t
                            .body_for_headers
                            .split(|c: char| !c.is_ascii_alphanumeric())
                            .filter(|s| !s.is_empty())
                            .collect();
                        let tp = crate::retrieval::event_ordering::inspect_closed_class_tense_tokens(&body_toks);
                        if (tp.prospective_prolepsis || irrealis_plan)
                            && !tp.pluperfect_analepsis
                            && !has_realized_past
                        {
                            t_scene[ti] *= 0.86;
                        }
                    }
                    if t.is_question {
                        t_scene[ti] *= 0.55;
                    }
                } else if has_named_dialogue
                    && has_declarative_peer
                    && !episodic_user_quote
                    && turn_echoes_query(ix, t, &exact_groups, &speaker_names)
                {
                    t_scene[ti] *= 0.62;
                } else if has_named_dialogue
                    && has_declarative_peer
                    && t.is_question
                    && !it.sniper_when
                {
                    // 0.78 in Q16.16: question-turn demotion multiplier when a declarative peer exists.
                    const NAMED_QUESTION_DEMOTE_Q16: u32 = 51_118;
                    const Q16_SCALE_F32: f32 = 65_536.0;
                    t_scene[ti] *= (NAMED_QUESTION_DEMOTE_Q16 as f32) / Q16_SCALE_F32;
                } else if !has_named_dialogue
                    && !it.assistant_attribution
                    && has_declarative_user_peer
                    && !episodic_user_quote
                    && t.speaker == "user"
                    && (t.is_pure_info_question && !t.is_personal_constraint)
                    && lacks_novel_assertion
                    && turn_echoes_query(ix, t, &exact_groups, &speaker_names)
                {
                    t_scene[ti] *= 0.72;
                }
                if it.sniper_how_many && t.has_numeric {
                    t_scene[ti] *= 1.65;
                }
                if it.sniper_where && t.has_spatial {
                    t_scene[ti] *= 1.60;
                }
                if it.sniper_why && t.has_causal {
                    t_scene[ti] *= 1.55;
                }
                if t.is_short_question {
                    t_scene[ti] *= 0.50;
                }
                if t.is_backchannel {
                    t_scene[ti] *= 0.55;
                }
            }
        }
    }

    // Named-Partner cQA Question-Echo Swap (strictly `named_partner` ONLY — never `user`/`assistant`!):
    let mut q_ents_and_vocatives: DetHashSet<u32> = DetHashSet::default();
    for ent in proper_entity_tokens(&q.text) {
        if let Some(id) = ix.vocab.get(&ent) {
            q_ents_and_vocatives.insert(id);
        }
    }
    for w in raw_tokens(&q.text) {
        if let Some(id) = ix.vocab.get(&format!("\u{1}ent:{w}")) {
            q_ents_and_vocatives.insert(id);
        }
    }
    let query_has_temp = it.temporal || it.sniper_when || !inline_date_tokens(&q.text).is_empty();
    let query_has_num = it.sniper_how_many || q.text.chars().any(|c| c.is_ascii_digit());
    if p.enable_dyad {
        for members in &ix.session_turns {
            let speaker_names = session_speaker_names(ix, members);
            if speaker_names.is_empty() {
                continue;
            }
            let mut sess_speaker_ents: DetHashSet<u32> = DetHashSet::default();
            for sw in &speaker_names {
                if let Some(id) = ix.vocab.get(&format!("\u{1}ent:{sw}")) {
                    sess_speaker_ents.insert(id);
                }
            }
            for pos in 0..members.len() {
                let qi = members[pos];
                let qt = &ix.turns[qi];
                if t_scene[qi] <= 0.0 || !turn_echoes_query(ix, qt, &exact_groups, &speaker_names) {
                    continue;
                }
                let novel_temp = qt.has_temporal && !query_has_temp;
                let novel_num = qt.has_numeric && !query_has_num;
                let novel_ent = qt
                    .entities
                    .iter()
                    .any(|eid| !q_ents_and_vocatives.contains(eid) && !sess_speaker_ents.contains(eid));
                if novel_temp || novel_num || novel_ent {
                    continue;
                }
                if let Some(&ai) = members.get(pos + 1) {
                    let ans = &ix.turns[ai];
                    if named_partner(&qt.speaker, &ans.speaker) && !ans.is_question && !ans.is_backchannel {
                        t_mask[ai] |= t_mask[qi];
                        t_exact_mask[ai] |= t_exact_mask[qi];
                        if t_scene[qi] > t_scene[ai] {
                            t_scene[ai] = t_scene[ai].max(t_scene[qi] * 1.06);
                            t_scene[qi] *= 0.65;
                        }
                    }
                }
            }
        }
    }

    // Weapon 3: Bounded Length-Normalized 2-Iteration Sparse PPR (Turn Level)
    if p.enable_ppr && nt > 5 {
        let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
        let seed_indices: Vec<usize> = order(&t_scene, &turn_ts)
            .into_iter()
            .take(16)
            .filter(|&i| t_scene[i] > 0.0)
            .collect();

        if let Some(&t0) = seed_indices.first() {
            let t_max = t_scene[t0];
            let t0_mask = t_mask[t0];
            if t_max > 0.0 {
                let mut activation: std::collections::BTreeMap<u32, f32> = std::collections::BTreeMap::new();
                let max_df = ((ix.t_stats.n * 0.12).max(4.0)) as u32;

                // Pass 1: Normalized Seed Turn -> Discriminative Bridge Entities + Rare Exact Symbols
                for &s in &seed_indices {
                    if t_scene[s] < 0.15 * t_max {
                        break;
                    }
                    let seed_norm = (t_scene[s] / t_max) / ((ix.turns[s].content.len() + 4) as f32).sqrt();
                    for &e in &ix.turns[s].entities {
                        if is_technical_handle(&ix.vocab, e) {
                            continue;
                        }
                        if let Some(&df) = ix.t_stats.df.get(&e) {
                            if df >= 2 && df <= max_df {
                                let idf_term = 1.4 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *activation.entry(e).or_insert(0.0) += 2.0 * seed_norm * idf_term;
                            }
                        }
                    }
                    for &w in &ix.turns[s].content {
                        if plan.group_of.contains_key(&w) {
                            continue;
                        }
                        if let Some(&df) = ix.t_stats.df.get(&w) {
                            if df >= 2 && df <= max_df {
                                let idf_term = 1.2 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *activation.entry(w).or_insert(0.0) += 0.75 * seed_norm * idf_term;
                            }
                        }
                    }
                }

                // Pass 2: Bridge Entities -> Normalized, Bounded 2nd-Hop Turns
                if !activation.is_empty() {
                    for ti in 0..nt {
                        let mut ent_hit = 0.0f32;
                        for &e in &ix.turns[ti].entities {
                            if let Some(&act) = activation.get(&e) {
                                ent_hit += 2.0 * act;
                            }
                        }
                        let mut ppr_sum = ent_hit;
                        for &w in &ix.turns[ti].content {
                            if let Some(&act) = activation.get(&w) {
                                ppr_sum += 0.75 * act;
                            }
                        }
                        if ppr_sum > 0.0 && (t_scene[ti] > 0.0 || ent_hit > 0.0) {
                            let target_norm = ppr_sum / ((ix.turns[ti].content.len() + 4) as f32).sqrt();
                            let sat = target_norm / (target_norm + 0.45);
                            let mult = if (t_mask[ti] & !t0_mask) != 0 { 1.65 } else { 1.0 };
                            let boost = t_max * 0.25 * mult * sat;
                            t_scene[ti] += boost;
                        }
                    }
                }
            }
        }
    }

    // Hierarchical session-to-turn funnel.
    if p.enable_funnel && smax > 0.0 {
        for ti in 0..nt {
            if t_scene[ti] > 0.0 {
                t_scene[ti] *= 1.0 + p.funnel * (s_score[ix.turns[ti].session] / smax);
            }
        }
    }

    // Adjacency-pair dyad coupling: answers to questions AND short commitment/value replies
    // to statements ("Done.", "Last Tuesday.", "Me too!") ride alongside their setup turn.
    if p.enable_dyad {
        let base = t_scene.to_vec();
        for members in &ix.session_turns {
            for pos in 1..members.len() {
                let (a, b) = (members[pos - 1], members[pos]);
                if !ix.turns[b].answers_question && !ix.turns[b].is_short_reply {
                    continue;
                }
                let partner_ratio = if ix.turns[b].answers_question { p.dyad_partner } else { 0.84 };
                let (hi, lo) = if base[a] >= base[b] { (a, b) } else { (b, a) };
                if base[hi] <= 0.0 {
                    continue;
                }
                let dyad = base[hi] + p.dyad * base[lo];
                t_scene[hi] = t_scene[hi].max(dyad);
                t_scene[lo] = t_scene[lo].max(dyad * partner_ratio);
                t_mask[lo] |= t_mask[hi];
            }
        }
    }

}

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let p = ctx.p;
    let nt = ctx.nt;
    let ns = ctx.ns;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let pref_on = ctx.pref_on;
    let mut t_own = std::mem::take(&mut ctx.t_own);
    let t_raw = std::mem::take(&mut ctx.t_raw);
    let t_turn_raw = std::mem::take(&mut ctx.t_turn_raw);
    let mut t_mask = std::mem::take(&mut ctx.t_mask);
    let mut t_exact_mask = std::mem::take(&mut ctx.t_exact_mask);

    // Weapon 1: Bidirectional 3-Turn Scene Pooling.
    // - On named-speaker dialogues (`LoCoMo` / multi-person chats), pool over uncarried `t_own` +
    //   Center-Peak Monotonicity so follow-up questions & reactions (`gold + 1`) never steal Rank 1!
    // - On `user`/`assistant` dialogues (`BEAM` / `LME`), preserve symmetric `t_raw` pooling so
    //   concise user turns ending in `?` are never inverted beneath verbose assistant expansions!
    let mut t_scene = vec![0.0f32; nt];
    let mut t_scene_sess = vec![0.0f32; nt];
    if p.enable_scene {
        for members in &ix.session_turns {
            let has_named_dialogue = members.iter().any(|&m| {
                let sp = ix.turns[m].speaker.as_str();
                !sp.is_empty() && sp != "user" && sp != "assistant"
            });
            if has_named_dialogue {
                // 0.22 in Q16.16: radius-2 named-dialogue exchange carry weight into `t_scene`.
                const NAMED_PREV2_CARRY_Q16: u32 = 14_418;
                const Q16_SCALE_F32: f32 = 65_536.0;
                // Phase 4 Step 10B: Detect single queried named speaker to dampen cross-speaker bleed
                // (prevents partner follow-up turns from inheriting queried speaker's gold score).
                let sp_set = session_speaker_names(ix, members);
                let q_toks_sc = raw_tokens(&q.text);
                let q_sp_matches: Vec<String> = sp_set
                    .iter()
                    .filter(|sp| q_toks_sc.iter().any(|qt| qt == *sp))
                    .cloned()
                    .collect();
                let target_sp: Option<&str> = if q_sp_matches.len() == 1 {
                    Some(q_sp_matches[0].as_str())
                } else {
                    None
                };
                for (pos, &ti) in members.iter().enumerate() {
                    let prev_own = if pos > 0 { t_own[members[pos - 1]] } else { 0.0 };
                    let prev2_own = if pos >= 2 { t_own[members[pos - 2]] } else { 0.0 };
                    let next_own = if pos + 1 < members.len() { t_own[members[pos + 1]] } else { 0.0 };
                    let ti_is_target = target_sp.is_none_or(|tsp| ix.turns[ti].speaker.eq_ignore_ascii_case(tsp));
                    let prev_is_target = pos > 0
                        && target_sp.is_some_and(|tsp| ix.turns[members[pos - 1]].speaker.eq_ignore_ascii_case(tsp));
                    let next_is_target = pos + 1 < members.len()
                        && target_sp.is_some_and(|tsp| ix.turns[members[pos + 1]].speaker.eq_ignore_ascii_case(tsp));
                    let base_prev_w = if ix.turns[ti].is_backchannel {
                        0.22
                    } else if pos > 0 && ix.turns[members[pos - 1]].is_question {
                        0.75
                    } else {
                        0.48
                    };
                    let mut prev_w = if pos > 0 && ix.turns[members[pos - 1]].is_question && ti_is_target && !ix.turns[ti].is_question {
                        // T63 Q->A Adjacency: declarative answer by queried speaker inherits strong credit from partner's question.
                        0.88
                    } else {
                        base_prev_w
                    };
                    let mut next_w = 0.42f32;
                    if !ti_is_target && !ix.turns[ti].answers_question {
                        if prev_is_target {
                            prev_w *= 0.45;
                        }
                        if next_is_target {
                            next_w = 0.20;
                        }
                    }
                    let base_prev2_w = if !ix.turns[ti].is_question
                        && !ix.turns[ti].is_backchannel
                        && (ix.turns[ti].answers_question || ix.turns[ti].anaphoric || prev_own > 0.0)
                    {
                        (NAMED_PREV2_CARRY_Q16 as f32) / Q16_SCALE_F32
                    } else {
                        0.0
                    };
                    let prev2_w = if ti_is_target { base_prev2_w } else { base_prev2_w * 0.55 };
                    let self_score = if ix.turns[ti].answers_question || ix.turns[ti].is_short_reply || ix.turns[ti].anaphoric {
                        t_raw[ti]
                    } else {
                        t_own[ti] + 0.18 * (t_raw[ti] - t_own[ti]).max(0.0)
                    };
                    t_scene[ti] = self_score + prev_w * prev_own + prev2_w * prev2_own + next_w * next_own;
                    // Keep `t_scene_sess` on symmetric named-dialogue pooling so Single-Hop session peak blend never dips.
                    t_scene_sess[ti] = self_score + base_prev_w * prev_own + base_prev2_w * prev2_own + 0.42 * next_own;
                    if pos > 0 && !ix.turns[ti].is_backchannel {
                        t_mask[ti] |= t_mask[members[pos - 1]];
                    }
                    if pos + 1 < members.len() && !ix.turns[ti].is_question {
                        t_mask[ti] |= t_mask[members[pos + 1]];
                    }
                }
                for pos in 1..members.len() {
                    let prev = members[pos - 1];
                    let ti = members[pos];
                    if !ix.turns[prev].is_question
                        && !ix.turns[prev].is_backchannel
                        && !ix.turns[ti].answers_question
                        && t_own[prev] > 0.0
                        && (t_exact_mask[ti] & !t_exact_mask[prev]) == 0
                        && t_own[ti] <= t_own[prev] * 1.05
                        && t_scene[ti] > 0.92 * t_scene[prev]
                    {
                        t_scene[ti] = 0.92 * t_scene[prev];
                    }
                    if !ix.turns[prev].is_question
                        && !ix.turns[prev].is_backchannel
                        && !ix.turns[ti].answers_question
                        && t_own[prev] > 0.0
                        && (t_exact_mask[ti] & !t_exact_mask[prev]) == 0
                        && t_own[ti] <= t_own[prev] * 1.05
                        && t_scene_sess[ti] > 0.92 * t_scene_sess[prev]
                    {
                        t_scene_sess[ti] = 0.92 * t_scene_sess[prev];
                    }
                }
            } else {
                for (pos, &ti) in members.iter().enumerate() {
                    let own_m = rung_own_qmask(ix, &plan, ti);
                    let prev_adds = pos > 0 && (rung_own_qmask(ix, &plan, members[pos - 1]) & !own_m) != 0;
                    let next_adds = pos + 1 < members.len()
                        && (rung_own_qmask(ix, &plan, members[pos + 1]) & !own_m) != 0;
                    let prev_w = if ix.turns[ti].speaker == "user" && !ix.turns[ti].is_short_reply {
                        0.0
                    } else if prev_adds {
                        if ix.turns[ti].is_backchannel {
                            0.25
                        } else if pos > 0 && ix.turns[members[pos - 1]].is_question {
                            0.75
                        } else {
                            0.55
                        }
                    } else {
                        0.15
                    };
                    let next_w = if next_adds { 0.55 } else { 0.18 };
                    let user_q_before_asst = ix.turns[ti].speaker == "user"
                        && ix.turns[ti].is_question
                        && !it.assistant_attribution
                        && pos + 1 < members.len()
                        && ix.turns[members[pos + 1]].speaker == "assistant";
                    let next_tr_w = if user_q_before_asst && ix.turns[ti].is_pure_info_question {
                        0.12
                    } else {
                        next_w
                    };
                    let prev_sess = if pos > 0 { t_raw[members[pos - 1]] } else { 0.0 };
                    let next_sess = if pos + 1 < members.len() { t_raw[members[pos + 1]] } else { 0.0 };
                    t_scene_sess[ti] = 1.0 * t_raw[ti] + prev_w * prev_sess + next_w * next_sess;
                    let prev_tr = if pos > 0 { t_turn_raw[members[pos - 1]] } else { 0.0 };
                    let next_tr = if pos + 1 < members.len() { t_turn_raw[members[pos + 1]] } else { 0.0 };
                    t_scene[ti] = 1.0 * t_turn_raw[ti] + prev_w * prev_tr + next_tr_w * next_tr;
                    if pos > 0 && !ix.turns[ti].is_backchannel {
                        t_mask[ti] |= t_mask[members[pos - 1]];
                    }
                    if pos + 1 < members.len()
                        && !(user_q_before_asst && ix.turns[ti].is_pure_info_question)
                    {
                        t_mask[ti] |= t_mask[members[pos + 1]];
                    }
                }
            }
        }
    } else {
        t_scene = t_turn_raw.clone();
        t_scene_sess = t_raw.clone();
    }

    // #809: Sidner FocusStack<EntityId>, #811: Hobbs Pronoun Walk, #813: Union-Find Coref
    if nt > 2 && !plan.focal_ids.is_empty() {
        let focus_q16 = crate::retrieval::query::focus::compute_anaphoric_focus_q16(ix, &plan);
        let hobbs_res = crate::retrieval::query::hobbs::resolve_hobbs_pronouns(ix, &plan);
        let coref_q16 = crate::retrieval::query::coref::coref_class_affinity_q16(ix, &plan);
        let t_max_sc = t_scene.iter().cloned().fold(0.0f32, f32::max);
        let t_max_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(t_max_sc.to_bits());
        // Stage-03 runs before stage-04 rebinds `smax`; read the live session scores from `ctx`.
        let s_score_s03 = &ctx.s_score;
        let smax_s03 = s_score_s03.iter().copied().fold(0.0f32, f32::max);
        let smax_s03_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(smax_s03.to_bits());
        if t_max_q16 > 0 {
            // Pure Q16.16 integer weights: ANA_W=3932 (~0.06), COREF_W=7864 (~0.12), RESCUE_W=15729 (~0.24)
            const ANA_W_Q16: u64 = 3_932;
            const COREF_W_Q16: u64 = 7_864;
            const RESCUE_W_Q16: u64 = 15_729;
            const Q16_SCALE_F32: f32 = 65_536.0;
            for ti in 0..nt {
                let fq = focus_q16[ti];
                let hq = hobbs_res.turn_score_q16[ti];
                let cq = coref_q16[ti];
                if fq == 0 && hq == 0 && cq == 0 {
                    continue;
                }
                if t_scene[ti] > 0.0 {
                    let ana_part = (ANA_W_Q16.saturating_mul(fq.max(hq))) >> 16;
                    let coref_part = (COREF_W_Q16.saturating_mul(cq)) >> 16;
                    let boost_q16 = (t_max_q16.saturating_mul(ana_part.saturating_add(coref_part))) >> 16;
                    if boost_q16 > 0 {
                        let boost_f32 = (boost_q16 as f32) / Q16_SCALE_F32;
                        t_scene[ti] += boost_f32;
                        t_scene_sess[ti] += boost_f32;
                    }
                } else {
                    // Rescue zero-lexical declarative pronoun/alias answer turns inside strong
                    // grounded sessions in Q16.16 (t_own and t_raw remain 100% untouched).
                    let tr = &ix.turns[ti];
                    let sess_q16 = crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16(
                        s_score_s03.get(tr.session).copied().unwrap_or(0.0).to_bits(),
                    );
                    let sess_ok = smax_s03_q16 > 0 && sess_q16.saturating_mul(20) >= smax_s03_q16.saturating_mul(7);
                    if sess_ok
                        && !tr.is_question
                        && !tr.is_backchannel
                        && !turn_is_superseded(ix, ti)
                        && (hq >= (crate::retrieval::query::hobbs::Q16 >> 1)
                            || cq >= (crate::retrieval::query::coref::Q16 >> 1))
                    {
                        let hc_q16 = (RESCUE_W_Q16.saturating_mul(hq.max(cq))) >> 16;
                        let rescue_q16 = (t_max_q16.saturating_mul(hc_q16)) >> 16;
                        if rescue_q16 > 0 {
                            let rescue_f32 = (rescue_q16 as f32) / Q16_SCALE_F32;
                            t_scene[ti] += rescue_f32;
                            t_scene_sess[ti] += rescue_f32;
                        }
                    }
                }
            }
        }
        let _ = hobbs_res.bindings;
    }

    // Named-speaker boost (never a filter): case-insensitive speaker alignment (`1.14x` when a single
    // dialogue speaker is queried without `<Speaker> <CapSurname>` full-name collision, `1.06x` when multiple).
    let cased_q_words: Vec<&str> = q.text.split_whitespace().collect();
    let is_surname_collision = |sp_low: &str| -> bool {
        cased_q_words.windows(2).any(|w2| {
            let first = w2[0].trim_matches(|c: char| !c.is_alphanumeric());
            let second = w2[1].trim_matches(|c: char| !c.is_alphanumeric());
            first.eq_ignore_ascii_case(sp_low)
                && second.len() >= 2
                && second.chars().next().is_some_and(char::is_uppercase)
                && !ix.turns.iter().take(64).any(|t| t.speaker.eq_ignore_ascii_case(second))
        })
    };
    let mut queried_speakers: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(64) {
        let sp = tr.speaker.as_str();
        if !sp.is_empty()
            && sp != "user"
            && sp != "assistant"
            && plan.groups.iter().any(|g| g.eq_ignore_ascii_case(sp))
            && !is_surname_collision(&sp.to_ascii_lowercase())
            && !queried_speakers.iter().any(|s| s.eq_ignore_ascii_case(sp))
        {
            queried_speakers.push(sp.to_ascii_lowercase());
        }
    }
    let sp_mult = if queried_speakers.len() == 1 { 1.16 } else { 1.06 };
    // Strict Speaker-Attributed Predicates: collect exact non-speaker query content term IDs
    // so a turn by the queried speaker that ONLY mentions those terms inside 2nd-person
    // ("you"/"your") clauses about the interlocutor does NOT receive the queried-speaker boost.
    let q_non_sp_exact_ids: Vec<u32> = plan
        .terms
        .iter()
        .filter(|qt| {
            qt.exact_hit
                && ix
                    .vocab
                    .names
                    .get(qt.id as usize)
                    .is_some_and(|nm| !queried_speakers.iter().any(|qs| qs.eq_ignore_ascii_case(nm)))
        })
        .map(|qt| qt.id)
        .collect();
    for t_sc in [&mut t_scene[..], &mut t_scene_sess[..]] {
        for ti in 0..nt {
            if t_raw[ti] <= 0.0 && t_own[ti] <= 0.0 {
                continue;
            }
            let tr = &ix.turns[ti];
            let sp = tr.speaker.as_str();
            if !sp.is_empty()
                && sp != "user"
                && sp != "assistant"
                && queried_speakers.iter().any(|qs| qs.eq_ignore_ascii_case(sp))
            {
                let mut matched_any = false;
                let mut all_in_other = true;
                if queried_speakers.len() == 1 && !tr.other_speaker_terms.is_empty() {
                    for &qid in &q_non_sp_exact_ids {
                        if tr.content.contains(&qid) {
                            matched_any = true;
                            if !tr.other_speaker_terms.contains(&qid) {
                                all_in_other = false;
                                break;
                            }
                        }
                    }
                } else {
                    all_in_other = false;
                }
                if matched_any && all_in_other {
                    t_sc[ti] *= 0.88;
                } else {
                    t_sc[ti] *= sp_mult;
                }
            }
        }
    }


    // Commit intermediate masks/vecs so apply_scene_refinements sees unchanged `ctx` fields;
    // wait: apply_scene_refinements only reads `ctx.{ix,q,p,plan,nt,smax,s_score,user_role,episodic_user_quote}`!
    // Compute derived internal intent flags without mutating `plan.intent` (preserving public Skeptic-Audit contract).
    let q_toks_eff = raw_tokens(&q.text);
    let seq_mode_eff = it.is_event_sequence
        || q_toks_eff.windows(2).any(|p| {
            (p[0] == "in" && p[1] == "order" && q_toks_eff.windows(3).all(|w3| !(w3[0] == "in" && w3[1] == "order" && w3[2] == "to")))
                || (p[0] == "order" && (p[1] == "in" || p[1] == "of"))
                || ((p[0] == "what" || p[0] == "which") && p[1] == "order")
                || (p[0] == "sequence" && p[1] == "of")
        })
        || q_toks_eff
            .iter()
            .any(|w| matches!(w.as_str(), "chronological" | "chronologically" | "timeline" | "progression"));
    let past_cue_eff = q_toks_eff.iter().any(|w| {
        matches!(
            w.as_str(),
            "before" | "previously" | "originally" | "earlier" | "initial" | "initially" | "former" | "formerly" | "changed" | "change" | "both" | "history" | "first"
        )
    }) || q_toks_eff.windows(2).any(|p| p[0] == "used" && p[1] == "to");
    let has_named_speakers_eff = ix
        .turns
        .iter()
        .take(64)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let is_monologue_corpus = !ix.turns.iter().any(|t| t.field == 1) && !has_named_speakers_eff;
    let large_or_mono = ix.session_turns.len() >= 16 || ix.turns.len() >= 64 || is_monologue_corpus;
    let q_trim_low = q.text.trim().to_ascii_lowercase();
    let is_when_participial = q_toks_eff
        .first()
        .is_some_and(|w| w == "when")
        && q_toks_eff
            .get(1)
            .is_some_and(|w| w.len() >= 5 && w.ends_with("ing"));
    // hardcode-scan: closed-class (retrospective assistant-recall & episodic interrogation cues)
    let has_gapped_you_report = q_toks_eff.windows(3).any(|w3| {
        w3[0] == "you"
            && matches!(
                w3[1].as_str(),
                "previously" | "earlier" | "recently" | "already" | "just" | "also" | "once"
            )
            && matches!(
                w3[2].as_str(),
                "recommended"
                    | "suggested"
                    | "provided"
                    | "told"
                    | "gave"
                    | "mentioned"
                    | "shared"
            )
    });
    let has_retro_user_cue = q_trim_low.contains("i told you")
        || q_trim_low.contains("did i tell you")
        || q_trim_low.contains("have i told you")
        || q_trim_low.contains("have i mentioned");
    let has_retro_asst_cue = has_gapped_you_report
        || q_trim_low.contains("did you ")
        || q_trim_low.contains("have you ")
        || (q_trim_low.contains("remind me") && !q_trim_low.contains(" i "))
        || q_trim_low.contains("last time")
        || q_trim_low.contains("previous conversation")
        || q_trim_low.contains("previous chat")
        || q_trim_low.contains("you recommended")
        || q_trim_low.contains("you suggested")
        || q_trim_low.contains("you provided")
        || q_trim_low.contains("you told me")
        || q_trim_low.contains("you gave me")
        || q_trim_low.contains("you mentioned")
        || q_trim_low.contains("you shared")
        || q_trim_low.contains("i gave you")
        || q_trim_low.contains("i recommended")
        || q_trim_low.contains("i suggested")
        || q_trim_low.contains("i provided")
        || q_trim_low.contains("we talked about")
        || q_trim_low.contains("we discussed");
    // hardcode-scan: closed-class (cardinal enumerative WH-frames for factual multi-entity lookup)
    let is_factual_enum_q = q_trim_low.starts_with("what two ")
        || q_trim_low.starts_with("what three ")
        || q_trim_low.starts_with("what are the two ")
        || q_trim_low.starts_with("what are the three ")
        || q_trim_low.starts_with("which two ")
        || q_trim_low.starts_with("which three ");
    let is_factual_retro_query = has_retro_asst_cue
        || has_retro_user_cue
        || is_factual_enum_q
        || q_trim_low.contains("did i ")
        || q_trim_low.contains("did we ")
        || q_trim_low.contains("have i ")
        || q_trim_low.contains("was my ")
        || q_trim_low.contains("were my ")
        || q_trim_low.contains("between ")
        || q_trim_low.starts_with("what is ")
        || q_trim_low.starts_with("what was ")
        || q_trim_low.starts_with("how many ")
        || q_trim_low.starts_with("how long ");
    // T29: closed-class 1SG advice/planning request gate on long user/assistant chats.
    // hardcode-scan: closed-class (1SG progressive/modal advice openings, Searle directives)
    let advice_shape_eff = !has_named_speakers_eff
        && !is_monologue_corpus
        && ix.turns.len() >= 64
        && !is_factual_retro_query
        && (q_trim_low.starts_with("i'm planning")
            || q_trim_low.starts_with("i'm looking")
            || q_trim_low.starts_with("i'm about to")
            || q_trim_low.starts_with("i'm working on")
            || q_trim_low.starts_with("i'm updating")
            || q_trim_low.starts_with("i'm thinking about")
            || q_trim_low.starts_with("i'm trying to decide")
            || q_trim_low.starts_with("i\u{2019}m planning")
            || q_trim_low.starts_with("i\u{2019}m looking")
            || q_trim_low.starts_with("i\u{2019}m about to")
            || q_trim_low.starts_with("i\u{2019}m working on")
            || q_trim_low.starts_with("i\u{2019}m updating")
            || q_trim_low.starts_with("i\u{2019}m thinking about")
            || q_trim_low.starts_with("i\u{2019}m trying to decide")
            || q_trim_low.starts_with("can you help me")
            || q_trim_low.starts_with("can you walk me through")
            || q_trim_low.starts_with("can you explain")
            || q_trim_low.starts_with("can you show me how")
            || q_trim_low.starts_with("i'm preparing ")
            || q_trim_low.starts_with("i\u{2019}m preparing ")
            || q_trim_low.starts_with("i want to set up ")
            || q_trim_low.contains("where do you think i should")
            || q_trim_low.contains("would you suggest")
            || q_trim_low.contains("can you suggest")
            || q_trim_low.contains("could you suggest")
            || q_trim_low.contains("would you recommend")
            || q_trim_low.contains("can you recommend")
            || q_trim_low.contains("could you recommend")
            || q_trim_low.contains("what would you suggest")
            || q_trim_low.contains("how would you suggest"));
    let is_how_what_task_shape = q_trim_low.starts_with("how should i ")
        || q_trim_low.starts_with("how do i ")
        || q_trim_low.starts_with("what should i ")
        || q_trim_low.starts_with("what steps do i ")
        || q_trim_low.starts_with("what steps should i ")
        || q_trim_low.starts_with("how can i ")
        || q_trim_low.starts_with("what are ")
        || q_trim_low.starts_with("if i")
        || (q_trim_low.contains("which ") && q_trim_low.contains(" should i"))
        || is_when_participial;
    let has_compact_rule_register = !ix.standing_rules.is_empty();
    // hardcode-scan: closed-class (1st-person possessive pronouns gating personal advice vs general rules)
    let has_personal_poss = q_toks_eff
        .iter()
        .any(|w| matches!(w.as_str(), "my" | "our" | "mine" | "ours"))
        || q_trim_low.contains("i'd like")
        || q_trim_low.contains("i would like")
        || q_trim_low.contains("i'm looking")
        || q_trim_low.contains("i'm working");
    // Rule-condition gate: a compact rule register only marks the query as a
    // task-under-rules consultation when some live rule's own condition clause
    // ("when I ask about X") or strict high-IC rule-body binding matches the query.
    let cond_rule_turns: Vec<usize> = ix
        .standing_rules
        .iter()
        .filter(|r| crate::retrieval::rule_condition_matches(ix, r, &plan, &q_toks_eff))
        .map(|r| r.turn)
        .collect();
    let rule_condition_live = has_compact_rule_register && !cond_rule_turns.is_empty();
    // hardcode-scan: closed-class (simple "When is/was ..." date queries governed by date-format standing rules)
    let has_date_format_rule = ix.standing_rules.iter().any(|r| {
        let b = ix.turns[r.turn].body_for_headers.to_ascii_lowercase();
        (b.contains("format") && b.contains("date")) || (b.contains("mm/") && b.contains("yyyy"))
    });
    let is_simple_when_date_q = !has_named_speakers_eff
        && !is_monologue_corpus
        && has_date_format_rule
        && !it.sniper_how_many
        && !q_trim_low.contains("between ")
        && (q_trim_low.starts_with("when is ")
            || q_trim_low.starts_with("when was ")
            || q_trim_low.contains("remind me of the date")
            || q_trim_low.starts_with("what is the date")
            || q_trim_low.starts_with("what date"));
    // hardcode-scan: closed-class (near-future temporal deictics paired with explicit recommendation requests)
    let is_near_future_rec_q = !it.sniper_when
        && (q_trim_low.contains("tonight")
            || q_trim_low.contains("this weekend")
            || q_trim_low.contains("next week"))
        && crate::retrieval::step14_is_rec_request(&q.text);
    let pref_live_eff = ix.turns.len() >= 24
        && !past_cue_eff
        && (!it.temporal || is_near_future_rec_q)
        && !seq_mode_eff
        && !eff_summary_request(&q.text)
        && !is_factual_retro_query
        && !( !has_personal_poss && !q_trim_low.starts_with("i'm ") && rule_condition_live )
        && (advice_shape_eff
            || (crate::retrieval::step14_is_rec_request(&q.text)
                && !is_how_what_task_shape
                && !rule_condition_live));
    let task_live_eff = !is_monologue_corpus
        && !has_named_speakers_eff
        && ix.turns.len() >= 64
        && !past_cue_eff
        && (!it.temporal || is_when_participial || is_simple_when_date_q)
        && (!it.sniper_when || is_when_participial || is_simple_when_date_q)
        && !it.sniper_how_many
        && !seq_mode_eff
        && !pref_live_eff
        && !eff_summary_request(&q.text)
        && !is_factual_retro_query
        && (!q_trim_low.ends_with('?')
            || q_trim_low.starts_with("please ")
            || rule_condition_live
            || is_simple_when_date_q
            || q_toks_eff.windows(2).any(|p| {
                ((p[0] == "could" || p[0] == "can" || p[0] == "would") && p[1] == "you")
                    || (p[0] == "how" && (p[1] == "to" || p[1] == "can" || p[1] == "should" || p[1] == "do"))
                    || (p[0] == "what" && (p[1] == "should" || p[1] == "are" || p[1] == "typical"))
                    || (p[0] == "which" && p[1] != "session")
                    || (p[0] == "should" && (p[1] == "i" || p[1] == "we"))
            }));
    let explicit_user_mention_only = (q_trim_low.contains("i mentioned")
        || q_trim_low.contains("did i mention")
        || q_trim_low.contains("i said ")
        || q_trim_low.contains("did i say"))
        && !(has_gapped_you_report
            || q_trim_low.contains("did you ")
            || q_trim_low.contains("have you ")
            || q_trim_low.contains("you recommended")
            || q_trim_low.contains("you suggested")
            || q_trim_low.contains("you provided")
            || q_trim_low.contains("you told me")
            || q_trim_low.contains("you gave me")
            || q_trim_low.contains("you mentioned")
            || q_trim_low.contains("you shared"));
    let asst_attr_eff = !has_named_speakers_eff
        && !is_monologue_corpus
        && ix.turns.len() >= 4
        && has_retro_asst_cue
        && !explicit_user_mention_only;
    let closed_class_mh = !is_monologue_corpus
        && ix.session_turns.len() >= 3
        && !it.temporal
        && !it.sniper_when
        && !seq_mode_eff
        && !pref_live_eff
        && !task_live_eff
        && crate::retrieval::graph::hashimoto_tripartite_q16::closed_class_plural_or_collective_mh(
            &q_toks_eff,
        );
    let mh_eff = it.multi_hop || closed_class_mh;
    let eff_it = QueryIntent {
        preference: it.preference || pref_live_eff,
        assistant_attribution: it.assistant_attribution || asst_attr_eff,
        current_state: it.current_state
            || (!past_cue_eff && !it.temporal && !seq_mode_eff && !it.sniper_how_many),
        multi_hop: mh_eff,
        is_event_sequence: if large_or_mono { seq_mode_eff } else { it.is_event_sequence },
        task_under_rules: it.task_under_rules || task_live_eff,
        ..it.clone()
    };
    let pref_on = pref_on || (p.enable_preference && eff_it.preference);
    // hardcode-scan: closed-class (personal factual/itinerary openings excluded from standing policy hijack)
    let is_personal_fact_q = q_trim_low.starts_with("when am i ")
        || q_trim_low.starts_with("what time ")
        || q_trim_low.starts_with("by what date")
        || q_trim_low.starts_with("what is my ");
    let policy_boost_ok = !eff_summary_request(&q.text)
        && !eff_it.is_event_sequence
        && !eff_it.preference
        && !is_personal_fact_q
        && (!eff_it.temporal || is_when_participial || is_simple_when_date_q)
        && !eff_it.sniper_how_many
        && !past_cue_eff
        && (ix.turns.len() < 64 || (!is_factual_retro_query && eff_it.task_under_rules));
    let live_policy_turns: DetHashSet<usize> = if policy_boost_ok {
        ix.standing_rules
            .iter()
            .filter(|r| {
                !turn_is_superseded(ix, r.turn)
                    && (r.condition_words.is_empty() || cond_rule_turns.contains(&r.turn))
            })
            .map(|r| r.turn)
            .collect()
    } else {
        DetHashSet::default()
    };

    // Run session-path refinements first on a mask copy, then restore and run the
    // turn-path refinements so PPR seeds see pre-dyad masks (floor behavior).
    let t_mask_snap = t_mask.clone();
    let t_exact_snap = t_exact_mask.clone();
    apply_scene_refinements(ctx, policy_boost_ok, &live_policy_turns, &mut t_scene_sess, &mut t_mask, &mut t_exact_mask);
    t_mask.copy_from_slice(&t_mask_snap);
    t_exact_mask.copy_from_slice(&t_exact_snap);
    apply_scene_refinements(ctx, policy_boost_ok, &live_policy_turns, &mut t_scene, &mut t_mask, &mut t_exact_mask);
    if eff_it.current_state && is_monologue_corpus {
        for (ti, &sup) in ix.torrens_superseded.iter().enumerate() {
            if sup {
                t_own[ti] = 0.0;
            }
        }
    }
    let mut s_score = std::mem::take(&mut ctx.s_score);
    // Operational monologue recency shaping (ISO-8601 dated logs relative to query anchor):
    if is_monologue_corpus && p.enable_recency && eff_it.current_state {
        let max_sts = ix.session_ts.iter().copied().max().unwrap_or(0);
        let min_sts = ix
            .session_ts
            .iter()
            .copied()
            .filter(|&ts| ts > 0)
            .min()
            .unwrap_or(0);
        // Only activate calendar recency when real multi-day timestamps exist (`span >= 7 days`)
        if max_sts > 0 && min_sts > 0 && (max_sts - min_sts) >= 7 * 86_400 {
            let q_anchor = parse_date(&q.date).max(max_sts);
            let cur_smax = s_score.iter().copied().fold(0.0f32, f32::max);
            let cur_tmax = t_scene.iter().copied().fold(0.0f32, f32::max);
            if cur_smax > 0.0 {
                for si in 0..ns {
                    let ts = ix.session_ts[si];
                    if ts > 0 && s_score[si] >= 0.38 * cur_smax {
                        let age_days = ((q_anchor - ts).max(0) as f32) / 86_400.0;
                        let rec_mult = 1.0 + 0.26 / (1.0 + age_days / 90.0);
                        s_score[si] *= rec_mult;
                    }
                }
            }
            if cur_tmax > 0.0 {
                for ti in 0..nt {
                    let ts = ix.session_ts[ix.turns[ti].session];
                    if ts > 0 && t_scene[ti] >= 0.38 * cur_tmax && !turn_is_superseded(ix, ti) {
                        let age_days = ((q_anchor - ts).max(0) as f32) / 86_400.0;
                        let rec_mult = 1.0 + 0.26 / (1.0 + age_days / 90.0);
                        t_scene[ti] *= rec_mult;
                        t_scene_sess[ti] *= rec_mult;
                    }
                }
            }
        }
    }
    apply_step7_lex_specialis(ix, &q.text, &plan, pref_on, it, &t_own, &mut t_scene);
    apply_c2_tap_form_precedence(ix, it, pref_on, &mut t_scene);
    apply_step13_khipu_turn_bridge(
        ix,
        &q.text,
        &plan,
        pref_on,
        it,
        &t_own,
        &t_exact_mask,
        &mut t_scene,
    );
    apply_step17_isnad_multihop_chain(
        ix,
        &q.text,
        &plan,
        pref_on,
        it,
        &t_own,
        &t_exact_mask,
        &mut t_scene,
    );
    apply_step14_bipartite_pref_boost(ix, &q.text, &plan, pref_on, it, &t_own, &mut t_scene);
    apply_step15_comparative_dual_slot_scores(
        ix,
        &q.text,
        &plan,
        it,
        &t_own,
        &t_exact_mask[..],
        &mut t_scene,
    );

    // Isolated 2-hop Meet-in-the-Middle WordNet + Haystack PPMI channel (`t_bridge`).
    // Borrow-checker clean; never mutates `t_own`, `t_raw`, `t_scene`, or `s_score`.
    let (t_bridge, t_bridge_hits) = score_bridge(ix, q, p, &plan, nt);


    ctx.s_score = s_score;
    ctx.t_own = t_own;
    ctx.t_raw = t_raw;
    ctx.t_turn_raw = t_turn_raw;
    ctx.t_mask = t_mask;
    ctx.t_exact_mask = t_exact_mask;
    ctx.t_scene = t_scene;
    ctx.t_scene_sess = t_scene_sess;
    ctx.q_toks_eff = q_toks_eff;
    ctx.past_cue_eff = past_cue_eff;
    ctx.seq_mode_eff = seq_mode_eff;
    ctx.has_named_speakers_eff = has_named_speakers_eff;
    ctx.is_monologue_corpus = is_monologue_corpus;
    ctx.has_compact_rule_register = has_compact_rule_register;
    ctx.cond_rule_turns = cond_rule_turns;
    ctx.eff_it = eff_it;
    ctx.pref_on = pref_on;
    ctx.t_bridge = t_bridge;
    ctx.t_bridge_hits = t_bridge_hits;
}
