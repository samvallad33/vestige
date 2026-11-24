//! Stage 02: Turn-level BM25F (`t_own` and `t_raw`), bigram bonus, and Rung 1 slot specificity.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let p = ctx.p;
    let plan = &ctx.plan;
    let it = &plan.intent;
    let pref_on = ctx.pref_on;
    let persona_infer_on = ctx.persona_infer_on;
    let fl = &ctx.fl;
    let mut s_score = std::mem::take(&mut ctx.s_score);

    // ---- Turn level: score BOTH uncarried `t_own` AND carried `t_raw` ----
    let nt = ix.turns.len();
    let mut t_own = vec![0.0f32; nt];
    let mut t_raw = vec![0.0f32; nt];
    let mut t_turn_raw = vec![0.0f32; ix.turns.len()];
    let mut t_mask = vec![0u64; nt];
    let mut t_exact_mask = vec![0u64; nt];

    // Q2a: Contiguous query bigram bonus (exact ordered query content-group pairs in t.seq)
    let mut g_idf_own = vec![0.0f32; plan.groups.len()];
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            if let Some(&df) = ix.t_own_stats.df.get(&qt.id) {
                let df = df as f32;
                let idf = ((ix.t_own_stats.n - df + 0.5) / (df + 0.5) + 1.0)
                    .ln()
                    .max(0.0)
                    .powf(p.idf_power);
                g_idf_own[g] = g_idf_own[g].max(idf * qt.weight);
            }
        }
    }
    let q_groups: Vec<usize> = raw_tokens(&q.text)
        .into_iter()
        .filter(|w| is_content(w))
        .filter_map(|w| ix.vocab.get(&w).and_then(|id| plan.group_of.get(&id).copied()))
        .collect();
    let mut q_bigrams: DetHashSet<(usize, usize)> = DetHashSet::default();
    for pair in q_groups.windows(2) {
        if pair[0] != pair[1] {
            q_bigrams.insert((pair[0], pair[1]));
        }
    }
    let g_surp = crate::retrieval::lexical::query_frame_ic::compute_query_surprisal_weights(ix, plan);
    let slot_kind = crate::retrieval::lexical::query_frame_ic::query_slot_witness_kind(&q.text);
    let has_surp_contrast = g_surp.iter().any(|&w| (w - 1.0).abs() > 0.02);
    let t_own_prep = prepare_plan(&plan, &ix.t_own_stats, p);
    let t_prep = prepare_plan(&plan, &ix.t_stats, p);
    for ti in collect_posting_candidates(&ix.t_postings, &plan, nt) {
        let t = &ix.turns[ti];
        let (mut sc_own, _, em_own) = bm25f_prepared(&ix.turn_own_bags[ti], &plan, &t_own_prep, &ix.t_own_stats, p, &fl);
        let (mut sc, m, _) = bm25f_prepared(&ix.turn_bags[ti], &plan, &t_prep, &ix.t_stats, p, &fl);
        if sc <= 0.0 && sc_own <= 0.0 {
            continue;
        }
        // #1 Meta-Discourse Scaffold Down-Weighting via Corpus-Induced Surprisal Gap:
        if has_surp_contrast && em_own != 0 {
            let mut sum_w = 0.0f32;
            let mut cnt = 0usize;
            for (gi, &gw) in g_surp.iter().enumerate().take(64) {
                if (em_own & (1u64 << gi)) != 0 {
                    sum_w += gw;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let mult = (sum_w / cnt as f32).clamp(0.72, 1.24);
                sc_own *= mult;
                sc *= mult;
            }
        }
        // #10 Syntactic Slot-Witness Presence Bonus:
        if slot_kind != crate::retrieval::lexical::query_frame_ic::SlotWitnessKind::None && sc_own > 0.0 {
            let sw_q16 = crate::retrieval::lexical::query_frame_ic::slot_witness_bonus_q16(ix, ti, slot_kind);
            if sw_q16 > 65_536 {
                let sw_mult = (sw_q16 as f32) / 65_536.0;
                sc_own *= sw_mult;
                sc *= sw_mult;
            }
        }
        let prox = p.enable_proximity && proximity_hit(t.seq.iter(), &plan, p.proximity_span);
        if prox {
            sc *= 1.0 + p.proximity_bonus;
            sc_own *= 1.0 + p.proximity_bonus;
        }
        if p.enable_carry && t.answers_question {
            sc *= p.answer_boost;
            sc_own *= p.answer_boost;
        } else if p.enable_carry && t.is_short_reply {
            sc *= 1.05;
            sc_own *= 1.05;
        }
        if pref_on && t.field == 0 {
            if t.speaker == "user" && turn_pos(ix, ti).map_or(false, |p| p <= 4) {
                sc *= p.preference_boost;
                sc_own *= p.preference_boost;
            }
        } else if persona_infer_on && (is_structural_personal_turn(ix, ti) || (t.is_habitual && !t.is_question)) {
            let mult = if is_structural_personal_turn(ix, ti) { 1.40 } else { 1.22 };
            sc *= mult;
            sc_own *= mult;
        }
        // Anderson ACT-R Fan Effect (rare personal-anchor lift on declarative turns):
        // When a turn matches low-fan (`df <= 3`) exact non-speaker query cues, amplify by
        // monotone ACT-R fan strength so one-of-a-kind personal anchors outrank high-fan hubs.
        if sc_own > 0.0 && !t.is_question && !t.is_backchannel && nt >= 16 {
            let q16_one = crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;
            let mut best_fan_q16: u64 = q16_one;
            for qt in &plan.terms {
                if !qt.exact_hit || !t.content.contains(&qt.id) {
                    continue;
                }
                let tdf = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
                if (1..=3).contains(&tdf) {
                    let f_q16 = crate::retrieval::graph::actr::actr_fan_strength_q16(tdf, nt as u32);
                    if f_q16 > best_fan_q16 {
                        best_fan_q16 = f_q16;
                    }
                }
            }
            if best_fan_q16 > q16_one {
                // Bounded gentle scale: up to ~1.09x for fan == 1 personal anchors.
                let extra = ((best_fan_q16 - q16_one) as f32) / (65_536.0 * 4.0);
                let fan_mult = 1.0 + extra.min(0.09);
                sc_own *= fan_mult;
                sc *= fan_mult;
            }
        }
        if sc_own > 0.0 {
            let dph_bonus = (crate::retrieval::lexical::dph::dph_bag_q16(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats) as f32) / 65536.0;
            // Pillar 7b: Clinchant-Gaussier Log-Logistic Information-Based (`LGD`) sharpening on short turns.
            let lgd_bonus = (crate::retrieval::lexical::lgd_q16::lgd_bag_q16(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats) as f32) / 65536.0;
            let add = 0.02 * dph_bonus + 0.018 * lgd_bonus;
            sc_own += add;
            sc += add;
        }
        t_own[ti] = sc_own;
        t_raw[ti] = sc;
        // Content-token count: TurnIx.content is DetHashSet<u32> of content term ids
        // (same field the s5 gate analysis used). This is NOT a UTF-8 char count.
        let content_tokens = t.content.len();
        let keep_full_carry = t.speaker == "assistant" || t.is_short_reply || content_tokens <= 8;
        let is_partner_q_reply = !t.is_question
            && !t.is_backchannel
            && (t.answers.is_some()
                || ix
                    .session_turns
                    .get(t.session)
                    .and_then(|m| {
                        m.iter()
                            .position(|&id| id == ti)
                            .and_then(|pos| pos.checked_sub(1).map(|p| m[p]))
                    })
                    .is_some_and(|prv| {
                        ix.turns[prv].is_question && ix.turns[prv].speaker != t.speaker
                    }));
        if carry_mode_is_parity() {
            t_turn_raw[ti] = sc_own + 1.0 * (sc - sc_own);
        } else if keep_full_carry {
            t_turn_raw[ti] = sc.max(sc_own);
        } else if is_partner_q_reply {
            t_turn_raw[ti] = sc_own + 0.62 * (sc - sc_own).max(0.0);
        } else {
            t_turn_raw[ti] = sc_own + 0.18 * (sc - sc_own).max(0.0);
        }
        if !q_bigrams.is_empty() && t.seq.len() >= 2 {
            let mut bigram_bonus = 0.0f32;
            let mut ppm_bonus = 0.0f32;
            let mut matched_bi: DetHashSet<(usize, usize)> = DetHashSet::default();
            for win in t.seq.windows(2) {
                let (ea, sa) = win[0];
                let (eb, sb) = win[1];
                if let (Some(&ga), Some(&gb)) = (
                    plan.group_of.get(&ea).or_else(|| plan.group_of.get(&sa)),
                    plan.group_of.get(&eb).or_else(|| plan.group_of.get(&sb)),
                ) {
                    if q_bigrams.contains(&(ga, gb)) && matched_bi.insert((ga, gb)) {
                        bigram_bonus += 0.28 * fl.w[t.field.min(2)] * (g_idf_own[ga] + g_idf_own[gb]);
                        // Order-2 PPM surprisal bonus when contiguous tokens form a rare collocation.
                        let surp_q16 = crate::retrieval::lexical::ppm_surprisal::ppm_collocation_surprise_q16(ix, ea, eb);
                        if surp_q16 > (4 * 65_536) {
                            ppm_bonus += 0.06 * ((surp_q16 as f32) / 65_536.0);
                        }
                    }
                }
            }
            if bigram_bonus > 0.0 {
                t_own[ti] += bigram_bonus + ppm_bonus;
                t_turn_raw[ti] += bigram_bonus + ppm_bonus;
            }
        }
        // #10 1-WL Local Neighborhood Structural Color alignment: mild tie-break boost when
        // interrogative query archetype aligns with structural slot shape (no shared words needed).
        let wl_c = crate::retrieval::lexical::ppm_surprisal::wl1_turn_color(ix, ti);
        let wl_align = !t.is_question
            && !t.is_backchannel
            && ((it.sniper_when && (wl_c & (1 << 6)) != 0)
                || (it.sniper_how_many && (wl_c & (1 << 7)) != 0)
                || (it.multi_hop && (wl_c & (1 << 11)) != 0 && t.field == 0));
        if wl_align && t_own[ti] > 0.0 {
            t_own[ti] *= 1.03;
            t_turn_raw[ti] *= 1.03;
        }
        t_mask[ti] = m;
        t_exact_mask[ti] = em_own;
    }

    apply_rung1_declarative_slot_specificity(
        ix,
        &plan,
        pref_on,
        it.is_event_sequence,
        &mut t_turn_raw,
    );
    apply_panini_karaka_slot_shape(ix, &q.text, &it, pref_on, &mut t_turn_raw);
    apply_step8_kelal_u_ferat(ix, &plan, &mut t_turn_raw);
    apply_step8_homeric_ring(ix, &plan, &mut t_turn_raw);
    apply_step9_method_of_loci(ix, &q.text, &mut t_turn_raw);
    if p.enable_deictic {
        apply_move3_heideltime_anchor(
            ix,
            q,
            it.multi_hop,
            &mut s_score,
            &mut t_turn_raw,
            &mut t_raw,
            &mut t_own,
        );
    }

    // Q16.16 Shannon Echo & Zero-Surprise Dampener (`Q16_QUARTER = 0.25x` turn prior):
    // quarantines high-echo subordinate deontic restatements (`rho_echo >= 0.70`),
    // zero-surprise polar acks (`IC_novel < 1.0 bit`), and unqueried synthetic time headers.
    let has_named_s2 = ix
        .turns
        .iter()
        .take(32)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    for ti in 0..ix.turns.len() {
        if t_own[ti] <= 0.0 && t_turn_raw[ti] <= 0.0 {
            continue;
        }
        if crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
            ix, ti, &q.text,
        ) {
            t_own[ti] *= 0.25;
            t_raw[ti] *= 0.25;
            t_turn_raw[ti] *= 0.25;
            continue;
        }
        if !has_named_s2
            && !plan.rare_entity_ids.is_empty()
            && plan.rare_entity_ids.iter().any(|rid| {
                ix.turns[ti].content.contains(rid)
                    || ix.turns[ti].entities.contains(rid)
                    || ix.turns[ti].feats.iter().any(|f| f.id == *rid)
            })
        {
            t_own[ti] *= 1.48;
            t_raw[ti] *= 1.48;
            t_turn_raw[ti] *= 1.48;
        }
        if !plan.excluded_ids.is_empty()
            && t_exact_mask[ti] == 0
            && ix.turns[ti].content.iter().any(|id| plan.excluded_ids.contains(id))
        {
            t_own[ti] *= 0.25;
            t_raw[ti] *= 0.25;
            t_turn_raw[ti] *= 0.25;
        }
    }

    ctx.s_score = s_score;
    ctx.t_own = t_own;
    ctx.t_raw = t_raw;
    ctx.t_turn_raw = t_turn_raw;
    ctx.t_mask = t_mask;
    ctx.t_exact_mask = t_exact_mask;
}
