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
    let t_own_prep = prepare_plan(&plan, &ix.t_own_stats, p);
    let t_prep = prepare_plan(&plan, &ix.t_stats, p);
    for ti in collect_posting_candidates(&ix.t_postings, &plan, nt) {
        let t = &ix.turns[ti];
        let (mut sc_own, _, em_own) = bm25f_prepared(&ix.turn_own_bags[ti], &plan, &t_own_prep, &ix.t_own_stats, p, &fl);
        let (mut sc, m, _) = bm25f_prepared(&ix.turn_bags[ti], &plan, &t_prep, &ix.t_stats, p, &fl);
        if sc <= 0.0 && sc_own <= 0.0 {
            continue;
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
        } else if persona_infer_on && is_structural_personal_turn(ix, ti) {
            sc *= 1.40;
            sc_own *= 1.40;
        }
        if sc_own > 0.0 {
            let dph_bonus = (crate::retrieval::lexical::dph::dph_bag_q16(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats) as f32) / 65536.0;
            let add = 0.02 * dph_bonus;
            sc_own += add;
            sc += add;
        }
        t_own[ti] = sc_own;
        t_raw[ti] = sc;
        // Content-token count: TurnIx.content is DetHashSet<u32> of content term ids
        // (same field the s5 gate analysis used). This is NOT a UTF-8 char count.
        let content_tokens = t.content.len();
        let keep_full_carry = t.speaker == "assistant" || t.is_short_reply || content_tokens <= 8;
        if carry_mode_is_parity() {
            t_turn_raw[ti] = sc_own + 1.0 * (sc - sc_own);
        } else if keep_full_carry {
            t_turn_raw[ti] = sc.max(sc_own);
        } else {
            t_turn_raw[ti] = sc_own + 0.18 * (sc - sc_own).max(0.0);
        }
        if !q_bigrams.is_empty() && t.seq.len() >= 2 {
            let mut bigram_bonus = 0.0f32;
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
                    }
                }
            }
            if bigram_bonus > 0.0 {
                t_own[ti] += bigram_bonus;
                t_turn_raw[ti] += bigram_bonus;
            }
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

    ctx.s_score = s_score;
    ctx.t_own = t_own;
    ctx.t_raw = t_raw;
    ctx.t_turn_raw = t_turn_raw;
    ctx.t_mask = t_mask;
    ctx.t_exact_mask = t_exact_mask;
}
