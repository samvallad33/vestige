//! Stage 01: Query plan initialization, role field weights, and session-level BM25F + window pooling.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run<'a>(
    ix: &'a Index,
    q: &'a Query,
    p: &'a Params,
    top_k: usize,
) -> RankCtx<'a> {
    let plan = plan_query(ix, q, p);
    let it = &plan.intent;
    let pref_on = p.enable_preference && it.preference;
    // Phrase triggers for persona, quoted-user, and summary routing are gone.
    let persona_infer_on = false;
    let episodic_user_quote = false;
    let summary_hold = false;
    let user_role = p.enable_roles
        && it.user_attribution
        && !pref_on
        && !it.task_under_rules
        && !summary_hold;
    let fl = if p.enable_roles && it.assistant_attribution {
        Fields { w: [1.0, p.assistant_attribution_weight, 0.75], b: [p.b_user, p.b_assistant_attributed, 0.0] }
    } else if user_role {
        Fields { w: [p.user_attribution_weight, 0.35, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else if p.enable_roles && pref_on {
        Fields { w: [1.45, 0.55, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else {
        Fields { w: [1.0, 1.0, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    };
    let ns = ix.session_bags.len();

    // ---- Session level: whole-session BM25F + max-passage window pooling ----
    let mut s_base = vec![0.0f32; ns];
    let mut s_mask = vec![0u64; ns];
    let mut s_exact_mask = vec![0u64; ns];
    let s_prep = prepare_plan(&plan, &ix.s_stats, p);
    for si in collect_posting_candidates(&ix.s_postings, &plan, ns) {
        let (sc, m, em) = bm25f_prepared(&ix.session_bags[si], &plan, &s_prep, &ix.s_stats, p, &fl);
        s_base[si] = sc;
        s_mask[si] = m;
        s_exact_mask[si] = em;
    }
    let mut s_win = vec![0.0f32; ns];
    if p.enable_window {
        let w_prep = prepare_plan(&plan, &ix.w_stats, p);
        for wi in collect_posting_candidates(&ix.w_postings, &plan, ix.windows.len()) {
            let (si, a, b) = ix.windows[wi];
            let (mut sc, m, em) = bm25f_prepared(&ix.window_bags[wi], &plan, &w_prep, &ix.w_stats, p, &fl);
            if sc <= 0.0 {
                continue;
            }
            let members = &ix.session_turns[si][a..=b];
            if p.enable_proximity
                && proximity_hit(members.iter().flat_map(|&t| ix.turns[t].seq.iter()), &plan, p.proximity_span)
            {
                sc *= 1.0 + p.proximity_bonus;
            }
            if pref_on {
                let mut best_pref = 1.0f32;
                for &t in members {
                    let tr = &ix.turns[t];
                    if tr.field == 0 {
                        if is_structural_personal_turn(ix, t) {
                            best_pref = best_pref.max(p.preference_boost);
                        }
                    }
                }
                sc *= best_pref;
            } else if persona_infer_on && members.iter().any(|&t| is_structural_personal_turn(ix, t)) {
                sc *= 1.38;
            }
            if members.iter().any(|&t| ix.turns[t].is_standing_policy) {
                sc *= 1.8;
            }
            if sc > s_win[si] {
                s_win[si] = sc;
            }
            s_mask[si] |= m;
            s_exact_mask[si] |= em;
        }
    }
    if pref_on {
        for si in 0..ns {
            let mut best = 1.0f32;
            for &ti in &ix.session_turns[si] {
                let tr = &ix.turns[ti];
                if tr.field != 0 {
                    continue;
                }
                let focal_hit = tr.content.iter().any(|id| plan.focal_ids.contains(id))
                    || tr.feats.iter().any(|f| plan.focal_ids.contains(&f.id));
                if is_structural_personal_turn(ix, ti) && focal_hit {
                    best = best.max(p.preference_boost);
                }
            }
            if best > 1.0 {
                s_base[si] *= best;
            }
        }
    }
    let bmax = s_base.iter().cloned().fold(0.0f32, f32::max);
    let wmax = s_win.iter().cloned().fold(0.0f32, f32::max);
    let mut s_score: Vec<f32> = (0..ns)
        .map(|i| {
            let b = if bmax > 0.0 { s_base[i] / bmax } else { 0.0 };
            if p.enable_window && wmax > 0.0 {
                p.window_mix * (s_win[i] / wmax) + (1.0 - p.window_mix) * b
            } else {
                b
            }
        })
        .collect();

    // ---- Causal supersession: newest high-affinity session wins current-state queries ----
    if p.enable_recency && it.current_state {
        let ord = order(&s_score, &ix.session_ts);
        if let Some(&top) = ord.first() {
            let floor = s_score[top] * p.recency_band;
            let band: Vec<usize> = ord.iter().take(p.recency_depth).copied().filter(|&i| s_score[i] > 0.0 && s_score[i] >= floor).collect();
            if band.len() > 1 {
                if let Some(&newest) = band.iter().max_by_key(|&&i| ix.session_ts[i]) {
                    if ix.session_ts[newest] > 0 {
                        s_score[newest] *= p.recency_boost;
                    }
                }
            }
        }
    }
    let smax = s_score.iter().cloned().fold(0.0f32, f32::max);

    let nt = ix.turns.len();
    let eff_it = plan.intent.clone();
    RankCtx {
        ix,
        q,
        p,
        k: top_k.max(1),
        ns,
        nt,
        plan,
        pref_on,
        persona_infer_on,
        episodic_user_quote,
        user_role,
        fl,
        s_mask,
        s_exact_mask,
        s_score,
        smax,
        t_own: Vec::new(),
        t_raw: Vec::new(),
        t_turn_raw: Vec::new(),
        t_mask: Vec::new(),
        t_exact_mask: Vec::new(),
        t_scene: Vec::new(),
        t_scene_sess: Vec::new(),
        q_toks_eff: Vec::new(),
        past_cue_eff: false,
        seq_mode_eff: false,
        has_named_speakers_eff: false,
        is_monologue_corpus: false,
        has_compact_rule_register: false,
        cond_rule_turns: Vec::new(),
        eff_it,
        t_bridge: Vec::new(),
        t_bridge_hits: Vec::new(),
        lexical_smax: 0.0,
        lexical_tmax_sess: 0.0,
        lexical_tmax: 0.0,
        standing_rules: Vec::new(),
        s_policy_floored: Vec::new(),
        t_policy_floored: Vec::new(),
        s_pref_floored: Vec::new(),
        t_pref_floored: Vec::new(),
        initiative_content: Vec::new(),
        bridge_boost: Vec::new(),
        s_ord_base: Vec::new(),
        total_q_groups: 0,
        mask_limit: 0,
        final_session_indices: Vec::new(),
        strong_top_sessions: 0,
        max_per_session_top5: 3,
        turn_ts: Vec::new(),
        final_turn_indices: Vec::new(),
        base_q16: Vec::new(),
        q_group_cnt: 0,
        q_group_mask: 0,
        grounded_group_mask: 0,
        top3_union_exact: 0,
        has_ungrounded_rare_oov: false,
        deficit_a: false,
        deficit_b: false,
        torrens_locked_head: false,
    }
}
