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
    let q_toks_s1 = raw_tokens(&q.text);
    let q_low_s1 = q.text.trim_start().to_ascii_lowercase();
    let has_named_s1 = ix
        .turns
        .iter()
        .take(32)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    // T58 Closed-class 2nd-person past attribution ("did you ...", "have you ...") on user/assistant chats:
    // weights assistant turns (`field 1`) without mutating public `QueryIntent::assistant_attribution`.
    // hardcode-scan: closed-class (2nd-person past attribution & retrospective recall cues)
    let eff_asst_attr = !has_named_s1
        && ix.turns.len() >= 24
        && (q_low_s1.contains("did you ")
            || q_low_s1.contains("have you ")
            || q_low_s1.contains("you recommend")
            || q_low_s1.contains("you suggest")
            || q_low_s1.contains("you advise")
            || q_low_s1.contains("you outline")
            || q_low_s1.contains("you propose")
            || q_low_s1.contains("you help")
            || q_low_s1.contains("you provided")
            || q_low_s1.contains("you told me")
            || q_low_s1.contains("you gave me")
            || q_low_s1.contains("you mentioned")
            || q_low_s1.contains("you shared")
            || q_low_s1.contains("remind me")
            || q_low_s1.contains("previous conversation")
            || q_low_s1.contains("previous chat")
            || q_low_s1.contains("we talked about")
            || q_low_s1.contains("we discussed"));
    let user_role = p.enable_roles
        && it.user_attribution
        && !eff_asst_attr
        && !pref_on
        && !it.task_under_rules
        && !summary_hold;
    let fl = if p.enable_roles && it.assistant_attribution {
        Fields { w: [0.65, p.assistant_attribution_weight, 0.75], b: [p.b_user, p.b_assistant_attributed, 0.0] }
    } else if p.enable_roles && eff_asst_attr {
        Fields { w: [1.10, p.assistant_attribution_weight, 0.75], b: [p.b_user, p.b_assistant_attributed, 0.0] }
    } else if user_role {
        Fields { w: [p.user_attribution_weight, 0.35, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else if p.enable_roles && pref_on {
        Fields { w: [1.45, 0.55, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else {
        Fields { w: [1.0, 1.0, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    };
    let is_when_participial = q_toks_s1
        .first()
        .is_some_and(|w| w == "when")
        && q_toks_s1
            .get(1)
            .is_some_and(|w| w.len() >= 5 && w.ends_with("ing"));
    let long_chat_s1 = ix.turns.len() >= 64;
    let policy_win_allowed = !eff_summary_request(&q.text)
        && !it.is_event_sequence
        && (!it.temporal || is_when_participial)
        && !it.sniper_how_many
        && !pref_on
        && !(long_chat_s1 && eff_asst_attr)
        && !crate::retrieval::step14_is_rec_request(&q.text)
        && (!long_chat_s1
            || (!q_low_s1.contains("did i ")
                && !q_low_s1.contains("did we ")
                && !q_low_s1.contains("have i ")
                && !q_low_s1.contains("was my ")
                && !q_low_s1.contains("were my ")
                && !q_low_s1.contains("between ")
                && !q_low_s1.starts_with("what is ")
                && !q_low_s1.starts_with("what was ")
                && !q_low_s1.starts_with("how many ")
                && !q_low_s1.starts_with("how long ")))
        && !q_toks_s1.windows(2).any(|w| w[0] == "order" && (w[1] == "in" || w[1] == "of"));
    let live_policy_turns: DetHashSet<usize> = if policy_win_allowed {
        ix.standing_rules
            .iter()
            .filter(|r| {
                !turn_is_superseded(ix, r.turn)
                    && (r.condition_words.is_empty()
                        || crate::retrieval::rule_condition_matches(ix, r, &plan, &q_toks_s1))
            })
            .map(|r| r.turn)
            .collect()
    } else {
        DetHashSet::default()
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
            if !live_policy_turns.is_empty()
                && members.iter().any(|&t| live_policy_turns.contains(&t))
            {
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
    // Impersonal 3rd-person haystack discount when the user asks an explicit 1st-person episodic/preference query.
    // hardcode-scan: closed-class (1st-person pronouns)
    let q_has_first_person = !has_named_s1
        && ns >= 16
        && ix.turns.iter().any(|t| t.field == 1)
        && !eff_asst_attr
        && !it.assistant_attribution
        && !it.task_under_rules
        && q.text
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '\u{2019}')
            .any(|w| {
                matches!(
                    w.to_ascii_lowercase().as_str(),
                    "i" | "my" | "me" | "mine" | "i'm" | "i've" | "i'd" | "i'll"
                        | "i\u{2019}m" | "i\u{2019}ve" | "i\u{2019}d" | "i\u{2019}ll"
                )
            });
    if q_has_first_person {
        for si in 0..ns {
            if s_score[si] <= 0.0 {
                continue;
            }
            let has_user_voice = ix.session_turns[si].iter().any(|&ti| {
                let tr = &ix.turns[ti];
                tr.field == 0
                    && (tr.is_standing_policy
                        || tr.is_personal_constraint
                        || tr.content.len() <= 10
                        || tr.body_for_headers
                            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '\u{2019}')
                            .any(|w| {
                                // hardcode-scan: closed-class (1st-person pronouns)
                                matches!(
                                    w.to_ascii_lowercase().as_str(),
                                    "i" | "my" | "me" | "mine" | "we" | "our" | "ours" | "us"
                                        | "i'm" | "i've" | "i'd" | "i'll" | "we're" | "we've"
                                        | "i\u{2019}m" | "i\u{2019}ve" | "i\u{2019}d" | "i\u{2019}ll"
                                        | "we\u{2019}re" | "we\u{2019}ve"
                                )
                            }))
            });
            if !has_user_voice {
                s_score[si] *= 0.62;
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
        implicit_disclosure_locked: false,
    }
}
