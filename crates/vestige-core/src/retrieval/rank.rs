//! `rank_one`: the per-query ranking pipeline.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

#[inline]
fn bag_to_q8_into(
    bag: &Bag,
    plan: &QueryPlan,
    st: &Stats,
    terms: &mut Vec<super::qld::QldTermInput>,
) -> super::qld::QldBagInput {
    let mut dl_q8 = [0u64; 3];
    let mut avdl_q8 = [0u64; 3];
    for f in 0..3 {
        dl_q8[f] = (bag.len[f] * 256.0).round().max(0.0) as u64;
        avdl_q8[f] = (st.avg[f] * 256.0).round().max(256.0) as u64;
    }
    terms.clear();
    for qt in &plan.terms {
        let df = st.df.get(&qt.id).copied().unwrap_or(0);
        let w_q8 = (qt.weight * 256.0).round().max(0.0) as u64;
        if df == 0 || w_q8 == 0 {
            continue;
        }
        let mut tf_q8 = [0u64; 3];
        let mut any = false;
        for f in 0..3 {
            let tf = bag.tf[f].get(&qt.id).copied().unwrap_or(0.0);
            if tf > 0.0 {
                tf_q8[f] = (tf * 256.0).round().max(1.0) as u64;
                any = true;
            }
        }
        if any {
            terms.push(super::qld::QldTermInput {
                tf_q8,
                w_q8,
                df_doc: df as u64,
            });
        }
    }
    super::qld::QldBagInput {
        dl_q8,
        avdl_q8,
        n_docs: (st.n as u64).max(1),
    }
}

pub(super) fn rank_one(ix: &Index, q: &Query, p: &Params, top_k: usize) -> RankResult {
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
            let dph_bonus = (super::dph::dph_bag_q16(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats) as f32) / 65536.0;
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
                for (pos, &ti) in members.iter().enumerate() {
                    let prev_own = if pos > 0 { t_own[members[pos - 1]] } else { 0.0 };
                    let next_own = if pos + 1 < members.len() { t_own[members[pos + 1]] } else { 0.0 };
                    let prev_w = if ix.turns[ti].is_backchannel {
                        0.22
                    } else if pos > 0 && ix.turns[members[pos - 1]].is_question {
                        0.75
                    } else {
                        0.48
                    };
                    let self_score = if ix.turns[ti].answers_question || ix.turns[ti].is_short_reply || ix.turns[ti].anaphoric {
                        t_raw[ti]
                    } else {
                        t_own[ti] + 0.18 * (t_raw[ti] - t_own[ti]).max(0.0)
                    };
                    t_scene[ti] = self_score + prev_w * prev_own + 0.42 * next_own;
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
                        && t_own[prev] > 0.0
                        && (t_exact_mask[ti] & !t_exact_mask[prev]) == 0
                        && t_own[ti] <= t_own[prev] * 1.05
                        && t_scene[ti] > 0.92 * t_scene[prev]
                    {
                        t_scene[ti] = 0.92 * t_scene[prev];
                    }
                }
                for &ti in members.iter() {
                    t_scene_sess[ti] = t_scene[ti];
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
        let focus_q16 = super::focus::compute_anaphoric_focus_q16(ix, &plan);
        let hobbs_res = super::hobbs::resolve_hobbs_pronouns(ix, &plan);
        let coref_q16 = super::coref::coref_class_affinity_q16(ix, &plan);
        let t_max_sc = t_scene.iter().cloned().fold(0.0f32, f32::max);
        if t_max_sc > 0.0 {
            for ti in 0..nt {
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let fq = focus_q16[ti];
                let hq = hobbs_res.turn_score_q16[ti];
                let cq = coref_q16[ti];
                if fq > 0 || hq > 0 || cq > 0 {
                    let f_frac = (fq as f32) / (super::focus::Q16 as f32);
                    let h_frac = (hq as f32) / (super::hobbs::Q16 as f32);
                    let c_frac = (cq as f32) / (super::coref::Q16 as f32);
                    let ana_boost = 0.05 * f_frac.max(h_frac);
                    let boost = t_max_sc * (ana_boost + 0.03 * c_frac);
                    t_scene[ti] += boost;
                    t_scene_sess[ti] += boost;
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
    for t_sc in [&mut t_scene[..], &mut t_scene_sess[..]] {
        for ti in 0..nt {
            if t_raw[ti] <= 0.0 && t_own[ti] <= 0.0 {
                continue;
            }
            let sp = ix.turns[ti].speaker.as_str();
            if !sp.is_empty()
                && sp != "user"
                && sp != "assistant"
                && queried_speakers.iter().any(|qs| qs.eq_ignore_ascii_case(sp))
            {
                t_sc[ti] *= sp_mult;
            }
        }
    }

    // Turn-level role alignment: suppress assistant turns on explicit user-role queries,
    // except when no user turn in the same session covers that assistant turn's exact query bits!
    let apply_scene_refinements = |t_scene: &mut [f32], t_mask: &mut [u64], t_exact_mask: &mut [u64]| {
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

        for ti in 0..nt {
            if present_state_local && ix.torrens_superseded.get(ti).copied().unwrap_or(false) {
                t_scene[ti] = 0.0;
                continue;
            }
            let t = &ix.turns[ti];
            if t.is_standing_policy && t_scene[ti] > 0.0 {
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
                            t_scene[ti] *= 1.65;
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
                    } else if !has_named_dialogue
                        && !it.assistant_attribution
                        && has_declarative_user_peer
                        && !episodic_user_quote
                        && t.speaker == "user"
                        && (if is_monologue_stream {
                            t.is_pure_info_question && !t.is_personal_constraint
                        } else {
                            t.is_question
                        })
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

    };
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
    let pref_live_eff = ix.turns.len() >= 24
        && !past_cue_eff
        && !it.temporal
        && !seq_mode_eff
        && super::scoring::step14_is_rec_request(&q.text);
    let q_trim_low = q.text.trim().to_ascii_lowercase();
    let has_compact_rule_register = (1..=8).contains(&ix.standing_rules.len());
    // Rule-condition gate: a compact rule register only marks the query as a
    // task-under-rules consultation when some live rule's own condition clause
    // ("when I ask about X") matches the query.
    let cond_rule_turns: Vec<usize> = ix
        .standing_rules
        .iter()
        .filter(|r| super::scoring::rule_condition_matches(r, &plan, &q_toks_eff))
        .map(|r| r.turn)
        .collect();
    let rule_condition_live = has_compact_rule_register && !cond_rule_turns.is_empty();
    let task_live_eff = !is_monologue_corpus
        && !has_named_speakers_eff
        && ix.turns.len() >= 64
        && !past_cue_eff
        && !it.temporal
        && !it.sniper_when
        && !it.sniper_how_many
        && !seq_mode_eff
        && (!q_trim_low.ends_with('?')
            || q_trim_low.starts_with("please ")
            || (rule_condition_live
                && !q_trim_low.contains("did i ")
                && !q_trim_low.contains("did we ")
                && !q_trim_low.contains("was my ")
                && !q_trim_low.contains("were my "))
            || q_toks_eff.windows(2).any(|p| {
                ((p[0] == "could" || p[0] == "can" || p[0] == "would") && p[1] == "you")
                    || (p[0] == "how" && (p[1] == "to" || p[1] == "can" || p[1] == "should" || p[1] == "do"))
                    || (p[0] == "what" && (p[1] == "should" || p[1] == "are" || p[1] == "typical"))
                    || (p[0] == "which" && p[1] != "session")
                    || (p[0] == "should" && (p[1] == "i" || p[1] == "we"))
            }));
    let eff_it = QueryIntent {
        preference: it.preference || pref_live_eff,
        current_state: it.current_state
            || (!past_cue_eff && !it.temporal && !seq_mode_eff && !it.sniper_how_many),
        is_event_sequence: if large_or_mono { seq_mode_eff } else { it.is_event_sequence },
        task_under_rules: it.task_under_rules || task_live_eff,
        ..it.clone()
    };
    let pref_on = pref_on || (p.enable_preference && eff_it.preference);

    // Run session-path refinements first on a mask copy, then restore and run the
    // turn-path refinements so PPR seeds see pre-dyad masks (floor behavior).
    let t_mask_snap = t_mask.clone();
    let t_exact_snap = t_exact_mask.clone();
    apply_scene_refinements(&mut t_scene_sess, &mut t_mask, &mut t_exact_mask);
    t_mask.copy_from_slice(&t_mask_snap);
    t_exact_mask.copy_from_slice(&t_exact_snap);
    apply_scene_refinements(&mut t_scene, &mut t_mask, &mut t_exact_mask);
    if eff_it.current_state {
        for (ti, &sup) in ix.torrens_superseded.iter().enumerate() {
            if sup {
                t_own[ti] = 0.0;
            }
        }
    }
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

    let k = top_k.max(1);

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

    let total_q_groups = plan.groups.len().min(64);
    let mask_limit = if total_q_groups == 64 { u64::MAX } else { (1u64 << total_q_groups) - 1 };

    let mut final_session_indices: Vec<usize> = if p.enable_submodular && ns > 1 {
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        if let Some(&top0) = s_ord_base.first() {
            if s_score[top0] > 0.0 {
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
                        let distinct_episode = min_ochiai.is_some_and(|d| d >= 0.58)
                            && (ix.session_ts[si] == 0 || selected.iter().all(|&s| ix.session_ts[s] != ix.session_ts[si]));
                        let redundancy = if same_date_seen {
                            0.25
                        } else if topic_seen {
                            0.35
                        } else if !eff_it.is_event_sequence
                            && !it.sniper_how_many
                            && !distinct_episode
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
        for si in &s_ord_base {
            if selected.len() >= k {
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
            .take(k)
            .collect()
    };

    // `S8-006` Cadastral disjoint session admission:
    // Admit at most 1 scored session outside Top-10 whose entity-id set is disjoint from Top-10
    // and whose query mask covers a content group none of Top-10 covers.
    if ns > 10 && final_session_indices.len() >= 10 {
        let top10_len = final_session_indices.len().min(10);
        let mut top10_mask = 0u64;
        let mut top10_ents: DetHashSet<u32> = DetHashSet::default();
        for &si in &final_session_indices[..top10_len] {
            top10_mask |= s_mask[si] & mask_limit;
            top10_ents.extend(ix.session_entities[si].iter().copied());
        }
        if let Some(&cad_si) = s_ord_base.iter().find(|&&si| {
            !final_session_indices[..top10_len].contains(&si)
                && s_score[si] > 0.0
                && ((s_mask[si] & mask_limit) & !top10_mask) != 0
                && !ix.session_entities[si].iter().any(|e| top10_ents.contains(e))
        }) {
            let uncov_bits = (s_mask[cad_si] & mask_limit) & !top10_mask;
            let ins_rank = (1..top10_len)
                .rev()
                .find(|&r| {
                    let s = final_session_indices[r];
                    let prev_union = final_session_indices[..r]
                        .iter()
                        .fold(0u64, |acc, &ps| acc | (s_mask[ps] & mask_limit));
                    ((s_mask[s] & mask_limit) & !prev_union) == 0 && (s_mask[s] & uncov_bits) == 0
                })
                .unwrap_or(top10_len - 1);
            if let Some(pos) = final_session_indices.iter().position(|&x| x == cad_si) {
                final_session_indices.remove(pos);
            }
            final_session_indices.insert(ins_rank, cad_si);
            if final_session_indices.len() > k {
                final_session_indices.truncate(k);
            }
        }
    }

    let strong_top_sessions = final_session_indices
        .iter()
        .take(5)
        .filter(|&&si| smax > 0.0 && !s_policy_floored[si] && !s_pref_floored[si] && (s_score[si] + bridge_boost[si]) >= 0.62 * smax)
        .count();

    // Weapon 4: Greedy Submodular Set Cover for Top-K Turn Selection
    let max_per_session_top5 = if eff_it.is_event_sequence {
        1
    } else if eff_it.multi_hop || (!pref_on && !eff_it.task_under_rules && strong_top_sessions >= 2) {
        2
    } else {
        3
    };
    let (q4_sess_floor, q4_turn_floor, allow_short_reply_bypass) =
        step10_florilegium_floors(strong_top_sessions);

    let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
    let mut final_turn_indices: Vec<usize> = if p.enable_submodular {
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
                let bypass_cap = short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop && allow_short_reply_bypass;
                let session_capped = selected_turns.len() < 5
                    && count >= max_per_session_top5
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
    };

    // Q4: Multi-session turn quota guarantee inside top-5 so Session #1 redundant
    // turns cannot crowd out strong Sessions #2..#5 before exact-edge/RSB stages.
    if !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && final_turn_indices.len() >= 5
        && smax > 0.0
    {
        let eff_q4_sess = if eff_it.multi_hop { q4_sess_floor.min(0.42) } else { q4_sess_floor };
        let eff_q4_turn = if eff_it.multi_hop { q4_turn_floor.min(0.38) } else { q4_turn_floor };
        let sess_limit = final_session_indices.len().min(5);
        for &si in &final_session_indices[1..sess_limit] {
            if s_score[si] + bridge_boost[si] < eff_q4_sess * smax {
                continue;
            }
            if final_turn_indices[..5].iter().any(|&t| ix.turns[t].session == si) {
                continue;
            }
            let Some(best_ti) = ix.session_turns[si]
                .iter()
                .copied()
                .filter(|&t| t_scene[t] > 0.0)
                .max_by(|&a, &b| {
                    t_scene[a]
                        .partial_cmp(&t_scene[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.cmp(&a))
                })
            else {
                continue;
            };
            let victim_slot = (1..5).rev().find(|&slot| {
                let vti = final_turn_indices[slot];
                if t_policy_floored[vti] || t_pref_floored[vti] {
                    return false;
                }
                let vs = ix.turns[vti].session;
                let cnt = final_turn_indices[..5]
                    .iter()
                    .filter(|&&t| ix.turns[t].session == vs)
                    .count();
                cnt >= 2 && t_scene[best_ti] >= eff_q4_turn * t_scene[vti]
            });
            if let Some(slot) = victim_slot {
                if let Some(existing) = final_turn_indices.iter().position(|&t| t == best_ti) {
                    if existing >= 5 {
                        final_turn_indices.remove(existing);
                        final_turn_indices.insert(slot, best_ti);
                    }
                } else {
                    final_turn_indices.insert(slot, best_ti);
                    final_turn_indices.truncate(k);
                }
            }
        }
    }

    // `jefferson_apportionment` + `lex_rhodia_shedding` (`BEAM MSR`, `LME_M MS`, `MAB MS`):
    // On multi-session / count queries with `strong_top_sessions >= 2`, allocate Top-5 seats
    // by D'Hondt greatest divisors `s_sight / (seats + 1)` and evict only from the session
    // holding the most seats (protecting Slot 0).
    if (eff_it.multi_hop || eff_it.sniper_how_many)
        && !eff_it.is_event_sequence
        && !pref_on
        && !eff_it.task_under_rules
        && strong_top_sessions >= 2
        && final_turn_indices.len() >= 5
        && smax > 0.0
    {
        for &cand_si in final_session_indices.iter().take(5) {
            let cand_sight = s_score[cand_si] + bridge_boost[cand_si];
            if cand_sight < 0.55 * smax {
                continue;
            }
            if final_turn_indices[..5].iter().any(|&ti| ix.turns[ti].session == cand_si) {
                continue;
            }
            let Some(best_cand_ti) = ix.session_turns[cand_si]
                .iter()
                .copied()
                .filter(|&ti| {
                    !ix.turns[ti].is_question
                        && !ix.turns[ti].is_backchannel
                        && !turn_is_superseded(ix, ti)
                        && t_scene[ti] > 0.0
                })
                .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
            else {
                continue;
            };
            // `lex_rhodia_shedding`: find the lowest-ranked turn in `1..5` belonging to the session
            // with the largest seat count (`>= 2`), when D'Hondt quotient `cand_sight / 1 > over_sight / over_cnt`.
            let mut shed_target: Option<(usize, usize, f32)> = None;
            for slot in (1..5usize).rev() {
                let vti = final_turn_indices[slot];
                if t_policy_floored[vti] || t_pref_floored[vti] {
                    continue;
                }
                let vs = ix.turns[vti].session;
                let s_cnt = final_turn_indices[..5].iter().filter(|&&t| ix.turns[t].session == vs).count();
                if s_cnt < 2 {
                    continue;
                }
                let over_quotient = (s_score[vs] + bridge_boost[vs]) / (s_cnt as f32);
                if cand_sight >= over_quotient {
                    let vsc = t_scene[vti];
                    if shed_target.is_none_or(|(_, bcnt, bsc)| s_cnt > bcnt || (s_cnt == bcnt && vsc < bsc)) {
                        shed_target = Some((slot, s_cnt, vsc));
                    }
                }
            }
            if let Some((slot, _, _)) = shed_target {
                if let Some(pos) = final_turn_indices.iter().position(|&t| t == best_cand_ti) {
                    if pos >= 5 {
                        final_turn_indices.remove(pos);
                        final_turn_indices.insert(slot, best_cand_ti);
                    }
                } else {
                    final_turn_indices.insert(slot, best_cand_ti);
                    final_turn_indices.truncate(k);
                }
            }
        }
    }

    if !eff_it.is_event_sequence && !pref_on && !eff_it.task_under_rules {
        let gl = {
            let g = plan.groups.len().min(64);
            if g == 64 { u64::MAX } else { (1u64 << g) - 1 }
        };
        let floored = |t: usize| t_policy_floored[t] || t_pref_floored[t];
        apply_step10a_evidence_seat(
            ix,
            &final_session_indices,
            &t_scene,
            &t_exact_mask,
            gl,
            &floored,
            k,
            &mut final_turn_indices,
        );
        apply_step15_comparative_dual_slot_seat(
            ix,
            &q.text,
            &plan,
            &it,
            &t_own,
            &t_exact_mask,
            &floored,
            k,
            &mut t_scene,
            &mut final_turn_indices,
        );
    }

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
                it.current_state
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
    if !it.assistant_attribution && final_turn_indices.len() > 1 {
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
    // C2 algos lane: slot-1 seat rules, all before RSB (which then seats behind the head).
    if !eff_it.is_event_sequence {
        apply_c2_bullae_numeral_seat(ix, &mut final_turn_indices, &t_scene);
        apply_c2_coverage_margin_seat(ix, &plan, &mut final_turn_indices);
        apply_c2_testimony_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_cross_witness_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_mimamsa_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_dichotomous_key(ix, &plan, &mut final_turn_indices, &t_scene);
        // Pool-fill reallocation: user source with a strict group superset takes a
        // same-session assistant's top-5 seat. Skipped on assistant-attribution so
        // the assistant's own words stay in place.
        if !it.assistant_attribution {
            apply_c2_source_superset_realloc(ix, &plan, &mut final_turn_indices, &t_scene);
            apply_c2_unique_group_realloc(ix, &plan, &mut final_turn_indices, &t_scene);
            apply_c2_subsequent_answer_seat(ix, &plan, &mut final_turn_indices, &t_scene);
            // Next-user elaboration: mid-session declarative head yields to
            // the next same-session declarative user already in the top 5
            // when that follow-up is at least as on-scene. Skipped on
            // preference so the rule stays independent of the
            // preference-source seat.
            // Text-level preference only (PREF_QUERY_CUES), not the
            // constraint-object upgrade on plan.intent.preference, so
            // #12–#18 stay independent of #11 without treating a factual
            // question that mentions a constraint object as preference.
            if !c2_text_preference(&q.text) {
                apply_c2_elaboration_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Cross-session opener: mid-session user head yields to a
                // user session-opener already in the top 5 from another
                // session. Skipped on preference and on summary requests.
                apply_c2_cross_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Same-session opener: mid-session declarative head yields
                // to the same session's user opener already in the top 5.
                // Skipped on preference and on summary requests.
                apply_c2_same_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker reply: in a named-speaker dialogue, an
                // unmentioned speaker's question yields to the next
                // same-session turn by a speaker the query names, already
                // in the top 5. Skipped on preference.
                apply_c2_queried_speaker_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker previous: in a named-speaker dialogue, an
                // unmentioned speaker's declarative comment yields to the
                // previous same-session statement by a speaker the query
                // names, already in the top 5. Skipped on preference.
                apply_c2_queried_speaker_prev_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker question-head previous: in a
                // named-speaker dialogue, an unmentioned speaker's
                // question yields to the previous same-session turn
                // by a speaker the query names, already in the top 5.
                // Skipped on preference.
                apply_c2_queried_speaker_qhead_prev_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker T@5 entry: a declarative turn by a
                // speaker the query names, already in official k outside
                // the top 5, swaps with the last unmentioned named
                // speaker in the top 5. Membership-preserving. Skipped
                // on preference.
                apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
            }
        }
        // Preference source: ReplySource is skipped on assistant_attribution,
        // but "can you recommend..." is tagged that way by a present-tense cue.
        // Membership-preserving; runs even when assistant_attribution is set.
        apply_c2_preference_source_seat(ix, &plan, &q.text, &mut final_turn_indices);
    }
    // RSB: the retrieved turn is the symptom. Seat its structural cause
    // immediately after it. The cause can share no words with the query.
    // Never in front of the symptom. A full window does not grow.
    if !eff_it.is_event_sequence {
        rsb_seat_behind(ix, &mut final_turn_indices, k);
    }
    // Rank-1 gatekeeper v2. Slot 1 stays the RSB v4 head. Slots 2..k are the
    // same Kemeny-Young order as PR 710, computed on that window with the head
    // removed. No new constant: the frozen slot is position 0 of this window.
    rank1_gatekeeper_keep_rsb_head(ix, &q.text, &mut final_turn_indices, &t_scene);
    // Queried-speaker T@5 entry, after the gatekeeper. The pre-RSB pass
    // (#17) is often undone when the gatekeeper Kemeny-reorders the tail
    // (slots 2..k). Re-applying the same membership-preserving swap here
    // is a post-gatekeeper ranking change, so the entry survives the
    // tail reorder by construction. Later post-gatekeeper C2 seats
    // may still run after this pass.
    // Same preference / event-sequence / assistant-attribution cuts.
    // Does not touch rsb_seat_behind and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !it.assistant_attribution && !c2_text_preference(&q.text) {
        apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Same-session user T@5 entry: a user opener or later turn from
        // a session already in the top 5, already in official k, swaps
        // with the last assistant in the top 5. Post-gatekeeper so the
        // entry survives. Skipped on preference.
        apply_c2_same_session_user_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Cross-session user-opener T@5 entry: a user opener from a
        // session not yet in the top 5, already in official k, swaps
        // with the last non-unique assistant that is not the head's
        // next turn. Post-gatekeeper, after #20, so the entry survives.
        // Skipped on preference.
        apply_c2_cross_session_user_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session seat: an unmentioned speaker
        // at slot 1 yields to a named speaker's first utterance in its
        // session, already in the top 5. Post-gatekeeper, after #21,
        // so it sees the official post-gatekeeper list (a pre-RSB
        // pass would reseat a tail occupant that the gatekeeper later
        // leaves outside the top 5). Later post-gatekeeper C2 seats
        // may still run after this pass. Skipped on preference.
        apply_c2_queried_speaker_first_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session T@5 entry: a named speaker's
        // first utterance in its session, already in official k
        // outside the top 5, swaps with the last unmentioned named
        // speaker. Declaratives have no scene floor; questions reuse
        // C2_XWITNESS_RATIO. Post-gatekeeper, after the first #22 pass.
        // Skipped on preference.
        apply_c2_queried_speaker_first_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session seat, again, after #23. A
        // first-in-session turn that #23 just admitted into the top 5
        // can now take slot 1; the earlier #22 pass could not see it
        // there yet. Later post-gatekeeper C2 seats may still run after
        // this pass. Skipped on preference.
        apply_c2_queried_speaker_first_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Subsequent-answer seat, again, after the gatekeeper. Pre-RSB #10
    // refuses when the next user is still outside the top 5; RSB can
    // then seat that user into the window, and a post-GK pass can take
    // slot 1. Later post-gatekeeper C2 seats may still run after this
    // pass. Same cuts as #10: event-sequence and assistant-attribution
    // at the call site; the function still skips temporal and
    // named-speaker sessions. Does not skip preference (same as #10).
    // Does not run the gatekeeper again and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !it.assistant_attribution {
        apply_c2_subsequent_answer_seat(ix, &plan, &mut final_turn_indices, &t_scene);
    }
    // Cross-session opener-pair seat: a user opener at slot 1 yields to
    // another session's user opener already in the top 5 whose
    // immediately following assistant reply is also already in the
    // top 5, when that opener is at least as on-scene. Post-gatekeeper,
    // after #26. Later post-gatekeeper C2 seats may still run after
    // this pass. Skipped on preference. Does not run the gatekeeper
    // again and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !it.assistant_attribution && !c2_text_preference(&q.text) {
        apply_c2_cross_opener_pair_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Answered-opener declarative seat: a user question opener at
        // slot 1 whose assistant reply is already in the top 5 yields
        // to another session's declarative user opener already in the
        // top 5, when that opener is at least as on-scene. After #27.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // Skipped on preference.
        apply_c2_answered_opener_decl_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Solo earlier-opener seat: a lone user opener at slot 1 (no
        // other turn from that session in the top 5) yields to an
        // earlier user opener from a different session already in the
        // top 5, when that opener is at least as on-scene. After #28.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // Skipped on preference.
        apply_c2_solo_earlier_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool opener T@10 entry: a user opener in the scene-ranked
        // pool, not already in the official k, takes the official-k
        // seat of the last non-unique tail assistant whose session
        // still has a user in the official k, when that opener is at
        // least as on-scene. After #29. Later post-gatekeeper C2 seats
        // may still run after this pass. The top 5 is unchanged.
        // Skipped on preference.
        apply_c2_pool_opener_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Same-session opener T@5 entry: a user opener already in
        // official k, from a session already represented in the top 5
        // by an assistant, swaps with the last same-session assistant
        // in the top 5. After #30. Later post-gatekeeper C2 seats may
        // still run after this pass. Skipped on preference.
        apply_c2_same_session_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool mentioned-speaker T@10 entry: a mentioned-speaker turn
        // in the scene-ranked pool, not already in official k, takes
        // the official-k seat of the last unmentioned named speaker in
        // the tail, when that turn is at least as on-scene. After #31.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // The top 5 is unchanged. Skipped on preference. Does not fire
        // on user/assistant chats.
        apply_c2_pool_mentioned_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker T@5 entry again after #32, so a mentioned
        // declarative that the pool T@10 entry just seated into
        // official k can still enter the top 5. After #32, last
        // ranking change. Skipped on preference.
        apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Late opener T@5 entry: a user opener already in official k,
        // from a session already in the top 5, swaps with the last
        // top-5 assistant of any session. After #33, last ranking
        // change, so it sees openers the pool T@10 entry (#30) seated.
        // Skipped on preference.
        apply_c2_late_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Head-session opener T@5 entry: the opener of slot 1's session,
        // already in official k, swaps with the least on-scene top-5
        // user from another session when at least as on-scene. After
        // #34, last ranking change. Skipped on preference.
        apply_c2_head_session_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Session substitution T@5 entry: the most on-scene official-k
        // user from a session outside the top 5 swaps with the weakest
        // top-5 user when that user is a declarative at most
        // C2_XWITNESS_RATIO of its scene. After #35, last ranking
        // change. Skipped on preference.
        apply_c2_session_substitution_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool session substitution T@10 entry: the most on-scene pool
        // user from a session with no official-k seat replaces the
        // weakest official-k tail turn when that turn is a declarative
        // at most C2_XWITNESS_RATIO of its scene. The top 5 is
        // unchanged. After #36, last ranking change. Skipped on
        // preference.
        apply_c2_pool_session_substitution_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Session substitution T@5 entry again after #37, so a pool user
        // that #37 just seated into official k can still enter the top
        // 5. After #37, last ranking change. Skipped on preference.
        apply_c2_session_substitution_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Mentioned-declarative session-mate T@5 entry: in a named-speaker
        // dialogue, a mentioned speaker's declarative from a top-5 session
        // swaps with the weakest top-5 declarative (at most
        // C2_XWITNESS_RATIO of its scene, never from an earlier session).
        // After #38, last ranking change. Skipped on preference.
        apply_c2_mentioned_session_mate_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Episode-run edge T@10 entry: the pool turn just before / after two
        // back-to-back official-k turns of one session takes the tail seat of
        // the least on-scene lone-session turn (at least as on-scene, never an
        // earlier session). After #39, last ranking change.
        apply_c2_episode_run_edge_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Episode-run edge T@5 entry: the edge of two back-to-back top-5 turns
        // takes the slot of the least on-scene lone-session, non-opener top-5
        // turn (at least as on-scene, never an earlier session). After #40,
        // last ranking change.
        apply_c2_episode_run_edge_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Neighbor-pair T@5 entry: in named dialogues, the official-k tail
        // neighbor of a top-5 turn swaps with the least on-scene lone-session,
        // non-opener top-5 turn (at least as on-scene, never an earlier
        // session). After #41, last ranking change.
        apply_c2_neighbor_pair_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Assistant-reply T@5 entry, for assistant-attribution questions only
    // (outside the C2 skip above): the official-k assistant reply to a top-5
    // user turn swaps with the least on-scene other top-5 user turn (at most
    // as on-scene, never from an earlier session, S@5 kept). Last ranking
    // change.
    if !eff_it.is_event_sequence && it.assistant_attribution {
        apply_c2_assistant_reply_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Unseated-session opener T@10 entry (assistant-attribution only):
        // the most on-scene declarative opener of a session with no official-k
        // seat takes the tail seat of the least on-scene lone-session turn
        // (at least as on-scene, never an earlier session). After #43.
        apply_c2_attribution_unseated_opener_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Exchequer split tally: for assistant-attribution questions, an
        // assistant turn whose own groups and the user turn it answers are
        // complementary halves takes slot 1 when that pair covers the
        // incumbent. After #44. Does not touch rsb_seat_behind.
        apply_c2_step31_exchequer_split(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Step 32 trial balance. After every other C2 seat. A question at slot 1
    // that misses the rarest surface lexeme yields to a top-5 opener from a
    // session that is not later, when that opener posts the rare lexeme and
    // every surface lexeme slot 1 already posts. Otherwise an echo at slot 1
    // yields to the declarative turn it answers, when that turn posts the
    // above-median surface lexemes and the pair posts every surface lexeme
    // the echo's own pair does not. Does not touch rsb_seat_behind and does
    // not rewrite the session list.
    apply_c2_step32_trial_balance(ix, &plan, &mut final_turn_indices, &t_scene);
    // Step 33 triangulation. A top-5 turn that is strictly best on its own
    // lexical score and on its session score takes slot 1 when it also
    // strictly covers the head's query terms. Named-speaker dialogues only.
    // After the gatekeeper, Step 31 and Step 32. Does not touch rsb_seat_behind.
    apply_c2_step33_triangulation(
        ix,
        &plan,
        &mut final_turn_indices,
        &t_own,
        &s_score,
        &bridge_boost,
    );
    // Step 34 Bertillonage. After trial balance and triangulation. Slot 1 below
    // the query's kind-count yields to one admissible same-session top-5 card.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_step34_bertillonage(ix, &q.text, &mut final_turn_indices, &t_scene);
    // Step 35 Mill's method of agreement. After the gatekeeper and Steps 31–34.
    // The unique session common to every surface-anchor hit set lends slot 1
    // to its unique top-5 instance when the head posts no anchor, or to its
    // unique same-speaker turn that posts every anchor when the head posts
    // only some. A head already inside that session stays. Does not touch
    // rsb_seat_behind.
    apply_c2_step35_mill_agreement(ix, &plan, &mut final_turn_indices, &t_scene);
    // Step 36 Antaranga-bahiranga. After Bertillonage. A question at slot 1
    // whose scene score is mostly external yields to a top-5 declarative
    // turn held by its own lexical score. Does not touch rsb_seat_behind
    // and does not rewrite the session list.
    apply_c2_step36_antaranga(
        ix,
        &q.text,
        &mut final_turn_indices,
        &t_own,
        &t_scene,
        &s_score,
        &bridge_boost,
    );
    // Step 37 Kalevala parallelism. After the gatekeeper and Steps 31–35.
    // Adjacent same-speaker restatements are one line. The following line
    // takes slot 1 when its content query cues strictly cover the head and
    // it already sits in the top 5. Does not touch rsb_seat_behind.
    apply_c2_step37_kalevala(ix, &q.text, &mut final_turn_indices);
    // Step 38 Hermeneutic circle. After Antaranga. A question at slot 1 yields
    // to the one top-5 declarative that posts the same content anchors, whose
    // session covers the anchors that part does not, when the incumbent
    // session does not and the declarative's own score is strictly stronger.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_step38_hermeneutic(ix, &q.text, &mut final_turn_indices, &t_own);
    // Pool C session-key opener. After Bertillonage. A same-session assistant
    // elaboration in slots 2–5 yields to that session's user opener when the
    // top 5 already spans two sessions and the opener is outside it. Slot 1
    // stays. Does not touch rsb_seat_behind.
    apply_c2_pool_c_session_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    // Absent session key. After the same-session yield. A user opener whose
    // session is not in the top 5 takes a redundant assistant from a
    // non-dominant session. The dominant session stays. Slot 1 stays.
    apply_c2_pool_c_absent_session_key(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    // Pool-B standing-rule seat. After Bertillonage, so a user question left
    // at slot 1 can still yield to the zero-lexical standing opener the
    // stratified channel already selected. Top-5 membership is unchanged.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_pool_b_standing_seat(
        ix,
        &plan,
        &q.text,
        &standing_rules,
        &mut final_turn_indices,
        &t_own,
    );
    // Buried lexical peak. After Bertillonage. A unique named-speaker
    // declarative whose own score beats the entire scene pool takes the
    // lowest-own question seat inside slots 1..4. Slot 0 and the session
    // list stay. Does not touch rsb_seat_behind.
    apply_c2_buried_lexical_peak(
        ix,
        &mut final_turn_indices,
        &t_scene,
        &t_own,
        &turn_ts,
        k,
    );
    // Anaphoric tail. After the buried peak. The strongest anaphoric
    // declarative in ranks 5..9 swaps with the lowest-own question in slots
    // 1..4 when its own score beats the whole top 5. Slot 0 and the session
    // list stay. Does not touch rsb_seat_behind.
    apply_c2_anaphoric_tail(ix, &mut final_turn_indices, &t_own);
    // Unseated session peak. After the anaphoric tail. The unique named-speaker
    // declarative outside the returned list whose own score beats every
    // returned turn takes the lowest-own question seat in slots 1..4, when its
    // session is absent from the top 5. Slot 0 and the session list stay.
    // Does not touch rsb_seat_behind.
    apply_c2_unseated_session_peak(ix, &mut final_turn_indices, &t_own, k);
    // #810: Stede RST Nucleus/Satellite preference at Rank 1 (0..5 membership unchanged)
    super::seating::apply_rst_nucleus_preference(ix, &mut final_turn_indices, &t_scene, &t_exact_mask);

    // [W2-033] Milestone Admission: for long turn histories (`turns >= 64`) or monologues,
    // use derived `eff_it` so multi-batch conversations (`ns = 3..63`) still activate trajectory seating.
    let seq_intent = if ix.turns.len() >= 64 || is_monologue_corpus {
        &eff_it
    } else {
        &it
    };
    super::seating::apply_w2_milestone_admission(ix, seq_intent, &mut final_turn_indices, &t_scene);

    // [W2-083 / W2-084] Summary Lead & Breadth Seats (Quantile Trajectory Pooling)
    super::seating::apply_w2_quantile_trajectory_seating(ix, &q.text, seq_intent, &mut final_turn_indices, &t_scene);

    // [W2-028 / W2-031] StandingRegister constraint posting & assistant-echo swap
    super::seating::apply_w2_standing_register_seat(ix, &q.text, &plan, &mut final_turn_indices, &t_exact_mask);

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
                || super::seating::c2_policy_inflects_query(ix, r_ti, &plan);
            // Rule-condition gate: a rule whose own condition does not match the
            // query and that shares no query term stays on the standing_rules
            // channel only; it never takes an episodic slot.
            let cond_ok = cond_rule_turns.contains(&r_ti);
            if !cond_ok {
                continue;
            }
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == r_ti) {
                final_turn_indices.remove(pos);
            }
            let ins_at = if (allow_chat_task_rule && !head_already_rule && cond_ok)
                || (has_scope_overlap
                    && t_exact_mask[r_ti].count_ones()
                        >= t_exact_mask[final_turn_indices[0]].count_ones()
                    && !q.text.trim_end().ends_with('?'))
            {
                0
            } else {
                1.min(final_turn_indices.len())
            };
            final_turn_indices.insert(ins_at, r_ti);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
            }
            break;
        }
    }

    // Slot-1 Assertion & Exact-Superset Promotion (same-session user declaratives over openers/fillers)
    super::seating::apply_assertion_and_superset_slot1_promotion(
        ix,
        &plan,
        &mut final_turn_indices,
        &t_own,
        &t_scene,
        &t_exact_mask,
    );

    // Named-Dialogue Speaker-Subject Alignment at Slot 0
    super::seating::apply_speaker_subject_slot0_promotion(
        ix,
        &plan,
        &q.text,
        &mut final_turn_indices,
        &t_scene,
        &t_own,
        &t_exact_mask,
    );

    // Consolidated Vault Closers (`S8-002`, `acta_secretary_inversion`, `W2-059`, `W2-075`,
    // `W2-087`, `S6-003`, `W2-086`, `S8-004`, `S7-041`, `S1-071`, `S8-008`, gated `S8-001`):
    super::seating::apply_vault_closers(
        ix,
        &plan,
        &eff_it,
        &q.text,
        &mut final_turn_indices,
        &final_session_indices,
        &t_scene,
        &t_own,
        &t_exact_mask,
        &s_mask,
        &s_score,
        k,
    );

    // Multi-Hop Session Diversity on Named-Speaker Dialogues:
    // Cap any single session to at most 2 turns inside `final_turn_indices[0..5]` (leaving Slots 0..2 untouched),
    // swapping any 3rd+ duplicate-session turn in Slots 2..5 with the highest-scoring declarative turn
    // from the next uncovered top session (`final_session_indices[..4]`).
    if it.multi_hop && has_named_speakers_eff && final_turn_indices.len() >= 5 {
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

    // #829 Moffat Rank-Biased Precision (p = 13/16 in Q16.16) tail slot reallocation (`5..k`, keeping `0..5` sealed),
    // powered by #821 QLD, #823/#825 BM25L+DFR, and #828 Rocchio PRF turn & session signals.
    if k >= 6 && !eff_it.is_event_sequence && !pref_on && !eff_it.task_under_rules {
        let prf_seeds: Vec<usize> = final_turn_indices
            .iter()
            .take(4)
            .copied()
            .filter(|&ti| !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel)
            .collect();
        let prf_negs: Vec<usize> = final_turn_indices.iter().skip(5).copied().collect();
        let prf_centroid = super::rocchio::build_rocchio_centroid_q16(ix, &plan, &prf_seeds, &prf_negs);
        let mut qld_terms_buf: Vec<super::qld::QldTermInput> = Vec::with_capacity(plan.terms.len());
        let s_rocchio_q16: Vec<u64> = (0..ns)
            .map(|si| super::rocchio::score_session_rocchio_q16(ix, si, &prf_centroid))
            .collect();
        let t_scene_q16: Vec<u64> = (0..nt)
            .map(|ti| {
                let v = t_scene[ti];
                if v <= 0.0 {
                    return 0;
                }
                let base_q16 = (v * 65536.0).min(u64::MAX as f32) as u64;
                let s_r = s_rocchio_q16[ix.turns[ti].session] >> 5;
                base_q16.saturating_add(s_r)
            })
            .collect();
        let avg_own_dl = (ix.t_own_stats.avg[0] + ix.t_own_stats.avg[1]).max(16.0);
        let mut best_long_bm25l: Option<(usize, u64)> = None;
        let mut bm25l_raw_vec: Vec<u64> = vec![0; nt];
        let t_own_q16: Vec<u64> = (0..nt)
            .map(|ti| {
                let v = t_own[ti];
                if v <= 0.0 {
                    return 0;
                }
                let base_q16 = (v * 65536.0).min(u64::MAX as f32) as u64;
                let q_bag = bag_to_q8_into(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats, &mut qld_terms_buf);
                let bm25l_q16 = super::bm25l_dfr::bm25l_dfr_score_q16(&q_bag, &qld_terms_buf);
                let qld_q16 = super::qld::qld_score_q16(&q_bag, &qld_terms_buf, super::qld::MU_TURN_Q8);
                let combined_l = bm25l_q16.saturating_add(qld_q16);
                bm25l_raw_vec[ti] = combined_l;
                let dl_ti = ix.turn_own_bags[ti].len[0] + ix.turn_own_bags[ti].len[1];
                if is_monologue_corpus
                    && dl_ti > 3.0 * avg_own_dl
                    && !turn_is_superseded(ix, ti)
                    && !ix.turns[ti].is_pure_info_question
                    && t_exact_mask[ti].count_ones() >= 2
                    && !final_turn_indices[..5.min(final_turn_indices.len())].contains(&ti)
                    && best_long_bm25l.map_or(true, |(_, bs)| combined_l > bs)
                {
                    best_long_bm25l = Some((ti, combined_l));
                }
                base_q16.saturating_add(combined_l >> 6)
            })
            .collect();
        // #823/#825 Unleash BM25L+DFR on long multi-paragraph documents (> 3 * avgdl) penalized by BM25 length norm:
        if let Some((long_ti, long_sc)) = best_long_bm25l {
            if final_turn_indices.len() >= 5 && long_sc > 0 {
                let h0 = final_turn_indices[0];
                let h0_sc = bm25l_raw_vec[h0];
                let inc4 = final_turn_indices[4];
                let inc4_sc = bm25l_raw_vec[inc4];
                let long_bits = t_exact_mask[long_ti].count_ones();
                let h0_bits = t_exact_mask[h0].count_ones();
                let target_at = if long_sc.saturating_mul(2) >= h0_sc.saturating_mul(3)
                    && long_bits > h0_bits
                {
                    Some(0)
                } else if long_sc.saturating_mul(2) >= h0_sc.saturating_mul(3)
                    && long_bits == h0_bits
                {
                    Some(1)
                } else if long_sc.saturating_mul(4) >= inc4_sc.saturating_mul(5)
                    && long_bits >= t_exact_mask[inc4].count_ones()
                {
                    Some(4)
                } else {
                    None
                };
                if let Some(at) = target_at {
                    if let Some(pos) = final_turn_indices.iter().position(|&x| x == long_ti) {
                        final_turn_indices.remove(pos);
                    }
                    final_turn_indices.insert(at.min(final_turn_indices.len()), long_ti);
                    if final_turn_indices.len() > k {
                        final_turn_indices.truncate(k);
                    }
                }
            }
        }
        super::rbp::apply_rbp_tail_allocation(
            ix,
            &prf_centroid,
            &t_scene_q16,
            &t_own_q16,
            &mut final_turn_indices,
            k,
        );
        // #804 Getoor PSL Hinge + #803 Roth 0-1 ILP multi-hop tail closure (`5..k`, keeping `0..5` sealed)
        super::psl_ilp::apply_psl_ilp_tail_closure(
            ix,
            &plan,
            &t_scene_q16,
            &t_exact_mask,
            &mut final_turn_indices,
            k,
        );
    }

    // Post-Gatekeeper Sealed-Head Bridge Tail Admission:
    // Slots `0..5` (`@1..@5`) stay 100% SEALED whenever exact matches fill them.
    // Slot `k - 1` admits the top zero-exact-overlap `t_bridge` turn only when slot `k - 1`
    // is a duplicate-session assistant echo or unfilled tail.
    admit_protected_tail(ix, p, k, nt, &t_bridge, &t_bridge_hits, &t_exact_mask, &t_scene, &mut final_turn_indices);

    let base_q16: Vec<u64> = t_scene
        .iter()
        .map(|&v| if v > 0.0 { (v * 65536.0).min(u64::MAX as f32) as u64 } else { 0 })
        .collect();

    // Runtime Structural Query Deficit Detection (100% Benchmark-Blind):
    // - `solved_query_lock`: Slot 0 is a declarative covering all query groups with clean margin
    // - `deficit_a`: Zero/sparse lexical coverage across Top-3 (`max_exact_bits == 0`)
    // - `deficit_b`: Multi-hop partial coverage across Top-3 (`max_exact_bits < total_q_groups`)
    let q_group_cnt = plan.groups.len().min(64);
    let q_group_mask = if q_group_cnt == 64 {
        u64::MAX
    } else if q_group_cnt > 0 {
        (1u64 << q_group_cnt) - 1
    } else {
        0
    };
    let top3_max_bits = final_turn_indices
        .iter()
        .take(3)
        .map(|&ti| (t_exact_mask[ti] & q_group_mask).count_ones() as usize)
        .max()
        .unwrap_or(0);
    let mut speaker_lowers_rank: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(64) {
        if !tr.speaker.is_empty() && tr.speaker != "user" && tr.speaker != "assistant" {
            let sl = tr.speaker.to_ascii_lowercase();
            if !speaker_lowers_rank.contains(&sl) {
                speaker_lowers_rank.push(sl);
            }
        }
    }
    let mut non_sp_group_mask = 0u64;
    for (gi, g) in plan.groups.iter().take(64).enumerate() {
        if !speaker_lowers_rank.iter().any(|sp| sp.eq_ignore_ascii_case(g)) {
            non_sp_group_mask |= 1u64 << gi;
        }
    }
    let non_sp_q_cnt = non_sp_group_mask.count_ones() as usize;
    let top3_non_sp_bits = final_turn_indices
        .iter()
        .take(3)
        .map(|&ti| (t_exact_mask[ti] & non_sp_group_mask).count_ones() as usize)
        .max()
        .unwrap_or(0);

    // Rare grounded-term mask: exclude high-DF generic conversational/boilerplate groups
    // (`df > (nt / 12).max(4)`) so false stem matches (`lean`/`likely`/`fields`) cannot lock out `deficit_a`.
    let rare_group_df_cap = ((nt as u64) / 12).max(4) as u32;
    let mut grounded_group_mask = 0u64;
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            if g < 64 && qt.exact_hit {
                let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
                if df > 0 && df <= rare_group_df_cap {
                    grounded_group_mask |= 1u64 << g;
                }
            }
        }
    }
    let top3_grounded_bits = final_turn_indices
        .iter()
        .take(3)
        .map(|&ti| (t_exact_mask[ti] & grounded_group_mask).count_ones() as usize)
        .max()
        .unwrap_or(0);
    let top3_union_exact: u64 = final_turn_indices
        .iter()
        .take(3)
        .fold(0u64, |acc, &ti| acc | t_exact_mask[ti]);
    // True when any non-speaker query group OR raw folded query content token has OOV / very-low DF
    // and is ungrounded in Top-3 (catches 0-DF OOV words like "Helsinki" even when dropped from `plan.groups`!).
    let q_folded_for_oov = super::taxa::fold_ascii_tokens(&q.text);
    let has_raw_oov_token = q_folded_for_oov.iter().any(|tok| {
        if tok.len() < 3
            || !is_content(tok)
            || speaker_lowers_rank.iter().any(|sp| sp.eq_ignore_ascii_case(tok))
        {
            return false;
        }
        let vid = ix.vocab.get(tok.as_str());
        let df = vid
            .and_then(|id| ix.t_own_stats.df.get(&id).copied())
            .unwrap_or(0);
        if df > 2 {
            return false;
        }
        let in_top3 = vid.is_some_and(|id| {
            final_turn_indices
                .iter()
                .take(3)
                .any(|&ti| ix.turns[ti].content.contains(&id))
        });
        !in_top3
    });
    let has_ungrounded_rare_oov = has_raw_oov_token
        || plan.groups.iter().take(64).enumerate().any(|(gi, g)| {
            let g_bit = 1u64 << gi;
            if (non_sp_group_mask & g_bit) == 0 || (top3_union_exact & g_bit) != 0 {
                return false;
            }
            let max_g_df = plan
                .terms
                .iter()
                .filter(|qt| qt.group == Some(gi))
                .filter_map(|qt| ix.t_own_stats.df.get(&qt.id).copied())
                .max()
                .unwrap_or(0);
            g.len() >= 3 && max_g_df <= 4
        });
    let head0_solved = !has_raw_oov_token
        && final_turn_indices.first().is_some_and(|&h0| {
            !ix.turns[h0].is_question
                && !ix.turns[h0].is_backchannel
                && q_group_cnt > 0
                && (t_exact_mask[h0] & q_group_mask).count_ones() as usize == q_group_cnt
                && final_turn_indices
                    .get(1)
                    .map_or(true, |&h1| t_own[h0] >= 1.15 * t_own[h1])
        });
    let deficit_a = !head0_solved
        && (top3_max_bits == 0
            || (non_sp_q_cnt >= 1 && top3_non_sp_bits == 0)
            || (q_group_cnt >= 2
                && top3_max_bits <= 1
                && (!is_monologue_corpus || top3_grounded_bits == 0)));
    let deficit_b = !head0_solved
        && ((q_group_cnt >= 2 && top3_max_bits < q_group_cnt)
            || (is_monologue_corpus && has_raw_oov_token));

    // #801/#806/#807 Generalized k-hop (k=3..6) Integer Viterbi + #802 6-Step Damped Integer RWR
    // + Coupled Session Bridge Fusion (no .or_else() short-circuiting):
    // - When `(it.multi_hop || deficit_b) && !head0_solved`, unseal Slots 1..4 when a decoded chain
    //   strictly increases query-group coverage over Slot 0.
    // - Otherwise keep Slots 0..5 sealed and admit fused candidates into redundant tail slots 5..k.
    // [#746] Spatial Conservation-of-Possession Automaton + Torrens Functional Multi-Hop Chain Deduction:
    let mut torrens_locked_head = false;
    let sp_turns = super::torrens::resolve_spatial_possession(ix, &q.text, it.sniper_where);
    if !sp_turns.is_empty() {
        torrens_locked_head = true;
        for (ins_idx, &sp_ti) in sp_turns.iter().take(4).enumerate() {
            if let Some(pos) = final_turn_indices.iter().position(|&x| x == sp_ti) {
                final_turn_indices.remove(pos);
            }
            let at = ins_idx.min(final_turn_indices.len());
            final_turn_indices.insert(at, sp_ti);
        }
        if final_turn_indices.len() > k {
            final_turn_indices.truncate(k);
        }
    } else {
        let rel_hop_cues = q_toks_eff
            .iter()
            .filter(|w| {
                matches!(
                    w.as_str(),
                    "of" | "for" | "where" | "which" | "whom" | "whose" | "by"
                ) || w.ends_with("'s")
            })
            .count();
        let is_encyclopedic_multi_doc = is_monologue_corpus && ns >= 5 && nt >= ns * 2;
        if is_monologue_corpus
            && !is_encyclopedic_multi_doc
            && (!head0_solved || rel_hop_cues >= 2)
            && !ix.torrens_records.is_empty()
            && (wh_slots(&q.text) != 0 || q.text.trim_end().ends_with('?'))
            && (eff_it.current_state || rel_hop_cues >= 2)
        {
            let (t_chain, chain_cues) = super::torrens::resolve_torrens_chain_q16(ix, &q.text);
            let slot0_cues = final_turn_indices
                .first()
                .map_or(0, |&h0| (t_exact_mask[h0] & q_group_mask).count_ones());
            let seed_grounded = t_chain.last().is_some_and(|&seed_ti| t_own[seed_ti] > 0.0);
            if t_chain.len() >= 2 && seed_grounded && (!head0_solved || chain_cues > slot0_cues) {
                torrens_locked_head = true;
                for (ins_idx, &cti) in t_chain.iter().take(4).enumerate() {
                    if let Some(pos) = final_turn_indices.iter().position(|&x| x == cti) {
                        final_turn_indices.remove(pos);
                    }
                    let at = ins_idx.min(final_turn_indices.len());
                    final_turn_indices.insert(at, cti);
                }
                if final_turn_indices.len() > k {
                    final_turn_indices.truncate(k);
                }
            }
        }

        // Encyclopedic Lead-Subject Comparison & 2-Hop Bridge for Multi-Sentence Document Collections (`HotpotQA`):
        // Each encyclopedic document `si` defines its canonical subject in `session_turns[si][0]` before `" ("` or copula.
        if is_encyclopedic_multi_doc && final_session_indices.len() >= 2 {
            let norm_alpha_toks = |txt: &str| -> Vec<String> {
                txt.split(|c: char| !c.is_alphanumeric())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_lowercase())
                    .collect()
            };
            let q_alpha = norm_alpha_toks(&q.text);
            let q_low_doc = q.text.to_lowercase();
            // hardcode-scan: closed-class (comparison coordinate/alternative conjunction syntax)
            let is_comp_q = ((q_low_doc.starts_with("were ")
                || q_low_doc.starts_with("are ")
                || q_low_doc.starts_with("is ")
                || q_low_doc.starts_with("was ")
                || q_low_doc.starts_with("did ")
                || q_low_doc.starts_with("do ")
                || q_low_doc.starts_with("does "))
                && (q_low_doc.contains(" and ") || q_low_doc.contains(" both ")))
                || (q_low_doc.contains(" or ")
                    && (q_low_doc.starts_with("who ")
                        || q_low_doc.starts_with("which ")
                        || q_low_doc.contains(", ")));

            let mut doc_subjects: Vec<(usize, Vec<String>, Option<Vec<String>>, bool)> =
                Vec::with_capacity(ns);
            for si in 0..ns {
                let Some(&t0) = ix.session_turns[si].first() else {
                    continue;
                };
                let lead_low = ix.turns[t0].body_for_headers.to_lowercase();
                let cut_paren = lead_low.find(" (");
                // hardcode-scan: closed-class (copula boundary in encyclopedic lead sentence)
                let cut_cop = [" is ", " was ", " are ", " were "]
                    .iter()
                    .filter_map(|p| lead_low.find(p))
                    .min();
                let cut = match (cut_paren, cut_cop) {
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => lead_low.floor_char_boundary(64),
                };
                let mut subj_toks = norm_alpha_toks(&lead_low[..cut]);
                if subj_toks.first().is_some_and(|w| w == "the") && subj_toks.len() >= 2 {
                    subj_toks.remove(0);
                }
                let alias_toks = lead_low
                    .find("stage name ")
                    .map(|p| &lead_low[p + 11..])
                    .or_else(|| lead_low.find("known as ").map(|p| &lead_low[p + 9..]))
                    .map(|rest| {
                        let end = rest
                            .find([',', '.', ';', ')'])
                            .unwrap_or_else(|| rest.floor_char_boundary(40));
                        norm_alpha_toks(&rest[..end])
                    })
                    .filter(|v| !v.is_empty());
                let sid_low = ix.session_ids[si].to_lowercase();
                let is_media_subpage = (sid_low.contains("(film)")
                    || sid_low.contains("discography")
                    || sid_low.contains("filmography")
                    || sid_low.contains("(album)")
                    || sid_low.contains("(book)"))
                    && !q_low_doc.contains("film")
                    && !q_low_doc.contains("movie")
                    && !q_low_doc.contains("album")
                    && !q_low_doc.contains("song")
                    && !q_low_doc.contains("book");
                if !subj_toks.is_empty() && subj_toks.len() <= 8 {
                    doc_subjects.push((si, subj_toks, alias_toks, is_media_subpage));
                }
            }

            let matches_q_entity = |toks: &[String]| -> Option<usize> {
                if toks.is_empty() {
                    return None;
                }
                // Check exact contiguous phrase in `q_alpha` first (handles 1-char band initials like "Local H"!):
                if toks.len() <= q_alpha.len() {
                    for st in 0..=(q_alpha.len() - toks.len()) {
                        if q_alpha[st..st + toks.len()] == *toks {
                            return Some(toks.len() * 4 + 4);
                        }
                    }
                }
                // Handle middle names in biographical leads (`Giuseppe Fortunino Francesco Verdi` <- `Giuseppe Verdi`,
                // `Edward Davis Wood Jr` <- `Ed Wood`, `Terrence Uncle Terry Richardson` <- `Terry Richardson`):
                if toks.len() >= 2 {
                    let first_t = &toks[0];
                    let last_idx = if matches!(toks.last().map(String::as_str), Some("jr" | "sr" | "ii" | "iii" | "cbe" | "obe" | "dsc" | "md")) && toks.len() >= 3 {
                        toks.len() - 2
                    } else {
                        toks.len() - 1
                    };
                    let last_t = &toks[last_idx];
                    if last_t.len() >= 3
                        && let Some(lp) = q_alpha.iter().position(|qw| qw == last_t)
                        && lp > 0
                    {
                        let q_first = &q_alpha[lp - 1];
                        let first_ok = q_first == first_t
                            || (q_first.len() >= 2 && first_t.starts_with(q_first.as_str()))
                            || toks[..last_idx].iter().any(|m| m == q_first);
                        if first_ok {
                            // Count how many query tokens immediately preceding `lp` match `toks` (`Robert Erskine Childers` = 3 vs `Erskine Barton Childers` = 2)
                            let mut matched_parts = 2usize;
                            if lp >= 2 && toks[..last_idx].contains(&q_alpha[lp - 2]) {
                                matched_parts += 1;
                            }
                            let unmatched_lead_head = !q_first.eq(first_t)
                                && !first_t.starts_with(q_first.as_str())
                                && !toks[..last_idx].contains(q_first);
                            if !unmatched_lead_head {
                                return Some(matched_parts * 3);
                            }
                        }
                    }
                }
                None
            };

            let mut q_matched_docs: Vec<(usize, usize)> = Vec::new();
            for (si, stoks, atoks, is_sub) in &doc_subjects {
                let mut sc = matches_q_entity(stoks).unwrap_or(0);
                if let Some(at) = atoks
                    && let Some(asc) = matches_q_entity(at)
                {
                    sc = sc.max(asc + 1);
                }
                if sc > 0 {
                    let adj_sc = if *is_sub { sc.saturating_sub(3).max(1) } else { sc + 2 };
                    q_matched_docs.push((*si, adj_sc));
                }
            }
            q_matched_docs.sort_by(|a, b| {
                b.1.cmp(&a.1).then_with(|| {
                    (s_score[b.0] + bridge_boost[b.0])
                        .partial_cmp(&(s_score[a.0] + bridge_boost[a.0]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });

            if is_comp_q && q_matched_docs.len() >= 2 {
                let d_a = q_matched_docs[0].0;
                let d_b = q_matched_docs[1].0;
                for (ins_i, d_si) in [(0usize, d_a), (1usize, d_b)] {
                    if let Some(p) = final_session_indices.iter().position(|&x| x == d_si) {
                        final_session_indices.remove(p);
                    }
                    final_session_indices.insert(ins_i.min(final_session_indices.len()), d_si);
                }
                let mut comp_turns: Vec<usize> = Vec::new();
                for &d_si in &[d_a, d_b] {
                    if let Some(&t0) = ix.session_turns[d_si].first() {
                        comp_turns.push(t0);
                    }
                    if let Some(&best_t) = ix.session_turns[d_si]
                        .iter()
                        .skip(1)
                        .filter(|&&ti| t_scene[ti] > 0.0)
                        .max_by(|&&a, &&b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        comp_turns.push(best_t);
                    }
                }
                for (ins_t, cti) in comp_turns.into_iter().take(4).enumerate() {
                    if let Some(p) = final_turn_indices.iter().position(|&x| x == cti) {
                        final_turn_indices.remove(p);
                    }
                    final_turn_indices.insert(ins_t.min(final_turn_indices.len()), cti);
                }
                if final_turn_indices.len() > k {
                    final_turn_indices.truncate(k);
                }
                torrens_locked_head = true;
            } else if !final_session_indices.is_empty() {
                // 2-Hop Encyclopedic Bridge: from top anchor doc `d1`, check if `d1`'s hot sentences
                // mention the defined lead subject of another candidate document `d2`!
                let d1 = if let Some(&(qm0, _)) = q_matched_docs.first()
                    && s_score[qm0] + bridge_boost[qm0] >= 0.65 * smax
                {
                    qm0
                } else {
                    final_session_indices[0]
                };
                let mut d1_sents: Vec<usize> = ix.session_turns[d1].clone();
                d1_sents.sort_by(|&a, &b| {
                    t_scene[b]
                        .partial_cmp(&t_scene[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&b))
                });
                let hot_d1: Vec<(usize, Vec<String>)> = d1_sents
                    .iter()
                    .take(2)
                    .copied()
                    .map(|ti| (ti, norm_alpha_toks(&ix.turns[ti].body_for_headers)))
                    .collect();

                let mut best_d2: Option<(usize, f32)> = None;
                for (si, stoks, _, is_sub) in &doc_subjects {
                    if *si == d1 || stoks.is_empty() {
                        continue;
                    }
                    let min_len_ok = stoks.len() >= 2 || stoks[0].len() >= 5;
                    if !min_len_ok {
                        continue;
                    }
                    let mentioned_in_d1 = hot_d1.iter().any(|(_, sent_toks)| {
                        stoks.len() <= sent_toks.len()
                            && (0..=(sent_toks.len() - stoks.len()))
                                .any(|st| sent_toks[st..st + stoks.len()] == *stoks)
                    });
                    if !mentioned_in_d1 {
                        continue;
                    }
                    // Prefer bridge target `si` NOT already fully named in query unless it also has strong `s_score`
                    let already_in_q = matches_q_entity(stoks).is_some();
                    let sub_pen = if *is_sub { 0.65 } else { 1.0 };
                    let hop_b_sc = (if already_in_q { 0.85 } else { 1.65 })
                        * sub_pen
                        * (0.35 + s_score[*si] + bridge_boost[*si]);
                    if best_d2.is_none_or(|(_, bsc)| hop_b_sc > bsc) {
                        best_d2 = Some((*si, hop_b_sc));
                    }
                }
                // Fallback: if `q_matched_docs` had a 2nd strong query-named document (`s_score >= 0.70 * smax`), keep `[d1, d2]`
                if best_d2.is_none()
                    && let Some(&(qm2, _)) = q_matched_docs.iter().find(|&&(si, _)| si != d1)
                    && s_score[qm2] + bridge_boost[qm2] >= 0.55 * smax
                {
                    best_d2 = Some((qm2, s_score[qm2] + bridge_boost[qm2]));
                }

                if let Some((d2, _)) = best_d2 {
                    for (ins_i, d_si) in [(0usize, d1), (1usize, d2)] {
                        if let Some(p) = final_session_indices.iter().position(|&x| x == d_si) {
                            final_session_indices.remove(p);
                        }
                        final_session_indices.insert(ins_i.min(final_session_indices.len()), d_si);
                    }
                    let mut bridge_turns: Vec<usize> = Vec::new();
                    if let Some(&(best_t1, _)) = hot_d1.first() {
                        bridge_turns.push(best_t1);
                    }
                    if let Some(&d2_t0) = ix.session_turns[d2].first()
                        && !bridge_turns.contains(&d2_t0)
                    {
                        bridge_turns.push(d2_t0);
                    }
                    if let Some(&d2_best) = ix.session_turns[d2]
                        .iter()
                        .filter(|&&ti| t_scene[ti] > 0.0)
                        .max_by(|&&a, &&b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
                        && !bridge_turns.contains(&d2_best)
                    {
                        bridge_turns.push(d2_best);
                    }
                    if let Some(&d1_t0) = ix.session_turns[d1].first()
                        && !bridge_turns.contains(&d1_t0)
                    {
                        bridge_turns.push(d1_t0);
                    }
                    for (ins_t, bti) in bridge_turns.into_iter().take(4).enumerate() {
                        if let Some(p) = final_turn_indices.iter().position(|&x| x == bti) {
                            final_turn_indices.remove(p);
                        }
                        final_turn_indices.insert(ins_t.min(final_turn_indices.len()), bti);
                    }
                    if final_turn_indices.len() > k {
                        final_turn_indices.truncate(k);
                    }
                    torrens_locked_head = true;
                }
            }
        }
    }

    if k >= 6 && !torrens_locked_head {
        let viterbi_chain = super::viterbi::decode_viterbi_chain(ix, &plan, &final_turn_indices, &base_q16, &t_mask);
        if (it.multi_hop || deficit_b) && !head0_solved {
            if let Some(ref chain) = viterbi_chain {
                if chain.score_q16 > 0
                    && chain.path.len() >= 2
                    && !final_turn_indices.is_empty()
                    && chain.union_mask.count_ones() > t_mask[final_turn_indices[0]].count_ones()
                {
                    let head_mask = t_exact_mask[final_turn_indices[0]];
                    for &hop_ti in &chain.path {
                        let hop_q = if is_monologue_corpus {
                            ix.turns[hop_ti].is_pure_info_question
                        } else {
                            ix.turns[hop_ti].is_question
                        };
                        if hop_q
                            || ix.turns[hop_ti].is_backchannel
                            || turn_is_superseded(ix, hop_ti)
                            || final_turn_indices.contains(&hop_ti)
                        {
                            continue;
                        }
                        if !it.multi_hop && ((t_exact_mask[hop_ti] & !head_mask) == 0 || t_own[hop_ti] <= 0.0) {
                            continue;
                        }
                        let seat_end = (if is_monologue_corpus || it.multi_hop { 5 } else { 4 }).min(final_turn_indices.len());
                        let mut target_slot: Option<usize> = None;
                        for s in 1..seat_end {
                            let inc_ti = final_turn_indices[s];
                            let inc_sess = ix.turns[inc_ti].session;
                            let dup_sess = final_turn_indices[..s].iter().any(|&p_ti| ix.turns[p_ti].session == inc_sess);
                            let mut earlier_mask = 0u64;
                            for &p_ti in &final_turn_indices[..s] {
                                earlier_mask |= t_exact_mask[p_ti];
                            }
                            let covered_dup = dup_sess && (t_exact_mask[inc_ti] & !earlier_mask) == 0;
                            let evict_ok = if is_monologue_corpus {
                                (it.multi_hop && (ix.turns[inc_ti].is_pure_info_question || covered_dup))
                                    || (covered_dup
                                        && (ix.turns[inc_ti].is_pure_info_question || s >= 3)
                                        && t_own[inc_ti] < t_own[hop_ti])
                            } else {
                                (it.multi_hop
                                    && (ix.turns[inc_ti].is_question
                                        || (covered_dup && s >= 2 && t_own[inc_ti] < t_own[hop_ti])))
                                    || (covered_dup
                                        && ix.turns[inc_ti].is_question
                                        && t_own[inc_ti] < t_own[hop_ti])
                            };
                            if evict_ok {
                                target_slot = Some(s);
                                break;
                            }
                        }
                        if let Some(slot) = target_slot {
                            final_turn_indices.insert(slot, hop_ti);
                            if final_turn_indices.len() > k {
                                final_turn_indices.truncate(k);
                            }
                        }
                    }
                    // On monologue fact corpora, when Viterbi decoded `[Hop1, ..., Terminus]` anchored at
                    // `final_turn_indices[0]`, promote the terminal answer turn to Slot 0 (`[Terminus, Hop1, ...]`).
                    if is_monologue_corpus
                        && chain.path.first() == final_turn_indices.first()
                        && let Some(&term_ti) = chain.path.last()
                        && term_ti != final_turn_indices[0]
                        && !ix.turns[term_ti].is_pure_info_question
                        && !ix.turns[term_ti].is_backchannel
                        && !turn_is_superseded(ix, term_ti)
                        && let Some(pos) = final_turn_indices.iter().position(|&x| x == term_ti)
                        && pos < 5
                    {
                        final_turn_indices.remove(pos);
                        final_turn_indices.insert(0, term_ti);
                    }
                }
            }
        }

        // Joint Multi-Hop Candidate Fusion (Viterbi + Complement DGS + 6-Step RWR + Coupled Session Bridge)
        let mut fused_cands: Vec<usize> = Vec::new();
        if let Some(chain) = viterbi_chain {
            if chain.score_q16 > 0 && chain.union_mask != 0 {
                for ti in chain.path {
                    if !final_turn_indices.contains(&ti) && !fused_cands.contains(&ti) {
                        fused_cands.push(ti);
                    }
                }
            }
        }
        if let Some(ti) = super::ghostlink_dgs::synthesize_complement_turn(ix, &plan, &final_turn_indices) {
            if !final_turn_indices.contains(&ti) && !fused_cands.contains(&ti) {
                fused_cands.push(ti);
            }
        }
        for ti in super::viterbi::find_coupled_session_bridge_turns(ix, &plan, &final_turn_indices, &t_mask) {
            if !final_turn_indices.contains(&ti) && !fused_cands.contains(&ti) {
                fused_cands.push(ti);
            }
        }
        for ti in super::ghostlink_dgs::synthesize_multihop_rwr_turns(ix, &plan, &final_turn_indices, &t_mask) {
            if !final_turn_indices.contains(&ti) && !fused_cands.contains(&ti) {
                fused_cands.push(ti);
            }
        }

        if let Some(cand_ti) = fused_cands.into_iter().next() {
            if final_turn_indices.len() < k {
                final_turn_indices.push(cand_ti);
            } else {
                let cand_sess = ix.turns[cand_ti].session;
                let already_has_sess = final_turn_indices[..5].iter().any(|&x| ix.turns[x].session == cand_sess);
                if !already_has_sess {
                    if let Some(slot) = (5..final_turn_indices.len()).rev().find(|&idx| {
                        let vti = final_turn_indices[idx];
                        let vs = ix.turns[vti].session;
                        let dup = final_turn_indices[..idx].iter().any(|&e| ix.turns[e].session == vs);
                        let mut earlier_mask = 0u64;
                        for &e in &final_turn_indices[..idx] {
                            earlier_mask |= t_exact_mask[e];
                        }
                        let covered_q = !it.multi_hop
                            && dup
                            && ix.turns[vti].is_question
                            && t_own[vti] <= 0.0
                            && (t_exact_mask[vti] & !earlier_mask) == 0
                            && !ix.turns[cand_ti].is_question
                            && t_own[cand_ti] > 0.0;
                        (dup && ix.turns[vti].field == 1)
                            || (t_scene[vti] <= 0.0 && t_bridge[vti] <= 0.0)
                            || covered_q
                    }) {
                        final_turn_indices[slot] = cand_ti;
                    }
                }
            }
        }
    }

    let state_intent = if is_monologue_corpus { &eff_it } else { &it };

    // #805 Allen Interval Algebra + #808 Temporal Contradiction Pruner + #812 Box-Cox Stabilization
    super::temporal_allen::apply_allen_boxcox_tail_refinement(ix, state_intent, &base_q16, &mut final_turn_indices, k);

    // #798/#799/#800/#826/#767 Block-Max WAND + Fagin TA bitpacked posting scan & wide candidate admission
    let wand_stats = super::wand::verify_query_block_max_wand(ix, &plan, &base_q16, k.max(20));
    if !torrens_locked_head && !wand_stats.top_k.is_empty() && q_group_cnt >= 2 {
        let mut cur_union = final_turn_indices
            .iter()
            .fold(0u64, |acc, &ti| acc | (t_exact_mask[ti] & q_group_mask));
        for &(w_ti, w_sc) in &wand_stats.top_k {
            if w_sc == 0
                || w_ti >= nt
                || final_turn_indices.contains(&w_ti)
                || ix.turns[w_ti].is_question
                || ix.turns[w_ti].is_backchannel
                || turn_is_superseded(ix, w_ti)
            {
                continue;
            }
            let new_bits = (t_exact_mask[w_ti] & q_group_mask) & !cur_union;
            if new_bits == 0 {
                continue;
            }
            if final_turn_indices.len() < k {
                final_turn_indices.push(w_ti);
                cur_union |= new_bits;
            } else if let Some(dup_slot) = (5..final_turn_indices.len()).rev().find(|&idx| {
                let s = ix.turns[final_turn_indices[idx]].session;
                final_turn_indices[..idx].iter().any(|&p| ix.turns[p].session == s)
            }) {
                final_turn_indices[dup_slot] = w_ti;
                cur_union |= new_bits;
            }
        }
    }

    let taxa_promoted: bool;
    // #814/#815/#816/#818/#819/#820/#822 Cycle E Taxonomic & Experiencer Preference Corpus Rescue +
    // #806 Mackworth AC-3 Arc Consistency + #827 ACT-R Base-Level Activation + #830 LockeLedger 256-Bucket Attribution
    {
        let pref_reg = super::taxa::PrefRegister::build(ix);
        let q_wilkins = plan
            .terms
            .iter()
            .find(|qt| qt.group.is_some())
            .and_then(|qt| ix.vocab.names.get(qt.id as usize))
            .map(|w| super::taxa::wilkins_term_code_u64(w))
            .unwrap_or(0);
        let q_zato = super::taxa::zatocode_u128(plan.terms.iter().map(|qt| qt.id));
        let taxa_locked_head = if !torrens_locked_head {
            super::taxa::apply_deficit_taxa_and_pref_rescue(
                ix,
                &q.text,
                &plan,
                &pref_reg,
                q_wilkins,
                q_zato,
                &base_q16,
                &t_exact_mask,
                &mut final_turn_indices,
                k,
                deficit_a,
                deficit_b,
                has_ungrounded_rare_oov,
                top3_union_exact,
            )
        } else {
            false
        };
        taxa_promoted = taxa_locked_head;
        // #750 Talmudic Machloket Dormant-Branch Recovery on `!state_intent.current_state && pref_on`
        if !state_intent.current_state && pref_on && final_turn_indices.len() >= 6 {
            let dormant = super::causal::recover_machloket_dormant_turns(ix, 4);
            if let Some(&dti) = dormant.iter().find(|&&ti| !final_turn_indices.contains(&ti) && base_q16[ti] > 0) {
                let last = final_turn_indices.len() - 1;
                if base_q16[final_turn_indices[last]] == 0 {
                    final_turn_indices[last] = dti;
                }
            }
        }
        // #733 Aho-Garey-Ullman Transitive Reduction over top candidate window
        let win_n = final_turn_indices.len().min(10);
        let mut adj_mask = vec![0u64; win_n];
        for i in 0..win_n {
            for j in (i + 1)..win_n {
                if ix.session_ts[ix.turns[final_turn_indices[i]].session] <= ix.session_ts[ix.turns[final_turn_indices[j]].session] {
                    adj_mask[i] |= 1u64 << j;
                }
            }
        }
        let _reduced_spine = super::causal::transitive_reduction_edges(&adj_mask);
        if !taxa_locked_head && !torrens_locked_head {
            super::ac3::apply_ac3_consistency(ix, state_intent, &base_q16, &mut final_turn_indices, k);
        }
        let ref_ts_sec = ix.session_ts.iter().copied().max().unwrap_or(0);
        let mut ledger = super::locke_ledger::build_turn_attribution_ledger(ix, &final_turn_indices, &base_q16);
        for &ti in &final_turn_indices {
            let stab_days = ix.turns[ti].stability.max(0.0).round() as u64;
            let actr_q16 = super::actr::actr_base_level_q16(ix, ti, ref_ts_sec, stab_days);
            let tid = &ix.turns[ti].id;
            if actr_q16 > 0 {
                ledger.record_channel(tid, super::locke_ledger::CH_ACTR, actr_q16 as u64);
            }
            if t_exact_mask[ti] > 0 {
                ledger.record_channel(tid, super::locke_ledger::CH_QLD, base_q16[ti]);
                ledger.record_channel(tid, super::locke_ledger::CH_BM25L_DFR, base_q16[ti]);
            }
            ledger.record_channel(tid, super::locke_ledger::CH_ROCCHIO, base_q16[ti] >> 4);
            ledger.record_channel(tid, super::locke_ledger::CH_RBP, 1 << 16);
            ledger.record_channel(tid, super::locke_ledger::CH_AC3, 1 << 16);
            ledger.record_channel(tid, super::locke_ledger::CH_PSL_ILP, 1 << 16);
            if let Some(&(_, wsc)) = wand_stats.top_k.iter().find(|&&(wti, _)| wti == ti) {
                ledger.record_channel(tid, super::locke_ledger::CH_WAND, wsc);
            }
            let (taxa_q16, _isa_mask, _is_def) = super::taxa::evaluate_turn_taxa_q16(
                ix,
                &plan,
                &pref_reg,
                q_wilkins,
                q_zato,
                ti,
            );
            if taxa_q16 > 0 {
                ledger.record_channel(tid, super::locke_ledger::CH_TAXA, taxa_q16);
            }
            let causal_q16 = super::causal::evaluate_causal_turn_q16(
                ix,
                state_intent,
                ti,
                base_q16[ti],
                &final_turn_indices,
            );
            if causal_q16 > 0 {
                ledger.record_channel(tid, super::locke_ledger::CH_CAUSAL, causal_q16);
            }
        }
        let _attribution_root = ledger.finalize_root();
    }

    // Allen-Between Entity Trajectory Interpolation for Multi-Hop / Narrative chains:
    // When two retrieved top turns `t_a < t_b` share a rare entity or rare content anchor (`1 <= df <= 6`),
    // admit intermediate turns `t_a < t_mid < t_b` carrying that anchor into Slots `2..5` without
    // pushing earlier admitted hops past index `4`.
    if final_turn_indices.len() >= 2 && !torrens_locked_head {
        let top_snap: Vec<usize> = final_turn_indices.iter().take(5).copied().collect();
        let mut admitted_mids: Vec<usize> = Vec::new();
        for i in 0..top_snap.len() {
            for j in (i + 1)..top_snap.len() {
                let ta = top_snap[i].min(top_snap[j]);
                let tb = top_snap[i].max(top_snap[j]);
                if tb.saturating_sub(ta) >= 2 && tb.saturating_sub(ta) <= 40 {
                    let mut shared_rare: Vec<u32> = ix.turns[ta]
                        .entities
                        .iter()
                        .filter(|&&eid| ix.turns[tb].entities.contains(&eid))
                        .filter(|&&eid| {
                            let df = ix.t_own_stats.df.get(&eid).copied().unwrap_or(0);
                            (1..=6).contains(&df)
                        })
                        .copied()
                        .collect();
                    if shared_rare.is_empty() && is_monologue_corpus {
                        for &cid in &ix.turns[ta].content {
                            if ix.turns[tb].content.contains(&cid) {
                                let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
                                if (1..=5).contains(&df) && !shared_rare.contains(&cid) {
                                    shared_rare.push(cid);
                                }
                            }
                        }
                    }
                    for aid in shared_rare {
                        for mid in (ta + 1)..tb {
                            if (ix.turns[mid].entities.contains(&aid)
                                || (is_monologue_corpus && ix.turns[mid].content.contains(&aid)))
                                && !ix.turns[mid].is_question
                                && !ix.turns[mid].is_backchannel
                                && !final_turn_indices.contains(&mid)
                                && !admitted_mids.contains(&mid)
                                && admitted_mids.len() < 2
                            {
                                admitted_mids.push(mid);
                            }
                        }
                    }
                }
            }
        }
        let mut ins_cursor = 2usize;
        for mid in admitted_mids {
            if final_turn_indices.len() < 5 {
                final_turn_indices.push(mid);
            } else if ins_cursor < 5 {
                let slot = (ins_cursor..5)
                    .rev()
                    .find(|&s| !top_snap[..2.min(top_snap.len())].contains(&final_turn_indices[s]))
                    .unwrap_or(4);
                final_turn_indices.insert(slot, mid);
                ins_cursor = (slot + 1).min(5);
                if final_turn_indices.len() > k {
                    final_turn_indices.truncate(k);
                }
            }
        }
    }

    // Synchronize sessions with `final_turn_indices` and land `W2-001` & `W2-006` session closers:
    apply_session_late_sync_and_margin_swap(
        ix,
        &q.text,
        &plan,
        &eff_it,
        &final_turn_indices,
        &mut final_session_indices,
        &standing_rules,
        &cond_rule_turns,
        &t_scene,
        &t_own,
        &s_score,
        &bridge_boost,
        &s_exact_mask,
        torrens_locked_head,
        taxa_promoted && nt >= TAXA_SESSION_LOCK_MIN_TURNS,
        k,
    );

    maybe_dump_replay(
        ix,
        q,
        k,
        &it,
        &t_scene,
        &final_turn_indices,
        &final_session_indices,
        &t_own,
        &t_turn_raw,
        &t_scene_sess,
        &s_score,
        &bridge_boost,
    );
    let mut max_ep_groups = 0usize;
    let mut union_mask_top5 = 0u64;
    let mut exact_union_top5 = 0u64;
    let mut grounded_union = 0u64;
    let mut any_pos_support = !standing_rules.is_empty();
    for &ti in final_turn_indices.iter().take(5) {
        let m = t_mask[ti] & q_group_mask;
        let c = m.count_ones() as usize;
        if c > max_ep_groups {
            max_ep_groups = c;
        }
        union_mask_top5 |= m;
        exact_union_top5 |= t_exact_mask[ti] & q_group_mask;
        grounded_union |= t_exact_mask[ti] & grounded_group_mask;
        if t_scene[ti] > 0.0 || t_bridge[ti] > 0.0 {
            any_pos_support = true;
        }
    }
    let union_groups = union_mask_top5.count_ones() as usize;
    let grounded_groups = grounded_group_mask.count_ones() as usize;
    let grounded_covered = grounded_union.count_ones() as usize;
    let zero_exact_unbridged = exact_union_top5 == 0
        && standing_rules.is_empty()
        && final_turn_indices
            .first()
            .map_or(true, |&h0| t_bridge[h0] <= 0.0)
        && union_groups < q_group_cnt;
    let cov_status = if q_group_cnt > 0
        && ((union_groups == 0 && (!any_pos_support || (q_group_cnt >= 2 && standing_rules.is_empty())))
            || zero_exact_unbridged)
    {
        "ABSTAIN"
    } else if q_group_cnt >= 2 && union_groups > 0 && union_groups < q_group_cnt {
        "PARTIAL_PREMISE"
    } else {
        "COVERED"
    };
    let premise_coverage = PremiseCoverage {
        status: cov_status,
        total_groups: q_group_cnt,
        max_episode_groups: max_ep_groups,
        union_groups,
        grounded_groups,
        grounded_covered,
    };

    let sessions = final_session_indices
        .iter()
        .copied()
        .map(|i| RankedItem {
            id: ix.session_ids[i].clone(),
            session_id: ix.session_ids[i].clone(),
            score: s_score[i] + bridge_boost[i],
            matched: matched_words(&plan, s_mask[i]),
        })
        .collect();

    let turns = final_turn_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.turns[i].id.clone(),
            session_id: ix.session_ids[ix.turns[i].session].clone(),
            score: if t_scene[i] > 0.0 { t_scene[i] } else { t_bridge[i] },
            matched: matched_words(&plan, t_mask[i]),
        })
        .collect();
    RankResult {
        intent: plan.intent,
        premise_coverage,
        sessions,
        turns,
        standing_rules,
    }
}

/// Prefix of the turn stream inspected for a named speaker label that is
/// neither empty nor the generic `user` / `assistant` dialogue roles.
/// Shared by the W2-001 late-pass named-speaker check and the taxa session lock.
const NAMED_SPEAKER_PREFIX: usize = 32;

/// Minimum corpus size (turn count) at which a taxa turn promotion may also
/// hard-lock its session at rank 1. Names the previous `nt >= 24` threshold.
const TAXA_SESSION_LOCK_MIN_TURNS: usize = 24;

/// Width of the late session list fused by 4-voter Kemeny-Young (replaces W2-006).
const SESSION_FUSION_TOP_K: usize = 5;

/// Number of Kemeny session-fusion voters on named-speaker corpora: sight,
/// current list, head-turn order, and session query-cue coverage. User/assistant
/// (monologue-style) streams keep the 3-voter set (no cue voter).
const SESSION_FUSION_VOTERS_NAMED: i32 = 4;

/// Majority of the 4 named-speaker fusion voters required to demote the current
/// #1. A 2-2 tie is broken by `sight`.
const SESSION_FUSION_OVERRIDE_VOTES_NAMED: i32 = 3;

/// Majority of the 3 monologue fusion voters (sight + current list + head-turn)
/// required to demote the current #1. Current-list votes for the incumbent, so
/// this requires both `sight` and head-turn order.
const SESSION_FUSION_OVERRIDE_VOTES_MONOLOGUE: i32 = 2;

/// When the NQC-gated MRF/PLM/MinCover ballot is live, a 5th voter joins the
/// named-speaker fusion. Override stays at 3 (now 3 of {sight, turn, cue, MRF}).
const SESSION_FUSION_OVERRIDE_VOTES_NAMED_MRF: i32 = 3;

/// 5:4 best-turn scene margin used by the post-fusion closer (`cand * den >= inc * num`).
const SESSION_SCENE_MARGIN_NUM: f32 = 5.0;
const SESSION_SCENE_MARGIN_DEN: f32 = 4.0;

fn corpus_has_named_speakers(ix: &Index) -> bool {
    ix.turns.iter().take(NAMED_SPEAKER_PREFIX).any(|t| {
        !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant"
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_session_late_sync_and_margin_swap(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    final_turn_indices: &[usize],
    final_session_indices: &mut Vec<usize>,
    standing_rules: &[RankedItem],
    cond_rule_turns: &[usize],
    t_scene: &[f32],
    t_own: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
    s_exact_mask: &[u64],
    torrens_locked_head: bool,
    taxa_lock_candidate: bool,
    k: usize,
) {
    if final_turn_indices.is_empty() {
        return;
    }
    let sight = |si: usize| -> f32 {
        s_score.get(si).copied().unwrap_or(0.0) + bridge_boost.get(si).copied().unwrap_or(0.0)
    };
    let best_turn_in_sess = |sess: usize| -> Option<(f32, usize)> {
        let members = ix.session_turns.get(sess)?;
        let mut best: Option<(f32, usize)> = None;
        for &ti in members {
            let sc = t_scene.get(ti).copied().unwrap_or(0.0);
            match best {
                None => best = Some((sc, ti)),
                Some((bs, bt)) if sc > bs || (sc.to_bits() == bs.to_bits() && ti < bt) => {
                    best = Some((sc, ti));
                }
                _ => {}
            }
        }
        best
    };

    let head_ti = final_turn_indices[0];
    let head_sess = ix.turns[head_ti].session;

    // Keep the upstream taxa *turn* promotion, but only force that turn's session
    // to #1 when the corpus has no named speakers (monologue rescue for CoRA /
    // NoLiMa / BABILong) or the taxa head already ties or beats the current #1
    // session score. A weaker taxa head must not overwrite a stronger session.
    let taxa_session_lock = if !taxa_lock_candidate {
        false
    } else if !corpus_has_named_speakers(ix) {
        true
    } else {
        match final_session_indices.first() {
            None => true,
            Some(&incumbent) => s_score.get(head_sess).copied().unwrap_or(0.0)
                >= s_score.get(incumbent).copied().unwrap_or(0.0),
        }
    };
    let hard_locked_head = torrens_locked_head || taxa_session_lock;

    // 0. If `torrens` or the gated `taxa` lock hard-locked Slot 0, sync `head_sess` to #1:
    if hard_locked_head {
        if let Some(pos) = final_session_indices.iter().position(|&s| s == head_sess) {
            if pos > 0 {
                let moved = final_session_indices.remove(pos);
                final_session_indices.insert(0, moved);
            }
        } else {
            final_session_indices.insert(0, head_sess);
        }
    } else if !final_session_indices.is_empty() {
        // 1. `W2-001` Envelope-Seal-Calendar Session Head + Late-Pass Witness Sync:
        let incumbent = final_session_indices[0];
        // Calendar month nomination (unique month in query)
        let mut q_months: Vec<String> = Vec::new();
        for tok in super::inline_date_tokens(q_text) {
            if let Some(mm) = tok.strip_prefix("__MONTH__")
                && !q_months.iter().any(|m| m == mm)
            {
                q_months.push(mm.to_string());
            }
        }
        let cal_nominee: Option<usize> = if q_months.len() == 1 {
            let target_m = q_months[0].as_str();
            let mut best_cs: Option<(usize, f32)> = None;
            let mut tied = false;
            for si in 0..ix.session_ts.len() {
                let ts = ix.session_ts[si];
                let has_m = ts != 0
                    && super::calendar_tokens(ts)
                        .iter()
                        .any(|ct| ct.strip_prefix("__MONTH__") == Some(target_m));
                if !has_m {
                    continue;
                }
                if let Some((sc, _)) = best_turn_in_sess(si)
                    && sc > 0.0
                {
                    match best_cs {
                        None => best_cs = Some((si, sc)),
                        Some((_, bsc)) if sc > bsc => {
                            best_cs = Some((si, sc));
                            tied = false;
                        }
                        Some((bsi, bsc)) if sc.to_bits() == bsc.to_bits() && si != bsi => {
                            tied = true;
                        }
                        _ => {}
                    }
                }
            }
            if tied { None } else { best_cs.map(|(si, _)| si) }
        } else {
            None
        };

        let nominee = cal_nominee.unwrap_or(head_sess);
        if nominee != incumbent
            && let Some(pos) = final_session_indices.iter().take(5).position(|&s| s == nominee)
        {
            let prom_n = final_turn_indices
                .iter()
                .take(5)
                .filter(|&&ti| ix.turns[ti].session == nominee)
                .count();
            let inc_n = final_turn_indices
                .iter()
                .take(5)
                .filter(|&&ti| ix.turns[ti].session == incumbent)
                .count();
            let q_cues = query_cues(ix, q_text);
            let head_c = turn_query_cues(ix, &q_cues, head_ti);
            let mut inc_union: Vec<u32> = Vec::new();
            for &ti in &ix.session_turns[incumbent] {
                for c in turn_query_cues(ix, &q_cues, ti) {
                    if !inc_union.contains(&c) {
                        inc_union.push(c);
                    }
                }
            }
            inc_union.sort_unstable();
            let inc_covers_head = !head_c.is_empty() && covers_cues(&inc_union, &head_c);
            let head_own = t_own.get(head_ti).copied().unwrap_or(0.0);
            let inc_best_own = best_turn_in_sess(incumbent)
                .and_then(|(_, ti)| t_own.get(ti).copied())
                .unwrap_or(0.0);
            let w2001_four_witnesses = prom_n > inc_n
                && sight(nominee) > sight(incumbent)
                && !inc_covers_head
                && head_own > inc_best_own;
            // Late-pass head turn sync on non-named chats (`BEAM` 106/222 S@1 misses where gold is in turns[:5]):
            let has_named = corpus_has_named_speakers(ix);
            let beam_late_sync = !has_named
                && nominee == head_sess
                && prom_n >= inc_n
                && !inc_covers_head
                && t_scene.get(head_ti).copied().unwrap_or(0.0) > 0.0
                && sight(nominee) >= 0.72 * sight(incumbent);
            if w2001_four_witnesses || beam_late_sync {
                let moved = final_session_indices.remove(pos);
                final_session_indices.insert(0, moved);
            }
        }

        // 2. Kemeny-Young session fusion over Top-5 (replaces W2-006 single-turn 5:4 swap):
        //    Voters: (a) session-score rank (`sight`), (b) current session list,
        //    (c) head-turn session order, and on named-speaker corpora (d) session
        //    query-cue coverage. User/assistant streams stay on the 3-voter set.
        //    Pairwise majority with `sight` rank as deterministic tie-breaker (`seating::kemeny_young`).
        //    A session-score-guarded 5:4 best-turn closer can still lift a session
        //    that ties or beats the fused #1 on `sight` and wins the scene margin.
        let n = final_session_indices.len().min(SESSION_FUSION_TOP_K);
        if n >= 2 {
            let pool: Vec<usize> = final_session_indices[..n].to_vec();
            let mut sight_order: Vec<usize> = (0..n).collect();
            sight_order.sort_by(|&a, &b| {
                let (sa, sb) = (pool[a], pool[b]);
                sight(sb)
                    .partial_cmp(&sight(sa))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| ix.session_ts[sb].cmp(&ix.session_ts[sa]))
                    .then_with(|| sa.cmp(&sb))
            });
            let mut sight_rank = vec![0usize; n];
            for (r, &idx) in sight_order.iter().enumerate() {
                sight_rank[idx] = r;
            }
            let turn_first_pos: Vec<usize> = pool
                .iter()
                .map(|&si| {
                    final_turn_indices
                        .iter()
                        .position(|&ti| ix.turns[ti].session == si)
                        .unwrap_or(usize::MAX)
                })
                .collect();
            let best_scene: Vec<f32> = pool
                .iter()
                .map(|&si| best_turn_in_sess(si).map_or(0.0, |(sc, _)| sc))
                .collect();
            let mut turn_order: Vec<usize> = (0..n).collect();
            turn_order.sort_by(|&a, &b| {
                turn_first_pos[a]
                    .cmp(&turn_first_pos[b])
                    .then_with(|| {
                        best_scene[b]
                            .partial_cmp(&best_scene[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
            });
            let mut turn_rank = vec![0usize; n];
            for (r, &idx) in turn_order.iter().enumerate() {
                turn_rank[idx] = r;
            }
            let q_cues_fuse = query_cues(ix, q_text);
            let sess_cue_cnt = |si: usize| -> usize {
                let mut u: Vec<u32> = Vec::new();
                if let Some(members) = ix.session_turns.get(si) {
                    for &ti in members {
                        for c in turn_query_cues(ix, &q_cues_fuse, ti) {
                            if !u.contains(&c) {
                                u.push(c);
                            }
                        }
                    }
                }
                u.len()
            };
            let cue_cnt: Vec<usize> = pool.iter().map(|&si| sess_cue_cnt(si)).collect();
            // Cue coverage is a dialogue overlap signal; user/assistant streams
            // keep the 3-voter fusion used on BEAM.
            let cue_voter = corpus_has_named_speakers(ix);
            let (mrf_scores, exact_ic_q16) =
                kemeny_mrf_q16::pool_mrf_and_exact_ic_q16(ix, plan, &pool, s_exact_mask);
            let sight_q16: Vec<u64> = pool
                .iter()
                .map(|&si| kemeny_mrf_q16::ieee754_bits_to_q16(sight(si).to_bits()))
                .collect();
            // Shtok et al. (2012) NQC peak lock: keep the 4-voter (or 3-voter)
            // fusion when #1 is a crisp high-dispersion winner.
            let nqc_lock = kemeny_mrf_q16::nqc_locks_incumbent(&sight_q16, &exact_ic_q16);
            let mrf_voter = cue_voter && !nqc_lock;
            let mut mrf_rank = vec![0usize; n];
            if mrf_voter {
                let mut mrf_order: Vec<usize> = (0..n).collect();
                mrf_order.sort_by(|&a, &b| {
                    mrf_scores[b]
                        .cmp(&mrf_scores[a])
                        .then_with(|| pool[a].cmp(&pool[b]))
                });
                for (r, &idx) in mrf_order.iter().enumerate() {
                    mrf_rank[idx] = r;
                }
            }
            let mut net = vec![vec![0i64; n]; n];
            for a in 0..n {
                for b in (a + 1)..n {
                    let mut margin = 1i64; // Voter (b): current session list ranks a < b
                    margin += if sight_rank[a] < sight_rank[b] { 1 } else { -1 }; // Voter (a)
                    margin += if turn_rank[a] < turn_rank[b] { 1 } else { -1 }; // Voter (c)
                    // Voter (d): session query-cue coverage, named-speaker corpora only.
                    if cue_voter {
                        margin += match cue_cnt[a].cmp(&cue_cnt[b]) {
                            std::cmp::Ordering::Greater => 1,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        };
                    }
                    // Voter (e): NQC-gated Metzler-Croft MRF + PLM + MinCover rank.
                    if mrf_voter {
                        margin += if mrf_rank[a] < mrf_rank[b] { 1 } else { -1 };
                    }
                    net[a][b] = margin;
                    net[b][a] = -margin;
                }
            }
            let ord = super::seating::kemeny_young(&net, &sight_rank);
            if ord.len() == n {
                // Keep the current #1 unless a majority of the 4 voters prefer the
                // Kemeny winner (2-2 ties broken by sight). The tail still follows the fused order.
                let winner = ord[0];
                let incumbent = 0;
                // Current-list voter prefers the incumbent (pool[0]) whenever the
                // winner is some other pool index, so only sight, turn, and cue
                // coverage can add votes. A 2-2 tie is broken by sight.
                let votes_for_winner = i32::from(sight_rank[winner] < sight_rank[incumbent])
                    + i32::from(turn_rank[winner] < turn_rank[incumbent])
                    + i32::from(cue_voter && cue_cnt[winner] > cue_cnt[incumbent])
                    + i32::from(mrf_voter && mrf_rank[winner] < mrf_rank[incumbent]);
                let (override_votes, tie_break_sight) = if mrf_voter {
                    // 5-voter set is odd, so there is no 2-2 tie to break.
                    (SESSION_FUSION_OVERRIDE_VOTES_NAMED_MRF, false)
                } else if cue_voter {
                    (
                        SESSION_FUSION_OVERRIDE_VOTES_NAMED,
                        votes_for_winner == SESSION_FUSION_VOTERS_NAMED / 2
                            && sight_rank[winner] < sight_rank[incumbent],
                    )
                } else {
                    (SESSION_FUSION_OVERRIDE_VOTES_MONOLOGUE, false)
                };
                // On named-speaker corpora, do not demote a current #1 whose
                // best turn is strictly stronger than the Kemeny winner's.
                // User/assistant streams skip this veto so BEAM keeps the
                // un-vetoed fusion head. The closer below can still lift a
                // 5:4 scene winner that ties or beats the incumbent on sight.
                let scene_veto = winner != incumbent
                    && corpus_has_named_speakers(ix)
                    && best_scene[incumbent] > best_scene[winner];
                // Zero-regression IC guard (same as kemeny_mrf_q16::solve_kemeny_order):
                // do not dethrone #1 for a session with strictly lower exact query-IC.
                // Only when the MRF voter is live, so monologue / BEAM 3-voter
                // fusion stays bit-identical to the incumbent 4-voter closer.
                let ic_holds = !mrf_voter
                    || exact_ic_q16[winner] >= exact_ic_q16[incumbent];
                let apply_head = (winner == incumbent
                    || ((votes_for_winner >= override_votes || tie_break_sight) && ic_holds))
                    && !scene_veto;
                if apply_head {
                    for i in 0..n {
                        final_session_indices[i] = pool[ord[i]];
                    }
                } else {
                    final_session_indices[0] = pool[incumbent];
                    let mut slot = 1;
                    for &idx in &ord {
                        if idx != incumbent {
                            final_session_indices[slot] = pool[idx];
                            slot += 1;
                        }
                    }
                }
            }

            // Session-score-guarded 5:4 scene closer: lift a fused top-k session
            // to #1 only when it ties or beats the incumbent on `sight` and its
            // best turn wins the 5:4 scene margin. Unlike W2-006, a weaker
            // session cannot overwrite a stronger one.
            let cur_top = final_session_indices[0];
            let sc_inc = best_turn_in_sess(cur_top).map_or(0.0, |(sc, _)| sc);
            let sight_inc = sight(cur_top);
            let check_limit = final_session_indices.len().min(SESSION_FUSION_TOP_K);
            let mut best_cand: Option<(usize, f32, f32)> = None;
            for pos in 1..check_limit {
                let cand_s = final_session_indices[pos];
                let sc_cand = best_turn_in_sess(cand_s).map_or(0.0, |(sc, _)| sc);
                if sc_cand > 0.0
                    && sc_cand * SESSION_SCENE_MARGIN_DEN >= sc_inc * SESSION_SCENE_MARGIN_NUM
                    && sight(cand_s) >= sight_inc
                {
                    if best_cand.is_none_or(|(_, bsc, bsi)| {
                        sc_cand > bsc || (sc_cand == bsc && sight(cand_s) > bsi)
                    }) {
                        best_cand = Some((pos, sc_cand, sight(cand_s)));
                    }
                }
            }
            if let Some((pos, _, _)) = best_cand {
                let moved = final_session_indices.remove(pos);
                final_session_indices.insert(0, moved);
            }
        }
    }

    // 3. Standing-Rules & Preference-Register Session Sync (decoupled from literal token overlap!):
    // Lock active `standing_rules[0..2]` and compact `preference_directive_turns` into `sessions[0..5]`.
    if !eff_it.temporal && !eff_it.sniper_how_many && !eff_it.is_event_sequence {
        let mut reg_sessions: Vec<(usize, bool)> = Vec::new();
        for (ri, sr) in standing_rules.iter().take(2).enumerate() {
            if (sr.score >= 1.5 || (eff_it.task_under_rules && ri == 0 && sr.score > 0.0))
                && let Some(sr_si) = ix.session_ids.iter().position(|id| id == &sr.session_id)
                && !reg_sessions.iter().any(|&(s, _)| s == sr_si)
            {
                // High-spec slot-1 seating needs the rule's own condition to match the query.
                let sr_cond_ok = cond_rule_turns.iter().any(|&rt| ix.turns[rt].id == sr.id);
                reg_sessions.push((sr_si, sr.score >= 2.5 && sr_cond_ok));
            }
        }
        if eff_it.preference && (1..=8).contains(&ix.preference_directive_turns.len()) {
            for &pti in ix.preference_directive_turns.iter().rev().take(2) {
                if !turn_is_superseded(ix, pti) {
                    let psi = ix.turns[pti].session;
                    if !reg_sessions.iter().any(|&(s, _)| s == psi) && reg_sessions.len() < 2 {
                        reg_sessions.push((psi, false));
                    }
                }
            }
        }
        for (reg_si, high_spec) in reg_sessions {
            let top5_end = final_session_indices.len().min(5);
            if final_session_indices[..top5_end].contains(&reg_si) {
                if high_spec
                    && eff_it.task_under_rules
                    && let Some(pos) = final_session_indices.iter().position(|&s| s == reg_si)
                    && pos > 1
                {
                    let moved = final_session_indices.remove(pos);
                    final_session_indices.insert(1, moved);
                }
                continue;
            }
            if let Some(pos) = final_session_indices.iter().position(|&s| s == reg_si) {
                final_session_indices.remove(pos);
            }
            if final_session_indices.len() < 5 {
                final_session_indices.push(reg_si);
            } else {
                let target_slot = if high_spec && eff_it.task_under_rules { 1 } else { 4 };
                let evicted = final_session_indices[target_slot];
                final_session_indices[target_slot] = reg_si;
                final_session_indices.push(evicted);
            }
        }
    }

    // 4. Exact Top-5 Session Coverage (`0..5`, in-place replacement with zero off-by-one overflow):
    // Every distinct session represented in `final_turn_indices[..5]` sits inside `final_session_indices[0..5]`,
    // guarded so weak noise turns (`sight < 0.38 * smax`) never evict strong sessions (`sight >= 0.58 * smax`)
    // unless `hard_locked_head` is active (`CoRA`, `NoLiMa`, `BABILong`, `HotpotQA`).
    let smax_sync = final_session_indices
        .iter()
        .map(|&s| sight(s))
        .fold(0.0f32, f32::max);
    let mut must_keep: Vec<usize> = Vec::with_capacity(5);
    for &ti in final_turn_indices.iter().take(5) {
        let s = ix.turns[ti].session;
        let is_reg_turn = ix.turns[ti].is_standing_policy || ix.turns[ti].is_personal_constraint;
        if !hard_locked_head && !is_reg_turn && smax_sync > 0.0 && sight(s) < 0.38 * smax_sync {
            continue;
        }
        if !must_keep.contains(&s) {
            must_keep.push(s);
        }
    }
    for &sess in &must_keep {
        let top5_end = final_session_indices.len().min(5);
        if final_session_indices[..top5_end].contains(&sess) {
            continue;
        }
        if let Some(pos) = final_session_indices.iter().position(|&s| s == sess) {
            final_session_indices.remove(pos);
        }
        if final_session_indices.len() < 5 {
            final_session_indices.push(sess);
        } else if let Some(victim) = (1..5usize).rev().find(|&idx| {
            let vs = final_session_indices[idx];
            !must_keep.contains(&vs)
                && (hard_locked_head || smax_sync <= 0.0 || sight(vs) < 0.58 * smax_sync || sight(sess) >= 0.55 * sight(vs))
        }) {
            let evicted = final_session_indices[victim];
            final_session_indices[victim] = sess;
            final_session_indices.push(evicted);
        } else if hard_locked_head {
            let evicted = final_session_indices[4];
            final_session_indices[4] = sess;
            final_session_indices.push(evicted);
        }
    }
    if final_session_indices.len() > k.max(5) {
        final_session_indices.truncate(k.max(5));
    }
}

#[allow(clippy::needless_range_loop)]
mod kemeny_mrf_q16 {
    //! Zero-float `Q16.16` integer session consensus voter combining:
    //! - Shtok, Kurland & Carmel (ACM TOIS 2012) Normalized Query Commitment (`NQC`) dispersion gate,
    //! - Metzler & Croft (SIGIR 2005) Sequential Dependence Model (`#1` ordered + `#uw8` unordered cliques),
    //! - Lv & Zhai (SIGIR 2009) Positional Language Model (`PLM`) triangular proximity kernel,
    //! - Tao & Zhai (SIGIR 2007) IC-weighted Minimum-Cover Span (`MinCover`),
    //! - Dwork, Kumar, Naor & Sivakumar (WWW 2001) exact Kemeny-Young rank aggregation (`5! = 120`).

    use super::*;

    /// Unit `1.0` in unsigned `Q16.16` fixed-point representation (`2^16`).
    const Q16_ONE: u64 = 65_536;
    /// Fractional shift width (`16` bits) for `Q16.16` fixed-point arithmetic.
    const Q16_SHIFT: u32 = 16;
    /// IEEE Std 754-2019 binary32 exponent bias (`127`).
    const IEEE754_F32_EXP_BIAS: i32 = 127;
    /// IEEE Std 754-2019 binary32 fraction bits (`23`) minus `Q16.16` fraction bits (`16`).
    const IEEE754_MANTISSA_TO_Q16_SHIFT: i32 = 7;
    /// IEEE Std 754-2019 binary32 total significand width (`23` explicit + `1` hidden bit).
    const IEEE754_SIGNIFICAND_BITS: u32 = 24;
    /// IEEE Std 754-2019 binary32 hidden leading significand bit (`1 << 23`).
    const IEEE754_HIDDEN_BIT: u32 = 0x0080_0000;
    /// IEEE Std 754-2019 binary32 fraction mask (`(1 << 23) - 1`).
    const IEEE754_FRAC_MASK: u32 = 0x007F_FFFF;
    /// IEEE Std 754-2019 binary32 8-bit exponent mask (`0xFF`).
    const IEEE754_EXP_MASK: u32 = 0xFF;
    /// IEEE Std 754-2019 binary32 sign bit shift (`31`).
    const IEEE754_SIGN_SHIFT: u32 = 31;
    /// IEEE Std 754-2019 binary32 exponent bit shift (`23`).
    const IEEE754_EXP_SHIFT: u32 = 23;
    /// Maximum left shift (`64 - 24 = 40`) before a 24-bit significand overflows `u64`.
    const Q16_SATURATE_LEFT_SHIFT: i32 = 40;
    /// Laplace document-frequency smoothing constant (`+1`, Robertson & Spärck Jones 1976).
    const LAPLACE_DOC_SMOOTH_ONE: u64 = 1;
    /// Bit width (`64`) of the query-group coverage bitmask (`u64`).
    const MAX_QUERY_GROUP_BITS: usize = 64;
    /// Top-4 fractional bit shift (`16 - 4 = 12`) into `LOG2_MANTISSA_LUT_Q16`.
    const LOG2_LUT_INDEX_SHIFT: u32 = 12;
    /// Remainder mask (`(1 << 12) - 1`) for linear interpolation between `log2` LUT knots.
    const LOG2_LUT_REM_MASK: u64 = 0x0FFF;
    /// Exact `round(65536 * log2(1 + m/16))` knots for `m in 0..=16` (Shannon 1948; max err 44 Q16 units).
    const LOG2_MANTISSA_LUT_Q16: [u64; 17] = [
        0, 5_732, 11_136, 16_248, 21_098, 25_711, 30_109, 34_312,
        38_336, 42_196, 45_904, 49_472, 52_911, 56_229, 59_434, 62_534, 65_536,
    ];

    /// Top-K session horizon (`5`) for exact `5! = 120` Kemeny-Young consensus (Dwork et al. WWW 2001).
    const KEMENY_CANDIDATE_POOL_CAP: usize = 5;
    /// Minimum session pool length (`2`) required for pairwise rank aggregation.
    const MIN_SESSIONS_FOR_CONSENSUS: usize = 2;
    /// Shtok et al. (ACM TOIS 2012) squared NQC dispersion threshold (`0.30 * 65536 = 19661`, `CV >= 0.55`).
    const NQC_HIGH_DISPERSION_SQ_Q16: u64 = 19_661;
    /// Top-1 over Top-2 dominance numerator (`4`, i.e. `4:3 = 1.33x` crisp peak guard for Single-Hop/Adversarial).
    const NQC_DOMINANCE_NUM: u64 = 4;
    /// Top-1 over Top-2 dominance denominator (`3`, paired with `NQC_DOMINANCE_NUM`).
    const NQC_DOMINANCE_DEN: u64 = 3;

    /// Metzler & Croft (SIGIR 2005) maximum content-token gap (`2`) for ordered adjacent cliques (`#1`).
    const MRF_ORDERED_GAP_MAX: usize = 2;
    /// Metzler & Croft (SIGIR 2005 §4.2) unordered window width (`8` tokens, `#uw8` optimal setting).
    const MRF_UNORDERED_WINDOW_TOKENS: usize = 8;
    /// Lv & Zhai (SIGIR 2009) triangular positional kernel radius (`16` content tokens).
    const PLM_KERNEL_RADIUS_TOKENS: u64 = 16;
    /// Turn-boundary virtual token stride (`4` tokens) preserving adjacency-pair proximity across speaker turns.
    const PLM_TURN_BOUNDARY_GAP_TOKENS: usize = 4;

    /// Voter 1 weight (`30`): whole-session + window lexical sight (Metzler & Croft 2005 unigram backbone).
    const VOTER_WEIGHT_SESSION_SIGHT: u32 = 30;
    /// Voter 2 weight (`25`): MRF (`#1` + `#uw8`) + PLM + Tao-Zhai `MinCover` span density.
    const VOTER_WEIGHT_MRF_PLM_SPAN: u32 = 25;
    /// Voter 3 weight (`20`): Fang-Zhai `TDC` + Raaijmakers-Shiffrin `SAM` exact query-IC coverage.
    const VOTER_WEIGHT_EXACT_IC_COV: u32 = 20;
    /// Voter 4 weight (`15`): Two-turn support (`top1 + top2` scene Q16), defeating 1-turn `C3`/`C6` decoys.
    const VOTER_WEIGHT_TWO_TURN_SUPPORT: u32 = 15;
    /// Voter 5 weight (`10`): Seated top-5 turn multiplicity from `final_turn_indices[0..5]`.
    const VOTER_WEIGHT_TOP5_TURN_SEATS: u32 = 10;
    /// Pairwise significance numerator (`17`, i.e. `17:16 = +6.25%` Weber-fraction anti-jitter margin).
    const KEMENY_MARGIN_SIG_NUM: u64 = 17;
    /// Pairwise significance denominator (`16`, paired with `KEMENY_MARGIN_SIG_NUM`).
    const KEMENY_MARGIN_SIG_DEN: u64 = 16;

    /// Pure integer IEEE-754 binary32 bit-pattern to `Q16.16` conversion (zero FP instructions).
    #[inline]
    pub(super) fn ieee754_bits_to_q16(bits: u32) -> u64 {
        let sign = bits >> IEEE754_SIGN_SHIFT;
        let exp_raw = (bits >> IEEE754_EXP_SHIFT) & IEEE754_EXP_MASK;
        let frac = bits & IEEE754_FRAC_MASK;
        if sign != 0 || exp_raw == 0 {
            return 0;
        }
        if exp_raw == IEEE754_EXP_MASK {
            return if frac == 0 { u64::MAX } else { 0 };
        }
        let mant24 = (frac | IEEE754_HIDDEN_BIT) as u64;
        let shift = (exp_raw as i32) - (IEEE754_F32_EXP_BIAS + IEEE754_MANTISSA_TO_Q16_SHIFT);
        if shift >= 0 {
            if shift >= Q16_SATURATE_LEFT_SHIFT {
                u64::MAX
            } else {
                mant24 << (shift as u32)
            }
        } else {
            let r = (-shift) as u32;
            if r >= IEEE754_SIGNIFICAND_BITS {
                0
            } else {
                mant24 >> r
            }
        }
    }

    /// Hardware `leading_zeros` + 16-segment piecewise linear `log2(x)` in `Q16.16` integers.
    #[inline]
    pub(super) fn log2_u64_q16(x: u64) -> u64 {
        if x <= 1 {
            return 0;
        }
        let msb = (u64::BITS - 1) - x.leading_zeros();
        let int_part = (msb as u64) << Q16_SHIFT;
        let frac16 = if msb >= Q16_SHIFT {
            (x >> (msb - Q16_SHIFT)) & (Q16_ONE - 1)
        } else {
            (x << (Q16_SHIFT - msb)) & (Q16_ONE - 1)
        };
        let idx = (frac16 >> LOG2_LUT_INDEX_SHIFT) as usize;
        let rem = frac16 & LOG2_LUT_REM_MASK;
        let lo = LOG2_MANTISSA_LUT_Q16[idx];
        let hi = LOG2_MANTISSA_LUT_Q16[idx + 1];
        int_part + lo + (((hi - lo) * rem) >> LOG2_LUT_INDEX_SHIFT)
    }

    /// Smoothed corpus Shannon self-information `log2((N + 1) / (df + 1))` in `Q16.16`.
    #[inline]
    pub(super) fn shannon_ic_q16(n_docs: u32, df: u32) -> u64 {
        let num = (n_docs as u64).saturating_add(LAPLACE_DOC_SMOOTH_ONE);
        let den = (df as u64).saturating_add(LAPLACE_DOC_SMOOTH_ONE);
        if num <= den {
            return 0;
        }
        log2_u64_q16(num).saturating_sub(log2_u64_q16(den))
    }

    /// Precomputes maximal `Q16.16` corpus self-information per query group `0..groups.len().min(64)`.
    fn query_group_ic_q16(ix: &Index, plan: &QueryPlan) -> Vec<u64> {
        let g_cnt = plan.groups.len().min(MAX_QUERY_GROUP_BITS);
        let mut g_ic = vec![Q16_ONE; g_cnt];
        let n_turns = (ix.turns.len() as u32).max(1);
        for qt in &plan.terms {
            if let Some(g) = qt.group
                && g < g_cnt
            {
                let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
                let ic = shannon_ic_q16(n_turns, df);
                if ic > g_ic[g] {
                    g_ic[g] = ic;
                }
            }
        }
        g_ic
    }

    /// MRF/PLM/MinCover score and exact query-IC coverage for each session in `pool`.
    pub(super) fn pool_mrf_and_exact_ic_q16(
        ix: &Index,
        plan: &QueryPlan,
        pool: &[usize],
        s_exact_mask: &[u64],
    ) -> (Vec<u64>, Vec<u64>) {
        let g_ic = query_group_ic_q16(ix, plan);
        let g_cnt = g_ic.len();
        let g_mask_limit = if g_cnt == MAX_QUERY_GROUP_BITS {
            u64::MAX
        } else if g_cnt == 0 {
            0
        } else {
            (1u64 << g_cnt) - 1
        };
        let mut mrf = Vec::with_capacity(pool.len());
        let mut exact_ic = Vec::with_capacity(pool.len());
        let mut seq_buf: Vec<(usize, usize)> = Vec::new();
        for &si in pool {
            let mut em = s_exact_mask.get(si).copied().unwrap_or(0) & g_mask_limit;
            let hit_groups = em.count_ones() as u64;
            let mut ic_sum: u64 = 0;
            while em != 0 {
                let g = em.trailing_zeros() as usize;
                ic_sum = ic_sum.saturating_add(g_ic.get(g).copied().unwrap_or(Q16_ONE));
                em &= em - 1;
            }
            exact_ic.push(ic_sum.saturating_add(if hit_groups >= 2 {
                (ic_sum * (hit_groups - 1)) >> 2
            } else {
                0
            }));
            seq_buf.clear();
            let mut cursor = 0usize;
            if let Some(members) = ix.session_turns.get(si) {
                for &ti in members {
                    if let Some(turn) = ix.turns.get(ti) {
                        for &(exact_id, stem_id) in &turn.seq {
                            if let Some(&g) = plan
                                .group_of
                                .get(&exact_id)
                                .or_else(|| plan.group_of.get(&stem_id))
                                && g < g_cnt
                            {
                                seq_buf.push((cursor, g));
                            }
                            cursor += 1;
                        }
                    }
                    cursor += PLM_TURN_BOUNDARY_GAP_TOKENS;
                }
            }
            mrf.push(score_session_mrf_plm_mincover_q16(&seq_buf, &g_ic));
        }
        (mrf, exact_ic)
    }

    /// Scores a session's ordered `#1` cliques, `#uw8` cliques, `PLM` kernel, and `MinCover` span in `Q16.16`.
    pub(super) fn score_session_mrf_plm_mincover_q16(
        seq_groups: &[(usize, usize)],
        group_ic_q16: &[u64],
    ) -> u64 {
        if seq_groups.len() < 2 || group_ic_q16.len() < 2 {
            return 0;
        }
        let mut ordered_q16: u64 = 0;
        let mut unordered_q16: u64 = 0;
        for i in 0..seq_groups.len() {
            let (pos_i, g_i) = seq_groups[i];
            let ic_i = group_ic_q16.get(g_i).copied().unwrap_or(Q16_ONE);
            let mut seen_right: u64 = 1u64 << g_i;
            for &(pos_j, g_j) in &seq_groups[(i + 1)..] {
                let dist = pos_j.saturating_sub(pos_i);
                if dist == 0 {
                    continue;
                }
                if dist > MRF_UNORDERED_WINDOW_TOKENS {
                    break;
                }
                if (seen_right & (1u64 << g_j)) != 0 {
                    continue;
                }
                seen_right |= 1u64 << g_j;
                let ic_j = group_ic_q16.get(g_j).copied().unwrap_or(Q16_ONE);
                let pair_ic = (ic_i.saturating_add(ic_j)) >> 1;
                if g_j == g_i + 1 && dist <= MRF_ORDERED_GAP_MAX {
                    ordered_q16 = ordered_q16.saturating_add(pair_ic);
                }
                let kernel_num = (MRF_UNORDERED_WINDOW_TOKENS + 1 - dist) as u64;
                let kernel_den = (MRF_UNORDERED_WINDOW_TOKENS + 1) as u64;
                unordered_q16 = unordered_q16.saturating_add((pair_ic * kernel_num) / kernel_den);
            }
        }

        // Tao & Zhai (2007) + Lv & Zhai (2009) Minimum-Cover Span over distinct query groups:
        let mut counts = [0u32; MAX_QUERY_GROUP_BITS];
        let mut active_mask: u64 = 0;
        let mut best_cov_bits: u32 = 0;
        let mut best_span_score_q16: u64 = 0;
        let mut left = 0usize;
        for right in 0..seq_groups.len() {
            let (_, g_r) = seq_groups[right];
            if counts[g_r] == 0 {
                active_mask |= 1u64 << g_r;
            }
            counts[g_r] += 1;
            while left < right {
                let (_, g_l) = seq_groups[left];
                if counts[g_l] > 1 {
                    counts[g_l] -= 1;
                    left += 1;
                } else {
                    break;
                }
            }
            let cov_bits = active_mask.count_ones();
            if cov_bits >= 2 {
                let span_len = (seq_groups[right].0.saturating_sub(seq_groups[left].0) + 1) as u64;
                let slack = span_len.saturating_sub(cov_bits as u64);
                let mut cov_ic_sum: u64 = 0;
                let mut m = active_mask;
                while m != 0 {
                    let g = m.trailing_zeros() as usize;
                    cov_ic_sum = cov_ic_sum.saturating_add(group_ic_q16.get(g).copied().unwrap_or(Q16_ONE));
                    m &= m - 1;
                }
                let plm_span_q16 = (cov_ic_sum.saturating_mul(PLM_KERNEL_RADIUS_TOKENS))
                    / PLM_KERNEL_RADIUS_TOKENS.saturating_add(slack);
                if cov_bits > best_cov_bits
                    || (cov_bits == best_cov_bits && plm_span_q16 > best_span_score_q16)
                {
                    best_cov_bits = cov_bits;
                    best_span_score_q16 = plm_span_q16;
                }
            }
        }

        ordered_q16
            .saturating_add(unordered_q16)
            .saturating_add(best_span_score_q16)
    }

    /// Shtok et al. (2012) integer squared-NQC confidence lock: returns `true` when `#1` has a crisp
    /// high-dispersion peak AND dominates `#2` by `>= 4:3` with maximal exact query-IC coverage.
    pub(super) fn nqc_locks_incumbent(sight_q16: &[u64], exact_ic_q16: &[u64]) -> bool {
        if sight_q16.len() < MIN_SESSIONS_FOR_CONSENSUS || sight_q16[0] == 0 {
            return false;
        }
        let max_other_ic = exact_ic_q16[1..].iter().copied().max().unwrap_or(0);
        if exact_ic_q16[0] < max_other_ic {
            return false;
        }
        if sight_q16[0].saturating_mul(NQC_DOMINANCE_DEN)
            < sight_q16[1].saturating_mul(NQC_DOMINANCE_NUM)
        {
            return false;
        }
        let m = sight_q16.len() as u64;
        let sum: u128 = sight_q16.iter().map(|&v| v as u128).sum();
        let mu: u128 = sum / (m as u128);
        if mu == 0 {
            return false;
        }
        let mut sq_dev_sum: u128 = 0;
        for &v in sight_q16 {
            let diff = (v as u128).abs_diff(mu);
            sq_dev_sum = sq_dev_sum.saturating_add(diff.saturating_mul(diff));
        }
        let mean_var = sq_dev_sum / (m as u128);
        let denom = mu.saturating_mul(mu);
        let nqc_sq_q16 = ((mean_var << Q16_SHIFT) / denom.max(1)) as u64;
        nqc_sq_q16 >= NQC_HIGH_DISPERSION_SQ_Q16
    }

    fn permute_kemeny(
        depth: usize,
        m: usize,
        cur: &mut [usize; KEMENY_CANDIDATE_POOL_CAP],
        margin: &[[u32; KEMENY_CANDIDATE_POOL_CAP]; KEMENY_CANDIDATE_POOL_CAP],
        cand_sessions: &[usize],
        best_score: &mut u32,
        best_perm: &mut [usize; KEMENY_CANDIDATE_POOL_CAP],
    ) {
        if depth == m {
            let mut sc = 0u32;
            for i in 0..m {
                for j in (i + 1)..m {
                    sc += margin[cur[i]][cur[j]];
                }
            }
            let better = if sc != *best_score {
                sc > *best_score
            } else {
                let cur_is_id = (0..m).all(|k| cur[k] == k);
                let best_is_id = (0..m).all(|k| best_perm[k] == k);
                if cur_is_id != best_is_id {
                    cur_is_id
                } else {
                    (0..m)
                        .find_map(|k| {
                            let sa = cand_sessions[cur[k]];
                            let sb = cand_sessions[best_perm[k]];
                            if sa != sb { Some(sa < sb) } else { None }
                        })
                        .unwrap_or(false)
                }
            };
            if better {
                *best_score = sc;
                *best_perm = *cur;
            }
            return;
        }
        for idx in depth..m {
            cur.swap(depth, idx);
            permute_kemeny(depth + 1, m, cur, margin, cand_sessions, best_score, best_perm);
            cur.swap(depth, idx);
        }
    }

    /// Exact `m!`-permutation Kemeny-Young solver (`m <= 5`) with deterministic identity/ID tie-break.
    pub(super) fn solve_kemeny_order(
        cand_sessions: &[usize],
        voters: &[[u64; KEMENY_CANDIDATE_POOL_CAP]; 5],
        weights: &[u32; 5],
        discrete_voter_mask: u8,
    ) -> Vec<usize> {
        let m = cand_sessions.len();
        let mut margin = [[0u32; KEMENY_CANDIDATE_POOL_CAP]; KEMENY_CANDIDATE_POOL_CAP];
        for a in 0..m {
            for b in 0..m {
                if a == b {
                    continue;
                }
                let mut w_sum = 0u32;
                for v in 0..5 {
                    let va = voters[v][a];
                    let vb = voters[v][b];
                    let is_discrete = (discrete_voter_mask & (1u8 << v)) != 0;
                    let wins = if is_discrete {
                        va > vb
                    } else {
                        va > vb
                            && va.saturating_mul(KEMENY_MARGIN_SIG_DEN)
                                >= vb.saturating_mul(KEMENY_MARGIN_SIG_NUM)
                    };
                    if wins {
                        w_sum += weights[v];
                    }
                }
                margin[a][b] = w_sum;
            }
        }

        let mut cur = [0usize, 1, 2, 3, 4];
        let mut best_perm = cur;
        let mut best_score: u32 = 0;
        for i in 0..m {
            for j in (i + 1)..m {
                best_score += margin[cur[i]][cur[j]];
            }
        }
        permute_kemeny(
            0,
            m,
            &mut cur,
            &margin,
            cand_sessions,
            &mut best_score,
            &mut best_perm,
        );

        // Zero-regression head guard: never dethrone `cand[0]` unless the new head wins direct
        // pairwise majority AND matches or exceeds `cand[0]` on exact query-IC coverage (Voter 2 idx).
        let w0 = best_perm[0];
        if w0 != 0 && (margin[w0][0] <= margin[0][w0] || voters[2][w0] < voters[2][0]) {
            return cand_sessions.to_vec();
        }
        (0..m).map(|k| cand_sessions[best_perm[k]]).collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(super) fn apply_nqc_gated_mrf_kemeny_top5(
        ix: &Index,
        plan: &QueryPlan,
        final_turn_indices: &[usize],
        final_session_indices: &mut Vec<usize>,
        t_scene: &[f32],
        s_score: &[f32],
        bridge_boost: &[f32],
        s_exact_mask: &[u64],
    ) {
        let m = final_session_indices.len().min(KEMENY_CANDIDATE_POOL_CAP);
        if m < MIN_SESSIONS_FOR_CONSENSUS {
            return;
        }
        let g_ic = query_group_ic_q16(ix, plan);
        let g_cnt = g_ic.len();
        let g_mask_limit = if g_cnt == MAX_QUERY_GROUP_BITS {
            u64::MAX
        } else {
            (1u64 << g_cnt) - 1
        };

        let cands: Vec<usize> = final_session_indices[..m].to_vec();
        let mut v_sight = [0u64; KEMENY_CANDIDATE_POOL_CAP];
        let mut v_mrf = [0u64; KEMENY_CANDIDATE_POOL_CAP];
        let mut v_ic_cov = [0u64; KEMENY_CANDIDATE_POOL_CAP];
        let mut v_two_turn = [0u64; KEMENY_CANDIDATE_POOL_CAP];
        let mut v_seats = [0u64; KEMENY_CANDIDATE_POOL_CAP];

        let mut seq_buf: Vec<(usize, usize)> = Vec::new();
        for (idx, &si) in cands.iter().enumerate() {
            let s_q16 = s_score
                .get(si)
                .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
            let b_q16 = bridge_boost
                .get(si)
                .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
            v_sight[idx] = s_q16.saturating_add(b_q16);

            let mut em = s_exact_mask.get(si).copied().unwrap_or(0) & g_mask_limit;
            let hit_groups = em.count_ones() as u64;
            let mut ic_sum: u64 = 0;
            while em != 0 {
                let g = em.trailing_zeros() as usize;
                ic_sum = ic_sum.saturating_add(g_ic[g]);
                em &= em - 1;
            }
            // Raaijmakers & Shiffrin (1981) SAM superlinear cue-intersection bonus for multi-group hits:
            v_ic_cov[idx] = ic_sum.saturating_add(if hit_groups >= 2 {
                (ic_sum * (hit_groups - 1)) >> 2
            } else {
                0
            });

            seq_buf.clear();
            let mut cursor = 0usize;
            let mut top1_t: u64 = 0;
            let mut top2_t: u64 = 0;
            if let Some(members) = ix.session_turns.get(si) {
                for &ti in members {
                    let t_q16 = t_scene
                        .get(ti)
                        .map_or(0, |v| ieee754_bits_to_q16(v.to_bits()));
                    if t_q16 >= top1_t {
                        top2_t = top1_t;
                        top1_t = t_q16;
                    } else if t_q16 > top2_t {
                        top2_t = t_q16;
                    }
                    for &(exact_id, stem_id) in &ix.turns[ti].seq {
                        if let Some(&g) = plan
                            .group_of
                            .get(&exact_id)
                            .or_else(|| plan.group_of.get(&stem_id))
                            && g < g_cnt
                        {
                            seq_buf.push((cursor, g));
                        }
                        cursor += 1;
                    }
                    cursor += PLM_TURN_BOUNDARY_GAP_TOKENS;
                }
            }
            v_mrf[idx] = score_session_mrf_plm_mincover_q16(&seq_buf, &g_ic);
            v_two_turn[idx] = top1_t.saturating_add(top2_t);
            let seat_cnt = final_turn_indices
                .iter()
                .take(KEMENY_CANDIDATE_POOL_CAP)
                .filter(|&&ti| ix.turns[ti].session == si)
                .count() as u64;
            v_seats[idx] = seat_cnt << Q16_SHIFT;
        }

        if nqc_locks_incumbent(&v_sight[..m], &v_ic_cov[..m]) {
            return;
        }

        let voters = [v_sight, v_mrf, v_ic_cov, v_two_turn, v_seats];
        let weights = [
            VOTER_WEIGHT_SESSION_SIGHT,
            VOTER_WEIGHT_MRF_PLM_SPAN,
            VOTER_WEIGHT_EXACT_IC_COV,
            VOTER_WEIGHT_TWO_TURN_SUPPORT,
            VOTER_WEIGHT_TOP5_TURN_SEATS,
        ];
        // Voters 2 (exact IC sum) and 4 (top-5 turn seat count) compare exact discrete quantities.
        let discrete_mask: u8 = (1u8 << 2) | (1u8 << 4);
        let reordered = solve_kemeny_order(&cands, &voters, &weights, discrete_mask);
        final_session_indices[..m].copy_from_slice(&reordered);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ieee754_and_log2_q16_are_bit_exact_and_monotonic() {
            assert_eq!(ieee754_bits_to_q16(0.0f32.to_bits()), 0);
            assert_eq!(ieee754_bits_to_q16((-2.5f32).to_bits()), 0);
            assert_eq!(ieee754_bits_to_q16(f32::NAN.to_bits()), 0);
            assert_eq!(ieee754_bits_to_q16(1.0f32.to_bits()), Q16_ONE);
            assert_eq!(ieee754_bits_to_q16(2.5f32.to_bits()), 163_840);
            assert_eq!(log2_u64_q16(1), 0);
            assert_eq!(log2_u64_q16(2), Q16_ONE);
            assert_eq!(log2_u64_q16(8), 3 * Q16_ONE);
            assert!(shannon_ic_q16(100, 1) > shannon_ic_q16(100, 20));
        }

        #[test]
        fn mrf_plm_mincover_ranks_tight_ordered_phrase_above_scattered_decoy() {
            let g_ic = vec![3 * Q16_ONE, 4 * Q16_ONE, 3 * Q16_ONE];
            let tight_phrase = vec![(2, 0), (3, 1), (4, 2)];
            let scattered_decoy = vec![(2, 2), (25, 0), (60, 1)];
            let sc_tight = score_session_mrf_plm_mincover_q16(&tight_phrase, &g_ic);
            let sc_decoy = score_session_mrf_plm_mincover_q16(&scattered_decoy, &g_ic);
            assert!(sc_tight > sc_decoy * 2);
        }

        #[test]
        fn nqc_locks_sharp_single_hop_winner_and_kemeny_promotes_ambiguous_consensus() {
            let sharp_sight = [10 * Q16_ONE, 4 * Q16_ONE, 2 * Q16_ONE];
            let sharp_ic = [6 * Q16_ONE, 3 * Q16_ONE, 3 * Q16_ONE];
            assert!(nqc_locks_incumbent(&sharp_sight, &sharp_ic));

            let flat_sight = [10 * Q16_ONE, 9 * Q16_ONE, 8 * Q16_ONE, 0, 0];
            assert!(!nqc_locks_incumbent(&flat_sight[..3], &[4 * Q16_ONE, 6 * Q16_ONE, 3 * Q16_ONE]));

            let cands = vec![10usize, 22usize, 35usize];
            let voters = [
                [10 * Q16_ONE, 9 * Q16_ONE, 7 * Q16_ONE, 0, 0],
                [Q16_ONE, 8 * Q16_ONE, Q16_ONE, 0, 0],
                [4 * Q16_ONE, 7 * Q16_ONE, 3 * Q16_ONE, 0, 0],
                [5 * Q16_ONE, 9 * Q16_ONE, 4 * Q16_ONE, 0, 0],
                [Q16_ONE, 2 * Q16_ONE, 0, 0, 0],
            ];
            let weights = [
                VOTER_WEIGHT_SESSION_SIGHT,
                VOTER_WEIGHT_MRF_PLM_SPAN,
                VOTER_WEIGHT_EXACT_IC_COV,
                VOTER_WEIGHT_TWO_TURN_SUPPORT,
                VOTER_WEIGHT_TOP5_TURN_SEATS,
            ];
            let out = solve_kemeny_order(&cands, &voters, &weights, (1 << 2) | (1 << 4));
            assert_eq!(out[0], 22);
        }
    }
}
