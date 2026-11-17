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

    // Named-speaker boost (never a filter): 1.06x tie-breaker gated on t_own[ti] > 0 or t_raw[ti] > 0.
    for t_sc in [&mut t_scene[..], &mut t_scene_sess[..]] {
        for ti in 0..nt {
            if t_raw[ti] <= 0.0 && t_own[ti] <= 0.0 {
                continue;
            }
            let sp = ix.turns[ti].speaker.as_str();
            if !sp.is_empty() && sp != "user" && sp != "assistant" && plan.groups.iter().any(|g| g == sp) {
                t_sc[ti] *= 1.06;
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

        // Standing policy invariant boost:
        for ti in 0..nt {
            let t = &ix.turns[ti];
            if t.is_standing_policy && t_scene[ti] > 0.0 {
                t_scene[ti] *= 2.5;
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
                        && t.is_question
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
    // Run session-path refinements first on a mask copy, then restore and run the
    // turn-path refinements so PPR seeds see pre-dyad masks (floor behavior).
    let t_mask_snap = t_mask.clone();
    let t_exact_snap = t_exact_mask.clone();
    apply_scene_refinements(&mut t_scene_sess, &mut t_mask, &mut t_exact_mask);
    t_mask.copy_from_slice(&t_mask_snap);
    t_exact_mask.copy_from_slice(&t_exact_snap);
    apply_scene_refinements(&mut t_scene, &mut t_mask, &mut t_exact_mask);
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
        || it.is_event_sequence;
    // Compound/prepositional task wording (" and ", "first ", "before ") must not
    // drop the standing-policy floor. Only count-snipers and event sequences do.
    let policy_blocked = it.sniper_how_many || it.is_event_sequence;
    let standing_rules = rank_standing_rules(ix, &plan, episodic_user_quote, &t_mask, k);
    // Winner floor stays inside task_under_rules, including multi-hop task rules.
    // v1 moved this seat outside that block and excluded multi-hop. Temporal queries
    // then received a global policy seat and displaced gold. v2 does not do that.
    let (s_policy_floored, t_policy_floored) = if it.task_under_rules && !policy_blocked {
        let mut s_floored = raise_floor(&mut s_score, lexical_smax, 0.68, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_standing_policy)
        });
        let mut t_floored = raise_floor(&mut t_scene, lexical_tmax, 0.68, |ti| ix.turns[ti].is_standing_policy);
        if let Some(top) = standing_rules.first() {
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
    if it.is_event_sequence {
        let _ = raise_floor(&mut s_score, lexical_smax, 0.42, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_initiative)
        });
        let _ = raise_floor(&mut t_scene, lexical_tmax, 0.38, |ti| ix.turns[ti].is_initiative);
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.38, |ti| ix.turns[ti].is_initiative);
    }
    // Mirror standing/pref floors onto the session-path t_scene (bridge_boost reads this).
    if it.task_under_rules && !policy_blocked {
        let _ = raise_floor(&mut t_scene_sess, lexical_tmax_sess, 0.68, |ti| ix.turns[ti].is_standing_policy);
        if let Some(top) = standing_rules.first() {
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
    let initiative_content: Vec<DetHashSet<u32>> = if it.is_event_sequence {
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
                if it.is_event_sequence {
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

    let final_session_indices: Vec<usize> = if p.enable_submodular && ns > 1 {
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
                        let same_date_seen = it.is_event_sequence && selected.iter().any(|&s| ix.session_ts[s] == ix.session_ts[si] && ix.session_ts[si] > 0);
                        let topic_seen = it.is_event_sequence
                            && !initiative_content[si].is_empty()
                            && selected.iter().any(|&s| jaccard(&initiative_content[s], &initiative_content[si]) >= 0.45);
                        let new_event = it.is_event_sequence && !initiative_content[si].is_empty() && !topic_seen;
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
                        } else if !it.is_event_sequence
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
            .into_iter()
            .filter(|&si| s_score[si] > 0.0 || bridge_boost[si] > 0.0)
            .take(k)
            .collect()
    };

    let strong_top_sessions = final_session_indices
        .iter()
        .take(5)
        .filter(|&&si| smax > 0.0 && !s_policy_floored[si] && !s_pref_floored[si] && (s_score[si] + bridge_boost[si]) >= 0.62 * smax)
        .count();
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

    // Weapon 4: Greedy Submodular Set Cover for Top-K Turn Selection
    let max_per_session_top5 = if it.is_event_sequence {
        1
    } else if it.multi_hop || (!pref_on && !it.task_under_rules && strong_top_sessions >= 2) {
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
                let topic_repeat = it.is_event_sequence
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
                let bypass_cap = short_reply && !it.is_event_sequence && allow_short_reply_bypass;
                let session_capped = selected_turns.len() < 5
                    && count >= max_per_session_top5
                    && !bypass_cap;
                if session_capped {
                    continue;
                }
                let diversity_factor = if short_reply && !it.is_event_sequence {
                    1.0
                } else if count == 0 && !topic_repeat {
                    1.0
                } else if it.is_event_sequence || topic_repeat {
                    0.15
                } else if it.multi_hop && count >= 1 && uncovered_count == 0.0 {
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
                let gain = sc * (1.0 + 0.60 * uncovered_ratio) * diversity_factor * couple;
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
                let bypass_cap = short_reply && !it.is_event_sequence && allow_short_reply_bypass;
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
    if !it.is_event_sequence
        && !pref_on
        && !it.task_under_rules
        && final_turn_indices.len() >= 5
        && smax > 0.0
    {
        let sess_limit = final_session_indices.len().min(5);
        for &si in &final_session_indices[1..sess_limit] {
            if s_score[si] + bridge_boost[si] < q4_sess_floor * smax {
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
                cnt >= 2 && t_scene[best_ti] >= q4_turn_floor * t_scene[vti]
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

    if !it.is_event_sequence && !pref_on && !it.task_under_rules {
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
            if !it.current_state {
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
            if it.is_event_sequence
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
                            if !it.is_event_sequence || final_turn_indices.len() >= 5 {
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
    if !it.is_event_sequence && !pref_on {
        apply_step11_kemeny_wh_slot_voter(ix, &q.text, &it, &t_scene, &mut final_turn_indices);
    }
    // C2 algos lane: slot-1 seat rules, all before RSB (which then seats behind the head).
    if !it.is_event_sequence {
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
    if !it.is_event_sequence {
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
    if !it.is_event_sequence && !it.assistant_attribution && !c2_text_preference(&q.text) {
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
    if !it.is_event_sequence && !it.assistant_attribution {
        apply_c2_subsequent_answer_seat(ix, &plan, &mut final_turn_indices, &t_scene);
    }
    // Cross-session opener-pair seat: a user opener at slot 1 yields to
    // another session's user opener already in the top 5 whose
    // immediately following assistant reply is also already in the
    // top 5, when that opener is at least as on-scene. Post-gatekeeper,
    // after #26. Later post-gatekeeper C2 seats may still run after
    // this pass. Skipped on preference. Does not run the gatekeeper
    // again and adds no WH-slot ballot.
    if !it.is_event_sequence && !it.assistant_attribution && !c2_text_preference(&q.text) {
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
    if !it.is_event_sequence && it.assistant_attribution {
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

    // [W2-033] Milestone Admission: ascending timestamp lattice for event sequences
    super::seating::apply_w2_milestone_admission(ix, &it, &mut final_turn_indices, &t_scene);

    // [W2-083 / W2-084] Summary Lead & Breadth Seats (Quantile Trajectory Pooling)
    super::seating::apply_w2_quantile_trajectory_seating(ix, &q.text, &it, &mut final_turn_indices, &t_scene);

    // [W2-028 / W2-031] StandingRegister constraint posting & assistant-echo swap
    super::seating::apply_w2_standing_register_seat(ix, &q.text, &plan, &mut final_turn_indices, &t_exact_mask);

    // #829 Moffat Rank-Biased Precision (p = 13/16 in Q16.16) tail slot reallocation (`5..k`, keeping `0..5` sealed),
    // powered by #821 QLD, #823/#825 BM25L+DFR, and #828 Rocchio PRF turn & session signals.
    if k >= 6 && !it.is_event_sequence && !pref_on && !it.task_under_rules {
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
                base_q16.saturating_add((bm25l_q16 + qld_q16) >> 6)
            })
            .collect();
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

    // #801/#806/#807 Generalized k-hop (k=3..6) Integer Viterbi + #802 6-Step Damped Integer RWR:
    // - For `it.multi_hop`, unseal Slots 1..3 when a decoded chain improves query-mask coverage
    //   and displace only redundant questions or duplicate-session covered turns.
    // - For literal / single-hop queries (`!it.multi_hop`), keep Slots 0..5 sealed and admit into
    //   redundant tail slots in 5..k.
    if k >= 6 {
        let viterbi_chain = super::viterbi::decode_viterbi_chain(ix, &plan, &final_turn_indices, &base_q16, &t_mask);
        if it.multi_hop {
            if let Some(ref chain) = viterbi_chain {
                if chain.score_q16 > 0
                    && chain.path.len() >= 2
                    && !final_turn_indices.is_empty()
                    && chain.union_mask.count_ones() > t_mask[final_turn_indices[0]].count_ones()
                {
                    for &hop_ti in &chain.path {
                        if ix.turns[hop_ti].is_question
                            || ix.turns[hop_ti].is_backchannel
                            || turn_is_superseded(ix, hop_ti)
                            || final_turn_indices.contains(&hop_ti)
                        {
                            continue;
                        }
                        let seat_end = 3.min(final_turn_indices.len());
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
                            if ix.turns[inc_ti].is_question || covered_dup {
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
                }
            }
        }

        let chain_turn = viterbi_chain
            .and_then(|chain| {
                if chain.score_q16 == 0 || chain.union_mask == 0 {
                    return None;
                }
                chain
                    .path
                    .into_iter()
                    .find(|ti| !final_turn_indices.contains(ti))
            })
            .or_else(|| super::ghostlink_dgs::synthesize_complement_turn(ix, &plan, &final_turn_indices))
            .or_else(|| {
                super::ghostlink_dgs::synthesize_multihop_rwr_turns(
                    ix,
                    &plan,
                    &final_turn_indices,
                    &t_mask,
                )
                .into_iter()
                .next()
            })
            .or_else(|| {
                super::viterbi::find_coupled_session_bridge_turns(
                    ix,
                    &plan,
                    &final_turn_indices,
                    &t_mask,
                )
                .into_iter()
                .next()
            });

        if let Some(cand_ti) = chain_turn {
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

    // #805 Allen Interval Algebra + #808 Temporal Contradiction Pruner + #812 Box-Cox Stabilization
    super::temporal_allen::apply_allen_boxcox_tail_refinement(ix, &it, &base_q16, &mut final_turn_indices, k);

    // #798/#799/#800/#826/#767 Block-Max WAND + Fagin TA bitpacked posting scan verification
    let wand_stats = super::wand::verify_query_block_max_wand(ix, &plan, &base_q16, k);

    // #814/#815/#816/#818/#819/#820/#822 Cycle E Taxonomic & Experiencer Preference Register +
    // #806 Mackworth AC-3 Arc Consistency + #827 ACT-R Base-Level Activation + #830 LockeLedger 256-Bucket Attribution
    {
        super::ac3::apply_ac3_consistency(ix, &it, &base_q16, &mut final_turn_indices, k);
        let pref_reg = super::taxa::PrefRegister::build(ix);
        let q_wilkins = plan
            .terms
            .iter()
            .find(|qt| qt.group.is_some())
            .and_then(|qt| ix.vocab.names.get(qt.id as usize))
            .map(|w| super::taxa::wilkins_term_code_u64(w))
            .unwrap_or(0);
        let q_zato = super::taxa::zatocode_u128(plan.terms.iter().map(|qt| qt.id));
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
        }
        let _attribution_root = ledger.finalize_root();
    }

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
    let turns = final_turn_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.turns[i].id.clone(),
            session_id: ix.session_ids[ix.turns[i].session].clone(),
            score: if t_scene[i] > 0.0 { t_scene[i] } else { t_bridge[i] },
            matched: matched_words(&plan, t_mask[i]),
        })
        .collect();
    RankResult { intent: plan.intent, sessions, turns, standing_rules }
}
