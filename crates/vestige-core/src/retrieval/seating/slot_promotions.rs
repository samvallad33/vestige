//! Seating pass: `slot_promotions`.

use crate::retrieval::*;


/// Slot-1 Assertion & Exact-Superset Promotion (`0..5` membership 100% invariant):
/// When `list[0]` in a user/assistant dialogue (`!c2_named_session`) was hoisted by
/// opener/elaboration heuristics despite having strictly weaker exact query-group coverage
/// and strictly lower `t_own` than a same-session user declarative in `list[1..5]`,
/// or when `list[0]` is a zero-exact-match assistant filler, swap `0` and `best_slot`.
pub(in crate::retrieval) fn apply_assertion_and_superset_slot1_promotion(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    list: &mut Vec<usize>,
    t_own: &[f32],
    t_scene: &[f32],
    t_exact_mask: &[u64],
) {
    if list.len() < 2 || eff_it.is_event_sequence {
        return;
    }
    let head = list[0];
    if c2_named_session(ix, head) {
        return;
    }
    let head_t = &ix.turns[head];
    let head_sess = head_t.session;
    let head_exact = t_exact_mask.get(head).copied().unwrap_or(0);
    let head_own = t_own.get(head).copied().unwrap_or(0.0);
    let head_sc = t_scene.get(head).copied().unwrap_or(0.0);
    if head_sc <= 0.0 {
        return;
    }
    let head_pos = turn_pos(ix, head);

    let window = list.len().min(5);
    // Assistant-attribution mode: when Slot 0 is a `field == 0` user prompt/question and a nearby
    // same-session substantive `field == 1` assistant answer sits in Slots 1..5 (`|pos| <= 2`),
    // promote the assistant answer turn to Slot 0 (`0..5` membership invariant).
    if eff_it.assistant_attribution {
        if head_t.field == 0 {
            let mut best_asst: Option<(usize, u32, f32)> = None;
            for slot in 1..window {
                let cand = list[slot];
                let cand_t = &ix.turns[cand];
                if cand_t.session != head_sess
                    || cand_t.field != 1
                    || cand_t.is_question
                    || cand_t.is_backchannel
                    || cand_t.body_for_headers.len() < 24
                    || turn_is_superseded(ix, cand)
                {
                    continue;
                }
                let near_dyad = match (head_pos, turn_pos(ix, cand)) {
                    (Some(hp), Some(cp)) => hp.abs_diff(cp) <= 2,
                    _ => false,
                };
                if !near_dyad {
                    continue;
                }
                let cand_exact = t_exact_mask.get(cand).copied().unwrap_or(0);
                let cand_own = t_own.get(cand).copied().unwrap_or(0.0);
                let cand_sc = t_scene.get(cand).copied().unwrap_or(0.0);
                if cand_sc >= 0.65 * head_sc
                    && (cand_exact != 0 || cand_own > 0.0 || head_t.is_question)
                {
                    let bits = cand_exact.count_ones();
                    let better = match best_asst {
                        None => true,
                        Some((_, b_bits, b_sc)) => bits > b_bits || (bits == b_bits && cand_sc > b_sc),
                    };
                    if better {
                        best_asst = Some((slot, bits, cand_sc));
                    }
                }
            }
            if let Some((slot, _, _)) = best_asst {
                list.swap(0, slot);
            }
        }
        return;
    }

    let mut best_slot: Option<(usize, u32, f32)> = None;
    for slot in 1..window {
        let cand = list[slot];
        let cand_t = &ix.turns[cand];
        if cand_t.session != head_sess
            || cand_t.field != 0
            || cand_t.is_question
            || cand_t.is_backchannel
            || turn_is_superseded(ix, cand)
        {
            continue;
        }
        if head_t.form_digit
            && !cand_t.form_digit
            && (plan.intent.sniper_how_many || plan.intent.sniper_when || head_exact == 0)
        {
            continue;
        }
        let cand_exact = t_exact_mask.get(cand).copied().unwrap_or(0);
        let cand_own = t_own.get(cand).copied().unwrap_or(0.0);
        let cand_sc = t_scene.get(cand).copied().unwrap_or(0.0);
        let near_dyad = match (head_pos, turn_pos(ix, cand)) {
            (Some(hp), Some(cp)) => hp.abs_diff(cp) <= 2,
            _ => false,
        };

        // 1. Assistant turn at Slot 0 yields to same-session user declarative assertion
        //    covering all exact groups of Slot 0.
        let asst_yields = head_t.field == 1
            && cand_exact != 0
            && (cand_exact & head_exact) == head_exact
            && cand_own > 0.0
            && cand_own >= 0.75 * head_own
            && cand_sc >= 0.80 * head_sc;
        // 2. Same-session user declarative with strict exact-group superset
        let strict_superset = head_t.field == 0
            && head_exact != 0
            && (cand_exact & head_exact) == head_exact
            && cand_exact.count_ones() > head_exact.count_ones()
            && cand_own >= 0.90 * head_own
            && cand_sc >= 0.72 * head_sc;
        // 3. Same-session user question opener yields to declarative covering all its exact groups
        let question_yields = head_t.field == 0
            && head_t.is_question
            && cand_exact != 0
            && (cand_exact & head_exact) == head_exact
            && cand_own >= 0.90 * head_own
            && cand_sc >= 0.80 * head_sc;
        // 4. Equal-bitmask user declarative tie: promote entity-richer, higher own-score,
        //    or substantive later assertion over casual pos-0 opener framing.
        let eq_mask_better = head_t.field == 0
            && cand_exact == head_exact
            && cand_exact != 0
            && cand_sc >= 0.88 * head_sc
            && ((cand_t.slot_entities_clean > head_t.slot_entities_clean && cand_own >= 0.92 * head_own)
                || (ix.session_turns.len() <= 4
                    && head_t.slot_entities_clean == 0
                    && turn_pos(ix, head) == Some(0)
                    && turn_pos(ix, cand).is_some_and(|p| p >= 2)
                    && !head_t.is_standing_policy
                    && !eff_it.task_under_rules
                    && cand_t.content.len() > head_t.content.len()
                    && cand_own >= 0.96 * head_own)
                || cand_own > 1.03 * head_own);
        // 5. Self-Payload vs. Carried-Echo Contrast (`|pos(r1) - pos(r2)| <= 2`):
        //    When Slot 0 won via neighbor carry (`t_own[r1] < 0.85 * t_own[r2]` or 0 clean slot entities
        //    while nearby declarative `r2` covers `>= r1` exact groups and carries clean slot entities),
        //    yield Slot 0 to the originating payload turn `r2`.
        let self_payload_wins = near_dyad
            && !head_t.is_standing_policy
            && cand_exact != 0
            && (cand_exact & head_exact) == head_exact
            && cand_sc >= 0.75 * head_sc
            && (head_own < 0.85 * cand_own
                || (head_t.slot_entities_clean == 0
                    && cand_t.slot_entities_clean >= 1
                    && cand_own >= 0.70 * head_own));

        if asst_yields || strict_superset || question_yields || eq_mask_better || self_payload_wins {
            let bits = cand_exact.count_ones();
            let better = match best_slot {
                None => true,
                Some((_, b_bits, b_own)) => bits > b_bits || (bits == b_bits && cand_own > b_own),
            };
            if better {
                best_slot = Some((slot, bits, cand_own));
            }
        }
    }

    if let Some((slot, _, _)) = best_slot {
        list.swap(0, slot);
    }
}

/// Speaker-Subject Alignment at Slot 0 for Named-Speaker Dialogues:
/// When the query names a single subject speaker `S` (not conjoined and not `WH_PERSON`),
/// if Slot 0 is spoken by a different speaker NOT named in the query (conversational mirror)
/// while a declarative turn in `list[1..]` is spoken by `S` and covers equal/higher exact
/// non-speaker groups with strong lexical grounding, promote `S`'s turn to Slot 0.
pub(in crate::retrieval) fn apply_speaker_subject_slot0_promotion(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
    t_own: &[f32],
    t_exact_mask: &[u64],
) {
    if list.len() < 2
        || plan.intent.is_event_sequence
        || plan.intent.assistant_attribution
        || c2_text_preference(query)
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return;
    }
    let head0 = list[0];
    if !c2_named_session(ix, head0) {
        return;
    }
    let q_nick_bind = ix.nicks.bindings_in(query);
    let mentions_sp_or_nick = |sp: &str| -> bool {
        c2_query_mentions_speaker(query, sp)
            || q_nick_bind.values().any(|full| full.eq_ignore_ascii_case(sp))
    };
    // Partner Question -> Queried-Speaker Answer Handoff across Top-4 (including Temporal queries!):
    // Whenever a question turn `q_ti` in `list[0..4]` is spoken by Partner A and `q_ti + 1` in the
    // same session is Speaker B's declarative answer (where the query asks about Speaker B),
    // seat `q_ti + 1` directly ahead of `q_ti` (so Slot 0 question hands Rank 1 to the actual answer!).
    for q_slot in 0..list.len().min(4) {
        let q_ti = list[q_slot];
        if ix.turns[q_ti].is_question
            && !mentions_sp_or_nick(&ix.turns[q_ti].speaker)
            && q_ti + 1 < ix.turns.len()
        {
            let reply_ti = q_ti + 1;
            let rt = &ix.turns[reply_ti];
            if rt.session == ix.turns[q_ti].session
                && !rt.is_pure_info_question
                && !rt.is_backchannel
                && !turn_is_superseded(ix, reply_ti)
                && mentions_sp_or_nick(&rt.speaker)
                && !c2_speaker_conjoined(query, &rt.speaker)
            {
                let orig_len = list.len();
                if let Some(pos) = list.iter().position(|&x| x == reply_ti) {
                    if pos <= q_slot {
                        continue;
                    }
                    list.remove(pos);
                }
                list.insert(q_slot, reply_ti);
                list.truncate(orig_len);
                if q_slot == 0 {
                    return;
                }
                break;
            }
        }
    }
    if plan.intent.temporal {
        return;
    }
    let head0 = list[0];
    let head_sp = ix.turns[head0].speaker.as_str();
    // Multi-Hop & Named-Dialogue 5:4 Witness Margin Swap for Ranks 2..5 -> Rank 1:
    // Even when `head0` shares the queried speaker, if `head0` is a question or a Rank 2..5
    // same-speaker declarative strictly covers more non-speaker query groups within 5:4 scene margin, promote it!
    let mut non_sp_mask_early = 0u64;
    for (g_idx, g_name) in plan.groups.iter().take(64).enumerate() {
        if !ix.turns.iter().any(|t| t.speaker.eq_ignore_ascii_case(g_name))
            && !q_nick_bind.contains_key(g_name.as_str())
        {
            non_sp_mask_early |= 1u64 << g_idx;
        }
    }
    if mentions_sp_or_nick(head_sp) {
        let h0_bits = (t_exact_mask.get(head0).copied().unwrap_or(0) & non_sp_mask_early).count_ones();
        let h0_sc = t_scene.get(head0).copied().unwrap_or(0.0);
        if h0_sc > 0.0 {
            let mut best_mh: Option<(usize, u32, f32)> = None;
            for slot in 1..list.len().min(5) {
                let cand = list[slot];
                let ct = &ix.turns[cand];
                if ct.is_question
                    || ct.is_backchannel
                    || turn_is_superseded(ix, cand)
                    || !mentions_sp_or_nick(&ct.speaker)
                {
                    continue;
                }
                let c_bits = (t_exact_mask.get(cand).copied().unwrap_or(0) & non_sp_mask_early).count_ones();
                let c_sc = t_scene.get(cand).copied().unwrap_or(0.0);
                let h0_is_q = ix.turns[head0].is_question;
                if (c_bits > h0_bits && c_sc * 5.0 >= h0_sc * 4.0)
                    || (h0_is_q && c_bits >= h0_bits && c_sc * 4.0 >= h0_sc * 3.0)
                {
                    if best_mh.is_none_or(|(_, bb, bs)| c_bits > bb || (c_bits == bb && c_sc > bs)) {
                        best_mh = Some((slot, c_bits, c_sc));
                    }
                }
            }
            if let Some((slot, _, _)) = best_mh {
                let cand = list.remove(slot);
                list.insert(0, cand);
            }
        }
        return;
    }
    let head_own = t_own.get(head0).copied().unwrap_or(0.0);
    let head_sc = t_scene.get(head0).copied().unwrap_or(0.0);
    if head_sc <= 0.0 {
        return;
    }

    // Build bitmask of non-speaker query groups so we preserve literal subject-predicate matches
    let mut non_sp_mask = 0u64;
    for (g_idx, g_name) in plan.groups.iter().enumerate() {
        if g_idx >= 64 {
            break;
        }
        if !ix.turns.iter().any(|t| t.speaker.eq_ignore_ascii_case(g_name)) {
            non_sp_mask |= 1u64 << g_idx;
        }
    }
    let head_non_sp = t_exact_mask.get(head0).copied().unwrap_or(0) & non_sp_mask;

    // Adversarial Presupposition Shield: if `head0` (declarative Partner B) is the sole
    // witness of a rare non-speaker query group (`df <= 8`) absent from all turns of the
    // queried speaker and exceeds max queried-speaker `t_own` IC by >= 0.5, preserve `head0`.
    let mut rare_non_sp_mask = 0u64;
    let nt_u32 = ix.turns.len().max(1) as u32;
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        if g >= 64 || (non_sp_mask & (1u64 << g)) == 0 {
            continue;
        }
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(nt_u32);
        if df <= 8 {
            rare_non_sp_mask |= 1u64 << g;
        }
    }
    let mut union_queried_rare_mask = 0u64;
    let mut max_queried_own = 0.0f32;
    for (ti, t) in ix.turns.iter().enumerate() {
        if c2_query_mentions_speaker(query, &t.speaker)
            && !c2_speaker_conjoined(query, &t.speaker)
        {
            union_queried_rare_mask |=
                t_exact_mask.get(ti).copied().unwrap_or(0) & rare_non_sp_mask;
            let o = t_own.get(ti).copied().unwrap_or(0.0);
            if o > max_queried_own {
                max_queried_own = o;
            }
        }
    }
    if !ix.turns[head0].is_pure_info_question
        && ((head_non_sp & rare_non_sp_mask) & !union_queried_rare_mask) != 0
        && head_own >= max_queried_own + 0.5
    {
        return;
    }

    let mut best: Option<(usize, u32, f32)> = None;
    for slot in 1..list.len().min(5) {
        let cand = list[slot];
        let ct = &ix.turns[cand];
        if ct.is_question
            || ct.is_backchannel
            || turn_is_superseded(ix, cand)
            || !c2_query_mentions_speaker(query, &ct.speaker)
            || c2_speaker_conjoined(query, &ct.speaker)
        {
            continue;
        }
        let cand_own = t_own.get(cand).copied().unwrap_or(0.0);
        let cand_sc = t_scene.get(cand).copied().unwrap_or(0.0);
        let cand_non_sp = t_exact_mask.get(cand).copied().unwrap_or(0) & non_sp_mask;
        if (cand_non_sp & head_non_sp) != head_non_sp {
            continue;
        }
        if cand_own >= 0.50 * head_own && cand_sc >= 0.45 * head_sc {
            let bits = cand_non_sp.count_ones();
            let better = match best {
                None => true,
                Some((_, bb, bs)) => bits > bb || (bits == bb && cand_sc > bs),
            };
            if better {
                best = Some((slot, bits, cand_sc));
            }
        }
    }

    if let Some((slot, _, _)) = best {
        let cand = list.remove(slot);
        list.insert(0, cand);
    }
}
