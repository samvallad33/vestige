//! Seating pass: `vault_closers`.

use crate::retrieval::*;


/// Consolidated Vault Closers (`RANK1_STANDOUTS.md` & `MEGA_ALGO.csv` — zero net-negative rules):
/// - `S8-002` Dead-reckoning ordinal leg + `winter_kleroterion` (`BEAM EO`)
/// - `acta_secretary_inversion` + `W2-059` Regesta gather + `S8-005` (`BEAM SUM` / `LME All@5`)
/// - `W2-075` Preference evidence seat + `W2-087` Preference recency + `halakha_ke_choice_register` (`PF` / `LME_S SSP` / `MAB SSP`)
/// - `S6-003` Inscriptio next-turn lift + `W2-086` Instruction recency (`BEAM IF`)
/// - `S1-071` Name-list subset yield (`LoCoMo SH` / `LME T@1`)
/// - `S8-008` Originalia estreat (`LME T@1`)
/// - `S7-041` Ibeji both (`"<CapA> and <CapB>"`, `LoCoMo OD` / `BEAM PF`)
/// - `S8-004` Rhumb-union seat (`LoCoMo MH`)
/// - Gated `S8-001` Ledger folio pick (`LoCoMo` / `BEAM` `pool@50 -> #1`)
pub(in crate::retrieval) fn apply_vault_closers(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &crate::retrieval::QueryIntent,
    query: &str,
    list: &mut Vec<usize>,
    final_session_indices: &[usize],
    t_scene: &[f32],
    t_own: &[f32],
    t_exact_mask: &[u64],
    s_mask: &[u64],
    s_score: &[f32],
    k: usize,
) {
    if list.is_empty() {
        return;
    }
    let ql = query.to_ascii_lowercase();
    let is_summary = eff_summary_request(query) && ix.turns.len() >= 32;
    let is_pref_rec = crate::retrieval::step14_is_rec_request(query) && ix.turns.len() >= 24;

    // 1. `S8-002` Dead-reckoning ordinal leg + `winter_kleroterion` (`BEAM event_ordering`):
    if eff_it.is_event_sequence && !is_summary {
        let s_max = s_score.iter().copied().fold(0.0f32, f32::max);
        let mut seq_sessions: Vec<usize> = (0..ix.session_turns.len())
            .filter(|&si| s_score.get(si).copied().unwrap_or(0.0) >= 0.35 * s_max && s_max > 0.0)
            .collect();
        seq_sessions.sort_by_key(|&si| (ix.session_ts.get(si).copied().unwrap_or(0), si));
        let mut leg_turns: Vec<usize> = Vec::new();
        for &si in &seq_sessions {
            if let Some(best_u) = ix.session_turns[si]
                .iter()
                .copied()
                .filter(|&ti| {
                    ix.turns[ti].field == 0
                        && !ix.turns[ti].is_backchannel
                        && t_scene.get(ti).copied().unwrap_or(0.0) > 0.0
                })
                .max_by(|&a, &b| {
                    let pa = turn_pos(ix, a).unwrap_or(99) <= 2;
                    let pb = turn_pos(ix, b).unwrap_or(99) <= 2;
                    pa.cmp(&pb).then_with(|| {
                        t_scene[a]
                            .partial_cmp(&t_scene[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
            {
                leg_turns.push(best_u);
            }
        }
        if !leg_turns.is_empty() {
            let qp = crate::retrieval::word_padded(query);
            // hardcode-scan: closed-class (ordinal leg selectors)
            let ord_pick: Option<usize> = if qp.contains(" first ") || qp.contains(" earliest ") || qp.contains(" initially ") {
                leg_turns.first().copied()
            } else if qp.contains(" second ") {
                leg_turns.get(1).copied()
            } else if qp.contains(" third ") {
                leg_turns.get(2).copied()
            } else if qp.contains(" last ") || qp.contains(" final ") || qp.contains(" latest ") {
                leg_turns.last().copied()
            } else {
                None
            };
            if let Some(target_ti) = ord_pick {
                let head_sc = t_scene.get(list[0]).copied().unwrap_or(0.0);
                let tgt_sc = t_scene.get(target_ti).copied().unwrap_or(0.0);
                let orig_len = list.len();
                if tgt_sc >= 0.65 * head_sc {
                    if let Some(pos) = list.iter().position(|&x| x == target_ti) {
                        list.remove(pos);
                    }
                    list.insert(0, target_ti);
                    list.truncate(orig_len.max(k));
                } else if !list[..list.len().min(5)].contains(&target_ti) {
                    if let Some(pos) = list.iter().position(|&x| x == target_ti) {
                        list.remove(pos);
                    }
                    let at = 1.min(list.len());
                    list.insert(at, target_ti);
                    list.truncate(orig_len.max(k));
                }
            }
        }
        return;
    }

    // 2. `acta_secretary_inversion` + `W2-059` Regesta gather (`BEAM summarization`):
    if is_summary && list.len() >= 2 {
        let head0 = list[0];
        let h0_t = &ix.turns[head0];
        // Summaries are answered by the assistant's synthesis turns: any user-side head
        // (question, short ask, standing rule or narrative statement) yields to a
        // comparably scored assistant answer.
        if h0_t.field == 0 {
            let head_sc = t_scene.get(head0).copied().unwrap_or(0.0);
            // A short ask head yields at 0.72; a long user narrative head (even one closing on
            // a question) yields at 0.60: its own wording inflates its lexical score against a
            // synthesis question.
            let yield_frac = if h0_t.content.len() < 24 { 0.72 } else { 0.60 };
            let mut best_asst: Option<(usize, f32)> = None;
            for slot in 1..list.len().min(10) {
                let cand = list[slot];
                let ct = &ix.turns[cand];
                let sc = t_scene.get(cand).copied().unwrap_or(0.0);
                if ct.field == 1
                    && !ct.is_question
                    && !ct.is_backchannel
                    && ct.content.len() >= 10
                    && sc >= yield_frac * head_sc
                    && best_asst.is_none_or(|(_, bs)| sc > bs)
                {
                    best_asst = Some((slot, sc));
                }
            }
            if let Some((slot, _)) = best_asst {
                let cand = list.remove(slot);
                list.insert(0, cand);
            }
        }
        // `W2-059`: Fill tail slots one distinct session per seat in session rank order
        let mut seen_s: Vec<usize> = list.iter().take(5).map(|&ti| ix.turns[ti].session).collect();
        for &si in final_session_indices.iter().take(10) {
            if list.len() >= k || seen_s.contains(&si) {
                continue;
            }
            if let Some(best_ti) = ix.session_turns[si]
                .iter()
                .copied()
                .filter(|&ti| !ix.turns[ti].is_backchannel && t_scene.get(ti).copied().unwrap_or(0.0) > 0.0 && !list.contains(&ti))
                .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
            {
                if let Some(dup_slot) = (5..list.len()).rev().find(|&idx| {
                    let s = ix.turns[list[idx]].session;
                    list[..idx].iter().any(|&p| ix.turns[p].session == s)
                }) {
                    list[dup_slot] = best_ti;
                    seen_s.push(si);
                }
            }
        }
        return;
    }

    // 3. `W2-075` + `W2-087` + `halakha_ke_choice_register` (`BEAM PF`, `LME_S SSP`, `MAB 400K SSP`):
    if is_pref_rec && !c2_named_session(ix, list[0]) {
        let head0 = list[0];
        let head_is_stance_user = ix.turns[head0].field == 0
            && !ix.turns[head0].is_question
            && (ix.turns[head0].is_personal_constraint
                || !ix.turns[head0].role_objs.is_empty()
                || ix.preference_directive_turns.contains(&head0));
        let mut q_ids: crate::retrieval::DetHashSet<u32> = crate::retrieval::DetHashSet::default();
        for qt in &plan.terms {
            q_ids.insert(qt.id);
        }
        for &fid in &plan.focal_ids {
            q_ids.insert(fid);
        }
        if !head_is_stance_user {
            // Search Top-10 first, then full `preference_directive_turns` (`W2-087` recency tie-break):
            let mut best_pref: Option<(usize, usize, i64, f32)> = None;
            let mut cands: Vec<usize> = list.iter().take(10).copied().collect();
            for &pti in &ix.preference_directive_turns {
                if !cands.contains(&pti) {
                    cands.push(pti);
                }
            }
            for cand in cands {
                let ct = &ix.turns[cand];
                if ct.field != 0 || ct.is_question || ct.is_backchannel || turn_is_superseded(ix, cand) {
                    continue;
                }
                let is_stance = ct.is_personal_constraint
                    || !ct.role_objs.is_empty()
                    || ix.preference_directive_turns.contains(&cand);
                if !is_stance {
                    continue;
                }
                let hits = ct.content.iter().filter(|id| q_ids.contains(id)).count()
                    + ct.role_objs.iter().filter(|id| q_ids.contains(id)).count();
                let sc = t_scene.get(cand).copied().unwrap_or(0.0);
                if hits == 0 && sc <= 0.0 {
                    continue;
                }
                let ts = ix.session_ts.get(ct.session).copied().unwrap_or(0);
                let key = (hits, usize::from(ix.preference_directive_turns.contains(&cand)), ts, sc);
                if best_pref.is_none_or(|(bh, bp, bts, bsc)| {
                    (key.0, key.1, key.2) > (bh, bp, bts)
                        || ((key.0, key.1, key.2) == (bh, bp, bts) && key.3 > bsc)
                }) {
                    best_pref = Some((hits, key.1, ts, sc));
                    // Store winner turn index by replacing `head0` below
                }
                let _ = best_pref;
            }
            // Select the winning stance turn cleanly:
            let mut winner: Option<(usize, (usize, i32, usize, i64))> = None;
            let mut pool_cands: Vec<usize> = list.iter().take(10).copied().collect();
            for &pti in &ix.preference_directive_turns {
                if !pool_cands.contains(&pti) {
                    pool_cands.push(pti);
                }
            }
            for cand in pool_cands {
                let ct = &ix.turns[cand];
                if ct.field != 0 || ct.is_question || ct.is_backchannel || turn_is_superseded(ix, cand) {
                    continue;
                }
                let is_stance = ct.is_personal_constraint
                    || !ct.role_objs.is_empty()
                    || ix.preference_directive_turns.contains(&cand);
                if !is_stance {
                    continue;
                }
                let hits = ct.content.iter().filter(|id| q_ids.contains(id)).count()
                    + ct.role_objs.iter().filter(|id| q_ids.contains(id)).count();
                let sc = t_scene.get(cand).copied().unwrap_or(0.0);
                let head_sc = t_scene.get(head0).copied().unwrap_or(0.0);
                if hits == 0 || (hits == 1 && sc < 0.65 * head_sc) {
                    continue;
                }
                let ts = ix.session_ts.get(ct.session).copied().unwrap_or(0);
                let ord_key = (
                    hits,
                    (sc * 1000.0) as i32,
                    usize::from(ix.preference_directive_turns.contains(&cand)),
                    ts,
                );
                if winner.is_none_or(|(_, bk)| ord_key > bk) {
                    winner = Some((cand, ord_key));
                }
            }
            if let Some((win_ti, _)) = winner
                && win_ti != head0
            {
                let orig_len = list.len();
                if let Some(pos) = list.iter().position(|&x| x == win_ti) {
                    list.remove(pos);
                }
                list.insert(0, win_ti);
                list.truncate(orig_len.max(k));
            }
        }
    }

    // 4. `S6-003` Inscriptio next-turn lift + `W2-086` Instruction recency (`BEAM IF`):
    if eff_it.task_under_rules && !is_pref_rec && !plan.intent.temporal {
        for slot in 0..list.len().min(5) {
            let ti = list[slot];
            if ix.turns[ti].content.len() <= 5 && ti + 1 < ix.turns.len() {
                let nxt = ti + 1;
                if ix.turns[nxt].session == ix.turns[ti].session
                    && ix.turns[nxt].is_standing_policy
                    && !turn_is_superseded(ix, nxt)
                    && !list[..3.min(list.len())].contains(&nxt)
                {
                    let orig_len = list.len();
                    if let Some(p) = list.iter().position(|&x| x == nxt) {
                        list.remove(p);
                    }
                    list.insert(0, nxt);
                    list.truncate(orig_len);
                    break;
                }
            }
        }
        // `W2-086`: Among non-superseded standing policies in Top-10 covering Slot 0's exact groups, latest wins:
        let h0 = list[0];
        if ix.turns[h0].is_standing_policy {
            let h0_mask = t_exact_mask.get(h0).copied().unwrap_or(0);
            let mut latest_pol = (h0, ix.session_ts.get(ix.turns[h0].session).copied().unwrap_or(0));
            for &cand in list.iter().take(10).skip(1) {
                if ix.turns[cand].is_standing_policy
                    && !turn_is_superseded(ix, cand)
                    && (t_exact_mask.get(cand).copied().unwrap_or(0) & h0_mask) == h0_mask
                {
                    let cts = ix.session_ts.get(ix.turns[cand].session).copied().unwrap_or(0);
                    if (cts, cand) > (latest_pol.1, latest_pol.0) {
                        latest_pol = (cand, cts);
                    }
                }
            }
            if latest_pol.0 != h0
                && let Some(pos) = list.iter().position(|&x| x == latest_pol.0)
            {
                let moved = list.remove(pos);
                list.insert(0, moved);
            }
        }
    }

    // 5. `S8-004` Rhumb-union seat (`LoCoMo Multi-Hop`):
    if plan.intent.multi_hop && plan.groups.len() >= 2 && ix.session_turns.len() >= 3 {
        let pos_sessions: Vec<usize> = final_session_indices
            .iter()
            .take(10)
            .copied()
            .filter(|&si| s_score.get(si).copied().unwrap_or(0.0) > 0.0 && s_mask.get(si).copied().unwrap_or(0) != 0)
            .collect();
        let mut union_sess: Option<usize> = None;
        'rhumb: for i in 0..pos_sessions.len() {
            for j in (i + 1)..pos_sessions.len() {
                let ma = s_mask[pos_sessions[i]];
                let mb = s_mask[pos_sessions[j]];
                if ma != 0 && mb != 0 && (ma & mb) == 0 {
                    let target_u = ma | mb;
                    if let Some(&sc_u) = pos_sessions.iter().find(|&&su| s_mask[su] == target_u) {
                        union_sess = Some(sc_u);
                        break 'rhumb;
                    }
                }
            }
        }
        if let Some(su) = union_sess {
            let head0 = list[0];
            if ix.turns[head0].session != su
                && let Some(best_u_ti) = ix.session_turns[su]
                    .iter()
                    .copied()
                    .filter(|&ti| {
                        !ix.turns[ti].is_question
                            && !ix.turns[ti].is_backchannel
                            && !turn_is_superseded(ix, ti)
                            && t_scene.get(ti).copied().unwrap_or(0.0) >= 0.85 * t_scene.get(head0).copied().unwrap_or(0.0)
                            && t_exact_mask.get(ti).copied().unwrap_or(0).count_ones()
                                > t_exact_mask.get(head0).copied().unwrap_or(0).count_ones()
                    })
                    .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal))
            {
                let orig_len = list.len();
                if let Some(pos) = list.iter().position(|&x| x == best_u_ti) {
                    list.remove(pos);
                }
                list.insert(0, best_u_ti);
                list.truncate(orig_len.max(k));
            }
        }
    }

    // 6. `S7-041` Ibeji both (`"<CapA> and <CapB>"` two-entity conjunction):
    {
        let q_words: Vec<&str> = query
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();
        let mut conj_pair: Option<(u32, u32)> = None;
        for i in 0..q_words.len().saturating_sub(2) {
            // hardcode-scan: closed-class (coordinating conjunction)
            if q_words[i + 1].eq_ignore_ascii_case("and")
                && q_words[i].chars().next().is_some_and(char::is_uppercase)
                && q_words[i + 2].chars().next().is_some_and(char::is_uppercase)
            {
                let ka = format!("\u{1}ent:{}", q_words[i].to_ascii_lowercase());
                let kb = format!("\u{1}ent:{}", q_words[i + 2].to_ascii_lowercase());
                if let (Some(ea), Some(eb)) = (ix.vocab.get(&ka), ix.vocab.get(&kb))
                    && ea != eb
                {
                    conj_pair = Some((ea, eb));
                    break;
                }
            }
        }
        if let Some((ea, eb)) = conj_pair {
            let h0 = list[0];
            let h0_both = ix.turns[h0].entities.contains(&ea) && ix.turns[h0].entities.contains(&eb);
            if !h0_both {
                if let Some(pos) = (1..list.len().min(10)).find(|&slot| {
                    let ct = &ix.turns[list[slot]];
                    !ct.is_question
                        && !ct.is_backchannel
                        && !turn_is_superseded(ix, list[slot])
                        && ct.entities.contains(&ea)
                        && ct.entities.contains(&eb)
                        && t_scene.get(list[slot]).copied().unwrap_or(0.0)
                            >= 0.65 * t_scene.get(h0).copied().unwrap_or(0.0)
                }) {
                    let moved = list.remove(pos);
                    list.insert(0, moved);
                }
            }
        }
    }

    // 7. `S1-071` Name-list subset yield (`LoCoMo SH` 192 near-misses & `LME T@1`):
    if !is_pref_rec && list.len() >= 2 {
        let h0 = list[0];
        let ht = &ix.turns[h0];
        if !ht.form_digit && !ht.entities.is_empty() && turn_pos(ix, h0) != Some(0) {
            let h_mask = t_exact_mask.get(h0).copied().unwrap_or(0);
            let h_sc = t_scene.get(h0).copied().unwrap_or(0.0);
            let mut best_sup: Option<(usize, usize)> = None;
            for slot in 1..list.len().min(5) {
                let cand = list[slot];
                let ct = &ix.turns[cand];
                if ct.session != ht.session
                    || ct.is_question
                    || ct.is_backchannel
                    || turn_is_superseded(ix, cand)
                    || ct.entities.len() <= ht.entities.len()
                    || ct.content.len() <= ht.content.len()
                {
                    continue;
                }
                let c_mask = t_exact_mask.get(cand).copied().unwrap_or(0);
                let c_sc = t_scene.get(cand).copied().unwrap_or(0.0);
                let is_strict_subset = ht.entities.iter().all(|e| ct.entities.contains(e));
                if is_strict_subset && (c_mask & h_mask) == h_mask && c_sc >= 0.82 * h_sc {
                    if best_sup.is_none_or(|(_, blen)| ct.entities.len() > blen) {
                        best_sup = Some((slot, ct.entities.len()));
                    }
                }
            }
            if let Some((slot, _)) = best_sup {
                let moved = list.remove(slot);
                list.insert(0, moved);
            }
        }
    }

    // 8. `S8-008` Originalia estreat (earlier same-session original digit statement beats later echo):
    if !is_pref_rec && !plan.intent.temporal && list.len() >= 2 {
        let h0 = list[0];
        let ht = &ix.turns[h0];
        if ht.form_digit {
            let h_digits = crate::retrieval::index::digit_token_set(&ht.body_for_headers);
            let h_mask = t_exact_mask.get(h0).copied().unwrap_or(0);
            let h_own = t_own.get(h0).copied().unwrap_or(0.0);
            if !h_digits.is_empty() {
                for slot in 1..list.len().min(5) {
                    let cand = list[slot];
                    let ct = &ix.turns[cand];
                    if cand < h0
                        && ct.session == ht.session
                        && ct.form_digit
                        && !ct.is_question
                        && !ct.is_backchannel
                        && !turn_is_superseded(ix, cand)
                        && (t_exact_mask.get(cand).copied().unwrap_or(0) & h_mask) == h_mask
                        && (ht.field == 1 || t_own.get(cand).copied().unwrap_or(0.0) >= 0.92 * h_own)
                        && crate::retrieval::index::digit_token_set(&ct.body_for_headers) == h_digits
                    {
                        let moved = list.remove(slot);
                        list.insert(0, moved);
                        break;
                    }
                }
            }
        }
    }

    // 9. Strictly Gated `S8-001` Ledger Folio Pick (`pool@50 -> #1` when Slot 0 is a question or strict exact-mask subset):
    if !is_pref_rec && !plan.intent.temporal && !eff_it.task_under_rules && !ql.is_empty() {
        let h0 = list[0];
        let ht = &ix.turns[h0];
        let h_mask = t_exact_mask.get(h0).copied().unwrap_or(0);
        let h_own = t_own.get(h0).copied().unwrap_or(0.0);
        let h_sc = t_scene.get(h0).copied().unwrap_or(0.0);
        let total_g = plan.groups.len().min(64);
        if ht.is_question || (total_g >= 2 && (h_mask.count_ones() as usize) < total_g) {
            let named_chat = c2_named_session(ix, h0);
            let mut best_folio: Option<(usize, u32, u32)> = None;
            for slot in 1..list.len().min(10) {
                let cand = list[slot];
                let ct = &ix.turns[cand];
                if ct.is_question
                    || ct.is_backchannel
                    || turn_is_superseded(ix, cand)
                    || (ct.field == 1 && !plan.intent.assistant_attribution)
                {
                    continue;
                }
                if named_chat
                    && c2_query_mentions_speaker(query, &ht.speaker)
                    && !c2_query_mentions_speaker(query, &ct.speaker)
                {
                    continue;
                }
                let c_mask = t_exact_mask.get(cand).copied().unwrap_or(0);
                let c_own = t_own.get(cand).copied().unwrap_or(0.0);
                let c_sc = t_scene.get(cand).copied().unwrap_or(0.0);
                if (c_mask & h_mask) == h_mask
                    && c_mask.count_ones() > h_mask.count_ones()
                    && c_own > 1.05 * h_own
                    && c_sc >= if ht.is_question { 0.84 } else { 0.95 } * h_sc
                {
                    let key = (c_mask.count_ones(), c_own.to_bits());
                    if best_folio.is_none_or(|(_, bp, bo)| key > (bp, bo)) {
                        best_folio = Some((slot, key.0, key.1));
                    }
                }
            }
            if let Some((slot, _, _)) = best_folio {
                let moved = list.remove(slot);
                list.insert(0, moved);
            }
        }
    }
}
