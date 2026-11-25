//! Seating pass: `w2_standing_register`.

use crate::retrieval::*;


/// [W2-028 / W2-031] Constraint Posting & Assistant Echo (`StandingRegister` 64-bit Bloom):
/// For closed-class planning/recommendation queries, checks the 64-bit `standing_register`
/// in O(1) and promotes a topic-matched durable user directive (`field == 0`) ahead of
/// same-session assistant echoes (`field == 1`) or zero-exact-match occupants in Slot 2 or 3.
pub(in crate::retrieval) fn apply_w2_standing_register_seat(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    final_turn_indices: &mut Vec<usize>,
    t_exact_mask: &[u64],
) {
    if ix.standing_register == 0
        || ix.standing_directive_turns.is_empty()
        || final_turn_indices.len() < 3
    {
        return;
    }
    let ql = q_text.to_ascii_lowercase();
    // hardcode-scan: closed-class (modal recommendation, planning, and task question openings)
    let is_planning_or_rec = [
        "what should", "how should", "where should", "which should",
        "can you", "could you", "would you", "should i", "should we",
        "do you recommend", "any suggestions", "any recommendations",
        "suggest", "recommend", "advice", "what can i", "how can i",
        "what are some", "when building", "i'm planning", "i'm looking",
        "i'm about to", "what typical", "how do i", "how to ",
    ]
    .iter()
    .any(|cue| ql.contains(cue));
    if !is_planning_or_rec {
        return;
    }

    let n_docs = (ix.t_own_stats.n as u32).max(1);
    let max_df = (n_docs / 16).clamp(3, 18);
    let q_domain: Vec<u32> = plan
        .terms
        .iter()
        .filter(|qt| qt.exact_hit)
        .filter_map(|qt| {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            (df > 0 && df <= max_df).then_some(qt.id)
        })
        .collect();
    if q_domain.is_empty() {
        return;
    }

    let mut q_bloom = 0u64;
    for &id in &q_domain {
        q_bloom |= 1u64 << (id % 64);
    }
    if (ix.standing_register & q_bloom) == 0 && ix.preference_directive_turns.is_empty() {
        return;
    }

    // hardcode-scan: closed-class (modal preference consultation, guarded against factual WH-past)
    let explicit_pref_consult = !ql.contains("did i ")
        && !ql.contains("did we ")
        && !ql.contains("did you ")
        && !ql.contains("have i ")
        && !ql.contains("was my ")
        && !ql.contains("were my ")
        && (ql.contains("suggest")
            || ql.contains("recommend")
            || ql.contains("advice")
            || ql.contains("any tips")
            || ql.contains("which should i")
            || ql.contains("what should i")
            || ql.contains("where do you think i should")
            || (wh_slots(q_text) == 0 && (ql.contains("should i ") || ql.contains("would you ")))
            || ql.starts_with("i'm planning")
            || ql.starts_with("i'm looking")
            || ql.starts_with("i'm about to")
            || ql.starts_with("i'm working on")
            || ql.starts_with("i'm updating"));

    let ns_u32 = (ix.session_turns.len() as u32).max(1);
    let mut q_hi_ic: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    for &id in &q_domain {
        let df = ix.s_stats.df.get(&id).copied().unwrap_or(1);
        q_hi_ic.insert(
            id,
            crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16(df, ns_u32),
        );
    }

    let mut best_dir: Option<(u64, usize, usize)> = None;
    let directive_pool: &[usize] = if explicit_pref_consult && !ix.preference_directive_turns.is_empty() {
        &ix.preference_directive_turns
    } else {
        &ix.standing_directive_turns
    };
    let q_toks_w2 = raw_tokens(q_text);
    let mut q_lex_nbr_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    if explicit_pref_consult && !ix.preference_directive_turns.is_empty() {
        for gw in plan.groups.iter().take(8) {
            if gw.starts_with('\u{1}') || !is_content(gw) {
                continue;
            }
            for (nb, _) in crate::wordnet::full_lexicon_neighbors(gw).into_iter().take(12) {
                if let Some(id) = ix.vocab.get(nb) {
                    q_lex_nbr_ids.insert(id);
                }
            }
        }
    }
    let check_already_seated = |dti: usize| -> bool {
        if explicit_pref_consult {
            final_turn_indices.first() == Some(&dti)
        } else {
            final_turn_indices[..3].contains(&dti)
        }
    };
    for &dti in directive_pool.iter().rev() {
        if turn_is_superseded(ix, dti) || check_already_seated(dti) {
            continue;
        }
        let turn_body_low = ix.turns[dti].body_for_headers.to_ascii_lowercase();
        // Guard against instruction echo / assistant restatement framing:
        if turn_body_low.starts_with("given that i have to")
            || turn_body_low.starts_with("given that i need")
            || turn_body_low.starts_with("since i have to")
            || turn_body_low.starts_with("since i need")
        {
            continue;
        }
        let toks_vec: Vec<&str> = ix.turns[dti]
            .body_for_headers
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();
        let is_deontic = ix.turns[dti].is_standing_policy
            || ix.preference_directive_turns.contains(&dti)
            || crate::retrieval::seating::standing_constraint_q16::is_standing_constraint_utterance(&toks_vec);
        let mut turn_ents: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &cid in &ix.turns[dti].content {
            turn_ents.insert(cid);
        }
        for f in &ix.turns[dti].feats {
            turn_ents.insert(f.id);
        }
        let ic_sc = crate::retrieval::seating::standing_constraint_q16::score_standing_constraint_q16(
            is_deontic,
            true,
            &turn_ents,
            &q_hi_ic,
            crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE,
        );
        let mut hits = q_domain
            .iter()
            .filter(|id| turn_ents.contains(id))
            .count();
        if hits == 0 && c2_policy_inflects_query(ix, dti, plan) {
            hits = 1;
        }
        if hits == 0 && ix.turns[dti].is_standing_policy {
            if ix
                .standing_rules
                .iter()
                .any(|r| r.turn == dti && rule_condition_matches(ix, r, plan, &q_toks_w2))
            {
                hits = 1;
            }
        }
        if hits == 0
            && ic_sc == 0
            && !q_lex_nbr_ids.is_empty()
            && ix.preference_directive_turns.contains(&dti)
            && turn_ents.iter().any(|id| q_lex_nbr_ids.contains(id))
        {
            hits = 1;
        }
        if hits == 0 && ic_sc == 0 {
            continue;
        }
        if best_dir.is_none_or(|(bic, bh, _)| (ic_sc, hits) > (bic, bh)) {
            best_dir = Some((ic_sc, hits, dti));
        }
    }

    // Fallback to general `standing_directive_turns` when `preference_directive_turns` had 0 domain overlap
    // (skipping standing policies on explicit preference consults so instructions never hijack preferences).
    if best_dir.is_none() && explicit_pref_consult && !ix.preference_directive_turns.is_empty() {
        for &dti in ix.standing_directive_turns.iter().rev() {
            if turn_is_superseded(ix, dti)
                || ix.turns[dti].is_standing_policy
                || final_turn_indices[..3].contains(&dti)
            {
                continue;
            }
            let hits = q_domain
                .iter()
                .filter(|id| ix.turns[dti].content.contains(id))
                .count();
            if hits > 0 && best_dir.is_none_or(|(_, bh, _)| hits > bh) {
                best_dir = Some((0, hits, dti));
            }
        }
    }

    let Some((_, best_hits, dir_ti)) = best_dir else { return };
    let dir_sess = ix.turns[dir_ti].session;

    // Explicit `"would you suggest"` / `"would you recommend"` preference consultation in long chats:
    // promote domain-matched compound preference directive (`"I prefer..."`, `"I'd like to..."`) directly to Slot 0!
    let dir_kratzer =
        crate::retrieval::seating::afriat_kratzer_garp_q16::analyze_kratzer_searle_turn(
            &ix.turns[dir_ti],
        );
    if explicit_pref_consult
        && ix.turns.len() >= 64
        && best_hits >= 1
        && ix.preference_directive_turns.contains(&dir_ti)
        && !(dir_kratzer.point
            == crate::retrieval::seating::afriat_kratzer_garp_q16::IllocPoint::Directive
            && ix.turns[dir_ti].is_pure_info_question)
    {
        let orig_len = final_turn_indices.len();
        if let Some(existing) = final_turn_indices.iter().position(|&t| t == dir_ti) {
            final_turn_indices.remove(existing);
        }
        final_turn_indices.insert(0, dir_ti);
        final_turn_indices.truncate(orig_len);
        return;
    }

    // Assistant echo swap: if slots 1 or 2 hold a same-session assistant echo (`field == 1`), swap it.
    if let Some(echo_slot) = (1..3usize).find(|&s| {
        let vti = final_turn_indices[s];
        ix.turns[vti].field == 1 && ix.turns[vti].session == dir_sess
    }) {
        if let Some(existing) = final_turn_indices.iter().position(|&t| t == dir_ti) {
            final_turn_indices.swap(echo_slot, existing);
        } else {
            final_turn_indices[echo_slot] = dir_ti;
        }
        return;
    }

    // Otherwise admit into Slot 2 (or Slot 1) only if current occupant is an assistant turn
    // or lacks exact query-term hits.
    for slot in [2usize, 1usize] {
        let vti = final_turn_indices[slot];
        if ix.turns[vti].field == 1 || t_exact_mask.get(vti).copied().unwrap_or(0) == 0 {
            if let Some(existing) = final_turn_indices.iter().position(|&t| t == dir_ti) {
                final_turn_indices.swap(slot, existing);
            } else {
                final_turn_indices[slot] = dir_ti;
            }
            return;
        }
    }
}
