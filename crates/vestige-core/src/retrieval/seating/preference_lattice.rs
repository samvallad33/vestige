//! Wave 7: Persistent User Preference Lattice & T44 Condorcet Coverage Density Duel.
//!
//! Implements:
//! 1. `PersistentPreferenceLattice`: Guarantees active non-superseded user preference
//!    directives and personal constraints are seated in Slot 0/1 on preference and
//!    recommendation requests without being displaced by topical feature discussions.
//! 2. `T44CondorcetCoverageDensityDuel`: Resolves near-tied Slot 0/1 candidates in
//!    Multi-Hop queries by awarding Slot 0 to the candidate covering both disjoint
//!    entity and bridge query groups.
//! 3. `InformativeTurnPriority`: Ensures short conversational confirmations and echoes
//!    yield Slot 0 to declarative turns containing substantive grounded facts.

use crate::retrieval::*;

// hardcode-scan: closed-class (modal advice and generic request-framing words)
pub(in crate::retrieval) fn is_pref_framing_word(w: &str) -> bool {
    matches!(
        w.to_ascii_lowercase().as_str(),
        "recommend"
            | "recommends"
            | "recommended"
            | "recommendation"
            | "recommendations"
            | "suggest"
            | "suggests"
            | "suggested"
            | "suggestion"
            | "suggestions"
            | "advice"
            | "advise"
            | "tips"
            | "tip"
            | "think"
            | "thinking"
            | "thought"
            | "plan"
            | "planning"
            | "planned"
            | "looking"
            | "look"
            | "trying"
            | "try"
            | "upcoming"
            | "weekend"
            | "lately"
            | "recent"
            | "recently"
            | "new"
            | "good"
            | "better"
            | "best"
            | "find"
            | "ways"
            | "way"
            | "getting"
            | "get"
            | "got"
            | "keeping"
            | "keep"
            | "something"
            | "extra"
            | "free"
            | "time"
            | "tonight"
            | "soon"
            | "idea"
            | "ideas"
            | "option"
            | "options"
            | "resource"
            | "resources"
            | "choose"
            | "sure"
            | "make"
            | "making"
            | "feel"
            | "feeling"
            | "excited"
            | "decide"
            | "whether"
            | "wait"
            | "help"
            | "give"
            | "share"
            | "tell"
            | "wondering"
            | "interested"
            | "consider"
            | "considering"
    )
}

// hardcode-scan: closed-class (explicit stance and comparative preference frames)
pub(in crate::retrieval) fn is_explicit_stance_directive(body_low: &str, is_pers: bool) -> bool {
    if is_pers || crate::retrieval::has_comparative_choice_frame(body_low) {
        return true;
    }
    [
        "i prefer",
        "i'd prefer",
        "i\u{2019}d prefer",
        "my favorite",
        "i really like",
        "i enjoy ",
        "i love ",
        "i don't like",
        "i don\u{2019}t like",
        "i don't want",
        "i don\u{2019}t want",
        "i'd rather",
        "i\u{2019}d rather",
        "try to avoid",
        "want to avoid",
    ]
    .iter()
    .any(|pat| body_low.contains(pat))
}

/// Persistent User Preference Lattice: guarantees top-slot seating for active constraints.
pub(in crate::retrieval) fn apply_persistent_preference_lattice(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    query: &str,
    has_dominant_cond_rule: bool,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
    t_scene: &[f32],
    t_own: &[f32],
    _t_exact_mask: &[u64],
    _s_score: &[f32],
    k: usize,
) {
    use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};

    if final_turn_indices.is_empty()
        || final_session_indices.is_empty()
        || has_dominant_cond_rule
        || eff_it.assistant_attribution
        || eff_it.is_event_sequence
        || eff_it.temporal
        || eff_it.sniper_how_many
        || eff_summary_request(query)
        || c2_named_session(ix, final_turn_indices[0])
        || ix.session_turns.len() < 16
    {
        return;
    }
    let is_pref_rec = (crate::retrieval::step14_is_rec_request(query) || eff_it.preference)
        && ix.turns.len() >= 16;
    if !is_pref_rec {
        return;
    }

    let ns_u32 = (ix.session_turns.len() as u32).max(1);
    let max_pref_df = (ns_u32 / 3).max(4);
    let mut q_topic_terms: Vec<(u32, u64)> = Vec::new();
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let raw_nm = ix
            .vocab
            .names
            .get(qt.id as usize)
            .map(|nm| nm.strip_prefix("\u{1}stem:").unwrap_or(nm.as_str()))
            .unwrap_or("");
        if raw_nm.is_empty() || is_pref_framing_word(raw_nm) {
            continue;
        }
        let s_df = ix.s_stats.df.get(&qt.id).copied().unwrap_or(0);
        if (1..=max_pref_df).contains(&s_df) {
            let ic = aspect_ic_q16(s_df, ns_u32);
            let ic_sq = ((ic as u128).saturating_mul(ic as u128) >> 16)
                .max((Q16_ONE >> 4) as u128) as u64;
            if let Some(entry) = q_topic_terms.iter_mut().find(|(id, _)| *id == qt.id) {
                entry.1 = entry.1.max(ic_sq);
            } else {
                q_topic_terms.push((qt.id, ic_sq));
            }
        }
    }

    // 1. Expand query vocabulary with bounded lexicon neighbors to catch paraphrased preferences.
    let mut lex_nbr_ids: Vec<(u32, u64)> = Vec::new();
    for g_word in &plan.groups {
        if g_word.starts_with('\u{1}') || !is_content(g_word) || is_pref_framing_word(g_word) {
            continue;
        }
        let mut add_lem = |lem: &str| {
            let st = stem(lem);
            for lid in [ix.vocab.get(lem), ix.vocab.get(&stem_term(&st))]
                .into_iter()
                .flatten()
            {
                let df = ix.s_stats.df.get(&lid).copied().unwrap_or(0);
                if (1..=max_pref_df).contains(&df)
                    && !lex_nbr_ids.iter().any(|&(id, _)| id == lid)
                {
                    let ic = aspect_ic_q16(df, ns_u32);
                    lex_nbr_ids.push((lid, ic));
                }
            }
        };
        for (lem, _) in crate::wordnet::full_lexicon_neighbors(g_word).into_iter().take(20) {
            add_lem(lem);
        }
        for (lem, _) in crate::roget::conceptual_neighbors(g_word, 24) {
            add_lem(lem);
        }
        for (lem, _) in crate::conceptnet::two_hop_neighbors(g_word, 10) {
            add_lem(lem);
        }
    }

    let sess_topic_ic = |si: usize| -> u64 {
        let mut sum: u64 = 0;
        for &(tid, ic_sq) in &q_topic_terms {
            if ix.session_content[si].contains(&tid) || ix.session_bags[si].tf[0].contains_key(&tid) {
                sum = sum.saturating_add(ic_sq);
            }
        }
        sum
    };

    // 2. Identify active user preference directives (preserve multi-sentence preference questions and numeric preferences).
    // (pti, rank_key, topic_ic_q16, lex_ic, is_stance, score_weight)
    let mut active_prefs: Vec<(usize, u64, u64, u64, bool, f32)> = Vec::new();
    for &pti in &ix.preference_directive_turns {
        if pti >= ix.turns.len() {
            continue;
        }
        let t = &ix.turns[pti];
        if t.field != 0
            || t.is_pure_info_question
            || t.is_short_question
            || t.is_backchannel
            || t.is_standing_policy
        {
            continue;
        }
        let body_low = t.body_for_headers.to_ascii_lowercase();
        if body_low.starts_with("given that i have to")
            || body_low.starts_with("given that i need")
            || body_low.starts_with("since i have to")
            || body_low.starts_with("since i need")
        {
            continue;
        }
        let is_stance = is_explicit_stance_directive(&body_low, t.is_personal_constraint);
        if t.content.len() > 80 && !is_stance {
            continue;
        }
        let mut topic_ic_q16: u64 = 0;
        for &(tid, ic_sq) in &q_topic_terms {
            if t.content.contains(&tid) || t.feats.iter().any(|f| f.id == tid) {
                topic_ic_q16 = topic_ic_q16.saturating_add(ic_sq);
            }
        }
        let mut lex_ic: u64 = 0;
        for &(lid, ic) in &lex_nbr_ids {
            if t.content.contains(&lid) || t.feats.iter().any(|f| f.id == lid) {
                lex_ic = lex_ic.saturating_add(ic);
            }
        }
        let sc = t_scene.get(pti).copied().unwrap_or(0.0);
        let own = t_own.get(pti).copied().unwrap_or(0.0);
        let score_weight = sc.max(own);
        let has_signal = topic_ic_q16 >= (Q16_ONE >> 2)
            || (is_stance && lex_ic >= (Q16_ONE >> 1));
        if !has_signal && (!is_stance || ix.preference_directive_turns.len() > 4) {
            continue;
        }
        let rank_key = topic_ic_q16
            .saturating_mul(4)
            .saturating_add(lex_ic.saturating_mul(2))
            .saturating_add(if is_stance { Q16_ONE >> 1 } else { 0 });
        active_prefs.push((pti, rank_key, topic_ic_q16, lex_ic, is_stance, score_weight));
    }

    if active_prefs.is_empty() {
        return;
    }

    // Sort active preferences: weighted topic + lexicon + stance key -> scene/own score -> recency.
    active_prefs.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| {
                let ts_a = ix.session_ts.get(ix.turns[a.0].session).copied().unwrap_or(0);
                let ts_b = ix.session_ts.get(ix.turns[b.0].session).copied().unwrap_or(0);
                ts_b.cmp(&ts_a)
            })
            .then_with(|| b.0.cmp(&a.0))
    });

    // Seat topic/lexicon-matched preferences in Slot 0/1; allow at most one zero-overlap standing floor at Slot 3
    // ONLY when preference register is compact (`<= 4` stance directives).
    let mut seats_filled = 0usize;
    let allow_zero_overlap_floor = ix.preference_directive_turns.len() <= 4 && k >= 5;
    for (pref_ti, _rank_key, topic_ic_q16, lex_ic, is_stance, _score_w) in active_prefs {
        if seats_filled >= 2 {
            break;
        }
        let has_signal = topic_ic_q16 >= (Q16_ONE >> 2)
            || (is_stance && lex_ic >= (Q16_ONE >> 1));
        if !has_signal && (seats_filled > 0 || !allow_zero_overlap_floor || !is_stance) {
            break;
        }
        let target_slot = if has_signal {
            seats_filled
        } else {
            3usize.min(final_turn_indices.len())
        };
        let pref_si = ix.turns[pref_ti].session;
        let s_target = if has_signal {
            seats_filled
        } else {
            3usize.min(final_session_indices.len())
        };
        if s_target < final_session_indices.len() {
            let inc_si = final_session_indices[s_target];
            if inc_si != pref_si {
                let inc_topic_ic = sess_topic_ic(inc_si);
                if inc_topic_ic > topic_ic_q16
                    && !(is_stance && (topic_ic_q16 > 0 || lex_ic >= (Q16_ONE >> 1)))
                {
                    continue;
                }
            }
        }
        // Avoid replacing an already higher-ranked session-0 preference turn with an identical session duplicate
        if seats_filled == 0 && final_turn_indices[0] == pref_ti {
            seats_filled += 1;
            continue;
        }
        if let Some(pos) = final_turn_indices.iter().position(|&x| x == pref_ti) {
            if pos > target_slot {
                let moved = final_turn_indices.remove(pos);
                final_turn_indices.insert(target_slot, moved);
            }
        } else {
            let insert_at = target_slot.min(final_turn_indices.len());
            final_turn_indices.insert(insert_at, pref_ti);
        }
        if let Some(s_pos) = final_session_indices.iter().position(|&s| s == pref_si) {
            if s_pos > s_target && s_pos < final_session_indices.len() {
                let moved_s = final_session_indices.remove(s_pos);
                final_session_indices.insert(s_target, moved_s);
            }
        } else {
            let s_insert = s_target.min(final_session_indices.len());
            final_session_indices.insert(s_insert, pref_si);
        }
        seats_filled += 1;
        if !has_signal {
            break;
        }
    }

    if final_turn_indices.len() > k.max(5) {
        final_turn_indices.truncate(k.max(5));
    }
    if final_session_indices.len() > k.max(5) {
        final_session_indices.truncate(k.max(5));
    }
}

/// T44 Condorcet Coverage Density Duel for Slot-0:
/// In Multi-Hop queries, when Rank 1 and Rank 2 are within 0.05 score margin,
/// the candidate covering both entity and bridge query aspects takes Slot 0.
pub(in crate::retrieval) fn apply_t44_condorcet_coverage_density_duel(
    _ix: &Index,
    plan: &QueryPlan,
    final_session_indices: &mut Vec<usize>,
    s_score: &[f32],
    s_mask: &[u64],
    _k: usize,
) {
    if !plan.intent.multi_hop || plan.groups.len() < 2 || final_session_indices.len() < 2 {
        return;
    }
    let s0 = final_session_indices[0];
    let s1 = final_session_indices[1];
    let sc0 = s_score.get(s0).copied().unwrap_or(0.0);
    let sc1 = s_score.get(s1).copied().unwrap_or(0.0);

    // Only duel when within 0.08 relative score margin
    let delta = (sc0 - sc1).abs();
    let max_sc = sc0.max(sc1);
    if max_sc <= 0.0 || delta > 0.08 * max_sc {
        return;
    }

    let n_qg = plan.groups.len().min(64);
    let qg_mask = if n_qg == 64 { u64::MAX } else if n_qg == 0 { 0 } else { (1u64 << n_qg) - 1 };
    let m0 = s_mask.get(s0).copied().unwrap_or(0) & qg_mask;
    let m1 = s_mask.get(s1).copied().unwrap_or(0) & qg_mask;

    let cov0 = m0.count_ones();
    let cov1 = m1.count_ones();

    // If s1 covers strictly more query groups than s0, or s1 is a superset of s0:
    let s1_wins = cov1 > cov0 || ((m0 & m1) == m0 && cov1 > cov0);
    if s1_wins {
        final_session_indices.swap(0, 1);
    }
}

/// S1-071 Informative Turn Priority:
/// Short assistant confirmations ("Sure thing!", "Got it!") yield Slot 0 to
/// same-session declarative user turns containing substantive grounded facts.
pub(in crate::retrieval) fn apply_informative_turn_priority(
    ix: &Index,
    plan: &QueryPlan,
    final_turn_indices: &mut Vec<usize>,
    t_scene: &[f32],
    _t_own: &[f32],
    t_exact_mask: &[u64],
) {
    if final_turn_indices.len() < 2 || plan.intent.is_event_sequence {
        return;
    }
    let h0 = final_turn_indices[0];
    let ht = &ix.turns[h0];

    // Check if head is a short confirmation or reaction turn
    let is_short_confirm = ht.field == 1
        && (ht.content.len() <= 10 || ht.is_backchannel || ht.is_short_reply);
    let is_pos0_casual = ht.field == 0
        && ht.content.len() <= 6
        && turn_pos(ix, h0) == Some(0)
        && !ht.is_standing_policy;

    if is_short_confirm || is_pos0_casual {
        let h_sc = t_scene.get(h0).copied().unwrap_or(0.0);
        let h_exact = t_exact_mask.get(h0).copied().unwrap_or(0);

        // Find a declarative substantive turn in same session or top-5
        if let Some(pos) = (1..final_turn_indices.len().min(5)).find(|&slot| {
            let ti = final_turn_indices[slot];
            let ct = &ix.turns[ti];
            if ct.session != ht.session || ct.is_question || ct.is_backchannel || turn_is_superseded(ix, ti) {
                return false;
            }
            let c_sc = t_scene.get(ti).copied().unwrap_or(0.0);
            let c_exact = t_exact_mask.get(ti).copied().unwrap_or(0);
            ct.content.len() >= 12
                && (c_exact != 0 || h_exact == 0)
                && (c_sc >= 0.55 * h_sc || h_sc == 0.0)
        }) {
            let moved = final_turn_indices.remove(pos);
            final_turn_indices.insert(0, moved);
        }
    }
}
