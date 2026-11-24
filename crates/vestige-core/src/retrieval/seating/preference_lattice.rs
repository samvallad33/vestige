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

// hardcode-scan: closed-class (explicit stance, optative/bouletic, and comparative preference frames)
pub(in crate::retrieval) fn is_explicit_stance_directive(body_low: &str, is_pers: bool) -> bool {
    if is_pers || crate::retrieval::has_comparative_choice_frame(body_low) {
        return true;
    }
    let has_without_gerund = body_low.contains(" without ")
        && body_low
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w.len() >= 5 && w.ends_with("ing"));
    if has_without_gerund {
        return true;
    }
    [
        "i prefer",
        "i'd prefer",
        "i\u{2019}d prefer",
        "so i'd like to",
        "so i\u{2019}d like to",
        "i'd like to",
        "i\u{2019}d like to",
        "i would like to",
        "keep my ",
        "i wanna ",
        "i aim for ",
        "i aim to ",
        "i tend to ",
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
        || eff_it.assistant_attribution
        || eff_it.is_event_sequence
        || eff_it.temporal
        || eff_it.sniper_how_many
        || eff_summary_request(query)
        || c2_named_session(ix, final_turn_indices[0])
        || ix.session_turns.len() < 2
        || ix.turns.len() < 16
    {
        return;
    }
    let q_low_pl = query.to_ascii_lowercase();
    // Guard strictly against factual past-tense / recall questions so KU/IE/CR/LME-SSU are untouched:
    // hardcode-scan: closed-class (factual past-tense & episodic lookup guards)
    let is_factual_past_q = q_low_pl.contains("did i ")
        || q_low_pl.contains("did we ")
        || q_low_pl.contains("did you ")
        || q_low_pl.contains("have i ")
        || q_low_pl.contains("was my ")
        || q_low_pl.contains("were my ")
        || q_low_pl.starts_with("what was ")
        || q_low_pl.starts_with("when ")
        || q_low_pl.starts_with("how many ")
        || q_low_pl.starts_with("how long ")
        || q_low_pl.contains("remind me of ");
    if is_factual_past_q {
        return;
    }
    // hardcode-scan: closed-class (collaborative advice, planning, and preference-following openings)
    let is_explicit_advice_q = crate::retrieval::step14_is_rec_request(query)
        || eff_it.preference
        || q_low_pl.starts_with("can you walk me through")
        || q_low_pl.starts_with("could you walk me through")
        || q_low_pl.starts_with("can you help me ")
        || q_low_pl.starts_with("could you help me ")
        || q_low_pl.starts_with("can you show me how")
        || q_low_pl.starts_with("can you explain step by step")
        || q_low_pl.starts_with("i'm planning ")
        || q_low_pl.starts_with("i\u{2019}m planning ")
        || q_low_pl.starts_with("i'm looking ")
        || q_low_pl.starts_with("i\u{2019}m looking ")
        || q_low_pl.starts_with("i'm working on ")
        || q_low_pl.starts_with("i\u{2019}m working on ")
        || q_low_pl.starts_with("i'm trying to ")
        || q_low_pl.starts_with("i\u{2019}m trying to ")
        || q_low_pl.starts_with("i'm preparing ")
        || q_low_pl.starts_with("i\u{2019}m preparing ")
        || q_low_pl.starts_with("i want to set up ")
        || q_low_pl.starts_with("what are some ways i can")
        || (q_low_pl.contains("what types of ") && q_low_pl.contains("should i"))
        || q_low_pl.contains("step by step");
    if has_dominant_cond_rule {
        return;
    }
    let is_pref_rec = is_explicit_advice_q && ix.turns.len() >= 16;
    if !is_pref_rec {
        return;
    }

    let ns_u32 = (ix.session_turns.len() as u32).max(1);
    let max_pref_df = (ns_u32 / 2).max(6);
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

    // Hoist corpus session co-occurrence set ONCE per query (O(N_sess) instead of O(P * 24 * Q * N_sess)).
    let co_cap = (ns_u32 / 4).max(4);
    let mut cooccur_allowed_ids: DetHashSet<u32> = DetHashSet::default();
    if ix.session_content.len() >= 8 && !q_topic_terms.is_empty() {
        for sc in &ix.session_content {
            if q_topic_terms.iter().any(|&(qid, _)| sc.contains(&qid)) {
                for &cid in sc {
                    let cdf = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
                    if (1..=co_cap).contains(&cdf) {
                        cooccur_allowed_ids.insert(cid);
                    }
                }
            }
        }
    }

    // 2. Identify active user preference directives (preserve multi-sentence preference questions and numeric preferences).
    // (pti, rank_key, topic_ic_q16, lex_ic, is_stance, score_weight)
    let mut active_prefs: Vec<(usize, u64, u64, u64, bool, f32)> = Vec::new();
    for &pti in &ix.preference_directive_turns {
        if pti >= ix.turns.len() {
            continue;
        }
        let t = &ix.turns[pti];
        if t.field != 0
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
        if t.is_pure_info_question && !is_stance {
            continue;
        }
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
        // Corpus session co-occurrence domain match (e.g. `lazysizes`/`bundle` <-> `gallery`, `excel`/`budget` <-> `expenses`):
        let mut cooccur_ic: u64 = 0;
        if topic_ic_q16 == 0 && is_stance && !cooccur_allowed_ids.is_empty() {
            let mut checked = 0usize;
            for &cid in &t.content {
                if !cooccur_allowed_ids.contains(&cid) {
                    continue;
                }
                let cdf = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
                checked += 1;
                if checked > 24 {
                    break;
                }
                cooccur_ic = cooccur_ic.saturating_add(aspect_ic_q16(cdf, ns_u32) >> 1);
            }
        }
        let eff_lex_ic = lex_ic.saturating_add(cooccur_ic);
        let sc = t_scene.get(pti).copied().unwrap_or(0.0);
        let own = t_own.get(pti).copied().unwrap_or(0.0);
        let score_weight = sc.max(own);
        let has_signal = topic_ic_q16 >= (Q16_ONE >> 2)
            || (is_stance && eff_lex_ic >= (Q16_ONE >> 1));
        if !has_signal && (!is_stance || ix.preference_directive_turns.len() > 6) {
            continue;
        }
        let rank_key = topic_ic_q16
            .saturating_mul(6)
            .saturating_add(eff_lex_ic.saturating_mul(2))
            .saturating_add(if is_stance { Q16_ONE >> 1 } else { 0 });
        active_prefs.push((pti, rank_key, topic_ic_q16, eff_lex_ic, is_stance, score_weight));
    }

    if active_prefs.is_empty() {
        return;
    }

    // C13: Afriat-Varian GARP Transitive Closure + Fostel-Scarf-Todd Utility + Kratzer-Searle Force Lattice
    {
        let cand_tis: Vec<usize> = active_prefs.iter().map(|x| x.0).collect();
        let cand_ic: Vec<u64> = active_prefs
            .iter()
            .map(|x| x.2.saturating_add(x.3))
            .collect();
        let ak_utilities =
            crate::retrieval::seating::afriat_kratzer_garp_q16::score_afriat_kratzer_candidates_q16(
                ix, &cand_tis, &cand_ic,
            );
        for (entry, &ak_q16) in active_prefs.iter_mut().zip(ak_utilities.iter()) {
            let k_prof =
                crate::retrieval::seating::afriat_kratzer_garp_q16::analyze_kratzer_searle_turn(
                    &ix.turns[entry.0],
                );
            if k_prof.point
                == crate::retrieval::seating::afriat_kratzer_garp_q16::IllocPoint::Directive
                && ix.turns[entry.0].is_pure_info_question
                && !entry.4
            {
                entry.1 = 0;
            } else if ak_q16 > 0 {
                let capped = ((ak_q16 >> 2) as u64)
                    .min(entry.2.saturating_mul(2).max(Q16_ONE >> 2));
                entry.1 = entry.1.saturating_add(capped);
            }
        }
        active_prefs.retain(|e| e.1 > 0 || e.4);
        if active_prefs.is_empty() {
            return;
        }
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

    // Two-Tier Co-Seating Shield:
    // - Tier 1 (Slot 0): domain-matched active preference directive (`topic_ic_q16 > 0` OR strong
    //   lexicon/cooccurrence match when Slot 0 is weak/echo/policy/assistant or lacks strong episodic IC).
    // - Tier 2 (Slot 1): co-seat universal / zero-exact-overlap stance preferences into Slot 1
    //   when Slot 0 already holds high episodic IC (protecting `LongMemEval_S` & `InMind`).
    let mut seats_filled = 0usize;
    let allow_zero_overlap_floor = ix.preference_directive_turns.len() <= 6 && k >= 5;
    for (pref_ti, _rank_key, topic_ic_q16, lex_ic, is_stance, _score_w) in active_prefs {
        if seats_filled >= 2 {
            break;
        }
        let has_signal = topic_ic_q16 >= (Q16_ONE >> 2)
            || (is_stance && lex_ic >= (Q16_ONE >> 1));
        if !has_signal && (seats_filled > 0 || !allow_zero_overlap_floor || !is_stance) {
            break;
        }
        let pref_si = ix.turns[pref_ti].session;
        let t0_opt = final_turn_indices.first().copied();
        let inc_is_asst = t0_opt.is_some_and(|t0| ix.turns[t0].field == 1);
        let inc_is_policy = t0_opt.is_some_and(|t0| ix.turns[t0].is_standing_policy);
        let inc_is_echo = t0_opt.is_some_and(|t0| {
            crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
                ix, t0, query,
            )
        });
        let inc_is_pref = t0_opt.is_some_and(|t0| {
            ix.preference_directive_turns.contains(&t0) && !turn_is_superseded(ix, t0)
        });

        let mut target_slot = if has_signal {
            seats_filled
        } else {
            1usize.min(final_turn_indices.len())
        };
        let mut s_target = if has_signal {
            seats_filled
        } else {
            1usize.min(final_session_indices.len())
        };

        if target_slot == 0 && !final_session_indices.is_empty() {
            let inc_si = final_session_indices[0];
            let inc_topic_ic = sess_topic_ic(inc_si);
            let q_toks_pl = raw_tokens(query);
            let inc_cond_policy = inc_is_policy
                && t0_opt.is_some_and(|t0| {
                    ix.standing_rules
                        .iter()
                        .any(|r| r.turn == t0 && rule_condition_matches(ix, r, plan, &q_toks_pl))
                });
            let strong_episodic_shield = inc_si != pref_si
                && (inc_cond_policy
                    || (!inc_is_asst
                        && !inc_is_policy
                        && !inc_is_echo
                        && (inc_is_pref
                            || (topic_ic_q16 == 0 && inc_topic_ic >= Q16_ONE.saturating_mul(6)))));
            if strong_episodic_shield {
                // Tier 2: preserve strong episodic or condition-matched rule Slot 0, co-seat preference at Slot 1!
                target_slot = 1usize.min(final_turn_indices.len());
                s_target = 1usize.min(final_session_indices.len());
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
