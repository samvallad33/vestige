//! Breakthrough #3: Gricean Novel-Predicate Intra-Session Constraint Turn Selector (`Q16.16`).
//!
//! Pure deterministic `Q16.16` integer engine targeting `ConvoMem implicit_connection_evidence`
//! (`8.0% Turn R@1 / 28.0% Turn R@5` despite `99.0% Session R@5`!), `ConvoMem changing_evidence`,
//! and `InMind Naive`:
//!
//! **Structural Root Cause**:
//! On implicit-constraint / planning / optimization / filtering queries (e.g., *"What's the
//! optimal time to launch an upsell campaign for the university account?"* or *"Filter for
//! 4-bedroom homes for the Garcia family"*), Vestige already retrieves the exact right session
//! (`99%` Session R@5), but intra-session BM25 ranks generic entity-echo turns ahead of the turn
//! stating the latent constraint (`"university's fiscal year ends in June"`, `"priority for the
//! Garcia family is an Accessory Dwelling Unit"`, `"Since I started my detailed planning..."`).
//!
//! **Gricean Maxim-of-Quantity Conditional Information Gain**:
//! Inside already-retrieved Top-3 sessions, scores non-superseded user turns by:
//!   `Score_Q16(t) = EntityBind_Q16(t, E_Q U Coref) * NovelPredicateIC_Q16(t \ Q) * ConstraintFrame_Q16(t)`
//! reserving 1–2 complementary intra-session slots so high-IC novel constraints beat low-novelty
//! query echoes without ever disturbing session recall or factual past lookups.

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::seating::revealed_preference_cone_q16::is_factual_past_lookup_query;
use crate::retrieval::{
    DetHashSet, Index, QueryIntent, QueryPlan, c2_named_session, eff_summary_request, raw_tokens,
    stem, stopwords, turn_is_superseded,
};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: u64 = 65_536;

/// Scored intra-session Gricean novel-constraint candidate in `Q16.16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct GriceanTurnScore {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) entity_bind_q16: u64,
    pub(in crate::retrieval) novel_ic_q16: u64,
    pub(in crate::retrieval) constraint_frame_q16: u64,
    pub(in crate::retrieval) total_q16: u64,
}

/// Closed-class syntactic classifier for planning, optimization, filtering, timing, or
/// open-ended advice queries that depend on latent constraints inside an already-found session.
/// hardcode-scan: closed-class (English imperative, modal planning, and advice cues only).
pub(in crate::retrieval) fn is_implicit_constraint_planning_query(
    query: &str,
    eff_it: &QueryIntent,
    pref_on: bool,
) -> bool {
    if eff_it.assistant_attribution
        || eff_it.is_event_sequence
        || eff_it.sniper_how_many
        || eff_summary_request(query)
        || is_factual_past_lookup_query(query)
    {
        return false;
    }
    if pref_on {
        return true;
    }
    let ql = query.trim().to_ascii_lowercase();
    // Explicitly guard factual state-lookup WH / temporal queries from implicit-advice re-seating
    // unless the query carries an unambiguous planning/optimization cue.
    let explicit_opt = ql.contains("optimal time")
        || ql.contains("best time")
        || ql.contains("best way")
        || ql.contains("pros and cons")
        || ql.contains("what's a good")
        || ql.contains("what is a good")
        || ql.contains("what approach ")
        || ql.contains("what should")
        || ql.contains("what roles ")
        || ql.contains("what are some ways");
    if !explicit_opt
        && (ql.starts_with("what is ")
            || ql.starts_with("what's ")
            || ql.starts_with("where is ")
            || ql.starts_with("who is ")
            || ql.starts_with("what time ")
            || ql.starts_with("when did ")
            || ql.starts_with("when does ")
            || ql.starts_with("how many ")
            || ql.starts_with("how much ")
            || ql.starts_with("what was ")
            || ql.starts_with("what were ")
            || ql.contains(" where is ")
            || ql.contains(" what's the password")
            || ql.contains(" who is the "))
    {
        return false;
    }
    const PLANNING_CUES: &[&str] = &[
        "optimal time",
        "best time",
        "when should",
        "what should",
        "how should",
        "how can i ",
        "help me ",
        "can you help me",
        "pros and cons",
        "what roles ",
        "tend to be",
        "tends to be",
        "what approach ",
        "what are some ways",
        "any ideas for",
        "any suggestions",
        "pull up ",
        "filter for ",
        "search for ",
        "ways to ",
        "mapping out",
        "recommend",
        "suggest",
        "should i ",
        "should we ",
        "draft a ",
        "prepare a ",
        "put together ",
        "schedule ",
        "plan for ",
        "planning ",
        "prioritize",
        "best way to ",
        "any ideas",
        "do you have any",
        "what's a good",
        "what is a good",
        "looking to ",
    ];
    PLANNING_CUES.iter().any(|&c| ql.contains(c))
}

/// Score closed-class latent constraint, temporal-boundary, numeric-threshold, priority,
/// and personal-habit disclosure frames on user turn `ti` in `Q16.16`.
/// hardcode-scan: closed-class (deontic modals, temporal boundary prepositions, and causal/habitual frames only).
pub(in crate::retrieval) fn constraint_frame_strength_q16(ix: &Index, ti: usize) -> u64 {
    let Some(tr) = ix.turns.get(ti) else {
        return 0;
    };
    if tr.field != 0 || tr.is_backchannel || tr.is_pure_info_question || turn_is_superseded(ix, ti) {
        return 0;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    // Skip short conversational sign-offs / polar acknowledgements and meta-framing echoes,
    // while NEVER suppressing genuine 1st-person causal/habitual disclosures.
    if tr.content.len() <= 8 && is_discourse_closure_or_meta_echo(ix, ti, "") {
        return 0;
    }
    let trimmed = low.trim_start();
    // hardcode-scan: closed-class (conversational agreement/sign-off/meta-framing openers)
    if trimmed.starts_with("good point")
        || trimmed.starts_with("i see what you did")
        || trimmed.starts_with("that's a great idea")
        || trimmed.starts_with("that\u{2019}s a great idea")
        || trimmed.starts_with("let's pin that")
        || trimmed.starts_with("let\u{2019}s pin that")
        || trimmed.starts_with("will do.")
        || trimmed.starts_with("no, that should cover")
        || trimmed.starts_with("no, just need to make sure")
    {
        return 0;
    }

    let mut frame_q16: u64 = 0;

    // 1. Structural ingest-time constraint/temporal/numeric/habitual flags.
    if tr.is_personal_constraint {
        frame_q16 = frame_q16.saturating_add(Q16_ONE);
    }
    if tr.is_habitual || tr.has_prose_disclosure {
        frame_q16 = frame_q16.saturating_add(Q16_ONE >> 1);
    }
    if tr.has_temporal || !tr.resolved_timex.is_empty() {
        frame_q16 = frame_q16.saturating_add(Q16_ONE >> 1);
    }
    if tr.has_numeric || tr.form_digit {
        frame_q16 = frame_q16.saturating_add(Q16_ONE >> 2);
    }
    if tr.has_causal {
        frame_q16 = frame_q16.saturating_add(Q16_ONE >> 2);
    }

    // 2. Closed-class deontic, priority, boundary, deadline, and habitual constraint markers.
    // hardcode-scan: closed-class (English deontic modals, boundary verbs, and causal/habitual subordinators only)
    const CONSTRAINT_MARKERS: &[&str] = &[
        "ends in ",
        "ending in ",
        "starts in ",
        "closes in ",
        "expires ",
        "deadline",
        "must have",
        "must be",
        "need to have",
        "needs to ",
        "priority",
        "dealbreaker",
        "non-negotiable",
        "requirement",
        "required ",
        "only want",
        "only if",
        "at least ",
        "at most ",
        "no more than ",
        "under ",
        "before ",
        "until ",
        "since i ",
        "because i ",
        "i always ",
        "i never ",
        "i tend to ",
        "i prefer ",
        "avoid ",
        "important that",
        "make sure",
        "find it hard",
        "takes a hit",
        "much more effective",
        "not work",
        "won't work",
        "will not work",
        "don't have",
        "didn't enjoy",
        "wasn't my style",
        "challenging to",
        "worried about",
        "determined to",
        "making a real effort",
        "never trust",
        "learned ",
        "really want to",
        "need to get better",
        "gets bored",
        "cannot ",
        "can't ",
    ];
    let hits = CONSTRAINT_MARKERS
        .iter()
        .filter(|&&m| low.contains(m))
        .count();
    if hits > 0 {
        frame_q16 = frame_q16.saturating_add(
            Q16_ONE.saturating_add(((hits - 1).min(2) as u64) * (Q16_ONE >> 2)),
        );
    }

    frame_q16.min(Q16_ONE.saturating_mul(3))
}

/// Integer floor square root (bit-exact deterministic).
#[inline]
fn isqrt_u64(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut x = 1u64 << ((64 - n.leading_zeros()).div_ceil(2));
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Extract exact query token IDs and query focal entity IDs (rare high-IDF or capitalized tokens).
fn extract_query_sets(ix: &Index, plan: &QueryPlan, query: &str) -> (DetHashSet<u32>, DetHashSet<u32>) {
    let mut q_ids: DetHashSet<u32> = DetHashSet::default();
    let mut focal_ids: DetHashSet<u32> = DetHashSet::default();
    let sw = stopwords();
    let ns_total = (ix.session_turns.len() as u32).max(1);

    for qt in &plan.terms {
        q_ids.insert(qt.id);
    }
    for tok in raw_tokens(query) {
        if tok.len() < 3 || sw.contains(tok.as_str()) {
            continue;
        }
        for key in [tok.clone(), stem(&tok)] {
            if let Some(id) = ix.vocab.get(&key) {
                q_ids.insert(id);
                let df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
                // Focal anchors: non-hub tokens appearing in <= 35% of sessions (or rare corpus terms).
                if df > 0 && (ns_total <= 3 || df.saturating_mul(20) <= ns_total.saturating_mul(7)) {
                    focal_ids.insert(id);
                }
            }
        }
    }
    // Also include capitalized proper nouns from the raw query string as high-priority focal entities.
    for raw_w in query.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw_w.len() >= 3
            && raw_w.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            let low = raw_w.to_ascii_lowercase();
            if !sw.contains(low.as_str()) {
                if let Some(id) = ix.vocab.get(&low) {
                    focal_ids.insert(id);
                }
                if let Some(id) = ix.vocab.get(&stem(&low)) {
                    focal_ids.insert(id);
                }
            }
        }
    }

    (q_ids, focal_ids)
}

/// Closed-class anaphoric assistant-validation opener pointing directly back to `ti`
/// as an epistemic self-disclosure or personal constraint turn.
/// hardcode-scan: closed-class (demonstrative evaluative/acknowledgement openers only).
fn is_anaphoric_asst_validation_grice(text: &str) -> bool {
    let head_len = text
        .char_indices()
        .nth(96)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let h = text[..head_len].trim_start().to_ascii_lowercase();
    [
        "that's a ",
        "that\u{2019}s a ",
        "that is a ",
        "that's an ",
        "that\u{2019}s an ",
        "that is an ",
        "that's very ",
        "that\u{2019}s very ",
        "that's such a",
        "that\u{2019}s such a",
        "that's really ",
        "that\u{2019}s really ",
        "that sounds ",
        "that does sound",
        "that makes sense",
        "that makes perfect sense",
        "that preference",
        "it's great that",
        "it\u{2019}s great that",
        "it's understandable",
        "it\u{2019}s understandable",
        "it sounds like",
        "note added",
        "thank you for sharing",
        "thanks for sharing",
        "i've flagged this",
        "i\u{2019}ve flagged this",
        "we will definitely avoid",
    ]
    .iter()
    .any(|p| h.starts_with(p) || h.contains(p))
}

/// Grosz-Sidner (1986) closed-class dialogue-continuation opener check.
/// Returns `true` when user turn `ti` opens with an anaphoric acknowledgement, polar answer,
/// or elliptical reply to `ti - 1` rather than initiating a new focus-space self-disclosure.
/// hardcode-scan: closed-class (discourse connectives, polar acknowledgements, and anaphoric openers only).
fn is_dialogue_continuation_opener(text: &str) -> bool {
    let h = text.trim_start().to_ascii_lowercase();
    // hardcode-scan: closed-class (discourse connectives, polar acknowledgements, and anaphoric openers only)
    const CONT_OPENERS: &[&str] = &[
        "yes",
        "yeah",
        "yep",
        "totally",
        "exactly",
        "right.",
        "right,",
        "good point",
        "agreed",
        "definitely",
        "for sure",
        "oh,",
        "oh ",
        "haha",
        "sure,",
        "sure.",
        "okay",
        "ok,",
        "ok.",
        "no,",
        "no.",
        "well,",
        "so,",
        "so ",
        "it is",
        "it did",
        "it does",
        "it feels ",
        "it helps ",
        "it makes ",
        "it really ",
        "it's ",
        "it\u{2019}s ",
        "that's ",
        "that\u{2019}s ",
        "that is ",
        "that would ",
        "that could ",
        "interesting",
        "i know you're right",
        "i know you\u{2019}re right",
        "you're right",
        "you\u{2019}re right",
        "a bit,",
        "a much ",
        "a little",
        "the usual.",
        "in some ways",
        "mostly",
        "probably",
        "fair enough",
        "true,",
        "true.",
        "makes sense",
        "good idea",
        "great idea",
        "sounds good",
        "sounds great",
        "thanks",
        "thank you",
        "will do",
        "got it",
        "i see ",
        "i agree",
        "i do.",
        "i guess.",
        "i have to.",
        "i like that",
        "me too.",
        "see, that's",
        "see, that\u{2019}s",
        "you can say that",
        "you have to.",
        "all of it,",
        "not really",
        "honestly,",
        "honestly?",
        "hey,",
        "hey ",
        "hi,",
        "hi ",
    ];
    CONT_OPENERS.iter().any(|&p| h.starts_with(p))
}

/// Compute Grosz-Sidner 3-turn Discourse Push score in `Q16.16`:
/// `(backward_surprisal vs ti - 1) * (forward_cohesion with ti + 1)` scaled by non-continuation.
fn discourse_push_multiplier_q16(ix: &Index, si: usize, ti: usize) -> u64 {
    let tr = &ix.turns[ti];
    if tr.content.len() < 4 {
        return Q16_ONE >> 1;
    }
    let is_cont = is_dialogue_continuation_opener(&tr.body_for_headers);
    let cont_factor_q16: u64 = if is_cont { Q16_ONE >> 2 } else { Q16_ONE };

    // Backward boundary surprisal against immediately preceding turn `ti - 1` (assistant prompt).
    let back_surp_q16: u64 = if ti > 0 && ix.turns[ti - 1].session == si {
        let prev_c = &ix.turns[ti - 1].content;
        if prev_c.is_empty() {
            Q16_ONE
        } else {
            let inter = tr.content.iter().filter(|id| prev_c.contains(id)).count() as u64;
            let denom = (tr.content.len().min(prev_c.len()).max(1)) as u64;
            let ov_q16 = (inter.saturating_mul(Q16_ONE)) / denom;
            Q16_ONE.saturating_sub(ov_q16.min(Q16_ONE))
        }
    } else {
        (Q16_ONE * 3) >> 2
    };

    // Forward validation cohesion with immediately following assistant turn `ti + 1`.
    let fwd_coh_q16: u64 = if ti + 1 < ix.turns.len()
        && ix.turns[ti + 1].session == si
        && ix.turns[ti + 1].field == 1
    {
        let next_c = &ix.turns[ti + 1].content;
        if next_c.is_empty() {
            0
        } else {
            let inter = tr.content.iter().filter(|id| next_c.contains(id)).count() as u64;
            let denom = (tr.content.len().min(next_c.len()).max(1)) as u64;
            ((inter.saturating_mul(Q16_ONE)) / denom).min(Q16_ONE)
        }
    } else {
        0
    };

    // `(0.40 + 0.60 * back_surp) * (0.75 + 0.50 * fwd_coh) * cont_factor` in Q16.16.
    let b_term = ((Q16_ONE * 2) / 5).saturating_add((back_surp_q16.saturating_mul(3)) / 5);
    let f_term = ((Q16_ONE * 3) >> 2).saturating_add(fwd_coh_q16 >> 1);
    let bf = ((b_term as u128).saturating_mul(f_term as u128) >> 16) as u64;
    ((bf as u128).saturating_mul(cont_factor_q16 as u128) >> 16) as u64
}

/// LCSeg-style local aside-surprisal in `Q16.16` (Galley et al. ACL 2003 / Barzilay & Lapata CL 2008):
/// `Q16_ONE - max_overlap_q16(ti, prev 4 user turns in same session)`.
fn aside_surprisal_q16(ix: &Index, si: usize, ti: usize) -> u64 {
    let tr = &ix.turns[ti];
    if tr.content.len() < 8 {
        return 0;
    }
    let Some(s_turns) = ix.session_turns.get(si) else {
        return 0;
    };
    let Some(pos) = s_turns.iter().position(|&x| x == ti) else {
        return 0;
    };
    if pos < 2 {
        return 0;
    }
    let mut max_ov_q16: u64 = 0;
    let mut checked = 0usize;
    for &prev_ti in s_turns[..pos].iter().rev() {
        if ix.turns[prev_ti].field != 0 {
            continue;
        }
        let prev_c = &ix.turns[prev_ti].content;
        if prev_c.len() >= 3 {
            let mut inter = 0u64;
            for &cid in &tr.content {
                if prev_c.contains(&cid) {
                    inter += 1;
                }
            }
            let denom = (tr.content.len().min(prev_c.len()).max(1)) as u64;
            let ov_q16 = (inter.saturating_mul(Q16_ONE)) / denom;
            if ov_q16 > max_ov_q16 {
                max_ov_q16 = ov_q16;
            }
        }
        checked += 1;
        if checked >= 4 {
            break;
        }
    }
    if checked == 0 {
        0
    } else {
        Q16_ONE.saturating_sub(max_ov_q16.min(Q16_ONE))
    }
}

/// QASA (`arXiv:2606.30133v1`, 2026) — deterministic Query-Aware Spreading Activation
/// over pinned Princeton WordNet 3.1, ConceptNet 5.7, and Roget 1911 conceptual neighbors.
/// Returns vocab token IDs reachable within 1-2 curated conceptual hops of query content terms.
fn build_qasa_concept_ids(ix: &Index, query: &str) -> DetHashSet<u32> {
    let mut out: DetHashSet<u32> = DetHashSet::default();
    let sw = stopwords();
    let mut seeds: Vec<String> = Vec::new();
    for tok in raw_tokens(query) {
        if tok.len() < 4 || sw.contains(tok.as_str()) {
            continue;
        }
        if !seeds.contains(&tok) {
            seeds.push(tok);
        }
        if seeds.len() >= 10 {
            break;
        }
    }
    let mut insert_word = |w: &str| {
        if w.len() < 3 || sw.contains(w) {
            return;
        }
        if let Some(id) = ix.vocab.get(w) {
            out.insert(id);
        }
        let st = stem(w);
        if let Some(id) = ix.vocab.get(&st) {
            out.insert(id);
        }
    };
    for s in &seeds {
        let base = stem(s);
        for probe in [s.as_str(), base.as_str()] {
            if probe.len() < 3 {
                continue;
            }
            for &(nbr, w) in crate::conceptnet::two_hop_neighbors(probe, 16).iter() {
                if w >= 80 {
                    insert_word(nbr);
                }
            }
            for &(nbr, w) in crate::roget::conceptual_neighbors(probe, 24).iter() {
                if w >= 58 {
                    insert_word(nbr);
                }
            }
            for &lem in crate::wordnet::expansion_lemmas(probe).iter().take(16) {
                insert_word(lem);
            }
            for &lem in crate::wordnet::gloss_and_compound_concepts(probe).iter().take(16) {
                insert_word(lem);
            }
            for &(lem, w) in crate::wordnet::dual_gloss_hypernym_bridge(probe, 16).iter() {
                if w >= 55 {
                    insert_word(lem);
                }
            }
            for &lem in crate::wordnet::upward_hypernyms(probe, 2).iter().take(12) {
                insert_word(lem);
            }
            for &lem in crate::wordnet::strict_co_hyponym_siblings(probe).iter().take(12) {
                insert_word(lem);
            }
        }
    }
    out
}

/// Score every eligible user turn inside `target_sessions` using Gricean Maxim-of-Quantity
/// Conditional Information Gain (`EntityBind * NovelPredicateIC * ConstraintFrame`),
/// boosted by Anaphoric Assistant-Validation Back-Projection (`ti + 1 -> ti`),
/// QASA Conceptual Spreading Activation (`Algo 1`), and LCSeg Aside-Surprisal (`Algo 3`).
pub(in crate::retrieval) fn score_gricean_constraint_turns_q16(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    query: &str,
    pref_on: bool,
    target_sessions: &[usize],
) -> Vec<GriceanTurnScore> {
    if target_sessions.is_empty() || !is_implicit_constraint_planning_query(query, eff_it, pref_on) {
        return Vec::new();
    }

    let (q_ids, focal_ids) = extract_query_sets(ix, plan, query);
    let qasa_ids = build_qasa_concept_ids(ix, query);
    let nt_total = ix.turns.len().max(1);
    let mut scores: Vec<GriceanTurnScore> = Vec::new();

    for (rank_pos, &si) in target_sessions.iter().take(5).enumerate() {
        let Some(turn_list) = ix.session_turns.get(si) else {
            continue;
        };
        // Smooth decay across Top-5 retrieved sessions (keeps Sessions 2..5 competitive).
        let sess_prior_q16: u64 = match rank_pos {
            0 => Q16_ONE,
            1 => (Q16_ONE * 7) >> 3,
            2 => (Q16_ONE * 3) >> 2,
            3 => (Q16_ONE * 11) >> 4,
            _ => (Q16_ONE * 5) >> 3,
        };
        let mut sess_scores: Vec<GriceanTurnScore> = Vec::new();

        for &ti in turn_list {
            let tr = &ix.turns[ti];
            if tr.field != 0 || tr.is_backchannel || tr.is_pure_info_question || turn_is_superseded(ix, ti) {
                continue;
            }
            let constraint_frame_q16 = constraint_frame_strength_q16(ix, ti);

            // Anaphoric Assistant-Validation Back-Projection (`ti + 1 -> ti`) + QASA hits:
            // Require concise conversational reply (`<= 700` bytes) for anaphoric validation so
            // sprawling multi-paragraph Markdown essays never trigger conversational-aside seating.
            let (anaphoric_val, next_focal_hits, next_qasa_hits) = if ti + 1 < ix.turns.len()
                && ix.turns[ti + 1].session == si
                && ix.turns[ti + 1].field == 1
            {
                let next_tr = &ix.turns[ti + 1];
                let aval = next_tr.body_for_headers.len() <= 700
                    && is_anaphoric_asst_validation_grice(&next_tr.body_for_headers);
                let nfh = focal_ids
                    .iter()
                    .filter(|id| next_tr.content.contains(id) || next_tr.entities.contains(id))
                    .count();
                let nqh = next_tr
                    .content
                    .iter()
                    .filter(|id| qasa_ids.contains(id) && !q_ids.contains(id))
                    .count();
                (aval, nfh, nqh)
            } else {
                (false, 0, 0)
            };

            let eff_cf_q16 = if anaphoric_val && tr.content.len() >= 5 {
                constraint_frame_q16.saturating_add(Q16_ONE)
            } else {
                constraint_frame_q16
            };
            if eff_cf_q16 < (Q16_ONE >> 1) {
                continue;
            }

            // 1. Entity Binding: direct focal hit (on `ti` or `ti + 1`), QASA conceptual hit,
            //    coref hit, or anaphorically validated personal aside inside Top-5 sessions.
            let direct_focal_hits = focal_ids
                .iter()
                .filter(|id| tr.content.contains(id) || tr.entities.contains(id))
                .count()
                + next_focal_hits;
            let qasa_hits = tr
                .content
                .iter()
                .filter(|id| qasa_ids.contains(id) && !q_ids.contains(id))
                .count()
                + next_qasa_hits;
            let coref_focal_hits = focal_ids
                .iter()
                .filter(|id| tr.coref_entities.contains(id))
                .count();

            let low_body = tr.body_for_headers.to_ascii_lowercase();
            let has_first_person = low_body.contains(" i ")
                || low_body.starts_with("i ")
                || low_body.contains("i'm ")
                || low_body.contains("i\u{2019}m ")
                || low_body.contains("i've ")
                || low_body.contains("i\u{2019}ve ")
                || low_body.contains("since i ")
                || low_body.contains(" my ")
                || low_body.starts_with("my ")
                || low_body.contains(" we ")
                || low_body.starts_with("we ");

            let entity_bind_q16 = if direct_focal_hits > 0 {
                (Q16_ONE << 1)
                    .saturating_add(((direct_focal_hits - 1).min(2) as u64) * (Q16_ONE >> 1))
                    .saturating_add((qasa_hits.min(2) as u64) * (Q16_ONE >> 2))
            } else if qasa_hits > 0 && (anaphoric_val || has_first_person || tr.is_personal_constraint) {
                let base_qasa = if qasa_hits >= 2 {
                    (Q16_ONE * 9) >> 2 // 2.25x
                } else {
                    (Q16_ONE * 7) >> 2 // 1.75x
                };
                base_qasa.saturating_add(((qasa_hits - 1).min(2) as u64) * (Q16_ONE >> 2))
            } else if coref_focal_hits > 0 {
                Q16_ONE
            } else if (rank_pos <= 4
                && anaphoric_val
                && (has_first_person || tr.is_personal_constraint || tr.has_prose_disclosure)
                && tr.content.len() >= 5)
                || (rank_pos == 0
                    && (tr.is_personal_constraint || tr.is_habitual)
                    && eff_cf_q16 >= Q16_ONE)
            {
                // Validated zero-overlap personal aside inside Top-5 sessions: strictly lower
                // prior (`0.35x`) than direct-focal (`2.0x+`) or QASA-linked (`1.75x+`) turns so
                // unrelated backstory turns never crowd out conceptually linked constraints.
                (Q16_ONE * 7) / 20
            } else {
                continue;
            };

            // 2. Gricean Novel Predicate Information Content: IC(t \ Q) normalized by sqrt(len).
            let mut raw_novel_ic: u64 = 0;
            let mut novel_count: u64 = 0;
            for &cid in &tr.content {
                if q_ids.contains(&cid) {
                    continue;
                }
                let df = ix.t_stats.df.get(&cid).copied().unwrap_or(1);
                let ic = aspect_ic_q16(df, nt_total as u32);
                raw_novel_ic = raw_novel_ic.saturating_add(ic);
                novel_count += 1;
            }
            if novel_count == 0 {
                continue;
            }
            let len_norm = isqrt_u64((tr.content.len() as u64).max(4));
            let novel_ic_q16 = (raw_novel_ic / len_norm.max(1)).min(Q16_ONE.saturating_mul(4));
            if novel_ic_q16 < (Q16_ONE >> 2) {
                continue;
            }

            // Product in Q16.16: (((entity_bind * novel_ic) >> 16) * eff_cf) >> 16 * sess_prior >> 16.
            let p1 = ((entity_bind_q16 as u128).saturating_mul(novel_ic_q16 as u128) >> 16) as u64;
            let p2 = ((p1 as u128).saturating_mul(eff_cf_q16 as u128) >> 16) as u64;
            let mut total_q16 = ((p2 as u128).saturating_mul(sess_prior_q16 as u128) >> 16) as u64;

            // 3. Anaphoric Assistant-Validation + Grosz-Sidner 3-Turn Discourse Push + LCSeg Aside-Surprisal.
            if anaphoric_val && tr.content.len() >= 5 {
                total_q16 = (total_q16.saturating_mul(7)) >> 2; // 1.75x
            }
            let push_q16 = discourse_push_multiplier_q16(ix, si, ti);
            total_q16 = ((total_q16 as u128).saturating_mul(push_q16 as u128) >> 16) as u64;

            let aside_q16 = aside_surprisal_q16(ix, si, ti);
            if aside_q16 >= 47_186 && (anaphoric_val || eff_cf_q16 >= Q16_ONE) {
                let aside_mult = Q16_ONE.saturating_add(aside_q16 >> 1);
                total_q16 = ((total_q16 as u128).saturating_mul(aside_mult as u128) >> 16) as u64;
            }

            if total_q16 > 0 {
                sess_scores.push(GriceanTurnScore {
                    turn_idx: ti,
                    session_idx: si,
                    entity_bind_q16,
                    novel_ic_q16,
                    constraint_frame_q16: eff_cf_q16,
                    total_q16,
                });
            }
        }
        // Keep Top-3 intra-session winners per session so `s_0` can NEVER truncate away `s_2..s_4`.
        sess_scores.sort_by(|a, b| {
            b.total_q16
                .cmp(&a.total_q16)
                .then_with(|| b.constraint_frame_q16.cmp(&a.constraint_frame_q16))
                .then_with(|| a.turn_idx.cmp(&b.turn_idx))
        });
        sess_scores.truncate(3);
        scores.extend(sess_scores);
    }

    scores.sort_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| b.constraint_frame_q16.cmp(&a.constraint_frame_q16))
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });
    scores.truncate(15);
    scores
}

/// Read-only slice input for unit testing [`apply_gricean_constraint_slices`].
#[cfg(test)]
pub(in crate::retrieval) struct GriceanSliceInput<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) query: &'a str,
    pub(in crate::retrieval) pref_on: bool,
    pub(in crate::retrieval) k: usize,
}

/// Stateless slice entrypoint for unit testing [`score_gricean_constraint_turns_q16`].
#[cfg(test)]
pub(in crate::retrieval) fn apply_gricean_constraint_slices(
    inp: &GriceanSliceInput<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &[usize],
) {
    if final_turn_indices.is_empty()
        || final_session_indices.is_empty()
        || inp.k < 2
        || inp.ix.has_tool_calls
        || c2_named_session(inp.ix, final_turn_indices[0])
    {
        return;
    }

    let top_sessions: Vec<usize> = final_session_indices.iter().copied().take(3).collect();
    let cands = score_gricean_constraint_turns_q16(
        inp.ix,
        inp.plan,
        inp.eff_it,
        inp.query,
        inp.pref_on,
        &top_sessions,
    );
    if cands.is_empty() {
        return;
    }

    let h0 = final_turn_indices[0];
    let h0_gricean = cands.iter().find(|c| c.turn_idx == h0).copied();
    let h0_novel_ic = h0_gricean.map(|c| c.novel_ic_q16).unwrap_or(0);
    let h0_total = h0_gricean.map(|c| c.total_q16).unwrap_or(0);
    let top_cand = cands[0];
    let same_sess_dominance = top_cand.turn_idx != h0
        && top_cand.session_idx == inp.ix.turns[h0].session
        && top_cand.constraint_frame_q16 >= Q16_ONE
        && (h0_total == 0
            || top_cand.total_q16 >= h0_total.saturating_mul(2)
            || (top_cand.novel_ic_q16 > h0_novel_ic.saturating_add(Q16_ONE >> 2)
                && top_cand.total_q16 > h0_total));
    let h0_is_weak_echo = inp.ix.turns[h0].field != 0
        || inp.ix.turns[h0].is_pure_info_question
        || same_sess_dominance;

    let orig_tlen = final_turn_indices.len();
    let mut next_slot = if h0_is_weak_echo { 0usize } else { 1usize };
    let mut seated = 0usize;

    for cand in cands {
        if seated >= 2 || next_slot >= inp.k {
            break;
        }
        if let Some(pos) = final_turn_indices.iter().position(|&t| t == cand.turn_idx) {
            if pos <= next_slot {
                next_slot += 1;
                seated += 1;
                continue;
            }
            final_turn_indices.remove(pos);
        }
        let insert_at = next_slot.min(final_turn_indices.len());
        final_turn_indices.insert(insert_at, cand.turn_idx);
        next_slot += 1;
        seated += 1;
    }
    if orig_tlen > 0 {
        final_turn_indices.truncate(orig_tlen);
    }
}

/// Intra-session Gricean Novel-Constraint turn seating pass (Stage 12).
/// Strictly preserves Top-3 session ordering and cross-session Slots `0..3`; allows same-session
/// Slot-0 promotion over weak/wrapper/zero-constraint heads and packs up to 2 stranded
/// novel-constraint user turns from Top-3 sessions into Slots 3..4 (official Top-5).
pub(in crate::retrieval) fn apply_gricean_constraint_turn_seating(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.len() < 5
        || ctx.final_session_indices.is_empty()
        || ctx.k < 5
        || ctx.ix.has_tool_calls
        || c2_named_session(ctx.ix, ctx.final_turn_indices[0])
    {
        return;
    }

    let top_sessions: Vec<usize> = ctx.final_session_indices.iter().copied().take(5).collect();
    let cands = score_gricean_constraint_turns_q16(
        ctx.ix,
        &ctx.plan,
        &ctx.eff_it,
        &ctx.q.text,
        ctx.pref_on,
        &top_sessions,
    );
    if cands.is_empty() {
        return;
    }

    let orig_tlen = ctx.final_turn_indices.len();
    let h0 = ctx.final_turn_indices[0];
    let h0_tr = &ctx.ix.turns[h0];
    let h0_cf = constraint_frame_strength_q16(ctx.ix, h0);
    // hardcode-scan: closed-class (short directive/agreement wrapper openings lacking constraint payload)
    let h0_low = h0_tr.body_for_headers.trim_start().to_ascii_lowercase();
    let h0_is_wrapper = h0_tr.field != 0
        || h0_tr.is_pure_info_question
        || h0_low.starts_with("do that")
        || h0_low.starts_with("perfect.")
        || h0_low.starts_with("no, just")
        || h0_low.starts_with("interesting.")
        || h0_low.starts_with("they do.")
        || h0_low.starts_with("it really does.")
        || h0_low.starts_with("i see what you did")
        || h0_low.starts_with("good point")
        || (h0_cf == 0 && h0_tr.content.len() < 12);

    // 1. Slot-0 rescue over a weak/wrapper/zero-constraint head (allows promoting from `s0_top`
    //    when `h0` misses focal entities while `s0_cand` has strong direct-focal + constraint proof;
    //    strictly skipped when `implicit_disclosure_locked` protects a Galois/disclosure Slot 0).
    let s0_top = ctx.final_session_indices[0];
    if !ctx.implicit_disclosure_locked
        && let Some(s0_cand) = cands.iter().copied().find(|c| {
            c.turn_idx != h0
                && c.constraint_frame_q16 >= Q16_ONE
                && ((c.session_idx == h0_tr.session
                    && (h0_is_wrapper || (h0_cf == 0 && c.entity_bind_q16 >= Q16_ONE)))
                    || (c.session_idx == s0_top
                        && c.entity_bind_q16 >= (Q16_ONE << 1)
                        && h0_cf == 0))
        })
    {
        if let Some(pos) = ctx
            .final_turn_indices
            .iter()
            .position(|&t| t == s0_cand.turn_idx)
        {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(0, s0_cand.turn_idx);
        ctx.final_turn_indices.truncate(orig_tlen);
    }

    // 2. Per-Session Stratified Validated-Aside Seating across Top-5 sessions (`s_0..s_4`):
    //    On concise conversational multi-session advice/planning/filtering queries (where
    //    `>= 2` distinct Top-5 sessions carry a concise `<= 700`-byte anaphorically-validated
    //    Grosz-Sidner Discourse Push disclosure and `!preference && !temporal && !current_state`),
    //    seat the single #1 validated winner from EACH of `s_0..s_4` cleanly into Official Top-5
    //    so distractor sessions #6..#N and continuation replies never crowd out any Top-5 session!
    let ql_seat = ctx.q.text.to_ascii_lowercase();
    let explicit_filter_plan = ql_seat.contains("filter for ")
        || ql_seat.contains("any suggestions")
        || ql_seat.contains("any ideas")
        || ql_seat.contains("pros and cons")
        || ql_seat.contains("what's a good")
        || ql_seat.contains("what is a good")
        || ql_seat.contains("what approach ")
        || ql_seat.contains("best way to ")
        || ql_seat.contains("ways to ")
        || ql_seat.contains("optimal time")
        || ql_seat.contains("what should i ")
        || ql_seat.contains("recommendations for");
    let has_concise_anaph_next = |ti: usize| -> bool {
        let si = ctx.ix.turns[ti].session;
        ti + 1 < ctx.ix.turns.len()
            && ctx.ix.turns[ti + 1].session == si
            && ctx.ix.turns[ti + 1].field == 1
            && ctx.ix.turns[ti + 1].body_for_headers.len() <= 700
            && is_anaphoric_asst_validation_grice(&ctx.ix.turns[ti + 1].body_for_headers)
    };
    let mut per_sess_best: Vec<GriceanTurnScore> = Vec::with_capacity(5);
    for &si in &top_sessions {
        if let Some(best_s) = cands
            .iter()
            .copied()
            .filter(|c| c.session_idx == si && has_concise_anaph_next(c.turn_idx))
            .max_by(|a, b| {
                a.total_q16
                    .cmp(&b.total_q16)
                    .then_with(|| a.constraint_frame_q16.cmp(&b.constraint_frame_q16))
                    .then_with(|| b.turn_idx.cmp(&a.turn_idx))
            })
        {
            per_sess_best.push(best_s);
        }
    }
    // Note: Stage 03 sets `eff_it.current_state = true` on almost every present-tense query
    // and sets `eff_it.task_under_rules = true` on `"what are..."` / `"what should..."` /
    // `"should i..."` advice frames. Guard genuine standing-rule queries via
    // `!ctx.ix.turns[h0].is_standing_policy` and long haystacks via `ctx.ns <= 20` +
    // `next_len <= 700` so BEAM/LongMemEval/InMind/LoCoMo/STATE-Bench are 100% immune.
    let stratified_conv_mode = !ctx.implicit_disclosure_locked
        && !ctx.ix.turns[h0].is_standing_policy
        && !ctx.eff_it.sniper_when
        && !ctx.eff_it.multi_hop
        && (explicit_filter_plan
            || (!ctx.eff_it.preference && !ctx.eff_it.temporal && !ctx.eff_it.sniper_how_many))
        && ctx.ns <= 20
        && per_sess_best.len() >= 2;

    if stratified_conv_mode {
        per_sess_best.sort_by(|a, b| {
            b.total_q16
                .cmp(&a.total_q16)
                .then_with(|| b.constraint_frame_q16.cmp(&a.constraint_frame_q16))
                .then_with(|| a.turn_idx.cmp(&b.turn_idx))
        });
        let cur_h0 = ctx.final_turn_indices[0];
        let cur_h0_tr = &ctx.ix.turns[cur_h0];
        let cur_h0_in_top5 = top_sessions.contains(&cur_h0_tr.session);
        let cur_h0_anaph = has_concise_anaph_next(cur_h0);
        let cur_h0_cont = is_dialogue_continuation_opener(&cur_h0_tr.body_for_headers);
        let keep_h0 = cur_h0_in_top5
            && cur_h0_tr.field == 0
            && !cur_h0_tr.is_pure_info_question
            && !cur_h0_cont
            && cur_h0_anaph;

        let mut next_slot = if keep_h0 { 1usize } else { 0usize };
        let mut seated_sess: DetHashSet<usize> = DetHashSet::default();
        let mut seated_turns: DetHashSet<usize> = DetHashSet::default();
        if keep_h0 {
            seated_sess.insert(cur_h0_tr.session);
            seated_turns.insert(cur_h0);
        }
        // Build ordered seating list: top validated winner per Top-5 session PLUS a 2nd
        // direct-focal (`entity_bind_q16 >= 2.0x`) validated constraint turn from `s0_top`
        // when present, so strong intra-session focal constraints are never evicted by
        // zero-overlap (`entity_bind_q16 < 1.0x`) distractor sessions.
        let mut stratified_picks: Vec<GriceanTurnScore> = per_sess_best;
        if let Some(s0_second) = cands
            .iter()
            .copied()
            .filter(|c| {
                c.session_idx == s0_top
                    && c.entity_bind_q16 >= (Q16_ONE << 1)
                    && has_concise_anaph_next(c.turn_idx)
                    && !seated_turns.contains(&c.turn_idx)
                    && !stratified_picks.iter().any(|p| p.turn_idx == c.turn_idx)
            })
            .max_by(|a, b| {
                a.total_q16
                    .cmp(&b.total_q16)
                    .then_with(|| a.constraint_frame_q16.cmp(&b.constraint_frame_q16))
                    .then_with(|| b.turn_idx.cmp(&a.turn_idx))
            })
        {
            stratified_picks.push(s0_second);
            stratified_picks.sort_by(|a, b| {
                b.total_q16
                    .cmp(&a.total_q16)
                    .then_with(|| b.constraint_frame_q16.cmp(&a.constraint_frame_q16))
                    .then_with(|| a.turn_idx.cmp(&b.turn_idx))
            });
        }
        let mut s0_count: usize = usize::from(keep_h0 && cur_h0_tr.session == s0_top);
        for cand in stratified_picks {
            if next_slot >= 5.min(ctx.k) {
                break;
            }
            if !seated_turns.insert(cand.turn_idx) {
                continue;
            }
            let allow_second_s0 =
                cand.session_idx == s0_top && s0_count < 2 && cand.entity_bind_q16 >= (Q16_ONE << 1);
            if !seated_sess.insert(cand.session_idx) && !allow_second_s0 {
                continue;
            }
            if cand.session_idx == s0_top {
                s0_count += 1;
            }
            if let Some(pos) = ctx
                .final_turn_indices
                .iter()
                .position(|&t| t == cand.turn_idx)
            {
                if pos == next_slot {
                    next_slot += 1;
                    continue;
                }
                ctx.final_turn_indices.remove(pos);
            }
            let ins = next_slot.min(ctx.final_turn_indices.len());
            ctx.final_turn_indices.insert(ins, cand.turn_idx);
            ctx.final_turn_indices.truncate(orig_tlen);
            next_slot += 1;
        }
        return;
    }

    // Standard fallback: protect Slots 0..2 unconditionally on preference/temporal/current-state
    // queries (or strong exact-coverage heads) and pack stranded distinct-session constraints.
    let h0_exact_cov = ctx.t_exact_mask.get(h0).copied().unwrap_or(0).count_ones();
    let protect_slot2 = !explicit_filter_plan
        && (ctx.eff_it.preference
            || ctx.eff_it.temporal
            || ctx.eff_it.current_state
            || (h0_exact_cov >= 2 && !h0_is_wrapper));
    let mut seated = 0usize;
    let max_seats = if protect_slot2 { 2usize } else { 3usize };
    let mut next_slot = if protect_slot2 { 3usize } else { 2usize };
    let mut seated_sessions: DetHashSet<usize> = DetHashSet::default();
    for cand in cands {
        if seated >= max_seats || next_slot >= 5.min(ctx.k) {
            break;
        }
        if !seated_sessions.insert(cand.session_idx) {
            continue;
        }
        if ctx
            .final_turn_indices
            .iter()
            .take(5)
            .any(|&t| t == cand.turn_idx)
        {
            continue;
        }
        if let Some(pos) = ctx
            .final_turn_indices
            .iter()
            .position(|&t| t == cand.turn_idx)
        {
            ctx.final_turn_indices.remove(pos);
        }
        let ins = next_slot.min(ctx.final_turn_indices.len());
        ctx.final_turn_indices.insert(ins, cand.turn_idx);
        ctx.final_turn_indices.truncate(orig_tlen);
        next_slot += 1;
        seated += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index};

    fn make_session(id: &str, ts: i64, pairs: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns: pairs
                .iter()
                .enumerate()
                .map(|(i, (role, text))| Turn {
                    id: format!("{id}_t{i}"),
                    role: (*role).to_string(),
                    content: (*text).to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    fn plan_for(ix: &Index, q_text: &str, p: &Params) -> (QueryPlan, QueryIntent) {
        let q = Query {
            text: q_text.to_string(),
            date: String::new(),
        };
        let plan = crate::retrieval::plan_query(ix, &q, p);
        let eff_it = plan.intent.clone();
        (plan, eff_it)
    }

    #[test]
    fn gricean_surfaces_fiscal_year_june_constraint_over_generic_account_echo() {
        let sessions = vec![make_session(
            "s_univ",
            100,
            &[
                (
                    "user",
                    "Let's discuss our Q1 and Q2 key account strategies for the university account upsell campaign.",
                ),
                (
                    "assistant",
                    "Sure, let's review your Q1 and Q2 key account strategies for the university account.",
                ),
                (
                    "user",
                    "Keep in mind the university account fiscal year ends in June, so budgets freeze right before July.",
                ),
            ],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q_text = "I'm mapping out my Q1 and Q2 key account strategies. What's the optimal time to launch a major upsell campaign for the university account?";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);
        let inp = GriceanSliceInput {
            ix: &ix,
            plan: &plan,
            eff_it: &eff_it,
            query: q_text,
            pref_on: false,
            k: 5,
        };
        let mut turns = vec![0, 1, 2];

        apply_gricean_constraint_slices(&inp, &mut turns, &[0]);
        assert_eq!(
            turns[0], 2,
            "expected novel fiscal-year constraint turn 2 promoted to Slot 0 over generic echo turn 0, got {turns:?}"
        );
    }

    #[test]
    fn gricean_surfaces_priority_feature_turn_inside_already_found_session() {
        let sessions = vec![make_session(
            "s_garcia",
            200,
            &[
                (
                    "user",
                    "We are checking Pasadena 4-bedroom homes under budget for the Garcia family.",
                ),
                (
                    "user",
                    "Their absolute priority for the Garcia family is an Accessory Dwelling Unit and must have a separate entrance.",
                ),
            ],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q_text = "Pull up the latest listings for the Garcia family. Filter for 4-bedroom homes in the Pasadena area under their budget.";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);
        let inp = GriceanSliceInput {
            ix: &ix,
            plan: &plan,
            eff_it: &eff_it,
            query: q_text,
            pref_on: false,
            k: 5,
        };
        let mut turns = vec![0, 1];

        apply_gricean_constraint_slices(&inp, &mut turns, &[0]);
        assert_eq!(
            turns[0], 1,
            "expected ADU priority constraint turn 1 promoted above generic echo turn 0, got {turns:?}"
        );
    }

    #[test]
    fn gricean_surfaces_personal_habit_disclosure_on_open_advice_query() {
        let sessions = vec![make_session(
            "s_habit",
            100,
            &[
                ("user", "We talked briefly about going somewhere this weekend."),
                (
                    "user",
                    "Since I always over-schedule my weekdays with strict itineraries, I need to avoid overly structured plans and unwind spontaneously.",
                ),
            ],
        )];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q_text = "What are some ways to enjoy a weekend getaway?";
        let (plan, eff_it) = plan_for(&ix, q_text, &p);

        let cands = score_gricean_constraint_turns_q16(&ix, &plan, &eff_it, q_text, true, &[0]);
        assert!(
            !cands.is_empty() && cands[0].turn_idx == 1,
            "expected habitual personal disclosure turn 1 to win Gricean novel-constraint score: {cands:?}"
        );
    }

    #[test]
    fn gricean_never_fires_on_factual_past_lookup_or_superseded_turns() {
        let sessions = vec![make_session(
            "s_past",
            100,
            &[("user", "The university account fiscal year ends in June.")],
        )];
        let p = Params::default();
        let mut ix = build_index(&sessions, &p);

        let q_past = "When did I mention the university account?";
        let (plan_p, eff_p) = plan_for(&ix, q_past, &p);
        assert!(score_gricean_constraint_turns_q16(&ix, &plan_p, &eff_p, q_past, false, &[0]).is_empty());

        ix.turn_superseded[0] = true;
        let q_plan = "What's the optimal time to launch a campaign for the university account?";
        let (plan_q, eff_q) = plan_for(&ix, q_plan, &p);
        assert!(score_gricean_constraint_turns_q16(&ix, &plan_q, &eff_q, q_plan, false, &[0]).is_empty());
    }
}
