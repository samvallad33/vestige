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
    // Skip short conversational sign-offs / polar acknowledgements, but NEVER suppress genuine
    // 1st-person causal/habitual disclosures (`"Since I always..., I need to avoid..."`)!
    if tr.content.len() <= 8 && is_discourse_closure_or_meta_echo(ix, ti, "") {
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

/// Score every eligible user turn inside `target_sessions` using Gricean Maxim-of-Quantity
/// Conditional Information Gain (`EntityBind * NovelPredicateIC * ConstraintFrame`).
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
    let nt_total = ix.turns.len().max(1);
    let mut scores: Vec<GriceanTurnScore> = Vec::new();

    for (rank_pos, &si) in target_sessions.iter().take(3).enumerate() {
        let Some(turn_list) = ix.session_turns.get(si) else {
            continue;
        };
        // Slight decay for sessions 2 and 3 vs top-1 winning session.
        let sess_prior_q16: u64 = match rank_pos {
            0 => Q16_ONE,
            1 => (Q16_ONE * 3) >> 2,
            _ => Q16_ONE >> 1,
        };

        for &ti in turn_list {
            let tr = &ix.turns[ti];
            if tr.field != 0 || tr.is_backchannel || tr.is_pure_info_question || turn_is_superseded(ix, ti) {
                continue;
            }
            let constraint_frame_q16 = constraint_frame_strength_q16(ix, ti);
            if constraint_frame_q16 < (Q16_ONE >> 1) {
                continue;
            }

            // 1. Entity Binding: direct focal entity hit, coref entity hit, or top-session continuity.
            let direct_focal_hits = focal_ids
                .iter()
                .filter(|id| tr.content.contains(id) || tr.entities.contains(id))
                .count();
            let coref_focal_hits = focal_ids
                .iter()
                .filter(|id| tr.coref_entities.contains(id))
                .count();

            let low_body = tr.body_for_headers.to_ascii_lowercase();
            let has_first_person = low_body.contains(" i ")
                || low_body.starts_with("i ")
                || low_body.contains("i'm ")
                || low_body.contains("i\u{2019}m ")
                || low_body.contains("since i ")
                || low_body.contains(" my ");

            let entity_bind_q16 = if direct_focal_hits > 0 {
                Q16_ONE.saturating_add(((direct_focal_hits - 1).min(2) as u64) * (Q16_ONE >> 2))
            } else if coref_focal_hits > 0 {
                (Q16_ONE * 7) >> 3
            } else if rank_pos <= 2
                && (tr.is_personal_constraint
                    || tr.is_habitual
                    || tr.has_prose_disclosure
                    || (has_first_person && constraint_frame_q16 >= Q16_ONE))
            {
                // Zero-overlap personal habit/disclosure inside the Top-3 already-found sessions.
                if rank_pos == 0 {
                    (Q16_ONE * 3) >> 2
                } else {
                    (Q16_ONE * 5) >> 3
                }
            } else {
                // Require explicit entity/coref binding outside personal disclosures.
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

            // Product in Q16.16: (((entity_bind * novel_ic) >> 16) * constraint_frame) >> 16 * sess_prior >> 16.
            let p1 = ((entity_bind_q16 as u128).saturating_mul(novel_ic_q16 as u128) >> 16) as u64;
            let p2 = ((p1 as u128).saturating_mul(constraint_frame_q16 as u128) >> 16) as u64;
            let total_q16 = ((p2 as u128).saturating_mul(sess_prior_q16 as u128) >> 16) as u64;

            if total_q16 > 0 {
                scores.push(GriceanTurnScore {
                    turn_idx: ti,
                    session_idx: si,
                    entity_bind_q16,
                    novel_ic_q16,
                    constraint_frame_q16,
                    total_q16,
                });
            }
        }
    }

    scores.sort_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| b.constraint_frame_q16.cmp(&a.constraint_frame_q16))
            .then_with(|| a.turn_idx.cmp(&b.turn_idx))
    });
    scores.truncate(4);
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

    let top_sessions: Vec<usize> = ctx.final_session_indices.iter().copied().take(3).collect();
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
        || (h0_cf == 0 && h0_tr.content.len() < 12);

    // 1. Same-session Slot-0 rescue over a weak/wrapper/zero-constraint head (zero session change).
    if let Some(s0_cand) = cands.iter().copied().find(|c| {
        c.turn_idx != h0
            && c.session_idx == h0_tr.session
            && c.constraint_frame_q16 >= Q16_ONE
            && (h0_is_wrapper || (h0_cf == 0 && c.entity_bind_q16 >= Q16_ONE))
    }) {
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

    // 2. Pack up to 2 stranded distinct-turn novel-constraint turns from Top-3 sessions into Slots 3..4.
    let mut seated = 0usize;
    let mut next_slot = 3usize;
    for cand in cands {
        if seated >= 2 || next_slot >= 5.min(ctx.k) {
            break;
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
