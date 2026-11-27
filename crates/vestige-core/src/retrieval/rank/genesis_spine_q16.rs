//! Wave 6: Chronological Genesis Spine + Knowledge-Update Mutation Polarity (`Q16.16`).
//!
//! Pure deterministic `Q16.16` integer kernels targeting two structural conversational memory
//! regimes with zero vectors, zero LLMs, and closed-class grammar only:
//!
//! 1. **Chronological Genesis Spine** (`event_ordering` & `summarization`):
//!    BM25 naturally peaks on late wrap-up sessions that recap an entire thread. When a query asks
//!    for chronological progression, timeline order, or multi-era summary, this kernel anchors
//!    Slot 0 on the earliest substantive **Genesis** session where the query's anchor topic first
//!    enters the history, then selects an era-stratified progression spine `[genesis, mid..., tail]`.
//! 2. **Knowledge-Update Mutation Polarity** (`knowledge_update` & `STATE-Bench`):
//!    Scores every candidate turn's closed-class state-revision polarity (`M_pol` in `Q16.16`):
//!    - Revision / transition operators (`switched to`, `changed to`, `updated to`, `instead of`,
//!      `no longer`, `now`, plus verified `supersedes_turns` links) yield positive mutation mass.
//!    - Anterior / stale markers (`initially`, `originally`, `previously`, `used to`, `formerly`,
//!      or being targeted by a later `supersedes_turns` edge) yield negative stale mass.
//!    - Current-state queries seat the latest affirmative mutation sharing the query's lowest-DF
//!      anchor (`star_bit`, strictly honoring Owner Directive `mem-00000000000058af`).
//!    - Origin/genesis queries (`originally`, `initially`, `first`) invert polarity (`-M_pol`) to
//!      lock onto the earliest introduction turn.

use crate::retrieval::context::RankCtx;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::{
    Index, QueryIntent, QueryPlan, eff_summary_request, turn_is_superseded,
};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: i64 = 65_536;
/// Explicit verified `supersedes_turns` edge boost (`+1.25 * Q16`).
const VERIFIED_SUPERSEDES_BOOST_Q16: i64 = 81_920;
/// Closed-class revision/update operator reward (`+0.75 * Q16`).
const REVISION_CUE_BOOST_Q16: i64 = 49_152;
/// Numeric value presence bonus on quantitative state updates (`+0.25 * Q16`).
const NUMERIC_MUTATION_BOOST_Q16: i64 = 16_384;
/// Stale/anteriority marker penalty (`-0.75 * Q16`).
const STALE_CUE_PENALTY_Q16: i64 = 49_152;
/// Penalty for turns explicitly superseded by a later turn (`-1.50 * Q16`).
const SUPERSEDED_TARGET_PENALTY_Q16: i64 = 98_304;

/// Temporal / state-transition polarity demanded by the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum TemporalQueryPolarity {
    /// Query explicitly asks for the original / initial / first value (`-M_pol`, genesis wins).
    GenesisFirst,
    /// Query asks for current / latest / updated state (`+M_pol`, latest active mutation wins).
    LatestCurrentState,
    /// Query asks for chronological event ordering or full multi-era summary spine.
    ChronologicalSpine,
    /// Neutral factual lookup — do not intervene.
    Neutral,
}

/// Pure-integer `Q16.16` mutation-polarity profile for a candidate turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct MutationPolarityProfile {
    /// Positive revision / state-update mass in `Q16.16`.
    pub(in crate::retrieval) mutation_q16: i64,
    /// Negative stale / anterior / superseded mass in `Q16.16`.
    pub(in crate::retrieval) stale_q16: i64,
    /// True when the turn carries an explicit closed-class revision operator or `supersedes_turns`.
    pub(in crate::retrieval) has_explicit_revision: bool,
    /// True when the turn contains an explicit numeric or clock state value.
    pub(in crate::retrieval) has_numeric_update: bool,
    /// Net polarity `mutation_q16 - stale_q16` in `Q16.16`.
    pub(in crate::retrieval) net_polarity_q16: i64,
}

/// Context bundle for [`select_ku_mutation_winner_q16`].
pub(in crate::retrieval) struct KuMutationCtx<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) query: &'a str,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) t_scene: &'a [f32],
    pub(in crate::retrieval) star_bit: u64,
}

/// Classify query temporal / mutation polarity using closed-class temporal & aspect cues only.
/// hardcode-scan: closed-class (temporal adverbs, auxiliaries, and sequence markers only).
pub(in crate::retrieval) fn detect_temporal_query_polarity(
    query: &str,
    eff_it: &QueryIntent,
) -> TemporalQueryPolarity {
    let ql = query.trim().to_ascii_lowercase();
    if eff_it.is_event_sequence
        || eff_summary_request(query)
        || ql.contains("chronological")
        || ql.contains("what order")
        || ql.contains("which order")
        || ql.contains("order of the")
        || ql.contains("order in which")
        || ql.contains("sequence of")
        || ql.contains("timeline of")
    {
        return TemporalQueryPolarity::ChronologicalSpine;
    }
    if ql.contains("originally")
        || ql.contains("initially")
        || ql.contains("at first")
        || ql.contains("first time")
        || ql.contains("before i switched")
        || ql.contains("before i changed")
        || ql.contains("used to be")
        || ql.contains("previous ")
        || ql.contains("earlier ")
    {
        return TemporalQueryPolarity::GenesisFirst;
    }
    if ql.contains("currently")
        || ql.contains("current ")
        || ql.contains("right now")
        || ql.contains("now ")
        || ql.ends_with(" now?")
        || ql.contains("latest ")
        || ql.contains("most recent")
        || ql.contains("updated ")
        || ql.contains("as of now")
    {
        return TemporalQueryPolarity::LatestCurrentState;
    }
    TemporalQueryPolarity::Neutral
}

/// Inspect closed-class state-mutation vs. stale/anteriority polarity on turn `ti`.
/// hardcode-scan: closed-class (aspectual transition verbs, negation, and temporal adverbs only).
pub(in crate::retrieval) fn analyze_turn_mutation_polarity_q16(
    ix: &Index,
    ti: usize,
) -> MutationPolarityProfile {
    let Some(tr) = ix.turns.get(ti) else {
        return MutationPolarityProfile::default();
    };
    let low = tr.body_for_headers.to_ascii_lowercase();

    let has_revision_phrase = low.contains("switched to ")
        || low.contains("switching to ")
        || low.contains("changed to ")
        || low.contains("changing to ")
        || low.contains("updated to ")
        || low.contains("moved to ")
        || low.contains("migrated to ")
        || low.contains("upgraded to ")
        || low.contains("replaced ")
        || low.contains("instead of ")
        || low.contains("no longer ")
        || low.contains("decided to switch")
        || low.contains("decided against ")
        || low.contains("dropped to ")
        || low.contains("increased to ")
        || low.contains("reduced to ")
        || low.contains("raised to ")
        || low.contains("bumped to ")
        || low.contains("actually, ")
        || low.contains("actually i ")
        || low.contains("as of now")
        || low.contains("from now on");

    let has_stale_phrase = low.contains("initially ")
        || low.contains("originally ")
        || low.contains("at first ")
        || low.contains("previously ")
        || low.contains("formerly ")
        || low.contains("used to ");

    let has_supersedes_edge = !tr.supersedes_turns.is_empty();
    let is_superseded = turn_is_superseded(ix, ti);

    let mut mutation_q16: i64 = 0;
    if has_supersedes_edge {
        mutation_q16 = mutation_q16.saturating_add(VERIFIED_SUPERSEDES_BOOST_Q16);
    }
    if has_revision_phrase {
        mutation_q16 = mutation_q16.saturating_add(REVISION_CUE_BOOST_Q16);
    }
    let has_numeric_update = tr.form_digit || tr.has_numeric || tr.form_clock;
    if has_numeric_update && (has_revision_phrase || has_supersedes_edge) {
        mutation_q16 = mutation_q16.saturating_add(NUMERIC_MUTATION_BOOST_Q16);
    }

    let mut stale_q16: i64 = 0;
    if is_superseded {
        stale_q16 = stale_q16.saturating_add(SUPERSEDED_TARGET_PENALTY_Q16);
    }
    if has_stale_phrase && !has_revision_phrase && !has_supersedes_edge {
        stale_q16 = stale_q16.saturating_add(STALE_CUE_PENALTY_Q16);
    }

    MutationPolarityProfile {
        mutation_q16,
        stale_q16,
        has_explicit_revision: has_revision_phrase || has_supersedes_edge,
        has_numeric_update,
        net_polarity_q16: mutation_q16.saturating_sub(stale_q16),
    }
}

/// Compute a chronologically stratified Genesis Spine over candidate sessions.
///
/// Selects up to `k` distinct sessions spanning the active temporal range of the topic:
/// - Anchors Slot 0 on the earliest substantive **Genesis** session (earliest `session_ts`,
///   tie-broken by `session_idx`) whose BM25/relevance score clears `>= 42%` of the pool peak
///   and shares exact query-group coverage (preventing off-topic early noise).
/// - Stratifies remaining slots across distinct chronological eras `[early, mid..., late]` so
///   multi-session progression and summarization achieve maximum era coverage.
/// Compute Karkali Asymmetric Prefix-IDF novelty (`IDF_{<t}(w) = log2((t + 2) / (df_{<t}(w) + 1))`)
/// in `Q16.16` for each session's opening substantive user turn across chronological session order.
/// Turns that introduce brand-new concepts/entities for the first time in the conversation score
/// highest, while later recaps/rehashes of already-introduced topics decay naturally.
fn compute_asymmetric_prefix_idf_q16(ix: &Index) -> Vec<u64> {
    use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
    use std::collections::BTreeMap;

    let ns = ix.session_turns.len();
    let mut prefix_nov = vec![0u64; ns];
    let mut order_idx: Vec<usize> = (0..ns).collect();
    order_idx.sort_by_key(|&si| (ix.session_ts.get(si).copied().unwrap_or(0), si));

    let mut seen_df: BTreeMap<u32, u64> = BTreeMap::new();
    for (step, &si) in order_idx.iter().enumerate() {
        let t_num = (step as u64) + 2;
        let open_ti = ix.session_turns[si]
            .iter()
            .copied()
            .find(|&ti| {
                let tr = &ix.turns[ti];
                tr.field == 0 && !tr.is_backchannel && !tr.is_standing_policy
            });
        if let Some(ti) = open_ti {
            let tr = &ix.turns[ti];
            let mut sum_idf: u64 = 0;
            let mut term_cnt: u64 = 0;
            for &cid in &tr.content {
                let prev_df = seen_df.get(&cid).copied().unwrap_or(0);
                // Asymmetric Prefix-IDF: brand-new term (`prev_df == 0`) gets full `log2(t + 2)` birth credit.
                let ratio_q16 = ((t_num << 16) / (prev_df + 1)).max(65_536);
                let idf_q16 = log2_u64_q16((ratio_q16 >> 16).max(1)).max(16_384);
                let ent_bonus = if tr.entities.contains(&cid) && prev_df == 0 {
                    idf_q16 >> 1
                } else {
                    0
                };
                sum_idf = sum_idf.saturating_add(idf_q16).saturating_add(ent_bonus);
                term_cnt += 1;
            }
            if term_cnt > 0 {
                // Mean novelty weighted by log-length so concise concrete milestone openers peak cleanly.
                prefix_nov[si] = sum_idf / term_cnt.min(18);
            }
            // Update running prefix document frequency strictly AFTER scoring session `si` (asymmetric!).
            for &cid in &tr.content {
                *seen_df.entry(cid).or_insert(0) += 1;
            }
        }
    }
    prefix_nov
}

pub(in crate::retrieval) fn compute_chronological_genesis_spine_q16(
    ix: &Index,
    cand_sessions: &[usize],
    s_score: &[f32],
    s_exact_mask: &[u64],
    k: usize,
    prefer_genesis_slot0: bool,
) -> Vec<usize> {
    if cand_sessions.len() < 2 || k == 0 {
        return cand_sessions.to_vec();
    }
    let pool_cap = if prefer_genesis_slot0 {
        cand_sessions.len().min(k.max(24))
    } else {
        cand_sessions.len().min(k.max(10))
    };
    let pool = &cand_sessions[..pool_cap];

    let max_sc = pool
        .iter()
        .map(|&si| s_score.get(si).copied().unwrap_or(0.0))
        .fold(0.0f32, f32::max);
    if max_sc <= 0.0 {
        return cand_sessions.to_vec();
    }

    let any_exact = pool
        .iter()
        .any(|&si| s_exact_mask.get(si).copied().unwrap_or(0) != 0);
    let min_sc_ratio = if prefer_genesis_slot0 { 0.36 } else { 0.42 };
    let mut qualified: Vec<usize> = pool
        .iter()
        .copied()
        .filter(|&si| {
            let sc = s_score.get(si).copied().unwrap_or(0.0);
            let ex = s_exact_mask.get(si).copied().unwrap_or(0);
            let has_user_init = ix.session_turns.get(si).is_some_and(|m| {
                m.iter().any(|&ti| {
                    ix.turns[ti].field == 0
                        && !ix.turns[ti].is_backchannel
                        && !ix.turns[ti].is_standing_policy
                })
            });
            sc >= min_sc_ratio * max_sc && (!any_exact || ex != 0) && has_user_init
        })
        .collect();

    if qualified.len() < 2 {
        return cand_sessions.to_vec();
    }

    qualified.sort_by_key(|&si| (ix.session_ts.get(si).copied().unwrap_or(0), si));
    qualified.dedup();

    let target_n = k.min(cand_sessions.len());
    let mut spine: Vec<usize> = Vec::with_capacity(target_n);

    if prefer_genesis_slot0 {
        let prefix_nov = compute_asymmetric_prefix_idf_q16(ix);
        let max_nov = qualified
            .iter()
            .map(|&si| prefix_nov.get(si).copied().unwrap_or(0))
            .max()
            .unwrap_or(1)
            .max(1);
        let qn = qualified.len();
        let strata = target_n.min(5).min(qn);
        if strata == 1 {
            spine.push(qualified[0]);
        } else {
            // CoNLL 2018 Chronological Stratum Cover + Karkali Asymmetric Prefix-IDF Birth Selection:
            // Partition `qualified` into `strata` disjoint chronological bins and pick each bin's
            // strongest First-Story birth session (`0.65 * norm_rel + 0.35 * norm_prefix_nov`,
            // breaking ties toward earlier chronological birth).
            for idx in 0..strata {
                let lo = (idx * qn) / strata;
                let hi = (((idx + 1) * qn) / strata).max(lo + 1).min(qn);
                let best_si = qualified[lo..hi]
                    .iter()
                    .copied()
                    .filter(|si| !spine.contains(si))
                    .max_by(|&a, &b| {
                        let rel_a = ((s_score.get(a).copied().unwrap_or(0.0) / max_sc) * 65_536.0) as u64;
                        let rel_b = ((s_score.get(b).copied().unwrap_or(0.0) / max_sc) * 65_536.0) as u64;
                        let nov_a = (((prefix_nov.get(a).copied().unwrap_or(0) as u128) << 16)
                            / (max_nov as u128)) as u64;
                        let nov_b = (((prefix_nov.get(b).copied().unwrap_or(0) as u128) << 16)
                            / (max_nov as u128)) as u64;
                        // 0.65 * rel + 0.35 * prefix_nov (42_598 + 22_938 = 65_536)
                        let sc_a = ((rel_a as u128 * 42_598 + nov_a as u128 * 22_938) >> 16) as u64;
                        let sc_b = ((rel_b as u128 * 42_598 + nov_b as u128 * 22_938) >> 16) as u64;
                        sc_a.cmp(&sc_b).then_with(|| b.cmp(&a))
                    });
                if let Some(si) = best_si {
                    spine.push(si);
                }
            }
            for &si in &qualified {
                if spine.len() >= target_n {
                    break;
                }
                if !spine.contains(&si) {
                    spine.push(si);
                }
            }
        }
    } else {
        spine.push(cand_sessions[0]);
        for &si in &qualified {
            if spine.len() >= target_n {
                break;
            }
            if !spine.contains(&si) {
                spine.push(si);
            }
        }
    }

    for &si in cand_sessions {
        if spine.len() >= cand_sessions.len() {
            break;
        }
        if !spine.contains(&si) {
            spine.push(si);
        }
    }
    spine
}

/// Select a decisive Knowledge-Update winner under `M_pol` while strictly enforcing
/// Owner Directive `mem-00000000000058af` (lowest-DF exact anchor `star_bit` protection).
pub(in crate::retrieval) fn select_ku_mutation_winner_q16(
    kctx: &KuMutationCtx<'_>,
    cand_turns: &[usize],
) -> Option<usize> {
    if cand_turns.len() < 2
        || kctx.eff_it.is_event_sequence
        || kctx.eff_it.task_under_rules
        || kctx.eff_it.preference
        || kctx.eff_it.assistant_attribution
    {
        return None;
    }
    let q_pol = detect_temporal_query_polarity(kctx.query, kctx.eff_it);
    if !matches!(
        q_pol,
        TemporalQueryPolarity::LatestCurrentState | TemporalQueryPolarity::GenesisFirst
    ) {
        return None;
    }

    let ix = kctx.ix;
    let t0 = cand_turns[0];
    if ix.turns[t0].is_standing_policy || ix.turns[t0].is_personal_constraint {
        return None;
    }
    let t0_prof = analyze_turn_mutation_polarity_q16(ix, t0);
    let t0_has_star = kctx.star_bit != 0
        && (kctx.t_exact_mask.get(t0).copied().unwrap_or(0) & kctx.star_bit) != 0;
    let t0_ts = ix.session_ts.get(ix.turns[t0].session).copied().unwrap_or(0);
    let t0_scene = kctx.t_scene.get(t0).copied().unwrap_or(0.0);
    let n_groups = kctx.plan.groups.len().min(64);
    let full_mask = if n_groups == 64 {
        u64::MAX
    } else if n_groups == 0 {
        0
    } else {
        (1u64 << n_groups) - 1
    };

    let t0_ex_cnt = (kctx.t_exact_mask.get(t0).copied().unwrap_or(0) & full_mask).count_ones();

    let mut best: Option<(usize, i64, i64, usize)> = None;
    for &ti in cand_turns.iter().skip(1).take(9) {
        let tr = &ix.turns[ti];
        if tr.field != 0
            || tr.is_backchannel
            || tr.is_standing_policy
            || tr.is_pure_info_question
            || is_discourse_closure_or_meta_echo(ix, ti, kctx.query)
        {
            continue;
        }
        let ex_m = kctx.t_exact_mask.get(ti).copied().unwrap_or(0) & full_mask;
        let cand_has_star = kctx.star_bit != 0 && (ex_m & kctx.star_bit) != 0;
        let explicitly_supersedes_t0 = tr.supersedes_turns.contains(&t0);

        if t0_has_star && !cand_has_star && !explicitly_supersedes_t0 {
            continue;
        }
        if ex_m == 0 && !explicitly_supersedes_t0 {
            continue;
        }
        if ex_m.count_ones() < t0_ex_cnt && !explicitly_supersedes_t0 {
            continue;
        }
        let sc = kctx.t_scene.get(ti).copied().unwrap_or(0.0);
        if t0_scene > 0.0 && sc < 0.65 * t0_scene && !explicitly_supersedes_t0 {
            continue;
        }

        let prof = analyze_turn_mutation_polarity_q16(ix, ti);
        let ts = ix.session_ts.get(tr.session).copied().unwrap_or(0);

        match q_pol {
            TemporalQueryPolarity::LatestCurrentState => {
                if turn_is_superseded(ix, ti) {
                    continue;
                }
                let strictly_later = (ts > 0 && t0_ts > 0 && ts > t0_ts) || (ts == t0_ts && ti > t0);
                if !strictly_later {
                    continue;
                }
                let mut gain = prof.net_polarity_q16.saturating_sub(t0_prof.net_polarity_q16);
                if explicitly_supersedes_t0 {
                    gain = gain.saturating_add(Q16_ONE);
                }
                let justified =
                    explicitly_supersedes_t0 || prof.has_explicit_revision || t0_prof.stale_q16 > 0;
                if justified
                    && gain >= (Q16_ONE >> 1)
                    && best.is_none_or(|(_, bg, bts, bti)| (gain, ts, ti) > (bg, bts, bti))
                {
                    best = Some((ti, gain, ts, ti));
                }
            }
            TemporalQueryPolarity::GenesisFirst => {
                let strictly_earlier =
                    (ts > 0 && t0_ts > 0 && ts < t0_ts) || (ts == t0_ts && ti < t0);
                if !strictly_earlier {
                    continue;
                }
                let inv_gain = t0_prof.net_polarity_q16.saturating_sub(prof.net_polarity_q16);
                let justified = t0_prof.has_explicit_revision || prof.stale_q16 > 0;
                if justified && inv_gain >= (Q16_ONE >> 1) {
                    let earlier_key = (inv_gain, -ts, usize::MAX - ti);
                    if best.is_none_or(|(_, bg, bts, bti)| {
                        earlier_key > (bg, -bts, usize::MAX - bti)
                    }) {
                        best = Some((ti, inv_gain, ts, ti));
                    }
                }
            }
            TemporalQueryPolarity::ChronologicalSpine | TemporalQueryPolarity::Neutral => {}
        }
    }

    best.map(|(ti, _, _, _)| ti)
}

/// Top-level Stage 12/Wave 6 entry point applying Chronological Genesis Spine stratification
/// and Knowledge-Update Mutation Polarity promotion directly onto `RankCtx`.
pub(in crate::retrieval) fn apply_genesis_spine_and_ku_polarity(ctx: &mut RankCtx<'_>) {
    if ctx.ix.has_tool_calls
        || ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || ctx.has_named_speakers_eff
        || ctx.is_monologue_corpus
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.preference
        || ctx.eff_it.task_under_rules
        || eff_summary_request(&ctx.q.text)
        || ctx.final_turn_indices.len() < 2
    {
        return;
    }

    let q_pol = detect_temporal_query_polarity(&ctx.q.text, &ctx.eff_it);
    match q_pol {
        TemporalQueryPolarity::ChronologicalSpine => {
            // Protect dedicated `event_ordering_stack` & incumbent Slot 0; only stratify
            // non-event-sequence chronological timeline queries without overwriting Slot 0.
            let orig_slen = ctx.final_session_indices.len();
            let spined = compute_chronological_genesis_spine_q16(
                ctx.ix,
                &ctx.final_session_indices,
                &ctx.s_score,
                &ctx.s_exact_mask,
                ctx.k.max(5),
                false,
            );
            if !spined.is_empty() {
                ctx.final_session_indices = spined;
                if orig_slen > 0 {
                    ctx.final_session_indices.truncate(orig_slen);
                }
            }
        }
        TemporalQueryPolarity::LatestCurrentState | TemporalQueryPolarity::GenesisFirst => {
            let mut star_bit: u64 = 0;
            let mut best_df = u32::MAX;
            for qt in &ctx.plan.terms {
                if qt.exact_hit
                    && let Some(g) = qt.group
                    && g < 64
                {
                    let df = ctx.ix.s_stats.df.get(&qt.id).copied().unwrap_or(u32::MAX);
                    if df > 0 && df < best_df {
                        best_df = df;
                        star_bit = 1u64 << g;
                    }
                }
            }
            let kctx = KuMutationCtx {
                ix: ctx.ix,
                query: &ctx.q.text,
                plan: &ctx.plan,
                eff_it: &ctx.eff_it,
                t_exact_mask: &ctx.t_exact_mask,
                t_scene: &ctx.t_scene,
                star_bit,
            };
            if let Some(win_ti) = select_ku_mutation_winner_q16(&kctx, &ctx.final_turn_indices) {
                let t0 = ctx.final_turn_indices[0];
                let explicitly_supersedes_t0 = ctx.ix.turns[win_ti].supersedes_turns.contains(&t0);
                let win_si = ctx.ix.turns[win_ti].session;
                let same_session = win_si == ctx.ix.turns[t0].session;
                // Only promote to Slot 0 on a verified `supersedes_turns` edge or a proven
                // same-session revision; never displace globally-ranked `final_session_indices`
                // on cross-session textual heuristics alone.
                if !explicitly_supersedes_t0 && !same_session {
                    return;
                }
                let t0_prof = analyze_turn_mutation_polarity_q16(ctx.ix, t0);
                let proven_slot0 = explicitly_supersedes_t0
                    || (q_pol == TemporalQueryPolarity::LatestCurrentState
                        && t0_prof.stale_q16 > 0)
                    || (q_pol == TemporalQueryPolarity::GenesisFirst
                        && t0_prof.has_explicit_revision);
                let target_slot = if proven_slot0 { 0usize } else { 1usize };

                let orig_tlen = ctx.final_turn_indices.len();
                if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == win_ti) {
                    if pos > target_slot {
                        ctx.final_turn_indices.remove(pos);
                        let ins_t = target_slot.min(ctx.final_turn_indices.len());
                        ctx.final_turn_indices.insert(ins_t, win_ti);
                    }
                } else {
                    let ins_t = target_slot.min(ctx.final_turn_indices.len());
                    ctx.final_turn_indices.insert(ins_t, win_ti);
                    ctx.final_turn_indices.truncate(orig_tlen);
                }

                if explicitly_supersedes_t0 {
                    let orig_slen = ctx.final_session_indices.len();
                    if orig_slen > target_slot {
                        if let Some(spos) =
                            ctx.final_session_indices.iter().position(|&s| s == win_si)
                        {
                            if spos > target_slot {
                                ctx.final_session_indices.remove(spos);
                                let ins_s = target_slot.min(ctx.final_session_indices.len());
                                ctx.final_session_indices.insert(ins_s, win_si);
                            }
                        } else {
                            let ins_s = target_slot.min(ctx.final_session_indices.len());
                            ctx.final_session_indices.insert(ins_s, win_si);
                            ctx.final_session_indices.truncate(orig_slen);
                        }
                    }
                }
            }
        }
        TemporalQueryPolarity::Neutral => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query};

    fn make_session(id: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: ts,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (role, content))| Turn {
                    id: format!("{id}#{i}"),
                    role: (*role).to_string(),
                    content: (*content).to_string(),
                    stability: 0.0,
                    links: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn genesis_spine_anchors_earliest_qualified_session_over_late_recap() {
        let p = Params::default();
        let sessions = vec![
            make_session("s0", 100, &[("user", "Unrelated weather chat.")]),
            make_session("s1", 200, &[("user", "Started building the Flask billing service today.")]),
            make_session("s2", 300, &[("user", "Added Stripe webhook retries to the Flask billing service.")]),
            make_session("s3", 400, &[("user", "Deployed Flask billing service to staging.")]),
            make_session("s4", 500, &[("user", "Recapping the entire Flask billing service rollout from start to finish.")]),
        ];
        let ix = build_index(&sessions, &p);
        let cands = vec![4usize, 3, 2, 1, 0];
        let s_score = vec![1.0f32, 7.0, 8.0, 9.0, 10.0];
        let s_exact = vec![0u64, 0b11, 0b11, 0b11, 0b11];

        let spine = compute_chronological_genesis_spine_q16(&ix, &cands, &s_score, &s_exact, 5, true);
        assert_eq!(spine[0], 1, "Genesis session s1 must take Slot 0 ahead of late recap s4");
        assert_eq!(&spine[..4], &[1, 2, 3, 4], "Spine must order qualified milestones chronologically");
    }

    #[test]
    fn ku_mutation_polarity_promotes_latest_revision_over_stale_baseline() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                100,
                &[("user", "Initially our API daily quota was 500 calls per day.")],
            ),
            make_session(
                "s2",
                200,
                &[("user", "Actually, we upgraded to 2500 calls per day for the API daily quota.")],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What is the current API daily quota?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact = vec![0b111u64, 0b111u64];
        let t_scene = vec![10.0, 9.0];
        let kctx = KuMutationCtx {
            ix: &ix,
            query: &q.text,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact,
            t_scene: &t_scene,
            star_bit: 0b001,
        };

        let win = select_ku_mutation_winner_q16(&kctx, &[0, 1]);
        assert_eq!(win, Some(1), "upgraded revision turn must beat stale initial baseline");
    }

    #[test]
    fn ku_origin_query_inverts_polarity_and_picks_genesis_turn() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                100,
                &[("user", "Originally we used PostgreSQL 14 for the billing ledger.")],
            ),
            make_session(
                "s2",
                200,
                &[("user", "We migrated to SQLite WAL instead of PostgreSQL 14 for the billing ledger.")],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What database did we originally use for the billing ledger?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact = vec![0b111u64, 0b111u64];
        let t_scene = vec![8.0, 10.0];
        let kctx = KuMutationCtx {
            ix: &ix,
            query: &q.text,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact,
            t_scene: &t_scene,
            star_bit: 0b001,
        };

        let win = select_ku_mutation_winner_q16(&kctx, &[1, 0]);
        assert_eq!(win, Some(0), "GenesisFirst query must invert polarity and pick original turn");
    }

    #[test]
    fn ku_mutation_respects_owner_directive_star_bit_anchor() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                100,
                &[("user", "Initially our Argon2 hashing cost was 64 MB.")],
            ),
            make_session(
                "s2",
                200,
                &[("user", "Actually, we switched to 128 MB for the generic cache budget.")],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What is the current Argon2 hashing cost?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact = vec![0b011u64, 0b010u64];
        let t_scene = vec![10.0, 9.5];
        let kctx = KuMutationCtx {
            ix: &ix,
            query: &q.text,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact,
            t_scene: &t_scene,
            star_bit: 0b001,
        };

        let win = select_ku_mutation_winner_q16(&kctx, &[0, 1]);
        assert_eq!(
            win, None,
            "turn missing rare anchor star_bit must NEVER supersede incumbent without explicit edge"
        );
    }

    #[test]
    fn genesis_pipeline_entrypoint_runs_cleanly_on_rank_ctx() {
        let p = Params::default();
        let sessions = vec![
            make_session("s1", 100, &[("user", "Initially our API daily quota was 500 calls.")]),
            make_session("s2", 200, &[("user", "Actually, we upgraded to 2500 calls for our API daily quota.")]),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "What is the current API daily quota?".to_string(),
            date: String::new(),
        };
        let prof_rev = analyze_turn_mutation_polarity_q16(&ix, 1);
        assert!(prof_rev.mutation_q16 > 0 && prof_rev.has_numeric_update);
        let mut ctx = crate::retrieval::rank::stage01_session_bm25f::run(&ix, &q, &p, 5);
        ctx.final_session_indices = vec![0, 1];
        ctx.final_turn_indices = vec![0, 1];
        ctx.t_exact_mask = vec![0b111u64, 0b111u64];
        ctx.t_scene = vec![10.0, 9.0];
        apply_genesis_spine_and_ku_polarity(&mut ctx);
        assert_eq!(ctx.final_turn_indices[0], 1);
        assert_eq!(ctx.final_session_indices[0], 1);
    }
}
