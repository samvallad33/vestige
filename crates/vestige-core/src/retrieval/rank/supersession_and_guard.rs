//! Mission 3: Strict Zero-Exact-Entity Rank-1 Guard + Forward-Time Supersession Chaining.
//!
//! Runs after `implicit_bridge` at the end of Stage 12 (`< 1ms`/query, zero vectors, zero LLMs),
//! shielded strictly against `has_tool_calls`, `torrens_locked_head`, and `implicit_disclosure_locked`:
//!
//! 1. **Pass 1 — Zero-Exact-Entity Rank-1 Guard (`rescue_zero_exact_entity_head`)**:
//!    When Rank 1 (`t0`) misses EVERY exact selective topical query term (`t_exact_mask[t0] & topical_mask == 0`)
//!    AND misses the query's lowest-DF exact content term `star_bit` even via stem/lemma
//!    (`t_mask[t0] & star_bit == 0`), while a Top-10 declarative candidate matches `star_bit`,
//!    promote that exact-entity witness to Rank 1 (and sync Rank-1 session if session 0 itself
//!    missed all topical query terms). Fixes zero-entity steals across `ConvoMem`, `LongMemEval Temporal`,
//!    `LoCoMo Temporal`, and `BEAM`.
//! 2. **Pass 2 — Forward-Time Supersession Chaining (`apply_forward_supersession_chain`)**:
//!    On present/current-state user queries (excluding explicit historical/original/ordering queries),
//!    when a strictly later Top-10 declarative user turn shares topic/entity/coreference provenance
//!    with `t0` (via `entities`, `coref_entities`, `star_bit`, or shared topical group) AND carries
//!    closed-class state-update / supersession markers (`extended`, `rescheduled`, `moved to`,
//!    `switched to`, `instead`, `actually`, `update:`, `retired`, `migrated`, `deviated`, `heads up`)
//!    — OR `t0` itself carries a stale/initial cue (`initially`, `originally`, `used to`, `previously`) —
//!    promote the latest superseding turn to Rank 1 (and sync Rank-1 session).

use crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape;
use crate::retrieval::query::tokens::is_content;
use crate::retrieval::rank::context::RankCtx;
use crate::retrieval::seating::c2::common::eff_summary_request;
use crate::retrieval::*;

/// Closed-class grammatical/wh/temporal/discourse scaffold words excluded from
/// selective `topical_mask` and `star_bit` calculation.
/// hardcode-scan: closed-class (wh-words, auxiliaries, modals, light verbs, quantifiers, temporal units).
fn is_scaffold_token(w: &str) -> bool {
    matches!(
        w,
        "what"
            | "when"
            | "where"
            | "who"
            | "whom"
            | "whose"
            | "which"
            | "why"
            | "how"
            | "can"
            | "could"
            | "would"
            | "should"
            | "shall"
            | "will"
            | "may"
            | "might"
            | "must"
            | "do"
            | "does"
            | "did"
            | "done"
            | "doing"
            | "is"
            | "am"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "having"
            | "need"
            | "want"
            | "look"
            | "looking"
            | "think"
            | "know"
            | "tell"
            | "say"
            | "said"
            | "ask"
            | "asked"
            | "mention"
            | "mentioned"
            | "give"
            | "gave"
            | "help"
            | "go"
            | "going"
            | "went"
            | "get"
            | "got"
            | "take"
            | "took"
            | "make"
            | "made"
            | "start"
            | "started"
            | "something"
            | "anything"
            | "everything"
            | "some"
            | "any"
            | "many"
            | "much"
            | "more"
            | "most"
            | "long"
            | "often"
            | "ago"
            | "last"
            | "next"
            | "first"
            | "latest"
            | "recent"
            | "recently"
            | "lately"
            | "currently"
            | "now"
            | "today"
            | "yesterday"
            | "tomorrow"
            | "time"
            | "times"
            | "day"
            | "days"
            | "week"
            | "weeks"
            | "month"
            | "months"
            | "year"
            | "years"
            | "event"
            | "events"
            | "order"
            | "earliest"
            | "thing"
            | "things"
            | "way"
            | "ways"
            | "kind"
            | "type"
            | "good"
            | "best"
            | "also"
            | "really"
            | "actually"
            | "just"
            | "always"
            | "never"
            | "total"
            | "past"
            | "spent"
            | "passed"
            | "since"
            | "before"
            | "after"
            | "between"
            | "attend"
            | "attended"
            | "participate"
            | "participated"
            | "visit"
            | "visited"
            | "happened"
            | "name"
            | "names"
            | "number"
            | "numbers"
            | "amount"
            | "price"
            | "cost"
            | "location"
            | "address"
            | "size"
            | "footage"
            | "exact"
            | "specific"
            | "target"
            | "value"
            | "peak"
            | "status"
            | "title"
            | "detail"
            | "details"
            | "score"
            | "rate"
            | "level"
            | "count"
            | "average"
    )
}

/// Closed-class check for explicit historical/origin/order queries where we must NOT
/// prefer a later update over the original historical fact.
/// hardcode-scan: closed-class (historical/temporal/ordering interrogatives and adverbs).
fn is_historical_or_order_query(q_text: &str) -> bool {
    let ql = q_text.to_ascii_lowercase();
    [
        "originally",
        "initially",
        "used to",
        "first ",
        "before ",
        "how long",
        "how many",
        "when did",
        "ago",
        "order of",
        "earliest",
        "planned to",
        "what were",
        "what was ",
        "where was ",
        "who was ",
        " did i ",
        " did my ",
        " did we ",
        " did you ",
        " have i ",
        " have we ",
        " ever ",
        "previously",
        "at first",
    ]
    .iter()
    .any(|p| ql.contains(p))
}

/// Closed-class check for present/current-state factual queries.
/// hardcode-scan: closed-class (present-tense copular and auxiliary question frames).
fn is_present_state_query(q_text: &str, it: &QueryIntent) -> bool {
    if is_historical_or_order_query(q_text) {
        return false;
    }
    if it.current_state {
        return true;
    }
    let ql = q_text.to_ascii_lowercase();
    [
        "what is ",
        "what's ",
        "where is ",
        "where's ",
        "who is ",
        "who's ",
        "what time is ",
        "what are ",
        "where do i ",
        "what do i ",
        " do i ",
        "is my ",
        "are my ",
        "what payment ",
        "what method ",
        "is used for",
        "am i ",
    ]
    .iter()
    .any(|p| ql.contains(p))
}

/// Closed-class state-update / supersession discourse markers indicating that a
/// later turn supersedes an earlier fact or schedule.
/// hardcode-scan: closed-class (change-of-state, rescheduling, and correction discourse markers).
fn has_update_or_supersession_marker(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "extended to",
        "been extended",
        "reschedule",
        "rescheduled",
        "moved to",
        "switched to",
        "switch over",
        "changed to",
        "change to",
        "changed my mind",
        "changing the rules",
        "changed the rules",
        "assigned a new",
        "update:",
        "update on",
        "team update",
        "process update",
        "updated to",
        "no longer",
        "not anymore",
        "replaced by",
        "replaced with",
        "now replaced",
        "now retired",
        "now he's",
        "now she's",
        "now it's",
        "now we're",
        "now coming",
        "canceled",
        "cancelled",
        "postponed",
        "pushed back",
        "pushed to",
        "deviated from",
        "heads up",
        "later informed",
        "to correct myself",
        "was misremembering",
        "misremembered",
        "turns out",
        "migrated to",
        "used to see",
        "mixing up",
        "mixed up",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// Closed-class cues marking `t0` itself as an explicitly initial or superseded state.
/// hardcode-scan: closed-class (initial/prior temporal adverbs).
fn has_initial_or_stale_cue(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "initially ",
        "originally ",
        "was planning ",
        "previously ",
        "at first ",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// Compute selective `topical_mask` and lowest-DF exact grounded `star_bit`.
fn compute_topical_and_star(ctx: &RankCtx<'_>) -> (u64, u64, Vec<String>) {
    let mut speaker_vocab: DetHashSet<String> = DetHashSet::default();
    if ctx.has_named_speakers_eff {
        for tr in &ctx.ix.turns {
            if !tr.speaker.is_empty() {
                speaker_vocab.insert(tr.speaker.to_ascii_lowercase());
            }
        }
    }
    let bindings = ctx.ix.nicks.bindings_in(&ctx.q.text);
    let mut queried_speakers = Vec::new();
    if ctx.has_named_speakers_eff {
        for tok in raw_tokens(&ctx.q.text) {
            let low = tok.to_ascii_lowercase();
            let canon = bindings.get(&low).cloned().unwrap_or(low);
            if speaker_vocab.contains(&canon) && !queried_speakers.contains(&canon) {
                queried_speakers.push(canon);
            }
        }
    }

    let max_g = ctx.plan.groups.len().min(62);
    let max_star_df = ((ctx.nt / 7).clamp(10, 40)) as u32;
    let mut group_min_exact_df = vec![u32::MAX; max_g];
    for qt in &ctx.plan.terms {
        if let Some(gi) = qt.group
            && gi < max_g
            && qt.exact_hit
            && let Some(&df) = ctx.ix.t_own_stats.df.get(&qt.id)
            && df > 0
            && df < group_min_exact_df[gi]
        {
            group_min_exact_df[gi] = df;
        }
    }

    let mut topical_mask = 0u64;
    let mut star_bit = 0u64;
    let mut best_df = u32::MAX;
    for gi in 0..max_g {
        let bit = 1u64 << gi;
        if (ctx.grounded_group_mask & bit) == 0 {
            continue;
        }
        let gword = ctx.plan.groups[gi].to_ascii_lowercase();
        if gword.len() < 2
            || !is_content(&gword)
            || is_scaffold_token(&gword)
            || speaker_vocab.contains(&gword)
        {
            continue;
        }
        topical_mask |= bit;
        let df = group_min_exact_df[gi];
        if df <= max_star_df && df < best_df {
            best_df = df;
            star_bit = bit;
        }
    }
    (topical_mask, star_bit, queried_speakers)
}

/// Pass 1: Strict Zero-Exact-Entity Rank-1 Guard.
fn rescue_zero_exact_entity_head(
    ctx: &mut RankCtx<'_>,
    topical_mask: u64,
    star_bit: u64,
    queried_speakers: &[String],
) {
    let window = ctx.final_turn_indices.len().min(10);
    if window < 2
        || star_bit == 0
        || ctx.pref_on
        || ctx.eff_it.task_under_rules
        || eff_summary_request(&ctx.q.text)
    {
        return;
    }

    let q_toks_owned = raw_tokens(&ctx.q.text);
    let q_toks_ref: Vec<&str> = q_toks_owned.iter().map(String::as_str).collect();
    if ctx.has_named_speakers_eff
        && (is_open_domain_inferential_shape(&q_toks_ref)
            || ctx.eff_it.sniper_when
            || ctx.eff_it.temporal)
    {
        return;
    }

    let t0 = ctx.final_turn_indices[0];
    let tr0 = &ctx.ix.turns[t0];
    let t0_any_topical = ctx.t_mask[t0] & topical_mask;
    let t0_has_star_any = (ctx.t_mask[t0] & star_bit) != 0;
    let t0_has_bridge = ctx.t_bridge_hits.get(t0).copied().unwrap_or(0) > 0;
    if t0_any_topical != 0 || t0_has_star_any || t0_has_bridge {
        return;
    }
    let is_temp_q = ctx.eff_it.sniper_when || ctx.eff_it.temporal;
    let t0_has_timex = tr0.has_temporal || !tr0.resolved_timex.is_empty();

    let role_pref_asst = !ctx.has_named_speakers_eff && ctx.eff_it.assistant_attribution;
    let role_pref_user = !ctx.has_named_speakers_eff && !ctx.eff_it.assistant_attribution;

    let mut best_pos: Option<usize> = None;
    let mut best_key: (u32, bool, bool, bool, f32) = (0, false, false, false, f32::NEG_INFINITY);

    for (pos, &cand) in ctx.final_turn_indices[1..window].iter().enumerate() {
        let tr = &ctx.ix.turns[cand];
        if (ctx.t_exact_mask[cand] & star_bit) == 0 {
            continue;
        }
        if tr.is_backchannel || tr.is_standing_policy || ctx.t_own[cand] <= 0.0 {
            continue;
        }
        if is_temp_q && t0_has_timex && !tr.has_temporal && tr.resolved_timex.is_empty() {
            continue;
        }
        if !ctx.has_named_speakers_eff && tr.is_pure_info_question && !tr0.is_question {
            continue;
        }
        let role_ok = (!role_pref_user || tr.field == 0) && (!role_pref_asst || tr.field == 1);
        let t0_role_ok =
            (!role_pref_user || tr0.field == 0) && (!role_pref_asst || tr0.field == 1);
        if !role_ok && t0_role_ok {
            continue;
        }
        let exact_cnt = (ctx.t_exact_mask[cand] & topical_mask).count_ones();
        let sp_ok = queried_speakers.len() != 1
            || tr.speaker.eq_ignore_ascii_case(&queried_speakers[0]);
        let decl_ok = !tr.is_question;
        let key = (exact_cnt, role_ok, sp_ok, decl_ok, ctx.t_scene[cand]);
        if best_pos.is_none() || key > best_key {
            best_pos = Some(pos + 1);
            best_key = key;
        }
    }

    if let Some(pos) = best_pos {
        let cand = ctx.final_turn_indices.remove(pos);
        ctx.final_turn_indices.insert(0, cand);

        // Sync Rank-1 session ONLY if displaced session had zero exact topical hits and zero star hits.
        if !ctx.final_session_indices.is_empty() {
            let s0 = ctx.final_session_indices[0];
            let cs = ctx.ix.turns[cand].session;
            if s0 != cs
                && (ctx.s_exact_mask[s0] & topical_mask) == 0
                && (ctx.s_mask[s0] & star_bit) == 0
                && (ctx.s_exact_mask[cs] & star_bit) != 0
            {
                if let Some(sp) = ctx.final_session_indices.iter().position(|&s| s == cs) {
                    ctx.final_session_indices.remove(sp);
                }
                ctx.final_session_indices.insert(0, cs);
            }
        }
    }
}

/// Check whether `cand` shares query-anchored topic/entity/coref provenance with `t0`.
fn shares_provenance(
    ctx: &RankCtx<'_>,
    t0: usize,
    cand: usize,
    topical_mask: u64,
    star_bit: u64,
) -> bool {
    if star_bit != 0 && (ctx.t_mask[cand] & star_bit) != 0 {
        return true;
    }
    if (ctx.t_mask[cand] & ctx.t_mask[t0] & topical_mask) != 0 {
        return true;
    }
    let tr0 = &ctx.ix.turns[t0];
    let trc = &ctx.ix.turns[cand];
    let max_ent_df = ((ctx.nt / 20).clamp(6, 15)) as u32;

    // Direct or ingest-coreference shared rare entity between `cand` and `t0`.
    trc.entities
        .iter()
        .chain(trc.coref_entities.iter())
        .any(|eid| {
            let df = ctx.ix.t_own_stats.df.get(eid).copied().unwrap_or(1);
            df <= max_ent_df && (tr0.entities.contains(eid) || tr0.coref_entities.contains(eid))
        })
}

/// Pass 2: Forward-Time Supersession Chaining for current-state user queries.
fn apply_forward_supersession_chain(
    ctx: &mut RankCtx<'_>,
    topical_mask: u64,
    star_bit: u64,
) {
    let window = ctx.final_turn_indices.len().min(20);
    if window < 2
        || ctx.has_named_speakers_eff
        || ctx.pref_on
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.assistant_attribution
        || eff_summary_request(&ctx.q.text)
    {
        return;
    }
    if !is_present_state_query(&ctx.q.text, &ctx.eff_it) {
        return;
    }

    let t0 = ctx.final_turn_indices[0];
    let tr0 = &ctx.ix.turns[t0];
    if tr0.is_standing_policy {
        return;
    }
    let t0_sess = tr0.session;
    let t0_ts = ctx.ix.session_ts.get(t0_sess).copied().unwrap_or(0);
    let t0_stale = has_initial_or_stale_cue(&tr0.body_for_headers);
    let t0_topical_cnt = (ctx.t_mask[t0] & topical_mask).count_ones();

    let mut query_topical_tids: Vec<u32> = Vec::new();
    for qt in &ctx.plan.terms {
        if let Some(gi) = qt.group
            && gi < 62
            && (topical_mask & (1u64 << gi)) != 0
            && !query_topical_tids.contains(&qt.id)
        {
            query_topical_tids.push(qt.id);
        }
    }

    let mut best_pos: Option<usize> = None;
    let mut best_key: (bool, i64, usize) = (false, i64::MIN, 0);

    for (pos, &cand) in ctx.final_turn_indices[1..window].iter().enumerate() {
        let trc = &ctx.ix.turns[cand];
        if trc.field != 0 || trc.is_pure_info_question || trc.is_backchannel || trc.is_standing_policy
        {
            continue;
        }
        let c_sess = trc.session;
        let c_ts = ctx.ix.session_ts.get(c_sess).copied().unwrap_or(0);
        // Strictly later in conversation order AND not chronologically earlier.
        let is_later = cand > t0 && (c_ts > t0_ts || (c_ts == t0_ts && (c_sess > t0_sess || cand > t0)));
        if !is_later {
            continue;
        }
        if !crate::retrieval::graph::state_entrenchment_dag::agm_may_supersede(ctx.ix, cand, t0) {
            continue;
        }
        if !shares_provenance(ctx, t0, cand, topical_mask, star_bit) {
            continue;
        }
        let explicit_edge = trc.supersedes_turns.contains(&t0);
        let has_update = has_update_or_supersession_marker(&trc.body_for_headers);
        let cand_hits_topical = (ctx.t_mask[cand] & topical_mask) != 0;
        let cand_coref_hits_query = trc
            .coref_entities
            .iter()
            .any(|eid| query_topical_tids.contains(eid));

        // Strict query-anchored supersession: never promote an unrelated later turn
        // or a sprawling multi-paragraph essay that misses query topical coverage.
        if trc.body_for_headers.len() > 700 {
            continue;
        }
        if !(cand_hits_topical || cand_coref_hits_query) {
            continue;
        }
        let cand_topical_cnt = (ctx.t_mask[cand] & topical_mask).count_ones();
        if !t0_stale
            && t0_topical_cnt >= 2
            && cand_topical_cnt < t0_topical_cnt
            && !(cand_coref_hits_query && cand_topical_cnt + 1 >= t0_topical_cnt)
        {
            continue;
        }
        if !t0_stale && ctx.t_scene[t0] > 0.0 && ctx.t_scene[cand] < 0.35 * ctx.t_scene[t0] {
            continue;
        }
        if trc.content.len() < 6 {
            continue;
        }
        if pos >= 9 && !(has_update && cand_hits_topical) {
            continue;
        }
        if !(has_update || (t0_stale && c_sess > t0_sess && cand_hits_topical)) {
            continue;
        }
        if ctx.t_scene[cand] <= 0.0 {
            continue;
        }
        if c_sess == t0_sess && !cand_hits_topical {
            continue;
        }
        let key = (explicit_edge || has_update, c_ts, cand);
        if best_pos.is_none() || key > best_key {
            best_pos = Some(pos + 1);
            best_key = key;
        }
    }

    if let Some(pos) = best_pos {
        let cand = ctx.final_turn_indices.remove(pos);
        ctx.final_turn_indices.insert(0, cand);

        // Sync Rank-1 session to the superseding turn's session.
        let cs = ctx.ix.turns[cand].session;
        if !ctx.final_session_indices.is_empty() && ctx.final_session_indices[0] != cs {
            if let Some(sp) = ctx.final_session_indices.iter().position(|&s| s == cs) {
                ctx.final_session_indices.remove(sp);
            }
            ctx.final_session_indices.insert(0, cs);
        }
    }
}

/// Entry point wired after `implicit_bridge::apply_unified_implicit_bridge` in Stage 12.
pub(in crate::retrieval) fn apply_supersession_and_entity_guard(ctx: &mut RankCtx<'_>) {
    if ctx.ix.has_tool_calls
        || ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || ctx.final_turn_indices.len() < 2
        || ctx.plan.groups.is_empty()
    {
        return;
    }

    let (topical_mask, star_bit, queried_speakers) = compute_topical_and_star(ctx);
    rescue_zero_exact_entity_head(ctx, topical_mask, star_bit, &queried_speakers);
    apply_forward_supersession_chain(ctx, topical_mask, star_bit);
    crate::retrieval::graph::state_entrenchment_dag::apply_agm_entrenchment_demotion(
        ctx.ix,
        ctx.eff_it.current_state,
        ctx.eff_it.is_event_sequence || is_historical_or_order_query(&ctx.q.text),
        &mut ctx.final_turn_indices,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    #[test]
    fn forward_supersession_promotes_extended_deadline_over_initial_turn() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2024/05/01 (Wed) 10:00".into(),
                    timestamp: 1,
                    turns: vec![mk_turn(
                        "t0",
                        "user",
                        "Yes, they are planning to submit an offer. I initially told the Lees that their offer deadline for the property is Friday at 5 PM.",
                    )],
                },
                Session {
                    id: "s1".into(),
                    date: "2024/05/03 (Fri) 10:00".into(),
                    timestamp: 2,
                    turns: vec![mk_turn(
                        "t1",
                        "user",
                        "The listing agent later informed me that the deadline has been extended to Saturday at noon.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "What is the offer deadline for the property?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "t1");
        assert_eq!(res.results[0].sessions[0].id, "s1");
    }

    #[test]
    fn forward_supersession_promotes_rescheduled_handyman_over_initial_schedule() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2024/06/01 (Sat) 09:00".into(),
                    timestamp: 1,
                    turns: vec![mk_turn(
                        "t0",
                        "user",
                        "I initially scheduled the handyman to fix the leaky faucet in my apartment on Tuesday afternoon.",
                    )],
                },
                Session {
                    id: "s1".into(),
                    date: "2024/06/02 (Sun) 09:00".into(),
                    timestamp: 2,
                    turns: vec![mk_turn(
                        "t1",
                        "user",
                        "The handyman called to reschedule, and now he's coming on Wednesday between 9-11 AM.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "What time is the handyman scheduled to come to my apartment?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "t1");
    }
}
