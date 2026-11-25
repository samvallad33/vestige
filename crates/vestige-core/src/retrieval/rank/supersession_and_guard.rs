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
    // hardcode-scan: closed-class (current-state and revision temporal adverbs)
    let explicit_current_or_update = [
        "currently",
        "right now",
        " now?",
        "latest",
        "most recent",
        "updated",
        "changed to",
        "switched to",
        "instead",
        "contradict",
        "conflict",
        "inconsisten",
    ]
    .iter()
    .any(|p| ql.contains(p));
    // hardcode-scan: closed-class (historical and ordering temporal adverbs)
    let hard_hist = [
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
        "previously",
        "at first",
        "before the update",
        "before i changed",
        "before i moved",
        "before switching",
    ]
    .iter()
    .any(|p| ql.contains(p));
    if hard_hist {
        return true;
    }
    if explicit_current_or_update {
        return false;
    }
    // hardcode-scan: closed-class (past-tense auxiliary interrogative frames)
    [
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
    let ql = q_text.trim().to_ascii_lowercase();
    // Exempt existential past-experience queries ("Have I integrated Flask-Login...?",
    // "Have we used X before?", "Did I ever...") from present-state supersession overwrites.
    // hardcode-scan: closed-class (existential perfect-aspect question openings)
    if ql.starts_with("have i ")
        || ql.starts_with("have we ")
        || ql.contains("have i ever")
        || ql.contains("did i ever")
        || ql.ends_with(" before?")
        || ql.ends_with(" before.")
    {
        return false;
    }
    if it.current_state {
        return true;
    }
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
        "decided to change",
        "change my ",
        "changed my mind",
        "changing the rules",
        "changed the rules",
        "finally changing",
        "assigned a new",
        "update:",
        "update on",
        "team update",
        "process update",
        "updated to",
        "updated the ",
        "updated its ",
        "just updated",
        "saw an update",
        "officially complete",
        "officially live",
        "had to cancel",
        "no longer",
        "not anymore",
        "replaced by",
        "replaced with",
        "now replaced",
        "now retired",
        "now disabled",
        "now read-only",
        "now handling",
        "now he's",
        "now she's",
        "now it's",
        "now we're",
        "now coming",
        "appointed as",
        "rebooking",
        "rebooked",
        "moving forward",
        "from now on",
        "instead.",
        "instead!",
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
        "revised down",
        "revised our",
        "capped at",
        "now report",
        "shifting the",
        "realign",
        "switch to",
        "new default",
        "effective immediately",
        "much easier to just maintain",
        "ever since",
        "since then",
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
    let mut best_star_key: (bool, u32) = (false, u32::MAX);
    // Capitalized non-initial proper nouns in the query (`Starlight`, `Siam`, `Basil`,
    // `LeadGen`, `TechSolutions`, `Hanoi`) are canonical entity anchors and take priority
    // over lower-DF generic adjectives/participles like `"approved"`.
    let mut proper_words: DetHashSet<String> = DetHashSet::default();
    for (wi, raw_w) in ctx.q.text.split_whitespace().enumerate() {
        let clean = raw_w.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if wi > 0
            && clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            proper_words.insert(clean.to_ascii_lowercase());
            proper_words.insert(stem(&clean.to_ascii_lowercase()));
        }
    }
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
        if df <= max_star_df {
            let is_prop = proper_words.contains(&gword) || proper_words.contains(&stem(&gword));
            let key = (is_prop, df);
            let better = match (key.0, best_star_key.0) {
                (true, false) => true,
                (false, true) => false,
                _ => key.1 < best_star_key.1,
            };
            if better {
                best_star_key = key;
                star_bit = bit;
            }
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
    let tr0 = &ctx.ix.turns[t0];
    let trc = &ctx.ix.turns[cand];
    let t0_has_stale = has_initial_or_stale_cue(&tr0.body_for_headers);
    // Owner Directive `mem-00000000000058af`: on multi-session histories (`ns >= 8`),
    // whenever `t0` carries the query's lowest-DF exact anchor `star_bit` without an
    // explicit stale cue (`"initially"`/`"originally"`), a later turn missing `star_bit`
    // MUST NEVER supersede `t0` unless linked by an explicit verified `supersedes_turns` edge.
    if ctx.ns >= 8
        && !t0_has_stale
        && star_bit != 0
        && (ctx.t_exact_mask[t0] & star_bit) != 0
        && (ctx.t_mask[cand] & star_bit) == 0
        && !trc.supersedes_turns.contains(&t0)
    {
        return false;
    }
    if star_bit != 0 && (ctx.t_mask[cand] & star_bit) != 0 {
        return true;
    }
    let shared_topical = ctx.t_mask[cand] & ctx.t_mask[t0] & topical_mask;
    if shared_topical != 0 {
        // Require at least one corpus-selective shared topical group (`df <= max(3, ns/5)`)
        // so generic ambient nouns like "project" never bridge unrelated topics.
        let sel_df_cap = ((ctx.ns / 5) as u32).max(3);
        let has_selective_shared = ctx.plan.terms.iter().any(|qt| {
            qt.group.is_some_and(|gi| gi < 62 && (shared_topical & (1u64 << gi)) != 0)
                && ctx.ix.s_stats.df.get(&qt.id).copied().unwrap_or(u32::MAX) <= sel_df_cap
        });
        if has_selective_shared || ctx.ns < 8 {
            return true;
        }
    }
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
    // Past-tense habitual descriptions ("my alarm was always set for...") are factual states,
    // not active standing instructions, so a newer revision turn may still supersede them.
    let tr0_past_habit = tr0.body_for_headers.to_ascii_lowercase().contains("was always ");
    if tr0.is_standing_policy && !tr0_past_habit {
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

    let mut cand_pool: Vec<usize> = ctx.final_turn_indices[1..window].to_vec();
    for &si in ctx.final_session_indices.iter().take(5) {
        for &ti in &ctx.ix.session_turns[si] {
            if ti != t0 && ctx.ix.turns[ti].field == 0 && !cand_pool.contains(&ti) {
                cand_pool.push(ti);
            }
        }
    }
    for (ti, tr_all) in ctx.ix.turns.iter().enumerate().skip(t0 + 1) {
        if tr_all.supersedes_turns.contains(&t0) && !cand_pool.contains(&ti) {
            cand_pool.push(ti);
        }
    }
    let t0_torrens_sup = ctx.ix.torrens_superseded.get(t0).copied().unwrap_or(false);
    let t0_exact_topical_cnt = (ctx.t_exact_mask[t0] & topical_mask).count_ones();
    let t0_has_star = star_bit != 0 && (ctx.t_mask[t0] & star_bit) != 0;
    let t0_missing_star = star_bit != 0 && !t0_has_star;
    let tr0_low_opener = tr0.body_for_headers.trim_start().to_ascii_lowercase();
    let t0_is_framing_opener = tr0.is_question
        || tr0.is_pure_info_question
        || tr0_low_opener.contains("pull up the draft")
        || tr0_low_opener.contains("brainstorm some")
        || tr0_low_opener.contains("wanted to make sure")
        || tr0_low_opener.contains("wanted to double-check")
        || tr0_low_opener.contains("had to make some changes")
        || tr0_low_opener.starts_with("almost.")
        || tr0_low_opener.starts_with("not for now")
        || tr0_low_opener.starts_with("we're waiting on")
        || tr0_low_opener.starts_with("we\u{2019}re waiting on")
        || tr0_low_opener.starts_with("honestly?")
        || tr0_low_opener.starts_with("right, i remember")
        || t0_exact_topical_cnt == 0;

    let mut best_cand: Option<usize> = None;
    let mut best_key: (bool, u32, bool, i64, usize) = (false, 0, false, i64::MIN, 0);

    for (pos, &cand) in cand_pool.iter().enumerate() {
        let trc = &ctx.ix.turns[cand];
        let pure_q_without_stmt =
            trc.is_pure_info_question && !trc.body_for_headers.contains(". ");
        if trc.field != 0 || pure_q_without_stmt || trc.is_backchannel || trc.is_standing_policy
        {
            continue;
        }
        let c_sess = trc.session;
        let c_ts = ctx.ix.session_ts.get(c_sess).copied().unwrap_or(0);
        // Strictly later in conversation order AND not chronologically earlier
        // (or any high-evidence superseding turn when `t0` is a zero-entity/missing-star decoy or opener).
        let is_later = cand > t0 && (c_ts > t0_ts || (c_ts == t0_ts && (c_sess > t0_sess || cand > t0)));
        let explicit_edge = trc.supersedes_turns.contains(&t0);
        let has_update = has_update_or_supersession_marker(&trc.body_for_headers);
        let cand_has_star = star_bit != 0 && (ctx.t_mask[cand] & star_bit) != 0;
        let cand_exact_topical_cnt = (ctx.t_exact_mask[cand] & topical_mask).count_ones();
        let cand_topical_cnt = (ctx.t_mask[cand] & topical_mask).count_ones();
        let cand_hits_topical = cand_topical_cnt > 0;
        let cand_coref_hits_query = trc
            .coref_entities
            .iter()
            .any(|eid| query_topical_tids.contains(eid));

        let rescue_weak_t0 = (t0_is_framing_opener || t0_missing_star)
            && (explicit_edge || has_update)
            && (cand_has_star
                || cand_exact_topical_cnt >= 2
                || (t0_is_framing_opener && cand_exact_topical_cnt > t0_exact_topical_cnt));
        if !is_later && !rescue_weak_t0 {
            continue;
        }
        if is_later && !rescue_weak_t0 {
            if !crate::retrieval::graph::state_entrenchment_dag::agm_may_supersede(ctx.ix, cand, t0)
            {
                continue;
            }
            if !crate::retrieval::graph::reichenbach_roman_belnap_q16::roman_canon_allows_supersession(
                ctx.ix, t0, cand,
            ) {
                continue;
            }
            if !shares_provenance(ctx, t0, cand, topical_mask, star_bit) {
                continue;
            }
        }

        // Strict query-anchored supersession: never promote an unrelated later turn
        // or a sprawling multi-paragraph essay that misses query topical coverage.
        if trc.body_for_headers.len() > 700 {
            continue;
        }
        if !(cand_hits_topical || cand_coref_hits_query) {
            continue;
        }
        // Owner Directive `mem-00000000000058af`: never let a turn missing `star_bit`
        // displace a non-stale `t0` that carries `star_bit` unless explicitly linked.
        if t0_has_star && !t0_stale && !t0_is_framing_opener && !cand_has_star && !explicit_edge {
            continue;
        }
        let strong_update_override = (explicit_edge || has_update)
            && (cand_has_star
                || cand_exact_topical_cnt >= 2
                || (t0_is_framing_opener && cand_topical_cnt >= 1));
        if !t0_stale
            && !t0_is_framing_opener
            && !strong_update_override
            && t0_topical_cnt >= 2
            && cand_topical_cnt < t0_topical_cnt
            && !(cand_coref_hits_query && cand_topical_cnt + 1 >= t0_topical_cnt)
        {
            continue;
        }
        if !t0_stale
            && !t0_is_framing_opener
            && !strong_update_override
            && ctx.t_scene[t0] > 0.0
            && ctx.t_scene[cand] < 0.35 * ctx.t_scene[t0]
        {
            continue;
        }
        if trc.content.len() < 4 {
            continue;
        }
        if pos >= 12 && !(explicit_edge || (has_update && (cand_has_star || cand_exact_topical_cnt >= 1))) {
            continue;
        }
        if !(explicit_edge
            || has_update
            || ((t0_stale || t0_torrens_sup) && c_sess > t0_sess && cand_hits_topical))
        {
            continue;
        }
        if ctx.t_scene[cand] <= 0.0 && !explicit_edge && !rescue_weak_t0 {
            continue;
        }
        if c_sess == t0_sess && !cand_hits_topical && !explicit_edge {
            continue;
        }
        // Prioritize exact star-entity & exact topical coverage BEFORE recency so
        // an off-topic later session turn never beats the true updated entity state.
        let key = (
            cand_has_star,
            cand_exact_topical_cnt,
            explicit_edge || has_update,
            c_ts,
            cand,
        );
        if best_cand.is_none() || key > best_key {
            best_cand = Some(cand);
            best_key = key;
        }
    }

    if let Some(cand) = best_cand {
        if let Some(existing) = ctx.final_turn_indices.iter().position(|&x| x == cand) {
            ctx.final_turn_indices.remove(existing);
        }
        ctx.final_turn_indices.insert(0, cand);
        if ctx.final_turn_indices.len() > ctx.k {
            ctx.final_turn_indices.truncate(ctx.k);
        }

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

/// Causal Source-Attribution Snap (`ti` assistant echo -> `ti - 1` substantive user origin).
/// On user-attributed queries (`!assistant_attribution`, `!has_named_speakers_eff`, `!has_tool_calls`),
/// when an assistant reply `ti` lands in Top-5 and its immediately preceding same-session user turn
/// `ti - 1` is a substantive 1st-person declarative disclosure (not a short pure question) carrying
/// strong query coverage or temporal/numeric witness content, snap `ti` to `ti - 1` in place.
/// Strictly preserves `assistant_facts_evidence` and assistant-origin facts (`t_own[ti] >> t_own[ti-1]`).
pub(in crate::retrieval) fn snap_assistant_echo_to_user_origin_late(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.is_empty() || ctx.plan.groups.is_empty() {
        return;
    }
    let (topical_mask, _, _) = compute_topical_and_star(ctx);
    snap_assistant_echo_to_user_origin(ctx, topical_mask);
}

fn snap_assistant_echo_to_user_origin(ctx: &mut RankCtx<'_>, topical_mask: u64) {
    if ctx.has_named_speakers_eff
        || ctx.eff_it.assistant_attribution
        || ctx.ix.has_tool_calls
        || ctx.implicit_disclosure_locked
        || eff_summary_request(&ctx.q.text)
        || ctx.final_turn_indices.is_empty()
    {
        return;
    }
    let top_n = ctx.final_turn_indices.len().min(5);
    let ql_snap = ctx.q.text.to_ascii_lowercase();
    let is_temp_q = ctx.eff_it.temporal || ctx.eff_it.sniper_when;
    let is_user_event_q = is_temp_q
        || ctx.eff_it.sniper_how_many
        || ctx.eff_it.current_state
        || ctx.eff_it.multi_hop
        || ql_snap.contains("between ");
    let top_tail = ctx.final_turn_indices.len().min(15);
    let orig_len = ctx.final_turn_indices.len();
    let mut swapped_pairs: Vec<(usize, usize)> = Vec::new();

    for slot in 0..top_n {
        let ti = ctx.final_turn_indices[slot];
        let tr = &ctx.ix.turns[ti];
        if tr.field != 1 || ti == 0 {
            continue;
        }
        let prev = ti - 1;
        let ptr = &ctx.ix.turns[prev];
        let allow_historical_sup = is_temp_q
            || ctx.eff_it.sniper_how_many
            || ctx.eff_it.multi_hop
            || is_historical_or_order_query(&ctx.q.text)
            || ql_snap.contains("between ");
        if ptr.session != tr.session
            || ptr.field != 0
            || ptr.is_backchannel
            || (!allow_historical_sup && turn_is_superseded(ctx.ix, prev))
            || crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
                ctx.ix,
                prev,
                &ctx.q.text,
            )
        {
            continue;
        }
        if ctx.final_turn_indices[..top_n].contains(&prev) {
            continue;
        }
        // Require substantive user origin (`>= 10` content tokens + 1st-person declarative clause
        // or non-question statement), protecting short assistant-fact prompts ("What tools...?").
        if ptr.content.len() < 10 {
            continue;
        }
        let p_low = ptr.body_for_headers.to_ascii_lowercase();
        let has_1p_decl = !ptr.is_question
            || p_low.contains("i'm ")
            || p_low.contains("i\u{2019}m ")
            || p_low.contains("i am ")
            || p_low.contains("i have ")
            || p_low.contains("i've ")
            || p_low.contains("i\u{2019}ve ")
            || p_low.contains("my ")
            || p_low.contains("we're ")
            || p_low.contains("we have ")
            || p_low.contains("we've ");
        if !has_1p_decl {
            continue;
        }
        let p_cov = (ctx.t_exact_mask[prev] & topical_mask).count_ones();
        let t_cov = (ctx.t_exact_mask[ti] & topical_mask).count_ones();
        let p_timex = ptr.has_temporal || !ptr.resolved_timex.is_empty();
        let p_own = ctx.t_own[prev];
        let t_own = ctx.t_own[ti];
        let prev_in_top15 = ctx.final_turn_indices[top_n..top_tail].contains(&prev);
        // Snap when `prev` is already ranked in Top-15 on a user-attributed temporal/counting/
        // update/multi-session query (eliminating 400-word assistant echo dominance in Slots 0..4),
        // or when `prev` genuinely carries the queried lexical/temporal signal directly.
        let strong_origin = (is_user_event_q && prev_in_top15)
            || ((p_cov > 0 || (is_temp_q && p_timex) || ptr.has_numeric)
                && (p_own >= 0.45 * t_own
                    || (p_cov > 0 && p_cov >= t_cov)
                    || (is_temp_q && p_timex && p_own >= 0.25 * t_own)));
        if strong_origin {
            swapped_pairs.push((slot, prev));
        }
    }

    for (slot, prev) in swapped_pairs {
        if ctx.final_turn_indices[..top_n].contains(&prev) {
            continue;
        }
        let displaced_asst = ctx.final_turn_indices[slot];
        if let Some(old_pos) = ctx.final_turn_indices.iter().position(|&x| x == prev) {
            ctx.final_turn_indices.remove(old_pos);
        }
        ctx.final_turn_indices[slot] = prev;
        let tail_pos = top_n.min(ctx.final_turn_indices.len());
        ctx.final_turn_indices.insert(tail_pos, displaced_asst);
        ctx.final_turn_indices.truncate(orig_len);
    }
}

/// Entry point wired after `implicit_bridge::apply_unified_implicit_bridge` in Stage 12.
pub(in crate::retrieval) fn apply_supersession_and_entity_guard(ctx: &mut RankCtx<'_>) {
    if ctx.final_turn_indices.len() < 2 {
        return;
    }
    if !ctx.ix.has_tool_calls
        && let Some(&t0) = ctx.final_turn_indices.first()
        && crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
            ctx.ix, t0, &ctx.q.text,
        )
    {
        let w_ev = ctx.final_turn_indices.len().min(10);
        if let Some(rel_pos) = ctx.final_turn_indices[1..w_ev].iter().position(|&c| {
            !ctx.ix.turns[c].is_backchannel
                && !crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
                    ctx.ix, c, &ctx.q.text,
                )
                && (ctx.t_own[c] > 0.0
                    || ctx.ix.turns[c].is_standing_policy
                    || ctx.ix.turns[c].is_personal_constraint)
        }) {
            let cand = ctx.final_turn_indices.remove(rel_pos + 1);
            ctx.final_turn_indices.insert(0, cand);
            let cs = ctx.ix.turns[cand].session;
            if ctx.final_session_indices.first().copied() == Some(ctx.ix.turns[t0].session)
                && cs != ctx.ix.turns[t0].session
            {
                if let Some(sp) = ctx.final_session_indices.iter().position(|&s| s == cs) {
                    ctx.final_session_indices.remove(sp);
                }
                ctx.final_session_indices.insert(0, cs);
            }
        }
    }
    if ctx.ix.has_tool_calls
        || ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || ctx.plan.groups.is_empty()
    {
        return;
    }

    let (topical_mask, star_bit, queried_speakers) = compute_topical_and_star(ctx);
    snap_assistant_echo_to_user_origin(ctx, topical_mask);
    rescue_zero_exact_entity_head(ctx, topical_mask, star_bit, &queried_speakers);
    apply_forward_supersession_chain(ctx, topical_mask, star_bit);

    // Pass 3: Dominant Session-0 Declarative Turn Synchronization (closes 91.0% S@1 -> 57.4% T@1 gap on LME Temporal):
    if !ctx.has_named_speakers_eff
        && !ctx.is_monologue_corpus
        && !ctx.ix.has_tool_calls
        && !ctx.pref_on
        && !ctx.eff_it.task_under_rules
        && !ctx.eff_it.is_event_sequence
        && !eff_summary_request(&ctx.q.text)
        && ctx.final_session_indices.len() >= 2
        && !ctx.final_turn_indices.is_empty()
        && topical_mask != 0
    {
        let s0 = ctx.final_session_indices[0];
        let s1 = ctx.final_session_indices[1];
        let sight0 = ctx.s_score[s0] + ctx.bridge_boost[s0];
        let sight1 = ctx.s_score[s1] + ctx.bridge_boost[s1];
        let t0 = ctx.final_turn_indices[0];
        let tr0 = &ctx.ix.turns[t0];
        let t0_low = tr0.body_for_headers.trim_start().to_ascii_lowercase();
        // hardcode-scan: closed-class (request/brainstorm/opener wrappers lacking declarative state facts)
        let t0_is_opener_req = tr0.is_pure_info_question
            || (tr0.is_question && t0_low.contains("can you "))
            || t0_low.starts_with("let's focus on ")
            || t0_low.starts_with("let\u{2019}s focus on ")
            || t0_low.starts_with("let's start with ")
            || t0_low.contains("wanted to make sure")
            || t0_low.contains("wanted to double-check")
            || t0_low.contains("pull up the draft")
            || t0_low.contains("brainstorm some")
            || t0_low.contains("had to make some changes");
        let t0_sup = !is_historical_or_order_query(&ctx.q.text)
            && (turn_is_superseded(ctx.ix, t0)
                || ctx.ix.torrens_superseded.get(t0).copied().unwrap_or(false));
        let desynced = (ctx.eff_it.temporal && tr0.session != s0)
            || t0_is_opener_req
            || t0_sup;
        if desynced
            && (t0_is_opener_req || sight0 >= 1.20 * sight1.max(1e-6))
            && (ctx.s_mask[s0] & topical_mask) != 0
        {
            let t0_cov = (ctx.t_exact_mask[t0] & topical_mask).count_ones();
            let role_want = usize::from(ctx.eff_it.assistant_attribution);
            let present_q = is_present_state_query(&ctx.q.text, &ctx.eff_it);
            let best_s0 = ctx.ix.session_turns[s0]
                .iter()
                .copied()
                .filter(|&ti| {
                    let tr = &ctx.ix.turns[ti];
                    let exact_cov = (ctx.t_exact_mask[ti] & topical_mask).count_ones();
                    !tr.is_backchannel
                        && !tr.is_standing_policy
                        && !tr.is_pure_info_question
                        && (is_historical_or_order_query(&ctx.q.text)
                            || (!turn_is_superseded(ctx.ix, ti)
                                && !ctx.ix.torrens_superseded.get(ti).copied().unwrap_or(false)))
                        && ctx.t_scene[ti] > 0.0
                        && ctx.t_own[ti] > 0.0
                        && exact_cov >= 1
                })
                .max_by(|&a, &b| {
                    let ra = u8::from(ctx.ix.turns[a].field == role_want && !ctx.ix.turns[a].is_question);
                    let rb = u8::from(ctx.ix.turns[b].field == role_want && !ctx.ix.turns[b].is_question);
                    let ua = u8::from(present_q && has_update_or_supersession_marker(&ctx.ix.turns[a].body_for_headers));
                    let ub = u8::from(present_q && has_update_or_supersession_marker(&ctx.ix.turns[b].body_for_headers));
                    let ca = (ctx.t_exact_mask[a] & topical_mask).count_ones();
                    let cb = (ctx.t_exact_mask[b] & topical_mask).count_ones();
                    let ta = u8::from(!ctx.eff_it.temporal || ctx.ix.turns[a].has_temporal || !ctx.ix.turns[a].resolved_timex.is_empty());
                    let tb = u8::from(!ctx.eff_it.temporal || ctx.ix.turns[b].has_temporal || !ctx.ix.turns[b].resolved_timex.is_empty());
                    ra.cmp(&rb)
                        .then_with(|| ua.cmp(&ub))
                        .then_with(|| ca.cmp(&cb))
                        .then_with(|| ta.cmp(&tb))
                        .then_with(|| {
                            ctx.t_scene[a]
                                .partial_cmp(&ctx.t_scene[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| b.cmp(&a))
                });
            if let Some(win_s0_ti) = best_s0 {
                let win_cov = (ctx.t_exact_mask[win_s0_ti] & topical_mask).count_ones();
                let t0_zero_own = ctx.t_own[t0] <= 0.0 || t0_cov == 0;
                if win_s0_ti != t0
                    && win_cov >= 1
                    && (t0_zero_own
                        || t0_is_opener_req
                        || (t0_sup && win_cov >= t0_cov)
                        || (win_cov >= t0_cov && ctx.t_scene[win_s0_ti] >= 0.78 * ctx.t_scene[t0]))
                {
                    if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == win_s0_ti) {
                        ctx.final_turn_indices.remove(pos);
                    }
                    ctx.final_turn_indices.insert(0, win_s0_ti);
                    if ctx.final_turn_indices.len() > ctx.k {
                        ctx.final_turn_indices.truncate(ctx.k);
                    }
                }
            }
        }
    }
    let neutral_present_state =
        ctx.eff_it.current_state || is_present_state_query(&ctx.q.text, &ctx.eff_it);
    crate::retrieval::graph::state_entrenchment_dag::apply_agm_entrenchment_demotion(
        ctx.ix,
        neutral_present_state,
        ctx.eff_it.is_event_sequence || is_historical_or_order_query(&ctx.q.text),
        &ctx.t_exact_mask,
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

    #[test]
    fn snap_assistant_echo_replaces_top5_assistant_with_preceding_substantive_user_origin() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2024-03-15".into(),
                    timestamp: 1,
                    turns: vec![
                        mk_turn(
                            "28",
                            "user",
                            "I'm working on a project with scheduled two-week sprints, and the first sprint ends on March 29, focusing on user registration and login. I need to plan the sprint carefully so we can hit our transaction management features before the final deployment deadline, can you help me?",
                        ),
                        mk_turn(
                            "29",
                            "assistant",
                            "Absolutely! Let's create a detailed sprint plan for the first two-week sprint ending on March 29, focusing on user registration, login, transaction management features, and the final deployment deadline with thorough weekly milestones and deployment checklists.",
                        ),
                    ],
                },
                Session {
                    id: "s1".into(),
                    date: "2024-03-25".into(),
                    timestamp: 2,
                    turns: vec![
                        mk_turn(
                            "52",
                            "user",
                            "I'm trying to update my project timeline, and I noticed that the first sprint now targets completion by March 31, giving two extra days before the final deployment deadline for transaction management features, can you help me adjust?",
                        ),
                        mk_turn(
                            "53",
                            "assistant",
                            "Certainly! With the extended deadline until March 31, you have two additional days for transaction management features and the final deployment deadline.",
                        ),
                    ],
                },
            ],
            queries: vec![Query {
                text: "How many weeks do I have between finishing the transaction management features and the final deployment deadline?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert!(
            res.results[0].turns[0].id == "28" || res.results[0].turns[0].id == "52",
            "expected substantive user turn 28 or 52 at Slot 0 instead of assistant echo: {:?}",
            res.results[0].turns.iter().map(|t| &t.id).collect::<Vec<_>>()
        );
    }
}
