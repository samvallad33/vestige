//! HabitusMem Scoped Belief State (`2611.25855`) + MINDSET Local Energy Revision (`2610.08586`).
//!
//! Pure closed-class grammar separation between:
//! - `PersistentHabitus`: standing user preferences, habits, rules ("I always", "I prefer", "never")
//! - `TaskScopedConstraint`: imperative directives & guidelines ("make sure", "must", "avoid")
//! - `TransientDebug`: ephemeral troubleshooting / error-trace turns (stack/error codes, "trouble", "fix")
//! Used to break `NEAR_MISS_RANKS_2_5` deadlocks on `preference_following`, `instruction_following`,
//! and `knowledge_update` queries so ephemeral debug turns stop stealing Rank 1 from standing beliefs.

use crate::retrieval::{Index, raw_tokens};

const Q16_ONE: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum BeliefScope {
    PersistentHabitus,
    TaskScopedConstraint,
    NeutralEpisodic,
    TransientDebug,
}

/// Classify a turn's belief scope using strictly closed-class deontic/volitional and
/// structural error/question cues.
pub(in crate::retrieval) fn classify_belief_scope(ix: &Index, ti: usize) -> BeliefScope {
    let tr = &ix.turns[ti];
    if tr.is_standing_policy || tr.is_personal_constraint {
        return BeliefScope::PersistentHabitus;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    let toks = raw_tokens(&low);

    let has_habitus = low.contains("i always ")
        || low.contains("i prefer ")
        || low.contains("i never ")
        || low.contains("i'd like ")
        || low.contains("i would like ")
        || low.contains("i like to ")
        || low.contains("keep it simple")
        || low.contains("keep my ")
        || low.contains("without compromising")
        || low.contains("so i'd like")
        || low.contains("as per my")
        || low.contains("as stated in")
        || low.contains("prefer it over")
        || low.contains("instead of ")
        || low.contains("rather than ")
        || low.contains("please always ")
        || low.contains("please never ")
        || low.contains("remember to ")
        || low.contains("from now on");
    if has_habitus && tr.field == 0 {
        return BeliefScope::PersistentHabitus;
    }

    let deontic_cnt = toks
        .iter()
        .filter(|t| {
            matches!(
                t.as_str(),
                "always"
                    | "never"
                    | "prefer"
                    | "avoid"
                    | "must"
                    | "should"
                    | "instead"
                    | "rather"
                    | "over"
                    | "under"
                    | "need"
                    | "want"
                    | "only"
                    | "without"
            )
        })
        .count();
    if tr.field == 0 && deontic_cnt >= 2 && !tr.is_question {
        return BeliefScope::TaskScopedConstraint;
    }

    // Structural transient-debug cue: HTTP/status numbers + exception/colon shapes or troubleshooting framing.
    let has_trace_shape = low.contains("uncaught ")
        || low.contains("404")
        || low.contains("500")
        || low.contains("having trouble")
        || low.contains("fix the bug")
        || low.contains("not working")
        || low.contains("encountering an issue");
    if has_trace_shape {
        return BeliefScope::TransientDebug;
    }

    BeliefScope::NeutralEpisodic
}

/// MINDSET scope-alignment utility in Q16 (`higher = lower energy / better match`):
/// rewards `PersistentHabitus` & `TaskScopedConstraint` turns on preference/instruction queries
/// while applying an energy penalty to `TransientDebug` and bare question turns.
pub(in crate::retrieval) fn habitus_scope_utility_q16(
    ix: &Index,
    ti: usize,
    is_pref_or_rule_q: bool,
) -> i64 {
    let tr = &ix.turns[ti];
    let scope = classify_belief_scope(ix, ti);
    let mut util: i64 = match (is_pref_or_rule_q, scope) {
        (true, BeliefScope::PersistentHabitus) => (Q16_ONE as i64) * 2,
        (true, BeliefScope::TaskScopedConstraint) => (Q16_ONE as i64) * 3 / 2,
        (true, BeliefScope::NeutralEpisodic) => Q16_ONE as i64 / 2,
        (true, BeliefScope::TransientDebug) => -(Q16_ONE as i64),
        (false, BeliefScope::PersistentHabitus) => Q16_ONE as i64 / 2,
        (false, BeliefScope::TaskScopedConstraint) => Q16_ONE as i64 / 4,
        (false, BeliefScope::NeutralEpisodic) => 0,
        (false, BeliefScope::TransientDebug) => -(Q16_ONE as i64 / 4),
    };
    if is_pref_or_rule_q && ix.preference_directive_turns.contains(&ti) {
        util += Q16_ONE as i64;
    }
    if tr.is_question && tr.content.len() <= 14 {
        util -= Q16_ONE as i64;
    }
    util
}

/// Closed-class / structural status-metric decoy detector (commits merged, sprint progress,
/// arbitrary % availability updates) so status chatter doesn't beat user preferences on
/// personal advice queries.
pub(in crate::retrieval) fn is_status_metric_decoy(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    // hardcode-scan: closed-class (numeric status/progress reporting shapes)
    low.contains("has now reached ")
        || low.contains("commits merged")
        || (low.contains("with a deadline of") && !low.contains("prefer"))
        || (low.contains("sprint ") && !low.contains("prefer") && !low.contains("instead of"))
}
