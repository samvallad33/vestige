//! C2 seating lane: `answers_and_pref`.

use crate::retrieval::*;



/// Share of slot 1's scene score a subsequent-answer challenger must keep.
pub(in crate::retrieval) const C2_SUBSEQUENT_RATIO: f32 = 0.5;

/// First later user turn in the same session, if any.
pub(in crate::retrieval) fn c2_next_user(ix: &Index, ti: usize) -> Option<usize> {
    let members = ix.session_turns.get(ix.turns[ti].session)?;
    let pos = members.iter().position(|&m| m == ti)?;
    members[pos + 1..]
        .iter()
        .copied()
        .find(|&m| ix.turns[m].speaker == "user")
}

/// Subsequent-answer seat. A mid-session user question at slot 1 often holds
/// the scene-setting inquiry ("by the way, …?"); the next user turn already
/// in the top 5 is the already-retrieved answer. When that follow-up is
/// declarative, keeps >= 50% of slot 1's scene score, and the head is not
/// the session opener, it takes slot 1. Top-5 membership never changes.
/// Session openers stay put (their follow-up is often more planning).
/// Skipped on temporal questions (those keep the existing timeline path),
/// event sequences and assistant-attribution (caller), and named-speaker
/// sessions. Distinct from primacy testimony (which promotes an earlier
/// first-half account) and from ReplySource (assistant at slot 1, previous
/// user). Runs after unique-group realloc, before RSB. A later
/// post-gatekeeper pass may run the same function again after RSB and
/// the tail reorder, so a next-user turn that only entered the top 5
/// via RSB can still take slot 1. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head, so a pre-RSB reseat also sticks; it
/// is not a Kemeny ballot (a tail vote cannot move T@1).
pub(in crate::retrieval) fn apply_c2_subsequent_answer_seat(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.temporal {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || !ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(cand) = c2_next_user(ix, top) else {
        return false;
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    let t = &ix.turns[cand];
    if t.speaker != "user" || t.is_question {
        return false;
    }
    if t_scene.get(cand).copied().unwrap_or(0.0) < C2_SUBSEQUENT_RATIO * sc_top {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}


/// True when the query cites a prior assistant utterance (collective chat,
/// "remind me", "you recommended", …) rather than only asking for a new
/// recommendation. The present-tense cue "you recommend" is excluded because
/// it also matches preference requests ("can you recommend...").
pub(in crate::retrieval) fn c2_retrospective_attribution(_text: &str) -> bool {
    // Assistant-citation phrase triggers are gone. Recorded speaker role
    // still distinguishes a turn; the query text no longer routes on a list.
    false
}

/// Preference source seat. ReplySource already yields an assistant at slot 1
/// to its previous user turn when that source is already in the top 5, but it
/// is skipped on assistant-attribution queries. Preference requests of the
/// form "can you recommend..." are tagged assistant_attribution by the
/// present-tense "you recommend" cue, so the user's preference-stating
/// source never takes slot 1. When the query is a preference intent and is
/// not a retrospective citation of a prior assistant utterance, apply the
/// same membership-preserving source move (position and role only). Skipped
/// on temporal questions, event sequences (caller), and named-speaker
/// sessions. Distinct from ReplySource (unchanged; still skipped on
/// assistant_attribution) and from subsequent-answer (#10; a user question
/// at slot 1). Runs after the other C2 seats, before RSB. Slot 1 is then
/// frozen by rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_preference_source_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
) -> bool {
    if list.len() < 2 || plan.intent.temporal || !plan.intent.preference {
        return false;
    }
    if c2_retrospective_attribution(query) {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "assistant" || c2_named_session(ix, top) {
        return false;
    }
    let members = &ix.session_turns[ix.turns[top].session];
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    if pos == 0 {
        return false;
    }
    let prev = members[pos - 1];
    if ix.turns[prev].speaker != "user" {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == prev) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, prev);
    true
}


/// Next-user elaboration seat. A mid-session declarative user turn at
/// slot 1 is often a list or scene overview; the next same-session user
/// turn already in the top 5, also declarative, with a scene score at
/// least that of the head, is the elaboration and takes slot 1. Top-5
/// membership never changes. Session openers stay put. Skipped on
/// temporal questions, preference questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. Sibling
/// of subsequent-answer (#10), which fires when the mid-session head is
/// a question and the follow-up holds >= 50% of its scene score. This
/// seat requires a declarative head and a follow-up that is at least as
/// on-scene as that head. Runs after subsequent-answer, before RSB.
/// Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_elaboration_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.temporal || c2_text_preference(query) {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(cand) = c2_next_user(ix, top) else {
        return false;
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    let t = &ix.turns[cand];
    if t.speaker != "user" || t.is_question {
        return false;
    }
    if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}
