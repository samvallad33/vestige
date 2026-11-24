//! Seating pass: `primacy`.

use crate::retrieval::*;


/// Closed-class same-session state revision detector:
/// Returns true when the later user turn (`field == 0`) explicitly revises an earlier
/// statement in the same session (via closed-class correction/update markers or a
/// conflicting numeric value), so primacy/given-new never rolls current state backward.
// hardcode-scan: closed-class (discourse revision, cancellation, and state-update markers)
fn has_same_session_state_revision(ix: &Index, earlier: usize, later: usize) -> bool {
    let lt = &ix.turns[later];
    if lt.field != 0 {
        return false;
    }
    let low = lt.body_for_headers.to_ascii_lowercase();
    // hardcode-scan: closed-class (discourse revision, cancellation, and state-update markers)
    let has_rev_cue = [
        "actually",
        "instead",
        "changed",
        "switched",
        "updated",
        "no longer",
        "anymore",
        "cancelled",
        "canceled",
        "rescheduled",
        "postponed",
        "replaced",
        "scratch that",
        "never mind",
    ]
    .iter()
    .any(|p| low.contains(p));
    if has_rev_cue {
        return true;
    }
    let ld = digit_token_set(&lt.body_for_headers);
    if !ld.is_empty() {
        let ed = digit_token_set(&ix.turns[earlier].body_for_headers);
        if !ed.is_empty() && ld != ed {
            return true;
        }
    }
    false
}

pub(in crate::retrieval) fn boundary_primacy_with_plan(
    ix: &Index,
    q_cues: &[u32],
    list: &mut Vec<usize>,
    guard: Option<&RecallCueGuard<'_>>,
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    if turn_pos(ix, top) == Some(0) {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let sess = ix.turns[top].session;
    let Some(&opener) = ix.session_turns[sess].first() else { return false };
    let window = list.len().min(PRIMACY_WINDOW);
    let Some(at) = list[1..window].iter().position(|&t| t == opener).map(|i| i + 1) else { return false };
    if !covers_cues(&turn_query_cues(ix, q_cues, opener), &top_cues) {
        return false;
    }
    if !recall_cue_promotion_allowed(ix, guard, top, opener) {
        return false;
    }
    if guard.is_some_and(|g| g.it.current_state)
        && has_same_session_state_revision(ix, opener, top)
    {
        return false;
    }
    list.remove(at);
    list.insert(0, opener);
    true
}

/// GivenNew window: the earliest covering turn must already sit in the kernel's top 10.
/// Fixed before any run.
pub(in crate::retrieval) const GIVEN_NEW_WINDOW: usize = 10;
/// GivenNew variant, fixed before the run. `false` = card 2a (any role). `true` = card 2b: fires
/// only when rank 1's role field is the chat schema value `user` or `assistant` (structural field).
pub(in crate::retrieval) const GIVEN_NEW_CHAT_ROLES_ONLY: bool = true;

pub(in crate::retrieval) fn given_new_with_plan(
    ix: &Index,
    q_cues: &[u32],
    list: &mut Vec<usize>,
    guard: Option<&RecallCueGuard<'_>>,
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    if GIVEN_NEW_CHAT_ROLES_ONLY {
        let sp = ix.turns[top].speaker.as_str();
        if sp != "user" && sp != "assistant" {
            return false;
        }
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    for &z in &ix.session_turns[ix.turns[top].session] {
        if z == top {
            return false;
        }
        if covers_cues(&turn_query_cues(ix, q_cues, z), &top_cues) {
            if !recall_cue_promotion_allowed(ix, guard, top, z) {
                continue;
            }
            if guard.is_some_and(|g| g.it.current_state)
                && has_same_session_state_revision(ix, z, top)
            {
                continue;
            }
            let window = list.len().min(GIVEN_NEW_WINDOW);
            let Some(at) = list[1..window].iter().position(|&t| t == z).map(|i| i + 1) else { return false };
            list.remove(at);
            list.insert(0, z);
            return true;
        }
    }
    false
}
