//! Seating pass: `primacy`.

use crate::retrieval::*;


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
            let window = list.len().min(GIVEN_NEW_WINDOW);
            let Some(at) = list[1..window].iter().position(|&t| t == z).map(|i| i + 1) else { return false };
            list.remove(at);
            list.insert(0, z);
            return true;
        }
    }
    false
}
