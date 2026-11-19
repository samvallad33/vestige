//! Seating pass: `opener_and_hearsay`.

use crate::retrieval::*;


/// SessionOpenerLift window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed at 5 before any run; the top-5 membership never changes.
pub(in crate::retrieval) const SESSION_OPENER_LIFT_WINDOW: usize = 5;

pub(in crate::retrieval) fn session_opener_lift_with_plan(
    ix: &Index,
    q_cues: &[u32],
    list: &mut Vec<usize>,
    guard: Option<&RecallCueGuard<'_>>,
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    let sp = ix.turns[top].speaker.as_str();
    if sp != "user" && sp != "assistant" {
        return false;
    }
    if turn_pos(ix, top) == Some(0) {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let window = list.len().min(SESSION_OPENER_LIFT_WINDOW);
    for i in 1..window {
        let cand = list[i];
        if turn_pos(ix, cand) != Some(0) {
            continue;
        }
        if let Some(g) = guard {
            if !rung2_opener_lift_allowed(ix, g.plan, top, cand) {
                continue;
            }
        }
        if !recall_cue_promotion_allowed(ix, guard, top, cand) {
            continue;
        }
        if !covers_cues(&turn_query_cues(ix, q_cues, cand), &top_cues) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Hearsay window: the covering user turn must already sit inside the top 5 the kernel
/// selected. Fixed at 5 before any run; the top-5 membership never changes.
pub(in crate::retrieval) const HEARSAY_WINDOW: usize = 5;

pub(in crate::retrieval) fn hearsay_with_plan(
    ix: &Index,
    q_cues: &[u32],
    list: &mut Vec<usize>,
    guard: Option<&RecallCueGuard<'_>>,
) -> bool {
    if list.len() < 2 {
        return false;
    }
    if let Some(g) = guard {
        if g.it.assistant_attribution {
            return false;
        }
    }
    let top = list[0];
    if ix.turns[top].speaker.as_str() != "assistant" {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let window = list.len().min(HEARSAY_WINDOW);
    for i in 1..window {
        let cand = list[i];
        if ix.turns[cand].speaker.as_str() != "user" {
            continue;
        }
        if let Some(g) = guard {
            let sc_top = g.t_scene.get(top).copied().unwrap_or(0.0);
            let sc_cand = g.t_scene.get(cand).copied().unwrap_or(0.0);
            if sc_top > 2.0 * sc_cand {
                continue;
            }
            if !g.it.is_event_sequence && turn_is_superseded(ix, cand) && !turn_is_superseded(ix, top) {
                continue;
            }
        }
        if !covers_cues(&turn_query_cues(ix, q_cues, cand), &top_cues) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}
