//! Seating pass: `cues`.

use crate::retrieval::*;

// =========================================================================
// RECALL-CUE SLOT-1 RULES (position, session, role field, token identity only)
// =========================================================================

/// BoundaryPrimacy window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed before any run; the top-5 membership never changes.
pub(in crate::retrieval) const PRIMACY_WINDOW: usize = 5;

/// Query cues: distinct kernel tokens of the query that occur somewhere in the haystack, minus
/// any token that occurs in a turn's role/speaker field (structural metadata, no word list).
pub(in crate::retrieval) fn query_cues(ix: &Index, text: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = cue_tokens(text)
        .into_iter()
        .filter_map(|w| ix.cue_vocab.get(&w).copied())
        .filter(|id| !ix.role_cues.contains(id))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// `cues(x)`: the query cues present on turn `ti` (both sides sorted, so a merge walk).
pub(in crate::retrieval) fn turn_query_cues(ix: &Index, q_cues: &[u32], ti: usize) -> Vec<u32> {
    let t = &ix.turn_cues[ti];
    let (mut i, mut j, mut out) = (0usize, 0usize, Vec::new());
    while i < t.len() && j < q_cues.len() {
        match t[i].cmp(&q_cues[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(t[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// `sup ⊇ sub` for sorted, deduplicated id lists.
pub(in crate::retrieval) fn covers_cues(sup: &[u32], sub: &[u32]) -> bool {
    let mut i = 0usize;
    for x in sub {
        while i < sup.len() && sup[i] < *x {
            i += 1;
        }
        if i >= sup.len() || sup[i] != *x {
            return false;
        }
        i += 1;
    }
    true
}

/// Position of turn `ti` inside its own session.
pub(in crate::retrieval) fn turn_pos(ix: &Index, ti: usize) -> Option<usize> {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti)
}

/// Structural user-preference / personal-constraint turn check (no PREF_MARKERS dictionary list).
pub(in crate::retrieval) fn is_structural_personal_turn(ix: &Index, ti: usize) -> bool {
    let tr = &ix.turns[ti];
    tr.field == 0
        && (tr.is_standing_policy
            || tr.is_personal_constraint
            || (tr.speaker == "user" && turn_pos(ix, ti).map_or(false, |pos| pos <= 4)))
}

pub(in crate::retrieval) struct RecallCueGuard<'a> {
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) it: &'a QueryIntent,
    pub(in crate::retrieval) t_scene: &'a [f32],
}

/// Score-dominance and state-polarity guard for `boundary_primacy`, `given_new`,
/// and `session_opener_lift_with_plan`. When `guard` is `None` (direct unit tests),
/// preserves baseline structural behavior.
pub(in crate::retrieval) fn recall_cue_promotion_allowed(
    ix: &Index,
    guard: Option<&RecallCueGuard<'_>>,
    top: usize,
    cand: usize,
) -> bool {
    let Some(g) = guard else {
        return true;
    };
    let t_top = &ix.turns[top];
    let t_cand = &ix.turns[cand];
    // Never demote a user turn to an assistant turn unless the query asks what the assistant said.
    if !g.it.assistant_attribution && t_top.speaker == "user" && t_cand.speaker == "assistant" {
        return false;
    }
    // Never promote a superseded turn over a non-superseded turn on non-sequence queries.
    if !g.it.is_event_sequence && turn_is_superseded(ix, cand) && !turn_is_superseded(ix, top) {
        return false;
    }
    let sc_top = g.t_scene.get(top).copied().unwrap_or(0.0);
    let sc_cand = g.t_scene.get(cand).copied().unwrap_or(0.0);
    // On assistant-attribution queries, never demote a clearly dominant assistant turn (>1.25x) to user.
    if g.it.assistant_attribution
        && t_top.speaker == "assistant"
        && t_cand.speaker == "user"
        && sc_top > 1.25 * sc_cand
    {
        return false;
    }
    let cand_is_opener = turn_pos(ix, cand) == Some(0);
    // Never promote a mid-session interrogative user turn over a stronger declarative user turn.
    if !cand_is_opener
        && t_cand.speaker == "user"
        && t_top.speaker == "user"
        && t_cand.is_question
        && !t_top.is_question
        && sc_top > 1.05 * sc_cand
    {
        return false;
    }
    // On current-state queries, never demote a stronger same-session later turn to a non-opener earlier turn.
    if g.it.current_state
        && !cand_is_opener
        && t_cand.session == t_top.session
        && t_cand.speaker == "user"
        && t_top.speaker == "user"
        && sc_top > 1.05 * sc_cand
    {
        return false;
    }
    // Score-dominance guard (>1.45x): block overriding a clearly dominant `top` unless
    // `cand` is a richer candidate against an interrogative `top` (`cand_ic > top_ic`) or
    // a high-information session opener (`cand_ic >= 1.5 * top_ic` or `top_ic < 15.0`).
    if sc_top > 1.45 * sc_cand {
        if g.it.assistant_attribution && t_top.speaker == "assistant" {
            return false;
        }
        let (cand_ic, _, _) = turn_novel_slot_profile(ix, g.plan, cand);
        let (top_ic, _, _) = turn_novel_slot_profile(ix, g.plan, top);
        if t_top.is_question {
            if cand_ic <= top_ic {
                return false;
            }
        } else if !cand_is_opener || (top_ic >= 15.0 && cand_ic < 1.5 * top_ic) {
            return false;
        }
    }
    true
}
