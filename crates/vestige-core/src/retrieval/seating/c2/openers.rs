//! C2 seating lane: `openers`.

use crate::retrieval::*;


/// Cross-session opener seat. When slot 1 is a mid-session user turn, a
/// user session-opener from a different user/assistant session already in
/// the top 5, that keeps slot 1's digit form and holds >= 70% of slot 1's
/// scene score, takes slot 1. The conversation start is the lead when the
/// sitting head has already drifted mid-topic. Top-5 membership never
/// changes. Skipped on summary requests (existing `summary_hold` shape:
/// those questions want the mid-conversation content, not an opener),
/// preference and temporal questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. Distinct
/// from cross-witness (any earlier session position, requires a query
/// group) and from SessionOpenerLift (requires the opener to cover every
/// query cue the head matched). Reuses `C2_XWITNESS_RATIO`. Runs after
/// the elaboration seat, before RSB. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_cross_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.session == ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Same-session opener seat. A mid-session declarative user turn at
/// slot 1 is often a later aside that still mentions the fact; the same
/// session's user opener already in the top 5, that keeps slot 1's
/// digit form and holds >= 70% of its scene score, is the conversation
/// start and takes slot 1. Top-5 membership never changes. Skipped on
/// summary requests (existing `summary_hold` shape), preference and
/// temporal questions, event sequences and assistant-attribution
/// (caller), and named-speaker sessions. Sibling of the cross-session
/// opener seat (#13), which requires a different session and allows a
/// question head. This seat requires the opener to be in the same
/// session and both the head and the opener to be declarative: a
/// question head already asking the query should stay, and a
/// question opener is still asking, not stating the fact. Reuses
/// `C2_XWITNESS_RATIO`. Runs after
/// the cross-session opener seat, before RSB. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_same_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
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
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.is_question
            || t.session != ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Cross-session opener-pair seat. A user session-opener at slot 1 is
/// often a sibling conversation start that still shares topic words; a
/// user opener from a different user/assistant session already in the
/// top 5, whose immediately following assistant reply is also already
/// in the top 5, and that is at least as on-scene, is a retrieved
/// conversation start and takes slot 1. Top-5 membership never changes.
/// The pair cut is the structure: an opener whose reply was not
/// retrieved with it is not promoted, which is what would otherwise
/// unseat a sitting gold opener that a higher-scoring lone opener
/// outranks. Distinct from the cross-session opener seat (#13), which
/// requires a mid-session head and does not require the candidate's
/// reply to be in the top 5, and from the same-session opener seat
/// (#14). Skipped on summary requests (existing `summary_hold` shape),
/// preference and temporal questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant: the scene floor is "at least as on-scene," the same
/// comparison #12 uses. Invoked after the gatekeeper, after #26. A
/// later post-gatekeeper C2 seat may still run after this pass.
pub(in crate::retrieval) fn apply_c2_cross_opener_pair_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.session == ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        let Some(members) = ix.session_turns.get(t.session) else {
            continue;
        };
        let Some(&reply) = members.get(1) else {
            continue;
        };
        if ix.turns[reply].speaker != "assistant" {
            continue;
        }
        if !list[..window].contains(&reply) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Answered-opener declarative seat. A user question opener at slot 1
/// whose immediately following assistant reply is already in the top 5
/// is a retrieved Q-A pair; a declarative user opener from a different
/// user/assistant session already in the top 5, that is at least as
/// on-scene, is a conversation start that states a fact and takes slot 1.
/// Top-5 membership never changes. The answered-head + declarative-cand
/// cut is the structure: a sitting gold question opener whose reply is
/// not in the top 5 is left alone, and a question challenger is not
/// promoted, which is what would otherwise unseat a sitting gold opener.
/// Distinct from the cross-session opener-pair seat (#27), which
/// requires the *candidate's* reply to be in the top 5 and does not
/// require a question head or a declarative candidate, from #13
/// (mid-session head), and from #14 (same session). Skipped on summary
/// requests (existing `summary_hold` shape), preference and temporal
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. No new constant: the scene floor is "at
/// least as on-scene," the same comparison #12 and #27 use. Invoked
/// after the gatekeeper, after #27. A later post-gatekeeper C2 seat
/// may still run after this pass.
pub(in crate::retrieval) fn apply_c2_answered_opener_decl_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || !ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(&reply) = members.get(1) else {
        return false;
    };
    if ix.turns[reply].speaker != "assistant" || !list[..window].contains(&reply) {
        return false;
    }
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.is_question
            || t.session == ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Solo earlier-opener seat. A lone user opener at slot 1 — no other
/// turn from that session is already in the top 5 — yields to an
/// earlier user opener from a different user/assistant session already
/// in the top 5, when that earlier opener is at least as on-scene and
/// both sessions have a known timestamp. Top-5 membership never changes.
/// The solo + earlier cut is the structure: a sitting gold opener that
/// arrived with another turn from its session is left alone, and a
/// same-date or later challenger is not promoted, which is what would
/// otherwise unseat a sitting gold opener. Distinct from #27 (candidate
/// reply already in the top 5; no solo or earlier requirement), from
/// #28 (answered question head and a declarative candidate), from #13
/// (mid-session head), and from #14 (same session). Skipped on summary
/// requests (existing `summary_hold` shape), preference and temporal
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. No new constant: the scene floor is "at
/// least as on-scene," the same comparison #12, #27, and #28 use.
/// Invoked after the gatekeeper, after #28. A later post-gatekeeper
/// C2 seat may still run after this pass.
pub(in crate::retrieval) fn apply_c2_solo_earlier_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let ts_top = ix.session_ts.get(ix.turns[top].session).copied().unwrap_or(0);
    if ts_top <= 0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top_sess = ix.turns[top].session;
    if list[1..window].iter().any(|&t| ix.turns[t].session == top_sess) {
        return false;
    }
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.session == top_sess
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        let ts_cand = ix.session_ts.get(t.session).copied().unwrap_or(0);
        if ts_cand <= 0 || ts_cand >= ts_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Pool opener T@10 entry. A user session opener that is in the
/// scene-ranked retrieval pool but not yet in the official k takes the
/// official-k seat of the last assistant in the official-k tail (slots
/// 6..k) that is not the unique official-k representative of its
/// session, is not slot 1's immediately following same-session turn,
/// and whose session still has a user turn in the official k, when that
/// opener is at least as on-scene as the evicted assistant. The top 5
/// is unchanged, so T@5 and All@5 cannot drop. The replacement is
/// refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. Distinct from #21 (the
/// candidate is already in the official k, and the victim is pulled
/// into the top 5) and from pull-outsider / drop-official-last (that
/// class evicts official last and can change the top 5). No new
/// constant: the scene floor is "at least as on-scene," the same
/// comparison #12 and #21 use. Skipped on summary requests (existing
/// `summary_hold` shape), preference, temporal, and WH-person questions,
/// event sequences and assistant-attribution (caller), and named-speaker
/// sessions. Invoked after the gatekeeper, after #29. A later
/// post-gatekeeper C2 seat may still run after this pass.
pub(in crate::retrieval) fn apply_c2_pool_opener_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if list.len() <= window {
        return false;
    }
    let k = list.len();
    let head_sess = ix.turns[top].session;
    let head_pos = c2_session_pos(ix, top);
    let mut vs = None;
    for i in window..k {
        let tid = list[i];
        if ix.turns[tid].speaker != "assistant" || c2_named_session(ix, tid) {
            continue;
        }
        let sess = ix.turns[tid].session;
        if list.iter().filter(|&&u| ix.turns[u].session == sess).count() < 2 {
            continue;
        }
        if sess == head_sess && c2_session_pos(ix, tid) == head_pos + 1 {
            continue;
        }
        if !list
            .iter()
            .any(|&u| u != tid && ix.turns[u].speaker == "user" && ix.turns[u].session == sess)
        {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    let sc_v = t_scene.get(list[vs]).copied().unwrap_or(0.0);
    if sc_v <= 0.0 {
        return false;
    }
    let turn_ts: Vec<i64> = ix
        .turns
        .iter()
        .map(|t| ix.session_ts.get(t.session).copied().unwrap_or(0))
        .collect();
    for ti in order(t_scene, &turn_ts) {
        if list.contains(&ti) || t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.speaker != "user" || c2_named_session(ix, ti) || c2_session_pos(ix, ti) != 0 {
            continue;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) < sc_v {
            continue;
        }
        if !c2_s5_replace_preserved(ix, list, vs, ti) {
            continue;
        }
        list[vs] = ti;
        return true;
    }
    false
}


/// Same-session opener T@5 entry. A user session opener already in the
/// official k but outside the top 5, from a session already represented
/// in the top 5 by an assistant, swaps into the top 5 in place of the
/// last same-session assistant in the top 5. The victim stays in the
/// official k, so T@10 cannot drop. The swap is refused when it would
/// push any session out of the first 5 unique-session slots, so S@5
/// cannot drop. Distinct from #20, which also promotes a later
/// non-opener user from a top-5 session and evicts the last assistant
/// of any session: that later-user clause is what would unseat a
/// sitting gold assistant (the ac031881 shape). The opener-only,
/// same-session-assistant victim cut is the structure. Skipped on
/// summary requests (existing `summary_hold` shape), preference,
/// temporal, and WH-person questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant. Invoked after the gatekeeper, after #30, as the last
/// ranking change, so it can see openers that the pool T@10 entry just
/// seated into official k. A later post-gatekeeper C2 seat may still
/// run after this pass.
pub(in crate::retrieval) fn apply_c2_same_session_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut top5_sessions = Vec::new();
    for &tid in &list[..window] {
        let s = ix.turns[tid].session;
        if !top5_sessions.contains(&s) {
            top5_sessions.push(s);
        }
    }
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
            || !top5_sessions.contains(&t.session)
            || t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0
        {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let cand_sess = ix.turns[list[ci]].session;
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker != "assistant" || ix.turns[tid].session != cand_sess {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Late opener T@5 entry. A user session opener already in the official k
/// but outside the top 5, from a session already represented in the top 5
/// (by any turn), swaps into the top 5 in place of the last assistant in
/// the top 5, of any session. Distinct from #31, whose victim must be a
/// same-session assistant (it refuses when the opener's session sits in
/// the top 5 only through a user turn), and from #20, which also seats a
/// later non-opener user and runs before #30 seats pool openers into
/// official k. The opener-only cut is what makes the #20 later-user loss
/// shape (a mid-session user evicting a seated gold assistant)
/// impossible. The victim stays in the official k, so T@10 cannot drop.
/// The swap is refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. Skipped on summary requests,
/// preference, temporal, and WH-person questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant. Invoked after the gatekeeper, after #33, as the last ranking
/// change, so it sees openers that the pool T@10 entry (#30) seated.
pub(in crate::retrieval) fn apply_c2_late_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5_sessions: Vec<usize> = list[..window].iter().map(|&t| ix.turns[t].session).collect();
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
            || !top5_sessions.contains(&t.session)
            || t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0
        {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker == "assistant" {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Head-session opener T@5 entry. The user opener of the session that holds
/// slot 1, already in the official k but outside the top 5, swaps into the
/// top 5 in place of the least on-scene user turn in slots 2..5 that
/// belongs to a different session, when the opener is at least as
/// on-scene as that turn. The slot-1 session is the strongest evidence
/// session; its opener frames it. Distinct from #31 and #34 (assistant
/// victims) and from evicting the last top-5 user (that victim choice
/// unseats sitting golds such as 8979f9ec): the victim here is the
/// weakest foreign-session user, and only the head session's opener
/// enters. The victim stays in the official k, so T@10 cannot drop. The
/// swap is refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. No new constant: the scene
/// floor is the same "at least as on-scene" comparison #12 and #30 use.
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #34, as
/// the last ranking change.
pub(in crate::retrieval) fn apply_c2_head_session_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let head_sess = ix.turns[top].session;
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker == "user"
            && t.session == head_sess
            && c2_session_pos(ix, cand) == 0
            && t_scene.get(cand).copied().unwrap_or(0.0) > 0.0
        {
            cand_slot = Some(i);
            break;
        }
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let sc_c = t_scene.get(list[ci]).copied().unwrap_or(0.0);
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        if ix.turns[tid].speaker != "user" || ix.turns[tid].session == head_sess {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}
