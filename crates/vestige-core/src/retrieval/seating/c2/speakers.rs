//! C2 seating lane: `speakers`.

use crate::retrieval::*;



/// Assistant-attribution left-out-session opener T@10 entry, for
/// assistant-attribution questions only (outside the C2 skip, beside #43).
/// The most on-scene declarative session opener (t_scene > 0) in the pool
/// whose session has no official-k seat at all takes the official-k tail
/// seat (slots 6..k) of the least on-scene turn that is the only official-k
/// turn of its session, when the opener is at least as on-scene and that
/// turn's session is not earlier than the opener's (primacy). The question
/// points back at a past conversation; an unseated conversation's opening
/// statement is where that conversation is anchored. Top 5 unchanged; C2
/// turn seats do not rewrite the session list. No new constant: "at least
/// as on-scene" is the existing floor. Distinct from #30 (user openers on
/// non-attribution questions, victim the last non-unique assistant) and #37
/// (any pool user, reused 0.7 ratio, declarative victim); both are skipped
/// on assistant-attribution questions. Skipped on temporal (Gemini Step
/// 17), summary, WH-person, named dialogues, and event sequences (caller).
/// Last ranking change.
pub(in crate::retrieval) fn apply_c2_attribution_unseated_opener_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
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
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    let k_sessions: Vec<usize> = list.iter().map(|&t| ix.turns[t].session).collect();
    let mut cand: Option<usize> = None;
    for (ti, t) in ix.turns.iter().enumerate() {
        if sc(ti) <= 0.0 || t.is_question || c2_session_pos(ix, ti) != 0 || k_sessions.contains(&t.session) {
            continue;
        }
        let tsp = t.speaker.as_str();
        if (tsp != "user" && tsp != "assistant") || c2_named_session(ix, ti) {
            continue;
        }
        // Exact scene ties go to the earliest session (the original
        // attestation; a later session repeating the opener scores the same),
        // the #37 tie rule.
        if cand.is_none_or(|c| sc(ti) > sc(c) || (sc(ti) == sc(c) && ts_of(ti) < ts_of(c))) {
            cand = Some(ti);
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let mut vs: Option<usize> = None;
    for i in window..list.len() {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc(cand) || list.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1 {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    list[vs] = cand;
    true
}


/// True when the padded query contains the speaker's name as a whole token.
/// Speakers are the conversation's own role labels (no name list). User and
/// assistant roles never count.
pub(in crate::retrieval) fn c2_query_mentions_speaker(query: &str, speaker: &str) -> bool {
    if speaker.is_empty() || speaker == "user" || speaker == "assistant" {
        return false;
    }
    let padded = word_padded(query);
    let sp = speaker.to_ascii_lowercase();
    // word_padded keeps a possessive apostrophe ("Melanie's" -> " melanie's ").
    padded.contains(&format!(" {sp} "))
        || padded.contains(&format!(" {sp}'s "))
        || padded.contains(&format!(" {sp}’s "))
}

/// True when a mentioned speaker is conjoined with someone else ("X and Y").
/// Those questions are often answered by a third-party report, not by X's
/// next reply.
pub(in crate::retrieval) fn c2_speaker_conjoined(query: &str, speaker: &str) -> bool {
    if speaker.is_empty() {
        return false;
    }
    let padded = word_padded(query);
    let sp = speaker.to_ascii_lowercase();
    padded.contains(&format!(" {sp} and ")) || padded.contains(&format!(" and {sp} "))
}

/// Queried-speaker reply seat. In a named-speaker dialogue, an unmentioned
/// speaker's question at slot 1 is often the prompt; the immediately
/// following same-session turn, by a speaker the query names, already in
/// the top 5, is that speaker's reply and takes slot 1. Top-5 membership
/// never changes. Skipped when the query conjoins the named speaker with
/// someone else ("X and Y": the answer is often a third-party report),
/// on WH-person questions (existing `wh_slots`, so this does not compete
/// with Gemini Step 11), preference and temporal questions, summary
/// requests, event sequences and assistant-attribution (caller). Does
/// not fire on user/assistant chats. Runs after the same-session opener
/// seat, before RSB. Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_queried_speaker_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty()
        || head_sp == "user"
        || head_sp == "assistant"
        || !ix.turns[top].is_question
        || c2_query_mentions_speaker(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    let Some(&cand) = members.get(pos + 1) else {
        return false;
    };
    let csp = ix.turns[cand].speaker.as_str();
    if !c2_query_mentions_speaker(query, csp) || c2_speaker_conjoined(query, csp) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}


/// Queried-speaker previous seat. In a named-speaker dialogue, an
/// unmentioned speaker's declarative turn at slot 1 is often a follow-up
/// comment; the immediately preceding same-session turn, also
/// declarative, by a speaker the query names, already in the top 5, is
/// the statement and takes slot 1. Top-5 membership never changes.
/// Sibling of the queried-speaker reply seat (#15), which requires a
/// question head and the next turn. This seat requires both sides
/// declarative and looks backward. The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply. Does not
/// fire on user/assistant chats. Runs after the queried-speaker reply
/// seat, before RSB. Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_queried_speaker_prev_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty()
        || head_sp == "user"
        || head_sp == "assistant"
        || ix.turns[top].is_question
        || c2_query_mentions_speaker(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    if pos == 0 {
        return false;
    }
    let cand = members[pos - 1];
    let csp = ix.turns[cand].speaker.as_str();
    if ix.turns[cand].is_question
        || !c2_query_mentions_speaker(query, csp)
        || c2_speaker_conjoined(query, csp)
    {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}

/// Queried-speaker T@5 entry. In a named-speaker dialogue, a declarative
/// turn by a speaker the query names, already in the official k but
/// outside the top 5, swaps into the top 5 in place of the last
/// unmentioned named speaker. The victim stays in the official k, so
/// T@10 cannot drop by construction. A question by a mentioned speaker
/// is not pulled: that is often a prompt, not the answer (the 4:74
/// shape). The swap is refused when it would push any session out of
/// the first 5 unique-session slots, so S@5 cannot drop by construction.
/// Reuses `C2_XWITNESS_RATIO`. The same mention, conjunction, WH-person,
/// preference, temporal, and summary cuts apply as #15/#16. Does not
/// fire on user/assistant chats. Invoked twice: after the queried-speaker
/// previous / question-head-previous seats and before RSB (#17), and
/// again after `rank1_gatekeeper_keep_rsb_head` so a tail reorder cannot
/// undo the entry (#19), and a third time after the pool
/// mentioned-speaker T@10 entry (#32) so a mentioned declarative that
/// pass just seated into official k can still enter the top 5 (#33).
/// The post-#32 call is the last ranking change; it does not touch
/// `rsb_seat_behind` and adds no WH-slot ballot.
pub(in crate::retrieval) fn apply_c2_queried_speaker_t5_entry(
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
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        let csp = t.speaker.as_str();
        if t.is_question
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
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
        let sp = ix.turns[tid].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
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


/// Queried-speaker question-head previous seat. In a named-speaker
/// dialogue, an unmentioned speaker's question at slot 1 is often a
/// follow-up prompt; the immediately preceding same-session turn, by a
/// speaker the query names, already in the top 5, is that speaker's
/// prior statement (or their own earlier question) and takes slot 1.
/// Top-5 membership never changes. Literal backward twin of the
/// queried-speaker reply seat (#15), which requires a question head
/// and the next turn. This seat keeps the question head and looks
/// backward; the previous turn may be a question or a statement
/// (both shapes are live golds). The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply. Does not
/// fire on user/assistant chats. Runs after the queried-speaker
/// previous seat, before the T@5 entry and RSB. Slot 1 is then frozen
/// by rank1_gatekeeper_keep_rsb_head.
pub(in crate::retrieval) fn apply_c2_queried_speaker_qhead_prev_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty()
        || head_sp == "user"
        || head_sp == "assistant"
        || !ix.turns[top].is_question
        || c2_query_mentions_speaker(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    if pos == 0 {
        return false;
    }
    let cand = members[pos - 1];
    let csp = ix.turns[cand].speaker.as_str();
    if !c2_query_mentions_speaker(query, csp) || c2_speaker_conjoined(query, csp) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}


/// Same-session user T@5 entry. In a user/assistant transcript, a user
/// turn already in the official k but outside the top 5, from a session
/// already represented in the top 5, that is the session opener or is
/// later than a same-session turn already in the top 5, swaps into the
/// top 5 in place of the last assistant. The victim stays in the official
/// k, so T@10 cannot drop by construction. The swap is refused when it
/// would push any session out of the first 5 unique-session slots, so
/// S@5 cannot drop by construction. No new ratio: the same-session
/// opener-or-later cut is the structure. Skipped on preference, temporal,
/// WH-person, summary requests, event sequences and assistant-attribution
/// (caller), and named-speaker sessions. Distinct from the source-superset
/// and unique-group reallocs (no group test) and from the subsequent-answer
/// seat (this changes official-k membership, not slot 1). Runs after
/// `rank1_gatekeeper_keep_rsb_head` so the entry survives the tail reorder.
pub(in crate::retrieval) fn apply_c2_same_session_user_t5_entry(
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
        if t.speaker != "user" {
            continue;
        }
        if !top5_sessions.contains(&t.session) {
            continue;
        }
        let cand_pos = c2_session_pos(ix, cand);
        let opener = cand_pos == 0;
        let later = list[..window].iter().any(|&tid| {
            ix.turns[tid].session == t.session && c2_session_pos(ix, tid) < cand_pos
        });
        if !opener && !later {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
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


/// Cross-session user-opener T@5 entry. In a user/assistant transcript, a
/// user session opener already in the official k but outside the top 5,
/// from a session not yet represented in the top 5, swaps into the top 5
/// in place of the last assistant that is not the unique top-5
/// representative of its session and is not slot 1's immediately
/// following same-session turn. The last cut refuses to break the seated
/// continuation of the head (a user-assistant pair starting at slot 1).
/// The victim stays in the official k, so T@10 cannot drop. The swap is
/// refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. No new ratio: the
/// cross-session opener cut, plus not evicting the head's next turn, is
/// the structure. Skipped on preference, temporal, WH-person, summary
/// requests, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Distinct from the same-session user T@5 entry
/// (#20: session already in the top 5) and from the source-superset and
/// unique-group reallocs (no group test). Runs after
/// `rank1_gatekeeper_keep_rsb_head`, after #20, as the last ranking change.
pub(in crate::retrieval) fn apply_c2_cross_session_user_t5_entry(
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
    let head_sess = ix.turns[top].session;
    let head_pos = c2_session_pos(ix, top);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user" {
            continue;
        }
        if top5_sessions.contains(&t.session) {
            continue;
        }
        if c2_session_pos(ix, cand) != 0 {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
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
        if ix.turns[tid].speaker != "assistant" {
            continue;
        }
        let sess = ix.turns[tid].session;
        if list[..window].iter().filter(|&&u| ix.turns[u].session == sess).count() < 2 {
            continue;
        }
        if sess == head_sess && c2_session_pos(ix, tid) == head_pos + 1 {
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

pub(in crate::retrieval) fn apply_c2_queried_speaker_first_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty()
        || head_sp == "user"
        || head_sp == "assistant"
        || c2_query_mentions_speaker(query, head_sp)
        || c2_speaker_conjoined(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in 1..window {
        let cand = list[i];
        let csp = ix.turns[cand].speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if c2_first_same_speaker(ix, cand) != Some(cand) {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(slot) = cand_slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


/// Queried-speaker first-in-session T@5 entry. In a named-speaker
/// dialogue, a turn by a speaker the query names that is that
/// speaker's first utterance in its session, already in official k
/// but outside the top 5, swaps into the top 5 in place of the last
/// unmentioned named speaker. The victim stays in official k, so
/// T@10 cannot drop by construction. A swap that would push any
/// session out of the first 5 unique-session slots is refused, so
/// S@5 cannot drop. Distinct from #17/#19: the cut is first-in-session,
/// not a scene floor on every candidate. Declaratives have no new
/// ratio; first-in-session is enough. Questions reuse
/// `C2_XWITNESS_RATIO`, the same floor #17 uses, because a
/// first-in-session question at r=0 can evict an unmentioned gold
/// (the 4:74 / 0:106 shapes). A later turn by the same speaker is
/// not pulled. The same mention, conjunction, WH-person, preference,
/// temporal, and summary cuts apply as #15–#19. Does not fire on
/// user/assistant chats. Invoked after the gatekeeper and after
/// #19–#22. A later #22 pass may still run after this entry, so a
/// newly admitted first-in-session turn can take slot 1.
pub(in crate::retrieval) fn apply_c2_queried_speaker_first_t5_entry(
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
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        let csp = t.speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        let sc = t_scene.get(cand).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if c2_first_same_speaker(ix, cand) != Some(cand) {
            continue;
        }
        if t.is_question && sc < C2_XWITNESS_RATIO * sc_top {
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
        let sp = ix.turns[tid].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
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
