//! C2 seating lane: `session_mates`.

use crate::retrieval::*;



/// Session substitution T@5 entry. The most on-scene user turn already in
/// the official k, outside the top 5, from a session not represented in the
/// top 5, swaps with the least on-scene user turn in slots 2..5 when that
/// turn is a declarative and holds at most `C2_XWITNESS_RATIO` of the
/// candidate's scene (the cross-witness dominance ratio, reused, not
/// re-tuned). The swap trades the weakest top-5 session for a clearly
/// stronger session the window left out, so the number of distinct
/// sessions in the top 5 does not fall when the victim was its session's
/// only seat. The declarative cut is load-bearing: a weakest user
/// question in the top 5 is often the sitting gold (the user asked about
/// the thing), as on BEAM 9:contradiction_resolution:0. Assistants are
/// never the victim. The victim stays in the official k, so T@10 cannot
/// drop, and C2 turn seats do not rewrite the session list. Distinct from
/// #21 (opener entry, assistant victim), #34/#35 (session already in the
/// top 5), and from evicting the last top-5 user (sitting-gold losses).
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #35, and
/// again after #37 (#38) so a pool user that #37 seats into official k can
/// still enter the top 5.
pub(in crate::retrieval) fn apply_c2_session_substitution_t5_entry(
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
    let mut cand: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let t = &ix.turns[tid];
        if t.speaker != "user" || c2_named_session(ix, tid) || top5_sessions.contains(&t.session) {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if cand.is_none_or(|(_, best)| sc > best) {
            cand = Some((i, sc));
        }
    }
    let Some((ci, sc_c)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        if ix.turns[tid].speaker != "user" {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if ix.turns[list[vs]].is_question {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Pool session substitution T@10 entry. The most on-scene user turn in the
/// scene-ranked pool (exact ties to the earliest session, the original
/// attestation) from a session with no turn in the official k takes
/// the official-k seat of the least on-scene turn in the official-k tail
/// (slots 6..k) when that turn is a declarative and holds at most
/// `C2_XWITNESS_RATIO` of the candidate's scene (reused, not re-tuned). The
/// T@10 analogue of #36: a clearly stronger session the official k left out
/// replaces the weakest tail seat. The top 5 is unchanged, so T@1, T@5 and
/// All@5 cannot drop; C2 turn seats do not rewrite the session list. The
/// declarative cut is the #36 cut (a weakest question is often a sitting
/// gold). Distinct from #30 (openers only, non-unique tail assistant
/// victim, "at least as on-scene" floor) and #32 (named speakers).
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #36, as the
/// last ranking change.
pub(in crate::retrieval) fn apply_c2_pool_session_substitution_t10_entry(
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
    if list.len() <= window {
        return false;
    }
    let k_sessions: Vec<usize> = list.iter().map(|&t| ix.turns[t].session).collect();
    // Most on-scene eligible pool user. Exact scene ties go to the earliest
    // session (the original attestation; a later session repeating the
    // same turn scores identically), then the lower turn index.
    let mut cand: Option<(usize, f32, i64)> = None;
    for (ti, t) in ix.turns.iter().enumerate() {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        if sc <= 0.0 || list.contains(&ti) {
            continue;
        }
        if t.speaker != "user" || c2_named_session(ix, ti) || k_sessions.contains(&t.session) {
            continue;
        }
        let ts = ix.session_ts.get(t.session).copied().unwrap_or(0);
        if cand.is_none_or(|(_, bs, bt)| sc > bs || (sc == bs && ts < bt)) {
            cand = Some((ti, sc, ts));
        }
    }
    let Some((cand, sc_c, _)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if ix.turns[list[vs]].is_question {
        return false;
    }
    list[vs] = cand;
    true
}


/// Mentioned-declarative session-mate T@5 entry. In a named-speaker
/// dialogue, the most on-scene declarative turn by a speaker the query
/// names (not conjoined with another name), already in the official k but
/// outside the top 5, from a session already represented in the top 5,
/// swaps with the least on-scene turn in slots 2..5 when that turn is a
/// declarative, holds at most `C2_XWITNESS_RATIO` of the candidate's scene
/// (reused, not re-tuned), and comes from a session no earlier than the
/// candidate's. The session cut is primacy: an earlier witness is never
/// evicted for a later or same-weight restatement (without it, LoCoMo
/// 4:57 loses a sitting earlier-session gold; unofficial All@5). The
/// declarative cuts follow #17 (a mentioned question is often a prompt)
/// and #36 (a weakest question is often a sitting gold). Distinct from
/// #17/#19/#33 (victim is the last unmentioned named speaker; any
/// mentioned declarative), #23 (first-in-session), and #36 (user/assistant
/// chats, session outside the top 5). The victim stays in the official k,
/// so T@10 cannot drop; C2 turn seats do not rewrite the session list.
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller). Does not
/// fire on user/assistant chats. Invoked after the gatekeeper, after #38,
/// as the last ranking change.
pub(in crate::retrieval) fn apply_c2_mentioned_session_mate_t5_entry(
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
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5_sessions: Vec<usize> = list[..window].iter().map(|&t| ix.turns[t].session).collect();
    let mut cand: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let t = &ix.turns[tid];
        let sp = t.speaker.as_str();
        if t.is_question
            || sp.is_empty()
            || sp == "user"
            || sp == "assistant"
            || !c2_query_mentions_speaker(query, sp)
            || c2_speaker_conjoined(query, sp)
            || !top5_sessions.contains(&t.session)
        {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if cand.is_none_or(|(_, best)| sc > best) {
            cand = Some((i, sc));
        }
    }
    let Some((ci, sc_c)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    let victim = list[vs];
    if ix.turns[victim].is_question {
        return false;
    }
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(victim) < ts_of(list[ci]) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Episode-run edge T@10 entry. Two or more official-k turns that sit
/// back to back in one session form an episode the retriever already
/// found; the pool turn just before or just after that run (same session,
/// t_scene > 0, not in the official k, a declarative, and, in a
/// named-speaker dialogue, by a speaker the query names and does not
/// conjoin) is the episode's unread edge. The most on-scene such edge
/// takes the official-k tail seat (slots 6..k) of the least on-scene
/// isolated turn (the only official-k turn of its session), when the
/// edge is at least as on-scene as that turn and the turn's session is
/// not earlier than the edge's (primacy cut: an earlier lone witness is
/// never evicted for a later episode). Multi-turn evidence, not a
/// per-turn feature: neither the candidate nor the victim is chosen by
/// its own text. Top 5 unchanged, so T@1 / T@5 / All@5 cannot drop; C2
/// turn seats do not rewrite the session list. No new constant: "at
/// least as on-scene" is the existing floor and a run is two adjacent
/// turns. Distinct from #30 / #32 / #37 (pool entry by opener, mentioned
/// speaker, or left-out session; none uses k-run adjacency) and from the
/// head-adjacency seats (#2 / #10 / #12 / #15 / #16 move turns next to slot
/// 1 into the top 5). Skipped on summary requests, preference, temporal,
/// and WH-person questions; event sequences and assistant-attribution
/// are skipped by the caller. Invoked after the gatekeeper, after #39,
/// as the last ranking change.
pub(in crate::retrieval) fn apply_c2_episode_run_edge_t10_entry(
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
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let mut by_sess: Vec<(usize, Vec<usize>)> = Vec::new();
    for &t in list.iter() {
        let s = ix.turns[t].session;
        let p = c2_session_pos(ix, t);
        match by_sess.iter_mut().find(|(bs, _)| *bs == s) {
            Some((_, ps)) => ps.push(p),
            None => by_sess.push((s, vec![p])),
        }
    }
    let eligible = |t: usize| -> bool {
        if list.contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
            return false;
        }
        let sp = ix.turns[t].speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" {
            return c2_query_mentions_speaker(query, sp) && !c2_speaker_conjoined(query, sp);
        }
        true
    };
    let mut cand: Option<usize> = None;
    for (s, ps) in by_sess.iter_mut() {
        ps.sort_unstable();
        let turns = &ix.session_turns[*s];
        let mut start = 0;
        while start < ps.len() {
            let mut end = start;
            while end + 1 < ps.len() && ps[end + 1] == ps[end] + 1 {
                end += 1;
            }
            if end > start {
                let mut edges: Vec<usize> = Vec::new();
                if ps[start] > 0 {
                    edges.push(turns[ps[start] - 1]);
                }
                if ps[end] + 1 < turns.len() {
                    edges.push(turns[ps[end] + 1]);
                }
                for t in edges {
                    if !eligible(t) {
                        continue;
                    }
                    if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                        cand = Some(t);
                    }
                }
            }
            start = end + 1;
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in window..list.len() {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c || list.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1 {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    list[vs] = cand;
    true
}


/// Episode-run edge T@5 entry. The top-5 analogue of #40: two or more
/// top-5 turns back to back in one session form an episode; the turn just
/// before or just after that run (same session, t_scene > 0, outside the
/// top 5, a declarative, and, in a named-speaker dialogue, by a speaker
/// the query names and does not conjoin) is its unread edge. The most
/// on-scene edge takes the slot (2..5) of the least on-scene top-5 turn
/// that is the only top-5 turn of its session and is not a session
/// opener, when the edge is at least as on-scene as that turn and the
/// turn's session is not earlier than the edge's (primacy cut, as #39 /
/// #40). An official-k edge swaps places with the victim (the victim stays
/// in the official k); a pool edge takes the victim's seat. The opener cut
/// keeps the opener seats' work (#13 / #21 / #29; without it LoCoMo 8:1
/// loses a seated gold opener). No new constant: "at least as on-scene"
/// is the existing floor. Skipped on summary requests, preference,
/// temporal, and WH-person questions; event sequences and
/// assistant-attribution are skipped by the caller. Invoked after the
/// gatekeeper, after #40, as the last ranking change.
pub(in crate::retrieval) fn apply_c2_episode_run_edge_t5_entry(
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
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let top: Vec<usize> = list[..window].to_vec();
    let mut by_sess: Vec<(usize, Vec<usize>)> = Vec::new();
    for &t in top.iter() {
        let s = ix.turns[t].session;
        let p = c2_session_pos(ix, t);
        match by_sess.iter_mut().find(|(bs, _)| *bs == s) {
            Some((_, ps)) => ps.push(p),
            None => by_sess.push((s, vec![p])),
        }
    }
    let eligible = |t: usize| -> bool {
        if top.contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
            return false;
        }
        let sp = ix.turns[t].speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" {
            return c2_query_mentions_speaker(query, sp) && !c2_speaker_conjoined(query, sp);
        }
        true
    };
    let mut cand: Option<usize> = None;
    for (s, ps) in by_sess.iter_mut() {
        ps.sort_unstable();
        let turns = &ix.session_turns[*s];
        let mut start = 0;
        while start < ps.len() {
            let mut end = start;
            while end + 1 < ps.len() && ps[end + 1] == ps[end] + 1 {
                end += 1;
            }
            if end > start {
                let mut edges: Vec<usize> = Vec::new();
                if ps[start] > 0 {
                    edges.push(turns[ps[start] - 1]);
                }
                if ps[end] + 1 < turns.len() {
                    edges.push(turns[ps[end] + 1]);
                }
                for t in edges {
                    if !eligible(t) {
                        continue;
                    }
                    if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                        cand = Some(t);
                    }
                }
            }
            start = end + 1;
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c
            || c2_session_pos(ix, v) == 0
            || top.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1
        {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    match list.iter().position(|&t| t == cand) {
        Some(ci) => list.swap(vs, ci),
        None => list[vs] = cand,
    }
    true
}


/// Neighbor-pair T@5 entry. In a named-speaker dialogue, an official-k
/// tail turn (slots 6..k) that sits directly next to a top-5 turn of the
/// same session completes a two-turn exchange the top 5 holds only half of.
/// The most on-scene such neighbor (declarative, t_scene > 0, by a speaker
/// the query names and does not conjoin) swaps places with the least
/// on-scene top-5 turn (slots 2..5) that is the only top-5 turn of its
/// session and not a session opener, when the neighbor is at least as
/// on-scene and that turn's session is not earlier than the neighbor's
/// (the #41 victim and primacy cuts). The victim stays in the official k,
/// so T@10 cannot drop; C2 turn seats do not rewrite the session list.
/// Official-k neighbors only: a pool neighbor (taking the seat outright)
/// loses LoCoMo 7:179. Named dialogues only: on user/assistant chats the
/// same move loses LME All@5 (4bc144e2). Distinct from #41 (the edge of a
/// back-to-back top-5 run, official k or pool) and from the slot-1
/// adjacency seats (#2 / #10 / #12 / #15 / #16). No new constant. Skipped
/// on summary requests, preference, temporal, and WH-person questions;
/// event sequences and assistant-attribution are skipped by the caller.
/// Invoked after the gatekeeper, after #41, as the last ranking change.
pub(in crate::retrieval) fn apply_c2_neighbor_pair_t5_entry(
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
    let head_sp = ix.turns[list[0]].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let top: Vec<usize> = list[..window].to_vec();
    let mut cand: Option<usize> = None;
    for &x in top.iter() {
        let turns = &ix.session_turns[ix.turns[x].session];
        let p = c2_session_pos(ix, x);
        let mut nbs: Vec<usize> = Vec::new();
        if p > 0 {
            nbs.push(turns[p - 1]);
        }
        if p + 1 < turns.len() {
            nbs.push(turns[p + 1]);
        }
        for t in nbs {
            if top.contains(&t) || !list[window..].contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
                continue;
            }
            let sp = ix.turns[t].speaker.as_str();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                if !c2_query_mentions_speaker(query, sp) || c2_speaker_conjoined(query, sp) {
                    continue;
                }
            }
            if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                cand = Some(t);
            }
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c
            || c2_session_pos(ix, v) == 0
            || top.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1
        {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    let Some(ci) = list.iter().position(|&t| t == cand) else {
        return false;
    };
    list.swap(vs, ci);
    true
}


/// Assistant-reply T@5 entry, for assistant-attribution questions only
/// (the question asks what the assistant said), which every other C2 seat
/// skips. The signal is the transcript's own role labels: an assistant turn
/// already in the official k but outside the top 5 (t_scene > 0) that sits
/// directly next to a top-5 user turn of the same session is that user
/// turn's reply. The most on-scene such reply swaps places with the least
/// on-scene top-5 user turn (slots 2..5) that (a) is not the reply's own
/// neighbor (the anchor: evicting it would only trade the exchange's halves;
/// without this cut LME 32260d93 loses its gold user turn), (b) is at most
/// as on-scene as the reply, and (c) comes from a session no earlier than
/// the reply's (primacy, as #39-#42). The victim stays in the official k,
/// so T@10 cannot drop; the swap is refused when it would push any session
/// out of the first 5 unique-session slots, so S@5 cannot drop; C2 turn
/// seats do not rewrite the session list. No new constant. Skipped on
/// temporal questions (Gemini Step 17 owns them), summary requests,
/// WH-person questions, named-speaker dialogues, and event sequences
/// (caller). Runs after the gatekeeper, outside the C2 assistant-attribution
/// / preference skip, as the last ranking change.
pub(in crate::retrieval) fn apply_c2_assistant_reply_t5_entry(
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
    let adjacent = |a: usize, b: usize| {
        ix.turns[a].session == ix.turns[b].session && c2_session_pos(ix, a).abs_diff(c2_session_pos(ix, b)) == 1
    };
    let mut cand: Option<usize> = None;
    for i in window..list.len() {
        let c = list[i];
        if ix.turns[c].speaker != "assistant" || sc(c) <= 0.0 {
            continue;
        }
        if !list[..window].iter().any(|&u| ix.turns[u].speaker == "user" && adjacent(u, c)) {
            continue;
        }
        if cand.is_none_or(|b| sc(c) > sc(list[b])) {
            cand = Some(i);
        }
    }
    let Some(ci) = cand else {
        return false;
    };
    let c = list[ci];
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        if ix.turns[v].speaker != "user" || sc(v) > sc(c) || adjacent(v, c) || ts_of(v) < ts_of(c) {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
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
