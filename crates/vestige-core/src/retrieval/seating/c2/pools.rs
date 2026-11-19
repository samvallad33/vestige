//! C2 seating lane: `pools`.

use crate::retrieval::*;



/// Pool mentioned-speaker T@10 entry. In a named-speaker dialogue, a
/// turn by a speaker the query names, in the scene-ranked pool but not
/// already in the official k, takes the official-k seat of the last
/// unmentioned named speaker in the official-k tail (slots 6..k), when
/// that turn is at least as on-scene as the evicted turn. The top 5 is
/// unchanged, so T@1 and T@5 cannot drop. The engine's session list is
/// not rewritten by C2 turn seats, so official Session cannot drop.
/// The replacement is also refused when it would push any session out
/// of the first 5 unique-session slots. Distinct from #17 and #23
/// (those candidates are already in official k and are pulled into the
/// top 5) and from #30 (user/assistant openers). No new constant: the
/// scene floor is "at least as on-scene," the same comparison #12 and
/// #30 use. Skipped when the query conjoins the named speaker with
/// someone else ("X and Y"), on summary requests (existing
/// `summary_hold` shape), preference, temporal, and WH-person
/// questions, and on event sequences and assistant-attribution
/// (caller). Does not fire on user/assistant chats. Invoked after the
/// gatekeeper, after #31. A later post-gatekeeper C2 seat may still
/// run after this pass.
pub(in crate::retrieval) fn apply_c2_pool_mentioned_t10_entry(
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
    if list.len() <= window {
        return false;
    }
    let k = list.len();
    let mut vs = None;
    for i in window..k {
        let sp = ix.turns[list[i]].speaker.as_str();
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
        let csp = ix.turns[ti].speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
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


/// Pool C session-key opener. The turn at session position 0 is the session
/// key. An assistant elaboration already in the top 5 yields that seat to the
/// user opener of the same session when the opener is not yet in the top 5,
/// posts at least one query group, and every query group the elaboration
/// attests is attested by the opener or by another turn that stays in the top
/// 5. The top 5 must already hold two sessions: a one-session list has no
/// second key to admit, and the elaboration there is the answer rather than
/// an echo. A digit on the elaboration that the opener does not post keeps
/// the elaboration. Slot 0 is frozen. Named-speaker corpora, text-level
/// preference, summaries, event sequences, and assistant-attribution questions
/// do not fire. Distinct from the cross-opener seat, which only reorders an
/// opener already inside the top 5. Does not read `TurnIx.body` and does not
/// touch `rsb_seat_behind`.
pub(in crate::retrieval) fn apply_c2_pool_c_session_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < C2_BULLAE_WINDOW
        || plan.groups.is_empty()
        || c2_text_preference(query)
        || plan.intent.assistant_attribution
        || plan.intent.is_event_sequence
        || c2_summary_request(query)
    {
        return false;
    }
    let named = ix.turns.iter().any(|t| {
        let sp = t.speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    });
    if named {
        return false;
    }
    let ng = plan.groups.len();
    let scene = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let window0 = list.len().min(C2_BULLAE_WINDOW);
    let mut sessions: DetHashSet<usize> = DetHashSet::default();
    for &t in &list[..window0] {
        sessions.insert(ix.turns[t].session);
    }
    // One session in the window: the elaboration is not an echo of a second key.
    if sessions.len() < 2 {
        return false;
    }
    let mut fired = false;
    for _ in 0..4 {
        let window = list.len().min(C2_BULLAE_WINDOW);
        let top: Vec<usize> = list[..window].to_vec();
        let mut best: Option<(usize, usize)> = None;
        for i in 1..window {
            let victim = list[i];
            if ix.turns[victim].speaker != "assistant" {
                continue;
            }
            let sess = ix.turns[victim].session;
            let Some(&opener) = ix.session_turns.get(sess).and_then(|m| m.first()) else {
                continue;
            };
            if opener == victim || top.contains(&opener) {
                continue;
            }
            if ix.turns[opener].speaker != "user" || c2_session_pos(ix, opener) != 0 {
                continue;
            }
            if scene(opener) <= 0.0 {
                continue;
            }
            if ix.turns[victim].form_digit && !ix.turns[opener].form_digit {
                continue;
            }
            let mut cover = c2_turn_groups(ix, plan, ng, opener);
            if !cover.contains(&true) {
                continue;
            }
            for (k, &other) in top.iter().enumerate() {
                if k == i {
                    continue;
                }
                let other_groups = c2_turn_groups(ix, plan, ng, other);
                for (slot, hit) in other_groups.into_iter().enumerate() {
                    if hit {
                        cover[slot] = true;
                    }
                }
            }
            let victim_groups = c2_turn_groups(ix, plan, ng, victim);
            if victim_groups.iter().zip(&cover).any(|(hit, kept)| *hit && !*kept) {
                continue;
            }
            let take = match best {
                None => true,
                Some((bi, bo)) => {
                    scene(opener) > scene(bo)
                        || (scene(opener) == scene(bo) && (opener < bo || (opener == bo && i < bi)))
                }
            };
            if take {
                best = Some((i, opener));
            }
        }
        let Some((i, opener)) = best else {
            break;
        };
        if let Some(j) = list.iter().position(|&t| t == opener) {
            list.swap(i, j);
        } else {
            list[i] = opener;
        }
        fired = true;
    }
    fired
}


/// Pool C absent session key. A user opener at session position 0 whose
/// session is outside the top 5 takes the seat of an assistant from a
/// non-dominant session, when that assistant's session keeps another turn and
/// that turn attests every query group the assistant attests. The dominant
/// session — the one with the most turns in the window, ties broken by the
/// earliest slot — is not a donor. The opener must post a query group and a
/// positive scene score. A digit on the assistant that the opener lacks keeps
/// the assistant. One-session windows do not fire. Does not read
/// `TurnIx.body` and does not touch `rsb_seat_behind`.
pub(in crate::retrieval) fn apply_c2_pool_c_absent_session_key(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < C2_BULLAE_WINDOW
        || plan.groups.is_empty()
        || c2_text_preference(query)
        || plan.intent.assistant_attribution
        || plan.intent.is_event_sequence
        || c2_summary_request(query)
    {
        return false;
    }
    let named = ix.turns.iter().any(|t| {
        let sp = t.speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    });
    if named {
        return false;
    }
    let ng = plan.groups.len();
    let scene = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let mut fired = false;
    for _ in 0..4 {
        let window = list.len().min(C2_BULLAE_WINDOW);
        let top: Vec<usize> = list[..window].to_vec();
        let mut counts: DetHashMap<usize, usize> = DetHashMap::default();
        for &t in &top {
            *counts.entry(ix.turns[t].session).or_insert(0) += 1;
        }
        if counts.len() < 2 {
            break;
        }
        let mut mode = ix.turns[top[0]].session;
        let mut mode_n = 0usize;
        let mut mode_at = window;
        for &t in &top {
            let s = ix.turns[t].session;
            let n = counts[&s];
            let at = top.iter().position(|&u| ix.turns[u].session == s).unwrap_or(window);
            if n > mode_n || (n == mode_n && at < mode_at) {
                mode = s;
                mode_n = n;
                mode_at = at;
            }
        }
        let mut best: Option<(usize, usize)> = None;
        for i in 1..window {
            let victim = list[i];
            if ix.turns[victim].speaker != "assistant" {
                continue;
            }
            let vs = ix.turns[victim].session;
            if vs == mode {
                continue;
            }
            let companions: Vec<usize> = top
                .iter()
                .copied()
                .filter(|&t| t != victim && ix.turns[t].session == vs)
                .collect();
            if companions.is_empty() {
                continue;
            }
            let mut cover = vec![false; ng];
            for c in &companions {
                for (slot, hit) in c2_turn_groups(ix, plan, ng, *c).into_iter().enumerate() {
                    if hit {
                        cover[slot] = true;
                    }
                }
            }
            let victim_groups = c2_turn_groups(ix, plan, ng, victim);
            if victim_groups.iter().zip(&cover).any(|(hit, kept)| *hit && !*kept) {
                continue;
            }
            for (sess, members) in ix.session_turns.iter().enumerate() {
                if counts.contains_key(&sess) {
                    continue;
                }
                let Some(&opener) = members.first() else {
                    continue;
                };
                if top.contains(&opener) {
                    continue;
                }
                if ix.turns[opener].speaker != "user" || c2_session_pos(ix, opener) != 0 {
                    continue;
                }
                if scene(opener) <= 0.0 {
                    continue;
                }
                if ix.turns[victim].form_digit && !ix.turns[opener].form_digit {
                    continue;
                }
                let opener_groups = c2_turn_groups(ix, plan, ng, opener);
                if !opener_groups.contains(&true) {
                    continue;
                }
                let take = match best {
                    None => true,
                    Some((bi, bo)) => {
                        scene(opener) > scene(bo)
                            || (scene(opener) == scene(bo)
                                && (opener < bo || (opener == bo && i < bi)))
                    }
                };
                if take {
                    best = Some((i, opener));
                }
            }
        }
        let Some((i, opener)) = best else {
            break;
        };
        if let Some(j) = list.iter().position(|&t| t == opener) {
            list.swap(i, j);
        } else {
            list[i] = opener;
        }
        fired = true;
    }
    fired
}


/// Pool-B standing-rule seat. `rank_standing_rules` already names the one
/// standing imperative that applies. When that rule is a user session-opener
/// already inside the top 5, its own lexical score is zero (the standing
/// floor, not a content match, is why it is seated), it carries an explicit
/// conditional, and one of its stored content terms is an inflection of a
/// query focal term, the rule takes slot 1 over a different user question.
/// Top-5 membership does not change. Does not read turn bodies, does not
/// touch `rsb_seat_behind`, and does not rewrite the session list.
///
/// A zero-specificity firing (the channel keeps every task-under-rules
/// policy when none of them hit the query) does not take slot 1. That is
/// the case that displaces a question which is itself the answer. Closed
/// class gates: task-under-rules, and not a text-level preference question,
/// summary request, temporal question, event sequence, multi-hop,
/// count/when/where/why sniper, current-state question, or
/// assistant-attribution question. Named-speaker sessions are skipped. A
/// rule that still carries its own lexical score stays where submodular
/// seating put it.
pub(in crate::retrieval) fn apply_c2_pool_b_standing_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    standing: &[RankedItem],
    list: &mut Vec<usize>,
    t_own: &[f32],
) -> bool {
    let it = &plan.intent;
    if list.len() < 2
        || standing.is_empty()
        || !it.task_under_rules
        || it.temporal
        || it.is_event_sequence
        || it.multi_hop
        || it.sniper_how_many
        || it.sniper_when
        || it.sniper_where
        || it.sniper_why
        || it.current_state
        || it.assistant_attribution
        || c2_text_preference(query)
        || c2_summary_request(query)
    {
        return false;
    }
    let head = list[0];
    let head_turn = &ix.turns[head];
    if head_turn.speaker != "user" || !head_turn.is_question || head_turn.is_standing_policy {
        return false;
    }
    let winner = standing[0].id.as_str();
    let Some(rule) = ix
        .standing_rules
        .iter()
        .find(|r| ix.turns[r.turn].id == winner)
    else {
        return false;
    };
    // The channel fires every task-under-rules policy when none of them hit
    // a focal id. That zero-specificity winner is what displaces a question
    // that is itself the answer. The scope set itself is fan-cut, so a rare
    // conditional word may be gone. Require an explicit conditional, then an
    // orthographic inflection between the policy's stored content terms and
    // a query focal term (`implement` / `implementation`). A shared prefix
    // that is not an inflection (`date` / `deadline`) does not count.
    if !rule.has_explicit_scope || !c2_policy_inflects_query(ix, rule.turn, plan) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window]
        .iter()
        .position(|&ti| ix.turns[ti].id == winner)
    else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    let cand = list[slot];
    let t = &ix.turns[cand];
    if t.speaker != "user"
        || t.is_question
        || !t.is_standing_policy
        || c2_named_session(ix, cand)
        || c2_session_pos(ix, cand) != 0
    {
        return false;
    }
    if t_own.get(cand).copied().unwrap_or(0.0) > 0.0 {
        return false;
    }
    let moved = list.remove(slot);
    list.insert(0, moved);
    true
}


/// Buried lexical peak. The scene pool is the same 50 the replay dump calls
/// pool@50: the seated list, then turns with `t_scene > 0` in scene order.
/// A declarative in a named-speaker session whose own lexical score strictly
/// exceeds every turn in that pool, when it is the only such turn, was cut
/// by scene penalties rather than by a missing posting. It takes the seat of
/// the lowest-own question in slots 1..4. Slot 0 stays. The displaced question
/// is kept: appended while the list is short of `k`, otherwise written over
/// the tail. The session list is not rewritten. Does not touch `rsb_seat_behind`.
pub(in crate::retrieval) fn apply_c2_buried_lexical_peak(
    ix: &Index,
    list: &mut Vec<usize>,
    t_scene: &[f32],
    t_own: &[f32],
    turn_ts: &[i64],
    k: usize,
) -> bool {
    if list.len() < 5 || k == 0 {
        return false;
    }
    let mut pool: Vec<usize> = list.clone();
    for ti in order(t_scene, turn_ts) {
        if pool.len() >= 50 {
            break;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if !pool.contains(&ti) {
            pool.push(ti);
        }
    }
    if pool.is_empty() {
        return false;
    }
    let mut pool_max = f32::MIN;
    for &ti in &pool {
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        if own > pool_max {
            pool_max = own;
        }
    }
    let mut peak: Option<usize> = None;
    for ti in 0..ix.turns.len() {
        if pool.contains(&ti) || ix.turns[ti].is_question || !c2_named_session(ix, ti) {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        if own <= pool_max || own <= 0.0 {
            continue;
        }
        if peak.is_some() {
            return false;
        }
        peak = Some(ti);
    }
    let Some(best) = peak else {
        return false;
    };
    let mut victim: Option<(f32, usize)> = None;
    for slot in 1..5.min(list.len()) {
        let ti = list[slot];
        if !ix.turns[ti].is_question {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        let take = match victim {
            None => true,
            Some((bow, bslot)) => own < bow || (own == bow && slot < bslot),
        };
        if take {
            victim = Some((own, slot));
        }
    }
    let Some((_, slot)) = victim else {
        return false;
    };
    let displaced = list[slot];
    list[slot] = best;
    if !list.contains(&displaced) {
        if list.len() < k {
            list.push(displaced);
        } else if !list.is_empty() {
            let last = list.len() - 1;
            list[last] = displaced;
        }
    }
    true
}


/// Anaphoric tail. The strongest non-question in ranks 5..9 of a named-speaker
/// session, when that turn is anaphoric and its own score strictly beats every
/// turn in the top 5, swaps with the lowest-own question in slots 1..4.
/// The swap stays inside the returned list, so the top-10 set is unchanged.
/// Slot 0 stays. Anaphora is the ingest flag (closed-class continuation or
/// pronoun), not a new word list. Does not read turn text and does not touch
/// `rsb_seat_behind` or the session list.
pub(in crate::retrieval) fn apply_c2_anaphoric_tail(ix: &Index, list: &mut Vec<usize>, t_own: &[f32]) -> bool {
    if list.len() < 6 {
        return false;
    }
    let top_end = 5.min(list.len());
    let tail_end = 10.min(list.len());
    let mut best: Option<usize> = None;
    for slot in 5..tail_end {
        let ti = list[slot];
        if ix.turns[ti].is_question || !c2_named_session(ix, ti) {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        let take = match best {
            None => true,
            Some(bslot) => {
                let bow = t_own.get(list[bslot]).copied().unwrap_or(0.0);
                own > bow || (own == bow && slot < bslot)
            }
        };
        if take {
            best = Some(slot);
        }
    }
    let Some(bslot) = best else {
        return false;
    };
    let best_ti = list[bslot];
    if !ix.turns[best_ti].anaphoric {
        return false;
    }
    let best_own = t_own.get(best_ti).copied().unwrap_or(0.0);
    let mut top_max = f32::MIN;
    for slot in 0..top_end {
        let own = t_own.get(list[slot]).copied().unwrap_or(0.0);
        if own > top_max {
            top_max = own;
        }
    }
    if best_own <= top_max {
        return false;
    }
    let mut victim: Option<usize> = None;
    for slot in 1..top_end {
        let ti = list[slot];
        if !ix.turns[ti].is_question {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        let take = match victim {
            None => true,
            Some(vslot) => {
                let vow = t_own.get(list[vslot]).copied().unwrap_or(0.0);
                own < vow || (own == vow && slot < vslot)
            }
        };
        if take {
            victim = Some(slot);
        }
    }
    let Some(vslot) = victim else {
        return false;
    };
    list.swap(bslot, vslot);
    true
}


/// Unseated session peak. A declarative outside the returned list whose own
/// score strictly beats every returned turn, when it is the only such turn and
/// its session has no turn in the top 5, takes the lowest-own question seat in
/// slots 1..4. The displaced question stays in the list. Named-speaker
/// sessions only. Slot 0 stays. Does not read turn text and does not touch
/// `rsb_seat_behind` or the session list.
pub(in crate::retrieval) fn apply_c2_unseated_session_peak(
    ix: &Index,
    list: &mut Vec<usize>,
    t_own: &[f32],
    k: usize,
) -> bool {
    if list.len() < 5 || k == 0 {
        return false;
    }
    let mut cap = f32::MIN;
    for &ti in list.iter() {
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        if own > cap {
            cap = own;
        }
    }
    let mut peak: Option<usize> = None;
    for ti in 0..ix.turns.len() {
        if list.contains(&ti) || ix.turns[ti].is_question || !c2_named_session(ix, ti) {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        if own <= cap {
            continue;
        }
        if peak.is_some() {
            return false;
        }
        peak = Some(ti);
    }
    let Some(best) = peak else {
        return false;
    };
    let sess = ix.turns[best].session;
    let top = 5.min(list.len());
    if list[..top].iter().any(|&ti| ix.turns[ti].session == sess) {
        return false;
    }
    let mut victim: Option<usize> = None;
    for slot in 1..top {
        let ti = list[slot];
        if !ix.turns[ti].is_question {
            continue;
        }
        let own = t_own.get(ti).copied().unwrap_or(0.0);
        let take = match victim {
            None => true,
            Some(vslot) => {
                let vow = t_own.get(list[vslot]).copied().unwrap_or(0.0);
                own < vow || (own == vow && slot < vslot)
            }
        };
        if take {
            victim = Some(slot);
        }
    }
    let Some(slot) = victim else {
        return false;
    };
    let displaced = list[slot];
    list[slot] = best;
    if !list.contains(&displaced) {
        if list.len() < k {
            list.push(displaced);
        } else if !list.is_empty() {
            let last = list.len() - 1;
            list[last] = displaced;
        }
    }
    true
}
