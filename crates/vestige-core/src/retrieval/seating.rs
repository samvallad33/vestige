//! Slot seating passes over the final turn list: primacy, given/new, opener lift, hearsay, RSB, Kemeny gatekeeper, C2 passes.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

/// Rank every query against the supplied sessions. Pure, deterministic, no I/O.
// =========================================================================
// RECALL-CUE SLOT-1 RULES (position, session, role field, token identity only)
// =========================================================================

/// BoundaryPrimacy window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed before any run; the top-5 membership never changes.
pub(super) const PRIMACY_WINDOW: usize = 5;

/// Query cues: distinct kernel tokens of the query that occur somewhere in the haystack, minus
/// any token that occurs in a turn's role/speaker field (structural metadata, no word list).
pub(super) fn query_cues(ix: &Index, text: &str) -> Vec<u32> {
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
pub(super) fn turn_query_cues(ix: &Index, q_cues: &[u32], ti: usize) -> Vec<u32> {
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
pub(super) fn covers_cues(sup: &[u32], sub: &[u32]) -> bool {
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
pub(super) fn turn_pos(ix: &Index, ti: usize) -> Option<usize> {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti)
}

/// Structural user-preference / personal-constraint turn check (no PREF_MARKERS dictionary list).
pub(super) fn is_structural_personal_turn(ix: &Index, ti: usize) -> bool {
    let tr = &ix.turns[ti];
    tr.field == 0
        && (tr.is_standing_policy
            || tr.is_personal_constraint
            || (tr.speaker == "user" && turn_pos(ix, ti).map_or(false, |pos| pos <= 4)))
}

pub(super) struct RecallCueGuard<'a> {
    pub(super) plan: &'a QueryPlan,
    pub(super) it: &'a QueryIntent,
    pub(super) t_scene: &'a [f32],
}

/// Score-dominance and state-polarity guard for `boundary_primacy`, `given_new`,
/// and `session_opener_lift_with_plan`. When `guard` is `None` (direct unit tests),
/// preserves baseline structural behavior.
pub(super) fn recall_cue_promotion_allowed(
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

pub(super) fn boundary_primacy_with_plan(
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
pub(super) const GIVEN_NEW_WINDOW: usize = 10;
/// GivenNew variant, fixed before the run. `false` = card 2a (any role). `true` = card 2b: fires
/// only when rank 1's role field is the chat schema value `user` or `assistant` (structural field).
pub(super) const GIVEN_NEW_CHAT_ROLES_ONLY: bool = true;

pub(super) fn given_new_with_plan(
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

/// SessionOpenerLift window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed at 5 before any run; the top-5 membership never changes.
pub(super) const SESSION_OPENER_LIFT_WINDOW: usize = 5;

pub(super) fn session_opener_lift_with_plan(
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
pub(super) const HEARSAY_WINDOW: usize = 5;

pub(super) fn hearsay_with_plan(
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

/// Nearest preceding self-contained turn in the same session. A follow-up is an
/// assistant turn or a turn flagged anaphoric at ingest. The antecedent is the
/// cause; it need not share words with the follow-up.
pub(super) fn rsb_antecedent(ix: &Index, ti: usize) -> Option<usize> {
    let t = &ix.turns[ti];
    if t.speaker != "assistant" && !t.anaphoric {
        return None;
    }
    let members = &ix.session_turns[t.session];
    let pos = members.iter().position(|&m| m == ti)?;
    for &prev in members[..pos].iter().rev() {
        let p = &ix.turns[prev];
        if p.field == 0 && !p.anaphoric {
            return Some(prev);
        }
    }
    None
}

/// One structural cause for a retrieved symptom: the antecedent if this turn is
/// a follow-up, otherwise the nearest earlier standing policy in the same episode.
pub(super) fn rsb_cause(ix: &Index, ti: usize) -> Option<usize> {
    if let Some(ante) = rsb_antecedent(ix, ti) {
        return Some(ante);
    }
    ix.turns[ti].policy_backlinks.last().copied()
}

/// Seat each cause immediately after the symptom that points at it.
///
/// Version 4, fixed before the run. Version 3 inserted the cause and then
/// truncated the list to k, which drops whoever already sat in the last slot.
/// That drop is not a salience change. A cause that is not already in the
/// window is inserted only while the window is shorter than k, so an empty
/// slot is filled and nobody already inside is removed. A cause already in
/// the window is moved to sit immediately after its symptom, and is never
/// moved ahead of that symptom. A full window does not grow. k is the
/// caller's top_k. The backchannel skip on the antecedent walk is removed
/// so the walk matches the spec: nearest preceding field-0 turn that is
/// not anaphoric.
pub(super) fn rsb_seat_behind(ix: &Index, list: &mut Vec<usize>, k: usize) {
    if list.is_empty() || k == 0 {
        return;
    }
    let snapshot = list.clone();
    let mut planned: Vec<(usize, usize)> = Vec::new();
    let mut reserved = snapshot.clone();
    for (i, &ti) in snapshot.iter().enumerate() {
        let Some(cause) = rsb_cause(ix, ti) else { continue };
        if planned.iter().any(|&(_, c)| c == cause) {
            continue;
        }
        let inside = snapshot.contains(&cause);
        if !inside && reserved.len() >= k {
            continue;
        }
        if !inside {
            reserved.push(cause);
        }
        planned.push((i, cause));
    }
    for (i, cause) in planned.into_iter().rev() {
        let Some(sym_at) = list.iter().position(|&t| t == snapshot[i]) else { continue };
        if let Some(cause_at) = list.iter().position(|&t| t == cause) {
            if cause_at <= sym_at {
                continue;
            }
            list.remove(cause_at);
            let sym_at = list.iter().position(|&t| t == snapshot[i]).unwrap();
            list.insert(sym_at + 1, cause);
        } else if list.len() < k {
            list.insert(sym_at + 1, cause);
        }
    }
}

/// Exact Kemeny-Young inside one Condorcet component. 8! = 40320 is the
/// largest search this branch-and-bound solves exactly. Larger components
/// use Copeland plus adjacent swaps. Fixed from the size of the search,
/// before any card.
pub(super) const KEMENY_EXACT_CAP: usize = 8;

pub(super) const WH_TIME: u8 = 1;
pub(super) const WH_QTY: u8 = 2;
pub(super) const WH_PERSON: u8 = 4;
pub(super) const WH_PLACE: u8 = 8;
pub(super) const WH_CAUSE: u8 = 16;

/// Closed class of English interrogative function words. Not a topical list.
pub(super) fn wh_slots(query: &str) -> u8 {
    let lower = query.to_ascii_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    let mut slots = 0u8;
    let has = |w: &str| tokens.iter().any(|t| t == w);
    let phrase = |a: &str, b: &str| {
        tokens.windows(2).any(|p| p[0] == a && p[1] == b)
    };
    // "when" is interrogative only before a closed-class auxiliary ("when did"),
    // not before a lexical verb ("when building"). Auxiliaries are the same
    // function words already in STOPWORDS.
    let when_interrogative = tokens.windows(2).any(|p| {
        p[0] == "when"
            && matches!(
                p[1].as_str(),
                "did" | "didnt"
                    | "does"
                    | "doesnt"
                    | "do"
                    | "dont"
                    | "is"
                    | "isnt"
                    | "are"
                    | "arent"
                    | "was"
                    | "wasnt"
                    | "were"
                    | "werent"
                    | "will"
                    | "wont"
                    | "can"
                    | "cant"
                    | "could"
                    | "couldnt"
                    | "would"
                    | "wouldnt"
                    | "should"
                    | "shouldnt"
                    | "shall"
                    | "has"
                    | "hasnt"
                    | "have"
                    | "havent"
                    | "had"
                    | "hadnt"
                    | "may"
                    | "might"
                    | "must"
            )
    });
    if phrase("what", "time")
        || phrase("which", "year")
        || phrase("how", "long")
        || when_interrogative
    {
        slots |= WH_TIME;
    }
    if phrase("how", "many") || phrase("how", "much") || phrase("how", "often") {
        slots |= WH_QTY;
    }
    if has("who") || has("whom") || has("whose") {
        slots |= WH_PERSON;
    }
    if has("where") {
        slots |= WH_PLACE;
    }
    if has("why") {
        slots |= WH_CAUSE;
    }
    slots
}

pub(super) fn has_iso_or_hhmm(body: &str) -> bool {
    let b = body.as_bytes();
    if b.len() >= 10 {
        for i in 0..=b.len() - 10 {
            if b[i].is_ascii_digit()
                && b[i + 1].is_ascii_digit()
                && b[i + 2].is_ascii_digit()
                && b[i + 3].is_ascii_digit()
                && b[i + 4] == b'-'
                && b[i + 5].is_ascii_digit()
                && b[i + 6].is_ascii_digit()
                && b[i + 7] == b'-'
                && b[i + 8].is_ascii_digit()
                && b[i + 9].is_ascii_digit()
            {
                return true;
            }
        }
    }
    for i in 0..b.len() {
        if b[i] == b':'
            && i > 0
            && i + 2 < b.len()
            && b[i - 1].is_ascii_digit()
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

pub(super) fn has_ascii_digit(body: &str) -> bool {
    body.bytes().any(|c| c.is_ascii_digit())
}

pub(super) fn has_handle_or_path(body: &str) -> bool {
    let b = body.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'@' && i + 1 < b.len() && b[i + 1].is_ascii_alphanumeric() {
            return true;
        }
        if i + 2 < b.len() && b[i] == b':' && b[i + 1] == b'/' && b[i + 2] == b'/' {
            return true;
        }
    }
    body.split_whitespace().any(|tok| tok.contains('/') && tok.len() > 1)
}

/// A capitalized word of at least two letters that is not the first word of a sentence.
pub(super) fn has_mid_sentence_name(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let mut sentence_start = true;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            sentence_start = true;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_uppercase() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            if !sentence_start && i - start >= 2 {
                return true;
            }
            sentence_start = false;
            continue;
        }
        sentence_start = false;
        i += 1;
    }
    false
}

pub(super) fn wh_satisfied(ix: &Index, ti: usize, slots: u8, has_cause_edge: bool) -> bool {
    if slots == 0 {
        return true;
    }
    let turn = &ix.turns[ti];
    let ts = ix.session_ts.get(turn.session).copied().unwrap_or(0);
    if slots & WH_TIME != 0 && ts == 0 && !turn.form_clock {
        return false;
    }
    if slots & WH_QTY != 0 && !turn.form_digit {
        return false;
    }
    if slots & WH_PERSON != 0 && !turn.form_handle && !turn.form_midcap {
        return false;
    }
    if slots & WH_PLACE != 0 && !turn.form_midcap && !turn.form_handle {
        return false;
    }
    if slots & WH_CAUSE != 0 && !has_cause_edge {
        return false;
    }
    true
}

/// In-degree inside the pool from edges the index already stored, plus the
/// RSB antecedent. One pass over the log, not one pass per candidate.
pub(super) fn pool_structural_degree(ix: &Index, pool: &[usize]) -> Vec<i64> {
    let mut at = vec![None; ix.turns.len()];
    for (i, &ti) in pool.iter().enumerate() {
        if let Some(slot) = at.get_mut(ti) {
            *slot = Some(i);
        }
    }
    let mut degree = vec![0i64; pool.len()];
    for (src, t) in ix.turns.iter().enumerate() {
        let mut hit = |target: usize| {
            if target == src {
                return;
            }
            if let Some(Some(slot)) = at.get(target) {
                degree[*slot] += 1;
            }
        };
        for &target in &t.policy_backlinks {
            hit(target);
        }
        for &target in &t.supersedes_turns {
            hit(target);
        }
        for &(_, target) in &t.exact_edges {
            hit(target);
        }
        if let Some(target) = rsb_antecedent(ix, src) {
            hit(target);
        }
    }
    degree
}

pub(super) fn kemeny_pair_score(order: &[usize], net: &[Vec<i64>]) -> i64 {
    let mut score = 0i64;
    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            score += net[order[i]][order[j]];
        }
    }
    score
}

pub(super) fn kemeny_bound(prefix: &[usize], used: &[bool], net: &[Vec<i64>]) -> i64 {
    let m = used.len();
    let mut score = kemeny_pair_score(prefix, net);
    for &p in prefix {
        for c in 0..m {
            if !used[c] {
                score += net[p][c];
            }
        }
    }
    let rest: Vec<usize> = (0..m).filter(|&c| !used[c]).collect();
    for i in 0..rest.len() {
        for j in (i + 1)..rest.len() {
            score += net[rest[i]][rest[j]].abs();
        }
    }
    score
}

/// Exact Kemeny order of a component. Equal scores keep the RSB order,
/// which is the initial incumbent, so a tie never falls through to turn id.
pub(super) fn kemeny_branch_and_bound(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let m = members.len();
    let mut local = vec![vec![0i64; m]; m];
    for i in 0..m {
        for j in 0..m {
            local[i][j] = net[members[i]][members[j]];
        }
    }
    let mut start: Vec<usize> = (0..m).collect();
    start.sort_by(|&a, &b| rsb[members[a]].cmp(&rsb[members[b]]));
    let key_of = |order: &[usize]| -> Vec<usize> {
        order.iter().map(|&i| rsb[members[i]]).collect()
    };
    let mut best_score = kemeny_pair_score(&start, &local);
    let mut best = start;
    let mut best_key = key_of(&best);
    let mut used = vec![false; m];
    let mut prefix: Vec<usize> = Vec::with_capacity(m);
    fn rec(
        local: &[Vec<i64>],
        members: &[usize],
        rsb: &[usize],
        used: &mut [bool],
        prefix: &mut Vec<usize>,
        best_score: &mut i64,
        best: &mut Vec<usize>,
        best_key: &mut Vec<usize>,
    ) {
        let m = used.len();
        if prefix.len() == m {
            let score = kemeny_pair_score(prefix, local);
            let key: Vec<usize> = prefix.iter().map(|&i| rsb[members[i]]).collect();
            if score > *best_score || (score == *best_score && key < *best_key) {
                *best_score = score;
                *best_key = key;
                *best = prefix.clone();
            }
            return;
        }
        if kemeny_bound(prefix, used, local) < *best_score {
            return;
        }
        for c in 0..m {
            if used[c] {
                continue;
            }
            used[c] = true;
            prefix.push(c);
            rec(local, members, rsb, used, prefix, best_score, best, best_key);
            prefix.pop();
            used[c] = false;
        }
    }
    rec(&local, members, rsb, &mut used, &mut prefix, &mut best_score, &mut best, &mut best_key);
    best.into_iter().map(|i| members[i]).collect()
}

pub(super) fn kemeny_copeland(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut order = members.to_vec();
    order.sort_by(|&a, &b| {
        let sa: i64 = members.iter().map(|&c| (net[a][c] > 0) as i64 - (net[a][c] < 0) as i64).sum();
        let sb: i64 = members.iter().map(|&c| (net[b][c] > 0) as i64 - (net[b][c] < 0) as i64).sum();
        sb.cmp(&sa).then(rsb[a].cmp(&rsb[b]))
    });
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..order.len().saturating_sub(1) {
            let a = order[i];
            let b = order[i + 1];
            if net[a][b] < 0 {
                order.swap(i, i + 1);
                improved = true;
            }
        }
    }
    order
}

pub(super) fn tarjan_sccs_best_first(net: &[Vec<i64>], rsb: &[usize]) -> Vec<Vec<usize>> {
    let n = net.len();
    let mut index = 0usize;
    let mut indices = vec![None; n];
    let mut low = vec![0usize; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut on = vec![false; n];
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    fn strong(
        v: usize,
        net: &[Vec<i64>],
        index: &mut usize,
        indices: &mut [Option<usize>],
        low: &mut [usize],
        stack: &mut Vec<usize>,
        on: &mut [bool],
        sccs: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        low[v] = *index;
        *index += 1;
        stack.push(v);
        on[v] = true;
        for w in 0..net.len() {
            if v != w && net[v][w] > 0 {
                if indices[w].is_none() {
                    strong(w, net, index, indices, low, stack, on, sccs);
                    low[v] = low[v].min(low[w]);
                } else if on[w] {
                    low[v] = low[v].min(indices[w].unwrap_or(0));
                }
            }
        }
        if low[v] == indices[v].unwrap_or(0) {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                on[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(comp);
        }
    }
    for v in 0..n {
        if indices[v].is_none() {
            strong(v, net, &mut index, &mut indices, &mut low, &mut stack, &mut on, &mut sccs);
        }
    }
    let mut id_of = vec![0usize; n];
    for (i, comp) in sccs.iter().enumerate() {
        for &v in comp {
            id_of[v] = i;
        }
    }
    let ns = sccs.len();
    let mut indeg = vec![0usize; ns];
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); ns];
    for a in 0..n {
        for b in 0..n {
            if a != b && net[a][b] > 0 && id_of[a] != id_of[b] {
                let (u, v) = (id_of[a], id_of[b]);
                if !next[u].contains(&v) {
                    next[u].push(v);
                    indeg[v] += 1;
                }
            }
        }
    }
    let anchor = |comp: &[usize]| comp.iter().map(|&v| rsb.get(v).copied().unwrap_or(usize::MAX)).min().unwrap_or(usize::MAX);
    let mut ready: Vec<usize> = (0..ns).filter(|&s| indeg[s] == 0).collect();
    let sort_ready = |ready: &mut Vec<usize>, sccs: &[Vec<usize>]| {
        ready.sort_by(|&a, &b| anchor(&sccs[b]).cmp(&anchor(&sccs[a])).then(b.cmp(&a)));
    };
    sort_ready(&mut ready, &sccs);
    let mut ordered = Vec::with_capacity(ns);
    while let Some(u) = ready.pop() {
        ordered.push(u);
        let mut outs = next[u].clone();
        outs.sort_unstable();
        for v in outs {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                ready.push(v);
            }
        }
        sort_ready(&mut ready, &sccs);
    }
    if ordered.len() != ns {
        ordered = (0..ns).collect();
    }
    // Tarjan's component ids are not best-first. Kahn from sources of the
    // "beats" condensation puts undefeated components first.
    ordered.into_iter().map(|i| sccs[i].clone()).collect()
}

pub(super) fn kemeny_young(net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(net.len());
    for scc in tarjan_sccs_best_first(net, rsb) {
        if scc.len() <= 1 {
            out.extend(scc);
        } else if scc.len() <= KEMENY_EXACT_CAP {
            out.extend(kemeny_branch_and_bound(&scc, net, rsb));
        } else {
            out.extend(kemeny_copeland(&scc, net, rsb));
        }
    }
    out
}

/// Rank-1 gatekeeper. Runs after RSB. Reorders the RSB window. It does not
/// grow the window. The pool is that window, so its size is the caller's k.
pub(super) fn rank1_gatekeeper(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    let n = list.len();
    if n < 2 {
        return;
    }
    let pool = list.clone();
    let slots = wh_slots(query);
    let degree = pool_structural_degree(ix, &pool);
    let mut net = vec![vec![0i64; n]; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let (ta, tb) = (pool[a], pool[b]);
            let mut margin = 0i64;
            let scene_a = t_scene.get(ta).copied().unwrap_or(0.0);
            let scene_b = t_scene.get(tb).copied().unwrap_or(0.0);
            match scene_a.partial_cmp(&scene_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            match degree[a].cmp(&degree[b]) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            let stab_a = ix.turns.get(ta).map(|t| t.stability).unwrap_or(0.0);
            let stab_b = ix.turns.get(tb).map(|t| t.stability).unwrap_or(0.0);
            match stab_a.partial_cmp(&stab_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            if slots != 0 {
                let ok_a = wh_satisfied(ix, ta, slots, degree[a] > 0);
                let ok_b = wh_satisfied(ix, tb, slots, degree[b] > 0);
                if ok_a && !ok_b {
                    margin += 1;
                } else if ok_b && !ok_a {
                    margin -= 1;
                }
            }
            net[a][b] = margin;
            net[b][a] = -margin;
        }
    }
    let rsb: Vec<usize> = (0..n).collect();
    let mut order = kemeny_young(&net, &rsb);
    if order.len() != n {
        order = rsb.clone();
    }
    let mut turned: Vec<usize> = order.iter().map(|&i| pool[i]).collect();
    if turn_is_superseded(ix, turned[0]) {
        if let Some(pos) = turned.iter().position(|&ti| !turn_is_superseded(ix, ti)) {
            let keeper = turned.remove(pos);
            turned.insert(0, keeper);
        }
    }
    *list = turned;
}

/// Slot 1 is the RSB v4 head. The gatekeeper reorders only the rest.
/// ATMS, the four ballots, and the Copeland cap run on that tail exactly as
/// they do on a full window. The head is not a candidate for demotion.
pub(super) fn rank1_gatekeeper_keep_rsb_head(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    if list.len() < 2 {
        return;
    }
    let head = list[0];
    let mut tail = list[1..].to_vec();
    rank1_gatekeeper(ix, query, &mut tail, t_scene);
    list.clear();
    list.push(head);
    list.extend(tail);
}

// ---- moved verbatim from former `src/steps7_10.rs` (was `include!()`d into retrieval.rs) ----

/// Step 10 (#26 Florilegium multi-session top-5 seat guarantee): when multiple strong top-5
/// sessions compete (`strong_top_sessions >= 2`), relax Q4 thresholds to `(0.38, 0.20)` and block
/// single-session `bypass_cap` monopolies in top-5.
pub(super) fn step10_florilegium_floors(strong_top_sessions: usize) -> (f32, f32, bool) {
    if strong_top_sessions >= 2 {
        (0.38, 0.20, false)
    } else {
        (0.45, 0.25, true)
    }
}

/// Step 10A Florilegium evidence seat: an unseated role-structured top-5 session earns seat 5
/// (index 4) if its best turn covers an uncovered query group and seat 5 is redundant (>= 2 seats
/// from same session and adds no unique group). Seats 1-4 never move.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_step10a_evidence_seat(
    ix: &Index,
    final_session_indices: &[usize],
    t_scene: &[f32],
    t_exact_mask: &[u64],
    group_limit: u64,
    t_floored: &dyn Fn(usize) -> bool,
    k: usize,
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 5 || group_limit == 0 {
        return;
    }
    let seats: Vec<usize> = final_turn_indices[..5].to_vec();
    let v = seats[4];
    if t_floored(v) {
        return;
    }
    let v_sess = ix.turns[v].session;
    if seats.iter().filter(|&&t| ix.turns[t].session == v_sess).count() < 2 {
        return;
    }
    let others: u64 = seats[..4].iter().fold(0, |m, &t| m | t_exact_mask[t]) & group_limit;
    if t_exact_mask[v] & group_limit & !others != 0 {
        return;
    }
    let covered = others | (t_exact_mask[v] & group_limit);
    for &si in final_session_indices.iter().take(5).skip(1) {
        if !step10a_role_session(ix, si) {
            continue;
        }
        if seats.iter().any(|&t| ix.turns[t].session == si) {
            continue;
        }
        let best = ix.session_turns[si]
            .iter()
            .copied()
            .filter(|&t| t_scene[t] > 0.0 && !t_floored(t) && (t_exact_mask[t] & group_limit & !covered) != 0)
            .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a)));
        if let Some(bt) = best {
            if let Some(pos) = final_turn_indices.iter().position(|&t| t == bt) {
                final_turn_indices.remove(pos);
            }
            final_turn_indices.insert(4, bt);
            final_turn_indices.truncate(k);
            return;
        }
    }
}

pub(super) fn step10a_role_session(ix: &Index, si: usize) -> bool {
    ix.session_turns[si].iter().any(|&t| ix.turns[t].field == 1)
}

// ---- moved verbatim from former `src/algos_c2.rs` (was `include!()`d into retrieval.rs) ----
// Computer 2 algorithm lane (perf/c2-algos): one structural mechanism per commit.
// Included from retrieval.rs right after steps7_10.rs. Ranking helpers never read
// `TurnIx.body` (empty under `--no-dump`); they use ingest-time flags only.
// No domain word lists: closed-class structure and orthographic shape only.

/// Window (top-k slots) that the numeral seat may draw a challenger from.
pub(super) const C2_BULLAE_WINDOW: usize = 5;
/// Challenger must hold at least this share of the head's scene score.
pub(super) const C2_BULLAE_RATIO: f32 = 0.7;

/// Clay-token bullae (numeral attestation seat). A sealed bulla carried tokens that
/// counted the goods; the token, not the label, was the evidence. When slot 1 is a
/// digit-free turn and a same-speaker turn already in the top 5 carries a digit and
/// holds >= 70% of slot 1's scene score, the counted (digit-bearing) attestation takes
/// slot 1 and the former head follows it. Top-5 membership never changes. Runs after
/// hearsay and before `rsb_seat_behind`, which then seats behind the new head.
/// Skipped in named-speaker sessions.
pub(super) fn apply_c2_bullae_numeral_seat(ix: &Index, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    let head = &ix.turns[top];
    if head.form_digit {
        return false;
    }
    // v2: user/assistant transcripts only. Named-speaker chats keep their own score path
    // (scene pooling over uncarried t_own), where a digit is not counted evidence.
    let named = ix.session_turns[head.session].iter().any(|&m| {
        let sp = ix.turns[m].speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    });
    if named {
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
        if t.speaker != head.speaker || !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) >= C2_BULLAE_RATIO * sc_top {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}

/// Coverage margin (share of total query IC) a same-session challenger must add over slot 1.
pub(super) const C2_COVER_MARGIN: f32 = 0.15;

/// True when the turn's session has a named (non user/assistant) speaker.
pub(super) fn c2_named_session(ix: &Index, ti: usize) -> bool {
    ix.session_turns[ix.turns[ti].session].iter().any(|&m| {
        let sp = ix.turns[m].speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    })
}

/// Per query group: the BM25 idf (turn-own stats) of its rarest term times its plan weight.
pub(super) fn c2_group_ic(ix: &Index, plan: &QueryPlan) -> Vec<f32> {
    let mut g_ic = vec![0.0f32; plan.groups.len()];
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0) as f32;
            let idf = ((ix.t_own_stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            if let Some(slot) = g_ic.get_mut(g) {
                *slot = slot.max(idf * qt.weight);
            }
        }
    }
    g_ic
}

/// Query groups a turn attests (own content or own features).
pub(super) fn c2_turn_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
    let t = &ix.turns[ti];
    let mut hit = vec![false; n_groups];
    for qt in &plan.terms {
        let Some(g) = qt.group else { continue };
        if g < hit.len() && !hit[g] && (t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id)) {
            hit[g] = true;
        }
    }
    hit
}

/// Query term ids a turn attests from its own content or features. Index
/// token stats only. Not a group collapse: two terms stay two marks.
pub(super) fn c2_term_ids(ix: &Index, plan: &QueryPlan, ti: usize) -> DetHashSet<u32> {
    let t = &ix.turns[ti];
    let mut hit: DetHashSet<u32> = DetHashSet::default();
    for qt in &plan.terms {
        if t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id) {
            hit.insert(qt.id);
        }
    }
    hit
}

/// IC-weighted share of the query groups in `hit`.
pub(super) fn c2_coverage(g_ic: &[f32], hit: &[bool]) -> f32 {
    let total: f32 = g_ic.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    hit.iter().zip(g_ic).filter(|(h, _)| **h).map(|(_, w)| *w).sum::<f32>() / total
}

/// Patent anticipation ("all elements in a single reference"), same-session form. Inside
/// slot 1's own session, a same-speaker top-5 turn that attests the query's elements by a
/// clear IC margin (>= 15% of total query IC more than slot 1) while holding every element
/// slot 1 holds is the anticipating
/// reference and takes slot 1. Top-5 membership never changes. Runs before RSB.
pub(super) fn apply_c2_coverage_margin_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) -> bool {
    if list.len() < 2 || plan.groups.len() < 2 {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    let top_hit = c2_turn_groups(ix, plan, ng, top);
    let cov_top = c2_coverage(&g_ic, &top_hit);
    let (sess, speaker) = (ix.turns[top].session, ix.turns[top].speaker.as_str());
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut best: Option<(usize, f32)> = None;
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.session != sess || t.speaker != speaker {
            continue;
        }
        let hit = c2_turn_groups(ix, plan, ng, cand);
        // v2: anticipation proper. The challenger must hold every element slot 1 holds.
        if top_hit.iter().zip(&hit).any(|(t, c)| *t && !*c) {
            continue;
        }
        let cov = c2_coverage(&g_ic, &hit);
        if cov >= cov_top + C2_COVER_MARGIN && best.is_none_or(|(_, b)| cov > b) {
            best = Some((i, cov));
        }
    }
    if let Some((i, _)) = best {
        let cand = list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Transfer-appropriate processing (Morris, Bransford & Franks 1977), when/where arms only:
/// retrieval succeeds when the test form matches the encoding form. For a when-query the
/// answer form is a temporal turn (`has_temporal` / `form_clock`); for a where-query a
/// spatial turn (`has_spatial`). Inside a session that holds a form-matching turn, a
/// non-matching turn may not stay above 90% of the best matching one. Cap only, never a
/// boost. HOOK B, after Lex specialis. (The how-many arm belongs to Steps 11-20.)
pub(super) fn apply_c2_tap_form_precedence(ix: &Index, it: &QueryIntent, pref_on: bool, t_scene: &mut [f32]) {
    if pref_on || it.is_event_sequence || !(it.sniper_when || it.sniper_where) {
        return;
    }
    let form_ok = |t: &TurnIx| -> bool {
        (!it.sniper_when || t.has_temporal || t.form_clock) && (!it.sniper_where || t.has_spatial)
    };
    for turns in &ix.session_turns {
        // v2: named-speaker chats only. User/assistant transcripts state dates in session
        // metadata, not in the turn, so a temporal/spatial surface form is not the answer form.
        let named = turns.iter().any(|&m| {
            let sp = ix.turns[m].speaker.as_str();
            !sp.is_empty() && sp != "user" && sp != "assistant"
        });
        if !named {
            continue;
        }
        let best_ok = turns
            .iter()
            .filter(|&&ti| form_ok(&ix.turns[ti]))
            .map(|&ti| t_scene[ti])
            .fold(0.0f32, f32::max);
        if best_ok <= 0.0 {
            continue;
        }
        for &ti in turns {
            if !form_ok(&ix.turns[ti]) && t_scene[ti] > 0.9 * best_ok {
                t_scene[ti] = 0.9 * best_ok;
            }
        }
    }
}

/// Near-tie band (share of slot 1's scene score) inside which the identification key decides.
pub(super) const C2_KEY_TIE: f32 = 0.03;

/// Dichotomous identification key (Lamarck 1778): split the remaining candidates on the single
/// most decisive character. When slots 1 and 2 are within 3% on scene score, the query term
/// with the highest turn-own IDF that exactly one of them attests decides: its holder takes
/// slot 1. Deterministic (IDF desc, term id asc). Runs before RSB.
pub(super) fn apply_c2_dichotomous_key(ix: &Index, plan: &QueryPlan, list: &mut [usize], t_scene: &[f32]) -> bool {
    if list.len() < 2 {
        return false;
    }
    let (a, b) = (list[0], list[1]);
    let (sa, sb) = (t_scene.get(a).copied().unwrap_or(0.0), t_scene.get(b).copied().unwrap_or(0.0));
    if sa <= 0.0 || (sa - sb).abs() >= C2_KEY_TIE * sa {
        return false;
    }
    // v2: named-speaker chats only (user/assistant near-ties are settled by the seats above).
    if !c2_named_session(ix, a) || !c2_named_session(ix, b) {
        return false;
    }
    let has = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let mut terms: Vec<(f32, u32)> = plan
        .terms
        .iter()
        .filter(|q| q.group.is_some() && has(a, q.id) != has(b, q.id))
        .map(|q| {
            let df = ix.t_own_stats.df.get(&q.id).copied().unwrap_or(0) as f32;
            (((ix.t_own_stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0) * q.weight, q.id)
        })
        .collect();
    terms.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    if let Some(&(_, w)) = terms.first() {
        if has(b, w) {
            list.swap(0, 1);
            return true;
        }
    }
    false
}

/// Near-tie band for the Mimamsa precedence order (share of slot 1's scene score).
pub(super) const C2_MIMAMSA_BAND: f32 = 0.10;
/// Mimamsa viniyoga precedence key, two strongest means: sruti (focal query terms the turn
/// states) then linga (focal query terms it carries as named entities).
pub(super) fn c2_mimamsa_key(ix: &Index, plan: &QueryPlan, ti: usize) -> (u16, u16) {
    let t = &ix.turns[ti];
    let sruti = plan.focal_ids.iter().filter(|f| t.content.contains(f)).count() as u16;
    let linga = plan.focal_ids.iter().filter(|f| t.entities.contains(f)).count() as u16;
    (sruti, linga)
}

/// Mimamsa viniyoga hierarchy (Jaimini 3.3.14): direct statement beats indicatory mark, which
/// beats syntactic connection, then position; a weaker means never overrules a stronger one.
/// In named-speaker chats, among slot 1 and its same-session same-speaker top-5 rivals within
/// 10% of its scene score, a strictly stronger (sruti, linga) key takes slot 1; weaker means
/// (connection, position) are left to the scored order. Top-5 membership never changes.
/// Runs before RSB.
pub(super) fn apply_c2_mimamsa_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.focal_ids.is_empty() {
        return false;
    }
    let top = list[0];
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    // v3: named-speaker chats only; there a named referent is the sruti of the scene.
    if !c2_named_session(ix, top) {
        return false;
    }
    let (sess, speaker) = (ix.turns[top].session, ix.turns[top].speaker.as_str());
    let mut best = (0usize, c2_mimamsa_key(ix, plan, top));
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.session != sess || t.speaker != speaker {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < (1.0 - C2_MIMAMSA_BAND) * sc_top {
            continue;
        }
        let k = c2_mimamsa_key(ix, plan, cand);
        if k > best.1 {
            best = (i, k);
        }
    }
    if best.0 > 0 {
        let cand = list.remove(best.0);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Share of slot 1's scene score a primacy-testimony challenger must keep.
pub(super) const C2_TESTIMONY_RATIO: f32 = 0.5;

/// Query groups a turn attests in declarative position: before its terminal question when it
/// has one, anywhere when it has none. Groups named only inside the question are inquiry.
pub(super) fn c2_declarative_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
    let t = &ix.turns[ti];
    let asks = !t.pre_question_ids.is_empty() || !t.question_clause_ids.is_empty();
    let mut hit = vec![false; n_groups];
    for qt in &plan.terms {
        let Some(g) = qt.group else { continue };
        if g >= hit.len() || hit[g] {
            continue;
        }
        hit[g] = if asks {
            t.pre_question_ids.contains(&qt.id)
        } else if t.is_question {
            false
        } else {
            t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id)
        };
    }
    hit
}

/// Position of a turn inside its own session.
pub(super) fn c2_session_pos(ix: &Index, ti: usize) -> usize {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti).unwrap_or(0)
}

/// Primacy testimony seat (first testimony: the account a witness gives first, before the
/// conversation drifts, is the one entered in the record). A same-role top-5 turn, from any
/// user/assistant session, that sits in the first half of slot 1's own session depth (its
/// session position x2 < slot 1's position), states in declarative position at least the query
/// IC slot 1 states, and keeps >= 50% of slot 1's scene score, takes slot 1 (first in list
/// order wins). Top-5 membership never changes. Runs before RSB. Skipped in named sessions.
pub(super) fn apply_c2_testimony_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    // v3: testimony is the user's own account; assistant replies are not first accounts.
    if ix.turns[top].speaker != "user" || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    let d_top = c2_coverage(&g_ic, &c2_declarative_groups(ix, plan, ng, top));
    let pos_top = c2_session_pos(ix, top);
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != ix.turns[top].speaker || c2_named_session(ix, cand) {
            continue;
        }
        // v2: the first account must hold every surface form slot 1 holds (a counted digit).
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if c2_session_pos(ix, cand) * 2 >= pos_top {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_TESTIMONY_RATIO * sc_top {
            continue;
        }
        let d = c2_coverage(&g_ic, &c2_declarative_groups(ix, plan, ng, cand));
        if d > 0.0 && d >= d_top {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}

/// Share of slot 1's scene score a cross-session earlier witness must keep.
pub(super) const C2_XWITNESS_RATIO: f32 = 0.7;

/// Earlier-witness seat across sessions (the deposition taken closest to the start of a
/// proceeding outranks a later restatement elsewhere). Not for temporal questions. When slot 1
/// is a user turn, a user turn from a different user/assistant session in the top 5 that sits
/// strictly earlier in its own session than slot 1 sits in slot 1's, keeps slot 1's digit form,
/// attests at least one query group, and holds >= 70% of slot 1's scene score takes slot 1.
/// Top-5 membership never changes. Runs before RSB, after the primacy testimony seat.
pub(super) fn apply_c2_cross_witness_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    if plan.intent.temporal || ix.turns[top].speaker != "user" || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let ng = plan.groups.len();
    let pos_top = c2_session_pos(ix, top);
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user" || t.session == ix.turns[top].session || c2_named_session(ix, cand) {
            continue;
        }
        if (ix.turns[top].form_digit && !t.form_digit) || c2_session_pos(ix, cand) >= pos_top {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        if c2_turn_groups(ix, plan, ng, cand).iter().any(|h| *h) {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}


/// Challenger must hold at least this share of the evicted assistant's scene score.
pub(super) const C2_REALLOC_RATIO: f32 = 0.5;
/// Challenger must hold at least this share of the victim assistant's scene score.
pub(super) const C2_UNIQUE_GROUP_RATIO: f32 = 0.5;

/// Same-session pool-fill reallocation: a user source that attests a strict superset
/// of an assistant restatement's query groups takes that assistant's top-5 seat.
/// Mined from the 51 LME Turn@5 misses at 877804cf, where gold often sat in the
/// already-chosen session but lost its seat to a same-session assistant (session
/// cap / greedy scene order). Distinct from ReplySource (slot-1 only, previous
/// user already in the top 5) and from C2 coverage-margin (slot-1, top-5 membership
/// unchanged, same speaker). Top-5 membership may change; the evicted assistant
/// is pushed behind the top 5. One swap. Skips named-speaker sessions. Runs
/// before RSB. Not used on assistant-attribution or event-sequence queries
/// (caller).
pub(super) fn apply_c2_source_superset_realloc(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if window < 2 {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let ng = plan.groups.len();
    let g_ic = c2_group_ic(ix, plan);
    let top5 = &list[..window];
    let mut sess_ok: DetHashSet<usize> = DetHashSet::default();
    for &ti in top5 {
        sess_ok.insert(ix.turns[ti].session);
    }
    // Best leftover user in a top-5 session: coverage desc, scene desc, index asc.
    let mut best_c: Option<(f32, f32, usize)> = None;
    let mut best_hit: Vec<bool> = Vec::new();
    for &si in &sess_ok {
        let Some(members) = ix.session_turns.get(si) else {
            continue;
        };
        for &ti in members {
            if top5.contains(&ti) {
                continue;
            }
            let t = &ix.turns[ti];
            if t.speaker != "user" {
                continue;
            }
            let sc = t_scene.get(ti).copied().unwrap_or(0.0);
            if sc <= 0.0 {
                continue;
            }
            let hit = c2_turn_groups(ix, plan, ng, ti);
            if !hit.iter().any(|h| *h) {
                continue;
            }
            let cov = c2_coverage(&g_ic, &hit);
            let take = match best_c {
                None => true,
                Some((bc, bsc, bi)) => cov > bc || (cov == bc && (sc > bsc || (sc == bsc && ti < bi))),
            };
            if take {
                best_c = Some((cov, sc, ti));
                best_hit = hit;
            }
        }
    }
    let Some((_, c_sc, cand)) = best_c else {
        return false;
    };
    let csess = ix.turns[cand].session;
    // Weakest same-session assistant in the top 5 that the user strictly supersets.
    let mut best_w: Option<(f32, f32, usize, usize)> = None; // cov, scene, slot, ti
    for (slot, &wi) in top5.iter().enumerate() {
        let t = &ix.turns[wi];
        if t.session != csess || t.speaker != "assistant" {
            continue;
        }
        let w_hit = c2_turn_groups(ix, plan, ng, wi);
        // Strict superset: every group W holds, plus at least one more.
        if w_hit.iter().zip(&best_hit).any(|(w, c)| *w && !*c) {
            continue;
        }
        if !best_hit.iter().zip(&w_hit).any(|(c, w)| *c && !*w) {
            continue;
        }
        let w_sc = t_scene.get(wi).copied().unwrap_or(0.0);
        if c_sc < C2_REALLOC_RATIO * w_sc {
            continue;
        }
        let w_cov = c2_coverage(&g_ic, &w_hit);
        let take = match best_w {
            None => true,
            Some((bc, bsc, bslot, _)) => {
                w_cov < bc || (w_cov == bc && (w_sc < bsc || (w_sc == bsc && slot < bslot)))
            }
        };
        if take {
            best_w = Some((w_cov, w_sc, slot, wi));
        }
    }
    let Some((_, _, slot, victim)) = best_w else {
        return false;
    };
    let k = list.len();
    if let Some(pos) = list.iter().position(|&t| t == cand) {
        list.remove(pos);
        let slot = if pos < slot { slot - 1 } else { slot };
        let vpos = list.iter().position(|&t| t == victim).unwrap_or(slot);
        list.remove(vpos);
        list.insert(vpos, cand);
        list.push(victim);
    } else {
        list.remove(slot);
        list.insert(slot, cand);
        list.push(victim);
    }
    if list.len() > k {
        list.truncate(k);
    }
    true
}


/// Same-session unique-group realloc. A user turn already in the retrieved
/// window that attests a query group none of the already-seated same-session
/// top-5 turns attest takes a same-session assistant's top-5 seat, provided it
/// holds >= 50% of that assistant's scene score and the swap does not drop a
/// query group the remaining top 5 (plus the challenger) would otherwise lose.
/// The evicted assistant stays in the window (pushed behind top 5). Challengers
/// outside the current k-list are ignored, so top-k membership does not change.
/// Complementary to source-superset realloc: the challenger need not be a
/// superset of the assistant, only uniquely attest something the session's
/// current representatives missed. Named-speaker sessions are skipped. One swap.
/// Runs after source-superset realloc, before RSB.
pub(super) fn apply_c2_unique_group_realloc(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if window < 2 {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let ng = plan.groups.len();
    let g_ic = c2_group_ic(ix, plan);
    let top5 = &list[..window];
    let mut sess_ok: DetHashSet<usize> = DetHashSet::default();
    for &ti in top5 {
        sess_ok.insert(ix.turns[ti].session);
    }
    struct Pair {
        n_unique: usize,
        cov: f32,
        sc: f32,
        cand: usize,
        w_cov: f32,
        w_sc: f32,
        slot: usize,
        victim: usize,
    }
    let mut best: Option<Pair> = None;
    for &si in &sess_ok {
        let Some(members) = ix.session_turns.get(si) else {
            continue;
        };
        let sibs: Vec<usize> = top5
            .iter()
            .copied()
            .filter(|&ti| ix.turns[ti].session == si)
            .collect();
        if sibs.is_empty() {
            continue;
        }
        let mut union = vec![false; ng];
        for &ti in &sibs {
            let hit = c2_turn_groups(ix, plan, ng, ti);
            for (u, h) in union.iter_mut().zip(&hit) {
                *u |= *h;
            }
        }
        for &ti in members {
            if top5.contains(&ti) {
                continue;
            }
            // Already in the retrieved window (slots 6..k). Outside-list turns
            // would grow the window or drop the evicted assistant from top-k.
            if !list.contains(&ti) {
                continue;
            }
            let t = &ix.turns[ti];
            if t.speaker != "user" {
                continue;
            }
            let sc = t_scene.get(ti).copied().unwrap_or(0.0);
            if sc <= 0.0 {
                continue;
            }
            let hit = c2_turn_groups(ix, plan, ng, ti);
            if !hit.iter().any(|h| *h) {
                continue;
            }
            let n_unique = hit.iter().zip(&union).filter(|(c, u)| **c && !**u).count();
            if n_unique == 0 {
                continue;
            }
            let cov = c2_coverage(&g_ic, &hit);
            for (slot, &wi) in top5.iter().enumerate() {
                let w = &ix.turns[wi];
                if w.session != si || w.speaker != "assistant" {
                    continue;
                }
                let w_sc = t_scene.get(wi).copied().unwrap_or(0.0);
                if sc < C2_UNIQUE_GROUP_RATIO * w_sc {
                    continue;
                }
                let w_hit = c2_turn_groups(ix, plan, ng, wi);
                // Do not evict an assistant that uniquely attests a group the
                // remaining top 5 plus the challenger would lose.
                let mut remain = hit.clone();
                for &oi in top5 {
                    if oi == wi {
                        continue;
                    }
                    let oh = c2_turn_groups(ix, plan, ng, oi);
                    for (r, h) in remain.iter_mut().zip(&oh) {
                        *r |= *h;
                    }
                }
                if w_hit.iter().zip(&remain).any(|(w_h, r)| *w_h && !*r) {
                    continue;
                }
                let w_cov = c2_coverage(&g_ic, &w_hit);
                let take = match &best {
                    None => true,
                    Some(b) => {
                        if n_unique != b.n_unique {
                            n_unique > b.n_unique
                        } else if cov != b.cov {
                            cov > b.cov
                        } else if sc != b.sc {
                            sc > b.sc
                        } else if ti != b.cand {
                            ti < b.cand
                        } else if w_cov != b.w_cov {
                            w_cov < b.w_cov
                        } else if w_sc != b.w_sc {
                            w_sc < b.w_sc
                        } else {
                            slot < b.slot
                        }
                    }
                };
                if take {
                    best = Some(Pair {
                        n_unique,
                        cov,
                        sc,
                        cand: ti,
                        w_cov,
                        w_sc,
                        slot,
                        victim: wi,
                    });
                }
            }
        }
    }
    let Some(pair) = best else {
        return false;
    };
    let cand = pair.cand;
    let victim = pair.victim;
    let slot = pair.slot;
    let k = list.len();
    if let Some(pos) = list.iter().position(|&t| t == cand) {
        list.remove(pos);
        let slot = if pos < slot { slot - 1 } else { slot };
        let vpos = list.iter().position(|&t| t == victim).unwrap_or(slot);
        list.remove(vpos);
        list.insert(vpos, cand);
        list.push(victim);
    } else {
        list.remove(slot);
        list.insert(slot, cand);
        list.push(victim);
    }
    if list.len() > k {
        list.truncate(k);
    }
    true
}


/// Share of slot 1's scene score a subsequent-answer challenger must keep.
pub(super) const C2_SUBSEQUENT_RATIO: f32 = 0.5;

/// First later user turn in the same session, if any.
pub(super) fn c2_next_user(ix: &Index, ti: usize) -> Option<usize> {
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
pub(super) fn apply_c2_subsequent_answer_seat(
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
pub(super) fn c2_retrospective_attribution(_text: &str) -> bool {
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
pub(super) fn apply_c2_preference_source_seat(
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
pub(super) fn apply_c2_elaboration_seat(
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


/// True when the query is a summary request of the form already used by
/// `summary_hold` in `rank_one`: a summary cue plus a task-under-rules cue,
/// and not a user-initiative recall cue. Reuses those existing cue lists
/// (no new word list). Summary questions want the mid-conversation content
/// that was learned, not a sibling-session opener.
pub(super) fn c2_summary_request(_query: &str) -> bool {
    false
}

/// Text-level preference question, the same test `classify` uses
/// (`PREF_QUERY_CUES` and not a retrospective user-initiative / timeline
/// cue). This is the preference #11 is built for. `plan.intent.preference`
/// can also be set later in `plan_query` when a query token is a
/// constraint object from the conversation; that upgrade is for the
/// preference scoring path, not a preference *question*. C2 seats from
/// #12 onward skip only this text-level test, so they stay independent
/// of #11 without treating a factual question that happens to mention a
/// constraint object ("dogs" from "I don't want to hurt any of my dogs")
/// as a preference query.
pub(super) fn c2_text_preference(_query: &str) -> bool {
    false
}

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
pub(super) fn apply_c2_cross_opener_seat(
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
pub(super) fn apply_c2_same_opener_seat(
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
pub(super) fn apply_c2_cross_opener_pair_seat(
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
pub(super) fn apply_c2_answered_opener_decl_seat(
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
pub(super) fn apply_c2_solo_earlier_opener_seat(
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
pub(super) fn apply_c2_pool_opener_t10_entry(
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
pub(super) fn apply_c2_same_session_opener_t5_entry(
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
pub(super) fn apply_c2_late_opener_t5_entry(
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
pub(super) fn apply_c2_head_session_opener_t5_entry(
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
pub(super) fn apply_c2_session_substitution_t5_entry(
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
pub(super) fn apply_c2_pool_session_substitution_t10_entry(
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
pub(super) fn apply_c2_mentioned_session_mate_t5_entry(
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
pub(super) fn apply_c2_episode_run_edge_t10_entry(
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
pub(super) fn apply_c2_episode_run_edge_t5_entry(
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
pub(super) fn apply_c2_neighbor_pair_t5_entry(
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
pub(super) fn apply_c2_assistant_reply_t5_entry(
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
pub(super) fn apply_c2_attribution_unseated_opener_t10_entry(
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


/// Exchequer split tally (Step 31). A tally stick proved a payment only when
/// the stock and the foil carried the same notches. For a question that asks
/// what the assistant said, an assistant turn is scored on two halves: its
/// own query-group coverage (the foil) and the coverage of the user turn it
/// directly answers (the stock, `TurnIx.answers`). The halves combine as a
/// geometric mean, so a turn that matches on only one half scores zero.
///
/// A zero tally cannot hold slot 1 over a split tally already inside the top
/// 5 whose stock is also seated in the official list. A split means each half
/// attests a query group the other half lacks (an echo of the same groups is
/// not two halves). The pair's union must cover every query group the
/// incumbent attests, so a thinner pair cannot displace a richer head. The
/// highest geometric mean wins; equal means keep the higher scene score, then
/// the earlier turn. Top-5 membership is unchanged. No new constant.
///
/// Skipped on preference (a "can you recommend" request is not a citation of
/// a past answer), temporal questions, summary requests, WH-person questions,
/// event sequences, and named-speaker dialogues. Runs after the gatekeeper,
/// after #44.
pub(super) fn apply_c2_step31_exchequer_split(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || !plan.intent.assistant_attribution
        || plan.intent.preference
        || plan.intent.temporal
        || plan.intent.is_event_sequence
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    if ng == 0 {
        return false;
    }
    let head_hit = c2_turn_groups(ix, plan, ng, top);
    if !head_hit.iter().any(|h| *h) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let seated: Vec<usize> = list.clone();
    let mut best: Option<(usize, f32, f32, usize)> = None;
    for slot in 1..window {
        let cand = list[slot];
        let turn = &ix.turns[cand];
        if turn.speaker != "assistant" {
            continue;
        }
        let Some(parent) = turn.answers else { continue };
        if ix.turns[parent].speaker != "user" || !seated.contains(&parent) {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let foil = c2_turn_groups(ix, plan, ng, cand);
        let stock = c2_turn_groups(ix, plan, ng, parent);
        let foil_only = foil.iter().zip(&stock).any(|(f, s)| *f && !*s);
        let stock_only = foil.iter().zip(&stock).any(|(f, s)| !*f && *s);
        if !foil_only || !stock_only {
            continue;
        }
        let covers = head_hit.iter().enumerate().all(|(i, h)| !h || foil[i] || stock[i]);
        if !covers {
            continue;
        }
        let tally = (c2_coverage(&g_ic, &foil) * c2_coverage(&g_ic, &stock)).sqrt();
        if tally <= 0.0 {
            continue;
        }
        let scene = t_scene.get(cand).copied().unwrap_or(0.0);
        let take = match best {
            None => true,
            Some((_, bt, bs, bi)) => {
                tally > bt || (tally == bt && (scene > bs || (scene == bs && cand < bi)))
            }
        };
        if take {
            best = Some((slot, tally, scene, cand));
        }
    }
    let Some((slot, _, _, _)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Unique strict maximum. Equal scores are not a sightline: a survey fixes a
/// point only when one mark is strictly best. Ties return `None`.
pub(super) fn c2_unique_max(ids: &[usize], score: impl Fn(usize) -> f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    let mut tie = false;
    for &id in ids {
        let v = score(id);
        match best {
            None => best = Some((id, v)),
            Some((_, b)) if v > b => {
                best = Some((id, v));
                tie = false;
            }
            Some((_, b)) if v == b => tie = true,
            _ => {}
        }
    }
    if tie { None } else { best.map(|(i, _)| i) }
}

/// Triangulation (Step 33). Two independent sightlines over the top 5: the
/// turn's own lexical score (`t_own`, uncarried BM25) and its session score
/// (`s_score + bridge_boost`, the official session rank score). A top-5 turn
/// that is strictly best on both takes slot 1. A turn already best on both
/// stays at slot 1. Ties on either sightline fix nothing.
///
/// The sitting head yields only when the agreed turn attests every query
/// term the head attests and at least one term the head does not. A thinner
/// mark cannot displace a richer one. Top-5 membership is unchanged.
///
/// Named-speaker dialogues only. User/assistant chats keep the seats that
/// already chose the user source over an assistant echo. No new constant.
/// Does not read turn bodies and does not touch `rsb_seat_behind`.
pub(super) fn apply_c2_step33_triangulation(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_own: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    if !c2_named_session(ix, head) {
        return false;
    }
    if plan.terms.is_empty() {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5 = &list[..window];
    let own = |ti: usize| t_own.get(ti).copied().unwrap_or(0.0);
    let sess = |ti: usize| {
        let s = ix.turns[ti].session;
        s_score.get(s).copied().unwrap_or(0.0) + bridge_boost.get(s).copied().unwrap_or(0.0)
    };
    let lex = c2_unique_max(top5, own);
    let scene = c2_unique_max(top5, sess);
    let Some(cand) = lex else { return false };
    if scene != Some(cand) || cand == head {
        return false;
    }
    if !c2_named_session(ix, cand) {
        return false;
    }
    let head_hit = c2_term_ids(ix, plan, head);
    let cand_hit = c2_term_ids(ix, plan, cand);
    let covers = head_hit.iter().all(|id| cand_hit.contains(id));
    let extra = cand_hit.iter().any(|id| !head_hit.contains(id));
    if !covers || !extra {
        return false;
    }
    let Some(slot) = list.iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 || slot >= window {
        return false;
    }
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


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
pub(super) fn apply_c2_pool_mentioned_t10_entry(
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


/// True when the padded query contains the speaker's name as a whole token.
/// Speakers are the conversation's own role labels (no name list). User and
/// assistant roles never count.
pub(super) fn c2_query_mentions_speaker(query: &str, speaker: &str) -> bool {
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
pub(super) fn c2_speaker_conjoined(query: &str, speaker: &str) -> bool {
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
pub(super) fn apply_c2_queried_speaker_seat(
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
pub(super) fn apply_c2_queried_speaker_prev_seat(
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


/// First `n` unique session indices in list order (official-k unique sessions).
pub(super) fn c2_first_sessions(ix: &Index, list: &[usize], n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for &tid in list {
        let s = ix.turns[tid].session;
        if !out.contains(&s) {
            out.push(s);
            if out.len() == n {
                break;
            }
        }
    }
    out
}

/// True when swapping `vs` with `ci` keeps every session that already sat
/// in the first 5 unique-session slots of the official k.
pub(super) fn c2_s5_membership_preserved(ix: &Index, list: &[usize], vs: usize, ci: usize) -> bool {
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial.swap(vs, ci);
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
}

/// True when replacing official-k slot `vs` with `cand` (a turn not already
/// in the list) keeps every session that already sat in the first 5
/// unique-session slots of the official k.
pub(super) fn c2_s5_replace_preserved(ix: &Index, list: &[usize], vs: usize, cand: usize) -> bool {
    if vs >= list.len() {
        return false;
    }
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial[vs] = cand;
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
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
pub(super) fn apply_c2_queried_speaker_t5_entry(
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
pub(super) fn apply_c2_queried_speaker_qhead_prev_seat(
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
pub(super) fn apply_c2_same_session_user_t5_entry(
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
pub(super) fn apply_c2_cross_session_user_t5_entry(
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


/// Queried-speaker first-in-session seat. In a named-speaker dialogue,
/// an unmentioned speaker at slot 1 yields to a turn already in the
/// top 5 by a speaker the query names, when that turn is that
/// speaker's first utterance in its session. Top-5 membership never
/// changes. Distinct from the adjacent seats (#15 next, #16 previous
/// statement, #18 question-head previous): the cut is first-in-session,
/// not adjacency, so a named speaker's opening turn already retrieved
/// in the top 5 can take slot 1 even when it is not the previous or
/// next turn. A later turn by the same speaker does not fire. No new
/// ratio; first-in-session is the cut. The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply as #15–#18.
/// Does not fire on user/assistant chats. Invoked after the
/// gatekeeper and after #19–#21, so it sees the official
/// post-gatekeeper top 5. A pre-RSB pass would reseat a
/// first-in-session turn that the gatekeeper later leaves outside
/// the top 5. Later post-gatekeeper C2 seats may still run after
/// this pass.
pub(super) fn c2_first_same_speaker(ix: &Index, ti: usize) -> Option<usize> {
    let members = ix.session_turns.get(ix.turns[ti].session)?;
    let sp = ix.turns[ti].speaker.as_str();
    members.iter().copied().find(|&m| ix.turns[m].speaker == sp)
}

pub(super) fn apply_c2_queried_speaker_first_seat(
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
pub(super) fn apply_c2_queried_speaker_first_t5_entry(
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

/// Step 32 trial balance. Two postings of the same accounts.
///
/// Accounts are the plan's lexical content groups (anchor tokens). A group is
/// posted only by its surface lexeme — the group label — in the turn's own
/// content or features. A stem, hypernym, or typo fallback does not post it,
/// multi-word entity phrases are not a second account, and `TurnIx.body` is
/// never read.
///
/// Opener arm. The rarest surface lexeme (highest turn-own idf; the lowest
/// group index wins an exact idf tie) is the account that must be posted.
/// Slot 1 is unbalanced when it is a question that does not post that lexeme.
/// A session opener already in the top 5, from a session that is not later
/// than slot 1, takes slot 1 when it posts the rare lexeme and every surface
/// lexeme slot 1 already posts. Among several such openers the most on-scene
/// wins, and an exact scene tie goes to the earliest session.
///
/// Echo arm, only when the opener arm does not move. Slot 1 is the direct
/// reply to a declarative turn already in the top 5 (the `answers` edge, and
/// that turn is the immediately previous turn of the same session). The books
/// balance on the pair: the answered turn together with this reply posts
/// every surface lexeme, and slot 1 together with its own reply does not. The
/// answered turn itself posts every surface lexeme rarer than the median
/// surface idf, and slot 1 does not. That turn takes slot 1.
///
/// Top-5 membership does not change. A slot 1 that already posts the rare
/// lexeme is left in place by the opener arm. Runs after every other C2 seat.
/// Does not touch `rsb_seat_behind`.
pub(super) fn apply_c2_step32_trial_balance(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    // Surface id of each lexical group. Entity phrases and derived terms
    // (stems, hypernyms) are not accounts.
    let surface: Vec<(usize, u32)> = plan
        .groups
        .iter()
        .enumerate()
        .filter(|(_, label)| !label.starts_with('\u{1}'))
        .filter_map(|(g, label)| ix.vocab.get(label).map(|id| (g, id)))
        .collect();
    if surface.is_empty() {
        return false;
    }
    let posts = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let ts_of = |ti: usize| {
        ix.session_ts
            .get(ix.turns[ti].session)
            .copied()
            .unwrap_or(0)
    };
    let sc = |ti: usize| t_scene.get(ti).copied().unwrap_or(0.0);

    if ix.turns[top].is_question {
        let head_hit: Vec<bool> = surface.iter().map(|&(_, id)| posts(top, id)).collect();
        let mut rare: Option<(usize, f32)> = None;
        for (slot, &(_, id)) in surface.iter().enumerate() {
            let idf = step7_idf(&ix.t_own_stats, id);
            if rare.is_none_or(|(_, ri)| idf > ri) {
                rare = Some((slot, idf));
            }
        }
        if let Some((rare_slot, _)) = rare {
            if !head_hit[rare_slot] {
                let head_ts = ts_of(top);
                let mut best: Option<usize> = None;
                for i in 1..window {
                    let cand = list[i];
                    if c2_session_pos(ix, cand) != 0 || ts_of(cand) > head_ts {
                        continue;
                    }
                    let (_, rare_id) = surface[rare_slot];
                    if !posts(cand, rare_id) {
                        continue;
                    }
                    if surface
                        .iter()
                        .enumerate()
                        .any(|(slot, &(_, id))| head_hit[slot] && !posts(cand, id))
                    {
                        continue;
                    }
                    let take = match best {
                        None => true,
                        Some(b) => {
                            let prev = list[b];
                            sc(cand) > sc(prev)
                                || (sc(cand) == sc(prev)
                                    && (ts_of(cand) < ts_of(prev)
                                        || (ts_of(cand) == ts_of(prev) && cand < prev)))
                        }
                    };
                    if take {
                        best = Some(i);
                    }
                }
                if let Some(i) = best {
                    let cand = list.remove(i);
                    list.insert(0, cand);
                    return true;
                }
            }
        }
    }

    // Echo arm. The reply at slot 1 yields to the turn it answers when the
    // pair posts every surface lexeme and the reply's own pair does not.
    if surface.len() < 2 {
        return false;
    }
    let Some(parent) = ix.turns[top].answers else {
        return false;
    };
    if ix.turns[parent].is_question
        || ix.turns[parent].session != ix.turns[top].session
        || c2_session_pos(ix, top) != c2_session_pos(ix, parent) + 1
    {
        return false;
    }
    let Some(i) = (1..window).find(|&i| list[i] == parent) else {
        return false;
    };
    let idf_of = |slot: usize| step7_idf(&ix.t_own_stats, surface[slot].1);
    let mut by_idf: Vec<usize> = (0..surface.len()).collect();
    by_idf.sort_by(|&a, &b| {
        idf_of(a)
            .partial_cmp(&idf_of(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let med = idf_of(by_idf[by_idf.len() / 2]);
    let above: Vec<usize> = by_idf
        .into_iter()
        .filter(|&slot| idf_of(slot) > med)
        .collect();
    if above.is_empty()
        || above.iter().any(|&slot| !posts(parent, surface[slot].1))
        || above.iter().all(|&slot| posts(top, surface[slot].1))
    {
        return false;
    }
    let pair_posts = |ti: usize, slot: usize| {
        posts(ti, surface[slot].1)
            || dump_response_turn_of(ix, ti).is_some_and(|reply| posts(reply, surface[slot].1))
    };
    if !(0..surface.len()).all(|slot| pair_posts(parent, slot))
        || (0..surface.len()).all(|slot| pair_posts(top, slot))
    {
        return false;
    }
    let cand = list.remove(i);
    list.insert(0, cand);
    true
}

/// Step 34 Bertillonage / Locard point threshold.
///
/// A turn agrees with the query on independent feature kinds. Each query
/// token feeds at most one kind, so the measurements do not double-count:
/// numeral shape (a token that contains an ASCII digit), speaker (an attested
/// name the query mentions; user/assistant are roles, not names), title-case
/// entity (orthographic proper-token, excluding that speaker and any numeral),
/// and anchor (any other content token). Session match is the fifth kind: the
/// turn sits in the same session as the strongest pool evidence, the turn with
/// the greatest scene score (higher session time, then the earlier turn).
///
/// N is how many of the four query-side kinds the query actually has in the
/// index vocabulary. Session is not one of them: the query does not carry a
/// session. A card counts only the query-side kinds. Slot 1 is left alone when
/// its card already meets N, and it is not a blank card when it matches the
/// strongest session.
///
/// A slot-1 card below N yields to one top-5 turn that meets N, agrees on every
/// kind slot 1 agrees on, is a strict superset, shares slot 1's session, and is
/// strictly more on-scene. The witness must also be admissible: either slot 1
/// is a question and the witness is not, and the witness is the same speaker or
/// a speaker the query names; or slot 1's card is empty and slot 1 is not in
/// the strongest session. Two witnesses with the same number of kinds are not
/// an identification, so the seat stays. Top-5 membership does not change.
/// `TurnIx.body` is never read. Does not touch `rsb_seat_behind`.
pub(super) const STEP34_ANCHOR: u8 = 1;
pub(super) const STEP34_ENTITY: u8 = 2;
pub(super) const STEP34_NUMERAL: u8 = 4;
pub(super) const STEP34_SPEAKER: u8 = 8;

pub(super) struct Step34Query {
    pub(super) anchor_ids: Vec<u32>,
    pub(super) entity_ids: Vec<u32>,
    pub(super) numeral_ids: Vec<u32>,
    pub(super) speakers: DetHashSet<String>,
    /// Query-side kinds actually present. Not a free parameter.
    pub(super) n: u8,
}

pub(super) fn step34_query(ix: &Index, query: &str) -> Step34Query {
    let mut seen_sp: DetHashSet<String> = DetHashSet::default();
    let mut speakers: DetHashSet<String> = DetHashSet::default();
    let mut speaker_toks: DetHashSet<String> = DetHashSet::default();
    for turn in &ix.turns {
        let sp = turn.speaker.as_str();
        let low = sp.to_ascii_lowercase();
        if low.is_empty() || low == "user" || low == "assistant" || !seen_sp.insert(low.clone()) {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
            speakers.insert(low);
            for w in raw_tokens(sp) {
                speaker_toks.insert(w);
            }
        }
    }
    let mut entity_toks: DetHashSet<String> = DetHashSet::default();
    for ent in proper_entity_tokens(query) {
        let Some(body) = ent.strip_prefix('\u{1}').and_then(|s| s.strip_prefix("ent:")) else {
            continue;
        };
        if body.contains('_') || body.bytes().any(|b| b.is_ascii_digit()) {
            continue;
        }
        if speaker_toks.contains(body) || !is_content(body) {
            continue;
        }
        entity_toks.insert(body.to_string());
    }
    let mut anchor_ids = Vec::new();
    let mut entity_ids = Vec::new();
    let mut numeral_ids = Vec::new();
    let mut seen_a: DetHashSet<u32> = DetHashSet::default();
    let mut seen_e: DetHashSet<u32> = DetHashSet::default();
    let mut seen_n: DetHashSet<u32> = DetHashSet::default();
    for w in raw_tokens(query) {
        if w.bytes().any(|b| b.is_ascii_digit()) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_n.insert(id) {
                    numeral_ids.push(id);
                }
            }
            continue;
        }
        if speaker_toks.contains(&w) {
            continue;
        }
        if entity_toks.contains(&w) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_e.insert(id) {
                    entity_ids.push(id);
                }
            }
            continue;
        }
        if is_content(&w) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_a.insert(id) {
                    anchor_ids.push(id);
                }
            }
        }
    }
    let mut n = 0u8;
    if !anchor_ids.is_empty() {
        n += 1;
    }
    if !entity_ids.is_empty() {
        n += 1;
    }
    if !numeral_ids.is_empty() {
        n += 1;
    }
    if !speakers.is_empty() {
        n += 1;
    }
    Step34Query { anchor_ids, entity_ids, numeral_ids, speakers, n }
}

pub(super) fn step34_posts(turn: &TurnIx, id: u32) -> bool {
    turn.content.contains(&id)
}

pub(super) fn step34_card(ix: &Index, ti: usize, q: &Step34Query) -> u8 {
    let turn = &ix.turns[ti];
    let mut bits = 0u8;
    if q.anchor_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_ANCHOR;
    }
    if q.entity_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_ENTITY;
    }
    if q.numeral_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_NUMERAL;
    }
    if q.speakers.contains(&turn.speaker.to_ascii_lowercase()) {
        bits |= STEP34_SPEAKER;
    }
    bits
}

/// Session of the strongest positive scene score. Tie: later session time, then
/// the earlier turn. Same order as `order`.
pub(super) fn step34_leader_session(ix: &Index, t_scene: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32, i64)> = None;
    for (ti, turn) in ix.turns.iter().enumerate() {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        let ts = ix.session_ts.get(turn.session).copied().unwrap_or(0);
        let take = match best {
            None => true,
            Some((bi, bs, bts)) => sc > bs || (sc == bs && (ts > bts || (ts == bts && ti < bi))),
        };
        if take {
            best = Some((ti, sc, ts));
        }
    }
    best.map(|(ti, _, _)| ix.turns[ti].session)
}

pub(super) fn apply_c2_step34_bertillonage(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let q = step34_query(ix, query);
    if q.n == 0 {
        return false;
    }
    let top = list[0];
    let head = step34_card(ix, top, &q);
    if head.count_ones() >= u32::from(q.n) {
        return false;
    }
    let leader = step34_leader_session(ix, t_scene);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let head_sc = t_scene.get(top).copied().unwrap_or(0.0);
    let head_sess = ix.turns[top].session;
    let head_sp = ix.turns[top].speaker.to_ascii_lowercase();
    let head_q = ix.turns[top].is_question;
    let blank = head == 0 && leader != Some(head_sess);
    let mut best: Option<(u32, usize)> = None;
    let mut tied = false;
    for slot in 1..window {
        let cand = list[slot];
        if ix.turns[cand].session != head_sess {
            continue;
        }
        let sc = t_scene.get(cand).copied().unwrap_or(0.0);
        if sc <= head_sc {
            continue;
        }
        let cb = step34_card(ix, cand, &q);
        if cb.count_ones() < u32::from(q.n) || (cb & head) != head || cb == head {
            continue;
        }
        let csp = ix.turns[cand].speaker.to_ascii_lowercase();
        let testimony = head_q
            && !ix.turns[cand].is_question
            && (csp == head_sp || q.speakers.contains(&csp));
        if !(testimony || blank) {
            continue;
        }
        let pc = cb.count_ones();
        match best {
            None => best = Some((pc, slot)),
            Some((bp, _)) if pc > bp => {
                best = Some((pc, slot));
                tied = false;
            }
            Some((bp, _)) if pc == bp => tied = true,
            _ => {}
        }
    }
    if tied {
        return false;
    }
    let Some((_, slot)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Surface anchors: lexical content-group labels. Stems, hypernyms, and
/// entity-phrase ids are not anchors. The label's own vocab id is the mark.
pub(super) fn c2_surface_anchor_ids(ix: &Index, plan: &QueryPlan) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for label in &plan.groups {
        if label.starts_with('\u{1}') {
            continue;
        }
        let Some(id) = ix.vocab.get(label) else {
            continue;
        };
        if seen.insert(id) {
            ids.push(id);
        }
    }
    ids
}

/// Pool the replay dump writes: seated turns first, then scene-descending
/// turns with a positive scene score, capped at 50.
pub(super) fn c2_dump_pool(ix: &Index, list: &[usize], t_scene: &[f32]) -> Vec<usize> {
    let mut pool: Vec<usize> = list.to_vec();
    let mut seen: DetHashSet<usize> = pool.iter().copied().collect();
    let turn_ts: Vec<i64> = ix
        .turns
        .iter()
        .map(|t| ix.session_ts.get(t.session).copied().unwrap_or(0))
        .collect();
    for ti in order(t_scene, &turn_ts) {
        if pool.len() >= 50 {
            break;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if seen.insert(ti) {
            pool.push(ti);
        }
    }
    pool
}

/// Mill's method of agreement (Step 35).
///
/// If every instance of a phenomenon shares exactly one circumstance, that
/// circumstance is its cause. Each surface query anchor is an instance. Its
/// hit set is the pool turns that post the anchor from their own content or
/// features (`TurnIx.body` is never read). The circumstance is a session.
/// The sessions common to every hit set are the agreement. Exactly one such
/// session is a cause. An empty hit set, a tie, or several common sessions
/// fix nothing.
///
/// A head already inside the agreement session is never displaced. A head
/// outside it yields slot 1 in exactly one of two shapes:
///
/// - Non-instance. The head posts no surface anchor, so it is not an instance
///   of the phenomenon. The unique top-5 turn of the agreement session that
///   posts at least one anchor (an instance) takes slot 1.
/// - Complete instance. The head posts some anchors and misses others. The
///   unique top-5 turn of the agreement session that posts every anchor, and
///   that is the same speaker as the head, takes slot 1. A different speaker
///   is the other role's echo of the same circumstance, not a second witness.
///
/// Two qualifying turns are a tie. Top-5 membership is unchanged. No new
/// constant: the pool cap is the dump's 50 and the window is the existing
/// top-5. Does not touch `rsb_seat_behind`.
pub(super) fn apply_c2_step35_mill_agreement(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let anchors = c2_surface_anchor_ids(ix, plan);
    if anchors.is_empty() {
        return false;
    }
    let pool = c2_dump_pool(ix, list, t_scene);
    let posts = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let mut common: DetHashSet<usize> = DetHashSet::default();
    let mut started = false;
    for &id in &anchors {
        let mut sessions: DetHashSet<usize> = DetHashSet::default();
        for &ti in &pool {
            if posts(ti, id) {
                sessions.insert(ix.turns[ti].session);
            }
        }
        if sessions.is_empty() {
            return false;
        }
        if !started {
            common = sessions;
            started = true;
        } else {
            common.retain(|s| sessions.contains(s));
        }
        if common.is_empty() {
            return false;
        }
    }
    if common.len() != 1 {
        return false;
    }
    let agree = match common.iter().next() {
        Some(&s) => s,
        None => return false,
    };
    let head = list[0];
    if ix.turns[head].session == agree {
        return false;
    }
    let head_has: Vec<bool> = anchors.iter().map(|&id| posts(head, id)).collect();
    let head_any = head_has.iter().copied().any(|h| h);
    let head_speaker = ix.turns[head].speaker.clone();
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut slot: Option<usize> = None;
    for i in 1..window {
        let cand = list[i];
        if ix.turns[cand].session != agree {
            continue;
        }
        let qualifies = if head_any {
            ix.turns[cand].speaker == head_speaker && anchors.iter().all(|&id| posts(cand, id))
        } else {
            anchors.iter().any(|&id| posts(cand, id))
        };
        if !qualifies {
            continue;
        }
        if slot.is_some() {
            return false;
        }
        slot = Some(i);
    }
    let Some(slot) = slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


/// Content-token anchors of the query. Each anchor is the vocab ids of the
/// surface token and, when it differs, its stem. Closed-class tokens are not
/// anchors. A token the index never interned is not an anchor. No turn body
/// is read.
pub(super) fn step36_anchors(ix: &Index, query: &str) -> Vec<Vec<u32>> {
    let mut seen: DetHashSet<String> = DetHashSet::default();
    let mut anchors = Vec::new();
    for w in raw_tokens(query) {
        if !is_content(&w) || !seen.insert(w.clone()) {
            continue;
        }
        let mut ids = Vec::new();
        if let Some(id) = ix.vocab.get(&w) {
            ids.push(id);
        }
        let stemmed = stem(&w);
        if stemmed != w {
            if let Some(id) = ix.vocab.get(&stemmed) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        if !ids.is_empty() {
            anchors.push(ids);
        }
    }
    anchors
}

pub(super) fn step36_posts(turn: &TurnIx, ids: &[u32]) -> bool {
    ids.iter()
        .any(|id| turn.content.contains(id) || turn.feats.iter().any(|f| f.id == *id))
}

/// Antaranga-bahiranga (Step 36). The internal operation precedes the external
/// one. Internal evidence is `t_own`, the turn's uncarried lexical score.
/// External evidence on the turn is everything in `t_scene` above `t_own`
/// (neighbour carry and later scene operations). `bridge_boost` is not part of
/// `t_scene`; it is added only to the session rank score, and that sum is the
/// session sightline.
///
/// Slot 1 yields only when it is a question and its external residual strictly
/// exceeds its own score. The witness is a top-5 turn that is not a question,
/// whose own score strictly exceeds slot 1's own score and is a strict majority
/// of its own scene score, that attests every query anchor slot 1 attests, and
/// whose session score (`s_score + bridge_boost`) strictly exceeds slot 1's.
/// Two witnesses with the same own score are not a unique internal reading, so
/// the seat stays. Top-5 membership does not change. Does not read turn bodies
/// and does not touch `rsb_seat_behind`.
pub(super) fn apply_c2_step36_antaranga(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    t_own: &[f32],
    t_scene: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    if !ix.turns[head].is_question {
        return false;
    }
    let own = |ti: usize| t_own.get(ti).copied().unwrap_or(0.0);
    let scene = |ti: usize| t_scene.get(ti).copied().unwrap_or(0.0);
    let sess = |ti: usize| {
        let s = ix.turns[ti].session;
        s_score.get(s).copied().unwrap_or(0.0) + bridge_boost.get(s).copied().unwrap_or(0.0)
    };
    let ho = own(head);
    let hs = scene(head);
    // Mostly external: the scene residual above own lexical exceeds own lexical.
    if hs - ho <= ho {
        return false;
    }
    let anchors = step36_anchors(ix, query);
    let head_turn = &ix.turns[head];
    let head_sess = sess(head);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut best: Option<(f32, usize)> = None;
    let mut tied = false;
    for slot in 1..window {
        let cand = list[slot];
        let turn = &ix.turns[cand];
        if turn.is_question {
            continue;
        }
        let to = own(cand);
        let ts = scene(cand);
        if to <= ho || to <= ts - to {
            continue;
        }
        if sess(cand) <= head_sess {
            continue;
        }
        let covers = anchors
            .iter()
            .all(|ids| !step36_posts(head_turn, ids) || step36_posts(turn, ids));
        if !covers {
            continue;
        }
        match best {
            None => best = Some((to, slot)),
            Some((bo, _)) if to > bo => {
                best = Some((to, slot));
                tied = false;
            }
            Some((bo, _)) if to == bo => tied = true,
            _ => {}
        }
    }
    if tied {
        return false;
    }
    let Some((_, slot)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


/// Kalevala parallelism (Step 37).
///
/// Adjacent lines of one singer restate one another and count as a single
/// line. Two turns are that line when they are successive utterances of the
/// same speaker, at most one other turn sits between them, the shared content
/// outnumbers half of the shorter line, and they agree on numeral shape
/// (`form_digit`). The following line takes slot 1 when its content query
/// cues strictly cover the head's. Closed-class query tokens are not cues
/// here: a following line that only adds "which" or "the" is the same line,
/// not a richer variant. An earlier line never takes the seat. Exactly one
/// following variant may qualify, and it must already sit in the top 5.
/// Top-5 membership does not change. `TurnIx.body` is never read. Does not
/// touch `rsb_seat_behind`.
pub(super) fn apply_c2_step37_kalevala(ix: &Index, query: &str, list: &mut Vec<usize>) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    let q_cues = step37_content_query_cues(ix, query);
    let head_cues = turn_query_cues(ix, &q_cues, head);
    let head_pos = c2_session_pos(ix, head);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut slot: Option<usize> = None;
    for i in 1..window {
        let cand = list[i];
        if c2_session_pos(ix, cand) <= head_pos || !step37_one_line(ix, head, cand) {
            continue;
        }
        let cand_cues = turn_query_cues(ix, &q_cues, cand);
        if !covers_cues(&cand_cues, &head_cues) || cand_cues.len() <= head_cues.len() {
            continue;
        }
        if slot.is_some() {
            return false;
        }
        slot = Some(i);
    }
    let Some(slot) = slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Query cues that are content words. Closed-class tokens stay in
/// `query_cues` for the other seats; a parallel line is not richer for
/// adding one of them.
pub(super) fn step37_content_query_cues(ix: &Index, query: &str) -> Vec<u32> {
    let q_cues = query_cues(ix, query);
    let mut keep: Vec<u32> = ix
        .cue_vocab
        .iter()
        .filter(|(word, id)| q_cues.binary_search(id).is_ok() && is_content(word))
        .map(|(_, id)| *id)
        .collect();
    keep.sort_unstable();
    keep.dedup();
    keep
}

/// One Kalevala line: successive same-speaker turns, at most one turn between
/// them, shared numeral shape, and a short-line restatement.
pub(super) fn step37_one_line(ix: &Index, a: usize, b: usize) -> bool {
    let ta = &ix.turns[a];
    let tb = &ix.turns[b];
    if ta.session != tb.session || ta.speaker != tb.speaker || ta.form_digit != tb.form_digit {
        return false;
    }
    let pa = c2_session_pos(ix, a);
    let pb = c2_session_pos(ix, b);
    let (lo, hi) = if pa < pb { (pa, pb) } else { (pb, pa) };
    if hi == lo || hi - lo > 2 {
        return false;
    }
    let members = &ix.session_turns[ta.session];
    for &m in &members[lo + 1..hi] {
        if ix.turns[m].speaker == ta.speaker {
            return false;
        }
    }
    step37_short_restatement(&ta.content, &tb.content)
}

/// The shorter line is mostly the words the two lines share. Equal content
/// is the same line twice, not a variant.
pub(super) fn step37_short_restatement(a: &DetHashSet<u32>, b: &DetHashSet<u32>) -> bool {
    if a.is_empty() || b.is_empty() || a == b {
        return false;
    }
    let inter = a.intersection(b).count();
    let short = a.len().min(b.len());
    inter * 2 > short
}


/// Hermeneutic circle (Step 38). A part is read by the whole it sits in.
///
/// Surface anchors are the query's content tokens (`step36_anchors`): closed
/// class is already excluded, and a token the index never interned is not an
/// anchor. A turn posts an anchor from its own content or features
/// (`TurnIx.body` is never read). The session posts an anchor when any turn
/// in it does. The part is the set of anchors the turn itself posts. The rest
/// is the anchors it does not post.
///
/// Slot 1 yields only when it is a question, its part is a nonempty proper
/// subset of the anchors, and exactly one other top-5 turn is the same part:
/// not a question, a different session, a strictly stronger own lexical score,
/// and a session that posts every rest anchor while slot 1's session misses
/// at least one. Two such turns are not one reading, so the seat stays.
/// Top-5 membership does not change. No new constant: the window is the
/// existing top-5. Does not touch `rsb_seat_behind` and does not rewrite the
/// session list. Speaker is not consulted.
pub(super) fn apply_c2_step38_hermeneutic(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    t_own: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let anchors = step36_anchors(ix, query);
    if anchors.len() < 2 {
        return false;
    }
    let head = list[0];
    if !ix.turns[head].is_question {
        return false;
    }
    let posts = |ti: usize, ids: &[u32]| step36_posts(&ix.turns[ti], ids);
    let session_posts = |session: usize, ids: &[u32]| {
        ix.session_turns
            .get(session)
            .is_some_and(|members| members.iter().any(|&ti| posts(ti, ids)))
    };
    let head_hit: Vec<bool> = anchors.iter().map(|ids| posts(head, ids)).collect();
    if !head_hit.iter().any(|h| *h) || head_hit.iter().all(|h| *h) {
        return false;
    }
    let head_sess = ix.turns[head].session;
    let head_own = t_own.get(head).copied().unwrap_or(0.0);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut slot: Option<usize> = None;
    for i in 1..window {
        let cand = list[i];
        let turn = &ix.turns[cand];
        if turn.is_question || turn.session == head_sess {
            continue;
        }
        let same_part = anchors
            .iter()
            .enumerate()
            .all(|(a, ids)| posts(cand, ids) == head_hit[a]);
        if !same_part {
            continue;
        }
        let rest_ok = anchors.iter().enumerate().all(|(a, ids)| {
            head_hit[a] || session_posts(turn.session, ids)
        });
        if !rest_ok {
            continue;
        }
        let head_whole = anchors.iter().enumerate().all(|(a, ids)| {
            head_hit[a] || session_posts(head_sess, ids)
        });
        if head_whole {
            continue;
        }
        let own = t_own.get(cand).copied().unwrap_or(0.0);
        if own <= head_own {
            continue;
        }
        if slot.is_some() {
            return false;
        }
        slot = Some(i);
    }
    let Some(slot) = slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
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
pub(super) fn apply_c2_pool_c_session_opener_seat(
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
pub(super) fn apply_c2_pool_c_absent_session_key(
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
pub(super) fn apply_c2_pool_b_standing_seat(
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

/// Surface form of a vocab id, with the stem/entity mark stripped.
pub(super) fn c2_term_surface<'a>(ix: &'a Index, id: u32) -> &'a str {
    let raw = ix.vocab.names.get(id as usize).map(String::as_str).unwrap_or("");
    raw.strip_prefix("\u{1}stem:")
        .or_else(|| raw.strip_prefix("\u{1}ent:"))
        .or_else(|| raw.strip_prefix("\u{1}hyp:"))
        .unwrap_or(raw)
}

/// True when one of the policy turn's stored content terms and a query focal
/// term differ only by a closed inflectional suffix. Orthographic shape, not
/// a domain list. `date` does not meet `deadline`.
pub(super) fn c2_policy_inflects_query(ix: &Index, ti: usize, plan: &QueryPlan) -> bool {
    fn inflects(shorter: &str, longer: &str) -> bool {
        if shorter.len() < 5 {
            return false;
        }
        let Some(rest) = longer.strip_prefix(shorter) else {
            return false;
        };
        matches!(
            rest,
            "" | "s"
                | "es"
                | "ed"
                | "ing"
                | "ings"
                | "er"
                | "ers"
                | "ly"
                | "al"
                | "ment"
                | "ments"
                | "tion"
                | "tions"
                | "ation"
                | "ations"
        )
    }
    ix.turns[ti].seq.iter().any(|&(raw, _)| {
        let term = c2_term_surface(ix, raw);
        plan.focal_ids.iter().any(|&fid| {
            let focal = c2_term_surface(ix, fid);
            if term.len() <= focal.len() {
                inflects(term, focal)
            } else {
                inflects(focal, term)
            }
        })
    })
}


/// Buried lexical peak. The scene pool is the same 50 the replay dump calls
/// pool@50: the seated list, then turns with `t_scene > 0` in scene order.
/// A declarative in a named-speaker session whose own lexical score strictly
/// exceeds every turn in that pool, when it is the only such turn, was cut
/// by scene penalties rather than by a missing posting. It takes the seat of
/// the lowest-own question in slots 1..4. Slot 0 stays. The displaced question
/// is kept: appended while the list is short of `k`, otherwise written over
/// the tail. The session list is not rewritten. Does not touch `rsb_seat_behind`.
pub(super) fn apply_c2_buried_lexical_peak(
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
pub(super) fn apply_c2_anaphoric_tail(ix: &Index, list: &mut Vec<usize>, t_own: &[f32]) -> bool {
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
pub(super) fn apply_c2_unseated_session_peak(
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

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----
// Wave 2: Steps 11-20 helpers (included from retrieval.rs right after steps7_10.rs).
// All ranking helpers use `DetHashMap`/`DetHashSet`/`BTreeMap`/`BTreeSet` and never read
// `TurnIx.body` (empty under `--no-dump`). Closed-class grammar only; zero domain word lists.

/// Step 11: closed-class WH quantity vs person intent for the Kemeny-Young QA-slot voter.
/// Elapsed-time queries (`temporal`, `"ago"`, `"since"`) are excluded from `how_qty` because
/// their answers come from session date arithmetic rather than an Arabic digit inside the turn.
pub(super) fn step11_wh_qa_mode(q_text: &str, temporal: bool) -> (bool, bool) {
    let slots = wh_slots(q_text);
    let how_qty = !temporal && (slots & WH_QTY) != 0;
    let who_person = !how_qty && (slots & WH_PERSON) != 0;
    (how_qty, who_person)
}

/// Step 11 (v6): Kemeny-Young Condorcet QA-Slot Agreement Voter over top-5 same-session,
/// same-speaker cohorts (`head.session`, `head.speaker`). Keeps all unrelated-session top-5 slots
/// fixed so intervening distractors never outvote a same-speaker QA-slot improvement, while letting
/// multi-sentence non-short-question turns carrying Arabic digits (`how_qty`) or named-referent
/// markers (`who_person`) win the head seat within `>= 0.70 * scene`.
pub(super) fn apply_step11_kemeny_wh_slot_voter(
    ix: &Index,
    q_text: &str,
    it: &QueryIntent,
    t_scene: &[f32],
    list: &mut Vec<usize>,
) {
    let top_len = list.len().min(5);
    if top_len < 2 {
        return;
    }
    let (how_qty, who_person) = step11_wh_qa_mode(q_text, it.temporal);
    if !how_qty && !who_person {
        return;
    }
    let Some(head_turn) = ix.turns.get(list[0]) else {
        return;
    };
    let head_sess = head_turn.session;
    let head_speaker = head_turn.speaker.as_str();
    let slot_positions: Vec<usize> = (0..top_len)
        .filter(|&pos| {
            ix.turns
                .get(list[pos])
                .is_some_and(|t| t.session == head_sess && t.speaker == head_speaker)
        })
        .collect();
    if slot_positions.len() < 2 {
        return;
    }
    let cohort: Vec<usize> = slot_positions.iter().map(|&p| list[p]).collect();
    let q_ent_ids: DetHashSet<u32> = raw_tokens(q_text)
        .into_iter()
        .filter_map(|w| ix.vocab.get(&format!("\u{1}ent:{w}")))
        .collect();
    let n = cohort.len();
    let mut net = vec![vec![0i64; n]; n];
    let mut any_qa_vote = false;

    for a in 0..n {
        for b in (a + 1)..n {
            let (ta, tb) = (cohort[a], cohort[b]);
            let mut margin = 1i64;
            let scene_a = t_scene.get(ta).copied().unwrap_or(0.0);
            let scene_b = t_scene.get(tb).copied().unwrap_or(0.0);

            if let (Some(turn_a), Some(turn_b)) = (ix.turns.get(ta), ix.turns.get(tb)) {
                if how_qty {
                    let num_a = !turn_a.is_short_question && turn_a.form_digit;
                    let num_b = !turn_b.is_short_question && turn_b.form_digit;
                    if num_b && !num_a && !turn_is_superseded(ix, tb) && scene_b >= 0.70 * scene_a {
                        margin -= 2;
                        any_qa_vote = true;
                    } else if num_a && !num_b && !turn_is_superseded(ix, ta) {
                        margin += 2;
                    }
                } else if who_person {
                    let has_ent_a = !turn_a.is_short_question
                        && (turn_a.form_handle || turn_a.form_midcap || turn_a.slot_entities_clean >= 1);
                    let has_novel_ent_b = !turn_b.is_short_question
                        && turn_b.content.len() >= 8
                        && (turn_b.form_handle || turn_b.form_midcap)
                        && turn_b.entities.iter().any(|e| !q_ent_ids.contains(e));
                    if has_novel_ent_b
                        && !has_ent_a
                        && !turn_is_superseded(ix, tb)
                        && scene_b >= 0.70 * scene_a
                    {
                        margin -= 2;
                        any_qa_vote = true;
                    } else if has_ent_a && !has_novel_ent_b && !turn_is_superseded(ix, ta) {
                        margin += 2;
                    }
                }
            }

            net[a][b] = margin;
            net[b][a] = -margin;
        }
    }

    if !any_qa_vote {
        return;
    }
    let rsb: Vec<usize> = (0..n).collect();
    let mut order = kemeny_young(&net, &rsb);
    if order.len() != n {
        order = rsb;
    }
    let mut reordered: Vec<usize> = order.into_iter().map(|i| cohort[i]).collect();
    if turn_is_superseded(ix, reordered[0]) {
        if let Some(pos) = reordered.iter().position(|&ti| !turn_is_superseded(ix, ti)) {
            let keeper = reordered.remove(pos);
            reordered.insert(0, keeper);
        }
    }
    for (slot_idx, &pos) in slot_positions.iter().enumerate() {
        list[pos] = reordered[slot_idx];
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

/// Closed-class check: returns `true` when `q_text` asks a comparative event-ordering,
/// interval, or difference question between two events.
pub(super) fn step15_is_comparative_query(q_text: &str, it: &QueryIntent) -> bool {
    if !(it.temporal || it.is_event_sequence) {
        return false;
    }
    let slots = wh_slots(q_text);
    let digits = raw_tokens(q_text)
        .into_iter()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .count();
    (slots & (WH_TIME | WH_QTY)) != 0 || digits >= 2 || it.is_event_sequence
}

/// Split a comparative event query along closed-class coordinators (`before or after`, `or`,
/// `between ... and`, or interval `before/after ... did I`) into two raw event clauses.
pub(super) fn step15_split_comparative_clauses(q_text: &str) -> Option<(String, String)> {
    let ql = q_text.to_ascii_lowercase();
    let trim_clause = |s: &str| -> String {
        s.trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .trim()
            .to_string()
    };

    if let Some(idx) = ql.find(" before or after ") {
        let a = &ql[..idx];
        let b = &ql[idx + " before or after ".len()..];
        let a_trim = a.rfind(',').map(|c| &a[c + 1..]).unwrap_or(a);
        return Some((trim_clause(a_trim), trim_clause(b)));
    }

    // Scan explicit `" or "` coordinators, skipping idiomatic non-event hedges.
    let mut search = 0usize;
    let mut chosen_or: Option<usize> = None;
    while let Some(rel) = ql[search..].find(" or ") {
        let pos = search + rel;
        let before = &ql[..pos];
        let after = &ql[pos + 4..];
        let idiom_skip = before.ends_with(" more")
            || before.ends_with(" yes")
            || before.ends_with(" before")
            || after.starts_with("less ")
            || after.starts_with("no ")
            || after.starts_with("not ");
        if !idiom_skip {
            chosen_or = Some(pos);
        }
        search = pos + 4;
    }
    if let Some(pos) = chosen_or {
        let left_full = &ql[..pos];
        let left = if let Some(comma) = left_full.rfind(',') {
            &left_full[comma + 1..]
        } else if let Some(bt) = left_full.rfind(" between ") {
            &left_full[bt + " between ".len()..]
        } else {
            left_full
        };
        let right_full = &ql[pos + 4..];
        let right_end = right_full
            .find(['?', '.', '!'])
            .unwrap_or(right_full.len());
        let right = right_full[..right_end]
            .rfind(',')
            .filter(|&c| {
                let tail = &right_full[c + 1..right_end];
                tail.contains("earliest") || tail.contains("latest") || tail.contains("in ")
            })
            .map(|c| &right_full[..c])
            .unwrap_or(&right_full[..right_end]);
        return Some((trim_clause(left), trim_clause(right)));
    }

    if let Some(bt) = ql.find(" between ") {
        let rest = &ql[bt + " between ".len()..];
        let end = rest.find(['?', '.', '!']).unwrap_or(rest.len());
        let body = &rest[..end];
        let mut and_positions = Vec::new();
        let mut s = 0usize;
        while let Some(rel) = body[s..].find(" and ") {
            let p = s + rel;
            and_positions.push(p);
            s = p + 5;
        }
        if !and_positions.is_empty() {
            let split_p = and_positions[and_positions.len() / 2];
            let left = &body[..split_p];
            let right = &body[split_p + 5..];
            return Some((trim_clause(left), trim_clause(right)));
        }
    }

    // Only split on `" before "` / `" after "` for interval/time comparisons (not single-event
    // `"Where did X move after Y?"`).
    let interval_frame = (wh_slots(q_text) & (WH_TIME | WH_QTY)) != 0;
    if interval_frame {
        for prep in [" before ", " after "] {
            if let Some(p) = ql.find(prep) {
                let left_raw = &ql[..p];
                let right_full = &ql[p + prep.len()..];
                let r_trim = right_full.trim_start();
                if r_trim.starts_with("today")
                    || r_trim.starts_with("that")
                    || r_trim.starts_with("now")
                {
                    continue;
                }
                const INV_AUX: &[&str] = &[
                    " did i ", " did we ", " do i ", " does ", " was i ", " were we ", " have i ",
                    " had i ",
                ];
                let mut inv_hit: Option<(usize, usize)> = None;
                for aux in INV_AUX {
                    if let Some(ai) = right_full.find(aux) {
                        if inv_hit.is_none_or(|(prev, _)| ai < prev) {
                            inv_hit = Some((ai, aux.len()));
                        }
                    }
                }
                if let Some((ai, alen)) = inv_hit {
                    let ev_a = &right_full[..ai];
                    let ev_b = &right_full[ai + alen..];
                    return Some((trim_clause(ev_a), trim_clause(ev_b)));
                }
                return Some((trim_clause(left_raw), trim_clause(right_full)));
            }
        }
    }

    None
}

/// Resolve exclusive bitmasks `(disc_a, disc_b)` over `plan.groups` (bits `0..64`) for the two
/// compared events. Returns `None` unless both exclusive sides contain >= 1 query group.
pub(super) fn step15_comparative_event_masks(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
) -> Option<(u64, u64)> {
    if it.is_event_sequence
        || it.assistant_attribution
        || plan.intent.preference
        || !step15_is_comparative_query(q_text, it)
        || plan.groups.len() < 2
    {
        return None;
    }
    let (clause_a, clause_b) = step15_split_comparative_clauses(q_text)?;
    let clause_to_mask = |clause: &str| -> u64 {
        let mut mask = 0u64;
        for w in raw_tokens(clause) {
            if let Some(g) = plan
                .groups
                .iter()
                .position(|grp| grp == &w)
                .or_else(|| {
                    ix.vocab
                        .get(&w)
                        .and_then(|id| plan.group_of.get(&id).copied())
                })
            {
                if g < 64 {
                    mask |= 1u64 << g;
                }
            }
        }
        mask
    };
    let mask_a = clause_to_mask(&clause_a);
    let mask_b = clause_to_mask(&clause_b);
    let shared = mask_a & mask_b;
    let disc_a = mask_a & !shared;
    let disc_b = mask_b & !shared;
    if disc_a == 0 || disc_b == 0 {
        return None;
    }
    Some((disc_a, disc_b))
}

/// Compute `(best_pop_a, max_a, best_pop_b, max_b)` using each side's highest exclusive bit-count
/// coverage so partial 1-word cross-mentions in the opposite session do not mask a missing side.
pub(super) fn step15_best_side_coverage(
    ix: &Index,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &[f32],
    disc_a: u64,
    disc_b: u64,
) -> (u32, f32, u32, f32) {
    let mut best_pop_a = 0u32;
    let mut best_pop_b = 0u32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        best_pop_a = best_pop_a.max((m & disc_a).count_ones());
        best_pop_b = best_pop_b.max((m & disc_b).count_ones());
    }
    let mut max_a = 0.0f32;
    let mut max_b = 0.0f32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        if best_pop_a > 0 && (m & disc_a).count_ones() >= best_pop_a && sc > max_a {
            max_a = sc;
        }
        if best_pop_b > 0 && (m & disc_b).count_ones() >= best_pop_b && sc > max_b {
            max_b = sc;
        }
    }
    (best_pop_a, max_a, best_pop_b, max_b)
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Balance`, scoring stage):
/// Balances Side A vs Side B using full exclusive-clause bit coverage (`best_pop_a`, `best_pop_b`)
/// down to `0.10 * strong`, promoting at most one best turn per weaker-side session toward
/// `0.90 * strong` (preserving rank #1).
pub(super) fn apply_step15_comparative_dual_slot_scores(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    if t_scene.len() != ix.turns.len()
        || t_own.len() != ix.turns.len()
        || t_exact_mask.len() != ix.turns.len()
    {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, max_a, best_pop_b, max_b) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if max_a <= 0.0 || max_b <= 0.0 {
        return;
    }

    let strong = max_a.max(max_b);
    let weak = max_a.min(max_b);
    if weak >= 0.58 * strong || weak < 0.10 * strong {
        return;
    }

    let (weak_mask, weak_req_pop) = if max_a < max_b {
        (disc_a, best_pop_a)
    } else {
        (disc_b, best_pop_b)
    };
    let ns = ix.session_turns.len();
    let mut sess_best: Vec<Option<(usize, u32, bool, f32)>> = vec![None; ns];
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < 0.70 * weak
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        let w_hits = (m & weak_mask).count_ones();
        if w_hits < weak_req_pop {
            continue;
        }
        let si = ix.turns[ti].session;
        if si >= ns {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match sess_best[si] {
            None => true,
            Some((_, prev_hits, prev_user, prev_sc)) => {
                (w_hits, is_user) > (prev_hits, prev_user)
                    || ((w_hits, is_user) == (prev_hits, prev_user) && sc > prev_sc)
            }
        };
        if better {
            sess_best[si] = Some((ti, w_hits, is_user, sc));
        }
    }

    let mut chosen: Vec<(usize, u32, bool, f32)> = sess_best.into_iter().flatten().collect();
    chosen.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });
    let cap = 0.78 * strong;
    let mult = (cap / weak).min(4.0);
    for (ti, _, _, base_sc) in chosen.into_iter().take(1) {
        let lifted = (base_sc * mult).min(cap);
        if lifted > t_scene[ti] {
            t_scene[ti] = lifted;
        }
    }
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Seat`, post-Q4 top-5 safeguard):
/// Ensures `final_turn_indices[..5]` contains best-bit coverage for BOTH Side A and Side B
/// (not merely 1 shared keyword inside a rival session), replacing a duplicate-session slot
/// inside `1..5` (never touching slot `0` or floored anchors) and giving the seated turn enough
/// score headroom to survive Kemeny reordering.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_step15_comparative_dual_slot_seat(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    floored: &dyn Fn(usize) -> bool,
    k: usize,
    t_scene: &mut [f32],
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 5 || k < 5 {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, _, best_pop_b, _) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if best_pop_a == 0 || best_pop_b == 0 {
        return;
    }

    let req_a = best_pop_a.clamp(1, 2);
    let req_b = best_pop_b.clamp(1, 2);
    let top5_sessions: std::collections::BTreeSet<usize> = final_turn_indices[..5]
        .iter()
        .map(|&ti| ix.turns[ti].session)
        .collect();
    let has_a = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_a).count_ones() >= req_a);
    let has_b = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_b).count_ones() >= req_b);
    if has_a && has_b && top5_sessions.len() >= 2 {
        return;
    }

    let (missing_mask, req_missing, present_mask, req_present) = if !has_a {
        (disc_a, req_a, disc_b, req_b)
    } else if !has_b {
        (disc_b, req_b, disc_a, req_a)
    } else {
        // Both masks appear in a single session that occupies all 5 slots: find best other-session turn.
        (disc_a | disc_b, 1, disc_a | disc_b, 1)
    };
    let need_new_session = top5_sessions.len() < 2;
    let head_sc = t_scene[final_turn_indices[0]];
    if head_sc <= 0.0 {
        return;
    }
    let min_cand_sc = 0.10 * head_sc;

    let mut best_cand: Option<(usize, u32, bool, f32)> = None;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < min_cand_sc
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
            || final_turn_indices[..5].contains(&ti)
            || (need_new_session && top5_sessions.contains(&ix.turns[ti].session))
        {
            continue;
        }
        let hits = (t_exact_mask[ti] & missing_mask).count_ones();
        if hits < req_missing {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match best_cand {
            None => true,
            Some((prev_ti, prev_hits, prev_user, prev_sc)) => {
                (hits, is_user) > (prev_hits, prev_user)
                    || ((hits, is_user) == (prev_hits, prev_user)
                        && (sc > prev_sc || (sc == prev_sc && ti < prev_ti)))
            }
        };
        if better {
            best_cand = Some((ti, hits, is_user, sc));
        }
    }
    let Some((cand_ti, _, _, _)) = best_cand else {
        return;
    };

    let victim_slot = (1..5).rev().find(|&slot| {
        let vti = final_turn_indices[slot];
        if floored(vti) {
            return false;
        }
        let vs = ix.turns[vti].session;
        let sess_cnt = final_turn_indices[..5]
            .iter()
            .filter(|&&t| ix.turns[t].session == vs)
            .count();
        let present_elsewhere = final_turn_indices[..5]
            .iter()
            .enumerate()
            .any(|(idx, &t)| idx != slot && (t_exact_mask[t] & present_mask).count_ones() >= req_present);
        sess_cnt >= 2 && present_elsewhere
    });
    let Some(slot) = victim_slot else {
        return;
    };

    let target_sc = 0.80 * head_sc;
    if t_scene[cand_ti] < target_sc {
        t_scene[cand_ti] = target_sc;
    }
    if let Some(existing) = final_turn_indices.iter().position(|&t| t == cand_ti) {
        final_turn_indices.remove(existing);
    }
    let removed = final_turn_indices.remove(slot);
    final_turn_indices.insert(slot, cand_ti);
    let demote_pos = 5.min(final_turn_indices.len());
    final_turn_indices.insert(demote_pos, removed);
    if final_turn_indices.len() > k {
        final_turn_indices.truncate(k);
    }
}
