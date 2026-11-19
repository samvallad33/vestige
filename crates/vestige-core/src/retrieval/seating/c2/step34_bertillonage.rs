//! C2 seating lane: `step34_bertillonage`.

use crate::retrieval::*;


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
pub(in crate::retrieval) const STEP34_ANCHOR: u8 = 1;
pub(in crate::retrieval) const STEP34_ENTITY: u8 = 2;
pub(in crate::retrieval) const STEP34_NUMERAL: u8 = 4;
pub(in crate::retrieval) const STEP34_SPEAKER: u8 = 8;

pub(in crate::retrieval) struct Step34Query {
    pub(in crate::retrieval) anchor_ids: Vec<u32>,
    pub(in crate::retrieval) entity_ids: Vec<u32>,
    pub(in crate::retrieval) numeral_ids: Vec<u32>,
    pub(in crate::retrieval) speakers: DetHashSet<String>,
    /// Query-side kinds actually present. Not a free parameter.
    pub(in crate::retrieval) n: u8,
}

pub(in crate::retrieval) fn step34_query(ix: &Index, query: &str) -> Step34Query {
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

pub(in crate::retrieval) fn step34_posts(turn: &TurnIx, id: u32) -> bool {
    turn.content.contains(&id)
}

pub(in crate::retrieval) fn step34_card(ix: &Index, ti: usize, q: &Step34Query) -> u8 {
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
pub(in crate::retrieval) fn step34_leader_session(ix: &Index, t_scene: &[f32]) -> Option<usize> {
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

pub(in crate::retrieval) fn apply_c2_step34_bertillonage(
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
