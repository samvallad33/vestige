//! C2 seating lane: `step36_38_classical`.

use crate::retrieval::*;



/// Content-token anchors of the query. Each anchor is the vocab ids of the
/// surface token and, when it differs, its stem. Closed-class tokens are not
/// anchors. A token the index never interned is not an anchor. No turn body
/// is read.
pub(in crate::retrieval) fn step36_anchors(ix: &Index, query: &str) -> Vec<Vec<u32>> {
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

pub(in crate::retrieval) fn step36_posts(turn: &TurnIx, ids: &[u32]) -> bool {
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
pub(in crate::retrieval) fn apply_c2_step36_antaranga(
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
pub(in crate::retrieval) fn apply_c2_step37_kalevala(ix: &Index, query: &str, list: &mut Vec<usize>) -> bool {
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
pub(in crate::retrieval) fn step37_content_query_cues(ix: &Index, query: &str) -> Vec<u32> {
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
pub(in crate::retrieval) fn step37_one_line(ix: &Index, a: usize, b: usize) -> bool {
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
pub(in crate::retrieval) fn step37_short_restatement(a: &DetHashSet<u32>, b: &DetHashSet<u32>) -> bool {
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
pub(in crate::retrieval) fn apply_c2_step38_hermeneutic(
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
