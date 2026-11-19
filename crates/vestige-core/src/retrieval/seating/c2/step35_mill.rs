//! C2 seating lane: `step35_mill`.

use crate::retrieval::*;


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
pub(in crate::retrieval) fn apply_c2_step35_mill_agreement(
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
