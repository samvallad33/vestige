//! Seating pass: `florilegium`.

use crate::retrieval::*;


// ---- moved verbatim from former `src/steps7_10.rs` (was `include!()`d into retrieval.rs) ----

/// Step 10 (#26 Florilegium multi-session top-5 seat guarantee): when multiple strong top-5
/// sessions compete (`strong_top_sessions >= 2`), relax Q4 thresholds to `(0.38, 0.20)` and block
/// single-session `bypass_cap` monopolies in top-5.
pub(in crate::retrieval) fn step10_florilegium_floors(strong_top_sessions: usize) -> (f32, f32, bool) {
    if strong_top_sessions >= 2 {
        (0.38, 0.20, false)
    } else {
        (0.45, 0.25, true)
    }
}

/// Step 10A Florilegium evidence seat: an unseated role-structured top-5 session earns seat 5
/// (index 4) if its best turn covers an uncovered query group and seat 5 is redundant (>= 2 seats
/// from same session and adds no unique group). Seats 1-4 never move.
pub(in crate::retrieval) fn apply_step10a_evidence_seat(
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

pub(in crate::retrieval) fn step10a_role_session(ix: &Index, si: usize) -> bool {
    ix.session_turns[si].iter().any(|&t| ix.turns[t].field == 1)
}
