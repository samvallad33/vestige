//! Seating pass: `rst_and_milestones`.

use crate::retrieval::*;


/// Issue #810: Stede RST Nucleus/Satellite preference at Rank 1 (`0..5` membership unchanged).
/// If slot 0 is a structural Satellite (backchannel, short reply, or zero-rare-payload turn)
/// and a Nucleus within slots `1..5` covers at least the same exact query groups with
/// `t_scene >= 0.72 * t_scene[0]`, promote that Nucleus to slot 0.
pub(in crate::retrieval) fn apply_rst_nucleus_preference(
    ix: &Index,
    list: &mut [usize],
    t_scene: &[f32],
    t_exact_mask: &[u64],
) {
    if list.len() < 2 {
        return;
    }
    let head = list[0];
    if crate::retrieval::query::rst::classify_rst_role(ix, head) != crate::retrieval::query::rst::RstRole::Satellite {
        return;
    }
    let head_sc = t_scene.get(head).copied().unwrap_or(0.0);
    let head_exact = t_exact_mask.get(head).copied().unwrap_or(0);
    if head_sc <= 0.0 {
        return;
    }
    let search_end = list.len().min(5);
    if let Some(pos) = (1..search_end).find(|&idx| {
        let cand = list[idx];
        crate::retrieval::query::rst::classify_rst_role(ix, cand) == crate::retrieval::query::rst::RstRole::Nucleus
            && (t_exact_mask.get(cand).copied().unwrap_or(0) & head_exact) == head_exact
            && t_scene.get(cand).copied().unwrap_or(0.0) >= 0.90 * head_sc
    }) {
        list.swap(0, pos);
    }
}

/// [W2-033] Milestone Admission: for chronological / multi-date event-sequence queries
/// (`it.is_event_sequence`), deduplicate repeated timestamps in Slots `1..5` using
/// unseen milestone timestamps from `5..` and enforce monotonic ascending timestamp order
/// across distinct non-zero timestamps in `1..5` (preserving slot 0 when dominant).
pub(in crate::retrieval) fn apply_w2_milestone_admission(
    ix: &Index,
    it: &QueryIntent,
    final_turn_indices: &mut Vec<usize>,
    t_scene: &[f32],
) {
    if !it.is_event_sequence || final_turn_indices.len() < 4 {
        return;
    }
    let top_n = final_turn_indices.len().min(5);
    let mut seen_ts: Vec<i64> = Vec::with_capacity(top_n);
    let mut dup_slots: Vec<usize> = Vec::new();
    for slot in 0..top_n {
        let ts = ix.session_ts[ix.turns[final_turn_indices[slot]].session];
        if ts > 0 {
            if seen_ts.contains(&ts) {
                if slot > 0 {
                    dup_slots.push(slot);
                }
            } else {
                seen_ts.push(ts);
            }
        }
    }

    for slot in dup_slots {
        let dup_sc = t_scene.get(final_turn_indices[slot]).copied().unwrap_or(0.0);
        if let Some(cand_pos) = (top_n..final_turn_indices.len()).find(|&idx| {
            let cti = final_turn_indices[idx];
            let cts = ix.session_ts[ix.turns[cti].session];
            cts > 0
                && !seen_ts.contains(&cts)
                && t_scene.get(cti).copied().unwrap_or(0.0) >= 0.45 * dup_sc
        }) {
            let cts = ix.session_ts[ix.turns[final_turn_indices[cand_pos]].session];
            seen_ts.push(cts);
            final_turn_indices.swap(slot, cand_pos);
        }
    }

    // Order slots 1..top_n in ascending timestamp lattice when all have valid timestamps.
    if top_n >= 3 {
        final_turn_indices[1..top_n].sort_by(|&a, &b| {
            let ta = ix.session_ts[ix.turns[a].session];
            let tb = ix.session_ts[ix.turns[b].session];
            match (ta > 0, tb > 0) {
                (true, true) => ta
                    .cmp(&tb)
                    .then_with(|| ix.turns[a].session.cmp(&ix.turns[b].session))
                    .then_with(|| a.cmp(&b)),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => a.cmp(&b),
            }
        });
    }
}
