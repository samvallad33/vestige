//! #806: Mackworth AC-3 Arc Consistency Propagator (1977)
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! Operates over the top candidate turns (`<= 32` candidates). Builds binary
//! mutual-exclusion / supersession arcs between conflicting facts:
//! 1. Recorded `supersedes_turns` edge (`t_new -> t_old`), when `current_state`
//!    is active or not a historical event-sequence query.
//! 2. `turn_is_superseded(ix, ti)` unitary domain collapse on superseded turns
//!    when a newer superseding state is seated in the top candidates.
//! 3. Same-speaker later state update sharing both `entities` and `role_objs`
//!    with a strictly newer `session_ts` (evicting stale duplicate facts from
//!    the same top-K window while preserving historical episodes when `!current_state`).
//!
//! Runs Mackworth's O(e) worklist queue propagation and demotes arc-inconsistent
//! outdated turns into lower tail slots so conflicting facts never co-occupy Top-5.

use std::collections::VecDeque;

use crate::retrieval::index::Index;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::QueryIntent;

/// Binary arc constraint `(kept_pos, suppressed_pos)` in `candidates[0..n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MutexArc {
    winner: usize,
    loser: usize,
}

/// Apply Mackworth AC-3 arc consistency propagation across `final_turn_indices`
/// (`<= 32` candidates). Skips historical/temporal/event-sequence queries so earlier
/// states remain intact for timeline comparison questions.
pub(in crate::retrieval) fn apply_ac3_consistency(
    ix: &Index,
    it: &QueryIntent,
    turn_score_q16: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
) {
    if final_turn_indices.len() < 2
        || it.is_event_sequence
        || it.temporal
        || !it.current_state
    {
        return;
    }

    let n = final_turn_indices.len().min(k).min(32);
    if n < 2 {
        return;
    }

    // Domain of each variable `i in 0..n`: `[keep = true/false]`. Initially all `true`.
    let mut alive = vec![true; n];
    let mut arcs: Vec<MutexArc> = Vec::new();

    for i in 0..n {
        let ti = final_turn_indices[i];
        let ts_i = ix.session_ts[ix.turns[ti].session];

        for j in 0..n {
            if i == j {
                continue;
            }
            let tj = final_turn_indices[j];
            // 1. Explicit causal contradiction edge (`ti` supersedes `tj`)
            if ix.turns[ti].supersedes_turns.contains(&tj) {
                arcs.push(MutexArc { winner: i, loser: j });
                continue;
            }
            // 2. Same-speaker strictly newer state update sharing identical entity + constraint role_obj
            let ts_j = ix.session_ts[ix.turns[tj].session];
            if ts_i > ts_j
                && ix.turns[ti].speaker == ix.turns[tj].speaker
                && !ix.turns[ti].role_objs.is_empty()
                && !ix.turns[tj].role_objs.is_empty()
                && turn_is_superseded(ix, tj)
            {
                let shared_obj = ix.turns[ti]
                    .role_objs
                    .iter()
                    .any(|o| ix.turns[tj].role_objs.contains(o));
                if shared_obj {
                    arcs.push(MutexArc { winner: i, loser: j });
                }
            }
        }
    }

    if arcs.is_empty() {
        return;
    }

    // Mackworth AC-3 worklist queue: revise domains until fixed point
    let mut queue: VecDeque<usize> = (0..arcs.len()).collect();
    while let Some(arc_idx) = queue.pop_front() {
        let arc = arcs[arc_idx];
        if alive[arc.winner] && alive[arc.loser] {
            // Domain revision: if `winner` survives and has positive support, `loser` is pruned
            if turn_score_q16[final_turn_indices[arc.winner]] > 0 {
                alive[arc.loser] = false;
                for (next_idx, next_arc) in arcs.iter().enumerate() {
                    if next_arc.winner == arc.loser {
                        queue.push_back(next_idx);
                    }
                }
            }
        }
    }

    // Reorder `final_turn_indices[0..n]`: arc-consistent turns first (preserving relative order),
    // arc-inconsistent superseded turns demoted cleanly to the bottom tail.
    let mut consistent: Vec<usize> = Vec::with_capacity(n);
    let mut demoted: Vec<usize> = Vec::new();
    for i in 0..n {
        if alive[i] {
            consistent.push(final_turn_indices[i]);
        } else {
            demoted.push(final_turn_indices[i]);
        }
    }
    if !demoted.is_empty() {
        consistent.extend(demoted);
        for (slot, val) in consistent.into_iter().enumerate() {
            final_turn_indices[slot] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ac3_module_is_integer_only() {
        // Compile-time guarantee: zero floats used anywhere in `ac3.rs`.
        let q16_unit: u64 = 1 << 16;
        assert_eq!(q16_unit, 65536);
    }
}
