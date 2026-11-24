//! Issues #805, #808, #812: Allen Interval Algebra, Temporal Contradiction Pruning,
//! and Pure-Integer Q16.16 Box-Cox / Power-Law Normalization.
//!
//! Zero floats (`f32`/`f64`), zero open-class word lists.
//! Computes deterministic Allen's 13 interval relations over session timestamps and
//! turn order, verifies temporal contradiction consistency on candidate tail chains,
//! and applies concave Box-Cox (lambda = 0 and lambda = 1/2) damping in Q16.16.

use crate::retrieval::graph::ghostlink_dgs::{integer_sqrt, Q16};
use crate::retrieval::index::Index;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::QueryIntent;

/// Allen's 13 mutually exclusive interval relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum AllenRel {
    Before,
    After,
    Meets,
    MetBy,
    Overlaps,
    OverlappedBy,
    Starts,
    StartedBy,
    During,
    Contains,
    Finishes,
    FinishedBy,
    Equals,
}

/// Closed integer time interval `[start, end]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct TimeInterval {
    pub(in crate::retrieval) start: i64,
    pub(in crate::retrieval) end: i64,
}

/// Classify the exact Allen interval relation between intervals `a` and `b`.
#[inline]
pub(in crate::retrieval) fn classify_allen_relation(a: TimeInterval, b: TimeInterval) -> AllenRel {
    if a.start == b.start && a.end == b.end {
        AllenRel::Equals
    } else if a.end < b.start {
        AllenRel::Before
    } else if a.start > b.end {
        AllenRel::After
    } else if a.end == b.start {
        AllenRel::Meets
    } else if a.start == b.end {
        AllenRel::MetBy
    } else if a.start < b.start && a.end > b.start && a.end < b.end {
        AllenRel::Overlaps
    } else if b.start < a.start && b.end > a.start && b.end < a.end {
        AllenRel::OverlappedBy
    } else if a.start == b.start && a.end < b.end {
        AllenRel::Starts
    } else if a.start == b.start && a.end > b.end {
        AllenRel::StartedBy
    } else if a.start > b.start && a.end < b.end {
        AllenRel::During
    } else if a.start < b.start && a.end > b.end {
        AllenRel::Contains
    } else if a.end == b.end && a.start > b.start {
        AllenRel::Finishes
    } else {
        AllenRel::FinishedBy
    }
}

/// Build an exact integer `[start, end]` interval for turn `ti` from its session
/// timestamp and ordinal index.
#[inline]
pub(in crate::retrieval) fn turn_time_interval(ix: &Index, ti: usize) -> TimeInterval {
    let sess = ix.turns[ti].session;
    let ts = ix.session_ts.get(sess).copied().unwrap_or(0);
    let base = ts.saturating_mul(1024).saturating_add((ti & 1023) as i64);
    TimeInterval {
        start: base,
        end: base + 1,
    }
}

/// #808 Temporal Contradiction Check:
/// True when placing `earlier_ti` ahead of `later_ti` violates causal state update ordering
/// for current-state queries (`later_ti` Allen-follows `earlier_ti` AND `later_ti` supersedes `earlier_ti`).
#[inline]
pub(in crate::retrieval) fn allen_pair_consistent(ix: &Index, it: &QueryIntent, a_ti: usize, b_ti: usize) -> bool {
    if !it.current_state || it.is_event_sequence || it.temporal {
        return true;
    }
    let rel = classify_allen_relation(turn_time_interval(ix, a_ti), turn_time_interval(ix, b_ti));
    let superseded_by_b = ix.turns[b_ti].supersedes_turns.contains(&a_ti);
    !((rel == AllenRel::Before || rel == AllenRel::Meets) && superseded_by_b)
}

/// #812 Pure-Integer Box-Cox Transformation (`lambda = 1/2`) in Q16.16:
/// `T_{1/2}(x) = 2 * (sqrt(1 + x) - 1)` computed cleanly in fixed-point integer math.
/// Stabilizes skewed posting / conductance variance without floating point operations.
#[inline]
pub(in crate::retrieval) fn box_cox_half_q16(val_q16: u64) -> u64 {
    // 1.0 in Q16.16 is Q16. sqrt((1 + x) * Q16) in Q16.16 = integer_sqrt((Q16 + val_q16) * Q16).
    let one_plus_x_scaled = Q16.saturating_add(val_q16).saturating_mul(Q16);
    let root_q16 = integer_sqrt(one_plus_x_scaled);
    root_q16.saturating_sub(Q16).saturating_mul(2)
}

/// Refine redundant tail slots `5..k` (leaving `0..5` sealed) using Allen temporal consistency (#808)
/// and integer Box-Cox stabilized utility (#812) when a superseded tail slot co-occurs with
/// its newer superseding state in `current_state` mode.
pub(in crate::retrieval) fn apply_allen_boxcox_tail_refinement(
    ix: &Index,
    it: &QueryIntent,
    turn_score_q16: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
) {
    if k < 6 || final_turn_indices.len() <= 5 || !it.current_state || it.is_event_sequence || it.temporal
    {
        return;
    }
    let len = final_turn_indices.len().min(k);
    for slot in 5..len {
        let cand_ti = final_turn_indices[slot];
        if !turn_is_superseded(ix, cand_ti) {
            continue;
        }
        let inconsistent_with_head = final_turn_indices[..5]
            .iter()
            .any(|&head_ti| !allen_pair_consistent(ix, it, cand_ti, head_ti));
        if inconsistent_with_head {
            // Look in remaining tail for a non-superseded positive Box-Cox turn
            if let Some(swap_idx) = ((slot + 1)..len).find(|&j| {
                let tj = final_turn_indices[j];
                !turn_is_superseded(ix, tj)
                    && turn_score_q16[tj] > turn_score_q16[cand_ti]
                    && box_cox_half_q16(turn_score_q16[tj]) > box_cox_half_q16(turn_score_q16[cand_ti])
            }) {
                final_turn_indices.swap(slot, swap_idx);
            }
        }
    }
}

/// IA-RAG (`2606.06044`, Jun 2026) Directional Allen Interval Cone:
/// Maps closed-class temporal prepositions onto Allen forward (`R_F`), backward (`R_B`), or
/// overlap cones for sub-graph time tightening around an anchor event turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) enum AllenDirCone {
    None,
    Forward,  // "after", "since", "following"
    Backward, // "before", "prior to", "earlier than"
    Overlap,  // "during", "while"
}

pub(in crate::retrieval) fn detect_ia_rag_cone(q_text: &str) -> AllenDirCone {
    let low = q_text.to_ascii_lowercase();
    if low.contains(" before ") || low.contains(" prior to ") || low.contains(" earlier than ") {
        AllenDirCone::Backward
    } else if low.contains(" after ") || low.contains(" since ") || low.contains(" following ") {
        AllenDirCone::Forward
    } else if low.contains(" during ") || low.contains(" while ") {
        AllenDirCone::Overlap
    } else {
        AllenDirCone::None
    }
}

/// Returns `true` when `cand_ti` satisfies the requested IA-RAG Allen directional cone relative
/// to `anchor_ti` (`Before/Meets` for `Backward`, `After/MetBy` for `Forward`, same-session for `Overlap`).
pub(in crate::retrieval) fn satisfies_ia_rag_cone(
    ix: &Index,
    cone: AllenDirCone,
    anchor_ti: usize,
    cand_ti: usize,
) -> bool {
    let a = turn_time_interval(ix, cand_ti);
    let b = turn_time_interval(ix, anchor_ti);
    let rel = classify_allen_relation(a, b);
    match cone {
        AllenDirCone::None => true,
        AllenDirCone::Backward => matches!(rel, AllenRel::Before | AllenRel::Meets),
        AllenDirCone::Forward => matches!(rel, AllenRel::After | AllenRel::MetBy),
        AllenDirCone::Overlap => {
            ix.turns[cand_ti].session == ix.turns[anchor_ti].session
                || matches!(
                    rel,
                    AllenRel::Equals | AllenRel::During | AllenRel::Contains | AllenRel::Overlaps | AllenRel::OverlappedBy
                )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allen_relations_classify_all_core_cases() {
        let a = TimeInterval { start: 10, end: 20 };
        let b = TimeInterval { start: 30, end: 40 };
        assert_eq!(classify_allen_relation(a, b), AllenRel::Before);
        assert_eq!(classify_allen_relation(b, a), AllenRel::After);
        assert_eq!(
            classify_allen_relation(a, TimeInterval { start: 20, end: 30 }),
            AllenRel::Meets
        );
        assert_eq!(classify_allen_relation(a, a), AllenRel::Equals);
    }

    #[test]
    fn box_cox_half_is_monotone_and_zero_at_origin() {
        assert_eq!(box_cox_half_q16(0), 0);
        // x = 3.0 -> sqrt(4) - 1 = 1 -> * 2 = 2.0 in Q16.16
        assert_eq!(box_cox_half_q16(3 * Q16), 2 * Q16);
        assert!(box_cox_half_q16(8 * Q16) > box_cox_half_q16(3 * Q16));
    }
}
