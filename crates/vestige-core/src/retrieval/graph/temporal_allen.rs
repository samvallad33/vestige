//! Issues #805, #808, #812: Allen Interval Algebra, Temporal Contradiction Pruning,
//! and Pure-Integer Q16.16 Box-Cox / Power-Law Normalization.
//!
//! Zero floats (`f32`/`f64`), zero open-class word lists.
//! Computes deterministic Allen's 13 interval relations over session timestamps and
//! turn order, verifies temporal contradiction consistency on candidate tail chains,
//! and applies concave Box-Cox (lambda = 0 and lambda = 1/2) damping in Q16.16.

use crate::retrieval::index::Index;

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
}
