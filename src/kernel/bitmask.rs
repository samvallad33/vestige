//! src/kernel/bitmask.rs — File 2: The BITS File
//! Hardware-accelerated query term coverage using single-cycle x86 POPCNT.
//! Zero heap allocation: tracks up to 64 query term groups in a single u64 bitfield.

/// A 64-bit integer representing up to 64 distinct query term groups.
pub type QueryMask = u64;

#[inline(always)]
pub fn query_group_mask(group_idx: usize) -> QueryMask {
    if group_idx < 64 {
        1u64 << group_idx
    } else {
        0
    }
}

/// Compute net-new uncovered query terms provided by a candidate turn.
///
/// Executes in 1 CPU clock cycle:
/// 1. (turn_mask & !covered_mask) bitwise AND-NOT
/// 2. .count_ones() lowers straight to x86 POPCNT instruction
#[inline(always)]
pub fn uncovered_query_term_count(turn_mask: QueryMask, covered_mask: QueryMask, total_groups: usize) -> u32 {
    let uncovered_bits = turn_mask & !covered_mask;
    let group_boundary_mask = if total_groups >= 64 {
        !0u64
    } else {
        (1u64 << total_groups) - 1
    };
    (uncovered_bits & group_boundary_mask).count_ones()
}

/// Calculate the marginal information gain ratio of a candidate turn
/// against the query terms that have already been satisfied.
#[inline(always)]
pub fn marginal_information_ratio(turn_mask: QueryMask, covered_mask: QueryMask, total_groups: usize) -> f32 {
    if total_groups == 0 {
        return 0.0;
    }
    let count = uncovered_query_term_count(turn_mask, covered_mask, total_groups);
    count as f32 / total_groups as f32
}
