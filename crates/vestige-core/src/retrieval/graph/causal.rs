//! Cycle G (#750): Talmudic `Machloket` Dormant-Branch Recovery
//! in pure deterministic integer arithmetic (zero floats, zero open-class word lists).

use crate::retrieval::index::Index;
use crate::retrieval::graph::supersession::turn_is_superseded;

/// #750 Talmudic `Machloket` Dormant-Branch Recovery:
/// Checks whether an earlier superseded turn `ti` was superseded ONLY by a turn `killer_ti`
/// that has itself subsequently been superseded (`turn_is_superseded(ix, killer_ti)`),
/// reopening the dormant branch as an eligible tail candidate when its defeating constraint lifts.
pub(in crate::retrieval) fn recover_machloket_dormant_turns(ix: &Index, max_scan: usize) -> Vec<usize> {
    let nt = ix.turns.len();
    let mut recovered: Vec<usize> = Vec::new();
    for killer_ti in 0..nt {
        if !turn_is_superseded(ix, killer_ti) {
            continue;
        }
        for &dormant_ti in &ix.turns[killer_ti].supersedes_turns {
            if dormant_ti < nt && !recovered.contains(&dormant_ti) {
                recovered.push(dormant_ti);
                if recovered.len() >= max_scan {
                    return recovered;
                }
            }
        }
    }
    recovered
}
