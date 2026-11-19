//! Supersession lookups used by ranking (`turn_is_superseded`). Edges are built in `index::build_index`.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use crate::retrieval::*;

#[inline]
pub(in crate::retrieval) fn turn_is_superseded(ix: &Index, ti: usize) -> bool {
    ix.turn_superseded.get(ti).copied().unwrap_or(false)
        || (ix.turns.len() <= 64 && ix.turns.iter().any(|t| t.supersedes_turns.contains(&ti)))
}
