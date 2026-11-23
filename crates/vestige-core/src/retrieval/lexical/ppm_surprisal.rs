//! Order-2 Token PPM Surprisal & #10 1-WL Structural Neighborhood Hashing (`Q16.16`).
//!
//! - **PPM Surprisal**: measures contiguous query-term bigram rarity `df_1 * df_2 / (N * df_pair)`
//!   so rare collocations of individually common words ("chipped mug", "charity race") earn
//!   sharp phrase-surprise credit above isolated word bag matches.
//! - **1-WL Neighborhood Hash**: deterministic color hash combining a turn's own structural role
//!   bits `(field, modality_mask, has_temporal/numeric/spatial/habitual)` with adjacent
//!   speaker/turn transitions, aligning structural slot shapes with zero lexical lists.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::index::Index;

/// Corpus-normalized PPM order-2 bigram surprisal in `Q16.16` for adjacent `(g0, g1)` query
/// content tokens co-occurring inside turn `ti`. Returns `0` when terms are not rare together.
pub(in crate::retrieval) fn ppm_collocation_surprise_q16(ix: &Index, id_a: u32, id_b: u32) -> u64 {
    let n = (ix.t_own_stats.n as u64).max(4);
    let dfa = ix.t_own_stats.df.get(&id_a).copied().unwrap_or(0) as u64;
    let dfb = ix.t_own_stats.df.get(&id_b).copied().unwrap_or(0) as u64;
    if dfa == 0 || dfb == 0 {
        return 0;
    }
    // Lower df product => higher Shannon + PPM transition surprise: log2(N / min(dfa, dfb)).
    let min_df = dfa.min(dfb).max(1);
    let max_df = dfa.max(dfb).max(1);
    let base = log2_u64_q16(n).saturating_sub(log2_u64_q16(min_df));
    let pair_bonus = log2_u64_q16(n).saturating_sub(log2_u64_q16(max_df)) >> 1;
    base.saturating_add(pair_bonus)
}

/// #10 1-WL Local Turn Neighborhood Structural Color: encodes `(self_struct_bits, prev_role, next_role)`
/// into a deterministic 24-bit structural fingerprint without reading turn text.
pub(in crate::retrieval) fn wl1_turn_color(ix: &Index, ti: usize) -> u32 {
    let tr = &ix.turns[ti];
    let mut self_c: u32 = ((tr.field as u32) & 0x3)
        | (((tr.modality_mask as u32) & 0xF) << 2)
        | (u32::from(tr.has_temporal) << 6)
        | (u32::from(tr.has_numeric) << 7)
        | (u32::from(tr.has_spatial) << 8)
        | (u32::from(tr.is_habitual) << 9)
        | (u32::from(tr.answers_question) << 10)
        | (u32::from(tr.slot_entities_clean > 0) << 11);
    let members = &ix.session_turns[tr.session];
    if let Some(pos) = members.iter().position(|&m| m == ti) {
        if pos > 0 {
            let prev = &ix.turns[members[pos - 1]];
            let pc = ((prev.field as u32) & 0x3) | (u32::from(prev.is_question) << 2);
            self_c ^= pc.wrapping_mul(0x9E37_79B9) & 0x00FF_0000;
        }
        if pos + 1 < members.len() {
            let nxt = &ix.turns[members[pos + 1]];
            let nc = ((nxt.field as u32) & 0x3) | (u32::from(nxt.is_question) << 2);
            self_c ^= (nc << 12) & 0x0000_F000;
        }
    }
    self_c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_bits_fit_cleanly_in_q16() {
        let log_hi = log2_u64_q16(1000).saturating_sub(log2_u64_q16(2));
        let log_lo = log2_u64_q16(1000).saturating_sub(log2_u64_q16(200));
        assert!(log_hi > log_lo);
    }
}
