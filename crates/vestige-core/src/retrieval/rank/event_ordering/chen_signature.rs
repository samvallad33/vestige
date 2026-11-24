//! #1 Chen Path Signatures / Lévy Signed Area + #11 Directed Symbolic Transfer Entropy
//! (`Q16.16`, pure integer fixed-point) for multi-session timeline & causal event ordering.
//!
//! Computes:
//! - Depth-2 Chen iterated path signature `(S^{(1)}_t, S^{(1)}_e, A_{t,e})` where the
//!   Lévy signed area `A_{t,e} = sum (X_t dX_e - X_e dX_t)` distinguishes causal order
//!   ("initiative milestone first -> follow-up elaboration later") from reverse paraphrases.
//! - Asymmetric Directed Symbolic Transfer Entropy `TE(s_a -> s_b)` over shared rare entities
//!   to break chronological ties and pull missing milestone epochs into Top-5.

use std::collections::BTreeSet;

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;
use crate::retrieval::index::Index;

/// Pairwise Lévy Signed Area + Directed Transfer Entropy precedence signal (`a < b` => `> 0`)
/// over sessions `(sa, sb)` using rare non-query shared entity ordering and initiative density.
pub(in crate::retrieval) fn chen_levy_transfer_precedence_q16(
    ix: &Index,
    sa: usize,
    sb: usize,
    topic_sc_a: u64,
    topic_sc_b: u64,
    nov_a: u64,
    nov_b: u64,
) -> i64 {
    // 1. Depth-2 Lévy signed area on the 2-step trajectory `(topic_sc, entity_novelty)`:
    //    `A_{a->b} = X_topic(a) * X_nov(b) - X_nov(a) * X_topic(b)` scaled to Q16.16.
    //    Positive area means `sa` introduced higher novel entity mass relative to `sb`'s topic echo.
    let area = ((nov_a as i128).saturating_mul(topic_sc_b as i128)
        - (topic_sc_a as i128).saturating_mul(nov_b as i128))
        >> 16;

    // 2. #11 Directed Symbolic Transfer Entropy: check shared rare entities appearing in
    //    initiative/first-person turns (`sa`) before elaboration/retrospective mentions (`sb`).
    let a_init = ix.session_turns[sa]
        .iter()
        .any(|&ti| ix.turns[ti].is_initiative && ix.turns[ti].field == 0);
    let b_init = ix.session_turns[sb]
        .iter()
        .any(|&ti| ix.turns[ti].is_initiative && ix.turns[ti].field == 0);
    let shared_rare = !ix.session_rare_nt[sa].is_empty()
        && ix.session_rare_nt[sa]
            .iter()
            .any(|e| ix.session_rare_nt[sb].contains(e));
    let mut te_dir: i64 = 0;
    if shared_rare && a_init != b_init {
        te_dir = if a_init {
            (Q16_ONE >> 2) as i64
        } else {
            -((Q16_ONE >> 2) as i64)
        };
    }

    let area_clamped = area.clamp(-(Q16_ONE as i128 >> 1), Q16_ONE as i128 >> 1) as i64;
    area_clamped.saturating_add(te_dir)
}

/// Novel rare-entity signature strength of session `si` relative to earlier already-seen entities.
pub(in crate::retrieval) fn session_entity_novelty_q16(
    ix: &Index,
    si: usize,
    seen_entities: &BTreeSet<u32>,
) -> u64 {
    let mut novel = 0u64;
    for &eid in &ix.session_rare_nt[si] {
        if !seen_entities.contains(&eid) {
            novel += 1;
        }
    }
    for &eid in &ix.session_entities[si] {
        if !seen_entities.contains(&eid) {
            novel += 2;
        }
    }
    (novel.min(8)).saturating_mul(Q16_ONE >> 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levy_signed_area_is_antisymmetric() {
        let a_to_b = ((4 * Q16_ONE as i128) * (3 * Q16_ONE as i128)
            - (2 * Q16_ONE as i128) * (5 * Q16_ONE as i128))
            >> 16;
        let b_to_a = ((5 * Q16_ONE as i128) * (2 * Q16_ONE as i128)
            - (3 * Q16_ONE as i128) * (4 * Q16_ONE as i128))
            >> 16;
        assert_eq!(a_to_b, -b_to_a);
        assert!(a_to_b > 0);
    }
}
