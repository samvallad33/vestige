//! Lachmann's Stemmatics & *Eliminatio Codicum Descriptorum* (`Q16.16`, Lachmann 1850; Maas 1958).
//!
//! Distinguishes primary *archetype* witness sessions (introducing novel rare tokens / *Leitfehler*
//! at their First Appearance Datum) from *codices descripti* (later sessions whose rare-token set
//! is a pure subset of an earlier session's rare tokens with zero independent innovations).
//! Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use std::collections::BTreeSet;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const LACHMANN_COEFF_Q16: u64 = 13_107; // 0.20

/// Scores each session's stemmatic archetype originality in `Q16.16`:
/// counts first-appearance rare tokens (`FAD == si`) and zeros out pure derivative copies
/// (*codices descripti* where `rare[si] ⊆ rare[earlier]` for some `earlier < si`).
pub(in crate::retrieval) fn compute_archetype_originality_q16(
    session_rare_nt: &[Vec<u32>],
) -> Vec<u64> {
    let ns = session_rare_nt.len();
    let mut seen_prior: BTreeSet<u32> = BTreeSet::new();
    let mut out = vec![0u64; ns];

    for (si, rares) in session_rare_nt.iter().enumerate() {
        if rares.is_empty() {
            continue;
        }
        // Eliminatio codicum descriptorum: check if an earlier session strictly contains all of `rares`.
        let is_descriptus = (0..si).any(|prev| {
            !session_rare_nt[prev].is_empty()
                && rares.iter().all(|tok| session_rare_nt[prev].binary_search(tok).is_ok())
        });
        if !is_descriptus {
            let novel_innovations = rares.iter().filter(|tok| !seen_prior.contains(tok)).count() as u64;
            out[si] = (novel_innovations.saturating_mul(Q16_ONE)) / (rares.len() as u64);
        }
        for &tok in rares {
            seen_prior.insert(tok);
        }
    }
    out
}

pub(in crate::retrieval) fn compute_lachmann_stemma_lifts_q16(
    ix: &Index,
    s_exact_mask: &[u64],
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let arch = compute_archetype_originality_q16(&ix.session_rare_nt);
    let top0_mask = s_exact_mask.get(top0).copied().unwrap_or(0);
    let adds_cov = |si: usize| -> bool {
        let m = s_exact_mask.get(si).copied().unwrap_or(0);
        m != 0 && ((m & !top0_mask) != 0 || top0_mask == 0)
    };
    let max_a = arch
        .iter()
        .enumerate()
        .filter(|&(si, _)| si != top0 && adds_cov(si))
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_a == 0 {
        return out;
    }
    for (si, &a) in arch.iter().enumerate() {
        if si != top0 && a > 0 && adds_cov(si) {
            let norm_q16 = (a.saturating_mul(Q16_ONE)) / max_a;
            out[si] = (((smax_q16.saturating_mul(LACHMANN_COEFF_Q16)) >> Q16_SHIFT)
                .saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eliminatio_codicum_descriptorum_zeros_derivative_copy() {
        let rares = vec![vec![10, 20, 30], vec![10, 20], vec![20, 40]];
        let sc = compute_archetype_originality_q16(&rares);
        assert_eq!(sc[0], Q16_ONE); // Archetype: 100% novel
        assert_eq!(sc[1], 0);       // Codex descriptus of session 0: eliminated!
        assert!(sc[2] > 0);         // Sibling witness introducing innovation 40: retained!
    }
}
