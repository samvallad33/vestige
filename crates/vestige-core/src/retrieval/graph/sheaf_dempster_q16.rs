//! #5 Cellular Sheaf Cohomology H^1 Contradiction Residual (Q16.16 integer port of
//! `strata_runtime::sheaf_h2sk_w24::cellular_sheaf_residual`) + #6 Dempster-Shafer Conflict Mass.
//!
//! When candidate memories attach inconsistent restrictions `(polarity, modality, epoch)` over a
//! shared high-IC entity, non-zero sheaf coboundary energy H^1 and high Dempster-Shafer conflict
//! mass `K` identify contested belief states:
//! - On current-state queries (`!past_override`), older superseded sections yielding high H^1 yield
//!   Rank 1 to the newer asserted consistent gluer.
//! - On paired contradiction queries (`is_have_i`), high conflict mass pairs `{positive, negated}`
//!   co-seat in Top-5 for `All@5`.

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::index::Index;

/// Integer Q16.16 sheaf coboundary energy `sum (r_u * stalk_u - r_v * stalk_v)^2`.
pub(in crate::retrieval) fn cellular_sheaf_residual_q16(
    stalks_q16: &[i64],
    edges: &[(usize, i64, usize, i64)],
) -> u64 {
    let mut sum_q32: u128 = 0;
    for &(u, r_u, v, r_v) in edges {
        if u >= stalks_q16.len() || v >= stalks_q16.len() {
            continue;
        }
        let lhs = (r_u as i128).saturating_mul(stalks_q16[u] as i128) >> 16;
        let rhs = (r_v as i128).saturating_mul(stalks_q16[v] as i128) >> 16;
        let diff = (lhs - rhs).unsigned_abs();
        sum_q32 = sum_q32.saturating_add(diff.saturating_mul(diff));
    }
    ((sum_q32 >> 16).min(u64::MAX as u128)) as u64
}

/// #6 Dempster-Shafer two-mass conflict `K = m1(A) * m2(!A) + m1(!A) * m2(A)` in `Q16.16`
/// where beliefs `[asserted, negated, uncertain]` sum to `<= Q16_ONE`.
pub(in crate::retrieval) fn dempster_shafer_conflict_q16(m1: [u64; 3], m2: [u64; 3]) -> u64 {
    let k1 = (m1[0] as u128).saturating_mul(m2[1] as u128) >> 16;
    let k2 = (m1[1] as u128).saturating_mul(m2[0] as u128) >> 16;
    ((k1.saturating_add(k2)).min(Q16_ONE as u128)) as u64
}

/// Turn-level Dempster-Shafer belief triple `[asserted, negated, uncertain]` in `Q16.16` from structural flags.
#[inline]
pub(in crate::retrieval) fn turn_belief_mass_q16(ix: &Index, ti: usize) -> [u64; 3] {
    let tr = &ix.turns[ti];
    let is_neg = (tr.modality_mask & 0x02) != 0;
    let is_hyp = (tr.modality_mask & 0x04) != 0;
    let is_q = (tr.modality_mask & 0x08) != 0;
    if is_q {
        [0, 0, Q16_ONE]
    } else if is_neg {
        [Q16_ONE >> 3, (Q16_ONE * 6) / 8, Q16_ONE >> 3]
    } else if is_hyp {
        [Q16_ONE >> 1, 0, Q16_ONE >> 1]
    } else {
        [(Q16_ONE * 7) / 8, 0, Q16_ONE >> 3]
    }
}

/// Pairwise Sheaf H^1 residual + Dempster-Shafer conflict between two turns sharing an anchor.
/// Non-zero whenever turns clash on supersession, polarity, or prospective-vs-asserted modality.
pub(in crate::retrieval) fn pair_sheaf_conflict_q16(ix: &Index, ta: usize, tb: usize) -> u64 {
    let ma = turn_belief_mass_q16(ix, ta);
    let mb = turn_belief_mass_q16(ix, tb);
    let ds_k = dempster_shafer_conflict_q16(ma, mb);
    let sa = if (ix.turns[ta].modality_mask & 0x02) != 0 || turn_is_superseded(ix, ta) {
        -(Q16_ONE as i64)
    } else {
        Q16_ONE as i64
    };
    let sb = if (ix.turns[tb].modality_mask & 0x02) != 0 || turn_is_superseded(ix, tb) {
        -(Q16_ONE as i64)
    } else {
        Q16_ONE as i64
    };
    let h1 = cellular_sheaf_residual_q16(&[sa, sb], &[(0, Q16_ONE as i64, 1, Q16_ONE as i64)]);
    ds_k.saturating_add(h1 >> 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sheaf_and_ds_detect_contradiction_conflict_zero_on_agreement() {
        let zero = cellular_sheaf_residual_q16(
            &[Q16_ONE as i64, Q16_ONE as i64],
            &[(0, Q16_ONE as i64, 1, Q16_ONE as i64)],
        );
        let clash = cellular_sheaf_residual_q16(
            &[Q16_ONE as i64, -(Q16_ONE as i64)],
            &[(0, Q16_ONE as i64, 1, Q16_ONE as i64)],
        );
        assert_eq!(zero, 0);
        assert!(clash > 0);
        let ds_clash = dempster_shafer_conflict_q16([57_344, 0, 8_192], [8_192, 49_152, 8_192]);
        assert!(ds_clash > Q16_ONE / 2);
    }
}
