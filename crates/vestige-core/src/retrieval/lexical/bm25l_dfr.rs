//! #823: Lv & Zhai (2011) BM25L length-shift + #825: Amati & Van Rijsbergen (2002)
//! DFR `InL2` Laplace `1 / (tf + 1)` after-effect in pure Q16.16 fixed-point integer arithmetic.
//! Zero floats, zero open-class word lists.
//!
//! 1. BM25L shifts length-normalized term frequency `c(q, D)` by `delta = 0.5` (`128` in Q8)
//!    so long multi-topic sessions are not over-penalized when `c(q, D) > 0`:
//!      `c_q8 = (tf_q8 * 256) / (256 - b_q8 + (b_q8 * dl_q8) / avdl_q8)`
//!      `c_plus_q8 = c_q8 + 128`
//!      `tf_bm25l_q16 = ((k1_q8 + 256) * c_plus_q8 * 65536) / (k1_q8 + c_plus_q8)`
//! 2. Amati DFR Laplace after-effect (`L` normalization: `1 / (tfn + 1)`) weights randomness
//!    divergence by diminishing marginal return on repeated bursty mentions:
//!      `dfr_ae_q16 = (c_q8 << 16) / (c_q8 + 256)`

use crate::retrieval::lexical::dph::{log2_q16, Q16};
use crate::retrieval::lexical::qld::{QldBagInput, QldTermInput};

const K1_Q8: u64 = 307;    // 1.20 * 256
const B_Q8: u64 = 166;     // 0.65 * 256
const DELTA_Q8: u64 = 128; // 0.50 * 256

/// Compute combined BM25L + Amati DFR `InL2` term score in Q16.16.
#[inline]
pub(in crate::retrieval) fn bm25l_dfr_term_q16(
    tf_q8: u64,
    dl_q8: u64,
    avdl_q8: u64,
    n_docs: u64,
    df_doc: u64,
) -> u64 {
    if tf_q8 == 0 || dl_q8 == 0 || avdl_q8 == 0 || n_docs == 0 || df_doc == 0 {
        return 0;
    }
    let norm_den_q8 = (256u64.saturating_sub(B_Q8))
        .saturating_add(((B_Q8 as u128 * dl_q8 as u128) / avdl_q8.max(1) as u128) as u64)
        .max(64);
    let c_q8 = ((tf_q8 as u128 * 256) / norm_den_q8 as u128) as u64;
    if c_q8 == 0 {
        return 0;
    }

    // #823 BM25L saturation with delta = 0.5 (`128` in Q8)
    let c_plus_q8 = c_q8.saturating_add(DELTA_Q8);
    let bm25l_sat_q16 = (((K1_Q8 + 256) as u128 * c_plus_q8 as u128 * Q16 as u128)
        / ((K1_Q8 + c_plus_q8).max(1) as u128 * 256)) as u64;

    // #825 Amati DFR Laplace after-effect `tfn / (tfn + 1)`
    let dfr_ae_q16 = ((c_q8 as u128 * Q16 as u128) / (c_q8 + 256) as u128) as u64;

    // Collection randomness divergence `log2(1 + (N + 1) / df)` in Q16.16
    let idf_ratio_q16 = (Q16 as u64).saturating_add(
        ((((n_docs + 1) as u128) << 16) / df_doc.max(1) as u128).min((u64::MAX / 2) as u128) as u64,
    );
    let idf_q16 = log2_q16(idf_ratio_q16).max(0) as u64;

    let sat_blend_q16 = (bm25l_sat_q16 + dfr_ae_q16) >> 1;
    ((sat_blend_q16 as u128 * idf_q16 as u128) >> 16) as u64
}

/// Score a bag using pure-integer Q8 inputs across 3 fields, returning Q16.16 fixed-point score.
pub(in crate::retrieval) fn bm25l_dfr_score_q16(bag: &QldBagInput, terms: &[QldTermInput]) -> u64 {
    let n_docs = bag.n_docs.max(1);
    let mut total_q16 = 0u64;
    for t in terms {
        if t.df_doc == 0 || t.w_q8 == 0 {
            continue;
        }
        let mut best_f_q16 = 0u64;
        for f in 0..3 {
            let tf_q8 = t.tf_q8[f];
            if tf_q8 == 0 {
                continue;
            }
            let dl_q8 = bag.dl_q8[f].max(256);
            let avdl_q8 = bag.avdl_q8[f].max(256);
            let w = bm25l_dfr_term_q16(tf_q8, dl_q8, avdl_q8, n_docs, t.df_doc);
            best_f_q16 = best_f_q16.max(w);
        }
        if best_f_q16 > 0 {
            total_q16 = total_q16.saturating_add(((best_f_q16 as u128 * t.w_q8 as u128) >> 8) as u64);
        }
    }
    total_q16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25l_dfr_rewards_higher_normalized_tf_with_diminishing_returns() {
        let tf1 = bm25l_dfr_term_q16(256, 30 * 256, 30 * 256, 100, 5);
        let tf2 = bm25l_dfr_term_q16(2 * 256, 30 * 256, 30 * 256, 100, 5);
        let tf3 = bm25l_dfr_term_q16(3 * 256, 30 * 256, 30 * 256, 100, 5);
        assert!(tf2 > tf1 && tf3 > tf2);
        assert!(tf2 - tf1 > tf3 - tf2);
    }
}
