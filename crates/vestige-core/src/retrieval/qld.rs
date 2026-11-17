//! #821: Integer Query Likelihood with Dirichlet Prior Smoothing (Zhai & Lafferty 2001)
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! Standard Dirichlet-smoothed log-likelihood ratio over matched query terms:
//!   score(D, Q) = sum_{q in Q \cap D} w_q * log2(1 + tf(q, D) / (mu * P(q | C)))
//!               * shrinkage(dl, mu)
//! where `P(q | C)` is estimated from corpus `df(q) / (N * avdl)`, and every
//! logarithm uses exact integer-seeded `super::dph::log2_q16`.

use super::dph::{log2_q16, Q16};

/// Default Dirichlet prior `mu` in Q8 (`384 * 256` = 384 tokens for session/window, `48 * 256` for turns).
#[allow(dead_code)]
pub(super) const MU_TURN_Q8: u64 = 48 * 256;
#[allow(dead_code)]
pub(super) const MU_SESSION_Q8: u64 = 384 * 256;

/// Pure-integer term entry passed across the thin `rank.rs` boundary (Q8 fixed-point).
#[derive(Clone, Copy)]
pub(super) struct QldTermInput {
    pub tf_q8: [u64; 3],
    pub w_q8: u64,
    pub df_doc: u64,
}

/// Pure-integer bag statistics passed across the thin `rank.rs` boundary (Q8 fixed-point).
#[derive(Clone, Copy)]
pub(super) struct QldBagInput {
    pub dl_q8: [u64; 3],
    pub avdl_q8: [u64; 3],
    pub n_docs: u64,
}

/// Compute Dirichlet log-likelihood term gain in Q16.16:
///   `log2(1 + (tf * N * avdl) / (mu * df))`
#[inline]
pub(super) fn qld_term_q16(
    tf_q8: u64,
    avdl_q8: u64,
    n_docs: u64,
    df_doc: u64,
    mu_q8: u64,
) -> u64 {
    if tf_q8 == 0 || avdl_q8 == 0 || n_docs == 0 || df_doc == 0 || mu_q8 == 0 {
        return 0;
    }
    let num = (tf_q8 as u128) * (n_docs as u128) * (avdl_q8 as u128);
    let den = (mu_q8 as u128) * (df_doc as u128);
    let ratio_q16 = ((num << 16) / den.max(1)).min((u64::MAX / 2) as u128) as u64;
    let one_plus_q16 = (Q16 as u64).saturating_add(ratio_q16);
    log2_q16(one_plus_q16).max(0) as u64
}

/// Compute Dirichlet log-likelihood over a bag's pure-integer Q8 inputs and apply
/// Dirichlet document-length regularization in Q16.16. Zero floats anywhere in this module.
pub(super) fn qld_score_q16(bag: &QldBagInput, terms: &[QldTermInput], mu_q8: u64) -> u64 {
    let n_docs = bag.n_docs.max(1);
    let mut sum_q16 = 0u64;
    let eff_dl_q8 = bag.dl_q8[0]
        .saturating_add(bag.dl_q8[1])
        .saturating_add(bag.dl_q8[2])
        .max(256);

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
            let avdl_q8 = bag.avdl_q8[f].max(256);
            let term_q16 = qld_term_q16(tf_q8, avdl_q8, n_docs, t.df_doc, mu_q8);
            best_f_q16 = best_f_q16.max(term_q16);
        }
        if best_f_q16 > 0 {
            sum_q16 = sum_q16.saturating_add(((best_f_q16 as u128 * t.w_q8 as u128) >> 8) as u64);
        }
    }

    if sum_q16 == 0 {
        return 0;
    }

    // Soft Dirichlet length regularization `(1 + mu / (dl + mu)) / 2` in Q16.16
    let shrink_q16 = ((mu_q8 as u128 * Q16 as u128) / (eff_dl_q8.saturating_add(mu_q8) as u128)) as u64;
    let reg_q16 = ((Q16 as u64) + shrink_q16) >> 1;
    ((sum_q16 as u128 * reg_q16 as u128) >> 16) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qld_term_monotonic_in_tf_and_inverse_in_df() {
        let low_tf = qld_term_q16(256, 20 * 256, 500, 10, MU_TURN_Q8);
        let high_tf = qld_term_q16(3 * 256, 20 * 256, 500, 10, MU_TURN_Q8);
        assert!(high_tf > low_tf);

        let rare_df = qld_term_q16(256, 20 * 256, 500, 2, MU_TURN_Q8);
        let common_df = qld_term_q16(256, 20 * 256, 500, 100, MU_TURN_Q8);
        assert!(rare_df > common_df);
    }

    #[test]
    fn qld_score_prefers_concise_match_over_long_bag() {
        let short_bag = QldBagInput {
            dl_q8: [12 * 256, 0, 0],
            avdl_q8: [20 * 256, 20 * 256, 20 * 256],
            n_docs: 200,
        };
        let long_bag = QldBagInput {
            dl_q8: [400 * 256, 0, 0],
            avdl_q8: [20 * 256, 20 * 256, 20 * 256],
            n_docs: 200,
        };
        let terms = [QldTermInput { tf_q8: [256, 0, 0], w_q8: 256, df_doc: 4 }];
        assert!(qld_score_q16(&short_bag, &terms, MU_TURN_Q8) > qld_score_q16(&long_bag, &terms, MU_TURN_Q8));
    }
}
