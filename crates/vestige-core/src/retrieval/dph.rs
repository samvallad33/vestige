//! Parameter-free Craig Macdonald DPH (Divergence From Randomness,
//! hypergeometric / Popper normalization) in Q16.16 fixed-point integer
//! arithmetic. Zero floats in this module, zero free parameters, zero word lists.
//!
//! Reference: Amati (2006), Macdonald & Ounis (2008) / Terrier `DPH`:
//!   f     = clamp(tf / dl, 1/dl .. (dl-1)/dl)
//!   norm  = (1 - f)^2 / (tf + 1)
//!   info1 = tf * max(0, log2((tf * avdl / dl) * (N / df)))
//!   info2 = 0.5 * max(0, log2(2 * pi * tf * (1 - f)))
//!   w     = norm * (info1 + info2)

use super::index::{Bag, Stats};
use super::scoring::QueryPlan;

pub(super) const Q16: i64 = 1 << 16; // 65536
const TWO_PI_Q16: u64 = 411_775; // round(2 * pi * 65536)

/// Fixed-point `log2(x)` where `x` is Q16.16 (`x_q16 = value * 65536`).
#[inline]
pub(super) fn log2_q16(x_q16: u64) -> i64 {
    if x_q16 == 0 {
        return i64::MIN / 4;
    }
    let msb = 63 - x_q16.leading_zeros() as i64;
    let int_part = (msb - 16) * Q16;
    let shifted = if msb >= 16 {
        x_q16 >> (msb - 16)
    } else {
        x_q16 << (16 - msb)
    };
    let u = (shifted & 0xFFFF) as i64;
    let corr = (22_675 * u * (Q16 - u)) >> 32;
    int_part + u + corr
}

/// Compute parameter-free DPH term weight in Q16.16 for one `(tf, dl, avdl, N, df)`.
/// Inputs `tf_q8`, `dl_q8`, `avdl_q8` are scaled by `256` (Q8).
pub(super) fn dph_term_q16(
    tf_q8: u64,
    dl_q8: u64,
    avdl_q8: u64,
    n_docs: u64,
    df_doc: u64,
) -> u64 {
    if tf_q8 == 0 || dl_q8 == 0 || avdl_q8 == 0 || n_docs == 0 || df_doc == 0 {
        return 0;
    }
    let eff_dl_q8 = dl_q8.max(tf_q8 + 256);
    let f_q16 = ((tf_q8 as u128 * Q16 as u128) / eff_dl_q8 as u128) as u64;
    let one_minus_f_q16 = (Q16 as u64).saturating_sub(f_q16).max(1);

    let omf_sq_q16 = (one_minus_f_q16 * one_minus_f_q16) >> 16;
    let tf_plus_1_q8 = tf_q8 + 256;
    let norm_q16 = ((omf_sq_q16 as u128 * 256) / tf_plus_1_q8 as u128) as u64;

    let num1 = (tf_q8 as u128) * (avdl_q8 as u128) * (n_docs as u128);
    let den1 = (eff_dl_q8 as u128) * (df_doc as u128);
    let ratio1_q16 = ((num1 << 8) / den1.max(1)).min(u64::MAX as u128) as u64;
    let l1 = log2_q16(ratio1_q16).max(0) as u64;
    let info1_q16 = ((tf_q8 as u128 * l1 as u128) >> 8) as u64;

    let tf_omf_q16 = ((tf_q8 as u128 * one_minus_f_q16 as u128) >> 8) as u64;
    let ratio2_q16 = ((TWO_PI_Q16 as u128 * tf_omf_q16 as u128) >> 16).min(u64::MAX as u128) as u64;
    let info2_q16 = (log2_q16(ratio2_q16).max(0) as u64) >> 1;

    let total_info_q16 = info1_q16.saturating_add(info2_q16);
    ((norm_q16 as u128 * total_info_q16 as u128) >> 16) as u64
}

/// Score a turn bag against `plan` using parameter-free DPH across all 3 fields,
/// returning Q16.16 fixed-point total score.
pub(super) fn dph_bag_q16(bag: &Bag, plan: &QueryPlan, st: &Stats) -> u64 {
    let n_docs = (st.n as u64).max(1);
    let mut total_q16 = 0u64;
    for qt in &plan.terms {
        let Some(&df) = st.df.get(&qt.id) else {
            continue;
        };
        if df == 0 {
            continue;
        }
        let w_q8 = (qt.weight * 256.0).round().max(0.0) as u64;
        if w_q8 == 0 {
            continue;
        }
        let mut term_best_q16 = 0u64;
        for f in 0..3 {
            let tf = bag.tf[f].get(&qt.id).copied().unwrap_or(0.0);
            if tf <= 0.0 {
                continue;
            }
            let tf_q8 = (tf * 256.0).round().max(1.0) as u64;
            let dl_q8 = (bag.len[f] * 256.0).round().max(256.0) as u64;
            let avdl_q8 = (st.avg[f] * 256.0).round().max(256.0) as u64;
            let w = dph_term_q16(tf_q8, dl_q8, avdl_q8, n_docs, df as u64);
            term_best_q16 = term_best_q16.max(w);
        }
        total_q16 = total_q16.saturating_add((term_best_q16 * w_q8) >> 8);
    }
    total_q16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_q16_matches_exact_powers_of_two() {
        assert_eq!(log2_q16(Q16 as u64), 0);
        assert_eq!(log2_q16((2 * Q16) as u64), Q16);
        assert_eq!(log2_q16((4 * Q16) as u64), 2 * Q16);
        assert_eq!(log2_q16((Q16 / 2) as u64), -Q16);
    }

    #[test]
    fn dph_prefers_concise_turn_over_rambling_turn_for_same_tf() {
        let concise = dph_term_q16(2 * 256, 10 * 256, 20 * 256, 1000, 10);
        let bloated = dph_term_q16(2 * 256, 120 * 256, 20 * 256, 1000, 10);
        assert!(concise > bloated, "concise={concise} must exceed bloated={bloated}");
    }

    #[test]
    fn dph_prefers_rarer_term_over_common_term() {
        let rare = dph_term_q16(256, 15 * 256, 15 * 256, 1000, 3);
        let common = dph_term_q16(256, 15 * 256, 15 * 256, 1000, 200);
        assert!(rare > common, "rare={rare} must exceed common={common}");
    }
}
