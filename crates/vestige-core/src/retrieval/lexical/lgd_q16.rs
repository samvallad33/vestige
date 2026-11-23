//! Clinchant & Gaussier Information-Based Log-Logistic (`LGD`) Scorer in Q16.16 integer math.
//!
//! `LGD(q, d) = sum_{t in q cap d} w_q(t) * -log2( lambda_t / (tf_norm(t, d) + lambda_t) )`
//! where `lambda_t = df_t / N` and `tf_norm = tf * log2(1 + avgdl / dl)`.
//! Empirically sharper than Okapi BM25 on short conversational turns with 1..2 rare factual tokens.

use crate::retrieval::index::{Bag, Stats};
use crate::retrieval::QueryPlan;

const Q16_ONE: u64 = 65_536;

/// Fast deterministic integer `log2(x)` in Q16.16 for `x >= Q16_ONE` (`x` scaled by Q16).
#[inline]
fn log2_q16_of_scaled(x_q16: u64) -> u64 {
    if x_q16 <= Q16_ONE {
        return 0;
    }
    // Integer part `k = floor(log2(x_q16 / Q16_ONE))`
    let lz = x_q16.leading_zeros();
    let bit = 63 - lz; // position of MSB
    let int_part = bit.saturating_sub(16) as u64;
    // Linear fractional interpolation on mantissa in `[1, 2)`
    let mant_q16 = if bit >= 16 {
        x_q16 >> (bit - 16)
    } else {
        x_q16 << (16 - bit)
    };
    let frac = mant_q16.saturating_sub(Q16_ONE);
    (int_part << 16).saturating_add(frac)
}

/// Compute Clinchant-Gaussier Log-Logistic `LGD` bag score in Q16.16 integer math.
pub(in crate::retrieval) fn lgd_bag_q16(bag: &Bag, plan: &QueryPlan, stats: &Stats) -> u64 {
    if stats.n <= 2.0 {
        return 0;
    }
    let dl: f32 = bag.len.iter().sum();
    if dl <= 0.0 {
        return 0;
    }
    let avgdl: f32 = stats.avg.iter().sum::<f32>().max(1.0);
    // H2 length normalization factor in Q16: `log2(1 + avgdl / dl)`
    let ratio_f32: f32 = (1.0f32 + avgdl / dl).clamp(1.0, 64.0);
    let ratio_q16 = (ratio_f32 * 65_536.0) as u64;
    let h2_q16 = log2_q16_of_scaled(ratio_q16).max(Q16_ONE >> 2);
    let n_u64 = (stats.n as u64).max(2);

    let mut total: u64 = 0;
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let mut tf_sum = 0.0f32;
        for f in 0..3 {
            if let Some(&v) = bag.tf[f].get(&qt.id) {
                tf_sum += v;
            }
        }
        if tf_sum <= 0.0 {
            continue;
        }
        let df = stats.df.get(&qt.id).copied().unwrap_or(1).max(1) as u64;
        // `lambda = df / N` -> `1 + tf_norm / lambda = 1 + tf_norm * N / df`
        let tf_q16 = (tf_sum.min(64.0) * 65_536.0) as u64;
        let tf_norm_q16 = ((tf_q16 as u128 * h2_q16 as u128) >> 16).max(1) as u64;
        let arg_q16 = Q16_ONE.saturating_add(
            ((tf_norm_q16 as u128 * n_u64 as u128) / (df as u128)).min((Q16_ONE as u128) * 4096)
                as u64,
        );
        let gain_q16 = log2_q16_of_scaled(arg_q16);
        let qw_q16 = (qt.weight.clamp(0.05, 4.0) * 65_536.0) as u64;
        total = total.saturating_add(((gain_q16 as u128 * qw_q16 as u128) >> 16) as u64);
    }
    total
}
