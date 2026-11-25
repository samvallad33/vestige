//! #827: Anderson (1974) ACT-R Fan Effect associative strength multiplier
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.

use crate::retrieval::lexical::dph::{log2_q16, Q16};

/// Anderson (1974) ACT-R Fan Effect associative strength multiplier in `Q16.16`:
///
///   `S_j = S_max - log2(max(1, fan_j))`
///
/// Normalized monotonically into `[0.55, 1.35]` (`36_045..=88_474` in `Q16.16`) so
/// one-of-a-kind personal anchors (`fan == 1..2`) are boosted up to `1.35x` while
/// high-fan ambient words (`fan` approaching `n_docs`) attenuate cleanly down to `0.55x`.
pub(in crate::retrieval) fn actr_fan_strength_q16(fan: u32, n_docs: u32) -> u64 {
    const Q16_U64: u64 = 65_536;
    const FAN_MIN_Q16: u64 = 36_045; // 0.55
    const FAN_MAX_Q16: u64 = 88_474; // 1.35
    let n = (n_docs as u64).max(2);
    let f = (fan as u64).clamp(1, n);
    let log_n = log2_q16(n.saturating_mul(Q16_U64)).max(Q16) as u64;
    let log_f = log2_q16(f.saturating_mul(Q16_U64)).max(0) as u64;
    let s_num = log_n.saturating_sub(log_f);
    // Ratio in [0, Q16]: 1.0 when fan == 1, 0.0 when fan == n.
    let ratio_q16 = (((s_num as u128) << 16) / (log_n as u128)).min(Q16_U64 as u128) as u64;
    let span = FAN_MAX_Q16 - FAN_MIN_Q16;
    FAN_MIN_Q16.saturating_add(((span as u128).saturating_mul(ratio_q16 as u128) >> 16) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actr_fan_effect_boosts_unique_anchors_and_attenuates_high_fan() {
        let rare = actr_fan_strength_q16(1, 200);
        let mid = actr_fan_strength_q16(10, 200);
        let hub = actr_fan_strength_q16(150, 200);
        assert!(rare > 65_536, "unique anchor should exceed 1.0x: {rare}");
        assert!(rare > mid && mid > hub, "strict fan monotonicity failed: {rare}, {mid}, {hub}");
        assert!(hub < 65_536, "high-fan cue should attenuate below 1.0x: {hub}");
    }
}
