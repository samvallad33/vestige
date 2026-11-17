//! FSRS-6 Core Algorithm Implementation
//!
//! Implements the mathematical formulas for the FSRS-6 algorithm.
//! All functions are pure and deterministic for testability.

use super::scheduler::Rating;

// ============================================================================
// FSRS-6 CONSTANTS (21 Parameters)
// ============================================================================

/// FSRS-6 default weights (w0 to w20)
/// Trained on millions of Anki reviews - 20-30% more efficient than SM-2
pub const FSRS6_WEIGHTS: [f64; 21] = [
    0.212,  // w0: Initial stability for Again
    1.2931, // w1: Initial stability for Hard
    2.3065, // w2: Initial stability for Good
    8.2956, // w3: Initial stability for Easy
    6.4133, // w4: Initial difficulty base
    0.8334, // w5: Initial difficulty grade modifier
    3.0194, // w6: Difficulty delta
    0.001,  // w7: Difficulty mean reversion
    1.8722, // w8: Stability increase base
    0.1666, // w9: Stability saturation
    0.796,  // w10: Retrievability influence on stability
    1.4835, // w11: Forget stability base
    0.0614, // w12: Forget difficulty influence
    0.2629, // w13: Forget stability influence
    1.6483, // w14: Forget retrievability influence
    0.6014, // w15: Hard penalty
    1.8729, // w16: Easy bonus
    0.5425, // w17: Same-day review base (NEW in FSRS-6)
    0.0912, // w18: Same-day review grade modifier (NEW in FSRS-6)
    0.0658, // w19: Same-day review stability influence (NEW in FSRS-6)
    0.1542, // w20: Forgetting curve decay (NEW in FSRS-6 - PERSONALIZABLE)
];

/// Maximum difficulty value
pub const MAX_DIFFICULTY: f64 = 10.0;

/// Minimum difficulty value
pub const MIN_DIFFICULTY: f64 = 1.0;

/// Minimum stability value (days)
pub const MIN_STABILITY: f64 = 0.1;

/// Maximum stability value (days) - 100 years
pub const MAX_STABILITY: f64 = 36500.0;

/// Default desired retention rate (90%)
pub const DEFAULT_RETENTION: f64 = 0.9;

/// Default forgetting curve decay (w20)
pub const DEFAULT_DECAY: f64 = 0.1542;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Clamp value to range
#[inline]
fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.clamp(min, max)
}

/// Calculate forgetting curve factor based on w20
/// FSRS-6: factor = 0.9^(-1/w20) - 1
#[inline]
fn forgetting_factor(w20: f64) -> f64 {
    0.9_f64.powf(-1.0 / w20) - 1.0
}

// ============================================================================
// RETRIEVABILITY (Probability of Recall)
// ============================================================================

/// Calculate retrievability (probability of recall)
///
/// FSRS-6 formula: R = (1 + factor * t / S)^(-w20)
/// where factor = 0.9^(-1/w20) - 1
///
/// This is the power forgetting curve - more accurate than exponential
/// for modeling human memory.
///
/// # Arguments
/// * `stability` - Memory stability in days
/// * `elapsed_days` - Days since last review
///
/// # Returns
/// Probability of recall (0.0 to 1.0)
pub fn retrievability(stability: f64, elapsed_days: f64) -> f64 {
    retrievability_with_decay(stability, elapsed_days, DEFAULT_DECAY)
}

/// Retrievability with custom decay parameter (for personalization)
///
/// # Arguments
/// * `stability` - Memory stability in days
/// * `elapsed_days` - Days since last review
/// * `w20` - Forgetting curve decay parameter
pub fn retrievability_with_decay(stability: f64, elapsed_days: f64, w20: f64) -> f64 {
    if stability <= 0.0 {
        return 0.0;
    }
    if elapsed_days <= 0.0 {
        return 1.0;
    }

    let factor = forgetting_factor(w20);
    let r = (1.0 + factor * elapsed_days / stability).powf(-w20);
    clamp(r, 0.0, 1.0)
}

// ============================================================================
// INITIAL VALUES
// ============================================================================

/// Calculate initial difficulty for a grade
/// D0(G) = w4 - e^(w5*(G-1)) + 1
pub fn initial_difficulty(grade: Rating) -> f64 {
    initial_difficulty_with_weights(grade, &FSRS6_WEIGHTS)
}

/// Calculate initial difficulty with custom weights
pub fn initial_difficulty_with_weights(grade: Rating, weights: &[f64; 21]) -> f64 {
    let w4 = weights[4];
    let w5 = weights[5];
    let g = grade.as_i32() as f64;
    let d = w4 - (w5 * (g - 1.0)).exp() + 1.0;
    clamp(d, MIN_DIFFICULTY, MAX_DIFFICULTY)
}

/// Calculate initial stability for a grade
/// S0(G) = w[G-1] (weights 0-3 are initial stabilities)
pub fn initial_stability(grade: Rating) -> f64 {
    initial_stability_with_weights(grade, &FSRS6_WEIGHTS)
}

/// Calculate initial stability with custom weights
pub fn initial_stability_with_weights(grade: Rating, weights: &[f64; 21]) -> f64 {
    weights[grade.as_index()].max(MIN_STABILITY)
}

// ============================================================================
// DIFFICULTY UPDATES
// ============================================================================

/// Calculate next difficulty after review
///
/// FSRS-6 formula with mean reversion:
/// D' = w7 * D0(3) + (1 - w7) * (D + delta * ((10 - D) / 9))
/// where delta = -w6 * (G - 3)
pub fn next_difficulty(current_d: f64, grade: Rating) -> f64 {
    next_difficulty_with_weights(current_d, grade, &FSRS6_WEIGHTS)
}

/// Calculate next difficulty with custom weights
pub fn next_difficulty_with_weights(current_d: f64, grade: Rating, weights: &[f64; 21]) -> f64 {
    let w6 = weights[6];
    let w7 = weights[7];
    let g = grade.as_i32() as f64;
    // FSRS-6 spec: Mean reversion target is D0(4) = initial difficulty for Easy
    let d0 = initial_difficulty_with_weights(Rating::Easy, weights);

    // Delta based on grade deviation from "Good" (3)
    let delta = -w6 * (g - 3.0);

    // FSRS-6: Apply mean reversion scaling ((10 - D) / 9)
    let mean_reversion_scale = (10.0 - current_d) / 9.0;
    let new_d = current_d + delta * mean_reversion_scale;

    // Convex combination with initial difficulty for stability
    let final_d = w7 * d0 + (1.0 - w7) * new_d;
    clamp(final_d, MIN_DIFFICULTY, MAX_DIFFICULTY)
}

// ============================================================================
// STABILITY UPDATES
// ============================================================================

/// Calculate stability after successful recall
///
/// S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) * HP * EB + 1)
pub fn next_recall_stability(current_s: f64, difficulty: f64, r: f64, grade: Rating) -> f64 {
    next_recall_stability_with_weights(current_s, difficulty, r, grade, &FSRS6_WEIGHTS)
}

/// Calculate stability after successful recall with custom weights
pub fn next_recall_stability_with_weights(
    current_s: f64,
    difficulty: f64,
    r: f64,
    grade: Rating,
    weights: &[f64; 21],
) -> f64 {
    if grade == Rating::Again {
        return next_forget_stability_with_weights(difficulty, current_s, r, weights);
    }

    let w8 = weights[8];
    let w9 = weights[9];
    let w10 = weights[10];
    let w15 = weights[15];
    let w16 = weights[16];

    let hard_penalty = if grade == Rating::Hard { w15 } else { 1.0 };
    let easy_bonus = if grade == Rating::Easy { w16 } else { 1.0 };

    let factor = w8.exp()
        * (11.0 - difficulty)
        * current_s.powf(-w9)
        * ((w10 * (1.0 - r)).exp() - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0;

    clamp(current_s * factor, MIN_STABILITY, MAX_STABILITY)
}

/// Calculate stability after lapse (forgetting)
///
/// S'f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))
pub fn next_forget_stability(difficulty: f64, current_s: f64, r: f64) -> f64 {
    next_forget_stability_with_weights(difficulty, current_s, r, &FSRS6_WEIGHTS)
}

/// Calculate stability after lapse with custom weights
pub fn next_forget_stability_with_weights(
    difficulty: f64,
    current_s: f64,
    r: f64,
    weights: &[f64; 21],
) -> f64 {
    let w11 = weights[11];
    let w12 = weights[12];
    let w13 = weights[13];
    let w14 = weights[14];

    let new_s =
        w11 * difficulty.powf(-w12) * ((current_s + 1.0).powf(w13) - 1.0) * (w14 * (1.0 - r)).exp();

    // FSRS-6 spec: Post-lapse stability cannot exceed pre-lapse stability
    let new_s = new_s.min(current_s);

    clamp(new_s, MIN_STABILITY, MAX_STABILITY)
}

/// Calculate stability for same-day reviews (NEW in FSRS-6)
///
/// S'(S,G) = S * e^(w17 * (G - 3 + w18)) * S^(-w19)
pub fn same_day_stability(current_s: f64, grade: Rating) -> f64 {
    same_day_stability_with_weights(current_s, grade, &FSRS6_WEIGHTS)
}

/// Calculate stability for same-day reviews with custom weights
pub fn same_day_stability_with_weights(current_s: f64, grade: Rating, weights: &[f64; 21]) -> f64 {
    let w17 = weights[17];
    let w18 = weights[18];
    let w19 = weights[19];
    let g = grade.as_i32() as f64;

    let new_s = current_s * (w17 * (g - 3.0 + w18)).exp() * current_s.powf(-w19);
    clamp(new_s, MIN_STABILITY, MAX_STABILITY)
}

// ============================================================================
// INTERVAL CALCULATION
// ============================================================================

/// Calculate next interval in days
///
/// FSRS-6 formula (inverse of retrievability):
/// t = S / factor * (R^(-1/w20) - 1)
pub fn next_interval(stability: f64, desired_retention: f64) -> i32 {
    next_interval_with_decay(stability, desired_retention, DEFAULT_DECAY)
}

/// Calculate next interval with custom decay
pub fn next_interval_with_decay(stability: f64, desired_retention: f64, w20: f64) -> i32 {
    if stability <= 0.0 {
        return 0;
    }
    if desired_retention >= 1.0 {
        return 0;
    }
    if desired_retention <= 0.0 {
        return MAX_STABILITY as i32;
    }

    let factor = forgetting_factor(w20);
    let interval = stability / factor * (desired_retention.powf(-1.0 / w20) - 1.0);

    interval.max(0.0).round() as i32
}

// ============================================================================
// FUZZING
// ============================================================================

/// Apply interval fuzzing to prevent review clustering
///
/// Uses deterministic fuzzing based on a seed to ensure reproducibility.
pub fn fuzz_interval(interval: i32, seed: u64) -> i32 {
    if interval <= 2 {
        return interval;
    }

    // Use simple LCG for deterministic fuzzing
    let fuzz_range = (interval as f64 * 0.05).max(1.0) as i32;
    let random = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % 32768) as i32;
    let offset = (random % (2 * fuzz_range + 1)) - fuzz_range;

    (interval + offset).max(1)
}

// ============================================================================
// SENTIMENT BOOST
// ============================================================================

/// Apply sentiment boost to stability (emotional memories last longer)
///
/// Research shows emotional memories are encoded more strongly due to
/// amygdala modulation of hippocampal consolidation.
///
/// # Arguments
/// * `stability` - Current memory stability
/// * `sentiment_intensity` - Emotional intensity (0.0 to 1.0)
/// * `max_boost` - Maximum boost multiplier (typically 1.5 to 3.0)
pub fn apply_sentiment_boost(stability: f64, sentiment_intensity: f64, max_boost: f64) -> f64 {
    let clamped_sentiment = clamp(sentiment_intensity, 0.0, 1.0);
    let clamped_max_boost = clamp(max_boost, 1.0, 3.0);
    let boost = 1.0 + (clamped_max_boost - 1.0) * clamped_sentiment;
    stability * boost
}

// ============================================================================
// WAVE 5 EPIGENETIC FSRS DYNAMICS (#413, #414, #415, #416)
// ============================================================================

/// #413: Supersession-weighted decay floor. Discounts retrievability cleanly when
/// later `supersedes`, `corrects`, or `closed_by` edges contest this record.
pub fn apply_supersession_decay_floor(retrievability: f64, superseding_edges: u32) -> f64 {
    let r = clamp(retrievability, 0.0, 1.0);
    if superseding_edges == 0 {
        return r;
    }
    let discounted = r / (1.0 + 0.35 * (superseding_edges as f64));
    discounted.max(0.05_f64.min(r))
}

/// #414: Lock ingest-time arousal axis deterministically from explicit severity (`0..=3`).
pub fn lock_arousal_from_severity(explicit_severity: u8) -> f64 {
    match explicit_severity {
        0 => 0.0,
        1 => 0.35,
        2 => 0.70,
        _ => 1.0,
    }
}

/// #415: Modulate FSRS stability gain by retrieval effort (Bjork desirable difficulty).
pub fn modulate_stability_by_effort(base_stability: f64, rank_position: usize, hops: u8) -> f64 {
    let s = clamp(base_stability, MIN_STABILITY, MAX_STABILITY);
    let rank_term = 0.08 * (rank_position.min(10) as f64);
    let hop_term = 0.12 * (hops.min(3) as f64);
    clamp(s * (1.0 + rank_term + hop_term), MIN_STABILITY, MAX_STABILITY)
}

/// #416: Apply exposure-ledger retrieval dividend: strengthens memories cited via
/// `derived_from` / `evidence_of`, and accelerates decay for repeatedly ignored exposures.
pub fn apply_exposure_dividend(
    stability: f64,
    cited_after_exposure: bool,
    ignored_streak: u32,
) -> f64 {
    let s = clamp(stability, MIN_STABILITY, MAX_STABILITY);
    if cited_after_exposure {
        clamp(s * 1.15, MIN_STABILITY, MAX_STABILITY)
    } else if ignored_streak > 0 {
        clamp(s * 0.90_f64.powi(ignored_streak.min(10) as i32), MIN_STABILITY, MAX_STABILITY)
    } else {
        s
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_fsrs6_constants() {
        assert_eq!(FSRS6_WEIGHTS.len(), 21);
        let w20 = FSRS6_WEIGHTS[20];
        assert!(w20 > 0.0 && w20 < 1.0);
    }

    #[test]
    fn test_forgetting_factor() {
        let factor = forgetting_factor(DEFAULT_DECAY);
        assert!(factor > 0.0, "Factor should be positive");
        assert!(
            factor > 0.5 && factor < 5.0,
            "Expected factor between 0.5 and 5.0, got {}",
            factor
        );
    }

    #[test]
    fn test_retrievability_at_zero_days() {
        let r = retrievability(10.0, 0.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_retrievability_decreases_over_time() {
        let stability = 10.0;
        let r1 = retrievability(stability, 1.0);
        let r5 = retrievability(stability, 5.0);
        let r10 = retrievability(stability, 10.0);

        assert!(r1 > r5);
        assert!(r5 > r10);
        assert!(r10 > 0.0);
    }

    #[test]
    fn test_retrievability_with_custom_decay() {
        let stability = 10.0;
        let elapsed = 5.0;

        let r_low_decay = retrievability_with_decay(stability, elapsed, 0.1);
        let r_high_decay = retrievability_with_decay(stability, elapsed, 0.5);

        // Higher decay = faster forgetting (lower retrievability for same time)
        assert!(r_low_decay < r_high_decay);
    }

    #[test]
    fn test_next_interval_round_trip() {
        let stability = 15.0;
        let desired_retention = 0.9;

        let interval = next_interval(stability, desired_retention);
        let actual_r = retrievability(stability, interval as f64);

        assert!(
            approx_eq(actual_r, desired_retention, 0.05),
            "Round-trip: interval={}, R={}, desired={}",
            interval,
            actual_r,
            desired_retention
        );
    }

    #[test]
    fn test_initial_difficulty_order() {
        let d_again = initial_difficulty(Rating::Again);
        let d_hard = initial_difficulty(Rating::Hard);
        let d_good = initial_difficulty(Rating::Good);
        let d_easy = initial_difficulty(Rating::Easy);

        assert!(d_again > d_hard);
        assert!(d_hard > d_good);
        assert!(d_good > d_easy);
    }

    #[test]
    fn test_initial_difficulty_bounds() {
        for rating in [Rating::Again, Rating::Hard, Rating::Good, Rating::Easy] {
            let d = initial_difficulty(rating);
            assert!((MIN_DIFFICULTY..=MAX_DIFFICULTY).contains(&d));
        }
    }

    #[test]
    fn test_next_difficulty_mean_reversion() {
        let high_d = 9.0;
        let new_d = next_difficulty(high_d, Rating::Good);
        assert!(new_d < high_d);

        let low_d = 2.0;
        let new_d_low = next_difficulty(low_d, Rating::Again);
        assert!(new_d_low > low_d);
    }

    #[test]
    fn test_same_day_stability() {
        let current_s = 5.0;

        let s_again = same_day_stability(current_s, Rating::Again);
        let s_good = same_day_stability(current_s, Rating::Good);
        let s_easy = same_day_stability(current_s, Rating::Easy);

        assert!(s_again < s_good);
        assert!(s_good < s_easy);
    }

    #[test]
    fn test_fuzz_interval() {
        let interval = 30;
        let fuzzed1 = fuzz_interval(interval, 12345);
        let fuzzed2 = fuzz_interval(interval, 12345);

        // Same seed = same result (deterministic)
        assert_eq!(fuzzed1, fuzzed2);

        // Fuzzing should keep it close
        assert!((fuzzed1 - interval).abs() <= 2);
    }

    #[test]
    fn test_sentiment_boost() {
        let stability = 10.0;
        let boosted = apply_sentiment_boost(stability, 1.0, 2.0);
        assert_eq!(boosted, 20.0); // Full boost = 2x

        let partial = apply_sentiment_boost(stability, 0.5, 2.0);
        assert_eq!(partial, 15.0); // 50% boost = 1.5x
    }

    #[test]
    fn test_wave5_epigenetic_fsrs_functions() {
        assert!((apply_supersession_decay_floor(0.9, 0) - 0.9).abs() < 1e-9);
        assert!(apply_supersession_decay_floor(0.9, 2) < 0.9);
        assert!((apply_supersession_decay_floor(0.9, 100) - 0.05).abs() < 1e-9);

        assert_eq!(lock_arousal_from_severity(0), 0.0);
        assert_eq!(lock_arousal_from_severity(1), 0.35);
        assert_eq!(lock_arousal_from_severity(2), 0.70);
        assert_eq!(lock_arousal_from_severity(3), 1.0);

        assert!(modulate_stability_by_effort(10.0, 5, 2) > 10.0);
        assert!(apply_exposure_dividend(10.0, true, 0) > 10.0);
        assert!(apply_exposure_dividend(10.0, false, 3) < 10.0);
    }
}
