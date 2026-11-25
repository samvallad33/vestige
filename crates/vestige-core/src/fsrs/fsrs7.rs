//! FSRS-7 Core Algorithm Implementation (34 Parameters, Dual-Stability `(S_long, S_short, D)`).
//!
//! Reference: `open-spaced-repetition/fsrs-rs` v7.0.0.
//! Pure deterministic Rust with an 11-parameter two-component mixture forgetting curve
//! modeling both fast short-term consolidation (`S_short`) and slow long-term retention (`S_long`).

use super::algorithm::{MAX_DIFFICULTY, MAX_STABILITY, MIN_DIFFICULTY, MIN_STABILITY};
use super::scheduler::Rating;

/// Official FSRS-7 34-parameter default weights (`w[0..34]`).
pub const FSRS7_WEIGHTS: [f64; 34] = [
    0.1104, 2.2395, 3.9221, 11.7841, // w0..w3: initial S_long(G=1..=4)
    6.1686, 0.6457, 3.6807,          // w4..w6: D0 base/slope, difficulty delta
    1.9795, 0.0, 1.3826, 0.7024, 0.5999, 0.8146, 0.6398, 1.0, // w7..w14: S_long updates (a=7)
    1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, // w15..w22: S_short updates (a=15)
    0.1567, 0.0801, 0.2421, 0.9464,  // w23..w26: mixture decays & anchors
    0.1433, 0.7145, 0.0, 0.5667,     // w27..w30: mixture component weights
    0.3734, 0.5333, 0.3048,          // w31..w33: difficulty & short-stability modulation
];

/// Dual-stability FSRS-7 memory state `(S_long, S_short, D)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fsrs7State {
    /// Long-term memory stability in days (`S_long`).
    pub s_long: f64,
    /// Short-term memory stability in days (`S_short <= 0.8 * S_long`).
    pub s_short: f64,
    /// Memory difficulty in `[1.0, 10.0]`.
    pub difficulty: f64,
}

/// FSRS-7 initial difficulty `D0(G) = clamp(w4 - exp(w5*(G-1)) + 1, 1, 10)`.
pub fn initial_difficulty_fsrs7(grade: Rating, w: &[f64; 34]) -> f64 {
    let g = grade.as_i32() as f64;
    (w[4] - (w[5] * (g - 1.0)).exp() + 1.0).clamp(MIN_DIFFICULTY, MAX_DIFFICULTY)
}

/// Compute initial FSRS-7 state `(S_long, S_short, D)` after the first rating.
pub fn initial_state_fsrs7(grade: Rating) -> Fsrs7State {
    initial_state_fsrs7_with_weights(grade, &FSRS7_WEIGHTS)
}

/// Compute initial FSRS-7 state with custom 34-parameter weights.
pub fn initial_state_fsrs7_with_weights(grade: Rating, w: &[f64; 34]) -> Fsrs7State {
    let s_long = w[grade.as_index()].clamp(MIN_STABILITY, MAX_STABILITY);
    let s_short = (0.8 * s_long).clamp(MIN_STABILITY, MAX_STABILITY);
    let difficulty = initial_difficulty_fsrs7(grade, w);
    Fsrs7State {
        s_long,
        s_short,
        difficulty,
    }
}

/// FSRS-7 11-parameter two-component mixture forgetting curve.
/// Returns `(R_mix, R_short, R_long)`.
pub fn retrievability_fsrs7_components(
    state: Fsrs7State,
    elapsed_days: f64,
    w: &[f64; 34],
) -> (f64, f64, f64) {
    if elapsed_days <= 0.0 {
        return (1.0, 1.0, 1.0);
    }
    let sl = state.s_long.max(MIN_STABILITY);
    let ss = state.s_short.max(MIN_STABILITY);
    let d = state.difficulty.clamp(MIN_DIFFICULTY, MAX_DIFFICULTY);

    let d_short = (w[23] * ss.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let f_short = w[25].powf(-1.0 / d_short) - 1.0;
    let r_short = (1.0 + f_short * elapsed_days / ss)
        .powf(-d_short)
        .clamp(0.0, 1.0);

    let d_long = w[24].clamp(0.01, 0.95);
    let f_long = w[26].powf(-1.0 / d_long) - 1.0;
    let r_long = (1.0 + f_long * elapsed_days / sl * ((w[32] - 0.3) * (d - 5.0)).exp())
        .powf(-d_long)
        .clamp(0.0, 1.0);

    let w_short = w[27] * ss.powf(-w[29]);
    let w_long = w[28] * sl.powf(w[30]) * ((w[31] - 0.5) * (d - 5.0)).exp();
    let denom = (w_short + w_long).max(1e-12);
    let r_mix = ((w_short * r_short + w_long * r_long) / denom).clamp(0.0, 1.0);
    (r_mix, r_short, r_long)
}

/// FSRS-7 mixture probability of recall `R(t)` using default 34-parameter weights.
pub fn retrievability_fsrs7(state: Fsrs7State, elapsed_days: f64) -> f64 {
    retrievability_fsrs7_components(state, elapsed_days, &FSRS7_WEIGHTS).0
}

fn update_component(s: f64, d: f64, r_comp: f64, grade: Rating, w: &[f64; 34], a: usize) -> f64 {
    if grade == Rating::Again {
        let s_fail = w[a + 3] * ((s + 1.0).powf(w[a + 4]) - 1.0) * ((1.0 - r_comp) * w[a + 5]).exp();
        s.min(s_fail).clamp(MIN_STABILITY, MAX_STABILITY)
    } else {
        let hard = if grade == Rating::Hard { w[a + 6] } else { 1.0 };
        let easy = if grade == Rating::Easy { w[a + 7] } else { 1.0 };
        let s_inc = 1.0
            + (w[a] - 1.5).exp()
                * (11.0 - d)
                * s.powf(-w[a + 1])
                * (((1.0 - r_comp) * w[a + 2]).exp() - 1.0)
                * hard
                * easy;
        (s * s_inc).clamp(MIN_STABILITY, MAX_STABILITY)
    }
}

/// Advance FSRS-7 state `(S_long, S_short, D)` after a review at `elapsed_days`.
pub fn next_state_fsrs7(state: Fsrs7State, elapsed_days: f64, grade: Rating) -> Fsrs7State {
    next_state_fsrs7_with_weights(state, elapsed_days, grade, &FSRS7_WEIGHTS)
}

/// Advance FSRS-7 state with custom 34-parameter weights.
pub fn next_state_fsrs7_with_weights(
    state: Fsrs7State,
    elapsed_days: f64,
    grade: Rating,
    w: &[f64; 34],
) -> Fsrs7State {
    let sl = state.s_long.clamp(MIN_STABILITY, MAX_STABILITY);
    let ss = state.s_short.clamp(MIN_STABILITY, MAX_STABILITY);
    let d = state.difficulty.clamp(MIN_DIFFICULTY, MAX_DIFFICULTY);

    let (r_mix, r_short, _) = retrievability_fsrs7_components(state, elapsed_days.max(0.0), w);
    let s_long = update_component(sl, d, r_mix, grade, w, 7);
    let s_short = update_component(ss, d, r_short, grade, w, 15)
        .min(0.8 * s_long)
        .clamp(MIN_STABILITY, MAX_STABILITY);

    let g = grade.as_i32() as f64;
    let l_factor = if grade == Rating::Again { r_mix + 0.1 } else { 1.0 };
    let delta_d = -w[6] * (g - 3.0) * ((10.0 - d) / 9.0) * l_factor;
    let d_init_4 = initial_difficulty_fsrs7(Rating::Easy, w);
    let difficulty = (0.01 * d_init_4 + 0.99 * (d + delta_d)).clamp(MIN_DIFFICULTY, MAX_DIFFICULTY);

    Fsrs7State {
        s_long,
        s_short,
        difficulty,
    }
}

/// Compute next review interval (days) where `retrievability_fsrs7(state, t) == desired_retention`
/// via deterministic monotone bisection.
pub fn next_interval_fsrs7(state: Fsrs7State, desired_retention: f64) -> f64 {
    let target = desired_retention.clamp(0.01, 0.99);
    let mut lo = 0.0_f64;
    let mut hi = MAX_STABILITY;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if retrievability_fsrs7(state, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).clamp(0.0, MAX_STABILITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsrs7_initial_and_mixture_curve_properties() {
        let st = initial_state_fsrs7(Rating::Good);
        assert!((st.s_long - 3.9221).abs() < 1e-9);
        assert!((st.s_short - 0.8 * 3.9221).abs() < 1e-9);
        assert!((MIN_DIFFICULTY..=MAX_DIFFICULTY).contains(&st.difficulty));

        let r0 = retrievability_fsrs7(st, 0.0);
        let r1 = retrievability_fsrs7(st, 1.0);
        let r10 = retrievability_fsrs7(st, 10.0);
        let r100 = retrievability_fsrs7(st, 100.0);
        assert_eq!(r0, 1.0);
        assert!(r1 < r0 && r10 < r1 && r100 < r10 && r100 > 0.0);
    }

    #[test]
    fn fsrs7_recall_grows_lapse_shrinks_and_interval_inverts() {
        let s0 = initial_state_fsrs7(Rating::Good);
        let s1 = next_state_fsrs7(s0, 4.0, Rating::Good);
        assert!(s1.s_long > s0.s_long);
        assert!(s1.s_short <= 0.8 * s1.s_long + 1e-9);

        let s_lapse = next_state_fsrs7(s1, 10.0, Rating::Again);
        assert!(s_lapse.s_long <= s1.s_long);
        assert!(s_lapse.difficulty >= s1.difficulty);

        let ivl = next_interval_fsrs7(s1, 0.90);
        let r_at_ivl = retrievability_fsrs7(s1, ivl);
        assert!((r_at_ivl - 0.90).abs() < 1e-4, "got R={r_at_ivl} at t={ivl}");
    }
}
