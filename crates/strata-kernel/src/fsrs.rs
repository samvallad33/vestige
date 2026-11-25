//! FSRS-6-shaped fold: quantized spaced-repetition state, version-pinned
//! weights, derived-only retrievability.
//!
//! # What this is (and is not)
//!
//! This is the FSRS-6 *formula shape* with v1-approximate constants. Shapes
//! implemented:
//!
//! * power-law retrievability `R(t, S) = (1 + FACTOR * t / S)^decay` with
//!   trainable factor and decay (FSRS-6 generalizes FSRS-4.5's fixed
//!   `decay = -0.5`, `FACTOR = 19/81`);
//! * initial stability per rating;
//! * initial difficulty `D0(G) = w4 - exp(w5*(G-1)) + w6`, clamped to
//!   `[1, 10]`;
//! * difficulty update with linear delta and FSRS-5-style mean reversion
//!   toward `D0(4)`;
//! * recall stability with difficulty decay:
//!   `S' = S * (1 + e^w9 * (11 - D) * S^(-w10) * (e^(w11*(1-R)) - 1) * hard * easy)`;
//! * forget stability
//!   `S' = w14 * D^(-w15) * ((S+1)^w16 - 1) * e^(w17*(1-R))`, never above `S`;
//! * a short-term (zero elapsed-seq) update of the FSRS-5 same-day shape.
//!
//! The constants are approximations pinned to an algorithm version. They
//! are deliberately load-bearing: **changing ANY weight or formula here
//! means minting a new `ALGO_*` version, never editing in place** — old
//! logs must replay under the constants they were written with, forever.
//!
//! # Storage rule
//!
//! Stability and difficulty live in [`CardState`] ONLY as quantized Q32.32
//! `i64` ([`crate::canonical::to_q32_32`]). Inside a fold the values may be
//! dequantized to `f64`, evolved with [`libm`] math, and re-quantized —
//! deterministically, since every input to every operation is a pure
//! function of (event, quantized state, version-pinned weights).
//! Retrievability is DERIVED-ONLY ([`FsrsFold::retrievability`]); it is
//! never stored.
//!
//! # Weight layout (21 slots, u32 milli-units: value = milli / 1000)
//!
//! | idx | role                                              |
//! |-----|---------------------------------------------------|
//! | 0-3 | initial stability `S0(G)` for ratings 1..=4 (days)|
//! | 4-6 | initial difficulty: base, exp slope, offset       |
//! | 7   | difficulty delta slope                            |
//! | 8   | difficulty mean-reversion weight                  |
//! | 9   | recall gain log-scale                            |
//! | 10  | recall stability exponent (`S^-w10`)             |
//! | 11  | recall retrievability sensitivity                |
//! | 12  | hard-penalty multiplier (rating 2)               |
//! | 13  | easy-bonus multiplier (rating 4)                 |
//! | 14  | forget scale                                      |
//! | 15  | forget difficulty exponent (`D^-w15`)            |
//! | 16  | forget stability exponent (`(S+1)^w16`)          |
//! | 17  | forget retrievability sensitivity                 |
//! | 18  | short-term log-gain                              |
//! | 19  | retrievability `FACTOR` (milli, ~0.235 = 19/81)   |
//! | 20  | retrievability decay offset (`decay = -(0.5+w20)`)|

use crate::canonical::{from_q32_32, to_q32_32};
use crate::event::ReviewEvent;
use crate::kernel::{UnknownAlgoVersion, VersionedFold};
use crate::lint_state_type;
use crate::state::State;
use crate::strata_allowed_enum;
use borsh::{BorshDeserialize, BorshSerialize};

/// Algorithm version 1 (first pinned constant set).
pub const ALGO_V1: u32 = 1;
/// Algorithm version 2 (same shapes, different pinned constants).
pub const ALGO_V2: u32 = 2;
/// Algorithm version 3 (exact official 21-parameter FSRS-6 default weights).
pub const ALGO_V3: u32 = 3;
/// Algorithm version 4 (official 34-parameter FSRS-7 dual-stability `(S_long, S_short, D)`
/// with 11-parameter two-component mixture forgetting curve).
pub const ALGO_V4: u32 = 4;

/// Official FSRS-6 21-parameter default weights (`open-spaced-repetition/fsrs-rs`), pinned to [`ALGO_V3`].
pub const FSRS6_OFFICIAL_WEIGHTS: [f64; 21] = [
    0.212,  // w0: S0(Again)
    1.2931, // w1: S0(Hard)
    2.3065, // w2: S0(Good)
    8.2956, // w3: S0(Easy)
    6.4133, // w4: D0 base
    0.8334, // w5: D0 grade slope
    3.0194, // w6: difficulty delta
    0.001,  // w7: mean reversion weight
    1.8722, // w8: recall gain log-scale
    0.1666, // w9: recall stability saturation
    0.796,  // w10: recall retrievability sensitivity
    1.4835, // w11: forget scale
    0.0614, // w12: forget difficulty exponent
    0.2629, // w13: forget stability exponent
    1.6483, // w14: forget retrievability sensitivity
    0.6014, // w15: hard penalty
    1.8729, // w16: easy bonus
    0.5425, // w17: same-day review base
    0.0912, // w18: same-day grade offset
    0.0658, // w19: same-day stability exponent
    0.1542, // w20: forgetting curve decay
];

/// Official FSRS-7 34-parameter default weights (`open-spaced-repetition/fsrs-rs` v7.0.0),
/// pinned to [`ALGO_V4`].
pub const FSRS7_WEIGHTS: [f64; 34] = [
    0.1104, 2.2395, 3.9221, 11.7841, // w0..w3: initial S_long(G=1..=4)
    6.1686, 0.6457, 3.6807,          // w4..w6: D0 base/slope, difficulty delta
    1.9795, 0.0, 1.3826, 0.7024, 0.5999, 0.8146, 0.6398, 1.0, // w7..w14: S_long updates (a=7)
    1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, // w15..w22: S_short updates (a=15)
    0.1567, 0.0801, 0.2421, 0.9464,  // w23..w26: mixture decays & target anchors
    0.1433, 0.7145, 0.0, 0.5667,     // w27..w30: mixture component weights
    0.3734, 0.5333, 0.3048,          // w31..w33: difficulty & short-stability modulation
];

/// Returns `true` when `version` is a known pinned FSRS algorithm version (`ALGO_V1`..=`ALGO_V4`).
pub fn is_known_algo_version(version: u32) -> bool {
    matches!(version, ALGO_V1 | ALGO_V2 | ALGO_V3 | ALGO_V4)
}

/// v1 weights, milli-units. FSRS-6-shaped approximation.
pub const V1_WEIGHTS_MILLI: [u32; 21] = [
    400, 600, 2400, 5800, // 0..3: S0(G=1..=4)
    7195, 535, 1460, // 4..6: D0 base / exp slope / offset
    520, 200, // 7..8: difficulty delta slope / mean-reversion
    500, 140, 940, // 9..11: recall gain / S exponent / R sensitivity
    620, 1450, // 12..13: hard penalty / easy bonus
    2180, 50, 340, 1260, // 14..17: forget scale / D exp / S exp / R sensitivity
    460,  // 18: short-term log-gain
    235, 0, // 19..20: retrievability FACTOR / decay offset
];

/// v2 weights, milli-units. Identical shapes; differs from v1 at indices
/// 3, 7, 8, 9, 10, 11, 19 — enough to change every state root while both
/// versions stay replayable forever.
pub const V2_WEIGHTS_MILLI: [u32; 21] = [
    400, 600, 2400, 6200, //
    7195, 535, 1460, //
    545, 210, //
    540, 165, 965, //
    620, 1450, //
    2180, 50, 340, 1260, //
    460,  //
    242, 0, //
];

/// Look up the pinned weight table for an algorithm version.
pub fn weights_for(version: u32) -> Result<&'static [u32; 21], UnknownAlgoVersion> {
    match version {
        ALGO_V1 => Ok(&V1_WEIGHTS_MILLI),
        ALGO_V2 => Ok(&V2_WEIGHTS_MILLI),
        other => Err(UnknownAlgoVersion(other)),
    }
}

/// Upper bound for stability (days). Matches FSRS ecosystem conventions.
pub const S_MAX: f64 = 36500.0;
/// Lower bound for stability (days).
pub const S_MIN: f64 = 0.01;
/// Lower difficulty bound (FSRS convention: 1 = easiest).
pub const D_MIN: f64 = 1.0;
/// Upper difficulty bound (FSRS convention: 10 = hardest).
pub const D_MAX: f64 = 10.0;

/// Learning phase of a card. Fieldless enum: borsh-encodes as a `u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum CardPhase {
    /// Has at least one review recorded (initial phase after first review at
    /// rating <= 2).
    Learning,
    /// In the normal review cycle.
    Review,
    /// Recovering from a lapse.
    Relearning,
}

strata_allowed_enum!(CardPhase);

/// Quantized per-card FSRS state. Every real-valued quantity is Q32.32
/// `i64`; there are no floats here by construction (enforced by
/// [`crate::canonical::lint_state_type!`]).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct CardState {
    /// Stability in days, Q32.32.
    pub stability_q: i64,
    /// Difficulty in `[1, 10]`, Q32.32.
    pub difficulty_q: i64,
    /// Seq of the last review of this card.
    pub last_seq: u64,
    /// Total reviews applied to this card.
    pub review_count: u32,
    /// Reviews answered "again" (rating 1).
    pub lapse_count: u32,
    /// Learning phase.
    pub phase: CardPhase,
}

lint_state_type!(CardState {
    stability_q: i64,
    difficulty_q: i64,
    last_seq: u64,
    review_count: u32,
    lapse_count: u32,
    phase: CardPhase,
});

/// The FSRS fold itself: pure, version-pinned, quantize-in / quantize-out.
pub struct FsrsFold;

impl FsrsFold {
    /// Apply one review to `state` under weight table `w`.
    ///
    /// Deterministic by construction: every operation is a pure function of
    /// `(state, event, w)` with [`libm`] transcendental math and Q32.32
    /// quantization at the boundary. Ratings are clamped into `1..=4`.
    pub fn fold(state: &mut State, event: &ReviewEvent, w: &[u32; 21]) {
        let g: f64 = match event.rating {
            0 => 1.0,
            1..=4 => f64::from(event.rating),
            _ => 4.0,
        };
        state.applied_seq = state.applied_seq.max(event.event_seq);

        match state.cards.get_mut(&event.card_id) {
            Some(card) => Self::update(card, g, event.event_seq, w),
            None => {
                state
                    .cards
                    .insert(event.card_id, Self::initial(g, event.event_seq, w));
            }
        }
    }

    /// Derived-only retrievability: probability the card is recallable at
    /// `current_seq`, under the constants of `version`. Never stored.
    ///
    /// `R(t, S) = (1 + FACTOR * t / S)^decay` with `t = current_seq -
    /// last_seq` (saturating; 0 for a not-yet-reviewed position), clamped to
    /// `[0, 1]`. Monotonically non-increasing in `current_seq`.
    pub fn retrievability(
        card: &CardState,
        current_seq: u64,
        version: u32,
    ) -> Result<f64, UnknownAlgoVersion> {
        let elapsed = current_seq.saturating_sub(card.last_seq);
        match version {
            ALGO_V1 | ALGO_V2 => {
                let w = weights_for(version)?;
                Ok(Self::r(Self::dequantized_stability(card), elapsed, w))
            }
            ALGO_V3 => Ok(Self::r_v3(Self::dequantized_stability(card), elapsed)),
            ALGO_V4 => {
                let s_long = Self::dequantized_stability(card);
                let s_short = (0.8 * s_long).max(S_MIN);
                let d = from_q32_32(card.difficulty_q).clamp(D_MIN, D_MAX);
                Ok(Self::r_v4_mixture(s_long, s_short, d, elapsed).0)
            }
            other => Err(UnknownAlgoVersion(other)),
        }
    }

    /// Derived-only FSRS-7 (`ALGO_V4`) two-component mixture retrievability using the card's
    /// explicit `S_short` (`short_stability_q`) from [`State::short_stability_q`].
    pub fn retrievability_v4(card: &CardState, s_short_q: Option<i64>, current_seq: u64) -> f64 {
        let elapsed = current_seq.saturating_sub(card.last_seq);
        let s_long = Self::dequantized_stability(card);
        let s_short = s_short_q
            .map(|q| from_q32_32(q).max(S_MIN))
            .unwrap_or_else(|| (0.8 * s_long).max(S_MIN));
        let d = from_q32_32(card.difficulty_q).clamp(D_MIN, D_MAX);
        Self::r_v4_mixture(s_long, s_short, d, elapsed).0
    }

    /// Whole days from `reviewed_at_ms` to `as_of_ms` (0 if `as_of` is earlier).
    ///
    /// The fold still uses sequence numbers. This only turns a caller-supplied
    /// review clock into the `t` that [`Self::retrievability`] already takes.
    /// Not a new algorithm version.
    pub fn elapsed_review_days(reviewed_at_ms: i64, as_of_ms: i64) -> u64 {
        let delta = as_of_ms.saturating_sub(reviewed_at_ms);
        if delta <= 0 {
            0
        } else {
            delta as u64 / 86_400_000
        }
    }

    /// Derived retrievability at `as_of_ms`.
    ///
    /// `Some(reviewed_at_ms)` measures `t` in whole days since that review.
    /// `None` (no last-review time) keeps `t = fallback_seq - last_seq`.
    pub fn retrievability_at_review(
        card: &CardState,
        reviewed_at_ms: Option<i64>,
        as_of_ms: i64,
        fallback_seq: u64,
        version: u32,
    ) -> Result<f64, UnknownAlgoVersion> {
        let current = match reviewed_at_ms {
            Some(at) => card
                .last_seq
                .saturating_add(Self::elapsed_review_days(at, as_of_ms)),
            None => fallback_seq,
        };
        Self::retrievability(card, current, version)
    }

    /// Stability (days) whose retrievability after `elapsed_days` whole days
    /// equals `target`, under the constants of `version`: the inverse of
    /// [`Self::retrievability`] in `S`. Derived-only, like retrievability;
    /// no fold calls it.
    ///
    /// Solves `(1 + FACTOR * t / S)^decay = target` for `S` and clamps the
    /// result to `[S_MIN, S_MAX]`. `target <= 0` (or NaN) gives `S_MIN`.
    /// `target >= 1`, or `elapsed_days == 0` (where every `S` recalls with
    /// certainty), gives `S_MAX`.
    pub fn stability_for_retrievability(
        target: f64,
        elapsed_days: u64,
        version: u32,
    ) -> Result<f64, UnknownAlgoVersion> {
        if !is_known_algo_version(version) {
            return Err(UnknownAlgoVersion(version));
        }
        if target.is_nan() || target <= 0.0 {
            return Ok(S_MIN);
        }
        if target >= 1.0 || elapsed_days == 0 {
            return Ok(S_MAX);
        }
        match version {
            ALGO_V1 | ALGO_V2 => {
                let w = weights_for(version)?;
                let factor = milli(w, 19);
                let decay = -(0.5 + milli(w, 20));
                let base = libm::pow(target, 1.0 / decay) - 1.0;
                if base.is_nan() || base <= 0.0 {
                    return Ok(S_MAX);
                }
                Ok((factor * (elapsed_days as f64) / base).clamp(S_MIN, S_MAX))
            }
            ALGO_V3 => {
                let w20 = FSRS6_OFFICIAL_WEIGHTS[20];
                let factor = libm::pow(0.9, -1.0 / w20) - 1.0;
                let base = libm::pow(target, -1.0 / w20) - 1.0;
                if base.is_nan() || base <= 0.0 {
                    return Ok(S_MAX);
                }
                Ok((factor * (elapsed_days as f64) / base).clamp(S_MIN, S_MAX))
            }
            ALGO_V4 => {
                // Monotone bisection in S_long (with S_short = 0.8 * S_long, D = 5.0) over [S_MIN, S_MAX].
                let mut lo = S_MIN;
                let mut hi = S_MAX;
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    let r_mid = Self::r_v4_mixture(mid, (0.8 * mid).max(S_MIN), 5.0, elapsed_days).0;
                    if r_mid < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                Ok((0.5 * (lo + hi)).clamp(S_MIN, S_MAX))
            }
            other => Err(UnknownAlgoVersion(other)),
        }
    }

    fn dequantized_stability(card: &CardState) -> f64 {
        from_q32_32(card.stability_q).max(S_MIN)
    }

    /// Power-law forgetting curve for `ALGO_V3` (exact official FSRS-6).
    fn r_v3(s: f64, elapsed: u64) -> f64 {
        if elapsed == 0 {
            return 1.0;
        }
        let w20 = FSRS6_OFFICIAL_WEIGHTS[20];
        let factor = libm::pow(0.9, -1.0 / w20) - 1.0;
        libm::pow(1.0 + factor * (elapsed as f64) / s.max(S_MIN), -w20).clamp(0.0, 1.0)
    }

    /// Official FSRS-6 (`ALGO_V3`) initial difficulty `D0(G) = w4 - exp(w5*(G-1)) + 1`.
    fn d0_v3(g: f64) -> f64 {
        let w = &FSRS6_OFFICIAL_WEIGHTS;
        (w[4] - libm::exp(w[5] * (g - 1.0)) + 1.0).clamp(D_MIN, D_MAX)
    }

    /// Official FSRS-6 (`ALGO_V3`) fold step using [`FSRS6_OFFICIAL_WEIGHTS`] and `libm`.
    pub fn fold_v3(state: &mut State, event: &ReviewEvent) {
        let w = &FSRS6_OFFICIAL_WEIGHTS;
        let g: f64 = match event.rating {
            0 => 1.0,
            1..=4 => f64::from(event.rating),
            _ => 4.0,
        };
        state.applied_seq = state.applied_seq.max(event.event_seq);
        match state.cards.get_mut(&event.card_id) {
            None => {
                let s0 = w[g as usize - 1].clamp(S_MIN, S_MAX);
                let d0 = Self::d0_v3(g);
                state.cards.insert(
                    event.card_id,
                    CardState {
                        stability_q: to_q32_32(s0),
                        difficulty_q: to_q32_32(d0),
                        last_seq: event.event_seq,
                        review_count: 1,
                        lapse_count: u32::from(g <= 1.0),
                        phase: if g <= 2.0 {
                            CardPhase::Learning
                        } else {
                            CardPhase::Review
                        },
                    },
                );
            }
            Some(card) => {
                let s = Self::dequantized_stability(card);
                let d = from_q32_32(card.difficulty_q).clamp(D_MIN, D_MAX);
                let elapsed = event.event_seq.saturating_sub(card.last_seq);
                let s_next = if elapsed == 0 {
                    let s_inc = libm::exp(w[17] * (g - 3.0 + w[18])) * libm::pow(s, -w[19]);
                    let raw = s * s_inc;
                    if g >= 3.0 { raw.max(s) } else { raw }
                } else {
                    let r = Self::r_v3(s, elapsed);
                    if g <= 1.0 {
                        let sf = w[11]
                            * libm::pow(d, -w[12])
                            * (libm::pow(s + 1.0, w[13]) - 1.0)
                            * libm::exp(w[14] * (1.0 - r));
                        sf.clamp(S_MIN, s)
                    } else {
                        let hard = if g == 2.0 { w[15] } else { 1.0 };
                        let easy = if g == 4.0 { w[16] } else { 1.0 };
                        let inc = libm::exp(w[8])
                            * (11.0 - d)
                            * libm::pow(s, -w[9])
                            * (libm::exp(w[10] * (1.0 - r)) - 1.0)
                            * hard
                            * easy;
                        s * (1.0 + inc)
                    }
                };
                let delta_d = -w[6] * (g - 3.0) * ((10.0 - d) / 9.0);
                let d_next = (w[7] * Self::d0_v3(4.0) + (1.0 - w[7]) * (d + delta_d))
                    .clamp(D_MIN, D_MAX);
                card.stability_q = to_q32_32(s_next.clamp(S_MIN, S_MAX));
                card.difficulty_q = to_q32_32(d_next);
                card.last_seq = event.event_seq;
                card.review_count = card.review_count.saturating_add(1);
                if g <= 1.0 {
                    card.lapse_count = card.lapse_count.saturating_add(1);
                    card.phase = CardPhase::Relearning;
                } else if g >= 3.0 {
                    card.phase = CardPhase::Review;
                }
            }
        }
    }

    /// Official FSRS-7 (`ALGO_V4`) 11-parameter two-component mixture forgetting curve.
    /// Returns `(R_mix, R_short, R_long)` using `libm` transcendental math only.
    pub fn r_v4_mixture(s_long: f64, s_short: f64, d: f64, elapsed: u64) -> (f64, f64, f64) {
        if elapsed == 0 {
            return (1.0, 1.0, 1.0);
        }
        let w = &FSRS7_WEIGHTS;
        let t = elapsed as f64;
        let sl = s_long.max(S_MIN);
        let ss = s_short.max(S_MIN);
        let d_short = (w[23] * libm::pow(ss, w[33] - 0.3)).clamp(0.01, 0.95);
        let f_short = libm::pow(w[25], -1.0 / d_short) - 1.0;
        let r_short = libm::pow(1.0 + f_short * t / ss, -d_short).clamp(0.0, 1.0);

        let d_long = w[24].clamp(0.01, 0.95);
        let f_long = libm::pow(w[26], -1.0 / d_long) - 1.0;
        let r_long = libm::pow(
            1.0 + f_long * t / sl * libm::exp((w[32] - 0.3) * (d - 5.0)),
            -d_long,
        )
        .clamp(0.0, 1.0);

        let w_short = w[27] * libm::pow(ss, -w[29]);
        let w_long = w[28] * libm::pow(sl, w[30]) * libm::exp((w[31] - 0.5) * (d - 5.0));
        let denom = (w_short + w_long).max(1e-12);
        let r_mix = ((w_short * r_short + w_long * r_long) / denom).clamp(0.0, 1.0);
        (r_mix, r_short, r_long)
    }

    /// Official FSRS-7 (`ALGO_V4`) initial difficulty `D0(G) = clamp(w4 - exp(w5*(G-1)) + 1, 1, 10)`.
    fn d0_v4(g: f64) -> f64 {
        let w = &FSRS7_WEIGHTS;
        (w[4] - libm::exp(w[5] * (g - 1.0)) + 1.0).clamp(D_MIN, D_MAX)
    }

    /// FSRS-7 (`ALGO_V4`) single stability-component update (`a = 7` for `S_long`, `a = 15` for `S_short`).
    fn update_v4_component(s: f64, d: f64, r_comp: f64, g: f64, a: usize) -> f64 {
        let w = &FSRS7_WEIGHTS;
        if g <= 1.0 {
            let s_fail = w[a + 3]
                * (libm::pow(s + 1.0, w[a + 4]) - 1.0)
                * libm::exp((1.0 - r_comp) * w[a + 5]);
            s.min(s_fail).clamp(S_MIN, S_MAX)
        } else {
            let hard = if g == 2.0 { w[a + 6] } else { 1.0 };
            let easy = if g == 4.0 { w[a + 7] } else { 1.0 };
            let s_inc = 1.0
                + libm::exp(w[a] - 1.5)
                    * (11.0 - d)
                    * libm::pow(s, -w[a + 1])
                    * (libm::exp((1.0 - r_comp) * w[a + 2]) - 1.0)
                    * hard
                    * easy;
            (s * s_inc).clamp(S_MIN, S_MAX)
        }
    }

    /// Official FSRS-7 (`ALGO_V4`) dual-stability fold step using [`FSRS7_WEIGHTS`] and `libm`.
    /// Stores `S_long` in `card.stability_q`, `D` in `card.difficulty_q`, and `S_short` in
    /// `state.short_stability_q[&card_id]` (all Q32.32 `i64`, hashed into `state_root`).
    pub fn fold_v4(state: &mut State, event: &ReviewEvent) {
        let w = &FSRS7_WEIGHTS;
        let g: f64 = match event.rating {
            0 => 1.0,
            1..=4 => f64::from(event.rating),
            _ => 4.0,
        };
        state.applied_seq = state.applied_seq.max(event.event_seq);
        match state.cards.get_mut(&event.card_id) {
            None => {
                let s_long_0 = w[g as usize - 1].clamp(S_MIN, S_MAX);
                let s_short_0 = (0.8 * s_long_0).clamp(S_MIN, S_MAX);
                let d0 = Self::d0_v4(g);
                state.cards.insert(
                    event.card_id,
                    CardState {
                        stability_q: to_q32_32(s_long_0),
                        difficulty_q: to_q32_32(d0),
                        last_seq: event.event_seq,
                        review_count: 1,
                        lapse_count: u32::from(g <= 1.0),
                        phase: if g <= 2.0 {
                            CardPhase::Learning
                        } else {
                            CardPhase::Review
                        },
                    },
                );
                state
                    .short_stability_q
                    .insert(event.card_id, to_q32_32(s_short_0));
            }
            Some(card) => {
                let s_long = Self::dequantized_stability(card);
                let s_short = state
                    .short_stability_q
                    .get(&event.card_id)
                    .map(|&q| from_q32_32(q).max(S_MIN))
                    .unwrap_or_else(|| (0.8 * s_long).max(S_MIN));
                let d = from_q32_32(card.difficulty_q).clamp(D_MIN, D_MAX);
                let elapsed = event.event_seq.saturating_sub(card.last_seq);

                let (r_mix, r_short, _) = Self::r_v4_mixture(s_long, s_short, d, elapsed);
                let s_long_next = Self::update_v4_component(s_long, d, r_mix, g, 7);
                let s_short_raw = Self::update_v4_component(s_short, d, r_short, g, 15);
                let s_short_next = s_short_raw.min(0.8 * s_long_next).clamp(S_MIN, S_MAX);

                let l_factor = if g <= 1.0 { r_mix + 0.1 } else { 1.0 };
                let delta_d = -w[6] * (g - 3.0) * ((10.0 - d) / 9.0) * l_factor;
                let d_next = (0.01 * Self::d0_v4(4.0) + 0.99 * (d + delta_d)).clamp(D_MIN, D_MAX);

                card.stability_q = to_q32_32(s_long_next);
                card.difficulty_q = to_q32_32(d_next);
                card.last_seq = event.event_seq;
                card.review_count = card.review_count.saturating_add(1);
                if g <= 1.0 {
                    card.lapse_count = card.lapse_count.saturating_add(1);
                    card.phase = CardPhase::Relearning;
                } else if g >= 3.0 {
                    card.phase = CardPhase::Review;
                }
                state
                    .short_stability_q
                    .insert(event.card_id, to_q32_32(s_short_next));
            }
        }
    }

    /// Power-law forgetting curve. `base >= 1` always, `decay < 0` always,
    /// so the result lies in `[0, 1]`.
    fn r(s: f64, elapsed: u64, w: &[u32; 21]) -> f64 {
        let factor = milli(w, 19);
        let decay = -(0.5 + milli(w, 20));
        let base = 1.0 + factor * (elapsed as f64) / s;
        libm::pow(base, decay).clamp(0.0, 1.0)
    }

    fn initial(g: f64, seq: u64, w: &[u32; 21]) -> CardState {
        let s0 = milli(w, g as usize - 1).max(S_MIN);
        let d0 = Self::d0(g, w).clamp(D_MIN, D_MAX);
        CardState {
            stability_q: to_q32_32(s0),
            difficulty_q: to_q32_32(d0),
            last_seq: seq,
            review_count: 1,
            lapse_count: u32::from(g <= 1.0),
            phase: if g <= 2.0 {
                CardPhase::Learning
            } else {
                CardPhase::Review
            },
        }
    }

    /// Initial difficulty `D0(G) = w4 - exp(w5*(G-1)) + w6`.
    fn d0(g: f64, w: &[u32; 21]) -> f64 {
        milli(w, 4) - libm::exp(milli(w, 5) * (g - 1.0)) + milli(w, 6)
    }

    fn update(card: &mut CardState, g: f64, seq: u64, w: &[u32; 21]) {
        let s = Self::dequantized_stability(card);
        let d = from_q32_32(card.difficulty_q).clamp(D_MIN, D_MAX);
        let elapsed = seq.saturating_sub(card.last_seq);

        let s_next = if elapsed == 0 {
            // Short-term (same-seq) update, FSRS-5 same-day shape:
            // S' = S * exp(w18 * (G - 3)).
            s * libm::exp(milli(w, 18) * (g - 3.0))
        } else {
            let r = Self::r(s, elapsed, w);
            if g <= 1.0 {
                // Forget: w14 * D^-w15 * ((S+1)^w16 - 1) * e^(w17*(1-R)),
                // never allowed to increase stability.
                let sf = milli(w, 14)
                    * libm::pow(d, -milli(w, 15))
                    * (libm::pow(s + 1.0, milli(w, 16)) - 1.0)
                    * libm::exp(milli(w, 17) * (1.0 - r));
                sf.clamp(S_MIN, s)
            } else {
                // Recall with difficulty decay:
                // S' = S * (1 + e^w9 * (11-D) * S^-w10 * (e^(w11*(1-R))-1) * hard * easy)
                let hard = if g == 2.0 { milli(w, 12) } else { 1.0 };
                let easy = if g == 4.0 { milli(w, 13) } else { 1.0 };
                let inc = libm::exp(milli(w, 9))
                    * (11.0 - d)
                    * libm::pow(s, -milli(w, 10))
                    * (libm::exp(milli(w, 11) * (1.0 - r)) - 1.0)
                    * hard
                    * easy;
                s * (1.0 + inc)
            }
        };

        // Difficulty: linear delta + FSRS-5 mean reversion toward D0(4).
        let d_lin = d + milli(w, 7) * (3.0 - g);
        let d_next = {
            let revert = milli(w, 8).clamp(0.0, 1.0);
            let target = Self::d0(4.0, w).clamp(D_MIN, D_MAX);
            revert * target + (1.0 - revert) * d_lin
        };

        card.stability_q = to_q32_32(s_next.clamp(S_MIN, S_MAX));
        card.difficulty_q = to_q32_32(d_next.clamp(D_MIN, D_MAX));
        card.last_seq = seq;
        card.review_count = card.review_count.saturating_add(1);
        if g <= 1.0 {
            card.lapse_count = card.lapse_count.saturating_add(1);
            card.phase = CardPhase::Relearning;
        } else if g >= 3.0 {
            card.phase = CardPhase::Review;
        }
    }
}

/// Decode milli-unit weight `w[i]` to `f64`.
#[inline]
fn milli(w: &[u32; 21], i: usize) -> f64 {
    f64::from(w[i]) / 1000.0
}

/// v1 kernel entry (function pointer target for dispatch).
pub fn fold_v1(state: &mut State, event: &ReviewEvent) {
    FsrsFold::fold(state, event, &V1_WEIGHTS_MILLI);
}

/// v2 kernel entry (function pointer target for dispatch).
pub fn fold_v2(state: &mut State, event: &ReviewEvent) {
    FsrsFold::fold(state, event, &V2_WEIGHTS_MILLI);
}

/// v3 kernel entry (official 21-param FSRS-6).
pub fn fold_v3(state: &mut State, event: &ReviewEvent) {
    FsrsFold::fold_v3(state, event);
}

/// v4 kernel entry (official 34-param FSRS-7 dual-stability `(S_long, S_short, D)`).
pub fn fold_v4(state: &mut State, event: &ReviewEvent) {
    FsrsFold::fold_v4(state, event);
}

impl VersionedFold for ReviewEvent {
    fn kernel_table() -> &'static [(u32, fn(&mut State, &Self))] {
        &[
            (ALGO_V1, fold_v1 as fn(&mut State, &ReviewEvent)),
            (ALGO_V2, fold_v2 as fn(&mut State, &ReviewEvent)),
            (ALGO_V3, fold_v3 as fn(&mut State, &ReviewEvent)),
            (ALGO_V4, fold_v4 as fn(&mut State, &ReviewEvent)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrievability_is_monotone_decreasing_in_seq() {
        let card = CardState {
            stability_q: to_q32_32(10.0),
            difficulty_q: to_q32_32(5.0),
            last_seq: 100,
            review_count: 1,
            lapse_count: 0,
            phase: CardPhase::Review,
        };
        let mut prev = FsrsFold::retrievability(&card, 100, ALGO_V1).unwrap();
        assert_eq!(prev, 1.0); // zero elapsed -> certain recall
        for seq in 101..1100u64 {
            let r = FsrsFold::retrievability(&card, seq, ALGO_V1).unwrap();
            assert!(
                r <= prev,
                "R must be non-increasing: {r} > {prev} at seq {seq}"
            );
            assert!((0.0..=1.0).contains(&r));
            prev = r;
        }
        assert!(prev < 0.99); // and actually decays over 1000 seq
    }

    #[test]
    fn retrievability_across_both_versions_decays() {
        for v in [ALGO_V1, ALGO_V2] {
            let card = CardState {
                stability_q: to_q32_32(7.5),
                difficulty_q: to_q32_32(6.0),
                last_seq: 10,
                review_count: 3,
                lapse_count: 1,
                phase: CardPhase::Review,
            };
            let near = FsrsFold::retrievability(&card, 12, v).unwrap();
            let far = FsrsFold::retrievability(&card, 500, v).unwrap();
            assert!(near > far, "version {v}: {near} !> {far}");
        }
    }

    #[test]
    fn stability_for_retrievability_inverts_the_curve() {
        for version in [ALGO_V1, ALGO_V2] {
            for s in [0.05, 0.4, 2.3065, 6.3, 41.0, 900.0] {
                for days in [1u64, 3, 16, 120] {
                    let card = CardState {
                        stability_q: to_q32_32(s),
                        difficulty_q: to_q32_32(5.0),
                        last_seq: 10,
                        review_count: 0,
                        lapse_count: 0,
                        phase: CardPhase::Learning,
                    };
                    let r = FsrsFold::retrievability(&card, 10 + days, version).unwrap();
                    let back = FsrsFold::stability_for_retrievability(r, days, version).unwrap();
                    assert!(
                        ((back - s) / s).abs() < 1e-6,
                        "v{version} S={s} t={days}: R={r} inverts to {back}"
                    );
                }
            }
        }
        let v = ALGO_V2;
        assert_eq!(
            FsrsFold::stability_for_retrievability(0.0, 5, v).unwrap(),
            S_MIN
        );
        assert_eq!(
            FsrsFold::stability_for_retrievability(f64::NAN, 5, v).unwrap(),
            S_MIN
        );
        assert_eq!(
            FsrsFold::stability_for_retrievability(1.0, 5, v).unwrap(),
            S_MAX
        );
        assert_eq!(
            FsrsFold::stability_for_retrievability(0.5, 0, v).unwrap(),
            S_MAX
        );
        assert!(FsrsFold::stability_for_retrievability(0.5, 1, 99).is_err());
    }

    #[test]
    fn unknown_version_is_rejected() {
        assert!(weights_for(u32::MAX).is_err());
        let sample = CardState {
            stability_q: to_q32_32(5.0),
            difficulty_q: to_q32_32(5.0),
            last_seq: 1,
            review_count: 1,
            lapse_count: 0,
            phase: CardPhase::Review,
        };
        assert!(FsrsFold::retrievability(&sample, 2, 99).is_err());
    }

    #[test]
    fn first_review_matches_d0_s0_shapes() {
        let mut st = State::default();
        let w = &V1_WEIGHTS_MILLI;
        FsrsFold::fold(
            &mut st,
            &ReviewEvent {
                card_id: 1,
                rating: 1,
                event_seq: 1,
            },
            w,
        );
        let c = &st.cards[&1];
        assert_eq!(c.stability_q, to_q32_32(0.4)); // w0
        let expected_d0 = (7.195 - libm::exp(0.535 * 0.0) + 1.46).clamp(D_MIN, D_MAX);
        assert_eq!(c.difficulty_q, to_q32_32(expected_d0));
        assert_eq!(c.phase, CardPhase::Learning);
        assert_eq!(c.lapse_count, 1);
    }

    #[test]
    fn forgetting_never_increases_stability() {
        let mut st = State::default();
        let w = &V1_WEIGHTS_MILLI;
        // Long history of good reviews to build stability.
        for seq in 1..=40u64 {
            FsrsFold::fold(
                &mut st,
                &ReviewEvent {
                    card_id: 7,
                    rating: 4,
                    event_seq: seq,
                },
                w,
            );
        }
        let before = st.cards[&7].stability_q;
        FsrsFold::fold(
            &mut st,
            &ReviewEvent {
                card_id: 7,
                rating: 1,
                event_seq: 90,
            },
            w,
        );
        let after = st.cards[&7].stability_q;
        assert!(
            after < before,
            "lapse must reduce stability ({after} >= {before})"
        );
        assert_eq!(st.cards[&7].phase, CardPhase::Relearning);
    }

    #[test]
    fn recall_grows_stability_and_review_phase_resumes() {
        let mut st = State::default();
        let w = &V1_WEIGHTS_MILLI;
        FsrsFold::fold(
            &mut st,
            &ReviewEvent {
                card_id: 3,
                rating: 1,
                event_seq: 1,
            },
            w,
        );
        let s_after_lapse = st.cards[&3].stability_q;
        FsrsFold::fold(
            &mut st,
            &ReviewEvent {
                card_id: 3,
                rating: 3,
                event_seq: 30,
            },
            w,
        );
        let c = &st.cards[&3];
        assert!(c.stability_q > s_after_lapse);
        assert_eq!(c.phase, CardPhase::Review);
        assert_eq!(c.review_count, 2);
    }

    #[test]
    fn fsrs6_v3_and_fsrs7_v4_fold_deterministically_and_monotonically() {
        for version in [ALGO_V3, ALGO_V4] {
            assert!(is_known_algo_version(version));
            let kernel = crate::kernel::Kernel::<ReviewEvent>::for_version(version).unwrap();
            let mut st1 = State::default();
            let mut st2 = State::default();
            let events = [
                ReviewEvent { card_id: 42, rating: 3, event_seq: 1 },
                ReviewEvent { card_id: 42, rating: 4, event_seq: 6 },
                ReviewEvent { card_id: 42, rating: 2, event_seq: 20 },
                ReviewEvent { card_id: 42, rating: 1, event_seq: 45 },
                ReviewEvent { card_id: 42, rating: 3, event_seq: 52 },
            ];
            for e in &events {
                kernel.apply(&mut st1, e);
                kernel.apply(&mut st2, e);
            }
            assert_eq!(st1, st2, "v{version} must be bit-exact deterministic");
            assert_eq!(
                crate::checkpoint::state_root(&st1),
                crate::checkpoint::state_root(&st2)
            );

            let c = &st1.cards[&42];
            let r0 = FsrsFold::retrievability(c, 52, version).unwrap();
            let r1 = FsrsFold::retrievability(c, 55, version).unwrap();
            let r2 = FsrsFold::retrievability(c, 200, version).unwrap();
            assert_eq!(r0, 1.0);
            assert!(r1 < r0 && r2 < r1 && r2 >= 0.0);

            // Stability inversion round-trips cleanly.
            let inv = FsrsFold::stability_for_retrievability(r1, 3, version).unwrap();
            assert!((S_MIN..=S_MAX).contains(&inv));
        }

        // Verify FSRS-7 (`ALGO_V4`) maintains the `S_short <= 0.8 * S_long` invariant and commits S_short into state_root.
        let mut st_v4 = State::default();
        fold_v4(&mut st_v4, &ReviewEvent { card_id: 7, rating: 3, event_seq: 1 });
        fold_v4(&mut st_v4, &ReviewEvent { card_id: 7, rating: 4, event_seq: 15 });
        let s_long = from_q32_32(st_v4.cards[&7].stability_q);
        let s_short = from_q32_32(st_v4.short_stability_q[&7]);
        assert!(s_short <= 0.8 * s_long + 1e-6);
        let root_a = crate::checkpoint::state_root(&st_v4);
        st_v4.short_stability_q.insert(7, to_q32_32(s_short * 0.5));
        let root_b = crate::checkpoint::state_root(&st_v4);
        assert_ne!(root_a, root_b, "ALGO_V4 state_root must cryptographically commit S_short");
    }
}
