//! Wh-slot tail ballot. Locked before the card. Nothing here was fit to a score.
//!
//! Four ballots, each at most one vote: scene, structural degree, stability,
//! and the wh residual. Scene is a ballot and the tie-break (a zero margin
//! keeps the input order). Wh overturns scene only when degree or stability
//! is on wh's side.
//!
//! The wh weight is the scarcity of the slot in the pool, scaled so it cannot
//! exceed one ballot:
//!
//! `round(1000 * ln((n + 1) / (p + 1)) / ln(n + 1))`
//!
//! This is 1000 when no turn passes and 0 when every turn passes. A 2–1
//! structural majority is never overturned by wh: wh plus one structural vote
//! is at most 2000 milli-votes, and scene plus the other structural vote is
//! 2000. A tie keeps the scene order.
//!
//! Below-window entry, the same rule as the suffix-automaton v2 admission:
//! the gatekeeper orders the RSB top-k tail only. A turn from beyond that
//! top k is seated only when its four-ballot margin is strictly positive
//! against every currently seated tail turn. Otherwise the seated top-k
//! order stays as that within-window gatekeeper leaves it. Slot 0 is not a
//! pool seat.

/// Tail pool. Ranks 2..=50 are eligible to be compared. The returned list
/// stays the caller's k. Fixed before the card.
pub const GATEKEEPER_POOL: usize = 50;

/// Selectivity for the guarded swap, measured on this many candidates.
pub const WH_SELECT_WINDOW: usize = 30;

/// A pass rate above this carries no information, so the wh ballot is withheld.
pub const WH_UNINFORMATIVE: f64 = 0.65;

/// The guarded swap requires the top-30 pass rate to be at most this.
pub const WH_SWAP_PASS_MAX: f64 = 0.35;

/// Slot 0 stays when its score exceeds the next score by this factor.
pub const WH_LOCK_RATIO: f32 = 1.25;

/// A rank-2 or rank-3 candidate may swap into slot 0 only at or above this
/// fraction of slot 0's score.
pub const WH_SWAP_RATIO: f32 = 0.82;

/// One ballot, in milli-votes. Scene, degree, and stability use this weight.
/// The wh ballot uses [`wh_weight_milli`], which is at most this value.
pub const WH_MARGIN_SCALE: i64 = 1000;

/// Scarcity of a wh slot, in milli-votes, capped at one ballot.
///
/// `n` is the pool size. `passing` is how many of those turns carry the slot.
/// Parameter-free: 1000 when `passing` is 0, 0 when `passing` is `n`.
pub fn wh_weight_milli(n: usize, passing: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    let denom = (n as f64 + 1.0).ln();
    if denom == 0.0 {
        return 0;
    }
    let numer = ((n as f64 + 1.0) / (passing as f64 + 1.0)).ln();
    let raw = (WH_MARGIN_SCALE as f64 * numer / denom).round() as i64;
    raw.clamp(0, WH_MARGIN_SCALE)
}
