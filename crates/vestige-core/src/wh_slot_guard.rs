//! Wh-slot tail ballot. Locked before the card. Nothing here was fit to a score.
//!
//! Four ballots, each at most one vote: scene, structural degree, stability,
//! and the wh residual. Scene is a ballot and the tie-break (a zero margin
//! keeps the input order). Wh is one vote, +1, -1, or 0, including on a query
//! with no wh class, where that vote is 0 and the other three ballots still run.
//!
//! The wh weight is the scarcity of the slot in the pool, scaled so it cannot
//! exceed one ballot:
//!
//! `round(1000 * ln((n + 1) / (p + 1)) / ln(n + 1))`
//!
//! clamped to 0..=1000. This is 1000 when no turn passes and 0 when every turn
//! passes. Two of the other ballots beat wh: 2000 milli-votes against at most
//! 1000. A 2-1 majority of those ballots is not overturned when wh joins the
//! single dissenting ballot, because that side is then at most 2000 as well.
//!
//! The first k turns are built and then passed to `rsb_seat_behind` with that
//! same k, which is what PR 715 does. A full window does not grow, so a cause
//! cannot push a seated turn past k. Only after that seat does the ranker
//! append positive-score turns, deduped, up to `GATEKEEPER_POOL`. Those
//! deeper turns never pass through `rsb_seat_behind`. A query with no wh
//! class does not admit them, so its list is the 715 window.
//!
//! Below-window entry, the same rule as the suffix-automaton v2 admission:
//! the gatekeeper orders the RSB top-k tail only. A turn from beyond that
//! top k is seated only when its four-ballot margin is strictly positive
//! against every currently seated tail turn. Otherwise the seated top-k
//! order is the within-window order. Slot 0 is not a pool seat. The
//! margin-guarded swap is the only move of that head.
//!
//! The query class is the closed interrogative set: when, who, whom, whose,
//! where, how many, how long. A residual is a character form with the query's
//! own tokens blanked first, so a turn that only repeats the query does not
//! pass. When is an ISO date, an HH:MM, or an ASCII digit. How long is a digit
//! or a clock. How many is a digit. Who is a mid-sentence capital. Where is an
//! @handle, a URL, or a path. Why, what-year, what-date, and what-city stay
//! outside this set. A session timestamp is not a clock.
//!
//! Provenance, fixed before the WhV6 run. None of these constants was fit to
//! LoCoMo, LME, BEAM, or MAB.
//!
//! - Scarcity weight: Sparck Jones, "A statistical interpretation of term
//!   specificity and its application in retrieval," Journal of Documentation
//!   28(1), 1972. The form here is `ln((n+1)/(p+1)) / ln(n+1)`, rounded to
//!   milli-votes and clamped to one ballot so a single feature cannot outvote
//!   two others. Manning, Raghavan, and Schütze, Introduction to Information
//!   Retrieval, Cambridge, 2008, chapter 6, is the linear-combination setting
//!   that clamp protects.
//! - Order: Kemeny, "Mathematics without numbers," Daedalus 88(4), 1959, and
//!   Young, "Condorcet's Theory of Voting," American Political Science Review
//!   82(4), 1988. Exact branch-and-bound through 8!; larger components use
//!   Copeland. Equal scores keep the RSB v4 order.
//! - Below-window entry: a candidate needs a strict pairwise majority against
//!   every seated tail turn (Condorcet). Same admission rule as the
//!   suffix-automaton v2.
//! - `GATEKEEPER_POOL` 50, `WH_SELECT_WINDOW` 30, `WH_UNINFORMATIVE` 0.65,
//!   `WH_SWAP_PASS_MAX` 0.35, `WH_LOCK_RATIO` 1.25, `WH_SWAP_RATIO` 0.82:
//!   the Wh v3 lock, written down before any card on this branch. Not
//!   estimated on a test set.
//! - RSB uses the caller's k, as in PR 715 (`5abd62a0`). The deeper pool is
//!   appended after that call.

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
