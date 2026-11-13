//! Wh-slot tail ballot. Locked before the WhV6 card. Nothing here was fit to a score.
//!
//! Four ballots, each at most one vote: scene, structural degree, stability,
//! and the wh residual. Scene is a ballot and the tie-break (a zero margin
//! keeps the input order). Wh is one vote, +1, -1, or 0. On a query with no
//! wh class that vote is 0 and the other three ballots still run.
//!
//! The wh weight is the scarcity of the slot inside the seated window, not
//! the deeper pool:
//!
//! `round(1000 * ln((n + 1) / (p + 1)) / ln(n + 1))`
//!
//! clamped to 0..=1000. This is 1000 when no seated turn passes and 0 when
//! every seated turn passes. Two of the other ballots beat wh. A 2-1 of those
//! ballots is not overturned when wh joins the dissenter.
//!
//! `rsb_seat_behind` runs on the first k turns with the caller's k, the same
//! call PR 715 makes. A full window does not grow. Only after that seat does
//! the ranker append positive-score turns, deduped, up to `GATEKEEPER_POOL`.
//! Those deeper turns never pass through `rsb_seat_behind`. A query with no
//! wh class does not admit them, so its list is the 715 window.
//!
//! A below-window turn is seated only when it passes the novel residual and
//! its four-ballot margin is strictly positive against every seated tail
//! turn. The returned tail keeps the seated length. Slot 0 is the RSB head
//! and stays there. There is no second pass and no swap.
//!
//! The class order is how long, then how many or how much, then when, then
//! who, whom, or whose, then where. Anything else is no class. A residual is
//! a character form read after the query's own tokens are blanked. When is an
//! ISO date, a slash date, or HH:MM. How long is a digit or that clock. How
//! many is a digit. Who and where take a novel TitleCase name or an @handle,
//! URL, or path. A TitleCase name is mid-sentence, length at least
//! `WH_MIN_NAME_LEN`, first letter upper, the rest lower. All-caps tokens are
//! rejected. The session timestamp is not a clock.
//!
//! Provenance, fixed before the run. None of these constants was fit to
//! LoCoMo, LME, BEAM, or MAB.
//!
//! - Scarcity weight: Sparck Jones, "A statistical interpretation of term
//!   specificity and its application in retrieval," Journal of Documentation
//!   28(1), 1972. Normalized by `ln(n+1)` so the weight is one ballot.
//!   Manning, Raghavan, and Schütze, Introduction to Information Retrieval,
//!   Cambridge, 2008, chapter 6.
//! - Order: Kemeny, "Mathematics without numbers," Daedalus 88(4), 1959, and
//!   Young, "Condorcet's Theory of Voting," American Political Science Review
//!   82(4), 1988. Exact branch-and-bound through 8!. Equal scores keep the
//!   RSB v4 order.
//! - Below-window entry: Condorcet pairwise majority, the suffix-automaton v2
//!   admission rule, plus the novel residual.
//! - `GATEKEEPER_POOL` 50, `WH_UNINFORMATIVE` 0.65, `WH_MARGIN_SCALE` 1000,
//!   `WH_MIN_NAME_LEN` 3: the WhV6 lock. Not estimated on a test set.
//! - RSB uses the caller's k, as in PR 715 (`5abd62a0`).

/// Deeper pool appended after RSB seats into k. The returned list stays k.
pub const GATEKEEPER_POOL: usize = 50;

/// A pass rate above this carries no information, so the wh ballot is withheld.
pub const WH_UNINFORMATIVE: f64 = 0.65;

/// One ballot, in milli-votes. Scene, degree, and stability use this weight.
/// The wh ballot uses [`wh_weight_milli`], which is at most this value.
pub const WH_MARGIN_SCALE: i64 = 1000;

/// Minimum length of a novel TitleCase name. All-caps tokens fail the
/// "rest lower" test on their own. Fixed before the card.
pub const WH_MIN_NAME_LEN: usize = 3;

/// Scarcity of a wh slot, in milli-votes, capped at one ballot.
///
/// `n` is the seated window. `passing` is how many of those turns carry the
/// novel residual. Parameter-free: 1000 when `passing` is 0, 0 when `passing`
/// is `n`.
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
