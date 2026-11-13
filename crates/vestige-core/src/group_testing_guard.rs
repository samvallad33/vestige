//! Parameters for combinatorial group-testing rule triggers.
//!
//! Fixed before any benchmark run. These are the algorithm's own constants.
//! They are not fitted to a score.
//!
//! v2 population, after the v1 card. v1 tested one OR over every stored edge
//! from every turn in the window. 303 of 355 BEAM queries were then impure:
//! each episode in the window contributed its own standing rule, so the
//! complement of a (1, n) find stayed positive and nothing was seated.
//! A family is now its own group. The edge family is the rule and preference
//! turns outside the window that the frozen head points at, plus a standing
//! policy in the head's episode. The token family is the rule and preference
//! turns outside the window that share one exact identical token with the
//! query. Hwang's (1, n) search has to identify a single defective inside a
//! family. An impure family identifies nobody and does not veto a family that
//! did. Two pure families that name different turns are a (2, n) population,
//! and nothing is seated.
//!
//! The episode member is a standing policy that comes later than the head.
//! `policy_backlinks` point from a later turn at an earlier policy. Seating
//! that earlier policy fired 86 times on the v2 card and moved BEAM turn
//! Any@5 from 59.15 to 58.87. The probed instruction sits in a later sub than
//! the retrieved head. The episode key is the stored session id (`_s` digits)
//! or the stored date. A later policy is one Boolean with the other edge
//! kinds, not a second vote.

/// Defective count for Hwang's generalized binary splitting.
///
/// One rule is seated, so the search is the (1, n) problem. For d = 1 the
/// procedure is binary splitting: α = floor(log2(n)) and the tested group has
/// size 2^α. F. K. Hwang, "A method for detecting all defective members in a
/// population by group testing", Journal of the American Statistical
/// Association 67 (1972). The same bound is Theorem 2.2.1 in Du and Hwang,
/// Combinatorial Group Testing and Its Applications (World Scientific, 1993).
pub const GBS_DEFECTIVE_COUNT: usize = 1;

/// Shortest digit run treated as an exact identity token.
///
/// A one-digit column is contained in a large fraction of turns, so it is not
/// a disjunct column: the union of other columns covers it. Kautz and
/// Singleton, "Nonrandom binary superimposed codes", IEEE Transactions on
/// Information Theory 10 (1964). Two digits is the shortest run that can be
/// its own column.
pub const EXACT_DIGIT_MIN_LEN: usize = 2;

/// Lowest seat index. Slot 0 is the RSB head and is not a seat.
pub const SEAT_FLOOR: usize = 1;
