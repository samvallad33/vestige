//! Parameters for combinatorial group-testing rule triggers.
//!
//! Fixed before any benchmark run. These are the algorithm's own constants.
//! They are not fitted to a score.
//!
//! The seated family is one standing policy in a later session of the frozen
//! head's episode. Hwang's (1, n) search identifies it. An impure family
//! identifies nobody.
//!
//! Measured and not seated. Each of these moved a rank-5 gold out and left
//! the preference slice at 12/30.
//!
//! - One OR over every window turn's edges: 303 of 355 BEAM queries were impure.
//! - Backward `policy_backlinks`: BEAM turn Any@5 moved from 59.15 to 58.87.
//! - A later policy in the head's own session, `rsb_antecedent`, and an exact
//!   query token: LongMemEval turn Any@5 moved from 376/470 to 372/470 and
//!   LoCoMo turn R@5 moved from 69.47 to 69.01. LoCoMo sessions do not share a
//!   date, so those fires were not this predicate.
//!
//! The episode key is the stored session id (`_s` digits) or the stored date.
//! A one-digit column is not disjunct (Kautz and Singleton, IEEE Trans. IT 10,
//! 1964). The token family that used that bound is one of the families above
//! and is not seated.

/// Defective count for Hwang's generalized binary splitting.
///
/// One rule is seated, so the search is the (1, n) problem. For d = 1 the
/// procedure is binary splitting: α = floor(log2(n)) and the tested group has
/// size 2^α. F. K. Hwang, "A method for detecting all defective members in a
/// population by group testing", Journal of the American Statistical
/// Association 67 (1972). The same bound is Theorem 2.2.1 in Du and Hwang,
/// Combinatorial Group Testing and Its Applications (World Scientific, 1993).
pub const GBS_DEFECTIVE_COUNT: usize = 1;

/// Lowest seat index. Slot 0 is the RSB head and is not a seat.
pub const SEAT_FLOOR: usize = 1;
