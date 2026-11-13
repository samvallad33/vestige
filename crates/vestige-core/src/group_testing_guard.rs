//! Parameters for combinatorial group-testing rule triggers.
//!
//! Fixed before any benchmark run. These are the algorithm's own constants.
//! They are not fitted to a score.

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
