//! Parameters for the suffix-automaton tail ballot.
//!
//! Locked before any benchmark run. Nothing in this file is a score weight.

/// Turns drawn from outside the current window into the tail pool.
///
/// The exact Kemeny solver stops at 8 candidates (`KEMENY_EXACT_CAP` in
/// `retrieval`). One component of that size, taken from below the window, is
/// the deeper pool: reordering the window alone cannot change the set at k.
/// The returned list stays the caller's length. Slot 0 is not a pool seat.
///
/// Event-sequence queries do not take these seats. That selector already
/// keeps one turn per session, and a longer echo of the question would stack
/// the same session back into the window.
pub const SUFFIX_BELOW_WINDOW: usize = 8;
