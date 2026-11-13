//! Structural factor-graph guard.
//!
//! Every quantity the loopy sum-product vote uses is fixed here, before any
//! benchmark card. Nothing in this file is a rank weight. The Kemeny ballot
//! that consumes the marginals is still +1, −1, or 0.
//!
//! Scene, structural degree, stability, and the wh slot are one ±1 each on
//! every pair, including a pair that crosses the cut. The factor marginal is
//! the same ±1, and only on a pair that crosses the cut. There is no extra
//! threshold. Degree is not silent across the cut: the forensics on `920928c0`
//! showed the displaced gold had the higher in-degree, and silencing that
//! ballot let a scene advantage win the pair alone.

/// One extra window past the caller's `top_k`.
///
/// A permutation of the current top-k cannot bring in a turn the selector
/// already cut. Recall at k moves only if the tail pool contains turns that
/// sit past that cut. The smallest such extension that is the same size as
/// the cut the caller already fixed is one further window. The tail pool
/// length is `top_k * (1 + FACTOR_EXTRA_WINDOWS) - 1`.
pub const FACTOR_EXTRA_WINDOWS: usize = 1;

/// Parallel sum-product iterations.
///
/// Kschischang, Frey, and Loeliger (IEEE Trans. Inf. Theory, 2001) note that
/// sum-product on a factor graph with cycles has no natural termination.
/// Murphy, Weiss, and Jordan (UAI 1999) run the parallel schedule and report
/// a mean of 10.2 iterations to convergence on PYRAMID. The guard stops at
/// 10, the integer part of that mean, whether or not a threshold would still
/// be moving. A threshold would make the iteration count data-dependent.
pub const FACTOR_ITERATIONS: usize = 10;

/// Weight on the previous message. The update is
/// `(1 − η) * computed + η * previous`.
///
/// Murphy, Weiss, and Jordan (UAI 1999) damp with momentum η = 0.1. The same
/// convex combination is the damping step in Koller and Friedman, Probabilistic
/// Graphical Models (2009), chapter 11.
pub const FACTOR_DAMPING: f64 = 0.1;

/// Log-potential unit. Every factor uses this same magnitude.
///
/// Strictly positive potentials are the regime in which a loopy sum-product
/// fixed point exists (Mooij and Kappen, "Sufficient conditions for
/// convergence of the sum-product algorithm", arXiv:cs/0504030). The unit
/// coupling puts one nat between the favored state and the opposite state.
/// It is not a per-edge weight and it does not scale the Kemeny ballot.
pub const FACTOR_COUPLING: f64 = 1.0;
