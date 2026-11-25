//! Strata-Weighted `BISCEPTER` Probability-Mass Bisection (`Q16.16`).
//!
//! Standard `git bisect` splits a commit range at the 50/50 commit-count
//! midpoint, ignoring empirical bug-inducing commit (BIC) latency priors and
//! Strata's recorded causal signals. Inspired by BISCEPTER (arXiv:2610.02995),
//! this module replaces uniform count midpoints with deterministic `Q16.16`
//! weighted-median probability-mass pivots combining:
//!
//! 1. **Power-law recency latency prior** (empirical BIC mass concentrates heavily
//!    in the newest commits of `good..bad`).
//! 2. **Exact file/hunk overlap** with `causal_walk` leads (`4.0x`).
//! 3. **Unwitnessed commit risk** — commits with no signed Reflex test witness (`2.0x`).
//! 4. **Revert / oscillation adjacency** — commits linked by `corrects` (`2.0x`).
//! 5. **Self-plasticizing Strata outcome conductance (`#854`)** — commits with
//!    historical `outcome:bad` vs `outcome:good` feedback (`0.25x..4.0x`).
//!
//! Zero floating-point nondeterminism, zero external dependencies, bit-exact
//! across platforms.

use vestige_core::composition::{CONDUCTANCE_Q16_ONE, outcome_conductance_multiplier_q16};

/// `1.0` in `Q16.16` fixed-point.
pub const Q16_ONE: u64 = CONDUCTANCE_Q16_ONE;

/// Deterministic Strata risk signals for a single commit in `good..bad`.
///
/// `age_from_bad` is `0` for the newest commit (`bad`) and `n - 1` for the oldest
/// candidate right after `good`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommitRiskSignals {
    /// Distance in commits from `bad` (`0` = newest / `bad` itself).
    pub age_from_bad: usize,
    /// True if the commit touched an exact `file:` or hunk named by `causal_walk`.
    pub touches_failing_anchor: bool,
    /// True if Strata holds zero passing Reflex witnesses for this commit.
    pub unwitnessed: bool,
    /// True if this commit participates in a `corrects` / revert oscillation chain.
    pub oscillation_neighbor: bool,
    /// Recorded Strata outcome counts `(bad_outcomes, good_outcomes)` for this commit/file.
    pub bad_outcomes: u32,
    /// Positive outcome counts (`outcome:good`) reducing failure prior.
    pub good_outcomes: u32,
}

/// Deterministic `Q16.16` probability-mass weight for one commit.
///
/// Never returns zero so every commit in `good..bad` retains strictly positive
/// probability mass and bisection always terminates in finite probes.
pub fn score_commit_prior_q16(signals: &CommitRiskSignals) -> u64 {
    // Power-law recency kernel: 1 / (age + 1), scaled in Q16.16 with a 1/16 floor
    // so ancient commits still keep positive mass while recent commits dominate
    // (mirroring BISCEPTER's heavy-tailed BIC latency distribution).
    let denom = (signals.age_from_bad as u64).saturating_add(1);
    let recency_q16 = (Q16_ONE / denom).max(Q16_ONE >> 4);

    let mut weight = recency_q16;
    if signals.touches_failing_anchor {
        weight = weight.saturating_mul(4);
    }
    if signals.unwitnessed {
        weight = weight.saturating_mul(2);
    }
    if signals.oscillation_neighbor {
        weight = weight.saturating_mul(2);
    }

    // Apply #854 self-plasticizing outcome conductance: recorded bad/proved-bug
    // outcomes reinforce suspect probability mass ("causal_walk_proved" -> up to 2.0x),
    // while verified-good outcomes damp suspect mass ("dead_end" -> down to 0.20x).
    let bad_n = signals.bad_outcomes.min(16) as usize;
    let good_n = signals.good_outcomes.min(16) as usize;
    let mut prior_outcomes: Vec<&'static str> = Vec::with_capacity(bad_n + good_n);
    prior_outcomes.extend(std::iter::repeat_n("causal_walk_proved", bad_n));
    prior_outcomes.extend(std::iter::repeat_n("dead_end", good_n));
    let outcome_mult_q16 = outcome_conductance_multiplier_q16(&prior_outcomes);
    let adjusted = ((weight as u128 * outcome_mult_q16 as u128) >> 16) as u64;
    adjusted.max(1)
}

/// Choose the weighted-median pivot index over a slice of non-zero `Q16.16` weights.
///
/// Given oldest-to-newest candidate weights `weights[0..n]`, returns the index `i`
/// that minimizes `|mass(0..=i) - mass(i+1..n)|`, breaking exact ties toward the
/// higher-mass (newer) side. Returns `None` only when `weights` is empty.
pub fn weighted_median_pivot(weights: &[u64]) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    if weights.len() == 1 {
        return Some(0);
    }
    let total: u128 = weights.iter().map(|&w| u128::from(w.max(1))).sum();
    let half = total / 2;
    let mut prefix: u128 = 0;
    let mut best_idx = 0usize;
    let mut best_gap = u128::MAX;
    for (idx, &w) in weights.iter().enumerate() {
        prefix += u128::from(w.max(1));
        let gap = prefix.abs_diff(half);
        if gap <= best_gap {
            best_gap = gap;
            best_idx = idx;
        }
    }
    Some(best_idx)
}

/// Simulate deterministic BISCEPTER weighted bisection over an oldest-to-newest
/// commit slice `weights`, returning the number of probes needed to isolate
/// `first_bad_idx` (where `0..first_bad_idx` are good and `first_bad_idx..n` are bad).
pub fn simulate_weighted_probes(weights: &[u64], first_bad_idx: usize) -> usize {
    if weights.len() <= 1 || first_bad_idx >= weights.len() {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = weights.len() - 1;
    let mut probes = 0usize;
    while lo < hi {
        let slice = &weights[lo..=hi];
        // Pick an interior pivot strictly inside [lo, hi - 1] so the interval
        // always strictly shrinks on both good and bad verdicts.
        let mut rel = weighted_median_pivot(slice).unwrap_or(slice.len() / 2);
        if rel >= slice.len() - 1 {
            rel = slice.len() - 2;
        }
        let mid = lo + rel;
        probes += 1;
        if mid >= first_bad_idx {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    probes
}

/// Standard uniform 50/50 midpoint probe count for the same `[0..n)` range.
pub fn simulate_uniform_probes(n: usize, first_bad_idx: usize) -> usize {
    if n <= 1 || first_bad_idx >= n {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    let mut probes = 0usize;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        probes += 1;
        if mid >= first_bad_idx {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    probes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biscepter_weighted_pivot_cuts_probes_on_recent_and_anchored_bics() {
        // 128 commits oldest (0) -> newest (127); true BIC sits near the newest end (index 124)
        // and touches the exact failing file anchor from causal_walk.
        let n = 128usize;
        let first_bad = 124usize;
        let weights: Vec<u64> = (0..n)
            .map(|idx| {
                score_commit_prior_q16(&CommitRiskSignals {
                    age_from_bad: (n - 1) - idx,
                    touches_failing_anchor: idx == first_bad || idx == 120,
                    unwitnessed: idx >= 122,
                    oscillation_neighbor: false,
                    bad_outcomes: u32::from(idx == first_bad),
                    good_outcomes: 0,
                })
            })
            .collect();

        let weighted = simulate_weighted_probes(&weights, first_bad);
        let uniform = simulate_uniform_probes(n, first_bad);
        assert!(
            weighted < uniform,
            "BISCEPTER weighted pivot ({weighted} probes) must beat uniform 50/50 ({uniform} probes)"
        );
    }

    #[test]
    fn biscepter_pivot_is_strictly_deterministic_and_always_converges() {
        let weights: Vec<u64> = (0..32)
            .map(|i| {
                score_commit_prior_q16(&CommitRiskSignals {
                    age_from_bad: 31 - i,
                    touches_failing_anchor: i % 7 == 0,
                    unwitnessed: i % 3 == 0,
                    oscillation_neighbor: i == 15,
                    bad_outcomes: u32::from(i == 29),
                    good_outcomes: u32::from(i < 5),
                })
            })
            .collect();
        for target in 0..weights.len() {
            let p1 = simulate_weighted_probes(&weights, target);
            let p2 = simulate_weighted_probes(&weights, target);
            assert_eq!(p1, p2, "probe count must be deterministic for target={target}");
            assert!(p1 <= weights.len(), "bisection must always converge");
        }
    }

    #[test]
    fn outcome_conductance_shifts_biscepter_mass_toward_bad_commits() {
        let clean = score_commit_prior_q16(&CommitRiskSignals {
            age_from_bad: 10,
            good_outcomes: 4,
            ..Default::default()
        });
        let suspect = score_commit_prior_q16(&CommitRiskSignals {
            age_from_bad: 10,
            bad_outcomes: 4,
            ..Default::default()
        });
        assert!(
            suspect > clean * 4,
            "bad-outcome commit ({suspect}) must outweigh good-outcome commit ({clean})"
        );
    }
}
