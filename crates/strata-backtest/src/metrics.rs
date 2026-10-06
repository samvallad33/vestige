//! AUC, NDCG@5, rank-score log-likelihood, and the paired bootstrap.
//!
//! Log-likelihood uses rank order only (`p ∝ exp(-rank)`), so a raw
//! retrievability and a raw Need cannot decide it by scale.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::mechanism::Subject;
use crate::protocol::{BOOTSTRAP_B, MIN_DELTA, NDCG_K, ci_indices};
use crate::rng::{SplitMix64, subseed};

/// Natural logarithm. `libm` spells the C function `log`.
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Mann–Whitney AUC. Equal scores contribute 0.5. `None` without both classes.
pub fn auc(
    scores: &BTreeMap<Subject, f64>,
    positives: &BTreeSet<Subject>,
    candidates: &BTreeSet<Subject>,
) -> Option<f64> {
    let pos: Vec<&Subject> = candidates.intersection(positives).collect();
    let neg: Vec<&Subject> = candidates.difference(positives).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    let mut pairs = 0.0;
    for positive in pos {
        let left = scores.get(positive).copied().unwrap_or(0.0);
        for negative in &neg {
            let right = scores.get(*negative).copied().unwrap_or(0.0);
            acc += match left.total_cmp(&right) {
                Ordering::Greater => 1.0,
                Ordering::Equal => 0.5,
                Ordering::Less => 0.0,
            };
            pairs += 1.0;
        }
    }
    Some(acc / pairs)
}

/// Binary NDCG@k. `None` when either class is missing or the ideal DCG is 0.
pub fn ndcg_at_k(
    order: &[Subject],
    positives: &BTreeSet<Subject>,
    candidates: &BTreeSet<Subject>,
) -> Option<f64> {
    let n_pos = candidates.intersection(positives).count();
    let n_neg = candidates.difference(positives).count();
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut gained = 0.0;
    for (index, subject) in order.iter().take(NDCG_K).enumerate() {
        if positives.contains(subject) && candidates.contains(subject) {
            gained += 1.0 / libm::log2(index as f64 + 2.0);
        }
    }
    let ideal = ideal_dcg(n_pos);
    if ideal == 0.0 {
        None
    } else {
        Some(gained / ideal)
    }
}

fn ideal_dcg(n_pos: usize) -> f64 {
    let mut ideal = 0.0;
    for index in 0..n_pos.min(NDCG_K) {
        ideal += 1.0 / libm::log2(index as f64 + 2.0);
    }
    ideal
}

/// Mean `ln p(rank)` of the positives. Rank 0 is best. `p ∝ exp(-rank)`.
pub fn suffix_log_likelihood(order: &[Subject], positives: &BTreeSet<Subject>) -> Option<f64> {
    let n = order.len();
    if n == 0 {
        return None;
    }
    let mut denom = 0.0;
    let mut weight = Vec::with_capacity(n);
    for rank in 0..n {
        let term = libm::exp(-(rank as f64));
        weight.push(term);
        denom += term;
    }
    if denom == 0.0 {
        return None;
    }
    let mut acc = 0.0;
    let mut count = 0.0;
    for (rank, subject) in order.iter().enumerate() {
        if positives.contains(subject) {
            acc += ln(weight[rank] / denom);
            count += 1.0;
        }
    }
    if count == 0.0 {
        None
    } else {
        Some(acc / count)
    }
}

/// Best-first order: score descending, subject canon ascending on ties.
pub fn order_by_score(scores: &BTreeMap<Subject, f64>) -> Vec<Subject> {
    let mut order: Vec<Subject> = scores.keys().cloned().collect();
    order.sort_by(|left, right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(right))
    });
    order
}

/// Mean, then the pinned percentile interval. Empty input is `(0, 0, 0)`.
pub fn bootstrap_mean(values: &[f64], tag: &str) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let samples = resample(values.len(), tag, |idx| values[idx]);
    let (lo, hi) = ci_indices(BOOTSTRAP_B);
    (mean, samples[lo], samples[hi])
}

/// Paired difference `a - b`. Both slices are the same cut order.
pub fn bootstrap_paired_diff(left: &[f64], right: &[f64], tag: &str) -> (f64, f64, f64) {
    assert_eq!(left.len(), right.len());
    if left.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / left.len() as f64;
    let samples = resample(left.len(), tag, |idx| left[idx] - right[idx]);
    let (lo, hi) = ci_indices(BOOTSTRAP_B);
    (mean, samples[lo], samples[hi])
}

fn resample(n: usize, tag: &str, value_at: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut rng = SplitMix64::new(subseed(tag));
    let mut samples = Vec::with_capacity(BOOTSTRAP_B);
    for _ in 0..BOOTSTRAP_B {
        let mut acc = 0.0;
        for _draw in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            acc += value_at(idx);
        }
        samples.push(acc / n as f64);
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    samples
}

/// Separated only outside ±[`MIN_DELTA`] with a CI that excludes 0.
pub fn claim_delta(
    diff: f64,
    lo: f64,
    hi: f64,
    positive: &'static str,
    negative: &'static str,
    n: usize,
) -> &'static str {
    if n == 0 {
        "undefined_no_cuts"
    } else if diff >= MIN_DELTA && lo > 0.0 {
        positive
    } else if diff <= -MIN_DELTA && hi < 0.0 {
        negative
    } else {
        "not_separated"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> Subject {
        Subject::Node(id.into())
    }

    #[test]
    fn auc_is_one_when_positives_strictly_outrank() {
        let mut scores = BTreeMap::new();
        scores.insert(node("p"), 1.0);
        scores.insert(node("n"), 0.0);
        let mut pos = BTreeSet::new();
        pos.insert(node("p"));
        let mut cand = BTreeSet::new();
        cand.insert(node("p"));
        cand.insert(node("n"));
        assert_eq!(auc(&scores, &pos, &cand), Some(1.0));
        scores.insert(node("n"), 1.0);
        assert_eq!(auc(&scores, &pos, &cand), Some(0.5));
        assert_eq!(auc(&scores, &BTreeSet::new(), &cand), None);
    }

    #[test]
    fn ndcg_of_the_ideal_order_is_one() {
        let order = vec![node("p"), node("n")];
        let mut pos = BTreeSet::new();
        pos.insert(node("p"));
        let cand = order.iter().cloned().collect();
        let score = ndcg_at_k(&order, &pos, &cand).unwrap();
        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_is_byte_stable_for_a_fixed_tag() {
        let values = [0.0, 1.0, 0.5, 0.25];
        let once = bootstrap_mean(&values, "boot/test/mean");
        let twice = bootstrap_mean(&values, "boot/test/mean");
        assert_eq!(once.0.to_bits(), twice.0.to_bits());
        assert_eq!(once.1.to_bits(), twice.1.to_bits());
        assert_eq!(once.2.to_bits(), twice.2.to_bits());
        assert!(once.1 <= once.0 && once.0 <= once.2);
    }
}
