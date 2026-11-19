//! Monotone submodular aspect-coverage selector for multi-session summarization.
//!
//! Grounded in Lin & Bilmes (ACL 2011) saturated coverage + diversity reward,
//! Nemhauser-Wolsey-Fisher (1978) `(1 - 1/e)` greedy bound, Fisher-Nemhauser-Wolsey (1978)
//! truncated partition matroid (one slot per session), and Barzilay & Elhadad (1997)
//! lexical-chain aspect cohesion.
//!
//! Pure `Q16.16` integer arithmetic (`Q16_ONE = 65_536`). Zero floats, zero embeddings.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;

/// Unit `1.0` in unsigned `Q16.16` fixed-point math (`2^16`).
pub(in crate::retrieval) const Q16_ONE: u64 = 65_536;

/// One candidate session for submodular aspect-coverage selection.
#[derive(Clone, Debug)]
pub(in crate::retrieval) struct AspectSessionCand {
    pub(in crate::retrieval) session_idx: usize,
    /// Base lexical/relevance score in `Q16.16` (used for deterministic tie-breaking & anchor).
    pub(in crate::retrieval) base_score_q16: u64,
    /// Session timestamp (seconds) for deterministic tie-breaking.
    pub(in crate::retrieval) session_ts: i64,
    /// Distinct aspect IDs (rare entity IDs, high-IC terms, or WordNet chain IDs) ->
    /// normalized per-session coverage contribution `c(a, s)` in `[0, Q16_ONE]`.
    pub(in crate::retrieval) aspect_cov_q16: BTreeMap<u32, u64>,
}

/// Deterministic corpus Shannon Information Content `IC(a) = log2((N + 1) / (df + 1))` in `Q16.16`.
#[inline]
pub(in crate::retrieval) fn aspect_ic_q16(df: u32, n_docs: u32) -> u64 {
    let num = (n_docs as u64).saturating_add(1).max(2);
    let den = (df as u64).saturating_add(1).min(num);
    let log_num = log2_u64_q16(num);
    let log_den = log2_u64_q16(den);
    log_num.saturating_sub(log_den)
}

/// Exact integer square root for concave diversity reward `sqrt(x)` in `Q16.16`.
/// Returns `floor(sqrt(x * Q16_ONE))` so `isqrt_q16(Q16_ONE) == Q16_ONE`.
#[inline]
pub(in crate::retrieval) fn isqrt_q16(x_q16: u64) -> u64 {
    let scaled: u128 = (x_q16 as u128).saturating_mul(Q16_ONE as u128);
    if scaled == 0 {
        return 0;
    }
    let mut hi: u128 = 1u128 << 64;
    let mut lo: u128 = 0;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1) + 1;
        if mid.saturating_mul(mid) <= scaled {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo.min(u64::MAX as u128) as u64
}

/// Evaluate `F(S) = sum_{a in A} w(a) * min(Q16_ONE, cov(a, S))` in `Q16.16`.
/// Monotone non-decreasing and submodular by concavity of `min(1, z)` composed with
/// non-negative modular sums `cov(a, S) = sum_{s in S} c(a, s)`.
#[cfg(test)]
pub(in crate::retrieval) fn evaluate_aspect_coverage_q16(
    selected: &[&AspectSessionCand],
    aspect_weights_q16: &BTreeMap<u32, u64>,
) -> u64 {
    let mut cov: BTreeMap<u32, u64> = BTreeMap::new();
    for cand in selected {
        for (&aid, &c_q16) in &cand.aspect_cov_q16 {
            let entry = cov.entry(aid).or_insert(0);
            *entry = entry.saturating_add(c_q16).min(Q16_ONE);
        }
    }
    let mut total: u64 = 0;
    for (aid, c_sat) in cov {
        if let Some(&w_q16) = aspect_weights_q16.get(&aid) {
            let contrib = ((w_q16 as u128).saturating_mul(c_sat as u128) >> 16) as u64;
            total = total.saturating_add(contrib);
        }
    }
    total
}

/// Greedy `(1 - 1/e)` monotone submodular selector enforcing strict **one slot per session**
/// (truncated partition matroid over `session_idx`) with a small concave relevance tie-breaker.
///
/// Gated by caller to `eff_summary_request(query)` so Single-Hop and Adversarial are untouched.
pub(in crate::retrieval) fn select_summary_aspect_sessions_q16(
    cands: &[AspectSessionCand],
    aspect_weights_q16: &BTreeMap<u32, u64>,
    k: usize,
    lambda_rel_q16: u64,
) -> Vec<usize> {
    if cands.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut selected: Vec<usize> = Vec::with_capacity(k.min(cands.len()));
    let mut used_sessions: BTreeSet<usize> = BTreeSet::new();
    let mut cov: BTreeMap<u32, u64> = BTreeMap::new();

    // Normalize base relevance against max candidate score for scale-free tie-breaking.
    let smax_q16 = cands
        .iter()
        .map(|c| c.base_score_q16)
        .max()
        .unwrap_or(0)
        .max(1);

    for _step in 0..k {
        let mut best: Option<(usize, u64, u64, i64, usize)> = None;
        for (idx, cand) in cands.iter().enumerate() {
            if used_sessions.contains(&cand.session_idx) {
                continue;
            }
            let mut marginal_cov_q16: u64 = 0;
            for (&aid, &c_q16) in &cand.aspect_cov_q16 {
                let cur = cov.get(&aid).copied().unwrap_or(0);
                if cur >= Q16_ONE {
                    continue;
                }
                let delta = cur.saturating_add(c_q16).min(Q16_ONE).saturating_sub(cur);
                if delta == 0 {
                    continue;
                }
                if let Some(&w_q16) = aspect_weights_q16.get(&aid) {
                    let gain = ((w_q16 as u128).saturating_mul(delta as u128) >> 16) as u64;
                    marginal_cov_q16 = marginal_cov_q16.saturating_add(gain);
                }
            }
            let rel_norm = (((cand.base_score_q16 as u128) << 16) / (smax_q16 as u128)) as u64;
            let rel_sqrt = isqrt_q16(rel_norm.min(Q16_ONE));
            let rel_term =
                ((lambda_rel_q16 as u128).saturating_mul(rel_sqrt as u128) >> 16) as u64;
            let total_gain = marginal_cov_q16.saturating_add(rel_term);

            let better = match best {
                None => true,
                Some((_, b_gain, b_base, b_ts, b_sid)) => {
                    total_gain > b_gain
                        || (total_gain == b_gain && cand.base_score_q16 > b_base)
                        || (total_gain == b_gain
                            && cand.base_score_q16 == b_base
                            && cand.session_ts > b_ts)
                        || (total_gain == b_gain
                            && cand.base_score_q16 == b_base
                            && cand.session_ts == b_ts
                            && cand.session_idx < b_sid)
                }
            };
            if better {
                best = Some((
                    idx,
                    total_gain,
                    cand.base_score_q16,
                    cand.session_ts,
                    cand.session_idx,
                ));
            }
        }
        let Some((best_idx, _, _, _, best_sid)) = best else {
            break;
        };
        used_sessions.insert(best_sid);
        selected.push(best_sid);
        for (&aid, &c_q16) in &cands[best_idx].aspect_cov_q16 {
            let entry = cov.entry(aid).or_insert(0);
            *entry = entry.saturating_add(c_q16).min(Q16_ONE);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_cand(sid: usize, base: u64, aspects: &[(u32, u64)]) -> AspectSessionCand {
        let mut m = BTreeMap::new();
        for &(a, c) in aspects {
            m.insert(a, c);
        }
        AspectSessionCand {
            session_idx: sid,
            base_score_q16: base,
            session_ts: (sid as i64) * 1000,
            aspect_cov_q16: m,
        }
    }

    #[test]
    fn submodular_diminishing_returns_holds_exactly() {
        let mut w = BTreeMap::new();
        w.insert(1, Q16_ONE * 3);
        w.insert(2, Q16_ONE * 2);
        let s1 = mk_cand(1, Q16_ONE, &[(1, Q16_ONE / 2)]);
        let s2 = mk_cand(2, Q16_ONE, &[(1, Q16_ONE / 2), (2, Q16_ONE / 2)]);
        let x = mk_cand(3, Q16_ONE, &[(1, Q16_ONE / 2), (2, Q16_ONE / 2)]);

        let f_s = evaluate_aspect_coverage_q16(&[&s1], &w);
        let f_sx = evaluate_aspect_coverage_q16(&[&s1, &x], &w);
        let f_t = evaluate_aspect_coverage_q16(&[&s1, &s2], &w);
        let f_tx = evaluate_aspect_coverage_q16(&[&s1, &s2, &x], &w);
        assert!(f_sx >= f_s);
        assert!(f_tx >= f_t);
        // Diminishing returns: Delta(x | S) >= Delta(x | T) for S subseteq T.
        assert!(f_sx - f_s >= f_tx - f_t);
    }

    #[test]
    fn near_duplicate_sessions_do_not_crowd_out_new_aspect_gold() {
        let mut w = BTreeMap::new();
        w.insert(10, Q16_ONE * 4);
        w.insert(20, Q16_ONE * 4);
        w.insert(30, Q16_ONE * 4);
        // Session 0 and 1 are near-duplicates covering only aspect 10 with high base score;
        // Sessions 2 and 3 cover distinct rare aspects 20 and 30 with slightly lower base score.
        let cands = vec![
            mk_cand(0, Q16_ONE * 10, &[(10, Q16_ONE)]),
            mk_cand(1, Q16_ONE * 9, &[(10, Q16_ONE)]),
            mk_cand(2, Q16_ONE * 6, &[(20, Q16_ONE)]),
            mk_cand(3, Q16_ONE * 5, &[(30, Q16_ONE)]),
        ];
        let picked = select_summary_aspect_sessions_q16(&cands, &w, 3, Q16_ONE / 4);
        assert_eq!(picked, vec![0, 2, 3]);
    }

    #[test]
    fn enforces_strict_one_slot_per_session() {
        let mut w = BTreeMap::new();
        w.insert(1, Q16_ONE * 5);
        let cands = vec![
            mk_cand(7, Q16_ONE * 10, &[(1, Q16_ONE / 2)]),
            mk_cand(7, Q16_ONE * 9, &[(1, Q16_ONE / 2)]),
            mk_cand(8, Q16_ONE * 2, &[(1, Q16_ONE / 2)]),
        ];
        let picked = select_summary_aspect_sessions_q16(&cands, &w, 3, Q16_ONE / 4);
        assert_eq!(picked, vec![7, 8]);
    }
}
