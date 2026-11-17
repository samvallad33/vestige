//! #828: Buckley et al. (1994) / Rocchio (1971) Pseudo-Relevance Feedback (PRF)
//! in pure Q16.16 fixed-point integer arithmetic. Zero floats, zero open-class word lists.
//!
//! Given top-scoring seed turns (positive centroid `R+`) and lower-ranked distractor
//! turns (negative centroid `R-`), extracts discriminative expansion terms `t` weighted by:
//!   `w_rocchio(t) = max(0, beta * mean_{d in R+} w(t, d) - gamma * mean_{d in R-} w(t, d))`
//! where `w(t, d)` is integer IDF-weighted term specificity (`log2(1 + N / df(t))`)
//! normalized by turn content length `integer_sqrt(|d| + 4)`.
//!
//! Candidate sessions and turns are then scored by their alignment with the positive
//! Rocchio centroid, breaking close ties in favor of sessions/turns that share the
//! discriminative episode vocabulary of the top-ranked evidence nuclei.

use super::dph::{log2_q16, Q16};
use super::ghostlink_dgs::integer_sqrt;
use super::index::Index;
use super::scoring::QueryPlan;
use std::collections::BTreeMap;

/// Rocchio feedback parameters in Q8 (`256 = 1.0`):
///   `BETA_Q8 = 192` (`0.75`), `GAMMA_Q8 = 64` (`0.25`).
const BETA_Q8: i64 = 192;
const GAMMA_Q8: i64 = 64;

/// Build a sparse Q16.16 Rocchio expansion vector `term_id -> weight_q16` from
/// top-ranked positive seed turns `pos_turns` and negative distractor turns `neg_turns`.
pub(super) fn build_rocchio_centroid_q16(
    ix: &Index,
    plan: &QueryPlan,
    pos_turns: &[usize],
    neg_turns: &[usize],
) -> BTreeMap<u32, u64> {
    let mut centroid: BTreeMap<u32, u64> = BTreeMap::new();
    if pos_turns.is_empty() {
        return centroid;
    }
    let n_turns = (ix.turns.len() as u64).max(1);
    // Restrict expansion to discriminative terms (`2 <= df <= max(4, N / 8)`)
    let max_df = ((n_turns / 8).max(4)) as u32;

    let mut pos_acc: BTreeMap<u32, i64> = BTreeMap::new();
    let pos_n = (pos_turns.len() as i64).max(1);
    for (rank, &ti) in pos_turns.iter().enumerate() {
        let t = &ix.turns[ti];
        if t.is_question || t.is_backchannel {
            continue;
        }
        let len_sqrt_q8 = integer_sqrt(((t.content.len() as u64) + 4) << 16).max(256) as i64;
        let rank_decay_q8 = (256i64 / ((rank as i64) + 1)).max(64);
        for &term_id in t.content.iter().chain(t.entities.iter()) {
            if plan.group_of.contains_key(&term_id) {
                continue;
            }
            let Some(&df) = ix.t_stats.df.get(&term_id) else {
                continue;
            };
            if df < 2 || df > max_df {
                continue;
            }
            let idf_ratio_q16 = (Q16 as u64).saturating_add(((n_turns << 16) / (df as u64).max(1)).min((u64::MAX / 2) as u64));
            let idf_q16 = log2_q16(idf_ratio_q16).max(0);
            let ent_mult = if t.entities.contains(&term_id) { 2i64 } else { 1i64 };
            let contrib = ((idf_q16 as i128 * rank_decay_q8 as i128 * ent_mult as i128)
                / (len_sqrt_q8 as i128)) as i64;
            *pos_acc.entry(term_id).or_insert(0) += contrib;
        }
    }

    let mut neg_acc: BTreeMap<u32, i64> = BTreeMap::new();
    let neg_n = (neg_turns.len() as i64).max(1);
    for &ti in neg_turns {
        let t = &ix.turns[ti];
        let len_sqrt_q8 = integer_sqrt(((t.content.len() as u64) + 4) << 16).max(256) as i64;
        for &term_id in t.content.iter().chain(t.entities.iter()) {
            if !pos_acc.contains_key(&term_id) {
                continue;
            }
            let Some(&df) = ix.t_stats.df.get(&term_id) else {
                continue;
            };
            let idf_ratio_q16 = (Q16 as u64).saturating_add(((n_turns << 16) / (df as u64).max(1)).min((u64::MAX / 2) as u64));
            let idf_q16 = log2_q16(idf_ratio_q16).max(0);
            let contrib = ((idf_q16 as i128 * 128) / (len_sqrt_q8 as i128)) as i64;
            *neg_acc.entry(term_id).or_insert(0) += contrib;
        }
    }

    // Keep top-12 positive Rocchio expansion terms
    let mut scored_terms: Vec<(u32, u64)> = Vec::new();
    for (term_id, p_sum) in pos_acc {
        let p_mean = p_sum / pos_n;
        let n_mean = neg_acc.get(&term_id).copied().unwrap_or(0) / neg_n;
        let net = ((BETA_Q8 * p_mean) - (GAMMA_Q8 * n_mean)) >> 8;
        if net > 0 {
            scored_terms.push((term_id, net as u64));
        }
    }
    scored_terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored_terms.truncate(12);
    for (tid, w) in scored_terms {
        centroid.insert(tid, w);
    }
    centroid
}

/// Score a session `si` against the Rocchio centroid in Q16.16, normalized by session length.
pub(super) fn score_session_rocchio_q16(
    ix: &Index,
    si: usize,
    centroid: &BTreeMap<u32, u64>,
) -> u64 {
    if centroid.is_empty() || si >= ix.session_content.len() {
        return 0;
    }
    let mut raw_q16 = 0u64;
    for &e in &ix.session_entities[si] {
        if let Some(&w) = centroid.get(&e) {
            raw_q16 = raw_q16.saturating_add(w.saturating_mul(2));
        }
    }
    for &w_id in &ix.session_content[si] {
        if let Some(&w) = centroid.get(&w_id) {
            raw_q16 = raw_q16.saturating_add(w);
        }
    }
    if raw_q16 == 0 {
        return 0;
    }
    let s_len = (ix.session_content[si].len() as u64) + 8;
    let norm_q8 = integer_sqrt(s_len << 16).max(256);
    ((raw_q16 as u128 * 256) / norm_q8 as u128) as u64
}

/// Score a turn `ti` against the Rocchio centroid in Q16.16, normalized by turn length.
pub(super) fn score_turn_rocchio_q16(
    ix: &Index,
    ti: usize,
    centroid: &BTreeMap<u32, u64>,
) -> u64 {
    if centroid.is_empty() || ti >= ix.turns.len() {
        return 0;
    }
    let t = &ix.turns[ti];
    if t.is_question || t.is_backchannel {
        return 0;
    }
    let mut raw_q16 = 0u64;
    for &e in &t.entities {
        if let Some(&w) = centroid.get(&e) {
            raw_q16 = raw_q16.saturating_add(w.saturating_mul(2));
        }
    }
    for &w_id in &t.content {
        if let Some(&w) = centroid.get(&w_id) {
            raw_q16 = raw_q16.saturating_add(w);
        }
    }
    if raw_q16 == 0 {
        return 0;
    }
    let t_len = (t.content.len() as u64) + 4;
    let norm_q8 = integer_sqrt(t_len << 16).max(256);
    ((raw_q16 as u128 * 256) / norm_q8 as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocchio_constants_are_positive_and_bounded() {
        assert!(BETA_Q8 > GAMMA_Q8);
        assert_eq!(BETA_Q8 + GAMMA_Q8, 256);
    }
}
