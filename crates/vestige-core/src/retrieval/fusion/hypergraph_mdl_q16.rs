//! #7 Hypergraph Minimum Hitting Set + #8 Integer Pivoted Cholesky IC Orthogonalization
//! + #12 Rissanen MDL Adaptive Seating (`Q16.16`).
//!
//! Used after Slot 0 is anchored on multi-aspect summarization and aggregation queries so
//! near-duplicate sessions covering the same saturated cluster lose residual Schur-complement
//! IC energy (`||v_perp||^2`), while compact sets hitting uncovered query/topic hyperedges minimize
//! Rissanen Minimum Description Length bits.
//! Pure integer arithmetic, deterministic `BTreeMap`/`BTreeSet` iteration.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::{AspectSessionCand, Q16_ONE};

/// Integer square root (`floor(sqrt(x))`) for exact Schur-complement normalization.
#[inline]
fn isqrt_u128(x: u128) -> u64 {
    if x == 0 {
        return 0;
    }
    let mut r = 1u128 << (127 - x.leading_zeros()).div_ceil(2);
    loop {
        let next = (r + x / r) >> 1;
        if next >= r {
            return r as u64;
        }
        r = next;
    }
}

/// Select `k` unique sessions from `cands`:
/// - Step 0 anchors the maximum-`base_score_q16` candidate (`@1` preservation).
/// - Steps `1..k` fuse **Integer Pivoted Cholesky Orthogonalization** (residual Schur IC norm
///   after projecting out previously selected sessions' aspect profiles), **Hypergraph Hitting Set**
///   bonus for uncovered aspects/hyperedges, and **Rissanen MDL** bit-efficiency regularization.
pub(in crate::retrieval) fn select_pivoted_cholesky_hypergraph_mdl_q16(
    cands: &[AspectSessionCand],
    aspect_weights_q16: &BTreeMap<u32, u64>,
    k: usize,
) -> Vec<usize> {
    if cands.is_empty() || k == 0 {
        return Vec::new();
    }
    let m = cands.len();
    let mut selected: Vec<usize> = Vec::with_capacity(k.min(m));
    let mut used_sessions: BTreeSet<usize> = BTreeSet::new();

    // Index all weighted aspects (`0..num_a`) so Gram / Cholesky projections run in dense arrays.
    let mut aid_list: Vec<u32> = aspect_weights_q16.keys().copied().collect();
    aid_list.sort_unstable();
    let mut aid_to_idx: BTreeMap<u32, usize> = BTreeMap::new();
    for (idx, &aid) in aid_list.iter().enumerate() {
        aid_to_idx.insert(aid, idx);
    }

    // Build weighted aspect profile vector v_i[a] in Q16.16 per candidate.
    let mut vecs: Vec<BTreeMap<usize, u64>> = Vec::with_capacity(m);
    let mut initial_norm_sq: Vec<u128> = vec![0; m];
    for (ci, cand) in cands.iter().enumerate() {
        let mut prof: BTreeMap<usize, u64> = BTreeMap::new();
        let mut nsq: u128 = 0;
        for (&aid, &cov) in &cand.aspect_cov_q16 {
            if let (Some(&a_idx), Some(&w_q16)) = (aid_to_idx.get(&aid), aspect_weights_q16.get(&aid)) {
                let val = ((cov.min(Q16_ONE) as u128).saturating_mul(w_q16 as u128) >> 16) as u64;
                if val > 0 {
                    prof.insert(a_idx, val);
                    nsq = nsq.saturating_add((val as u128).saturating_mul(val as u128));
                }
            }
        }
        initial_norm_sq[ci] = nsq;
        vecs.push(prof);
    }

    let smax_q16 = cands
        .iter()
        .map(|c| c.base_score_q16)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut uncovered_hyperedges: BTreeSet<usize> = (0..aid_list.len()).collect();
    // Orthonormal Cholesky basis vectors stored as sparse `BTreeMap<usize, i64>` in Q16.16.
    let mut basis: Vec<BTreeMap<usize, i64>> = Vec::new();
    let mut resid_norm_sq: Vec<u128> = initial_norm_sq;

    for step in 0..k {
        if step == 0 {
            // Preserve Slot 0 (@1): purely highest base_score_q16.
            let mut best0: Option<(usize, u64, i64, usize)> = None;
            for (ci, cand) in cands.iter().enumerate() {
                let better = match best0 {
                    None => true,
                    Some((_, b_sc, b_ts, b_sid)) => {
                        cand.base_score_q16 > b_sc
                            || (cand.base_score_q16 == b_sc && cand.session_ts > b_ts)
                            || (cand.base_score_q16 == b_sc
                                && cand.session_ts == b_ts
                                && cand.session_idx < b_sid)
                    }
                };
                if better {
                    best0 = Some((ci, cand.base_score_q16, cand.session_ts, cand.session_idx));
                }
            }
            let Some((best_ci, _, _, best_sid)) = best0 else {
                break;
            };
            used_sessions.insert(best_sid);
            selected.push(best_sid);
            for &a_idx in vecs[best_ci].keys() {
                uncovered_hyperedges.remove(&a_idx);
            }
            if let Some(unit) = normalize_sparse_q16(&vecs[best_ci]) {
                subtract_projection(&unit, &vecs, &mut resid_norm_sq);
                basis.push(unit);
            }
            continue;
        }

        // Compute max residual norm & hitting weight to normalize Q16 gains cleanly.
        let mut max_chol: u64 = 1;
        let mut hit_gains = vec![0u64; m];
        let mut max_hit: u64 = 1;
        let mut chol_scores = vec![0u64; m];
        for (ci, cand) in cands.iter().enumerate() {
            if used_sessions.contains(&cand.session_idx) {
                continue;
            }
            let chol = isqrt_u128(resid_norm_sq[ci]);
            chol_scores[ci] = chol;
            if chol > max_chol {
                max_chol = chol;
            }
            let mut h_sum: u64 = 0;
            for (&a_idx, &val) in &vecs[ci] {
                if uncovered_hyperedges.contains(&a_idx) {
                    h_sum = h_sum.saturating_add(val);
                }
            }
            // Rissanen MDL compactness adjustment: penalize candidates covering only already-hit noise
            // relative to description cost `log2(2 + covered_count)`.
            let mdl_cost_bits = (u64::BITS - (2u64 + vecs[ci].len() as u64).leading_zeros()) as u64;
            let eff_hit = if h_sum > 0 {
                h_sum.saturating_mul(2) / mdl_cost_bits.max(1)
            } else {
                0
            };
            hit_gains[ci] = eff_hit;
            if eff_hit > max_hit {
                max_hit = eff_hit;
            }
        }

        let mut best: Option<(usize, u64, u64, i64, usize)> = None;
        for (ci, cand) in cands.iter().enumerate() {
            if used_sessions.contains(&cand.session_idx) {
                continue;
            }
            let rel_n = (((cand.base_score_q16 as u128) << 16) / (smax_q16 as u128))
                .min(Q16_ONE as u128) as u64;
            let chol_n = (((chol_scores[ci] as u128) << 16) / (max_chol as u128))
                .min(Q16_ONE as u128) as u64;
            let hit_n = (((hit_gains[ci] as u128) << 16) / (max_hit as u128))
                .min(Q16_ONE as u128) as u64;
            // Fused Q16 objective: 40% relevance + 35% Cholesky orthogonal novelty + 25% Hypergraph/MDL.
            let score = ((rel_n.saturating_mul(40)
                .saturating_add(chol_n.saturating_mul(35))
                .saturating_add(hit_n.saturating_mul(25)))
                / 100) as u64;
            let better = match best {
                None => true,
                Some((_, b_sc, b_base, b_ts, b_sid)) => {
                    score > b_sc
                        || (score == b_sc && cand.base_score_q16 > b_base)
                        || (score == b_sc
                            && cand.base_score_q16 == b_base
                            && cand.session_ts > b_ts)
                        || (score == b_sc
                            && cand.base_score_q16 == b_base
                            && cand.session_ts == b_ts
                            && cand.session_idx < b_sid)
                }
            };
            if better {
                best = Some((ci, score, cand.base_score_q16, cand.session_ts, cand.session_idx));
            }
        }
        let Some((best_ci, _, _, _, best_sid)) = best else {
            break;
        };
        used_sessions.insert(best_sid);
        selected.push(best_sid);
        for &a_idx in vecs[best_ci].keys() {
            uncovered_hyperedges.remove(&a_idx);
        }
        // Orthogonalize `vecs[best_ci]` against existing basis before adding as next Cholesky direction.
        let mut orth: BTreeMap<usize, i64> = vecs[best_ci].iter().map(|(&k, &v)| (k, v as i64)).collect();
        for b_vec in &basis {
            let mut dot_q32: i128 = 0;
            for (&k, &bv) in b_vec {
                if let Some(&ov) = orth.get(&k) {
                    dot_q32 += (ov as i128) * (bv as i128);
                }
            }
            let proj_q16 = (dot_q32 >> 16) as i64;
            if proj_q16 != 0 {
                for (&k, &bv) in b_vec {
                    let sub = ((proj_q16 as i128 * bv as i128) >> 16) as i64;
                    let e = orth.entry(k).or_insert(0);
                    *e -= sub;
                }
            }
        }
        if let Some(unit) = normalize_signed_sparse_q16(&orth) {
            subtract_projection(&unit, &vecs, &mut resid_norm_sq);
            basis.push(unit);
        }
    }
    selected
}

fn normalize_sparse_q16(v: &BTreeMap<usize, u64>) -> Option<BTreeMap<usize, i64>> {
    let nsq: u128 = v.values().map(|&x| (x as u128) * (x as u128)).sum();
    let norm = isqrt_u128(nsq);
    if norm == 0 {
        return None;
    }
    let mut out = BTreeMap::new();
    for (&k, &val) in v {
        let q = (((val as u128) << 16) / (norm as u128)) as i64;
        if q != 0 {
            out.insert(k, q);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn normalize_signed_sparse_q16(v: &BTreeMap<usize, i64>) -> Option<BTreeMap<usize, i64>> {
    let nsq: u128 = v
        .values()
        .map(|&x| {
            let ax = x.unsigned_abs() as u128;
            ax * ax
        })
        .sum();
    let norm = isqrt_u128(nsq);
    if norm == 0 {
        return None;
    }
    let mut out = BTreeMap::new();
    for (&k, &val) in v {
        let sign: i64 = if val < 0 { -1 } else { 1 };
        let q = ((((val.unsigned_abs() as u128) << 16) / (norm as u128)) as i64) * sign;
        if q != 0 {
            out.insert(k, q);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn subtract_projection(
    unit: &BTreeMap<usize, i64>,
    vecs: &[BTreeMap<usize, u64>],
    resid_norm_sq: &mut [u128],
) {
    for (ci, v) in vecs.iter().enumerate() {
        let mut dot_q16: i128 = 0;
        for (&k, &uv) in unit {
            if let Some(&val) = v.get(&k) {
                dot_q16 += ((val as i128) * (uv as i128)) >> 16;
            }
        }
        let proj_sq = dot_q16.saturating_mul(dot_q16) as u128;
        resid_norm_sq[ci] = resid_norm_sq[ci].saturating_sub(proj_sq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(sid: usize, base: u64, asps: &[(u32, u64)]) -> AspectSessionCand {
        let mut m = BTreeMap::new();
        for &(a, c) in asps {
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
    fn pivoted_cholesky_hypergraph_anchors_slot0_and_orthogonalizes_tail() {
        let mut w = BTreeMap::new();
        w.insert(1, Q16_ONE * 4);
        w.insert(2, Q16_ONE * 4);
        w.insert(3, Q16_ONE * 4);
        let cands = vec![
            mk(0, Q16_ONE * 10, &[(1, Q16_ONE)]),
            mk(1, Q16_ONE * 9, &[(1, Q16_ONE)]),
            mk(2, Q16_ONE * 5, &[(2, Q16_ONE)]),
            mk(3, Q16_ONE * 5, &[(3, Q16_ONE)]),
        ];
        let picked = select_pivoted_cholesky_hypergraph_mdl_q16(&cands, &w, 3);
        assert_eq!(picked[0], 0);
        assert!(picked.contains(&2) && picked.contains(&3));
        assert!(!picked.contains(&1));
    }
}
