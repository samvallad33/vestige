//! ScalDPP Exact Incremental Q16 Cholesky MAP-DPP + GeoRAG Facility Location (`2604.03240`, `2606.29328`).
//!
//! Solves the Single-Point Coverage Barrier on multi-session `summarization`, `event_ordering`,
//! and `multi_session_reasoning` by maximizing `log det(Gamma_Y)` where `Gamma_{ij} = q_i * S_{ij} * q_j`
//! over the exact sparse entity + rare-term + temporal-phase similarity kernel `S_{ij}` in pure integer math.

use crate::retrieval::graph::ghostlink_dgs::integer_sqrt;
use crate::retrieval::{DetHashSet, Index};

const Q16_ONE: u64 = 65_536;

/// Select `k` sessions from `cands` (`(session_idx, quality_q16)`) maximizing the ScalDPP
/// quality-diversity log-determinant via greedy incremental Cholesky factorization (`O(N k^2)`).
/// Always locks `cands[0]` at slot 0 when `lock_head` is true so `@1` precision is preserved.
pub(in crate::retrieval) fn select_scaldpp_cholesky_sessions_q16(
    ix: &Index,
    cands: &[(usize, u64)],
    k: usize,
    lock_head: bool,
) -> Vec<usize> {
    let n = cands.len().min(32);
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let target_k = k.min(n);
    if target_k <= 1 {
        return vec![cands[0].0];
    }

    let ns = ix.session_turns.len().max(1);
    let max_q = cands.iter().take(n).map(|&(_, q)| q).max().unwrap_or(1).max(1);

    // Precompute normalized quality `q_i in [1, Q16_ONE]` and sparse signature sets per candidate.
    let mut qual = vec![1u64; n];
    let mut sigs: Vec<DetHashSet<u32>> = Vec::with_capacity(n);
    let mut phases = vec![0usize; n];
    for i in 0..n {
        let (si, q) = cands[i];
        qual[i] = ((q as u128 * Q16_ONE as u128) / (max_q as u128)).clamp(1_024, Q16_ONE as u128) as u64;
        let mut s: DetHashSet<u32> = DetHashSet::default();
        for &e in &ix.session_entities[si] {
            s.insert(e);
        }
        for &r in ix.session_rare_nt[si].iter().take(16) {
            s.insert(r);
        }
        sigs.push(s);
        phases[i] = (si.saturating_mul(4) / ns).min(3);
    }

    // Sparse positive-definite similarity kernel `S_{ij} in [0, Q16_ONE]` (`S_{ii} = Q16_ONE`):
    // blends exact entity/rare-term Jaccard (75%) + same chronological quarter overlap (25%).
    let sim_ij = |i: usize, j: usize| -> i64 {
        if i == j {
            return Q16_ONE as i64;
        }
        let a = &sigs[i];
        let b = &sigs[j];
        let jacc_q16 = if a.is_empty() || b.is_empty() {
            0u64
        } else {
            let inter = a.iter().filter(|x| b.contains(x)).count() as u64;
            let uni = (a.len() + b.len()) as u64 - inter;
            if uni == 0 {
                0
            } else {
                inter.saturating_mul(Q16_ONE) / uni
            }
        };
        let phase_q16 = if phases[i] == phases[j] { Q16_ONE } else { 0 };
        // Keep off-diagonal strictly < 0.88 * Q16 so the kernel stays well-conditioned PSD.
        let s = (jacc_q16.saturating_mul(66) + phase_q16.saturating_mul(22)) / 100;
        s.min(57_671) as i64
    };

    // Quality-weighted Gram kernel `L_{ij} = (q_i * S_{ij} * q_j) >> 32` in Q16.
    let gram_ij = |i: usize, j: usize| -> i64 {
        let qi = qual[i] as i128;
        let qj = qual[j] as i128;
        let s = sim_ij(i, j) as i128;
        (((qi * s) >> 16) * qj >> 16) as i64
    };

    // Incremental Cholesky state: `d2[i]` = residual diagonal variance `L_{ii} - sum_{t} c[i][t]^2`.
    let mut d2: Vec<i64> = (0..n).map(|i| gram_ij(i, i).max(1)).collect();
    let mut c_mat: Vec<Vec<i64>> = vec![Vec::with_capacity(target_k); n];
    let mut chosen = vec![false; n];
    let mut selected: Vec<usize> = Vec::with_capacity(target_k);

    for step in 0..target_k {
        let pick = if step == 0 && lock_head {
            0usize
        } else {
            let mut best_i = None;
            let mut best_val = i64::MIN;
            for i in 0..n {
                if !chosen[i] && d2[i] > best_val {
                    best_val = d2[i];
                    best_i = Some(i);
                }
            }
            match best_i {
                Some(i) => i,
                None => break,
            }
        };
        chosen[pick] = true;
        selected.push(cands[pick].0);
        if selected.len() >= target_k {
            break;
        }

        // `d_pick = sqrt(d2[pick] * Q16_ONE)` in Q16.
        let d2_pos = d2[pick].max(1) as u64;
        let d_pick = integer_sqrt((d2_pos as u128 * Q16_ONE as u128).min(u64::MAX as u128) as u64)
            .max(1) as i64;

        for i in 0..n {
            if chosen[i] {
                continue;
            }
            let mut dot: i128 = 0;
            for t in 0..step {
                dot += (c_mat[pick][t] as i128 * c_mat[i][t] as i128) >> 16;
            }
            let num = (gram_ij(pick, i) as i128).saturating_sub(dot);
            let e_i = ((num << 16) / (d_pick as i128)).clamp(-1_048_576, 1_048_576) as i64;
            c_mat[i].push(e_i);
            let e_sq = ((e_i as i128 * e_i as i128) >> 16) as i64;
            d2[i] = d2[i].saturating_sub(e_sq).max(0);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn scaldpp_breaks_single_cluster_redundancy() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "user".into(),
                    content: "Deploying alpha kubernetes cluster ingress controller".into(),
                    ..Default::default()
                }],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 2000,
                turns: vec![Turn {
                    id: "t1".into(),
                    role: "user".into(),
                    content: "Updating alpha kubernetes cluster ingress controller again".into(),
                    ..Default::default()
                }],
            },
            Session {
                id: "s2".into(),
                date: String::new(),
                timestamp: 3000,
                turns: vec![Turn {
                    id: "t2".into(),
                    role: "user".into(),
                    content: "Migrating postgres ledger schema and backup snapshots".into(),
                    ..Default::default()
                }],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        // s1 has slightly higher raw quality than s2, but is near-duplicate of s0.
        let cands = vec![(0, 65_536), (1, 60_000), (2, 56_000)];
        let picked = select_scaldpp_cholesky_sessions_q16(&ix, &cands, 2, true);
        assert_eq!(picked, vec![0, 2], "DPP must prefer orthogonal session s2 over redundant s1");
    }
}
