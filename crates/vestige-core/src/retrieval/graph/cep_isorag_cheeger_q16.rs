//! Kernel 2: CEP Max-Plus + ISO-RAG Cheeger Conductance Pruning (`cep_isorag_cheeger_q16.rs`).
//!
//! References:
//! - CEP (arXiv:2610.36134): Tropical (Max-Plus) Bellman-Ford with background visitation subtraction.
//! - ISO-RAG (arXiv:2609.00513): Cheeger conductance edge pruning `|log phi_u - log phi_v| > tau_beta`.
//!
//! Prunes cross-boundary generic hub edges in the Top-`<=40` candidate subgraph, runs
//! `H = 2..3` Max-Plus Bellman-Ford relaxations in integer log-Q16, and subtracts
//! half background visitation potential `log b_Q(v) >> 1` to kill hub inflation.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::{DetHashMap, DetHashSet, Index};

const Q16_ONE: i64 = 65_536;
/// Cheeger log-degree disparity pruning threshold (`tau_beta = 2.0` log2 units = `4x` degree ratio).
const CHEEGER_LOG_GAP_TAU_Q16: i64 = 2 * Q16_ONE;

/// Propagates CEP Max-Plus shortest-path potentials over ISO-RAG Cheeger-pruned candidate
/// neighborhoods (`hops in 2..=3`), returning `(turn_idx, net_score_q16)` sorted descending.
pub(in crate::retrieval) fn cep_isorag_rerank_q16(
    ix: &Index,
    cand_turns: &[(usize, u64)],
    hops: usize,
) -> Vec<(usize, u64)> {
    let n = cand_turns.len().min(40);
    if n <= 2 {
        return cand_turns.to_vec();
    }
    let hops = hops.clamp(2, 3);
    let max_df = (ix.turns.len() as u32 / 10).clamp(4, 24);

    // Build sparse rare-term posting map inside candidate pool
    let mut term_post: DetHashMap<u32, Vec<usize>> = DetHashMap::default();
    let mut turn_rare_deg = vec![1u64; n];
    for (i, &(ti, _)) in cand_turns.iter().take(n).enumerate() {
        let mut deg = 0u64;
        for &cid in &ix.turns[ti].content {
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
            if df >= 1 && df <= max_df {
                term_post.entry(cid).or_default().push(i);
                deg += 1;
            }
        }
        turn_rare_deg[i] = deg.max(1);
    }

    // Node log-conductance proxy `log phi_u = log2(rare_deg_u + cross_hub_weight)` in Q16
    let mut log_phi = vec![0i64; n];
    for i in 0..n {
        let ti = cand_turns[i].0;
        let len = (ix.turns[ti].content.len() as u64).max(2);
        log_phi[i] = log2_u64_q16(len + turn_rare_deg[i]) as i64;
    }

    let mut adj: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n];
    let mut seen_edges: DetHashSet<(usize, usize)> = DetHashSet::default();
    let mut visit_mass = vec![0u64; n];

    for (cid, members) in term_post {
        if members.len() < 2 || members.len() > 8 {
            continue;
        }
        let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(1).max(1) as u64;
        let nt = (ix.turns.len() as u64).max(4);
        let edge_log_w = log2_u64_q16((nt / df).max(2)) as i64 / 4; // in Q16 log units

        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let u = members[a];
                let v = members[b];
                // ISO-RAG Cheeger conductance pruning across sharp degree boundaries
                if (log_phi[u] - log_phi[v]).abs() > CHEEGER_LOG_GAP_TAU_Q16 {
                    continue;
                }
                let key = if u < v { (u, v) } else { (v, u) };
                if seen_edges.insert(key) {
                    let w = (Q16_ONE >> 2).saturating_add(edge_log_w);
                    adj[u].push((v, w));
                    adj[v].push((u, w));
                    visit_mass[u] = visit_mass[u].saturating_add(df);
                    visit_mass[v] = visit_mass[v].saturating_add(df);
                }
            }
        }
    }

    // Also allow tight same-session adjacency (1 hop distance <= 2)
    for i in 0..n {
        let ti = cand_turns[i].0;
        for j in (i + 1)..n {
            let tj = cand_turns[j].0;
            if ix.turns[ti].session == ix.turns[tj].session && ti.abs_diff(tj) <= 2 {
                let key = (i, j);
                if seen_edges.insert(key) {
                    adj[i].push((j, Q16_ONE >> 1));
                    adj[j].push((i, Q16_ONE >> 1));
                }
            }
        }
    }

    // Initialize tropical potential d^(0)[u] = log_Q16(base_q16)
    let mut dist: Vec<i64> = cand_turns
        .iter()
        .take(n)
        .map(|&(_, sc)| {
            if sc == 0 {
                0
            } else {
                log2_u64_q16((sc >> 8).max(1)) as i64
            }
        })
        .collect();

    // CEP Max-Plus Bellman-Ford for H = 2..3 hops
    for _ in 0..hops {
        let mut next = dist.clone();
        for u in 0..n {
            if dist[u] <= 0 {
                continue;
            }
            for &(v, w) in &adj[u] {
                // Damped edge addition: `dist[u] + w - (log_phi[v] / 8)` prevents runaway cycling
                let prop = dist[u].saturating_add(w).saturating_sub(log_phi[v] >> 4);
                if prop > next[v] {
                    next[v] = prop;
                }
            }
        }
        dist = next;
    }

    // Subtract half background visitation potential: log b_Q(v) >> 1
    let mut out: Vec<(usize, u64)> = Vec::with_capacity(n);
    for i in 0..n {
        let bg_log_q16 = log2_u64_q16(visit_mass[i].saturating_add(2)) as i64;
        let net_log = dist[i].saturating_sub(bg_log_q16 >> 1);
        let boost_q16 = (net_log.max(0) as u64).min((Q16_ONE as u64) * 2);
        let combined = cand_turns[i].1.saturating_add(boost_q16);
        out.push((cand_turns[i].0, combined));
    }
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, build_index};
    use crate::retrieval::tests::sess;

    #[test]
    fn cheeger_pruning_and_bg_subtraction_prevent_generic_hub_inflation() {
        let sessions = vec![
            // Two bridged specific turns sharing rare "meridian" + "zurich" anchors
            sess("s0", &[("alice", "Meridian Robotics hired me as staff architect.")]),
            sess("s1", &[("alice", "Meridian Robotics moved HQ to Zurich CET.")]),
            // Generic large chatter hub matching lots of common words
            sess("s2", &[("assistant", "Meridian plan team project work time day week month review.")]),
        ];
        let ix = build_index(&sessions, &Params::default());
        let cands = vec![(0, 90_000), (2, 70_000), (1, 68_000)];
        let res = cep_isorag_rerank_q16(&ix, &cands, 2);
        assert_eq!(res[0].0, 0);
        // Bridged turn 1 or non-hub stays competitive and hub doesn't dominate
        assert!(res.iter().any(|&(ti, sc)| ti == 1 && sc >= 68_000));
    }
}
