//! Intelligence Link Analysis & Contact Tracing (`Q16.16`, Freeman 1977; Krebs 2002).
//!
//! Computes exact integer Brandes/Freeman betweenness centrality over shortest paths
//! (`dist <= 3`) connecting terminal sessions that match distinct query groups, surfacing
//! low-degree intermediate broker sessions that bridge separated evidence clusters.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::{BTreeMap, VecDeque};

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const MAX_BROKER_DIST: usize = 3;
const ENT_MIN_DF: u32 = 2;
const ENT_MAX_DF: u32 = 8;
const BROKER_COEFF_Q16: u64 = 16_384; // 0.25

pub(in crate::retrieval) fn query_terminal_betweenness_q16(
    ns: usize,
    adj: &[Vec<usize>],
    sources: &[usize],
    targets: &[usize],
) -> Vec<u64> {
    let mut betw = vec![0u64; ns];
    for &src in sources {
        let mut dist = vec![usize::MAX; ns];
        let mut sigma = vec![0u64; ns];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); ns];
        let mut order: Vec<usize> = Vec::new();
        let mut q = VecDeque::new();

        dist[src] = 0;
        sigma[src] = 1;
        q.push_back(src);

        while let Some(u) = q.pop_front() {
            order.push(u);
            if dist[u] >= MAX_BROKER_DIST {
                continue;
            }
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
                if dist[v] == dist[u] + 1 {
                    sigma[v] = sigma[v].saturating_add(sigma[u]);
                    preds[v].push(u);
                }
            }
        }

        let mut delta_q16 = vec![0u64; ns];
        for &t in targets {
            if t != src && (2..=MAX_BROKER_DIST).contains(&dist[t]) {
                delta_q16[t] = delta_q16[t].saturating_add(Q16_ONE);
            }
        }
        for &w in order.iter().rev() {
            if delta_q16[w] == 0 || sigma[w] == 0 {
                continue;
            }
            for &p in &preds[w] {
                let share = (sigma[p].saturating_mul(delta_q16[w])) / sigma[w];
                if p != src {
                    betw[p] = betw[p].saturating_add(share);
                    delta_q16[p] = delta_q16[p].saturating_add(share);
                }
            }
        }
    }
    betw
}

pub(in crate::retrieval) fn compute_freeman_brokerage_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    s_ord_base: &[usize],
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let Some(&top0) = s_ord_base.first() else {
        return out;
    };
    let n_groups = plan.groups.len().min(64);
    if ns <= 5 || top0 >= ns || n_groups < 2 || smax_q16 == 0 {
        return out;
    }

    let mut ent_post: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (si, ents) in ix.session_entities.iter().enumerate() {
        for &eid in ents {
            ent_post.entry(eid).or_default().push(si);
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ns];
    let mut add_u = |a: usize, b: usize| {
        if a != b && !adj[a].contains(&b) {
            adj[a].push(b);
            adj[b].push(a);
        }
    };
    for sessions in ent_post.values() {
        let df = sessions.len() as u32;
        if (ENT_MIN_DF..=ENT_MAX_DF).contains(&df) {
            for (i, &sa) in sessions.iter().enumerate() {
                for &sb in &sessions[(i + 1)..] {
                    add_u(sa, sb);
                }
            }
        }
    }
    for si in 0..ns {
        for &(_, prev) in &ix.session_rsb_back[si] {
            if prev < ns {
                add_u(si, prev);
            }
        }
    }

    let sources: Vec<usize> = s_ord_base
        .iter()
        .copied()
        .filter(|&si| si < ns && (s_exact_mask[si] & 1u64) != 0)
        .take(3)
        .collect();
    let other_mask = if n_groups == 64 {
        u64::MAX ^ 1u64
    } else {
        ((1u64 << n_groups) - 1) ^ 1u64
    };
    let targets: Vec<usize> = s_ord_base
        .iter()
        .copied()
        .filter(|&si| si < ns && (s_exact_mask[si] & other_mask) != 0 && !sources.contains(&si))
        .take(4)
        .collect();
    if sources.is_empty() || targets.is_empty() {
        return out;
    }

    let betw = query_terminal_betweenness_q16(ns, &adj, &sources, &targets);
    let max_b = betw
        .iter()
        .enumerate()
        .filter(|&(si, _)| si != top0)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_b == 0 {
        return out;
    }

    for (si, &b) in betw.iter().enumerate() {
        if si != top0 && b > 0 {
            let norm_q16 = (b.saturating_mul(Q16_ONE)) / max_b;
            out[si] = (((smax_q16.saturating_mul(BROKER_COEFF_Q16)) >> Q16_SHIFT)
                .saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freeman_betweenness_isolates_shortest_path_broker() {
        let mut adj = vec![Vec::new(); 5];
        adj[0].push(2);
        adj[2].push(0);
        adj[2].push(4);
        adj[4].push(2);
        let betw = query_terminal_betweenness_q16(5, &adj, &[0], &[4]);
        assert_eq!(betw[2], Q16_ONE);
        assert_eq!(betw[1], 0);
    }
}
