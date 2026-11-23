//! Prize-Collecting Group Steiner Tree Subgraph Bridge (Q16.16 integer).
//!
//! Inspired by G-Retriever PCST (`2402.07630`) & Group Steiner Tree Retrieval (`2508.20543`).
//! Given an undirected weighted session graph `adj` and per-session query group coverage masks
//! `s_mask`, connects the highest-prize root session to at least one terminal session per
//! uncovered query group `g` via minimum-cost integer Dijkstra paths, lifting both the connected
//! terminals and any intermediate Steiner bridge sessions where `prize > path_cost`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

const Q16_ONE: u64 = 65_536;

/// Compute Q16.16 Steiner bridge bonuses for sessions forming the minimum-cost Group Steiner
/// subgraph connecting distinct query-group terminals `T_g`.
pub(in crate::retrieval) fn compute_group_steiner_lifts_q16(
    ns: usize,
    adj: &[Vec<(usize, u64)>],
    base_sight_q16: &[u64],
    s_mask: &[u64],
    mask_limit: u64,
    root: usize,
    cap_q16: u64,
) -> Vec<u64> {
    let mut lifts = vec![0u64; ns];
    if ns < 3 || root >= ns || cap_q16 == 0 || mask_limit == 0 {
        return lifts;
    }

    // Determine which query groups actually exist in the corpus and aren't yet covered by `root`.
    let mut union_mask = 0u64;
    for &m in s_mask.iter().take(ns) {
        union_mask |= m & mask_limit;
    }
    let mut covered_mask = s_mask.get(root).copied().unwrap_or(0) & mask_limit;
    let mut uncovered = union_mask & !covered_mask;
    if uncovered == 0 {
        return lifts;
    }

    let mut tree_nodes = vec![false; ns];
    tree_nodes[root] = true;
    let mut steiner_votes = vec![0u64; ns];

    // Iteratively connect the nearest uncovered terminal group via multi-source Dijkstra from `tree_nodes`.
    let max_rounds = (uncovered.count_ones() as usize).min(6);
    for _ in 0..max_rounds {
        if uncovered == 0 {
            break;
        }
        let mut dist = vec![u64::MAX; ns];
        let mut parent = vec![usize::MAX; ns];
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        for (s, &in_tree) in tree_nodes.iter().enumerate() {
            if in_tree {
                dist[s] = 0;
                heap.push(Reverse((0, s)));
            }
        }

        let mut best_target: Option<(usize, u64)> = None;
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u] {
                continue;
            }
            if !tree_nodes[u] && ((s_mask[u] & uncovered) != 0) {
                let prize = base_sight_q16[u].max(Q16_ONE >> 2);
                // Prize-collecting score: accept shortest path whose cost is bounded relative to prize.
                if best_target.is_none_or(|(_, bd)| d < bd) && d <= prize.saturating_mul(4) {
                    best_target = Some((u, d));
                }
                // First hit from Dijkstra is minimum distance; stop expanding past 2x its distance.
                if let Some((_, bd)) = best_target
                    && d > bd.saturating_mul(2)
                {
                    break;
                }
            }
            for &(v, w) in &adj[u] {
                if w == 0 {
                    continue;
                }
                // Integer edge traversal cost inversely proportional to DWPR edge strength.
                let edge_cost = (Q16_ONE.saturating_mul(Q16_ONE)) / (Q16_ONE.saturating_add(w));
                let nd = d.saturating_add(edge_cost.max(256));
                if nd < dist[v] {
                    dist[v] = nd;
                    parent[v] = u;
                    heap.push(Reverse((nd, v)));
                }
            }
        }

        let Some((target, path_cost)) = best_target else {
            break;
        };
        let mut cur = target;
        let mut hops = 0usize;
        while cur < ns && !tree_nodes[cur] && hops <= 4 {
            tree_nodes[cur] = true;
            let weight = if cur == target {
                Q16_ONE
            } else {
                // Intermediate Steiner bridge node connecting disjoint query groups!
                (Q16_ONE * 3) / 4
            };
            let bonus = weight.saturating_mul(Q16_ONE) / (Q16_ONE.saturating_add(path_cost >> 1));
            steiner_votes[cur] = steiner_votes[cur].saturating_add(bonus);
            covered_mask |= s_mask[cur] & mask_limit;
            let p = parent[cur];
            if p == usize::MAX {
                break;
            }
            cur = p;
            hops += 1;
        }
        uncovered = union_mask & !covered_mask;
    }

    let max_vote = steiner_votes
        .iter()
        .enumerate()
        .filter(|&(s, _)| s != root)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_vote == 0 {
        return lifts;
    }
    for s in 0..ns {
        if s != root && steiner_votes[s] > 0 {
            lifts[s] = ((cap_q16 as u128)
                .saturating_mul(steiner_votes[s] as u128)
                / (max_vote as u128)) as u64;
        }
    }
    lifts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_steiner_connects_bridge_between_disjoint_query_groups() {
        // Session 0 covers group 0, session 2 covers group 1, session 1 is the Steiner bridge.
        let mut adj = vec![Vec::new(); 4];
        adj[0].push((1, Q16_ONE * 2));
        adj[1].push((0, Q16_ONE * 2));
        adj[1].push((2, Q16_ONE * 2));
        adj[2].push((1, Q16_ONE * 2));
        let base_sight = vec![Q16_ONE * 2, Q16_ONE / 4, Q16_ONE, 0];
        let s_mask = vec![0b01, 0b00, 0b10, 0b00];
        let lifts = compute_group_steiner_lifts_q16(4, &adj, &base_sight, &s_mask, 0b11, 0, Q16_ONE);
        assert!(lifts[1] > 0, "intermediate Steiner bridge must be lifted");
        assert!(lifts[2] > 0, "target terminal group session must be lifted");
        assert_eq!(lifts[3], 0, "disconnected session must remain zero");
    }
}
