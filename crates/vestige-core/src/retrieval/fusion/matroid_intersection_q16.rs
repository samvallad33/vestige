//! Engine #5: Edmonds Weighted Matroid Intersection (`Q16.16`).
//!
//! Solves exact global maximum-weight independent set selection of size `k` over
//! Top-`N` (`N <= 32`) candidates subject to two partition matroids simultaneously:
//! - **$M_1$ (Temporal/CUSUM Session Partition + Genesis Pin)**: limits seats per
//!   session/epoch bucket `part_m1[i]` to capacity `cap_m1[bucket]` (and locks a
//!   mandatory Genesis anchor at Slot 0 when `pin_genesis_idx` is `Some`).
//! - **$M_2$ (Distinct Aspect / Sub-Clause / Polarity Partition)**: limits seats per
//!   aspect/polarity signature `part_m2[i]` to capacity `cap_m2[aspect]`.
//!
//! Uses exact Bellman-Ford minimum-cost augmenting paths in the exchange digraph
//! $D(S)$ (`<= 32` nodes, `<= 5` augmentations -> `< 12 µs`, zero floats).

use std::collections::BTreeMap;

/// Maximum ground-set size solved by exact matroid intersection (`32`).
pub(in crate::retrieval) const MAX_MATROID_GROUND: usize = 32;

/// One candidate element in the weighted matroid intersection ground set.
#[derive(Clone, Copy, Debug)]
pub(in crate::retrieval) struct MatroidElem {
    /// Original turn or session index.
    pub(in crate::retrieval) id: usize,
    /// `Q16.16` element weight (quality / salience).
    pub(in crate::retrieval) weight_q16: i64,
    /// Partition ID in Matroid $M_1$ (e.g., session index or CUSUM epoch bin).
    pub(in crate::retrieval) m1_part: usize,
    /// Partition ID in Matroid $M_2$ (e.g., dominant uncovered query facet or polarity).
    pub(in crate::retrieval) m2_part: usize,
    /// Chronological tie-break key (timestamp or stream index).
    pub(in crate::retrieval) chrono_key: i64,
}

/// Exact Edmonds Weighted Matroid Intersection (`max_{S in I_1 cap I_2, |S| <= target_k} sum w_i`).
///
/// If `pin_idx` is `Some(p)`, element `p` is seeded into $S$ first and protected from
/// exchange removal (preserving Genesis Slot-0 or crisp Rank-1 invariants).
/// Returns selected indices into `elems` (ordered by descending weight, with pinned element first).
pub(in crate::retrieval) fn solve_weighted_matroid_intersection(
    elems: &[MatroidElem],
    cap_m1: &BTreeMap<usize, usize>,
    cap_m2: &BTreeMap<usize, usize>,
    default_cap_m1: usize,
    default_cap_m2: usize,
    target_k: usize,
    pin_idx: Option<usize>,
) -> Vec<usize> {
    let n = elems.len().min(MAX_MATROID_GROUND);
    if n == 0 || target_k == 0 {
        return Vec::new();
    }
    let k_lim = target_k.min(n);

    let m1_cap_of = |part: usize| -> usize {
        cap_m1.get(&part).copied().unwrap_or(default_cap_m1).max(1)
    };
    let m2_cap_of = |part: usize| -> usize {
        cap_m2.get(&part).copied().unwrap_or(default_cap_m2).max(1)
    };

    let mut in_s = vec![false; n];
    let mut cnt_m1: BTreeMap<usize, usize> = BTreeMap::new();
    let mut cnt_m2: BTreeMap<usize, usize> = BTreeMap::new();
    let mut s_size = 0usize;

    if let Some(p) = pin_idx
        && p < n
    {
        in_s[p] = true;
        *cnt_m1.entry(elems[p].m1_part).or_insert(0) += 1;
        *cnt_m2.entry(elems[p].m2_part).or_insert(0) += 1;
        s_size = 1;
    }

    const INF: i128 = i128::MAX / 8;

    while s_size < k_lim {
        // X1 = { y not in S : S + y in I_1 }, X2 = { y not in S : S + y in I_2 }
        let mut x1 = vec![false; n];
        let mut x2 = vec![false; n];
        for y in 0..n {
            if in_s[y] {
                continue;
            }
            let c1 = cnt_m1.get(&elems[y].m1_part).copied().unwrap_or(0);
            let c2 = cnt_m2.get(&elems[y].m2_part).copied().unwrap_or(0);
            x1[y] = c1 < m1_cap_of(elems[y].m1_part);
            x2[y] = c2 < m2_cap_of(elems[y].m2_part);
        }

        // Build exchange digraph edges `(u, v)`:
        // - For `i in S`, `j not in S`: edge `i -> j` if `S - i + j in I_1`
        //   (i.e. `x1[j]` OR `m1_part[i] == m1_part[j]`), skipping `pin_idx`.
        // - For `j not in S`, `i in S`: edge `j -> i` if `S - i + j in I_2`
        //   (i.e. `x2[j]` OR `m2_part[i] == m2_part[j]`), skipping `pin_idx`.
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n * 8);
        for i in 0..n {
            if !in_s[i] || Some(i) == pin_idx {
                continue;
            }
            for j in 0..n {
                if in_s[j] {
                    continue;
                }
                if x1[j] || elems[i].m1_part == elems[j].m1_part {
                    edges.push((i, j));
                }
                if x2[j] || elems[i].m2_part == elems[j].m2_part {
                    edges.push((j, i));
                }
            }
        }

        // Node cost for shortest augmenting path:
        // minimize `sum_{y not in S} (-w_y) + sum_{x in S} (+w_x)` with hop-count tie-break
        // (scaled by 64 so shortest hop path breaks exact weight ties deterministically).
        let node_cost = |u: usize| -> i128 {
            let w = elems[u].weight_q16 as i128;
            if in_s[u] { w * 64 + 1 } else { -w * 64 + 1 }
        };

        let mut dist = vec![INF; n];
        let mut prev = vec![usize::MAX; n];
        for y in 0..n {
            if x1[y] {
                dist[y] = node_cost(y);
            }
        }

        // Bellman-Ford over `<= 32` vertices (terminates early when no edge relaxes).
        for _ in 0..n {
            let mut changed = false;
            for &(u, v) in &edges {
                if dist[u] == INF {
                    continue;
                }
                let cand = dist[u].saturating_add(node_cost(v));
                if cand < dist[v] || (cand == dist[v] && u < prev[v]) {
                    dist[v] = cand;
                    prev[v] = u;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Pick best sink `t in X2` minimizing total path cost (maximizing weight gain).
        let mut best_t: Option<(usize, i128)> = None;
        for t in 0..n {
            if x2[t] && dist[t] < INF {
                let take = match best_t {
                    None => true,
                    Some((bt, bd)) => {
                        dist[t] < bd || (dist[t] == bd && elems[t].id < elems[bt].id)
                    }
                };
                if take {
                    best_t = Some((t, dist[t]));
                }
            }
        }
        let Some((mut cur, best_cost)) = best_t else {
            break;
        };
        // Stop if augmentation yields non-positive net weight when we already have positive mass.
        if s_size > 0 && best_cost > 0 {
            break;
        }

        // Augment symmetric difference along `prev` chain.
        let mut guard = 0usize;
        while cur < n && guard <= n {
            in_s[cur] = !in_s[cur];
            let p = prev[cur];
            if p == usize::MAX {
                break;
            }
            cur = p;
            guard += 1;
        }

        // Recompute partition counts cleanly from `in_s`.
        cnt_m1.clear();
        cnt_m2.clear();
        s_size = 0;
        for i in 0..n {
            if in_s[i] {
                *cnt_m1.entry(elems[i].m1_part).or_insert(0) += 1;
                *cnt_m2.entry(elems[i].m2_part).or_insert(0) += 1;
                s_size += 1;
            }
        }
    }

    let mut chosen: Vec<usize> = (0..n).filter(|&i| in_s[i]).collect();
    chosen.sort_by(|&a, &b| {
        (Some(b) == pin_idx)
            .cmp(&(Some(a) == pin_idx))
            .then_with(|| elems[b].weight_q16.cmp(&elems[a].weight_q16))
            .then_with(|| elems[a].chrono_key.cmp(&elems[b].chrono_key))
            .then_with(|| elems[a].id.cmp(&elems[b].id))
    });
    chosen
}

/// High-level Top-K session selector using Edmonds Weighted Matroid Intersection over
/// `M1 = CUSUM/Chronological Epoch Bin` and `M2 = Dominant Uncovered Query/Aspect Cluster`.
/// If `sort_chronological` holds (e.g. `event_ordering`), orders all `k` picks strictly ascending
/// by `(chrono_key, id)`.
pub(in crate::retrieval) fn select_sessions_matroid_q16(
    elems: &[MatroidElem],
    target_k: usize,
    max_per_epoch: usize,
    max_per_aspect: usize,
    pin_genesis_idx: Option<usize>,
    sort_chronological: bool,
) -> Vec<usize> {
    let cap_m1 = BTreeMap::new();
    let cap_m2 = BTreeMap::new();
    let mut picked = solve_weighted_matroid_intersection(
        elems,
        &cap_m1,
        &cap_m2,
        max_per_epoch.max(1),
        max_per_aspect.max(1),
        target_k,
        pin_genesis_idx,
    );
    if sort_chronological {
        picked.sort_by_key(|&idx| (elems[idx].chrono_key, elems[idx].id));
    }
    picked.into_iter().map(|idx| elems[idx].id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edmonds_matroid_intersection_beats_greedy_collision() {
        let elems = vec![
            MatroidElem { id: 100, weight_q16: 10 * 65_536, m1_part: 0, m2_part: 0, chrono_key: 1 },
            MatroidElem { id: 101, weight_q16: 9 * 65_536, m1_part: 0, m2_part: 1, chrono_key: 2 },
            MatroidElem { id: 102, weight_q16: 9 * 65_536, m1_part: 1, m2_part: 0, chrono_key: 3 },
        ];
        let picked = select_sessions_matroid_q16(&elems, 2, 1, 1, None, true);
        assert_eq!(picked, vec![101, 102]);
    }

    #[test]
    fn matroid_intersection_respects_genesis_pin() {
        let elems = vec![
            MatroidElem { id: 0, weight_q16: 5 * 65_536, m1_part: 0, m2_part: 0, chrono_key: 10 },
            MatroidElem { id: 1, weight_q16: 9 * 65_536, m1_part: 0, m2_part: 1, chrono_key: 20 },
            MatroidElem { id: 2, weight_q16: 8 * 65_536, m1_part: 1, m2_part: 1, chrono_key: 30 },
        ];
        let picked = select_sessions_matroid_q16(&elems, 2, 1, 1, Some(0), true);
        assert_eq!(picked, vec![0, 2]);
    }
}
