//! Wave 17 (#727, #730, #731, #732, #734):
//! Shachter Bayes-Ball d-separation, Single-World Intervention Graph (SWIG)
//! node splitting, Markov blanket minimal shield, Rota Möbius inversion on
//! causal posets, and Alexandrov intervals + chain/antichain structure.

use std::collections::{BTreeSet, VecDeque};

/// #727: Shachter's exact Bayes-Ball d-separation test on a DAG (`0..n`).
/// Returns `true` iff `source` and `target` are d-separated given `conditioned`.
pub fn bayes_ball_d_separated(
    n: usize,
    edges: &[(usize, usize)], // directed u -> v
    source: usize,
    target: usize,
    conditioned: &BTreeSet<usize>,
) -> bool {
    let mut children = vec![Vec::new(); n];
    let mut parents = vec![Vec::new(); n];
    for &(u, v) in edges {
        children[u].push(v);
        parents[v].push(u);
    }

    // Precompute nodes that have a conditioned descendant (or are themselves conditioned)
    let mut has_cond_desc = vec![false; n];
    for &c in conditioned {
        let mut q = VecDeque::from([c]);
        while let Some(curr) = q.pop_front() {
            if !has_cond_desc[curr] {
                has_cond_desc[curr] = true;
                for &p in &parents[curr] {
                    q.push_back(p);
                }
            }
        }
    }

    // State: `(node, arrived_from_child)`
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();
    for &c in &children[source] {
        queue.push_back((c, false)); // arrived at `c` from parent `source`
    }
    for &p in &parents[source] {
        queue.push_back((p, true)); // arrived at `p` from child `source`
    }

    while let Some((node, from_child)) = queue.pop_front() {
        if !visited.insert((node, from_child)) {
            continue;
        }
        if node == target {
            return false; // active path exists -> NOT d-separated!
        }
        let is_cond = conditioned.contains(&node);
        if from_child {
            // Arrived from child (`node -> prev`): chain/fork upward & fork downward
            if !is_cond {
                for &p in &parents[node] {
                    queue.push_back((p, true));
                }
                for &c in &children[node] {
                    queue.push_back((c, false));
                }
            }
        } else {
            // Arrived from parent (`prev -> node`): chain downward if unconditioned;
            // collider `prev -> node <- other_p` activates if `has_cond_desc[node]`!
            if !is_cond {
                for &c in &children[node] {
                    queue.push_back((c, false));
                }
            }
            if has_cond_desc[node] {
                for &p in &parents[node] {
                    queue.push_back((p, true));
                }
            }
        }
    }
    true
}

/// #730: Single-World Intervention Graph (SWIG) node splitting on supersession/intervention `do(V = v)`.
/// Splits node `split_node` into:
/// - `split_node` (natural pre-intervention node receiving original incoming edges `u -> split_node`)
/// - `new_node_id = n` (post-intervention value node emitting original outgoing edges `new_node_id -> w`)
pub fn swig_node_split(
    n: usize,
    edges: &[(usize, usize)],
    split_node: usize,
) -> (usize, Vec<(usize, usize)>) {
    let intervened_id = n;
    let mut out = Vec::with_capacity(edges.len());
    for &(u, v) in edges {
        let src = if u == split_node { intervened_id } else { u };
        out.push((src, v));
    }
    (n + 1, out)
}

/// #731: Markov blanket minimal structural shield `Pa(X) U Ch(X) U CoPa(X)`.
pub fn markov_blanket(n: usize, edges: &[(usize, usize)], x: usize) -> BTreeSet<usize> {
    let _ = n;
    let mut mb = BTreeSet::new();
    let mut children = BTreeSet::new();
    for &(u, v) in edges {
        if v == x {
            mb.insert(u);
        }
        if u == x {
            mb.insert(v);
            children.insert(v);
        }
    }
    for &(u, v) in edges {
        if children.contains(&v) && u != x {
            mb.insert(u);
        }
    }
    mb
}

/// #732: Rota Möbius inversion `mu(x, y)` on a topologically ordered causal poset `0..n`
/// where `leq[x][y] == true` iff `x <= y` in the causal order.
pub fn rota_moebius_matrix(leq: &[Vec<bool>]) -> Vec<Vec<i64>> {
    let n = leq.len();
    let mut mu = vec![vec![0i64; n]; n];
    for (i, row) in mu.iter_mut().enumerate() {
        row[i] = 1;
    }
    for length in 1..n {
        for x in 0..(n - length) {
            let y = x + length;
            if leq[x][y] {
                let mut s = 0i64;
                for z in x..y {
                    if leq[x][z] && leq[z][y] {
                        s += mu[x][z];
                    }
                }
                mu[x][y] = -s;
            }
        }
    }
    mu
}

/// #734: Alexandrov causal interval `I(x, y) = { z | x <= z <= y }` & longest chain depth.
pub fn alexandrov_interval(leq: &[Vec<bool>], x: usize, y: usize) -> Vec<usize> {
    (0..leq.len())
        .filter(|&z| leq[x][z] && leq[z][y])
        .collect()
}

pub fn longest_causal_chain_len(n: usize, dag_edges: &[(usize, usize)]) -> usize {
    if n == 0 {
        return 0;
    }
    let mut dp = vec![1usize; n];
    let mut changed = true;
    while changed {
        changed = false;
        for &(u, v) in dag_edges {
            if dp[u] + 1 > dp[v] {
                dp[v] = dp[u] + 1;
                changed = true;
            }
        }
    }
    dp.into_iter().max().unwrap_or(0)
}
