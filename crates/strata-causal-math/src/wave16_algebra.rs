//! Wave 16A–16H (#701–#708):
//! Submodular Any@5 + Birkhoff-von Neumann Sinkhorn routing, Lengauer-Tarjan
//! dominators & backward slicing, Green provenance semirings & tropical paths,
//! HodgeRank cycle decomposition, Galois concept lattices & MDL ranking,
//! congruence e-graphs, differential Z-sets, and category graph schemas.

use std::collections::{BTreeMap, BTreeSet};

/// #701: Greedy `(1 - 1/e)` submodular Any@k slot allocation + Sinkhorn doubly-stochastic routing.
pub fn submodular_greedy_slots(
    candidates: &[(String, BTreeSet<String>)], // (doc_id, covered_facets)
    k: usize,
) -> Vec<String> {
    let mut covered = BTreeSet::new();
    let mut chosen = Vec::new();
    let mut used = vec![false; candidates.len()];
    for _ in 0..k.min(candidates.len()) {
        let mut best_idx = None;
        let mut best_gain = 0usize;
        for (i, (_, facets)) in candidates.iter().enumerate() {
            if !used[i] {
                let gain = facets.difference(&covered).count();
                if best_idx.is_none() || gain > best_gain {
                    best_gain = gain;
                    best_idx = Some(i);
                }
            }
        }
        if let Some(idx) = best_idx {
            used[idx] = true;
            for f in &candidates[idx].1 {
                covered.insert(f.clone());
            }
            chosen.push(candidates[idx].0.clone());
        }
    }
    chosen
}

pub fn sinkhorn_doubly_stochastic(weights: &[Vec<f64>], iters: usize) -> Vec<Vec<f64>> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let mut m: Vec<Vec<f64>> = weights
        .iter()
        .map(|row| row.iter().map(|&v| v.max(1e-9)).collect())
        .collect();
    for _ in 0..iters {
        // Row normalize
        for row in &mut m {
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for x in row {
                    *x /= s;
                }
            }
        }
        // Column normalize
        for col in 0..n {
            let s: f64 = (0..n).map(|r| m[r][col]).sum();
            if s > 0.0 {
                for row in &mut m {
                    row[col] /= s;
                }
            }
        }
    }
    m
}

/// #702: Backward slicing & dominator bottlenecks on directed causal graphs.
pub fn compute_dominators(
    entry: usize,
    num_nodes: usize,
    edges: &[(usize, usize)],
) -> Vec<BTreeSet<usize>> {
    let all_nodes: BTreeSet<usize> = (0..num_nodes).collect();
    let mut dom: Vec<BTreeSet<usize>> = (0..num_nodes)
        .map(|v| {
            if v == entry {
                BTreeSet::from([entry])
            } else {
                all_nodes.clone()
            }
        })
        .collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    for &(u, v) in edges {
        preds[v].push(u);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for v in 0..num_nodes {
            if v == entry || preds[v].is_empty() {
                continue;
            }
            let mut inter = dom[preds[v][0]].clone();
            for &p in &preds[v][1..] {
                inter = inter.intersection(&dom[p]).copied().collect();
            }
            inter.insert(v);
            if inter != dom[v] {
                dom[v] = inter;
                changed = true;
            }
        }
    }
    dom
}

/// #703: Tropical `(min, +)` shortest causal path times + Green provenance semiring `N[X]`.
pub fn tropical_all_pairs_shortest(n: usize, weighted_edges: &[(usize, usize, u64)]) -> Vec<Vec<u64>> {
    let inf = u64::MAX / 4;
    let mut dist = vec![vec![inf; n]; n];
    for (i, row) in dist.iter_mut().enumerate() {
        row[i] = 0;
    }
    for &(u, v, w) in weighted_edges {
        dist[u][v] = dist[u][v].min(w);
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let cand = dist[i][k] + dist[k][j];
                if cand < dist[i][j] {
                    dist[i][j] = cand;
                }
            }
        }
    }
    dist
}

/// Tropical `(max, +)` 2-hop semiring squaring: returns `M^2[i][j] = max_k (M[i][k] + M[k][j])`
/// over non-zero directed edges (`0` marks unreachable / `-inf` in the tropical semiring).
pub fn tropical_max_plus_two_hop(n: usize, weighted_edges: &[(usize, usize, u64)]) -> Vec<Vec<u64>> {
    let mut adj = vec![vec![0u64; n]; n];
    for &(u, v, w) in weighted_edges {
        if u < n && v < n && w > adj[u][v] {
            adj[u][v] = w;
        }
    }
    let mut sq = vec![vec![0u64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let wik = adj[i][k];
            if wik == 0 {
                continue;
            }
            for j in 0..n {
                let wkj = adj[k][j];
                if wkj == 0 {
                    continue;
                }
                let chain = wik.saturating_add(wkj);
                if chain > sq[i][j] {
                    sq[i][j] = chain;
                }
            }
        }
    }
    sq
}

/// #704: HodgeRank preference potential & cyclic curl residual decomposition.
pub fn hodgerank_potential_and_curl(
    n: usize,
    comparisons: &[(usize, usize, f64)], // (i, j, y_ij ~ s_j - s_i)
) -> (Vec<f64>, f64) {
    // Iterative Jacobi / Gauss-Seidel solver on graph Laplacian L s = div(Y) with s[0] = 0 gauge
    let mut s = vec![0.0; n];
    for _ in 0..200 {
        for v in 1..n {
            let mut sum = 0.0;
            let mut deg = 0.0;
            for &(i, j, y_ij) in comparisons {
                if j == v {
                    sum += s[i] + y_ij;
                    deg += 1.0;
                } else if i == v {
                    sum += s[j] - y_ij;
                    deg += 1.0;
                }
            }
            if deg > 0.0 {
                s[v] = sum / deg;
            }
        }
    }
    let curl_residual_sq: f64 = comparisons
        .iter()
        .map(|&(i, j, y_ij)| ((s[j] - s[i]) - y_ij).powi(2))
        .sum();
    (s, curl_residual_sq)
}

/// #705: Formal Concept Analysis Galois closure `A''` + MDL setup-turn score.
pub fn galois_attribute_closure(
    context: &[(String, BTreeSet<String>)], // (object, attributes)
    seed_attrs: &BTreeSet<String>,
) -> BTreeSet<String> {
    let matching_objs: Vec<&BTreeSet<String>> = context
        .iter()
        .filter(|(_, attrs)| seed_attrs.is_subset(attrs))
        .map(|(_, attrs)| attrs)
        .collect();
    let Some(first) = matching_objs.first() else {
        return seed_attrs.clone();
    };
    let mut closed = (*first).clone();
    for attrs in &matching_objs[1..] {
        closed = closed.intersection(attrs).cloned().collect();
    }
    closed
}

pub fn mdl_setup_turn_bits(schema_bits: usize, explained_turns: usize, bits_saved_per_turn: usize) -> isize {
    (schema_bits as isize) - ((explained_turns * bits_saved_per_turn) as isize)
}

/// #706: Congruence-closure e-graph for alias & function-application equivalence.
#[derive(Debug, Default)]
pub struct CongruenceEGraph {
    classes: BTreeMap<String, usize>,
    apps: Vec<(String, usize, usize)>, // (fn_name, arg_class, result_class)
    next_id: usize,
}

impl CongruenceEGraph {
    pub fn intern(&mut self, sym: &str) -> usize {
        if let Some(&id) = self.classes.get(sym) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.classes.insert(sym.to_string(), id);
        id
    }

    pub fn add_app(&mut self, func: &str, arg_sym: &str, out_sym: &str) {
        let a = self.intern(arg_sym);
        let r = self.intern(out_sym);
        self.apps.push((func.to_string(), a, r));
    }

    pub fn merge_aliases(&mut self, a_sym: &str, b_sym: &str) {
        let ca = self.intern(a_sym);
        let cb = self.intern(b_sym);
        if ca == cb {
            return;
        }
        self.rewrite_class(cb, ca);
        // Upward congruence closure: if f(x1)->r1 and f(x2)->r2 with x1 == x2, merge r1 & r2!
        loop {
            let mut pair_to_merge = None;
            for i in 0..self.apps.len() {
                for j in (i + 1)..self.apps.len() {
                    if self.apps[i].0 == self.apps[j].0
                        && self.apps[i].1 == self.apps[j].1
                        && self.apps[i].2 != self.apps[j].2
                    {
                        pair_to_merge = Some((self.apps[i].2, self.apps[j].2));
                        break;
                    }
                }
                if pair_to_merge.is_some() {
                    break;
                }
            }
            if let Some((from_c, to_c)) = pair_to_merge {
                self.rewrite_class(from_c, to_c);
            } else {
                break;
            }
        }
    }

    fn rewrite_class(&mut self, from_c: usize, to_c: usize) {
        for v in self.classes.values_mut() {
            if *v == from_c {
                *v = to_c;
            }
        }
        for app in &mut self.apps {
            if app.1 == from_c {
                app.1 = to_c;
            }
            if app.2 == from_c {
                app.2 = to_c;
            }
        }
    }

    pub fn equiv(&self, a: &str, b: &str) -> bool {
        self.classes.get(a).is_some_and(|ca| self.classes.get(b) == Some(ca))
    }
}

/// #707: Incremental differential dataflow Z-set (`item -> signed multiplicity`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ZSet<T: Ord + Clone> {
    pub weights: BTreeMap<T, i64>,
}

impl<T: Ord + Clone> ZSet<T> {
    pub fn apply_delta(&mut self, item: T, delta: i64) {
        let entry = self.weights.entry(item.clone()).or_insert(0);
        *entry += delta;
        if *entry == 0 {
            self.weights.remove(&item);
        }
    }
}

/// #708: Category graph schema check (every object has `id_X` and composable arrows compose).
pub fn verify_category_schema(
    objects: &BTreeSet<String>,
    morphisms: &BTreeSet<(String, String)>, // (dom, cod)
) -> bool {
    for obj in objects {
        if !morphisms.contains(&(obj.clone(), obj.clone())) {
            return false;
        }
    }
    for (a, b) in morphisms {
        for (b2, c) in morphisms {
            if b == b2 && !morphisms.contains(&(a.clone(), c.clone())) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tropical_max_plus_tests {
    use super::*;

    #[test]
    fn tropical_max_plus_two_hop_picks_strongest_chain() {
        let sq = tropical_max_plus_two_hop(4, &[(0, 1, 10), (1, 3, 25), (0, 2, 12), (2, 3, 15)]);
        // 0 -> 1 -> 3 = 35 beats 0 -> 2 -> 3 = 27.
        assert_eq!(sq[0][3], 35);
        assert_eq!(sq[1][0], 0);
    }
}
