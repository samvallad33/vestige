//! Kemeny-Young exact Condorcet rank aggregation and Copeland fallback.

use crate::retrieval::*;


/// Exact Kemeny-Young inside one Condorcet component. 8! = 40320 is the
/// largest search this branch-and-bound solves exactly. Larger components
/// use Copeland plus adjacent swaps. Fixed from the size of the search,
/// before any card.
pub(in crate::retrieval) const KEMENY_EXACT_CAP: usize = 8;

/// In-degree inside the pool from edges the index already stored, plus the
/// RSB antecedent. One pass over the log, not one pass per candidate.
pub(in crate::retrieval) fn pool_structural_degree(ix: &Index, pool: &[usize]) -> Vec<i64> {
    let mut at = vec![None; ix.turns.len()];
    for (i, &ti) in pool.iter().enumerate() {
        if let Some(slot) = at.get_mut(ti) {
            *slot = Some(i);
        }
    }
    let mut degree = vec![0i64; pool.len()];
    for (src, t) in ix.turns.iter().enumerate() {
        let mut hit = |target: usize| {
            if target == src {
                return;
            }
            if let Some(Some(slot)) = at.get(target) {
                degree[*slot] += 1;
            }
        };
        for &target in &t.policy_backlinks {
            hit(target);
        }
        for &target in &t.supersedes_turns {
            hit(target);
        }
        for &(_, target) in &t.exact_edges {
            hit(target);
        }
        if let Some(target) = rsb_antecedent(ix, src) {
            hit(target);
        }
    }
    degree
}

pub(in crate::retrieval) fn kemeny_pair_score(order: &[usize], net: &[Vec<i64>]) -> i64 {
    let mut score = 0i64;
    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            score += net[order[i]][order[j]];
        }
    }
    score
}

pub(in crate::retrieval) fn kemeny_bound(prefix: &[usize], used: &[bool], net: &[Vec<i64>]) -> i64 {
    let m = used.len();
    let mut score = kemeny_pair_score(prefix, net);
    for &p in prefix {
        for c in 0..m {
            if !used[c] {
                score += net[p][c];
            }
        }
    }
    let rest: Vec<usize> = (0..m).filter(|&c| !used[c]).collect();
    for i in 0..rest.len() {
        for j in (i + 1)..rest.len() {
            score += net[rest[i]][rest[j]].abs();
        }
    }
    score
}

/// Exact Kemeny order of a component. Equal scores keep the RSB order,
/// which is the initial incumbent, so a tie never falls through to turn id.
pub(in crate::retrieval) fn kemeny_branch_and_bound(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let m = members.len();
    let mut local = vec![vec![0i64; m]; m];
    for i in 0..m {
        for j in 0..m {
            local[i][j] = net[members[i]][members[j]];
        }
    }
    let mut start: Vec<usize> = (0..m).collect();
    start.sort_by(|&a, &b| rsb[members[a]].cmp(&rsb[members[b]]));
    let key_of = |order: &[usize]| -> Vec<usize> {
        order.iter().map(|&i| rsb[members[i]]).collect()
    };
    let mut best_score = kemeny_pair_score(&start, &local);
    let mut best = start;
    let mut best_key = key_of(&best);
    let mut used = vec![false; m];
    let mut prefix: Vec<usize> = Vec::with_capacity(m);
    fn rec(
        local: &[Vec<i64>],
        members: &[usize],
        rsb: &[usize],
        used: &mut [bool],
        prefix: &mut Vec<usize>,
        best_score: &mut i64,
        best: &mut Vec<usize>,
        best_key: &mut Vec<usize>,
    ) {
        let m = used.len();
        if prefix.len() == m {
            let score = kemeny_pair_score(prefix, local);
            let key: Vec<usize> = prefix.iter().map(|&i| rsb[members[i]]).collect();
            if score > *best_score || (score == *best_score && key < *best_key) {
                *best_score = score;
                *best_key = key;
                *best = prefix.clone();
            }
            return;
        }
        if kemeny_bound(prefix, used, local) < *best_score {
            return;
        }
        for c in 0..m {
            if used[c] {
                continue;
            }
            used[c] = true;
            prefix.push(c);
            rec(local, members, rsb, used, prefix, best_score, best, best_key);
            prefix.pop();
            used[c] = false;
        }
    }
    rec(&local, members, rsb, &mut used, &mut prefix, &mut best_score, &mut best, &mut best_key);
    best.into_iter().map(|i| members[i]).collect()
}

pub(in crate::retrieval) fn kemeny_copeland(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut order = members.to_vec();
    order.sort_by(|&a, &b| {
        let sa: i64 = members.iter().map(|&c| (net[a][c] > 0) as i64 - (net[a][c] < 0) as i64).sum();
        let sb: i64 = members.iter().map(|&c| (net[b][c] > 0) as i64 - (net[b][c] < 0) as i64).sum();
        sb.cmp(&sa).then(rsb[a].cmp(&rsb[b]))
    });
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..order.len().saturating_sub(1) {
            let a = order[i];
            let b = order[i + 1];
            if net[a][b] < 0 {
                order.swap(i, i + 1);
                improved = true;
            }
        }
    }
    order
}

pub(in crate::retrieval) fn tarjan_sccs_best_first(net: &[Vec<i64>], rsb: &[usize]) -> Vec<Vec<usize>> {
    let n = net.len();
    let mut index = 0usize;
    let mut indices = vec![None; n];
    let mut low = vec![0usize; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut on = vec![false; n];
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    fn strong(
        v: usize,
        net: &[Vec<i64>],
        index: &mut usize,
        indices: &mut [Option<usize>],
        low: &mut [usize],
        stack: &mut Vec<usize>,
        on: &mut [bool],
        sccs: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        low[v] = *index;
        *index += 1;
        stack.push(v);
        on[v] = true;
        for w in 0..net.len() {
            if v != w && net[v][w] > 0 {
                if indices[w].is_none() {
                    strong(w, net, index, indices, low, stack, on, sccs);
                    low[v] = low[v].min(low[w]);
                } else if on[w] {
                    low[v] = low[v].min(indices[w].unwrap_or(0));
                }
            }
        }
        if low[v] == indices[v].unwrap_or(0) {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                on[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(comp);
        }
    }
    for v in 0..n {
        if indices[v].is_none() {
            strong(v, net, &mut index, &mut indices, &mut low, &mut stack, &mut on, &mut sccs);
        }
    }
    let mut id_of = vec![0usize; n];
    for (i, comp) in sccs.iter().enumerate() {
        for &v in comp {
            id_of[v] = i;
        }
    }
    let ns = sccs.len();
    let mut indeg = vec![0usize; ns];
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); ns];
    for a in 0..n {
        for b in 0..n {
            if a != b && net[a][b] > 0 && id_of[a] != id_of[b] {
                let (u, v) = (id_of[a], id_of[b]);
                if !next[u].contains(&v) {
                    next[u].push(v);
                    indeg[v] += 1;
                }
            }
        }
    }
    let anchor = |comp: &[usize]| comp.iter().map(|&v| rsb.get(v).copied().unwrap_or(usize::MAX)).min().unwrap_or(usize::MAX);
    let mut ready: Vec<usize> = (0..ns).filter(|&s| indeg[s] == 0).collect();
    let sort_ready = |ready: &mut Vec<usize>, sccs: &[Vec<usize>]| {
        ready.sort_by(|&a, &b| anchor(&sccs[b]).cmp(&anchor(&sccs[a])).then(b.cmp(&a)));
    };
    sort_ready(&mut ready, &sccs);
    let mut ordered = Vec::with_capacity(ns);
    while let Some(u) = ready.pop() {
        ordered.push(u);
        let mut outs = next[u].clone();
        outs.sort_unstable();
        for v in outs {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                ready.push(v);
            }
        }
        sort_ready(&mut ready, &sccs);
    }
    if ordered.len() != ns {
        ordered = (0..ns).collect();
    }
    // Tarjan's component ids are not best-first. Kahn from sources of the
    // "beats" condensation puts undefeated components first.
    ordered.into_iter().map(|i| sccs[i].clone()).collect()
}

pub(in crate::retrieval) fn kemeny_young(net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(net.len());
    for scc in tarjan_sccs_best_first(net, rsb) {
        if scc.len() <= 1 {
            out.extend(scc);
        } else if scc.len() <= KEMENY_EXACT_CAP {
            out.extend(kemeny_branch_and_bound(&scc, net, rsb));
        } else {
            out.extend(kemeny_copeland(&scc, net, rsb));
        }
    }
    out
}
