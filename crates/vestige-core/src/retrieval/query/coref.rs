//! Issue #813: Session-Scoped Tarjan Union-Find Coreference Classes in pure
//! integer arithmetic (path compression + union by rank). Zero floats, zero word lists.
//!
//! Clusters attested speaker nicknames (`SpeakerNickIndex`: e.g. `mel` <-> `melanie`)
//! and bound Hobbs antecedent mentions into equivalence classes over dense `TokenId`s
//! (`0..vocab.names.len()`), then computes class-equivalence query coverage affinity.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;

pub(in crate::retrieval) const Q16: u64 = 1 << 16;

/// Tarjan Disjoint-Set Union with path compression and union-by-rank.
pub(in crate::retrieval) struct UnionFindCoref {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFindCoref {
    pub(in crate::retrieval) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0u8; n],
        }
    }

    #[inline]
    pub(in crate::retrieval) fn find(&mut self, x: u32) -> u32 {
        let xi = x as usize;
        if xi >= self.parent.len() {
            return x;
        }
        let mut root = xi;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        let mut cur = xi;
        while cur != root {
            let nxt = self.parent[cur] as usize;
            self.parent[cur] = root as u32;
            cur = nxt;
        }
        root as u32
    }

    pub(in crate::retrieval) fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb || ra as usize >= self.parent.len() || rb as usize >= self.parent.len() {
            return;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] = self.rank[ra as usize].saturating_add(1);
            }
        }
    }
}

/// Build session-agnostic nickname + attested-alias equivalence classes from `Index`
/// and score turns that match a query term's coreference class (`Q16 / 2` when the
/// turn lacks direct surface equality but shares a Union-Find root with a query term).
pub(in crate::retrieval) fn coref_class_affinity_q16(ix: &Index, plan: &QueryPlan) -> Vec<u64> {
    let nt = ix.turns.len();
    let mut scores = vec![0u64; nt];
    let n_vocab = ix.vocab.names.len();
    if nt <= 2 || n_vocab == 0 || plan.terms.is_empty() {
        return scores;
    }

    let mut uf = UnionFindCoref::new(n_vocab);
    let mut any_union = false;

    for (nick, full) in &ix.nicks.prefix_of {
        if let (Some(nid), Some(fid)) = (ix.vocab.get(nick), ix.vocab.get(full)) {
            uf.union(nid, fid);
            any_union = true;
        }
        let ent_n = format!("\u{1}ent:{nick}");
        let ent_f = format!("\u{1}ent:{full}");
        if let (Some(enid), Some(efid)) = (ix.vocab.get(&ent_n), ix.vocab.get(&ent_f)) {
            uf.union(enid, efid);
            any_union = true;
        }
    }

    if !any_union {
        return scores;
    }

    let n_docs = (ix.t_own_stats.n as u32).max(1);
    let q_roots: Vec<(u32, u32)> = plan
        .terms
        .iter()
        .filter(|qt| qt.exact_hit)
        .filter_map(|qt| {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            if df > 0 && df * 2 < n_docs {
                Some((uf.find(qt.id), qt.id))
            } else {
                None
            }
        })
        .collect();

    if q_roots.is_empty() {
        return scores;
    }

    for (ti, t) in ix.turns.iter().enumerate() {
        if t.is_backchannel {
            continue;
        }
        let mut hits = 0u64;
        for &cid in &t.content {
            let r = uf.find(cid);
            if q_roots.iter().any(|&(qr, orig_id)| qr == r && orig_id != cid && !t.content.contains(&orig_id)) {
                hits += 1;
            }
        }
        if hits > 0 {
            scores[ti] = (hits * (Q16 >> 1)).min(Q16);
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_clusters_transitively() {
        let mut uf = UnionFindCoref::new(6);
        uf.union(1, 2);
        uf.union(2, 4);
        assert_eq!(uf.find(1), uf.find(4));
        assert_ne!(uf.find(1), uf.find(3));
    }
}
