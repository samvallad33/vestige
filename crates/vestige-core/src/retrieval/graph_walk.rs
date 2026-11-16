//! Bounded walk over the memory graph: recall by association.
//!
//! Remembering one thing pulls up what it is linked to. A memory that shares
//! no words with the cue can still surface because it is tied to something
//! the cue did reach: the reply to a statement, the cause behind a symptom
//! (RSB), the rule a later turn invoked, the older fact a newer one revised.
//!
//! The walk uses only edges the index already holds, made undirected so
//! association runs both ways:
//!
//! * `answers`: same-session adjacency pairs (who replied to whom),
//! * `exact_edges`: caller and ingest-time RSB links,
//! * `policy_backlinks`: a turn that invoked an earlier standing policy,
//! * `supersedes_turns`: state revisions and contradictions.
//!
//! Seeds are the recalled top-k (seed `i` starts with mass `1 / (i + 1)`) and,
//! in the combined variant, the self-statement channel. Each hop keeps
//! `HOP_DECAY` of the mass and splits it evenly over the node's links, so a
//! hub spreads thin and a dedicated link carries strongly. A reached memory
//! surfaces by graph mass alone, never by word overlap with the query.

use super::{DetHashSet, TurnIx};

/// Most walk-reached memories returned per query. Small next to top-k, like
/// the self-statement channel; every item is reported against k by the scorer.
pub(super) const WALK_CAP: usize = 3;

/// Hop limit. One hop reaches direct associates; a second reaches the
/// associate's associate (a symptom's cause's reply). Beyond two hops spreading
/// activation is dominated by hubs.
const MAX_HOPS: usize = 2;

/// Fraction of activation kept per hop, the classic halving of spreading
/// activation (Collins & Loftus 1975; Anderson ACT-R fan effect).
const HOP_DECAY: f32 = 0.5;

/// Ingest-time undirected adjacency over the existing structural turn edges.
pub(super) fn build_adjacency(turns: &[TurnIx]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); turns.len()];
    let mut link = |a: usize, b: usize| {
        if a != b && a < adj.len() && b < adj.len() {
            adj[a].push(b);
            adj[b].push(a);
        }
    };
    for (ti, t) in turns.iter().enumerate() {
        if let Some(prev) = t.answers {
            link(ti, prev);
        }
        for &(_, target) in &t.exact_edges {
            link(ti, target);
        }
        for &policy in &t.policy_backlinks {
            link(ti, policy);
        }
        for &older in &t.supersedes_turns {
            link(ti, older);
        }
    }
    for neighbours in &mut adj {
        neighbours.sort_unstable();
        neighbours.dedup();
    }
    adj
}

/// Spreads activation from `seeds` (in recall order) over `adj` for up to
/// `MAX_HOPS` hops. Returns up to `cap` reached turns that are not seeds,
/// strongest first, ties by index.
pub(super) fn walk(adj: &[Vec<usize>], seeds: &[usize], cap: usize) -> Vec<(usize, f32)> {
    let n = adj.len();
    if seeds.is_empty() || n == 0 || cap == 0 {
        return Vec::new();
    }
    let seed_set: DetHashSet<usize> = seeds.iter().copied().collect();
    let mut reached = vec![0.0f32; n];
    let mut frontier: Vec<(usize, f32)> = seeds
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s < n)
        .map(|(rank, &s)| (s, 1.0 / (rank as f32 + 1.0)))
        .collect();
    for _ in 0..MAX_HOPS {
        let mut next = vec![0.0f32; n];
        let mut touched: Vec<usize> = Vec::new();
        for &(node, mass) in &frontier {
            let neighbours = &adj[node];
            if neighbours.is_empty() {
                continue;
            }
            let share = mass * HOP_DECAY / neighbours.len() as f32;
            for &d in neighbours {
                if next[d] == 0.0 {
                    touched.push(d);
                }
                next[d] += share;
            }
        }
        touched.sort_unstable();
        frontier = touched.into_iter().map(|d| (d, next[d])).collect();
        for &(d, mass) in &frontier {
            reached[d] += mass;
        }
    }
    let mut out: Vec<(usize, f32)> = reached
        .into_iter()
        .enumerate()
        .filter(|&(ti, mass)| mass > 0.0 && !seed_set.contains(&ti))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(cap);
    out
}

#[cfg(test)]
mod tests {
    use super::walk;

    /// 0 - 1 - 2 - 3 chain plus a hub 4 linked to 5, 6, 7.
    fn graph() -> Vec<Vec<usize>> {
        vec![
            vec![1],
            vec![0, 2],
            vec![1, 3],
            vec![2],
            vec![5, 6, 7],
            vec![4],
            vec![4],
            vec![4],
        ]
    }

    #[test]
    fn walk_stops_after_two_hops_and_skips_seeds() {
        let got: Vec<usize> = walk(&graph(), &[0], 10)
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(
            got,
            vec![1, 2],
            "node 3 is three hops away; seed 0 is never returned"
        );
    }

    #[test]
    fn hub_spreads_mass_thin() {
        let from_chain = walk(&graph(), &[0], 1)[0].1;
        let from_hub = walk(&graph(), &[4], 1)[0].1;
        assert!(
            from_hub < from_chain,
            "a hub's mass is split over its links"
        );
    }

    #[test]
    fn empty_inputs_return_nothing() {
        assert!(walk(&graph(), &[], 3).is_empty());
        assert!(walk(&[], &[0], 3).is_empty());
        assert!(walk(&graph(), &[0], 0).is_empty());
    }
}
