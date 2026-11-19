//! Packed ConceptNet 5.7.0 commonsense graph for Ounis-gated zero-overlap
//! query/index bridging (Issue #817).
//!
//! Covers `{AtLocation, UsedFor, CapableOf, HasA}` English single-word edges
//! packed deterministically by `scripts/build_conceptnet_table.py` from
//! `conceptnet-assertions-5.7.0.csv.gz` (CC-BY-SA 4.0).
//!
//! sha256 of `conceptnet-5.7-4rel.tsv` is pinned in [`DATA_SHA256`] and
//! verified by a unit test. Zero floats, zero hand-typed word lists.

use std::collections::HashMap;
use std::sync::OnceLock;

const DATA_TSV: &str = include_str!("../data/conceptnet-5.7-4rel.tsv");

pub const DATA_SHA256: &str =
    "440bab391f707506677d24236ace7ba17cb7905c6e56c11ae4bb0ed83d2d766f";

struct Graph {
    fwd: HashMap<&'static str, Vec<(&'static str, u16)>>,
    rev: HashMap<&'static str, Vec<(&'static str, u16)>>,
}

fn graph() -> &'static Graph {
    static G: OnceLock<Graph> = OnceLock::new();
    G.get_or_init(|| {
        let mut fwd: HashMap<&'static str, Vec<(&'static str, u16)>> = HashMap::new();
        let mut rev: HashMap<&'static str, Vec<(&'static str, u16)>> = HashMap::new();
        for line in DATA_TSV.lines() {
            let mut it = line.split('\t');
            let (Some(src), Some(_rel), Some(dst), Some(w_str)) =
                (it.next(), it.next(), it.next(), it.next())
            else {
                continue;
            };
            let w: u16 = w_str.parse().unwrap_or(100);
            fwd.entry(src).or_default().push((dst, w));
            rev.entry(dst).or_default().push((src, w / 2));
        }
        for list in fwd.values_mut().chain(rev.values_mut()) {
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        }
        Graph { fwd, rev }
    })
}

/// Return up to `max_out` deterministic 1-hop and 2-hop ConceptNet neighbors
/// `(term, integer_weight)` for `word` (where 100 = raw ConceptNet weight 1.0).
/// Zero floats; deterministic ordering.
pub fn two_hop_neighbors(word: &str, max_out: usize) -> Vec<(&'static str, u16)> {
    let g = graph();
    let mut scores: HashMap<&'static str, u16> = HashMap::new();

    if let Some(edges) = g.fwd.get(word) {
        for &(dst, w) in edges.iter().take(6) {
            if dst != word {
                let e = scores.entry(dst).or_insert(0);
                *e = (*e).max(w);
            }
            if let Some(hop2) = g.fwd.get(dst) {
                for &(dst2, w2) in hop2.iter().take(3) {
                    if dst2 != word && dst2 != dst {
                        let hw = w.min(w2) / 3;
                        if hw > 0 {
                            let e = scores.entry(dst2).or_insert(0);
                            *e = (*e).max(hw);
                        }
                    }
                }
            }
        }
    }

    if let Some(edges) = g.rev.get(word) {
        for &(src, w) in edges.iter().take(6) {
            if src != word && w > 0 {
                let e = scores.entry(src).or_insert(0);
                *e = (*e).max(w);
            }
        }
    }

    let mut out: Vec<(&'static str, u16)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    out.truncate(max_out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn conceptnet_table_matches_pinned_sha256() {
        let normalized = DATA_TSV.replace("\r\n", "\n");
        let hash = Sha256::digest(normalized.as_bytes());
        let digest: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(digest, DATA_SHA256);
    }

    #[test]
    fn conceptnet_two_hop_returns_deterministic_neighbors() {
        let n1 = two_hop_neighbors("accommodation", 8);
        let n2 = two_hop_neighbors("accommodation", 8);
        assert_eq!(n1, n2);
        assert!(!n1.is_empty(), "expected ConceptNet neighbors for accommodation");
    }
}
