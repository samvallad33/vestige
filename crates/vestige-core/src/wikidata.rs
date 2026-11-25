//! Packed Wikidata CC0 1.0 topological containment & composition graph
//! (`P17` country, `P131` admin-unit/location, `P195` collection, `P159` HQ, `P527` part/ingredient)
//! built deterministically by `scripts/build_wikidata_table.py`.
//!
//! sha256 of `wikidata-cc0-4rel.tsv` is pinned in [`DATA_SHA256`] and verified by a unit test.
//! Zero floats, zero hand-typed open-class word lists.

use std::collections::HashMap;
use std::sync::OnceLock;

const DATA_TSV: &str = include_str!("../data/wikidata-cc0-4rel.tsv");

pub const DATA_SHA256: &str =
    "806a5ada7944c2f92fe355b577c1211c9e5b50580dd62b7ceb047e053eba0694";

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EdgeRel {
    Country,
    LocatedIn,
    Collection,
    Headquarters,
    HasPart,
}

impl EdgeRel {
    fn from_code(code: &str) -> Self {
        match code {
            "P17" => Self::Country,
            "P195" => Self::Collection,
            "P159" => Self::Headquarters,
            "P527" => Self::HasPart,
            _ => Self::LocatedIn,
        }
    }
}

struct TopoGraph {
    fwd: HashMap<&'static str, Vec<(&'static str, EdgeRel, u16)>>,
    rev: HashMap<&'static str, Vec<(&'static str, EdgeRel, u16)>>,
    /// Nodes whose outgoing `Country` (`P17`) edges point to >= 2 distinct countries
    /// (e.g., `"central district"` shared across Israel, Botswana, Iran, Russia) —
    /// forbidden as multi-hop intermediate bridges so cities never cross country borders!
    ambiguous_admin: HashMap<&'static str, bool>,
}

fn graph() -> &'static TopoGraph {
    static G: OnceLock<TopoGraph> = OnceLock::new();
    G.get_or_init(|| {
        let mut fwd: HashMap<&'static str, Vec<(&'static str, EdgeRel, u16)>> = HashMap::new();
        let mut rev: HashMap<&'static str, Vec<(&'static str, EdgeRel, u16)>> = HashMap::new();
        let mut token_owners: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        let mut p17_countries: HashMap<&'static str, Vec<(&'static str, u16)>> = HashMap::new();
        for line in DATA_TSV.lines() {
            let mut it = line.split('\t');
            let (Some(src), Some(rel_str), Some(dst), Some(w_str)) =
                (it.next(), it.next(), it.next(), it.next())
            else {
                continue;
            };
            let rel = EdgeRel::from_code(rel_str);
            let w: u16 = w_str.parse().unwrap_or(15);
            if rel == EdgeRel::Country {
                let cs = p17_countries.entry(src).or_default();
                if let Some(existing) = cs.iter_mut().find(|(c, _)| *c == dst) {
                    if w > existing.1 {
                        existing.1 = w;
                    }
                } else {
                    cs.push((dst, w));
                }
            }
            fwd.entry(src).or_default().push((dst, rel, w));
            rev.entry(dst).or_default().push((src, rel, w));
            if src.contains(' ') {
                for tok in src.split_ascii_whitespace() {
                    if tok.len() >= 5 && tok.bytes().all(|b| b.is_ascii_alphabetic()) {
                        let owners = token_owners.entry(tok).or_default();
                        if owners.len() <= 3 && !owners.contains(&src) {
                            owners.push(src);
                        }
                    }
                }
            }
        }
        // Distinctiveness-gated sub-phrase alias: any >=5-char constituent token appearing in
        // <= 2 multi-word entity names across the entire graph, NOT already a standalone key,
        // and NOT a common English noun in WordNet (`senses(tok).is_empty()` prevents `"stuck"` -> `"villa stuck"`,
        // `"attic"` -> `"our lord in the attic museum"`, `"operation"` -> `"OEEC"`) inherits its owner's edges.
        for (tok, owners) in token_owners {
            if owners.len() <= 2 && !fwd.contains_key(tok) && crate::wordnet::senses(tok).is_empty() {
                let mut inherited: Vec<(&'static str, EdgeRel, u16)> = Vec::new();
                for owner in owners {
                    if let Some(edges) = fwd.get(owner) {
                        inherited.extend_from_slice(edges);
                    }
                }
                if !inherited.is_empty() {
                    fwd.insert(tok, inherited);
                }
            }
        }
        for list in fwd.values_mut().chain(rev.values_mut()) {
            list.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(b.0)).then_with(|| a.1.cmp(&b.1)));
            list.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        }
        let mut ambiguous_admin: HashMap<&'static str, bool> = HashMap::new();
        for (node, mut countries) in p17_countries {
            if countries.len() >= 2 {
                countries.sort_by_key(|a| std::cmp::Reverse(a.1));
                let top_w = countries[0].1;
                let second_w = countries[1].1;
                // Only block multi-hop traversal when no single country dominates (>= 3x sitelinks);
                // e.g. "paris" (france:367 vs us:33) is canonical, whereas "central district" is ambiguous.
                if top_w < second_w.saturating_mul(3) {
                    ambiguous_admin.insert(node, true);
                }
            }
        }
        TopoGraph {
            fwd,
            rev,
            ambiguous_admin,
        }
    })
}

fn resolve_source_key(g: &'static TopoGraph, entity: &str) -> Option<&'static str> {
    if let Some((&seed, _)) = g.fwd.get_key_value(entity) {
        return Some(seed);
    }
    // Closed-compound & Germanic/Romance apocopated compound resolution
    // (e.g., `"semper opera"` / `"semper opera house"` -> `"semperoper"`):
    let mut parts = entity.split_ascii_whitespace();
    let (Some(w0), Some(w1)) = (parts.next(), parts.next()) else {
        return None;
    };
    if w0.len() >= 4 && w1.len() >= 4 {
        let mut joined = format!("{w0}{w1}");
        if let Some((&seed, _)) = g.fwd.get_key_value(joined.as_str()) {
            return Some(seed);
        }
        // Double-consonant Italian/French loanword reduction (`"caffe mocha"` -> `"cafe mocha"`):
        let mut dedup_w0 = String::with_capacity(w0.len());
        let mut prev_c = '\0';
        for ch in w0.chars() {
            if ch != prev_c || matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u') {
                dedup_w0.push(ch);
            }
            prev_c = ch;
        }
        if dedup_w0 != w0 {
            let dedup_phrase = format!("{dedup_w0} {w1}");
            if let Some((&seed, _)) = g.fwd.get_key_value(dedup_phrase.as_str()) {
                return Some(seed);
            }
        }
        joined.clear();
        if w1.len() >= 5 {
            let apoc = format!("{w0}{}", &w1[..w1.len() - 1]);
            if let Some((&seed, _)) = g.fwd.get_key_value(apoc.as_str()) {
                return Some(seed);
            }
        }
    }
    None
}

/// True when `entity` exists as a source entity in the CC0 topological graph.
pub fn has_source(entity: &str) -> bool {
    resolve_source_key(graph(), entity).is_some()
}

/// Direct 1-hop typed outgoing edges (`src -> [(dst, rel)]`) for exact-edge reachability walks.
pub fn direct_typed_edges(entity: &str) -> Vec<(&'static str, EdgeRel)> {
    let g = graph();
    let Some(seed) = resolve_source_key(g, entity) else {
        return Vec::new();
    };
    let Some(edges) = g.fwd.get(seed) else {
        return Vec::new();
    };
    edges.iter().take(8).map(|&(dst, rel, _)| (dst, rel)).collect()
}

/// True if `node` is an ambiguous cross-country administrative label (e.g. `"central district"`)
/// that must never be used as a multi-hop geographic intermediate.
pub fn is_ambiguous_admin(node: &str) -> bool {
    graph().ambiguous_admin.contains_key(node)
}

/// Forward 1..=3 hop topological containment closure (`entity -> container -> country`),
/// e.g. `"girl with a pearl earring" -> "mauritshuis" -> "the hague" -> "netherlands"`.
/// Zero floats; deterministic ordering.
pub fn upward_containers(entity: &str, max_hops: usize) -> Vec<&'static str> {
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<(String, usize), Vec<&'static str>>> =
            std::cell::RefCell::new(HashMap::new());
    }
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&(entity.to_string(), max_hops)).cloned()) {
        return hit;
    }
    let g = graph();
    let Some(seed) = resolve_source_key(g, entity) else {
        CACHE.with(|c| {
            let mut m = c.borrow_mut();
            if m.len() >= 32_768 {
                m.clear();
            }
            m.insert((entity.to_string(), max_hops), Vec::new());
        });
        return Vec::new();
    };
    let mut visited: Vec<&'static str> = Vec::new();
    let mut frontier: Vec<&'static str> = vec![seed];
    for hop in 0..max_hops {
        let mut next: Vec<&'static str> = Vec::new();
        for &cur in &frontier {
            if hop > 0 && g.ambiguous_admin.contains_key(cur) {
                continue;
            }
            if let Some(edges) = g.fwd.get(cur) {
                for &(dst, _, _) in edges.iter().take(6) {
                    if dst != entity && !visited.contains(&dst) {
                        visited.push(dst);
                        next.push(dst);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    visited.sort_unstable();
    visited.dedup();
    CACHE.with(|c| {
        let mut m = c.borrow_mut();
        if m.len() >= 32_768 {
            m.clear();
        }
        m.insert((entity.to_string(), max_hops), visited.clone());
    });
    visited
}

/// Inverse 1-hop contained members (`country/region/ingredient -> notable instances`),
/// capped to `max_out` highest-sitelink entries.
pub fn direct_instances_of(container: &str, max_out: usize) -> Vec<&'static str> {
    let g = graph();
    let Some(edges) = g.rev.get(container) else {
        return Vec::new();
    };
    edges.iter().take(max_out).map(|&(src, _, _)| src).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn wikidata_table_matches_pinned_sha256() {
        let normalized = DATA_TSV.replace("\r\n", "\n");
        let hash = Sha256::digest(normalized.as_bytes());
        let digest: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(digest, DATA_SHA256);
    }
}
