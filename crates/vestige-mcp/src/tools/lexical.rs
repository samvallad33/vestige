//! Exact-token lexical ranking over stored memory text.
//!
//! This is BM25 with a fixed formula: same log, same query, same order.
//! Tokens are case-folded splits on non-letters and non-digits. There is no
//! stemmer, no edit distance, no embedding, and no score that means "similar
//! in meaning." A term scores only when it occurs as a whole token in the
//! memory's content or in one of its tags.

use std::collections::{BTreeMap, BTreeSet};

use vestige_core::KnowledgeNode;

/// About 3_000 tokens of hook text, counted the same way the rest of the
/// server counts tokens: UTF-8 bytes divided by 4.
pub const HOOK_TOKEN_BUDGET: usize = 3_000;
pub const HOOK_CHAR_BUDGET: usize = HOOK_TOKEN_BUDGET * 4;

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Identity tags a two-hop bridge may walk. Exact prefix, case-sensitive.
const IDENTITY_PREFIXES: &[&str] = &["person:", "place:", "food:", "file:", "service:", "symbol:"];

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "for", "on", "with", "at", "by", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "and", "or",
    "but", "if", "then", "so", "than", "too", "very", "can", "will", "just", "not", "no", "nor",
    "you", "your", "we", "our", "they", "them", "he", "she", "his", "her", "their", "i", "me",
    "my", "what", "which", "who", "whom", "how", "when", "where", "why", "into", "about", "over",
    "after", "before", "do", "does", "did", "have", "has", "had", "should",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalHit {
    pub index: usize,
    /// BM25 rounded to millionths. Higher is a closer exact-token match.
    /// Ties break by caller-supplied salience, then recency, then id.
    pub score_micro: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagBridge {
    pub index: usize,
    /// `[shared identity, middle memory id, onward identity]`.
    pub path: [String; 3],
}

/// Case-folded exact tokens. One-character pieces are dropped.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            push_token(&mut current, &mut out);
        }
    }
    if !current.is_empty() {
        push_token(&mut current, &mut out);
    }
    out
}

fn push_token(current: &mut String, out: &mut Vec<String>) {
    if current.chars().count() >= 2 && !is_stopword(current) {
        out.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

fn document_text(node: &KnowledgeNode) -> String {
    let mut text = node.content.clone();
    for tag in &node.tags {
        text.push(' ');
        text.push_str(tag);
    }
    text
}

/// Rank `nodes` by exact-token BM25 against `query`. Score 0 is omitted.
/// Order is score, then the `tie` key descending, then id descending, so the
/// same store always returns the same list.
pub fn rank(
    nodes: &[KnowledgeNode],
    query: &str,
    tie: impl Fn(&KnowledgeNode) -> i128,
) -> Vec<LexicalHit> {
    let query_terms: Vec<String> = {
        let mut seen = BTreeSet::new();
        tokens(query)
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect()
    };
    if query_terms.is_empty() || nodes.is_empty() {
        return Vec::new();
    }
    let docs: Vec<Vec<String>> = nodes.iter().map(|n| tokens(&document_text(n))).collect();
    let lengths: Vec<u32> = docs.iter().map(|d| d.len() as u32).collect();
    let avgdl = lengths.iter().map(|n| *n as f64).sum::<f64>() / docs.len() as f64;
    let n_docs = docs.len() as f64;
    let mut df: BTreeMap<&str, usize> = BTreeMap::new();
    for doc in &docs {
        let mut unique = BTreeSet::new();
        for term in doc {
            unique.insert(term.as_str());
        }
        for term in unique {
            *df.entry(term).or_default() += 1;
        }
    }
    let mut hits = Vec::new();
    for (index, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            continue;
        }
        let mut tf: BTreeMap<&str, u32> = BTreeMap::new();
        for term in doc {
            *tf.entry(term.as_str()).or_default() += 1;
        }
        let mut score = 0.0;
        for term in &query_terms {
            let Some(freq) = tf.get(term.as_str()).copied() else {
                continue;
            };
            let df = df.get(term.as_str()).copied().unwrap_or(0) as f64;
            let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
            let freq = freq as f64;
            let denom = freq + K1 * (1.0 - B + B * (lengths[index] as f64 / avgdl.max(1.0)));
            score += idf * (freq * (K1 + 1.0)) / denom;
        }
        let score_micro = (score * 1_000_000.0).round() as i64;
        if score_micro > 0 {
            hits.push(LexicalHit { index, score_micro });
        }
    }
    hits.sort_by(|a, b| {
        b.score_micro
            .cmp(&a.score_micro)
            .then_with(|| tie(&nodes[b.index]).cmp(&tie(&nodes[a.index])))
            .then_with(|| nodes[b.index].created_at.cmp(&nodes[a.index].created_at))
            .then_with(|| nodes[b.index].id.cmp(&nodes[a.index].id))
    });
    hits
}

/// Take whole records until `char_budget` is spent. The first hit is kept
/// even when it is the only record and it is long; later records that fit
/// the remaining budget are added while skipping any oversized intermediate
/// record. `max_items` is a hard cap.
pub fn pack<'a>(
    nodes: &'a [KnowledgeNode],
    hits: &[LexicalHit],
    char_budget: usize,
    max_items: usize,
) -> Vec<&'a KnowledgeNode> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for hit in hits {
        if out.len() >= max_items {
            break;
        }
        let node = &nodes[hit.index];
        let cost = node
            .content
            .len()
            .saturating_add(node.id.len())
            .saturating_add(32);
        if !out.is_empty() && used.saturating_add(cost) > char_budget {
            continue;
        }
        used = used.saturating_add(cost);
        out.push(node);
    }
    out
}

fn identity_tags(node: &KnowledgeNode) -> Vec<&str> {
    node.tags
        .iter()
        .map(String::as_str)
        .filter(|tag| {
            IDENTITY_PREFIXES
                .iter()
                .any(|prefix| tag.starts_with(prefix))
        })
        .collect()
}

/// Memories two exact identity tags away from `seeds`: a seed shares one
/// identity with a middle memory, and that middle shares a different identity
/// with the far memory. The far memory shares none of the seed identities, so
/// a one-hop neighbor is not reported again as a bridge.
pub fn bridges(nodes: &[KnowledgeNode], seed_indexes: &[usize], limit: usize) -> Vec<TagBridge> {
    if limit == 0 || seed_indexes.is_empty() {
        return Vec::new();
    }
    let mut start: BTreeSet<&str> = BTreeSet::new();
    for index in seed_indexes {
        for tag in identity_tags(&nodes[*index]) {
            start.insert(tag);
        }
    }
    if start.is_empty() {
        return Vec::new();
    }
    let mut by_tag: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        for tag in identity_tags(node) {
            by_tag.entry(tag).or_default().push(index);
        }
    }
    let seed_set: BTreeSet<usize> = seed_indexes.iter().copied().collect();
    let mut direct: Vec<(usize, &str)> = Vec::new();
    for tag in &start {
        for index in by_tag.get(tag).into_iter().flatten() {
            if seed_set.contains(index) {
                continue;
            }
            if !direct.iter().any(|(have, _)| have == index) {
                direct.push((*index, *tag));
            }
        }
    }
    let mut found = Vec::new();
    for (middle, shared) in direct {
        for onward in identity_tags(&nodes[middle]) {
            if start.contains(onward) {
                continue;
            }
            for far in by_tag.get(onward).into_iter().flatten().copied() {
                if far == middle || seed_set.contains(&far) {
                    continue;
                }
                if identity_tags(&nodes[far])
                    .iter()
                    .any(|tag| start.contains(tag))
                {
                    continue;
                }
                if found.iter().any(|bridge: &TagBridge| bridge.index == far) {
                    continue;
                }
                found.push(TagBridge {
                    index: far,
                    path: [
                        shared.to_string(),
                        nodes[middle].id.clone(),
                        onward.to_string(),
                    ],
                });
                if found.len() >= limit {
                    return found;
                }
            }
        }
    }
    found
}

/// Person tags literally named by `prompt` (the tag, or the text after `person:`).
pub fn named_people(nodes: &[KnowledgeNode], prompt: &str) -> BTreeSet<String> {
    let mut people = BTreeSet::new();
    for node in nodes {
        for tag in &node.tags {
            let Some(name) = tag.strip_prefix("person:") else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            if phrase_in(prompt, tag) || phrase_in(prompt, name) {
                people.insert(tag.clone());
            }
        }
    }
    people
}

fn phrase_in(text: &str, surface: &str) -> bool {
    let text = text.to_lowercase();
    let surface = surface.to_lowercase();
    if surface.is_empty() {
        return false;
    }
    text.match_indices(&surface).any(|(at, _)| {
        let word = |c: char| c.is_alphanumeric() || c == '_';
        text[..at].chars().next_back().is_none_or(|c| !word(c))
            && text[at + surface.len()..]
                .chars()
                .next()
                .is_none_or(|c| !word(c))
    })
}

/// Drop a memory that is about some other recorded person when the prompt
/// named one. A memory with no `person:` tag can still match on its words.
pub fn allows_person(node: &KnowledgeNode, named: &BTreeSet<String>) -> bool {
    if named.is_empty() {
        return true;
    }
    let people: Vec<&str> = node
        .tags
        .iter()
        .filter_map(|tag| tag.strip_prefix("person:").map(|_| tag.as_str()))
        .collect();
    people.is_empty() || people.iter().any(|tag| named.contains(*tag))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, content: &str, tags: &[&str]) -> KnowledgeNode {
        let mut node = KnowledgeNode::new(content);
        node.id = id.to_string();
        node.tags = tags.iter().map(|tag| (*tag).to_string()).collect();
        node
    }

    #[test]
    fn exact_tokens_rank_the_closer_memory_and_ignore_a_miss() {
        let nodes = vec![
            node("a", "the deploy timeout is two seconds", &["deploy-env"]),
            node("b", "unrelated gardening note", &[]),
            node("c", "deploy checklist", &[]),
        ];
        let hits = rank(&nodes, "what is the deploy timeout", |_| 0);
        assert!(hits.len() >= 2, "{hits:?}");
        assert_eq!(hits[0].index, 0, "both terms outrank one term");
        assert!(hits.iter().all(|hit| hit.index != 1));
        assert!(hits[0].score_micro > hits[1].score_micro);
    }

    #[test]
    fn a_repeated_query_is_the_same_order() {
        let nodes = vec![
            node(
                "a",
                "Maya felt off after dinner at Bangkok Garden",
                &["person:Maya"],
            ),
            node("b", "Maya has a peanut allergy", &["person:Maya"]),
            node(
                "c",
                "Liam felt off after dinner at Bangkok Garden",
                &["person:Liam"],
            ),
        ];
        let once = rank(
            &nodes,
            "where should we take Maya for her birthday dinner",
            |_| 0,
        );
        let twice = rank(
            &nodes,
            "where should we take Maya for her birthday dinner",
            |_| 0,
        );
        assert_eq!(once, twice);
        assert_eq!(once[0].index, 0);
    }

    #[test]
    fn two_hop_bridge_walks_a_different_identity_and_skips_the_direct_neighbor() {
        let nodes = vec![
            node("seed", "Maya allergy", &["person:Maya"]),
            node("middle", "dinner", &["person:Maya", "place:Lisbon"]),
            node("far", "Lisbon trip", &["place:Lisbon", "person:Ana"]),
            node("direct-only", "also Maya", &["person:Maya"]),
        ];
        let bridges = bridges(&nodes, &[0], 4);
        assert_eq!(bridges.len(), 1, "{bridges:?}");
        assert_eq!(bridges[0].index, 2);
        assert_eq!(bridges[0].path[0], "person:Maya");
        assert_eq!(bridges[0].path[1], "middle");
        assert_eq!(bridges[0].path[2], "place:Lisbon");
    }

    #[test]
    fn packing_keeps_whole_records_inside_the_budget() {
        let nodes = vec![
            node("a", &"a".repeat(100), &[]),
            node("b", &"b".repeat(5_000), &[]),
            node("c", &"c".repeat(100), &[]),
        ];
        let hits = vec![
            LexicalHit {
                index: 0,
                score_micro: 3,
            },
            LexicalHit {
                index: 1,
                score_micro: 2,
            },
            LexicalHit {
                index: 2,
                score_micro: 1,
            },
        ];
        let packed = pack(&nodes, &hits, 300, 8);
        let ids: Vec<_> = packed.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["a", "c"], "oversized middle record is skipped so later fitting records still pack");
        assert_eq!(packed[0].content.len(), 100, "the record is whole");
    }
}
