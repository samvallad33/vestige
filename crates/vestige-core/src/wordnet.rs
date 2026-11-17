//! Princeton WordNet 3.1 noun relations for gated query expansion.
//!
//! Source: <https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz> (`dict/data.noun`),
//! bundled verbatim at `crates/vestige-core/data/wordnet-3.1/data.noun`.
//! sha256 `2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94`.
//! License: the WordNet 3.1 license (Princeton University), reproduced in
//! `crates/vestige-core/data/wordnet-3.1/LICENSE` and in the file's own header.
//!
//! The whole file is parsed. Nothing is filtered or pruned: every synset, its
//! lemmas, and its hypernym (`@`) and instance-hypernym (`@i`) pointers are
//! kept; hyponyms are the inverse of those pointers. Multi-word lemmas
//! (`ice_cream`) are not keyed, because the retriever looks up single tokens
//! and such a key could never be hit.
//!
//! This module only answers "which WordNet lemmas are one relation away from
//! this word". When expansion may run, and how it is weighted and filtered,
//! is decided by the retriever (see `retrieval::plan_query`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// The bundled WordNet 3.1 `data.noun`, byte-for-byte as distributed.
const DATA_NOUN: &str = include_str!("../data/wordnet-3.1/data.noun");

/// sha256 of [`DATA_NOUN`]; checked by a unit test.
pub const DATA_NOUN_SHA256: &str = "2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94";

#[derive(Default)]
struct Tables {
    /// lemma -> the noun synsets it belongs to (every sense).
    senses: HashMap<String, Vec<u32>>,
    /// synset -> its single-word lemmas, lowercased.
    lemmas: HashMap<u32, Vec<String>>,
    /// synset -> direct hypernym synsets.
    parents: HashMap<u32, Vec<u32>>,
    /// synset -> direct hyponym synsets (inverse of `parents`).
    children: HashMap<u32, Vec<u32>>,
}

fn push_unique<T: PartialEq>(v: &mut Vec<T>, x: T) {
    if !v.contains(&x) {
        v.push(x);
    }
}

fn parse(data: &str) -> Tables {
    let mut out = Tables::default();
    for line in data.lines() {
        // License header lines start with two spaces; synset lines with an offset.
        if !line.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            continue;
        }
        let head = line.split(" | ").next().unwrap_or(line);
        let f: Vec<&str> = head.split_ascii_whitespace().collect();
        // synset_offset lex_filenum ss_type w_cnt(hex) [word lex_id]... p_cnt [ptr offset pos st]...
        let Some(own) = f.first().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Some(w_cnt) = f.get(3).and_then(|s| usize::from_str_radix(s, 16).ok()) else {
            continue;
        };
        let p_at = 4 + 2 * w_cnt;
        let Some(p_cnt) = f.get(p_at).and_then(|s| s.parse::<usize>().ok()) else {
            continue;
        };
        for k in 0..p_cnt {
            let b = p_at + 1 + 4 * k;
            let (Some(sym), Some(off), Some(pos)) = (f.get(b), f.get(b + 1), f.get(b + 2)) else {
                break;
            };
            if (*sym == "@" || *sym == "@i")
                && *pos == "n"
                && let Ok(o) = off.parse::<u32>()
            {
                push_unique(out.parents.entry(own).or_default(), o);
                push_unique(out.children.entry(o).or_default(), own);
            }
        }
        for i in 0..w_cnt {
            let Some(lemma) = f.get(4 + 2 * i) else { break };
            if lemma.contains('_') {
                continue;
            }
            let key = lemma.to_lowercase();
            push_unique(out.senses.entry(key.clone()).or_default(), own);
            push_unique(out.lemmas.entry(own).or_default(), key);
        }
    }
    out
}

fn table() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| parse(DATA_NOUN))
}

/// WordNet's own noun detachment rules (morphy), applied when the surface
/// form is not itself a lemma. Inflectional morphology, not vocabulary.
const NOUN_DETACH: &[(&str, &str)] = &[
    ("ses", "s"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("men", "man"),
    ("ies", "y"),
    ("s", ""),
];

/// The noun synsets `word` belongs to, after morphy's detachment rules.
pub fn senses(word: &str) -> &'static [u32] {
    let t = &table().senses;
    if let Some(v) = t.get(word) {
        return v;
    }
    for (suf, rep) in NOUN_DETACH {
        if let Some(base) = word.strip_suffix(suf)
            && !base.is_empty()
            && let Some(v) = t.get(&format!("{base}{rep}"))
        {
            return v;
        }
    }
    &[]
}

/// Lemmas one WordNet relation away from `word` across all its noun senses:
/// synonyms (same synset), direct hypernyms and direct hyponyms. Sorted and
/// de-duplicated so the result is deterministic; `word` itself is excluded.
pub fn expansion_lemmas(word: &str) -> Vec<&'static str> {
    let t = table();
    let mut synsets: Vec<u32> = Vec::new();
    for &s in senses(word) {
        push_unique(&mut synsets, s);
        for &p in t.parents.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            push_unique(&mut synsets, p);
        }
        for &c in t.children.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            push_unique(&mut synsets, c);
        }
    }
    let mut out: Vec<&'static str> = synsets
        .iter()
        .flat_map(|s| t.lemmas.get(s).map(Vec::as_slice).unwrap_or(&[]))
        .map(String::as_str)
        .filter(|l| *l != word)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Eduard Hovy upward-only hypernym expansion: walks `@`/`@i` parent pointers up to `max_hops`
/// (never downward `children`) so concrete facts ("asthma") meet category queries ("condition", "disease")
/// without downward candidate explosion.
pub fn upward_hypernyms(word: &str, max_hops: usize) -> Vec<&'static str> {
    let t = table();
    let mut visited: Vec<u32> = Vec::new();
    let mut frontier: Vec<u32> = senses(word).to_vec();
    for &s in &frontier {
        push_unique(&mut visited, s);
    }
    for _ in 0..max_hops {
        let mut next = Vec::new();
        for s in frontier {
            for &p in t.parents.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                if !visited.contains(&p) {
                    visited.push(p);
                    next.push(p);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    let mut out: Vec<&'static str> = visited
        .iter()
        .flat_map(|s| t.lemmas.get(s).map(Vec::as_slice).unwrap_or(&[]))
        .map(String::as_str)
        .filter(|l| *l != word)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn bundled_file_matches_recorded_sha256() {
        let digest = Sha256::digest(DATA_NOUN.as_bytes());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, DATA_NOUN_SHA256);
    }

    #[test]
    fn license_header_is_present() {
        assert!(DATA_NOUN.contains("WordNet 3.1 Copyright 2011 by Princeton University"));
    }

    #[test]
    fn parses_every_synset_line() {
        let synsets = DATA_NOUN
            .lines()
            .filter(|l| l.as_bytes().first().is_some_and(u8::is_ascii_digit))
            .count();
        assert!(synsets > 80_000, "data.noun synsets: {synsets}");
        assert!(table().senses.len() > 50_000, "keyed lemmas: {}", table().senses.len());
    }

    #[test]
    fn expansion_reaches_hyponyms_hypernyms_and_plurals() {
        let dog = expansion_lemmas("dog");
        assert!(dog.contains(&"puppy"), "hyponym missing");
        assert!(dog.contains(&"canine"), "hypernym missing");
        let plural: Vec<&str> = expansion_lemmas("dogs").into_iter().filter(|l| *l != "dog").collect();
        assert_eq!(plural, dog);
        assert!(!dog.contains(&"dog"));
        assert!(expansion_lemmas("qzxv").is_empty());
        let mut sorted = dog.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, dog, "deterministic order");
    }
}
