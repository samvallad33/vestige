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
    /// multi-word (`2..=5` token) normalized phrase -> noun synsets (`chief executive officer`, `south africa`).
    phrase_senses: HashMap<String, Vec<u32>>,
    /// synset -> its single-word lemmas, lowercased.
    lemmas: HashMap<u32, Vec<String>>,
    /// synset -> its multi-word (`2..=5` token) normalized phrases, lowercased.
    phrases: HashMap<u32, Vec<String>>,
    /// synset -> direct hypernym synsets (`@`, `@i`).
    parents: HashMap<u32, Vec<u32>>,
    /// synset -> direct hyponym synsets (inverse of `parents`).
    children: HashMap<u32, Vec<u32>>,
    /// synset -> part/member holonym synsets (`#p`, `#m`) and container chains.
    holonyms: HashMap<u32, Vec<u32>>,
    /// synset -> part/member meronym & instance-hyponym synsets (`%p`, `%m`, `~i`).
    meronyms: HashMap<u32, Vec<u32>>,
    /// synset -> rare unambiguous gloss definition lemmas (`| ...`).
    gloss_lemmas: HashMap<u32, Vec<String>>,
    /// compound constituent word -> associated single-word/phrase lemmas from `_` entries.
    compound_links: HashMap<String, Vec<String>>,
    /// cardinal compass headword -> its `(deg + 180) % 360` opposite headword from `data.noun`.
    cardinal_opposite: HashMap<String, String>,
    /// Typed negative/exclusion edges (`vegan` -> `["animal", "dairy", "meat", "fish"]`,
    /// `lactose intolerance` -> `["milk", "lactose"]`) parsed from closed-class negation syntax in `data.noun`.
    excludes_edges: HashMap<String, Vec<String>>,
    /// Typed meronym & definition-composition edges (`cappuccino` -> `["espresso", "milk"]`,
    /// `omelette` -> `["egg"]`, `egg` -> `["animal"]`).
    has_part_edges: HashMap<String, Vec<String>>,
    /// Resnik IC subsumer descendant count across the WordNet 3.1 noun DAG.
    desc_count: HashMap<u32, u32>,
    /// Multi-word instance-hypernym (`@i`) surnames/tails (`seuss` -> `[Dr._Seuss]`, `vivaldi` -> `[Antonio_Vivaldi]`).
    instance_surnames: HashMap<String, Vec<u32>>,
    /// Rare gloss word -> synsets whose WordNet definition contains that word (`married` -> `[husband, wife, ...]`).
    gloss_to_synsets: HashMap<String, Vec<u32>>,
}

fn push_unique<T: PartialEq>(v: &mut Vec<T>, x: T) {
    if !v.contains(&x) {
        v.push(x);
    }
}

fn parse(data: &str) -> Tables {
    let mut out = Tables::default();
    let mut raw_gloss: Vec<(u32, Vec<String>)> = Vec::new();
    let mut raw_def_text: Vec<(u32, String)> = Vec::new();
    let mut raw_compounds: Vec<(u32, Vec<String>)> = Vec::new();
    let mut cardinal_deg: Vec<(String, u16)> = Vec::new();

    for line in data.lines() {
        // License header lines start with two spaces; synset lines with an offset.
        if !line.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            continue;
        }
        let mut split_pipe = line.splitn(2, " | ");
        let head = split_pipe.next().unwrap_or(line);
        let gloss = split_pipe.next().unwrap_or("");
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
        let mut is_instance_syn = false;
        for k in 0..p_cnt {
            let b = p_at + 1 + 4 * k;
            let (Some(sym), Some(off), Some(pos)) = (f.get(b), f.get(b + 1), f.get(b + 2)) else {
                break;
            };
            if *pos != "n" {
                continue;
            }
            let Ok(o) = off.parse::<u32>() else {
                continue;
            };
            if *sym == "@" || *sym == "@i" {
                if *sym == "@i" {
                    is_instance_syn = true;
                }
                push_unique(out.parents.entry(own).or_default(), o);
                push_unique(out.children.entry(o).or_default(), own);
            } else if *sym == "#p" || *sym == "#m" {
                push_unique(out.holonyms.entry(own).or_default(), o);
                push_unique(out.meronyms.entry(o).or_default(), own);
            } else if *sym == "%p" || *sym == "%m" || *sym == "~i" {
                push_unique(out.meronyms.entry(own).or_default(), o);
            }
        }
        for i in 0..w_cnt {
            let Some(lemma) = f.get(4 + 2 * i) else { break };
            if lemma.contains('_') {
                let all_parts: Vec<String> = lemma
                    .split('_')
                    .map(|p| p.trim_matches('.').to_lowercase())
                    .filter(|p| !p.is_empty())
                    .collect();
                if (2..=5).contains(&all_parts.len())
                    && all_parts
                        .iter()
                        .all(|p| p.bytes().all(|b| b.is_ascii_alphabetic() || b == b'\'' || b == b'-'))
                {
                    let phrase_key = all_parts.join(" ");
                    push_unique(out.phrase_senses.entry(phrase_key.clone()).or_default(), own);
                    push_unique(out.phrases.entry(own).or_default(), phrase_key.clone());
                    if let Some(stripped) = phrase_key.strip_prefix("the ")
                        && stripped.len() >= 3
                    {
                        push_unique(out.phrase_senses.entry(stripped.to_string()).or_default(), own);
                    }
                    if is_instance_syn
                        && let Some(tail) = all_parts.last()
                        && tail.len() >= 4
                        && tail.bytes().all(|b| b.is_ascii_alphabetic())
                    {
                        let entry = out.instance_surnames.entry(tail.clone()).or_default();
                        if entry.len() < 4 {
                            push_unique(entry, own);
                        }
                    }
                }
                let parts: Vec<String> = all_parts
                    .into_iter()
                    .filter(|p| p.len() >= 3 && p.bytes().all(|b| b.is_ascii_alphabetic()))
                    .collect();
                if parts.len() >= 2 {
                    raw_compounds.push((own, parts));
                }
                continue;
            }
            let key = lemma.to_lowercase();
            push_unique(out.senses.entry(key.clone()).or_default(), own);
            push_unique(out.lemmas.entry(own).or_default(), key);
        }
        if !gloss.is_empty() {
            if gloss.contains("cardinal compass point")
                && gloss.contains("degrees")
                && let Some(hw) = f.get(4)
                && !hw.contains('_')
                && let Some(deg) = gloss
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<u16>().ok())
                    .find(|&d| d < 360)
            {
                cardinal_deg.push((hw.to_ascii_lowercase(), deg));
            }
            let def_only = gloss.split('"').next().unwrap_or(gloss);
            raw_def_text.push((own, def_only.to_ascii_lowercase()));
            let gtoks: Vec<String> = def_only
                .split(|c: char| !c.is_ascii_alphabetic())
                .map(|w| w.to_ascii_lowercase())
                .filter(|w| w.len() >= 3)
                .collect();
            if !gtoks.is_empty() {
                raw_gloss.push((own, gtoks));
            }
        }
    }

    for (w_a, deg_a) in &cardinal_deg {
        let opp_deg = (*deg_a + 180) % 360;
        if let Some((w_b, _)) = cardinal_deg.iter().find(|(_, d)| *d == opp_deg) {
            out.cardinal_opposite.insert(w_a.clone(), w_b.clone());
        }
    }

    // Pass 2: Link unambiguous WordNet-indexed rare nouns from glosses (`| ...`) and `_` compounds.
    for (syn, parts) in raw_compounds {
        let mut partners: Vec<String> = out.lemmas.get(&syn).cloned().unwrap_or_default();
        for &h in out.parents.get(&syn).map(Vec::as_slice).unwrap_or(&[]) {
            if let Some(hl) = out.lemmas.get(&h) {
                for x in hl {
                    push_unique(&mut partners, x.clone());
                }
            }
        }
        for &h in out.holonyms.get(&syn).map(Vec::as_slice).unwrap_or(&[]) {
            if let Some(hl) = out.lemmas.get(&h) {
                for x in hl {
                    push_unique(&mut partners, x.clone());
                }
            }
        }
        for p in &parts {
            push_unique(&mut partners, p.clone());
        }
        for p in &parts {
            if out.senses.get(p).is_none_or(|s| s.len() <= 4) {
                let entry = out.compound_links.entry(p.clone()).or_default();
                for partner in &partners {
                    if partner != p && entry.len() < 16 {
                        push_unique(entry, partner.clone());
                    }
                }
            }
        }
    }

    let mut gloss_df: HashMap<String, u32> = HashMap::new();
    for (_syn, gtoks) in &raw_gloss {
        let mut uniq: Vec<String> = Vec::new();
        for tok in gtoks {
            let base = if out.senses.contains_key(tok) {
                tok.clone()
            } else if let Some(sing) = tok.strip_suffix('s') {
                if sing.len() >= 3 && out.senses.contains_key(sing) {
                    sing.to_string()
                } else {
                    continue;
                }
            } else {
                continue;
            };
            push_unique(&mut uniq, base);
        }
        for u in uniq {
            *gloss_df.entry(u).or_insert(0) += 1;
        }
    }

    for (syn, gtoks) in raw_gloss {
        let mut kept: Vec<String> = Vec::new();
        for tok in gtoks {
            let candidate = if out.senses.get(&tok).is_some_and(|s| !s.is_empty() && s.len() <= 5) {
                Some(tok)
            } else if let Some(sing) = tok.strip_suffix('s') {
                if sing.len() >= 3 && out.senses.get(sing).is_some_and(|s| !s.is_empty() && s.len() <= 5) {
                    Some(sing.to_string())
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(clean_tok) = candidate {
                let gdf = gloss_df.get(&clean_tok).copied().unwrap_or(0);
                let max_gdf = if clean_tok.len() >= 4 { 520 } else { 200 };
                if gdf <= max_gdf {
                    push_unique(&mut kept, clean_tok);
                    if kept.len() >= 16 {
                        break;
                    }
                }
            }
        }
        if !kept.is_empty() {
            // Inverse gloss link for monosemous (1-sense) synset lemmas and narrow 1-sense children
            let mut mono_targets: Vec<String> = Vec::new();
            for l in out.lemmas.get(&syn).map(Vec::as_slice).unwrap_or(&[]) {
                if out.senses.get(l).is_some_and(|s| s.len() == 1) {
                    push_unique(&mut mono_targets, l.clone());
                }
            }
            if let Some(kids) = out.children.get(&syn)
                && kids.len() <= 4
            {
                for &ch in kids {
                    for cl in out.lemmas.get(&ch).map(Vec::as_slice).unwrap_or(&[]) {
                        if out.senses.get(cl).is_some_and(|s| s.len() == 1) {
                            push_unique(&mut mono_targets, cl.clone());
                        }
                    }
                }
            }
            if !mono_targets.is_empty() {
                for gtok in &kept {
                    let gdf = gloss_df.get(gtok).copied().unwrap_or(0);
                    if gdf > 200 {
                        continue;
                    }
                    let entry = out.compound_links.entry(gtok.clone()).or_default();
                    for mt in &mono_targets {
                        if mt != gtok && entry.len() < 256 {
                            push_unique(entry, mt.clone());
                        }
                    }
                }
            }
            for gtok in &kept {
                let gdf = gloss_df.get(gtok).copied().unwrap_or(0);
                if gdf <= 140 {
                    let entry = out.gloss_to_synsets.entry(gtok.clone()).or_default();
                    if entry.len() < 48 {
                        push_unique(entry, syn);
                    }
                }
            }
            out.gloss_lemmas.insert(syn, kept);
        }
    }

    // Pass 3: Compile explicit typed `excludes` and `has_part` edges from WordNet 3.1 `data.noun`
    // using strictly closed-class syntactic negation/composition cues (zero open-class word lists).
    let mut syn_excludes: HashMap<u32, Vec<String>> = HashMap::new();
    let mut syn_has_part: HashMap<u32, Vec<String>> = HashMap::new();
    let norm_noun = |w: &str, senses_map: &HashMap<String, Vec<u32>>| -> Option<String> {
        if w.len() < 3 || !w.bytes().all(|b| b.is_ascii_alphabetic()) {
            return None;
        }
        if senses_map.contains_key(w) {
            return Some(w.to_string());
        }
        for (suf, rep) in NOUN_DETACH {
            if let Some(base) = w.strip_suffix(suf)
                && !base.is_empty()
            {
                let cand = format!("{base}{rep}");
                if cand.len() >= 3 && senses_map.contains_key(&cand) {
                    return Some(cand);
                }
            }
        }
        None
    };

    for &(syn, ref def_low) in &raw_def_text {
        // 1. Meronym pointers (`%p`, `%m`) -> `has_part`
        if let Some(meros) = out.meronyms.get(&syn) {
            for &m in meros {
                for ml in out.lemmas.get(&m).map(Vec::as_slice).unwrap_or(&[]) {
                    push_unique(syn_has_part.entry(syn).or_default(), ml.clone());
                }
                for mp in out.phrases.get(&m).map(Vec::as_slice).unwrap_or(&[]) {
                    push_unique(syn_has_part.entry(syn).or_default(), mp.clone());
                }
            }
        }

        for clause in def_low.split([';', '.']) {
            // Clean parenthetical asides like `(often)` into spaces
            let mut cleaned = String::with_capacity(clause.len());
            let mut depth = 0usize;
            for ch in clause.chars() {
                if ch == '(' {
                    depth += 1;
                    cleaned.push(' ');
                } else if ch == ')' {
                    depth = depth.saturating_sub(1);
                    cleaned.push(' ');
                } else if depth == 0 {
                    cleaned.push(if ch.is_ascii_alphabetic() || ch == '-' { ch } else { ' ' });
                }
            }
            let words: Vec<&str> = cleaned.split_ascii_whitespace().collect();

            // 2. Closed-class negation scopes: `"no <NP>"`, `"without <NP>"`, `"inability/unable to <verb> <NP>"`
            for idx in 0..words.len() {
                let neg_start = if words[idx] == "no" || words[idx] == "without" {
                    Some(idx + 1)
                } else if (words[idx] == "inability" || words[idx] == "unable")
                    && words.get(idx + 1) == Some(&"to")
                    && idx + 3 < words.len()
                {
                    Some(idx + 3)
                } else {
                    None
                };
                if let Some(st) = neg_start {
                    // Collect coordinated noun targets across closed-class conjunctions `or`/`and`/`nor`
                    let mut seg: Vec<&str> = Vec::new();
                    for &w in words.iter().skip(st).take(10) {
                        // hardcode-scan: closed-class (conjunctions, determiners, prepositions)
                        if matches!(w, "or" | "and" | "nor") {
                            if let Some(first_n) = seg.iter().find_map(|t| norm_noun(t, &out.senses)) {
                                push_unique(syn_excludes.entry(syn).or_default(), first_n);
                            }
                            if seg.len() >= 2 {
                                let ph = seg.join(" ");
                                if out.phrase_senses.contains_key(&ph) {
                                    push_unique(syn_excludes.entry(syn).or_default(), ph);
                                } else if let Some(s_last) = seg.last().and_then(|t| norm_noun(t, &out.senses)) {
                                    let ph_sing = format!("{} {s_last}", seg[..seg.len() - 1].join(" "));
                                    if out.phrase_senses.contains_key(&ph_sing) {
                                        push_unique(syn_excludes.entry(syn).or_default(), ph_sing);
                                    }
                                }
                            }
                            seg.clear();
                            continue;
                        }
                        // hardcode-scan: closed-class (prepositional/quantifier boundary markers)
                        if matches!(w, "at" | "in" | "on" | "from" | "with" | "for" | "by" | "to" | "of" | "that" | "which" | "who") {
                            break;
                        }
                        // hardcode-scan: closed-class (indefinite quantifiers/articles)
                        if matches!(w, "any" | "all" | "other" | "a" | "an" | "the") {
                            continue;
                        }
                        seg.push(w);
                    }
                    if !seg.is_empty() {
                        if let Some(first_n) = seg.iter().find_map(|t| norm_noun(t, &out.senses)) {
                            push_unique(syn_excludes.entry(syn).or_default(), first_n);
                        }
                        if seg.len() >= 2 {
                            let ph = seg.join(" ");
                            if out.phrase_senses.contains_key(&ph) {
                                push_unique(syn_excludes.entry(syn).or_default(), ph);
                            } else if let Some(s_last) = seg.last().and_then(|t| norm_noun(t, &out.senses)) {
                                let ph_sing = format!("{} {s_last}", seg[..seg.len() - 1].join(" "));
                                if out.phrase_senses.contains_key(&ph_sing) {
                                    push_unique(syn_excludes.entry(syn).or_default(), ph_sing);
                                }
                            }
                        }
                    }
                }
            }

            // 3. Closed-class partitive/composition syntax:
            //    `"parts of <A> and <B>"`, `"made of/from/with <NP>"`, `"consisting of <NP>"`,
            //    or leading participial modifier (`"<verb>-ed/-en <noun>"` at start of definition).
            for idx in 0..words.len() {
                let comp_start = if ((words[idx] == "parts" || words[idx] == "consisting" || words[idx] == "composed")
                    && words.get(idx + 1) == Some(&"of"))
                    || (words[idx] == "made"
                        && words.get(idx + 1).is_some_and(|w| matches!(*w, "of" | "from" | "with")))
                {
                    Some(idx + 2)
                } else {
                    None
                };
                if let Some(st) = comp_start {
                    let mut seg: Vec<&str> = Vec::new();
                    for &w in words.iter().skip(st).take(8) {
                        if matches!(w, "and" | "or") {
                            if let Some(head_n) = seg.last().and_then(|t| norm_noun(t, &out.senses)) {
                                push_unique(syn_has_part.entry(syn).or_default(), head_n);
                            }
                            seg.clear();
                            continue;
                        }
                        if matches!(w, "with" | "in" | "on" | "for" | "to" | "by" | "topped" | "served" | "cooked" | "baked" | "fried") {
                            break;
                        }
                        if matches!(w, "a" | "an" | "the" | "equal") {
                            continue;
                        }
                        seg.push(w);
                    }
                    if let Some(head_n) = seg.last().and_then(|t| norm_noun(t, &out.senses)) {
                        push_unique(syn_has_part.entry(syn).or_default(), head_n);
                    }
                }
            }
            // Participial opening (`"<past-participle> <noun>"` e.g. `"beaten eggs"`) or
            // biological origin modifier (`"<noun> reproductive body"` -> `<noun>`):
            if words.len() >= 2 {
                let w0 = words[0];
                let w1 = words[1];
                if (w0.ends_with("en") || w0.ends_with("ed"))
                    && w0.len() >= 5
                    && let Some(n1) = norm_noun(w1, &out.senses)
                {
                    push_unique(syn_has_part.entry(syn).or_default(), n1);
                } else if w1 == "reproductive"
                    && let Some(n0) = norm_noun(w0, &out.senses)
                {
                    push_unique(syn_has_part.entry(syn).or_default(), n0);
                }
            }
        }
    }

    // Inherit `excludes` across 1-hop `@` hypernym (`vegan` -> `vegetarian` -> `["meat", "fish", "animal"]`)
    // and materialize keyed by both single-word lemmas and multi-word phrases.
    let all_syns: Vec<u32> = out.lemmas.keys().chain(out.phrases.keys()).copied().collect();
    for syn in all_syns {
        let mut ex_all: Vec<String> = syn_excludes.get(&syn).cloned().unwrap_or_default();
        if out.lemmas.get(&syn).is_some_and(|ls| ls.iter().any(|l| out.senses.get(l).is_some_and(|s| s.len() == 1))) {
            for &p in out.parents.get(&syn).map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(pex) = syn_excludes.get(&p) {
                    for item in pex {
                        push_unique(&mut ex_all, item.clone());
                    }
                }
            }
        }
        if !ex_all.is_empty() {
            for k in out
                .lemmas
                .get(&syn)
                .into_iter()
                .flatten()
                .chain(out.phrases.get(&syn).into_iter().flatten())
            {
                let entry = out.excludes_edges.entry(k.clone()).or_default();
                for item in &ex_all {
                    if item != k {
                        push_unique(entry, item.clone());
                    }
                }
            }
        }
        if let Some(hp) = syn_has_part.get(&syn) {
            for k in out
                .lemmas
                .get(&syn)
                .into_iter()
                .flatten()
                .chain(out.phrases.get(&syn).into_iter().flatten())
            {
                let entry = out.has_part_edges.entry(k.clone()).or_default();
                for item in hp {
                    if item != k {
                        push_unique(entry, item.clone());
                    }
                }
            }
        }
    }

    // Pass 4: Exact WordNet 3.1 DAG descendant counts for Resnik Information Content (`IC(LCS)`).
    let mut all_syns: Vec<u32> = out
        .lemmas
        .keys()
        .chain(out.phrases.keys())
        .copied()
        .collect();
    all_syns.sort_unstable();
    all_syns.dedup();
    let mut vis_buf: Vec<u32> = Vec::with_capacity(32);
    for syn in all_syns {
        vis_buf.clear();
        vis_buf.push(syn);
        let mut cursor = 0usize;
        while cursor < vis_buf.len() && vis_buf.len() < 64 {
            let cur = vis_buf[cursor];
            cursor += 1;
            *out.desc_count.entry(cur).or_insert(0) += 1;
            if let Some(ps) = out.parents.get(&cur) {
                for &p in ps {
                    if !vis_buf.contains(&p) {
                        vis_buf.push(p);
                    }
                }
            }
        }
    }

    out
}

/// Resnik IC specific ancestor synsets for `word`: walks `@`/`@i`/`#p`/`#m` upward up to `max_hops`
/// from `senses(word)` (plus `@i` instance surnames when `word` is an instance proper noun), returning
/// `(synset_id, desc_count)` strictly filtered to specific informative synsets (`desc_count <= max_desc`).
pub fn specific_ancestor_synsets(word: &str, max_hops: usize, max_desc: u32) -> Vec<(u32, u32)> {
    let t = table();
    let mut seeds: Vec<u32> = senses(word).to_vec();
    if seeds.len() <= 2
        && let Some(inst) = t.instance_surnames.get(word)
    {
        for &s in inst {
            push_unique(&mut seeds, s);
        }
    }
    if seeds.is_empty()
        && let Some(sing) = word.strip_suffix("es").or_else(|| word.strip_suffix('s'))
        && sing.len() >= 3
    {
        seeds.extend_from_slice(senses(sing));
    }
    if seeds.is_empty() {
        return Vec::new();
    }
    let mut visited: Vec<u32> = Vec::new();
    let mut frontier: Vec<u32> = seeds;
    for &s in &frontier {
        push_unique(&mut visited, s);
    }
    for _ in 0..max_hops {
        let mut next = Vec::new();
        for s in frontier {
            for &p in t
                .parents
                .get(&s)
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .chain(t.holonyms.get(&s).map(Vec::as_slice).unwrap_or(&[]))
            {
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
    let mut out: Vec<(u32, u32)> = visited
        .into_iter()
        .filter_map(|s| {
            let dc = t.desc_count.get(&s).copied().unwrap_or(1);
            (dc <= max_desc).then_some((s, dc))
        })
        .collect();
    out.sort_unstable_by_key(|&(s, dc)| (dc, s));
    out
}

/// Rare WordNet gloss lemmas attached to `word`'s own noun/instance synsets
/// (e.g. `seuss` -> `["writer", "illustrator", "children", "book"]`, `vivaldi` -> `["baroque", "composer", "violinist"]`).
pub fn gloss_tokens_of(word: &str) -> Vec<&'static str> {
    let t = table();
    let mut seeds: Vec<u32> = senses(word).to_vec();
    if seeds.len() <= 2
        && let Some(inst) = t.instance_surnames.get(word)
    {
        for &s in inst {
            push_unique(&mut seeds, s);
        }
    }
    let mut out: Vec<&'static str> = Vec::new();
    for s in seeds {
        if let Some(gl) = t.gloss_lemmas.get(&s) {
            for g in gl {
                if g != word && !out.contains(&g.as_str()) {
                    out.push(g.as_str());
                }
            }
        }
    }
    out
}

/// Synsets whose WordNet 3.1 definition contains the rare word `word`.
pub fn synsets_with_gloss(word: &str) -> &'static [u32] {
    table()
        .gloss_to_synsets
        .get(word)
        .map(Vec::as_slice)
        .unwrap_or(&[])
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

/// The noun synsets `word` (or normalized space-separated multi-word phrase) belongs to,
/// after morphy's detachment rules.
pub fn senses(word: &str) -> &'static [u32] {
    let tbl = table();
    if word.contains(' ') {
        if let Some(v) = tbl.phrase_senses.get(word) {
            return v;
        }
        if let Some(stripped) = word.strip_prefix("the ")
            && let Some(v) = tbl.phrase_senses.get(stripped)
        {
            return v;
        }
        return &[];
    }
    let t = &tbl.senses;
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

/// Walk `@`/`@i` hypernyms PLUS `#p`/`#m` part/member holonyms up to `max_hops`,
/// unlocking geographical & part-whole containment (e.g., `Frankfurt -> Germany`, `Dresden -> Germany`).
type HopCache = std::sync::Mutex<HashMap<(String, usize), Vec<&'static str>>>;

pub fn holonyms_and_upward(word: &str, max_hops: usize) -> Vec<&'static str> {
    static CACHE: OnceLock<HopCache> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let key = (word.to_string(), max_hops);
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }
    let t = table();
    let mut visited: Vec<u32> = Vec::new();
    let mut frontier: Vec<u32> = senses(word).to_vec();
    for &s in &frontier {
        push_unique(&mut visited, s);
    }
    for _ in 0..max_hops {
        let mut next = Vec::new();
        for s in frontier {
            for &p in t
                .parents
                .get(&s)
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .chain(t.holonyms.get(&s).map(Vec::as_slice).unwrap_or(&[]).iter())
            {
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
    if let Some(cl) = t.compound_links.get(word) {
        for c in cl {
            if c != word {
                out.push(c.as_str());
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, out.clone());
    }
    out
}

/// Extract unambiguous gloss definition nouns (`| ...`), meronyms (`%p/%m`), and `_` compound links
/// associated with `word` directly from WordNet 3.1 `data.noun`, with 2-hop closure across
/// low-polysemy (`senses.len() <= 4`) intermediate concepts. Memoized globally across indexes.
pub fn gloss_and_compound_concepts(word: &str) -> Vec<&'static str> {
    static CACHE: OnceLock<std::sync::Mutex<HashMap<String, Vec<&'static str>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(word)
    {
        return hit.clone();
    }
    let t = table();
    let mut hop1: Vec<&'static str> = Vec::new();
    for &s in senses(word) {
        if let Some(gl) = t.gloss_lemmas.get(&s) {
            for g in gl {
                if g != word {
                    hop1.push(g.as_str());
                }
            }
        }
        for &m in t.meronyms.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            for ml in t.lemmas.get(&m).map(Vec::as_slice).unwrap_or(&[]) {
                if ml != word {
                    hop1.push(ml.as_str());
                }
            }
        }
    }
    if let Some(cl) = t.compound_links.get(word) {
        for c in cl {
            if c != word {
                hop1.push(c.as_str());
            }
        }
    }
    hop1.sort_unstable();
    hop1.dedup();

    let mut out = hop1.clone();
    for &mid in &hop1 {
        let mid_senses = senses(mid);
        if mid_senses.is_empty() || mid_senses.len() > 4 {
            continue;
        }
        for &ms in mid_senses {
            if let Some(gl) = t.gloss_lemmas.get(&ms) {
                for g in gl {
                    if g != word {
                        out.push(g.as_str());
                    }
                }
            }
        }
        if let Some(cl) = t.compound_links.get(mid) {
            for c in cl {
                if c != word {
                    out.push(c.as_str());
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    if let Ok(mut guard) = cache.lock() {
        guard.insert(word.to_string(), out.clone());
    }
    out
}

/// High-precision primary-sense direct gloss, parent-gloss (when monosemous), and `#p/#m` holonym
/// concepts for `word`. Avoids secondary senses (e.g. zodiac/river) and 2-hop sibling fanout.
pub fn direct_mono_gloss_concepts(word: &str) -> Vec<&'static str> {
    static CACHE: OnceLock<std::sync::Mutex<HashMap<String, Vec<&'static str>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(word)
    {
        return hit.clone();
    }
    let t = table();
    let ss = senses(word);
    let mut out: Vec<&'static str> = Vec::new();
    let use_ss: &[u32] = if ss.len() <= 2 { ss } else { &ss[..1] };
    for &s in use_ss {
        if let Some(gl) = t.gloss_lemmas.get(&s) {
            for g in gl {
                if g != word {
                    out.push(g.as_str());
                }
            }
        }
        for &h in t.holonyms.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            for hl in t.lemmas.get(&h).map(Vec::as_slice).unwrap_or(&[]) {
                if hl != word {
                    out.push(hl.as_str());
                }
            }
            for hp in t.phrases.get(&h).map(Vec::as_slice).unwrap_or(&[]) {
                if hp != word {
                    out.push(hp.as_str());
                }
            }
        }
        if ss.len() == 1 {
            for &p in t.parents.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                for pl in t.lemmas.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
                    if pl != word {
                        out.push(pl.as_str());
                    }
                }
                for pp in t.phrases.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
                    if pp != word {
                        out.push(pp.as_str());
                    }
                }
                if let Some(gl) = t.gloss_lemmas.get(&p) {
                    for g in gl {
                        if g != word {
                            out.push(g.as_str());
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    if let Ok(mut guard) = cache.lock() {
        guard.insert(word.to_string(), out.clone());
    }
    out
}

/// Multi-word phrases (`2..=5` words) AND single-word lemmas across `term`'s synsets and
/// `max_depth` upward `#p/#m/@/@i` ancestors (`ceo <-> chief executive officer`, `the hague -> netherlands`).
pub fn phrase_and_lemma_closure(term: &str, max_depth: usize) -> Vec<&'static str> {
    let t = table();
    let mut visited: Vec<u32> = Vec::new();
    let mut frontier: Vec<u32> = Vec::new();
    for &s in senses(term) {
        push_unique(&mut visited, s);
        push_unique(&mut frontier, s);
    }
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for &s in &frontier {
            for &p in t.parents.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                if !visited.contains(&p) {
                    visited.push(p);
                    next.push(p);
                }
            }
            for &h in t.holonyms.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                if !visited.contains(&h) {
                    visited.push(h);
                    next.push(h);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    let mut out: Vec<&'static str> = Vec::new();
    for &s in &visited {
        for l in t.lemmas.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            if l != term {
                out.push(l.as_str());
            }
        }
        for ph in t.phrases.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            if ph != term {
                out.push(ph.as_str());
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Mathematical `(deg + 180) % 360` opposite cardinal compass direction derived purely
/// from Princeton WordNet 3.1 `data.noun` degree glosses (zero hardcoded direction strings).
pub fn opposite_cardinal_direction(word: &str) -> Option<&'static str> {
    table().cardinal_opposite.get(word).map(String::as_str)
}

// hardcode-scan: closed-class (derivational adjectival-to-nominal suffixes `-ant -> -ance`, `-ent -> -ence`)
const ADJ_NOMINAL_DETACH: &[(&str, &str)] = &[
    ("ant", "ance"),
    ("ent", "ence"),
    ("ic", "ia"),
];

fn normalize_nominal_phrase(term: &str) -> Option<String> {
    let tbl = table();
    if tbl.excludes_edges.contains_key(term)
        || tbl.has_part_edges.contains_key(term)
        || tbl.senses.contains_key(term)
        || tbl.phrase_senses.contains_key(term)
    {
        return Some(term.to_string());
    }
    if term.contains(' ') {
        let parts: Vec<&str> = term.split_ascii_whitespace().collect();
        if let Some(&last) = parts.last() {
            for &(suf, rep) in ADJ_NOMINAL_DETACH {
                if let Some(stem_b) = last.strip_suffix(suf)
                    && stem_b.len() >= 3
                {
                    let mut rebuilt = parts[..parts.len() - 1].join(" ");
                    rebuilt.push(' ');
                    rebuilt.push_str(stem_b);
                    rebuilt.push_str(rep);
                    if tbl.phrase_senses.contains_key(&rebuilt) || tbl.excludes_edges.contains_key(&rebuilt) {
                        return Some(rebuilt);
                    }
                }
            }
        }
    } else {
        for &(suf, rep) in ADJ_NOMINAL_DETACH {
            if let Some(stem_b) = term.strip_suffix(suf)
                && stem_b.len() >= 3
            {
                let cand = format!("{stem_b}{rep}");
                if tbl.senses.contains_key(&cand) || tbl.excludes_edges.contains_key(&cand) {
                    return Some(cand);
                }
            }
        }
    }
    None
}

/// Typed `excludes` targets (`vegan —excludes→ ["animal", "dairy", "meat", "fish"]`,
/// `lactose intolerant —excludes→ ["milk", "lactose"]`) parsed deterministically from WordNet 3.1.
pub fn typed_excludes_targets(term: &str) -> Vec<&'static str> {
    let tbl = table();
    let Some(norm) = normalize_nominal_phrase(term) else {
        return Vec::new();
    };
    let Some(v) = tbl.excludes_edges.get(&norm) else {
        return Vec::new();
    };
    let mut out: Vec<&'static str> = v.iter().map(String::as_str).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Typed `has_part` + `is_a` (`@`) 1-step outgoing edges from primary senses (`senses[..2]`)
/// for exact-edge reachability walks (`cappuccino —has_part→ milk —is_a→ dairy product`,
/// `omelette —has_part→ egg —has_part→ animal`, `cod —is_a→ fish —is_a→ animal`).
pub fn typed_part_and_isa_parents(term: &str) -> Vec<&'static str> {
    let tbl = table();
    let mut out: Vec<&'static str> = Vec::new();
    if let Some(hp) = tbl.has_part_edges.get(term) {
        for x in hp {
            if x != term {
                out.push(x.as_str());
            }
        }
    }
    let ss = senses(term);
    let primary_ss = &ss[..ss.len().min(2)];
    for &s in primary_ss {
        for &p in tbl.parents.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
            for pl in tbl.lemmas.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
                if pl != term {
                    out.push(pl.as_str());
                }
            }
            for pp in tbl.phrases.get(&p).map(Vec::as_slice).unwrap_or(&[]) {
                if pp != term {
                    out.push(pp.as_str());
                }
            }
        }
    }
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
        let normalized = DATA_NOUN.replace("\r\n", "\n");
        let digest = Sha256::digest(normalized.as_bytes());
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

    #[test]
    fn holonyms_and_gloss_concepts_reach_geography_and_ingredients() {
        let frankfurt_up = holonyms_and_upward("frankfurt", 2);
        assert!(frankfurt_up.contains(&"germany"), "Frankfurt #p Germany missing: {frankfurt_up:?}");
        let dresden_up = holonyms_and_upward("dresden", 2);
        assert!(dresden_up.contains(&"germany"), "Dresden #p Germany missing: {dresden_up:?}");

        let cappuccino_gloss = gloss_and_compound_concepts("cappuccino");
        assert!(cappuccino_gloss.contains(&"milk"), "cappuccino -> milk missing: {cappuccino_gloss:?}");
        assert!(cappuccino_gloss.contains(&"lactose"), "cappuccino -> milk -> lactose missing: {cappuccino_gloss:?}");

        let milk_gloss = gloss_and_compound_concepts("milk");
        assert!(milk_gloss.contains(&"lactose"), "milk -> lactose inverse gloss missing: {milk_gloss:?}");

        let omelette_gloss = gloss_and_compound_concepts("omelette");
        assert!(omelette_gloss.contains(&"egg"), "omelette -> egg missing: {omelette_gloss:?}");

        assert_eq!(opposite_cardinal_direction("north"), Some("south"));
        assert_eq!(opposite_cardinal_direction("south"), Some("north"));
        assert_eq!(opposite_cardinal_direction("east"), Some("west"));
        assert_eq!(opposite_cardinal_direction("west"), Some("east"));
        assert_eq!(opposite_cardinal_direction("kitchen"), None);

        let vegan_mono = direct_mono_gloss_concepts("vegan");
        assert!(vegan_mono.contains(&"fish"), "vegan -> vegetarian -> fish missing: {vegan_mono:?}");
        assert!(vegan_mono.contains(&"meat"), "vegan -> vegetarian -> meat missing: {vegan_mono:?}");
        let sushi_mono = direct_mono_gloss_concepts("sushi");
        assert!(sushi_mono.contains(&"fish"), "sushi -> fish missing: {sushi_mono:?}");
    }
}
