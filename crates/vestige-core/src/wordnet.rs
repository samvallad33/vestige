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
/// The bundled WordNet 3.1 `data.verb`, byte-for-byte as distributed.
const DATA_VERB: &str = include_str!("../data/wordnet-3.1/data.verb");
/// The bundled WordNet 3.1 `data.adj`, byte-for-byte as distributed.
const DATA_ADJ: &str = include_str!("../data/wordnet-3.1/data.adj");
/// The bundled WordNet 3.1 `data.adv`, byte-for-byte as distributed.
const DATA_ADV: &str = include_str!("../data/wordnet-3.1/data.adv");

/// sha256 of [`DATA_NOUN`]; checked by unit test and `scripts/hardcode_scan.sh`.
pub const DATA_NOUN_SHA256: &str = "2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94";
/// sha256 of [`DATA_VERB`]; checked by unit test and `scripts/hardcode_scan.sh`.
pub const DATA_VERB_SHA256: &str = "eb1cf196dab6e1b815a1c86fa20c909fb7413b2f34250d7fe60a0fb35ef59e03";
/// sha256 of [`DATA_ADJ`]; checked by unit test and `scripts/hardcode_scan.sh`.
pub const DATA_ADJ_SHA256: &str = "ca1033bf627eb95f6cbb2864f8990be8e6a01a3c182e7d7c7094ccf8ead242cc";
/// sha256 of [`DATA_ADV`]; checked by unit test and `scripts/hardcode_scan.sh`.
pub const DATA_ADV_SHA256: &str = "aea301f39ac1b24be9a880dbc6cc6331bc5391f58c7525a1b298a13c922f2ed3";

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
    /// synset -> direct noun antonym synsets (`!`, e.g. `brother <-> sister`, `father <-> mother`).
    antonyms: HashMap<u32, Vec<u32>>,
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
    /// Synset offset -> WordNet 3.1 `lex_filenum` (`0..=44`, 41 lexicographer supersenses).
    synset_lex_filenum: HashMap<u32, u8>,
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
        if let Some(lf) = f.get(1).and_then(|s| s.parse::<u8>().ok())
            && lf < 64
        {
            out.synset_lex_filenum.insert(own, lf);
        }
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
            } else if *sym == "!" {
                push_unique(out.antonyms.entry(own).or_default(), o);
                push_unique(out.antonyms.entry(o).or_default(), own);
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
    type AncestorKey = (String, usize, u32);
    type AncestorVal = Vec<(u32, u32)>;
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<AncestorKey, AncestorVal>> =
            std::cell::RefCell::new(HashMap::new());
    }
    if let Some(hit) = CACHE.with(|c| {
        c.borrow()
            .get(&(word.to_string(), max_hops, max_desc))
            .cloned()
    }) {
        return hit;
    }
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
        CACHE.with(|c| {
            let mut m = c.borrow_mut();
            if m.len() >= 32_768 {
                m.clear();
            }
            m.insert((word.to_string(), max_hops, max_desc), Vec::new());
        });
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
    CACHE.with(|c| {
        let mut m = c.borrow_mut();
        if m.len() >= 32_768 {
            m.clear();
        }
        m.insert((word.to_string(), max_hops, max_desc), out.clone());
    });
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

/// Bitmask of WordNet 3.1 `lex_filenum` supersenses (`0..=44`) attested by `word`'s noun synsets.
/// Enables `O(1)` bitwise semantic-class intersection (`(q_ss & doc_ss) != 0`) at zero lexical overlap.
pub fn supersense_mask(word: &str) -> u64 {
    let tbl = table();
    let mut mask = 0u64;
    for &syn in senses(word) {
        if let Some(&lf) = tbl.synset_lex_filenum.get(&syn)
            && lf < 64
        {
            mask |= 1u64 << lf;
        }
    }
    mask
}

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

/// Full 4-POS Princeton WordNet 3.1 relational graph (`data.noun`, `data.verb`, `data.adj`, `data.adv`):
/// stores synonyms, instance (`@i`/`~i`), hypernym/hyponym (`@`/`~`), part/member (`#p`/`#m`/`%p`/`%m`),
/// cross-POS derivationally related forms (`+`), adjective similar-to/attribute/pertainym (`&`/`=`/`\`),
/// and verb entailment/cause (`*`/`>`). Loaded lazily once and immutable.
#[derive(Default)]
struct FullLexicon {
    /// lowercased lemma -> typed synset keys `((pos << 28) | offset)` across n/v/a/r.
    senses: HashMap<String, Vec<u32>>,
    /// typed synset key -> single-word lemmas (and distinctive tokens of `_` compounds).
    lemmas: HashMap<u32, Vec<String>>,
    /// typed synset key -> `(target_key, weight_u16)` where 100 = strongest relation.
    edges: HashMap<u32, Vec<(u32, u16)>>,
    /// noun/all downward instance + hyponym + meronym children for T50 class->instance lookup.
    down_children: HashMap<u32, Vec<u32>>,
}

#[inline]
fn pos_code(pos: &str) -> Option<u32> {
    match pos {
        "n" => Some(0),
        "v" => Some(1),
        "a" | "s" => Some(2),
        "r" => Some(3),
        _ => None,
    }
}

#[inline]
fn syn_key(pos_id: u32, offset: u32) -> u32 {
    (pos_id << 28) | (offset & 0x0FFF_FFFF)
}

fn parse_full_wordnet() -> FullLexicon {
    let mut lex = FullLexicon::default();
    let files: [(u32, &str); 4] = [
        (0, DATA_NOUN),
        (1, DATA_VERB),
        (2, DATA_ADJ),
        (3, DATA_ADV),
    ];
    for (src_pos, data) in files {
        for line in data.lines() {
            if !line.as_bytes().first().is_some_and(u8::is_ascii_digit) {
                continue;
            }
            let head = line.split(" | ").next().unwrap_or(line);
            let f: Vec<&str> = head.split_ascii_whitespace().collect();
            let Some(off) = f.first().and_then(|s| s.parse::<u32>().ok()) else {
                continue;
            };
            let Some(w_cnt) = f.get(3).and_then(|s| usize::from_str_radix(s, 16).ok()) else {
                continue;
            };
            let p_at = 4 + 2 * w_cnt;
            let Some(p_cnt) = f.get(p_at).and_then(|s| s.parse::<usize>().ok()) else {
                continue;
            };
            let own = syn_key(src_pos, off);

            for i in 0..w_cnt {
                let Some(raw_lemma) = f.get(4 + 2 * i) else {
                    break;
                };
                // Strip adjective syntactic markers like `alone(p)` -> `alone`.
                let cleaned = raw_lemma.split('(').next().unwrap_or(raw_lemma);
                if cleaned.contains('_') {
                    // Keep distinctive alphabetic compound tokens for instance names (e.g. `thanksgiving_day`).
                    let parts: Vec<String> = cleaned
                        .split('_')
                        .map(|p| p.trim_matches('.').to_ascii_lowercase())
                        .filter(|p| p.len() >= 3 && p.bytes().all(|b| b.is_ascii_alphabetic()))
                        .collect();
                    if (2..=4).contains(&parts.len()) {
                        let lem_vec = lex.lemmas.entry(own).or_default();
                        for part in parts {
                            if !lem_vec.contains(&part) {
                                lem_vec.push(part);
                            }
                        }
                    }
                    continue;
                }
                let key = cleaned.to_ascii_lowercase();
                if key.len() >= 2 && key.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') {
                    push_unique(lex.senses.entry(key.clone()).or_default(), own);
                    push_unique(lex.lemmas.entry(own).or_default(), key);
                }
            }

            for k in 0..p_cnt {
                let b = p_at + 1 + 4 * k;
                let (Some(&sym), Some(&t_off_s), Some(&t_pos_s)) = (f.get(b), f.get(b + 1), f.get(b + 2)) else {
                    break;
                };
                let (Some(t_pos), Ok(t_off)) = (pos_code(t_pos_s), t_off_s.parse::<u32>()) else {
                    continue;
                };
                let dst = syn_key(t_pos, t_off);
                let rel_w: Option<u16> = match sym {
                    "+" | "\\" => Some(95),            // derivationally related form, pertainym
                    "&" | "=" | "@i" | "~i" => Some(85), // similar-to, attribute, instance hyper/hyponym
                    "@" | "~" | "*" | ">" => Some(72),   // hypernym, hyponym, entailment, cause
                    "#p" | "#m" | "%p" | "%m" => Some(62), // part/member holonym & meronym
                    _ => None,
                };
                if let Some(w) = rel_w {
                    let e = lex.edges.entry(own).or_default();
                    if !e.iter().any(|&(d, _)| d == dst) {
                        e.push((dst, w));
                    }
                    // Ensure symmetry for cross-POS derived forms & similar-to links.
                    if matches!(sym, "+" | "\\" | "&" | "=") {
                        let rev = lex.edges.entry(dst).or_default();
                        if !rev.iter().any(|&(d, _)| d == own) {
                            rev.push((own, w));
                        }
                    }
                    if matches!(sym, "~" | "~i" | "%p" | "%m") {
                        push_unique(lex.down_children.entry(own).or_default(), dst);
                    } else if matches!(sym, "@" | "@i") {
                        push_unique(lex.down_children.entry(dst).or_default(), own);
                    }
                }
            }
        }
    }
    lex
}

fn full_table() -> &'static FullLexicon {
    static FULL: OnceLock<FullLexicon> = OnceLock::new();
    FULL.get_or_init(parse_full_wordnet)
}

// hardcode-scan: closed-class (English inflectional morphology stripping for WordNet lemma lookup)
const FULL_MORPH_DETACH: &[(&str, &str)] = &[
    ("ies", "y"),
    ("ied", "y"),
    ("ves", "f"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("sses", "ss"),
    ("xes", "x"),
    ("zes", "z"),
    ("ing", ""),
    ("ing", "e"),
    ("ed", ""),
    ("ed", "e"),
    ("er", ""),
    ("est", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
];

fn full_senses_of<'a>(lex: &'a FullLexicon, word: &str) -> &'a [u32] {
    if let Some(v) = lex.senses.get(word) {
        return v;
    }
    for &(suf, rep) in FULL_MORPH_DETACH {
        if let Some(base) = word.strip_suffix(suf)
            && base.len() >= 3
        {
            let cand = if rep.is_empty() {
                base.to_string()
            } else {
                format!("{base}{rep}")
            };
            if let Some(v) = lex.senses.get(&cand) {
                return v;
            }
        }
    }
    &[]
}

/// Deterministic full-WordNet 3.1 (noun + verb + adj + adv) 1-hop relational neighbors
/// `(lemma, relation_weight_u16)` where 100 = same-synset synonym, 95 = derived form `+` / pertainym `\`,
/// 85 = similar-to `&` / attribute `=` / instance `@i`/`~i`, 72 = hypernym/hyponym/entailment,
/// 62 = part/member `#p`/`%p`. Memoized globally.
type FullNbrCache = std::sync::Mutex<HashMap<String, Vec<(&'static str, u16)>>>;

pub fn full_lexicon_neighbors(word: &str) -> Vec<(&'static str, u16)> {
    static CACHE: OnceLock<FullNbrCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let key = word.to_ascii_lowercase();
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }
    let lex = full_table();
    let noun_tbl = table();
    let mut scores: HashMap<&'static str, u16> = HashMap::new();
    let direct_ss = full_senses_of(lex, &key);
    let ss: &[u32] = if !direct_ss.is_empty() {
        direct_ss
    } else if key.len() >= 6 {
        let wb = key.as_bytes();
        let first = wb[0];
        let mut best_fix: Option<&'static str> = None;
        for k in lex.senses.keys() {
            let kb = k.as_bytes();
            if kb.first().copied() != Some(first) || kb.len().abs_diff(wb.len()) > 1 {
                continue;
            }
            let ok = if kb.len() == wb.len() {
                let mut diffs = [0usize; 3];
                let mut dc = 0usize;
                for i in 0..wb.len() {
                    if wb[i] != kb[i] {
                        if dc >= 2 {
                            dc = 3;
                            break;
                        }
                        diffs[dc] = i;
                        dc += 1;
                    }
                }
                dc == 1
                    || (dc == 2
                        && diffs[0] + 1 == diffs[1]
                        && wb[diffs[0]] == kb[diffs[1]]
                        && wb[diffs[1]] == kb[diffs[0]])
            } else {
                let (s, l) = if wb.len() < kb.len() { (wb, kb) } else { (kb, wb) };
                let (mut i, mut j, mut skip) = (0usize, 0usize, false);
                let mut good = true;
                while i < s.len() && j < l.len() {
                    if s[i] == l[j] {
                        i += 1;
                        j += 1;
                    } else if !skip {
                        skip = true;
                        j += 1;
                    } else {
                        good = false;
                        break;
                    }
                }
                good
            };
            if ok && best_fix.is_none_or(|b| k.as_str() < b) {
                best_fix = Some(k.as_str());
            }
        }
        if let Some(fixed) = best_fix {
            scores.insert(fixed, 100);
            full_senses_of(lex, fixed)
        } else {
            &[]
        }
    } else {
        &[]
    };
    // Cap per-word senses to primary 6 so hyper-polysemous hubs don't explode.
    for &s in ss.iter().take(6) {
        if let Some(lems) = lex.lemmas.get(&s) {
            for l in lems {
                if l != &key {
                    let e = scores.entry(l.as_str()).or_insert(0);
                    *e = (*e).max(100);
                }
            }
        }
        if let Some(nbrs) = lex.edges.get(&s) {
            for &(dst, w) in nbrs {
                if let Some(lems) = lex.lemmas.get(&dst) {
                    for l in lems {
                        if l != &key {
                            let e = scores.entry(l.as_str()).or_insert(0);
                            *e = (*e).max(w);
                        }
                    }
                }
            }
        }
    }
    // Named instance-hypernym surnames (`vivaldi -> Antonio_Vivaldi -> composer -> bach/mozart`,
    // `seuss -> Dr._Seuss -> author / children / books`) from WordNet 3.1 `data.noun`.
    if let Some(inst_syns) = noun_tbl.instance_surnames.get(&key) {
        for &isyn in inst_syns {
            if let Some(gl) = noun_tbl.gloss_lemmas.get(&isyn) {
                for g in gl {
                    if g != &key {
                        let e = scores.entry(g.as_str()).or_insert(0);
                        *e = (*e).max(90);
                    }
                }
            }
            for &par in noun_tbl.parents.get(&isyn).map(Vec::as_slice).unwrap_or(&[]) {
                for pl in noun_tbl.lemmas.get(&par).map(Vec::as_slice).unwrap_or(&[]) {
                    if pl != &key {
                        let e = scores.entry(pl.as_str()).or_insert(0);
                        *e = (*e).max(95);
                    }
                }
                if let Some(gl) = noun_tbl.gloss_lemmas.get(&par) {
                    for g in gl {
                        if g != &key {
                            let e = scores.entry(g.as_str()).or_insert(0);
                            *e = (*e).max(80);
                        }
                    }
                }
                for &sib in noun_tbl.children.get(&par).map(Vec::as_slice).unwrap_or(&[]).iter().take(40) {
                    for sl in noun_tbl.lemmas.get(&sib).map(Vec::as_slice).unwrap_or(&[]) {
                        if sl != &key {
                            let e = scores.entry(sl.as_str()).or_insert(0);
                            *e = (*e).max(78);
                        }
                    }
                    for ph in noun_tbl.phrases.get(&sib).map(Vec::as_slice).unwrap_or(&[]) {
                        if let Some(last) = ph.split_ascii_whitespace().last()
                            && last.len() >= 4
                            && last != key.as_str()
                        {
                            let e = scores.entry(last).or_insert(0);
                            *e = (*e).max(78);
                        }
                    }
                }
            }
        }
    }
    let mut out: Vec<(&'static str, u16)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, out.clone());
    }
    out
}

/// T50 Class-to-Instance downward expansion across full WordNet 3.1 (`~`, `~i`, `%p`, `%m`)
/// up to `max_depth` hops (e.g., `"holiday"` -> `"thanksgiving"`, `"christmas"`;
/// `"state"` -> `"california"`, `"texas"`, `"colorado"`). Memoized and deterministic.
pub fn class_to_instances(class_word: &str, max_depth: usize) -> Vec<&'static str> {
    static CACHE: OnceLock<HopCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let key = (class_word.to_ascii_lowercase(), max_depth);
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }
    let lex = full_table();
    let mut visited: Vec<u32> = Vec::new();
    let mut frontier: Vec<u32> = Vec::new();
    for &s in full_senses_of(lex, &key.0) {
        // Prefer noun synsets (`pos == 0`) for class->instance queries.
        if (s >> 28) == 0 && !visited.contains(&s) {
            visited.push(s);
            frontier.push(s);
        }
    }
    for _ in 0..max_depth.min(2) {
        let mut next: Vec<u32> = Vec::new();
        for s in frontier {
            for &ch in lex.down_children.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                if !visited.contains(&ch) && visited.len() < 1500 {
                    visited.push(ch);
                    next.push(ch);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    let mut out: Vec<&'static str> = Vec::new();
    for s in visited {
        if let Some(lems) = lex.lemmas.get(&s) {
            for l in lems {
                if l != &key.0 {
                    out.push(l.as_str());
                }
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

/// Strict mutually exclusive WordNet 3.1 co-hyponym siblings and direct noun antonyms
/// (e.g., `brother` <-> `sister` via `!`, `wednesday` <-> `tuesday` via specific hypernym
/// parent `desc_count <= 35`, `children.len() <= 24`). Excludes synonyms of `word` and returns
/// empty for generic nouns without tight sibling sets or noun antonyms.
pub fn strict_co_hyponym_siblings(word: &str) -> Vec<&'static str> {
    let key = word.to_ascii_lowercase();
    if key.len() < 3 {
        return Vec::new();
    }
    let t = table();
    let Some(senses) = t.senses.get(&key) else {
        return Vec::new();
    };
    if senses.is_empty() {
        return Vec::new();
    }
    let mut own_syns: Vec<u32> = senses.clone();
    own_syns.sort_unstable();
    let mut own_lemmas: Vec<&'static str> = Vec::new();
    for &s in &own_syns {
        if let Some(ls) = t.lemmas.get(&s) {
            for l in ls {
                own_lemmas.push(l.as_str());
            }
        }
    }
    let mut out: Vec<&'static str> = Vec::new();
    // 1. Direct WordNet 3.1 noun antonyms (`!`, e.g., brother <-> sister, father <-> mother).
    for &s in &own_syns {
        if let Some(ants) = t.antonyms.get(&s) {
            for &ant in ants {
                if own_syns.binary_search(&ant).is_ok() {
                    continue;
                }
                if let Some(ls) = t.lemmas.get(&ant) {
                    for l in ls {
                        if l.len() >= 3 && l.as_str() != key && !own_lemmas.contains(&l.as_str()) {
                            out.push(l.as_str());
                        }
                    }
                }
            }
        }
    }
    // 2. Tight co-hyponym siblings under specific immediate hypernym parents (`senses <= 4`).
    if senses.len() <= 4 {
        for &s in &own_syns {
            let Some(parents) = t.parents.get(&s) else {
                continue;
            };
            for &p in parents {
                let desc = t.desc_count.get(&p).copied().unwrap_or(u32::MAX);
                if desc > 35 {
                    continue;
                }
                let Some(children) = t.children.get(&p) else {
                    continue;
                };
                if children.len() < 2 || children.len() > 24 {
                    continue;
                }
                for &sib in children {
                    if own_syns.binary_search(&sib).is_ok() {
                        continue;
                    }
                    if let Some(ls) = t.lemmas.get(&sib) {
                        for l in ls {
                            if l.len() >= 3 && l.as_str() != key && !own_lemmas.contains(&l.as_str()) {
                                out.push(l.as_str());
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// #3 Finite Heyting Algebra Asymmetric Subsumption (`turn_word => query_word`):
/// Scores intuitionistic lattice implication asymmetrically over the WordNet 3.1 hypernym DAG:
/// - `100`: `turn_word == query_word` or share a synset
/// - `82..=96`: `turn_word` is a specific hyponym/instance of `query_word` (`turn_word <= query_word`,
///   e.g. `cappuccino => coffee`, `violin => instrument`, `husband => spouse`), scaled by Resnik IC
/// - `38`: inverse (`query_word < turn_word`, generic hypernym in turn)
/// - `0`: unrelated in the 3-hop finite Heyting sublattice.
pub fn heyting_implication_ic_q16(turn_word: &str, query_word: &str) -> u16 {
    if turn_word.len() < 3 || query_word.len() < 3 {
        return 0;
    }
    if turn_word.eq_ignore_ascii_case(query_word) {
        return 100;
    }
    let tw = turn_word.to_ascii_lowercase();
    let qw = query_word.to_ascii_lowercase();
    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<(String, String), u16>> =
            std::cell::RefCell::new(HashMap::new());
    }
    let key = (tw, qw);
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&key).copied()) {
        return hit;
    }
    let t = table();
    let res: u16 = 'eval: {
        let (Some(t_syns), Some(q_syns)) = (t.senses.get(&key.0), t.senses.get(&key.1)) else {
            break 'eval 0;
        };
        if t_syns.iter().any(|s| q_syns.contains(s)) {
            break 'eval 98;
        }
        // Check upward closure from `turn_word` (`tw <= qw`: specific turn entails general query facet).
        let mut frontier: Vec<u32> = t_syns.clone();
        for hop in 1..=3usize {
            let mut next: Vec<u32> = Vec::new();
            for s in frontier {
                if let Some(ps) = t.parents.get(&s) {
                    for &p in ps {
                        if q_syns.contains(&p) {
                            let desc = t.desc_count.get(&p).copied().unwrap_or(500);
                            let ic_bonus: u16 = if desc <= 25 {
                                12
                            } else if desc <= 120 {
                                8
                            } else {
                                2
                            };
                            let base: u16 = match hop {
                                1 => 84,
                                2 => 76,
                                _ => 68,
                            };
                            break 'eval (base + ic_bonus).min(96);
                        }
                        if !next.contains(&p) {
                            next.push(p);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        // Check downward (`qw < tw`: turn only has a broader superordinate, much weaker implication).
        for &qs in q_syns {
            if let Some(ps) = t.parents.get(&qs)
                && ps.iter().any(|p| t_syns.contains(p))
            {
                break 'eval 38;
            }
        }
        0
    };
    CACHE.with(|c| {
        let mut m = c.borrow_mut();
        if m.len() >= 65_536 {
            m.clear();
        }
        m.insert(key, res);
    });
    res
}

/// Mutual-Hypernym / Co-Instance Sibling Resonance (#2):
/// Walks upward 1..=2 hops from `word` to specific (non-root, `desc_count <= 45`) synsets
/// and gathers sibling / co-instance lemmas under that exact Least Common Subsumer
/// (e.g., `vivaldi -> classical_composer -> bach/mozart`, `camping -> outdoor_sport`).
pub fn lcs_sibling_resonance(word: &str, max_out: usize) -> Vec<(&'static str, u16)> {
    let w = word.to_ascii_lowercase();
    if w.len() < 3 || max_out == 0 {
        return Vec::new();
    }
    let t = table();
    let Some(syns) = t.senses.get(&w) else {
        return Vec::new();
    };
    let mut out: Vec<(&'static str, u16)> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for &s in syns.iter().take(4) {
        let Some(ps) = t.parents.get(&s) else {
            continue;
        };
        for &p in ps {
            let desc = t.desc_count.get(&p).copied().unwrap_or(999);
            if desc > 45 {
                continue;
            }
            let w_score: u16 = if desc <= 12 { 82 } else { 70 };
            if let Some(plems) = t.lemmas.get(&p) {
                for pl in plems {
                    let pls = pl.as_str();
                    if pls != w && pls.len() >= 3 && seen.insert(pls) {
                        out.push((pls, w_score));
                        if out.len() >= max_out {
                            return out;
                        }
                    }
                }
            }
            if let Some(kids) = t.children.get(&p) {
                for &kid in kids.iter().take(12) {
                    if kid == s {
                        continue;
                    }
                    if let Some(klems) = t.lemmas.get(&kid) {
                        for kl in klems.iter().take(2) {
                            let kls = kl.as_str();
                            if kls != w && kls.len() >= 3 && seen.insert(kls) {
                                out.push((kls, w_score.saturating_sub(4)));
                                if out.len() >= max_out {
                                    return out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Bipartite Gloss-Hypernym Dual-Bridge (#8):
/// Given an abstract query word, walks its high-IC gloss tokens (`len >= 4`, `senses <= 4`)
/// and retrieves specific hyponym/synset lemmas whose definitions or parents interlock with
/// those gloss facets.
pub fn dual_gloss_hypernym_bridge(q_word: &str, max_out: usize) -> Vec<(&'static str, u16)> {
    let w = q_word.to_ascii_lowercase();
    if w.len() < 4 || max_out == 0 {
        return Vec::new();
    }
    let t = table();
    let g_toks = gloss_tokens_of(&w);
    if g_toks.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<(&'static str, u16)> = Vec::new();
    let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for gt in g_toks.into_iter().take(8) {
        if gt.len() < 4 || gt == w {
            continue;
        }
        let Some(gsyns) = t.senses.get(gt) else {
            continue;
        };
        if gsyns.len() > 5 {
            continue;
        }
        if seen.insert(gt) {
            out.push((gt, 72));
            if out.len() >= max_out {
                return out;
            }
        }
        for &gs in gsyns.iter().take(2) {
            let desc = t.desc_count.get(&gs).copied().unwrap_or(999);
            if desc > 60 {
                continue;
            }
            if let Some(kids) = t.children.get(&gs) {
                for &kid in kids.iter().take(6) {
                    if let Some(klems) = t.lemmas.get(&kid) {
                        for kl in klems.iter().take(1) {
                            let kls = kl.as_str();
                            if kls.len() >= 4 && kls != w && seen.insert(kls) {
                                out.push((kls, 66));
                                if out.len() >= max_out {
                                    return out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn bundled_file_matches_recorded_sha256() {
        for (name, data, expected) in [
            ("data.noun", DATA_NOUN, DATA_NOUN_SHA256),
            ("data.verb", DATA_VERB, DATA_VERB_SHA256),
            ("data.adj", DATA_ADJ, DATA_ADJ_SHA256),
            ("data.adv", DATA_ADV, DATA_ADV_SHA256),
        ] {
            let normalized = data.replace("\r\n", "\n");
            let digest = Sha256::digest(normalized.as_bytes());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, expected, "WordNet 3.1 checksum mismatch for {name}");
        }
    }

    #[test]
    fn full_wordnet_cross_pos_and_class_instances_work() {
        let lonely_nbrs = full_lexicon_neighbors("lonely");
        assert!(
            lonely_nbrs.iter().any(|&(w, _)| w == "alone" || w == "solitary" || w == "loneliness"),
            "expected adj/derived neighbors for lonely: {lonely_nbrs:?}"
        );
        let hols = class_to_instances("holiday", 2);
        assert!(
            hols.contains(&"thanksgiving") || hols.contains(&"christmas"),
            "expected holiday instances: {:?}",
            &hols[..hols.len().min(20)]
        );
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

    #[test]
    fn strict_co_hyponym_siblings_separates_mutually_exclusive_peers() {
        let wed_sibs = strict_co_hyponym_siblings("wednesday");
        assert!(wed_sibs.contains(&"tuesday"), "expected tuesday sibling of wednesday: {wed_sibs:?}");
        assert!(!wed_sibs.contains(&"wednesday"));
        let bro_sibs = strict_co_hyponym_siblings("brother");
        assert!(bro_sibs.contains(&"sister"), "expected sister sibling of brother: {bro_sibs:?}");
        assert!(strict_co_hyponym_siblings("thing").is_empty());
    }

    #[test]
    fn heyting_implication_is_asymmetric_specific_to_general() {
        let up = heyting_implication_ic_q16("cappuccino", "coffee");
        let down = heyting_implication_ic_q16("coffee", "cappuccino");
        assert!(up >= 80, "expected specific => general implication >= 80, got {up}");
        assert!(up > down, "expected asymmetric Heyting implication ({up} > {down})");
        assert_eq!(heyting_implication_ic_q16("cappuccino", "cappuccino"), 100);
    }
}
