//! Inverted postings and bridge/PPMI postings.

use crate::retrieval::*;

pub(in crate::retrieval) fn build_postings(bags: &[Bag], vocab_len: usize) -> Vec<Vec<u32>> {
    let mut postings = vec![Vec::new(); vocab_len];
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for (doc_idx, b) in bags.iter().enumerate() {
        seen.clear();
        for f in 0..3 {
            for (&t, &v) in &b.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
        }
        let d = doc_idx as u32;
        for &t in &seen {
            if let Some(list) = postings.get_mut(t as usize) {
                list.push(d);
            }
        }
    }
    postings
}

pub(in crate::retrieval) fn build_turn_postings(
    turn_bags: &[Bag],
    turn_own_bags: &[Bag],
    vocab_len: usize,
) -> Vec<Vec<u32>> {
    let mut postings = vec![Vec::new(); vocab_len];
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for (ti, (tb, ob)) in turn_bags.iter().zip(turn_own_bags.iter()).enumerate() {
        seen.clear();
        for f in 0..3 {
            for (&t, &v) in &tb.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
            for (&t, &v) in &ob.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
        }
        let d = ti as u32;
        for &t in &seen {
            if let Some(list) = postings.get_mut(t as usize) {
                list.push(d);
            }
        }
    }
    postings
}

pub(in crate::retrieval) fn collect_posting_candidates(postings: &[Vec<u32>], plan: &QueryPlan, total_docs: usize) -> Vec<usize> {
    if total_docs == 0 || plan.terms.is_empty() {
        return Vec::new();
    }
    let mut seen = vec![0u8; total_docs];
    let mut out = Vec::new();
    for qt in &plan.terms {
        if let Some(list) = postings.get(qt.id as usize) {
            for &doc in list {
                let idx = doc as usize;
                if seen[idx] == 0 {
                    seen[idx] = 1;
                    out.push(idx);
                }
            }
        }
    }
    out
}

/// Kernel tokens of `text` for recall-cue matching: the same tokenizer and stemmer as ranking.
pub(in crate::retrieval) fn cue_tokens(text: &str) -> Vec<String> {
    raw_tokens(text).into_iter().map(|w| stem(&w)).collect()
}

pub(in crate::retrieval) fn intern_cues(text: &str, vocab: &mut DetHashMap<String, u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = cue_tokens(text)
        .into_iter()
        .map(|w| {
            let next = vocab.len() as u32;
            *vocab.entry(w).or_insert(next)
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Ingest-time 1-hop WordNet/ConceptNet bridge postings and Haystack Positive PMI neighbors.
pub(in crate::retrieval) fn build_bridge_postings_and_ppmi(
    p: &Params,
    vocab: &Vocab,
    turns: &[TurnIx],
    t_own_stats: &Stats,
) -> (Vocab, Vec<Vec<u32>>, DetHashMap<u32, Vec<(u32, f32)>>) {
    // 1-hop ingest WordNet bridge postings in private `bridge_vocab` (`ix.vocab` untouched!).
    // Memoize WordNet + stem expansion once per unique rare `cid` (0-diff exact parity, ~5x faster).
    let n_turns_u32 = (t_own_stats.n as u32).max(1);
    let max_rare_df = (n_turns_u32 / 8).max(4);
    let mut bridge_vocab = Vocab::default();
    let bridge_df_cap = max_rare_df.min(24);
    let mut cid_to_bridge: DetHashMap<u32, Vec<u32>> = DetHashMap::default();
    if p.enable_hypernyms {
        thread_local! {
            static BRIDGE_LEMMA_CACHE: std::cell::RefCell<DetHashMap<(String, bool), Vec<String>>> =
                std::cell::RefCell::new(DetHashMap::default());
        }
        BRIDGE_LEMMA_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.len() >= 65_536 {
                cache.clear();
            }
            for (&cid, &df) in &t_own_stats.df {
                if df == 0 || df > bridge_df_cap {
                    continue;
                }
                let word = &vocab.names[cid as usize];
                if word.starts_with('\u{1}') {
                    continue;
                }
                let key = (word.clone(), p.enable_stemming);
                if !cache.contains_key(&key) {
                    let mut acc: Vec<String> = Vec::new();
                    let mut seen: DetHashSet<String> = DetHashSet::default();
                    let mut push_str = |s: &str| {
                        if seen.insert(s.to_string()) {
                            acc.push(s.to_string());
                        }
                        if p.enable_stemming {
                            let st = stem_term(&stem(s));
                            if seen.insert(st.clone()) {
                                acc.push(st);
                            }
                        }
                    };
                    for lemma in wordnet_lemmas_with_morph(word) {
                        push_str(lemma);
                    }
                    // Hovy 3-hop upward-only `@` hypernyms across surface + noun-singular forms
                    // (never bare suffix stems like "landed"->"land" which conflate verbs with nouns!).
                    let mut hyp_forms: Vec<String> = vec![word.clone()];
                    if let Some(sing) = word.strip_suffix("ies") {
                        if sing.len() >= 3 {
                            hyp_forms.push(format!("{sing}y"));
                        }
                    } else if !word.ends_with("ss")
                        && !word.ends_with("us")
                        && !word.ends_with("is")
                        && let Some(sing) = word.strip_suffix("es").or_else(|| word.strip_suffix('s'))
                        && sing.len() >= 4
                    {
                        hyp_forms.push(sing.to_string());
                    }
                    for form in &hyp_forms {
                        for lemma in crate::wordnet::upward_hypernyms(form, 3) {
                            push_str(lemma);
                        }
                    }
                    // #817: sha256-pinned ConceptNet 5.7.0 (AtLocation, UsedFor, CapableOf, HasA)
                    for (cn_lemma, _w) in crate::conceptnet::two_hop_neighbors(word, 4) {
                        push_str(cn_lemma);
                    }
                    cache.insert(key.clone(), acc);
                }
                if let Some(strs) = cache.get(&key) {
                    if !strs.is_empty() {
                        let mut bids = Vec::with_capacity(strs.len());
                        for s in strs {
                            bids.push(bridge_vocab.intern(s));
                        }
                        cid_to_bridge.insert(cid, bids);
                    }
                }
            }
        });
    }
    let mut turn_bridge_postings = vec![Vec::new(); bridge_vocab.names.len()];
    if !cid_to_bridge.is_empty() {
        let mut scratch: Vec<u32> = Vec::with_capacity(256);
        for (ti, tr) in turns.iter().enumerate() {
            if tr.is_backchannel {
                continue;
            }
            scratch.clear();
            for &cid in &tr.content {
                if let Some(bids) = cid_to_bridge.get(&cid) {
                    scratch.extend_from_slice(bids);
                }
            }
            if scratch.is_empty() {
                continue;
            }
            scratch.sort_unstable();
            scratch.dedup();
            for &bid in &scratch {
                if let Some(list) = turn_bridge_postings.get_mut(bid as usize) {
                    list.push(ti as u32);
                }
            }
        }
    }

    // Haystack Positive PMI associations over rare content tokens (`2 <= df <= max_rare_df`).
    let mut pair_counts: DetHashMap<(u32, u32), u32> = DetHashMap::default();
    for ti in 0..turns.len() {
        let mut rare_toks: Vec<u32> = turns[ti]
            .content
            .iter()
            .copied()
            .filter(|id| {
                let df = t_own_stats.df.get(id).copied().unwrap_or(0);
                df >= 2 && df <= max_rare_df
            })
            .collect();
        if let Some(ans_prev) = turns[ti].answers {
            for &id in &turns[ans_prev].content {
                let df = t_own_stats.df.get(&id).copied().unwrap_or(0);
                if df >= 2 && df <= max_rare_df {
                    rare_toks.push(id);
                }
            }
        }
        rare_toks.sort_unstable();
        rare_toks.dedup();
        if rare_toks.len() > 48 {
            rare_toks.truncate(48);
        }
        if rare_toks.len() <= 24 {
            for i in 0..rare_toks.len() {
                for j in (i + 1)..rare_toks.len() {
                    *pair_counts.entry((rare_toks[i], rare_toks[j])).or_insert(0) += 1;
                }
            }
        }
    }
    let mut ppmi_neighbors: DetHashMap<u32, Vec<(u32, f32)>> = DetHashMap::default();
    let n_f = t_own_stats.n.max(1.0);
    for ((u, v), c_uv) in pair_counts {
        if c_uv < 2 {
            continue;
        }
        let df_u = t_own_stats.df.get(&u).copied().unwrap_or(1) as f32;
        let df_v = t_own_stats.df.get(&v).copied().unwrap_or(1) as f32;
        let ppmi = ((c_uv as f32 * n_f) / (df_u * df_v)).log2();
        if ppmi >= 2.5 {
            let w = (ppmi / 10.0).clamp(0.18, 0.45);
            ppmi_neighbors.entry(u).or_default().push((v, w));
            ppmi_neighbors.entry(v).or_default().push((u, w));
        }
    }
    for list in ppmi_neighbors.values_mut() {
        list.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        list.truncate(6);
    }

    (bridge_vocab, turn_bridge_postings, ppmi_neighbors)
}
