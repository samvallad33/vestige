//! Zero-overlap bridge lane: WordNet morphology lookup, GhostLink damping, `t_bridge` scorer and protected tail admission.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use crate::retrieval::*;

/// Morphology-aware WordNet 1-hop expansion: checks exact word, singular `-s`/`-es` strip,
/// and deterministic suffix `stem(&word)` so plural/inflected nouns ("macarons", "walnuts")
/// resolve to their canonical WordNet synsets.
pub(in crate::retrieval) fn wordnet_lemmas_with_morph(word: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = crate::wordnet::expansion_lemmas(word).to_vec();
    let mut push_all = |cand: &str| {
        for l in crate::wordnet::expansion_lemmas(cand) {
            if !out.contains(&l) {
                out.push(l);
            }
        }
        for l in crate::wordnet::upward_hypernyms(cand, 3) {
            if !out.contains(&l) {
                out.push(l);
            }
        }
        for (l, _) in crate::conceptnet::two_hop_neighbors(cand, 8) {
            if !out.contains(&l) {
                out.push(l);
            }
        }
    };
    push_all(word);
    if let Some(sing) = word.strip_suffix("es").or_else(|| word.strip_suffix('s')) {
        if sing.len() >= 3 {
            push_all(sing);
        }
    }
    let st = stem(word);
    if st != word && st.len() >= 3 {
        push_all(&st);
    }
    out
}

pub(in crate::retrieval) fn ghostlink_hub_damp(deg: u32, own: u32) -> f32 {
    let extra = deg.saturating_sub(own);
    1.0 / (2.0 + extra as f32).log2()
}

pub(in crate::retrieval) fn ochiai_divergence(a: &[u32], b: &[u32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let denom = ((a.len() * b.len()) as f32).sqrt();
    if denom <= 0.0 {
        None
    } else {
        Some((1.0 - (inter as f32) / denom).clamp(0.0, 1.0))
    }
}

/// Isolated 2-hop Meet-in-the-Middle WordNet + Haystack PPMI channel (`t_bridge`).
/// Moved verbatim out of `rank_one`. Reads only `&Index`, the query, params and plan; it never
/// sees `t_own`, `t_raw`, `t_scene` or `s_score`.
pub(in crate::retrieval) fn score_bridge(ix: &Index, q: &Query, p: &Params, plan: &QueryPlan, nt: usize) -> (Vec<f32>, Vec<u8>) {
    let mut t_bridge = vec![0.0f32; nt];
    let mut t_bridge_hits = vec![0u8; nt];
    if p.enable_hypernyms {
        let n_turns_u32 = (ix.t_own_stats.n as u32).max(1);
        let max_seed_df = (n_turns_u32 / 8).max(3);
        let max_bridge_hub_df = ((ix.t_own_stats.n as usize) / 16).clamp(2, 12);
        let max_genus_hub_df = ((ix.t_own_stats.n as usize) / 10).clamp(4, 28);

        let mut seeds: Vec<(u32, String)> = raw_tokens(&q.text)
            .into_iter()
            .filter(|w| is_content(w))
            .map(|w| {
                let df = ix.vocab.get(&w).and_then(|id| ix.t_own_stats.df.get(&id).copied()).unwrap_or(0);
                (df, w)
            })
            .filter(|(df, _)| crate::zero_overlap::term_rarity_gate(*df, max_seed_df))
            .collect();
        seeds.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        seeds.dedup_by(|a, b| a.1 == b.1);
        seeds.truncate(4);

        let exact_q_ids: DetHashSet<u32> = plan.terms.iter().map(|qt| qt.id).collect();
        let mut direct_bridge_terms: DetHashMap<u32, (f32, u8)> = DetHashMap::default();
        let mut genus_bridge_terms: DetHashMap<u32, (f32, u8)> = DetHashMap::default();
        let mut meet_bridge_terms: DetHashMap<u32, (f32, u8)> = DetHashMap::default();

        for (s_idx, (seed_df, w)) in seeds.iter().enumerate() {
            let bit = 1u8 << (s_idx.min(7));
            // 0. Direct OOV Genus -> Instance lookup (`*seed_df == 0` zero-overlap gate!):
            //    When a query genus word `w` ("activities", "career", "hobbies") never occurs verbatim
            //    in the corpus (`df == 0`), match it directly against upward hypernyms in `ix.bridge_vocab`.
            if *seed_df == 0 {
                let mut genus_keys: Vec<String> = vec![w.clone()];
                if let Some(sing) = w.strip_suffix("ies") {
                    if sing.len() >= 3 {
                        genus_keys.push(format!("{sing}y"));
                    }
                } else if let Some(sing) = w.strip_suffix("es").or_else(|| w.strip_suffix('s')) {
                    if sing.len() >= 3 {
                        genus_keys.push(sing.to_string());
                    }
                }
                if p.enable_stemming {
                    genus_keys.push(stem_term(&stem(w)));
                }
                for gk in &genus_keys {
                    if let Some(gid) = ix.bridge_vocab.get(gk) {
                        let entry = genus_bridge_terms.entry(gid).or_insert((0.0, 0));
                        entry.0 = entry.0.max(1.0);
                        entry.1 |= bit;
                    }
                }
            }
            if let Some(qid) = ix.vocab.get(w) {
                if let Some(ppmi_list) = ix.ppmi_neighbors.get(&qid) {
                    for &(pid, pw) in ppmi_list {
                        if !exact_q_ids.contains(&pid) {
                            let entry = direct_bridge_terms.entry(pid).or_insert((0.0, 0));
                            entry.0 = entry.0.max(pw);
                            entry.1 |= bit;
                        }
                    }
                }
            }
            for lemma in wordnet_lemmas_with_morph(w) {
                let st_lemma = stem_term(&stem(lemma));
                // 1. Direct corpus vocabulary (`ix.vocab`)
                if let Some(lid) = ix.vocab.get(lemma) {
                    if !exact_q_ids.contains(&lid) {
                        let entry = direct_bridge_terms.entry(lid).or_insert((0.0, 0));
                        entry.0 = entry.0.max(p.hypernym_weight);
                        entry.1 |= bit;
                    }
                }
                if p.enable_stemming {
                    if let Some(sid) = ix.vocab.get(&st_lemma) {
                        if !exact_q_ids.contains(&sid) {
                            let entry = direct_bridge_terms.entry(sid).or_insert((0.0, 0));
                            entry.0 = entry.0.max(p.hypernym_weight * 0.85);
                            entry.1 |= bit;
                        }
                    }
                }
                // 2. Private ingest-bridge vocabulary (`ix.bridge_vocab`)
                if let Some(blid) = ix.bridge_vocab.get(lemma) {
                    let entry = meet_bridge_terms.entry(blid).or_insert((0.0, 0));
                    entry.0 = entry.0.max(p.hypernym_weight);
                    entry.1 |= bit;
                }
                if p.enable_stemming {
                    if let Some(bsid) = ix.bridge_vocab.get(&st_lemma) {
                        let entry = meet_bridge_terms.entry(bsid).or_insert((0.0, 0));
                        entry.0 = entry.0.max(p.hypernym_weight * 0.85);
                        entry.1 |= bit;
                    }
                }
            }
        }

        let mut seed_mask_hit = vec![0u8; nt];
        for (&bid, &(bw, s_bit)) in &direct_bridge_terms {
            if let Some(list) = ix.t_postings.get(bid as usize) {
                if !list.is_empty() && list.len() <= max_bridge_hub_df {
                    let damp = ghostlink_hub_damp(list.len() as u32, 1);
                    let idf = ((ix.t_own_stats.n + 1.0) / (list.len() as f32 + 1.0)).ln().max(0.5);
                    for &doc in list {
                        let ti = doc as usize;
                        t_bridge[ti] += bw * idf * damp;
                        seed_mask_hit[ti] |= s_bit;
                    }
                }
            }
        }
        for (&bid, &(bw, s_bit)) in &genus_bridge_terms {
            if let Some(blist) = ix.turn_bridge_postings.get(bid as usize) {
                if !blist.is_empty() && blist.len() <= max_genus_hub_df {
                    let damp = ghostlink_hub_damp(blist.len() as u32, 1);
                    let idf = ((ix.t_own_stats.n + 1.0) / (blist.len() as f32 + 1.0)).ln().max(0.5);
                    for &doc in blist {
                        let ti = doc as usize;
                        t_bridge[ti] += 0.90 * bw * idf * damp;
                        seed_mask_hit[ti] |= s_bit;
                    }
                }
            }
        }
        for (&bid, &(bw, s_bit)) in &meet_bridge_terms {
            if let Some(blist) = ix.turn_bridge_postings.get(bid as usize) {
                if !blist.is_empty() && blist.len() <= max_bridge_hub_df {
                    let damp = ghostlink_hub_damp(blist.len() as u32, 1);
                    let idf = ((ix.t_own_stats.n + 1.0) / (blist.len() as f32 + 1.0)).ln().max(0.5);
                    for &doc in blist {
                        let ti = doc as usize;
                        t_bridge[ti] += 0.65 * bw * idf * damp;
                        seed_mask_hit[ti] |= s_bit;
                    }
                }
            }
        }
        for ti in 0..nt {
            if t_bridge[ti] > 0.0 {
                let len_norm = ((ix.turns[ti].content.len() + 6) as f32).sqrt();
                t_bridge[ti] /= len_norm;
                t_bridge_hits[ti] = seed_mask_hit[ti].count_ones() as u8;
            }
        }
    }
    (t_bridge, t_bridge_hits)
}

/// Post-Gatekeeper Sealed-Head Bridge Tail Admission, moved verbatim out of `rank_one`.
/// Slots `0..5` stay sealed; only slot `k - 1` (or an unfilled tail) can take a bridge turn.
pub(in crate::retrieval) fn admit_protected_tail(
    ix: &Index,
    p: &Params,
    k: usize,
    nt: usize,
    t_bridge: &[f32],
    t_bridge_hits: &[u8],
    t_exact_mask: &[u64],
    t_scene: &[f32],
    final_turn_indices: &mut Vec<usize>,
) {
    if p.enable_hypernyms && k >= 6 {
        let best_bridge = (0..nt)
            .filter(|&ti| {
                t_bridge[ti] > 0.0
                    && t_exact_mask[ti] == 0
                    && ix.turns[ti].field == 0
                    && !ix.turns[ti].is_question
                    && !ix.turns[ti].is_backchannel
                    && !turn_is_superseded(ix, ti)
                    && !final_turn_indices.contains(&ti)
            })
            .max_by(|&a, &b| {
                t_bridge_hits[a]
                    .cmp(&t_bridge_hits[b])
                    .then(t_bridge[a].partial_cmp(&t_bridge[b]).unwrap_or(std::cmp::Ordering::Equal))
                    .then(b.cmp(&a))
            });

        if let Some(b_ti) = best_bridge {
            if final_turn_indices.len() < k {
                final_turn_indices.push(b_ti);
            } else {
                let last_slot = k - 1;
                let vti = final_turn_indices[last_slot];
                let v_sess = ix.turns[vti].session;
                let dup_sess = final_turn_indices[..last_slot]
                    .iter()
                    .any(|&earlier| ix.turns[earlier].session == v_sess);
                if (dup_sess && ix.turns[vti].field == 1) || t_scene[vti] <= 0.0 {
                    final_turn_indices[last_slot] = b_ti;
                }
            }
        }
    }
}
