//! Kernel 5: MWE Masking + Asymmetric SYNAPSE + CABLE Abstract Ingest Tags (`mwe_synapse_cable_q16.rs`).
//!
//! References:
//! - LiteSemRAG (arXiv:2604.16350): Multi-word expression masking before ambiguous unigram tokenization.
//! - SYNAPSE (arXiv:2601.02744): Asymmetric directional edge weights (0.85 attr/hyponym vs 0.25 hypernym),
//!   fan dilution `1/deg`, and Top-7 Winner-Take-All (WTA) lateral inhibition.
//! - CABLE (arXiv:2608.17911): Ingest-time zero-overlap abstract ancestor/concept tags (`0.65x`).

use crate::retrieval::{DetHashMap, DetHashSet, Feat, Index, Vocab, is_content, raw_tokens};
use crate::wordnet;

/// Asymmetric attribute / hyponym weight (`0.85` in Q16).
const ASYM_HYPONYM_Q16: u64 = 55_706;
/// Asymmetric hypernym upward weight (`0.25` in Q16).
const ASYM_HYPERNYM_Q16: u64 = 16_384;
/// Winner-Take-All top-K active neighbor fanout cap (`7` per SYNAPSE paper).
const SYNAPSE_WTA_CAP: usize = 7;

/// Identifies WordNet multi-word phrases (`w_i w_{i+1}`) in a raw token slice and returns
/// `(mwe_phrases, masked_token_indices)` so isolated ambiguous head unigrams inside an MWE
/// (e.g., `"lean"` inside `"political leaning"`, `"park"` inside `"theme park"`) can be masked.
pub(in crate::retrieval) fn detect_mwe_masked_indices(tokens: &[String]) -> (Vec<String>, DetHashSet<usize>) {
    let mut phrases = Vec::new();
    let mut masked = DetHashSet::default();
    let mut i = 0usize;
    while i + 1 < tokens.len() {
        let a = &tokens[i];
        let b = &tokens[i + 1];
        if a.len() >= 3 && b.len() >= 3 && is_content(a) && is_content(b) {
            let cand = format!("{a} {b}");
            if !wordnet::senses(&cand).is_empty() {
                phrases.push(format!("{a}_{b}"));
                masked.insert(i);
                masked.insert(i + 1);
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    (phrases, masked)
}

/// Emits CABLE-style abstract ingest features (`0.65x` = `42598/65536`) for rare declarative
/// content tokens using WordNet specific ancestor synsets & MWE compound identifiers so
/// zero-overlap commonsense & preference queries connect at ingest without adding runtime vectors.
pub(in crate::retrieval) fn emit_mwe_and_cable_feats(
    vocab: &mut Vocab,
    tokens: &[String],
    enable_cable_tags: bool,
) -> (Vec<Feat>, DetHashSet<String>) {
    let (mwes, masked_idx) = detect_mwe_masked_indices(tokens);
    let mut masked_words: DetHashSet<String> = DetHashSet::default();
    let mut out = Vec::new();

    for idx in &masked_idx {
        masked_words.insert(tokens[*idx].clone());
    }
    for mwe in mwes {
        let id = vocab.intern(&format!("mwe:{mwe}"));
        out.push(Feat {
            id,
            w: 0.75,
            counts_len: false,
            inheritable: false,
        });
    }

    if enable_cable_tags {
        let mut emitted = DetHashSet::default();
        let mut count = 0usize;
        for (i, tok) in tokens.iter().enumerate() {
            if masked_idx.contains(&i) || tok.len() < 4 || !is_content(tok) {
                continue;
            }
            for (anc_id, _) in wordnet::specific_ancestor_synsets(tok, 2, 96).into_iter().take(2) {
                let tag = format!("cable:wn:{anc_id}");
                if emitted.insert(tag.clone()) {
                    let id = vocab.intern(&tag);
                    out.push(Feat {
                        id,
                        w: 0.65,
                        counts_len: false,
                        inheritable: false,
                    });
                    count += 1;
                    if count >= 6 {
                        return (out, masked_words);
                    }
                }
            }
        }
    }
    (out, masked_words)
}

/// Asymmetric SYNAPSE spreading activation with `1/deg` fan dilution, asymmetric edge weights
/// (`0.85` attribute/hyponym vs `0.25` hypernym), and Top-7 Winner-Take-All lateral inhibition.
pub(in crate::retrieval) fn asymmetric_synapse_wta_q16(
    ix: &Index,
    q_text: &str,
    cand_turns: &[(usize, u64)],
) -> Vec<(usize, u64)> {
    let n = cand_turns.len().min(28);
    if n <= 1 {
        return cand_turns.to_vec();
    }
    let q_toks: Vec<String> = raw_tokens(q_text)
        .into_iter()
        .filter(|w| w.len() >= 3 && is_content(w))
        .collect();
    let (q_mwes, _) = detect_mwe_masked_indices(&q_toks);
    let q_mwe_ids: DetHashSet<u32> = q_mwes
        .iter()
        .filter_map(|m| ix.vocab.get(&format!("mwe:{m}")))
        .collect();

    // Directed weighted adjacency (u -> v, weight_q16)
    let max_df = (ix.turns.len() as u32 / 10).clamp(4, 24);
    let mut term_turns: DetHashMap<u32, Vec<usize>> = DetHashMap::default();
    for (i, &(ti, _)) in cand_turns.iter().take(n).enumerate() {
        for &cid in &ix.turns[ti].content {
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
            if df >= 1 && df <= max_df {
                term_turns.entry(cid).or_default().push(i);
            }
        }
        for feat in &ix.turns[ti].feats {
            if let Some(name) = ix.vocab.names.get(feat.id as usize)
                && (name.starts_with("mwe:") || name.starts_with("cable:"))
            {
                term_turns.entry(feat.id).or_default().push(i);
            }
        }
    }

    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];
    for (tid, members) in term_turns {
        if members.len() < 2 || members.len() > 8 {
            continue;
        }
        let fan_div = members.len() as u64;
        for &u in &members {
            let u_is_decl = !ix.turns[cand_turns[u].0].is_question;
            for &v in &members {
                if u == v {
                    continue;
                }
                let v_is_decl = !ix.turns[cand_turns[v].0].is_question;
                let base_asym = if v_is_decl && !u_is_decl {
                    ASYM_HYPONYM_Q16 // question -> answer hyponym/attribute direction (0.85)
                } else if !v_is_decl && u_is_decl {
                    ASYM_HYPERNYM_Q16 // answer -> question weaker upward path (0.25)
                } else {
                    (ASYM_HYPONYM_Q16 + ASYM_HYPERNYM_Q16) >> 1
                };
                let diluted = (base_asym / fan_div.max(1)).max(4_096);
                let w = if q_mwe_ids.contains(&tid) {
                    diluted.saturating_add(diluted >> 1)
                } else {
                    diluted
                };
                adj[u].push((v, w));
            }
        }
    }

    // Lateral inhibition with Top-7 Winner-Take-All cap per hop
    let mut act: Vec<u64> = cand_turns
        .iter()
        .take(n)
        .map(|&(ti, base)| {
            let mut s = base.max(1);
            if !q_mwe_ids.is_empty() && ix.turns[ti].feats.iter().any(|f| q_mwe_ids.contains(&f.id)) {
                s = s.saturating_add(s >> 2);
            }
            s
        })
        .collect();

    for _hop in 0..2 {
        let mut next = vec![0u64; n];
        for i in 0..n {
            let mut ranked_nbrs: Vec<u64> = adj[i]
                .iter()
                .map(|&(nbr, w)| ((act[nbr] as u128 * w as u128) >> 16) as u64)
                .collect();
            ranked_nbrs.sort_unstable_by(|a, b| b.cmp(a));
            let wta_sum: u64 = ranked_nbrs.into_iter().take(SYNAPSE_WTA_CAP).sum();
            let self_keep = if adj[i].len() > SYNAPSE_WTA_CAP {
                (act[i] * 3) >> 2
            } else {
                act[i]
            };
            next[i] = self_keep.saturating_add(wta_sum >> 2);
        }
        act = next;
    }

    let mut out: Vec<(usize, u64)> = (0..n).map(|i| (cand_turns[i].0, act[i])).collect();
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mwe_detection_masks_polysemous_compound_constituents() {
        let toks: Vec<String> = vec!["love", "theme", "park", "rides"]
            .into_iter()
            .map(String::from)
            .collect();
        let (phrases, masked) = detect_mwe_masked_indices(&toks);
        assert!(phrases.contains(&"theme_park".to_string()), "{phrases:?}");
        assert!(masked.contains(&1) && masked.contains(&2), "{masked:?}");
        assert!(!masked.contains(&0));
    }
}
