//! Kernel 6: MGMem + SAG2 Interleaved Bridge + SentGraph (`mgmem_sag2_sentgraph_q16.rs`).
//!
//! References:
//! - MGMem (arXiv:2611.33174): Mention-incidence graph scored with exact `omega(v)^2` priority.
//! - SAG2 (arXiv:2610.22588): Round-2 per-term rank-interleaved unseen BM25 over Top-12 bridge terms.
//! - SentGraph (arXiv:2601.03014): Preceding Nucleus subject inheritance (`0.35x` in Q16).
//!
//! Eliminates single-bridge-term monopolization in multi-hop second-hop discovery and enables
//! anaphoric passage sentences to carry their document title + preceding Nucleus subject.

use crate::retrieval::{DetHashMap, DetHashSet, Feat, Index, is_content};
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;

const Q16_ONE: u64 = 65_536;
/// SentGraph `0.35x` subject inheritance factor (`22938 / 65536`).
pub(in crate::retrieval) const SENTGRAPH_SUBJECT_WEIGHT: f32 = 0.35;
/// Maximum Round-2 bridge terms interleaved cleanly per SAG2.
const SAG2_MAX_BRIDGE_TERMS: usize = 12;

/// Extracts candidate rare subject tokens from a sentence/turn (Capitalized nouns or rare
/// exact tokens) to propagate downstream at `0.35x` via SentGraph inside `index/mod.rs`.
pub(in crate::retrieval) fn extract_nucleus_subject_ids(
    ix_body: &str,
    vocab: &mut crate::retrieval::Vocab,
) -> Vec<Feat> {
    let mut out = Vec::new();
    let mut seen = DetHashSet::default();
    for raw in ix_body.split_whitespace().take(24) {
        let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.len() < 4 {
            continue;
        }
        let first = trimmed.chars().next().unwrap_or('a');
        let low = trimmed.to_ascii_lowercase();
        if first.is_ascii_uppercase() && is_content(&low) && seen.insert(low.clone()) {
            let id = vocab.intern(&low);
            out.push(Feat {
                id,
                w: SENTGRAPH_SUBJECT_WEIGHT,
                counts_len: false,
                inheritable: true,
            });
            if out.len() >= 4 {
                break;
            }
        }
    }
    out
}

/// Discovers Round-2 unseen bridge turns using MGMem `omega(v)^2` mention-incidence priority
/// over Top-12 bridge terms extracted from Stage-1 seed turns, interleaving Round-2 unseen
/// candidate turns rank-by-rank across all 12 terms so no single high-DF bridge term hogs the pool.
pub(in crate::retrieval) fn mgmem_sag2_interleaved_bridge_q16(
    ix: &Index,
    seed_turns: &[usize],
    already_top: &DetHashSet<usize>,
    base_scores: &[f32],
    limit: usize,
) -> Vec<(usize, u64)> {
    if seed_turns.is_empty() || limit == 0 {
        return Vec::new();
    }
    let nt = ix.turns.len();
    let max_df = (nt as u32 / 8).clamp(3, 36);

    // 1. MGMem Mention-Incidence counts `omega(v)` across Top seed turns
    let mut omega: DetHashMap<u32, u64> = DetHashMap::default();
    for &ti in seed_turns.iter().take(6) {
        if ti >= nt {
            continue;
        }
        let tr = &ix.turns[ti];
        for &cid in tr.content.iter().chain(tr.entities.iter()) {
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
            if df >= 2 && df <= max_df {
                *omega.entry(cid).or_default() += 1;
            }
        }
    }

    // Score each bridge term by exact `omega(v)^2 * IDF_approx`:
    // multi-mention bridge vertices (`omega >= 2`) naturally receive >= 4x weight over `omega == 1`.
    let mut bridge_terms: Vec<(u32, u64)> = omega
        .into_iter()
        .map(|(cid, w_cnt)| {
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(2).max(1) as u64;
            let sq = w_cnt.saturating_mul(w_cnt);
            let mult = if w_cnt >= 2 { 4 } else { 1 };
            let priority = ((sq * mult * nt as u64) / df).max(1);
            (cid, priority)
        })
        .collect();
    bridge_terms.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    bridge_terms.truncate(SAG2_MAX_BRIDGE_TERMS);

    if bridge_terms.is_empty() {
        return Vec::new();
    }

    // 2. Per-term ranked list of unseen candidate turns (SAG2)
    let mut per_term_lists: Vec<Vec<(usize, u64)>> = Vec::with_capacity(bridge_terms.len());
    for &(cid, term_prio) in &bridge_terms {
        let Some(postings) = ix.t_postings.get(cid as usize) else {
            continue;
        };
        let mut candidates: Vec<(usize, u64)> = postings
            .iter()
            .map(|&raw_ti| raw_ti as usize)
            .filter(|&ti| ti < nt && !already_top.contains(&ti) && !ix.turns[ti].is_question)
            .map(|ti| {
                let base_sc = ieee754_bits_to_q16(base_scores.get(ti).copied().unwrap_or(0.0f32).to_bits());
                let tr = &ix.turns[ti];
                let entity_bonus = if tr.entities.contains(&cid) { Q16_ONE >> 1 } else { 0 };
                let score = (term_prio.min(Q16_ONE * 4) >> 2)
                    .saturating_add(entity_bonus)
                    .saturating_add(base_sc >> 1);
                (ti, score)
            })
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(8);
        if !candidates.is_empty() {
            per_term_lists.push(candidates);
        }
    }

    // 3. Round-robin rank-interleave across all bridge-term lists so high-DF bridges never hog!
    let mut out = Vec::with_capacity(limit);
    let mut seen_turns = DetHashSet::default();
    let max_depth = per_term_lists.iter().map(|v| v.len()).max().unwrap_or(0);
    for rank in 0..max_depth {
        for list in &per_term_lists {
            if let Some(&(ti, sc)) = list.get(rank)
                && seen_turns.insert(ti)
            {
                out.push((ti, sc));
                if out.len() >= limit {
                    return out;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, build_index};
    use crate::retrieval::tests::sess;

    #[test]
    fn sag2_interleaves_bridge_terms_without_high_df_monopolization() {
        let sessions = vec![
            sess("s0", &[("user", "Northbridge founded by Helena in Prague")]),
            sess("s1", &[("user", "Helena moved to Oslo in 2021")]),
            sess("s2", &[("user", "Helena painted landscapes")]),
            sess("s3", &[("user", "Prague clock tower chimes at noon")]),
        ];
        let ix = build_index(&sessions, &Params::default());
        let mut already = DetHashSet::default();
        already.insert(0);
        let base = vec![1.0, 0.5, 0.4, 0.6];
        let bridges = mgmem_sag2_interleaved_bridge_q16(&ix, &[0], &already, &base, 4);
        let ids: Vec<usize> = bridges.iter().map(|&(ti, _)| ti).collect();
        assert!(ids.contains(&1) && ids.contains(&3), "Must interleave both bridge concepts: {ids:?}");
    }
}
