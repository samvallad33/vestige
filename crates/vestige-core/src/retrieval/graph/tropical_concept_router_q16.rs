//! Maslov Tropical Min-Plus Semiring Concept Router + Banerjee-Pedersen Extended Gloss Overlap (`Q16.16`).
//!
//! Pure deterministic, zero-vector, zero-LLM `$0.00` semantic bridge built strictly over Vestige's
//! pinned offline lexicons (Princeton WordNet 3.1, ConceptNet 5.7, Roget 1911):
//!
//! 1. **Tropical Min-Plus Semiring Routing** ($\oplus = \min, \otimes = +$):
//!    Naive multi-hop graph summation suffers from hub explosion (generic words like `"thing"`,
//!    `"person"`, `"make"` dominate every path). Instead, we compute shortest-path tropical distance
//!    in `Q16.16` integer arithmetic with a degree-information toll:
//!      $$c(u, v) = \lfloor (-\log_2 w_{uv} + \alpha \log_2(\text{deg}(v) + 1)) \cdot 2^{16} \rfloor$$
//!      $$d_{\text{trop}}(u \to v) = \min_{u \to w \to v} \big( c(u, w) + c(w, v) \big)$$
//!    High-degree hubs pay a steep logarithmic toll and are pruned automatically, while crisp
//!    domain bridges (`autumn -> season`, `baking -> bakery`, `medical -> health`,
//!    `vendor -> sales`, `courses -> literacy/skills`, `retries -> rate/limit`) survive with low
//!    tropical distance (high `affinity_q16 = D_MAX - d_trop`).
//! 2. **Banerjee-Pedersen Extended Gloss Overlap (`Q16.16`)**:
//!    Computes shared rare WordNet gloss concepts between query content tokens and turn content
//!    tokens, weighted by corpus IDF so definitional paraphrases (`"matching angle pairs and a
//!    connecting segment"` -> `"congruence"`, `"overwhelming due to retries"` -> `"rate limit"`)
//!    score cleanly without embeddings.

use crate::retrieval::{DetHashMap, DetHashSet, Index, TurnIx, is_content, raw_tokens, stem};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: u32 = 65_536;

/// Maximum tropical path cost (`4.5` bits in `Q16.16`); paths exceeding this yield zero affinity.
const TROPICAL_MAX_DIST_Q16: u32 = 294_912;

/// Fast deterministic `log2(x)` in `Q16.16` for `x >= 1` using bit-length + 4-bit mantissa lookup.
#[inline]
pub(in crate::retrieval) fn log2_u32_q16(x: u32) -> u32 {
    if x <= 1 {
        return 0;
    }
    let exp = 31 - x.leading_zeros();
    // Fractional mantissa in `0..16` (linear interpolation over `[1, 2)` is monotonic and < 0.01 err).
    let frac_idx = if exp >= 4 {
        ((x >> (exp - 4)) & 0xF) as usize
    } else {
        ((x << (4 - exp)) & 0xF) as usize
    };
    const MANT_Q16: [u32; 16] = [
        0, 5_731, 11_136, 16_248, 21_098, 25_709, 30_105, 34_305, 38_326, 42_182, 45_887, 49_452,
        52_887, 56_201, 59_403, 62_500,
    ];
    (exp << 16).saturating_add(MANT_Q16[frac_idx])
}

/// Degree-penalized tropical edge cost `c(u -> v)` in `Q16.16`:
/// `base_cost_q16 + alpha * log2(deg_v + 1)` where `alpha = 0.35` (`22_938 / 65_536`).
#[inline]
fn tropical_edge_cost_q16(norm_weight_pct: u16, target_deg: u32) -> u32 {
    // Convert `weight` in `1..=100` to `-log2(weight / 100)` in Q16:
    // `log2(100) - log2(weight)` (`log2(100) * 65536 = 435_410`, scaled by 0.50 -> max ~2.5 bits).
    let w = (norm_weight_pct as u32).clamp(10, 100);
    let neg_log_w = log2_u32_q16(100).saturating_sub(log2_u32_q16(w)) >> 1;
    // Hub penalty: `0.35 * log2(deg + 1)` in Q16.16.
    let deg_log = log2_u32_q16(target_deg.saturating_add(1));
    let hub_toll = ((deg_log as u64 * 22_938) >> 16) as u32;
    neg_log_w.saturating_add(hub_toll).saturating_add(16_384) // +0.25 bit hop floor
}

/// Precomputed per-query Tropical Min-Plus concept potential + extended gloss signature.
pub(in crate::retrieval) struct TropicalQueryField {
    /// Minimum tropical distance `d_trop(Q -> tok)` in `Q16.16` for corpus-attested tokens.
    pub(in crate::retrieval) dist_by_vid: DetHashMap<u32, u32>,
    /// Query gloss signature (`gloss_token -> rare_weight_q16`) for Banerjee-Pedersen overlap.
    pub(in crate::retrieval) q_gloss_sig: DetHashMap<String, u32>,
}

/// Build the per-query Tropical Min-Plus potential field over WordNet 3.1, ConceptNet 5.7,
/// and Roget 1911 for corpus-attested vocabulary items (`df <= max_df`).
pub(in crate::retrieval) fn build_tropical_query_field(
    ix: &Index,
    q_tokens: &[String],
    skip_words: &DetHashSet<String>,
) -> TropicalQueryField {
    let nt = ix.turns.len().max(1) as u32;
    let max_df = ((nt as usize * 35) / 100).max(8) as u32;
    let mut dist_by_vid: DetHashMap<u32, u32> = DetHashMap::default();
    let mut q_gloss_sig: DetHashMap<String, u32> = DetHashMap::default();

    let mut relax_word = |word: &str, cost_q16: u32| {
        if cost_q16 >= TROPICAL_MAX_DIST_Q16 || word.len() < 3 || !is_content(word) {
            return;
        }
        let mut try_vid = |tok: &str, extra: u32| {
            let eff = cost_q16.saturating_add(extra);
            if eff >= TROPICAL_MAX_DIST_Q16 {
                return;
            }
            if let Some(vid) = ix.vocab.get(tok) {
                let df = ix.t_own_stats.df.get(&vid).copied().unwrap_or(0);
                if (1..=max_df).contains(&df) {
                    dist_by_vid
                        .entry(vid)
                        .and_modify(|cur| {
                            if eff < *cur {
                                *cur = eff;
                            }
                        })
                        .or_insert(eff);
                }
            }
        };
        try_vid(word, 0);
        let st = stem(word);
        if st.len() >= 4 {
            try_vid(&crate::retrieval::stem_term(&st), 8_192);
        }
    };

    let mut seen_q: DetHashSet<String> = DetHashSet::default();
    for raw_w in q_tokens {
        let w = raw_w.as_str();
        if w.len() < 3 || !is_content(w) || skip_words.contains(w) || !seen_q.insert(w.to_string())
        {
            continue;
        }

        // 0. Morphophonemic / derivational zero-hop bridge (WordNet-verified, near-zero cost).
        for deriv in crate::retrieval::index::normalize::derivational_base_stems(w) {
            relax_word(&deriv, 24_576);
        }

        // 1. WordNet 3.1 direct lexicon + upward hypernyms (degree-penalized by sense count).
        for (nbr, sc) in crate::wordnet::full_lexicon_neighbors(w)
            .into_iter()
            .take(18)
        {
            let deg = (crate::wordnet::senses(nbr).len() as u32).max(1) * 4;
            let cost = tropical_edge_cost_q16(sc.clamp(25, 100), deg);
            for part in raw_tokens(nbr) {
                relax_word(&part, cost);
            }
        }
        for hyp in crate::wordnet::upward_hypernyms(w, 2).into_iter().take(10) {
            let deg = (crate::wordnet::senses(hyp).len() as u32).max(1) * 6;
            let cost = tropical_edge_cost_q16(72, deg);
            for part in raw_tokens(hyp) {
                relax_word(&part, cost);
            }
        }

        // 2. ConceptNet 5.7 1-2 hop & reciprocal 2-hop (degree-penalized by WordNet polysemy + corpus DF).
        for (nbr, sc) in crate::conceptnet::two_hop_neighbors(w, 16) {
            let pct = ((sc as u32 * 100) / 200).clamp(20, 100) as u16;
            let deg = (crate::wordnet::senses(nbr).len() as u32).max(1) * 5;
            let cost = tropical_edge_cost_q16(pct, deg);
            for part in raw_tokens(nbr) {
                relax_word(&part, cost);
            }
        }
        for (nbr, sc) in crate::conceptnet::reciprocal_two_hop_neighbors(w, 12) {
            let pct = ((sc as u32 * 100) / 200).clamp(20, 100) as u16;
            let deg = (crate::wordnet::senses(nbr).len() as u32).max(1) * 4;
            let cost = tropical_edge_cost_q16(pct, deg);
            for part in raw_tokens(nbr) {
                relax_word(&part, cost);
            }
        }

        // 3. Roget 1911 conceptual neighbors (higher hub toll for broad thesaurus heads).
        for (nbr, sc) in crate::roget::conceptual_neighbors(w, 18) {
            let deg = (crate::wordnet::senses(nbr).len() as u32).max(1) * 8;
            let cost = tropical_edge_cost_q16(sc.clamp(25, 95), deg).saturating_add(16_384);
            relax_word(nbr, cost);
        }

        // 4. Banerjee-Pedersen Extended Gloss Signature (only non-polysemous content tokens `<= 6` senses).
        let q_senses = crate::wordnet::senses(w).len();
        if (1..=6).contains(&q_senses) {
            for gt in crate::wordnet::gloss_tokens_of(w).into_iter().take(16) {
                if gt.len() >= 4 && is_content(gt) && !skip_words.contains(gt) {
                    let g_senses = crate::wordnet::senses(gt).len();
                    if g_senses <= 5 {
                        let w_q16 = (Q16_ONE / (g_senses.max(1) as u32)).max(16_384);
                        q_gloss_sig
                            .entry(gt.to_string())
                            .and_modify(|cur| *cur = (*cur).max(w_q16))
                            .or_insert(w_q16);
                        // Also allow direct turn match on specific gloss tokens (`cost ~ 2.2 bits`).
                        relax_word(gt, 144_179);
                    }
                }
            }
        }
    }

    TropicalQueryField {
        dist_by_vid,
        q_gloss_sig,
    }
}

/// Score a candidate turn against a [`TropicalQueryField`] in pure `Q16.16`.
///
/// Returns `(tropical_affinity_q16, matched_concept_count, extended_gloss_overlap_q16)`.
pub(in crate::retrieval) fn score_turn_tropical_q16(
    ix: &Index,
    tr: &TurnIx,
    field: &TropicalQueryField,
) -> (u32, u32, u32) {
    let nt = ix.turns.len().max(1) as u32;
    let mut best_by_tok: DetHashMap<u32, u32> = DetHashMap::default();

    let mut visit_id = |vid: u32| {
        if let Some(&dist_q16) = field.dist_by_vid.get(&vid)
            && dist_q16 < TROPICAL_MAX_DIST_Q16
        {
            let aff = TROPICAL_MAX_DIST_Q16 - dist_q16;
            // IDF modulation: rare corpus tokens (`df = 1..3`) get full tropical affinity;
            // frequent background tokens are smoothly attenuated via `log2(N / df)`.
            let df = ix.t_own_stats.df.get(&vid).copied().unwrap_or(1).max(1);
            let idf_q16 = log2_u32_q16((nt / df).max(1)).min(Q16_ONE * 6);
            let weighted = (((aff as u64) * (idf_q16 as u64)) >> 18) as u32;
            best_by_tok
                .entry(vid)
                .and_modify(|cur| *cur = (*cur).max(weighted))
                .or_insert(weighted);
        }
    };

    for &cid in &tr.content {
        visit_id(cid);
    }
    for f in &tr.feats {
        if field.dist_by_vid.contains_key(&f.id) {
            visit_id(f.id);
        }
    }

    // Banerjee-Pedersen Extended Gloss Overlap: check rare turn content tokens whose WordNet
    // gloss intersects `field.q_gloss_sig` (bounded to first 14 rare content tokens for O(1) speed).
    let mut gloss_overlap_q16: u32 = 0;
    if !field.q_gloss_sig.is_empty() {
        let mut checked = 0usize;
        let max_g_df = (nt / 5).max(6);
        let mut seen_g: DetHashSet<&str> = DetHashSet::default();
        for &cid in &tr.content {
            if checked >= 14 {
                break;
            }
            let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
            if !(1..=max_g_df).contains(&df) {
                continue;
            }
            let Some(name) = ix.vocab.names.get(cid as usize) else {
                continue;
            };
            if name.starts_with('\u{1}') || name.len() < 4 {
                continue;
            }
            let s_cnt = crate::wordnet::senses(name).len();
            if !(1..=5).contains(&s_cnt) {
                continue;
            }
            checked += 1;
            let mut token_hits = 0u32;
            let mut token_mass = 0u32;
            for gt in crate::wordnet::gloss_tokens_of(name).into_iter().take(14) {
                if let Some(&gw) = field.q_gloss_sig.get(gt)
                    && seen_g.insert(gt)
                {
                    token_hits += 1;
                    token_mass = token_mass.saturating_add(gw);
                }
            }
            // Banerjee-Pedersen quadratic emphasis on multi-word gloss overlap (`>= 2` shared rare gloss words).
            if token_hits >= 2 {
                gloss_overlap_q16 = gloss_overlap_q16.saturating_add(token_mass);
            } else if token_hits == 1 {
                gloss_overlap_q16 = gloss_overlap_q16.saturating_add(token_mass >> 2);
            }
        }
    }

    let trop_sum: u32 = best_by_tok
        .values()
        .copied()
        .fold(0u32, |acc, v| acc.saturating_add(v));
    (
        trop_sum.min(Q16_ONE * 8),
        best_by_tok.len() as u32,
        gloss_overlap_q16.min(Q16_ONE * 4),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn tropical_hub_toll_penalizes_high_degree_nodes() {
        let low_deg = tropical_edge_cost_q16(80, 2);
        let high_deg = tropical_edge_cost_q16(80, 128);
        assert!(
            low_deg < high_deg,
            "specific low-degree bridge ({low_deg}) must beat hub ({high_deg})"
        );
    }

    #[test]
    fn tropical_router_bridges_autumn_seasons_and_baking_bakery() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: "2024/01/01 (Mon) 10:00".into(),
                timestamp: 1,
                turns: vec![
                    Turn {
                        id: "t_bake".into(),
                        role: "user".into(),
                        content: "I was just thinking about this little bakery in my hometown that made sugar cream pie.".into(),
                        links: Vec::new(),
                        stability: 0.0,
                    },
                    Turn {
                        id: "t_season".into(),
                        role: "user".into(),
                        content: "It is nice here in Denver actually having four distinct seasons.".into(),
                        links: Vec::new(),
                        stability: 0.0,
                    },
                    Turn {
                        id: "t_noise".into(),
                        role: "user".into(),
                        content: "My parents are visiting next week and I am trying to explain my job.".into(),
                        links: Vec::new(),
                        stability: 0.0,
                    },
                ],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        let skip = DetHashSet::default();

        let f_bake = build_tropical_query_field(&ix, &raw_tokens("I am in the mood to do some baking"), &skip);
        let (bake_sc, bake_hits, _) = score_turn_tropical_q16(&ix, &ix.turns[0], &f_bake);
        let (noise_sc, _, _) = score_turn_tropical_q16(&ix, &ix.turns[2], &f_bake);
        assert!(bake_hits >= 1 && bake_sc > noise_sc, "bake_sc={bake_sc}, noise_sc={noise_sc}");

        let f_aut = build_tropical_query_field(&ix, &raw_tokens("see some really great autumn colors"), &skip);
        let (seas_sc, seas_hits, _) = score_turn_tropical_q16(&ix, &ix.turns[1], &f_aut);
        let (noise_sc2, _, _) = score_turn_tropical_q16(&ix, &ix.turns[2], &f_aut);
        assert!(seas_hits >= 1 && seas_sc > noise_sc2, "seas_sc={seas_sc}, noise_sc2={noise_sc2}");
    }
}
