//! Engine #6: Signed Galois Lattice & Exclusion-Cone Resonance (`Q16.16`).
//!
//! Pure deterministic integer fixed-point math over pinned WordNet 3.1, ConceptNet 5.7,
//! and Wikidata CC0 typed relations (zero LLMs, zero vectors, zero open-class wordlists):
//! 1. **Positive Cone $R^+(q)$**: upward `has_part` / `is_a` / hypernym closure per query facet.
//! 2. **Exclusion Cone $R^-(u)$**: `wordnet::typed_excludes_targets` + closed-class negation /
//!    prohibition / avoidance frames scoped to declarative `field == 0` turns.
//! 3. **Galois Closed-Intent Meet**: concepts witnessed by `>= 2` distinct query facets.
//! 4. **Signed Hazard Resonance $\Phi_{\text{hazard}}(q, u)$**: elevates user exclusion /
//!    constraint turns straight to Rank 1 on advice/safety frames and preserves direct
//!    refutation witnesses on Adversarial queries.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::graph::taxa::fold_ascii_tokens;
use crate::retrieval::index::Index;
use crate::retrieval::query::tokens::is_content;

/// Per-turn Signed Galois resonance profile in `Q16.16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct SignedGaloisScore {
    /// Multi-facet Galois closed-intent meet (`>= 2` query facets intersecting in $R^+$).
    pub(in crate::retrieval) galois_meet_q16: u64,
    /// Signed hazard resonance $\Phi_{\text{hazard}} = \sum_z R^+(q \to z) R^-(u \to z)$.
    pub(in crate::retrieval) hazard_resonance_q16: u64,
    /// True when turn `u` asserts direct exclusion/negation against a positive query target.
    pub(in crate::retrieval) has_exclusion_conflict: bool,
}

/// Closed-class negation, avoidance, and prohibition triggers opening an exclusion scope.
/// hardcode-scan: closed-class (negation, prohibition, and avoidance function words only).
fn is_closed_class_exclusion_trigger(w: &str) -> bool {
    matches!(
        w,
        "no" | "not"
            | "never"
            | "without"
            | "avoid"
            | "avoids"
            | "avoiding"
            | "cannot"
            | "cant"
            | "dont"
            | "wont"
            | "shouldnt"
            | "forbid"
            | "forbidden"
            | "except"
            | "excluding"
            | "exclude"
            | "free"
    )
}

/// Build the Positive Cone $R^+(q)$: maps each normalized concept `z` to `(facet_mask, weight_q16)`
/// across up to 3 upward `has_part` / `is_a` / hypernym / Wikidata hops from each query facet.
pub(in crate::retrieval) fn build_query_positive_cone_q16(
    q_text: &str,
    exclude_speakers: &BTreeSet<String>,
) -> BTreeMap<String, (u32, u64)> {
    let toks = fold_ascii_tokens(q_text);
    let mut facets: Vec<String> = Vec::new();
    // Collect longest 2-word spans first if attested in WordNet/Wikidata, then single content tokens.
    let mut used_pos = vec![false; toks.len()];
    for st in 0..toks.len().saturating_sub(1) {
        let ph = format!("{} {}", toks[st], toks[st + 1]);
        if !exclude_speakers.contains(&toks[st])
            && !exclude_speakers.contains(&toks[st + 1])
            && (crate::wikidata::has_source(&ph) || !crate::wordnet::senses(&ph).is_empty())
            && !facets.contains(&ph)
            && facets.len() < 16
        {
            facets.push(ph);
            used_pos[st] = true;
            used_pos[st + 1] = true;
        }
    }
    for (i, tok) in toks.iter().enumerate() {
        if used_pos[i] || tok.len() < 3 || !is_content(tok) || exclude_speakers.contains(tok) {
            continue;
        }
        if !facets.contains(tok) && facets.len() < 16 {
            facets.push(tok.clone());
        }
    }

    let mut cone: BTreeMap<String, (u32, u64)> = BTreeMap::new();
    let mut insert_cone = |concept: &str, fbit: u32, w: u64| {
        if concept.len() < 3 || w == 0 {
            return;
        }
        cone.entry(concept.to_string())
            .and_modify(|(mask, cur_w)| {
                *mask |= fbit;
                if w > *cur_w {
                    *cur_w = w;
                }
            })
            .or_insert((fbit, w));
    };

    for (fi, facet) in facets.iter().enumerate() {
        let fbit = 1u32 << fi;
        insert_cone(facet, fbit, Q16_ONE);
        if let Some(sing) = facet.strip_suffix('s')
            && sing.len() >= 3
        {
            insert_cone(sing, fbit, Q16_ONE);
        }

        let mut frontier: Vec<String> = vec![facet.clone()];
        let mut visited: BTreeSet<String> = BTreeSet::new();
        visited.insert(facet.clone());

        for hop in 1..=3u32 {
            let hop_w = match hop {
                1 => (Q16_ONE * 3) >> 2, // 0.75
                2 => Q16_ONE >> 1,       // 0.50
                _ => Q16_ONE >> 2,       // 0.25
            };
            let mut next_f: Vec<String> = Vec::new();
            for cur in &frontier {
                for (dst, rel) in crate::wikidata::direct_typed_edges(cur) {
                    if matches!(
                        rel,
                        crate::wikidata::EdgeRel::HasPart | crate::wikidata::EdgeRel::LocatedIn
                    ) {
                        insert_cone(dst, fbit, hop_w);
                        if visited.insert(dst.to_string()) {
                            next_f.push(dst.to_string());
                        }
                    }
                }
                for wn_p in crate::wordnet::typed_part_and_isa_parents(cur) {
                    insert_cone(wn_p, fbit, hop_w);
                    if visited.insert(wn_p.to_string()) {
                        next_f.push(wn_p.to_string());
                    }
                    if let Some((_, head_w)) = wn_p.rsplit_once(' ')
                        && head_w.len() >= 4
                        && !crate::wordnet::senses(head_w).is_empty()
                    {
                        insert_cone(head_w, fbit, (hop_w * 3) >> 2);
                        if visited.insert(head_w.to_string()) {
                            next_f.push(head_w.to_string());
                        }
                    }
                }
                if hop == 1 {
                    for (cn, cw) in crate::conceptnet::two_hop_neighbors(cur, 12) {
                        let scaled = ((u64::from(cw.min(200)) * hop_w) / 200).max(Q16_ONE >> 3);
                        insert_cone(cn, fbit, scaled);
                    }
                }
            }
            if next_f.is_empty() {
                break;
            }
            frontier = next_f;
        }
    }
    cone
}

/// Build the Exclusion Cone $R^-(u)$ for a single turn:
/// 1. Direct WordNet lexical exclusions (`wordnet::typed_excludes_targets`) on rare 1..=2 word spans.
/// 2. Scoped grammatical exclusions: content tokens within 1..=4 tokens after a closed-class
///    exclusion/negation trigger (or `"<noun>-free"`), expanded 1 hop upward.
pub(in crate::retrieval) fn build_turn_exclusion_cone_q16(
    text: &str,
    modality_mask: u8,
) -> BTreeMap<String, u64> {
    let toks = fold_ascii_tokens(text);
    let mut excl: BTreeMap<String, u64> = BTreeMap::new();
    let mut push_ex = |term: &str, w: u64| {
        if term.len() < 3 || w == 0 {
            return;
        }
        excl.entry(term.to_string())
            .and_modify(|cur| {
                if w > *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    };

    // 1. Pinned WordNet typed exclusions (e.g., dietary / material / biological exclusion terms).
    for i in 0..toks.len() {
        for len in 1..=2usize {
            if i + len > toks.len() {
                break;
            }
            let span = if len == 1 {
                toks[i].clone()
            } else {
                format!("{} {}", toks[i], toks[i + 1])
            };
            for target in crate::wordnet::typed_excludes_targets(&span) {
                push_ex(target, Q16_ONE);
                for up in crate::wordnet::typed_part_and_isa_parents(target) {
                    push_ex(up, (Q16_ONE * 3) >> 2);
                }
            }
        }
    }

    // 2. Closed-class negation / avoidance / suffix `-free` scope extraction.
    let neg_bit_set = (modality_mask & 0x02) != 0;
    for (i, tok) in toks.iter().enumerate() {
        if tok == "free" && i > 0 && toks[i - 1].len() >= 3 && is_content(&toks[i - 1]) {
            push_ex(&toks[i - 1], Q16_ONE);
            for up in crate::wordnet::typed_part_and_isa_parents(&toks[i - 1]) {
                push_ex(up, (Q16_ONE * 3) >> 2);
            }
        }
        if is_closed_class_exclusion_trigger(tok) {
            for target in toks.iter().skip(i + 1).take(4) {
                if target.len() < 3
                    || !is_content(target)
                    || is_closed_class_exclusion_trigger(target)
                {
                    continue;
                }
                let base_w = if neg_bit_set { Q16_ONE } else { (Q16_ONE * 3) >> 2 };
                push_ex(target, base_w);
                if let Some(sing) = target.strip_suffix('s')
                    && sing.len() >= 3
                {
                    push_ex(sing, base_w);
                }
                for up in crate::wordnet::typed_part_and_isa_parents(target)
                    .into_iter()
                    .take(4)
                {
                    push_ex(up, base_w >> 1);
                }
            }
        }
    }
    excl
}

/// Evaluate Signed Galois Lattice Meet + Exclusion-Cone Hazard Resonance over `cand_turns`.
pub(in crate::retrieval) fn score_signed_galois_candidates_q16(
    ix: &Index,
    q_text: &str,
    cand_turns: &[usize],
    exclude_speakers: &BTreeSet<String>,
) -> BTreeMap<usize, SignedGaloisScore> {
    let mut out = BTreeMap::new();
    if cand_turns.is_empty() {
        return out;
    }
    let pos_cone = build_query_positive_cone_q16(q_text, exclude_speakers);
    if pos_cone.is_empty() {
        return out;
    }

    let nt_u32 = (ix.turns.len() as u32).max(2);
    let max_df = (nt_u32 / 4).max(6);

    // Precompute Galois Closed-Intent Meet concepts (`>= 2` query facets).
    let closed_meet_concepts: BTreeMap<&str, u64> = pos_cone
        .iter()
        .filter_map(|(c, &(mask, w))| {
            let f_cnt = mask.count_ones();
            (f_cnt >= 2).then_some((c.as_str(), w.saturating_mul(u64::from(f_cnt))))
        })
        .collect();

    for &ti in cand_turns {
        if ti >= ix.turns.len() {
            continue;
        }
        let tr = &ix.turns[ti];
        if tr.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        let d_toks = fold_ascii_tokens(&tr.body_for_headers);

        // 1. Galois Closed-Intent Meet score over turn's positive tokens.
        let mut galois_meet: u64 = 0;
        let mut hit_meet_terms: BTreeSet<&str> = BTreeSet::new();
        for tok in &d_toks {
            if let Some(&mw) = closed_meet_concepts.get(tok.as_str())
                && hit_meet_terms.insert(tok.as_str())
            {
                let df = ix
                    .vocab
                    .get(tok)
                    .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                    .unwrap_or(1);
                if df <= max_df {
                    let ic = aspect_ic_q16(df, nt_u32).max(Q16_ONE >> 1);
                    galois_meet = galois_meet.saturating_add(
                        ((mw as u128).saturating_mul(ic as u128) >> 16) as u64,
                    );
                }
            }
        }

        // 2. Signed Hazard Resonance: Phi_hazard = sum_z R+(q -> z) * R-(u -> z).
        let excl_cone = build_turn_exclusion_cone_q16(&tr.body_for_headers, tr.modality_mask);
        let mut hazard: u64 = 0;
        for (z, ex_w) in &excl_cone {
            if let Some(&(_mask, pos_w)) = pos_cone.get(z.as_str()) {
                let prod = ((pos_w as u128).saturating_mul(*ex_w as u128) >> 16) as u64;
                hazard = hazard.saturating_add(prod);
            }
        }

        if galois_meet > 0 || hazard > 0 {
            out.insert(
                ti,
                SignedGaloisScore {
                    galois_meet_q16: galois_meet,
                    hazard_resonance_q16: hazard,
                    has_exclusion_conflict: hazard >= (Q16_ONE >> 1),
                },
            );
        }
    }
    out
}

/// Seat the strongest exclusion-cone hazard turn (`field == 0`) at Slot 0 on advice/safety frames
/// and sync its session to `final_session_indices[0]`.
pub(in crate::retrieval) fn apply_signed_galois_hazard_seat_q16(
    ix: &Index,
    q_text: &str,
    is_advice_frame: bool,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) -> bool {
    if !is_advice_frame || final_turn_indices.len() < 2 {
        return false;
    }
    let scan_len = final_turn_indices.len().min(16);
    let mut cands: Vec<usize> = final_turn_indices[..scan_len].to_vec();
    for &pti in &ix.preference_directive_turns {
        if !cands.contains(&pti) {
            cands.push(pti);
        }
    }
    let empty_sp = BTreeSet::new();
    let scores = score_signed_galois_candidates_q16(ix, q_text, &cands, &empty_sp);
    let t0 = final_turn_indices[0];
    let t0_haz = scores.get(&t0).map_or(0, |s| s.hazard_resonance_q16);

    let mut best: Option<(usize, u64, u64)> = None;
    for (ti, sc) in scores {
        let tr = &ix.turns[ti];
        if tr.field != 0 || tr.is_question || tr.is_standing_policy {
            continue;
        }
        if sc.has_exclusion_conflict
            && sc.hazard_resonance_q16 >= Q16_ONE
            && sc.hazard_resonance_q16 > t0_haz
        {
            let key = (sc.hazard_resonance_q16, sc.galois_meet_q16);
            if best.is_none_or(|(_, bh, bg)| key > (bh, bg)) {
                best = Some((ti, key.0, key.1));
            }
        }
    }
    let Some((win_ti, _, _)) = best else {
        return false;
    };
    let orig_len = final_turn_indices.len();
    if let Some(pos) = final_turn_indices.iter().position(|&x| x == win_ti) {
        final_turn_indices.remove(pos);
    }
    final_turn_indices.insert(0, win_ti);
    final_turn_indices.truncate(orig_len);

    let win_sess = ix.turns[win_ti].session;
    if let Some(spos) = final_session_indices.iter().position(|&s| s == win_sess) {
        final_session_indices.remove(spos);
    }
    final_session_indices.insert(0, win_sess);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusion_cone_resonates_with_positive_query_cone() {
        let pos = build_query_positive_cone_q16("Can I drink milk?", &BTreeSet::new());
        let excl = build_turn_exclusion_cone_q16("I avoid milk completely.", 0x03);
        let overlap: u64 = excl
            .iter()
            .filter_map(|(z, &ew)| {
                pos.get(z)
                    .map(|&(_, pw)| ((pw as u128 * ew as u128) >> 16) as u64)
            })
            .sum();
        assert!(
            overlap >= Q16_ONE,
            "expected strong signed hazard resonance, got {overlap}"
        );
    }
}
