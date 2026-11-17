//! Cycle E (#814, #815, #816, #818, #819, #820, #822):
//! Taxonomic Signatures, Superimposed Bitmasks, Structural Aligners & Durable Preference Register
//! in pure integer / Q16.16 fixed-point arithmetic (zero floats, zero open-class word lists).
//!
//! - #814: Wilkins u64 taxonomic code (hierarchical WordNet hypernym prefix, XOR + leading_zeros)
//! - #815: Mooers Zatocode u128 superimposed coding (k=4 splitmix64 bit positions)
//! - #816: Ranganathan 5-facet PMEST signature (Personality, Matter, Energy, Space, Time)
//! - #818: Ounis-gated Cyc isa/partOf 64-bit bitmask resonance over WordNet/ConceptNet
//! - #819: Forbus SME structural predicate skeleton alignment
//! - #820: Canonical entity equivalence bridge
//! - #822: Agichtein typographical definition/theorem zone detector (punctuation + structure only)
//! - PrefRegister: Durable 64-bit Experiencer Preference Register over constraint/directive turns

use super::ghostlink_dgs::Q16;
use super::index::Index;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;

#[inline]
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn fnv1a_16(s: &str) -> u16 {
    let mut h: u32 = 0x811C_9DC5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    ((h >> 16) ^ (h & 0xFFFF)) as u16
}

/// #814 Wilkins u64 Taxonomic Prefix Code for a vocabulary term:
/// Packs hierarchical WordNet ancestor levels into upper bits `[63..16]` and the
/// canonical leaf/stem hash into `[15..0]`, enabling XOR + `leading_zeros()` LCP matching.
pub(super) fn wilkins_term_code_u64(word: &str) -> u64 {
    let hypers = crate::wordnet::upward_hypernyms(word, 4);
    let mut l1 = fnv1a_16(word);
    let mut l2 = l1;
    let mut l3 = l1;
    if !hypers.is_empty() {
        let len = hypers.len();
        l1 = fnv1a_16(hypers[len - 1]);
        l2 = fnv1a_16(hypers[len / 2]);
        l3 = fnv1a_16(hypers[0]);
    }
    let l0 = fnv1a_16(word);
    ((l1 as u64) << 48) | ((l2 as u64) << 32) | ((l3 as u64) << 16) | (l0 as u64)
}

/// #814 Integer Wilkins Longest-Common-Prefix taxonomic similarity in Q16.16 `[0, 65536]`.
#[inline]
pub(super) fn wilkins_lcp_similarity_q16(a: u64, b: u64) -> u64 {
    let lcp = (a ^ b).leading_zeros() as u64;
    (lcp.min(64) * Q16) >> 6
}

/// Build a turn-level Wilkins code from its rarest content/entity token in `ix.vocab`.
pub(super) fn turn_wilkins_code_u64(ix: &Index, ti: usize) -> u64 {
    let mut best_id: Option<(u32, u32)> = None;
    for &e in &ix.turns[ti].entities {
        let df = ix.t_stats.df.get(&e).copied().unwrap_or(u32::MAX);
        if best_id.map_or(true, |(bdf, _)| df < bdf) {
            best_id = Some((df, e));
        }
    }
    if best_id.is_none() {
        for &w in &ix.turns[ti].content {
            let df = ix.t_own_stats.df.get(&w).copied().unwrap_or(u32::MAX);
            if df >= 1 && best_id.map_or(true, |(bdf, _)| df < bdf) {
                best_id = Some((df, w));
            }
        }
    }
    let Some((_, id)) = best_id else {
        return 0;
    };
    let word = ix
        .vocab
        .names
        .get(id as usize)
        .map(String::as_str)
        .unwrap_or("")
        .strip_prefix("\u{1}ent:")
        .unwrap_or_else(|| ix.vocab.names.get(id as usize).map(String::as_str).unwrap_or(""));
    wilkins_term_code_u64(word)
}

/// #815 Mooers Zatocode u128 superimposed coding (`k = 4` bit positions per token id).
#[inline]
pub(super) fn zatocode_u128(ids: impl IntoIterator<Item = u32>) -> u128 {
    let mut sig: u128 = 0;
    for id in ids {
        let h1 = splitmix64(id as u64);
        let h2 = splitmix64(h1 ^ 0xA5A5_5A5A_1234_5678);
        let b0 = (h1 & 127) as u32;
        let b1 = ((h1 >> 7) & 127) as u32;
        let b2 = (h2 & 127) as u32;
        let b3 = ((h2 >> 7) & 127) as u32;
        sig |= (1u128 << b0) | (1u128 << b1) | (1u128 << b2) | (1u128 << b3);
    }
    sig
}

/// Bitwise Zatocode intersection score normalized against query signature weight in Q16.16.
#[inline]
pub(super) fn zatocode_overlap_q16(q_sig: u128, doc_sig: u128) -> u64 {
    let q_bits = q_sig.count_ones() as u64;
    if q_bits == 0 || doc_sig == 0 {
        return 0;
    }
    let hit = (q_sig & doc_sig).count_ones() as u64;
    (hit.saturating_mul(Q16)) / q_bits
}

/// #816 Ranganathan 5-Facet PMEST Signature (Personality, Matter, Energy, Space, Time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PmestSignature {
    pub(super) personality: u16,
    pub(super) matter: u16,
    pub(super) energy: u16,
    pub(super) space: u16,
    pub(super) time: u16,
}

impl PmestSignature {
    pub(super) fn for_turn(ix: &Index, ti: usize) -> Self {
        let t = &ix.turns[ti];
        let p_bits = (t.entities.len().min(63) as u16)
            | (if !t.speaker.is_empty() { 1 << 8 } else { 0 });
        let m_bits = (t.slot_entities_clean.min(255) as u16)
            | ((t.role_objs.len().min(63) as u16) << 8);
        let e_bits = ((t.answers_question as u16) << 0)
            | ((t.has_causal as u16) << 1)
            | ((t.is_initiative as u16) << 2)
            | ((t.is_personal_constraint as u16) << 3);
        let s_bits = ((t.has_spatial as u16) << 0) | ((t.form_handle as u16) << 1);
        let t_bits = ((t.has_temporal as u16) << 0)
            | ((t.form_clock as u16) << 1)
            | ((t.form_digit as u16) << 2);
        Self {
            personality: p_bits,
            matter: m_bits,
            energy: e_bits,
            space: s_bits,
            time: t_bits,
        }
    }

    pub(super) fn for_query(plan: &QueryPlan) -> Self {
        let it = &plan.intent;
        Self {
            personality: (plan.focal_ids.len().min(63) as u16) | (1 << 8),
            matter: plan.groups.len().min(255) as u16,
            energy: ((it.sniper_why as u16) << 1) | ((it.preference as u16) << 3),
            space: (it.sniper_where as u16) << 0,
            time: (((it.temporal || it.sniper_when || it.is_event_sequence) as u16) << 0),
        }
    }

    #[inline]
    pub(super) fn align_q16(self, other: Self) -> u64 {
        let mut active = 0u64;
        let mut matched = 0u64;
        for (a, b) in [
            (self.personality, other.personality),
            (self.matter, other.matter),
            (self.energy, other.energy),
            (self.space, other.space),
            (self.time, other.time),
        ] {
            if a != 0 {
                active += 1;
                if (a & b) != 0 || (a > 0 && b > 0) {
                    matched += 1;
                }
            }
        }
        if active == 0 {
            0
        } else {
            (matched * Q16) / active
        }
    }
}

/// #818 Ounis-Gated Cyc isa/partOf Bitmask + #820 Entity Canonical Equivalence Bridge:
/// Encodes up to 64 relation bits over WordNet hypernyms and ConceptNet two-hop neighbors.
pub(super) fn isa_part_of_bitmask_u64(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
    let mut mask = 0u64;
    let t = &ix.turns[ti];
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(q_word) = ix.vocab.names.get(qt.id as usize) else {
            continue;
        };
        let clean_q = q_word
            .strip_prefix("\u{1}stem:")
            .or_else(|| q_word.strip_prefix("\u{1}ent:"))
            .unwrap_or(q_word.as_str());
        for cn in crate::conceptnet::two_hop_neighbors(clean_q, 6) {
            if let Some(lid) = ix.vocab.get(cn.0) {
                if t.content.contains(&lid) {
                    let bit = (splitmix64(lid as u64) & 63) as u32;
                    mask |= 1u64 << bit;
                }
            }
        }
        for hyp in crate::wordnet::upward_hypernyms(clean_q, 2) {
            if let Some(lid) = ix.vocab.get(hyp) {
                if t.content.contains(&lid) {
                    let bit = (splitmix64((lid as u64) ^ 0x5555) & 63) as u32;
                    mask |= 1u64 << bit;
                }
            }
        }
    }
    mask
}

/// #819 Forbus Structure-Mapping Engine (SME) relational skeleton similarity in Q16.16.
#[inline]
pub(super) fn forbus_sme_alignment_q16(ix: &Index, plan: &QueryPlan, ti: usize) -> u64 {
    let q_pmest = PmestSignature::for_query(plan);
    let d_pmest = PmestSignature::for_turn(ix, ti);
    let facet_align = q_pmest.align_q16(d_pmest);
    let rel_bonus = if !ix.turns[ti].is_question && !ix.turns[ti].role_objs.is_empty() {
        Q16 >> 2
    } else {
        0
    };
    facet_align.saturating_add(rel_bonus).min(Q16)
}

/// #822 Agichtein Typographical Definition / Theorem Zone Detector:
/// Identifies definitional / specification / structured-list turns from pure punctuation
/// and structural layout signals (no open-class cue-word lists).
#[inline]
pub(super) fn is_definition_or_theorem_zone(ix: &Index, ti: usize) -> bool {
    let t = &ix.turns[ti];
    if t.list_item_count >= 2 || (t.form_midcap && t.form_digit && t.slot_entities_clean >= 2) {
        return true;
    }
    let body = t.body_for_headers.as_str();
    body.contains(":=") || body.contains("=>") || (body.matches(':').count() >= 2 && t.content.len() >= 6)
}

/// Durable Experiencer Preference Register (`PrefRegister`):
/// Indexes durable constraint/standing-directive turns with their u128 Zatocode over
/// entity/vocab tokens for zero-overlap lifestyle/preference routing.
#[derive(Clone, Debug, Default)]
pub(super) struct PrefRegister {
    pub(super) slot_mask_64: u64,
    pub(super) entries: Vec<(usize, u128)>,
}

impl PrefRegister {
    pub(super) fn build(ix: &Index) -> Self {
        let mut entries = Vec::new();
        for (ti, t) in ix.turns.iter().enumerate() {
            if !turn_is_superseded(ix, ti) && (t.is_personal_constraint || t.is_standing_policy) {
                let code = zatocode_u128(
                    t.role_objs
                        .iter()
                        .copied()
                        .chain(t.entities.iter().copied())
                        .chain(t.content.iter().copied().take(8)),
                );
                entries.push((ti, code));
            }
        }
        Self {
            slot_mask_64: ix.standing_register,
            entries,
        }
    }

    pub(super) fn resonance_q16(&self, q_code: u128, ti: usize) -> u64 {
        if let Some((_, d_code)) = self.entries.iter().find(|&&(etri, _)| etri == ti) {
            zatocode_overlap_q16(q_code, *d_code)
        } else {
            0
        }
    }
}

/// Evaluate combined Cycle E taxonomic/structural resonance score in Q16.16 for `ti`
/// and return `(taxa_score_q16, isa_mask, is_def_zone)` for LockeLedger auditing.
pub(super) fn evaluate_turn_taxa_q16(
    ix: &Index,
    plan: &QueryPlan,
    pref_reg: &PrefRegister,
    q_wilkins: u64,
    q_zato: u128,
    ti: usize,
) -> (u64, u64, bool) {
    let t_wilkins = turn_wilkins_code_u64(ix, ti);
    let wilkins_q16 = if q_wilkins != 0 && t_wilkins != 0 {
        wilkins_lcp_similarity_q16(q_wilkins, t_wilkins)
    } else {
        0
    };
    let d_zato = zatocode_u128(
        ix.turns[ti]
            .entities
            .iter()
            .copied()
            .chain(ix.turns[ti].content.iter().copied().take(12)),
    );
    let zato_q16 = zatocode_overlap_q16(q_zato, d_zato);
    let sme_q16 = forbus_sme_alignment_q16(ix, plan, ti);
    let isa_mask = if crate::zero_overlap::ounis_zero_overlap_gate(
        plan.terms
            .iter()
            .filter_map(|qt| qt.group.map(|_| ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0))),
    ) {
        isa_part_of_bitmask_u64(ix, plan, ti)
    } else {
        0
    };
    let isa_q16 = ((isa_mask.count_ones() as u64).min(8) * Q16) >> 3;
    let is_def = is_definition_or_theorem_zone(ix, ti);
    let def_q16 = if is_def { Q16 >> 3 } else { 0 };
    let pref_q16 = if pref_reg.slot_mask_64 != 0 || !pref_reg.entries.is_empty() {
        pref_reg.resonance_q16(q_zato, ti)
    } else {
        0
    };

    let combined = wilkins_q16
        .saturating_add(zato_q16)
        .saturating_add(sme_q16 >> 1)
        .saturating_add(isa_q16)
        .saturating_add(def_q16)
        .saturating_add(pref_q16);
    (combined, isa_mask, is_def)
}

/// Conditional Deficit-Gated Corpus Rescue (`Deficit A` / `Deficit B` / `preference`):
/// Scans unseated corpus turns `0..nt` (not just `final_turn_indices`) and seats the strongest
/// taxonomic (`isa_mask != 0`) or Durable Experiencer Preference (`PrefRegister`) turn into
/// a redundant slot when the query has a zero-overlap or preference coverage deficit.
pub(super) fn apply_deficit_taxa_and_pref_rescue(
    ix: &Index,
    plan: &QueryPlan,
    pref_reg: &PrefRegister,
    q_wilkins: u64,
    q_zato: u128,
    base_q16: &[u64],
    t_exact_mask: &[u64],
    final_turn_indices: &mut Vec<usize>,
    k: usize,
    deficit_a: bool,
    deficit_b: bool,
) {
    if k < 6 || (!deficit_a && !deficit_b && !plan.intent.preference) {
        return;
    }
    let nt = ix.turns.len();
    let mut best_cand: Option<(usize, u64)> = None;
    for ti in 0..nt {
        let t = &ix.turns[ti];
        if t.is_question
            || t.is_backchannel
            || turn_is_superseded(ix, ti)
            || final_turn_indices.contains(&ti)
        {
            continue;
        }
        let pref_res = if plan.intent.preference {
            pref_reg.resonance_q16(q_zato, ti)
        } else {
            0
        };
        let isa_mask = if deficit_a {
            isa_part_of_bitmask_u64(ix, plan, ti)
        } else {
            0
        };
        if pref_res == 0 && isa_mask == 0 {
            continue;
        }
        let (taxa_q16, _, _) = evaluate_turn_taxa_q16(ix, plan, pref_reg, q_wilkins, q_zato, ti);
        let score = taxa_q16.saturating_add(pref_res);
        if score > (Q16 >> 1) && best_cand.map_or(true, |(_, bs)| score > bs) {
            best_cand = Some((ti, score));
        }
    }

    let Some((cand_ti, _)) = best_cand else {
        return;
    };
    if final_turn_indices.len() < k {
        final_turn_indices.push(cand_ti);
        return;
    }
    let cand_sess = ix.turns[cand_ti].session;
    if final_turn_indices[..5].iter().any(|&t| ix.turns[t].session == cand_sess) {
        return;
    }
    if let Some(slot) = (5..final_turn_indices.len()).rev().find(|&idx| {
        let vti = final_turn_indices[idx];
        let vs = ix.turns[vti].session;
        let dup = final_turn_indices[..idx].iter().any(|&e| ix.turns[e].session == vs);
        let mut earlier_mask = 0u64;
        for &e in &final_turn_indices[..idx] {
            earlier_mask |= t_exact_mask[e];
        }
        (dup && ix.turns[vti].field == 1)
            || base_q16[vti] == 0
            || (dup && ix.turns[vti].is_question && (t_exact_mask[vti] & !earlier_mask) == 0)
    }) {
        final_turn_indices[slot] = cand_ti;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilkins_zatocode_and_pmest_are_pure_integer_deterministic() {
        let c_dog = wilkins_term_code_u64("dog");
        let c_puppy = wilkins_term_code_u64("puppy");
        assert_ne!(c_dog, 0);
        assert!(wilkins_lcp_similarity_q16(c_dog, c_dog) == Q16);
        assert!(wilkins_lcp_similarity_q16(c_dog, c_puppy) <= Q16);

        let z1 = zatocode_u128([10, 20, 30]);
        let z2 = zatocode_u128([10, 20, 99]);
        assert!(zatocode_overlap_q16(z1, z1) == Q16);
        assert!(zatocode_overlap_q16(z1, z2) > 0);

        let p = PmestSignature {
            personality: 1,
            matter: 2,
            energy: 0,
            space: 1,
            time: 0,
        };
        assert_eq!(p.align_q16(p), Q16);
    }
}
