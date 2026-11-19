//! Fellegi–Sunter Record Linkage (`Q16.16` integer log-likelihood ratios, JASA 1969).
//!
//! Resolves cross-session `(possessive_role_noun, proper_entity)` aliases (e.g. `"my sister"`
//! <-> `"Mia"`) using strictly closed-class English possessive determiners and copula verbs,
//! speaker corroboration, entity rarity, and high-fanout disagreement weights.
//! Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::{BTreeMap, BTreeSet};

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const W_POSS_PROX_Q16: i64 = 131_072;           // +2.0
const W_COPULA_BRIDGE_Q16: i64 = 98_304;        // +1.5
const W_MULTI_TURN_Q16: i64 = 65_536;           // +1.0
const W_RARE_ENT_Q16: i64 = 65_536;             // +1.0
const DISAGREE_HIGH_FANOUT_Q16: i64 = -196_608; // -3.0
const MATCH_THRESHOLD_Q16: i64 = 196_608;       // >= 3.0 LLR
const MAX_FANOUT_ENTITIES: usize = 3;
const MAX_PROX_SPAN: usize = 3;
const ALIAS_COEFF_Q16: u64 = 19_661;            // 0.30

#[derive(Default)]
struct AliasEvidenceAcc {
    speakers: BTreeSet<String>,
    has_copula: bool,
    hits: u32,
}

/// Computes Fellegi-Sunter match score in signed `Q16.16` log-likelihood units.
#[inline]
pub(in crate::retrieval) fn fellegi_sunter_llr_q16(
    has_copula: bool,
    single_speaker_multi_turn: bool,
    rare_entity: bool,
    role_fanout: usize,
) -> i64 {
    let mut llr = W_POSS_PROX_Q16;
    if has_copula {
        llr += W_COPULA_BRIDGE_Q16;
    }
    if single_speaker_multi_turn {
        llr += W_MULTI_TURN_Q16;
    }
    if rare_entity {
        llr += W_RARE_ENT_Q16;
    }
    if role_fanout > MAX_FANOUT_ENTITIES {
        llr += DISAGREE_HIGH_FANOUT_Q16;
    }
    llr
}

pub(in crate::retrieval) fn discover_fellegi_sunter_aliases(ix: &Index) -> Vec<(u32, u32)> {
    // hardcode-scan: closed-class (possessive determiners and copula/naming verbs only)
    let is_poss_det = |w: &str| matches!(w, "my" | "our" | "his" | "her" | "their");
    let is_copula_or_name = |w: &str| matches!(w, "is" | "was" | "are" | "were" | "named" | "called");

    let ns = ix.session_turns.len().max(1);
    let mut pair_ev: BTreeMap<(u32, u32), AliasEvidenceAcc> = BTreeMap::new();
    let mut role_to_ents: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();

    for t in &ix.turns {
        if t.field != 0 || t.is_question || t.is_backchannel || t.seq.len() < 2 || t.entities.is_empty() {
            continue;
        }
        let mut ent_positions: Vec<(usize, u32)> = Vec::new();
        for (pos, &(ex_id, _)) in t.seq.iter().enumerate() {
            let Some(name) = ix.vocab.names.get(ex_id as usize) else {
                continue;
            };
            let is_ent = t.entities.contains(&ex_id)
                || ix
                    .vocab
                    .get(&format!("\u{1}ent:{name}"))
                    .is_some_and(|eid| t.entities.contains(&eid));
            if is_ent {
                ent_positions.push((pos, ex_id));
            }
        }
        if ent_positions.is_empty() {
            continue;
        }

        let has_poss = t.feats.iter().any(|f| {
            ix.vocab
                .names
                .get(f.id as usize)
                .is_some_and(|w| is_poss_det(w.as_str()))
        });
        if !has_poss {
            continue;
        }
        let has_copula = t.feats.iter().any(|f| {
            ix.vocab
                .names
                .get(f.id as usize)
                .is_some_and(|w| is_copula_or_name(w.as_str()))
        });

        for &(epos, eid) in &ent_positions {
            let lo = epos.saturating_sub(MAX_PROX_SPAN);
            let hi = (epos + MAX_PROX_SPAN).min(t.seq.len().saturating_sub(1));
            for pos in lo..=hi {
                if pos == epos {
                    continue;
                }
                let role_id = t.seq[pos].0;
                if role_id == eid {
                    continue;
                }
                let Some(rname) = ix.vocab.names.get(role_id as usize) else {
                    continue;
                };
                if rname.len() < 3 || t.entities.contains(&role_id) {
                    continue;
                }
                let rdf = ix.s_stats.df.get(&role_id).copied().unwrap_or(0) as usize;
                if rdf == 0 || rdf * 2 > ns {
                    continue;
                }
                let acc = pair_ev.entry((role_id, eid)).or_default();
                acc.speakers.insert(t.speaker.clone());
                acc.has_copula |= has_copula;
                acc.hits = acc.hits.saturating_add(1);
                role_to_ents.entry(role_id).or_default().insert(eid);
            }
        }
    }

    let mut linked = Vec::new();
    for ((role_id, eid), acc) in pair_ev {
        let edf = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
        let fanout = role_to_ents.get(&role_id).map_or(0, BTreeSet::len);
        let llr = fellegi_sunter_llr_q16(
            acc.has_copula,
            acc.hits >= 2 && acc.speakers.len() == 1,
            (1..=8).contains(&edf),
            fanout,
        );
        if llr >= MATCH_THRESHOLD_Q16 {
            linked.push((role_id, eid));
        }
    }
    linked
}

pub(in crate::retrieval) fn compute_fellegi_sunter_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let pairs = discover_fellegi_sunter_aliases(ix);
    if pairs.is_empty() {
        return out;
    }
    let q_ids: BTreeSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| qt.exact_hit)
        .map(|qt| qt.id)
        .collect();
    let mut targets: BTreeSet<u32> = BTreeSet::new();
    for (a, b) in pairs {
        if q_ids.contains(&a) && !q_ids.contains(&b) {
            targets.insert(b);
        } else if q_ids.contains(&b) && !q_ids.contains(&a) {
            targets.insert(a);
        }
    }
    if targets.is_empty() {
        return out;
    }

    let mut hits = vec![0u64; ns];
    for aid in targets {
        if let Some(postings) = ix.s_postings.get(aid as usize) {
            for &si_u32 in postings {
                let si = si_u32 as usize;
                if si < ns && si != top0 {
                    hits[si] = hits[si].saturating_add(Q16_ONE);
                }
            }
        }
    }
    let max_h = hits.iter().copied().max().unwrap_or(0);
    if max_h == 0 {
        return out;
    }
    for (si, &h) in hits.iter().enumerate() {
        if si != top0 && h > 0 {
            let norm_q16 = (h.saturating_mul(Q16_ONE)) / max_h;
            out[si] = (((smax_q16.saturating_mul(ALIAS_COEFF_Q16)) >> Q16_SHIFT)
                .saturating_mul(norm_q16))
                >> Q16_SHIFT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fellegi_sunter_llr_links_specific_alias_and_rejects_high_fanout() {
        let specific = fellegi_sunter_llr_q16(true, false, true, 1);
        let generic = fellegi_sunter_llr_q16(true, false, true, 6);
        assert!(specific >= MATCH_THRESHOLD_Q16);
        assert!(generic < MATCH_THRESHOLD_Q16);
    }
}
