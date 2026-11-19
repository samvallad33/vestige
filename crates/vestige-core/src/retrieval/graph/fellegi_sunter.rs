//! Fellegi–Sunter Record Linkage (`Q16.16` integer log-likelihood ratios, JASA 1969).
//!
//! Resolves cross-session `(possessive_role_noun, proper_entity)` aliases (e.g. `"my sister"`
//! <-> `"Mia"`) by scanning `t.body_for_headers` (preserving closed-class possessive determiners
//! and copulas stripped from `t.feats`), corroborated by `t.role_objs`, speaker identity, entity
//! rarity, and high-fanout disagreement weights. Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::{BTreeMap, BTreeSet};

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const W_POSS_PROX_Q16: i64 = 131_072;           // +2.0: <poss_det> <role_noun> near proper entity
const W_COPULA_BRIDGE_Q16: i64 = 98_304;        // +1.5: copula/naming verb or role_objs attestation
const W_MULTI_TURN_Q16: i64 = 65_536;           // +1.0: corroborated across >=2 turns by same speaker
const W_RARE_ENT_Q16: i64 = 65_536;             // +1.0: proper entity session df <= 10
const DISAGREE_HIGH_FANOUT_Q16: i64 = -196_608; // -3.0: generic noun binding >4 distinct entities
const MATCH_THRESHOLD_Q16: i64 = 196_608;       // >= 3.0 total LLR required for union
const MAX_FANOUT_ENTITIES: usize = 4;
const MAX_WORD_WINDOW: usize = 5;
const ALIAS_COEFF_Q16: u64 = 22_938;            // 0.35 in Q16.16

#[derive(Default)]
struct AliasEvidenceAcc {
    speakers: BTreeSet<String>,
    has_copula_or_role_obj: bool,
    hits: u32,
}

#[inline]
pub(in crate::retrieval) fn fellegi_sunter_llr_q16(
    has_copula_or_role_obj: bool,
    single_speaker_multi_turn: bool,
    rare_entity: bool,
    role_fanout: usize,
) -> i64 {
    let mut llr = W_POSS_PROX_Q16;
    if has_copula_or_role_obj {
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
    let is_copula_or_name =
        |w: &str| matches!(w, "is" | "was" | "are" | "were" | "named" | "called");

    let ns = ix.session_turns.len().max(1);
    let mut pair_ev: BTreeMap<(u32, u32), AliasEvidenceAcc> = BTreeMap::new();
    let mut role_to_ents: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();

    for t in &ix.turns {
        if t.field != 0 || t.is_question || t.is_backchannel || t.entities.is_empty() {
            continue;
        }
        let words: Vec<(String, bool)> = t
            .body_for_headers
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let cap = s.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && s.len() >= 2;
                (s.to_ascii_lowercase(), cap)
            })
            .collect();
        if words.len() < 3 {
            continue;
        }

        let has_copula = words.iter().any(|(w, _)| is_copula_or_name(w.as_str()));

        // Locate proper entity token positions inside `words` attested in `t.entities`.
        let mut ent_pos: Vec<(usize, u32)> = Vec::new();
        for (idx, (w, cap)) in words.iter().enumerate() {
            if !*cap {
                continue;
            }
            let Some(vid) = ix.vocab.get(w) else {
                continue;
            };
            let ent_tag = format!("\u{1}ent:{w}");
            let is_ent = t.entities.contains(&vid)
                || ix.vocab.get(&ent_tag).is_some_and(|eid| t.entities.contains(&eid));
            if is_ent {
                ent_pos.push((idx, vid));
            }
        }
        if ent_pos.is_empty() {
            continue;
        }

        // Scan `<poss_det> <role_noun>` or `<poss_det> <adj> <role_noun>` within MAX_WORD_WINDOW of entity.
        for i in 0..words.len() {
            if !is_poss_det(words[i].0.as_str()) {
                continue;
            }
            for step in 1..=2usize {
                if i + step >= words.len() {
                    break;
                }
                let (ref cand_role, cand_cap) = words[i + step];
                if cand_cap || cand_role.len() < 3 || is_copula_or_name(cand_role.as_str()) {
                    continue;
                }
                let Some(role_id) = ix.vocab.get(cand_role) else {
                    continue;
                };
                if !t.content.contains(&role_id) || t.entities.contains(&role_id) {
                    continue;
                }
                let rdf = ix.s_stats.df.get(&role_id).copied().unwrap_or(0) as usize;
                if rdf == 0 || rdf * 2 > ns {
                    continue;
                }
                let rpos = i + step;
                let in_role_objs = t.role_objs.contains(&role_id);
                for &(ep, eid) in &ent_pos {
                    if eid != role_id && ep.abs_diff(rpos) <= MAX_WORD_WINDOW {
                        let appos_adj = ep.abs_diff(rpos) <= 2;
                        let acc = pair_ev.entry((role_id, eid)).or_default();
                        acc.speakers.insert(t.speaker.clone());
                        acc.has_copula_or_role_obj |= has_copula || in_role_objs || appos_adj;
                        acc.hits = acc.hits.saturating_add(1);
                        role_to_ents.entry(role_id).or_default().insert(eid);
                    }
                }
            }
        }
    }

    let mut linked = Vec::new();
    for ((role_id, eid), acc) in pair_ev {
        let edf = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
        let fanout = role_to_ents.get(&role_id).map_or(0, BTreeSet::len);
        let llr = fellegi_sunter_llr_q16(
            acc.has_copula_or_role_obj,
            acc.hits >= 2 && acc.speakers.len() == 1,
            (1..=10).contains(&edf),
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
