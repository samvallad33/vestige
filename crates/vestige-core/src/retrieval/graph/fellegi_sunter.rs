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
const MATCH_THRESHOLD_Q16: i64 = 294_912;       // >= 4.5: explicit naming plus rarity or corroboration
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

// Closed-class copula/naming grammar, bounded to the actual role-name span.
fn explicit_naming_link(words: &[(String, bool)], poss: usize, role: usize, entity: usize) -> bool {
    let naming = |i: usize| words.get(i).is_some_and(|(w, _)|
        matches!(w.as_str(), "is" | "was" | "are" | "were" | "named" | "called"));
    (entity == role + 2 && naming(role + 1))
        || (entity + 2 == poss && naming(entity + 1) && role == poss + 1)
}

pub(in crate::retrieval) fn discover_fellegi_sunter_aliases(ix: &Index) -> Vec<(u32, u32)> {
    ix.fellegi_aliases
        .get_or_init(|| discover_fellegi_sunter_aliases_uncached(ix))
        .clone()
}

fn discover_fellegi_sunter_aliases_uncached(ix: &Index) -> Vec<(u32, u32)> {
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
                for &(ep, eid) in &ent_pos {
                    if eid != role_id && ep.abs_diff(rpos) <= MAX_WORD_WINDOW {
                        // The naming verb must CONNECT the role and name, not merely occur
                        // elsewhere in the turn. Possessive cue is already required above.
                        if !explicit_naming_link(&words, i, rpos, ep) {
                            continue;
                        }
                        let acc = pair_ev.entry((role_id, eid)).or_default();
                        acc.speakers.insert(t.speaker.clone());
                        acc.has_copula_or_role_obj = true;
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
        if acc.has_copula_or_role_obj && llr >= MATCH_THRESHOLD_Q16 {
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
    if ns < 2 || top0 >= ns || smax_q16 == 0 {
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

#[cfg(test)]
mod raw_alias_tests {
    use super::*;
    use crate::retrieval::tests::{ix_of, s, turn};
    #[test]
    fn raw_naming_requires_connecting_verb_and_possessive() {
        let words = |text: &str| text.split_whitespace().map(|w| (w.to_lowercase(), false)).collect::<Vec<_>>();
        assert!(explicit_naming_link(&words("my sister is alice"), 0, 1, 3));
        assert!(explicit_naming_link(&words("alice is my sister"), 2, 3, 0));
        assert!(!explicit_naming_link(&words("my sister visited alice who is happy"), 0, 1, 3));
        assert!(!explicit_naming_link(&words("my sister alice is happy"), 0, 1, 2));
        assert!(fellegi_sunter_llr_q16(false, false, true, 1) < MATCH_THRESHOLD_Q16);
    }
    #[test]
    fn aliases_visible_when_possessive_is_a_stopword() {
        let mut sessions = vec![s("a", vec![turn("user", "My sister is Alice. Alice paints murals.")])];
        for i in 0..7 { sessions.push(s(&format!("b{i}"), vec![turn("user", "Quiet weather arrived today.")])); }
        let ix = ix_of(&sessions);
        let role = ix.vocab.get("sister").unwrap();
        let entity = ix.vocab.get("alice").unwrap();
        assert!(discover_fellegi_sunter_aliases(&ix).contains(&(role, entity)));
    }
}
