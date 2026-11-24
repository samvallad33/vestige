//! Tier C Engine #1 — Hashimoto Non-Backtracking Operator + Tripartite Residual `S_tri(q, b, c)` (`Q16.16`)
//! plus Pigeonhole Set-Cover Multi-Hop Detection (`Zero LLM, Zero Vectors`).
//!
//! `B_{(u -> v), (w -> x)} = w * I[v = w and u != x]`
//! `S_tri(q, b, c) = IC_Hash(b ~> c \ Q) * IC(c cap (Q and !mask(b))) / Q16_ONE`

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
use crate::retrieval::graph::actr::actr_fan_strength_q16;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::index::Index;
use crate::retrieval::query::intent::QueryPlan;

const MAX_ROUND1_SEEDS: usize = 6;
const MAX_BRIDGE_ENTS_PER_SEED: usize = 10;

/// Pigeonhole Set-Cover Theorem for corpus-split Multi-Hop queries:
/// If `union_cov >= 2` distinct non-speaker query groups are attested in the corpus
/// yet no single session covers all of them (`max_sess_cov < union_cov`), then by
/// the Pigeonhole Principle answering the query strictly requires >= 2 sessions.
/// Conversely, Single-Hop queries satisfied inside one session have `max_sess_cov == union_cov`.
pub(in crate::retrieval) fn pigeonhole_corpus_split_mh(
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    excluded_speakers: &[String],
) -> bool {
    let mut non_sp_gmask = 0u64;
    for (gi, g) in plan.groups.iter().take(64).enumerate() {
        if !excluded_speakers.iter().any(|sp| sp.eq_ignore_ascii_case(g)) {
            non_sp_gmask |= 1u64 << gi;
        }
    }
    if non_sp_gmask == 0 {
        return false;
    }
    let union_cov = s_exact_mask
        .iter()
        .fold(0u64, |acc, &m| acc | (m & non_sp_gmask))
        .count_ones();
    let max_sess_cov = s_exact_mask
        .iter()
        .map(|&m| (m & non_sp_gmask).count_ones())
        .max()
        .unwrap_or(0);
    union_cov >= 2 && max_sess_cov < union_cov
}

/// Closed-class plural WH (`what`/`which` + `>=4`-char plural word ending in `s`, excluding
/// closed-class auxiliaries/determiners) OR closed-class collective quantifiers
/// (`both`/`common`/`shared`/`together`). Zero open-class nouns.
pub(in crate::retrieval) fn closed_class_plural_or_collective_mh(q_toks_eff: &[String]) -> bool {
    // hardcode-scan: closed-class (WH interrogatives + closed-class auxiliaries/determiners)
    let plural_wh = q_toks_eff.windows(2).any(|p| {
        (p[0] == "what" || p[0] == "which")
            && p[1].len() >= 4
            && p[1].ends_with('s')
            && !p[1].ends_with("ss")
            && !p[1].ends_with("us")
            && !p[1].ends_with("is")
            && !matches!(
                p[1].as_str(),
                "was" | "does" | "has" | "this" | "is" | "kind" | "sort" | "else" | "most"
            )
    });
    // hardcode-scan: closed-class (collective/conjunctive quantifiers)
    let collective = q_toks_eff
        .iter()
        .any(|w| matches!(w.as_str(), "both" | "common" | "shared" | "together"));
    plural_wh || collective
}

pub(in crate::retrieval) fn hashimoto_tripartite_bridge_q16(
    ix: &Index,
    plan: &QueryPlan,
    t_own: &[f32],
    t_exact_mask: &[u64],
    q_group_mask: u64,
    queried_speaker: Option<&str>,
) -> BTreeMap<usize, u64> {
    let nt = ix.turns.len();
    let ns = ix.session_turns.len();
    if nt < 4 || ns < 2 || q_group_mask == 0 {
        return BTreeMap::new();
    }
    let nt_u32 = (nt as u32).max(1);
    let ns_u32 = (ns as u32).max(1);
    let max_bridge_sdf = ((ns as u32) / 4).clamp(3, 8);
    let max_bridge_tdf = ((nt as u32) / 10).clamp(4, 24);

    let mut q_term_ids: BTreeSet<u32> = BTreeSet::new();
    let mut group_ic_q16 = [Q16_ONE; 64];
    for qt in &plan.terms {
        q_term_ids.insert(qt.id);
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(1).max(1);
            let ic = aspect_ic_q16(df, nt_u32).max(Q16_ONE >> 1);
            if ic > group_ic_q16[g] {
                group_ic_q16[g] = ic;
            }
        }
    }
    let mask_ic_q16 = |mut m: u64| -> u64 {
        let mut sum: u64 = 0;
        while m != 0 {
            let g = m.trailing_zeros() as usize;
            sum = sum.saturating_add(group_ic_q16[g]);
            m &= m - 1;
        }
        sum
    };

    let mut seed_cands: Vec<(u64, u32, usize)> = Vec::new();
    for ti in 0..nt {
        let sc_f = t_own.get(ti).copied().unwrap_or(0.0);
        if sc_f <= 0.0 {
            continue;
        }
        let tr = &ix.turns[ti];
        if tr.is_question || tr.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        let hits = (t_exact_mask.get(ti).copied().unwrap_or(0) & q_group_mask).count_ones();
        if hits == 0 {
            continue;
        }
        let sc_q16 = ieee754_bits_to_q16(sc_f.to_bits());
        if sc_q16 > 0 {
            seed_cands.push((sc_q16, hits, ti));
        }
    }
    if seed_cands.is_empty() {
        return BTreeMap::new();
    }
    seed_cands.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));
    let max_seed_q16 = seed_cands[0].0.max(1);
    let seeds: Vec<(usize, u64, u64)> = seed_cands
        .into_iter()
        .take(MAX_ROUND1_SEEDS)
        .filter(|&(sc, _, _)| sc.saturating_mul(3) >= max_seed_q16)
        .map(|(sc, _, ti)| {
            let norm = (((sc as u128) << 16) / (max_seed_q16 as u128)).min(Q16_ONE as u128) as u64;
            let mask_b = t_exact_mask.get(ti).copied().unwrap_or(0) & q_group_mask;
            (ti, norm, mask_b)
        })
        .collect();
    if seeds.is_empty() {
        return BTreeMap::new();
    }
    let seed_set: BTreeSet<usize> = seeds.iter().map(|&(ti, _, _)| ti).collect();

    let mut hashimoto_edges: BTreeMap<u32, Vec<(usize, u64, u64, u64)>> = BTreeMap::new();
    for &(b_ti, b_w_q16, mask_b) in &seeds {
        let members = &ix.session_turns[ix.turns[b_ti].session];
        let mut extract_turns: Vec<(usize, u64)> = vec![(b_ti, Q16_ONE)];
        if let Some(pos) = members.iter().position(|&m| m == b_ti) {
            if pos >= 1 {
                extract_turns.push((members[pos - 1], (Q16_ONE * 3) / 4));
            }
            if pos + 1 < members.len() {
                extract_turns.push((members[pos + 1], (Q16_ONE * 3) / 4));
            }
        }
        let mut local_ents: Vec<(u64, u32)> = Vec::new();
        for (eti, src_scale_q16) in extract_turns {
            let tr = &ix.turns[eti];
            for &eid in tr.entities.iter().chain(tr.content.iter()) {
                if q_term_ids.contains(&eid) {
                    continue;
                }
                if let Some(nm) = ix.vocab.names.get(eid as usize)
                    && (nm.starts_with("\u{1}stem:") || nm.starts_with("\u{1}hyp:"))
                {
                    continue;
                }
                let tdf = ix.t_own_stats.df.get(&eid).copied().unwrap_or(0);
                let sdf = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
                if !(2..=max_bridge_tdf).contains(&tdf) || !(1..=max_bridge_sdf).contains(&sdf) {
                    continue;
                }
                let is_proper = tr.entities.contains(&eid);
                if !is_proper && tdf > 6 {
                    continue;
                }
                let ic_q16 = aspect_ic_q16(sdf, ns_u32);
                let fan_q16 = actr_fan_strength_q16(tdf, nt_u32);
                let mut w_e = (((ic_q16 as u128).saturating_mul(fan_q16 as u128)) >> 16) as u64;
                w_e = ((w_e as u128).saturating_mul(src_scale_q16 as u128) >> 16) as u64;
                if is_proper {
                    w_e = w_e.saturating_add(w_e >> 2);
                }
                local_ents.push((w_e, eid));
            }
        }
        local_ents.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        local_ents.dedup_by_key(|x| x.1);
        for (w_e, eid) in local_ents.into_iter().take(MAX_BRIDGE_ENTS_PER_SEED) {
            hashimoto_edges.entry(eid).or_default().push((b_ti, b_w_q16, mask_b, w_e));
        }
    }
    if hashimoto_edges.is_empty() {
        return BTreeMap::new();
    }

    let mut cand_acc: BTreeMap<usize, (u64, u32)> = BTreeMap::new();
    for (eid, incoming_b) in &hashimoto_edges {
        let Some(posting) = ix.t_postings.get(*eid as usize) else {
            continue;
        };
        let tdf = ix.t_own_stats.df.get(eid).copied().unwrap_or(1);
        let hop2_ic_q16 = (((aspect_ic_q16(tdf, nt_u32) as u128)
            .saturating_mul(actr_fan_strength_q16(tdf, nt_u32) as u128))
            >> 16) as u64;

        for &c_u32 in posting {
            let c_ti = c_u32 as usize;
            if c_ti >= nt || seed_set.contains(&c_ti) {
                continue;
            }
            let tr_c = &ix.turns[c_ti];
            if tr_c.is_question || tr_c.is_backchannel || turn_is_superseded(ix, c_ti) {
                continue;
            }
            if !tr_c.content.contains(eid) && !tr_c.entities.contains(eid) {
                continue;
            }
            let mask_c = t_exact_mask.get(c_ti).copied().unwrap_or(0) & q_group_mask;

            let mut best_for_eid: u64 = 0;
            for &(b_ti, b_w_q16, mask_b, w_be_q16) in incoming_b {
                if c_ti == b_ti {
                    continue;
                }
                let ic_hash = ((b_w_q16 as u128)
                    .saturating_mul(w_be_q16.saturating_add(hop2_ic_q16) as u128)
                    >> 16) as u64;
                let residual_mask = mask_c & !mask_b;
                let resid_ic_q16 = if residual_mask != 0 {
                    (mask_ic_q16(residual_mask).saturating_mul(185)) / 100
                } else if mask_c != 0 {
                    Q16_ONE / 2
                } else {
                    Q16_ONE / 3
                };
                let mut s_tri = ((ic_hash as u128).saturating_mul(resid_ic_q16 as u128) >> 16) as u64;
                if tr_c.session != ix.turns[b_ti].session {
                    s_tri = s_tri.saturating_add(s_tri >> 2);
                }
                if let Some(qsp) = queried_speaker
                    && tr_c.speaker.eq_ignore_ascii_case(qsp)
                {
                    s_tri = s_tri.saturating_add(s_tri / 5);
                }
                if s_tri > best_for_eid {
                    best_for_eid = s_tri;
                }
            }
            if best_for_eid > 0 {
                cand_acc
                    .entry(c_ti)
                    .and_modify(|(best_w, cnt)| {
                        if best_for_eid > *best_w {
                            *best_w = best_for_eid;
                        }
                        *cnt = cnt.saturating_add(1);
                    })
                    .or_insert((best_for_eid, 1));
            }
        }
    }

    let mut out: BTreeMap<usize, u64> = BTreeMap::new();
    for (c_ti, (best_w, cnt)) in cand_acc {
        let mult = match cnt {
            0 | 1 => 100u64,
            2 => 130u64,
            3 => 150u64,
            _ => 165u64,
        };
        let fused = best_w.saturating_mul(mult) / 100;
        if fused > 0 {
            out.insert(c_ti, fused);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, Session, Turn, build_index, plan_query, raw_tokens};

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.into(),
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn pigeonhole_set_cover_distinguishes_split_multihop_from_single_hop() {
        let sessions = vec![
            Session {
                id: "D1".into(),
                date: "2023-05-01".into(),
                timestamp: 0,
                turns: vec![mk_turn(
                    "D1:1",
                    "Alice",
                    "I joined the robotics club with Mira.",
                )],
            },
            Session {
                id: "D2".into(),
                date: "2023-05-10".into(),
                timestamp: 0,
                turns: vec![mk_turn(
                    "D2:1",
                    "Alice",
                    "Mira won the regional trophy in Seattle.",
                )],
            },
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Where did Alice's robotics partner win a trophy?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let speakers = vec!["alice".to_string()];

        // Pick the first two non-speaker query group bits in `plan.groups`.
        let non_sp_bits: Vec<usize> = plan
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.eq_ignore_ascii_case("alice"))
            .map(|(i, _)| i)
            .collect();
        assert!(non_sp_bits.len() >= 2, "groups={:?}", plan.groups);
        let b0 = 1u64 << non_sp_bits[0];
        let b1 = 1u64 << non_sp_bits[1];

        // Split across D1 (b0) and D2 (b1): Pigeonhole fires!
        let split_masks = vec![b0, b1];
        assert!(pigeonhole_corpus_split_mh(&plan, &split_masks, &speakers));

        // Single-Hop where D1 already covers both groups: Pigeonhole stays FALSE!
        let single_masks = vec![b0 | b1, b0];
        assert!(!pigeonhole_corpus_split_mh(&plan, &single_masks, &speakers));
    }

    #[test]
    fn closed_class_plural_wh_and_collective_quantifiers_match_cleanly() {
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "What books has Alice read?"
        )));
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "Which places did both Alice and Bob visit?"
        )));
        assert!(closed_class_plural_or_collective_mh(&raw_tokens(
            "What do Alice and Bob have in common?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What was Alice doing yesterday?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What does Alice like?"
        )));
        assert!(!closed_class_plural_or_collective_mh(&raw_tokens(
            "What kind of dog does Alice own?"
        )));
    }

    #[test]
    fn hashimoto_tripartite_bridge_discovers_non_backtracking_hop2_turn() {
        let sessions = vec![
            Session {
                id: "D1".into(),
                date: "2023-05-01".into(),
                timestamp: 0,
                turns: vec![
                    mk_turn(
                        "D1:1",
                        "Alice",
                        "I signed up for the pottery workshop with Kazuki.",
                    ),
                    mk_turn("D1:2", "Bob", "That sounds relaxing."),
                ],
            },
            Session {
                id: "D2".into(),
                date: "2023-05-12".into(),
                timestamp: 0,
                turns: vec![
                    mk_turn(
                        "D2:1",
                        "Alice",
                        "Kazuki invited me to the kiln exhibition in Kyoto.",
                    ),
                    mk_turn("D2:2", "Bob", "Have fun on the trip!"),
                ],
            },
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Where was the exhibition connected to Alice's pottery workshop?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let q_group_mask = if plan.groups.len() >= 64 {
            u64::MAX
        } else {
            (1u64 << plan.groups.len()) - 1
        };

        // Simulate Pass-1 seed at D1:1 (turn 0) with exact mask on pottery/workshop,
        // and D2:1 (turn 2) carrying residual exhibition group + shared rare bridge entity Kazuki.
        let t_own = vec![5.0f32, 0.0, 0.5, 0.0];
        let mut t_exact_mask = vec![0u64; ix.turns.len()];
        for (ti, tr) in ix.turns.iter().enumerate() {
            for qt in &plan.terms {
                if qt.exact_hit
                    && let Some(g) = qt.group
                    && g < 64
                    && tr.content.contains(&qt.id)
                {
                    t_exact_mask[ti] |= 1u64 << g;
                }
            }
        }
        let scores = hashimoto_tripartite_bridge_q16(
            &ix,
            &plan,
            &t_own,
            &t_exact_mask,
            q_group_mask,
            Some("alice"),
        );
        // Non-backtracking: seed turn 0 is excluded, Hop-2 target turn 2 (`D2:1`) is discovered via Kazuki!
        assert!(!scores.contains_key(&0), "seed turn must not backtrack onto itself");
        assert!(
            scores.get(&2).copied().unwrap_or(0) > 0,
            "Hop-2 endpoint turn 2 must receive positive Hashimoto tripartite bridge score: {scores:?}"
        );
    }
}
