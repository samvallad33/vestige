//! Tropical `(max, +)` 2-Hop Squaring & Entity-Denoised 2-Hop Lexical Bridging (`Q16.16`).
//!
//! Takes the top `3..=5` Pass-1 declarative seed turns sharing a high-IC anchor with the query,
//! extracts ONLY their *new* high-IC entities/rare anchors (`df <= hub_df`, pruning ambient hubs),
//! and squares the sparse turn-entity bipartite graph in the `(max, +)` tropical semiring:
//!
//!   `T^2[q -> t2] = max_{t1 in seeds, e in (E(t1) \ Q) cap E(t2)} (w(q -> t1) + ic(e))`
//!
//! fused with a bounded multiplicative chain multiplier when multiple distinct rare bridge
//! entities independently witness the same candidate turn `t2`.
//! Pure integer `Q16.16`, deterministic `BTreeMap`/`BTreeSet`, zero floats in bridge scoring.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::{aspect_ic_q16, Q16_ONE};
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
use crate::retrieval::graph::actr::actr_fan_strength_q16;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::index::Index;
use crate::retrieval::query::intent::QueryPlan;

/// Maximum Pass-1 seed turns used to extract bridge entities (`5`).
const MAX_PASS1_SEEDS: usize = 5;
/// Maximum bridge entities kept per Pass-1 seed turn (`8`).
const MAX_BRIDGE_ENTS_PER_SEED: usize = 8;

/// Computes exact integer `(max, +)` tropical 2-hop chain scores in `Q16.16` over candidate turns.
///
/// Returns `BTreeMap<turn_idx, score_q16>` for turns connected via at least one non-query,
/// non-hub bridge entity from the top `3..=5` Pass-1 seed turns.
pub(in crate::retrieval) fn tropical_max_plus_entity_bridge_q16(
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
    let hub_turn_df = ((nt as u32) / 10).clamp(4, 28);
    let hub_sess_df = ((ns as u32) / 3).clamp(3, 14);

    // Collect exact query term IDs (plus their high-IC subset) so bridge extraction pulls ONLY *new* anchors.
    let mut q_term_ids: BTreeSet<u32> = BTreeSet::new();
    let mut q_hi_ic_mask: u64 = 0;
    for qt in &plan.terms {
        q_term_ids.insert(qt.id);
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            let tdf = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            if (1..=hub_turn_df).contains(&tdf) {
                q_hi_ic_mask |= 1u64 << g;
            }
        }
    }
    let anchor_mask = if q_hi_ic_mask != 0 {
        q_hi_ic_mask
    } else {
        q_group_mask
    };

    // Pass 1: select top `3..=5` declarative turns sharing a high-IC anchor with the query.
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
        let hits = (t_exact_mask.get(ti).copied().unwrap_or(0) & anchor_mask).count_ones();
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
    seed_cands.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let max_seed_q16 = seed_cands[0].0.max(1);
    let seeds: Vec<(usize, u64)> = seed_cands
        .into_iter()
        .take(MAX_PASS1_SEEDS)
        .filter(|&(sc, _, _)| sc.saturating_mul(3) >= max_seed_q16)
        .map(|(sc, _, ti)| {
            let norm = (((sc as u128) << 16) / (max_seed_q16 as u128)).min(Q16_ONE as u128) as u64;
            (ti, norm)
        })
        .collect();
    if seeds.is_empty() {
        return BTreeMap::new();
    }
    let seed_set: BTreeSet<usize> = seeds.iter().map(|&(ti, _)| ti).collect();

    // Extract ONLY new high-IC entities / rare anchors from Pass-1 seed turns (ACT-R fan attenuated).
    // Map: bridge_entity_id -> max_{t1 in seeds containing e} (seed_w(t1) + ic_fan(e))
    let mut bridge_ent_best_path: BTreeMap<u32, (u64, usize)> = BTreeMap::new();
    for &(seed_ti, seed_w_q16) in &seeds {
        let tr = &ix.turns[seed_ti];
        let mut local_ents: Vec<(u64, u32)> = Vec::new();
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
            if !(2..=hub_turn_df).contains(&tdf) || !(1..=hub_sess_df).contains(&sdf) {
                continue;
            }
            let is_proper = tr.entities.contains(&eid);
            // Prefer proper entities and very rare concrete nouns (`tdf <= 6`) to prevent hub drift.
            if !is_proper && tdf > 6 {
                continue;
            }
            let ic_q16 = aspect_ic_q16(sdf, ns_u32);
            let fan_q16 = actr_fan_strength_q16(tdf, nt_u32);
            let mut w_e = (((ic_q16 as u128).saturating_mul(fan_q16 as u128)) >> 16) as u64;
            if is_proper {
                w_e = w_e.saturating_add(w_e >> 2);
            }
            local_ents.push((w_e, eid));
        }
        local_ents.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        local_ents.dedup_by_key(|x| x.1);
        for (w_e, eid) in local_ents.into_iter().take(MAX_BRIDGE_ENTS_PER_SEED) {
            let hop1_plus_e = seed_w_q16.saturating_add(w_e);
            bridge_ent_best_path
                .entry(eid)
                .and_modify(|cur| {
                    if hop1_plus_e > cur.0 {
                        *cur = (hop1_plus_e, seed_ti);
                    }
                })
                .or_insert((hop1_plus_e, seed_ti));
        }
    }
    if bridge_ent_best_path.is_empty() {
        return BTreeMap::new();
    }

    // Tropical `(max, +)` squaring into candidate turns `t2` via inverted turn postings.
    // Track `(max_chain_q16, distinct_bridge_ents)` so multi-entity bridge chains get multiplicative confirmation.
    let mut cand_acc: BTreeMap<usize, (u64, u32)> = BTreeMap::new();
    for (&eid, &(hop1_w, seed_ti)) in &bridge_ent_best_path {
        let Some(posting) = ix.t_postings.get(eid as usize) else {
            continue;
        };
        let seed_sess = ix.turns[seed_ti].session;
        let tdf = ix.t_own_stats.df.get(&eid).copied().unwrap_or(1);
        let hop2_w = (((aspect_ic_q16(tdf, nt_u32) as u128)
            .saturating_mul(actr_fan_strength_q16(tdf, nt_u32) as u128))
            >> 16) as u64;
        for &t2_u32 in posting {
            let t2 = t2_u32 as usize;
            if t2 >= nt || seed_set.contains(&t2) {
                continue;
            }
            let tr2 = &ix.turns[t2];
            if tr2.is_question || tr2.is_backchannel || turn_is_superseded(ix, t2) {
                continue;
            }
            // Must carry the bridge entity in its own content/entities (not merely carried noise).
            if !tr2.content.contains(&eid) && !tr2.entities.contains(&eid) {
                continue;
            }
            let mut chain_w = hop1_w.saturating_add(hop2_w);
            // Cross-session 2-hop chains get +20% (primary Multi-Hop topology).
            if tr2.session != seed_sess {
                chain_w = chain_w.saturating_add(chain_w / 5);
            }
            if let Some(qsp) = queried_speaker {
                if tr2.speaker.eq_ignore_ascii_case(qsp) {
                    chain_w = chain_w.saturating_add(chain_w / 6);
                }
            }
            cand_acc
                .entry(t2)
                .and_modify(|(best_w, cnt)| {
                    if chain_w > *best_w {
                        *best_w = chain_w;
                    }
                    *cnt = cnt.saturating_add(1);
                })
                .or_insert((chain_w, 1));
        }
    }

    // Multiplicative chain fusion: reward turns confirmed by >= 2 independent rare bridge entities.
    let mut out: BTreeMap<usize, u64> = BTreeMap::new();
    for (t2, (best_w, cnt)) in cand_acc {
        let mult_num = match cnt {
            0 | 1 => 100u64,
            2 => 125u64,
            3 => 142u64,
            _ => 155u64,
        };
        let fused = best_w.saturating_mul(mult_num) / 100;
        if fused > 0 {
            out.insert(t2, fused);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{build_index, plan_query, Params, Query, Session, Turn};

    #[test]
    fn tropical_max_plus_bridge_links_two_hop_entity_and_prunes_hub() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: "2024-01-01".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "Alice".into(),
                    content: "I bought a telescope for Project Orion last week.".into(),
                    stability: 1.0,
                    links: vec![],
                }],
            },
            Session {
                id: "s1".into(),
                date: "2024-01-05".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t1".into(),
                    role: "Alice".into(),
                    content: "Project Orion is headed to the Atacama observatory in Chile.".into(),
                    stability: 1.0,
                    links: vec![],
                }],
            },
            Session {
                id: "s2".into(),
                date: "2024-01-08".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t2".into(),
                    role: "Bob".into(),
                    content: "I enjoyed dinner with friends last week.".into(),
                    stability: 1.0,
                    links: vec![],
                }],
            },
            Session {
                id: "s3".into(),
                date: "2024-01-10".into(),
                timestamp: 0,
                turns: vec![Turn {
                    id: "t3".into(),
                    role: "Bob".into(),
                    content: "Weather looks sunny next week too.".into(),
                    stability: 1.0,
                    links: vec![],
                }],
            },
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "Where is Alice taking her telescope?".into(),
            date: "2024-02-01".into(),
        };
        let plan = plan_query(&ix, &q, &p);
        let mut t_own = vec![0.0f32; ix.turns.len()];
        let mut t_exact_mask = vec![0u64; ix.turns.len()];
        // Seed Pass 1 on t0 ("telescope").
        t_own[0] = 5.0;
        t_exact_mask[0] = 1;
        let bridges = tropical_max_plus_entity_bridge_q16(
            &ix,
            &plan,
            &t_own,
            &t_exact_mask,
            1,
            Some("alice"),
        );
        assert!(
            bridges.get(&1).copied().unwrap_or(0) > bridges.get(&2).copied().unwrap_or(0),
            "expected t1 (Project Orion -> Atacama) to outscore hub turn t2: {bridges:?}"
        );
    }
}
