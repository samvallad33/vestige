//! Deterministic Topic-Thread Retriever & Retroactive Salience Backfill (RSB)
//! for chronological event-ordering queries.
//!
//! Grounded in:
//! - Allen (1983) interval algebra & Pustejovsky et al. (2003) TimeML,
//! - Barzilay & Lapata (2008) entity-grid local coherence,
//! - Chambers & Jurafsky (2008) PPMI narrative event chains,
//! - Josselyn & Frankland (2018) / Rashid et al. (2016) Retroactive Salience Backfill (RSB),
//! - Lin & Bilmes (2011) concave day-bucket diversity & temporal-span facility location.
//!
//! Pure `Q16.16` integer arithmetic. Zero floats, zero embeddings, zero topic wordlists.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::{isqrt_q16, Q16_ONE};

/// Seconds per Gregorian day for day-bucket diversity partitioning.
const SECS_PER_DAY: i64 = 86_400;
/// PPR damping factor `alpha = 0.75` in `Q16.16`.
const PPR_ALPHA_Q16: u64 = 49_152;
/// PPR restart mass `1 - alpha = 0.25` in `Q16.16`.
const PPR_RESTART_Q16: u64 = Q16_ONE - PPR_ALPHA_Q16;

/// Input descriptor for a session participating in bipartite entity-session PPR + RSB.
#[derive(Clone, Debug)]
pub(in crate::retrieval) struct ThreadSessionNode {
    pub(in crate::retrieval) session_idx: usize,
    pub(in crate::retrieval) session_ts: i64,
    pub(in crate::retrieval) base_score_q16: u64,
    /// `(entity_id, weight_q16)` pairs (where `weight_q16 = tf * IC(entity)`).
    pub(in crate::retrieval) entities: Vec<(u32, u64)>,
    /// Earlier sessions linked via backward RSB or causal/supersession arcs.
    pub(in crate::retrieval) rsb_antecedents: Vec<usize>,
}

/// Deterministic bipartite Entity <-> Session Personalized PageRank (8 power iterations)
/// seeded strictly on the highest-IC query entities `seed_entities_q16`.
pub(in crate::retrieval) fn bipartite_entity_session_ppr_q16(
    nodes: &[ThreadSessionNode],
    seed_entities_q16: &BTreeMap<u32, u64>,
    iters: usize,
) -> BTreeMap<usize, u64> {
    let mut s_mass: BTreeMap<usize, u64> = BTreeMap::new();
    if nodes.is_empty() || seed_entities_q16.is_empty() {
        return s_mass;
    }

    // Normalize seed teleportation vector v over query entities.
    let v_sum: u64 = seed_entities_q16
        .values()
        .copied()
        .fold(0u64, u64::saturating_add)
        .max(1);
    let mut v_ent: BTreeMap<u32, u64> = BTreeMap::new();
    for (&eid, &w) in seed_entities_q16 {
        let norm = (((w as u128) << 16) / (v_sum as u128)) as u64;
        if norm > 0 {
            v_ent.insert(eid, norm);
        }
    }

    // Build sparse bipartite adjacency: entity -> [(session_idx, w)], session -> [(entity_id, w)].
    let mut ent_adj: BTreeMap<u32, Vec<(usize, u64)>> = BTreeMap::new();
    let mut ent_deg: BTreeMap<u32, u64> = BTreeMap::new();
    let mut sess_deg: BTreeMap<usize, u64> = BTreeMap::new();
    let mut sess_map: BTreeMap<usize, &ThreadSessionNode> = BTreeMap::new();

    for n in nodes {
        sess_map.insert(n.session_idx, n);
        let mut sd = 0u64;
        for &(eid, w) in &n.entities {
            if w == 0 {
                continue;
            }
            sd = sd.saturating_add(w);
            ent_adj.entry(eid).or_default().push((n.session_idx, w));
            let ed = ent_deg.entry(eid).or_insert(0);
            *ed = ed.saturating_add(w);
        }
        sess_deg.insert(n.session_idx, sd.max(1));
    }

    let mut e_mass: BTreeMap<u32, u64> = v_ent.clone();
    let steps = iters.clamp(2, 12);

    for _ in 0..steps {
        // Entity -> Session step.
        let mut next_s: BTreeMap<usize, u64> = BTreeMap::new();
        for (&eid, &em) in &e_mass {
            if em == 0 {
                continue;
            }
            let deg = ent_deg.get(&eid).copied().unwrap_or(0);
            if deg == 0 {
                continue;
            }
            if let Some(nbrs) = ent_adj.get(&eid) {
                for &(sid, w) in nbrs {
                    let flow = ((em as u128).saturating_mul(w as u128) / (deg as u128)) as u64;
                    let damped = ((flow as u128).saturating_mul(PPR_ALPHA_Q16 as u128) >> 16) as u64;
                    if damped > 0 {
                        let e = next_s.entry(sid).or_insert(0);
                        *e = e.saturating_add(damped);
                    }
                }
            }
        }
        s_mass = next_s;

        // Session -> Entity step + restart on query seeds.
        let mut next_e: BTreeMap<u32, u64> = BTreeMap::new();
        for (&eid, &vm) in &v_ent {
            let restart = ((vm as u128).saturating_mul(PPR_RESTART_Q16 as u128) >> 16) as u64;
            if restart > 0 {
                next_e.insert(eid, restart);
            }
        }
        for (&sid, &sm) in &s_mass {
            if sm == 0 {
                continue;
            }
            let sd = sess_deg.get(&sid).copied().unwrap_or(1);
            if let Some(node) = sess_map.get(&sid) {
                for &(eid, w) in &node.entities {
                    let flow = ((sm as u128).saturating_mul(w as u128) / (sd as u128)) as u64;
                    let damped = ((flow as u128).saturating_mul(PPR_ALPHA_Q16 as u128) >> 16) as u64;
                    if damped > 0 {
                        let e = next_e.entry(eid).or_insert(0);
                        *e = e.saturating_add(damped);
                    }
                }
            }
        }
        e_mass = next_e;
    }

    s_mass
}

/// Retroactive Salience Backfill (Josselyn & Frankland 2018; Rashid et al. 2016):
/// Propagates salience backward from later high-scoring sessions `s_j` (`t_j > t_i`)
/// to earlier weak-lexical sessions `s_i` linked via shared rare entities or causal/RSB arcs.
pub(in crate::retrieval) fn apply_rsb_retroactive_backfill_q16(
    nodes: &[ThreadSessionNode],
    scores_q16: &mut BTreeMap<usize, u64>,
    max_backfill_cap_q16: u64,
) {
    if nodes.len() < 2 || max_backfill_cap_q16 == 0 {
        return;
    }
    let mut ent_sets: BTreeMap<usize, BTreeMap<u32, u64>> = BTreeMap::new();
    for n in nodes {
        let mut m = BTreeMap::new();
        for &(eid, w) in &n.entities {
            let e = m.entry(eid).or_insert(0);
            *e = (*e).max(w);
        }
        ent_sets.insert(n.session_idx, m);
    }

    let mut deltas: BTreeMap<usize, u64> = BTreeMap::new();
    for later in nodes {
        let later_sc = scores_q16.get(&later.session_idx).copied().unwrap_or(0);
        if later_sc == 0 {
            continue;
        }
        let later_ents = &ent_sets[&later.session_idx];
        let later_norm: u64 = later_ents
            .values()
            .copied()
            .fold(0u64, u64::saturating_add)
            .max(1);
        let ant_set: BTreeSet<usize> = later.rsb_antecedents.iter().copied().collect();

        for earlier in nodes {
            if earlier.session_idx == later.session_idx || earlier.session_ts >= later.session_ts {
                continue;
            }
            let has_causal = ant_set.contains(&earlier.session_idx);
            let earlier_ents = &ent_sets[&earlier.session_idx];
            let mut shared_w: u64 = 0;
            for (&eid, &w_l) in later_ents {
                if let Some(&w_e) = earlier_ents.get(&eid) {
                    shared_w = shared_w.saturating_add(w_l.min(w_e));
                }
            }
            if shared_w == 0 && !has_causal {
                continue;
            }
            let overlap_q16 = (((shared_w as u128) << 16) / (later_norm as u128)).min(Q16_ONE as u128) as u64;
            let causal_bonus = if has_causal { Q16_ONE / 2 } else { 0 };
            let coupling_q16 = overlap_q16.saturating_add(causal_bonus).min(Q16_ONE);
            // Alpha = 0.25 retroactive transfer bounded by cap.
            let raw_lift = ((later_sc as u128).saturating_mul(coupling_q16 as u128) >> 18) as u64;
            if raw_lift > 0 {
                let d = deltas.entry(earlier.session_idx).or_insert(0);
                *d = d.saturating_add(raw_lift).min(max_backfill_cap_q16);
            }
        }
    }

    for (sid, lift) in deltas {
        let e = scores_q16.entry(sid).or_insert(0);
        *e = e.saturating_add(lift);
    }
}

/// Select `k` sessions maximizing concave day-bucket diversity (`sqrt` per day bucket)
/// so picks spread across distinct narrative epochs instead of clustering on one day,
/// then sort the selected set chronologically by ascending `session_ts`.
pub(in crate::retrieval) fn select_and_order_topic_thread_q16(
    nodes: &[ThreadSessionNode],
    seed_entities_q16: &BTreeMap<u32, u64>,
    k: usize,
) -> Vec<usize> {
    if nodes.is_empty() || k == 0 {
        return Vec::new();
    }
    let ppr = bipartite_entity_session_ppr_q16(nodes, seed_entities_q16, 8);
    let mut scores: BTreeMap<usize, u64> = BTreeMap::new();
    let max_base = nodes.iter().map(|n| n.base_score_q16).max().unwrap_or(0).max(1);
    let max_ppr = ppr.values().copied().max().unwrap_or(0).max(1);

    for n in nodes {
        let b_norm = (((n.base_score_q16 as u128) << 16) / (max_base as u128)) as u64;
        let p_norm = (((ppr.get(&n.session_idx).copied().unwrap_or(0) as u128) << 16)
            / (max_ppr as u128)) as u64;
        // Equal 50/50 fusion of lexical base and bipartite topic-thread PPR.
        let fused = (b_norm >> 1).saturating_add(p_norm >> 1);
        scores.insert(n.session_idx, fused);
    }

    // Apply bounded Josselyn RSB backfill (capped at 0.30 of max score).
    let top_sc = scores.values().copied().max().unwrap_or(0);
    let cap = ((top_sc as u128).saturating_mul(19_661) >> 16) as u64;
    apply_rsb_retroactive_backfill_q16(nodes, &mut scores, cap);

    // Greedy concave day-bucket selection: F(S) = sum_d sqrt(sum_{s in S cap B_d} r(s)).
    let mut selected: Vec<(i64, usize)> = Vec::with_capacity(k.min(nodes.len()));
    let mut used: BTreeSet<usize> = BTreeSet::new();
    let mut day_sums: BTreeMap<i64, u64> = BTreeMap::new();

    for _ in 0..k {
        let mut best: Option<(usize, i64, i64, u64)> = None;
        for n in nodes {
            if used.contains(&n.session_idx) {
                continue;
            }
            let r = scores.get(&n.session_idx).copied().unwrap_or(0);
            if r == 0 && !selected.is_empty() {
                continue;
            }
            let day = n.session_ts.div_euclid(SECS_PER_DAY);
            let cur = day_sums.get(&day).copied().unwrap_or(0);
            let gain = isqrt_q16(cur.saturating_add(r)).saturating_sub(isqrt_q16(cur));
            let better = match best {
                None => true,
                Some((b_sid, _, _, b_gain)) => {
                    gain > b_gain || (gain == b_gain && n.session_idx < b_sid)
                }
            };
            if better {
                best = Some((n.session_idx, n.session_ts, day, gain));
            }
        }
        let Some((sid, ts, day, _)) = best else {
            break;
        };
        used.insert(sid);
        let r = scores.get(&sid).copied().unwrap_or(0);
        let e = day_sums.entry(day).or_insert(0);
        *e = e.saturating_add(r);
        selected.push((ts, sid));
    }

    // Final chronological ordering by ascending session_ts (deterministic session_idx tie-break).
    selected.sort_by_key(|&(ts, sid)| (ts, sid));
    selected.into_iter().map(|(_, sid)| sid).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsb_backfills_earlier_weak_lexical_session_and_orders_chronologically() {
        // Session 1 (Day 1) has weak lexical score (0) but shares rare entity 99 with Session 3 (Day 5)
        // and is an RSB antecedent. Session 2 (Day 5) is a same-day duplicate of Session 3.
        let nodes = vec![
            ThreadSessionNode {
                session_idx: 1,
                session_ts: SECS_PER_DAY,
                base_score_q16: 0,
                entities: vec![(99, Q16_ONE * 4)],
                rsb_antecedents: vec![],
            },
            ThreadSessionNode {
                session_idx: 2,
                session_ts: 5 * SECS_PER_DAY + 60,
                base_score_q16: Q16_ONE * 8,
                entities: vec![(99, Q16_ONE * 4)],
                rsb_antecedents: vec![],
            },
            ThreadSessionNode {
                session_idx: 3,
                session_ts: 5 * SECS_PER_DAY,
                base_score_q16: Q16_ONE * 10,
                entities: vec![(99, Q16_ONE * 4)],
                rsb_antecedents: vec![1],
            },
            ThreadSessionNode {
                session_idx: 4,
                session_ts: 9 * SECS_PER_DAY,
                base_score_q16: Q16_ONE * 7,
                entities: vec![(99, Q16_ONE * 4)],
                rsb_antecedents: vec![3],
            },
        ];
        let mut seeds = BTreeMap::new();
        seeds.insert(99, Q16_ONE * 4);
        let out = select_and_order_topic_thread_q16(&nodes, &seeds, 3);
        // Must include Day 1 (session 1), Day 5 (session 3), Day 9 (session 4) in chronological order.
        assert_eq!(out, vec![1, 3, 4]);
    }
}
