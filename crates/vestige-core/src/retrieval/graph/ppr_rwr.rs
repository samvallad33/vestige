//! Bipartite Session–Entity–RareTerm 2-Step Personalized PageRank / RWR (`Q16.16`).
//!
//! Combines Haveliwala (WWW 2002) topic-sensitive Personalized PageRank, Collins & Loftus
//! (Psych. Rev. 1975) spreading activation, and Josselyn & Frankland (2018) bidirectional
//! engram co-allocation. Zero floats, zero open-class wordlists.

use crate::retrieval::index::Index;
use crate::retrieval::QueryPlan;
use std::collections::BTreeMap;

const Q16_ONE: u64 = 65_536;
const Q16_SHIFT: u32 = 16;
const MAX_PPR_SEEDS: usize = 3;
const PPR_STEPS: usize = 2;
const PPR_ALPHA_NUM: u64 = 3;
const PPR_ALPHA_DEN: u64 = 4;
const ENT_MIN_DF: u32 = 2;
const ENT_MAX_DF: u32 = 8;
const RARE_MIN_DF: u32 = 2;
const RARE_MAX_DF: u32 = 6;
const ENT_EDGE_W: u64 = 2;
const RARE_EDGE_W: u64 = 1;
const RSB_EDGE_W: u64 = 2;
const PPR_COEFF_MULTI_Q16: u64 = 24_904; // 0.38 in Q16.16
const PPR_COEFF_BASE_Q16: u64 = 13_107;  // 0.20 in Q16.16

/// Runs `PPR_STEPS` deterministic integer power iterations over weighted undirected `adj`.
pub(in crate::retrieval) fn run_ppr_rwr_q16(
    ns: usize,
    adj: &[Vec<(usize, u64)>],
    restart_q16: &[u64],
) -> Vec<u64> {
    let mut deg = vec![0u64; ns];
    for (u, nbrs) in adj.iter().enumerate() {
        deg[u] = nbrs.iter().map(|&(_, w)| w).sum();
    }
    let mut p = restart_q16.to_vec();
    // Truncated PPR series: accumulate visit mass over every step so a
    // 1-hop neighbour is never outscored by a 2-hop one through walk parity.
    let mut acc = vec![0u64; ns];
    for _ in 0..PPR_STEPS {
        let mut walked = vec![0u64; ns];
        for (u, nbrs) in adj.iter().enumerate() {
            if p[u] == 0 || deg[u] == 0 {
                continue;
            }
            for &(v, w) in nbrs {
                let flow = (p[u].saturating_mul(w)) / deg[u];
                walked[v] = walked[v].saturating_add(flow);
            }
        }
        for (s, slot) in p.iter_mut().enumerate() {
            let r_part = restart_q16[s] / PPR_ALPHA_DEN;
            let w_part = (walked[s].saturating_mul(PPR_ALPHA_NUM)) / PPR_ALPHA_DEN;
            *slot = r_part.saturating_add(w_part);
            acc[s] = acc[s].saturating_add(w_part);
        }
    }
    acc
}

/// Computes `Q16.16` session lift vector from 2-step Bipartite PPR / RWR.
pub(in crate::retrieval) fn compute_bipartite_ppr_lifts_q16(
    ix: &Index,
    plan: &QueryPlan,
    multi_or_partial: bool,
    base_sight_q16: &[u64],
    s_ord_base: &[usize],
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    let mut out = vec![0u64; ns];
    let Some(&top0) = s_ord_base.first() else {
        return out;
    };
    if ns <= 5 || top0 >= ns || smax_q16 == 0 {
        return out;
    }
    let top0_sight = base_sight_q16[top0];
    if top0_sight == 0 {
        return out;
    }

    let mut seeds: Vec<(usize, u64)> = Vec::new();
    for &si in s_ord_base.iter().take(MAX_PPR_SEEDS) {
        if si < ns && base_sight_q16[si].saturating_mul(5) >= top0_sight.saturating_mul(2) {
            seeds.push((si, base_sight_q16[si]));
        }
    }
    let seed_sum: u64 = seeds.iter().map(|&(_, w)| w).sum();
    if seed_sum == 0 {
        return out;
    }
    let mut restart_q16 = vec![0u64; ns];
    for &(si, w) in &seeds {
        restart_q16[si] = (w.saturating_mul(Q16_ONE)) / seed_sum;
    }

    let mut q_term_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut s_mask_local = vec![0u64; ns];
    let n_qg = plan.groups.len().min(64);
    let mask_limit = if n_qg == 64 {
        u64::MAX
    } else if n_qg == 0 {
        0
    } else {
        (1u64 << n_qg) - 1
    };
    for qt in &plan.terms {
        q_term_ids.insert(qt.id);
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            let bit = 1u64 << g;
            if let Some(post) = ix.s_postings.get(qt.id as usize) {
                for &si_u32 in post {
                    let si = si_u32 as usize;
                    if si < ns {
                        s_mask_local[si] |= bit;
                    }
                }
            }
        }
    }

    let mut ent_postings: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (si, ents) in ix.session_entities.iter().enumerate() {
        for &eid in ents {
            ent_postings.entry(eid).or_default().push(si);
        }
    }
    let mut rare_postings: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (si, rares) in ix.session_rare_nt.iter().enumerate() {
        for &tid in rares {
            rare_postings.entry(tid).or_default().push(si);
        }
    }

    let mut pair_w: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut add_edge = |a: usize, b: usize, w: u64| {
        if a != b && w > 0 {
            let key = if a < b { (a, b) } else { (b, a) };
            *pair_w.entry(key).or_insert(0) += w;
        }
    };

    // MemORAI DWPR (2026): query-conditioned edge weights amplify query-relevant entity/rare
    // bridges by 2.5x while high-DF hubs stay damped by 1/df.
    for (eid, sessions) in &ent_postings {
        let df = sessions.len() as u32;
        if (ENT_MIN_DF..=ENT_MAX_DF).contains(&df) {
            let base_w = (ENT_EDGE_W.saturating_mul(Q16_ONE)) / (df as u64);
            let w = if q_term_ids.contains(eid) {
                base_w.saturating_mul(5) / 2
            } else {
                base_w
            };
            for (i, &sa) in sessions.iter().enumerate() {
                for &sb in &sessions[(i + 1)..] {
                    add_edge(sa, sb, w);
                }
            }
        }
    }
    for (tid, sessions) in &rare_postings {
        let df = sessions.len() as u32;
        if (RARE_MIN_DF..=RARE_MAX_DF).contains(&df) {
            let base_w = (RARE_EDGE_W.saturating_mul(Q16_ONE)) / (df as u64);
            let w = if q_term_ids.contains(tid) {
                base_w.saturating_mul(5) / 2
            } else {
                base_w
            };
            for (i, &sa) in sessions.iter().enumerate() {
                for &sb in &sessions[(i + 1)..] {
                    add_edge(sa, sb, w);
                }
            }
        }
    }
    for si in 0..ns {
        for &(deg, prev) in &ix.session_rsb_back[si] {
            if prev < ns {
                let w = (RSB_EDGE_W.saturating_mul(Q16_ONE)) / (deg.max(2) as u64);
                add_edge(si, prev, w);
            }
        }
        // EngramRAG (2026) R_SUPERSEDES = 2.5 weight along explicit state-revision edges.
        if let Some(sups) = ix.session_supersedes_back.get(si) {
            for &prev in sups {
                if prev < ns {
                    add_edge(si, prev, (Q16_ONE * 5) / 2);
                }
            }
        }
        if let Some(prev_ep) = ix.session_episode_prev[si] {
            if prev_ep < ns {
                add_edge(si, prev_ep, Q16_ONE >> 1);
            }
        }
    }
    if pair_w.is_empty() {
        return out;
    }

    let mut raw_adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); ns];
    for ((a, b), w) in pair_w {
        raw_adj[a].push((b, w));
        raw_adj[b].push((a, w));
    }

    // ISO-RAG (Sep 2026) Localized Cheeger Isoperimetric Edge Mask:
    // Discrete node volume `v(u) = Q16 / (1 + deg(u))`; compute 1-hop ball volume `V(S_u)`
    // and 2-hop boundary volume `V(dS_u)` -> local Cheeger ratio `phi_u = V(dS_u) * Q16 / (V(S_u) + 1)`.
    // Prune hub-leaking edges where `min(phi_u, phi_v) * Q16 < beta * max(phi_u, phi_v)` (`beta = 0.35`),
    // while always keeping each node's top-2 strongest incident edges so no node is isolated.
    let node_vol: Vec<u64> = raw_adj
        .iter()
        .map(|nbrs| Q16_ONE / (1 + nbrs.len() as u64))
        .collect();
    let mut phi_q16 = vec![Q16_ONE; ns];
    for u in 0..ns {
        if raw_adj[u].is_empty() {
            continue;
        }
        let mut s_u = std::collections::BTreeSet::new();
        s_u.insert(u);
        let mut vol_s = node_vol[u];
        for &(v, _) in &raw_adj[u] {
            if s_u.insert(v) {
                vol_s = vol_s.saturating_add(node_vol[v]);
            }
        }
        let mut bd_seen = std::collections::BTreeSet::new();
        let mut vol_bd = 0u64;
        for &(v, _) in &raw_adj[u] {
            for &(w, _) in &raw_adj[v] {
                if !s_u.contains(&w) && bd_seen.insert(w) {
                    vol_bd = vol_bd.saturating_add(node_vol[w]);
                }
            }
        }
        phi_q16[u] = (vol_bd.saturating_mul(Q16_ONE)) / vol_s.max(1);
    }
    const CHEEGER_BETA_Q16: u64 = 22_938; // 0.35 in Q16.16
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); ns];
    for u in 0..ns {
        let mut sorted = raw_adj[u].clone();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (idx, &(v, w)) in sorted.iter().enumerate() {
            let pu = phi_q16[u];
            let pv = phi_q16[v];
            let lo = pu.min(pv);
            let hi = pu.max(pv);
            let iso_ok = idx < 2 || hi == 0 || lo.saturating_mul(Q16_ONE) >= hi.saturating_mul(CHEEGER_BETA_Q16);
            if iso_ok {
                adj[u].push((v, w));
            }
        }
    }

    let ppr = run_ppr_rwr_q16(ns, &adj, &restart_q16);
    let max_non_top0 = ppr
        .iter()
        .enumerate()
        .filter(|&(si, _)| si != top0)
        .map(|(_, &v)| v)
        .max()
        .unwrap_or(0);
    if max_non_top0 > 0 {
        let coeff_q16 = if multi_or_partial {
            PPR_COEFF_MULTI_Q16
        } else {
            PPR_COEFF_BASE_Q16
        };
        for (si, &mass) in ppr.iter().enumerate() {
            if si != top0 && mass > 0 {
                let norm_q16 = (mass.saturating_mul(Q16_ONE)) / max_non_top0;
                out[si] = (((smax_q16.saturating_mul(coeff_q16)) >> Q16_SHIFT)
                    .saturating_mul(norm_q16))
                    >> Q16_SHIFT;
            }
        }
    }

    // Group Steiner Tree Subgraph Bridge (G-Retriever `2402.07630` / Group Steiner `2508.20543`):
    // Connect uncovered query-group terminals through minimum-cost ISO-RAG pruned Steiner paths.
    if multi_or_partial && n_qg >= 2 {
        let steiner_cap = (smax_q16.saturating_mul(7_209)) >> Q16_SHIFT; // 0.11 * smax
        let steiner_lifts = crate::retrieval::graph::group_steiner_q16::compute_group_steiner_lifts_q16(
            ns,
            &adj,
            base_sight_q16,
            &s_mask_local,
            mask_limit,
            top0,
            steiner_cap,
        );
        for si in 0..ns {
            if si != top0 && steiner_lifts[si] > 0 {
                out[si] = out[si].saturating_add(steiner_lifts[si]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppr_rwr_diffuses_across_two_hops() {
        let mut adj = vec![Vec::new(); 5];
        adj[0].push((2, Q16_ONE));
        adj[2].push((0, Q16_ONE));
        adj[2].push((4, Q16_ONE));
        adj[4].push((2, Q16_ONE));
        let mut restart = vec![0u64; 5];
        restart[0] = Q16_ONE;
        let ppr = run_ppr_rwr_q16(5, &adj, &restart);
        assert!(ppr[2] > ppr[4] && ppr[4] > 0);
        assert_eq!(ppr[3], 0);
    }
}
