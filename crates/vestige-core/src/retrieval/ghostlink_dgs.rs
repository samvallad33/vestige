//! GhostLink Divergent Graph Synthesis (Issue #802 / Faloutsos Connection Subgraph):
//!
//!   G_bar = (V x V) \ E
//!   Synthesis(phi_i, phi_j) = Path_min(G_bar) * (1 - Ochiai(phi_i, phi_j))
//!   Conductance(u, v) = (Q16 / df(c)) * (1 - Ochiai(c, e_div))
//!   p^(h+1) = ((1 - alpha) * p^(0) + alpha * W * p^(h)) >> 16,  alpha = 55705 (0.85 Q16.16)
//!
//! Pure integer / Q16.16 fixed-point arithmetic (zero floats, zero open-class word lists).

use std::collections::BTreeMap;

use super::index::Index;
use super::is_technical_handle;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;
use super::viterbi::{transition_contradicts, turn_carriers};

pub(super) const Q16: u64 = 1 << 16;
pub(super) const ALPHA_Q16: u64 = 55_705; // 0.85 in Q16.16
pub(super) const ONE_MINUS_ALPHA_Q16: u64 = Q16 - ALPHA_Q16; // 9831 (0.15 in Q16.16)
const RWR_STEPS: usize = 6;
const RWR_MAX_NODES: usize = 96;

#[inline]
pub(super) fn integer_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }
    x
}

/// Integer Ochiai graph divergence `1 - inter / sqrt(|a| * |b|)` in Q16.16 `[0, 65536]`.
/// Maximal (`Q16`) when `(phi_i, phi_j)` has never co-occurred (edge in complement graph G_bar).
pub(super) fn ochiai_divergence_q16(a: &[u32], b: &[u32]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return Q16;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let denom = integer_sqrt((a.len() as u64).saturating_mul(b.len() as u64)).max(1);
    let sim_q16 = (inter.saturating_mul(Q16) / denom).min(Q16);
    Q16.saturating_sub(sim_q16)
}

/// 6-step Damped Integer Random Walk with Restart (Q16.16 RWR) over Faloutsos conductance edges
/// across multi-hop complement turns. Returns ranked unseated complement turn indices.
pub(super) fn synthesize_multihop_rwr_turns(
    ix: &Index,
    plan: &QueryPlan,
    seed_turns: &[usize],
    t_mask: &[u64],
) -> Vec<usize> {
    let nt = ix.turns.len();
    if nt <= 5 || seed_turns.is_empty() {
        return Vec::new();
    }

    let max_df = ((nt as u64 * 10) / 100).max(4) as u32;
    let mut nodes: Vec<usize> = Vec::with_capacity(RWR_MAX_NODES);
    let mut seed_count = 0usize;
    for &s in seed_turns.iter().take(6) {
        if !ix.turns[s].is_backchannel && !turn_is_superseded(ix, s) && !nodes.contains(&s) {
            nodes.push(s);
            seed_count += 1;
        }
    }
    if seed_count == 0 {
        return Vec::new();
    }

    // 3-wave rare-carrier expansion ordered by lowest df (highest Faloutsos conductance)
    let mut wave_start = 0usize;
    for _wave in 0..3 {
        if nodes.len() >= RWR_MAX_NODES || wave_start >= nodes.len() {
            break;
        }
        let wave_end = nodes.len();
        let mut carriers: BTreeMap<(u32, u32), ()> = BTreeMap::new();
        for &ti in &nodes[wave_start..wave_end] {
            for e in turn_carriers(ix, plan, ti, max_df) {
                if let Some(&df) = ix.t_stats.df.get(&e) {
                    if df >= 2 && df <= max_df {
                        carriers.insert((df, e), ());
                    }
                }
            }
        }
        wave_start = wave_end;

        for (&(_df, e), _) in &carriers {
            if let Some(post) = ix.t_postings.get(e as usize) {
                for &cand_u32 in post {
                    let cand_ti = cand_u32 as usize;
                    if cand_ti < nt
                        && !ix.turns[cand_ti].is_backchannel
                        && !turn_is_superseded(ix, cand_ti)
                        && !nodes.contains(&cand_ti)
                    {
                        nodes.push(cand_ti);
                        if nodes.len() >= RWR_MAX_NODES {
                            break;
                        }
                    }
                }
            }
            if nodes.len() >= RWR_MAX_NODES {
                break;
            }
        }
    }

    let n = nodes.len();
    if n <= seed_count {
        return Vec::new();
    }

    // Column-normalized integer transition conductance matrix `adj[u] = [(v, weight_q16)]`
    let node_carriers: Vec<Vec<u32>> = nodes
        .iter()
        .map(|&ti| turn_carriers(ix, plan, ti, max_df))
        .collect();
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];
    for ui in 0..n {
        let u_ti = nodes[ui];
        let tu = &ix.turns[u_ti];
        let u_carr = &node_carriers[ui];
        if u_carr.is_empty() {
            continue;
        }
        let mut row_raw: Vec<(usize, u64)> = Vec::new();
        let mut row_sum: u64 = 0;

        for vi in 0..n {
            if ui == vi {
                continue;
            }
            let v_ti = nodes[vi];
            if transition_contradicts(ix, u_ti, v_ti) {
                continue;
            }
            let tv = &ix.turns[v_ti];
            let v_carr = &node_carriers[vi];
            let mut cond_best: u64 = 0;
            for &c in u_carr {
                if !v_carr.contains(&c) {
                    continue;
                }
                let df_c = ix.t_stats.df.get(&c).copied().unwrap_or(0);
                if df_c < 2 || df_c > max_df {
                    continue;
                }
                let base_cond = Q16 / (df_c as u64);
                let post_c = ix.t_postings.get(c as usize).map(|p| p.as_slice()).unwrap_or(&[]);
                let mut div_best = Q16 / 2;
                for &e_div in v_carr {
                    if e_div == c || u_carr.contains(&e_div) {
                        continue;
                    }
                    let post_d = ix
                        .t_postings
                        .get(e_div as usize)
                        .map(|p| p.as_slice())
                        .unwrap_or(&[]);
                    let d = ochiai_divergence_q16(post_c, post_d);
                    if d > div_best {
                        div_best = d;
                        if div_best >= Q16 {
                            break;
                        }
                    }
                }
                let mut w = (base_cond.saturating_mul(div_best)) >> 16;
                if tu.session != tv.session {
                    w = w.saturating_mul(3) >> 1;
                }
                if (t_mask[v_ti] & !t_mask[u_ti]) != 0 {
                    w = w.saturating_add(Q16 >> 4);
                }
                if w > cond_best {
                    cond_best = w;
                }
            }
            if cond_best > 0 {
                row_raw.push((vi, cond_best));
                row_sum = row_sum.saturating_add(cond_best);
            }
        }

        if row_sum > 0 {
            adj[ui] = row_raw
                .into_iter()
                .map(|(vi, w)| (vi, ((w as u128 * Q16 as u128) / (row_sum as u128)) as u64))
                .collect();
        }
    }

    // Seed restart distribution p^(0)
    let mut p0 = vec![0u64; n];
    for rank in 0..seed_count {
        p0[rank] = Q16 / (rank as u64 + 1);
    }
    let p0_sum: u64 = p0.iter().sum::<u64>().max(1);
    for v in &mut p0 {
        *v = ((*v as u128 * Q16 as u128) / (p0_sum as u128)) as u64;
    }

    // 6-hop Damped Integer Random Walk with Restart:
    // p^(h+1) = ((1 - alpha) * p^(0) + alpha * W * p^(h)) >> 16
    let mut p = p0.clone();
    let mut next = vec![0u64; n];
    for _step in 0..RWR_STEPS {
        for v in &mut next {
            *v = 0;
        }
        for ui in 0..n {
            let pu = p[ui];
            if pu == 0 || adj[ui].is_empty() {
                continue;
            }
            for &(vi, w_q16) in &adj[ui] {
                let flow = ((pu as u128 * w_q16 as u128) >> 16) as u64;
                next[vi] = next[vi].saturating_add(flow);
            }
        }
        for i in 0..n {
            let restart = ((ONE_MINUS_ALPHA_Q16 as u128 * p0[i] as u128) >> 16) as u64;
            let walk = ((ALPHA_Q16 as u128 * next[i] as u128) >> 16) as u64;
            p[i] = restart.saturating_add(walk);
        }
    }

    let mut ranked: Vec<(usize, u64)> = Vec::new();
    for (idx, &ti) in nodes.iter().enumerate() {
        if seed_turns.contains(&ti) {
            continue;
        }
        let prob = p[idx];
        if prob > 0 {
            let len_d = integer_sqrt((ix.turns[ti].content.len() as u64) + 6).max(2);
            let score = prob / len_d;
            if score > 0 {
                ranked.push((ti, score));
            }
        }
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(4).map(|(ti, _)| ti).collect()
}

/// Find the highest-synthesis 2-hop complement turn not yet in `final_turn_indices`.
pub(super) fn synthesize_complement_turn(
    ix: &Index,
    plan: &QueryPlan,
    final_turn_indices: &[usize],
) -> Option<usize> {
    let nt = ix.turns.len();
    if nt <= 5 || final_turn_indices.is_empty() {
        return None;
    }

    let max_df = ((nt as u64 * 10) / 100).max(4) as u32;
    let mut seed_entities: BTreeMap<u32, u64> = BTreeMap::new();
    let mut seed_ent_set: Vec<u32> = Vec::new();

    for (rank, &s) in final_turn_indices.iter().take(4).enumerate() {
        let rank_w = Q16 / (rank as u64 + 1);
        for &e in &ix.turns[s].entities {
            if plan.group_of.contains_key(&e) || is_technical_handle(&ix.vocab, e) {
                continue;
            }
            let Some(&df) = ix.t_stats.df.get(&e) else {
                continue;
            };
            if df < 2 || df > max_df {
                continue;
            }
            let v = seed_entities.entry(e).or_insert(0);
            *v = (*v).max(rank_w);
            if !seed_ent_set.contains(&e) {
                seed_ent_set.push(e);
            }
        }
    }

    if seed_entities.is_empty() {
        return None;
    }

    let mut best_ti: Option<usize> = None;
    let mut best_score_q16: u64 = 0;

    for ti in 0..nt {
        let t = &ix.turns[ti];
        if t.is_backchannel
            || t.entities.is_empty()
            || turn_is_superseded(ix, ti)
            || final_turn_indices.contains(&ti)
        {
            continue;
        }
        let mut best_carrier_q16: u64 = 0;
        let mut max_div_q16: u64 = 0;
        let mut has_divergent = false;

        for &c in &t.entities {
            if let Some(&pot_q16) = seed_entities.get(&c) {
                let df_c = ix.t_stats.df.get(&c).copied().unwrap_or(2).max(2) as u64;
                let cond_q16 = pot_q16 / df_c;
                best_carrier_q16 = best_carrier_q16.saturating_add(cond_q16);
                let post_c = ix.t_postings.get(c as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                for &e_div in &t.entities {
                    if e_div == c || seed_ent_set.contains(&e_div) {
                        continue;
                    }
                    let post_div = ix.t_postings.get(e_div as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                    let div_q16 = ochiai_divergence_q16(post_c, post_div);
                    if div_q16 > max_div_q16 {
                        max_div_q16 = div_q16;
                    }
                    has_divergent = true;
                }
            }
        }

        if has_divergent && best_carrier_q16 > 0 && max_div_q16 >= (Q16 / 2) {
            let synth_q16 = (best_carrier_q16.saturating_mul(max_div_q16)) >> 16;
            let len_d = integer_sqrt((t.content.len() as u64) + 6).max(2);
            let norm_q16 = synth_q16 / len_d;
            if norm_q16 > best_score_q16 {
                best_score_q16 = norm_q16;
                best_ti = Some(ti);
            }
        }
    }

    best_ti
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ochiai_divergence_is_maximal_in_complement_graph() {
        let a = [1u32, 3, 5];
        let b = [2u32, 4, 6];
        assert_eq!(ochiai_divergence_q16(&a, &b), Q16);
        assert_eq!(ochiai_divergence_q16(&a, &a), 0);
        assert_eq!(ALPHA_Q16 + ONE_MINUS_ALPHA_Q16, Q16);
    }
}
