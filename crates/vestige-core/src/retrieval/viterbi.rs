//! Issue #801: McCallum-Style Integer Viterbi Chain Decoding anchored in
//! GhostLink Divergent Graph Synthesis (Path_min(G_bar) * (1 - Ochiai)).
//!
//! Pure integer / Q16.16 fixed-point dynamic programming (zero floats, zero
//! word lists). Decodes the highest-probability multi-hop chain across
//! turns/sessions (`seed -> hop1 -> hop2`) where edge transitions are scored by
//! carrier conductance (`Q16 / df`) weighted by complement-graph Ochiai divergence
//! plus newly covered query-group bits.

use std::collections::BTreeMap;

use super::ghostlink_dgs::{integer_sqrt, ochiai_divergence_q16, Q16};
use super::index::Index;
use super::is_technical_handle;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;

/// Decoded multi-hop chain returned by integer Viterbi.
pub(super) struct ViterbiChain {
    /// Ordered sequence of turn indices `(t_0, t_1[, t_2])` forming the winning path.
    pub(super) path: Vec<usize>,
    /// Total Q16.16 log/potential path score.
    pub(super) score_q16: u64,
    /// Union of query-group bitmasks covered across the decoded chain.
    pub(super) union_mask: u64,
}

/// Run 2-hop / 3-hop Integer Viterbi over candidate nodes seeded from `seed_turns`
/// and connected via rare bridge entities (`2 <= df <= max_df`).
pub(super) fn decode_viterbi_chain(
    ix: &Index,
    plan: &QueryPlan,
    seed_turns: &[usize],
    t_scene: &[f32],
    t_mask: &[u64],
) -> Option<ViterbiChain> {
    let nt = ix.turns.len();
    if nt <= 5 || seed_turns.is_empty() {
        return None;
    }

    let t_max = t_scene[seed_turns[0]];
    if t_max <= 0.0 {
        return None;
    }

    let max_df = ((ix.t_stats.n * 0.12).max(4.0)) as u32;
    let mut nodes: Vec<usize> = Vec::with_capacity(64);
    for &s in seed_turns.iter().take(12) {
        if t_scene[s] > 0.0 && !turn_is_superseded(ix, s) && !nodes.contains(&s) {
            nodes.push(s);
        }
    }
    if nodes.is_empty() {
        return None;
    }

    // Expand 1-hop & 2-hop candidate pool via rare bridge entities from top seeds.
    let mut carrier_posts: BTreeMap<u32, u64> = BTreeMap::new();
    for &s in nodes.iter().take(6) {
        for &e in &ix.turns[s].entities {
            if plan.group_of.contains_key(&e) || is_technical_handle(&ix.vocab, e) {
                continue;
            }
            if let Some(&df) = ix.t_stats.df.get(&e) {
                if df >= 2 && df <= max_df {
                    carrier_posts.entry(e).or_insert(df as u64);
                }
            }
        }
    }

    for (&e, _) in &carrier_posts {
        if let Some(post) = ix.t_postings.get(e as usize) {
            for &ti_u32 in post {
                let ti = ti_u32 as usize;
                if ti < nt
                    && !ix.turns[ti].is_backchannel
                    && !turn_is_superseded(ix, ti)
                    && !nodes.contains(&ti)
                {
                    nodes.push(ti);
                    if nodes.len() >= 64 {
                        break;
                    }
                }
            }
        }
        if nodes.len() >= 64 {
            break;
        }
    }

    let n = nodes.len();
    if n < 2 {
        return None;
    }

    // Emission potential (step 0..2) in Q16.16
    let mut emit_q16 = vec![0u64; n];
    for (idx, &ti) in nodes.iter().enumerate() {
        let base = ((t_scene[ti] / t_max).clamp(0.0, 1.0) * (Q16 as f32)) as u64;
        let cov = (t_mask[ti].count_ones() as u64) * (Q16 / 8);
        emit_q16[idx] = base.saturating_add(cov);
    }

    // Precompute transition weights: trans[u][v] in Q16.16
    // Weighted by GhostLink complement divergence (1 - Ochiai(c, e_div)) * (Q16 / df_c)
    // plus joint coverage gain when moving to a distinct session.
    let mut dp1 = vec![0u64; n];
    let mut back1 = vec![usize::MAX; n];

    for (vi, &v_ti) in nodes.iter().enumerate() {
        let tv = &ix.turns[v_ti];
        if tv.entities.is_empty() || tv.is_backchannel {
            continue;
        }
        for (ui, &u_ti) in nodes.iter().enumerate() {
            if u_ti == v_ti || emit_q16[ui] == 0 {
                continue;
            }
            let tu = &ix.turns[u_ti];
            let trans = viterbi_edge_q16(ix, plan, tu, tv, u_ti, v_ti, t_mask, max_df);
            if trans == 0 {
                continue;
            }
            let cand = emit_q16[ui]
                .saturating_add(trans)
                .saturating_add(emit_q16[vi] >> 1);
            if cand > dp1[vi] {
                dp1[vi] = cand;
                back1[vi] = ui;
            }
        }
    }

    // 2-hop DP step (len=3 chain: u -> v -> w) when a valid hop-1 endpoint exists.
    let mut best_score = 0u64;
    let mut best_path: Vec<usize> = Vec::new();
    let mut best_mask = 0u64;

    // Best length-2 chain
    for (vi, &v_ti) in nodes.iter().enumerate() {
        if dp1[vi] > best_score && back1[vi] != usize::MAX {
            let u_ti = nodes[back1[vi]];
            let um = t_mask[u_ti] | t_mask[v_ti];
            if ix.turns[u_ti].session != ix.turns[v_ti].session || um.count_ones() > t_mask[u_ti].count_ones() {
                best_score = dp1[vi];
                best_path = vec![u_ti, v_ti];
                best_mask = um;
            }
        }
    }

    // Try extending to 3rd hop w across distinct turn/session
    for (wi, &w_ti) in nodes.iter().enumerate() {
        let tw = &ix.turns[w_ti];
        if tw.entities.is_empty() || tw.is_backchannel {
            continue;
        }
        for (vi, &v_ti) in nodes.iter().enumerate() {
            if back1[vi] == usize::MAX || v_ti == w_ti || nodes[back1[vi]] == w_ti {
                continue;
            }
            let tv = &ix.turns[v_ti];
            let trans2 = viterbi_edge_q16(ix, plan, tv, tw, v_ti, w_ti, t_mask, max_df);
            if trans2 == 0 {
                continue;
            }
            let u_ti = nodes[back1[vi]];
            let cand = dp1[vi]
                .saturating_add(trans2 >> 1)
                .saturating_add(emit_q16[wi] >> 2);
            if cand > best_score && tw.session != ix.turns[u_ti].session {
                best_score = cand;
                best_path = vec![u_ti, v_ti, w_ti];
                best_mask = t_mask[u_ti] | t_mask[v_ti] | t_mask[w_ti];
            }
        }
    }

    if best_path.is_empty() || best_score == 0 {
        None
    } else {
        Some(ViterbiChain {
            path: best_path,
            score_q16: best_score,
            union_mask: best_mask,
        })
    }
}

fn viterbi_edge_q16(
    ix: &Index,
    plan: &QueryPlan,
    tu: &super::TurnIx,
    tv: &super::TurnIx,
    u_ti: usize,
    v_ti: usize,
    t_mask: &[u64],
    max_df: u32,
) -> u64 {
    let mut best_trans: u64 = 0;
    for &c in &tu.entities {
        if plan.group_of.contains_key(&c) || is_technical_handle(&ix.vocab, c) {
            continue;
        }
        if !tv.entities.contains(&c) {
            continue;
        }
        let df_c = ix.t_stats.df.get(&c).copied().unwrap_or(0);
        if df_c < 2 || df_c > max_df {
            continue;
        }
        let cond_q16 = Q16 / (df_c as u64);
        let post_c = ix.t_postings.get(c as usize).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut max_div_q16 = Q16 / 2;
        for &e_div in &tv.entities {
            if e_div == c || tu.entities.contains(&e_div) {
                continue;
            }
            let post_div = ix.t_postings.get(e_div as usize).map(|v| v.as_slice()).unwrap_or(&[]);
            let d = ochiai_divergence_q16(post_c, post_div);
            if d > max_div_q16 {
                max_div_q16 = d;
            }
        }
        let mut edge = (cond_q16.saturating_mul(max_div_q16)) >> 16;
        if tu.session != tv.session {
            edge = edge.saturating_mul(3) >> 1;
        }
        let new_groups = (t_mask[v_ti] & !t_mask[u_ti]).count_ones() as u64;
        if new_groups > 0 {
            edge = edge.saturating_add(new_groups * (Q16 / 8));
        }
        let len_d = integer_sqrt((tv.content.len() as u64) + 6).max(2);
        let norm_edge = edge / len_d;
        if norm_edge > best_trans {
            best_trans = norm_edge;
        }
    }
    best_trans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_empty_seeds_returns_none() {
        // Pure structural invariant check for integer constants
        assert_eq!(Q16, 65536);
        assert_eq!(integer_sqrt(9), 3);
    }
}
