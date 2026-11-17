//! GhostLink Divergent Graph Synthesis (Issue #802 / Faloutsos Connection Subgraph):
//!
//!   G_bar = (V x V) \ E
//!   Synthesis(phi_i, phi_j) = Path_min(G_bar) * (1 - Ochiai(phi_i, phi_j))
//!
//! Pure integer / Q16.16 fixed-point arithmetic (zero floats, zero word lists).

use std::collections::BTreeMap;

use super::index::Index;
use super::is_technical_handle;
use super::scoring::QueryPlan;
use super::supersession::turn_is_superseded;

pub(super) const Q16: u64 = 1 << 16;

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

    let max_df = ((ix.t_stats.n * 0.10).max(4.0)) as u32;
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
    }
}
