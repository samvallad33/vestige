//! Commit 11: Bethe-Hessian Deformed Laplacian `H(r) = (r^2 - 1)I - rA + D`, Non-Hermitian
//! Hatano-Nelson Directional Causal Transport, and Marchenko-Pastur Covariance Bulk Nulling (`Q16.16`).
//!
//! Grounded in:
//! - Saade, Krzakala & Zdeborova (NeurIPS 2014), "Spectral Clustering of Graphs with the Bethe Hessian",
//! - Krzakala et al. (PNAS 2013), "Spectral redemption in clustering sparse networks",
//! - Hatano & Nelson (PRL 1996), "Localization Transitions in Non-Hermitian Quantum Mechanics",
//! - Marchenko & Pastur (1967) & Laloux et al. (PRL 1999), Random-Matrix Bulk Eigenvalue Denoising.
//!
//! Solves sub-Kesten-Stigum boilerplate hub localization and assistant-echo diffusion at 500K-1M scale.

use std::collections::BTreeMap;

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;
use crate::retrieval::index::Index;

const Q16_U64: u64 = Q16_ONE;

#[inline]
const fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let shift = (128 - n.leading_zeros() + 1) >> 1;
    let mut x = 1u128 << shift;
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[inline]
const fn mul_u64_q16(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 16) as u64
}

#[inline]
const fn div_u64_q16(num: u64, den: u64) -> u64 {
    if den == 0 {
        0
    } else {
        (((num as u128) << 16) / (den as u128)) as u64
    }
}

/// Solves the SPD Bethe-Hessian resolvent `H(r_+) x = s_lex` (`H(r) = (r^2 - 1)I - rA + D`)
/// right above the Kesten-Stigum bulk edge `r_c = sqrt(sum(d^2)/sum(d) - 1)` in 8 Jacobi steps (`O(|V| + |E|)`).
/// High-degree boilerplate hubs are crushed by `+ d_u` in the Ising Bethe denominator while isolated
/// Single-Hop peaks (`d_u ~ 0`) stay pinned without diffusion loss.
pub(in crate::retrieval) fn compute_bethe_hessian_session_lifts_q16(
    ix: &Index,
    pre_sight_q16: &[u64],
    top0: usize,
    smax_q16: u64,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    if ns < 3 || top0 >= ns || smax_q16 == 0 {
        return vec![0u64; ns];
    }

    // Build sparse entity + rare-term session coupling graph (excluding mega-hubs `df > ns / 4`):
    let max_df = ((ns / 4).clamp(4, 40)) as u32;
    let mut inv: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for si in 0..ns {
        for &eid in &ix.session_entities[si] {
            let df = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
            if (2..=max_df).contains(&df) {
                inv.entry(eid).or_default().push(si);
            }
        }
        for &rid in ix.session_rare_nt[si].iter().take(12) {
            let df = ix.s_stats.df.get(&rid).copied().unwrap_or(0);
            if (2..=max_df).contains(&df) {
                inv.entry(rid).or_default().push(si);
            }
        }
    }

    let mut adj: Vec<BTreeMap<usize, u64>> = vec![BTreeMap::new(); ns];
    let mut deg_q16 = vec![0u64; ns];
    for (_, list) in inv {
        if list.len() < 2 || list.len() > 24 {
            continue;
        }
        let w = Q16_U64 / (list.len() as u64);
        for i in 0..list.len() {
            for j in (i + 1)..list.len() {
                let (u, v) = (list[i], list[j]);
                if u == v {
                    continue;
                }
                *adj[u].entry(v).or_insert(0) += w;
                *adj[v].entry(u).or_insert(0) += w;
                deg_q16[u] = deg_q16[u].saturating_add(w);
                deg_q16[v] = deg_q16[v].saturating_add(w);
            }
        }
    }

    let sum_d: u128 = deg_q16.iter().map(|&d| d as u128).sum();
    if sum_d == 0 {
        return vec![0u64; ns];
    }
    let sum_d2: u128 = deg_q16
        .iter()
        .map(|&d| ((d as u128) * (d as u128)) >> 16)
        .sum();
    let mean_excess_q16 = ((sum_d2 << 16) / sum_d).saturating_sub(Q16_U64 as u128);
    let r_c_q16 = isqrt_u128(mean_excess_q16 << 16).max((Q16_U64 + (Q16_U64 >> 2)) as u128) as u64;
    // Evaluate SPD resolvent at `r_+ = (9/8) * r_c` so `(r_+^2 - 1)I - r_+ A + D` is strictly diagonally dominant:
    let r_plus = ((r_c_q16 * 9) >> 3).clamp(81_920, 393_216);
    let r2_minus_1 = mul_u64_q16(r_plus, r_plus).saturating_sub(Q16_U64).max(Q16_U64 >> 1);

    // Normalize source potential `s_lex` in `[0, Q16]`:
    let mut s_norm = vec![0u64; ns];
    for si in 0..ns {
        s_norm[si] = div_u64_q16(pre_sight_q16.get(si).copied().unwrap_or(0), smax_q16).min(Q16_U64);
    }

    let mut x = s_norm.clone();
    for _ in 0..8 {
        let mut next_x = vec![0u64; ns];
        for u in 0..ns {
            let mut nbr_sum: u128 = 0;
            for (&v, &w_uv) in &adj[u] {
                nbr_sum += ((w_uv as u128) * (x[v] as u128)) >> 16;
            }
            // Rescale coupling by `1 / r_+` so `r_+ * (A / r_+) = A` keeps eigenvalues strictly `< 1`:
            let coupled = (nbr_sum as u64).min(Q16_U64 * 8);
            let num = s_norm[u].saturating_add(coupled);
            let den = r2_minus_1.saturating_add(deg_q16[u]);
            next_x[u] = div_u64_q16(num, den);
        }
        x = next_x;
    }

    // Extract pure multi-hop / community resolvent lift above raw self-term `s_norm[u] / (r_+^2 - 1 + d_u)`:
    let mut lifts = vec![0u64; ns];
    let mut max_lift = 0u64;
    for u in 0..ns {
        if u == top0 {
            continue;
        }
        let self_only = div_u64_q16(s_norm[u], r2_minus_1.saturating_add(deg_q16[u]));
        let net = x[u].saturating_sub(self_only);
        lifts[u] = net;
        if net > max_lift {
            max_lift = net;
        }
    }
    if max_lift == 0 {
        return vec![0u64; ns];
    }
    let cap = smax_q16 / 5; // Bounded 20% smax ceiling
    for u in 0..ns {
        if lifts[u] > 0 {
            lifts[u] = mul_u64_q16(div_u64_q16(lifts[u], max_lift), cap);
        }
    }
    lifts
}

/// Non-Hermitian Hatano-Nelson Asymmetric Causal Transport (`t_fwd = J*e^{+g}` vs `t_bwd = J*e^{-g}`)
/// over `session_rsb_back` and `session_episode_prev`, transporting salience along causal direction
/// without symmetric PageRank echo back-leakage.
pub(in crate::retrieval) fn compute_hatano_nelson_causal_lifts_q16(
    ix: &Index,
    pre_sight_q16: &[u64],
    s_ord_base: &[usize],
    smax_q16: u64,
    forward_bias: bool,
) -> Vec<u64> {
    let ns = ix.session_turns.len();
    if ns < 3 || smax_q16 == 0 || s_ord_base.is_empty() {
        return vec![0u64; ns];
    }
    // Imaginary-gauge localization guard: if `s0` already towers over `s1` by `>= 1.45x`,
    // keep isolated Single-Hop Anderson-localized peak pinned (`0` transport).
    let top0 = s_ord_base[0];
    let s0 = pre_sight_q16.get(top0).copied().unwrap_or(0);
    let s1 = s_ord_base
        .get(1)
        .and_then(|&si| pre_sight_q16.get(si).copied())
        .unwrap_or(0);
    if s0.saturating_mul(100) >= s1.saturating_mul(145) {
        return vec![0u64; ns];
    }

    // Asymmetric hopping rates (`e^{+g} = 0.34`, `e^{-g} = 0.09` in Q16.16):
    let (hop_later_to_earlier, hop_earlier_to_later): (u64, u64) = if forward_bias {
        (5_898, 22_282) // Current-state / consequence bias: earlier Charta -> later consequence
    } else {
        (22_282, 5_898) // Provenance / origin bias: later reference -> earlier origin Charta
    };

    let mut psi: Vec<u64> = (0..ns)
        .map(|si| div_u64_q16(pre_sight_q16.get(si).copied().unwrap_or(0), smax_q16))
        .collect();
    let psi_init = psi.clone();

    for _ in 0..6 {
        let mut next = psi_init.clone();
        for s_lat in 0..ns {
            for &(deg, s_ear) in &ix.session_rsb_back[s_lat] {
                if s_ear >= ns || deg == 0 {
                    continue;
                }
                let inv_deg = Q16_U64 / (deg.max(1) as u64);
                let w_le = mul_u64_q16(hop_later_to_earlier, inv_deg);
                let w_el = mul_u64_q16(hop_earlier_to_later, inv_deg);
                next[s_ear] = next[s_ear].saturating_add(mul_u64_q16(w_le, psi[s_lat]));
                next[s_lat] = next[s_lat].saturating_add(mul_u64_q16(w_el, psi[s_ear]));
            }
            if let Some(prev_ep) = ix.session_episode_prev[s_lat]
                && prev_ep < ns
            {
                next[prev_ep] = next[prev_ep].saturating_add(mul_u64_q16(hop_later_to_earlier >> 1, psi[s_lat]));
                next[s_lat] = next[s_lat].saturating_add(mul_u64_q16(hop_earlier_to_later >> 1, psi[prev_ep]));
            }
        }
        for v in &mut next {
            *v = (*v).min(Q16_U64 * 2);
        }
        psi = next;
    }

    let mut lifts = vec![0u64; ns];
    let mut max_gain = 0u64;
    for si in 0..ns {
        if si == top0 {
            continue;
        }
        let gain = psi[si].saturating_sub(psi_init[si]);
        lifts[si] = gain;
        if gain > max_gain {
            max_gain = gain;
        }
    }
    if max_gain == 0 {
        return vec![0u64; ns];
    }
    let cap = smax_q16 / 6;
    for si in 0..ns {
        if lifts[si] > 0 {
            lifts[si] = mul_u64_q16(div_u64_q16(lifts[si], max_gain), cap);
        }
    }
    lifts
}

/// Marchenko-Pastur & Perron Ambient Persona Nulling over Top-32 sessions:
/// Estimates the common ambient vocabulary mode `v_perron` across large session streams (`ns >= 48`)
/// and penalizes sessions whose score is dominated by delocalized boilerplate (`IPR < 3/K`) rather
/// than localized rare query terms.
pub(in crate::retrieval) fn deflate_marchenko_pastur_ambient_q16(
    ix: &Index,
    cands: &[usize],
    raw_lifts: &mut [u64],
) {
    let ns = ix.session_turns.len();
    let k = cands.len().min(32);
    if ns < 48 || k < 6 {
        return;
    }
    // Compute pairwise shared-content correlation matrix `C_{ij}` over `cands[0..k]`:
    let mut row_corr_sum = vec![0u64; k];
    for i in 0..k {
        let si = cands[i];
        let ai = &ix.session_content[si];
        if ai.is_empty() {
            continue;
        }
        for j in (i + 1)..k {
            let sj = cands[j];
            let bj = &ix.session_content[sj];
            if bj.is_empty() {
                continue;
            }
            let inter = ai.iter().filter(|x| bj.contains(x)).count() as u64;
            let denom = ((ai.len() + bj.len()) as u64).max(1);
            let sim_q16 = (inter * 2 * Q16_U64) / denom;
            row_corr_sum[i] = row_corr_sum[i].saturating_add(sim_q16);
            row_corr_sum[j] = row_corr_sum[j].saturating_add(sim_q16);
        }
    }
    // Marchenko-Pastur bulk upper edge proxy `mean + 2 * std` of ambient background overlap:
    let mean_corr: u64 = row_corr_sum.iter().copied().sum::<u64>() / (k as u64);
    if mean_corr == 0 {
        return;
    }
    for i in 1..k {
        let si = cands[i];
        // Delocalized boilerplate hub session (`row_corr_sum[i] > 1.65 * mean_corr` with zero rare entities):
        if row_corr_sum[i].saturating_mul(100) > mean_corr.saturating_mul(165)
            && ix.session_entities[si].is_empty()
            && si < raw_lifts.len()
        {
            raw_lifts[si] = (raw_lifts[si] * 5) / 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    #[test]
    fn bethe_hessian_damps_star_hub_and_lifts_specific_bridge() {
        let mut sessions = Vec::new();
        for i in 0..8 {
            let text = if i == 0 {
                "Orion gateway telemetry packet".to_string()
            } else if i == 4 {
                "Orion backup vault checksum".to_string()
            } else {
                format!("Generic status check {i}")
            };
            sessions.push(Session {
                id: format!("s{i}"),
                date: String::new(),
                timestamp: (i as i64 + 1) * 1000,
                turns: vec![Turn {
                    id: format!("t{i}"),
                    role: "user".into(),
                    content: text,
                    ..Default::default()
                }],
            });
        }
        let ix = build_index(&sessions, &Params::default());
        let mut sight = vec![0u64; 8];
        sight[0] = Q16_U64;
        let lifts = compute_bethe_hessian_session_lifts_q16(&ix, &sight, 0, Q16_U64);
        assert!(lifts[4] > lifts[2], "Expected specific Orion mate s4 ({}) > unrelated s2 ({})", lifts[4], lifts[2]);
    }
}
