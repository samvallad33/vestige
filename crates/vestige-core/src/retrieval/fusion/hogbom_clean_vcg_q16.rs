//! Commit 10: Damped Hogbom CLEAN-OMP Interferometric Deconvolution, Green's Commutative
//! Provenance Semiring (`K[X]`), Jonker-Volgenant Bipartite Facet Marriage, and VCG Clarke-Pivot
//! Externality Slate Selection (`Q16.16`).
//!
//! Grounded in:
//! - Hogbom (1974), "Aperture Synthesis with a Non-Regular Distribution of Interferometer Baselines",
//! - Tropp & Gilbert (2007), "Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit",
//! - Green, Karvounarakis & Tannen (2007), "Provenance Semirings" (PODS '07),
//! - Jonker & Volgenant (1987), "A Shortest Augmenting Path Algorithm for Linear Assignment",
//! - Vickrey (1961), Clarke (1971), & Groves (1973), VCG Marginal Externality Pricing.
//!
//! 100% deterministic `Q16.16` integer arithmetic. Locks Slot 0 (`r^(0) == q`) for zero `@1` regression.

pub(in crate::retrieval) const Q16_ONE: i64 = 65_536;

#[inline]
const fn mul_q16(a: i64, b: i64) -> i64 {
    ((a as i128 * b as i128) >> 16) as i64
}

#[inline]
const fn div_q16(num: i64, den: i64) -> i64 {
    if den == 0 {
        0
    } else {
        (((num as i128) << 16) / (den as i128)) as i64
    }
}

/// Damped Hogbom CLEAN + Modified Gram-Schmidt Orthogonal Matching Pursuit (`O(top_s * M * K)`).
/// At `step == 0`, `r^(0) == q_ic`, preserving the exact Slot-0 `@1` winner when `pin_first` holds.
/// Subsequent steps peel the orthogonalized dirty-beam sidelobe of selected candidates from `r_q`,
/// collapsing Hop-1 paraphrase clones to `0` and unmasking faint Hop-2/Hop-3 supporting turns.
pub(in crate::retrieval) fn hogbom_omp_peel_q16(
    w_mk: &[Vec<i64>],
    q_ic: &[i64],
    k_cand: usize,
    top_s: usize,
    loop_gain_q16: i64,
    min_energy_frac_q16: i64,
    pin_first: Option<usize>,
) -> Vec<usize> {
    let m_terms = q_ic.len();
    if m_terms == 0 || k_cand == 0 || top_s == 0 {
        return Vec::new();
    }
    let mut r_q = q_ic.to_vec();
    let total_q_ic: i64 = q_ic.iter().copied().sum();
    if total_q_ic <= 0 {
        return Vec::new();
    }
    let min_gain = mul_q16(total_q_ic, min_energy_frac_q16).max(1);

    let mut selected: Vec<usize> = Vec::with_capacity(top_s);
    let mut used = vec![false; k_cand];
    let mut ortho_cols: Vec<Vec<i64>> = Vec::with_capacity(top_s);
    let mut ortho_norms: Vec<i64> = Vec::with_capacity(top_s);

    for step in 0..top_s.min(k_cand) {
        let best_u = if step == 0
            && let Some(p0) = pin_first
            && p0 < k_cand
        {
            p0
        } else {
            let mut bu = usize::MAX;
            let mut b_dirty = 0i64;
            for (u, is_used) in used.iter().enumerate().take(k_cand) {
                if *is_used {
                    continue;
                }
                let mut dirty_u = 0i64;
                for m in 0..m_terms {
                    dirty_u = dirty_u.saturating_add(mul_q16(w_mk[m][u], r_q[m]));
                }
                if dirty_u > b_dirty {
                    b_dirty = dirty_u;
                    bu = u;
                }
            }
            if bu == usize::MAX || (step > 0 && b_dirty <= min_gain) {
                break;
            }
            bu
        };

        let mut col_u: Vec<i64> = (0..m_terms).map(|m| w_mk[m][best_u]).collect();
        for (j, prev_col) in ortho_cols.iter().enumerate() {
            if ortho_norms[j] <= 0 {
                continue;
            }
            let mut dot_q = 0i64;
            for m in 0..m_terms {
                dot_q = dot_q.saturating_add(mul_q16(mul_q16(col_u[m], prev_col[m]), q_ic[m]));
            }
            let proj = div_q16(dot_q, ortho_norms[j]);
            for m in 0..m_terms {
                col_u[m] = col_u[m].saturating_sub(mul_q16(proj, prev_col[m]));
            }
        }

        let mut norm_u = 0i64;
        let mut numer_u = 0i64;
        for m in 0..m_terms {
            norm_u = norm_u.saturating_add(mul_q16(mul_q16(col_u[m], col_u[m]), q_ic[m]));
            numer_u = numer_u.saturating_add(mul_q16(col_u[m], r_q[m]));
        }
        if step > 0 && numer_u <= min_gain {
            break;
        }
        if norm_u > 0 && numer_u > 0 {
            let alpha = mul_q16(loop_gain_q16, div_q16(numer_u, norm_u));
            for m in 0..m_terms {
                let peel = mul_q16(alpha, mul_q16(col_u[m], q_ic[m]));
                r_q[m] = (r_q[m] - peel).max(0);
            }
            ortho_cols.push(col_u);
            ortho_norms.push(norm_u);
        }
        used[best_u] = true;
        selected.push(best_u);
    }
    selected
}

/// Square-free Witness Monomial in Green's Absorptive Provenance Semiring `K[X]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::retrieval) struct WitnessMonomial {
    pub(in crate::retrieval) cost_q16: i64,
    pub(in crate::retrieval) clause_mask: u64,
    pub(in crate::retrieval) visit_mask: u64,
    pub(in crate::retrieval) path_len: u8,
    pub(in crate::retrieval) path: [u32; 5],
}

/// Loop-erased Provenance Semiring skyline DP (`O(W * |E|)`) preventing 2-cycle walk inflation
/// and extracting non-dominated multi-hop witness turn tuples.
pub(in crate::retrieval) fn provenance_semiring_skyline_q16(
    n_nodes: usize,
    base_clauses: &[u64],
    base_cost_q16: &[i64],
    edges: &[(usize, usize, i64)],
    max_hops: usize,
    skyline_cap: usize,
) -> Vec<WitnessMonomial> {
    if n_nodes == 0 || edges.is_empty() {
        return Vec::new();
    }
    let mut poly: Vec<Vec<WitnessMonomial>> = (0..n_nodes)
        .map(|u| {
            if base_clauses[u] == 0 {
                return Vec::new();
            }
            let mut path = [u32::MAX; 5];
            path[0] = u as u32;
            vec![WitnessMonomial {
                cost_q16: base_cost_q16[u],
                clause_mask: base_clauses[u],
                visit_mask: 1u64 << (u & 63),
                path_len: 1,
                path,
            }]
        })
        .collect();

    for _ in 1..max_hops.min(5) {
        let mut next_poly = poly.clone();
        for &(u, v, edge_cost) in edges {
            if u >= n_nodes || v >= n_nodes || base_clauses[v] == 0 {
                continue;
            }
            let v_bit = 1u64 << (v & 63);
            for m_u in &poly[u] {
                if (m_u.visit_mask & v_bit) != 0 || (m_u.path_len as usize) >= 5 {
                    continue;
                }
                if (base_clauses[v] & !m_u.clause_mask) == 0 {
                    continue;
                }
                let mut path = m_u.path;
                path[m_u.path_len as usize] = v as u32;
                let cand = WitnessMonomial {
                    cost_q16: m_u.cost_q16.saturating_add(edge_cost).saturating_add(base_cost_q16[v]),
                    clause_mask: m_u.clause_mask | base_clauses[v],
                    visit_mask: m_u.visit_mask | v_bit,
                    path_len: m_u.path_len + 1,
                    path,
                };
                let dominated = next_poly[v].iter().any(|m_a| {
                    (m_a.clause_mask & cand.clause_mask) == cand.clause_mask
                        && m_a.cost_q16 <= cand.cost_q16
                });
                if !dominated {
                    next_poly[v].retain(|m_b| {
                        !((cand.clause_mask & m_b.clause_mask) == m_b.clause_mask
                            && cand.cost_q16 <= m_b.cost_q16)
                    });
                    next_poly[v].push(cand);
                    next_poly[v].sort_by_key(|m| (-(m.clause_mask.count_ones() as i32), m.cost_q16));
                    next_poly[v].truncate(skyline_cap);
                }
            }
        }
        poly = next_poly;
    }

    let mut global_skyline: Vec<WitnessMonomial> = poly
        .into_iter()
        .flatten()
        .filter(|m| m.path_len >= 2)
        .collect();
    global_skyline.sort_by_key(|m| (-(m.clause_mask.count_ones() as i32), m.cost_q16, m.path[0]));
    global_skyline.truncate(skyline_cap);
    global_skyline
}

/// Slot-0-Anchored Jonker-Volgenant Shortest Augmenting Path Assignment + VCG Clarke-Pivot
/// Marginal Externality Filter (`O(R^2 * C)`).
/// Assigns `rows` (uncovered query facets or chronological strata, `R <= 9`) 1-to-1 to distinct
/// candidates (`cols`, `C <= 64`) maximizing total `Q16.16` bipartite welfare, then filters out any
/// redundant clone whose VCG Clarke externality `W(S) - W(S \ {m})` is zero when an uncovered atom exists.
pub(in crate::retrieval) fn jonker_volgenant_vcg_slate_q16(
    payoff_rc: &[Vec<i64>], // [R][C] non-negative Q16.16 affinity of candidate `c` for facet/stratum `r`
    atom_masks: &[u64],     // [C] bitmask of distinct information atoms covered by candidate `c`
    atom_ic_q16: &[i64],    // [<=64] Q16.16 value per information atom
    pin_col0: usize,        // Locked @1 winner at Slot 0
    target_k: usize,
) -> Vec<usize> {
    let n_cols = atom_masks.len();
    if n_cols == 0 || target_k == 0 {
        return Vec::new();
    }
    let mut slate: Vec<usize> = Vec::with_capacity(target_k);
    let c0 = pin_col0.min(n_cols - 1);
    slate.push(c0);
    if target_k == 1 || n_cols == 1 {
        return slate;
    }

    let r_raw = payoff_rc.len();
    let r_eff = r_raw.min(target_k - 1).min(n_cols - 1);
    if r_eff > 0 {
        // Classic Kuhn-Munkres / Jonker-Volgenant shortest augmenting path on minimization cost `-payoff`:
        let max_w = payoff_rc
            .iter()
            .take(r_eff)
            .flat_map(|row| row.iter().copied())
            .max()
            .unwrap_or(0)
            .max(1);
        // 1-indexed potentials for clean augmenting path backpointers (`0` = dummy free sentinel)
        let dim_r = r_eff;
        let dim_c = n_cols;
        let mut u = vec![0i64; dim_r + 1];
        let mut v = vec![0i64; dim_c + 1];
        let mut p = vec![0usize; dim_c + 1];
        let mut way = vec![0usize; dim_c + 1];

        for i in 1..=dim_r {
            p[0] = i;
            let mut j0 = 0usize;
            let mut minv = vec![i64::MAX / 4; dim_c + 1];
            let mut used = vec![false; dim_c + 1];
            loop {
                used[j0] = true;
                let i0 = p[j0];
                let mut delta = i64::MAX / 4;
                let mut j1 = 0usize;
                for j in 1..=dim_c {
                    if used[j] || (j - 1) == c0 {
                        continue;
                    }
                    let w_ij = payoff_rc[i0 - 1].get(j - 1).copied().unwrap_or(0).max(0);
                    let cost = max_w - w_ij;
                    let cur = cost - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
                if j1 == 0 {
                    break;
                }
                for j in 0..=dim_c {
                    if used[j] {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
                if p[j0] == 0 {
                    break;
                }
            }
            loop {
                let j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }

        let mut row_match = vec![usize::MAX; dim_r];
        for j in 1..=dim_c {
            if p[j] > 0 && p[j] <= dim_r {
                row_match[p[j] - 1] = j - 1;
            }
        }
        for &col in &row_match {
            if col < n_cols
                && col != c0
                && !slate.contains(&col)
                && slate.len() < target_k
            {
                slate.push(col);
            }
        }
    }

    // VCG Clarke-Pivot Marginal Externality Completion & Clone Eviction:
    // Any remaining seat (or zero-marginal-atom clone in `1..slate.len()`) is filled by maximizing
    // exact marginal atom IC `Delta_VCG(c | S) = sum_{a in A(c) \ A(S)} IC(a)`.
    let mask_ic = |mut m: u64| -> i64 {
        let mut s = 0i64;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            s = s.saturating_add(atom_ic_q16.get(b).copied().unwrap_or(Q16_ONE >> 2));
            m &= m - 1;
        }
        s
    };

    // Evict zero-externality clones in Slots `1..slate.len()` if an unselected candidate adds positive VCG welfare:
    for slot in 1..slate.len() {
        let mut others_mask = 0u64;
        for (idx, &sc) in slate.iter().enumerate() {
            if idx != slot {
                others_mask |= atom_masks[sc];
            }
        }
        let cur_c = slate[slot];
        let cur_vcg = mask_ic(atom_masks[cur_c] & !others_mask);
        if cur_vcg == 0 {
            let mut best_rep: Option<(i64, usize)> = None;
            for (cand_c, &cm) in atom_masks.iter().enumerate() {
                if slate.contains(&cand_c) {
                    continue;
                }
                let gain = mask_ic(cm & !others_mask);
                if gain > 0 && best_rep.is_none_or(|(bg, _)| gain > bg) {
                    best_rep = Some((gain, cand_c));
                }
            }
            if let Some((_, win_c)) = best_rep {
                slate[slot] = win_c;
            }
        }
    }

    while slate.len() < target_k && slate.len() < n_cols {
        let cur_union = slate.iter().fold(0u64, |acc, &c| acc | atom_masks[c]);
        let mut best_next: Option<(i64, usize)> = None;
        for (cand_c, &cm) in atom_masks.iter().enumerate() {
            if slate.contains(&cand_c) {
                continue;
            }
            let gain = mask_ic(cm & !cur_union);
            if best_next.is_none_or(|(bg, _)| gain > bg) {
                best_next = Some((gain, cand_c));
            }
        }
        if let Some((_, nc)) = best_next {
            slate.push(nc);
        } else {
            break;
        }
    }

    slate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hogbom_clean_omp_peels_hop1_clone_and_surfaces_hop2_needle() {
        // 3 terms: t0 (Hop-1 entity, IC=4.0), t1 (Hop-1 verb, IC=2.0), t2 (Hop-2 rare bridge, IC=1.5)
        let q_ic = vec![Q16_ONE * 4, Q16_ONE * 2, Q16_ONE + (Q16_ONE >> 1)];
        // Cand 0: Hop-1 anchor (hits t0, t1)
        // Cand 1: Hop-1 near-duplicate clone (hits t0, t1 slightly weaker)
        // Cand 2: Hop-2 quiet supporting turn (hits ONLY t2 -> lower raw dirty score than Cand 1!)
        let w_mk = vec![
            vec![Q16_ONE, 58_000, 0],
            vec![Q16_ONE, 58_000, 0],
            vec![0, 0, Q16_ONE],
        ];
        let picked = hogbom_omp_peel_q16(&w_mk, &q_ic, 3, 2, 62_259, 3_277, Some(0));
        assert_eq!(picked, vec![0, 2], "CLEAN-OMP must peel Cand 1 clone and pick Hop-2 Cand 2");
    }

    #[test]
    fn jonker_volgenant_vcg_slate_locks_slot0_and_evicts_zero_externality_clone() {
        let payoff = vec![
            vec![Q16_ONE * 4, Q16_ONE * 3, Q16_ONE],
            vec![Q16_ONE * 4, Q16_ONE * 3, Q16_ONE * 2],
        ];
        // Cand 0 & Cand 1 are clones covering atom 0b001; Cand 2 covers uncovered atom 0b010
        let masks = vec![0b001u64, 0b001u64, 0b010u64];
        let ic = vec![Q16_ONE * 3, Q16_ONE * 2];
        let slate = jonker_volgenant_vcg_slate_q16(&payoff, &masks, &ic, 0, 2);
        assert_eq!(slate, vec![0, 2]);
    }
}
