//! Commit 9: Signed Cellular Sheaf Kron Reduction, Birkhoff-Rota Mobius Poset Deconvolution,
//! Finite Heyting Intuitionistic Residuation, and Augmented Forman-Ricci Curvature Hub Sieve (`Q16.16`).
//!
//! Grounded in:
//! - Hansen & Ghrist (2019), "Toward a Spectral Theory of Cellular Sheaves" (arXiv:1808.01513),
//! - Hansen & Ghrist (2021), "Opinion Dynamics on Discourse Sheaves" (arXiv:2005.12798),
//! - Rota (1964), "Theory of Mobius Functions" & Birkhoff (1937), "Rings of Sets",
//! - Forman (2003) & Samal et al. (2018), Augmented Forman-Ricci Curvature `F#(e) = 4 - d_u - d_v + 3*m_e`.
//!
//! 100% deterministic integer (`i64` / `u128` / `Q16.16`) math over pinned WordNet 3.1,
//! ConceptNet 5.7, and Wikidata CC0 edges. Zero floats, zero vectors, zero open-class wordlists.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::graph::taxa::fold_ascii_tokens;
use crate::retrieval::index::Index;
use crate::retrieval::query::tokens::is_content;

pub(in crate::retrieval) const Q16_ONE_I64: i64 = 65_536;

#[inline]
pub(in crate::retrieval) const fn q16_mul(a: i64, b: i64) -> i64 {
    ((a as i128 * b as i128) >> 16) as i64
}

#[inline]
pub(in crate::retrieval) const fn q16_div(num: i64, den: i64) -> i64 {
    if den == 0 {
        0
    } else {
        (((num as i128) << 16) / (den as i128)) as i64
    }
}

/// Closed-class syntactic negation/exclusion scope detector for signed sheaf boundary stalks.
/// hardcode-scan: closed-class (negation, avoidance, and privative function words only).
fn closed_negation_parity(tokens: &[String]) -> Vec<i8> {
    let mut signs = vec![1i8; tokens.len()];
    let mut neg_window = 0usize;
    for (i, w) in tokens.iter().enumerate() {
        if matches!(
            w.as_str(),
            "no" | "not"
                | "never"
                | "without"
                | "avoid"
                | "avoids"
                | "avoiding"
                | "cannot"
                | "cant"
                | "dont"
                | "wont"
                | "shouldnt"
                | "neither"
                | "none"
                | "except"
                | "excluding"
        ) {
            neg_window = 4;
            signs[i] = -1;
            continue;
        }
        // hardcode-scan: closed-class (adversative clause boundary conjunctions)
        if matches!(w.as_str(), "but" | "however" | "although" | "yet") {
            neg_window = 0;
            continue;
        }
        if neg_window > 0 {
            signs[i] = -1;
            neg_window -= 1;
        }
    }
    signs
}

/// Augmented Forman-Ricci curvature `F#(e) = 4 - d_u - d_v + 3 * triangles(u, v)` mapped to
/// a `Q16.16` conductance multiplier in `[8_192, 81_920]` (`0.125 .. 1.25`).
/// Star hubs (`d_u + d_v >> 4` with `triangles == 0`) have strongly negative curvature and are damped,
/// whereas triangle-supported or concise conceptual bridges preserve full conductance.
#[inline]
pub(in crate::retrieval) fn forman_ricci_bridge_weight_q16(
    deg_u: usize,
    deg_v: usize,
    shared_triangles: usize,
) -> i64 {
    let f_sharp: i32 = 4 - (deg_u as i32) - (deg_v as i32) + 3 * (shared_triangles as i32);
    if f_sharp >= 2 {
        81_920 // 1.25x positive-curvature clustered bridge
    } else if f_sharp >= -4 {
        Q16_ONE_I64 // 1.00x neutral tree-like geodesic bridge
    } else if f_sharp >= -12 {
        36_864 // 0.5625x mild hub attenuation
    } else {
        12_288 // 0.1875x star-hub curvature suppression
    }
}

/// Combined per-turn topological deduction result in `Q16.16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::retrieval) struct SheafMobiusScore {
    /// Asymmetric Birkhoff-Heyting entailment `Entail_Q16(Cand |- Query)` with Rota Mobius IC (`0..=65536`).
    pub(in crate::retrieval) heyting_entail_q16: i64,
    /// Intuitionistic open-world negation compatibility (`+Q16` valid disjoint sibling, `-Q16` refuted specialization/superordinate).
    pub(in crate::retrieval) heyting_neg_compat_q16: i64,
    /// Signed Cellular Sheaf Kron-reduced consonance minus cohomological frustration (`i64` Q16.16).
    pub(in crate::retrieval) sheaf_harmony_q16: i64,
}

/// Computes Rota Mobius-deconvolved Heyting residuation (`Cand |- Query` & intuitionistic `~U`)
/// over a finite poset of up to 128 concepts (`parent_mask[i]` = direct hypernyms/containers of `i`).
pub(in crate::retrieval) fn mobius_heyting_entailment_q16(
    parent_adj: &[u128],
    raw_ic_q16: &[i64],
    q_pos_seed: u128,
    q_neg_seed: u128,
    cand_pos_seed: u128,
) -> (i64, i64) {
    let n = parent_adj.len().min(128);
    if n == 0 || (q_pos_seed == 0 && q_neg_seed == 0) || cand_pos_seed == 0 {
        return (0, 0);
    }

    // 1. Transitive upward filter `up[x]` (`z in up[x] <=> x <= z`, i.e., `z` is ancestor of `x`)
    //    and downward order ideal `down[z]` (`x in down[z] <=> x <= z`).
    let mut up = vec![0u128; n];
    for i in 0..n {
        up[i] = (1u128 << i) | parent_adj[i];
    }
    for k in 0..n {
        let up_k = up[k];
        let k_bit = 1u128 << k;
        for i in 0..n {
            if (up[i] & k_bit) != 0 {
                up[i] |= up_k;
            }
        }
    }
    let mut down = vec![0u128; n];
    for x in 0..n {
        let mut m = up[x];
        while m != 0 {
            let z = m.trailing_zeros() as usize;
            if z < n {
                down[z] |= 1u128 << x;
            }
            m &= m - 1;
        }
    }

    // 2. Rota Incidence Algebra Mobius Deconvolution:
    //    Ensure non-negative atomic specificity `delta_mu[z]` so DAG diamond ancestors are never double-counted.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| up[i].count_ones()); // Roots (`up == self`) first, leaves last
    let mut delta_mu = vec![0i64; n];
    for &x in &order {
        let mut anc_sum = 0i64;
        let mut m = up[x] & !(1u128 << x);
        while m != 0 {
            let z = m.trailing_zeros() as usize;
            if z < n {
                anc_sum = anc_sum.saturating_add(delta_mu[z]);
            }
            m &= m - 1;
        }
        // Incremental specificity gained by specializing from `up[x] \ {x}` to `x`:
        let target_ic = raw_ic_q16[x].max(anc_sum + (Q16_ONE_I64 >> 3));
        delta_mu[x] = (target_ic - anc_sum).max(Q16_ONE_I64 >> 4);
    }

    let closure_up = |seed: u128| -> u128 {
        let mut out = 0u128;
        let mut m = seed;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            if i < n {
                out |= up[i];
            }
            m &= m - 1;
        }
        out
    };
    let closure_down = |seed: u128| -> u128 {
        let mut out = 0u128;
        let mut m = seed;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            if i < n {
                out |= down[i];
            }
            m &= m - 1;
        }
        out
    };

    let sum_delta = |mut mask: u128| -> i64 {
        let mut s = 0i64;
        while m_nonzero(mask) {
            let z = mask.trailing_zeros() as usize;
            if z < n {
                s = s.saturating_add(delta_mu[z]);
            }
            mask &= mask - 1;
        }
        s
    };

    // 3. Asymmetric Heyting Entailment `Entail_Q16(Cand |- Query)`:
    //    `Cand` entails `Query` iff `up(Query) subseteq up(Cand)` (`down(Cand) subseteq down(Query)`).
    let up_q = closure_up(q_pos_seed);
    let up_c = closure_up(cand_pos_seed);
    let q_mass = sum_delta(up_q);
    let inter_mass = sum_delta(up_q & up_c);
    let entail_q16 = if q_mass > 0 {
        q16_div(inter_mass, q_mass).clamp(0, Q16_ONE_I64)
    } else {
        0
    };

    // 4. Intuitionistic Pseudocomplement `~U_neg = { x : (down[x] & down(U_neg)) == 0 }`:
    //    Excludes negated target, all specializations (`down`), AND ambiguous superordinates whose `down[x]` hits `U_neg`!
    let neg_compat_q16 = if q_neg_seed != 0 {
        let u_neg_ideal = closure_down(q_neg_seed);
        let mut heyting_neg_mask = 0u128;
        for x in 0..n {
            if (down[x] & u_neg_ideal) == 0 {
                heyting_neg_mask |= 1u128 << x;
            }
        }
        if (cand_pos_seed & u_neg_ideal) != 0 {
            -Q16_ONE_I64 // Direct specialization conflict (`lobster` under `not shellfish`)
        } else if (cand_pos_seed & heyting_neg_mask) == cand_pos_seed {
            Q16_ONE_I64 >> 1 // Certified disjoint in Heyting open-world pseudocomplement
        } else {
            -(Q16_ONE_I64 >> 2) // Ambiguous superordinate (`seafood` under `not shellfish`)
        }
    } else {
        0
    };

    (entail_q16, neg_compat_q16)
}

#[inline]
const fn m_nonzero(m: u128) -> bool {
    m != 0
}

/// Factor-Width-2 Signed Cellular Sheaf Kron Reduction (`L_F = delta^T W delta`, `sigma_e in {-1, +1}`).
/// Eliminates interior bridge stalks `s` (`deg <= 8`) via exact Schur star-mesh reduction and evaluates
/// boundary consonance minus cohomological frustration `E_F(x|_B)`.
pub(in crate::retrieval) fn score_signed_sheaf_kron_q16(
    n_nodes: usize,
    edges: &[(usize, usize, i64, i8)], // (u, v, w_q16, sigma in {-1, +1})
    q_boundary: &[(usize, i64)],       // (u, x_u in {-65536, +65536})
    c_boundary: &[(usize, i64)],       // (v, x_v in {-65536, +65536})
) -> i64 {
    if n_nodes == 0 || q_boundary.is_empty() || c_boundary.is_empty() {
        return 0;
    }
    let mut is_boundary = vec![false; n_nodes];
    let mut x_val = vec![0i64; n_nodes];
    for &(u, xv) in q_boundary {
        if u < n_nodes {
            is_boundary[u] = true;
            x_val[u] = xv;
        }
    }
    for &(v, xv) in c_boundary {
        if v < n_nodes {
            if is_boundary[v] {
                // Direct boundary overlap: concordant sign `+w`, opposite sign `-2w` frustration!
                let prod = q16_mul(x_val[v], xv);
                if prod < 0 {
                    return -Q16_ONE_I64;
                }
            }
            is_boundary[v] = true;
            x_val[v] = xv;
        }
    }

    // Build signed adjacency `adj[u][v] = (net_signed_w, total_abs_w)` for Factor-Width-2 Schur/Kron elimination:
    let mut adj: Vec<BTreeMap<usize, (i64, i64)>> = vec![BTreeMap::new(); n_nodes];
    for &(u, v, w, sig) in edges {
        if u >= n_nodes || v >= n_nodes || u == v || w <= 0 || sig == 0 {
            continue;
        }
        let sw = w * (sig as i64);
        let eu = adj[u].entry(v).or_insert((0, 0));
        eu.0 = eu.0.saturating_add(sw);
        eu.1 = eu.1.saturating_add(w);
        let ev = adj[v].entry(u).or_insert((0, 0));
        ev.0 = ev.0.saturating_add(sw);
        ev.1 = ev.1.saturating_add(w);
    }

    // Eliminate interior bridge nodes `s` (`!is_boundary[s]`, `1 <= deg <= 8`) via exact Kron star-mesh:
    for s in 0..n_nodes {
        if is_boundary[s] || adj[s].is_empty() || adj[s].len() > 8 {
            continue;
        }
        let nbrs: Vec<(usize, (i64, i64))> = std::mem::take(&mut adj[s]).into_iter().collect();
        let mut d_s: i64 = 0;
        for &(_, (_, abs_w)) in &nbrs {
            d_s = d_s.saturating_add(abs_w);
        }
        for &(u, _) in &nbrs {
            adj[u].remove(&s);
        }
        if d_s <= 0 {
            continue;
        }
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                let (u, (sw_us, aw_us)) = nbrs[i];
                let (v, (sw_sv, aw_sv)) = nbrs[j];
                let dw_abs = q16_div(q16_mul(aw_us, aw_sv), d_s);
                if dw_abs <= 0 {
                    continue;
                }
                let sig_uv: i64 = if (sw_us >= 0) == (sw_sv >= 0) { 1 } else { -1 };
                let dw_signed = dw_abs * sig_uv;
                let eu = adj[u].entry(v).or_insert((0, 0));
                eu.0 = eu.0.saturating_add(dw_signed);
                eu.1 = eu.1.saturating_add(dw_abs);
                let ev = adj[v].entry(u).or_insert((0, 0));
                ev.0 = ev.0.saturating_add(dw_signed);
                ev.1 = ev.1.saturating_add(dw_abs);
            }
        }
    }

    // 6-step Gauss-Seidel harmonic relaxation on any remaining high-degree interior stalks:
    for _ in 0..6 {
        for s in 0..n_nodes {
            if is_boundary[s] || adj[s].is_empty() {
                continue;
            }
            let mut num: i64 = 0;
            let mut den: i64 = 0;
            for (&z, &(sw, aw)) in &adj[s] {
                num = num.saturating_add(q16_mul(sw, x_val[z]));
                den = den.saturating_add(aw);
            }
            if den > 0 {
                x_val[s] = q16_div(num, den).clamp(-Q16_ONE_I64, Q16_ONE_I64);
            }
        }
    }

    // Evaluate cross-boundary sheaf consonance minus cohomological frustration:
    let q_set: BTreeSet<usize> = q_boundary.iter().map(|&(u, _)| u).collect();
    let c_set: BTreeSet<usize> = c_boundary.iter().map(|&(v, _)| v).collect();
    let mut consonance: i64 = 0;
    let mut frustration: i64 = 0;
    for &u in &q_set {
        if u >= n_nodes {
            continue;
        }
        for (&v, &(sw, aw)) in &adj[u] {
            if !c_set.contains(&v) {
                continue;
            }
            // Parallel opposite-parity paths have `|sw| < aw`; difference `aw - |sw|` is pure cycle frustration!
            let cycle_frust = aw.saturating_sub(sw.abs());
            let agree = q16_mul(sw, q16_mul(x_val[u], x_val[v]));
            if agree >= 0 {
                consonance = consonance.saturating_add(agree);
            } else {
                frustration = frustration.saturating_add(-agree);
            }
            frustration = frustration.saturating_add(cycle_frust);
        }
    }
    consonance.saturating_sub(frustration + (frustration >> 1))
}

/// Evaluates all candidate turns in `pool` against `q_text` using the unified Curvature-Sieved
/// Signed Sheaf + Mobius-Heyting poset engine over WordNet 3.1, ConceptNet 5.7, and Wikidata CC0.
pub(in crate::retrieval) fn evaluate_sheaf_mobius_pool_q16(
    ix: &Index,
    q_text: &str,
    pool: &[usize],
    exclude_speakers: &BTreeSet<String>,
) -> BTreeMap<usize, SheafMobiusScore> {
    let mut out = BTreeMap::new();
    if pool.is_empty() {
        return out;
    }
    let q_toks = fold_ascii_tokens(q_text);
    let q_signs = closed_negation_parity(&q_toks);

    let mut concept_id: BTreeMap<String, usize> = BTreeMap::new();
    let mut concepts: Vec<String> = Vec::new();
    let intern_c = |s: &str, map: &mut BTreeMap<String, usize>, vec: &mut Vec<String>| -> Option<usize> {
        if s.len() < 3 || !is_content(s) || exclude_speakers.contains(s) {
            return None;
        }
        if let Some(&id) = map.get(s) {
            return Some(id);
        }
        if vec.len() >= 120 {
            return None;
        }
        let id = vec.len();
        map.insert(s.to_string(), id);
        vec.push(s.to_string());
        Some(id)
    };

    let mut q_boundary: Vec<(usize, i64)> = Vec::new();
    let mut q_pos_seed: u128 = 0;
    let mut q_neg_seed: u128 = 0;
    for (idx, w) in q_toks.iter().enumerate() {
        if let Some(cid) = intern_c(w, &mut concept_id, &mut concepts) {
            let sgn = q_signs[idx];
            q_boundary.push((cid, (sgn as i64) * Q16_ONE_I64));
            if sgn > 0 {
                q_pos_seed |= 1u128 << cid;
            } else {
                q_neg_seed |= 1u128 << cid;
            }
        }
    }
    if q_boundary.is_empty() {
        return out;
    }

    // Expand 1..=2 hop taxonomic poset & signed sheaf edges from query seeds + candidate rare tokens:
    let mut turn_seeds: Vec<(usize, Vec<(usize, i64)>, u128)> = Vec::with_capacity(pool.len());
    let nt_u32 = (ix.turns.len() as u32).max(1);
    let max_cand_df = (nt_u32 / 4).clamp(4, 32);
    for &ti in pool.iter().take(48) {
        let tr = &ix.turns[ti];
        let t_toks = fold_ascii_tokens(&tr.body_for_headers);
        let t_signs = closed_negation_parity(&t_toks);
        let mut c_bnd: Vec<(usize, i64)> = Vec::new();
        let mut c_pos: u128 = 0;
        for (i, tw) in t_toks.iter().enumerate() {
            if tw.len() < 3 || !is_content(tw) || exclude_speakers.contains(tw) {
                continue;
            }
            let df = ix
                .vocab
                .get(tw)
                .and_then(|vid| ix.t_own_stats.df.get(&vid).copied())
                .unwrap_or(1);
            if df > max_cand_df && !concept_id.contains_key(tw) {
                continue;
            }
            if let Some(cid) = intern_c(tw, &mut concept_id, &mut concepts) {
                let sgn = t_signs[i];
                if !c_bnd.iter().any(|&(u, _)| u == cid) {
                    c_bnd.push((cid, (sgn as i64) * Q16_ONE_I64));
                }
                if sgn > 0 {
                    c_pos |= 1u128 << cid;
                }
            }
            if c_bnd.len() >= 10 {
                break;
            }
        }
        if !c_bnd.is_empty() {
            turn_seeds.push((ti, c_bnd, c_pos));
        }
    }

    let base_len = concepts.len();
    let mut raw_edges: Vec<(usize, usize, i8)> = Vec::new();
    let mut parent_isa: Vec<(usize, usize)> = Vec::new();
    for u in 0..base_len {
        let w = concepts[u].clone();
        for hyp in crate::wordnet::upward_hypernyms(&w, 2).into_iter().take(5) {
            if let Some(v) = intern_c(hyp, &mut concept_id, &mut concepts)
                && u != v
            {
                parent_isa.push((u, v));
                raw_edges.push((u, v, 1));
            }
        }
        for cont in crate::wikidata::upward_containers(&w, 2).into_iter().take(3) {
            if let Some(v) = intern_c(cont, &mut concept_id, &mut concepts)
                && u != v
            {
                parent_isa.push((u, v));
                raw_edges.push((u, v, 1));
            }
        }
        for ex in crate::wordnet::typed_excludes_targets(&w).into_iter().take(4) {
            if let Some(v) = intern_c(ex, &mut concept_id, &mut concepts)
                && u != v
            {
                raw_edges.push((u, v, -1));
            }
        }
        for (cn, _) in crate::conceptnet::two_hop_neighbors(&w, 8) {
            if let Some(v) = intern_c(cn, &mut concept_id, &mut concepts)
                && u != v
            {
                raw_edges.push((u, v, 1));
            }
        }
    }

    let n_all = concepts.len().min(120);
    let mut nbr_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_all];
    for &(u, v, _) in &raw_edges {
        if u < n_all && v < n_all {
            nbr_sets[u].insert(v);
            nbr_sets[v].insert(u);
        }
    }
    let mut sheaf_edges: Vec<(usize, usize, i64, i8)> = Vec::with_capacity(raw_edges.len());
    for &(u, v, sig) in &raw_edges {
        if u < n_all && v < n_all {
            let tri = nbr_sets[u].intersection(&nbr_sets[v]).count();
            let w_curv = forman_ricci_bridge_weight_q16(nbr_sets[u].len(), nbr_sets[v].len(), tri);
            sheaf_edges.push((u, v, w_curv, sig));
        }
    }

    let mut parent_adj = vec![0u128; n_all];
    for &(u, v) in &parent_isa {
        if u < n_all && v < n_all && u != v {
            parent_adj[u] |= 1u128 << v;
        }
    }
    let mut raw_ic = vec![Q16_ONE_I64; n_all];
    for (i, cname) in concepts.iter().take(n_all).enumerate() {
        let df = ix
            .vocab
            .get(cname)
            .and_then(|vid| ix.t_own_stats.df.get(&vid).copied())
            .unwrap_or(1)
            .max(1);
        let ic = crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16(df, nt_u32) as i64;
        raw_ic[i] = ic.max(Q16_ONE_I64 >> 1);
    }

    for (ti, c_bnd, c_pos) in turn_seeds {
        let (entail_q16, neg_compat_q16) = mobius_heyting_entailment_q16(
            &parent_adj,
            &raw_ic,
            q_pos_seed,
            q_neg_seed,
            c_pos,
        );
        let sheaf_harmony_q16 = score_signed_sheaf_kron_q16(n_all, &sheaf_edges, &q_boundary, &c_bnd);
        if entail_q16 != 0 || neg_compat_q16 != 0 || sheaf_harmony_q16 != 0 {
            out.insert(
                ti,
                SheafMobiusScore {
                    heyting_entail_q16: entail_q16,
                    heyting_neg_compat_q16: neg_compat_q16,
                    sheaf_harmony_q16,
                },
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_sheaf_kron_separates_consonance_from_antonym_frustration() {
        // 0 (query: -1.0 negated root) --(+1)--> 1 (interior bridge) --(+1)--> 2 (leaf)
        let edges = vec![
            (0, 1, 57_344, 1),
            (1, 2, 57_344, 1),
        ];
        let q_bnd = vec![(0, -Q16_ONE_I64)];
        let c_violating = vec![(2, Q16_ONE_I64)];   // Positive assertion of negated leaf -> discord!
        let c_consonant = vec![(2, -Q16_ONE_I64)];  // Avoided leaf -> harmonic global section!
        let sc_bad = score_signed_sheaf_kron_q16(3, &edges, &q_bnd, &c_violating);
        let sc_good = score_signed_sheaf_kron_q16(3, &edges, &q_bnd, &c_consonant);
        assert!(sc_good > 0 && sc_bad < 0, "good={sc_good}, bad={sc_bad}");
    }

    #[test]
    fn mobius_heyting_entailment_is_asymmetric_and_respects_pseudocomplement() {
        // 0 (specific leaf) <= 1 (mid class) <= 2 (broad root); 3 (disjoint sibling) <= 2
        let mut parent = vec![0u128; 4];
        parent[0] = 1 << 1;
        parent[1] = 1 << 2;
        parent[3] = 1 << 2;
        let ic = vec![Q16_ONE_I64 * 3, Q16_ONE_I64 * 2, Q16_ONE_I64, Q16_ONE_I64 * 3];

        // Specialization `0` entails `1` at full 1.0 (`65_536`), whereas broad `2` does NOT fully entail `0`:
        let (spec_to_gen, _) = mobius_heyting_entailment_q16(&parent, &ic, 1 << 1, 0, 1 << 0);
        let (gen_to_spec, _) = mobius_heyting_entailment_q16(&parent, &ic, 1 << 0, 0, 1 << 2);
        assert_eq!(spec_to_gen, Q16_ONE_I64);
        assert!(gen_to_spec < spec_to_gen);

        // Negating `1` (`q_neg_seed = 1 << 1`) refutes specialization `0` (`-65536`) and accepts disjoint `3` (`> 0`):
        let (_, neg_sub) = mobius_heyting_entailment_q16(&parent, &ic, 0, 1 << 1, 1 << 0);
        let (_, neg_disj) = mobius_heyting_entailment_q16(&parent, &ic, 0, 1 << 1, 1 << 3);
        assert!(neg_sub < 0 && neg_disj > 0);
    }
}
