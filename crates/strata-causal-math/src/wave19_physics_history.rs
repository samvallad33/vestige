//! Wave 19 (#739–#743, #746, #751):
//! Z4 discrete path integral destructive interference, Fink time-reversal DORT
//! focusing, Feinberg-Horn-Jackson deficiency-zero audit, Hodge 1-Laplacian,
//! Laplacian renormalization pyramid, Torrens-Pacioli trial-balance ledger,
//! and Inca Khipu base-10 hierarchical episode rollup cords.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

/// #739: Discrete path integral with Z4 edge phases `{0: +1, 1: +i, 2: -1, 3: -i}`.
/// Returns `(real_sum, imag_sum)` across paths; opposite phases (`0` vs `2`) cancel to `(0, 0)`!
pub fn z4_path_integral_sum(path_phases_mod4: &[u8]) -> (i64, i64) {
    let mut re = 0i64;
    let mut im = 0i64;
    for &p in path_phases_mod4 {
        match p % 4 {
            0 => re += 1,
            1 => im += 1,
            2 => re -= 1,
            3 => im -= 1,
            _ => unreachable!(),
        }
    }
    (re, im)
}

/// #740: Time-reversal focusing (Fink) & DORT principal mode on directed cause matrix `G`
/// (`y_symptom = G x_cause` -> dominant eigenvector of `K^T K = G^T G` via power iteration).
pub fn dort_dominant_cause_focus(g_matrix: &[Vec<f64>], iters: usize) -> Vec<f64> {
    let rows = g_matrix.len();
    let cols = g_matrix.first().map_or(0, Vec::len);
    if cols == 0 {
        return Vec::new();
    }
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    for _ in 0..iters {
        // w = G v
        let mut w = vec![0.0; rows];
        for r in 0..rows {
            for c in 0..cols {
                w[r] += g_matrix[r][c] * v[c];
            }
        }
        // next = G^T w
        let mut next = vec![0.0; cols];
        for c in 0..cols {
            for r in 0..rows {
                next[c] += g_matrix[r][c] * w[r];
            }
        }
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut next {
                *x /= norm;
            }
            v = next;
        }
    }
    v
}

/// #741: Feinberg-Horn-Jackson deficiency-zero audit `delta = n - l - s`.
pub fn feinberg_deficiency(
    num_species: usize,
    complexes: &[Vec<f64>],            // `n` complexes, each of length `num_species`
    reactions: &[(usize, usize)],      // directed edges between complex indices `0..n`
) -> usize {
    let n = complexes.len();
    if n == 0 {
        return 0;
    }
    // 1. Linkage classes `l` (connected components of undirected reaction graph)
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in reactions {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut seen = vec![false; n];
    let mut linkage_classes = 0usize;
    for start in 0..n {
        if !seen[start] {
            linkage_classes += 1;
            let mut q = VecDeque::from([start]);
            seen[start] = true;
            while let Some(u) = q.pop_front() {
                for &w in &adj[u] {
                    if !seen[w] {
                        seen[w] = true;
                        q.push_back(w);
                    }
                }
            }
        }
    }

    // 2. Stoichiometric subspace dimension `s = rank({ y_v - y_u })` via Gaussian elimination
    let mut mat: Vec<Vec<f64>> = reactions
        .iter()
        .map(|&(u, v)| {
            (0..num_species)
                .map(|sp| complexes[v][sp] - complexes[u][sp])
                .collect()
        })
        .collect();

    let mut rank = 0usize;
    for col in 0..num_species {
        if let Some(piv) = (rank..mat.len()).find(|&r| mat[r][col].abs() > 1e-9) {
            mat.swap(rank, piv);
            let pv = mat[rank][col];
            for val in &mut mat[rank][col..num_species] {
                *val /= pv;
            }
            for r in 0..mat.len() {
                if r != rank && mat[r][col].abs() > 1e-9 {
                    let factor = mat[r][col];
                    let piv_row = mat[rank][col..num_species].to_vec();
                    for (dst, &pv_elem) in mat[r][col..num_species].iter_mut().zip(piv_row.iter()) {
                        *dst -= factor * pv_elem;
                    }
                }
            }
            rank += 1;
        }
    }

    n.saturating_sub(linkage_classes + rank)
}

/// #742: Hodge 1-Laplacian action `Delta_1 f = (B1^T B1 + B2 B2^T) f` on edge 1-cochains.
pub fn hodge_1_laplacian_apply(
    num_vertices: usize,
    edges: &[(usize, usize)],              // `m` oriented edges `(tail, head)`
    triangles: &[(usize, usize, usize)],   // oriented 2-simplices `(e0, e1, e2)` with `e0 + e1 - e2` boundary
    edge_flow: &[f64],
) -> Vec<f64> {
    let m = edges.len();
    // Gradient/divergence part: B1^T (B1 f)
    let mut div = vec![0.0; num_vertices];
    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        div[v] += edge_flow[e_idx];
        div[u] -= edge_flow[e_idx];
    }
    let mut out = vec![0.0; m];
    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        out[e_idx] += div[v] - div[u];
    }
    // Curl part: B2 (B2^T f)
    for &(e0, e1, e2) in triangles {
        let curl = edge_flow[e0] + edge_flow[e1] - edge_flow[e2];
        out[e0] += curl;
        out[e1] += curl;
        out[e2] -= curl;
    }
    out
}

/// #743: Multi-scale Laplacian renormalization pyramid + synaptic tagging & capture.
pub fn renormalize_haystack_pyramid(
    fine_scores: &[f64],
    synaptic_tags: &BTreeSet<usize>,
    tag_capture_thresh: f64,
) -> (Vec<f64>, Vec<usize>) {
    let mut coarse = Vec::with_capacity(fine_scores.len().div_ceil(2));
    for pair in fine_scores.chunks(2) {
        let mean = pair.iter().sum::<f64>() / (pair.len() as f64);
        coarse.push(mean);
    }
    let captured: Vec<usize> = synaptic_tags
        .iter()
        .copied()
        .filter(|&idx| fine_scores.get(idx).is_some_and(|&s| s >= tag_capture_thresh))
        .collect();
    (coarse, captured)
}

/// #746: Torrens live title register with Pacioli double-entry trial-balance audit.
#[derive(Debug, Default)]
pub struct TorrensPacioliRegister {
    live_titles: BTreeMap<String, String>, // parcel_slot -> active_deed_mem_id
    debits_cents: i64,
    credits_cents: i64,
}

impl TorrensPacioliRegister {
    pub fn transfer_title(&mut self, slot: &str, new_deed_id: &str) -> Option<String> {
        // Double-entry weight conservation: +1 live title debit, +1 retired/minted credit
        self.debits_cents += 100;
        self.credits_cents += 100;
        self.live_titles.insert(slot.to_string(), new_deed_id.to_string())
    }

    pub fn current_deed(&self, slot: &str) -> Option<&str> {
        self.live_titles.get(slot).map(String::as_str)
    }

    pub fn trial_balance_ok(&self) -> bool {
        self.debits_cents == self.credits_cents
    }
}

/// #751: Inca Khipu base-10 positional knot cords + top-cord subsidiary rollup audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KhipuCluster {
    pub pendant_counts: Vec<u64>,
    pub top_cord_rollup: u64,
    pub base10_knots: Vec<Vec<u8>>, // hundreds..tens..units digits per pendant
}

pub fn encode_khipu_rollup(pendant_counts: Vec<u64>) -> KhipuCluster {
    let top_cord_rollup: u64 = pendant_counts.iter().sum();
    let base10_knots = pendant_counts
        .iter()
        .map(|&n| {
            n.to_string()
                .bytes()
                .map(|b| b - b'0')
                .collect::<Vec<u8>>()
        })
        .collect();
    KhipuCluster {
        pendant_counts,
        top_cord_rollup,
        base10_knots,
    }
}

pub fn verify_khipu_audit(cluster: &KhipuCluster) -> bool {
    let sum: u64 = cluster.pendant_counts.iter().sum();
    sum == cluster.top_cord_rollup
}
