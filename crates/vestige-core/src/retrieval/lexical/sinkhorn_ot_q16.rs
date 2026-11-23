//! Discrete Sinkhorn-Knopp Optimal Transport over Pinned Lexical/Graph Ground Metric (Q16.16).
//!
//! Inspired by Sinkhorn Associative Memory / Optimal Transport (`2603.20656`, Mar 2026) and
//! GeoRAG multi-demand coverage (`2606.29328`, Jun 2026).
//! Runs 4 pure-integer Q16 Sinkhorn scaling iterations over the small `(m x n)` discrete kernel
//! `K_{ab}` linking query content tokens `q_a` to candidate turn tokens `d_b` via exact match,
//! WordNet Heyting implication, and shared WordNet ancestor synsets — rewarding turns that
//! holistically transport ALL query demand instead of repeating one surface token.

use crate::retrieval::{Index, is_content, raw_tokens};
use std::collections::BTreeSet;

const Q16_ONE: u64 = 65_536;
const SINKHORN_ITERS: usize = 4;
const KERNEL_EPS: u64 = 128; // ~0.002 in Q16 to guarantee strictly positive scaling

/// Precomputed query token descriptors for fast Top-K candidate OT re-scoring.
pub(in crate::retrieval) struct SinkhornQuery {
    pub(in crate::retrieval) words: Vec<String>,
    /// Normalized source marginal `p_a` in Q16 summing to `Q16_ONE`.
    pub(in crate::retrieval) marginal_q16: Vec<u64>,
    /// Specific ancestor synset sets per query token (cached once per query).
    pub(in crate::retrieval) synsets: Vec<BTreeSet<u32>>,
}

impl SinkhornQuery {
    pub(in crate::retrieval) fn build(
        ix: &Index,
        q_text: &str,
        exclude_speakers: &BTreeSet<String>,
    ) -> Self {
        let mut seen = BTreeSet::new();
        let mut words: Vec<String> = Vec::new();
        for tok in raw_tokens(q_text) {
            if tok.len() >= 3
                && is_content(&tok)
                && !exclude_speakers.contains(&tok)
                && seen.insert(tok.clone())
            {
                words.push(tok);
                if words.len() >= 8 {
                    break;
                }
            }
        }
        let m = words.len();
        if m == 0 {
            return Self {
                words,
                marginal_q16: Vec::new(),
                synsets: Vec::new(),
            };
        }
        let nt = (ix.turns.len() as u64).max(4);
        let mut raw_w = Vec::with_capacity(m);
        let mut sum_w = 0u64;
        let mut synsets = Vec::with_capacity(m);
        for w in &words {
            let df = ix
                .vocab
                .get(w)
                .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                .unwrap_or(1) as u64;
            let ratio = nt / df.max(1);
            let ic = ((64 - ratio.leading_zeros()) as u64).max(1) * Q16_ONE;
            raw_w.push(ic);
            sum_w = sum_w.saturating_add(ic);

            let mut ss = BTreeSet::new();
            for &s in crate::wordnet::senses(w).iter().take(3) {
                ss.insert(s);
            }
            for (anc, _) in crate::wordnet::specific_ancestor_synsets(w, 2, 120)
                .into_iter()
                .take(4)
            {
                ss.insert(anc);
            }
            synsets.push(ss);
        }
        let marginal_q16 = raw_w
            .into_iter()
            .map(|w| ((w as u128 * Q16_ONE as u128) / sum_w.max(1) as u128).max(1) as u64)
            .collect();
        Self {
            words,
            marginal_q16,
            synsets,
        }
    }

    /// Compute Q16 discrete Sinkhorn Optimal Transport alignment score `in [0, Q16_ONE]`
    /// between `self` and turn `ti`.
    pub(in crate::retrieval) fn score_turn_q16(&self, ix: &Index, ti: usize) -> u64 {
        let m = self.words.len();
        if m == 0 || ti >= ix.turns.len() {
            return 0;
        }
        let tr = &ix.turns[ti];
        let mut d_words: Vec<String> = Vec::new();
        let mut d_seen = BTreeSet::new();
        for tok in raw_tokens(&tr.body_for_headers) {
            if tok.len() >= 3 && is_content(&tok) && d_seen.insert(tok.clone()) {
                d_words.push(tok);
                if d_words.len() >= 20 {
                    break;
                }
            }
        }
        let n = d_words.len();
        if n == 0 {
            return 0;
        }

        // Precompute turn token synsets for fast O(m*n) kernel evaluation.
        let mut d_syns: Vec<BTreeSet<u32>> = Vec::with_capacity(n);
        for dw in &d_words {
            let mut ss = BTreeSet::new();
            for &s in crate::wordnet::senses(dw).iter().take(3) {
                ss.insert(s);
            }
            for (anc, _) in crate::wordnet::specific_ancestor_synsets(dw, 2, 120)
                .into_iter()
                .take(4)
            {
                ss.insert(anc);
            }
            d_syns.push(ss);
        }

        // Build discrete Gibbs kernel K[a][b] in Q16 (`exp(-C_{ab} / eps)` proxy).
        let mut k_mat = vec![KERNEL_EPS; m * n];
        let mut any_non_eps = false;
        for a in 0..m {
            let qw = &self.words[a];
            let qs = &self.synsets[a];
            for b in 0..n {
                let dw = &d_words[b];
                let sim: u64 = if qw == dw {
                    Q16_ONE
                } else if qw.len() >= 4
                    && dw.len() >= 4
                    && (qw.starts_with(dw.as_str()) || dw.starts_with(qw.as_str()))
                {
                    (Q16_ONE * 82) / 100
                } else {
                    let heyt = crate::wordnet::heyting_implication_ic_q16(dw, qw);
                    if heyt >= 68 {
                        (Q16_ONE * u64::from(heyt)) / 100
                    } else if !qs.is_empty() && !d_syns[b].is_empty() && qs.iter().any(|s| d_syns[b].contains(s)) {
                        (Q16_ONE * 58) / 100
                    } else {
                        KERNEL_EPS
                    }
                };
                if sim > KERNEL_EPS {
                    any_non_eps = true;
                }
                k_mat[a * n + b] = sim;
            }
        }
        if !any_non_eps {
            return 0;
        }

        // Uniform target marginal `q_b = Q16_ONE / n` (encourages balanced multi-slot alignment).
        let q_b = (Q16_ONE / (n as u64)).max(1);
        let mut u = vec![Q16_ONE; m];
        let mut v = vec![Q16_ONE; n];

        for _ in 0..SINKHORN_ITERS {
            // u_a = p_a / sum_b K_{ab} v_b
            for a in 0..m {
                let mut denom: u128 = 0;
                let row = &k_mat[a * n..(a + 1) * n];
                for b in 0..n {
                    denom = denom.saturating_add((row[b] as u128 * v[b] as u128) >> 16);
                }
                let d_clamped = denom.max(1);
                u[a] = (((self.marginal_q16[a] as u128) << 16) / d_clamped)
                    .min((Q16_ONE as u128) * 64) as u64;
            }
            // v_b = q_b / sum_a K_{ab} u_a
            for b in 0..n {
                let mut denom: u128 = 0;
                for a in 0..m {
                    denom = denom.saturating_add((k_mat[a * n + b] as u128 * u[a] as u128) >> 16);
                }
                let d_clamped = denom.max(1);
                v[b] = (((q_b as u128) << 16) / d_clamped)
                    .min((Q16_ONE as u128) * 64) as u64;
            }
        }

        // Transported similarity `<P*, K> = sum_{a,b} (u_a K_{ab} v_b) * K_{ab}` in Q16.
        let mut total: u128 = 0;
        for a in 0..m {
            let row = &k_mat[a * n..(a + 1) * n];
            for b in 0..n {
                let kab = row[b] as u128;
                if kab <= KERNEL_EPS as u128 {
                    continue;
                }
                let p_ab = (((u[a] as u128 * kab) >> 16) * (v[b] as u128)) >> 16;
                total = total.saturating_add((p_ab * kab) >> 16);
            }
        }
        total.min(Q16_ONE as u128) as u64
    }
}
