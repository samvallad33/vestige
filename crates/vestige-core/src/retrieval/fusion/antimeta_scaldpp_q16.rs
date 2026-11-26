//! Kernel 4: Anti-Meta Q_ii + Echo Damper + GraphMemix MST + ScalDPP (`antimeta_scaldpp_q16.rs`).
//!
//! References:
//! - ScalDPP (arXiv:2604.03240): Incremental Cholesky DPP over sparse rare-token Gram matrix.
//! - GraphMemix (arXiv:2608.26983): Kruskal MST chronological step linkage over novel events.
//! - CoD (arXiv:2611.02480) & ContextFact (arXiv:2611.29155): Content-IDF density & meta dampening.
//!
//! Computes Anti-Meta Q_ii turn quality (`IDF_Density * (65536 - gamma * MetaRatio)`),
//! damps assistant echoes against preceding user turn `u_{t-1}`, and diversifies
//! Top-k seats via sparse TF-IDF Cholesky DPP + Kruskal MST novelty chaining.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::{DetHashSet, Index, is_content, raw_tokens};

const Q16_ONE: u64 = 65_536;
/// Gamma = 0.75 in Q16.16 for MetaRatio quality dampening.
const META_GAMMA_Q16: u64 = 49_152;

// hardcode-scan: closed-class (conversational recap/meta markers and hedges only, zero topic words)
const META_MARKERS: &[&str] = &[
    "recap", "summary", "summarize", "overview", "timeline", "noted", "sounds", "great",
    "awesome", "helpful", "clarify", "agree", "agreed", "definitely", "certainly", "glad",
    "happy", "thanks", "thank", "welcome", "sure", "okay", "understands", "understood",
];

/// Returns turn quality multiplier `in [16384, 78643]` (`[0.25, 1.20]` in Q16.16) combining:
/// - Corpus content-IDF density `IDF_Density_Q16`
/// - Closed-class meta/recap dampening `(65536 - gamma * MetaRatio)`
/// - Derivative assistant echo penalty against immediately preceding user turn `u_{t-1}`.
pub(in crate::retrieval) fn antimeta_echo_quality_q16(ix: &Index, ti: usize) -> u64 {
    if ti >= ix.turns.len() {
        return Q16_ONE;
    }
    let tr = &ix.turns[ti];
    if tr.is_standing_policy || tr.is_habitual {
        return Q16_ONE;
    }
    let toks = raw_tokens(&tr.body_for_headers);
    if toks.len() < 4 {
        return Q16_ONE;
    }
    let nt = (ix.turns.len() as u64).max(4);
    let mut meta_hits = 0u64;
    let mut ic_sum = 0u64;
    let mut content_cnt = 0u64;
    for tok in &toks {
        if META_MARKERS.contains(&tok.as_str()) {
            meta_hits += 1;
        }
        if is_content(tok) {
            content_cnt += 1;
            let df = ix
                .vocab
                .get(tok)
                .and_then(|id| ix.t_own_stats.df.get(&id).copied())
                .unwrap_or(1)
                .max(1) as u64;
            ic_sum = ic_sum.saturating_add(log2_u64_q16((nt + 1) / df));
        }
    }

    let total = (toks.len() as u64).max(1);
    let meta_ratio_q16 = ((meta_hits as u128 * Q16_ONE as u128) / total as u128)
        .min(Q16_ONE as u128) as u64;
    let avg_ic = if content_cnt > 0 { ic_sum / content_cnt } else { Q16_ONE };
    // Only damp turns carrying conversational meta/recap markers or assistant echoes;
    // never distort clean declarative turns upward or downward solely by corpus average IDF.
    let mut q_ii = if meta_hits > 0 {
        let idf_density_q16 = ((avg_ic * 3) >> 4).clamp(49_152, Q16_ONE);
        let meta_factor_q16 =
            Q16_ONE.saturating_sub(((META_GAMMA_Q16 as u128 * meta_ratio_q16 as u128) >> 16) as u64);
        ((idf_density_q16 as u128 * meta_factor_q16 as u128) >> 16) as u64
    } else {
        Q16_ONE
    };

    // Echo Damper: if assistant turn copies >= 70% of preceding user turn `u_{t-1}` rare content
    // and adds <= 1 new rare token, damp to 0.35x so original user evidence wins.
    if tr.field == 1 && ti > 0 && ix.turns[ti - 1].session == tr.session && ix.turns[ti - 1].field == 0 {
        let prev = &ix.turns[ti - 1];
        if !prev.content.is_empty() && !tr.content.is_empty() {
            let shared = tr.content.iter().filter(|id| prev.content.contains(id)).count();
            let novel = tr.content.iter().filter(|id| !prev.content.contains(id)).count();
            if shared * 10 >= prev.content.len() * 7 && novel <= 1 {
                q_ii = ((q_ii as u128 * 22_938u128) >> 16) as u64; // 0.35x
            }
        }
    }
    q_ii.clamp(16_384, 78_643)
}

/// Computes Kruskal MST bonus edges across chronologically ordered turns + incremental
/// orthogonalized Cholesky ScalDPP diversity selection (`<= 24` candidates -> `k` turns).
/// Always keeps index `0` anchored when `protect_top0` is true.
pub(in crate::retrieval) fn scaldpp_mst_select_q16(
    ix: &Index,
    cand_turns: &[(usize, u64)],
    k: usize,
    protect_top0: bool,
) -> Vec<usize> {
    let n = cand_turns.len().min(24);
    if n <= 2 || k <= 1 {
        return cand_turns.iter().take(n).map(|&(ti, _)| ti).collect();
    }
    let rare_cap = (ix.turns.len() as u32 / 8).clamp(4, 40);
    let nt = (ix.turns.len() as u64).max(2);

    // Sparse weighted rare TF vectors per candidate (normalized to sum-of-squares <= Q16_ONE)
    let mut vecs: Vec<Vec<(u32, u64)>> = Vec::with_capacity(n);
    let mut diag = vec![0u64; n];

    for (idx, &(ti, base_q16)) in cand_turns.iter().take(n).enumerate() {
        let tr = &ix.turns[ti];
        let q_mult = antimeta_echo_quality_q16(ix, ti);
        let mut weights: Vec<(u32, u64)> = tr
            .content
            .iter()
            .copied()
            .filter_map(|cid| {
                let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0) as u64;
                if df >= 1 && df <= rare_cap as u64 {
                    let w = log2_u64_q16((nt + 1) / df.max(1)).max(Q16_ONE);
                    Some((cid, w))
                } else {
                    None
                }
            })
            .collect();
        weights.sort_unstable_by_key(|&(id, _)| id);
        weights.truncate(20);
        let norm_sq: u128 = weights.iter().map(|&(_, w)| (w as u128 * w as u128) >> 16).sum::<u128>().max(1);
        let max_w_scale = Q16_ONE; // normalized similarity contribution
        for entry in &mut weights {
            entry.1 = (((entry.1 as u128 * entry.1 as u128) >> 16) * max_w_scale as u128 / norm_sq) as u64;
        }
        // Quality-scaled diagonal L_ii in Q16
        let q_sq = (q_mult as u128 * base_q16.max(1) as u128) >> 16;
        diag[idx] = q_sq.max(1) as u64;
        vecs.push(weights);
    }

    // GraphMemix Kruskal MST bonus over distinct chronological novel pairs:
    // Reward nodes that introduce new chronological anchors (distinct sessions / timestamps).
    let mut mst_novel_bonus = vec![0u64; n];
    let mut chrono_order: Vec<usize> = (0..n).collect();
    chrono_order.sort_unstable_by_key(|&i| {
        let ti = cand_turns[i].0;
        (ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0), ti)
    });
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut edges: Vec<(u64, usize, usize)> = Vec::new();
    for pos in 0..chrono_order.len().saturating_sub(1) {
        let a = chrono_order[pos];
        let b = chrono_order[pos + 1];
        if ix.turns[cand_turns[a].0].session != ix.turns[cand_turns[b].0].session {
            // Dot overlap over rare terms
            let overlap = sparse_dot_q16(&vecs[a], &vecs[b]);
            let link_score = (Q16_ONE >> 2).saturating_add(overlap >> 1);
            edges.push((link_score, a, b));
        }
    }
    edges.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (w, u, v) in edges {
        let ru = find(&mut parent, u);
        let rv = find(&mut parent, v);
        if ru != rv {
            parent[ru] = rv;
            mst_novel_bonus[u] = mst_novel_bonus[u].saturating_add(w >> 3);
            mst_novel_bonus[v] = mst_novel_bonus[v].saturating_add(w >> 3);
        }
    }

    // Incremental Cholesky ScalDPP greedy selection: d_i^2 <- d_i^2 - c_{i,s}^2
    let mut resid_diag_q16: Vec<u64> = (0..n)
        .map(|i| diag[i].saturating_add(mst_novel_bonus[i]))
        .collect();
    let max_d = resid_diag_q16.iter().copied().max().unwrap_or(1).max(1);
    for d in &mut resid_diag_q16 {
        *d = ((*d as u128 * Q16_ONE as u128) / max_d as u128).max(1) as u64;
    }

    let mut used = vec![false; n];
    let mut c_cols: Vec<Vec<i64>> = Vec::with_capacity(k.min(n));
    let mut out = Vec::with_capacity(n);

    let seed = if protect_top0 { 0 } else { (0..n).max_by_key(|&i| resid_diag_q16[i]).unwrap_or(0) };
    let mut last_picked = seed;
    used[seed] = true;
    out.push(cand_turns[seed].0);

    while out.len() < k.min(n) {
        let d_last = resid_diag_q16[last_picked].max(1);
        let mut col = vec![0i64; n];
        for i in 0..n {
            if used[i] {
                continue;
            }
            let dot = sparse_dot_q16(&vecs[i], &vecs[last_picked]) as i64;
            let mut proj_sum: i128 = 0;
            for c_prev in &c_cols {
                proj_sum += (c_prev[i] as i128 * c_prev[last_picked] as i128) >> 16;
            }
            let numer = (dot as i128).saturating_sub(proj_sum);
            // c_i = numer / sqrt(d_last) approximated in Q16 without floats
            let c_val = ((numer * Q16_ONE as i128) / (d_last.max(256) as i128)).clamp(-(Q16_ONE as i128), Q16_ONE as i128) as i64;
            col[i] = c_val;
            let sq = ((c_val as i128 * c_val as i128) >> 16).unsigned_abs() as u64;
            resid_diag_q16[i] = resid_diag_q16[i].saturating_sub(sq);
        }
        c_cols.push(col);

        let next = (0..n)
            .filter(|&i| !used[i])
            .max_by_key(|&i| resid_diag_q16[i]);
        let Some(ni) = next else { break };
        used[ni] = true;
        last_picked = ni;
        out.push(cand_turns[ni].0);
    }

    let mut seen: DetHashSet<usize> = out.iter().copied().collect();
    for &(ti, _) in cand_turns.iter().take(n) {
        if seen.insert(ti) {
            out.push(ti);
        }
    }
    out
}

#[inline]
fn sparse_dot_q16(a: &[(u32, u64)], b: &[(u32, u64)]) -> u64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut sum = 0u64;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                // Both already normalized density shares -> geometric overlap bounded by min share
                sum = sum.saturating_add(a[i].1.min(b[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    sum.min(Q16_ONE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, build_index};
    use crate::retrieval::tests::sess;

    #[test]
    fn antimeta_damps_recap_and_echo_while_preserving_original_fact() {
        let sessions = vec![
            sess("s0", &[
                ("user", "My telemetry beacon uses port 9443 on staging."),
                ("assistant", "Telemetry beacon uses port 9443 on staging."),
            ]),
            sess("s1", &[
                ("assistant", "Summary recap overview: noted, sounds great, clarify welcome thanks!"),
            ]),
        ];
        let ix = build_index(&sessions, &Params::default());
        let q_orig = antimeta_echo_quality_q16(&ix, 0);
        let q_echo = antimeta_echo_quality_q16(&ix, 1);
        let q_meta = antimeta_echo_quality_q16(&ix, 2);
        assert!(q_orig > q_echo, "origin={q_orig} must beat parrot echo={q_echo}");
        assert!(q_orig > q_meta, "origin={q_orig} must beat meta recap={q_meta}");
    }
}
