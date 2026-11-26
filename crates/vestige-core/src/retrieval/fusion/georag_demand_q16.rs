//! Kernel 1: GeoRAG Demand Coverage (`georag_demand_q16.rs`, arXiv:2606.29328).
//!
//! Splits multi-part queries into weighted demand facets `{q_j}` using corpus-IDF
//! and Reverse-Validation uniqueness weights `q_tilde_j` (rewarding facets covered by
//! few candidate turns/sessions and damping generic recap cliches), then greedily seats
//! Top-k candidates maximizing marginal demand gain weighted by `(Q16_ONE - max Jaccard)`:
//!   `Delta F_Q(c | S) = sum_j q_tilde_j * max(0, cov(c, q_j) - M_j(S)) * (65536 - max_s Jaccard_Q16(c, s)) >> 16`.
//!
//! Strictly preserves Slot 0 (Rank-1 crisp anchor) when called on Slots `1..k`.

use crate::retrieval::fusion::kemeny_mrf::log2_u64_q16;
use crate::retrieval::{DetHashMap, DetHashSet, Index, QueryPlan};

const Q16_ONE: u64 = 65_536;

/// Candidate turn descriptor for GeoRAG submodular demand coverage seating.
#[derive(Clone)]
pub(in crate::retrieval) struct DemandCand {
    pub(in crate::retrieval) turn_idx: usize,
    pub(in crate::retrieval) base_q16: u64,
    pub(in crate::retrieval) facet_cov_q16: Vec<u64>,
    pub(in crate::retrieval) rare_terms: Vec<u32>,
}

/// Computes `Jaccard_Q16(a, b)` over sorted deduplicated `rare_terms` sets.
#[inline]
fn jaccard_rare_q16(a: &[u32], b: &[u32]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut inter = 0u64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    if inter == 0 {
        return 0;
    }
    let union = (a.len() + b.len()) as u64 - inter;
    ((inter as u128 * Q16_ONE as u128) / union.max(1) as u128).min(Q16_ONE as u128) as u64
}

/// Rescores/reorders candidate turns `[0..pool_len]` via GeoRAG reverse-validation
/// demand coverage, keeping index `0` anchored when `protect_top0` is true so crisp
/// `@1` lexical winners are never displaced while stopping redundant recaps from
/// filling Slots `1..k`.
pub(in crate::retrieval) fn georag_seat_turns_q16(
    ix: &Index,
    plan: &QueryPlan,
    cand_turns: &[(usize, u64)],
    k: usize,
    protect_top0: bool,
) -> Vec<usize> {
    let n = cand_turns.len().min(32);
    if n <= 2 || k <= 1 || plan.groups.is_empty() {
        return cand_turns.iter().take(n).map(|&(ti, _)| ti).collect();
    }
    let facets = plan.groups.len().min(16);
    let nt = (ix.turns.len() as u64).max(2);

    // Group term ids per facet index [0..facets)
    let mut facet_terms: Vec<Vec<u32>> = vec![Vec::new(); facets];
    let mut facet_df: Vec<u32> = vec![nt as u32; facets];
    for qt in &plan.terms {
        if let Some(g) = qt.group
            && g < facets
        {
            facet_terms[g].push(qt.id);
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(nt as u32);
            facet_df[g] = facet_df[g].min(df.max(1));
        }
    }

    let rare_cap = (ix.turns.len() as u32 / 8).clamp(6, 48);
    let mut cands: Vec<DemandCand> = Vec::with_capacity(n);
    let mut facet_pool_freq = vec![0u64; facets];

    for &(ti, base_q16) in cand_turns.iter().take(n) {
        let tr = &ix.turns[ti];
        let mut cov = vec![0u64; facets];
        for (fi, tids) in facet_terms.iter().enumerate() {
            if tids.is_empty() {
                continue;
            }
            let own_hit = tids.iter().any(|id| tr.content.contains(id));
            let feat_hit = tr.feats.iter().any(|f| tids.contains(&f.id));
            let val = if own_hit && !tr.is_question && !tr.is_backchannel {
                Q16_ONE
            } else if own_hit {
                (Q16_ONE * 3) >> 2
            } else if feat_hit {
                Q16_ONE >> 1
            } else {
                0
            };
            cov[fi] = val;
            if val > 0 {
                facet_pool_freq[fi] += 1;
            }
        }
        let mut rare: Vec<u32> = tr
            .content
            .iter()
            .copied()
            .filter(|cid| {
                let df = ix.t_own_stats.df.get(cid).copied().unwrap_or(0);
                df >= 1 && df <= rare_cap
            })
            .collect();
        rare.sort_unstable();
        rare.dedup();
        rare.truncate(24);

        cands.push(DemandCand {
            turn_idx: ti,
            base_q16,
            facet_cov_q16: cov,
            rare_terms: rare,
        });
    }

    // Reverse-Validation uniqueness weight q_tilde_j in Q16:
    // IC(q_j) / (1 + pool_freq_j / 2), strongly uplifting scarce query demands.
    let mut q_tilde_q16 = vec![Q16_ONE; facets];
    for fi in 0..facets {
        let ratio = (nt + 1) / (facet_df[fi] as u64 + 1).max(1);
        let ic = log2_u64_q16(ratio.max(2));
        let denom_q16 = Q16_ONE.saturating_add(facet_pool_freq[fi] * (Q16_ONE >> 1));
        q_tilde_q16[fi] = ((ic as u128 * Q16_ONE as u128) / denom_q16.max(1) as u128)
            .clamp(Q16_ONE as u128 >> 2, (Q16_ONE as u128) * 8) as u64;
    }

    let mut used = vec![false; n];
    let mut max_covered = vec![0u64; facets];
    let mut selected_indices: Vec<usize> = Vec::with_capacity(k.min(n));
    let mut sess_counts: DetHashMap<usize, usize> = DetHashMap::default();
    let mut out = Vec::with_capacity(n);

    if protect_top0 {
        used[0] = true;
        selected_indices.push(0);
        for fi in 0..facets {
            max_covered[fi] = max_covered[fi].max(cands[0].facet_cov_q16[fi]);
        }
        *sess_counts.entry(ix.turns[cands[0].turn_idx].session).or_default() += 1;
        out.push(cands[0].turn_idx);
    }

    let max_base = cands.iter().map(|c| c.base_q16).max().unwrap_or(1).max(1);
    let target_slots = k.min(n);

    while out.len() < target_slots {
        let mut best_pos: Option<usize> = None;
        let mut best_obj: u128 = 0;

        for (i, c) in cands.iter().enumerate() {
            if used[i] {
                continue;
            }
            let mut max_jac = 0u64;
            for &si_pos in &selected_indices {
                let jac = jaccard_rare_q16(&c.rare_terms, &cands[si_pos].rare_terms);
                if jac > max_jac {
                    max_jac = jac;
                }
            }
            let novelty_q16 = Q16_ONE.saturating_sub(max_jac);

            let mut demand_gain: u128 = 0;
            for fi in 0..facets {
                let delta = c.facet_cov_q16[fi].saturating_sub(max_covered[fi]);
                if delta > 0 {
                    demand_gain += (q_tilde_q16[fi] as u128 * delta as u128) >> 16;
                }
            }
            let weighted_gain = (demand_gain * novelty_q16 as u128) >> 16;
            let rel_norm_q16 = ((c.base_q16 as u128) << 16) / (max_base as u128);
            let same_sess = sess_counts
                .get(&ix.turns[c.turn_idx].session)
                .copied()
                .unwrap_or(0);
            let sess_damp = if same_sess >= 2 { 3 } else if same_sess == 1 { 5 } else { 8 };
            // Combine GeoRAG submodular demand coverage (70%) + base relevance (30%), damped on recap repetition.
            let obj = (((weighted_gain * 7 + (rel_norm_q16 * 3)) * sess_damp) / 80) + 1;

            let is_better = match best_pos {
                None => true,
                Some(bp) => obj > best_obj || (obj == best_obj && c.turn_idx < cands[bp].turn_idx),
            };
            if is_better {
                best_obj = obj;
                best_pos = Some(i);
            }
        }

        let Some(pos) = best_pos else {
            break;
        };
        used[pos] = true;
        selected_indices.push(pos);
        for fi in 0..facets {
            max_covered[fi] = max_covered[fi].max(cands[pos].facet_cov_q16[fi]);
        }
        *sess_counts.entry(ix.turns[cands[pos].turn_idx].session).or_default() += 1;
        out.push(cands[pos].turn_idx);
    }

    let mut seen_ti: DetHashSet<usize> = out.iter().copied().collect();
    for c in &cands {
        if seen_ti.insert(c.turn_idx) {
            out.push(c.turn_idx);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Query, build_index, plan_query};
    use crate::retrieval::tests::sess;

    #[test]
    fn georag_stops_redundant_recap_turns_and_covers_missing_facet() {
        let sessions = vec![
            sess("s0", &[("alice", "I adopted a golden retriever named Cooper in Seattle.")]),
            sess("s1", &[("assistant", "Recap: Cooper is a golden retriever adopted in Seattle!")]),
            sess("s2", &[("alice", "Cooper passed therapy dog certification at Harbor Clinic.")]),
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query { text: "Where did Cooper pass therapy certification in Seattle?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &p);
        let raw = vec![(0, 100_000), (1, 98_000), (2, 75_000)];
        let seated = georag_seat_turns_q16(&ix, &plan, &raw, 3, true);
        assert_eq!(seated[0], 0, "Slot 0 protected: {seated:?}");
        assert_eq!(seated[1], 2, "Missing certification facet beat duplicate recap: {seated:?}");
    }
}
