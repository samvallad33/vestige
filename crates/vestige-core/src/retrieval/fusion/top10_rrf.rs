//! Slot-0-Locked Top-10 5-Ballot Reciprocal Rank Fusion (`Q16.16`, Cormack et al. SIGIR 2009).
//!
//! Stage 12's 5-voter Kemeny fusion only inspects `SESSION_FUSION_TOP_K = 5`, leaving
//! candidate sessions sitting at ranks `5..9` unranked by multi-voter consensus.
//! This module fuses 5 zero-float `Q16.16` ballots across Top-10 (`k_rrf = 60`):
//! 1. Session sight (`s_score + bridge_boost`),
//! 2. Metzler-Croft MRF / PLM / MinCover span rank,
//! 3. Turn-evidence seat & peak scene rank,
//! 4. Dialogue query-cue coverage rank,
//! 5. Langlois-Seignobos firsthand witness corroboration rank,
//! while **strictly locking `final_session_indices[0]`** so Single-Hop & Adversarial R@1
//! cannot drop by a single query.

use crate::retrieval::fusion::kemeny_mrf::pool_mrf_and_exact_ic_q16;
use crate::retrieval::graph::historiography_source::session_firsthand_witness_scores_q16;
use crate::retrieval::index::Index;
use crate::retrieval::seating::{query_cues, turn_query_cues};
use crate::retrieval::QueryPlan;

const Q16_ONE: u64 = 65_536;
const TOP10_POOL_CAP: usize = 10;
const MIN_POOL_LEN: usize = 6;
const RRF_K: u64 = 60;
const W_SIGHT: u64 = 3;
const W_MRF: u64 = 2;
const W_TURN: u64 = 2;
const W_CUE: u64 = 1;
const W_WITNESS: u64 = 1;

pub(in crate::retrieval) struct Top10RrfInputs<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) q_text: &'a str,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) final_turn_indices: &'a [usize],
    pub(in crate::retrieval) best_scene_q16: &'a [u64],
    pub(in crate::retrieval) sight_all_q16: &'a [u64],
    pub(in crate::retrieval) s_exact_mask: &'a [u64],
}

pub(in crate::retrieval) fn rrf_reorder_tail_5ballot_q16(
    cands: &[usize],
    sight_rank: &[usize],
    mrf_rank: &[usize],
    turn_rank: &[usize],
    cue_rank: &[usize],
    witness_rank: &[usize],
) -> Vec<usize> {
    let n = cands.len();
    if n < 3 {
        return cands.to_vec();
    }
    let mut rrf_q16 = vec![0u64; n];
    for i in 1..n {
        let s = (W_SIGHT.saturating_mul(Q16_ONE)) / (RRF_K + sight_rank[i] as u64);
        let m = (W_MRF.saturating_mul(Q16_ONE)) / (RRF_K + mrf_rank[i] as u64);
        let t = (W_TURN.saturating_mul(Q16_ONE)) / (RRF_K + turn_rank[i] as u64);
        let c = (W_CUE.saturating_mul(Q16_ONE)) / (RRF_K + cue_rank[i] as u64);
        let w = (W_WITNESS.saturating_mul(Q16_ONE)) / (RRF_K + witness_rank[i] as u64);
        rrf_q16[i] = s
            .saturating_add(m)
            .saturating_add(t)
            .saturating_add(c)
            .saturating_add(w);
    }
    let mut tail: Vec<usize> = (1..n).collect();
    tail.sort_by(|&a, &b| {
        rrf_q16[b]
            .cmp(&rrf_q16[a])
            .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
            .then_with(|| cands[a].cmp(&cands[b]))
    });
    let mut out = Vec::with_capacity(n);
    out.push(cands[0]);
    for idx in tail {
        out.push(cands[idx]);
    }
    out
}

pub(in crate::retrieval) fn apply_slot0_locked_top10_rrf(
    inp: &Top10RrfInputs<'_>,
    final_session_indices: &mut [usize],
) {
    let n = final_session_indices.len().min(TOP10_POOL_CAP);
    if n < MIN_POOL_LEN {
        return;
    }
    let pool: Vec<usize> = final_session_indices[..n].to_vec();

    // 1. Sight rank
    let mut sight_ord: Vec<usize> = (0..n).collect();
    sight_ord.sort_by(|&a, &b| {
        inp.sight_all_q16[pool[b]]
            .cmp(&inp.sight_all_q16[pool[a]])
            .then_with(|| pool[a].cmp(&pool[b]))
    });
    let mut sight_rank = vec![0usize; n];
    for (r, &idx) in sight_ord.iter().enumerate() {
        sight_rank[idx] = r;
    }

    // 2. MRF / PLM / MinCover rank
    let (mrf_scores, _) = pool_mrf_and_exact_ic_q16(inp.ix, inp.plan, &pool, inp.s_exact_mask);
    let mut mrf_ord: Vec<usize> = (0..n).collect();
    mrf_ord.sort_by(|&a, &b| {
        mrf_scores[b]
            .cmp(&mrf_scores[a])
            .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
            .then_with(|| pool[a].cmp(&pool[b]))
    });
    let mut mrf_rank = vec![0usize; n];
    for (r, &idx) in mrf_ord.iter().enumerate() {
        mrf_rank[idx] = r;
    }

    // 3. Turn-evidence rank
    let turn_pos: Vec<usize> = pool
        .iter()
        .map(|&si| {
            inp.final_turn_indices
                .iter()
                .position(|&ti| inp.ix.turns[ti].session == si)
                .unwrap_or(usize::MAX)
        })
        .collect();
    let mut turn_ord: Vec<usize> = (0..n).collect();
    turn_ord.sort_by(|&a, &b| {
        turn_pos[a]
            .cmp(&turn_pos[b])
            .then_with(|| inp.best_scene_q16[pool[b]].cmp(&inp.best_scene_q16[pool[a]]))
            .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
    });
    let mut turn_rank = vec![0usize; n];
    for (r, &idx) in turn_ord.iter().enumerate() {
        turn_rank[idx] = r;
    }

    // 4. Query-cue coverage rank
    let q_cues = query_cues(inp.ix, inp.q_text);
    let cue_cnt: Vec<usize> = pool
        .iter()
        .map(|&si| {
            let mut u: Vec<u32> = Vec::new();
            if let Some(members) = inp.ix.session_turns.get(si) {
                for &ti in members {
                    for c in turn_query_cues(inp.ix, &q_cues, ti) {
                        if !u.contains(&c) {
                            u.push(c);
                        }
                    }
                }
            }
            u.len()
        })
        .collect();
    let mut cue_ord: Vec<usize> = (0..n).collect();
    cue_ord.sort_by(|&a, &b| {
        cue_cnt[b]
            .cmp(&cue_cnt[a])
            .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
            .then_with(|| pool[a].cmp(&pool[b]))
    });
    let mut cue_rank = vec![0usize; n];
    for (r, &idx) in cue_ord.iter().enumerate() {
        cue_rank[idx] = r;
    }

    // 5. Historiographical firsthand-witness corroboration rank
    let witness_all = session_firsthand_witness_scores_q16(inp.ix, inp.plan);
    let mut witness_ord: Vec<usize> = (0..n).collect();
    witness_ord.sort_by(|&a, &b| {
        witness_all[pool[b]]
            .cmp(&witness_all[pool[a]])
            .then_with(|| sight_rank[a].cmp(&sight_rank[b]))
            .then_with(|| pool[a].cmp(&pool[b]))
    });
    let mut witness_rank = vec![0usize; n];
    for (r, &idx) in witness_ord.iter().enumerate() {
        witness_rank[idx] = r;
    }

    let reordered = rrf_reorder_tail_5ballot_q16(
        &pool,
        &sight_rank,
        &mrf_rank,
        &turn_rank,
        &cue_rank,
        &witness_rank,
    );
    final_session_indices[..n].copy_from_slice(&reordered);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5ballot_rrf_locks_slot0_and_promotes_rank8_consensus_session() {
        let cands = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let sr = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mr = vec![0, 6, 7, 5, 4, 3, 2, 1];
        let tr = vec![0, 7, 6, 5, 4, 3, 2, 1];
        let cr = vec![0, 7, 6, 5, 4, 3, 2, 1];
        let wr = vec![0, 7, 6, 5, 4, 3, 2, 1];
        let out = rrf_reorder_tail_5ballot_q16(&cands, &sr, &mr, &tr, &cr, &wr);
        assert_eq!(out[0], 10);
        assert!(out[..5].contains(&80));
    }
}
