//! Stage 6: Straight-line 13-stage per-query ranking pipeline (`rank_one`).

pub(in crate::retrieval) mod context;
pub(in crate::retrieval) mod event_ordering;
pub(in crate::retrieval) mod tool_receipt;
mod stage01_session_bm25f;
mod stage02_turn_scores;
mod stage03_role_and_rungs;
mod stage04_floors_and_bridge;
mod stage05_submodular_sessions;
mod stage06_turn_cover_and_quota;
mod stage07_edges_and_cues;
mod stage08_c2_seating;
mod stage09_w2_and_closers;
mod stage10_prf_and_tail;
mod stage11_multihop_and_torrens;
mod stage12_session_sync_and_fusion;
mod stage13_finalize;

use crate::retrieval::*;

pub(in crate::retrieval) fn rank_one(
    ix: &Index,
    q: &Query,
    p: &Params,
    top_k: usize,
) -> RankResult {
    let mut ctx = stage01_session_bm25f::run(ix, q, p, top_k);
    stage02_turn_scores::run(&mut ctx);
    stage03_role_and_rungs::run(&mut ctx);
    stage04_floors_and_bridge::run(&mut ctx);
    stage05_submodular_sessions::run(&mut ctx);
    stage06_turn_cover_and_quota::run(&mut ctx);
    stage07_edges_and_cues::run(&mut ctx);
    stage08_c2_seating::run(&mut ctx);
    stage09_w2_and_closers::run(&mut ctx);
    stage10_prf_and_tail::run(&mut ctx);
    stage11_multihop_and_torrens::run(&mut ctx);
    stage12_session_sync_and_fusion::run(&mut ctx);
    stage13_finalize::run(ctx)
}
