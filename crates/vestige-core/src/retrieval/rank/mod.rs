//! Stage 6: Straight-line 13-stage per-query ranking pipeline (`rank_one`).

pub(in crate::retrieval) mod context;
pub(in crate::retrieval) mod event_ordering;
pub(in crate::retrieval) mod genesis_spine_q16;
pub(in crate::retrieval) mod hawkes;
pub(in crate::retrieval) mod implicit_bridge;
pub(in crate::retrieval) mod speaker_disposition_cone_q16;
pub(in crate::retrieval) mod supersession_and_guard;
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
use std::cell::RefCell;
use std::sync::OnceLock;

pub(in crate::retrieval) fn waterfall_log_path() -> Option<&'static str> {
    static WF_PATH: OnceLock<Option<String>> = OnceLock::new();
    WF_PATH
        .get_or_init(|| std::env::var("VESTIGE_WATERFALL_LOG").ok().filter(|s| !s.is_empty()))
        .as_deref()
}

thread_local! {
    static WF_FLAGS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

pub(in crate::retrieval) fn record_wf_flag(flag: &'static str) {
    if waterfall_log_path().is_some() {
        WF_FLAGS.with(|f| {
            let mut v = f.borrow_mut();
            if !v.contains(&flag) {
                v.push(flag);
            }
        });
    }
}

pub(in crate::retrieval) fn take_wf_flags() -> Vec<&'static str> {
    WF_FLAGS.with(|f| std::mem::take(&mut *f.borrow_mut()))
}

pub(in crate::retrieval) fn rank_one(
    ix: &Index,
    q: &Query,
    p: &Params,
    top_k: usize,
) -> RankResult {
    let wf_on = waterfall_log_path().is_some();
    if wf_on {
        let _ = take_wf_flags();
    }
    let mut ctx = stage01_session_bm25f::run(ix, q, p, top_k);
    stage02_turn_scores::run(&mut ctx);
    stage03_role_and_rungs::run(&mut ctx);
    stage04_floors_and_bridge::run(&mut ctx);
    if wf_on {
        let mut s_ord: Vec<usize> = (0..ctx.ns).collect();
        s_ord.sort_by(|&a, &b| {
            let sa = ctx.s_score[a] + ctx.bridge_boost.get(a).copied().unwrap_or(0.0);
            let sb = ctx.s_score[b] + ctx.bridge_boost.get(b).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        ctx.wf_g1_s10 = s_ord
            .into_iter()
            .take(10)
            .map(|si| ix.session_ids[si].clone())
            .collect();
    }
    stage05_submodular_sessions::run(&mut ctx);
    stage06_turn_cover_and_quota::run(&mut ctx);
    stage07_edges_and_cues::run(&mut ctx);
    stage08_c2_seating::run(&mut ctx);
    stage09_w2_and_closers::run(&mut ctx);
    stage10_prf_and_tail::run(&mut ctx);
    stage11_multihop_and_torrens::run(&mut ctx);
    if wf_on {
        ctx.wf_g3_t10 = ctx
            .final_turn_indices
            .iter()
            .take(10)
            .map(|&ti| ix.turns[ti].id.clone())
            .collect();
    }
    stage12_session_sync_and_fusion::run(&mut ctx);
    stage13_finalize::run(ctx)
}
