//! Unlock 3: Structured `[tool_call ...]` Receipt Outcome Priority.
//!
//! Strictly gated on `ctx.ix.has_tool_calls` (zero execution and zero diff on
//! plain conversational corpora). When a corpus contains serialized tool-call
//! execution receipts (`[tool_call fn(args) -> result]`), pairs index-time
//! `counts_len: false` length isolation with execution-outcome priority:
//! 1. Refines Top-10 session ranking using executed `[tool_call]` lexical +
//!    exact technical handle (`ORD-6014`, `BK-1000`) and numeric/amount evidence.
//! 2. Seats the best executed `[tool_call]` turn from each top session ahead of
//!    plain user request/confirmation turns so verification queries hit the
//!    ground-truth state mutation directly at Rank 1 and Top-5.

use crate::retrieval::index::normalize::{digit_token_set, extract_exact_technical_handles};
use crate::retrieval::rank::context::RankCtx;
use crate::retrieval::*;

/// Apply structured `[tool_call]` receipt refinement and outcome priority.
pub(in crate::retrieval) fn apply_tool_receipt_priority(ctx: &mut RankCtx<'_>) {
    if !ctx.ix.has_tool_calls || ctx.final_turn_indices.is_empty() {
        return;
    }

    // Collect query technical handles (`ORD-6014`, `BK-1000`, `UA100`) and numeric tokens (`249`, `52`).
    let mut q_handle_ids: Vec<u32> = Vec::new();
    for h in extract_exact_technical_handles(&ctx.q.text) {
        if let Some(id) = ctx.ix.vocab.get(&h)
            && !q_handle_ids.contains(&id)
        {
            q_handle_ids.push(id);
        }
    }
    let mut q_digit_ids: Vec<u32> = Vec::new();
    for d in digit_token_set(&ctx.q.text) {
        if let Some(id) = ctx.ix.vocab.get(&d)
            && !q_digit_ids.contains(&id)
        {
            q_digit_ids.push(id);
        }
    }

    let count_hits = |tr: &TurnIx, ids: &[u32]| -> u32 {
        ids.iter()
            .filter(|id| tr.entities.contains(id) || tr.content.contains(id))
            .count() as u32
    };

    // Rare handle IDs (`df <= 5`) strongly pinpoint the exact episode/order.
    let rare_handle_ids: Vec<u32> = q_handle_ids
        .iter()
        .copied()
        .filter(|id| {
            ctx.ix
                .t_own_stats
                .df
                .get(id)
                .copied()
                .is_some_and(|df| (1..=5).contains(&df))
        })
        .collect();

    // Step 1: Refine Top-10 sessions using executed tool-receipt turn evidence + rare handles.
    let s_window = ctx.final_session_indices.len().min(10);
    let mut cand_sessions: Vec<usize> = ctx.final_session_indices[..s_window].to_vec();
    if !rare_handle_ids.is_empty() {
        for tr in &ctx.ix.turns {
            if tr.has_tool_call
                && count_hits(tr, &rare_handle_ids) > 0
                && !cand_sessions.contains(&tr.session)
            {
                cand_sessions.push(tr.session);
            }
        }
    }

    let mut sess_scored: Vec<(usize, f32)> = Vec::with_capacity(cand_sessions.len());
    for &si in &cand_sessions {
        let mut best_tool = 0.0f32;
        let mut rare_hit = false;
        for &ti in &ctx.ix.session_turns[si] {
            let tr = &ctx.ix.turns[ti];
            if !tr.has_tool_call {
                continue;
            }
            let h = count_hits(tr, &q_handle_ids) as f32;
            let rh = count_hits(tr, &rare_handle_ids);
            if rh > 0 {
                rare_hit = true;
            }
            let d = count_hits(tr, &q_digit_ids) as f32;
            let ex = (ctx.t_exact_mask[ti] & ctx.grounded_group_mask).count_ones() as f32;
            let sc = ctx.t_own[ti].max(0.0)
                + 0.45 * ctx.t_scene[ti].max(0.0)
                + 3.0 * h
                + 1.10 * d
                + 0.55 * ex;
            if sc > best_tool {
                best_tool = sc;
            }
        }
        let combined = ctx.s_score[si].max(0.0)
            + 0.55 * best_tool
            + if rare_hit { 6.0 } else { 0.0 };
        sess_scored.push((si, combined));
    }
    sess_scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    if !sess_scored.is_empty() {
        let mut new_sessions: Vec<usize> = sess_scored.iter().map(|&(si, _)| si).collect();
        for &si in &ctx.final_session_indices {
            if !new_sessions.contains(&si) {
                new_sessions.push(si);
            }
        }
        ctx.final_session_indices = new_sessions;
    }

    // Step 2: Score executed `[tool_call]` turns across Top-10 sessions + Top-20 turns.
    let top_s_limit = ctx.final_session_indices.len().min(10);
    let mut sess_rank_map: DetHashMap<usize, usize> = DetHashMap::default();
    for (rk, &si) in ctx.final_session_indices[..top_s_limit].iter().enumerate() {
        sess_rank_map.insert(si, rk);
    }

    let mut tool_turns: Vec<usize> = Vec::new();
    let mut seen_t: DetHashSet<usize> = DetHashSet::default();
    for &si in &ctx.final_session_indices[..top_s_limit] {
        for &ti in &ctx.ix.session_turns[si] {
            if ctx.ix.turns[ti].has_tool_call && seen_t.insert(ti) {
                tool_turns.push(ti);
            }
        }
    }
    for &ti in ctx.final_turn_indices.iter().take(20) {
        if ctx.ix.turns[ti].has_tool_call && seen_t.insert(ti) {
            tool_turns.push(ti);
        }
    }

    if tool_turns.is_empty() {
        return;
    }

    let smax = tool_turns
        .iter()
        .map(|&ti| ctx.s_score[ctx.ix.turns[ti].session].max(0.0))
        .fold(1e-6f32, f32::max);
    let tmax = tool_turns
        .iter()
        .map(|&ti| ctx.t_own[ti].max(0.0))
        .fold(1e-6f32, f32::max);

    let mut scored_tools: Vec<(usize, usize, f32)> = Vec::with_capacity(tool_turns.len());
    for &ti in &tool_turns {
        let tr = &ctx.ix.turns[ti];
        let si = tr.session;
        let s_rank = sess_rank_map.get(&si).copied().unwrap_or(10);
        let s_norm = (ctx.s_score[si].max(0.0) / smax).clamp(0.0, 2.0);
        let t_norm = (ctx.t_own[ti].max(0.0) / tmax).clamp(0.0, 2.0);
        let ex = (ctx.t_exact_mask[ti] & ctx.grounded_group_mask).count_ones() as f32;
        let h = count_hits(tr, &q_handle_ids) as f32;
        let rh = count_hits(tr, &rare_handle_ids) as f32;
        let d = count_hits(tr, &q_digit_ids) as f32;
        let lead_bonus = match s_rank {
            0 => 1.25,
            1 => 0.50,
            2 => 0.25,
            _ => 0.0,
        };
        let sc = 1.65 * s_norm
            + 1.15 * t_norm
            + 0.22 * ex
            + 1.40 * h
            + 1.20 * rh
            + 0.45 * d
            + lead_bonus;
        scored_tools.push((ti, si, sc));
    }
    scored_tools.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Pass 1: Seat the best executed `[tool_call]` turn per distinct session first
    // (maximizing both Turn R@1 on winning session and Turn R@5 across Top-5 sessions).
    let mut reordered: Vec<usize> = Vec::with_capacity(ctx.final_turn_indices.len().max(10));
    let mut used_t: DetHashSet<usize> = DetHashSet::default();
    let mut used_s: DetHashSet<usize> = DetHashSet::default();
    for &(ti, si, _) in &scored_tools {
        if used_s.insert(si) && used_t.insert(ti) {
            reordered.push(ti);
        }
    }
    // Pass 2: Append remaining executed `[tool_call]` turns (secondary mutations in Top-10).
    for &(ti, _, _) in &scored_tools {
        if used_t.insert(ti) {
            reordered.push(ti);
        }
    }
    // Pass 3: Append remaining non-tool turns preserving original order.
    for &ti in &ctx.final_turn_indices {
        if used_t.insert(ti) {
            reordered.push(ti);
        }
    }
    ctx.final_turn_indices = reordered;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    #[test]
    fn tool_call_receipt_beats_user_confirmation_without_length_penalty() {
        let big_json = format!(
            "[tool_call get_order({{\"order_id\": \"ORD-6014\"}}) -> {{\"order_id\": \"ORD-6014\", \"status\": \"returned\", \"refund\": 94, \"payload\": \"{}\"}}]",
            "item metadata field ".repeat(120)
        );
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "ep_return".into(),
                    date: "2026/07/02 (Thu) 10:00".into(),
                    timestamp: 1,
                    turns: vec![
                        mk_turn(
                            "t0",
                            "user",
                            "Please process the full order return for all three items — the shirt, book, and phone case in ORD-6014.",
                        ),
                        mk_turn(
                            "t1",
                            "assistant",
                            &format!(
                                "I have processed the returns for all three items in order ORD-6014 for a total refund of $94.\n{big_json}"
                            ),
                        ),
                    ],
                },
                Session {
                    id: "ep_other".into(),
                    date: "2026/07/03 (Fri) 10:00".into(),
                    timestamp: 2,
                    turns: vec![
                        mk_turn("o0", "user", "Can you check my shipping status for ORD-9999?"),
                        mk_turn(
                            "o1",
                            "assistant",
                            "Order ORD-9999 is currently in transit.\n[tool_call get_order({\"order_id\": \"ORD-9999\"}) -> {\"status\": \"shipped\"}]",
                        ),
                    ],
                },
            ],
            queries: vec![Query {
                text: "Thanks. Just to confirm, the full order return is complete for all three items — the shirt, book, and phone case in ORD-6014?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        assert_eq!(res.results[0].turns[0].id, "t1");
        assert_eq!(res.results[0].sessions[0].id, "ep_return");
    }
}
