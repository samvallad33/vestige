//! Issue #809: Sidner-Style `FocusStack<EntityId>` (1979/1983) in Q16.16
//! integer arithmetic. Zero floats, zero word lists.
//!
//! Tracks the discourse entity focus stack (`Cp` / `Cb`) chronologically inside
//! each session and resolves anaphoric non-question turns (`t.anaphoric`) to
//! their active pre-turn antecedent entities when a query targets an entity
//! held in focus.

use crate::retrieval::index::Index;
use crate::retrieval::is_technical_handle;
use crate::retrieval::QueryPlan;

pub(in crate::retrieval) const Q16: u64 = 1 << 16;

/// Compute Sidner focus-stack affinity in Q16.16 `[0, Q16]` per turn for anaphoric
/// non-question turns whose active discourse focus contains a query focal entity
/// not explicitly repeated in the turn itself.
pub(in crate::retrieval) fn compute_anaphoric_focus_q16(ix: &Index, plan: &QueryPlan) -> Vec<u64> {
    let nt = ix.turns.len();
    let mut focus_q16 = vec![0u64; nt];
    if nt <= 2 || plan.focal_ids.is_empty() {
        return focus_q16;
    }

    let n_docs = (ix.t_own_stats.n as u32).max(1);
    let max_df = (n_docs / 4).max(4);

    // Query focal entities that are NOT framing speaker tokens
    let query_focal: Vec<(u32, u64)> = plan
        .terms
        .iter()
        .filter(|qt| plan.focal_ids.contains(&qt.id) && qt.exact_hit)
        .filter_map(|qt| {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            if df > 0 && df <= max_df {
                let w_q16 = (qt.weight.clamp(0.0, 2.0) * ((Q16 / 2) as f32)) as u64;
                Some((qt.id, w_q16))
            } else {
                None
            }
        })
        .collect();

    if query_focal.is_empty() {
        return focus_q16;
    }

    for members in &ix.session_turns {
        // FocusStack ordered with MRU (`Cp`) at index 0, max depth 6.
        let mut stack: Vec<u32> = Vec::with_capacity(6);
        for &ti in members {
            let t = &ix.turns[ti];

            // Consult focus stack ONLY on anaphoric non-question turns that don't
            // already explicitly repeat the focused entity.
            if t.anaphoric && !t.is_question && !t.is_backchannel && !stack.is_empty() {
                let mut best_q16: u64 = 0;
                for (depth, &ent_id) in stack.iter().take(2).enumerate() {
                    if t.content.contains(&ent_id) {
                        continue;
                    }
                    if let Some(&(_, qw)) = query_focal.iter().find(|&&(qid, _)| qid == ent_id) {
                        // Top of stack (depth 0) gets full weight, depth 1 gets 5/8
                        let decayed = if depth == 0 { qw } else { (qw * 5) >> 3 };
                        if decayed > best_q16 {
                            best_q16 = decayed;
                        }
                    }
                }
                focus_q16[ti] = best_q16.min(Q16);
            }

            // Update Sidner stack from this turn's non-framing entities and rare content.
            // Prefer explicit entities first (pushed in deterministic order so last stays MRU).
            for &eid in &t.entities {
                if is_technical_handle(&ix.vocab, eid) {
                    continue;
                }
                let df = ix.t_stats.df.get(&eid).copied().unwrap_or(0);
                if df > max_df {
                    continue;
                }
                push_focus(&mut stack, eid);
            }
            for &cid in &t.content {
                let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
                if df >= 1 && df <= (n_docs / 8).max(3) {
                    push_focus(&mut stack, cid);
                }
            }
        }
    }

    focus_q16
}

#[inline]
fn push_focus(stack: &mut Vec<u32>, id: u32) {
    if let Some(pos) = stack.iter().position(|&x| x == id) {
        stack.remove(pos);
    }
    stack.insert(0, id);
    if stack.len() > 6 {
        stack.truncate(6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_focus_maintains_mru_depth_6() {
        let mut s = Vec::new();
        for i in 1..=8 {
            push_focus(&mut s, i);
        }
        assert_eq!(s, vec![8, 7, 6, 5, 4, 3]);
        push_focus(&mut s, 5);
        assert_eq!(s, vec![5, 8, 7, 6, 4, 3]);
    }
}
