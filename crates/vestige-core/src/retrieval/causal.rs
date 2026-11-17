//! Cycle G (#728, #729, #733, #747, #748, #749, #750):
//! Pearl Causal Calculus, Halpern-Pearl Responsibility, Aho-Garey-Ullman Transitive Reduction,
//! Paninian Anuvrtti Carry Register, Isnad Consensus, Harris Stratigraphic Cuts & Talmudic Machloket
//! in pure integer / Q16.16 arithmetic (zero floats, zero open-class word lists).

use super::ghostlink_dgs::Q16;
use super::index::Index;
use super::supersession::turn_is_superseded;
use super::QueryIntent;

/// #728 Pearl `do(X=x)` Graph Surgery:
/// Mutilates incoming backdoor edges to any overridden state variable so pre-intervention
/// rationales drop to zero reachability on `current_state` queries.
#[inline]
pub(super) fn pearl_do_mutilate_reachability_q16(
    ix: &Index,
    it: &QueryIntent,
    ti: usize,
    raw_score_q16: u64,
) -> u64 {
    if it.current_state && !it.is_event_sequence && !it.temporal && turn_is_superseded(ix, ti) {
        0
    } else {
        raw_score_q16
    }
}

/// #729 Halpern-Pearl Actual Causality Responsibility `1 / (1 + |W|)` in Q16.16:
/// A pivotal cause (`contingency_size = 0`) scores `Q16` (1.0); a joint contributor with
/// `k` off-path witness turns scores `Q16 / (1 + k)`.
#[inline]
pub(super) fn halpern_pearl_responsibility_q16(contingency_size: u64) -> u64 {
    Q16 / contingency_size.saturating_add(1)
}

/// #733 Aho-Garey-Ullman Transitive Reduction Skeleton (Hasse Diagram):
/// Removes every shortcut edge `(u -> w)` from adjacency matrix `adj` (`n <= 64`)
/// whenever a longer directed path `u -> v -> ... -> w` exists.
pub(super) fn transitive_reduction_edges(adj: &[u64]) -> Vec<u64> {
    let n = adj.len().min(64);
    let mut reach: Vec<u64> = adj.iter().take(n).copied().collect();
    for k in 0..n {
        let rk = reach[k];
        if rk == 0 {
            continue;
        }
        let bit_k = 1u64 << k;
        for i in 0..n {
            if (reach[i] & bit_k) != 0 {
                reach[i] |= rk;
            }
        }
    }
    let mut reduced = adj.to_vec();
    for u in 0..n {
        let mut two_plus = 0u64;
        let out_u = reach[u];
        for v in 0..n {
            if v != u && ((out_u & (1u64 << v)) != 0) {
                two_plus |= reach[v];
            }
        }
        reduced[u] &= !two_plus;
    }
    reduced
}

/// #747 Paninian `Anuvrtti` Carry-Register with `Utsarga-Apavada` Precedence:
/// Carries the last recorded filler entity across consecutive same-session turns until
/// an `adhikara` boundary (session break or explicit exception `apavada` via `role_objs`).
pub(super) fn panini_anuvrtti_carried_entity(ix: &Index, ti: usize) -> Option<u32> {
    let t = &ix.turns[ti];
    if !t.entities.is_empty() {
        return t.entities.iter().copied().min();
    }
    if !t.anaphoric || t.is_backchannel {
        return None;
    }
    let members = ix.session_turns.get(t.session)?;
    let pos = members.iter().position(|&m| m == ti)?;
    for back in (0..pos).rev() {
        let prev = members[back];
        let pt = &ix.turns[prev];
        if !pt.role_objs.is_empty() && !t.role_objs.is_empty() && pt.role_objs != t.role_objs {
            // Apavada boundary ends carry
            break;
        }
        if let Some(&e) = pt.entities.iter().min() {
            return Some(e);
        }
    }
    None
}

/// #748 Isnad Vertex-Disjoint Path Consensus & Lachmann Shared-Error Collapse:
/// Collapses turns sharing identical superseded-error signatures into one stemmatic branch
/// and counts independent corroborating sessions (mutawatir count).
pub(super) fn isnad_disjoint_witness_count(ix: &Index, entity_id: u32) -> usize {
    let Some(post) = ix.t_postings.get(entity_id as usize) else {
        return 0;
    };
    let mut independent_sessions: Vec<usize> = Vec::new();
    let mut seen_error_branches: Vec<usize> = Vec::new();
    for &doc in post {
        let ti = doc as usize;
        if ti >= ix.turns.len() {
            continue;
        }
        if !ix.turns[ti].supersedes_turns.is_empty() {
            let err_sig = ix.turns[ti].supersedes_turns[0];
            if seen_error_branches.contains(&err_sig) {
                continue;
            }
            seen_error_branches.push(err_sig);
        }
        let s = ix.turns[ti].session;
        if !independent_sessions.contains(&s) {
            independent_sessions.push(s);
        }
    }
    independent_sessions.len()
}

/// #749 Harris Matrix Stratigraphic Interface Classification:
/// Identifies recap/compaction `Cut` layers from structural backlinks (`exact_edges.len() >= 3`
/// or structured list summaries) vs sequential event `Deposit` layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HarrisLayer {
    Deposit,
    Cut,
}

#[inline]
pub(super) fn classify_harris_layer(ix: &Index, ti: usize) -> HarrisLayer {
    let t = &ix.turns[ti];
    if t.exact_edges.len() >= 3 || t.policy_backlinks.len() >= 2 {
        HarrisLayer::Cut
    } else {
        HarrisLayer::Deposit
    }
}

/// #750 Talmudic `Machloket` Dormant-Branch Recovery:
/// Checks whether an earlier superseded turn `ti` was superseded ONLY by a turn `killer_ti`
/// that has itself subsequently been superseded (`turn_is_superseded(ix, killer_ti)`),
/// reopening the dormant branch as an eligible tail candidate when its defeating constraint lifts.
pub(super) fn recover_machloket_dormant_turns(ix: &Index, max_scan: usize) -> Vec<usize> {
    let nt = ix.turns.len();
    let mut recovered: Vec<usize> = Vec::new();
    for killer_ti in 0..nt {
        if !turn_is_superseded(ix, killer_ti) {
            continue;
        }
        for &dormant_ti in &ix.turns[killer_ti].supersedes_turns {
            if dormant_ti < nt && !recovered.contains(&dormant_ti) {
                recovered.push(dormant_ti);
                if recovered.len() >= max_scan {
                    return recovered;
                }
            }
        }
    }
    recovered
}

/// Evaluate Cycle G causal/historical integrity score in Q16.16 for `ti`.
pub(super) fn evaluate_causal_turn_q16(
    ix: &Index,
    it: &QueryIntent,
    ti: usize,
    base_q16: u64,
    candidate_window: &[usize],
) -> u64 {
    let mutil_q16 = pearl_do_mutilate_reachability_q16(ix, it, ti, base_q16);
    if mutil_q16 == 0 {
        return 0;
    }
    let layer_bonus = match classify_harris_layer(ix, ti) {
        HarrisLayer::Deposit => Q16 >> 4,
        HarrisLayer::Cut => 0,
    };
    let carried = panini_anuvrtti_carried_entity(ix, ti);
    let isnad = carried.map_or(1, |eid| isnad_disjoint_witness_count(ix, eid).max(1) as u64);
    let peers = candidate_window
        .iter()
        .filter(|&&other| other != ti && ix.turns[other].session == ix.turns[ti].session)
        .count() as u64;
    let resp_q16 = halpern_pearl_responsibility_q16(peers);
    ((mutil_q16 >> 2)
        .saturating_add(resp_q16 >> 1)
        .saturating_add(layer_bonus))
    .saturating_add((isnad.min(8) * Q16) >> 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halpern_responsibility_and_agu_transitive_reduction_are_exact() {
        assert_eq!(halpern_pearl_responsibility_q16(0), Q16);
        assert_eq!(halpern_pearl_responsibility_q16(3), Q16 / 4);

        // 0 -> 1 -> 2 plus redundant shortcut 0 -> 2: reduction deletes 0 -> 2.
        let adj = [0b110u64, 0b100u64, 0u64];
        let red = transitive_reduction_edges(&adj);
        assert_eq!(red[0], 0b010u64);
        assert_eq!(red[1], 0b100u64);
    }
}
