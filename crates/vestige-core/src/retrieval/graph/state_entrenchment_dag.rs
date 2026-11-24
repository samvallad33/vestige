//! Engine #3: Transitive `derived_from` State Invalidation + AGM Epistemic Entrenchment DAG.
//!
//! Pure deterministic integer graph pass (zero LLMs, zero vectors, zero open-class lists):
//! 1. **AGM Epistemic Entrenchment $\mathcal{E}(t) \in \{0, 1, 2, 3\}$**:
//!    - `0`: Interrogative or purely hypothetical (`is_question` or `modality_mask & 0x04`).
//!    - `1`: Temporary / ephemeral one-off exception (`"for today"`, `"just this time"`, `"temporarily"`).
//!    - `2`: Standard asserted episodic state.
//!    - `3`: Permanent standing policy / durable user constraint (`is_standing_policy`,
//!           `is_personal_constraint`, `"always"`, `"never"`, `"from now on"`).
//!    Contraction Law: `t_new` may supersede `t_old` only if $\mathcal{E}(t_{\text{new}}) \ge \mathcal{E}(t_{\text{old}})$
//!    or `t_new` carries an explicit permanent revocation cue.
//! 2. **Transitive `derived_from` Stale Invalidation (1-pass BFS)**:
//!    When root turn `r` is legitimately superseded, propagates `transitive_stale = true`
//!    downstream across `exact_edges`, `policy_backlinks`, `answers`, and same-session
//!    anaphoric continuations (`depth <= 6`).

use std::collections::VecDeque;

use crate::retrieval::index::Index;

/// AGM Epistemic Entrenchment tier (`0 = lowest`, `3 = highest`).
pub(in crate::retrieval) const ENTRENCH_HYPOTHETICAL: u8 = 0;
pub(in crate::retrieval) const ENTRENCH_TEMPORARY: u8 = 1;
pub(in crate::retrieval) const ENTRENCH_EPISODIC: u8 = 2;
pub(in crate::retrieval) const ENTRENCH_PERMANENT: u8 = 3;

/// Maximum downstream derivation hops invalidated by one superseded root.
const MAX_DERIVATION_HOPS: u8 = 6;

/// Classify a turn's AGM Epistemic Entrenchment rank $\mathcal{E}(t) \in \{0..=3\}$
/// strictly from structural flags and closed-class temporal/deontic grammar.
/// hardcode-scan: closed-class (ephemeral vs permanent temporal/deontic markers only).
pub(in crate::retrieval) fn classify_agm_entrenchment(ix: &Index, ti: usize) -> u8 {
    let tr = &ix.turns[ti];
    if tr.is_question || (tr.modality_mask & 0x04) != 0 {
        return ENTRENCH_HYPOTHETICAL;
    }
    let low = tr.body_for_headers.to_ascii_lowercase();
    let is_temporary = [
        "for today",
        "just today",
        "just this time",
        "just this once",
        "this one time",
        "for now only",
        "temporarily",
        "temporary ",
        "for tonight",
        "this afternoon only",
        "right now only",
    ]
    .iter()
    .any(|p| low.contains(p));
    if is_temporary && !tr.is_standing_policy {
        return ENTRENCH_TEMPORARY;
    }

    let is_permanent = tr.is_standing_policy
        || tr.is_personal_constraint
        || tr.is_habitual
        || [
            "from now on",
            "going forward",
            "permanently",
            "i always ",
            "i never ",
            "please always ",
            "please never ",
        ]
        .iter()
        .any(|p| low.contains(p));
    if is_permanent {
        return ENTRENCH_PERMANENT;
    }
    ENTRENCH_EPISODIC
}

/// Closed-class permanent revocation cue allowing a newer turn to contract an entrenched belief.
/// hardcode-scan: closed-class (explicit belief-revocation markers only).
fn has_explicit_permanent_revocation(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    [
        "no longer",
        "not anymore",
        "anymore",
        "from now on",
        "permanently",
        "stop using",
        "stopped using",
        "switched to",
        "replaced with",
        "changed my mind",
        "cancel that rule",
        "ignore my previous",
    ]
    .iter()
    .any(|p| low.contains(p))
}

/// AGM Contraction Gate: true iff `newer` has sufficient epistemic entrenchment to supersede `older`.
#[inline]
pub(in crate::retrieval) fn agm_may_supersede(ix: &Index, newer: usize, older: usize) -> bool {
    if newer >= ix.turns.len() || older >= ix.turns.len() || newer <= older {
        return false;
    }
    let e_new = classify_agm_entrenchment(ix, newer);
    let e_old = classify_agm_entrenchment(ix, older);
    if e_new == ENTRENCH_HYPOTHETICAL {
        return false;
    }
    if e_new >= e_old {
        return true;
    }
    // Temporary (`1`) cannot overwrite Permanent (`3`) even if it mentions an entity;
    // Episodic (`2`) can contract Permanent (`3`) only with explicit revocation phrasing.
    e_new == ENTRENCH_EPISODIC && has_explicit_permanent_revocation(&ix.turns[newer].body_for_headers)
}

/// Output of AGM Entrenchment + Transitive `derived_from` Stale Invalidation.
pub(in crate::retrieval) struct EntrenchmentDagState {
    pub(in crate::retrieval) entrenchment: Vec<u8>,
    /// True if `ti` is directly superseded under AGM rules OR transitively derived from a stale root.
    pub(in crate::retrieval) transitive_stale: Vec<bool>,
}

/// Run 1-pass BFS transitive staleness propagation across `derived_from` links,
/// protecting entrenched policies (`E = 3`) from temporary (`E = 1`) overrides.
pub(in crate::retrieval) fn compute_entrenchment_staleness_dag(ix: &Index) -> EntrenchmentDagState {
    let nt = ix.turns.len();
    let entrenchment: Vec<u8> = (0..nt).map(|ti| classify_agm_entrenchment(ix, ti)).collect();
    let mut stale = vec![false; nt];
    let mut queue: VecDeque<(usize, u8)> = VecDeque::new();

    // 1. Seed valid superseded roots respecting AGM Entrenchment.
    for (newer, tr) in ix.turns.iter().enumerate() {
        for &older in &tr.supersedes_turns {
            if older < nt && agm_may_supersede(ix, newer, older) && !stale[older] {
                stale[older] = true;
                queue.push_back((older, 0));
            }
        }
    }
    for ti in 0..nt {
        if ix.torrens_superseded.get(ti).copied().unwrap_or(false) && !stale[ti] {
            stale[ti] = true;
            queue.push_back((ti, 0));
        }
    }

    if queue.is_empty() {
        return EntrenchmentDagState {
            entrenchment,
            transitive_stale: stale,
        };
    }

    // 2. Build forward `derived_by[parent] -> Vec<child>` adjacency (`child > parent`).
    let mut derived_by: Vec<Vec<usize>> = vec![Vec::new(); nt];
    let mut add_deriv = |parent: usize, child: usize| {
        if parent < nt && child < nt && child > parent && !derived_by[parent].contains(&child) {
            derived_by[parent].push(child);
        }
    };
    for (child, tr) in ix.turns.iter().enumerate() {
        for &(_, parent) in &tr.exact_edges {
            add_deriv(parent, child);
        }
        for &parent in &tr.policy_backlinks {
            add_deriv(parent, child);
        }
        // Assistant recap/plan directly answering a superseded user turn inherits staleness.
        if let Some(parent) = tr.answers
            && tr.field == 1
        {
            add_deriv(parent, child);
        }
    }
    // Same-session anaphoric continuations carrying coref entities from an immediately preceding turn.
    for members in &ix.session_turns {
        for pos in 1..members.len() {
            let prev = members[pos - 1];
            let cur = members[pos];
            if ix.turns[cur].anaphoric
                && !ix.turns[cur].coref_entities.is_empty()
                && ix.turns[cur]
                    .coref_entities
                    .iter()
                    .any(|e| ix.turns[prev].entities.contains(e))
            {
                add_deriv(prev, cur);
            }
        }
    }

    // 3. Single-pass BFS transitive invalidation (`t1 -> t2 -> t3`).
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= MAX_DERIVATION_HOPS {
            continue;
        }
        for &child in &derived_by[node] {
            if stale[child] {
                continue;
            }
            // Never mark a superseding replacement turn itself as stale if it replaced `node`.
            if ix.turns[child].supersedes_turns.contains(&node) {
                continue;
            }
            // Do not cascade episodic staleness onto an independent permanent standing rule.
            if entrenchment[child] == ENTRENCH_PERMANENT && entrenchment[node] < ENTRENCH_PERMANENT {
                continue;
            }
            stale[child] = true;
            queue.push_back((child, depth + 1));
        }
    }

    EntrenchmentDagState {
        entrenchment,
        transitive_stale: stale,
    }
}

/// Demote transitively stale turns behind fresh candidates on current-state queries
/// (preserving Top-10 membership and never mutating historical/event-sequence queries).
pub(in crate::retrieval) fn apply_agm_entrenchment_demotion(
    ix: &Index,
    current_state: bool,
    is_historical_or_seq: bool,
    final_turn_indices: &mut Vec<usize>,
) {
    if !current_state || is_historical_or_seq || final_turn_indices.len() < 2 {
        return;
    }
    let dag = compute_entrenchment_staleness_dag(ix);
    let head = final_turn_indices[0];
    if !dag.transitive_stale[head] {
        return;
    }
    let window = final_turn_indices.len().min(10);
    if let Some(fresh_pos) = (1..window).find(|&slot| {
        let ti = final_turn_indices[slot];
        !dag.transitive_stale[ti]
            && !ix.turns[ti].is_question
            && !ix.turns[ti].is_backchannel
            && dag.entrenchment[ti] >= ENTRENCH_TEMPORARY
    }) {
        let fresh_ti = final_turn_indices.remove(fresh_pos);
        final_turn_indices.insert(0, fresh_ti);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{build_index, Params, Session, Turn, TurnLink};

    #[test]
    fn temporary_override_cannot_erase_permanent_policy_and_bfs_invalidates_chain() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "rule".into(),
                    role: "user".into(),
                    content: "Always use metric units when formatting measurements.".into(),
                    ..Turn::default()
                }],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 200,
                turns: vec![Turn {
                    id: "temp".into(),
                    role: "user".into(),
                    content: "Just for today, show imperial units for this measurement.".into(),
                    ..Turn::default()
                }],
            },
            Session {
                id: "s2".into(),
                date: String::new(),
                timestamp: 300,
                turns: vec![
                    Turn {
                        id: "plan_v1".into(),
                        role: "user".into(),
                        content: "Budget is set to 200 dollars for Orion.".into(),
                        ..Turn::default()
                    },
                    Turn {
                        id: "derived_v1".into(),
                        role: "assistant".into(),
                        content: "I built the Orion schedule from the 200 dollar budget.".into(),
                        links: vec![TurnLink {
                            kind: "derived_from".into(),
                            to: "plan_v1".into(),
                            degree: 2,
                        }],
                        ..Turn::default()
                    },
                ],
            },
            Session {
                id: "s3".into(),
                date: String::new(),
                timestamp: 400,
                turns: vec![Turn {
                    id: "plan_v2".into(),
                    role: "user".into(),
                    content: "Actually Orion budget changed from 200 to 500 dollars.".into(),
                    ..Turn::default()
                }],
            },
        ];
        let mut ix = build_index(&sessions, &Params::default());
        assert_eq!(classify_agm_entrenchment(&ix, 0), ENTRENCH_PERMANENT);
        assert_eq!(classify_agm_entrenchment(&ix, 1), ENTRENCH_TEMPORARY);
        assert!(!agm_may_supersede(&ix, 1, 0));

        // Ensure plan_v2 -> plan_v1 supersession cascades to derived_v1 via BFS.
        if !ix.turns[4].supersedes_turns.contains(&2) {
            ix.turns[4].supersedes_turns.push(2);
        }
        let dag = compute_entrenchment_staleness_dag(&ix);
        assert!(!dag.transitive_stale[0]);
        assert!(dag.transitive_stale[2]);
        assert!(dag.transitive_stale[3]);
        assert!(!dag.transitive_stale[4]);
    }
}
