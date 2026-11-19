//! Closed-Class Deontic & First-Person Standing-Constraint Matcher for
//! long-horizon Preference Following and Instruction Following.
//!
//! Grounded in:
//! - Palmer (2001) Mood and Modality (deontic obligation/prohibition & bouletic preference),
//! - Karttunen (1973, 1977) presupposition projection & question answer-set semantics.
//!
//! Slot-protection guarantee: standing constraints only fire when sharing high-IC content
//! entities with the query (`shared_ic_q16 >= min_entity_ic_q16`) and are gated strictly
//! to `eff_it.preference || eff_it.task_under_rules`, preventing unrelated slot theft.

use std::collections::{BTreeMap, BTreeSet};

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;

/// Closed-class first-person standing preference / deontic constraint detector.
/// hardcode-scan: closed-class (1SG pronouns, modals, negation, universal adverbs, imperatives only).
pub(in crate::retrieval) fn is_standing_constraint_utterance(tokens: &[&str]) -> bool {
    if tokens.len() < 3 {
        return false;
    }
    let low: Vec<String> = tokens.iter().map(|t| t.to_ascii_lowercase()).collect();

    // Pattern 1: First-person pronoun + closed-class deontic/habitual marker.
    let has_1sg = low.iter().any(|w| matches!(w.as_str(), "i" | "my" | "me" | "we" | "our"));
    let has_deontic = low.iter().any(|w| {
        matches!(
            w.as_str(),
            "always" | "never" | "must" | "should" | "don't" | "dont" | "cannot" | "can't"
        )
    });
    if has_1sg && has_deontic {
        return true;
    }

    // Pattern 2: Clause-initial imperative constraint ("always ...", "never ...", "don't ...", "make sure ...").
    if matches!(low[0].as_str(), "always" | "never" | "don't" | "dont" | "please") {
        return true;
    }
    if low.windows(2).any(|w| w[0] == "make" && w[1] == "sure") {
        return true;
    }
    false
}

/// Score a candidate standing constraint turn `turn_entities` against query high-IC entities
/// `query_hi_ic_q16` in `Q16.16`.
///
/// Returns `0` unless the constraint is active (not superseded), passes the closed-class
/// deontic grammar check, and shares at least `min_shared_ic_q16` entity information content
/// with the query (guaranteeing zero unrelated-slot theft).
pub(in crate::retrieval) fn score_standing_constraint_q16(
    is_deontic_constraint: bool,
    is_active: bool,
    turn_entities: &BTreeSet<u32>,
    query_hi_ic_q16: &BTreeMap<u32, u64>,
    min_shared_ic_q16: u64,
) -> u64 {
    if !is_deontic_constraint || !is_active || query_hi_ic_q16.is_empty() {
        return 0;
    }
    let mut shared_ic: u64 = 0;
    let mut total_q_ic: u64 = 0;
    for (&eid, &ic_q16) in query_hi_ic_q16 {
        total_q_ic = total_q_ic.saturating_add(ic_q16);
        if turn_entities.contains(&eid) {
            shared_ic = shared_ic.saturating_add(ic_q16);
        }
    }
    if shared_ic < min_shared_ic_q16 || total_q_ic == 0 {
        return 0;
    }
    (((shared_ic as u128) << 16) / (total_q_ic as u128)).min(Q16_ONE as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_closed_class_deontic_and_first_person_constraints() {
        assert!(is_standing_constraint_utterance(&["I", "always", "use", "metric", "units"]));
        assert!(is_standing_constraint_utterance(&["Never", "reply", "with", "bullet", "points"]));
        assert!(is_standing_constraint_utterance(&["Make", "sure", "to", "cite", "sources"]));
        assert!(!is_standing_constraint_utterance(&["The", "train", "arrived", "at", "noon"]));
    }

    #[test]
    fn entity_ic_binding_blocks_unrelated_slot_theft() {
        let mut q_ic = BTreeMap::new();
        q_ic.insert(42, Q16_ONE * 4);
        let mut related = BTreeSet::new();
        related.insert(42);
        let mut unrelated = BTreeSet::new();
        unrelated.insert(99);

        assert_eq!(
            score_standing_constraint_q16(true, true, &related, &q_ic, Q16_ONE * 2),
            Q16_ONE
        );
        assert_eq!(
            score_standing_constraint_q16(true, true, &unrelated, &q_ic, Q16_ONE * 2),
            0
        );
        // Superseded constraint scores 0.
        assert_eq!(
            score_standing_constraint_q16(true, false, &related, &q_ic, Q16_ONE * 2),
            0
        );
    }
}
