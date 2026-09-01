//! Arm C2 — evidence-typed causal ranking (the v2 recovery layer).
//!
//! The v2 FAIL diagnosed the exact defect in C1: every backward same-entity
//! candidate carries a near-identical edge strength (entity_term dominates,
//! recency ≤0.3 breaks the tie), so recency ranks routine chatter above the
//! true cause, and depth-2 paths compound two lag decays until roots sink
//! below depth-1 chatter.
//!
//! C2 ranks on WHAT KIND of evidence a candidate is, derived from content and
//! corpus structure only — never from node metadata (`node_type` is not read;
//! a dataset that gives causes a distinctive type must not be won by reading
//! the label):
//!
//! - **Mutation** — the candidate *changes* the shared entity: the entity
//!   occurs in content immediately followed by an assignment/version operator
//!   (`=`, `:`, `@`), or the memory couples ≥2 distinct identifier entities
//!   (a transfer/copy/pin touching this entity and another). Root causes in
//!   real corpora are exactly this shape: `ORT_PIN=1.19.2`,
//!   `features=["fp16lib"]`, a value copied between configs.
//! - **Origin** — the earliest memory in the store that carries the shared
//!   entity: the provenance head of the entity's timeline.
//! - **Mention** — everything else: audits, listings, confirmations. Mentions
//!   never mutate state, so they are ranked below both, and they can never
//!   serve as a multi-hop intermediate.
//!
//! The additive bonuses (Mutation 2.0, Origin 1.0) dominate the base edge
//! strength (squashed ≤1.0 × lag decay), so evidence class decides rank and
//! the C1 score only breaks ties within a class. This is a prior about what
//! causes ARE, not a similarity: it is exactly the claim the spike exists to
//! test, and it is content-derived, so the seeder's adversarial change-verb
//! chatter ("Rotated the comment on X") stays a Mention — no assignment, one
//! entity, not first.

use super::causal_graph::lag_decay;
use super::{ArmRun, load_all_nodes};
use crate::types::{EVIDENCE_FLOOR, LOOKBACK_DAYS, QueryResult, TOP_K, squash_strength};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use vestige_core::KnowledgeNode;
use vestige_core::Storage;
use vestige_core::advanced::retroactive_backfill::extract_entities;

pub const MUTATION_BONUS: f64 = 2.0;
pub const ORIGIN_BONUS: f64 = 1.0;
/// Path credit for a depth-2 hop, applied to the compounded edge product.
pub const HOP_CREDIT: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    Mutation,
    Origin,
    Mention,
}

impl EvidenceClass {
    pub fn bonus(self) -> f64 {
        match self {
            EvidenceClass::Mutation => MUTATION_BONUS,
            EvidenceClass::Origin => ORIGIN_BONUS,
            EvidenceClass::Mention => 0.0,
        }
    }
}

/// True when `entity` occurs in `content` immediately followed (after
/// optional spaces) by an assignment or version operator. `API_TIMEOUT=2`,
/// `ORT_PIN: 1.19.2`, `usearch@2.16` all count; `Audited API_TIMEOUT during
/// the review` does not.
pub fn assignment_adjacent(content: &str, entity: &str) -> bool {
    if entity.is_empty() {
        return false;
    }
    // `extract_entities` lowercases identifiers for storage, so the entity
    // arrives lowercased while content keeps its original case (env vars are
    // usually UPPER_SNAKE). Fold both; positions below index the folded copy.
    let content = content.to_lowercase();
    let entity = entity.to_lowercase();
    let (content, entity) = (content.as_str(), entity.as_str());
    let mut start = 0usize;
    while let Some(pos) = content[start..].find(entity) {
        let at = start + pos;
        let end = at + entity.len();
        // Must be a whole-token occurrence: the char before must not be
        // alphanumeric/underscore (avoid matching inside a longer identifier).
        let ok_left = at == 0
            || !content[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if ok_left {
            let tail = content[end..].trim_start_matches(' ');
            if matches!(tail.chars().next(), Some('=') | Some(':') | Some('@')) {
                return true;
            }
        }
        start = end;
    }
    false
}

/// Identifier entities extracted from content AND tags, deduplicated.
fn node_entities(node: &KnowledgeNode) -> Vec<String> {
    let mut ents = extract_entities(&node.content, &node.tags);
    ents.sort();
    ents.dedup();
    ents
}

/// Classify one candidate's evidence with respect to the entity it shares
/// with the downstream node. `first_touch` maps entity → id of the earliest
/// memory carrying it (corpus structure, computed once per store).
pub fn classify(
    node: &KnowledgeNode,
    shared_entity: &str,
    entity_count: usize,
    first_touch: &HashMap<String, String>,
) -> EvidenceClass {
    if assignment_adjacent(&node.content, shared_entity) || entity_count >= 2 {
        return EvidenceClass::Mutation;
    }
    if first_touch.get(shared_entity).map(String::as_str) == Some(node.id.as_str()) {
        return EvidenceClass::Origin;
    }
    EvidenceClass::Mention
}

/// entity → id of the earliest memory carrying it (ties broken by id so the
/// map is deterministic).
fn first_touch_map(all: &[KnowledgeNode], entities_of: &HashMap<String, Vec<String>>) -> HashMap<String, String> {
    let mut first: HashMap<String, (chrono::DateTime<chrono::Utc>, String)> = HashMap::new();
    for node in all {
        for ent in entities_of.get(&node.id).into_iter().flatten() {
            match first.get(ent) {
                Some((t, id)) if (*t, id.as_str()) <= (node.created_at, node.id.as_str()) => {}
                _ => {
                    first.insert(ent.clone(), (node.created_at, node.id.clone()));
                }
            }
        }
    }
    first.into_iter().map(|(e, (_, id))| (e, id)).collect()
}

struct Candidate {
    id: String,
    class: EvidenceClass,
    base: f64,
}

/// Backward same-entity candidates for `target`, classified and base-scored.
/// Mirrors the B/C1 admission rule (strictly earlier, within lookback, ≥1
/// shared entity, evidence floor on the squashed base).
fn backward_candidates(
    target: &KnowledgeNode,
    all: &[KnowledgeNode],
    entities_of: &HashMap<String, Vec<String>>,
    first_touch: &HashMap<String, String>,
) -> Vec<Candidate> {
    let target_ents: HashSet<&str> = entities_of
        .get(&target.id)
        .into_iter()
        .flatten()
        .map(String::as_str)
        .collect();
    if target_ents.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for node in all {
        if node.id == target.id || node.created_at >= target.created_at {
            continue;
        }
        let days = (target.created_at - node.created_at).num_seconds() as f64 / 86_400.0;
        if days <= 0.0 || days > LOOKBACK_DAYS as f64 {
            continue;
        }
        let ents = entities_of.get(&node.id).into_iter().flatten();
        let shared: Vec<&String> = ents.filter(|e| target_ents.contains(e.as_str())).collect();
        let Some(first_shared) = shared.first() else {
            continue;
        };
        let base = squash_strength(1.0) * lag_decay(days);
        if base < EVIDENCE_FLOOR * squash_strength(1.0) {
            // Same spirit as the C1 floor: a hopeless-lag edge contributes
            // nothing, so it can never launder a bonus into the ranking.
            continue;
        }
        let n_ents = entities_of.get(&node.id).map_or(0, Vec::len);
        let class = classify(node, first_shared, n_ents, first_touch);
        out.push(Candidate {
            id: node.id.clone(),
            class,
            base,
        });
    }
    out
}

pub fn run(storage: &Storage, failure_ids: &[String]) -> Result<ArmRun> {
    let all = load_all_nodes(storage)?;
    let acc_t0 = Instant::now();
    let entities_of: HashMap<String, Vec<String>> = all
        .iter()
        .map(|n| (n.id.clone(), node_entities(n)))
        .collect();
    let first_touch = first_touch_map(&all, &entities_of);
    let by_id: HashMap<&str, &KnowledgeNode> = all.iter().map(|n| (n.id.as_str(), n)).collect();
    let accumulation_ms = acc_t0.elapsed().as_secs_f64() * 1000.0;

    let mut queries = Vec::with_capacity(failure_ids.len());
    for failure_id in failure_ids {
        let t0 = Instant::now();
        let mut scores: HashMap<String, f64> = HashMap::new();
        if let Some(failure) = by_id.get(failure_id.as_str()) {
            let d1 = backward_candidates(failure, &all, &entities_of, &first_touch);
            for c in &d1 {
                let s = c.class.bonus() + c.base;
                scores
                    .entry(c.id.clone())
                    .and_modify(|g| *g = g.max(s))
                    .or_insert(s);
                // Depth 2 only through a mutation: a mention cannot transmit
                // causality it never exerted.
                if c.class != EvidenceClass::Mutation {
                    continue;
                }
                let Some(mid) = by_id.get(c.id.as_str()) else {
                    continue;
                };
                for r in backward_candidates(mid, &all, &entities_of, &first_touch) {
                    if r.class == EvidenceClass::Mention || r.id == *failure_id {
                        continue;
                    }
                    let s2 = r.class.bonus() * HOP_CREDIT + r.base * c.base;
                    scores
                        .entry(r.id.clone())
                        .and_modify(|g| *g = g.max(s2))
                        .or_insert(s2);
                }
            }
        }
        scores.remove(failure_id);
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(TOP_K);
        queries.push(QueryResult {
            failure_id: failure_id.clone(),
            ranked_ids: ranked.iter().map(|(id, _)| id.clone()).collect(),
            scores: ranked.iter().map(|(_, s)| *s).collect(),
            wall_clock_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    Ok(ArmRun {
        queries,
        search_mode: "causal_graph_v2_evidence_typed".into(),
        accumulation_ms: Some(accumulation_ms),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_adjacency_hits_operators_only() {
        assert!(assignment_adjacent("Set API_TIMEOUT=2 in the env", "API_TIMEOUT"));
        assert!(assignment_adjacent("ORT_PIN: 1.19.2 pinned", "ORT_PIN"));
        assert!(assignment_adjacent("bumped usearch@2.16 today", "usearch"));
        assert!(assignment_adjacent("Set API_TIMEOUT = 2", "API_TIMEOUT"));
        // The seeder's adversarial change-verb chatter stays a mention.
        assert!(!assignment_adjacent(
            "Rotated the comment on API_TIMEOUT during the quarterly catalog review",
            "API_TIMEOUT"
        ));
        assert!(!assignment_adjacent("Audited API_TIMEOUT during review", "API_TIMEOUT"));
        // No partial-identifier matches.
        assert!(!assignment_adjacent("MY_API_TIMEOUT=9", "API_TIMEOUT"));
    }

    #[test]
    fn mention_never_outranks_mutation_regardless_of_recency() {
        // Mention at the best possible lag vs mutation at the worst admissible
        // lag: bonus must dominate.
        let mention_best = EvidenceClass::Mention.bonus() + squash_strength(1.0) * lag_decay(1.0);
        let mutation_worst =
            EvidenceClass::Mutation.bonus() + squash_strength(1.0) * lag_decay(LOOKBACK_DAYS as f64);
        assert!(mutation_worst > mention_best);
        let origin_worst =
            EvidenceClass::Origin.bonus() + squash_strength(1.0) * lag_decay(LOOKBACK_DAYS as f64);
        assert!(origin_worst > mention_best);
        assert!(mutation_worst > origin_worst + squash_strength(1.0));
    }
}
