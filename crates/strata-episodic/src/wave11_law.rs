//! Wave 11 (#651–#660): Legal & evidentiary operators (stare decisis, adverse
//! possession, chain of custody, poisonous tree, res ipsa, hearsay, laches,
//! force majeure, mens rea / actus reus, race-notice).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

/// #651: Owner-rule precedence hierarchy (`StareDecisis`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrecedenceCourt {
    AgentObservation = 1,
    SessionPreference = 2,
    ProjectRule = 3,
    OwnerConstitutional = 4,
}

pub fn resolve_stare_decisis<T>(candidates: &[(PrecedenceCourt, T)]) -> Option<&T> {
    candidates.iter().max_by_key(|(court, _)| *court).map(|(_, v)| v)
}

/// #652: Observed routine proposals with mandatory owner confirmation (`AdversePossession`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutineProposal {
    pub pattern_id: String,
    pub observed_count: u32,
    pub owner_confirmed: bool,
}

impl RoutineProposal {
    pub fn is_binding_rule(&self, min_observations: u32) -> bool {
        self.observed_count >= min_observations && self.owner_confirmed
    }
}

/// #653: Packet provenance chain of custody.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustodyHop {
    pub stage: String,
    pub actor: String,
    pub prev_blake3: [u8; 32],
    pub hop_blake3: [u8; 32],
}

pub fn append_custody_hop(chain: &mut Vec<CustodyHop>, stage: &str, actor: &str) {
    let prev = chain.last().map_or([0u8; 32], |h| h.hop_blake3);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&prev);
    hasher.update(stage.as_bytes());
    hasher.update(actor.as_bytes());
    let hop_blake3 = *hasher.finalize().as_bytes();
    chain.push(CustodyHop {
        stage: stage.to_string(),
        actor: actor.to_string(),
        prev_blake3: prev,
        hop_blake3,
    });
}

/// #654: Retraction-aware dependency invalidation (`Fruit of the Poisonous Tree`),
/// respecting the Independent Source exception!
pub fn poisonous_tree_invalidate(
    retracted_root: &str,
    derived_edges: &[(String, String)], // (premise -> conclusion)
    independent_clean_roots: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut tainted = BTreeSet::new();
    let mut queue = VecDeque::new();
    tainted.insert(retracted_root.to_string());
    queue.push_back(retracted_root.to_string());

    while let Some(curr) = queue.pop_front() {
        for (prem, concl) in derived_edges {
            if prem == &curr && !tainted.contains(concl) && !independent_clean_roots.contains(concl)
            {
                tainted.insert(concl.clone());
                queue.push_back(concl.clone());
            }
        }
    }
    tainted
}

/// #655: Temporal hazard hypotheses with explicit confirmation (`ResIpsaLoquitur`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardHypothesis {
    pub action_sig: String,
    pub failure_sig: String,
    pub co_occurrences: u32,
    pub confirmed: bool,
}

/// #656: Source attribution and urgency separation (`Hearsay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestimonyKind {
    FirsthandReceipted,
    SecondhandHearsay,
}

pub fn hearsay_admissible_for_mutation(kind: TestimonyKind, _urgent_claimed: bool) -> bool {
    kind == TestimonyKind::FirsthandReceipted
}

/// #657: Age- and relevance-aware interruption restraint (`Laches`).
pub fn barred_by_laches(created_ms: u64, now_ms: u64, max_actionable_delay_ms: u64, high_severity: bool) -> bool {
    !high_severity && now_ms.saturating_sub(created_ms) > max_actionable_delay_ms
}

/// #658: User-declared crisis pause intervals (`ForceMajeure`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForceMajeureInterval {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl ForceMajeureInterval {
    pub fn suspends_routine_nudges(&self, now_ms: u64) -> bool {
        now_ms >= self.start_ms && now_ms <= self.end_ms
    }
}

/// #659: Separate intention (`MensRea`), attempt, and outcome (`ActusReus`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionTriad {
    pub intention_id: String,
    pub attempt_receipt_id: Option<String>,
    pub outcome_receipt_id: Option<String>,
}

/// #660: Conflict-preserving offline synchronization priority (`RaceNotice`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncPriorityOutcome {
    Winner(String),
    ConcurrentSplitPreserved(Vec<String>),
}

pub fn resolve_race_notice(
    branches: &BTreeMap<String, (bool, u64)>, // branch_id -> (had_clean_notice_of_peer, recorded_seq)
) -> SyncPriorityOutcome {
    let clean: Vec<(&String, u64)> = branches
        .iter()
        .filter(|(_, (had_notice, _))| !*had_notice)
        .map(|(id, (_, seq))| (id, *seq))
        .collect();
    if clean.len() == 1 {
        return SyncPriorityOutcome::Winner(clean[0].0.clone());
    }
    let min_seq = clean.iter().map(|(_, s)| *s).min();
    let tied: Vec<String> = clean
        .iter()
        .filter(|(_, s)| Some(*s) == min_seq)
        .map(|(id, _)| (*id).clone())
        .collect();
    if tied.len() == 1 {
        SyncPriorityOutcome::Winner(tied[0].clone())
    } else {
        SyncPriorityOutcome::ConcurrentSplitPreserved(branches.keys().cloned().collect())
    }
}
