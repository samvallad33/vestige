//! Wave 8 (#621–#630): Physics-inspired source-preserving state operators.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #621: Ferromagnetic hysteresis salience update (remanence prevents single-dip erasure).
pub fn hysteresis_salience_step(current_milli: i64, delta_milli: i64, remanence_floor: i64) -> i64 {
    let raw = (current_milli + delta_milli).clamp(0, 1000);
    if current_milli >= 800 && delta_milli < 0 {
        raw.max(remanence_floor)
    } else {
        raw
    }
}

/// #622: Schmitt-trigger dead-band transition (`low_thresh < high_thresh`).
pub fn schmitt_deadband_state(currently_active: bool, signal_milli: i64, low: i64, high: i64) -> bool {
    if currently_active {
        signal_milli >= low
    } else {
        signal_milli >= high
    }
}

/// #623: Destructive-interference packet admission (conflicting polarities hold both).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacketAdmission {
    AdmitSingle(String),
    HoldConflict { pos_id: String, neg_id: String },
}

pub fn destructive_interference_admit(slot_claims: &[(String, bool)]) -> Option<PacketAdmission> {
    let pos = slot_claims.iter().find(|(_, p)| *p).map(|(id, _)| id.clone());
    let neg = slot_claims.iter().find(|(_, p)| !*p).map(|(id, _)| id.clone());
    match (pos, neg) {
        (Some(p), Some(n)) => Some(PacketAdmission::HoldConflict { pos_id: p, neg_id: n }),
        (Some(p), None) => Some(PacketAdmission::AdmitSingle(p)),
        (None, Some(n)) => Some(PacketAdmission::AdmitSingle(n)),
        (None, None) => None,
    }
}

/// #624: Constructive-resonance receipt-backed recurrence accumulator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResonanceEvidence {
    pub memory_id: String,
    pub distinct_receipts: BTreeSet<String>,
    pub resonance_score_milli: u64,
}

pub fn constructive_resonance_recurrence(memory_id: &str, receipts: &[String]) -> ResonanceEvidence {
    let set: BTreeSet<String> = receipts.iter().cloned().collect();
    let score = (set.len() as u64 * 250).min(1000);
    ResonanceEvidence {
        memory_id: memory_id.to_string(),
        distinct_receipts: set,
        resonance_score_milli: score,
    }
}

/// #625: Exclusive current-value slots (Pauli exclusion: one live occupant per `(entity, attr)`).
#[derive(Debug, Default)]
pub struct ExclusiveSlotRegister {
    live: BTreeMap<(String, String), String>,
    retired: Vec<(String, String)>, // (old_mem_id, superseded_by_mem_id)
}

impl ExclusiveSlotRegister {
    pub fn assign(&mut self, entity: &str, attr: &str, new_mem_id: &str) -> Option<String> {
        let prev = self
            .live
            .insert((entity.to_string(), attr.to_string()), new_mem_id.to_string());
        if let Some(old) = &prev {
            self.retired.push((old.clone(), new_mem_id.to_string()));
        }
        prev
    }

    pub fn current(&self, entity: &str, attr: &str) -> Option<&str> {
        self.live
            .get(&(entity.to_string(), attr.to_string()))
            .map(String::as_str)
    }
}

/// #626: Cherenkov anomalous mutation-burst detector.
pub fn cherenkov_mutation_burst(timestamps_ms: &[u64], window_ms: u64, max_normal_rate: usize) -> bool {
    if timestamps_ms.len() <= max_normal_rate {
        return false;
    }
    let mut sorted = timestamps_ms.to_vec();
    sorted.sort_unstable();
    let mut left = 0usize;
    for right in 0..sorted.len() {
        while sorted[right].saturating_sub(sorted[left]) > window_ms {
            left += 1;
        }
        if right - left + 1 > max_normal_rate {
            return true;
        }
    }
    false
}

/// #627: Traceable episodic-to-semantic distillation (decay chains).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecayChainLink {
    pub parent_ids: Vec<String>,
    pub distilled_id: String,
    pub chain_blake3: String,
}

pub fn episodic_decay_chain(parent_ids: Vec<String>, distilled_id: impl Into<String>) -> DecayChainLink {
    let distilled_id = distilled_id.into();
    let mut h = blake3::Hasher::new();
    for p in &parent_ids {
        h.update(p.as_bytes());
        h.update(b"->");
    }
    h.update(distilled_id.as_bytes());
    DecayChainLink {
        parent_ids,
        distilled_id,
        chain_blake3: h.finalize().to_hex().to_string(),
    }
}

/// #628: Milestone-driven episode nucleation clustering.
pub fn nucleate_episode_clusters(
    milestones: &[String],
    edges: &[(String, String)],
) -> BTreeMap<String, BTreeSet<String>> {
    let milestone_set: BTreeSet<&String> = milestones.iter().collect();
    let mut clusters: BTreeMap<String, BTreeSet<String>> = milestones
        .iter()
        .map(|m| (m.clone(), BTreeSet::from([m.clone()])))
        .collect();
    for (a, b) in edges {
        if milestone_set.contains(a) {
            clusters.entry(a.clone()).or_default().insert(b.clone());
        }
        if milestone_set.contains(b) {
            clusters.entry(b.clone()).or_default().insert(a.clone());
        }
    }
    clusters
}

/// #629: Temporary (elastic) vs durable (plastic) context changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextDeformation {
    ElasticTemporary { baseline: String, active_until_ms: u64 },
    PlasticDurable { new_baseline: String },
}

pub fn resolve_elastic_or_plastic(state: &ContextDeformation, now_ms: u64, temp_val: &str) -> String {
    match state {
        ContextDeformation::ElasticTemporary {
            baseline,
            active_until_ms,
        } => {
            if now_ms <= *active_until_ms {
                temp_val.to_string()
            } else {
                baseline.clone()
            }
        }
        ContextDeformation::PlasticDurable { new_baseline } => new_baseline.clone(),
    }
}

/// #630: Light-cone causal-horizon validation (`cause_seq < effect_seq`).
pub fn validate_causal_light_cone(cause_seq: u64, effect_seq: u64, reachable_in_dag: bool) -> bool {
    cause_seq < effect_seq && reachable_in_dag
}
