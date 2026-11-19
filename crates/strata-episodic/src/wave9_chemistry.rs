//! Wave 9 (#631–#640): Chemistry-inspired reaction, buffering, chelation & gating operators.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #631: Le Chatelier user-controlled grounding recall under stress.
pub fn le_chatelier_grounding_recall<'a>(
    stress_mode_enabled: bool,
    grounding_anchors: &'a [String],
    normal_candidates: &'a [String],
) -> &'a [String] {
    if stress_mode_enabled && !grounding_anchors.is_empty() {
        grounding_anchors
    } else {
        normal_candidates
    }
}

/// #632: Sensitive-memory activation-energy gate (`E >= E_a` or explicit owner handle).
pub fn activation_energy_gate(
    is_sensitive: bool,
    explicit_handle_requested: bool,
    context_energy: u32,
    threshold_ea: u32,
) -> bool {
    if !is_sensitive {
        return true;
    }
    explicit_handle_requested || context_energy >= threshold_ea
}

/// #633: Limiting-reagent all-required-participants gate.
pub fn limiting_reagent_gate(required: &BTreeSet<String>, signed: &BTreeSet<String>) -> Result<(), Vec<String>> {
    let missing: Vec<String> = required.difference(signed).cloned().collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}

/// #634: Directed chiral role bindings `(subject, predicate, object) != (object, predicate, subject)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChiralRoleBinding {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl ChiralRoleBinding {
    pub fn enantiomer(&self) -> Self {
        Self {
            subject: self.object.clone(),
            predicate: self.predicate.clone(),
            object: self.subject.clone(),
        }
    }
}

/// #635: Buffered response to emotional hyperbole (clamps single-turn salience spike).
pub fn buffer_hyperbole_salience(base_milli: u64, raw_spike_milli: u64, max_step_milli: u64) -> u64 {
    base_milli + raw_spike_milli.min(max_step_milli)
}

/// #636: Reversible dependency quarantine (chelation).
#[derive(Debug, Default)]
pub struct ChelationQuarantine {
    chelated: BTreeSet<String>,
}

impl ChelationQuarantine {
    pub fn chelate(&mut self, mem_id: impl Into<String>) {
        self.chelated.insert(mem_id.into());
    }

    pub fn dechelate(&mut self, mem_id: &str) -> bool {
        self.chelated.remove(mem_id)
    }

    pub fn is_active(&self, mem_id: &str) -> bool {
        !self.chelated.contains(mem_id)
    }
}

/// #637: Recipient-scoped constraint routing (allosteric inhibition).
pub fn allosteric_recipient_filter(
    memories: &[(String, BTreeSet<String>)], // (mem_id, allowed_recipients)
    recipient: &str,
) -> Vec<String> {
    memories
        .iter()
        .filter(|(_, allowed)| allowed.contains("*") || allowed.contains(recipient))
        .map(|(id, _)| id.clone())
        .collect()
}

/// #638: Supersaturation repeated-intention escalation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationLevel {
    Normal,
    Elevated,
    CrystallizedUrgent,
}

pub fn supersaturation_escalation(deferral_count: u32, saturation_limit: u32) -> EscalationLevel {
    if deferral_count >= saturation_limit {
        EscalationLevel::CrystallizedUrgent
    } else if deferral_count * 2 >= saturation_limit {
        EscalationLevel::Elevated
    } else {
        EscalationLevel::Normal
    }
}

/// #639: Ephemeral episode links (weak dispersion links expiring outside `ttl_seq`).
pub fn active_ephemeral_links(
    links: &[(String, String, u64)], // (from, to, created_seq)
    current_seq: u64,
    ttl_seq: u64,
) -> Vec<(String, String)> {
    links
        .iter()
        .filter(|(_, _, seq)| current_seq.saturating_sub(*seq) <= ttl_seq)
        .map(|(a, b, _)| (a.clone(), b.clone()))
        .collect()
}

/// #640: Evidence-threshold titration transition receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitrationReceipt {
    pub hypothesis_id: String,
    pub witness_receipts: Vec<String>,
    pub promoted: bool,
    pub receipt_blake3: String,
}

pub fn evaluate_titration_threshold(
    hypothesis_id: &str,
    witnesses: &[String],
    equivalence_threshold: usize,
) -> TitrationReceipt {
    let unique: BTreeMap<&String, ()> = witnesses.iter().map(|w| (w, ())).collect();
    let list: Vec<String> = unique.keys().map(|k| (*k).clone()).collect();
    let promoted = list.len() >= equivalence_threshold;
    let mut h = blake3::Hasher::new();
    h.update(hypothesis_id.as_bytes());
    h.update(&[u8::from(promoted)]);
    for w in &list {
        h.update(w.as_bytes());
    }
    TitrationReceipt {
        hypothesis_id: hypothesis_id.to_string(),
        witness_receipts: list,
        promoted,
        receipt_blake3: h.finalize().to_hex().to_string(),
    }
}
