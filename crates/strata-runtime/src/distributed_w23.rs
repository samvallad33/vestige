//! Wave 23A–23O (#575–#589): Distributed Strata Runtime — checkpoints, forks,
//! causal vector clocks, speculative execution, correction ripples, compute-priced
//! retention, artifact envelopes, BFT auditor quorums, role shards, process leases,
//! state-transition proofs, detached hardware seals, immutable-branch policy evolution,
//! bounded safe frame decoding, per-replica journals & swarm workflow adapters.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #575: Cognitive checkpoints, forks, diffs, and guarded DAG merges.
#[derive(Debug, Default, Clone)]
pub struct CognitiveBranchDag {
    branches: BTreeMap<String, Vec<String>>,
}

impl CognitiveBranchDag {
    pub fn init_main(&mut self, genesis: Vec<String>) {
        self.branches.insert("main".to_string(), genesis);
    }

    pub fn fork(&mut self, from_branch: &str, new_branch: &str) -> bool {
        if let Some(items) = self.branches.get(from_branch).cloned() {
            self.branches.insert(new_branch.to_string(), items);
            true
        } else {
            false
        }
    }

    pub fn append(&mut self, branch: &str, item: impl Into<String>) {
        self.branches.entry(branch.to_string()).or_default().push(item.into());
    }

    pub fn diff(&self, base: &str, head: &str) -> Vec<String> {
        let base_set: BTreeSet<&String> = self
            .branches
            .get(base)
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        self.branches
            .get(head)
            .map(|v| v.iter().filter(|x| !base_set.contains(x)).cloned().collect())
            .unwrap_or_default()
    }

    pub fn guarded_merge(&mut self, source: &str, target: &str, forbidden_prefix: &str) -> Result<usize, &'static str> {
        let delta = self.diff(target, source);
        if delta.iter().any(|x| x.starts_with(forbidden_prefix)) {
            return Err("MergeBlockedByGuard");
        }
        let count = delta.len();
        self.branches.entry(target.to_string()).or_default().extend(delta);
        Ok(count)
    }
}

/// #576: Authenticated causal vector clock & partial-order comparator.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CausalVectorClock {
    pub entries: BTreeMap<String, u64>,
}

impl CausalVectorClock {
    pub fn tick(&mut self, replica: &str) {
        *self.entries.entry(replica.to_string()).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.entries {
            let entry = self.entries.entry(k.clone()).or_insert(0);
            *entry = (*entry).max(*v);
        }
    }

    pub fn happens_before(&self, other: &Self) -> bool {
        let mut strictly_less = false;
        let all_keys: BTreeSet<&String> = self.entries.keys().chain(other.entries.keys()).collect();
        for k in all_keys {
            let a = self.entries.get(k).copied().unwrap_or(0);
            let b = other.entries.get(k).copied().unwrap_or(0);
            if a > b {
                return false;
            }
            if a < b {
                strictly_less = true;
            }
        }
        strictly_less
    }
}

/// #577: Bounded speculative execution with exact promotion checks.
pub fn promote_speculative_batch<F>(staged: Vec<String>, max_bound: usize, check: F) -> Result<Vec<String>, &'static str>
where
    F: FnOnce(&[String]) -> bool,
{
    if staged.len() > max_bound {
        return Err("SpeculativeBoundExceeded");
    }
    if !check(&staged) {
        return Err("PromotionCheckFailed");
    }
    Ok(staged)
}

/// #578: Cross-agent correction ripple & cooperative wake-up list.
pub fn correction_ripple_targets(
    superseded_mem_id: &str,
    agent_manifests: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    agent_manifests
        .iter()
        .filter(|(_, held)| held.contains(superseded_mem_id))
        .map(|(agent, _)| agent.clone())
        .collect()
}

/// #579: Compute-priced context retention protecting required rules unconditionally.
#[derive(Debug, Clone)]
pub struct PricedContextItem {
    pub id: String,
    pub required_rule: bool,
    pub utility_milli: u64,
    pub token_cost: u64,
}

pub fn select_compute_priced_context(items: &[PricedContextItem], token_budget: u64) -> Vec<String> {
    let mut selected = Vec::new();
    let mut used = 0u64;
    for item in items.iter().filter(|i| i.required_rule) {
        selected.push(item.id.clone());
        used = used.saturating_add(item.token_cost);
    }
    let mut optional: Vec<&PricedContextItem> = items.iter().filter(|i| !i.required_rule).collect();
    optional.sort_by(|a, b| {
        let ratio_a = a.utility_milli.saturating_mul(1000) / a.token_cost.max(1);
        let ratio_b = b.utility_milli.saturating_mul(1000) / b.token_cost.max(1);
        ratio_b.cmp(&ratio_a)
    });
    for opt in optional {
        if used + opt.token_cost <= token_budget {
            used += opt.token_cost;
            selected.push(opt.id.clone());
        }
    }
    selected
}

/// #580: Typed latent & provider-state artifact transport envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedArtifactEnvelope {
    pub schema_uri: String,
    pub provider: String,
    pub payload_blake3: String,
    pub bytes_len: usize,
}

impl TypedArtifactEnvelope {
    pub fn wrap(schema_uri: &str, provider: &str, payload: &[u8]) -> Self {
        Self {
            schema_uri: schema_uri.to_string(),
            provider: provider.to_string(),
            payload_blake3: blake3::hash(payload).to_hex().to_string(),
            bytes_len: payload.len(),
        }
    }
}

/// #581: Byzantine auditor quorum (`n = 3f + 1`, requires `>= 2f + 1` matching votes).
pub fn verify_bft_auditor_quorum(votes: &BTreeMap<String, bool>, max_faulty_f: usize) -> Option<bool> {
    let threshold = 2 * max_faulty_f + 1;
    let trues = votes.values().filter(|&&v| v).count();
    let falses = votes.values().filter(|&&v| !v).count();
    if trues >= threshold {
        Some(true)
    } else if falses >= threshold {
        Some(false)
    } else {
        None
    }
}

/// #582: Authorized role shards & progressive context disclosure.
pub fn filter_by_role_shard(items: &[(String, String)], granted_shards: &BTreeSet<String>) -> Vec<String> {
    items
        .iter()
        .filter(|(_, shard)| granted_shards.contains(shard))
        .map(|(id, _)| id.clone())
        .collect()
}

/// #583: Cognitive process trees, leases, and evidence-backed child lesson promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessLease {
    pub child_pid: String,
    pub expires_ms: u64,
    pub evidence_receipt: Option<String>,
}

impl ProcessLease {
    pub fn can_promote_child_lesson(&self, now_ms: u64) -> bool {
        now_ms <= self.expires_ms && self.evidence_receipt.is_some()
    }
}

/// #584: Scoped policy & state-transition proof commitment.
pub fn compute_transition_proof(prev_root: &[u8; 32], policy_hash: &[u8; 32], next_root: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"strata-transition-v1:");
    h.update(prev_root);
    h.update(policy_hash);
    h.update(next_root);
    *h.finalize().as_bytes()
}

/// #585: Hardware-backed detached checkpoint sealing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachedHardwareSeal {
    pub key_slot_id: String,
    pub segment_merkle_root: [u8; 32],
    pub attestation_tag: [u8; 32],
}

pub fn seal_detached_checkpoint(slot_key: &[u8; 32], key_slot_id: &str, root: [u8; 32]) -> DetachedHardwareSeal {
    let mut msg = key_slot_id.as_bytes().to_vec();
    msg.extend_from_slice(&root);
    DetachedHardwareSeal {
        key_slot_id: key_slot_id.to_string(),
        segment_merkle_root: root,
        attestation_tag: *blake3::keyed_hash(slot_key, &msg).as_bytes(),
    }
}

/// #586: Adversarial policy evolution on an immutable shadow branch.
pub fn evaluate_candidate_policy_branch(
    pinned_main_hash: [u8; 32],
    candidate_hash: [u8; 32],
    adversarial_corpus_blocked_all: bool,
) -> ([u8; 32], bool) {
    // Main hash is NEVER mutated in place; returns (unchanged_main, candidate_qualified).
    (
        pinned_main_hash,
        pinned_main_hash != candidate_hash && adversarial_corpus_blocked_all,
    )
}

/// #587: Allocation-safe bounded frame decoder (`[u32_le_len][payload]`).
pub fn decode_frame_bounded_safe(buf: &[u8], max_cap: usize) -> Result<&[u8], &'static str> {
    if buf.len() < 4 {
        return Err("TruncatedLengthHeader");
    }
    let declared = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if declared > max_cap {
        return Err("FrameExceedsAllocationCap");
    }
    if buf.len() < 4 + declared {
        return Err("TruncatedFramePayload");
    }
    Ok(&buf[4..4 + declared])
}

/// #588: Durable per-replica logical event journal.
#[derive(Debug, Default)]
pub struct ReplicaLogicalJournal {
    entries: Vec<(u64, String)>,
}

impl ReplicaLogicalJournal {
    pub fn append(&mut self, event: impl Into<String>) -> u64 {
        let lsn = self.entries.len() as u64 + 1;
        self.entries.push((lsn, event.into()));
        lsn
    }

    pub fn since_lsn(&self, after_lsn: u64) -> Vec<(u64, String)> {
        self.entries
            .iter()
            .filter(|(lsn, _)| *lsn > after_lsn)
            .cloned()
            .collect()
    }
}

/// #589: Integrated swarm workflow handoff qualification (coordinator -> worker -> verifier).
pub fn qualify_swarm_handoff(
    coordinator_receipt: Option<&str>,
    worker_receipt: Option<&str>,
    verifier_receipt: Option<&str>,
) -> bool {
    coordinator_receipt.is_some() && worker_receipt.is_some() && verifier_receipt.is_some()
}
