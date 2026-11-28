//! Deterministic merge fold across per-writer chains.
//!
//! Each device keeps one local signing log and a replica per granted writer.
//! A replica stores the writer's verifying key in `writer.pub` and refuses to
//! open if `strata.key` is present. The fold reads recorded frames only.
//! `state_digest` does not include this state, so a peerless store stays
//! byte-identical.
//!
//! Issue #401's signed tree head, #302's claim slot, #402's wrapped data key,
//! and #384's scope set are not in this tree. This module adds the minimum
//! each acceptance criterion needs: a signed head over the existing merkle
//! root, an `AssertClaim` op, a local data-key file, and an empty scope list.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use borsh::{BorshDeserialize, BorshSerialize};
use ed25519_dalek::VerifyingKey;
use strata::{
    consistency_proof, merkle_root, verify_consistency, verify_head, FrameRecord, ReplicaLog,
    SignedHead,
};
use strata_gate::log::EventLog;
use strata_gate::record::{GateEvent, GateRecord, RecordKind, Verdict};
use strata_gate::{rederive_verdicts, sweep, Policy};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::ALGO_V2;
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;

use super::{
    classify_write_payload, handle_of, hash32, mesh_grant_params, AdmissionContext, StoreError,
    StrataStore, WritePayload, INGEST_RATING,
};
use crate::canonical::canonical_hash;
use crate::card::CardEvent;
use crate::op::{is_mesh_receipt, ClaimRef, FrameFail, StoreOp};

const MANIFEST: &str = "chain.manifest";
const KEY_FILE: &str = "strata.key";
const ACTIVE_WRITER: &str = "mesh/active-writer";
const DATA_KEY_DOMAIN: &[u8] = b"vestige.mesh.data-key.v1\0";
const B58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const ED25519_MULTICODEC_PREFIX: [u8; 2] = [0xED, 0x01];

/// Log directory `open` uses. Absent marker means `<dir>/log`.
pub(crate) fn active_log_dir(dir: &Path) -> PathBuf {
    let marker = dir.join(ACTIVE_WRITER);
    let Ok(text) = fs::read_to_string(&marker) else {
        return dir.join(super::LOG_DIR);
    };
    let rel = text.trim();
    if rel.is_empty() || rel.starts_with('/') || rel.contains("..") || rel.contains('\\') {
        return dir.join(super::LOG_DIR);
    }
    dir.join(rel)
}

/// did:key of a log verifying key. Same multicodec and base58btc as
/// `vestige_core::did_key_from_ed25519_public_key`. The store does not depend
/// on vestige-core.
pub fn writer_did(verifying_key: &[u8; 32]) -> String {
    let mut payload = [0_u8; 34];
    payload[..2].copy_from_slice(&ED25519_MULTICODEC_PREFIX);
    payload[2..].copy_from_slice(verifying_key);
    format!("did:key:z{}", base58btc_encode(&payload))
}

/// blake3 hex of the verifying key. Matches strata-verify's fingerprint.
pub fn fingerprint(verifying_key: &[u8; 32]) -> String {
    blake3::hash(verifying_key).to_hex().to_string()
}

/// Canonical handle of a record that lives on one writer's chain.
pub fn canonical_handle(writer_did: &str, id: &str) -> String {
    format!("{writer_did}/{id}")
}

/// Human-confirmed mesh grant. The CLI sets this. An MCP caller does not.
pub fn mesh_grant_context() -> AdmissionContext {
    AdmissionContext {
        rule_id: Some(super::RULE_MESH_GRANT.to_string()),
        confirm: false,
        human_confirmed: true,
    }
}

fn base58btc_encode(bytes: &[u8]) -> String {
    let zeros = bytes.iter().take_while(|&&b| b == 0).count();
    let mut digits: Vec<u8> = Vec::with_capacity(bytes.len() * 138 / 100 + 1);
    for &byte in &bytes[zeros..] {
        let mut carry = byte as u32;
        for digit in digits.iter_mut() {
            carry += (*digit as u32) * 256;
            *digit = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let mut out = String::with_capacity(zeros + digits.len());
    out.extend(std::iter::repeat_n('1', zeros));
    for digit in digits.iter().rev() {
        out.push(B58_ALPHABET[*digit as usize] as char);
    }
    out
}

/// What a device has admitted about other writers. Rebuilt by replay.
#[derive(Debug, Default)]
pub(crate) struct MeshSide {
    grants: BTreeMap<String, PinnedGrant>,
    merges: Vec<MergeNote>,
    conflicts: Vec<ConflictNote>,
    quarantined: BTreeSet<String>,
    tombstones: Vec<String>,
    forks: Vec<ForkNote>,
    policy_mismatches: Vec<([u8; 32], [u8; 32])>,
    denies: Vec<(String, Vec<FrameFail>)>,
    index_mismatches: Vec<(String, u64, String)>,
}

#[derive(Debug, Clone)]
struct PinnedGrant {
    verifying_key: [u8; 32],
    policy_hash: [u8; 32],
    scopes: Vec<String>,
    endpoint_binding: Vec<u8>,
}

#[derive(Debug, Clone)]
struct MergeNote {
    writer_did: String,
    from_seq: u64,
    to_seq: u64,
    effect_seq: u64,
    signed_head: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ConflictNote {
    subject_id: String,
    field_id: String,
    left: ClaimRef,
    right: ClaimRef,
    effect_seq: u64,
}

#[derive(Debug, Clone)]
struct ForkNote {
    derived_from_did: String,
    derived_from_seq: u64,
    derived_from_frame: [u8; 32],
}

impl MeshSide {
    pub(crate) fn observe(&mut self, dir: &Path, op: &StoreOp) {
        match op {
            StoreOp::GrantWriter {
                writer_did,
                verifying_key,
                policy_hash,
                scopes,
                endpoint_binding,
            } => {
                self.grants.insert(
                    writer_did.clone(),
                    PinnedGrant {
                        verifying_key: *verifying_key,
                        policy_hash: *policy_hash,
                        scopes: scopes.clone(),
                        endpoint_binding: endpoint_binding.clone(),
                    },
                );
            }
            StoreOp::MergeChain {
                writer_did,
                from_seq,
                to_seq,
                signed_head,
                ..
            } => {
                self.merges.push(MergeNote {
                    writer_did: writer_did.clone(),
                    from_seq: *from_seq,
                    to_seq: *to_seq,
                    effect_seq: 0,
                    signed_head: signed_head.clone(),
                });
            }
            StoreOp::ConflictRecord {
                subject_id,
                field_id,
                left,
                right,
            } => {
                self.conflicts.push(ConflictNote {
                    subject_id: subject_id.clone(),
                    field_id: field_id.clone(),
                    left: left.clone(),
                    right: right.clone(),
                    effect_seq: 0,
                });
            }
            StoreOp::Equivocation { writer_did, .. } => {
                self.quarantined.insert(writer_did.clone());
            }
            StoreOp::PolicyMismatch {
                local_hash,
                peer_hash,
            } => {
                self.policy_mismatches.push((*local_hash, *peer_hash));
            }
            StoreOp::IndexMismatch {
                sender_did,
                segment,
                handle,
            } => {
                self.index_mismatches
                    .push((sender_did.clone(), *segment, handle.clone()));
            }
            StoreOp::DenyRange {
                writer_did,
                failures,
            } => {
                self.denies.push((writer_did.clone(), failures.clone()));
            }
            StoreOp::ForkGenesis {
                derived_from_did,
                derived_from_seq,
                derived_from_frame,
            } => {
                self.forks.push(ForkNote {
                    derived_from_did: derived_from_did.clone(),
                    derived_from_seq: *derived_from_seq,
                    derived_from_frame: *derived_from_frame,
                });
            }
            StoreOp::TombstoneReceipt { memory_id, .. } | StoreOp::PurgeTombstone { memory_id } => {
                destroy_data_key(dir, memory_id);
                self.tombstones.push(memory_id.clone());
            }
            _ => {}
        }
    }
}

/// One open claim-slot contradiction. Both claims stay in the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenConflict {
    /// Subject the claims share.
    pub subject_id: String,
    /// Field the claims share.
    pub field_id: String,
    /// The side with the lower frame hash.
    pub left: ClaimRef,
    /// The side with the higher frame hash.
    pub right: ClaimRef,
}

/// A later write of the same canonical content, kept as a reinforcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reinforcement {
    /// Qualified id of the node that was kept.
    pub kept: String,
    /// Writer of the later effect.
    pub later_writer: String,
    /// Data-frame seq of the later effect.
    pub later_seq: u64,
    /// Frame hash of the earlier effect.
    pub earlier_frame: [u8; 32],
    /// Frame hash of the later effect.
    pub later_frame: [u8; 32],
}

/// The merged projection. `digest` excludes frontiers and local receipts.
#[derive(Debug, Clone)]
pub struct MeshProjection {
    /// blake3 of the canonical projection.
    pub digest: [u8; 32],
    /// Writer did -> last included frame seq.
    pub frontier: BTreeMap<String, u64>,
    /// Qualified id -> canonical content hash, for live nodes.
    pub nodes: BTreeMap<String, [u8; 32]>,
    /// Reinforcements, in digest order.
    pub reinforcements: Vec<Reinforcement>,
    /// Open conflicts, in digest order.
    pub conflicts: Vec<OpenConflict>,
    /// Review count per merged card handle.
    pub review_counts: BTreeMap<u64, u32>,
}

/// Public half of an admitted writer grant. Never a signing seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedWriter {
    /// 32-byte ed25519 verifying key.
    pub verifying_key: [u8; 32],
    /// Policy hash the grant pinned.
    pub policy_hash: [u8; 32],
    /// Scope names. Empty means the whole store.
    pub scopes: Vec<String>,
    /// Writer binding to a transport endpoint, when pairing recorded one.
    pub endpoint_binding: Vec<u8>,
}

/// Result of importing one chain directory.
#[derive(Debug, Clone)]
pub struct ImportReport {
    /// Effect seq of the merge, denial, or mismatch receipt, when one was written.
    pub receipt_seq: Option<u64>,
    /// Mesh digest after the import.
    pub mesh_digest: [u8; 32],
    /// The new frames were receipts, so no merge receipt was written.
    pub receipt_only: bool,
    /// The range was denied as a whole.
    pub denied: bool,
    /// The peer's policy hash did not match.
    pub policy_mismatch: bool,
    /// The writer signed two diverging prefixes.
    pub equivocation: bool,
}

/// Borsh manifest written beside the segment files. No signing seed.
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct ChainManifest {
    /// Manifest version. 1.
    pub version: u32,
    /// did:key of `verifying_key`.
    pub writer_did: String,
    /// 32-byte ed25519 verifying key.
    pub verifying_key: [u8; 32],
    /// Policy hash the chain was written under.
    pub policy_hash: [u8; 32],
    /// Borsh of the signed head.
    pub signed_head: Vec<u8>,
    /// Scopes the grant should carry. Empty means the whole store.
    pub scopes: Vec<String>,
    /// Transport binding. Empty when the chain was exported offline.
    pub endpoint_binding: Vec<u8>,
}

#[derive(BorshSerialize, BorshDeserialize)]
struct HeadWire {
    seq: u64,
    frame_hash: [u8; 32],
    merkle_root: [u8; 32],
    signature: [u8; 64],
}

#[derive(BorshSerialize)]
struct ProjectionBytes<'a> {
    nodes: Vec<(&'a str, [u8; 32])>,
    edges: &'a [(String, String, String)],
    fsrs_root: [u8; 32],
    reviewed_at: Vec<(u64, i64)>,
    intentions: Vec<(String, [u8; 32])>,
    anchors: Vec<(String, [u8; 32])>,
    conflicts: Vec<([u8; 32], [u8; 32])>,
    reinforcements: Vec<(String, [u8; 32], [u8; 32])>,
}

struct ChainIn {
    writer: String,
    frames: Vec<FrameRecord>,
    /// Include effects at or below this seq, then a receipt-only suffix.
    admitted_to: u64,
}

struct EffectIn {
    writer: String,
    op: StoreOp,
    data_seq: u64,
    frame_hash: [u8; 32],
}

struct ReadLog {
    events: Vec<GateEvent>,
}

impl EventLog for ReadLog {
    fn events_before(&self, bound: u64) -> Vec<GateEvent> {
        self.events
            .iter()
            .take_while(|event| event.seq < bound)
            .cloned()
            .collect()
    }

    fn append(&mut self, _kind: RecordKind, _payload: Vec<u8>) -> strata_gate::SeqAck {
        panic!("a mesh re-derivation log is read-only");
    }
}

/// Re-derive a peer range under `policy`. A mismatch or a sweep gap names
/// every failing `(writer, seq)` and the caller denies the whole range.
pub fn check_peer_range(
    events: &[GateEvent],
    policy: &Policy,
    writer: &str,
    log_seq_of_gate: &BTreeMap<u64, u64>,
) -> Vec<FrameFail> {
    let log = ReadLog {
        events: events.to_vec(),
    };
    let mut failures = Vec::new();
    match rederive_verdicts(&log, policy) {
        Ok(derived) => {
            for (seq, verdict) in derived {
                let Some(event) = events.iter().find(|event| {
                    event.seq == seq && event.kind == RecordKind::Gate
                }) else {
                    continue;
                };
                let Ok(stored) = GateRecord::try_from_slice(&event.payload) else {
                    failures.push(fail(writer, log_seq_of_gate, seq));
                    continue;
                };
                if stored.verdict != verdict || stored.policy_hash != policy.policy_hash() {
                    failures.push(fail(writer, log_seq_of_gate, seq));
                }
            }
        }
        Err(_) => failures.push(FrameFail {
            writer_did: writer.to_string(),
            seq: 0,
        }),
    }
    for gap in sweep(&log) {
        let seq = gap_seq(&gap).unwrap_or(0);
        failures.push(fail(writer, log_seq_of_gate, seq));
    }
    failures.sort_by(|a, b| a.seq.cmp(&b.seq).then(a.writer_did.cmp(&b.writer_did)));
    failures.dedup();
    failures
}

fn fail(writer: &str, log_seq_of_gate: &BTreeMap<u64, u64>, gate_seq: u64) -> FrameFail {
    FrameFail {
        writer_did: writer.to_string(),
        seq: log_seq_of_gate.get(&gate_seq).copied().unwrap_or(gate_seq),
    }
}

fn gap_seq(gap: &strata_gate::record::GapRecord) -> Option<u64> {
    use strata_gate::record::GapDetail;
    match &gap.detail {
        GapDetail::DutySeqGap { found, .. } => Some(*found),
        GapDetail::ReadNoReceipt { reader_seq, .. } => Some(*reader_seq),
        GapDetail::OrphanEffect { effect_seq, .. } => Some(*effect_seq),
    }
}

impl StrataStore {
    /// Sign `message` with this store's log key. The seed is not returned.
    pub fn sign_bytes(&self, message: &[u8]) -> [u8; 64] {
        self.log.sign_bytes(message)
    }

    /// This store's log verifying key. Not the seed.
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.log.verifying_key().to_bytes()
    }

    /// Last acked `(seq, frame hash)` of a replica, when one has been written.
    pub fn replica_head(
        &self,
        verifying_key: [u8; 32],
    ) -> Result<Option<(u64, [u8; 32])>, StoreError> {
        let dir = self.chain_dir(&fingerprint(&verifying_key));
        if !dir.exists() {
            return Ok(None);
        }
        let key = VerifyingKey::from_bytes(&verifying_key).map_err(|_| {
            StoreError::InvalidInput("verifying key is not ed25519".into())
        })?;
        let frames = strata::read_replica_frames(&dir, &key)?;
        Ok(frames.last().map(|frame| (frame.seq, frame.frame_hash)))
    }

    /// did:key of this store's log verifying key.
    pub fn writer_did(&self) -> String {
        writer_did(&self.log.verifying_key().to_bytes())
    }

    /// Writer-qualified handle. Local ids stay `mem-…`; this is the mesh key.
    pub fn canonical_handle(&self, id: &str) -> String {
        canonical_handle(&self.writer_did(), id)
    }

    /// Data directory the store was opened from.
    pub fn data_dir(&self) -> &Path {
        &self.dir
    }

    /// blake3 of the merged fold. A peerless store still has one: it is the
    /// projection of its own chain, and it is not `state_digest`.
    pub fn mesh_digest(&self) -> [u8; 32] {
        self.project().digest
    }

    /// Admitted `UpsertNode` contents on this writer and on every granted replica.
    ///
    /// Peer memories stay in the replica chain. Callers that want the same
    /// view the mesh fold hashed read them here instead of `nodes()`.
    pub fn mesh_upsert_contents(&self) -> Vec<String> {
        let mut out = Vec::new();
        for chain in self.chains(None) {
            for effect in owned_included(&chain) {
                if let StoreOp::UpsertNode { record } = effect.op
                    && record.is_live()
                {
                    out.push(record.content);
                }
            }
        }
        out
    }

    /// Writer -> last included frame seq. Reported beside [`Self::mesh_digest`].
    pub fn mesh_frontier(&self) -> BTreeMap<String, u64> {
        self.project().frontier
    }

    /// The merged projection: nodes, reinforcements, conflicts, digest.
    pub fn mesh_projection(&self) -> MeshProjection {
        self.project()
    }

    /// Conflict receipts this device appended.
    pub fn mesh_conflict_receipts(&self) -> Vec<OpenConflict> {
        self.mesh
            .conflicts
            .iter()
            .map(|note| OpenConflict {
                subject_id: note.subject_id.clone(),
                field_id: note.field_id.clone(),
                left: note.left.clone(),
                right: note.right.clone(),
            })
            .collect()
    }

    /// Grant receipts: writer did and the effect is resolved by scanning.
    pub fn mesh_grants(&self) -> Vec<String> {
        self.mesh.grants.keys().cloned().collect()
    }

    /// The pinned public key and policy for a granted writer.
    pub fn pinned_writer(&self, writer_did: &str) -> Option<PinnedWriter> {
        self.mesh.grants.get(writer_did).map(|grant| PinnedWriter {
            verifying_key: grant.verifying_key,
            policy_hash: grant.policy_hash,
            scopes: grant.scopes.clone(),
            endpoint_binding: grant.endpoint_binding.clone(),
        })
    }

    /// Admitted ranges: writer, from seq, to seq, effect seq.
    pub fn mesh_merges(&self) -> Vec<(String, u64, u64, u64)> {
        self.mesh
            .merges
            .iter()
            .map(|merge| {
                (
                    merge.writer_did.clone(),
                    merge.from_seq,
                    merge.to_seq,
                    merge.effect_seq,
                )
            })
            .collect()
    }

    /// Memory ids whose data keys this device destroyed.
    pub fn mesh_tombstones(&self) -> Vec<String> {
        self.mesh.tombstones.clone()
    }

    /// True when `writer` is quarantined for equivocation.
    pub fn mesh_quarantined(&self, writer: &str) -> bool {
        self.mesh.quarantined.contains(writer)
    }

    /// Policy-mismatch receipts, local hash then peer hash.
    pub fn mesh_policy_mismatches(&self) -> Vec<([u8; 32], [u8; 32])> {
        self.mesh.policy_mismatches.clone()
    }

    /// Range denials this device recorded.
    pub fn mesh_denials(&self) -> Vec<(String, Vec<FrameFail>)> {
        self.mesh.denies.clone()
    }

    /// Index-mismatch receipts: sender, segment, handle.
    pub fn mesh_index_mismatches(&self) -> Vec<(String, u64, String)> {
        self.mesh.index_mismatches.clone()
    }

    /// Fork genesis records on this writer chain.
    pub fn mesh_forks(&self) -> Vec<(String, u64, [u8; 32])> {
        self.mesh
            .forks
            .iter()
            .map(|fork| {
                (
                    fork.derived_from_did.clone(),
                    fork.derived_from_seq,
                    fork.derived_from_frame,
                )
            })
            .collect()
    }

    /// Resolve a bare id.
    ///
    /// A local record wins, so existing ids and `eff-` receipts stay put.
    /// A bare id that is not local and names more than one foreign chain is
    /// refused with the qualified candidates. One foreign hit returns that
    /// qualified handle.
    pub fn resolve_bare_id(&self, id: &str) -> Result<String, StoreError> {
        if self.nodes.contains_key(id) || self.effect_index_has_receipt(id) {
            return Ok(id.to_string());
        }
        let found = self.foreign_ids(id);
        match found.len() {
            0 => Err(StoreError::NotFound(format!("no record {id}"))),
            1 => Ok(found.into_iter().next().unwrap_or_default()),
            _ => Err(StoreError::InvalidInput(format!(
                "bare id {id} is ambiguous: {}",
                found.join(", ")
            ))),
        }
    }

    /// Admit a writer. Held unless `ctx` is the human-confirmed mesh-grant.
    pub fn grant_writer(
        &mut self,
        verifying_key: [u8; 32],
        scopes: Vec<String>,
        endpoint_binding: Vec<u8>,
        ctx: &AdmissionContext,
    ) -> Result<u64, StoreError> {
        let op = StoreOp::GrantWriter {
            writer_did: writer_did(&verifying_key),
            verifying_key,
            policy_hash: self.policy.policy_hash(),
            scopes,
            endpoint_binding,
        };
        self.admit_mesh(op, strata_gate::record::action_kind::GRANT, ctx)
    }

    /// Assert one claim slot.
    pub fn assert_claim(
        &mut self,
        subject_id: &str,
        field_id: &str,
        value: &[u8],
        valid_from_ms: i64,
        valid_until_ms: i64,
    ) -> Result<u64, StoreError> {
        self.admit_mesh(
            StoreOp::AssertClaim {
                subject_id: subject_id.to_string(),
                field_id: field_id.to_string(),
                value: value.to_vec(),
                valid_from_ms,
                valid_until_ms,
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )
    }

    /// Bind a local data-key file for `memory_id`. The bytes are derived from
    /// the public verifying key and the id. They are not the signing seed,
    /// and this function does not log them. Issue #402's wrapped keys are not
    /// in this tree; destroying this file is the purge proof.
    pub fn bind_data_key(&self, memory_id: &str) -> Result<(), StoreError> {
        let path = data_key_path(&self.dir, memory_id)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut material = Vec::new();
        material.extend_from_slice(DATA_KEY_DOMAIN);
        material.extend_from_slice(&self.log.verifying_key().to_bytes());
        material.extend_from_slice(memory_id.as_bytes());
        let bytes = blake3::hash(&material);
        write_private(&path, bytes.as_bytes())
    }

    /// True when the data-key file is still on disk.
    pub fn data_key_present(&self, memory_id: &str) -> bool {
        data_key_path(&self.dir, memory_id)
            .ok()
            .is_some_and(|path| path.is_file())
    }

    /// Destroy the local data key and append a purge plus a tombstone receipt.
    pub fn purge_local(&mut self, memory_id: &str) -> Result<u64, StoreError> {
        let purge = self.admit_mesh(
            StoreOp::PurgeTombstone {
                memory_id: memory_id.to_string(),
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )?;
        self.admit_mesh(
            StoreOp::TombstoneReceipt {
                memory_id: memory_id.to_string(),
                writer_did: self.writer_did(),
                peer_seq: 0,
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )?;
        let _ = purge;
        Ok(purge)
    }

    /// Copy peer segment bytes into the replica without admitting them.
    ///
    /// `src` must not contain `strata.key`. The returned seq is the replica
    /// watermark after the copy. A later [`Self::import_chain`] of the same
    /// bytes admits the range. A crash between this call and the import leaves
    /// the frames durable and unadmitted.
    pub fn replicate_from(&self, verifying_key: [u8; 32], src: &Path) -> Result<u64, StoreError> {
        if src.join(KEY_FILE).exists() {
            return Err(StoreError::InvalidInput(
                "replica refused a source that contains strata.key".into(),
            ));
        }
        let key = VerifyingKey::from_bytes(&verifying_key).map_err(|_| {
            StoreError::InvalidInput("verifying key is not ed25519".into())
        })?;
        let mut replica = ReplicaLog::open(self.chain_dir(&fingerprint(&verifying_key)), key)?;
        replica.sync_from_dir(src)?;
        Ok(replica.last_acked_seq()?)
    }

    /// Make one more source frame durable. `None` when the replica matches `src`.
    pub fn replicate_frame(
        &self,
        verifying_key: [u8; 32],
        src: &Path,
    ) -> Result<Option<(u64, [u8; 32])>, StoreError> {
        if src.join(KEY_FILE).exists() {
            return Err(StoreError::InvalidInput(
                "replica refused a source that contains strata.key".into(),
            ));
        }
        let key = VerifyingKey::from_bytes(&verifying_key).map_err(|_| {
            StoreError::InvalidInput("verifying key is not ed25519".into())
        })?;
        let mut replica = ReplicaLog::open(self.chain_dir(&fingerprint(&verifying_key)), key)?;
        for _ in 0..8 {
            let step = replica.sync_frames(src, 1)?;
            if step.last_ack.is_some() {
                return Ok(step.last_ack);
            }
            if step.caught_up {
                return Ok(None);
            }
        }
        Err(StoreError::InvalidInput(
            "replica made no progress copying the peer chain".into(),
        ))
    }

    /// Copy this writer's segments and a public manifest to `dest`.
    /// `strata.key` is not copied.
    pub fn export_chain(&self, dest: &Path) -> Result<(), StoreError> {
        if dest.join(KEY_FILE).exists() {
            return Err(StoreError::InvalidInput(
                "export destination already contains strata.key".into(),
            ));
        }
        fs::create_dir_all(dest)?;
        let source = self.log.directory();
        for path in seg_files(source)? {
            let name = path
                .file_name()
                .ok_or_else(|| StoreError::InvalidInput("segment has no name".into()))?;
            fs::copy(&path, dest.join(name))?;
        }
        let head = self.log.sign_mesh_head()?;
        let key = self.log.verifying_key().to_bytes();
        let manifest = ChainManifest {
            version: 1,
            writer_did: writer_did(&key),
            verifying_key: key,
            policy_hash: self.policy.policy_hash(),
            signed_head: borsh::to_vec(&HeadWire {
                seq: head.seq,
                frame_hash: head.frame_hash,
                merkle_root: head.merkle_root,
                signature: head.signature,
            })
            .map_err(|err| StoreError::Encode(err.to_string()))?,
            scopes: Vec::new(),
            endpoint_binding: Vec::new(),
        };
        fs::write(
            dest.join(MANIFEST),
            borsh::to_vec(&manifest).map_err(|err| StoreError::Encode(err.to_string()))?,
        )?;
        fs::write(dest.join("writer.pub"), key)?;
        if dest.join(KEY_FILE).exists() {
            return Err(StoreError::InvalidInput(
                "export wrote strata.key; that file must not leave the writer".into(),
            ));
        }
        Ok(())
    }

    /// Verify `src` and fold it. A chain with no admitted grant is not folded
    /// unless `ctx` is the human-confirmed grant path, which admits the grant
    /// first. `src/strata.key` is refused before the file is opened.
    pub fn import_chain(&mut self, src: &Path, ctx: &AdmissionContext) -> Result<ImportReport, StoreError> {
        if src.join(KEY_FILE).exists() {
            return Err(StoreError::InvalidInput(
                "import refused: source contains strata.key, which a replica never reads".into(),
            ));
        }
        let manifest = read_manifest(src)?;
        if writer_did(&manifest.verifying_key) != manifest.writer_did {
            return Err(StoreError::InvalidInput(
                "manifest writer did does not match the verifying key".into(),
            ));
        }
        let key = VerifyingKey::from_bytes(&manifest.verifying_key).map_err(|_| {
            StoreError::InvalidInput("manifest verifying key is not ed25519".into())
        })?;
        if manifest.writer_did == self.writer_did() {
            return Err(StoreError::InvalidInput(
                "this store's writer key already matches the source; run vestige mesh fork-identity"
                    .into(),
            ));
        }
        let local_policy = self.policy.policy_hash();
        if manifest.policy_hash != local_policy {
            let seq = self.admit_mesh(
                StoreOp::PolicyMismatch {
                    local_hash: local_policy,
                    peer_hash: manifest.policy_hash,
                },
                strata_gate::record::action_kind::WRITE,
                &AdmissionContext::default(),
            )?;
            return Ok(self.report(Some(seq), false, false, true, false));
        }
        if !self.mesh.grants.contains_key(&manifest.writer_did) {
            self.grant_writer(
                manifest.verifying_key,
                manifest.scopes.clone(),
                manifest.endpoint_binding.clone(),
                ctx,
            )?;
        }
        if self.mesh.quarantined.contains(&manifest.writer_did) {
            return Err(StoreError::InvalidInput(format!(
                "writer {} is quarantined",
                manifest.writer_did
            )));
        }
        let replica_dir = self.chain_dir(&fingerprint(&manifest.verifying_key));
        let mut replica = ReplicaLog::open(&replica_dir, key)?;
        if let Err(err) = replica.sync_from_dir(src) {
            if err.to_string().contains("equivocation") {
                let seq = self.record_equivocation(&manifest, &err.to_string())?;
                return Ok(self.report(Some(seq), false, false, false, true));
            }
            return Err(err.into());
        }
        let frames = replica.read_frames()?;
        self.verify_signed_head(&manifest, &frames)?;
        let (events, gate_seqs) = gate_events_of(&frames);
        let failures = check_peer_range(&events, &self.policy, &manifest.writer_did, &gate_seqs);
        if !failures.is_empty() {
            let seq = self.admit_mesh(
                StoreOp::DenyRange {
                    writer_did: manifest.writer_did.clone(),
                    failures,
                },
                strata_gate::record::action_kind::WRITE,
                &AdmissionContext::default(),
            )?;
            return Ok(self.report(Some(seq), false, true, false, false));
        }
        let previous = self
            .mesh
            .merges
            .iter()
            .filter(|merge| merge.writer_did == manifest.writer_did)
            .map(|merge| merge.to_seq)
            .max()
            .unwrap_or(0);
        let head = frames.last().map(|frame| frame.seq).unwrap_or(0);
        if head <= previous {
            return Ok(self.report(None, true, false, false, false));
        }
        let effects = admitted_effects(&manifest.writer_did, &frames);
        let fresh: Vec<&EffectIn> = effects.iter().filter(|effect| effect.data_seq > previous).collect();
        if fresh.iter().all(|effect| is_mesh_receipt(&effect.op)) {
            return Ok(self.report(None, true, false, false, false));
        }
        let hashes: Vec<[u8; 32]> = frames.iter().map(|frame| frame.payload_blake3).collect();
        let proof = consistency_proof(&hashes, previous as usize);
        let old_root = merkle_root(&hashes[..previous as usize]);
        let new_root = merkle_root(&hashes);
        if previous > 0
            && !verify_consistency(previous as usize, hashes.len(), &old_root, &new_root, &proof)
        {
            let seq = self.record_equivocation(&manifest, "consistency proof failed")?;
            return Ok(self.report(Some(seq), false, false, false, true));
        }
        let proof_hash = hash_nodes(&proof);
        let rederive_hash = hash32(
            &rederive_verdicts(
                &ReadLog {
                    events: events.clone(),
                },
                &self.policy,
            )
            .map_err(|err| StoreError::Gate(err.to_string()))?
            .iter()
            .flat_map(|(seq, verdict)| {
                let mut bytes = seq.to_le_bytes().to_vec();
                bytes.push(verdict_byte(*verdict));
                bytes
            })
            .collect::<Vec<_>>(),
        );
        let projected = self.project_with(Some((manifest.writer_did.clone(), head)));
        let conflicts: Vec<ClaimRef> = projected
            .conflicts
            .iter()
            .flat_map(|conflict| [conflict.left.clone(), conflict.right.clone()])
            .collect();
        let digest = projected.digest;
        let seq = self.admit_mesh(
            StoreOp::MergeChain {
                writer_did: manifest.writer_did.clone(),
                from_seq: previous.saturating_add(1),
                to_seq: head,
                signed_head: manifest.signed_head.clone(),
                consistency_proof_hash: proof_hash,
                rederive_hash,
                mesh_digest_after: digest,
                conflicts: conflicts.clone(),
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )?;
        self.record_new_conflicts(&projected)?;
        self.fold_peer_purges(&manifest.writer_did, &effects, previous)?;
        let _ = (key, conflicts);
        Ok(self.report(Some(seq), false, false, false, false))
    }

    /// Start a new writer chain whose genesis names the current fork point.
    /// The current log's `strata.key` stays where it is. The new chain is
    /// used on the next `open` of this data directory.
    pub fn fork_identity(&mut self) -> Result<String, StoreError> {
        let old_key = self.log.verifying_key().to_bytes();
        let old_did = writer_did(&old_key);
        let frames = self.log.read_frames(1)?;
        let (seq, frame) = frames
            .last()
            .map(|frame| (frame.seq, frame.frame_hash))
            .unwrap_or((0, [0u8; 32]));
        let export_dir = self.dir.join("mesh/fork-export");
        let _ = fs::remove_dir_all(&export_dir);
        self.export_chain(&export_dir)?;
        let new_log = self.dir.join("mesh/fork-log");
        if new_log.exists() {
            return Err(StoreError::InvalidInput(
                "mesh/fork-log already exists".into(),
            ));
        }
        {
            let mut fresh =
                StrataStore::open_log_with_policy(&self.dir, &new_log, self.policy.clone())?;
            fresh.admit_mesh(
                StoreOp::ForkGenesis {
                    derived_from_did: old_did.clone(),
                    derived_from_seq: seq,
                    derived_from_frame: frame,
                },
                strata_gate::record::action_kind::WRITE,
                &AdmissionContext::default(),
            )?;
            fresh.import_chain(&export_dir, &mesh_grant_context())?;
            let new_did = fresh.writer_did();
            drop(fresh);
            fs::write(self.dir.join(ACTIVE_WRITER), "mesh/fork-log\n")?;
            let _ = fs::remove_dir_all(&export_dir);
            Ok(new_did)
        }
    }

    /// Record that a peer range failed re-derivation. The range is not folded.
    pub fn deny_range(&mut self, writer_did: &str, failures: Vec<FrameFail>) -> Result<u64, StoreError> {
        self.admit_mesh(
            StoreOp::DenyRange {
                writer_did: writer_did.to_string(),
                failures,
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )
    }

    /// Record an index lie. Admission is unchanged: the hint never selects frames.
    pub fn record_index_mismatch(
        &mut self,
        sender_did: &str,
        segment: u64,
        handle: &str,
    ) -> Result<u64, StoreError> {
        self.admit_mesh(
            StoreOp::IndexMismatch {
                sender_did: sender_did.to_string(),
                segment,
                handle: handle.to_string(),
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )
    }

    fn effect_index_has_receipt(&self, id: &str) -> bool {
        let Some(rest) = id.strip_prefix("eff-") else {
            return false;
        };
        let Ok(seq) = u64::from_str_radix(rest, 16) else {
            return false;
        };
        self.effect_by_seq(seq).ok().flatten().is_some()
    }

    fn foreign_ids(&self, id: &str) -> Vec<String> {
        let mut found = Vec::new();
        let Ok(dirs) = fs::read_dir(self.dir.join("mesh/chains")) else {
            return found;
        };
        for entry in dirs.flatten() {
            let writer_pub = entry.path().join("writer.pub");
            let Ok(bytes) = fs::read(&writer_pub) else {
                continue;
            };
            if bytes.len() != 32 {
                continue;
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes);
            let Ok(verifying) = VerifyingKey::from_bytes(&key) else {
                continue;
            };
            let path = entry.path();
            let Ok(frames) = strata::read_replica_frames(&path, &verifying) else {
                continue;
            };
            let did = writer_did(&key);
            if !self.mesh.grants.contains_key(&did) {
                continue;
            }
            let effects = admitted_effects(&did, &frames);
            if effects.iter().any(|effect| effect_names(&effect.op, id)) {
                found.push(canonical_handle(&did, id));
            }
        }
        found.sort();
        found.dedup();
        found
    }

    fn report(
        &self,
        receipt_seq: Option<u64>,
        receipt_only: bool,
        denied: bool,
        policy_mismatch: bool,
        equivocation: bool,
    ) -> ImportReport {
        ImportReport {
            receipt_seq,
            mesh_digest: self.mesh_digest(),
            receipt_only,
            denied,
            policy_mismatch,
            equivocation,
        }
    }

    fn record_equivocation(&mut self, manifest: &ChainManifest, _why: &str) -> Result<u64, StoreError> {
        let previous = self
            .mesh
            .merges
            .iter()
            .rev()
            .find(|merge| merge.writer_did == manifest.writer_did)
            .map(|merge| merge.signed_head.clone())
            .unwrap_or_default();
        let seq = self.admit_mesh(
            StoreOp::Equivocation {
                writer_did: manifest.writer_did.clone(),
                head_a: previous,
                head_b: manifest.signed_head.clone(),
            },
            strata_gate::record::action_kind::WRITE,
            &AdmissionContext::default(),
        )?;
        Ok(seq)
    }

    fn verify_signed_head(&self, manifest: &ChainManifest, frames: &[FrameRecord]) -> Result<(), StoreError> {
        let wire = HeadWire::try_from_slice(&manifest.signed_head)
            .map_err(|err| StoreError::InvalidInput(format!("signed head does not decode: {err}")))?;
        let head = SignedHead {
            seq: wire.seq,
            frame_hash: wire.frame_hash,
            merkle_root: wire.merkle_root,
            signature: wire.signature,
        };
        let key = VerifyingKey::from_bytes(&manifest.verifying_key).map_err(|_| {
            StoreError::InvalidInput("verifying key is not ed25519".into())
        })?;
        if !verify_head(&key, &head) {
            return Err(StoreError::InvalidInput(
                "signed head does not verify under the pinned key".into(),
            ));
        }
        let hashes: Vec<[u8; 32]> = frames.iter().map(|frame| frame.payload_blake3).collect();
        if merkle_root(&hashes) != head.merkle_root {
            return Err(StoreError::InvalidInput(
                "signed head merkle root does not match the frames".into(),
            ));
        }
        let seq = frames.last().map(|frame| frame.seq).unwrap_or(0);
        let frame_hash = frames.last().map(|frame| frame.frame_hash).unwrap_or([0u8; 32]);
        if head.seq != seq || head.frame_hash != frame_hash {
            return Err(StoreError::InvalidInput(
                "signed head does not name the chain tail".into(),
            ));
        }
        Ok(())
    }

    fn record_new_conflicts(&mut self, projected: &MeshProjection) -> Result<(), StoreError> {
        for conflict in &projected.conflicts {
            let already = self.mesh.conflicts.iter().any(|note| {
                note.left.frame_hash == conflict.left.frame_hash
                    && note.right.frame_hash == conflict.right.frame_hash
            });
            if already {
                continue;
            }
            self.admit_mesh(
                StoreOp::ConflictRecord {
                    subject_id: conflict.subject_id.clone(),
                    field_id: conflict.field_id.clone(),
                    left: conflict.left.clone(),
                    right: conflict.right.clone(),
                },
                strata_gate::record::action_kind::WRITE,
                &AdmissionContext::default(),
            )?;
        }
        Ok(())
    }

    fn fold_peer_purges(
        &mut self,
        writer: &str,
        effects: &[EffectIn],
        previous: u64,
    ) -> Result<(), StoreError> {
        for effect in effects {
            if effect.data_seq <= previous {
                continue;
            }
            if let StoreOp::PurgeTombstone { memory_id } = &effect.op {
                self.admit_mesh(
                    StoreOp::TombstoneReceipt {
                        memory_id: memory_id.clone(),
                        writer_did: writer.to_string(),
                        peer_seq: effect.data_seq,
                    },
                    strata_gate::record::action_kind::WRITE,
                    &AdmissionContext::default(),
                )?;
            }
        }
        Ok(())
    }

    fn admit_mesh(
        &mut self,
        op: StoreOp,
        kind: u8,
        ctx: &AdmissionContext,
    ) -> Result<u64, StoreError> {
        let params = if kind == strata_gate::record::action_kind::GRANT {
            mesh_grant_params(ctx)
        } else {
            None
        };
        let (effect_seq, _) = self.admit_write_with_params(op, kind, Vec::new(), params)?;
        if let Some(note) = self.mesh.merges.last_mut()
            && note.effect_seq == 0
        {
            note.effect_seq = effect_seq;
        }
        if let Some(note) = self.mesh.conflicts.last_mut()
            && note.effect_seq == 0
        {
            note.effect_seq = effect_seq;
        }
        Ok(effect_seq)
    }

    fn chain_dir(&self, fp: &str) -> PathBuf {
        self.dir.join("mesh/chains").join(fp)
    }

    fn project(&self) -> MeshProjection {
        self.project_with(None)
    }

    fn project_with(&self, extra: Option<(String, u64)>) -> MeshProjection {
        fold_chains(&self.chains(extra))
    }

    fn chains(&self, extra: Option<(String, u64)>) -> Vec<ChainIn> {
        let local = self.writer_did();
        let mut chains = vec![ChainIn {
            writer: local.clone(),
            frames: self.log.read_frames(1).unwrap_or_default(),
            admitted_to: u64::MAX,
        }];
        if let Ok(entries) = fs::read_dir(self.dir.join("mesh/chains")) {
            for entry in entries.flatten() {
                let writer_pub = entry.path().join("writer.pub");
                let Ok(bytes) = fs::read(&writer_pub) else {
                    continue;
                };
                if bytes.len() != 32 {
                    continue;
                }
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                let did = writer_did(&key);
                if !self.mesh.grants.contains_key(&did) || self.mesh.quarantined.contains(&did) {
                    continue;
                }
                let Ok(verifying) = VerifyingKey::from_bytes(&key) else {
                    continue;
                };
                let path = entry.path();
                let Ok(frames) = strata::read_replica_frames(&path, &verifying) else {
                    continue;
                };
                let mut admitted = self
                    .mesh
                    .merges
                    .iter()
                    .filter(|merge| merge.writer_did == did)
                    .map(|merge| merge.to_seq)
                    .max()
                    .unwrap_or(0);
                if let Some((writer, to)) = &extra
                    && writer == &did
                {
                    admitted = admitted.max(*to);
                }
                chains.push(ChainIn {
                    writer: did,
                    frames,
                    admitted_to: admitted,
                });
            }
        }
        chains
    }
}

fn effect_names(op: &StoreOp, id: &str) -> bool {
    match op {
        StoreOp::UpsertNode { record } => record.id == id,
        StoreOp::ReviewNode { .. } => false,
        StoreOp::AssertClaim { subject_id, .. } => subject_id == id,
        _ => false,
    }
}

fn read_manifest(src: &Path) -> Result<ChainManifest, StoreError> {
    let bytes = fs::read(src.join(MANIFEST))?;
    ChainManifest::try_from_slice(&bytes)
        .map_err(|err| StoreError::InvalidInput(format!("chain manifest does not decode: {err}")))
}

fn seg_files(dir: &Path) -> Result<Vec<PathBuf>, StoreError> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("seg") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn data_key_path(dir: &Path, memory_id: &str) -> Result<PathBuf, StoreError> {
    if memory_id.is_empty()
        || !memory_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
    {
        return Err(StoreError::InvalidInput(
            "memory id is not a safe data-key filename".into(),
        ));
    }
    Ok(dir.join("mesh/keys").join(memory_id))
}

fn destroy_data_key(dir: &Path, memory_id: &str) {
    if let Ok(path) = data_key_path(dir, memory_id) {
        let _ = fs::remove_file(path);
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    use std::io::Write;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn hash_nodes(nodes: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for node in nodes {
        hasher.update(node);
    }
    *hasher.finalize().as_bytes()
}

fn verdict_byte(verdict: Verdict) -> u8 {
    match verdict {
        Verdict::Allow => 0,
        Verdict::Deny => 1,
        Verdict::Hold => 2,
    }
}

fn gate_events_of(frames: &[FrameRecord]) -> (Vec<GateEvent>, BTreeMap<u64, u64>) {
    let mut events = Vec::new();
    let mut log_seq = BTreeMap::new();
    let mut gate_seq = 0u64;
    for frame in frames {
        let Some(kind) = RecordKind::from_u8(frame.kind) else {
            continue;
        };
        log_seq.insert(gate_seq, frame.seq);
        events.push(GateEvent {
            seq: gate_seq,
            kind,
            payload: frame.payload.clone(),
        });
        gate_seq += 1;
    }
    (events, log_seq)
}

fn admitted_effects(writer: &str, frames: &[FrameRecord]) -> Vec<EffectIn> {
    let mut propose_at: HashMap<u64, strata_gate::record::Propose> = HashMap::new();
    let mut gates_for: HashMap<u64, Vec<(u64, GateRecord)>> = HashMap::new();
    let mut pending: HashMap<[u8; 32], VecDeque<u64>> = HashMap::new();
    let mut gate_seq = 0u64;
    let mut out = Vec::new();
    for frame in frames {
        if let Some(kind) = RecordKind::from_u8(frame.kind) {
            let gseq = gate_seq;
            gate_seq += 1;
            match kind {
                RecordKind::Propose => {
                    if let Ok(propose) =
                        strata_gate::record::Propose::try_from_slice(&frame.payload)
                    {
                        propose_at.insert(gseq, propose);
                    }
                }
                RecordKind::Gate => {
                    if let Ok(gate) = GateRecord::try_from_slice(&frame.payload) {
                        gates_for
                            .entry(gate.propose_seq)
                            .or_default()
                            .push((gseq, gate));
                    }
                }
                RecordKind::Effect => {
                    if let Ok(effect) = strata_gate::record::EffectRecord::try_from_slice(&frame.payload)
                    {
                        let covering = propose_at
                            .get(&effect.propose_seq)
                            .is_some_and(|propose| propose.action_hash == effect.action_hash);
                        let admitting = gates_for.get(&effect.propose_seq).is_some_and(|gates| {
                            gates.iter().any(|(seq, gate)| {
                                *seq == effect.gate_seq && gate.verdict == Verdict::Allow && *seq < gseq
                            })
                        });
                        if covering && admitting {
                            pending
                                .entry(effect.payload_digest)
                                .or_default()
                                .push_back(gseq);
                        }
                    }
                }
                _ => {}
            }
        } else if frame.kind == crate::KIND_STORE_WRITE
            && let WritePayload::StoreOp(op) = classify_write_payload(&frame.payload)
        {
            let digest = hash32(&frame.payload);
            if pending.get_mut(&digest).and_then(|queue| queue.pop_front()).is_some() {
                out.push(EffectIn {
                    writer: writer.to_string(),
                    op,
                    data_seq: frame.seq,
                    frame_hash: frame.frame_hash,
                });
            }
        }
    }
    out
}

fn owned_included(chain: &ChainIn) -> Vec<EffectIn> {
    let effects = admitted_effects(&chain.writer, &chain.frames);
    let head = effects.last().map(|effect| effect.data_seq).unwrap_or(0);
    let through = if chain.admitted_to == u64::MAX {
        head
    } else {
        let suffix_is_receipts = effects
            .iter()
            .filter(|effect| effect.data_seq > chain.admitted_to)
            .all(|effect| is_mesh_receipt(&effect.op));
        if suffix_is_receipts { head } else { chain.admitted_to }
    };
    effects
        .into_iter()
        .filter(|effect| effect.data_seq <= through)
        .collect()
}

struct Ready {
    frame: [u8; 32],
    writer: String,
    seq: u64,
    index: usize,
}

impl PartialEq for Ready {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ready {}
impl PartialOrd for Ready {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ready {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .frame
            .cmp(&self.frame)
            .then(other.writer.cmp(&self.writer))
            .then(other.seq.cmp(&self.seq))
            .then(other.index.cmp(&self.index))
    }
}

fn fold_chains(chains: &[ChainIn]) -> MeshProjection {
    let mut effects = Vec::new();
    let mut frontier = BTreeMap::new();
    for chain in chains {
        let included = owned_included(chain);
        if let Some(last) = included.last() {
            frontier.insert(chain.writer.clone(), last.data_seq);
        } else if chain.admitted_to == u64::MAX {
            frontier.insert(chain.writer.clone(), 0);
        }
        effects.extend(included);
    }
    let order = linearize(&effects);
    let mut forward = vec![Vec::new(); effects.len()];
    let mut last_of: HashMap<String, usize> = HashMap::new();
    for (index, effect) in effects.iter().enumerate() {
        if let Some(prev) = last_of.insert(effect.writer.clone(), index) {
            forward[prev].push(index);
        }
    }
    for (index, effect) in effects.iter().enumerate() {
        if let StoreOp::MergeChain {
            writer_did,
            from_seq,
            to_seq,
            ..
        } = &effect.op
        {
            for (other, covered) in effects.iter().enumerate() {
                if covered.writer == *writer_did
                    && covered.data_seq >= *from_seq
                    && covered.data_seq <= *to_seq
                {
                    forward[other].push(index);
                }
            }
        }
    }
    let mut nodes: BTreeMap<String, [u8; 32]> = BTreeMap::new();
    let mut live: BTreeMap<String, bool> = BTreeMap::new();
    let mut by_content: HashMap<(String, [u8; 32]), (String, [u8; 32])> = HashMap::new();
    let mut cards: HashMap<(String, u64), String> = HashMap::new();
    let mut reinforcements = Vec::new();
    let mut edges = Vec::new();
    let mut intentions = BTreeMap::new();
    let mut anchors = BTreeMap::new();
    let mut claims = Vec::new();
    let mut edge_items = Vec::new();
    let mut fsrs = State::default();
    let mut reviewed_at: BTreeMap<u64, i64> = BTreeMap::new();
    let mut review_counts: BTreeMap<u64, u32> = BTreeMap::new();
    let mut ordinal = 0u64;
    for index in &order {
        let effect = &effects[*index];
        if is_mesh_receipt(&effect.op) {
            continue;
        }
        ordinal += 1;
        match &effect.op {
            StoreOp::UpsertNode { record } => {
                let qualified = canonical_handle(&effect.writer, &record.id);
                if nodes.contains_key(&qualified) {
                    nodes.insert(qualified.clone(), canonical_hash(&record.content));
                    live.insert(qualified, record.is_live());
                } else if let Some((kept, earlier)) = by_content
                    .get(&(record.scope.clone(), canonical_hash(&record.content)))
                    .filter(|(kept, _)| live.get(kept).copied().unwrap_or(false))
                    .cloned()
                {
                    reinforcements.push(Reinforcement {
                        kept: kept.clone(),
                        later_writer: effect.writer.clone(),
                        later_seq: effect.data_seq,
                        earlier_frame: earlier,
                        later_frame: effect.frame_hash,
                    });
                    cards.insert((effect.writer.clone(), handle_of(&record.id)), kept.clone());
                    fold_mesh_review(
                        &mut fsrs,
                        &mut review_counts,
                        handle_of(&kept),
                        INGEST_RATING,
                        ordinal,
                    );
                } else {
                    nodes.insert(qualified.clone(), canonical_hash(&record.content));
                    live.insert(qualified.clone(), record.is_live());
                    by_content.insert(
                        (record.scope.clone(), canonical_hash(&record.content)),
                        (qualified.clone(), effect.frame_hash),
                    );
                    cards.insert((effect.writer.clone(), handle_of(&record.id)), qualified.clone());
                    fold_mesh_review(
                        &mut fsrs,
                        &mut review_counts,
                        handle_of(&qualified),
                        INGEST_RATING,
                        ordinal,
                    );
                }
            }
            StoreOp::ReviewNode {
                card_id,
                rating,
                reviewed_at_ms,
            } => {
                if let Some(kept) = cards.get(&(effect.writer.clone(), *card_id)) {
                    let handle = handle_of(kept);
                    fold_mesh_review(&mut fsrs, &mut review_counts, handle, *rating, ordinal);
                    if let Some(ms) = reviewed_at_ms {
                        reviewed_at.insert(handle, *ms);
                    }
                }
            }
            StoreOp::SaveEdge { edge } => {
                let source = qualify(&effect.writer, &edge.source_id, &cards);
                let target = qualify(&effect.writer, &edge.target_id, &cards);
                edges.push((source, target, edge.link_type.clone()));
                edge_items.push((*index, edge.link_type.clone(), edge.source_id.clone(), edge.target_id.clone()));
            }
            StoreOp::SaveEdges { edges: batch } => {
                for edge in batch {
                    let source = qualify(&effect.writer, &edge.source_id, &cards);
                    let target = qualify(&effect.writer, &edge.target_id, &cards);
                    edges.push((source, target, edge.link_type.clone()));
                    edge_items.push((
                        *index,
                        edge.link_type.clone(),
                        edge.source_id.clone(),
                        edge.target_id.clone(),
                    ));
                }
            }
            StoreOp::AssertClaim {
                subject_id,
                field_id,
                value,
                valid_from_ms,
                valid_until_ms,
            } => {
                claims.push(ClaimAt {
                    index: *index,
                    writer: effect.writer.clone(),
                    seq: effect.data_seq,
                    frame: effect.frame_hash,
                    subject: subject_id.clone(),
                    field: field_id.clone(),
                    value: value.clone(),
                    from: *valid_from_ms,
                    until: *valid_until_ms,
                });
            }
            StoreOp::UpsertIntentions { records } => {
                for record in records {
                    let qualified = canonical_handle(&effect.writer, &record.id);
                    intentions.insert(qualified, hash32(&borsh_bytes(record)));
                }
            }
            StoreOp::RecordAnchors { anchors: rows }
            | StoreOp::ReplaceAnchors { anchors: rows, .. } => {
                for anchor in rows {
                    let qualified = canonical_handle(&effect.writer, &anchor.id);
                    anchors.insert(qualified, hash32(&borsh_bytes(anchor)));
                }
            }
            StoreOp::SupersedeNode { id, .. } => {
                let qualified = canonical_handle(&effect.writer, id);
                live.insert(qualified, false);
            }
            StoreOp::PurgeTombstone { memory_id } => {
                let qualified = canonical_handle(&effect.writer, memory_id);
                nodes.remove(&qualified);
                live.insert(qualified, false);
            }
            _ => {}
        }
    }
    let conflicts = open_conflicts(&claims, &edge_items, &forward);
    let mut reinforcements_sorted = reinforcements;
    reinforcements_sorted.sort_by(|a, b| {
        a.kept
            .cmp(&b.kept)
            .then(a.later_frame.cmp(&b.later_frame))
            .then(a.earlier_frame.cmp(&b.earlier_frame))
    });
    let conflict_pairs: Vec<([u8; 32], [u8; 32])> = conflicts
        .iter()
        .map(|conflict| (conflict.left.frame_hash, conflict.right.frame_hash))
        .collect();
    let reinforce_bytes: Vec<(String, [u8; 32], [u8; 32])> = reinforcements_sorted
        .iter()
        .map(|item| (item.kept.clone(), item.earlier_frame, item.later_frame))
        .collect();
    let node_bytes: Vec<(&str, [u8; 32])> = nodes
        .iter()
        .filter(|(id, _)| live.get(*id).copied().unwrap_or(true))
        .map(|(id, hash)| (id.as_str(), *hash))
        .collect();
    let projection = ProjectionBytes {
        nodes: node_bytes,
        edges: &edges,
        fsrs_root: strata_kernel::checkpoint::state_root(&fsrs),
        reviewed_at: reviewed_at.iter().map(|(k, v)| (*k, *v)).collect(),
        intentions: intentions.into_iter().collect(),
        anchors: anchors.into_iter().collect(),
        conflicts: conflict_pairs,
        reinforcements: reinforce_bytes,
    };
    let digest = hash32(&borsh_bytes(&projection));
    MeshProjection {
        digest,
        frontier,
        nodes,
        reinforcements: reinforcements_sorted,
        conflicts,
        review_counts,
    }
}

struct ClaimAt {
    index: usize,
    writer: String,
    seq: u64,
    frame: [u8; 32],
    subject: String,
    field: String,
    value: Vec<u8>,
    from: i64,
    until: i64,
}

fn qualify(writer: &str, id: &str, cards: &HashMap<(String, u64), String>) -> String {
    if let Some(kept) = cards.get(&(writer.to_string(), handle_of(id))) {
        return kept.clone();
    }
    if id.contains('/') {
        id.to_string()
    } else {
        canonical_handle(writer, id)
    }
}

fn fold_mesh_review(
    state: &mut State,
    counts: &mut BTreeMap<u64, u32>,
    card: u64,
    rating: u8,
    ordinal: u64,
) {
    let event = CardEvent::Review(ReviewEvent {
        card_id: card,
        rating,
        event_seq: ordinal,
    });
    if let Ok(kernel) = Kernel::<CardEvent>::for_version(ALGO_V2) {
        kernel.apply(state, &event);
        *counts.entry(card).or_default() += 1;
    }
}

fn open_conflicts(
    claims: &[ClaimAt],
    edges: &[(usize, String, String, String)],
    forward: &[Vec<usize>],
) -> Vec<OpenConflict> {
    let mut out = Vec::new();
    for (i, left) in claims.iter().enumerate() {
        for right in claims.iter().skip(i + 1) {
            if left.subject != right.subject || left.field != right.field || left.value == right.value {
                continue;
            }
            let overlaps = left.from <= right.until && right.from <= left.until;
            if !overlaps {
                continue;
            }
            if reaches(left.index, right.index, forward) || reaches(right.index, left.index, forward) {
                continue;
            }
            let closed = edges.iter().any(|(index, kind, source, target)| {
                (*kind == "corrects" || *kind == "supersedes")
                    && mentions(&left.subject, source, target)
                    && reaches(left.index, *index, forward)
                    && reaches(right.index, *index, forward)
            });
            if closed {
                continue;
            }
            let (first, second) = if left.frame <= right.frame {
                (left, right)
            } else {
                (right, left)
            };
            out.push(OpenConflict {
                subject_id: left.subject.clone(),
                field_id: left.field.clone(),
                left: ClaimRef {
                    writer_did: first.writer.clone(),
                    seq: first.seq,
                    frame_hash: first.frame,
                },
                right: ClaimRef {
                    writer_did: second.writer.clone(),
                    seq: second.seq,
                    frame_hash: second.frame,
                },
            });
        }
    }
    out.sort_by(|a, b| a.left.frame_hash.cmp(&b.left.frame_hash).then(a.right.frame_hash.cmp(&b.right.frame_hash)));
    out
}

fn mentions(subject: &str, source: &str, target: &str) -> bool {
    source == subject || target == subject || source.ends_with(&format!("/{subject}")) || target.ends_with(&format!("/{subject}"))
}

fn reaches(from: usize, to: usize, forward: &[Vec<usize>]) -> bool {
    let mut seen = vec![false; forward.len()];
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if seen.get(node).copied().unwrap_or(true) {
            continue;
        }
        seen[node] = true;
        if let Some(next) = forward.get(node) {
            stack.extend(next);
        }
    }
    false
}

fn linearize(effects: &[EffectIn]) -> Vec<usize> {
    let mut forward = vec![Vec::new(); effects.len()];
    let mut indegree = vec![0usize; effects.len()];
    let mut last_of: HashMap<String, usize> = HashMap::new();
    for (index, effect) in effects.iter().enumerate() {
        if let Some(prev) = last_of.insert(effect.writer.clone(), index) {
            forward[prev].push(index);
            indegree[index] += 1;
        }
    }
    for (index, effect) in effects.iter().enumerate() {
        if let StoreOp::MergeChain {
            writer_did,
            from_seq,
            to_seq,
            ..
        } = &effect.op
        {
            for (other, covered) in effects.iter().enumerate() {
                if covered.writer == *writer_did
                    && covered.data_seq >= *from_seq
                    && covered.data_seq <= *to_seq
                    && other != index
                {
                    forward[other].push(index);
                    indegree[index] += 1;
                }
            }
        }
    }
    let mut heap = BinaryHeap::new();
    for (index, effect) in effects.iter().enumerate() {
        if indegree[index] == 0 {
            heap.push(Ready {
                frame: effect.frame_hash,
                writer: effect.writer.clone(),
                seq: effect.data_seq,
                index,
            });
        }
    }
    let mut order = Vec::with_capacity(effects.len());
    while let Some(ready) = heap.pop() {
        order.push(ready.index);
        for next in &forward[ready.index] {
            indegree[*next] -= 1;
            if indegree[*next] == 0 {
                let effect = &effects[*next];
                heap.push(Ready {
                    frame: effect.frame_hash,
                    writer: effect.writer.clone(),
                    seq: effect.data_seq,
                    index: *next,
                });
            }
        }
    }
    if order.len() != effects.len() {
        let seen: BTreeSet<usize> = order.iter().copied().collect();
        order.extend((0..effects.len()).filter(|index| !seen.contains(index)));
    }
    order
}

fn borsh_bytes<T: BorshSerialize>(value: &T) -> Vec<u8> {
    borsh::to_vec(value).unwrap_or_default()
}

/// Items and happens-before edges, for the linearization tests.
#[cfg(test)]
pub(crate) fn linearize_for_test(
    items: &[(String, u64, [u8; 32])],
    edges: &[(usize, usize)],
) -> Vec<usize> {
    let effects: Vec<EffectIn> = items
        .iter()
        .map(|(writer, seq, frame)| EffectIn {
            writer: writer.clone(),
            op: StoreOp::PurgeTombstone {
                memory_id: "unused".into(),
            },
            data_seq: *seq,
            frame_hash: *frame,
        })
        .collect();
    // Chain order is derived from writer+seq inside linearize. Extra edges
    // are applied by wrapping them as synthetic precedence via a second pass.
    let mut order_graph = linearize_with_edges(&effects, edges);
    if order_graph.is_empty() {
        order_graph = linearize(&effects);
    }
    order_graph
}

#[cfg(test)]
fn linearize_with_edges(effects: &[EffectIn], extra: &[(usize, usize)]) -> Vec<usize> {
    let mut forward = vec![Vec::new(); effects.len()];
    let mut indegree = vec![0usize; effects.len()];
    let mut last_of: HashMap<String, usize> = HashMap::new();
    for (index, effect) in effects.iter().enumerate() {
        if let Some(prev) = last_of.insert(effect.writer.clone(), index) {
            forward[prev].push(index);
            indegree[index] += 1;
        }
    }
    for (from, to) in extra {
        forward[*from].push(*to);
        indegree[*to] += 1;
    }
    let mut heap = BinaryHeap::new();
    for (index, effect) in effects.iter().enumerate() {
        if indegree[index] == 0 {
            heap.push(Ready {
                frame: effect.frame_hash,
                writer: effect.writer.clone(),
                seq: effect.data_seq,
                index,
            });
        }
    }
    let mut order = Vec::new();
    while let Some(ready) = heap.pop() {
        order.push(ready.index);
        for next in &forward[ready.index] {
            indegree[*next] -= 1;
            if indegree[*next] == 0 {
                let effect = &effects[*next];
                heap.push(Ready {
                    frame: effect.frame_hash,
                    writer: effect.writer.clone(),
                    seq: effect.data_seq,
                    index: *next,
                });
            }
        }
    }
    order
}

#[cfg(test)]
mod linearize_tests {
    use super::linearize_for_test;

    fn hash(byte: u8) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0] = byte;
        out
    }

    #[test]
    fn chain_order_follows_the_writer() {
        let items = vec![
            ("A".into(), 1, hash(3)),
            ("A".into(), 2, hash(1)),
            ("A".into(), 3, hash(2)),
        ];
        assert_eq!(linearize_for_test(&items, &[]), vec![0, 1, 2]);
    }

    #[test]
    fn fork_join_breaks_ties_by_frame_hash() {
        let items = vec![
            ("A".into(), 1, hash(1)),
            ("B".into(), 1, hash(9)),
            ("C".into(), 1, hash(2)),
            ("D".into(), 1, hash(4)),
        ];
        // A before B and C; B and C before D.
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
        assert_eq!(linearize_for_test(&items, &edges), vec![0, 2, 1, 3]);
    }

    #[test]
    fn diamond_uses_the_same_tie_break() {
        let items = vec![
            ("A".into(), 1, hash(5)),
            ("B".into(), 1, hash(7)),
            ("C".into(), 1, hash(3)),
            ("D".into(), 1, hash(8)),
        ];
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
        assert_eq!(linearize_for_test(&items, &edges), vec![0, 2, 1, 3]);
    }
}
