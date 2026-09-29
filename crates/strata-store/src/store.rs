//! `StrataStore` — the STRATA-native memory store.
//!
//! The durable source of truth is a [`strata::StrataLog`] under `<dir>/log`.
//! Everything else (node registry, edge indexes, FSRS fold, checkpoint list)
//! is a derived index rebuilt by replay on open — proven bit-identical by
//! [`StrataStore::state_digest`] across open/close/open cycles.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};

use borsh::{BorshDeserialize, BorshSerialize};
use strata::StrataLog;
use strata_gate::policy::{ANY_KIND, WILDCARD_PREFIX};
use strata_gate::record::{action_kind, EffectRecord, GateRecord, Propose, RecordKind, Verdict};
use strata_gate::{GateRuntime, Policy, Rule, SeqAck};
use strata_kernel::checkpoint::{checkpoint_hash, Checkpoint};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{FsrsFold, ALGO_V2};
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;
use strata_kernel::verify::verify_with_head;

use crate::error::StoreError;
use crate::gate_log::StrataEventLog;
use crate::op::{StoreOp, KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
use crate::types::{
    ConnectionRecord, EdgeDirection, EdgeKind, IngestInput, NodeRecord, VALID_FOREVER_MS,
};

/// Subdirectory holding the durable log.
const LOG_DIR: &str = "log";
/// Anchor file: hash of the head checkpoint (tamper-evidence for the
/// successor-less head, per the strata-kernel verify contract).
const META_NAME: &str = "store.meta";
/// Rating folded for a brand-new node's ingest review ("Good").
const INGEST_RATING: u8 = 3;
/// Default node type when an ingest input leaves it empty.
const DEFAULT_NODE_TYPE: &str = "fact";

#[derive(BorshSerialize, BorshDeserialize)]
struct StoreMeta {
    magic: [u8; 8],
    head_checkpoint_hash: [u8; 32],
    head_log_seq: u64,
}

/// The default pinned policy: allow writes, hold destructive actions.
///
/// Rule 1 holds every `RETIRE` action (supersession); rule 2 allows anything
/// else under a generous blast-radius cap. First match wins; empty policy =
/// deny everything (useful for tests).
pub fn default_policy() -> Policy {
    Policy {
        rules: vec![
            Rule {
                match_kind: action_kind::RETIRE,
                match_params_hash_prefix: WILDCARD_PREFIX,
                max_blast_radius: u32::MAX,
                forbid_forgotten_lessons: false,
                require_human: false,
                verdict: Verdict::Hold,
            },
            Rule {
                match_kind: ANY_KIND,
                match_params_hash_prefix: WILDCARD_PREFIX,
                max_blast_radius: 10_000,
                forbid_forgotten_lessons: false,
                require_human: false,
                verdict: Verdict::Allow,
            },
        ],
    }
}

/// Stable u64 card handle for a node id: first 8 bytes of blake3(id),
/// little-endian. Collision probability is negligible (documented, not
/// guarded: 2^-64 birthday bound per pair).
pub fn handle_of(id: &str) -> u64 {
    let digest = blake3::hash(id.as_bytes());
    let bytes: [u8; 8] = digest.as_bytes()[0..8]
        .try_into()
        .expect("blake3 gives 32 bytes");
    u64::from_le_bytes(bytes)
}

fn hash32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn borsh_vec<T: BorshSerialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    borsh::to_vec(value).map_err(|e| StoreError::Encode(e.to_string()))
}

/// Canonical projection of the derived state used by [`StrataStore::state_digest`].
#[derive(BorshSerialize)]
struct StateDigest<'a> {
    nodes: Vec<(&'a str, &'a NodeRecord)>,
    origins: Vec<(&'a str, u64)>,
    edges: &'a [ConnectionRecord],
    fsrs_root: [u8; 32],
    checkpoints: Vec<[u8; 32]>,
    orphan_writes: u64,
}

/// The STRATA-native memory store.
///
/// See the crate docs for the write path and determinism contract. v1 is
/// single-threaded single-writer (`!Send` through the gate-log cache).
pub struct StrataStore {
    dir: PathBuf,
    log: StrataLog,
    gate_log: StrataEventLog,
    policy: Policy,
    /// Node registry (derived).
    nodes: BTreeMap<String, NodeRecord>,
    /// node id -> gate-space effect seq of the WRITE that created it (gate
    /// context ids; keeps the gate's `ReadNoReceipt` duty clean).
    origins: BTreeMap<String, u64>,
    /// Edge list in landing order (derived).
    edges: Vec<ConnectionRecord>,
    /// source id -> edge indexes (forward).
    forward: BTreeMap<String, Vec<usize>>,
    /// target id -> edge indexes (reverse).
    reverse: BTreeMap<String, Vec<usize>>,
    /// FSRS fold state (derived, kernel-canonical).
    fsrs: State,
    /// Review events in fold order with their hashes (reused by verification).
    review_events: Vec<(u64, [u8; 32], ReviewEvent)>,
    /// `event_seq -> created_at_ms` for a review folded on a node upsert.
    /// An explicit [`StoreOp::ReviewNode`] frame has no timestamp, so it is
    /// absent here. Not part of [`StrataStore::state_digest`].
    review_timestamps: BTreeMap<u64, i64>,
    /// Sealed checkpoints in log order.
    checkpoints: Vec<Checkpoint>,
    /// Data frames that had no admitting effect in the log (ignored).
    orphan_writes: u64,
}

/// Whether the card's last review frame carries a wall-clock timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewClock {
    /// Last review was folded on a node upsert. The value is that record's
    /// `created_at_ms`.
    Mapped(i64),
    /// Last review is an explicit review frame. Those frames store a log
    /// sequence and no timestamp.
    Unmapped,
}

impl StrataStore {
    /// Open (or create) a store under `dir` with the default policy
    /// (allow writes, hold destructive).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_policy(dir, default_policy())
    }

    /// Open (or create) a store under `dir` with a caller-pinned policy.
    ///
    /// Replay only honors recorded gate verdicts, so opening with a different
    /// policy than the one that wrote the log is safe; it only affects new
    /// admissions (and `rederive_verdicts`).
    pub fn open_with_policy(dir: impl AsRef<Path>, policy: Policy) -> Result<Self, StoreError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let log = StrataLog::open(dir.join(LOG_DIR))?;
        let gate_log = StrataEventLog::new(log.clone())?;
        let mut store = Self {
            dir,
            log,
            gate_log,
            policy,
            nodes: BTreeMap::new(),
            origins: BTreeMap::new(),
            edges: Vec::new(),
            forward: BTreeMap::new(),
            reverse: BTreeMap::new(),
            fsrs: State::default(),
            review_events: Vec::new(),
            review_timestamps: BTreeMap::new(),
            checkpoints: Vec::new(),
            orphan_writes: 0,
        };
        store.replay()?;
        store.verify_checkpoint_chain()?;
        Ok(store)
    }

    // ------------------------------------------------------------------
    // Replay: rebuild every derived map from the log alone
    // ------------------------------------------------------------------

    fn replay(&mut self) -> Result<(), StoreError> {
        let frames = self.log.read_frames(1)?;

        // Gate-space structures (gate frames only, dense 0-based seqs).
        let mut propose_at: HashMap<u64, Propose> = HashMap::new();
        let mut gates_for: HashMap<u64, Vec<(u64, GateRecord)>> = HashMap::new();
        // payload_digest -> admitted-but-unconsumed effect gate seqs.
        let mut pending: HashMap<[u8; 32], VecDeque<u64>> = HashMap::new();
        let mut gate_seq_counter: u64 = 0;

        for frame in frames {
            let seq = frame.seq;
            if let Some(kind) = RecordKind::from_u8(frame.kind) {
                let gseq = gate_seq_counter;
                gate_seq_counter += 1;
                match kind {
                    RecordKind::Propose => {
                        if let Ok(p) = Propose::try_from_slice(&frame.payload) {
                            propose_at.insert(gseq, p);
                        }
                    }
                    RecordKind::Gate => {
                        if let Ok(g) = GateRecord::try_from_slice(&frame.payload) {
                            gates_for.entry(g.propose_seq).or_default().push((gseq, g));
                        }
                    }
                    RecordKind::Effect => {
                        if let Ok(effect) = EffectRecord::try_from_slice(&frame.payload) {
                            let covering_propose = propose_at
                                .get(&effect.propose_seq)
                                .is_some_and(|p| p.action_hash == effect.action_hash);
                            let admitting_gate =
                                gates_for.get(&effect.propose_seq).is_some_and(|gs| {
                                    gs.iter().any(|(gsq, g)| {
                                        *gsq == effect.gate_seq
                                            && g.verdict == Verdict::Allow
                                            && *gsq < gseq
                                    })
                                });
                            if covering_propose && admitting_gate {
                                pending
                                    .entry(effect.payload_digest)
                                    .or_default()
                                    .push_back(gseq);
                            }
                        }
                    }
                    _ => {}
                }
            } else if frame.kind == KIND_STORE_WRITE {
                let digest = hash32(&frame.payload);
                let admitted = pending.get_mut(&digest).and_then(|queue| queue.pop_front());
                match (admitted, StoreOp::try_from_slice(&frame.payload).ok()) {
                    (Some(gseq), Some(op)) => self.apply_op(&op, gseq, seq)?,
                    _ => self.orphan_writes += 1,
                }
            } else if frame.kind == KIND_STORE_CHECKPOINT {
                if let Ok(cp) = Checkpoint::try_from_slice(&frame.payload) {
                    self.checkpoints.push(cp);
                }
            }
            // Unknown kinds are ignored: forward compatibility.
        }
        Ok(())
    }

    /// Apply one admitted op to the derived maps. Live writes and replay call
    /// this with the same inputs, which is what makes reopen bit-identical.
    ///
    /// `gate_effect_seq` is the admitting effect's gate-space seq (used for
    /// origins/gate contexts); `frame_seq` is the data frame's log seq (used
    /// as the FSRS `event_seq`).
    fn apply_op(
        &mut self,
        op: &StoreOp,
        gate_effect_seq: u64,
        frame_seq: u64,
    ) -> Result<(), StoreError> {
        match op {
            StoreOp::UpsertNode { record } => {
                let handle = handle_of(&record.id);
                let is_new = !self.fsrs.cards.contains_key(&handle);
                self.origins.insert(record.id.clone(), gate_effect_seq);
                self.nodes.insert(record.id.clone(), record.clone());
                if is_new {
                    // Every ingest folds one ReviewEvent ("Good") into the
                    // kernel state under the record's kernel version. That
                    // review's only timestamp is this record's created_at_ms.
                    self.fold_review(handle, INGEST_RATING, record.kernel_id, frame_seq)?;
                    self.review_timestamps
                        .insert(frame_seq, record.created_at_ms);
                }
            }
            StoreOp::SaveEdge { edge } => {
                let idx = self.edges.len();
                self.edges.push(edge.clone());
                self.forward
                    .entry(edge.source_id.clone())
                    .or_default()
                    .push(idx);
                self.reverse
                    .entry(edge.target_id.clone())
                    .or_default()
                    .push(idx);
            }
            StoreOp::SupersedeNode { id, superseded_by } => {
                if let Some(record) = self.nodes.get_mut(id) {
                    record.superseded_by = Some(superseded_by.clone());
                }
            }
            StoreOp::ReviewNode { card_id, rating } => {
                self.fold_review(*card_id, *rating, ALGO_V2, frame_seq)?;
            }
        }
        Ok(())
    }

    fn fold_review(
        &mut self,
        card_id: u64,
        rating: u8,
        kernel_id: u32,
        event_seq: u64,
    ) -> Result<(), StoreError> {
        let event = ReviewEvent {
            card_id,
            rating,
            event_seq,
        };
        let event_hash = hash32(&borsh_vec(&event)?);
        let kernel = Kernel::<ReviewEvent>::for_version(kernel_id)
            .map_err(|e| StoreError::Verify(e.to_string()))?;
        kernel.apply(&mut self.fsrs, &event);
        self.review_events.push((event_seq, event_hash, event));
        Ok(())
    }

    // ------------------------------------------------------------------
    // Admission: every mutation passes PROPOSE -> GATE -> EFFECT -> data
    // ------------------------------------------------------------------

    fn admit_write(
        &mut self,
        op: StoreOp,
        action_kind_code: u8,
        context: Vec<u64>,
    ) -> Result<(u64, u64), StoreError> {
        let op_bytes = borsh_vec(&op)?;
        let action_hash = hash32(&op_bytes);

        let mut runtime = GateRuntime::new(self.gate_log.clone(), self.policy.clone());
        let propose = Propose {
            action_hash,
            action_kind: action_kind_code,
            params_hash: action_hash,
            context,
        };
        let propose_ack = runtime.commit_propose(propose);
        let gate_ack = runtime
            .commit_gate(propose_ack.seq)
            .map_err(|e| StoreError::Gate(e.to_string()))?;
        let (_, verdict) = runtime
            .latest_gate(propose_ack.seq)
            .expect("commit_gate just appended the gate");
        match verdict {
            Verdict::Allow => {}
            Verdict::Deny => {
                return Err(StoreError::Denied {
                    propose_seq: propose_ack.seq,
                });
            }
            Verdict::Hold => {
                return Err(StoreError::Held {
                    propose_seq: propose_ack.seq,
                });
            }
        }

        let effect = EffectRecord {
            propose_seq: propose_ack.seq,
            gate_seq: gate_ack.seq,
            action_hash,
            payload_digest: hash32(&op_bytes),
        };
        let effect_ack: SeqAck = runtime
            .commit_effect(effect)
            .map_err(|r| StoreError::Rejected(r.to_string()))?;

        // The data frame lands only after an admitted effect cites its digest.
        let data_acks = self.log.append_batch(vec![(KIND_STORE_WRITE, op_bytes)])?;
        let data_seq = data_acks[0].seq;

        self.apply_op(&op, effect_ack.seq, data_seq)?;
        Ok((effect_ack.seq, data_seq))
    }

    fn context_for(&self, ids: &[&str]) -> Vec<u64> {
        ids.iter()
            .filter_map(|id| self.origins.get(*id).copied())
            .collect()
    }

    fn require_node(&self, id: &str) -> Result<&NodeRecord, StoreError> {
        self.nodes
            .get(id)
            .ok_or_else(|| StoreError::NotFound(format!("node {id}")))
    }

    // ------------------------------------------------------------------
    // vestige-core memory surface
    // ------------------------------------------------------------------

    /// Ingest a memory into the default scope (""), returning its id.
    pub fn ingest(&mut self, input: IngestInput) -> Result<String, StoreError> {
        self.ingest_in_scope(input, "")
    }

    /// Ingest a memory into a named scope, returning its id.
    ///
    /// The id is derived from the log head (`mem-<next-seq-hex>`), so it is
    /// deterministic under replay.
    pub fn ingest_in_scope(
        &mut self,
        input: IngestInput,
        scope: &str,
    ) -> Result<String, StoreError> {
        if input.content.trim().is_empty() {
            return Err(StoreError::InvalidInput("content must not be empty".into()));
        }
        let id = format!("mem-{:016x}", self.log.head().next_seq);
        let created_at_ms = input.created_at_ms.unwrap_or(0);
        let mut tags = input.tags;
        tags.sort_unstable();
        tags.dedup();
        let record = NodeRecord {
            id: id.clone(),
            kernel_id: ALGO_V2,
            scope: scope.to_string(),
            content: input.content,
            node_type: if input.node_type.is_empty() {
                DEFAULT_NODE_TYPE.to_string()
            } else {
                input.node_type
            },
            tags,
            created_at_ms,
            valid_from_ms: input.valid_from_ms.unwrap_or(created_at_ms),
            valid_until_ms: input.valid_until_ms.unwrap_or(VALID_FOREVER_MS),
            superseded_by: None,
        };
        // A brand-new fact references nothing yet: empty context.
        self.admit_write(
            StoreOp::UpsertNode { record },
            action_kind::WRITE,
            Vec::new(),
        )?;
        Ok(id)
    }

    /// Gate-space effect seq of the write that created `id`, if it exists.
    pub fn origin_seq(&self, id: &str) -> Option<u64> {
        self.origins.get(id).copied()
    }

    /// Every node id and the effect seq that admitted it, in id order.
    pub fn origins(&self) -> Vec<(String, u64)> {
        self.origins
            .iter()
            .map(|(id, seq)| (id.clone(), *seq))
            .collect()
    }

    /// Fetch one node by id.
    pub fn get_node(&self, id: &str) -> Option<NodeRecord> {
        self.nodes.get(id).cloned()
    }

    /// All live (non-superseded) nodes in a scope, ordered by id.
    pub fn get_all_nodes_in_scope(&self, scope: &str) -> Vec<NodeRecord> {
        self.nodes
            .values()
            .filter(|r| r.scope == scope && r.is_live())
            .cloned()
            .collect()
    }

    /// Rewrite a node's creation time (admitted as a WRITE; no new review is
    /// folded for an existing card).
    pub fn set_created_at(&mut self, id: &str, created_at_ms: i64) -> Result<(), StoreError> {
        let mut record = self.require_node(id)?.clone();
        record.created_at_ms = created_at_ms;
        let context = self.context_for(&[id]);
        self.admit_write(StoreOp::UpsertNode { record }, action_kind::WRITE, context)?;
        Ok(())
    }

    /// Append one typed edge (vocabulary-validated; the source node must
    /// exist — targets may point at non-memory artifacts like file anchors).
    pub fn save_connection(&mut self, connection: &ConnectionRecord) -> Result<(), StoreError> {
        if EdgeKind::parse(&connection.link_type).is_none() {
            return Err(StoreError::InvalidInput(format!(
                "link_type '{}' is not in the typed-edge vocabulary",
                connection.link_type
            )));
        }
        if connection.strength_milli < 0 {
            return Err(StoreError::InvalidInput(
                "strength_milli must be >= 0".into(),
            ));
        }
        self.require_node(&connection.source_id)?;
        let mut context = self.context_for(&[&connection.source_id]);
        if self.nodes.contains_key(&connection.target_id) {
            context.extend(self.context_for(&[&connection.target_id]));
        }
        context.sort_unstable();
        context.dedup();
        self.admit_write(
            StoreOp::SaveEdge {
                edge: connection.clone(),
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(())
    }

    /// All edges touching a memory: outgoing first, then incoming.
    pub fn get_connections_for_memory(&self, memory_id: &str) -> Vec<ConnectionRecord> {
        self.get_edges_for(memory_id, EdgeDirection::Both, None)
    }

    /// Typed edge query in one or both directions, optionally filtered by kind.
    pub fn get_edges_for(
        &self,
        node_id: &str,
        direction: EdgeDirection,
        kind: Option<EdgeKind>,
    ) -> Vec<ConnectionRecord> {
        let matches_kind = |e: &ConnectionRecord| kind.is_none_or(|k| e.link_type == k.as_str());
        let outgoing: Vec<ConnectionRecord> = self
            .forward
            .get(node_id)
            .map(|idxs| idxs.iter().map(|&i| self.edges[i].clone()).collect())
            .unwrap_or_default();
        let incoming: Vec<ConnectionRecord> = self
            .reverse
            .get(node_id)
            .map(|idxs| idxs.iter().map(|&i| self.edges[i].clone()).collect())
            .unwrap_or_default();
        match direction {
            EdgeDirection::Outgoing => outgoing.into_iter().filter(matches_kind).collect(),
            EdgeDirection::Incoming => incoming.into_iter().filter(matches_kind).collect(),
            EdgeDirection::Both => outgoing
                .into_iter()
                .chain(incoming)
                .filter(matches_kind)
                .collect(),
        }
    }

    /// Mark `id` as superseded by `superseded_by`.
    ///
    /// Routed as a destructive `RETIRE` action: the default policy HOLDS it;
    /// a review-gated (permissive) policy lands it.
    pub fn supersede(&mut self, id: &str, superseded_by: &str) -> Result<(), StoreError> {
        self.require_node(id)?;
        self.require_node(superseded_by)?;
        if id == superseded_by {
            return Err(StoreError::InvalidInput(
                "a node cannot supersede itself".into(),
            ));
        }
        if self.require_node(id)?.superseded_by.is_some() {
            return Err(StoreError::InvalidInput(format!(
                "node {id} is already superseded"
            )));
        }
        let context = self.context_for(&[id, superseded_by]);
        self.admit_write(
            StoreOp::SupersedeNode {
                id: id.to_string(),
                superseded_by: superseded_by.to_string(),
            },
            action_kind::RETIRE,
            context,
        )?;
        Ok(())
    }

    /// All (superseded, superseder) pairs, ordered by superseded id.
    pub fn supersession_pairs(&self) -> Vec<(String, String)> {
        self.nodes
            .iter()
            .filter_map(|(id, r)| r.superseded_by.clone().map(|by| (id.clone(), by)))
            .collect()
    }

    /// Fold an explicit FSRS review for a node (rating 1..=4).
    pub fn review(&mut self, id: &str, rating: u8) -> Result<(), StoreError> {
        self.require_node(id)?;
        if !(1..=4).contains(&rating) {
            return Err(StoreError::InvalidInput("rating must be 1..=4".into()));
        }
        let context = self.context_for(&[id]);
        self.admit_write(
            StoreOp::ReviewNode {
                card_id: handle_of(id),
                rating,
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(())
    }

    /// Wall-clock time of the card's last review, when the frame at
    /// `last_seq` is the node upsert that folded it.
    ///
    /// `Some(Mapped)` is that upsert's `created_at_ms`. `Some(Unmapped)`
    /// means the last review is an explicit review frame, which stores a
    /// log sequence and no timestamp. `None` means the id has no card.
    /// A later `set_created_at` does not move this clock.
    pub fn review_clock(&self, id: &str) -> Option<ReviewClock> {
        let card = self.card_state(id)?;
        Some(match self.review_timestamps.get(&card.last_seq) {
            Some(ms) => ReviewClock::Mapped(*ms),
            None => ReviewClock::Unmapped,
        })
    }

    /// Current FSRS scheduling card for a node (derived state, cloned).
    pub fn card_state(&self, id: &str) -> Option<strata_kernel::fsrs::CardState> {
        self.fsrs.cards.get(&handle_of(id)).cloned()
    }

    /// Retrievability of a node at the current log head — derived on read,
    /// never stored, and reads append nothing (v1).
    pub fn retrievability(&self, id: &str) -> Result<Option<f64>, StoreError> {
        let Some(card) = self.fsrs.cards.get(&handle_of(id)) else {
            return Ok(None);
        };
        FsrsFold::retrievability(card, self.log.head().last_acked_seq, ALGO_V2)
            .map(Some)
            .map_err(|e| StoreError::Verify(e.to_string()))
    }

    /// Does the stored node read like a failure? (`None` if the id is
    /// unknown.) Compatible with vestige-core's marker heuristic.
    pub fn is_failure_memory(&self, id: &str) -> Option<bool> {
        self.nodes
            .get(id)
            .map(|r| crate::types::looks_like_failure(&r.content, &r.tags))
    }

    /// Never-composed pairs: live node pairs in a scope with NO edge between
    /// them in either direction, (a, b) ordered by id, up to `limit`.
    ///
    /// This is the store-side query behind the graph tool's
    /// `never_composed` fusion candidates.
    pub fn get_never_composed(&self, scope: &str, limit: usize) -> Vec<(String, String)> {
        let ids: Vec<String> = self
            .get_all_nodes_in_scope(scope)
            .into_iter()
            .map(|r| r.id)
            .collect();
        let mut pairs = Vec::new();
        'outer: for (i, a) in ids.iter().enumerate() {
            for b in ids.iter().skip(i + 1) {
                let linked =
                    self.forward
                        .get(a)
                        .is_some_and(|out| out.iter().any(|&idx| self.edges[idx].target_id == *b))
                        || self.reverse.get(a).is_some_and(|inc| {
                            inc.iter().any(|&idx| self.edges[idx].source_id == *b)
                        });
                if !linked {
                    pairs.push((a.clone(), b.clone()));
                    if pairs.len() >= limit {
                        break 'outer;
                    }
                }
            }
        }
        pairs
    }

    // ------------------------------------------------------------------
    // Checkpoints, verification, backup
    // ------------------------------------------------------------------

    /// Seal an FSRS checkpoint over the current fold and anchor its hash in
    /// `store.meta`. No-op (returns the existing head) when no new reviews
    /// landed since the last seal.
    pub fn seal_checkpoint(&mut self) -> Result<Checkpoint, StoreError> {
        let log_seq = self.fsrs.applied_seq;
        if let Some(last) = self.checkpoints.last() {
            if log_seq <= last.log_seq {
                return Ok(*last);
            }
        }
        let prev = self.checkpoints.last().map_or([0u8; 32], checkpoint_hash);
        let cp = Checkpoint::seal(ALGO_V2, log_seq, prev, &self.fsrs);
        let payload = borsh_vec(&cp)?;
        self.log
            .append_batch(vec![(KIND_STORE_CHECKPOINT, payload)])?;
        self.write_anchor(&cp)?;
        self.checkpoints.push(cp);
        Ok(cp)
    }

    fn meta_path(&self) -> PathBuf {
        self.dir.join(META_NAME)
    }

    fn write_anchor(&self, cp: &Checkpoint) -> Result<(), StoreError> {
        let meta = StoreMeta {
            magic: *b"STRSTME1",
            head_checkpoint_hash: checkpoint_hash(cp),
            head_log_seq: cp.log_seq,
        };
        let bytes = borsh_vec(&meta)?;
        let tmp = self.dir.join(format!("{META_NAME}.tmp"));
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, self.meta_path())?;
        Ok(())
    }

    fn read_meta(&self) -> Option<StoreMeta> {
        let bytes = std::fs::read(self.meta_path()).ok()?;
        let meta = StoreMeta::try_from_slice(&bytes).ok()?;
        (meta.magic == *b"STRSTME1").then_some(meta)
    }

    /// Verify the checkpoint chain (and the folded roots it commits) with the
    /// strata-kernel verifier, anchored by the externally persisted head hash
    /// when `store.meta` exists. Runs on every open.
    pub fn verify_checkpoint_chain(&self) -> Result<(), StoreError> {
        let anchor = match (self.checkpoints.last(), self.read_meta()) {
            (None, None) => return Ok(()),
            (None, Some(_)) => {
                return Err(StoreError::Verify(
                    "store.meta exists but the log carries no checkpoint".into(),
                ));
            }
            (Some(_), None) => None,
            (Some(head), Some(meta)) => {
                if meta.head_log_seq != head.log_seq {
                    return Err(StoreError::Verify(format!(
                        "anchor log_seq {} does not match head checkpoint log_seq {}",
                        meta.head_log_seq, head.log_seq
                    )));
                }
                Some(meta.head_checkpoint_hash)
            }
        };
        let last_log_seq = self.checkpoints.last().map_or(0, |head| head.log_seq);
        let events: Vec<(u64, [u8; 32], ReviewEvent)> = self
            .review_events
            .iter()
            .filter(|(seq, _, _)| *seq <= last_log_seq)
            .cloned()
            .collect();
        verify_with_head(&self.checkpoints, anchor, events.into_iter()).map_err(StoreError::from)
    }

    /// Back the store up: seal the active segment (signed trailer; a fresh
    /// active segment is rolled so the live store keeps appending), then copy
    /// the sealed segments plus `head.state`, the signing key, and the anchor
    /// file into `dest`. The copy opens as a store via [`StrataStore::open`].
    ///
    /// `strata.lock` is deliberately NOT copied (it names this process).
    pub fn backup_to(&self, dest: impl AsRef<Path>) -> Result<(), StoreError> {
        let dest = dest.as_ref();
        self.log.seal()?;
        let dest_log = dest.join(LOG_DIR);
        std::fs::create_dir_all(&dest_log)?;
        for entry in std::fs::read_dir(self.dir.join(LOG_DIR))? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name == "strata.lock" {
                continue;
            }
            std::fs::copy(&path, dest_log.join(name))?;
        }
        if self.meta_path().exists() {
            std::fs::copy(self.meta_path(), dest.join(META_NAME))?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Gate introspection + derived-state digest
    // ------------------------------------------------------------------

    /// Structural sweep over the gate log (orphan effects, dangling reads,
    /// duty-seq holes).
    pub fn sweep(&self) -> Vec<strata_gate::record::GapRecord> {
        GateRuntime::new(self.gate_log.clone(), self.policy.clone()).sweep()
    }

    /// Re-derive every recorded gate verdict under the currently pinned
    /// policy.
    pub fn rederive_verdicts(&self) -> Result<Vec<(u64, Verdict)>, StoreError> {
        GateRuntime::new(self.gate_log.clone(), self.policy.clone())
            .rederive_verdicts()
            .map_err(|e| StoreError::Gate(e.to_string()))
    }

    /// blake3 digest over the canonical projection of every derived map
    /// (nodes, origins, edges, FSRS state root, checkpoint hashes, orphan
    /// count). Two stores replaying the same log produce the same digest.
    pub fn state_digest(&self) -> [u8; 32] {
        let digest = StateDigest {
            nodes: self.nodes.iter().map(|(k, v)| (k.as_str(), v)).collect(),
            origins: self.origins.iter().map(|(k, v)| (k.as_str(), *v)).collect(),
            edges: &self.edges,
            fsrs_root: strata_kernel::checkpoint::state_root(&self.fsrs),
            checkpoints: self.checkpoints.iter().map(checkpoint_hash).collect(),
            orphan_writes: self.orphan_writes,
        };
        hash32(&borsh_vec(&digest).expect("state digest serialization is infallible"))
    }

    /// The pinned policy.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// The durable log handle.
    pub fn log(&self) -> &StrataLog {
        &self.log
    }

    /// Every node record, in id order (includes superseded).
    pub fn nodes(&self) -> Vec<NodeRecord> {
        self.nodes.values().cloned().collect()
    }

    /// Every typed edge, in landing order.
    pub fn edges(&self) -> Vec<ConnectionRecord> {
        self.edges.clone()
    }

    /// Number of live nodes (any scope).
    pub fn node_count(&self) -> usize {
        self.nodes.values().filter(|r| r.is_live()).count()
    }

    /// Number of typed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Data frames ignored at replay (no admitting effect).
    pub fn orphan_write_count(&self) -> u64 {
        self.orphan_writes
    }

    /// Sealed checkpoints, in chain order.
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Review events retained for verification, in fold order.
    pub fn review_event_count(&self) -> usize {
        self.review_events.len()
    }
}
