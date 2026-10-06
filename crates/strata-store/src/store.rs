//! `StrataStore` — the STRATA-native memory store.
//!
//! The durable source of truth is a [`strata::StrataLog`] under `<dir>/log`.
//! Everything else (node registry, edge indexes, FSRS fold, checkpoint list)
//! is a derived index rebuilt by replay on open — proven bit-identical by
//! [`StrataStore::state_digest`] across open/close/open cycles.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};

use borsh::{BorshDeserialize, BorshSerialize};
use strata::StrataLog;
use strata_gate::policy::{ANY_KIND, WILDCARD_PREFIX};
use strata_gate::record::{action_kind, EffectRecord, GateRecord, Propose, RecordKind, Verdict};
use strata_gate::{GateRuntime, Policy, Rule, SeqAck};
use strata_kernel::checkpoint::{checkpoint_hash, Checkpoint};
use strata_kernel::event::{ReviewEvent, StrataEvent};
use strata_kernel::fsrs::{FsrsFold, ALGO_V2};
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;
use strata_kernel::verify::verify_with_head;

use crate::anchor::AnchorIndex;
use crate::card::{CardEvent, ImportedCard};
use crate::error::StoreError;
use crate::gate_log::StrataEventLog;
use crate::op::{StoreOp, KIND_STORE_CHECKPOINT, KIND_STORE_WRITE};
use crate::types::{
    AnchorRecord, ConnectionRecord, EdgeDirection, EdgeKind, IngestInput, IntentionRecord,
    NodeRecord, VALID_FOREVER_MS,
};

/// Subdirectory holding the durable log.
const LOG_DIR: &str = "log";

/// Create `path` and any missing parents; directories made here are owner-only
/// (0700) on unix.
fn create_private_dir_all(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)?;
        set_private_dir(path)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path)
    }
}

/// Restrict an existing directory to its owner (0700) on unix.
fn set_private_dir(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Copy a file, creating the destination owner-only (0600) on unix so the
/// copy is never readable by others, even briefly.
fn copy_private_file(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut source = std::fs::File::open(from)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut target = options.open(to)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        target.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    std::io::copy(&mut source, &mut target)?;
    Ok(())
}
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

/// Policy that admits `RETIRE` (supersede) as well as writes.
///
/// The MCP server pins [`default_policy`], which holds every retire. Tests
/// that need a recorded supersession chain open with this policy instead.
pub fn permissive_policy() -> Policy {
    Policy {
        rules: vec![Rule {
            match_kind: ANY_KIND,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: u32::MAX,
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Allow,
        }],
    }
}

/// Domain separator for a RETIRE rule id. The prefix is not a hash of node
/// content, ids, or names.
const RETIRE_RULE_DOMAIN: &[u8] = b"strata.retire.v1\0";

/// Named RETIRE rule: successor was admitted in this tool call.
pub const RULE_EDIT: &str = "edit";
/// Named RETIRE rule: suppression.
pub const RULE_SUPPRESS: &str = "suppress";
/// Named RETIRE rule: intention update.
pub const RULE_INTENTIONS: &str = "intentions";
/// Named RETIRE rule: purge. Also requires [`AdmissionContext::confirm`].
pub const RULE_PURGE: &str = "purge";

const NAMED_RETIRE_RULES: [&str; 4] = [RULE_EDIT, RULE_SUPPRESS, RULE_INTENTIONS, RULE_PURGE];

/// What the caller claims about a RETIRE. The gate matches the rule id here,
/// never node content or names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdmissionContext {
    /// Exact rule id: `edit`, `suppress`, `intentions`, or `purge`.
    pub rule_id: Option<String>,
    /// Required for `purge`. Ignored by the other three rules.
    pub confirm: bool,
}

/// Receipt for an admitted RETIRE. `receipt_id` is `eff-` plus the effect seq.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetireReceipt {
    /// `eff-` + 16 lowercase hex digits of [`Self::effect_seq`].
    pub receipt_id: String,
    /// Named rule that authorized this RETIRE, when one did.
    pub rule_id: Option<&'static str>,
    /// Gate-space seq of the admitting [`EffectRecord`].
    pub effect_seq: u64,
}

/// `eff-` + 16 lowercase hex digits of the admitting effect's gate seq.
pub fn effect_receipt_id(effect_seq: u64) -> String {
    format!("eff-{effect_seq:016x}")
}

/// blake3(domain || rule id). Admission copies this into `PROPOSE.params_hash`
/// only after the context's rule id matches exactly.
fn rule_id_hash(rule_id: &str) -> [u8; 32] {
    let mut body = Vec::with_capacity(RETIRE_RULE_DOMAIN.len() + rule_id.len());
    body.extend_from_slice(RETIRE_RULE_DOMAIN);
    body.extend_from_slice(rule_id.as_bytes());
    hash32(&body)
}

fn rule_id_prefix(rule_id: &str) -> [u8; 8] {
    let hash = rule_id_hash(rule_id);
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&hash[..8]);
    prefix
}

/// Rule id encoded in `params_hash`, or `None` when it is not one of the four.
pub fn retire_rule_id(params_hash: &[u8; 32]) -> Option<&'static str> {
    NAMED_RETIRE_RULES
        .into_iter()
        .find(|id| params_hash[..8] == rule_id_prefix(id))
}

/// `edit` also needs the successor in this tool call. `purge` also needs
/// `confirm`. Any other id, including a missing one, does not authorize.
fn named_retire_rule(
    rule_id: Option<&str>,
    confirm: bool,
    successor_in_call: bool,
) -> Option<&'static str> {
    match rule_id {
        Some(RULE_EDIT) if successor_in_call => Some(RULE_EDIT),
        Some(RULE_SUPPRESS) => Some(RULE_SUPPRESS),
        Some(RULE_INTENTIONS) => Some(RULE_INTENTIONS),
        Some(RULE_PURGE) if confirm => Some(RULE_PURGE),
        _ => None,
    }
}

fn retire_allow(rule_id: &str) -> Rule {
    Rule {
        match_kind: action_kind::RETIRE,
        match_params_hash_prefix: rule_id_prefix(rule_id),
        max_blast_radius: u32::MAX,
        forbid_forgotten_lessons: false,
        require_human: false,
        verdict: Verdict::Allow,
    }
}

/// The default pinned policy: four named RETIRE allows, then hold every
/// other RETIRE, then allow anything else under a blast-radius cap.
///
/// First match wins. A named rule matches only when admission copied that
/// rule id's hash into `params_hash` from [`AdmissionContext`]. Empty policy
/// denies everything (useful for tests).
pub fn default_policy() -> Policy {
    Policy {
        rules: vec![
            retire_allow(RULE_EDIT),
            retire_allow(RULE_SUPPRESS),
            retire_allow(RULE_INTENTIONS),
            retire_allow(RULE_PURGE),
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

/// What an admitted node or intention effect did. Derived by replaying the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectAction {
    /// First upsert of a node (ingest). Folds one Good review.
    Create,
    /// Later upsert of an existing node (`set_created_at`). No new review.
    Rewrite,
    /// Successor admitted under the `edit` RETIRE rule. The predecessor stays
    /// in the log; its card is not copied.
    Edit,
    /// Explicit FSRS review. `rating` is 1..=4.
    Review,
    /// Intention row inserted or replaced by `UpsertIntentions`. A batch
    /// proves one effect per row, all citing the same EFFECT. Not a card.
    Intention,
    /// Code anchor recorded by `RecordAnchors` or `ReplaceAnchors`. One
    /// effect per anchor row, named by the anchor id. Not a card.
    Anchor,
    /// Verification verdict cached by `RecordAnchorVerdict`, named by the
    /// anchor id. Not a card.
    AnchorVerdict,
    /// Typed edge admitted by `SaveEdge`, named by its source id; `edge`
    /// carries the target and kind. Not a card: it never shadows the
    /// source's own receipt in [`StrataStore::latest_effect`].
    Edge,
}

/// One node, intention or anchor effect proved from the log: covering
/// propose, Allow gate, and a data frame whose blake3 matches the effect's
/// payload digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectProof {
    /// Gate-space seq of the EFFECT record (`eff-` receipt id).
    pub effect_seq: u64,
    /// Log seq of the STORE_WRITE frame (FSRS `event_seq` for reviews).
    pub data_seq: u64,
    /// Node the effect names. An intention id for [`EffectAction::Intention`];
    /// an anchor id for [`EffectAction::Anchor`] and
    /// [`EffectAction::AnchorVerdict`].
    pub node_id: String,
    /// Which mutation landed.
    pub action: EffectAction,
    /// blake3 of the `StoreOp` payload. Equals `EFFECT.payload_digest`.
    pub payload_digest: [u8; 32],
    /// Review rating when `action` is [`EffectAction::Review`].
    pub rating: Option<u8>,
    /// `(target_id, link_type)` when `action` is [`EffectAction::Edge`].
    pub edge: Option<(String, String)>,
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

/// Unix epoch milliseconds. Used only to stamp an explicit review.
fn admission_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
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
    /// card handle → latest explicit `reviewed_at_ms`.
    reviewed_at: Vec<(u64, i64)>,
    intentions: Vec<(&'a str, &'a IntentionRecord)>,
    /// Code anchors in (node id, anchor id) order.
    anchors: Vec<&'a AnchorRecord>,
}

/// A payload is this type only when borsh consumes it exactly. Kind bytes
/// 0x20 and 0x21 are shared by store frames and migration frames.
pub(crate) fn decode_exact<T>(payload: &[u8]) -> Option<T>
where
    T: BorshDeserialize + BorshSerialize,
{
    let value = T::try_from_slice(payload).ok()?;
    let encoded = borsh::to_vec(&value).ok()?;
    (encoded == payload).then_some(value)
}

/// What a kind-0x20 payload is. A store write wins when both decoders accept it.
#[derive(Debug)]
pub(crate) enum WritePayload {
    StoreOp(StoreOp),
    ImportedNode(strata_migrate::NodeRecord),
    Neither,
}

/// What a kind-0x21 payload is. A checkpoint whose magic matches wins when both accept it.
#[derive(Debug)]
pub(crate) enum CheckpointPayload {
    Checkpoint(Checkpoint),
    ImportedEdge(strata_migrate::EdgeRecord),
    Neither,
}

/// A migration node only when the payload round-trips and its version is current.
pub(crate) fn migration_node(payload: &[u8]) -> Option<strata_migrate::NodeRecord> {
    decode_exact::<strata_migrate::NodeRecord>(payload)
        .filter(|node| node.record_version == strata_migrate::RECORD_VERSION)
}

/// A migration edge only when the payload round-trips and its version is current.
pub(crate) fn migration_edge(payload: &[u8]) -> Option<strata_migrate::EdgeRecord> {
    decode_exact::<strata_migrate::EdgeRecord>(payload)
        .filter(|edge| edge.record_version == strata_migrate::RECORD_VERSION)
}

/// The proofs one admitted `StoreOp` lands. `prove_effects` (full log scan)
/// and the incremental effect index share this, so they cannot diverge.
/// `handles` maps card handles to node ids seen so far (admitted upserts and
/// imported nodes).
fn effect_proofs_for_op(
    op: &StoreOp,
    effect_seq: u64,
    data_seq: u64,
    digest: [u8; 32],
    rule: Option<&'static str>,
    handles: &mut HashMap<u64, String>,
) -> Result<Vec<EffectProof>, StoreError> {
    let single = |node_id: String, action, rating, edge| {
        Ok(vec![EffectProof {
            effect_seq,
            data_seq,
            node_id,
            action,
            payload_digest: digest,
            rating,
            edge,
        }])
    };
    match op {
        StoreOp::UpsertNode { record } => {
            let handle = handle_of(&record.id);
            let action = if handles.contains_key(&handle) {
                EffectAction::Rewrite
            } else {
                EffectAction::Create
            };
            handles.insert(handle, record.id.clone());
            single(record.id.clone(), action, None, None)
        }
        StoreOp::SupersedeNode { superseded_by, .. } if rule == Some(RULE_EDIT) => {
            single(superseded_by.clone(), EffectAction::Edit, None, None)
        }
        StoreOp::ReviewNode {
            card_id, rating, ..
        } => {
            let Some(node_id) = handles.get(card_id).cloned() else {
                return Err(StoreError::Verify(format!(
                    "review effect {effect_seq} names an unknown card"
                )));
            };
            single(node_id, EffectAction::Review, Some(*rating), None)
        }
        // One admitted batch: every row cites this effect.
        StoreOp::UpsertIntentions { records } => Ok(records
            .iter()
            .map(|record| EffectProof {
                effect_seq,
                data_seq,
                node_id: record.id.clone(),
                action: EffectAction::Intention,
                payload_digest: digest,
                rating: None,
                edge: None,
            })
            .collect()),
        StoreOp::RecordAnchors { anchors } | StoreOp::ReplaceAnchors { anchors, .. } => Ok(anchors
            .iter()
            .map(|anchor| EffectProof {
                effect_seq,
                data_seq,
                node_id: anchor.id.clone(),
                action: EffectAction::Anchor,
                payload_digest: digest,
                rating: None,
                edge: None,
            })
            .collect()),
        StoreOp::RecordAnchorVerdict { anchor_id, .. } => {
            single(anchor_id.clone(), EffectAction::AnchorVerdict, None, None)
        }
        StoreOp::SaveEdge { edge } => single(
            edge.source_id.clone(),
            EffectAction::Edge,
            None,
            Some((edge.target_id.clone(), edge.link_type.clone())),
        ),
        StoreOp::SupersedeNode { .. } => Ok(Vec::new()),
    }
}

/// Incremental index over the proved effects: lookups by receipt seq or by
/// node id without re-reading the log. Derived state, rebuilt by replay and
/// extended by every admitted write, like the other registries.
#[derive(Default)]
struct EffectIndex {
    proofs: Vec<EffectProof>,
    /// Effect seq -> first proof that cites it.
    by_seq: BTreeMap<u64, usize>,
    /// Node id -> latest non-edge proof (highest effect seq; later wins ties).
    latest: BTreeMap<String, usize>,
    /// Card handle -> node id, including imported nodes.
    handles: HashMap<u64, String>,
    /// First review naming a card that was never created. Lookups report it
    /// the way a full scan would.
    poisoned: Option<String>,
}

impl EffectIndex {
    fn note_imported(&mut self, legacy_id: &str) {
        self.handles
            .insert(handle_of(legacy_id), legacy_id.to_string());
    }

    fn record(
        &mut self,
        op: &StoreOp,
        effect_seq: u64,
        data_seq: u64,
        digest: [u8; 32],
        rule: Option<&'static str>,
    ) {
        if self.poisoned.is_some() {
            return;
        }
        match effect_proofs_for_op(op, effect_seq, data_seq, digest, rule, &mut self.handles) {
            Ok(rows) => {
                for proof in rows {
                    let idx = self.proofs.len();
                    self.by_seq.entry(proof.effect_seq).or_insert(idx);
                    if proof.action != EffectAction::Edge {
                        let newer = self
                            .latest
                            .get(&proof.node_id)
                            .is_none_or(|&at| self.proofs[at].effect_seq <= proof.effect_seq);
                        if newer {
                            self.latest.insert(proof.node_id.clone(), idx);
                        }
                    }
                    self.proofs.push(proof);
                }
            }
            Err(StoreError::Verify(message)) => self.poisoned = Some(message),
            Err(other) => self.poisoned = Some(other.to_string()),
        }
    }

    fn check(&self) -> Result<(), StoreError> {
        match &self.poisoned {
            Some(message) => Err(StoreError::Verify(message.clone())),
            None => Ok(()),
        }
    }
}

pub(crate) fn classify_write_payload(payload: &[u8]) -> WritePayload {
    if let Some(op) = decode_exact::<StoreOp>(payload) {
        WritePayload::StoreOp(op)
    } else if let Some(node) = migration_node(payload) {
        WritePayload::ImportedNode(node)
    } else {
        WritePayload::Neither
    }
}

pub(crate) fn classify_checkpoint_payload(payload: &[u8]) -> CheckpointPayload {
    if let Some(cp) = decode_exact::<Checkpoint>(payload)
        .filter(|cp| cp.magic == strata_kernel::checkpoint::MAGIC)
    {
        CheckpointPayload::Checkpoint(cp)
    } else if let Some(edge) = migration_edge(payload) {
        CheckpointPayload::ImportedEdge(edge)
    } else {
        CheckpointPayload::Neither
    }
}

/// One recorded supersession hop, in log order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupersedeHop {
    /// Node that was superseded.
    pub id: String,
    /// Node that replaced `id`.
    pub superseded_by: String,
    /// Log seq of the frame that recorded the hop.
    pub frame_seq: u64,
    /// Chain hash of that frame.
    pub frame_hash: [u8; 32],
    /// `SupersedeNode` for a retire op, `supersedes` for a typed edge.
    pub recorded_as: &'static str,
}

/// Creating frame of one node, plus the supersession chain that names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedOrigin {
    /// Log seq of the first admitted `UpsertNode` for this id.
    pub frame_seq: u64,
    /// Frame kind byte (`KIND_STORE_WRITE`).
    pub frame_kind: u8,
    /// Chain hash of the creating frame.
    pub frame_hash: [u8; 32],
    /// `payload_blake3` of the creating frame.
    pub payload_blake3: [u8; 32],
    /// Log seq of the admitting `PROPOSE`, when present.
    pub propose_frame_seq: Option<u64>,
    /// Log seq of the admitting `GATE`, when present.
    pub gate_frame_seq: Option<u64>,
    /// Log seq of the admitting `EFFECT`, when present.
    pub effect_frame_seq: Option<u64>,
    /// Node record carried by the creating frame.
    pub record: NodeRecord,
    /// Recorded supersession hops in the connected component of this node.
    pub supersede_chain: Vec<SupersedeHop>,
}

struct AdmittedEffect {
    effect_log_seq: u64,
    propose_seq: u64,
    gate_seq: u64,
}

/// The STRATA-native memory store.
///
/// See the crate docs for the write path and determinism contract. v1 is
/// single-threaded single-writer (`!Send` through the gate-log cache).
pub struct StrataStore {
    dir: PathBuf,
    /// The log directory: `<dir>/log`, or the staged log the upgrade admits
    /// into before renaming it there.
    log_dir: PathBuf,
    log: StrataLog,
    gate_log: StrataEventLog,
    policy: Policy,
    /// Node registry (derived).
    nodes: BTreeMap<String, NodeRecord>,
    /// id -> gate-space effect seq of the WRITE that last admitted it (node
    /// or intention). Gate context ids; keeps `ReadNoReceipt` clean.
    origins: BTreeMap<String, u64>,
    /// Intention registry (derived). Not FSRS cards.
    intentions: BTreeMap<String, IntentionRecord>,
    /// Edge list in landing order (derived).
    edges: Vec<ConnectionRecord>,
    /// source id -> edge indexes (forward).
    forward: BTreeMap<String, Vec<usize>>,
    /// target id -> edge indexes (reverse).
    reverse: BTreeMap<String, Vec<usize>>,
    /// FSRS fold state (derived, kernel-canonical).
    fsrs: State,
    /// Card-fold events (reviews and imported v3 cards) in fold order with
    /// their hashes (reused by verification).
    review_events: Vec<(u64, [u8; 32], CardEvent)>,
    /// Latest review clock per card. Rebuilt from `ReviewNode` payloads and
    /// imported `FSRS_REVIEW` / `FSRS_STATE` frames. Empty when the latest
    /// review frame omitted the field.
    reviewed_at: BTreeMap<u64, i64>,
    /// Sealed checkpoints in log order.
    checkpoints: Vec<Checkpoint>,
    /// Data frames that had no admitting effect in the log (ignored).
    orphan_writes: u64,
    /// Tool call is open. `edit` may retire only a successor admitted here.
    tool_call_open: bool,
    /// Node ids admitted since [`StrataStore::begin_tool_call`].
    call_admitted: BTreeSet<String>,
    /// Admitting effect seq -> named rule, for RETIREs the context authorized.
    retire_rules: BTreeMap<u64, &'static str>,
    /// Admitted `UpsertNode` frames per node id, in log order. Derived: replay
    /// rebuilds it. Undo reads it to append a compensating record; it never
    /// rewrites or truncates the log.
    upserts: BTreeMap<String, Vec<(u64, NodeRecord)>>,
    /// Code anchors (derived). Rows of a retired node stay here; reads
    /// filter them out.
    anchors: AnchorIndex,
    /// Proved effects by receipt seq and node id (derived). Lets receipt
    /// lookups answer without re-reading the log.
    effect_index: EffectIndex,
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
        let dir = dir.as_ref();
        Self::open_log_with_policy(dir, dir.join(LOG_DIR), policy)
    }

    /// Open a store whose log lives at `log_dir` instead of `<dir>/log`.
    /// `store.meta` is still read from and written to `dir`.
    ///
    /// The v3 upgrade admits carried-over rows into the staged log
    /// (`<dir>/log.strata-staging`) with the normal write path, then renames
    /// it onto `<dir>/log`. The frames do not name their directory, so
    /// [`StrataStore::open`] on `dir` after the rename replays the same state.
    pub fn open_log_with_policy(
        dir: impl AsRef<Path>,
        log_dir: impl AsRef<Path>,
        policy: Policy,
    ) -> Result<Self, StoreError> {
        let dir = dir.as_ref().to_path_buf();
        let log_dir = log_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let log = StrataLog::open(&log_dir)?;
        let gate_log = StrataEventLog::new(log.clone())?;
        let mut store = Self {
            dir,
            log_dir,
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
            reviewed_at: BTreeMap::new(),
            checkpoints: Vec::new(),
            orphan_writes: 0,
            intentions: BTreeMap::new(),
            tool_call_open: false,
            call_admitted: BTreeSet::new(),
            retire_rules: BTreeMap::new(),
            upserts: BTreeMap::new(),
            anchors: AnchorIndex::default(),
            effect_index: EffectIndex::default(),
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
        // Importer kernel id -> v3 memory id. Imported FSRS frames name
        // cards by kernel id; the store keys cards by `handle_of(id)`.
        let mut imported_ids: HashMap<u64, String> = HashMap::new();

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
                                if let Some(propose) = propose_at.get(&effect.propose_seq) {
                                    if let Some(rule) = retire_rule_id(&propose.params_hash) {
                                        self.retire_rules.insert(gseq, rule);
                                    }
                                }
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
                // 0x20 is also a migration KIND_NODE.
                match classify_write_payload(&frame.payload) {
                    WritePayload::StoreOp(op) => {
                        let digest = hash32(&frame.payload);
                        let admitted = pending.get_mut(&digest).and_then(|queue| queue.pop_front());
                        if let Some(gseq) = admitted {
                            self.apply_op(&op, gseq, seq)?;
                            let rule = self.retire_rules.get(&gseq).copied();
                            self.effect_index.record(&op, gseq, seq, digest, rule);
                        } else {
                            self.orphan_writes += 1;
                        }
                    }
                    WritePayload::ImportedNode(node) => {
                        imported_ids.insert(node.kernel_id, node.legacy_id.clone());
                        self.effect_index.note_imported(&node.legacy_id);
                        self.apply_imported_node(&node);
                    }
                    WritePayload::Neither => self.orphan_writes += 1,
                }
            } else if frame.kind == KIND_STORE_CHECKPOINT {
                // 0x21 is also a migration KIND_EDGE.
                match classify_checkpoint_payload(&frame.payload) {
                    CheckpointPayload::Checkpoint(cp) => self.checkpoints.push(cp),
                    CheckpointPayload::ImportedEdge(edge) => self.apply_imported_edge(&edge),
                    CheckpointPayload::Neither => {}
                }
            } else if frame.kind == strata_migrate::records::KIND_SUPERSESSION {
                // A v3 `superseded_by` link, carried by the importer. The old
                // node stops being live and its successor stays the answer.
                if let Ok(link) = strata_migrate::records::decode_supersession(&frame.payload) {
                    self.apply_imported_supersession(&link);
                }
            } else if frame.kind == strata_migrate::records::KIND_FSRS_REVIEW {
                // A synthetic review from a v3 `fsrs_cards` row.
                self.apply_imported_review(&frame.payload, seq, &imported_ids)?;
            } else if frame.kind == strata_migrate::records::KIND_FSRS_STATE {
                // A v3 card carried from the `knowledge_nodes` columns.
                if let Ok(record) = strata_migrate::records::decode_fsrs_state(&frame.payload) {
                    self.apply_imported_fsrs_state(&record, seq)?;
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
                self.upserts
                    .entry(record.id.clone())
                    .or_default()
                    .push((frame_seq, record.clone()));
                self.nodes.insert(record.id.clone(), record.clone());
                if is_new {
                    // Every ingest folds one ReviewEvent ("Good") into the
                    // kernel state under the record's kernel version.
                    self.fold_review(handle, INGEST_RATING, record.kernel_id, frame_seq)?;
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
            StoreOp::ReviewNode {
                card_id,
                rating,
                reviewed_at_ms,
            } => {
                self.fold_review(*card_id, *rating, ALGO_V2, frame_seq)?;
                // Latest explicit review wins. `None` drops any earlier clock
                // so retrievability falls back to sequence distance.
                match reviewed_at_ms {
                    Some(ms) => {
                        self.reviewed_at.insert(*card_id, *ms);
                    }
                    None => {
                        self.reviewed_at.remove(card_id);
                    }
                }
            }
            StoreOp::UpsertIntentions { records } => {
                for record in records {
                    self.origins.insert(record.id.clone(), gate_effect_seq);
                    self.intentions.insert(record.id.clone(), record.clone());
                }
            }
            // Anchors are not origins: a receipt replay of a memory resolves
            // through the node's own write, never through its anchors.
            StoreOp::RecordAnchors { anchors } => self.anchors.record(anchors),
            StoreOp::ReplaceAnchors { node_id, anchors } => {
                self.anchors.replace(node_id, anchors);
            }
            StoreOp::RecordAnchorVerdict {
                anchor_id,
                status,
                checked_at_ms,
            } => self.anchors.verdict(anchor_id, status, *checked_at_ms),
        }
        Ok(())
    }

    /// Imported node. Kind stays on the edge records; this only fills the registry.
    fn apply_imported_node(&mut self, node: &strata_migrate::NodeRecord) {
        // The importer keys every carried v3 column as `<table>.<column>`.
        let legacy = |column: &str| {
            let qualified = format!("knowledge_nodes.{column}");
            node.legacy
                .iter()
                .find(|(key, _)| *key == qualified)
                .map(|(_, value)| value.as_str())
        };
        // v3 kept each memory's project namespace in `scope`; keep it, or
        // every project's memories would merge into `user`.
        let scope = legacy("scope")
            .map(str::trim)
            .filter(|scope| !scope.is_empty() && *scope != "NULL")
            .unwrap_or("user")
            .to_string();
        // A v3-suppressed memory was out of retrieval. Keep it that way: the
        // record stays on the log and in the backup, but it is not live.
        let suppressed = legacy("suppression_count")
            .and_then(|count| count.trim().parse::<i64>().ok())
            .is_some_and(|count| count > 0);
        let record = NodeRecord {
            id: node.legacy_id.clone(),
            kernel_id: ALGO_V2,
            scope,
            content: node.content.clone(),
            node_type: if node.node_type.is_empty() {
                DEFAULT_NODE_TYPE.to_string()
            } else {
                node.node_type.clone()
            },
            tags: node.tags.clone(),
            created_at_ms: node.created_ms,
            valid_from_ms: node.created_ms,
            valid_until_ms: VALID_FOREVER_MS,
            superseded_by: suppressed.then(|| IMPORTED_SUPPRESSED_MARKER.to_string()),
            source: node.source.as_ref().map(|key| crate::types::SourceKey {
                system: key.system.clone(),
                project: key.project.clone(),
                id: key.id.clone(),
            }),
            source_updated_at_ms: node.source_updated_at_ms,
        };
        self.nodes.insert(record.id.clone(), record);
    }

    /// Imported v3 supersession: both ends must be imported nodes.
    fn apply_imported_supersession(&mut self, link: &strata_migrate::records::SupersessionRecord) {
        if !self.nodes.contains_key(&link.superseded_by_legacy_id) {
            return;
        }
        if let Some(record) = self.nodes.get_mut(&link.superseded_legacy_id) {
            record.superseded_by = Some(link.superseded_by_legacy_id.clone());
        }
    }

    /// Imported edge. `link_type` is copied, including `legacy_inferred`.
    fn apply_imported_edge(&mut self, edge: &strata_migrate::EdgeRecord) {
        let milli = (strata_kernel::canonical::from_q32_32(edge.strength_q32) * 1000.0).round();
        let strength_milli = if milli.is_finite() && milli >= 0.0 {
            milli as i64
        } else {
            0
        };
        let record = ConnectionRecord {
            source_id: edge.source_legacy_id.clone(),
            target_id: edge.target_legacy_id.clone(),
            strength_milli,
            link_type: edge.link_type.clone(),
            meta_sha: None,
            created_at_ms: edge.created_ms,
            activation_count: i64::from(edge.activation_count),
        };
        let idx = self.edges.len();
        self.forward
            .entry(record.source_id.clone())
            .or_default()
            .push(idx);
        self.reverse
            .entry(record.target_id.clone())
            .or_default()
            .push(idx);
        self.edges.push(record);
    }

    /// Imported `FSRS_REVIEW` frame (a v3 `fsrs_cards` rating series).
    ///
    /// The importer names the card by its kernel id; the fold keys it by
    /// the node's handle, like every admitted review. The frame's seq is the
    /// event seq. The payload's `reviewed_at_ms` sets the review clock as a
    /// `ReviewNode` would. A review naming no imported node is ignored.
    fn apply_imported_review(
        &mut self,
        payload: &[u8],
        frame_seq: u64,
        imported_ids: &HashMap<u64, String>,
    ) -> Result<(), StoreError> {
        let (Ok(event), Ok(reviewed_at_ms)) = (
            strata_migrate::records::decode_review(payload),
            strata_migrate::records::decode_reviewed_at_ms(payload),
        ) else {
            return Ok(());
        };
        let Some(id) = imported_ids.get(&event.card_id) else {
            return Ok(());
        };
        if !self.nodes.contains_key(id) {
            return Ok(());
        }
        let handle = handle_of(id);
        self.fold_review(handle, event.rating, ALGO_V2, frame_seq)?;
        match reviewed_at_ms {
            Some(ms) => {
                self.reviewed_at.insert(handle, ms);
            }
            None => {
                self.reviewed_at.remove(&handle);
            }
        }
        Ok(())
    }

    /// Imported `FSRS_STATE` frame: the v3 card itself.
    ///
    /// Folds a [`CardEvent::Import`] at the frame's seq and sets the review
    /// clock to v3's `last_accessed`. Only a node with no card yet takes
    /// it, so a later frame never rewinds a card that already folded
    /// reviews. A record for an unknown node, or from another wire version,
    /// is ignored.
    fn apply_imported_fsrs_state(
        &mut self,
        record: &strata_migrate::FsrsStateRecord,
        frame_seq: u64,
    ) -> Result<(), StoreError> {
        if record.record_version != strata_migrate::RECORD_VERSION
            || !self.nodes.contains_key(&record.legacy_id)
        {
            return Ok(());
        }
        let handle = handle_of(&record.legacy_id);
        if self.fsrs.cards.contains_key(&handle) {
            return Ok(());
        }
        self.fold_card(
            CardEvent::Import(ImportedCard {
                card_id: handle,
                event_seq: frame_seq,
                stability_q: record.stability_q,
                difficulty_q: record.difficulty_q,
                review_count: record.review_count,
                lapse_count: record.lapse_count,
                phase: record.phase,
            }),
            ALGO_V2,
        )?;
        self.reviewed_at.insert(handle, record.reviewed_at_ms);
        Ok(())
    }

    fn fold_review(
        &mut self,
        card_id: u64,
        rating: u8,
        kernel_id: u32,
        event_seq: u64,
    ) -> Result<(), StoreError> {
        self.fold_card(
            CardEvent::Review(ReviewEvent {
                card_id,
                rating,
                event_seq,
            }),
            kernel_id,
        )
    }

    fn fold_card(&mut self, event: CardEvent, kernel_id: u32) -> Result<(), StoreError> {
        let event_hash = hash32(&borsh_vec(&event)?);
        let kernel = Kernel::<CardEvent>::for_version(kernel_id)
            .map_err(|e| StoreError::Verify(e.to_string()))?;
        kernel.apply(&mut self.fsrs, &event);
        self.review_events.push((event.seq(), event_hash, event));
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
        self.admit_write_with_params(op, action_kind_code, context, None)
    }

    /// `params_hash` overrides the default (the op hash) when a named RETIRE
    /// rule authorized this admission. The override is the rule-id digest,
    /// never a digest of node content.
    fn admit_write_with_params(
        &mut self,
        op: StoreOp,
        action_kind_code: u8,
        context: Vec<u64>,
        params_hash: Option<[u8; 32]>,
    ) -> Result<(u64, u64), StoreError> {
        // The gate frames below have no error channel: refuse up front when
        // the log has already found damage in acked history.
        self.log.ensure_writable()?;
        let fresh_id = match &op {
            StoreOp::UpsertNode { record }
                if self.tool_call_open && !self.nodes.contains_key(&record.id) =>
            {
                Some(record.id.clone())
            }
            _ => None,
        };
        let op_bytes = borsh_vec(&op)?;
        let action_hash = hash32(&op_bytes);
        let params_hash = params_hash.unwrap_or(action_hash);

        let mut runtime = GateRuntime::new(self.gate_log.clone(), self.policy.clone());
        let propose = Propose {
            action_hash,
            action_kind: action_kind_code,
            params_hash,
            context,
        };
        // A full volume refuses a frame instead of aborting; each gate step
        // is checked so a refused write stops here with the error.
        let _ = self.gate_log.take_refusal();
        let propose_ack = runtime.commit_propose(propose);
        self.refused_append()?;
        let gate_ack = runtime.commit_gate(propose_ack.seq);
        self.refused_append()?;
        let gate_ack = gate_ack.map_err(|e| StoreError::Gate(e.to_string()))?;
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
        let effect_ack = runtime.commit_effect(effect);
        self.refused_append()?;
        let effect_ack: SeqAck = effect_ack.map_err(|r| StoreError::Rejected(r.to_string()))?;
        if let Some(rule) = retire_rule_id(&params_hash) {
            self.retire_rules.insert(effect_ack.seq, rule);
        }

        // The data frame lands only after an admitted effect cites its digest.
        let data_acks = self.log.append_batch(vec![(KIND_STORE_WRITE, op_bytes)])?;
        let data_seq = data_acks[0].seq;

        self.apply_op(&op, effect_ack.seq, data_seq)?;
        let rule = self.retire_rules.get(&effect_ack.seq).copied();
        self.effect_index
            .record(&op, effect_ack.seq, data_seq, action_hash, rule);
        if let Some(id) = fresh_id {
            self.call_admitted.insert(id);
        }
        Ok((effect_ack.seq, data_seq))
    }

    /// Surface a gate frame the log refused for lack of space.
    fn refused_append(&self) -> Result<(), StoreError> {
        match self.gate_log.take_refusal() {
            Some(e) => Err(StoreError::Log(e)),
            None => Ok(()),
        }
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
        self.ingest_in_scope_with_receipt(input, scope)
            .map(|(id, _)| id)
    }

    /// [`Self::ingest_in_scope`], also returning the gate-space seq of the
    /// admitting EFFECT (the `eff-` receipt id, see [`effect_receipt_id`]).
    pub fn ingest_in_scope_with_receipt(
        &mut self,
        input: IngestInput,
        scope: &str,
    ) -> Result<(String, u64), StoreError> {
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
            source: input.source.clone(),
            source_updated_at_ms: input.source_updated_at_ms,
        };
        // A brand-new fact references nothing yet: empty context.
        let (effect_seq, _) = self.admit_write(
            StoreOp::UpsertNode { record },
            action_kind::WRITE,
            Vec::new(),
        )?;
        Ok((id, effect_seq))
    }

    /// Insert or replace intentions through one admitted write.
    ///
    /// Returns the gate-space effect seq. The caller has already checked a
    /// compare-and-swap when the batch is a check claim; this method does
    /// not re-read the previous rows.
    pub fn upsert_intentions(&mut self, records: Vec<IntentionRecord>) -> Result<u64, StoreError> {
        if records.is_empty() {
            return Err(StoreError::InvalidInput("intention batch is empty".into()));
        }
        let mut seen = BTreeSet::new();
        for record in &records {
            if record.id.is_empty() {
                return Err(StoreError::InvalidInput(
                    "intention id must not be empty".into(),
                ));
            }
            if !seen.insert(record.id.as_str()) {
                let id = &record.id;
                return Err(StoreError::InvalidInput(format!(
                    "duplicate intention id {id}"
                )));
            }
        }
        let ids: Vec<&str> = records.iter().map(|record| record.id.as_str()).collect();
        let context = self.context_for(&ids);
        let (effect_seq, _) = self.admit_write(
            StoreOp::UpsertIntentions { records },
            action_kind::WRITE,
            context,
        )?;
        Ok(effect_seq)
    }

    /// One intention by id.
    pub fn get_intention(&self, id: &str) -> Option<IntentionRecord> {
        self.intentions.get(id).cloned()
    }

    /// Every intention, in id order.
    pub fn intentions(&self) -> Vec<IntentionRecord> {
        self.intentions.values().cloned().collect()
    }

    /// A node that exists and is not retired, or why not.
    fn require_live_node(&self, id: &str) -> Result<&NodeRecord, StoreError> {
        let record = self.require_node(id)?;
        if !record.is_live() {
            return Err(StoreError::InvalidInput(format!("node {id} is retired")));
        }
        Ok(record)
    }

    /// Shape checks shared by record and replace. Writes nothing.
    fn check_anchor_batch(&self, anchors: &[AnchorRecord]) -> Result<(), StoreError> {
        if anchors.is_empty() {
            return Err(StoreError::InvalidInput("anchor batch is empty".into()));
        }
        let mut seen = BTreeSet::new();
        for anchor in anchors {
            if anchor.id.is_empty() || anchor.node_id.is_empty() || anchor.file_path.is_empty() {
                return Err(StoreError::InvalidInput(
                    "anchor id, node id and file path must not be empty".into(),
                ));
            }
            if !seen.insert(anchor.id.as_str()) {
                let id = &anchor.id;
                return Err(StoreError::InvalidInput(format!(
                    "duplicate anchor id {id}"
                )));
            }
            self.require_live_node(&anchor.node_id)?;
        }
        Ok(())
    }

    /// Insert or replace code anchors by anchor id through one admitted
    /// write. Every anchor must name a live node. Returns the gate-space
    /// effect seq; a refused batch appends nothing.
    pub fn record_anchors(&mut self, anchors: Vec<AnchorRecord>) -> Result<u64, StoreError> {
        self.check_anchor_batch(&anchors)?;
        let nodes: BTreeSet<&str> = anchors
            .iter()
            .map(|anchor| anchor.node_id.as_str())
            .collect();
        let context = self.context_for(&nodes.into_iter().collect::<Vec<_>>());
        let (effect_seq, _) = self.admit_write(
            StoreOp::RecordAnchors { anchors },
            action_kind::WRITE,
            context,
        )?;
        Ok(effect_seq)
    }

    /// Replace every anchor of `node_id` with `anchors` through one admitted
    /// write. The memory itself is not rewritten. Every row must name
    /// `node_id`, and the node must be live.
    pub fn replace_anchors(
        &mut self,
        node_id: &str,
        anchors: Vec<AnchorRecord>,
    ) -> Result<u64, StoreError> {
        self.require_live_node(node_id)?;
        self.check_anchor_batch(&anchors)?;
        if anchors.iter().any(|anchor| anchor.node_id != node_id) {
            return Err(StoreError::InvalidInput(format!(
                "every replacement anchor must name node {node_id}"
            )));
        }
        let context = self.context_for(&[node_id]);
        let (effect_seq, _) = self.admit_write(
            StoreOp::ReplaceAnchors {
                node_id: node_id.to_string(),
                anchors,
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(effect_seq)
    }

    /// Cache one anchor's latest verdict. `Ok(None)` and nothing appended
    /// when the anchor is unknown or its node is retired (the SQLite store's
    /// `UPDATE` of zero rows).
    pub fn record_anchor_verdict(
        &mut self,
        anchor_id: &str,
        status: &str,
        checked_at_ms: i64,
    ) -> Result<Option<u64>, StoreError> {
        if status.is_empty() {
            return Err(StoreError::InvalidInput(
                "anchor verdict must not be empty".into(),
            ));
        }
        let Some(node_id) = self.anchor(anchor_id).map(|anchor| anchor.node_id) else {
            return Ok(None);
        };
        let context = self.context_for(&[node_id.as_str()]);
        let (effect_seq, _) = self.admit_write(
            StoreOp::RecordAnchorVerdict {
                anchor_id: anchor_id.to_string(),
                status: status.to_string(),
                checked_at_ms,
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(Some(effect_seq))
    }

    /// Anchors of a live node, ordered by file path, start line, then id.
    /// A retired or unknown node has none.
    pub fn anchors_for(&self, node_id: &str) -> Vec<AnchorRecord> {
        if !self.nodes.get(node_id).is_some_and(NodeRecord::is_live) {
            return Vec::new();
        }
        let mut rows = self.anchors.rows_of(node_id);
        rows.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.start_line.cmp(&b.start_line))
                .then(a.id.cmp(&b.id))
        });
        rows
    }

    /// One anchor by id, when its node is live.
    pub fn anchor(&self, anchor_id: &str) -> Option<AnchorRecord> {
        let row = self.anchors.get(anchor_id)?;
        self.nodes
            .get(&row.node_id)
            .is_some_and(NodeRecord::is_live)
            .then(|| row.clone())
    }

    /// Gate-space effect seq of the write that created `id`, if it exists.
    pub fn origin_seq(&self, id: &str) -> Option<u64> {
        self.origins.get(id).copied()
    }

    /// Origin of `id` read from the log: the first admitted `UpsertNode`
    /// frame, the gate frames that admitted it, and every recorded
    /// supersession hop in that node's chain.
    ///
    /// `None` when no admitted upsert names `id`. The log records no actor
    /// and no source envelope; callers surface those as absent.
    pub fn recorded_origin(&self, id: &str) -> Result<Option<RecordedOrigin>, StoreError> {
        let frames = self.log.read_frames(1)?;
        let mut propose_at: HashMap<u64, Propose> = HashMap::new();
        let mut gates_for: HashMap<u64, Vec<(u64, GateRecord)>> = HashMap::new();
        let mut propose_log: HashMap<u64, u64> = HashMap::new();
        let mut gate_log: HashMap<u64, u64> = HashMap::new();
        let mut pending: HashMap<[u8; 32], VecDeque<AdmittedEffect>> = HashMap::new();
        let mut gate_seq_counter: u64 = 0;
        let mut origin: Option<RecordedOrigin> = None;
        let mut hops: Vec<SupersedeHop> = Vec::new();

        for frame in &frames {
            let seq = frame.seq;
            if let Some(kind) = RecordKind::from_u8(frame.kind) {
                let gseq = gate_seq_counter;
                gate_seq_counter += 1;
                match kind {
                    RecordKind::Propose => {
                        propose_log.insert(gseq, seq);
                        if let Ok(propose) = Propose::try_from_slice(&frame.payload) {
                            propose_at.insert(gseq, propose);
                        }
                    }
                    RecordKind::Gate => {
                        gate_log.insert(gseq, seq);
                        if let Ok(gate) = GateRecord::try_from_slice(&frame.payload) {
                            gates_for
                                .entry(gate.propose_seq)
                                .or_default()
                                .push((gseq, gate));
                        }
                    }
                    RecordKind::Effect => {
                        if let Ok(effect) = EffectRecord::try_from_slice(&frame.payload) {
                            let covering = propose_at
                                .get(&effect.propose_seq)
                                .is_some_and(|propose| propose.action_hash == effect.action_hash);
                            let admitting =
                                gates_for.get(&effect.propose_seq).is_some_and(|gates| {
                                    gates.iter().any(|(gate_seq, gate)| {
                                        *gate_seq == effect.gate_seq
                                            && gate.verdict == Verdict::Allow
                                            && *gate_seq < gseq
                                    })
                                });
                            if covering && admitting {
                                pending.entry(effect.payload_digest).or_default().push_back(
                                    AdmittedEffect {
                                        effect_log_seq: seq,
                                        propose_seq: effect.propose_seq,
                                        gate_seq: effect.gate_seq,
                                    },
                                );
                            }
                        }
                    }
                    _ => {}
                }
                continue;
            }
            if frame.kind != KIND_STORE_WRITE {
                continue;
            }
            let digest = hash32(&frame.payload);
            let admitted = pending.get_mut(&digest).and_then(|queue| queue.pop_front());
            let Some(admitted) = admitted else {
                continue;
            };
            let Ok(op) = StoreOp::try_from_slice(&frame.payload) else {
                continue;
            };
            match op {
                StoreOp::UpsertNode { record } if record.id == id && origin.is_none() => {
                    origin = Some(RecordedOrigin {
                        frame_seq: seq,
                        frame_kind: frame.kind,
                        frame_hash: frame.frame_hash,
                        payload_blake3: frame.payload_blake3,
                        propose_frame_seq: propose_log.get(&admitted.propose_seq).copied(),
                        gate_frame_seq: gate_log.get(&admitted.gate_seq).copied(),
                        effect_frame_seq: Some(admitted.effect_log_seq),
                        record,
                        supersede_chain: Vec::new(),
                    });
                }
                StoreOp::SupersedeNode {
                    id: sid,
                    superseded_by,
                } => {
                    hops.push(SupersedeHop {
                        id: sid,
                        superseded_by,
                        frame_seq: seq,
                        frame_hash: frame.frame_hash,
                        recorded_as: "SupersedeNode",
                    });
                }
                StoreOp::SaveEdge { edge } if edge.link_type == EdgeKind::Supersedes.as_str() => {
                    hops.push(SupersedeHop {
                        id: edge.target_id,
                        superseded_by: edge.source_id,
                        frame_seq: seq,
                        frame_hash: frame.frame_hash,
                        recorded_as: "supersedes",
                    });
                }
                _ => {}
            }
        }

        let Some(mut origin) = origin else {
            return Ok(None);
        };
        origin.supersede_chain = supersede_component(&hops, id);
        Ok(Some(origin))
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
    pub fn save_connection(&mut self, connection: &ConnectionRecord) -> Result<u64, StoreError> {
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
        let (effect_seq, _) = self.admit_write(
            StoreOp::SaveEdge {
                edge: connection.clone(),
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(effect_seq)
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

    /// Open a tool call. `edit` may retire a node only when its successor was
    /// admitted after this and before [`Self::end_tool_call`].
    pub fn begin_tool_call(&mut self) {
        self.tool_call_open = true;
        self.call_admitted.clear();
    }

    /// Close the tool call and drop the successor set.
    pub fn end_tool_call(&mut self) {
        self.tool_call_open = false;
        self.call_admitted.clear();
    }

    /// Mark `id` as superseded by `superseded_by`.
    ///
    /// Routed as a `RETIRE` with no rule id. The default policy holds it.
    pub fn supersede(&mut self, id: &str, superseded_by: &str) -> Result<(), StoreError> {
        self.admit_supersede(id, superseded_by, &AdmissionContext::default())
            .map(|_| ())
    }

    /// RETIRE `id` in favor of `superseded_by` when `ctx` carries a named rule.
    ///
    /// The default policy allows exactly `edit` (successor admitted in this
    /// tool call), `suppress`, `intentions`, and `purge` (`confirm` set).
    /// Every other RETIRE is held. An allowed RETIRE returns an `eff-`
    /// receipt naming the rule.
    pub fn retire(
        &mut self,
        id: &str,
        superseded_by: &str,
        ctx: &AdmissionContext,
    ) -> Result<RetireReceipt, StoreError> {
        let (seq, rule_id) = self.admit_supersede(id, superseded_by, ctx)?;
        Ok(RetireReceipt {
            receipt_id: effect_receipt_id(seq),
            rule_id,
            effect_seq: seq,
        })
    }

    /// Receipt for an allowed RETIRE, including one rebuilt by replay.
    pub fn retire_receipt(&self, effect_seq: u64) -> Option<RetireReceipt> {
        self.retire_rules
            .get(&effect_seq)
            .map(|rule| RetireReceipt {
                receipt_id: effect_receipt_id(effect_seq),
                rule_id: Some(*rule),
                effect_seq,
            })
    }

    fn admit_supersede(
        &mut self,
        id: &str,
        superseded_by: &str,
        ctx: &AdmissionContext,
    ) -> Result<(u64, Option<&'static str>), StoreError> {
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
        let successor_in_call = self.call_admitted.contains(superseded_by);
        let rule = named_retire_rule(ctx.rule_id.as_deref(), ctx.confirm, successor_in_call);
        let params = rule.map(rule_id_hash);
        let context = self.context_for(&[id, superseded_by]);
        let (effect_seq, _) = self.admit_write_with_params(
            StoreOp::SupersedeNode {
                id: id.to_string(),
                superseded_by: superseded_by.to_string(),
            },
            action_kind::RETIRE,
            context,
            params,
        )?;
        Ok((effect_seq, rule))
    }

    /// All (superseded, superseder) pairs, ordered by superseded id.
    pub fn supersession_pairs(&self) -> Vec<(String, String)> {
        self.nodes
            .iter()
            .filter_map(|(id, r)| r.superseded_by.clone().map(|by| (id.clone(), by)))
            .collect()
    }

    /// Fold an explicit FSRS review for a node (rating 1..=4).
    ///
    /// Returns the gate-space effect seq of the admitted review.
    /// `reviewed_at_ms` is the admission clock: unix epoch milliseconds.
    pub fn review(&mut self, id: &str, rating: u8) -> Result<u64, StoreError> {
        self.review_at(id, rating, Some(admission_now_ms()))
    }

    /// Same as [`Self::review`] with a caller-supplied clock.
    ///
    /// `None` is written as `borsh` option tag `0`, not an omitted field.
    /// Returns the gate-space effect seq of the admitted review.
    pub fn review_at(
        &mut self,
        id: &str,
        rating: u8,
        reviewed_at_ms: Option<i64>,
    ) -> Result<u64, StoreError> {
        self.require_node(id)?;
        if !(1..=4).contains(&rating) {
            return Err(StoreError::InvalidInput("rating must be 1..=4".into()));
        }
        let context = self.context_for(&[id]);
        let (effect_seq, _) = self.admit_write(
            StoreOp::ReviewNode {
                card_id: handle_of(id),
                rating,
                reviewed_at_ms,
            },
            action_kind::WRITE,
            context,
        )?;
        Ok(effect_seq)
    }

    /// Replace `id` with a successor, then retire `id` under rule `edit`.
    ///
    /// Lands `UpsertNode` then `SupersedeNode` inside one tool call. The
    /// successor is a new ingest (its own FSRS card). The predecessor's card
    /// and bytes stay on the retired node.
    ///
    /// A context that is not exactly rule `edit` returns [`StoreError::Held`]
    /// and writes nothing. An upsert cannot be rolled back, so a RETIRE that
    /// would hold must not be preceded by a live successor.
    pub fn edit(
        &mut self,
        id: &str,
        content: &str,
        ctx: &AdmissionContext,
    ) -> Result<(String, RetireReceipt), StoreError> {
        if content.trim().is_empty() {
            return Err(StoreError::InvalidInput("content must not be empty".into()));
        }
        let old = self.require_node(id)?.clone();
        if old.superseded_by.is_some() {
            return Err(StoreError::InvalidInput(format!(
                "node {id} is already superseded"
            )));
        }
        if named_retire_rule(ctx.rule_id.as_deref(), ctx.confirm, true) != Some(RULE_EDIT) {
            return Err(StoreError::Held {
                propose_seq: self.gate_log.gate_frame_count(),
            });
        }
        let opened = !self.tool_call_open;
        if opened {
            self.begin_tool_call();
        }
        let result = (|| {
            let input = IngestInput {
                content: content.to_string(),
                source: old.source.clone(),
                source_updated_at_ms: old.source_updated_at_ms,
                node_type: old.node_type.clone(),
                tags: old.tags.clone(),
                created_at_ms: Some(old.created_at_ms),
                valid_from_ms: Some(old.valid_from_ms),
                valid_until_ms: Some(old.valid_until_ms),
            };
            let successor = self.ingest_in_scope(input, &old.scope)?;
            let receipt = self.retire(id, &successor, ctx)?;
            Ok((successor, receipt))
        })();
        if opened {
            self.end_tool_call();
        }
        result
    }

    /// Review events in fold order (imported cards left out). The kernel
    /// test replays these independently.
    #[cfg(test)]
    pub(crate) fn review_events(&self) -> Vec<ReviewEvent> {
        self.review_events
            .iter()
            .filter_map(|(_, _, event)| match event {
                CardEvent::Review(review) => Some(*review),
                CardEvent::Import(_) => None,
            })
            .collect()
    }

    /// Every card-fold event (reviews and imported cards) in fold order.
    #[cfg(test)]
    pub(crate) fn card_events(&self) -> Vec<CardEvent> {
        self.review_events
            .iter()
            .map(|(_, _, event)| *event)
            .collect()
    }

    /// Every node, intention and code-anchor effect proved from the log, in
    /// effect-seq order.
    ///
    /// The log read is strict: every segment's hash chain and every sealed
    /// segment's signed trailer is checked, and damage anywhere is an error
    /// rather than a shorter answer. Each effect must cite an Allow gate and
    /// a data frame with the same payload digest.
    pub fn prove_effects(&self) -> Result<Vec<EffectProof>, StoreError> {
        self.log.verify_tail()?;
        let frames = self.log.read_frames(1)?;
        let mut propose_at: HashMap<u64, Propose> = HashMap::new();
        let mut gates_for: HashMap<u64, Vec<(u64, GateRecord)>> = HashMap::new();
        let mut pending: HashMap<[u8; 32], VecDeque<(u64, Option<&'static str>)>> = HashMap::new();
        let mut handles: HashMap<u64, String> = HashMap::new();
        let mut proofs = Vec::new();
        let mut gate_seq_counter: u64 = 0;

        for frame in frames {
            if frame.payload_blake3 != strata::payload_blake3(frame.kind, &frame.payload) {
                return Err(StoreError::Verify(format!(
                    "frame {} payload blake3 does not match its bytes",
                    frame.seq
                )));
            }
            if let Some(kind) = RecordKind::from_u8(frame.kind) {
                let gseq = gate_seq_counter;
                gate_seq_counter += 1;
                match kind {
                    RecordKind::Propose => {
                        if let Ok(propose) = Propose::try_from_slice(&frame.payload) {
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
                        if let Ok(effect) = EffectRecord::try_from_slice(&frame.payload) {
                            let covering = propose_at
                                .get(&effect.propose_seq)
                                .is_some_and(|propose| propose.action_hash == effect.action_hash);
                            let allowed = gates_for.get(&effect.propose_seq).is_some_and(|gates| {
                                gates.iter().any(|(gate_seq, gate)| {
                                    *gate_seq == effect.gate_seq
                                        && gate.verdict == Verdict::Allow
                                        && *gate_seq < gseq
                                })
                            });
                            if covering && allowed && effect.action_hash == effect.payload_digest {
                                let rule = propose_at
                                    .get(&effect.propose_seq)
                                    .and_then(|propose| retire_rule_id(&propose.params_hash));
                                pending
                                    .entry(effect.payload_digest)
                                    .or_default()
                                    .push_back((gseq, rule));
                            }
                        }
                    }
                    _ => {}
                }
            } else if frame.kind == KIND_STORE_WRITE {
                let digest = hash32(&frame.payload);
                let Some((effect_seq, rule)) =
                    pending.get_mut(&digest).and_then(|queue| queue.pop_front())
                else {
                    // Not an admitted write: an imported v3 node is still a
                    // card later reviews may name (promote/demote on it).
                    if let Some(node) = migration_node(&frame.payload) {
                        handles.insert(handle_of(&node.legacy_id), node.legacy_id);
                    }
                    continue;
                };
                let Some(op) = StoreOp::try_from_slice(&frame.payload).ok() else {
                    return Err(StoreError::Verify(format!(
                        "admitted frame {} is not a StoreOp",
                        frame.seq
                    )));
                };
                proofs.extend(effect_proofs_for_op(
                    &op,
                    effect_seq,
                    frame.seq,
                    digest,
                    rule,
                    &mut handles,
                )?);
            }
        }
        Ok(proofs)
    }

    /// The proved effect at `effect_seq`, if the log admits one.
    ///
    /// Answered from the effect index built when the log was replayed and
    /// extended by each admitted write, so the cost does not grow with the
    /// log. [`StrataStore::prove_effects`] is the full re-verifying scan.
    pub fn effect_by_seq(&self, effect_seq: u64) -> Result<Option<EffectProof>, StoreError> {
        self.effect_index.check()?;
        Ok(self
            .effect_index
            .by_seq
            .get(&effect_seq)
            .map(|&idx| self.effect_index.proofs[idx].clone()))
    }

    /// The latest proved effect for `node_id` (a node or an intention id).
    ///
    /// Answered from the effect index, like [`StrataStore::effect_by_seq`].
    /// Edge effects are excluded: they must not hide the source node's own
    /// receipt. Use [`Self::edge_proofs`] for those.
    pub fn latest_effect(&self, node_id: &str) -> Result<Option<EffectProof>, StoreError> {
        self.effect_index.check()?;
        Ok(self
            .effect_index
            .latest
            .get(node_id)
            .map(|&idx| self.effect_index.proofs[idx].clone()))
    }

    /// Proved `SaveEdge` effects from `source` to `target` of `link_type`,
    /// oldest first.
    ///
    /// The effect index keeps edge proofs out of [`Self::latest_effect`].
    /// This scans that index. Replay rebuilds it, so the answer matches a
    /// full log read.
    pub fn edge_proofs(
        &self,
        source: &str,
        target: &str,
        link_type: &str,
    ) -> Result<Vec<EffectProof>, StoreError> {
        self.effect_index.check()?;
        Ok(self
            .effect_index
            .proofs
            .iter()
            .filter(|proof| {
                proof.action == EffectAction::Edge
                    && proof.node_id == source
                    && proof
                        .edge
                        .as_ref()
                        .is_some_and(|(got_target, kind)| got_target == target && kind == link_type)
            })
            .cloned()
            .collect())
    }

    /// Review clock recorded on the latest explicit review of `id`.
    pub fn reviewed_at_ms(&self, id: &str) -> Option<i64> {
        self.reviewed_at.get(&handle_of(id)).copied()
    }

    /// Current FSRS scheduling card for a node (derived state, cloned).
    pub fn card_state(&self, id: &str) -> Option<strata_kernel::fsrs::CardState> {
        self.fsrs.cards.get(&handle_of(id)).cloned()
    }

    /// Retrievability of a node at the admission clock — derived on read,
    /// never stored, and reads append nothing (v1).
    ///
    /// An explicit review with `reviewed_at_ms` measures elapsed whole days
    /// from that timestamp; a card still at its ingest review measures from
    /// the node's creation time. Otherwise sequence distance from `last_seq`
    /// to the log head is used.
    pub fn retrievability(&self, id: &str) -> Result<Option<f64>, StoreError> {
        self.retrievability_at(id, admission_now_ms())
    }

    /// [`Self::retrievability`] evaluated at `as_of_ms` instead of now.
    pub fn retrievability_at(&self, id: &str, as_of_ms: i64) -> Result<Option<f64>, StoreError> {
        let handle = handle_of(id);
        let Some(card) = self.fsrs.cards.get(&handle) else {
            return Ok(None);
        };
        // A card still at its ingest review has no explicit review clock;
        // the node's recorded creation time stands in, so retention follows
        // elapsed time rather than log writes. A node with no recorded
        // creation time (0) keeps the sequence-distance fallback.
        let clock = self.reviewed_at.get(&handle).copied().or_else(|| {
            (card.review_count <= 1)
                .then(|| self.nodes.get(id).map(|record| record.created_at_ms))
                .flatten()
                .filter(|ms| *ms > 0)
        });
        FsrsFold::retrievability_at_review(
            card,
            clock,
            as_of_ms,
            self.log.head().last_acked_seq,
            ALGO_V2,
        )
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
    /// strata-kernel verifier, anchored by `store.meta`. Runs on every open.
    /// A log that already has frames and no anchor file is an error.
    pub fn verify_checkpoint_chain(&self) -> Result<(), StoreError> {
        let anchor = match (self.checkpoints.last(), self.read_meta()) {
            (None, None) => return Ok(()),
            (None, Some(_)) => {
                return Err(StoreError::Verify(
                    "store.meta exists but the log carries no checkpoint".into(),
                ));
            }
            (Some(_), None) => {
                return Err(StoreError::Verify(
                    "store.meta is missing but the log has frames".into(),
                ));
            }
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
        let events: Vec<(u64, [u8; 32], CardEvent)> = self
            .review_events
            .iter()
            .filter(|(seq, _, _)| *seq <= last_log_seq)
            .cloned()
            .collect();
        verify_with_head(&self.checkpoints, anchor, events.into_iter()).map_err(StoreError::from)
    }

    /// Back the store up: verify every segment, seal the active segment
    /// (signed trailer; a fresh active segment is rolled so the live store
    /// keeps appending), then copy the sealed segments plus `head.state`, the
    /// signing key, and the anchor file into `dest`. The copy opens as a
    /// store via [`StrataStore::open`].
    ///
    /// A log that fails verification is not backed up: the call fails before
    /// anything is sealed or copied.
    ///
    /// `strata.lock` is deliberately NOT copied (it names this process).
    pub fn backup_to(&self, dest: impl AsRef<Path>) -> Result<(), StoreError> {
        let dest = dest.as_ref();
        self.log.verify_log()?;
        self.log.seal()?;
        let dest_log = dest.join(LOG_DIR);
        create_private_dir_all(&dest_log)?;
        set_private_dir(dest)?;
        for entry in std::fs::read_dir(&self.log_dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name == "strata.lock" {
                continue;
            }
            copy_private_file(&path, &dest_log.join(name))?;
        }
        if self.meta_path().exists() {
            copy_private_file(&self.meta_path(), &dest.join(META_NAME))?;
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
    /// count, explicit review clocks, intentions, code anchors). Two stores
    /// replaying the same log produce the same digest.
    pub fn state_digest(&self) -> [u8; 32] {
        let digest = StateDigest {
            nodes: self.nodes.iter().map(|(k, v)| (k.as_str(), v)).collect(),
            origins: self.origins.iter().map(|(k, v)| (k.as_str(), *v)).collect(),
            edges: &self.edges,
            fsrs_root: strata_kernel::checkpoint::state_root(&self.fsrs),
            checkpoints: self.checkpoints.iter().map(checkpoint_hash).collect(),
            orphan_writes: self.orphan_writes,
            reviewed_at: self.reviewed_at.iter().map(|(k, v)| (*k, *v)).collect(),
            intentions: self
                .intentions
                .iter()
                .map(|(id, record)| (id.as_str(), record))
                .collect(),
            anchors: self.anchors.iter().collect(),
        };
        hash32(&borsh_vec(&digest).expect("state digest serialization is infallible"))
    }

    /// Re-read the log, refuse a broken frame, and fold a fresh copy of the
    /// derived maps. Does not append, truncate, or seal. The fold is the same
    /// `replay` live admits use, so intentions, review clocks, and imported
    /// frames stay in the digest.
    pub fn refold(&self) -> Result<Refold, StoreError> {
        self.log.verify_log()?;
        self.verify_segments()?;
        self.log.verify_tail()?;
        let frames = self.log.read_frames(1)?;
        let frame_count = frames.len() as u64;
        let gate_mismatches = gate_verdict_mismatches(self, &frames)?;
        let mut scratch = Self {
            dir: self.dir.clone(),
            log_dir: self.log_dir.clone(),
            log: self.log.clone(),
            gate_log: self.gate_log.clone(),
            policy: self.policy.clone(),
            nodes: BTreeMap::new(),
            origins: BTreeMap::new(),
            intentions: BTreeMap::new(),
            edges: Vec::new(),
            forward: BTreeMap::new(),
            reverse: BTreeMap::new(),
            fsrs: State::default(),
            review_events: Vec::new(),
            reviewed_at: BTreeMap::new(),
            checkpoints: Vec::new(),
            orphan_writes: 0,
            tool_call_open: false,
            call_admitted: BTreeSet::new(),
            retire_rules: BTreeMap::new(),
            upserts: BTreeMap::new(),
            anchors: AnchorIndex::default(),
            effect_index: EffectIndex::default(),
        };
        scratch.replay()?;
        let mut retrievability = BTreeMap::new();
        for id in scratch.nodes.keys() {
            if let Some(score) = scratch.retrievability(id)? {
                retrievability.insert(id.clone(), score);
            }
        }
        let gaps = self.sweep().iter().map(gap_label).collect::<Vec<_>>();
        Ok(Refold {
            frames: frame_count,
            state_digest: scratch.state_digest(),
            nodes: scratch.nodes,
            origins: scratch.origins,
            retrievability,
            gate_mismatches,
            gaps,
        })
    }

    /// Strict read of every segment. A torn frame, blake3 miss, or broken
    /// chain is an error.
    fn verify_segments(&self) -> Result<(), StoreError> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(&self.log_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("seg") {
                paths.push(path);
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(StoreError::Verify("log has no segments".into()));
        }
        let mut expected_prev = strata::GENESIS_PREV_SEGMENT_HASH;
        for (i, path) in paths.iter().enumerate() {
            let is_last = i + 1 == paths.len();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("segment");
            let bytes = std::fs::read(path)?;
            let scanned = scan_segment(&bytes, name)?;
            if scanned.header.prev_segment_hash != expected_prev {
                return Err(StoreError::Verify(format!(
                    "{name}: segment chain link mismatch"
                )));
            }
            if !is_last && scanned.trailer.is_none() {
                return Err(StoreError::Verify(format!(
                    "{name}: sealed segment is missing its trailer"
                )));
            }
            if !is_last {
                expected_prev = hash32(&bytes);
            }
        }
        Ok(())
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

    /// Ids of nodes (native and imported) that satisfy `keep`, in id order.
    /// Borrows each record instead of cloning its content.
    pub fn node_ids_where(&self, keep: impl Fn(&NodeRecord) -> bool) -> Vec<String> {
        self.nodes
            .values()
            .filter(|record| keep(record))
            .map(|record| record.id.clone())
            .collect()
    }

    /// Every typed edge, in landing order.
    pub fn edges(&self) -> Vec<ConnectionRecord> {
        self.edges.clone()
    }

    /// The node registry, borrowed (GhostLink reads it without cloning).
    pub(crate) fn node_map(&self) -> &BTreeMap<String, NodeRecord> {
        &self.nodes
    }

    /// Every edge in landing order, borrowed.
    pub(crate) fn edge_list(&self) -> &[ConnectionRecord] {
        &self.edges
    }

    /// The latest clock the log records: the greatest node `created_at_ms`
    /// or explicit review clock. Derived from the log alone, so a read that
    /// evaluates retention at this clock is deterministic for a given head.
    pub fn head_clock_ms(&self) -> i64 {
        self.nodes
            .values()
            .map(|record| record.created_at_ms)
            .chain(self.reviewed_at.values().copied())
            .max()
            .unwrap_or(0)
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

    /// Card-fold events (reviews and imported v3 cards) retained for
    /// verification, in fold order.
    pub fn review_event_count(&self) -> usize {
        self.review_events.len()
    }

    /// Every admitted node upsert, oldest first.
    ///
    /// Rebuilt by replay. A compensating undo is itself an upsert (`write`
    /// restored, or `superseded_by = undo:<seq>` for a create), so the
    /// classification survives reopen without a new op kind.
    pub fn node_writes(&self) -> Vec<NodeWrite> {
        let mut out = Vec::new();
        for versions in self.upserts.values() {
            for (idx, (frame_seq, record)) in versions.iter().enumerate() {
                let (op_type, reverts_frame_seq) = classify_upsert(versions, idx);
                let status = if idx + 1 == versions.len() {
                    "applied"
                } else {
                    "reverted"
                };
                out.push(NodeWrite {
                    frame_seq: *frame_seq,
                    record: record.clone(),
                    op_type,
                    status,
                    reverts_frame_seq,
                });
            }
        }
        out.sort_by_key(|write| write.frame_seq);
        out
    }

    /// Append a compensating `UpsertNode` that reverses the admitted upsert at
    /// `frame_seq`.
    ///
    /// The target must be that node's latest upsert and must not itself be an
    /// undo. A later `SupersedeNode` (the map record diverges from the upsert
    /// history) is a conflict and appends nothing. The returned seq is the new
    /// data frame. The prior frames stay in the log.
    pub fn undo_node_write(&mut self, frame_seq: u64) -> Result<u64, StoreError> {
        let (id, idx) = self
            .upserts
            .iter()
            .find_map(|(id, versions)| {
                versions
                    .iter()
                    .position(|(seq, _)| *seq == frame_seq)
                    .map(|idx| (id.clone(), idx))
            })
            .ok_or_else(|| StoreError::NotFound(format!("operation op-{frame_seq:016x}")))?;
        let versions = self
            .upserts
            .get(&id)
            .expect("node id was just found in upserts")
            .clone();
        if idx + 1 != versions.len() {
            return Err(StoreError::InvalidInput(
                "undo conflicts with later memory changes; no changes applied".into(),
            ));
        }
        let (op_type, _) = classify_upsert(&versions, idx);
        if op_type == "undo" {
            return Err(StoreError::InvalidInput(
                "cannot undo an undo operation".into(),
            ));
        }
        let history_tip = versions[idx].1.clone();
        let current = self.require_node(&id)?.clone();
        if current != history_tip {
            return Err(StoreError::InvalidInput(
                "undo conflicts with later memory changes; no changes applied".into(),
            ));
        }
        if idx != 0 {
            let record = versions[idx - 1].1.clone();
            let context = self.context_for(&[&id]);
            let (_effect_seq, data_seq) =
                self.admit_write(StoreOp::UpsertNode { record }, action_kind::WRITE, context)?;
            return Ok(data_seq);
        }
        // Undoing the first upsert of a node retires it. When that node was
        // the successor of an edit, the versions it retired come back live
        // and the code anchors that moved with the edit move back, so the
        // undo leaves the pre-edit memory rather than no memory at all.
        let restores: Vec<NodeRecord> = self
            .nodes
            .values()
            .filter(|node| node.superseded_by.as_deref() == Some(id.as_str()))
            .map(|node| {
                let mut restored = node.clone();
                restored.superseded_by = None;
                restored
            })
            .collect();
        let moved_anchors = self.anchors.rows_of(&id);
        let mut tomb = history_tip;
        tomb.superseded_by = Some(format!("undo:{frame_seq:016x}"));
        let context = self.context_for(&[&id]);
        let (_effect_seq, data_seq) = self.admit_write(
            StoreOp::UpsertNode { record: tomb },
            action_kind::WRITE,
            context,
        )?;
        for record in &restores {
            let context = self.context_for(&[&record.id]);
            self.admit_write(
                StoreOp::UpsertNode {
                    record: record.clone(),
                },
                action_kind::WRITE,
                context,
            )?;
        }
        if let Some(restored) = restores.first() {
            if !moved_anchors.is_empty() {
                let rows = moved_anchors
                    .into_iter()
                    .map(|anchor| AnchorRecord {
                        node_id: restored.id.clone(),
                        ..anchor
                    })
                    .collect();
                self.record_anchors(rows)?;
            }
        }
        Ok(data_seq)
    }
}

/// Marker prefix on `superseded_by` for an undo of a node's first upsert.
///
/// The node stays in the log and in the derived map, and [`NodeRecord::is_live`]
/// is false, so reads that honor liveness do not return it.
const UNDO_MARKER_PREFIX: &str = "undo:";

/// `superseded_by` marker on an imported memory that v3 had suppressed.
/// Like the undo marker it names no node; [`NodeRecord::is_live`] is false.
const IMPORTED_SUPPRESSED_MARKER: &str = "v3:suppressed";

/// One admitted `UpsertNode`, classified for the reversible operation log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWrite {
    /// Log sequence of the data frame.
    pub frame_seq: u64,
    /// Record exactly as that frame admitted it.
    pub record: NodeRecord,
    /// `write` for a caller mutation, `undo` for a compensating record.
    pub op_type: &'static str,
    /// `applied` when this frame is the node's latest upsert, otherwise `reverted`.
    pub status: &'static str,
    /// Data-frame seq this undo reverses, when [`Self::op_type`] is `undo`.
    pub reverts_frame_seq: Option<u64>,
}

fn undo_marker(record: &NodeRecord) -> Option<u64> {
    let marker = record.superseded_by.as_deref()?;
    let rest = marker.strip_prefix(UNDO_MARKER_PREFIX)?;
    u64::from_str_radix(rest, 16).ok()
}

fn classify_upsert(versions: &[(u64, NodeRecord)], idx: usize) -> (&'static str, Option<u64>) {
    let record = &versions[idx].1;
    if let Some(seq) = undo_marker(record) {
        return ("undo", Some(seq));
    }
    // A compensating restore is an exact copy of the version before the one
    // it reverses. Caller edits that happen to land on an older body are
    // indistinguishable and treated as undo; the only writer of an exact
    // prior body today is `undo_node_write`.
    if idx >= 2 && record == &versions[idx - 2].1 && record != &versions[idx - 1].1 {
        return ("undo", Some(versions[idx - 1].0));
    }
    // A version re-admitted unchanged after its retirement was undone: the
    // record matches the one before it exactly, which no caller edit yields.
    if idx >= 1 && record == &versions[idx - 1].1 {
        return ("undo", None);
    }
    ("write", None)
}

/// Hops in the undirected supersession component of `id`, in log order.
fn supersede_component(hops: &[SupersedeHop], id: &str) -> Vec<SupersedeHop> {
    let mut ids = BTreeSet::new();
    ids.insert(id.to_string());
    loop {
        let mut grew = false;
        for hop in hops {
            if ids.contains(&hop.id) || ids.contains(&hop.superseded_by) {
                grew |= ids.insert(hop.id.clone());
                grew |= ids.insert(hop.superseded_by.clone());
            }
        }
        if !grew {
            break;
        }
    }
    hops.iter()
        .filter(|hop| ids.contains(&hop.id) && ids.contains(&hop.superseded_by))
        .cloned()
        .collect()
}

/// Fresh fold of one log. Receipt replay compares this to the live maps.
pub struct Refold {
    /// Frames the refold read.
    pub frames: u64,
    /// State digest of the refolded copy.
    pub state_digest: [u8; 32],
    /// Refolded node registry.
    pub nodes: BTreeMap<String, NodeRecord>,
    /// Refolded origin seq per node.
    pub origins: BTreeMap<String, u64>,
    /// FSRS retrievability per node in the refolded copy.
    pub retrievability: BTreeMap<String, f64>,
    /// Gate verdicts the refold could not re-derive.
    pub gate_mismatches: Vec<String>,
    /// Admission gaps the sweep found.
    pub gaps: Vec<String>,
}

struct ScannedSegment {
    header: strata::SegmentHeader,
    trailer: Option<strata::SegmentTrailer>,
}

fn scan_segment(bytes: &[u8], name: &str) -> Result<ScannedSegment, StoreError> {
    if bytes.len() < strata::HEADER_WIRE_SIZE {
        return Err(StoreError::Verify(format!(
            "{name}: segment header unreadable"
        )));
    }
    let header: strata::SegmentHeader = borsh::from_slice(&bytes[..strata::HEADER_WIRE_SIZE])
        .map_err(|_| StoreError::Verify(format!("{name}: segment header unreadable")))?;
    if header.magic != strata::SEGMENT_MAGIC || header.version != strata::SEGMENT_VERSION {
        return Err(StoreError::Verify(format!(
            "{name}: segment header unreadable"
        )));
    }
    let mut prev = strata::header_hash(&header);
    let mut off = strata::HEADER_WIRE_SIZE;
    let mut leaves = Vec::new();
    let mut frames = 0u64;
    loop {
        let rem = bytes.len() - off;
        if rem == 0 {
            return Ok(ScannedSegment {
                header,
                trailer: None,
            });
        }
        if rem == strata::TRAILER_WIRE_SIZE {
            // A 35-byte-payload frame is also 104 bytes; a frame that parses,
            // hashes and chains is a frame, as in strata's own recovery.
            let chained = match strata::parse_frame(&bytes[off..]) {
                Ok((frame, used))
                    if used == rem
                        && frame.payload_blake3
                            == strata::payload_blake3(frame.kind, &frame.payload)
                        && frame.prev_frame_hash == prev =>
                {
                    Some((frame, used))
                }
                _ => None,
            };
            if let Some((frame, used)) = chained {
                prev = strata::frame_hash(&frame);
                leaves.push(frame.payload_blake3);
                frames += 1;
                off += used;
                continue;
            }
            let trailer: strata::SegmentTrailer =
                borsh::from_slice(&bytes[off..]).map_err(|_| {
                    StoreError::Verify(format!("{name}: trailer-sized tail failed to parse"))
                })?;
            if trailer.frame_count != frames {
                return Err(StoreError::Verify(format!(
                    "{name}: trailer frame_count {} != scanned {frames}",
                    trailer.frame_count
                )));
            }
            if trailer.merkle_root != strata::merkle_root(&leaves) {
                return Err(StoreError::Verify(format!(
                    "{name}: trailer merkle root mismatch"
                )));
            }
            return Ok(ScannedSegment {
                header,
                trailer: Some(trailer),
            });
        }
        if rem < strata::FRAME_FIXED_WIRE_SIZE {
            return Err(StoreError::Verify(format!(
                "{name}: short frame header at offset {off}"
            )));
        }
        let (frame, used) = strata::parse_frame(&bytes[off..]).map_err(|e| {
            StoreError::Verify(format!("{name}: frame parse failed at offset {off}: {e}"))
        })?;
        if frame.payload_blake3 != strata::payload_blake3(frame.kind, &frame.payload) {
            return Err(StoreError::Verify(format!(
                "{name}: payload blake3 mismatch at offset {off}"
            )));
        }
        if frame.prev_frame_hash != prev {
            return Err(StoreError::Verify(format!(
                "{name}: frame chain link mismatch at offset {off}"
            )));
        }
        prev = strata::frame_hash(&frame);
        leaves.push(frame.payload_blake3);
        frames += 1;
        off += used;
    }
}

fn gate_verdict_mismatches(
    store: &StrataStore,
    frames: &[strata::FrameRecord],
) -> Result<Vec<String>, StoreError> {
    let recomputed = store.rederive_verdicts()?;
    let mut stored = Vec::new();
    let mut gate_seq = 0u64;
    for frame in frames {
        let Some(kind) = RecordKind::from_u8(frame.kind) else {
            continue;
        };
        let gseq = gate_seq;
        gate_seq += 1;
        if kind != RecordKind::Gate {
            continue;
        }
        let gate = GateRecord::try_from_slice(&frame.payload).map_err(|error| {
            StoreError::Verify(format!("malformed gate record at seq {gseq}: {error}"))
        })?;
        stored.push((gseq, gate.verdict));
    }
    let mut out = Vec::new();
    if stored.len() != recomputed.len() {
        out.push("gate:count".into());
    }
    for (left, right) in stored.iter().zip(recomputed.iter()) {
        if left != right {
            out.push(format!("gate:{}", left.0));
        }
    }
    Ok(out)
}

fn gap_label(gap: &strata_gate::record::GapRecord) -> String {
    use strata_gate::record::GapDetail;
    match &gap.detail {
        GapDetail::OrphanEffect { effect_seq, .. } => format!("gap:orphan_effect:{effect_seq}"),
        GapDetail::ReadNoReceipt { reader_seq, .. } => {
            format!("gap:read_no_receipt:{reader_seq}")
        }
        GapDetail::DutySeqGap {
            source,
            expected,
            found,
        } => format!("gap:duty_seq_gap:{source}:{expected}:{found}"),
    }
}
