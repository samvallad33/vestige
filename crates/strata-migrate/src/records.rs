//! Wire records the migration writes into the STRATA log.
//!
//! Kind-byte allocation: `strata-gate` owns codes `1..=7` and `0` is reserved
//! ("unknown") by the log layer. The migration family starts at `0x1F`
//! (`GENESIS`) so it cannot collide with gate records in a shared log.
//! `MIGRATION_RECEIPT` is frame kind 46 (`0x2E`) per the PR-0a spec; the
//! `kinds.rs` registry adopts it in PR-0c.
//!
//! Every payload begins with a little-endian `u16` `record_version` so a
//! future migration format can evolve without kind renegotiation. Version 1
//! is the layout documented here.
//!
//! The `FSRS_REVIEW` payload is NOT a bespoke struct: it is the kernel's own
//! `strata_kernel::event::ReviewEvent` borsh encoding, and the `CHECKPOINT`
//! payload is `strata_kernel::checkpoint::Checkpoint` verbatim. Migration
//! records reuse kernel wire types wherever one exists — one encoding per
//! concept, forever.

use borsh::{BorshDeserialize, BorshSerialize};
use ed25519_dalek::Signer;
use strata_kernel::checkpoint::Checkpoint;
use strata_kernel::event::ReviewEvent;

/// First frame of a fresh migration log: provenance for everything after it.
pub const KIND_GENESIS: u8 = 0x1F;
/// Migration parameter set `v4-migrate/1`: schema version, source BLAKE3,
/// envelope chain head. Written immediately after `GENESIS`.
pub const KIND_PARAMS: u8 = 0x26;
/// One `knowledge_nodes` row (or a V40 `walk_receipts` reference node).
pub const KIND_NODE: u8 = 0x20;
/// One `memory_connections` row mapped into the 8-type STRATA vocabulary.
pub const KIND_EDGE: u8 = 0x21;
/// A v3 association row (`semantic` / `similarity`). Not an EDGE: causal
/// walks and Backfill do not traverse this kind.
pub const KIND_LEGACY_LINK: u8 = 0x27;
/// One synthesized review event (payload = kernel `ReviewEvent`).
pub const KIND_FSRS_REVIEW: u8 = 0x22;
/// One `sync_tombstones` or `deletion_tombstones` row.
pub const KIND_TOMBSTONE: u8 = 0x23;
/// One supersession lineage pointer (`knowledge_nodes.superseded_by`).
pub const KIND_SUPERSESSION: u8 = 0x24;
/// Sealed fold checkpoint (payload = kernel `Checkpoint`).
pub const KIND_CHECKPOINT: u8 = 0x25;
/// Final frame of a migration: signed MIGRATION_RECEIPT (kind 46; PR-0c's
/// `kinds.rs` adopts this code).
pub const KIND_MIGRATION_RECEIPT: u8 = 46;

/// Current wire version of every migration record below.
pub const RECORD_VERSION: u16 = 1;

/// Identity string recorded in every receipt body (the authorship proof is
/// the ed25519 signature + in-log verifying key; this names the scheme).
pub const RECEIPT_SIGNING_KEY_ID: &str = "vestige-migrate-receipt-v1";

/// The receipt-signing key file, written next to the destination log with
/// 0600 permissions. It is NOT part of the log: the log itself carries only
/// the verifying key. Losing it only loses the ability to sign further
/// receipts for that destination; verification uses the in-log public key.
pub const RECEIPT_KEY_FILE: &str = "receipt-signing.key";

/// ed25519 signing domain for the MIGRATION_RECEIPT signature.
const RECEIPT_SIGNATURE_CONTEXT: &[u8] = b"vestige strata migration receipt v1";

/// The blake3 CHECKSUM over the receipt body: tamper-evidence only. It
/// proves nothing about authorship (anyone can recompute it) — the ed25519
/// signature is the authorship proof. (Audit finding: a keyed-BLAKE3
/// "signature" was presented as verification while anyone could re-derive
/// both the key and the digest.)

/// First frame of a fresh migration log: provenance for everything after it.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct GenesisRecord {
    pub record_version: u16,
    /// Source archive format identifier (expected `vestige.portable.v1`).
    pub archive_format: String,
    /// Vestige version that produced the source archive.
    pub vestige_version: String,
    /// SQLite schema version of the source database.
    pub schema_version: u32,
}

/// Second frame of a fresh migration log: the parameter set actually used.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ParamsRecord {
    pub record_version: u16,
    /// Parameter set id (`v4-migrate/1`).
    pub params_id: String,
    /// Schema version of the migrated source.
    pub schema_version: u32,
    /// BLAKE3 hex of the source files (db, then `-wal`, then `-shm` in
    /// canonical order) taken before any byte was read through SQL.
    pub source_blake3: String,
    /// `entry_digest` of the last verified `receipt_envelopes` row
    /// (empty string when the source kept no envelopes).
    pub envelope_head: String,
    /// Ed25519 verifying key for the log segments. The private key is not
    /// in the log directory.
    pub log_verifying_key: [u8; 32],
}

/// A migrated knowledge node. `kernel_id` is the dense 1-based STRATA
/// identity assigned in source row order; `legacy_id` keeps the original
/// UUID so nothing is lost and back-references stay possible.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct NodeRecord {
    pub record_version: u16,
    pub legacy_id: String,
    pub kernel_id: u64,
    pub content: String,
    pub node_type: String,
    pub tags: Vec<String>,
    /// Milliseconds since the Unix epoch.
    pub created_ms: i64,
    pub updated_ms: i64,
    pub last_accessed_ms: i64,
    /// EVERY other source column, verbatim (ints/floats/text as canonical
    /// strings, blobs hex). FSRS state, scope, source, suppression,
    /// sentiment, and the rest of the 52 columns ride here (blocker 4):
    /// nothing a v3 row carried is silently dropped.
    pub legacy: Vec<(String, String)>,
}

/// A migrated typed edge. Legacy link types are folded into the 8-type
/// STRATA vocabulary: any `link_type` outside the vocabulary becomes
/// `derived_from` with `legacy_inferred = 1` (H4: only the 8 types exist;
/// migration never invents new edge semantics for inferred history).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct EdgeRecord {
    pub record_version: u16,
    pub source_kernel_id: u64,
    pub target_kernel_id: u64,
    pub source_legacy_id: String,
    pub target_legacy_id: String,
    /// STRATA vocabulary type (`derived_from` when rewritten from legacy).
    pub link_type: String,
    /// True when `link_type` was rewritten from the legacy vocabulary.
    pub legacy_inferred: bool,
    /// Original legacy link type, kept for provenance only.
    pub legacy_link_type: String,
    /// Edge strength quantized to Q32.32 (`strata_kernel::canonical`).
    pub strength_q32: i64,
    pub created_ms: i64,
    pub last_activated_ms: i64,
    pub activation_count: i32,
    /// Every other source column, verbatim (e.g. v39 edge_meta).
    pub legacy: Vec<(String, String)>,
}

/// A v3 association row kept for provenance. It is not a causal edge.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct LegacyLinkRecord {
    pub record_version: u16,
    pub source_kernel_id: u64,
    pub target_kernel_id: u64,
    pub source_legacy_id: String,
    pub target_legacy_id: String,
    /// Original v3 link type (`semantic` or `similarity`).
    pub legacy_link_type: String,
    pub strength_q32: i64,
    pub created_ms: i64,
    pub last_activated_ms: i64,
    pub activation_count: i32,
    pub legacy: Vec<(String, String)>,
}

/// A migrated tombstone row (`sync_tombstones` or `deletion_tombstones`).
/// The two source tables have different column sets; fields only present in
/// `deletion_tombstones` are `None`/empty for sync tombstones.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct TombstoneRecord {
    pub record_version: u16,
    /// Source table the tombstone came from.
    pub origin_table: String,
    /// The table the tombstone targets (sync_tombstones.table_name); None
    /// where the source has no such column.
    pub source_table: Option<String>,
    /// Tombstoned row id (memory id for deletion tombstones).
    pub row_id: String,
    pub deleted_ms: i64,
    pub reason: Option<String>,
    /// deletion_tombstones only.
    pub node_type: Option<String>,
    /// deletion_tombstones only.
    pub tags: Vec<String>,
}

/// A supersession lineage pointer from the bitemporal store
/// (`knowledge_nodes.superseded_by`): A was superseded by B, and both ids are
/// kept — legacy strings always, kernel ids when both endpoints were mapped
/// (0 otherwise).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct SupersessionRecord {
    pub record_version: u16,
    pub superseded_legacy_id: String,
    pub superseded_by_legacy_id: String,
    pub superseded_kernel_id: u64,
    pub superseded_by_kernel_id: u64,
}

/// Body of the signed MIGRATION_RECEIPT. All collections are ordered Vecs
/// (H6: no HashMap in hashed state); `counts` is sorted by table name.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ReceiptBody {
    pub record_version: u16,
    /// BLAKE3 hex of the source files before the migration read them.
    pub source_blake3_before: String,
    /// BLAKE3 hex of the same files re-hashed after the migration; the
    /// migrator refuses to seal a receipt whose before != after, so a
    /// sealed receipt always carries equal values.
    pub source_blake3_after: String,
    /// Schema version of the migrated source.
    pub schema_version: u32,
    /// Last verified `receipt_envelopes` entry digest (empty = none).
    pub envelope_head: String,
    /// Source row counts per table, sorted by table name.
    pub counts: Vec<(String, u64)>,
    /// `node_embeddings` rows whose vector values were never read (H1:
    /// vectors do not survive into STRATA; this counts what was dropped).
    pub dropped_vectors: u64,
    /// Source columns that did NOT ride into the log, named (blocker 4:
    /// the receipt must never present itself as a fuller copy than it is).
    pub dropped_columns: Vec<String>,
    /// [`RECEIPT_SIGNING_KEY_ID`].
    pub signing_key_id: String,
}

/// Wire form of frame kind 46: body, checksum, and an ed25519 signature
/// made by a non-derivable random key stored OUTSIDE the log (0600). The
/// log carries only the verifying key.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct MigrationReceipt {
    pub record_version: u16,
    pub body: ReceiptBody,
    /// Ed25519 verifying key (the signing key never touches the log).
    pub verifying_key: [u8; 32],
    /// Ed25519 signature over borsh(body).
    pub signature: [u8; 64],
    /// Plain blake3 checksum of borsh(body): tamper-evidence, not authorship.
    pub checksum: [u8; 32],
}

impl MigrationReceipt {
    /// Seal a body with a caller-managed ed25519 signing key.
    pub fn seal(body: ReceiptBody, signing: &ed25519_dalek::SigningKey) -> Self {
        let bytes = borsh::to_vec(&body).expect("borsh encode receipt body");
        let checksum = *blake3::hash(&bytes).as_bytes();
        let mut msg = Vec::with_capacity(bytes.len() + RECEIPT_SIGNATURE_CONTEXT.len());
        msg.extend_from_slice(RECEIPT_SIGNATURE_CONTEXT);
        msg.extend_from_slice(&bytes);
        let signature = signing.sign(&msg).to_bytes();
        Self {
            record_version: RECORD_VERSION,
            body,
            verifying_key: signing.verifying_key().to_bytes(),
            signature,
            checksum,
        }
    }

    /// The checksum binds the body (tamper-evidence).
    pub fn verify_checksum(&self) -> bool {
        match borsh::to_vec(&self.body) {
            Ok(bytes) => *blake3::hash(&bytes).as_bytes() == self.checksum,
            Err(_) => false,
        }
    }

    /// The signature proves authorship under the in-log verifying key.
    pub fn verify_signature(&self) -> bool {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};
        let Ok(vk) = VerifyingKey::from_bytes(&self.verifying_key) else {
            return false;
        };
        match borsh::to_vec(&self.body) {
            Ok(bytes) => {
                let mut msg = Vec::with_capacity(bytes.len() + RECEIPT_SIGNATURE_CONTEXT.len());
                msg.extend_from_slice(RECEIPT_SIGNATURE_CONTEXT);
                msg.extend_from_slice(&bytes);
                vk.verify(&msg, &Signature::from_bytes(&self.signature))
                    .is_ok()
            }
            Err(_) => false,
        }
    }
}

/// Load the destination's receipt-signing key (0600) or create one from the
/// OS entropy pool. The key is NEVER derived from public data (audit: the
/// old seed was blake3 over the source BLAKE3 written into the log, so any
/// reader could re-derive strata.key and forge segments).
pub fn load_or_create_receipt_key(
    dir: &std::path::Path,
) -> Result<ed25519_dalek::SigningKey, std::io::Error> {
    let path = dir.join(RECEIPT_KEY_FILE);
    if path.exists() {
        let bytes = std::fs::read(&path)?;
        let seed: [u8; 32] = bytes.try_into().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{RECEIPT_KEY_FILE} is not 32 bytes"),
            )
        })?;
        return Ok(ed25519_dalek::SigningKey::from_bytes(&seed));
    }
    let mut seed = [0u8; 32];
    urandom_fill(&mut seed)?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        f.write_all(&seed)?;
    }
    #[cfg(not(unix))]
    std::fs::write(&path, &seed)?;
    Ok(ed25519_dalek::SigningKey::from_bytes(&seed))
}

/// Read OS entropy (unix /dev/urandom). No weak fallback: a platform
/// without urandom fails the migration instead of shipping a derivable key.
fn urandom_fill(buf: &mut [u8]) -> Result<(), std::io::Error> {
    use std::io::Read;
    std::fs::File::open("/dev/urandom")?.read_exact(buf)
}

/// Decode a `KIND_GENESIS` payload.
pub fn decode_genesis(payload: &[u8]) -> Result<GenesisRecord, borsh::io::Error> {
    GenesisRecord::try_from_slice(payload)
}

/// Decode a `KIND_PARAMS` payload.
pub fn decode_params(payload: &[u8]) -> Result<ParamsRecord, borsh::io::Error> {
    ParamsRecord::try_from_slice(payload)
}

/// Decode a `KIND_NODE` payload.
pub fn decode_node(payload: &[u8]) -> Result<NodeRecord, borsh::io::Error> {
    NodeRecord::try_from_slice(payload)
}

/// Decode a `KIND_EDGE` payload.
pub fn decode_edge(payload: &[u8]) -> Result<EdgeRecord, borsh::io::Error> {
    EdgeRecord::try_from_slice(payload)
}

/// Decode a `KIND_LEGACY_LINK` payload.
pub fn decode_legacy_link(payload: &[u8]) -> Result<LegacyLinkRecord, borsh::io::Error> {
    LegacyLinkRecord::try_from_slice(payload)
}

/// Decode a `KIND_FSRS_REVIEW` payload (kernel wire type).
pub fn decode_review(payload: &[u8]) -> Result<ReviewEvent, borsh::io::Error> {
    ReviewEvent::try_from_slice(payload)
}

/// Decode a `KIND_TOMBSTONE` payload.
pub fn decode_tombstone(payload: &[u8]) -> Result<TombstoneRecord, borsh::io::Error> {
    TombstoneRecord::try_from_slice(payload)
}

/// Decode a `KIND_SUPERSESSION` payload.
pub fn decode_supersession(payload: &[u8]) -> Result<SupersessionRecord, borsh::io::Error> {
    SupersessionRecord::try_from_slice(payload)
}

/// Decode a `KIND_CHECKPOINT` payload (kernel wire type).
pub fn decode_checkpoint(payload: &[u8]) -> Result<Checkpoint, borsh::io::Error> {
    Checkpoint::try_from_slice(payload)
}

/// Decode a `KIND_MIGRATION_RECEIPT` payload.
pub fn decode_receipt(payload: &[u8]) -> Result<MigrationReceipt, borsh::io::Error> {
    MigrationReceipt::try_from_slice(payload)
}
