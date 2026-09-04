//! Controlled, post-retrieval context-ablation replay.
//!
//! A replay capsule freezes the exact, final, token-budgeted evidence pack that
//! crossed the MCP boundary. Replaying that capsule only removes named,
//! receipt-local evidence slots. It never reruns retrieval, backfills a removed
//! candidate, expands the graph, calls a model, or mutates memory state.
//!
//! Raw memory text, queries, prompts, and model output are deliberately absent
//! from every public replay result in this module. Evidence is represented by
//! an opaque local slot and a keyed private digest. Suppression makes a capsule
//! permanently non-replayable, removes public evidence/size fingerprints, and
//! retains only a private memory dependency locator so a later purge can find
//! it. Purge then erases that locator and the residual item-row cardinality.

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};
use crate::trace::Receipt;

/// The only selection boundary implemented by replay v1.
pub const REPLAY_SELECTION_BOUNDARY: &str = "post_retrieval_context_ablation";
/// Stable algorithm identifier included in every capsule, digest, and replay.
pub const REPLAY_ALGORITHM_VERSION: &str = "vestige.post_retrieval_context_ablation.v1";
/// Schema version for persisted capsule and replay payloads.
pub const REPLAY_SCHEMA_VERSION: u32 = 1;
/// Public claim boundary. Product surfaces must show this verbatim.
pub const REPLAY_CLAIM_BOUNDARY: &str = "Controlled replay shows how the recorded memory context changes when specified evidence is withheld. It does not establish that a memory caused an agent answer or any real-world outcome.";

const PRIVATE_DIGEST_DOMAIN: &[u8] = b"vestige.replay.private-item.v1";
const POLICY_DIGEST_DOMAIN: &[u8] = b"vestige.replay.policy.v1";
const ITEM_LEAF_DOMAIN: &[u8] = b"vestige.replay.item-leaf.v1";
const SET_DIGEST_DOMAIN: &[u8] = b"vestige.replay.ordered-set.v1";
const EMPTY_MERKLE_DOMAIN: &[u8] = b"vestige.replay.merkle-empty.v1";
const MERKLE_PARENT_DOMAIN: &[u8] = b"vestige.replay.merkle-parent.v1";
const IDEMPOTENCY_DOMAIN: &[u8] = b"vestige.replay.idempotency.v1";

/// Privacy state of a frozen capsule or replay record.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPrivacyState {
    Active,
    Redacted,
    Purged,
}

impl ReplayPrivacyState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Redacted => "redacted",
            Self::Purged => "purged",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, ReplayBuildError> {
        match value {
            "active" => Ok(Self::Active),
            "redacted" => Ok(Self::Redacted),
            "purged" => Ok(Self::Purged),
            other => Err(ReplayBuildError::InvalidPersistedState(format!(
                "unknown replay privacy state `{other}`"
            ))),
        }
    }
}

/// Coarse decay signal frozen with one evidence item.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDecayRisk {
    Low,
    Medium,
    High,
}

impl ReplayDecayRisk {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn parse(value: &str) -> std::result::Result<Self, ReplayBuildError> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(ReplayBuildError::InvalidPersistedState(format!(
                "unknown replay decay risk `{other}`"
            ))),
        }
    }
}

/// Whether privacy invalidation came from reversible suppression or purge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayInvalidationReason {
    Suppressed,
    Purged,
}

/// Validation errors from the pure replay engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayBuildError {
    EmptySourceReceipt,
    InvalidSourceReceipt,
    EmptyPolicyDigest,
    InvalidPolicyDigest,
    EmptyEvidencePack,
    InvalidEvidenceSlot(String),
    InvalidMemoryIdentifier(String),
    DuplicateEvidenceSlot(String),
    UnknownWithheldSlot(String),
    InvalidPrivateDigest(String),
    DuplicatePrivateDigest(String),
    InvalidTrustScore(String),
    TokenEstimateOverflow,
    InvalidPersistedState(String),
    CapsuleNotReplayable(String),
    FrozenCapsuleIntegrityMismatch(String),
    ConflictingFrozenCapsule(String),
    ConflictingReceiptLink(String),
    ReceiptCapsuleMismatch,
    ConflictingIdempotentReplay(String),
    PersistedReplayIntegrityMismatch(String),
    NumericOverflow(&'static str),
}

impl fmt::Display for ReplayBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySourceReceipt => write!(f, "source receipt id must not be empty"),
            Self::InvalidSourceReceipt => {
                write!(f, "source receipt id is not a valid opaque identifier")
            }
            Self::EmptyPolicyDigest => write!(f, "policy digest must not be empty"),
            Self::InvalidPolicyDigest => {
                write!(f, "policy digest must be a canonical BLAKE3 digest")
            }
            Self::EmptyEvidencePack => write!(f, "frozen evidence pack must not be empty"),
            Self::InvalidEvidenceSlot(slot) => {
                write!(f, "invalid receipt-local evidence slot `{slot}`")
            }
            Self::InvalidMemoryIdentifier(slot) => {
                write!(
                    f,
                    "invalid opaque memory identifier at evidence slot `{slot}`"
                )
            }
            Self::DuplicateEvidenceSlot(slot) => {
                write!(f, "duplicate receipt-local evidence slot `{slot}`")
            }
            Self::UnknownWithheldSlot(slot) => {
                write!(
                    f,
                    "withheld evidence slot `{slot}` is not in the frozen pack"
                )
            }
            Self::InvalidPrivateDigest(slot) => {
                write!(f, "invalid private digest for evidence slot `{slot}`")
            }
            Self::DuplicatePrivateDigest(slot) => {
                write!(f, "duplicate private digest at evidence slot `{slot}`")
            }
            Self::InvalidTrustScore(slot) => {
                write!(f, "invalid trust score for evidence slot `{slot}`")
            }
            Self::TokenEstimateOverflow => {
                write!(
                    f,
                    "frozen evidence token estimates exceed the supported range"
                )
            }
            Self::InvalidPersistedState(message) => write!(f, "{message}"),
            Self::CapsuleNotReplayable(id) => {
                write!(f, "replay capsule `{id}` is not replayable")
            }
            Self::FrozenCapsuleIntegrityMismatch(id) => {
                write!(
                    f,
                    "replay capsule `{id}` failed frozen-evidence integrity checks"
                )
            }
            Self::ConflictingFrozenCapsule(id) => {
                write!(
                    f,
                    "source receipt `{id}` already has a different frozen capsule"
                )
            }
            Self::ConflictingReceiptLink(id) => {
                write!(f, "replay `{id}` is already linked to a different receipt")
            }
            Self::ReceiptCapsuleMismatch => {
                write!(
                    f,
                    "receipt id does not match replay capsule source receipt id"
                )
            }
            Self::ConflictingIdempotentReplay(id) => {
                write!(f, "idempotent replay `{id}` conflicts with frozen evidence")
            }
            Self::PersistedReplayIntegrityMismatch(id) => {
                write!(f, "persisted replay `{id}` failed integrity checks")
            }
            Self::NumericOverflow(field) => write!(f, "{field} exceeds SQLite INTEGER range"),
        }
    }
}

impl std::error::Error for ReplayBuildError {}

/// One item in the exact evidence pack supplied by the retrieval boundary.
///
/// `memory_id` is only an internal dependency locator. It is never copied into
/// replay result JSON. `private_digest` must be a keyed digest, not a public
/// hash of memory content.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReplayItemDraft {
    pub evidence_slot: String,
    pub memory_id: String,
    pub private_digest: String,
    pub token_estimate: u64,
    pub trust_score: f64,
    pub decay_risk: ReplayDecayRisk,
}

/// Draft frozen alongside one retrieval receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReplayCapsuleDraft {
    pub source_receipt_id: String,
    pub policy_digest: String,
    pub items: Vec<RetrievalReplayItemDraft>,
    pub created_at: DateTime<Utc>,
}

impl RetrievalReplayCapsuleDraft {
    pub fn new(
        source_receipt_id: impl Into<String>,
        policy_digest: impl Into<String>,
        items: Vec<RetrievalReplayItemDraft>,
    ) -> Self {
        Self {
            source_receipt_id: source_receipt_id.into(),
            policy_digest: policy_digest.into(),
            items,
            created_at: Utc::now(),
        }
    }
}

/// Persisted frozen item. Suppression nulls evidence and size-derived fields;
/// the memory locator remains private until purge deletes the item row.
#[derive(Debug, Clone, PartialEq)]
struct RetrievalReplayItem {
    pub ordinal: u32,
    pub evidence_slot: String,
    pub memory_id: Option<String>,
    pub private_digest: Option<String>,
    pub token_estimate: Option<u64>,
    pub trust_score: Option<f64>,
    pub decay_risk: Option<ReplayDecayRisk>,
}

/// Frozen retrieval capsule as stored locally. This is an internal audit type;
/// public replay evidence is [`CounterfactualReplayResult`], which carries no
/// memory ids or item digests.
#[derive(Debug, Clone, PartialEq)]
struct RetrievalReplayCapsule {
    pub capsule_id: String,
    pub source_receipt_id: String,
    pub schema_version: u32,
    pub algorithm_version: String,
    pub selection_boundary: String,
    pub redaction_generation: u64,
    pub privacy_state: ReplayPrivacyState,
    pub replayable: bool,
    pub policy_digest: String,
    pub baseline_evidence_digest: Option<String>,
    pub baseline_merkle_root: Option<String>,
    pub item_count: Option<u64>,
    pub total_token_estimate: Option<u64>,
    pub trust_floor: Option<f64>,
    pub decay_risk: Option<ReplayDecayRisk>,
    pub items: Vec<RetrievalReplayItem>,
    pub created_at: DateTime<Utc>,
}

/// Public capsule projection. Stable memory ids and private item digests never
/// enter this type; non-active capsules expose no item rows at all.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalReplayCapsuleSummary {
    pub capsule_id: String,
    pub source_receipt_id: String,
    pub schema_version: u32,
    pub algorithm_version: String,
    pub selection_boundary: String,
    pub redaction_generation: u64,
    pub privacy_state: ReplayPrivacyState,
    pub replayable: bool,
    pub policy_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_evidence_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_merkle_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_estimate: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_floor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decay_risk: Option<ReplayDecayRisk>,
    pub items: Vec<ReplayEvidenceItemSummary>,
    pub created_at: DateTime<Utc>,
}

/// Result of durable capsule creation.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableRetrievalReplayCapsule {
    pub capsule: RetrievalReplayCapsuleSummary,
    pub reused_existing: bool,
}

/// Privacy-safe item consumed by the pure ablation engine.
#[derive(Debug, Clone, PartialEq)]
pub struct FrozenReplayItem {
    pub ordinal: u32,
    pub evidence_slot: String,
    pub private_digest: String,
    pub token_estimate: u64,
    pub trust_score: f64,
    pub decay_risk: ReplayDecayRisk,
}

/// Exact aggregate of one ordered evidence set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEvidenceItemSummary {
    pub evidence_slot: String,
    /// One-based rank in the exact final evidence order.
    pub rank: u32,
    pub token_estimate: u64,
    pub trust_score: f64,
    pub decay_risk: ReplayDecayRisk,
}

/// Exact aggregate of one ordered evidence set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEvidenceSetSummary {
    pub items: Vec<ReplayEvidenceItemSummary>,
    pub ordered_slots: Vec<String>,
    pub item_count: u64,
    pub token_estimate: u64,
    pub trust_floor: f64,
    pub decay_risk: ReplayDecayRisk,
    pub ordered_evidence_digest: String,
    pub merkle_root: String,
}

/// Measured structural difference between baseline and ablated context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayInfluence {
    pub removed_item_count: u64,
    pub removed_token_estimate: u64,
    pub trust_floor_delta: f64,
    pub decay_risk_changed: bool,
    pub ordered_evidence_digest_changed: bool,
    pub merkle_root_changed: bool,
}

/// Privacy-safe evidence payload for one controlled replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CounterfactualReplayResult {
    pub source_receipt_id: String,
    pub schema_version: u32,
    pub algorithm_version: String,
    pub selection_boundary: String,
    pub policy_digest: String,
    pub redaction_generation: u64,
    pub withheld_slots: Vec<String>,
    pub baseline: ReplayEvidenceSetSummary,
    pub counterfactual: ReplayEvidenceSetSummary,
    pub replay_influence: ReplayInfluence,
    /// Replay persisted an audit record but did not mutate cognitive state.
    pub memory_state_was_read_only: bool,
    pub claim_boundary: String,
}

/// Stored counterfactual replay. `result` becomes `None` after privacy
/// invalidation; structural audit linkage may remain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StoredCounterfactualReplay {
    pub replay_id: String,
    pub idempotency_key: String,
    pub capsule_id: String,
    pub source_receipt_id: String,
    pub receipt_id: Option<String>,
    pub algorithm_version: String,
    pub redaction_generation: u64,
    pub withheld_slots: Vec<String>,
    pub privacy_state: ReplayPrivacyState,
    pub result: Option<CounterfactualReplayResult>,
    pub created_at: DateTime<Utc>,
}

/// Result of replay creation or an idempotent retry.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableCounterfactualReplay {
    pub replay: StoredCounterfactualReplay,
    pub reused_existing: bool,
}

/// Count of privacy records affected by one invalidation operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayPrivacyInvalidation {
    pub capsules_invalidated: u64,
    pub replays_invalidated: u64,
}

/// Outcome of checking fresh content against a frozen item digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMaterializationCheck {
    Match,
    ContentChanged,
    Unavailable,
}

/// Build a non-public, keyed digest of one exact evidence fragment.
///
/// The key must come from a local secret and must not be persisted next to the
/// digest. Including the receipt-local slot prevents cross-slot equality leaks.
pub fn private_evidence_digest(
    private_key: &[u8; 32],
    evidence_slot: &str,
    evidence_bytes: &[u8],
) -> String {
    let mut hasher = blake3::Hasher::new_keyed(private_key);
    put_field(&mut hasher, PRIVATE_DIGEST_DOMAIN);
    put_field(&mut hasher, evidence_slot.as_bytes());
    put_field(&mut hasher, evidence_bytes);
    format!("b3k:{}", hasher.finalize().to_hex())
}

/// Digest a canonical, non-content selection-policy representation.
pub fn replay_policy_digest(canonical_policy: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    put_field(&mut hasher, POLICY_DIGEST_DOMAIN);
    put_field(&mut hasher, canonical_policy);
    format!("b3:{}", hasher.finalize().to_hex())
}

/// Generate the only accepted receipt-local slot format for a one-based rank.
pub fn replay_evidence_slot(rank: usize) -> String {
    format!("evidence_{rank}")
}

/// Canonical idempotency key for a replay request.
///
/// Withheld slots are sorted and deduplicated, so retry order and accidental
/// duplicate arguments do not mint a second replay.
pub fn replay_idempotency_key(
    algorithm_version: &str,
    source_receipt_id: &str,
    redaction_generation: u64,
    withheld_slots: &[String],
) -> String {
    let normalized = normalize_withheld_slots(withheld_slots);
    let mut hasher = blake3::Hasher::new();
    put_field(&mut hasher, IDEMPOTENCY_DOMAIN);
    put_field(&mut hasher, algorithm_version.as_bytes());
    put_field(&mut hasher, source_receipt_id.as_bytes());
    put_field(&mut hasher, &redaction_generation.to_be_bytes());
    for slot in normalized {
        put_field(&mut hasher, slot.as_bytes());
    }
    format!("b3:{}", hasher.finalize().to_hex())
}

/// Pure post-retrieval context ablation.
///
/// The counterfactual is the baseline items, in the same order, excluding the
/// named slots. There is intentionally no callback through which search,
/// candidate refill, graph expansion, an LLM, or memory writes could occur.
pub fn ablate_frozen_context(
    source_receipt_id: &str,
    policy_digest: &str,
    redaction_generation: u64,
    items: &[FrozenReplayItem],
    withheld_slots: &[String],
) -> std::result::Result<CounterfactualReplayResult, ReplayBuildError> {
    if source_receipt_id.trim().is_empty() {
        return Err(ReplayBuildError::EmptySourceReceipt);
    }
    if !valid_opaque_identifier(source_receipt_id) {
        return Err(ReplayBuildError::InvalidSourceReceipt);
    }
    if !valid_blake3_digest(policy_digest, "b3:") {
        return Err(ReplayBuildError::InvalidPolicyDigest);
    }
    validate_frozen_items(items)?;

    let withheld_slots = normalize_withheld_slots(withheld_slots);
    let available: HashSet<&str> = items
        .iter()
        .map(|item| item.evidence_slot.as_str())
        .collect();
    for slot in &withheld_slots {
        if !available.contains(slot.as_str()) {
            return Err(ReplayBuildError::UnknownWithheldSlot(slot.clone()));
        }
    }

    let withheld: HashSet<&str> = withheld_slots.iter().map(String::as_str).collect();
    let counterfactual_items: Vec<FrozenReplayItem> = items
        .iter()
        .filter(|item| !withheld.contains(item.evidence_slot.as_str()))
        .cloned()
        .collect();

    let baseline = summarize_evidence_set(items);
    let counterfactual = summarize_evidence_set(&counterfactual_items);
    let replay_influence = ReplayInfluence {
        removed_item_count: baseline.item_count - counterfactual.item_count,
        removed_token_estimate: baseline.token_estimate - counterfactual.token_estimate,
        trust_floor_delta: round_score(counterfactual.trust_floor - baseline.trust_floor),
        decay_risk_changed: baseline.decay_risk != counterfactual.decay_risk,
        ordered_evidence_digest_changed: baseline.ordered_evidence_digest
            != counterfactual.ordered_evidence_digest,
        merkle_root_changed: baseline.merkle_root != counterfactual.merkle_root,
    };

    Ok(CounterfactualReplayResult {
        source_receipt_id: source_receipt_id.to_string(),
        schema_version: REPLAY_SCHEMA_VERSION,
        algorithm_version: REPLAY_ALGORITHM_VERSION.to_string(),
        selection_boundary: REPLAY_SELECTION_BOUNDARY.to_string(),
        policy_digest: policy_digest.to_string(),
        redaction_generation,
        withheld_slots,
        baseline,
        counterfactual,
        replay_influence,
        memory_state_was_read_only: true,
        claim_boundary: REPLAY_CLAIM_BOUNDARY.to_string(),
    })
}

impl SqliteMemoryStore {
    /// Atomically persist a retrieval receipt and the exact replay capsule it
    /// names. Raw query text is never copied into the receipt row.
    pub fn save_retrieval_receipt_with_replay_capsule(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> Result<DurableRetrievalReplayCapsule> {
        if receipt.receipt_id != draft.source_receipt_id {
            return Err(replay_error(ReplayBuildError::ReceiptCapsuleMismatch));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx =
            Self::begin_write_transaction(&writer, "save_retrieval_receipt_with_replay_capsule")?;
        insert_or_validate_receipt(&tx, receipt, run_id, tool)?;
        let durable = Self::save_retrieval_replay_capsule_in_transaction(&tx, draft)?;
        tx.commit()?;
        Ok(durable)
    }

    /// Save one frozen retrieval capsule in its own immediate transaction.
    pub fn save_retrieval_replay_capsule(
        &self,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> Result<DurableRetrievalReplayCapsule> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_retrieval_replay_capsule")?;
        let durable = Self::save_retrieval_replay_capsule_in_transaction(&tx, draft)?;
        tx.commit()?;
        Ok(durable)
    }

    /// Transactional seam used when the generic receipt and its frozen capsule
    /// must commit atomically.
    pub(crate) fn save_retrieval_replay_capsule_in_transaction(
        tx: &Transaction<'_>,
        draft: &RetrievalReplayCapsuleDraft,
    ) -> Result<DurableRetrievalReplayCapsule> {
        let frozen = capsule_from_draft(draft).map_err(replay_error)?;
        if let Some(existing) = load_capsule_by_source(tx, &draft.source_receipt_id)? {
            if capsules_have_same_frozen_evidence(&existing, &frozen) {
                return Ok(DurableRetrievalReplayCapsule {
                    capsule: capsule_summary(&existing),
                    reused_existing: true,
                });
            }
            return Err(replay_error(ReplayBuildError::ConflictingFrozenCapsule(
                draft.source_receipt_id.clone(),
            )));
        }

        tx.execute(
            "INSERT INTO retrieval_replay_capsules
                 (capsule_id, source_receipt_id, schema_version, algorithm_version,
                  selection_boundary, redaction_generation, privacy_state, replayable,
                  policy_digest, baseline_evidence_digest, baseline_merkle_root,
                  item_count, total_token_estimate, trust_floor, decay_risk, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', 1, ?7, ?8, ?9,
                     ?10, ?11, ?12, ?13, ?14)",
            params![
                frozen.capsule_id,
                frozen.source_receipt_id,
                frozen.schema_version as i64,
                frozen.algorithm_version,
                frozen.selection_boundary,
                to_sql_i64(frozen.redaction_generation, "redaction_generation")?,
                frozen.policy_digest,
                frozen.baseline_evidence_digest,
                frozen.baseline_merkle_root,
                frozen
                    .item_count
                    .map(|value| to_sql_i64(value, "item_count"))
                    .transpose()?,
                frozen
                    .total_token_estimate
                    .map(|value| to_sql_i64(value, "total_token_estimate"))
                    .transpose()?,
                frozen.trust_floor,
                frozen.decay_risk.map(ReplayDecayRisk::as_str),
                frozen.created_at.to_rfc3339(),
            ],
        )?;

        for item in &frozen.items {
            tx.execute(
                "INSERT INTO retrieval_replay_items
                     (capsule_id, ordinal, evidence_slot, memory_id, private_digest,
                      token_estimate, trust_score, decay_risk)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    frozen.capsule_id,
                    item.ordinal as i64,
                    item.evidence_slot,
                    item.memory_id,
                    item.private_digest,
                    item.token_estimate
                        .map(|value| to_sql_i64(value, "token_estimate"))
                        .transpose()?,
                    item.trust_score,
                    item.decay_risk.map(ReplayDecayRisk::as_str),
                ],
            )?;
        }

        Ok(DurableRetrievalReplayCapsule {
            capsule: capsule_summary(&frozen),
            reused_existing: false,
        })
    }

    /// Load a capsule by the retrieval receipt that froze it.
    pub fn get_retrieval_replay_capsule(
        &self,
        source_receipt_id: &str,
    ) -> Result<Option<RetrievalReplayCapsuleSummary>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        Ok(load_capsule_by_source(&reader, source_receipt_id)?
            .as_ref()
            .map(capsule_summary))
    }

    /// Create or idempotently return one controlled context-ablation replay.
    ///
    /// This writes only the replay audit row. It does not read or mutate
    /// `knowledge_nodes`, rerun retrieval, or write any cognitive state.
    pub fn create_context_ablation_replay(
        &self,
        source_receipt_id: &str,
        withheld_slots: &[String],
    ) -> Result<DurableCounterfactualReplay> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "create_context_ablation_replay")?;
        let capsule = load_capsule_by_source(&tx, source_receipt_id)?.ok_or_else(|| {
            StorageError::NotFound(format!("replay capsule for receipt {source_receipt_id}"))
        })?;
        let frozen_items = active_frozen_items(&capsule).map_err(replay_error)?;
        let result = ablate_frozen_context(
            &capsule.source_receipt_id,
            &capsule.policy_digest,
            capsule.redaction_generation,
            &frozen_items,
            withheld_slots,
        )
        .map_err(replay_error)?;
        let idempotency_key = replay_idempotency_key(
            &capsule.algorithm_version,
            &capsule.source_receipt_id,
            capsule.redaction_generation,
            &result.withheld_slots,
        );

        if let Some(existing) = load_replay_by_idempotency_key(&tx, &idempotency_key)? {
            if existing.capsule_id != capsule.capsule_id
                || existing.source_receipt_id != capsule.source_receipt_id
                || existing.algorithm_version != capsule.algorithm_version
                || existing.redaction_generation != capsule.redaction_generation
                || existing.withheld_slots != result.withheld_slots
                || existing.privacy_state != ReplayPrivacyState::Active
                || existing.result.as_ref() != Some(&result)
            {
                return Err(replay_error(ReplayBuildError::ConflictingIdempotentReplay(
                    existing.replay_id,
                )));
            }
            tx.commit()?;
            return Ok(DurableCounterfactualReplay {
                replay: existing,
                reused_existing: true,
            });
        }

        let replay_id = format!("replay_{}", Uuid::new_v4().simple());
        let result_json = serde_json::to_string(&result)
            .map_err(|error| StorageError::Init(format!("replay result serialize: {error}")))?;
        let withheld_slots_json =
            serde_json::to_string(&result.withheld_slots).map_err(|error| {
                StorageError::Init(format!("replay withheld slots serialize: {error}"))
            })?;
        let created_at = Utc::now();
        tx.execute(
            "INSERT INTO counterfactual_replays
                 (replay_id, idempotency_key, capsule_id, source_receipt_id, receipt_id,
                  algorithm_version, redaction_generation, withheld_slots_json,
                  result_json, privacy_state, created_at)
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, 'active', ?9)",
            params![
                replay_id,
                idempotency_key,
                capsule.capsule_id,
                capsule.source_receipt_id,
                capsule.algorithm_version,
                to_sql_i64(capsule.redaction_generation, "redaction_generation")?,
                withheld_slots_json,
                result_json,
                created_at.to_rfc3339(),
            ],
        )?;
        tx.commit()?;

        Ok(DurableCounterfactualReplay {
            replay: StoredCounterfactualReplay {
                replay_id,
                idempotency_key,
                capsule_id: capsule.capsule_id,
                source_receipt_id: capsule.source_receipt_id,
                receipt_id: None,
                algorithm_version: capsule.algorithm_version,
                redaction_generation: capsule.redaction_generation,
                withheld_slots: result.withheld_slots.clone(),
                privacy_state: ReplayPrivacyState::Active,
                result: Some(result),
                created_at,
            },
            reused_existing: false,
        })
    }

    /// Load one persisted replay. A privacy-invalidated replay retains its
    /// linkage but returns no result payload.
    pub fn get_context_ablation_replay(
        &self,
        replay_id: &str,
    ) -> Result<Option<StoredCounterfactualReplay>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        load_replay_by_id(&reader, replay_id)
    }

    /// Link a replay to its generic receipt exactly once. An idempotent retry
    /// with the same receipt succeeds; a different receipt is rejected.
    pub fn link_context_ablation_receipt(&self, replay_id: &str, receipt_id: &str) -> Result<()> {
        if !valid_random_public_id(replay_id, "replay_") || !valid_opaque_identifier(receipt_id) {
            return Err(replay_error(ReplayBuildError::ConflictingReceiptLink(
                replay_id.to_string(),
            )));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "link_context_ablation_receipt")?;
        let receipt_exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM memory_receipts WHERE receipt_id = ?1)",
            params![receipt_id],
            |row| row.get(0),
        )?;
        if !receipt_exists {
            return Err(StorageError::NotFound(receipt_id.to_string()));
        }
        tx.execute(
            "UPDATE counterfactual_replays SET receipt_id = ?1
             WHERE replay_id = ?2 AND (receipt_id IS NULL OR receipt_id = ?1)",
            params![receipt_id, replay_id],
        )?;
        let linked: Option<Option<String>> = tx
            .query_row(
                "SELECT receipt_id FROM counterfactual_replays WHERE replay_id = ?1",
                params![replay_id],
                |row| row.get(0),
            )
            .optional()?;
        match linked {
            None => Err(StorageError::NotFound(replay_id.to_string())),
            Some(Some(current)) if current == receipt_id => {
                tx.commit()?;
                Ok(())
            }
            _ => Err(replay_error(ReplayBuildError::ConflictingReceiptLink(
                replay_id.to_string(),
            ))),
        }
    }

    /// Atomically persist a typed replay receipt and link it to the replay row.
    /// A crash before commit leaves the replay unlinked; an idempotent retry can
    /// safely create a fresh receipt id and complete the same transaction.
    pub fn save_counterfactual_replay_receipt(
        &self,
        replay_id: &str,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_counterfactual_replay_receipt")?;
        let replay = load_replay_by_id(&tx, replay_id)?
            .ok_or_else(|| StorageError::NotFound(replay_id.to_string()))?;
        if replay.privacy_state != ReplayPrivacyState::Active {
            return Err(replay_error(ReplayBuildError::CapsuleNotReplayable(
                replay.capsule_id,
            )));
        }
        let evidence_matches = matches!(
            &receipt.evidence,
            Some(crate::trace::ReceiptEvidence::CounterfactualReplay {
                replay_id: evidence_replay_id,
                capsule_id,
                result: evidence_result,
                ..
            }) if evidence_replay_id == replay_id
                && capsule_id == &replay.capsule_id
                && replay.result.as_ref() == Some(evidence_result)
        );
        if !evidence_matches {
            return Err(replay_error(
                ReplayBuildError::PersistedReplayIntegrityMismatch(replay_id.to_string()),
            ));
        }
        if let Some(existing_receipt_id) = replay.receipt_id {
            if existing_receipt_id == receipt.receipt_id {
                insert_or_validate_receipt(&tx, receipt, run_id, tool)?;
                tx.commit()?;
                return Ok(());
            }
            return Err(replay_error(ReplayBuildError::ConflictingReceiptLink(
                replay_id.to_string(),
            )));
        }

        insert_or_validate_receipt(&tx, receipt, run_id, tool)?;
        let changed = tx.execute(
            "UPDATE counterfactual_replays SET receipt_id = ?1
             WHERE replay_id = ?2 AND receipt_id IS NULL AND privacy_state = 'active'",
            params![receipt.receipt_id, replay_id],
        )?;
        if changed != 1 {
            return Err(replay_error(ReplayBuildError::ConflictingReceiptLink(
                replay_id.to_string(),
            )));
        }
        tx.commit()?;
        Ok(())
    }

    /// Compare a caller-computed keyed digest of current materialization with
    /// the frozen digest. The raw materialized content never enters storage.
    pub fn verify_replay_materialization_digest(
        &self,
        source_receipt_id: &str,
        evidence_slot: &str,
        current_private_digest: &str,
    ) -> Result<ReplayMaterializationCheck> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(String, i64, Option<String>)> = reader
            .query_row(
                "SELECT c.privacy_state, c.replayable, i.private_digest
                 FROM retrieval_replay_capsules c
                 JOIN retrieval_replay_items i ON i.capsule_id = c.capsule_id
                 WHERE c.source_receipt_id = ?1 AND i.evidence_slot = ?2",
                params![source_receipt_id, evidence_slot],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((state, replayable, frozen_digest)) = row else {
            return Ok(ReplayMaterializationCheck::Unavailable);
        };
        if ReplayPrivacyState::parse(&state).map_err(replay_error)? != ReplayPrivacyState::Active
            || replayable != 1
        {
            return Ok(ReplayMaterializationCheck::Unavailable);
        }
        match frozen_digest {
            Some(digest)
                if constant_time_eq(digest.as_bytes(), current_private_digest.as_bytes()) =>
            {
                Ok(ReplayMaterializationCheck::Match)
            }
            Some(_) => Ok(ReplayMaterializationCheck::ContentChanged),
            None => Ok(ReplayMaterializationCheck::Unavailable),
        }
    }

    /// Privacy-invalidate every active capsule that still references a memory.
    pub fn invalidate_replay_evidence_for_memory(
        &self,
        memory_id: &str,
        reason: ReplayInvalidationReason,
    ) -> Result<ReplayPrivacyInvalidation> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "invalidate_replay_evidence_for_memory")?;
        let report =
            Self::invalidate_replay_evidence_for_memory_in_transaction(&tx, memory_id, reason)?;
        tx.commit()?;
        Ok(report)
    }

    /// Transactional privacy seam for suppression/purge paths.
    pub(crate) fn invalidate_replay_evidence_for_memory_in_transaction(
        tx: &Transaction<'_>,
        memory_id: &str,
        reason: ReplayInvalidationReason,
    ) -> Result<ReplayPrivacyInvalidation> {
        let mut stmt = tx.prepare(
            "SELECT DISTINCT c.capsule_id
             FROM retrieval_replay_capsules c
             JOIN retrieval_replay_items i ON i.capsule_id = c.capsule_id
             WHERE i.memory_id = ?1",
        )?;
        let rows = stmt.query_map(params![memory_id], |row| row.get::<_, String>(0))?;
        let mut capsule_ids = Vec::new();
        for row in rows {
            capsule_ids.push(row?);
        }
        drop(stmt);
        invalidate_capsules_in_transaction(tx, &capsule_ids, reason)
    }

    /// Privacy-invalidate a known capsule. Purge callers can use this after a
    /// prior suppression has already scrubbed its memory dependency locator.
    pub fn invalidate_replay_capsule(
        &self,
        capsule_id: &str,
        reason: ReplayInvalidationReason,
    ) -> Result<ReplayPrivacyInvalidation> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "invalidate_replay_capsule")?;
        let report = invalidate_capsules_in_transaction(&tx, &[capsule_id.to_string()], reason)?;
        tx.commit()?;
        Ok(report)
    }
}

fn capsule_from_draft(
    draft: &RetrievalReplayCapsuleDraft,
) -> std::result::Result<RetrievalReplayCapsule, ReplayBuildError> {
    if draft.source_receipt_id.trim().is_empty() {
        return Err(ReplayBuildError::EmptySourceReceipt);
    }
    if !valid_opaque_identifier(&draft.source_receipt_id) {
        return Err(ReplayBuildError::InvalidSourceReceipt);
    }
    if draft.policy_digest.trim().is_empty() {
        return Err(ReplayBuildError::EmptyPolicyDigest);
    }
    if !valid_blake3_digest(&draft.policy_digest, "b3:") {
        return Err(ReplayBuildError::InvalidPolicyDigest);
    }
    if draft.items.is_empty() {
        return Err(ReplayBuildError::EmptyEvidencePack);
    }
    let mut private_digests = HashSet::new();
    for (index, item) in draft.items.iter().enumerate() {
        if item.evidence_slot != replay_evidence_slot(index + 1) {
            return Err(ReplayBuildError::InvalidEvidenceSlot(
                item.evidence_slot.clone(),
            ));
        }
        if !valid_opaque_identifier(&item.memory_id)
            || item.evidence_slot == item.memory_id
            || (item.memory_id.len() >= 8 && item.evidence_slot.contains(&item.memory_id))
        {
            return Err(ReplayBuildError::InvalidMemoryIdentifier(
                item.evidence_slot.clone(),
            ));
        }
        if !private_digests.insert(item.private_digest.as_str()) {
            return Err(ReplayBuildError::DuplicatePrivateDigest(
                item.evidence_slot.clone(),
            ));
        }
    }

    let items: Vec<FrozenReplayItem> = draft
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| FrozenReplayItem {
            ordinal: index as u32,
            evidence_slot: item.evidence_slot.clone(),
            private_digest: item.private_digest.clone(),
            token_estimate: item.token_estimate,
            trust_score: item.trust_score,
            decay_risk: item.decay_risk,
        })
        .collect();
    validate_frozen_items(&items)?;
    let summary = summarize_evidence_set(&items);

    let stored_items = draft
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| RetrievalReplayItem {
            ordinal: index as u32,
            evidence_slot: item.evidence_slot.clone(),
            memory_id: Some(item.memory_id.clone()),
            private_digest: Some(item.private_digest.clone()),
            token_estimate: Some(item.token_estimate),
            trust_score: Some(item.trust_score),
            decay_risk: Some(item.decay_risk),
        })
        .collect();

    Ok(RetrievalReplayCapsule {
        capsule_id: format!("rcap_{}", Uuid::new_v4().simple()),
        source_receipt_id: draft.source_receipt_id.clone(),
        schema_version: REPLAY_SCHEMA_VERSION,
        algorithm_version: REPLAY_ALGORITHM_VERSION.to_string(),
        selection_boundary: REPLAY_SELECTION_BOUNDARY.to_string(),
        redaction_generation: 0,
        privacy_state: ReplayPrivacyState::Active,
        replayable: true,
        policy_digest: draft.policy_digest.clone(),
        baseline_evidence_digest: Some(summary.ordered_evidence_digest),
        baseline_merkle_root: Some(summary.merkle_root),
        item_count: Some(summary.item_count),
        total_token_estimate: Some(summary.token_estimate),
        trust_floor: Some(summary.trust_floor),
        decay_risk: Some(summary.decay_risk),
        items: stored_items,
        created_at: draft.created_at,
    })
}

fn active_frozen_items(
    capsule: &RetrievalReplayCapsule,
) -> std::result::Result<Vec<FrozenReplayItem>, ReplayBuildError> {
    if capsule.privacy_state != ReplayPrivacyState::Active
        || !capsule.replayable
        || capsule.schema_version != REPLAY_SCHEMA_VERSION
        || capsule.algorithm_version != REPLAY_ALGORITHM_VERSION
        || capsule.selection_boundary != REPLAY_SELECTION_BOUNDARY
    {
        return Err(ReplayBuildError::CapsuleNotReplayable(
            capsule.capsule_id.clone(),
        ));
    }
    let mut frozen = Vec::with_capacity(capsule.items.len());
    for item in &capsule.items {
        frozen.push(FrozenReplayItem {
            ordinal: item.ordinal,
            evidence_slot: item.evidence_slot.clone(),
            private_digest: item.private_digest.clone().ok_or_else(|| {
                ReplayBuildError::CapsuleNotReplayable(capsule.capsule_id.clone())
            })?,
            token_estimate: item.token_estimate.ok_or_else(|| {
                ReplayBuildError::CapsuleNotReplayable(capsule.capsule_id.clone())
            })?,
            trust_score: item.trust_score.ok_or_else(|| {
                ReplayBuildError::CapsuleNotReplayable(capsule.capsule_id.clone())
            })?,
            decay_risk: item.decay_risk.ok_or_else(|| {
                ReplayBuildError::CapsuleNotReplayable(capsule.capsule_id.clone())
            })?,
        });
    }
    validate_frozen_items(&frozen)?;
    let recomputed = summarize_evidence_set(&frozen);
    if !valid_blake3_digest(&capsule.policy_digest, "b3:")
        || capsule.baseline_evidence_digest.as_deref()
            != Some(recomputed.ordered_evidence_digest.as_str())
        || capsule.baseline_merkle_root.as_deref() != Some(recomputed.merkle_root.as_str())
        || capsule.item_count != Some(recomputed.item_count)
        || capsule.total_token_estimate != Some(recomputed.token_estimate)
        || capsule.trust_floor != Some(recomputed.trust_floor)
        || capsule.decay_risk != Some(recomputed.decay_risk)
    {
        return Err(ReplayBuildError::FrozenCapsuleIntegrityMismatch(
            capsule.capsule_id.clone(),
        ));
    }
    Ok(frozen)
}

fn capsule_summary(capsule: &RetrievalReplayCapsule) -> RetrievalReplayCapsuleSummary {
    let items = if capsule.privacy_state == ReplayPrivacyState::Active && capsule.replayable {
        capsule
            .items
            .iter()
            .filter_map(|item| {
                Some(ReplayEvidenceItemSummary {
                    evidence_slot: item.evidence_slot.clone(),
                    rank: item.ordinal.checked_add(1)?,
                    token_estimate: item.token_estimate?,
                    trust_score: round_score(item.trust_score?),
                    decay_risk: item.decay_risk?,
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    RetrievalReplayCapsuleSummary {
        capsule_id: capsule.capsule_id.clone(),
        source_receipt_id: capsule.source_receipt_id.clone(),
        schema_version: capsule.schema_version,
        algorithm_version: capsule.algorithm_version.clone(),
        selection_boundary: capsule.selection_boundary.clone(),
        redaction_generation: capsule.redaction_generation,
        privacy_state: capsule.privacy_state,
        replayable: capsule.replayable,
        policy_digest: capsule.policy_digest.clone(),
        baseline_evidence_digest: capsule.baseline_evidence_digest.clone(),
        baseline_merkle_root: capsule.baseline_merkle_root.clone(),
        item_count: capsule.item_count,
        total_token_estimate: capsule.total_token_estimate,
        trust_floor: capsule.trust_floor,
        decay_risk: capsule.decay_risk,
        items,
        created_at: capsule.created_at,
    }
}

fn validate_frozen_items(items: &[FrozenReplayItem]) -> std::result::Result<(), ReplayBuildError> {
    if items.is_empty() {
        return Err(ReplayBuildError::EmptyEvidencePack);
    }
    let mut slots = HashSet::new();
    let mut token_total = 0u64;
    for (expected_ordinal, item) in items.iter().enumerate() {
        if item.ordinal as usize != expected_ordinal
            || item.evidence_slot != replay_evidence_slot(expected_ordinal + 1)
        {
            return Err(ReplayBuildError::InvalidEvidenceSlot(
                item.evidence_slot.clone(),
            ));
        }
        if !slots.insert(item.evidence_slot.as_str()) {
            return Err(ReplayBuildError::DuplicateEvidenceSlot(
                item.evidence_slot.clone(),
            ));
        }
        if !valid_blake3_digest(&item.private_digest, "b3k:") {
            return Err(ReplayBuildError::InvalidPrivateDigest(
                item.evidence_slot.clone(),
            ));
        }
        if !item.trust_score.is_finite() || !(0.0..=1.0).contains(&item.trust_score) {
            return Err(ReplayBuildError::InvalidTrustScore(
                item.evidence_slot.clone(),
            ));
        }
        if item.token_estimate > i64::MAX as u64 {
            return Err(ReplayBuildError::TokenEstimateOverflow);
        }
        token_total = token_total
            .checked_add(item.token_estimate)
            .ok_or(ReplayBuildError::TokenEstimateOverflow)?;
    }
    if token_total > i64::MAX as u64 {
        return Err(ReplayBuildError::TokenEstimateOverflow);
    }
    Ok(())
}

fn valid_opaque_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
}

fn valid_random_public_id(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(|suffix| {
        suffix.len() == 32
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_evidence_slot_shape(slot: &str) -> bool {
    slot.strip_prefix("evidence_")
        .and_then(|rank| rank.parse::<usize>().ok())
        .is_some_and(|rank| rank > 0 && replay_evidence_slot(rank) == slot)
}

fn valid_evidence_summary(summary: &ReplayEvidenceSetSummary) -> bool {
    if summary.item_count != summary.items.len() as u64
        || summary.ordered_slots
            != summary
                .items
                .iter()
                .map(|item| item.evidence_slot.clone())
                .collect::<Vec<_>>()
        || !valid_blake3_digest(&summary.ordered_evidence_digest, "b3:")
        || !valid_blake3_digest(&summary.merkle_root, "b3:")
    {
        return false;
    }
    let mut token_estimate = 0u64;
    for item in &summary.items {
        if !valid_evidence_slot_shape(&item.evidence_slot)
            || item.rank == 0
            || replay_evidence_slot(item.rank as usize) != item.evidence_slot
            || !item.trust_score.is_finite()
            || !(0.0..=1.0).contains(&item.trust_score)
        {
            return false;
        }
        let Some(total) = token_estimate.checked_add(item.token_estimate) else {
            return false;
        };
        token_estimate = total;
    }
    let trust_floor = summary
        .items
        .iter()
        .map(|item| item.trust_score)
        .reduce(f64::min)
        .map(round_score)
        .unwrap_or(0.0);
    let decay_risk = summary
        .items
        .iter()
        .map(|item| item.decay_risk)
        .max()
        .unwrap_or(ReplayDecayRisk::High);
    summary.token_estimate == token_estimate
        && summary.trust_floor == trust_floor
        && summary.decay_risk == decay_risk
}

fn valid_counterfactual_result(
    result: &CounterfactualReplayResult,
    source_receipt_id: &str,
    policy_digest: &str,
    redaction_generation: u64,
    withheld_slots: &[String],
) -> bool {
    if result.source_receipt_id != source_receipt_id
        || result.schema_version != REPLAY_SCHEMA_VERSION
        || result.algorithm_version != REPLAY_ALGORITHM_VERSION
        || result.selection_boundary != REPLAY_SELECTION_BOUNDARY
        || result.policy_digest != policy_digest
        || result.redaction_generation != redaction_generation
        || result.withheld_slots != withheld_slots
        || !result.memory_state_was_read_only
        || result.claim_boundary != REPLAY_CLAIM_BOUNDARY
        || !valid_evidence_summary(&result.baseline)
        || !valid_evidence_summary(&result.counterfactual)
    {
        return false;
    }
    let withheld: HashSet<&str> = withheld_slots.iter().map(String::as_str).collect();
    let expected_counterfactual: Vec<ReplayEvidenceItemSummary> = result
        .baseline
        .items
        .iter()
        .filter(|item| !withheld.contains(item.evidence_slot.as_str()))
        .cloned()
        .collect();
    if result.counterfactual.items != expected_counterfactual {
        return false;
    }
    result.replay_influence
        == ReplayInfluence {
            removed_item_count: result.baseline.item_count - result.counterfactual.item_count,
            removed_token_estimate: result.baseline.token_estimate
                - result.counterfactual.token_estimate,
            trust_floor_delta: round_score(
                result.counterfactual.trust_floor - result.baseline.trust_floor,
            ),
            decay_risk_changed: result.baseline.decay_risk != result.counterfactual.decay_risk,
            ordered_evidence_digest_changed: result.baseline.ordered_evidence_digest
                != result.counterfactual.ordered_evidence_digest,
            merkle_root_changed: result.baseline.merkle_root != result.counterfactual.merkle_root,
        }
}

fn valid_blake3_digest(value: &str, prefix: &str) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn summarize_evidence_set(items: &[FrozenReplayItem]) -> ReplayEvidenceSetSummary {
    let token_estimate = items.iter().map(|item| item.token_estimate).sum();
    let trust_floor = items
        .iter()
        .map(|item| item.trust_score)
        .reduce(f64::min)
        .map(round_score)
        .unwrap_or(0.0);
    let decay_risk = items
        .iter()
        .map(|item| item.decay_risk)
        .max()
        .unwrap_or(ReplayDecayRisk::High);
    ReplayEvidenceSetSummary {
        items: items
            .iter()
            .map(|item| ReplayEvidenceItemSummary {
                evidence_slot: item.evidence_slot.clone(),
                rank: item.ordinal + 1,
                token_estimate: item.token_estimate,
                trust_score: round_score(item.trust_score),
                decay_risk: item.decay_risk,
            })
            .collect(),
        ordered_slots: items
            .iter()
            .map(|item| item.evidence_slot.clone())
            .collect(),
        item_count: items.len() as u64,
        token_estimate,
        trust_floor,
        decay_risk,
        ordered_evidence_digest: ordered_evidence_set_digest(items),
        merkle_root: evidence_merkle_root(items),
    }
}

fn ordered_evidence_set_digest(items: &[FrozenReplayItem]) -> String {
    let mut hasher = blake3::Hasher::new();
    put_field(&mut hasher, SET_DIGEST_DOMAIN);
    put_field(&mut hasher, &(items.len() as u64).to_be_bytes());
    for item in items {
        let leaf = item_leaf_digest(item);
        put_field(&mut hasher, leaf.as_bytes());
    }
    format!("b3:{}", hasher.finalize().to_hex())
}

fn evidence_merkle_root(items: &[FrozenReplayItem]) -> String {
    if items.is_empty() {
        let mut hasher = blake3::Hasher::new();
        put_field(&mut hasher, EMPTY_MERKLE_DOMAIN);
        return format!("b3:{}", hasher.finalize().to_hex());
    }

    let mut level: Vec<blake3::Hash> = items.iter().map(item_leaf_digest).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let left = pair[0];
            let right = pair.get(1).copied().unwrap_or(left);
            let mut hasher = blake3::Hasher::new();
            put_field(&mut hasher, MERKLE_PARENT_DOMAIN);
            put_field(&mut hasher, left.as_bytes());
            put_field(&mut hasher, right.as_bytes());
            next.push(hasher.finalize());
        }
        level = next;
    }
    format!("b3:{}", level[0].to_hex())
}

fn item_leaf_digest(item: &FrozenReplayItem) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    put_field(&mut hasher, ITEM_LEAF_DOMAIN);
    put_field(&mut hasher, &item.ordinal.to_be_bytes());
    put_field(&mut hasher, item.evidence_slot.as_bytes());
    put_field(&mut hasher, item.private_digest.as_bytes());
    put_field(&mut hasher, &item.token_estimate.to_be_bytes());
    put_field(&mut hasher, &item.trust_score.to_bits().to_be_bytes());
    put_field(&mut hasher, item.decay_risk.as_str().as_bytes());
    hasher.finalize()
}

fn put_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn normalize_withheld_slots(withheld_slots: &[String]) -> Vec<String> {
    withheld_slots
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn round_score(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn replay_error(error: ReplayBuildError) -> StorageError {
    StorageError::Init(format!("controlled replay: {error}"))
}

fn to_sql_i64(value: u64, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| replay_error(ReplayBuildError::NumericOverflow(field)))
}

fn from_sql_u64(value: i64, field: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        replay_error(ReplayBuildError::InvalidPersistedState(format!(
            "negative persisted {field}"
        )))
    })
}

fn parse_datetime(value: &str, field: &'static str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| {
            replay_error(ReplayBuildError::InvalidPersistedState(format!(
                "invalid {field}: {error}"
            )))
        })
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let max_len = left.len().max(right.len());
    for index in 0..max_len {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        difference |= (left_byte ^ right_byte) as usize;
    }
    difference == 0
}

fn capsules_have_same_frozen_evidence(
    existing: &RetrievalReplayCapsule,
    proposed: &RetrievalReplayCapsule,
) -> bool {
    existing.privacy_state == ReplayPrivacyState::Active
        && existing.replayable
        && existing.source_receipt_id == proposed.source_receipt_id
        && existing.schema_version == proposed.schema_version
        && existing.algorithm_version == proposed.algorithm_version
        && existing.selection_boundary == proposed.selection_boundary
        && existing.redaction_generation == 0
        && existing.policy_digest == proposed.policy_digest
        && existing.baseline_evidence_digest == proposed.baseline_evidence_digest
        && existing.baseline_merkle_root == proposed.baseline_merkle_root
        && existing.item_count == proposed.item_count
        && existing.total_token_estimate == proposed.total_token_estimate
        && existing.trust_floor == proposed.trust_floor
        && existing.decay_risk == proposed.decay_risk
        && existing.items == proposed.items
}

fn insert_or_validate_receipt(
    tx: &Transaction<'_>,
    receipt: &Receipt,
    run_id: Option<&str>,
    tool: Option<&str>,
) -> Result<()> {
    let payload = serde_json::to_string(receipt)
        .map_err(|error| StorageError::Init(format!("receipt serialize: {error}")))?;
    let existing: Option<String> = tx
        .query_row(
            "SELECT payload FROM memory_receipts WHERE receipt_id = ?1",
            params![receipt.receipt_id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        if constant_time_eq(existing.as_bytes(), payload.as_bytes()) {
            return Ok(());
        }
        return Err(replay_error(ReplayBuildError::ConflictingFrozenCapsule(
            receipt.receipt_id.clone(),
        )));
    }
    tx.execute(
        "INSERT INTO memory_receipts
             (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
              trust_floor, decay_risk, payload, created_at)
         VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            receipt.receipt_id,
            run_id,
            tool,
            receipt.retrieved.len() as i64,
            receipt.suppressed.len() as i64,
            receipt.trust_floor,
            receipt.decay_risk.as_str(),
            payload,
            Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(())
}

#[derive(Debug)]
struct CapsuleRow {
    capsule_id: String,
    source_receipt_id: String,
    schema_version: i64,
    algorithm_version: String,
    selection_boundary: String,
    redaction_generation: i64,
    privacy_state: String,
    replayable: i64,
    policy_digest: String,
    baseline_evidence_digest: Option<String>,
    baseline_merkle_root: Option<String>,
    item_count: Option<i64>,
    total_token_estimate: Option<i64>,
    trust_floor: Option<f64>,
    decay_risk: Option<String>,
    created_at: String,
}

fn load_capsule_by_source(
    conn: &Connection,
    source_receipt_id: &str,
) -> Result<Option<RetrievalReplayCapsule>> {
    let header: Option<CapsuleRow> = conn
        .query_row(
            "SELECT capsule_id, source_receipt_id, schema_version, algorithm_version,
                    selection_boundary, redaction_generation, privacy_state, replayable,
                    policy_digest, baseline_evidence_digest, baseline_merkle_root,
                    item_count, total_token_estimate, trust_floor, decay_risk, created_at
             FROM retrieval_replay_capsules WHERE source_receipt_id = ?1",
            params![source_receipt_id],
            |row| {
                Ok(CapsuleRow {
                    capsule_id: row.get(0)?,
                    source_receipt_id: row.get(1)?,
                    schema_version: row.get(2)?,
                    algorithm_version: row.get(3)?,
                    selection_boundary: row.get(4)?,
                    redaction_generation: row.get(5)?,
                    privacy_state: row.get(6)?,
                    replayable: row.get(7)?,
                    policy_digest: row.get(8)?,
                    baseline_evidence_digest: row.get(9)?,
                    baseline_merkle_root: row.get(10)?,
                    item_count: row.get(11)?,
                    total_token_estimate: row.get(12)?,
                    trust_floor: row.get(13)?,
                    decay_risk: row.get(14)?,
                    created_at: row.get(15)?,
                })
            },
        )
        .optional()?;
    let Some(header) = header else {
        return Ok(None);
    };

    let mut stmt = conn.prepare(
        "SELECT ordinal, evidence_slot, memory_id, private_digest,
                token_estimate, trust_score, decay_risk
         FROM retrieval_replay_items WHERE capsule_id = ?1 ORDER BY ordinal ASC",
    )?;
    let rows = stmt.query_map(params![header.capsule_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<f64>>(5)?,
            row.get::<_, Option<String>>(6)?,
        ))
    })?;
    let mut items = Vec::new();
    for row in rows {
        let (ordinal, evidence_slot, memory_id, private_digest, tokens, trust, decay) = row?;
        items.push(RetrievalReplayItem {
            ordinal: u32::try_from(ordinal).map_err(|_| {
                replay_error(ReplayBuildError::InvalidPersistedState(
                    "invalid replay item ordinal".into(),
                ))
            })?,
            evidence_slot,
            memory_id,
            private_digest,
            token_estimate: tokens
                .map(|value| from_sql_u64(value, "token_estimate"))
                .transpose()?,
            trust_score: trust,
            decay_risk: decay
                .as_deref()
                .map(ReplayDecayRisk::parse)
                .transpose()
                .map_err(replay_error)?,
        });
    }

    Ok(Some(RetrievalReplayCapsule {
        capsule_id: header.capsule_id,
        source_receipt_id: header.source_receipt_id,
        schema_version: u32::try_from(header.schema_version).map_err(|_| {
            replay_error(ReplayBuildError::InvalidPersistedState(
                "invalid replay schema version".into(),
            ))
        })?,
        algorithm_version: header.algorithm_version,
        selection_boundary: header.selection_boundary,
        redaction_generation: from_sql_u64(header.redaction_generation, "redaction_generation")?,
        privacy_state: ReplayPrivacyState::parse(&header.privacy_state).map_err(replay_error)?,
        replayable: header.replayable == 1,
        policy_digest: header.policy_digest,
        baseline_evidence_digest: header.baseline_evidence_digest,
        baseline_merkle_root: header.baseline_merkle_root,
        item_count: header
            .item_count
            .map(|value| from_sql_u64(value, "item_count"))
            .transpose()?,
        total_token_estimate: header
            .total_token_estimate
            .map(|value| from_sql_u64(value, "total_token_estimate"))
            .transpose()?,
        trust_floor: header.trust_floor,
        decay_risk: header
            .decay_risk
            .as_deref()
            .map(ReplayDecayRisk::parse)
            .transpose()
            .map_err(replay_error)?,
        items,
        created_at: parse_datetime(&header.created_at, "capsule created_at")?,
    }))
}

#[derive(Debug)]
struct ReplayRow {
    replay_id: String,
    idempotency_key: String,
    capsule_id: String,
    source_receipt_id: String,
    receipt_id: Option<String>,
    algorithm_version: String,
    redaction_generation: i64,
    withheld_slots_json: String,
    result_json: Option<String>,
    privacy_state: String,
    created_at: String,
    capsule_policy_digest: String,
    capsule_source_receipt_id: String,
    capsule_algorithm_version: String,
    capsule_redaction_generation: i64,
    capsule_privacy_state: String,
}

fn load_replay_by_idempotency_key(
    conn: &Connection,
    idempotency_key: &str,
) -> Result<Option<StoredCounterfactualReplay>> {
    load_replay_with_query(
        conn,
        "SELECT r.replay_id, r.idempotency_key, r.capsule_id, r.source_receipt_id,
                r.receipt_id, r.algorithm_version, r.redaction_generation,
                r.withheld_slots_json, r.result_json, r.privacy_state, r.created_at,
                c.policy_digest, c.source_receipt_id, c.algorithm_version,
                c.redaction_generation, c.privacy_state
         FROM counterfactual_replays r
         JOIN retrieval_replay_capsules c ON c.capsule_id = r.capsule_id
         WHERE r.idempotency_key = ?1",
        idempotency_key,
    )
}

fn load_replay_by_id(
    conn: &Connection,
    replay_id: &str,
) -> Result<Option<StoredCounterfactualReplay>> {
    load_replay_with_query(
        conn,
        "SELECT r.replay_id, r.idempotency_key, r.capsule_id, r.source_receipt_id,
                r.receipt_id, r.algorithm_version, r.redaction_generation,
                r.withheld_slots_json, r.result_json, r.privacy_state, r.created_at,
                c.policy_digest, c.source_receipt_id, c.algorithm_version,
                c.redaction_generation, c.privacy_state
         FROM counterfactual_replays r
         JOIN retrieval_replay_capsules c ON c.capsule_id = r.capsule_id
         WHERE r.replay_id = ?1",
        replay_id,
    )
}

fn load_replay_with_query(
    conn: &Connection,
    sql: &str,
    value: &str,
) -> Result<Option<StoredCounterfactualReplay>> {
    let row: Option<ReplayRow> = conn
        .query_row(sql, params![value], |row| {
            Ok(ReplayRow {
                replay_id: row.get(0)?,
                idempotency_key: row.get(1)?,
                capsule_id: row.get(2)?,
                source_receipt_id: row.get(3)?,
                receipt_id: row.get(4)?,
                algorithm_version: row.get(5)?,
                redaction_generation: row.get(6)?,
                withheld_slots_json: row.get(7)?,
                result_json: row.get(8)?,
                privacy_state: row.get(9)?,
                created_at: row.get(10)?,
                capsule_policy_digest: row.get(11)?,
                capsule_source_receipt_id: row.get(12)?,
                capsule_algorithm_version: row.get(13)?,
                capsule_redaction_generation: row.get(14)?,
                capsule_privacy_state: row.get(15)?,
            })
        })
        .optional()?;
    row.map(replay_from_row).transpose()
}

fn replay_from_row(row: ReplayRow) -> Result<StoredCounterfactualReplay> {
    let withheld_slots: Vec<String> =
        serde_json::from_str(&row.withheld_slots_json).map_err(|error| {
            replay_error(ReplayBuildError::InvalidPersistedState(format!(
                "invalid replay withheld slots: {error}"
            )))
        })?;
    let privacy_state = ReplayPrivacyState::parse(&row.privacy_state).map_err(replay_error)?;
    let capsule_privacy_state =
        ReplayPrivacyState::parse(&row.capsule_privacy_state).map_err(replay_error)?;
    let redaction_generation = from_sql_u64(row.redaction_generation, "redaction_generation")?;
    let capsule_redaction_generation = from_sql_u64(
        row.capsule_redaction_generation,
        "capsule redaction_generation",
    )?;
    let normalized_slots = normalize_withheld_slots(&withheld_slots);
    let expected_idempotency_key = replay_idempotency_key(
        &row.algorithm_version,
        &row.source_receipt_id,
        redaction_generation,
        &withheld_slots,
    );
    let basic_integrity_ok = valid_random_public_id(&row.replay_id, "replay_")
        && valid_random_public_id(&row.capsule_id, "rcap_")
        && valid_blake3_digest(&row.idempotency_key, "b3:")
        && row.idempotency_key == expected_idempotency_key
        && valid_opaque_identifier(&row.source_receipt_id)
        && row.algorithm_version == REPLAY_ALGORITHM_VERSION
        && row.source_receipt_id == row.capsule_source_receipt_id
        && row.algorithm_version == row.capsule_algorithm_version
        && redaction_generation <= capsule_redaction_generation
        && privacy_state == capsule_privacy_state
        && valid_blake3_digest(&row.capsule_policy_digest, "b3:")
        && withheld_slots == normalized_slots
        && withheld_slots
            .iter()
            .all(|slot| valid_evidence_slot_shape(slot));
    if !basic_integrity_ok {
        return Err(replay_error(
            ReplayBuildError::PersistedReplayIntegrityMismatch(row.replay_id),
        ));
    }

    let result = if privacy_state == ReplayPrivacyState::Active {
        let result: CounterfactualReplayResult =
            serde_json::from_str(row.result_json.as_deref().ok_or_else(|| {
                replay_error(ReplayBuildError::PersistedReplayIntegrityMismatch(
                    row.replay_id.clone(),
                ))
            })?)
            .map_err(|error| {
                replay_error(ReplayBuildError::InvalidPersistedState(format!(
                    "invalid replay result: {error}"
                )))
            })?;
        if !valid_counterfactual_result(
            &result,
            &row.source_receipt_id,
            &row.capsule_policy_digest,
            redaction_generation,
            &withheld_slots,
        ) {
            return Err(replay_error(
                ReplayBuildError::PersistedReplayIntegrityMismatch(row.replay_id),
            ));
        }
        Some(result)
    } else {
        // Fail closed even if a corrupt/legacy row retained stale JSON.
        None
    };
    Ok(StoredCounterfactualReplay {
        replay_id: row.replay_id,
        idempotency_key: row.idempotency_key,
        capsule_id: row.capsule_id,
        source_receipt_id: row.source_receipt_id,
        receipt_id: row.receipt_id,
        algorithm_version: row.algorithm_version,
        redaction_generation,
        withheld_slots,
        privacy_state,
        result,
        created_at: parse_datetime(&row.created_at, "replay created_at")?,
    })
}

fn invalidate_capsules_in_transaction(
    tx: &Transaction<'_>,
    capsule_ids: &[String],
    reason: ReplayInvalidationReason,
) -> Result<ReplayPrivacyInvalidation> {
    let mut capsules_invalidated = 0u64;
    let mut replays_invalidated = 0u64;
    for capsule_id in capsule_ids {
        let (state, changed) = match reason {
            ReplayInvalidationReason::Suppressed => {
                let changed = tx.execute(
                    "UPDATE retrieval_replay_capsules
                     SET redaction_generation = redaction_generation + 1,
                         privacy_state = 'redacted',
                         replayable = 0,
                         baseline_evidence_digest = NULL,
                         baseline_merkle_root = NULL,
                         item_count = NULL,
                         total_token_estimate = NULL,
                         trust_floor = NULL,
                         decay_risk = NULL
                     WHERE capsule_id = ?1 AND privacy_state = 'active'",
                    params![capsule_id],
                )?;
                (ReplayPrivacyState::Redacted, changed)
            }
            ReplayInvalidationReason::Purged => {
                let changed = tx.execute(
                    "UPDATE retrieval_replay_capsules
                     SET redaction_generation = redaction_generation + 1,
                         privacy_state = 'purged',
                         replayable = 0,
                         baseline_evidence_digest = NULL,
                         baseline_merkle_root = NULL,
                         item_count = NULL,
                         total_token_estimate = NULL,
                         trust_floor = NULL,
                         decay_risk = NULL
                     WHERE capsule_id = ?1 AND privacy_state <> 'purged'",
                    params![capsule_id],
                )?;
                (ReplayPrivacyState::Purged, changed)
            }
        };
        if changed == 0 {
            continue;
        }
        capsules_invalidated += 1;
        match reason {
            ReplayInvalidationReason::Suppressed => {
                // Keep only the private memory dependency locator. This is not
                // returned by public reads and exists solely so a later purge
                // by memory id can discover and erase the redacted capsule.
                tx.execute(
                    "UPDATE retrieval_replay_items
                     SET private_digest = NULL,
                         token_estimate = NULL,
                         trust_score = NULL,
                         decay_risk = NULL
                     WHERE capsule_id = ?1",
                    params![capsule_id],
                )?;
            }
            ReplayInvalidationReason::Purged => {
                // Deleting the rows erases memory ids and the residual item-row
                // cardinality fingerprint. The capsule audit tombstone remains.
                tx.execute(
                    "DELETE FROM retrieval_replay_items WHERE capsule_id = ?1",
                    params![capsule_id],
                )?;
            }
        }
        scrub_dependent_replay_receipts(tx, capsule_id)?;
        replays_invalidated += tx.execute(
            "UPDATE counterfactual_replays
             SET privacy_state = ?1, result_json = NULL
             WHERE capsule_id = ?2
               AND privacy_state <> 'purged'
               AND (privacy_state <> ?1 OR result_json IS NOT NULL)",
            params![state.as_str(), capsule_id],
        )? as u64;
    }
    Ok(ReplayPrivacyInvalidation {
        capsules_invalidated,
        replays_invalidated,
    })
}

fn scrub_dependent_replay_receipts(tx: &Transaction<'_>, capsule_id: &str) -> Result<()> {
    let mut stmt = tx.prepare(
        "SELECT receipt_id FROM counterfactual_replays
         WHERE capsule_id = ?1 AND receipt_id IS NOT NULL",
    )?;
    let rows = stmt.query_map(params![capsule_id], |row| row.get::<_, String>(0))?;
    let mut receipt_ids = Vec::new();
    for row in rows {
        receipt_ids.push(row?);
    }
    drop(stmt);
    for receipt_id in receipt_ids {
        let payload: Option<String> = tx
            .query_row(
                "SELECT payload FROM memory_receipts WHERE receipt_id = ?1",
                params![receipt_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(payload) = payload else {
            continue;
        };
        let mut receipt: Receipt = serde_json::from_str(&payload).map_err(|error| {
            StorageError::Init(format!(
                "replay receipt deserialize during privacy scrub: {error}"
            ))
        })?;
        receipt.retrieved.clear();
        receipt.suppressed.clear();
        receipt.activation_path.clear();
        receipt.mutations.clear();
        receipt.trust_floor = 0.0;
        receipt.decay_risk = crate::trace::DecayRisk::High;
        receipt.evidence = None;
        let scrubbed = serde_json::to_string(&receipt).map_err(|error| {
            StorageError::Init(format!(
                "replay receipt serialize during privacy scrub: {error}"
            ))
        })?;
        tx.execute(
            "UPDATE memory_receipts
             SET retrieved_count = 0, suppressed_count = 0,
                 trust_floor = 0, decay_risk = 'high', payload = ?1
             WHERE receipt_id = ?2",
            params![scrubbed, receipt_id],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the rollback fixture drives a transaction by hand; production
    // writers go through SqliteMemoryStore::begin_write_transaction.
    use rusqlite::TransactionBehavior;

    fn keyed(slot: &str, text: &str) -> String {
        private_evidence_digest(&[7; 32], slot, text.as_bytes())
    }

    fn policy() -> String {
        replay_policy_digest(b"max_tokens=4096;rerank=v3")
    }

    fn frozen_items() -> Vec<FrozenReplayItem> {
        vec![
            FrozenReplayItem {
                ordinal: 0,
                evidence_slot: "evidence_1".into(),
                private_digest: keyed("evidence_1", "alpha"),
                token_estimate: 21,
                trust_score: 0.91,
                decay_risk: ReplayDecayRisk::Low,
            },
            FrozenReplayItem {
                ordinal: 1,
                evidence_slot: "evidence_2".into(),
                private_digest: keyed("evidence_2", "beta"),
                token_estimate: 34,
                trust_score: 0.62,
                decay_risk: ReplayDecayRisk::Medium,
            },
            FrozenReplayItem {
                ordinal: 2,
                evidence_slot: "evidence_3".into(),
                private_digest: keyed("evidence_3", "gamma"),
                token_estimate: 13,
                trust_score: 0.80,
                decay_risk: ReplayDecayRisk::Low,
            },
        ]
    }

    fn capsule_draft(receipt_id: &str) -> RetrievalReplayCapsuleDraft {
        let items = frozen_items()
            .into_iter()
            .map(|item| RetrievalReplayItemDraft {
                evidence_slot: item.evidence_slot,
                memory_id: format!("memory_{}", item.ordinal + 1),
                private_digest: item.private_digest,
                token_estimate: item.token_estimate,
                trust_score: item.trust_score,
                decay_risk: item.decay_risk,
            })
            .collect();
        RetrievalReplayCapsuleDraft {
            source_receipt_id: receipt_id.into(),
            policy_digest: replay_policy_digest(b"max_tokens=4096;rerank=v3"),
            items,
            created_at: DateTime::parse_from_rfc3339("2026-08-10T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    fn test_store(path: &std::path::Path) -> SqliteMemoryStore {
        SqliteMemoryStore::new(Some(path.to_path_buf())).expect("test store")
    }

    fn seed_receipt(store: &SqliteMemoryStore, receipt_id: &str) {
        let writer = store.writer.lock().unwrap();
        writer
            .execute(
                "INSERT OR IGNORE INTO memory_receipts
                     (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
                      trust_floor, decay_risk, payload, created_at)
                 VALUES (?1, NULL, 'test', NULL, 0, 0, 0, 'high', '{}', ?2)",
                params![receipt_id, "2026-08-10T12:00:00Z"],
            )
            .unwrap();
    }

    #[test]
    fn pure_ablation_removes_only_named_slots_without_refill() {
        let items = frozen_items();
        let result =
            ablate_frozen_context("r_source", &policy(), 0, &items, &["evidence_2".into()])
                .unwrap();

        assert_eq!(
            result.baseline.ordered_slots,
            vec!["evidence_1", "evidence_2", "evidence_3"]
        );
        assert_eq!(
            result.counterfactual.ordered_slots,
            vec!["evidence_1", "evidence_3"]
        );
        assert_eq!(result.baseline.item_count, 3);
        assert_eq!(result.counterfactual.item_count, 2);
        assert_eq!(result.replay_influence.removed_item_count, 1);
        assert_eq!(result.replay_influence.removed_token_estimate, 34);
        assert_eq!(result.baseline.trust_floor, 0.62);
        assert_eq!(result.counterfactual.trust_floor, 0.80);
        assert!(result.replay_influence.ordered_evidence_digest_changed);
        assert!(result.replay_influence.merkle_root_changed);
        assert!(result.memory_state_was_read_only);
        assert_eq!(result.baseline.items[1].rank, 2);
        assert_eq!(result.baseline.items[1].token_estimate, 34);
        assert_eq!(result.selection_boundary, REPLAY_SELECTION_BOUNDARY);
        assert_eq!(result.claim_boundary, REPLAY_CLAIM_BOUNDARY);
    }

    #[test]
    fn empty_counterfactual_has_explicit_conservative_metrics() {
        let items = frozen_items();
        let result = ablate_frozen_context(
            "r_source",
            &policy(),
            0,
            &items,
            &[
                "evidence_3".into(),
                "evidence_1".into(),
                "evidence_2".into(),
            ],
        )
        .unwrap();
        assert!(result.counterfactual.ordered_slots.is_empty());
        assert_eq!(result.counterfactual.item_count, 0);
        assert_eq!(result.counterfactual.token_estimate, 0);
        assert_eq!(result.counterfactual.trust_floor, 0.0);
        assert_eq!(result.counterfactual.decay_risk, ReplayDecayRisk::High);
        assert!(
            result
                .counterfactual
                .ordered_evidence_digest
                .starts_with("b3:")
        );
        assert!(result.counterfactual.merkle_root.starts_with("b3:"));
    }

    #[test]
    fn replay_idempotency_normalizes_slot_order_and_duplicates() {
        let first = replay_idempotency_key(
            REPLAY_ALGORITHM_VERSION,
            "r_source",
            3,
            &[
                "evidence_2".into(),
                "evidence_1".into(),
                "evidence_2".into(),
            ],
        );
        let second = replay_idempotency_key(
            REPLAY_ALGORITHM_VERSION,
            "r_source",
            3,
            &["evidence_1".into(), "evidence_2".into()],
        );
        assert_eq!(first, second);
        assert_ne!(
            first,
            replay_idempotency_key(
                REPLAY_ALGORITHM_VERSION,
                "r_source",
                4,
                &["evidence_1".into(), "evidence_2".into()]
            )
        );
    }

    #[test]
    fn digests_are_keyed_deterministic_and_order_sensitive() {
        let first = private_evidence_digest(&[1; 32], "evidence_1", b"same");
        assert_eq!(
            first,
            private_evidence_digest(&[1; 32], "evidence_1", b"same")
        );
        assert_ne!(
            first,
            private_evidence_digest(&[2; 32], "evidence_1", b"same")
        );
        assert_ne!(
            first,
            private_evidence_digest(&[1; 32], "evidence_2", b"same")
        );

        let items = frozen_items();
        let mut reordered = items.clone();
        reordered.swap(0, 1);
        for (index, item) in reordered.iter_mut().enumerate() {
            item.ordinal = index as u32;
        }
        assert_ne!(
            ordered_evidence_set_digest(&items),
            ordered_evidence_set_digest(&reordered)
        );
        assert_ne!(
            evidence_merkle_root(&items),
            evidence_merkle_root(&reordered)
        );
    }

    #[test]
    fn unknown_withheld_slot_is_rejected_instead_of_rerunning_search() {
        let error = ablate_frozen_context(
            "r_source",
            &policy(),
            0,
            &frozen_items(),
            &["evidence_99".into()],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ReplayBuildError::UnknownWithheldSlot("evidence_99".into())
        );
    }

    #[test]
    fn capsule_and_replay_are_idempotent_across_restart() {
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("replay.db");
        let replay_id;
        {
            let store = test_store(&db_path);
            seed_receipt(&store, "r_restart");
            let draft = capsule_draft("r_restart");
            let first = store.save_retrieval_replay_capsule(&draft).unwrap();
            let retry = store.save_retrieval_replay_capsule(&draft).unwrap();
            assert!(!first.reused_existing);
            assert!(retry.reused_existing);
            assert_eq!(first.capsule.capsule_id, retry.capsule.capsule_id);

            let first_replay = store
                .create_context_ablation_replay("r_restart", &["evidence_2".into()])
                .unwrap();
            assert!(!first_replay.reused_existing);
            assert!(valid_random_public_id(&first.capsule.capsule_id, "rcap_"));
            assert!(valid_random_public_id(
                &first_replay.replay.replay_id,
                "replay_"
            ));
            assert!(!first.capsule.capsule_id.contains("r_restart"));
            assert!(
                !first_replay.replay.replay_id.contains(
                    first_replay
                        .replay
                        .idempotency_key
                        .trim_start_matches("b3:")
                )
            );
            let capsule_json = serde_json::to_string(&first.capsule).unwrap();
            assert!(!capsule_json.contains("memory_1"));
            assert!(!capsule_json.contains("b3k:"));
            replay_id = first_replay.replay.replay_id;
        }
        {
            let reopened = test_store(&db_path);
            let retry = reopened
                .create_context_ablation_replay(
                    "r_restart",
                    &["evidence_2".into(), "evidence_2".into()],
                )
                .unwrap();
            assert!(retry.reused_existing);
            assert_eq!(retry.replay.replay_id, replay_id);
            assert_eq!(
                reopened
                    .get_context_ablation_replay(&replay_id)
                    .unwrap()
                    .unwrap(),
                retry.replay
            );
        }
    }

    #[test]
    fn conflicting_capsule_retry_cannot_replace_frozen_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("capsule-conflict.db"));
        seed_receipt(&store, "r_conflict");
        let draft = capsule_draft("r_conflict");
        store.save_retrieval_replay_capsule(&draft).unwrap();

        let mut conflicting = draft;
        conflicting.items[0].token_estimate += 1;
        assert!(store.save_retrieval_replay_capsule(&conflicting).is_err());
        let frozen = store
            .get_retrieval_replay_capsule("r_conflict")
            .unwrap()
            .unwrap();
        assert_eq!(frozen.items[0].token_estimate, 21);
    }

    #[test]
    fn capsule_integrity_mismatch_blocks_replay() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("integrity.db"));
        seed_receipt(&store, "r_integrity");
        let capsule = store
            .save_retrieval_replay_capsule(&capsule_draft("r_integrity"))
            .unwrap()
            .capsule;
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE retrieval_replay_items SET token_estimate = token_estimate + 1
                     WHERE capsule_id = ?1 AND evidence_slot = 'evidence_1'",
                    params![capsule.capsule_id],
                )
                .unwrap();
        }
        assert!(
            store
                .create_context_ablation_replay("r_integrity", &["evidence_2".into()])
                .is_err()
        );
        let replay_count: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row("SELECT COUNT(*) FROM counterfactual_replays", [], |row| {
                    row.get(0)
                })
                .unwrap()
        };
        assert_eq!(replay_count, 0);
    }

    #[test]
    fn replay_rows_fail_closed_on_privacy_or_payload_inconsistency() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("replay-integrity.db"));
        seed_receipt(&store, "r_replay_integrity");
        store
            .save_retrieval_replay_capsule(&capsule_draft("r_replay_integrity"))
            .unwrap();
        let replay_id = store
            .create_context_ablation_replay("r_replay_integrity", &["evidence_2".into()])
            .unwrap()
            .replay
            .replay_id;

        // A stale payload on a redacted row is never parsed or returned.
        let stale_payload: String = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT result_json FROM counterfactual_replays WHERE replay_id = ?1",
                    params![replay_id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        store
            .invalidate_replay_evidence_for_memory("memory_1", ReplayInvalidationReason::Suppressed)
            .unwrap();
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE counterfactual_replays SET result_json = ?1
                     WHERE replay_id = ?2",
                    params![stale_payload, replay_id],
                )
                .unwrap();
        }
        let redacted = store
            .get_context_ablation_replay(&replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(redacted.privacy_state, ReplayPrivacyState::Redacted);
        assert!(redacted.result.is_none());

        // Active requires a complete, internally consistent result.
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE counterfactual_replays
                     SET privacy_state = 'active', result_json = NULL
                     WHERE replay_id = ?1",
                    params![replay_id],
                )
                .unwrap();
        }
        assert!(store.get_context_ablation_replay(&replay_id).is_err());
    }

    #[test]
    fn tampered_idempotent_replay_is_not_authoritative() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("idempotency-integrity.db"));
        seed_receipt(&store, "r_idempotency_integrity");
        store
            .save_retrieval_replay_capsule(&capsule_draft("r_idempotency_integrity"))
            .unwrap();
        let replay_id = store
            .create_context_ablation_replay("r_idempotency_integrity", &["evidence_2".into()])
            .unwrap()
            .replay
            .replay_id;
        {
            let writer = store.writer.lock().unwrap();
            let payload: String = writer
                .query_row(
                    "SELECT result_json FROM counterfactual_replays WHERE replay_id = ?1",
                    params![replay_id],
                    |row| row.get(0),
                )
                .unwrap();
            let mut value: serde_json::Value = serde_json::from_str(&payload).unwrap();
            value["policyDigest"] = serde_json::Value::String(format!("b3:{}", "0".repeat(64)));
            writer
                .execute(
                    "UPDATE counterfactual_replays SET result_json = ?1 WHERE replay_id = ?2",
                    params![serde_json::to_string(&value).unwrap(), replay_id],
                )
                .unwrap();
        }
        assert!(
            store
                .create_context_ablation_replay("r_idempotency_integrity", &["evidence_2".into()])
                .is_err()
        );
    }

    #[test]
    fn receipt_linking_is_idempotent_and_conflict_safe() {
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("receipt-link.db");
        let store = test_store(&db_path);
        seed_receipt(&store, "r_link_source");
        seed_receipt(&store, "r_link_a");
        seed_receipt(&store, "r_link_b");
        store
            .save_retrieval_replay_capsule(&capsule_draft("r_link_source"))
            .unwrap();
        let replay_id = store
            .create_context_ablation_replay("r_link_source", &["evidence_2".into()])
            .unwrap()
            .replay
            .replay_id;
        store
            .link_context_ablation_receipt(&replay_id, "r_link_a")
            .unwrap();
        store
            .link_context_ablation_receipt(&replay_id, "r_link_a")
            .unwrap();
        assert!(
            store
                .link_context_ablation_receipt(&replay_id, "r_link_b")
                .is_err()
        );
        assert_eq!(
            store
                .get_context_ablation_replay(&replay_id)
                .unwrap()
                .unwrap()
                .receipt_id
                .as_deref(),
            Some("r_link_a")
        );
    }

    #[test]
    fn typed_replay_receipt_persists_and_links_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("typed-replay-receipt.db"));
        seed_receipt(&store, "r_typed_source");
        store
            .save_retrieval_replay_capsule(&capsule_draft("r_typed_source"))
            .unwrap();
        let replay = store
            .create_context_ablation_replay("r_typed_source", &["evidence_2".into()])
            .unwrap()
            .replay;
        let result = replay.result.clone().unwrap();
        let receipt = Receipt::build(
            Utc::now(),
            "replay",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[result.counterfactual.trust_floor],
            Vec::new(),
        )
        .with_evidence(crate::trace::ReceiptEvidence::CounterfactualReplay {
            schema: "https://vestige.dev/schemas/receipt/counterfactual-replay/v1".into(),
            schema_version: 1,
            replay_id: replay.replay_id.clone(),
            capsule_id: replay.capsule_id.clone(),
            result,
        });

        store
            .save_counterfactual_replay_receipt(
                &replay.replay_id,
                &receipt,
                Some("run_replay"),
                Some("receipt"),
            )
            .unwrap();
        let linked = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            linked.receipt_id.as_deref(),
            Some(receipt.receipt_id.as_str())
        );
        let persisted = store.get_receipt(&receipt.receipt_id).unwrap().unwrap();
        assert_eq!(persisted, receipt);
        assert!(matches!(
            persisted.evidence,
            Some(crate::trace::ReceiptEvidence::CounterfactualReplay { .. })
        ));

        store
            .invalidate_replay_evidence_for_memory("memory_1", ReplayInvalidationReason::Suppressed)
            .unwrap();
        let scrubbed = store.get_receipt(&receipt.receipt_id).unwrap().unwrap();
        assert!(scrubbed.evidence.is_none());
        assert_eq!(scrubbed.trust_floor, 0.0);
        assert_eq!(scrubbed.decay_risk, crate::trace::DecayRisk::High);
        let payload: String = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT payload FROM memory_receipts WHERE receipt_id = ?1",
                    params![receipt.receipt_id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert!(!payload.contains("counterfactual_replay"));
        assert!(!payload.contains("evidence_1"));
        assert!(!payload.contains(REPLAY_CLAIM_BOUNDARY));
    }

    #[test]
    fn legacy_receipt_without_capsule_is_not_replayed() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("legacy.db"));
        seed_receipt(&store, "r_legacy");
        assert!(
            store
                .create_context_ablation_replay("r_legacy", &["evidence_1".into()])
                .is_err()
        );
        let replay_count: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row("SELECT COUNT(*) FROM counterfactual_replays", [], |row| {
                    row.get(0)
                })
                .unwrap()
        };
        assert_eq!(replay_count, 0);
    }

    #[test]
    fn receipt_and_capsule_transaction_rolls_back_as_one_unit() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("atomic-capsule.db"));
        {
            let mut writer = store.writer.lock().unwrap();
            let tx = writer
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            tx.execute(
                "INSERT INTO memory_receipts
                     (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
                      trust_floor, decay_risk, payload, created_at)
                 VALUES ('r_atomic_rollback', NULL, 'test', NULL, 0, 0, 0,
                         'high', '{}', '2026-08-10T12:00:00Z')",
                [],
            )
            .unwrap();
            SqliteMemoryStore::save_retrieval_replay_capsule_in_transaction(
                &tx,
                &capsule_draft("r_atomic_rollback"),
            )
            .unwrap();
            // Simulate receipt serialization/dispatch failure: dropping the
            // transaction rolls both rows back.
        }
        let (receipt_count, capsule_count): (i64, i64) = {
            let reader = store.reader.lock().unwrap();
            (
                reader
                    .query_row(
                        "SELECT COUNT(*) FROM memory_receipts
                         WHERE receipt_id = 'r_atomic_rollback'",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap(),
                reader
                    .query_row(
                        "SELECT COUNT(*) FROM retrieval_replay_capsules
                         WHERE source_receipt_id = 'r_atomic_rollback'",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap(),
            )
        };
        assert_eq!((receipt_count, capsule_count), (0, 0));
    }

    #[test]
    fn replay_does_not_mutate_knowledge_node_state() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("read-only.db"));
        seed_receipt(&store, "r_read_only");
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "INSERT INTO knowledge_nodes
                         (id, content, created_at, updated_at, last_accessed,
                          reps, retrieval_strength, retention_strength)
                     VALUES ('memory_1', 'private content', ?1, ?1, ?1, 7, 0.43, 0.67)",
                    params!["2026-08-10T00:00:00Z"],
                )
                .unwrap();
        }
        let before: (String, i64, f64, f64, String) = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT content, reps, retrieval_strength, retention_strength, updated_at
                     FROM knowledge_nodes WHERE id = 'memory_1'",
                    [],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .unwrap()
        };

        store
            .save_retrieval_replay_capsule(&capsule_draft("r_read_only"))
            .unwrap();
        store
            .create_context_ablation_replay("r_read_only", &["evidence_1".into()])
            .unwrap();

        let after: (String, i64, f64, f64, String) = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT content, reps, retrieval_strength, retention_strength, updated_at
                     FROM knowledge_nodes WHERE id = 'memory_1'",
                    [],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .unwrap()
        };
        assert_eq!(before, after);
    }

    #[test]
    fn suppression_and_purge_scrub_private_and_size_fingerprints() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("privacy.db"));
        seed_receipt(&store, "r_privacy");
        let capsule = store
            .save_retrieval_replay_capsule(&capsule_draft("r_privacy"))
            .unwrap()
            .capsule;
        let replay = store
            .create_context_ablation_replay("r_privacy", &["evidence_1".into()])
            .unwrap()
            .replay;

        let report = store
            .invalidate_replay_evidence_for_memory("memory_1", ReplayInvalidationReason::Suppressed)
            .unwrap();
        assert_eq!(report.capsules_invalidated, 1);
        assert_eq!(report.replays_invalidated, 1);

        let redacted = store
            .get_retrieval_replay_capsule("r_privacy")
            .unwrap()
            .unwrap();
        assert_eq!(redacted.privacy_state, ReplayPrivacyState::Redacted);
        assert!(!redacted.replayable);
        assert_eq!(redacted.redaction_generation, 1);
        assert!(redacted.baseline_evidence_digest.is_none());
        assert!(redacted.baseline_merkle_root.is_none());
        assert!(redacted.item_count.is_none());
        assert!(redacted.total_token_estimate.is_none());
        assert!(redacted.items.is_empty());
        let public_json = serde_json::to_string(&redacted).unwrap();
        assert!(!public_json.contains("memory_1"));
        assert!(!public_json.contains("evidence_1"));
        assert!(!public_json.contains("b3k:"));
        let retained_private_locators: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM retrieval_replay_items
                     WHERE capsule_id = ?1
                       AND memory_id IS NOT NULL
                       AND private_digest IS NULL
                       AND token_estimate IS NULL",
                    params![capsule.capsule_id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(retained_private_locators, 3);
        let redacted_replay = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(redacted_replay.privacy_state, ReplayPrivacyState::Redacted);
        assert!(redacted_replay.result.is_none());
        assert_eq!(
            store
                .verify_replay_materialization_digest(
                    "r_privacy",
                    "evidence_1",
                    &keyed("evidence_1", "alpha")
                )
                .unwrap(),
            ReplayMaterializationCheck::Unavailable
        );
        assert!(
            store
                .create_context_ablation_replay("r_privacy", &["evidence_2".into()])
                .is_err()
        );

        let purge = store
            .invalidate_replay_evidence_for_memory("memory_1", ReplayInvalidationReason::Purged)
            .unwrap();
        assert_eq!(purge.capsules_invalidated, 1);
        assert_eq!(purge.replays_invalidated, 1);
        let purged = store
            .get_retrieval_replay_capsule("r_privacy")
            .unwrap()
            .unwrap();
        assert_eq!(purged.privacy_state, ReplayPrivacyState::Purged);
        assert_eq!(purged.redaction_generation, 2);
        assert!(purged.items.is_empty());
        let purged_replay = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(purged_replay.privacy_state, ReplayPrivacyState::Purged);
        assert!(purged_replay.result.is_none());

        // State transitions are monotonic; suppression can never downgrade a
        // purged audit tombstone back to redacted.
        let downgrade = store
            .invalidate_replay_capsule(&capsule.capsule_id, ReplayInvalidationReason::Suppressed)
            .unwrap();
        assert_eq!(downgrade.capsules_invalidated, 0);
        assert_eq!(
            store
                .get_retrieval_replay_capsule("r_privacy")
                .unwrap()
                .unwrap()
                .privacy_state,
            ReplayPrivacyState::Purged
        );
    }

    #[test]
    fn memory_lifecycle_hooks_invalidate_replay_and_erase_dependency_rows() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("lifecycle-hooks.db"));
        let node = store
            .ingest(crate::memory::IngestInput {
                content: "memory whose replay evidence follows lifecycle privacy".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .unwrap();
        seed_receipt(&store, "r_lifecycle_hooks");
        let mut draft = capsule_draft("r_lifecycle_hooks");
        draft.items[0].memory_id = node.id.clone();
        let capsule = store.save_retrieval_replay_capsule(&draft).unwrap().capsule;
        let replay = store
            .create_context_ablation_replay("r_lifecycle_hooks", &["evidence_1".into()])
            .unwrap()
            .replay;

        store.suppress_memory(&node.id).unwrap();

        let redacted = store
            .get_retrieval_replay_capsule("r_lifecycle_hooks")
            .unwrap()
            .unwrap();
        assert_eq!(redacted.privacy_state, ReplayPrivacyState::Redacted);
        assert!(!redacted.replayable);
        assert!(redacted.items.is_empty());
        let redacted_replay = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(redacted_replay.privacy_state, ReplayPrivacyState::Redacted);
        assert!(redacted_replay.result.is_none());
        assert!(
            store
                .create_context_ablation_replay("r_lifecycle_hooks", &["evidence_2".into()],)
                .is_err(),
            "suppression must make the public replay surface unavailable"
        );
        let (retained_locators, retained_protected_values): (i64, i64) = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*),
                            COUNT(private_digest) + COUNT(token_estimate)
                              + COUNT(trust_score) + COUNT(decay_risk)
                     FROM retrieval_replay_items WHERE capsule_id = ?1",
                    params![capsule.capsule_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap()
        };
        assert_eq!(retained_locators, 3);
        assert_eq!(retained_protected_values, 0);

        store
            .purge_node(&node.id, Some("replay lifecycle integration test"))
            .unwrap();

        let purged = store
            .get_retrieval_replay_capsule("r_lifecycle_hooks")
            .unwrap()
            .unwrap();
        assert_eq!(purged.privacy_state, ReplayPrivacyState::Purged);
        assert!(!purged.replayable);
        assert!(purged.items.is_empty());
        let purged_replay = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(purged_replay.privacy_state, ReplayPrivacyState::Purged);
        assert!(purged_replay.result.is_none());
        let residual_dependency_rows: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM retrieval_replay_items WHERE capsule_id = ?1",
                    params![capsule.capsule_id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(residual_dependency_rows, 0);
    }

    #[test]
    fn legacy_delete_invalidates_replay_without_claiming_full_erasure() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("legacy-delete-hook.db"));
        let node = store
            .ingest(crate::memory::IngestInput {
                content: "memory removed through the legacy delete path".into(),
                node_type: "fact".into(),
                ..Default::default()
            })
            .unwrap();
        seed_receipt(&store, "r_legacy_delete_hook");
        let mut draft = capsule_draft("r_legacy_delete_hook");
        draft.items[0].memory_id = node.id.clone();
        let capsule = store.save_retrieval_replay_capsule(&draft).unwrap().capsule;
        let replay = store
            .create_context_ablation_replay("r_legacy_delete_hook", &["evidence_1".into()])
            .unwrap()
            .replay;

        assert!(store.delete_node(&node.id).unwrap());

        let deleted_capsule = store
            .get_retrieval_replay_capsule("r_legacy_delete_hook")
            .unwrap()
            .unwrap();
        assert_eq!(deleted_capsule.privacy_state, ReplayPrivacyState::Purged);
        assert!(!deleted_capsule.replayable);
        assert!(deleted_capsule.items.is_empty());
        let deleted_replay = store
            .get_context_ablation_replay(&replay.replay_id)
            .unwrap()
            .unwrap();
        assert_eq!(deleted_replay.privacy_state, ReplayPrivacyState::Purged);
        assert!(deleted_replay.result.is_none());
        let residual_dependency_rows: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM retrieval_replay_items WHERE capsule_id = ?1",
                    params![capsule.capsule_id],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(residual_dependency_rows, 0);
    }

    #[test]
    fn materialization_requires_current_content_digest_match() {
        let temp = tempfile::tempdir().unwrap();
        let store = test_store(&temp.path().join("materialize.db"));
        seed_receipt(&store, "r_materialize");
        store
            .save_retrieval_replay_capsule(&capsule_draft("r_materialize"))
            .unwrap();
        assert_eq!(
            store
                .verify_replay_materialization_digest(
                    "r_materialize",
                    "evidence_1",
                    &keyed("evidence_1", "alpha")
                )
                .unwrap(),
            ReplayMaterializationCheck::Match
        );
        assert_eq!(
            store
                .verify_replay_materialization_digest(
                    "r_materialize",
                    "evidence_1",
                    &keyed("evidence_1", "edited")
                )
                .unwrap(),
            ReplayMaterializationCheck::ContentChanged
        );
    }

    #[test]
    fn replay_result_json_contains_no_memory_id_or_raw_content() {
        let result = ablate_frozen_context(
            "r_source",
            &policy(),
            0,
            &frozen_items(),
            &["evidence_1".into()],
        )
        .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("memory_1"));
        assert!(!json.contains("alpha"));
        assert!(!json.contains("b3k:"));
        assert!(json.contains(REPLAY_CLAIM_BOUNDARY));
    }
}
