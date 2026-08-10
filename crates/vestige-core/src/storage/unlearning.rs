//! Pure models and verification primitives for Verified Local Unlearning.
//!
//! This module deliberately does **not** delete data. It defines the contract
//! that the V22 storage/execution layer must satisfy before Vestige may return a
//! verified result:
//!
//! - post-lineage artifacts are closed transitively;
//! - anti-resurrection records contain commitments, never the erased plaintext;
//! - every required local postcondition is positively checked; and
//! - the result carries explicit exclusions for copies Vestige does not control.
//!
//! Two scopes are intentionally distinct. [`UnlearningScope::LegacyAuditedPurge`]
//! can report what an older purge inspected, but can never become a verified
//! influence-erasure claim because legacy artifacts have incomplete lineage.
//! [`UnlearningScope::PostLineageVerifiedLocal`] is eligible for verification,
//! but only inside Vestige-controlled local state and registered managed
//! artifacts. Neither scope claims erasure of unmanaged copies, forensic media
//! remnants, provider-retained backups, or external model weights.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Stable schema marker for signed/serialized unlearning results.
pub const VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1: &str =
    "https://vestige.dev/schemas/unlearning/verified-local/v1";

/// Domain used to derive the keyed BLAKE3 commitment key.
const COMMITMENT_KEY_CONTEXT: &str = "vestige.dev verified-local-unlearning commitment key v1";

/// Minimum installation/sync-group secret size accepted by the commitment API.
///
/// The KDF cannot add entropy to a weak secret. The integration layer should
/// obtain this from an OS keystore or an equivalently protected 256-bit secret.
pub const MIN_COMMITMENT_SECRET_BYTES: usize = 32;

/// Largest SQLite-compatible generation accepted by the pure model.
pub const MAX_ERASURE_GENERATION: u64 = i64::MAX as u64;

/// Maximum byte length for implementation-controlled ledger codes.
pub const MAX_CLOSED_CODE_BYTES: usize = 96;

/// Proposed V22 schema consumed by the future SQLite integration.
///
/// This constant is intentionally not wired into `migrations.rs` in this patch.
/// The integration migration may add timestamps/indexes, but must preserve these
/// semantic constraints:
///
/// - lineage rows are the only raw-ID table introduced here and are removed as
///   part of the same unlearning transaction as their artifacts;
/// - jobs/tombstones/steps retain commitments only, never target ID or content;
/// - commitment key identity is retained so key rotation cannot silently make
///   an old tombstone unverifiable;
/// - generation allocation is serialized through a singleton counter; and
/// - step details are closed codes, not arbitrary content-bearing text.
pub const V22_UNLEARNING_SCHEMA_ASSUMPTION: &str = r#"
CREATE TABLE artifact_lineage (
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    derived_kind TEXT NOT NULL,
    derived_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_kind, source_id, derived_kind, derived_id, relation)
) STRICT;
CREATE INDEX idx_artifact_lineage_source
    ON artifact_lineage(source_kind, source_id);
CREATE INDEX idx_artifact_lineage_derived
    ON artifact_lineage(derived_kind, derived_id);

CREATE TABLE erasure_jobs (
    erasure_id TEXT PRIMARY KEY,
    schema_uri TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    generation INTEGER NOT NULL UNIQUE CHECK (generation > 0),
    commitment_key_id TEXT NOT NULL,
    lineage_epoch INTEGER NOT NULL CHECK (lineage_epoch >= 0),
    fence_commitment TEXT NOT NULL,
    target_commitment TEXT NOT NULL,
    exact_content_commitment TEXT NOT NULL,
    source_locator_commitment TEXT,
    closure_commitment TEXT NOT NULL,
    started_at TEXT NOT NULL,
    committed_at TEXT,
    result_json TEXT,
    signature_json TEXT
) STRICT;

CREATE TABLE erasure_steps (
    erasure_id TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    surface TEXT NOT NULL,
    action TEXT NOT NULL,
    matched_count INTEGER NOT NULL,
    changed_count INTEGER NOT NULL,
    verification_status TEXT NOT NULL,
    detail_code TEXT NOT NULL,
    PRIMARY KEY (erasure_id, step_order),
    FOREIGN KEY (erasure_id) REFERENCES erasure_jobs(erasure_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE erasure_tombstones (
    target_commitment TEXT PRIMARY KEY,
    exact_content_commitment TEXT NOT NULL,
    source_locator_commitment TEXT,
    commitment_key_id TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation > 0),
    scope TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    erased_at TEXT NOT NULL
) STRICT;
CREATE UNIQUE INDEX idx_erasure_tombstones_generation
    ON erasure_tombstones(generation);
CREATE INDEX idx_erasure_tombstones_content
    ON erasure_tombstones(exact_content_commitment);
CREATE INDEX idx_erasure_tombstones_source
    ON erasure_tombstones(source_locator_commitment)
    WHERE source_locator_commitment IS NOT NULL;

CREATE TABLE erasure_generation_counter (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_generation INTEGER NOT NULL CHECK (last_generation >= 0)
) STRICT;
INSERT INTO erasure_generation_counter(singleton, last_generation) VALUES (1, 0);

CREATE TABLE managed_artifacts (
    artifact_commitment TEXT PRIMARY KEY,
    artifact_kind TEXT NOT NULL,
    locator_commitment TEXT NOT NULL,
    commitment_key_id TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_verified_at TEXT
) STRICT;
"#;

/// Claim scope for one unlearning operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnlearningScope {
    /// Pre-lineage data. This can prove only which surfaces were inspected and
    /// changed, not absence of influence in unknown derived artifacts.
    LegacyAuditedPurge,
    /// Data created after complete lineage capture was enabled. Verification is
    /// limited to Vestige-controlled local state and registered managed artifacts.
    PostLineageVerifiedLocal,
}

impl UnlearningScope {
    /// Whether this scope may ever produce a verified-within-scope verdict.
    pub const fn is_verifiable(self) -> bool {
        matches!(self, Self::PostLineageVerifiedLocal)
    }

    /// Non-negotiable exclusions attached to this scope.
    pub fn exclusions(self) -> Vec<GuaranteeExclusion> {
        let mut exclusions = vec![
            GuaranteeExclusion::UnmanagedCopies,
            GuaranteeExclusion::PlaintextMediaForensics,
            GuaranteeExclusion::ProviderRetainedBackups,
            GuaranteeExclusion::ExternalModelWeights,
            GuaranteeExclusion::FreshUserReingest,
        ];
        if self == Self::LegacyAuditedPurge {
            exclusions.push(GuaranteeExclusion::UntrackedLegacyDerivedInfluence);
        }
        exclusions
    }
}

/// Boundaries that no local receipt may silently elide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuaranteeExclusion {
    UnmanagedCopies,
    PlaintextMediaForensics,
    ProviderRetainedBackups,
    ExternalModelWeights,
    FreshUserReingest,
    UntrackedLegacyDerivedInfluence,
}

/// Kinds of artifacts that may participate in lineage closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    KnowledgeNode,
    Embedding,
    FtsDocument,
    VectorEntry,
    Insight,
    TemporalSummary,
    DomainProjection,
    Intention,
    MergePlan,
    MergeOperation,
    CompositionEvent,
    TraceEvent,
    Receipt,
    MemoryPr,
    SynapticTag,
    SynapticEvent,
    SynapticCaptureItem,
    RuntimeProjection,
    ManagedBackup,
    ManagedExport,
    ManagedSyncObject,
    ConnectorSource,
}

impl ArtifactKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KnowledgeNode => "knowledge_node",
            Self::Embedding => "embedding",
            Self::FtsDocument => "fts_document",
            Self::VectorEntry => "vector_entry",
            Self::Insight => "insight",
            Self::TemporalSummary => "temporal_summary",
            Self::DomainProjection => "domain_projection",
            Self::Intention => "intention",
            Self::MergePlan => "merge_plan",
            Self::MergeOperation => "merge_operation",
            Self::CompositionEvent => "composition_event",
            Self::TraceEvent => "trace_event",
            Self::Receipt => "receipt",
            Self::MemoryPr => "memory_pr",
            Self::SynapticTag => "synaptic_tag",
            Self::SynapticEvent => "synaptic_event",
            Self::SynapticCaptureItem => "synaptic_capture_item",
            Self::RuntimeProjection => "runtime_projection",
            Self::ManagedBackup => "managed_backup",
            Self::ManagedExport => "managed_export",
            Self::ManagedSyncObject => "managed_sync_object",
            Self::ConnectorSource => "connector_source",
        }
    }
}

/// A typed artifact identifier used only while computing/executing closure.
///
/// Raw identifiers from this type must not be copied into the durable erasure
/// ledger. The ledger stores [`Commitment`] values instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub id: String,
}

impl ArtifactRef {
    pub fn new(kind: ArtifactKind, id: impl Into<String>) -> Result<Self, UnlearningModelError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(UnlearningModelError::EmptyArtifactId);
        }
        Ok(Self { kind, id })
    }
}

/// Why one artifact depends on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineageRelation {
    DerivedFrom,
    Embeds,
    Indexes,
    Summarizes,
    Aggregates,
    Mentions,
    Mutates,
    Mirrors,
}

/// Directed lineage edge: `source -> derived`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineageEdge {
    pub source: ArtifactRef,
    pub derived: ArtifactRef,
    pub relation: LineageRelation,
}

/// One member of a deterministic transitive closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClosureMember {
    artifact: ArtifactRef,
    /// Minimum directed distance from the root. Root distance is zero.
    distance: usize,
}

impl ClosureMember {
    pub fn artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    pub const fn distance(&self) -> usize {
        self.distance
    }
}

/// Complete transitive lineage closure for one root artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LineageClosure {
    root: ArtifactRef,
    members: Vec<ClosureMember>,
}

impl LineageClosure {
    pub fn root(&self) -> &ArtifactRef {
        &self.root
    }

    pub fn members(&self) -> &[ClosureMember] {
        &self.members
    }

    pub fn contains(&self, artifact: &ArtifactRef) -> bool {
        self.members
            .iter()
            .any(|member| &member.artifact == artifact)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Compute the deterministic, cycle-safe transitive closure of `root`.
///
/// Edge order and duplicates do not affect the result. Members are sorted by
/// minimum distance and then by typed artifact key, making the closure stable
/// enough to commit into a receipt.
pub fn compute_lineage_closure(root: &ArtifactRef, edges: &[LineageEdge]) -> LineageClosure {
    let mut adjacency: BTreeMap<ArtifactRef, BTreeSet<ArtifactRef>> = BTreeMap::new();
    for edge in edges {
        adjacency
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.derived.clone());
    }

    let mut minimum_distance = BTreeMap::<ArtifactRef, usize>::new();
    let mut queue = VecDeque::new();
    minimum_distance.insert(root.clone(), 0);
    queue.push_back(root.clone());

    while let Some(current) = queue.pop_front() {
        let distance = minimum_distance[&current];
        let Some(children) = adjacency.get(&current) else {
            continue;
        };
        for child in children {
            if !minimum_distance.contains_key(child) {
                minimum_distance.insert(child.clone(), distance + 1);
                queue.push_back(child.clone());
            }
        }
    }

    let mut members: Vec<_> = minimum_distance
        .into_iter()
        .map(|(artifact, distance)| ClosureMember { artifact, distance })
        .collect();
    members.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then_with(|| a.artifact.cmp(&b.artifact))
    });

    LineageClosure {
        root: root.clone(),
        members,
    }
}

/// Secret key used for keyed anti-resurrection commitments.
///
/// It intentionally cannot be cloned, implements a redacted `Debug`, and
/// overwrites its owned bytes on drop. The integration layer remains responsible
/// for sourcing the secret from protected storage and using a guaranteed-zeroize
/// wrapper when the storage integration adds that dependency.
pub struct CommitmentKey {
    bytes: [u8; 32],
    key_id: String,
}

impl CommitmentKey {
    pub fn derive(secret: &[u8]) -> Result<Self, UnlearningModelError> {
        if secret.len() < MIN_COMMITMENT_SECRET_BYTES {
            return Err(UnlearningModelError::CommitmentSecretTooShort {
                minimum: MIN_COMMITMENT_SECRET_BYTES,
                actual: secret.len(),
            });
        }
        let bytes = blake3::derive_key(COMMITMENT_KEY_CONTEXT, secret);
        let mut id_hasher = blake3::Hasher::new_derive_key(
            "vestige.dev verified-local-unlearning commitment key id v1",
        );
        id_hasher.update(&bytes);
        let mut key_id = id_hasher.finalize().to_hex().to_string();
        key_id.truncate(32);
        Ok(Self { bytes, key_id })
    }

    /// Non-secret identifier that binds tombstones to their commitment key.
    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

impl fmt::Debug for CommitmentKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CommitmentKey(<redacted>)")
    }
}

impl Drop for CommitmentKey {
    fn drop(&mut self) {
        self.bytes.fill(0);
    }
}

/// Domain of one anti-resurrection commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentKind {
    TargetIdentifier,
    ExactContent,
    SourceLocator,
    LineageClosure,
    ExecutionFence,
    PostconditionReport,
    ManagedArtifactLocator,
}

impl CommitmentKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TargetIdentifier => "target_identifier",
            Self::ExactContent => "exact_content",
            Self::SourceLocator => "source_locator",
            Self::LineageClosure => "lineage_closure",
            Self::ExecutionFence => "execution_fence",
            Self::PostconditionReport => "postcondition_report",
            Self::ManagedArtifactLocator => "managed_artifact_locator",
        }
    }
}

/// A versioned keyed digest safe to persist after its input is erased.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Commitment {
    pub version: u32,
    pub kind: CommitmentKind,
    pub key_id: String,
    pub digest: String,
}

impl Commitment {
    pub fn matches(&self, key: &CommitmentKey, value: &[u8]) -> bool {
        *self == commit_bytes(key, self.kind, value)
    }

    pub fn is_well_formed_for(&self, key: &CommitmentKey, kind: CommitmentKind) -> bool {
        self.version == 1
            && self.kind == kind
            && self.key_id == key.key_id()
            && is_lower_hex_digest(&self.digest)
    }
}

/// Domain-separated keyed commitment over exact bytes.
pub fn commit_bytes(key: &CommitmentKey, kind: CommitmentKind, value: &[u8]) -> Commitment {
    let kind_bytes = kind.as_str().as_bytes();
    let mut hasher = blake3::Hasher::new_keyed(&key.bytes);
    hasher.update(b"vestige.verified-local-unlearning.commitment.v1");
    hasher.update(&(kind_bytes.len() as u64).to_le_bytes());
    hasher.update(kind_bytes);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
    Commitment {
        version: 1,
        kind,
        key_id: key.key_id.clone(),
        digest: hasher.finalize().to_hex().to_string(),
    }
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Commitments used by restore, import, sync and connector gates.
///
/// `exact_content` matches byte-for-byte resurrection. It intentionally does not
/// claim to detect paraphrases or independently re-entered facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AntiResurrectionCommitments {
    pub target_identifier: Commitment,
    pub exact_content: Commitment,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_locator: Option<Commitment>,
}

pub fn anti_resurrection_commitments(
    key: &CommitmentKey,
    target_identifier: &str,
    exact_content: &[u8],
    source_locator: Option<&str>,
) -> AntiResurrectionCommitments {
    AntiResurrectionCommitments {
        target_identifier: commit_bytes(
            key,
            CommitmentKind::TargetIdentifier,
            target_identifier.as_bytes(),
        ),
        exact_content: commit_bytes(key, CommitmentKind::ExactContent, exact_content),
        source_locator: source_locator.map(|source| {
            commit_bytes(key, CommitmentKind::SourceLocator, source.trim().as_bytes())
        }),
    }
}

/// Commit a closure without exposing any of its identifiers in the ledger.
pub fn commit_lineage_closure(
    key: &CommitmentKey,
    closure: &LineageClosure,
) -> Result<Commitment, UnlearningModelError> {
    validate_lineage_closure(closure)?;
    let canonical = serde_json::to_vec(closure)
        .map_err(|error| UnlearningModelError::Serialization(error.to_string()))?;
    Ok(commit_bytes(
        key,
        CommitmentKind::LineageClosure,
        &canonical,
    ))
}

fn validate_lineage_closure(closure: &LineageClosure) -> Result<(), UnlearningModelError> {
    if closure.members.is_empty()
        || closure.members[0].artifact != closure.root
        || closure.members[0].distance != 0
    {
        return Err(UnlearningModelError::InvalidLineageClosure);
    }

    let mut previous: Option<(usize, &ArtifactRef)> = None;
    for member in &closure.members {
        if let Some((distance, artifact)) = previous
            && (member.distance, &member.artifact) <= (distance, artifact)
        {
            return Err(UnlearningModelError::InvalidLineageClosure);
        }
        previous = Some((member.distance, &member.artifact));
    }
    Ok(())
}

/// Typestate token created only after the integration layer has blocked local
/// writes, imports, restores, sync, and connector mutation for one lineage epoch.
///
/// This type does not itself acquire a database lock. Its private fields make the
/// handoff explicit: the executor must hold the real fence before constructing
/// the token, keep it held through deletion and verification, and compare the
/// epoch again before committing the ledger record.
#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationFence {
    lineage_epoch: u64,
    commitment: Commitment,
}

impl VerificationFence {
    pub fn from_held_epoch(
        key: &CommitmentKey,
        lineage_epoch: u64,
        opaque_fence_token: &[u8],
    ) -> Result<Self, UnlearningModelError> {
        if opaque_fence_token.len() < 16 {
            return Err(UnlearningModelError::FenceTokenTooShort {
                minimum: 16,
                actual: opaque_fence_token.len(),
            });
        }
        let mut material = Vec::with_capacity(8 + opaque_fence_token.len());
        material.extend_from_slice(&lineage_epoch.to_le_bytes());
        material.extend_from_slice(opaque_fence_token);
        Ok(Self {
            lineage_epoch,
            commitment: commit_bytes(key, CommitmentKind::ExecutionFence, &material),
        })
    }

    pub const fn lineage_epoch(&self) -> u64 {
        self.lineage_epoch
    }

    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
}

/// Storage/runtime surfaces reported by an execution step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnlearningSurface {
    CanonicalSqlite,
    LinkedSqliteArtifacts,
    FullTextIndex,
    EmbeddingStore,
    VectorIndex,
    CognitiveRuntime,
    TraceAndReceiptEvidence,
    ReviewAndUndoQueues,
    ConnectorAndRestoreGates,
    ManagedBackups,
    ManagedExports,
    ManagedSyncObjects,
    ErasureLedgerAnchor,
}

/// Every managed surface that must be inspected for post-lineage verification.
/// A feature that is not configured still reports `Passed` with zero counts after
/// its registry is checked; it is not omitted or marked out-of-scope.
pub const VERIFIED_REQUIRED_SURFACES: &[UnlearningSurface] = &[
    UnlearningSurface::CanonicalSqlite,
    UnlearningSurface::LinkedSqliteArtifacts,
    UnlearningSurface::FullTextIndex,
    UnlearningSurface::EmbeddingStore,
    UnlearningSurface::VectorIndex,
    UnlearningSurface::CognitiveRuntime,
    UnlearningSurface::TraceAndReceiptEvidence,
    UnlearningSurface::ReviewAndUndoQueues,
    UnlearningSurface::ConnectorAndRestoreGates,
    UnlearningSurface::ManagedBackups,
    UnlearningSurface::ManagedExports,
    UnlearningSurface::ManagedSyncObjects,
    UnlearningSurface::ErasureLedgerAnchor,
];

/// Legacy audit is deliberately narrower and can never imply complete lineage.
pub const LEGACY_AUDIT_REQUIRED_SURFACES: &[UnlearningSurface] = &[
    UnlearningSurface::CanonicalSqlite,
    UnlearningSurface::FullTextIndex,
    UnlearningSurface::EmbeddingStore,
    UnlearningSurface::VectorIndex,
    UnlearningSurface::CognitiveRuntime,
    UnlearningSurface::TraceAndReceiptEvidence,
    UnlearningSurface::ReviewAndUndoQueues,
];

/// Closed action code; arbitrary user text cannot enter the durable ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceAction {
    Delete,
    Redact,
    Rebuild,
    Reset,
    RevokeKey,
    InstallGate,
    Verify,
}

/// Closed execution detail vocabulary. Additions require a schema-versioned
/// code change; callers cannot place erased text into a free-form detail field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceDetailCode {
    TargetRowsDeleted,
    ArtifactRedacted,
    IndexRebuiltNoTargetHit,
    RuntimeResetAndRehydrated,
    MutationGateInstalled,
    ManagedArtifactResolved,
    LedgerAnchorVerified,
    VerifiedNoTargetHit,
    FeatureNotConfigured,
    LegacySurfaceAudited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Passed,
    Failed,
    NotChecked,
    OutOfScope,
}

/// One content-free step result suitable for an erasure ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceResult {
    pub surface: UnlearningSurface,
    pub action: SurfaceAction,
    pub matched_count: u64,
    pub changed_count: u64,
    pub status: CheckStatus,
    pub detail_code: SurfaceDetailCode,
}

/// Required postconditions for a verified-within-scope result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostconditionKind {
    ConcurrentMutationFenceHeld,
    LineageEpochStable,
    CanonicalTargetAbsent,
    LineageClosureResolved,
    LinkedSqliteArtifactsAbsent,
    FullTextTargetAbsent,
    EmbeddingTargetAbsent,
    VectorTargetAbsent,
    RuntimeResetAndRehydrated,
    TraceReceiptIdentifiersAbsent,
    PendingMutationPathsDisabled,
    AntiResurrectionGateInstalled,
    ManagedArtifactsResolved,
    LedgerAnchoredOutsideReplaceableDatabase,
}

/// Exact set that gates [`UnlearningVerdict::VerifiedWithinScope`].
pub const VERIFIED_REQUIRED_POSTCONDITIONS: &[PostconditionKind] = &[
    PostconditionKind::ConcurrentMutationFenceHeld,
    PostconditionKind::LineageEpochStable,
    PostconditionKind::CanonicalTargetAbsent,
    PostconditionKind::LineageClosureResolved,
    PostconditionKind::LinkedSqliteArtifactsAbsent,
    PostconditionKind::FullTextTargetAbsent,
    PostconditionKind::EmbeddingTargetAbsent,
    PostconditionKind::VectorTargetAbsent,
    PostconditionKind::RuntimeResetAndRehydrated,
    PostconditionKind::TraceReceiptIdentifiersAbsent,
    PostconditionKind::PendingMutationPathsDisabled,
    PostconditionKind::AntiResurrectionGateInstalled,
    PostconditionKind::ManagedArtifactsResolved,
    PostconditionKind::LedgerAnchoredOutsideReplaceableDatabase,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostconditionCheck {
    pub kind: PostconditionKind,
    pub status: CheckStatus,
    /// Content-free observation count. Zero commonly means absence.
    pub observed_count: u64,
    /// Closed verifier code, never a raw query result or target value.
    pub verifier_code: VerifierCode,
}

/// Closed verifier vocabulary. Detailed diagnostics remain in transient logs;
/// the durable erasure ledger records only these non-content-bearing codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierCode {
    VerifiedAbsent,
    ClosureResolved,
    MutationFenceHeld,
    LineageEpochStable,
    RuntimeResetAndRehydrated,
    MutationPathDisabled,
    GateInstalled,
    ManagedArtifactsResolved,
    LedgerAnchorVerified,
    CheckFailed,
    NotChecked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostconditionReport {
    pub verifier_version: u32,
    pub checks: Vec<PostconditionCheck>,
}

impl PostconditionReport {
    pub fn status_for(&self, kind: PostconditionKind) -> Option<CheckStatus> {
        self.checks
            .iter()
            .find(|check| check.kind == kind)
            .map(|check| check.status)
    }

    pub fn missing_verified_requirements(&self) -> Vec<PostconditionKind> {
        VERIFIED_REQUIRED_POSTCONDITIONS
            .iter()
            .copied()
            .filter(|kind| self.status_for(*kind) != Some(CheckStatus::Passed))
            .collect()
    }

    pub fn has_failure(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == CheckStatus::Failed)
    }

    pub fn commitment(&self, key: &CommitmentKey) -> Result<Commitment, UnlearningModelError> {
        validate_postcondition_report_shape(self)?;
        let mut canonical_report = self.clone();
        canonical_report.checks.sort_by_key(|check| check.kind);
        let canonical = serde_json::to_vec(&canonical_report)
            .map_err(|error| UnlearningModelError::Serialization(error.to_string()))?;
        Ok(commit_bytes(
            key,
            CommitmentKind::PostconditionReport,
            &canonical,
        ))
    }
}

fn is_erasure_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_CLOSED_CODE_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
}

fn validate_postcondition_report_shape(
    report: &PostconditionReport,
) -> Result<(), UnlearningModelError> {
    if report.verifier_version == 0 {
        return Err(UnlearningModelError::InvalidVerifierVersion);
    }

    let mut kinds = BTreeSet::new();
    for check in &report.checks {
        if !kinds.insert(check.kind) {
            return Err(UnlearningModelError::DuplicatePostcondition(check.kind));
        }
        let code_is_consistent = match check.status {
            CheckStatus::Passed => check.verifier_code == success_code_for(check.kind),
            CheckStatus::Failed => check.verifier_code == VerifierCode::CheckFailed,
            CheckStatus::NotChecked | CheckStatus::OutOfScope => {
                check.verifier_code == VerifierCode::NotChecked
            }
        };
        if !code_is_consistent {
            return Err(UnlearningModelError::InconsistentVerifierCode(check.kind));
        }
    }
    Ok(())
}

const fn success_code_for(kind: PostconditionKind) -> VerifierCode {
    match kind {
        PostconditionKind::ConcurrentMutationFenceHeld => VerifierCode::MutationFenceHeld,
        PostconditionKind::LineageEpochStable => VerifierCode::LineageEpochStable,
        PostconditionKind::LineageClosureResolved => VerifierCode::ClosureResolved,
        PostconditionKind::RuntimeResetAndRehydrated => VerifierCode::RuntimeResetAndRehydrated,
        PostconditionKind::PendingMutationPathsDisabled => VerifierCode::MutationPathDisabled,
        PostconditionKind::AntiResurrectionGateInstalled => VerifierCode::GateInstalled,
        PostconditionKind::ManagedArtifactsResolved => VerifierCode::ManagedArtifactsResolved,
        PostconditionKind::LedgerAnchoredOutsideReplaceableDatabase => {
            VerifierCode::LedgerAnchorVerified
        }
        PostconditionKind::CanonicalTargetAbsent
        | PostconditionKind::LinkedSqliteArtifactsAbsent
        | PostconditionKind::FullTextTargetAbsent
        | PostconditionKind::EmbeddingTargetAbsent
        | PostconditionKind::VectorTargetAbsent
        | PostconditionKind::TraceReceiptIdentifiersAbsent => VerifierCode::VerifiedAbsent,
    }
}

fn validate_surface_results_shape(surfaces: &[SurfaceResult]) -> Result<(), UnlearningModelError> {
    let mut kinds = BTreeSet::new();
    for result in surfaces {
        if !kinds.insert(result.surface) {
            return Err(UnlearningModelError::DuplicateSurface(result.surface));
        }
    }
    Ok(())
}

fn required_surfaces_passed(surfaces: &[SurfaceResult], required: &[UnlearningSurface]) -> bool {
    required.iter().all(|surface| {
        surfaces
            .iter()
            .find(|result| result.surface == *surface)
            .is_some_and(|result| result.status == CheckStatus::Passed)
    })
}

fn surface_failure(surfaces: &[SurfaceResult]) -> bool {
    surfaces
        .iter()
        .any(|result| result.status == CheckStatus::Failed)
}

/// Falsifiable outcome label. `VerifiedWithinScope` is unavailable to legacy data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnlearningVerdict {
    VerifiedWithinScope,
    AuditedPurgeOnly,
    Incomplete,
    Failed,
}

pub fn evaluate_unlearning_verdict(
    scope: UnlearningScope,
    report: &PostconditionReport,
    surfaces: &[SurfaceResult],
) -> UnlearningVerdict {
    if validate_postcondition_report_shape(report).is_err()
        || validate_surface_results_shape(surfaces).is_err()
        || report.has_failure()
        || surface_failure(surfaces)
    {
        return UnlearningVerdict::Failed;
    }
    if scope == UnlearningScope::LegacyAuditedPurge {
        return if required_surfaces_passed(surfaces, LEGACY_AUDIT_REQUIRED_SURFACES) {
            UnlearningVerdict::AuditedPurgeOnly
        } else {
            UnlearningVerdict::Incomplete
        };
    }
    if report.missing_verified_requirements().is_empty()
        && required_surfaces_passed(surfaces, VERIFIED_REQUIRED_SURFACES)
    {
        UnlearningVerdict::VerifiedWithinScope
    } else {
        UnlearningVerdict::Incomplete
    }
}

/// Durable, content-free result payload. A signer may wrap this in
/// [`SignedErasureReceipt`] after verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErasureLedgerRecord {
    schema: String,
    erasure_id: String,
    generation: u64,
    commitment_key_id: String,
    lineage_epoch: u64,
    scope: UnlearningScope,
    target: AntiResurrectionCommitments,
    closure_commitment: Commitment,
    fence_commitment: Commitment,
    postcondition_commitment: Commitment,
    surfaces: Vec<SurfaceResult>,
    postconditions: PostconditionReport,
    verdict: UnlearningVerdict,
    exclusions: Vec<GuaranteeExclusion>,
    committed_at: DateTime<Utc>,
}

impl ErasureLedgerRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        key: &CommitmentKey,
        erasure_id: impl Into<String>,
        generation: u64,
        scope: UnlearningScope,
        target: AntiResurrectionCommitments,
        closure: &LineageClosure,
        fence: &VerificationFence,
        mut postconditions: PostconditionReport,
        mut surfaces: Vec<SurfaceResult>,
        committed_at: DateTime<Utc>,
    ) -> Result<Self, UnlearningModelError> {
        let erasure_id = erasure_id.into();
        if !is_erasure_id(&erasure_id) {
            return Err(UnlearningModelError::InvalidClosedCode("erasure_id"));
        }
        if generation == 0 || generation > MAX_ERASURE_GENERATION {
            return Err(UnlearningModelError::InvalidGeneration(generation));
        }
        validate_target_commitments(key, &target)?;
        validate_postcondition_report_shape(&postconditions)?;
        validate_surface_results_shape(&surfaces)?;
        if !fence
            .commitment()
            .is_well_formed_for(key, CommitmentKind::ExecutionFence)
        {
            return Err(UnlearningModelError::InvalidCommitment(
                CommitmentKind::ExecutionFence,
            ));
        }

        reject_ledger_string_leak(key, &target, &erasure_id, "erasure_id")?;

        let closure_commitment = commit_lineage_closure(key, closure)?;
        let postcondition_commitment = postconditions.commitment(key)?;
        let verdict = evaluate_unlearning_verdict(scope, &postconditions, &surfaces);
        postconditions.checks.sort_by_key(|check| check.kind);
        surfaces.sort_by_key(|surface| surface.surface);
        Ok(Self {
            schema: VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1.to_string(),
            erasure_id,
            generation,
            commitment_key_id: key.key_id().to_string(),
            lineage_epoch: fence.lineage_epoch(),
            scope,
            target,
            closure_commitment,
            fence_commitment: fence.commitment().clone(),
            postcondition_commitment,
            surfaces,
            postconditions,
            verdict,
            exclusions: scope.exclusions(),
            committed_at,
        })
    }

    pub const fn verdict(&self) -> UnlearningVerdict {
        self.verdict
    }

    pub const fn scope(&self) -> UnlearningScope {
        self.scope
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn lineage_epoch(&self) -> u64 {
        self.lineage_epoch
    }

    pub fn exclusions(&self) -> &[GuaranteeExclusion] {
        &self.exclusions
    }

    pub fn commitment_key_id(&self) -> &str {
        &self.commitment_key_id
    }

    /// Versioned byte-exact JSON payload for an integration-supplied signer.
    /// All variable-order collections are sorted by the checked constructor.
    /// Cross-language verifiers must implement this schema's exact encoding or
    /// introduce a later schema version backed by JCS/deterministic CBOR.
    pub fn signing_payload(&self) -> Result<Vec<u8>, UnlearningModelError> {
        serde_json::to_vec(self)
            .map_err(|error| UnlearningModelError::Serialization(error.to_string()))
    }
}

fn validate_target_commitments(
    key: &CommitmentKey,
    target: &AntiResurrectionCommitments,
) -> Result<(), UnlearningModelError> {
    for (commitment, kind) in [
        (&target.target_identifier, CommitmentKind::TargetIdentifier),
        (&target.exact_content, CommitmentKind::ExactContent),
    ] {
        if !commitment.is_well_formed_for(key, kind) {
            return Err(UnlearningModelError::InvalidCommitment(kind));
        }
    }
    if let Some(source) = &target.source_locator
        && !source.is_well_formed_for(key, CommitmentKind::SourceLocator)
    {
        return Err(UnlearningModelError::InvalidCommitment(
            CommitmentKind::SourceLocator,
        ));
    }
    Ok(())
}

fn reject_ledger_string_leak(
    key: &CommitmentKey,
    target: &AntiResurrectionCommitments,
    value: &str,
    field: &'static str,
) -> Result<(), UnlearningModelError> {
    if target.target_identifier.matches(key, value.as_bytes())
        || target.exact_content.matches(key, value.as_bytes())
    {
        return Err(UnlearningModelError::LedgerFieldMatchesTarget(field));
    }
    Ok(())
}

/// Signature metadata is a transport envelope, not independent proof of
/// deletion. Verification establishes that one key attested to this immutable
/// payload; it does not prove absence outside the receipt's declared scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureEnvelope {
    pub algorithm: String,
    pub key_id: String,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SignedErasureReceipt {
    record: ErasureLedgerRecord,
    signature: SignatureEnvelope,
}

impl SignedErasureReceipt {
    /// Attach signer output. Cryptographic verification belongs to the receipt
    /// integration; this envelope never expands the local erasure scope.
    pub fn attach(record: ErasureLedgerRecord, signature: SignatureEnvelope) -> Self {
        Self { record, signature }
    }

    pub fn record(&self) -> &ErasureLedgerRecord {
        &self.record
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnlearningModelError {
    EmptyArtifactId,
    CommitmentSecretTooShort { minimum: usize, actual: usize },
    FenceTokenTooShort { minimum: usize, actual: usize },
    InvalidGeneration(u64),
    InvalidLineageClosure,
    InvalidCommitment(CommitmentKind),
    DuplicatePostcondition(PostconditionKind),
    DuplicateSurface(UnlearningSurface),
    InconsistentVerifierCode(PostconditionKind),
    InvalidVerifierVersion,
    InvalidClosedCode(&'static str),
    LedgerFieldMatchesTarget(&'static str),
    Serialization(String),
}

impl fmt::Display for UnlearningModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArtifactId => formatter.write_str("artifact id cannot be empty"),
            Self::CommitmentSecretTooShort { minimum, actual } => write!(
                formatter,
                "commitment secret must be at least {minimum} bytes (received {actual})"
            ),
            Self::FenceTokenTooShort { minimum, actual } => write!(
                formatter,
                "opaque fence token must be at least {minimum} bytes (received {actual})"
            ),
            Self::InvalidGeneration(generation) => {
                write!(formatter, "invalid SQLite erasure generation: {generation}")
            }
            Self::InvalidLineageClosure => formatter.write_str("invalid lineage closure"),
            Self::InvalidCommitment(kind) => {
                write!(formatter, "invalid {kind:?} commitment")
            }
            Self::DuplicatePostcondition(kind) => {
                write!(formatter, "duplicate {kind:?} postcondition")
            }
            Self::DuplicateSurface(surface) => {
                write!(formatter, "duplicate {surface:?} surface result")
            }
            Self::InconsistentVerifierCode(kind) => {
                write!(
                    formatter,
                    "verifier code is inconsistent with {kind:?} status"
                )
            }
            Self::InvalidVerifierVersion => formatter.write_str("verifier version must be nonzero"),
            Self::InvalidClosedCode(field) => {
                write!(formatter, "invalid content-free code in {field}")
            }
            Self::LedgerFieldMatchesTarget(field) => {
                write!(
                    formatter,
                    "ledger field {field} matches erased target material"
                )
            }
            Self::Serialization(error) => write!(formatter, "serialization failed: {error}"),
        }
    }
}

impl std::error::Error for UnlearningModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(kind: ArtifactKind, id: &str) -> ArtifactRef {
        ArtifactRef::new(kind, id).unwrap()
    }

    fn key() -> CommitmentKey {
        CommitmentKey::derive(b"0123456789abcdef0123456789abcdef").unwrap()
    }

    fn fence(key: &CommitmentKey) -> VerificationFence {
        VerificationFence::from_held_epoch(key, 42, b"opaque-fence-token").unwrap()
    }

    fn passing_surfaces() -> Vec<SurfaceResult> {
        VERIFIED_REQUIRED_SURFACES
            .iter()
            .copied()
            .map(|surface| SurfaceResult {
                surface,
                action: SurfaceAction::Verify,
                matched_count: 0,
                changed_count: 0,
                status: CheckStatus::Passed,
                detail_code: SurfaceDetailCode::VerifiedNoTargetHit,
            })
            .collect()
    }

    fn passing_report() -> PostconditionReport {
        PostconditionReport {
            verifier_version: 1,
            checks: VERIFIED_REQUIRED_POSTCONDITIONS
                .iter()
                .copied()
                .map(|kind| PostconditionCheck {
                    kind,
                    status: CheckStatus::Passed,
                    observed_count: 0,
                    verifier_code: success_code_for(kind),
                })
                .collect(),
        }
    }

    #[test]
    fn closure_is_transitive_deterministic_and_cycle_safe() {
        let root = artifact(ArtifactKind::KnowledgeNode, "memory-a");
        let embedding = artifact(ArtifactKind::Embedding, "embedding-a");
        let insight = artifact(ArtifactKind::Insight, "insight-a");
        let receipt = artifact(ArtifactKind::Receipt, "receipt-a");
        let unrelated = artifact(ArtifactKind::KnowledgeNode, "memory-b");
        let edges = vec![
            LineageEdge {
                source: insight.clone(),
                derived: receipt.clone(),
                relation: LineageRelation::DerivedFrom,
            },
            LineageEdge {
                source: root.clone(),
                derived: embedding.clone(),
                relation: LineageRelation::Embeds,
            },
            LineageEdge {
                source: embedding.clone(),
                derived: insight.clone(),
                relation: LineageRelation::DerivedFrom,
            },
            // Cycle must terminate without duplicating a member.
            LineageEdge {
                source: receipt.clone(),
                derived: embedding.clone(),
                relation: LineageRelation::Mentions,
            },
            LineageEdge {
                source: unrelated.clone(),
                derived: artifact(ArtifactKind::Insight, "other-insight"),
                relation: LineageRelation::DerivedFrom,
            },
        ];

        let forward = compute_lineage_closure(&root, &edges);
        let mut reversed_edges = edges.clone();
        reversed_edges.reverse();
        let reversed = compute_lineage_closure(&root, &reversed_edges);

        assert_eq!(forward, reversed);
        assert_eq!(forward.len(), 4);
        assert!(forward.contains(&root));
        assert!(forward.contains(&receipt));
        assert!(!forward.contains(&unrelated));
        assert_eq!(forward.members[0].distance, 0);
        assert_eq!(forward.members[3].distance, 3);
    }

    #[test]
    fn commitment_domains_are_separated_and_keyed() {
        let key = key();
        let other_key = CommitmentKey::derive(b"abcdef0123456789abcdef0123456789").unwrap();
        let raw = b"89b5851f-7ca1-4984-b751-11c108f422fe";
        let id_commitment = commit_bytes(&key, CommitmentKind::TargetIdentifier, raw);
        let content_commitment = commit_bytes(&key, CommitmentKind::ExactContent, raw);

        assert_ne!(id_commitment.digest, content_commitment.digest);
        assert_eq!(id_commitment.key_id, key.key_id());
        assert_ne!(id_commitment.key_id, other_key.key_id());
        assert!(id_commitment.matches(&key, raw));
        assert!(!id_commitment.matches(&other_key, raw));
        assert!(!id_commitment.digest.contains("89b5851f"));
        assert!(CommitmentKey::derive(b"short").is_err());
        assert_eq!(format!("{key:?}"), "CommitmentKey(<redacted>)");
    }

    #[test]
    fn anti_resurrection_commitments_match_exact_inputs_only() {
        let key = key();
        let commitments = anti_resurrection_commitments(
            &key,
            "memory-id",
            b"exact content bytes",
            Some(" github|sam/repo|issue|7 "),
        );

        assert!(commitments.target_identifier.matches(&key, b"memory-id"));
        assert!(
            commitments
                .exact_content
                .matches(&key, b"exact content bytes")
        );
        assert!(
            !commitments
                .exact_content
                .matches(&key, b"Exact content bytes")
        );
        assert!(
            commitments
                .source_locator
                .unwrap()
                .matches(&key, b"github|sam/repo|issue|7")
        );
    }

    #[test]
    fn legacy_scope_can_never_be_verified() {
        let report = passing_report();
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::LegacyAuditedPurge,
                &report,
                &passing_surfaces(),
            ),
            UnlearningVerdict::AuditedPurgeOnly
        );
        assert!(
            UnlearningScope::LegacyAuditedPurge
                .exclusions()
                .contains(&GuaranteeExclusion::UntrackedLegacyDerivedInfluence)
        );
    }

    #[test]
    fn verified_scope_requires_every_positive_postcondition() {
        let mut report = passing_report();
        let surfaces = passing_surfaces();
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &surfaces,
            ),
            UnlearningVerdict::VerifiedWithinScope
        );

        report.checks.pop();
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &surfaces,
            ),
            UnlearningVerdict::Incomplete
        );

        report.checks.push(PostconditionCheck {
            kind: PostconditionKind::LedgerAnchoredOutsideReplaceableDatabase,
            status: CheckStatus::Failed,
            observed_count: 1,
            verifier_code: VerifierCode::CheckFailed,
        });
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &surfaces,
            ),
            UnlearningVerdict::Failed
        );
    }

    #[test]
    fn ledger_payload_contains_commitments_and_explicit_boundaries_not_target() {
        let key = key();
        let raw_id = "raw-memory-id-must-not-survive";
        let raw_content = "raw content must not survive";
        let raw_source = "github|sam/repo|issue|raw-source-must-not-survive";
        let target =
            anti_resurrection_commitments(&key, raw_id, raw_content.as_bytes(), Some(raw_source));
        let closure = compute_lineage_closure(&artifact(ArtifactKind::KnowledgeNode, raw_id), &[]);
        let report = passing_report();
        let fence = fence(&key);
        let ledger = ErasureLedgerRecord::build(
            &key,
            "erase-1",
            9,
            UnlearningScope::PostLineageVerifiedLocal,
            target,
            &closure,
            &fence,
            report,
            passing_surfaces(),
            DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
        )
        .unwrap();

        assert_eq!(ledger.verdict(), UnlearningVerdict::VerifiedWithinScope);
        assert_eq!(ledger.lineage_epoch(), 42);
        assert_eq!(ledger.commitment_key_id(), key.key_id());
        let serialized = String::from_utf8(ledger.signing_payload().unwrap()).unwrap();
        assert!(!serialized.contains(raw_id));
        assert!(!serialized.contains(raw_content));
        assert!(!serialized.contains(raw_source));
        assert!(!serialized.contains("opaque-fence-token"));
        assert!(serialized.contains("unmanaged_copies"));
        assert!(serialized.contains("external_model_weights"));
        assert!(serialized.contains(VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1));
    }

    #[test]
    fn closure_and_postcondition_commitments_are_stable_and_distinct() {
        let key = key();
        let root = artifact(ArtifactKind::KnowledgeNode, "root");
        let child = artifact(ArtifactKind::Insight, "child");
        let edge = LineageEdge {
            source: root.clone(),
            derived: child,
            relation: LineageRelation::DerivedFrom,
        };
        let a = compute_lineage_closure(&root, std::slice::from_ref(&edge));
        let b = compute_lineage_closure(&root, &[edge.clone(), edge]);
        let closure_a = commit_lineage_closure(&key, &a).unwrap();
        let closure_b = commit_lineage_closure(&key, &b).unwrap();
        let report = passing_report().commitment(&key).unwrap();

        assert_eq!(closure_a, closure_b);
        assert_ne!(closure_a.digest, report.digest);
    }

    #[test]
    fn verification_requires_real_surface_evidence() {
        let report = passing_report();
        assert_eq!(
            evaluate_unlearning_verdict(UnlearningScope::PostLineageVerifiedLocal, &report, &[],),
            UnlearningVerdict::Incomplete
        );

        let mut surfaces = passing_surfaces();
        surfaces.pop();
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &surfaces,
            ),
            UnlearningVerdict::Incomplete
        );

        surfaces.push(surfaces[0].clone());
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &surfaces,
            ),
            UnlearningVerdict::Failed
        );
    }

    #[test]
    fn duplicate_postconditions_are_failed_not_first_match_wins() {
        let mut report = passing_report();
        report.checks.push(PostconditionCheck {
            kind: PostconditionKind::CanonicalTargetAbsent,
            status: CheckStatus::NotChecked,
            observed_count: 1,
            verifier_code: VerifierCode::NotChecked,
        });
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &passing_surfaces(),
            ),
            UnlearningVerdict::Failed
        );
        assert!(matches!(
            report.commitment(&key()),
            Err(UnlearningModelError::DuplicatePostcondition(
                PostconditionKind::CanonicalTargetAbsent
            ))
        ));
    }

    #[test]
    fn passed_status_requires_the_kind_specific_closed_code() {
        let mut report = passing_report();
        report.checks[0].verifier_code = VerifierCode::CheckFailed;
        assert_eq!(
            evaluate_unlearning_verdict(
                UnlearningScope::PostLineageVerifiedLocal,
                &report,
                &passing_surfaces(),
            ),
            UnlearningVerdict::Failed
        );
        assert!(matches!(
            report.commitment(&key()),
            Err(UnlearningModelError::InconsistentVerifierCode(_))
        ));
    }

    #[test]
    fn builder_binds_commitment_kinds_key_and_fence_epoch() {
        let key = key();
        let other_key = CommitmentKey::derive(b"abcdef0123456789abcdef0123456789").unwrap();
        let closure =
            compute_lineage_closure(&artifact(ArtifactKind::KnowledgeNode, "target-id"), &[]);
        let fence = fence(&key);
        let wrong_target =
            anti_resurrection_commitments(&other_key, "target-id", b"target-content", None);
        let result = ErasureLedgerRecord::build(
            &key,
            "erase-2",
            2,
            UnlearningScope::PostLineageVerifiedLocal,
            wrong_target,
            &closure,
            &fence,
            passing_report(),
            passing_surfaces(),
            Utc::now(),
        );
        assert!(matches!(
            result,
            Err(UnlearningModelError::InvalidCommitment(
                CommitmentKind::TargetIdentifier
            ))
        ));
        assert!(VerificationFence::from_held_epoch(&key, 1, b"short").is_err());
    }

    #[test]
    fn ledger_rejects_free_form_or_target_bearing_ids() {
        let key = key();
        let closure =
            compute_lineage_closure(&artifact(ArtifactKind::KnowledgeNode, "target-id"), &[]);
        let fence = fence(&key);

        let result = ErasureLedgerRecord::build(
            &key,
            "free form id",
            3,
            UnlearningScope::PostLineageVerifiedLocal,
            anti_resurrection_commitments(
                &key,
                "target-id",
                b"raw target content with spaces",
                None,
            ),
            &closure,
            &fence,
            passing_report(),
            passing_surfaces(),
            Utc::now(),
        );
        assert!(matches!(
            result,
            Err(UnlearningModelError::InvalidClosedCode("erasure_id"))
        ));

        let result = ErasureLedgerRecord::build(
            &key,
            "target-id",
            4,
            UnlearningScope::PostLineageVerifiedLocal,
            anti_resurrection_commitments(&key, "target-id", b"content", None),
            &closure,
            &fence,
            passing_report(),
            passing_surfaces(),
            Utc::now(),
        );
        assert!(matches!(
            result,
            Err(UnlearningModelError::LedgerFieldMatchesTarget("erasure_id"))
        ));
    }

    #[test]
    fn generation_is_nonzero_and_sqlite_compatible() {
        let key = key();
        let closure =
            compute_lineage_closure(&artifact(ArtifactKind::KnowledgeNode, "target-id"), &[]);
        let fence = fence(&key);
        for generation in [0, MAX_ERASURE_GENERATION + 1] {
            let result = ErasureLedgerRecord::build(
                &key,
                "erase-4",
                generation,
                UnlearningScope::PostLineageVerifiedLocal,
                anti_resurrection_commitments(&key, "target-id", b"content", None),
                &closure,
                &fence,
                passing_report(),
                passing_surfaces(),
                Utc::now(),
            );
            assert!(matches!(
                result,
                Err(UnlearningModelError::InvalidGeneration(value)) if value == generation
            ));
        }
    }

    #[test]
    fn empty_identifiers_are_rejected() {
        assert!(ArtifactRef::new(ArtifactKind::KnowledgeNode, "  ").is_err());
        let key = key();
        let report = passing_report();
        let closure = compute_lineage_closure(&artifact(ArtifactKind::KnowledgeNode, "id"), &[]);
        let fence = fence(&key);
        let result = ErasureLedgerRecord::build(
            &key,
            "",
            1,
            UnlearningScope::LegacyAuditedPurge,
            anti_resurrection_commitments(&key, "id", b"content", None),
            &closure,
            &fence,
            report,
            passing_surfaces(),
            Utc::now(),
        );
        assert!(matches!(
            result,
            Err(UnlearningModelError::InvalidClosedCode("erasure_id"))
        ));
    }
}
