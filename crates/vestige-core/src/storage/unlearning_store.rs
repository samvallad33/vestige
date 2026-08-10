//! SQLite execution helpers for Verified Local Unlearning.
//!
//! This is deliberately an *inner* storage layer.  It owns no runtime state,
//! does not decide which rows to delete, and cannot make an unlearning claim by
//! itself.  The integration layer must hold the write/import/restore fence,
//! perform the actual deletion and runtime rebuild, then use these helpers to
//! persist the content-free evidence and verify the remaining local surfaces.
//!
//! The V25 migration is intentionally wired elsewhere.  Keeping the SQL
//! contract here lets the migration, purge coordinator, and MCP surface agree
//! on one fail-closed representation without making this module another
//! mutable-memory implementation.

use std::fmt;

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::Deserialize;

use super::unlearning::{
    AntiResurrectionCommitments, ArtifactKind, ArtifactRef, CheckStatus, Commitment,
    CommitmentKind, ErasureLedgerRecord, LineageEdge, LineageRelation, SurfaceAction,
    SurfaceDetailCode, SurfaceResult, UnlearningScope, UnlearningVerdict,
    VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1, compute_lineage_closure,
};

/// The migration version reserved for the local unlearning execution schema.
pub const V25_UNLEARNING_STORAGE_SCHEMA_VERSION: u32 = 25;

/// V25 contract consumed by this module.
///
/// The columns contain only closed codes, counters, timestamps, random erasure
/// identifiers, or keyed commitments.  In particular, neither a target ID nor
/// target plaintext may be written to any `erasure_*` table.  `artifact_lineage`
/// is the one raw-identifier table; the coordinator must remove its relevant
/// rows in the same fenced transaction as the derived artifacts.
///
/// This supersedes the un-wired V22 draft in `unlearning.rs`.  The migration
/// must preserve these names and constraints exactly, or this module must be
/// versioned alongside it.
pub const V25_UNLEARNING_STORAGE_SCHEMA_EXPECTATION: &str = r#"
CREATE TABLE IF NOT EXISTS artifact_lineage (
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    derived_kind TEXT NOT NULL,
    derived_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN (
        'derived_from', 'embeds', 'indexes', 'summarizes', 'aggregates',
        'mentions', 'mutates', 'mirrors'
    )),
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_kind, source_id, derived_kind, derived_id, relation)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_artifact_lineage_source
    ON artifact_lineage(source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_artifact_lineage_derived
    ON artifact_lineage(derived_kind, derived_id);

CREATE TABLE IF NOT EXISTS erasure_generation_counter (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_generation INTEGER NOT NULL CHECK (last_generation >= 0)
) STRICT;
INSERT OR IGNORE INTO erasure_generation_counter(singleton, last_generation) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS erasure_jobs (
    erasure_id TEXT PRIMARY KEY,
    schema_uri TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN (
        'legacy_audited_purge', 'post_lineage_verified_local'
    )),
    status TEXT NOT NULL CHECK (status IN ('running', 'committed', 'failed')),
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
    signature_json TEXT,
    failure_code TEXT CHECK (failure_code IS NULL OR failure_code IN (
        'verification_failed', 'lineage_incomplete', 'tombstone_conflict',
        'durability_failure', 'runtime_rebuild_failed', 'operator_cancelled'
    ))
) STRICT;

CREATE TABLE IF NOT EXISTS erasure_steps (
    erasure_id TEXT NOT NULL,
    step_order INTEGER NOT NULL CHECK (step_order >= 0),
    surface TEXT NOT NULL CHECK (surface IN (
        'canonical_sqlite', 'linked_sqlite_artifacts', 'full_text_index',
        'embedding_store', 'vector_index', 'cognitive_runtime',
        'trace_and_receipt_evidence', 'review_and_undo_queues',
        'connector_and_restore_gates', 'managed_backups', 'managed_exports',
        'managed_sync_objects', 'erasure_ledger_anchor'
    )),
    action TEXT NOT NULL CHECK (action IN (
        'delete', 'redact', 'rebuild', 'reset', 'revoke_key', 'install_gate', 'verify'
    )),
    matched_count INTEGER NOT NULL CHECK (matched_count >= 0),
    changed_count INTEGER NOT NULL CHECK (changed_count >= 0),
    verification_status TEXT NOT NULL CHECK (verification_status IN (
        'passed', 'failed', 'not_checked', 'out_of_scope'
    )),
    detail_code TEXT NOT NULL CHECK (detail_code IN (
        'target_rows_deleted', 'artifact_redacted', 'index_rebuilt_no_target_hit',
        'runtime_reset_and_rehydrated', 'mutation_gate_installed',
        'managed_artifact_resolved', 'ledger_anchor_verified',
        'verified_no_target_hit', 'feature_not_configured', 'legacy_surface_audited'
    )),
    PRIMARY KEY (erasure_id, step_order),
    FOREIGN KEY (erasure_id) REFERENCES erasure_jobs(erasure_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS erasure_tombstones (
    target_commitment TEXT PRIMARY KEY,
    exact_content_commitment TEXT NOT NULL,
    source_locator_commitment TEXT,
    commitment_key_id TEXT NOT NULL,
    generation INTEGER NOT NULL UNIQUE CHECK (generation > 0),
    scope TEXT NOT NULL CHECK (scope IN (
        'legacy_audited_purge', 'post_lineage_verified_local'
    )),
    receipt_id TEXT NOT NULL,
    erased_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_erasure_tombstones_content
    ON erasure_tombstones(exact_content_commitment, commitment_key_id);
CREATE INDEX IF NOT EXISTS idx_erasure_tombstones_source
    ON erasure_tombstones(source_locator_commitment, commitment_key_id)
    WHERE source_locator_commitment IS NOT NULL;

CREATE TABLE IF NOT EXISTS managed_artifacts (
    artifact_commitment TEXT PRIMARY KEY,
    artifact_kind TEXT NOT NULL,
    locator_commitment TEXT NOT NULL,
    commitment_key_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'erased', 'unavailable')),
    created_at TEXT NOT NULL,
    last_verified_at TEXT
) STRICT;
"#;

/// Runtime/configuration error from the V25 execution helpers.
#[derive(Debug, thiserror::Error)]
pub enum UnlearningStoreError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("V25 unlearning schema returned an unknown {kind} code `{value}`")]
    UnknownClosedCode { kind: &'static str, value: String },
    #[error("V25 unlearning schema is missing required table `{0}")]
    MissingRequiredTable(&'static str),
    #[error("invalid content-free erasure identifier")]
    InvalidErasureId,
    #[error("SQLite INTEGER overflow in `{field}`")]
    IntegerOverflow { field: &'static str },
    #[error("commitment has an invalid shape for {0:?}")]
    InvalidCommitment(CommitmentKind),
    #[error("commitments in one operation do not use the same commitment key")]
    MixedCommitmentKeys,
    #[error("tombstone conflict for an existing target commitment")]
    TombstoneConflict,
    #[error("job `{0}` is not in the running state")]
    JobNotRunning(String),
    #[error("audit result is not eligible for a verified or legacy audit record: {0:?}")]
    IneligibleAuditVerdict(UnlearningVerdict),
    #[error("persisted audit payload is malformed: {0}")]
    MalformedAuditPayload(String),
}

pub type UnlearningStoreResult<T> = std::result::Result<T, UnlearningStoreError>;

/// Closed lifecycle state for an execution job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErasureJobStatus {
    Running,
    Committed,
    Failed,
}

impl ErasureJobStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Committed => "committed",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> UnlearningStoreResult<Self> {
        match value {
            "running" => Ok(Self::Running),
            "committed" => Ok(Self::Committed),
            "failed" => Ok(Self::Failed),
            _ => Err(unknown_code("erasure job status", value)),
        }
    }
}

/// Closed non-content-bearing reason persisted when a job cannot finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErasureFailureCode {
    VerificationFailed,
    LineageIncomplete,
    TombstoneConflict,
    DurabilityFailure,
    RuntimeRebuildFailed,
    OperatorCancelled,
}

impl ErasureFailureCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::VerificationFailed => "verification_failed",
            Self::LineageIncomplete => "lineage_incomplete",
            Self::TombstoneConflict => "tombstone_conflict",
            Self::DurabilityFailure => "durability_failure",
            Self::RuntimeRebuildFailed => "runtime_rebuild_failed",
            Self::OperatorCancelled => "operator_cancelled",
        }
    }
}

/// Immutable inputs for the `running` job row.  They are all commitments or
/// closed metadata; raw target values stay in the fenced executor only.
pub struct ErasureJobStart<'a> {
    pub erasure_id: &'a str,
    pub scope: UnlearningScope,
    pub lineage_epoch: u64,
    pub fence_commitment: &'a Commitment,
    pub target: &'a AntiResurrectionCommitments,
    pub closure_commitment: &'a Commitment,
    pub started_at: DateTime<Utc>,
}

/// Durable, content-free job metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredErasureJob {
    pub erasure_id: String,
    pub scope: UnlearningScope,
    pub status: ErasureJobStatus,
    pub generation: u64,
    pub commitment_key_id: String,
    pub lineage_epoch: u64,
    pub started_at: DateTime<Utc>,
    pub committed_at: Option<DateTime<Utc>>,
    pub failure_code: Option<ErasureFailureCode>,
}

/// Persisted status from an anti-resurrection lookup.  A content or source
/// match must block re-ingest even if the caller assigned a new target ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AntiResurrectionGateStatus {
    pub target_identifier_blocked: bool,
    pub exact_content_blocked: bool,
    pub source_locator_blocked: bool,
}

impl AntiResurrectionGateStatus {
    pub const fn blocks_any(self) -> bool {
        self.target_identifier_blocked || self.exact_content_blocked || self.source_locator_blocked
    }
}

/// Result of an idempotent tombstone write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TombstoneWriteOutcome {
    Inserted,
    AlreadyPresent,
}

/// Exact random canaries used only during a post-delete scan.  They are never
/// stored in the erasure ledger or returned in scan results.
pub struct ExactCanary<'a> {
    pub identifier: &'a str,
    pub content: &'a str,
}

impl ExactCanary<'_> {
    fn validate(&self) -> UnlearningStoreResult<()> {
        if self.identifier.is_empty() || self.content.is_empty() {
            return Err(UnlearningStoreError::MalformedAuditPayload(
                "exact canaries must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

/// Closed list of local persistence tables that can contain raw target IDs or
/// content as of V25.  `erasure_*` and `managed_artifacts` are intentionally
/// absent: they must contain commitments only and are checked structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalCanaryTable {
    KnowledgeNodes,
    KnowledgeFts,
    NodeEmbeddings,
    MemoryConnections,
    MemoryStates,
    FsrsCards,
    Intentions,
    Insights,
    MergePlans,
    MergeOperations,
    CompositionEvents,
    CompositionMembers,
    CompositionOutcomes,
    AgentTraces,
    MemoryReceipts,
    MemoryPrs,
    SynapticTags,
    SynapticEvents,
    SynapticCaptureItems,
    ReplayCapsules,
    ReplayItems,
    CounterfactualReplays,
    ArtifactLineage,
    SyncTombstones,
    DeletionTombstones,
}

/// Required full local scan for a verified-within-scope result.
pub const V25_REQUIRED_LOCAL_CANARY_TABLES: &[LocalCanaryTable] = &[
    LocalCanaryTable::KnowledgeNodes,
    LocalCanaryTable::KnowledgeFts,
    LocalCanaryTable::NodeEmbeddings,
    LocalCanaryTable::MemoryConnections,
    LocalCanaryTable::MemoryStates,
    LocalCanaryTable::FsrsCards,
    LocalCanaryTable::Intentions,
    LocalCanaryTable::Insights,
    LocalCanaryTable::MergePlans,
    LocalCanaryTable::MergeOperations,
    LocalCanaryTable::CompositionEvents,
    LocalCanaryTable::CompositionMembers,
    LocalCanaryTable::CompositionOutcomes,
    LocalCanaryTable::AgentTraces,
    LocalCanaryTable::MemoryReceipts,
    LocalCanaryTable::MemoryPrs,
    LocalCanaryTable::SynapticTags,
    LocalCanaryTable::SynapticEvents,
    LocalCanaryTable::SynapticCaptureItems,
    LocalCanaryTable::ReplayCapsules,
    LocalCanaryTable::ReplayItems,
    LocalCanaryTable::CounterfactualReplays,
    LocalCanaryTable::ArtifactLineage,
    LocalCanaryTable::SyncTombstones,
    LocalCanaryTable::DeletionTombstones,
];

/// Content-free finding count for one required local table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanaryScanResult {
    pub table: LocalCanaryTable,
    pub matching_rows: u64,
}

impl CanaryScanResult {
    pub const fn is_clear(&self) -> bool {
        self.matching_rows == 0
    }
}

/// An eligible completed record plus the stable JSON bytes that can be signed
/// by the DSSE integration.  The bytes contain commitments, not erased input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibleAuditRecord {
    erasure_id: String,
    generation: u64,
    scope: UnlearningScope,
    verdict: UnlearningVerdict,
    signing_payload: Vec<u8>,
}

impl EligibleAuditRecord {
    pub fn erasure_id(&self) -> &str {
        &self.erasure_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn scope(&self) -> UnlearningScope {
        self.scope
    }

    pub const fn verdict(&self) -> UnlearningVerdict {
        self.verdict
    }

    pub fn signing_payload(&self) -> &[u8] {
        &self.signing_payload
    }
}

/// Read-only SQL helpers used under the coordinator's fenced transaction.
pub struct UnlearningStore;

impl UnlearningStore {
    /// Load a deterministic, cycle-safe transitive closure from the V25 lineage
    /// table.  Unknown kind/relation codes are a schema-integrity failure.
    pub fn load_lineage_closure(
        tx: &Transaction<'_>,
        root: &ArtifactRef,
    ) -> UnlearningStoreResult<super::unlearning::LineageClosure> {
        require_table(tx, "artifact_lineage")?;
        let mut statement = tx.prepare(
            r#"WITH RECURSIVE walk(kind, id, path) AS (
                 SELECT ?1, ?2, ',' || hex(?1) || ':' || hex(?2) || ','
                 UNION ALL
                 SELECT edge.derived_kind, edge.derived_id,
                        walk.path || hex(edge.derived_kind) || ':' || hex(edge.derived_id) || ','
                   FROM artifact_lineage AS edge
                   JOIN walk ON edge.source_kind = walk.kind AND edge.source_id = walk.id
                  WHERE instr(
                      walk.path,
                      ',' || hex(edge.derived_kind) || ':' || hex(edge.derived_id) || ','
                  ) = 0
             )
             SELECT DISTINCT edge.source_kind, edge.source_id, edge.derived_kind,
                             edge.derived_id, edge.relation
               FROM artifact_lineage AS edge
               JOIN walk ON edge.source_kind = walk.kind AND edge.source_id = walk.id
             ORDER BY edge.source_kind, edge.source_id, edge.derived_kind, edge.derived_id, edge.relation"#,
        )?;
        let rows = statement.query_map(params![root.kind.as_str(), &root.id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;

        let mut edges = Vec::new();
        for row in rows {
            let (source_kind, source_id, derived_kind, derived_id, relation) = row?;
            edges.push(LineageEdge {
                source: ArtifactRef::new(parse_artifact_kind(&source_kind)?, source_id)
                    .map_err(|error| malformed_model(error))?,
                derived: ArtifactRef::new(parse_artifact_kind(&derived_kind)?, derived_id)
                    .map_err(|error| malformed_model(error))?,
                relation: parse_lineage_relation(&relation)?,
            });
        }
        Ok(compute_lineage_closure(root, &edges))
    }

    /// Reserve the next strictly monotonic erasure generation inside the same
    /// immediate transaction that creates the job.  Callers must not allocate
    /// a generation and then commit it without a job/tombstone.
    pub fn allocate_generation(tx: &Transaction<'_>) -> UnlearningStoreResult<u64> {
        require_table(tx, "erasure_generation_counter")?;
        let current: i64 = tx.query_row(
            "SELECT last_generation FROM erasure_generation_counter WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        let next = current
            .checked_add(1)
            .ok_or(UnlearningStoreError::IntegerOverflow {
                field: "erasure generation",
            })?;
        tx.execute(
            "UPDATE erasure_generation_counter SET last_generation = ?1 WHERE singleton = 1",
            params![next],
        )?;
        Ok(next as u64)
    }

    /// Insert one immutable `running` job.  The caller supplies the generation
    /// from [`Self::allocate_generation`] in the same transaction.
    pub fn create_job(
        tx: &Transaction<'_>,
        generation: u64,
        input: &ErasureJobStart<'_>,
    ) -> UnlearningStoreResult<()> {
        require_table(tx, "erasure_jobs")?;
        validate_start(input, generation)?;
        tx.execute(
            r#"INSERT INTO erasure_jobs (
                erasure_id, schema_uri, scope, status, generation, commitment_key_id,
                lineage_epoch, fence_commitment, target_commitment, exact_content_commitment,
                source_locator_commitment, closure_commitment, started_at
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13
             )"#,
            params![
                input.erasure_id,
                VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1,
                scope_code(input.scope),
                ErasureJobStatus::Running.as_str(),
                to_sqlite_i64(generation, "erasure generation")?,
                input.target.target_identifier.key_id,
                to_sqlite_i64(input.lineage_epoch, "lineage epoch")?,
                input.fence_commitment.digest,
                input.target.target_identifier.digest,
                input.target.exact_content.digest,
                input
                    .target
                    .source_locator
                    .as_ref()
                    .map(|value| &value.digest),
                input.closure_commitment.digest,
                input.started_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Persist one closed-code execution step.  Counts are observations only;
    /// detailed diagnostics must remain transient and never enter this ledger.
    pub fn record_surface_step(
        tx: &Transaction<'_>,
        erasure_id: &str,
        step_order: u64,
        result: &SurfaceResult,
    ) -> UnlearningStoreResult<()> {
        require_table(tx, "erasure_steps")?;
        validate_erasure_id(erasure_id)?;
        tx.execute(
            r#"INSERT INTO erasure_steps (
                 erasure_id, step_order, surface, action, matched_count, changed_count,
                 verification_status, detail_code
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
            params![
                erasure_id,
                to_sqlite_i64(step_order, "erasure step order")?,
                surface_code(result.surface),
                action_code(result.action),
                to_sqlite_i64(result.matched_count, "surface matched count")?,
                to_sqlite_i64(result.changed_count, "surface changed count")?,
                check_status_code(result.status),
                detail_code(result.detail_code),
            ],
        )?;
        Ok(())
    }

    /// Install the anti-resurrection tombstone.  Duplicate calls are idempotent
    /// only when every persisted commitment and the original generation agree;
    /// otherwise the coordinator must stop rather than overwrite evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn write_tombstone(
        tx: &Transaction<'_>,
        receipt_id: &str,
        generation: u64,
        scope: UnlearningScope,
        target: &AntiResurrectionCommitments,
        erased_at: DateTime<Utc>,
    ) -> UnlearningStoreResult<TombstoneWriteOutcome> {
        require_table(tx, "erasure_tombstones")?;
        validate_closed_identifier(receipt_id)?;
        validate_target_commitments(target)?;
        let generation = to_sqlite_i64(generation, "erasure generation")?;
        let inserted = tx.execute(
            r#"INSERT INTO erasure_tombstones (
                 target_commitment, exact_content_commitment, source_locator_commitment,
                 commitment_key_id, generation, scope, receipt_id, erased_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(target_commitment) DO NOTHING"#,
            params![
                target.target_identifier.digest,
                target.exact_content.digest,
                target.source_locator.as_ref().map(|value| &value.digest),
                target.target_identifier.key_id,
                generation,
                scope_code(scope),
                receipt_id,
                erased_at.to_rfc3339(),
            ],
        )?;
        if inserted == 1 {
            return Ok(TombstoneWriteOutcome::Inserted);
        }

        let existing = tx
            .query_row(
                r#"SELECT exact_content_commitment, source_locator_commitment, commitment_key_id,
                        generation, scope, receipt_id
                   FROM erasure_tombstones WHERE target_commitment = ?1"#,
                params![target.target_identifier.digest],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()?;
        let Some((content, source, key_id, old_generation, old_scope, old_receipt)) = existing
        else {
            return Err(UnlearningStoreError::TombstoneConflict);
        };
        let expected_source = target
            .source_locator
            .as_ref()
            .map(|value| value.digest.as_str());
        if content == target.exact_content.digest
            && source.as_deref() == expected_source
            && key_id == target.target_identifier.key_id
            && old_generation == generation
            && old_scope == scope_code(scope)
            && old_receipt == receipt_id
        {
            Ok(TombstoneWriteOutcome::AlreadyPresent)
        } else {
            Err(UnlearningStoreError::TombstoneConflict)
        }
    }

    /// Check the three exact anti-resurrection gates without returning any raw
    /// target material.  Imports/restores/connectors should reject if *any*
    /// field is true.
    pub fn check_anti_resurrection(
        tx: &Transaction<'_>,
        target: &AntiResurrectionCommitments,
    ) -> UnlearningStoreResult<AntiResurrectionGateStatus> {
        require_table(tx, "erasure_tombstones")?;
        validate_target_commitments(target)?;
        let key_id = &target.target_identifier.key_id;
        let target_identifier_blocked = exists(
            tx,
            "SELECT 1 FROM erasure_tombstones WHERE target_commitment = ?1 AND commitment_key_id = ?2 LIMIT 1",
            params![target.target_identifier.digest, key_id],
        )?;
        let exact_content_blocked = exists(
            tx,
            "SELECT 1 FROM erasure_tombstones WHERE exact_content_commitment = ?1 AND commitment_key_id = ?2 LIMIT 1",
            params![target.exact_content.digest, key_id],
        )?;
        let source_locator_blocked = match &target.source_locator {
            Some(source) => exists(
                tx,
                "SELECT 1 FROM erasure_tombstones WHERE source_locator_commitment = ?1 AND commitment_key_id = ?2 LIMIT 1",
                params![source.digest, key_id],
            )?,
            None => false,
        };
        Ok(AntiResurrectionGateStatus {
            target_identifier_blocked,
            exact_content_blocked,
            source_locator_blocked,
        })
    }

    /// Run the mandatory V25 local scan.  A missing expected table is a failed
    /// verification, never an implicit "not configured" success.
    pub fn scan_required_local_canaries(
        tx: &Transaction<'_>,
        canary: &ExactCanary<'_>,
    ) -> UnlearningStoreResult<Vec<CanaryScanResult>> {
        Self::scan_exact_canary_tables(tx, V25_REQUIRED_LOCAL_CANARY_TABLES, canary)
    }

    /// Scan a selected closed subset.  This is useful for isolated tests and
    /// for legacy audited purge, whereas verified-local execution must call
    /// [`Self::scan_required_local_canaries`].
    pub fn scan_exact_canary_tables(
        tx: &Transaction<'_>,
        tables: &[LocalCanaryTable],
        canary: &ExactCanary<'_>,
    ) -> UnlearningStoreResult<Vec<CanaryScanResult>> {
        canary.validate()?;
        tables
            .iter()
            .copied()
            .map(|table| {
                let (name, sql) = canary_query(table);
                require_table(tx, name)?;
                let matched: i64 =
                    tx.query_row(sql, params![canary.identifier, canary.content], |row| {
                        row.get(0)
                    })?;
                Ok(CanaryScanResult {
                    table,
                    matching_rows: nonnegative_u64(matched, "canary matching rows")?,
                })
            })
            .collect()
    }

    /// Extract a signable record only when the pure model returned a truthful
    /// verified-local or legacy-audited verdict.  `Incomplete` and `Failed`
    /// results must instead use [`Self::mark_job_failed`].
    pub fn assemble_eligible_audit(
        record: &ErasureLedgerRecord,
    ) -> UnlearningStoreResult<EligibleAuditRecord> {
        match record.verdict() {
            UnlearningVerdict::VerifiedWithinScope | UnlearningVerdict::AuditedPurgeOnly => {}
            other => return Err(UnlearningStoreError::IneligibleAuditVerdict(other)),
        }
        let signing_payload = record
            .signing_payload()
            .map_err(|error| UnlearningStoreError::MalformedAuditPayload(error.to_string()))?;
        let identity = parse_audit_identity(&signing_payload)?;
        if identity.schema != VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1
            || identity.generation != record.generation()
            || identity.scope != record.scope()
        {
            return Err(UnlearningStoreError::MalformedAuditPayload(
                "ledger identity does not match its typed record".into(),
            ));
        }
        Ok(EligibleAuditRecord {
            erasure_id: identity.erasure_id,
            generation: identity.generation,
            scope: identity.scope,
            verdict: identity.verdict,
            signing_payload,
        })
    }

    /// Atomically attach an eligible audit record to its running job.  The
    /// optional signature is opaque transport metadata from the DSSE layer;
    /// this storage helper does not treat it as proof of deletion.
    pub fn commit_eligible_audit(
        tx: &Transaction<'_>,
        audit: &EligibleAuditRecord,
        signature_json: Option<&str>,
        committed_at: DateTime<Utc>,
    ) -> UnlearningStoreResult<()> {
        require_table(tx, "erasure_jobs")?;
        validate_erasure_id(&audit.erasure_id)?;
        let identity = parse_audit_identity(&audit.signing_payload)?;
        if identity.schema != VERIFIED_LOCAL_UNLEARNING_SCHEMA_V1
            || identity.erasure_id != audit.erasure_id
            || identity.generation != audit.generation
            || identity.scope != audit.scope
            || identity.verdict != audit.verdict
            || !matches!(
                identity.verdict,
                UnlearningVerdict::VerifiedWithinScope | UnlearningVerdict::AuditedPurgeOnly
            )
        {
            return Err(UnlearningStoreError::MalformedAuditPayload(
                "eligible audit payload does not match its sealed identity".into(),
            ));
        }
        let status = current_job_status(tx, &audit.erasure_id)?;
        if status != ErasureJobStatus::Running {
            return Err(UnlearningStoreError::JobNotRunning(
                audit.erasure_id.clone(),
            ));
        }
        let changed = tx.execute(
            r#"UPDATE erasure_jobs
                SET status = ?1, committed_at = ?2, result_json = ?3, signature_json = ?4,
                    failure_code = NULL
              WHERE erasure_id = ?5 AND generation = ?6 AND scope = ?7 AND status = 'running'"#,
            params![
                ErasureJobStatus::Committed.as_str(),
                committed_at.to_rfc3339(),
                std::str::from_utf8(&audit.signing_payload).map_err(|error| {
                    UnlearningStoreError::MalformedAuditPayload(error.to_string())
                })?,
                signature_json,
                audit.erasure_id,
                to_sqlite_i64(audit.generation, "erasure generation")?,
                scope_code(audit.scope),
            ],
        )?;
        if changed != 1 {
            return Err(UnlearningStoreError::JobNotRunning(
                audit.erasure_id.clone(),
            ));
        }
        Ok(())
    }

    /// Close an unsuccessful execution with a closed failure code.  This path
    /// cannot store failure diagnostics or target material.
    pub fn mark_job_failed(
        tx: &Transaction<'_>,
        erasure_id: &str,
        code: ErasureFailureCode,
    ) -> UnlearningStoreResult<()> {
        require_table(tx, "erasure_jobs")?;
        validate_erasure_id(erasure_id)?;
        let changed = tx.execute(
            r#"UPDATE erasure_jobs
                SET status = ?1, failure_code = ?2
              WHERE erasure_id = ?3 AND status = 'running'"#,
            params![ErasureJobStatus::Failed.as_str(), code.as_str(), erasure_id],
        )?;
        if changed != 1 {
            return Err(UnlearningStoreError::JobNotRunning(erasure_id.to_string()));
        }
        Ok(())
    }

    /// Load the content-free lifecycle metadata for one job.
    pub fn load_job(
        tx: &Transaction<'_>,
        erasure_id: &str,
    ) -> UnlearningStoreResult<Option<StoredErasureJob>> {
        require_table(tx, "erasure_jobs")?;
        validate_erasure_id(erasure_id)?;
        tx.query_row(
            r#"SELECT erasure_id, scope, status, generation, commitment_key_id, lineage_epoch,
                    started_at, committed_at, failure_code
               FROM erasure_jobs WHERE erasure_id = ?1"#,
            params![erasure_id],
            |row| {
                let scope: String = row.get(1)?;
                let status: String = row.get(2)?;
                let generation: i64 = row.get(3)?;
                let lineage_epoch: i64 = row.get(5)?;
                let started_at: String = row.get(6)?;
                let committed_at: Option<String> = row.get(7)?;
                let failure_code: Option<String> = row.get(8)?;
                Ok((
                    row.get::<_, String>(0)?,
                    scope,
                    status,
                    generation,
                    row.get::<_, String>(4)?,
                    lineage_epoch,
                    started_at,
                    committed_at,
                    failure_code,
                ))
            },
        )
        .optional()?
        .map(|row| {
            Ok(StoredErasureJob {
                erasure_id: row.0,
                scope: parse_scope(&row.1)?,
                status: ErasureJobStatus::parse(&row.2)?,
                generation: nonnegative_u64(row.3, "stored erasure generation")?,
                commitment_key_id: row.4,
                lineage_epoch: nonnegative_u64(row.5, "stored lineage epoch")?,
                started_at: parse_timestamp(&row.6)?,
                committed_at: row.7.as_deref().map(parse_timestamp).transpose()?,
                failure_code: row.8.as_deref().map(parse_failure_code).transpose()?,
            })
        })
        .transpose()
    }
}

fn validate_start(input: &ErasureJobStart<'_>, generation: u64) -> UnlearningStoreResult<()> {
    validate_erasure_id(input.erasure_id)?;
    if generation == 0 {
        return Err(UnlearningStoreError::IntegerOverflow {
            field: "erasure generation",
        });
    }
    validate_target_commitments(input.target)?;
    validate_commitment(input.fence_commitment, CommitmentKind::ExecutionFence)?;
    validate_commitment(input.closure_commitment, CommitmentKind::LineageClosure)?;
    let key_id = &input.target.target_identifier.key_id;
    if input.fence_commitment.key_id != *key_id || input.closure_commitment.key_id != *key_id {
        return Err(UnlearningStoreError::MixedCommitmentKeys);
    }
    Ok(())
}

fn validate_target_commitments(target: &AntiResurrectionCommitments) -> UnlearningStoreResult<()> {
    validate_commitment(&target.target_identifier, CommitmentKind::TargetIdentifier)?;
    validate_commitment(&target.exact_content, CommitmentKind::ExactContent)?;
    if target.target_identifier.key_id != target.exact_content.key_id {
        return Err(UnlearningStoreError::MixedCommitmentKeys);
    }
    if let Some(source) = &target.source_locator {
        validate_commitment(source, CommitmentKind::SourceLocator)?;
        if source.key_id != target.target_identifier.key_id {
            return Err(UnlearningStoreError::MixedCommitmentKeys);
        }
    }
    Ok(())
}

fn validate_commitment(value: &Commitment, expected: CommitmentKind) -> UnlearningStoreResult<()> {
    if value.version != 1
        || value.kind != expected
        || !is_lower_hex(&value.digest, 64)
        || !is_lower_hex(&value.key_id, 32)
    {
        return Err(UnlearningStoreError::InvalidCommitment(expected));
    }
    Ok(())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_erasure_id(value: &str) -> UnlearningStoreResult<()> {
    if !value.is_empty()
        && value.len() <= 96
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
    {
        Ok(())
    } else {
        Err(UnlearningStoreError::InvalidErasureId)
    }
}

fn validate_closed_identifier(value: &str) -> UnlearningStoreResult<()> {
    validate_erasure_id(value)
}

fn to_sqlite_i64(value: u64, field: &'static str) -> UnlearningStoreResult<i64> {
    i64::try_from(value).map_err(|_| UnlearningStoreError::IntegerOverflow { field })
}

fn nonnegative_u64(value: i64, field: &'static str) -> UnlearningStoreResult<u64> {
    u64::try_from(value).map_err(|_| UnlearningStoreError::IntegerOverflow { field })
}

fn unknown_code(kind: &'static str, value: &str) -> UnlearningStoreError {
    UnlearningStoreError::UnknownClosedCode {
        kind,
        value: value.to_owned(),
    }
}

fn malformed_model(error: impl fmt::Display) -> UnlearningStoreError {
    UnlearningStoreError::MalformedAuditPayload(error.to_string())
}

fn parse_timestamp(value: &str) -> UnlearningStoreResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| UnlearningStoreError::MalformedAuditPayload(error.to_string()))
}

fn parse_scope(value: &str) -> UnlearningStoreResult<UnlearningScope> {
    match value {
        "legacy_audited_purge" => Ok(UnlearningScope::LegacyAuditedPurge),
        "post_lineage_verified_local" => Ok(UnlearningScope::PostLineageVerifiedLocal),
        _ => Err(unknown_code("unlearning scope", value)),
    }
}

const fn scope_code(value: UnlearningScope) -> &'static str {
    match value {
        UnlearningScope::LegacyAuditedPurge => "legacy_audited_purge",
        UnlearningScope::PostLineageVerifiedLocal => "post_lineage_verified_local",
    }
}

fn parse_failure_code(value: &str) -> UnlearningStoreResult<ErasureFailureCode> {
    match value {
        "verification_failed" => Ok(ErasureFailureCode::VerificationFailed),
        "lineage_incomplete" => Ok(ErasureFailureCode::LineageIncomplete),
        "tombstone_conflict" => Ok(ErasureFailureCode::TombstoneConflict),
        "durability_failure" => Ok(ErasureFailureCode::DurabilityFailure),
        "runtime_rebuild_failed" => Ok(ErasureFailureCode::RuntimeRebuildFailed),
        "operator_cancelled" => Ok(ErasureFailureCode::OperatorCancelled),
        _ => Err(unknown_code("erasure failure", value)),
    }
}

fn parse_artifact_kind(value: &str) -> UnlearningStoreResult<ArtifactKind> {
    match value {
        "knowledge_node" => Ok(ArtifactKind::KnowledgeNode),
        "embedding" => Ok(ArtifactKind::Embedding),
        "fts_document" => Ok(ArtifactKind::FtsDocument),
        "vector_entry" => Ok(ArtifactKind::VectorEntry),
        "insight" => Ok(ArtifactKind::Insight),
        "temporal_summary" => Ok(ArtifactKind::TemporalSummary),
        "domain_projection" => Ok(ArtifactKind::DomainProjection),
        "intention" => Ok(ArtifactKind::Intention),
        "merge_plan" => Ok(ArtifactKind::MergePlan),
        "merge_operation" => Ok(ArtifactKind::MergeOperation),
        "composition_event" => Ok(ArtifactKind::CompositionEvent),
        "trace_event" => Ok(ArtifactKind::TraceEvent),
        "receipt" => Ok(ArtifactKind::Receipt),
        "memory_pr" => Ok(ArtifactKind::MemoryPr),
        "synaptic_tag" => Ok(ArtifactKind::SynapticTag),
        "synaptic_event" => Ok(ArtifactKind::SynapticEvent),
        "synaptic_capture_item" => Ok(ArtifactKind::SynapticCaptureItem),
        "runtime_projection" => Ok(ArtifactKind::RuntimeProjection),
        "managed_backup" => Ok(ArtifactKind::ManagedBackup),
        "managed_export" => Ok(ArtifactKind::ManagedExport),
        "managed_sync_object" => Ok(ArtifactKind::ManagedSyncObject),
        "connector_source" => Ok(ArtifactKind::ConnectorSource),
        _ => Err(unknown_code("artifact kind", value)),
    }
}

fn parse_lineage_relation(value: &str) -> UnlearningStoreResult<LineageRelation> {
    match value {
        "derived_from" => Ok(LineageRelation::DerivedFrom),
        "embeds" => Ok(LineageRelation::Embeds),
        "indexes" => Ok(LineageRelation::Indexes),
        "summarizes" => Ok(LineageRelation::Summarizes),
        "aggregates" => Ok(LineageRelation::Aggregates),
        "mentions" => Ok(LineageRelation::Mentions),
        "mutates" => Ok(LineageRelation::Mutates),
        "mirrors" => Ok(LineageRelation::Mirrors),
        _ => Err(unknown_code("lineage relation", value)),
    }
}

const fn surface_code(value: super::unlearning::UnlearningSurface) -> &'static str {
    use super::unlearning::UnlearningSurface;
    match value {
        UnlearningSurface::CanonicalSqlite => "canonical_sqlite",
        UnlearningSurface::LinkedSqliteArtifacts => "linked_sqlite_artifacts",
        UnlearningSurface::FullTextIndex => "full_text_index",
        UnlearningSurface::EmbeddingStore => "embedding_store",
        UnlearningSurface::VectorIndex => "vector_index",
        UnlearningSurface::CognitiveRuntime => "cognitive_runtime",
        UnlearningSurface::TraceAndReceiptEvidence => "trace_and_receipt_evidence",
        UnlearningSurface::ReviewAndUndoQueues => "review_and_undo_queues",
        UnlearningSurface::ConnectorAndRestoreGates => "connector_and_restore_gates",
        UnlearningSurface::ManagedBackups => "managed_backups",
        UnlearningSurface::ManagedExports => "managed_exports",
        UnlearningSurface::ManagedSyncObjects => "managed_sync_objects",
        UnlearningSurface::ErasureLedgerAnchor => "erasure_ledger_anchor",
    }
}

const fn action_code(value: SurfaceAction) -> &'static str {
    match value {
        SurfaceAction::Delete => "delete",
        SurfaceAction::Redact => "redact",
        SurfaceAction::Rebuild => "rebuild",
        SurfaceAction::Reset => "reset",
        SurfaceAction::RevokeKey => "revoke_key",
        SurfaceAction::InstallGate => "install_gate",
        SurfaceAction::Verify => "verify",
    }
}

const fn check_status_code(value: CheckStatus) -> &'static str {
    match value {
        CheckStatus::Passed => "passed",
        CheckStatus::Failed => "failed",
        CheckStatus::NotChecked => "not_checked",
        CheckStatus::OutOfScope => "out_of_scope",
    }
}

const fn detail_code(value: SurfaceDetailCode) -> &'static str {
    match value {
        SurfaceDetailCode::TargetRowsDeleted => "target_rows_deleted",
        SurfaceDetailCode::ArtifactRedacted => "artifact_redacted",
        SurfaceDetailCode::IndexRebuiltNoTargetHit => "index_rebuilt_no_target_hit",
        SurfaceDetailCode::RuntimeResetAndRehydrated => "runtime_reset_and_rehydrated",
        SurfaceDetailCode::MutationGateInstalled => "mutation_gate_installed",
        SurfaceDetailCode::ManagedArtifactResolved => "managed_artifact_resolved",
        SurfaceDetailCode::LedgerAnchorVerified => "ledger_anchor_verified",
        SurfaceDetailCode::VerifiedNoTargetHit => "verified_no_target_hit",
        SurfaceDetailCode::FeatureNotConfigured => "feature_not_configured",
        SurfaceDetailCode::LegacySurfaceAudited => "legacy_surface_audited",
    }
}

fn exists(
    tx: &Transaction<'_>,
    sql: &str,
    params: impl rusqlite::Params,
) -> UnlearningStoreResult<bool> {
    tx.query_row(sql, params, |_| Ok(()))
        .optional()
        .map(|value| value.is_some())
        .map_err(Into::into)
}

fn require_table(tx: &Transaction<'_>, table: &'static str) -> UnlearningStoreResult<()> {
    let present = tx
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1 LIMIT 1",
            params![table],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if present {
        Ok(())
    } else {
        Err(UnlearningStoreError::MissingRequiredTable(table))
    }
}

fn current_job_status(
    tx: &Transaction<'_>,
    erasure_id: &str,
) -> UnlearningStoreResult<ErasureJobStatus> {
    let status: Option<String> = tx
        .query_row(
            "SELECT status FROM erasure_jobs WHERE erasure_id = ?1",
            params![erasure_id],
            |row| row.get(0),
        )
        .optional()?;
    status
        .as_deref()
        .ok_or_else(|| UnlearningStoreError::JobNotRunning(erasure_id.to_owned()))
        .and_then(ErasureJobStatus::parse)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuditIdentity {
    schema: String,
    erasure_id: String,
    generation: u64,
    scope: UnlearningScope,
    verdict: UnlearningVerdict,
}

fn parse_audit_identity(bytes: &[u8]) -> UnlearningStoreResult<AuditIdentity> {
    serde_json::from_slice(bytes)
        .map_err(|error| UnlearningStoreError::MalformedAuditPayload(error.to_string()))
}

fn canary_query(table: LocalCanaryTable) -> (&'static str, &'static str) {
    match table {
        LocalCanaryTable::KnowledgeNodes => (
            "knowledge_nodes",
            r#"SELECT count(*) FROM knowledge_nodes WHERE id = ?1 OR instr(content, ?2) > 0
             OR instr(COALESCE(source, ''), ?2) > 0 OR instr(COALESCE(tags, ''), ?2) > 0
             OR instr(COALESCE(source_id, ''), ?1) > 0 OR instr(COALESCE(source_url, ''), ?2) > 0
             OR instr(COALESCE(source_author, ''), ?2) > 0"#,
        ),
        LocalCanaryTable::KnowledgeFts => (
            "knowledge_fts",
            r#"SELECT count(*) FROM knowledge_fts WHERE id = ?1 OR instr(content, ?2) > 0
             OR instr(tags, ?2) > 0"#,
        ),
        LocalCanaryTable::NodeEmbeddings => (
            "node_embeddings",
            "SELECT count(*) FROM node_embeddings WHERE node_id = ?1",
        ),
        LocalCanaryTable::MemoryConnections => (
            "memory_connections",
            "SELECT count(*) FROM memory_connections WHERE source_id = ?1 OR target_id = ?1",
        ),
        LocalCanaryTable::MemoryStates => (
            "memory_states",
            "SELECT count(*) FROM memory_states WHERE memory_id = ?1 OR instr(COALESCE(suppressed_by, ''), ?1) > 0",
        ),
        LocalCanaryTable::FsrsCards => (
            "fsrs_cards",
            "SELECT count(*) FROM fsrs_cards WHERE memory_id = ?1",
        ),
        LocalCanaryTable::Intentions => (
            "intentions",
            r#"SELECT count(*) FROM intentions WHERE instr(content, ?2) > 0 OR instr(trigger_data, ?2) > 0
             OR instr(COALESCE(notes, ''), ?2) > 0 OR instr(COALESCE(tags, ''), ?2) > 0
             OR instr(COALESCE(related_memories, ''), ?1) > 0 OR instr(COALESCE(source_data, ''), ?2) > 0"#,
        ),
        LocalCanaryTable::Insights => (
            "insights",
            r#"SELECT count(*) FROM insights WHERE instr(insight, ?2) > 0 OR instr(source_memories, ?1) > 0
             OR instr(COALESCE(tags, ''), ?2) > 0"#,
        ),
        LocalCanaryTable::MergePlans => (
            "merge_plans",
            r#"SELECT count(*) FROM merge_plans WHERE survivor_id = ?1 OR instr(member_ids, ?1) > 0
             OR instr(payload, ?1) > 0 OR instr(payload, ?2) > 0"#,
        ),
        LocalCanaryTable::MergeOperations => (
            "merge_operations",
            r#"SELECT count(*) FROM merge_operations WHERE survivor_id = ?1 OR instr(affected_ids, ?1) > 0
             OR instr(COALESCE(signals, ''), ?1) > 0 OR instr(COALESCE(signals, ''), ?2) > 0
             OR instr(COALESCE(reason, ''), ?2) > 0 OR instr(undo_payload, ?1) > 0
             OR instr(undo_payload, ?2) > 0"#,
        ),
        LocalCanaryTable::CompositionEvents => (
            "composition_events",
            r#"SELECT count(*) FROM composition_events WHERE instr(COALESCE(query, ''), ?2) > 0
             OR instr(COALESCE(output_preview, ''), ?2) > 0 OR instr(metadata, ?1) > 0
             OR instr(metadata, ?2) > 0"#,
        ),
        LocalCanaryTable::CompositionMembers => (
            "composition_members",
            r#"SELECT count(*) FROM composition_members WHERE memory_id = ?1 OR instr(COALESCE(preview, ''), ?2) > 0
             OR instr(metadata, ?1) > 0 OR instr(metadata, ?2) > 0"#,
        ),
        LocalCanaryTable::CompositionOutcomes => (
            "composition_outcomes",
            r#"SELECT count(*) FROM composition_outcomes WHERE instr(COALESCE(notes, ''), ?2) > 0
             OR instr(metadata, ?1) > 0 OR instr(metadata, ?2) > 0"#,
        ),
        LocalCanaryTable::AgentTraces => (
            "agent_traces",
            "SELECT count(*) FROM agent_traces WHERE instr(payload, ?1) > 0 OR instr(payload, ?2) > 0",
        ),
        LocalCanaryTable::MemoryReceipts => (
            "memory_receipts",
            r#"SELECT count(*) FROM memory_receipts WHERE instr(COALESCE(query, ''), ?2) > 0
             OR instr(payload, ?1) > 0 OR instr(payload, ?2) > 0"#,
        ),
        LocalCanaryTable::MemoryPrs => (
            "memory_prs",
            r#"SELECT count(*) FROM memory_prs WHERE subject_id = ?1 OR instr(title, ?2) > 0
             OR instr(diff, ?1) > 0 OR instr(diff, ?2) > 0
             OR instr(signals, ?1) > 0 OR instr(signals, ?2) > 0"#,
        ),
        LocalCanaryTable::SynapticTags => (
            "synaptic_tags",
            "SELECT count(*) FROM synaptic_tags WHERE memory_id = ?1 OR instr(COALESCE(encoding_context, ''), ?2) > 0",
        ),
        LocalCanaryTable::SynapticEvents => (
            "synaptic_events",
            "SELECT count(*) FROM synaptic_events WHERE trigger_memory_id = ?1",
        ),
        LocalCanaryTable::SynapticCaptureItems => (
            "synaptic_capture_items",
            r#"SELECT count(*) FROM synaptic_capture_items WHERE memory_id = ?1
             OR instr(COALESCE(reason, ''), ?2) > 0"#,
        ),
        LocalCanaryTable::ReplayCapsules => (
            "retrieval_replay_capsules",
            "SELECT count(*) FROM retrieval_replay_capsules WHERE instr(source_receipt_id, ?1) > 0",
        ),
        LocalCanaryTable::ReplayItems => (
            "retrieval_replay_items",
            "SELECT count(*) FROM retrieval_replay_items WHERE memory_id = ?1",
        ),
        LocalCanaryTable::CounterfactualReplays => (
            "counterfactual_replays",
            r#"SELECT count(*) FROM counterfactual_replays WHERE instr(source_receipt_id, ?1) > 0
             OR instr(COALESCE(result_json, ''), ?1) > 0 OR instr(COALESCE(result_json, ''), ?2) > 0"#,
        ),
        LocalCanaryTable::ArtifactLineage => (
            "artifact_lineage",
            "SELECT count(*) FROM artifact_lineage WHERE source_id = ?1 OR derived_id = ?1",
        ),
        LocalCanaryTable::SyncTombstones => (
            "sync_tombstones",
            "SELECT count(*) FROM sync_tombstones WHERE row_id = ?1 OR instr(COALESCE(reason, ''), ?2) > 0",
        ),
        LocalCanaryTable::DeletionTombstones => (
            "deletion_tombstones",
            r#"SELECT count(*) FROM deletion_tombstones WHERE memory_id = ?1 OR instr(tags, ?2) > 0
             OR instr(COALESCE(reason, ''), ?2) > 0"#,
        ),
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rusqlite::{Connection, TransactionBehavior, params};

    use super::*;
    use crate::storage::unlearning::{
        CommitmentKey, SurfaceAction, SurfaceDetailCode, UnlearningSurface,
        anti_resurrection_commitments, commit_lineage_closure,
    };

    fn key() -> CommitmentKey {
        CommitmentKey::derive(b"0123456789abcdef0123456789abcdef").unwrap()
    }

    fn root() -> ArtifactRef {
        ArtifactRef::new(ArtifactKind::KnowledgeNode, "node-canary").unwrap()
    }

    #[test]
    fn loads_cycle_safe_closure_and_writes_content_free_tombstone() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(V25_UNLEARNING_STORAGE_SCHEMA_EXPECTATION)
            .unwrap();
        connection.execute_batch(
            "CREATE TABLE knowledge_nodes (id TEXT PRIMARY KEY, content TEXT, source TEXT, tags TEXT,\
               source_id TEXT, source_url TEXT, source_author TEXT);",
        ).unwrap();
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        tx.execute(
            "INSERT INTO artifact_lineage(source_kind, source_id, derived_kind, derived_id, relation, created_at)\
             VALUES ('knowledge_node', 'node-canary', 'insight', 'insight-a', 'summarizes', ?1),\
                    ('insight', 'insight-a', 'knowledge_node', 'node-canary', 'mentions', ?1)",
            params![Utc::now().to_rfc3339()],
        ).unwrap();
        let closure = UnlearningStore::load_lineage_closure(&tx, &root()).unwrap();
        assert_eq!(closure.len(), 2);

        let commitment_key = key();
        let target = anti_resurrection_commitments(
            &commitment_key,
            "node-canary",
            b"content-canary",
            Some("connector://record"),
        );
        let fence = super::super::unlearning::VerificationFence::from_held_epoch(
            &commitment_key,
            7,
            b"opaque-fence-token",
        )
        .unwrap();
        let closure_commitment = commit_lineage_closure(&commitment_key, &closure).unwrap();
        let generation = UnlearningStore::allocate_generation(&tx).unwrap();
        UnlearningStore::create_job(
            &tx,
            generation,
            &ErasureJobStart {
                erasure_id: "erase-test-1",
                scope: UnlearningScope::PostLineageVerifiedLocal,
                lineage_epoch: 7,
                fence_commitment: fence.commitment(),
                target: &target,
                closure_commitment: &closure_commitment,
                started_at: Utc::now(),
            },
        )
        .unwrap();
        UnlearningStore::record_surface_step(
            &tx,
            "erase-test-1",
            0,
            &SurfaceResult {
                surface: UnlearningSurface::CanonicalSqlite,
                action: SurfaceAction::Delete,
                matched_count: 1,
                changed_count: 1,
                status: CheckStatus::Passed,
                detail_code: SurfaceDetailCode::TargetRowsDeleted,
            },
        )
        .unwrap();
        assert_eq!(
            UnlearningStore::write_tombstone(
                &tx,
                "erase-test-1",
                generation,
                UnlearningScope::PostLineageVerifiedLocal,
                &target,
                Utc::now(),
            )
            .unwrap(),
            TombstoneWriteOutcome::Inserted,
        );
        assert!(
            UnlearningStore::check_anti_resurrection(&tx, &target)
                .unwrap()
                .blocks_any()
        );
        tx.commit().unwrap();
    }

    #[test]
    fn selected_canary_scan_returns_counts_without_returning_canary_values() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection.execute_batch(
            "CREATE TABLE knowledge_nodes (id TEXT PRIMARY KEY, content TEXT, source TEXT, tags TEXT,\
               source_id TEXT, source_url TEXT, source_author TEXT);",
        ).unwrap();
        let tx = connection.transaction().unwrap();
        tx.execute(
            "INSERT INTO knowledge_nodes VALUES ('node-canary', 'content-canary', '', '[]', '', '', '')",
            [],
        ).unwrap();
        let result = UnlearningStore::scan_exact_canary_tables(
            &tx,
            &[LocalCanaryTable::KnowledgeNodes],
            &ExactCanary {
                identifier: "node-canary",
                content: "content-canary",
            },
        )
        .unwrap();
        assert_eq!(result[0].matching_rows, 1);
    }
}
