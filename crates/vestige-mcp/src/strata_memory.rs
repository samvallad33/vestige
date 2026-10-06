//! `MemoryStore` adapter over [`strata_store::StrataStore`].
//!
//! Writes go through the store's PROPOSE → GATE → EFFECT path. Reads are the
//! replayed registry. Similarity (embeddings, cosine, BM25, FTS, Jaccard,
//! keyword or name match) is refused; a link exists only when the log
//! recorded an edge.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use strata_store::VALID_FOREVER_MS;
use vestige_core::codebase::{AnchorStatus, CodeAnchor};
use vestige_core::intention_graph::Command;
use vestige_core::storage::{
    CompositionEventRecord, ConnectionRecord as VestigeEdge, ConsolidationHistoryRecord,
    CoverageSnapshot, HANDLE_REQUIRED_DETAIL, HandleKind, HandleResolution, HealthStatus,
    HygieneNodeSummary, HygieneSnapshot, MAX_CANDIDATES, MemoryEdge, MemoryRecord,
    MemoryStoreError, MemoryStoreResult, MemoryStoreSend, ModelSignature, NeverComposedCandidate,
    ReceiptAttestationStatus, SchedulingState, SearchQuery, StateTransitionRecord, Storage,
    StorageError, StoreStats, WalCheckpointMode, WalCheckpointStatus,
};
use vestige_core::{
    ConsolidationResult, DecayRisk, IngestInput, KnowledgeNode, MemoryStats, Rating, Receipt,
    SecretPolicy, SourceEnvelope, scan_secrets,
};

pub mod ghostlink;

const Q32_SCALE: f64 = 4294967296.0;
const RECEIPT_PREFIX: &str = "eff-";
const STRATA_REPLAY_BOUNDARY: &str = "Replay re-derives state from the Strata log and compares it to the receipt. A match checks the log; it is not a claim about the world.";

fn node_field_mismatches(
    id: &str,
    live: &strata_store::NodeRecord,
    folded: &strata_store::NodeRecord,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |field: &str| out.push(format!("node:{id}:{field}"));
    if live.content != folded.content {
        push("content");
    }
    if live.node_type != folded.node_type {
        push("node_type");
    }
    if live.tags != folded.tags {
        push("tags");
    }
    if live.scope != folded.scope {
        push("scope");
    }
    if live.kernel_id != folded.kernel_id {
        push("kernel_id");
    }
    if live.created_at_ms != folded.created_at_ms {
        push("created_at_ms");
    }
    if live.valid_from_ms != folded.valid_from_ms {
        push("valid_from_ms");
    }
    if live.valid_until_ms != folded.valid_until_ms {
        push("valid_until_ms");
    }
    if live.superseded_by != folded.superseded_by {
        push("superseded_by");
    }
    out
}

/// FSRS-6 Easy. Explicit "this helped" review; the kernel's recall update.
const PROMOTE_RATING: u8 = 4;
/// FSRS-6 Again. Explicit "this failed" review; the kernel's forget update.
const DEMOTE_RATING: u8 = 1;

/// Side file for the Fellegi-Sunter thresholds. Not a log frame: `StoreOp`
/// has no policy variant, and this file is not read by replay.
const MERGE_POLICY_FILE: &str = "merge-policy.json";

/// The durable directory this process opened is a Strata log, not a SQLite file.
pub fn is_strata_backend(storage: &Storage) -> bool {
    storage.is_strata() || storage.is_strata_log()
}

/// Open (or create) a Strata log under `dir`. Creates no SQLite file.
pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Storage>, StorageError> {
    let memory = Arc::new(StrataMemory::open(dir)?);
    register_open(&memory);
    Ok(memory)
}

static OPEN_LOGS: Mutex<Vec<(PathBuf, Weak<StrataMemory>)>> = Mutex::new(Vec::new());

pub(crate) fn register_open(memory: &Arc<StrataMemory>) {
    let mut open = OPEN_LOGS.lock().unwrap_or_else(|err| err.into_inner());
    open.retain(|(_, weak)| weak.strong_count() > 0);
    open.push((memory.log_dir.clone(), Arc::downgrade(memory)));
}

fn live_memory(storage: &Storage) -> Option<Arc<StrataMemory>> {
    if !is_strata_backend(storage) {
        return None;
    }
    let path = storage.db_path().to_path_buf();
    let mut open = OPEN_LOGS.lock().unwrap_or_else(|err| err.into_inner());
    open.retain(|(_, weak)| weak.strong_count() > 0);
    open.iter()
        .find(|(log_dir, _)| log_dir == &path)
        .and_then(|(_, weak)| weak.upgrade())
}

/// What a whole-log verification covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogCheck {
    /// Segments scanned, sealed and active.
    pub segments: u32,
    /// Sealed segments whose signed trailer was checked.
    pub sealed_segments: u32,
    /// Frames read back and re-hashed.
    pub frames: u64,
}

/// Re-verify every segment of the Strata log this process has open: frame and
/// segment hash chains, payload hashes, each sealed segment's signed trailer.
/// Damage is an error, and the log then refuses further writes.
pub fn verify_log(storage: &Storage) -> Result<LogCheck, String> {
    let memory =
        live_memory(storage).ok_or_else(|| "strata log is not open in this process".to_string())?;
    let store = memory.lock();
    store
        .log()
        .verify_log()
        .map(|report| LogCheck {
            segments: report.segments,
            sealed_segments: report.sealed_segments,
            frames: report.frames,
        })
        .map_err(|err| err.to_string())
}

/// `memory_status` view `provenance` on a Strata log.
///
/// The creating frame and any recorded supersede chain, read from the log.
/// The log records no actor and no source envelope; both are null.
pub fn execute_node_provenance(storage: &Storage, args: Option<&Value>) -> Result<Value, String> {
    let id = args
        .and_then(|value| value.get("memoryId").or_else(|| value.get("id")))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or("provenance requires memoryId")?;
    let memory = live_memory(storage)
        .ok_or_else(|| "provenance: strata log is not open in this process".to_string())?;
    let store = memory.lock();
    let origin = store.recorded_origin(id).map_err(|err| err.to_string())?;
    let current = store.get_node(id);
    let Some(origin) = origin else {
        // A memory imported by the v3 upgrade has no admitted write on this
        // log: its record came from the migration frames the signed
        // migration receipt covers.
        if let Some(record) = current.filter(retrievable) {
            let valid_until =
                (record.valid_until_ms != VALID_FOREVER_MS).then_some(record.valid_until_ms);
            return Ok(json!({
                "view": "provenance",
                "status": "completed",
                "found": true,
                "memoryId": record.id,
                "origin": {
                    "kind": "v3_import",
                    "note": "Imported from a v3 store by the upgrade. The signed migration receipt covers it; no admitted write frame exists for it on this log.",
                    "actor": Value::Null,
                    "timestamps": {
                        "createdAtMs": record.created_at_ms,
                        "validFromMs": record.valid_from_ms,
                        "validUntilMs": valid_until,
                    },
                    "record": {
                        "id": record.id,
                        "scope": record.scope,
                        "content": record.content,
                        "nodeType": record.node_type,
                        "tags": record.tags,
                        "kernelId": record.kernel_id,
                    },
                },
                "supersedeChain": [],
            }));
        }
        return Ok(json!({
            "view": "provenance",
            "status": "completed",
            "found": false,
            "memoryId": id,
        }));
    };
    // A memory hidden without a successor (v3-suppressed, or an undone
    // creation) keeps its provenance but not its content, since every other
    // read hides it. An edited or merged memory names its successor and its
    // original text is history, so that stays readable.
    let hidden = current
        .as_ref()
        .and_then(|record| record.superseded_by.as_deref())
        .is_some_and(|successor| store.get_node(successor).is_none());
    let record = &origin.record;
    let valid_until = (record.valid_until_ms != VALID_FOREVER_MS).then_some(record.valid_until_ms);
    let chain: Vec<Value> = origin
        .supersede_chain
        .iter()
        .map(|hop| {
            json!({
                "id": hop.id,
                "supersededBy": hop.superseded_by,
                "frameSeq": hop.frame_seq,
                "frameHash": hex32(&hop.frame_hash),
                "recordedAs": hop.recorded_as,
            })
        })
        .collect();
    Ok(json!({
        "view": "provenance",
        "status": "completed",
        "found": true,
        "memoryId": record.id,
        "origin": {
            "frame": {
                "seq": origin.frame_seq,
                "kind": origin.frame_kind,
                "kindName": "STORE_WRITE",
                "op": "UpsertNode",
                "frameHash": hex32(&origin.frame_hash),
                "payloadBlake3": hex32(&origin.payload_blake3),
                "proposeSeq": origin.propose_frame_seq,
                "gateSeq": origin.gate_frame_seq,
                "effectSeq": origin.effect_frame_seq,
            },
            "actor": Value::Null,
            "timestamps": {
                "createdAtMs": record.created_at_ms,
                "validFromMs": record.valid_from_ms,
                "validUntilMs": valid_until,
            },
            "source": Value::Null,
            "record": {
                "id": record.id,
                "scope": record.scope,
                "content": if hidden { Value::Null } else { json!(record.content) },
                "withheld": hidden,
                "nodeType": record.node_type,
                "tags": record.tags,
                "kernelId": record.kernel_id,
            },
        },
        "supersedeChain": chain,
    }))
}

fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub struct StrataMemory {
    data_dir: PathBuf,
    log_dir: PathBuf,
    store: Mutex<strata_store::StrataStore>,
    actor: Mutex<Option<String>>,
    /// `Some` only after `set_merge_policy`. A get with `None` reads the side
    /// file, then env, then the built-in defaults.
    merge_policy: Mutex<Option<vestige_core::MergePolicy>>,
}

/// File in the data directory recording when maintenance last completed:
/// `{"lastBackupMs": .., "lastDreamMs": ..}`. It holds timestamps only, never
/// memory content, and is read by `session_start` for `needsBackup` and
/// `needsDream`.
const MAINTENANCE_STAMPS: &str = "maintenance-stamps.json";

impl StrataMemory {
    /// When `key` last completed, or `None` when it never has or the stamp
    /// file is missing or unreadable.
    fn maintenance_stamp(&self, key: &str) -> Option<DateTime<Utc>> {
        let raw = std::fs::read(self.data_dir.join(MAINTENANCE_STAMPS)).ok()?;
        let stamps: Value = serde_json::from_slice(&raw).ok()?;
        DateTime::from_timestamp_millis(stamps.get(key)?.as_i64()?)
    }

    /// Record that `key` completed at `at`. A stamp only moves forward, the
    /// other stamps are kept, and the file is replaced by rename so a crash
    /// never leaves it torn.
    fn record_maintenance_stamp(&self, key: &str, at: DateTime<Utc>) -> std::io::Result<()> {
        let path = self.data_dir.join(MAINTENANCE_STAMPS);
        let mut stamps = std::fs::read(&path)
            .ok()
            .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let at_ms = at.timestamp_millis();
        if stamps
            .get(key)
            .and_then(Value::as_i64)
            .is_some_and(|prev| prev >= at_ms)
        {
            return Ok(());
        }
        stamps[key] = json!(at_ms);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&stamps)?)?;
        std::fs::rename(&tmp, &path)
    }
}

impl StrataMemory {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, StorageError> {
        Self::open_with_policy(dir, strata_store::default_policy())
    }

    pub fn open_with_policy(
        dir: impl AsRef<Path>,
        policy: strata_gate::Policy,
    ) -> Result<Self, StorageError> {
        let data_dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;
        let store =
            strata_store::StrataStore::open_with_policy(&data_dir, policy).map_err(map_store)?;
        Ok(Self {
            log_dir: data_dir.join("log"),
            data_dir,
            store: Mutex::new(store),
            actor: Mutex::new(None),
            merge_policy: Mutex::new(None),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, strata_store::StrataStore> {
        self.store
            .lock()
            .unwrap_or_else(|err| panic!("strata memory lock poisoned: {err}"))
    }

    fn nodes(&self) -> Vec<strata_store::NodeRecord> {
        self.lock()
            .nodes()
            .into_iter()
            .filter(|record| record.is_live())
            .collect()
    }
}

/// What kind of stored record an audit row describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditKind {
    Memory,
    Intention,
}

/// One stored record with every text field it carries, for the credential
/// audit. Retired and suppressed memories are included: their bytes stay in
/// the append-only log.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    pub id: String,
    pub kind: AuditKind,
    pub retired: bool,
    pub created_at: DateTime<Utc>,
    pub texts: Vec<String>,
}

/// Every record of the open Strata log, for `scan-secrets`. `None` when
/// `storage` is not a Strata log opened in this process.
pub fn secret_audit_records(storage: &Storage) -> Option<Vec<AuditRecord>> {
    live_memory(storage).map(|memory| memory.audit_records())
}

impl StrataMemory {
    fn audit_records(&self) -> Vec<AuditRecord> {
        let store = self.lock();
        let mut rows: Vec<AuditRecord> = store
            .nodes()
            .iter()
            .map(|record| {
                let mut texts = vec![
                    record.content.clone(),
                    record.node_type.clone(),
                    record.scope.clone(),
                ];
                texts.extend(record.tags.iter().cloned());
                if let Some(source) = record.source.as_ref() {
                    texts.extend([
                        source.system.clone(),
                        source.project.clone(),
                        source.id.clone(),
                    ]);
                }
                AuditRecord {
                    id: record.id.clone(),
                    kind: AuditKind::Memory,
                    retired: !record.is_live(),
                    created_at: ms_to_dt(record.created_at_ms),
                    texts,
                }
            })
            .collect();
        rows.extend(store.intentions().iter().map(|record| {
            let mut texts = vec![record.content.clone(), record.trigger_data.clone()];
            texts.extend(record.notes.clone());
            texts.extend(record.source_data.clone());
            texts.extend(record.scope.clone());
            texts.extend(record.tags.iter().cloned());
            AuditRecord {
                id: record.id.clone(),
                kind: AuditKind::Intention,
                retired: false,
                created_at: ms_to_dt(record.created_at_ms),
                texts,
            }
        }));
        rows.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        rows
    }
}

fn map_store(err: strata_store::StoreError) -> StorageError {
    use strata_store::StoreError::*;
    match err {
        Denied { propose_seq } => {
            StorageError::Init(format!("gate_denied: propose_seq {propose_seq}"))
        }
        Held { propose_seq } => StorageError::Init(format!("gate_held: propose_seq {propose_seq}")),
        NotFound(msg) => StorageError::NotFound(msg),
        InvalidInput(msg) => StorageError::Init(msg),
        Io(err) => StorageError::Io(err),
        other => StorageError::Database(other.to_string()),
    }
}

fn similarity(op: &str) -> StorageError {
    StorageError::Init(format!(
        "similarity_disabled: {op}: embeddings, cosine, BM25, FTS, Jaccard, and keyword or name matching are not Strata operations; pass an exact handle"
    ))
}

/// Vestige 4.0 withholds erasure-class operations on a Strata log. The log
/// is append-only: retiring a memory hides it from every read but keeps its
/// bytes, so purge, delete and suppress would promise more than they do.
/// Real erasure is a later release. Shared by the store and the MCP server
/// so every surface (tools, dashboard, CLI, review approvals) refuses alike.
pub fn withheld_message(what: &str) -> String {
    format!(
        "unavailable_in_4_0: {what} is withheld on Strata in Vestige 4.0. The log is append-only, so it can hide a memory but cannot erase its bytes yet, and this call must not claim erasure. Use memory with action='demote' to lower a memory's retrieval strength."
    )
}

fn withheld(what: &str) -> StorageError {
    StorageError::Init(withheld_message(what))
}

fn pending(op: &str) -> StorageError {
    StorageError::Init(format!(
        "pending_strata: {op} is not admitted on the Strata log yet"
    ))
}

/// Reject a blank, huge, or control-character scope. Matching stays exact,
/// the same compare `node_is_in_scope` uses.
fn projection_scope(scope: &str) -> Result<&str, StorageError> {
    if scope.is_empty()
        || scope.len() > 200
        || scope.chars().any(char::is_control)
        || scope.trim().is_empty()
    {
        return Err(StorageError::InvalidScope(
            "expected a non-empty identifier of at most 200 visible characters".into(),
        ));
    }
    Ok(scope)
}

/// Decision and pattern always. Fact and note only with an exact durable tag.
fn projection_durable(record: &strata_store::NodeRecord) -> bool {
    matches!(record.node_type.as_str(), "decision" | "pattern")
        || (matches!(record.node_type.as_str(), "fact" | "note")
            && record.tags.iter().any(|tag| {
                matches!(
                    tag.to_ascii_lowercase().as_str(),
                    "rule" | "preference" | "convention"
                )
            }))
}

fn projection_rank(node_type: &str) -> u8 {
    match node_type {
        "decision" => 0,
        "pattern" => 1,
        _ => 2,
    }
}

fn sim_async(op: &str) -> MemoryStoreError {
    MemoryStoreError::Init(similarity(op).to_string())
}

fn pending_async(op: &str) -> MemoryStoreError {
    MemoryStoreError::Init(pending(op).to_string())
}

fn dt_ms(ms: i64) -> Result<DateTime<Utc>, StorageError> {
    DateTime::from_timestamp_millis(ms)
        .ok_or_else(|| StorageError::Init(format!("intention timestamp {ms} is out of range")))
}

fn dt_ms_opt(ms: Option<i64>) -> Result<Option<DateTime<Utc>>, StorageError> {
    ms.map(dt_ms).transpose()
}

fn stored_intention(
    intention: &vestige_core::storage::IntentionRecord,
) -> strata_store::IntentionRecord {
    strata_store::IntentionRecord {
        id: intention.id.clone(),
        content: intention.content.clone(),
        trigger_type: intention.trigger_type.clone(),
        trigger_data: intention.trigger_data.clone(),
        priority: intention.priority,
        status: intention.status.clone(),
        created_at_ms: intention.created_at.timestamp_millis(),
        deadline_ms: intention.deadline.map(|t| t.timestamp_millis()),
        fulfilled_at_ms: intention.fulfilled_at.map(|t| t.timestamp_millis()),
        reminder_count: intention.reminder_count,
        last_reminded_at_ms: intention.last_reminded_at.map(|t| t.timestamp_millis()),
        notes: intention.notes.clone(),
        tags: intention.tags.clone(),
        related_memories: intention.related_memories.clone(),
        snoozed_until_ms: intention.snoozed_until.map(|t| t.timestamp_millis()),
        source_type: intention.source_type.clone(),
        source_data: intention.source_data.clone(),
        scope: intention.scope.clone(),
    }
}

fn core_intention(
    record: &strata_store::IntentionRecord,
) -> Result<vestige_core::storage::IntentionRecord, StorageError> {
    Ok(vestige_core::storage::IntentionRecord {
        id: record.id.clone(),
        content: record.content.clone(),
        trigger_type: record.trigger_type.clone(),
        trigger_data: record.trigger_data.clone(),
        priority: record.priority,
        status: record.status.clone(),
        created_at: dt_ms(record.created_at_ms)?,
        deadline: dt_ms_opt(record.deadline_ms)?,
        fulfilled_at: dt_ms_opt(record.fulfilled_at_ms)?,
        reminder_count: record.reminder_count,
        last_reminded_at: dt_ms_opt(record.last_reminded_at_ms)?,
        notes: record.notes.clone(),
        tags: record.tags.clone(),
        related_memories: record.related_memories.clone(),
        snoozed_until: dt_ms_opt(record.snoozed_until_ms)?,
        source_type: record.source_type.clone(),
        source_data: record.source_data.clone(),
        scope: record.scope.clone(),
    })
}

/// A code anchor as the log stores it: integer times, verdict as its string.
fn stored_anchor(anchor: &CodeAnchor) -> strata_store::AnchorRecord {
    strata_store::AnchorRecord {
        id: anchor.id.clone(),
        node_id: anchor.node_id.clone(),
        file_path: anchor.file_path.clone(),
        symbol: anchor.symbol.clone(),
        symbol_kind: anchor.symbol_kind.clone(),
        start_line: anchor.start_line,
        end_line: anchor.end_line,
        span_lines: anchor.span_lines,
        content_hash: anchor.content_hash.clone(),
        captured_at_ms: anchor.captured_at.timestamp_millis(),
        last_verified_at_ms: anchor.last_verified_at.map(|at| at.timestamp_millis()),
        last_status: anchor.last_status.map(|status| status.as_str().to_string()),
    }
}

/// A stored anchor back as the tool surface reads it. An unknown verdict
/// string reads as never checked, never as stale.
fn core_anchor(record: &strata_store::AnchorRecord) -> CodeAnchor {
    CodeAnchor {
        id: record.id.clone(),
        node_id: record.node_id.clone(),
        file_path: record.file_path.clone(),
        symbol: record.symbol.clone(),
        symbol_kind: record.symbol_kind.clone(),
        start_line: record.start_line,
        end_line: record.end_line,
        span_lines: record.span_lines,
        content_hash: record.content_hash.clone(),
        captured_at: ms_to_dt(record.captured_at_ms),
        last_verified_at: record.last_verified_at_ms.map(ms_to_dt),
        last_status: record
            .last_status
            .as_deref()
            .and_then(AnchorStatus::parse_status),
    }
}

fn is_prospective(row: &vestige_core::storage::IntentionRecord) -> bool {
    row.source_type != crate::intention_graph_log::SOURCE
}

fn sort_intentions(
    mut rows: Vec<vestige_core::storage::IntentionRecord>,
) -> Vec<vestige_core::storage::IntentionRecord> {
    rows.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(a.created_at.cmp(&b.created_at))
            .then(a.id.cmp(&b.id))
    });
    rows
}

/// RETIRE `id` under an existing named rule, then RETIRE the successor anchor
/// so the anchor stays out of reads. No new op. The log keeps both records.
fn retire_live(
    store: &mut strata_store::StrataStore,
    id: &str,
    rule: &'static str,
    confirm: bool,
) -> Result<strata_store::RetireReceipt, StorageError> {
    let Some(record) = store.get_node(id) else {
        return Err(StorageError::NotFound(id.to_string()));
    };
    if !record.is_live() {
        return Err(StorageError::NotFound(format!("{id} is already retired")));
    }
    let scope = record.scope.clone();
    // SupersedeNode needs a successor. This anchor is retired in the same call.
    let anchor = store
        .ingest_in_scope(
            strata_store::IngestInput {
                content: ".".into(),
                source: None,
                source_updated_at_ms: None,
                node_type: "fact".into(),
                tags: Vec::new(),
                created_at_ms: Some(0),
                valid_from_ms: None,
                valid_until_ms: None,
            },
            &scope,
        )
        .map_err(map_store)?;
    let ctx = strata_store::AdmissionContext {
        rule_id: Some(rule.to_string()),
        confirm,
    };
    let receipt = store.retire(id, &anchor, &ctx).map_err(map_store)?;
    if receipt.rule_id != Some(rule) {
        return Err(StorageError::Init(format!(
            "{rule} retire was not admitted under {rule}"
        )));
    }
    store.retire(&anchor, id, &ctx).map_err(map_store)?;
    Ok(receipt)
}

fn is_mem_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("mem-") else {
        return false;
    };
    rest.len() == 16 && rest.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_memory_id(id: &str) -> bool {
    uuid::Uuid::parse_str(id).is_ok() || is_mem_id(id)
}

/// Namespaces are identifiers. Blank, oversized, and control-character values
/// are refused before a frame is appended, the same gate v3 applied on write.
fn normalize_scope(scope: &str) -> Result<&str, StorageError> {
    let normalized = scope.trim();
    if normalized.is_empty() || normalized.len() > 200 || normalized.chars().any(char::is_control) {
        return Err(StorageError::InvalidScope(
            "expected a non-empty identifier of at most 200 visible characters".into(),
        ));
    }
    Ok(normalized)
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or(DateTime::UNIX_EPOCH)
}

fn q32(q: i64) -> f64 {
    q as f64 / Q32_SCALE
}

/// A link a caller declares when it saves a memory, named from the new
/// memory's side. `supersedes` is caller-declarable on Strata (INGEST V5):
/// it records a full-replacement trail — for a correction that keeps the
/// old memory in play, use `corrects` via the review path instead. The
/// remaining provenance kinds stay closed: `touched` / `anchored_to` /
/// `projected_to` are recorded by the code and projection paths themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredLink {
    /// The new memory derives from the target: `new -derived_from-> target`.
    DerivedFrom,
    /// The new memory is evidence about the target: `new -evidence_of-> target`.
    EvidenceOf,
    /// The new memory closes the target: `target -closed_by-> new`.
    Closes,
    /// The new memory fully replaces the target:
    /// `new -supersedes-> target`. Caller-declared; unlike a review-gated
    /// `corrects` correction, the old memory is a full replacement, and the
    /// edge alone retires nothing — retiring stays a RETIRE.
    Supersedes,
}

impl DeclaredLink {
    pub const NAMES: [&'static str; 4] = ["derived_from", "evidence_of", "closes", "supersedes"];

    pub fn parse(kind: &str) -> Option<Self> {
        match kind {
            "derived_from" => Some(Self::DerivedFrom),
            "evidence_of" => Some(Self::EvidenceOf),
            "closes" => Some(Self::Closes),
            "supersedes" => Some(Self::Supersedes),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::EvidenceOf => "evidence_of",
            Self::Closes => "closes",
            Self::Supersedes => "supersedes",
        }
    }

    /// The recorded edge kind and its direction for a link from `new_id`.
    fn edge(self, new_id: &str, target: &str) -> (strata_store::EdgeKind, String, String) {
        match self {
            Self::DerivedFrom => (
                strata_store::EdgeKind::DerivedFrom,
                new_id.to_string(),
                target.to_string(),
            ),
            Self::EvidenceOf => (
                strata_store::EdgeKind::EvidenceOf,
                new_id.to_string(),
                target.to_string(),
            ),
            Self::Closes => (
                strata_store::EdgeKind::ClosedBy,
                target.to_string(),
                new_id.to_string(),
            ),
            // Caller target is the OLD memory: the edge reads
            // `new -supersedes-> old`, the same direction `recorded_origin`
            // already walks for typed supersession hops.
            Self::Supersedes => (
                strata_store::EdgeKind::Supersedes,
                new_id.to_string(),
                target.to_string(),
            ),
        }
    }
}

/// Check declared links before anything is written: each target is a live
/// memory of this Strata log, in `scope`, and no link is repeated.
pub fn check_links(
    storage: &Storage,
    scope: &str,
    links: &[(DeclaredLink, String)],
) -> Result<(), String> {
    let memory = live_memory(storage)
        .ok_or_else(|| "links need a Strata log open in this process".to_string())?;
    let store = memory.lock();
    let mut seen = BTreeSet::new();
    for (kind, target) in links {
        if !seen.insert((kind.as_str(), target.as_str())) {
            return Err(format!(
                "link {} -> {target} is listed twice",
                kind.as_str()
            ));
        }
        let Some(record) = store.get_node(target) else {
            return Err(format!(
                "link target {target} is not a memory in this store"
            ));
        };
        if !retrievable(&record) {
            return Err(format!("link target {target} is retired"));
        }
        if record.scope != scope {
            return Err(format!(
                "link target {target} is in scope '{}', not '{scope}'; links stay within one scope",
                record.scope
            ));
        }
    }
    Ok(())
}

/// Write declared links from the memory `new_id`, each through the gate with
/// its own receipt. Call [`check_links`] first; a link that is refused here is
/// reported with the receipts of the links already written.
pub fn save_links(
    storage: &Storage,
    new_id: &str,
    links: &[(DeclaredLink, String)],
) -> Result<Vec<Value>, String> {
    let memory = live_memory(storage)
        .ok_or_else(|| "links need a Strata log open in this process".to_string())?;
    let mut store = memory.lock();
    let now = Utc::now().timestamp_millis();
    let mut written = Vec::with_capacity(links.len());
    for (kind, target) in links {
        if target == new_id {
            return Err(format!("{new_id} cannot link to itself"));
        }
        let (edge_kind, source_id, target_id) = kind.edge(new_id, target);
        let edge = strata_store::ConnectionRecord {
            source_id: source_id.clone(),
            target_id: target_id.clone(),
            strength_milli: 1000,
            link_type: edge_kind.as_str().to_string(),
            meta_sha: None,
            created_at_ms: now,
            activation_count: 0,
        };
        match store.save_connection(&edge) {
            Ok(effect) => written.push(json!({
                "kind": kind.as_str(),
                "edge": edge_kind.as_str(),
                "source": source_id,
                "target": target_id,
                "receiptId": receipt_id_for(effect),
            })),
            Err(err) => {
                return Err(format!(
                    "{new_id} was saved, but its {} link to {target} was not admitted: {}; links written before it: {}",
                    kind.as_str(),
                    map_store(err),
                    Value::Array(written)
                ));
            }
        }
    }
    Ok(written)
}

fn receipt_id_for(seq: u64) -> String {
    format!("{RECEIPT_PREFIX}{seq:016x}")
}

fn parse_receipt_seq(receipt_id: &str) -> Option<u64> {
    let rest = receipt_id.strip_prefix(RECEIPT_PREFIX)?;
    u64::from_str_radix(rest, 16).ok()
}

fn resolve_proof(
    store: &strata_store::StrataStore,
    receipt_or_node: &str,
) -> Result<Option<strata_store::EffectProof>, strata_store::StoreError> {
    if let Some(seq) = parse_receipt_seq(receipt_or_node) {
        return store.effect_by_seq(seq);
    }
    store.latest_effect(receipt_or_node)
}

/// An edge receipt names both endpoints. When either was retired, the
/// receipt stays hidden like the retired memory itself.
fn edge_proof_hidden(store: &strata_store::StrataStore, proof: &strata_store::EffectProof) -> bool {
    let Some((target, _)) = &proof.edge else {
        return false;
    };
    [proof.node_id.as_str(), target.as_str()].iter().any(|id| {
        store
            .get_node(id)
            .is_some_and(|record| !retrievable(&record))
    })
}

fn mutation_kind(proof: &strata_store::EffectProof) -> &'static str {
    match (proof.action, proof.rating) {
        (strata_store::EffectAction::Create, _) => "created",
        (strata_store::EffectAction::Rewrite, _) => "rewritten",
        (strata_store::EffectAction::Edit, _) => "edited",
        (strata_store::EffectAction::Review, Some(PROMOTE_RATING)) => "promoted",
        (strata_store::EffectAction::Review, Some(DEMOTE_RATING)) => "demoted",
        (strata_store::EffectAction::Review, _) => "reviewed",
        (strata_store::EffectAction::Intention, _) => "intention_upserted",
        (strata_store::EffectAction::Anchor, _) => "anchor_recorded",
        (strata_store::EffectAction::AnchorVerdict, _) => "anchor_verified",
        (strata_store::EffectAction::Edge, _) => "edge_recorded",
    }
}

fn receipt_from_proof(proof: &strata_store::EffectProof, trust: f64) -> Receipt {
    let mut note = format!(
        "digest={} event_seq={} effect_seq={}",
        hex32(&proof.payload_digest),
        proof.data_seq,
        proof.effect_seq
    );
    if let Some(rating) = proof.rating {
        note.push_str(&format!(" rating={rating}"));
    }
    if proof.action == strata_store::EffectAction::Edit {
        note.push_str(" rule=edit");
    }
    if let Some((target, kind)) = &proof.edge {
        note.push_str(&format!(" edge={kind} target={target}"));
    }
    Receipt {
        receipt_id: receipt_id_for(proof.effect_seq),
        retrieved: vec![proof.node_id.clone()],
        suppressed: Vec::new(),
        activation_path: Vec::new(),
        trust_floor: trust,
        decay_risk: DecayRisk::from_trust_floor(trust),
        mutations: vec![vestige_core::ReceiptMutation {
            id: proof.node_id.clone(),
            kind: mutation_kind(proof).to_string(),
            note: Some(note),
        }],
        evidence: None,
        actor: None,
    }
}

fn blocking_secrets(text: &str) -> Vec<String> {
    scan_secrets(text)
        .into_iter()
        .filter(|finding| finding.blocks_ingestion())
        .map(|finding| finding.kind.to_string())
        .collect()
}

/// Credential kinds found across `texts`, each kind once, in first-seen order.
pub(crate) fn blocking_secrets_in<'a>(texts: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut kinds: Vec<String> = Vec::new();
    for text in texts {
        for kind in blocking_secrets(text) {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
    }
    kinds
}

/// Every text field an ingest stores: content, type, tags and provenance.
fn stored_ingest_texts(input: &IngestInput) -> Vec<&str> {
    let mut texts = vec![input.content.as_str(), input.node_type.as_str()];
    texts.extend(input.tags.iter().map(String::as_str));
    texts.extend(input.source.as_deref());
    if let Some(envelope) = input.source_envelope.as_ref() {
        texts.extend(envelope.source_system.as_deref());
        texts.extend(envelope.source_id.as_deref());
        texts.extend(envelope.source_project.as_deref());
    }
    texts
}

/// The credential gate for one ingest. The scope is checked under every
/// policy: it names a namespace that reads and write responses repeat, so the
/// explicit override covers the memory's own text, never the scope.
fn gate_ingest(input: &IngestInput, scope: &str, policy: SecretPolicy) -> Result<(), StorageError> {
    let mut texts = vec![scope];
    if policy != SecretPolicy::AllowExplicitly {
        texts.extend(stored_ingest_texts(input));
    }
    let kinds = blocking_secrets_in(texts);
    if kinds.is_empty() {
        Ok(())
    } else {
        Err(StorageError::SecretDetected { kinds })
    }
}

/// The credential gate for one intention: every text field it stores.
fn gate_intention(intention: &vestige_core::storage::IntentionRecord) -> Result<(), StorageError> {
    let mut texts = vec![intention.content.as_str(), intention.trigger_data.as_str()];
    texts.extend(intention.notes.as_deref());
    texts.extend(intention.source_data.as_deref());
    texts.extend(intention.scope.as_deref());
    texts.extend(intention.tags.iter().map(String::as_str));
    let kinds = blocking_secrets_in(texts);
    if kinds.is_empty() {
        Ok(())
    } else {
        Err(StorageError::SecretDetected { kinds })
    }
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|text| !text.is_empty())
}

/// Map a vestige ingest onto the store provenance fields.
///
/// A connector envelope with `(source_system, source_id)` becomes a
/// [`strata_store::SourceKey`]. A free-form `source` string with no key
/// becomes `system = <text>`. `source_updated_at` is the timestamp. Both are
/// `None` when the input has neither.
fn store_provenance(input: &IngestInput) -> (Option<strata_store::SourceKey>, Option<i64>) {
    let envelope = input.source_envelope.as_ref();
    let updated =
        envelope.and_then(|env| env.source_updated_at.map(|time| time.timestamp_millis()));
    if let Some(env) = envelope
        && let (Some(system), Some(id)) = (
            nonempty(env.source_system.as_deref()),
            nonempty(env.source_id.as_deref()),
        )
    {
        return (
            Some(strata_store::SourceKey {
                system: system.to_string(),
                project: env.source_project.clone().unwrap_or_default(),
                id: id.to_string(),
            }),
            updated,
        );
    }
    if let Some(label) = nonempty(input.source.as_deref()) {
        return (
            Some(strata_store::SourceKey {
                system: label.to_string(),
                project: String::new(),
                id: String::new(),
            }),
            updated,
        );
    }
    (None, updated)
}

fn to_store_input(input: &IngestInput) -> strata_store::IngestInput {
    let (source, source_updated_at_ms) = store_provenance(input);
    strata_store::IngestInput {
        content: input.content.clone(),
        source,
        source_updated_at_ms,
        node_type: input.node_type.clone(),
        tags: input.tags.clone(),
        created_at_ms: Some(Utc::now().timestamp_millis()),
        valid_from_ms: input.valid_from.map(|t| t.timestamp_millis()),
        valid_until_ms: input.valid_until.map(|t| t.timestamp_millis()),
    }
}

fn project_node(
    store: &strata_store::StrataStore,
    record: &strata_store::NodeRecord,
) -> KnowledgeNode {
    let card = store.card_state(&record.id);
    let retrieval = store
        .retrievability(&record.id)
        .ok()
        .flatten()
        .unwrap_or(0.0);
    let mut node = KnowledgeNode::default();
    node.id = record.id.clone();
    node.content = record.content.clone();
    node.node_type = record.node_type.clone();
    node.created_at = ms_to_dt(record.created_at_ms);
    node.updated_at = node.created_at;
    node.last_accessed = node.created_at;
    node.tags = record.tags.clone();
    node.valid_from = Some(ms_to_dt(record.valid_from_ms));
    node.valid_until =
        (record.valid_until_ms != VALID_FOREVER_MS).then(|| ms_to_dt(record.valid_until_ms));
    // Kernel retrievability is the only strength the log can justify.
    node.stability = card.as_ref().map(|c| q32(c.stability_q)).unwrap_or(0.0);
    node.difficulty = card.as_ref().map(|c| q32(c.difficulty_q)).unwrap_or(0.0);
    node.reps = card.as_ref().map(|c| c.review_count as i32).unwrap_or(0);
    node.lapses = card.as_ref().map(|c| c.lapse_count as i32).unwrap_or(0);
    node.storage_strength = retrieval;
    node.retrieval_strength = retrieval;
    node.retention_strength = retrieval;
    node.has_embedding = Some(false);
    apply_store_provenance(&mut node, record);
    node
}

fn envelope_with(
    system: Option<String>,
    project: Option<String>,
    id: Option<String>,
    updated: Option<DateTime<Utc>>,
) -> SourceEnvelope {
    let mut envelope = SourceEnvelope::default();
    envelope.source_system = system;
    envelope.source_project = project;
    envelope.source_id = id;
    envelope.source_updated_at = updated;
    envelope
}

fn apply_store_provenance(node: &mut KnowledgeNode, record: &strata_store::NodeRecord) {
    let updated = record
        .source_updated_at_ms
        .and_then(DateTime::from_timestamp_millis);
    let Some(key) = &record.source else {
        if updated.is_some() {
            node.source_envelope = Some(envelope_with(None, None, None, updated));
        }
        return;
    };
    if key.id.is_empty() && key.project.is_empty() {
        node.source = Some(key.system.clone());
        if updated.is_some() {
            node.source_envelope = Some(envelope_with(None, None, None, updated));
        }
        return;
    }
    node.source_envelope = Some(envelope_with(
        Some(key.system.clone()),
        (!key.project.is_empty()).then(|| key.project.clone()),
        (!key.id.is_empty()).then(|| key.id.clone()),
        updated,
    ));
}

fn retrievable(record: &strata_store::NodeRecord) -> bool {
    record.superseded_by.is_none()
}

/// A node-id lookup of a missing or retired node. `eff-` ids and intention ids
/// still resolve.
fn node_lookup_hidden(store: &strata_store::StrataStore, key: &str) -> bool {
    parse_receipt_seq(key).is_none()
        && store.get_intention(key).is_none()
        && !store.get_node(key).as_ref().is_some_and(retrievable)
}

fn endpoint_retired(store: &strata_store::StrataStore, id: &str) -> bool {
    store
        .get_node(id)
        .is_some_and(|record| record.superseded_by.is_some())
}

fn edge_visible(store: &strata_store::StrataStore, edge: &strata_store::ConnectionRecord) -> bool {
    !endpoint_retired(store, &edge.source_id) && !endpoint_retired(store, &edge.target_id)
}

fn project_edge(edge: &strata_store::ConnectionRecord) -> VestigeEdge {
    let created = ms_to_dt(edge.created_at_ms);
    VestigeEdge {
        source_id: edge.source_id.clone(),
        target_id: edge.target_id.clone(),
        strength: edge.strength_milli as f64 / 1000.0,
        link_type: edge.link_type.clone(),
        created_at: created,
        last_activated: created,
        activation_count: i32::try_from(edge.activation_count).unwrap_or(i32::MAX),
    }
}

fn to_strata_edge(edge: &VestigeEdge) -> strata_store::ConnectionRecord {
    strata_store::ConnectionRecord {
        source_id: edge.source_id.clone(),
        target_id: edge.target_id.clone(),
        strength_milli: (edge.strength * 1000.0).round() as i64,
        link_type: edge.link_type.clone(),
        meta_sha: None,
        created_at_ms: edge.created_at.timestamp_millis(),
        activation_count: i64::from(edge.activation_count),
    }
}

fn page_bounds(limit: usize, after: Option<&str>, budget_ms: u64) -> Result<(), StorageError> {
    if !(1..=1000).contains(&limit) || !(1..=10_000).contains(&budget_ms) {
        return Err(StorageError::Init(
            "batchSize must be 1..1000 and budgetMs 1..10000".into(),
        ));
    }
    if after.is_some_and(|id| !is_memory_id(id)) {
        return Err(StorageError::Init("after must be a memory id".into()));
    }
    Ok(())
}

fn retention_bucket(value: f64) -> &'static str {
    if value < 0.0 {
        "below0"
    } else if value < 0.2 {
        "0-20%"
    } else if value < 0.4 {
        "20-40%"
    } else if value < 0.6 {
        "40-60%"
    } else if value < 0.8 {
        "60-80%"
    } else if value <= 1.0 {
        "80-100%"
    } else {
        "above100%"
    }
}

fn in_scope<'a>(
    nodes: &'a [strata_store::NodeRecord],
    scope: Option<&str>,
) -> Vec<&'a strata_store::NodeRecord> {
    nodes
        .iter()
        .filter(|node| scope.is_none_or(|scope| node.scope == scope))
        .collect()
}

impl MemoryStoreSend for StrataMemory {
    async fn init(&self) -> MemoryStoreResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> MemoryStoreResult<HealthStatus> {
        let _guard = self.lock();
        Ok(HealthStatus::Healthy)
    }

    async fn registered_model(&self) -> MemoryStoreResult<Option<ModelSignature>> {
        Ok(None)
    }

    async fn register_model(&self, _sig: &ModelSignature) -> MemoryStoreResult<()> {
        Err(pending_async("register_model"))
    }

    async fn insert(&self, _record: &MemoryRecord) -> MemoryStoreResult<uuid::Uuid> {
        Err(pending_async("insert"))
    }

    async fn get(&self, _id: uuid::Uuid) -> MemoryStoreResult<Option<MemoryRecord>> {
        Err(pending_async("get"))
    }

    async fn update(&self, _record: &MemoryRecord) -> MemoryStoreResult<()> {
        Err(pending_async("update"))
    }

    async fn delete(&self, _id: uuid::Uuid) -> MemoryStoreResult<()> {
        Err(pending_async("delete"))
    }

    async fn search_records(
        &self,
        _query: &SearchQuery,
    ) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(sim_async("search_records"))
    }

    async fn fts_search(
        &self,
        _text: &str,
        _limit: usize,
    ) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(sim_async("fts_search"))
    }

    async fn vector_search(
        &self,
        _embedding: &[f32],
        _limit: usize,
    ) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(sim_async("vector_search"))
    }

    async fn get_scheduling(
        &self,
        _memory_id: uuid::Uuid,
    ) -> MemoryStoreResult<Option<SchedulingState>> {
        Err(pending_async("get_scheduling"))
    }

    async fn update_scheduling(&self, _state: &SchedulingState) -> MemoryStoreResult<()> {
        Err(pending_async("update_scheduling"))
    }

    async fn get_due_memories(
        &self,
        _before: DateTime<Utc>,
        _limit: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, SchedulingState)>> {
        Err(pending_async("get_due_memories"))
    }

    async fn add_edge(&self, _edge: &MemoryEdge) -> MemoryStoreResult<()> {
        Err(pending_async("add_edge"))
    }

    async fn get_edges(
        &self,
        _node_id: uuid::Uuid,
        _edge_type: Option<&str>,
    ) -> MemoryStoreResult<Vec<MemoryEdge>> {
        Err(pending_async("get_edges"))
    }

    async fn remove_edge(&self, _source: uuid::Uuid, _target: uuid::Uuid) -> MemoryStoreResult<()> {
        Err(pending_async("remove_edge"))
    }

    async fn get_neighbors(
        &self,
        _node_id: uuid::Uuid,
        _depth: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, f64)>> {
        Err(sim_async("get_neighbors"))
    }

    async fn list_domains(&self) -> MemoryStoreResult<Vec<vestige_core::storage::Domain>> {
        Ok(Vec::new())
    }

    async fn get_domain(
        &self,
        _id: &str,
    ) -> MemoryStoreResult<Option<vestige_core::storage::Domain>> {
        Ok(None)
    }

    async fn upsert_domain(
        &self,
        _domain: &vestige_core::storage::Domain,
    ) -> MemoryStoreResult<()> {
        Err(pending_async("upsert_domain"))
    }

    async fn delete_domain(&self, _id: &str) -> MemoryStoreResult<()> {
        Err(pending_async("delete_domain"))
    }

    async fn classify(&self, _embedding: &[f32]) -> MemoryStoreResult<Vec<(String, f64)>> {
        Err(sim_async("classify"))
    }

    async fn count(&self) -> MemoryStoreResult<usize> {
        Ok(self.lock().node_count())
    }

    async fn get_store_stats(&self) -> MemoryStoreResult<StoreStats> {
        let store = self.lock();
        Ok(StoreStats {
            total_memories: store.node_count(),
            memories_with_embeddings: 0,
            total_edges: store.edge_count(),
            total_domains: 0,
            registered_model_name: None,
            registered_model_dim: None,
        })
    }

    async fn vacuum(&self) -> MemoryStoreResult<()> {
        Ok(())
    }

    fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    fn db_path(&self) -> &Path {
        &self.log_dir
    }

    fn is_strata(&self) -> bool {
        true
    }

    fn is_strata_log(&self) -> bool {
        true
    }

    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        self.maintenance_stamp("lastBackupMs")
    }

    fn get_last_dream(&self) -> Result<Option<DateTime<Utc>>, StorageError> {
        Ok(self.maintenance_stamp("lastDreamMs"))
    }

    fn save_dream_history(
        &self,
        record: &vestige_core::DreamHistoryRecord,
    ) -> Result<i64, StorageError> {
        self.record_maintenance_stamp("lastDreamMs", record.dreamed_at)
            .map_err(|err| StorageError::Init(format!("dream stamp not recorded: {err}")))?;
        Ok(0)
    }

    fn count_memories_since(&self, since: DateTime<Utc>) -> Result<i64, StorageError> {
        let since_ms = since.timestamp_millis();
        let store = self.lock();
        Ok(store
            .nodes()
            .into_iter()
            .filter(retrievable)
            .filter(|record| record.created_at_ms > since_ms)
            .count() as i64)
    }

    fn process_actor_did(&self) -> Option<String> {
        self.actor
            .lock()
            .unwrap_or_else(|err| panic!("strata actor lock poisoned: {err}"))
            .clone()
    }

    fn set_process_actor(&self, did: &str) -> Result<(), StorageError> {
        *self
            .actor
            .lock()
            .unwrap_or_else(|err| panic!("strata actor lock poisoned: {err}")) =
            Some(did.to_string());
        Ok(())
    }

    fn get_merge_policy(&self) -> Result<vestige_core::MergePolicy, StorageError> {
        let mut slot = self
            .merge_policy
            .lock()
            .unwrap_or_else(|err| panic!("strata merge policy lock poisoned: {err}"));
        if let Some(policy) = *slot {
            return Ok(policy);
        }
        if let Some(policy) = read_merge_policy_file(&self.data_dir)? {
            *slot = Some(policy);
            return Ok(policy);
        }
        Ok(merge_policy_from_env())
    }

    fn set_merge_policy(&self, policy: vestige_core::MergePolicy) -> Result<(), StorageError> {
        let policy = vestige_core::MergePolicy::new(
            policy.match_threshold,
            policy.possible_threshold,
            policy.auto_apply,
        );
        write_merge_policy_file(&self.data_dir, &policy)?;
        *self
            .merge_policy
            .lock()
            .unwrap_or_else(|err| panic!("strata merge policy lock poisoned: {err}")) =
            Some(policy);
        Ok(())
    }

    fn sidecar_dir(&self, name: &str) -> PathBuf {
        let path = self.data_dir.join(name);
        let _ = std::fs::create_dir_all(&path);
        path
    }

    fn resolve_handle(&self, query: &str) -> HandleResolution {
        let query = query.trim();
        let store = self.lock();
        // Every live node, native or imported from v3: origins only list
        // natively admitted writes, so an upgraded store's memories were
        // unreachable by handle. Retired nodes stay out.
        let ids: Vec<String> = store.node_ids_where(retrievable);
        // Exact tag handle (case-sensitive, like the SQLite resolver): every
        // live node carrying exactly this tag. No prefix or fuzzy matching.
        let tagged: Vec<String> = if query.is_empty() {
            Vec::new()
        } else {
            store.node_ids_where(|record| {
                retrievable(record) && record.tags.iter().any(|tag| tag == query)
            })
        };
        drop(store);
        if query.is_empty() {
            return HandleResolution {
                kind: HandleKind::Unknown,
                ids: Vec::new(),
                exact: false,
                candidates: Vec::new(),
                handle_required: Some(HANDLE_REQUIRED_DETAIL.to_string()),
            };
        }
        if let Some(id) = ids.iter().find(|id| id.as_str() == query) {
            return HandleResolution {
                kind: HandleKind::Memory,
                ids: vec![id.clone()],
                exact: true,
                candidates: Vec::new(),
                handle_required: None,
            };
        }
        if query.len() >= 8 {
            let hits: Vec<&String> = ids.iter().filter(|id| id.starts_with(query)).collect();
            if hits.len() == 1 {
                return HandleResolution {
                    kind: HandleKind::Memory,
                    ids: vec![hits[0].clone()],
                    exact: false,
                    candidates: Vec::new(),
                    handle_required: None,
                };
            }
            if hits.len() > 1 {
                return HandleResolution {
                    kind: HandleKind::Memory,
                    ids: Vec::new(),
                    exact: false,
                    candidates: hits
                        .into_iter()
                        .take(MAX_CANDIDATES)
                        .map(|id| (id.clone(), HandleKind::Memory))
                        .collect(),
                    handle_required: None,
                };
            }
        }
        if !tagged.is_empty() {
            return HandleResolution {
                kind: HandleKind::Tag,
                ids: tagged,
                exact: true,
                candidates: Vec::new(),
                handle_required: None,
            };
        }
        HandleResolution {
            kind: HandleKind::Unknown,
            ids: Vec::new(),
            exact: false,
            candidates: Vec::new(),
            handle_required: Some(HANDLE_REQUIRED_DETAIL.to_string()),
        }
    }

    fn retire_affected(
        &self,
        ids: &[&str],
        _reason: &str,
    ) -> Vec<vestige_core::storage::RetireOutcome> {
        ids.iter()
            .map(|id| vestige_core::storage::RetireOutcome {
                id: (*id).to_string(),
                suppressed: false,
                suppression_count: 0,
                error: Some(pending("retire_affected").to_string()),
            })
            .collect()
    }

    fn backup_to(&self, path: &Path) -> Result<(), StorageError> {
        self.lock().backup_to(path).map_err(map_store)?;
        // The backup is complete; a stamp that fails to write only leaves
        // `needsBackup` set, it never fails the backup.
        let _ = self.record_maintenance_stamp("lastBackupMs", Utc::now());
        Ok(())
    }

    fn checkpoint_wal(
        &self,
        _mode: WalCheckpointMode,
    ) -> Result<WalCheckpointStatus, StorageError> {
        Ok(WalCheckpointStatus {
            busy: 0,
            log_frames: 0,
            checkpointed_frames: 0,
        })
    }

    fn prune_agent_traces(&self) -> Result<i64, StorageError> {
        Ok(0)
    }

    fn get_last_consolidation(&self) -> Result<Option<DateTime<Utc>>, StorageError> {
        Ok(None)
    }

    fn run_consolidation(&self) -> Result<ConsolidationResult, StorageError> {
        Ok(ConsolidationResult::default())
    }

    fn ingest(&self, input: IngestInput) -> Result<KnowledgeNode, StorageError> {
        self.ingest_in_scope_with_secret_policy(
            input,
            vestige_core::DEFAULT_MEMORY_SCOPE,
            SecretPolicy::Reject,
        )
    }

    fn ingest_in_scope(
        &self,
        input: IngestInput,
        scope: &str,
    ) -> Result<KnowledgeNode, StorageError> {
        self.ingest_in_scope_with_secret_policy(input, scope, SecretPolicy::Reject)
    }

    fn ingest_with_secret_policy(
        &self,
        input: IngestInput,
        policy: SecretPolicy,
    ) -> Result<KnowledgeNode, StorageError> {
        self.ingest_in_scope_with_secret_policy(input, vestige_core::DEFAULT_MEMORY_SCOPE, policy)
    }

    fn ingest_in_scope_with_secret_policy(
        &self,
        input: IngestInput,
        scope: &str,
        policy: SecretPolicy,
    ) -> Result<KnowledgeNode, StorageError> {
        gate_ingest(&input, scope, policy)?;
        if input.content.trim().is_empty() {
            return Err(StorageError::Init("content must not be empty".into()));
        }
        let scope = normalize_scope(scope)?;
        let mut store = self.lock();
        let id = store
            .ingest_in_scope(to_store_input(&input), scope)
            .map_err(map_store)?;
        let record = store
            .get_node(&id)
            .ok_or_else(|| StorageError::NotFound(id.clone()))?;
        Ok(project_node(&store, &record))
    }

    fn find_duplicate_by_canonical_hash(
        &self,
        scope: &str,
        content: &str,
    ) -> Result<Option<String>, StorageError> {
        let store = self.lock();
        store
            .find_node_by_canonical_hash(scope, &strata_store::canonical_hash(content))
            .map_err(map_store)
    }

    fn find_intent_record(
        &self,
        scope: &str,
        intent_id: &str,
    ) -> Result<Option<(String, u64, String)>, StorageError> {
        self.lock()
            .find_intent(scope, intent_id)
            .map_err(map_store)
    }

    fn record_intent_entry(
        &self,
        scope: &str,
        intent_id: &str,
        node_id: &str,
        effect_seq: u64,
        response_digest: &str,
    ) -> Result<(), StorageError> {
        self.lock()
            .record_intent(scope, intent_id, node_id, effect_seq, response_digest)
            .map_err(map_store)
    }

    /// Latest `eff-` receipt naming `node_id`, via the same effect-index
    /// resolution `get_receipt` uses. `None` when no admitted effect names it.
    fn latest_receipt_id_for_node(&self, node_id: &str) -> Option<String> {
        let store = self.lock();
        resolve_proof(&store, node_id)
            .ok()
            .flatten()
            .map(|proof| receipt_id_for(proof.effect_seq))
    }

    fn node_effect_seq(&self, node_id: &str) -> Option<u64> {
        let store = self.lock();
        resolve_proof(&store, node_id)
            .ok()
            .flatten()
            .map(|proof| proof.effect_seq)
    }

    fn get_node(&self, id: &str) -> Result<Option<KnowledgeNode>, StorageError> {
        let store = self.lock();
        Ok(store
            .get_node(id)
            .filter(retrievable)
            .as_ref()
            .map(|record| project_node(&store, record)))
    }

    fn get_all_nodes(&self, limit: i32, offset: i32) -> Result<Vec<KnowledgeNode>, StorageError> {
        self.page_nodes(None, limit, offset)
    }

    fn get_all_nodes_in_scope(
        &self,
        scope: &str,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        self.page_nodes(Some(scope), limit, offset)
    }

    fn node_is_in_scope(&self, id: &str, scope: &str) -> Result<bool, StorageError> {
        Ok(self
            .lock()
            .get_node(id)
            .is_some_and(|record| retrievable(&record) && record.scope == scope))
    }

    fn get_connections_for_memory(
        &self,
        memory_id: &str,
    ) -> Result<Vec<VestigeEdge>, StorageError> {
        let store = self.lock();
        Ok(store
            .get_connections_for_memory(memory_id)
            .iter()
            .filter(|edge| edge_visible(&store, edge))
            .map(project_edge)
            .collect())
    }

    fn get_all_connections(&self) -> Result<Vec<VestigeEdge>, StorageError> {
        let store = self.lock();
        Ok(store
            .edges()
            .iter()
            .filter(|edge| edge_visible(&store, edge))
            .map(project_edge)
            .collect())
    }

    fn save_connection(&self, connection: &VestigeEdge) -> Result<(), StorageError> {
        let edge = to_strata_edge(connection);
        self.lock()
            .save_connection(&edge)
            .map_err(map_store)
            .map(|_| ())
    }

    fn admit_projection(
        &self,
        memory_ids: &[String],
        target: &str,
        region: &[u8],
    ) -> Result<(String, String), StorageError> {
        let hash = blake3::hash(region).to_hex().to_string();
        let mut store = self.lock();
        for id in memory_ids {
            if store.get_node(id).is_none() {
                return Err(StorageError::NotFound(id.clone()));
            }
        }
        let mut sources: Vec<String> = memory_ids.to_vec();
        if sources.is_empty() {
            // No memory to point from. A projection record is the source of
            // the one projected_to edge. It is not a decision, pattern, or
            // durable tag, so a later preview does not select it.
            let id = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: target.to_string(),
                        source: None,
                        source_updated_at_ms: None,
                        node_type: "projection".into(),
                        tags: Vec::new(),
                        created_at_ms: Some(0),
                        valid_from_ms: None,
                        valid_until_ms: None,
                    },
                    "projection",
                )
                .map_err(map_store)?;
            sources.push(id);
        }
        let mut effect_seq = 0u64;
        for source in &sources {
            let edge = strata_store::ConnectionRecord {
                source_id: source.clone(),
                target_id: target.to_string(),
                strength_milli: 1000,
                link_type: strata_store::EdgeKind::ProjectedTo.as_str().to_string(),
                meta_sha: Some(hash.clone()),
                created_at_ms: 0,
                activation_count: 0,
            };
            effect_seq = store.save_connection(&edge).map_err(map_store)?;
        }
        Ok((receipt_id_for(effect_seq), hash))
    }

    fn maintenance_memory_page(
        &self,
        limit: usize,
        after: Option<&str>,
        scope: &str,
    ) -> Result<(Vec<KnowledgeNode>, bool), StorageError> {
        page_bounds(limit, after, 10_000)?;
        let store = self.lock();
        let mut records: Vec<strata_store::NodeRecord> = store
            .nodes()
            .into_iter()
            .filter(|record| record.scope == scope && record.is_live())
            .filter(|record| after.is_none_or(|cursor| record.id.as_str() > cursor))
            .collect();
        records.sort_by(|left, right| left.id.cmp(&right.id));
        let has_more = records.len() > limit;
        records.truncate(limit);
        let nodes = records
            .iter()
            .map(|record| project_node(&store, record))
            .collect();
        Ok((nodes, has_more))
    }

    fn mark_reviewed(&self, id: &str, rating: Rating) -> Result<KnowledgeNode, StorageError> {
        let mut store = self.lock();
        let score = u8::try_from(rating.as_i32())
            .map_err(|_| StorageError::Init("rating must be 1..=4".into()))?;
        store.review(id, score).map_err(map_store)?;
        let record = store
            .get_node(id)
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
        Ok(project_node(&store, &record))
    }

    fn superseded_node_ids(&self) -> Result<HashSet<String>, StorageError> {
        Ok(self
            .lock()
            .supersession_pairs()
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    /// Recorded-edge subgraph for `graph` action `memory_graph`.
    ///
    /// Walks only edges the log admitted. Superseded nodes are omitted when
    /// that trail exists, so a retired record cannot stay in the picture or
    /// bridge two live ones. Node order is BFS by hop, each hop sorted by id;
    /// edges are sorted by source, target, link type, then creation time.
    /// The live memory with the most recorded links: a count of link records
    /// on the log, never a similarity. Only links whose two ends are both
    /// live count, the same test the subgraph walk applies, so the answer is
    /// always a center that walk can open. `None` when no memory has a link.
    /// Ties go to the newest memory, then to the smallest id.
    fn get_most_connected_memory(&self) -> Result<Option<String>, StorageError> {
        let store = self.lock();
        let superseded: HashSet<String> = store
            .supersession_pairs()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        // id -> (links, created_at_ms); `None` marks an end that is not live.
        let mut seen: BTreeMap<String, Option<(usize, i64)>> = BTreeMap::new();
        let mut live = |id: &str| -> bool {
            if let Some(entry) = seen.get(id) {
                return entry.is_some();
            }
            let entry = store
                .get_node(id)
                .filter(|record| record.is_live() && !superseded.contains(id))
                .map(|record| (0, record.created_at_ms));
            let is_live = entry.is_some();
            seen.insert(id.to_string(), entry);
            is_live
        };
        let mut linked = Vec::new();
        for edge in store.edges().iter() {
            if edge.source_id != edge.target_id && live(&edge.source_id) && live(&edge.target_id) {
                linked.push((edge.source_id.clone(), edge.target_id.clone()));
            }
        }
        for (source, target) in linked {
            for id in [source, target] {
                if let Some(Some((links, _))) = seen.get_mut(&id) {
                    *links += 1;
                }
            }
        }
        Ok(seen
            .into_iter()
            .filter_map(|(id, entry)| entry.filter(|(links, _)| *links > 0).map(|e| (id, e)))
            .max_by(
                |(a_id, (a_links, a_created)), (b_id, (b_links, b_created))| {
                    a_links
                        .cmp(b_links)
                        .then(a_created.cmp(b_created))
                        .then_with(|| b_id.cmp(a_id))
                },
            )
            .map(|(id, _)| id))
    }

    fn get_memory_subgraph(
        &self,
        center_id: &str,
        depth: u32,
        max_nodes: usize,
    ) -> Result<(Vec<KnowledgeNode>, Vec<VestigeEdge>), StorageError> {
        let store = self.lock();
        let superseded: HashSet<String> = store
            .supersession_pairs()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let live = |id: &str| {
            !superseded.contains(id) && store.get_node(id).is_some_and(|record| record.is_live())
        };
        if !live(center_id) {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut visited = vec![center_id.to_string()];
        let mut seen = HashSet::from([center_id.to_string()]);
        let mut frontier = vec![center_id.to_string()];
        for _ in 0..depth {
            if visited.len() >= max_nodes {
                break;
            }
            let mut next = BTreeSet::new();
            for id in &frontier {
                for edge in store.get_connections_for_memory(id) {
                    let other = if edge.source_id == *id {
                        edge.target_id
                    } else {
                        edge.source_id
                    };
                    if !seen.contains(&other) && live(&other) {
                        next.insert(other);
                    }
                }
            }
            let room = max_nodes.saturating_sub(visited.len());
            let taken: Vec<String> = next.into_iter().take(room).collect();
            if taken.is_empty() {
                break;
            }
            for id in &taken {
                seen.insert(id.clone());
                visited.push(id.clone());
            }
            frontier = taken;
        }

        let mut edges: Vec<VestigeEdge> = store
            .edges()
            .iter()
            .filter(|edge| seen.contains(&edge.source_id) && seen.contains(&edge.target_id))
            .map(project_edge)
            .collect();
        edges.sort_by(|a, b| {
            (&a.source_id, &a.target_id, &a.link_type, a.created_at).cmp(&(
                &b.source_id,
                &b.target_id,
                &b.link_type,
                b.created_at,
            ))
        });
        let nodes = visited
            .iter()
            .filter_map(|id| {
                store
                    .get_node(id)
                    .filter(|record| record.is_live())
                    .map(|record| project_node(&store, &record))
            })
            .collect();
        Ok((nodes, edges))
    }

    fn supersession_pairs(&self) -> Result<Vec<(String, String)>, StorageError> {
        Ok(self.lock().supersession_pairs())
    }

    fn get_stats(&self) -> Result<MemoryStats, StorageError> {
        let store = self.lock();
        let nodes: Vec<_> = store.nodes().into_iter().filter(retrievable).collect();
        let strengths: Vec<f64> = nodes
            .iter()
            .map(|record| {
                store
                    .retrievability(&record.id)
                    .ok()
                    .flatten()
                    .unwrap_or(0.0)
            })
            .collect();
        let total = strengths.len() as i64;
        let average = if strengths.is_empty() {
            0.0
        } else {
            strengths.iter().sum::<f64>() / strengths.len() as f64
        };
        let oldest = nodes.iter().map(|n| n.created_at_ms).min().map(ms_to_dt);
        let newest = nodes.iter().map(|n| n.created_at_ms).max().map(ms_to_dt);
        let mut stats = MemoryStats::default();
        stats.total_nodes = total;
        stats.average_retention = average;
        stats.average_storage_strength = average;
        stats.average_retrieval_strength = average;
        stats.oldest_memory = oldest;
        stats.newest_memory = newest;
        Ok(stats)
    }

    fn state_distribution(&self) -> Result<(i64, i64, i64, i64), StorageError> {
        let store = self.lock();
        let mut bands = [0i64; 4];
        for record in store.nodes().into_iter().filter(retrievable) {
            let score = store
                .retrievability(&record.id)
                .ok()
                .flatten()
                .unwrap_or(0.0);
            let idx = if score >= 0.7 {
                0
            } else if score >= 0.4 {
                1
            } else if score >= 0.1 {
                2
            } else {
                3
            };
            bands[idx] += 1;
        }
        Ok((bands[0], bands[1], bands[2], bands[3]))
    }

    fn get_avg_retention(&self) -> Result<f64, StorageError> {
        Ok(self.get_stats()?.average_retention)
    }

    fn get_retention_distribution(&self) -> Result<Vec<(String, i64)>, StorageError> {
        let store = self.lock();
        let mut counts: BTreeMap<&'static str, i64> = BTreeMap::new();
        for record in store.nodes().into_iter().filter(retrievable) {
            let score = store
                .retrievability(&record.id)
                .ok()
                .flatten()
                .unwrap_or(0.0);
            *counts.entry(retention_bucket(score)).or_default() += 1;
        }
        Ok(counts
            .into_iter()
            .map(|(bucket, count)| (bucket.to_string(), count))
            .collect())
    }

    fn get_retention_trend(&self) -> Result<String, StorageError> {
        Ok("unknown".into())
    }

    fn count_memories_below_retention(&self, threshold: f64) -> Result<i64, StorageError> {
        let store = self.lock();
        let count = store
            .nodes()
            .iter()
            .filter(|record| retrievable(record))
            .filter(|record| {
                store
                    .retrievability(&record.id)
                    .ok()
                    .flatten()
                    .unwrap_or(0.0)
                    < threshold
            })
            .count();
        Ok(count as i64)
    }

    fn get_receipt(&self, receipt_id: &str) -> Result<Option<Receipt>, StorageError> {
        let store = self.lock();
        if node_lookup_hidden(&store, receipt_id) {
            return Ok(None);
        }
        let Some(proof) = resolve_proof(&store, receipt_id).map_err(map_store)? else {
            return Ok(None);
        };
        if edge_proof_hidden(&store, &proof) {
            return Ok(None);
        }
        let trust = store
            .retrievability(&proof.node_id)
            .ok()
            .flatten()
            .unwrap_or(0.0);
        Ok(Some(receipt_from_proof(&proof, trust)))
    }

    fn get_retrieval_replay_capsule(
        &self,
        _source_receipt_id: &str,
    ) -> Result<Option<vestige_core::storage::RetrievalReplayCapsuleSummary>, StorageError> {
        Ok(None)
    }

    fn receipt_attestation_status(
        &self,
        receipt_id: &str,
    ) -> Result<Option<ReceiptAttestationStatus>, StorageError> {
        let store = self.lock();
        if node_lookup_hidden(&store, receipt_id) {
            return Ok(None);
        }
        // The receipt tool replaces this with the log proof. LegacyUnsigned
        // only means "no DSSE envelope"; the effect itself is checked in get_receipt.
        Ok(resolve_proof(&store, receipt_id)
            .map_err(map_store)?
            .filter(|proof| !edge_proof_hidden(&store, proof))
            .map(|_| ReceiptAttestationStatus::LegacyUnsigned))
    }

    fn create_context_ablation_replay(
        &self,
        _source_receipt_id: &str,
        _withheld_slots: &[String],
    ) -> Result<vestige_core::storage::DurableCounterfactualReplay, StorageError> {
        Err(pending("create_context_ablation_replay"))
    }

    fn replay_receipt(&self, receipt_id: &str) -> Result<Value, StorageError> {
        let store = self.lock();
        let folded = store.refold().map_err(map_store)?;
        if let Some(edge_proof) = parse_receipt_seq(receipt_id)
            .map(|seq| store.effect_by_seq(seq))
            .transpose()
            .map_err(map_store)?
            .flatten()
            .filter(|proof| proof.action == strata_store::EffectAction::Edge)
        {
            return replay_edge_receipt(&store, &folded, &edge_proof);
        }
        let Some((node_id, seq)) = lookup_origin(&store, receipt_id) else {
            return Err(StorageError::NotFound(format!(
                "Receipt '{receipt_id}' was not found"
            )));
        };
        let mut mismatches = Vec::new();
        let live_digest = store.state_digest();
        if live_digest != folded.state_digest {
            mismatches.push("state_digest".to_string());
        }
        match (store.get_node(&node_id), folded.nodes.get(&node_id)) {
            (Some(live), Some(replayed)) => {
                mismatches.extend(node_field_mismatches(&node_id, &live, replayed));
            }
            _ => mismatches.push(format!("node:{node_id}:missing")),
        }
        if folded.origins.get(&node_id).copied() != Some(seq) {
            mismatches.push(format!("origin:{node_id}"));
        }
        let live_score = store.retrievability(&node_id).ok().flatten().unwrap_or(0.0);
        let folded_score = folded.retrievability.get(&node_id).copied().unwrap_or(0.0);
        if live_score.to_bits() != folded_score.to_bits() {
            mismatches.push(format!("retrievability:{node_id}"));
        }
        mismatches.extend(folded.gate_mismatches);
        mismatches.extend(folded.gaps);
        mismatches.sort();
        mismatches.dedup();
        let matched = mismatches.is_empty();
        Ok(json!({
            "action": "replay",
            "kind": "strata",
            "readOnly": true,
            "receiptId": receipt_id_for(seq),
            "nodeId": node_id,
            "effectSeq": seq,
            "matched": matched,
            "mismatches": mismatches,
            "stateDigest": hex32(&live_digest),
            "replayedDigest": hex32(&folded.state_digest),
            "frames": folded.frames,
            "claimBoundary": STRATA_REPLAY_BOUNDARY,
        }))
    }

    fn current_code_context_nodes(
        &self,
        node_type: &str,
        tag: Option<&str>,
        scope: &str,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        let store = self.lock();
        let mut out = Vec::new();
        let cap = usize::try_from(limit).unwrap_or(0);
        for record in store.nodes() {
            if !retrievable(&record) || record.node_type != node_type || record.scope != scope {
                continue;
            }
            if tag.is_some_and(|wanted| !record.tags.iter().any(|stored| stored == wanted)) {
                continue;
            }
            out.push(project_node(&store, &record));
            if out.len() >= cap {
                break;
            }
        }
        Ok(out)
    }

    fn current_code_context_scope_counts(
        &self,
        node_type: &str,
        tag: Option<&str>,
    ) -> Result<Vec<(String, usize)>, StorageError> {
        let store = self.lock();
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for record in store.nodes() {
            if !retrievable(&record) || record.node_type != node_type {
                continue;
            }
            if tag.is_some_and(|wanted| !record.tags.iter().any(|stored| stored == wanted)) {
                continue;
            }
            *counts.entry(record.scope.clone()).or_default() += 1;
        }
        Ok(counts.into_iter().collect())
    }

    fn code_anchors_for_node(&self, node_id: &str) -> Result<Vec<CodeAnchor>, StorageError> {
        Ok(self
            .lock()
            .anchors_for(node_id)
            .iter()
            .map(core_anchor)
            .collect())
    }

    fn code_anchors_for_nodes(
        &self,
        node_ids: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<CodeAnchor>>, StorageError> {
        // A memory with no anchors (or a retired one) is absent from the map;
        // the retrieval path reports that as unverifiable, never as stale.
        let store = self.lock();
        let mut out = std::collections::HashMap::new();
        for node_id in node_ids {
            let rows = store.anchors_for(node_id);
            if !rows.is_empty() {
                out.insert(node_id.clone(), rows.iter().map(core_anchor).collect());
            }
        }
        Ok(out)
    }

    fn record_code_anchors(&self, anchors: &[CodeAnchor]) -> Result<usize, StorageError> {
        if anchors.is_empty() {
            return Ok(0);
        }
        // SQLite's INSERT OR REPLACE keeps the last row for a repeated id.
        // The log admits one row per id, so fold repeats the same way.
        let mut rows: Vec<strata_store::AnchorRecord> = Vec::with_capacity(anchors.len());
        for anchor in anchors {
            let row = stored_anchor(anchor);
            match rows.iter_mut().find(|kept| kept.id == row.id) {
                Some(kept) => *kept = row,
                None => rows.push(row),
            }
        }
        self.lock().record_anchors(rows).map_err(map_store)?;
        Ok(anchors.len())
    }

    fn replace_code_anchors(
        &self,
        node_id: &str,
        scope: &str,
        anchors: &[CodeAnchor],
    ) -> Result<usize, StorageError> {
        if anchors.is_empty()
            || anchors
                .iter()
                .any(|anchor| anchor.node_id != node_id || !anchor.is_verifiable())
        {
            return Err(StorageError::Init(
                "Replacement requires complete verifiable anchors for this memory".into(),
            ));
        }
        let mut store = self.lock();
        let is_code_memory = store.get_node(node_id).is_some_and(|record| {
            retrievable(&record)
                && record.scope == scope.trim()
                && matches!(record.node_type.as_str(), "pattern" | "decision")
        });
        if !is_code_memory {
            return Err(StorageError::Init(
                "Code memory not found in requested scope".into(),
            ));
        }
        // Replaced evidence has not been checked yet, same as the SQLite path.
        let rows = anchors
            .iter()
            .map(|anchor| strata_store::AnchorRecord {
                last_verified_at_ms: None,
                last_status: None,
                ..stored_anchor(anchor)
            })
            .collect();
        store.replace_anchors(node_id, rows).map_err(map_store)?;
        Ok(anchors.len())
    }

    fn record_anchor_verification(
        &self,
        anchor_id: &str,
        status: AnchorStatus,
        checked_at: DateTime<Utc>,
    ) -> Result<(), StorageError> {
        // An unknown anchor, or one whose memory is retired, appends nothing.
        self.lock()
            .record_anchor_verdict(anchor_id, status.as_str(), checked_at.timestamp_millis())
            .map_err(map_store)?;
        Ok(())
    }

    fn projection_candidates(
        &self,
        scope: &str,
        min_retention: f64,
        limit: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        // Same predicate as the SQLite projection query. Content is not scanned.
        // The log has no suppression record, so that filter is empty. Read-only.
        let scope = projection_scope(scope)?;
        let store = self.lock();
        let now_ms = Utc::now().timestamp_millis();
        let cap = usize::try_from(limit.max(0)).unwrap_or(0);
        let nodes = store.nodes();
        let mut matched: Vec<strata_store::NodeRecord> = nodes
            .into_iter()
            .filter(|record| {
                if record.scope != scope || record.superseded_by.is_some() {
                    return false;
                }
                if record.valid_from_ms > now_ms || record.valid_until_ms <= now_ms {
                    return false;
                }
                if !projection_durable(record) {
                    return false;
                }
                let retention = store
                    .retrievability(&record.id)
                    .ok()
                    .flatten()
                    .unwrap_or(0.0);
                retention >= min_retention
            })
            .collect();
        matched.sort_by(|a, b| {
            projection_rank(&a.node_type)
                .cmp(&projection_rank(&b.node_type))
                .then_with(|| b.created_at_ms.cmp(&a.created_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        matched.truncate(cap);
        Ok(matched
            .iter()
            .map(|record| project_node(&store, record))
            .collect())
    }

    fn get_walk_receipt(
        &self,
        _receipt_id: &str,
    ) -> Result<Option<vestige_core::storage::StoredWalkReceipt>, StorageError> {
        Ok(None)
    }

    fn hybrid_search(
        &self,
        _query: &str,
        _limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
    ) -> Result<Vec<vestige_core::memory::SearchResult>, StorageError> {
        Err(similarity("hybrid_search"))
    }

    fn hybrid_search_filtered(
        &self,
        _query: &str,
        _limit: i32,
        _keyword_weight: f32,
        _semantic_weight: f32,
        _include_types: Option<&[String]>,
        _exclude_types: Option<&[String]>,
    ) -> Result<Vec<vestige_core::memory::SearchResult>, StorageError> {
        Err(similarity("hybrid_search_filtered"))
    }

    fn concrete_search_filtered(
        &self,
        _query: &str,
        _limit: i32,
        _include_types: Option<&[String]>,
        _exclude_types: Option<&[String]>,
    ) -> Result<Vec<vestige_core::memory::SearchResult>, StorageError> {
        // Query recall is not a Strata operation, even when the text happens
        // to equal a node's content: discovery is by exact handle only
        // (recall with `handle`). Refusing here keeps the auto-routed
        // "concrete" path from answering what the hybrid path refuses.
        Err(similarity("recall"))
    }

    fn blast_radius(
        &self,
        root_id: &str,
        open_only: bool,
    ) -> Result<vestige_core::BlastReport, StorageError> {
        self.blast_radius_with_link_types(root_id, open_only, &vestige_core::BLAST_LINK_TYPES)
    }

    fn blast_radius_with_link_types(
        &self,
        root_id: &str,
        open_only: bool,
        link_types: &[&str],
    ) -> Result<vestige_core::BlastReport, StorageError> {
        let store = self.lock();
        let root = store
            .get_node(root_id)
            .ok_or_else(|| StorageError::NotFound(root_id.to_string()))?;
        let root_node = project_node(&store, &root);
        let now = Utc::now();
        let open = |node: &KnowledgeNode| node.valid_until.is_none_or(|until| until > now);
        let mut affected = vec![vestige_core::BlastAffected {
            id: root_node.id.clone(),
            via: "root".to_string(),
            depth: 0,
        }];
        let mut visited = HashSet::from([root_node.id.clone()]);
        let mut queue = std::collections::VecDeque::from([(root_node.id.clone(), 0u32)]);
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= vestige_core::BLAST_MAX_DEPTH {
                continue;
            }
            for edge in store.get_connections_for_memory(&current) {
                if edge.source_id != current
                    || !link_types.contains(&edge.link_type.as_str())
                    || edge.target_id == current
                {
                    continue;
                }
                let target = edge.target_id.clone();
                if !visited.insert(target.clone()) {
                    continue;
                }
                let Some(record) = store.get_node(&target) else {
                    continue;
                };
                let node = project_node(&store, &record);
                if open_only && !open(&node) {
                    continue;
                }
                affected.push(vestige_core::BlastAffected {
                    id: target.clone(),
                    via: edge.link_type.clone(),
                    depth: depth + 1,
                });
                queue.push_back((target, depth + 1));
            }
        }
        affected[1..].sort_by(|a, b| (a.depth, &a.id).cmp(&(b.depth, &b.id)));
        let total = affected.len();
        Ok(vestige_core::BlastReport {
            root_id: root_node.id,
            affected,
            total,
        })
    }

    fn recall(
        &self,
        _input: vestige_core::RecallInput,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        Err(similarity("recall"))
    }

    fn tag_vocabulary(
        &self,
        _scope: Option<&str>,
    ) -> Result<vestige_core::storage::TagVocabulary, StorageError> {
        Err(similarity("tag_vocabulary"))
    }

    fn maintain_lifecycle_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
    ) -> Result<Value, StorageError> {
        page_bounds(limit, after, budget_ms)?;
        Ok(json!({
            "phase": "lifecycle",
            "dryRun": dry_run,
            "selected": 0,
            "processed": 0,
            "changed": 0,
            "hasMore": false
        }))
    }

    fn maintain_log_batch(&self, limit: usize, dry_run: bool) -> Result<Value, StorageError> {
        page_bounds(limit, None, 1)?;
        Ok(json!({
            "phase": "logs",
            "dryRun": dry_run,
            "selected": 0,
            "deleted": 0,
            "hasMore": false
        }))
    }

    fn maintain_gc_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
        _min_retention: f64,
        _max_age_days: Option<u64>,
    ) -> Result<Value, StorageError> {
        page_bounds(limit, after, budget_ms)?;
        Ok(json!({
            "tool": "gc",
            "dryRun": dry_run,
            "candidateCount": 0,
            "deleted": 0,
            "errors": 0,
            "processed": 0,
            "hasMore": false,
            "nextCursor": null,
            "atomic": true
        }))
    }

    fn preview_tag_mutation(
        &self,
        source_tags: &[String],
        target_tag: &str,
        scope: Option<&str>,
    ) -> Result<Value, StorageError> {
        if source_tags.is_empty()
            || source_tags.iter().any(|tag| tag.trim().is_empty())
            || target_tag.trim().is_empty()
        {
            return Err(StorageError::Init(
                "tag mutation requires nonempty exact tags".into(),
            ));
        }
        let target = target_tag.trim();
        let kinds = blocking_secrets(target);
        if !kinds.is_empty() {
            return Err(StorageError::SecretDetected { kinds });
        }
        let nodes = self.nodes();
        let selected = in_scope(&nodes, scope);
        let mut affected = Vec::new();
        let mut source_counts = serde_json::Map::new();
        for tag in source_tags {
            let tag = tag.trim();
            let mut count = 0i64;
            for node in &selected {
                if node.tags.iter().any(|stored| stored == tag) && !affected.contains(&node.id) {
                    affected.push(node.id.clone());
                }
                if node.tags.iter().any(|stored| stored == tag) {
                    count += 1;
                }
            }
            source_counts.insert(tag.to_string(), json!(count));
        }
        affected.sort();
        let target_count = selected
            .iter()
            .filter(|node| node.tags.iter().any(|stored| stored == target))
            .count();
        Ok(json!({
            "sourceTags": source_tags,
            "targetTag": target,
            "scope": scope,
            "allScopes": scope.is_none(),
            "sourceTagCounts": source_counts,
            "targetTagCount": target_count,
            "affectedMemoryCount": affected.len(),
            "affectedMemoryIds": affected,
            "affectedMemoryIdsTruncated": false,
            "withinOperationLimit": true,
            "previewToken": format!(
                "strata-preview:{}:{}:{target}",
                scope.unwrap_or("*"),
                source_tags.join("\u{1f}")
            ),
            "requiresConfirmation": true,
            "wouldWrite": false
        }))
    }

    fn get_never_composed_candidates(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        self.never_composed(Some(vestige_core::DEFAULT_MEMORY_SCOPE), limit, tag_filter)
    }

    fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        self.never_composed(
            scope.or(Some(vestige_core::DEFAULT_MEMORY_SCOPE)),
            limit,
            tag_filter,
        )
    }

    fn get_recent_composition_events(
        &self,
        _limit: i32,
    ) -> Result<Vec<CompositionEventRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn get_recent_composition_events_page(
        &self,
        _limit: i32,
        _offset: i32,
    ) -> Result<Vec<CompositionEventRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn query_time_range(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: i32,
        node_type: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        let store = self.lock();
        let mut nodes: Vec<KnowledgeNode> = store
            .nodes()
            .iter()
            .filter(|record| record.is_live())
            .filter(|record| {
                if !retrievable(record) {
                    return false;
                }
                let created = ms_to_dt(record.created_at_ms);
                start.is_none_or(|start| created >= start)
                    && end.is_none_or(|end| created <= end)
                    && node_type.is_none_or(|kind| record.node_type == kind)
                    && tags.is_none_or(|tags| {
                        tags.is_empty()
                            || tags
                                .iter()
                                .any(|tag| record.tags.iter().any(|stored| stored == tag))
                    })
            })
            .map(|record| project_node(&store, record))
            .collect();
        nodes.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));
        if limit >= 0 {
            nodes.truncate(limit as usize);
        }
        Ok(nodes)
    }

    fn hygiene_snapshot(&self, scope: Option<&str>) -> Result<HygieneSnapshot, StorageError> {
        let store = self.lock();
        let nodes = store
            .nodes()
            .into_iter()
            .filter(|record| retrievable(record) && scope.is_none_or(|scope| record.scope == scope))
            .map(|record| {
                let retrieval = store
                    .retrievability(&record.id)
                    .ok()
                    .flatten()
                    .unwrap_or(0.0);
                HygieneNodeSummary {
                    id: record.id,
                    node_type: record.node_type,
                    created_at: ms_to_dt(record.created_at_ms),
                    retention_strength: retrieval,
                    tags: record.tags,
                    valid_from: Some(ms_to_dt(record.valid_from_ms)),
                    valid_until: (record.valid_until_ms != VALID_FOREVER_MS)
                        .then(|| ms_to_dt(record.valid_until_ms)),
                    superseded: false,
                    content_bytes: record.content.len(),
                    content_preview: record.content.chars().take(140).collect(),
                    never_accessed: false,
                    access_unknown: true,
                }
            })
            .collect();
        Ok(HygieneSnapshot {
            nodes,
            malformed_tag_rows: 0,
            malformed_tag_row_ids: Vec::new(),
            malformed_tag_row_ids_truncated: false,
            defaulted_retention_rows: 0,
        })
    }

    fn list_tag_operations(
        &self,
        _limit: usize,
        _scope: Option<&str>,
    ) -> Result<Vec<vestige_core::advanced::MergeOperation>, StorageError> {
        Ok(Vec::new())
    }

    fn list_merge_operations(
        &self,
        limit: usize,
    ) -> Result<Vec<vestige_core::advanced::MergeOperation>, StorageError> {
        let mut writes = live_node_writes(&self.lock());
        writes.sort_by_key(|write| std::cmp::Reverse(write.frame_seq));
        writes.truncate(limit);
        Ok(writes.into_iter().map(merge_operation).collect())
    }

    fn get_merge_operation(
        &self,
        operation_id: &str,
    ) -> Result<Option<vestige_core::advanced::MergeOperation>, StorageError> {
        let Some(frame) = parse_op_frame(operation_id) else {
            return Ok(None);
        };
        Ok(live_node_writes(&self.lock())
            .into_iter()
            .find(|write| write.frame_seq == frame)
            .map(merge_operation))
    }

    fn merge_undo(
        &self,
        op_id: &str,
    ) -> Result<vestige_core::advanced::MergeOperation, StorageError> {
        let frame = parse_op_frame(op_id)
            .ok_or_else(|| StorageError::NotFound(format!("operation {op_id}")))?;
        let mut store = self.lock();
        // Versions this write's node retired (an edit's predecessors). The
        // undo brings them back, so the operation reports them as affected.
        let predecessors: Vec<String> = store
            .node_writes()
            .into_iter()
            .find(|write| write.frame_seq == frame)
            .map(|write| {
                store
                    .supersession_pairs()
                    .into_iter()
                    .filter(|(_, successor)| *successor == write.record.id)
                    .map(|(old, _)| old)
                    .collect()
            })
            .unwrap_or_default();
        let new_seq = store.undo_node_write(frame).map_err(map_store)?;
        let mut op = store
            .node_writes()
            .into_iter()
            .find(|write| write.frame_seq == new_seq)
            .map(merge_operation)
            .ok_or_else(|| {
                StorageError::Init("compensating record vanished after append".into())
            })?;
        for id in predecessors {
            if store.get_node(&id).is_some_and(|node| node.is_live())
                && !op.affected_ids.contains(&id)
            {
                op.affected_ids.push(id);
            }
        }
        Ok(op)
    }

    fn get_consolidation_history(
        &self,
        _limit: i32,
    ) -> Result<Vec<ConsolidationHistoryRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn get_recent_state_transitions(
        &self,
        _limit: i32,
    ) -> Result<Vec<StateTransitionRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn coverage_snapshot(&self) -> Result<CoverageSnapshot, StorageError> {
        let store = self.lock();
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for edge in store
            .edges()
            .into_iter()
            .filter(|edge| edge_visible(&store, edge))
        {
            *counts.entry(edge.link_type).or_default() += 1;
        }
        Ok(CoverageSnapshot {
            total_nodes: store.node_count() as u64,
            anchored_nodes: 0,
            anchor_coverage_pct: 0.0,
            edge_counts_by_type: counts.into_iter().collect(),
            newest_git_commit_record: None,
            newest_git_commit_record_age_days: None,
            newest_agent_trace_at: None,
            newest_agent_trace_age_hours: None,
        })
    }

    fn load_active_synaptic_tags(
        &self,
    ) -> Result<Vec<vestige_core::neuroscience::SynapticTag>, StorageError> {
        Ok(Vec::new())
    }

    fn promote_memory(&self, id: &str) -> Result<KnowledgeNode, StorageError> {
        self.review_memory(id, PROMOTE_RATING)
    }

    fn promote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> Result<vestige_core::storage::ActorMutationOutcome, StorageError> {
        self.actor_review(id, claimed_role, tool, PROMOTE_RATING, "support")
    }

    fn demote_memory(&self, id: &str) -> Result<KnowledgeNode, StorageError> {
        self.review_memory(id, DEMOTE_RATING)
    }

    fn demote_memory_as_actor(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
    ) -> Result<vestige_core::storage::ActorMutationOutcome, StorageError> {
        self.actor_review(id, claimed_role, tool, DEMOTE_RATING, "oppose")
    }

    fn purge_node(
        &self,
        _id: &str,
        _reason: Option<&str>,
    ) -> Result<vestige_core::storage::PurgeReport, StorageError> {
        Err(withheld("purge"))
    }

    fn delete_node(&self, _id: &str) -> Result<bool, StorageError> {
        Err(pending("delete_node"))
    }

    fn get_active_intentions(
        &self,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        let rows = self.intentions()?;
        Ok(sort_intentions(
            rows.into_iter()
                .filter(|row| is_prospective(row) && row.status == "active")
                .collect(),
        ))
    }

    fn get_active_intentions_in_scope(
        &self,
        scope: &str,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        let scope = scope.trim();
        let rows = self.intentions()?;
        Ok(sort_intentions(
            rows.into_iter()
                .filter(|row| {
                    is_prospective(row) && row.status == "active" && row.effective_scope() == scope
                })
                .collect(),
        ))
    }

    fn get_intention(
        &self,
        id: &str,
    ) -> Result<Option<vestige_core::storage::IntentionRecord>, StorageError> {
        match self.lock().get_intention(id) {
            Some(record) => {
                let row = core_intention(&record)?;
                if !is_prospective(&row) {
                    return Ok(None);
                }
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    fn get_intentions_by_status(
        &self,
        status: &str,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        let rows = self.intentions()?;
        Ok(sort_intentions(
            rows.into_iter()
                .filter(|row| is_prospective(row) && row.status == status)
                .collect(),
        ))
    }

    fn get_overdue_intentions(
        &self,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        let now = Utc::now();
        let mut rows = self.intentions()?;
        rows.retain(|row| {
            is_prospective(row)
                && row.status == "active"
                && row.deadline.is_some_and(|deadline| deadline < now)
        });
        rows.sort_by(|a, b| a.deadline.cmp(&b.deadline).then(a.id.cmp(&b.id)));
        Ok(rows)
    }

    fn save_intention(
        &self,
        intention: &vestige_core::storage::IntentionRecord,
    ) -> Result<(), StorageError> {
        gate_intention(intention)?;
        self.lock()
            .upsert_intentions(vec![stored_intention(intention)])
            .map_err(map_store)
            .map(|_| ())
    }

    fn snooze_intention(&self, id: &str, until: DateTime<Utc>) -> Result<bool, StorageError> {
        let mut store = self.lock();
        let Some(current) = store.get_intention(id) else {
            return Ok(false);
        };
        if current.source_type == crate::intention_graph_log::SOURCE {
            return Ok(false);
        }
        let mut record = core_intention(&current)?;
        if matches!(record.status.as_str(), "fulfilled" | "cancelled") {
            return Ok(false);
        }
        record.status = "snoozed".to_string();
        record.snoozed_until = Some(until);
        store
            .upsert_intentions(vec![stored_intention(&record)])
            .map_err(map_store)?;
        Ok(true)
    }

    fn update_intention_status(&self, id: &str, status: &str) -> Result<bool, StorageError> {
        let mut store = self.lock();
        let Some(current) = store.get_intention(id) else {
            return Ok(false);
        };
        if current.source_type == crate::intention_graph_log::SOURCE {
            return Ok(false);
        }
        let mut record = core_intention(&current)?;
        record.status = status.to_string();
        record.fulfilled_at = if status == "fulfilled" {
            Some(Utc::now())
        } else {
            None
        };
        store
            .upsert_intentions(vec![stored_intention(&record)])
            .map_err(map_store)?;
        Ok(true)
    }

    fn commit_intention_check(
        &self,
        changes: &[(
            vestige_core::storage::IntentionRecord,
            vestige_core::storage::IntentionRecord,
        )],
    ) -> Result<(), String> {
        if changes.is_empty() {
            return Ok(());
        }
        let mut store = self.lock();
        let mut next = Vec::with_capacity(changes.len());
        for (old, new) in changes {
            if old.id != new.id {
                return Err("intention check cannot change record identity".into());
            }
            let Some(current) = store.get_intention(&old.id) else {
                let id = &old.id;
                return Err(format!(
                    "Intention '{id}' changed during check; retry the check"
                ));
            };
            if !current.same_claim(&stored_intention(old)) {
                let id = &old.id;
                return Err(format!(
                    "Intention '{id}' changed during check; retry the check"
                ));
            }
            next.push(stored_intention(new));
        }
        store
            .upsert_intentions(next)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    fn apply_intention_graph(
        &self,
        scope: &str,
        command: Command,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        crate::intention_graph_log::apply(&mut self.lock(), scope, command, now)
    }

    fn replay_intention_graph(&self, scope: &str) -> Result<Value, String> {
        crate::intention_graph_log::replay(&self.lock(), scope)
    }

    fn intention_memory_snapshot(
        &self,
        scope: &str,
        memory_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        crate::intention_graph_log::memory_snapshot(&self.lock(), scope, memory_id, now)
    }

    fn suppress_memory(&self, id: &str) -> Result<KnowledgeNode, StorageError> {
        // A Strata suppression hides the memory from every read. The log
        // keeps its bytes, which is what suppress has always meant (it never
        // deleted); on Strata it cannot be reversed.
        let mut store = self.lock();
        let receipt = retire_live(&mut store, id, strata_store::RULE_SUPPRESS, false)?;
        // A GhostLink composition record names both members in its source
        // key. Suppression hides a memory from every read, so each live
        // record composed from it is withdrawn with it, through the gate
        // with its own receipt.
        let compositions: Vec<String> = store
            .nodes()
            .into_iter()
            .filter(|record| record.is_live())
            .filter(|record| {
                strata_store::composition_pair(record).is_some_and(|(a, b)| a == id || b == id)
            })
            .map(|record| record.id)
            .collect();
        for record in compositions {
            retire_live(&mut store, &record, strata_store::RULE_SUPPRESS, false)?;
        }
        // The trait returns a node, not a receipt. `source` carries the eff-
        // id for this call only; the log record is unchanged.
        let mut node = KnowledgeNode::default();
        node.id = id.to_string();
        node.source = Some(receipt.receipt_id);
        node.suppression_count = 1;
        Ok(node)
    }

    fn update_node_content(&self, id: &str, new_content: &str) -> Result<(), StorageError> {
        if new_content.trim().is_empty() {
            return Err(StorageError::Init("content must not be empty".into()));
        }
        let kinds = blocking_secrets(new_content);
        if !kinds.is_empty() {
            return Err(StorageError::SecretDetected { kinds });
        }
        let mut store = self.lock();
        if store.get_node(id).is_none() {
            return Err(StorageError::NotFound(id.to_string()));
        }
        // An edit admits a successor and retires this node, and a retired
        // node has no anchors. Read them first so the code evidence moves
        // with the memory (SQLite edits in place and keeps them).
        let anchors = store.anchors_for(id);
        let (successor, _) = store
            .edit(
                id,
                new_content,
                &strata_store::AdmissionContext {
                    rule_id: Some(strata_store::RULE_EDIT.to_string()),
                    confirm: false,
                },
            )
            .map_err(map_store)?;
        if !anchors.is_empty() {
            let moved = anchors
                .into_iter()
                .map(|anchor| strata_store::AnchorRecord {
                    node_id: successor.clone(),
                    ..anchor
                })
                .collect();
            // Same anchor ids, so they move rather than duplicate. The edit
            // already stands; a failure here is reported, not rolled back.
            if let Err(err) = store.record_anchors(moved) {
                tracing::warn!(memory_id = %successor, error = %err, "edit kept the memory but not its code anchors");
            }
        }
        Ok(())
    }
}

impl StrataMemory {
    fn intentions(&self) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        self.lock()
            .intentions()
            .iter()
            .map(core_intention)
            .collect()
    }

    fn review_memory(&self, id: &str, rating: u8) -> Result<KnowledgeNode, StorageError> {
        let mut store = self.lock();
        if store.get_node(id).is_none() {
            return Err(StorageError::NotFound(id.to_string()));
        }
        store.review(id, rating).map_err(map_store)?;
        let record = store
            .get_node(id)
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
        Ok(project_node(&store, &record))
    }

    /// Bound-actor path. The admitted fact is the FSRS review. The endorsement
    /// block quotes this call; it is not a second log vote.
    fn actor_review(
        &self,
        id: &str,
        claimed_role: Option<&str>,
        tool: &str,
        rating: u8,
        endorsement_kind: &str,
    ) -> Result<vestige_core::storage::ActorMutationOutcome, StorageError> {
        let actor_did = self
            .process_actor_did()
            .ok_or_else(|| StorageError::Init("no process actor is bound to this store".into()))?;
        let before = self
            .get_node(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
        let node = self.review_memory(id, rating)?;
        let mut receipt = self
            .get_receipt(id)?
            .ok_or_else(|| StorageError::Init(format!("write receipt missing for {id}")))?;
        let resolution =
            vestige_core::ActorPolicySnapshot::flat_v1().resolve(&actor_did, claimed_role);
        receipt.actor = Some(vestige_core::ActorProvenance::from_resolution(
            &actor_did,
            &resolution,
        ));
        let digest = vestige_core::revision_digest(&before.content);
        let event_id =
            vestige_core::endorsement_event_id(&actor_did, id, &digest, endorsement_kind);
        let receipt_id = receipt.receipt_id.clone();
        Ok(vestige_core::storage::ActorMutationOutcome {
            before,
            node,
            receipt,
            endorsement: vestige_core::storage::EndorsementEventRecord {
                event_id,
                memory_id: id.to_string(),
                actor_did,
                claimed_role: resolution.claimed_role.clone(),
                effective_role: resolution.effective_role,
                resolved_weight: resolution.resolved_weight,
                resolution_disposition: resolution.disposition.as_str().to_string(),
                policy_version: resolution.policy_version,
                endorsement_kind: endorsement_kind.to_string(),
                revision_digest: digest,
                independent_prior: resolution.resolved_weight,
                tool: tool.to_string(),
                receipt_id: Some(receipt_id),
                created_at: Utc::now().to_rfc3339(),
            },
            already_recorded: false,
        })
    }

    /// Quantized FSRS card: stability, difficulty, last_seq, reps, lapses.
    pub fn card_q(&self, id: &str) -> Option<(i64, i64, u64, u32, u32)> {
        self.lock().card_state(id).map(|card| {
            (
                card.stability_q,
                card.difficulty_q,
                card.last_seq,
                card.review_count,
                card.lapse_count,
            )
        })
    }

    fn page_nodes(
        &self,
        scope: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        let store = self.lock();
        let mut nodes: Vec<KnowledgeNode> = store
            .nodes()
            .iter()
            .filter(|record| retrievable(record) && scope.is_none_or(|scope| record.scope == scope))
            .map(|record| project_node(&store, record))
            .collect();
        nodes.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));
        let offset = usize::try_from(offset).unwrap_or(0);
        if offset >= nodes.len() || limit <= 0 {
            return Ok(Vec::new());
        }
        nodes.drain(0..offset);
        nodes.truncate(limit as usize);
        Ok(nodes)
    }

    /// GhostLink bridge lens (the owner ruling of 2026-09-28): pool pairs
    /// within three undirected hops over recorded touched / derived_from /
    /// closed_by edges, never woven, scored by hop proximity, composition
    /// novelty, retention trust and prior outcomes. Tags filter by exact
    /// identity; no content, tag-name or term overlap is computed.
    fn never_composed(
        &self,
        scope: Option<&str>,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        let Some(scope) = scope else {
            return Ok(Vec::new());
        };
        let scope = normalize_scope(scope)?;
        let limit = usize::try_from(limit).unwrap_or(0).max(1);
        Ok(ghostlink::bridge_trait_candidates(
            self,
            Some(scope),
            tag_filter.filter(|tags| !tags.is_empty()),
            limit,
        ))
    }
}

fn merge_policy_path(data_dir: &Path) -> PathBuf {
    data_dir.join(MERGE_POLICY_FILE)
}

/// Same precedence as the SQLite `fsrs_config` read when no row is stored:
/// env, then [`vestige_core::MergePolicy::default`], then `MergePolicy::new`
/// (clamp, and `possible <= match`).
fn merge_policy_from_env() -> vestige_core::MergePolicy {
    let default = vestige_core::MergePolicy::default();
    let env_f32 = |name: &str, fallback: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(fallback)
    };
    let auto_apply = std::env::var("VESTIGE_MERGE_AUTO_APPLY")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(default.auto_apply);
    vestige_core::MergePolicy::new(
        env_f32("VESTIGE_MERGE_MATCH_THRESHOLD", default.match_threshold),
        env_f32(
            "VESTIGE_MERGE_POSSIBLE_THRESHOLD",
            default.possible_threshold,
        ),
        auto_apply,
    )
}

fn read_merge_policy_file(
    data_dir: &Path,
) -> Result<Option<vestige_core::MergePolicy>, StorageError> {
    let path = merge_policy_path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(StorageError::Io(err)),
    };
    let policy: vestige_core::MergePolicy = serde_json::from_slice(&bytes)
        .map_err(|err| StorageError::Init(format!("merge policy file is not readable: {err}")))?;
    Ok(Some(vestige_core::MergePolicy::new(
        policy.match_threshold,
        policy.possible_threshold,
        policy.auto_apply,
    )))
}

fn write_merge_policy_file(
    data_dir: &Path,
    policy: &vestige_core::MergePolicy,
) -> Result<(), StorageError> {
    let bytes = serde_json::to_vec(policy)
        .map_err(|err| StorageError::Init(format!("merge policy encode failed: {err}")))?;
    let path = merge_policy_path(data_dir);
    let tmp = data_dir.join(format!("{MERGE_POLICY_FILE}.tmp"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn parse_op_frame(operation_id: &str) -> Option<u64> {
    let rest = operation_id.strip_prefix("op-")?;
    if rest.len() != 16 || !rest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(rest, 16).ok()
}

/// Node writes whose node is still live. The changelog and operation reads
/// are read tools, so a retired node's id stays out of them like every other
/// read; the frames themselves remain on the log.
fn live_node_writes(store: &strata_store::StrataStore) -> Vec<strata_store::NodeWrite> {
    store
        .node_writes()
        .into_iter()
        .filter(|write| {
            store
                .get_node(&write.record.id)
                .is_some_and(|node| node.is_live())
        })
        .collect()
}

fn merge_operation(write: strata_store::NodeWrite) -> vestige_core::advanced::MergeOperation {
    let reverts = write.reverts_frame_seq.map(|seq| format!("op-{seq:016x}"));
    let created_at = ms_to_dt(write.record.created_at_ms).to_rfc3339();
    let reason = if write.op_type == "undo" {
        Some(format!(
            "Reverted {} by appending a compensating record",
            reverts.as_deref().unwrap_or("the prior write")
        ))
    } else {
        Some("admitted node write".into())
    };
    vestige_core::advanced::MergeOperation {
        id: format!("op-{:016x}", write.frame_seq),
        plan_id: None,
        op_type: write.op_type.to_string(),
        status: write.status.to_string(),
        created_at: created_at.clone(),
        reverted_at: (write.status == "reverted").then_some(created_at),
        reverts_op_id: reverts,
        survivor_id: Some(write.record.id.clone()),
        affected_ids: vec![write.record.id],
        confidence: None,
        signals: None,
        reason,
    }
}

/// Replay an edge receipt: the refolded log must reach the live state, and
/// the edge the receipt names must be in it.
fn replay_edge_receipt(
    store: &strata_store::StrataStore,
    folded: &strata_store::Refold,
    proof: &strata_store::EffectProof,
) -> Result<Value, StorageError> {
    let receipt_id = receipt_id_for(proof.effect_seq);
    if edge_proof_hidden(store, proof) {
        return Err(StorageError::NotFound(format!(
            "Receipt '{receipt_id}' was not found"
        )));
    }
    let (target, kind) = proof.edge.clone().unwrap_or_default();
    let mut mismatches = Vec::new();
    let live_digest = store.state_digest();
    if live_digest != folded.state_digest {
        mismatches.push("state_digest".to_string());
    }
    let present = store.edges().iter().any(|edge| {
        edge.source_id == proof.node_id && edge.target_id == target && edge.link_type == kind
    });
    if !present {
        mismatches.push(format!("edge:{}:{kind}:{target}:missing", proof.node_id));
    }
    mismatches.extend(folded.gate_mismatches.iter().cloned());
    mismatches.extend(folded.gaps.iter().cloned());
    mismatches.sort();
    mismatches.dedup();
    Ok(json!({
        "action": "replay",
        "kind": "strata",
        "readOnly": true,
        "receiptId": receipt_id,
        "edge": { "source": proof.node_id, "target": target, "kind": kind },
        "effectSeq": proof.effect_seq,
        "matched": mismatches.is_empty(),
        "mismatches": mismatches,
        "stateDigest": hex32(&live_digest),
        "replayedDigest": hex32(&folded.state_digest),
        "frames": folded.frames,
        "claimBoundary": STRATA_REPLAY_BOUNDARY,
    }))
}

fn lookup_origin(
    store: &strata_store::StrataStore,
    receipt_or_node: &str,
) -> Option<(String, u64)> {
    if let Some(seq) = parse_receipt_seq(receipt_or_node) {
        return store
            .origins()
            .into_iter()
            .find(|(_, origin)| *origin == seq);
    }
    store
        .origin_seq(receipt_or_node)
        .map(|seq| (receipt_or_node.to_string(), seq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};
    use serde_json::json;
    use vestige_core::IngestInput;
    use vestige_core::SourceEnvelope;

    fn no_sqlite(dir: &Path) -> bool {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".sqlite")
                    || name.ends_with(".sqlite3")
                    || name.ends_with(".db")
                    || name.ends_with(".db-wal")
                    || name.ends_with(".db-shm")
                {
                    return false;
                }
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        true
    }

    #[test]
    fn empty_dir_reopens_without_sqlite_and_keeps_the_receipt() {
        let dir = tempfile::TempDir::new().unwrap();
        let first = StrataMemory::open(dir.path()).unwrap();
        let node = first
            .ingest_in_scope(
                IngestInput {
                    content: "strata boot fixture".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        assert!(is_mem_id(&node.id));
        let receipt = first.get_receipt(&node.id).unwrap().expect("write receipt");
        assert_eq!(receipt.retrieved, vec![node.id.clone()]);
        assert!(receipt.activation_path.is_empty());
        drop(first);

        let second = StrataMemory::open(dir.path()).unwrap();
        let again = second.get_node(&node.id).unwrap().expect("replayed node");
        assert_eq!(again.content, "strata boot fixture");
        let replayed = second.get_receipt(&receipt.receipt_id).unwrap().unwrap();
        assert_eq!(replayed.retrieved, vec![node.id.clone()]);
        assert!(no_sqlite(dir.path()));
        let err = second
            .hybrid_search_filtered("strata", 5, 1.0, 0.0, None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("similarity_disabled"), "{err}");
        assert!(again.source.is_none());
        assert!(again.source_envelope.is_none());
    }

    fn remove_segments(dir: &Path) {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "seg") {
                    std::fs::remove_file(&path).unwrap();
                }
            }
        }
    }

    #[test]
    fn receipt_lookup_after_edit_is_served_without_rereading_the_log() {
        let dir = tempfile::TempDir::new().unwrap();
        let memory = StrataMemory::open(dir.path()).unwrap();
        let node = memory
            .ingest_in_scope(
                IngestInput {
                    content: "receipt lookup fixture".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        memory
            .update_node_content(&node.id, "receipt lookup fixture, edited")
            .unwrap();
        let successor = memory
            .lock()
            .supersession_pairs()
            .into_iter()
            .find(|(old, _)| old == &node.id)
            .map(|(_, successor)| successor)
            .expect("edit retired the original");

        // With the segment files gone, only the in-memory effect index can answer.
        remove_segments(dir.path());

        let receipt = memory
            .get_receipt(&successor)
            .unwrap()
            .expect("edit receipt for the successor");
        assert_eq!(receipt.retrieved, vec![successor.clone()]);
        let by_id = memory
            .get_receipt(&receipt.receipt_id)
            .unwrap()
            .expect("lookup by receipt id");
        assert_eq!(by_id.receipt_id, receipt.receipt_id);
        assert!(
            memory
                .receipt_attestation_status(&successor)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn ingest_keeps_source_and_source_updated_at() {
        let dir = tempfile::TempDir::new().unwrap();
        let memory = StrataMemory::open(dir.path()).unwrap();
        let updated = DateTime::parse_from_rfc3339("2026-02-20T11:30:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let mut label_envelope = SourceEnvelope::default();
        label_envelope.source_updated_at = Some(updated);
        let sourced = memory
            .ingest(IngestInput {
                content: "v3 source label".into(),
                source: Some("fixture".into()),
                source_envelope: Some(label_envelope),
                ..IngestInput::default()
            })
            .unwrap();
        assert_eq!(sourced.source.as_deref(), Some("fixture"));
        assert_eq!(
            sourced
                .source_envelope
                .as_ref()
                .and_then(|env| env.source_updated_at),
            Some(updated)
        );

        let mut keyed_envelope = SourceEnvelope::default();
        keyed_envelope.source_system = Some("github".into());
        keyed_envelope.source_project = Some("vestige".into());
        keyed_envelope.source_id = Some("310".into());
        keyed_envelope.source_updated_at = Some(updated);
        let keyed = memory
            .ingest(IngestInput {
                content: "connector row".into(),
                source_envelope: Some(keyed_envelope),
                ..IngestInput::default()
            })
            .unwrap();
        let envelope = keyed.source_envelope.expect("connector key dropped");
        assert_eq!(envelope.source_system.as_deref(), Some("github"));
        assert_eq!(envelope.source_project.as_deref(), Some("vestige"));
        assert_eq!(envelope.source_id.as_deref(), Some("310"));
        assert_eq!(envelope.source_updated_at, Some(updated));
    }

    #[test]
    fn malformed_scopes_are_refused_and_a_padded_scope_is_trimmed() {
        let dir = tempfile::TempDir::new().unwrap();
        let memory = StrataMemory::open(dir.path()).unwrap();
        let input = || IngestInput {
            content: "scoped fixture memory".into(),
            ..IngestInput::default()
        };
        for scope in ["", "   ", &"s".repeat(250), "bad\u{7}scope"] {
            let err = memory.ingest_in_scope(input(), scope).unwrap_err();
            assert!(
                matches!(err, StorageError::InvalidScope(_)),
                "{scope:?}: {err}"
            );
        }
        let node = memory.ingest_in_scope(input(), "  user  ").unwrap();
        let stored = memory.lock().get_node(&node.id).expect("stored node");
        assert_eq!(stored.scope, "user");
    }

    #[test]
    fn intention_check_commit_rejects_a_stale_snapshot() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = StrataMemory::open(dir.path()).unwrap();
        let created_at = "2026-01-01T00:00:00Z".parse().unwrap();
        let record = vestige_core::storage::IntentionRecord {
            id: "int-stale".into(),
            content: "Synthetic reminder".into(),
            trigger_type: "time".into(),
            trigger_data: "{}".into(),
            priority: 2,
            status: "active".into(),
            created_at,
            deadline: None,
            fulfilled_at: None,
            reminder_count: 0,
            last_reminded_at: None,
            notes: None,
            tags: Vec::new(),
            related_memories: Vec::new(),
            snoozed_until: None,
            source_type: "mcp".into(),
            source_data: None,
            scope: Some("user".into()),
        };
        store.save_intention(&record).unwrap();
        let mut delivered = record.clone();
        delivered.reminder_count = 1;
        delivered.last_reminded_at = Some(created_at);
        store
            .commit_intention_check(&[(record.clone(), delivered.clone())])
            .unwrap();
        let err = store
            .commit_intention_check(&[(record.clone(), delivered)])
            .unwrap_err();
        assert!(err.contains("changed during check"), "{err}");
        let kept = store.get_intention(&record.id).unwrap().unwrap();
        assert_eq!(kept.reminder_count, 1);
        let receipt = store.get_receipt(&record.id).unwrap().unwrap();
        assert_eq!(receipt.retrieved, vec![record.id]);
        assert!(receipt.receipt_id.starts_with("eff-"));
        assert!(no_sqlite(dir.path()));
    }

    fn sample_intention(id: &str) -> vestige_core::storage::IntentionRecord {
        vestige_core::storage::IntentionRecord {
            id: id.into(),
            content: "Synthetic reminder".into(),
            trigger_type: "time".into(),
            trigger_data: "{}".into(),
            priority: 2,
            status: "active".into(),
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
            deadline: None,
            fulfilled_at: None,
            reminder_count: 0,
            last_reminded_at: None,
            notes: None,
            tags: Vec::new(),
            related_memories: Vec::new(),
            snoozed_until: None,
            source_type: "mcp".into(),
            source_data: None,
            scope: Some("user".into()),
        }
    }

    #[test]
    fn intention_check_commit_delivers_the_batch_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = StrataMemory::open(dir.path()).unwrap();
        let left = sample_intention("int-left");
        let right = sample_intention("int-right");
        store.save_intention(&left).unwrap();
        store.save_intention(&right).unwrap();
        let reminded_at = left.created_at;
        let mut left_next = left.clone();
        left_next.reminder_count = 1;
        left_next.last_reminded_at = Some(reminded_at);
        let mut right_next = right.clone();
        right_next.reminder_count = 1;
        right_next.last_reminded_at = Some(reminded_at);
        store
            .commit_intention_check(&[
                (left.clone(), left_next.clone()),
                (right.clone(), right_next.clone()),
            ])
            .unwrap();
        let left_receipt = store.get_receipt(&left.id).unwrap().unwrap();
        let right_receipt = store.get_receipt(&right.id).unwrap().unwrap();
        assert_eq!(left_receipt.receipt_id, right_receipt.receipt_id);
        assert!(left_receipt.receipt_id.starts_with("eff-"));

        let mut left_again = left_next.clone();
        left_again.reminder_count = 2;
        let err = store
            .commit_intention_check(&[(left_next, left_again), (right.clone(), right_next.clone())])
            .unwrap_err();
        assert!(err.contains("changed during check"), "{err}");
        assert_eq!(
            store
                .get_intention(&left.id)
                .unwrap()
                .unwrap()
                .reminder_count,
            1
        );
        assert_eq!(
            store
                .get_intention(&right.id)
                .unwrap()
                .unwrap()
                .reminder_count,
            1
        );
        assert_eq!(
            store.get_receipt(&right.id).unwrap().unwrap().receipt_id,
            right_receipt.receipt_id
        );
        assert!(no_sqlite(dir.path()));
    }

    #[test]
    fn intention_graph_rows_stay_out_of_prospective_lists() {
        use vestige_core::storage::MemoryStoreSend;

        let dir = tempfile::TempDir::new().unwrap();
        let store = StrataMemory::open(dir.path()).unwrap();
        let at = "2026-10-01T09:00:00Z".parse().unwrap();
        let command = serde_json::from_value(json!({
            "action": "plan",
            "id": "p1",
            "description": "Synthetic graph plan",
            "requirements": [],
            "conflict_keys": []
        }))
        .unwrap();
        let written = store.apply_intention_graph("user", command, at).unwrap();
        assert!(written["journal_seq"].as_i64().unwrap() >= 1);
        assert!(store.get_active_intentions().unwrap().is_empty());
        assert!(store.get_intentions_by_status("active").unwrap().is_empty());
        assert!(store.get_intention("igs|user").unwrap().is_none());
        let replayed = store.replay_intention_graph("user").unwrap();
        assert_eq!(replayed["matched"], true);
        assert_eq!(replayed["commands"], 1);
        assert!(no_sqlite(dir.path()));
    }

    #[test]
    fn node_field_mismatch_names_only_the_changed_field() {
        let live = strata_store::NodeRecord {
            id: "mem-0000000000000001".into(),
            kernel_id: 2,
            scope: "user".into(),
            content: "original".into(),
            node_type: "fact".into(),
            tags: vec!["a".into()],
            created_at_ms: 1,
            valid_from_ms: 1,
            valid_until_ms: i64::MAX,
            superseded_by: None,
            source: None,
            source_updated_at_ms: None,
        };
        let mut folded = live.clone();
        folded.content = "tampered".into();
        assert_eq!(
            node_field_mismatches(&live.id, &live, &folded),
            vec!["node:mem-0000000000000001:content".to_string()]
        );
        assert!(node_field_mismatches(&live.id, &live, &live).is_empty());
    }

    fn log_files(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut stack = vec![dir.join("log")];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(bytes) = std::fs::read(&path) {
                    let rel = path
                        .strip_prefix(dir)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .into_owned();
                    out.push((rel, bytes));
                }
            }
        }
        out.sort();
        out
    }

    /// 4.0 withholds erasure on Strata: purge refuses through the tool and
    /// the store and writes nothing. Suppress works as an honest hide: the
    /// memory leaves every read, the log keeps its bytes, and reverse is
    /// refused.
    #[tokio::test]
    async fn purge_is_withheld_and_suppress_hides_without_erasing() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = open(dir.path()).unwrap();
        let marker = "WITHHELD_MARKER_STAYS";
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: marker.into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        let before = log_files(dir.path());

        let purged = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "purge", "id": node.id, "confirm": true })),
        )
        .await
        .unwrap_err();
        assert!(purged.contains("unavailable_in_4_0"), "{purged}");
        let store_purge = storage.purge_node(&node.id, None).unwrap_err().to_string();
        assert!(store_purge.contains("unavailable_in_4_0"), "{store_purge}");
        assert_eq!(
            log_files(dir.path()),
            before,
            "a withheld purge wrote to the log"
        );
        assert_eq!(storage.get_node(&node.id).unwrap().unwrap().content, marker);

        let suppressed = crate::tools::suppress::execute(
            &storage,
            Some(serde_json::json!({ "id": node.id, "reason": "fixture" })),
        )
        .await
        .unwrap();
        assert_eq!(suppressed["success"], true, "{suppressed}");
        assert!(!suppressed.to_string().contains(marker));
        assert!(storage.get_node(&node.id).unwrap().is_none());
        assert!(storage.resolve_handle(&node.id).ids.is_empty());
        // Not erasure: the bytes are still on the log.
        let raw: Vec<u8> = log_files(dir.path())
            .into_iter()
            .flat_map(|(_, bytes)| bytes)
            .collect();
        assert!(
            raw.windows(marker.len())
                .any(|window| window == marker.as_bytes()),
            "suppress must not claim erasure; the log keeps the bytes"
        );
        let reversed = crate::tools::suppress::execute(
            &storage,
            Some(serde_json::json!({ "id": node.id, "reverse": true })),
        )
        .await
        .unwrap_err();
        assert!(reversed.contains("unavailable_in_4_0"), "{reversed}");
    }

    #[tokio::test]
    async fn retired_nodes_stay_out_of_every_read_tool() {
        let dir = tempfile::TempDir::new().unwrap();
        let marker = "RETIRED_READ_MARKER_7c2e";
        let (doomed, kept) = {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let doomed = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: marker.into(),
                        node_type: "fact".into(),
                        ..strata_store::IngestInput::default()
                    },
                    "user",
                )
                .unwrap();
            let kept = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: "KEPT_READ_VISIBLE".into(),
                        node_type: "fact".into(),
                        ..strata_store::IngestInput::default()
                    },
                    "user",
                )
                .unwrap();
            store
                .save_connection(&strata_store::ConnectionRecord {
                    source_id: kept.clone(),
                    target_id: doomed.clone(),
                    strength_milli: 1000,
                    link_type: "derived_from".into(),
                    meta_sha: None,
                    created_at_ms: 0,
                    activation_count: 0,
                })
                .unwrap();
            store
                .retire(
                    &doomed,
                    &kept,
                    &strata_store::AdmissionContext {
                        rule_id: Some(strata_store::RULE_SUPPRESS.to_string()),
                        confirm: false,
                    },
                )
                .unwrap();
            (doomed, kept)
        };

        let storage = open(dir.path()).unwrap();
        let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        cognitive.lock().await.hydrate(&storage);
        let config = vestige_core::OutputConfig::default();

        let prefix = &doomed[..12];
        let resolved = storage.resolve_handle(&doomed);
        assert!(
            resolved.ids.is_empty(),
            "exact handle returned {resolved:?}"
        );
        let prefixed = storage.resolve_handle(prefix);
        assert!(
            !prefixed.ids.iter().any(|id| id == &doomed)
                && !prefixed.candidates.iter().any(|(id, _)| id == &doomed),
            "prefix handle returned {prefixed:?}"
        );
        assert!(storage.get_node(&doomed).unwrap().is_none());
        assert!(storage.get_receipt(&doomed).unwrap().is_none());
        assert!(
            !storage
                .get_connections_for_memory(&kept)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == doomed || edge.target_id == doomed)
        );
        assert!(
            !storage
                .hygiene_snapshot(None)
                .unwrap()
                .nodes
                .iter()
                .any(|node| node.id == doomed || node.content_preview.contains(marker))
        );

        fn show(result: Result<serde_json::Value, String>) -> String {
            match result {
                Ok(value) => value.to_string(),
                Err(error) => error,
            }
        }
        fn hidden(tool: &str, text: &str, marker: &str, id: &str) {
            assert!(!text.contains(marker), "{tool} leaked content: {text}");
            assert!(!text.contains(id), "{tool} leaked id: {text}");
        }
        fn content_hidden(tool: &str, text: &str, marker: &str) {
            assert!(!text.contains(marker), "{tool} leaked content: {text}");
        }

        let kept_get = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "get", "id": kept })),
        )
        .await
        .unwrap();
        assert!(kept_get.to_string().contains("KEPT_READ_VISIBLE"));
        hidden("memory get kept", &kept_get.to_string(), marker, &doomed);

        let doomed_get = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "get", "id": doomed })),
        )
        .await
        .unwrap();
        assert_eq!(doomed_get["found"], false);
        content_hidden("memory get", &doomed_get.to_string(), marker);

        let batch = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "get_batch", "ids": [doomed, kept] })),
        )
        .await
        .unwrap();
        assert_eq!(batch["found"], 1);
        content_hidden("memory get_batch", &batch.to_string(), marker);

        content_hidden(
            "memory state",
            &show(
                crate::tools::memory_unified::execute(
                    &storage,
                    &cognitive,
                    Some(serde_json::json!({ "action": "state", "id": doomed })),
                )
                .await,
            ),
            marker,
        );

        let handle_kept = crate::tools::recall::execute(
            &storage,
            &cognitive,
            &config,
            Some(serde_json::json!({ "handle": kept })),
        )
        .await
        .unwrap();
        assert!(handle_kept.to_string().contains("KEPT_READ_VISIBLE"));
        hidden("recall handle", &handle_kept.to_string(), marker, &doomed);
        content_hidden(
            "recall handle retired",
            &show(
                crate::tools::recall::execute(
                    &storage,
                    &cognitive,
                    &config,
                    Some(serde_json::json!({ "handle": doomed })),
                )
                .await,
            ),
            marker,
        );
        for (tool, args) in [
            (
                "recall lookup",
                serde_json::json!({ "mode": "lookup", "query": "unrelated fixture" }),
            ),
            (
                "recall reason",
                serde_json::json!({ "mode": "reason", "query": "unrelated fixture" }),
            ),
            (
                "recall contradictions",
                serde_json::json!({ "mode": "contradictions" }),
            ),
        ] {
            hidden(
                tool,
                &show(
                    crate::tools::recall::execute(&storage, &cognitive, &config, Some(args)).await,
                ),
                marker,
                &doomed,
            );
        }

        hidden(
            "receipt",
            &show(
                crate::tools::receipt::execute(
                    &storage,
                    Some(serde_json::json!({ "action": "get", "receipt_id": kept })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        content_hidden(
            "receipt retired",
            &show(
                crate::tools::receipt::execute(
                    &storage,
                    Some(serde_json::json!({ "action": "get", "receipt_id": doomed })),
                )
                .await,
            ),
            marker,
        );

        for (tool, args) in [
            (
                "graph chain",
                serde_json::json!({ "action": "chain", "from": kept, "to": kept }),
            ),
            (
                "graph associations",
                serde_json::json!({ "action": "associations", "from": kept }),
            ),
            (
                "graph bridges",
                serde_json::json!({ "action": "bridges", "from": kept, "to": kept }),
            ),
            (
                "graph neighbors",
                serde_json::json!({ "action": "neighbors", "memory_id": kept }),
            ),
            (
                "graph memory",
                serde_json::json!({ "action": "memory", "memory_id": kept }),
            ),
            (
                "graph memory_graph",
                serde_json::json!({ "action": "memory_graph", "center_id": kept }),
            ),
            (
                "graph recent",
                serde_json::json!({ "action": "recent", "limit": 5 }),
            ),
            (
                "graph never_composed",
                serde_json::json!({ "action": "never_composed", "limit": 5 }),
            ),
            ("graph predict", serde_json::json!({ "action": "predict" })),
        ] {
            hidden(
                tool,
                &show(crate::tools::graph_unified::execute(&storage, &cognitive, Some(args)).await),
                marker,
                &doomed,
            );
        }
        content_hidden(
            "graph chain to retired",
            &show(
                crate::tools::graph_unified::execute(
                    &storage,
                    &cognitive,
                    Some(serde_json::json!({ "action": "chain", "from": kept, "to": doomed })),
                )
                .await,
            ),
            marker,
        );

        for view in [
            "health",
            "retention",
            "timeline",
            "changelog",
            "stats",
            "coverage",
            "provenance",
            "tools",
        ] {
            hidden(
                &format!("memory_status {view}"),
                &show(
                    crate::tools::memory_status::execute(
                        &storage,
                        &cognitive,
                        &config,
                        Some(serde_json::json!({ "view": view })),
                    )
                    .await,
                ),
                marker,
                &doomed,
            );
        }

        hidden(
            "codebase",
            &show(
                crate::tools::codebase_unified::execute(
                    &storage,
                    &cognitive,
                    &config,
                    Some(serde_json::json!({ "action": "get_context", "codebase": "fixture" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "session_start",
            &show(
                crate::tools::session_context::execute(
                    &storage,
                    &cognitive,
                    &config,
                    Some(serde_json::json!({
                        "queries": ["unrelated fixture"],
                        "include_predictions": false,
                        "include_intentions": false
                    })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "dedup",
            &show(
                crate::tools::dedup::execute_unified(
                    &storage,
                    Some(&cognitive),
                    Some(serde_json::json!({ "action": "scan" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "maintain",
            &show(
                crate::tools::maintain::execute(
                    &storage,
                    &cognitive,
                    Some(serde_json::json!({ "action": "export", "format": "json" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "project",
            &show(
                crate::tools::project::execute(
                    &storage,
                    Some(serde_json::json!({ "action": "preview" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "intention",
            &show(
                crate::tools::intention_unified::execute(
                    &storage,
                    &cognitive,
                    Some(serde_json::json!({ "action": "list" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        #[cfg(feature = "connectors")]
        hidden(
            "source_sync",
            &show(
                crate::tools::source_sync::execute(
                    &storage,
                    Some(serde_json::json!({ "source": "gitlab", "repo": "a/b" })),
                )
                .await,
            ),
            marker,
            &doomed,
        );
        hidden(
            "causal_walk",
            &show(crate::tools::causal_walk::execute(&storage, None).await),
            marker,
            &doomed,
        );
        hidden(
            "selftest",
            &show(crate::tools::selftest::execute(&storage, None).await),
            marker,
            &doomed,
        );
        content_hidden(
            "forgotten_lesson",
            &show(
                crate::tools::forgotten_lesson::execute(
                    &storage,
                    Some(serde_json::json!({ "failure_id": doomed })),
                )
                .await,
            ),
            marker,
        );
    }

    #[test]
    fn merge_policy_roundtrip_survives_reopen_without_sqlite() {
        let dir = tempfile::TempDir::new().unwrap();
        let first = StrataMemory::open(dir.path()).unwrap();
        let unset = first.get_merge_policy().unwrap();
        assert!(unset.possible_threshold <= unset.match_threshold);
        assert!(!merge_policy_path(dir.path()).exists());
        let saved = vestige_core::MergePolicy::new(0.91, 0.99, true);
        first.set_merge_policy(saved).unwrap();
        let got = first.get_merge_policy().unwrap();
        assert!((got.match_threshold - 0.91).abs() < 1e-6);
        assert!((got.possible_threshold - 0.91).abs() < 1e-6);
        assert!(got.auto_apply);
        drop(first);

        let second = StrataMemory::open(dir.path()).unwrap();
        let again = second.get_merge_policy().unwrap();
        assert!((again.match_threshold - got.match_threshold).abs() < 1e-6);
        assert!((again.possible_threshold - got.possible_threshold).abs() < 1e-6);
        assert!(again.auto_apply);
        assert!(no_sqlite(dir.path()));
        assert_eq!(second.lock().node_count(), 0);
    }

    #[test]
    fn undo_appends_a_compensating_record_and_hides_it_from_reads() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = super::open(dir.path()).unwrap();
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: "undo me please".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().content,
            "undo me please"
        );
        let ops = storage.list_merge_operations(10).unwrap();
        let op = ops
            .iter()
            .find(|op| op.survivor_id.as_deref() == Some(node.id.as_str()))
            .expect("ingest is an undoable write");
        assert_eq!(op.op_type, "write");
        assert_eq!(op.status, "applied");
        let undone = storage.merge_undo(&op.id).unwrap();
        assert_eq!(undone.op_type, "undo");
        assert_eq!(undone.reverts_op_id.as_deref(), Some(op.id.as_str()));
        assert!(storage.get_node(&node.id).unwrap().is_none());
        assert!(
            storage
                .get_all_nodes(50, 0)
                .unwrap()
                .iter()
                .all(|listed| listed.content != "undo me please")
        );
        drop(storage);
        let reopened = super::open(dir.path()).unwrap();
        assert!(reopened.get_node(&node.id).unwrap().is_none());
        assert!(no_sqlite(dir.path()));
    }

    #[test]
    fn suppressing_a_woven_member_withdraws_its_composition_records() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = super::open(dir.path()).unwrap();
        let ingest = |content: &str| {
            storage
                .ingest(vestige_core::IngestInput {
                    content: content.into(),
                    node_type: "fact".into(),
                    ..vestige_core::IngestInput::default()
                })
                .unwrap()
                .id
        };
        let kept = ingest("KEPT_WOVEN_VISIBLE");
        let doomed = ingest("DOOMED_WOVEN_MARKER_41d9");
        let woven =
            super::ghostlink::weave(storage.as_ref(), &kept, &doomed, "helpful", None).unwrap();
        let record = woven["recordId"].as_str().unwrap().to_string();
        assert!(storage.get_node(&record).unwrap().is_some());

        let suppressed = storage.suppress_memory(&doomed).unwrap();
        assert_eq!(suppressed.id, doomed);

        // Neither the suppressed member's id nor its record surfaces in any
        // GhostLink read or in a plain read of the surviving member.
        let hidden = |what: &str, text: String| {
            assert!(
                !text.contains(&doomed),
                "{what} leaked the suppressed id: {text}"
            );
            assert!(
                !text.contains(&record),
                "{what} leaked the withdrawn record: {text}"
            );
        };
        for view in ["recent", "memory"] {
            hidden(
                view,
                super::ghostlink::inspect(storage.as_ref(), view, None, Some(&kept), 10)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|err| err),
            );
        }
        hidden(
            "associations",
            super::ghostlink::explore(storage.as_ref(), "associations", &kept, None, 10)
                .map(|value| value.to_string())
                .unwrap_or_else(|err| err),
        );
        // A chain asked for by the suppressed id echoes only the caller's own
        // argument: no path, no record, no content.
        let chain =
            super::ghostlink::explore(storage.as_ref(), "chain", &kept, Some(&doomed), 10).unwrap();
        assert_eq!(chain["steps"], json!([]), "{chain}");
        let text = chain.to_string();
        assert!(
            !text.contains(&record) && !text.contains("DOOMED_WOVEN_MARKER_41d9"),
            "{text}"
        );
        assert!(
            storage.get_node(&record).unwrap().is_none(),
            "record withdrawn"
        );
        let kept_node = storage.get_node(&kept).unwrap().expect("kept stays");
        hidden(
            "memory get kept",
            serde_json::to_string(&kept_node).unwrap(),
        );
        hidden(
            "connections of kept",
            format!("{:?}", storage.get_connections_for_memory(&kept).unwrap()),
        );

        // The withdrawal is a gated write that survives a reopen.
        drop(storage);
        let reopened = super::open(dir.path()).unwrap();
        assert!(reopened.get_node(&record).unwrap().is_none());
        assert!(reopened.get_node(&kept).unwrap().is_some());
    }

    fn token(fill: &str) -> String {
        format!("ghp_{}", fill.repeat(36))
    }

    fn plain_input() -> IngestInput {
        IngestInput {
            content: "Synthetic note with no credential".into(),
            node_type: "fact".into(),
            ..Default::default()
        }
    }

    #[test]
    fn ingest_gate_refuses_a_credential_in_any_stored_field() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = StrataMemory::open(dir.path()).unwrap();
        let secret = token("A");

        let mut in_tag = plain_input();
        in_tag.tags = vec!["safe".into(), secret.clone()];
        let mut in_source = plain_input();
        in_source.source = Some(format!("https://example.invalid/?t={secret}"));
        let mut in_envelope = plain_input();
        let mut envelope = SourceEnvelope::default();
        envelope.source_system = Some("tracker".into());
        envelope.source_id = Some(secret.clone());
        in_envelope.source_envelope = Some(envelope);
        let mut in_project = plain_input();
        let mut envelope = SourceEnvelope::default();
        envelope.source_system = Some("tracker".into());
        envelope.source_id = Some("42".into());
        envelope.source_project = Some(secret.clone());
        in_project.source_envelope = Some(envelope);
        let mut in_type = plain_input();
        in_type.node_type = secret.clone();

        for (field, input) in [
            ("tag", in_tag),
            ("source", in_source),
            ("envelope id", in_envelope),
            ("envelope project", in_project),
            ("node type", in_type),
        ] {
            let err = store.ingest(input).expect_err(field);
            assert!(
                matches!(err, StorageError::SecretDetected { .. }),
                "{field}: {err}"
            );
            assert!(
                !err.to_string().contains(&secret),
                "{field}: the refusal must not echo the credential"
            );
        }
        let err = store
            .ingest_in_scope(plain_input(), &secret)
            .expect_err("scope");
        assert!(
            matches!(err, StorageError::SecretDetected { .. }),
            "scope: {err}"
        );
        assert!(!err.to_string().contains(&secret));
        assert!(
            store.nodes().is_empty(),
            "a refused write must leave nothing in the log"
        );

        // The explicit override covers the memory's own text, never a scope
        // name, which is echoed by every read and write response.
        let mut allowed = plain_input();
        allowed.tags = vec![secret.clone()];
        store
            .ingest_with_secret_policy(allowed, SecretPolicy::AllowExplicitly)
            .expect("explicit override keeps working for tags");
        let err = store
            .ingest_in_scope_with_secret_policy(
                plain_input(),
                &secret,
                SecretPolicy::AllowExplicitly,
            )
            .expect_err("scope stays gated under the override");
        assert!(matches!(err, StorageError::SecretDetected { .. }));
        assert!(store.ingest(plain_input()).is_ok(), "clean input passes");
    }

    #[test]
    fn save_intention_refuses_a_credential_in_any_stored_field() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = StrataMemory::open(dir.path()).unwrap();
        let secret = token("B");

        let mut in_content = sample_intention("int-content");
        in_content.content = format!("rotate {secret} on Friday");
        let mut in_tags = sample_intention("int-tags");
        in_tags.tags = vec![secret.clone()];
        let mut in_notes = sample_intention("int-notes");
        in_notes.notes = Some(secret.clone());
        let mut in_scope = sample_intention("int-scope");
        in_scope.scope = Some(secret.clone());
        let mut in_trigger = sample_intention("int-trigger");
        in_trigger.trigger_data = format!("{{\"condition\":\"{secret}\"}}");
        let mut in_source = sample_intention("int-source");
        in_source.source_data = Some(secret.clone());

        for (field, record) in [
            ("content", in_content),
            ("tags", in_tags),
            ("notes", in_notes),
            ("scope", in_scope),
            ("trigger", in_trigger),
            ("source data", in_source),
        ] {
            let err = store.save_intention(&record).expect_err(field);
            assert!(
                matches!(err, StorageError::SecretDetected { .. }),
                "{field}: {err}"
            );
            assert!(
                !err.to_string().contains(&secret),
                "{field}: the refusal must not echo the credential"
            );
        }
        assert!(
            store.lock().intentions().is_empty(),
            "a refused intention must leave nothing in the log"
        );
        store
            .save_intention(&sample_intention("int-clean"))
            .expect("clean intention passes");
    }

    #[test]
    fn audit_records_reach_retired_memories_scopes_and_intentions() {
        let dir = tempfile::TempDir::new().unwrap();
        let secret = token("C");
        // Records written before the gate covered every field still sit in
        // the log. Seed them below the gate, as an older binary would have.
        let (retired_id, scoped_id) = {
            let mut raw = strata_store::StrataStore::open(dir.path()).unwrap();
            let node = |tags: Vec<String>| strata_store::IngestInput {
                content: "Synthetic note".into(),
                source: None,
                source_updated_at_ms: None,
                node_type: "fact".into(),
                tags,
                created_at_ms: Some(1),
                valid_from_ms: None,
                valid_until_ms: None,
            };
            let retired = raw
                .ingest_in_scope(node(vec![secret.clone()]), "user")
                .unwrap();
            let scoped = raw.ingest_in_scope(node(Vec::new()), &secret).unwrap();
            let mut intention = stored_intention(&sample_intention("int-audit"));
            intention.content = format!("rotate {secret}");
            raw.upsert_intentions(vec![intention]).unwrap();
            (retired, scoped)
        };
        let store = StrataMemory::open(dir.path()).unwrap();
        store.suppress_memory(&retired_id).unwrap();
        assert!(store.get_node(&retired_id).unwrap().is_none());

        let records = store.audit_records();
        let find = |id: &str| {
            records
                .iter()
                .find(|record| record.id == id)
                .unwrap_or_else(|| panic!("{id} missing from the audit set"))
        };
        let hit = |id: &str| {
            find(id)
                .texts
                .iter()
                .any(|text| !blocking_secrets(text).is_empty())
        };
        assert!(find(&retired_id).retired, "suppressed memory is audited");
        assert!(hit(&retired_id), "tag on a suppressed memory is scanned");
        assert!(hit(&scoped_id), "a scope name is scanned");
        assert!(hit("int-audit"), "an intention is scanned");
        assert_eq!(find("int-audit").kind, AuditKind::Intention);
    }

    // -----------------------------------------------------------------
    // INGEST V5 (Lane A): declarable supersedes, canonical duplicate
    // lookup, intent idempotency, node receipt resolution.
    // -----------------------------------------------------------------

    #[test]
    fn supersedes_declared_link_parses_and_records_new_to_old() {
        assert!(DeclaredLink::NAMES.contains(&"supersedes"));
        assert_eq!(DeclaredLink::parse("supersedes"), Some(DeclaredLink::Supersedes));
        assert_eq!(DeclaredLink::parse("nonsense"), None);

        let dir = tempfile::TempDir::new().unwrap();
        let storage = open(dir.path()).unwrap();
        let scope = "user";
        let old = storage
            .ingest_in_scope(
                IngestInput {
                    content: "the old truth".into(),
                    ..IngestInput::default()
                },
                scope,
            )
            .unwrap();
        let new = storage
            .ingest_in_scope(
                IngestInput {
                    content: "the full replacement".into(),
                    ..IngestInput::default()
                },
                scope,
            )
            .unwrap();

        // Caller target = the OLD memory.
        let links = vec![(DeclaredLink::Supersedes, old.id.clone())];
        check_links(storage.as_ref(), scope, &links).expect("target checks");
        let written = save_links(storage.as_ref(), &new.id, &links).expect("link written");
        assert_eq!(written.len(), 1);
        assert_eq!(written[0]["kind"], "supersedes");
        assert_eq!(written[0]["edge"], "supersedes");
        assert_eq!(written[0]["source"], json!(new.id));
        assert_eq!(written[0]["target"], json!(old.id));
        assert!(
            written[0]["receiptId"]
                .as_str()
                .is_some_and(|id| id.starts_with("eff-")),
            "each link carries its own effect receipt"
        );

        // The edge is persisted with direction new -supersedes-> old, and
        // the old memory stays live (an edge retires nothing).
        let edges = storage.get_connections_for_memory(&old.id).unwrap();
        let edge = edges
            .iter()
            .find(|edge| edge.link_type == "supersedes")
            .expect("supersedes edge persisted");
        assert_eq!(edge.source_id, new.id);
        assert_eq!(edge.target_id, old.id);
        assert!(storage.get_node(&old.id).unwrap().is_some());
    }

    #[test]
    fn latest_receipt_id_for_node_and_node_effect_seq_resolve_the_write() {
        let dir = tempfile::TempDir::new().unwrap();
        let memory = StrataMemory::open(dir.path()).unwrap();
        let (id, effect_seq) = {
            let mut store = memory.lock();
            store
                .ingest_in_scope_with_receipt(
                    strata_store::IngestInput {
                        content: "receipt resolution fixture".into(),
                        source: None,
                        source_updated_at_ms: None,
                        node_type: "fact".into(),
                        tags: Vec::new(),
                        created_at_ms: Some(0),
                        valid_from_ms: None,
                        valid_until_ms: None,
                    },
                    "user",
                )
                .unwrap()
        };
        assert_eq!(
            memory.latest_receipt_id_for_node(&id),
            Some(format!("eff-{effect_seq:016x}")),
            "the receipt id is eff- plus the admitting effect seq"
        );
        assert_eq!(memory.node_effect_seq(&id), Some(effect_seq));
        assert_eq!(memory.latest_receipt_id_for_node("mem-does-not-exist"), None);
        assert_eq!(memory.node_effect_seq("mem-does-not-exist"), None);
    }

    #[test]
    fn canonical_duplicate_and_intent_surface_answer_through_the_storage_trait() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = open(dir.path()).unwrap();
        let scope = "user";
        // Before any write: no duplicate, no intent record.
        assert_eq!(
            storage
                .find_duplicate_by_canonical_hash(scope, "dup wiring probe")
                .unwrap(),
            None
        );
        assert_eq!(storage.find_intent_record(scope, "run-42").unwrap(), None);

        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: "Dup Wiring Probe".into(),
                    ..IngestInput::default()
                },
                scope,
            )
            .unwrap();
        // NFC-free variants (case, zero-width, whitespace) resolve to it.
        assert_eq!(
            storage
                .find_duplicate_by_canonical_hash(scope, "  dup\u{200b} WIRING\tprobe ")
                .unwrap(),
            Some(node.id.clone())
        );
        // Another scope stays clean.
        assert_eq!(
            storage.find_duplicate_by_canonical_hash("other", "Dup Wiring Probe").unwrap(),
            None
        );

        // Intent entries roundtrip through the same surface and are
        // first-write-wins.
        let seq = storage.node_effect_seq(&node.id).expect("effect seq");
        storage
            .record_intent_entry(scope, "run-42", &node.id, seq, "digest-one")
            .unwrap();
        storage
            .record_intent_entry(scope, "run-42", &node.id, seq + 9, "digest-two")
            .unwrap();
        assert_eq!(
            storage.find_intent_record(scope, "run-42").unwrap(),
            Some((node.id.clone(), seq, "digest-one".to_string()))
        );

        // And they survive a reopen of the same directory.
        drop(storage);
        let reopened = open(dir.path()).unwrap();
        assert_eq!(
            reopened.find_intent_record(scope, "run-42").unwrap(),
            Some((node.id, seq, "digest-one".to_string()))
        );
    }
}
