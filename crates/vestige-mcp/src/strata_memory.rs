//! `MemoryStore` adapter over [`strata_store::StrataStore`].
//!
//! Writes go through the store's PROPOSE → GATE → EFFECT path. Reads are the
//! replayed registry. Similarity (embeddings, cosine, BM25, FTS, Jaccard,
//! keyword or name match) is refused; a link exists only when the log
//! recorded an edge.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use strata_store::VALID_FOREVER_MS;
use vestige_core::storage::{
    CompositionEventRecord, ConnectionRecord as VestigeEdge, ConsolidationHistoryRecord,
    CoverageSnapshot, HandleKind, HandleResolution, HealthStatus, HygieneNodeSummary,
    HygieneSnapshot, MemoryEdge, MemoryRecord, MemoryStoreError, MemoryStoreResult,
    MemoryStoreSend, ModelSignature, NeverComposedCandidate, ReceiptAttestationStatus,
    SchedulingState, SearchQuery, StateTransitionRecord, Storage, StorageError, StoreStats,
    WalCheckpointMode, WalCheckpointStatus, HANDLE_REQUIRED_DETAIL, MAX_CANDIDATES,
};
use vestige_core::{
    scan_secrets, ConsolidationResult, DecayRisk, IngestInput, KnowledgeNode, MemoryStats, Receipt,
    SecretPolicy,
};

const Q32_SCALE: f64 = 4294967296.0;
const RECEIPT_PREFIX: &str = "eff-";

/// The durable directory this process opened is a Strata log, not a SQLite file.
pub fn is_strata_backend(storage: &Storage) -> bool {
    storage.db_path().file_name().and_then(|name| name.to_str()) == Some("log")
}

/// Open (or create) a Strata log under `dir`. Creates no SQLite file.
pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Storage>, StorageError> {
    Ok(Arc::new(StrataMemory::open(dir)?))
}

pub struct StrataMemory {
    data_dir: PathBuf,
    log_dir: PathBuf,
    store: Mutex<strata_store::StrataStore>,
    actor: Mutex<Option<String>>,
}

impl StrataMemory {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, StorageError> {
        let data_dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;
        let store = strata_store::StrataStore::open(&data_dir).map_err(map_store)?;
        Ok(Self {
            log_dir: data_dir.join("log"),
            data_dir,
            store: Mutex::new(store),
            actor: Mutex::new(None),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, strata_store::StrataStore> {
        self.store
            .lock()
            .unwrap_or_else(|err| panic!("strata memory lock poisoned: {err}"))
    }

    fn nodes(&self) -> Vec<strata_store::NodeRecord> {
        self.lock().nodes()
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

fn pending(op: &str) -> StorageError {
    StorageError::Init(format!(
        "pending_strata: {op} is not admitted on the Strata log yet"
    ))
}

fn sim_async(op: &str) -> MemoryStoreError {
    MemoryStoreError::Init(similarity(op).to_string())
}

fn pending_async(op: &str) -> MemoryStoreError {
    MemoryStoreError::Init(pending(op).to_string())
}

/// RETIRE `id` under an existing named rule.
///
/// `SupersedeNode` requires a successor, so this admits a content-free anchor
/// and retires that anchor in the same call. Both records stay on the log.
/// The returned receipt is the one that retires `id`.
fn retire_live(
    store: &mut strata_store::StrataStore,
    id: &str,
    rule_id: &'static str,
    confirm: bool,
) -> Result<strata_store::RetireReceipt, StorageError> {
    let record = store
        .get_node(id)
        .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
    if !record.is_live() {
        return Err(StorageError::Init(format!("node {id} is already retired")));
    }
    let scope = record.scope.clone();
    let anchor = store
        .ingest_in_scope(
            strata_store::IngestInput {
                content: ".".into(),
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
        rule_id: Some(rule_id.to_string()),
        confirm,
    };
    let receipt = store.retire(id, &anchor, &ctx).map_err(map_store)?;
    if receipt.rule_id != Some(rule_id) {
        return Err(StorageError::Init(format!(
            "{rule_id} retire was not admitted under {rule_id}"
        )));
    }
    store.retire(&anchor, id, &ctx).map_err(map_store)?;
    Ok(receipt)
}

fn purge_report(
    id: &str,
    deleted: bool,
    receipt_id: Option<String>,
) -> vestige_core::storage::PurgeReport {
    vestige_core::storage::PurgeReport {
        memory_id: id.to_string(),
        deleted,
        deleted_at: Utc::now(),
        edges_pruned: 0,
        insights_rewritten: 0,
        insights_deleted: 0,
        children_orphaned: 0,
        unlearning_scope: vestige_core::storage::UnlearningScope::LegacyAuditedPurge,
        unlearning_verdict: vestige_core::storage::UnlearningVerdict::Incomplete,
        unlearning_claim_boundary: "Retired; can't be retrieved.",
        receipt_id,
    }
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

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or(DateTime::UNIX_EPOCH)
}

fn q32(q: i64) -> f64 {
    q as f64 / Q32_SCALE
}

fn receipt_id_for(seq: u64) -> String {
    format!("{RECEIPT_PREFIX}{seq:016x}")
}

fn parse_receipt_seq(receipt_id: &str) -> Option<u64> {
    let rest = receipt_id.strip_prefix(RECEIPT_PREFIX)?;
    u64::from_str_radix(rest, 16).ok()
}

fn blocking_secrets(text: &str) -> Vec<String> {
    scan_secrets(text)
        .into_iter()
        .filter(|finding| finding.blocks_ingestion())
        .map(|finding| finding.kind.to_string())
        .collect()
}

fn to_store_input(input: &IngestInput) -> strata_store::IngestInput {
    strata_store::IngestInput {
        content: input.content.clone(),
        node_type: input.node_type.clone(),
        tags: input.tags.clone(),
        created_at_ms: Some(Utc::now().timestamp_millis()),
        valid_from_ms: input.valid_from.map(|t| t.timestamp_millis()),
        valid_until_ms: input.valid_until.map(|t| t.timestamp_millis()),
    }
}

fn project_node(store: &strata_store::StrataStore, record: &strata_store::NodeRecord) -> KnowledgeNode {
    let card = store.card_state(&record.id);
    let retrieval = store.retrievability(&record.id).ok().flatten().unwrap_or(0.0);
    let mut node = KnowledgeNode::default();
    node.id = record.id.clone();
    node.content = record.content.clone();
    node.node_type = record.node_type.clone();
    node.created_at = ms_to_dt(record.created_at_ms);
    node.updated_at = node.created_at;
    node.last_accessed = node.created_at;
    node.tags = record.tags.clone();
    node.valid_from = Some(ms_to_dt(record.valid_from_ms));
    node.valid_until = (record.valid_until_ms != VALID_FOREVER_MS).then(|| ms_to_dt(record.valid_until_ms));
    // Kernel retrievability is the only strength the log can justify.
    node.stability = card.as_ref().map(|c| q32(c.stability_q)).unwrap_or(0.0);
    node.difficulty = card.as_ref().map(|c| q32(c.difficulty_q)).unwrap_or(0.0);
    node.reps = card.as_ref().map(|c| c.review_count as i32).unwrap_or(0);
    node.lapses = card.as_ref().map(|c| c.lapse_count as i32).unwrap_or(0);
    node.storage_strength = retrieval;
    node.retrieval_strength = retrieval;
    node.retention_strength = retrieval;
    node.has_embedding = Some(false);
    node
}

fn retrievable(record: &strata_store::NodeRecord) -> bool {
    record.superseded_by.is_none()
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

    async fn remove_edge(
        &self,
        _source: uuid::Uuid,
        _target: uuid::Uuid,
    ) -> MemoryStoreResult<()> {
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

    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        None
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
            .unwrap_or_else(|err| panic!("strata actor lock poisoned: {err}")) = Some(did.to_string());
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
        // Origins keep retired ids. Handle recall only sees nodes with no successor.
        let ids: Vec<String> = store
            .origins()
            .into_iter()
            .filter(|(id, _)| store.get_node(id).is_some_and(|record| retrievable(&record)))
            .map(|(id, _)| id)
            .collect();
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
        self.lock().backup_to(path).map_err(map_store)
    }

    fn checkpoint_wal(&self, _mode: WalCheckpointMode) -> Result<WalCheckpointStatus, StorageError> {
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
        self.ingest_in_scope_with_secret_policy(input, vestige_core::DEFAULT_MEMORY_SCOPE, SecretPolicy::Reject)
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
        if policy != SecretPolicy::AllowExplicitly {
            let kinds = blocking_secrets(&input.content);
            if !kinds.is_empty() {
                return Err(StorageError::SecretDetected { kinds });
            }
        }
        if input.content.trim().is_empty() {
            return Err(StorageError::Init("content must not be empty".into()));
        }
        let mut store = self.lock();
        let id = store
            .ingest_in_scope(to_store_input(&input), scope)
            .map_err(map_store)?;
        let record = store
            .get_node(&id)
            .ok_or_else(|| StorageError::NotFound(id.clone()))?;
        Ok(project_node(&store, &record))
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
        Ok(self.lock().get_node(id).is_some_and(|record| {
            retrievable(&record) && record.scope == scope
        }))
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
        self.lock().save_connection(&edge).map_err(map_store)
    }

    fn superseded_node_ids(&self) -> Result<HashSet<String>, StorageError> {
        Ok(self
            .lock()
            .supersession_pairs()
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    fn supersession_pairs(&self) -> Result<Vec<(String, String)>, StorageError> {
        Ok(self.lock().supersession_pairs())
    }

    fn get_stats(&self) -> Result<MemoryStats, StorageError> {
        let store = self.lock();
        let nodes: Vec<_> = store.nodes().into_iter().filter(retrievable).collect();
        let strengths: Vec<f64> = nodes
            .iter()
            .map(|record| store.retrievability(&record.id).ok().flatten().unwrap_or(0.0))
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
        let Some((id, seq)) = lookup_origin(&store, receipt_id) else {
            return Ok(None);
        };
        let trust = store.retrievability(&id).ok().flatten().unwrap_or(0.0);
        Ok(Some(Receipt {
            receipt_id: receipt_id_for(seq),
            retrieved: vec![id],
            suppressed: Vec::new(),
            activation_path: Vec::new(),
            trust_floor: trust,
            decay_risk: DecayRisk::from_trust_floor(trust),
            mutations: Vec::new(),
            evidence: None,
            actor: None,
        }))
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
        Ok(lookup_origin(&store, receipt_id).map(|_| ReceiptAttestationStatus::LegacyUnsigned))
    }

    fn create_context_ablation_replay(
        &self,
        _source_receipt_id: &str,
        _withheld_slots: &[String],
    ) -> Result<vestige_core::storage::DurableCounterfactualReplay, StorageError> {
        Err(pending("create_context_ablation_replay"))
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

    fn code_anchors_for_nodes(
        &self,
        _node_ids: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<vestige_core::codebase::CodeAnchor>>, StorageError>
    {
        // Anchors are not admitted on this log, so the recorded set is empty.
        Ok(std::collections::HashMap::new())
    }

    fn projection_candidates(
        &self,
        _scope: &str,
        _min_retention: f64,
        _limit: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        Err(pending("projection_candidates"))
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
        Err(similarity("concrete_search_filtered"))
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
        self.never_composed(None, limit, tag_filter)
    }

    fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        self.never_composed(scope.or(Some(vestige_core::DEFAULT_MEMORY_SCOPE)), limit, tag_filter)
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
                            || tags.iter().any(|tag| record.tags.iter().any(|stored| stored == tag))
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
        for edge in store.edges().into_iter().filter(|edge| edge_visible(&store, edge)) {
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

    fn promote_memory(&self, _id: &str) -> Result<KnowledgeNode, StorageError> {
        Err(pending("promote_memory"))
    }

    fn promote_memory_as_actor(
        &self,
        _id: &str,
        _claimed_role: Option<&str>,
        _tool: &str,
    ) -> Result<vestige_core::storage::ActorMutationOutcome, StorageError> {
        Err(pending("promote_memory_as_actor"))
    }

    fn demote_memory(&self, _id: &str) -> Result<KnowledgeNode, StorageError> {
        Err(pending("demote_memory"))
    }

    fn demote_memory_as_actor(
        &self,
        _id: &str,
        _claimed_role: Option<&str>,
        _tool: &str,
    ) -> Result<vestige_core::storage::ActorMutationOutcome, StorageError> {
        Err(pending("demote_memory_as_actor"))
    }

    fn purge_node(
        &self,
        id: &str,
        reason: Option<&str>,
    ) -> Result<vestige_core::storage::PurgeReport, StorageError> {
        // Reason stays off the log. Real erasure is a later change.
        if let Some(reason) = reason {
            tracing::info!(memory_id = %id, reason, "retiring memory");
        }
        let mut store = self.lock();
        let Some(record) = store.get_node(id) else {
            return Ok(purge_report(id, false, None));
        };
        if !record.is_live() {
            return Ok(purge_report(id, false, None));
        }
        let receipt = retire_live(&mut store, id, strata_store::RULE_PURGE, true)?;
        Ok(purge_report(id, true, Some(receipt.receipt_id)))
    }

    fn delete_node(&self, _id: &str) -> Result<bool, StorageError> {
        Err(pending("delete_node"))
    }

    fn get_active_intentions(
        &self,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn get_active_intentions_in_scope(
        &self,
        _scope: &str,
    ) -> Result<Vec<vestige_core::storage::IntentionRecord>, StorageError> {
        Ok(Vec::new())
    }

    fn save_intention(
        &self,
        _intention: &vestige_core::storage::IntentionRecord,
    ) -> Result<(), StorageError> {
        Err(pending("save_intention"))
    }

    fn suppress_memory(&self, id: &str) -> Result<KnowledgeNode, StorageError> {
        let mut store = self.lock();
        let receipt = retire_live(&mut store, id, strata_store::RULE_SUPPRESS, false)?;
        // The receipt is an in-memory handoff for the tool. It is not a node field
        // on the log. The record stays, with superseded_by set, and reads drop it.
        let mut node = KnowledgeNode::default();
        node.id = id.to_string();
        node.suppression_count = 1;
        node.suppressed_at = Some(Utc::now());
        node.source = Some(receipt.receipt_id);
        Ok(node)
    }

    fn update_node_content(&self, _id: &str, _new_content: &str) -> Result<(), StorageError> {
        Err(pending("update_node_content"))
    }
}

impl StrataMemory {
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

    fn never_composed(
        &self,
        scope: Option<&str>,
        limit: i32,
        tag_filter: Option<&[String]>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        let Some(scope) = scope else {
            return Ok(Vec::new());
        };
        let limit = usize::try_from(limit).unwrap_or(0);
        let store = self.lock();
        let records = store.nodes();
        let pairs = store.get_never_composed(scope, limit.saturating_mul(4).max(limit));
        let mut out = Vec::new();
        for (first, second) in pairs {
            let Some(a) = records.iter().find(|record| record.id == first) else {
                continue;
            };
            let Some(b) = records.iter().find(|record| record.id == second) else {
                continue;
            };
            if !retrievable(a) || !retrievable(b) {
                continue;
            }
            if let Some(tags) = tag_filter.filter(|tags| !tags.is_empty()) {
                let has = |record: &strata_store::NodeRecord| {
                    record.tags.iter().any(|tag| tags.iter().any(|want| want == tag))
                };
                if !has(a) || !has(b) {
                    continue;
                }
            }
            out.push(NeverComposedCandidate {
                first_id: first,
                second_id: second,
                score: 0.0,
                novelty_score: 0.0,
                bridge_score: 0.0,
                trust_score: 0.0,
                outcome_score_adjustment: 0.0,
                shared_tags: Vec::new(),
                boundary_tags: Vec::new(),
                shared_terms: Vec::new(),
                prior_outcomes: Vec::new(),
                outcome_signal: String::new(),
                first_node_type: a.node_type.clone(),
                second_node_type: b.node_type.clone(),
                first_preview: a.content.chars().take(140).collect(),
                second_preview: b.content.chars().take(140).collect(),
                reason: "no recorded edge".into(),
                composition_question: String::new(),
            });
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }
}

fn lookup_origin(store: &strata_store::StrataStore, receipt_or_node: &str) -> Option<(String, u64)> {
    let found = if let Some(seq) = parse_receipt_seq(receipt_or_node) {
        store
            .origins()
            .into_iter()
            .find(|(_, origin)| *origin == seq)
    } else {
        store
            .origin_seq(receipt_or_node)
            .map(|seq| (receipt_or_node.to_string(), seq))
    }?;
    store
        .get_node(&found.0)
        .is_some_and(|record| retrievable(&record))
        .then_some(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

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

    #[tokio::test]
    async fn purge_with_confirm_returns_a_receipt_and_hides_content() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = open(dir.path()).unwrap();
        let marker = "PURGE_MARKER_SECRET";
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: marker.into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        let kept = storage
            .ingest_in_scope(
                IngestInput {
                    content: "KEEP_VISIBLE".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        let before = log_files(dir.path());
        let refused = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "purge", "id": node.id })),
        )
        .await
        .unwrap_err();
        assert!(refused.contains("confirm=true"), "{refused}");
        assert_eq!(log_files(dir.path()), before);
        assert_eq!(storage.get_node(&node.id).unwrap().unwrap().content, marker);

        let purged = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({
                "action": "purge",
                "id": node.id,
                "confirm": true
            })),
        )
        .await
        .unwrap();
        assert_eq!(purged["rule"], "purge");
        assert_eq!(purged["nodeId"], node.id);
        let receipt_id = purged["receiptId"].as_str().unwrap();
        assert!(receipt_id.starts_with("eff-"), "{receipt_id}");
        assert!(storage.get_node(&node.id).unwrap().is_none());
        assert!(storage.resolve_handle(&node.id).ids.is_empty());
        assert_eq!(
            storage.get_node(&kept.id).unwrap().unwrap().content,
            "KEEP_VISIBLE"
        );
        let listed = serde_json::to_string(&storage.get_all_nodes(50, 0).unwrap()).unwrap();
        assert!(!listed.contains(marker), "{listed}");
        let ranged = serde_json::to_string(
            &storage
                .query_time_range(None, None, 50, None, None)
                .unwrap(),
        )
        .unwrap();
        assert!(!ranged.contains(marker), "{ranged}");
        let hygiene = storage.hygiene_snapshot(None).unwrap();
        assert!(
            hygiene
                .nodes
                .iter()
                .all(|node| !node.content_preview.contains(marker))
        );
        let got = crate::tools::memory_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "get", "id": node.id })),
        )
        .await
        .unwrap();
        assert_eq!(got["message"], "retired, can't be retrieved");
        assert!(!got.to_string().contains(marker));

        let seq = u64::from_str_radix(receipt_id.trim_start_matches("eff-"), 16).unwrap();
        drop(storage);
        let reopened = strata_store::StrataStore::open(dir.path()).unwrap();
        let again = reopened.retire_receipt(seq).unwrap();
        assert_eq!(again.rule_id, Some(strata_store::RULE_PURGE));
        assert_eq!(again.receipt_id, receipt_id);
        let stored = reopened.get_node(&node.id).unwrap();
        assert!(stored.superseded_by.is_some());
        assert_eq!(stored.content, marker);
    }

    #[tokio::test]
    async fn suppress_returns_a_receipt_and_hides_content() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = open(dir.path()).unwrap();
        let marker = "SUPPRESS_MARKER_SECRET";
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: marker.into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        let kept = storage
            .ingest_in_scope(
                IngestInput {
                    content: "KEEP_VISIBLE".into(),
                    ..IngestInput::default()
                },
                "user",
            )
            .unwrap();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: kept.id.clone(),
                target_id: node.id.clone(),
                strength: 1.0,
                link_type: "derived_from".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();
        let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        cognitive.lock().await.hydrate(&storage);
        let before = log_files(dir.path());
        let missing = crate::tools::suppress::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "id": "mem-00000000000000ab" })),
        )
        .await
        .unwrap_err();
        assert!(
            missing.contains("not found") || missing.contains("Not found"),
            "{missing}"
        );
        assert_eq!(log_files(dir.path()), before, "unknown id wrote the log");

        let suppressed = crate::tools::suppress::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "id": node.id, "reason": "fixture" })),
        )
        .await
        .unwrap();
        assert_eq!(suppressed["success"], true);
        assert_eq!(suppressed["rule"], "suppress");
        assert_eq!(suppressed["id"], node.id);
        let receipt_id = suppressed["receiptId"].as_str().unwrap();
        assert!(receipt_id.starts_with("eff-"), "{receipt_id}");
        assert!(!suppressed.to_string().contains(marker));
        assert!(storage.get_node(&node.id).unwrap().is_none());
        assert!(
            storage
                .get_connections_for_memory(&kept.id)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            storage.get_node(&kept.id).unwrap().unwrap().content,
            "KEEP_VISIBLE"
        );

        let after = log_files(dir.path());
        let associations = crate::tools::graph_unified::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "action": "associations", "from": kept.id })),
        )
        .await
        .unwrap();
        assert!(!associations.to_string().contains(&node.id), "{associations}");
        assert!(!associations.to_string().contains(marker), "{associations}");

        let second = crate::tools::suppress::execute(
            &storage,
            &cognitive,
            Some(serde_json::json!({ "id": node.id })),
        )
        .await
        .unwrap_err();
        assert!(second.contains("already retired"), "{second}");
        assert_eq!(log_files(dir.path()), after, "second suppress wrote the log");

        let seq = u64::from_str_radix(receipt_id.trim_start_matches("eff-"), 16).unwrap();
        drop(storage);
        let reopened = strata_store::StrataStore::open(dir.path()).unwrap();
        let again = reopened.retire_receipt(seq).unwrap();
        assert_eq!(again.rule_id, Some(strata_store::RULE_SUPPRESS));
        assert_eq!(again.receipt_id, receipt_id);
        assert_eq!(reopened.get_node(&node.id).unwrap().content, marker);
        drop(reopened);
        let memory = open(dir.path()).unwrap();
        assert!(memory.get_node(&node.id).unwrap().is_none());
        assert_eq!(
            memory.get_node(&kept.id).unwrap().unwrap().content,
            "KEEP_VISIBLE"
        );
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
        assert!(resolved.ids.is_empty(), "exact handle returned {resolved:?}");
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
                &show(crate::tools::recall::execute(&storage, &cognitive, &config, Some(args)).await),
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
}
