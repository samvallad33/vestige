//! # strata-bridge — the STRATA Causal Proof Engine behind the trait wall
//!
//! Implements vestige-core's storage seam ([`MemoryStoreSend`], which the
//! blanket impls adapt into `LocalMemoryStore` and the dyn-compatible
//! `MemoryStore`) over `strata_store::StrataStore`, so a legacy-sqlite-free
//! build can boot on STRATA as its runtime store.
//!
//! ## Direction of the dependency
//!
//! `vestige-core` must not depend on strata crates. This crate is the place
//! where the two type families meet: it depends on BOTH and owns every
//! conversion (`KnowledgeNode` <-> `NodeRecord`, `ConnectionRecord` <-> the
//! milli-integer strata mirror, `IngestInput` <-> its float-free mirror).
//!
//! ## Threading model
//!
//! `StrataStore` v1 is single-writer, single-threaded, `!Send` (the gate-log
//! cache is `Rc<RefCell<..>>`; the log's `strata.lock` enforces one store per
//! directory). The trait object must be `Send + Sync + 'static`. The bridge
//! resolves that the only honest way: [`StrataBackend`] owns a dedicated OS
//! thread that exclusively holds the `StrataStore`; every trait call ships
//! one closure to that thread and blocks on the reply channel. Round-trips
//! are microseconds (uncontended channel + in-memory derived maps); writes
//! additionally durably append to the log, exactly as the store demands.
//!
//! A panic inside a job never kills the worker thread (each job runs under
//! `catch_unwind`); the caller sees [`BridgeError::JobPanicked`] instead.
//!
//! ## Mapping policy (loud failures, no stubs)
//!
//! Only the strata-store v1 surface is implemented: ingest / get_node /
//! get_all_nodes_in_scope / set_created_at / save_connection /
//! get_connections_for_memory / supersession_pairs / never-composed
//! candidates / review (`mark_reviewed`) / `backup_to`, plus the
//! boot-critical `data_dir` / `db_path` / `sidecar_dir` /
//! `process_actor_did` and truthful async lifecycle answers. Every other
//! method keeps the trait's loud-fail default (`StorageError::Init("... is
//! not implemented by this backend")` / `unimplemented!`) — the bridge never
//! fabricates a silent success.
//!
//! ## Coercion notes (vestige-core type -> strata-store type)
//!
//! * `created_at` is caller data in strata (no hidden clock). Plain `ingest`
//!   passes `None` (deterministic 0 = epoch); SQLite's `Utc::now()` stamping
//!   is deliberately NOT imitated. Use `set_created_at` for real timestamps.
//! * Edge strength `f64` <-> `strength_milli: i64` (rounded at the milli
//!   boundary; strata persists no floats).
//! * `valid_from == created_at` reads back as `None` (strata normalizes an
//!   absent window start to the creation time; the distinction is not
//!   persisted). `VALID_FOREVER_MS` reads back as `valid_until: None`.
//! * Bjork `storage_strength` is not modeled by the kernel; it reads back as
//!   the constant 1.0 when a card exists. FSRS `stability`/`difficulty`
//!   dequantize from the kernel's Q32.32 card (`q / 2^32`, exact);
//!   `retrieval_strength`/`retention_strength` are the derived
//!   retrievability at the log head.
//! * `sentiment_score` / `sentiment_magnitude` / `source` /
//!   `source_envelope` on ingest have no strata representation and are
//!   dropped at the boundary.
//! * The bridge compiles against vestige-core's storage seam, whose sync
//!   product surface references types that core re-exports only when the
//!   workspace graph unifies `legacy-sqlite` on (the trait file itself
//!   imports them ungated). Standalone legacy-free core builds are a
//!   pre-existing breakage at this base, not something this crate widens.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use strata_kernel::fsrs::CardState;
use strata_store::{
    ConnectionRecord as StrataConnection, IngestInput as StrataIngestInput, NodeRecord, StoreError,
    StrataStore, VALID_FOREVER_MS,
};
use vestige_core::storage::{
    ConnectionRecord, HealthStatus, MemoryStore, MemoryStoreError as AsyncError,
    MemoryStoreResult as AsyncResult, MemoryStoreSend, ModelSignature, NeverComposedCandidate,
    StorageError, StoreStats, DEFAULT_MEMORY_SCOPE,
};
use vestige_core::{IngestInput, KnowledgeNode, Rating};

// ---------------------------------------------------------------------------
// Thread plumbing
// ---------------------------------------------------------------------------

/// One unit of work for the store thread: run against the exclusively-held
/// store. Replies travel on a per-call channel captured by the closure.
type Job = Box<dyn FnOnce(&mut StrataStore) + Send + 'static>;

/// Errors of the bridge itself (store errors are mapped separately).
#[derive(Debug)]
enum BridgeError {
    /// The store thread is gone (store open failed or the channel closed).
    ThreadGone(String),
    /// The job panicked on the store thread; the thread survives.
    JobPanicked,
}

impl BridgeError {
    fn into_store_error(self) -> StorageError {
        match self {
            BridgeError::ThreadGone(reason) => {
                StorageError::Init(format!("strata store thread unavailable: {reason}"))
            }
            BridgeError::JobPanicked => {
                StorageError::Init("strata store job panicked on the store thread".into())
            }
        }
    }

    fn into_async_error(self) -> AsyncError {
        match self {
            BridgeError::ThreadGone(reason) => {
                AsyncError::Init(format!("strata store thread unavailable: {reason}"))
            }
            BridgeError::JobPanicked => {
                AsyncError::Init("strata store job panicked on the store thread".into())
            }
        }
    }
}

/// The `Send + Sync + 'static` handle onto a dedicated-thread `StrataStore`.
pub struct StrataBackend {
    dir: PathBuf,
    /// The durable artifact path (`<dir>/log`); the trait's `db_path` must
    /// return a reference, so it is owned here.
    db_path: PathBuf,
    tx: Mutex<mpsc::Sender<Job>>,
}

impl StrataBackend {
    /// Open (or create) a STRATA store under `dir` on a dedicated thread and
    /// return the trait-object handle. Fails when the store's open-time
    /// verification (replay + checkpoint chain anchor) fails — fail-closed,
    /// same as `StrataStore::open`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>, String> {
        let dir = dir.as_ref().to_path_buf();
        let (tx, rx) = mpsc::channel::<Job>();
        let (open_tx, open_rx) = mpsc::channel::<Result<(), String>>();
        let thread_dir = dir.clone();
        std::thread::Builder::new()
            .name("strata-store".into())
            .spawn(move || {
                let mut store = match StrataStore::open(&thread_dir) {
                    Ok(store) => {
                        let _ = open_tx.send(Ok(()));
                        store
                    }
                    Err(e) => {
                        // Drain nothing; signal the failure and exit. Later
                        // sends on `tx` will fail loudly.
                        let _ = open_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    // A panicking job must not kill the thread: the store is
                    // process-global state (one writer per directory).
                    let _ =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&mut store)));
                }
            })
            .map_err(|e| format!("failed to spawn the strata store thread: {e}"))?;
        let db_path = dir.join("log");
        match open_rx.recv() {
            Ok(Ok(())) => Ok(Arc::new(Self {
                dir,
                db_path,
                tx: Mutex::new(tx),
            })),
            Ok(Err(open_failure)) => Err(open_failure),
            Err(_) => Err("strata store thread died before opening".into()),
        }
    }

    /// Ship one closure to the store thread and block for its reply.
    fn call<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut StrataStore) -> R + Send + 'static,
    ) -> Result<R, BridgeError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let job: Job = Box::new(move |store| {
            // Send failures (caller dropped / channel closed) are irrelevant
            // to the store thread; it keeps serving.
            let _ = reply_tx.send(f(store));
        });
        {
            let guard = self
                .tx
                .lock()
                .map_err(|_| BridgeError::ThreadGone("sender lock poisoned".into()))?;
            guard.send(job).map_err(|_| {
                BridgeError::ThreadGone("store thread exited (open failed?)".into())
            })?;
        }
        reply_rx.recv().map_err(|_| BridgeError::JobPanicked)
    }

    /// The store's directory (its `data_dir`).
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

// ---------------------------------------------------------------------------
// Constructor hook entry (the fn pointer vestige-core's STORE_CONSTRUCTOR
// installs; signature fixed by the hook: fn(PathBuf) -> Result<_, String>)
// ---------------------------------------------------------------------------

/// Resolve the STRATA directory for a constructor call.
///
/// `$VESTIGE_STRATA_DIR` wins outright. Otherwise the incoming path is
/// interpreted like the SQLite constructor path (`<data dir>/vestige.db` or a
/// bare data dir): a `vestige.db` file name (or any file-looking path) maps
/// to its parent's `strata/` subdir; a directory maps to `<dir>/strata`.
pub fn strata_dir_for(path: &Path) -> PathBuf {
    if let Some(env) = std::env::var_os("VESTIGE_STRATA_DIR") {
        if !env.is_empty() {
            return PathBuf::from(env);
        }
    }
    let looks_like_db_file =
        path.file_name().is_some_and(|n| n == "vestige.db") || path.extension().is_some();
    if looks_like_db_file {
        path.parent().unwrap_or(Path::new(".")).join("strata")
    } else {
        path.join("strata")
    }
}

/// The store constructor installable into
/// `vestige_core::storage::STORE_CONSTRUCTOR`. Opens a `StrataBackend` over
/// the STRATA directory resolved by [`strata_dir_for`].
pub fn strata_constructor(path: PathBuf) -> Result<Arc<dyn MemoryStore>, String> {
    let dir = strata_dir_for(&path);
    let backend = StrataBackend::open(&dir)?;
    Ok(backend as Arc<dyn MemoryStore>)
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Dequantize a Q32.32 fixed-point integer (the kernel's canonical format)
/// to `f64`. Division by 2^32 is exact for every input.
fn from_q32_32(q: i64) -> f64 {
    q as f64 / 4_294_967_296.0
}

fn ms_to_datetime(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or(DateTime::UNIX_EPOCH)
}

fn datetime_to_ms(dt: &DateTime<Utc>) -> i64 {
    dt.timestamp_millis()
}

fn map_store_error(e: StoreError) -> StorageError {
    match e {
        StoreError::NotFound(what) => StorageError::NotFound(what),
        StoreError::InvalidInput(what) => {
            StorageError::Init(format!("strata: invalid input: {what}"))
        }
        other => StorageError::Init(format!("strata: {other}")),
    }
}

/// Build the product node view of one strata record, folding in its FSRS
/// card and derived retrievability when present.
fn node_from_record(
    record: &NodeRecord,
    card: Option<&CardState>,
    retrievability: Option<f64>,
) -> KnowledgeNode {
    let mut node = KnowledgeNode::new(record.content.clone());
    node.id = record.id.clone();
    node.node_type = record.node_type.clone();
    node.tags = record.tags.clone();
    let created = ms_to_datetime(record.created_at_ms);
    node.created_at = created;
    node.updated_at = created;
    node.last_accessed = created;
    if record.valid_from_ms != record.created_at_ms {
        node.valid_from = Some(ms_to_datetime(record.valid_from_ms));
    }
    if record.valid_until_ms != VALID_FOREVER_MS {
        node.valid_until = Some(ms_to_datetime(record.valid_until_ms));
    }
    if let Some(card) = card {
        node.stability = from_q32_32(card.stability_q);
        node.difficulty = from_q32_32(card.difficulty_q);
        node.reps = card.review_count as i32;
        node.lapses = card.lapse_count as i32;
        // Bjork storage strength is not modeled by the kernel; a card that
        // exists has been encoded once, so report full storage strength.
        node.storage_strength = 1.0;
    }
    if let Some(r) = retrievability {
        node.retrieval_strength = r;
        node.retention_strength = r;
    }
    node
}

/// Read one node (record + card + retrievability) off the store. `None`
/// mirrors "no such id" without an error.
fn fetch_node(store: &StrataStore, id: &str) -> Option<KnowledgeNode> {
    let record = store.get_node(id)?;
    let card = store.card_state(id);
    let retrievability = store.retrievability(id).ok().flatten();
    Some(node_from_record(&record, card.as_ref(), retrievability))
}

/// Convert a product ingest input into the float-free strata mirror.
fn ingest_input_into_strata(input: &IngestInput) -> StrataIngestInput {
    StrataIngestInput {
        content: input.content.clone(),
        node_type: input.node_type.clone(),
        tags: input.tags.clone(),
        // No hidden clock: plain ingest leaves creation time to the caller
        // (deterministic 0). Explicit validity windows are caller data and
        // pass through.
        created_at_ms: None,
        valid_from_ms: input.valid_from.as_ref().map(datetime_to_ms),
        valid_until_ms: input.valid_until.as_ref().map(datetime_to_ms),
    }
}

fn connection_into_strata(connection: &ConnectionRecord) -> StrataConnection {
    StrataConnection {
        source_id: connection.source_id.clone(),
        target_id: connection.target_id.clone(),
        strength_milli: (connection.strength * 1000.0).round() as i64,
        link_type: connection.link_type.clone(),
        meta_sha: None,
        created_at_ms: datetime_to_ms(&connection.created_at),
        activation_count: connection.activation_count as i64,
    }
}

fn connection_from_strata(edge: &StrataConnection) -> ConnectionRecord {
    let at = ms_to_datetime(edge.created_at_ms);
    ConnectionRecord {
        source_id: edge.source_id.clone(),
        target_id: edge.target_id.clone(),
        strength: edge.strength(),
        link_type: edge.link_type.clone(),
        created_at: at,
        last_activated: at,
        activation_count: edge.activation_count as i32,
    }
}

// ---------------------------------------------------------------------------
// The trait implementation
// ---------------------------------------------------------------------------

impl MemoryStoreSend for StrataBackend {
    // --- Lifecycle (truthful answers for an already-open store) ---

    async fn init(&self) -> AsyncResult<()> {
        // Open ran replay + checkpoint-chain verification (fail-closed);
        // nothing further to initialize.
        self.call(|_| ()).map_err(BridgeError::into_async_error)
    }

    async fn health_check(&self) -> AsyncResult<HealthStatus> {
        self.call(|_| HealthStatus::Healthy)
            .map_err(BridgeError::into_async_error)
    }

    // --- Embedding model registry (STRATA v1 has no embedding surface) ---

    async fn registered_model(&self) -> AsyncResult<Option<ModelSignature>> {
        Ok(None)
    }

    async fn register_model(&self, _sig: &ModelSignature) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "register_model is not implemented by the STRATA backend (no embedding surface)".into(),
        ))
    }

    // --- Phase 1 row surface (Uuid-keyed; STRATA ids are log-derived
    //     strings and the product seam below is the sanctioned path) ---

    async fn insert(
        &self,
        _record: &vestige_core::storage::MemoryRecord,
    ) -> AsyncResult<uuid::Uuid> {
        Err(AsyncError::Init(
            "insert is not implemented by the STRATA backend (Uuid row surface; use ingest)".into(),
        ))
    }

    async fn get(
        &self,
        _id: uuid::Uuid,
    ) -> AsyncResult<Option<vestige_core::storage::MemoryRecord>> {
        Err(AsyncError::Init(
            "get is not implemented by the STRATA backend (Uuid row surface; use get_node)".into(),
        ))
    }

    async fn update(&self, _record: &vestige_core::storage::MemoryRecord) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "update is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn delete(&self, _id: uuid::Uuid) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "delete is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn search_records(
        &self,
        _query: &vestige_core::storage::SearchQuery,
    ) -> AsyncResult<Vec<vestige_core::storage::SearchResult>> {
        Err(AsyncError::Init(
            "search_records is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn fts_search(
        &self,
        _text: &str,
        _limit: usize,
    ) -> AsyncResult<Vec<vestige_core::storage::SearchResult>> {
        Err(AsyncError::Init(
            "fts_search is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn vector_search(
        &self,
        _embedding: &[f32],
        _limit: usize,
    ) -> AsyncResult<Vec<vestige_core::storage::SearchResult>> {
        Err(AsyncError::Init(
            "vector_search is not implemented by the STRATA backend (no embedding surface)".into(),
        ))
    }

    async fn get_scheduling(
        &self,
        _memory_id: uuid::Uuid,
    ) -> AsyncResult<Option<vestige_core::storage::SchedulingState>> {
        Err(AsyncError::Init(
            "get_scheduling is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn update_scheduling(
        &self,
        _state: &vestige_core::storage::SchedulingState,
    ) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "update_scheduling is not implemented by the STRATA backend (Uuid row surface)".into(),
        ))
    }

    async fn get_due_memories(
        &self,
        _before: DateTime<Utc>,
        _limit: usize,
    ) -> AsyncResult<
        Vec<(
            vestige_core::storage::MemoryRecord,
            vestige_core::storage::SchedulingState,
        )>,
    > {
        Err(AsyncError::Init(
            "get_due_memories is not implemented by the STRATA backend (no wall clock; review drives the fold)"
                .into(),
        ))
    }

    async fn add_edge(&self, _edge: &vestige_core::storage::MemoryEdge) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "add_edge is not implemented by the STRATA backend (Uuid graph surface; use save_connection)"
                .into(),
        ))
    }

    async fn get_edges(
        &self,
        _node_id: uuid::Uuid,
        _edge_type: Option<&str>,
    ) -> AsyncResult<Vec<vestige_core::storage::MemoryEdge>> {
        Err(AsyncError::Init(
            "get_edges is not implemented by the STRATA backend (Uuid graph surface; use get_connections_for_memory)"
                .into(),
        ))
    }

    async fn remove_edge(&self, _source: uuid::Uuid, _target: uuid::Uuid) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "remove_edge is not implemented by the STRATA backend (Uuid graph surface)".into(),
        ))
    }

    async fn get_neighbors(
        &self,
        _node_id: uuid::Uuid,
        _depth: usize,
    ) -> AsyncResult<Vec<(vestige_core::storage::MemoryRecord, f64)>> {
        Err(AsyncError::Init(
            "get_neighbors is not implemented by the STRATA backend (Uuid graph surface)".into(),
        ))
    }

    // --- Domains: the trait declares these Phase-1 stubs returning empty;
    //     mirror that sanctioned behavior. Mutations stay loud. ---

    async fn list_domains(&self) -> AsyncResult<Vec<vestige_core::storage::Domain>> {
        Ok(Vec::new())
    }

    async fn get_domain(&self, _id: &str) -> AsyncResult<Option<vestige_core::storage::Domain>> {
        Ok(None)
    }

    async fn upsert_domain(&self, _domain: &vestige_core::storage::Domain) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "upsert_domain is not implemented by the STRATA backend (no domain store)".into(),
        ))
    }

    async fn delete_domain(&self, _id: &str) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "delete_domain is not implemented by the STRATA backend (no domain store)".into(),
        ))
    }

    async fn classify(&self, _embedding: &[f32]) -> AsyncResult<Vec<(String, f64)>> {
        Ok(Vec::new())
    }

    // --- Bulk / maintenance (truthful counts; vacuum is not applicable) ---

    async fn count(&self) -> AsyncResult<usize> {
        self.call(|store| store.node_count())
            .map_err(BridgeError::into_async_error)
    }

    async fn get_store_stats(&self) -> AsyncResult<StoreStats> {
        self.call(|store| StoreStats {
            total_memories: store.node_count(),
            memories_with_embeddings: 0,
            total_edges: store.edge_count(),
            total_domains: 0,
            registered_model_name: None,
            registered_model_dim: None,
        })
        .map_err(BridgeError::into_async_error)
    }

    async fn vacuum(&self) -> AsyncResult<()> {
        Err(AsyncError::Init(
            "vacuum is not implemented by the STRATA backend (append-only log; no compaction)"
                .into(),
        ))
    }

    // -----------------------------------------------------------------------
    // PHASE 4 PRODUCT SEAM — the strata-store v1 surface
    // -----------------------------------------------------------------------

    /// Ingest into the default scope (`"user"`, matching the SQLite
    /// reference's `DEFAULT_MEMORY_SCOPE`).
    fn ingest(&self, input: IngestInput) -> Result<KnowledgeNode, StorageError> {
        // Qualified: the blanket `impl MemoryStore for T` makes the plain
        // call ambiguous between the two traits.
        <Self as MemoryStoreSend>::ingest_in_scope(self, input, DEFAULT_MEMORY_SCOPE)
    }

    fn ingest_in_scope(
        &self,
        input: IngestInput,
        scope: &str,
    ) -> Result<KnowledgeNode, StorageError> {
        let strata_input = ingest_input_into_strata(&input);
        let scope = scope.to_string();
        self.call(move |store| {
            let id = store
                .ingest_in_scope(strata_input, &scope)
                .map_err(map_store_error)?;
            Ok::<_, StorageError>(
                fetch_node(store, &id).expect("just-ingested node must read back"),
            )
        })
        .map_err(BridgeError::into_store_error)?
    }

    fn get_node(&self, id: &str) -> Result<Option<KnowledgeNode>, StorageError> {
        let id = id.to_string();
        self.call(move |store| Ok::<_, StorageError>(fetch_node(store, &id)))
            .map_err(BridgeError::into_store_error)?
    }

    fn get_all_nodes_in_scope(
        &self,
        scope: &str,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<KnowledgeNode>, StorageError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let scope = scope.to_string();
        let skip = offset.max(0) as usize;
        let take = limit as usize;
        self.call(move |store| {
            Ok::<_, StorageError>(
                store
                    .get_all_nodes_in_scope(&scope)
                    .into_iter()
                    .skip(skip)
                    .take(take)
                    .map(|record| {
                        let card = store.card_state(&record.id);
                        let r = store.retrievability(&record.id).ok().flatten();
                        node_from_record(&record, card.as_ref(), r)
                    })
                    .collect(),
            )
        })
        .map_err(BridgeError::into_store_error)?
    }

    fn set_created_at(&self, id: &str, when: DateTime<Utc>) -> Result<(), StorageError> {
        let id = id.to_string();
        let ms = datetime_to_ms(&when);
        self.call(move |store| store.set_created_at(&id, ms).map_err(map_store_error))
            .map_err(BridgeError::into_store_error)?
    }

    fn save_connection(&self, connection: &ConnectionRecord) -> Result<(), StorageError> {
        let strata_connection = connection_into_strata(connection);
        self.call(move |store| {
            store
                .save_connection(&strata_connection)
                .map_err(map_store_error)
        })
        .map_err(BridgeError::into_store_error)?
    }

    fn get_connections_for_memory(
        &self,
        memory_id: &str,
    ) -> Result<Vec<ConnectionRecord>, StorageError> {
        let id = memory_id.to_string();
        self.call(move |store| {
            Ok::<_, StorageError>(
                store
                    .get_connections_for_memory(&id)
                    .iter()
                    .map(connection_from_strata)
                    .collect(),
            )
        })
        .map_err(BridgeError::into_store_error)?
    }

    fn supersession_pairs(&self) -> Result<Vec<(String, String)>, StorageError> {
        self.call(|store| Ok::<_, StorageError>(store.supersession_pairs()))
            .map_err(BridgeError::into_store_error)?
    }

    fn get_never_composed_candidates_in_scope(
        &self,
        limit: i32,
        tag_filter: Option<&[String]>,
        scope: Option<&str>,
    ) -> Result<Vec<NeverComposedCandidate>, StorageError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let scope = scope.unwrap_or(DEFAULT_MEMORY_SCOPE).to_string();
        let take = limit as usize;
        let tags: Vec<String> = tag_filter.unwrap_or(&[]).to_vec();
        self.call(move |store| {
            let mut out = Vec::new();
            for (first_id, second_id) in store.get_never_composed(&scope, take) {
                let (Some(first), Some(second)) =
                    (store.get_node(&first_id), store.get_node(&second_id))
                else {
                    continue;
                };
                let shared_tags: Vec<String> = first
                    .tags
                    .iter()
                    .filter(|t| second.tags.contains(t))
                    .cloned()
                    .collect();
                if !tags.is_empty() && !tags.iter().any(|t| shared_tags.contains(t)) {
                    continue;
                }
                let preview = |content: &str| -> String { content.chars().take(80).collect() };
                out.push(NeverComposedCandidate {
                    first_id: first_id.clone(),
                    second_id: second_id.clone(),
                    // strata-store computes no fusion scores; the pair is the
                    // store-side fact, the scores are the tool layer's job.
                    score: 0.0,
                    novelty_score: 0.0,
                    bridge_score: 0.0,
                    trust_score: 0.0,
                    outcome_score_adjustment: 0.0,
                    shared_tags,
                    boundary_tags: Vec::new(),
                    shared_terms: Vec::new(),
                    prior_outcomes: Vec::new(),
                    outcome_signal: String::new(),
                    first_node_type: first.node_type.clone(),
                    second_node_type: second.node_type.clone(),
                    first_preview: preview(&first.content),
                    second_preview: preview(&second.content),
                    reason: "no edge between the pair in either direction (strata derived index)"
                        .into(),
                    composition_question: String::new(),
                });
            }
            Ok::<_, StorageError>(out)
        })
        .map_err(BridgeError::into_store_error)?
    }

    fn backup_to(&self, path: &Path) -> Result<(), StorageError> {
        let dest = path.to_path_buf();
        self.call(move |store| store.backup_to(&dest).map_err(map_store_error))
            .map_err(BridgeError::into_store_error)?
    }

    fn mark_reviewed(&self, id: &str, rating: Rating) -> Result<KnowledgeNode, StorageError> {
        let id = id.to_string();
        let rating_code = rating as u8;
        self.call(move |store| {
            store.review(&id, rating_code).map_err(map_store_error)?;
            Ok::<_, StorageError>(
                fetch_node(store, &id).ok_or_else(|| StorageError::NotFound(id.clone()))?,
            )
        })
        .map_err(BridgeError::into_store_error)?
    }

    // --- Boot-critical infrastructure answers (defaults panic) ---

    fn data_dir(&self) -> &Path {
        &self.dir
    }

    fn db_path(&self) -> &Path {
        // The durable artifact of a STRATA store is its append-only log.
        &self.db_path
    }

    fn sidecar_dir(&self, name: &str) -> PathBuf {
        let dir = self.dir.join(name);
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn process_actor_did(&self) -> Option<String> {
        // No actor table in STRATA v1; `set_process_actor` fails (logged as
        // a warning by the boot path), so "no actor bound" is the truth.
        None
    }

    fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        // STRATA v1 seals+copies backups but records no timestamp of the
        // last one; `None` ("never / unknown") is the truthful answer and
        // keeps read-only status surfaces (session_start, maintenance)
        // working instead of hitting the default `unimplemented!`.
        None
    }

    fn resolve_handle(&self, _query: &str) -> vestige_core::storage::HandleResolution {
        // The STRATA backend has no handle resolver yet. The type's own
        // negative answer (Unknown + guidance) is its designed failure
        // channel — recall's handle mode then reports `handle_required`
        // instead of panicking on the default.
        vestige_core::storage::HandleResolution {
            kind: vestige_core::storage::HandleKind::Unknown,
            ids: Vec::new(),
            exact: false,
            candidates: Vec::new(),
            handle_required: Some(vestige_core::storage::HANDLE_REQUIRED_DETAIL.to_string()),
        }
    }
}
