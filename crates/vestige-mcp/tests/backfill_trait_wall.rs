//! Phase 4 storage-wall proof.
//!
//! The backfill MCP tool runs end-to-end against `WallMockStore`, a pure
//! in-memory mock that implements the storage trait (`MemoryStoreSend`) and
//! overrides ONLY the seven methods backfill needs. Every other trait method
//! resolves to the wall's loud defaults (`Err(StorageError::Init("... not
//! implemented by this backend"))` / `unimplemented!`), and the async seam is
//! stubbed with `MemoryStoreError::Backend`. There is no SQLite anywhere in
//! this process. If the tool — or anything it calls — reached behind the
//! trait for a SQLite-only path, this test would fail loudly instead of
//! silently passing. This is the compile-time-and-runtime receipt that the
//! tools layer is engine-agnostic: a second backend (STRATA) that implements
//! the same trait can serve the same tool.

use std::sync::{Arc, Mutex};

use serde_json::json;
use vestige_core::storage::{
    Domain, HealthStatus, MemoryEdge, MemoryStoreError, MemoryStoreResult, MemoryStoreSend,
    MemoryRecord, ModelSignature, SchedulingState, SearchQuery, StorageError, StoreStats,
};
use vestige_core::storage::{ConnectionRecord, Storage};
use vestige_core::KnowledgeNode;
use vestige_mcp::tools::backfill;

type MockResult<T> = std::result::Result<T, StorageError>;

/// In-memory mock store. Implements the storage seam; nothing else.
struct WallMockStore {
    nodes: Mutex<Vec<KnowledgeNode>>,
    connections: Mutex<Vec<ConnectionRecord>>,
    promoted: Mutex<Vec<String>>,
    saved_edges: Mutex<Vec<String>>,
}

impl WallMockStore {
    fn new() -> Self {
        Self {
            nodes: Mutex::new(Vec::new()),
            connections: Mutex::new(Vec::new()),
            promoted: Mutex::new(Vec::new()),
            saved_edges: Mutex::new(Vec::new()),
        }
    }

    fn seed(&self, node: KnowledgeNode) {
        self.nodes.lock().unwrap().push(node);
    }
}

fn node(id: &str, content: &str, tags: &[&str], days_ago: i64) -> KnowledgeNode {
    // KnowledgeNode is non_exhaustive: default-then-mutate is the only
    // cross-crate construction.
    let mut n = KnowledgeNode::default();
    n.id = id.to_string();
    n.content = content.to_string();
    n.tags = tags.iter().map(|t| t.to_string()).collect();
    n.created_at = chrono::Utc::now() - chrono::Duration::days(days_ago);
    n.suppression_count = 0;
    n
}

impl MemoryStoreSend for WallMockStore {
    // ---- the seven methods the backfill tool actually touches ----

    fn get_node(&self, id: &str) -> MockResult<Option<KnowledgeNode>> {
        Ok(self
            .nodes
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.id == id)
            .cloned())
    }

    fn node_is_in_scope(&self, id: &str, _scope: &str) -> MockResult<bool> {
        Ok(self.nodes.lock().unwrap().iter().any(|n| n.id == id))
    }

    fn get_all_nodes_in_scope(&self, _scope: &str, limit: i32, _offset: i32) -> MockResult<Vec<KnowledgeNode>> {
        let mut nodes = self.nodes.lock().unwrap().clone();
        // newest first, mirroring the reference backend's ordering contract
        nodes.sort_by_key(|n| std::cmp::Reverse(n.created_at));
        nodes.truncate(limit.max(0) as usize);
        Ok(nodes)
    }

    fn supersession_pairs(&self) -> MockResult<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    fn get_connections_for_memory(&self, memory_id: &str) -> MockResult<Vec<ConnectionRecord>> {
        Ok(self
            .connections
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.source_id == memory_id || c.target_id == memory_id)
            .cloned()
            .collect())
    }

    fn save_connection(&self, connection: &ConnectionRecord) -> MockResult<()> {
        self.saved_edges
            .lock()
            .unwrap()
            .push(format!("{}->{}", connection.source_id, connection.target_id));
        self.connections.lock().unwrap().push(connection.clone());
        Ok(())
    }

    fn promote_memory_backfill(&self, id: &str) -> MockResult<KnowledgeNode> {
        self.promoted.lock().unwrap().push(id.to_string());
        let mut nodes = self.nodes.lock().unwrap();
        let n = nodes.iter_mut().find(|n| n.id == id).expect("node exists");
        n.reps += 1;
        Ok(n.clone())
    }

    // ---- async seam: stubbed, never reached by this test ----

    async fn init(&self) -> MemoryStoreResult<()> {
        Err(unsup("init"))
    }
    async fn health_check(&self) -> MemoryStoreResult<HealthStatus> {
        Err(unsup("health_check"))
    }
    async fn registered_model(&self) -> MemoryStoreResult<Option<ModelSignature>> {
        Err(unsup("registered_model"))
    }
    async fn register_model(&self, _sig: &ModelSignature) -> MemoryStoreResult<()> {
        Err(unsup("register_model"))
    }
    async fn insert(&self, _record: &MemoryRecord) -> MemoryStoreResult<uuid::Uuid> {
        Err(unsup("insert"))
    }
    async fn get(&self, _id: uuid::Uuid) -> MemoryStoreResult<Option<MemoryRecord>> {
        Err(unsup("get"))
    }
    async fn update(&self, _record: &MemoryRecord) -> MemoryStoreResult<()> {
        Err(unsup("update"))
    }
    async fn delete(&self, _id: uuid::Uuid) -> MemoryStoreResult<()> {
        Err(unsup("delete"))
    }
    async fn search_records(&self, _query: &SearchQuery) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(unsup("search_records"))
    }
    async fn fts_search(
        &self,
        _text: &str,
        _limit: usize,
    ) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(unsup("fts_search"))
    }
    async fn vector_search(
        &self,
        _embedding: &[f32],
        _limit: usize,
    ) -> MemoryStoreResult<Vec<vestige_core::storage::SearchResult>> {
        Err(unsup("vector_search"))
    }
    async fn get_scheduling(
        &self,
        _memory_id: uuid::Uuid,
    ) -> MemoryStoreResult<Option<SchedulingState>> {
        Err(unsup("get_scheduling"))
    }
    async fn update_scheduling(&self, _state: &SchedulingState) -> MemoryStoreResult<()> {
        Err(unsup("update_scheduling"))
    }
    async fn get_due_memories(
        &self,
        _before: chrono::DateTime<chrono::Utc>,
        _limit: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, SchedulingState)>> {
        Err(unsup("get_due_memories"))
    }
    async fn add_edge(&self, _edge: &MemoryEdge) -> MemoryStoreResult<()> {
        Err(unsup("add_edge"))
    }
    async fn get_edges(
        &self,
        _node_id: uuid::Uuid,
        _edge_type: Option<&str>,
    ) -> MemoryStoreResult<Vec<MemoryEdge>> {
        Err(unsup("get_edges"))
    }
    async fn remove_edge(&self, _source: uuid::Uuid, _target: uuid::Uuid) -> MemoryStoreResult<()> {
        Err(unsup("remove_edge"))
    }
    async fn get_neighbors(
        &self,
        _node_id: uuid::Uuid,
        _depth: usize,
    ) -> MemoryStoreResult<Vec<(MemoryRecord, f64)>> {
        Err(unsup("get_neighbors"))
    }
    async fn list_domains(&self) -> MemoryStoreResult<Vec<Domain>> {
        Err(unsup("list_domains"))
    }
    async fn get_domain(&self, _id: &str) -> MemoryStoreResult<Option<Domain>> {
        Err(unsup("get_domain"))
    }
    async fn upsert_domain(&self, _domain: &Domain) -> MemoryStoreResult<()> {
        Err(unsup("upsert_domain"))
    }
    async fn delete_domain(&self, _id: &str) -> MemoryStoreResult<()> {
        Err(unsup("delete_domain"))
    }
    async fn classify(&self, _embedding: &[f32]) -> MemoryStoreResult<Vec<(String, f64)>> {
        Err(unsup("classify"))
    }
    async fn count(&self) -> MemoryStoreResult<usize> {
        Err(unsup("count"))
    }
    async fn get_store_stats(&self) -> MemoryStoreResult<StoreStats> {
        Err(unsup("get_store_stats"))
    }
    async fn vacuum(&self) -> MemoryStoreResult<()> {
        Err(unsup("vacuum"))
    }
}

fn unsup(name: &str) -> MemoryStoreError {
    MemoryStoreError::Backend(format!("mock store: {name} is not supported in this test"))
}

#[tokio::test]
async fn backfill_tool_runs_purely_against_the_trait_wall() {
    let mock = Arc::new(WallMockStore::new());

    // The quiet cause: an env-var edit sharing the API_TIMEOUT entity with
    // the failure, three days older (the canonical backfill scenario).
    mock.seed(node(
        "cause",
        "Set API_TIMEOUT=2 in the deploy env to speed up cold starts",
        &["API_TIMEOUT", "deploy-env"],
        3,
    ));
    // A semantic distractor: crash-adjacent wording, no shared entity.
    mock.seed(node(
        "distractor",
        "An outage hit the billing system last month",
        &["billing-system"],
        20,
    ));
    // The failure, recorded most recently.
    mock.seed(node(
        "failure",
        "Service crashed: 500 Internal Server Error on the auth endpoint",
        &["auth-service", "API_TIMEOUT", "crash"],
        0,
    ));

    // The ONLY conversion in this test: Arc<WallMockStore> -> Arc<dyn MemoryStore>.
    // From here on the tool cannot tell which engine it is talking to.
    let storage: Arc<Storage> = mock.clone();

    let out = backfill::execute(
        &storage,
        Some(json!({ "failure_id": "failure", "promote": true })),
    )
    .await
    .expect("backfill must run against the mock store");

    assert_eq!(out["triggered"], json!(true), "run: {out}");
    let causes = out["causes"].as_array().expect("causes array");
    assert!(!causes.is_empty());

    // The top cause is the env-var memory, surfaced by the shared entity.
    let top = &causes[0];
    let content = top["content_preview"].as_str().unwrap_or("");
    assert!(
        content.contains("API_TIMEOUT") && content.contains("deploy"),
        "top cause must be the env-var edit, got: {content}"
    );
    assert_eq!(top["promoted"], json!(true));
    assert_eq!(top["candidate_edge_persisted"], json!(true));
    assert_eq!(out["causality_verified"], json!(false));

    // The wall held: writes landed in the mock, via trait methods only.
    assert!(mock.promoted.lock().unwrap().contains(&"cause".to_string()));
    assert!(mock
        .saved_edges
        .lock()
        .unwrap()
        .contains(&"cause->failure".to_string()));
}
