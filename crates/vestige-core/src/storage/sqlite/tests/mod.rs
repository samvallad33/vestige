//! Unit tests for the SQLite store, split by the same domains as the source
//! modules. Shared fixtures and helpers live here; each submodule tests one
//! sibling of `storage/sqlite/`.

use super::*;
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use crate::advanced::{MatchClass, MergePolicy};
use crate::storage::memory_store::{
    MemoryEdge, MemoryRecord, MemoryStore, MemoryStoreError, ModelSignature, SchedulingState,
};
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
#[cfg(unix)]
use std::process::{Command, Stdio};
#[cfg(unix)]
use std::sync::mpsc;
use tempfile::tempdir;
// The public struct was renamed from Storage to SqliteMemoryStore; this
// alias keeps all existing tests compiling without modification.
use SqliteMemoryStore as Storage;

mod admin;
mod connectors;
mod embeddings;
mod ingest;
mod lifecycle;
mod lint;
mod merge;
mod purge;
mod records;
mod search;
mod store_trait;
mod sync;

fn create_test_storage() -> Storage {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    Storage::new(Some(db_path)).unwrap()
}

fn create_test_storage_at(dir: &tempfile::TempDir, name: &str) -> Storage {
    Storage::new(Some(dir.path().join(name))).unwrap()
}

/// Build an `IngestInput` carrying a source envelope for a GitHub-ish issue.
fn source_input(id: &str, content: &str, hash: &str) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        node_type: "fact".to_string(),
        source_envelope: Some(crate::memory::SourceEnvelope {
            source_system: Some("github".to_string()),
            source_id: Some(id.to_string()),
            source_url: Some(format!("https://github.com/o/r/issues/{id}")),
            content_hash: Some(hash.to_string()),
            source_project: Some("o/r".to_string()),
            source_type: Some("issue".to_string()),
            source_author: Some("octocat".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Run `f` with vector search disabled for this thread only. It used to
/// set `VESTIGE_DISABLE_VECTOR_SEARCH` in the process environment under
/// `ENV_LOCK`, but every other test thread building a `Storage` during
/// that window got no vector index: `cargo test --lib vector` failed the
/// three `peer_*` tests on every run, and the full suite passed only by
/// scheduling luck.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn with_vector_search_disabled<T>(f: impl FnOnce() -> T) -> T {
    VECTOR_SEARCH_DISABLED_FOR_TEST.with(|cell| cell.set(true));
    let result = catch_unwind(AssertUnwindSafe(f));
    VECTOR_SEARCH_DISABLED_FOR_TEST.with(|cell| cell.set(false));

    match result {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    }
}

fn ingest_tagged_in_scope(
    storage: &Storage,
    scope: &str,
    content: &str,
    tags: &[&str],
) -> KnowledgeNode {
    storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                tags: tags.iter().map(|tag| tag.to_string()).collect(),
                ..Default::default()
            },
            scope,
        )
        .unwrap()
}

/// Ingest a node and seed it with a controllable embedding under the active
/// model so similarity is deterministic in tests.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn seed_node(storage: &Storage, content: &str, tags: &[&str], vector: Vec<f32>) -> String {
    let node = storage
        .ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            ..Default::default()
        })
        .unwrap();
    let bytes = Embedding::new(vector).to_bytes();
    let active = storage.embedding_service.model_name().to_string();
    let writer = storage.writer.lock().unwrap();
    writer
        .execute(
            "INSERT OR REPLACE INTO node_embeddings
                 (node_id, embedding, dimensions, model, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                &node.id,
                &bytes,
                EMBEDDING_DIMENSIONS as i32,
                active,
                Utc::now().to_rfc3339()
            ],
        )
        .unwrap();
    writer
        .execute(
            "UPDATE knowledge_nodes SET has_embedding = 1 WHERE id = ?1",
            rusqlite::params![&node.id],
        )
        .unwrap();
    node.id
}

/// A near-unit vector pointing mostly along `axis`, so two nodes sharing an
/// axis are highly similar and nodes on different axes are not.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn axis_vector(axis: usize, jitter: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBEDDING_DIMENSIONS];
    v[axis % EMBEDDING_DIMENSIONS] = 1.0;
    v[(axis + 1) % EMBEDDING_DIMENSIONS] = jitter;
    v
}

/// Write a caller-supplied vector for `node_id` through the production
/// funnel every embedding write uses, so the row lands in the active
/// profile's table, the legacy mirror, the `has_embedding` flag and THIS
/// store's in-memory index exactly as a real ingest would.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn persist_test_vector(storage: &Storage, node_id: &str, vector: &[f32]) {
    let bytes = Embedding::new(vector.to_vec()).to_bytes();
    storage
        .persist_node_embedding(node_id, &bytes, vector.len(), "test-model", vector, true)
        .unwrap();
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn index_contains(storage: &Storage, node_id: &str) -> bool {
    storage
        .vector_index
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .contains(node_id)
}

/// Id of the top hit for `vector` in this store's in-memory index.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn nearest(storage: &Storage, vector: &[f32]) -> Option<String> {
    storage
        .vector_index
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .search(vector, 1)
        .unwrap()
        .into_iter()
        .next()
        .map(|(id, _)| id)
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

/// Run `f` with VESTIGE_AUTO_CONSOLIDATE_MERGE pinned to `value` for this
/// thread (None = pinned-unset, the documented fail-closed default).
/// Sibling of `with_vector_search_disabled`; like it, this no longer
/// touches the process environment, so consolidation tests on other
/// threads keep reading the real one.
#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn with_auto_merge_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    AUTO_CONSOLIDATE_MERGE_FOR_TEST
        .with(|cell| *cell.borrow_mut() = Some(value.map(str::to_string)));
    let result = catch_unwind(AssertUnwindSafe(f));
    AUTO_CONSOLIDATE_MERGE_FOR_TEST.with(|cell| *cell.borrow_mut() = None);
    match result {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    }
}
