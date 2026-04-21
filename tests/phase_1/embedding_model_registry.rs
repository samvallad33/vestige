//! Phase 1 integration tests: embedding model registry.

use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;
use vestige_core::storage::{
    MemoryRecord, MemoryStore, MemoryStoreError, ModelSignature, SqliteMemoryStore,
};

fn make_store() -> Arc<dyn MemoryStore> {
    let dir = tempdir().unwrap();
    let db = dir.path().join("test.db");
    std::mem::forget(dir);
    let store = SqliteMemoryStore::new(Some(db)).expect("create store");
    Arc::new(store)
}

fn sig_a() -> ModelSignature {
    ModelSignature {
        name: "model-a".to_string(),
        dimension: 256,
        hash: "a".repeat(64),
    }
}

fn sig_b() -> ModelSignature {
    ModelSignature {
        name: "model-b".to_string(),
        dimension: 256,
        hash: "b".repeat(64),
    }
}

fn record_without_embedding() -> MemoryRecord {
    MemoryRecord {
        id: Uuid::new_v4(),
        domains: vec![],
        domain_scores: Default::default(),
        content: "plain text memory".to_string(),
        node_type: "fact".to_string(),
        tags: vec![],
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: serde_json::json!({}),
    }
}

#[tokio::test]
async fn first_embedded_insert_auto_registers() {
    // fresh store; register a model, then check registered_model() returns Some
    let store = make_store();
    let sig = sig_a();
    store.register_model(&sig).await.expect("register");
    let got = store.registered_model().await.expect("registered_model");
    assert_eq!(got, Some(sig));
}

#[tokio::test]
async fn second_insert_with_same_signature_succeeds() {
    let store = make_store();
    let sig = sig_a();
    store.register_model(&sig).await.expect("first register");
    store.register_model(&sig).await.expect("second register idempotent");
}

#[tokio::test]
async fn second_insert_with_different_dimension_refused() {
    let store = make_store();
    let sig = sig_a(); // dim 256
    store.register_model(&sig).await.expect("register 256");
    // Try inserting a 512-dim vector into a store registered for 256
    let mut rec = record_without_embedding();
    rec.embedding = Some(vec![0.0f32; 512]);
    rec.metadata = serde_json::json!({
        "model_name": "model-a",
        "model_dim": 256_u64,
        "model_hash": "a".repeat(64),
    });
    let err = store.insert(&rec).await.unwrap_err();
    assert!(
        matches!(err, MemoryStoreError::InvalidInput(_)),
        "expected InvalidInput for dim mismatch, got {:?}", err
    );
}

#[tokio::test]
async fn second_insert_with_different_model_name_refused() {
    let store = make_store();
    store.register_model(&sig_a()).await.expect("register a");
    let err = store.register_model(&sig_b()).await.unwrap_err();
    assert!(
        matches!(err, MemoryStoreError::ModelMismatch { .. }),
        "expected ModelMismatch, got {:?}", err
    );
}

#[tokio::test]
async fn second_insert_with_different_hash_refused() {
    let store = make_store();
    let sig = sig_a();
    store.register_model(&sig).await.expect("register");
    let sig_diff_hash = ModelSignature {
        name: "model-a".to_string(),
        dimension: 256,
        hash: "c".repeat(64), // different hash
    };
    let err = store.register_model(&sig_diff_hash).await.unwrap_err();
    assert!(
        matches!(err, MemoryStoreError::ModelMismatch { .. }),
        "expected ModelMismatch for different hash, got {:?}", err
    );
}

#[tokio::test]
async fn no_embedding_insert_allowed_before_registration() {
    let store = make_store();
    // registered_model() should be None
    assert!(store.registered_model().await.expect("registered_model").is_none());
    // A plain text memory without an embedding must insert successfully
    let rec = record_without_embedding();
    store.insert(&rec).await.expect("plain insert before registration");
}

#[tokio::test]
async fn stats_reports_registered_model_after_first_write() {
    let store = make_store();
    let sig = sig_a();
    store.register_model(&sig).await.expect("register");
    let stats = store.get_stats().await.expect("stats");
    assert_eq!(stats.registered_model_name, Some("model-a".to_string()));
    assert_eq!(stats.registered_model_dim, Some(256));
}
