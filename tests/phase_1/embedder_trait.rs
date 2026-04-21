//! Phase 1 integration tests: Embedder trait and FastembedEmbedder.

use vestige_core::embedder::{Embedder, FastembedEmbedder};
use vestige_core::storage::MemoryStore;
use std::sync::Arc;
use tempfile::tempdir;
use vestige_core::storage::SqliteMemoryStore;

fn make_store() -> Arc<dyn MemoryStore> {
    let dir = tempdir().unwrap();
    let db = dir.path().join("test.db");
    std::mem::forget(dir);
    Arc::new(SqliteMemoryStore::new(Some(db)).expect("create"))
}

#[tokio::test]
async fn fastembed_implements_embedder_trait() {
    // The key test: `Box<dyn Embedder>` compiles
    let e: Box<dyn Embedder> = Box::new(FastembedEmbedder::new());
    assert_eq!(e.dimension(), 256, "dimension must be 256");
    assert!(!e.model_name().is_empty(), "model_name must not be empty");
    assert!(!e.model_hash().is_empty(), "model_hash must not be empty");
    assert_eq!(e.model_hash().len(), 64, "hash must be 64 hex chars");
}

#[tokio::test]
async fn signature_matches_memory_store_registry() {
    let e = FastembedEmbedder::new();
    let sig = e.signature();
    let store = make_store();
    store.register_model(&sig).await.expect("register via Embedder::signature");
    let got = store.registered_model().await.expect("registered_model").expect("Some");
    assert_eq!(got.name, sig.name);
    assert_eq!(got.dimension, sig.dimension);
    assert_eq!(got.hash, sig.hash);
}
