//! First-run honesty. While the embedding runtime is still loading (or, on a
//! fresh install, downloading about 130 MB), retrieval is keyword-only and
//! saves have no vector yet. The responses say so, so the agent can tell the
//! user why the first minute looks thin instead of guessing.

use serde_json::{Value, json};
use vestige_core::Storage;

/// The `warming` block for a response, or `None` once the runtime is ready.
/// Silent in builds without an embedding runtime: nothing is warming there.
pub fn embedding_warming(storage: &Storage) -> Option<Value> {
    if !crate::embeddings_compiled_in() || storage.is_embedding_ready() {
        return None;
    }
    Some(json!({
        "embeddingRuntime": "not ready",
        "effect": "keyword-only retrieval and vector-less saves until the embedding model is loaded; a first run downloads about 130 MB",
        "check": "memory_status view='health' reports embeddingReady; saves are back-filled with vectors once it is ready",
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use vestige_core::Storage;

    fn storage_without_runtime() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    #[test]
    fn warming_is_reported_until_the_runtime_is_ready() {
        let (storage, _dir) = storage_without_runtime();
        let block = super::embedding_warming(&storage).expect("runtime is not ready yet");
        assert_eq!(block["embeddingRuntime"], "not ready");
        assert!(block["effect"].as_str().unwrap().contains("130 MB"));
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    #[test]
    fn warming_is_silent_when_no_runtime_is_compiled_in() {
        let (storage, _dir) = storage_without_runtime();
        assert!(super::embedding_warming(&storage).is_none());
    }
}
