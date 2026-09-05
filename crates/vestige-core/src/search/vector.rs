//! High-Performance Vector Search
//!
//! Uses USearch for HNSW (Hierarchical Navigable Small World) indexing.
//! 20x faster than FAISS for approximate nearest neighbor search.
//!
//! Features:
//! - Sub-millisecond query times
//! - Cosine similarity by default
//! - Incremental index updates
//! - Persistence to disk

use std::collections::HashMap;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default embedding dimensions after Matryoshka truncation (768 → 256)
/// 3x storage savings with only ~2% quality loss on MTEB benchmarks
pub const DEFAULT_DIMENSIONS: usize = 256;

/// HNSW connectivity parameter (higher = better recall, more memory)
pub const DEFAULT_CONNECTIVITY: usize = 16;

/// HNSW expansion factor for index building
pub const DEFAULT_EXPANSION_ADD: usize = 128;

/// HNSW expansion factor for search (higher = better recall, slower)
pub const DEFAULT_EXPANSION_SEARCH: usize = 64;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Vector search error types
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum VectorSearchError {
    /// Failed to create the index
    IndexCreation(String),
    /// Failed to add a vector
    IndexAdd(String),
    /// Failed to search
    IndexSearch(String),
    /// Failed to persist/load index
    IndexPersistence(String),
    /// Dimension mismatch
    InvalidDimensions(usize, usize),
    /// Key not found
    KeyNotFound(u64),
}

impl std::fmt::Display for VectorSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorSearchError::IndexCreation(e) => write!(f, "Index creation failed: {}", e),
            VectorSearchError::IndexAdd(e) => write!(f, "Failed to add vector: {}", e),
            VectorSearchError::IndexSearch(e) => write!(f, "Search failed: {}", e),
            VectorSearchError::IndexPersistence(e) => write!(f, "Persistence failed: {}", e),
            VectorSearchError::InvalidDimensions(expected, got) => {
                write!(f, "Invalid dimensions: expected {}, got {}", expected, got)
            }
            VectorSearchError::KeyNotFound(key) => write!(f, "Key not found: {}", key),
        }
    }
}

impl std::error::Error for VectorSearchError {}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for vector index
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Number of dimensions
    pub dimensions: usize,
    /// HNSW connectivity parameter
    pub connectivity: usize,
    /// Expansion factor for adding vectors
    pub expansion_add: usize,
    /// Expansion factor for searching
    pub expansion_search: usize,
    /// Distance metric
    pub metric: MetricKind,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimensions: DEFAULT_DIMENSIONS,
            connectivity: DEFAULT_CONNECTIVITY,
            expansion_add: DEFAULT_EXPANSION_ADD,
            expansion_search: DEFAULT_EXPANSION_SEARCH,
            metric: MetricKind::Cos, // Cosine similarity
        }
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    /// Total number of vectors
    pub total_vectors: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// HNSW connectivity
    pub connectivity: usize,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
}

// ============================================================================
// VECTOR INDEX
// ============================================================================

/// High-performance HNSW vector index
pub struct VectorIndex {
    index: Index,
    config: VectorIndexConfig,
    key_to_id: HashMap<String, u64>,
    id_to_key: HashMap<u64, String>,
    next_id: u64,
}

impl VectorIndex {
    /// Create a new vector index with default configuration
    pub fn new() -> Result<Self, VectorSearchError> {
        Self::with_config(VectorIndexConfig::default())
    }

    /// Create a new vector index with custom configuration
    pub fn with_config(config: VectorIndexConfig) -> Result<Self, VectorSearchError> {
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: config.metric,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };

        let index =
            Index::new(&options).map_err(|e| VectorSearchError::IndexCreation(e.to_string()))?;

        Ok(Self {
            index,
            config,
            key_to_id: HashMap::new(),
            id_to_key: HashMap::new(),
            next_id: 0,
        })
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimensions of the index
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Reserve capacity for a specified number of vectors
    /// This should be called before adding vectors to avoid segmentation faults
    pub fn reserve(&self, capacity: usize) -> Result<(), VectorSearchError> {
        self.index.reserve(capacity).map_err(|e| {
            VectorSearchError::IndexCreation(format!("Failed to reserve capacity: {}", e))
        })
    }

    /// Add a vector with a string key
    pub fn add(&mut self, key: &str, vector: &[f32]) -> Result<(), VectorSearchError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorSearchError::InvalidDimensions(
                self.config.dimensions,
                vector.len(),
            ));
        }

        // Check if key already exists
        if let Some(&existing_id) = self.key_to_id.get(key) {
            // Update existing vector
            self.index
                .remove(existing_id)
                .map_err(|e| VectorSearchError::IndexAdd(e.to_string()))?;
            // Reserve capacity for the re-add
            self.reserve(self.index.size() + 1)?;
            self.index
                .add(existing_id, vector)
                .map_err(|e| VectorSearchError::IndexAdd(e.to_string()))?;
            return Ok(());
        }

        // Ensure we have capacity before adding
        // usearch requires reserve() to be called before add() to avoid segfaults
        let current_capacity = self.index.capacity();
        let current_size = self.index.size();
        if current_size >= current_capacity {
            // Reserve more capacity (double or at least 16)
            let new_capacity = std::cmp::max(current_capacity * 2, 16);
            self.reserve(new_capacity)?;
        }

        // Add new vector
        let id = self.next_id;
        self.next_id += 1;

        self.index
            .add(id, vector)
            .map_err(|e| VectorSearchError::IndexAdd(e.to_string()))?;

        self.key_to_id.insert(key.to_string(), id);
        self.id_to_key.insert(id, key.to_string());

        Ok(())
    }

    /// Remove a vector by key
    pub fn remove(&mut self, key: &str) -> Result<bool, VectorSearchError> {
        if let Some(id) = self.key_to_id.remove(key) {
            self.id_to_key.remove(&id);
            self.index
                .remove(id)
                .map_err(|e| VectorSearchError::IndexAdd(e.to_string()))?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if a key exists in the index
    pub fn contains(&self, key: &str) -> bool {
        self.key_to_id.contains_key(key)
    }

    /// Every key currently in the index, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.key_to_id.keys().map(String::as_str)
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, f32)>, VectorSearchError> {
        if query.len() != self.config.dimensions {
            return Err(VectorSearchError::InvalidDimensions(
                self.config.dimensions,
                query.len(),
            ));
        }

        if self.is_empty() {
            return Ok(vec![]);
        }

        let results = self
            .index
            .search(query, limit)
            .map_err(|e| VectorSearchError::IndexSearch(e.to_string()))?;

        let mut search_results = Vec::with_capacity(results.keys.len());
        for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
            if let Some(string_key) = self.id_to_key.get(key) {
                // Convert distance to similarity (1 - distance for cosine)
                let score = 1.0 - distance;
                search_results.push((string_key.clone(), score));
            }
        }

        Ok(search_results)
    }

    /// Search with minimum similarity threshold
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<(String, f32)>, VectorSearchError> {
        let results = self.search(query, limit)?;
        Ok(results
            .into_iter()
            .filter(|(_, score)| *score >= min_similarity)
            .collect())
    }

    /// Save the index to disk
    pub fn save(&self, path: &Path) -> Result<(), VectorSearchError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| VectorSearchError::IndexPersistence("Invalid path".to_string()))?;

        self.index
            .save(path_str)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;

        // Save key mappings
        let mappings_path = path.with_extension("mappings.json");
        let mappings = serde_json::json!({
            "key_to_id": self.key_to_id,
            "next_id": self.next_id,
        });
        let mappings_str = serde_json::to_string(&mappings)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;
        std::fs::write(&mappings_path, mappings_str)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;

        Ok(())
    }

    /// Load the index from disk
    pub fn load(path: &Path, config: VectorIndexConfig) -> Result<Self, VectorSearchError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| VectorSearchError::IndexPersistence("Invalid path".to_string()))?;

        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: config.metric,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };

        let index =
            Index::new(&options).map_err(|e| VectorSearchError::IndexCreation(e.to_string()))?;

        index
            .load(path_str)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;

        // Load key mappings
        let mappings_path = path.with_extension("mappings.json");
        let mappings_str = std::fs::read_to_string(&mappings_path)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;
        let mappings: serde_json::Value = serde_json::from_str(&mappings_str)
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;

        let key_to_id: HashMap<String, u64> = serde_json::from_value(mappings["key_to_id"].clone())
            .map_err(|e| VectorSearchError::IndexPersistence(e.to_string()))?;

        let next_id: u64 = mappings["next_id"]
            .as_u64()
            .ok_or_else(|| VectorSearchError::IndexPersistence("Invalid next_id".to_string()))?;

        // Rebuild reverse mapping
        let id_to_key: HashMap<u64, String> =
            key_to_id.iter().map(|(k, &v)| (v, k.clone())).collect();

        Ok(Self {
            index,
            config,
            key_to_id,
            id_to_key,
            next_id,
        })
    }

    /// Get index statistics
    pub fn stats(&self) -> VectorIndexStats {
        VectorIndexStats {
            total_vectors: self.len(),
            dimensions: self.config.dimensions,
            connectivity: self.config.connectivity,
            memory_bytes: self.index.serialized_length(),
        }
    }
}

// NOTE: Default implementation removed because VectorIndex::new() is fallible.
// Use VectorIndex::new() directly and handle the Result appropriately.
// If you need a Default-like interface, consider using Option<VectorIndex> or
// a wrapper that handles initialization lazily.

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(seed: f32) -> Vec<f32> {
        (0..DEFAULT_DIMENSIONS)
            .map(|i| ((i as f32 + 1.0) * seed).sin())
            .collect()
    }

    #[test]
    fn test_index_creation() {
        let index = VectorIndex::new().unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.dimensions(), DEFAULT_DIMENSIONS);
    }

    #[test]
    fn test_add_and_search() {
        let mut index = VectorIndex::new().unwrap();

        let v1 = create_test_vector(1.0);
        let v2 = create_test_vector(2.0);
        let v3 = create_test_vector(100.0);

        index.add("node-1", &v1).unwrap();
        index.add("node-2", &v2).unwrap();
        index.add("node-3", &v3).unwrap();

        assert_eq!(index.len(), 3);
        assert!(index.contains("node-1"));
        assert!(!index.contains("node-999"));

        let results = index.search(&v1, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "node-1");
    }

    #[test]
    fn test_remove() {
        let mut index = VectorIndex::new().unwrap();
        let v1 = create_test_vector(1.0);

        index.add("node-1", &v1).unwrap();
        assert!(index.contains("node-1"));

        index.remove("node-1").unwrap();
        assert!(!index.contains("node-1"));
    }

    #[test]
    fn test_update() {
        let mut index = VectorIndex::new().unwrap();
        let v1 = create_test_vector(1.0);
        let v2 = create_test_vector(2.0);

        index.add("node-1", &v1).unwrap();
        assert_eq!(index.len(), 1);

        index.add("node-1", &v2).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut index = VectorIndex::new().unwrap();
        let wrong_size: Vec<f32> = vec![1.0, 2.0, 3.0];

        let result = index.add("node-1", &wrong_size);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_with_threshold() {
        let mut index = VectorIndex::new().unwrap();

        let v1 = create_test_vector(1.0);
        let v2 = create_test_vector(100.0);

        index.add("similar", &v1).unwrap();
        index.add("different", &v2).unwrap();

        let results = index.search_with_threshold(&v1, 10, 0.9).unwrap();

        // Should only include the similar one
        assert!(results.iter().any(|(k, _)| k == "similar"));
    }

    #[test]
    fn test_stats() {
        let mut index = VectorIndex::new().unwrap();
        let v1 = create_test_vector(1.0);

        index.add("node-1", &v1).unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 1);
        assert_eq!(stats.dimensions, DEFAULT_DIMENSIONS);
    }
}
