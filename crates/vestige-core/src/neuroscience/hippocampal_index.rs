//! # Hippocampal Indexing Theory Implementation
//!
//! Based on Teyler and Rudy's (2007) indexing theory: The hippocampus stores
//! INDICES (pointers), not content. Content is distributed across neocortex.
//!
//! ## Theory Background
//!
//! Just as the hippocampus creates sparse, orthogonal representations that serve
//! as indices to cortical memories, this system separates:
//!
//! - **Index Layer**: Compact, searchable, in-memory (like hippocampus)
//! - **Content Layer**: Detailed, distributed storage (like neocortex)
//!
//! ## Two-Phase Retrieval
//!
//! 1. **Phase 1 (Hippocampal)**: Fast search over compact indices
//!    - Semantic summary embeddings (compressed)
//!    - Temporal markers
//!    - Importance flags
//!
//! 2. **Phase 2 (Neocortical)**: Full content retrieval
//!    - Follow content pointers
//!    - Retrieve from appropriate storage
//!    - Reconstruct full memory
//!
//! ## References
//!
//! - Teyler, T. J., & Rudy, J. W. (2007). The hippocampal indexing theory and
//!   episodic memory: Updating the index. Hippocampus, 17(12), 1158-1169.
//! - McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).
//!   Why there are complementary learning systems in the hippocampus and neocortex.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

// Note: When using with the embeddings feature, cosine_similarity
// and EMBEDDING_DIMENSIONS can be imported from crate::embeddings

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors for hippocampal index operations
#[derive(Debug, Clone)]
pub enum HippocampalIndexError {
    /// Memory not found in index
    NotFound(String),
    /// Content retrieval failed
    ContentRetrievalFailed(String),
    /// Invalid barcode
    InvalidBarcode(String),
    /// Storage error
    StorageError(String),
    /// Index corruption detected
    IndexCorruption(String),
    /// Lock acquisition failed
    LockError(String),
    /// Migration error
    MigrationError(String),
    /// Embedding error
    EmbeddingError(String),
}

impl std::fmt::Display for HippocampalIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HippocampalIndexError::NotFound(id) => write!(f, "Memory not found: {}", id),
            HippocampalIndexError::ContentRetrievalFailed(e) => {
                write!(f, "Content retrieval failed: {}", e)
            }
            HippocampalIndexError::InvalidBarcode(e) => write!(f, "Invalid barcode: {}", e),
            HippocampalIndexError::StorageError(e) => write!(f, "Storage error: {}", e),
            HippocampalIndexError::IndexCorruption(e) => write!(f, "Index corruption: {}", e),
            HippocampalIndexError::LockError(e) => write!(f, "Lock error: {}", e),
            HippocampalIndexError::MigrationError(e) => write!(f, "Migration error: {}", e),
            HippocampalIndexError::EmbeddingError(e) => write!(f, "Embedding error: {}", e),
        }
    }
}

impl std::error::Error for HippocampalIndexError {}

pub type Result<T> = std::result::Result<T, HippocampalIndexError>;

// ============================================================================
// MEMORY BARCODE
// ============================================================================

/// Unique barcode for each memory (inspired by chickadee hippocampus)
///
/// The barcode provides:
/// - Unique identification across the entire memory system
/// - Temporal information (when created)
/// - Content fingerprint (what it represents)
///
/// This is analogous to how hippocampal neurons create sparse,
/// orthogonal patterns for different memories.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryBarcode {
    /// Sequential unique identifier
    pub id: u64,
    /// Hash of creation timestamp (temporal signature)
    pub creation_hash: u32,
    /// Hash of content (content fingerprint)
    pub content_fingerprint: u32,
}

impl MemoryBarcode {
    /// Create a new barcode
    pub fn new(id: u64, creation_hash: u32, content_fingerprint: u32) -> Self {
        Self {
            id,
            creation_hash,
            content_fingerprint,
        }
    }

    /// Convert to a compact string representation
    pub fn to_compact_string(&self) -> String {
        format!(
            "{:016x}-{:08x}-{:08x}",
            self.id, self.creation_hash, self.content_fingerprint
        )
    }

    /// Parse from string representation
    pub fn from_string(s: &str) -> std::result::Result<Self, HippocampalIndexError> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return Err(HippocampalIndexError::InvalidBarcode(
                "Expected 3 parts separated by '-'".to_string(),
            ));
        }

        let id = u64::from_str_radix(parts[0], 16)
            .map_err(|e| HippocampalIndexError::InvalidBarcode(format!("Invalid id: {}", e)))?;
        let creation_hash = u32::from_str_radix(parts[1], 16).map_err(|e| {
            HippocampalIndexError::InvalidBarcode(format!("Invalid creation_hash: {}", e))
        })?;
        let content_fingerprint = u32::from_str_radix(parts[2], 16).map_err(|e| {
            HippocampalIndexError::InvalidBarcode(format!("Invalid content_fingerprint: {}", e))
        })?;

        Ok(Self {
            id,
            creation_hash,
            content_fingerprint,
        })
    }

    /// Check if two barcodes have the same content (ignoring temporal info)
    pub fn same_content(&self, other: &Self) -> bool {
        self.content_fingerprint == other.content_fingerprint
    }

    /// Check if created around the same time (within hash collision probability)
    pub fn similar_time(&self, other: &Self) -> bool {
        self.creation_hash == other.creation_hash
    }
}

impl std::fmt::Display for MemoryBarcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:016x}-{:08x}-{:08x}",
            self.id, self.creation_hash, self.content_fingerprint
        )
    }
}

// ============================================================================
// BARCODE GENERATOR
// ============================================================================

/// Generator for unique memory barcodes
///
/// Creates barcodes that encode:
/// - Sequential ID (uniqueness)
/// - Temporal signature (when)
/// - Content fingerprint (what)
pub struct BarcodeGenerator {
    /// Next sequential ID
    next_id: u64,
    /// Salt for hashing (instance-specific)
    hash_salt: u64,
}

impl BarcodeGenerator {
    /// Create a new barcode generator
    pub fn new() -> Self {
        Self {
            next_id: 0,
            hash_salt: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }

    /// Create a generator starting from a specific ID
    pub fn with_starting_id(starting_id: u64) -> Self {
        Self {
            next_id: starting_id,
            hash_salt: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }

    /// Generate a unique barcode for new memory
    pub fn generate(&mut self, content: &str, timestamp: DateTime<Utc>) -> MemoryBarcode {
        let id = self.next_id;
        self.next_id += 1;

        let creation_hash = self.hash_timestamp(timestamp);
        let content_fingerprint = self.hash_content(content);

        MemoryBarcode::new(id, creation_hash, content_fingerprint)
    }

    /// Generate barcode for existing memory with known ID
    pub fn generate_with_id(
        &self,
        id: u64,
        content: &str,
        timestamp: DateTime<Utc>,
    ) -> MemoryBarcode {
        let creation_hash = self.hash_timestamp(timestamp);
        let content_fingerprint = self.hash_content(content);

        MemoryBarcode::new(id, creation_hash, content_fingerprint)
    }

    /// Hash timestamp to 32-bit signature
    fn hash_timestamp(&self, timestamp: DateTime<Utc>) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        timestamp
            .timestamp_nanos_opt()
            .unwrap_or(0)
            .hash(&mut hasher);
        self.hash_salt.hash(&mut hasher);
        (hasher.finish() & 0xFFFFFFFF) as u32
    }

    /// Hash content to 32-bit fingerprint
    fn hash_content(&self, content: &str) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        content.hash(&mut hasher);
        (hasher.finish() & 0xFFFFFFFF) as u32
    }

    /// Get the current ID counter (for persistence)
    pub fn current_id(&self) -> u64 {
        self.next_id
    }
}

impl Default for BarcodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TEMPORAL MARKER
// ============================================================================

/// Temporal information for a memory index
///
/// Encodes when the memory was created and when it's valid,
/// enabling temporal queries without accessing full content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMarker {
    /// When the memory was created
    pub created_at: DateTime<Utc>,
    /// When the memory was last accessed
    pub last_accessed: DateTime<Utc>,
    /// When the memory becomes valid (optional)
    pub valid_from: Option<DateTime<Utc>>,
    /// When the memory expires (optional)
    pub valid_until: Option<DateTime<Utc>>,
    /// Access count for frequency-based retrieval
    pub access_count: u32,
}

impl TemporalMarker {
    /// Create a new temporal marker
    pub fn new(created_at: DateTime<Utc>) -> Self {
        Self {
            created_at,
            last_accessed: created_at,
            valid_from: None,
            valid_until: None,
            access_count: 0,
        }
    }

    /// Check if valid at a specific time
    pub fn is_valid_at(&self, time: DateTime<Utc>) -> bool {
        let after_start = self.valid_from.map(|t| time >= t).unwrap_or(true);
        let before_end = self.valid_until.map(|t| time <= t).unwrap_or(true);
        after_start && before_end
    }

    /// Check if currently valid
    pub fn is_currently_valid(&self) -> bool {
        self.is_valid_at(Utc::now())
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Get age in days since creation
    pub fn age_days(&self) -> f64 {
        (Utc::now() - self.created_at).num_seconds() as f64 / 86400.0
    }

    /// Get recency (days since last access)
    pub fn recency_days(&self) -> f64 {
        (Utc::now() - self.last_accessed).num_seconds() as f64 / 86400.0
    }
}

// ============================================================================
// IMPORTANCE FLAGS
// ============================================================================

/// Importance flags for a memory (compact, bit-packed)
///
/// These flags enable fast filtering without content access.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportanceFlags {
    bits: u32,
}

impl ImportanceFlags {
    // Flag bit positions
    const EMOTIONAL: u32 = 1 << 0;
    const FREQUENTLY_ACCESSED: u32 = 1 << 1;
    const RECENTLY_CREATED: u32 = 1 << 2;
    const HAS_ASSOCIATIONS: u32 = 1 << 3;
    const USER_STARRED: u32 = 1 << 4;
    const HIGH_RETENTION: u32 = 1 << 5;
    const CONSOLIDATED: u32 = 1 << 6;
    const COMPRESSED: u32 = 1 << 7;

    /// Create empty flags
    pub fn empty() -> Self {
        Self { bits: 0 }
    }

    /// Create with all flags set
    pub fn all() -> Self {
        Self {
            bits: Self::EMOTIONAL
                | Self::FREQUENTLY_ACCESSED
                | Self::RECENTLY_CREATED
                | Self::HAS_ASSOCIATIONS
                | Self::USER_STARRED
                | Self::HIGH_RETENTION
                | Self::CONSOLIDATED
                | Self::COMPRESSED,
        }
    }

    /// Set emotional flag
    pub fn set_emotional(&mut self, value: bool) {
        if value {
            self.bits |= Self::EMOTIONAL;
        } else {
            self.bits &= !Self::EMOTIONAL;
        }
    }

    /// Check emotional flag
    pub fn is_emotional(&self) -> bool {
        self.bits & Self::EMOTIONAL != 0
    }

    /// Set frequently accessed flag
    pub fn set_frequently_accessed(&mut self, value: bool) {
        if value {
            self.bits |= Self::FREQUENTLY_ACCESSED;
        } else {
            self.bits &= !Self::FREQUENTLY_ACCESSED;
        }
    }

    /// Check frequently accessed flag
    pub fn is_frequently_accessed(&self) -> bool {
        self.bits & Self::FREQUENTLY_ACCESSED != 0
    }

    /// Set recently created flag
    pub fn set_recently_created(&mut self, value: bool) {
        if value {
            self.bits |= Self::RECENTLY_CREATED;
        } else {
            self.bits &= !Self::RECENTLY_CREATED;
        }
    }

    /// Check recently created flag
    pub fn is_recently_created(&self) -> bool {
        self.bits & Self::RECENTLY_CREATED != 0
    }

    /// Set has associations flag
    pub fn set_has_associations(&mut self, value: bool) {
        if value {
            self.bits |= Self::HAS_ASSOCIATIONS;
        } else {
            self.bits &= !Self::HAS_ASSOCIATIONS;
        }
    }

    /// Check has associations flag
    pub fn has_associations(&self) -> bool {
        self.bits & Self::HAS_ASSOCIATIONS != 0
    }

    /// Set user starred flag
    pub fn set_user_starred(&mut self, value: bool) {
        if value {
            self.bits |= Self::USER_STARRED;
        } else {
            self.bits &= !Self::USER_STARRED;
        }
    }

    /// Check user starred flag
    pub fn is_user_starred(&self) -> bool {
        self.bits & Self::USER_STARRED != 0
    }

    /// Set high retention flag
    pub fn set_high_retention(&mut self, value: bool) {
        if value {
            self.bits |= Self::HIGH_RETENTION;
        } else {
            self.bits &= !Self::HIGH_RETENTION;
        }
    }

    /// Check high retention flag
    pub fn has_high_retention(&self) -> bool {
        self.bits & Self::HIGH_RETENTION != 0
    }

    /// Set consolidated flag
    pub fn set_consolidated(&mut self, value: bool) {
        if value {
            self.bits |= Self::CONSOLIDATED;
        } else {
            self.bits &= !Self::CONSOLIDATED;
        }
    }

    /// Check consolidated flag
    pub fn is_consolidated(&self) -> bool {
        self.bits & Self::CONSOLIDATED != 0
    }

    /// Set compressed flag
    pub fn set_compressed(&mut self, value: bool) {
        if value {
            self.bits |= Self::COMPRESSED;
        } else {
            self.bits &= !Self::COMPRESSED;
        }
    }

    /// Check compressed flag
    pub fn is_compressed(&self) -> bool {
        self.bits & Self::COMPRESSED != 0
    }

    /// Get raw bits (for persistence)
    pub fn to_bits(&self) -> u32 {
        self.bits
    }

    /// Create from raw bits
    pub fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    /// Count number of flags set
    pub fn count_set(&self) -> u32 {
        self.bits.count_ones()
    }
}

impl Default for ImportanceFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// CONTENT TYPES AND STORAGE LOCATIONS
// ============================================================================

/// Type of content stored
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContentType {
    /// Plain text content
    Text,
    /// Source code
    Code,
    /// Structured data (JSON, etc.)
    StructuredData,
    /// Embedding vector
    Embedding,
    /// Metadata only
    Metadata,
    /// Binary data
    Binary,
    /// Reference to external resource
    ExternalReference,
}

/// Location where content is stored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageLocation {
    /// SQLite database
    SQLite {
        /// Table name
        table: String,
        /// Row ID
        row_id: i64,
    },
    /// Vector store
    VectorStore {
        /// Index name
        index: String,
        /// Vector ID
        id: u64,
    },
    /// File system
    FileSystem {
        /// File path
        path: PathBuf,
    },
    /// Inline (stored directly in the pointer)
    Inline {
        /// Raw data
        data: Vec<u8>,
    },
    /// Content was compressed/archived
    Archived {
        /// Archive identifier
        archive_id: String,
        /// Offset in archive
        offset: u64,
    },
}

// ============================================================================
// CONTENT POINTER
// ============================================================================

/// Pointer to actual content in distributed storage
///
/// This is the "neocortical" reference - pointing to where
/// the actual memory content lives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPointer {
    /// Type of content at this location
    pub content_type: ContentType,
    /// Where the content is stored
    pub storage_location: StorageLocation,
    /// Byte range within the content (for chunked storage)
    pub chunk_range: Option<(usize, usize)>,
    /// Size in bytes (for pre-allocation)
    pub size_bytes: Option<usize>,
    /// Content hash for integrity verification
    pub content_hash: Option<u64>,
}

impl ContentPointer {
    /// Create a pointer to SQLite storage
    pub fn sqlite(table: &str, row_id: i64, content_type: ContentType) -> Self {
        Self {
            content_type,
            storage_location: StorageLocation::SQLite {
                table: table.to_string(),
                row_id,
            },
            chunk_range: None,
            size_bytes: None,
            content_hash: None,
        }
    }

    /// Create a pointer to vector store
    pub fn vector_store(index: &str, id: u64) -> Self {
        Self {
            content_type: ContentType::Embedding,
            storage_location: StorageLocation::VectorStore {
                index: index.to_string(),
                id,
            },
            chunk_range: None,
            size_bytes: None,
            content_hash: None,
        }
    }

    /// Create a pointer to file system
    pub fn file_system(path: PathBuf, content_type: ContentType) -> Self {
        Self {
            content_type,
            storage_location: StorageLocation::FileSystem { path },
            chunk_range: None,
            size_bytes: None,
            content_hash: None,
        }
    }

    /// Create an inline pointer (for small data)
    pub fn inline(data: Vec<u8>, content_type: ContentType) -> Self {
        let size = data.len();
        Self {
            content_type,
            storage_location: StorageLocation::Inline { data },
            chunk_range: None,
            size_bytes: Some(size),
            content_hash: None,
        }
    }

    /// Set chunk range
    pub fn with_chunk_range(mut self, start: usize, end: usize) -> Self {
        self.chunk_range = Some((start, end));
        self
    }

    /// Set size
    pub fn with_size(mut self, size: usize) -> Self {
        self.size_bytes = Some(size);
        self
    }

    /// Set content hash
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.content_hash = Some(hash);
        self
    }

    /// Check if this is inline storage
    pub fn is_inline(&self) -> bool {
        matches!(self.storage_location, StorageLocation::Inline { .. })
    }
}

// ============================================================================
// INDEX LINK
// ============================================================================

/// Link between memory indices (associations)
///
/// These links form the "web" of memory associations,
/// enabling pattern completion and spreading activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexLink {
    /// Target barcode
    pub target_barcode: MemoryBarcode,
    /// Link strength (0.0 to 1.0)
    pub strength: f32,
    /// Type of association
    pub link_type: AssociationLinkType,
    /// When the link was created
    pub created_at: DateTime<Utc>,
    /// Number of times the link was activated
    pub activation_count: u32,
}

/// Type of association between memories (in hippocampal index)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssociationLinkType {
    /// Temporal co-occurrence
    Temporal,
    /// Semantic similarity
    Semantic,
    /// Causal relationship
    Causal,
    /// Part-of relationship
    PartOf,
    /// User-defined association
    UserDefined,
    /// Derived from same source
    SameSource,
}

impl IndexLink {
    /// Create a new link
    pub fn new(target: MemoryBarcode, strength: f32, link_type: AssociationLinkType) -> Self {
        Self {
            target_barcode: target,
            strength: strength.clamp(0.0, 1.0),
            link_type,
            created_at: Utc::now(),
            activation_count: 0,
        }
    }

    /// Strengthen the link (Hebbian learning)
    pub fn strengthen(&mut self, amount: f32) {
        self.strength = (self.strength + amount).clamp(0.0, 1.0);
        self.activation_count = self.activation_count.saturating_add(1);
    }

    /// Decay the link strength
    pub fn decay(&mut self, factor: f32) {
        self.strength *= factor.clamp(0.0, 1.0);
    }
}

// ============================================================================
// MEMORY INDEX (The "Hippocampal" Entry)
// ============================================================================

/// Compressed index dimension for semantic summary
/// (Smaller than full embedding for efficiency)
pub const INDEX_EMBEDDING_DIM: usize = 128;

/// Compact index entry - what the "hippocampus" stores
///
/// This is the core data structure that enables fast search.
/// It contains only enough information to:
/// 1. Identify the memory (barcode)
/// 2. Match semantic queries (compressed embedding)
/// 3. Filter by time and importance
/// 4. Find associated memories
/// 5. Locate the full content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIndex {
    /// Unique identifier (barcode)
    pub barcode: MemoryBarcode,
    /// Original memory ID (e.g., UUID from KnowledgeNode)
    pub memory_id: String,
    /// Compressed semantic embedding (smaller dimension)
    pub semantic_summary: Vec<f32>,
    /// Temporal information
    pub temporal_marker: TemporalMarker,
    /// Pointers to actual content
    pub content_pointers: Vec<ContentPointer>,
    /// Links to associated memories
    pub association_links: Vec<IndexLink>,
    /// Importance flags
    pub importance_flags: ImportanceFlags,
    /// Node type (fact, concept, etc.)
    pub node_type: String,
    /// Brief content preview (first ~100 chars)
    pub preview: String,
}

impl MemoryIndex {
    /// Create a new memory index
    pub fn new(
        barcode: MemoryBarcode,
        memory_id: String,
        node_type: String,
        created_at: DateTime<Utc>,
        preview: String,
    ) -> Self {
        Self {
            barcode,
            memory_id,
            semantic_summary: Vec::new(),
            temporal_marker: TemporalMarker::new(created_at),
            content_pointers: Vec::new(),
            association_links: Vec::new(),
            importance_flags: ImportanceFlags::empty(),
            node_type,
            preview: preview.chars().take(100).collect(),
        }
    }

    /// Set semantic summary (compressed embedding)
    pub fn with_semantic_summary(mut self, summary: Vec<f32>) -> Self {
        self.semantic_summary = summary;
        self
    }

    /// Add a content pointer
    pub fn add_content_pointer(&mut self, pointer: ContentPointer) {
        self.content_pointers.push(pointer);
    }

    /// Add an association link
    pub fn add_link(&mut self, link: IndexLink) {
        // Check for existing link to same target
        if let Some(existing) = self
            .association_links
            .iter_mut()
            .find(|l| l.target_barcode == link.target_barcode)
        {
            // Strengthen existing link
            existing.strengthen(link.strength * 0.5);
        } else {
            self.association_links.push(link);
        }
    }

    /// Remove weak links (below threshold)
    pub fn prune_weak_links(&mut self, threshold: f32) {
        self.association_links.retain(|l| l.strength >= threshold);
    }

    /// Record an access event
    pub fn record_access(&mut self) {
        self.temporal_marker.record_access();

        // Update importance flags based on access patterns
        if self.temporal_marker.access_count > 10 {
            self.importance_flags.set_frequently_accessed(true);
        }
    }

    /// Get total size of all content (for memory estimation)
    pub fn estimated_content_size(&self) -> usize {
        self.content_pointers
            .iter()
            .filter_map(|p| p.size_bytes)
            .sum()
    }

    /// Check if this index matches importance criteria
    pub fn matches_importance(&self, min_flags: u32) -> bool {
        self.importance_flags.to_bits() & min_flags == min_flags
    }
}

// ============================================================================
// INDEX QUERY
// ============================================================================

/// Query for searching the index
#[derive(Debug, Clone)]
pub struct IndexQuery {
    /// Semantic query embedding (optional)
    pub semantic_embedding: Option<Vec<f32>>,
    /// Text query (for preview matching)
    pub text_query: Option<String>,
    /// Time range filter
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Required importance flags
    pub required_flags: Option<ImportanceFlags>,
    /// Node type filter
    pub node_types: Option<Vec<String>>,
    /// Minimum semantic similarity threshold
    pub min_similarity: f32,
    /// Maximum results
    pub limit: usize,
}

impl IndexQuery {
    /// Create query from text
    pub fn from_text(query: &str) -> Self {
        Self {
            semantic_embedding: None,
            text_query: Some(query.to_string()),
            time_range: None,
            required_flags: None,
            node_types: None,
            min_similarity: 0.3,
            limit: 10,
        }
    }

    /// Create query from embedding
    pub fn from_embedding(embedding: Vec<f32>) -> Self {
        Self {
            semantic_embedding: Some(embedding),
            text_query: None,
            time_range: None,
            required_flags: None,
            node_types: None,
            min_similarity: 0.3,
            limit: 10,
        }
    }

    /// Set time range filter
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.time_range = Some((start, end));
        self
    }

    /// Set required importance flags
    pub fn with_required_flags(mut self, flags: ImportanceFlags) -> Self {
        self.required_flags = Some(flags);
        self
    }

    /// Set node type filter
    pub fn with_node_types(mut self, types: Vec<String>) -> Self {
        self.node_types = Some(types);
        self
    }

    /// Set minimum similarity
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold;
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

impl Default for IndexQuery {
    fn default() -> Self {
        Self {
            semantic_embedding: None,
            text_query: None,
            time_range: None,
            required_flags: None,
            node_types: None,
            min_similarity: 0.3,
            limit: 10,
        }
    }
}

// ============================================================================
// INDEX MATCH
// ============================================================================

/// Result of an index search
#[derive(Debug, Clone)]
pub struct IndexMatch {
    /// The matched index entry
    pub index: MemoryIndex,
    /// Semantic similarity score (0.0 to 1.0)
    pub semantic_score: f32,
    /// Text match score (0.0 to 1.0)
    pub text_score: f32,
    /// Temporal relevance score (0.0 to 1.0)
    pub temporal_score: f32,
    /// Importance score (0.0 to 1.0)
    pub importance_score: f32,
    /// Combined relevance score
    pub combined_score: f32,
}

impl IndexMatch {
    /// Create a new index match
    pub fn new(index: MemoryIndex) -> Self {
        Self {
            index,
            semantic_score: 0.0,
            text_score: 0.0,
            temporal_score: 0.0,
            importance_score: 0.0,
            combined_score: 0.0,
        }
    }

    /// Calculate combined score with weights
    pub fn calculate_combined(
        &mut self,
        semantic_weight: f32,
        text_weight: f32,
        temporal_weight: f32,
        importance_weight: f32,
    ) {
        self.combined_score = self.semantic_score * semantic_weight
            + self.text_score * text_weight
            + self.temporal_score * temporal_weight
            + self.importance_score * importance_weight;
    }
}

// ============================================================================
// FULL MEMORY (Retrieved Content)
// ============================================================================

/// Complete memory with all content retrieved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullMemory {
    /// The index entry
    pub barcode: MemoryBarcode,
    /// Original memory ID
    pub memory_id: String,
    /// Full text content
    pub content: String,
    /// Node type
    pub node_type: String,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last accessed time
    pub last_accessed: DateTime<Utc>,
    /// Full embedding (if available)
    pub embedding: Option<Vec<f32>>,
    /// All tags
    pub tags: Vec<String>,
    /// Source information
    pub source: Option<String>,
    /// FSRS scheduling state
    pub stability: f64,
    pub difficulty: f64,
    pub next_review: Option<DateTime<Utc>>,
    /// Retention strength
    pub retention_strength: f64,
}

// ============================================================================
// CONTENT STORE
// ============================================================================

/// Abstract content storage backend
///
/// This represents the "neocortex" - the distributed storage
/// where actual memory content lives.
pub struct ContentStore {
    /// SQLite connection (if available)
    sqlite_path: Option<PathBuf>,
    /// File storage root
    file_root: Option<PathBuf>,
    /// In-memory cache for recently accessed content
    cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size
    current_cache_size: Arc<RwLock<usize>>,
}

impl ContentStore {
    /// Create a new content store
    pub fn new() -> Self {
        Self {
            sqlite_path: None,
            file_root: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: 10 * 1024 * 1024, // 10 MB default
            current_cache_size: Arc::new(RwLock::new(0)),
        }
    }

    /// Configure SQLite backend
    pub fn with_sqlite(mut self, path: PathBuf) -> Self {
        self.sqlite_path = Some(path);
        self
    }

    /// Configure file storage backend
    pub fn with_file_root(mut self, path: PathBuf) -> Self {
        self.file_root = Some(path);
        self
    }

    /// Set maximum cache size
    pub fn with_max_cache(mut self, size_bytes: usize) -> Self {
        self.max_cache_size = size_bytes;
        self
    }

    /// Retrieve content from a pointer
    pub fn retrieve(&self, pointer: &ContentPointer) -> Result<Vec<u8>> {
        // Check cache first
        let cache_key = self.cache_key(pointer);
        if let Ok(cache) = self.cache.read()
            && let Some(data) = cache.get(&cache_key)
        {
            return Ok(data.clone());
        }

        // Retrieve from storage
        let data = match &pointer.storage_location {
            StorageLocation::Inline { data } => data.clone(),
            StorageLocation::SQLite { table, row_id } => {
                self.retrieve_from_sqlite(table, *row_id)?
            }
            StorageLocation::FileSystem { path } => self.retrieve_from_file(path)?,
            StorageLocation::VectorStore { index, id } => {
                self.retrieve_from_vector_store(index, *id)?
            }
            StorageLocation::Archived { archive_id, offset } => {
                self.retrieve_from_archive(archive_id, *offset)?
            }
        };

        // Apply chunk range if specified
        let data = if let Some((start, end)) = pointer.chunk_range {
            data.get(start..end).unwrap_or(&data).to_vec()
        } else {
            data
        };

        // Update cache
        self.cache_content(&cache_key, &data);

        Ok(data)
    }

    /// Generate cache key for a pointer
    fn cache_key(&self, pointer: &ContentPointer) -> String {
        match &pointer.storage_location {
            StorageLocation::Inline { .. } => "inline".to_string(),
            StorageLocation::SQLite { table, row_id } => format!("sqlite:{}:{}", table, row_id),
            StorageLocation::FileSystem { path } => format!("file:{}", path.display()),
            StorageLocation::VectorStore { index, id } => format!("vector:{}:{}", index, id),
            StorageLocation::Archived { archive_id, offset } => {
                format!("archive:{}:{}", archive_id, offset)
            }
        }
    }

    /// Add content to cache
    fn cache_content(&self, key: &str, data: &[u8]) {
        let data_size = data.len();

        // Don't cache if too large
        if data_size > self.max_cache_size / 4 {
            return;
        }

        if let Ok(mut cache) = self.cache.write()
            && let Ok(mut size) = self.current_cache_size.write()
        {
            // Evict if necessary
            while *size + data_size > self.max_cache_size && !cache.is_empty() {
                // Simple eviction: remove first entry
                if let Some(key_to_remove) = cache.keys().next().cloned() {
                    if let Some(removed) = cache.remove(&key_to_remove) {
                        *size = size.saturating_sub(removed.len());
                    }
                } else {
                    break;
                }
            }

            cache.insert(key.to_string(), data.to_vec());
            *size += data_size;
        }
    }

    /// Retrieve from SQLite (placeholder - to be integrated with Storage)
    fn retrieve_from_sqlite(&self, table: &str, row_id: i64) -> Result<Vec<u8>> {
        // This would connect to SQLite and retrieve the content
        // For now, return an error indicating it needs integration
        Err(HippocampalIndexError::ContentRetrievalFailed(format!(
            "SQLite retrieval not yet integrated: {}:{}",
            table, row_id
        )))
    }

    /// Retrieve from file system
    fn retrieve_from_file(&self, path: &PathBuf) -> Result<Vec<u8>> {
        std::fs::read(path).map_err(|e| {
            HippocampalIndexError::ContentRetrievalFailed(format!(
                "File read failed for {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Retrieve from vector store (placeholder)
    fn retrieve_from_vector_store(&self, index: &str, id: u64) -> Result<Vec<u8>> {
        Err(HippocampalIndexError::ContentRetrievalFailed(format!(
            "Vector store retrieval not yet integrated: {}:{}",
            index, id
        )))
    }

    /// Retrieve from archive (placeholder)
    fn retrieve_from_archive(&self, archive_id: &str, offset: u64) -> Result<Vec<u8>> {
        Err(HippocampalIndexError::ContentRetrievalFailed(format!(
            "Archive retrieval not yet implemented: {}:{}",
            archive_id, offset
        )))
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut size) = self.current_cache_size.write() {
            *size = 0;
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let entries = self.cache.read().map(|c| c.len()).unwrap_or(0);
        let size = self.current_cache_size.read().map(|s| *s).unwrap_or(0);
        (entries, size)
    }
}

impl Default for ContentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HIPPOCAMPAL INDEX CONFIGURATION
// ============================================================================

/// Configuration for the hippocampal index
#[derive(Debug, Clone)]
pub struct HippocampalIndexConfig {
    /// Dimension for semantic summaries (compressed embedding)
    pub summary_dimensions: usize,
    /// Minimum link strength to keep
    pub link_prune_threshold: f32,
    /// Days before "recently created" flag is cleared
    pub recently_created_days: u32,
    /// Access count threshold for "frequently accessed" flag
    pub frequently_accessed_threshold: u32,
    /// Weights for combined score calculation
    pub semantic_weight: f32,
    pub text_weight: f32,
    pub temporal_weight: f32,
    pub importance_weight: f32,
}

impl Default for HippocampalIndexConfig {
    fn default() -> Self {
        Self {
            summary_dimensions: INDEX_EMBEDDING_DIM, // 128 vs 384 for full
            link_prune_threshold: 0.1,
            recently_created_days: 7,
            frequently_accessed_threshold: 10,
            semantic_weight: 0.5,
            text_weight: 0.2,
            temporal_weight: 0.15,
            importance_weight: 0.15,
        }
    }
}

// ============================================================================
// HIPPOCAMPAL INDEX
// ============================================================================

/// Separates memory index from content storage
///
/// Based on Teyler and Rudy's hippocampal indexing theory:
/// - Index is compact and fast to search (hippocampus)
/// - Content is detailed and stored separately (neocortex)
pub struct HippocampalIndex {
    /// Index entries by barcode
    indices: Arc<RwLock<HashMap<String, MemoryIndex>>>,
    /// Content store reference
    content_store: ContentStore,
    /// Barcode generator
    barcode_generator: Arc<RwLock<BarcodeGenerator>>,
    /// Configuration
    config: HippocampalIndexConfig,
}

impl HippocampalIndex {
    /// Create a new hippocampal index
    pub fn new() -> Self {
        Self::with_config(HippocampalIndexConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: HippocampalIndexConfig) -> Self {
        Self {
            indices: Arc::new(RwLock::new(HashMap::new())),
            content_store: ContentStore::new(),
            barcode_generator: Arc::new(RwLock::new(BarcodeGenerator::new())),
            config,
        }
    }

    /// Set the content store
    pub fn with_content_store(mut self, store: ContentStore) -> Self {
        self.content_store = store;
        self
    }

    /// Index a new memory
    pub fn index_memory(
        &self,
        memory_id: &str,
        content: &str,
        node_type: &str,
        created_at: DateTime<Utc>,
        semantic_embedding: Option<Vec<f32>>,
    ) -> Result<MemoryBarcode> {
        // Generate barcode
        let barcode = {
            let mut generator = self
                .barcode_generator
                .write()
                .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;
            generator.generate(content, created_at)
        };

        // Create preview
        let preview: String = content.chars().take(100).collect();

        // Create index entry
        let mut index = MemoryIndex::new(
            barcode,
            memory_id.to_string(),
            node_type.to_string(),
            created_at,
            preview,
        );

        // Compress embedding if provided
        if let Some(embedding) = semantic_embedding {
            let summary = self.compress_embedding(&embedding);
            index.semantic_summary = summary;
        }

        // Set initial importance flags
        index.importance_flags.set_recently_created(true);

        // Add default content pointer (assumes SQLite storage)
        index.add_content_pointer(ContentPointer::sqlite(
            "knowledge_nodes",
            barcode.id as i64,
            ContentType::Text,
        ));

        // Store in index
        {
            let mut indices = self
                .indices
                .write()
                .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;
            indices.insert(memory_id.to_string(), index);
        }

        Ok(barcode)
    }

    /// Compress a full embedding to index dimensions
    fn compress_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        if embedding.len() <= self.config.summary_dimensions {
            return embedding.to_vec();
        }

        // Simple compression: take evenly spaced samples
        // In production, would use PCA or learned compression
        let step = embedding.len() as f32 / self.config.summary_dimensions as f32;
        let mut compressed = Vec::with_capacity(self.config.summary_dimensions);

        for i in 0..self.config.summary_dimensions {
            let idx = (i as f32 * step) as usize;
            compressed.push(embedding[idx.min(embedding.len() - 1)]);
        }

        // Normalize
        let norm: f32 = compressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut compressed {
                *x /= norm;
            }
        }

        compressed
    }

    /// Phase 1: Fast index search (hippocampus-like)
    pub fn search_indices(&self, query: &IndexQuery) -> Result<Vec<IndexMatch>> {
        let indices = self
            .indices
            .read()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        let mut matches: Vec<IndexMatch> = Vec::new();

        for index in indices.values() {
            // Apply filters
            if !self.passes_filters(index, query) {
                continue;
            }

            let mut match_result = IndexMatch::new(index.clone());

            // Calculate semantic score
            if let Some(ref query_embedding) = query.semantic_embedding
                && !index.semantic_summary.is_empty()
            {
                let query_compressed = self.compress_embedding(query_embedding);
                match_result.semantic_score =
                    self.cosine_similarity(&query_compressed, &index.semantic_summary);

                if match_result.semantic_score < query.min_similarity {
                    continue;
                }
            }

            // Calculate text score
            if let Some(ref text_query) = query.text_query {
                match_result.text_score = self.text_match_score(text_query, &index.preview);
            }

            // Calculate temporal score (recency)
            match_result.temporal_score = self.temporal_score(&index.temporal_marker);

            // Calculate importance score
            match_result.importance_score = self.importance_score(&index.importance_flags);

            // Calculate combined score
            match_result.calculate_combined(
                self.config.semantic_weight,
                self.config.text_weight,
                self.config.temporal_weight,
                self.config.importance_weight,
            );

            matches.push(match_result);
        }

        // Sort by combined score
        matches.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        matches.truncate(query.limit);

        Ok(matches)
    }

    /// Check if an index passes query filters
    fn passes_filters(&self, index: &MemoryIndex, query: &IndexQuery) -> bool {
        // Time range filter
        if let Some((start, end)) = query.time_range
            && (index.temporal_marker.created_at < start || index.temporal_marker.created_at > end)
        {
            return false;
        }

        // Importance flags filter
        if let Some(ref required) = query.required_flags
            && !index.matches_importance(required.to_bits())
        {
            return false;
        }

        // Node type filter
        if let Some(ref types) = query.node_types
            && !types.contains(&index.node_type)
        {
            return false;
        }

        true
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Calculate text match score
    fn text_match_score(&self, query: &str, preview: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let preview_lower = preview.to_lowercase();

        // Simple scoring: check for word matches
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let preview_words: Vec<&str> = preview_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let matches = query_words
            .iter()
            .filter(|q| preview_words.iter().any(|p| p.contains(*q)))
            .count();

        matches as f32 / query_words.len() as f32
    }

    /// Calculate temporal score (recency-based)
    fn temporal_score(&self, temporal: &TemporalMarker) -> f32 {
        let recency_days = temporal.recency_days();

        // Exponential decay with 14-day half-life
        let recency_score = 0.5_f32.powf(recency_days as f32 / 14.0);

        // Boost for frequently accessed
        let frequency_boost = if temporal.access_count > 10 {
            1.2
        } else if temporal.access_count > 5 {
            1.1
        } else {
            1.0
        };

        (recency_score * frequency_boost).min(1.0)
    }

    /// Calculate importance score from flags
    fn importance_score(&self, flags: &ImportanceFlags) -> f32 {
        let mut score = 0.0_f32;

        if flags.is_emotional() {
            score += 0.2;
        }
        if flags.is_frequently_accessed() {
            score += 0.2;
        }
        if flags.is_user_starred() {
            score += 0.25;
        }
        if flags.has_high_retention() {
            score += 0.15;
        }
        if flags.has_associations() {
            score += 0.1;
        }
        if flags.is_recently_created() {
            score += 0.1;
        }

        score.min(1.0)
    }

    /// Phase 2: Content retrieval (neocortex-like)
    pub fn retrieve_content(&self, index: &MemoryIndex) -> Result<FullMemory> {
        // For now, return a partial memory with available index data
        // Full retrieval would require integration with Storage
        Ok(FullMemory {
            barcode: index.barcode,
            memory_id: index.memory_id.clone(),
            content: index.preview.clone(), // Would retrieve full content
            node_type: index.node_type.clone(),
            created_at: index.temporal_marker.created_at,
            last_accessed: index.temporal_marker.last_accessed,
            embedding: None, // Would retrieve from vector store
            tags: Vec::new(),
            source: None,
            stability: 1.0,
            difficulty: 5.0,
            next_review: None,
            retention_strength: 1.0,
        })
    }

    /// Combined retrieval: search then retrieve
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<FullMemory>> {
        let index_query = IndexQuery::from_text(query).with_limit(limit);
        let matches = self.search_indices(&index_query)?;

        let mut memories = Vec::with_capacity(matches.len());
        for m in matches {
            // Record access
            if let Ok(mut indices) = self.indices.write()
                && let Some(index) = indices.get_mut(&m.index.memory_id)
            {
                index.record_access();
            }

            match self.retrieve_content(&m.index) {
                Ok(memory) => memories.push(memory),
                Err(e) => {
                    tracing::warn!(
                        "Failed to retrieve content for {}: {}",
                        m.index.memory_id,
                        e
                    )
                }
            }
        }

        Ok(memories)
    }

    /// Recall with semantic embedding
    pub fn recall_semantic(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<FullMemory>> {
        let query = IndexQuery::from_embedding(embedding)
            .with_limit(limit)
            .with_min_similarity(min_similarity);

        let matches = self.search_indices(&query)?;

        let mut memories = Vec::with_capacity(matches.len());
        for m in matches {
            if let Ok(memory) = self.retrieve_content(&m.index) {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    /// Add association between memories
    pub fn add_association(
        &self,
        from_id: &str,
        to_id: &str,
        strength: f32,
        link_type: AssociationLinkType,
    ) -> Result<()> {
        let mut indices = self
            .indices
            .write()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        // Get target barcode
        let to_barcode = indices
            .get(to_id)
            .map(|i| i.barcode)
            .ok_or_else(|| HippocampalIndexError::NotFound(to_id.to_string()))?;

        // Add link to source
        if let Some(from_index) = indices.get_mut(from_id) {
            let link = IndexLink::new(to_barcode, strength, link_type);
            from_index.add_link(link);

            // Update has_associations flag
            from_index.importance_flags.set_has_associations(true);
        } else {
            return Err(HippocampalIndexError::NotFound(from_id.to_string()));
        }

        Ok(())
    }

    /// Get associated memories (spreading activation)
    pub fn get_associations(&self, memory_id: &str, depth: usize) -> Result<Vec<IndexMatch>> {
        let indices = self
            .indices
            .read()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        let source = indices
            .get(memory_id)
            .ok_or_else(|| HippocampalIndexError::NotFound(memory_id.to_string()))?;

        let mut associations = Vec::new();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(memory_id.to_string());

        self.collect_associations(
            &indices,
            source,
            &mut associations,
            &mut visited,
            depth,
            1.0,
        );

        // Sort by combined score
        associations.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(associations)
    }

    /// Recursively collect associations
    #[allow(clippy::only_used_in_recursion)]
    fn collect_associations(
        &self,
        indices: &HashMap<String, MemoryIndex>,
        source: &MemoryIndex,
        associations: &mut Vec<IndexMatch>,
        visited: &mut std::collections::HashSet<String>,
        remaining_depth: usize,
        decay_factor: f32,
    ) {
        if remaining_depth == 0 {
            return;
        }

        for link in &source.association_links {
            // Find target by barcode
            if let Some((target_id, target)) = indices
                .iter()
                .find(|(_, i)| i.barcode == link.target_barcode)
            {
                if visited.contains(target_id) {
                    continue;
                }
                visited.insert(target_id.clone());

                let mut match_result = IndexMatch::new(target.clone());
                match_result.combined_score = link.strength * decay_factor;
                associations.push(match_result);

                // Recurse with decay
                self.collect_associations(
                    indices,
                    target,
                    associations,
                    visited,
                    remaining_depth - 1,
                    decay_factor * 0.7, // Decay for each hop
                );
            }
        }
    }

    /// Update importance flags for all indices
    pub fn update_importance_flags(&self) -> Result<()> {
        let mut indices = self
            .indices
            .write()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        let now = Utc::now();
        let recently_threshold = Duration::days(self.config.recently_created_days as i64);

        for index in indices.values_mut() {
            // Update recently_created flag
            let age = now - index.temporal_marker.created_at;
            index
                .importance_flags
                .set_recently_created(age < recently_threshold);

            // Update frequently_accessed flag
            index.importance_flags.set_frequently_accessed(
                index.temporal_marker.access_count >= self.config.frequently_accessed_threshold,
            );
        }

        Ok(())
    }

    /// Prune weak association links
    pub fn prune_weak_links(&self) -> Result<usize> {
        let mut indices = self
            .indices
            .write()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        let mut pruned_count = 0;
        for index in indices.values_mut() {
            let before = index.association_links.len();
            index.prune_weak_links(self.config.link_prune_threshold);
            pruned_count += before - index.association_links.len();
        }

        Ok(pruned_count)
    }

    /// Get index by memory ID
    pub fn get_index(&self, memory_id: &str) -> Result<Option<MemoryIndex>> {
        let indices = self
            .indices
            .read()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        Ok(indices.get(memory_id).cloned())
    }

    /// Remove an index
    pub fn remove_index(&self, memory_id: &str) -> Result<Option<MemoryIndex>> {
        let mut indices = self
            .indices
            .write()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        Ok(indices.remove(memory_id))
    }

    /// Get total number of indices
    pub fn len(&self) -> usize {
        self.indices.read().map(|i| i.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics
    pub fn stats(&self) -> HippocampalIndexStats {
        let indices = self.indices.read().ok();
        let (cache_entries, cache_size) = self.content_store.cache_stats();

        let (total_indices, total_links, total_pointers) = indices
            .map(|i| {
                let total = i.len();
                let links: usize = i.values().map(|idx| idx.association_links.len()).sum();
                let pointers: usize = i.values().map(|idx| idx.content_pointers.len()).sum();
                (total, links, pointers)
            })
            .unwrap_or((0, 0, 0));

        HippocampalIndexStats {
            total_indices,
            total_association_links: total_links,
            total_content_pointers: total_pointers,
            cache_entries,
            cache_size_bytes: cache_size,
            index_dimensions: self.config.summary_dimensions,
        }
    }
}

impl Default for HippocampalIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the hippocampal index
#[derive(Debug, Clone)]
pub struct HippocampalIndexStats {
    /// Total number of indices
    pub total_indices: usize,
    /// Total number of association links
    pub total_association_links: usize,
    /// Total number of content pointers
    pub total_content_pointers: usize,
    /// Number of entries in content cache
    pub cache_entries: usize,
    /// Size of content cache in bytes
    pub cache_size_bytes: usize,
    /// Index embedding dimensions
    pub index_dimensions: usize,
}

// ============================================================================
// MIGRATION SUPPORT
// ============================================================================

/// Result of migrating existing memories to indexed format
#[derive(Debug, Clone, Default)]
pub struct MigrationResult {
    /// Number of memories successfully migrated
    pub migrated: usize,
    /// Number of memories that failed migration
    pub failed: usize,
    /// Number of memories skipped (already indexed)
    pub skipped: usize,
    /// Error messages for failures
    pub errors: Vec<String>,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl HippocampalIndex {
    /// Migrate a KnowledgeNode to indexed format
    #[allow(clippy::too_many_arguments)]
    pub fn migrate_node(
        &self,
        node_id: &str,
        content: &str,
        node_type: &str,
        created_at: DateTime<Utc>,
        embedding: Option<Vec<f32>>,
        retention_strength: f64,
        sentiment_magnitude: f64,
    ) -> Result<MemoryBarcode> {
        // Check if already indexed
        if let Ok(indices) = self.indices.read()
            && indices.contains_key(node_id)
        {
            return Err(HippocampalIndexError::MigrationError(
                "Node already indexed".to_string(),
            ));
        }

        // Create the index
        let barcode = self.index_memory(node_id, content, node_type, created_at, embedding)?;

        // Update importance flags based on existing data
        if let Ok(mut indices) = self.indices.write()
            && let Some(index) = indices.get_mut(node_id)
        {
            // Set high retention flag if applicable
            if retention_strength > 0.7 {
                index.importance_flags.set_high_retention(true);
            }

            // Set emotional flag if applicable
            if sentiment_magnitude > 0.5 {
                index.importance_flags.set_emotional(true);
            }

            // Add SQLite content pointer
            index.content_pointers.clear();
            index.add_content_pointer(ContentPointer::sqlite(
                "knowledge_nodes",
                barcode.id as i64,
                ContentType::Text,
            ));
        }

        Ok(barcode)
    }

    /// Batch migrate multiple nodes
    pub fn migrate_batch(&self, nodes: Vec<MigrationNode>) -> MigrationResult {
        let start = std::time::Instant::now();
        let mut result = MigrationResult::default();

        for node in nodes {
            match self.migrate_node(
                &node.id,
                &node.content,
                &node.node_type,
                node.created_at,
                node.embedding,
                node.retention_strength,
                node.sentiment_magnitude,
            ) {
                Ok(_) => result.migrated += 1,
                Err(HippocampalIndexError::MigrationError(msg))
                    if msg == "Node already indexed" =>
                {
                    result.skipped += 1;
                }
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(format!("{}: {}", node.id, e));
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Create associations from semantic similarity
    pub fn create_semantic_associations(
        &self,
        memory_id: &str,
        similarity_threshold: f32,
        max_associations: usize,
    ) -> Result<usize> {
        let indices = self
            .indices
            .read()
            .map_err(|e| HippocampalIndexError::LockError(e.to_string()))?;

        let source = indices
            .get(memory_id)
            .ok_or_else(|| HippocampalIndexError::NotFound(memory_id.to_string()))?;

        if source.semantic_summary.is_empty() {
            return Ok(0);
        }

        // Snapshot the (id, summary) pairs under the read lock, then DROP the lock
        // before the O(n) cosine scan. Holding the read lock across the whole scan
        // blocks all writers (index_memory/add_association/migrate_node) for its
        // full duration on a large index.
        let source_summary = source.semantic_summary.clone();
        let pairs: Vec<(String, Vec<f32>)> = indices
            .iter()
            .filter(|(id, index)| id.as_str() != memory_id && !index.semantic_summary.is_empty())
            .map(|(id, index)| (id.clone(), index.semantic_summary.clone()))
            .collect();
        drop(indices); // release read lock BEFORE the expensive scan

        // Find similar memories (lock-free)
        let mut candidates: Vec<(String, f32)> = Vec::new();
        for (id, summary) in &pairs {
            let similarity = self.cosine_similarity(&source_summary, summary);
            if similarity >= similarity_threshold {
                candidates.push((id.clone(), similarity));
            }
        }

        // Sort by similarity
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_associations);

        // Add associations
        let mut added = 0;
        for (target_id, strength) in candidates {
            if self
                .add_association(
                    memory_id,
                    &target_id,
                    strength,
                    AssociationLinkType::Semantic,
                )
                .is_ok()
            {
                added += 1;
            }
        }

        Ok(added)
    }
}

/// Node data for migration
#[derive(Debug, Clone)]
pub struct MigrationNode {
    /// Node ID
    pub id: String,
    /// Content
    pub content: String,
    /// Node type
    pub node_type: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Embedding (optional)
    pub embedding: Option<Vec<f32>>,
    /// Retention strength
    pub retention_strength: f64,
    /// Sentiment magnitude
    pub sentiment_magnitude: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barcode_generation() {
        let mut generator = BarcodeGenerator::new();
        let now = Utc::now();

        let barcode1 = generator.generate("content1", now);
        let barcode2 = generator.generate("content2", now);

        assert_ne!(barcode1.id, barcode2.id);
        assert_ne!(barcode1.content_fingerprint, barcode2.content_fingerprint);
    }

    #[test]
    fn test_barcode_string_roundtrip() {
        let barcode = MemoryBarcode::new(12345, 0xABCD1234, 0xDEADBEEF);
        let s = barcode.to_string();
        let parsed = MemoryBarcode::from_string(&s).unwrap();

        assert_eq!(barcode, parsed);
    }

    #[test]
    fn test_importance_flags() {
        let mut flags = ImportanceFlags::empty();
        assert!(!flags.is_emotional());
        assert!(!flags.is_frequently_accessed());

        flags.set_emotional(true);
        assert!(flags.is_emotional());

        flags.set_frequently_accessed(true);
        assert!(flags.is_frequently_accessed());

        assert_eq!(flags.count_set(), 2);
    }

    #[test]
    fn test_temporal_marker() {
        let now = Utc::now();
        let mut marker = TemporalMarker::new(now);

        assert!(marker.is_currently_valid());
        assert_eq!(marker.access_count, 0);

        marker.record_access();
        assert_eq!(marker.access_count, 1);
    }

    #[test]
    fn test_index_memory() {
        let index = HippocampalIndex::new();
        let now = Utc::now();

        let barcode = index
            .index_memory(
                "test-id",
                "This is test content for indexing",
                "fact",
                now,
                None,
            )
            .unwrap();

        // barcode.id is u64, verify it was assigned
        let _ = barcode.id;
        assert_eq!(index.len(), 1);

        let retrieved = index.get_index("test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().node_type, "fact");
    }

    #[test]
    fn test_search_indices() {
        let index = HippocampalIndex::new();
        let now = Utc::now();

        index
            .index_memory("mem-1", "The quick brown fox", "fact", now, None)
            .unwrap();
        index
            .index_memory("mem-2", "jumps over the lazy dog", "fact", now, None)
            .unwrap();
        index
            .index_memory("mem-3", "completely unrelated content", "fact", now, None)
            .unwrap();

        let query = IndexQuery::from_text("fox").with_limit(10);
        let results = index.search_indices(&query).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].index.memory_id, "mem-1");
    }

    #[test]
    fn test_associations() {
        let index = HippocampalIndex::new();
        let now = Utc::now();

        index
            .index_memory("mem-1", "Content A", "fact", now, None)
            .unwrap();
        index
            .index_memory("mem-2", "Content B", "fact", now, None)
            .unwrap();

        index
            .add_association("mem-1", "mem-2", 0.8, AssociationLinkType::Semantic)
            .unwrap();

        let associations = index.get_associations("mem-1", 1).unwrap();
        assert_eq!(associations.len(), 1);
        assert_eq!(associations[0].index.memory_id, "mem-2");
    }

    #[test]
    fn test_compress_embedding() {
        let index = HippocampalIndex::new();

        // Create a 768-dim embedding (like BGE-base-en-v1.5)
        let embedding: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0).sin()).collect();

        let compressed = index.compress_embedding(&embedding);

        assert_eq!(compressed.len(), INDEX_EMBEDDING_DIM);

        // Check normalization
        let norm: f32 = compressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_migration() {
        let index = HippocampalIndex::new();
        let now = Utc::now();

        let nodes = vec![
            MigrationNode {
                id: "node-1".to_string(),
                content: "First node content".to_string(),
                node_type: "fact".to_string(),
                created_at: now,
                embedding: None,
                retention_strength: 0.8,
                sentiment_magnitude: 0.6,
            },
            MigrationNode {
                id: "node-2".to_string(),
                content: "Second node content".to_string(),
                node_type: "concept".to_string(),
                created_at: now,
                embedding: None,
                retention_strength: 0.3,
                sentiment_magnitude: 0.1,
            },
        ];

        let result = index.migrate_batch(nodes);

        assert_eq!(result.migrated, 2);
        assert_eq!(result.failed, 0);
        assert_eq!(index.len(), 2);

        // Check that flags were set correctly
        let idx1 = index.get_index("node-1").unwrap().unwrap();
        assert!(idx1.importance_flags.has_high_retention());
        assert!(idx1.importance_flags.is_emotional());

        let idx2 = index.get_index("node-2").unwrap().unwrap();
        assert!(!idx2.importance_flags.has_high_retention());
        assert!(!idx2.importance_flags.is_emotional());
    }

    #[test]
    fn test_content_pointer() {
        let sqlite_ptr = ContentPointer::sqlite("knowledge_nodes", 42, ContentType::Text);
        assert!(!sqlite_ptr.is_inline());

        let inline_ptr = ContentPointer::inline(vec![1, 2, 3, 4], ContentType::Binary);
        assert!(inline_ptr.is_inline());
        assert_eq!(inline_ptr.size_bytes, Some(4));
    }

    #[test]
    fn test_index_link_strengthen() {
        let barcode = MemoryBarcode::new(1, 0, 0);
        let mut link = IndexLink::new(barcode, 0.5, AssociationLinkType::Semantic);

        assert_eq!(link.activation_count, 0);

        link.strengthen(0.2);
        assert!(link.strength > 0.5);
        assert_eq!(link.activation_count, 1);
    }

    #[test]
    fn test_prune_weak_links() {
        let index = HippocampalIndex::new();
        let now = Utc::now();

        index
            .index_memory("mem-1", "Content A", "fact", now, None)
            .unwrap();
        index
            .index_memory("mem-2", "Content B", "fact", now, None)
            .unwrap();
        index
            .index_memory("mem-3", "Content C", "fact", now, None)
            .unwrap();

        // Add strong and weak links
        index
            .add_association("mem-1", "mem-2", 0.8, AssociationLinkType::Semantic)
            .unwrap();
        index
            .add_association("mem-1", "mem-3", 0.05, AssociationLinkType::Semantic)
            .unwrap();

        let pruned = index.prune_weak_links().unwrap();
        assert_eq!(pruned, 1);

        let idx = index.get_index("mem-1").unwrap().unwrap();
        assert_eq!(idx.association_links.len(), 1);
    }
}
