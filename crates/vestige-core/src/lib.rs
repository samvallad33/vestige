//! # Vestige Core
//!
//! Cognitive memory engine for AI systems. Implements bleeding-edge 2026 memory science:
//!
//! - **FSRS-6**: 21-parameter spaced repetition (30% more efficient than SM-2)
//! - **Dual-Strength Model**: Bjork & Bjork (1992) storage/retrieval strength
//! - **Semantic Embeddings**: Local fastembed v5 (nomic-embed-text-v1.5, 768 dimensions)
//! - **HNSW Vector Search**: USearch (20x faster than FAISS)
//! - **Temporal Memory**: Bi-temporal model with validity periods
//! - **Hybrid Search**: RRF fusion of keyword (BM25/FTS5) + semantic
//!
//! ## Advanced Features (Bleeding Edge 2026)
//!
//! - **Speculative Retrieval**: Predict needed memories before they're requested
//! - **Importance Evolution**: Memory importance evolves based on actual usage
//! - **Semantic Compression**: Compress old memories while preserving meaning
//! - **Cross-Project Learning**: Learn patterns that apply across all projects
//! - **Intent Detection**: Understand why the user is doing something
//! - **Memory Chains**: Build chains of reasoning from memory
//! - **Adaptive Embedding**: Different embedding strategies for different content
//! - **Memory Dreams**: Enhanced consolidation that creates new insights
//!
//! ## Neuroscience-Inspired Features
//!
//! - **Synaptic Tagging and Capture (STC)**: Memories can become important RETROACTIVELY
//!   based on subsequent events. Based on Frey & Morris (1997) finding that weak
//!   stimulation creates "synaptic tags" that can be captured by later PRPs.
//!   Successful STC observed even with 9-hour intervals.
//!
//! - **Context-Dependent Memory**: Encoding Specificity Principle (Tulving & Thomson, 1973).
//!   Memory retrieval is most effective when the retrieval context matches the encoding
//!   context. Captures temporal, topical, session, and emotional context.
//!
//! - **Multi-channel Importance Signaling**: Inspired by neuromodulator systems
//!   (dopamine, norepinephrine, acetylcholine). Different signals capture different
//!   types of importance: novelty (prediction error), arousal (emotional intensity),
//!   reward (positive outcomes), and attention (focused learning).
//!
//! - **Hippocampal Indexing**: Based on Teyler & Rudy (2007) indexing theory.
//!   The hippocampus stores INDICES (pointers), not content. Content is distributed
//!   across neocortex. Enables fast search with compact index while storing full
//!   content separately. Two-phase retrieval: fast index search, then content retrieval.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use vestige_core::{Storage, IngestInput, Rating};
//!
//! // Create storage (uses default platform-specific location)
//! let mut storage = Storage::new(None)?;
//!
//! // Ingest a memory
//! let input = IngestInput {
//!     content: "The mitochondria is the powerhouse of the cell".to_string(),
//!     node_type: "fact".to_string(),
//!     ..Default::default()
//! };
//! let node = storage.ingest(input)?;
//!
//! // Review the memory
//! let updated = storage.mark_reviewed(&node.id, Rating::Good)?;
//!
//! // Search semantically
//! let results = storage.semantic_search("cellular energy", 10, 0.5)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `embeddings` (default): Enable local embedding generation with fastembed
//! - `vector-search` (default): Enable HNSW vector search with USearch
//! - `full`: All features including MCP protocol support
//! - `mcp`: Model Context Protocol for Claude integration

#![cfg_attr(docsrs, feature(doc_cfg))]
// Only warn about missing docs for public items exported from the crate root
// Internal struct fields and enum variants don't need documentation
#![warn(rustdoc::missing_crate_level_docs)]

// ============================================================================
// MODULES
// ============================================================================

pub mod consolidation;
pub mod fsrs;
pub mod memory;
pub mod storage;

#[cfg(feature = "embeddings")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddings")))]
pub mod embeddings;

#[cfg(feature = "vector-search")]
#[cfg_attr(docsrs, doc(cfg(feature = "vector-search")))]
pub mod search;

/// Advanced memory features - bleeding edge 2026 cognitive capabilities
pub mod advanced;

/// Codebase memory - Vestige's killer differentiator for AI code understanding
pub mod codebase;

/// Neuroscience-inspired memory mechanisms
///
/// Implements cutting-edge neuroscience findings including:
/// - Synaptic Tagging and Capture (STC) for retroactive importance
/// - Context-dependent memory retrieval
/// - Spreading activation networks
pub mod neuroscience;

// ============================================================================
// PUBLIC API RE-EXPORTS
// ============================================================================

// Memory types
pub use memory::{
    ConsolidationResult, EmbeddingResult, IngestInput, KnowledgeNode, MatchType, MemoryStats,
    NodeType, RecallInput, SearchMode, SearchResult, SimilarityResult, TemporalRange,
    // GOD TIER 2026: New types
    EdgeType, KnowledgeEdge, MemoryScope, MemorySystem,
};

// FSRS-6 algorithm
pub use fsrs::{
    initial_difficulty,
    initial_stability,
    next_interval,
    // Core functions for advanced usage
    retrievability,
    retrievability_with_decay,
    FSRSParameters,
    FSRSScheduler,
    FSRSState,
    LearningState,
    PreviewResults,
    Rating,
    ReviewResult,
};

// Storage layer
pub use storage::{
    ConsolidationHistoryRecord, InsightRecord, IntentionRecord, Result, SmartIngestResult, Storage,
    StorageError,
};

// Consolidation (sleep-inspired memory processing)
pub use consolidation::SleepConsolidation;

// Advanced features (bleeding edge 2026)
pub use advanced::{
    AccessContext,
    AccessTrigger,
    ActionType,
    ActivityStats,
    ActivityTracker,
    // Adaptive embedding
    AdaptiveEmbedder,
    ApplicableKnowledge,
    AppliedModification,
    ChainStep,
    ChangeSummary,
    CompressedMemory,
    CompressionConfig,
    CompressionStats,
    ConnectionGraph,
    ConnectionReason,
    ConnectionStats,
    ConnectionType,
    ConsolidationReport,
    // Sleep consolidation (automatic background consolidation)
    ConsolidationScheduler,
    ContentType,
    // Cross-project learning
    CrossProjectLearner,
    DetectedIntent,
    DreamConfig,
    // DreamMemory - input type for dreaming
    DreamMemory,
    DreamResult,
    EmbeddingStrategy,
    ImportanceDecayConfig,
    ImportanceScore,
    // Importance tracking
    ImportanceTracker,
    // Intent detection
    IntentDetector,
    LabileState,
    Language,
    MaintenanceType,
    // Memory chains
    MemoryChainBuilder,
    // Memory compression
    MemoryCompressor,
    MemoryConnection,
    // Memory dreams
    MemoryDreamer,
    MemoryPath,
    MemoryReplay,
    MemorySnapshot,
    Modification,
    Pattern,
    PatternType,
    PredictedMemory,
    PredictionContext,
    ProjectContext,
    ReasoningChain,
    ReconsolidatedMemory,
    // Reconsolidation (memories become modifiable on retrieval)
    ReconsolidationManager,
    ReconsolidationStats,
    RelationshipType,
    RetrievalRecord,
    // Speculative retrieval
    SpeculativeRetriever,
    SynthesizedInsight,
    UniversalPattern,
    UsageEvent,
    UsagePattern,
    UserAction,
    // Prediction Error Gating (solves bad vs good similar memory problem)
    CandidateMemory,
    CreateReason,
    EvaluationIntent,
    GateDecision,
    GateStats,
    MergeStrategy,
    PredictionErrorConfig,
    PredictionErrorGate,
    SimilarityResult as PredictionSimilarityResult,
    SupersedeReason,
    UpdateType,
};

// Codebase memory (Vestige's killer differentiator)
pub use codebase::{
    // Types
    ArchitecturalDecision,
    BugFix,
    CodePattern,
    CodebaseError,
    // Main interface
    CodebaseMemory,
    CodebaseNode,
    CodebaseStats,
    // Watcher
    CodebaseWatcher,
    CodingPreference,
    // Git analysis
    CommitInfo,
    // Context
    ContextCapture,
    FileContext,
    FileEvent,
    FileRelationship,
    Framework,
    GitAnalyzer,
    GitContext,
    HistoryAnalysis,
    LearningResult,
    // Patterns
    PatternDetector,
    PatternMatch,
    PatternSuggestion,
    ProjectType,
    RelatedFile,
    // Relationships
    RelationshipGraph,
    RelationshipTracker,
    WatcherConfig,
    WorkContext,
    WorkingContext,
};

// Neuroscience-inspired memory mechanisms
pub use neuroscience::{
    AccessPattern,
    AccessibilityCalculator,
    // Spreading Activation (Associative Memory Network)
    ActivatedMemory,
    ActivationConfig,
    ActivationNetwork,
    ActivationNode,
    ArousalExplanation,
    ArousalSignal,
    AssociatedMemory,
    AssociationEdge,
    AssociationLinkType,
    AttentionExplanation,
    AttentionSignal,
    BarcodeGenerator,
    BatchUpdateResult,
    CaptureResult,
    CaptureWindow,
    CapturedMemory,
    CompetitionCandidate,
    CompetitionConfig,
    CompetitionEvent,
    CompetitionManager,
    CompetitionResult,
    CompositeWeights,
    ConsolidationPriority,
    ContentPointer,
    ContentStore,
    ContentType as HippocampalContentType,
    Context as ImportanceContext,
    // Context-Dependent Memory (Encoding Specificity Principle)
    ContextMatcher,
    ContextReinstatement,
    ContextWeights,
    DecayFunction,
    EmotionalContext,
    EmotionalMarker,
    EncodingContext,
    FullMemory,
    // Hippocampal Indexing (Teyler & Rudy, 2007)
    HippocampalIndex,
    HippocampalIndexConfig,
    HippocampalIndexError,
    ImportanceCluster,
    ImportanceConsolidationConfig,
    ImportanceEncodingConfig,
    ImportanceEvent,
    ImportanceEventType,
    ImportanceFlags,
    ImportanceRetrievalConfig,
    // Multi-channel Importance Signaling (Neuromodulator-inspired)
    ImportanceSignals,
    IndexLink,
    IndexMatch,
    IndexQuery,
    LifecycleSummary,
    LinkType,
    MarkerType,
    MemoryBarcode,
    MemoryIndex,
    MemoryLifecycle,
    // Memory States (accessibility continuum)
    MemoryState,
    MemoryStateInfo,
    MigrationNode,
    MigrationResult,
    NoveltyExplanation,
    NoveltySignal,
    Outcome,
    OutcomeType,
    RecencyBucket,
    RewardExplanation,
    RewardSignal,
    ScoredMemory,
    SentimentAnalyzer,
    SentimentResult,
    Session as AttentionSession,
    SessionContext,
    StateDecayConfig,
    StatePercentages,
    StateTimeAccumulator,
    StateTransition,
    StateTransitionReason,
    StateUpdateService,
    StorageLocation,
    // Synaptic Tagging and Capture (retroactive importance)
    SynapticTag,
    SynapticTaggingConfig,
    SynapticTaggingSystem,
    TaggingStats,
    TemporalContext,
    TemporalMarker,
    TimeOfDay,
    TopicalContext,
    INDEX_EMBEDDING_DIM,
};

// Embeddings (when feature enabled)
#[cfg(feature = "embeddings")]
pub use embeddings::{
    cosine_similarity, euclidean_distance, Embedding, EmbeddingError, EmbeddingService,
    EMBEDDING_DIMENSIONS,
};

// Search (when feature enabled)
#[cfg(feature = "vector-search")]
pub use search::{
    linear_combination,
    reciprocal_rank_fusion,
    HybridSearchConfig,
    // Hybrid search
    HybridSearcher,
    // Keyword search
    KeywordSearcher,
    VectorIndex,
    VectorIndexConfig,
    VectorIndexStats,
    VectorSearchError,
    // GOD TIER 2026: Reranking
    Reranker,
    RerankerConfig,
    RerankerError,
    RerankedResult,
};

// ============================================================================
// VERSION INFO
// ============================================================================

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// FSRS algorithm version (6 = 21 parameters)
pub const FSRS_VERSION: u8 = 6;

/// Default embedding model (2026 GOD TIER: nomic-embed-text-v1.5)
/// 8192 token context, Matryoshka support, fully open source
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

// ============================================================================
// PRELUDE
// ============================================================================

/// Convenient imports for common usage
pub mod prelude {
    pub use crate::{
        ConsolidationResult, FSRSScheduler, FSRSState, IngestInput, KnowledgeNode, MemoryStats,
        NodeType, Rating, RecallInput, Result, SearchMode, Storage, StorageError,
    };

    #[cfg(feature = "embeddings")]
    pub use crate::{Embedding, EmbeddingService};

    #[cfg(feature = "vector-search")]
    pub use crate::{HybridSearcher, VectorIndex};

    // Advanced features
    pub use crate::{
        ActivityTracker,
        AdaptiveEmbedder,
        ConnectionGraph,
        ConsolidationReport,
        // Sleep consolidation
        ConsolidationScheduler,
        CrossProjectLearner,
        ImportanceTracker,
        IntentDetector,
        LabileState,
        MemoryChainBuilder,
        MemoryCompressor,
        MemoryDreamer,
        MemoryReplay,
        Modification,
        PredictedMemory,
        ReconsolidatedMemory,
        // Reconsolidation
        ReconsolidationManager,
        SpeculativeRetriever,
        // Prediction Error Gating
        PredictionErrorGate,
        GateDecision,
        EvaluationIntent,
    };

    // Codebase memory
    pub use crate::{
        ArchitecturalDecision, BugFix, CodePattern, CodebaseMemory, CodebaseNode, WorkingContext,
    };

    // Neuroscience-inspired mechanisms
    pub use crate::{
        AccessPattern,
        AccessibilityCalculator,
        ArousalSignal,
        AttentionSession,
        AttentionSignal,
        BarcodeGenerator,
        CapturedMemory,
        CompetitionManager,
        CompositeWeights,
        ConsolidationPriority,
        ContentPointer,
        ContentStore,
        // Context-dependent memory
        ContextMatcher,
        ContextReinstatement,
        EmotionalContext,
        EncodingContext,
        // Hippocampal indexing (Teyler & Rudy)
        HippocampalIndex,
        ImportanceCluster,
        ImportanceContext,
        ImportanceEvent,
        // Multi-channel importance signaling
        ImportanceSignals,
        IndexMatch,
        IndexQuery,
        MemoryBarcode,
        MemoryIndex,
        MemoryLifecycle,
        // Memory states
        MemoryState,
        NoveltySignal,
        Outcome,
        OutcomeType,
        RewardSignal,
        ScoredMemory,
        SessionContext,
        StateUpdateService,
        SynapticTag,
        SynapticTaggingSystem,
        TemporalContext,
        TopicalContext,
    };
}
