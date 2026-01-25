//! # Advanced Memory Features
//!
//! Bleeding-edge 2026 cognitive memory capabilities that make Vestige
//! the most advanced memory system in existence.
//!
//! ## Features
//!
//! - **Speculative Retrieval**: Predict what memories the user will need BEFORE they ask
//! - **Importance Evolution**: Memories evolve in importance based on actual usage
//! - **Semantic Compression**: Compress old memories while preserving meaning
//! - **Cross-Project Learning**: Learn patterns that apply across ALL projects
//! - **Intent Detection**: Understand WHY the user is doing something
//! - **Memory Chains**: Build chains of reasoning from memory
//! - **Adaptive Embedding**: Use DIFFERENT embedding models for different content
//! - **Memory Dreams**: Enhanced consolidation that creates NEW insights
//! - **Sleep Consolidation**: Automatic background consolidation during idle periods
//! - **Reconsolidation**: Memories become modifiable on retrieval (Nader's theory)

pub mod adaptive_embedding;
pub mod chains;
pub mod compression;
pub mod cross_project;
pub mod dreams;
pub mod importance;
pub mod intent;
pub mod prediction_error;
pub mod reconsolidation;
pub mod speculative;

// Re-exports for convenient access
pub use adaptive_embedding::{AdaptiveEmbedder, ContentType, EmbeddingStrategy, Language};
pub use chains::{ChainStep, ConnectionType, MemoryChainBuilder, MemoryPath, ReasoningChain};
pub use compression::{CompressedMemory, CompressionConfig, CompressionStats, MemoryCompressor};
pub use cross_project::{
    ApplicableKnowledge, CrossProjectLearner, ProjectContext, UniversalPattern,
};
pub use dreams::{
    ActivityStats,
    ActivityTracker,
    ConnectionGraph,
    ConnectionReason,
    ConnectionStats,
    ConsolidationReport,
    // Sleep Consolidation types
    ConsolidationScheduler,
    DreamConfig,
    // DreamMemory - input type for dreaming
    DreamMemory,
    DreamResult,
    MemoryConnection,
    MemoryDreamer,
    MemoryReplay,
    Pattern,
    PatternType,
    SynthesizedInsight,
};
pub use importance::{ImportanceDecayConfig, ImportanceScore, ImportanceTracker, UsageEvent};
pub use intent::{ActionType, DetectedIntent, IntentDetector, MaintenanceType, UserAction};
pub use reconsolidation::{
    AccessContext, AccessTrigger, AppliedModification, ChangeSummary, LabileState, MemorySnapshot,
    Modification, ReconsolidatedMemory, ReconsolidationManager, ReconsolidationStats,
    RelationshipType, RetrievalRecord,
};
pub use prediction_error::{
    CandidateMemory, CreateReason, EvaluationIntent, GateDecision, GateStats, MergeStrategy,
    PredictionErrorConfig, PredictionErrorGate, SimilarityResult, SupersedeReason, UpdateType,
    cosine_similarity,
};
pub use speculative::{PredictedMemory, PredictionContext, SpeculativeRetriever, UsagePattern};
