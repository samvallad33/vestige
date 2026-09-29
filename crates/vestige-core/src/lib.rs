//! # Vestige Core
//!
//! Cognitive memory engine for AI systems. Implements bleeding-edge 2026 memory science:
//!
//! - **FSRS-6**: 21-parameter spaced repetition (30% more efficient than SM-2)
//! - **Dual-Strength Model**: Bjork & Bjork (1992) storage/retrieval strength
//! - **Temporal Memory**: Bi-temporal model with validity periods
//! - **Hybrid Search Fusion**: RRF fusion of ranked result lists
//!
//! ## Advanced Features (Bleeding Edge 2026)
//!
//! - **Speculative Retrieval**: Predict needed memories before they're requested
//! - **Importance Evolution**: Memory importance evolves based on actual usage
//! - **Semantic Compression**: Compress old memories while preserving meaning
//! - **Cross-Project Learning**: Learn patterns that apply across all projects
//! - **Intent Detection**: Understand why the user is doing something
//! - **Memory Chains**: Build chains of reasoning from memory
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
//! let storage = open_storage(None)?;
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
//! - `bundled-sqlite` (default): Bundle SQLite
//! - `codebase-git` (default): Git history analysis for codebase memory
//! - `encryption`: Encrypted SQLite via SQLCipher
//! - `connectors`: Network-backed external-source connectors
//! - `cloud-sync`: Hosted managed-sync backend

#![cfg_attr(docsrs, feature(doc_cfg))]
// Only warn about missing docs for public items exported from the crate root
// Internal struct fields and enum variants don't need documentation
#![warn(rustdoc::missing_crate_level_docs)]

// ============================================================================
// MODULES
// ============================================================================

/// Actor provenance (#252 Phase A): process did:key identity, the
/// operator-controlled versioned role policy, and neutral resolution.
pub mod actor;

/// Optional `vestige.toml` configuration (Phase 2: Configurable Output).
pub mod config;
pub mod connectors;
pub mod consolidation;
/// Durable profile contracts for local embedding vector spaces.
pub mod fsrs;
/// SQLite FTS5 query sanitization — quarantined with the SQLite backend
/// behind `legacy-sqlite` (build/t5-legacy-isolation).
#[cfg(feature = "legacy-sqlite")]
pub mod fts;
/// Evidence-aware future intentions with deterministic local evaluation.
pub mod intention_graph;
pub mod memory;
pub mod security;
pub mod storage;

/// Agent Black Box, Memory Receipts & Memory PRs — the cognitive flight
/// recorder, immune system, and reviewable-diff model for agent memory.
pub mod trace;

/// Hybrid and temporal search over ranked result lists.
pub mod search;

/// Advanced memory features - bleeding edge 2026 cognitive capabilities
pub mod advanced;

/// Codebase memory - Vestige's killer differentiator for AI code understanding
pub mod codebase;
pub mod projection;

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
    ConsolidationResult,
    // GOD TIER 2026: New types
    EdgeType,
    EmbeddingResult,
    IngestInput,
    KnowledgeEdge,
    KnowledgeNode,
    MatchType,
    MemoryScope,
    MemoryStats,
    MemorySystem,
    NodeType,
    RecallInput,
    SchemaIntrospection,
    SearchMode,
    SearchResult,
    SimilarityResult,
    SourceEnvelope,
    TableIntrospection,
    TemporalRange,
};

// Local secret-detection primitives used by every memory write boundary.
pub use security::{SecretConfidence, SecretFinding, SecretKind, SecretPolicy, scan_secrets};

// FSRS-6 algorithm
pub use fsrs::{
    FSRSParameters,
    FSRSScheduler,
    FSRSState,
    LearningState,
    PreviewResults,
    Rating,
    ReviewResult,
    initial_difficulty,
    initial_stability,
    next_interval,
    // Core functions for advanced usage
    retrievability,
    retrievability_with_decay,
};

// Configuration (vestige.toml output profiles / defaults)
pub use config::{CONFIG_FILE, OutputConfig, OutputDefaults, OutputProfile, VestigeConfig};

// Actor provenance (#252 Phase A)
pub use actor::{
    ACTOR_KEY_FILE, ActorIdentityError, ActorPolicySnapshot, ED25519_MULTICODEC_PREFIX,
    FLAT_POLICY_V1, INITIAL_POLICY_VERSION, MAX_AGGREGATE_ENDORSEMENT_WEIGHT, MAX_ROLE_WEIGHT,
    NEUTRAL_WEIGHT, ProcessActor, ResolutionDisposition, RoleResolution, UNATTRIBUTED_ROLE,
    actor_key_path_for_data_dir, base58btc_decode, base58btc_encode, bounded_aggregate,
    did_key_from_ed25519_public_key, ed25519_public_key_from_did_key, endorsement_event_id,
    revision_digest,
};

// Agent Black Box / Receipts / Memory PRs (the cognitive flight recorder)
pub use trace::ActorProvenance;
pub use trace::{
    BACKFILL_RECEIPT_CLAIM_BOUNDARY, BACKFILL_RECEIPT_SCHEMA_V1, BackfillCandidateEvidence,
    DecayRisk, HIGH_TRUST_FLOOR, LOW_CONFIDENCE_FLOOR, MemoryPr, MemoryPrAction, MemoryPrKind,
    MemoryPrStatus, MemoryTraceEvent, Receipt, ReceiptEvidence, ReceiptMutation, ReviewMode,
    RiskClass, RiskSignal, StrengthDelta, SuppressReason, SuppressedReceiptEntry,
    SynapticCaptureCandidate, SynapticCaptureDisposition, SynapticCaptureEvidence,
    SynapticCaptureTrigger, SynapticCaptureWindow, SynapticStrengthChange,
    WALK_RECEIPT_CLAIM_BOUNDARY, WALK_RECEIPT_SCHEMA_V1, WriteContext, WriteSource, classify_write,
};

// Storage layer
// Storage: backend-agnostic surface (always available).
pub use storage::LegacySqliteDisabled;
// v3 SQLite guard: refuses the legacy engine at every 4.0 entry point
// (ungated — the refusal must fire in Strata-only builds too).
pub use storage::v3_guard::{
    MIGRATION_HINT, SQLITE_MAGIC, V3Info, detect_v3, ensure_not_v3, v3_rw_guard_armed,
};
#[cfg(feature = "legacy-sqlite")]
pub use storage::{
    ClassificationResult, Domain, HealthStatus, LocalMemoryStore, MemoryEdge, MemoryRecord,
    MemoryStore, MemoryStoreError, MemoryStoreResult, ModelSignature, SchedulingState, SearchQuery,
    StoreStats, open_storage,
};

// Storage: legacy SQLite surface (quarantined behind `legacy-sqlite`,
// build/t5-legacy-isolation; default ON, flips off when STRATA lands).
#[cfg(feature = "legacy-sqlite")]
pub use storage::{
    ACCESS_LOG_RETENTION_DAYS,
    AgentRunSummary,
    BLAST_LINK_TYPES,
    BLAST_MAX_DEPTH,
    BLAST_SCAN_NODE_CAP,
    BlastAffected,
    BlastReport,
    CompositionEventRecord,
    CompositionMemberRecord,
    CompositionNeighborRecord,
    CompositionOutcomeRecord,
    ConnectionRecord,
    ConnectorCursor,
    ConsolidationHistoryRecord,
    CounterfactualReplayResult,
    DEFAULT_MEMORY_SCOPE,
    DreamHistoryRecord,
    DurableCounterfactualReplay,
    DurableRetrievalReplayCapsule,
    DurableSynapticCapture,
    DurableSynapticPairReceipt,
    EmbeddingProfileIntegrityManifest,
    EmbeddingProfileMigrationNodeCheckpoint,
    EmbeddingProfileMigrationRecord,
    EmbeddingProfileVector,
    FrozenReplayItem,
    HygieneNodeSummary,
    HygieneSnapshot,
    InsightRecord,
    IntentionRecord,
    NeverComposedCandidate,
    PORTABLE_ARCHIVE_FORMAT,
    PendingMemoryMutationDecision,
    PendingMemoryMutationEffect,
    PortableArchive,
    PortableImportMode,
    PortableImportReport,
    PortableSyncReport,
    REPLAY_ALGORITHM_VERSION,
    REPLAY_CLAIM_BOUNDARY,
    REPLAY_SCHEMA_VERSION,
    REPLAY_SELECTION_BOUNDARY,
    ReconcileReport,
    ReplayBuildError,
    ReplayDecayRisk,
    ReplayEvidenceItemSummary,
    ReplayEvidenceSetSummary,
    ReplayInfluence,
    ReplayInvalidationReason,
    ReplayMaterializationCheck,
    ReplayPrivacyInvalidation,
    ReplayPrivacyState,
    Result,
    RetireOutcome,
    RetrievalReplayCapsuleDraft,
    RetrievalReplayCapsuleSummary,
    RetrievalReplayItemDraft,
    SYNAPTIC_CAPTURE_ALGORITHM_V1,
    SYNAPTIC_CAPTURE_ALGORITHM_V2,
    SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
    SYNAPTIC_CAPTURE_SCHEMA_V1,
    SYNAPTIC_CAPTURE_SCHEMA_V2,
    SYNAPTIC_CONTEXT_ALGORITHM_V1,
    SYNAPTIC_CONTEXT_THRESHOLD_V1,
    // Note: storage::SearchResult is intentionally not re-exported here to avoid
    // collision with memory::SearchResult. Use vestige_core::storage::SearchResult directly.
    SmartIngestResult,
    SourceUpsertOutcome,
    SourceUpsertResult,
    SqliteMemoryStore,
    StateTransitionRecord,
    Storage,
    StorageError,
    StoredCounterfactualReplay,
    SynapticCapturePolicy,
    SynapticCaptureRequest,
    SynapticImportanceEvent,
    SynapticIngestOutcome,
    SynapticIngestRequest,
    SynapticSignalSnapshot,
    TagVocabulary,
    ablate_frozen_context,
    commit_sha_of,
    db_path_for_data_dir,
    default_db_path,
    private_evidence_digest,
    replay_evidence_slot,
    replay_idempotency_key,
    replay_policy_digest,
};

// Embedding profile contracts are feature-independent so profile discovery,
// storage metadata, and explicit install workflows remain available in a
// lightweight build without an inference runtime.

// Consolidation (sleep-inspired memory processing)
pub use consolidation::SleepConsolidation;
pub use consolidation::{
    CreativeConnection, CreativeConnectionType, DreamEngine, DreamInsight, DreamPhase,
    FourPhaseDreamResult, PhaseResult, TriageCategory, TriagedMemory,
};
// Dream compile is wired to the legacy SQLite store (`build/t5-legacy-isolation`).
#[cfg(feature = "legacy-sqlite")]
pub use consolidation::{
    DreamCompileConfig, DreamCompilePhase, DreamCompilePr, DreamCompileReport, run_dream_compile,
};

// Advanced features (bleeding edge 2026)
pub use advanced::{
    AccessContext,
    AccessTrigger,
    ActionType,
    ActivityStats,
    ActivityTracker,
    ApplicableKnowledge,
    AppliedModification,
    // Prediction Error Gating (solves bad vs good similar memory problem)
    CandidateMemory,
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
    CreateReason,
    // Cross-project learning
    CrossProjectLearner,
    DetectedIntent,
    DiscoveredConnection,
    DiscoveredConnectionType,
    DreamConfig,
    // DreamMemory - input type for dreaming
    DreamMemory,
    DreamResult,
    EvaluationIntent,
    GateDecision,
    GateStats,
    ImportanceDecayConfig,
    ImportanceScore,
    // Importance tracking
    ImportanceTracker,
    // Intent detection
    IntentDetector,
    // Reconsolidation (memories become modifiable on retrieval)
    LabileCandidate,
    LabileState,
    MaintenanceType,
    // Merge / Supersede controls (Phase 3)
    MatchClass,
    MatchSignals,
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
    MergeCandidate,
    MergeOperation,
    MergePlan,
    MergePolicy,
    MergeStrategy,
    Modification,
    Pattern,
    PatternType,
    PlanKind,
    PredictedMemory,
    PredictionContext,
    PredictionErrorConfig,
    PredictionErrorGate,
    ProjectContext,
    ReasoningChain,
    ReconsolidatedMemory,
    ReconsolidationManager,
    ReconsolidationMeta,
    ReconsolidationStats,
    RelationshipType,
    RetrievalRecord,
    SimilarityResult as PredictionSimilarityResult,
    // Speculative retrieval
    SpeculativeRetriever,
    SupersedeReason,
    SynthesizedInsight,
    UniversalPattern,
    UpdateType,
    UsageEvent,
    UsagePattern,
    UserAction,
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
    // Emotional Memory (Brown & Kulik 1977, Bower 1981, LaBar & Cabeza 2006)
    EmotionCategory,
    EmotionalContext,
    EmotionalEvaluation,
    EmotionalMarker,
    EmotionalMemory,
    EmotionalMemoryStats,
    EncodingContext,
    FullMemory,
    // Hippocampal Indexing (Teyler & Rudy, 2007)
    HippocampalIndex,
    HippocampalIndexConfig,
    HippocampalIndexError,
    INDEX_EMBEDDING_DIM,
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
};

// Search fusion (RRF + linear combination over ranked result lists)
pub use search::{HybridSearchConfig, HybridSearcher, linear_combination, reciprocal_rank_fusion};

// ============================================================================
// VERSION INFO
// ============================================================================

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// FSRS algorithm version (6 = 21 parameters)
pub const FSRS_VERSION: u8 = 6;

// ============================================================================
// PRELUDE
// ============================================================================

/// Convenient imports for common usage
pub mod prelude {
    pub use crate::{
        ConsolidationResult, FSRSScheduler, FSRSState, IngestInput, KnowledgeNode, MemoryStats,
        NodeType, Rating, RecallInput, SearchMode,
    };

    // Legacy SQLite surface (behind `legacy-sqlite`, build/t5-legacy-isolation).
    #[cfg(feature = "legacy-sqlite")]
    pub use crate::{Result, Storage, StorageError};

    pub use crate::HybridSearcher;

    // Advanced features
    pub use crate::{
        ActivityTracker,
        ConnectionGraph,
        ConsolidationReport,
        // Sleep consolidation
        ConsolidationScheduler,
        CrossProjectLearner,
        EvaluationIntent,
        GateDecision,
        ImportanceTracker,
        IntentDetector,
        LabileCandidate,
        LabileState,
        MemoryChainBuilder,
        MemoryCompressor,
        MemoryDreamer,
        MemoryReplay,
        Modification,
        PredictedMemory,
        // Prediction Error Gating
        PredictionErrorGate,
        ReconsolidatedMemory,
        // Reconsolidation
        ReconsolidationManager,
        SpeculativeRetriever,
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

/// Wave-S UX: the latest published version of `vestige-mcp-server` on the
/// npm registry, or None on any failure (offline, timeout, parse). Used by
/// the server's start-time check-and-hint — never a self-update. Two-second
/// budget so a slow registry cannot delay anything.
#[cfg(feature = "cloud-sync")]
pub async fn latest_npm_version() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let resp: serde_json::Value = client
        .get("https://registry.npmjs.org/vestige-mcp-server/latest")
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    resp.get("version")?.as_str().map(String::from)
}
