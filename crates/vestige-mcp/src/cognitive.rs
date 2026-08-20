//! CognitiveEngine — Stateful neuroscience modules that persist across tool calls.
//!
//! v1.5.0: Wires ALL unused vestige-core features into the MCP server.
//! Each module is initialized once at startup and shared via Arc<Mutex<>>
//! across all tool invocations.

use vestige_core::neuroscience::predictive_retrieval::PredictiveMemory;
use vestige_core::neuroscience::prospective_memory::{IntentionParser, ProspectiveMemory};
#[cfg(feature = "vector-search")]
use vestige_core::search::TemporalSearcher;
use vestige_core::{
    AccessibilityCalculator,
    // Neuroscience modules
    ActivationNetwork,
    ActivityTracker,
    AdaptiveEmbedder,
    ArousalSignal,
    AttentionSignal,
    CompetitionManager,
    ConsolidationScheduler,
    ContextMatcher,
    CrossProjectLearner,
    EmotionalMemory,
    HippocampalIndex,
    ImportanceSignals,
    // Advanced modules
    ImportanceTracker,
    IntentDetector,
    LinkType,
    MemoryChainBuilder,
    MemoryCompressor,
    MemoryDreamer,
    NoveltySignal,
    ReconsolidationManager,
    RewardSignal,
    SpeculativeRetriever,
    StateUpdateService,
    // Storage
    Storage,
    SynapticTaggingSystem,
};
#[cfg(feature = "vector-search")]
use vestige_core::{Reranker, RerankerConfig};

/// Stateful cognitive engine holding all neuroscience modules.
///
/// Lives on McpServer as `Arc<Mutex<CognitiveEngine>>` and is passed
/// to tools that need persistent cross-call state (search, ingest,
/// feedback, consolidation, new tools).
pub struct CognitiveEngine {
    // -- Neuroscience --
    pub activation_network: ActivationNetwork,
    pub synaptic_tagging: SynapticTaggingSystem,
    pub hippocampal_index: HippocampalIndex,
    pub context_matcher: ContextMatcher,
    pub accessibility_calc: AccessibilityCalculator,
    pub competition_mgr: CompetitionManager,
    pub state_service: StateUpdateService,
    pub importance_signals: ImportanceSignals,
    pub novelty_signal: NoveltySignal,
    pub arousal_signal: ArousalSignal,
    pub reward_signal: RewardSignal,
    pub attention_signal: AttentionSignal,
    pub emotional_memory: EmotionalMemory,
    pub predictive_memory: PredictiveMemory,
    pub prospective_memory: ProspectiveMemory,
    pub intention_parser: IntentionParser,

    // -- Advanced --
    pub importance_tracker: ImportanceTracker,
    pub reconsolidation: ReconsolidationManager,
    pub intent_detector: IntentDetector,
    pub activity_tracker: ActivityTracker,
    pub dreamer: MemoryDreamer,
    pub chain_builder: MemoryChainBuilder,
    pub compressor: MemoryCompressor,
    pub cross_project: CrossProjectLearner,
    pub adaptive_embedder: AdaptiveEmbedder,
    pub speculative_retriever: SpeculativeRetriever,
    pub consolidation_scheduler: ConsolidationScheduler,

    // -- Search --
    #[cfg(feature = "vector-search")]
    pub reranker: Reranker,
    #[cfg(feature = "vector-search")]
    pub temporal_searcher: TemporalSearcher,
}

impl Default for CognitiveEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveEngine {
    /// Load persisted source-of-truth state into in-memory cognitive modules.
    ///
    /// Hydrates `ActivationNetwork` plus the active synaptic-tag projection.
    /// Other modules (MemoryChainBuilder, HippocampalIndex) require full
    /// MemoryNode content and are deferred to a follow-up.
    pub fn hydrate(&mut self, storage: &Storage) {
        match storage.get_all_connections() {
            Ok(connections) => {
                for conn in &connections {
                    let link_type = match conn.link_type.as_str() {
                        "semantic" => LinkType::Semantic,
                        "temporal" => LinkType::Temporal,
                        "causal" => LinkType::Causal,
                        "spatial" => LinkType::Spatial,
                        "shared_concepts" | "complementary" => LinkType::Semantic,
                        _ => LinkType::Semantic,
                    };
                    self.activation_network.add_edge(
                        conn.source_id.clone(),
                        conn.target_id.clone(),
                        link_type,
                        conn.strength,
                    );
                }
                tracing::info!(
                    count = connections.len(),
                    "Hydrated cognitive modules from persisted connections"
                );
            }
            Err(e) => {
                tracing::warn!("Failed to hydrate cognitive modules: {}", e);
            }
        }

        match storage.load_active_synaptic_tags() {
            Ok(tags) => {
                let count = tags.len();
                for tag in tags {
                    self.synaptic_tagging.restore_tag(tag);
                }
                tracing::info!(count, "Hydrated active synaptic tags from durable storage");
            }
            Err(e) => {
                tracing::warn!("Failed to hydrate synaptic tags: {}", e);
            }
        }
    }

    /// Initialize all cognitive modules with default configurations.
    pub fn new() -> Self {
        Self {
            // Neuroscience
            activation_network: ActivationNetwork::new(),
            synaptic_tagging: SynapticTaggingSystem::new(),
            hippocampal_index: HippocampalIndex::new(),
            context_matcher: ContextMatcher::new(),
            accessibility_calc: AccessibilityCalculator::default(),
            competition_mgr: CompetitionManager::new(),
            state_service: StateUpdateService::new(),
            importance_signals: ImportanceSignals::new(),
            novelty_signal: NoveltySignal::new(),
            arousal_signal: ArousalSignal::new(),
            reward_signal: RewardSignal::new(),
            attention_signal: AttentionSignal::new(),
            emotional_memory: EmotionalMemory::new(),
            predictive_memory: PredictiveMemory::new(),
            prospective_memory: ProspectiveMemory::new(),
            intention_parser: IntentionParser::new(),

            // Advanced
            importance_tracker: ImportanceTracker::new(),
            reconsolidation: ReconsolidationManager::new(),
            intent_detector: IntentDetector::new(),
            activity_tracker: ActivityTracker::new(),
            dreamer: MemoryDreamer::new(),
            chain_builder: MemoryChainBuilder::new(),
            compressor: MemoryCompressor::new(),
            cross_project: CrossProjectLearner::new(),
            adaptive_embedder: AdaptiveEmbedder::new(),
            speculative_retriever: SpeculativeRetriever::new(),
            consolidation_scheduler: ConsolidationScheduler::new(),

            // Search
            #[cfg(feature = "vector-search")]
            reranker: Reranker::new(RerankerConfig::default()),
            #[cfg(feature = "vector-search")]
            temporal_searcher: TemporalSearcher::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;
    use vestige_core::{ConnectionRecord, IngestInput};

    fn create_test_storage() -> (Storage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    fn ingest_memory(storage: &Storage, content: &str) -> String {
        let result = storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["test".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        result.id
    }

    #[test]
    fn test_hydrate_empty_storage() {
        let (storage, _dir) = create_test_storage();
        let mut engine = CognitiveEngine::new();
        engine.hydrate(&storage);
        // Should succeed with 0 connections
        let assocs = engine.activation_network.get_associations("nonexistent");
        assert!(assocs.is_empty());
    }

    #[test]
    fn test_hydrate_loads_connections() {
        let (storage, _dir) = create_test_storage();

        // Create two memories so FK constraints pass
        let id1 = ingest_memory(&storage, "Memory about Rust programming");
        let id2 = ingest_memory(&storage, "Memory about Cargo build system");

        // Save a connection between them
        let now = Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: id1.clone(),
                target_id: id2.clone(),
                strength: 0.85,
                link_type: "semantic".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();

        // Hydrate engine
        let mut engine = CognitiveEngine::new();
        engine.hydrate(&storage);

        // Verify activation network has the connection
        let assocs = engine.activation_network.get_associations(&id1);
        assert!(
            !assocs.is_empty(),
            "Hydrated engine should have associations for {}",
            id1
        );
        assert!(
            assocs.iter().any(|a| a.memory_id == id2),
            "Should find connection to {}",
            id2
        );
    }

    #[test]
    fn test_hydrate_multiple_link_types() {
        let (storage, _dir) = create_test_storage();

        let id1 = ingest_memory(&storage, "Event A happened");
        let id2 = ingest_memory(&storage, "Event B followed");
        let id3 = ingest_memory(&storage, "Event C was caused by A");

        let now = Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: id1.clone(),
                target_id: id2.clone(),
                strength: 0.7,
                link_type: "temporal".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();
        storage
            .save_connection(&ConnectionRecord {
                source_id: id1.clone(),
                target_id: id3.clone(),
                strength: 0.9,
                link_type: "causal".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();

        let mut engine = CognitiveEngine::new();
        engine.hydrate(&storage);

        let assocs = engine.activation_network.get_associations(&id1);
        assert!(
            assocs.len() >= 2,
            "Should have at least 2 associations, got {}",
            assocs.len()
        );
    }

    #[test]
    fn test_hydrate_restores_active_synaptic_tags() {
        let (storage, _dir) = create_test_storage();
        let id = ingest_memory(&storage, "Decision waiting for a later outcome");
        storage
            .save_synaptic_tag(&vestige_core::SynapticTag::new(&id))
            .unwrap();

        let mut engine = CognitiveEngine::new();
        engine.hydrate(&storage);

        assert!(engine.synaptic_tagging.has_active_tag(&id));
    }
}
