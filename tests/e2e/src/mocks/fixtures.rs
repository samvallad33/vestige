//! Test Data Factory
//!
//! Provides utilities for generating realistic test data:
//! - Memory nodes with various properties
//! - Batch generation for stress testing
//! - Pre-built scenarios for common test cases

use chrono::{DateTime, Duration, Utc};
use vestige_core::{KnowledgeNode, Rating, Storage};

/// Helper to create IngestInput (works around non_exhaustive)
#[allow(clippy::too_many_arguments)]
fn make_ingest_input(
    content: String,
    node_type: String,
    tags: Vec<String>,
    sentiment_score: f64,
    sentiment_magnitude: f64,
    source: Option<String>,
    valid_from: Option<DateTime<Utc>>,
    valid_until: Option<DateTime<Utc>>,
) -> vestige_core::IngestInput {
    vestige_core::IngestInput {
        content,
        node_type,
        tags,
        sentiment_score,
        sentiment_magnitude,
        source,
        valid_from,
        valid_until,
        validity_inferred: false,
        source_envelope: None,
    }
}

/// Factory for creating test data
///
/// Generates realistic test data with configurable properties.
/// Designed for creating comprehensive test scenarios.
///
/// # Example
///
/// ```rust,ignore
/// let mut storage = Storage::new(Some(path))?;
///
/// // Create a single memory
/// let node = TestDataFactory::create_memory(&mut storage, "test content");
///
/// // Create a batch
/// let nodes = TestDataFactory::create_batch(&mut storage, 100);
///
/// // Create a specific scenario
/// let scenario = TestDataFactory::create_decay_scenario(&mut storage);
/// ```
pub struct TestDataFactory;

/// Configuration for batch memory generation
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of memories to create
    pub count: usize,
    /// Node type to use (None = random)
    pub node_type: Option<String>,
    /// Base content prefix
    pub content_prefix: String,
    /// Tags to apply
    pub tags: Vec<String>,
    /// Whether to add sentiment
    pub with_sentiment: bool,
    /// Whether to add temporal validity
    pub with_temporal: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            count: 10,
            node_type: None,
            content_prefix: "Test memory".to_string(),
            tags: vec![],
            with_sentiment: false,
            with_temporal: false,
        }
    }
}

/// Scenario containing related test data
#[derive(Debug)]
pub struct TestScenario {
    /// IDs of created nodes
    pub node_ids: Vec<String>,
    /// Description of the scenario
    pub description: String,
    /// Metadata for test assertions
    pub metadata: std::collections::HashMap<String, String>,
}

impl TestDataFactory {
    // ========================================================================
    // SINGLE MEMORY CREATION
    // ========================================================================

    /// Create a simple memory with content
    pub fn create_memory(storage: &mut Storage, content: &str) -> Option<KnowledgeNode> {
        let input = make_ingest_input(
            content.to_string(),
            "fact".to_string(),
            vec![],
            0.0,
            0.0,
            None,
            None,
            None,
        );
        storage.ingest(input).ok()
    }

    /// Create a memory with full configuration
    pub fn create_memory_full(
        storage: &mut Storage,
        content: &str,
        node_type: &str,
        source: Option<&str>,
        tags: Vec<&str>,
        sentiment_score: f64,
        sentiment_magnitude: f64,
    ) -> Option<KnowledgeNode> {
        let input = make_ingest_input(
            content.to_string(),
            node_type.to_string(),
            tags.iter().map(|s| s.to_string()).collect(),
            sentiment_score,
            sentiment_magnitude,
            source.map(String::from),
            None,
            None,
        );
        storage.ingest(input).ok()
    }

    /// Create a memory with temporal validity
    pub fn create_temporal_memory(
        storage: &mut Storage,
        content: &str,
        valid_from: Option<DateTime<Utc>>,
        valid_until: Option<DateTime<Utc>>,
    ) -> Option<KnowledgeNode> {
        let input = make_ingest_input(
            content.to_string(),
            "fact".to_string(),
            vec![],
            0.0,
            0.0,
            None,
            valid_from,
            valid_until,
        );
        storage.ingest(input).ok()
    }

    /// Create an emotional memory
    pub fn create_emotional_memory(
        storage: &mut Storage,
        content: &str,
        sentiment: f64,
        magnitude: f64,
    ) -> Option<KnowledgeNode> {
        let input = make_ingest_input(
            content.to_string(),
            "event".to_string(),
            vec![],
            sentiment,
            magnitude,
            None,
            None,
            None,
        );
        storage.ingest(input).ok()
    }

    // ========================================================================
    // BATCH CREATION
    // ========================================================================

    /// Create a batch of memories
    pub fn create_batch(storage: &mut Storage, count: usize) -> Vec<String> {
        Self::create_batch_with_config(
            storage,
            BatchConfig {
                count,
                ..Default::default()
            },
        )
    }

    /// Create a batch with custom configuration
    pub fn create_batch_with_config(storage: &mut Storage, config: BatchConfig) -> Vec<String> {
        let node_types = ["fact", "concept", "procedure", "event", "code"];
        let mut ids = Vec::with_capacity(config.count);

        for i in 0..config.count {
            let node_type = config
                .node_type
                .clone()
                .unwrap_or_else(|| node_types[i % node_types.len()].to_string());

            let sentiment_score = if config.with_sentiment {
                ((i as f64) / (config.count as f64) * 2.0) - 1.0
            } else {
                0.0
            };

            let sentiment_magnitude = if config.with_sentiment {
                (i as f64) / (config.count as f64)
            } else {
                0.0
            };

            let (valid_from, valid_until) = if config.with_temporal {
                let now = Utc::now();
                if i % 3 == 0 {
                    (
                        Some(now - Duration::days(30)),
                        Some(now + Duration::days(30)),
                    )
                } else if i % 3 == 1 {
                    (
                        Some(now - Duration::days(60)),
                        Some(now - Duration::days(30)),
                    )
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let input = make_ingest_input(
                format!("{} {}", config.content_prefix, i),
                node_type,
                config.tags.clone(),
                sentiment_score,
                sentiment_magnitude,
                None,
                valid_from,
                valid_until,
            );

            if let Ok(node) = storage.ingest(input) {
                ids.push(node.id);
            }
        }

        ids
    }

    // ========================================================================
    // SCENARIO CREATION
    // ========================================================================

    /// Create a scenario for testing memory decay
    pub fn create_decay_scenario(storage: &mut Storage) -> TestScenario {
        let mut ids = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        // High stability memory (should decay slowly)
        let high_stab = Self::create_memory_full(
            storage,
            "Well-learned fact about photosynthesis",
            "fact",
            Some("biology textbook"),
            vec!["biology", "science"],
            0.3,
            0.5,
        );
        if let Some(node) = high_stab {
            metadata.insert("high_stability".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Low stability memory (should decay quickly)
        let low_stab = Self::create_memory(storage, "Random fact I just learned");
        if let Some(node) = low_stab {
            metadata.insert("low_stability".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Emotional memory (decay should be affected by sentiment)
        let emotional = Self::create_emotional_memory(storage, "Important life event", 0.9, 0.95);
        if let Some(node) = emotional {
            metadata.insert("emotional".to_string(), node.id.clone());
            ids.push(node.id);
        }

        TestScenario {
            node_ids: ids,
            description: "Decay testing scenario with varied stability".to_string(),
            metadata,
        }
    }

    /// Create a scenario for testing review scheduling
    pub fn create_scheduling_scenario(storage: &mut Storage) -> TestScenario {
        let mut ids = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        // New card (never reviewed)
        let new_card = Self::create_memory(storage, "Brand new memory");
        if let Some(node) = new_card {
            metadata.insert("new".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Learning card (few reviews)
        if let Some(node) = Self::create_memory(storage, "Learning memory") {
            let _ = storage.mark_reviewed(&node.id, Rating::Good);
            metadata.insert("learning".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Review card (many reviews)
        if let Some(node) = Self::create_memory(storage, "Well-reviewed memory") {
            for _ in 0..5 {
                let _ = storage.mark_reviewed(&node.id, Rating::Good);
            }
            metadata.insert("review".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Relearning card (had lapses)
        if let Some(node) = Self::create_memory(storage, "Struggling memory") {
            let _ = storage.mark_reviewed(&node.id, Rating::Good);
            let _ = storage.mark_reviewed(&node.id, Rating::Again);
            metadata.insert("relearning".to_string(), node.id.clone());
            ids.push(node.id);
        }

        TestScenario {
            node_ids: ids,
            description: "Scheduling scenario with cards in different learning states".to_string(),
            metadata,
        }
    }

    /// Create a scenario for testing search
    pub fn create_search_scenario(storage: &mut Storage) -> TestScenario {
        let mut ids = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        // Programming memories
        for content in [
            "Rust programming language uses ownership for memory safety",
            "Python is great for data science and machine learning",
            "JavaScript runs in web browsers and Node.js",
        ] {
            if let Some(node) = Self::create_memory_full(
                storage,
                content,
                "fact",
                Some("programming docs"),
                vec!["programming", "code"],
                0.0,
                0.0,
            ) {
                ids.push(node.id);
            }
        }
        metadata.insert("programming_count".to_string(), "3".to_string());

        // Science memories
        for content in [
            "Mitochondria is the powerhouse of the cell",
            "DNA contains genetic information",
            "Gravity is the force of attraction between masses",
        ] {
            if let Some(node) = Self::create_memory_full(
                storage,
                content,
                "fact",
                Some("science textbook"),
                vec!["science"],
                0.0,
                0.0,
            ) {
                ids.push(node.id);
            }
        }
        metadata.insert("science_count".to_string(), "3".to_string());

        // Recipe memories
        for content in [
            "To make pasta, boil water and add salt",
            "Chocolate cake requires cocoa powder and eggs",
        ] {
            if let Some(node) = Self::create_memory_full(
                storage,
                content,
                "procedure",
                Some("cookbook"),
                vec!["cooking", "recipes"],
                0.0,
                0.0,
            ) {
                ids.push(node.id);
            }
        }
        metadata.insert("recipe_count".to_string(), "2".to_string());

        TestScenario {
            node_ids: ids,
            description: "Search scenario with categorized content".to_string(),
            metadata,
        }
    }

    /// Create a scenario for testing temporal queries
    pub fn create_temporal_scenario(storage: &mut Storage) -> TestScenario {
        let now = Utc::now();
        let mut ids = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        // Currently valid
        if let Some(node) = Self::create_temporal_memory(
            storage,
            "Currently valid memory",
            Some(now - Duration::days(10)),
            Some(now + Duration::days(10)),
        ) {
            metadata.insert("current".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Expired
        if let Some(node) = Self::create_temporal_memory(
            storage,
            "Expired memory",
            Some(now - Duration::days(60)),
            Some(now - Duration::days(30)),
        ) {
            metadata.insert("expired".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // Future
        if let Some(node) = Self::create_temporal_memory(
            storage,
            "Future memory",
            Some(now + Duration::days(30)),
            Some(now + Duration::days(60)),
        ) {
            metadata.insert("future".to_string(), node.id.clone());
            ids.push(node.id);
        }

        // No bounds (always valid)
        if let Some(node) = Self::create_temporal_memory(storage, "Always valid memory", None, None)
        {
            metadata.insert("always_valid".to_string(), node.id.clone());
            ids.push(node.id);
        }

        TestScenario {
            node_ids: ids,
            description: "Temporal scenario with different validity periods".to_string(),
            metadata,
        }
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Get a random node type
    pub fn random_node_type(seed: usize) -> &'static str {
        const TYPES: [&str; 9] = [
            "fact",
            "concept",
            "procedure",
            "event",
            "relationship",
            "quote",
            "code",
            "question",
            "insight",
        ];
        TYPES[seed % TYPES.len()]
    }

    /// Generate lorem ipsum-like content
    pub fn lorem_content(words: usize, seed: usize) -> String {
        const WORDS: [&str; 20] = [
            "the",
            "memory",
            "learning",
            "knowledge",
            "algorithm",
            "data",
            "system",
            "process",
            "function",
            "method",
            "class",
            "object",
            "variable",
            "constant",
            "type",
            "structure",
            "pattern",
            "design",
            "architecture",
            "code",
        ];

        (0..words)
            .map(|i| WORDS[(seed + i * 7) % WORDS.len()])
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate tags
    pub fn generate_tags(count: usize, seed: usize) -> Vec<String> {
        const TAGS: [&str; 10] = [
            "important",
            "review",
            "todo",
            "concept",
            "fact",
            "code",
            "note",
            "idea",
            "question",
            "reference",
        ];

        (0..count)
            .map(|i| TAGS[(seed + i) % TAGS.len()].to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_storage() -> Storage {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        Storage::new(Some(db_path)).unwrap()
    }

    #[test]
    fn test_create_memory() {
        let mut storage = create_test_storage();
        let node = TestDataFactory::create_memory(&mut storage, "test content");

        assert!(node.is_some());
        assert_eq!(node.unwrap().content, "test content");
    }

    #[test]
    fn test_create_batch() {
        let mut storage = create_test_storage();
        let ids = TestDataFactory::create_batch(&mut storage, 10);

        assert_eq!(ids.len(), 10);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 10);
    }

    #[test]
    fn test_create_decay_scenario() {
        let mut storage = create_test_storage();
        let scenario = TestDataFactory::create_decay_scenario(&mut storage);

        assert!(!scenario.node_ids.is_empty());
        assert!(scenario.metadata.contains_key("high_stability"));
        assert!(scenario.metadata.contains_key("low_stability"));
        assert!(scenario.metadata.contains_key("emotional"));
    }

    #[test]
    fn test_create_scheduling_scenario() {
        let mut storage = create_test_storage();
        let scenario = TestDataFactory::create_scheduling_scenario(&mut storage);

        assert!(!scenario.node_ids.is_empty());
        assert!(scenario.metadata.contains_key("new"));
        assert!(scenario.metadata.contains_key("learning"));
        assert!(scenario.metadata.contains_key("review"));
    }

    #[test]
    fn test_lorem_content() {
        let content = TestDataFactory::lorem_content(10, 42);
        let words: Vec<_> = content.split_whitespace().collect();

        assert_eq!(words.len(), 10);
    }

    #[test]
    fn test_generate_tags() {
        let tags = TestDataFactory::generate_tags(5, 0);

        assert_eq!(tags.len(), 5);
        assert!(tags.iter().all(|t| !t.is_empty()));
    }
}
