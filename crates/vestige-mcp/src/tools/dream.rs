//! Dream tool — Explicit dream trigger that returns insights.
//! v1.5.0: Wires MemoryDreamer into an MCP tool.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use chrono::Utc;
use vestige_core::{DreamHistoryRecord, InsightRecord, LinkType, Storage};

pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memory_count": {
                "type": "integer",
                "description": "Number of recent memories to dream about (default: 50)",
                "default": 50
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity for connection discovery (0.0-1.0, default: 0.5)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5
            }
        }
    })
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let memory_count = args
        .as_ref()
        .and_then(|a| a.get("memory_count"))
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(500) as usize; // Cap at 500 to prevent O(N^2) hang
    let min_similarity = args
        .as_ref()
        .and_then(|a| a.get("min_similarity"))
        .and_then(|v| v.as_f64())
        .map(|v| v.clamp(0.0, 1.0));

    // v1.9.0: Waking SWR tagging — preferential replay of tagged memories (70/30 split)
    let tagged_nodes = storage
        .get_waking_tagged_memories(memory_count as i32)
        .unwrap_or_default();
    let tagged_count = tagged_nodes.len();

    // Calculate how many tagged vs random to include
    let tagged_target = (memory_count * 7 / 10).min(tagged_count); // 70% tagged
    let _random_target = memory_count.saturating_sub(tagged_target); // 30% random (used for logging)

    // Build the dream memory set: tagged memories first, then fill with random
    let tagged_ids: std::collections::HashSet<String> = tagged_nodes
        .iter()
        .take(tagged_target)
        .map(|n| n.id.clone())
        .collect();

    let random_nodes = storage
        .get_all_nodes(memory_count as i32, 0)
        .map_err(|e| format!("Failed to load memories: {}", e))?;

    let mut all_nodes: Vec<_> = tagged_nodes.into_iter().take(tagged_target).collect();
    for node in random_nodes {
        if !tagged_ids.contains(&node.id) && all_nodes.len() < memory_count {
            all_nodes.push(node);
        }
    }
    // If still under capacity (e.g., all memories are tagged), fill from remaining tagged
    if all_nodes.len() < memory_count {
        let used_ids: std::collections::HashSet<String> =
            all_nodes.iter().map(|n| n.id.clone()).collect();
        let remaining_tagged = storage
            .get_waking_tagged_memories(memory_count as i32)
            .unwrap_or_default();
        for node in remaining_tagged {
            if !used_ids.contains(&node.id) && all_nodes.len() < memory_count {
                all_nodes.push(node);
            }
        }
    }

    if all_nodes.len() < 5 {
        return Ok(serde_json::json!({
            "status": "insufficient_memories",
            "message": format!("Need at least 5 memories to dream. Current count: {}", all_nodes.len()),
            "count": all_nodes.len()
        }));
    }

    let dream_memories: Vec<vestige_core::DreamMemory> = all_nodes
        .iter()
        .map(|n| vestige_core::DreamMemory {
            id: n.id.clone(),
            content: n.content.clone(),
            embedding: storage.get_node_embedding(&n.id).ok().flatten(),
            tags: n.tags.clone(),
            created_at: n.created_at,
            access_count: n.reps as u32,
        })
        .collect();

    // Run the dream OFF the engine lock. Snapshot the dreamer under a short
    // lock (its history/insights/connections are Arc-shared, so the engine's
    // dreamer still records this run), then do the synchronous O(n²) pairwise
    // scan on the blocking pool. Holding `CognitiveEngine` across that scan
    // starved every other tool that takes the same mutex (explore, predict,
    // session_context, memory_unified, autopilot, ...) for the whole dream,
    // which presented as a server hang under multi-agent use.
    let dreamer = {
        let cog = cognitive.lock().await;
        cog.dreamer.clone()
    };
    let (dream_result, new_connections, insights, dream_memories) =
        tokio::task::spawn_blocking(move || {
            let (dream_result, new_connections) = if let Some(min_similarity) = min_similarity {
                let config = vestige_core::DreamConfig {
                    min_similarity,
                    ..vestige_core::DreamConfig::default()
                };
                dreamer.dream_with_config_and_connections_blocking(&dream_memories, config)
            } else {
                dreamer.dream_with_connections_blocking(&dream_memories)
            };
            let insights = dreamer.synthesize_insights(&dream_memories);
            (dream_result, new_connections, insights, dream_memories)
        })
        .await
        .map_err(|e| format!("Dream scan task failed: {e}"))?;

    // v2.1.0: Persist dream insights to database (Bug #4 fix)
    let mut insights_persisted = 0u64;
    for insight in &insights {
        let record = InsightRecord {
            id: insight.id.clone(),
            insight: insight.insight.clone(),
            source_memories: insight.source_memories.clone(),
            confidence: insight.confidence,
            novelty_score: insight.novelty_score,
            insight_type: format!("{:?}", insight.insight_type),
            generated_at: insight.generated_at,
            tags: insight.tags.clone(),
            feedback: None,
            applied_count: 0,
        };
        if storage.save_insight(&record).is_ok() {
            insights_persisted += 1;
        }
    }

    let mut connections_persisted = 0u64;
    {
        let now = Utc::now();
        for conn in &new_connections {
            let link_type = match conn.connection_type {
                vestige_core::DiscoveredConnectionType::Semantic => "semantic",
                vestige_core::DiscoveredConnectionType::SharedConcept => "shared_concepts",
                vestige_core::DiscoveredConnectionType::Temporal => "temporal",
                vestige_core::DiscoveredConnectionType::Complementary => "complementary",
                vestige_core::DiscoveredConnectionType::CausalChain => "causal",
            };
            let record = vestige_core::ConnectionRecord {
                source_id: conn.from_id.clone(),
                target_id: conn.to_id.clone(),
                strength: conn.similarity,
                link_type: link_type.to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            };
            match storage.save_connection(&record) {
                Ok(_) => connections_persisted += 1,
                Err(e) => {
                    tracing::warn!(
                        source = %conn.from_id,
                        target = %conn.to_id,
                        link_type = %link_type,
                        "Failed to persist dream connection: {}",
                        e
                    );
                }
            }
        }
        if connections_persisted > 0 {
            tracing::info!(
                connections_persisted = connections_persisted,
                "Dream: persisted {} connections to database",
                connections_persisted
            );
        }
    }

    // Hydrate live cognitive engine with newly persisted connections
    if connections_persisted > 0 {
        let mut cog = cognitive.lock().await;
        for conn in &new_connections {
            let link_type_enum = match conn.connection_type {
                vestige_core::DiscoveredConnectionType::Semantic => LinkType::Semantic,
                vestige_core::DiscoveredConnectionType::SharedConcept => LinkType::Semantic,
                vestige_core::DiscoveredConnectionType::Temporal => LinkType::Temporal,
                vestige_core::DiscoveredConnectionType::Complementary => LinkType::Semantic,
                vestige_core::DiscoveredConnectionType::CausalChain => LinkType::Causal,
            };
            cog.activation_network.add_edge(
                conn.from_id.clone(),
                conn.to_id.clone(),
                link_type_enum,
                conn.similarity,
            );
        }
    }

    // Persist dream history (non-fatal on failure — dream still happened)
    {
        let record = DreamHistoryRecord {
            dreamed_at: Utc::now(),
            duration_ms: dream_result.duration_ms as i64,
            memories_replayed: dream_memories.len() as i32,
            connections_found: dream_result.new_connections_found as i32,
            insights_generated: dream_result.insights_generated.len() as i32,
            memories_strengthened: dream_result.memories_strengthened as i32,
            memories_compressed: dream_result.memories_compressed as i32,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        };
        if let Err(e) = storage.save_dream_history(&record) {
            tracing::warn!("Failed to persist dream history: {}", e);
        }
    }

    // v1.9.0: Clear waking tags after dream processes them
    let tags_cleared = storage.clear_waking_tags().unwrap_or(0);

    Ok(serde_json::json!({
        "status": "dreamed",
        "memoriesReplayed": dream_memories.len(),
        "wakingTagsProcessed": tagged_target,
        "wakingTagsCleared": tags_cleared,
        "insights": insights.iter().map(|i| serde_json::json!({
            "insight_type": format!("{:?}", i.insight_type),
            "insight": i.insight,
            "source_memories": i.source_memories,
            "confidence": i.confidence,
            "novelty_score": i.novelty_score,
        })).collect::<Vec<_>>(),
        "connectionsPersisted": connections_persisted,
        "insightsPersisted": insights_persisted,
        "stats": {
            "new_connections_found": dream_result.new_connections_found,
            "connections_persisted": connections_persisted,
            "insights_persisted": insights_persisted,
            "memories_strengthened": dream_result.memories_strengthened,
            "memories_compressed": dream_result.memories_compressed,
            "insights_generated": dream_result.insights_generated.len(),
            "duration_ms": dream_result.duration_ms,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_n_memories(storage: &Arc<Storage>, n: usize) {
        for i in 0..n {
            storage
                .ingest(vestige_core::IngestInput {
                    content: format!("Dream test memory number {}", i),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["dream-test".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }
    }

    #[test]
    fn test_schema_has_properties() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["memory_count"].is_object());
        assert_eq!(s["properties"]["memory_count"]["default"], 50);
        assert!(s["properties"]["min_similarity"].is_object());
        assert_eq!(s["properties"]["min_similarity"]["minimum"], 0.0);
        assert_eq!(s["properties"]["min_similarity"]["maximum"], 1.0);
    }

    #[tokio::test]
    async fn test_dream_insufficient_memories() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 3).await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["count"], 3);
    }

    #[tokio::test]
    async fn test_dream_empty_database() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["count"], 0);
    }

    #[tokio::test]
    async fn test_dream_with_enough_memories() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 10).await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "dreamed");
        assert!(value["memoriesReplayed"].as_u64().unwrap() >= 5);
        assert!(value["insights"].is_array());
        assert!(value["stats"].is_object());
    }

    #[tokio::test]
    async fn test_dream_custom_memory_count() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 10).await;
        let args = serde_json::json!({ "memory_count": 7 });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "dreamed");
        assert!(value["memoriesReplayed"].as_u64().unwrap() <= 7);
    }

    #[tokio::test]
    async fn test_dream_with_exactly_5_memories() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 5).await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "dreamed");
    }

    #[tokio::test]
    async fn test_dream_stats_fields_present() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 6).await;
        let result = execute(&storage, &test_cognitive(), None).await;
        let value = result.unwrap();
        assert!(value["stats"]["new_connections_found"].is_number());
        assert!(value["stats"]["memories_strengthened"].is_number());
        assert!(value["stats"]["memories_compressed"].is_number());
        assert!(value["stats"]["insights_generated"].is_number());
        assert!(value["stats"]["duration_ms"].is_number());
    }

    #[tokio::test]
    async fn test_dream_persists_to_database() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 10).await;

        // Before dream: no dream history
        {
            assert!(storage.get_last_dream().unwrap().is_none());
        }

        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["status"], "dreamed");

        // After dream: dream history should exist
        {
            let last = storage.get_last_dream().unwrap();
            assert!(
                last.is_some(),
                "Dream should have been persisted to database"
            );
        }
    }

    #[tokio::test]
    async fn test_dream_connections_round_trip() {
        // Verify dream → persist → query round-trip
        let (storage, _dir) = test_storage().await;

        // Create enough diverse memories to trigger connection discovery
        for i in 0..15 {
            storage
                .ingest(vestige_core::IngestInput {
                    content: format!(
                        "Memory {} about topic {}: detailed content for connection discovery",
                        i,
                        if i % 3 == 0 {
                            "rust"
                        } else if i % 3 == 1 {
                            "cargo"
                        } else {
                            "testing"
                        }
                    ),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["dream-roundtrip".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }

        let cognitive = test_cognitive();
        let result = execute(&storage, &cognitive, None).await.unwrap();
        assert_eq!(result["status"], "dreamed");

        let persisted = result["connectionsPersisted"].as_u64().unwrap_or(0);
        if persisted > 0 {
            // Verify connections are queryable from storage
            let all_conns = storage.get_all_connections().unwrap();
            assert!(
                !all_conns.is_empty(),
                "Persisted connections should be queryable"
            );

            // Verify connection IDs reference valid memories
            let all_nodes = storage.get_all_nodes(100, 0).unwrap();
            let valid_ids: std::collections::HashSet<String> =
                all_nodes.iter().map(|n| n.id.clone()).collect();
            for conn in &all_conns {
                assert!(
                    valid_ids.contains(&conn.source_id),
                    "Connection source_id {} should reference a valid memory",
                    conn.source_id
                );
                assert!(
                    valid_ids.contains(&conn.target_id),
                    "Connection target_id {} should reference a valid memory",
                    conn.target_id
                );
            }

            // Verify live cognitive engine was hydrated
            let cog = cognitive.lock().await;
            let first_conn = &all_conns[0];
            let assocs = cog
                .activation_network
                .get_associations(&first_conn.source_id);
            assert!(
                !assocs.is_empty(),
                "Live cognitive engine should have been hydrated with dream connections"
            );
        }
    }

    /// Directly test save_connection with real memory IDs — isolates the persistence layer.
    #[tokio::test]
    async fn test_save_connection_with_dream_ids() {
        let (storage, _dir) = test_storage().await;

        // Ingest memories and collect their IDs
        let mut ids = Vec::new();
        for i in 0..5 {
            let result = storage
                .ingest(vestige_core::IngestInput {
                    content: format!("Save connection test memory {}", i),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec!["save-conn-test".to_string()],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
            ids.push(result.id);
        }

        // Simulate what dream does: save connections between real memory IDs
        let now = chrono::Utc::now();
        let mut saved = 0u32;
        let mut errors = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let record = vestige_core::ConnectionRecord {
                    source_id: ids[i].clone(),
                    target_id: ids[j].clone(),
                    strength: 0.75,
                    link_type: "semantic".to_string(),
                    created_at: now,
                    last_activated: now,
                    activation_count: 1,
                };
                match storage.save_connection(&record) {
                    Ok(_) => saved += 1,
                    Err(e) => errors.push(format!("{} -> {}: {}", ids[i], ids[j], e)),
                }
            }
        }

        assert!(
            errors.is_empty(),
            "save_connection failed for {} of {} connections:\n{}",
            errors.len(),
            saved + errors.len() as u32,
            errors.join("\n")
        );
        assert!(saved > 0, "Should have saved at least one connection");

        // Verify they're queryable
        let all = storage.get_all_connections().unwrap();
        assert_eq!(all.len(), saved as usize);

        // Verify per-memory query
        let conns = storage.get_connections_for_memory(&ids[0]).unwrap();
        assert!(
            !conns.is_empty(),
            "get_connections_for_memory should return connections for {}",
            ids[0]
        );
    }

    /// Test that dream actually discovers connections and they persist.
    /// Unlike test_dream_connections_round_trip, this ASSERTS on the dream
    /// discovering connections (not just conditionally checking).
    #[tokio::test]
    async fn test_dream_discovers_and_persists_connections() {
        let (storage, _dir) = test_storage().await;

        // Ingest memories with known high-similarity content (shared tags + similar text)
        let topics = [
            (
                "Rust borrow checker prevents data races at compile time",
                vec!["rust", "safety"],
            ),
            (
                "Rust ownership model ensures memory safety without GC",
                vec!["rust", "safety"],
            ),
            (
                "Cargo is the Rust package manager and build system",
                vec!["rust", "cargo"],
            ),
            (
                "Cargo.toml defines dependencies for Rust projects",
                vec!["rust", "cargo"],
            ),
            (
                "Unit tests in Rust use #[test] attribute",
                vec!["rust", "testing"],
            ),
            (
                "Integration tests in Rust live in the tests/ directory",
                vec!["rust", "testing"],
            ),
            (
                "Clippy is a Rust linter that catches common mistakes",
                vec!["rust", "tooling"],
            ),
            (
                "Rustfmt formats Rust code according to style guidelines",
                vec!["rust", "tooling"],
            ),
        ];

        for (content, tags) in &topics {
            storage
                .ingest(vestige_core::IngestInput {
                    content: content.to_string(),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }

        let cognitive = test_cognitive();
        let result = execute(&storage, &cognitive, None).await.unwrap();
        assert_eq!(result["status"], "dreamed");

        let found = result["stats"]["new_connections_found"]
            .as_u64()
            .unwrap_or(0);
        let persisted = result["connectionsPersisted"].as_u64().unwrap_or(0);

        // Dream should discover connections between these related memories
        // (they share tags and have similar content)
        assert!(
            found > 0,
            "Dream should discover connections between related memories (found: {})",
            found
        );

        // Key assertion: if connections were found, they should persist
        assert_eq!(
            persisted, found,
            "All {} discovered connections should persist, but only {} did. \
             Check tracing output for save_connection errors.",
            found, persisted
        );

        // Verify round-trip through storage
        let stored = storage.get_all_connections().unwrap();
        assert_eq!(
            stored.len(),
            persisted as usize,
            "Storage should contain exactly {} connections",
            persisted
        );
    }

    #[tokio::test]
    async fn test_dream_persists_dense_connection_set_above_legacy_buffer_cap() {
        let (storage, _dir) = test_storage().await;
        ingest_n_memories(&storage, 50).await;

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "memory_count": 50,
                "min_similarity": 0.1
            })),
        )
        .await
        .unwrap();

        assert_eq!(result["status"], "dreamed");
        let found = result["stats"]["new_connections_found"]
            .as_u64()
            .unwrap_or(0);
        let persisted = result["connectionsPersisted"].as_u64().unwrap_or(0);

        assert!(
            found > 1_000,
            "test setup should discover more than the legacy 1,000 connection cap"
        );
        assert_eq!(
            persisted, found,
            "dense dreams should persist every connection discovered in the run"
        );
        assert_eq!(
            storage.get_all_connections().unwrap().len(),
            persisted as usize
        );
    }

    #[tokio::test]
    async fn test_dream_persists_insights() {
        let (storage, _dir) = test_storage().await;

        // Create diverse tagged memories to encourage insight generation
        let topics = [
            (
                "Rust borrow checker prevents data races",
                vec!["rust", "safety"],
            ),
            (
                "Rust ownership model ensures memory safety",
                vec!["rust", "safety"],
            ),
            (
                "Cargo manages Rust project dependencies",
                vec!["rust", "cargo"],
            ),
            (
                "Cargo.toml defines project configuration",
                vec!["rust", "cargo"],
            ),
            (
                "Unit tests use the #[test] attribute",
                vec!["rust", "testing"],
            ),
            (
                "Integration tests live in the tests directory",
                vec!["rust", "testing"],
            ),
            (
                "Clippy catches common Rust mistakes",
                vec!["rust", "tooling"],
            ),
            (
                "Rustfmt automatically formats code",
                vec!["rust", "tooling"],
            ),
        ];
        for (content, tags) in &topics {
            storage
                .ingest(vestige_core::IngestInput {
                    content: content.to_string(),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }

        let result = execute(&storage, &test_cognitive(), None).await.unwrap();
        assert_eq!(result["status"], "dreamed");

        let response_insights = result["insights"].as_array().unwrap();
        let persisted_count = result["insightsPersisted"].as_u64().unwrap_or(0);

        // If insights were generated, they should be persisted
        if !response_insights.is_empty() {
            assert!(
                persisted_count > 0,
                "Generated insights should be persisted to database"
            );
            let stored = storage.get_insights(100).unwrap();
            assert_eq!(
                stored.len(),
                persisted_count as usize,
                "All {} persisted insights should be retrievable",
                persisted_count
            );
            // Verify insight fields
            for insight in &stored {
                assert!(!insight.id.is_empty(), "Insight ID should not be empty");
                assert!(
                    !insight.insight.is_empty(),
                    "Insight text should not be empty"
                );
                assert!(insight.confidence >= 0.0 && insight.confidence <= 1.0);
                assert!(insight.novelty_score >= 0.0);
                assert!(
                    insight.feedback.is_none(),
                    "Fresh insight should have no feedback"
                );
                assert_eq!(insight.applied_count, 0);
            }
        }
    }
}
