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
            "scope":{"type":"string","default":"user"},
            "after":{"type":"string","description":"ID cursor from previous dream page. Cross-page pairs are outside this page's analysis."},
            "max_pairs":{"type":"integer","minimum":10,"maximum":124750,"default":1225,"description":"Upper bound for within-page memory pairs; reduces selected memory count."},
            "memory_count": {
                "type": "integer",
                "description": "Maximum memories in a scoped ID page (default: 50)",
                "default": 50, "minimum":5, "maximum":500
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
    #[derive(serde::Deserialize, Default)]
    struct Args {
        memory_count: Option<usize>,
        min_similarity: Option<f64>,
        max_pairs: Option<usize>,
        after: Option<String>,
        scope: Option<String>,
    }
    let parsed: Args = serde_json::from_value(args.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|e| e.to_string())?;
    let requested = parsed.memory_count.unwrap_or(50);
    let max_pairs = parsed.max_pairs.unwrap_or(1225);
    if !(5..=500).contains(&requested)
        || !(10..=124750).contains(&max_pairs)
        || parsed
            .min_similarity
            .is_some_and(|n| !n.is_finite() || !(0.0..=1.0).contains(&n))
    {
        return Err("invalid memory_count, max_pairs or min_similarity".into());
    }
    let mut memory_count = requested;
    while memory_count * (memory_count - 1) / 2 > max_pairs {
        memory_count -= 1;
    }
    let scope = parsed.scope.unwrap_or_else(|| "user".into());
    if scope.trim().is_empty() {
        return Err("scope must not be empty".into());
    }
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        // Recorded edges and integer FSRS state only. `min_similarity`, when
        // the caller set it, is a milli-unit floor on those edges — never a
        // content, keyword, or embedding score.
        let floor = strength_floor_milli(parsed.min_similarity);
        let storage = Arc::clone(storage);
        let after = parsed.after.clone();
        return tokio::task::spawn_blocking(move || {
            storage.dream_recorded_consolidation(
                &scope,
                after.as_deref(),
                memory_count,
                max_pairs,
                floor,
            )
        })
        .await
        .map_err(|e| format!("dream task failed: {e}"))?
        .map_err(|e| e.to_string());
    }
    let dream_started_at = Utc::now();
    let (mut all_nodes, has_more) = storage
        .maintenance_memory_page(memory_count, parsed.after.as_deref(), &scope)
        .map_err(|e| e.to_string())?;
    let next_cursor = all_nodes.last().map(|n| n.id.clone());
    all_nodes.retain(|n| n.suppression_count == 0 && n.is_currently_valid());
    let min_similarity = parsed.min_similarity;
    if all_nodes.len() < 5 {
        return Ok(serde_json::json!({
            "hasMore":has_more,"nextCursor":next_cursor,"scope":scope,"maxPairs":max_pairs,
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
            // w1b: vector embeddings removed; the dreamer's content-word
            // similarity path is the only connection scorer left.
            embedding: None,
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
    let page_ids: Vec<String> = all_nodes.iter().map(|node| node.id.clone()).collect();
    let tags_cleared = storage
        .clear_dream_page_tags(&page_ids, dream_started_at)
        .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "hasMore":has_more,"nextCursor":next_cursor,"scope":scope,"maxPairs":max_pairs,
        "status": "dreamed",
        "memoriesReplayed": dream_memories.len(),
        "selection":"scoped_id_page",
        "wakingTagsProcessed": tags_cleared,
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

/// Quantize an explicit similarity argument into a recorded-edge strength
/// floor. Omitted means every recorded edge is eligible. This is not a
/// matcher: nothing is scored against content, tags, or embeddings.
fn strength_floor_milli(min_similarity: Option<f64>) -> i64 {
    let Some(value) = min_similarity else {
        return 0;
    };
    let milli = (value * 1000.0).round();
    if !milli.is_finite() {
        return 0;
    }
    let clamped = milli.clamp(0.0, 1000.0);
    clamped as i64
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
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

#[cfg(test)]
mod strata_tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use vestige_core::IngestInput;

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    fn ingest(storage: &Storage, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.into(),
                ..IngestInput::default()
            })
            .unwrap()
            .id
    }

    fn link(storage: &Storage, source: &str, target: &str, strength: f64) {
        let now = Utc::now();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength,
                link_type: "derived_from".into(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();
    }

    fn strata_verify_bin() -> PathBuf {
        if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
            return PathBuf::from(dir).join("debug").join("strata-verify");
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/strata-verify")
    }

    fn assert_strata_verify(dir: &Path) {
        let bin = strata_verify_bin();
        assert!(
            bin.is_file(),
            "strata-verify binary missing at {}; build it before this test",
            bin.display()
        );
        let output = Command::new(&bin)
            .arg(dir)
            .output()
            .unwrap_or_else(|err| panic!("spawn strata-verify: {err}"));
        assert!(
            output.status.success(),
            "strata-verify failed: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("OK"), "{stdout}");
    }

    #[tokio::test]
    async fn strata_dream_completes_from_recorded_edges_and_fsrs_only() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        // Identical text must not become an edge. Only the recorded links move FSRS.
        let linked = ingest(storage.as_ref(), "alpha quartz ledger entry");
        let twin = ingest(storage.as_ref(), "alpha quartz ledger entry");
        let mid = ingest(storage.as_ref(), "boron carbide kiln log");
        let far = ingest(storage.as_ref(), "delta manifold pressure");
        let tail = ingest(storage.as_ref(), "epsilon unused isolator");
        link(storage.as_ref(), &linked, &mid, 1.0);
        link(storage.as_ref(), &mid, &far, 1.0);

        let edges_before = storage.get_all_connections().unwrap().len();
        let before = storage.get_node(&linked).unwrap().unwrap().reps;
        let twin_before = storage.get_node(&twin).unwrap().unwrap().reps;

        let value = execute(&storage, &cognitive(), None).await.unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["basis"], "recorded_edges+fsrs");
        assert_eq!(value["connectionsPersisted"], 0);
        assert_eq!(value["connectionsFound"], 0);
        assert_eq!(value["insights"], serde_json::json!([]));
        assert_eq!(value["edgesConsidered"], 2);
        assert_eq!(value["memoriesReviewed"], 3);
        assert_eq!(storage.get_all_connections().unwrap().len(), edges_before);

        let reviewed: HashSet<String> = value["reviews"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_string())
            .collect();
        assert!(reviewed.contains(&linked));
        assert!(reviewed.contains(&mid));
        assert!(reviewed.contains(&far));
        assert!(!reviewed.contains(&twin), "identical text is not an edge");
        assert!(
            !reviewed.contains(&tail),
            "an unlinked memory is not reviewed"
        );
        for row in value["reviews"].as_array().unwrap() {
            assert_eq!(row["rating"], 4);
            assert_eq!(row["reviewCount"], 2);
        }

        let after = storage.get_node(&linked).unwrap().unwrap().reps;
        assert!(after > before);
        assert_eq!(storage.get_node(&twin).unwrap().unwrap().reps, twin_before);
        drop(storage);

        assert_strata_verify(dir.path());
        let reopened = crate::strata_memory::open(dir.path()).unwrap();
        assert_eq!(reopened.get_node(&linked).unwrap().unwrap().reps, after);
        let again = execute(&reopened, &cognitive(), None).await.unwrap();
        assert_eq!(again["status"], "completed");
        assert_eq!(reopened.get_node(&linked).unwrap().unwrap().reps, after + 1);
        assert_eq!(reopened.get_node(&twin).unwrap().unwrap().reps, twin_before);
    }

    #[tokio::test]
    async fn strata_dream_below_five_does_not_review() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let id = ingest(storage.as_ref(), "only one");
        let value = execute(&storage, &cognitive(), None).await.unwrap();
        assert_eq!(value["status"], "insufficient_memories");
        assert_eq!(value["count"], 1);
        assert!(value["message"].as_str().unwrap().contains('5'));
        assert_eq!(storage.get_node(&id).unwrap().unwrap().reps, 1);
    }

    #[tokio::test]
    async fn strata_dream_strength_floor_ignores_weak_edges() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let mut ids = Vec::new();
        for i in 0..5 {
            ids.push(ingest(storage.as_ref(), &format!("floor fixture {i}")));
        }
        link(storage.as_ref(), &ids[0], &ids[1], 0.4);
        let before = storage.get_node(&ids[0]).unwrap().unwrap().reps;
        let value = execute(
            &storage,
            &cognitive(),
            Some(serde_json::json!({ "min_similarity": 0.5 })),
        )
        .await
        .unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["edgesConsidered"], 0);
        assert_eq!(value["memoriesReviewed"], 0);
        assert_eq!(storage.get_node(&ids[0]).unwrap().unwrap().reps, before);
    }

    #[tokio::test]
    async fn strata_dream_rejects_bad_bounds_and_cursors() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let err = execute(
            &storage,
            &cognitive(),
            Some(serde_json::json!({ "memory_count": 1 })),
        )
        .await
        .unwrap_err();
        assert!(err.contains("invalid"), "{err}");
        let err = execute(
            &storage,
            &cognitive(),
            Some(serde_json::json!({ "after": "not-an-id" })),
        )
        .await
        .unwrap_err();
        assert!(err.contains("after must be a memory id"), "{err}");
    }
}
