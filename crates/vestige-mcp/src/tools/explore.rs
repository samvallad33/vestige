//! Explore connections tool — Graph exploration, chain building, bridge discovery.
//! v1.5.0: Wires MemoryChainBuilder + ActivationNetwork + HippocampalIndex.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::Storage;
use vestige_core::advanced::{Connection, ConnectionType, MemoryChainBuilder, MemoryNode};

pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["chain", "associations", "bridges"],
                "description": "Type of exploration: 'chain' builds reasoning path, 'associations' finds related memories, 'bridges' finds connecting memories"
            },
            "from": {
                "type": "string",
                "description": "Source memory ID"
            },
            "to": {
                "type": "string",
                "description": "Target memory ID (required for 'chain' and 'bridges')"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 10)",
                "default": 10
            }
        },
        "required": ["action", "from"]
    })
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let args = args.ok_or("Missing arguments")?;
    let action = args
        .get("action")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'action'")?;
    let from = args
        .get("from")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'from'")?;
    let to = args.get("to").and_then(|v| v.as_str());
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let cog = cognitive.lock().await;

    match action {
        "chain" => {
            let to_id = to.ok_or("'to' is required for chain action")?;
            let chain_result = cog.chain_builder.build_chain(from, to_id);
            let from_owned = from.to_string();
            let to_owned = to_id.to_string();
            drop(cog); // release lock before potential storage fallback

            let chain_opt = if chain_result.is_some() {
                chain_result
            } else {
                // Storage fallback: build temporary chain from persisted connections
                build_chain_from_storage(storage, &from_owned, &to_owned)
            };

            match chain_opt {
                Some(chain) => Ok(serde_json::json!({
                    "action": "chain",
                    "from": from_owned,
                    "to": to_owned,
                    "steps": chain.steps.iter().map(|s| serde_json::json!({
                        "memory_id": s.memory_id,
                        "memory_preview": s.memory_preview,
                        "connection_type": format!("{:?}", s.connection_type),
                        "connection_strength": s.connection_strength,
                        "reasoning": s.reasoning,
                    })).collect::<Vec<_>>(),
                    "confidence": chain.confidence,
                    "total_hops": chain.total_hops,
                })),
                None => Ok(serde_json::json!({
                    "action": "chain",
                    "from": from_owned,
                    "to": to_owned,
                    "steps": [],
                    "message": "No chain found between these memories"
                })),
            }
        }
        "associations" => {
            let activation_assocs = cog.activation_network.get_associations(from);
            let hippocampal_assocs = cog
                .hippocampal_index
                .get_associations(from, 2)
                .unwrap_or_default();
            let from_owned = from.to_string();
            drop(cog); // release lock consistently (matches chain/bridges pattern)

            let mut all_associations: Vec<serde_json::Value> = Vec::new();

            for assoc in activation_assocs.iter().take(limit) {
                all_associations.push(serde_json::json!({
                    "memory_id": assoc.memory_id,
                    "strength": assoc.association_strength,
                    "link_type": format!("{:?}", assoc.link_type),
                    "source": "spreading_activation",
                }));
            }
            for m in hippocampal_assocs.iter().take(limit) {
                all_associations.push(serde_json::json!({
                    "memory_id": m.index.memory_id,
                    "semantic_score": m.semantic_score,
                    "text_score": m.text_score,
                    "source": "hippocampal_index",
                }));
            }

            all_associations.truncate(limit);

            // Fallback: if in-memory modules are empty, query storage directly
            if all_associations.is_empty()
                && let Ok(connections) = storage.get_connections_for_memory(&from_owned)
            {
                for conn in connections.iter().take(limit) {
                    let other_id = if conn.source_id == from_owned {
                        &conn.target_id
                    } else {
                        &conn.source_id
                    };
                    all_associations.push(serde_json::json!({
                        "memory_id": other_id,
                        "strength": conn.strength,
                        "link_type": conn.link_type,
                        "source": "persistent_graph",
                    }));
                }
            }

            Ok(serde_json::json!({
                "action": "associations",
                "from": from_owned,
                "associations": all_associations,
                "count": all_associations.len(),
            }))
        }
        "bridges" => {
            let to_id = to.ok_or("'to' is required for bridges action")?;
            let bridges = cog.chain_builder.find_bridge_memories(from, to_id);
            let from_owned = from.to_string();
            let to_owned = to_id.to_string();
            drop(cog); // release lock before potential storage fallback

            let final_bridges = if !bridges.is_empty() {
                bridges
            } else {
                // Storage fallback: build temporary graph and find bridges
                let temp_builder = build_temp_chain_builder(storage, &from_owned, &to_owned);
                temp_builder.find_bridge_memories(&from_owned, &to_owned)
            };

            let limited: Vec<_> = final_bridges.iter().take(limit).collect();
            Ok(serde_json::json!({
                "action": "bridges",
                "from": from_owned,
                "to": to_owned,
                "bridges": limited,
                "count": limited.len(),
            }))
        }
        _ => Err(format!(
            "Unknown action: '{}'. Expected: chain, associations, bridges",
            action
        )),
    }
}

/// Build a temporary MemoryChainBuilder from persisted connections for fallback queries.
fn build_temp_chain_builder(
    storage: &Arc<Storage>,
    from_id: &str,
    to_id: &str,
) -> MemoryChainBuilder {
    let mut builder = MemoryChainBuilder::new();

    // Load connections involving either endpoint
    let mut all_conns = Vec::new();
    if let Ok(conns) = storage.get_connections_for_memory(from_id) {
        all_conns.extend(conns);
    }
    if let Ok(conns) = storage.get_connections_for_memory(to_id) {
        all_conns.extend(conns);
    }

    // Deduplicate edges and load referenced memory nodes
    let mut seen_edges = std::collections::HashSet::new();
    all_conns.retain(|c| seen_edges.insert((c.source_id.clone(), c.target_id.clone())));

    let mut seen_ids = std::collections::HashSet::new();
    for conn in &all_conns {
        for id in [&conn.source_id, &conn.target_id] {
            if seen_ids.insert(id.clone())
                && let Ok(Some(node)) = storage.get_node(id)
            {
                builder.add_memory(MemoryNode {
                    id: node.id.clone(),
                    content_preview: node.content.chars().take(100).collect(),
                    tags: node.tags.clone(),
                    connections: vec![],
                });
            }
        }
    }

    // Add edges
    for conn in &all_conns {
        builder.add_connection(Connection {
            from_id: conn.source_id.clone(),
            to_id: conn.target_id.clone(),
            connection_type: link_type_to_connection_type(&conn.link_type),
            strength: conn.strength,
            created_at: conn.created_at,
        });
    }

    builder
}

/// Build a chain from storage when in-memory chain_builder is empty.
fn build_chain_from_storage(
    storage: &Arc<Storage>,
    from_id: &str,
    to_id: &str,
) -> Option<vestige_core::advanced::ReasoningChain> {
    let builder = build_temp_chain_builder(storage, from_id, to_id);
    builder.build_chain(from_id, to_id)
}

/// Convert storage link_type string to ConnectionType enum.
fn link_type_to_connection_type(link_type: &str) -> ConnectionType {
    match link_type {
        "temporal" => ConnectionType::TemporalProximity,
        "causal" => ConnectionType::Causal,
        "part_of" => ConnectionType::PartOf,
        _ => ConnectionType::SemanticSimilarity,
    }
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

    #[test]
    fn test_schema_has_required_fields() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["action"].is_object());
        assert!(s["properties"]["from"].is_object());
        assert!(s["properties"]["to"].is_object());
        assert!(s["properties"]["limit"].is_object());
        let required = s["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("action")));
        assert!(required.contains(&serde_json::json!("from")));
    }

    #[test]
    fn test_schema_action_enum() {
        let s = schema();
        let action_enum = s["properties"]["action"]["enum"].as_array().unwrap();
        assert!(action_enum.contains(&serde_json::json!("chain")));
        assert!(action_enum.contains(&serde_json::json!("associations")));
        assert!(action_enum.contains(&serde_json::json!("bridges")));
    }

    #[tokio::test]
    async fn test_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_missing_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "from": "some-id" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'action'"));
    }

    #[tokio::test]
    async fn test_missing_from_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "associations" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'from'"));
    }

    #[tokio::test]
    async fn test_unknown_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "invalid", "from": "id1" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown action"));
    }

    #[tokio::test]
    async fn test_chain_missing_to_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "chain", "from": "id1" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'to' is required"));
    }

    #[tokio::test]
    async fn test_bridges_missing_to_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "bridges", "from": "id1" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'to' is required"));
    }

    #[tokio::test]
    async fn test_associations_succeeds_empty() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "associations",
            "from": "00000000-0000-0000-0000-000000000000"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "associations");
        assert!(value["associations"].is_array());
        assert_eq!(value["count"], 0);
    }

    #[tokio::test]
    async fn test_chain_no_path_found() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "chain",
            "from": "00000000-0000-0000-0000-000000000001",
            "to": "00000000-0000-0000-0000-000000000002"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "chain");
        assert_eq!(value["steps"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_bridges_no_results() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "bridges",
            "from": "00000000-0000-0000-0000-000000000001",
            "to": "00000000-0000-0000-0000-000000000002"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "bridges");
        assert_eq!(value["count"], 0);
    }

    #[tokio::test]
    async fn test_associations_with_limit() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "associations",
            "from": "00000000-0000-0000-0000-000000000000",
            "limit": 5
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_associations_storage_fallback() {
        let (storage, _dir) = test_storage().await;

        // Create two memories and a direct connection in storage
        let id1 = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory about Rust".to_string(),
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
            .unwrap()
            .id;

        let id2 = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory about Cargo".to_string(),
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
            .unwrap()
            .id;

        // Save connection directly to storage (bypassing cognitive engine)
        let now = chrono::Utc::now();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: id1.clone(),
                target_id: id2.clone(),
                strength: 0.9,
                link_type: "semantic".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            })
            .unwrap();

        // Execute with empty cognitive engine — should fall back to storage
        let cognitive = test_cognitive();
        let args = serde_json::json!({
            "action": "associations",
            "from": id1,
        });
        let result = execute(&storage, &cognitive, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let associations = value["associations"].as_array().unwrap();
        assert!(
            !associations.is_empty(),
            "Should find associations via storage fallback"
        );
        assert_eq!(associations[0]["source"], "persistent_graph");
        assert_eq!(associations[0]["memory_id"], id2);
    }

    #[tokio::test]
    async fn test_chain_storage_fallback() {
        let (storage, _dir) = test_storage().await;

        // Create 3 memories: A -> B -> C
        let make = |content: &str| vestige_core::IngestInput {
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
        };
        let id_a = storage.ingest(make("Memory A about databases")).unwrap().id;
        let id_b = storage.ingest(make("Memory B about indexes")).unwrap().id;
        let id_c = storage
            .ingest(make("Memory C about performance"))
            .unwrap()
            .id;

        // Save connections A->B and B->C to storage
        let now = chrono::Utc::now();
        for (src, tgt) in [(&id_a, &id_b), (&id_b, &id_c)] {
            storage
                .save_connection(&vestige_core::ConnectionRecord {
                    source_id: src.clone(),
                    target_id: tgt.clone(),
                    strength: 0.9,
                    link_type: "semantic".to_string(),
                    created_at: now,
                    last_activated: now,
                    activation_count: 1,
                })
                .unwrap();
        }

        // Execute chain with empty cognitive engine — should fall back to storage
        let args = serde_json::json!({ "action": "chain", "from": id_a, "to": id_c });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "chain");
        let steps = value["steps"].as_array().unwrap();
        assert!(
            !steps.is_empty(),
            "Chain should find path A->B->C via storage fallback"
        );
    }

    #[tokio::test]
    async fn test_bridges_storage_fallback() {
        let (storage, _dir) = test_storage().await;

        // Create 3 memories: A -> B -> C (B is the bridge)
        let make = |content: &str| vestige_core::IngestInput {
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
        };
        let id_a = storage.ingest(make("Bridge test memory A")).unwrap().id;
        let id_b = storage.ingest(make("Bridge test memory B")).unwrap().id;
        let id_c = storage.ingest(make("Bridge test memory C")).unwrap().id;

        let now = chrono::Utc::now();
        for (src, tgt) in [(&id_a, &id_b), (&id_b, &id_c)] {
            storage
                .save_connection(&vestige_core::ConnectionRecord {
                    source_id: src.clone(),
                    target_id: tgt.clone(),
                    strength: 0.9,
                    link_type: "semantic".to_string(),
                    created_at: now,
                    last_activated: now,
                    activation_count: 1,
                })
                .unwrap();
        }

        // Execute bridges with empty cognitive engine
        let args = serde_json::json!({ "action": "bridges", "from": id_a, "to": id_c });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "bridges");
        let bridges = value["bridges"].as_array().unwrap();
        assert!(
            !bridges.is_empty(),
            "Should find B as bridge between A and C via storage fallback"
        );
    }
}
