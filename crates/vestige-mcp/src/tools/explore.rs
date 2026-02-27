//! Explore connections tool — Graph exploration, chain building, bridge discovery.
//! v2.0.0: Rewritten to use SQLite storage directly (matching graph.rs pattern).
//!
//! Previously relied on in-memory CognitiveEngine data structures that were
//! never populated, causing all actions to return empty results. Now queries
//! the same SQLite connection data that `memory_graph` uses successfully.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;
use vestige_core::Storage;

/// Maximum depth for chain building
const MAX_CHAIN_DEPTH: usize = 10;

/// Maximum states to explore in BFS
const MAX_STATES_TO_EXPLORE: usize = 1000;

/// Minimum connection strength to traverse
const MIN_CONNECTION_STRENGTH: f64 = 0.2;

/// Depth to pull subgraph neighborhoods
const SUBGRAPH_DEPTH: u32 = 3;

/// Max nodes per subgraph pull
const SUBGRAPH_MAX_NODES: usize = 200;

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

/// Adjacency list edge: (target_id, strength, link_type)
type AdjEdge = (String, f64, String);

/// Build a bidirectional adjacency list from ConnectionRecord edges.
fn build_adjacency(edges: &[vestige_core::ConnectionRecord]) -> HashMap<String, Vec<AdjEdge>> {
    let mut adj: HashMap<String, Vec<AdjEdge>> = HashMap::new();
    for e in edges {
        adj.entry(e.source_id.clone())
            .or_default()
            .push((e.target_id.clone(), e.strength, e.link_type.clone()));
        adj.entry(e.target_id.clone())
            .or_default()
            .push((e.source_id.clone(), e.strength, e.link_type.clone()));
    }
    adj
}

/// State for BFS priority queue (best-first by score).
#[derive(Debug, Clone)]
struct SearchState {
    node_id: String,
    path: Vec<String>,
    /// (from, to, strength, link_type) for each hop
    hops: Vec<(String, String, f64, String)>,
    score: f64,
    depth: usize,
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Find best paths from `from_id` to `to_id` using best-first search on the adjacency list.
fn bfs_find_paths(
    adj: &HashMap<String, Vec<AdjEdge>>,
    from_id: &str,
    to_id: &str,
    max_paths: usize,
) -> Vec<SearchState> {
    let mut found: Vec<SearchState> = Vec::new();
    let mut queue = BinaryHeap::new();
    let mut explored: usize = 0;

    queue.push(SearchState {
        node_id: from_id.to_string(),
        path: vec![from_id.to_string()],
        hops: vec![],
        score: 1.0,
        depth: 0,
    });

    while let Some(state) = queue.pop() {
        explored += 1;
        if explored > MAX_STATES_TO_EXPLORE {
            break;
        }

        // Reached target
        if state.node_id == to_id {
            found.push(state);
            if found.len() >= max_paths {
                break;
            }
            continue;
        }

        if state.depth >= MAX_CHAIN_DEPTH {
            continue;
        }

        if let Some(neighbors) = adj.get(&state.node_id) {
            for (target, strength, link_type) in neighbors {
                if *strength < MIN_CONNECTION_STRENGTH {
                    continue;
                }
                // Avoid cycles
                if state.path.contains(target) {
                    continue;
                }

                let mut new_path = state.path.clone();
                new_path.push(target.clone());

                let mut new_hops = state.hops.clone();
                new_hops.push((
                    state.node_id.clone(),
                    target.clone(),
                    *strength,
                    link_type.clone(),
                ));

                let new_score = state.score * strength * 0.9;

                queue.push(SearchState {
                    node_id: target.clone(),
                    path: new_path,
                    hops: new_hops,
                    score: new_score,
                    depth: state.depth + 1,
                });
            }
        }
    }

    // Sort by score descending
    found.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    found
}

pub async fn execute(
    storage: &Arc<Storage>,
    args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let args = args.ok_or("Missing arguments")?;
    let action = args.get("action").and_then(|v| v.as_str()).ok_or("Missing 'action'")?;
    let from = args.get("from").and_then(|v| v.as_str()).ok_or("Missing 'from'")?;
    let to = args.get("to").and_then(|v| v.as_str());
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    match action {
        "associations" => {
            // Pull depth-1 neighborhood from SQLite
            let (_nodes, edges) = storage
                .get_memory_subgraph(from, 1, limit.max(50))
                .map_err(|e| format!("Failed to get subgraph: {}", e))?;

            // Filter edges connected to `from`
            let mut associations: Vec<serde_json::Value> = Vec::new();
            for e in &edges {
                let (neighbor_id, direction) = if e.source_id == from {
                    (&e.target_id, "outgoing")
                } else if e.target_id == from {
                    (&e.source_id, "incoming")
                } else {
                    continue;
                };

                associations.push(serde_json::json!({
                    "memory_id": neighbor_id,
                    "strength": e.strength,
                    "link_type": &e.link_type,
                    "direction": direction,
                    "source": "sqlite",
                }));
            }

            // Sort by strength descending, then truncate
            associations.sort_by(|a, b| {
                let sa = a["strength"].as_f64().unwrap_or(0.0);
                let sb = b["strength"].as_f64().unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
            });
            associations.truncate(limit);

            Ok(serde_json::json!({
                "action": "associations",
                "from": from,
                "associations": associations,
                "count": associations.len(),
            }))
        }

        "chain" => {
            let to_id = to.ok_or("'to' is required for chain action")?;

            // Pull neighborhoods around both endpoints and merge
            let (nodes_from, edges_from) = storage
                .get_memory_subgraph(from, SUBGRAPH_DEPTH, SUBGRAPH_MAX_NODES)
                .map_err(|e| format!("Failed to get subgraph for 'from': {}", e))?;

            let mut all_edges = edges_from;
            let mut all_node_ids: HashSet<String> = nodes_from.iter().map(|n| n.id.clone()).collect();

            // If target isn't in the from-neighborhood, also pull its neighborhood
            if !all_node_ids.contains(to_id) {
                let (_nodes_to, edges_to) = storage
                    .get_memory_subgraph(to_id, SUBGRAPH_DEPTH, SUBGRAPH_MAX_NODES)
                    .map_err(|e| format!("Failed to get subgraph for 'to': {}", e))?;

                // Merge edges (dedup by source+target)
                let existing: HashSet<(String, String)> = all_edges
                    .iter()
                    .map(|e| (e.source_id.clone(), e.target_id.clone()))
                    .collect();
                for e in edges_to {
                    if !existing.contains(&(e.source_id.clone(), e.target_id.clone())) {
                        all_edges.push(e);
                    }
                }
                // Update node set
                for n in &_nodes_to {
                    all_node_ids.insert(n.id.clone());
                }
            }

            let adj = build_adjacency(&all_edges);
            let paths = bfs_find_paths(&adj, from, to_id, 5);

            if let Some(best) = paths.first() {
                // Build steps from the best path's hops
                let steps: Vec<serde_json::Value> = best.hops.iter().map(|(f, t, s, lt)| {
                    serde_json::json!({
                        "from": f,
                        "to": t,
                        "connection_strength": s,
                        "link_type": lt,
                    })
                }).collect();

                // Confidence = geometric mean of hop strengths
                let confidence = if best.hops.is_empty() {
                    0.0
                } else {
                    let product: f64 = best.hops.iter().map(|(_, _, s, _)| s).product();
                    product.powf(1.0 / best.hops.len() as f64)
                };

                Ok(serde_json::json!({
                    "action": "chain",
                    "from": from,
                    "to": to_id,
                    "path": best.path,
                    "steps": steps,
                    "confidence": (confidence * 1000.0).round() / 1000.0,
                    "total_hops": best.hops.len(),
                    "paths_found": paths.len(),
                }))
            } else {
                Ok(serde_json::json!({
                    "action": "chain",
                    "from": from,
                    "to": to_id,
                    "path": [],
                    "steps": [],
                    "message": "No chain found between these memories",
                    "paths_found": 0,
                }))
            }
        }

        "bridges" => {
            let to_id = to.ok_or("'to' is required for bridges action")?;

            // Reuse chain logic — find multiple paths, extract intermediates
            let (nodes_from, edges_from) = storage
                .get_memory_subgraph(from, SUBGRAPH_DEPTH, SUBGRAPH_MAX_NODES)
                .map_err(|e| format!("Failed to get subgraph for 'from': {}", e))?;

            let mut all_edges = edges_from;
            let all_node_ids: HashSet<String> = nodes_from.iter().map(|n| n.id.clone()).collect();

            if !all_node_ids.contains(to_id) {
                let (_nodes_to, edges_to) = storage
                    .get_memory_subgraph(to_id, SUBGRAPH_DEPTH, SUBGRAPH_MAX_NODES)
                    .map_err(|e| format!("Failed to get subgraph for 'to': {}", e))?;

                let existing: HashSet<(String, String)> = all_edges
                    .iter()
                    .map(|e| (e.source_id.clone(), e.target_id.clone()))
                    .collect();
                for e in edges_to {
                    if !existing.contains(&(e.source_id.clone(), e.target_id.clone())) {
                        all_edges.push(e);
                    }
                }
            }

            let adj = build_adjacency(&all_edges);
            let paths = bfs_find_paths(&adj, from, to_id, 10);

            // Count frequency of intermediate nodes across all paths
            let mut bridge_counts: HashMap<String, usize> = HashMap::new();
            for path in &paths {
                if path.path.len() > 2 {
                    // Skip first (from) and last (to) — intermediates only
                    for node_id in &path.path[1..path.path.len() - 1] {
                        *bridge_counts.entry(node_id.clone()).or_insert(0) += 1;
                    }
                }
            }

            // Sort by frequency descending
            let mut bridge_list: Vec<_> = bridge_counts.into_iter().collect();
            bridge_list.sort_by(|a, b| b.1.cmp(&a.1));
            bridge_list.truncate(limit);

            let bridges: Vec<serde_json::Value> = bridge_list.iter().map(|(id, count)| {
                serde_json::json!({
                    "memory_id": id,
                    "frequency": count,
                    "paths_total": paths.len(),
                })
            }).collect();

            Ok(serde_json::json!({
                "action": "bridges",
                "from": from,
                "to": to_id,
                "bridges": bridges,
                "count": bridges.len(),
            }))
        }

        _ => Err(format!("Unknown action: '{}'. Expected: chain, associations, bridges", action)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    /// Helper: ingest a memory and return its ID.
    fn ingest_memory(storage: &Storage, content: &str) -> String {
        let node = storage.ingest(vestige_core::IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec!["test".to_string()],
            valid_from: None,
            valid_until: None,
        }).unwrap();
        node.id
    }

    /// Helper: save a connection between two memories.
    fn connect(storage: &Storage, from: &str, to: &str, strength: f64, link_type: &str) {
        use chrono::Utc;
        storage.save_connection(&vestige_core::ConnectionRecord {
            source_id: from.to_string(),
            target_id: to.to_string(),
            strength,
            link_type: link_type.to_string(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 1,
        }).unwrap();
    }

    // ---- Schema tests ----

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

    // ---- Validation tests ----

    #[tokio::test]
    async fn test_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_missing_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "from": "some-id" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'action'"));
    }

    #[tokio::test]
    async fn test_missing_from_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "associations" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'from'"));
    }

    #[tokio::test]
    async fn test_unknown_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "invalid", "from": "id1" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown action"));
    }

    #[tokio::test]
    async fn test_chain_missing_to_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "chain", "from": "id1" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'to' is required"));
    }

    #[tokio::test]
    async fn test_bridges_missing_to_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "bridges", "from": "id1" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'to' is required"));
    }

    // ---- Empty results tests (backwards compat) ----

    #[tokio::test]
    async fn test_associations_succeeds_empty() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "associations",
            "from": "00000000-0000-0000-0000-000000000000"
        });
        let result = execute(&storage, Some(args)).await;
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
        let result = execute(&storage, Some(args)).await;
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
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "bridges");
        assert_eq!(value["count"], 0);
    }

    // ---- Integration tests with actual connections ----

    #[tokio::test]
    async fn test_associations_returns_neighbors() {
        let (storage, _dir) = test_storage().await;

        let id_a = ingest_memory(&storage, "Memory A about databases");
        let id_b = ingest_memory(&storage, "Memory B about indexing");
        let id_c = ingest_memory(&storage, "Memory C about queries");

        connect(&storage, &id_a, &id_b, 0.8, "semantic");
        connect(&storage, &id_a, &id_c, 0.6, "temporal");

        let args = serde_json::json!({
            "action": "associations",
            "from": &id_a,
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "associations");
        let count = value["count"].as_u64().unwrap();
        assert!(count >= 2, "Expected at least 2 associations, got {}", count);

        let assocs = value["associations"].as_array().unwrap();
        let neighbor_ids: Vec<&str> = assocs.iter()
            .map(|a| a["memory_id"].as_str().unwrap())
            .collect();
        assert!(neighbor_ids.contains(&id_b.as_str()));
        assert!(neighbor_ids.contains(&id_c.as_str()));
    }

    #[tokio::test]
    async fn test_chain_finds_path() {
        let (storage, _dir) = test_storage().await;

        let id_a = ingest_memory(&storage, "Start node");
        let id_b = ingest_memory(&storage, "Middle node");
        let id_c = ingest_memory(&storage, "End node");

        connect(&storage, &id_a, &id_b, 0.9, "causal");
        connect(&storage, &id_b, &id_c, 0.85, "causal");

        let args = serde_json::json!({
            "action": "chain",
            "from": &id_a,
            "to": &id_c,
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "chain");
        let hops = value["total_hops"].as_u64().unwrap();
        assert!(hops >= 2, "Expected at least 2 hops, got {}", hops);
        assert!(value["confidence"].as_f64().unwrap() > 0.0);

        let path = value["path"].as_array().unwrap();
        assert_eq!(path.first().unwrap().as_str().unwrap(), id_a);
        assert_eq!(path.last().unwrap().as_str().unwrap(), id_c);
    }

    #[tokio::test]
    async fn test_bridges_finds_intermediate_nodes() {
        let (storage, _dir) = test_storage().await;

        let id_a = ingest_memory(&storage, "Node A");
        let id_bridge = ingest_memory(&storage, "Bridge node");
        let id_c = ingest_memory(&storage, "Node C");

        connect(&storage, &id_a, &id_bridge, 0.9, "semantic");
        connect(&storage, &id_bridge, &id_c, 0.8, "semantic");

        let args = serde_json::json!({
            "action": "bridges",
            "from": &id_a,
            "to": &id_c,
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "bridges");
        let count = value["count"].as_u64().unwrap();
        assert!(count >= 1, "Expected at least 1 bridge, got {}", count);

        let bridges = value["bridges"].as_array().unwrap();
        let bridge_ids: Vec<&str> = bridges.iter()
            .map(|b| b["memory_id"].as_str().unwrap())
            .collect();
        assert!(bridge_ids.contains(&id_bridge.as_str()));
    }

    #[tokio::test]
    async fn test_associations_respects_limit() {
        let (storage, _dir) = test_storage().await;

        let id_center = ingest_memory(&storage, "Center node");
        for i in 0..10 {
            let neighbor = ingest_memory(&storage, &format!("Neighbor {}", i));
            connect(&storage, &id_center, &neighbor, 0.5 + (i as f64 * 0.03), "semantic");
        }

        let args = serde_json::json!({
            "action": "associations",
            "from": &id_center,
            "limit": 3,
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let count = value["count"].as_u64().unwrap();
        assert!(count <= 3, "Expected at most 3, got {}", count);
    }

    // ---- Unit tests for internal helpers ----

    #[test]
    fn test_build_adjacency_bidirectional() {
        use chrono::Utc;
        let edges = vec![
            vestige_core::ConnectionRecord {
                source_id: "a".to_string(),
                target_id: "b".to_string(),
                strength: 0.8,
                link_type: "semantic".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            },
        ];
        let adj = build_adjacency(&edges);
        assert!(adj.contains_key("a"));
        assert!(adj.contains_key("b"));
        assert_eq!(adj["a"].len(), 1);
        assert_eq!(adj["b"].len(), 1);
        assert_eq!(adj["a"][0].0, "b");
        assert_eq!(adj["b"][0].0, "a");
    }

    #[test]
    fn test_bfs_finds_direct_path() {
        use chrono::Utc;
        let edges = vec![
            vestige_core::ConnectionRecord {
                source_id: "a".to_string(),
                target_id: "b".to_string(),
                strength: 0.9,
                link_type: "test".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            },
        ];
        let adj = build_adjacency(&edges);
        let paths = bfs_find_paths(&adj, "a", "b", 5);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].path, vec!["a", "b"]);
    }

    #[test]
    fn test_bfs_finds_multi_hop_path() {
        use chrono::Utc;
        let edges = vec![
            vestige_core::ConnectionRecord {
                source_id: "a".to_string(),
                target_id: "b".to_string(),
                strength: 0.9,
                link_type: "test".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            },
            vestige_core::ConnectionRecord {
                source_id: "b".to_string(),
                target_id: "c".to_string(),
                strength: 0.8,
                link_type: "test".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            },
        ];
        let adj = build_adjacency(&edges);
        let paths = bfs_find_paths(&adj, "a", "c", 5);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_bfs_skips_weak_connections() {
        use chrono::Utc;
        let edges = vec![
            vestige_core::ConnectionRecord {
                source_id: "a".to_string(),
                target_id: "b".to_string(),
                strength: 0.1, // Below MIN_CONNECTION_STRENGTH
                link_type: "test".to_string(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 1,
            },
        ];
        let adj = build_adjacency(&edges);
        let paths = bfs_find_paths(&adj, "a", "b", 5);
        assert!(paths.is_empty(), "Should not traverse weak connections");
    }

    #[test]
    fn test_bfs_no_path() {
        let adj: HashMap<String, Vec<AdjEdge>> = HashMap::new();
        let paths = bfs_find_paths(&adj, "a", "b", 5);
        assert!(paths.is_empty());
    }
}
