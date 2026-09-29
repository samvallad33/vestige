//! memory_graph tool — Subgraph export with force-directed layout for visualization.
//! v1.9.0: Computes Fruchterman-Reingold layout server-side.

use std::sync::Arc;
use vestige_core::Storage;

pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "center_id": {
                "type": "string",
                "description": "Memory ID to center the graph on. Required if no query."
            },
            "query": {
                "type": "string",
                "description": "Search query to find center node. Used if center_id not provided."
            },
            "depth": {
                "type": "integer",
                "description": "How many hops from center to include (1-3, default: 2)",
                "default": 2,
                "minimum": 1,
                "maximum": 3
            },
            "max_nodes": {
                "type": "integer",
                "description": "Maximum number of nodes to include (default: 50)",
                "default": 50,
                "maximum": 200
            }
        }
    })
}

/// Simple Fruchterman-Reingold force-directed layout
fn fruchterman_reingold(
    node_count: usize,
    edges: &[(usize, usize, f64)],
    width: f64,
    height: f64,
    iterations: usize,
) -> Vec<(f64, f64)> {
    if node_count == 0 {
        return Vec::new();
    }
    if node_count == 1 {
        return vec![(width / 2.0, height / 2.0)];
    }

    let area = width * height;
    let k = (area / node_count as f64).sqrt();

    // Initialize positions in a circle
    let mut positions: Vec<(f64, f64)> = (0..node_count)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / node_count as f64;
            (
                width / 2.0 + (width / 3.0) * angle.cos(),
                height / 2.0 + (height / 3.0) * angle.sin(),
            )
        })
        .collect();

    let mut temperature = width / 10.0;
    let cooling = temperature / iterations as f64;

    for _ in 0..iterations {
        let mut displacements = vec![(0.0f64, 0.0f64); node_count];

        // Repulsive forces between all pairs
        for i in 0..node_count {
            for j in (i + 1)..node_count {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);
                let force = k * k / dist;
                let fx = dx / dist * force;
                let fy = dy / dist * force;
                displacements[i].0 += fx;
                displacements[i].1 += fy;
                displacements[j].0 -= fx;
                displacements[j].1 -= fy;
            }
        }

        // Attractive forces along edges
        for &(u, v, weight) in edges {
            let dx = positions[u].0 - positions[v].0;
            let dy = positions[u].1 - positions[v].1;
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);
            let force = dist * dist / k * weight;
            let fx = dx / dist * force;
            let fy = dy / dist * force;
            displacements[u].0 -= fx;
            displacements[u].1 -= fy;
            displacements[v].0 += fx;
            displacements[v].1 += fy;
        }

        // Apply displacements with temperature limiting
        for i in 0..node_count {
            let dx = displacements[i].0;
            let dy = displacements[i].1;
            let dist = (dx * dx + dy * dy).sqrt().max(0.01);
            let capped = dist.min(temperature);
            positions[i].0 += dx / dist * capped;
            positions[i].1 += dy / dist * capped;

            // Clamp to bounds
            positions[i].0 = positions[i].0.clamp(10.0, width - 10.0);
            positions[i].1 = positions[i].1.clamp(10.0, height - 10.0);
        }

        temperature -= cooling;
        if temperature < 0.1 {
            break;
        }
    }

    positions
}

pub async fn execute(
    storage: &Arc<Storage>,
    args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let depth = args
        .as_ref()
        .and_then(|a| a.get("depth"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2)
        .min(3) as u32;

    let max_nodes = args
        .as_ref()
        .and_then(|a| a.get("max_nodes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .min(200) as usize;

    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());

    // Determine center node
    let center_id = if let Some(id) = args
        .as_ref()
        .and_then(|a| a.get("center_id"))
        .and_then(|v| v.as_str())
    {
        id.to_string()
    } else if let Some(query) = args
        .as_ref()
        .and_then(|a| a.get("query"))
        .and_then(|v| v.as_str())
    {
        if strata {
            // Query would be keyword/FTS search. Strata has no such op.
            return Err(
                "similarity_disabled: memory_graph query is keyword search, not a recorded edge; pass center_id"
                    .to_string(),
            );
        }
        // Search for center node
        let results = storage
            .search(query, 1)
            .map_err(|e| format!("Search failed: {}", e))?;
        results
            .first()
            .map(|n| n.id.clone())
            .ok_or_else(|| "No memories found matching query".to_string())?
    } else {
        // Default: use the most recent memory
        let recent = storage
            .get_all_nodes(1, 0)
            .map_err(|e| format!("Failed to get recent node: {}", e))?;
        recent
            .first()
            .map(|n| n.id.clone())
            .ok_or_else(|| "No memories in database".to_string())?
    };

    // Recorded edges on Strata. The SQLite path keeps its own subgraph.
    let (nodes, edges) = if strata {
        recorded_subgraph(storage.as_ref(), &center_id, depth, max_nodes)?
    } else {
        storage
            .get_memory_subgraph(&center_id, depth, max_nodes)
            .map_err(|e| format!("Failed to get subgraph: {}", e))?
    };

    if nodes.is_empty() || !nodes.iter().any(|n| n.id == center_id) {
        return Err(format!(
            "Memory '{}' not found or has no accessible data",
            center_id
        ));
    }

    // Build index map for FR layout
    let id_to_idx: std::collections::HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    let layout_edges: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter_map(|e| {
            let u = id_to_idx.get(e.source_id.as_str())?;
            let v = id_to_idx.get(e.target_id.as_str())?;
            Some((*u, *v, e.strength))
        })
        .collect();

    // Compute force-directed layout
    let positions = fruchterman_reingold(nodes.len(), &layout_edges, 800.0, 600.0, 50);

    // Build response
    let nodes_json: Vec<serde_json::Value> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let (x, y) = positions.get(i).copied().unwrap_or((400.0, 300.0));
            serde_json::json!({
                "id": n.id,
                "label": if n.content.chars().count() > 60 {
                    format!("{}...", n.content.chars().take(57).collect::<String>())
                } else {
                    n.content.clone()
                },
                "type": n.node_type,
                "retention": n.retention_strength,
                "tags": n.tags,
                "x": (x * 100.0).round() / 100.0,
                "y": (y * 100.0).round() / 100.0,
                "isCenter": n.id == center_id,
                // v2.0.5 Active Forgetting — dashboard uses these to dim suppressed nodes
                "suppression_count": n.suppression_count,
                "suppressed_at": n.suppressed_at.map(|t| t.to_rfc3339()),
            })
        })
        .collect();

    let edges_json: Vec<serde_json::Value> = edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "source": e.source_id,
                "target": e.target_id,
                "weight": e.strength,
                "type": e.link_type,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
        "center_id": center_id,
        "depth": depth,
        "nodeCount": nodes.len(),
        "edgeCount": edges.len(),
    }))
}

/// BFS over logged edges. Superseded ids are dropped when the store can name them.
fn recorded_subgraph(
    storage: &vestige_core::Storage,
    center_id: &str,
    depth: u32,
    max_nodes: usize,
) -> Result<
    (
        Vec<vestige_core::KnowledgeNode>,
        Vec<vestige_core::ConnectionRecord>,
    ),
    String,
> {
    let superseded = storage.superseded_node_ids().unwrap_or_default();
    if superseded.contains(center_id) {
        return Ok((Vec::new(), Vec::new()));
    }
    let Some(center) = storage
        .get_node(center_id)
        .map_err(|e| format!("Failed to get subgraph: {}", e))?
    else {
        return Ok((Vec::new(), Vec::new()));
    };

    let mut ids = vec![center_id.to_string()];
    let mut seen = std::collections::BTreeSet::from([center_id.to_string()]);
    let mut frontier = vec![center_id.to_string()];

    for _ in 0..depth {
        let mut next = std::collections::BTreeSet::new();
        for id in &frontier {
            let mut conns = storage
                .get_connections_for_memory(id)
                .map_err(|e| format!("Failed to get subgraph: {}", e))?;
            conns.sort_by(|a, b| {
                other_end(a, id)
                    .cmp(other_end(b, id))
                    .then(a.source_id.cmp(&b.source_id))
                    .then(a.target_id.cmp(&b.target_id))
                    .then(a.link_type.cmp(&b.link_type))
            });
            for edge in &conns {
                let other = other_end(edge, id);
                if other == id
                    || seen.contains(other)
                    || superseded.contains(other)
                    || next.contains(other)
                {
                    continue;
                }
                let live = storage
                    .get_node(other)
                    .map_err(|e| format!("Failed to get subgraph: {}", e))?
                    .is_some();
                if live {
                    next.insert(other.to_string());
                }
            }
        }
        let mut added = Vec::new();
        for id in next {
            if ids.len() >= max_nodes {
                break;
            }
            seen.insert(id.clone());
            ids.push(id.clone());
            added.push(id);
        }
        if added.is_empty() || ids.len() >= max_nodes {
            break;
        }
        frontier = added;
    }

    let mut nodes = Vec::with_capacity(ids.len());
    nodes.push(center);
    for id in ids.iter().skip(1) {
        if let Some(node) = storage
            .get_node(id)
            .map_err(|e| format!("Failed to get subgraph: {}", e))?
        {
            nodes.push(node);
        }
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));

    let mut edges = storage
        .get_all_connections()
        .map_err(|e| format!("Failed to get subgraph: {}", e))?;
    edges.retain(|edge| seen.contains(&edge.source_id) && seen.contains(&edge.target_id));
    edges.sort_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.target_id.cmp(&b.target_id))
            .then(a.link_type.cmp(&b.link_type))
            .then(a.created_at.cmp(&b.created_at))
    });
    Ok((nodes, edges))
}

fn other_end<'a>(edge: &'a vestige_core::ConnectionRecord, id: &str) -> &'a str {
    if edge.source_id == id {
        edge.target_id.as_str()
    } else {
        edge.source_id.as_str()
    }
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    #[test]
    fn test_schema_is_valid() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["center_id"].is_object());
        assert!(s["properties"]["query"].is_object());
        assert!(s["properties"]["depth"].is_object());
        assert!(s["properties"]["max_nodes"].is_object());
    }

    #[test]
    fn test_fruchterman_reingold_empty() {
        let positions = fruchterman_reingold(0, &[], 800.0, 600.0, 50);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_fruchterman_reingold_single_node() {
        let positions = fruchterman_reingold(1, &[], 800.0, 600.0, 50);
        assert_eq!(positions.len(), 1);
        assert!((positions[0].0 - 400.0).abs() < 0.01);
        assert!((positions[0].1 - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_fruchterman_reingold_two_nodes() {
        let edges = vec![(0, 1, 1.0)];
        let positions = fruchterman_reingold(2, &edges, 800.0, 600.0, 50);
        assert_eq!(positions.len(), 2);
        // Nodes should be within bounds
        for (x, y) in &positions {
            assert!(*x >= 10.0 && *x <= 790.0);
            assert!(*y >= 10.0 && *y <= 590.0);
        }
    }

    #[test]
    fn test_fruchterman_reingold_connected_graph() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let positions = fruchterman_reingold(3, &edges, 800.0, 600.0, 50);
        assert_eq!(positions.len(), 3);
        // Connected nodes should be closer than disconnected nodes in a larger graph
        for (x, y) in &positions {
            assert!(*x >= 10.0 && *x <= 790.0);
            assert!(*y >= 10.0 && *y <= 590.0);
        }
    }

    #[tokio::test]
    async fn test_graph_empty_database() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err()); // No memories to center on
    }

    #[tokio::test]
    async fn test_graph_with_center_id() {
        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Graph test memory".to_string(),
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

        let args = serde_json::json!({ "center_id": node.id });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["center_id"], node.id);
        assert_eq!(value["nodeCount"], 1);
        let nodes = value["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0]["isCenter"], true);
    }

    #[tokio::test]
    async fn test_graph_with_query() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(vestige_core::IngestInput {
                content: "Quantum computing fundamentals".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["science".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();

        let args = serde_json::json!({ "query": "quantum" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value["nodeCount"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_graph_node_has_position() {
        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Position test memory".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();

        let args = serde_json::json!({ "center_id": node.id });
        let result = execute(&storage, Some(args)).await.unwrap();
        let nodes = result["nodes"].as_array().unwrap();
        assert!(nodes[0]["x"].is_number());
        assert!(nodes[0]["y"].is_number());
    }
}

#[cfg(test)]
mod strata_stdio {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use crate::server::McpServer;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::sync::Mutex;

    fn plant() -> (
        tempfile::TempDir,
        String,
        String,
        String,
        String,
        String,
        String,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        // Default policy holds RETIRE. Copy the allow verdict onto that rule so
        // the planted supersession is a real admitted log record.
        let mut policy = strata_store::default_policy();
        let allow = policy.rules[1].verdict;
        policy.rules[0].verdict = allow;
        let mut store = strata_store::StrataStore::open_with_policy(dir.path(), policy).unwrap();
        let ingest = |store: &mut strata_store::StrataStore, content: &str| {
            store
                .ingest(strata_store::IngestInput {
                    content: content.into(),
                    ..strata_store::IngestInput::default()
                })
                .unwrap()
        };
        let a = ingest(&mut store, "alpha");
        let b = ingest(&mut store, "beta");
        let c = ingest(&mut store, "gamma");
        let superseded = ingest(&mut store, "superseded");
        let beyond = ingest(&mut store, "beyond superseded");
        let isolated = ingest(&mut store, "isolated");
        let edge =
            |source: &str, target: &str, kind: &str, milli: i64| strata_store::ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength_milli: milli,
                link_type: kind.to_string(),
                created_at_ms: 1,
                ..strata_store::ConnectionRecord::default()
            };
        store
            .save_connection(&edge(&a, &b, "derived_from", 800))
            .unwrap();
        store
            .save_connection(&edge(&b, &c, "touched", 500))
            .unwrap();
        store
            .save_connection(&edge(&a, &superseded, "evidence_of", 1000))
            .unwrap();
        store
            .save_connection(&edge(&superseded, &beyond, "touched", 1000))
            .unwrap();
        store.supersede(&superseded, &b).unwrap();
        (dir, a, b, c, superseded, beyond, isolated)
    }

    async fn drive(storage: Arc<Storage>, input: &str) -> Vec<Value> {
        let server = McpServer::new(storage, Arc::new(Mutex::new(CognitiveEngine::new())));
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, mut client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            crate::protocol::stdio::run_io(server, None, BufReader::new(server_r), server_w).await
        });
        client_w.write_all(input.as_bytes()).await.unwrap();
        drop(client_w);
        let mut buf = String::new();
        client_r.read_to_string(&mut buf).await.unwrap();
        handle.await.unwrap().unwrap();
        buf.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("stdio line is JSON"))
            .collect()
    }

    #[tokio::test]
    async fn memory_graph_stdio_returns_planted_recorded_edges() {
        let (dir, a, b, c, superseded, beyond, isolated) = plant();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let init = json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "memory-graph", "version": "1"}
            }
        });
        let list = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"});
        let call = json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "graph",
                "arguments": {"action": "memory_graph", "center_id": a, "depth": 2}
            }
        });
        let input = format!(
            "{init}\n{}\n{list}\n{call}\n",
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
        );
        let out = drive(storage, &input).await;
        let listed = out
            .iter()
            .find(|v| v["id"] == json!(1))
            .expect("tools/list");
        let graph_tool = listed["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .find(|tool| tool["name"] == "graph")
            .expect("graph advertises memory_graph");
        let advertised = graph_tool.to_string();
        assert!(
            advertised.contains("memory_graph"),
            "graph schema must advertise memory_graph: {graph_tool}"
        );

        let response = out
            .iter()
            .find(|v| v["id"] == json!(2))
            .expect("tools/call");
        assert!(
            response.get("error").is_none(),
            "protocol error: {response}"
        );
        let body = &response["result"];
        assert_eq!(body["isError"], json!(false), "{body}");
        let graph = &body["structuredContent"];
        assert_eq!(graph["center_id"], json!(a));
        assert_eq!(graph["nodeCount"], json!(3));
        assert_eq!(graph["edgeCount"], json!(2));
        let node_ids: Vec<&str> = graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| node["id"].as_str().unwrap())
            .collect();
        assert_eq!(node_ids, vec![a.as_str(), b.as_str(), c.as_str()]);
        let edges = graph["edges"].as_array().unwrap();
        assert_eq!(edges[0]["source"], json!(a));
        assert_eq!(edges[0]["target"], json!(b));
        assert_eq!(edges[0]["type"], json!("derived_from"));
        assert_eq!(edges[0]["weight"], json!(0.8));
        assert_eq!(edges[1]["source"], json!(b));
        assert_eq!(edges[1]["target"], json!(c));
        assert_eq!(edges[1]["type"], json!("touched"));
        assert_eq!(edges[1]["weight"], json!(0.5));
        let blob = graph.to_string();
        assert!(!blob.contains(&superseded), "{graph}");
        assert!(!blob.contains(&beyond), "{graph}");
        assert!(!blob.contains(&isolated), "{graph}");
    }
}
