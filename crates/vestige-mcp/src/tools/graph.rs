//! memory_graph tool — Subgraph export with force-directed layout for visualization.
//! v1.9.0: Computes Fruchterman-Reingold layout server-side.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

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

    // Strata has no FTS, embeddings, or inferred links. The subgraph is the
    // center plus nodes reached by recorded edges, with superseded nodes left
    // out when the log actually has that lineage.
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return recorded_memory_graph(storage.as_ref(), args.as_ref(), depth, max_nodes);
    }

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

    // Get subgraph
    let (nodes, edges) = storage
        .get_memory_subgraph(&center_id, depth, max_nodes)
        .map_err(|e| format!("Failed to get subgraph: {}", e))?;

    if nodes.is_empty() || !nodes.iter().any(|n| n.id == center_id) {
        return Err(format!(
            "Memory '{}' not found or has no accessible data",
            center_id
        ));
    }

    Ok(render_subgraph(&center_id, depth, &nodes, &edges))
}

/// Nodes reachable from `center_id` by recorded edges only.
///
/// Neighbor expansion is sorted by id, and the returned nodes and edges are
/// sorted, so the same log always yields the same subgraph. Superseded ids
/// are dropped when the backend can name them. Missing lineage data is an
/// empty exclusion set, not a failure. Nothing here searches text, embeddings,
/// or entity overlap, and no edge is invented for a pair the log did not record.
fn recorded_memory_graph(
    storage: &Storage,
    args: Option<&serde_json::Value>,
    depth: u32,
    max_nodes: usize,
) -> Result<serde_json::Value, String> {
    let superseded = match storage.superseded_node_ids() {
        Ok(ids) => ids,
        Err(err) => {
            let text = err.to_string();
            if text.contains("not implemented") {
                HashSet::new()
            } else {
                return Err(text);
            }
        }
    };
    let center_id = resolve_recorded_center(storage, args, &superseded)?;
    let (nodes, edges) = recorded_subgraph(storage, &center_id, depth, max_nodes, &superseded)?;
    Ok(render_subgraph(&center_id, depth, &nodes, &edges))
}

fn resolve_recorded_center(
    storage: &Storage,
    args: Option<&serde_json::Value>,
    superseded: &HashSet<String>,
) -> Result<String, String> {
    if let Some(id) = args
        .and_then(|value| value.get("center_id"))
        .and_then(|value| value.as_str())
        .filter(|id| !id.is_empty())
    {
        return Ok(id.to_string());
    }
    if let Some(query) = args
        .and_then(|value| value.get("query"))
        .and_then(|value| value.as_str())
        .filter(|query| !query.is_empty())
    {
        let exact = storage
            .get_node(query)
            .map_err(|err| err.to_string())?
            .filter(|node| !superseded.contains(&node.id));
        if exact.is_some() {
            return Ok(query.to_string());
        }
        return Err(
            "similarity_disabled: memory_graph query is not an exact node id; embeddings, cosine, BM25, FTS, Jaccard, and keyword or name matching are not Strata operations; pass center_id"
                .into(),
        );
    }

    let mut offset = 0i32;
    loop {
        let page = storage
            .get_all_nodes(64, offset)
            .map_err(|err| format!("Failed to get recent node: {err}"))?;
        if page.is_empty() {
            break;
        }
        let count = i32::try_from(page.len()).unwrap_or(i32::MAX);
        if let Some(node) = page.into_iter().find(|node| !superseded.contains(&node.id)) {
            return Ok(node.id);
        }
        offset = offset.saturating_add(count);
    }
    Err("No memories in database".into())
}

fn recorded_subgraph(
    storage: &Storage,
    center_id: &str,
    depth: u32,
    max_nodes: usize,
    superseded: &HashSet<String>,
) -> Result<(Vec<KnowledgeNode>, Vec<ConnectionRecord>), String> {
    let hidden = |id: &str| superseded.contains(id);
    if hidden(center_id)
        || storage
            .get_node(center_id)
            .map_err(|err| err.to_string())?
            .is_none()
    {
        return Err(format!(
            "Memory '{center_id}' not found or has no accessible data"
        ));
    }

    let limit = max_nodes.max(1);
    let mut ids = vec![center_id.to_string()];
    let mut id_set = BTreeSet::from([center_id.to_string()]);
    let mut frontier = vec![center_id.to_string()];

    for _ in 0..depth {
        if ids.len() >= limit {
            break;
        }
        let mut next = BTreeSet::new();
        for id in &frontier {
            let connections = storage
                .get_connections_for_memory(id)
                .map_err(|err| err.to_string())?;
            for conn in connections {
                let other = if conn.source_id == *id {
                    conn.target_id
                } else {
                    conn.source_id
                };
                if id_set.contains(&other) || next.contains(&other) || hidden(&other) {
                    continue;
                }
                let is_node = storage
                    .get_node(&other)
                    .map_err(|err| err.to_string())?
                    .is_some();
                if is_node {
                    next.insert(other);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        let mut advanced = Vec::new();
        for id in next {
            if ids.len() >= limit {
                break;
            }
            id_set.insert(id.clone());
            ids.push(id.clone());
            advanced.push(id);
        }
        frontier = advanced;
    }

    let mut nodes = Vec::with_capacity(ids.len());
    for id in &ids {
        if hidden(id) {
            continue;
        }
        if let Some(node) = storage.get_node(id).map_err(|err| err.to_string())? {
            nodes.push(node);
        }
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));

    let included: BTreeSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    let mut edges = storage
        .get_all_connections()
        .map_err(|err| err.to_string())?;
    edges.retain(|edge| {
        included.contains(edge.source_id.as_str())
            && included.contains(edge.target_id.as_str())
            && !hidden(&edge.source_id)
            && !hidden(&edge.target_id)
    });
    edges.sort_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.target_id.cmp(&b.target_id))
            .then(a.link_type.cmp(&b.link_type))
            .then(a.created_at.cmp(&b.created_at))
            .then(a.activation_count.cmp(&b.activation_count))
    });
    Ok((nodes, edges))
}

fn render_subgraph(
    center_id: &str,
    depth: u32,
    nodes: &[KnowledgeNode],
    edges: &[ConnectionRecord],
) -> serde_json::Value {
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

    let positions = fruchterman_reingold(nodes.len(), &layout_edges, 800.0, 600.0, 50);

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

    serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
        "center_id": center_id,
        "depth": depth,
        "nodeCount": nodes.len(),
        "edgeCount": edges.len(),
    })
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
