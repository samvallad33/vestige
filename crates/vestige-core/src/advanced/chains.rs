//! # Memory Chains (Reasoning)
//!
//! Build chains of reasoning from memory, connecting concepts through
//! their relationships. This enables Vestige to explain HOW it arrived
//! at a conclusion, not just WHAT the conclusion is.
//!
//! ## Use Cases
//!
//! - **Explanation**: "Why do you think X is related to Y?"
//! - **Discovery**: Find non-obvious connections between concepts
//! - **Debugging**: Trace how a bug in A could affect component B
//! - **Learning**: Understand relationships in a domain
//!
//! ## How It Works
//!
//! 1. **Graph Traversal**: Navigate the knowledge graph using BFS/DFS
//! 2. **Path Scoring**: Score paths by relevance and connection strength
//! 3. **Chain Building**: Construct reasoning chains from paths
//! 4. **Explanation Generation**: Generate human-readable explanations
//!
//! ## Example
//!
//! ```rust,ignore
//! let builder = MemoryChainBuilder::new();
//!
//! // Build a reasoning chain from "database" to "performance"
//! let chain = builder.build_chain("database", "performance");
//!
//! // Shows: database -> indexes -> query optimization -> performance
//! for step in chain.steps {
//!     println!("{}: {} -> {}", step.reasoning, step.memory, step.connection_type);
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Maximum depth for chain building
const MAX_CHAIN_DEPTH: usize = 10;

/// Maximum paths to explore
const MAX_PATHS_TO_EXPLORE: usize = 1000;

/// Minimum connection strength to consider
const MIN_CONNECTION_STRENGTH: f64 = 0.2;

/// Types of connections between memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConnectionType {
    /// Direct semantic similarity
    SemanticSimilarity,
    /// Same topic/tag
    SharedTopic,
    /// Temporal proximity (happened around same time)
    TemporalProximity,
    /// Causal relationship (A causes B)
    Causal,
    /// Part-whole relationship
    PartOf,
    /// Example-of relationship
    ExampleOf,
    /// Prerequisite relationship (need A to understand B)
    Prerequisite,
    /// Contradiction/conflict
    Contradicts,
    /// Elaboration (B provides more detail on A)
    Elaborates,
    /// Same entity/concept
    SameEntity,
    /// Used together
    UsedTogether,
    /// Custom relationship
    Custom(String),
}

impl ConnectionType {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            Self::SemanticSimilarity => "is semantically similar to",
            Self::SharedTopic => "shares topic with",
            Self::TemporalProximity => "happened around the same time as",
            Self::Causal => "causes or leads to",
            Self::PartOf => "is part of",
            Self::ExampleOf => "is an example of",
            Self::Prerequisite => "is a prerequisite for",
            Self::Contradicts => "contradicts",
            Self::Elaborates => "provides more detail about",
            Self::SameEntity => "refers to the same thing as",
            Self::UsedTogether => "is commonly used with",
            Self::Custom(_) => "is related to",
        }
    }

    /// Get default strength for this connection type
    pub fn default_strength(&self) -> f64 {
        match self {
            Self::SameEntity => 1.0,
            Self::Causal | Self::PartOf => 0.9,
            Self::Prerequisite | Self::Elaborates => 0.8,
            Self::SemanticSimilarity => 0.7,
            Self::SharedTopic | Self::UsedTogether => 0.6,
            Self::ExampleOf => 0.7,
            Self::TemporalProximity => 0.4,
            Self::Contradicts => 0.5,
            Self::Custom(_) => 0.5,
        }
    }
}

/// A step in a reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    /// Memory at this step
    pub memory_id: String,
    /// Content preview
    pub memory_preview: String,
    /// How this connects to the next step
    pub connection_type: ConnectionType,
    /// Strength of this connection (0.0 to 1.0)
    pub connection_strength: f64,
    /// Human-readable reasoning for this step
    pub reasoning: String,
}

/// A complete reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    /// Starting concept/memory
    pub from: String,
    /// Ending concept/memory
    pub to: String,
    /// Steps in the chain
    pub steps: Vec<ChainStep>,
    /// Overall confidence in this chain
    pub confidence: f64,
    /// Total number of hops
    pub total_hops: usize,
    /// Human-readable explanation of the chain
    pub explanation: String,
}

impl ReasoningChain {
    /// Check if this is a valid chain (reaches destination)
    pub fn is_complete(&self) -> bool {
        if let Some(last) = self.steps.last() {
            last.memory_id == self.to || self.steps.iter().any(|s| s.memory_id == self.to)
        } else {
            false
        }
    }

    /// Get the path as a list of memory IDs
    pub fn path_ids(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.memory_id.clone()).collect()
    }
}

/// A path between memories (used during search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPath {
    /// Memory IDs in order
    pub memories: Vec<String>,
    /// Connections between consecutive memories
    pub connections: Vec<Connection>,
    /// Total path score
    pub score: f64,
}

/// A connection between two memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source memory
    pub from_id: String,
    /// Target memory
    pub to_id: String,
    /// Type of connection
    pub connection_type: ConnectionType,
    /// Strength (0.0 to 1.0)
    pub strength: f64,
    /// When this connection was established
    pub created_at: DateTime<Utc>,
}

/// Memory node for graph operations
#[derive(Debug, Clone)]
pub struct MemoryNode {
    /// Memory ID
    pub id: String,
    /// Content preview
    pub content_preview: String,
    /// Tags/topics
    pub tags: Vec<String>,
    /// Connections to other memories
    pub connections: Vec<Connection>,
}

/// State for path search (used in priority queue)
#[derive(Debug, Clone)]
struct SearchState {
    memory_id: String,
    path: Vec<String>,
    connections: Vec<Connection>,
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
        // Higher score = higher priority
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Builder for memory reasoning chains
pub struct MemoryChainBuilder {
    /// Memory graph (loaded from storage)
    graph: HashMap<String, MemoryNode>,
    /// Reverse index: tag -> memory IDs
    tag_index: HashMap<String, Vec<String>>,
}

impl MemoryChainBuilder {
    /// Create a new chain builder
    pub fn new() -> Self {
        Self {
            graph: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Load a memory node into the graph
    pub fn add_memory(&mut self, node: MemoryNode) {
        // Update tag index
        for tag in &node.tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .push(node.id.clone());
        }

        self.graph.insert(node.id.clone(), node);
    }

    /// Add a connection between memories
    pub fn add_connection(&mut self, connection: Connection) {
        if let Some(node) = self.graph.get_mut(&connection.from_id) {
            node.connections.push(connection);
        }
    }

    /// Build a reasoning chain from one concept to another
    pub fn build_chain(&self, from: &str, to: &str) -> Option<ReasoningChain> {
        // Find all paths and pick the best one
        let paths = self.find_paths(from, to);

        if paths.is_empty() {
            return None;
        }

        // Convert best path to chain
        let best_path = paths.into_iter().next()?;
        self.path_to_chain(from, to, best_path)
    }

    /// Find all paths between two concepts
    pub fn find_paths(&self, concept_a: &str, concept_b: &str) -> Vec<MemoryPath> {
        // Resolve concepts to memory IDs
        let start_ids = self.resolve_concept(concept_a);
        let end_ids: HashSet<_> = self.resolve_concept(concept_b).into_iter().collect();

        if start_ids.is_empty() || end_ids.is_empty() {
            return vec![];
        }

        let mut all_paths = Vec::new();

        // BFS from each starting point
        for start_id in start_ids {
            let paths = self.bfs_find_paths(&start_id, &end_ids);
            all_paths.extend(paths);
        }

        // Sort by score (descending)
        all_paths.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top paths
        all_paths.into_iter().take(10).collect()
    }

    /// Build a chain explaining why two concepts are related
    pub fn explain_relationship(&self, from: &str, to: &str) -> Option<String> {
        let chain = self.build_chain(from, to)?;
        Some(chain.explanation)
    }

    /// Find memories that connect two concepts
    pub fn find_bridge_memories(&self, concept_a: &str, concept_b: &str) -> Vec<String> {
        let paths = self.find_paths(concept_a, concept_b);

        // Collect memories that appear as intermediate steps
        let mut bridges: HashMap<String, usize> = HashMap::new();

        for path in paths {
            if path.memories.len() > 2 {
                for mem in &path.memories[1..path.memories.len() - 1] {
                    *bridges.entry(mem.clone()).or_insert(0) += 1;
                }
            }
        }

        // Sort by frequency
        let mut bridge_list: Vec<_> = bridges.into_iter().collect();
        bridge_list.sort_by_key(|b| std::cmp::Reverse(b.1));

        bridge_list.into_iter().map(|(id, _)| id).collect()
    }

    /// Get the number of memories in the graph
    pub fn memory_count(&self) -> usize {
        self.graph.len()
    }

    /// Get the number of connections in the graph
    pub fn connection_count(&self) -> usize {
        self.graph.values().map(|n| n.connections.len()).sum()
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    fn resolve_concept(&self, concept: &str) -> Vec<String> {
        // First, check if it's a direct memory ID
        if self.graph.contains_key(concept) {
            return vec![concept.to_string()];
        }

        // Check tag index
        if let Some(ids) = self.tag_index.get(concept) {
            return ids.clone();
        }

        // Search by content (simplified - would use embeddings in production)
        let concept_lower = concept.to_lowercase();
        self.graph
            .values()
            .filter(|node| node.content_preview.to_lowercase().contains(&concept_lower))
            .map(|node| node.id.clone())
            .take(10)
            .collect()
    }

    fn bfs_find_paths(&self, start: &str, targets: &HashSet<String>) -> Vec<MemoryPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = BinaryHeap::new();

        queue.push(SearchState {
            memory_id: start.to_string(),
            path: vec![start.to_string()],
            connections: vec![],
            score: 1.0,
            depth: 0,
        });

        let mut explored = 0;

        while let Some(state) = queue.pop() {
            explored += 1;
            if explored > MAX_PATHS_TO_EXPLORE {
                break;
            }

            // Check if we reached a target
            if targets.contains(&state.memory_id) {
                paths.push(MemoryPath {
                    memories: state.path,
                    connections: state.connections,
                    score: state.score,
                });
                continue;
            }

            // Don't revisit or go too deep
            if state.depth >= MAX_CHAIN_DEPTH {
                continue;
            }

            let visit_key = (state.memory_id.clone(), state.depth);
            if visited.contains(&visit_key) {
                continue;
            }
            visited.insert(visit_key);

            // Expand neighbors
            if let Some(node) = self.graph.get(&state.memory_id) {
                for conn in &node.connections {
                    if conn.strength < MIN_CONNECTION_STRENGTH {
                        continue;
                    }

                    if state.path.contains(&conn.to_id) {
                        continue; // Avoid cycles
                    }

                    let mut new_path = state.path.clone();
                    new_path.push(conn.to_id.clone());

                    let mut new_connections = state.connections.clone();
                    new_connections.push(conn.clone());

                    // Score decays with depth and connection strength
                    let new_score = state.score * conn.strength * 0.9;

                    queue.push(SearchState {
                        memory_id: conn.to_id.clone(),
                        path: new_path,
                        connections: new_connections,
                        score: new_score,
                        depth: state.depth + 1,
                    });
                }
            }

            // Also explore tag-based connections
            if let Some(node) = self.graph.get(&state.memory_id) {
                for tag in &node.tags {
                    if let Some(related_ids) = self.tag_index.get(tag) {
                        for related_id in related_ids {
                            if state.path.contains(related_id) {
                                continue;
                            }

                            let mut new_path = state.path.clone();
                            new_path.push(related_id.clone());

                            let mut new_connections = state.connections.clone();
                            new_connections.push(Connection {
                                from_id: state.memory_id.clone(),
                                to_id: related_id.clone(),
                                connection_type: ConnectionType::SharedTopic,
                                strength: 0.5,
                                created_at: Utc::now(),
                            });

                            let new_score = state.score * 0.5 * 0.9;

                            queue.push(SearchState {
                                memory_id: related_id.clone(),
                                path: new_path,
                                connections: new_connections,
                                score: new_score,
                                depth: state.depth + 1,
                            });
                        }
                    }
                }
            }
        }

        paths
    }

    fn path_to_chain(&self, from: &str, to: &str, path: MemoryPath) -> Option<ReasoningChain> {
        if path.memories.is_empty() {
            return None;
        }

        let mut steps = Vec::new();

        for (i, mem_id) in path.memories.iter().enumerate() {
            let preview = self
                .graph
                .get(mem_id)
                .map(|n| n.content_preview.clone())
                .unwrap_or_default();

            // The connection that LED INTO memories[i] is connections[i-1] (the
            // edge memories[i-1] -> memories[i]). The start step (i==0) has no
            // incoming edge. The old code paired memories[i] with connections[i]
            // (the OUTGOING edge) and appended a synthetic connection to mask the
            // resulting off-by-one, mislabeling every step's connection type.
            let incoming = if i == 0 {
                None
            } else {
                path.connections.get(i - 1)
            };

            let reasoning = if i == 0 {
                format!("Starting from '{}'", preview)
            } else {
                let prev_preview = self
                    .graph
                    .get(&path.memories[i - 1])
                    .map(|n| n.content_preview.as_str())
                    .unwrap_or("");
                let rel = incoming
                    .map(|c| c.connection_type.description())
                    .unwrap_or("relates to");
                format!("'{}' {} '{}'", prev_preview, rel, preview)
            };

            steps.push(ChainStep {
                memory_id: mem_id.clone(),
                memory_preview: preview,
                connection_type: incoming
                    .map(|c| c.connection_type.clone())
                    .unwrap_or(ConnectionType::SemanticSimilarity),
                connection_strength: incoming.map(|c| c.strength).unwrap_or(1.0),
                reasoning,
            });
        }

        // Calculate overall confidence
        let confidence = path
            .connections
            .iter()
            .map(|c| c.strength)
            .fold(1.0, |acc, s| acc * s)
            .powf(1.0 / path.memories.len() as f64); // Geometric mean

        // Generate explanation
        let explanation = self.generate_explanation(&steps);

        Some(ReasoningChain {
            from: from.to_string(),
            to: to.to_string(),
            steps,
            confidence,
            total_hops: path.memories.len(),
            explanation,
        })
    }

    fn generate_explanation(&self, steps: &[ChainStep]) -> String {
        if steps.is_empty() {
            return "No reasoning chain found.".to_string();
        }

        let mut parts = Vec::new();

        for (i, step) in steps.iter().enumerate() {
            if i == 0 {
                parts.push(format!("Starting from '{}'", step.memory_preview));
            } else {
                parts.push(format!(
                    "which {} '{}'",
                    step.connection_type.description(),
                    step.memory_preview
                ));
            }
        }

        parts.join(", ")
    }
}

impl Default for MemoryChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_graph() -> MemoryChainBuilder {
        let mut builder = MemoryChainBuilder::new();

        // Add test memories
        builder.add_memory(MemoryNode {
            id: "database".to_string(),
            content_preview: "Database design patterns".to_string(),
            tags: vec!["database".to_string(), "architecture".to_string()],
            connections: vec![],
        });

        builder.add_memory(MemoryNode {
            id: "indexes".to_string(),
            content_preview: "Database indexing strategies".to_string(),
            tags: vec!["database".to_string(), "performance".to_string()],
            connections: vec![],
        });

        builder.add_memory(MemoryNode {
            id: "query-opt".to_string(),
            content_preview: "Query optimization techniques".to_string(),
            tags: vec!["performance".to_string(), "sql".to_string()],
            connections: vec![],
        });

        builder.add_memory(MemoryNode {
            id: "perf".to_string(),
            content_preview: "Performance best practices".to_string(),
            tags: vec!["performance".to_string()],
            connections: vec![],
        });

        // Add connections
        builder.add_connection(Connection {
            from_id: "database".to_string(),
            to_id: "indexes".to_string(),
            connection_type: ConnectionType::PartOf,
            strength: 0.9,
            created_at: Utc::now(),
        });

        builder.add_connection(Connection {
            from_id: "indexes".to_string(),
            to_id: "query-opt".to_string(),
            connection_type: ConnectionType::Causal,
            strength: 0.8,
            created_at: Utc::now(),
        });

        builder.add_connection(Connection {
            from_id: "query-opt".to_string(),
            to_id: "perf".to_string(),
            connection_type: ConnectionType::Causal,
            strength: 0.85,
            created_at: Utc::now(),
        });

        builder
    }

    #[test]
    fn test_build_chain() {
        let builder = build_test_graph();

        let chain = builder.build_chain("database", "perf");
        assert!(chain.is_some());

        let chain = chain.unwrap();
        assert!(chain.total_hops >= 2);
        assert!(chain.confidence > 0.0);
    }

    #[test]
    fn test_find_paths() {
        let builder = build_test_graph();

        let paths = builder.find_paths("database", "performance");
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_connection_description() {
        assert_eq!(ConnectionType::Causal.description(), "causes or leads to");
        assert_eq!(ConnectionType::PartOf.description(), "is part of");
    }

    #[test]
    fn test_find_bridge_memories() {
        let builder = build_test_graph();

        let bridges = builder.find_bridge_memories("database", "perf");
        // Indexes and query-opt should be bridges
        assert!(
            bridges.contains(&"indexes".to_string()) || bridges.contains(&"query-opt".to_string())
        );
    }
}
