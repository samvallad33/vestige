//! # Spreading Activation Network
//!
//! Implementation of Collins & Loftus (1975) Spreading Activation Theory
//! for semantic memory retrieval.
//!
//! ## Theory
//!
//! Memory is organized as a semantic network where:
//! - Concepts are nodes with activation levels
//! - Related concepts are connected by weighted edges
//! - Activating one concept spreads activation to related concepts
//! - Activation decays with distance and time
//!
//! ## References
//!
//! - Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic
//!   processing. Psychological Review, 82(6), 407-428.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default decay factor per hop in the network
const DEFAULT_DECAY_FACTOR: f64 = 0.7;

/// Maximum activation level
const MAX_ACTIVATION: f64 = 1.0;

/// Minimum activation threshold for propagation
const MIN_ACTIVATION_THRESHOLD: f64 = 0.1;

// ============================================================================
// LINK TYPES
// ============================================================================

/// Types of associative links between memories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum LinkType {
    /// Same topic/category
    #[default]
    Semantic,
    /// Occurred together in time
    Temporal,
    /// Spatial co-occurrence
    Spatial,
    /// Causal relationship
    Causal,
    /// Part-whole relationship
    PartOf,
    /// User-defined association
    UserDefined,
}


// ============================================================================
// ASSOCIATION EDGE
// ============================================================================

/// An edge connecting two nodes in the activation network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssociationEdge {
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
    /// Strength of the association (0.0-1.0)
    pub strength: f64,
    /// Type of association
    pub link_type: LinkType,
    /// When the association was created
    pub created_at: DateTime<Utc>,
    /// When the association was last reinforced
    pub last_activated: DateTime<Utc>,
    /// Number of times this link was traversed
    pub activation_count: u32,
}

impl AssociationEdge {
    /// Create a new association edge
    pub fn new(source_id: String, target_id: String, link_type: LinkType, strength: f64) -> Self {
        let now = Utc::now();
        Self {
            source_id,
            target_id,
            strength: strength.clamp(0.0, 1.0),
            link_type,
            created_at: now,
            last_activated: now,
            activation_count: 0,
        }
    }

    /// Reinforce the edge (increases strength)
    pub fn reinforce(&mut self, amount: f64) {
        self.strength = (self.strength + amount).min(1.0);
        self.last_activated = Utc::now();
        self.activation_count += 1;
    }

    /// Decay the edge strength over time
    pub fn apply_decay(&mut self, decay_rate: f64) {
        self.strength *= decay_rate;
    }
}

// ============================================================================
// ACTIVATION NODE
// ============================================================================

/// A node in the activation network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationNode {
    /// Unique node ID (typically memory ID)
    pub id: String,
    /// Current activation level (0.0-1.0)
    pub activation: f64,
    /// When this node was last activated
    pub last_activated: DateTime<Utc>,
    /// Outgoing edges
    pub edges: Vec<String>,
}

impl ActivationNode {
    /// Create a new node
    pub fn new(id: String) -> Self {
        Self {
            id,
            activation: 0.0,
            last_activated: Utc::now(),
            edges: Vec::new(),
        }
    }

    /// Activate this node
    pub fn activate(&mut self, level: f64) {
        self.activation = level.clamp(0.0, MAX_ACTIVATION);
        self.last_activated = Utc::now();
    }

    /// Add activation (accumulates)
    pub fn add_activation(&mut self, amount: f64) {
        self.activation = (self.activation + amount).min(MAX_ACTIVATION);
        self.last_activated = Utc::now();
    }

    /// Check if node is above activation threshold
    pub fn is_active(&self) -> bool {
        self.activation >= MIN_ACTIVATION_THRESHOLD
    }
}

// ============================================================================
// ACTIVATED MEMORY
// ============================================================================

/// A memory that has been activated through spreading activation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivatedMemory {
    /// Memory ID
    pub memory_id: String,
    /// Activation level (0.0-1.0)
    pub activation: f64,
    /// Distance from source (number of hops)
    pub distance: u32,
    /// Path from source to this memory
    pub path: Vec<String>,
    /// Type of link that brought activation here
    pub link_type: LinkType,
}

// ============================================================================
// ASSOCIATED MEMORY
// ============================================================================

/// A memory associated with another through the network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssociatedMemory {
    /// Memory ID
    pub memory_id: String,
    /// Association strength
    pub association_strength: f64,
    /// Type of association
    pub link_type: LinkType,
}

// ============================================================================
// ACTIVATION CONFIG
// ============================================================================

/// Configuration for spreading activation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationConfig {
    /// Decay factor per hop (0.0-1.0)
    pub decay_factor: f64,
    /// Maximum hops to propagate
    pub max_hops: u32,
    /// Minimum activation threshold
    pub min_threshold: f64,
    /// Whether to allow activation cycles
    pub allow_cycles: bool,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            decay_factor: DEFAULT_DECAY_FACTOR,
            max_hops: 3,
            min_threshold: MIN_ACTIVATION_THRESHOLD,
            allow_cycles: false,
        }
    }
}

// ============================================================================
// ACTIVATION NETWORK
// ============================================================================

/// The spreading activation network
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationNetwork {
    /// All nodes in the network
    nodes: HashMap<String, ActivationNode>,
    /// All edges in the network
    edges: HashMap<(String, String), AssociationEdge>,
    /// Configuration
    config: ActivationConfig,
}

impl Default for ActivationNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationNetwork {
    /// Create a new empty network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            config: ActivationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ActivationConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            config,
        }
    }

    /// Add a node to the network
    pub fn add_node(&mut self, id: String) {
        self.nodes
            .entry(id.clone())
            .or_insert_with(|| ActivationNode::new(id));
    }

    /// Add an edge between two nodes
    pub fn add_edge(
        &mut self,
        source: String,
        target: String,
        link_type: LinkType,
        strength: f64,
    ) {
        // Ensure both nodes exist
        self.add_node(source.clone());
        self.add_node(target.clone());

        // Add edge
        let edge = AssociationEdge::new(source.clone(), target.clone(), link_type, strength);
        self.edges.insert((source.clone(), target.clone()), edge);

        // Update node's edge list
        if let Some(node) = self.nodes.get_mut(&source) {
            if !node.edges.contains(&target) {
                node.edges.push(target);
            }
        }
    }

    /// Activate a node and spread activation through the network
    pub fn activate(&mut self, source_id: &str, initial_activation: f64) -> Vec<ActivatedMemory> {
        let mut results = Vec::new();
        let mut visited = HashMap::new();

        // Activate source node
        if let Some(node) = self.nodes.get_mut(source_id) {
            node.activate(initial_activation);
        }

        // BFS to spread activation
        let mut queue = vec![(
            source_id.to_string(),
            initial_activation,
            0u32,
            vec![source_id.to_string()],
        )];

        while let Some((current_id, current_activation, hops, path)) = queue.pop() {
            // Skip if we've visited this node with higher activation
            if let Some(&prev_activation) = visited.get(&current_id) {
                if prev_activation >= current_activation {
                    continue;
                }
            }
            visited.insert(current_id.clone(), current_activation);

            // Check hop limit
            if hops >= self.config.max_hops {
                continue;
            }

            // Get outgoing edges
            if let Some(node) = self.nodes.get(&current_id) {
                for target_id in node.edges.clone() {
                    let edge_key = (current_id.clone(), target_id.clone());
                    if let Some(edge) = self.edges.get(&edge_key) {
                        // Calculate propagated activation
                        let propagated =
                            current_activation * edge.strength * self.config.decay_factor;

                        if propagated >= self.config.min_threshold {
                            // Activate target node
                            if let Some(target_node) = self.nodes.get_mut(&target_id) {
                                target_node.add_activation(propagated);
                            }

                            // Add to results
                            let mut new_path = path.clone();
                            new_path.push(target_id.clone());

                            results.push(ActivatedMemory {
                                memory_id: target_id.clone(),
                                activation: propagated,
                                distance: hops + 1,
                                path: new_path.clone(),
                                link_type: edge.link_type,
                            });

                            // Add to queue for further propagation
                            queue.push((target_id.clone(), propagated, hops + 1, new_path));
                        }
                    }
                }
            }
        }

        // Sort by activation level
        results.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get directly associated memories for a given memory
    pub fn get_associations(&self, memory_id: &str) -> Vec<AssociatedMemory> {
        let mut associations = Vec::new();

        if let Some(node) = self.nodes.get(memory_id) {
            for target_id in &node.edges {
                let edge_key = (memory_id.to_string(), target_id.clone());
                if let Some(edge) = self.edges.get(&edge_key) {
                    associations.push(AssociatedMemory {
                        memory_id: target_id.clone(),
                        association_strength: edge.strength,
                        link_type: edge.link_type,
                    });
                }
            }
        }

        associations.sort_by(|a, b| {
            b.association_strength
                .partial_cmp(&a.association_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        associations
    }

    /// Reinforce an edge (called when both nodes are accessed together)
    pub fn reinforce_edge(&mut self, source: &str, target: &str, amount: f64) {
        let key = (source.to_string(), target.to_string());
        if let Some(edge) = self.edges.get_mut(&key) {
            edge.reinforce(amount);
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Clear all activations
    pub fn clear_activations(&mut self) {
        for node in self.nodes.values_mut() {
            node.activation = 0.0;
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = ActivationNetwork::new();
        assert_eq!(network.node_count(), 0);
        assert_eq!(network.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let mut network = ActivationNetwork::new();

        network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.8);
        network.add_edge("b".to_string(), "c".to_string(), LinkType::Temporal, 0.6);

        assert_eq!(network.node_count(), 3);
        assert_eq!(network.edge_count(), 2);
    }

    #[test]
    fn test_spreading_activation() {
        let mut network = ActivationNetwork::new();

        network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.8);
        network.add_edge("b".to_string(), "c".to_string(), LinkType::Semantic, 0.8);
        network.add_edge("a".to_string(), "d".to_string(), LinkType::Semantic, 0.5);

        let results = network.activate("a", 1.0);

        // Should have activated b, c, and d
        assert!(!results.is_empty());

        // b should have higher activation than c (closer to source)
        let b_activation = results
            .iter()
            .find(|r| r.memory_id == "b")
            .map(|r| r.activation);
        let c_activation = results
            .iter()
            .find(|r| r.memory_id == "c")
            .map(|r| r.activation);

        assert!(b_activation.unwrap_or(0.0) > c_activation.unwrap_or(0.0));
    }

    #[test]
    fn test_get_associations() {
        let mut network = ActivationNetwork::new();

        network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.9);
        network.add_edge("a".to_string(), "c".to_string(), LinkType::Temporal, 0.5);

        let associations = network.get_associations("a");

        assert_eq!(associations.len(), 2);
        assert_eq!(associations[0].memory_id, "b"); // Sorted by strength
        assert_eq!(associations[0].association_strength, 0.9);
    }

    #[test]
    fn test_reinforce_edge() {
        let mut network = ActivationNetwork::new();

        network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.5);
        network.reinforce_edge("a", "b", 0.2);

        let associations = network.get_associations("a");
        assert!(associations[0].association_strength > 0.5);
    }

    #[test]
    fn test_activation_threshold() {
        let mut network = ActivationNetwork::with_config(ActivationConfig {
            decay_factor: 0.1, // Very high decay
            min_threshold: 0.5, // High threshold
            ..Default::default()
        });

        network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.5);
        network.add_edge("b".to_string(), "c".to_string(), LinkType::Semantic, 0.5);

        let results = network.activate("a", 1.0);

        // c should not be activated due to high decay and threshold
        let c_activated = results.iter().any(|r| r.memory_id == "c");
        assert!(!c_activated);
    }
}
