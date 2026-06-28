//! File relationship tracking for codebase memory
//!
//! This module tracks relationships between files:
//! - Co-edit patterns (files edited together)
//! - Import/dependency relationships
//! - Test-implementation relationships
//! - Domain groupings
//!
//! Understanding file relationships helps:
//! - Suggest related files when editing
//! - Provide better context for code generation
//! - Identify architectural boundaries

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::types::{FileRelationship, RelationType, RelationshipSource};

// ============================================================================
// ERRORS
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum RelationshipError {
    #[error("Relationship not found: {0}")]
    NotFound(String),
    #[error("Invalid relationship: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, RelationshipError>;

// ============================================================================
// RELATED FILE
// ============================================================================

/// A file that is related to another file
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelatedFile {
    /// Path to the related file
    pub path: PathBuf,
    /// Type of relationship
    pub relationship_type: RelationType,
    /// Strength of the relationship (0.0 - 1.0)
    pub strength: f64,
    /// Human-readable description
    pub description: String,
}

// ============================================================================
// RELATIONSHIP GRAPH
// ============================================================================

/// Graph structure for visualizing file relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipGraph {
    /// Nodes (files) in the graph
    pub nodes: Vec<GraphNode>,
    /// Edges (relationships) in the graph
    pub edges: Vec<GraphEdge>,
    /// Graph metadata
    pub metadata: GraphMetadata,
}

/// A node in the relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    /// Unique ID for this node
    pub id: String,
    /// File path
    pub path: PathBuf,
    /// Display label
    pub label: String,
    /// Node type (for styling)
    pub node_type: String,
    /// Number of connections
    pub degree: usize,
}

/// An edge in the relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Relationship type
    pub relationship_type: RelationType,
    /// Edge weight (strength)
    pub weight: f64,
    /// Edge label
    pub label: String,
}

/// Metadata about the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphMetadata {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// When the graph was built
    pub built_at: DateTime<Utc>,
    /// Average relationship strength
    pub average_strength: f64,
}

// ============================================================================
// CO-EDIT SESSION
// ============================================================================

/// Tracks files edited together in a session
#[derive(Debug, Clone)]
struct CoEditSession {
    /// Files in this session
    files: HashSet<PathBuf>,
    /// When the session started (for analytics/debugging)
    #[allow(dead_code)]
    started_at: DateTime<Utc>,
    /// When the session was last updated
    last_updated: DateTime<Utc>,
}

// ============================================================================
// RELATIONSHIP TRACKER
// ============================================================================

/// Tracks relationships between files in a codebase
pub struct RelationshipTracker {
    /// All relationships indexed by ID
    relationships: HashMap<String, FileRelationship>,
    /// Relationships indexed by file for fast lookup
    file_relationships: HashMap<PathBuf, Vec<String>>,
    /// Current co-edit session
    current_session: Option<CoEditSession>,
    /// Co-edit counts between file pairs
    coedit_counts: HashMap<(PathBuf, PathBuf), u32>,
    /// ID counter for new relationships
    next_id: u32,
}

impl RelationshipTracker {
    /// Create a new relationship tracker
    pub fn new() -> Self {
        Self {
            relationships: HashMap::new(),
            file_relationships: HashMap::new(),
            current_session: None,
            coedit_counts: HashMap::new(),
            next_id: 1,
        }
    }

    /// Generate a new relationship ID
    fn new_id(&mut self) -> String {
        let id = format!("rel-{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a relationship
    pub fn add_relationship(&mut self, relationship: FileRelationship) -> Result<String> {
        if relationship.files.len() < 2 {
            return Err(RelationshipError::Invalid(
                "Relationship must have at least 2 files".to_string(),
            ));
        }

        let id = relationship.id.clone();

        // Reject duplicate ids: a second insert with the same id but different
        // files would overwrite the relationship while leaving the stale id in
        // each previous file's index, corrupting get_related_files.
        if self.relationships.contains_key(&id) {
            return Err(RelationshipError::Invalid(format!(
                "duplicate relationship id: {id}"
            )));
        }

        // Index by each file
        for file in &relationship.files {
            self.file_relationships
                .entry(file.clone())
                .or_default()
                .push(id.clone());
        }

        self.relationships.insert(id.clone(), relationship);

        Ok(id)
    }

    /// Record that files were edited together
    pub fn record_coedit(&mut self, files: &[PathBuf]) -> Result<()> {
        if files.len() < 2 {
            return Ok(()); // Need at least 2 files for a relationship
        }

        let now = Utc::now();

        // Update or create session
        match &mut self.current_session {
            Some(session) => {
                // Check if session is still active (within 30 minutes)
                let elapsed = now.signed_duration_since(session.last_updated);
                if elapsed.num_minutes() > 30 {
                    // Session expired, finalize it and start new
                    self.finalize_session()?;
                    self.current_session = Some(CoEditSession {
                        files: files.iter().cloned().collect(),
                        started_at: now,
                        last_updated: now,
                    });
                } else {
                    // Add files to current session
                    session.files.extend(files.iter().cloned());
                    session.last_updated = now;
                }
            }
            None => {
                // Start new session
                self.current_session = Some(CoEditSession {
                    files: files.iter().cloned().collect(),
                    started_at: now,
                    last_updated: now,
                });
            }
        }

        // Update co-edit counts for each pair
        for i in 0..files.len() {
            for j in (i + 1)..files.len() {
                let pair = if files[i] < files[j] {
                    (files[i].clone(), files[j].clone())
                } else {
                    (files[j].clone(), files[i].clone())
                };
                *self.coedit_counts.entry(pair).or_insert(0) += 1;
            }
        }

        Ok(())
    }

    /// Finalize the current session and create relationships
    fn finalize_session(&mut self) -> Result<()> {
        if let Some(session) = self.current_session.take() {
            let files: Vec<_> = session.files.into_iter().collect();

            if files.len() >= 2 {
                // Create relationships for frequent co-edits
                for i in 0..files.len() {
                    for j in (i + 1)..files.len() {
                        let pair = if files[i] < files[j] {
                            (files[i].clone(), files[j].clone())
                        } else {
                            (files[j].clone(), files[i].clone())
                        };

                        let count = self.coedit_counts.get(&pair).copied().unwrap_or(0);

                        // Only create relationship if edited together multiple times
                        if count >= 3 {
                            let strength = (count as f64 / 10.0).min(1.0);
                            let id = self.new_id();

                            let relationship = FileRelationship {
                                id: id.clone(),
                                files: vec![pair.0.clone(), pair.1.clone()],
                                relationship_type: RelationType::FrequentCochange,
                                strength,
                                description: format!(
                                    "Edited together {} times in recent sessions",
                                    count
                                ),
                                created_at: Utc::now(),
                                last_confirmed: Some(Utc::now()),
                                source: RelationshipSource::UserDefined,
                                observation_count: count,
                            };

                            // Check if relationship already exists
                            let exists = self
                                .relationships
                                .values()
                                .any(|r| r.files.contains(&pair.0) && r.files.contains(&pair.1));

                            if !exists {
                                self.add_relationship(relationship)?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get files related to a given file
    pub fn get_related_files(&self, file: &Path) -> Result<Vec<RelatedFile>> {
        let path = file.to_path_buf();

        let relationship_ids = self.file_relationships.get(&path);

        let related: Vec<_> = relationship_ids
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.relationships.get(id))
                    .flat_map(|rel| {
                        rel.files
                            .iter()
                            .filter(|f| *f != &path)
                            .map(|f| RelatedFile {
                                path: f.clone(),
                                relationship_type: rel.relationship_type,
                                strength: rel.strength,
                                description: rel.description.clone(),
                            })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Also check for test file relationships
        let mut additional = self.infer_test_relationships(file);
        additional.extend(related);

        // Deduplicate by path
        let mut seen = HashSet::new();
        let deduped: Vec<_> = additional
            .into_iter()
            .filter(|r| seen.insert(r.path.clone()))
            .collect();

        Ok(deduped)
    }

    /// Infer test file relationships based on naming conventions
    fn infer_test_relationships(&self, file: &Path) -> Vec<RelatedFile> {
        let mut related = Vec::new();

        let file_stem = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let extension = file
            .extension()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let parent = file.parent().unwrap_or(Path::new("."));

        // Check for test file naming patterns
        let is_test = file_stem.contains("test")
            || file_stem.contains("spec")
            || file_stem.ends_with("_test")
            || file_stem.starts_with("test_");

        if is_test {
            // This is a test file - find the implementation
            let impl_stem = file_stem
                .replace("_test", "")
                .replace(".test", "")
                .replace("_spec", "")
                .replace(".spec", "")
                .trim_start_matches("test_")
                .to_string();

            let impl_path = parent.join(format!("{}.{}", impl_stem, extension));

            if impl_path.exists() {
                related.push(RelatedFile {
                    path: impl_path,
                    relationship_type: RelationType::TestsImplementation,
                    strength: 0.9,
                    description: "Implementation file for this test".to_string(),
                });
            }
        } else {
            // This is an implementation - find the test file
            let test_patterns = [
                format!("{}_test.{}", file_stem, extension),
                format!("{}.test.{}", file_stem, extension),
                format!("test_{}.{}", file_stem, extension),
                format!("{}_spec.{}", file_stem, extension),
                format!("{}.spec.{}", file_stem, extension),
            ];

            for pattern in &test_patterns {
                let test_path = parent.join(pattern);
                if test_path.exists() {
                    related.push(RelatedFile {
                        path: test_path,
                        relationship_type: RelationType::TestsImplementation,
                        strength: 0.9,
                        description: "Test file for this implementation".to_string(),
                    });
                    break;
                }
            }

            // Check tests/ directory
            if let Some(grandparent) = parent.parent() {
                let tests_dir = grandparent.join("tests");
                if tests_dir.exists() {
                    for pattern in &test_patterns {
                        let test_path = tests_dir.join(pattern);
                        if test_path.exists() {
                            related.push(RelatedFile {
                                path: test_path,
                                relationship_type: RelationType::TestsImplementation,
                                strength: 0.8,
                                description: "Test file in tests/ directory".to_string(),
                            });
                        }
                    }
                }
            }
        }

        related
    }

    /// Build a relationship graph for visualization
    pub fn build_graph(&self) -> Result<RelationshipGraph> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_ids: HashMap<PathBuf, String> = HashMap::new();
        let mut node_degrees: HashMap<String, usize> = HashMap::new();

        // Build nodes from all files in relationships
        for relationship in self.relationships.values() {
            for file in &relationship.files {
                if !node_ids.contains_key(file) {
                    let id = format!("node-{}", node_ids.len());
                    node_ids.insert(file.clone(), id.clone());

                    let label = file
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| file.to_string_lossy().to_string());

                    let node_type = file
                        .extension()
                        .map(|e| e.to_string_lossy().to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    nodes.push(GraphNode {
                        id: id.clone(),
                        path: file.clone(),
                        label,
                        node_type,
                        degree: 0, // Will update later
                    });
                }
            }
        }

        // Build edges from relationships
        for relationship in self.relationships.values() {
            if relationship.files.len() >= 2 {
                // Skip relationships where files aren't in the node map
                let Some(source_id) = node_ids.get(&relationship.files[0]).cloned() else {
                    continue;
                };
                let Some(target_id) = node_ids.get(&relationship.files[1]).cloned() else {
                    continue;
                };

                // Update degrees
                *node_degrees.entry(source_id.clone()).or_insert(0) += 1;
                *node_degrees.entry(target_id.clone()).or_insert(0) += 1;

                let label = format!("{:?}", relationship.relationship_type);

                edges.push(GraphEdge {
                    source: source_id,
                    target: target_id,
                    relationship_type: relationship.relationship_type,
                    weight: relationship.strength,
                    label,
                });
            }
        }

        // Update node degrees
        for node in &mut nodes {
            node.degree = node_degrees.get(&node.id).copied().unwrap_or(0);
        }

        // Calculate metadata
        let average_strength = if edges.is_empty() {
            0.0
        } else {
            edges.iter().map(|e| e.weight).sum::<f64>() / edges.len() as f64
        };

        let metadata = GraphMetadata {
            node_count: nodes.len(),
            edge_count: edges.len(),
            built_at: Utc::now(),
            average_strength,
        };

        Ok(RelationshipGraph {
            nodes,
            edges,
            metadata,
        })
    }

    /// Get a specific relationship by ID
    pub fn get_relationship(&self, id: &str) -> Option<&FileRelationship> {
        self.relationships.get(id)
    }

    /// Get all relationships
    pub fn get_all_relationships(&self) -> Vec<&FileRelationship> {
        self.relationships.values().collect()
    }

    /// Delete a relationship
    pub fn delete_relationship(&mut self, id: &str) -> Result<()> {
        if let Some(relationship) = self.relationships.remove(id) {
            // Remove from file index
            for file in &relationship.files {
                if let Some(ids) = self.file_relationships.get_mut(file) {
                    ids.retain(|i| i != id);
                }
            }
            Ok(())
        } else {
            Err(RelationshipError::NotFound(id.to_string()))
        }
    }

    /// Get relationships by type
    pub fn get_relationships_by_type(&self, rel_type: RelationType) -> Vec<&FileRelationship> {
        self.relationships
            .values()
            .filter(|r| r.relationship_type == rel_type)
            .collect()
    }

    /// Update relationship strength
    pub fn update_strength(&mut self, id: &str, delta: f64) -> Result<()> {
        if let Some(relationship) = self.relationships.get_mut(id) {
            relationship.strength = (relationship.strength + delta).clamp(0.0, 1.0);
            relationship.last_confirmed = Some(Utc::now());
            relationship.observation_count += 1;
            Ok(())
        } else {
            Err(RelationshipError::NotFound(id.to_string()))
        }
    }

    /// Load relationships from storage
    pub fn load_relationships(&mut self, relationships: Vec<FileRelationship>) -> Result<()> {
        for relationship in relationships {
            // Advance next_id past any loaded "rel-N" id so a later new_id() can't
            // collide with a persisted one (next_id starts at 1 and is otherwise
            // never reconciled with loaded data).
            if let Some(n) = relationship
                .id
                .strip_prefix("rel-")
                .and_then(|s| s.parse::<u32>().ok())
            {
                self.next_id = self.next_id.max(n + 1);
            }
            self.add_relationship(relationship)?;
        }
        Ok(())
    }

    /// Export all relationships for storage
    pub fn export_relationships(&self) -> Vec<FileRelationship> {
        self.relationships.values().cloned().collect()
    }

    /// Get the most connected files (highest degree in graph)
    pub fn get_hub_files(&self, limit: usize) -> Vec<(PathBuf, usize)> {
        let mut file_degrees: HashMap<PathBuf, usize> = HashMap::new();

        for relationship in self.relationships.values() {
            for file in &relationship.files {
                *file_degrees.entry(file.clone()).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<_> = file_degrees.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        sorted.truncate(limit);

        sorted
    }
}

impl Default for RelationshipTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_relationship() -> FileRelationship {
        FileRelationship::new(
            "test-rel-1".to_string(),
            vec![PathBuf::from("src/main.rs"), PathBuf::from("src/lib.rs")],
            RelationType::SharedDomain,
            "Core entry points".to_string(),
        )
    }

    #[test]
    fn test_add_relationship() {
        let mut tracker = RelationshipTracker::new();
        let rel = create_test_relationship();

        let result = tracker.add_relationship(rel);
        assert!(result.is_ok());

        let stored = tracker.get_relationship("test-rel-1");
        assert!(stored.is_some());
    }

    #[test]
    fn test_get_related_files() {
        let mut tracker = RelationshipTracker::new();
        let rel = create_test_relationship();
        tracker.add_relationship(rel).unwrap();

        let related = tracker.get_related_files(Path::new("src/main.rs")).unwrap();

        assert!(!related.is_empty());
        assert!(related.iter().any(|r| r.path == Path::new("src/lib.rs")));
    }

    #[test]
    fn test_build_graph() {
        let mut tracker = RelationshipTracker::new();
        let rel = create_test_relationship();
        tracker.add_relationship(rel).unwrap();

        let graph = tracker.build_graph().unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.metadata.node_count, 2);
        assert_eq!(graph.metadata.edge_count, 1);
    }

    #[test]
    fn test_delete_relationship() {
        let mut tracker = RelationshipTracker::new();
        let rel = create_test_relationship();
        tracker.add_relationship(rel).unwrap();

        assert!(tracker.get_relationship("test-rel-1").is_some());

        tracker.delete_relationship("test-rel-1").unwrap();

        assert!(tracker.get_relationship("test-rel-1").is_none());
    }

    #[test]
    fn test_record_coedit() {
        let mut tracker = RelationshipTracker::new();

        let files = vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")];

        // Record multiple coedits
        for _ in 0..5 {
            tracker.record_coedit(&files).unwrap();
        }

        // Finalize should create a relationship
        tracker.finalize_session().unwrap();

        // Should have a co-change relationship
        let relationships = tracker.get_relationships_by_type(RelationType::FrequentCochange);
        assert!(!relationships.is_empty());
    }

    #[test]
    fn test_get_hub_files() {
        let mut tracker = RelationshipTracker::new();

        // Create a hub file (main.rs) connected to multiple others
        for i in 0..5 {
            let rel = FileRelationship::new(
                format!("rel-{}", i),
                vec![
                    PathBuf::from("src/main.rs"),
                    PathBuf::from(format!("src/module{}.rs", i)),
                ],
                RelationType::ImportsDependency,
                "Import relationship".to_string(),
            );
            tracker.add_relationship(rel).unwrap();
        }

        let hubs = tracker.get_hub_files(3);

        assert!(!hubs.is_empty());
        assert_eq!(hubs[0].0, PathBuf::from("src/main.rs"));
        assert_eq!(hubs[0].1, 5);
    }
}
