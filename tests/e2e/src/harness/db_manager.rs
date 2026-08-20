//! Test Database Manager
//!
//! Provides isolated database instances for testing:
//! - Temporary databases that are automatically cleaned up
//! - Pre-seeded databases with test data
//! - Database snapshots and restoration
//! - Concurrent test isolation

use std::path::PathBuf;
use tempfile::TempDir;
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
    valid_from: Option<chrono::DateTime<chrono::Utc>>,
    valid_until: Option<chrono::DateTime<chrono::Utc>>,
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

/// Manager for test databases
///
/// Creates isolated database instances for each test to prevent interference.
/// Automatically cleans up temporary databases when dropped.
///
/// # Example
///
/// ```rust,ignore
/// let mut db = TestDatabaseManager::new_temp();
///
/// // Use the storage
/// db.storage.ingest(IngestInput { ... });
///
/// // Database is automatically deleted when `db` goes out of scope
/// ```
pub struct TestDatabaseManager {
    /// The storage instance
    pub storage: Storage,
    /// Temporary directory (kept alive to prevent premature deletion)
    _temp_dir: Option<TempDir>,
    /// Path to the database file
    db_path: PathBuf,
    /// Snapshot data for restore operations
    snapshot: Option<Vec<KnowledgeNode>>,
}

impl TestDatabaseManager {
    /// Create a new test database in a temporary directory
    ///
    /// The database is automatically deleted when the manager is dropped.
    pub fn new_temp() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let db_path = temp_dir.path().join("test_vestige.db");

        let storage = Storage::new(Some(db_path.clone())).expect("Failed to create test storage");

        Self {
            storage,
            _temp_dir: Some(temp_dir),
            db_path,
            snapshot: None,
        }
    }

    /// Create a test database at a specific path
    ///
    /// The database is NOT automatically deleted.
    pub fn new_at_path(path: PathBuf) -> Self {
        let storage = Storage::new(Some(path.clone())).expect("Failed to create test storage");

        Self {
            storage,
            _temp_dir: None,
            db_path: path,
            snapshot: None,
        }
    }

    /// Get the database path
    pub fn path(&self) -> &PathBuf {
        &self.db_path
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.storage
            .get_stats()
            .map(|s| s.total_nodes == 0)
            .unwrap_or(true)
    }

    /// Get the number of nodes in the database
    pub fn node_count(&self) -> i64 {
        self.storage.get_stats().map(|s| s.total_nodes).unwrap_or(0)
    }

    // ========================================================================
    // SEEDING METHODS
    // ========================================================================

    /// Seed the database with a specified number of test nodes
    pub fn seed_nodes(&mut self, count: usize) -> Vec<String> {
        let mut ids = Vec::with_capacity(count);

        for i in 0..count {
            let input = make_ingest_input(
                format!("Test memory content {}", i),
                "fact".to_string(),
                vec![format!("test-{}", i % 5)],
                0.0,
                0.0,
                None,
                None,
                None,
            );

            if let Ok(node) = self.storage.ingest(input) {
                ids.push(node.id);
            }
        }

        ids
    }

    /// Seed with diverse node types
    pub fn seed_diverse(&mut self, count_per_type: usize) -> Vec<String> {
        let types = ["fact", "concept", "procedure", "event", "code"];
        let mut ids = Vec::with_capacity(count_per_type * types.len());

        for node_type in types {
            for i in 0..count_per_type {
                let input = make_ingest_input(
                    format!("Test {} content {}", node_type, i),
                    node_type.to_string(),
                    vec![node_type.to_string()],
                    0.0,
                    0.0,
                    None,
                    None,
                    None,
                );

                if let Ok(node) = self.storage.ingest(input) {
                    ids.push(node.id);
                }
            }
        }

        ids
    }

    /// Seed with nodes having various retention states
    pub fn seed_with_retention_states(&mut self) -> Vec<String> {
        let mut ids = Vec::new();

        // New node (never reviewed)
        let input = make_ingest_input(
            "New memory - never reviewed".to_string(),
            "fact".to_string(),
            vec!["new".to_string()],
            0.0,
            0.0,
            None,
            None,
            None,
        );
        if let Ok(node) = self.storage.ingest(input) {
            ids.push(node.id);
        }

        // Well-learned node (multiple good reviews)
        let input = make_ingest_input(
            "Well-learned memory - reviewed multiple times".to_string(),
            "fact".to_string(),
            vec!["learned".to_string()],
            0.0,
            0.0,
            None,
            None,
            None,
        );
        if let Ok(node) = self.storage.ingest(input) {
            let _ = self.storage.mark_reviewed(&node.id, Rating::Good);
            let _ = self.storage.mark_reviewed(&node.id, Rating::Good);
            let _ = self.storage.mark_reviewed(&node.id, Rating::Easy);
            ids.push(node.id);
        }

        // Struggling node (multiple lapses)
        let input = make_ingest_input(
            "Struggling memory - has lapses".to_string(),
            "fact".to_string(),
            vec!["struggling".to_string()],
            0.0,
            0.0,
            None,
            None,
            None,
        );
        if let Ok(node) = self.storage.ingest(input) {
            let _ = self.storage.mark_reviewed(&node.id, Rating::Again);
            let _ = self.storage.mark_reviewed(&node.id, Rating::Hard);
            let _ = self.storage.mark_reviewed(&node.id, Rating::Again);
            ids.push(node.id);
        }

        ids
    }

    /// Seed with emotional memories (different sentiment magnitudes)
    pub fn seed_emotional(&mut self, count: usize) -> Vec<String> {
        let mut ids = Vec::with_capacity(count);

        for i in 0..count {
            let magnitude = (i as f64) / (count as f64);
            let input = make_ingest_input(
                format!("Emotional memory with magnitude {:.2}", magnitude),
                "event".to_string(),
                vec!["emotional".to_string()],
                if i % 2 == 0 { 0.8 } else { -0.8 },
                magnitude,
                None,
                None,
                None,
            );

            if let Ok(node) = self.storage.ingest(input) {
                ids.push(node.id);
            }
        }

        ids
    }

    // ========================================================================
    // SNAPSHOT/RESTORE
    // ========================================================================

    /// Take a snapshot of current database state
    pub fn take_snapshot(&mut self) {
        let nodes = self.storage.get_all_nodes(10000, 0).unwrap_or_default();
        self.snapshot = Some(nodes);
    }

    /// Restore from the last snapshot
    ///
    /// Note: This clears the database and re-inserts all nodes from snapshot.
    /// IDs will NOT be preserved (new UUIDs are generated).
    pub fn restore_snapshot(&mut self) -> bool {
        if let Some(nodes) = self.snapshot.take() {
            // Clear current data by recreating storage
            // Delete the database file first
            let _ = std::fs::remove_file(&self.db_path);
            self.storage = Storage::new(Some(self.db_path.clone()))
                .expect("Failed to recreate storage for restore");

            // Re-insert nodes
            for node in nodes {
                let input = make_ingest_input(
                    node.content,
                    node.node_type,
                    node.tags,
                    node.sentiment_score,
                    node.sentiment_magnitude,
                    node.source,
                    node.valid_from,
                    node.valid_until,
                );
                let _ = self.storage.ingest(input);
            }

            true
        } else {
            false
        }
    }

    /// Check if a snapshot exists
    pub fn has_snapshot(&self) -> bool {
        self.snapshot.is_some()
    }

    // ========================================================================
    // CLEANUP
    // ========================================================================

    /// Clear all data from the database
    pub fn clear(&mut self) {
        // Get all node IDs and delete them
        if let Ok(nodes) = self.storage.get_all_nodes(10000, 0) {
            for node in nodes {
                let _ = self.storage.delete_node(&node.id);
            }
        }
    }

    /// Recreate the database (useful for testing migrations)
    pub fn recreate(&mut self) {
        // Delete the database file
        let _ = std::fs::remove_file(&self.db_path);

        // Recreate storage
        self.storage =
            Storage::new(Some(self.db_path.clone())).expect("Failed to recreate storage");
    }
}

impl Drop for TestDatabaseManager {
    fn drop(&mut self) {
        // Storage is dropped automatically
        // TempDir (if Some) will clean up the temp directory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_database_creation() {
        let db = TestDatabaseManager::new_temp();
        assert!(db.is_empty());
        assert!(db.path().exists());
    }

    #[test]
    fn test_seed_nodes() {
        let mut db = TestDatabaseManager::new_temp();
        let ids = db.seed_nodes(10);

        assert_eq!(ids.len(), 10);
        assert_eq!(db.node_count(), 10);
    }

    #[test]
    fn test_seed_diverse() {
        let mut db = TestDatabaseManager::new_temp();
        let ids = db.seed_diverse(3);

        // 5 types * 3 each = 15
        assert_eq!(ids.len(), 15);
        assert_eq!(db.node_count(), 15);
    }

    #[test]
    fn test_clear_database() {
        let mut db = TestDatabaseManager::new_temp();
        db.seed_nodes(5);
        assert_eq!(db.node_count(), 5);

        db.clear();
        assert!(db.is_empty());
    }

    #[test]
    fn test_snapshot_restore() {
        let mut db = TestDatabaseManager::new_temp();
        db.seed_nodes(5);

        db.take_snapshot();
        assert!(db.has_snapshot());

        db.clear();
        assert!(db.is_empty());

        db.restore_snapshot();
        assert_eq!(db.node_count(), 5);
    }
}
