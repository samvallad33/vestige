//! # Memory Reconsolidation
//!
//! Implements Nader's reconsolidation theory: "Memories are rebuilt every time they're recalled."
//!
//! When a memory is accessed, it enters a "labile" (modifiable) state. During this window:
//! - New context can be integrated
//! - Connections can be strengthened
//! - Related information can be linked
//! - Emotional associations can be updated
//!
//! After the labile window closes, the memory is "reconsolidated" with any modifications.
//!
//! ## Scientific Background
//!
//! Based on Karim Nader's groundbreaking 2000 research showing that:
//! - Retrieved memories become temporarily unstable
//! - Protein synthesis is required to re-store them
//! - This window allows memories to be updated or modified
//! - Memories are not static recordings but dynamic reconstructions
//!
//! ## Example
//!
//! ```rust,ignore
//! use vestige_core::advanced::reconsolidation::ReconsolidationManager;
//!
//! let mut manager = ReconsolidationManager::new();
//!
//! // Memory becomes labile on access
//! manager.mark_labile("memory-123");
//!
//! // Check if memory is still modifiable
//! if manager.is_labile("memory-123") {
//!     // Add new context during labile window
//!     manager.apply_modification("memory-123", Modification::AddContext {
//!         context: "Related to project X".to_string(),
//!     });
//! }
//!
//! // Later: reconsolidate with modifications
//! let result = manager.reconsolidate("memory-123");
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default labile window duration (5 minutes)
const DEFAULT_LABILE_WINDOW_SECS: i64 = 300;

/// Maximum modifications per memory during labile window
const MAX_MODIFICATIONS_PER_WINDOW: usize = 10;

/// How long to keep retrieval history
const RETRIEVAL_HISTORY_DAYS: i64 = 30;

// ============================================================================
// LABILE STATE
// ============================================================================

/// State of a memory that has become labile (modifiable) after access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabileState {
    /// Memory ID
    pub memory_id: String,
    /// When the memory was accessed (became labile)
    pub accessed_at: DateTime<Utc>,
    /// Snapshot of the original memory state
    pub original_state: MemorySnapshot,
    /// Modifications applied during labile window
    pub modifications: Vec<Modification>,
    /// Access context (what triggered the retrieval)
    pub access_context: Option<AccessContext>,
    /// Whether this memory has been reconsolidated
    pub reconsolidated: bool,
}

impl LabileState {
    /// Create a new labile state for a memory
    pub fn new(memory_id: String, original: MemorySnapshot) -> Self {
        Self {
            memory_id,
            accessed_at: Utc::now(),
            original_state: original,
            modifications: Vec::new(),
            access_context: None,
            reconsolidated: false,
        }
    }

    /// Check if still within labile window
    pub fn is_within_window(&self, window: Duration) -> bool {
        Utc::now() - self.accessed_at < window
    }

    /// Add a modification
    pub fn add_modification(&mut self, modification: Modification) -> bool {
        if self.modifications.len() < MAX_MODIFICATIONS_PER_WINDOW {
            self.modifications.push(modification);
            true
        } else {
            false
        }
    }

    /// Set access context
    pub fn with_context(mut self, context: AccessContext) -> Self {
        self.access_context = Some(context);
        self
    }
}

/// Snapshot of a memory's state before modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Memory content at time of access
    pub content: String,
    /// Tags at time of access
    pub tags: Vec<String>,
    /// Retention strength at time of access
    pub retention_strength: f64,
    /// Storage strength at time of access
    pub storage_strength: f64,
    /// Retrieval strength at time of access
    pub retrieval_strength: f64,
    /// Connection IDs at time of access
    pub connection_ids: Vec<String>,
    /// Snapshot timestamp
    pub captured_at: DateTime<Utc>,
}

impl MemorySnapshot {
    /// Create a snapshot from memory data
    pub fn capture(
        content: String,
        tags: Vec<String>,
        retention_strength: f64,
        storage_strength: f64,
        retrieval_strength: f64,
        connection_ids: Vec<String>,
    ) -> Self {
        Self {
            content,
            tags,
            retention_strength,
            storage_strength,
            retrieval_strength,
            connection_ids,
            captured_at: Utc::now(),
        }
    }
}

// ============================================================================
// MODIFICATIONS
// ============================================================================

/// Types of modifications that can be applied during the labile window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modification {
    /// Add contextual information
    AddContext {
        /// New context to add
        context: String,
    },
    /// Strengthen connection to another memory
    StrengthenConnection {
        /// Connected memory ID
        target_memory_id: String,
        /// Strength boost (0.0 to 1.0)
        boost: f64,
    },
    /// Add a new tag
    AddTag {
        /// Tag to add
        tag: String,
    },
    /// Remove a tag
    RemoveTag {
        /// Tag to remove
        tag: String,
    },
    /// Update emotional association
    UpdateEmotion {
        /// New sentiment score (-1.0 to 1.0)
        sentiment_score: Option<f64>,
        /// New sentiment magnitude (0.0 to 1.0)
        sentiment_magnitude: Option<f64>,
    },
    /// Link to related memory
    LinkMemory {
        /// Memory to link to
        related_memory_id: String,
        /// Type of relationship
        relationship: RelationshipType,
    },
    /// Correct or update content
    UpdateContent {
        /// Updated content (or None to keep original)
        new_content: Option<String>,
        /// Whether this is a correction
        is_correction: bool,
    },
    /// Add source/provenance information
    AddSource {
        /// Source information
        source: String,
    },
    /// Boost retrieval strength (successful recall)
    BoostRetrieval {
        /// Boost amount
        boost: f64,
    },
}

/// Types of relationships between memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    /// Memory A supports/reinforces Memory B
    Supports,
    /// Memory A contradicts Memory B
    Contradicts,
    /// Memory A is an elaboration of Memory B
    Elaborates,
    /// Memory A is a generalization of Memory B
    Generalizes,
    /// Memory A is a specific example of Memory B
    Exemplifies,
    /// Memory A is temporally related to Memory B
    TemporallyRelated,
    /// Memory A caused Memory B
    Causes,
    /// General semantic similarity
    SimilarTo,
}

impl Modification {
    /// Get a description of this modification
    pub fn description(&self) -> String {
        match self {
            Self::AddContext { context } => format!("Add context: {}", truncate(context, 50)),
            Self::StrengthenConnection {
                target_memory_id,
                boost,
            } => format!(
                "Strengthen connection to {} by {:.2}",
                target_memory_id, boost
            ),
            Self::AddTag { tag } => format!("Add tag: {}", tag),
            Self::RemoveTag { tag } => format!("Remove tag: {}", tag),
            Self::UpdateEmotion {
                sentiment_score,
                sentiment_magnitude,
            } => format!(
                "Update emotion: score={:?}, magnitude={:?}",
                sentiment_score, sentiment_magnitude
            ),
            Self::LinkMemory {
                related_memory_id,
                relationship,
            } => format!("Link to {} ({:?})", related_memory_id, relationship),
            Self::UpdateContent { is_correction, .. } => {
                format!("Update content (correction={})", is_correction)
            }
            Self::AddSource { source } => format!("Add source: {}", truncate(source, 50)),
            Self::BoostRetrieval { boost } => format!("Boost retrieval by {:.2}", boost),
        }
    }
}

// ============================================================================
// ACCESS CONTEXT
// ============================================================================

/// Context about how/why a memory was accessed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessContext {
    /// What triggered the retrieval
    pub trigger: AccessTrigger,
    /// Search query if applicable
    pub query: Option<String>,
    /// Other memories retrieved in same session
    pub co_retrieved: Vec<String>,
    /// Session or task identifier
    pub session_id: Option<String>,
}

/// What triggered memory retrieval
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccessTrigger {
    /// Direct search by user
    Search,
    /// Automatic retrieval (speculative, context-based)
    Automatic,
    /// Consolidation replay
    ConsolidationReplay,
    /// Linked from another memory
    LinkedRetrieval,
    /// User explicitly accessed
    DirectAccess,
    /// Review/study session
    Review,
}

// ============================================================================
// RECONSOLIDATED MEMORY
// ============================================================================

/// Result of reconsolidating a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconsolidatedMemory {
    /// Memory ID
    pub memory_id: String,
    /// When reconsolidation occurred
    pub reconsolidated_at: DateTime<Utc>,
    /// Duration of labile window
    pub labile_duration: Duration,
    /// Modifications that were applied
    pub applied_modifications: Vec<AppliedModification>,
    /// Whether any modifications were made
    pub was_modified: bool,
    /// Summary of changes
    pub change_summary: ChangeSummary,
    /// New retrieval count
    pub retrieval_count: u32,
}

/// A modification that was successfully applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedModification {
    /// The modification
    pub modification: Modification,
    /// When it was applied
    pub applied_at: DateTime<Utc>,
    /// Whether it succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Summary of changes made during reconsolidation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeSummary {
    /// Number of tags added
    pub tags_added: usize,
    /// Number of tags removed
    pub tags_removed: usize,
    /// Number of connections strengthened
    pub connections_strengthened: usize,
    /// Number of new links created
    pub links_created: usize,
    /// Whether content was updated
    pub content_updated: bool,
    /// Whether emotion was updated
    pub emotion_updated: bool,
    /// Total retrieval boost applied
    pub retrieval_boost: f64,
}

impl ChangeSummary {
    /// Check if any changes were made
    pub fn has_changes(&self) -> bool {
        self.tags_added > 0
            || self.tags_removed > 0
            || self.connections_strengthened > 0
            || self.links_created > 0
            || self.content_updated
            || self.emotion_updated
            || self.retrieval_boost > 0.0
    }
}

// ============================================================================
// RETRIEVAL HISTORY
// ============================================================================

/// Record of a memory retrieval event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRecord {
    /// Memory ID
    pub memory_id: String,
    /// When retrieval occurred
    pub retrieved_at: DateTime<Utc>,
    /// Access context
    pub context: Option<AccessContext>,
    /// Whether memory was modified during labile window
    pub was_modified: bool,
    /// Retrieval strength at time of access
    pub retrieval_strength_at_access: f64,
}

// ============================================================================
// RECONSOLIDATION MANAGER
// ============================================================================

/// Manages memory reconsolidation
///
/// Tracks labile memories and applies modifications during the labile window.
/// Inspired by Nader's research on memory reconsolidation.
#[derive(Debug)]
pub struct ReconsolidationManager {
    /// Currently labile memories
    labile_memories: HashMap<String, LabileState>,
    /// Duration of labile window
    labile_window: Duration,
    /// Retrieval history
    retrieval_history: Arc<RwLock<Vec<RetrievalRecord>>>,
    /// Reconsolidation statistics
    stats: ReconsolidationStats,
    /// Whether reconsolidation is enabled
    enabled: bool,
}

impl Default for ReconsolidationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ReconsolidationManager {
    /// Create a new reconsolidation manager
    pub fn new() -> Self {
        Self {
            labile_memories: HashMap::new(),
            labile_window: Duration::seconds(DEFAULT_LABILE_WINDOW_SECS),
            retrieval_history: Arc::new(RwLock::new(Vec::new())),
            stats: ReconsolidationStats::default(),
            enabled: true,
        }
    }

    /// Create with custom labile window
    pub fn with_window(window_seconds: i64) -> Self {
        let mut manager = Self::new();
        manager.labile_window = Duration::seconds(window_seconds);
        manager
    }

    /// Enable or disable reconsolidation
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if reconsolidation is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Mark a memory as labile (accessed)
    ///
    /// Call this when a memory is retrieved. The memory will be modifiable
    /// during the labile window.
    pub fn mark_labile(&mut self, memory_id: &str, snapshot: MemorySnapshot) {
        if !self.enabled {
            return;
        }

        let state = LabileState::new(memory_id.to_string(), snapshot);
        self.labile_memories.insert(memory_id.to_string(), state);
        self.stats.total_marked_labile += 1;
    }

    /// Mark a memory as labile with context
    pub fn mark_labile_with_context(
        &mut self,
        memory_id: &str,
        snapshot: MemorySnapshot,
        context: AccessContext,
    ) {
        if !self.enabled {
            return;
        }

        let state = LabileState::new(memory_id.to_string(), snapshot).with_context(context);
        self.labile_memories.insert(memory_id.to_string(), state);
        self.stats.total_marked_labile += 1;
    }

    /// Check if a memory is currently labile (modifiable)
    pub fn is_labile(&self, memory_id: &str) -> bool {
        self.labile_memories
            .get(memory_id)
            .map(|state| state.is_within_window(self.labile_window))
            .unwrap_or(false)
    }

    /// Get the labile state for a memory
    pub fn get_labile_state(&self, memory_id: &str) -> Option<&LabileState> {
        self.labile_memories
            .get(memory_id)
            .filter(|state| state.is_within_window(self.labile_window))
    }

    /// Get remaining labile window time
    pub fn remaining_labile_time(&self, memory_id: &str) -> Option<Duration> {
        self.labile_memories.get(memory_id).and_then(|state| {
            let elapsed = Utc::now() - state.accessed_at;
            if elapsed < self.labile_window {
                Some(self.labile_window - elapsed)
            } else {
                None
            }
        })
    }

    /// Apply a modification to a labile memory
    ///
    /// Returns true if the modification was applied, false if the memory
    /// is not labile or the modification limit was reached.
    pub fn apply_modification(&mut self, memory_id: &str, modification: Modification) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(state) = self.labile_memories.get_mut(memory_id)
            && state.is_within_window(self.labile_window)
        {
            let success = state.add_modification(modification);
            if success {
                self.stats.total_modifications += 1;
            }
            return success;
        }
        false
    }

    /// Apply multiple modifications at once
    pub fn apply_modifications(
        &mut self,
        memory_id: &str,
        modifications: Vec<Modification>,
    ) -> usize {
        let mut applied = 0;
        for modification in modifications {
            if self.apply_modification(memory_id, modification) {
                applied += 1;
            }
        }
        applied
    }

    /// Reconsolidate a memory (finalize modifications)
    ///
    /// This should be called when:
    /// - The labile window expires
    /// - Explicitly by the system when appropriate
    ///
    /// Returns the reconsolidation result with all applied modifications.
    pub fn reconsolidate(&mut self, memory_id: &str) -> Option<ReconsolidatedMemory> {
        // remove() already guarantees idempotency: a second call finds no entry
        // and returns None via `?`. The `reconsolidated` flag was never set to
        // true anywhere, so the guard below was dead — but keep it as a correct,
        // explicit belt-and-suspenders in case the entry is ever retained.
        let state = self.labile_memories.remove(memory_id)?;

        if state.reconsolidated {
            return None;
        }

        let labile_duration = Utc::now() - state.accessed_at;

        // Build change summary
        let mut change_summary = ChangeSummary::default();
        let mut applied_modifications = Vec::new();

        for modification in &state.modifications {
            let applied = AppliedModification {
                modification: modification.clone(),
                applied_at: Utc::now(),
                success: true,
                error: None,
            };

            // Update summary based on modification type
            match modification {
                Modification::AddTag { .. } => change_summary.tags_added += 1,
                Modification::RemoveTag { .. } => change_summary.tags_removed += 1,
                Modification::StrengthenConnection { .. } => {
                    change_summary.connections_strengthened += 1
                }
                Modification::LinkMemory { .. } => change_summary.links_created += 1,
                Modification::UpdateContent { .. } => change_summary.content_updated = true,
                Modification::UpdateEmotion { .. } => change_summary.emotion_updated = true,
                Modification::BoostRetrieval { boost } => change_summary.retrieval_boost += boost,
                _ => {}
            }

            applied_modifications.push(applied);
        }

        let was_modified = change_summary.has_changes();

        // Record retrieval in history
        self.record_retrieval(RetrievalRecord {
            memory_id: memory_id.to_string(),
            retrieved_at: state.accessed_at,
            context: state.access_context,
            was_modified,
            retrieval_strength_at_access: state.original_state.retrieval_strength,
        });

        self.stats.total_reconsolidated += 1;
        if was_modified {
            self.stats.total_modified += 1;
        }

        Some(ReconsolidatedMemory {
            memory_id: memory_id.to_string(),
            reconsolidated_at: Utc::now(),
            labile_duration,
            applied_modifications,
            was_modified,
            change_summary,
            retrieval_count: self.get_retrieval_count(memory_id),
        })
    }

    /// Force reconsolidation of all expired labile memories
    pub fn reconsolidate_expired(&mut self) -> Vec<ReconsolidatedMemory> {
        let expired_ids: Vec<_> = self
            .labile_memories
            .iter()
            .filter(|(_, state)| !state.is_within_window(self.labile_window))
            .map(|(id, _)| id.clone())
            .collect();

        expired_ids
            .into_iter()
            .filter_map(|id| self.reconsolidate(&id))
            .collect()
    }

    /// Get all currently labile memory IDs
    pub fn get_labile_memory_ids(&self) -> Vec<String> {
        self.labile_memories
            .iter()
            .filter(|(_, state)| state.is_within_window(self.labile_window))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Record a retrieval event
    fn record_retrieval(&self, record: RetrievalRecord) {
        if let Ok(mut history) = self.retrieval_history.write() {
            history.push(record);

            // Trim old records
            let cutoff = Utc::now() - Duration::days(RETRIEVAL_HISTORY_DAYS);
            history.retain(|r| r.retrieved_at >= cutoff);
        }
    }

    /// Get retrieval count for a memory
    pub fn get_retrieval_count(&self, memory_id: &str) -> u32 {
        self.retrieval_history
            .read()
            .map(|history| history.iter().filter(|r| r.memory_id == memory_id).count() as u32)
            .unwrap_or(0)
    }

    /// Get retrieval history for a memory
    pub fn get_retrieval_history(&self, memory_id: &str) -> Vec<RetrievalRecord> {
        self.retrieval_history
            .read()
            .map(|history| {
                history
                    .iter()
                    .filter(|r| r.memory_id == memory_id)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get most recently retrieved memories
    pub fn get_recent_retrievals(&self, limit: usize) -> Vec<RetrievalRecord> {
        self.retrieval_history
            .read()
            .map(|history| {
                let mut recent: Vec<_> = history.iter().cloned().collect();
                recent.sort_by_key(|b| std::cmp::Reverse(b.retrieved_at));
                recent.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    /// Get memories frequently retrieved together
    pub fn get_co_retrieved_memories(&self, memory_id: &str) -> HashMap<String, usize> {
        let mut co_retrieved = HashMap::new();

        if let Ok(history) = self.retrieval_history.read() {
            for record in history.iter() {
                if record.memory_id == memory_id
                    && let Some(context) = &record.context
                {
                    for co_id in &context.co_retrieved {
                        if co_id != memory_id {
                            *co_retrieved.entry(co_id.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        co_retrieved
    }

    /// Get reconsolidation statistics
    pub fn get_stats(&self) -> &ReconsolidationStats {
        &self.stats
    }

    /// Get current labile window duration
    pub fn get_labile_window(&self) -> Duration {
        self.labile_window
    }

    /// Set labile window duration
    pub fn set_labile_window(&mut self, window: Duration) {
        self.labile_window = window;
    }

    /// Clear all labile states (for cleanup)
    pub fn clear_labile_states(&mut self) {
        self.labile_memories.clear();
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics about reconsolidation operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReconsolidationStats {
    /// Total memories marked labile
    pub total_marked_labile: usize,
    /// Total memories reconsolidated
    pub total_reconsolidated: usize,
    /// Total memories modified during labile window
    pub total_modified: usize,
    /// Total modifications applied
    pub total_modifications: usize,
}

impl ReconsolidationStats {
    /// Get modification rate (modifications per labile memory)
    pub fn modification_rate(&self) -> f64 {
        if self.total_marked_labile > 0 {
            self.total_modifications as f64 / self.total_marked_labile as f64
        } else {
            0.0
        }
    }

    /// Get modified rate (% of labile memories that were modified)
    pub fn modified_rate(&self) -> f64 {
        if self.total_reconsolidated > 0 {
            self.total_modified as f64 / self.total_reconsolidated as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Truncate string for display
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..s.floor_char_boundary(max_len)]
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot() -> MemorySnapshot {
        MemorySnapshot::capture(
            "Test content".to_string(),
            vec!["test".to_string()],
            0.8,
            5.0,
            0.9,
            vec![],
        )
    }

    #[test]
    fn test_manager_new() {
        let manager = ReconsolidationManager::new();
        assert!(manager.is_enabled());
        assert_eq!(
            manager.get_labile_window(),
            Duration::seconds(DEFAULT_LABILE_WINDOW_SECS)
        );
    }

    #[test]
    fn test_mark_labile() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        manager.mark_labile("mem-1", snapshot);

        assert!(manager.is_labile("mem-1"));
        assert!(!manager.is_labile("mem-2")); // Not marked
    }

    #[test]
    fn test_apply_modification() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        manager.mark_labile("mem-1", snapshot);

        let success = manager.apply_modification(
            "mem-1",
            Modification::AddTag {
                tag: "new-tag".to_string(),
            },
        );

        assert!(success);
        assert_eq!(manager.get_stats().total_modifications, 1);
    }

    #[test]
    fn test_apply_modification_not_labile() {
        let mut manager = ReconsolidationManager::new();

        // Try to modify a memory that's not labile
        let success = manager.apply_modification(
            "mem-1",
            Modification::AddTag {
                tag: "new-tag".to_string(),
            },
        );

        assert!(!success);
    }

    #[test]
    fn test_reconsolidate() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        manager.mark_labile("mem-1", snapshot);
        manager.apply_modification(
            "mem-1",
            Modification::AddTag {
                tag: "new-tag".to_string(),
            },
        );

        let result = manager.reconsolidate("mem-1");

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.was_modified);
        assert_eq!(result.change_summary.tags_added, 1);
    }

    #[test]
    fn test_remaining_labile_time() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        manager.mark_labile("mem-1", snapshot);

        let remaining = manager.remaining_labile_time("mem-1");
        assert!(remaining.is_some());
        assert!(remaining.unwrap() > Duration::zero());
    }

    #[test]
    fn test_modification_types() {
        let modifications = vec![
            Modification::AddContext {
                context: "test".to_string(),
            },
            Modification::StrengthenConnection {
                target_memory_id: "other".to_string(),
                boost: 0.5,
            },
            Modification::AddTag {
                tag: "tag".to_string(),
            },
            Modification::RemoveTag {
                tag: "old".to_string(),
            },
            Modification::UpdateEmotion {
                sentiment_score: Some(0.5),
                sentiment_magnitude: None,
            },
            Modification::LinkMemory {
                related_memory_id: "rel".to_string(),
                relationship: RelationshipType::Supports,
            },
            Modification::UpdateContent {
                new_content: None,
                is_correction: true,
            },
            Modification::AddSource {
                source: "web".to_string(),
            },
            Modification::BoostRetrieval { boost: 0.1 },
        ];

        for modification in modifications {
            assert!(!modification.description().is_empty());
        }
    }

    #[test]
    fn test_relationship_types() {
        let relationships = [
            RelationshipType::Supports,
            RelationshipType::Contradicts,
            RelationshipType::Elaborates,
            RelationshipType::Generalizes,
            RelationshipType::Exemplifies,
            RelationshipType::TemporallyRelated,
            RelationshipType::Causes,
            RelationshipType::SimilarTo,
        ];

        // Just ensure all variants exist
        assert_eq!(relationships.len(), 8);
    }

    #[test]
    fn test_change_summary() {
        let mut summary = ChangeSummary::default();
        assert!(!summary.has_changes());

        summary.tags_added = 1;
        assert!(summary.has_changes());
    }

    #[test]
    fn test_labile_state() {
        let snapshot = make_snapshot();
        let mut state = LabileState::new("mem-1".to_string(), snapshot);

        assert!(state.is_within_window(Duration::seconds(300)));
        assert!(!state.reconsolidated);

        // Add modifications
        for i in 0..MAX_MODIFICATIONS_PER_WINDOW {
            assert!(state.add_modification(Modification::AddTag {
                tag: format!("tag-{}", i),
            }));
        }

        // Should fail now (limit reached)
        assert!(!state.add_modification(Modification::AddTag {
            tag: "overflow".to_string(),
        }));
    }

    #[test]
    fn test_retrieval_history() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        // Mark and reconsolidate multiple times
        for _ in 0..3 {
            manager.mark_labile("mem-1", snapshot.clone());
            manager.reconsolidate("mem-1");
        }

        assert_eq!(manager.get_retrieval_count("mem-1"), 3);
        assert_eq!(manager.get_retrieval_history("mem-1").len(), 3);
    }

    #[test]
    fn test_stats() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();

        manager.mark_labile("mem-1", snapshot.clone());
        manager.apply_modification(
            "mem-1",
            Modification::AddTag {
                tag: "t".to_string(),
            },
        );
        manager.reconsolidate("mem-1");

        let stats = manager.get_stats();
        assert_eq!(stats.total_marked_labile, 1);
        assert_eq!(stats.total_reconsolidated, 1);
        assert_eq!(stats.total_modified, 1);
        assert_eq!(stats.total_modifications, 1);
    }

    #[test]
    fn test_disabled_manager() {
        let mut manager = ReconsolidationManager::new();
        manager.set_enabled(false);

        let snapshot = make_snapshot();
        manager.mark_labile("mem-1", snapshot);

        // Should not be labile when disabled
        assert!(!manager.is_labile("mem-1"));
    }

    #[test]
    fn test_access_context() {
        let mut manager = ReconsolidationManager::new();
        let snapshot = make_snapshot();
        let context = AccessContext {
            trigger: AccessTrigger::Search,
            query: Some("test query".to_string()),
            co_retrieved: vec!["mem-2".to_string(), "mem-3".to_string()],
            session_id: Some("session-1".to_string()),
        };

        manager.mark_labile_with_context("mem-1", snapshot, context);

        let state = manager.get_labile_state("mem-1");
        assert!(state.is_some());
        assert!(state.unwrap().access_context.is_some());
    }

    #[test]
    fn test_get_labile_memory_ids() {
        let mut manager = ReconsolidationManager::new();

        manager.mark_labile("mem-1", make_snapshot());
        manager.mark_labile("mem-2", make_snapshot());
        manager.mark_labile("mem-3", make_snapshot());

        let ids = manager.get_labile_memory_ids();
        assert_eq!(ids.len(), 3);
    }
}
