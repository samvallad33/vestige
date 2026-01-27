//! # Memory Dreams (Enhanced Consolidation)
//!
//! Enhanced sleep-inspired consolidation that creates NEW insights from
//! existing memories. Like how the brain consolidates and generates novel
//! connections during sleep, Memory Dreams finds hidden patterns and
//! synthesizes new knowledge.
//!
//! ## Dream Cycle (Sleep Stages)
//!
//! 1. **Stage 1 - Replay**: Replay recent memories in sequence
//! 2. **Stage 2 - Cross-reference**: Find connections with existing knowledge
//! 3. **Stage 3 - Strengthen**: Reinforce connections that fire together
//! 4. **Stage 4 - Prune**: Remove weak connections not reactivated
//! 5. **Stage 5 - Transfer**: Move consolidated from episodic to semantic
//!
//! ## Consolidation Scheduler
//!
//! Automatically detects low-activity periods and triggers consolidation:
//! - Tracks user activity patterns
//! - Runs during detected idle periods
//! - Configurable consolidation interval
//!
//! ## Memory Replay
//!
//! Simulates hippocampal replay during sleep:
//! - Replays recent memory sequences
//! - Tests synthetic pattern combinations
//! - Discovers emergent patterns
//!
//! ## Novelty Detection
//!
//! The system measures how "new" an insight is based on:
//! - Distance from existing memories in embedding space
//! - Uniqueness of the combination that produced it
//! - Information gain over source memories
//!
//! ## Example
//!
//! ```rust,ignore
//! use vestige_core::advanced::dreams::{ConsolidationScheduler, MemoryDreamer};
//!
//! // Create scheduler with activity tracking
//! let mut scheduler = ConsolidationScheduler::new();
//!
//! // Check if consolidation should run (low activity detected)
//! if scheduler.should_consolidate() {
//!     let report = scheduler.run_consolidation_cycle(&storage).await;
//!     println!("Consolidation complete: {:?}", report);
//! }
//!
//! // Or run dream cycle directly
//! let dreamer = MemoryDreamer::new();
//! let result = dreamer.dream(&memories).await;
//!
//! println!("Found {} new connections", result.new_connections_found);
//! println!("Generated {} insights", result.insights_generated.len());
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use uuid::Uuid;

/// Minimum similarity for connection discovery
const MIN_SIMILARITY_FOR_CONNECTION: f64 = 0.5;

/// Maximum insights to generate per dream cycle
const MAX_INSIGHTS_PER_DREAM: usize = 10;

/// Minimum novelty score for insights
const MIN_NOVELTY_SCORE: f64 = 0.3;

/// Minimum memories needed for insight generation
const MIN_MEMORIES_FOR_INSIGHT: usize = 2;

/// Default consolidation interval (6 hours)
const DEFAULT_CONSOLIDATION_INTERVAL_HOURS: i64 = 6;

/// Default activity window for tracking (5 minutes)
const DEFAULT_ACTIVITY_WINDOW_SECS: i64 = 300;

/// Minimum idle time before consolidation can run (30 minutes)
const MIN_IDLE_TIME_FOR_CONSOLIDATION_MINS: i64 = 30;

/// Connection strength decay factor
const CONNECTION_DECAY_FACTOR: f64 = 0.95;

/// Minimum connection strength to keep
const MIN_CONNECTION_STRENGTH: f64 = 0.1;

/// Maximum memories to replay per cycle
const MAX_REPLAY_MEMORIES: usize = 100;

// ============================================================================
// ACTIVITY TRACKING
// ============================================================================

/// Tracks user activity to detect low-activity periods
#[derive(Debug, Clone)]
pub struct ActivityTracker {
    /// Recent activity timestamps
    activity_log: VecDeque<DateTime<Utc>>,
    /// Maximum activity log size
    max_log_size: usize,
    /// Activity window duration for rate calculation
    activity_window: Duration,
}

impl Default for ActivityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivityTracker {
    /// Create a new activity tracker
    pub fn new() -> Self {
        Self {
            activity_log: VecDeque::with_capacity(1000),
            max_log_size: 1000,
            activity_window: Duration::seconds(DEFAULT_ACTIVITY_WINDOW_SECS),
        }
    }

    /// Record an activity event
    pub fn record_activity(&mut self) {
        let now = Utc::now();
        self.activity_log.push_back(now);

        // Trim old entries
        while self.activity_log.len() > self.max_log_size {
            self.activity_log.pop_front();
        }
    }

    /// Get activity rate (events per minute) in the recent window
    pub fn activity_rate(&self) -> f64 {
        let now = Utc::now();
        let window_start = now - self.activity_window;

        let recent_count = self
            .activity_log
            .iter()
            .filter(|&&t| t >= window_start)
            .count();

        let window_minutes = self.activity_window.num_seconds() as f64 / 60.0;
        if window_minutes > 0.0 {
            recent_count as f64 / window_minutes
        } else {
            0.0
        }
    }

    /// Get time since last activity
    pub fn time_since_last_activity(&self) -> Option<Duration> {
        self.activity_log.back().map(|&last| Utc::now() - last)
    }

    /// Check if system is idle (no recent activity)
    pub fn is_idle(&self) -> bool {
        self.time_since_last_activity()
            .map(|d| d >= Duration::minutes(MIN_IDLE_TIME_FOR_CONSOLIDATION_MINS))
            .unwrap_or(true) // No activity ever = idle
    }

    /// Get activity statistics
    pub fn get_stats(&self) -> ActivityStats {
        ActivityStats {
            total_events: self.activity_log.len(),
            events_per_minute: self.activity_rate(),
            last_activity: self.activity_log.back().copied(),
            is_idle: self.is_idle(),
        }
    }
}

/// Activity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityStats {
    /// Total activity events tracked
    pub total_events: usize,
    /// Current activity rate (events per minute)
    pub events_per_minute: f64,
    /// Timestamp of last activity
    pub last_activity: Option<DateTime<Utc>>,
    /// Whether system is currently idle
    pub is_idle: bool,
}

// ============================================================================
// CONSOLIDATION SCHEDULER
// ============================================================================

/// Schedules and manages memory consolidation cycles
///
/// Inspired by sleep-based memory consolidation, this scheduler:
/// - Detects low-activity periods (like sleep)
/// - Runs consolidation cycles during these periods
/// - Tracks consolidation history and effectiveness
#[derive(Debug)]
pub struct ConsolidationScheduler {
    /// Timestamp of last consolidation
    last_consolidation: DateTime<Utc>,
    /// Minimum interval between consolidations
    consolidation_interval: Duration,
    /// Activity tracker for detecting idle periods
    activity_tracker: ActivityTracker,
    /// Consolidation history
    consolidation_history: Vec<ConsolidationReport>,
    /// Whether automatic consolidation is enabled
    auto_enabled: bool,
    /// Memory dreamer for insight generation
    dreamer: MemoryDreamer,
    /// Connection manager for tracking memory connections
    connections: Arc<RwLock<ConnectionGraph>>,
}

impl Default for ConsolidationScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationScheduler {
    /// Create a new consolidation scheduler
    pub fn new() -> Self {
        Self {
            last_consolidation: Utc::now() - Duration::hours(DEFAULT_CONSOLIDATION_INTERVAL_HOURS),
            consolidation_interval: Duration::hours(DEFAULT_CONSOLIDATION_INTERVAL_HOURS),
            activity_tracker: ActivityTracker::new(),
            consolidation_history: Vec::new(),
            auto_enabled: true,
            dreamer: MemoryDreamer::new(),
            connections: Arc::new(RwLock::new(ConnectionGraph::new())),
        }
    }

    /// Create with custom consolidation interval
    pub fn with_interval(interval_hours: i64) -> Self {
        let mut scheduler = Self::new();
        scheduler.consolidation_interval = Duration::hours(interval_hours);
        scheduler
    }

    /// Record user activity (call this on memory operations)
    pub fn record_activity(&mut self) {
        self.activity_tracker.record_activity();
    }

    /// Check if consolidation should run
    ///
    /// Returns true if:
    /// - Auto consolidation is enabled
    /// - Sufficient time has passed since last consolidation
    /// - System is currently idle
    pub fn should_consolidate(&self) -> bool {
        if !self.auto_enabled {
            return false;
        }

        let time_since_last = Utc::now() - self.last_consolidation;
        let interval_passed = time_since_last >= self.consolidation_interval;
        let is_idle = self.activity_tracker.is_idle();

        interval_passed && is_idle
    }

    /// Force check if consolidation should run (ignoring idle check)
    pub fn should_consolidate_force(&self) -> bool {
        let time_since_last = Utc::now() - self.last_consolidation;
        time_since_last >= self.consolidation_interval
    }

    /// Run a complete consolidation cycle
    ///
    /// This implements the 5-stage sleep consolidation model:
    /// 1. Replay recent memories
    /// 2. Cross-reference with existing knowledge
    /// 3. Strengthen co-activated connections
    /// 4. Prune weak connections
    /// 5. Transfer consolidated memories
    pub async fn run_consolidation_cycle(
        &mut self,
        memories: &[DreamMemory],
    ) -> ConsolidationReport {
        let start = Instant::now();
        let mut report = ConsolidationReport::new();

        // Stage 1: Memory Replay
        let replay = self.stage1_replay(memories);
        report.stage1_replay = Some(replay.clone());

        // Stage 2: Cross-reference
        let cross_refs = self.stage2_cross_reference(memories, &replay);
        report.stage2_connections = cross_refs;

        // Stage 3: Strengthen connections
        let strengthened = self.stage3_strengthen(&replay);
        report.stage3_strengthened = strengthened;

        // Stage 4: Prune weak connections
        let pruned = self.stage4_prune();
        report.stage4_pruned = pruned;

        // Stage 5: Transfer (identify memories for semantic storage)
        let transferred = self.stage5_transfer(memories);
        report.stage5_transferred = transferred;

        // Run dream cycle for insights
        let dream_result = self.dreamer.dream(memories).await;
        report.dream_result = Some(dream_result);

        // Update state
        self.last_consolidation = Utc::now();
        report.duration_ms = start.elapsed().as_millis() as u64;
        report.completed_at = Utc::now();

        // Store in history
        self.consolidation_history.push(report.clone());
        if self.consolidation_history.len() > 100 {
            self.consolidation_history.remove(0);
        }

        report
    }

    /// Stage 1: Replay recent memories in sequence
    fn stage1_replay(&self, memories: &[DreamMemory]) -> MemoryReplay {
        // Sort by creation time for sequential replay
        let mut sorted: Vec<_> = memories.iter().take(MAX_REPLAY_MEMORIES).collect();
        sorted.sort_by_key(|m| m.created_at);

        let sequence: Vec<String> = sorted.iter().map(|m| m.id.clone()).collect();

        // Generate synthetic combinations (test pairs that might have hidden connections)
        let mut synthetic_combinations = Vec::new();
        for i in 0..sorted.len().saturating_sub(1) {
            for j in (i + 1)..sorted.len().min(i + 5) {
                // Only combine memories within a close window
                synthetic_combinations.push((sorted[i].id.clone(), sorted[j].id.clone()));
            }
        }

        // Discover patterns from replay
        let discovered_patterns = self.discover_replay_patterns(&sorted);

        MemoryReplay {
            sequence,
            synthetic_combinations,
            discovered_patterns,
            replayed_at: Utc::now(),
        }
    }

    /// Discover patterns during replay
    fn discover_replay_patterns(&self, memories: &[&DreamMemory]) -> Vec<Pattern> {
        let mut patterns = Vec::new();
        let mut tag_sequences: HashMap<String, Vec<DateTime<Utc>>> = HashMap::new();

        // Track tag occurrence patterns
        for memory in memories {
            for tag in &memory.tags {
                tag_sequences
                    .entry(tag.clone())
                    .or_default()
                    .push(memory.created_at);
            }
        }

        // Identify recurring patterns
        for (tag, timestamps) in tag_sequences {
            if timestamps.len() >= 3 {
                patterns.push(Pattern {
                    id: format!("pattern-{}", Uuid::new_v4()),
                    pattern_type: PatternType::Recurring,
                    description: format!(
                        "Recurring theme '{}' across {} memories",
                        tag,
                        timestamps.len()
                    ),
                    memory_ids: memories
                        .iter()
                        .filter(|m| m.tags.contains(&tag))
                        .map(|m| m.id.clone())
                        .collect(),
                    confidence: (timestamps.len() as f64 / memories.len() as f64).min(1.0),
                    discovered_at: Utc::now(),
                });
            }
        }

        patterns
    }

    /// Stage 2: Cross-reference with existing knowledge
    fn stage2_cross_reference(&self, memories: &[DreamMemory], replay: &MemoryReplay) -> usize {
        let memory_map: HashMap<_, _> = memories.iter().map(|m| (m.id.clone(), m)).collect();

        let mut connections_found = 0;

        if let Ok(mut graph) = self.connections.write() {
            for (id_a, id_b) in &replay.synthetic_combinations {
                if let (Some(mem_a), Some(mem_b)) = (memory_map.get(id_a), memory_map.get(id_b)) {
                    // Check for connection potential
                    let similarity = calculate_memory_similarity(mem_a, mem_b);
                    if similarity >= MIN_SIMILARITY_FOR_CONNECTION {
                        graph.add_connection(
                            id_a,
                            id_b,
                            similarity,
                            ConnectionReason::CrossReference,
                        );
                        connections_found += 1;
                    }
                }
            }
        }

        connections_found
    }

    /// Stage 3: Strengthen connections that fired together
    fn stage3_strengthen(&self, replay: &MemoryReplay) -> usize {
        let mut strengthened = 0;

        if let Ok(mut graph) = self.connections.write() {
            // Strengthen connections between sequentially replayed memories
            for window in replay.sequence.windows(2) {
                if let [id_a, id_b] = window {
                    if graph.strengthen_connection(id_a, id_b, 0.1) {
                        strengthened += 1;
                    }
                }
            }

            // Also strengthen based on discovered patterns
            for pattern in &replay.discovered_patterns {
                for i in 0..pattern.memory_ids.len() {
                    for j in (i + 1)..pattern.memory_ids.len() {
                        if graph.strengthen_connection(
                            &pattern.memory_ids[i],
                            &pattern.memory_ids[j],
                            0.05 * pattern.confidence,
                        ) {
                            strengthened += 1;
                        }
                    }
                }
            }
        }

        strengthened
    }

    /// Stage 4: Prune weak connections not reactivated
    fn stage4_prune(&self) -> usize {
        let mut pruned = 0;

        if let Ok(mut graph) = self.connections.write() {
            // Apply decay to all connections
            graph.apply_decay(CONNECTION_DECAY_FACTOR);

            // Remove connections below threshold
            pruned = graph.prune_weak(MIN_CONNECTION_STRENGTH);
        }

        pruned
    }

    /// Stage 5: Identify memories ready for semantic storage transfer
    fn stage5_transfer(&self, memories: &[DreamMemory]) -> Vec<String> {
        // Memories with high access count and strong connections are candidates
        // for transfer from episodic to semantic storage
        let mut candidates = Vec::new();

        if let Ok(graph) = self.connections.read() {
            for memory in memories {
                let connection_count = graph.connection_count(&memory.id);
                let total_strength = graph.total_connection_strength(&memory.id);

                // Criteria for semantic transfer:
                // - Accessed multiple times
                // - Has multiple strong connections
                // - Is part of discovered patterns
                if memory.access_count >= 3 && connection_count >= 2 && total_strength >= 1.0 {
                    candidates.push(memory.id.clone());
                }
            }
        }

        candidates
    }

    /// Enable or disable automatic consolidation
    pub fn set_auto_enabled(&mut self, enabled: bool) {
        self.auto_enabled = enabled;
    }

    /// Get consolidation history
    pub fn get_history(&self) -> &[ConsolidationReport] {
        &self.consolidation_history
    }

    /// Get activity statistics
    pub fn get_activity_stats(&self) -> ActivityStats {
        self.activity_tracker.get_stats()
    }

    /// Get time until next scheduled consolidation
    pub fn time_until_next(&self) -> Duration {
        let elapsed = Utc::now() - self.last_consolidation;
        if elapsed >= self.consolidation_interval {
            Duration::zero()
        } else {
            self.consolidation_interval - elapsed
        }
    }

    /// Get the memory dreamer for direct access
    pub fn dreamer(&self) -> &MemoryDreamer {
        &self.dreamer
    }

    /// Get connection graph statistics
    pub fn get_connection_stats(&self) -> Option<ConnectionStats> {
        self.connections.read().ok().map(|g| g.get_stats())
    }
}

// ============================================================================
// MEMORY REPLAY
// ============================================================================

/// Result of memory replay during consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReplay {
    /// Memory IDs in replay order (chronological)
    pub sequence: Vec<String>,
    /// Synthetic combinations tested for connections
    pub synthetic_combinations: Vec<(String, String)>,
    /// Patterns discovered during replay
    pub discovered_patterns: Vec<Pattern>,
    /// When replay occurred
    pub replayed_at: DateTime<Utc>,
}

/// A discovered pattern from memory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique pattern ID
    pub id: String,
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Human-readable description
    pub description: String,
    /// Memory IDs that form this pattern
    pub memory_ids: Vec<String>,
    /// Confidence in this pattern (0.0 to 1.0)
    pub confidence: f64,
    /// When this pattern was discovered
    pub discovered_at: DateTime<Utc>,
}

/// Types of patterns that can be discovered
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PatternType {
    /// Recurring theme across memories
    Recurring,
    /// Sequential pattern (A followed by B)
    Sequential,
    /// Co-occurrence pattern
    CoOccurrence,
    /// Temporal pattern (time-based)
    Temporal,
    /// Causal pattern
    Causal,
}

// ============================================================================
// CONNECTION GRAPH
// ============================================================================

/// Graph of connections between memories
#[derive(Debug, Clone)]
pub struct ConnectionGraph {
    /// Adjacency list: memory_id -> [(connected_id, strength, reason)]
    connections: HashMap<String, Vec<MemoryConnection>>,
    /// Total connections ever created
    total_created: usize,
    /// Total connections pruned
    total_pruned: usize,
}

/// A connection between two memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConnection {
    /// Connected memory ID
    pub target_id: String,
    /// Connection strength (0.0 to 1.0+)
    pub strength: f64,
    /// Why this connection exists
    pub reason: ConnectionReason,
    /// When this connection was created
    pub created_at: DateTime<Utc>,
    /// When this connection was last strengthened
    pub last_strengthened: DateTime<Utc>,
}

/// Reason for a memory connection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionReason {
    /// Semantic similarity
    Semantic,
    /// Cross-reference during consolidation
    CrossReference,
    /// Sequential access pattern
    Sequential,
    /// Shared tags/concepts
    SharedConcepts,
    /// User-defined link
    UserDefined,
    /// Discovered pattern
    Pattern,
}

impl Default for ConnectionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionGraph {
    /// Create a new connection graph
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            total_created: 0,
            total_pruned: 0,
        }
    }

    /// Add a connection between two memories
    pub fn add_connection(
        &mut self,
        from_id: &str,
        to_id: &str,
        strength: f64,
        reason: ConnectionReason,
    ) {
        let now = Utc::now();

        // Add bidirectional connection
        for (a, b) in [(from_id, to_id), (to_id, from_id)] {
            let connections = self.connections.entry(a.to_string()).or_default();

            // Check if connection already exists
            if let Some(existing) = connections.iter_mut().find(|c| c.target_id == b) {
                existing.strength = (existing.strength + strength).min(2.0);
                existing.last_strengthened = now;
            } else {
                connections.push(MemoryConnection {
                    target_id: b.to_string(),
                    strength,
                    reason: reason.clone(),
                    created_at: now,
                    last_strengthened: now,
                });
                self.total_created += 1;
            }
        }
    }

    /// Strengthen an existing connection
    pub fn strengthen_connection(&mut self, from_id: &str, to_id: &str, boost: f64) -> bool {
        let now = Utc::now();
        let mut strengthened = false;

        for (a, b) in [(from_id, to_id), (to_id, from_id)] {
            if let Some(connections) = self.connections.get_mut(a) {
                if let Some(conn) = connections.iter_mut().find(|c| c.target_id == b) {
                    conn.strength = (conn.strength + boost).min(2.0);
                    conn.last_strengthened = now;
                    strengthened = true;
                }
            }
        }

        strengthened
    }

    /// Apply decay to all connections
    pub fn apply_decay(&mut self, decay_factor: f64) {
        for connections in self.connections.values_mut() {
            for conn in connections.iter_mut() {
                conn.strength *= decay_factor;
            }
        }
    }

    /// Prune connections below threshold
    pub fn prune_weak(&mut self, min_strength: f64) -> usize {
        let mut pruned = 0;

        for connections in self.connections.values_mut() {
            let before = connections.len();
            connections.retain(|c| c.strength >= min_strength);
            pruned += before - connections.len();
        }

        self.total_pruned += pruned;
        pruned
    }

    /// Get number of connections for a memory
    pub fn connection_count(&self, memory_id: &str) -> usize {
        self.connections
            .get(memory_id)
            .map(|c| c.len())
            .unwrap_or(0)
    }

    /// Get total connection strength for a memory
    pub fn total_connection_strength(&self, memory_id: &str) -> f64 {
        self.connections
            .get(memory_id)
            .map(|connections| connections.iter().map(|c| c.strength).sum())
            .unwrap_or(0.0)
    }

    /// Get all connections for a memory
    pub fn get_connections(&self, memory_id: &str) -> Vec<&MemoryConnection> {
        self.connections
            .get(memory_id)
            .map(|c| c.iter().collect())
            .unwrap_or_default()
    }

    /// Get statistics about the connection graph
    pub fn get_stats(&self) -> ConnectionStats {
        let total_connections: usize = self.connections.values().map(|c| c.len()).sum();
        let total_strength: f64 = self
            .connections
            .values()
            .flat_map(|c| c.iter())
            .map(|c| c.strength)
            .sum();

        ConnectionStats {
            total_memories: self.connections.len(),
            total_connections: total_connections / 2, // Bidirectional, so divide by 2
            average_strength: if total_connections > 0 {
                total_strength / total_connections as f64
            } else {
                0.0
            },
            total_created: self.total_created / 2,
            total_pruned: self.total_pruned / 2,
        }
    }
}

/// Statistics about the connection graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    /// Number of memories with connections
    pub total_memories: usize,
    /// Total number of connections
    pub total_connections: usize,
    /// Average connection strength
    pub average_strength: f64,
    /// Total connections ever created
    pub total_created: usize,
    /// Total connections pruned
    pub total_pruned: usize,
}

// ============================================================================
// CONSOLIDATION REPORT
// ============================================================================

/// Report from a consolidation cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Stage 1: Memory replay results
    pub stage1_replay: Option<MemoryReplay>,
    /// Stage 2: Number of cross-references found
    pub stage2_connections: usize,
    /// Stage 3: Number of connections strengthened
    pub stage3_strengthened: usize,
    /// Stage 4: Number of connections pruned
    pub stage4_pruned: usize,
    /// Stage 5: Memory IDs transferred to semantic storage
    pub stage5_transferred: Vec<String>,
    /// Dream cycle results
    pub dream_result: Option<DreamResult>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// When consolidation completed
    pub completed_at: DateTime<Utc>,
}

impl ConsolidationReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self {
            stage1_replay: None,
            stage2_connections: 0,
            stage3_strengthened: 0,
            stage4_pruned: 0,
            stage5_transferred: Vec::new(),
            dream_result: None,
            duration_ms: 0,
            completed_at: Utc::now(),
        }
    }

    /// Get total insights generated
    pub fn total_insights(&self) -> usize {
        self.dream_result
            .as_ref()
            .map(|r| r.insights_generated.len())
            .unwrap_or(0)
    }

    /// Get total new connections discovered
    pub fn total_new_connections(&self) -> usize {
        self.stage2_connections
            + self
                .dream_result
                .as_ref()
                .map(|r| r.new_connections_found)
                .unwrap_or(0)
    }
}

impl Default for ConsolidationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate similarity between two memories
fn calculate_memory_similarity(a: &DreamMemory, b: &DreamMemory) -> f64 {
    // Use embeddings if available
    if let (Some(emb_a), Some(emb_b)) = (&a.embedding, &b.embedding) {
        return cosine_similarity(emb_a, emb_b);
    }

    // Fallback to tag + content similarity
    let tag_sim = tag_similarity(&a.tags, &b.tags);
    let content_sim = content_word_similarity(&a.content, &b.content);

    tag_sim * 0.4 + content_sim * 0.6
}

/// Calculate tag similarity (Jaccard index)
fn tag_similarity(tags_a: &[String], tags_b: &[String]) -> f64 {
    if tags_a.is_empty() && tags_b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<_> = tags_a.iter().collect();
    let set_b: HashSet<_> = tags_b.iter().collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Calculate content similarity via word overlap
fn content_word_similarity(content_a: &str, content_b: &str) -> f64 {
    let words_a: HashSet<_> = content_a
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 3)
        .collect();

    let words_b: HashSet<_> = content_b
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 3)
        .collect();

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Result of a dream cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamResult {
    /// Number of new connections discovered
    pub new_connections_found: usize,
    /// Number of memories that were strengthened
    pub memories_strengthened: usize,
    /// Number of memories that were compressed
    pub memories_compressed: usize,
    /// Insights generated during the dream
    pub insights_generated: Vec<SynthesizedInsight>,
    /// Dream cycle duration in milliseconds
    pub duration_ms: u64,
    /// Timestamp of the dream
    pub dreamed_at: DateTime<Utc>,
    /// Statistics about the dream
    pub stats: DreamStats,
}

/// Statistics from a dream cycle
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamStats {
    /// Memories analyzed
    pub memories_analyzed: usize,
    /// Potential connections evaluated
    pub connections_evaluated: usize,
    /// Pattern clusters found
    pub clusters_found: usize,
    /// Candidate insights considered
    pub candidates_considered: usize,
}

/// A synthesized insight from memory combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedInsight {
    /// Unique ID for this insight
    pub id: String,
    /// The insight itself
    pub insight: String,
    /// Memory IDs that contributed to this insight
    pub source_memories: Vec<String>,
    /// Confidence in this insight (0.0 to 1.0)
    pub confidence: f64,
    /// Novelty score - how "new" is this insight (0.0 to 1.0)
    pub novelty_score: f64,
    /// Category/type of insight
    pub insight_type: InsightType,
    /// When this insight was generated
    pub generated_at: DateTime<Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Types of insights that can be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    /// Connection between seemingly unrelated concepts
    HiddenConnection,
    /// Recurring pattern across memories
    RecurringPattern,
    /// Generalization from specific examples
    Generalization,
    /// Contradiction or tension between memories
    Contradiction,
    /// Gap in knowledge that should be filled
    KnowledgeGap,
    /// Trend or evolution over time
    TemporalTrend,
    /// Synthesis of multiple sources
    Synthesis,
}

impl InsightType {
    /// Get description of insight type
    pub fn description(&self) -> &str {
        match self {
            Self::HiddenConnection => "Hidden connection discovered between concepts",
            Self::RecurringPattern => "Recurring pattern identified across memories",
            Self::Generalization => "General principle derived from specific cases",
            Self::Contradiction => "Potential contradiction detected",
            Self::KnowledgeGap => "Gap in knowledge identified",
            Self::TemporalTrend => "Trend or evolution observed over time",
            Self::Synthesis => "New understanding from combining sources",
        }
    }
}

/// Configuration for dream cycles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Maximum memories to analyze per dream
    pub max_memories_per_dream: usize,
    /// Minimum similarity for connection discovery
    pub min_similarity: f64,
    /// Maximum insights to generate
    pub max_insights: usize,
    /// Minimum novelty required for insights
    pub min_novelty: f64,
    /// Enable compression during dreams
    pub enable_compression: bool,
    /// Enable strengthening during dreams
    pub enable_strengthening: bool,
    /// Focus on specific tags (empty = all)
    pub focus_tags: Vec<String>,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            max_memories_per_dream: 1000,
            min_similarity: MIN_SIMILARITY_FOR_CONNECTION,
            max_insights: MAX_INSIGHTS_PER_DREAM,
            min_novelty: MIN_NOVELTY_SCORE,
            enable_compression: true,
            enable_strengthening: true,
            focus_tags: vec![],
        }
    }
}

/// Memory input for dreaming
#[derive(Debug, Clone)]
pub struct DreamMemory {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Access count
    pub access_count: u32,
}

/// A discovered connection between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredConnection {
    /// Source memory ID
    pub from_id: String,
    /// Target memory ID
    pub to_id: String,
    /// Similarity score
    pub similarity: f64,
    /// Type of connection discovered
    pub connection_type: DiscoveredConnectionType,
    /// Reasoning for this connection
    pub reasoning: String,
}

/// Types of connections discovered during dreaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveredConnectionType {
    /// Semantic similarity
    Semantic,
    /// Shared concepts/entities
    SharedConcept,
    /// Temporal correlation
    Temporal,
    /// Complementary information
    Complementary,
    /// Cause-effect relationship
    CausalChain,
}

/// Memory dreamer for enhanced consolidation
#[derive(Debug)]
pub struct MemoryDreamer {
    /// Configuration
    config: DreamConfig,
    /// Dream history
    dream_history: Arc<RwLock<Vec<DreamResult>>>,
    /// Generated insights (persisted separately)
    insights: Arc<RwLock<Vec<SynthesizedInsight>>>,
    /// Discovered connections
    connections: Arc<RwLock<Vec<DiscoveredConnection>>>,
}

impl MemoryDreamer {
    /// Create a new memory dreamer with default config
    pub fn new() -> Self {
        Self::with_config(DreamConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DreamConfig) -> Self {
        Self {
            config,
            dream_history: Arc::new(RwLock::new(Vec::new())),
            insights: Arc::new(RwLock::new(Vec::new())),
            connections: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Run a dream cycle on provided memories
    pub async fn dream(&self, memories: &[DreamMemory]) -> DreamResult {
        let start = std::time::Instant::now();
        let mut stats = DreamStats::default();

        // Filter memories based on config
        let working_memories: Vec<_> = if self.config.focus_tags.is_empty() {
            memories
                .iter()
                .take(self.config.max_memories_per_dream)
                .collect()
        } else {
            memories
                .iter()
                .filter(|m| m.tags.iter().any(|t| self.config.focus_tags.contains(t)))
                .take(self.config.max_memories_per_dream)
                .collect()
        };

        stats.memories_analyzed = working_memories.len();

        // Phase 1: Discover new connections
        let new_connections = self.discover_connections(&working_memories, &mut stats);

        // Phase 2: Find clusters/patterns
        let clusters = self.find_clusters(&working_memories, &new_connections);
        stats.clusters_found = clusters.len();

        // Phase 3: Generate insights
        let insights = self.generate_insights(&working_memories, &clusters, &mut stats);

        // Phase 4: Strengthen important memories (would update storage)
        let memories_strengthened = if self.config.enable_strengthening {
            self.identify_memories_to_strengthen(&working_memories, &new_connections)
        } else {
            0
        };

        // Phase 5: Identify compression candidates (would compress in storage)
        let memories_compressed = if self.config.enable_compression {
            self.identify_compression_candidates(&working_memories)
        } else {
            0
        };

        // Store results
        self.store_connections(&new_connections);
        self.store_insights(&insights);

        let result = DreamResult {
            new_connections_found: new_connections.len(),
            memories_strengthened,
            memories_compressed,
            insights_generated: insights,
            duration_ms: start.elapsed().as_millis() as u64,
            dreamed_at: Utc::now(),
            stats,
        };

        // Store in history
        if let Ok(mut history) = self.dream_history.write() {
            history.push(result.clone());
            // Keep last 100 dreams
            if history.len() > 100 {
                history.remove(0);
            }
        }

        result
    }

    /// Synthesize insights from memories without full dream cycle
    pub fn synthesize_insights(&self, memories: &[DreamMemory]) -> Vec<SynthesizedInsight> {
        let mut stats = DreamStats::default();

        // Find clusters
        let connections =
            self.discover_connections(&memories.iter().collect::<Vec<_>>(), &mut stats);
        let clusters = self.find_clusters(&memories.iter().collect::<Vec<_>>(), &connections);

        // Generate insights
        self.generate_insights(&memories.iter().collect::<Vec<_>>(), &clusters, &mut stats)
    }

    /// Get all generated insights
    pub fn get_insights(&self) -> Vec<SynthesizedInsight> {
        self.insights.read().map(|i| i.clone()).unwrap_or_default()
    }

    /// Get insights by type
    pub fn get_insights_by_type(&self, insight_type: &InsightType) -> Vec<SynthesizedInsight> {
        self.insights
            .read()
            .map(|insights| {
                insights
                    .iter()
                    .filter(|i| {
                        std::mem::discriminant(&i.insight_type)
                            == std::mem::discriminant(insight_type)
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get dream history
    pub fn get_dream_history(&self) -> Vec<DreamResult> {
        self.dream_history
            .read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Get discovered connections
    pub fn get_connections(&self) -> Vec<DiscoveredConnection> {
        self.connections
            .read()
            .map(|c| c.clone())
            .unwrap_or_default()
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    fn discover_connections(
        &self,
        memories: &[&DreamMemory],
        stats: &mut DreamStats,
    ) -> Vec<DiscoveredConnection> {
        let mut connections = Vec::new();

        // Compare each pair of memories
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                stats.connections_evaluated += 1;

                let mem_a = &memories[i];
                let mem_b = &memories[j];

                // Calculate similarity
                let similarity = self.calculate_similarity(mem_a, mem_b);

                if similarity >= self.config.min_similarity {
                    let connection_type = self.determine_connection_type(mem_a, mem_b, similarity);
                    let reasoning =
                        self.generate_connection_reasoning(mem_a, mem_b, &connection_type);

                    connections.push(DiscoveredConnection {
                        from_id: mem_a.id.clone(),
                        to_id: mem_b.id.clone(),
                        similarity,
                        connection_type,
                        reasoning,
                    });
                }
            }
        }

        connections
    }

    fn calculate_similarity(&self, a: &DreamMemory, b: &DreamMemory) -> f64 {
        // Primary: embedding similarity
        if let (Some(emb_a), Some(emb_b)) = (&a.embedding, &b.embedding) {
            return cosine_similarity(emb_a, emb_b);
        }

        // Fallback: tag overlap + content similarity
        let tag_sim = self.tag_similarity(&a.tags, &b.tags);
        let content_sim = self.content_similarity(&a.content, &b.content);

        tag_sim * 0.4 + content_sim * 0.6
    }

    fn tag_similarity(&self, tags_a: &[String], tags_b: &[String]) -> f64 {
        if tags_a.is_empty() && tags_b.is_empty() {
            return 0.0;
        }

        let set_a: HashSet<_> = tags_a.iter().collect();
        let set_b: HashSet<_> = tags_b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn content_similarity(&self, content_a: &str, content_b: &str) -> f64 {
        // Simple word overlap (Jaccard)
        let words_a: HashSet<_> = content_a
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let words_b: HashSet<_> = content_b
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn determine_connection_type(
        &self,
        a: &DreamMemory,
        b: &DreamMemory,
        similarity: f64,
    ) -> DiscoveredConnectionType {
        // Check for shared concepts (via tags)
        let shared_tags = a.tags.iter().filter(|t| b.tags.contains(t)).count();
        if shared_tags >= 2 {
            return DiscoveredConnectionType::SharedConcept;
        }

        // Check for temporal correlation
        let time_diff = (a.created_at - b.created_at).num_hours().abs();
        if time_diff <= 24 && similarity > 0.6 {
            return DiscoveredConnectionType::Temporal;
        }

        // High semantic similarity
        if similarity > 0.8 {
            return DiscoveredConnectionType::Semantic;
        }

        // Default to complementary
        DiscoveredConnectionType::Complementary
    }

    fn generate_connection_reasoning(
        &self,
        a: &DreamMemory,
        b: &DreamMemory,
        conn_type: &DiscoveredConnectionType,
    ) -> String {
        match conn_type {
            DiscoveredConnectionType::Semantic => format!(
                "High semantic similarity between '{}...' and '{}...'",
                truncate(&a.content, 30),
                truncate(&b.content, 30)
            ),
            DiscoveredConnectionType::SharedConcept => {
                let shared: Vec<_> = a.tags.iter().filter(|t| b.tags.contains(t)).collect();
                format!("Shared concepts: {:?}", shared)
            }
            DiscoveredConnectionType::Temporal => "Created within close time proximity".to_string(),
            DiscoveredConnectionType::Complementary => {
                "Memories provide complementary information".to_string()
            }
            DiscoveredConnectionType::CausalChain => {
                "Potential cause-effect relationship".to_string()
            }
        }
    }

    fn find_clusters(
        &self,
        _memories: &[&DreamMemory],
        connections: &[DiscoveredConnection],
    ) -> Vec<Vec<String>> {
        // Simple clustering based on connections
        let mut clusters: Vec<HashSet<String>> = Vec::new();

        for conn in connections {
            // Find existing cluster containing either endpoint
            let mut found_cluster = None;
            for (i, cluster) in clusters.iter().enumerate() {
                if cluster.contains(&conn.from_id) || cluster.contains(&conn.to_id) {
                    found_cluster = Some(i);
                    break;
                }
            }

            match found_cluster {
                Some(i) => {
                    clusters[i].insert(conn.from_id.clone());
                    clusters[i].insert(conn.to_id.clone());
                }
                None => {
                    let mut new_cluster = HashSet::new();
                    new_cluster.insert(conn.from_id.clone());
                    new_cluster.insert(conn.to_id.clone());
                    clusters.push(new_cluster);
                }
            }
        }

        // Merge overlapping clusters
        let mut merged = true;
        while merged {
            merged = false;
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    if !clusters[i].is_disjoint(&clusters[j]) {
                        let to_merge: HashSet<_> = clusters[j].drain().collect();
                        clusters[i].extend(to_merge);
                        merged = true;
                        break;
                    }
                }
                if merged {
                    clusters.retain(|c| !c.is_empty());
                    break;
                }
            }
        }

        // Convert to Vec<Vec<String>>
        clusters
            .into_iter()
            .filter(|c| c.len() >= MIN_MEMORIES_FOR_INSIGHT)
            .map(|c| c.into_iter().collect())
            .collect()
    }

    fn generate_insights(
        &self,
        memories: &[&DreamMemory],
        clusters: &[Vec<String>],
        stats: &mut DreamStats,
    ) -> Vec<SynthesizedInsight> {
        let mut insights = Vec::new();
        let memory_map: HashMap<_, _> = memories.iter().map(|m| (&m.id, *m)).collect();

        for cluster in clusters {
            stats.candidates_considered += 1;

            // Get memories in this cluster
            let cluster_memories: Vec<_> = cluster
                .iter()
                .filter_map(|id| memory_map.get(&id).copied())
                .collect();

            if cluster_memories.len() < MIN_MEMORIES_FOR_INSIGHT {
                continue;
            }

            // Try to generate insight from this cluster
            if let Some(insight) = self.generate_insight_from_cluster(&cluster_memories) {
                if insight.novelty_score >= self.config.min_novelty {
                    insights.push(insight);
                }
            }

            if insights.len() >= self.config.max_insights {
                break;
            }
        }

        insights
    }

    fn generate_insight_from_cluster(
        &self,
        memories: &[&DreamMemory],
    ) -> Option<SynthesizedInsight> {
        if memories.is_empty() {
            return None;
        }

        // Collect all tags
        let all_tags: HashSet<_> = memories
            .iter()
            .flat_map(|m| m.tags.iter().cloned())
            .collect();

        // Find common themes
        let common_tags: Vec<_> = all_tags
            .iter()
            .filter(|t| {
                memories.iter().filter(|m| m.tags.contains(*t)).count() > memories.len() / 2
            })
            .cloned()
            .collect();

        // Generate insight based on cluster characteristics
        let (insight_text, insight_type) = self.synthesize_insight_text(memories, &common_tags);

        // Calculate novelty (simplified)
        let novelty = self.calculate_novelty(&insight_text, memories);

        // Calculate confidence based on cluster cohesion
        let confidence = self.calculate_insight_confidence(memories);

        Some(SynthesizedInsight {
            id: format!("insight-{}", Uuid::new_v4()),
            insight: insight_text,
            source_memories: memories.iter().map(|m| m.id.clone()).collect(),
            confidence,
            novelty_score: novelty,
            insight_type,
            generated_at: Utc::now(),
            tags: common_tags,
        })
    }

    fn synthesize_insight_text(
        &self,
        memories: &[&DreamMemory],
        common_tags: &[String],
    ) -> (String, InsightType) {
        // Determine insight type based on memory characteristics
        let time_range = memories
            .iter()
            .map(|m| m.created_at)
            .fold((Utc::now(), Utc::now() - Duration::days(365)), |acc, t| {
                (acc.0.min(t), acc.1.max(t))
            });

        let time_span_days = (time_range.1 - time_range.0).num_days();

        if time_span_days > 30 {
            // Temporal trend
            let insight = format!(
                "Pattern observed over {} days in '{}': recurring theme across {} related memories",
                time_span_days,
                common_tags.first().map(|s| s.as_str()).unwrap_or("topic"),
                memories.len()
            );
            (insight, InsightType::TemporalTrend)
        } else if common_tags.len() >= 2 {
            // Hidden connection
            let insight = format!(
                "Connection between '{}' and '{}' found across {} memories",
                common_tags.first().map(|s| s.as_str()).unwrap_or("A"),
                common_tags.get(1).map(|s| s.as_str()).unwrap_or("B"),
                memories.len()
            );
            (insight, InsightType::HiddenConnection)
        } else if memories.len() >= 3 {
            // Recurring pattern
            let insight = format!(
                "Recurring pattern in '{}': {} instances identified with common characteristics",
                common_tags.first().map(|s| s.as_str()).unwrap_or("topic"),
                memories.len()
            );
            (insight, InsightType::RecurringPattern)
        } else {
            // Synthesis
            let insight = format!(
                "Synthesis: {} related memories about '{}' suggest broader understanding",
                memories.len(),
                common_tags.first().map(|s| s.as_str()).unwrap_or("topic")
            );
            (insight, InsightType::Synthesis)
        }
    }

    fn calculate_novelty(&self, insight: &str, source_memories: &[&DreamMemory]) -> f64 {
        // Novelty = how different is the insight from source memories

        // Count unique words in insight not heavily present in sources
        let insight_words: HashSet<_> = insight
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let source_words: HashSet<_> = source_memories
            .iter()
            .flat_map(|m| m.content.split_whitespace())
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let novel_words = insight_words.difference(&source_words).count();
        let total_words = insight_words.len();

        if total_words == 0 {
            return 0.3; // Default low novelty
        }

        // Base novelty from word difference
        let word_novelty = (novel_words as f64 / total_words as f64) * 0.5;

        // Boost novelty if connecting multiple sources
        let source_bonus = ((source_memories.len() as f64 - 2.0) * 0.1)
            .max(0.0)
            .min(0.3);

        (word_novelty + source_bonus + 0.2).min(1.0)
    }

    fn calculate_insight_confidence(&self, memories: &[&DreamMemory]) -> f64 {
        // Confidence based on:
        // 1. Number of supporting memories
        // 2. Access patterns of source memories
        // 3. Tag overlap

        let count_factor = (memories.len() as f64 / 5.0).min(1.0) * 0.4;

        let avg_access =
            memories.iter().map(|m| m.access_count as f64).sum::<f64>() / memories.len() as f64;
        let access_factor = (avg_access / 10.0).min(1.0) * 0.3;

        let tag_overlap = self.average_tag_overlap(memories);
        let tag_factor = tag_overlap * 0.3;

        (count_factor + access_factor + tag_factor).min(0.95)
    }

    fn average_tag_overlap(&self, memories: &[&DreamMemory]) -> f64 {
        if memories.len() < 2 {
            return 0.0;
        }

        let mut total_overlap = 0.0;
        let mut comparisons = 0;

        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                total_overlap += self.tag_similarity(&memories[i].tags, &memories[j].tags);
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            0.0
        } else {
            total_overlap / comparisons as f64
        }
    }

    fn identify_memories_to_strengthen(
        &self,
        _memories: &[&DreamMemory],
        connections: &[DiscoveredConnection],
    ) -> usize {
        // Memories with many connections should be strengthened
        let mut connection_counts: HashMap<&str, usize> = HashMap::new();

        for conn in connections {
            *connection_counts.entry(&conn.from_id).or_insert(0) += 1;
            *connection_counts.entry(&conn.to_id).or_insert(0) += 1;
        }

        // Count memories with above-average connections
        let avg_connections = if connection_counts.is_empty() {
            0.0
        } else {
            connection_counts.values().sum::<usize>() as f64 / connection_counts.len() as f64
        };

        connection_counts
            .values()
            .filter(|&&count| count as f64 > avg_connections)
            .count()
    }

    fn identify_compression_candidates(&self, memories: &[&DreamMemory]) -> usize {
        // Old memories with low access that are similar to others
        let now = Utc::now();
        let old_threshold = now - Duration::days(60);

        memories
            .iter()
            .filter(|m| m.created_at < old_threshold && m.access_count < 3)
            .count()
            / 3 // Rough estimate of compressible groups
    }

    fn store_connections(&self, connections: &[DiscoveredConnection]) {
        if let Ok(mut stored) = self.connections.write() {
            stored.extend(connections.iter().cloned());
            // Keep last 1000 connections
            let len = stored.len();
            if len > 1000 {
                stored.drain(0..(len - 1000));
            }
        }
    }

    fn store_insights(&self, insights: &[SynthesizedInsight]) {
        if let Ok(mut stored) = self.insights.write() {
            stored.extend(insights.iter().cloned());
            // Keep last 500 insights
            let len = stored.len();
            if len > 500 {
                stored.drain(0..(len - 500));
            }
        }
    }
}

impl Default for MemoryDreamer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    (dot / (mag_a * mag_b)) as f64
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(id: &str, content: &str, tags: Vec<&str>) -> DreamMemory {
        DreamMemory {
            id: id.to_string(),
            content: content.to_string(),
            embedding: None,
            tags: tags.into_iter().map(String::from).collect(),
            created_at: Utc::now(),
            access_count: 1,
        }
    }

    fn make_memory_with_time(
        id: &str,
        content: &str,
        tags: Vec<&str>,
        hours_ago: i64,
    ) -> DreamMemory {
        DreamMemory {
            id: id.to_string(),
            content: content.to_string(),
            embedding: None,
            tags: tags.into_iter().map(String::from).collect(),
            created_at: Utc::now() - Duration::hours(hours_ago),
            access_count: 1,
        }
    }

    #[tokio::test]
    async fn test_dream_cycle() {
        let dreamer = MemoryDreamer::new();

        let memories = vec![
            make_memory(
                "1",
                "Database indexing improves query performance",
                vec!["database", "performance"],
            ),
            make_memory(
                "2",
                "Query optimization techniques for SQL",
                vec!["database", "sql"],
            ),
            make_memory(
                "3",
                "Performance tuning in database systems",
                vec!["database", "performance"],
            ),
            make_memory(
                "4",
                "Understanding B-tree indexes",
                vec!["database", "indexing"],
            ),
        ];

        let result = dreamer.dream(&memories).await;

        assert!(result.stats.memories_analyzed == 4);
        assert!(result.stats.connections_evaluated > 0);
    }

    #[test]
    fn test_tag_similarity() {
        let dreamer = MemoryDreamer::new();

        let tags_a = vec!["rust".to_string(), "programming".to_string()];
        let tags_b = vec!["rust".to_string(), "memory".to_string()];

        let sim = dreamer.tag_similarity(&tags_a, &tags_b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_insight_type_description() {
        assert!(!InsightType::HiddenConnection.description().is_empty());
        assert!(!InsightType::RecurringPattern.description().is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    // ========== Activity Tracker Tests ==========

    #[test]
    fn test_activity_tracker_new() {
        let tracker = ActivityTracker::new();
        assert!(tracker.is_idle());
        assert_eq!(tracker.activity_rate(), 0.0);
    }

    #[test]
    fn test_activity_tracker_record() {
        let mut tracker = ActivityTracker::new();

        tracker.record_activity();
        assert!(!tracker.is_idle()); // Just recorded activity

        let stats = tracker.get_stats();
        assert_eq!(stats.total_events, 1);
        assert!(stats.last_activity.is_some());
    }

    #[test]
    fn test_activity_rate() {
        let mut tracker = ActivityTracker::new();

        // Record 10 events
        for _ in 0..10 {
            tracker.record_activity();
        }

        // Rate should be > 0
        assert!(tracker.activity_rate() > 0.0);
    }

    // ========== Consolidation Scheduler Tests ==========

    #[test]
    fn test_scheduler_new() {
        let scheduler = ConsolidationScheduler::new();
        // Should consolidate immediately (interval passed since "past" initialization)
        assert!(scheduler.should_consolidate_force());
    }

    #[test]
    fn test_scheduler_with_interval() {
        let scheduler = ConsolidationScheduler::with_interval(12);
        assert!(scheduler.time_until_next() <= Duration::hours(12));
    }

    #[test]
    fn test_scheduler_activity_tracking() {
        let mut scheduler = ConsolidationScheduler::new();

        scheduler.record_activity();

        let stats = scheduler.get_activity_stats();
        assert_eq!(stats.total_events, 1);
        assert!(!stats.is_idle);
    }

    #[tokio::test]
    async fn test_consolidation_cycle() {
        let mut scheduler = ConsolidationScheduler::new();

        let memories = vec![
            make_memory_with_time("1", "First memory about rust", vec!["rust"], 5),
            make_memory_with_time(
                "2",
                "Second memory about rust programming",
                vec!["rust", "programming"],
                4,
            ),
            make_memory_with_time("3", "Third memory about systems", vec!["systems"], 3),
            make_memory_with_time(
                "4",
                "Fourth memory about rust systems",
                vec!["rust", "systems"],
                2,
            ),
        ];

        let report = scheduler.run_consolidation_cycle(&memories).await;

        // Should have completed all stages
        assert!(report.stage1_replay.is_some());
        assert!(report.duration_ms >= 0);
        assert!(report.completed_at <= Utc::now());
    }

    // ========== Memory Replay Tests ==========

    #[test]
    fn test_memory_replay_structure() {
        let replay = MemoryReplay {
            sequence: vec!["1".to_string(), "2".to_string()],
            synthetic_combinations: vec![("1".to_string(), "2".to_string())],
            discovered_patterns: vec![],
            replayed_at: Utc::now(),
        };

        assert_eq!(replay.sequence.len(), 2);
        assert_eq!(replay.synthetic_combinations.len(), 1);
    }

    // ========== Connection Graph Tests ==========

    #[test]
    fn test_connection_graph_add() {
        let mut graph = ConnectionGraph::new();

        graph.add_connection("a", "b", 0.8, ConnectionReason::Semantic);

        assert_eq!(graph.connection_count("a"), 1);
        assert_eq!(graph.connection_count("b"), 1);
        assert!((graph.total_connection_strength("a") - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_connection_graph_strengthen() {
        let mut graph = ConnectionGraph::new();

        graph.add_connection("a", "b", 0.5, ConnectionReason::Semantic);
        assert!(graph.strengthen_connection("a", "b", 0.2));

        // Strength should be approximately 0.7
        let strength = graph.total_connection_strength("a");
        assert!(strength >= 0.7);
    }

    #[test]
    fn test_connection_graph_decay_and_prune() {
        let mut graph = ConnectionGraph::new();

        graph.add_connection("a", "b", 0.2, ConnectionReason::Semantic);

        // Apply decay multiple times
        for _ in 0..10 {
            graph.apply_decay(0.8);
        }

        // Prune weak connections
        let pruned = graph.prune_weak(0.1);

        // Connection should be pruned
        assert!(pruned > 0 || graph.connection_count("a") == 0);
    }

    #[test]
    fn test_connection_graph_stats() {
        let mut graph = ConnectionGraph::new();

        graph.add_connection("a", "b", 0.8, ConnectionReason::Semantic);
        graph.add_connection("b", "c", 0.6, ConnectionReason::CrossReference);

        let stats = graph.get_stats();
        assert_eq!(stats.total_connections, 2);
        assert!(stats.average_strength > 0.0);
    }

    // ========== Consolidation Report Tests ==========

    #[test]
    fn test_consolidation_report_new() {
        let report = ConsolidationReport::new();

        assert_eq!(report.stage2_connections, 0);
        assert_eq!(report.total_insights(), 0);
        assert_eq!(report.total_new_connections(), 0);
    }

    // ========== Pattern Tests ==========

    #[test]
    fn test_pattern_types() {
        let pattern = Pattern {
            id: "test".to_string(),
            pattern_type: PatternType::Recurring,
            description: "Test pattern".to_string(),
            memory_ids: vec!["1".to_string(), "2".to_string()],
            confidence: 0.8,
            discovered_at: Utc::now(),
        };

        assert_eq!(pattern.pattern_type, PatternType::Recurring);
        assert_eq!(pattern.memory_ids.len(), 2);
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_calculate_memory_similarity() {
        let mem_a = make_memory(
            "1",
            "Rust programming language",
            vec!["rust", "programming"],
        );
        let mem_b = make_memory("2", "Rust systems programming", vec!["rust", "systems"]);

        let similarity = calculate_memory_similarity(&mem_a, &mem_b);
        assert!(similarity > 0.0); // Should have some similarity due to shared "rust" tag
    }

    #[test]
    fn test_tag_similarity_function() {
        let tags_a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tags_b = vec!["b".to_string(), "c".to_string(), "d".to_string()];

        let sim = tag_similarity(&tags_a, &tags_b);
        // Jaccard: 2 / 4 = 0.5
        assert!((sim - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_content_word_similarity() {
        let content_a = "The quick brown fox jumps over the lazy dog";
        let content_b = "The quick brown cat jumps over the lazy dog";

        let sim = content_word_similarity(content_a, content_b);
        assert!(sim > 0.5); // High overlap
    }
}
