//! # Synaptic Tagging and Capture (STC)
//!
//! Implements the neuroscience finding that memories can become important RETROACTIVELY
//! based on subsequent events. This is a fundamental capability that distinguishes
//! biological memory from traditional AI memory systems.
//!
//! ## The Neuroscience
//!
//! Synaptic Tagging and Capture (STC) explains how memories can be consolidated
//! hours after initial encoding:
//!
//! 1. **Weak stimulation** creates a "synaptic tag" - a temporary molecular marker
//! 2. **Strong stimulation** (important event) triggers production of
//!    Plasticity-Related Products (PRPs)
//! 3. **PRPs can be captured** by tagged synapses within a temporal window
//! 4. **Captured memories** are consolidated to long-term storage
//!
//! > "Successful STC is observed even with a 9-hour interval between weak and strong
//! > stimulation, suggesting a broader temporal flexibility for tag-PRP interactions."
//! > - Redondo & Morris (2011)
//!
//! ## Why This Matters for AI
//!
//! Traditional AI memory systems determine importance at encoding time. But in reality:
//! - A conversation about a coworker's vacation might seem trivial
//! - Hours later, you learn that coworker is leaving the company
//! - Suddenly, that vacation conversation becomes important context
//!
//! STC enables this retroactive importance assignment - something no other AI memory
//! system does.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use vestige_core::neuroscience::{SynapticTaggingSystem, ImportanceEvent, ImportanceEventType};
//! use chrono::Utc;
//!
//! let mut stc = SynapticTaggingSystem::new();
//!
//! // Memory is encoded (automatically tagged)
//! stc.tag_memory("mem-123");
//!
//! // Hours later, user explicitly flags something as important
//! let captured = stc.trigger_prp(ImportanceEvent {
//!     event_type: ImportanceEventType::UserFlag,
//!     memory_id: Some("mem-456".to_string()),
//!     timestamp: Utc::now(),
//!     strength: 1.0,
//!     context: Some("User said 'remember this'".to_string()),
//! });
//!
//! // PRPs sweep backward through time, capturing tagged memories
//! for memory in captured {
//!     println!("Retroactively consolidated: {}", memory.memory_id);
//! }
//! ```
//!
//! ## Configuration
//!
//! The system is highly configurable to match different use cases:
//!
//! - **Capture Window**: How far back/forward to look for tagged memories
//! - **Decay Function**: How tag strength decays over time
//! - **PRP Threshold**: Minimum importance to trigger PRP production
//! - **Cluster Settings**: How to group related important memories

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default backward capture window (hours) - based on neuroscience research
/// showing successful STC even with 9-hour intervals
const DEFAULT_BACKWARD_HOURS: f64 = 9.0;

/// Default forward capture window (hours) - smaller since we're looking ahead
const DEFAULT_FORWARD_HOURS: f64 = 2.0;

/// Default tag lifetime before complete decay (hours)
const DEFAULT_TAG_LIFETIME_HOURS: f64 = 12.0;

/// Default PRP threshold - minimum importance to trigger PRP production
const DEFAULT_PRP_THRESHOLD: f64 = 0.7;

/// Default minimum tag strength for capture
const DEFAULT_MIN_TAG_STRENGTH: f64 = 0.3;

/// Maximum importance cluster size
const DEFAULT_MAX_CLUSTER_SIZE: usize = 50;

// ============================================================================
// DECAY FUNCTIONS
// ============================================================================

/// Decay function for synaptic tag strength
///
/// Different decay functions model different memory characteristics:
/// - Exponential: Rapid initial decay, slow tail (default for short-term)
/// - Linear: Constant decay rate
/// - Power: Slow initial decay, accelerating over time
/// - Logarithmic: Very slow decay, good for important memories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum DecayFunction {
    /// Exponential decay: strength = initial * e^(-lambda * t)
    /// Best for modeling biological tag decay
    #[default]
    Exponential,
    /// Linear decay: strength = initial * (1 - t/lifetime)
    /// Simple, predictable decay
    Linear,
    /// Power law decay: strength = initial * (1 + t)^(-alpha)
    /// Matches FSRS-6 forgetting curve
    Power,
    /// Logarithmic decay: strength = initial * (1 - ln(1+t)/ln(1+lifetime))
    /// Very slow decay for persistent tags
    Logarithmic,
}


impl DecayFunction {
    /// Calculate decayed strength
    ///
    /// # Arguments
    /// * `initial_strength` - Initial tag strength (0.0 to 1.0)
    /// * `hours_elapsed` - Time since tag creation
    /// * `lifetime_hours` - Total lifetime before complete decay
    ///
    /// # Returns
    /// Decayed strength (0.0 to 1.0)
    pub fn apply(&self, initial_strength: f64, hours_elapsed: f64, lifetime_hours: f64) -> f64 {
        if hours_elapsed <= 0.0 {
            return initial_strength;
        }
        if hours_elapsed >= lifetime_hours {
            return 0.0;
        }

        let t = hours_elapsed;
        let l = lifetime_hours;

        let decayed = match self {
            DecayFunction::Exponential => {
                // lambda = -ln(0.01) / lifetime for 99% decay at lifetime
                let lambda = 4.605 / l;
                initial_strength * (-lambda * t).exp()
            }
            DecayFunction::Linear => initial_strength * (1.0 - t / l),
            DecayFunction::Power => {
                // alpha = 0.5 matches FSRS-6
                let alpha = 0.5;
                initial_strength * (1.0 + t / l).powf(-alpha)
            }
            DecayFunction::Logarithmic => {
                initial_strength * (1.0 - (1.0 + t).ln() / (1.0 + l).ln())
            }
        };

        decayed.clamp(0.0, 1.0)
    }
}

// ============================================================================
// SYNAPTIC TAG
// ============================================================================

/// A synaptic tag marking a memory for potential consolidation
///
/// In neuroscience, a synaptic tag is a temporary molecular marker created at
/// a synapse after weak stimulation. It marks the synapse as "eligible" for
/// consolidation if plasticity-related products (PRPs) arrive within the
/// capture window.
///
/// ## Lifecycle
///
/// 1. Created when a memory is encoded
/// 2. Strength decays over time
/// 3. If PRP arrives while strength > threshold, memory is captured
/// 4. Captured memories are promoted to long-term storage
/// 5. Uncaptured tags eventually decay to zero and are cleaned up
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapticTag {
    /// The memory this tag is attached to
    pub memory_id: String,
    /// When the tag was created (memory encoding time)
    pub created_at: DateTime<Utc>,
    /// Current tag strength (decays over time)
    pub tag_strength: f64,
    /// Initial tag strength at creation
    pub initial_strength: f64,
    /// Whether this memory has been captured (consolidated)
    pub captured: bool,
    /// The event that captured this memory (if any)
    pub capture_event: Option<String>,
    /// When the memory was captured
    pub captured_at: Option<DateTime<Utc>>,
    /// Context about why this tag was created
    pub encoding_context: Option<String>,
}

impl SynapticTag {
    /// Create a new synaptic tag for a memory
    pub fn new(memory_id: &str) -> Self {
        Self {
            memory_id: memory_id.to_string(),
            created_at: Utc::now(),
            tag_strength: 1.0,
            initial_strength: 1.0,
            captured: false,
            capture_event: None,
            captured_at: None,
            encoding_context: None,
        }
    }

    /// Create with custom initial strength
    pub fn with_strength(memory_id: &str, strength: f64) -> Self {
        Self {
            memory_id: memory_id.to_string(),
            created_at: Utc::now(),
            tag_strength: strength.clamp(0.0, 1.0),
            initial_strength: strength.clamp(0.0, 1.0),
            captured: false,
            capture_event: None,
            captured_at: None,
            encoding_context: None,
        }
    }

    /// Create with encoding context
    pub fn with_context(memory_id: &str, context: &str) -> Self {
        Self {
            memory_id: memory_id.to_string(),
            created_at: Utc::now(),
            tag_strength: 1.0,
            initial_strength: 1.0,
            captured: false,
            capture_event: None,
            captured_at: None,
            encoding_context: Some(context.to_string()),
        }
    }

    /// Calculate current tag strength with decay
    pub fn current_strength(&self, decay_fn: DecayFunction, lifetime_hours: f64) -> f64 {
        // Use milliseconds for precise timing (important for tests with short lifetimes)
        let hours_elapsed = (Utc::now() - self.created_at).num_milliseconds() as f64 / 3_600_000.0;
        decay_fn.apply(self.initial_strength, hours_elapsed, lifetime_hours)
    }

    /// Check if the tag is still active (above minimum threshold)
    pub fn is_active(
        &self,
        decay_fn: DecayFunction,
        lifetime_hours: f64,
        min_strength: f64,
    ) -> bool {
        !self.captured && self.current_strength(decay_fn, lifetime_hours) >= min_strength
    }

    /// Mark this tag as captured
    pub fn capture(&mut self, event_id: &str) {
        self.captured = true;
        self.capture_event = Some(event_id.to_string());
        self.captured_at = Some(Utc::now());
    }

    /// Get the age of this tag in hours
    pub fn age_hours(&self) -> f64 {
        (Utc::now() - self.created_at).num_milliseconds() as f64 / 3_600_000.0
    }
}

// ============================================================================
// CAPTURE WINDOW
// ============================================================================

/// Temporal window for PRP capture
///
/// When an important event occurs, PRPs are produced and can be captured by
/// tagged memories within this temporal window. The window extends both
/// backward (already encoded memories) and forward (memories about to be
/// encoded).
///
/// ## Biological Basis
///
/// Research shows that STC can occur with intervals up to 9 hours between
/// weak and strong stimulation. This suggests a broader temporal flexibility
/// for tag-PRP interactions than previously thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureWindow {
    /// How far back to look for tagged memories (hours)
    pub backward_hours: f64,
    /// How far forward to look for tagged memories (hours)
    pub forward_hours: f64,
    /// Decay function for capture probability
    pub decay_function: DecayFunction,
}

impl Default for CaptureWindow {
    fn default() -> Self {
        Self {
            backward_hours: DEFAULT_BACKWARD_HOURS,
            forward_hours: DEFAULT_FORWARD_HOURS,
            decay_function: DecayFunction::Exponential,
        }
    }
}

impl CaptureWindow {
    /// Create a new capture window
    pub fn new(backward_hours: f64, forward_hours: f64) -> Self {
        Self {
            backward_hours,
            forward_hours,
            decay_function: DecayFunction::Exponential,
        }
    }

    /// Create with custom decay function
    pub fn with_decay(backward_hours: f64, forward_hours: f64, decay_fn: DecayFunction) -> Self {
        Self {
            backward_hours,
            forward_hours,
            decay_function: decay_fn,
        }
    }

    /// Calculate capture probability based on temporal distance
    ///
    /// Memories closer to the importance event have higher capture probability.
    /// The probability decays with distance according to the configured decay function.
    ///
    /// # Arguments
    /// * `memory_time` - When the memory was encoded
    /// * `event_time` - When the importance event occurred
    ///
    /// # Returns
    /// Capture probability (0.0 to 1.0), or None if outside window
    pub fn capture_probability(
        &self,
        memory_time: DateTime<Utc>,
        event_time: DateTime<Utc>,
    ) -> Option<f64> {
        let diff_hours = (event_time - memory_time).num_minutes() as f64 / 60.0;

        if diff_hours > 0.0 {
            // Memory was encoded before event (backward capture)
            if diff_hours > self.backward_hours {
                return None;
            }
            Some(
                self.decay_function
                    .apply(1.0, diff_hours, self.backward_hours),
            )
        } else {
            // Memory was encoded after event (forward capture)
            let abs_diff = diff_hours.abs();
            if abs_diff > self.forward_hours {
                return None;
            }
            Some(self.decay_function.apply(1.0, abs_diff, self.forward_hours))
        }
    }

    /// Get the start of the capture window
    pub fn window_start(&self, event_time: DateTime<Utc>) -> DateTime<Utc> {
        event_time - Duration::minutes((self.backward_hours * 60.0) as i64)
    }

    /// Get the end of the capture window
    pub fn window_end(&self, event_time: DateTime<Utc>) -> DateTime<Utc> {
        event_time + Duration::minutes((self.forward_hours * 60.0) as i64)
    }

    /// Check if a time is within the capture window
    pub fn is_in_window(&self, memory_time: DateTime<Utc>, event_time: DateTime<Utc>) -> bool {
        memory_time >= self.window_start(event_time) && memory_time <= self.window_end(event_time)
    }
}

// ============================================================================
// IMPORTANCE EVENTS
// ============================================================================

/// Types of events that trigger PRP production
///
/// Each type has different characteristics:
/// - **UserFlag**: Highest priority, explicit user action
/// - **EmotionalContent**: Detected via sentiment analysis
/// - **NoveltySpike**: High prediction error indicates something unexpected
/// - **RepeatedAccess**: Pattern of repeated retrieval
/// - **CrossReference**: Important memory references this one
/// - **TemporalProximity**: Close in time to confirmed important memory
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ImportanceEventType {
    /// Explicit user flag ("remember this", "important")
    UserFlag,
    /// Detected emotional content via sentiment analysis
    EmotionalContent,
    /// High prediction error (novelty detection)
    NoveltySpike,
    /// Memory accessed multiple times in short period
    RepeatedAccess,
    /// Referenced by other important memories
    CrossReference,
    /// Temporally close to confirmed important memory
    TemporalProximity,
}

impl ImportanceEventType {
    /// Get the base PRP strength for this event type
    ///
    /// Different event types have different inherent importance:
    /// - UserFlag has highest strength (explicit user intent)
    /// - NoveltySpike has high strength (surprising = memorable)
    /// - EmotionalContent and RepeatedAccess have medium strength
    /// - CrossReference and TemporalProximity have lower strength (indirect)
    pub fn base_strength(&self) -> f64 {
        match self {
            ImportanceEventType::UserFlag => 1.0,
            ImportanceEventType::NoveltySpike => 0.9,
            ImportanceEventType::EmotionalContent => 0.8,
            ImportanceEventType::RepeatedAccess => 0.75,
            ImportanceEventType::CrossReference => 0.6,
            ImportanceEventType::TemporalProximity => 0.5,
        }
    }

    /// Get the capture radius multiplier
    ///
    /// Some event types should have wider capture windows:
    /// - UserFlag: Standard window (1.0x)
    /// - EmotionalContent: Wider window (1.5x) - emotions spread context
    /// - NoveltySpike: Narrower window (0.7x) - novelty is specific
    pub fn capture_radius_multiplier(&self) -> f64 {
        match self {
            ImportanceEventType::EmotionalContent => 1.5,
            ImportanceEventType::UserFlag => 1.0,
            ImportanceEventType::RepeatedAccess => 1.2,
            ImportanceEventType::CrossReference => 1.0,
            ImportanceEventType::TemporalProximity => 0.8,
            ImportanceEventType::NoveltySpike => 0.7,
        }
    }
}

impl std::fmt::Display for ImportanceEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportanceEventType::UserFlag => write!(f, "user_flag"),
            ImportanceEventType::EmotionalContent => write!(f, "emotional"),
            ImportanceEventType::NoveltySpike => write!(f, "novelty"),
            ImportanceEventType::RepeatedAccess => write!(f, "repeated"),
            ImportanceEventType::CrossReference => write!(f, "cross_ref"),
            ImportanceEventType::TemporalProximity => write!(f, "temporal"),
        }
    }
}

/// An event that triggers PRP production
///
/// When an importance event occurs, the system produces Plasticity-Related
/// Products that can be captured by nearby tagged memories, consolidating
/// them retroactively.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportanceEvent {
    /// Type of importance event
    pub event_type: ImportanceEventType,
    /// Memory that triggered the event (if any)
    pub memory_id: Option<String>,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Event strength (0.0 to 1.0)
    pub strength: f64,
    /// Additional context about the event
    pub context: Option<String>,
}

impl ImportanceEvent {
    /// Create a new importance event
    pub fn new(event_type: ImportanceEventType) -> Self {
        Self {
            event_type,
            memory_id: None,
            timestamp: Utc::now(),
            strength: event_type.base_strength(),
            context: None,
        }
    }

    /// Create with memory ID
    pub fn for_memory(memory_id: &str, event_type: ImportanceEventType) -> Self {
        Self {
            event_type,
            memory_id: Some(memory_id.to_string()),
            timestamp: Utc::now(),
            strength: event_type.base_strength(),
            context: None,
        }
    }

    /// Create with custom strength
    pub fn with_strength(event_type: ImportanceEventType, strength: f64) -> Self {
        Self {
            event_type,
            memory_id: None,
            timestamp: Utc::now(),
            strength: strength.clamp(0.0, 1.0),
            context: None,
        }
    }

    /// Create a user flag event
    pub fn user_flag(memory_id: &str, context: Option<&str>) -> Self {
        Self {
            event_type: ImportanceEventType::UserFlag,
            memory_id: Some(memory_id.to_string()),
            timestamp: Utc::now(),
            strength: 1.0,
            context: context.map(|s| s.to_string()),
        }
    }

    /// Create an emotional content event
    pub fn emotional(memory_id: &str, sentiment_magnitude: f64) -> Self {
        Self {
            event_type: ImportanceEventType::EmotionalContent,
            memory_id: Some(memory_id.to_string()),
            timestamp: Utc::now(),
            strength: sentiment_magnitude.clamp(0.0, 1.0),
            context: None,
        }
    }

    /// Create a novelty spike event
    pub fn novelty(memory_id: &str, prediction_error: f64) -> Self {
        Self {
            event_type: ImportanceEventType::NoveltySpike,
            memory_id: Some(memory_id.to_string()),
            timestamp: Utc::now(),
            strength: prediction_error.clamp(0.0, 1.0),
            context: None,
        }
    }

    /// Create a repeated access event
    pub fn repeated_access(memory_id: &str, access_count: u32) -> Self {
        // Strength scales with access count but caps at 1.0
        let strength = (access_count as f64 / 5.0).min(1.0);
        Self {
            event_type: ImportanceEventType::RepeatedAccess,
            memory_id: Some(memory_id.to_string()),
            timestamp: Utc::now(),
            strength,
            context: Some(format!("{} accesses", access_count)),
        }
    }

    /// Generate unique event ID
    pub fn event_id(&self) -> String {
        format!(
            "{}-{}-{}",
            self.event_type,
            self.timestamp.timestamp_millis(),
            self.memory_id.as_deref().unwrap_or("none")
        )
    }
}

// ============================================================================
// CAPTURED MEMORY
// ============================================================================

/// A memory that was captured (retroactively consolidated)
///
/// This represents the result of successful STC - a previously ordinary memory
/// that has been promoted to long-term storage due to a subsequent importance
/// event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapturedMemory {
    /// The memory that was captured
    pub memory_id: String,
    /// When the memory was originally encoded
    pub encoded_at: DateTime<Utc>,
    /// The event that caused capture
    pub capture_event_id: String,
    /// The type of event that caused capture
    pub capture_event_type: ImportanceEventType,
    /// When the memory was captured
    pub captured_at: DateTime<Utc>,
    /// Capture probability at time of capture
    pub capture_probability: f64,
    /// Tag strength at time of capture
    pub tag_strength_at_capture: f64,
    /// Final consolidated importance score
    pub consolidated_importance: f64,
    /// Temporal distance from trigger event (hours)
    pub temporal_distance_hours: f64,
}

impl CapturedMemory {
    /// Check if this was a backward capture (memory before event)
    pub fn is_backward_capture(&self) -> bool {
        self.temporal_distance_hours > 0.0
    }

    /// Check if this was a forward capture (memory after event)
    pub fn is_forward_capture(&self) -> bool {
        self.temporal_distance_hours < 0.0
    }
}

// ============================================================================
// IMPORTANCE CLUSTER
// ============================================================================

/// A cluster of important memories around a significant moment
///
/// When an importance event occurs, it often captures multiple related memories.
/// These form an "importance cluster" - a group of memories that collectively
/// provide context around a significant moment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportanceCluster {
    /// Unique cluster ID
    pub cluster_id: String,
    /// The triggering importance event
    pub trigger_event_id: String,
    /// Type of the triggering event
    pub trigger_event_type: ImportanceEventType,
    /// Center time of the cluster
    pub center_time: DateTime<Utc>,
    /// Memory IDs in this cluster
    pub memory_ids: Vec<String>,
    /// Average importance of memories in cluster
    pub average_importance: f64,
    /// When this cluster was created
    pub created_at: DateTime<Utc>,
    /// Temporal span of the cluster (hours)
    pub temporal_span_hours: f64,
}

impl ImportanceCluster {
    /// Create a new cluster
    pub fn new(trigger_event: &ImportanceEvent, captured: &[CapturedMemory]) -> Self {
        let memory_ids: Vec<String> = captured.iter().map(|c| c.memory_id.clone()).collect();

        let average_importance = if captured.is_empty() {
            0.0
        } else {
            captured
                .iter()
                .map(|c| c.consolidated_importance)
                .sum::<f64>()
                / captured.len() as f64
        };

        let temporal_span = if captured.len() < 2 {
            0.0
        } else {
            // Safe: captured.len() >= 2 guarantees non-empty iterator
            match (
                captured.iter().map(|c| c.encoded_at).min(),
                captured.iter().map(|c| c.encoded_at).max(),
            ) {
                (Some(min_time), Some(max_time)) => {
                    (max_time - min_time).num_minutes() as f64 / 60.0
                }
                _ => 0.0,
            }
        };

        Self {
            cluster_id: uuid::Uuid::new_v4().to_string(),
            trigger_event_id: trigger_event.event_id(),
            trigger_event_type: trigger_event.event_type,
            center_time: trigger_event.timestamp,
            memory_ids,
            average_importance,
            created_at: Utc::now(),
            temporal_span_hours: temporal_span,
        }
    }

    /// Get the number of memories in this cluster
    pub fn size(&self) -> usize {
        self.memory_ids.len()
    }

    /// Check if a memory is in this cluster
    pub fn contains(&self, memory_id: &str) -> bool {
        self.memory_ids.iter().any(|id| id == memory_id)
    }
}

// ============================================================================
// CAPTURE RESULT
// ============================================================================

/// Result of a PRP trigger operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureResult {
    /// The event that triggered capture
    pub event: ImportanceEvent,
    /// Memories that were captured
    pub captured_memories: Vec<CapturedMemory>,
    /// Tags that were considered but not captured
    pub considered_count: usize,
    /// The importance cluster created (if any)
    pub cluster: Option<ImportanceCluster>,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

impl CaptureResult {
    /// Get the number of captured memories
    pub fn captured_count(&self) -> usize {
        self.captured_memories.len()
    }

    /// Check if any memories were captured
    pub fn has_captures(&self) -> bool {
        !self.captured_memories.is_empty()
    }

    /// Get the capture rate (captured / considered)
    pub fn capture_rate(&self) -> f64 {
        if self.considered_count == 0 {
            return 0.0;
        }
        self.captured_memories.len() as f64 / self.considered_count as f64
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Synaptic Tagging System
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapticTaggingConfig {
    /// Capture window configuration
    pub capture_window: CaptureWindow,
    /// Minimum event strength to trigger PRP production
    pub prp_threshold: f64,
    /// Tag lifetime before complete decay (hours)
    pub tag_lifetime_hours: f64,
    /// Minimum tag strength for capture eligibility
    pub min_tag_strength: f64,
    /// Maximum memories in a single cluster
    pub max_cluster_size: usize,
    /// Whether to create importance clusters
    pub enable_clustering: bool,
    /// Whether to auto-decay tags
    pub auto_decay: bool,
    /// Interval for automatic tag cleanup (hours)
    pub cleanup_interval_hours: f64,
}

impl Default for SynapticTaggingConfig {
    fn default() -> Self {
        Self {
            capture_window: CaptureWindow::default(),
            prp_threshold: DEFAULT_PRP_THRESHOLD,
            tag_lifetime_hours: DEFAULT_TAG_LIFETIME_HOURS,
            min_tag_strength: DEFAULT_MIN_TAG_STRENGTH,
            max_cluster_size: DEFAULT_MAX_CLUSTER_SIZE,
            enable_clustering: true,
            auto_decay: true,
            cleanup_interval_hours: 1.0,
        }
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics about the synaptic tagging system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaggingStats {
    /// Total tags created
    pub total_tags_created: u64,
    /// Currently active tags
    pub active_tags: usize,
    /// Total memories captured
    pub total_captures: u64,
    /// Total importance events processed
    pub total_events: u64,
    /// Total clusters created
    pub total_clusters: u64,
    /// Average capture rate
    pub average_capture_rate: f64,
    /// Average captures per event
    pub average_captures_per_event: f64,
    /// Tags expired without capture
    pub tags_expired: u64,
    /// Last cleanup time
    pub last_cleanup: Option<DateTime<Utc>>,
}

// ============================================================================
// SYNAPTIC TAGGING SYSTEM
// ============================================================================

/// The Synaptic Tagging and Capture (STC) system
///
/// This is the main entry point for retroactive importance assignment.
/// It manages synaptic tags, processes importance events, and captures
/// memories for consolidation.
///
/// ## Thread Safety
///
/// The system is thread-safe and can be shared across threads using Arc.
/// All internal state is protected by RwLock.
///
/// ## Usage
///
/// ```rust,ignore
/// let mut stc = SynapticTaggingSystem::new();
///
/// // Tag memories as they are encoded
/// stc.tag_memory("mem-123");
///
/// // Later, process importance events
/// let result = stc.trigger_prp(ImportanceEvent::user_flag("mem-456", None));
/// for captured in result.captured_memories {
///     // Promote to long-term storage
///     storage.promote_memory(&captured.memory_id, captured.consolidated_importance)?;
/// }
/// ```
pub struct SynapticTaggingSystem {
    /// Active synaptic tags
    tags: Arc<RwLock<HashMap<String, SynapticTag>>>,
    /// Importance clusters
    clusters: Arc<RwLock<Vec<ImportanceCluster>>>,
    /// Configuration
    config: SynapticTaggingConfig,
    /// Statistics
    stats: Arc<RwLock<TaggingStats>>,
}

impl Default for SynapticTaggingSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SynapticTaggingSystem {
    /// Create a new STC system with default configuration
    pub fn new() -> Self {
        Self::with_config(SynapticTaggingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SynapticTaggingConfig) -> Self {
        Self {
            tags: Arc::new(RwLock::new(HashMap::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: Arc::new(RwLock::new(TaggingStats::default())),
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &SynapticTaggingConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: SynapticTaggingConfig) {
        self.config = config;
    }

    /// Tag a memory for potential capture
    ///
    /// This should be called when a memory is encoded. The tag will remain
    /// active for the configured lifetime, eligible for capture if an
    /// importance event occurs nearby.
    ///
    /// # Arguments
    /// * `memory_id` - The ID of the memory to tag
    ///
    /// # Returns
    /// The created synaptic tag
    pub fn tag_memory(&mut self, memory_id: &str) -> SynapticTag {
        let tag = SynapticTag::new(memory_id);

        if let Ok(mut tags) = self.tags.write() {
            tags.insert(memory_id.to_string(), tag.clone());
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.total_tags_created += 1;
            stats.active_tags = self.tags.read().map(|t| t.len()).unwrap_or(0);
        }

        tag
    }

    /// Tag a memory with custom strength
    ///
    /// Use this for memories that have initial importance signals (e.g., emotional content)
    /// but haven't crossed the threshold for full importance yet.
    pub fn tag_memory_with_strength(&mut self, memory_id: &str, strength: f64) -> SynapticTag {
        let tag = SynapticTag::with_strength(memory_id, strength);

        if let Ok(mut tags) = self.tags.write() {
            tags.insert(memory_id.to_string(), tag.clone());
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.total_tags_created += 1;
            stats.active_tags = self.tags.read().map(|t| t.len()).unwrap_or(0);
        }

        tag
    }

    /// Tag a memory with encoding context
    pub fn tag_memory_with_context(&mut self, memory_id: &str, context: &str) -> SynapticTag {
        let tag = SynapticTag::with_context(memory_id, context);

        if let Ok(mut tags) = self.tags.write() {
            tags.insert(memory_id.to_string(), tag.clone());
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.total_tags_created += 1;
            stats.active_tags = self.tags.read().map(|t| t.len()).unwrap_or(0);
        }

        tag
    }

    /// Trigger PRP production from an importance event
    ///
    /// This is the core STC mechanism. When an importance event occurs:
    /// 1. PRPs are produced (if event strength >= threshold)
    /// 2. System sweeps for eligible tagged memories
    /// 3. Eligible memories are captured (consolidated)
    /// 4. Optionally, an importance cluster is created
    ///
    /// # Arguments
    /// * `event` - The importance event
    ///
    /// # Returns
    /// Result containing captured memories and cluster info
    pub fn trigger_prp(&mut self, event: ImportanceEvent) -> CaptureResult {
        let start = std::time::Instant::now();

        // Check if event is strong enough to trigger PRPs
        if event.strength < self.config.prp_threshold {
            return CaptureResult {
                event,
                captured_memories: vec![],
                considered_count: 0,
                cluster: None,
                processing_time_us: start.elapsed().as_micros() as u64,
            };
        }

        // Sweep for eligible tags
        let (captured, considered_count) = self.sweep_for_capture_internal(&event);

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.total_events += 1;
            stats.total_captures += captured.len() as u64;

            // Update rolling average
            let total = stats.total_events as f64;
            stats.average_captures_per_event =
                (stats.average_captures_per_event * (total - 1.0) + captured.len() as f64) / total;

            if considered_count > 0 {
                let rate = captured.len() as f64 / considered_count as f64;
                stats.average_capture_rate =
                    (stats.average_capture_rate * (total - 1.0) + rate) / total;
            }
        }

        // Create cluster if enabled and we have captures
        let cluster = if self.config.enable_clustering && !captured.is_empty() {
            let cluster = ImportanceCluster::new(&event, &captured);

            if let Ok(mut clusters) = self.clusters.write() {
                clusters.push(cluster.clone());
            }

            if let Ok(mut stats) = self.stats.write() {
                stats.total_clusters += 1;
            }

            Some(cluster)
        } else {
            None
        };

        CaptureResult {
            event,
            captured_memories: captured,
            considered_count,
            cluster,
            processing_time_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Internal sweep implementation
    fn sweep_for_capture_internal(
        &mut self,
        event: &ImportanceEvent,
    ) -> (Vec<CapturedMemory>, usize) {
        let mut captured = Vec::new();
        let mut considered = 0;

        // Calculate capture window with event-type-specific multiplier
        let multiplier = event.event_type.capture_radius_multiplier();
        let effective_backward = self.config.capture_window.backward_hours * multiplier;
        let effective_forward = self.config.capture_window.forward_hours * multiplier;

        let effective_window = CaptureWindow::new(effective_backward, effective_forward);

        if let Ok(mut tags) = self.tags.write() {
            let event_id = event.event_id();

            for tag in tags.values_mut() {
                // Skip already captured tags
                if tag.captured {
                    continue;
                }

                // Check if in temporal window
                if !effective_window.is_in_window(tag.created_at, event.timestamp) {
                    continue;
                }

                considered += 1;

                // Calculate current tag strength
                let current_strength = tag.current_strength(
                    self.config.capture_window.decay_function,
                    self.config.tag_lifetime_hours,
                );

                // Check if tag is strong enough
                if current_strength < self.config.min_tag_strength {
                    continue;
                }

                // Calculate capture probability
                let capture_prob = effective_window
                    .capture_probability(tag.created_at, event.timestamp)
                    .unwrap_or(0.0);

                // Check if we should capture (probabilistic based on strength and proximity)
                let capture_score = current_strength * capture_prob * event.strength;

                if capture_score >= self.config.min_tag_strength {
                    // Calculate temporal distance
                    let temporal_distance =
                        (event.timestamp - tag.created_at).num_minutes() as f64 / 60.0;

                    // Calculate consolidated importance
                    let consolidated_importance =
                        (capture_score * 0.6 + event.strength * 0.4).min(1.0);

                    // Mark tag as captured
                    tag.capture(&event_id);

                    captured.push(CapturedMemory {
                        memory_id: tag.memory_id.clone(),
                        encoded_at: tag.created_at,
                        capture_event_id: event_id.clone(),
                        capture_event_type: event.event_type,
                        captured_at: Utc::now(),
                        capture_probability: capture_prob,
                        tag_strength_at_capture: current_strength,
                        consolidated_importance,
                        temporal_distance_hours: temporal_distance,
                    });

                    // Limit cluster size
                    if captured.len() >= self.config.max_cluster_size {
                        break;
                    }
                }
            }
        }

        (captured, considered)
    }

    /// Sweep for capture around a specific time
    ///
    /// Use this when you want to retroactively check for captures without
    /// a specific importance event (e.g., during periodic consolidation).
    ///
    /// # Arguments
    /// * `center_time` - The center time to sweep around
    ///
    /// # Returns
    /// List of memory IDs that could be captured
    pub fn sweep_for_capture(&mut self, center_time: DateTime<Utc>) -> Vec<String> {
        let mut eligible = Vec::new();

        if let Ok(tags) = self.tags.read() {
            for tag in tags.values() {
                if tag.captured {
                    continue;
                }

                if !self
                    .config
                    .capture_window
                    .is_in_window(tag.created_at, center_time)
                {
                    continue;
                }

                let current_strength = tag.current_strength(
                    self.config.capture_window.decay_function,
                    self.config.tag_lifetime_hours,
                );

                if current_strength >= self.config.min_tag_strength {
                    eligible.push(tag.memory_id.clone());
                }
            }
        }

        eligible
    }

    /// Decay all tags and clean up expired ones
    ///
    /// Should be called periodically (e.g., every hour) to:
    /// 1. Update tag strengths based on decay
    /// 2. Remove tags that have decayed below threshold
    /// 3. Remove captured tags that are no longer needed
    pub fn decay_tags(&mut self) {
        let mut expired_count = 0;

        if let Ok(mut tags) = self.tags.write() {
            tags.retain(|_, tag| {
                // Keep captured tags for a while (for reference)
                if tag.captured {
                    // Keep for 24 hours after capture
                    if let Some(captured_at) = tag.captured_at {
                        return (Utc::now() - captured_at).num_hours() < 24;
                    }
                    return false;
                }

                // Check if tag has decayed
                let current_strength = tag.current_strength(
                    self.config.capture_window.decay_function,
                    self.config.tag_lifetime_hours,
                );

                if current_strength < self.config.min_tag_strength * 0.1 {
                    expired_count += 1;
                    return false;
                }

                // Update stored strength
                tag.tag_strength = current_strength;
                true
            });
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.tags_expired += expired_count;
            stats.active_tags = self.tags.read().map(|t| t.len()).unwrap_or(0);
            stats.last_cleanup = Some(Utc::now());
        }
    }

    /// Get a specific tag
    pub fn get_tag(&self, memory_id: &str) -> Option<SynapticTag> {
        self.tags.read().ok()?.get(memory_id).cloned()
    }

    /// Check if a memory has an active tag
    pub fn has_active_tag(&self, memory_id: &str) -> bool {
        self.tags
            .read()
            .ok()
            .and_then(|tags| tags.get(memory_id).cloned())
            .map(|tag| {
                tag.is_active(
                    self.config.capture_window.decay_function,
                    self.config.tag_lifetime_hours,
                    self.config.min_tag_strength,
                )
            })
            .unwrap_or(false)
    }

    /// Check if a memory was captured
    pub fn is_captured(&self, memory_id: &str) -> bool {
        self.tags
            .read()
            .ok()
            .and_then(|tags| tags.get(memory_id).map(|t| t.captured))
            .unwrap_or(false)
    }

    /// Get all active tags
    pub fn get_active_tags(&self) -> Vec<SynapticTag> {
        self.tags
            .read()
            .ok()
            .map(|tags| {
                tags.values()
                    .filter(|tag| {
                        tag.is_active(
                            self.config.capture_window.decay_function,
                            self.config.tag_lifetime_hours,
                            self.config.min_tag_strength,
                        )
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all captured tags
    pub fn get_captured_tags(&self) -> Vec<SynapticTag> {
        self.tags
            .read()
            .ok()
            .map(|tags| tags.values().filter(|tag| tag.captured).cloned().collect())
            .unwrap_or_default()
    }

    /// Get clusters containing a memory
    pub fn get_clusters_for_memory(&self, memory_id: &str) -> Vec<ImportanceCluster> {
        self.clusters
            .read()
            .ok()
            .map(|clusters| {
                clusters
                    .iter()
                    .filter(|c| c.contains(memory_id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all clusters
    pub fn get_all_clusters(&self) -> Vec<ImportanceCluster> {
        self.clusters
            .read()
            .ok()
            .map(|clusters| clusters.clone())
            .unwrap_or_default()
    }

    /// Get statistics
    pub fn stats(&self) -> TaggingStats {
        self.stats
            .read()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Clear all state (for testing)
    pub fn clear(&mut self) {
        if let Ok(mut tags) = self.tags.write() {
            tags.clear();
        }
        if let Ok(mut clusters) = self.clusters.write() {
            clusters.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            *stats = TaggingStats::default();
        }
    }

    /// Get memory IDs that are candidates for capture within a time range
    ///
    /// This is useful for batch processing - you can get candidates first,
    /// then process importance events for them.
    pub fn get_capture_candidates(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<String> {
        self.tags
            .read()
            .ok()
            .map(|tags| {
                tags.values()
                    .filter(|tag| {
                        !tag.captured
                            && tag.created_at >= start
                            && tag.created_at <= end
                            && tag.is_active(
                                self.config.capture_window.decay_function,
                                self.config.tag_lifetime_hours,
                                self.config.min_tag_strength,
                            )
                    })
                    .map(|tag| tag.memory_id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Bulk tag multiple memories
    pub fn tag_memories(&mut self, memory_ids: &[&str]) -> Vec<SynapticTag> {
        memory_ids.iter().map(|id| self.tag_memory(id)).collect()
    }

    /// Process multiple importance events
    pub fn trigger_prp_batch(&mut self, events: Vec<ImportanceEvent>) -> Vec<CaptureResult> {
        events.into_iter().map(|e| self.trigger_prp(e)).collect()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_function_exponential() {
        let decay = DecayFunction::Exponential;

        // At t=0, full strength
        assert!((decay.apply(1.0, 0.0, 12.0) - 1.0).abs() < 0.01);

        // At halfway, significant decay
        let mid = decay.apply(1.0, 6.0, 12.0);
        assert!(mid > 0.0 && mid < 0.5);

        // At lifetime, near zero
        let end = decay.apply(1.0, 12.0, 12.0);
        assert!(end < 0.02);
    }

    #[test]
    fn test_decay_function_linear() {
        let decay = DecayFunction::Linear;

        assert!((decay.apply(1.0, 0.0, 10.0) - 1.0).abs() < 0.01);
        assert!((decay.apply(1.0, 5.0, 10.0) - 0.5).abs() < 0.01);
        assert!((decay.apply(1.0, 10.0, 10.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_synaptic_tag_creation() {
        let tag = SynapticTag::new("mem-123");

        assert_eq!(tag.memory_id, "mem-123");
        assert_eq!(tag.tag_strength, 1.0);
        assert!(!tag.captured);
        assert!(tag.capture_event.is_none());
    }

    #[test]
    fn test_synaptic_tag_capture() {
        let mut tag = SynapticTag::new("mem-123");
        tag.capture("event-456");

        assert!(tag.captured);
        assert_eq!(tag.capture_event.as_deref(), Some("event-456"));
        assert!(tag.captured_at.is_some());
    }

    #[test]
    fn test_capture_window_probability() {
        let window = CaptureWindow::new(9.0, 2.0);
        let event_time = Utc::now();

        // Memory just before event - high probability
        let before = event_time - Duration::hours(1);
        let prob_before = window.capture_probability(before, event_time).unwrap();
        assert!(prob_before > 0.5);

        // Memory long before event - lower probability
        let long_before = event_time - Duration::hours(8);
        let prob_long_before = window.capture_probability(long_before, event_time).unwrap();
        assert!(prob_long_before < prob_before);

        // Memory outside window - None
        let outside = event_time - Duration::hours(10);
        assert!(window.capture_probability(outside, event_time).is_none());
    }

    #[test]
    fn test_importance_event_types() {
        assert!(
            ImportanceEventType::UserFlag.base_strength()
                > ImportanceEventType::TemporalProximity.base_strength()
        );
        assert!(ImportanceEventType::EmotionalContent.capture_radius_multiplier() > 1.0);
        assert!(ImportanceEventType::NoveltySpike.capture_radius_multiplier() < 1.0);
    }

    #[test]
    fn test_synaptic_tagging_system_basic() {
        let mut stc = SynapticTaggingSystem::new();

        // Tag a memory
        let tag = stc.tag_memory("mem-123");
        assert_eq!(tag.memory_id, "mem-123");

        // Check it's active
        assert!(stc.has_active_tag("mem-123"));
        assert!(!stc.is_captured("mem-123"));
    }

    #[test]
    fn test_prp_trigger_captures_tagged_memory() {
        let mut stc = SynapticTaggingSystem::new();

        // Tag a memory
        stc.tag_memory("mem-123");

        // Trigger importance event
        let event = ImportanceEvent::user_flag("mem-456", Some("Remember this!"));
        let result = stc.trigger_prp(event);

        // Should capture the tagged memory
        assert!(result.has_captures());
        assert_eq!(result.captured_memories[0].memory_id, "mem-123");
        assert!(stc.is_captured("mem-123"));
    }

    #[test]
    fn test_weak_event_does_not_trigger_prp() {
        let mut stc = SynapticTaggingSystem::new();
        stc.tag_memory("mem-123");

        // Weak event below threshold
        let event = ImportanceEvent::with_strength(ImportanceEventType::TemporalProximity, 0.3);
        let result = stc.trigger_prp(event);

        assert!(!result.has_captures());
    }

    #[test]
    fn test_clustering() {
        let mut stc = SynapticTaggingSystem::new();

        // Tag multiple memories
        stc.tag_memory("mem-1");
        stc.tag_memory("mem-2");
        stc.tag_memory("mem-3");

        // Trigger event
        let event = ImportanceEvent::user_flag("mem-trigger", None);
        let result = stc.trigger_prp(event);

        // Should create cluster
        assert!(result.cluster.is_some());
        let cluster = result.cluster.unwrap();
        assert!(cluster.size() >= 3);
    }

    #[test]
    fn test_decay_cleans_old_tags() {
        let mut stc = SynapticTaggingSystem::with_config(SynapticTaggingConfig {
            // 0.000003 hours = ~10.8ms (so 100ms sleep should be enough)
            tag_lifetime_hours: 0.000003,
            min_tag_strength: 0.01,
            ..Default::default()
        });

        stc.tag_memory("mem-123");

        // Simulate time passing - sleep much longer than tag lifetime
        std::thread::sleep(std::time::Duration::from_millis(100));

        stc.decay_tags();

        // Check the tag state for debugging
        let tag = stc.get_tag("mem-123");
        let has_active = stc.has_active_tag("mem-123");

        // After sufficient time, tag should be removed OR marked inactive
        // decay_tags removes tags, so get_tag should return None
        assert!(
            tag.is_none() || !has_active,
            "Tag should be cleaned up or inactive. tag={:?}, has_active={}",
            tag.map(|t| t.tag_strength),
            has_active
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut stc = SynapticTaggingSystem::new();

        stc.tag_memory("mem-1");
        stc.tag_memory("mem-2");

        let event = ImportanceEvent::user_flag("trigger", None);
        let _ = stc.trigger_prp(event);

        let stats = stc.stats();
        assert_eq!(stats.total_tags_created, 2);
        assert_eq!(stats.total_events, 1);
        assert!(stats.total_captures >= 2);
    }

    #[test]
    fn test_captured_memory_direction() {
        let captured = CapturedMemory {
            memory_id: "test".to_string(),
            encoded_at: Utc::now() - Duration::hours(2),
            capture_event_id: "event".to_string(),
            capture_event_type: ImportanceEventType::UserFlag,
            captured_at: Utc::now(),
            capture_probability: 0.8,
            tag_strength_at_capture: 0.9,
            consolidated_importance: 0.85,
            temporal_distance_hours: 2.0,
        };

        assert!(captured.is_backward_capture());
        assert!(!captured.is_forward_capture());
    }

    #[test]
    fn test_importance_cluster_creation() {
        let event = ImportanceEvent::user_flag("trigger", None);
        let captured = vec![
            CapturedMemory {
                memory_id: "mem-1".to_string(),
                encoded_at: Utc::now() - Duration::hours(1),
                capture_event_id: "event".to_string(),
                capture_event_type: ImportanceEventType::UserFlag,
                captured_at: Utc::now(),
                capture_probability: 0.8,
                tag_strength_at_capture: 0.9,
                consolidated_importance: 0.85,
                temporal_distance_hours: 1.0,
            },
            CapturedMemory {
                memory_id: "mem-2".to_string(),
                encoded_at: Utc::now() - Duration::hours(2),
                capture_event_id: "event".to_string(),
                capture_event_type: ImportanceEventType::UserFlag,
                captured_at: Utc::now(),
                capture_probability: 0.7,
                tag_strength_at_capture: 0.8,
                consolidated_importance: 0.75,
                temporal_distance_hours: 2.0,
            },
        ];

        let cluster = ImportanceCluster::new(&event, &captured);

        assert_eq!(cluster.size(), 2);
        assert!(cluster.contains("mem-1"));
        assert!(cluster.contains("mem-2"));
        assert!(!cluster.contains("mem-3"));
        assert!(cluster.average_importance > 0.0);
    }

    #[test]
    fn test_batch_operations() {
        let mut stc = SynapticTaggingSystem::new();

        // Bulk tag
        let tags = stc.tag_memories(&["mem-1", "mem-2", "mem-3"]);
        assert_eq!(tags.len(), 3);

        // Batch trigger
        let events = vec![
            ImportanceEvent::user_flag("trigger-1", None),
            ImportanceEvent::emotional("trigger-2", 0.9),
        ];
        let results = stc.trigger_prp_batch(events);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_capture_candidates() {
        let mut stc = SynapticTaggingSystem::new();

        stc.tag_memory("mem-1");
        stc.tag_memory("mem-2");

        let start = Utc::now() - Duration::hours(1);
        let end = Utc::now() + Duration::hours(1);

        let candidates = stc.get_capture_candidates(start, end);
        assert_eq!(candidates.len(), 2);
    }
}
