//! # Prospective Memory
//!
//! Implementation of prospective memory - "remember to do X when Y happens."
//! This is a distinct cognitive system for future intentions, separate from
//! retrospective memory (remembering past events).
//!
//! ## Theoretical Foundation
//!
//! Based on neuroscience research on prospective memory (Einstein & McDaniel, 1990):
//! - **Event-based**: Triggered by external cues (seeing someone, entering a location)
//! - **Time-based**: Triggered by time passing (in 2 hours, at 3pm)
//! - **Activity-based**: Triggered by completing an activity
//!
//! Key cognitive processes:
//! - **Intention formation**: Creating the future intention
//! - **Retention**: Maintaining the intention during delay
//! - **Intention retrieval**: Recognizing the trigger and recalling the intention
//! - **Execution**: Carrying out the intended action
//!
//! ## How It Works
//!
//! 1. **Parse intentions** from natural language or explicit API
//! 2. **Monitor context** continuously for trigger matches
//! 3. **Escalate priority** as deadlines approach
//! 4. **Surface proactively** when triggers are detected
//! 5. **Track fulfillment** and learn from patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! use vestige_core::neuroscience::{ProspectiveMemory, Intention, IntentionTrigger};
//!
//! let mut pm = ProspectiveMemory::new();
//!
//! // Time-based intention
//! pm.create_intention(Intention::new(
//!     "Send weekly report to team",
//!     IntentionTrigger::TimeBased {
//!         at: next_friday_at_3pm,
//!     },
//! ));
//!
//! // Event-based intention
//! pm.create_intention(Intention::new(
//!     "Ask John about the API design",
//!     IntentionTrigger::EventBased {
//!         condition: "meeting with John".to_string(),
//!         pattern: TriggerPattern::Contains("john".to_string()),
//!     },
//! ));
//!
//! // Context-based intention
//! pm.create_intention(Intention::new(
//!     "Review the error handling in payments module",
//!     IntentionTrigger::ContextBased {
//!         context_match: ContextPattern::InCodebase("payments".to_string()),
//!     },
//! ));
//!
//! // Check for triggered intentions
//! let triggered = pm.check_triggers(&current_context);
//! for intention in triggered {
//!     notify_user(&intention);
//! }
//! ```

use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Maximum active intentions to track
const MAX_INTENTIONS: usize = 1000;

/// Default priority escalation threshold (hours before deadline)
const DEFAULT_ESCALATION_THRESHOLD_HOURS: i64 = 24;

/// Maximum times to remind for a single intention
const MAX_REMINDERS_PER_INTENTION: u32 = 5;

/// Minimum interval between reminders (minutes)
const MIN_REMINDER_INTERVAL_MINUTES: i64 = 30;

/// Maximum age for completed intentions in history (days)
const COMPLETED_INTENTION_RETENTION_DAYS: i64 = 30;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur in prospective memory operations
#[derive(Debug, Error)]
pub enum ProspectiveMemoryError {
    /// Failed to create intention
    #[error("Failed to create intention: {0}")]
    IntentionCreation(String),

    /// Intention not found
    #[error("Intention not found: {0}")]
    NotFound(String),

    /// Invalid trigger configuration
    #[error("Invalid trigger: {0}")]
    InvalidTrigger(String),

    /// Failed to parse natural language intention
    #[error("Failed to parse intention: {0}")]
    ParseError(String),

    /// Lock poisoned during concurrent access
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    /// Maximum intentions reached
    #[error("Maximum intentions reached ({0})")]
    MaxIntentionsReached(usize),
}

/// Result type for prospective memory operations
pub type Result<T> = std::result::Result<T, ProspectiveMemoryError>;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Priority levels for intentions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority - nice to remember
    Low = 1,
    /// Normal priority - should remember
    Normal = 2,
    /// High priority - important to remember
    High = 3,
    /// Critical priority - must not forget
    Critical = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl Priority {
    /// Get numeric value for comparison
    pub fn value(&self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Normal => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }

    /// Create from numeric value
    pub fn from_value(value: u8) -> Self {
        match value {
            1 => Self::Low,
            2 => Self::Normal,
            3 => Self::High,
            _ => Self::Critical,
        }
    }

    /// Escalate to next level
    pub fn escalate(&self) -> Self {
        match self {
            Self::Low => Self::Normal,
            Self::Normal => Self::High,
            Self::High => Self::Critical,
            Self::Critical => Self::Critical,
        }
    }
}

/// Status of an intention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionStatus {
    /// Intention is active and being monitored
    Active,
    /// Intention has been triggered but not yet fulfilled
    Triggered,
    /// Intention has been fulfilled
    Fulfilled,
    /// Intention was cancelled
    Cancelled,
    /// Intention expired (deadline passed without fulfillment)
    Expired,
    /// Intention is snoozed until a specific time
    Snoozed,
}

impl Default for IntentionStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// Pattern for matching trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerPattern {
    /// Exact string match
    Exact(String),
    /// Contains substring (case-insensitive)
    Contains(String),
    /// Matches regex pattern
    Regex(String),
    /// Matches any of the given patterns
    AnyOf(Vec<TriggerPattern>),
    /// Matches all of the given patterns
    AllOf(Vec<TriggerPattern>),
}

impl TriggerPattern {
    /// Check if input matches this pattern
    pub fn matches(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();

        match self {
            Self::Exact(s) => input_lower == s.to_lowercase(),
            Self::Contains(s) => input_lower.contains(&s.to_lowercase()),
            Self::Regex(pattern) => {
                // Simple regex matching (in production, use the regex crate)
                input_lower.contains(&pattern.to_lowercase())
            }
            Self::AnyOf(patterns) => patterns.iter().any(|p| p.matches(input)),
            Self::AllOf(patterns) => patterns.iter().all(|p| p.matches(input)),
        }
    }

    /// Create a contains pattern
    pub fn contains(s: impl Into<String>) -> Self {
        Self::Contains(s.into())
    }

    /// Create an exact match pattern
    pub fn exact(s: impl Into<String>) -> Self {
        Self::Exact(s.into())
    }
}

/// Pattern for matching context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextPattern {
    /// Working in a specific codebase/project
    InCodebase(String),
    /// Working with a specific file pattern
    FilePattern(String),
    /// Specific topic/tag is active
    TopicActive(String),
    /// User is in a specific mode (debugging, reviewing, etc.)
    UserMode(String),
    /// Multiple conditions
    Composite {
        /// All conditions that must match
        all: Vec<ContextPattern>,
        /// Any conditions (at least one must match)
        any: Vec<ContextPattern>,
    },
}

impl ContextPattern {
    /// Check if context matches this pattern
    pub fn matches(&self, context: &Context) -> bool {
        match self {
            Self::InCodebase(name) => context
                .project_name
                .as_ref()
                .map(|p| p.to_lowercase().contains(&name.to_lowercase()))
                .unwrap_or(false),
            Self::FilePattern(pattern) => context
                .active_files
                .iter()
                .any(|f| f.to_lowercase().contains(&pattern.to_lowercase())),
            Self::TopicActive(topic) => context
                .active_topics
                .iter()
                .any(|t| t.to_lowercase().contains(&topic.to_lowercase())),
            Self::UserMode(mode) => context
                .user_mode
                .as_ref()
                .map(|m| m.to_lowercase() == mode.to_lowercase())
                .unwrap_or(false),
            Self::Composite { all, any } => {
                let all_match = all.is_empty() || all.iter().all(|p| p.matches(context));
                let any_match = any.is_empty() || any.iter().any(|p| p.matches(context));
                all_match && any_match
            }
        }
    }

    /// Create a codebase pattern
    pub fn in_codebase(name: impl Into<String>) -> Self {
        Self::InCodebase(name.into())
    }

    /// Create a file pattern
    pub fn file_pattern(pattern: impl Into<String>) -> Self {
        Self::FilePattern(pattern.into())
    }

    /// Create a topic pattern
    pub fn topic_active(topic: impl Into<String>) -> Self {
        Self::TopicActive(topic.into())
    }
}

/// Trigger types for intentions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentionTrigger {
    /// Trigger at a specific time
    TimeBased {
        /// The time to trigger
        at: DateTime<Utc>,
    },

    /// Trigger after a duration from creation
    DurationBased {
        /// Duration to wait before triggering
        after: Duration,
        /// Calculated trigger time (set on creation)
        trigger_at: Option<DateTime<Utc>>,
    },

    /// Trigger based on an event/condition
    EventBased {
        /// Description of the condition
        condition: String,
        /// Pattern to match
        pattern: TriggerPattern,
    },

    /// Trigger based on context
    ContextBased {
        /// Context pattern to match
        context_match: ContextPattern,
    },

    /// Trigger when an activity is completed
    ActivityBased {
        /// Activity that must complete
        activity: String,
        /// Pattern to match completion
        completion_pattern: TriggerPattern,
    },

    /// Recurring trigger (repeats)
    Recurring {
        /// Base trigger type
        base: Box<IntentionTrigger>,
        /// Recurrence pattern
        recurrence: RecurrencePattern,
        /// Next occurrence
        next_occurrence: Option<DateTime<Utc>>,
    },

    /// Compound trigger (multiple conditions)
    Compound {
        /// All triggers that must fire
        all_of: Vec<IntentionTrigger>,
        /// Any triggers (at least one must fire)
        any_of: Vec<IntentionTrigger>,
    },
}

impl IntentionTrigger {
    /// Create a time-based trigger
    pub fn at_time(time: DateTime<Utc>) -> Self {
        Self::TimeBased { at: time }
    }

    /// Create a duration-based trigger
    pub fn after_duration(duration: Duration) -> Self {
        Self::DurationBased {
            after: duration,
            trigger_at: Some(Utc::now() + duration),
        }
    }

    /// Create an event-based trigger
    pub fn on_event(condition: impl Into<String>, pattern: TriggerPattern) -> Self {
        Self::EventBased {
            condition: condition.into(),
            pattern,
        }
    }

    /// Create a context-based trigger
    pub fn on_context(context_match: ContextPattern) -> Self {
        Self::ContextBased { context_match }
    }

    /// Check if this trigger matches the current state
    pub fn is_triggered(&self, context: &Context, events: &[String]) -> bool {
        let now = Utc::now();

        match self {
            Self::TimeBased { at } => now >= *at,
            Self::DurationBased { trigger_at, .. } => trigger_at.map(|t| now >= t).unwrap_or(false),
            Self::EventBased { pattern, .. } => events.iter().any(|e| pattern.matches(e)),
            Self::ContextBased { context_match } => context_match.matches(context),
            Self::ActivityBased {
                completion_pattern, ..
            } => events.iter().any(|e| completion_pattern.matches(e)),
            Self::Recurring {
                next_occurrence, ..
            } => next_occurrence.map(|t| now >= t).unwrap_or(false),
            Self::Compound { all_of, any_of } => {
                let all_match =
                    all_of.is_empty() || all_of.iter().all(|t| t.is_triggered(context, events));
                let any_match =
                    any_of.is_empty() || any_of.iter().any(|t| t.is_triggered(context, events));
                all_match && any_match
            }
        }
    }

    /// Get a human-readable description of the trigger
    pub fn description(&self) -> String {
        match self {
            Self::TimeBased { at } => format!("At {}", at.format("%Y-%m-%d %H:%M")),
            Self::DurationBased { after, .. } => {
                let hours = after.num_hours();
                let minutes = after.num_minutes() % 60;
                if hours > 0 {
                    format!("In {} hours {} minutes", hours, minutes)
                } else {
                    format!("In {} minutes", minutes)
                }
            }
            Self::EventBased { condition, .. } => format!("When: {}", condition),
            Self::ContextBased { context_match } => match context_match {
                ContextPattern::InCodebase(name) => format!("In {} codebase", name),
                ContextPattern::FilePattern(pattern) => format!("Working on {}", pattern),
                ContextPattern::TopicActive(topic) => format!("Discussing {}", topic),
                ContextPattern::UserMode(mode) => format!("In {} mode", mode),
                ContextPattern::Composite { .. } => "Complex context".to_string(),
            },
            Self::ActivityBased { activity, .. } => format!("After completing: {}", activity),
            Self::Recurring {
                base, recurrence, ..
            } => {
                format!("{} ({})", base.description(), recurrence.description())
            }
            Self::Compound { all_of, any_of } => {
                let parts: Vec<String> = all_of
                    .iter()
                    .chain(any_of.iter())
                    .map(|t| t.description())
                    .collect();
                parts.join(" and ")
            }
        }
    }
}

/// Recurrence patterns for recurring intentions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurrencePattern {
    /// Every N minutes
    EveryMinutes(i64),
    /// Every N hours
    EveryHours(i64),
    /// Daily at specific time
    Daily { hour: u32, minute: u32 },
    /// Weekly on specific days
    Weekly {
        days: Vec<chrono::Weekday>,
        hour: u32,
        minute: u32,
    },
    /// Monthly on specific day
    Monthly { day: u32, hour: u32, minute: u32 },
    /// Custom interval
    Custom { interval: Duration },
}

impl RecurrencePattern {
    /// Get the next occurrence from a given time
    pub fn next_occurrence(&self, from: DateTime<Utc>) -> DateTime<Utc> {
        match self {
            Self::EveryMinutes(mins) => from + Duration::minutes(*mins),
            Self::EveryHours(hours) => from + Duration::hours(*hours),
            Self::Daily { hour, minute } => {
                let today = from.date_naive();
                // Default to midnight if invalid time (00:00:00 is always valid)
                let time = chrono::NaiveTime::from_hms_opt(*hour, *minute, 0)
                    .unwrap_or(chrono::NaiveTime::MIN);
                let datetime = today.and_time(time);
                let result = DateTime::<Utc>::from_naive_utc_and_offset(datetime, Utc);

                if result <= from {
                    result + Duration::days(1)
                } else {
                    result
                }
            }
            Self::Weekly { days, hour, minute } => {
                // Find next matching day
                let mut candidate = from + Duration::days(1);
                for _ in 0..7 {
                    if days.contains(&candidate.weekday()) {
                        let date = candidate.date_naive();
                        // Default to midnight if invalid time
                        let time = chrono::NaiveTime::from_hms_opt(*hour, *minute, 0)
                            .unwrap_or(chrono::NaiveTime::MIN);
                        return DateTime::<Utc>::from_naive_utc_and_offset(
                            date.and_time(time),
                            Utc,
                        );
                    }
                    candidate = candidate + Duration::days(1);
                }
                from + Duration::days(7) // Fallback
            }
            Self::Monthly { day, hour, minute } => {
                let current_month = from.month();
                let current_year = from.year();

                let target_date = chrono::NaiveDate::from_ymd_opt(
                    current_year,
                    current_month,
                    (*day).min(28), // Safe day
                )
                .unwrap_or_else(|| from.date_naive());

                // Default to midnight if invalid time
                let time = chrono::NaiveTime::from_hms_opt(*hour, *minute, 0)
                    .unwrap_or(chrono::NaiveTime::MIN);

                let result =
                    DateTime::<Utc>::from_naive_utc_and_offset(target_date.and_time(time), Utc);

                if result <= from {
                    // Go to next month
                    let next_month = if current_month == 12 {
                        1
                    } else {
                        current_month + 1
                    };
                    let next_year = if current_month == 12 {
                        current_year + 1
                    } else {
                        current_year
                    };

                    let next_date =
                        chrono::NaiveDate::from_ymd_opt(next_year, next_month, (*day).min(28))
                            .unwrap_or_else(|| from.date_naive());

                    DateTime::<Utc>::from_naive_utc_and_offset(next_date.and_time(time), Utc)
                } else {
                    result
                }
            }
            Self::Custom { interval } => from + *interval,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            Self::EveryMinutes(mins) => format!("every {} minutes", mins),
            Self::EveryHours(hours) => format!("every {} hours", hours),
            Self::Daily { hour, minute } => format!("daily at {:02}:{:02}", hour, minute),
            Self::Weekly { days, hour, minute } => {
                let day_names: Vec<_> = days.iter().map(|d| format!("{:?}", d)).collect();
                format!(
                    "every {} at {:02}:{:02}",
                    day_names.join(", "),
                    hour,
                    minute
                )
            }
            Self::Monthly { day, hour, minute } => {
                format!("monthly on day {} at {:02}:{:02}", day, hour, minute)
            }
            Self::Custom { interval } => format!("every {} minutes", interval.num_minutes()),
        }
    }
}

/// A future intention to be remembered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Unique identifier
    pub id: String,
    /// What to remember to do
    pub content: String,
    /// When/how to trigger
    pub trigger: IntentionTrigger,
    /// Priority level
    pub priority: Priority,
    /// Current status
    pub status: IntentionStatus,
    /// When the intention was created
    pub created_at: DateTime<Utc>,
    /// Optional deadline
    pub deadline: Option<DateTime<Utc>>,
    /// When the intention was fulfilled (if fulfilled)
    pub fulfilled_at: Option<DateTime<Utc>>,
    /// Number of times this has been reminded
    pub reminder_count: u32,
    /// Last reminder time
    pub last_reminded_at: Option<DateTime<Utc>>,
    /// Optional notes/context
    pub notes: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Related memory IDs
    pub related_memories: Vec<String>,
    /// Snoozed until (if snoozed)
    pub snoozed_until: Option<DateTime<Utc>>,
    /// Source of the intention (natural language, API, etc.)
    pub source: IntentionSource,
}

impl Intention {
    /// Create a new intention
    pub fn new(content: impl Into<String>, trigger: IntentionTrigger) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: content.into(),
            trigger,
            priority: Priority::Normal,
            status: IntentionStatus::Active,
            created_at: Utc::now(),
            deadline: None,
            fulfilled_at: None,
            reminder_count: 0,
            last_reminded_at: None,
            notes: None,
            tags: Vec::new(),
            related_memories: Vec::new(),
            snoozed_until: None,
            source: IntentionSource::Api,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add related memory
    pub fn with_related_memory(mut self, memory_id: String) -> Self {
        self.related_memories.push(memory_id);
        self
    }

    /// Check if the intention is overdue
    pub fn is_overdue(&self) -> bool {
        self.deadline.map(|d| Utc::now() > d).unwrap_or(false)
    }

    /// Check if deadline is approaching
    pub fn is_deadline_approaching(&self, threshold: Duration) -> bool {
        self.deadline
            .map(|d| {
                let now = Utc::now();
                now < d && (d - now) < threshold
            })
            .unwrap_or(false)
    }

    /// Check if should remind again
    pub fn should_remind(&self) -> bool {
        if self.status != IntentionStatus::Active && self.status != IntentionStatus::Triggered {
            return false;
        }

        if self.reminder_count >= MAX_REMINDERS_PER_INTENTION {
            return false;
        }

        // Check snoozed
        if let Some(snoozed_until) = self.snoozed_until {
            if Utc::now() < snoozed_until {
                return false;
            }
        }

        // Check minimum interval
        if let Some(last) = self.last_reminded_at {
            if (Utc::now() - last) < Duration::minutes(MIN_REMINDER_INTERVAL_MINUTES) {
                return false;
            }
        }

        true
    }

    /// Mark as triggered
    pub fn mark_triggered(&mut self) {
        self.status = IntentionStatus::Triggered;
        self.reminder_count += 1;
        self.last_reminded_at = Some(Utc::now());
    }

    /// Mark as fulfilled
    pub fn mark_fulfilled(&mut self) {
        self.status = IntentionStatus::Fulfilled;
        self.fulfilled_at = Some(Utc::now());
    }

    /// Snooze for a duration
    pub fn snooze(&mut self, duration: Duration) {
        self.status = IntentionStatus::Snoozed;
        self.snoozed_until = Some(Utc::now() + duration);
    }

    /// Wake from snooze
    pub fn wake(&mut self) {
        if self.status == IntentionStatus::Snoozed {
            self.status = IntentionStatus::Active;
            self.snoozed_until = None;
        }
    }

    /// Get effective priority (accounting for deadline proximity)
    pub fn effective_priority(&self) -> Priority {
        let mut priority = self.priority;

        // Escalate if deadline is approaching
        if self.is_deadline_approaching(Duration::hours(1)) {
            priority = priority.escalate().escalate();
        } else if self.is_deadline_approaching(Duration::hours(DEFAULT_ESCALATION_THRESHOLD_HOURS))
        {
            priority = priority.escalate();
        }

        // Escalate if overdue
        if self.is_overdue() {
            priority = Priority::Critical;
        }

        priority
    }
}

/// Source of an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentionSource {
    /// Created via API
    Api,
    /// Parsed from natural language
    NaturalLanguage {
        /// Original text
        original_text: String,
        /// Confidence in parsing
        confidence: f64,
    },
    /// Inferred from user behavior
    Inferred {
        /// What triggered inference
        trigger: String,
        /// Confidence in inference
        confidence: f64,
    },
    /// Imported from external system
    Imported {
        /// Source system
        source: String,
    },
}

// ============================================================================
// CONTEXT
// ============================================================================

/// Current context for trigger matching
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// Current time
    pub timestamp: DateTime<Utc>,
    /// Current project name
    pub project_name: Option<String>,
    /// Current project path
    pub project_path: Option<String>,
    /// Active files being worked on
    pub active_files: Vec<String>,
    /// Active topics/tags
    pub active_topics: Vec<String>,
    /// Current user mode (debugging, reviewing, etc.)
    pub user_mode: Option<String>,
    /// Recent events (for event-based triggers)
    pub recent_events: Vec<String>,
    /// People/entities mentioned recently
    pub mentioned_entities: Vec<String>,
    /// Current conversation context
    pub conversation_context: Option<String>,
}

impl Context {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            ..Default::default()
        }
    }

    /// Set project
    pub fn with_project(mut self, name: impl Into<String>, path: impl Into<String>) -> Self {
        self.project_name = Some(name.into());
        self.project_path = Some(path.into());
        self
    }

    /// Add active file
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.active_files.push(file.into());
        self
    }

    /// Add topic
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.active_topics.push(topic.into());
        self
    }

    /// Set user mode
    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.user_mode = Some(mode.into());
        self
    }

    /// Add event
    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.recent_events.push(event.into());
        self
    }

    /// Add mentioned entity
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.mentioned_entities.push(entity.into());
        self
    }
}

/// Context monitor for checking triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMonitor {
    /// IDs of intentions currently being monitored
    pub active_intentions: Vec<String>,
    /// Current context snapshot
    pub current_context: Context,
    /// Last check time
    pub last_check: DateTime<Utc>,
}

impl Default for ContextMonitor {
    fn default() -> Self {
        Self {
            active_intentions: Vec::new(),
            current_context: Context::new(),
            last_check: Utc::now(),
        }
    }
}

impl ContextMonitor {
    /// Create a new context monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the current context
    pub fn update_context(&mut self, context: Context) {
        self.current_context = context;
        self.last_check = Utc::now();
    }
}

// ============================================================================
// NATURAL LANGUAGE PARSING
// ============================================================================

/// Parser for natural language intentions
pub struct IntentionParser {
    /// Time-related keywords
    time_keywords: HashMap<String, Duration>,
}

impl IntentionParser {
    /// Create a new intention parser
    pub fn new() -> Self {
        let mut time_keywords = HashMap::new();

        // Duration keywords
        time_keywords.insert("in a minute".to_string(), Duration::minutes(1));
        time_keywords.insert("in 5 minutes".to_string(), Duration::minutes(5));
        time_keywords.insert("in 10 minutes".to_string(), Duration::minutes(10));
        time_keywords.insert("in 15 minutes".to_string(), Duration::minutes(15));
        time_keywords.insert("in 30 minutes".to_string(), Duration::minutes(30));
        time_keywords.insert("in an hour".to_string(), Duration::hours(1));
        time_keywords.insert("in 2 hours".to_string(), Duration::hours(2));
        time_keywords.insert("tomorrow".to_string(), Duration::hours(24));
        time_keywords.insert("next week".to_string(), Duration::days(7));

        Self { time_keywords }
    }

    /// Parse a natural language intention
    pub fn parse(&self, text: &str) -> Result<Intention> {
        let text_lower = text.to_lowercase();

        // Detect trigger type and extract content
        let (trigger, content) = self.extract_trigger_and_content(&text_lower, text)?;

        let mut intention = Intention::new(content, trigger);
        intention.source = IntentionSource::NaturalLanguage {
            original_text: text.to_string(),
            confidence: 0.7, // Base confidence for pattern matching
        };

        // Detect priority from keywords
        if text_lower.contains("urgent")
            || text_lower.contains("important")
            || text_lower.contains("critical")
            || text_lower.contains("asap")
        {
            intention.priority = Priority::High;
        }

        Ok(intention)
    }

    /// Extract trigger and content from text
    fn extract_trigger_and_content(
        &self,
        text_lower: &str,
        original: &str,
    ) -> Result<(IntentionTrigger, String)> {
        // Check for "remind me to X when Y" pattern
        if let Some(when_byte_idx) = text_lower.find(" when ") {
            // Convert byte index to char index for safe slicing
            let when_char_idx = text_lower[..when_byte_idx].chars().count();

            let content_part: String = if text_lower.starts_with("remind me to ") {
                original.chars().skip(13).take(when_char_idx.saturating_sub(13)).collect()
            } else if text_lower.starts_with("remind me ") {
                original.chars().skip(10).take(when_char_idx.saturating_sub(10)).collect()
            } else {
                original.chars().take(when_char_idx).collect()
            };

            let condition_part: String = original.chars().skip(when_char_idx + 6).collect();

            return Ok((
                IntentionTrigger::EventBased {
                    condition: condition_part.clone(),
                    pattern: TriggerPattern::contains(&condition_part),
                },
                content_part,
            ));
        }

        // Check for time-based patterns
        for (keyword, duration) in &self.time_keywords {
            if text_lower.contains(keyword) {
                let content = self.extract_content(text_lower, original, keyword);
                return Ok((IntentionTrigger::after_duration(*duration), content));
            }
        }

        // Check for "at X" time pattern
        if text_lower.contains(" at ") {
            // For now, treat as a simple event trigger
            let parts: Vec<&str> = original.splitn(2, " at ").collect();
            if parts.len() == 2 {
                let part0_lower = parts[0].to_lowercase();
                let content: String = if part0_lower.starts_with("remind me to ") {
                    parts[0].chars().skip(13).collect()
                } else if part0_lower.starts_with("remind me ") {
                    parts[0].chars().skip(10).collect()
                } else {
                    parts[0].to_string()
                };

                // Try to parse time (simplified - just use duration for now)
                return Ok((
                    IntentionTrigger::after_duration(Duration::hours(1)),
                    content,
                ));
            }
        }

        // Check for implicit intentions ("I should tell Sarah about this")
        if text_lower.starts_with("i should ")
            || text_lower.starts_with("i need to ")
            || text_lower.starts_with("don't forget to ")
            || text_lower.starts_with("remember to ")
        {
            // Use char-aware slicing to avoid UTF-8 boundary issues
            let content: String = if text_lower.starts_with("i should ") {
                original.chars().skip(9).collect()
            } else if text_lower.starts_with("i need to ") {
                original.chars().skip(10).collect()
            } else if text_lower.starts_with("don't forget to ") {
                original.chars().skip(16).collect()
            } else {
                original.chars().skip(12).collect()
            };

            // Extract entity if mentioned
            if let Some(entity) = self.extract_entity(&text_lower) {
                return Ok((
                    IntentionTrigger::EventBased {
                        condition: format!("Meeting or conversation with {}", entity),
                        pattern: TriggerPattern::contains(&entity),
                    },
                    content,
                ));
            }

            // Default to time-based
            return Ok((
                IntentionTrigger::after_duration(Duration::hours(1)),
                content,
            ));
        }

        // Default fallback
        Err(ProspectiveMemoryError::ParseError(
            "Could not parse intention from text".to_string(),
        ))
    }

    /// Extract content from text, removing trigger keywords
    fn extract_content(&self, _text_lower: &str, original: &str, keyword: &str) -> String {
        let content = original
            .replace(keyword, "")
            .replace(&keyword.to_uppercase(), "")
            .replace("remind me to ", "")
            .replace("Remind me to ", "")
            .replace("remind me ", "")
            .replace("Remind me ", "")
            .trim()
            .to_string();

        content
    }

    /// Extract entity names from text
    fn extract_entity(&self, text_lower: &str) -> Option<String> {
        // Simple pattern: look for "tell X about" or "ask X about" or "with X"
        let patterns = ["tell ", "ask ", "with ", "to "];

        for pattern in patterns {
            if let Some(idx) = text_lower.find(pattern) {
                let after = &text_lower[idx + pattern.len()..];
                // Get first word as entity
                if let Some(space_idx) = after.find(' ') {
                    let entity = &after[..space_idx];
                    if !["the", "a", "an", "about", "to", "for"].contains(&entity) {
                        return Some(entity.to_string());
                    }
                }
            }
        }

        None
    }
}

impl Default for IntentionParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PROSPECTIVE MEMORY ENGINE
// ============================================================================

/// Configuration for the prospective memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemoryConfig {
    /// Maximum active intentions
    pub max_intentions: usize,
    /// Enable priority escalation
    pub enable_escalation: bool,
    /// Hours before deadline to start escalation
    pub escalation_threshold_hours: i64,
    /// Maximum reminders per intention
    pub max_reminders: u32,
    /// Minimum minutes between reminders
    pub min_reminder_interval_minutes: i64,
    /// Auto-expire intentions after deadline
    pub auto_expire: bool,
    /// Days to retain completed intentions
    pub completed_retention_days: i64,
}

impl Default for ProspectiveMemoryConfig {
    fn default() -> Self {
        Self {
            max_intentions: MAX_INTENTIONS,
            enable_escalation: true,
            escalation_threshold_hours: DEFAULT_ESCALATION_THRESHOLD_HOURS,
            max_reminders: MAX_REMINDERS_PER_INTENTION,
            min_reminder_interval_minutes: MIN_REMINDER_INTERVAL_MINUTES,
            auto_expire: true,
            completed_retention_days: COMPLETED_INTENTION_RETENTION_DAYS,
        }
    }
}

/// The main prospective memory engine
pub struct ProspectiveMemory {
    /// Active intentions
    intentions: Arc<RwLock<HashMap<String, Intention>>>,
    /// Context monitors
    monitors: Arc<RwLock<Vec<ContextMonitor>>>,
    /// Natural language parser
    parser: IntentionParser,
    /// Configuration
    config: ProspectiveMemoryConfig,
    /// History of fulfilled intentions (for learning)
    history: Arc<RwLock<VecDeque<Intention>>>,
}

impl ProspectiveMemory {
    /// Create a new prospective memory engine
    pub fn new() -> Self {
        Self::with_config(ProspectiveMemoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ProspectiveMemoryConfig) -> Self {
        Self {
            intentions: Arc::new(RwLock::new(HashMap::new())),
            monitors: Arc::new(RwLock::new(vec![ContextMonitor::new()])),
            parser: IntentionParser::new(),
            config,
            history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ProspectiveMemoryConfig {
        &self.config
    }

    /// Create a new intention
    pub fn create_intention(&self, intention: Intention) -> Result<String> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        // Check capacity
        if intentions.len() >= self.config.max_intentions {
            return Err(ProspectiveMemoryError::MaxIntentionsReached(
                self.config.max_intentions,
            ));
        }

        let id = intention.id.clone();
        intentions.insert(id.clone(), intention);

        Ok(id)
    }

    /// Create intention from natural language
    pub fn create_from_text(&self, text: &str) -> Result<String> {
        let intention = self.parser.parse(text)?;
        self.create_intention(intention)
    }

    /// Get an intention by ID
    pub fn get_intention(&self, id: &str) -> Result<Option<Intention>> {
        let intentions = self
            .intentions
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(intentions.get(id).cloned())
    }

    /// Get all active intentions
    pub fn get_active_intentions(&self) -> Result<Vec<Intention>> {
        let intentions = self
            .intentions
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(intentions
            .values()
            .filter(|i| {
                i.status == IntentionStatus::Active || i.status == IntentionStatus::Triggered
            })
            .cloned()
            .collect())
    }

    /// Get intentions by priority
    pub fn get_by_priority(&self, min_priority: Priority) -> Result<Vec<Intention>> {
        let intentions = self
            .intentions
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        let mut result: Vec<_> = intentions
            .values()
            .filter(|i| i.effective_priority() >= min_priority)
            .filter(|i| {
                i.status == IntentionStatus::Active || i.status == IntentionStatus::Triggered
            })
            .cloned()
            .collect();

        // Sort by effective priority (highest first)
        result.sort_by(|a, b| b.effective_priority().cmp(&a.effective_priority()));

        Ok(result)
    }

    /// Get overdue intentions
    pub fn get_overdue(&self) -> Result<Vec<Intention>> {
        let intentions = self
            .intentions
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(intentions
            .values()
            .filter(|i| i.is_overdue())
            .filter(|i| {
                i.status == IntentionStatus::Active || i.status == IntentionStatus::Triggered
            })
            .cloned()
            .collect())
    }

    /// Check triggers against current context
    pub fn check_triggers(&self, context: &Context) -> Result<Vec<Intention>> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        let mut triggered = Vec::new();

        for intention in intentions.values_mut() {
            // Skip non-active intentions
            if intention.status != IntentionStatus::Active {
                // Check if snoozed intention should wake
                if intention.status == IntentionStatus::Snoozed {
                    if let Some(until) = intention.snoozed_until {
                        if Utc::now() >= until {
                            intention.wake();
                        }
                    }
                }
                continue;
            }

            // Check if triggered
            if intention
                .trigger
                .is_triggered(context, &context.recent_events)
            {
                if intention.should_remind() {
                    intention.mark_triggered();
                    triggered.push(intention.clone());
                }
            }

            // Check for deadline escalation
            if self.config.enable_escalation {
                let threshold = Duration::hours(self.config.escalation_threshold_hours);
                if intention.is_deadline_approaching(threshold) {
                    // Priority will be automatically escalated via effective_priority()
                }
            }

            // Auto-expire overdue intentions
            if self.config.auto_expire && intention.is_overdue() {
                intention.status = IntentionStatus::Expired;
            }
        }

        // Sort triggered by effective priority
        triggered.sort_by(|a, b| b.effective_priority().cmp(&a.effective_priority()));

        Ok(triggered)
    }

    /// Update context and check for triggers
    pub fn update_context(&self, context: Context) -> Result<Vec<Intention>> {
        // Update monitors
        {
            let mut monitors = self
                .monitors
                .write()
                .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

            if let Some(monitor) = monitors.first_mut() {
                monitor.update_context(context.clone());
            }
        }

        // Check triggers
        self.check_triggers(&context)
    }

    /// Mark intention as fulfilled
    pub fn fulfill(&self, id: &str) -> Result<()> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        if let Some(intention) = intentions.get_mut(id) {
            intention.mark_fulfilled();

            // Add to history
            let fulfilled_intention = intention.clone();

            let mut history = self
                .history
                .write()
                .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

            history.push_back(fulfilled_intention);

            // Maintain history size
            let retention_cutoff =
                Utc::now() - Duration::days(self.config.completed_retention_days);
            while history
                .front()
                .map(|i| i.fulfilled_at.unwrap_or(i.created_at) < retention_cutoff)
                .unwrap_or(false)
            {
                history.pop_front();
            }

            Ok(())
        } else {
            Err(ProspectiveMemoryError::NotFound(id.to_string()))
        }
    }

    /// Snooze an intention
    pub fn snooze(&self, id: &str, duration: Duration) -> Result<()> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        if let Some(intention) = intentions.get_mut(id) {
            intention.snooze(duration);
            Ok(())
        } else {
            Err(ProspectiveMemoryError::NotFound(id.to_string()))
        }
    }

    /// Cancel an intention
    pub fn cancel(&self, id: &str) -> Result<()> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        if let Some(intention) = intentions.get_mut(id) {
            intention.status = IntentionStatus::Cancelled;
            Ok(())
        } else {
            Err(ProspectiveMemoryError::NotFound(id.to_string()))
        }
    }

    /// Update intention priority
    pub fn set_priority(&self, id: &str, priority: Priority) -> Result<()> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        if let Some(intention) = intentions.get_mut(id) {
            intention.priority = priority;
            Ok(())
        } else {
            Err(ProspectiveMemoryError::NotFound(id.to_string()))
        }
    }

    /// Get intention statistics
    pub fn stats(&self) -> Result<IntentionStats> {
        let intentions = self
            .intentions
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        let history = self
            .history
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        let active = intentions
            .values()
            .filter(|i| i.status == IntentionStatus::Active)
            .count();

        let triggered = intentions
            .values()
            .filter(|i| i.status == IntentionStatus::Triggered)
            .count();

        let overdue = intentions.values().filter(|i| i.is_overdue()).count();

        let fulfilled = history.len();

        let high_priority = intentions
            .values()
            .filter(|i| i.effective_priority() >= Priority::High)
            .filter(|i| {
                i.status == IntentionStatus::Active || i.status == IntentionStatus::Triggered
            })
            .count();

        Ok(IntentionStats {
            total_active: active,
            triggered: triggered,
            overdue: overdue,
            fulfilled_lifetime: fulfilled,
            high_priority: high_priority,
        })
    }

    /// Clean up old/completed intentions
    pub fn cleanup(&self) -> Result<usize> {
        let mut intentions = self
            .intentions
            .write()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        let before = intentions.len();

        // Remove fulfilled, cancelled, and expired intentions
        intentions.retain(|_, i| {
            matches!(
                i.status,
                IntentionStatus::Active | IntentionStatus::Triggered | IntentionStatus::Snoozed
            )
        });

        Ok(before - intentions.len())
    }

    /// Get fulfillment history
    pub fn get_history(&self, limit: usize) -> Result<Vec<Intention>> {
        let history = self
            .history
            .read()
            .map_err(|e| ProspectiveMemoryError::LockPoisoned(e.to_string()))?;

        Ok(history.iter().rev().take(limit).cloned().collect())
    }
}

impl Default for ProspectiveMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about intentions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionStats {
    /// Number of active intentions
    pub total_active: usize,
    /// Number of triggered (pending action) intentions
    pub triggered: usize,
    /// Number of overdue intentions
    pub overdue: usize,
    /// Total fulfilled in history
    pub fulfilled_lifetime: usize,
    /// High priority intentions needing attention
    pub high_priority: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_escalation() {
        assert_eq!(Priority::Low.escalate(), Priority::Normal);
        assert_eq!(Priority::Normal.escalate(), Priority::High);
        assert_eq!(Priority::High.escalate(), Priority::Critical);
        assert_eq!(Priority::Critical.escalate(), Priority::Critical);
    }

    #[test]
    fn test_trigger_pattern_matches() {
        let pattern = TriggerPattern::contains("john");
        assert!(pattern.matches("Meeting with John"));
        assert!(pattern.matches("john's project"));
        assert!(!pattern.matches("Meeting with Jane"));
    }

    #[test]
    fn test_trigger_pattern_any_of() {
        let pattern = TriggerPattern::AnyOf(vec![
            TriggerPattern::contains("john"),
            TriggerPattern::contains("jane"),
        ]);

        assert!(pattern.matches("Meeting with John"));
        assert!(pattern.matches("Meeting with Jane"));
        assert!(!pattern.matches("Meeting with Bob"));
    }

    #[test]
    fn test_context_pattern_matches() {
        let context = Context::new()
            .with_project("payments-service", "/code/payments")
            .with_file("/code/payments/src/auth.rs")
            .with_topic("authentication");

        let pattern = ContextPattern::in_codebase("payments");
        assert!(pattern.matches(&context));

        let pattern = ContextPattern::topic_active("auth");
        assert!(pattern.matches(&context));
    }

    #[test]
    fn test_intention_creation() {
        let intention = Intention::new(
            "Review code",
            IntentionTrigger::after_duration(Duration::hours(1)),
        )
        .with_priority(Priority::High)
        .with_tags(vec!["code-review".to_string()]);

        assert_eq!(intention.priority, Priority::High);
        assert!(!intention.tags.is_empty());
        assert_eq!(intention.status, IntentionStatus::Active);
    }

    #[test]
    fn test_time_trigger() {
        let trigger = IntentionTrigger::at_time(Utc::now() - Duration::hours(1));
        let context = Context::new();

        assert!(trigger.is_triggered(&context, &[]));
    }

    #[test]
    fn test_duration_trigger() {
        let trigger = IntentionTrigger::after_duration(Duration::seconds(-1));
        let context = Context::new();

        assert!(trigger.is_triggered(&context, &[]));
    }

    #[test]
    fn test_event_trigger() {
        let trigger =
            IntentionTrigger::on_event("Meeting with John", TriggerPattern::contains("john"));
        let context = Context::new();

        assert!(!trigger.is_triggered(&context, &[]));
        assert!(trigger.is_triggered(&context, &["Scheduled meeting with John".to_string()]));
    }

    #[test]
    fn test_prospective_memory_create() {
        let pm = ProspectiveMemory::new();

        let intention = Intention::new(
            "Test intention",
            IntentionTrigger::after_duration(Duration::hours(1)),
        );

        let id = pm.create_intention(intention).unwrap();
        assert!(!id.is_empty());

        let retrieved = pm.get_intention(&id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_parse_natural_language() {
        let parser = IntentionParser::new();

        // Test "remind me to X in Y" pattern
        let result = parser.parse("remind me to check email in 30 minutes");
        assert!(result.is_ok());

        // Test "when" pattern
        let result = parser.parse("remind me to ask about API when meeting with John");
        assert!(result.is_ok());

        // Test implicit intention
        let result = parser.parse("I should tell Sarah about the bug");
        assert!(result.is_ok());
    }

    #[test]
    fn test_intention_snooze() {
        let pm = ProspectiveMemory::new();

        let intention = Intention::new(
            "Test",
            IntentionTrigger::after_duration(Duration::seconds(-1)),
        );

        let id = pm.create_intention(intention).unwrap();

        pm.snooze(&id, Duration::hours(1)).unwrap();

        let intention = pm.get_intention(&id).unwrap().unwrap();
        assert_eq!(intention.status, IntentionStatus::Snoozed);
    }

    #[test]
    fn test_intention_fulfill() {
        let pm = ProspectiveMemory::new();

        let intention =
            Intention::new("Test", IntentionTrigger::after_duration(Duration::hours(1)));

        let id = pm.create_intention(intention).unwrap();

        pm.fulfill(&id).unwrap();

        let intention = pm.get_intention(&id).unwrap().unwrap();
        assert_eq!(intention.status, IntentionStatus::Fulfilled);
    }

    #[test]
    fn test_recurrence_pattern() {
        let now = Utc::now();

        let pattern = RecurrencePattern::EveryHours(2);
        let next = pattern.next_occurrence(now);
        assert!(next > now);
        assert!((next - now) == Duration::hours(2));
    }

    #[test]
    fn test_stats() {
        let pm = ProspectiveMemory::new();

        // Create some intentions
        for i in 0..5 {
            let intention = Intention::new(
                format!("Intention {}", i),
                IntentionTrigger::after_duration(Duration::hours(1)),
            );
            pm.create_intention(intention).unwrap();
        }

        let stats = pm.stats().unwrap();
        assert_eq!(stats.total_active, 5);
    }
}
