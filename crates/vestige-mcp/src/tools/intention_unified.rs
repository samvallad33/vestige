//! Unified Intention Tool
//!
//! A single unified tool that merges all 5 intention operations:
//! - set_intention -> action: "set"
//! - check_intentions -> action: "check"
//! - complete_intention -> action: "update" with status: "complete"
//! - snooze_intention -> action: "update" with status: "snooze"
//! - list_intentions -> action: "list"

use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::cognitive::CognitiveEngine;
use vestige_core::IntentionRecord;
use vestige_core::Storage;
use vestige_core::neuroscience::{
    ContextPattern, IntentionTrigger as ProspectiveTrigger, ProspectiveContext, RecurrencePattern,
    TriggerPattern,
};

const MAX_TRIGGER_DEPTH: usize = 5;
const MAX_TRIGGER_NODES: usize = 32;
const MAX_TRIGGER_BRANCHES: usize = 16;
const MAX_TRIGGER_TEXT_BYTES: usize = 4_096;
const MAX_ARGUMENT_BYTES: usize = 128 * 1_024;
const MAX_CONTEXT_ITEMS: usize = 32;
const MAX_DURATION_MINUTES: i64 = 10 * 365 * 24 * 60;
const MAX_LIST_LIMIT: i32 = 1_000;
const MAX_ONE_SHOT_REMINDERS: i32 = 5;
const MIN_ONE_SHOT_REMINDER_INTERVAL_MINUTES: i64 = 30;

/// Unified schema for the `intention` tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "description": "Unified intention management tool. Supports setting, checking, updating (complete/snooze/cancel), and listing intentions.",
        "$defs": {
            "trigger": {
                "type": "object",
                "description": "A simple, recurring, activity, or bounded compound trigger.",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["time", "context", "event", "activity", "recurring", "compound"]
                    },
                    "at": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Absolute RFC3339 trigger or recurrence anchor (UTC-normalized on storage)"
                    },
                    "in_minutes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_DURATION_MINUTES,
                        "description": "Positive minutes from intention creation for a one-shot time trigger"
                    },
                    "codebase": { "type": "string", "maxLength": MAX_TRIGGER_TEXT_BYTES },
                    "file_pattern": { "type": "string", "maxLength": MAX_TRIGGER_TEXT_BYTES },
                    "topic": { "type": "string", "maxLength": MAX_TRIGGER_TEXT_BYTES },
                    "condition": {
                        "type": "string",
                        "maxLength": MAX_TRIGGER_TEXT_BYTES,
                        "description": "Case-insensitive text matched against check context events"
                    },
                    "activity": {
                        "type": "string",
                        "maxLength": MAX_TRIGGER_TEXT_BYTES,
                        "description": "Activity whose completion is matched against check context events"
                    },
                    "recurrence": {
                        "description": "Named cadence or a bounded absolute interval. Fortnightly is exactly 20,160 minutes.",
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "hourly, daily, weekly, fortnightly, or 'every N minutes/hours/days/weeks/fortnights'"
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "every": { "type": "integer", "minimum": 1 },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "fortnight", "fortnights"]
                                    },
                                    "every_minutes": { "type": "integer", "minimum": 1, "maximum": MAX_DURATION_MINUTES },
                                    "interval_minutes": { "type": "integer", "minimum": 1, "maximum": MAX_DURATION_MINUTES }
                                }
                            }
                        ]
                    },
                    "base": { "$ref": "#/$defs/trigger" },
                    "all_of": {
                        "type": "array",
                        "maxItems": MAX_TRIGGER_BRANCHES,
                        "items": { "$ref": "#/$defs/trigger" }
                    },
                    "any_of": {
                        "type": "array",
                        "maxItems": MAX_TRIGGER_BRANCHES,
                        "items": { "$ref": "#/$defs/trigger" }
                    }
                }
            }
        },
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "check", "update", "list"],
                "description": "'set' creates, 'check' finds triggered intentions, 'update' changes status (complete, snooze, cancel), 'list' shows them"
            },
            // SET action parameters
            "description": {
                "type": "string",
                "description": "[set] What to remember to do"
            },
            "trigger": {
                "$ref": "#/$defs/trigger",
                "description": "[set] When to trigger this intention"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "critical"],
                "default": "normal",
                "description": "[set] Priority level"
            },
            "deadline": {
                "type": "string",
                "description": "[set] Optional deadline (ISO timestamp)"
            },
            // UPDATE action parameters
            "id": {
                "type": "string",
                "description": "[update] ID of the intention to update"
            },
            "status": {
                "type": "string",
                "enum": ["complete", "snooze", "cancel"],
                "description": "[update] New status: 'complete' marks as fulfilled, 'snooze' delays, 'cancel' cancels"
            },
            "snooze_minutes": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_DURATION_MINUTES,
                "default": 30,
                "description": "[update] Minutes to snooze for (when status is 'snooze')"
            },
            // CHECK action parameters
            "context": {
                "type": "object",
                "description": "[check] Current context for matching intentions",
                "properties": {
                    "current_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Current RFC3339 timestamp (defaults to now)"
                    },
                    "codebase": {
                        "type": "string",
                        "maxLength": MAX_TRIGGER_TEXT_BYTES,
                        "description": "Current codebase/project name"
                    },
                    "file": {
                        "type": "string",
                        "maxLength": MAX_TRIGGER_TEXT_BYTES,
                        "description": "Current file path"
                    },
                    "topics": {
                        "type": "array",
                        "maxItems": MAX_CONTEXT_ITEMS,
                        "items": { "type": "string", "maxLength": MAX_TRIGGER_TEXT_BYTES },
                        "description": "Current discussion topics"
                    },
                    "events": {
                        "type": "array",
                        "maxItems": MAX_CONTEXT_ITEMS,
                        "items": { "type": "string", "maxLength": MAX_TRIGGER_TEXT_BYTES },
                        "description": "Recent event or activity descriptions for event/activity triggers"
                    }
                }
            },
            "include_snoozed": {
                "type": "boolean",
                "default": false,
                "description": "[check] Include snoozed intentions"
            },
            // LIST action parameters
            "filter_status": {
                "type": "string",
                "enum": ["active", "fulfilled", "cancelled", "snoozed", "all"],
                "default": "active",
                "description": "[list] Filter by status"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LIST_LIMIT,
                "default": 20,
                "description": "[list] Maximum number to return"
            }
        },
        "required": ["action"]
    })
}

// ============================================================================
// ARGUMENT STRUCTS
// ============================================================================

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
struct TriggerSpec {
    #[serde(rename = "type")]
    trigger_type: Option<String>,
    at: Option<String>,
    #[serde(alias = "inMinutes")]
    in_minutes: Option<i64>,
    codebase: Option<String>,
    #[serde(alias = "filePattern")]
    file_pattern: Option<String>,
    topic: Option<String>,
    condition: Option<String>,
    activity: Option<String>,
    recurrence: Option<Value>,
    base: Option<Box<TriggerSpec>>,
    #[serde(default, alias = "allOf")]
    all_of: Vec<TriggerSpec>,
    #[serde(default, alias = "anyOf")]
    any_of: Vec<TriggerSpec>,
    #[serde(alias = "nextOccurrence")]
    next_occurrence: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ContextSpec {
    #[serde(alias = "currentTime")]
    current_time: Option<String>,
    codebase: Option<String>,
    file: Option<String>,
    topics: Option<Vec<String>>,
    #[serde(default, alias = "recentEvents", alias = "recent_events")]
    events: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct UnifiedIntentionArgs {
    action: String,
    // SET parameters
    description: Option<String>,
    trigger: Option<TriggerSpec>,
    priority: Option<String>,
    deadline: Option<String>,
    // UPDATE parameters
    id: Option<String>,
    status: Option<String>,
    #[serde(alias = "snoozeMinutes")]
    snooze_minutes: Option<i64>,
    // CHECK parameters
    context: Option<ContextSpec>,
    #[serde(alias = "includeSnoozed")]
    #[allow(dead_code)]
    include_snoozed: Option<bool>,
    // LIST parameters
    #[serde(alias = "filterStatus")]
    filter_status: Option<String>,
    limit: Option<i32>,
}

fn parse_rfc3339(value: &str, field: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| format!("'{field}' must be a valid RFC3339 timestamp"))
}

fn checked_minutes(amount: i64, multiplier: i64) -> Result<i64, String> {
    let minutes = amount
        .checked_mul(multiplier)
        .ok_or_else(|| "Recurrence interval is too large".to_string())?;
    if !(1..=MAX_DURATION_MINUTES).contains(&minutes) {
        return Err(format!(
            "Recurrence interval must be between 1 and {MAX_DURATION_MINUTES} minutes"
        ));
    }
    Ok(minutes)
}

fn recurrence_minutes(value: &Value) -> Result<i64, String> {
    if let Some(name) = value.as_str() {
        let normalized = name.trim().to_ascii_lowercase();
        let named = match normalized.as_str() {
            "hourly" | "every hour" => Some(60),
            "daily" | "every day" => Some(1_440),
            "weekly" | "every week" => Some(10_080),
            "fortnightly" | "fortnight" | "every fortnight" | "every two weeks"
            | "every other week" | "every 2 weeks" => Some(20_160),
            _ => None,
        };
        if let Some(minutes) = named {
            return Ok(minutes);
        }

        let words: Vec<&str> = normalized.split_whitespace().collect();
        if words.len() == 3 && words[0] == "every" {
            let amount = words[1]
                .parse::<i64>()
                .map_err(|_| "Recurrence must use a positive integer amount".to_string())?;
            let multiplier = match words[2] {
                "minute" | "minutes" => 1,
                "hour" | "hours" => 60,
                "day" | "days" => 1_440,
                "week" | "weeks" => 10_080,
                "fortnight" | "fortnights" => 20_160,
                _ => {
                    return Err(
                        "Unsupported recurrence unit; use minutes, hours, days, weeks, or fortnights"
                            .to_string(),
                    );
                }
            };
            return checked_minutes(amount, multiplier);
        }
        return Err(
            "Unsupported recurrence; use hourly, daily, weekly, fortnightly, or 'every N <unit>'"
                .to_string(),
        );
    }

    let object = value
        .as_object()
        .ok_or_else(|| "'recurrence' must be a string or object".to_string())?;
    for key in [
        "every_minutes",
        "everyMinutes",
        "interval_minutes",
        "intervalMinutes",
    ] {
        if let Some(raw) = object.get(key) {
            let minutes = raw
                .as_i64()
                .ok_or_else(|| format!("'{key}' must be an integer"))?;
            return checked_minutes(minutes, 1);
        }
    }

    let amount = object.get("every").and_then(Value::as_i64).ok_or_else(|| {
        "Recurrence object requires 'every' with 'unit', or 'every_minutes'".to_string()
    })?;
    let unit = object
        .get("unit")
        .and_then(Value::as_str)
        .ok_or_else(|| "Recurrence object with 'every' requires string 'unit'".to_string())?
        .to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "minute" | "minutes" => 1,
        "hour" | "hours" => 60,
        "day" | "days" => 1_440,
        "week" | "weeks" => 10_080,
        "fortnight" | "fortnights" => 20_160,
        _ => {
            return Err(
                "Unsupported recurrence unit; use minutes, hours, days, weeks, or fortnights"
                    .to_string(),
            );
        }
    };
    checked_minutes(amount, multiplier)
}

fn normalize_text(value: &mut Option<String>, field: &str) -> Result<(), String> {
    let Some(text) = value else {
        return Ok(());
    };
    *text = text.trim().to_string();
    if text.is_empty() {
        return Err(format!("'{field}' cannot be empty"));
    }
    if text.len() > MAX_TRIGGER_TEXT_BYTES {
        return Err(format!(
            "'{field}' is too large (max {MAX_TRIGGER_TEXT_BYTES} bytes)"
        ));
    }
    Ok(())
}

impl TriggerSpec {
    fn inferred_type(&self) -> &str {
        if self.recurrence.is_some() {
            "recurring"
        } else if !self.all_of.is_empty() || !self.any_of.is_empty() {
            "compound"
        } else if self.activity.is_some() {
            "activity"
        } else if self.condition.is_some() {
            "event"
        } else if self.codebase.is_some() || self.file_pattern.is_some() || self.topic.is_some() {
            "context"
        } else {
            "time"
        }
    }

    fn normalize(
        &mut self,
        now: DateTime<Utc>,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(), String> {
        if depth > MAX_TRIGGER_DEPTH {
            return Err(format!(
                "Trigger nesting exceeds maximum depth {MAX_TRIGGER_DEPTH}"
            ));
        }
        *nodes += 1;
        if *nodes > MAX_TRIGGER_NODES {
            return Err(format!(
                "Trigger contains more than {MAX_TRIGGER_NODES} total nodes"
            ));
        }

        for (value, field) in [
            (&mut self.codebase, "codebase"),
            (&mut self.file_pattern, "file_pattern"),
            (&mut self.topic, "topic"),
            (&mut self.condition, "condition"),
            (&mut self.activity, "activity"),
        ] {
            normalize_text(value, field)?;
        }

        let trigger_type = self
            .trigger_type
            .as_deref()
            .unwrap_or_else(|| self.inferred_type())
            .trim()
            .to_ascii_lowercase();
        self.trigger_type = Some(trigger_type.clone());

        match trigger_type.as_str() {
            "time" => {
                if self.at.is_some() == self.in_minutes.is_some() {
                    return Err("Time trigger requires exactly one of 'at' or 'in_minutes'".into());
                }
                if let Some(at) = &self.at {
                    self.at = Some(parse_rfc3339(at, "trigger.at")?.to_rfc3339());
                }
                if let Some(minutes) = self.in_minutes
                    && !(1..=MAX_DURATION_MINUTES).contains(&minutes)
                {
                    return Err(format!(
                        "'in_minutes' must be between 1 and {MAX_DURATION_MINUTES}"
                    ));
                }
            }
            "context" => {
                if self.codebase.is_none() && self.file_pattern.is_none() && self.topic.is_none() {
                    return Err(
                        "Context trigger requires at least one of 'codebase', 'file_pattern', or 'topic'"
                            .into(),
                    );
                }
            }
            "event" => {
                if self.condition.is_none() {
                    return Err("Event trigger requires 'condition'".into());
                }
            }
            "activity" => {
                if self.activity.is_none() {
                    return Err("Activity trigger requires 'activity'".into());
                }
            }
            "recurring" => {
                let recurrence = self
                    .recurrence
                    .as_ref()
                    .ok_or("Recurring trigger requires 'recurrence'")?;
                let minutes = recurrence_minutes(recurrence)?;
                self.recurrence = Some(serde_json::json!({ "every_minutes": minutes }));

                let next = if let Some(value) = &self.next_occurrence {
                    parse_rfc3339(value, "trigger.next_occurrence")?
                } else if let Some(value) = &self.at {
                    parse_rfc3339(value, "trigger.at")?
                } else if let Some(delay) = self.in_minutes {
                    if !(1..=MAX_DURATION_MINUTES).contains(&delay) {
                        return Err(format!(
                            "'in_minutes' must be between 1 and {MAX_DURATION_MINUTES}"
                        ));
                    }
                    now + Duration::minutes(delay)
                } else {
                    now + Duration::minutes(minutes)
                };
                self.next_occurrence = Some(next.to_rfc3339());
                if let Some(at) = &self.at {
                    self.at = Some(parse_rfc3339(at, "trigger.at")?.to_rfc3339());
                }

                if let Some(base) = &mut self.base {
                    base.normalize(now, depth + 1, nodes)?;
                } else {
                    *nodes += 1;
                    if *nodes > MAX_TRIGGER_NODES {
                        return Err(format!(
                            "Trigger contains more than {MAX_TRIGGER_NODES} total nodes"
                        ));
                    }
                    self.base = Some(Box::new(Self {
                        trigger_type: Some("time".to_string()),
                        at: Some(next.to_rfc3339()),
                        in_minutes: None,
                        codebase: None,
                        file_pattern: None,
                        topic: None,
                        condition: None,
                        activity: None,
                        recurrence: None,
                        base: None,
                        all_of: Vec::new(),
                        any_of: Vec::new(),
                        next_occurrence: None,
                    }));
                }
            }
            "compound" => {
                if self.all_of.is_empty() && self.any_of.is_empty() {
                    return Err("Compound trigger requires non-empty 'all_of' or 'any_of'".into());
                }
                if self.all_of.len() > MAX_TRIGGER_BRANCHES
                    || self.any_of.len() > MAX_TRIGGER_BRANCHES
                {
                    return Err(format!(
                        "Each compound branch list is limited to {MAX_TRIGGER_BRANCHES} triggers"
                    ));
                }
                for child in self.all_of.iter_mut().chain(self.any_of.iter_mut()) {
                    child.normalize(now, depth + 1, nodes)?;
                }
            }
            other => {
                return Err(format!(
                    "Unknown trigger type '{other}'. Valid types: time, context, event, activity, recurring, compound"
                ));
            }
        }
        Ok(())
    }

    fn to_prospective(&self, created_at: DateTime<Utc>) -> Result<ProspectiveTrigger, String> {
        match self.trigger_type.as_deref().unwrap_or("time") {
            "time" => {
                if let Some(at) = &self.at {
                    Ok(ProspectiveTrigger::TimeBased {
                        at: parse_rfc3339(at, "trigger.at")?,
                    })
                } else if let Some(minutes) = self.in_minutes {
                    Ok(ProspectiveTrigger::DurationBased {
                        after: Duration::minutes(minutes),
                        trigger_at: Some(created_at + Duration::minutes(minutes)),
                    })
                } else {
                    Err("Stored time trigger has no time".into())
                }
            }
            "context" => {
                let mut all = Vec::new();
                if let Some(value) = &self.codebase {
                    all.push(ContextPattern::InCodebase(value.clone()));
                }
                if let Some(value) = &self.file_pattern {
                    all.push(ContextPattern::FilePattern(value.clone()));
                }
                if let Some(value) = &self.topic {
                    all.push(ContextPattern::TopicActive(value.clone()));
                }
                let context_match = if all.len() == 1 {
                    all.pop().expect("one context pattern")
                } else {
                    ContextPattern::Composite {
                        all,
                        any: Vec::new(),
                    }
                };
                Ok(ProspectiveTrigger::ContextBased { context_match })
            }
            "event" => {
                let condition = self
                    .condition
                    .clone()
                    .ok_or("Stored event has no condition")?;
                Ok(ProspectiveTrigger::EventBased {
                    pattern: TriggerPattern::contains(&condition),
                    condition,
                })
            }
            "activity" => {
                let activity = self
                    .activity
                    .clone()
                    .ok_or("Stored activity has no activity")?;
                let match_text = self.condition.clone().unwrap_or_else(|| activity.clone());
                Ok(ProspectiveTrigger::ActivityBased {
                    activity,
                    completion_pattern: TriggerPattern::contains(match_text),
                })
            }
            "recurring" => {
                let minutes = recurrence_minutes(
                    self.recurrence
                        .as_ref()
                        .ok_or("Stored recurrence has no cadence")?,
                )?;
                let next = self
                    .next_occurrence
                    .as_deref()
                    .ok_or("Stored recurrence has no next occurrence")?;
                Ok(ProspectiveTrigger::Recurring {
                    base: Box::new(
                        self.base
                            .as_ref()
                            .ok_or("Stored recurrence has no base")?
                            .to_prospective(created_at)?,
                    ),
                    recurrence: RecurrencePattern::EveryMinutes(minutes),
                    next_occurrence: Some(parse_rfc3339(next, "trigger.next_occurrence")?),
                })
            }
            "compound" => Ok(ProspectiveTrigger::Compound {
                all_of: self
                    .all_of
                    .iter()
                    .map(|trigger| trigger.to_prospective(created_at))
                    .collect::<Result<Vec<_>, _>>()?,
                any_of: self
                    .any_of
                    .iter()
                    .map(|trigger| trigger.to_prospective(created_at))
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            other => Err(format!("Unsupported stored trigger type '{other}'")),
        }
    }

    fn from_prospective(trigger: &ProspectiveTrigger) -> Self {
        match trigger {
            ProspectiveTrigger::TimeBased { at } => Self::time_at(*at),
            ProspectiveTrigger::DurationBased { after, .. } => Self {
                trigger_type: Some("time".into()),
                in_minutes: Some(after.num_minutes().max(1)),
                ..Self::empty()
            },
            ProspectiveTrigger::EventBased { condition, .. } => Self {
                trigger_type: Some("event".into()),
                condition: Some(condition.clone()),
                ..Self::empty()
            },
            ProspectiveTrigger::ContextBased { context_match } => {
                Self::from_context_pattern(context_match)
            }
            ProspectiveTrigger::ActivityBased { activity, .. } => Self {
                trigger_type: Some("activity".into()),
                activity: Some(activity.clone()),
                ..Self::empty()
            },
            ProspectiveTrigger::Recurring {
                base,
                recurrence,
                next_occurrence,
            } => {
                let minutes = match recurrence {
                    RecurrencePattern::EveryMinutes(value) => *value,
                    RecurrencePattern::EveryHours(value) => value.saturating_mul(60),
                    RecurrencePattern::Daily { .. } => 1_440,
                    RecurrencePattern::Weekly { .. } => 10_080,
                    RecurrencePattern::Monthly { .. } => 43_200,
                    RecurrencePattern::Custom { interval } => interval.num_minutes(),
                };
                Self {
                    trigger_type: Some("recurring".into()),
                    recurrence: Some(serde_json::json!({ "every_minutes": minutes.max(1) })),
                    base: Some(Box::new(Self::from_prospective(base))),
                    next_occurrence: next_occurrence.map(|value| value.to_rfc3339()),
                    ..Self::empty()
                }
            }
            ProspectiveTrigger::Compound { all_of, any_of } => Self {
                trigger_type: Some("compound".into()),
                all_of: all_of.iter().map(Self::from_prospective).collect(),
                any_of: any_of.iter().map(Self::from_prospective).collect(),
                ..Self::empty()
            },
        }
    }

    fn from_context_pattern(pattern: &ContextPattern) -> Self {
        match pattern {
            ContextPattern::InCodebase(value) => Self {
                trigger_type: Some("context".into()),
                codebase: Some(value.clone()),
                ..Self::empty()
            },
            ContextPattern::FilePattern(value) => Self {
                trigger_type: Some("context".into()),
                file_pattern: Some(value.clone()),
                ..Self::empty()
            },
            ContextPattern::TopicActive(value) => Self {
                trigger_type: Some("context".into()),
                topic: Some(value.clone()),
                ..Self::empty()
            },
            ContextPattern::UserMode(value) => Self {
                trigger_type: Some("event".into()),
                condition: Some(value.clone()),
                ..Self::empty()
            },
            ContextPattern::Composite { all, any } => Self {
                trigger_type: Some("compound".into()),
                all_of: all.iter().map(Self::from_context_pattern).collect(),
                any_of: any.iter().map(Self::from_context_pattern).collect(),
                ..Self::empty()
            },
        }
    }

    fn empty() -> Self {
        Self {
            trigger_type: None,
            at: None,
            in_minutes: None,
            codebase: None,
            file_pattern: None,
            topic: None,
            condition: None,
            activity: None,
            recurrence: None,
            base: None,
            all_of: Vec::new(),
            any_of: Vec::new(),
            next_occurrence: None,
        }
    }

    fn time_at(at: DateTime<Utc>) -> Self {
        Self {
            trigger_type: Some("time".into()),
            at: Some(at.to_rfc3339()),
            ..Self::empty()
        }
    }
}

// ============================================================================
// MAIN EXECUTE FUNCTION
// ============================================================================

/// Execute the unified intention tool
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: UnifiedIntentionArgs = match args {
        Some(v) => {
            let encoded_size = serde_json::to_vec(&v)
                .map_err(|error| format!("Invalid arguments: {error}"))?
                .len();
            if encoded_size > MAX_ARGUMENT_BYTES {
                return Err(format!(
                    "Arguments exceed the {MAX_ARGUMENT_BYTES}-byte limit"
                ));
            }
            serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?
        }
        None => return Err("Missing arguments".to_string()),
    };

    match args.action.as_str() {
        "set" => execute_set(storage, cognitive, &args).await,
        "check" => execute_check(storage, cognitive, &args).await,
        "update" => execute_update(storage, &args).await,
        "list" => execute_list(storage, &args).await,
        _ => Err(format!(
            "Unknown action: '{}'. Valid actions are: set, check, update, list",
            args.action
        )),
    }
}

// ============================================================================
// ACTION IMPLEMENTATIONS
// ============================================================================

/// Execute "set" action - create a new intention
async fn execute_set(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: &UnifiedIntentionArgs,
) -> Result<Value, String> {
    let description = args
        .description
        .as_ref()
        .ok_or("Missing 'description' for set action")?;

    if description.trim().is_empty() {
        return Err("Description cannot be empty".to_string());
    }

    if description.len() > 100_000 {
        return Err("Description too large (max 100KB)".to_string());
    }

    let now = Utc::now();
    let id = Uuid::new_v4().to_string();

    // ====================================================================
    // COGNITIVE: NLP parsing + intent auto-tagging
    // ====================================================================
    let mut nlp_parsed = false;
    let mut nlp_trigger = None;
    let mut nlp_priority = None;
    let mut tags = Vec::new();

    if let Ok(cog) = cognitive.try_lock() {
        // 8A. Try NLP parsing when no explicit trigger is provided
        if args.trigger.is_none()
            && let Ok(parsed) = cog.intention_parser.parse(description)
        {
            nlp_parsed = true;
            // Preserve the full parsed trigger tree, including recurrence base,
            // cadence, next occurrence, activity, and compound branches.
            nlp_trigger = Some(TriggerSpec::from_prospective(&parsed.trigger));

            // Use NLP-detected priority if user didn't specify one
            if args.priority.is_none() {
                nlp_priority = Some(parsed.priority);
            }
        }

        // Auto-tag with detected intent
        let intent_result = cog.intent_detector.detect_intent();
        if intent_result.confidence > 0.5 {
            let intent_tag = format!("intent:{:?}", intent_result.primary_intent);
            let intent_tag = if intent_tag.len() > 50 {
                format!("{}...", &intent_tag[..intent_tag.floor_char_boundary(47)])
            } else {
                intent_tag
            };
            tags.push(intent_tag);
        }
    }

    // Determine and validate trigger (explicit > NLP > manual). Manual
    // intentions preserve the existing non-triggering `{}` storage shape.
    let mut normalized_trigger = args.trigger.clone().or(nlp_trigger);
    let (trigger_type, trigger_data, trigger_at, next_occurrence) =
        if let Some(trigger) = &mut normalized_trigger {
            let mut nodes = 0;
            trigger.normalize(now, 1, &mut nodes)?;
            let trigger_type = trigger
                .trigger_type
                .clone()
                .expect("normalization assigns trigger type");
            let trigger_at = if trigger_type == "time" {
                if let Some(at) = &trigger.at {
                    Some(parse_rfc3339(at, "trigger.at")?)
                } else {
                    trigger
                        .in_minutes
                        .map(|minutes| now + Duration::minutes(minutes))
                }
            } else {
                None
            };
            let next_occurrence = trigger
                .next_occurrence
                .as_deref()
                .map(|value| parse_rfc3339(value, "trigger.next_occurrence"))
                .transpose()?;
            let data = serde_json::to_string(trigger)
                .map_err(|error| format!("Failed to encode trigger: {error}"))?;
            (trigger_type, data, trigger_at, next_occurrence)
        } else {
            ("manual".to_string(), "{}".to_string(), None, None)
        };

    // Parse priority (explicit > NLP > normal)
    let priority = match args.priority.as_deref() {
        Some("low") => 1,
        Some("high") => 3,
        Some("critical") => 4,
        Some("normal") => 2,
        Some(_) => 2,
        None => {
            // Use NLP-detected priority if available
            if let Some(nlp_p) = nlp_priority {
                use vestige_core::neuroscience::prospective_memory::Priority;
                match nlp_p {
                    Priority::Low => 1,
                    Priority::Normal => 2,
                    Priority::High => 3,
                    Priority::Critical => 4,
                }
            } else {
                2 // normal default
            }
        }
    };

    // Parse deadline
    let deadline = args
        .deadline
        .as_deref()
        .map(|value| parse_rfc3339(value, "deadline"))
        .transpose()?;

    let record = IntentionRecord {
        id: id.clone(),
        content: description.clone(),
        trigger_type,
        trigger_data,
        priority,
        status: "active".to_string(),
        created_at: now,
        deadline,
        fulfilled_at: None,
        reminder_count: 0,
        last_reminded_at: None,
        notes: None,
        tags,
        related_memories: vec![],
        snoozed_until: None,
        source_type: if nlp_parsed { "nlp" } else { "mcp" }.to_string(),
        source_data: None,
    };

    storage.save_intention(&record).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": true,
        "action": "set",
        "intentionId": id,
        "message": format!("Intention created: {}", description),
        "priority": priority,
        "triggerAt": trigger_at.map(|dt| dt.to_rfc3339()),
        "nextOccurrence": next_occurrence.map(|dt| dt.to_rfc3339()),
        "deadline": deadline.map(|dt| dt.to_rfc3339()),
        "nlpParsed": nlp_parsed,
    }))
}

/// Execute "check" action - find triggered intentions
async fn execute_check(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: &UnifiedIntentionArgs,
) -> Result<Value, String> {
    let now = args
        .context
        .as_ref()
        .and_then(|context| context.current_time.as_deref())
        .map(|value| parse_rfc3339(value, "context.current_time"))
        .transpose()?
        .unwrap_or_else(Utc::now);

    let mut prospective_ctx = ProspectiveContext::new();
    prospective_ctx.timestamp = now;
    if let Some(ctx) = &args.context {
        for (value, field) in [(&ctx.codebase, "codebase"), (&ctx.file, "file")] {
            if let Some(value) = value
                && value.len() > MAX_TRIGGER_TEXT_BYTES
            {
                return Err(format!(
                    "'context.{field}' is limited to {MAX_TRIGGER_TEXT_BYTES} bytes"
                ));
            }
        }
        if ctx.topics.as_ref().map(Vec::len).unwrap_or(0) > MAX_CONTEXT_ITEMS {
            return Err(format!(
                "'context.topics' is limited to {MAX_CONTEXT_ITEMS} items"
            ));
        }
        if let Some(topics) = &ctx.topics {
            for topic in topics {
                if topic.len() > MAX_TRIGGER_TEXT_BYTES {
                    return Err(format!(
                        "Each context topic is limited to {MAX_TRIGGER_TEXT_BYTES} bytes"
                    ));
                }
            }
        }
        if ctx.events.len() > MAX_CONTEXT_ITEMS {
            return Err(format!(
                "'context.events' is limited to {MAX_CONTEXT_ITEMS} items"
            ));
        }
        for event in &ctx.events {
            if event.len() > MAX_TRIGGER_TEXT_BYTES {
                return Err(format!(
                    "Each context event is limited to {MAX_TRIGGER_TEXT_BYTES} bytes"
                ));
            }
        }
        if let Some(codebase) = &ctx.codebase {
            prospective_ctx.project_name = Some(codebase.clone());
        }
        if let Some(file) = &ctx.file {
            prospective_ctx.active_files = vec![file.clone()];
        }
        if let Some(topics) = &ctx.topics {
            prospective_ctx.active_topics = topics.clone();
        }
        prospective_ctx.recent_events = ctx.events.clone();
    }

    // ====================================================================
    // COGNITIVE: Update prospective memory context
    // ====================================================================
    if args.context.is_some()
        && let Ok(cog) = cognitive.try_lock()
    {
        // Update context on prospective memory (triggers internal monitoring)
        let _ = cog
            .prospective_memory
            .update_context(prospective_ctx.clone());
    }

    // Always inspect snoozed records so an expired snooze can wake on this
    // check. `include_snoozed` controls only whether records whose snooze is
    // still in force are returned as pending.
    let mut intentions = storage.get_active_intentions().map_err(|e| e.to_string())?;
    let snoozed = storage
        .get_intentions_by_status("snoozed")
        .map_err(|e| e.to_string())?;
    use std::collections::HashSet;
    let seen: HashSet<String> = intentions.iter().map(|i| i.id.clone()).collect();
    for intention in snoozed {
        let expired = intention
            .snoozed_until
            .map(|until| now >= until)
            .unwrap_or(true);
        if (expired || args.include_snoozed.unwrap_or(false)) && !seen.contains(&intention.id) {
            intentions.push(intention);
        }
    }

    let mut triggered = Vec::new();
    let mut pending = Vec::new();
    let mut changes = Vec::new();

    for mut intention in intentions {
        let original = intention.clone();
        let mut changed = false;
        let snooze_in_force = intention
            .snoozed_until
            .map(|until| now < until)
            .unwrap_or(false)
            && intention.status == "snoozed";
        if intention.status == "snoozed" && !snooze_in_force {
            intention.status = "active".to_string();
            intention.snoozed_until = None;
            changed = true;
        }

        let (mut trigger, prospective_trigger, invalid_trigger) =
            if intention.trigger_type == "manual" && intention.trigger_data.trim() == "{}" {
                (None, None, None)
            } else {
                match serde_json::from_str::<TriggerSpec>(&intention.trigger_data) {
                    Ok(mut spec) => {
                        let mut nodes = 0;
                        match spec
                            .normalize(intention.created_at, 1, &mut nodes)
                            .and_then(|_| spec.to_prospective(intention.created_at))
                        {
                            Ok(prospective) => (Some(spec), Some(prospective), None),
                            Err(error) => (Some(spec), None, Some(error)),
                        }
                    }
                    Err(error) => (
                        None,
                        None,
                        Some(format!("Stored trigger JSON is invalid: {error}")),
                    ),
                }
            };
        let trigger_matched = !snooze_in_force
            && prospective_trigger
                .as_ref()
                .map(|value| {
                    value.is_triggered_at(&prospective_ctx, &prospective_ctx.recent_events, now)
                })
                .unwrap_or(false);

        // Snooze suppresses both trigger and overdue delivery until it expires.
        let is_overdue = !snooze_in_force
            && intention
                .deadline
                .map(|deadline| deadline < now)
                .unwrap_or(false);
        // Determine whether a recurring branch actually fired by advancing a
        // clone. Tree membership alone is insufficient: an `any_of` may match
        // its event branch while a sibling recurrence is still in the future.
        let mut advanced_trigger = prospective_trigger.clone();
        let scheduled_recurrence = trigger_matched
            && advanced_trigger
                .as_mut()
                .map(|value| {
                    value.re_arm_triggered(&prospective_ctx, &prospective_ctx.recent_events, now)
                })
                .unwrap_or(false);
        let within_one_shot_limits = intention.reminder_count < MAX_ONE_SHOT_REMINDERS
            && intention
                .last_reminded_at
                .map(|last| now - last >= Duration::minutes(MIN_ONE_SHOT_REMINDER_INTERVAL_MINUTES))
                .unwrap_or(true);
        let should_deliver =
            (trigger_matched || is_overdue) && (scheduled_recurrence || within_one_shot_limits);

        if should_deliver {
            intention.reminder_count = intention.reminder_count.saturating_add(1);
            intention.last_reminded_at = Some(now);
            changed = true;

            if scheduled_recurrence && let Some(value) = &advanced_trigger {
                let advanced = TriggerSpec::from_prospective(value);
                intention.trigger_type = advanced
                    .trigger_type
                    .clone()
                    .unwrap_or_else(|| intention.trigger_type.clone());
                intention.trigger_data = serde_json::to_string(&advanced)
                    .map_err(|error| format!("Failed to persist recurrence: {error}"))?;
                trigger = Some(advanced);
            }
        }

        let next_occurrence = trigger
            .as_ref()
            .and_then(|value| value.next_occurrence.clone());

        let item = serde_json::json!({
            "id": intention.id,
            "description": intention.content,
            "status": intention.status,
            "priority": match intention.priority {
                1 => "low",
                3 => "high",
                4 => "critical",
                _ => "normal",
            },
            "createdAt": intention.created_at.to_rfc3339(),
            "deadline": intention.deadline.map(|d| d.to_rfc3339()),
            "snoozedUntil": intention.snoozed_until.map(|d| d.to_rfc3339()),
            "isOverdue": is_overdue,
            "reminderCount": intention.reminder_count,
            "nextOccurrence": next_occurrence,
            "invalid_trigger": invalid_trigger,
        });

        if should_deliver {
            triggered.push(item);
        } else {
            pending.push(item);
        }

        if changed {
            changes.push((original, intention));
        }
    }

    // Claim every due occurrence and wake every expired snooze in one
    // compare-and-swap transaction. A concurrent check that read the same old
    // state loses the CAS and returns an error instead of duplicating delivery;
    // a conflict also rolls back the whole batch so no prefix is consumed.
    if !changes.is_empty() {
        storage.commit_intention_check(&changes)?;
    }

    Ok(serde_json::json!({
        "action": "check",
        "triggered": triggered,
        "pending": pending,
        "checkedAt": now.to_rfc3339(),
    }))
}

/// Execute "update" action - complete, snooze, or cancel an intention
async fn execute_update(
    storage: &Arc<Storage>,
    args: &UnifiedIntentionArgs,
) -> Result<Value, String> {
    let intention_id = args.id.as_ref().ok_or("Missing 'id' for update action")?;

    let status = args
        .status
        .as_ref()
        .ok_or("Missing 'status' for update action")?;

    match status.as_str() {
        "complete" => {
            let updated = storage
                .update_intention_status(intention_id, "fulfilled")
                .map_err(|e| e.to_string())?;

            if updated {
                Ok(serde_json::json!({
                    "success": true,
                    "action": "update",
                    "status": "complete",
                    "message": "Intention marked as complete",
                    "intentionId": intention_id,
                }))
            } else {
                Err(format!("Intention not found: {}", intention_id))
            }
        }
        "snooze" => {
            let minutes = args.snooze_minutes.unwrap_or(30);
            if !(1..=MAX_DURATION_MINUTES).contains(&minutes) {
                return Err(format!(
                    "'snooze_minutes' must be between 1 and {MAX_DURATION_MINUTES}"
                ));
            }
            let snooze_until = Utc::now() + Duration::minutes(minutes);

            let updated = storage
                .snooze_intention(intention_id, snooze_until)
                .map_err(|e| e.to_string())?;

            if updated {
                Ok(serde_json::json!({
                    "success": true,
                    "action": "update",
                    "status": "snooze",
                    "message": format!("Intention snoozed for {} minutes", minutes),
                    "intentionId": intention_id,
                    "snoozedUntil": snooze_until.to_rfc3339(),
                }))
            } else {
                Err(format!("Intention not found: {}", intention_id))
            }
        }
        "cancel" => {
            let updated = storage
                .update_intention_status(intention_id, "cancelled")
                .map_err(|e| e.to_string())?;

            if updated {
                Ok(serde_json::json!({
                    "success": true,
                    "action": "update",
                    "status": "cancel",
                    "message": "Intention cancelled",
                    "intentionId": intention_id,
                }))
            } else {
                Err(format!("Intention not found: {}", intention_id))
            }
        }
        _ => Err(format!(
            "Unknown status: '{}'. Valid statuses are: complete, snooze, cancel",
            status
        )),
    }
}

/// Execute "list" action - list intentions with optional filtering
async fn execute_list(
    storage: &Arc<Storage>,
    args: &UnifiedIntentionArgs,
) -> Result<Value, String> {
    let filter_status = args.filter_status.as_deref().unwrap_or("active");

    let intentions = if filter_status == "all" {
        // Get all by combining different statuses
        let mut all = storage.get_active_intentions().map_err(|e| e.to_string())?;
        all.extend(
            storage
                .get_intentions_by_status("fulfilled")
                .map_err(|e| e.to_string())?,
        );
        all.extend(
            storage
                .get_intentions_by_status("cancelled")
                .map_err(|e| e.to_string())?,
        );
        all.extend(
            storage
                .get_intentions_by_status("snoozed")
                .map_err(|e| e.to_string())?,
        );
        all
    } else if filter_status == "active" {
        // Use get_active_intentions for proper priority ordering
        storage.get_active_intentions().map_err(|e| e.to_string())?
    } else {
        storage
            .get_intentions_by_status(filter_status)
            .map_err(|e| e.to_string())?
    };

    let requested_limit = args.limit.unwrap_or(20);
    if !(1..=MAX_LIST_LIMIT).contains(&requested_limit) {
        return Err(format!("'limit' must be between 1 and {MAX_LIST_LIMIT}"));
    }
    let limit = requested_limit as usize;
    let now = Utc::now();

    let items: Vec<Value> = intentions
        .into_iter()
        .take(limit)
        .map(|i| {
            let is_overdue = i.deadline.map(|d| d < now).unwrap_or(false);
            serde_json::json!({
                "id": i.id,
                "description": i.content,
                "status": i.status,
                "priority": match i.priority {
                    1 => "low",
                    3 => "high",
                    4 => "critical",
                    _ => "normal",
                },
                "createdAt": i.created_at.to_rfc3339(),
                "deadline": i.deadline.map(|d| d.to_rfc3339()),
                "isOverdue": is_overdue,
                "snoozedUntil": i.snoozed_until.map(|d| d.to_rfc3339()),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "list",
        "intentions": items,
        "total": items.len(),
        "status": filter_status,
    }))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    /// Helper to create an intention and return its ID
    async fn create_test_intention(storage: &Arc<Storage>, description: &str) -> String {
        let args = serde_json::json!({
            "action": "set",
            "description": description
        });
        let result = execute(storage, &test_cognitive(), Some(args))
            .await
            .unwrap();
        result["intentionId"].as_str().unwrap().to_string()
    }

    // ========================================================================
    // ACTION ROUTING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_missing_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({});
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    #[tokio::test]
    async fn test_unknown_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "unknown" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown action"));
    }

    #[tokio::test]
    async fn test_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    // ========================================================================
    // SET ACTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_set_action_basic_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": "Remember to write unit tests"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "set");
        assert!(value["intentionId"].is_string());
        assert!(
            value["message"]
                .as_str()
                .unwrap()
                .contains("Intention created")
        );
    }

    #[tokio::test]
    async fn test_set_action_missing_description_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "set" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'description'"));
    }

    #[tokio::test]
    async fn test_set_action_empty_description_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": ""
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_set_action_with_priority() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": "Critical bug fix needed",
            "priority": "critical"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["priority"], 4);
    }

    #[tokio::test]
    async fn test_set_action_with_time_trigger() {
        let (storage, _dir) = test_storage().await;
        let future_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
        let args = serde_json::json!({
            "action": "set",
            "description": "Meeting reminder",
            "trigger": {
                "type": "time",
                "at": future_time
            }
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["triggerAt"].is_string());
    }

    #[tokio::test]
    async fn test_set_action_with_duration_trigger() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": "Check build status",
            "trigger": {
                "type": "time",
                "inMinutes": 30
            }
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["triggerAt"].is_string());
    }

    #[tokio::test]
    async fn test_set_action_with_duration_trigger_snake_case() {
        // The public JSON schema (see schema() above) declares `in_minutes` in
        // snake_case. The TriggerSpec struct uses `rename_all = "camelCase"` so
        // without an explicit `#[serde(alias = "in_minutes")]` the snake_case
        // input is silently dropped (becomes None), `triggerAt` becomes null,
        // and the time-based intention never fires.
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": "Check build status",
            "trigger": {
                "type": "time",
                "in_minutes": 30
            }
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(
            value["triggerAt"].is_string(),
            "snake_case in_minutes should derive triggerAt; got: {:?}",
            value["triggerAt"]
        );
    }

    #[tokio::test]
    async fn test_set_action_with_file_pattern_snake_case() {
        // The public JSON schema declares `file_pattern` in snake_case. Verify
        // it survives deserialization by setting an intention with ONLY
        // file_pattern (no codebase — otherwise the check-side codebase branch
        // would short-circuit and mask a dropped file_pattern field).
        //
        // Note: file_pattern matching currently uses substring containment, not
        // glob, so the "pattern" must be a plain substring of the file path.
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "set",
            "description": "Review test files",
            "trigger": {
                "type": "context",
                "file_pattern": ".test.cjs"
            }
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(
            result.is_ok(),
            "set should succeed with snake_case file_pattern"
        );

        // Check should fire when a matching file is in context.
        let check_args = serde_json::json!({
            "action": "check",
            "context": {
                "file": "tests/neural-cascade.test.cjs"
            }
        });
        let check = execute(&storage, &test_cognitive(), Some(check_args))
            .await
            .unwrap();
        let triggered = check["triggered"].as_array().expect("triggered array");
        assert!(
            !triggered.is_empty(),
            "file_pattern must survive snake_case deserialization and match on file substring; \
             got triggered: {:?}, pending: {:?}",
            check["triggered"],
            check["pending"]
        );
    }

    #[tokio::test]
    async fn test_set_action_with_deadline() {
        let (storage, _dir) = test_storage().await;
        let deadline = (Utc::now() + Duration::days(7)).to_rfc3339();
        let args = serde_json::json!({
            "action": "set",
            "description": "Complete feature by end of week",
            "deadline": deadline
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["deadline"].is_string());
    }

    // ========================================================================
    // CHECK ACTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_check_action_empty_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "check" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["action"], "check");
        assert!(value["triggered"].is_array());
        assert!(value["pending"].is_array());
        assert!(value["checkedAt"].is_string());
    }

    #[tokio::test]
    async fn test_check_action_returns_pending() {
        let (storage, _dir) = test_storage().await;
        create_test_intention(&storage, "Future task").await;

        let args = serde_json::json!({ "action": "check" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let pending = value["pending"].as_array().unwrap();
        assert!(!pending.is_empty());
    }

    #[tokio::test]
    async fn test_check_action_with_context() {
        let (storage, _dir) = test_storage().await;

        // Create context-triggered intention
        let set_args = serde_json::json!({
            "action": "set",
            "description": "Check tests in payments",
            "trigger": {
                "type": "context",
                "codebase": "payments"
            }
        });
        execute(&storage, &test_cognitive(), Some(set_args))
            .await
            .unwrap();

        // Check with matching context
        let check_args = serde_json::json!({
            "action": "check",
            "context": {
                "codebase": "payments-service"
            }
        });
        let result = execute(&storage, &test_cognitive(), Some(check_args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let triggered = value["triggered"].as_array().unwrap();
        assert!(!triggered.is_empty());
    }

    #[tokio::test]
    async fn test_check_action_time_triggered() {
        let (storage, _dir) = test_storage().await;

        // Create time-triggered intention in the past
        let past_time = (Utc::now() - Duration::hours(1)).to_rfc3339();
        let set_args = serde_json::json!({
            "action": "set",
            "description": "Past due task",
            "trigger": {
                "type": "time",
                "at": past_time
            }
        });
        execute(&storage, &test_cognitive(), Some(set_args))
            .await
            .unwrap();

        let check_args = serde_json::json!({ "action": "check" });
        let result = execute(&storage, &test_cognitive(), Some(check_args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let triggered = value["triggered"].as_array().unwrap();
        assert!(!triggered.is_empty());
    }

    // ========================================================================
    // UPDATE ACTION TESTS - COMPLETE
    // ========================================================================

    #[tokio::test]
    async fn test_update_action_complete_succeeds() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "complete"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "update");
        assert_eq!(value["status"], "complete");
        assert!(value["message"].as_str().unwrap().contains("complete"));
    }

    #[tokio::test]
    async fn test_update_action_complete_nonexistent_fails() {
        let (storage, _dir) = test_storage().await;
        let fake_id = Uuid::new_v4().to_string();

        let args = serde_json::json!({
            "action": "update",
            "id": fake_id,
            "status": "complete"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_update_action_missing_id_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "update",
            "status": "complete"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'id'"));
    }

    #[tokio::test]
    async fn test_update_action_missing_status_fails() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'status'"));
    }

    // ========================================================================
    // UPDATE ACTION TESTS - SNOOZE
    // ========================================================================

    #[tokio::test]
    async fn test_update_action_snooze_succeeds() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to snooze").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "snooze",
            "snooze_minutes": 30
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["status"], "snooze");
        assert!(value["snoozedUntil"].is_string());
        assert!(value["message"].as_str().unwrap().contains("snoozed"));
    }

    #[tokio::test]
    async fn test_update_action_snooze_default_minutes() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task with default snooze").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "snooze"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["message"].as_str().unwrap().contains("30 minutes"));
    }

    // ========================================================================
    // UPDATE ACTION TESTS - CANCEL
    // ========================================================================

    #[tokio::test]
    async fn test_update_action_cancel_succeeds() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to cancel").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "cancel"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["status"], "cancel");
        assert!(value["message"].as_str().unwrap().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_update_action_unknown_status_fails() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task").await;

        let args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "invalid"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown status"));
    }

    // ========================================================================
    // LIST ACTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_list_action_empty_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "list" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["action"], "list");
        assert!(value["intentions"].is_array());
        assert_eq!(value["total"], 0);
        assert_eq!(value["status"], "active");
    }

    #[tokio::test]
    async fn test_list_action_returns_created() {
        let (storage, _dir) = test_storage().await;
        create_test_intention(&storage, "First task").await;
        create_test_intention(&storage, "Second task").await;

        let args = serde_json::json!({ "action": "list" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["total"], 2);
    }

    #[tokio::test]
    async fn test_list_action_filter_by_status() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;

        // Complete one
        let complete_args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "complete"
        });
        execute(&storage, &test_cognitive(), Some(complete_args))
            .await
            .unwrap();

        // Create another active one
        create_test_intention(&storage, "Active task").await;

        // List fulfilled
        let list_args = serde_json::json!({
            "action": "list",
            "filter_status": "fulfilled"
        });
        let result = execute(&storage, &test_cognitive(), Some(list_args))
            .await
            .unwrap();
        assert_eq!(result["total"], 1);
        assert_eq!(result["status"], "fulfilled");
    }

    #[tokio::test]
    async fn test_list_action_with_limit() {
        let (storage, _dir) = test_storage().await;
        for i in 0..5 {
            create_test_intention(&storage, &format!("Task {}", i)).await;
        }

        let args = serde_json::json!({
            "action": "list",
            "limit": 3
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let intentions = value["intentions"].as_array().unwrap();
        assert!(intentions.len() <= 3);
    }

    #[tokio::test]
    async fn test_list_action_all_status() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;
        create_test_intention(&storage, "Active task").await;

        // Complete one
        let complete_args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "complete"
        });
        execute(&storage, &test_cognitive(), Some(complete_args))
            .await
            .unwrap();

        // List all
        let list_args = serde_json::json!({
            "action": "list",
            "filter_status": "all"
        });
        let result = execute(&storage, &test_cognitive(), Some(list_args))
            .await
            .unwrap();
        assert_eq!(result["total"], 2);
    }

    // ========================================================================
    // FULL LIFECYCLE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_intention_full_lifecycle() {
        let (storage, _dir) = test_storage().await;

        // 1. Create intention
        let intention_id = create_test_intention(&storage, "Full lifecycle test").await;

        // 2. Verify it appears in list
        let list_args = serde_json::json!({ "action": "list" });
        let list_result = execute(&storage, &test_cognitive(), Some(list_args))
            .await
            .unwrap();
        assert_eq!(list_result["total"], 1);

        // 3. Snooze it
        let snooze_args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "snooze",
            "snooze_minutes": 5
        });
        let snooze_result = execute(&storage, &test_cognitive(), Some(snooze_args)).await;
        assert!(snooze_result.is_ok());

        // 4. Complete it
        let complete_args = serde_json::json!({
            "action": "update",
            "id": intention_id,
            "status": "complete"
        });
        let complete_result = execute(&storage, &test_cognitive(), Some(complete_args)).await;
        assert!(complete_result.is_ok());

        // 5. Verify it's no longer active
        let final_list_args = serde_json::json!({ "action": "list" });
        let final_list = execute(&storage, &test_cognitive(), Some(final_list_args))
            .await
            .unwrap();
        assert_eq!(final_list["total"], 0);

        // 6. Verify it's in fulfilled list
        let fulfilled_args = serde_json::json!({
            "action": "list",
            "filter_status": "fulfilled"
        });
        let fulfilled_list = execute(&storage, &test_cognitive(), Some(fulfilled_args))
            .await
            .unwrap();
        assert_eq!(fulfilled_list["total"], 1);
    }

    #[tokio::test]
    async fn test_intention_priority_ordering() {
        let (storage, _dir) = test_storage().await;

        // Create intentions with different priorities
        let args_low = serde_json::json!({
            "action": "set",
            "description": "Low priority task",
            "priority": "low"
        });
        execute(&storage, &test_cognitive(), Some(args_low))
            .await
            .unwrap();

        let args_critical = serde_json::json!({
            "action": "set",
            "description": "Critical task",
            "priority": "critical"
        });
        execute(&storage, &test_cognitive(), Some(args_critical))
            .await
            .unwrap();

        let args_normal = serde_json::json!({
            "action": "set",
            "description": "Normal task",
            "priority": "normal"
        });
        execute(&storage, &test_cognitive(), Some(args_normal))
            .await
            .unwrap();

        // List and verify ordering (critical should be first due to priority DESC ordering)
        let list_args = serde_json::json!({ "action": "list" });
        let list_result = execute(&storage, &test_cognitive(), Some(list_args))
            .await
            .unwrap();
        let intentions = list_result["intentions"].as_array().unwrap();

        assert!(intentions.len() >= 3);
        // Critical (4) should come before normal (2) and low (1)
        let first_priority = intentions[0]["priority"].as_str().unwrap();
        assert_eq!(first_priority, "critical");
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_required_action() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["action"].is_object());
        assert!(
            schema_value["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("action"))
        );
    }

    #[test]
    fn test_schema_has_action_enum() {
        let schema_value = schema();
        let action_enum = schema_value["properties"]["action"]["enum"]
            .as_array()
            .unwrap();
        assert!(action_enum.contains(&serde_json::json!("set")));
        assert!(action_enum.contains(&serde_json::json!("check")));
        assert!(action_enum.contains(&serde_json::json!("update")));
        assert!(action_enum.contains(&serde_json::json!("list")));
    }

    #[test]
    fn test_schema_has_set_parameters() {
        let schema_value = schema();
        assert!(schema_value["properties"]["description"].is_object());
        assert!(schema_value["properties"]["trigger"].is_object());
        assert!(schema_value["properties"]["priority"].is_object());
        assert!(schema_value["properties"]["deadline"].is_object());
    }

    #[test]
    fn test_schema_has_update_parameters() {
        let schema_value = schema();
        assert!(schema_value["properties"]["id"].is_object());
        assert!(schema_value["properties"]["status"].is_object());
        assert!(schema_value["properties"]["snooze_minutes"].is_object());
    }

    #[test]
    fn test_schema_has_check_parameters() {
        let schema_value = schema();
        assert!(schema_value["properties"]["context"].is_object());
        assert!(schema_value["properties"]["include_snoozed"].is_object());
    }

    #[test]
    fn test_schema_has_list_parameters() {
        let schema_value = schema();
        assert!(schema_value["properties"]["filter_status"].is_object());
        assert!(schema_value["properties"]["limit"].is_object());
    }

    // ========================================================================
    // v2.0.7 REGRESSION COVERAGE — include_snoozed actually wires through
    // ========================================================================

    /// `include_snoozed=true` must fold snoozed intentions back into the
    /// check pool so their triggers can still fire. Before v2.0.7 the flag
    /// was schema-advertised but runtime-ignored.
    #[tokio::test]
    async fn test_check_includes_snoozed_when_flag_set() {
        let (storage, _dir) = test_storage().await;

        // Create an intention, then snooze it.
        let id = create_test_intention(&storage, "snoozed test intention").await;
        let snooze_args = serde_json::json!({
            "action": "update",
            "id": id,
            "status": "snooze",
            "snooze_minutes": 1
        });
        execute(&storage, &test_cognitive(), Some(snooze_args))
            .await
            .unwrap();

        // Check with include_snoozed=true; snoozed intention should appear
        // in either triggered or pending.
        let check_args = serde_json::json!({
            "action": "check",
            "include_snoozed": true
        });
        let result = execute(&storage, &test_cognitive(), Some(check_args))
            .await
            .unwrap();
        let triggered = result["triggered"].as_array().unwrap();
        let pending = result["pending"].as_array().unwrap();
        let appears_anywhere = triggered
            .iter()
            .chain(pending.iter())
            .any(|v| v["id"].as_str() == Some(id.as_str()));
        assert!(
            appears_anywhere,
            "snoozed intention should be visible when include_snoozed=true"
        );
    }

    /// Default (include_snoozed omitted) must NOT surface snoozed intentions
    /// — this preserves the pre-v2.0.7 behavior for every caller that
    /// doesn't opt in.
    #[tokio::test]
    async fn test_check_excludes_snoozed_by_default() {
        let (storage, _dir) = test_storage().await;

        let id = create_test_intention(&storage, "default-excluded snoozed intention").await;
        let snooze_args = serde_json::json!({
            "action": "update",
            "id": id,
            "status": "snooze",
            "snooze_minutes": 1
        });
        execute(&storage, &test_cognitive(), Some(snooze_args))
            .await
            .unwrap();

        // Default check — no include_snoozed in args.
        let check_args = serde_json::json!({ "action": "check" });
        let result = execute(&storage, &test_cognitive(), Some(check_args))
            .await
            .unwrap();
        let triggered = result["triggered"].as_array().unwrap();
        let pending = result["pending"].as_array().unwrap();
        let appears_anywhere = triggered
            .iter()
            .chain(pending.iter())
            .any(|v| v["id"].as_str() == Some(id.as_str()));
        assert!(
            !appears_anywhere,
            "snoozed intention must NOT surface without include_snoozed=true"
        );
    }

    /// v2.0.7 also adds a `status` field to each check-result item so
    /// callers can tell active-triggered from snoozed-overdue. Verify the
    /// field is present and reflects the real storage state.
    #[tokio::test]
    async fn test_check_item_exposes_status_field() {
        let (storage, _dir) = test_storage().await;
        let _id = create_test_intention(&storage, "status-field test").await;
        let check_args = serde_json::json!({ "action": "check" });
        let result = execute(&storage, &test_cognitive(), Some(check_args))
            .await
            .unwrap();
        let pending = result["pending"].as_array().unwrap();
        assert!(!pending.is_empty(), "setup should produce one pending item");
        assert_eq!(
            pending[0]["status"], "active",
            "freshly-created intention must report status=\"active\""
        );
    }

    #[tokio::test]
    async fn test_public_recurring_trigger_advances_persistently_without_immediate_duplicate() {
        let (storage, _dir) = test_storage().await;
        let anchor = Utc::now() - Duration::minutes(1);
        let first_check = anchor + Duration::minutes(2);
        let set = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "repeatable reminder",
                "trigger": {
                    "type": "recurring",
                    "at": anchor.to_rfc3339(),
                    "recurrence": { "every": 5, "unit": "minutes" }
                }
            })),
        )
        .await
        .unwrap();
        let id = set["intentionId"].as_str().unwrap();

        let check = |current_time: DateTime<Utc>| {
            serde_json::json!({
                "action": "check",
                "context": { "current_time": current_time.to_rfc3339() }
            })
        };
        let first = execute(&storage, &test_cognitive(), Some(check(first_check)))
            .await
            .unwrap();
        assert_eq!(first["triggered"].as_array().unwrap().len(), 1);
        let next = parse_rfc3339(
            first["triggered"][0]["nextOccurrence"].as_str().unwrap(),
            "nextOccurrence",
        )
        .unwrap();
        assert!(next > first_check);

        let duplicate = execute(&storage, &test_cognitive(), Some(check(first_check)))
            .await
            .unwrap();
        assert!(duplicate["triggered"].as_array().unwrap().is_empty());

        let repeated = execute(&storage, &test_cognitive(), Some(check(next)))
            .await
            .unwrap();
        assert_eq!(repeated["triggered"].as_array().unwrap().len(), 1);
        let stored = storage.get_intention(id).unwrap().unwrap();
        assert_eq!(stored.reminder_count, 2);
        assert!(stored.last_reminded_at.is_some());
    }

    #[tokio::test]
    async fn test_recurring_base_event_is_required_and_snooze_suppresses_delivery() {
        let (storage, _dir) = test_storage().await;
        let anchor = Utc::now() - Duration::minutes(1);
        let set = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "event-gated recurrence",
                "trigger": {
                    "type": "recurring",
                    "at": anchor.to_rfc3339(),
                    "recurrence": "every 5 minutes",
                    "base": {
                        "type": "event",
                        "condition": "deployment completed"
                    }
                }
            })),
        )
        .await
        .unwrap();
        let id = set["intentionId"].as_str().unwrap().to_string();
        let due = anchor + Duration::minutes(2);

        let missing_event = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": { "current_time": due.to_rfc3339() }
            })),
        )
        .await
        .unwrap();
        assert!(missing_event["triggered"].as_array().unwrap().is_empty());

        execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "update",
                "id": id,
                "status": "snooze",
                "snooze_minutes": 1
            })),
        )
        .await
        .unwrap();
        let while_snoozed = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "include_snoozed": true,
                "context": {
                    "current_time": Utc::now().to_rfc3339(),
                    "events": ["deployment completed"]
                }
            })),
        )
        .await
        .unwrap();
        assert!(while_snoozed["triggered"].as_array().unwrap().is_empty());

        let after_snooze = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": {
                    "current_time": (Utc::now() + Duration::minutes(2)).to_rfc3339(),
                    "events": ["deployment completed successfully"]
                }
            })),
        )
        .await
        .unwrap();
        assert_eq!(after_snooze["triggered"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_compound_context_is_conjunctive_and_activity_event_branches_work() {
        let (storage, _dir) = test_storage().await;
        execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "compound public trigger",
                "trigger": {
                    "type": "compound",
                    "all_of": [{
                        "type": "context",
                        "codebase": "vestige",
                        "file_pattern": "intention_unified.rs"
                    }],
                    "any_of": [
                        { "type": "event", "condition": "review approved" },
                        { "type": "activity", "activity": "tests completed" }
                    ]
                }
            })),
        )
        .await
        .unwrap();

        let missing_file = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": {
                    "codebase": "vestige",
                    "file": "other.rs",
                    "events": ["tests completed"]
                }
            })),
        )
        .await
        .unwrap();
        assert!(missing_file["triggered"].as_array().unwrap().is_empty());

        let matching = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": {
                    "codebase": "vestige",
                    "file": "crates/vestige-mcp/src/tools/intention_unified.rs",
                    "events": ["all tests completed"]
                }
            })),
        )
        .await
        .unwrap();
        assert_eq!(matching["triggered"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_future_recurring_any_of_branch_does_not_disable_event_cooldown() {
        let (storage, _dir) = test_storage().await;
        let now = Utc::now();
        execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "mixed any-of cooldown",
                "trigger": {
                    "type": "compound",
                    "any_of": [
                        { "type": "event", "condition": "review approved" },
                        {
                            "type": "recurring",
                            "at": (now + Duration::hours(2)).to_rfc3339(),
                            "recurrence": "daily"
                        }
                    ]
                }
            })),
        )
        .await
        .unwrap();

        let check = serde_json::json!({
            "action": "check",
            "context": {
                "current_time": now.to_rfc3339(),
                "events": ["review approved"]
            }
        });
        let first = execute(&storage, &test_cognitive(), Some(check.clone()))
            .await
            .unwrap();
        assert_eq!(first["triggered"].as_array().unwrap().len(), 1);

        let duplicate = execute(&storage, &test_cognitive(), Some(check))
            .await
            .unwrap();
        assert!(duplicate["triggered"].as_array().unwrap().is_empty());
        assert_eq!(duplicate["pending"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_invalid_stored_trigger_is_visible_in_pending_item() {
        let (storage, _dir) = test_storage().await;
        let now = Utc::now();
        storage
            .save_intention(&IntentionRecord {
                id: "invalid-recurring".to_string(),
                content: "legacy partial recurrence".to_string(),
                trigger_type: "recurring".to_string(),
                trigger_data: serde_json::json!({ "type": "recurring" }).to_string(),
                priority: 2,
                status: "active".to_string(),
                created_at: now,
                deadline: None,
                fulfilled_at: None,
                reminder_count: 0,
                last_reminded_at: None,
                notes: None,
                tags: Vec::new(),
                related_memories: Vec::new(),
                snoozed_until: None,
                source_type: "legacy".to_string(),
                source_data: None,
            })
            .unwrap();

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "action": "check" })),
        )
        .await
        .unwrap();
        assert!(result["triggered"].as_array().unwrap().is_empty());
        assert!(result["pending"][0]["invalid_trigger"].is_string());
    }

    #[tokio::test]
    async fn test_nlp_fortnight_recurrence_is_persisted_in_full() {
        let (storage, _dir) = test_storage().await;
        let set = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "remind me to review the roadmap every fortnight"
            })),
        )
        .await
        .unwrap();
        assert_eq!(set["nlpParsed"], true);
        let id = set["intentionId"].as_str().unwrap();
        let stored = storage.get_intention(id).unwrap().unwrap();
        let trigger: Value = serde_json::from_str(&stored.trigger_data).unwrap();
        assert_eq!(trigger["type"], "recurring");
        assert_eq!(trigger["recurrence"]["every_minutes"], 20_160);
        assert!(trigger["base"].is_object());
        assert!(trigger["next_occurrence"].is_string());
    }

    #[tokio::test]
    async fn test_invalid_trigger_time_recurrence_depth_and_public_limits_fail_closed() {
        let (storage, _dir) = test_storage().await;
        for trigger in [
            serde_json::json!({ "type": "time", "at": "tomorrow" }),
            serde_json::json!({
                "type": "recurring",
                "recurrence": { "every_minutes": 0 }
            }),
            serde_json::json!({ "type": "compound", "all_of": [], "any_of": [] }),
            serde_json::json!({
                "type": "compound",
                "all_of": [{
                    "type": "compound",
                    "all_of": [{
                        "type": "compound",
                        "all_of": [{
                            "type": "compound",
                            "all_of": [{
                                "type": "compound",
                                "all_of": [{
                                    "type": "context",
                                    "topic": "too deep"
                                }]
                            }]
                        }]
                    }]
                }]
            }),
        ] {
            let result = execute(
                &storage,
                &test_cognitive(),
                Some(serde_json::json!({
                    "action": "set",
                    "description": "invalid",
                    "trigger": trigger
                })),
            )
            .await;
            assert!(result.is_err());
        }

        let bad_time = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": { "current_time": "not-rfc3339" }
            })),
        )
        .await;
        assert!(bad_time.is_err());

        let bad_snooze = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "update",
                "id": "missing",
                "status": "snooze",
                "snooze_minutes": 0
            })),
        )
        .await;
        assert!(bad_snooze.is_err());

        let bad_limit = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "action": "list", "limit": -1 })),
        )
        .await;
        assert!(bad_limit.is_err());

        let oversized_args = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "set",
                "description": "x".repeat(MAX_ARGUMENT_BYTES)
            })),
        )
        .await;
        assert!(oversized_args.is_err());

        let too_many_topics = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "action": "check",
                "context": {
                    "topics": vec!["topic"; MAX_CONTEXT_ITEMS + 1]
                }
            })),
        )
        .await;
        assert!(too_many_topics.is_err());
    }
}
