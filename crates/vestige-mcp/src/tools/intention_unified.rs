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
use vestige_core::neuroscience::ProspectiveContext;
use vestige_core::neuroscience::prospective_memory::IntentionTrigger as ProspectiveTrigger;

/// Unified schema for the `intention` tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "description": "Unified intention management tool. Supports setting, checking, updating (complete/snooze/cancel), and listing intentions.",
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
                "type": "object",
                "description": "[set] When to trigger this intention",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["time", "context", "event"],
                        "description": "Trigger type: time-based, context-based, or event-based"
                    },
                    "at": {
                        "type": "string",
                        "description": "ISO timestamp for time-based triggers"
                    },
                    "in_minutes": {
                        "type": "integer",
                        "description": "Minutes from now for duration-based triggers"
                    },
                    "codebase": {
                        "type": "string",
                        "description": "Trigger when working in this codebase"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Trigger when editing files matching this pattern"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Trigger when discussing this topic"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Natural language condition for event triggers"
                    }
                }
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
                        "description": "Current ISO timestamp (defaults to now)"
                    },
                    "codebase": {
                        "type": "string",
                        "description": "Current codebase/project name"
                    },
                    "file": {
                        "type": "string",
                        "description": "Current file path"
                    },
                    "topics": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Current discussion topics"
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

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TriggerSpec {
    #[serde(rename = "type")]
    trigger_type: Option<String>,
    at: Option<String>,
    #[serde(alias = "in_minutes")]
    in_minutes: Option<i64>,
    codebase: Option<String>,
    #[serde(alias = "file_pattern")]
    file_pattern: Option<String>,
    topic: Option<String>,
    condition: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContextSpec {
    #[allow(dead_code)]
    current_time: Option<String>,
    codebase: Option<String>,
    file: Option<String>,
    topics: Option<Vec<String>>,
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
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
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
    let mut nlp_trigger_type = None;
    let mut nlp_trigger_data = None;
    let mut nlp_priority = None;
    let mut tags = Vec::new();

    if let Ok(cog) = cognitive.try_lock() {
        // 8A. Try NLP parsing when no explicit trigger is provided
        if args.trigger.is_none()
            && let Ok(parsed) = cog.intention_parser.parse(description)
        {
            nlp_parsed = true;
            // Extract trigger info from parsed intention
            let (t_type, t_data) = match &parsed.trigger {
                ProspectiveTrigger::TimeBased { .. } => (
                    "time".to_string(),
                    serde_json::json!({"type": "time"}).to_string(),
                ),
                ProspectiveTrigger::DurationBased { after, .. } => {
                    let mins = after.num_minutes();
                    (
                        "time".to_string(),
                        serde_json::json!({"type": "time", "in_minutes": mins}).to_string(),
                    )
                }
                ProspectiveTrigger::EventBased { condition, .. } => (
                    "event".to_string(),
                    serde_json::json!({"type": "event", "condition": condition}).to_string(),
                ),
                ProspectiveTrigger::ContextBased { context_match } => (
                    "context".to_string(),
                    serde_json::json!({"type": "context", "topic": format!("{:?}", context_match)})
                        .to_string(),
                ),
                ProspectiveTrigger::Recurring { .. } => (
                    "recurring".to_string(),
                    serde_json::json!({"type": "recurring"}).to_string(),
                ),
                _ => (
                    "event".to_string(),
                    serde_json::json!({"type": "event"}).to_string(),
                ),
            };
            nlp_trigger_type = Some(t_type);
            nlp_trigger_data = Some(t_data);

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

    // Determine trigger type and data (explicit > NLP > manual)
    let (trigger_type, trigger_data) = if let Some(trigger) = &args.trigger {
        let t_type = trigger
            .trigger_type
            .clone()
            .unwrap_or_else(|| "time".to_string());
        let data = serde_json::to_string(trigger).unwrap_or_else(|_| "{}".to_string());
        (t_type, data)
    } else if let (Some(t_type), Some(t_data)) = (nlp_trigger_type, nlp_trigger_data) {
        (t_type, t_data)
    } else {
        ("manual".to_string(), "{}".to_string())
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
    let deadline = args.deadline.as_ref().and_then(|s| {
        DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    });

    // Calculate trigger time if specified
    let trigger_at = if let Some(trigger) = &args.trigger {
        if let Some(at) = &trigger.at {
            DateTime::parse_from_rfc3339(at)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        } else {
            trigger.in_minutes.map(|mins| now + Duration::minutes(mins))
        }
    } else {
        None
    };

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
    let now = Utc::now();

    // ====================================================================
    // COGNITIVE: Update prospective memory context
    // ====================================================================
    if let Some(ctx) = &args.context
        && let Ok(cog) = cognitive.try_lock()
    {
        let mut prospective_ctx = ProspectiveContext::new();
        if let Some(codebase) = &ctx.codebase {
            prospective_ctx.project_name = Some(codebase.clone());
        }
        if let Some(file) = &ctx.file {
            prospective_ctx.active_files = vec![file.clone()];
        }
        if let Some(topics) = &ctx.topics {
            prospective_ctx.active_topics = topics.clone();
        }
        // Update context on prospective memory (triggers internal monitoring)
        let _ = cog.prospective_memory.update_context(prospective_ctx);
    }

    // Get active intentions. `include_snoozed=true` folds snoozed intentions
    // back into the check pool so time/context triggers can wake them up when
    // their snooze window has elapsed or a matching context appears — before
    // v2.0.7 this flag was advertised in the schema but silently ignored, so
    // snoozed intentions were invisible to `check` regardless of the arg.
    let mut intentions = storage.get_active_intentions().map_err(|e| e.to_string())?;
    if args.include_snoozed.unwrap_or(false) {
        let snoozed = storage
            .get_intentions_by_status("snoozed")
            .map_err(|e| e.to_string())?;
        // Deduplicate defensively — an intention should never be in both lists
        // but we pay the O(n) hash-set cost to stay correct if storage semantics
        // shift under us.
        use std::collections::HashSet;
        let seen: HashSet<String> = intentions.iter().map(|i| i.id.clone()).collect();
        for s in snoozed {
            if !seen.contains(&s.id) {
                intentions.push(s);
            }
        }
    }

    let mut triggered = Vec::new();
    let mut pending = Vec::new();

    for intention in intentions {
        // Parse trigger data
        let trigger: Option<TriggerSpec> = serde_json::from_str(&intention.trigger_data).ok();

        // Check if triggered
        let is_triggered = if let Some(t) = &trigger {
            match t.trigger_type.as_deref() {
                Some("time") => {
                    if let Some(at) = &t.at {
                        if let Ok(trigger_time) = DateTime::parse_from_rfc3339(at) {
                            trigger_time.with_timezone(&Utc) <= now
                        } else {
                            false
                        }
                    } else if let Some(mins) = t.in_minutes {
                        let trigger_time = intention.created_at + Duration::minutes(mins);
                        trigger_time <= now
                    } else {
                        false
                    }
                }
                Some("context") => {
                    if let Some(ctx) = &args.context {
                        // Check codebase match
                        if let (Some(trigger_codebase), Some(current_codebase)) =
                            (&t.codebase, &ctx.codebase)
                        {
                            current_codebase
                                .to_lowercase()
                                .contains(&trigger_codebase.to_lowercase())
                        // Check file pattern match
                        } else if let (Some(pattern), Some(file)) = (&t.file_pattern, &ctx.file) {
                            file.contains(pattern)
                        // Check topic match
                        } else if let (Some(topic), Some(topics)) = (&t.topic, &ctx.topics) {
                            topics
                                .iter()
                                .any(|t| t.to_lowercase().contains(&topic.to_lowercase()))
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        };

        // Check if overdue
        let is_overdue = intention.deadline.map(|d| d < now).unwrap_or(false);

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
        });

        if is_triggered || is_overdue {
            triggered.push(item);
        } else {
            pending.push(item);
        }
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

    let limit = args.limit.unwrap_or(20) as usize;
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
}
