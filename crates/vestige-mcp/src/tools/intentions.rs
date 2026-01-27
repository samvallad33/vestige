#![allow(dead_code)]
//! Intentions Tools (Deprecated - use intention_unified instead)
//!
//! Prospective memory tools for setting and checking future intentions.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

use vestige_core::{IntentionRecord, Storage};

/// Schema for set_intention tool
pub fn set_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What to remember to do"
            },
            "trigger": {
                "type": "object",
                "description": "When to trigger this intention",
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
                "description": "Priority level"
            },
            "deadline": {
                "type": "string",
                "description": "Optional deadline (ISO timestamp)"
            }
        },
        "required": ["description"]
    })
}

/// Schema for check_intentions tool
pub fn check_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "context": {
                "type": "object",
                "description": "Current context for matching intentions",
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
                "description": "Include snoozed intentions"
            }
        }
    })
}

/// Schema for complete_intention tool
pub fn complete_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "intentionId": {
                "type": "string",
                "description": "ID of the intention to mark as complete"
            }
        },
        "required": ["intentionId"]
    })
}

/// Schema for snooze_intention tool
pub fn snooze_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "intentionId": {
                "type": "string",
                "description": "ID of the intention to snooze"
            },
            "minutes": {
                "type": "integer",
                "description": "Minutes to snooze for",
                "default": 30
            }
        },
        "required": ["intentionId"]
    })
}

/// Schema for list_intentions tool
pub fn list_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "fulfilled", "cancelled", "snoozed", "all"],
                "default": "active",
                "description": "Filter by status"
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Maximum number to return"
            }
        }
    })
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TriggerSpec {
    #[serde(rename = "type")]
    trigger_type: Option<String>,
    at: Option<String>,
    in_minutes: Option<i64>,
    codebase: Option<String>,
    file_pattern: Option<String>,
    topic: Option<String>,
    condition: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetIntentionArgs {
    description: String,
    trigger: Option<TriggerSpec>,
    priority: Option<String>,
    deadline: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContextSpec {
    #[allow(dead_code)] // Deserialized from JSON but not yet used in context matching
    current_time: Option<String>,
    codebase: Option<String>,
    file: Option<String>,
    topics: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckIntentionsArgs {
    context: Option<ContextSpec>,
    #[allow(dead_code)] // Deserialized from JSON for future snoozed intentions filter
    include_snoozed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IntentionIdArgs {
    intention_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnoozeArgs {
    intention_id: String,
    minutes: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListArgs {
    status: Option<String>,
    limit: Option<i32>,
}

/// Execute set_intention tool
pub async fn execute_set(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SetIntentionArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.description.trim().is_empty() {
        return Err("Description cannot be empty".to_string());
    }

    let now = Utc::now();
    let id = Uuid::new_v4().to_string();

    // Determine trigger type and data
    let (trigger_type, trigger_data) = if let Some(trigger) = &args.trigger {
        let t_type = trigger.trigger_type.clone().unwrap_or_else(|| "time".to_string());
        let data = serde_json::to_string(trigger).unwrap_or_else(|_| "{}".to_string());
        (t_type, data)
    } else {
        ("manual".to_string(), "{}".to_string())
    };

    // Parse priority
    let priority = match args.priority.as_deref() {
        Some("low") => 1,
        Some("high") => 3,
        Some("critical") => 4,
        _ => 2, // normal
    };

    // Parse deadline
    let deadline = args.deadline.and_then(|s| {
        DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))
    });

    // Calculate trigger time if specified
    let trigger_at = if let Some(trigger) = &args.trigger {
        if let Some(at) = &trigger.at {
            DateTime::parse_from_rfc3339(at).ok().map(|dt| dt.with_timezone(&Utc))
        } else if let Some(mins) = trigger.in_minutes {
            Some(now + Duration::minutes(mins))
        } else {
            None
        }
    } else {
        None
    };

    let record = IntentionRecord {
        id: id.clone(),
        content: args.description.clone(),
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
        tags: vec![],
        related_memories: vec![],
        snoozed_until: None,
        source_type: "mcp".to_string(),
        source_data: None,
    };

    let mut storage = storage.lock().await;
    storage.save_intention(&record).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "success": true,
        "intentionId": id,
        "message": format!("Intention created: {}", args.description),
        "priority": priority,
        "triggerAt": trigger_at.map(|dt| dt.to_rfc3339()),
        "deadline": deadline.map(|dt| dt.to_rfc3339()),
    }))
}

/// Execute check_intentions tool
pub async fn execute_check(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: CheckIntentionsArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => CheckIntentionsArgs { context: None, include_snoozed: None },
    };

    let now = Utc::now();
    let storage = storage.lock().await;

    // Get active intentions
    let intentions = storage.get_active_intentions().map_err(|e| e.to_string())?;

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
                        if let (Some(trigger_codebase), Some(current_codebase)) = (&t.codebase, &ctx.codebase) {
                            current_codebase.to_lowercase().contains(&trigger_codebase.to_lowercase())
                        // Check file pattern match
                        } else if let (Some(pattern), Some(file)) = (&t.file_pattern, &ctx.file) {
                            file.contains(pattern)
                        // Check topic match
                        } else if let (Some(topic), Some(topics)) = (&t.topic, &ctx.topics) {
                            topics.iter().any(|t| t.to_lowercase().contains(&topic.to_lowercase()))
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
            "priority": match intention.priority {
                1 => "low",
                3 => "high",
                4 => "critical",
                _ => "normal",
            },
            "createdAt": intention.created_at.to_rfc3339(),
            "deadline": intention.deadline.map(|d| d.to_rfc3339()),
            "isOverdue": is_overdue,
        });

        if is_triggered || is_overdue {
            triggered.push(item);
        } else {
            pending.push(item);
        }
    }

    Ok(serde_json::json!({
        "triggered": triggered,
        "pending": pending,
        "checkedAt": now.to_rfc3339(),
    }))
}

/// Execute complete_intention tool
pub async fn execute_complete(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: IntentionIdArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing intention_id".to_string()),
    };

    let mut storage = storage.lock().await;
    let updated = storage.update_intention_status(&args.intention_id, "fulfilled")
        .map_err(|e| e.to_string())?;

    if updated {
        Ok(serde_json::json!({
            "success": true,
            "message": "Intention marked as complete",
            "intentionId": args.intention_id,
        }))
    } else {
        Err(format!("Intention not found: {}", args.intention_id))
    }
}

/// Execute snooze_intention tool
pub async fn execute_snooze(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SnoozeArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing intention_id".to_string()),
    };

    let minutes = args.minutes.unwrap_or(30);
    let snooze_until = Utc::now() + Duration::minutes(minutes);

    let mut storage = storage.lock().await;
    let updated = storage.snooze_intention(&args.intention_id, snooze_until)
        .map_err(|e| e.to_string())?;

    if updated {
        Ok(serde_json::json!({
            "success": true,
            "message": format!("Intention snoozed for {} minutes", minutes),
            "intentionId": args.intention_id,
            "snoozedUntil": snooze_until.to_rfc3339(),
        }))
    } else {
        Err(format!("Intention not found: {}", args.intention_id))
    }
}

/// Execute list_intentions tool
pub async fn execute_list(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: ListArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => ListArgs { status: None, limit: None },
    };

    let status = args.status.as_deref().unwrap_or("active");
    let storage = storage.lock().await;

    let intentions = if status == "all" {
        // Get all by combining different statuses
        let mut all = storage.get_active_intentions().map_err(|e| e.to_string())?;
        all.extend(storage.get_intentions_by_status("fulfilled").map_err(|e| e.to_string())?);
        all.extend(storage.get_intentions_by_status("cancelled").map_err(|e| e.to_string())?);
        all.extend(storage.get_intentions_by_status("snoozed").map_err(|e| e.to_string())?);
        all
    } else if status == "active" {
        // Use get_active_intentions for proper priority ordering
        storage.get_active_intentions().map_err(|e| e.to_string())?
    } else {
        storage.get_intentions_by_status(status).map_err(|e| e.to_string())?
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
        "intentions": items,
        "total": items.len(),
        "status": status,
    }))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Mutex<Storage>>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(Mutex::new(storage)), dir)
    }

    /// Helper to create an intention and return its ID
    async fn create_test_intention(storage: &Arc<Mutex<Storage>>, description: &str) -> String {
        let args = serde_json::json!({
            "description": description
        });
        let result = execute_set(storage, Some(args)).await.unwrap();
        result["intentionId"].as_str().unwrap().to_string()
    }

    // ========================================================================
    // SET_INTENTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_set_intention_empty_description_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "description": "" });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_set_intention_whitespace_only_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "description": "   \t\n  " });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_set_intention_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_set(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_set_intention_basic_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "description": "Remember to write unit tests"
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["intentionId"].is_string());
        assert!(value["message"].as_str().unwrap().contains("Intention created"));
    }

    #[tokio::test]
    async fn test_set_intention_with_priority() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "description": "Critical bug fix needed",
            "priority": "critical"
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["priority"], 4); // critical = 4
    }

    #[tokio::test]
    async fn test_set_intention_with_time_trigger() {
        let (storage, _dir) = test_storage().await;
        let future_time = (Utc::now() + Duration::hours(1)).to_rfc3339();
        let args = serde_json::json!({
            "description": "Meeting reminder",
            "trigger": {
                "type": "time",
                "at": future_time
            }
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["triggerAt"].is_string());
    }

    #[tokio::test]
    async fn test_set_intention_with_duration_trigger() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "description": "Check build status",
            "trigger": {
                "type": "time",
                "inMinutes": 30
            }
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["triggerAt"].is_string());
    }

    #[tokio::test]
    async fn test_set_intention_with_context_trigger() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "description": "Review error handling",
            "trigger": {
                "type": "context",
                "codebase": "payments"
            }
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_set_intention_with_deadline() {
        let (storage, _dir) = test_storage().await;
        let deadline = (Utc::now() + Duration::days(7)).to_rfc3339();
        let args = serde_json::json!({
            "description": "Complete feature by end of week",
            "deadline": deadline
        });
        let result = execute_set(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["deadline"].is_string());
    }

    // ========================================================================
    // CHECK_INTENTIONS TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_check_intentions_empty_succeeds() {
        let (storage, _dir) = test_storage().await;
        let result = execute_check(&storage, None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["triggered"].is_array());
        assert!(value["pending"].is_array());
        assert!(value["checkedAt"].is_string());
    }

    #[tokio::test]
    async fn test_check_intentions_returns_pending() {
        let (storage, _dir) = test_storage().await;
        // Create an intention without immediate trigger
        create_test_intention(&storage, "Future task").await;

        let result = execute_check(&storage, None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let pending = value["pending"].as_array().unwrap();
        assert!(!pending.is_empty());
    }

    #[tokio::test]
    async fn test_check_intentions_with_context() {
        let (storage, _dir) = test_storage().await;

        // Create context-triggered intention
        let args = serde_json::json!({
            "description": "Check tests in payments",
            "trigger": {
                "type": "context",
                "codebase": "payments"
            }
        });
        execute_set(&storage, Some(args)).await.unwrap();

        // Check with matching context
        let check_args = serde_json::json!({
            "context": {
                "codebase": "payments-service"
            }
        });
        let result = execute_check(&storage, Some(check_args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let triggered = value["triggered"].as_array().unwrap();
        assert!(!triggered.is_empty());
    }

    #[tokio::test]
    async fn test_check_intentions_time_triggered() {
        let (storage, _dir) = test_storage().await;

        // Create time-triggered intention in the past
        let past_time = (Utc::now() - Duration::hours(1)).to_rfc3339();
        let args = serde_json::json!({
            "description": "Past due task",
            "trigger": {
                "type": "time",
                "at": past_time
            }
        });
        execute_set(&storage, Some(args)).await.unwrap();

        let result = execute_check(&storage, None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let triggered = value["triggered"].as_array().unwrap();
        assert!(!triggered.is_empty());
    }

    // ========================================================================
    // COMPLETE_INTENTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_complete_intention_succeeds() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;

        let args = serde_json::json!({
            "intentionId": intention_id
        });
        let result = execute_complete(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["message"].as_str().unwrap().contains("complete"));
    }

    #[tokio::test]
    async fn test_complete_intention_nonexistent_fails() {
        let (storage, _dir) = test_storage().await;
        let fake_id = Uuid::new_v4().to_string();

        let args = serde_json::json!({
            "intentionId": fake_id
        });
        let result = execute_complete(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_complete_intention_missing_id_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_complete(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing intention_id"));
    }

    #[tokio::test]
    async fn test_completed_intention_not_in_active_list() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to hide").await;

        // Complete it
        let args = serde_json::json!({ "intentionId": intention_id });
        execute_complete(&storage, Some(args)).await.unwrap();

        // Check active intentions - should not include completed
        let list_args = serde_json::json!({ "status": "active" });
        let result = execute_list(&storage, Some(list_args)).await.unwrap();
        let intentions = result["intentions"].as_array().unwrap();

        let ids: Vec<&str> = intentions
            .iter()
            .map(|i| i["id"].as_str().unwrap())
            .collect();
        assert!(!ids.contains(&intention_id.as_str()));
    }

    // ========================================================================
    // SNOOZE_INTENTION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_snooze_intention_succeeds() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to snooze").await;

        let args = serde_json::json!({
            "intentionId": intention_id,
            "minutes": 30
        });
        let result = execute_snooze(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["snoozedUntil"].is_string());
        assert!(value["message"].as_str().unwrap().contains("snoozed"));
    }

    #[tokio::test]
    async fn test_snooze_intention_default_minutes() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task with default snooze").await;

        let args = serde_json::json!({
            "intentionId": intention_id
        });
        let result = execute_snooze(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["message"].as_str().unwrap().contains("30 minutes"));
    }

    #[tokio::test]
    async fn test_snooze_intention_nonexistent_fails() {
        let (storage, _dir) = test_storage().await;
        let fake_id = Uuid::new_v4().to_string();

        let args = serde_json::json!({
            "intentionId": fake_id,
            "minutes": 15
        });
        let result = execute_snooze(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_snooze_intention_missing_id_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute_snooze(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing intention_id"));
    }

    // ========================================================================
    // LIST_INTENTIONS TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_list_intentions_empty_succeeds() {
        let (storage, _dir) = test_storage().await;
        let result = execute_list(&storage, None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["intentions"].is_array());
        assert_eq!(value["total"], 0);
        assert_eq!(value["status"], "active");
    }

    #[tokio::test]
    async fn test_list_intentions_returns_created() {
        let (storage, _dir) = test_storage().await;
        create_test_intention(&storage, "First task").await;
        create_test_intention(&storage, "Second task").await;

        let result = execute_list(&storage, None).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["total"], 2);
    }

    #[tokio::test]
    async fn test_list_intentions_filter_by_status() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;

        // Complete one
        let args = serde_json::json!({ "intentionId": intention_id });
        execute_complete(&storage, Some(args)).await.unwrap();

        // Create another active one
        create_test_intention(&storage, "Active task").await;

        // List fulfilled
        let list_args = serde_json::json!({ "status": "fulfilled" });
        let result = execute_list(&storage, Some(list_args)).await.unwrap();
        assert_eq!(result["total"], 1);
        assert_eq!(result["status"], "fulfilled");
    }

    #[tokio::test]
    async fn test_list_intentions_with_limit() {
        let (storage, _dir) = test_storage().await;
        for i in 0..5 {
            create_test_intention(&storage, &format!("Task {}", i)).await;
        }

        let args = serde_json::json!({ "limit": 3 });
        let result = execute_list(&storage, Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let intentions = value["intentions"].as_array().unwrap();
        assert!(intentions.len() <= 3);
    }

    #[tokio::test]
    async fn test_list_intentions_all_status() {
        let (storage, _dir) = test_storage().await;
        let intention_id = create_test_intention(&storage, "Task to complete").await;
        create_test_intention(&storage, "Active task").await;

        // Complete one
        let args = serde_json::json!({ "intentionId": intention_id });
        execute_complete(&storage, Some(args)).await.unwrap();

        // List all
        let list_args = serde_json::json!({ "status": "all" });
        let result = execute_list(&storage, Some(list_args)).await.unwrap();
        assert_eq!(result["total"], 2);
    }

    // ========================================================================
    // INTENTION LIFECYCLE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_intention_full_lifecycle() {
        let (storage, _dir) = test_storage().await;

        // 1. Create intention
        let intention_id = create_test_intention(&storage, "Full lifecycle test").await;

        // 2. Verify it appears in list
        let list_result = execute_list(&storage, None).await.unwrap();
        assert_eq!(list_result["total"], 1);

        // 3. Snooze it
        let snooze_args = serde_json::json!({
            "intentionId": intention_id,
            "minutes": 5
        });
        let snooze_result = execute_snooze(&storage, Some(snooze_args)).await;
        assert!(snooze_result.is_ok());

        // 4. Complete it
        let complete_args = serde_json::json!({ "intentionId": intention_id });
        let complete_result = execute_complete(&storage, Some(complete_args)).await;
        assert!(complete_result.is_ok());

        // 5. Verify it's no longer active
        let final_list = execute_list(&storage, None).await.unwrap();
        assert_eq!(final_list["total"], 0);

        // 6. Verify it's in fulfilled list
        let fulfilled_args = serde_json::json!({ "status": "fulfilled" });
        let fulfilled_list = execute_list(&storage, Some(fulfilled_args)).await.unwrap();
        assert_eq!(fulfilled_list["total"], 1);
    }

    #[tokio::test]
    async fn test_intention_priority_ordering() {
        let (storage, _dir) = test_storage().await;

        // Create intentions with different priorities
        let args_low = serde_json::json!({
            "description": "Low priority task",
            "priority": "low"
        });
        execute_set(&storage, Some(args_low)).await.unwrap();

        let args_critical = serde_json::json!({
            "description": "Critical task",
            "priority": "critical"
        });
        execute_set(&storage, Some(args_critical)).await.unwrap();

        let args_normal = serde_json::json!({
            "description": "Normal task",
            "priority": "normal"
        });
        execute_set(&storage, Some(args_normal)).await.unwrap();

        // List and verify ordering (critical should be first due to priority DESC ordering)
        let list_result = execute_list(&storage, None).await.unwrap();
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
    fn test_set_schema_has_required_fields() {
        let schema_value = set_schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["description"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("description")));
    }

    #[test]
    fn test_complete_schema_has_required_fields() {
        let schema_value = complete_schema();
        assert!(schema_value["properties"]["intentionId"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("intentionId")));
    }

    #[test]
    fn test_snooze_schema_has_required_fields() {
        let schema_value = snooze_schema();
        assert!(schema_value["properties"]["intentionId"].is_object());
        assert!(schema_value["properties"]["minutes"].is_object());
        assert!(schema_value["required"].as_array().unwrap().contains(&serde_json::json!("intentionId")));
    }

    #[test]
    fn test_list_schema_has_optional_fields() {
        let schema_value = list_schema();
        assert!(schema_value["properties"]["status"].is_object());
        assert!(schema_value["properties"]["limit"].is_object());
    }

    #[test]
    fn test_check_schema_has_context_field() {
        let schema_value = check_schema();
        assert!(schema_value["properties"]["context"].is_object());
    }
}
