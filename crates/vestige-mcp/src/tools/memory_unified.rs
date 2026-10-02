//! Unified Memory Tool
//!
//! Merges get_knowledge, delete_knowledge, and get_memory_state into a single
//! `memory` tool with action-based dispatch.

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{MemoryState, Modification, OutcomeType, Storage};

// Accessibility thresholds based on retention strength
const ACCESSIBILITY_ACTIVE: f64 = 0.7;
const ACCESSIBILITY_DORMANT: f64 = 0.4;
const ACCESSIBILITY_SILENT: f64 = 0.1;

/// UUID or a Strata node id (`mem-` + 16 hex digits).
pub fn is_memory_id(id: &str) -> bool {
    if uuid::Uuid::parse_str(id).is_ok() {
        return true;
    }
    let Some(rest) = id.strip_prefix("mem-") else {
        return false;
    };
    rest.len() == 16 && rest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Compute accessibility score from memory strengths
/// Combines retention, retrieval, and storage strengths
fn compute_accessibility(retention: f64, retrieval: f64, storage: f64) -> f64 {
    // Weighted combination: retention is most important for accessibility
    retention * 0.5 + retrieval * 0.3 + storage * 0.2
}

/// Determine memory state from accessibility score
fn state_from_accessibility(accessibility: f64) -> MemoryState {
    if accessibility >= ACCESSIBILITY_ACTIVE {
        MemoryState::Active
    } else if accessibility >= ACCESSIBILITY_DORMANT {
        MemoryState::Dormant
    } else if accessibility >= ACCESSIBILITY_SILENT {
        MemoryState::Silent
    } else {
        MemoryState::Unavailable
    }
}

/// Input schema for the unified memory tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "get_batch", "delete", "purge", "state", "promote", "demote", "edit"],
                "description": "'get', 'get_batch' (ids), 'state', 'promote' / 'demote' (retrieval strength; demote never deletes), 'edit' (admit a successor under rule edit, then retire the previous node), 'purge' (retired, can't be retrieved; confirm=true). 'delete' aliases purge"
            },
            "id": {
                "type": "string",
                "description": "Memory id (single-memory actions)."
            },
            "ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "[get_batch] Memory ids, max 20."
            },
            "reason": {
                "type": "string",
                "description": "Why (optional, logged)."
            },
            "role": {
                "type": "string",
                "description": "[promote/demote] CLAIMED role for provenance. Resolved against the operator-controlled policy; claims never override the process identity or self-grant authority. Unregistered claims stay neutral at 1.0."
            },
            "confirm": {
                "type": "boolean",
                "description": "Required for purge and delete. Retires the memory so it can't be retrieved.",
                "default": false
            },
            "content": {
                "type": "string",
                "description": "[edit] Content for the successor. The previous node is retired under rule edit; its FSRS card is not copied."
            }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MemoryArgs {
    action: String,
    id: Option<String>,
    ids: Option<Vec<String>>,
    reason: Option<String>,
    confirm: Option<bool>,
    content: Option<String>,
    /// #252 Phase A: claimed role for provenance. Never overrides the
    /// process identity and never self-grants authority.
    #[serde(alias = "claimed_role")]
    role: Option<String>,
}

/// Execute the unified memory tool
/// Standalone `purge` tool schema (#219). Same parameters as
/// `memory(action='purge')`. Hosts can prompt on this tool alone.
pub fn purge_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "description": "Memory id to retire (confirm=true).",
                "type": "string"
            },
            "role": {
                "type": "string",
                "description": "[promote/demote] CLAIMED role for provenance. Resolved against the operator-controlled policy; claims never override the process identity or self-grant authority. Unregistered claims stay neutral at 1.0."
            },
            "confirm": {
                "description": "Required. Retires the memory so it can't be retrieved.",
                "type": "boolean"
            },
            "reason": {
                "description": "Why (optional, logged).",
                "type": "string"
            }
        },
        "required": ["id", "confirm"]
    })
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: MemoryArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // get_batch uses 'ids' array, all other actions use 'id'
    if args.action == "get_batch" {
        let ids = args.ids.ok_or("get_batch requires 'ids' array")?;
        if ids.is_empty() {
            return Err("ids array cannot be empty".to_string());
        }
        if ids.len() > 20 {
            return Err("get_batch supports max 20 IDs per call".to_string());
        }
        for id in &ids {
            if !is_memory_id(id) {
                return Err(format!("Invalid memory ID format: {}", id));
            }
        }
        return execute_get_batch(storage, &ids).await;
    }

    // All other actions require 'id'
    let id = args.id.ok_or("This action requires 'id' parameter")?;
    if !is_memory_id(&id) {
        return Err("Invalid memory ID format".to_string());
    }

    match args.action.as_str() {
        "get" => execute_get(storage, &id).await,
        "delete" => {
            execute_purge(
                storage,
                cognitive,
                &id,
                args.reason,
                args.confirm.unwrap_or(false),
                "delete",
            )
            .await
        }
        "purge" => {
            execute_purge(
                storage,
                cognitive,
                &id,
                args.reason,
                args.confirm.unwrap_or(false),
                "purge",
            )
            .await
        }
        "state" => execute_state(storage, &id).await,
        "promote" => {
            execute_promote(storage, cognitive, &id, args.reason, args.role.as_deref()).await
        }
        "demote" => {
            execute_demote(storage, cognitive, &id, args.reason, args.role.as_deref()).await
        }
        "edit" => execute_edit(storage, &id, args.content).await,
        _ => Err(format!(
            "Invalid action '{}'. Must be one of: get, get_batch, delete, purge, state, promote, demote, edit",
            args.action
        )),
    }
}

/// Get full memory node with all metadata
async fn execute_get(storage: &Arc<Storage>, id: &str) -> Result<Value, String> {
    let node = storage.get_node(id).map_err(|e| e.to_string())?;

    match node {
        Some(n) => Ok(serde_json::json!({
            "action": "get",
            "found": true,
            "node": {
                "id": n.id,
                "content": n.content,
                "nodeType": n.node_type,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "lastAccessed": n.last_accessed.to_rfc3339(),
                "stability": n.stability,
                "difficulty": n.difficulty,
                "reps": n.reps,
                "lapses": n.lapses,
                "storageStrength": n.storage_strength,
                "retrievalStrength": n.retrieval_strength,
                "retentionStrength": n.retention_strength,
                "sentimentScore": n.sentiment_score,
                "sentimentMagnitude": n.sentiment_magnitude,
                "nextReview": n.next_review.map(|d| d.to_rfc3339()),
                "source": n.source,
                "tags": n.tags,
                "hasEmbedding": n.has_embedding,
                "embeddingModel": n.embedding_model,
            }
        })),
        None => Ok(serde_json::json!({
            "action": "get",
            "found": false,
            "nodeId": id,
            "message": absent_message(storage, id),
        })),
    }
}

/// Get multiple full memory nodes by ID (batch retrieval for expandable IDs)
async fn execute_get_batch(storage: &Arc<Storage>, ids: &[String]) -> Result<Value, String> {
    let mut results = Vec::with_capacity(ids.len());
    let mut found_count = 0;

    for id in ids {
        match storage.get_node(id) {
            Ok(Some(n)) => {
                found_count += 1;
                results.push(serde_json::json!({
                    "id": n.id,
                    "content": n.content,
                    "nodeType": n.node_type,
                    "createdAt": n.created_at.to_rfc3339(),
                    "updatedAt": n.updated_at.to_rfc3339(),
                    "tags": n.tags,
                    "retentionStrength": n.retention_strength,
                    "source": n.source,
                }));
            }
            Ok(None) => {
                results.push(serde_json::json!({
                    "id": id,
                    "found": false,
                    "message": absent_message(storage, id),
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "id": id,
                    "error": e.to_string(),
                }));
            }
        }
    }

    Ok(serde_json::json!({
        "action": "get_batch",
        "requested": ids.len(),
        "found": found_count,
        "results": results,
    }))
}

fn absent_message(storage: &Arc<Storage>, id: &str) -> &'static str {
    if storage
        .superseded_node_ids()
        .ok()
        .is_some_and(|ids| ids.contains(id))
    {
        "retired, can't be retrieved"
    } else {
        "Memory not found"
    }
}

/// Retire a memory. On Strata this is a confirmed `purge` RETIRE.
async fn execute_purge(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    id: &str,
    reason: Option<String>,
    confirm: bool,
    action: &str,
) -> Result<Value, String> {
    if !confirm {
        return Err("Pass confirm=true to retire this memory. It can't be retrieved.".to_string());
    }

    let report = storage
        .purge_node(id, reason.as_deref())
        .map_err(|e| e.to_string())?;

    // Drop in-process projections of the retired id. Hydration reads only
    // what the store still returns.
    let runtime_rebuilt = if report.deleted {
        let mut rebuilt = CognitiveEngine::new();
        rebuilt.hydrate(storage);
        *cognitive.lock().await = rebuilt;
        true
    } else {
        false
    };

    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return Ok(match report.receipt_id {
            Some(receipt_id) => serde_json::json!({
                "action": action,
                "success": true,
                "nodeId": id,
                "receiptId": receipt_id,
                "rule": "purge",
                "message": "Retired; can't be retrieved.",
            }),
            None => serde_json::json!({
                "action": action,
                "success": false,
                "nodeId": id,
                "message": absent_message(storage, id),
            }),
        });
    }

    Ok(serde_json::json!({
        "action": action,
        "success": report.deleted,
        "nodeId": id,
        "message": if report.deleted {
            "Memory purged; content and embeddings removed. Legacy audit/sync records retain only opaque markers and limited metadata, so this is not a verified-local unlearning result."
        } else {
            "Memory not found"
        },
        "deletedAt": report.deleted_at.to_rfc3339(),
        "edgesPruned": report.edges_pruned,
        "insightsRewritten": report.insights_rewritten,
        "insightsDeleted": report.insights_deleted,
        "childrenOrphaned": report.children_orphaned,
        "unlearning": {
            "scope": report.unlearning_scope,
            "verdict": report.unlearning_verdict,
            "claimBoundary": report.unlearning_claim_boundary,
            "exclusions": report.unlearning_scope.exclusions(),
            "runtimeRebuilt": runtime_rebuilt,
            "runtimeHydrationAttempted": runtime_rebuilt,
            // `CognitiveEngine::hydrate` currently logs individual load
            // failures instead of returning a Result, so the public response
            // must not convert an attempted hydration into a verified claim.
            "runtimeHydrationVerified": false,
        },
    }))
}

/// Get accessibility state of a memory (Active/Dormant/Silent/Unavailable)
async fn execute_state(storage: &Arc<Storage>, id: &str) -> Result<Value, String> {
    // Get the memory
    let memory = storage
        .get_node(id)
        .map_err(|e| format!("Error: {}", e))?
        .ok_or_else(|| absent_message(storage, id))?;

    // Calculate accessibility score
    let accessibility = compute_accessibility(
        memory.retention_strength,
        memory.retrieval_strength,
        memory.storage_strength,
    );

    // Determine state
    let state = state_from_accessibility(accessibility);

    let state_description = match state {
        MemoryState::Active => "Easily retrievable - this memory is fresh and accessible",
        MemoryState::Dormant => "Retrievable with effort - may need cues to recall",
        MemoryState::Silent => "Difficult to retrieve - exists but hard to access",
        MemoryState::Unavailable => "Cannot be retrieved - needs significant reinforcement",
    };

    Ok(serde_json::json!({
        "action": "state",
        "memoryId": id,
        "content": memory.content,
        "state": format!("{:?}", state),
        "accessibility": accessibility,
        "description": state_description,
        "components": {
            "retentionStrength": memory.retention_strength,
            "retrievalStrength": memory.retrieval_strength,
            "storageStrength": memory.storage_strength
        },
        "thresholds": {
            "active": ACCESSIBILITY_ACTIVE,
            "dormant": ACCESSIBILITY_DORMANT,
            "silent": ACCESSIBILITY_SILENT
        }
    }))
}

fn change_delta(before: f64, after: f64, legacy: &str, strata: bool) -> String {
    if strata {
        format!("{:+.4}", after - before)
    } else {
        legacy.to_string()
    }
}

fn change_multiplier(before: f64, after: f64, legacy: &str, strata: bool) -> String {
    if !strata {
        return legacy.to_string();
    }
    if before.abs() < f64::EPSILON {
        return "fsrs-6".to_string();
    }
    format!("{:.4}x", after / before)
}

fn strata_receipt_id(storage: &Arc<Storage>, id: &str) -> Result<String, String> {
    storage
        .get_receipt(id)
        .map_err(|err| err.to_string())?
        .map(|receipt| receipt.receipt_id)
        .ok_or_else(|| format!("write receipt missing for {id}"))
}

/// Promote a memory (thumbs up) — increases retrieval strength with cognitive feedback pipeline
async fn execute_promote(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    id: &str,
    reason: Option<String>,
    claimed_role: Option<&str>,
) -> Result<Value, String> {
    let before = storage
        .get_node(id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", id))?;

    // #252 Phase A: with a bound process actor, the promote is an
    // actor-attributed endorsement — mutation, evidence, and receipt commit
    // in one transaction, and same-actor retries never accumulate votes.
    // Without a bound actor the historical path runs unchanged.
    let endorsement = if storage.process_actor_did().is_some() {
        Some(
            storage
                .promote_memory_as_actor(id, claimed_role, "memory")
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let node = match &endorsement {
        Some(outcome) => outcome.node.clone(),
        None => storage.promote_memory(id).map_err(|e| e.to_string())?,
    };

    // Cognitive feedback pipeline
    if let Ok(mut cog) = cognitive.try_lock() {
        cog.reward_signal.record_outcome(id, OutcomeType::Helpful);
        cog.importance_tracker.on_retrieved(id, true);
        if cog.reconsolidation.is_labile(id) {
            cog.reconsolidation.apply_modification(
                id,
                Modification::StrengthenConnection {
                    target_memory_id: id.to_string(),
                    boost: 0.2,
                },
            );
        }
    }

    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let mut result = serde_json::json!({
        "success": true,
        "action": "promoted",
        "nodeId": node.id,
        "reason": reason,
        "changes": {
            "retrievalStrength": {
                "before": before.retrieval_strength,
                "after": node.retrieval_strength,
                "delta": change_delta(before.retrieval_strength, node.retrieval_strength, "+0.20", strata)
            },
            "retentionStrength": {
                "before": before.retention_strength,
                "after": node.retention_strength,
                "delta": change_delta(before.retention_strength, node.retention_strength, "+0.10", strata)
            },
            "stability": {
                "before": before.stability,
                "after": node.stability,
                "multiplier": change_multiplier(before.stability, node.stability, "1.5x", strata)
            }
        },
        "message": if strata {
            format!(
                "Memory promoted under FSRS-6 (Easy). Stability {:.4} -> {:.4}. Reps {} -> {}. Lapses {} -> {}.",
                before.stability, node.stability, before.reps, node.reps, before.lapses, node.lapses
            )
        } else {
            format!(
                "Memory promoted. It will now surface more often in searches. Retrieval: {:.2} -> {:.2}",
                before.retrieval_strength, node.retrieval_strength
            )
        },
    });
    if strata {
        let receipt_id = match &endorsement {
            Some(outcome) => outcome.receipt.receipt_id.clone(),
            None => strata_receipt_id(storage, id)?,
        };
        result["receiptId"] = serde_json::json!(receipt_id);
        result["changes"]["reps"] = serde_json::json!({"before": before.reps, "after": node.reps});
        result["changes"]["lapses"] =
            serde_json::json!({"before": before.lapses, "after": node.lapses});
        result["changes"]["difficulty"] = serde_json::json!({
            "before": before.difficulty,
            "after": node.difficulty
        });
    }
    if let Some(outcome) = &endorsement {
        result["actor"] = crate::actor_surface::actor_block(&outcome.endorsement);
        result["endorsement"] = crate::actor_surface::endorsement_block(outcome);
    }
    Ok(result)
}

/// Demote a memory (thumbs down) — decreases retrieval strength with cognitive feedback pipeline
async fn execute_demote(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    id: &str,
    reason: Option<String>,
    claimed_role: Option<&str>,
) -> Result<Value, String> {
    let before = storage
        .get_node(id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Node not found: {}", id))?;

    // #252 Phase A: negative feedback carries actor identity too (same
    // transactional and idempotence contract as promote).
    let endorsement = if storage.process_actor_did().is_some() {
        Some(
            storage
                .demote_memory_as_actor(id, claimed_role, "memory")
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let node = match &endorsement {
        Some(outcome) => outcome.node.clone(),
        None => storage.demote_memory(id).map_err(|e| e.to_string())?,
    };

    // Cognitive feedback pipeline
    if let Ok(mut cog) = cognitive.try_lock() {
        cog.reward_signal
            .record_outcome(id, OutcomeType::NotHelpful);
        cog.importance_tracker.on_retrieved(id, false);
        if cog.reconsolidation.is_labile(id) {
            cog.reconsolidation.apply_modification(
                id,
                Modification::AddContext {
                    context: "User reported this memory was wrong/unhelpful".to_string(),
                },
            );
        }
    }

    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let mut result = serde_json::json!({
        "success": true,
        "action": "demoted",
        "nodeId": node.id,
        "reason": reason,
        "changes": {
            "retrievalStrength": {
                "before": before.retrieval_strength,
                "after": node.retrieval_strength,
                "delta": change_delta(before.retrieval_strength, node.retrieval_strength, "-0.30", strata)
            },
            "retentionStrength": {
                "before": before.retention_strength,
                "after": node.retention_strength,
                "delta": change_delta(before.retention_strength, node.retention_strength, "-0.15", strata)
            },
            "stability": {
                "before": before.stability,
                "after": node.stability,
                "multiplier": change_multiplier(before.stability, node.stability, "0.5x", strata)
            }
        },
        "message": if strata {
            format!(
                "Memory demoted under FSRS-6 (Again). Stability {:.4} -> {:.4}. Reps {} -> {}. Lapses {} -> {}. Not deleted.",
                before.stability, node.stability, before.reps, node.reps, before.lapses, node.lapses
            )
        } else {
            format!(
                "Memory demoted. Better alternatives will now surface instead. Retrieval: {:.2} -> {:.2}",
                before.retrieval_strength, node.retrieval_strength
            )
        },
        // A Strata log has no ranking for a demote to lower, and
        // retrievability reads 1.0 right after any review, so "after: 1.0"
        // beside a demote needs saying out loud.
        "note": if strata {
            "The memory is not deleted and is still found by its id and tags. Retrievability reads 1.0 right after any review; the lower stability makes it fade faster from here."
        } else {
            "Memory is NOT deleted - it remains searchable but ranks lower."
        }
    });
    if strata {
        let receipt_id = match &endorsement {
            Some(outcome) => outcome.receipt.receipt_id.clone(),
            None => strata_receipt_id(storage, id)?,
        };
        result["receiptId"] = serde_json::json!(receipt_id);
        result["changes"]["reps"] = serde_json::json!({"before": before.reps, "after": node.reps});
        result["changes"]["lapses"] =
            serde_json::json!({"before": before.lapses, "after": node.lapses});
        result["changes"]["difficulty"] = serde_json::json!({
            "before": before.difficulty,
            "after": node.difficulty
        });
    }
    if let Some(outcome) = &endorsement {
        result["actor"] = crate::actor_surface::actor_block(&outcome.endorsement);
        result["endorsement"] = crate::actor_surface::endorsement_block(outcome);
    }
    Ok(result)
}

/// Edit content. Strata admits a new event; the SQLite path updates the row.
async fn execute_edit(
    storage: &Arc<Storage>,
    id: &str,
    content: Option<String>,
) -> Result<Value, String> {
    let new_content = content.ok_or("Missing 'content' field. Required for edit action.")?;

    if new_content.trim().is_empty() {
        return Err("Content cannot be empty".to_string());
    }

    // Get existing node to capture old content
    let old_node = storage
        .get_node(id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Memory not found: {}", id))?;

    storage
        .update_node_content(id, &new_content)
        .map_err(|e| e.to_string())?;

    // Truncate previews for response (char-safe to avoid UTF-8 panics)
    let old_preview = if old_node.content.chars().count() > 200 {
        let truncated: String = old_node.content.chars().take(197).collect();
        format!("{}...", truncated)
    } else {
        old_node.content.clone()
    };
    let new_preview = if new_content.chars().count() > 200 {
        let truncated: String = new_content.chars().take(197).collect();
        format!("{}...", truncated)
    } else {
        new_content.clone()
    };

    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let node_id = if strata {
        storage
            .supersession_pairs()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|(old, _)| old == id)
            .map(|(_, successor)| successor)
            .ok_or_else(|| format!("edit did not retire {id}"))?
    } else {
        id.to_string()
    };
    let embedding_status = if strata {
        "refused"
    } else if storage
        .get_node(&node_id)
        .map_err(|e| e.to_string())?
        .is_some_and(|node| node.has_embedding == Some(true))
    {
        "available"
    } else {
        "pending"
    };
    let mut result = serde_json::json!({
        "success": true,
        "action": "edit",
        "nodeId": node_id,
        "oldContentPreview": old_preview,
        "newContentPreview": new_preview,
        "embeddingStatus": embedding_status,
        "note": if strata {
            "Successor admitted under rule edit. The previous node is retired and stays in the log. Its FSRS card is not copied."
        } else {
            "FSRS state preserved (stability, difficulty, reps, lapses unchanged). Embedding state was invalidated with the content update; regeneration may complete now or in maintenance. Inspect embeddingStatus."
        }
    });
    if strata {
        result["supersedes"] = serde_json::json!(id);
        result["rule"] = serde_json::json!("edit");
        result["receiptId"] = serde_json::json!(strata_receipt_id(storage, &node_id)?);
    }
    Ok(result)
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    #[test]
    fn test_accessibility_thresholds() {
        // Test Active state
        let accessibility = compute_accessibility(0.9, 0.8, 0.7);
        assert!(accessibility >= ACCESSIBILITY_ACTIVE);
        assert!(matches!(
            state_from_accessibility(accessibility),
            MemoryState::Active
        ));

        // Test Dormant state
        let accessibility = compute_accessibility(0.5, 0.5, 0.5);
        assert!((ACCESSIBILITY_DORMANT..ACCESSIBILITY_ACTIVE).contains(&accessibility));
        assert!(matches!(
            state_from_accessibility(accessibility),
            MemoryState::Dormant
        ));

        // Test Silent state
        let accessibility = compute_accessibility(0.2, 0.2, 0.2);
        assert!((ACCESSIBILITY_SILENT..ACCESSIBILITY_DORMANT).contains(&accessibility));
        assert!(matches!(
            state_from_accessibility(accessibility),
            MemoryState::Silent
        ));

        // Test Unavailable state
        let accessibility = compute_accessibility(0.05, 0.05, 0.05);
        assert!(accessibility < ACCESSIBILITY_SILENT);
        assert!(matches!(
            state_from_accessibility(accessibility),
            MemoryState::Unavailable
        ));
    }

    #[test]
    fn test_schema_structure() {
        let schema = schema();
        assert!(schema["properties"]["action"].is_object());
        assert!(schema["properties"]["id"].is_object());
        assert!(schema["properties"]["reason"].is_object());
        assert_eq!(schema["required"], serde_json::json!(["action"]));
        assert!(schema["properties"]["ids"].is_object()); // get_batch support
        // Verify all 8 actions are in enum
        let actions = schema["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 8);
        assert!(actions.contains(&serde_json::json!("get_batch")));
        assert!(actions.contains(&serde_json::json!("purge")));
        assert!(actions.contains(&serde_json::json!("edit")));
        assert!(actions.contains(&serde_json::json!("promote")));
        assert!(actions.contains(&serde_json::json!("demote")));
        assert!(schema["properties"]["confirm"].is_object());
    }

    // === INTEGRATION TESTS ===

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    async fn ingest_memory(storage: &Arc<Storage>) -> String {
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory unified test content".to_string(),
                node_type: "fact".to_string(),
                source: Some("test".to_string()),
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["test-tag".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        node.id
    }

    #[tokio::test]
    async fn test_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_invalid_action_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "invalid", "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid action"));
    }

    #[tokio::test]
    async fn test_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "get", "id": "not-a-uuid" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory ID format"));
    }

    #[tokio::test]
    async fn test_get_existing_memory() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "get", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "get");
        assert_eq!(value["found"], true);
        assert_eq!(value["node"]["id"], id);
        assert_eq!(value["node"]["content"], "Memory unified test content");
        assert_eq!(value["node"]["nodeType"], "fact");
        assert!(value["node"]["createdAt"].is_string());
        assert!(value["node"]["tags"].is_array());
    }

    #[tokio::test]
    async fn test_get_nonexistent_memory() {
        let (storage, _dir) = test_storage().await;
        let args =
            serde_json::json!({ "action": "get", "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["found"], false);
        assert_eq!(value["message"], "Memory not found");
    }

    #[tokio::test]
    async fn test_delete_requires_confirm() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "delete", "id": id.clone() });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("confirm=true"));
        assert!(storage.get_node(&id).unwrap().is_some());
    }

    #[tokio::test]
    async fn test_delete_existing_memory_with_confirm() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "delete", "id": id, "confirm": true });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "delete");
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_memory() {
        let (storage, _dir) = test_storage().await;
        // Ingest+delete a throwaway memory to warm writer after WAL migration
        let warmup_id = storage
            .ingest(vestige_core::IngestInput {
                content: "warmup".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap()
            .id;
        let _ = storage.delete_node(&warmup_id);
        let args = serde_json::json!({
            "action": "delete",
            "id": "00000000-0000-0000-0000-000000000000",
            "confirm": true
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], false);
        assert!(value["message"].as_str().unwrap().contains("not found"));
        assert_eq!(
            value["unlearning"]["verdict"],
            serde_json::json!("incomplete")
        );
        assert_eq!(
            value["unlearning"]["runtimeRebuilt"],
            serde_json::json!(false)
        );
    }

    #[tokio::test]
    async fn test_delete_then_get_returns_not_found() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let del_args = serde_json::json!({ "action": "delete", "id": id, "confirm": true });
        execute(&storage, &test_cognitive(), Some(del_args))
            .await
            .unwrap();
        let get_args = serde_json::json!({ "action": "get", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(get_args)).await;
        let value = result.unwrap();
        assert_eq!(value["found"], false);
    }

    #[tokio::test]
    async fn test_purge_requires_confirm() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "purge", "id": id.clone() });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("confirm=true"));
        assert!(storage.get_node(&id).unwrap().is_some());
    }

    #[tokio::test]
    async fn test_purge_existing_memory() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let cognitive = test_cognitive();
        {
            let mut engine = cognitive.lock().await;
            engine.activation_network.add_edge(
                id.clone(),
                "runtime-only-neighbor".to_string(),
                vestige_core::LinkType::Semantic,
                1.0,
            );
            engine.synaptic_tagging.tag_memory(&id);
            assert!(engine.synaptic_tagging.has_active_tag(&id));
            assert!(!engine.activation_network.activate(&id, 1.0).is_empty());
        }
        let args = serde_json::json!({
            "action": "purge",
            "id": id,
            "confirm": true,
            "reason": "test cleanup"
        });
        let result = execute(&storage, &cognitive, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "purge");
        assert_eq!(value["success"], true);
        assert!(
            value["message"]
                .as_str()
                .unwrap()
                .contains("opaque markers")
        );
        assert_eq!(value["edgesPruned"], 0);
        assert!(storage.get_node(&id).unwrap().is_none());
        assert_eq!(
            value["unlearning"]["scope"],
            serde_json::json!("legacy_audited_purge")
        );
        assert_eq!(
            value["unlearning"]["verdict"],
            serde_json::json!("incomplete")
        );
        assert_eq!(
            value["unlearning"]["runtimeRebuilt"],
            serde_json::json!(true)
        );
        assert_eq!(
            value["unlearning"]["runtimeHydrationAttempted"],
            serde_json::json!(true)
        );
        assert_eq!(
            value["unlearning"]["runtimeHydrationVerified"],
            serde_json::json!(false)
        );
        assert!(
            value["unlearning"]["claimBoundary"]
                .as_str()
                .unwrap()
                .contains("does not establish complete machine unlearning")
        );
        let mut rebuilt = cognitive.lock().await;
        assert!(!rebuilt.synaptic_tagging.has_active_tag(&id));
        assert!(rebuilt.activation_network.activate(&id, 1.0).is_empty());
    }

    #[tokio::test]
    async fn test_state_existing_memory() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "state", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["action"], "state");
        assert_eq!(value["memoryId"], id);
        assert!(value["accessibility"].is_number());
        assert!(value["state"].is_string());
        assert!(value["description"].is_string());
        assert!(value["components"]["retentionStrength"].is_number());
        assert!(value["components"]["retrievalStrength"].is_number());
        assert!(value["components"]["storageStrength"].is_number());
        assert_eq!(value["thresholds"]["active"], 0.7);
        assert_eq!(value["thresholds"]["dormant"], 0.4);
        assert_eq!(value["thresholds"]["silent"], 0.1);
    }

    #[tokio::test]
    async fn test_state_nonexistent_memory_fails() {
        let (storage, _dir) = test_storage().await;
        let args =
            serde_json::json!({ "action": "state", "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_accessibility_boundary_active() {
        let a = compute_accessibility(1.0, 0.7, 0.5);
        assert!(a >= ACCESSIBILITY_ACTIVE);
        assert!(matches!(state_from_accessibility(a), MemoryState::Active));
    }

    #[test]
    fn test_accessibility_boundary_zero() {
        let a = compute_accessibility(0.0, 0.0, 0.0);
        assert_eq!(a, 0.0);
        assert!(matches!(
            state_from_accessibility(a),
            MemoryState::Unavailable
        ));
    }

    // ========================================================================
    // PROMOTE/DEMOTE TESTS (ported from feedback.rs, v1.7.0 merge)
    // ========================================================================

    #[tokio::test]
    async fn test_promote_missing_id_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "promote", "id": "not-a-uuid" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory ID format"));
    }

    #[tokio::test]
    async fn test_promote_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "promote", "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Node not found"));
    }

    #[tokio::test]
    async fn test_promote_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "promote", "id": id, "reason": "It was helpful" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "promoted");
        assert_eq!(value["nodeId"], id);
        assert_eq!(value["reason"], "It was helpful");
        assert!(value["changes"]["retrievalStrength"].is_object());
    }

    #[tokio::test]
    async fn test_promote_without_reason_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "promote", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["reason"].is_null());
    }

    #[tokio::test]
    async fn test_promote_changes_contain_expected_fields() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "promote", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        assert!(value["changes"]["retrievalStrength"]["before"].is_number());
        assert!(value["changes"]["retrievalStrength"]["after"].is_number());
        assert_eq!(value["changes"]["retrievalStrength"]["delta"], "+0.20");
        assert!(value["changes"]["retentionStrength"]["before"].is_number());
        assert_eq!(value["changes"]["retentionStrength"]["delta"], "+0.10");
        assert_eq!(value["changes"]["stability"]["multiplier"], "1.5x");
    }

    #[tokio::test]
    async fn test_demote_invalid_uuid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "action": "demote", "id": "bad-id" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory ID format"));
    }

    #[tokio::test]
    async fn test_demote_nonexistent_node_fails() {
        let (storage, _dir) = test_storage().await;
        let args =
            serde_json::json!({ "action": "demote", "id": "00000000-0000-0000-0000-000000000000" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Node not found"));
    }

    #[tokio::test]
    async fn test_demote_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "demote", "id": id, "reason": "It was wrong" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "demoted");
        assert_eq!(value["nodeId"], id);
        assert_eq!(value["reason"], "It was wrong");
        assert!(value["note"].as_str().unwrap().contains("NOT deleted"));
    }

    #[tokio::test]
    async fn test_demote_changes_contain_expected_fields() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "demote", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        assert!(value["changes"]["retrievalStrength"]["before"].is_number());
        assert_eq!(value["changes"]["retrievalStrength"]["delta"], "-0.30");
        assert_eq!(value["changes"]["retentionStrength"]["delta"], "-0.15");
        assert_eq!(value["changes"]["stability"]["multiplier"], "0.5x");
    }

    // ========================================================================
    // EDIT TESTS (v1.9.2)
    // ========================================================================

    #[tokio::test]
    async fn test_edit_succeeds() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({
            "action": "edit",
            "id": id,
            "content": "Updated memory content"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["action"], "edit");
        assert_eq!(value["nodeId"], id);
        assert!(
            value["oldContentPreview"]
                .as_str()
                .unwrap()
                .contains("Memory unified test content")
        );
        assert!(
            value["newContentPreview"]
                .as_str()
                .unwrap()
                .contains("Updated memory content")
        );
        assert!(
            value["note"]
                .as_str()
                .unwrap()
                .contains("FSRS state preserved")
        );
    }

    #[tokio::test]
    async fn test_edit_preserves_fsrs_state() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;

        // Get FSRS state before edit
        let before = storage.get_node(&id).unwrap().unwrap();

        // Edit content
        let args = serde_json::json!({
            "action": "edit",
            "id": id,
            "content": "Completely new content after edit"
        });
        execute(&storage, &test_cognitive(), Some(args))
            .await
            .unwrap();

        // Verify FSRS state preserved
        let after = storage.get_node(&id).unwrap().unwrap();
        assert_eq!(after.stability, before.stability);
        assert_eq!(after.difficulty, before.difficulty);
        assert_eq!(after.reps, before.reps);
        assert_eq!(after.lapses, before.lapses);
        assert_eq!(after.retention_strength, before.retention_strength);
        // Content should be updated
        assert_eq!(after.content, "Completely new content after edit");
        assert_ne!(after.content, before.content);
    }

    #[tokio::test]
    async fn test_edit_missing_content_fails() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "edit", "id": id });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("content"));
    }

    #[tokio::test]
    async fn test_edit_empty_content_fails() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "edit", "id": id, "content": "  " });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_edit_nonexistent_memory_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "action": "edit",
            "id": "00000000-0000-0000-0000-000000000000",
            "content": "New content"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_with_multibyte_utf8_content() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        // Content with emoji and CJK characters (multi-byte UTF-8)
        let long_content = "🧠".repeat(100); // 100 brain emoji = 400 bytes but only 100 chars
        let args = serde_json::json!({
            "action": "edit",
            "id": id,
            "content": long_content
        });
        // This must NOT panic (previous code would panic on byte-level truncation)
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    // === #252 PHASE A: ACTOR PROVENANCE (gates 1 and 2 at the tool layer) ===

    #[tokio::test]
    async fn test_promote_with_claimed_role_stays_neutral_and_records_endorsement() {
        let (storage, _dir) = test_storage().await;
        // Ingest BEFORE binding so the node is unattributed and a later
        // support is a plain "support", not self-support.
        let id = ingest_memory(&storage).await;
        let actor = vestige_core::actor::ProcessActor::mint();
        storage.set_process_actor(actor.did()).unwrap();

        // Gate 1: claiming a privileged role without operator membership
        // leaves the actor neutral at 1.0 — the claim never self-grants.
        let args = serde_json::json!({ "action": "promote", "id": id, "role": "operator" });
        let value = execute(&storage, &test_cognitive(), Some(args))
            .await
            .unwrap();
        assert_eq!(value["actor"]["claimedRole"], "operator");
        assert_eq!(value["actor"]["effectiveRole"], "unattributed");
        assert_eq!(value["actor"]["resolvedWeight"], 1.0);
        assert_eq!(
            value["actor"]["resolutionDisposition"],
            "unregistered_claim"
        );
        assert_eq!(value["actor"]["policyVersion"], 1);
        assert_eq!(value["endorsement"]["kind"], "support");
        assert!(value["endorsement"]["eventId"].as_str().is_some());
        assert_eq!(value["endorsement"]["alreadyRecorded"], false);

        // Gate 2: the same actor retrying — even under a different hat —
        // does not create a second vote.
        let retry = serde_json::json!({ "action": "promote", "id": id, "role": "qa" });
        let value = execute(&storage, &test_cognitive(), Some(retry))
            .await
            .unwrap();
        assert_eq!(value["endorsement"]["alreadyRecorded"], true);
        let events = storage
            .list_endorsement_events(Some(&id), None, 50)
            .unwrap();
        assert_eq!(events.len(), 1, "same-actor hats/retries are one stance");

        // The endorsement binds the exact content revision.
        assert_eq!(
            events[0].revision_digest,
            vestige_core::actor::revision_digest("Memory unified test content")
        );
    }

    #[tokio::test]
    async fn test_demote_records_opposition_with_actor_identity() {
        let (storage, _dir) = test_storage().await;
        storage
            .set_process_actor(vestige_core::actor::ProcessActor::mint().did())
            .unwrap();
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "demote", "id": id });
        let value = execute(&storage, &test_cognitive(), Some(args))
            .await
            .unwrap();
        assert_eq!(value["endorsement"]["kind"], "oppose");
        assert!(
            value["actor"]["id"]
                .as_str()
                .unwrap()
                .starts_with("did:key:z6Mk")
        );
    }

    #[tokio::test]
    async fn test_promote_without_bound_actor_keeps_legacy_shape() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let args = serde_json::json!({ "action": "promote", "id": id });
        let value = execute(&storage, &test_cognitive(), Some(args))
            .await
            .unwrap();
        assert_eq!(value["success"], true);
        assert!(
            value.get("actor").is_none(),
            "no bound actor, no provenance claim"
        );
        assert!(value.get("endorsement").is_none());
        assert!(
            storage
                .list_endorsement_events(Some(&id), None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_operator_granted_role_resolves_on_promote() {
        let (storage, _dir) = test_storage().await;
        let id = ingest_memory(&storage).await;
        let did = vestige_core::actor::ProcessActor::mint().did().to_string();
        storage.set_process_actor(&did).unwrap();
        // Operator grants qa through the store (never through a tool call).
        storage
            .grant_actor_role(&did, "qa", Some("operator reviewed"))
            .unwrap();
        let args = serde_json::json!({ "action": "promote", "id": id, "role": "qa" });
        let value = execute(&storage, &test_cognitive(), Some(args))
            .await
            .unwrap();
        assert_eq!(value["actor"]["effectiveRole"], "qa");
        assert_eq!(value["actor"]["resolutionDisposition"], "granted");
        assert!((value["actor"]["resolvedWeight"].as_f64().unwrap() - 1.10).abs() < 1e-9);
        assert_eq!(value["endorsement"]["independentPrior"], 1.10);
    }
}

#[cfg(test)]
mod strata_tests {
    use super::*;
    use crate::strata_memory::StrataMemory;
    use crate::tools::receipt;
    use std::path::Path;
    use vestige_core::MemoryStore;

    fn open() -> (Arc<StrataMemory>, Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let mem = Arc::new(StrataMemory::open(dir.path()).unwrap());
        // Registered like the server's own open, so receipt checks can
        // re-verify the log.
        crate::strata_memory::register_open(&mem);
        let storage: Arc<Storage> = mem.clone();
        (mem, storage, dir)
    }

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    fn ingest(storage: &Arc<Storage>, content: &str) -> String {
        storage
            .ingest(vestige_core::IngestInput {
                content: content.to_string(),
                ..vestige_core::IngestInput::default()
            })
            .unwrap()
            .id
    }

    fn log_contains(dir: &Path, needle: &str) -> bool {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if std::fs::read(&path)
                    .unwrap_or_default()
                    .windows(needle.len())
                    .any(|window| window == needle.as_bytes())
                {
                    return true;
                }
            }
        }
        false
    }

    async fn proved(storage: &Arc<Storage>, receipt_id: &str, node_id: &str, kind: &str) {
        let got = receipt::execute(
            storage,
            Some(serde_json::json!({"action": "get", "receipt_id": receipt_id})),
        )
        .await
        .unwrap_or_else(|err| panic!("receipt get failed: {err}"));
        assert_eq!(got["receipt"]["receipt_id"], receipt_id);
        assert_eq!(got["receipt"]["retrieved"][0], node_id);
        assert_eq!(got["receipt"]["mutations"][0]["kind"], kind);
        assert_eq!(got["receipt"]["mutations"][0]["id"], node_id);
        assert_eq!(got["attestation"]["verification"]["locallyVerified"], true);
        assert_eq!(got["attestation"]["verification"]["chainValid"], true);
        assert_eq!(got["attestation"]["verification"]["gateAllowed"], true);
        let digest = got["attestation"]["verification"]["payloadDigest"]
            .as_str()
            .unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert!(
            got["receipt"]["mutations"][0]["note"]
                .as_str()
                .unwrap()
                .contains(&format!("digest={digest}"))
        );
    }

    #[tokio::test]
    async fn promote_demote_and_edit_admit_proved_receipts() {
        let dir = tempfile::TempDir::new().unwrap();
        let (receipts, cards) = {
            let mem = Arc::new(StrataMemory::open(dir.path()).unwrap());
            crate::strata_memory::register_open(&mem);
            mem.set_process_actor("did:key:z6Mkstratafixture").unwrap();
            let storage: Arc<Storage> = mem.clone();
            let id = ingest(&storage, "original cause text");
            let before = mem.card_q(&id).unwrap();

            let promoted = execute(
                &storage,
                &cognitive(),
                Some(serde_json::json!({
                    "action": "promote",
                    "id": id,
                    "reason": "it held",
                    "role": "operator"
                })),
            )
            .await
            .unwrap();
            assert_eq!(promoted["action"], "promoted");
            assert_eq!(promoted["success"], true);
            let promote_receipt = promoted["receiptId"].as_str().unwrap().to_string();
            proved(&storage, &promote_receipt, &id, "promoted").await;
            assert_eq!(
                promoted["actor"]["resolutionDisposition"],
                "unregistered_claim"
            );
            assert_eq!(promoted["actor"]["resolvedWeight"], 1.0);
            let after_promote = mem.card_q(&id).unwrap();
            assert_eq!(after_promote.3, before.3 + 1, "promote folds one review");
            assert_eq!(after_promote.4, before.4, "Easy is not a lapse");
            assert_ne!(after_promote.0, before.0, "Easy changes stability");

            let demoted = execute(
                &storage,
                &cognitive(),
                Some(serde_json::json!({
                    "action": "demote",
                    "id": id,
                    "reason": "it failed"
                })),
            )
            .await
            .unwrap();
            assert_eq!(demoted["action"], "demoted");
            assert!(demoted["note"].as_str().unwrap().contains("not deleted"));
            let demote_receipt = demoted["receiptId"].as_str().unwrap().to_string();
            assert_ne!(demote_receipt, promote_receipt);
            proved(&storage, &demote_receipt, &id, "demoted").await;
            let after_demote = mem.card_q(&id).unwrap();
            assert_eq!(after_demote.3, after_promote.3 + 1);
            assert_eq!(after_demote.4, after_promote.4 + 1, "Again records a lapse");
            assert!(storage.get_node(&id).unwrap().is_some());

            let edited = execute(
                &storage,
                &cognitive(),
                Some(serde_json::json!({
                    "action": "edit",
                    "id": id,
                    "content": "replacement cause text"
                })),
            )
            .await
            .unwrap();
            assert_eq!(edited["action"], "edit");
            assert_eq!(edited["embeddingStatus"], "refused");
            assert_eq!(edited["rule"], "edit");
            assert_eq!(edited["supersedes"], id);
            let successor = edited["nodeId"].as_str().unwrap().to_string();
            assert_ne!(successor, id);
            let edit_receipt = edited["receiptId"].as_str().unwrap().to_string();
            proved(&storage, &edit_receipt, &successor, "edited").await;
            let note = receipt::execute(
                &storage,
                Some(serde_json::json!({"action": "get", "receipt_id": edit_receipt})),
            )
            .await
            .unwrap();
            assert!(
                note["receipt"]["mutations"][0]["note"]
                    .as_str()
                    .unwrap()
                    .contains("rule=edit"),
                "receipt names edit"
            );
            assert_eq!(mem.card_q(&id).unwrap(), after_demote, "old card stays");
            let successor_card = mem.card_q(&successor).unwrap();
            assert_eq!(successor_card.3, 1, "successor is a fresh ingest fold");
            assert_eq!(successor_card.4, 0, "successor has no lapses");
            assert!(storage.get_node(&id).unwrap().is_none());
            assert_eq!(
                storage.supersession_pairs().unwrap(),
                vec![(id.clone(), successor.clone())]
            );
            assert_eq!(
                storage.get_node(&successor).unwrap().unwrap().content,
                "replacement cause text"
            );
            let live_recall = crate::tools::recall::execute(
                &storage,
                &cognitive(),
                &vestige_core::OutputConfig::default(),
                Some(serde_json::json!({"handle": successor})),
            )
            .await
            .unwrap();
            assert_eq!(live_recall["nodes"][0]["id"], successor);
            assert_eq!(live_recall["nodes"][0]["content"], "replacement cause text");
            let query = crate::tools::recall::execute(
                &storage,
                &cognitive(),
                &vestige_core::OutputConfig::default(),
                Some(serde_json::json!({"query": "replacement cause text"})),
            )
            .await
            .unwrap_err();
            assert!(query.contains("similarity_disabled"), "{query}");
            assert!(log_contains(dir.path(), "original cause text"));
            assert!(log_contains(dir.path(), "replacement cause text"));
            (
                (promote_receipt, demote_receipt, edit_receipt, id, successor),
                after_demote,
            )
        };

        let reopened = Arc::new(StrataMemory::open(dir.path()).unwrap());
        crate::strata_memory::register_open(&reopened);
        let storage: Arc<Storage> = reopened.clone();
        let (promote_receipt, demote_receipt, edit_receipt, id, successor) = receipts;
        proved(&storage, &promote_receipt, &id, "promoted").await;
        proved(&storage, &demote_receipt, &id, "demoted").await;
        proved(&storage, &edit_receipt, &successor, "edited").await;
        assert!(reopened.get_node(&id).unwrap().is_none());
        let card = reopened.card_q(&id).unwrap();
        assert_eq!(card.3, cards.3);
        assert_eq!(card.4, cards.4);
        assert_eq!(
            storage.supersession_pairs().unwrap(),
            vec![(id.clone(), successor.clone())]
        );
        let successor_node = storage.get_node(&successor).unwrap().unwrap();
        assert_eq!(successor_node.content, "replacement cause text");
        assert_eq!(successor_node.reps, 1);
        assert_eq!(successor_node.lapses, 0);
    }

    #[tokio::test]
    async fn invalid_and_unknown_handles_are_explicit_errors() {
        let (_mem, storage, _dir) = open();
        let cog = cognitive();
        for action in ["promote", "demote", "edit"] {
            let mut args = serde_json::json!({"action": action, "id": "not-a-handle"});
            if action == "edit" {
                args["content"] = serde_json::json!("text");
            }
            let err = execute(&storage, &cog, Some(args)).await.unwrap_err();
            assert!(err.contains("Invalid memory ID format"), "{action}: {err}");
        }

        let unknown = "mem-ffffffffffffffff";
        for (action, extra) in [
            ("promote", serde_json::json!({})),
            ("demote", serde_json::json!({})),
            ("edit", serde_json::json!({"content": "next"})),
        ] {
            let mut args = serde_json::json!({"action": action, "id": unknown});
            if let Some(obj) = extra.as_object() {
                for (key, value) in obj {
                    args[key] = value.clone();
                }
            }
            let err = execute(&storage, &cog, Some(args)).await.unwrap_err();
            assert!(
                err.to_ascii_lowercase().contains("not found"),
                "{action}: {err}"
            );
        }

        let id = ingest(&storage, "stays");
        let missing = execute(
            &storage,
            &cog,
            Some(serde_json::json!({"action": "edit", "id": id})),
        )
        .await
        .unwrap_err();
        assert!(missing.contains("content"), "{missing}");
        let empty = execute(
            &storage,
            &cog,
            Some(serde_json::json!({"action": "edit", "id": id, "content": "   "})),
        )
        .await
        .unwrap_err();
        assert!(empty.contains("empty"), "{empty}");
        let secret = format!("ghp_{}", "A".repeat(36));
        let refused = execute(
            &storage,
            &cog,
            Some(serde_json::json!({"action": "edit", "id": id, "content": secret})),
        )
        .await
        .unwrap_err();
        assert!(
            refused.contains("Refused to store probable credential"),
            "{refused}"
        );
        assert_eq!(storage.get_node(&id).unwrap().unwrap().content, "stays");
    }

    /// Predecessor of an edit must not come back from handle recall.
    #[tokio::test]
    async fn recall_withholds_the_node_retired_by_edit() {
        let (_mem, storage, _dir) = open();
        let id = ingest(&storage, "original cause text");
        let edited = execute(
            &storage,
            &cognitive(),
            Some(serde_json::json!({
                "action": "edit",
                "id": id,
                "content": "replacement cause text"
            })),
        )
        .await
        .unwrap();
        let successor = edited["nodeId"].as_str().unwrap();
        assert_ne!(successor, id);
        let old_recall = crate::tools::recall::execute(
            &storage,
            &cognitive(),
            &vestige_core::OutputConfig::default(),
            Some(serde_json::json!({"handle": id})),
        )
        .await
        .unwrap();
        assert!(
            !old_recall.to_string().contains("original cause text"),
            "retired node is not returned by recall: {old_recall}"
        );
    }
}
