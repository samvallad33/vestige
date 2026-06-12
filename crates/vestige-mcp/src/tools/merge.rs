//! Merge / Supersede control tools (Phase 3 — v2.1.25)
//!
//! Diff-previewed, confidence-gated, reversible, self-explaining
//! combine/dedupe/supersede on a never-delete (bitemporal) store. The default
//! is always preview/review — these tools never silently mutate memory.
//!
//! Tool surface (each registered as its own MCP tool name, all routed here):
//!
//! - `merge_candidates` — surface likely duplicate clusters with confidence +
//!   the signals behind each (Fellegi-Sunter match / possible / non-match).
//! - `plan_merge` — previewable merge PLAN (a diff) without applying it.
//! - `plan_supersede` — preview superseding A with B (bitemporal invalidation,
//!   audit-preserving) without applying.
//! - `apply_plan` — execute a previously-generated plan id; recorded as a
//!   reversible operation.
//! - `merge_undo` — reverse a prior merge/supersede operation (the reflog).
//! - `protect` — pin a memory so it can never be auto-merged/superseded/forgotten.
//! - `merge_policy` — get/set the two confidence thresholds + auto_apply.
//!
//! The actual logic lives in `vestige_core` (`storage::Storage` +
//! `advanced::merge_supersede`); this layer only validates arguments and shapes
//! JSON.

use serde_json::{Value, json};
use std::sync::Arc;
use vestige_core::Storage;

// ============================================================================
// SCHEMAS
// ============================================================================

/// `merge_candidates` input schema.
pub fn merge_candidates_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max candidate clusters to return (default 20).",
                "default": 20, "minimum": 1, "maximum": 100
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional: only consider memories with these tags (ANY match)."
            }
        }
    })
}

/// `plan_merge` input schema.
pub fn plan_merge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "member_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "IDs of the memories to merge (>= 2). The survivor is kept; the rest are bitemporally invalidated (kept for audit)."
            },
            "survivor_id": {
                "type": "string",
                "description": "Optional: which member to keep. Defaults to the highest-retention member."
            }
        },
        "required": ["member_ids"]
    })
}

/// `plan_supersede` input schema.
pub fn plan_supersede_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "old_id": { "type": "string", "description": "Memory being superseded (kept, marked invalid)." },
            "new_id": { "type": "string", "description": "Memory that supersedes the old one." }
        },
        "required": ["old_id", "new_id"]
    })
}

/// `apply_plan` input schema.
pub fn apply_plan_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "plan_id": { "type": "string", "description": "ID of a plan produced by plan_merge / plan_supersede." },
            "confirm": {
                "type": "boolean",
                "description": "Required true for 'possible'/'non_match' plans. 'match' plans apply only if the policy has auto_apply=true, else confirm is required too.",
                "default": false
            }
        },
        "required": ["plan_id"]
    })
}

/// `merge_undo` input schema.
pub fn merge_undo_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation_id": {
                "type": "string",
                "description": "ID of the merge/supersede operation to reverse. Omit to list recent operations (the reflog)."
            }
        }
    })
}

/// `protect` input schema.
pub fn protect_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "id": { "type": "string", "description": "Memory id to protect/unprotect." },
            "protected": {
                "type": "boolean",
                "description": "true to pin (block auto-merge/supersede/forget), false to unpin. Default true.",
                "default": true
            }
        },
        "required": ["id"]
    })
}

/// `merge_policy` input schema.
pub fn merge_policy_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "match_threshold": {
                "type": "number",
                "description": "Score >= this => 'match' (auto-merge eligible). 0-1.",
                "minimum": 0.0, "maximum": 1.0
            },
            "possible_threshold": {
                "type": "number",
                "description": "Score in [possible, match) => 'possible' (review). Below => not offered. 0-1.",
                "minimum": 0.0, "maximum": 1.0
            },
            "auto_apply": {
                "type": "boolean",
                "description": "Allow 'match'-class plans to apply without confirm. Default false (review-first)."
            }
        }
    })
}

// ============================================================================
// DISPATCH
// ============================================================================

/// Route a merge/supersede tool call by tool name.
pub async fn execute(storage: &Arc<Storage>, tool: &str, args: Option<Value>) -> Result<Value, String> {
    match tool {
        "merge_candidates" => merge_candidates(storage, args),
        "plan_merge" => plan_merge(storage, args),
        "plan_supersede" => plan_supersede(storage, args),
        "apply_plan" => apply_plan(storage, args),
        "merge_undo" => merge_undo(storage, args),
        "protect" => protect(storage, args),
        "merge_policy" => merge_policy(storage, args),
        other => Err(format!("unknown merge tool: {other}")),
    }
}

fn obj(args: &Option<Value>) -> serde_json::Map<String, Value> {
    args.as_ref()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

// ============================================================================
// merge_candidates
// ============================================================================

fn merge_candidates(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let a = obj(&args);
        let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let tags: Vec<String> = a
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let policy = storage.get_merge_policy().map_err(|e| e.to_string())?;
        let candidates = storage
            .merge_candidates(policy, limit, &tags)
            .map_err(|e| e.to_string())?;

        let out: Vec<Value> = candidates
            .iter()
            .map(|c| {
                json!({
                    "memberIds": c.member_ids,
                    "previews": c.previews,
                    "survivorId": c.survivor_id,
                    "confidence": format!("{:.3}", c.confidence),
                    "classification": c.classification.as_str(),
                    "hasProtectedMember": c.has_protected_member,
                    "signals": {
                        "embeddingSimilarity": format!("{:.3}", c.signals.embedding_similarity),
                        "tagOverlap": format!("{:.3}", c.signals.tag_overlap),
                        "tokenOverlap": format!("{:.3}", c.signals.token_overlap),
                        "combinedScore": format!("{:.3}", c.signals.combined_score)
                    },
                    "nextStep": if c.has_protected_member {
                        "A member is protected — unprotect it or pick it as survivor before plan_merge."
                    } else {
                        "Call plan_merge with these memberIds to preview the combined result."
                    }
                })
            })
            .collect();

        let policy = storage.get_merge_policy().map_err(|e| e.to_string())?;
        Ok(json!({
            "candidates": out,
            "totalCandidates": out.len(),
            "policy": {
                "matchThreshold": policy.match_threshold,
                "possibleThreshold": policy.possible_threshold,
                "autoApply": policy.auto_apply
            },
            "note": "Nothing was changed. These are review candidates only."
        }))
    }
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = (storage, args);
        Ok(json!({ "error": "Embeddings feature not enabled.", "candidates": [] }))
    }
}

// ============================================================================
// plan_merge
// ============================================================================

fn plan_merge(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let a = obj(&args);
        let member_ids: Vec<String> = a
            .get("member_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if member_ids.len() < 2 {
            return Err("member_ids must contain at least 2 ids".into());
        }
        let survivor = a.get("survivor_id").and_then(|v| v.as_str());
        let policy = storage.get_merge_policy().map_err(|e| e.to_string())?;
        let plan = storage
            .plan_merge(&member_ids, survivor, policy)
            .map_err(|e| e.to_string())?;
        Ok(plan_to_json(&plan, &policy))
    }
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = (storage, args);
        Err("Embeddings feature not enabled.".into())
    }
}

// ============================================================================
// plan_supersede
// ============================================================================

fn plan_supersede(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let a = obj(&args);
        let old_id = a
            .get("old_id")
            .and_then(|v| v.as_str())
            .ok_or("old_id is required")?;
        let new_id = a
            .get("new_id")
            .and_then(|v| v.as_str())
            .ok_or("new_id is required")?;
        let policy = storage.get_merge_policy().map_err(|e| e.to_string())?;
        let plan = storage
            .plan_supersede(old_id, new_id, policy)
            .map_err(|e| e.to_string())?;
        Ok(plan_to_json(&plan, &policy))
    }
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = (storage, args);
        Err("Embeddings feature not enabled.".into())
    }
}

#[cfg(all(feature = "embeddings", feature = "vector-search"))]
fn plan_to_json(plan: &vestige_core::MergePlan, policy: &vestige_core::MergePolicy) -> Value {
    let requires_confirm = plan.classification != vestige_core::MatchClass::Match || !policy.auto_apply;
    json!({
        "planId": plan.id,
        "kind": plan.kind.as_str(),
        "survivorId": plan.survivor_id,
        "memberIds": plan.member_ids,
        "diff": {
            "resultContent": plan.result_content,
            "resultTags": plan.result_tags,
            "resultSource": plan.result_source,
            "invalidatedIds": plan.invalidated_ids
        },
        "confidence": format!("{:.3}", plan.confidence),
        "classification": plan.classification.as_str(),
        "signals": {
            "embeddingSimilarity": format!("{:.3}", plan.signals.embedding_similarity),
            "tagOverlap": format!("{:.3}", plan.signals.tag_overlap),
            "tokenOverlap": format!("{:.3}", plan.signals.token_overlap),
            "combinedScore": format!("{:.3}", plan.signals.combined_score)
        },
        "explanation": plan.explanation,
        "requiresConfirm": requires_confirm,
        "nextStep": format!(
            "Review the diff. To execute: apply_plan with plan_id='{}'{}.",
            plan.id,
            if requires_confirm { " and confirm=true" } else { "" }
        ),
        "note": "Nothing was changed. This is a preview plan — apply_plan applies it; merge_undo reverses it."
    })
}

// ============================================================================
// apply_plan
// ============================================================================

fn apply_plan(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let a = obj(&args);
        let plan_id = a
            .get("plan_id")
            .and_then(|v| v.as_str())
            .ok_or("plan_id is required")?;
        let confirm = a.get("confirm").and_then(|v| v.as_bool()).unwrap_or(false);
        let op = storage
            .apply_plan(plan_id, confirm)
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "operationId": op.id,
            "opType": op.op_type,
            "status": op.status,
            "survivorId": op.survivor_id,
            "affectedIds": op.affected_ids,
            "reason": op.reason,
            "appliedAt": op.created_at,
            "reversible": true,
            "nextStep": format!("To reverse this, call merge_undo with operation_id='{}'.", op.id),
            "note": "Old memories were bitemporally invalidated (valid_until stamped), NOT deleted. They remain queryable for audit."
        }))
    }
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = (storage, args);
        Err("Embeddings feature not enabled.".into())
    }
}

// ============================================================================
// merge_undo (also lists the reflog when no id given)
// ============================================================================

fn merge_undo(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let a = obj(&args);
        match a.get("operation_id").and_then(|v| v.as_str()) {
            Some(op_id) => {
                let op = storage.merge_undo(op_id).map_err(|e| e.to_string())?;
                Ok(json!({
                    "undoOperationId": op.id,
                    "revertedOperationId": op.reverts_op_id,
                    "status": "reverted",
                    "affectedIds": op.affected_ids,
                    "reason": op.reason,
                    "note": "The original operation was reversed: survivor content/tags restored and invalidation cleared. The plan is re-openable."
                }))
            }
            None => {
                // No id => return the reflog so the caller can pick one.
                let ops = storage.list_merge_operations(20).map_err(|e| e.to_string())?;
                let log: Vec<Value> = ops
                    .iter()
                    .map(|op| {
                        json!({
                            "operationId": op.id,
                            "opType": op.op_type,
                            "status": op.status,
                            "survivorId": op.survivor_id,
                            "affectedIds": op.affected_ids,
                            "confidence": op.confidence.map(|c| format!("{:.3}", c)),
                            "reason": op.reason,
                            "createdAt": op.created_at,
                            "revertedAt": op.reverted_at
                        })
                    })
                    .collect();
                Ok(json!({
                    "operations": log,
                    "totalOperations": log.len(),
                    "note": "This is the reversible operation log (the memory reflog). Pass operation_id to reverse one."
                }))
            }
        }
    }
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let _ = (storage, args);
        Err("Embeddings feature not enabled.".into())
    }
}

// ============================================================================
// protect
// ============================================================================

fn protect(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let a = obj(&args);
    let id = a
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or("id is required")?;
    let protected = a.get("protected").and_then(|v| v.as_bool()).unwrap_or(true);
    storage
        .set_protected(id, protected)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "id": id,
        "protected": protected,
        "note": if protected {
            "Memory pinned. It can never be auto-merged, superseded, or garbage-collected until unprotected."
        } else {
            "Memory unprotected. It is now eligible for merge/supersede/forget again."
        }
    }))
}

// ============================================================================
// merge_policy (get when no args, set otherwise)
// ============================================================================

fn merge_policy(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let a = obj(&args);
    let current = storage.get_merge_policy().map_err(|e| e.to_string())?;

    let has_update = a.contains_key("match_threshold")
        || a.contains_key("possible_threshold")
        || a.contains_key("auto_apply");

    if has_update {
        let match_t = a
            .get("match_threshold")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(current.match_threshold);
        let possible_t = a
            .get("possible_threshold")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(current.possible_threshold);
        let auto = a
            .get("auto_apply")
            .and_then(|v| v.as_bool())
            .unwrap_or(current.auto_apply);
        let policy = vestige_core::MergePolicy::new(match_t, possible_t, auto);
        storage.set_merge_policy(policy).map_err(|e| e.to_string())?;
        Ok(json!({
            "updated": true,
            "matchThreshold": policy.match_threshold,
            "possibleThreshold": policy.possible_threshold,
            "autoApply": policy.auto_apply,
            "note": "Policy saved. Fellegi-Sunter: score>=match => auto-merge eligible; [possible,match) => review; below => not offered."
        }))
    } else {
        Ok(json!({
            "matchThreshold": current.match_threshold,
            "possibleThreshold": current.possible_threshold,
            "autoApply": current.auto_apply,
            "note": "Two-threshold merge policy. Pass match_threshold / possible_threshold / auto_apply to change it."
        }))
    }
}

// ============================================================================
// TESTS — see tests/merge_supersede_test.rs for full integration coverage.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_are_objects() {
        for s in [
            merge_candidates_schema(),
            plan_merge_schema(),
            plan_supersede_schema(),
            apply_plan_schema(),
            merge_undo_schema(),
            protect_schema(),
            merge_policy_schema(),
        ] {
            assert_eq!(s["type"], "object");
        }
    }

    #[test]
    fn plan_merge_requires_two_ids() {
        assert!(plan_merge_schema()["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "member_ids"));
    }
}
