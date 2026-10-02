//! Find Duplicates Tool
//!
//! Detects duplicate memory clusters by EXACT EQUALITY ONLY (owner decision
//! 2026-09-28: no similarity anywhere in dedup nomination). Clusters form
//! when memories share an identical content identity (the stored envelope
//! `content_hash`, or byte-identical content when no hash was recorded) or
//! the same declared source key (`source_system` + `source_id`). The former
//! cosine-similarity clustering over stored embeddings was removed.

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::cognitive::CognitiveEngine;
use tokio::sync::Mutex;

use vestige_core::Storage;

/// Input schema for find_duplicates tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of duplicate clusters to return (default: 20)",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional: only check memories with these tags (ANY match)"
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DedupArgs {
    limit: Option<usize>,
    tags: Option<Vec<String>>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: DedupArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => DedupArgs {
            limit: None,
            tags: None,
        },
    };
    let limit = args.limit.unwrap_or(20);
    let tag_filter = args.tags.unwrap_or_default();

    // Load nodes for metadata (content, retention, tags, source envelope).
    let mut all_nodes = Vec::new();
    let mut offset = 0;
    loop {
        let batch = storage
            .get_all_nodes(500, offset)
            .map_err(|e| format!("Failed to load nodes: {}", e))?;
        let batch_len = batch.len();
        all_nodes.extend(batch);
        if batch_len < 500 {
            break;
        }
        offset += 500;
    }

    // Filter by tags if specified
    let filtered: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|n| tag_filter.is_empty() || tag_filter.iter().any(|t| n.tags.contains(t)))
        .collect();

    let n = filtered.len();

    // Union-find over the two exact-equality keys.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }
    let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    };

    // Key 1: identical content identity — the stored envelope content hash
    // when present, else the exact content itself.
    let mut by_content: HashMap<String, Vec<usize>> = HashMap::new();
    // Key 2: the same declared source key (source_system + source_id).
    let mut by_source: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, node) in filtered.iter().enumerate() {
        let env = node.source_envelope.as_ref();
        let content_key = env
            .and_then(|e| e.content_hash.clone())
            .unwrap_or_else(|| node.content.clone());
        by_content.entry(content_key).or_default().push(i);
        if let Some(env) = env
            && let (Some(system), Some(id)) = (&env.source_system, &env.source_id)
        {
            // Same granularity as the store's UNIQUE source-key index:
            // (system, project, id). Two projects' "issue 42" stay separate.
            let project = env.source_project.as_deref().unwrap_or("");
            by_source
                .entry(format!("{system}:{project}:{id}"))
                .or_default()
                .push(i);
        }
    }
    for group in by_content.values().chain(by_source.values()) {
        for pair in group.windows(2) {
            union(&mut parent, pair[0], pair[1]);
        }
    }

    // Group into clusters
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        cluster_map.entry(root).or_default().push(i);
    }

    // Only keep clusters with >1 member, sorted by size descending
    let mut clusters: Vec<Vec<usize>> = cluster_map.into_values().filter(|c| c.len() > 1).collect();
    clusters.sort_by_key(|b| std::cmp::Reverse(b.len()));
    clusters.truncate(limit);

    // Format output
    let cluster_results: Vec<Value> = clusters
        .iter()
        .enumerate()
        .map(|(ci, members)| {
            let anchor = members[0];
            let anchor_env = filtered[anchor].source_envelope.as_ref();
            let anchor_content_key = anchor_env
                .and_then(|e| e.content_hash.clone())
                .unwrap_or_else(|| filtered[anchor].content.clone());
            let anchor_source_key = anchor_env.and_then(|e| {
                Some(format!(
                    "{}:{}:{}",
                    e.source_system.as_deref()?,
                    e.source_project.as_deref().unwrap_or(""),
                    e.source_id.as_deref()?
                ))
            });
            let member_results: Vec<Value> = members
                .iter()
                .map(|&idx| {
                    let node = filtered[idx];
                    let content_preview = {
                        let c = node.content.replace('\n', " ");
                        if c.len() > 120 {
                            format!("{}...", &c[..c.floor_char_boundary(120)])
                        } else {
                            c
                        }
                    };
                    let env = node.source_envelope.as_ref();
                    let member_content_key = env
                        .and_then(|e| e.content_hash.clone())
                        .unwrap_or_else(|| node.content.clone());
                    let member_source_key = env.and_then(|e| {
                        Some(format!(
                            "{}:{}:{}",
                            e.source_system.as_deref()?,
                            e.source_project.as_deref().unwrap_or(""),
                            e.source_id.as_deref()?
                        ))
                    });

                    // Relation to the cluster anchor: exact content identity,
                    // exact source key, or reached transitively through a
                    // chain of exact equalities.
                    let relation = if idx == anchor {
                        "anchor"
                    } else if member_content_key == anchor_content_key {
                        "content"
                    } else if member_source_key.as_deref() == anchor_source_key.as_deref() {
                        "source_key"
                    } else {
                        "transitive"
                    };

                    serde_json::json!({
                        "id": node.id,
                        "contentPreview": content_preview,
                        "retention": node.retention_strength,
                        "createdAt": node.created_at.to_rfc3339(),
                        "tags": &node.tags,
                        "matchRelation": relation
                    })
                })
                .collect();

            serde_json::json!({
                "clusterId": ci,
                "size": members.len(),
                "members": member_results,
                "suggestedAction": if members.len() > 3 { "review" } else { "merge" }
            })
        })
        .collect();

    Ok(serde_json::json!({
        "clusters": cluster_results,
        "totalClusters": cluster_results.len(),
        "totalMemories": all_nodes.len(),
        "totalScanned": n,
        "note": "Exact-equality grouping only: identical content hash (or byte-identical content) and identical declared source keys. Similarity-based clustering was removed."
    }))
}

// ============================================================================
// UNIFIED `dedup` TOOL (v2.2 — Tool Consolidation)
//
// Folds the 8 former dedup/merge tools into a single action-dispatched surface:
//   action = scan (default) | plan_merge | plan_supersede | apply | undo
//          | tag_rename | tag_merge | protect | policy
//
// `scan` combines exact-equality duplicate clusters (this module's `execute`:
// identical content hash or declared source key) with Fellegi-Sunter labelled
// merge candidates (`merge::merge_candidates`, itself nominated by exact
// equality only), returning both in separate fields. The mutate/preview/reverse
// actions delegate to `super::merge::execute` verbatim, preserving
// plan_id → apply → undo, confirm-gating, and bitemporal-never-delete
// byte-for-byte.
// ============================================================================

/// Discriminated-union schema for the unified `dedup` tool.
pub fn unified_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "plan_merge", "plan_supersede", "apply", "undo", "verdict", "tag_rename", "tag_merge", "protect", "policy"],
                "default": "scan",
                "description": "'scan' (default, read-only): duplicate clusters, merge candidates, pending reconsolidation plans. 'plan_merge' / 'plan_supersede': preview a reversible plan. 'apply': run a plan_id. 'undo': reverse an operation_id, or list the reflog. 'verdict': approve|reject|quarantine a reconsolidation plan. 'tag_rename' / 'tag_merge': preview-token gated. 'protect': pin a memory. 'policy': thresholds."
            },
            "limit": {
                "type": "integer",
                "description": "[scan] Max clusters and candidates (default 20).",
                "minimum": 1, "maximum": 100
            },
            "tags": {
                "type": "array", "items": { "type": "string" },
                "description": "[scan] Only memories with any of these tags."
            },
            "member_ids": {
                "type": "array", "items": { "type": "string" },
                "description": "[plan_merge] Two or more memory ids; the survivor is kept, the rest invalidated."
            },
            "survivor_id": { "type": "string", "description": "[plan_merge] Member to keep (default: highest retention)." },
            "old_id": { "type": "string", "description": "[plan_supersede] Memory being superseded (kept, marked invalid)." },
            "new_id": { "type": "string", "description": "[plan_supersede] Memory that supersedes it." },
            "plan_id": { "type": "string", "description": "[apply, verdict] Plan id from plan_merge, plan_supersede, or a reconsolidation plan surfaced by smart_ingest/scan." },
            "verdict": {
                "type": "string",
                "enum": ["approve", "reject", "quarantine"],
                "description": "[verdict] Reconsolidation verdict: approve applies the plan (old memory superseded, rollback via undo); reject discards it (memory stays); quarantine suppresses the memory (existing suppress path) and closes the plan."
            },
            "confirm": { "type": "boolean", "default": false, "description": "[apply, undo, tag_*] Explicit confirmation. On Strata, undo of a logged write needs confirm=true. Tag actions preview when false and need preview_token when true." },
            "operation_id": { "type": "string", "description": "[undo] Operation to reverse. Omit to list recent operations plus tagOperations." },
            "source_tag": { "type": "string", "description": "[tag_rename] Exact source tag to rename." },
            "source_tags": { "type": "array", "items": { "type": "string" }, "minItems": 2, "maxItems": 50, "description": "[tag_merge] Two or more exact source tags." },
            "target_tag": { "type": "string", "description": "[tag_rename/tag_merge] Exact destination tag." },
            "scope": { "type": "string", "default": "user", "description": "[tag_*] Project scope (default 'user')." },
            "all_scopes": { "type": "boolean", "default": false, "description": "[tag_*] Operate across every scope. Default false." },
            "preview_token": { "type": "string", "description": "[tag_* confirm=true] Token from the matching preview." },
            "reason": { "type": "string", "minLength": 1, "maxLength": 1000, "description": "[tag_* confirm=true] Audit reason, required." },
            "id": { "type": "string", "description": "[protect] Memory id to protect/unprotect." },
            "protected": { "type": "boolean", "default": true, "description": "[protect] true to pin, false to unpin." },
            "match_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "[policy] Score >= this => 'match'." },
            "possible_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "[policy] Score in [possible, match) => review." },
            "auto_apply": { "type": "boolean", "description": "[policy] Apply 'match' plans without confirm. Default false." }
        }
    })
}

/// Unified dispatcher for the `dedup` tool. Routes on `action` (default `scan`).
///
/// `cognitive` is used only by the `verdict` action: quarantine must close the
/// in-memory labile window alongside the storage suppression, and the window
/// lives in the cognitive engine. Callers without cognitive access pass
/// `None`; the window then simply self-expires, which is always safe.
pub async fn execute_unified(
    storage: &Arc<Storage>,
    cognitive: Option<&Arc<Mutex<CognitiveEngine>>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let action = args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(|v| v.as_str())
        .unwrap_or("scan")
        .to_string();

    match action.as_str() {
        "scan" => {
            // Exact-equality duplicate clusters (this module).
            let clusters = execute(storage, args.clone()).await?;
            // Fellegi-Sunter labelled merge candidates, nominated by exact
            // equality in storage (merge module, name-dispatched).
            let candidates =
                super::merge::execute(storage, "merge_candidates", args.clone()).await?;
            // Pending reconsolidation verdicts — expired plans are swept by
            // the listing itself, so only live windows are surfaced.
            let reconsolidation_plans = reconsolidation_plan_entries(storage);
            // Merge planning is withheld on this build when its candidate
            // scoring is unavailable; do not point the caller at it.
            let next_step = if super::unavailable::is_unavailable(&candidates) {
                "Exact duplicates are listed under duplicateClusters. Merge planning (plan_merge, plan_supersede, apply, verdict) is withheld in this build, so there is nothing to apply; use action='tag_rename' or 'tag_merge' (preview first) for tag cleanup, or action='undo' to list recent reversible operations."
            } else {
                "Use action='plan_merge' (member_ids) or action='plan_supersede' (old_id,new_id) to preview a reversible plan, then action='apply' (plan_id). Reconsolidation plans take action='verdict' (verdict=approve|reject|quarantine)."
            };
            Ok(serde_json::json!({
                "action": "scan",
                "duplicateClusters": clusters,
                "mergeCandidates": candidates,
                "reconsolidationPlans": reconsolidation_plans,
                "nextStep": next_step
            }))
        }
        "plan_merge" => super::merge::execute(storage, "plan_merge", args).await,
        "plan_supersede" => super::merge::execute(storage, "plan_supersede", args).await,
        "apply" => super::merge::execute(storage, "apply_plan", args).await,
        "undo" => super::merge::execute(storage, "merge_undo", args).await,
        "verdict" => execute_verdict(storage, cognitive, args),
        "tag_rename" => execute_tag_mutation(storage, args, false),
        "tag_merge" => execute_tag_mutation(storage, args, true),
        "protect" => super::merge::execute(storage, "protect", args).await,
        "policy" => super::merge::execute(storage, "merge_policy", args).await,
        other => Err(format!(
            "Unknown dedup action '{other}'. Use scan|plan_merge|plan_supersede|apply|undo|verdict|tag_rename|tag_merge|protect|policy."
        )),
    }
}

/// Pending reconsolidation plans shaped for the scan surface. Read-only; the
/// storage listing runs the expiry sweep first so nothing stale is offered.
#[cfg(vestige_embeddings_removed)]
fn reconsolidation_plan_entries(storage: &Arc<Storage>) -> Vec<Value> {
    match storage.list_reconsolidation_plans(20) {
        Ok(plans) => plans
            .into_iter()
            .map(|(plan, status)| {
                let meta = plan.reconsolidation.as_ref();
                serde_json::json!({
                    "planId": plan.id,
                    "status": status,
                    "targetMemoryId": meta.map(|m| m.target_memory_id.clone()),
                    "trigger": meta.map(|m| m.trigger.clone()),
                    "windowExpiresAt": meta.map(|m| m.window_expires_at.to_rfc3339()),
                    "survivorId": plan.survivor_id,
                    "explanation": plan.explanation,
                    "nextStep": "decide with action='verdict' (verdict=approve|reject|quarantine, plan_id)"
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(not(vestige_embeddings_removed))]
fn reconsolidation_plan_entries(_storage: &Arc<Storage>) -> Vec<Value> {
    Vec::new()
}

/// Apply an explicit reconsolidation verdict. Approve delegates to the
/// existing apply path (reversible via undo); reject closes the plan leaving
/// the memory untouched; quarantine suppresses the memory through the
/// existing suppress path and closes the in-memory labile window.
#[cfg(vestige_embeddings_removed)]
fn obj(args: &Option<Value>) -> serde_json::Map<String, Value> {
    args.as_ref()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn execute_verdict(
    storage: &Arc<Storage>,
    cognitive: Option<&Arc<Mutex<CognitiveEngine>>>,
    args: Option<Value>,
) -> Result<Value, String> {
    #[cfg(vestige_embeddings_removed)]
    {
        let a = obj(&args);
        let plan_id = a
            .get("plan_id")
            .and_then(|v| v.as_str())
            .ok_or("plan_id is required for the verdict action")?;
        let verdict = a
            .get("verdict")
            .and_then(|v| v.as_str())
            .ok_or("verdict is required (approve | reject | quarantine)")?;
        let reason = a.get("reason").and_then(|v| v.as_str());

        // Capture the target before the verdict so quarantine can close the
        // cognitive-side window afterwards.
        let target_id = storage
            .get_plan(plan_id)
            .map_err(|e| e.to_string())?
            .and_then(|plan| plan.reconsolidation.map(|meta| meta.target_memory_id));

        let op = storage
            .verdict_reconsolidation_plan(plan_id, verdict, reason)
            .map_err(|e| e.to_string())?;

        // Quarantine: the storage layer suppressed the memory; close the
        // in-memory labile window so no further modification slips in while
        // the memory is inhibited. Best-effort: the window also self-expires,
        // and callers without cognitive access pass None.
        if verdict == "quarantine"
            && let Some(target) = target_id.as_deref()
            && let Some(cognitive) = cognitive
            && let Ok(mut cog) = cognitive.try_lock()
        {
            cog.reconsolidation.close_window(target);
        }

        Ok(serde_json::json!({
            "action": "verdict",
            "verdict": verdict,
            "planId": plan_id,
            "operationId": op.id,
            "status": op.status,
            "targetMemoryId": target_id,
            "survivorId": op.survivor_id,
            "reason": op.reason,
            "reversible": verdict == "approve",
            "nextStep": if verdict == "approve" {
                format!("To reverse the approval, call merge_undo with operation_id='{}'.", op.id)
            } else if verdict == "quarantine" {
                "The target memory is suppressed (top-down inhibition). suppress with reverse=true within 24h un-suppresses it.".to_string()
            } else {
                "The plan was discarded and the target memory is unchanged.".to_string()
            },
            "note": "Reconsolidation verdicts are recorded in the merge_operations reflog. Expired plans cannot be verdicted; they auto-close with their labile window."
        }))
    }
    #[cfg(not(vestige_embeddings_removed))]
    {
        let _ = (storage, cognitive, args);
        Err("Reconsolidation verdicts require embeddings and vector-search features.".into())
    }
}

fn execute_tag_mutation(
    storage: &Arc<Storage>,
    args: Option<Value>,
    merge: bool,
) -> Result<Value, String> {
    let args = args
        .as_ref()
        .and_then(Value::as_object)
        .ok_or("tag action arguments must be an object")?;
    let target_tag = args
        .get("target_tag")
        .and_then(Value::as_str)
        .ok_or("target_tag is required")?;
    let source_tags = if merge {
        let tags = args
            .get("source_tags")
            .and_then(Value::as_array)
            .ok_or("source_tags is required for tag_merge")?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or("source_tags must contain only strings")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tags.len() < 2 {
            return Err("tag_merge requires at least two source_tags".into());
        }
        tags
    } else {
        vec![
            args.get("source_tag")
                .and_then(Value::as_str)
                .ok_or("source_tag is required for tag_rename")?
                .to_string(),
        ]
    };
    let all_scopes = args
        .get("all_scopes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // An explicit scope alongside all_scopes=true is contradictory: silently
    // dropping the scope would rewrite every scope while the caller believes
    // the mutation is project-scoped.
    if all_scopes && args.get("scope").is_some_and(|value| !value.is_null()) {
        return Err("pass either scope or all_scopes=true, not both".into());
    }
    let scope = if all_scopes {
        None
    } else {
        Some(
            args.get("scope")
                .and_then(Value::as_str)
                .unwrap_or(vestige_core::DEFAULT_MEMORY_SCOPE),
        )
    };
    let confirm = args
        .get("confirm")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let op_type = if merge { "tag_merge" } else { "tag_rename" };

    if !confirm {
        let mut preview = storage
            .preview_tag_mutation(&source_tags, target_tag, scope)
            .map_err(|error| error.to_string())?;
        if let Some(object) = preview.as_object_mut() {
            object.insert("action".into(), Value::String(op_type.into()));
            object.insert(
                "nextStep".into(),
                Value::String(format!(
                    "Review this preview, then call dedup action='{op_type}' with confirm=true, this preview_token, and a nonempty reason."
                )),
            );
        }
        return Ok(preview);
    }

    let preview_token = args
        .get("preview_token")
        .and_then(Value::as_str)
        .ok_or("preview_token is required when confirm=true")?;
    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .ok_or("reason is required when confirm=true")?;
    let operation = storage
        .apply_tag_mutation(
            &source_tags,
            target_tag,
            scope,
            preview_token,
            op_type,
            reason,
        )
        .map_err(|error| error.to_string())?;
    let operation_id = operation.id.clone();
    let operation_scope = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("scope"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let operation_all_scopes = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("allScopes"))
        .and_then(Value::as_bool)
        .unwrap_or(all_scopes);
    let operation_source_tags = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("sourceTags"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!(source_tags));
    let operation_target_tag = operation
        .signals
        .as_ref()
        .and_then(|signals| signals.get("targetTag"))
        .cloned()
        .unwrap_or_else(|| Value::String(target_tag.to_string()));
    Ok(serde_json::json!({
        "action": op_type,
        "status": "applied",
        "operationId": operation_id,
        "affectedMemoryCount": operation.affected_ids.len(),
        "affectedMemoryIds": operation.affected_ids,
        "scope": operation_scope,
        "allScopes": operation_all_scopes,
        "sourceTags": operation_source_tags,
        "targetTag": operation_target_tag,
        "reason": operation.reason,
        "reversible": true,
        "nextStep": format!("To reverse this operation without overwriting later tag edits, call dedup action='undo' with operation_id='{operation_id}'."),
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    #[test]
    fn test_schema() {
        let schema = schema();
        assert_eq!(schema["type"], "object");
        // The cosine similarity_threshold parameter is gone: nomination is
        // exact equality only.
        assert!(schema["properties"].get("similarity_threshold").is_none());
        assert!(schema["properties"]["limit"].is_object());
        assert!(schema["properties"]["tags"].is_object());
    }

    #[test]
    fn test_unified_schema() {
        let schema = unified_schema();
        assert_eq!(schema["type"], "object");
        let actions = schema["properties"]["action"]["enum"].as_array().unwrap();
        assert_eq!(actions.len(), 10);
        assert!(
            actions.contains(&serde_json::json!("verdict")),
            "the reconsolidation verdict action must be advertised"
        );
        assert_eq!(schema["properties"]["action"]["default"], "scan");
    }

    #[tokio::test]
    async fn test_unified_scan_empty_storage() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let storage = Arc::new(storage);
        // Default action (scan) on empty storage must not error.
        let result = execute_unified(&storage, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn tag_rename_requires_preview_token_and_supports_agent_visible_undo() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let node = storage
            .ingest(IngestInput {
                content: "MCP tag rename fixture".to_string(),
                tags: vec!["old".to_string(), "keep".to_string()],
                ..Default::default()
            })
            .unwrap();

        let preview = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user "
            })),
        )
        .await
        .unwrap();
        assert_eq!(preview["requiresConfirmation"], true);
        assert_eq!(preview["affectedMemoryCount"], 1);
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old", "keep"]
        );

        let missing_token = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user ",
                "confirm": true,
                "reason": "normalize tag"
            })),
        )
        .await;
        assert!(missing_token.unwrap_err().contains("preview_token"));

        let applied = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": " user ",
                "confirm": true,
                "preview_token": preview["previewToken"],
                "reason": "normalize tag"
            })),
        )
        .await
        .unwrap();
        assert_eq!(applied["status"], "applied");
        assert_eq!(applied["scope"], "user");
        assert_eq!(applied["sourceTags"], serde_json::json!(["old"]));
        assert_eq!(applied["targetTag"], "new");
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["new", "keep"]
        );

        let undone = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "undo",
                "operation_id": applied["operationId"]
            })),
        )
        .await
        .unwrap();
        assert_eq!(undone["status"], "reverted");
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["old", "keep"]
        );
    }

    #[tokio::test]
    async fn tag_actions_reject_scope_combined_with_all_scopes() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        storage
            .ingest(IngestInput {
                content: "scope conflict fixture".to_string(),
                tags: vec!["old".to_string()],
                ..Default::default()
            })
            .unwrap();

        let conflict = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": "project-a",
                "all_scopes": true
            })),
        )
        .await
        .unwrap_err();
        assert!(conflict.contains("not both"));

        let scoped_only = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "scope": "user"
            })),
        )
        .await
        .unwrap();
        assert_eq!(scoped_only["allScopes"], false);
        assert_eq!(scoped_only["affectedMemoryCount"], 1);

        let all_scopes_only = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": "old",
                "target_tag": "new",
                "all_scopes": true
            })),
        )
        .await
        .unwrap();
        assert_eq!(all_scopes_only["allScopes"], true);
        assert_eq!(all_scopes_only["affectedMemoryCount"], 1);
    }

    #[tokio::test]
    async fn overlong_source_tag_is_renameable_end_to_end() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let overlong = "z".repeat(250);
        let node = storage
            .ingest(IngestInput {
                content: "overlong tag repair fixture".to_string(),
                tags: vec![overlong.clone()],
                ..Default::default()
            })
            .unwrap();

        let preview = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": overlong,
                "target_tag": "short-tag",
                "scope": "user"
            })),
        )
        .await
        .unwrap();
        assert_eq!(preview["affectedMemoryCount"], 1);

        let applied = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_rename",
                "source_tag": overlong,
                "target_tag": "short-tag",
                "scope": "user",
                "confirm": true,
                "preview_token": preview["previewToken"],
                "reason": "repair an overlong stored tag"
            })),
        )
        .await
        .unwrap();
        assert_eq!(applied["status"], "applied");
        assert_eq!(
            storage.get_node(&node.id).unwrap().unwrap().tags,
            vec!["short-tag"],
            "the overlong tag must be gone after the rename"
        );
    }

    #[tokio::test]
    async fn tag_merge_requires_multiple_sources() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let error = execute_unified(
            &storage,
            None,
            Some(serde_json::json!({
                "action": "tag_merge",
                "source_tags": ["one"],
                "target_tag": "target"
            })),
        )
        .await
        .unwrap_err();
        assert!(error.contains("at least two"));
    }

    #[tokio::test]
    async fn test_empty_storage() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let storage = Arc::new(storage);
        let result = execute(&storage, None).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result["clusters"], serde_json::json!([]));
        assert_eq!(result["totalClusters"], 0);
    }

    #[tokio::test]
    async fn identical_content_clusters_by_exact_equality() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Deploy the gateway before Friday".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Deploy the gateway before Friday".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "An unrelated cooking note".to_string(),
                ..Default::default()
            })
            .unwrap();

        let result = execute(&storage, None).await.unwrap();
        assert_eq!(result["totalClusters"], 1, "{result}");
        let cluster = &result["clusters"][0];
        assert_eq!(cluster["size"], 2);
        let relations: Vec<&str> = cluster["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["matchRelation"].as_str().unwrap())
            .collect();
        assert!(relations.contains(&"anchor"));
        assert!(relations.contains(&"content"));
        assert_eq!(result["totalMemories"], 3);
    }

    #[tokio::test]
    async fn identical_content_hash_clusters_across_different_text() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        // SourceEnvelope is #[non_exhaustive]: build via Default + field
        // mutation instead of a struct expression. Two renderings of the same
        // upstream record share its declared content hash. (Same-source-key
        // duplicates are blocked by the store's UNIQUE index, so that legacy
        // shape is covered by the vestige-core nomination tests instead.)
        let mut envelope = vestige_core::SourceEnvelope::default();
        envelope.content_hash = Some("sha256:abc".to_string());
        storage
            .ingest(vestige_core::IngestInput {
                content: "Redmine 42 original description".to_string(),
                source_envelope: Some(envelope.clone()),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Redmine 42 edited description after upstream change".to_string(),
                source_envelope: Some(envelope),
                ..Default::default()
            })
            .unwrap();

        let result = execute(&storage, None).await.unwrap();
        assert_eq!(result["totalClusters"], 1, "{result}");
        let relations: Vec<&str> = result["clusters"][0]["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["matchRelation"].as_str().unwrap())
            .collect();
        assert!(relations.contains(&"content"));
    }

    #[tokio::test]
    async fn near_identical_content_is_not_clustered() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Use tokio runtime for async Rust services".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Use the tokio runtime for async Rust services".to_string(),
                ..Default::default()
            })
            .unwrap();

        let result = execute(&storage, None).await.unwrap();
        assert_eq!(
            result["totalClusters"], 0,
            "near-identical content must not cluster without an exact key: {result}"
        );
    }

    #[tokio::test]
    async fn tag_filter_restricts_clusters() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Duplicated release note".to_string(),
                tags: vec!["rust".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "Duplicated release note".to_string(),
                tags: vec!["python".to_string()],
                ..Default::default()
            })
            .unwrap();

        let unfiltered = execute(&storage, None).await.unwrap();
        assert_eq!(unfiltered["totalClusters"], 1);

        let filtered = execute(&storage, Some(serde_json::json!({ "tags": ["rust"] })))
            .await
            .unwrap();
        assert_eq!(
            filtered["totalClusters"], 0,
            "filtering out one member dissolves the cluster: {filtered}"
        );
    }
}
