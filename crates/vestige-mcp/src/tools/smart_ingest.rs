//! Smart Ingest Tool
//!
//! Intelligent memory ingestion with Prediction Error Gating.
//! Automatically decides whether to create, update, or supersede memories
//! based on semantic similarity to existing content.
//!
//! This solves the "bad vs good similar memory" problem by:
//! - Detecting when new content is similar to existing memories
//! - Updating existing memories when appropriate (low prediction error)
//! - Creating new memories when content is substantially different (high PE)
//! - Superseding demoted/outdated memories with better alternatives
//!
//! v1.5.0: Enhanced with cognitive pipeline:
//!   Pre-ingest: importance scoring (4-channel) + intent detection → auto-tag
//!   Post-ingest: synaptic tagging + novelty model update + hippocampal indexing

use chrono::Utc;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{
    CapturedMemory, ContentType, ImportanceContext, ImportanceEvent, ImportanceEventType,
    IngestInput, Receipt, ReceiptMutation, SecretPolicy, Storage, StorageError, scan_secrets,
};

/// Input schema for smart_ingest tool
///
/// Supports two modes:
/// - **Single mode**: provide `content` (required) + optional fields
/// - **Batch mode**: provide `items` array (max 20), each with full cognitive pipeline
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember. Will be compared against existing memories. (Single mode)"
            },
            "node_type": {
                "type": "string",
                "description": "Type of knowledge: fact, concept, event, person, place, note, pattern, decision",
                "default": "fact"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tags for categorization"
            },
            "source": {
                "type": "string",
                "description": "Source or reference for this knowledge"
            },
            "forceCreate": {
                "type": "boolean",
                "description": "Force creation of a new memory even if similar content exists",
                "default": false
            },
            "allowSecrets": {
                "type": "boolean",
                "description": "Allow a detected credential to be stored for this single item. Dangerous: normally redact the value or store a secret-manager reference instead.",
                "default": false
            },
            "batchMergePolicy": {
                "type": "string",
                "enum": ["force_create", "smart"],
                "description": "Batch mode only. Defaults to 'force_create' so caller-separated items stay separate. Use 'smart' to allow Prediction Error Gating against existing memories.",
                "default": "force_create"
            },
            "items": {
                "type": "array",
                "description": "Batch mode: array of items to save (max 20). Defaults to force-creating each caller-separated item; set batchMergePolicy='smart' to allow Prediction Error Gating against existing memories. Use at session end or before context compaction.",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for categorization"
                        },
                        "node_type": {
                            "type": "string",
                            "description": "Type: fact, concept, event, person, place, note, pattern, decision",
                            "default": "fact"
                        },
                        "source": {
                            "type": "string",
                            "description": "Source reference"
                        },
                        "forceCreate": {
                            "type": "boolean",
                            "description": "Force creation of this item even if similar content exists",
                            "default": false
                        },
                        "allowSecrets": {
                            "type": "boolean",
                            "description": "Allow a detected credential for this item only. Defaults to false; do not use for ordinary session summaries.",
                            "default": false
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SmartIngestArgs {
    content: Option<String>,
    #[serde(alias = "node_type")]
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    source: Option<String>,
    force_create: Option<bool>,
    allow_secrets: Option<bool>,
    batch_merge_policy: Option<String>,
    items: Option<Vec<BatchItem>>,
}

/// A single item in batch mode
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchItem {
    content: String,
    tags: Option<Vec<String>>,
    #[serde(alias = "node_type")]
    node_type: Option<String>,
    source: Option<String>,
    force_create: Option<bool>,
    allow_secrets: Option<bool>,
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SmartIngestArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    // Detect mode: batch (items present) vs single (content present)
    if let Some(items) = args.items {
        let batch_merge_policy = args
            .batch_merge_policy
            .unwrap_or_else(|| "force_create".to_string());
        let default_force_create = match batch_merge_policy.as_str() {
            "force_create" => true,
            "smart" => false,
            other => {
                return Err(format!(
                    "Invalid batchMergePolicy '{}'. Must be 'force_create' or 'smart'.",
                    other
                ));
            }
        };
        let global_force = match args.force_create {
            // An EXPLICIT forceCreate is authoritative and must be honored in both
            // policies. Previously `Some(false)` under the default 'force_create'
            // policy fell through to `default_force_create` (= true), silently
            // inverting the caller's explicit false into a force-create.
            Some(explicit) => explicit,
            None => default_force_create,
        };
        return execute_batch(storage, cognitive, items, global_force, &batch_merge_policy).await;
    }

    // Single mode: content is required
    let content = args.content.ok_or(
        "Missing 'content' field. Provide 'content' for single mode or 'items' for batch mode.",
    )?;
    let secret_policy = if args.allow_secrets.unwrap_or(false) {
        SecretPolicy::AllowExplicitly
    } else {
        SecretPolicy::Reject
    };
    let input_has_secret_finding = !scan_secrets(&content).is_empty();

    // Validate content
    if content.trim().is_empty() {
        return Err("Content cannot be empty".to_string());
    }

    if content.len() > 1_000_000 {
        return Err("Content too large (max 1MB)".to_string());
    }

    // ====================================================================
    // COGNITIVE PRE-INGEST: importance scoring + intent detection + content analysis
    // ====================================================================
    let mut importance_composite = 0.0_f64;
    let mut tags = args.tags.unwrap_or_default();

    if let Ok(cog) = cognitive.try_lock() {
        // 4A. Full 4-channel importance scoring
        let context = ImportanceContext::current();
        let importance = cog
            .importance_signals
            .compute_importance(&content, &context);
        importance_composite = importance.composite;

        // 4B. Intent detection → auto-tag
        let intent_result = cog.intent_detector.detect_intent();
        if intent_result.confidence > 0.5 {
            let intent_tag = format!("intent:{:?}", intent_result.primary_intent);
            // Truncate long intent tags
            let intent_tag = if intent_tag.len() > 50 {
                format!("{}...", &intent_tag[..intent_tag.floor_char_boundary(47)])
            } else {
                intent_tag
            };
            tags.push(intent_tag);
        }

        // 4D. Adaptive embedding — detect content type for logging
        let _content_type = ContentType::detect(&content);
    }

    let input = IngestInput {
        content: content.clone(),
        node_type: args.node_type.unwrap_or_else(|| "fact".to_string()),
        source: args.source,
        sentiment_score: 0.0,
        // Store importance composite as sentiment_magnitude for FSRS encoding boost
        sentiment_magnitude: importance_composite,
        tags,
        valid_from: None,
        valid_until: None,
        source_envelope: None,
    };

    // ====================================================================
    // INGEST (storage lock)
    // ====================================================================

    // Check if force_create is enabled
    if args.force_create.unwrap_or(false) {
        let node = storage
            .ingest_with_secret_policy(input, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();
        let has_embedding = node.has_embedding.unwrap_or(false);

        // Post-ingest cognitive side effects
        let synaptic_capture = run_post_ingest(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
        );

        return Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "message": "Memory created (force_create=true)",
            "hasEmbedding": has_embedding,
            "predictionError": 1.0,
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": "Forced creation - skipped similarity check"
        }));
    }

    // Use smart ingest with prediction error gating
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let result = storage
            .smart_ingest_with_secret_policy(input, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = result.node.id.clone();
        let node_content = result.node.content.clone();
        let node_type = result.node.node_type.clone();
        let has_embedding = result.node.has_embedding.unwrap_or(false);
        let previous_content = if input_has_secret_finding {
            Some("[redacted: credential-bearing ingest]".to_string())
        } else {
            result.previous_content.clone()
        };
        let merge_preview = if input_has_secret_finding {
            Some("[redacted: credential-bearing ingest]".to_string())
        } else {
            result.merge_preview.clone()
        };

        // Post-ingest cognitive side effects
        let synaptic_capture = run_post_ingest(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
        );

        Ok(serde_json::json!({
            "success": true,
            "decision": result.decision,
            "nodeId": node_id,
            "message": format!("Smart ingest complete: {}", result.reason),
            "hasEmbedding": has_embedding,
            "similarity": result.similarity,
            "predictionError": result.prediction_error,
            "supersededId": result.superseded_id,
            "previousContent": previous_content,
            "mergedFrom": result.merged_from,
            "mergePreview": merge_preview,
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": result.reason,
            "explanation": match result.decision.as_str() {
                "create" => "Created new memory - content was different enough from existing memories",
                "update" => "Updated existing memory - content was similar to an existing memory",
                "reinforce" => "Reinforced existing memory - content was nearly identical",
                "supersede" => "Superseded old memory - new content is an improvement/correction",
                "merge" => "Merged with related memories - content connects multiple topics",
                "replace" => "Replaced existing memory content entirely",
                "add_context" => "Added new content as context to existing memory",
                _ => "Memory processed successfully"
            }
        }))
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let node = storage
            .ingest_with_secret_policy(input, secret_policy)
            .map_err(|e| e.to_string())?;
        let node_id = node.id.clone();
        let node_content = node.content.clone();
        let node_type = node.node_type.clone();

        let synaptic_capture = run_post_ingest(
            storage,
            cognitive,
            &node_id,
            &node_content,
            &node_type,
            importance_composite,
        );

        Ok(serde_json::json!({
            "success": true,
            "decision": "create",
            "nodeId": node_id,
            "message": "Memory created (smart ingest requires embeddings feature)",
            "hasEmbedding": false,
            "predictionError": 1.0,
            "importanceScore": importance_composite,
            "synapticCapture": synaptic_capture,
            "reason": "Embeddings not available - used regular ingest"
        }))
    }
}

/// Execute batch mode: process up to 20 items, each with full cognitive pipeline.
///
/// Unlike the old `session_checkpoint` tool, batch mode runs the full cognitive
/// pre-ingest (importance scoring, intent detection) and post-ingest (synaptic
/// tagging, novelty update, hippocampal indexing) pipelines per item.
async fn execute_batch(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    items: Vec<BatchItem>,
    global_force_create: bool,
    batch_merge_policy: &str,
) -> Result<Value, String> {
    if items.is_empty() {
        return Err("Items array cannot be empty".to_string());
    }
    if items.len() > 20 {
        return Err("Maximum 20 items per batch".to_string());
    }

    let mut results = Vec::new();
    let mut created = 0u32;
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    let mut updated = 0u32;
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    let updated = 0u32;
    let mut skipped = 0u32;
    let mut errors = 0u32;
    let mut batch_created_node_ids: Vec<String> = Vec::new();

    for (i, item) in items.into_iter().enumerate() {
        // Skip empty content
        if item.content.trim().is_empty() {
            results.push(serde_json::json!({
                "index": i,
                "status": "skipped",
                "reason": "Empty content"
            }));
            skipped += 1;
            continue;
        }

        // Skip content > 1MB
        if item.content.len() > 1_000_000 {
            results.push(serde_json::json!({
                "index": i,
                "status": "skipped",
                "reason": "Content too large (max 1MB)"
            }));
            skipped += 1;
            continue;
        }

        // Extract per-item force_create before consuming other fields
        let item_force_create = item.force_create.unwrap_or(false);
        let secret_policy = if item.allow_secrets.unwrap_or(false) {
            SecretPolicy::AllowExplicitly
        } else {
            SecretPolicy::Reject
        };
        let input_has_secret_finding = !scan_secrets(&item.content).is_empty();

        // ================================================================
        // COGNITIVE PRE-INGEST (per item)
        // ================================================================
        let mut importance_composite = 0.0_f64;
        let mut tags = item.tags.unwrap_or_default();

        if let Ok(cog) = cognitive.try_lock() {
            let context = ImportanceContext::current();
            let importance = cog
                .importance_signals
                .compute_importance(&item.content, &context);
            importance_composite = importance.composite;

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

            let _content_type = ContentType::detect(&item.content);
        }

        let input = IngestInput {
            content: item.content.clone(),
            node_type: item.node_type.unwrap_or_else(|| "fact".to_string()),
            source: item.source,
            sentiment_score: 0.0,
            sentiment_magnitude: importance_composite,
            tags,
            valid_from: None,
            valid_until: None,
            source_envelope: None,
        };

        // ================================================================
        // INGEST (storage lock per item)
        // ================================================================

        // Check force_create: global flag OR per-item flag
        let item_force = global_force_create || item_force_create;
        if item_force {
            match storage.ingest_with_secret_policy(input, secret_policy) {
                Ok(node) => {
                    let node_id = node.id.clone();
                    let node_content = node.content.clone();
                    let node_type = node.node_type.clone();

                    created += 1;
                    batch_created_node_ids.push(node_id.clone());
                    let synaptic_capture = run_post_ingest(
                        storage,
                        cognitive,
                        &node_id,
                        &node_content,
                        &node_type,
                        importance_composite,
                    );

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": "Forced creation - skipped similarity check"
                    }));
                }
                Err(e) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": if matches!(&e, StorageError::SecretDetected { .. }) { "rejected" } else { "error" },
                        "reason": e.to_string()
                    }));
                }
            }
            continue;
        }

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            match storage.smart_ingest_excluding_with_secret_policy(
                input,
                &batch_created_node_ids,
                secret_policy,
            ) {
                Ok(result) => {
                    let node_id = result.node.id.clone();
                    let node_content = result.node.content.clone();
                    let node_type = result.node.node_type.clone();
                    let previous_content = if input_has_secret_finding {
                        Some("[redacted: credential-bearing ingest]".to_string())
                    } else {
                        result.previous_content.clone()
                    };
                    let merge_preview = if input_has_secret_finding {
                        Some("[redacted: credential-bearing ingest]".to_string())
                    } else {
                        result.merge_preview.clone()
                    };

                    match result.decision.as_str() {
                        "create" | "supersede" | "merge" => {
                            created += 1;
                            batch_created_node_ids.push(node_id.clone());
                        }
                        "update" | "reinforce" | "replace" | "add_context" => updated += 1,
                        _ => created += 1,
                    }

                    // Post-ingest cognitive side effects
                    let synaptic_capture = run_post_ingest(
                        storage,
                        cognitive,
                        &node_id,
                        &node_content,
                        &node_type,
                        importance_composite,
                    );

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": result.decision,
                        "nodeId": node_id,
                        "similarity": result.similarity,
                        "predictionError": result.prediction_error,
                        "supersededId": result.superseded_id,
                        "previousContent": previous_content,
                        "mergedFrom": result.merged_from,
                        "mergePreview": merge_preview,
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": result.reason
                    }));
                }
                Err(e) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": if matches!(&e, StorageError::SecretDetected { .. }) { "rejected" } else { "error" },
                        "reason": e.to_string()
                    }));
                }
            }
        }

        #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
        {
            match storage.ingest_with_secret_policy(input, secret_policy) {
                Ok(node) => {
                    let node_id = node.id.clone();
                    let node_content = node.content.clone();
                    let node_type = node.node_type.clone();

                    created += 1;
                    batch_created_node_ids.push(node_id.clone());
                    let synaptic_capture = run_post_ingest(
                        storage,
                        cognitive,
                        &node_id,
                        &node_content,
                        &node_type,
                        importance_composite,
                    );

                    results.push(serde_json::json!({
                        "index": i,
                        "status": "saved",
                        "decision": "create",
                        "nodeId": node_id,
                        "importanceScore": importance_composite,
                        "synapticCapture": synaptic_capture,
                        "reason": "Embeddings not available - used regular ingest"
                    }));
                }
                Err(e) => {
                    errors += 1;
                    results.push(serde_json::json!({
                        "index": i,
                        "status": if matches!(&e, StorageError::SecretDetected { .. }) { "rejected" } else { "error" },
                        "reason": e.to_string()
                    }));
                }
            }
        }
    }

    Ok(serde_json::json!({
        "success": errors == 0,
        "mode": "batch",
        "batchMergePolicy": batch_merge_policy,
        "summary": {
            "total": results.len(),
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "errors": errors
        },
        "results": results
    }))
}

/// Cognitive post-ingest side effects: synaptic tagging, novelty update, hippocampal indexing.
///
/// A high-salience ingest can capture nearby *previously tagged* memories. Each
/// actual capture is promoted and persisted as a normal, fetchable receipt. The
/// receipt contains ids and measured strength changes only — never copied memory
/// content — so purge remains authoritative and suppressed memories stay out.
///
/// Uses try_lock() for non-blocking access. If cognitive is locked, side effects are skipped.
fn run_post_ingest(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    node_id: &str,
    content: &str,
    node_type: &str,
    importance_composite: f64,
) -> Option<Value> {
    let mut synaptic_capture = None;

    if let Ok(mut cog) = cognitive.try_lock() {
        // 4C. Synaptic tagging for retroactive capture. Trigger *before* adding
        // the triggering memory's own tag: a high-salience incident must make
        // an earlier ordinary decision important, not self-capture at t=0.
        if importance_composite > 0.3 {
            let trigger_is_eligible = storage
                .get_node(node_id)
                .ok()
                .flatten()
                .is_some_and(|node| node.suppression_count == 0);

            if importance_composite > 0.7 && trigger_is_eligible {
                let window = cog.synaptic_tagging.config().capture_window.clone();
                let radius = ImportanceEventType::NoveltySpike.capture_radius_multiplier();
                let event = ImportanceEvent::for_memory(node_id, ImportanceEventType::NoveltySpike);
                let capture = cog.synaptic_tagging.trigger_prp(event);
                synaptic_capture = persist_synaptic_capture_receipt(
                    storage,
                    node_id,
                    importance_composite,
                    window.backward_hours * radius,
                    window.forward_hours * radius,
                    &capture.captured_memories,
                );
            }

            // A just-ingested memory is eligible only for a *future* event.
            // Do not tag explicitly suppressed memories for later promotion.
            if trigger_is_eligible {
                cog.synaptic_tagging.tag_memory(node_id);
            }
        }

        // 4E. Update novelty model with new content
        cog.importance_signals.learn_content(content);

        // 4F. Record in hippocampal index
        let _ = cog.hippocampal_index.index_memory(
            node_id,
            content,
            node_type,
            Utc::now(),
            None, // semantic_embedding — generated separately
        );

        // 4G. Cross-project pattern recording
        cog.cross_project
            .record_project_memory(node_id, "default", None);
    }

    synaptic_capture
}

/// Persist the observable part of a synaptic capture without copying either
/// memory's content. Suppressed and purged nodes cannot be promoted; their ids
/// are surfaced as withheld evidence rather than silently changing them.
fn persist_synaptic_capture_receipt(
    storage: &Arc<Storage>,
    trigger_memory_id: &str,
    trigger_importance: f64,
    backward_hours: f64,
    forward_hours: f64,
    captures: &[CapturedMemory],
) -> Option<Value> {
    let mut retrieved = Vec::new();
    let mut trust_scores = Vec::new();
    let mut activation_path = Vec::new();
    let mut mutations = Vec::new();
    let mut captured = Vec::new();
    let mut withheld_suppressed_ids = Vec::new();

    for capture in captures {
        let Some(before) = storage.get_node(&capture.memory_id).ok().flatten() else {
            // A concurrently purged memory has no content or strength left to
            // change, so it cannot be included as evidence.
            continue;
        };
        if before.suppression_count > 0 {
            withheld_suppressed_ids.push(before.id);
            continue;
        }

        let Ok(after) = storage.promote_memory_backfill(&capture.memory_id) else {
            continue;
        };
        retrieved.push(capture.memory_id.clone());
        trust_scores.push(before.retention_strength);
        activation_path.push(format!(
            "{} --[tagged; {:.2}h before event]--> {}",
            capture.memory_id, capture.temporal_distance_hours, trigger_memory_id
        ));
        mutations.push(ReceiptMutation {
            id: capture.memory_id.clone(),
            kind: "synaptic_capture".to_string(),
            note: Some(format!(
                "Evidence-backed temporal association; capture score {:.2}",
                capture.consolidated_importance
            )),
        });
        captured.push(serde_json::json!({
            "memoryId": capture.memory_id,
            "encodedAt": capture.encoded_at.to_rfc3339(),
            "temporalDistanceHours": capture.temporal_distance_hours,
            "captureProbability": capture.capture_probability,
            "tagStrengthAtCapture": capture.tag_strength_at_capture,
            "captureImportance": capture.consolidated_importance,
            "strengthChange": {
                "retrievalStrength": { "before": before.retrieval_strength, "after": after.retrieval_strength },
                "retentionStrength": { "before": before.retention_strength, "after": after.retention_strength },
                "stability": { "before": before.stability, "after": after.stability }
            }
        }));
    }

    if captured.is_empty() {
        return None;
    }

    let receipt = Receipt::build(
        Utc::now(),
        trigger_memory_id,
        retrieved,
        vec![],
        activation_path,
        &trust_scores,
        mutations,
    );
    if let Err(error) = storage.save_receipt(&receipt, None, Some("smart_ingest"), None) {
        tracing::warn!(%error, "synaptic capture receipt save failed");
    }

    Some(serde_json::json!({
        "receiptId": receipt.receipt_id,
        "receipt": receipt,
        "trigger": {
            "memoryId": trigger_memory_id,
            "eventType": "novelty_spike",
            "importanceScore": trigger_importance
        },
        "captureWindow": {
            "backwardHours": backward_hours,
            "forwardHours": forward_hours
        },
        "captured": captured,
        "withheldSuppressedIds": withheld_suppressed_ids,
        "claimBoundary": "Evidence-backed temporal association, not proof that the trigger caused the earlier memory."
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

    #[tokio::test]
    async fn test_smart_ingest_empty_content_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_smart_ingest_basic_content_succeeds() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "This is a test fact to remember."
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
        assert!(value["decision"].is_string());
    }

    #[tokio::test]
    async fn test_smart_ingest_force_create() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Force create test content.",
            "forceCreate": true
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["decision"], "create");
        assert!(
            value["reason"].as_str().unwrap().contains("Forced")
                || value["reason"]
                    .as_str()
                    .unwrap()
                    .contains("Embeddings not available")
        );
    }

    #[tokio::test]
    async fn high_salience_ingest_promotes_earlier_tag_and_saves_capture_receipt() {
        let (storage, _dir) = test_storage().await;
        let cognitive = test_cognitive();
        let earlier = storage
            .ingest(IngestInput {
                content: "We selected the retry policy for the deploy worker.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert!(
            run_post_ingest(
                &storage,
                &cognitive,
                &earlier.id,
                &earlier.content,
                &earlier.node_type,
                0.5,
            )
            .is_none(),
            "an ordinary decision should be tagged but not self-promoted"
        );
        let before = storage.demote_memory(&earlier.id).unwrap();

        let trigger = storage
            .ingest(IngestInput {
                content: "Production deployment failed after the retry policy exhausted."
                    .to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        let capture = run_post_ingest(
            &storage,
            &cognitive,
            &trigger.id,
            &trigger.content,
            &trigger.node_type,
            0.9,
        )
        .expect("high-salience trigger captures the earlier tagged decision");

        let receipt_id = capture["receiptId"].as_str().expect("receipt id");
        let saved = storage
            .get_receipt(receipt_id)
            .unwrap()
            .expect("persisted receipt");
        assert_eq!(saved.retrieved, vec![earlier.id.clone()]);
        assert_eq!(saved.mutations[0].kind, "synaptic_capture");
        assert!(
            saved.activation_path[0].contains(&trigger.id),
            "receipt records the trigger reference"
        );
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert!(after.retrieval_strength > before.retrieval_strength);
        assert_eq!(capture["captured"][0]["memoryId"], earlier.id);
        assert_eq!(capture["trigger"]["memoryId"], trigger.id);
        assert_eq!(
            capture["claimBoundary"],
            "Evidence-backed temporal association, not proof that the trigger caused the earlier memory."
        );
    }

    #[tokio::test]
    async fn suppressed_tag_is_not_promoted_or_exposed_in_a_capture_receipt() {
        let (storage, _dir) = test_storage().await;
        let cognitive = test_cognitive();
        let earlier = storage
            .ingest(IngestInput {
                content: "An ordinary approval was recorded before the incident.".to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .unwrap();
        run_post_ingest(
            &storage,
            &cognitive,
            &earlier.id,
            &earlier.content,
            &earlier.node_type,
            0.5,
        );
        let suppressed = storage.suppress_memory(&earlier.id).unwrap();

        let trigger = storage
            .ingest(IngestInput {
                content: "A high-salience incident arrived after the approval.".to_string(),
                node_type: "event".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert!(
            run_post_ingest(
                &storage,
                &cognitive,
                &trigger.id,
                &trigger.content,
                &trigger.node_type,
                0.9,
            )
            .is_none(),
            "a capture with only suppressed candidates emits no receipt"
        );
        let after = storage.get_node(&earlier.id).unwrap().unwrap();
        assert_eq!(after.suppression_count, suppressed.suppression_count);
        assert_eq!(after.retrieval_strength, suppressed.retrieval_strength);
    }

    #[tokio::test]
    async fn test_smart_ingest_rejects_secret_even_when_force_created() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": format!("Store this token: {secret}"),
                "forceCreate": true
            })),
        )
        .await;

        let err = result.unwrap_err();
        assert!(err.contains("Refused to store probable credential"));
        assert!(
            !err.contains(&secret),
            "MCP errors must not echo the rejected credential"
        );
        assert_eq!(
            storage.get_stats().unwrap().total_nodes,
            0,
            "forceCreate must not bypass the credential guard"
        );
    }

    #[tokio::test]
    async fn test_explicit_secret_override_does_not_echo_content_in_response() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let response = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": format!("intentional local credential: {secret}"),
                "forceCreate": true,
                "allowSecrets": true
            })),
        )
        .await
        .unwrap();

        assert_eq!(response["success"], true);
        assert!(
            !serde_json::to_string(&response).unwrap().contains(&secret),
            "the override response must not copy the credential into an MCP transcript"
        );
    }

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["content"].is_object());
        assert!(schema_value["properties"]["forceCreate"].is_object());
        assert!(schema_value["properties"]["batchMergePolicy"].is_object());
        assert!(schema_value["properties"]["items"].is_object());
        // v1.7: no top-level required — content for single mode, items for batch mode
        assert!(schema_value.get("required").is_none() || schema_value["required"].is_null());
    }

    #[tokio::test]
    async fn test_smart_ingest_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_smart_ingest_whitespace_only_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "   \t\n  " });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_smart_ingest_too_large_fails() {
        let (storage, _dir) = test_storage().await;
        let large = "x".repeat(1_000_001);
        let args = serde_json::json!({ "content": large });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[tokio::test]
    async fn test_smart_ingest_exactly_1mb_succeeds() {
        let (storage, _dir) = test_storage().await;
        let content = "x".repeat(1_000_000);
        let args = serde_json::json!({ "content": content });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_node_type() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "A concept to remember",
            "node_type": "concept"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_tags_and_source() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Tagged and sourced memory",
            "tags": ["test", "smart-ingest"],
            "source": "unit-test"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_smart_ingest_response_has_importance_score() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "Important memory content" });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        let value = result.unwrap();
        assert!(value["importanceScore"].is_number());
    }

    #[tokio::test]
    async fn test_smart_ingest_missing_content_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "tags": ["test"] });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("content"));
    }

    // ========================================================================
    // TESTS PORTED FROM ingest.rs (v1.7.0 merge)
    // ========================================================================

    #[tokio::test]
    async fn test_smart_ingest_with_all_optional_fields() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "Complex memory with all metadata.",
            "node_type": "decision",
            "tags": ["architecture", "design"],
            "source": "team meeting notes"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
        assert!(value["nodeId"].is_string());
    }

    #[tokio::test]
    async fn test_smart_ingest_default_node_type_is_fact() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "content": "Default type test content." });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let node_id = result.unwrap()["nodeId"].as_str().unwrap().to_string();
        let node = storage.get_node(&node_id).unwrap().unwrap();
        assert_eq!(node.node_type, "fact");
    }

    #[test]
    fn test_schema_has_optional_fields() {
        let schema_value = schema();
        assert!(schema_value["properties"]["node_type"].is_object());
        assert!(schema_value["properties"]["tags"].is_object());
        assert!(schema_value["properties"]["source"].is_object());
    }

    #[tokio::test]
    async fn test_smart_ingest_with_source() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "content": "MCP protocol version 2024-11-05 is the current standard.",
            "source": "https://modelcontextprotocol.io/spec"
        });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["success"], true);
    }

    // ========================================================================
    // BATCH MODE TESTS (ported from checkpoint.rs, v1.7.0 merge)
    // ========================================================================

    #[tokio::test]
    async fn test_batch_empty_items_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": [] })),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_batch_ingest() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "First batch item", "tags": ["test"] },
                    { "content": "Second batch item", "tags": ["test"] }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["mode"], "batch");
        assert_eq!(value["batchMergePolicy"], "force_create");
        assert_eq!(value["summary"]["total"], 2);
    }

    #[tokio::test]
    async fn test_batch_ingest_saves_safe_item_and_rejects_secret_item() {
        let (storage, _dir) = test_storage().await;
        let secret = format!("ghp_{}", "A".repeat(36));
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "safe batch memory" },
                    { "content": format!("batch credential: {secret}") }
                ]
            })),
        )
        .await
        .unwrap();

        assert_eq!(result["mode"], "batch");
        assert_eq!(result["results"][0]["status"], "saved");
        assert_eq!(result["results"][1]["status"], "rejected");
        assert!(
            !result["results"][1]["reason"]
                .as_str()
                .unwrap()
                .contains(&secret),
            "batch result must not echo the rejected credential"
        );
        assert_eq!(result["summary"]["created"], 1);
        assert_eq!(result["summary"]["errors"], 1);
        assert_eq!(
            storage.get_stats().unwrap().total_nodes,
            1,
            "safe batch entries may persist while rejected entries never do"
        );
    }

    #[tokio::test]
    async fn test_batch_defaults_to_force_create_for_caller_separated_items() {
        // Default policy (no explicit forceCreate) force-creates each
        // caller-separated item so they stay separate.
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Jira tickets should not auto-assign sprint fields." },
                    { "content": "Sprint planning summaries should not append Jira status labels." }
                ]
            })),
        )
        .await;

        let value = result.unwrap();
        assert_eq!(value["batchMergePolicy"], "force_create");
        assert_eq!(value["summary"]["created"], 2);
        assert_eq!(value["summary"]["updated"], 0);
        for item in value["results"].as_array().unwrap() {
            assert_eq!(item["decision"], "create");
            assert!(item["reason"].as_str().unwrap().contains("Forced creation"));
        }
    }

    #[tokio::test]
    async fn test_batch_explicit_force_create_false_is_honored() {
        // Regression (#130): an EXPLICIT forceCreate:false must NOT be silently
        // inverted to force-create by the default policy. Distinct/novel items are
        // still created (PE gating creates novel content), but NOT via the
        // "Forced creation" path.
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "forceCreate": false,
                "items": [
                    { "content": "Jira tickets should not auto-assign sprint fields." },
                    { "content": "Sprint planning summaries should not append Jira status labels." }
                ]
            })),
        )
        .await;

        let value = result.unwrap();
        assert_eq!(value["summary"]["created"], 2, "novel items still created");
        for item in value["results"].as_array().unwrap() {
            assert!(
                !item["reason"].as_str().unwrap().contains("Forced creation"),
                "explicit forceCreate:false must not force-create"
            );
        }
    }

    #[tokio::test]
    async fn test_batch_rejects_invalid_merge_policy() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "batchMergePolicy": "merge_everything",
                "items": [{ "content": "Invalid policy should fail." }]
            })),
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("batchMergePolicy"));
    }

    #[tokio::test]
    async fn test_batch_skips_empty_content() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Valid item" },
                    { "content": "" },
                    { "content": "Another valid item" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["skipped"], 1);
    }

    #[tokio::test]
    async fn test_batch_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_batch_exceeds_20_items_fails() {
        let (storage, _dir) = test_storage().await;
        let items: Vec<serde_json::Value> = (0..21)
            .map(|i| serde_json::json!({ "content": format!("Item {}", i) }))
            .collect();
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": items })),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Maximum 20 items"));
    }

    #[tokio::test]
    async fn test_batch_exactly_20_items_succeeds() {
        let (storage, _dir) = test_storage().await;
        let items: Vec<serde_json::Value> = (0..20)
            .map(|i| serde_json::json!({ "content": format!("Item {}", i) }))
            .collect();
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "items": items })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["total"], 20);
    }

    #[tokio::test]
    async fn test_batch_skips_whitespace_only_content() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "   \t\n  " },
                    { "content": "Valid content" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["skipped"], 1);
        assert_eq!(value["summary"]["created"], 1);
    }

    #[tokio::test]
    async fn test_batch_single_item_succeeds() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{ "content": "Single item" }]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["total"], 1);
        assert_eq!(value["success"], true);
    }

    #[tokio::test]
    async fn test_batch_items_with_all_fields() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{
                    "content": "Full fields item",
                    "tags": ["test", "batch"],
                    "node_type": "decision",
                    "source": "test-suite"
                }]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["summary"]["created"], 1);
    }

    #[tokio::test]
    async fn test_batch_results_array_matches_items() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "First" },
                    { "content": "" },
                    { "content": "Third" }
                ]
            })),
        )
        .await;
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["index"], 0);
        assert_eq!(results[1]["index"], 1);
        assert_eq!(results[1]["status"], "skipped");
        assert_eq!(results[2]["index"], 2);
    }

    #[tokio::test]
    async fn test_batch_success_true_when_only_skipped() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "" },
                    { "content": "   " }
                ]
            })),
        )
        .await;
        let value = result.unwrap();
        assert_eq!(value["success"], true); // skipped ≠ errors
        assert_eq!(value["summary"]["errors"], 0);
        assert_eq!(value["summary"]["skipped"], 2);
    }

    #[tokio::test]
    async fn test_batch_has_importance_scores() {
        let (storage, _dir) = test_storage().await;
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [{ "content": "Important batch memory content" }]
            })),
        )
        .await;
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results[0]["importanceScore"].is_number());
    }

    #[tokio::test]
    async fn test_batch_force_create_global() {
        let (storage, _dir) = test_storage().await;
        // Three items with very similar content + global forceCreate
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "forceCreate": true,
                "items": [
                    { "content": "Physics question about quantum mechanics and wave functions" },
                    { "content": "Physics question about quantum mechanics and wave equations" },
                    { "content": "Physics question about quantum mechanics and wave behavior" }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["mode"], "batch");
        // All 3 should be created separately, not merged
        assert_eq!(value["summary"]["created"], 3);
        assert_eq!(value["summary"]["updated"], 0);
        // Each result should say "Forced creation"
        let results = value["results"].as_array().unwrap();
        for r in results {
            assert_eq!(r["decision"], "create");
            assert!(r["reason"].as_str().unwrap().contains("Forced"));
        }
    }

    #[tokio::test]
    async fn test_batch_force_create_per_item() {
        let (storage, _dir) = test_storage().await;
        // Mix of forced and non-forced items
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "items": [
                    { "content": "Forced item one", "forceCreate": true },
                    { "content": "Normal item two" },
                    { "content": "Forced item three", "forceCreate": true }
                ]
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        // Forced items should say "Forced creation"
        assert_eq!(results[0]["decision"], "create");
        assert!(results[0]["reason"].as_str().unwrap().contains("Forced"));
        // Non-forced item gets normal processing
        assert_eq!(results[1]["status"], "saved");
        // Third forced item
        assert_eq!(results[2]["decision"], "create");
        assert!(results[2]["reason"].as_str().unwrap().contains("Forced"));
    }

    #[tokio::test]
    async fn test_no_content_no_items_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "tags": ["orphan"] });
        let result = execute(&storage, &test_cognitive(), Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("content"));
    }
}
