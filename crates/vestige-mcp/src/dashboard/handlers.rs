//! Dashboard API endpoint handlers
//!
//! v2.0: Adds cognitive operation endpoints (dream, explore, predict, importance, consolidation)

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path as FsPath, PathBuf};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Json, Redirect};
use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::events::VestigeEvent;
use super::state::AppState;

/// Redirect root to the SvelteKit dashboard
pub async fn serve_dashboard() -> Redirect {
    Redirect::permanent("/dashboard")
}

#[derive(Debug, Deserialize)]
pub struct MemoryListParams {
    pub q: Option<String>,
    pub node_type: Option<String>,
    pub tag: Option<String>,
    pub min_retention: Option<f64>,
    pub sort: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

/// List memories with optional search
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<MemoryListParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(50).clamp(1, 200);
    let offset = params.offset.unwrap_or(0).max(0);

    if let Some(query) = params.q.as_ref().filter(|q| !q.trim().is_empty()) {
        // Use hybrid search
        let results = state
            .storage
            .hybrid_search(query, limit, 0.3, 0.7)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let formatted: Vec<Value> = results
            .into_iter()
            .filter(|r| {
                if let Some(min_ret) = params.min_retention {
                    r.node.retention_strength >= min_ret
                } else {
                    true
                }
            })
            .map(|r| {
                serde_json::json!({
                    "id": r.node.id,
                    "content": r.node.content,
                    "nodeType": r.node.node_type,
                    "tags": r.node.tags,
                    "retentionStrength": r.node.retention_strength,
                    "storageStrength": r.node.storage_strength,
                    "retrievalStrength": r.node.retrieval_strength,
                    "createdAt": r.node.created_at.to_rfc3339(),
                    "updatedAt": r.node.updated_at.to_rfc3339(),
                    "combinedScore": r.combined_score,
                    "source": r.node.source,
                    "reviewCount": r.node.reps,
                })
            })
            .collect();

        return Ok(Json(serde_json::json!({
            "total": formatted.len(),
            "memories": formatted,
        })));
    }

    // No search query — list all memories
    let mut nodes = state
        .storage
        .get_all_nodes(limit, offset)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Apply filters
    if let Some(ref node_type) = params.node_type {
        nodes.retain(|n| n.node_type == *node_type);
    }
    if let Some(ref tag) = params.tag {
        nodes.retain(|n| n.tags.iter().any(|t| t == tag));
    }
    if let Some(min_ret) = params.min_retention {
        nodes.retain(|n| n.retention_strength >= min_ret);
    }

    let formatted: Vec<Value> = nodes
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content,
                "nodeType": n.node_type,
                "tags": n.tags,
                "retentionStrength": n.retention_strength,
                "storageStrength": n.storage_strength,
                "retrievalStrength": n.retrieval_strength,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "source": n.source,
                "reviewCount": n.reps,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "total": formatted.len(),
        "memories": formatted,
    })))
}

/// Get a single memory by ID
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .get_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(serde_json::json!({
        "id": node.id,
        "content": node.content,
        "nodeType": node.node_type,
        "tags": node.tags,
        "retentionStrength": node.retention_strength,
        "storageStrength": node.storage_strength,
        "retrievalStrength": node.retrieval_strength,
        "sentimentScore": node.sentiment_score,
        "sentimentMagnitude": node.sentiment_magnitude,
        "source": node.source,
        "createdAt": node.created_at.to_rfc3339(),
        "updatedAt": node.updated_at.to_rfc3339(),
        "lastAccessedAt": node.last_accessed.to_rfc3339(),
        "nextReviewAt": node.next_review.map(|dt| dt.to_rfc3339()),
        "reviewCount": node.reps,
        "validFrom": node.valid_from.map(|dt| dt.to_rfc3339()),
        "validUntil": node.valid_until.map(|dt| dt.to_rfc3339()),
    })))
}

/// Delete a memory by ID
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let deleted = state
        .storage
        .delete_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if deleted {
        state.emit(VestigeEvent::MemoryDeleted {
            id: id.clone(),
            timestamp: chrono::Utc::now(),
        });
        Ok(Json(serde_json::json!({ "deleted": true, "id": id })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Promote a memory
pub async fn promote_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .promote_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state.emit(VestigeEvent::MemoryPromoted {
        id: node.id.clone(),
        new_retention: node.retention_strength,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "promoted": true,
        "id": node.id,
        "retentionStrength": node.retention_strength,
    })))
}

/// Demote a memory
pub async fn demote_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let node = state
        .storage
        .demote_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state.emit(VestigeEvent::MemoryDemoted {
        id: node.id.clone(),
        new_retention: node.retention_strength,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "demoted": true,
        "id": node.id,
        "retentionStrength": node.retention_strength,
    })))
}

/// Actively suppress a memory via top-down inhibitory control.
///
/// Optional JSON body: `{"reason": "..."}`. Each call compounds — the
/// `suppression_count` field on the memory increments, FSRS takes a hit,
/// and the background Rac1 worker fades co-activated neighbors over the
/// next 72 hours. Emits a `MemorySuppressed` event so the 3D graph plays
/// the violet implosion animation.
///
/// Reversible within the 24-hour labile window via `unsuppress_memory`.
///
/// Fixes the v2.0.5 UI gap: `suppress` had full graph event handlers and
/// MCP tool exposure, but zero HTTP endpoint and no dashboard trigger.
pub async fn suppress_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<Value>>,
) -> Result<Json<Value>, StatusCode> {
    use vestige_core::neuroscience::active_forgetting::{
        ActiveForgettingSystem, DEFAULT_LABILE_HOURS,
    };

    let reason = body
        .as_ref()
        .and_then(|Json(v)| v.get("reason"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let sys = ActiveForgettingSystem::new();

    // Pre-count to surface in the response + for the event payload.
    let before_count = state
        .storage
        .get_node(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map(|n| n.suppression_count)
        .unwrap_or(0);

    let node = state
        .storage
        .suppress_memory(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Estimate cascade size for the UX; capped at 100 so the number is
    // stable even on highly-connected nodes.
    let estimated_cascade = state
        .storage
        .get_connections_for_memory(&id)
        .map(|edges| edges.len().min(100))
        .unwrap_or(0);

    let reversible_until = node
        .suppressed_at
        .map(|t| sys.reversible_until(t))
        .unwrap_or_else(chrono::Utc::now);
    let retrieval_penalty = sys.retrieval_penalty(node.suppression_count);

    tracing::info!(
        id = %id,
        count = node.suppression_count,
        reason = reason.as_deref().unwrap_or(""),
        "Memory suppressed via dashboard"
    );

    state.emit(VestigeEvent::MemorySuppressed {
        id: node.id.clone(),
        suppression_count: node.suppression_count,
        estimated_cascade,
        reversible_until,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "suppressed": true,
        "id": node.id,
        "suppressionCount": node.suppression_count,
        "priorCount": before_count,
        "retrievalPenalty": retrieval_penalty,
        "retentionStrength": node.retention_strength,
        "retrievalStrength": node.retrieval_strength,
        "stability": node.stability,
        "estimatedCascadeNeighbors": estimated_cascade,
        "reversibleUntil": reversible_until.to_rfc3339(),
        "labileWindowHours": DEFAULT_LABILE_HOURS,
        "reason": reason,
    })))
}

/// Reverse a prior suppression, if still inside the 24-hour labile
/// window. Emits `MemoryUnsuppressed` so the graph plays the rainbow
/// reversal burst. Returns the current suppression state so the UI
/// knows whether a single click fully cleared the suppression or whether
/// the memory still has compounded suppressions remaining.
pub async fn unsuppress_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    use vestige_core::neuroscience::active_forgetting::ActiveForgettingSystem;

    let sys = ActiveForgettingSystem::new();
    let node = state
        .storage
        .reverse_suppression(&id, sys.labile_hours)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let still_suppressed = node.suppression_count > 0;

    state.emit(VestigeEvent::MemoryUnsuppressed {
        id: node.id.clone(),
        remaining_count: node.suppression_count,
        timestamp: chrono::Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "unsuppressed": true,
        "id": node.id,
        "suppressionCount": node.suppression_count,
        "stillSuppressed": still_suppressed,
        "retentionStrength": node.retention_strength,
        "retrievalStrength": node.retrieval_strength,
        "stability": node.stability,
    })))
}

#[derive(Debug, Deserialize)]
pub struct SanhedrinAppealRequest {
    pub reason: String,
    pub note: Option<String>,
    #[serde(rename = "receiptId")]
    pub receipt_id: Option<String>,
    #[serde(rename = "claimId")]
    pub claim_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SanhedrinTelemetryParams {
    pub days: Option<i64>,
}

/// Return the latest Sanhedrin receipt written by the Stop-hook bridge.
pub async fn get_sanhedrin_latest() -> Result<Json<Value>, StatusCode> {
    let state_dir = sanhedrin_state_dir();
    let latest_path = state_dir.join("latest.json");
    if !latest_path.exists() {
        return Ok(Json(serde_json::json!({
            "receipt": null,
            "stateDir": state_dir,
        })));
    }

    let raw = fs::read_to_string(&latest_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let receipt: Value =
        serde_json::from_str(&raw).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let schema_warning = sanhedrin_schema_warning(&receipt);

    Ok(Json(serde_json::json!({
        "receipt": receipt,
        "stateDir": state_dir,
        "receiptPath": latest_path,
        "htmlPath": state_dir.join("latest.html"),
        "schemaWarning": schema_warning,
    })))
}

/// Return rolling Sanhedrin receipts, appeals, and fail-open counters.
pub async fn get_sanhedrin_telemetry(
    Query(params): Query<SanhedrinTelemetryParams>,
) -> Result<Json<Value>, StatusCode> {
    let state_dir = sanhedrin_state_dir();
    let days = params.days.unwrap_or(7).clamp(1, 90);
    let telemetry = tokio::task::spawn_blocking(move || build_sanhedrin_telemetry(state_dir, days))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
    Ok(Json(telemetry))
}

fn build_sanhedrin_telemetry(state_dir: PathBuf, days: i64) -> Result<Value, StatusCode> {
    let cutoff = Utc::now() - Duration::days(days);
    let receipts_dir = state_dir.join("receipts");
    let mut by_verdict = serde_json::Map::new();
    for verdict in ["PASS", "NOTE", "CAUTION", "VETO", "APPEALED"] {
        by_verdict.insert(verdict.to_string(), Value::from(0));
    }
    let mut by_class: BTreeMap<String, i64> = BTreeMap::new();
    let mut daily: BTreeMap<String, serde_json::Map<String, Value>> = BTreeMap::new();
    let mut total_runs = 0i64;
    let mut last_run_at: Option<DateTime<Utc>> = None;
    let mut truncated = false;

    if let Ok(entries) = bounded_receipt_entries(&receipts_dir, cutoff) {
        for path in entries {
            let Ok(raw) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(receipt) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };
            let Some(created_at) = parse_sanhedrin_timestamp(&receipt["createdAt"]) else {
                continue;
            };
            if created_at < cutoff {
                continue;
            }
            total_runs += 1;
            if last_run_at.map(|last| created_at > last).unwrap_or(true) {
                last_run_at = Some(created_at);
            }
            let verdict = receipt
                .get("verdictBar")
                .and_then(Value::as_str)
                .unwrap_or("NOTE")
                .to_ascii_uppercase();
            increment_json_counter(&mut by_verdict, &verdict);
            let day = created_at.date_naive().to_string();
            let bucket = daily.entry(day).or_insert_with(|| {
                let mut map = serde_json::Map::new();
                map.insert("date".to_string(), Value::String(String::new()));
                map.insert("total".to_string(), Value::from(0));
                map.insert("pass".to_string(), Value::from(0));
                map.insert("note".to_string(), Value::from(0));
                map.insert("caution".to_string(), Value::from(0));
                map.insert("veto".to_string(), Value::from(0));
                map.insert("appealed".to_string(), Value::from(0));
                map.insert("failOpen".to_string(), Value::from(0));
                map
            });
            increment_json_counter(bucket, "total");
            match verdict.as_str() {
                "PASS" => increment_json_counter(bucket, "pass"),
                "NOTE" => increment_json_counter(bucket, "note"),
                "CAUTION" => increment_json_counter(bucket, "caution"),
                "VETO" => increment_json_counter(bucket, "veto"),
                "APPEALED" => increment_json_counter(bucket, "appealed"),
                _ => increment_json_counter(bucket, "note"),
            }

            if let Some(claims) = receipt.get("claims").and_then(Value::as_array) {
                for claim in claims {
                    let class = known_sanhedrin_class(
                        claim
                            .get("class")
                            .and_then(Value::as_str)
                            .unwrap_or("UNKNOWN"),
                    );
                    *by_class.entry(class).or_insert(0) += 1;
                }
            }
        }
        truncated = total_runs >= 5_000;
    }

    let appeals = count_jsonl_since(&state_dir.join("appeals.jsonl"), cutoff, false);
    let fail_open = count_jsonl_since(&state_dir.join("fail-open.jsonl"), cutoff, true);
    for (date, bucket) in daily.iter_mut() {
        bucket.insert("date".to_string(), Value::String(date.clone()));
    }

    Ok(serde_json::json!({
        "days": days,
        "stateDir": state_dir,
        "totalRuns": total_runs,
        "byVerdict": by_verdict,
        "byClass": by_class,
        "appeals": appeals,
        "failOpen": fail_open,
        "truncated": truncated,
        "lastRunAt": last_run_at.map(|dt| dt.to_rfc3339()),
        "daily": daily.into_values().map(Value::Object).collect::<Vec<_>>(),
    }))
}

/// Record feedback that a Sanhedrin veto was stale, wrong, or too strict.
///
/// This intentionally does not promote, demote, suppress, edit, or delete any
/// memory. The hook reads this ledger and suppresses future same-fingerprint
/// vetoes, which keeps appeal training scoped to Sanhedrin behavior.
pub async fn appeal_sanhedrin(
    State(state): State<AppState>,
    Json(req): Json<SanhedrinAppealRequest>,
) -> Result<Json<Value>, StatusCode> {
    let reason = req.reason.trim().to_ascii_lowercase();
    if !matches!(reason.as_str(), "stale" | "wrong" | "too_strict") {
        return Err(StatusCode::BAD_REQUEST);
    }

    let state_dir = sanhedrin_state_dir();
    let latest_path = state_dir.join("latest.json");
    let raw = match fs::read_to_string(&latest_path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(StatusCode::NOT_FOUND);
        }
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };
    let mut receipt: Value = serde_json::from_str(&raw).map_err(|_| StatusCode::BAD_REQUEST)?;
    let original_receipt = receipt.clone();
    let note = req.note.unwrap_or_default();
    let receipt_id = receipt
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let receipt_id_ref = receipt_id.as_deref().ok_or(StatusCode::BAD_REQUEST)?;
    let _ = sanitize_receipt_id(receipt_id_ref)?;
    let expected_receipt_id = req.receipt_id.as_deref().ok_or(StatusCode::BAD_REQUEST)?;
    if expected_receipt_id != receipt_id_ref {
        return Err(StatusCode::CONFLICT);
    }
    if receipt
        .get("verdictBar")
        .and_then(Value::as_str)
        .map(|v| v != "VETO")
        .unwrap_or(true)
    {
        return Err(StatusCode::CONFLICT);
    }
    let claim = mark_sanhedrin_claim(&mut receipt, &reason, &note, req.claim_id.as_deref())?;

    let appeal = serde_json::json!({
        "timestamp": Utc::now().to_rfc3339(),
        "receiptId": receipt_id.as_deref(),
        "claimId": claim.get("id").and_then(Value::as_str),
        "claimFingerprint": claim.get("fingerprint").and_then(Value::as_str),
        "claim": claim.get("text").and_then(Value::as_str),
        "reason": &reason,
        "note": &note,
        "status": "active",
    });

    set_json_field(&mut receipt, "overall", "appealed");
    set_json_field(&mut receipt, "verdictBar", "APPEALED");
    set_json_field(&mut receipt, "summary", &format!("Appealed as {}.", reason));
    save_sanhedrin_receipt(&state_dir, &receipt)?;
    if let Err(err) = append_sanhedrin_appeal(&state_dir, &appeal) {
        let _ = save_sanhedrin_receipt(&state_dir, &original_receipt);
        return Err(err);
    }

    state.emit(VestigeEvent::HookVerdictRecorded {
        hook: "sanhedrin".to_string(),
        verdict: "APPEALED".to_string(),
        phase: "appeal".to_string(),
        reason: reason.clone(),
        receipt_id: receipt_id.clone(),
        timestamp: Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "appeal": appeal,
        "receipt": receipt,
    })))
}

fn sanhedrin_state_dir() -> PathBuf {
    std::env::var_os("VESTIGE_SANHEDRIN_STATE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".vestige/sanhedrin"))
        })
        .unwrap_or_else(|| PathBuf::from(".vestige/sanhedrin"))
}

fn sanhedrin_schema_warning(receipt: &Value) -> Option<String> {
    let schema = receipt.get("schema").and_then(Value::as_str).unwrap_or("");
    if schema.is_empty() || schema == "vestige.sanhedrin.receipt.v1" {
        None
    } else {
        Some(format!(
            "Unsupported Sanhedrin receipt schema '{}'; dashboard expects vestige.sanhedrin.receipt.v1",
            schema
        ))
    }
}

fn parse_sanhedrin_timestamp(value: &Value) -> Option<DateTime<Utc>> {
    let raw = value.as_str()?;
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .ok()
}

fn bounded_receipt_entries(
    receipts_dir: &FsPath,
    cutoff: DateTime<Utc>,
) -> Result<Vec<PathBuf>, StatusCode> {
    const MAX_RECEIPTS: usize = 5_000;
    const MAX_RECEIPT_BYTES: u64 = 256 * 1024;

    let mut entries: Vec<(Option<DateTime<Utc>>, PathBuf)> = Vec::new();
    let Ok(read_dir) = fs::read_dir(receipts_dir) else {
        return Ok(Vec::new());
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() > MAX_RECEIPT_BYTES
        {
            continue;
        }
        let modified = metadata.modified().ok().map(DateTime::<Utc>::from);
        if modified.map(|mtime| mtime < cutoff).unwrap_or(false) {
            continue;
        }
        entries.push((modified, path));
    }

    entries.sort_by(|(left_time, _), (right_time, _)| right_time.cmp(left_time));
    Ok(entries
        .into_iter()
        .take(MAX_RECEIPTS)
        .map(|(_, path)| path)
        .collect())
}

fn increment_json_counter(map: &mut serde_json::Map<String, Value>, key: &str) {
    let current = map.get(key).and_then(Value::as_i64).unwrap_or(0);
    map.insert(key.to_string(), Value::from(current + 1));
}

fn count_jsonl_since(path: &FsPath, cutoff: DateTime<Utc>, distinct_run: bool) -> i64 {
    const MAX_LEDGER_BYTES: u64 = 2 * 1024 * 1024;
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return 0;
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > MAX_LEDGER_BYTES
    {
        return 0;
    }
    let Ok(file) = fs::File::open(path) else {
        return 0;
    };
    let mut seen_runs = HashSet::new();
    let mut count = 0i64;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(item) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(timestamp) = parse_sanhedrin_timestamp(&item["timestamp"]) else {
            continue;
        };
        if timestamp < cutoff {
            continue;
        }
        if distinct_run
            && let Some(run_id) = item.get("runId").and_then(Value::as_str)
            && !seen_runs.insert(run_id.to_string())
        {
            continue;
        }
        count += 1;
    }
    count
}

fn known_sanhedrin_class(class: &str) -> String {
    match class {
        "receipt_lock"
        | "TECHNICAL"
        | "BIOGRAPHICAL"
        | "FINANCIAL"
        | "ACHIEVEMENT"
        | "TIMELINE"
        | "QUANTITATIVE"
        | "ATTRIBUTION"
        | "CAUSAL"
        | "COMPARATIVE"
        | "EXISTENTIAL"
        | "VAGUE-QUANTIFIER"
        | "UNVERIFIED-POSITIVE" => class.to_string(),
        _ => "OTHER".to_string(),
    }
}

fn ensure_sanhedrin_dirs(state_dir: &FsPath) -> Result<(), StatusCode> {
    fs::create_dir_all(state_dir.join("receipts")).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn mark_sanhedrin_claim(
    receipt: &mut Value,
    reason: &str,
    note: &str,
    claim_id: Option<&str>,
) -> Result<Value, StatusCode> {
    let claim_id = claim_id.ok_or(StatusCode::BAD_REQUEST)?;
    let claims = receipt
        .get_mut("claims")
        .and_then(Value::as_array_mut)
        .ok_or(StatusCode::BAD_REQUEST)?;

    if claims.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let selected = claims
        .iter()
        .position(|claim| claim.get("id").and_then(Value::as_str) == Some(claim_id))
        .ok_or(StatusCode::NOT_FOUND)?;

    if claims
        .get(selected)
        .and_then(|claim| claim.get("decision"))
        .and_then(Value::as_str)
        != Some("veto")
    {
        return Err(StatusCode::CONFLICT);
    }

    let claim = claims
        .get_mut(selected)
        .and_then(Value::as_object_mut)
        .ok_or(StatusCode::BAD_REQUEST)?;

    claim.insert(
        "decision".to_string(),
        Value::String("appealed".to_string()),
    );
    claim.insert(
        "evidence_state".to_string(),
        Value::String("appealed".to_string()),
    );
    claim.insert(
        "appeal".to_string(),
        serde_json::json!({
            "status": "appealed",
            "lastReason": reason,
            "note": note,
            "actions": ["stale", "wrong", "too_strict"],
        }),
    );

    Ok(Value::Object(claim.clone()))
}

fn set_json_field(receipt: &mut Value, key: &str, value: &str) {
    if let Some(obj) = receipt.as_object_mut() {
        obj.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn save_sanhedrin_receipt(state_dir: &FsPath, receipt: &Value) -> Result<(), StatusCode> {
    ensure_sanhedrin_dirs(state_dir)?;
    let rendered = render_sanhedrin_receipt_html(receipt);
    let pretty = serde_json::to_string_pretty(receipt).map_err(|_| StatusCode::BAD_REQUEST)?;
    let safe_id = receipt
        .get("id")
        .and_then(Value::as_str)
        .map(sanitize_receipt_id)
        .transpose()?;

    if let Some(safe_id) = safe_id {
        write_atomic(
            &state_dir.join("receipts").join(format!("{}.json", safe_id)),
            pretty.as_bytes(),
        )?;
        write_atomic(
            &state_dir.join("receipts").join(format!("{}.html", safe_id)),
            rendered.as_bytes(),
        )?;
    }

    write_atomic(&state_dir.join("latest.json"), pretty.as_bytes())?;
    write_atomic(&state_dir.join("latest.html"), rendered.as_bytes())?;
    Ok(())
}

fn append_sanhedrin_appeal(state_dir: &FsPath, appeal: &Value) -> Result<(), StatusCode> {
    ensure_sanhedrin_dirs(state_dir)?;
    let mut appeals = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_dir.join("appeals.jsonl"))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    writeln!(appeals, "{}", appeal).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn sanitize_receipt_id(id: &str) -> Result<&str, StatusCode> {
    if !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        Ok(id)
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}

fn write_atomic(path: &FsPath, bytes: &[u8]) -> Result<(), StatusCode> {
    let parent = path.parent().ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    fs::create_dir_all(parent).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tmp = path.with_extension(format!(
        "{}.tmp",
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::write(&tmp, bytes).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    fs::rename(&tmp, path).map_err(|_| {
        let _ = fs::remove_file(&tmp);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

fn render_sanhedrin_receipt_html(receipt: &Value) -> String {
    let verdict = escape_html(
        receipt
            .get("verdictBar")
            .and_then(Value::as_str)
            .unwrap_or("PASS"),
    );
    let summary = escape_html(receipt.get("summary").and_then(Value::as_str).unwrap_or(""));
    let mut claims_html = String::new();

    if let Some(claims) = receipt.get("claims").and_then(Value::as_array) {
        for claim in claims {
            let text = escape_html(claim.get("text").and_then(Value::as_str).unwrap_or(""));
            let decision = escape_html(claim.get("decision").and_then(Value::as_str).unwrap_or(""));
            let evidence_state = escape_html(
                claim
                    .get("evidence_state")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            let fix = escape_html(
                claim
                    .get("fix")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("No change required."),
            );
            let mut precedents = String::new();
            if let Some(items) = claim.get("precedent").and_then(Value::as_array) {
                for item in items {
                    let summary = item
                        .get("summary")
                        .and_then(Value::as_str)
                        .unwrap_or("Precedent recorded.");
                    precedents.push_str(&format!("<li>{}</li>", escape_html(summary)));
                }
            }
            claims_html.push_str(&format!(
                "<section class='claim'><div class='meta'>{} / {}</div><h2>{}</h2><p><strong>Fix:</strong> {}</p><p><strong>Appeal:</strong> stale | wrong | too_strict</p><ul>{}</ul></section>",
                decision, evidence_state, text, fix, precedents
            ));
        }
    }

    format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8"><title>Vestige Veto Receipt</title>
<style>
body{{margin:0;background:#050509;color:#e7e7f4;font-family:Inter,ui-sans-serif,system-ui;padding:32px}}
.bar{{display:inline-flex;gap:10px;align-items:center;border:1px solid #6d5dfc66;border-radius:8px;padding:10px 14px;background:#171528}}
.status{{font-weight:800;color:#fff;letter-spacing:.08em}}
.claim{{margin-top:18px;border:1px solid #ffffff1a;border-radius:8px;padding:16px;background:#0e0f18}}
.meta{{font-size:12px;color:#a8a8c8;text-transform:uppercase;letter-spacing:.08em}}
h1{{font-size:24px;margin:18px 0 4px}} h2{{font-size:16px;line-height:1.4}} p,li{{color:#c7c7dd}}
</style></head><body>
<div class="bar"><span>Verdict</span><span class="status">{}</span></div>
<h1>Veto Receipt</h1><p>{}</p>{}
</body></html>"#,
        verdict, summary, claims_html
    )
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Get system stats
pub async fn get_stats(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let stats = state
        .storage
        .get_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };

    Ok(Json(serde_json::json!({
        "totalMemories": stats.total_nodes,
        "dueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "averageStorageStrength": stats.average_storage_strength,
        "averageRetrievalStrength": stats.average_retrieval_strength,
        "withEmbeddings": stats.nodes_with_embeddings,
        "embeddingCoverage": embedding_coverage,
        "embeddingModel": stats.embedding_model,
        "oldestMemory": stats.oldest_memory.map(|dt| dt.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|dt| dt.to_rfc3339()),
    })))
}

#[derive(Debug, Deserialize)]
pub struct TimelineParams {
    pub days: Option<i64>,
    pub limit: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct ChangelogParams {
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<i32>,
}

/// Get timeline data
pub async fn get_timeline(
    State(state): State<AppState>,
    Query(params): Query<TimelineParams>,
) -> Result<Json<Value>, StatusCode> {
    let days = params.days.unwrap_or(7).clamp(1, 90);
    let limit = params.limit.unwrap_or(200).clamp(1, 500);

    let start = Utc::now() - Duration::days(days);
    let nodes = state
        .storage
        .query_time_range(Some(start), Some(Utc::now()), limit, None, None)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Group by day
    let mut by_day: std::collections::BTreeMap<String, Vec<Value>> =
        std::collections::BTreeMap::new();
    for node in &nodes {
        let date = node.created_at.format("%Y-%m-%d").to_string();
        let content_preview: String = {
            let preview: String = node.content.chars().take(100).collect();
            if preview.len() < node.content.len() {
                format!("{}...", preview)
            } else {
                preview
            }
        };
        by_day.entry(date).or_default().push(serde_json::json!({
            "id": node.id,
            "content": content_preview,
            "nodeType": node.node_type,
            "retentionStrength": node.retention_strength,
            "createdAt": node.created_at.to_rfc3339(),
        }));
    }

    let timeline: Vec<Value> = by_day
        .into_iter()
        .rev()
        .map(|(date, memories)| {
            serde_json::json!({
                "date": date,
                "count": memories.len(),
                "memories": memories,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "days": days,
        "totalMemories": nodes.len(),
        "timeline": timeline,
    })))
}

/// Recent cognitive events in the same envelope used by the WebSocket event
/// stream. The pulse hook polls this endpoint once per Claude wake, so keep it
/// cheap, bounded, and tolerant of empty history.
pub async fn get_changelog(
    State(state): State<AppState>,
    Query(params): Query<ChangelogParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(50).clamp(1, 100);
    let start = parse_changelog_bound(params.start.as_deref())?;
    let end = parse_changelog_bound(params.end.as_deref())?;
    let fetch_limit = if start.is_some() || end.is_some() {
        limit.saturating_mul(4)
    } else {
        limit
    };

    let mut events: Vec<(DateTime<Utc>, Value)> = Vec::new();

    let dreams = state
        .storage
        .get_dream_history(fetch_limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    for dream in dreams {
        if changelog_window_contains(dream.dreamed_at, start.as_ref(), end.as_ref()) {
            events.push((dream.dreamed_at, dream_changelog_event(&dream)));
        }
    }

    // Connections are currently persisted as graph edges rather than as audit
    // rows, so filter by created_at from the connection table.
    let connections = state
        .storage
        .get_all_connections()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    for conn in connections {
        if changelog_window_contains(conn.created_at, start.as_ref(), end.as_ref()) {
            events.push((conn.created_at, connection_changelog_event(&conn)));
        }
    }

    events.sort_by_key(|event| Reverse(event.0));
    events.truncate(limit as usize);
    let formatted_events: Vec<Value> = events.into_iter().map(|(_, event)| event).collect();
    let total_events = formatted_events.len();

    Ok(Json(serde_json::json!({
        "events": formatted_events,
        "totalEvents": total_events,
        "filter": {
            "start": start.as_ref().map(DateTime::to_rfc3339),
            "end": end.as_ref().map(DateTime::to_rfc3339),
            "limit": limit,
        },
    })))
}

fn parse_changelog_bound(raw: Option<&str>) -> Result<Option<DateTime<Utc>>, StatusCode> {
    match raw {
        Some(value) if !value.trim().is_empty() => DateTime::parse_from_rfc3339(value)
            .map(|dt| Some(dt.with_timezone(&Utc)))
            .map_err(|_| StatusCode::BAD_REQUEST),
        _ => Ok(None),
    }
}

fn changelog_window_contains(
    ts: DateTime<Utc>,
    start: Option<&DateTime<Utc>>,
    end: Option<&DateTime<Utc>>,
) -> bool {
    start.is_none_or(|s| ts >= *s) && end.is_none_or(|e| ts <= *e)
}

fn dream_changelog_event(dream: &vestige_core::DreamHistoryRecord) -> Value {
    serde_json::json!({
        "type": "DreamCompleted",
        "timestamp": dream.dreamed_at.to_rfc3339(),
        "data": {
            "memories_replayed": dream.memories_replayed,
            "connections_found": dream.connections_found,
            "connections_persisted": dream.connections_found,
            "insights_generated": dream.insights_generated,
            "duration_ms": dream.duration_ms,
            "timestamp": dream.dreamed_at.to_rfc3339(),
        },
    })
}

fn connection_changelog_event(conn: &vestige_core::ConnectionRecord) -> Value {
    serde_json::json!({
        "type": "ConnectionDiscovered",
        "timestamp": conn.created_at.to_rfc3339(),
        "data": {
            "source_id": &conn.source_id,
            "target_id": &conn.target_id,
            "connection_type": &conn.link_type,
            "weight": conn.strength,
            "timestamp": conn.created_at.to_rfc3339(),
        },
    })
}

/// Health check
pub async fn health_check(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let stats = state
        .storage
        .get_stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    Ok(Json(serde_json::json!({
        "status": status,
        "totalMemories": stats.total_nodes,
        "averageRetention": stats.average_retention,
        "version": env!("CARGO_PKG_VERSION"),
    })))
}

// ============================================================================
// MEMORY GRAPH
// ============================================================================

/// Redirect legacy graph to SvelteKit dashboard graph page
pub async fn serve_graph() -> Redirect {
    Redirect::permanent("/dashboard/graph")
}

#[derive(Debug, Deserialize)]
pub struct GraphParams {
    pub query: Option<String>,
    pub center_id: Option<String>,
    pub depth: Option<u32>,
    pub max_nodes: Option<usize>,
    /// How to choose the default center when neither `query` nor `center_id`
    /// is provided. "recent" (default) uses the newest memory — matches
    /// what users actually expect ("show me my recent stuff"). "connected"
    /// uses the most-connected memory for a richer initial subgraph; used
    /// to be the default but ended up clustering on historical hotspots
    /// and hiding fresh memories that hadn't accumulated edges yet.
    /// Unknown values fall back to "recent".
    pub sort: Option<String>,
}

/// Which memory to center the default subgraph on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphSort {
    Recent,
    Connected,
}

impl GraphSort {
    fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::to_ascii_lowercase).as_deref() {
            Some("connected") => Self::Connected,
            _ => Self::Recent,
        }
    }
}

/// Get memory graph data (nodes + edges with layout positions)
pub async fn get_graph(
    State(state): State<AppState>,
    Query(params): Query<GraphParams>,
) -> Result<Json<Value>, StatusCode> {
    let depth = params.depth.unwrap_or(2).clamp(1, 3);
    let max_nodes = params.max_nodes.unwrap_or(50).clamp(1, 200);
    let sort = GraphSort::parse(params.sort.as_deref());

    // Determine center node
    let explicit_center = params.center_id.is_some() || params.query.is_some();
    let center_id = if let Some(ref id) = params.center_id {
        id.clone()
    } else if let Some(ref query) = params.query {
        let results = state
            .storage
            .search(query, 1)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        results
            .first()
            .map(|n| n.id.clone())
            .ok_or(StatusCode::NOT_FOUND)?
    } else {
        default_center_id(&state.storage, sort)?
    };

    // Get subgraph
    let (mut nodes, mut edges) = state
        .storage
        .get_memory_subgraph(&center_id, depth, max_nodes)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Default-load fallback: if the newest memory is isolated (1 node, 0 edges),
    // silently re-resolve via Connected so the user sees the densest cluster
    // instead of a lonely orb. Explicit query/center_id requests are honored
    // as-is — the user asked for that specific subgraph.
    let mut center_id = center_id;
    if !explicit_center
        && sort == GraphSort::Recent
        && nodes.len() <= 1
        && edges.is_empty()
        && let Ok(fallback) = default_center_id(&state.storage, GraphSort::Connected)
        && fallback != center_id
        && let Ok((n2, e2)) = state
            .storage
            .get_memory_subgraph(&fallback, depth, max_nodes)
        && n2.len() > nodes.len()
    {
        center_id = fallback;
        nodes = n2;
        edges = e2;
    }

    if nodes.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Build nodes JSON with timestamps for recency calculation
    let nodes_json: Vec<Value> = nodes
        .iter()
        .map(|n| {
            let label = if n.content.chars().count() > 80 {
                format!("{}...", n.content.chars().take(77).collect::<String>())
            } else {
                n.content.clone()
            };
            serde_json::json!({
                "id": n.id,
                "label": label,
                "type": n.node_type,
                "retention": n.retention_strength,
                "tags": n.tags,
                "createdAt": n.created_at.to_rfc3339(),
                "updatedAt": n.updated_at.to_rfc3339(),
                "isCenter": n.id == center_id,
            })
        })
        .collect();

    let edges_json: Vec<Value> = edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "source": e.source_id,
                "target": e.target_id,
                "weight": e.strength,
                "type": e.link_type,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
        "center_id": center_id,
        "depth": depth,
        "nodeCount": nodes.len(),
        "edgeCount": edges.len(),
    })))
}

/// Pick the default subgraph center when neither `query` nor `center_id`
/// was provided. Factored out so both the route handler and unit tests can
/// exercise the same branching (recent vs connected + empty-db fallback)
/// without spinning up a full axum server.
fn default_center_id(
    storage: &std::sync::Arc<vestige_core::Storage>,
    sort: GraphSort,
) -> Result<String, StatusCode> {
    match sort {
        GraphSort::Recent => {
            let recent = storage
                .get_all_nodes(1, 0)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            recent
                .first()
                .map(|n| n.id.clone())
                .ok_or(StatusCode::NOT_FOUND)
        }
        GraphSort::Connected => {
            let most_connected = storage
                .get_most_connected_memory()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            if let Some(id) = most_connected {
                Ok(id)
            } else {
                // Nothing connected yet (fresh DB, or every node is isolated) —
                // fall through to the newest memory so the user still sees
                // SOMETHING rather than a 404.
                let recent = storage
                    .get_all_nodes(1, 0)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                recent
                    .first()
                    .map(|n| n.id.clone())
                    .ok_or(StatusCode::NOT_FOUND)
            }
        }
    }
}

// ============================================================================
// SEARCH (dedicated endpoint)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<i32>,
    pub min_retention: Option<f64>,
}

/// Search memories with hybrid search
pub async fn search_memories(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(20).clamp(1, 100);
    let start = std::time::Instant::now();

    let results = state
        .storage
        .hybrid_search(&params.q, limit, 0.3, 0.7)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let result_ids: Vec<String> = results.iter().map(|r| r.node.id.clone()).collect();

    // Emit search event
    state.emit(VestigeEvent::SearchPerformed {
        query: params.q.clone(),
        result_count: results.len(),
        result_ids: result_ids.clone(),
        duration_ms,
        timestamp: Utc::now(),
    });

    let formatted: Vec<Value> = results
        .into_iter()
        .filter(|r| {
            params
                .min_retention
                .is_none_or(|min| r.node.retention_strength >= min)
        })
        .map(|r| {
            serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
                "combinedScore": r.combined_score,
                "createdAt": r.node.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "query": params.q,
        "total": formatted.len(),
        "durationMs": duration_ms,
        "results": formatted,
    })))
}

// ============================================================================
// COGNITIVE OPERATIONS (v2.0)
// ============================================================================

/// Trigger a dream cycle via CognitiveEngine
pub async fn trigger_dream(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let cognitive = state
        .cognitive
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let start = std::time::Instant::now();
    let memory_count: usize = 50;

    // Load memories for dreaming
    let all_nodes = state
        .storage
        .get_all_nodes(memory_count as i32, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if all_nodes.len() < 5 {
        return Ok(Json(serde_json::json!({
            "status": "insufficient_memories",
            "message": format!("Need at least 5 memories. Current: {}", all_nodes.len()),
        })));
    }

    // Emit start event
    state.emit(VestigeEvent::DreamStarted {
        memory_count: all_nodes.len(),
        timestamp: Utc::now(),
    });

    // Build dream memories
    let dream_memories: Vec<vestige_core::DreamMemory> = all_nodes
        .iter()
        .map(|n| vestige_core::DreamMemory {
            id: n.id.clone(),
            content: n.content.clone(),
            embedding: state.storage.get_node_embedding(&n.id).ok().flatten(),
            tags: n.tags.clone(),
            created_at: n.created_at,
            access_count: n.reps as u32,
        })
        .collect();

    // Run dream through CognitiveEngine
    let cog = cognitive.lock().await;
    let (dream_result, new_connections) = cog.dreamer.dream_with_connections(&dream_memories).await;
    let insights = cog.dreamer.synthesize_insights(&dream_memories);
    drop(cog);

    // Persist new connections
    let mut connections_persisted = 0u64;
    let now = Utc::now();
    for conn in &new_connections {
        let link_type = match conn.connection_type {
            vestige_core::DiscoveredConnectionType::Semantic => "semantic",
            vestige_core::DiscoveredConnectionType::SharedConcept => "shared_concepts",
            vestige_core::DiscoveredConnectionType::Temporal => "temporal",
            vestige_core::DiscoveredConnectionType::Complementary => "complementary",
            vestige_core::DiscoveredConnectionType::CausalChain => "causal",
        };
        let record = vestige_core::ConnectionRecord {
            source_id: conn.from_id.clone(),
            target_id: conn.to_id.clone(),
            strength: conn.similarity,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 1,
        };
        if state.storage.save_connection(&record).is_ok() {
            connections_persisted += 1;
        }

        // Emit connection events
        state.emit(VestigeEvent::ConnectionDiscovered {
            source_id: conn.from_id.clone(),
            target_id: conn.to_id.clone(),
            connection_type: link_type.to_string(),
            weight: conn.similarity,
            timestamp: now,
        });
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    let completed_at = Utc::now();
    let insights_generated = insights.len();

    if let Err(e) = state
        .storage
        .save_dream_history(&vestige_core::DreamHistoryRecord {
            dreamed_at: completed_at,
            duration_ms: duration_ms as i64,
            memories_replayed: dream_memories.len() as i32,
            connections_found: connections_persisted as i32,
            insights_generated: insights_generated as i32,
            memories_strengthened: dream_result.memories_strengthened as i32,
            memories_compressed: dream_result.memories_compressed as i32,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        })
    {
        tracing::warn!("Failed to persist dashboard dream history: {}", e);
    }

    // Emit completion event
    state.emit(VestigeEvent::DreamCompleted {
        memories_replayed: dream_memories.len(),
        connections_found: connections_persisted as usize,
        insights_generated,
        duration_ms,
        timestamp: completed_at,
    });

    Ok(Json(serde_json::json!({
        "status": "dreamed",
        "memoriesReplayed": dream_memories.len(),
        "connectionsPersisted": connections_persisted,
        "insights": insights.iter().map(|i| serde_json::json!({
            "type": format!("{:?}", i.insight_type),
            "insight": i.insight,
            "sourceMemories": i.source_memories,
            "confidence": i.confidence,
            "noveltyScore": i.novelty_score,
        })).collect::<Vec<Value>>(),
        "stats": {
            "newConnectionsFound": dream_result.new_connections_found,
            "connectionsPersisted": connections_persisted,
            "memoriesStrengthened": dream_result.memories_strengthened,
            "memoriesCompressed": dream_result.memories_compressed,
            "insightsGenerated": dream_result.insights_generated.len(),
            "durationMs": duration_ms,
        }
    })))
}

#[derive(Debug, Deserialize)]
pub struct ExploreRequest {
    pub from_id: String,
    pub to_id: Option<String>,
    pub action: Option<String>, // "associations", "chains", "bridges"
    pub limit: Option<usize>,
}

/// Explore connections between memories
pub async fn explore_connections(
    State(state): State<AppState>,
    Json(req): Json<ExploreRequest>,
) -> Result<Json<Value>, StatusCode> {
    let action = req.action.as_deref().unwrap_or("associations");
    let limit = req.limit.unwrap_or(10).clamp(1, 50);

    match action {
        "associations" => {
            // Get the source memory content for similarity search
            let source_node = state
                .storage
                .get_node(&req.from_id)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .ok_or(StatusCode::NOT_FOUND)?;

            // Use hybrid search with source content to find associated memories
            let results = state
                .storage
                .hybrid_search(&source_node.content, limit as i32, 0.3, 0.7)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let formatted: Vec<Value> = results
                .iter()
                .filter(|r| r.node.id != req.from_id) // Exclude self
                .map(|r| {
                    serde_json::json!({
                        "id": r.node.id,
                        "content": r.node.content,
                        "nodeType": r.node.node_type,
                        "score": r.combined_score,
                        "retention": r.node.retention_strength,
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({
                "action": "associations",
                "fromId": req.from_id,
                "results": formatted,
            })))
        }
        "chains" | "bridges" => {
            let to_id = req.to_id.as_deref().ok_or(StatusCode::BAD_REQUEST)?;

            let (nodes, edges) = state
                .storage
                .get_memory_subgraph(&req.from_id, 2, limit)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let nodes_json: Vec<Value> = nodes
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "id": n.id,
                        "content": n.content.chars().take(100).collect::<String>(),
                        "nodeType": n.node_type,
                        "retention": n.retention_strength,
                    })
                })
                .collect();

            let edges_json: Vec<Value> = edges
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "source": e.source_id,
                        "target": e.target_id,
                        "weight": e.strength,
                        "type": e.link_type,
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({
                "action": action,
                "fromId": req.from_id,
                "toId": to_id,
                "nodes": nodes_json,
                "edges": edges_json,
            })))
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

/// Predict which memories will be needed
pub async fn predict_memories(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    // Get recent memories as predictions based on activity
    let recent = state
        .storage
        .get_all_nodes(10, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let predictions: Vec<Value> = recent
        .iter()
        .map(|n| {
            serde_json::json!({
                "id": n.id,
                "content": n.content.chars().take(100).collect::<String>(),
                "nodeType": n.node_type,
                "retention": n.retention_strength,
                "predictedNeed": "high",
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "predictions": predictions,
        "basedOn": "recent_activity",
    })))
}

#[derive(Debug, Deserialize)]
pub struct ImportanceRequest {
    pub content: String,
}

/// Score content importance using 4-channel model
pub async fn score_importance(
    State(state): State<AppState>,
    Json(req): Json<ImportanceRequest>,
) -> Result<Json<Value>, StatusCode> {
    if let Some(ref cognitive) = state.cognitive {
        let context = vestige_core::ImportanceContext::current();
        let cog = cognitive.lock().await;
        let score = cog
            .importance_signals
            .compute_importance(&req.content, &context);
        drop(cog);

        let composite = score.composite;
        let novelty = score.novelty;
        let arousal = score.arousal;
        let reward = score.reward;
        let attention = score.attention;

        state.emit(VestigeEvent::ImportanceScored {
            memory_id: None, // /api/importance scores arbitrary content, not a stored memory
            content_preview: req.content.chars().take(80).collect(),
            composite_score: composite,
            novelty,
            arousal,
            reward,
            attention,
            timestamp: Utc::now(),
        });

        Ok(Json(serde_json::json!({
            "composite": composite,
            "channels": {
                "novelty": novelty,
                "arousal": arousal,
                "reward": reward,
                "attention": attention,
            },
            "recommendation": if composite > 0.6 { "save" } else { "skip" },
        })))
    } else {
        // Fallback: basic heuristic scoring
        let word_count = req.content.split_whitespace().count();
        let has_code = req.content.contains("```") || req.content.contains("fn ");
        let composite = if has_code {
            0.7
        } else {
            (word_count as f64 / 100.0).min(0.8)
        };

        Ok(Json(serde_json::json!({
            "composite": composite,
            "channels": {
                "novelty": composite,
                "arousal": 0.5,
                "reward": 0.5,
                "attention": composite,
            },
            "recommendation": if composite > 0.6 { "save" } else { "skip" },
        })))
    }
}

/// Trigger consolidation
pub async fn trigger_consolidation(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    state.emit(VestigeEvent::ConsolidationStarted {
        timestamp: Utc::now(),
    });

    let start = std::time::Instant::now();

    let result = state
        .storage
        .run_consolidation()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let duration_ms = start.elapsed().as_millis() as u64;

    state.emit(VestigeEvent::ConsolidationCompleted {
        nodes_processed: result.nodes_processed as usize,
        decay_applied: result.decay_applied as usize,
        embeddings_generated: result.embeddings_generated as usize,
        duration_ms,
        timestamp: Utc::now(),
    });

    Ok(Json(serde_json::json!({
        "nodesProcessed": result.nodes_processed,
        "decayApplied": result.decay_applied,
        "embeddingsGenerated": result.embeddings_generated,
        "duplicatesMerged": result.duplicates_merged,
        "activationsComputed": result.activations_computed,
        "durationMs": duration_ms,
    })))
}

/// Get retention distribution (for histogram visualization)
pub async fn retention_distribution(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    // Cap at 1000 to prevent excessive memory usage on large databases
    let nodes = state
        .storage
        .get_all_nodes(1000, 0)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Build distribution buckets
    let mut buckets = [0u32; 10]; // 0-10%, 10-20%, ..., 90-100%
    let mut by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut endangered = Vec::new();

    for node in &nodes {
        let bucket = ((node.retention_strength * 10.0).floor() as usize).min(9);
        buckets[bucket] += 1;
        *by_type.entry(node.node_type.clone()).or_default() += 1;

        // Endangered: retention below 30%
        if node.retention_strength < 0.3 {
            endangered.push(serde_json::json!({
                "id": node.id,
                "content": node.content.chars().take(60).collect::<String>(),
                "retention": node.retention_strength,
                "nodeType": node.node_type,
            }));
        }
    }

    let distribution: Vec<Value> = buckets
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            serde_json::json!({
                "range": format!("{}-{}%", i * 10, (i + 1) * 10),
                "count": count,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "distribution": distribution,
        "byType": by_type,
        "endangered": endangered,
        "total": nodes.len(),
    })))
}

// ============================================================================
// INTENTIONS (v2.0)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IntentionListParams {
    pub status: Option<String>,
}

/// List intentions
pub async fn list_intentions(
    State(state): State<AppState>,
    Query(params): Query<IntentionListParams>,
) -> Result<Json<Value>, StatusCode> {
    let status_filter = params.status.unwrap_or_else(|| "active".to_string());

    let intentions = if status_filter == "all" {
        // Get all statuses
        let mut all = state.storage.get_active_intentions().unwrap_or_default();
        all.extend(
            state
                .storage
                .get_intentions_by_status("fulfilled")
                .unwrap_or_default(),
        );
        all.extend(
            state
                .storage
                .get_intentions_by_status("cancelled")
                .unwrap_or_default(),
        );
        all.extend(
            state
                .storage
                .get_intentions_by_status("snoozed")
                .unwrap_or_default(),
        );
        all
    } else if status_filter == "active" {
        state
            .storage
            .get_active_intentions()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        state
            .storage
            .get_intentions_by_status(&status_filter)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let count = intentions.len();
    Ok(Json(serde_json::json!({
        "intentions": intentions,
        "total": count,
        "filter": status_filter,
    })))
}

// ============================================================================
// DEEP REFERENCE (Reasoning Theater, v2.0.8)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DeepReferenceBody {
    pub query: String,
    pub depth: Option<i32>,
}

/// Run the 8-stage deep_reference cognitive pipeline over HTTP.
///
/// Wraps the existing `crate::tools::cross_reference::execute` tool so the
/// dashboard can surface the same reasoning chain the MCP clients see. Emits
/// a `DeepReferenceCompleted` event with primary + supporting + contradicting
/// memory IDs so Graph3D can camera-glide, pulse evidence nodes, and draw
/// contradiction arcs in the 3D scene.
pub async fn deep_reference_query(
    State(state): State<AppState>,
    Json(body): Json<DeepReferenceBody>,
) -> Result<Json<Value>, StatusCode> {
    let cognitive = state
        .cognitive
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    if body.query.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let args = serde_json::json!({
        "query": body.query.clone(),
        "depth": body.depth.unwrap_or(20).clamp(5, 50),
    });

    let start = std::time::Instant::now();
    let response = crate::tools::cross_reference::execute(&state.storage, cognitive, Some(args))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let duration_ms = start.elapsed().as_millis() as u64;

    // Pull evidence IDs out for the WebSocket event so Graph3D can glide,
    // pulse, and arc. We intentionally read from the serialized JSON rather
    // than re-running the pipeline — whatever the tool decided is what the
    // Theater visualizes.
    let primary_id = response
        .get("recommended")
        .and_then(|r| r.get("memory_id"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let supporting_ids: Vec<String> = response
        .get("evidence")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let role = e.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    if role == "supporting" || role == "primary" {
                        e.get("id").and_then(|v| v.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let contradicting_ids: Vec<String> = response
        .get("evidence")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let role = e.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    if role == "contradicting" {
                        e.get("id").and_then(|v| v.as_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let contradiction_pairs: Vec<(String, String)> = response
        .get("contradictions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let a = c.get("a_id").and_then(|v| v.as_str())?.to_string();
                    let b = c.get("b_id").and_then(|v| v.as_str())?.to_string();
                    Some((a, b))
                })
                .collect()
        })
        .unwrap_or_default();

    let memories_analyzed = response
        .get("memoriesAnalyzed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let confidence = response
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let intent = response
        .get("intent")
        .and_then(|v| v.as_str())
        .unwrap_or("Synthesis")
        .to_string();

    let status = response
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    state.emit(VestigeEvent::DeepReferenceCompleted {
        query: body.query,
        intent,
        status,
        confidence,
        primary_id,
        supporting_ids,
        contradicting_ids,
        contradiction_pairs,
        memories_analyzed,
        duration_ms,
        timestamp: Utc::now(),
    });

    Ok(Json(response))
}

// ============================================================================
// AGENT BLACK BOX (v2.2) — replayable agent-run traces
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TraceListParams {
    pub limit: Option<usize>,
    /// Optional run filter — receipts/traces scoped to one run (B5).
    pub run: Option<String>,
}

/// List recent agent runs (newest activity first) for the Black Box run picker.
pub async fn list_traces(
    State(state): State<AppState>,
    Query(params): Query<TraceListParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(50).clamp(1, 500);
    let runs = state
        .storage
        .list_agent_runs(limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let runs_json: Vec<Value> = runs
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "runId": r.run_id,
                "firstTool": r.first_tool,
                "eventCount": r.event_count,
                "retrievedCount": r.retrieved_count,
                "suppressedCount": r.suppressed_count,
                "writeCount": r.write_count,
                "vetoCount": r.veto_count,
                "startedAt": r.started_at,
                "lastAt": r.last_at,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "total": runs_json.len(),
        "runs": runs_json,
    })))
}

/// Fetch the full event timeline for one run — the black-box replay payload.
pub async fn get_trace(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let events = state
        .storage
        .get_trace(&run_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if events.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    let summary = state.storage.get_agent_run(&run_id).ok().flatten();
    Ok(Json(serde_json::json!({
        "runId": run_id,
        "summary": summary.map(|s| serde_json::json!({
            "firstTool": s.first_tool,
            "eventCount": s.event_count,
            "retrievedCount": s.retrieved_count,
            "suppressedCount": s.suppressed_count,
            "writeCount": s.write_count,
            "vetoCount": s.veto_count,
            "startedAt": s.started_at,
            "lastAt": s.last_at,
        })),
        "events": events,
    })))
}

/// Export a run as a downloadable `.vestige-trace.json` artifact.
pub async fn export_trace(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<([(axum::http::HeaderName, String); 2], Json<Value>), StatusCode> {
    let events = state
        .storage
        .get_trace(&run_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if events.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    let summary = state.storage.get_agent_run(&run_id).ok().flatten();
    let body = serde_json::json!({
        "format": "vestige-trace",
        "version": 1,
        "runId": run_id,
        "exportedAt": Utc::now().to_rfc3339(),
        "summary": summary.map(|s| serde_json::json!({
            "firstTool": s.first_tool,
            "eventCount": s.event_count,
            "retrievedCount": s.retrieved_count,
            "suppressedCount": s.suppressed_count,
            "writeCount": s.write_count,
            "vetoCount": s.veto_count,
            "startedAt": s.started_at,
            "lastAt": s.last_at,
        })),
        "events": events,
    });
    // B7: sanitize the run_id before putting it in the download filename so a
    // crafted run_id (quotes, path separators, control chars) can't break the
    // Content-Disposition header or the filename. Falls back to "trace".
    let safe: String = run_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let safe = if safe.trim_matches('_').is_empty() {
        "trace".to_string()
    } else {
        safe
    };
    let headers = [
        (
            axum::http::header::CONTENT_TYPE,
            "application/json".to_string(),
        ),
        (
            axum::http::header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{safe}.vestige-trace.json\""),
        ),
    ];
    Ok((headers, Json(body)))
}

// ============================================================================
// MEMORY RECEIPTS (v2.2)
// ============================================================================

/// List recent retrieval receipts.
pub async fn list_receipts(
    State(state): State<AppState>,
    Query(params): Query<TraceListParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(50).clamp(1, 500);
    // B5: when a run is given, scope to that run's receipts so the Black Box
    // panel shows only receipts that actually belong to the selected run.
    let receipts = match params.run.as_deref().filter(|r| !r.is_empty()) {
        Some(run_id) => state.storage.list_receipts_for_run(run_id, limit),
        None => state.storage.list_receipts(limit),
    }
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "total": receipts.len(),
        "receipts": receipts,
    })))
}

/// Fetch one receipt by id — the payload behind "Open receipt in Cinema".
pub async fn get_receipt(
    State(state): State<AppState>,
    Path(receipt_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let receipt = state
        .storage
        .get_receipt(&receipt_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::to_value(receipt).unwrap_or_default()))
}

// ============================================================================
// MEMORY PRs (v2.2) — risk-gated brain-change review queue
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct MemoryPrListParams {
    pub status: Option<String>,
    pub limit: Option<usize>,
}

/// List Memory PRs, optionally filtered by status.
pub async fn list_memory_prs(
    State(state): State<AppState>,
    Query(params): Query<MemoryPrListParams>,
) -> Result<Json<Value>, StatusCode> {
    let limit = params.limit.unwrap_or(100).clamp(1, 500);
    let status = params.status.as_deref().and_then(|s| {
        serde_json::from_value::<vestige_core::MemoryPrStatus>(serde_json::Value::String(
            s.to_string(),
        ))
        .ok()
    });
    let prs = state
        .storage
        .list_memory_prs(status, limit)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let pending = state.storage.count_pending_memory_prs().unwrap_or(0);
    Ok(Json(serde_json::json!({
        "total": prs.len(),
        "pendingCount": pending,
        "mode": read_review_mode(&state).as_str(),
        "prs": prs,
    })))
}

/// Fetch one Memory PR by id.
pub async fn get_memory_pr(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let pr = state
        .storage
        .get_memory_pr(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::to_value(pr).unwrap_or_default()))
}

/// Act on a Memory PR: promote / merge / supersede / quarantine / forget /
/// ask_agent_why. `ask_agent_why` is read-only and returns the risk signals.
pub async fn act_on_memory_pr(
    State(state): State<AppState>,
    Path((id, action)): Path<(String, String)>,
) -> Result<Json<Value>, StatusCode> {
    let action = vestige_core::MemoryPrAction::from_label(&action)
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Ask Agent Why is read-only — return the self-explaining signals.
    if matches!(action, vestige_core::MemoryPrAction::AskAgentWhy) {
        let pr = state
            .storage
            .get_memory_pr(&id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .ok_or(StatusCode::NOT_FOUND)?;
        return Ok(Json(serde_json::json!({
            "id": pr.id,
            "kind": pr.kind.as_str(),
            "title": pr.title,
            "why": pr.signals,
            "explanation": "These are the risk signals that opened this Memory PR.",
        })));
    }

    let decided = state
        .storage
        .decide_memory_pr(&id, action)
        .map_err(|_| StatusCode::NOT_FOUND)?;

    // B1: an accept action (promote/merge/supersede) must RELEASE the subject
    // memory from quarantine — gate_writes suppressed it, so deciding the PR
    // without un-suppressing would leave it "promoted" yet still held out of
    // retrieval. Forget/Quarantine intentionally keep it suppressed.
    let mut released = false;
    if action.releases_memory()
        && let Some(subject_id) = decided.subject_id.as_deref()
    {
        // Use the UNCONDITIONAL quarantine release, not reverse_suppression:
        // approving a PR must restore the memory even if reviewed days later,
        // past the active-forgetting labile window (the C1 fix).
        match state.storage.release_quarantine(subject_id) {
            Ok(node) => {
                released = true;
                state.emit(VestigeEvent::MemoryUnsuppressed {
                    id: node.id.clone(),
                    remaining_count: node.suppression_count,
                    timestamp: Utc::now(),
                });
            }
            Err(e) => {
                // Best-effort: the PR is decided regardless, but surface the
                // failure so a stuck-suppressed memory isn't silent.
                tracing::warn!(
                    "memory PR {} {}d but failed to release subject {}: {}",
                    id,
                    action_label(action),
                    subject_id,
                    e
                );
            }
        }
    }

    state.emit(VestigeEvent::MemoryPrDecided {
        id: decided.id.clone(),
        decision: decided
            .decision
            .and_then(|d| serde_json::to_value(d).ok())
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default(),
        status: decided.status.as_str().to_string(),
        timestamp: Utc::now(),
    });

    let mut out = serde_json::to_value(&decided).unwrap_or_default();
    if let Some(obj) = out.as_object_mut() {
        obj.insert("subjectReleased".to_string(), serde_json::json!(released));
    }
    Ok(Json(out))
}

/// Short label for a Memory PR action, for log lines.
fn action_label(action: vestige_core::MemoryPrAction) -> &'static str {
    use vestige_core::MemoryPrAction::*;
    match action {
        Promote => "promote",
        Merge => "merge",
        Supersede => "supersede",
        Quarantine => "quarantine",
        Forget => "forget",
        AskAgentWhy => "ask_agent_why",
    }
}

#[derive(Debug, Deserialize)]
pub struct ReviewModeBody {
    pub mode: String,
}

/// Get the current review mode (fast / risk_gated / paranoid).
pub async fn get_review_mode(State(state): State<AppState>) -> Json<Value> {
    let mode = read_review_mode(&state);
    Json(serde_json::json!({
        "mode": mode.as_str(),
        "pendingCount": state.storage.count_pending_memory_prs().unwrap_or(0),
    }))
}

/// Set the review mode. Persisted to a small JSON file in the data dir so it
/// survives restarts (local-first, no extra config service).
pub async fn set_review_mode(
    State(state): State<AppState>,
    Json(body): Json<ReviewModeBody>,
) -> Result<Json<Value>, StatusCode> {
    let mode = vestige_core::ReviewMode::from_label(&body.mode);
    let path = review_mode_path(&state);
    let payload = serde_json::json!({ "mode": mode.as_str() });
    // B7: atomic write (temp + rename) so a concurrent read can never see a
    // partially-written / corrupt review_mode.json, reusing the same helper the
    // Sanhedrin receipt path uses.
    write_atomic(&path, &serde_json::to_vec_pretty(&payload).unwrap_or_default())?;
    Ok(Json(serde_json::json!({ "mode": mode.as_str() })))
}

/// Path to the persisted review-mode file.
fn review_mode_path(state: &AppState) -> PathBuf {
    state.storage.data_dir().join("review_mode.json")
}

/// Read the persisted review mode, defaulting to RiskGated.
pub fn read_review_mode(state: &AppState) -> vestige_core::ReviewMode {
    let path = review_mode_path(state);
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.get("mode").and_then(|m| m.as_str()).map(String::from))
        .map(|s| vestige_core::ReviewMode::from_label(&s))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::Arc;
    use tempfile::tempdir;
    use vestige_core::memory::IngestInput;
    use vestige_core::{ConnectionRecord, DreamHistoryRecord, Storage};

    #[test]
    fn graph_sort_parse_defaults_to_recent() {
        assert_eq!(GraphSort::parse(None), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("recent")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("RECENT")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("Recent")), GraphSort::Recent);
        assert_eq!(GraphSort::parse(Some("garbage")), GraphSort::Recent);
    }

    #[test]
    fn graph_sort_parse_accepts_connected_case_insensitive() {
        assert_eq!(GraphSort::parse(Some("connected")), GraphSort::Connected);
        assert_eq!(GraphSort::parse(Some("CONNECTED")), GraphSort::Connected);
        assert_eq!(GraphSort::parse(Some("Connected")), GraphSort::Connected);
    }

    #[test]
    fn changelog_dream_event_uses_pulse_compatible_shape() {
        let now = Utc::now();
        let event = dream_changelog_event(&DreamHistoryRecord {
            dreamed_at: now,
            duration_ms: 1234,
            memories_replayed: 12,
            connections_found: 3,
            insights_generated: 2,
            memories_strengthened: 0,
            memories_compressed: 0,
            phase_nrem1_ms: None,
            phase_nrem3_ms: None,
            phase_rem_ms: None,
            phase_integration_ms: None,
            summaries_generated: None,
            emotional_memories_processed: None,
            creative_connections_found: None,
        });

        assert_eq!(event["type"], "DreamCompleted");
        assert_eq!(event["data"]["insights_generated"], 2);
        assert_eq!(event["data"]["connections_persisted"], 3);
        assert_eq!(event["data"]["timestamp"], now.to_rfc3339());
    }

    #[test]
    fn changelog_connection_event_uses_pulse_compatible_shape() {
        let now = Utc::now();
        let event = connection_changelog_event(&ConnectionRecord {
            source_id: "source-memory".to_string(),
            target_id: "target-memory".to_string(),
            strength: 0.82,
            link_type: "semantic".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 1,
        });

        assert_eq!(event["type"], "ConnectionDiscovered");
        assert_eq!(event["data"]["source_id"], "source-memory");
        assert_eq!(event["data"]["target_id"], "target-memory");
        assert_eq!(event["data"]["connection_type"], "semantic");
    }

    fn seed_storage() -> (tempfile::TempDir, Arc<Storage>) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = Arc::new(Storage::new(Some(db_path)).unwrap());
        (dir, storage)
    }

    fn ingest(storage: &Storage, content: &str) -> String {
        let node = storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        node.id
    }

    #[test]
    fn default_center_id_recent_returns_newest_node() {
        let (_dir, storage) = seed_storage();
        ingest(&storage, "first");
        ingest(&storage, "second");
        let newest = ingest(&storage, "third");

        let center = default_center_id(&storage, GraphSort::Recent).unwrap();
        assert_eq!(
            center, newest,
            "Recent mode should pick the newest ingested memory"
        );
    }

    fn link(storage: &Storage, source: &str, target: &str) {
        let now = Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 0.9,
                link_type: "semantic".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    #[test]
    fn default_center_id_connected_prefers_hub_over_newest() {
        let (_dir, storage) = seed_storage();
        let hub = ingest(&storage, "hub node");
        let spoke_a = ingest(&storage, "spoke A");
        let spoke_b = ingest(&storage, "spoke B");
        let spoke_c = ingest(&storage, "spoke C");
        // Wire the spokes into `hub` so it has the most connections. Leave
        // the final `lonely` node unconnected — it's the newest by
        // insertion order and would win in Recent mode.
        for spoke in [&spoke_a, &spoke_b, &spoke_c] {
            link(&storage, &hub, spoke);
        }
        let _lonely = ingest(&storage, "lonely newcomer");

        let center = default_center_id(&storage, GraphSort::Connected).unwrap();
        assert_eq!(
            center, hub,
            "Connected mode should pick the densest node, not the newest"
        );
    }

    #[test]
    fn default_center_id_connected_falls_back_to_recent_when_no_edges() {
        let (_dir, storage) = seed_storage();
        ingest(&storage, "alpha");
        let newest = ingest(&storage, "beta");

        // No connections exist — Connected mode should degrade to Recent
        // rather than returning 404.
        let center = default_center_id(&storage, GraphSort::Connected).unwrap();
        assert_eq!(center, newest);
    }

    #[test]
    fn default_center_id_returns_not_found_on_empty_db() {
        let (_dir, storage) = seed_storage();

        let recent_err = default_center_id(&storage, GraphSort::Recent).unwrap_err();
        assert_eq!(recent_err, StatusCode::NOT_FOUND);

        let connected_err = default_center_id(&storage, GraphSort::Connected).unwrap_err();
        assert_eq!(connected_err, StatusCode::NOT_FOUND);
    }
}
