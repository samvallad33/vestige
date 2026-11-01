//! Maintenance MCP Tools
//!
//! Exposes CLI-only operations as MCP tools so Claude can trigger them automatically:
//! system_status, consolidate, backup, export, gc.

use chrono::{NaiveDate, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{FSRSScheduler, Storage};

fn create_private_file(path: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
    }

    #[cfg(not(unix))]
    {
        std::fs::File::create(path)
    }
}

// ============================================================================
// SCHEMAS
// ============================================================================

pub fn consolidate_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "budgetMs": {"type":"integer", "minimum":1,"maximum":10000,"default":1000},
            "phase": {"type": "string", "enum": ["all", "lifecycle", "logs"], "default": "all"},
            "batchSize": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10,
                "description": "Bounded phases: selected memories or log rows per call."},
            "after": {"type": "string", "description": "Lifecycle phase: nextCursor from the previous page. Omit after a sweep to retry failures and discover earlier inserts."},
            "dry_run": {"type": "boolean", "default": true, "description": "Bounded phases: preview the selected batch without inference or mutation."}
        }
    })
}

pub fn backup_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

pub fn export_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Export format: 'json' (default), 'jsonl', or 'portable' for exact Vestige-to-Vestige transfer",
                "enum": ["json", "jsonl", "portable"],
                "default": "json"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Filter by tags (ALL must match)"
            },
            "since": {
                "type": "string",
                "description": "Only export memories created after this date (YYYY-MM-DD)"
            },
            "path": {
                "type": "string",
                "description": "Custom filename (not path). File is saved in the active Vestige data directory's exports/ folder. Default: memories-{timestamp}.{format}"
            }
        }
    })
}

pub fn gc_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "batchSize":{"type":"integer","minimum":1,"maximum":1000,"default":100},
            "after":{"type":"string","description":"Live scan cursor from previous page; omit after completing a sweep."},
            "budgetMs":{"type":"integer","minimum":1,"maximum":10000,"default":1000},
            "min_retention": {
                "type": "number",
                "description": "Legacy engine only: delete memories with retention below this threshold (default: 0.1). Strata lists no candidates and deletes nothing.",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "max_age_days": {
                "type": "integer",
                "description": "Legacy engine only: only delete memories older than this many days (optional additional filter)",
                "minimum": 1
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true (default), only report what would be deleted without actually deleting",
                "default": true
            }
        }
    })
}

/// Combined system status schema (replaces health_check + stats in v1.7.0)
pub fn system_status_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "schema_introspection": {
                "type": "boolean",
                "description": "When true, extends the response with a 'schema' block carrying the SQLite schema version, per-table row counts + column lists, and embedding-coverage convenience fields. Default: false (response shape unchanged). Use this for audit / migration-guard / downstream-upgrade scripts that otherwise have to read SQLite directly.",
                "default": false
            },
            "detail": {
                "type": "string",
                "enum": ["compact", "full"],
                "description": "compact (default) omits scheduler, FSRS and embedding blocks; full returns them.",
                "default": "compact"
            }
        }
    })
}

/// Arguments for the system_status tool. All optional.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SystemStatusArgs {
    #[serde(alias = "schema_introspection")]
    schema_introspection: Option<bool>,
    detail: Option<String>,
}

// ============================================================================
// EXECUTE FUNCTIONS
// ============================================================================

/// Combined system status tool (merges health_check + stats, v1.7.0)
///
/// Returns system health status, full statistics, FSRS preview,
/// cognitive module health, state distribution, and actionable recommendations.
///
/// v2.1.24+: when `schema_introspection: true` is passed, the response
/// additionally carries a `schema` block with the live SQLite schema version,
/// per-table row counts + column lists, and embedding-coverage convenience
/// fields. Default off; response shape unchanged when omitted.
pub async fn execute_system_status(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    // Parse arguments (all optional, including the args envelope itself).
    let parsed: SystemStatusArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => SystemStatusArgs::default(),
    };
    let include_schema = parsed.schema_introspection.unwrap_or(false);

    let stats = storage.get_stats().map_err(|e| e.to_string())?;

    // === Health assessment ===
    let status = if stats.total_nodes == 0 {
        "empty"
    } else if stats.average_retention < 0.3 {
        "critical"
    } else if stats.average_retention < 0.5 {
        "degraded"
    } else {
        "healthy"
    };

    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };

    // w1b: vector code removed; there is no embedding runtime to be ready.
    let embedding_ready = false;
    let embeddings_compiled_in = crate::embeddings_compiled_in();

    let mut warnings = Vec::new();
    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings.push("Low average retention - promote or review the memories still needed");
    }
    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }
    // A build without an embedding runtime has nothing to generate, so the
    // coverage warnings below would read as a failure that never happened.
    if !embeddings_compiled_in {
        warnings.push(
            "No embedding runtime in this build - memories are found by exact handle only (a full id or an exact tag)",
        );
    }
    if embeddings_compiled_in && stats.total_nodes > 0 && stats.nodes_with_active_embeddings == 0 {
        warnings.push("No active-model embeddings generated - semantic search unavailable");
    }
    if embeddings_compiled_in && embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push("Low embedding coverage - run consolidate to improve semantic search");
    }
    if stats.nodes_with_mismatched_embeddings > 0 {
        warnings.push(
            "Stored embeddings from another model are present - run consolidate after changing embedding models",
        );
    }

    // === Automation trigger timestamps (read early: the stale-consolidation
    // diagnostic below consumes last_consolidation) ===
    let last_consolidation = storage.get_last_consolidation().ok().flatten();
    let last_dream = storage.get_last_dream().ok().flatten();
    let saves_since_last_dream = match &last_dream {
        Some(dt) => storage.count_memories_since(*dt).unwrap_or(0),
        None => stats.total_nodes,
    };
    let last_backup = storage.last_backup_timestamp();

    // === Structured diagnostics (upgrade/memory-status) ===
    // Warnings above stay byte-compatible strings for audit scripts. The
    // diagnostics array is the machine-readable panel: each entry names a
    // detected problem, the affected count, up to 10 example entity IDs, and
    // a concrete next action referencing an existing tool. Reads are
    // failure-tolerant: a diagnostic source failing must never fail health.
    const DIAGNOSTIC_ID_LIMIT: usize = 10;
    let mut diagnostics: Vec<Value> = Vec::new();
    // `consolidate` is withheld on a Strata log, so a diagnostic must not point
    // there. `dream` replays recorded edges and folds one FSRS review per
    // endpoint, which is the maintenance pass a Strata log does run.
    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let upkeep_action = if strata { "dream" } else { "consolidate" };

    // Decay risk: memories below the retention floor, with the worst IDs.
    let below_30 = storage.count_memories_below_retention(0.3).unwrap_or(0);
    let worst = storage
        .lowest_retention_nodes(DIAGNOSTIC_ID_LIMIT)
        .unwrap_or_default();
    if below_30 > 0 {
        let severity = if stats.total_nodes > 0 && below_30 * 2 >= stats.total_nodes {
            "critical"
        } else {
            "warning"
        };
        diagnostics.push(serde_json::json!({
            "code": "low_retention",
            "severity": severity,
            "detail": format!(
                "{below_30} memories below 30% retention; worst listed first by id"
            ),
            "count": below_30,
            "memoryIds": worst.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            "memoryIdsTruncated": below_30 > DIAGNOSTIC_ID_LIMIT as i64,
            "nextAction": { "tool": "maintain", "args": { "action": upkeep_action } },
        }));
    }

    // Review backlog: due memories with example IDs.
    let due_ids = storage
        .due_for_review_node_ids(DIAGNOSTIC_ID_LIMIT)
        .unwrap_or_default();
    if stats.nodes_due_for_review > 10 {
        diagnostics.push(serde_json::json!({
            "code": "due_for_review",
            "severity": "warning",
            "detail": format!(
                "{} memories are past their next_review timestamp; oldest due listed first",
                stats.nodes_due_for_review
            ),
            "count": stats.nodes_due_for_review,
            "memoryIds": due_ids,
            "memoryIdsTruncated": stats.nodes_due_for_review > DIAGNOSTIC_ID_LIMIT as i64,
            "nextAction": { "tool": "maintain", "args": { "action": upkeep_action } },
        }));
    }

    // Embedding gaps: only meaningful in builds that can generate embeddings.
    if embeddings_compiled_in && stats.total_nodes > 0 {
        let missing = stats.total_nodes - stats.nodes_with_active_embeddings;
        if stats.nodes_with_active_embeddings == 0 {
            diagnostics.push(serde_json::json!({
                "code": "embedding_coverage_gap",
                "severity": "critical",
                "detail": "No memory has an active-model embedding; semantic recall is blind",
                "count": missing,
                "nextAction": {
                    "tool": "maintain",
                    "args": { "action": "consolidate", "phase": "embeddings" }
                },
            }));
        } else if embedding_coverage < 50.0 && stats.total_nodes > 10 {
            diagnostics.push(serde_json::json!({
                "code": "embedding_coverage_gap",
                "severity": "warning",
                "detail": format!(
                    "Embedding coverage is {embedding_coverage:.1}% ({} of {} memories)",
                    stats.nodes_with_active_embeddings, stats.total_nodes
                ),
                "count": missing,
                "nextAction": {
                    "tool": "maintain",
                    "args": { "action": "consolidate", "phase": "embeddings" }
                },
            }));
        }
        if stats.nodes_with_mismatched_embeddings > 0 {
            diagnostics.push(serde_json::json!({
                "code": "embedding_model_mismatch",
                "severity": "warning",
                "detail": format!(
                    "{} memories hold embeddings from a model other than the active profile",
                    stats.nodes_with_mismatched_embeddings
                ),
                "count": stats.nodes_with_mismatched_embeddings,
                "nextAction": { "tool": "maintain", "args": { "action": "consolidate" } },
            }));
        }
    }

    // Consolidation staleness: never run, or older than 7 days. A Strata log
    // withholds consolidation, so "never run" there is not a finding.
    if !strata {
        match last_consolidation {
            Some(ts) => {
                let age_days = (Utc::now() - ts).num_days();
                if age_days > 7 {
                    diagnostics.push(serde_json::json!({
                        "code": "stale_consolidation",
                        "severity": "info",
                        "detail": format!(
                            "Last consolidation ran {age_days} days ago; FSRS decay scores go stale between runs"
                        ),
                        "nextAction": { "tool": "maintain", "args": { "action": "consolidate" } },
                    }));
                }
            }
            None => {
                if stats.total_nodes > 0 {
                    diagnostics.push(serde_json::json!({
                        "code": "stale_consolidation",
                        "severity": "info",
                        "detail": "Consolidation has never run on this store",
                        "nextAction": { "tool": "maintain", "args": { "action": "consolidate" } },
                    }));
                }
            }
        }
    }

    // Retention trajectory from the consolidation snapshots.
    if storage.get_retention_trend().unwrap_or_default() == "declining" {
        diagnostics.push(serde_json::json!({
            "code": "retention_trend_declining",
            "severity": "warning",
            "detail": "Average retention is declining across recent consolidation snapshots",
            "nextAction": { "tool": "memory_status", "args": { "view": "retention" } },
        }));
    }

    let mut recommendations = Vec::new();
    if status == "critical" {
        recommendations
            .push("CRITICAL: Many memories have very low retention. Review important memories.");
    }
    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }
    if embeddings_compiled_in && stats.nodes_with_active_embeddings < stats.total_nodes {
        recommendations.push("Run 'consolidate' to generate active-model embeddings.");
    }
    if stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider reviewing memories periodically.");
    }
    if status == "healthy" && recommendations.is_empty() {
        recommendations.push("The store is healthy.");
    }

    // === State distribution ===
    // Computed in SQL over EVERY stored row (upgrade/memory-status). The old
    // version blended strengths over only the first 500 rows in insertion
    // order, so on a large store the "distribution" silently described the
    // oldest slice of the data. Thresholds and the blend
    // (0.5*retention + 0.3*retrieval + 0.2*storage) are unchanged.
    let (active, dormant, silent, unavailable) =
        storage.state_distribution().map_err(|e| e.to_string())?;
    let total = active + dormant + silent + unavailable;

    // === FSRS Preview ===
    // Representative = newest memory (get_all_nodes is created_at DESC),
    // labeled as such so the preview is not mistaken for a store-wide
    // projection. One row, not 500.
    let scheduler = FSRSScheduler::default();
    let representative = storage
        .get_all_nodes(1, 0)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next();
    let fsrs_preview = if let Some(representative) = representative {
        let mut state = scheduler.new_card();
        state.difficulty = representative.difficulty;
        state.stability = representative.stability;
        state.reps = representative.reps;
        state.lapses = representative.lapses;
        state.last_review = representative.last_accessed;
        let elapsed = scheduler.days_since_review(&state.last_review);
        let preview = scheduler.preview_reviews(&state, elapsed);
        Some(serde_json::json!({
            "representativeMemoryId": representative.id,
            "representativeBasis": "newest memory",
            "elapsedDays": format!("{:.1}", elapsed),
            "intervalIfGood": preview.good.interval,
            "intervalIfEasy": preview.easy.interval,
            "intervalIfHard": preview.hard.interval,
            "currentRetrievability": format!("{:.3}", preview.good.retrievability),
        }))
    } else {
        None
    };

    // === Cognitive health ===
    let cognitive_health = if let Ok(cog) = cognitive.try_lock() {
        let activation_count = cog.activation_network.edge_count();
        let scheduler_stats = cog.consolidation_scheduler.get_activity_stats();
        Some(serde_json::json!({
            "activationNetworkSize": activation_count,
            "predictionAccuracy": "0.00",
            // Compiled-in module count (single source: COGNITIVE_MODULE_COUNT).
            // Cognitive modules are in-process structs with no failure
            // channel; this is a build fact, not a runtime probe.
            "modulesActive": crate::cognitive::COGNITIVE_MODULE_COUNT,
            "schedulerStats": {
                "totalEvents": scheduler_stats.total_events,
                "eventsPerMinute": scheduler_stats.events_per_minute,
                "isIdle": scheduler_stats.is_idle,
                "timeUntilNextConsolidation": format!("{:?}", cog.consolidation_scheduler.time_until_next()),
            },
        }))
    } else {
        None
    };

    // === Automation triggers (for conditional dream/backup/gc at session start) ===
    // Reads happen early in this function (the stale-consolidation diagnostic
    // consumes last_consolidation); only response assembly happens here.

    let mut response = serde_json::json!({
        "tool": "system_status",
        // Health
        "status": status,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "embeddingReady": embedding_ready,
        "embeddingsCompiledIn": embeddings_compiled_in,
        // Stats
        "totalMemories": stats.total_nodes,
        "dueForReview": stats.nodes_due_for_review,
        "averageRetention": stats.average_retention,
        "averageStorageStrength": stats.average_storage_strength,
        "averageRetrievalStrength": stats.average_retrieval_strength,
        "withEmbeddings": stats.nodes_with_embeddings,
        "withActiveEmbeddings": stats.nodes_with_active_embeddings,
        "mismatchedEmbeddings": stats.nodes_with_mismatched_embeddings,
        "embeddingCoverage": format!("{:.1}%", embedding_coverage),
        // (embeddingsCompiledIn was serialized twice here; the duplicate
        //  literal above this block is the one that survived serde_json's
        //  last-key-wins behavior. Kept once.)
        "embeddingModel": stats.embedding_model,
        "activeEmbeddingModel": stats.active_embedding_model,
        "oldestMemory": stats.oldest_memory.map(|dt| dt.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|dt| dt.to_rfc3339()),
        // Distribution — full-population SQL aggregate; `sampled` keeps its
        // key for compatibility but now equals the whole store.
        "stateDistribution": {
            "active": active,
            "dormant": dormant,
            "silent": silent,
            "unavailable": unavailable,
            "sampled": total,
            "basis": "full",
        },
        // FSRS
        "fsrsPreview": fsrs_preview,
        // Cognitive
        "cognitiveHealth": cognitive_health,
        // Automation triggers — Claude uses these to decide when to dream/backup/gc
        "automationTriggers": {
            "lastDreamTimestamp": last_dream.map(|dt| dt.to_rfc3339()),
            "savesSinceLastDream": saves_since_last_dream,
            "lastBackupTimestamp": last_backup.map(|dt| dt.to_rfc3339()),
            "lastConsolidationTimestamp": last_consolidation.map(|dt| dt.to_rfc3339()),
        },
    });

    // The default reply is what an agent reads. Scheduler dumps, an FSRS
    // preview of one card, and embedding counters that are zero in this build
    // are available with detail=full and are not part of the compact reply.
    if parsed.detail.as_deref() != Some("full") {
        if let Some(object) = response.as_object_mut() {
            object.remove("cognitiveHealth");
            object.remove("fsrsPreview");
            if !embeddings_compiled_in {
                for key in [
                    "embeddingReady",
                    "withEmbeddings",
                    "withActiveEmbeddings",
                    "mismatchedEmbeddings",
                    "embeddingCoverage",
                    "embeddingModel",
                    "activeEmbeddingModel",
                ] {
                    object.remove(key);
                }
            }
        }
        if let Some(warnings) = response["warnings"].as_array_mut() {
            warnings.retain(|warning| {
                !warning
                    .as_str()
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .contains("embedding")
            });
        }
        if let Some(diagnostics) = response["diagnostics"].as_array_mut() {
            diagnostics.retain(|row| !row["code"].as_str().unwrap_or("").starts_with("embedding_"));
        }
    }

    // v2.1.24+: optional schema introspection block. Default off; response
    // shape unchanged when omitted.
    if include_schema {
        let intro = storage.schema_introspection().map_err(|e| e.to_string())?;
        let tables_json: Vec<Value> = intro
            .tables
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "rows": t.rows,
                    "columns": t.columns,
                })
            })
            .collect();
        response["schema"] = serde_json::json!({
            "schemaVersion": intro.schema_version,
            "schemaVersionAppliedAt": intro.schema_version_applied_at.map(|dt| dt.to_rfc3339()),
            "tables": tables_json,
            "embeddingNullCount": intro.embedding_null_count,
            "activeEmbeddingModel": intro.active_embedding_model,
            "activeEmbeddingDimensions": intro.active_embedding_dimensions,
        });
    }

    Ok(response)
}

/// Consolidate tool
pub async fn execute_consolidate(
    storage: &Arc<Storage>,
    args: Option<Value>,
) -> Result<Value, String> {
    #[derive(Default, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Args {
        phase: Option<String>,
        batch_size: Option<usize>,
        budget_ms: Option<u64>,
        after: Option<String>,
        #[serde(alias = "dry_run")]
        dry_run: Option<bool>,
    }
    let parsed: Args = serde_json::from_value(args.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|error| error.to_string())?;
    if !matches!(
        parsed.phase.as_deref().unwrap_or("all"),
        "all" | "lifecycle" | "logs"
    ) {
        return Err("phase must be all, lifecycle or logs".into());
    }
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        // `maintain action=consolidate` is withheld before it reaches here; the
        // hidden `consolidate` alias carries no `action`, so it lands here and
        // gets the same refusal instead of a no-op's all-zero "success".
        return Err(super::unavailable::withheld_in_4_0(
            "maintain action 'consolidate'",
            super::unavailable::CONSOLIDATE_NOOP,
        ));
    }
    match parsed.phase.as_deref().unwrap_or("all") {
        // w1b: the "embeddings" phase was removed with the vector runtime;
        // only lifecycle/log row batches and the full sweep remain.
        "lifecycle" | "logs" => {
            let storage = Arc::clone(storage);
            return tokio::task::spawn_blocking(move || {
                if parsed.phase.as_deref() == Some("logs") {
                    if parsed.after.is_some() || parsed.budget_ms.is_some() {
                        return Err(vestige_core::storage::StorageError::Init(
                            "logs uses repeatable row batches; after and budgetMs are unsupported"
                                .into(),
                        ));
                    }
                    storage.maintain_log_batch(
                        parsed.batch_size.unwrap_or(100),
                        parsed.dry_run.unwrap_or(true),
                    )
                } else {
                    storage.maintain_lifecycle_batch(
                        parsed.batch_size.unwrap_or(100),
                        parsed.after.as_deref(),
                        parsed.budget_ms.unwrap_or(1000),
                        parsed.dry_run.unwrap_or(true),
                    )
                }
            })
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string());
        }
        "all" => {
            if parsed.batch_size.is_some()
                || parsed.after.is_some()
                || parsed.dry_run.is_some()
                || parsed.budget_ms.is_some()
            {
                return Err("batchSize, after and dry_run require a bounded phase".into());
            }
        }
        _ => return Err("phase must be all, lifecycle or logs".into()),
    }
    let result = storage.run_consolidation().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "tool": "consolidate",
        "nodesProcessed": result.nodes_processed,
        "nodesPromoted": result.nodes_promoted,
        "nodesPruned": result.nodes_pruned,
        "decayApplied": result.decay_applied,
        "embeddingsGenerated": result.embeddings_generated,
        "duplicatesMerged": result.duplicates_merged,
        "activationsComputed": result.activations_computed,
        "w20Optimized": result.w20_optimized,
        "durationMs": result.duration_ms,
    }))
}

/// Backup tool
/// Bytes on disk for a backup: the file, or every file under a directory.
fn path_size(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| path_size(&entry.path()))
                .sum()
        })
        .unwrap_or(0)
}

pub async fn execute_backup(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    // Determine backup path
    let backup_dir = storage.sidecar_dir("backups");

    std::fs::create_dir_all(&backup_dir)
        .map_err(|e| format!("Failed to create backup directory: {}", e))?;

    let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
    // A SQLite backup is one file; a Strata backup is a copy of the log
    // directory, so it gets no `.db` suffix.
    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
    let name = if strata {
        format!("vestige-{timestamp}.strata")
    } else {
        format!("vestige-{timestamp}.db")
    };
    let backup_path = backup_dir.join(name);

    // SQLite: VACUUM INTO (consistent under WAL). Strata: sealed log copy.
    {
        storage
            .backup_to(&backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;
    }

    let file_size = path_size(&backup_path);

    Ok(serde_json::json!({
        "tool": "backup",
        "path": backup_path.display().to_string(),
        "sizeBytes": file_size,
        "timestamp": Utc::now().to_rfc3339(),
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportArgs {
    format: Option<String>,
    tags: Option<Vec<String>>,
    since: Option<String>,
    path: Option<String>,
}

/// Export tool
pub async fn execute_export(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ExportArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => ExportArgs {
            format: None,
            tags: None,
            since: None,
            path: None,
        },
    };

    let format = args.format.unwrap_or_else(|| "json".to_string());
    if format != "json" && format != "jsonl" && format != "portable" {
        return Err(format!(
            "Invalid format '{}'. Must be 'json', 'jsonl', or 'portable'.",
            format
        ));
    }

    if format == "portable" {
        if args.tags.as_ref().is_some_and(|tags| !tags.is_empty()) || args.since.is_some() {
            return Err(
                "Portable export is exact and does not support tags or since filters.".to_string(),
            );
        }

        let export_dir = storage.sidecar_dir("exports");
        std::fs::create_dir_all(&export_dir)
            .map_err(|e| format!("Failed to create export directory: {}", e))?;

        let export_path = match args.path {
            Some(ref p) => {
                let filename = std::path::Path::new(p)
                    .file_name()
                    .ok_or("Invalid export filename: must be a simple filename, not a path")?;
                let name_str = filename.to_str().ok_or("Invalid filename encoding")?;
                if name_str.contains("..") {
                    return Err("Invalid export filename: '..' not allowed".to_string());
                }
                export_dir.join(filename)
            }
            None => {
                let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
                export_dir.join(format!("vestige-portable-{}.json", timestamp))
            }
        };

        let archive = storage
            .export_portable_archive_to_path(&export_path)
            .map_err(|e| e.to_string())?;
        let file_size = std::fs::metadata(&export_path)
            .map(|m| m.len())
            .unwrap_or(0);

        return Ok(serde_json::json!({
            "tool": "export",
            "path": export_path.display().to_string(),
            "format": "portable",
            "archiveFormat": archive.archive_format,
            "schemaVersion": archive.schema_version,
            "tablesExported": archive.tables.len(),
            "rowsExported": archive.total_rows(),
            "sizeBytes": file_size,
        }));
    }

    // Parse since date
    let since_date = match &args.since {
        Some(date_str) => {
            let naive = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|e| format!("Invalid date '{}': {}. Use YYYY-MM-DD.", date_str, e))?;
            Some(naive.and_hms_opt(0, 0, 0).unwrap().and_utc())
        }
        None => None,
    };

    let tag_filter: Vec<String> = args.tags.unwrap_or_default();

    // Fetch all nodes (capped at 100K to prevent OOM)
    let mut all_nodes = Vec::new();
    let page_size = 500;
    let max_nodes = 100_000;
    let mut offset = 0;
    loop {
        let batch = storage
            .get_all_nodes(page_size, offset)
            .map_err(|e| e.to_string())?;
        let batch_len = batch.len();
        all_nodes.extend(batch);
        if batch_len < page_size as usize || all_nodes.len() >= max_nodes {
            break;
        }
        offset += page_size;
    }

    // Apply filters
    let filtered: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            if since_date
                .as_ref()
                .is_some_and(|since_dt| node.created_at < *since_dt)
            {
                return false;
            }
            if !tag_filter.is_empty() {
                for tag in &tag_filter {
                    if !node.tags.iter().any(|t| t == tag) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    // Determine export path — always constrained to vestige exports directory
    let export_dir = storage.sidecar_dir("exports");
    std::fs::create_dir_all(&export_dir)
        .map_err(|e| format!("Failed to create export directory: {}", e))?;

    let export_path = match args.path {
        Some(ref p) => {
            // Only allow a filename, not a path — prevent path traversal
            let filename = std::path::Path::new(p)
                .file_name()
                .ok_or("Invalid export filename: must be a simple filename, not a path")?;
            let name_str = filename.to_str().ok_or("Invalid filename encoding")?;
            if name_str.contains("..") {
                return Err("Invalid export filename: '..' not allowed".to_string());
            }
            export_dir.join(filename)
        }
        None => {
            let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
            export_dir.join(format!("memories-{}.{}", timestamp, format))
        }
    };

    // Write export
    let file = create_private_file(&export_path)
        .map_err(|e| format!("Failed to create export file: {}", e))?;
    let mut writer = std::io::BufWriter::new(file);

    use std::io::Write;
    match format.as_str() {
        "json" => {
            serde_json::to_writer_pretty(&mut writer, &filtered)
                .map_err(|e| format!("Failed to write JSON: {}", e))?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        "jsonl" => {
            for node in &filtered {
                serde_json::to_writer(&mut writer, node)
                    .map_err(|e| format!("Failed to write JSONL: {}", e))?;
                writer.write_all(b"\n").map_err(|e| e.to_string())?;
            }
        }
        // Defensive: the `format != "json" && format != "jsonl"` early-return
        // above should already catch every unsupported format, but that gate is
        // at the arg-validation layer. If it ever grows a bug (e.g. case
        // sensitivity drift, a new branch, refactor) we return a clean error
        // instead of `unreachable!()` — no panic can reach a user via the MCP
        // dispatcher.
        other => {
            return Err(format!(
                "unsupported export format: {:?}. Expected 'json' or 'jsonl'.",
                other
            ));
        }
    }
    writer.flush().map_err(|e| e.to_string())?;

    let file_size = std::fs::metadata(&export_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(serde_json::json!({
        "tool": "export",
        "path": export_path.display().to_string(),
        "format": format,
        "memoriesExported": filtered.len(),
        "totalMemories": all_nodes.len(),
        "sizeBytes": file_size,
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GcArgs {
    batch_size: Option<usize>,
    after: Option<String>,
    budget_ms: Option<u64>,
    #[serde(alias = "min_retention")]
    min_retention: Option<f64>,
    #[serde(alias = "max_age_days")]
    max_age_days: Option<u64>,
    #[serde(alias = "dry_run")]
    dry_run: Option<bool>,
}

/// Garbage collection tool
pub async fn execute_gc(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: GcArgs = serde_json::from_value(args.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|e| e.to_string())?;
    let storage = Arc::clone(storage);
    tokio::task::spawn_blocking(move || {
        storage.maintain_gc_batch(
            args.batch_size.unwrap_or(100),
            args.after.as_deref(),
            args.budget_ms.unwrap_or(1000),
            args.dry_run.unwrap_or(true),
            args.min_retention.unwrap_or(0.1),
            args.max_age_days,
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}

// ============================================================================
// TESTS
// ============================================================================
