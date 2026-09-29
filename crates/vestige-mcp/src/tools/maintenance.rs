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
                "description": "Delete memories with retention below this threshold (default: 0.1)",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "max_age_days": {
                "type": "integer",
                "description": "Only delete memories older than this many days (optional additional filter)",
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
        warnings.push("Low average retention - consider running consolidation");
    }
    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }
    // A build without an embedding runtime has nothing to generate, so the
    // coverage warnings below would read as a failure that never happened.
    if !embeddings_compiled_in {
        warnings.push(
            "Built without embeddings - semantic recall and the prediction-error gate are unavailable in this build (keyword search only)",
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

    // Decay risk: memories below the retention floor, with the worst IDs.
    let below_30 = storage.count_memories_below_retention(0.3).unwrap_or(0);
    let worst = storage.lowest_retention_nodes(DIAGNOSTIC_ID_LIMIT).unwrap_or_default();
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
            "nextAction": { "tool": "maintain", "args": { "action": "consolidate" } },
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
            "nextAction": { "tool": "maintain", "args": { "action": "consolidate" } },
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

    // Consolidation staleness: never run, or older than 7 days.
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
        recommendations.push("Consider running periodic consolidation.");
    }
    if status == "healthy" && recommendations.is_empty() {
        recommendations.push("Memory system is healthy!");
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
        let prediction_accuracy = cog.predictive_memory.prediction_accuracy().unwrap_or(0.0);
        let scheduler_stats = cog.consolidation_scheduler.get_activity_stats();
        Some(serde_json::json!({
            "activationNetworkSize": activation_count,
            "predictionAccuracy": format!("{:.2}", prediction_accuracy),
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
    match parsed.phase.as_deref().unwrap_or("all") {
        // Vector runtime is gone. Keep a bounded empty page so callers still
        // see dryRun/selected/hasMore; nothing is embedded.
        "embeddings" => {
            if parsed.budget_ms.is_some() {
                return Err("embedding inference supports a row bound, not budgetMs".into());
            }
            let batch = parsed.batch_size.unwrap_or(10);
            if !(1..=100).contains(&batch) {
                return Err("embeddings batchSize must be 1..=100".into());
            }
            return Ok(serde_json::json!({
                "phase": "embeddings",
                "dryRun": parsed.dry_run.unwrap_or(true),
                "selected": 0,
                "processed": 0,
                "hasMore": false,
            }));
        }
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
pub async fn execute_backup(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    // Determine backup path
    let backup_dir = storage.sidecar_dir("backups");

    std::fs::create_dir_all(&backup_dir)
        .map_err(|e| format!("Failed to create backup directory: {}", e))?;

    let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
    let backup_path = backup_dir.join(format!("vestige-{}.db", timestamp));

    // Use VACUUM INTO for a consistent backup (handles WAL properly)
    {
        storage
            .backup_to(&backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;
    }

    let file_size = std::fs::metadata(&backup_path)
        .map(|m| m.len())
        .unwrap_or(0);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    #[test]
    fn test_system_status_schema() {
        let schema = system_status_schema();
        assert_eq!(schema["type"], "object");
    }

    #[tokio::test]
    async fn test_system_status_empty_db() {
        let (storage, _dir) = test_storage().await;
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["tool"], "system_status");
        assert_eq!(value["status"], "empty");
        assert_eq!(value["totalMemories"], 0);
        assert!(value["warnings"].is_array());
        assert!(value["recommendations"].is_array());
    }

    #[tokio::test]
    async fn test_system_status_with_memories() {
        let (storage, _dir) = test_storage().await;
        {
            storage
                .ingest(vestige_core::IngestInput {
                    content: "Test memory for status".to_string(),
                    node_type: "fact".to_string(),
                    source: None,
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    tags: vec![],
                    valid_from: None,
                    valid_until: None,
                    validity_inferred: false,
                    source_envelope: None,
                })
                .unwrap();
        }
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 1);
        assert!(value["stateDistribution"].is_object());
        assert!(value["embeddingCoverage"].is_string());
    }

    #[tokio::test]
    async fn test_system_status_has_cognitive_health() {
        let (storage, _dir) = test_storage().await;
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        let value = result.unwrap();
        assert!(value["cognitiveHealth"].is_object());
        assert_eq!(
            value["cognitiveHealth"]["modulesActive"],
            crate::cognitive::COGNITIVE_MODULE_COUNT,
            "modulesActive must come from the maintained constant, not a literal"
        );
    }

    #[tokio::test]
    async fn test_system_status_has_automation_triggers() {
        let (storage, _dir) = test_storage().await;
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();

        let triggers = &value["automationTriggers"];
        assert!(triggers.is_object(), "automationTriggers should be present");
        assert!(triggers["lastDreamTimestamp"].is_null(), "No dreams yet");
        assert_eq!(triggers["savesSinceLastDream"], 0, "Empty DB = 0 saves");
        assert!(
            triggers["lastConsolidationTimestamp"].is_null(),
            "No consolidation yet"
        );
        // lastBackupTimestamp depends on filesystem state, just check it exists
        assert!(triggers.get("lastBackupTimestamp").is_some());
    }

    #[tokio::test]
    async fn test_system_status_automation_triggers_with_memories() {
        let (storage, _dir) = test_storage().await;
        {
            for i in 0..3 {
                storage
                    .ingest(vestige_core::IngestInput {
                        content: format!("Automation trigger test memory {}", i),
                        node_type: "fact".to_string(),
                        source: None,
                        sentiment_score: 0.0,
                        sentiment_magnitude: 0.0,
                        tags: vec![],
                        valid_from: None,
                        valid_until: None,
                        validity_inferred: false,
                        source_envelope: None,
                    })
                    .unwrap();
            }
        }
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        let value = result.unwrap();

        let triggers = &value["automationTriggers"];
        // No dream ever → savesSinceLastDream == totalMemories
        assert_eq!(triggers["savesSinceLastDream"], 3);
        assert!(triggers["lastDreamTimestamp"].is_null());
    }

    // ========================================================================
    // STRUCTURED DIAGNOSTICS TESTS (upgrade/memory-status)
    // ========================================================================

    /// An empty store must not invent problems: no decay, review, embedding
    /// or staleness diagnostics may fire.
    #[tokio::test]
    async fn test_system_status_empty_store_has_no_diagnostic_noise() {
        let (storage, _dir) = test_storage().await;
        let value = execute_system_status(&storage, &test_cognitive(), None)
            .await
            .unwrap();
        let diagnostics = value["diagnostics"].as_array().unwrap();
        assert!(
            diagnostics.is_empty(),
            "empty store must produce no diagnostics, got {diagnostics:?}"
        );
    }

    /// Health must DETECT a real problem, not just print counts: repeatedly
    /// demoting a memory drives its retention below the 0.3 floor via the
    /// public API, and the low_retention diagnostic must name that memory's
    /// id and a concrete next action. Consolidation has never run either, so
    /// the staleness diagnostic must fire as well.
    #[tokio::test]
    async fn test_system_status_diagnostics_name_decayed_memory_ids() {
        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory that will be driven into decay".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec![],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();
        // 7 demotions at -0.15 retention each bottom out at the 0.05 floor.
        for _ in 0..7 {
            storage.demote_memory(&node.id).unwrap();
        }

        let value = execute_system_status(&storage, &test_cognitive(), None)
            .await
            .unwrap();
        let diagnostics = value["diagnostics"].as_array().unwrap();

        let low = diagnostics
            .iter()
            .find(|d| d["code"] == "low_retention")
            .unwrap_or_else(|| panic!("low_retention diagnostic missing: {diagnostics:?}"));
        assert_eq!(low["count"].as_i64().unwrap(), 1);
        assert_eq!(
            low["memoryIds"].as_array().unwrap(),
            &vec![serde_json::json!(node.id)],
            "diagnostic must name the decayed memory"
        );
        assert!(low["memoryIdsTruncated"] == false);
        assert_eq!(low["nextAction"]["tool"], "maintain");
        assert_eq!(low["nextAction"]["args"]["action"], "consolidate");
        assert!(
            low["severity"] == "critical" || low["severity"] == "warning",
            "severity must be a known level, got {low:?}"
        );

        assert!(
            diagnostics
                .iter()
                .any(|d| d["code"] == "stale_consolidation"),
            "a store that never consolidated must report staleness"
        );

        // Full-population state distribution: sampled now equals the store.
        assert_eq!(value["stateDistribution"]["basis"], "full");
        assert_eq!(
            value["stateDistribution"]["sampled"],
            value["totalMemories"]
        );
        let distributed: i64 = ["active", "dormant", "silent", "unavailable"]
            .iter()
            .map(|k| value["stateDistribution"][k].as_i64().unwrap())
            .sum();
        assert_eq!(
            distributed, 1,
            "state distribution must account for every memory"
        );
    }

    // ========================================================================
    // SCHEMA INTROSPECTION TESTS (PR2)
    // ========================================================================

    #[test]
    fn test_system_status_schema_has_schema_introspection_flag() {
        let schema = system_status_schema();
        let props = &schema["properties"];
        let flag = &props["schema_introspection"];
        assert!(flag.is_object(), "schema_introspection property must exist");
        assert_eq!(flag["type"], "boolean");
        assert_eq!(flag["default"], false);
        // Top-level required must NOT include this — flag is opt-in.
        let required = schema.get("required");
        if let Some(req) = required {
            let req_arr = req.as_array().unwrap();
            assert!(!req_arr.contains(&serde_json::json!("schema_introspection")));
        }
    }

    #[tokio::test]
    async fn test_system_status_without_schema_flag_omits_schema_block() {
        // Backwards-compat: when the flag is not set (or false), the response
        // shape is unchanged — no `schema` key.
        let (storage, _dir) = test_storage().await;
        let result = execute_system_status(&storage, &test_cognitive(), None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(
            value.get("schema").is_none(),
            "schema block must NOT be present when flag is unset, got {:?}",
            value.get("schema")
        );

        // Explicit false → still no schema block.
        let result = execute_system_status(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "schema_introspection": false })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.get("schema").is_none());
    }

    #[tokio::test]
    async fn test_system_status_with_schema_flag_emits_schema_block() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(vestige_core::IngestInput {
                content: "Schema introspection seed memory".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["schema-test".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();

        let result = execute_system_status(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "schema_introspection": true })),
        )
        .await;
        assert!(result.is_ok(), "{:?}", result);
        let value = result.unwrap();

        // Shape assertions.
        let schema_block = value
            .get("schema")
            .expect("schema block must be present when flag is true");
        assert!(schema_block.is_object());
        assert!(
            schema_block["schemaVersion"].is_number(),
            "schemaVersion must be a number, got {:?}",
            schema_block["schemaVersion"]
        );
        // Schema version should be >= 13 (V13 is the highest landed migration
        // at the time this PR was authored).
        let v = schema_block["schemaVersion"].as_u64().unwrap();
        assert!(v >= 13, "expected schema_version >= 13, got {}", v);

        // tables should be a non-empty array of {name, rows, columns}.
        let tables = schema_block["tables"].as_array().unwrap();
        assert!(!tables.is_empty(), "expected at least one table");
        let kn = tables
            .iter()
            .find(|t| t["name"] == "knowledge_nodes")
            .expect("knowledge_nodes table must be present");
        assert_eq!(kn["rows"], 1, "ingested exactly one memory");
        let cols = kn["columns"].as_array().unwrap();
        assert!(!cols.is_empty(), "knowledge_nodes must have columns");
        // The id column is universally present.
        let col_names: Vec<&str> = cols.iter().filter_map(|c| c.as_str()).collect();
        assert!(
            col_names.contains(&"id"),
            "knowledge_nodes.id must be in columns list: {:?}",
            col_names
        );

        // Convenience fields.
        assert!(schema_block["embeddingNullCount"].is_number());
        // activeEmbeddingModel may be null if the `embeddings` feature is
        // not enabled in the test build; just check the key exists.
        assert!(schema_block.get("activeEmbeddingModel").is_some());
        assert!(schema_block.get("activeEmbeddingDimensions").is_some());
    }

    #[tokio::test]
    async fn test_system_status_camelcase_alias() {
        // Accept both `schema_introspection` (snake) and `schemaIntrospection`
        // (camel) per the #[serde(rename_all = "camelCase")] + alias attr.
        let (storage, _dir) = test_storage().await;
        let result = execute_system_status(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "schemaIntrospection": true })),
        )
        .await;
        assert!(result.is_ok(), "{:?}", result);
        let value = result.unwrap();
        assert!(
            value.get("schema").is_some(),
            "camelCase form must also trigger schema block"
        );
    }

    #[test]
    fn test_storage_schema_introspection_method() {
        // Direct test on the Storage method, independent of the MCP layer.
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        let intro = storage
            .schema_introspection()
            .expect("schema_introspection must succeed on a fresh DB");

        // Schema version pulled from the schema_version table.
        assert!(
            intro.schema_version >= 13,
            "fresh DB should be at schema_version >= 13, got {}",
            intro.schema_version
        );
        // At least one walked table should exist.
        assert!(
            !intro.tables.is_empty(),
            "expected at least one user-data table"
        );
        // Empty DB → no embeddings → embedding_null_count == 0 (no rows to
        // count). Once we ingest, it should be > 0 (no embeddings generated
        // in tests by default).
        assert_eq!(intro.embedding_null_count, 0);
    }

    #[tokio::test]
    async fn test_portable_export_writes_archive_to_storage_exports_dir() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(vestige_core::IngestInput {
                content: "Portable MCP export test memory".to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["portable".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap();

        let result = execute_export(
            &storage,
            Some(serde_json::json!({
                "format": "portable",
                "path": "portable-test.json"
            })),
        )
        .await
        .unwrap();

        let path = result["path"].as_str().unwrap();
        assert_eq!(result["format"], "portable");
        assert!(path.ends_with("exports/portable-test.json"));
        assert!(std::path::Path::new(path).exists());
        assert_eq!(
            result["archiveFormat"],
            vestige_core::PORTABLE_ARCHIVE_FORMAT
        );
        assert!(result["rowsExported"].as_u64().unwrap() > 0);
    }
}
