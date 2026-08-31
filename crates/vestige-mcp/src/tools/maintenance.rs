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
        "properties": {}
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

    let embedding_ready = storage.is_embedding_ready();

    let mut warnings = Vec::new();
    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings.push("Low average retention - consider running consolidation");
    }
    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }
    if stats.total_nodes > 0 && stats.nodes_with_active_embeddings == 0 {
        warnings.push("No active-model embeddings generated - semantic search unavailable");
    }
    if embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push("Low embedding coverage - run consolidate to improve semantic search");
    }
    if stats.nodes_with_mismatched_embeddings > 0 {
        warnings.push(
            "Stored embeddings from another model are present - run consolidate after changing embedding models",
        );
    }

    let mut recommendations = Vec::new();
    if status == "critical" {
        recommendations
            .push("CRITICAL: Many memories have very low retention. Review important memories.");
    }
    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }
    if stats.nodes_with_active_embeddings < stats.total_nodes {
        recommendations.push("Run 'consolidate' to generate active-model embeddings.");
    }
    if stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider running periodic consolidation.");
    }
    if status == "healthy" && recommendations.is_empty() {
        recommendations.push("Memory system is healthy!");
    }

    // === State distribution ===
    let nodes = storage.get_all_nodes(500, 0).map_err(|e| e.to_string())?;
    let total = nodes.len();
    let (active, dormant, silent, unavailable) = if total > 0 {
        let mut a = 0usize;
        let mut d = 0usize;
        let mut s = 0usize;
        let mut u = 0usize;
        for node in &nodes {
            let accessibility = node.retention_strength * 0.5
                + node.retrieval_strength * 0.3
                + node.storage_strength * 0.2;
            if accessibility >= 0.7 {
                a += 1;
            } else if accessibility >= 0.4 {
                d += 1;
            } else if accessibility >= 0.1 {
                s += 1;
            } else {
                u += 1;
            }
        }
        (a, d, s, u)
    } else {
        (0, 0, 0, 0)
    };

    // === FSRS Preview ===
    let scheduler = FSRSScheduler::default();
    let fsrs_preview = if let Some(representative) = nodes.first() {
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
            "modulesActive": 28,
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
    let last_consolidation = storage.get_last_consolidation().ok().flatten();
    let last_dream = storage.get_last_dream().ok().flatten();
    let saves_since_last_dream = match &last_dream {
        Some(dt) => storage.count_memories_since(*dt).unwrap_or(0),
        None => stats.total_nodes,
    };
    let last_backup = storage.last_backup_timestamp();

    let mut response = serde_json::json!({
        "tool": "system_status",
        // Health
        "status": status,
        "warnings": warnings,
        "recommendations": recommendations,
        "embeddingReady": embedding_ready,
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
        "embeddingModel": stats.embedding_model,
        "activeEmbeddingModel": stats.active_embedding_model,
        "oldestMemory": stats.oldest_memory.map(|dt| dt.to_rfc3339()),
        "newestMemory": stats.newest_memory.map(|dt| dt.to_rfc3339()),
        // Distribution
        "stateDistribution": {
            "active": active,
            "dormant": dormant,
            "silent": silent,
            "unavailable": unavailable,
            "sampled": total,
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
    _args: Option<Value>,
) -> Result<Value, String> {
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
    min_retention: Option<f64>,
    max_age_days: Option<u64>,
    dry_run: Option<bool>,
}

/// Garbage collection tool
pub async fn execute_gc(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: GcArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => GcArgs {
            min_retention: None,
            max_age_days: None,
            dry_run: None,
        },
    };

    let min_retention = args.min_retention.unwrap_or(0.1).clamp(0.0, 1.0);
    let max_age_days = args.max_age_days;
    let dry_run = args.dry_run.unwrap_or(true); // Default to dry_run for safety

    let now = Utc::now();

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

    // Find candidates
    let candidates: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            if node.retention_strength >= min_retention {
                return false;
            }
            if let Some(max_days) = max_age_days {
                let age_days = (now - node.created_at).num_days();
                if age_days < 0 || (age_days as u64) < max_days {
                    return false;
                }
            }
            true
        })
        .collect();

    let candidate_count = candidates.len();

    // Build sample for display
    let sample: Vec<Value> = candidates
        .iter()
        .take(10)
        .map(|node| {
            let age_days = (now - node.created_at).num_days();
            let content_preview: String = {
                let preview: String = node.content.chars().take(60).collect();
                if preview.len() < node.content.len() {
                    format!("{}...", preview)
                } else {
                    preview
                }
            };
            serde_json::json!({
                "id": &node.id[..8.min(node.id.len())],
                "retention": node.retention_strength,
                "ageDays": age_days,
                "contentPreview": content_preview,
            })
        })
        .collect();

    if dry_run {
        return Ok(serde_json::json!({
            "tool": "gc",
            "dryRun": true,
            "minRetention": min_retention,
            "maxAgeDays": max_age_days,
            "candidateCount": candidate_count,
            "totalMemories": all_nodes.len(),
            "sample": sample,
            "message": format!("{} memories would be deleted. Set dry_run=false to delete.", candidate_count),
        }));
    }

    // Perform actual deletion
    let mut deleted = 0usize;
    let mut errors = 0usize;
    let ids: Vec<String> = candidates.iter().map(|n| n.id.clone()).collect();

    for id in &ids {
        match storage.delete_node(id) {
            Ok(true) => deleted += 1,
            Ok(false) => errors += 1,
            Err(_) => errors += 1,
        }
    }

    Ok(serde_json::json!({
        "tool": "gc",
        "dryRun": false,
        "minRetention": min_retention,
        "maxAgeDays": max_age_days,
        "deleted": deleted,
        "errors": errors,
        "totalBefore": all_nodes.len(),
        "totalAfter": all_nodes.len() - deleted,
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

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
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
        assert_eq!(value["cognitiveHealth"]["modulesActive"], 28);
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
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
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
