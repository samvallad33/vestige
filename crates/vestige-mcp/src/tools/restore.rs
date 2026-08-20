//! Restore Tool
//!
//! Restores memories from a JSON backup file.
//! Previously CLI-only (vestige-restore binary), now available as an MCP tool
//! so Claude Code can trigger restores directly.

use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

use vestige_core::{IngestInput, PortableArchive, PortableImportMode, Storage};

/// Input schema for restore tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the backup JSON file to restore from"
            },
            "allowAnyPath": {
                "type": "boolean",
                "description": "Allow restoring from a file outside the active Vestige backups/ or exports/ directories. Only set true for trusted local files.",
                "default": false
            },
            "merge": {
                "type": "boolean",
                "description": "For portable archives, merge into the current database instead of requiring an empty target. Applies sync tombstones and keeps newer local memory rows on conflict.",
                "default": false
            }
        },
        "required": ["path"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RestoreArgs {
    path: String,
    #[serde(default)]
    allow_any_path: bool,
    #[serde(default)]
    merge: bool,
}

#[derive(Deserialize)]
struct BackupWrapper {
    #[serde(rename = "type")]
    _type: String,
    text: String,
}

#[derive(Deserialize)]
struct RecallResult {
    results: Vec<MemoryBackup>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MemoryBackup {
    content: String,
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    source: Option<String>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: RestoreArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    let path = Path::new(&args.path);
    if !path.exists() {
        return Err(format!("Backup file not found: {}", args.path));
    }

    if !args.allow_any_path {
        ensure_restore_path_allowed(storage, path)?;
    }

    // Read and parse backup
    let backup_bytes = std::fs::read(path).map_err(|e| format!("Failed to read backup: {}", e))?;
    if backup_bytes.starts_with(b"SQLite format 3\0") {
        return Err(
            "Restore expected JSON, but this file is a raw SQLite database backup. Use portable export/import for cross-device transfer, or replace the database file manually while Vestige is stopped."
                .to_string(),
        );
    }
    let backup_content = String::from_utf8(backup_bytes)
        .map_err(|_| "Restore file is not UTF-8 JSON".to_string())?;

    if let Ok(archive) = serde_json::from_str::<PortableArchive>(&backup_content)
        && archive.archive_format == vestige_core::PORTABLE_ARCHIVE_FORMAT
    {
        let rows = archive.total_rows();
        let tables = archive.tables.len();
        let mode = if args.merge {
            PortableImportMode::Merge
        } else {
            PortableImportMode::EmptyOnly
        };
        let report = storage
            .import_portable_archive(&archive, mode)
            .map_err(|e| e.to_string())?;
        return Ok(serde_json::json!({
            "tool": "restore",
            "success": true,
            "mode": if args.merge { "portable-merge" } else { "portable" },
            "tables": tables,
            "rows": rows,
            "tablesImported": report.tables_imported,
            "rowsImported": report.rows_imported,
            "tablesSkipped": report.tables_skipped,
            "rowsInserted": report.rows_inserted,
            "rowsUpdated": report.rows_updated,
            "rowsDeleted": report.rows_deleted,
            "rowsSkipped": report.rows_skipped,
            "conflictsKeptLocal": report.conflicts_kept_local,
            "ftsRebuilt": report.fts_rebuilt,
            "message": format!("Imported {} rows from portable archive.", report.rows_imported),
        }));
    }

    // Try parsing as wrapped format first (MCP response wrapper),
    // then fall back to direct RecallResult
    let memories: Vec<MemoryBackup> =
        if let Ok(wrapper) = serde_json::from_str::<Vec<BackupWrapper>>(&backup_content) {
            if let Some(first) = wrapper.first() {
                let recall: RecallResult = serde_json::from_str(&first.text)
                    .map_err(|e| format!("Failed to parse backup contents: {}", e))?;
                recall.results
            } else {
                return Err("Empty backup file".to_string());
            }
        } else if let Ok(recall) = serde_json::from_str::<RecallResult>(&backup_content) {
            recall.results
        } else if let Ok(nodes) = serde_json::from_str::<Vec<MemoryBackup>>(&backup_content) {
            nodes
        } else {
            return Err(
            "Unrecognized backup format. Expected MCP wrapper, RecallResult, or array of memories."
                .to_string(),
        );
        };

    let total = memories.len();
    if total == 0 {
        return Ok(serde_json::json!({
            "tool": "restore",
            "success": true,
            "restored": 0,
            "total": 0,
            "message": "No memories found in backup file.",
        }));
    }

    let mut success_count = 0_usize;
    let mut error_count = 0_usize;

    for memory in &memories {
        let input = IngestInput {
            content: memory.content.clone(),
            node_type: memory
                .node_type
                .clone()
                .unwrap_or_else(|| "fact".to_string()),
            source: memory.source.clone(),
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: memory.tags.clone().unwrap_or_default(),
            valid_from: None,
            valid_until: None,
            validity_inferred: false,
            source_envelope: None,
        };

        match storage.ingest(input) {
            Ok(_) => success_count += 1,
            Err(_) => error_count += 1,
        }
    }

    Ok(serde_json::json!({
        "tool": "restore",
        "success": true,
        "restored": success_count,
        "errors": error_count,
        "total": total,
        "message": format!("Restored {}/{} memories from backup.", success_count, total),
    }))
}

fn ensure_restore_path_allowed(storage: &Storage, path: &Path) -> Result<(), String> {
    let canonical_path = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve restore path: {}", e))?;

    for dir in [
        storage.sidecar_dir("exports"),
        storage.sidecar_dir("backups"),
    ] {
        if !dir.exists() {
            continue;
        }
        let canonical_dir = dir
            .canonicalize()
            .map_err(|e| format!("Failed to resolve allowed restore directory: {}", e))?;
        if canonical_path.starts_with(&canonical_dir) {
            return Ok(());
        }
    }

    Err(format!(
        "MCP restore is restricted to {} and {} by default. Pass allowAnyPath=true only for trusted local files.",
        storage.sidecar_dir("exports").display(),
        storage.sidecar_dir("backups").display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    fn write_temp_file(dir: &TempDir, name: &str, content: &str) -> String {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_schema_has_required_fields() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["path"].is_object());
        assert!(s["properties"]["allowAnyPath"].is_object());
        assert!(
            s["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("path"))
        );
    }

    #[tokio::test]
    async fn test_missing_args_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_missing_path_field_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, Some(serde_json::json!({}))).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    #[tokio::test]
    async fn test_nonexistent_file_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "path": "/tmp/does_not_exist_vestige_test.json" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_malformed_json_fails() {
        let (storage, dir) = test_storage().await;
        let path = write_temp_file(&dir, "bad.json", "this is not json {{{");
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unrecognized backup format"));
    }

    #[tokio::test]
    async fn test_restore_rejects_arbitrary_path_by_default() {
        let (storage, dir) = test_storage().await;
        let path = write_temp_file(&dir, "outside_exports.json", "[]");
        let args = serde_json::json!({ "path": path });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("restricted"));
    }

    #[tokio::test]
    async fn test_restore_direct_array_format() {
        let (storage, dir) = test_storage().await;
        let backup = serde_json::json!([
            { "content": "Memory one", "nodeType": "fact", "tags": ["test"] },
            { "content": "Memory two", "nodeType": "concept" }
        ]);
        let path = write_temp_file(&dir, "backup.json", &backup.to_string());
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["tool"], "restore");
        assert_eq!(value["success"], true);
        assert_eq!(value["restored"], 2);
        assert_eq!(value["errors"], 0);
        assert_eq!(value["total"], 2);
    }

    #[tokio::test]
    async fn test_restore_recall_result_format() {
        let (storage, dir) = test_storage().await;
        let backup = serde_json::json!({
            "results": [
                { "content": "Recall memory one" },
                { "content": "Recall memory two" },
                { "content": "Recall memory three" }
            ]
        });
        let path = write_temp_file(&dir, "recall.json", &backup.to_string());
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["restored"], 3);
        assert_eq!(value["total"], 3);
    }

    #[tokio::test]
    async fn test_restore_portable_archive_from_allowed_exports_dir() {
        let (source, _source_dir) = test_storage().await;
        let (target, _target_dir) = test_storage().await;

        source
            .ingest(vestige_core::IngestInput {
                content: "Portable MCP restore test memory".to_string(),
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

        let export_dir = target.sidecar_dir("exports");
        std::fs::create_dir_all(&export_dir).unwrap();
        let archive_path = export_dir.join("portable-restore.json");
        source
            .export_portable_archive_to_path(&archive_path)
            .unwrap();

        let args = serde_json::json!({ "path": archive_path });
        let result = execute(&target, Some(args)).await.unwrap();

        assert_eq!(result["tool"], "restore");
        assert_eq!(result["success"], true);
        assert_eq!(result["mode"], "portable");
        assert!(result["rowsImported"].as_u64().unwrap() > 0);
        assert_eq!(target.get_stats().unwrap().total_nodes, 1);
    }

    #[tokio::test]
    async fn test_restore_empty_results_array() {
        let (storage, dir) = test_storage().await;
        let backup = serde_json::json!({ "results": [] });
        let path = write_temp_file(&dir, "empty.json", &backup.to_string());
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["restored"], 0);
        assert_eq!(value["total"], 0);
    }

    #[tokio::test]
    async fn test_restore_empty_array_returns_error() {
        // Empty [] parses as Vec<BackupWrapper> first, which has no items → "Empty backup file"
        let (storage, dir) = test_storage().await;
        let path = write_temp_file(&dir, "empty_arr.json", "[]");
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Empty backup file"));
    }

    #[tokio::test]
    async fn test_restore_defaults_node_type_to_fact() {
        let (storage, dir) = test_storage().await;
        let backup = serde_json::json!([{ "content": "No type specified" }]);
        let path = write_temp_file(&dir, "notype.json", &backup.to_string());
        let args = serde_json::json!({ "path": path, "allowAnyPath": true });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["restored"], 1);
    }
}
