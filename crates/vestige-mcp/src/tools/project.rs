//! `project`: render the durable subset of a scope into a client rule file.
//!
//! Preview is the default and writes nothing. Write replaces only the fenced
//! region of the target file and needs `confirm: true`; the target must stay
//! inside `root` (the working directory unless given), so a projection can
//! never land outside the repository the agent is working in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};
use vestige_core::Storage;
use vestige_core::projection::{self, ProjectionFormat, ProjectionOptions};

/// Longest diff excerpt returned in a preview.
const DIFF_LINES: usize = 200;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["preview", "write"],
                "default": "preview",
                "description": "'preview' (default) renders and diffs, writes nothing. 'write' replaces the fenced region in 'path'; needs confirm=true."
            },
            "format": {
                "type": "string",
                "enum": ["claude-md", "memory-md"],
                "default": "claude-md",
                "description": "'claude-md': a grouped section for CLAUDE.md or AGENTS.md. 'memory-md': one line per memory for an index file."
            },
            "scope": { "type": "string", "description": "Project namespace to project (default 'user')." },
            "path": { "type": "string", "description": "Target file, relative to 'root'. Required for write; a preview diffs against it when given." },
            "root": { "type": "string", "description": "Directory the target must stay inside (default: the server's working directory)." },
            "confirm": { "type": "boolean", "default": false, "description": "Required for write. Only the fenced region changes; the rest of the file is kept byte for byte." },
            "min_retention": { "type": "number", "default": 0.3, "minimum": 0, "maximum": 1, "description": "Leave out memories below this retention." },
            "max_items": { "type": "integer", "default": 60, "minimum": 1, "maximum": 500, "description": "At most this many memories." }
        }
    })
}

#[derive(Debug, Deserialize)]
struct ProjectArgs {
    #[serde(default = "default_action")]
    action: String,
    #[serde(default = "default_format")]
    format: String,
    scope: Option<String>,
    path: Option<String>,
    root: Option<String>,
    #[serde(default)]
    confirm: bool,
    min_retention: Option<f64>,
    max_items: Option<usize>,
}

fn default_action() -> String {
    "preview".to_string()
}

fn default_format() -> String {
    "claude-md".to_string()
}

/// Resolve `path` under `root` and refuse anything that escapes it. The file
/// itself may not exist yet, so the check runs on its parent directory.
fn resolve_target(root: Option<&str>, path: &str) -> Result<PathBuf, String> {
    let root = match root {
        Some(dir) => PathBuf::from(dir),
        None => std::env::current_dir().map_err(|e| format!("cannot read the working directory: {e}"))?,
    };
    let root = root
        .canonicalize()
        .map_err(|e| format!("root {} is not a readable directory: {e}", root.display()))?;
    let candidate = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
    };
    let file_name = candidate
        .file_name()
        .ok_or_else(|| format!("{path} has no file name"))?
        .to_owned();
    let parent = candidate.parent().unwrap_or(&root);
    let parent = parent
        .canonicalize()
        .map_err(|e| format!("directory {} does not exist: {e}", parent.display()))?;
    if !parent.starts_with(&root) {
        return Err(format!(
            "{path} resolves outside {}; projections stay inside root",
            root.display()
        ));
    }
    Ok(parent.join(file_name))
}

fn items_json(items: &[projection::ProjectedItem]) -> Vec<Value> {
    items
        .iter()
        .map(|item| {
            json!({
                "id": item.id,
                "nodeType": item.node_type,
                "tags": item.tags,
                "retention": (item.retention * 100.0).round() / 100.0,
            })
        })
        .collect()
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ProjectArgs = match args {
        Some(value) => serde_json::from_value(value).map_err(|e| format!("Invalid arguments: {e}"))?,
        None => ProjectArgs {
            action: default_action(),
            format: default_format(),
            scope: None,
            path: None,
            root: None,
            confirm: false,
            min_retention: None,
            max_items: None,
        },
    };
    let format = ProjectionFormat::parse(&args.format)
        .ok_or_else(|| format!("unknown format '{}'; use claude-md or memory-md", args.format))?;
    let opts = ProjectionOptions {
        scope: args.scope.clone().unwrap_or_else(|| "user".to_string()),
        format,
        min_retention: args.min_retention.unwrap_or(0.3).clamp(0.0, 1.0),
        max_items: args.max_items.unwrap_or(60).clamp(1, 500),
    };
    let projection = projection::project(storage, &opts).map_err(|e| e.to_string())?;

    let target = match args.path.as_deref() {
        Some(path) => Some(resolve_target(args.root.as_deref(), path)?),
        None => None,
    };
    let existing = match &target {
        Some(path) if path.exists() => {
            Some(std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?)
        }
        _ => None,
    };
    let new_text = projection::splice(existing.as_deref().unwrap_or(""), &projection.region);
    let diff = projection::line_diff(existing.as_deref().unwrap_or(""), &new_text);
    let (added, removed) = projection::diff_summary(&diff);

    match args.action.as_str() {
        "preview" => {
            let mut response = json!({
                "action": "preview",
                "format": format.label(),
                "scope": opts.scope,
                "itemCount": projection.items.len(),
                "items": items_json(&projection.items),
                "region": projection.region,
                "nextStep": "Call again with action='write', the same path, and confirm=true to apply. Only the fenced region changes.",
            });
            if let Some(path) = &target {
                response["target"] = json!({
                    "path": path.display().to_string(),
                    "exists": existing.is_some(),
                    "added": added,
                    "removed": removed,
                    "diff": projection::unified(&diff, DIFF_LINES),
                });
            }
            Ok(response)
        }
        "write" => {
            let path = target.ok_or("write needs 'path'")?;
            if !args.confirm {
                return Err(
                    "Preview first, then pass confirm=true to write. Only the fenced region changes."
                        .to_string(),
                );
            }
            if added == 0 && removed == 0 {
                return Ok(json!({
                    "action": "write",
                    "written": false,
                    "path": path.display().to_string(),
                    "itemCount": projection.items.len(),
                    "added": 0,
                    "removed": 0,
                    "note": "The file already holds this projection; nothing to change.",
                }));
            }
            let tmp = path.with_extension("vestige-projection.tmp");
            std::fs::write(&tmp, &new_text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &path).map_err(|e| format!("cannot replace {}: {e}", path.display()))?;
            Ok(json!({
                "action": "write",
                "written": true,
                "path": path.display().to_string(),
                "bytes": new_text.len(),
                "itemCount": projection.items.len(),
                "added": added,
                "removed": removed,
                "note": "Only the fenced region changed. Re-run preview any time; an unchanged store projects to an unchanged file.",
            }))
        }
        other => Err(format!("unknown action '{other}'; use preview or write")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    fn storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    fn ingest(storage: &Storage, content: &str, node_type: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: node_type.to_string(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    #[tokio::test]
    async fn preview_writes_nothing_and_shows_the_diff() {
        let (storage, _db) = storage();
        let decision = ingest(&storage, "Release from an integration branch", "decision");
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("CLAUDE.md"), "# Mine\n\nKeep me.\n").unwrap();

        let value = execute(
            &storage,
            Some(json!({ "path": "CLAUDE.md", "root": root.path() })),
        )
        .await
        .unwrap();
        assert_eq!(value["action"], "preview");
        assert_eq!(value["itemCount"], 1);
        assert!(value["region"].as_str().unwrap().contains(&decision));
        assert_eq!(value["target"]["exists"], true);
        assert!(value["target"]["added"].as_u64().unwrap() > 0);
        assert_eq!(
            std::fs::read_to_string(root.path().join("CLAUDE.md")).unwrap(),
            "# Mine\n\nKeep me.\n",
            "preview must not touch the file"
        );
    }

    #[tokio::test]
    async fn write_needs_confirm_then_replaces_only_the_fence_and_is_idempotent() {
        let (storage, _db) = storage();
        let pattern = ingest(&storage, "Touch files after scripted edits", "pattern");
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("CLAUDE.md"), "# Mine\n\nKeep me.\n").unwrap();

        let refused = execute(
            &storage,
            Some(json!({ "action": "write", "path": "CLAUDE.md", "root": root.path() })),
        )
        .await
        .unwrap_err();
        assert!(refused.contains("confirm"), "{refused}");

        let written = execute(
            &storage,
            Some(json!({ "action": "write", "path": "CLAUDE.md", "root": root.path(), "confirm": true })),
        )
        .await
        .unwrap();
        assert_eq!(written["written"], true, "{written}");
        let file = std::fs::read_to_string(root.path().join("CLAUDE.md")).unwrap();
        assert!(file.starts_with("# Mine\n\nKeep me.\n"), "{file}");
        assert!(file.contains(projection::BEGIN_MARKER) && file.contains(&pattern));

        let again = execute(
            &storage,
            Some(json!({ "action": "write", "path": "CLAUDE.md", "root": root.path(), "confirm": true })),
        )
        .await
        .unwrap();
        assert_eq!(again["written"], false, "{again}");
        assert_eq!(again["added"], 0);
    }

    #[tokio::test]
    async fn targets_outside_root_are_refused() {
        let (storage, _db) = storage();
        let root = tempfile::tempdir().unwrap();
        let escape = execute(
            &storage,
            Some(json!({ "action": "write", "path": "../escape.md", "root": root.path(), "confirm": true })),
        )
        .await
        .unwrap_err();
        assert!(escape.contains("outside"), "{escape}");
        let unknown = execute(&storage, Some(json!({ "format": "yaml" }))).await.unwrap_err();
        assert!(unknown.contains("unknown format"), "{unknown}");
    }
}
