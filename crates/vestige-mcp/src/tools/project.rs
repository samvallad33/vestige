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

/// Largest existing target file the tool will read. Rule files are small
/// text; anything bigger is not a projection target.
const MAX_TARGET_BYTES: u64 = 2 * 1024 * 1024;

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
        // One server process can serve several agents in different projects,
        // so its own working directory says nothing about the caller's.
        None if !Path::new(path).is_absolute() => {
            return Err(format!(
                "path '{path}' is relative and no root was given: pass root (the absolute directory \
                 the file belongs in) or an absolute path. A relative path would resolve against the \
                 memory server's working directory, which may be another project."
            ));
        }
        None => std::env::current_dir()
            .map_err(|e| format!("cannot read the working directory: {e}"))?,
    };
    let root = root
        .canonicalize()
        .map_err(|e| format!("root {} is not a readable directory: {e}", root.display()))?;
    if root.parent().is_none() {
        return Err(
            "root must not be the filesystem root; name the directory projections stay inside"
                .into(),
        );
    }
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
    let target = parent.join(file_name);
    if std::fs::symlink_metadata(&target).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Err("projection target must not be a symlink".into());
    }
    Ok(target)
}

/// Read an existing target: a regular file no larger than `MAX_TARGET_BYTES`.
/// The type and size are checked on the opened handle, and the open does not
/// wait on a pipe, so a special file cannot stall or exhaust the call.
fn read_target(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!(
            "{} is not a regular file; projections only read regular files",
            path.display()
        ));
    }
    if meta.len() > MAX_TARGET_BYTES {
        return Err(format!(
            "{} is too large to project into (limit {MAX_TARGET_BYTES} bytes)",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_TARGET_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > MAX_TARGET_BYTES {
        return Err(format!(
            "{} is too large to project into (limit {MAX_TARGET_BYTES} bytes)",
            path.display()
        ));
    }
    String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8 text", path.display()))
}

fn gate_refused(err: &str) -> bool {
    err.contains("gate_denied") || err.contains("gate_held")
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
        Some(value) => {
            serde_json::from_value(value).map_err(|e| format!("Invalid arguments: {e}"))?
        }
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
    let format = ProjectionFormat::parse(&args.format).ok_or_else(|| {
        format!(
            "unknown format '{}'; use claude-md or memory-md",
            args.format
        )
    })?;
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
        Some(path) if path.exists() => Some(read_target(path)?),
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
            // The gate admits every projected_to edge before any byte of the
            // target file is replaced. A refusal leaves the file alone.
            let receipt = if crate::strata_memory::is_strata_backend(storage.as_ref()) {
                let ids: Vec<String> = projection
                    .items
                    .iter()
                    .map(|item| item.id.clone())
                    .collect();
                match storage.admit_projection(
                    &ids,
                    &path.display().to_string(),
                    projection.region.as_bytes(),
                ) {
                    Ok((receipt_id, hash)) => Some(json!({
                        "receiptId": receipt_id,
                        "hash": hash,
                    })),
                    Err(err) if gate_refused(&err.to_string()) => {
                        return Ok(json!({
                            "action": "write",
                            "written": false,
                            "refused": true,
                            "path": path.display().to_string(),
                            "note": "The gate refused this projection. The file was not changed.",
                        }));
                    }
                    Err(err) => return Err(err.to_string()),
                }
            } else {
                None
            };
            if added == 0 && removed == 0 {
                let mut response = json!({
                    "action": "write",
                    "written": false,
                    "path": path.display().to_string(),
                    "itemCount": projection.items.len(),
                    "added": 0,
                    "removed": 0,
                    "note": "The file already holds this projection; nothing to change.",
                });
                if let Some(receipt) = receipt {
                    response["receipt"] = receipt;
                }
                return Ok(response);
            }
            projection::write_projection(&path, existing.as_deref(), &new_text)
                .map_err(|e| format!("cannot update projection: {e}"))?;
            let mut response = json!({
                "action": "write",
                "written": true,
                "path": path.display().to_string(),
                "bytes": new_text.len(),
                "itemCount": projection.items.len(),
                "added": added,
                "removed": removed,
                "note": "Only the fenced region changed. Re-run preview any time; an unchanged store projects to an unchanged file.",
            });
            if let Some(receipt) = receipt {
                response["receipt"] = receipt;
            }
            Ok(response)
        }
        other => Err(format!("unknown action '{other}'; use preview or write")),
    }
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    fn storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    fn ingest(storage: &Arc<Storage>, content: &str, node_type: &str) -> String {
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
        let unknown = execute(&storage, Some(json!({ "format": "yaml" })))
            .await
            .unwrap_err();
        assert!(unknown.contains("unknown format"), "{unknown}");
    }
}

#[cfg(test)]
mod strata_preview {
    use super::*;
    use chrono::{Duration, Utc};
    use strata_gate::policy::{ANY_KIND, WILDCARD_PREFIX};
    use strata_gate::record::Verdict;
    use strata_gate::{Policy, Rule};
    use vestige_core::IngestInput;

    fn permissive_policy() -> Policy {
        Policy {
            rules: vec![Rule {
                match_kind: ANY_KIND,
                match_params_hash_prefix: WILDCARD_PREFIX,
                max_blast_radius: u32::MAX,
                forbid_forgotten_lessons: false,
                require_human: false,
                verdict: Verdict::Allow,
            }],
        }
    }

    fn blake3_tree(root: &std::path::Path) -> blake3::Hash {
        let mut files = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    files.push(path);
                }
            }
        }
        files.sort();
        let mut hasher = blake3::Hasher::new();
        for path in files {
            let rel = path.strip_prefix(root).unwrap();
            hasher.update(rel.to_string_lossy().as_bytes());
            hasher.update(&[0]);
            hasher.update(&std::fs::read(&path).unwrap());
            hasher.update(&[0]);
        }
        hasher.finalize()
    }

    fn no_sqlite(dir: &std::path::Path) -> bool {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".sqlite")
                    || name.ends_with(".sqlite3")
                    || name.ends_with(".db")
                    || name.ends_with(".db-wal")
                    || name.ends_with(".db-shm")
                {
                    return false;
                }
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        true
    }

    fn ingest(
        storage: &Arc<Storage>,
        content: &str,
        node_type: &str,
        tags: &[&str],
        scope: &str,
        valid_from: Option<chrono::DateTime<Utc>>,
        valid_until: Option<chrono::DateTime<Utc>>,
    ) -> String {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    node_type: node_type.to_string(),
                    tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
                    valid_from,
                    valid_until,
                    ..Default::default()
                },
                scope,
            )
            .unwrap()
            .id
    }

    #[tokio::test]
    async fn preview_reads_the_log_and_leaves_it_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let decision = ingest(
            &storage,
            "Ship releases from an integration branch",
            "decision",
            &[],
            "user",
            None,
            None,
        );
        let superseded = ingest(
            &storage,
            "Old branch policy that was replaced",
            "decision",
            &[],
            "user",
            None,
            None,
        );
        let pattern = ingest(
            &storage,
            "Touch files after scripted edits",
            "pattern",
            &[],
            "user",
            None,
            None,
        );
        let rule = ingest(
            &storage,
            "Prefer tabs in Svelte files",
            "fact",
            &["Preference"],
            "user",
            None,
            None,
        );
        let convention = ingest(
            &storage,
            "Commit messages stay imperative",
            "note",
            &["convention"],
            "user",
            None,
            None,
        );
        let plain = ingest(
            &storage,
            "The office moved in spring",
            "fact",
            &[],
            "user",
            None,
            None,
        );
        let keyword = ingest(
            &storage,
            "A fact that mentions a decision and a preference rule",
            "fact",
            &["notes"],
            "user",
            None,
            None,
        );
        let expired = ingest(
            &storage,
            "Expired decision that was withdrawn",
            "decision",
            &[],
            "user",
            None,
            Some(Utc::now() - Duration::days(1)),
        );
        let future = ingest(
            &storage,
            "Future decision not yet valid",
            "decision",
            &[],
            "user",
            Some(Utc::now() + Duration::days(2)),
            None,
        );
        let other = ingest(
            &storage,
            "Decision that belongs to another scope",
            "decision",
            &[],
            "other-proj",
            None,
            None,
        );
        drop(storage);
        let aged;
        {
            let mut store =
                strata_store::StrataStore::open_with_policy(dir.path(), permissive_policy())
                    .unwrap();
            store.supersede(&superseded, &decision).unwrap();
            // A decision written long ago has decayed with elapsed time.
            aged = store
                .ingest_in_scope(
                    strata_store::IngestInput {
                        content: "Decision written a year ago".into(),
                        source: None,
                        source_updated_at_ms: None,
                        node_type: "decision".into(),
                        tags: Vec::new(),
                        created_at_ms: Some(Utc::now().timestamp_millis() - 365 * 86_400_000),
                        valid_from_ms: None,
                        valid_until_ms: None,
                    },
                    "user",
                )
                .unwrap();
        }
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("CLAUDE.md");
        std::fs::write(&target, "# Mine\n\nKeep me.\n").unwrap();
        let log_before = blake3_tree(dir.path());
        let file_before = blake3::hash(&std::fs::read(&target).unwrap());

        let value = execute(
            &storage,
            Some(json!({ "action": "preview", "path": "CLAUDE.md", "root": root.path() })),
        )
        .await
        .unwrap();

        let log_after = blake3_tree(dir.path());
        let file_after = blake3::hash(&std::fs::read(&target).unwrap());
        assert_eq!(log_before, log_after, "preview must not append to the log");
        assert_eq!(file_before, file_after, "preview must not write the target");
        assert!(no_sqlite(dir.path()));
        assert_eq!(value["action"], "preview");
        assert_eq!(value["scope"], "user");
        assert_eq!(value["format"], "claude-md");
        let ids: Vec<&str> = value["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&decision.as_str()), "{ids:?}");
        assert!(ids.contains(&pattern.as_str()), "{ids:?}");
        assert!(ids.contains(&rule.as_str()), "{ids:?}");
        assert!(ids.contains(&convention.as_str()), "{ids:?}");
        for excluded in [
            &superseded,
            &plain,
            &keyword,
            &expired,
            &future,
            &other,
            &aged,
        ] {
            assert!(!ids.contains(&excluded.as_str()), "{excluded} in {ids:?}");
        }
        let region = value["region"].as_str().unwrap();
        assert!(region.contains(&decision));
        assert!(!region.contains("office moved"));
        assert!(!region.contains("mentions a decision"));
        assert!(!region.contains("Old branch policy"));
        assert_eq!(value["items"][0]["nodeType"], "decision");
        assert!(value["target"]["added"].as_u64().unwrap() > 0);
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "# Mine\n\nKeep me.\n"
        );

        let again = execute(
            &storage,
            Some(json!({ "action": "preview", "path": "CLAUDE.md", "root": root.path() })),
        )
        .await
        .unwrap();
        assert_eq!(again["region"], value["region"]);
        assert_eq!(blake3_tree(dir.path()), log_after);

        let floor = execute(
            &storage,
            Some(json!({ "action": "preview", "min_retention": 1.0 })),
        )
        .await
        .unwrap();
        // Fresh memories meet a floor of 1.0; the year-old decision does not.
        assert_eq!(floor["itemCount"], 4, "{floor}");
        assert_eq!(blake3_tree(dir.path()), log_after);

        let capped = execute(
            &storage,
            Some(json!({ "action": "preview", "max_items": 1 })),
        )
        .await
        .unwrap();
        assert_eq!(capped["itemCount"], 1);
        assert_eq!(capped["items"][0]["id"], decision);
    }

    #[tokio::test]
    async fn preview_rejects_invalid_input() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let before = blake3_tree(dir.path());
        let bad_type = execute(&storage, Some(json!({ "max_items": "lots" })))
            .await
            .unwrap_err();
        assert!(bad_type.contains("Invalid arguments"), "{bad_type}");
        let bad_format = execute(&storage, Some(json!({ "format": "yaml" })))
            .await
            .unwrap_err();
        assert!(bad_format.contains("unknown format"), "{bad_format}");
        let bad_scope = execute(&storage, Some(json!({ "scope": "" })))
            .await
            .unwrap_err();
        assert!(bad_scope.contains("Invalid memory scope"), "{bad_scope}");
        let root = tempfile::tempdir().unwrap();
        let escape = execute(
            &storage,
            Some(json!({ "path": "../escape.md", "root": root.path() })),
        )
        .await
        .unwrap_err();
        assert!(escape.contains("outside"), "{escape}");
        assert_eq!(blake3_tree(dir.path()), before);
        assert!(no_sqlite(dir.path()));
    }

    #[tokio::test]
    async fn preview_rejects_unknown_action() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let before = blake3_tree(dir.path());
        let unknown = execute(&storage, Some(json!({ "action": "frobnicate" })))
            .await
            .unwrap_err();
        assert!(unknown.contains("unknown action 'frobnicate'"), "{unknown}");
        assert_eq!(blake3_tree(dir.path()), before);
    }

    fn deny_policy() -> Policy {
        Policy {
            rules: vec![Rule {
                match_kind: ANY_KIND,
                match_params_hash_prefix: WILDCARD_PREFIX,
                max_blast_radius: u32::MAX,
                forbid_forgotten_lessons: false,
                require_human: false,
                verdict: Verdict::Deny,
            }],
        }
    }

    fn frame_count(dir: &std::path::Path, kind: u8) -> usize {
        let store = strata_store::StrataStore::open(dir).unwrap();
        store
            .log()
            .read_frames(1)
            .unwrap()
            .iter()
            .filter(|frame| frame.kind == kind)
            .count()
    }

    #[tokio::test]
    async fn admitted_write_returns_a_receipt_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let decision = ingest(
            &storage,
            "Ship releases from an integration branch",
            "decision",
            &[],
            "user",
            None,
            None,
        );
        let pattern = ingest(
            &storage,
            "Touch files after scripted edits",
            "pattern",
            &[],
            "user",
            None,
            None,
        );
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("CLAUDE.md");
        let value = execute(
            &storage,
            Some(json!({
                "action": "write",
                "path": "CLAUDE.md",
                "root": root.path(),
                "confirm": true,
            })),
        )
        .await
        .unwrap();
        assert_eq!(value["written"], true, "{value}");
        assert_eq!(value["refused"], Value::Null);
        let hash = value["receipt"]["hash"].as_str().unwrap().to_string();
        let receipt_id = value["receipt"]["receiptId"].as_str().unwrap();
        assert!(receipt_id.starts_with("eff-"), "{receipt_id}");
        let bytes = std::fs::read(&target).unwrap();
        assert_eq!(hash, blake3::hash(&bytes).to_hex().as_str());
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(
            text.contains(&decision) && text.contains(&pattern),
            "{text}"
        );
        drop(storage);

        let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
        let edges: Vec<_> = store
            .edges()
            .into_iter()
            .filter(|edge| edge.link_type == "projected_to")
            .collect();
        assert_eq!(edges.len(), 2, "{edges:?}");
        let sources: Vec<&str> = edges.iter().map(|edge| edge.source_id.as_str()).collect();
        assert!(sources.contains(&decision.as_str()) && sources.contains(&pattern.as_str()));
        assert!(
            edges
                .iter()
                .all(|edge| edge.meta_sha.as_deref() == Some(hash.as_str()))
        );
        // The write records the canonical path (macOS temp dirs sit behind the
        // /var -> /private/var symlink), so compare against the canonical form.
        let canonical_target = target.canonicalize().unwrap().display().to_string();
        assert!(
            edges.iter().all(|edge| edge.target_id == canonical_target),
            "{edges:?}"
        );
        store.seal_checkpoint().unwrap();
        drop(store);
        let report = strata_verify::verify_path(dir.path());
        assert!(report.ok, "{}", report.json);
    }

    #[tokio::test]
    async fn refused_write_leaves_the_file_and_appends_no_effect() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        ingest(
            &storage,
            "Ship releases from an integration branch",
            "decision",
            &[],
            "user",
            None,
            None,
        );
        drop(storage);
        let effects_before =
            frame_count(dir.path(), strata_gate::record::RecordKind::Effect.to_u8());
        let writes_before = frame_count(dir.path(), strata_store::KIND_STORE_WRITE);
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("CLAUDE.md");
        std::fs::write(&target, "# Mine\n\nKeep me.\n").unwrap();
        let before = blake3::hash(&std::fs::read(&target).unwrap());

        let storage: Arc<Storage> = Arc::new(
            crate::strata_memory::StrataMemory::open_with_policy(dir.path(), deny_policy())
                .unwrap(),
        );
        let value = execute(
            &storage,
            Some(json!({
                "action": "write",
                "path": "CLAUDE.md",
                "root": root.path(),
                "confirm": true,
            })),
        )
        .await
        .unwrap();
        assert_eq!(value["written"], false, "{value}");
        assert_eq!(value["refused"], true, "{value}");
        assert!(
            value["note"].as_str().unwrap().contains("refused"),
            "{value}"
        );
        drop(storage);

        assert_eq!(blake3::hash(&std::fs::read(&target).unwrap()), before);
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "# Mine\n\nKeep me.\n"
        );
        assert_eq!(
            frame_count(dir.path(), strata_gate::record::RecordKind::Effect.to_u8()),
            effects_before
        );
        assert_eq!(
            frame_count(dir.path(), strata_store::KIND_STORE_WRITE),
            writes_before
        );
    }

    #[tokio::test]
    async fn preview_still_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        ingest(
            &storage,
            "Ship releases from an integration branch",
            "decision",
            &[],
            "user",
            None,
            None,
        );
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("CLAUDE.md");
        std::fs::write(&target, "# Mine\n").unwrap();
        let log_before = blake3_tree(dir.path());
        let file_before = blake3::hash(&std::fs::read(&target).unwrap());
        let value = execute(
            &storage,
            Some(json!({
                "action": "preview",
                "path": "CLAUDE.md",
                "root": root.path(),
            })),
        )
        .await
        .unwrap();
        assert_eq!(value["action"], "preview");
        assert!(value.get("receipt").is_none());
        assert_eq!(blake3_tree(dir.path()), log_before);
        assert_eq!(blake3::hash(&std::fs::read(&target).unwrap()), file_before);
        drop(storage);
        let store = strata_store::StrataStore::open(dir.path()).unwrap();
        assert!(
            store
                .edges()
                .iter()
                .all(|edge| edge.link_type != "projected_to")
        );
    }

    #[tokio::test]
    async fn oversized_target_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("CLAUDE.md"), vec![b'a'; 8 * 1024 * 1024]).unwrap();
        let err = execute(
            &storage,
            Some(json!({ "path": "CLAUDE.md", "root": root.path() })),
        )
        .await
        .unwrap_err();
        assert!(err.contains("too large"), "{err}");
    }

    #[tokio::test]
    async fn directory_target_is_refused_as_not_a_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("sub")).unwrap();
        let err = execute(
            &storage,
            Some(json!({ "path": "sub", "root": root.path() })),
        )
        .await
        .unwrap_err();
        assert!(err.contains("regular file"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn fifo_target_is_refused_without_blocking() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let root = tempfile::tempdir().unwrap();
        let fifo = root.path().join("CLAUDE.md");
        let c_path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let root_path = root.path().to_path_buf();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let result = runtime.block_on(execute(
                &storage,
                Some(json!({ "path": "CLAUDE.md", "root": root_path })),
            ));
            let _ = tx.send(result);
        });
        let outcome = rx.recv_timeout(std::time::Duration::from_secs(10));
        if outcome.is_err() {
            // Release a blocked reader so the worker thread can finish.
            let _ = std::fs::OpenOptions::new().write(true).open(&fifo);
        }
        let result = outcome.expect("project blocked on a FIFO target");
        let err = result.unwrap_err();
        assert!(err.contains("regular file"), "{err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn filesystem_root_is_not_an_acceptable_root() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let err = execute(&storage, Some(json!({ "path": "dev/null", "root": "/" })))
            .await
            .unwrap_err();
        assert!(err.contains("filesystem root"), "{err}");
    }
}
