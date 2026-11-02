//! `session` MCP tool — Boot, Intentions, Status & Projection.
//!
//! Absorbs: `session_start`, `intention`, `memory_status(stats|health)`, `project`.
//! Actions: "start" | "status" | "set_intention" | "check_intentions" | "complete_intention" | "project".

use std::sync::Arc;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "set_intention", "check_intentions", "complete_intention", "project"],
                "default": "status",
                "description": "'start': session boot context; 'status': memory graph health & stats; 'set_intention'/'check_intentions'/'complete_intention': manage session intentions; 'project': write CLAUDE.md / AGENTS.md fences directly."
            },
            "cwd": { "type": "string", "description": "Current working directory for session start or projection." },
            "repo_tag": { "type": "boolean", "default": false, "description": "For 'start': also resolve repository name tag." },
            "view": { "type": "string", "enum": ["system", "health", "tools", "provenance", "changelog"], "description": "For 'status': detail view to return (default: system)." },
            "intention": { "type": "string", "description": "For 'set_intention': what you intend to do." },
            "id": { "type": "string", "description": "For 'complete_intention': intention id." },
            "format": { "type": "string", "enum": ["claude-md", "memory-md"], "default": "claude-md", "description": "For 'project': format of rule fence." },
            "path": { "type": "string", "description": "For 'project': target rule file path (e.g. CLAUDE.md)." },
            "write": { "type": "boolean", "default": false, "description": "For 'project': apply write directly if true, preview if false." },
            "confirm": { "type": "boolean", "default": true, "description": "For 'project' write: confirm changes." },
            "min_retention": { "type": "number", "default": 0.3, "description": "For 'project': min retention threshold." },
            "max_items": { "type": "integer", "default": 60, "description": "Max items to project or return." }
        }
    })
}

#[derive(Debug, Deserialize)]
struct SessionArgs {
    #[serde(default = "default_action")]
    action: String,
    cwd: Option<String>,
    #[serde(default)]
    repo_tag: bool,
    view: Option<String>,
    intention: Option<String>,
    id: Option<String>,
    format: Option<String>,
    path: Option<String>,
    #[serde(default)]
    write: bool,
    #[serde(default = "default_confirm")]
    confirm: bool,
    min_retention: Option<f64>,
    max_items: Option<usize>,
}

fn default_action() -> String {
    "status".to_string()
}

fn default_confirm() -> bool {
    true
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SessionArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid session arguments: {e}"))?,
        None => SessionArgs {
            action: default_action(),
            cwd: None,
            repo_tag: false,
            view: None,
            intention: None,
            id: None,
            format: None,
            path: None,
            write: false,
            confirm: true,
            min_retention: None,
            max_items: None,
        },
    };

    match args.action.as_str() {
        "start" => {
            let cwd_path = args.cwd.as_deref().map(std::path::Path::new);
            let fallback = std::env::current_dir().ok();
            let effective_cwd = cwd_path.or(fallback.as_deref()).unwrap_or(std::path::Path::new("."));
            let tags = crate::rules_hook::rule_tags(effective_cwd, args.repo_tag);
            let mut resolved = Vec::new();
            for tag in &tags {
                let val = tools::recall::execute(
                    storage,
                    cognitive,
                    &vestige_core::OutputConfig::default(),
                    Some(json!({ "handle": tag })),
                ).await.unwrap_or_default();
                resolved.push((tag.clone(), crate::rules_hook::nodes_from_recall(&val)));
            }
            let limits = crate::rules_hook::Limits {
                max_items: args.max_items.unwrap_or(crate::rules_hook::DEFAULT_MAX_ITEMS),
                max_bytes: crate::rules_hook::DEFAULT_MAX_BYTES,
            };
            let context = crate::rules_hook::render_context(&resolved, "startup", limits);
            Ok(json!({
                "action": "start",
                "cwd": effective_cwd.display().to_string(),
                "resolved_tags": tags,
                "context": context,
            }))
        }
        "status" => {
            let view = args.view.unwrap_or_else(|| "system".to_string());
            tools::memory_status::execute(
                storage,
                cognitive,
                Some(json!({ "view": view })),
            ).await
        }
        "set_intention" => {
            let desc = args.intention.ok_or("set_intention requires 'intention'")?;
            tools::intention_unified::execute(
                storage,
                Some(json!({ "action": "set", "description": desc })),
            ).await
        }
        "check_intentions" => {
            tools::intention_unified::execute(
                storage,
                Some(json!({ "action": "check" })),
            ).await
        }
        "complete_intention" => {
            let id = args.id.ok_or("complete_intention requires 'id'")?;
            tools::intention_unified::execute(
                storage,
                Some(json!({ "action": "complete", "id": id })),
            ).await
        }
        "project" => {
            let action = if args.write { "write" } else { "preview" };
            let format = args.format.unwrap_or_else(|| "claude-md".to_string());
            let target_path = args.path.unwrap_or_else(|| "CLAUDE.md".to_string());
            tools::project::execute(
                storage,
                Some(json!({
                    "action": action,
                    "format": format,
                    "path": target_path,
                    "confirm": args.confirm,
                    "min_retention": args.min_retention,
                    "max_items": args.max_items,
                })),
            ).await
        }
        other => Err(format!("unknown session action '{other}'; use start, status, set_intention, check_intentions, complete_intention, or project")),
    }
}
