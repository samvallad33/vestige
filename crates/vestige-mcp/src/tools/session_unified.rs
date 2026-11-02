//! `session` MCP tool — Boot, Intentions, Status & Projection.
//!
//! Absorbs: `session_start`, `intention`, `memory_status(stats|health)`, `project`.
//! Actions: "start" | "status" | "set_intention" | "check_intentions" | "complete_intention" | "project".

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
use vestige_core::Storage;

use crate::cognitive::CognitiveEngine;
use crate::tools;

/// Native schemas are embedded below the six-tool envelope. JSON Pointer refs
/// are document-root-relative, so retain their definitions and rebase the refs
/// to the embedded location (including recursive refs) rather than inlining.
pub(crate) fn embedded_schema(mut schema: Value, pointer: &str) -> Value {
    fn visit(value: &mut Value, pointer: &str) {
        match value {
            Value::Object(object) => {
                if let Some(Value::String(reference)) = object.get_mut("$ref") {
                    if reference == "#" || reference.starts_with("#/") {
                        *reference = format!("#{pointer}{}", &reference[1..]);
                    }
                }
                for child in object.values_mut() {
                    visit(child, pointer);
                }
            }
            Value::Array(values) => {
                for child in values {
                    visit(child, pointer);
                }
            }
            _ => {}
        }
    }
    visit(&mut schema, pointer);
    schema
}

pub fn schema() -> Value {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "reflex", "intention", "codebase", "set_intention", "check_intentions", "complete_intention", "project"],
                "default": "status",
                "description": "'start': session boot context; 'status': memory graph health & stats; 'set_intention'/'check_intentions'/'complete_intention': manage session intentions; 'project': write CLAUDE.md / AGENTS.md fences directly."
            },
            "params": {"type":"object", "description":"For reflex, intention, or codebase: the complete operation payload, including its own action. Full per-action schemas are available through session status view=tools tool=session."},
            "cwd": { "type": "string", "description": "Current working directory for session start or projection." },
            "repo_tag": { "type": "boolean", "default": false, "description": "For 'start': also resolve repository name tag." },
            "view": { "type": "string", "enum": ["system", "health", "tools", "provenance", "changelog"], "description": "For 'status': detail view to return (default: system)." },
            "tool": { "type": "string", "description": "For status view tools: exact advertised tool whose full schema to inspect." },
            "intention": { "type": "string", "description": "For 'set_intention': what you intend to do." },
            "id": { "type": "string", "description": "For 'complete_intention': intention id." },
            "format": { "type": "string", "enum": ["claude-md", "memory-md"], "default": "claude-md", "description": "For 'project': format of rule fence." },
            "path": { "type": "string", "description": "For 'project': target rule file path (e.g. CLAUDE.md)." },
            "write": { "type": "boolean", "default": false, "description": "For 'project': apply write directly if true, preview if false." },
            "confirm": { "type": "boolean", "default": true, "description": "For 'project' write: confirm changes." },
            "min_retention": { "type": "number", "default": 0.3, "description": "For 'project': min retention threshold." },
            "max_items": { "type": "integer", "default": 60, "description": "Max items to project or return." }
        }
    });
    // Keep the boot and hook contract discoverable without changing the cwd
    // rules/projection route. The compact catalog retains these root fields.
    for (name, value) in tools::session_context::schema()["properties"]
        .as_object()
        .unwrap()
    {
        schema["properties"][name] = value.clone();
    }
    for (name, value) in tools::memory_status::schema()["properties"]
        .as_object()
        .unwrap()
    {
        if schema["properties"].get(name).is_none() || name == "view" {
            schema["properties"][name] = value.clone();
        }
    }
    if let Some(views) = schema["properties"]["view"]["enum"].as_array_mut() {
        views.push(json!("system"));
        views.push(json!("memory_prs"));
    }
    schema["properties"]["status"] =
        json!({"type":"string", "description":"For memory_prs: native review status filter."});
    schema["properties"]["limit"] = json!({"type":"integer", "minimum":1, "maximum":500, "description":"For memory_prs: maximum review rows."});
    for field in ["session", "host", "event", "source"] {
        schema["properties"]["memory_hook"]["properties"][field] = json!({"type":"string"});
    }
    schema["allOf"] = json!([
        {"if":{"properties":{"action":{"const":"reflex"}},"required":["action"]},
         "then":{"properties":{"params":embedded_schema(tools::reflex::schema(), "/allOf/0/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"intention"}},"required":["action"]},
         "then":{"properties":{"params":embedded_schema(tools::intention_graph::schema(), "/allOf/1/then/properties/params")},"required":["params"]}},
        {"if":{"properties":{"action":{"const":"codebase"}},"required":["action"]},
         "then":{"properties":{"params":embedded_schema(tools::codebase_unified::schema(), "/allOf/2/then/properties/params")},"required":["params"]}}
    ]);
    schema
}

/// The cwd/rule start remains compatible; explicit boot fields select the
/// context packet, which is also the memory hook's native implementation.
pub(crate) fn uses_context_packet(args: &Value) -> bool {
    [
        "memory_hook",
        "context",
        "include_status",
        "include_intentions",
        "include_predictions",
        "queries",
        "token_budget",
        "scope",
        "changed_files",
        "changedFiles",
    ]
    .iter()
    .any(|field| args.get(*field).is_some())
}

/// Same queue contract as the local dashboard. Strata explicitly reports
/// unavailability; it must never be mistaken for a successfully empty queue.
pub(crate) fn memory_prs(storage: &Arc<Storage>, args: &Value) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return Ok(json!({
            "total":0, "pendingCount":0, "mode":vestige_core::ReviewMode::Fast.as_str(),
            "prs":[], "available":false,
            "reason":"Memory PR review is not available on a Strata log in Vestige 4.0. Every write passes the log's gate and leaves a receipt instead."
        }));
    }
    let limit = args["limit"].as_u64().unwrap_or(100).clamp(1, 500) as usize;
    let status = args
        .get("status")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let prs = storage
        .list_memory_prs(status, limit)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "total":prs.len(), "pendingCount":storage.count_pending_memory_prs().unwrap_or(0),
        "mode":crate::trace_recorder::read_review_mode(storage).as_str(), "prs":prs
    }))
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
        Some(v) => {
            serde_json::from_value(v).map_err(|e| format!("Invalid session arguments: {e}"))?
        }
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
            let effective_cwd = cwd_path
                .or(fallback.as_deref())
                .unwrap_or(std::path::Path::new("."));
            let tags = crate::rules_hook::rule_tags(effective_cwd, args.repo_tag);
            let mut resolved = Vec::new();
            for tag in &tags {
                let val = tools::recall::execute(
                    storage,
                    cognitive,
                    &vestige_core::OutputConfig::default(),
                    Some(json!({ "handle": tag })),
                )
                .await
                .unwrap_or_default();
                resolved.push((tag.clone(), crate::rules_hook::nodes_from_recall(&val)));
            }
            let limits = crate::rules_hook::Limits {
                max_items: args
                    .max_items
                    .unwrap_or(crate::rules_hook::DEFAULT_MAX_ITEMS),
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
            let view = args
                .view
                .filter(|v| v != "system")
                .unwrap_or_else(|| "health".to_string());
            tools::memory_status::execute(
                storage,
                cognitive,
                &vestige_core::OutputConfig::default(),
                Some(json!({ "view": view })),
            )
            .await
        }
        "set_intention" => {
            let desc = args.intention.ok_or("set_intention requires 'intention'")?;
            tools::intention_unified::execute(
                storage,
                cognitive,
                Some(json!({ "action": "set", "description": desc })),
            )
            .await
        }
        "check_intentions" => {
            tools::intention_unified::execute(
                storage,
                cognitive,
                Some(json!({ "action": "check" })),
            )
            .await
        }
        "complete_intention" => {
            let id = args.id.ok_or("complete_intention requires 'id'")?;
            tools::intention_unified::execute(
                storage,
                cognitive,
                Some(json!({ "action": "update", "status": "complete", "id": id })),
            )
            .await
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
            )
            .await
        }
        other => Err(format!(
            "unknown session action '{other}'; use start, status, set_intention, check_intentions, complete_intention, or project"
        )),
    }
}
