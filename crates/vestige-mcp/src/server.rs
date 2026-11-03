//! MCP Server Core
//!
//! Handles the main MCP server logic, routing requests to appropriate
//! tool and resource handlers.

use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, warn};

use crate::cognitive::CognitiveEngine;
use crate::dashboard::events::VestigeEvent;
use crate::protocol::messages::{
    CallToolRequest, CallToolResult, InitializeRequest, InitializeResult, ListResourcesResult,
    ListToolsResult, ReadResourceRequest, ReadResourceResult, ResourceDescription,
    ServerCapabilities, ServerInfo, ToolAnnotations, ToolDescription,
};
use crate::protocol::types::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, LEGACY_PROTOCOL_VERSIONS, MCP_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
};
use crate::resources;
use crate::tools;
use vestige_core::{OutputConfig, Storage, VestigeConfig};

const CANONICAL_TOOLS: [&str; 6] = [
    "session",
    "remember",
    "recall",
    "causal_walk",
    "curate",
    "precedent",
];
static ALIAS_AUDIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Migration audit only: never persist arguments, prompts, or credentials.
// Separate from Strata evidence receipts; these operational receipts are unsigned.
fn audit_legacy_call(storage: &Storage, name: &str) {
    if CANONICAL_TOOLS.contains(&name) {
        return;
    }
    let receipt = serde_json::json!({
        "receipt_id":format!("legacy-{}", uuid::Uuid::new_v4()),
        "timestamp":Utc::now().to_rfc3339(), "pid":std::process::id(),
        "tool":name, "event":"legacy_tool_call", "signed":false,
        "message":"Non-canonical MCP name called; migrate to the six-tool catalog."
    });
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let _guard = ALIAS_AUDIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(storage.data_dir().join("legacy-tool-calls.jsonl"))?;
        writeln!(file, "{receipt}")?;
        file.sync_data()
    })();
    match result {
        Ok(()) => warn!(tool = name, receipt_id = %receipt["receipt_id"], "Legacy MCP tripwire"),
        Err(error) => warn!(tool = name, %error, "Legacy MCP tripwire persistence failed"),
    }
}

fn migration_runtime_status(storage: &Storage) -> serde_json::Value {
    use std::io::{BufRead, BufReader};
    let path = storage.data_dir().join("legacy-tool-calls.jsonl");
    let mut counts = std::collections::BTreeMap::<String, u64>::new();
    let mut first = None;
    let mut last = None;
    let mut read_error = None;
    let _guard = ALIAS_AUDIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    match std::fs::File::open(&path) {
        Ok(file) => {
            for line in BufReader::new(file).lines() {
                match line.and_then(|l| {
                    serde_json::from_str::<serde_json::Value>(&l).map_err(std::io::Error::other)
                }) {
                    Ok(row) => {
                        *counts
                            .entry(row["tool"].as_str().unwrap_or("unknown").to_owned())
                            .or_default() += 1;
                        if first.is_none() {
                            first = Some(row["timestamp"].clone());
                        }
                        last = Some(row["timestamp"].clone());
                    }
                    Err(e) => {
                        read_error = Some(e.to_string());
                        break;
                    }
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => read_error = Some(e.to_string()),
    }
    serde_json::json!({
        "pid":std::process::id(), "version":env!("CARGO_PKG_VERSION"),
        "executable":std::env::current_exe().ok(),
        "legacy_aliases":{"total":counts.values().sum::<u64>(), "by_tool":counts,
            "first_call":first,"last_call":last,"receipts_path":path,"read_error":read_error,
            "receipt_kind":"unsigned operational warning; includes unknown non-canonical names"}
    })
}

/// A `memory` get that carries a non-empty `reason` and found its memory: the
/// agent is showing that memory to its user. Returns the memory id (as the
/// store resolved it) and the reason.
fn shown_memory(
    tool_name: &str,
    args: &Option<serde_json::Value>,
    result: &serde_json::Value,
) -> Option<(String, String)> {
    if tool_name != "memory" {
        return None;
    }
    let args = args.as_ref()?;
    if args.get("action").and_then(|v| v.as_str()) != Some("get") {
        return None;
    }
    let reason = args.get("reason").and_then(|v| v.as_str())?.trim();
    if reason.is_empty() {
        return None;
    }
    let node = result.get("node")?;
    let id = node
        .get("id")
        .and_then(|v| v.as_str())
        .or_else(|| args.get("id").and_then(|v| v.as_str()))?;
    Some((id.to_string(), reason.chars().take(2000).collect()))
}

/// The dashboard's port for links handed to agents: the one this process is
/// serving, else VESTIGE_DASHBOARD_PORT, else 3927.
fn dashboard_port() -> u16 {
    let serving = crate::dashboard::SERVING_PORT.load(Ordering::Relaxed);
    if serving > 0 {
        return serving;
    }
    std::env::var("VESTIGE_DASHBOARD_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3927)
}

/// Build the MCP `instructions` string injected into every connecting client's
/// system prompt.
///
/// Default ("minimal", a few sentences) is safe for any user: competitive coders,
/// hobbyists saving recipes, Rails devs saving bug fixes, enterprise deployments.
/// It earns its per-session token cost by telling the client *how* to use
/// Vestige without imposing one maintainer's workflow on strangers.
///
/// The "full" variant is the composition mandate that enforces the
/// Composing / Never-composed / Recommendation response shape. It can misfire
/// on trivial retrievals for a general audience, so it is opt-in via
/// `VESTIGE_SYSTEM_PROMPT_MODE=full`.
///
/// Anything other than `full` falls back to minimal.
fn build_instructions() -> String {
    let mode = std::env::var("VESTIGE_SYSTEM_PROMPT_MODE").unwrap_or_default();
    let mut instructions = if mode.eq_ignore_ascii_case("full") {
        "Vestige is the Causal Proof Engine and the operating system for AI agents: zero vectors, \
         zero RAG, no lookalike text. \
         Every retrieval MUST be composed into a recommendation, never summarized.\
         \n\nCOMPOSITION MANDATE: When you receive memories from recall or ghostlink, your response MUST follow this shape. \
         (a) Composing: [memory IDs], followed by a brief composition rationale \
         about how the memories relate, NOT a restatement of their contents). \
         (b) Never-composed detected: list pairs curate(action='bridge') proves from recorded \
         edges (bridge) or forces with no recorded relation (divergent), or write 'None.' \
         (c) Recommendation: what the user should DO, as a concrete executable action. \
         If your draft begins 'Memory A says X. Memory B says Y.' STOP and rewrite.\
         \n\nBLOCKING PHRASE: If retrieved high-trust memories (retention > 0.7, reps > 0) \
         contradict what you were about to say, start your response with 'Vestige is blocking this:' \
         and surface the contradiction before proceeding. FSRS scores reflect memory history, not truth; verify against current evidence.\
         \n\nFEEDBACK: If the user confirms a memory was helpful, call curate(action='promote'). \
         If they correct it, call curate(action='demote'). Do not ask permission, just act."
            .to_string()
    } else {
        "Vestige is the Causal Proof Engine and the operating system for AI agents. Its kernel, \
         Strata, is an append-only, hash-chained, signed log: zero vectors, zero RAG, \
         no lookalike text. Memories are found by exact handle (a full id or an exact tag) and every \
         result carries its proof (memory id, edge path, receipt). \
         Compose retrievals into recommendations \
         rather than listing their contents when the user is making a decision. \
         On user feedback, call curate(action='promote') for helpful retrievals and \
         curate(action='demote') for wrong ones — do not ask permission, just act."
            .to_string()
    };
    instructions.push_str("\nUse session for boot, intentions and status; remember for gated saves and inline salience receipts; recall for exact handles or ids; causal_walk for recorded evidence; curate for memory changes; precedent for receipts and verification.");
    instructions.push_str("\nDiscover actions with session(action='status', view='tools'); pass tool='<name>' for its exact schema. Choose calls that serve the task; no tool-call quota is required.");
    instructions.push_str("\nUse recall(id='<memory-id>') to read a memory. Native stability is a ranking signal, not proof of truth. Verify retrieved claims against current evidence.");
    instructions
}

fn supported_protocol_versions() -> &'static [&'static str] {
    crate::protocol::types::SUPPORTED_PROTOCOL_VERSIONS
}

/// Which protocol era — and handshake requirement — a request speaks.
///
/// The 2026-07-28 revision removed the handshake and moved version identity
/// onto every request (`_meta["io.modelcontextprotocol/protocolVersion"]`),
/// so the era is a per-request property, not connection state:
///
/// * [`Era::Modern`] — the request declares the modern revision. Served
///   statelessly: the `initialize` flag is ignored, the request must
///   validate its own metadata, and every result gets the modern envelope
///   (`resultType`, `_meta.serverInfo`).
/// * [`Era::LegacyDeclared`] — the request declares a LEGACY revision through
///   the modern `_meta` mechanism (a mixed client). Served statelessly — it
///   declared its own revision, so no handshake gate — but with legacy result
///   shape: a client on an older revision must not receive fields its
///   revision does not define (`resultType` above all).
/// * [`Era::LegacyHandshake`] — no declared version, or the request is
///   `initialize` itself (a dual-era server selects legacy semantics from
///   that method name per the 2026-07-28 versioning page). Requires the
///   handshake, exactly as before this branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Era {
    LegacyHandshake,
    LegacyDeclared,
    Modern,
}

/// The `_meta` key that carries the per-request protocol version.
const META_PROTOCOL_VERSION_KEY: &str = "io.modelcontextprotocol/protocolVersion";
/// The `_meta` key that carries the client's capabilities per request.
const META_CLIENT_CAPABILITIES_KEY: &str = "io.modelcontextprotocol/clientCapabilities";
/// The result `_meta` key carrying server identity without connection state.
const META_SERVER_INFO_KEY: &str = "io.modelcontextprotocol/serverInfo";

/// Classify an incoming request by protocol era (see [`Era`]).
fn request_era(request: &JsonRpcRequest) -> Era {
    // A dual-era server selects its behavior from how the client opens:
    // `initialize` always selects legacy handshake semantics, even when the
    // envelope happens to carry modern `_meta`.
    if request.method == "initialize" {
        return Era::LegacyHandshake;
    }
    let meta = request.params.as_ref().and_then(|p| p.get("_meta"));
    let declared_version = meta.and_then(|m| m.get(META_PROTOCOL_VERSION_KEY));
    match declared_version.and_then(|v| v.as_str()) {
        None => {
            // A `_meta` block carrying the reserved per-request keys without
            // a version is still a modern attempt — a legacy client sends no
            // `_meta` at all — and must fail modern validation with -32602
            // ("missing required field"), not the legacy "server not
            // initialized" gate.
            if meta.is_some_and(|m| m.get(META_CLIENT_CAPABILITIES_KEY).is_some()) {
                Era::Modern
            } else {
                Era::LegacyHandshake
            }
        }
        Some(version) => {
            if LEGACY_PROTOCOL_VERSIONS.contains(&version) {
                // A known legacy revision declared per-request: serve that
                // revision's shape, statelessly.
                Era::LegacyDeclared
            } else {
                // The modern revision itself, or an unknown one —
                // validate_modern_meta turns unknown into -32022 with the
                // supported list.
                Era::Modern
            }
        }
    }
}

/// Validate the required per-request metadata of a modern (2026-07-28)
/// request and return the requested protocol version.
///
/// Modern requests MUST carry `io.modelcontextprotocol/protocolVersion` and
/// `io.modelcontextprotocol/clientCapabilities` in `_meta`; a request missing
/// either is malformed and MUST be rejected with `-32602`. A version the
/// server does not serve (unknown, or known-but-declined) MUST be rejected
/// with `-32022` listing the versions it does support, so the client can
/// retry with a mutually supported revision.
fn validate_modern_meta(request: &JsonRpcRequest) -> Result<String, JsonRpcError> {
    let meta = request
        .params
        .as_ref()
        .and_then(|p| p.get("_meta"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let version = meta
        .get(META_PROTOCOL_VERSION_KEY)
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            JsonRpcError::invalid_params(
                "modern requests must carry a string \
                 _meta['io.modelcontextprotocol/protocolVersion']",
            )
        })?;

    if meta.get(META_CLIENT_CAPABILITIES_KEY).is_none() {
        return Err(JsonRpcError::invalid_params(
            "modern requests must carry \
             _meta['io.modelcontextprotocol/clientCapabilities']",
        ));
    }

    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
        return Err(JsonRpcError::unsupported_protocol_version(version));
    }

    Ok(version.to_string())
}

/// Make a successful result conform to the modern result envelope.
///
/// 2026-07-28 requires `resultType` on every result (`"complete"` for a
/// finished request; this server never produces `"input_required"` because no
/// tool needs a mid-call follow-up — see `dispatch_tool`) and recommends
/// `io.modelcontextprotocol/serverInfo` in the result `_meta` so a client can
/// identify the server without connection state. When the tool result carries
/// a persisted retrieval receipt, the `_meta.ui.resourceUri` points at its
/// rendered MCP App card, which is how Claude Desktop opens the receipt view
/// inline (SEP-1865).
fn decorate_modern_result(result: &mut serde_json::Value) {
    // Read the receipt id before taking the mutable borrow below.
    let receipt_id = result
        .get("structuredContent")
        .and_then(|sc| sc.get("receiptId"))
        .and_then(|id| id.as_str())
        .map(String::from);
    let Some(object) = result.as_object_mut() else {
        return;
    };
    object
        .entry("resultType".to_string())
        .or_insert_with(|| serde_json::json!("complete"));

    let meta = object
        .entry("_meta".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(meta_object) = meta.as_object_mut() {
        meta_object
            .entry(META_SERVER_INFO_KEY.to_string())
            .or_insert_with(|| {
                serde_json::json!({
                    "name": "vestige",
                    "version": env!("CARGO_PKG_VERSION"),
                })
            });
        // Link the receipt card for any tool result that persisted one. The
        // id comes from the structured payload the receipt attach step wrote;
        // no additional lookup, no memory content added to the wire.
        if let Some(receipt_id) = receipt_id {
            meta_object.entry("ui".to_string()).or_insert_with(|| {
                serde_json::json!({
                    "resourceUri":
                        crate::resources::receipt_card::resource_uri(&receipt_id),
                })
            });
        }
    }
}

/// Cache hint for `server/discover`. The payload is compile-time constant
/// (supported revisions, capabilities and identity are baked into the binary),
/// so an hour is an honest hint, and "stable across calls within its own TTL"
/// holds trivially.
const DISCOVER_TTL_MS: u64 = 3_600_000;

/// Reject a pagination cursor this server never issued.
///
/// `tools/list`, `resources/list` and `resources/templates/list` return every
/// entry in one page and never emit a `nextCursor`, so any cursor a client sends
/// back is by definition one Vestige did not hand out. The spec says an unknown
/// cursor SHOULD be answered with `-32602` rather than silently ignored, and
/// ignoring it is worse than pedantic: a client that believes it is paginating
/// would loop on page one forever. Flagged by the conformance suite in #175.
///
/// `null` and `""` are treated as absent, so a client that always serialises
/// the field is not punished for a cursor it never really set.
/// Tools a Strata log withholds in 4.0, with every hidden alias. Erasure:
/// the log is append-only, so these would hide a memory but keep its bytes.
const STRATA_WITHHELD_TOOLS: &[&str] = &["purge", "delete_knowledge"];

/// `(tool, action, why)` a Strata log cannot honor in 4.0. `None` is erasure
/// (shared wording with the store). Each is dropped from the advertised
/// schema and refused on call, so every advertised action works.
const STRATA_WITHHELD_ACTIONS: &[(&str, &str, Option<&str>)] = &[
    ("memory", "purge", None),
    ("memory", "delete", None),
    ("blast_radius", "retire", None),
    (
        "dedup",
        "plan_merge",
        Some("merge planning needs embeddings, which 4.0 removed"),
    ),
    (
        "dedup",
        "plan_supersede",
        Some("supersede planning needs embeddings, which 4.0 removed"),
    ),
    (
        "dedup",
        "apply",
        Some("there are no merge or supersede plans without embeddings"),
    ),
    (
        "dedup",
        "verdict",
        Some("reconsolidation verdicts need embeddings, which 4.0 removed"),
    ),
    (
        "dedup",
        "protect",
        Some("the Strata log has no protect flag yet"),
    ),
    (
        "receipt",
        "save_walk",
        Some("walk receipts are not recorded on a Strata log yet"),
    ),
    (
        "maintain",
        "consolidate",
        Some(tools::unavailable::CONSOLIDATE_NOOP),
    ),
    (
        "maintain",
        "restore",
        Some(
            "Strata backups are directory copies; stop Vestige and copy a backup's log/ into the data directory",
        ),
    ),
];

/// The refusal for a call a Strata log withholds in 4.0, or `None`.
fn strata_withheld_call(tool: &str, arguments: Option<&serde_json::Value>) -> Option<String> {
    withheld_call_in(STRATA_WITHHELD_ACTIONS, tool, arguments)
}

/// Resolve a six-tool operation to its existing implementation. This is an
/// internal route only: tools/list and the opening MCP trace retain the public
/// six-tool name. Review gates, receipt builders, and Reflex must observe the
/// same effective operation/arguments as the implementation they protect.
fn unified_operation(request: &CallToolRequest) -> Result<CallToolRequest, String> {
    let mut name = request.name.clone();
    let mut args = request.arguments.clone();
    let action = args
        .as_ref()
        .and_then(|a| a.get("action"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let nested = match (name.as_str(), action) {
        ("session", "reflex") => Some("reflex"),
        ("session", "intention") => Some("intention"),
        ("session", "codebase") => Some("codebase"),
        ("remember", "backfill") => Some("backfill"),
        ("curate", "memory") => Some("memory"),
        ("curate", "graph") => Some("graph"),
        ("curate", "ghostlink") => Some("ghostlink"),
        ("curate", "dream") => Some("dream"),
        ("curate", "maintain") => Some("maintain"),
        _ => None,
    };
    if let Some(target) = nested {
        let raw = args.as_ref().unwrap();
        let mut params = raw
            .get("params")
            .filter(|p| p.is_object())
            .cloned()
            .ok_or_else(|| format!("{} action={action} requires object 'params'", request.name))?;
        // Outer fields are never silently dropped: in particular a misplaced
        // confirm/scope must not be mistaken for authority on the inner action.
        if raw
            .as_object()
            .unwrap()
            .keys()
            .any(|k| k != "action" && k != "params")
        {
            return Err(format!(
                "{} action={action} takes only action and params; put operation fields inside params",
                request.name
            ));
        }
        tools::compact::unfold_arguments(target, &mut params);
        name = target.into();
        args = Some(params);
    } else if name == "session"
        && action == "start"
        && args
            .as_ref()
            .is_some_and(tools::session_unified::uses_context_packet)
    {
        name = "session_start".into();
        args.as_mut()
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("action");
    } else if name == "session"
        && (action == "status" || action.is_empty())
        && args
            .as_ref()
            .is_none_or(|a| a["view"] != "tools" && a["view"] != "memory_prs")
    {
        name = "memory_status".into();
        let raw = args.get_or_insert_with(|| serde_json::json!({}));
        raw.as_object_mut().unwrap().remove("action");
        if raw.get("view").is_none() || raw["view"] == "system" {
            raw["view"] = serde_json::json!("health");
        }
    }
    Ok(CallToolRequest {
        name,
        arguments: args,
    })
}

/// `ghostlink` is keyed by `mode` (+ `view` / `kind`), `graph` and
/// `composed_graph` by `action`: all three resolve to one graph action, so a
/// `("graph", action)` row withholds the same call on every surface.
fn graph_family_action(tool: &str, arguments: Option<&serde_json::Value>) -> Option<String> {
    match tool {
        "ghostlink" => tools::ghostlink::graph_action_for(arguments),
        "graph" | "composed_graph" => arguments
            .and_then(|args| args.get("action"))
            .and_then(|value| value.as_str())
            .map(str::to_string),
        _ => None,
    }
}

fn withheld_call_in(
    table: &[(&str, &str, Option<&str>)],
    tool: &str,
    arguments: Option<&serde_json::Value>,
) -> Option<String> {
    let field = |name: &str| {
        arguments
            .and_then(|args| args.get(name))
            .and_then(|value| value.as_str())
    };
    if let Some(action) = graph_family_action(tool, arguments) {
        let (_, _, why) = table
            .iter()
            .find(|(name, withheld, _)| *name == "graph" && *withheld == action)?;
        return Some(match why {
            None => crate::strata_memory::withheld_message(&format!("{tool} '{action}'")),
            Some(why) => tools::unavailable::withheld_in_4_0(&format!("{tool} '{action}'"), why),
        });
    }
    if STRATA_WITHHELD_TOOLS.contains(&tool) {
        return Some(crate::strata_memory::withheld_message(tool));
    }
    if tool == "memory_status"
        && field("view") == Some("changelog")
        && arguments.and_then(|args| args.get("memory_id")).is_some()
    {
        return Some(
            "unavailable_in_4_0: memory_status changelog for one memory_id is not available on Strata in Vestige 4.0: a Strata log does not record per-memory state transitions. Use memory_status view='provenance' with memoryId, or receipt get, for that memory's recorded history.".to_string(),
        );
    }
    let action = field("action")?;
    if tool == "maintain" && action == "export" && field("format") == Some("portable") {
        return Some(
            "unavailable_in_4_0: maintain export format 'portable' is not written from a Strata log yet; use format 'json' or 'jsonl'.".to_string(),
        );
    }
    let (_, _, why) = table
        .iter()
        .find(|(name, withheld, _)| *name == tool && *withheld == action)?;
    Some(match why {
        None => crate::strata_memory::withheld_message(&format!("{tool} action '{action}'")),
        Some(why) => tools::unavailable::withheld_in_4_0(&format!("{tool} action '{action}'"), why),
    })
}

/// Drop what a Strata log withholds from the advertised tool list, so the
/// surface matches what the log can honestly do.
fn withhold_on_strata(tools: &mut Vec<ToolDescription>) {
    tools.retain(|tool| !STRATA_WITHHELD_TOOLS.contains(&tool.name.as_str()));
    for tool in tools.iter_mut() {
        let removed = strip_withheld_actions(&tool.name, &mut tool.input_schema);
        if tool.name == "suppress" {
            // On Strata a suppression hides a memory for good (the log keeps
            // the bytes), so the host should prompt, and reverse is refused.
            if let Some(annotations) = tool.annotations.as_mut() {
                annotations.destructive_hint = true;
            }
            tool.description = Some("Hide a memory from every read; the log keeps its bytes. On Strata in 4.0 it cannot be undone (reverse=true is refused) and is not erasure.".to_string());
        } else if tool.name == "memory" {
            tool.description = Some("Manage one memory: 'get', 'get_batch', 'state', 'promote' / 'demote' (demote never deletes), 'edit' (admit a successor, retire the previous node). Erasure is withheld on Strata in 4.0.".to_string());
        } else if tool.name == "recall" {
            // The v3 text sold keyword search and similarity modes, which a
            // Strata log answers with similarity_disabled.
            tool.description = Some("Exact handle (id, 8+ char prefix, tag) or exact-token BM25 over stored text. No embeddings.".to_string());
            if let Some(props) = tool
                .input_schema
                .pointer_mut("/properties")
                .and_then(|v| v.as_object_mut())
            {
                for legacy_field in [
                    "source",
                    "filters",
                    "context_topics",
                    "depth",
                    "topic",
                    "since",
                    "min_trust",
                ] {
                    props.remove(legacy_field);
                }
            }
        } else if tool.name == "smart_ingest" {
            tool.description = Some("Save one memory ('content') or up to 20 ('items'); each write passes the gate and returns a receipt. Secret-shaped content is refused unless allowSecrets is set.".to_string());
        } else if tool.name == "receipt" {
            tool.description = Some("'get' shows what a write did from its receipt id; 'replay' re-derives that state from the log and reports any mismatch.".to_string());
        } else if !removed.is_empty() {
            let note = format!(" Withheld on Strata in 4.0: {}.", removed.join(", "));
            tool.description = tool.description.take().map(|text| text + &note);
        }
    }
}

/// Remove withheld selector values from one tool's schema; returns them.
fn strip_withheld_actions(tool: &str, schema: &mut serde_json::Value) -> Vec<&'static str> {
    strip_withheld_in(STRATA_WITHHELD_ACTIONS, tool, schema)
}

fn strip_withheld_in(
    table: &[(&str, &'static str, Option<&str>)],
    tool: &str,
    schema: &mut serde_json::Value,
) -> Vec<&'static str> {
    if tool == "ghostlink" {
        // Mode, view and kind values whose graph action a row withholds.
        let withheld: Vec<&'static str> = table
            .iter()
            .filter(|(name, _, _)| *name == "graph")
            .map(|(_, action, _)| *action)
            .collect();
        let mut removed = Vec::new();
        for (selector, pointer) in [
            ("mode", "/properties/mode/enum"),
            ("view", "/properties/view/enum"),
            ("kind", "/properties/kind/enum"),
        ] {
            if let Some(values) = schema.pointer_mut(pointer).and_then(|v| v.as_array_mut()) {
                values.retain(|value| {
                    let Some(value) = value.as_str() else {
                        return true;
                    };
                    // A mode maps to one action (inspect / explore defer to
                    // their view / kind); a view or kind is the action.
                    let action = if selector == "mode" {
                        tools::ghostlink::graph_action_for(Some(
                            &serde_json::json!({ "mode": value }),
                        ))
                    } else {
                        Some(value.to_string())
                    };
                    match action.and_then(|a| withheld.iter().find(|w| **w == a).copied()) {
                        Some(hit) => {
                            removed.push(hit);
                            false
                        }
                        None => true,
                    }
                });
            }
        }
        return removed;
    }
    let withheld: Vec<&'static str> = table
        .iter()
        .filter(|(name, _, _)| *name == tool)
        .map(|(_, action, _)| *action)
        .collect();
    if let Some(values) = schema
        .pointer_mut("/properties/action/enum")
        .and_then(|values| values.as_array_mut())
    {
        values.retain(|value| {
            value
                .as_str()
                .is_none_or(|value| !withheld.contains(&value))
        });
    }
    if tool == "memory"
        && let Some(description) = schema.pointer_mut("/properties/action/description")
    {
        *description = serde_json::json!(
            "'get', 'get_batch' (ids), 'state', 'promote' / 'demote' (demote never deletes), 'edit' (admit a successor, retire the previous node). Erasure is withheld on Strata in 4.0."
        );
    }
    if tool == "recall"
        && let Some(description) = schema.pointer_mut("/properties/mode/description")
    {
        *description = serde_json::json!(
            "'lookup' (default). 'reason'/'contradictions': similarity_disabled."
        );
    }
    if tool == "maintain"
        && let Some(formats) = schema
            .pointer_mut("/properties/format/enum")
            .and_then(|values| values.as_array_mut())
    {
        formats.retain(|value| value != "portable");
    }
    withheld
}

/// Strata wording for the selector descriptions of the full schema that
/// `memory_status` `view='tools'` serves. The wire catalog truncates these to
/// 50 characters, so the longer Strata text lives here and costs the
/// `tools/list` budget nothing.
fn describe_full_schema_for_strata(tool: &str, schema: &mut serde_json::Value) {
    let text = match tool {
        "maintain" => {
            "Store-wide maintenance: dream and dream_compile (replay recorded edges; no memory row is rewritten), gc (lists nothing and deletes nothing on Strata), importance_score (id: a count of recorded edges and reviews, never a text score), backup, export. Inspect the selected action's schema. Export uses since; start/end are unsupported."
        }
        "dedup" => {
            "'scan' (default, read-only): exact-equality duplicate clusters. 'undo': reverse an operation_id, or list the reflog. 'tag_rename' / 'tag_merge': preview-token gated. 'policy': thresholds."
        }
        _ => return,
    };
    if let Some(description) = schema.pointer_mut("/properties/action/description") {
        *description = serde_json::json!(text);
    }
}

fn reject_unknown_cursor(params: Option<&serde_json::Value>) -> Result<(), JsonRpcError> {
    let Some(cursor) = params.and_then(|p| p.get("cursor")) else {
        return Ok(());
    };
    match cursor {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::String(s) if s.is_empty() => Ok(()),
        _ => Err(JsonRpcError::invalid_params(
            "unknown pagination cursor: this server returns a single page and never issues one",
        )),
    }
}

/// Whether `VESTIGE_TRACE` enables Black Box trace recording. ON by default:
/// unset, empty, or any malformed value → true (fail-open to the documented
/// default); only an explicit `0` / `false` / `off` / `no` turns it off.
/// Parsed exactly like the sibling `VESTIGE_AUTO_CONSOLIDATE_MERGE` gate.
fn parse_trace_enabled(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let v = v.trim();
            !(v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no")
                || v == "0")
        }
        None => true,
    }
}

/// OPT-OUT (Black Box trace recorder): every MCP tool call writes trace rows
/// (`agent_traces`/`agent_runs`) to the user's DB. A consumer that does not
/// want per-call persistence can turn it off with `VESTIGE_TRACE=0` (or
/// false/off/no). Read ONCE per process (the recorder sits on the hot path of
/// every tool call), so flipping the env mid-process has no effect.
fn trace_enabled() -> bool {
    static TRACE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE_ENABLED
        .get_or_init(|| parse_trace_enabled(std::env::var("VESTIGE_TRACE").ok().as_deref()))
}

fn log_level_rank(level: &str) -> Option<usize> {
    [
        "debug",
        "info",
        "notice",
        "warning",
        "error",
        "critical",
        "alert",
        "emergency",
    ]
    .iter()
    .position(|item| *item == level)
}

/// MCP Server implementation
pub struct McpServer {
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    /// The stable process actor (#252 Phase A): a did:key minted per data
    /// directory at first run and persisted at `<data_dir>/actor.key`.
    /// Identity comes from this field only; per-call `role` arguments supply
    /// a claimed role and can never override it.
    actor: Arc<vestige_core::actor::ProcessActor>,
    /// Handshake flag. Atomic rather than `bool` so `handle_request` can take
    /// `&self` and the stdio transport can dispatch requests concurrently
    /// behind an `Arc<McpServer>` (see `protocol/stdio.rs::run_io`).
    initialized: AtomicBool,
    /// Minimum MCP log level the client asked for, as a rank. Atomic for the
    /// same reason as `initialized`.
    logging_level: AtomicUsize,
    /// Tool call counter for inline consolidation trigger (every 100 calls)
    tool_call_count: AtomicU64,
    /// Set while an inline consolidation worker is running. The trigger
    /// predicate below claims nothing and resets nothing, so with requests
    /// dispatched concurrently several handlers can reach the trigger together
    /// and each spawn its own synchronous consolidation.
    consolidating: Arc<AtomicBool>,
    /// Optional event broadcast channel for dashboard real-time updates.
    event_tx: Option<broadcast::Sender<VestigeEvent>>,
    /// Resolved output config from `<data_dir>/vestige.toml` (Phase 2). Tools
    /// use it as the fallback for detail/limit when no explicit MCP param is
    /// given; explicit params always win.
    output_config: Arc<OutputConfig>,
    /// The connected client's name from `initialize` (clientInfo.name), so the
    /// dashboard can say which agent is showing a memory.
    client_name: std::sync::OnceLock<String>,
    /// The host session this connection's hooks report to `reflex`, so the
    /// memories this agent reads can be named at a later Operator stop.
    reflex: tools::reflex::ConnectionState,
}

/// Load `vestige.toml` from the storage's data directory and resolve it to an
/// effective [`OutputConfig`]. A missing/malformed file yields the built-in
/// default, which preserves historical behavior.
fn load_output_config(storage: &Arc<Storage>) -> Arc<OutputConfig> {
    let config = VestigeConfig::load_from_data_dir(storage.data_dir());
    Arc::new(config.output())
}

/// Load or mint the process actor for this data directory and bind it to the
/// store so node writes stamp `author_actor_did`. A mint failure must not
/// take the whole server down: identity is provenance metadata, and storage
/// paths fail closed (no endorsement) when no actor is bound.
fn load_process_actor(storage: &Arc<Storage>) -> Arc<vestige_core::actor::ProcessActor> {
    let key_path = vestige_core::actor::actor_key_path_for_data_dir(storage.data_dir());
    match vestige_core::actor::ProcessActor::load_or_mint(&key_path) {
        Ok(actor) => {
            if let Err(error) = storage.set_process_actor(actor.did()) {
                tracing::warn!(%error, "could not bind the process actor; endorsements stay disabled");
            }
            Arc::new(actor)
        }
        Err(error) => {
            tracing::warn!(
                %error,
                path = %key_path.display(),
                "could not load or mint the actor key; endorsements stay disabled"
            );
            Arc::new(vestige_core::actor::ProcessActor::mint())
        }
    }
}

/// Holds the right to run one inline consolidation, and releases it when the
/// worker finishes, returns early, or unwinds.
struct ConsolidationClaim(Arc<AtomicBool>);

impl Drop for ConsolidationClaim {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl McpServer {
    pub fn new(storage: Arc<Storage>, cognitive: Arc<Mutex<CognitiveEngine>>) -> Self {
        let output_config = load_output_config(&storage);
        let actor = load_process_actor(&storage);
        Self {
            storage,
            cognitive,
            actor,
            initialized: AtomicBool::new(false),
            logging_level: AtomicUsize::new(1),
            tool_call_count: AtomicU64::new(0),
            consolidating: Arc::new(AtomicBool::new(false)),
            event_tx: None,
            output_config,
            client_name: std::sync::OnceLock::new(),
            reflex: tools::reflex::new_connection(),
        }
    }

    /// Create an MCP server that broadcasts events to the dashboard.
    pub fn new_with_events(
        storage: Arc<Storage>,
        cognitive: Arc<Mutex<CognitiveEngine>>,
        event_tx: broadcast::Sender<VestigeEvent>,
    ) -> Self {
        let output_config = load_output_config(&storage);
        let actor = load_process_actor(&storage);
        Self {
            storage,
            cognitive,
            actor,
            initialized: AtomicBool::new(false),
            logging_level: AtomicUsize::new(1),
            tool_call_count: AtomicU64::new(0),
            consolidating: Arc::new(AtomicBool::new(false)),
            event_tx: Some(event_tx),
            output_config,
            client_name: std::sync::OnceLock::new(),
            reflex: tools::reflex::new_connection(),
        }
    }

    /// The process actor did:key, ONLY when it is actually bound to the
    /// store. An unbound identity (key-file failure at startup) never
    /// claims provenance.
    fn bound_actor_did(&self) -> Option<String> {
        let did = self.actor.did();
        (self.storage.process_actor_did().as_deref() == Some(did)).then(|| did.to_string())
    }

    /// Resolve the actor provenance for one tool call: the process identity
    /// plus the effective role the operator policy assigns to the call's
    /// CLAIMED role. The claimed role never overrides the identity.
    pub fn resolve_actor_provenance(
        &self,
        claimed_role: Option<&str>,
    ) -> Option<(String, vestige_core::actor::RoleResolution)> {
        let did = self.bound_actor_did()?;
        let snapshot = self.storage.actor_policy_snapshot().ok()?;
        let resolution = snapshot.resolve(&did, claimed_role);
        Some((did, resolution))
    }

    /// The JSON `actor` block shared by mutation responses (#252 Phase A).
    pub fn actor_response_block(&self, claimed_role: Option<&str>) -> Option<serde_json::Value> {
        let (did, resolution) = self.resolve_actor_provenance(claimed_role)?;
        Some(serde_json::json!({
            "id": did,
            "claimedRole": resolution.claimed_role,
            "effectiveRole": resolution.effective_role,
            "resolvedWeight": resolution.resolved_weight,
            "resolutionDisposition": resolution.disposition.as_str(),
            "policyVersion": resolution.policy_version,
        }))
    }

    /// Take the right to run one inline consolidation, or `None` when one is
    /// already running.
    ///
    /// `ConsolidationScheduler::should_consolidate` is a pure read: it claims
    /// nothing and resets nothing, so it answers the same way for every caller
    /// that asks in the same moment. With one request served at a time that
    /// could not matter. With requests dispatched concurrently, a burst of
    /// handlers reaching the trigger together would each spawn their own
    /// `run_consolidation`, and that call is synchronous, so each one occupies
    /// a runtime worker for its whole run.
    fn claim_consolidation(&self) -> Option<ConsolidationClaim> {
        self.consolidating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| ConsolidationClaim(Arc::clone(&self.consolidating)))
    }

    /// Emit an event to the dashboard (no-op if no event channel).
    fn emit(&self, event: VestigeEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Handle an incoming JSON-RPC request
    pub async fn handle_request(&self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        debug!("Handling request: {}", request.method);

        if request.id.is_none() {
            if request.method != "notifications/initialized" {
                debug!("Dropping JSON-RPC notification '{}'", request.method);
            }
            return None;
        }

        // Classify the request by protocol era, and validate the per-request
        // metadata a modern (2026-07-28) request must carry.
        //
        // The stateless core (SEP-2567/SEP-2575) means a modern request
        // arrives with no handshake at all: it declares its own protocol
        // version and client capabilities in `_meta`, and the server accepts
        // or rejects it independently of anything that came before. So a
        // modern request NEVER hits the `initialized` gate below — the gate
        // exists to enforce the legacy handshake, and the spec forbids
        // inferring context from prior requests on the same connection.
        let era = request_era(&request);
        if era == Era::Modern
            && let Err(error) = validate_modern_meta(&request)
        {
            return Some(JsonRpcResponse::error(request.id, error));
        }

        // Check initialization for handshake-era requests.
        //
        // `server/discover` is deliberately exempt. Its entire purpose is to let
        // a client learn what this server speaks BEFORE committing to a protocol
        // revision, and MCP 2026-07-28 -- which removes the handshake altogether
        // -- explicitly sanctions using it as a backward-compatibility probe on
        // stdio. Gating it behind the very handshake it exists to precede makes
        // it useless: a modern client probing this server would get
        // "Server not initialized" and have to guess.
        if era == Era::LegacyHandshake
            && !self.initialized.load(Ordering::Acquire)
            && request.method != "initialize"
            && request.method != "notifications/initialized"
            && request.method != "server/discover"
        {
            warn!(
                "Rejecting request '{}': server not initialized",
                request.method
            );
            return Some(JsonRpcResponse::error(
                request.id,
                JsonRpcError::server_not_initialized(),
            ));
        }

        let mut result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params).await,
            "notifications/initialized" => Err(JsonRpcError::invalid_request(
                "notifications/initialized must be sent without an id",
            )),
            "tools/list" => self.handle_tools_list(request.params.as_ref(), era).await,
            "tools/call" => self.handle_tools_call(request.params).await,
            "resources/list" => {
                self.handle_resources_list(request.params.as_ref(), era)
                    .await
            }
            "resources/templates/list" => {
                self.handle_resources_templates_list(request.params.as_ref(), era)
            }
            "resources/read" => self.handle_resources_read(request.params, era).await,
            "server/discover" => self.handle_server_discover(),
            "ping" => Ok(serde_json::json!({})),
            // The server only emits info and warning messages about its own
            // startup (model downloads), so any requested level is accepted.
            "logging/setLevel" => {
                match request
                    .params
                    .as_ref()
                    .and_then(|p| p.get("level"))
                    .and_then(|v| v.as_str())
                    .and_then(log_level_rank)
                {
                    Some(level) => {
                        self.logging_level.store(level, Ordering::Release);
                        Ok(serde_json::json!({}))
                    }
                    None => Err(JsonRpcError::invalid_params(
                        "level must be debug, info, notice, warning, error, critical, alert, or emergency",
                    )),
                }
            }
            method => {
                warn!("Unknown method: {}", method);
                Err(JsonRpcError::method_not_found())
            }
        };

        // Modern result envelope: `resultType` on every result and
        // `io.modelcontextprotocol/serverInfo` in the result `_meta` (both
        // required or recommended by 2026-07-28; see
        // `decorate_modern_result`). Legacy results are left byte-identical
        // to before — unknown result fields are ignored by legacy clients,
        // but there is no reason to grow the legacy wire at all when the
        // distinction costs one `if`.
        if era == Era::Modern
            && let Ok(value) = result.as_mut()
        {
            decorate_modern_result(value);
        }

        Some(match result {
            Ok(result) => JsonRpcResponse::success(request.id, result),
            Err(error) => JsonRpcResponse::error(request.id, error),
        })
    }

    /// `server/discover` — advertise supported protocol versions, capabilities and
    /// identity WITHOUT a handshake.
    ///
    /// MCP 2026-07-28 makes this mandatory: it removes the
    /// `initialize`/`notifications/initialized` handshake entirely and makes the
    /// protocol stateless, so a client needs some way to learn what a server
    /// speaks before it commits to a protocol revision. On stdio the spec
    /// explicitly allows using this as a backward-compatibility probe, which is
    /// exactly how a dual-era client decides between our modern path and the
    /// legacy handshake.
    ///
    /// Since #241 the server serves BOTH eras, so `supportedVersions` names
    /// 2026-07-28 first (the spec's own example orders newest first) followed
    /// by the legacy revisions negotiable through `initialize`. The result
    /// shape is the `DiscoverResult` schema (`resultType`, `supportedVersions`,
    /// `capabilities`, `ttlMs`, `cacheScope` required; `instructions` optional;
    /// server identity in `_meta` under `io.modelcontextprotocol/serverInfo`).
    /// The first version of this handler invented its own field names
    /// (`protocolVersions`, `serverInfo`); a conforming client could not read
    /// them, concluded the server offered no revision at all, and tested the
    /// newest one anyway (#175).
    ///
    /// Unlike `initialize`, this neither takes params nor mutates session state,
    /// so it is safe to call at any point, including before initialization.
    ///
    /// The shape is `DiscoverResult` from the 2026-07-28 schema, because that is
    /// the only schema that defines the method: `resultType`,
    /// `supportedVersions`, `capabilities`, `ttlMs` and `cacheScope` are
    /// required, `instructions` is optional, and server identity lives in
    /// `_meta` under `io.modelcontextprotocol/serverInfo`. The first version of
    /// this handler invented its own field names (`protocolVersions`,
    /// `serverInfo`); a conforming client could not read them, concluded the
    /// server offered no revision at all, and tested the newest one anyway
    /// (#175).
    fn handle_server_discover(&self) -> Result<serde_json::Value, JsonRpcError> {
        Ok(serde_json::json!({
            "resultType": "complete",
            "supportedVersions": supported_protocol_versions(),
            "capabilities": {
                "tools": { "listChanged": false },
                "resources": { "listChanged": false },
                "logging": {},
            },
            "instructions": build_instructions(),
            "ttlMs": DISCOVER_TTL_MS,
            // Nothing here is caller-specific: revisions, capabilities and build
            // identity are the same for everyone, so a shared cache is fine.
            "cacheScope": "public",
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "vestige",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            },
        }))
    }

    /// Handle initialize request
    async fn handle_initialize(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, JsonRpcError> {
        let request: InitializeRequest = match params {
            Some(p) => serde_json::from_value(p)
                .map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => {
                return Err(JsonRpcError::invalid_params(
                    "initialize params are required",
                ));
            }
        };

        // `initialize` is a legacy-era method: negotiation happens over the
        // handshake revisions only. A client that wants the modern stateless
        // revision does not shake hands at all — it sends requests carrying
        // `_meta['io.modelcontextprotocol/protocolVersion']`. Echoing
        // 2026-07-28 here would commit the session to handshake semantics the
        // modern revision removed, so it is excluded from this list and the
        // handshake falls back to the newest legacy revision.
        let negotiated_version =
            if LEGACY_PROTOCOL_VERSIONS.contains(&request.protocol_version.as_str()) {
                info!(
                    "Client requested supported protocol version {}, using it",
                    request.protocol_version
                );
                request.protocol_version.clone()
            } else {
                info!(
                    "Client requested unsupported protocol version {}, using {}",
                    request.protocol_version, MCP_VERSION
                );
                MCP_VERSION.to_string()
            };

        self.initialized.store(true, Ordering::Release);
        let _ = self.client_name.set(request.client_info.name.clone());
        info!(
            "MCP session initialized with protocol version {}",
            negotiated_version
        );

        let result = InitializeResult {
            protocol_version: negotiated_version,
            server_info: ServerInfo {
                name: "vestige".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some({
                    let mut map = HashMap::new();
                    map.insert("listChanged".to_string(), serde_json::json!(false));
                    map
                }),
                resources: Some({
                    let mut map = HashMap::new();
                    map.insert("listChanged".to_string(), serde_json::json!(false));
                    map
                }),
                prompts: None,
                logging: Some(HashMap::new()),
            },
            instructions: Some(build_instructions()),
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
    }

    /// Whether the client has completed the `initialize` handshake. The stdio
    /// transport holds server-initiated notifications until then, so nothing
    /// precedes the initialize response on the wire.
    pub fn logging_allows(&self, notification: &serde_json::Value) -> bool {
        notification["params"]["level"]
            .as_str()
            .and_then(log_level_rank)
            .is_some_and(|level| level >= self.logging_level.load(Ordering::Acquire))
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// The advertised tool catalog. Single source of the full schemas:
    /// `handle_tools_list` compacts it for the wire (#212), and
    /// `tools::compact::full_schema` serves the same schemas in full through
    /// `memory_status` `view='tools'`. The parity guard test keeps this
    /// function and that registry name-for-name identical.
    fn tool_catalog() -> Vec<ToolDescription> {
        vec![
            ToolDescription {
                name: "session".to_string(),
                title: Some("Session".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: false,
                    destructive_hint: false,
                    idempotent_hint: false,
                    open_world_hint: false,
                }),
                description: Some(
                    "Session lifecycle, memory hooks, Reflex, intentions, codebase context and rule projection. Use params for action=reflex/intention/codebase; other actions use root fields.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::session_unified::schema()),
                ..Default::default()
            },
            ToolDescription {
                name: "remember".to_string(),
                title: Some("Remember".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: false,
                    destructive_hint: false,
                    idempotent_hint: false,
                    open_world_hint: false,
                }),
                description: Some(
                    "Single gated write and inline backfill. Save uses root fields; action=backfill takes its native payload in params. Anchors codebase records and recompiles fences on 'pinned'.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::remember_unified::schema()),
                ..Default::default()
            },
            ToolDescription {
                name: "recall".to_string(),
                title: Some("Recall".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: false,
                }),
                description: Some(
                    "Single exact-handle & context reader. Parameters must be flat JSON keys. Do not nest arguments. Returns discovered_handles on refusal if an unindexed identifier is passed.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::recall::schema()),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "query": {"type": "string"},
                        "discovered_handles": {"type": "array"}
                    }
                })),
                meta: Some(serde_json::json!({
                    "ui": { "resourceUri": "ui://vestige/receipt/{id}" }
                })),
            },
            ToolDescription {
                name: "causal_walk".to_string(),
                title: Some("Causal Walk & Prove".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: false,
                    destructive_hint: false,
                    idempotent_hint: false,
                    open_world_hint: false,
                }),
                description: Some(
                    "Recorded causal investigation and commit witness bundles. action='walk' follows explicit starts; action='prove' bundles already recorded test witnesses using repoPath, commit and out. It does not run tests or bisect.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::causal_walk::schema()),
                ..Default::default()
            },
            ToolDescription {
                name: "curate".to_string(),
                title: Some("Curate".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: false,
                    destructive_hint: true,
                    idempotent_hint: false,
                    open_world_hint: false,
                }),
                description: Some(
                    "Graph and state operations: promote, demote, edit, native typed link, bridge and suppress. Complete memory/graph/ghostlink/dream/maintain operations use action plus params.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::curate_unified::schema()),
                ..Default::default()
            },
            ToolDescription {
                name: "precedent".to_string(),
                title: Some("Precedent".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: false,
                    destructive_hint: false,
                    idempotent_hint: false,
                    open_world_hint: false,
                }),
                description: Some(
                    "Guard stops, receipts, replay, commit proof, forgotten lessons, BLAKE3/Ed25519 verify_log, scan_secrets, and selftest. Proof may write artifacts when explicitly requested.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::precedent::schema()),
                ..Default::default()
            },
        ]
    }

    /// Handle tools/list request
    async fn handle_tools_list(
        &self,
        params: Option<&serde_json::Value>,
        era: Era,
    ) -> Result<serde_json::Value, JsonRpcError> {
        reject_unknown_cursor(params)?;

        // Backfill and causal_walk are distinct advertised tools. Deprecated
        // folded names remain hidden redirects in handle_tools_call; the full
        // schema registry in tools/compact.rs must match this catalog.
        let mut tools = Self::tool_catalog();
        if crate::strata_memory::is_strata_backend(self.storage.as_ref()) {
            withhold_on_strata(&mut tools);
        }

        // Per-tool result-size annotation `_meta["anthropic/maxResultSizeChars"]`.
        //
        // Claude Code v2.1.91+ honors this annotation to override its 50K default
        // `CallToolResult` truncation. Without it, large Vestige payloads
        // (`search` with `detail_level="full"` at `limit=20` has been observed
        // at ~135K chars; `memory_timeline` at `limit=30` at ~84K chars) are
        // silently truncated and spilled to disk, forcing the parent agent to
        // chunk-read them.
        //
        // Per-tool caps below are sized at ~2× observed peak with growth
        // headroom; max permitted by Anthropic is 500_000. Only the
        // high-payload tools carry the annotation (recall, memory_status,
        // memory, codebase, dedup, graph); the remaining advertised tools
        // deliberately do NOT (cargo-cult prevention — annotating a
        // small-payload tool dilutes the signal).
        //
        // Other tools that COULD plausibly grow into the annotated set with
        // future workload (`deep_reference`, `cross_reference`, `memory_graph`,
        // `explore_connections`, `session_context`) are left unannotated until
        // empirical measurement shows truncation under realistic use.
        for tool in tools.iter_mut() {
            let max_chars: Option<u64> = match tool.name.as_str() {
                // v2.2: search folded into recall (mode='lookup'); annotation moved.
                "recall" => Some(300_000),
                "memory_status" => Some(200_000),
                "memory" => Some(100_000),
                "codebase" => Some(100_000),
                // v2.2: dedup action='scan' returns duplicate clusters +
                // merge candidates + policy in one payload.
                "dedup" => Some(150_000),
                // v2.2: graph action='memory_graph' (force-directed layout) and
                // 'bounty_mode' pagination can both produce large payloads.
                // 4.0: ghostlink carries the same payloads (map, bounty,
                // proofs on every candidate); the hidden graph alias keeps
                // its entry.
                "ghostlink" | "graph" => Some(250_000),
                _ => None,
            };
            if let Some(n) = max_chars {
                // Merge into the tool's existing `_meta` instead of replacing
                // it: `recall` already carries `ui.resourceUri` (MCP Apps), and
                // overwriting the map dropped that key from tools/list.
                let mut meta = match tool.meta.take() {
                    Some(serde_json::Value::Object(existing)) => existing,
                    _ => serde_json::Map::new(),
                };
                meta.insert(
                    "anthropic/maxResultSizeChars".to_string(),
                    serde_json::Value::from(n),
                );
                tool.meta = Some(serde_json::Value::Object(meta));
            }
        }

        // Deterministic order. MCP 2026-07-28 says servers SHOULD return tools
        // from tools/list in a stable order so clients can cache the list and so
        // the bytes land identically in an LLM prompt cache. The vec above is
        // hand-ordered by theme, which is good for a human reading the source and
        // useless as a cache key the moment anyone reorders it.
        tools.sort_by(|a, b| a.name.cmp(&b.name));

        let mut result = serde_json::to_value(ListToolsResult { tools })
            .map_err(|e| JsonRpcError::internal_error(&e.to_string()))?;

        // Freshness hints (MCP 2026-07-28 `CacheableResult`). Measured on a real
        // install: this response is 28,506 bytes and was served 1,161 times from
        // one log -- roughly 7,000 tokens of tool schema pushed into model context
        // on every single session start, re-sent forever, with nothing telling the
        // client it could have kept the previous copy.
        //
        // The advertised surface only changes when the binary changes, so an hour
        // is conservative rather than aggressive. `private` because the list can
        // vary with per-install configuration; a shared intermediary must not
        // serve one install's tool list to another.
        //
        // Emitting these at 2025-11-25 is forward-compatible for a client
        // that negotiated through the handshake; but a client that DECLARED
        // 2025-11-25 per-request gets strict old-revision shape (no
        // newer-revision fields at all), so suppress them there.
        if era != Era::LegacyDeclared
            && let Some(object) = result.as_object_mut()
        {
            object.insert("ttlMs".to_string(), serde_json::json!(3_600_000u64));
            object.insert("cacheScope".to_string(), serde_json::json!("private"));
        }
        Ok(result)
    }

    /// Handle tools/call request. Crate-visible so the dashboard's tool bridge
    /// (`POST /api/mcp/{tool}`) runs the exact path an agent's call runs:
    /// Strata withholding, trace, review gate, receipt, events.
    pub(crate) async fn handle_tools_call(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, JsonRpcError> {
        // Claude Code names each call's tool_use_id in `_meta`; reflex ties the
        // call to the agent whose PreToolUse firing recorded the same id.
        let call_id = params
            .as_ref()
            .and_then(|p| p.pointer("/_meta/claudecode~1toolUseId"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let mut request: CallToolRequest = match params {
            Some(p) => serde_json::from_value(p)
                .map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => return Err(JsonRpcError::invalid_params("Missing tool call parameters")),
        };
        if let Some(arguments) = &request.arguments
            && !arguments.is_object()
        {
            return Err(JsonRpcError::invalid_params(
                "tools/call arguments must be an object",
            ));
        }
        // `tools/list` advertises filter fields grouped under `filters` and
        // `source`; the handlers read them at the top level.
        if let Some(arguments) = request.arguments.as_mut() {
            tools::compact::unfold_arguments(&request.name, arguments);
        }

        let operation =
            unified_operation(&request).map_err(|error| JsonRpcError::invalid_params(&error))?;
        // Inspect the public name, never the normalized internal operation.
        // Runs before Reflex and withheld-call early returns and with tracing off.
        audit_legacy_call(self.storage.as_ref(), &request.name);
        let operation_name = operation.name.clone();
        let operation_args = operation.arguments.clone();

        // Hook-only: `reflex` answers hooks in hook JSON. It skips the trace
        // recorder, the review gate, the consolidation counter and dashboard
        // events, so hook firings never count as agent activity.
        if operation_name == tools::reflex::NAME {
            let value = tools::reflex::execute(
                &self.storage,
                &self.cognitive,
                &self.output_config,
                &self.reflex,
                operation.arguments,
            )
            .await
            .unwrap_or_else(|error| {
                warn!(%error, "reflex answered empty");
                serde_json::json!({})
            });
            let call_result = CallToolResult {
                content: vec![crate::protocol::messages::ToolResultContent {
                    content_type: "text".to_string(),
                    text: value.to_string(),
                }],
                structured_content: Some(value),
                is_error: Some(false),
            };
            return serde_json::to_value(call_result)
                .map_err(|e| JsonRpcError::internal_error(&e.to_string()));
        }

        // 4.0: erasure-class calls are withheld on a Strata log. Refuse before
        // tracing and before the Memory-PR review gate, so a withheld call
        // cannot open a pending purge or suppress review either.
        if crate::strata_memory::is_strata_backend(self.storage.as_ref())
            && let Some(message) = strata_withheld_call(&operation_name, operation_args.as_ref())
        {
            let error_content = serde_json::json!({ "error": message });
            let call_result = CallToolResult {
                content: vec![crate::protocol::messages::ToolResultContent {
                    content_type: "text".to_string(),
                    text: error_content.to_string(),
                }],
                structured_content: Some(error_content),
                is_error: Some(true),
            };
            return serde_json::to_value(call_result)
                .map_err(|e| JsonRpcError::internal_error(&e.to_string()));
        }

        // Record activity on every tool call (non-blocking)
        if let Ok(mut cog) = self.cognitive.try_lock() {
            cog.activity_tracker.record_activity();
            cog.consolidation_scheduler.record_activity();
        }

        // Capture the args for tracing/event emission BEFORE tool dispatch
        // consumes request.arguments. The Black Box must populate even in pure
        // stdio mode (no dashboard socket), so this is NOT gated on event_tx —
        // only the WebSocket broadcast inside record()/emit is.
        let saved_args = request.arguments.clone();
        // Backfill previews promise zero durable writes, including the generic
        // call/result recorder. A normal run keeps its trace and own receipts.
        let backfill_preview = operation_name == "backfill"
            && operation_args
                .as_ref()
                .and_then(|args| args.get("dry_run"))
                .and_then(serde_json::Value::as_bool)
                == Some(true);
        let trace_this_call = trace_enabled() && !backfill_preview;

        // Agent Black Box: record the opening mcp.call event for this tool
        // invocation. run_id groups the events of one agent turn; it is derived
        // from the args (an explicit runId, or a fresh id) so record_result below
        // can attach the downstream memory events (retrieve/suppress/veto) to the
        // same run.
        let trace_run_id = crate::trace_recorder::run_id_for(&operation_args);
        if trace_this_call {
            crate::trace_recorder::record_call(
                &self.storage,
                self.event_tx.as_ref(),
                &trace_run_id,
                &request.name,
                &saved_args,
            );
        }

        // Destructive and suppressive mutations must be reviewed before their
        // tool implementation can remove a node or change its retrieval
        // influence. This deliberately runs after the opening trace event but
        // before dispatch. Safety is intentionally independent of VESTIGE_TRACE:
        // disabling the recorder must not silently permit destructive writes.
        // Normal calls continue to #150's unchanged record_result -> receipt ->
        // post-commit gate -> event order when tracing is enabled.
        let mode = crate::trace_recorder::read_review_mode(&self.storage);
        let pending = crate::trace_recorder::gate_pending_memory_mutation(
            &self.storage,
            self.event_tx.as_ref(),
            &trace_run_id,
            &operation_name,
            &operation_args,
            mode,
        );
        match pending {
            Ok(Some(content)) => {
                // The pre-gate already emitted MemoryPrOpened. `success`
                // is false, so emitting the normal tool event cannot claim
                // a deletion or suppression that did not happen.
                self.emit_tool_event(&operation_name, &operation_args, &content);
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: content.to_string(),
                    }],
                    structured_content: Some(content),
                    is_error: Some(false),
                };
                return serde_json::to_value(call_result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()));
            }
            Ok(None) => {}
            Err(error) => {
                let error_content = serde_json::json!({ "error": error });
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: error_content.to_string(),
                    }],
                    structured_content: Some(error_content),
                    is_error: Some(true),
                };
                return serde_json::to_value(call_result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()));
            }
        }

        // `mut` so the post-call block can annotate a successful result with any
        // Memory PRs or receipts it attaches to a successful result.
        // Outer layer: protocol errors (unknown tool). Inner layer: the
        // tool's own result, where `Err` becomes an is_error CallToolResult.
        let is_session_status = request.name == "session"
            && request
                .arguments
                .as_ref()
                .and_then(|a| a["action"].as_str())
                .unwrap_or("status")
                == "status";
        let mut result = match self.dispatch_tool(operation).await {
            Ok(result) => result,
            Err(protocol_error) => return Err(protocol_error),
        };
        if is_session_status && let Ok(content) = &mut result {
            content["runtime"] = migration_runtime_status(self.storage.as_ref());
        }

        // Purge rebuilds this cache inside the tool. Suppress retires through
        // the same read filter, so the in-process graph has to drop the id too.
        if (operation_name == "suppress"
            || (operation_name == "curate"
                && operation_args
                    .as_ref()
                    .is_some_and(|a| a["action"] == "suppress")))
            && crate::strata_memory::is_strata_backend(self.storage.as_ref())
            && result
                .as_ref()
                .ok()
                .and_then(|content| content.get("success"))
                .and_then(|value| value.as_bool())
                == Some(true)
        {
            let mut rebuilt = CognitiveEngine::new();
            rebuilt.hydrate(&self.storage);
            *self.cognitive.lock().await = rebuilt;
        }

        // ================================================================
        // DASHBOARD EVENT EMISSION (v2.0)
        // Emit real-time events to WebSocket clients after successful tool calls.
        // ================================================================
        if let Ok(ref mut content) = result {
            // Agent Black Box: inspect the successful result and record the
            // downstream memory events (retrieve/suppress/veto/dream) under the
            // same run_id as the opening mcp.call, so /api/traces, /api/receipts
            // and the trace:// resource are actually populated.
            if trace_this_call {
                crate::trace_recorder::record_result(
                    &self.storage,
                    self.event_tx.as_ref(),
                    &trace_run_id,
                    &operation_name,
                    content,
                );
                // Persist the receipt for this exact retrieval run, then attach
                // its stable reference to the structured response. The Black Box
                // can now answer "why did the agent do that?" from the same
                // evidence the tool actually used, rather than reconstructing an
                // explanation after the fact. Non-retrieval tools safely return
                // None and keep their existing response shape.
                // #252 Phase A: the receipt records who made the call —
                // process identity + the resolution of the call's CLAIMED
                // role. Claims never override identity or grant authority.
                let claimed_role = operation_args
                    .as_ref()
                    .and_then(|args| args.get("role"))
                    .and_then(|r| r.as_str())
                    .map(str::trim)
                    .filter(|role| !role.is_empty())
                    .or_else(|| {
                        operation_args
                            .as_ref()
                            .and_then(|args| args.get("claimedRole"))
                            .and_then(|r| r.as_str())
                            .map(str::trim)
                            .filter(|role| !role.is_empty())
                    });
                let actor_provenance =
                    self.resolve_actor_provenance(claimed_role)
                        .map(|(did, resolution)| {
                            vestige_core::trace::ActorProvenance::from_resolution(&did, &resolution)
                        });
                if let Some(receipt) = crate::trace_recorder::build_and_save_receipt(
                    &self.storage,
                    &trace_run_id,
                    &operation_name,
                    content,
                    actor_provenance,
                ) && let Some(obj) = content.as_object_mut()
                {
                    let receipt_id = receipt
                        .get("receipt_id")
                        .and_then(|value| value.as_str())
                        .map(String::from);
                    obj.entry("runId".to_string())
                        .or_insert_with(|| serde_json::json!(trace_run_id));
                    obj.insert("receiptId".to_string(), serde_json::json!(receipt_id));
                    // A caller that asked for `detail_level: "brief"` wants the
                    // smallest useful answer. The full receipt is persisted and
                    // one `receipt` call away by id, so only the id ships.
                    let brief = obj.get("detailLevel").and_then(|v| v.as_str()) == Some("brief");
                    if !brief {
                        obj.insert("receipt".to_string(), receipt);
                    }
                }

                // Memory PR gate: classify the writes this tool just made under
                // the active ReviewMode and, for risky ones, quarantine the new
                // node and open a Memory PR. `gate_writes` no-ops for non-write
                // tools, and `classify_write` auto-commits everything in Fast
                // mode, so this is safe on every call.
                //
                // This closes the dead seam: the gate and its tests existed but
                // it had no production caller, so `ReviewMode` was inert and
                // nothing ever landed in `memory_prs` outside tests.
                //
                // Note this rides on `trace_enabled()` with the rest of the black
                // box: a Memory PR is an auditable trace artifact and is reviewed
                // through the same surfaces, so disabling tracing disables gating.
                let mode = crate::trace_recorder::read_review_mode(&self.storage);
                let opened = crate::trace_recorder::gate_writes(
                    &self.storage,
                    self.event_tx.as_ref(),
                    &trace_run_id,
                    &operation_name,
                    content,
                    mode,
                );
                if !opened.is_empty()
                    && let Some(obj) = content.as_object_mut()
                {
                    // Tell the calling agent exactly what happened. A held write
                    // is quarantined until the PR is decided; a destructive write
                    // (or a failed suppression) is NOT held — the PR is an audit
                    // record of something that already happened. Saying
                    // "quarantined" for those would be false.
                    let held = opened
                        .iter()
                        .filter(|o| o.get("held").and_then(|v| v.as_bool()) == Some(true))
                        .count();
                    let recorded = opened.len() - held;
                    let mut parts = Vec::new();
                    if held > 0 {
                        parts.push(format!(
                            "{held} write(s) quarantined until their Memory PR is decided"
                        ));
                    }
                    if recorded > 0 {
                        parts.push(format!(
                            "{recorded} write(s) already applied but recorded for review \
                             (destructive or unsuppressable — nothing is held)"
                        ));
                    }
                    obj.insert("memoryPrs".to_string(), serde_json::json!(opened));
                    obj.insert(
                        "memoryPrNotice".to_string(),
                        serde_json::json!(format!(
                            "Review mode '{}': {}. Review in the dashboard under Memory PRs \
                             or via GET /api/memory-prs.",
                            mode.as_str(),
                            parts.join("; ")
                        )),
                    );
                }
            }
            // Reason mode budgets complete evidence groups before recording
            // the retrieval receipt. Account for the actual attached metadata
            // here; a caller-supplied trace ID may be arbitrarily long, so omit
            // that optional echo if necessary (the receipt keeps correlation).
            if let Some(budget) = content.get("tokenBudgetLimit").and_then(|v| v.as_u64()) {
                content["budgetUnit"] = serde_json::json!("utf8_bytes_div_4_ceiling");
                for _ in 0..3 {
                    let bytes = content.to_string().len();
                    content["tokensUsed"] = serde_json::json!(bytes.div_ceil(4));
                }
                if content.to_string().len() > budget as usize * 4 {
                    if let Some(object) = content.as_object_mut() {
                        object.remove("runId");
                        object.insert("runIdOmitted".into(), serde_json::json!(true));
                    }
                    for _ in 0..3 {
                        let bytes = content.to_string().len();
                        content["tokensUsed"] = serde_json::json!(bytes.div_ceil(4));
                    }
                }
            }
            // A memory read with a reason is an agent showing that memory to
            // its user: the reply carries the dashboard view of it, and the
            // emit below takes every open dashboard inside it.
            if let Some((id, _)) = shown_memory(&operation_name, &operation_args, content)
                && let Some(object) = content.as_object_mut()
            {
                object.insert(
                    "dashboard".to_string(),
                    serde_json::json!({
                        "url": format!("http://127.0.0.1:{}/dashboard/memory?id={}", dashboard_port(), id),
                        "note": "Every open Vestige dashboard is now inside this memory, showing your reason. Give the user this link if theirs is not open."
                    }),
                );
            }
            // Emit after receipt attachment and gating so the dashboard sees the
            // same final evidence and review state returned to the calling agent.
            self.emit_tool_event(&operation_name, &operation_args, content);
            // Last, so receipts and the dashboard keep the whole result: reflex
            // records what this read put in front of the agent, folds bodies it
            // already holds word for word, and keeps the memory hook from
            // injecting a memory twice.
            tools::reflex::record_retirement(&self.reflex, &self.storage, &operation_name, content);
            let hook = operation_args
                .as_ref()
                .and_then(|a| a.get("memory_hook"))
                .filter(|h| h.get("session").is_some());
            match hook {
                Some(hook) if operation_name == "session_start" => {
                    tools::reflex::admit_injection(&self.storage, hook, content)
                }
                _ => tools::reflex::fold(
                    &self.reflex,
                    &self.storage,
                    &operation_name,
                    operation_args.as_ref(),
                    call_id.as_deref(),
                    content,
                ),
            }
        }

        let response = match result {
            Ok(content) => {
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: content.to_string(),
                    }],
                    structured_content: Some(content),
                    is_error: Some(false),
                };
                serde_json::to_value(call_result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
            Err(e) => {
                let error_content = serde_json::json!({ "error": e });
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: error_content.to_string(),
                    }],
                    structured_content: Some(error_content),
                    is_error: Some(true),
                };
                serde_json::to_value(call_result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
        };

        // Inline consolidation trigger: uses ConsolidationScheduler instead of fixed count
        let count = self.tool_call_count.fetch_add(1, Ordering::Relaxed) + 1;
        let should_consolidate = self
            .cognitive
            .try_lock()
            .ok()
            .map(|cog| cog.consolidation_scheduler.should_consolidate())
            .unwrap_or(count.is_multiple_of(100)); // Fallback to count-based if lock unavailable

        // The claim is what holds this to one worker at a time; the predicate
        // above cannot, because it reads state it never changes.
        let claim = if should_consolidate {
            self.claim_consolidation()
        } else {
            None
        };

        if let Some(claim) = claim {
            let storage_clone = Arc::clone(&self.storage);
            let cognitive_clone = Arc::clone(&self.cognitive);
            tokio::spawn(async move {
                let _claim = claim;
                // Expire labile reconsolidation windows
                if let Ok(mut cog) = cognitive_clone.try_lock() {
                    let _expired = cog.reconsolidation.reconsolidate_expired();
                }
                // Auto-close reconsolidation merge plans whose labile window
                // expired without a verdict — no zombie plans.
                match storage_clone.expire_stale_reconsolidation_plans() {
                    Ok(closed) if !closed.is_empty() => {
                        tracing::info!(
                            count = closed.len(),
                            "auto-closed expired reconsolidation merge plans"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "reconsolidation plan expiry sweep failed");
                    }
                    Ok(_) => {}
                }

                match storage_clone.run_consolidation() {
                    Ok(result) => {
                        tracing::info!(
                            tool_calls = count,
                            decay_applied = result.decay_applied,
                            duplicates_merged = result.duplicates_merged,
                            activations_computed = result.activations_computed,
                            duration_ms = result.duration_ms,
                            "Inline consolidation triggered (scheduler)"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Inline consolidation failed: {}", e);
                    }
                }
            });
        }

        response
    }

    /// Route a `tools/call` request to its tool implementation.
    ///
    /// Extracted verbatim from `handle_tools_call` (which keeps the trace
    /// preamble and the post-call receipt/PR annotation) so the dispatch
    /// table stays readable on its own.
    ///
    /// The outer `Result` is the protocol layer: only an unknown tool name
    /// errors here, as a JSON-RPC `invalid_params`. The inner `Result` is the
    /// tool layer: an `Err` is rendered by the caller as an `is_error`
    /// `CallToolResult`, exactly as the inline dispatch did.
    async fn dispatch_tool(
        &self,
        request: CallToolRequest,
    ) -> Result<Result<serde_json::Value, String>, JsonRpcError> {
        let result = match request.name.as_str() {
            "session" if request.arguments.as_ref().is_some_and(|a| (a["action"] == "status" || a.get("action").is_none()) && a["view"] == "tools") => {
                let catalog = self.handle_tools_list(None, Era::LegacyHandshake).await?;
                tools::memory_status::tool_guide(&catalog, request.arguments.as_ref().unwrap())
            }
            "session" if request.arguments.as_ref().is_some_and(|a| (a["action"] == "status" || a.get("action").is_none()) && a["view"] == "memory_prs") => {
                tools::session_unified::memory_prs(&self.storage, request.arguments.as_ref().unwrap())
            }
            "session" => tools::session_unified::execute(&self.storage, &self.cognitive, request.arguments).await,
            "remember" => tools::remember_unified::execute(&self.storage, &self.cognitive, request.arguments).await,
            "curate" => tools::curate_unified::execute(&self.storage, &self.cognitive, request.arguments).await,

            // ================================================================
            // UNIFIED TOOLS (v1.1+) - Preferred API
            // ================================================================
            // RECALL — unified retrieval tool (v2.2). HOT PATH.
            // mode = lookup (default, zero-overhead) | reason | contradictions
            "recall" => {
                tools::recall::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }
            "receipt" => tools::receipt::execute(&self.storage, request.arguments).await,
            // REMOVED (2026-09-28): vector search is not the product.
            // Retrieval by exact handle lives in `recall`.
            "search" => Err(
                "tool 'search' is removed: vector search is not part of Vestige; use 'recall' with an exact handle (a full memory id or an exact tag)".to_string(),
            ),
            "memory" => {
                tools::memory_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            "project" => tools::project::execute(&self.storage, request.arguments).await,
            "codebase" => {
                tools::codebase_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }
            "intention" => {
                tools::intention_graph::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // Core memory (v1.7: smart_ingest absorbs ingest + checkpoint)
            // ================================================================
            "smart_ingest" => {
                tools::smart_ingest::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // External-source connectors (#57). Without the feature this name
            // falls through to the unknown-tool protocol error below.
            // ================================================================
            #[cfg(feature = "connectors")]
            "source_sync" => tools::source_sync::execute(&self.storage, request.arguments).await,

            // ================================================================
            // CAUSAL WALK — investigate recorded causal evidence
            // ================================================================
            "causal_walk" => tools::causal_walk::execute(&self.storage, request.arguments).await,
            "gather" => {
                tools::gather::execute(&self.storage, Some(&self.cognitive), request.arguments)
                    .await
            }

            // ================================================================
            // BACKFILL — revise salience, inspect receipts, or undo a revision
            // ================================================================
            "backfill" => tools::backfill::execute(&self.storage, request.arguments).await,

            // ================================================================
            // w3d: planted-cause selftest + forgotten-lesson detection
            // ================================================================
            "selftest" => tools::selftest::execute(&self.storage, request.arguments).await,
            "forgotten_lesson" => {
                tools::forgotten_lesson::execute(&self.storage, request.arguments).await
            }
            "precedent" => {
                tools::precedent::execute(&self.storage, request.arguments).await
            }

            // ================================================================
            // DEPRECATED (v1.7): ingest → smart_ingest
            // ================================================================
            "ingest" => {
                warn!("Tool 'ingest' is deprecated in v1.7. Use 'smart_ingest' instead.");
                tools::smart_ingest::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // DEPRECATED (v1.7): session_checkpoint → smart_ingest (batch mode)
            // ================================================================
            "session_checkpoint" => {
                warn!(
                    "Tool 'session_checkpoint' is deprecated in v1.7. Use 'smart_ingest' with 'items' parameter instead."
                );
                tools::smart_ingest::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // DEPRECATED (v1.7): promote_memory → curate(action='promote')
            // ================================================================
            "promote_memory" => {
                warn!(
                    "Tool 'promote_memory' is deprecated in v1.7. Use 'memory' with action='promote' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("promote"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "promote"})),
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }

            // #219: the standalone purge tool. Forces action='purge' and runs
            // the identical path as memory(action='purge') — same review gate,
            // same tombstone.
            "purge" => {
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("purge"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "purge"})),
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }
            "demote_memory" => {
                warn!(
                    "Tool 'demote_memory' is deprecated in v1.7. Use 'memory' with action='demote' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("demote"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "demote"})),
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }

            // ================================================================
            // DEPRECATED (v1.7): health_check, stats → system_status
            // ================================================================
            "health_check" => {
                warn!("Tool 'health_check' is deprecated in v1.7. Use 'system_status' instead.");
                tools::maintenance::execute_system_status(
                    &self.storage,
                    &self.cognitive,
                    request.arguments,
                )
                .await
            }
            "stats" => {
                warn!("Tool 'stats' is deprecated in v1.7. Use 'system_status' instead.");
                tools::maintenance::execute_system_status(
                    &self.storage,
                    &self.cognitive,
                    request.arguments,
                )
                .await
            }

            // ================================================================
            // MEMORY STATUS — unified status/temporal tool (v2.2)
            // view = health (default) | retention | timeline | changelog
            // ================================================================
            "memory_status"
                if request
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("view"))
                    .and_then(|v| v.as_str())
                    == Some("tools") =>
            {
                // This is tool RESULT content (data inside structuredContent),
                // not a response envelope, so the strict
                // no-newer-revision-fields rule for declared-legacy requests
                // does not apply here; the catalog rides with the same hints
                // it has always carried.
                let catalog = self.handle_tools_list(None, Era::LegacyHandshake).await?;
                let mut guide =
                    tools::memory_status::tool_guide(&catalog, request.arguments.as_ref().unwrap());
                if crate::strata_memory::is_strata_backend(self.storage.as_ref())
                    && let Ok(guide) = guide.as_mut()
                    && let Some(entries) = guide["tools"].as_array_mut()
                {
                    for entry in entries.iter_mut() {
                        let name = entry["name"].as_str().unwrap_or_default().to_string();
                        if let Some(schema) = entry.get_mut("inputSchema") {
                            strip_withheld_actions(&name, schema);
                            describe_full_schema_for_strata(&name, schema);
                        }
                    }
                }
                guide
            }
            "memory_status" => {
                tools::memory_status::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }

            // DEPRECATED (v2.2): folded into `memory_status`. Hidden aliases —
            // each calls the same underlying handler verbatim.
            "system_status" => {
                warn!(
                    "Tool 'system_status' is deprecated in v2.2. Use 'memory_status' (view='health')."
                );
                tools::maintenance::execute_system_status(
                    &self.storage,
                    &self.cognitive,
                    request.arguments,
                )
                .await
            }

            "mark_reviewed" => tools::review::execute(&self.storage, request.arguments).await,

            // ================================================================
            // DEPRECATED: legacy search aliases — redirect to `recall` lookup.
            // ('recall' itself is now the unified retrieval tool, handled above.)
            // ================================================================
            "semantic_search" | "hybrid_search" => {
                Err(
                    "tool 'semantic_search'/'hybrid_search' is removed: vector search is not part of Vestige; use 'recall' with an exact handle (a full memory id or an exact tag)".to_string(),
                )
            }

            // ================================================================
            // DEPRECATED: Memory tools - redirect to unified 'memory'
            // ================================================================
            "get_knowledge" => {
                warn!(
                    "Tool 'get_knowledge' is deprecated. Use 'memory' with action='get' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("id").cloned().unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "get",
                            "id": id
                        }))
                    }
                    None => None,
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }
            "delete_knowledge" => {
                warn!(
                    "Tool 'delete_knowledge' is deprecated. Use 'memory' with action='purge', confirm=true instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("id").cloned().unwrap_or(serde_json::Value::Null);
                        let confirm = args
                            .get("confirm")
                            .cloned()
                            .unwrap_or(serde_json::Value::Bool(false));
                        Some(serde_json::json!({
                            "action": "delete",
                            "id": id,
                            "confirm": confirm
                        }))
                    }
                    None => None,
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }
            "get_memory_state" => {
                warn!(
                    "Tool 'get_memory_state' is deprecated. Use 'memory' with action='state' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args
                            .get("memory_id")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "state",
                            "id": id
                        }))
                    }
                    None => None,
                };
                tools::memory_unified::execute(&self.storage, &self.cognitive, unified_args).await
            }

            // ================================================================
            // DEPRECATED: Codebase tools - redirect to unified 'codebase'
            // ================================================================
            "remember_pattern" => {
                warn!(
                    "Tool 'remember_pattern' is deprecated. Use 'codebase' with action='remember_pattern' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("remember_pattern"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "remember_pattern"})),
                };
                tools::codebase_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    unified_args,
                )
                .await
            }
            "remember_decision" => {
                warn!(
                    "Tool 'remember_decision' is deprecated. Use 'codebase' with action='remember_decision' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert(
                                "action".to_string(),
                                serde_json::json!("remember_decision"),
                            );
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "remember_decision"})),
                };
                tools::codebase_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    unified_args,
                )
                .await
            }
            "get_codebase_context" => {
                warn!(
                    "Tool 'get_codebase_context' is deprecated. Use 'codebase' with action='get_context' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("get_context"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "get_context"})),
                };
                tools::codebase_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    unified_args,
                )
                .await
            }

            // ================================================================
            // DEPRECATED: Intention tools - redirect to unified 'intention'
            // ================================================================
            "set_intention" => {
                warn!(
                    "Tool 'set_intention' is deprecated. Use 'intention' with action='set' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("set"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "set"})),
                };
                tools::intention_unified::execute(&self.storage, &self.cognitive, unified_args)
                    .await
            }
            "check_intentions" => {
                warn!(
                    "Tool 'check_intentions' is deprecated. Use 'intention' with action='check' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("check"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "check"})),
                };
                tools::intention_unified::execute(&self.storage, &self.cognitive, unified_args)
                    .await
            }
            "complete_intention" => {
                warn!(
                    "Tool 'complete_intention' is deprecated. Use 'intention' with action='update', status='complete' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args
                            .get("intentionId")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "update",
                            "id": id,
                            "status": "complete"
                        }))
                    }
                    None => None,
                };
                tools::intention_unified::execute(&self.storage, &self.cognitive, unified_args)
                    .await
            }
            "snooze_intention" => {
                warn!(
                    "Tool 'snooze_intention' is deprecated. Use 'intention' with action='update', status='snooze' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args
                            .get("intentionId")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let minutes = args
                            .get("minutes")
                            .cloned()
                            .unwrap_or(serde_json::json!(30));
                        Some(serde_json::json!({
                            "action": "update",
                            "id": id,
                            "status": "snooze",
                            "snooze_minutes": minutes
                        }))
                    }
                    None => None,
                };
                tools::intention_unified::execute(&self.storage, &self.cognitive, unified_args)
                    .await
            }
            "list_intentions" => {
                warn!(
                    "Tool 'list_intentions' is deprecated. Use 'intention' with action='list' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("list"));
                            if let Some(status) = obj.remove("status") {
                                obj.insert("filter_status".to_string(), status);
                            }
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "list"})),
                };
                tools::intention_unified::execute(&self.storage, &self.cognitive, unified_args)
                    .await
            }

            // ================================================================
            // Neuroscience tools (internal, not in tools/list)
            // ================================================================
            "list_by_state" => {
                tools::memory_states::execute_list(&self.storage, request.arguments).await
            }
            "state_stats" => tools::memory_states::execute_stats(&self.storage).await,
            "trigger_importance" => {
                tools::tagging::execute_trigger(&self.storage, request.arguments).await
            }
            "find_tagged" => tools::tagging::execute_find(&self.storage, request.arguments).await,
            "tagging_stats" => tools::tagging::execute_stats(&self.storage).await,

            // ================================================================
            // Feedback (internal, still used by request_feedback)
            // ================================================================
            "request_feedback" => {
                tools::feedback::execute_request_feedback(&self.storage, request.arguments).await
            }

            // ================================================================
            // TEMPORAL TOOLS (v1.2+) — DEPRECATED (v2.2): folded into
            // `memory_status` (view='timeline' / view='changelog'). Hidden aliases.
            // ================================================================
            "memory_timeline" => {
                warn!(
                    "Tool 'memory_timeline' is deprecated in v2.2. Use 'memory_status' (view='timeline')."
                );
                tools::timeline::execute(&self.storage, &self.output_config, request.arguments)
                    .await
            }
            "memory_changelog" => {
                warn!(
                    "Tool 'memory_changelog' is deprecated in v2.2. Use 'memory_status' (view='changelog')."
                );
                tools::changelog::execute(&self.storage, request.arguments).await
            }

            // ================================================================
            // MAINTAIN — unified maintenance/lifecycle tool (v2.2)
            // action = consolidate | dream | gc | importance_score | backup
            //        | export | restore
            // ================================================================
            "maintain" => {
                // Mirror the pre-dispatch *Started* events that the standalone
                // consolidate/dream arms emit, keyed off the action.
                match request
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("action"))
                    .and_then(|v| v.as_str())
                {
                    Some("consolidate") => self.emit(VestigeEvent::ConsolidationStarted {
                        timestamp: chrono::Utc::now(),
                    }),
                    Some("dream") => self.emit(VestigeEvent::DreamStarted {
                        memory_count: self
                            .storage
                            .get_stats()
                            .map(|s| s.total_nodes as usize)
                            .unwrap_or(0),
                        timestamp: chrono::Utc::now(),
                    }),
                    _ => {}
                }
                tools::maintain::execute(&self.storage, &self.cognitive, request.arguments).await
            }

            // ================================================================
            // MAINTENANCE TOOLS (v1.2+) — DEPRECATED (v2.2): folded into
            // `maintain`. Hidden aliases; pre-emit Started events preserved.
            // ================================================================
            "consolidate" => {
                warn!(
                    "Tool 'consolidate' is deprecated in v2.2. Use 'maintain' (action='consolidate')."
                );
                self.emit(VestigeEvent::ConsolidationStarted {
                    timestamp: chrono::Utc::now(),
                });
                tools::maintenance::execute_consolidate(&self.storage, request.arguments).await
            }
            "backup" => {
                warn!("Tool 'backup' is deprecated in v2.2. Use 'maintain' (action='backup').");
                tools::maintenance::execute_backup(&self.storage, request.arguments).await
            }
            "export" => {
                warn!("Tool 'export' is deprecated in v2.2. Use 'maintain' (action='export').");
                tools::maintenance::execute_export(&self.storage, request.arguments).await
            }
            "gc" => {
                warn!("Tool 'gc' is deprecated in v2.2. Use 'maintain' (action='gc').");
                tools::maintenance::execute_gc(&self.storage, request.arguments).await
            }

            // ================================================================
            // AUTO-SAVE & DEDUP TOOLS (v1.3+)
            // ================================================================
            // DEPRECATED (v2.2): folded into `maintain` (action='importance_score').
            "importance_score" => {
                warn!(
                    "Tool 'importance_score' is deprecated in v2.2. Use 'maintain' (action='importance_score')."
                );
                tools::importance::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            // ================================================================
            // DEDUP / MERGE / SUPERSEDE — unified `dedup` tool (v2.2)
            // ================================================================
            "dedup" => {
                tools::dedup::execute_unified(
                    &self.storage,
                    Some(&self.cognitive),
                    request.arguments,
                )
                .await
            }

            // DEPRECATED (v2.2): folded into `dedup`. Kept as hidden back-compat
            // aliases (≥1 minor release) — they call the same underlying handlers
            // verbatim, so envelopes/plan_id/confirm-gating/bitemporal are intact.
            "find_duplicates" => {
                warn!(
                    "Tool 'find_duplicates' is deprecated in v2.2. Use 'dedup' with action='scan'."
                );
                tools::dedup::execute(&self.storage, request.arguments).await
            }
            "merge_candidates" | "plan_merge" | "plan_supersede" | "apply_plan" | "merge_undo"
            | "protect" | "merge_policy" => {
                warn!(
                    "Tool '{}' is deprecated in v2.2. Use 'dedup' (action={}).",
                    request.name,
                    match request.name.as_str() {
                        "merge_candidates" => "scan",
                        "apply_plan" => "apply",
                        "merge_undo" => "undo",
                        "merge_policy" => "policy",
                        other => other,
                    }
                );
                tools::merge::execute(&self.storage, request.name.as_str(), request.arguments).await
            }

            // ================================================================
            // COGNITIVE TOOLS (v1.5+) — DEPRECATED (v2.2): dream folded into
            // `maintain` (action='dream'). Hidden alias; DreamStarted preserved.
            // ================================================================
            "dream" => {
                warn!("Tool 'dream' is deprecated in v2.2. Use 'maintain' (action='dream').");
                self.emit(VestigeEvent::DreamStarted {
                    memory_count: self
                        .storage
                        .get_stats()
                        .map(|s| s.total_nodes as usize)
                        .unwrap_or(0),
                    timestamp: chrono::Utc::now(),
                });
                tools::dream::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            // ================================================================
            // GRAPH — unified graph/association/prediction tool (v2.2)
            // ================================================================
            "ghostlink" => {
                tools::ghostlink::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            // DEPRECATED (4.0): renamed to `ghostlink`. Hidden alias: still
            // dispatches, and on a Strata log each action runs the matching
            // GhostLink mode, so the two cannot disagree.
            "graph" => {
                warn!(
                    "Tool 'graph' is deprecated in 4.0. Use 'ghostlink' (modes: propose|bounty|weave|map|inspect|explore|predict|harden)."
                );
                tools::graph_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            "predict" => {
                warn!("Tool 'predict' is deprecated in v2.2. Use 'graph' (action='predict').");
                tools::predict::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            // DEPRECATED (v2.2): folded into `maintain` (action='restore').
            "restore" => {
                warn!("Tool 'restore' is deprecated in v2.2. Use 'maintain' (action='restore').");
                tools::restore::execute(&self.storage, request.arguments).await
            }

            // ================================================================
            // CONTEXT PACKETS (v1.8+) — `session_start` (renamed v2.2)
            // ================================================================
            "session_start" => {
                tools::session_context::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }
            // DEPRECATED (v2.2): renamed to `session_start`. Hidden alias.
            "session_context" => {
                warn!("Tool 'session_context' is deprecated in v2.2. Use 'session_start'.");
                tools::session_context::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }

            // ================================================================
            // AUTONOMIC TOOLS (v1.9+)
            // ================================================================
            // DEPRECATED (v2.2): folded into `memory_status` (view='retention').
            "memory_health" => {
                warn!(
                    "Tool 'memory_health' is deprecated in v2.2. Use 'memory_status' (view='retention')."
                );
                tools::health::execute(&self.storage, request.arguments).await
            }
            // DEPRECATED (v2.2): folded into `graph`. Hidden aliases.
            "memory_graph" => {
                warn!(
                    "Tool 'memory_graph' is deprecated in v2.2. Use 'graph' (action='memory_graph')."
                );
                tools::graph::execute(&self.storage, request.arguments).await
            }
            "composed_graph" => {
                warn!(
                    "Tool 'composed_graph' is deprecated in v2.2. Use 'graph' (action='recent'|'get'|'memory'|'neighbors'|'never_composed'|'bounty_mode'|'label')."
                );
                tools::composed_graph::execute(&self.storage, request.arguments).await
            }
            // DEPRECATED (v2.2): folded into `recall`. Hidden aliases.
            "deep_reference" | "cross_reference" => {
                warn!(
                    "Tool '{}' is deprecated in v2.2. Use 'recall' (mode='reason').",
                    request.name
                );
                tools::cross_reference::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            "contradictions" => {
                warn!(
                    "Tool 'contradictions' is deprecated in v2.2. Use 'recall' (mode='contradictions')."
                );
                tools::contradictions::execute(&self.storage, request.arguments).await
            }

            // ================================================================
            // ACTIVE FORGETTING (v2.0.5) — top-down suppression
            // ================================================================
            "suppress" => tools::suppress::execute(&self.storage, request.arguments).await,

            // ================================================================
            // BLAST RADIUS — exact downstream reach of a cause/source record.
            // Dispatched by name; advertising it in tools/list additionally
            // needs a compact.rs registry entry + the catalog-count test
            // update. Retire routes every id through
            // the same pre-call Memory-PR gate as suppress/purge.
            // ================================================================
            "blast_radius" => {
                tools::blast_radius::execute(&self.storage, request.arguments).await
            }

            // Unit tests only: wait for the cognitive engine's mutex and do
            // nothing else, so a transport test can park a handler
            // deterministically without leaning on a real tool's locking.
            // No release build contains this arm.
            #[cfg(test)]
            "test_park_on_engine" => {
                let _engine = self.cognitive.lock().await;
                Ok(serde_json::json!({ "parked": false }))
            }

            name => {
                return Err(JsonRpcError::invalid_params(&format!(
                    "Unknown tool: {}",
                    name
                )));
            }
        };
        Ok(result)
    }

    /// Handle resources/list request
    async fn handle_resources_list(
        &self,
        params: Option<&serde_json::Value>,
        era: Era,
    ) -> Result<serde_json::Value, JsonRpcError> {
        reject_unknown_cursor(params)?;

        let resources = vec![
            // Memory resources
            ResourceDescription {
                uri: "memory://stats".to_string(),
                name: "Memory Statistics".to_string(),
                description: Some("Memory counts, average retention and health status of the log".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://recent".to_string(),
                name: "Recent Memories".to_string(),
                description: Some("Recently added memories (last 10)".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://decaying".to_string(),
                name: "Decaying Memories".to_string(),
                description: Some("Memories with FSRS retention below 0.5, lowest first (20 shown from the first 100 read)".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://due".to_string(),
                name: "Due for Review".to_string(),
                description: Some("Memories due for FSRS review today. Legacy engine only: Strata returns an error.".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            // Codebase resources
            ResourceDescription {
                uri: "codebase://structure".to_string(),
                name: "Codebase Structure".to_string(),
                description: Some("Remembered project structure and organization".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "codebase://patterns".to_string(),
                name: "Code Patterns".to_string(),
                description: Some("Remembered code patterns and conventions".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "codebase://decisions".to_string(),
                name: "Architectural Decisions".to_string(),
                description: Some("Remembered architectural and design decisions".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            // Consolidation resources
            ResourceDescription {
                uri: "memory://insights".to_string(),
                name: "Consolidation Insights".to_string(),
                description: Some("Insights recorded by consolidation runs. Legacy engine only: Strata returns an error.".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://consolidation-log".to_string(),
                name: "Consolidation Log".to_string(),
                description: Some("History of consolidation runs. Legacy engine only: empty on Strata, where consolidate is withheld.".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            // Prospective memory resources
            ResourceDescription {
                uri: "memory://intentions".to_string(),
                name: "Active Intentions".to_string(),
                description: Some("Open intentions waiting to be triggered".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://intentions/due".to_string(),
                name: "Triggered Intentions".to_string(),
                description: Some("Intentions that have been triggered or are overdue".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ];

        let result = ListResourcesResult { resources };
        let mut value = serde_json::to_value(result)
            .map_err(|e| JsonRpcError::internal_error(&e.to_string()))?;
        // Cache hints (`CacheableResult`): the list of advertised resources is
        // compile-time constant per binary, an hour is conservative;
        // `private` because feature flags can differ per install. Suppressed
        // for a client that declared a legacy revision per-request (no
        // newer-revision fields); forward-compatible for everyone else.
        if era != Era::LegacyDeclared
            && let Some(object) = value.as_object_mut()
        {
            object.insert("ttlMs".to_string(), serde_json::json!(3_600_000u64));
            object.insert("cacheScope".to_string(), serde_json::json!("private"));
        }
        Ok(value)
    }

    /// Handle resources/templates/list request.
    ///
    /// One URI template: the MCP App receipt card (`ui://vestige/receipt/{id}`,
    /// SEP-1865). Every other Vestige resource is a fixed `memory://` URI.
    /// The answer includes the template rather than `-32601`: the method
    /// belongs to the `resources` capability we declare, and a client
    /// discovering templates should learn what exists instead of
    /// "method not found". The conformance suite could not verify this surface
    /// while it errored (#175).
    fn handle_resources_templates_list(
        &self,
        params: Option<&serde_json::Value>,
        era: Era,
    ) -> Result<serde_json::Value, JsonRpcError> {
        reject_unknown_cursor(params)?;
        // Cache hints (`CacheableResult`): the template list is compile-time
        // constant per binary, an hour is conservative. `private` because a
        // build's feature set can differ per install. Suppressed for a client
        // that declared a legacy revision per-request.
        let mut result = serde_json::json!({
            "resourceTemplates": [
                {
                    "uriTemplate": resources::receipt_card::URI_TEMPLATE,
                    "name": "Retrieval Receipt Card",
                    "description": "MCP App rendering one receipt as server-side HTML: the ids it \
                     covers, suppressed entries with reasons, the recorded path, any mutation, \
                     and the FSRS retention floor. \
                     Read with resources/read using a receipt id from a tool result. \
                     No network; no memory content beyond the receipt.",
                    "mimeType": resources::receipt_card::MIME_TYPE,
                },
            ]
        });
        if era != Era::LegacyDeclared
            && let Some(object) = result.as_object_mut()
        {
            object.insert("ttlMs".to_string(), serde_json::json!(3_600_000u64));
            object.insert("cacheScope".to_string(), serde_json::json!("private"));
        }
        Ok(result)
    }

    /// Handle resources/read request
    async fn handle_resources_read(
        &self,
        params: Option<serde_json::Value>,
        era: Era,
    ) -> Result<serde_json::Value, JsonRpcError> {
        let request: ReadResourceRequest = match params {
            Some(p) => serde_json::from_value(p)
                .map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => return Err(JsonRpcError::invalid_params("Missing resource URI")),
        };

        let uri = &request.uri;
        // Normalize URI: strip provider prefix (e.g., "vestige/") for scheme matching
        // OpenCode and other MCP clients may send "vestige/memory://recent"
        // but we register resources as "memory://recent"
        let normalized_uri = uri.strip_prefix("vestige/").unwrap_or(uri);
        // The receipt card is the one non-JSON resource: an MCP App HTML
        // document (`text/html;profile=mcp-app`, SEP-1865) rendered
        // server-side from the persisted receipt.
        let is_receipt_card = resources::receipt_card::parse_uri(normalized_uri).is_some();
        let content = if is_receipt_card {
            resources::receipt_card::read(&self.storage, normalized_uri).await
        } else if normalized_uri.starts_with("memory://") {
            resources::memory::read(&self.storage, normalized_uri).await
        } else if normalized_uri.starts_with("codebase://") {
            resources::codebase::read(&self.storage, normalized_uri).await
        } else {
            Err(format!("Unknown resource scheme: {}", uri))
        };

        match content {
            Ok(text) => {
                let mime_type = if is_receipt_card {
                    resources::receipt_card::MIME_TYPE.to_string()
                } else {
                    "application/json".to_string()
                };
                let result = ReadResourceResult {
                    contents: vec![crate::protocol::messages::ResourceContent {
                        uri: uri.clone(),
                        mime_type: Some(mime_type),
                        text: Some(text),
                        blob: None,
                    }],
                };
                let mut value = serde_json::to_value(result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()))?;
                // Cache hints (`CacheableResult`). Resource content is
                // user-data backed and can change on any write, so the honest
                // hint is one second and `private` — the field pair is
                // required at 2026-07-28, and its value is advisory.
                // Suppressed for a client that declared a legacy revision
                // per-request (no newer-revision fields).
                if era != Era::LegacyDeclared
                    && let Some(object) = value.as_object_mut()
                {
                    object.insert("ttlMs".to_string(), serde_json::json!(1_000u64));
                    object.insert("cacheScope".to_string(), serde_json::json!("private"));
                }
                Ok(value)
            }
            Err(e) => {
                let not_found = e.to_ascii_lowercase().contains("unknown")
                    || e.to_ascii_lowercase().contains("not found");
                if not_found {
                    // 2026-07-28 retires `-32002` (resource not found):
                    // implementations of the modern revision MUST NOT emit
                    // it; the failure is expressed as `-32602` invalid
                    // params instead. Legacy clients keep the historical
                    // code they were built to match on.
                    if era == Era::Modern {
                        Err(JsonRpcError::invalid_params(&format!(
                            "Resource not found: {}",
                            uri
                        )))
                    } else {
                        Err(JsonRpcError::resource_not_found(uri))
                    }
                } else {
                    Err(JsonRpcError::internal_error(&e))
                }
            }
        }
    }

    /// Extract event data from tool results and emit to dashboard.
    fn emit_tool_event(
        &self,
        tool_name: &str,
        args: &Option<serde_json::Value>,
        result: &serde_json::Value,
    ) {
        if self.event_tx.is_none() {
            return;
        }
        let now = Utc::now();

        // v2.2: the unified `maintain` tool folds consolidate/dream/importance_score
        // (the three maintenance actions that emit). Normalize its name to the
        // effective action so the existing emit arms below fire unchanged. Old
        // standalone names still arrive verbatim and match directly.
        let tool_name = if tool_name == "maintain" {
            args.as_ref()
                .and_then(|a| a.get("action"))
                .and_then(|v| v.as_str())
                .unwrap_or("maintain")
        } else if tool_name == "recall" {
            // The unified `recall` tool fires SearchPerformed only for the lookup
            // path (the former `search`). reason/contradictions do not emit, so
            // map them to a non-emitting name.
            match args
                .as_ref()
                .and_then(|a| a.get("mode"))
                .and_then(|v| v.as_str())
            {
                Some("reason") | Some("contradictions") => "recall_noemit",
                _ => "search", // lookup (default) → SearchPerformed
            }
        } else {
            tool_name
        };

        match tool_name {
            // -- smart_ingest: memory created/updated --
            "remember" | "smart_ingest" | "ingest" | "session_checkpoint" => {
                // Single mode: result has "decision" (create/update/supersede/reinforce/merge/replace/add_context)
                if let Some(decision) = result.get("decision").and_then(|a| a.as_str()) {
                    let id = result
                        .get("nodeId")
                        .or(result.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let preview = result
                        .get("contentPreview")
                        .or(result.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    match decision {
                        "create" => {
                            let node_type = result
                                .get("nodeType")
                                .and_then(|v| v.as_str())
                                .unwrap_or("fact")
                                .to_string();
                            let tags = result
                                .get("tags")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|t| t.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default();
                            self.emit(VestigeEvent::MemoryCreated {
                                id,
                                content_preview: preview,
                                node_type,
                                tags,
                                timestamp: now,
                            });
                        }
                        "update" | "supersede" | "reinforce" | "merge" | "replace"
                        | "add_context" => {
                            self.emit(VestigeEvent::MemoryUpdated {
                                id,
                                content_preview: preview,
                                field: decision.to_string(),
                                timestamp: now,
                            });
                        }
                        _ => {}
                    }
                }
                // Batch mode: result has "results" array
                if let Some(results) = result.get("results").and_then(|r| r.as_array()) {
                    for item in results {
                        let decision = item.get("decision").and_then(|a| a.as_str()).unwrap_or("");
                        let id = item
                            .get("nodeId")
                            .or(item.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let preview = item
                            .get("contentPreview")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if decision == "create" {
                            self.emit(VestigeEvent::MemoryCreated {
                                id,
                                content_preview: preview,
                                node_type: "fact".to_string(),
                                tags: vec![],
                                timestamp: now,
                            });
                        } else if !decision.is_empty() {
                            self.emit(VestigeEvent::MemoryUpdated {
                                id,
                                content_preview: preview,
                                field: decision.to_string(),
                                timestamp: now,
                            });
                        }
                    }
                }
            }

            // -- memory: get/delete/promote/demote --
            "memory" | "promote_memory" | "demote_memory" | "delete_knowledge"
            | "get_memory_state" => {
                let action = args
                    .as_ref()
                    .and_then(|a| a.get("action"))
                    .and_then(|a| a.as_str())
                    .unwrap_or(if tool_name == "promote_memory" {
                        "promote"
                    } else if tool_name == "demote_memory" {
                        "demote"
                    } else if tool_name == "delete_knowledge" {
                        "delete"
                    } else {
                        ""
                    });
                let id = args
                    .as_ref()
                    .and_then(|a| a.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                match action {
                    "delete" | "purge"
                        if result
                            .get("success")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false) =>
                    {
                        let node_id = result
                            .get("nodeId")
                            .and_then(|value| value.as_str())
                            .unwrap_or(&id)
                            .to_string();
                        self.emit(VestigeEvent::MemoryDeleted {
                            id: node_id,
                            timestamp: now,
                        });
                    }
                    "promote" => {
                        let retention = result
                            .get("newRetention")
                            .or(result.get("retrievalStrength"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        self.emit(VestigeEvent::MemoryPromoted {
                            id,
                            new_retention: retention,
                            timestamp: now,
                        });
                    }
                    "demote" => {
                        let retention = result
                            .get("newRetention")
                            .or(result.get("retrievalStrength"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        self.emit(VestigeEvent::MemoryDemoted {
                            id,
                            new_retention: retention,
                            timestamp: now,
                        });
                    }
                    "get" => {
                        if let Some((shown_id, reason)) = shown_memory(tool_name, args, result) {
                            self.emit(VestigeEvent::MemoryShown {
                                id: shown_id,
                                by: self
                                    .client_name
                                    .get()
                                    .cloned()
                                    .unwrap_or_else(|| "Your agent".to_string()),
                                reason,
                                timestamp: now,
                            });
                        }
                    }
                    _ => {}
                }
            }

            // -- search --
            "search" | "recall" | "semantic_search" | "hybrid_search" => {
                let query = args
                    .as_ref()
                    .and_then(|a| a.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let results = result.get("results").and_then(|r| r.as_array());
                let result_count = results.map(|r| r.len()).unwrap_or(0);
                let result_ids: Vec<String> = results
                    .map(|r| {
                        r.iter()
                            .filter_map(|item| {
                                item.get("id").and_then(|v| v.as_str()).map(String::from)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let duration_ms = result
                    .get("durationMs")
                    .or(result.get("duration_ms"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                self.emit(VestigeEvent::SearchPerformed {
                    query,
                    result_count,
                    result_ids,
                    duration_ms,
                    timestamp: now,
                });
            }

            // -- dream --
            "dream" => {
                let replayed = result
                    .get("memoriesReplayed")
                    .or(result.get("memories_replayed"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                // On Strata, connectionsFound counts the recorded edges the
                // dream replayed; the connections it found that were not
                // there before are stats.new_connections_found. The event
                // means the new ones.
                let connections = result
                    .get("stats")
                    .and_then(|stats| stats.get("new_connections_found"))
                    .or(result.get("connectionsFound"))
                    .or(result.get("connections_found"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let insights = result
                    .get("insightsGenerated")
                    .or(result.get("insights"))
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let duration_ms = result
                    .get("durationMs")
                    .or(result.get("duration_ms"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                self.emit(VestigeEvent::DreamCompleted {
                    memories_replayed: replayed,
                    connections_found: connections,
                    insights_generated: insights,
                    duration_ms,
                    timestamp: now,
                });
            }

            // -- consolidate --
            "consolidate" => {
                let processed = result
                    .get("nodesProcessed")
                    .or(result.get("nodes_processed"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let decay = result
                    .get("decayApplied")
                    .or(result.get("decay_applied"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let embeddings = result
                    .get("embeddingsGenerated")
                    .or(result.get("embeddings_generated"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let duration_ms = result
                    .get("durationMs")
                    .or(result.get("duration_ms"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                self.emit(VestigeEvent::ConsolidationCompleted {
                    nodes_processed: processed,
                    decay_applied: decay,
                    embeddings_generated: embeddings,
                    duration_ms,
                    timestamp: now,
                });
            }

            // -- importance_score --
            "importance_score" => {
                let preview = args
                    .as_ref()
                    .and_then(|a| a.get("content"))
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        if s.len() > 100 {
                            format!("{}...", &s[..s.floor_char_boundary(100)])
                        } else {
                            s.to_string()
                        }
                    })
                    .unwrap_or_default();
                let composite = result
                    .get("compositeScore")
                    .or(result.get("composite_score"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let channels = result.get("channels").or(result.get("breakdown"));
                let novelty = channels
                    .and_then(|c| c.get("novelty"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let arousal = channels
                    .and_then(|c| c.get("arousal"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let reward = channels
                    .and_then(|c| c.get("reward"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let attention = channels
                    .and_then(|c| c.get("attention"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                self.emit(VestigeEvent::ImportanceScored {
                    memory_id: None, // importance_score tool runs on arbitrary content
                    content_preview: preview,
                    composite_score: composite,
                    novelty,
                    arousal,
                    reward,
                    attention,
                    timestamp: now,
                });
            }

            // Other tools don't emit events
            _ => {}
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod unified_operation_tests {
    use super::*;

    #[test]
    fn nested_native_run_id_survives_public_six_tool_envelope() {
        for field in ["runId", "run_id"] {
            let mut params = serde_json::json!({"action":"get", "id":"mem-example"});
            params[field] = serde_json::json!("native-hook-run");
            let request = CallToolRequest {
                name: "curate".into(),
                arguments: Some(serde_json::json!({"action":"memory", "params":params})),
            };
            let operation = unified_operation(&request).unwrap();
            assert_eq!(request.name, "curate");
            assert_eq!(operation.name, "memory");
            assert_eq!(
                crate::trace_recorder::run_id_for(&operation.arguments),
                "native-hook-run"
            );
            assert_eq!(
                request.arguments.as_ref().unwrap()["params"][field],
                "native-hook-run"
            );
        }
    }
}
