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
         (b) Never-composed detected: list pairs ghostlink(mode='propose') proves from recorded \
         edges (bridge) or forces with no recorded relation (divergent), or write 'None.' \
         (c) Recommendation: what the user should DO, as a concrete executable action. \
         If your draft begins 'Memory A says X. Memory B says Y.' STOP and rewrite.\
         \n\nBLOCKING PHRASE: If retrieved high-trust memories (retention > 0.7, reps > 0) \
         contradict what you were about to say, start your response with 'Vestige is blocking this:' \
         and surface the contradiction before proceeding. FSRS scores reflect memory history, not truth; verify against current evidence.\
         \n\nFEEDBACK: If the user confirms a memory was helpful, call memory(action='promote'). \
         If they correct it, call memory(action='demote'). Do not ask permission, just act."
            .to_string()
    } else {
        "Vestige is the Causal Proof Engine and the operating system for AI agents. Its kernel, \
         Strata, is an append-only, hash-chained, signed log: zero vectors, zero RAG, \
         no lookalike text. Memories are found by exact handle (a full id or an exact tag) and every \
         result carries its proof (memory id, edge path, receipt). \
         Compose retrievals into recommendations \
         rather than listing their contents when the user is making a decision. \
         On user feedback, call memory(action='promote') for helpful retrievals and \
         memory(action='demote') for wrong ones — do not ask permission, just act."
            .to_string()
    };
    instructions.push_str("\nIn a codebase, save durable conventions and design decisions with codebase(action='remember_pattern'|'remember_decision', files=['path.rs#symbol']); anchored code memories self-check against the source and are flagged when the code drifts.");
    instructions.push_str("\nghostlink(mode='propose') surfaces never-composed memory pairs from recorded edges, each with its proof; record what a tested pair showed with mode='weave'.");
    instructions.push_str("\nDiscover all available actions with memory_status(view='tools'); pass tool='<name>' for its exact schema. Choose calls that serve the task; no tool-call quota is required.");
    instructions.push_str("\nTo show the user one memory, read it with memory(action='get', id, reason='what you will tell them'): every open Vestige dashboard goes inside that memory and shows your words, and the reply carries its dashboard link.");
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
            tool.description = Some("Find memories by exact handle: a full memory id or an exact tag; a unique id prefix of 8+ characters also resolves. Free text and the 'reason' / 'contradictions' modes return similarity_disabled.".to_string());
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
            "'lookup' (default): by handle. 'reason' and 'contradictions' return similarity_disabled in 4.0."
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
                    "Cognitive session lifecycle: boot, intentions, status & lock-free projection. Parameters must be flat JSON keys. Do not nest arguments.".to_string(),
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
                    "Single gated write and inline backfill. Parameters must be flat JSON keys. Do not nest arguments. Anchors codebase records, runs salience backfill inline, and recompiles fences on 'pinned'.".to_string(),
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
                    "Root-cause investigation, gather, and bisect engine. action='walk' traces backwards from from_handle and attaches forgotten lessons; action='prove' executes deterministic bisect and seals signed witness proof cards.".to_string(),
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
                    "Graph & state mutation: promote, demote, edit, link, bridge ghostlinks, suppress, and connect memory identities.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::curate_unified::schema()),
                ..Default::default()
            },
            ToolDescription {
                name: "precedent".to_string(),
                title: Some("Precedent".to_string()),
                annotations: Some(ToolAnnotations {
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: false,
                }),
                description: Some(
                    "Guard stops, receipts, and cryptographic verification: operator stops, receipts, replay, BLAKE3/Ed25519 verify_log, scan_secrets, and selftest.".to_string(),
                ),
                input_schema: tools::compact::of(&tools::precedent::schema()),
                ..Default::default()
            },
        ]
    }

    /// Dispatch incoming tool calls to the 6 unified cognitive engines,
    /// preserving backward-compatible routing for legacy client calls.
    async fn dispatch_tool(
        &self,
        request: CallToolRequest,
    ) -> Result<Result<serde_json::Value, String>, JsonRpcError> {
        let tool_name = request.name.as_str();
        let args = request.arguments;

        let result = match tool_name {
            // 1. SESSION
            "session" | "session_start" | "intention" | "memory_status" | "project" => {
                tools::session_unified::execute(&self.storage, &self.cognitive, args).await
            }

            // 2. REMEMBER
            "remember" | "smart_ingest" | "ingest" | "session_checkpoint" | "backfill" | "dedup" => {
                tools::remember_unified::execute(&self.storage, &self.cognitive, args).await
            }

            // 3. RECALL
            "recall" | "memory_get" | "codebase_get_context" | "search" => {
                tools::recall::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    args,
                )
                .await
            }

            // 4. CAUSAL_WALK & PROVE
            "causal_walk" | "gather" | "forgotten_lesson" | "ingest_repo" | "verify" => {
                tools::causal_walk::execute(&self.storage, args).await
            }

            // 5. CURATE
            "curate" | "memory" | "memory_edit" | "promote_memory" | "purge" | "suppress"
            | "graph" | "ghostlink" | "connect_identities" => {
                tools::curate_unified::execute(&self.storage, &self.cognitive, args).await
            }

            // 6. PRECEDENT
            "precedent" | "stops" | "receipt" | "replay" | "verify_log" | "scan_secrets"
            | "selftest" => {
                tools::precedent::execute(&self.storage, args).await
            }

            // Internal reflex hook transport
            "reflex" => {
                tools::reflex::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    &self.conn,
                    args,
                )
                .await
            }

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
            "smart_ingest" | "ingest" | "session_checkpoint" => {
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
