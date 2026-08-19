//! MCP Server Core
//!
//! Handles the main MCP server logic, routing requests to appropriate
//! tool and resource handlers.

use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, warn};

use crate::cognitive::CognitiveEngine;
use crate::dashboard::events::VestigeEvent;
use crate::protocol::messages::{
    CallToolRequest, CallToolResult, InitializeRequest, InitializeResult, ListResourcesResult,
    ListToolsResult, ReadResourceRequest, ReadResourceResult, ResourceDescription,
    ServerCapabilities, ServerInfo, ToolDescription,
};
use crate::protocol::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, MCP_VERSION};
use crate::resources;
use crate::tools;
use vestige_core::{OutputConfig, Storage, VestigeConfig};

/// Build the MCP `instructions` string injected into every connecting client's
/// system prompt.
///
/// Default ("minimal", 3 sentences) is safe for any user: competitive coders,
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
    if mode.eq_ignore_ascii_case("full") {
        "Vestige is your long-term cognitive memory AND reasoning engine, not a RAG database. \
         Every retrieval MUST be composed into a recommendation, never summarized.\
         \n\nCOMPOSITION MANDATE: When you receive memories from search, deep_reference, \
         cross_reference, or explore_connections, your response MUST follow this shape. \
         (a) Composing: [memory IDs], followed by a brief composition rationale \
         about how the memories relate, NOT a restatement of their contents). \
         (b) Never-composed detected: list combinations of retrieved memories that share \
         tags/topics but have never been referenced together, or write 'None.' \
         (c) Recommendation: what the user should DO, as a concrete executable action. \
         If your draft begins 'Memory A says X. Memory B says Y.' STOP and rewrite.\
         \n\nBLOCKING PHRASE: If retrieved high-trust memories (retention > 0.7, reps > 0) \
         contradict what you were about to say, start your response with 'Vestige is blocking this:' \
         and surface the contradiction verbatim before proceeding. FSRS trust overrides fresh guesses.\
         \n\nFEEDBACK: If the user confirms a memory was helpful, call memory(action='promote'). \
         If they correct it, call memory(action='demote'). Do not ask permission, just act."
            .to_string()
    } else {
        "Vestige is your long-term memory system. Compose retrievals into recommendations \
         rather than listing their contents when the user is making a decision. \
         On user feedback, call memory(action='promote') for helpful retrievals and \
         memory(action='demote') for wrong ones — do not ask permission, just act."
            .to_string()
    }
}

fn supported_protocol_versions() -> &'static [&'static str] {
    &["2024-11-05", "2025-03-26", "2025-06-18", MCP_VERSION]
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

/// MCP Server implementation
pub struct McpServer {
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    initialized: bool,
    /// Tool call counter for inline consolidation trigger (every 100 calls)
    tool_call_count: AtomicU64,
    /// Optional event broadcast channel for dashboard real-time updates.
    event_tx: Option<broadcast::Sender<VestigeEvent>>,
    /// Resolved output config from `<data_dir>/vestige.toml` (Phase 2). Tools
    /// use it as the fallback for detail/limit when no explicit MCP param is
    /// given; explicit params always win.
    output_config: Arc<OutputConfig>,
}

/// Load `vestige.toml` from the storage's data directory and resolve it to an
/// effective [`OutputConfig`]. A missing/malformed file yields the built-in
/// default, which preserves historical behavior.
fn load_output_config(storage: &Arc<Storage>) -> Arc<OutputConfig> {
    let config = VestigeConfig::load_from_data_dir(storage.data_dir());
    Arc::new(config.output())
}

impl McpServer {
    #[allow(dead_code)]
    pub fn new(storage: Arc<Storage>, cognitive: Arc<Mutex<CognitiveEngine>>) -> Self {
        let output_config = load_output_config(&storage);
        Self {
            storage,
            cognitive,
            initialized: false,
            tool_call_count: AtomicU64::new(0),
            event_tx: None,
            output_config,
        }
    }

    /// Create an MCP server that broadcasts events to the dashboard.
    pub fn new_with_events(
        storage: Arc<Storage>,
        cognitive: Arc<Mutex<CognitiveEngine>>,
        event_tx: broadcast::Sender<VestigeEvent>,
    ) -> Self {
        let output_config = load_output_config(&storage);
        Self {
            storage,
            cognitive,
            initialized: false,
            tool_call_count: AtomicU64::new(0),
            event_tx: Some(event_tx),
            output_config,
        }
    }

    /// Emit an event to the dashboard (no-op if no event channel).
    fn emit(&self, event: VestigeEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Handle an incoming JSON-RPC request
    pub async fn handle_request(&mut self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        debug!("Handling request: {}", request.method);

        if request.id.is_none() {
            if request.method != "notifications/initialized" {
                debug!("Dropping JSON-RPC notification '{}'", request.method);
            }
            return None;
        }

        // Check initialization for non-initialize requests
        if !self.initialized
            && request.method != "initialize"
            && request.method != "notifications/initialized"
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

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params).await,
            "notifications/initialized" => Err(JsonRpcError::invalid_request(
                "notifications/initialized must be sent without an id",
            )),
            "tools/list" => self.handle_tools_list().await,
            "tools/call" => self.handle_tools_call(request.params).await,
            "resources/list" => self.handle_resources_list().await,
            "resources/read" => self.handle_resources_read(request.params).await,
            "ping" => Ok(serde_json::json!({})),
            method => {
                warn!("Unknown method: {}", method);
                Err(JsonRpcError::method_not_found())
            }
        };

        Some(match result {
            Ok(result) => JsonRpcResponse::success(request.id, result),
            Err(error) => JsonRpcResponse::error(request.id, error),
        })
    }

    /// Handle initialize request
    async fn handle_initialize(
        &mut self,
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

        let negotiated_version =
            if supported_protocol_versions().contains(&request.protocol_version.as_str()) {
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

        self.initialized = true;
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
            },
            instructions: Some(build_instructions()),
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self) -> Result<serde_json::Value, JsonRpcError> {
        // v2.3: 14 advertised tools after adding the controlled `receipt`
        // surface and retaining the distinct flagship `backfill` primitive.
        // 22 deprecated/folded names still work as hidden redirects in
        // handle_tools_call. See docs/launch/tool-consolidation-v2.2.0.md.
        let mut tools = vec![
            // ================================================================
            // RECALL — unified retrieval tool (v2.2). HOT PATH.
            // Folds search + deep_reference + cross_reference + contradictions.
            // mode='lookup' (default) is a zero-overhead pass-through to search.
            // ================================================================
            ToolDescription {
                name: "recall".to_string(),
                description: Some("Retrieve from memory. Modes: 'lookup' (default — fast hybrid search: keyword + semantic + convex fusion; retrieval is audit-only, and promote is the explicit usefulness signal), 'reason' (deep cognitive reasoning across memories with FSRS-6 trust scoring, spreading activation, supersession, and contradiction analysis; use when accuracy matters, needs 'query'), 'contradictions' (surface trust-weighted disagreement pairs for a 'topic'). Default mode is fast — only 'reason' pays the deep-analysis cost.".to_string()),
                input_schema: tools::recall::schema(),
                ..Default::default()
            },
            ToolDescription {
                name: "receipt".to_string(),
                description: Some("Inspect a persisted memory receipt or run a controlled post-retrieval context ablation. Actions: 'get' returns the receipt and a privacy-safe frozen-capsule summary; 'replay' compares the exact final recorded evidence pack with the same pack minus specified receipt-local slots. Replay never reruns search, backfills candidates, expands the graph, calls a model, or establishes causality.".to_string()),
                input_schema: tools::receipt::schema(),
                ..Default::default()
            },
            // ================================================================
            // UNIFIED TOOLS (v1.1+)
            // ================================================================
            ToolDescription {
                name: "memory".to_string(),
                description: Some("Unified memory management tool. Actions: 'get' (retrieve full node), 'purge' (irreversibly remove content/embeddings with confirm=true), 'delete' (legacy alias for purge), 'state' (get accessibility state), 'promote' (thumbs up — increases retrieval strength), 'demote' (thumbs down — decreases retrieval strength, does NOT delete), 'edit' (update content in-place, preserves FSRS state).".to_string()),
                input_schema: tools::memory_unified::schema(),
                ..Default::default()
            },
            ToolDescription {
                name: "codebase".to_string(),
                description: Some("Unified codebase tool. Actions: 'remember_pattern' (store code pattern), 'remember_decision' (store architectural decision), 'get_context' (retrieve patterns and decisions).".to_string()),
                input_schema: tools::codebase_unified::schema(),
                ..Default::default()
            },
            ToolDescription {
                name: "intention".to_string(),
                description: Some("Unified intention management tool. Actions: 'set' (create), 'check' (find triggered), 'update' (complete/snooze/cancel), 'list' (show intentions).".to_string()),
                input_schema: tools::intention_unified::schema(),
                ..Default::default()
            },
            // ================================================================
            // CORE MEMORY (v1.7: smart_ingest absorbs ingest + checkpoint)
            // ================================================================
            ToolDescription {
                name: "smart_ingest".to_string(),
                description: Some("INTELLIGENT memory ingestion with Prediction Error Gating. Single mode: provide 'content' to auto-decide CREATE/UPDATE/SUPERSEDE. Batch mode: provide 'items' array (max 20) for session-end saves — each item runs the full cognitive pipeline (importance scoring, intent detection, synaptic tagging).".to_string()),
                input_schema: tools::smart_ingest::schema(),
                ..Default::default()
            },
            // ================================================================
            // EXTERNAL-SOURCE CONNECTORS (#57)
            // ================================================================
            ToolDescription {
                name: "source_sync".to_string(),
                description: Some("Index an external system into Vestige as a durable, offline, semantically-searchable index that cites back to the canonical record. GitHub: source='github', repo='owner/name' (auth via GITHUB_TOKEN env). Redmine: source='redmine', project='<id>' (host via REDMINE_URL, auth via REDMINE_API_KEY env). Idempotent: re-running updates changed issues without duplicating; set reconcile=true to tombstone issues removed upstream.".to_string()),
                input_schema: tools::source_sync::schema(),
                ..Default::default()
            },
            // ================================================================
            // STATUS / TEMPORAL — unified `memory_status` tool (v2.2)
            // Folds system_status + memory_health + memory_timeline +
            // memory_changelog into one view-dispatched surface, plus the
            // full-store hygiene-statistics view.
            // ================================================================
            ToolDescription {
                name: "memory_status".to_string(),
                description: Some("Memory status & history. Views: 'health' (default — full system health + stats + FSRS preview + cognitive-module health + warnings + recommendations), 'retention' (lightweight retention dashboard: avg, distribution, trend), 'timeline' (browse memories chronologically, grouped by day), 'changelog' (audit trail of memory state changes — per-memory transitions or system-wide), 'stats' (full-store hygiene counts by type/tag/age/retention/lifecycle, bounded never-accessed and largest-node details, and recent tag-operation audit).".to_string()),
                input_schema: tools::memory_status::schema(),
                ..Default::default()
            },
            // ================================================================
            // MAINTAIN — unified maintenance/lifecycle tool (v2.2)
            // Folds consolidate + dream + gc + importance_score + backup +
            // export + restore into one action-dispatched surface.
            // ================================================================
            ToolDescription {
                name: "maintain".to_string(),
                description: Some("Memory maintenance & lifecycle. Actions: 'consolidate' (run FSRS-6 decay/embedding cycle), 'dream' (replay memories → insights/connections + strengthen patterns), 'gc' (garbage-collect stale memories; dry_run=true by default for safety), 'importance_score' (4-channel neuroscience score for 'content'), 'backup' (SQLite DB backup), 'export' (memories as JSON/JSONL with tag/date filters), 'restore' (restore from a JSON backup at 'path').".to_string()),
                input_schema: tools::maintain::schema(),
                ..Default::default()
            },
            // ================================================================
            // DEDUP / MERGE / SUPERSEDE — unified `dedup` tool (v2.2)
            // Folds find_duplicates + the 7 Phase-3 merge tools into one
            // action-dispatched surface. Diff-previewed, confidence-gated,
            // reversible, never silent; bitemporal-never-delete preserved.
            // ================================================================
            ToolDescription {
                name: "dedup".to_string(),
                description: Some("Deduplication & merge/supersede. Actions: 'scan' (default — surface duplicate clusters via cosine + merge candidates via Fellegi-Sunter, read-only), 'plan_merge' (preview a reversible merge plan for 2+ member_ids → plan_id), 'plan_supersede' (preview superseding old_id with new_id → plan_id), 'apply' (execute a plan_id; 'possible'/'non_match' need confirm=true), 'undo' (reverse an operation_id, or omit to list the reflog), 'protect' (pin a memory against auto-merge/supersede/forget), 'policy' (get/set Fellegi-Sunter thresholds). Old memories are invalidated, never deleted.".to_string()),
                input_schema: tools::dedup::unified_schema(),
                ..Default::default()
            },
            // ================================================================
            // COGNITIVE TOOLS (v1.5+)
            // (dream folded into `maintain` action='dream' in v2.2)
            // ================================================================
            // ================================================================
            // GRAPH — unified graph/association/prediction tool (v2.2)
            // Folds explore_connections + predict + memory_graph + composed_graph.
            // ================================================================
            ToolDescription {
                name: "graph".to_string(),
                description: Some("Memory graph & associations. Actions: 'chain' (reasoning path from→to), 'associations' (related memories via spreading activation, needs 'from'), 'bridges' (connectors between from/to), 'predict' (what memories you'll need next, from 'context'), 'memory_graph' (force-directed subgraph for viz, from center_id or query), 'recent'/'get'/'memory'/'neighbors'/'never_composed'/'bounty_mode' (composition topology), 'label' (record a composition outcome — the only write).".to_string()),
                input_schema: tools::graph_unified::schema(),
                ..Default::default()
            },
            // ================================================================
            // RESTORE TOOL (v1.5+)
            // (folded into `maintain` action='restore' in v2.2)
            // ================================================================
            // ================================================================
            // CONTEXT PACKETS (v1.8+)
            // ================================================================
            ToolDescription {
                name: "session_start".to_string(),
                description: Some("One-call session initialization. Combines search, intentions, status, predictions, and codebase context into a single token-budgeted response. Call this once at the start of a session instead of 5 separate calls. (Renamed from 'session_context' in v2.2.)".to_string()),
                input_schema: tools::session_context::schema(),
                ..Default::default()
            },
            // ================================================================
            // AUTONOMIC TOOLS (v1.9+)
            // (memory_health → `memory_status` view='retention';
            //  memory_graph + composed_graph → `graph`, all in v2.2)
            // ================================================================
            // ================================================================
            // DEEP REFERENCE (v2.0.4+) — folded into `recall` (mode='reason' /
            // 'contradictions') in v2.2. deep_reference/cross_reference/
            // contradictions remain hidden dispatch aliases.
            // ================================================================
            // ================================================================
            // ACTIVE FORGETTING (v2.0.5) — top-down suppression
            // Anderson et al. 2025 Nat Rev Neurosci + Davis Rac1
            // ================================================================
            ToolDescription {
                name: "suppress".to_string(),
                description: Some("Actively suppress a memory via top-down inhibitory control (Anderson 2025 SIF + Davis Rac1). Distinct from delete: the memory persists but is inhibited from retrieval and actively decays. Each call compounds. A background Rac1 worker cascades decay to co-activated neighbors. Reversible within 24 hours via reverse=true.".to_string()),
                input_schema: tools::suppress::schema(),
                ..Default::default()
            },
            // ================================================================
            // RETROACTIVE SALIENCE BACKFILL — Cai 2024 Nature
            // "Memory with hindsight": failure -> backward causal reach.
            // A flagship v2.2 capability, kept as its own advertised tool — it
            // is a distinct cognitive primitive (backward causal promotion),
            // not a maintenance op that folds into `maintain`.
            // ================================================================
            ToolDescription {
                name: "backfill".to_string(),
                description: Some("Memory with hindsight. When a FAILURE (bug/crash/regression) is recorded, reach BACKWARD in time and promote the quiet earlier memory that caused it — the root cause a vector search structurally cannot surface because it isn't similar to the failure, only causally upstream (shares an entity: same file/env-var/service). Faithful port of Cai 2024 Nature; backward-only by construction. Pass failure_id (or it auto-finds the latest failure), manual=true to force, promote=false for a dry run.".to_string()),
                input_schema: tools::backfill::schema(),
                ..Default::default()
            },
        ];

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
                "graph" => Some(250_000),
                _ => None,
            };
            if let Some(n) = max_chars {
                let mut meta = serde_json::Map::new();
                meta.insert(
                    "anthropic/maxResultSizeChars".to_string(),
                    serde_json::Value::from(n),
                );
                tool.meta = Some(serde_json::Value::Object(meta));
            }
        }

        let result = ListToolsResult { tools };
        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
    }

    /// Handle tools/call request
    async fn handle_tools_call(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, JsonRpcError> {
        let request: CallToolRequest = match params {
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

        // Agent Black Box: record the opening mcp.call event for this tool
        // invocation. run_id groups the events of one agent turn; it is derived
        // from the args (an explicit runId, or a fresh id) so record_result below
        // can attach the downstream memory events (retrieve/suppress/veto) to the
        // same run.
        let trace_run_id = crate::trace_recorder::run_id_for(&saved_args);
        if trace_enabled() {
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
            &request.name,
            &saved_args,
            mode,
        );
        match pending {
            Ok(Some(content)) => {
                // The pre-gate already emitted MemoryPrOpened. `success`
                // is false, so emitting the normal tool event cannot claim
                // a deletion or suppression that did not happen.
                self.emit_tool_event(&request.name, &saved_args, &content);
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: serde_json::to_string_pretty(&content)
                            .unwrap_or_else(|_| content.to_string()),
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
                        text: serde_json::to_string_pretty(&error_content)
                            .unwrap_or_else(|_| error_content.to_string()),
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
        let mut result = match request.name.as_str() {
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
            // DEPRECATED (v2.2): folded into `recall` (mode='lookup'). Hidden alias.
            "search" => {
                warn!(
                    "Tool 'search' is deprecated in v2.2. Use 'recall' (mode='lookup', the default)."
                );
                tools::search_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
            }
            "memory" => {
                tools::memory_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
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
                tools::intention_unified::execute(&self.storage, &self.cognitive, request.arguments)
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
            // External-source connectors (#57)
            // ================================================================
            "source_sync" => tools::source_sync::execute(&self.storage, request.arguments).await,

            // ================================================================
            // Retroactive Salience Backfill (Cai 2024 Nature) — flagship v2.2
            // ================================================================
            "backfill" => tools::backfill::execute(&self.storage, request.arguments).await,

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
            // DEPRECATED (v1.7): promote_memory → memory(action='promote')
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
                warn!(
                    "Tool '{}' is deprecated. Use 'recall' (mode='lookup') instead.",
                    request.name
                );
                tools::search_unified::execute(
                    &self.storage,
                    &self.cognitive,
                    &self.output_config,
                    request.arguments,
                )
                .await
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
            "match_context" => tools::context::execute(&self.storage, request.arguments).await,

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
            "dedup" => tools::dedup::execute_unified(&self.storage, request.arguments).await,

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
            "graph" => {
                tools::graph_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            // DEPRECATED (v2.2): folded into `graph`. Hidden aliases.
            "explore_connections" => {
                warn!(
                    "Tool 'explore_connections' is deprecated in v2.2. Use 'graph' (action='chain'|'associations'|'bridges')."
                );
                tools::explore::execute(&self.storage, &self.cognitive, request.arguments).await
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

            name => {
                return Err(JsonRpcError::invalid_params(&format!(
                    "Unknown tool: {}",
                    name
                )));
            }
        };

        // ================================================================
        // DASHBOARD EVENT EMISSION (v2.0)
        // Emit real-time events to WebSocket clients after successful tool calls.
        // ================================================================
        if let Ok(ref mut content) = result {
            // Agent Black Box: inspect the successful result and record the
            // downstream memory events (retrieve/suppress/veto/dream) under the
            // same run_id as the opening mcp.call, so /api/traces, /api/receipts
            // and the trace:// resource are actually populated.
            if trace_enabled() {
                crate::trace_recorder::record_result(
                    &self.storage,
                    self.event_tx.as_ref(),
                    &trace_run_id,
                    &request.name,
                    content,
                );
                // Persist the receipt for this exact retrieval run, then attach
                // its stable reference to the structured response. The Black Box
                // can now answer "why did the agent do that?" from the same
                // evidence the tool actually used, rather than reconstructing an
                // explanation after the fact. Non-retrieval tools safely return
                // None and keep their existing response shape.
                if let Some(receipt) = crate::trace_recorder::build_and_save_receipt(
                    &self.storage,
                    &trace_run_id,
                    &request.name,
                    content,
                ) && let Some(obj) = content.as_object_mut()
                {
                    let receipt_id = receipt
                        .get("receipt_id")
                        .and_then(|value| value.as_str())
                        .map(String::from);
                    obj.entry("runId".to_string())
                        .or_insert_with(|| serde_json::json!(trace_run_id));
                    obj.insert("receiptId".to_string(), serde_json::json!(receipt_id));
                    obj.insert("receipt".to_string(), receipt);
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
                    &request.name,
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
            // Emit after receipt attachment and gating so the dashboard sees the
            // same final evidence and review state returned to the calling agent.
            self.emit_tool_event(&request.name, &saved_args, content);
        }

        let response = match result {
            Ok(content) => {
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: serde_json::to_string_pretty(&content)
                            .unwrap_or_else(|_| content.to_string()),
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

        if should_consolidate {
            let storage_clone = Arc::clone(&self.storage);
            let cognitive_clone = Arc::clone(&self.cognitive);
            tokio::spawn(async move {
                // Expire labile reconsolidation windows
                if let Ok(mut cog) = cognitive_clone.try_lock() {
                    let _expired = cog.reconsolidation.reconsolidate_expired();
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

    /// Handle resources/list request
    async fn handle_resources_list(&self) -> Result<serde_json::Value, JsonRpcError> {
        let resources = vec![
            // Memory resources
            ResourceDescription {
                uri: "memory://stats".to_string(),
                name: "Memory Statistics".to_string(),
                description: Some("Current memory system statistics and health status".to_string()),
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
                description: Some("Memories with low retention that need review".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://due".to_string(),
                name: "Due for Review".to_string(),
                description: Some("Memories scheduled for review today".to_string()),
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
                description: Some("Insights generated during memory consolidation".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            ResourceDescription {
                uri: "memory://consolidation-log".to_string(),
                name: "Consolidation Log".to_string(),
                description: Some("History of memory consolidation runs".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            // Prospective memory resources
            ResourceDescription {
                uri: "memory://intentions".to_string(),
                name: "Active Intentions".to_string(),
                description: Some(
                    "Future intentions (prospective memory) waiting to be triggered".to_string(),
                ),
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
        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
    }

    /// Handle resources/read request
    async fn handle_resources_read(
        &self,
        params: Option<serde_json::Value>,
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
        let content = if normalized_uri.starts_with("memory://") {
            resources::memory::read(&self.storage, normalized_uri).await
        } else if normalized_uri.starts_with("codebase://") {
            resources::codebase::read(&self.storage, normalized_uri).await
        } else {
            Err(format!("Unknown resource scheme: {}", uri))
        };

        match content {
            Ok(text) => {
                let result = ReadResourceResult {
                    contents: vec![crate::protocol::messages::ResourceContent {
                        uri: uri.clone(),
                        mime_type: Some("application/json".to_string()),
                        text: Some(text),
                        blob: None,
                    }],
                };
                serde_json::to_value(result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
            Err(e) => {
                if e.to_ascii_lowercase().contains("unknown")
                    || e.to_ascii_lowercase().contains("not found")
                {
                    Err(JsonRpcError::resource_not_found(uri))
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
                let connections = result
                    .get("connectionsFound")
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
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    /// Create a test server with temporary storage
    async fn test_server() -> (McpServer, TempDir) {
        let (storage, dir) = test_storage().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let server = McpServer::new(storage, cognitive);
        (server, dir)
    }

    /// Create a JSON-RPC request
    fn make_request(method: &str, params: Option<serde_json::Value>) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: method.to_string(),
            params,
        }
    }

    fn make_notification(method: &str, params: Option<serde_json::Value>) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params,
        }
    }

    fn init_params() -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
    }

    // ========================================================================
    // MEMORY PR GATE WIRING
    // ========================================================================

    /// Regression test for the dead seam: `gate_writes` and its unit tests
    /// existed, but nothing in production ever called it, so `ReviewMode` was
    /// inert and `memory_prs` only ever filled up inside `#[cfg(test)]`.
    ///
    /// This drives a real `tools/call` through `handle_request` and asserts a
    /// Memory PR actually lands. Paranoid mode is used deliberately so the test
    /// pins the WIRING, not the risk heuristic (which is covered by the unit
    /// tests in `trace_recorder`).
    #[tokio::test]
    async fn memory_pr_gate_is_wired_into_the_real_tool_path() {
        use vestige_core::MemoryPrStatus;

        let (storage, _dir) = test_storage().await;
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            r#"{"mode":"paranoid"}"#,
        )
        .unwrap();

        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new(storage.clone(), cognitive);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        assert_eq!(
            storage
                .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
                .unwrap()
                .len(),
            0,
            "no Memory PRs before the call"
        );

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "smart_ingest",
                    "arguments": { "content": "Staging was migrated to Postgres 17." }
                })),
            ))
            .await
            .unwrap();
        assert!(
            response.error.is_none(),
            "smart_ingest should succeed: {:?}",
            response.error
        );

        let prs = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(
            prs.len(),
            1,
            "a write through the real tool path must open a Memory PR in Paranoid mode"
        );

        // The calling agent must be told its write is held, otherwise it will
        // assume the memory is live and act on it.
        let text = serde_json::to_string(&response.result).unwrap();
        assert!(
            text.contains("memoryPrs"),
            "response must carry the opened PRs: {text}"
        );
        assert!(
            text.contains("memoryPrNotice"),
            "response must carry the human-readable notice: {text}"
        );
        // Truthfulness: this was a normal (non-destructive) write, so it must
        // report held=true and the notice must say quarantined.
        assert!(
            text.contains("\"held\":true"),
            "a suppressed write must report held=true: {text}"
        );
        assert!(
            text.contains("quarantined until"),
            "notice for a held write must say quarantined: {text}"
        );
    }

    /// The third leg of the mode matrix: Fast mode must open NO Memory PR
    /// through the real tool path — every write auto-commits.
    #[tokio::test]
    async fn fast_mode_opens_no_memory_pr() {
        use vestige_core::MemoryPrStatus;

        let (storage, _dir) = test_storage().await;
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            r#"{"mode":"fast"}"#,
        )
        .unwrap();

        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new(storage.clone(), cognitive);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "smart_ingest",
                    "arguments": { "content": "Fast mode write that must auto-commit." }
                })),
            ))
            .await
            .unwrap();
        assert!(response.error.is_none());

        assert_eq!(
            storage
                .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
                .unwrap()
                .len(),
            0,
            "Fast mode must never open a Memory PR"
        );
        let text = serde_json::to_string(&response.result).unwrap();
        assert!(
            !text.contains("memoryPrNotice"),
            "Fast mode must not attach a gating notice: {text}"
        );
    }

    /// Deprecated aliases share the same destructive policy; otherwise an
    /// older MCP client could bypass the canonical `memory` pre-gate.
    #[tokio::test]
    async fn legacy_delete_knowledge_is_pre_gated_on_the_real_mcp_path() {
        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory preserved through the legacy delete alias.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new(storage.clone(), cognitive);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "delete_knowledge",
                    "arguments": {
                        "id": node.id.clone(),
                        "confirm": true,
                        "reason": "exercise the pre-execution review gate"
                    }
                })),
            ))
            .await
            .unwrap();
        assert!(
            response.error.is_none(),
            "pending review is a valid MCP result"
        );
        assert!(
            storage.get_node(&node.id).unwrap().is_some(),
            "the legacy alias must not bypass pre-execution review"
        );

        let structured = response
            .result
            .as_ref()
            .and_then(|result| result.get("structuredContent"))
            .expect("structured tool result");
        assert_eq!(structured["action"], "delete_pending_review");
        assert_eq!(structured["pendingReview"], true);
        let prs = storage
            .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(prs.len(), 1);
        assert_eq!(prs[0].diff["pendingAction"], "delete");
    }

    /// Regression #117: purge must be intercepted on the real MCP path before
    /// `memory_unified::execute` can delete the node. The response and dashboard
    /// event must describe a pending review, not a completed deletion.
    #[tokio::test]
    async fn destructive_purge_is_pre_gated_on_the_real_mcp_path() {
        use crate::dashboard::events::VestigeEvent;
        use std::time::Duration;
        use vestige_core::MemoryPrStatus;

        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory that must survive pre-execution review.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let (event_tx, mut events) = tokio::sync::broadcast::channel(16);
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new_with_events(storage.clone(), cognitive, event_tx);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "memory",
                    "arguments": {
                        "action": "purge",
                        "id": node.id,
                        "confirm": true,
                        "reason": "exercise the pre-execution review gate",
                        "runId": "run_pre_execution_purge"
                    }
                })),
            ))
            .await
            .unwrap();
        assert!(
            response.error.is_none(),
            "a pending-review response is a valid MCP result"
        );

        let structured = response
            .result
            .as_ref()
            .and_then(|result| result.get("structuredContent"))
            .expect("structured tool result");
        assert_eq!(structured["success"], false);
        assert_eq!(structured["pendingReview"], true);
        assert_eq!(structured["action"], "purge_pending_review");
        assert_eq!(structured["nodeId"], node.id);
        assert!(
            storage.get_node(&node.id).unwrap().is_some(),
            "pre-execution gate must prevent the purge"
        );

        let prs = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(prs.len(), 1);
        assert_eq!(prs[0].diff["pendingAction"], "purge");
        assert_eq!(prs[0].diff["node"]["deleted"], false);

        let mut saw_pr_opened = false;
        for _ in 0..8 {
            match tokio::time::timeout(Duration::from_millis(100), events.recv()).await {
                Ok(Ok(VestigeEvent::MemoryPrOpened { .. })) => {
                    saw_pr_opened = true;
                }
                Ok(Ok(VestigeEvent::MemoryDeleted { .. })) => {
                    panic!("a blocked purge must not emit MemoryDeleted")
                }
                Ok(Ok(_)) => {}
                Ok(Err(_)) | Err(_) => break,
            }
        }
        assert!(saw_pr_opened, "pre-gate must announce the opened Memory PR");
    }

    /// Suppression changes retrieval influence immediately, so it is subject to
    /// the same production pre-execution gate as an irreversible purge.
    #[tokio::test]
    async fn suppress_is_pre_gated_on_the_real_mcp_path() {
        use vestige_core::MemoryPrStatus;

        let (storage, _dir) = test_storage().await;
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Memory that must not be inhibited before review.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new(storage.clone(), cognitive);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "suppress",
                    "arguments": { "id": node.id, "reason": "requires review first" }
                })),
            ))
            .await
            .unwrap();
        assert!(response.error.is_none());

        let structured = response
            .result
            .as_ref()
            .and_then(|result| result.get("structuredContent"))
            .expect("structured tool result");
        assert_eq!(structured["action"], "suppress_pending_review");
        assert_eq!(structured["pendingReview"], true);
        assert_eq!(
            storage
                .get_node(&node.id)
                .unwrap()
                .unwrap()
                .suppression_count,
            0,
            "pre-execution gate must not change retrieval influence"
        );
        let prs = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(prs.len(), 1);
        assert_eq!(prs[0].diff["pendingAction"], "suppress");
    }

    /// Fast mode is the documented explicit opt-out: it keeps legacy direct
    /// execution instead of manufacturing a pending review.
    #[tokio::test]
    async fn fast_mode_allows_destructive_call_on_the_real_mcp_path() {
        let (storage, _dir) = test_storage().await;
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            r#"{"mode":"fast"}"#,
        )
        .unwrap();
        let node = storage
            .ingest(vestige_core::IngestInput {
                content: "Fast mode purge target.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new(storage.clone(), cognitive);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "memory",
                    "arguments": { "action": "purge", "id": node.id, "confirm": true }
                })),
            ))
            .await
            .unwrap();
        let structured = response
            .result
            .as_ref()
            .and_then(|result| result.get("structuredContent"))
            .expect("structured tool result");
        assert_eq!(structured["success"], true);
        assert!(storage.get_node(&node.id).unwrap().is_none());
        assert!(
            storage
                .list_memory_prs(Some(vestige_core::MemoryPrStatus::Pending), 10)
                .unwrap()
                .is_empty(),
            "Fast mode must not pre-gate the mutation"
        );
    }

    /// Dashboard events must not announce a write as landed before the Memory
    /// PR is opened. The receipt/gate merge seam is deliberately exercised
    /// through the real MCP dispatch path, not by calling either helper alone.
    #[tokio::test]
    async fn memory_pr_event_precedes_memory_created_event() {
        use crate::dashboard::events::VestigeEvent;
        use std::time::Duration;

        let (storage, _dir) = test_storage().await;
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            r#"{"mode":"paranoid"}"#,
        )
        .unwrap();
        let (event_tx, mut events) = tokio::sync::broadcast::channel(16);
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let mut server = McpServer::new_with_events(storage, cognitive, event_tx);
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "smart_ingest",
                    "arguments": { "content": "A write whose dashboard order is audited." }
                })),
            ))
            .await
            .unwrap();
        assert!(response.error.is_none(), "smart_ingest should succeed");

        let mut observed = Vec::new();
        for _ in 0..8 {
            let event = tokio::time::timeout(Duration::from_millis(100), events.recv())
                .await
                .expect("expected a queued dashboard event")
                .expect("dashboard event channel should remain open");
            match event {
                VestigeEvent::MemoryPrOpened { .. } => observed.push("memory-pr-opened"),
                VestigeEvent::MemoryCreated { .. } => {
                    observed.push("memory-created");
                    break;
                }
                _ => {}
            }
        }
        let opened_at = observed
            .iter()
            .position(|event| *event == "memory-pr-opened")
            .expect("gating must open a Memory PR");
        let created_at = observed
            .iter()
            .position(|event| *event == "memory-created")
            .expect("emit_tool_event must publish the created memory");
        assert!(
            opened_at < created_at,
            "dashboard must observe the Memory PR before the memory-created event: {observed:?}"
        );
    }

    /// A corrupt or missing `review_mode.json` must never silently disable
    /// gating: it falls back to the default RiskGated.
    #[tokio::test]
    async fn review_mode_falls_back_to_risk_gated() {
        let (storage, _dir) = test_storage().await;
        assert_eq!(
            crate::trace_recorder::read_review_mode(&storage),
            vestige_core::ReviewMode::RiskGated,
            "missing file defaults to RiskGated"
        );

        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            "{ not valid json",
        )
        .unwrap();
        assert_eq!(
            crate::trace_recorder::read_review_mode(&storage),
            vestige_core::ReviewMode::RiskGated,
            "corrupt file defaults to RiskGated, never Fast"
        );

        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            r#"{"mode":"fast"}"#,
        )
        .unwrap();
        assert_eq!(
            crate::trace_recorder::read_review_mode(&storage),
            vestige_core::ReviewMode::Fast,
            "a valid mode is honored"
        );
    }

    // ========================================================================
    // TRACE GATE TESTS
    // ========================================================================

    #[test]
    fn test_parse_trace_enabled_defaults_on_and_honors_opt_out() {
        // Default ON: unset, empty, or malformed values all fail open.
        assert!(parse_trace_enabled(None));
        assert!(parse_trace_enabled(Some("")));
        assert!(parse_trace_enabled(Some("1")));
        assert!(parse_trace_enabled(Some("true")));
        assert!(parse_trace_enabled(Some("banana")));
        // Explicit opt-out only: 0/false/off/no (case-insensitive, trimmed).
        assert!(!parse_trace_enabled(Some("0")));
        assert!(!parse_trace_enabled(Some("false")));
        assert!(!parse_trace_enabled(Some("FALSE")));
        assert!(!parse_trace_enabled(Some("OFF")));
        assert!(!parse_trace_enabled(Some(" no ")));
    }

    // ========================================================================
    // INITIALIZATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_initialize_sets_initialized_flag() {
        let (mut server, _dir) = test_server().await;
        assert!(!server.initialized);

        let request = make_request(
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            })),
        );

        let response = server.handle_request(request).await;
        assert!(response.is_some());
        let response = response.unwrap();
        assert!(response.result.is_some());
        assert!(response.error.is_none());
        assert!(server.initialized);
    }

    #[tokio::test]
    async fn test_initialize_returns_server_info() {
        let (mut server, _dir) = test_server().await;
        // Send with current protocol version to get it back
        let params = serde_json::json!({
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "1.0" }
        });
        let request = make_request("initialize", Some(params));

        let response = server.handle_request(request).await.unwrap();
        let result = response.result.unwrap();

        assert_eq!(result["protocolVersion"], MCP_VERSION);
        assert_eq!(result["serverInfo"]["name"], "vestige");
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
        assert!(result["instructions"].is_string());
    }

    #[tokio::test]
    async fn test_initialize_unsupported_protocol_falls_back_to_latest() {
        let (mut server, _dir) = test_server().await;
        let params = serde_json::json!({
            "protocolVersion": "1.0.0",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "1.0" }
        });
        let request = make_request("initialize", Some(params));

        let response = server.handle_request(request).await.unwrap();
        let result = response.result.unwrap();

        assert_eq!(result["protocolVersion"], MCP_VERSION);
    }

    #[tokio::test]
    async fn test_initialize_missing_params_returns_error() {
        let (mut server, _dir) = test_server().await;
        let request = make_request("initialize", None);

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
        assert!(!server.initialized);
    }

    // ========================================================================
    // UNINITIALIZED SERVER TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_request_before_initialize_returns_error() {
        let (mut server, _dir) = test_server().await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert_eq!(error.code, -32003); // ServerNotInitialized
    }

    #[tokio::test]
    async fn test_ping_before_initialize_returns_error() {
        let (mut server, _dir) = test_server().await;

        let request = make_request("ping", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32003);
    }

    // ========================================================================
    // NOTIFICATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_initialized_notification_returns_none() {
        let (mut server, _dir) = test_server().await;

        // First initialize
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        // Send initialized notification
        let notification = make_notification("notifications/initialized", None);
        let response = server.handle_request(notification).await;

        // Notifications should return None
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn test_initialized_notification_with_id_returns_invalid_request() {
        let (mut server, _dir) = test_server().await;

        let request = make_request("notifications/initialized", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32600);
    }

    #[tokio::test]
    async fn test_notification_does_not_emit_response_or_side_effect() {
        let (mut server, _dir) = test_server().await;

        let notification = make_notification("initialize", None);
        let response = server.handle_request(notification).await;

        assert!(response.is_none());
        assert!(!server.initialized);
    }

    // ========================================================================
    // TOOLS/LIST TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_tools_list_returns_all_tools() {
        let (mut server, _dir) = test_server().await;

        // Initialize first
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        // v2.2 Tool Consolidation (Layer 1): 34 → 27 after `dedup` folds
        // find_duplicates + the 7 Phase-3 merge tools (8 → 1). Old names remain
        // dispatchable as hidden back-compat aliases but drop off the advertised list.
        assert_eq!(
            tools.len(),
            14,
            "Expected exactly 14 tools after v2.3 receipt replay integration \
             (12 consolidated: dedup + memory_status + graph + maintain + recall; \
             session_context renamed) plus `receipt` and the flagship `backfill`"
        );

        let tool_names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();

        // Unified tools
        // (search folded into `recall` mode='lookup' in v2.2)
        assert!(tool_names.contains(&"recall"));
        assert!(tool_names.contains(&"receipt"));
        assert!(tool_names.contains(&"memory"));
        assert!(tool_names.contains(&"codebase"));
        assert!(tool_names.contains(&"intention"));

        // Flagship retroactive-salience backfill stays advertised (not folded).
        assert!(tool_names.contains(&"backfill"));

        // Core memory (smart_ingest absorbs ingest + checkpoint in v1.7)
        assert!(tool_names.contains(&"smart_ingest"));

        // External-source connectors (#57)
        assert!(tool_names.contains(&"source_sync"));
        assert!(
            !tool_names.contains(&"ingest"),
            "ingest should be removed in v1.7"
        );
        assert!(
            !tool_names.contains(&"session_checkpoint"),
            "session_checkpoint should be removed in v1.7"
        );

        // Feedback merged into memory tool (v1.7)
        assert!(
            !tool_names.contains(&"promote_memory"),
            "promote_memory should be removed in v1.7"
        );
        assert!(
            !tool_names.contains(&"demote_memory"),
            "demote_memory should be removed in v1.7"
        );

        // Status / temporal — unified `memory_status` tool (v2.2).
        // system_status + memory_health + memory_timeline + memory_changelog
        // folded in; old names dispatch as hidden aliases but are off the list.
        assert!(tool_names.contains(&"memory_status"));
        for old in [
            "system_status",
            "memory_health",
            "memory_timeline",
            "memory_changelog",
        ] {
            assert!(
                !tool_names.contains(&old),
                "{old} should be folded into 'memory_status' in v2.2"
            );
        }
        assert!(
            !tool_names.contains(&"health_check"),
            "health_check should be removed in v1.7"
        );
        assert!(
            !tool_names.contains(&"stats"),
            "stats should be removed in v1.7"
        );
        // Maintenance / lifecycle — unified `maintain` tool (v2.2).
        // consolidate + dream + gc + importance_score + backup + export + restore
        // folded in; old names dispatch as hidden aliases but are off the list.
        assert!(tool_names.contains(&"maintain"));
        for old in [
            "consolidate",
            "dream",
            "gc",
            "importance_score",
            "backup",
            "export",
            "restore",
        ] {
            assert!(
                !tool_names.contains(&old),
                "{old} should be folded into 'maintain' in v2.2"
            );
        }

        // Dedup / merge / supersede — unified `dedup` tool (v2.2).
        // find_duplicates + the 7 Phase-3 merge tools folded in; still
        // dispatchable as hidden back-compat aliases, but off the advertised list.
        assert!(tool_names.contains(&"dedup"));
        for old in [
            "find_duplicates",
            "merge_candidates",
            "plan_merge",
            "plan_supersede",
            "apply_plan",
            "merge_undo",
            "protect",
            "merge_policy",
        ] {
            assert!(
                !tool_names.contains(&old),
                "{old} should be folded into 'dedup' in v2.2"
            );
        }

        // Cognitive tools (v1.5): explore_connections + predict → `graph`;
        // dream + restore → `maintain` (all v2.2). Nothing left advertised here.

        // Context packets (v1.8) — renamed session_context → session_start (v2.2)
        assert!(tool_names.contains(&"session_start"));
        assert!(
            !tool_names.contains(&"session_context"),
            "session_context renamed to 'session_start' in v2.2"
        );

        // Graph — unified `graph` tool (v2.2). explore_connections + predict +
        // memory_graph + composed_graph folded in; old names dispatch as hidden
        // aliases but are off the advertised list. (memory_health → memory_status.)
        assert!(tool_names.contains(&"graph"));
        for old in [
            "explore_connections",
            "predict",
            "memory_graph",
            "composed_graph",
        ] {
            assert!(
                !tool_names.contains(&old),
                "{old} should be folded into 'graph' in v2.2"
            );
        }

        // Retrieval — unified `recall` tool (v2.2). search + deep_reference +
        // cross_reference + contradictions folded in; old names dispatch as
        // hidden aliases but are off the advertised list.
        for old in [
            "search",
            "deep_reference",
            "cross_reference",
            "contradictions",
        ] {
            assert!(
                !tool_names.contains(&old),
                "{old} should be folded into 'recall' in v2.2"
            );
        }

        // Active forgetting (v2.0.5) — Anderson 2025 + Davis Rac1
        assert!(tool_names.contains(&"suppress"));
    }

    /// v2.2: the 8 tools folded into `dedup` must still dispatch (hidden
    /// back-compat aliases), i.e. they must NOT return the "Unknown tool"
    /// InvalidParams (-32602) error. Read-only/list-style actions are used so
    /// the call resolves without mutating or requiring extra setup.
    #[tokio::test]
    async fn test_deprecated_dedup_aliases_redirect() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        // (tool name, args) — all read-only / list-style so they resolve cleanly.
        let calls: Vec<(&str, serde_json::Value)> = vec![
            ("find_duplicates", serde_json::json!({})),
            ("merge_candidates", serde_json::json!({})),
            ("merge_undo", serde_json::json!({})), // no operation_id => lists the reflog
            ("merge_policy", serde_json::json!({})), // no args => returns current policy
            ("dedup", serde_json::json!({"action": "policy"})),
            ("dedup", serde_json::json!({})), // default action = scan
        ];

        for (name, args) in calls {
            let request = make_request(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": args })),
            );
            let response = server.handle_request(request).await.unwrap();
            // The call may succeed (result) or fail for a domain reason, but it
            // must NOT be the unknown-tool InvalidParams error.
            if let Some(err) = response.error {
                assert_ne!(
                    err.code, -32602,
                    "'{name}' should still dispatch (hidden alias), got unknown-tool error: {}",
                    err.message
                );
            }
        }
    }

    /// v2.2: the 4 tools folded into `memory_status` must still dispatch, and
    /// each `view` of the new tool must resolve.
    #[tokio::test]
    async fn test_memory_status_views_and_aliases() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let calls: Vec<(&str, serde_json::Value)> = vec![
            // Deprecated aliases must still dispatch.
            ("system_status", serde_json::json!({})),
            ("memory_health", serde_json::json!({})),
            ("memory_timeline", serde_json::json!({})),
            ("memory_changelog", serde_json::json!({})),
            // New unified views.
            ("memory_status", serde_json::json!({})), // default view = health
            ("memory_status", serde_json::json!({"view": "retention"})),
            ("memory_status", serde_json::json!({"view": "timeline"})),
            ("memory_status", serde_json::json!({"view": "changelog"})),
            ("memory_status", serde_json::json!({"view": "stats"})),
        ];

        for (name, args) in calls {
            let request = make_request(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": args })),
            );
            let response = server.handle_request(request).await.unwrap();
            assert!(
                response.error.is_none(),
                "'{name}' {args} should resolve, got error: {:?}",
                response.error
            );
        }
    }

    /// v2.2: the 4 tools folded into `graph` must still dispatch, and the
    /// read-only `graph` actions must resolve. (memory_graph is sync — this also
    /// guards the no-`.await` facade branch.)
    #[tokio::test]
    async fn test_graph_actions_and_aliases() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let calls: Vec<(&str, serde_json::Value)> = vec![
            // Deprecated aliases must still dispatch (not unknown-tool).
            ("predict", serde_json::json!({})),
            ("memory_graph", serde_json::json!({})),
            ("composed_graph", serde_json::json!({"action": "recent"})),
            // New unified actions (read-only).
            ("graph", serde_json::json!({"action": "predict"})),
            ("graph", serde_json::json!({"action": "memory_graph"})),
            ("graph", serde_json::json!({"action": "recent"})),
            ("graph", serde_json::json!({"action": "never_composed"})),
        ];

        for (name, args) in calls {
            let request = make_request(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": args })),
            );
            let response = server.handle_request(request).await.unwrap();
            if let Some(err) = response.error {
                assert_ne!(
                    err.code, -32602,
                    "'{name}' {args} should dispatch (not unknown-tool): {}",
                    err.message
                );
            }
        }
    }

    /// A real retrieval must create one durable receipt that the Black Box can
    /// fetch by the caller-supplied run id, and return that same receipt inline.
    #[tokio::test]
    async fn recall_run_produces_fetchable_decision_receipt() {
        let (mut server, _dir) = test_server().await;
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let seeded = server
            .storage
            .ingest(vestige_core::IngestInput {
                content: "The dashboard development server runs on port 5199.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        let run_id = "run_decision_receipt";
        let response = server
            .handle_request(make_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": "recall",
                    "arguments": { "query": "port 5199", "runId": run_id }
                })),
            ))
            .await
            .expect("recall response");

        assert!(response.error.is_none(), "recall should succeed");
        let receipts = server.storage.list_receipts_for_run(run_id, 10).unwrap();
        assert_eq!(receipts.len(), 1, "one retrieval produces one receipt");
        assert!(receipts[0].retrieved.contains(&seeded.id));

        let structured = response
            .result
            .as_ref()
            .and_then(|result| result.get("structuredContent"))
            .expect("structured content");
        assert_eq!(structured["runId"], run_id);
        assert_eq!(
            structured["receiptId"],
            serde_json::json!(receipts[0].receipt_id)
        );
        assert!(structured.get("receipt").is_some());
    }

    /// v2.2: the 7 tools folded into `maintain` must still dispatch, the new
    /// actions must resolve, gc must default to dry_run, and restore must keep
    /// path validation (a nonexistent path errors rather than silently no-op).
    #[tokio::test]
    async fn test_maintain_actions_and_safety() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        // Aliases + safe new actions must dispatch (not unknown-tool).
        let dispatch_ok: Vec<(&str, serde_json::Value)> = vec![
            ("consolidate", serde_json::json!({})),
            ("backup", serde_json::json!({})),
            ("dream", serde_json::json!({})),
            ("maintain", serde_json::json!({"action": "consolidate"})),
            ("maintain", serde_json::json!({"action": "gc"})),
            ("maintain", serde_json::json!({"action": "backup"})),
        ];
        for (name, args) in dispatch_ok {
            let request = make_request(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": args })),
            );
            let response = server.handle_request(request).await.unwrap();
            if let Some(err) = response.error {
                assert_ne!(
                    err.code, -32602,
                    "'{name}' {args} should dispatch: {}",
                    err.message
                );
            }
        }

        // gc via maintain defaults to dry_run=true (no deletion).
        let gc_req = make_request(
            "tools/call",
            Some(serde_json::json!({ "name": "maintain", "arguments": {"action": "gc"} })),
        );
        let gc_resp = server.handle_request(gc_req).await.unwrap();
        let text = gc_resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("\"dryRun\": true") || text.contains("\"dryRun\":true"),
            "maintain action=gc must default to dry_run=true; got: {text}"
        );

        // restore keeps path validation: a missing file must error, not no-op.
        let restore_req = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "maintain",
                "arguments": {"action": "restore", "path": "/nonexistent/vestige-backup-xyz.json"}
            })),
        );
        let restore_resp = server.handle_request(restore_req).await.unwrap();
        // Either a JSON-RPC error or an error envelope is acceptable; a silent
        // success is NOT (that would mean confinement/validation was bypassed).
        let validated = restore_resp.error.is_some()
            || restore_resp
                .result
                .map(|r| {
                    r["content"][0]["text"]
                        .as_str()
                        .map(|t| {
                            t.to_lowercase().contains("not found")
                                || t.to_lowercase().contains("error")
                        })
                        .unwrap_or(false)
                })
                .unwrap_or(false);
        assert!(
            validated,
            "maintain action=restore must validate a missing path"
        );
    }

    /// v2.2 HOT PATH: `recall` defaults to mode='lookup' (search), the folded
    /// names still dispatch, and the reason/contradictions modes resolve.
    #[tokio::test]
    async fn test_recall_modes_and_aliases() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let calls: Vec<(&str, serde_json::Value)> = vec![
            // Deprecated aliases must still dispatch.
            ("search", serde_json::json!({"query": "x"})),
            ("deep_reference", serde_json::json!({"query": "x"})),
            ("cross_reference", serde_json::json!({"query": "x"})),
            ("contradictions", serde_json::json!({})),
            ("semantic_search", serde_json::json!({"query": "x"})),
            // New unified modes.
            ("recall", serde_json::json!({"query": "x"})), // default mode = lookup
            (
                "recall",
                serde_json::json!({"mode": "lookup", "query": "x"}),
            ),
            (
                "recall",
                serde_json::json!({"mode": "reason", "query": "x"}),
            ),
            ("recall", serde_json::json!({"mode": "contradictions"})),
        ];

        for (name, args) in calls {
            let request = make_request(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": args })),
            );
            let response = server.handle_request(request).await.unwrap();
            assert!(
                response.error.is_none(),
                "'{name}' {args} should resolve, got error: {:?}",
                response.error
            );
        }
    }

    /// v2.2: `recall` mode='lookup' (the default) must produce the same result
    /// shape as the former standalone `search` — i.e. the no-mode default is a
    /// faithful pass-through, not a reasoning call.
    #[tokio::test]
    async fn test_recall_lookup_matches_search_shape() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let args = serde_json::json!({ "query": "anything" });
        let via_recall = make_request(
            "tools/call",
            Some(serde_json::json!({ "name": "recall", "arguments": args })),
        );
        let via_search = make_request(
            "tools/call",
            Some(serde_json::json!({ "name": "search", "arguments": args })),
        );
        let r1 = server.handle_request(via_recall).await.unwrap();
        let r2 = server.handle_request(via_search).await.unwrap();
        assert!(r1.error.is_none() && r2.error.is_none());
        // The unified-tool wrapper text (the search payload) must match.
        assert_eq!(
            r1.result.unwrap()["content"][0]["text"],
            r2.result.unwrap()["content"][0]["text"],
            "recall(mode=lookup) must equal search byte-for-byte"
        );
    }

    #[tokio::test]
    async fn test_tools_have_descriptions_and_schemas() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        for tool in tools {
            assert!(tool["name"].is_string(), "Tool should have a name");
            assert!(
                tool["description"].is_string(),
                "Tool should have a description"
            );
            assert!(
                tool["inputSchema"].is_object(),
                "Tool should have an input schema"
            );
        }
    }

    // ========================================================================
    // RESOURCES/LIST TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_resources_list_returns_all_resources() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("resources/list", None);
        let response = server.handle_request(request).await.unwrap();

        let result = response.result.unwrap();
        let resources = result["resources"].as_array().unwrap();

        // Verify expected resources are present
        let resource_uris: Vec<&str> = resources
            .iter()
            .map(|r| r["uri"].as_str().unwrap())
            .collect();

        assert!(resource_uris.contains(&"memory://stats"));
        assert!(resource_uris.contains(&"memory://recent"));
        assert!(resource_uris.contains(&"memory://decaying"));
        assert!(resource_uris.contains(&"memory://due"));
        assert!(resource_uris.contains(&"memory://intentions"));
        assert!(resource_uris.contains(&"codebase://structure"));
        assert!(resource_uris.contains(&"codebase://patterns"));
        assert!(resource_uris.contains(&"codebase://decisions"));
    }

    #[tokio::test]
    async fn test_resources_have_descriptions() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("resources/list", None);
        let response = server.handle_request(request).await.unwrap();

        let result = response.result.unwrap();
        let resources = result["resources"].as_array().unwrap();

        for resource in resources {
            assert!(resource["uri"].is_string(), "Resource should have a URI");
            assert!(resource["name"].is_string(), "Resource should have a name");
            assert!(
                resource["description"].is_string(),
                "Resource should have a description"
            );
        }
    }

    // ========================================================================
    // UNKNOWN METHOD TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_unknown_method_returns_error() {
        let (mut server, _dir) = test_server().await;

        // Initialize first
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("unknown/method", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert_eq!(error.code, -32601); // MethodNotFound
    }

    #[tokio::test]
    async fn test_unknown_tool_returns_error() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "nonexistent_tool",
                "arguments": {}
            })),
        );

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }

    // ========================================================================
    // PING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_ping_returns_empty_object() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("ping", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.result.is_some());
        assert!(response.error.is_none());
        assert_eq!(response.result.unwrap(), serde_json::json!({}));
    }

    // ========================================================================
    // TOOLS/CALL TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_tools_call_missing_params_returns_error() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/call", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602); // InvalidParams
    }

    #[tokio::test]
    async fn test_tool_call_populates_agent_black_box_trace() {
        // Regression (#117): the trace recorder was dead code — no production
        // caller — so agent_traces/receipts/memory_prs never populated. A tool
        // call must now record at least the opening mcp.call event under the
        // supplied runId.
        let (mut server, _dir) = test_server().await;
        server
            .handle_request(make_request("initialize", Some(init_params())))
            .await;

        let run_id = "test-run-blackbox";
        let request = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "search",
                "arguments": { "query": "anything", "runId": run_id }
            })),
        );
        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_none(), "tool call should succeed");

        let events = server.storage.get_trace(run_id).expect("read trace");
        assert!(
            !events.is_empty(),
            "the Black Box must record at least the mcp.call event for this run"
        );
    }

    #[tokio::test]
    async fn test_tools_call_invalid_params_returns_error() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request(
            "tools/call",
            Some(serde_json::json!({
                "invalid": "params"
            })),
        );

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_tools_call_rejects_non_object_arguments() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request(
            "tools/call",
            Some(serde_json::json!({
                "name": "search",
                "arguments": "not-an-object"
            })),
        );

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }

    // ========================================================================
    // Per-tool result-size annotation tests
    // (`_meta["anthropic/maxResultSizeChars"]`, CC v2.1.91+)
    //
    // The annotation lives on the Tool definition in `tools/list`, so CC reads
    // it once when the MCP session opens and applies the override to every
    // invocation of that tool. These tests pin the wire-form so a future
    // refactor of `ToolDescription` cannot silently drop the annotation.
    // ========================================================================

    /// Expected per-tool caps. Returns `Some(cap)` for tools the discipline
    /// annotates, `None` for tools that MUST NOT carry the annotation
    /// (cargo-cult prevention).
    fn expected_max_result_size(name: &str) -> Option<u64> {
        match name {
            // v2.2: search folded into recall (mode='lookup'); annotation moved.
            "recall" => Some(300_000),
            // v2.2: memory_timeline folded into memory_status (view='timeline');
            // the high-payload annotation moved with it.
            "memory_status" => Some(200_000),
            "memory" => Some(100_000),
            "codebase" => Some(100_000),
            // v2.2: dedup action='scan' returns clusters + candidates + policy.
            "dedup" => Some(150_000),
            // v2.2: graph memory_graph layout + bounty_mode pagination.
            "graph" => Some(250_000),
            _ => None,
        }
    }

    #[tokio::test]
    async fn test_high_payload_tools_have_max_result_size_annotation() {
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();
        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        for name in [
            "recall",
            "memory_status",
            "memory",
            "codebase",
            "dedup",
            "graph",
        ] {
            let tool = tools
                .iter()
                .find(|t| t["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("Tool '{}' missing from tools/list", name));

            let expected = expected_max_result_size(name).unwrap();
            let meta = tool.get("_meta").unwrap_or_else(|| {
                panic!("Tool '{}' is missing the `_meta` field on the wire", name)
            });
            let actual = meta
                .get("anthropic/maxResultSizeChars")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| {
                    panic!(
                        "Tool '{}' _meta lacks integer 'anthropic/maxResultSizeChars'",
                        name
                    )
                });
            assert_eq!(
                actual, expected,
                "Tool '{}' cap drift: expected {} got {}",
                name, expected, actual
            );
            assert!(
                actual <= 500_000,
                "Tool '{}' cap {} exceeds Anthropic 500K ceiling",
                name,
                actual
            );
        }
    }

    #[tokio::test]
    async fn test_other_tools_do_not_carry_max_result_size_annotation() {
        // Cargo-cult prevention. Dynamically derived from tools/list so this
        // test is robust to new tools being added: any tool that is NOT in
        // the discipline-prescribed set MUST NOT carry the annotation.
        // Adding the annotation to a small-payload tool dilutes the signal
        // and trains future maintainers that the value is arbitrary.
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();
        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        for tool in tools {
            let name = tool["name"].as_str().unwrap();
            if expected_max_result_size(name).is_some() {
                continue; // covered by the annotated-tools test
            }

            // Either the `_meta` key is absent OR it is an object without the
            // anthropic key — both are acceptable. The forbidden case is the
            // anthropic key present on this tool.
            let has_max_size = tool
                .get("_meta")
                .and_then(|m| m.get("anthropic/maxResultSizeChars"))
                .is_some();
            assert!(
                !has_max_size,
                "Tool '{}' should NOT carry maxResultSizeChars annotation \
                 (not in the discipline-prescribed set: search, memory_timeline, \
                 memory, codebase). If this tool's realistic max-payload now \
                 routinely exceeds 50K, update expected_max_result_size() + the \
                 annotation loop in handle_tools_list together.",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_meta_wire_shape_uses_underscore_meta_field() {
        // Anthropic's MCP spec is explicit: the field on the wire is `_meta`,
        // NOT `meta`. The Rust struct uses `meta: Option<Value>` with
        // `#[serde(rename = "_meta")]` — assert the rename actually fired.
        let (mut server, _dir) = test_server().await;
        let init_request = make_request("initialize", Some(init_params()));
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();
        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        // v2.2: `recall` is the annotated retrieval tool (search folded in).
        let recall_tool = tools
            .iter()
            .find(|t| t["name"].as_str() == Some("recall"))
            .expect("'recall' tool present");

        // Wire-form: `_meta` must exist; `meta` (un-renamed) must NOT exist.
        assert!(
            recall_tool.get("_meta").is_some(),
            "recall tool missing `_meta` key (serde rename to _meta did not apply)"
        );
        assert!(
            recall_tool.get("meta").is_none(),
            "recall tool has un-renamed `meta` key (regression — serde rename broke)"
        );
    }
}
