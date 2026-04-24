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
use vestige_core::Storage;

/// Build the MCP `instructions` string injected into every connecting client's
/// system prompt.
///
/// Default ("minimal", 3 sentences) is safe for any user: competitive coders,
/// hobbyists saving recipes, Rails devs saving bug fixes, enterprise deployments.
/// It earns its per-session token cost by telling the client *how* to use
/// Vestige without imposing one maintainer's workflow on strangers.
///
/// The "full" variant is the composition mandate that enforces the
/// Composing / Never-composed / Recommendation response shape, names the
/// AIMO3 36/50 case study as the origin, and includes the "Vestige is
/// blocking this:" refusal phrase. It is load-bearing for Sam's own
/// decision-adjacent work but would misfire on trivial retrievals for a
/// general audience, so it is opt-in via `VESTIGE_SYSTEM_PROMPT_MODE=full`.
///
/// Anything other than `full` falls back to minimal.
fn build_instructions() -> String {
    let mode = std::env::var("VESTIGE_SYSTEM_PROMPT_MODE").unwrap_or_default();
    if mode.eq_ignore_ascii_case("full") {
        "Vestige is your long-term cognitive memory AND reasoning engine, not a RAG database. \
         Every retrieval MUST be composed into a recommendation, never summarized.\
         \n\nCOMPOSITION MANDATE: When you receive memories from search, deep_reference, \
         cross_reference, or explore_connections, your response MUST follow this shape. \
         (a) Composing: [memory IDs], followed by your composition logic (your chain-of-thought \
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

/// MCP Server implementation
pub struct McpServer {
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    initialized: bool,
    /// Tool call counter for inline consolidation trigger (every 100 calls)
    tool_call_count: AtomicU64,
    /// Optional event broadcast channel for dashboard real-time updates.
    event_tx: Option<broadcast::Sender<VestigeEvent>>,
}

impl McpServer {
    #[allow(dead_code)]
    pub fn new(storage: Arc<Storage>, cognitive: Arc<Mutex<CognitiveEngine>>) -> Self {
        Self {
            storage,
            cognitive,
            initialized: false,
            tool_call_count: AtomicU64::new(0),
            event_tx: None,
        }
    }

    /// Create an MCP server that broadcasts events to the dashboard.
    pub fn new_with_events(
        storage: Arc<Storage>,
        cognitive: Arc<Mutex<CognitiveEngine>>,
        event_tx: broadcast::Sender<VestigeEvent>,
    ) -> Self {
        Self {
            storage,
            cognitive,
            initialized: false,
            tool_call_count: AtomicU64::new(0),
            event_tx: Some(event_tx),
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
            "notifications/initialized" => {
                // Notification, no response needed
                return None;
            }
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
            None => InitializeRequest::default(),
        };

        // Version negotiation: use client's version if older than server's
        // Claude Desktop rejects servers with newer protocol versions
        let negotiated_version = if request.protocol_version.as_str() < MCP_VERSION {
            info!(
                "Client requested older protocol version {}, using it",
                request.protocol_version
            );
            request.protocol_version.clone()
        } else {
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
        // v2.0.5+: 24 tools (verified by the `tools.len() == 24` assertion in the
        // handle_tools_list test below — the `suppress` tool landed in v2.0.5).
        // Deprecated tools still work via redirects in handle_tools_call.
        let tools = vec![
            // ================================================================
            // UNIFIED TOOLS (v1.1+)
            // ================================================================
            ToolDescription {
                name: "search".to_string(),
                description: Some("Unified search tool. Uses hybrid search (keyword + semantic + convex combination fusion) internally. Auto-strengthens memories on access (Testing Effect).".to_string()),
                input_schema: tools::search_unified::schema(),
            },
            ToolDescription {
                name: "memory".to_string(),
                description: Some("Unified memory management tool. Actions: 'get' (retrieve full node), 'delete' (remove memory), 'state' (get accessibility state), 'promote' (thumbs up — increases retrieval strength), 'demote' (thumbs down — decreases retrieval strength, does NOT delete), 'edit' (update content in-place, preserves FSRS state).".to_string()),
                input_schema: tools::memory_unified::schema(),
            },
            ToolDescription {
                name: "codebase".to_string(),
                description: Some("Unified codebase tool. Actions: 'remember_pattern' (store code pattern), 'remember_decision' (store architectural decision), 'get_context' (retrieve patterns and decisions).".to_string()),
                input_schema: tools::codebase_unified::schema(),
            },
            ToolDescription {
                name: "intention".to_string(),
                description: Some("Unified intention management tool. Actions: 'set' (create), 'check' (find triggered), 'update' (complete/snooze/cancel), 'list' (show intentions).".to_string()),
                input_schema: tools::intention_unified::schema(),
            },
            // ================================================================
            // CORE MEMORY (v1.7: smart_ingest absorbs ingest + checkpoint)
            // ================================================================
            ToolDescription {
                name: "smart_ingest".to_string(),
                description: Some("INTELLIGENT memory ingestion with Prediction Error Gating. Single mode: provide 'content' to auto-decide CREATE/UPDATE/SUPERSEDE. Batch mode: provide 'items' array (max 20) for session-end saves — each item runs the full cognitive pipeline (importance scoring, intent detection, synaptic tagging).".to_string()),
                input_schema: tools::smart_ingest::schema(),
            },
            // ================================================================
            // TEMPORAL TOOLS (v1.2+)
            // ================================================================
            ToolDescription {
                name: "memory_timeline".to_string(),
                description: Some("Browse memories chronologically. Returns memories in a time range, grouped by day. Defaults to last 7 days.".to_string()),
                input_schema: tools::timeline::schema(),
            },
            ToolDescription {
                name: "memory_changelog".to_string(),
                description: Some("View audit trail of memory changes. Per-memory: state transitions. System-wide: consolidations + recent state changes.".to_string()),
                input_schema: tools::changelog::schema(),
            },
            // ================================================================
            // MAINTENANCE TOOLS (v1.7: system_status replaces health_check + stats)
            // ================================================================
            ToolDescription {
                name: "system_status".to_string(),
                description: Some("Combined system health and statistics. Returns status (healthy/degraded/critical/empty), full stats, FSRS preview, cognitive module health, state distribution, warnings, and recommendations.".to_string()),
                input_schema: tools::maintenance::system_status_schema(),
            },
            ToolDescription {
                name: "consolidate".to_string(),
                description: Some("Run FSRS-6 memory consolidation cycle. Applies decay, generates embeddings, and performs maintenance. Use when memories seem stale.".to_string()),
                input_schema: tools::maintenance::consolidate_schema(),
            },
            ToolDescription {
                name: "backup".to_string(),
                description: Some("Create a SQLite database backup. Returns the backup file path.".to_string()),
                input_schema: tools::maintenance::backup_schema(),
            },
            ToolDescription {
                name: "export".to_string(),
                description: Some("Export memories as JSON or JSONL. Supports tag and date filters.".to_string()),
                input_schema: tools::maintenance::export_schema(),
            },
            ToolDescription {
                name: "gc".to_string(),
                description: Some("Garbage collect stale memories below retention threshold. Defaults to dry_run=true for safety.".to_string()),
                input_schema: tools::maintenance::gc_schema(),
            },
            // ================================================================
            // AUTO-SAVE & DEDUP TOOLS (v1.3+)
            // ================================================================
            ToolDescription {
                name: "importance_score".to_string(),
                description: Some("Score content importance using 4-channel neuroscience model (novelty/arousal/reward/attention). Returns composite score, channel breakdown, encoding boost, and explanations.".to_string()),
                input_schema: tools::importance::schema(),
            },
            ToolDescription {
                name: "find_duplicates".to_string(),
                description: Some("Find duplicate and near-duplicate memory clusters using cosine similarity on embeddings. Returns clusters with suggested actions (merge/review). Use to clean up redundant memories.".to_string()),
                input_schema: tools::dedup::schema(),
            },
            // ================================================================
            // COGNITIVE TOOLS (v1.5+)
            // ================================================================
            ToolDescription {
                name: "dream".to_string(),
                description: Some("Trigger memory dreaming — replays recent memories to discover hidden connections, synthesize insights, and strengthen important patterns. Returns insights, connections, and dream stats.".to_string()),
                input_schema: tools::dream::schema(),
            },
            ToolDescription {
                name: "explore_connections".to_string(),
                description: Some("Graph exploration tool for memory connections. Actions: 'chain' (build reasoning path between memories), 'associations' (find related memories via spreading activation + hippocampal index), 'bridges' (find connecting memories between two nodes).".to_string()),
                input_schema: tools::explore::schema(),
            },
            ToolDescription {
                name: "predict".to_string(),
                description: Some("Proactive memory prediction — predicts what memories you'll need next based on context, recent activity, and learned patterns. Returns predictions, suggestions, and speculative retrievals.".to_string()),
                input_schema: tools::predict::schema(),
            },
            // ================================================================
            // RESTORE TOOL (v1.5+)
            // ================================================================
            ToolDescription {
                name: "restore".to_string(),
                description: Some("Restore memories from a JSON backup file. Supports MCP wrapper format, RecallResult format, and direct memory array format.".to_string()),
                input_schema: tools::restore::schema(),
            },
            // ================================================================
            // CONTEXT PACKETS (v1.8+)
            // ================================================================
            ToolDescription {
                name: "session_context".to_string(),
                description: Some("One-call session initialization. Combines search, intentions, status, predictions, and codebase context into a single token-budgeted response. Replaces 5 separate calls at session start.".to_string()),
                input_schema: tools::session_context::schema(),
            },
            // ================================================================
            // AUTONOMIC TOOLS (v1.9+)
            // ================================================================
            ToolDescription {
                name: "memory_health".to_string(),
                description: Some("Retention dashboard. Returns avg retention, retention distribution (buckets: 0-20%, 20-40%, etc.), trend (improving/declining/stable), and recommendation. Lightweight alternative to full system_status focused on memory quality.".to_string()),
                input_schema: tools::health::schema(),
            },
            ToolDescription {
                name: "memory_graph".to_string(),
                description: Some("Subgraph export for visualization. Input: center_id or query, depth (1-3), max_nodes. Returns nodes with force-directed layout positions and edges with weights. Powers memory graph visualization.".to_string()),
                input_schema: tools::graph::schema(),
            },
            // ================================================================
            // DEEP REFERENCE (v2.0.4+) — replaces cross_reference
            // ================================================================
            ToolDescription {
                name: "deep_reference".to_string(),
                description: Some("Deep cognitive reasoning across memories. Combines FSRS-6 trust scoring, spreading activation, temporal supersession, dream insights, and contradiction analysis to build a complete understanding of a topic. Returns trust-scored evidence, fact evolution timeline, and a recommended answer. Use this when accuracy matters.".to_string()),
                input_schema: tools::cross_reference::schema(),
            },
            ToolDescription {
                name: "cross_reference".to_string(),
                description: Some("Alias for deep_reference. Connect the dots across memories with cognitive reasoning.".to_string()),
                input_schema: tools::cross_reference::schema(),
            },
            // ================================================================
            // ACTIVE FORGETTING (v2.0.5) — top-down suppression
            // Anderson et al. 2025 Nat Rev Neurosci + Davis Rac1
            // ================================================================
            ToolDescription {
                name: "suppress".to_string(),
                description: Some("Actively suppress a memory via top-down inhibitory control (Anderson 2025 SIF + Davis Rac1). Distinct from delete: the memory persists but is inhibited from retrieval and actively decays. Each call compounds. A background Rac1 worker cascades decay to co-activated neighbors. Reversible within 24 hours via reverse=true.".to_string()),
                input_schema: tools::suppress::schema(),
            },
        ];

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

        // Record activity on every tool call (non-blocking)
        if let Ok(mut cog) = self.cognitive.try_lock() {
            cog.activity_tracker.record_activity();
            cog.consolidation_scheduler.record_activity();
        }

        // Save args for event emission (tool dispatch consumes request.arguments)
        let saved_args = if self.event_tx.is_some() {
            request.arguments.clone()
        } else {
            None
        };

        let result = match request.name.as_str() {
            // ================================================================
            // UNIFIED TOOLS (v1.1+) - Preferred API
            // ================================================================
            "search" => {
                tools::search_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            "memory" => {
                tools::memory_unified::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }
            "codebase" => {
                tools::codebase_unified::execute(&self.storage, &self.cognitive, request.arguments)
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
            // SYSTEM STATUS (v1.7: replaces health_check + stats)
            // ================================================================
            "system_status" => {
                tools::maintenance::execute_system_status(
                    &self.storage,
                    &self.cognitive,
                    request.arguments,
                )
                .await
            }

            "mark_reviewed" => tools::review::execute(&self.storage, request.arguments).await,

            // ================================================================
            // DEPRECATED: Search tools - redirect to unified 'search'
            // ================================================================
            "recall" | "semantic_search" | "hybrid_search" => {
                warn!(
                    "Tool '{}' is deprecated. Use 'search' instead.",
                    request.name
                );
                tools::search_unified::execute(&self.storage, &self.cognitive, request.arguments)
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
                    "Tool 'delete_knowledge' is deprecated. Use 'memory' with action='delete' instead."
                );
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("id").cloned().unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "delete",
                            "id": id
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
                tools::codebase_unified::execute(&self.storage, &self.cognitive, unified_args).await
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
                tools::codebase_unified::execute(&self.storage, &self.cognitive, unified_args).await
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
                tools::codebase_unified::execute(&self.storage, &self.cognitive, unified_args).await
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
            // TEMPORAL TOOLS (v1.2+)
            // ================================================================
            "memory_timeline" => tools::timeline::execute(&self.storage, request.arguments).await,
            "memory_changelog" => tools::changelog::execute(&self.storage, request.arguments).await,

            // ================================================================
            // MAINTENANCE TOOLS (v1.2+, non-deprecated)
            // ================================================================
            "consolidate" => {
                self.emit(VestigeEvent::ConsolidationStarted {
                    timestamp: chrono::Utc::now(),
                });
                tools::maintenance::execute_consolidate(&self.storage, request.arguments).await
            }
            "backup" => tools::maintenance::execute_backup(&self.storage, request.arguments).await,
            "export" => tools::maintenance::execute_export(&self.storage, request.arguments).await,
            "gc" => tools::maintenance::execute_gc(&self.storage, request.arguments).await,

            // ================================================================
            // AUTO-SAVE & DEDUP TOOLS (v1.3+)
            // ================================================================
            "importance_score" => {
                tools::importance::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            "find_duplicates" => tools::dedup::execute(&self.storage, request.arguments).await,

            // ================================================================
            // COGNITIVE TOOLS (v1.5+)
            // ================================================================
            "dream" => {
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
            "explore_connections" => {
                tools::explore::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            "predict" => {
                tools::predict::execute(&self.storage, &self.cognitive, request.arguments).await
            }
            "restore" => tools::restore::execute(&self.storage, request.arguments).await,

            // ================================================================
            // CONTEXT PACKETS (v1.8+)
            // ================================================================
            "session_context" => {
                tools::session_context::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // AUTONOMIC TOOLS (v1.9+)
            // ================================================================
            "memory_health" => tools::health::execute(&self.storage, request.arguments).await,
            "memory_graph" => tools::graph::execute(&self.storage, request.arguments).await,
            "deep_reference" | "cross_reference" => {
                tools::cross_reference::execute(&self.storage, &self.cognitive, request.arguments)
                    .await
            }

            // ================================================================
            // ACTIVE FORGETTING (v2.0.5) — top-down suppression
            // ================================================================
            "suppress" => tools::suppress::execute(&self.storage, request.arguments).await,

            name => {
                return Err(JsonRpcError::method_not_found_with_message(&format!(
                    "Unknown tool: {}",
                    name
                )));
            }
        };

        // ================================================================
        // DASHBOARD EVENT EMISSION (v2.0)
        // Emit real-time events to WebSocket clients after successful tool calls.
        // ================================================================
        if let Ok(ref content) = result {
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
                    is_error: Some(false),
                };
                serde_json::to_value(call_result)
                    .map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
            Err(e) => {
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: serde_json::json!({ "error": e }).to_string(),
                    }],
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
            Err(e) => Err(JsonRpcError::internal_error(&e)),
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
                    "delete" => {
                        self.emit(VestigeEvent::MemoryDeleted { id, timestamp: now });
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
    async fn test_initialize_with_default_params() {
        let (mut server, _dir) = test_server().await;
        let request = make_request("initialize", None);

        let response = server.handle_request(request).await.unwrap();
        assert!(response.result.is_some());
        assert!(response.error.is_none());
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
        let init_request = make_request("initialize", None);
        server.handle_request(init_request).await;

        // Send initialized notification
        let notification = make_request("notifications/initialized", None);
        let response = server.handle_request(notification).await;

        // Notifications should return None
        assert!(response.is_none());
    }

    // ========================================================================
    // TOOLS/LIST TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_tools_list_returns_all_tools() {
        let (mut server, _dir) = test_server().await;

        // Initialize first
        let init_request = make_request("initialize", None);
        server.handle_request(init_request).await;

        let request = make_request("tools/list", None);
        let response = server.handle_request(request).await.unwrap();

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();

        // v2.0.5: 24 tools (4 unified + 1 core + 2 temporal + 5 maintenance + 2 auto-save + 3 cognitive + 1 restore + 1 session_context + 2 autonomic + 1 deep_reference + 1 cross_reference alias + 1 suppress)
        assert_eq!(tools.len(), 24, "Expected exactly 24 tools in v2.0.5+");

        let tool_names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();

        // Unified tools
        assert!(tool_names.contains(&"search"));
        assert!(tool_names.contains(&"memory"));
        assert!(tool_names.contains(&"codebase"));
        assert!(tool_names.contains(&"intention"));

        // Core memory (smart_ingest absorbs ingest + checkpoint in v1.7)
        assert!(tool_names.contains(&"smart_ingest"));
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

        // Temporal tools (v1.2)
        assert!(tool_names.contains(&"memory_timeline"));
        assert!(tool_names.contains(&"memory_changelog"));

        // Maintenance tools (v1.7: system_status replaces health_check + stats)
        assert!(tool_names.contains(&"system_status"));
        assert!(
            !tool_names.contains(&"health_check"),
            "health_check should be removed in v1.7"
        );
        assert!(
            !tool_names.contains(&"stats"),
            "stats should be removed in v1.7"
        );
        assert!(tool_names.contains(&"consolidate"));
        assert!(tool_names.contains(&"backup"));
        assert!(tool_names.contains(&"export"));
        assert!(tool_names.contains(&"gc"));

        // Auto-save & dedup tools (v1.3)
        assert!(tool_names.contains(&"importance_score"));
        assert!(tool_names.contains(&"find_duplicates"));

        // Cognitive tools (v1.5)
        assert!(tool_names.contains(&"dream"));
        assert!(tool_names.contains(&"explore_connections"));
        assert!(tool_names.contains(&"predict"));
        assert!(tool_names.contains(&"restore"));

        // Context packets (v1.8)
        assert!(tool_names.contains(&"session_context"));

        // Autonomic tools (v1.9)
        assert!(tool_names.contains(&"memory_health"));
        assert!(tool_names.contains(&"memory_graph"));

        // Deep reference + cross_reference alias (v2.0.4)
        assert!(tool_names.contains(&"deep_reference"));
        assert!(tool_names.contains(&"cross_reference"));

        // Active forgetting (v2.0.5) — Anderson 2025 + Davis Rac1
        assert!(tool_names.contains(&"suppress"));
    }

    #[tokio::test]
    async fn test_tools_have_descriptions_and_schemas() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", None);
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

        let init_request = make_request("initialize", None);
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

        let init_request = make_request("initialize", None);
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
        let init_request = make_request("initialize", None);
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

        let init_request = make_request("initialize", None);
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
        assert_eq!(response.error.unwrap().code, -32601);
    }

    // ========================================================================
    // PING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_ping_returns_empty_object() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", None);
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

        let init_request = make_request("initialize", None);
        server.handle_request(init_request).await;

        let request = make_request("tools/call", None);
        let response = server.handle_request(request).await.unwrap();

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602); // InvalidParams
    }

    #[tokio::test]
    async fn test_tools_call_invalid_params_returns_error() {
        let (mut server, _dir) = test_server().await;

        let init_request = make_request("initialize", None);
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
}
