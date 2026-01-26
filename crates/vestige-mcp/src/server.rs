//! MCP Server Core
//!
//! Handles the main MCP server logic, routing requests to appropriate
//! tool and resource handlers.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::protocol::messages::{
    CallToolRequest, CallToolResult, InitializeRequest, InitializeResult,
    ListResourcesResult, ListToolsResult, ReadResourceRequest, ReadResourceResult,
    ResourceDescription, ServerCapabilities, ServerInfo, ToolDescription,
};
use crate::protocol::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, MCP_VERSION};
use crate::resources;
use crate::tools;
use vestige_core::Storage;

/// MCP Server implementation
pub struct McpServer {
    storage: Arc<Mutex<Storage>>,
    initialized: bool,
}

impl McpServer {
    pub fn new(storage: Arc<Mutex<Storage>>) -> Self {
        Self {
            storage,
            initialized: false,
        }
    }

    /// Handle an incoming JSON-RPC request
    pub async fn handle_request(&mut self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        debug!("Handling request: {}", request.method);

        // Check initialization for non-initialize requests
        if !self.initialized && request.method != "initialize" && request.method != "notifications/initialized" {
            warn!("Rejecting request '{}': server not initialized", request.method);
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
        let _request: InitializeRequest = match params {
            Some(p) => serde_json::from_value(p).map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => InitializeRequest::default(),
        };

        self.initialized = true;
        info!("MCP session initialized");

        let result = InitializeResult {
            protocol_version: MCP_VERSION.to_string(),
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
            instructions: Some(
                "Vestige is your long-term memory system. Use it to remember important information, \
                 recall past knowledge, and maintain context across sessions. The system uses \
                 FSRS-6 spaced repetition to naturally decay memories over time - review important \
                 memories to strengthen them.".to_string()
            ),
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
    }

    /// Handle tools/list request
    async fn handle_tools_list(&self) -> Result<serde_json::Value, JsonRpcError> {
        let tools = vec![
            // ================================================================
            // UNIFIED TOOLS (v1.1+) - Preferred API
            // ================================================================
            ToolDescription {
                name: "search".to_string(),
                description: Some("Unified search tool. Uses hybrid search (keyword + semantic + RRF fusion) internally. Auto-strengthens memories on access (Testing Effect).".to_string()),
                input_schema: tools::search_unified::schema(),
            },
            ToolDescription {
                name: "memory".to_string(),
                description: Some("Unified memory management tool. Actions: 'get' (retrieve full node), 'delete' (remove memory), 'state' (get accessibility state).".to_string()),
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
            // Core memory tools
            // ================================================================
            ToolDescription {
                name: "ingest".to_string(),
                description: Some("Add new knowledge to memory. Use for facts, concepts, decisions, or any information worth remembering.".to_string()),
                input_schema: tools::ingest::schema(),
            },
            ToolDescription {
                name: "smart_ingest".to_string(),
                description: Some("INTELLIGENT memory ingestion with Prediction Error Gating. Automatically decides whether to CREATE new, UPDATE existing, or SUPERSEDE outdated memories based on semantic similarity. Solves the 'bad vs good similar memory' problem.".to_string()),
                input_schema: tools::smart_ingest::schema(),
            },
            ToolDescription {
                name: "recall".to_string(),
                description: Some("(deprecated) Use 'search' instead. Search and retrieve knowledge from memory.".to_string()),
                input_schema: tools::recall::schema(),
            },
            ToolDescription {
                name: "semantic_search".to_string(),
                description: Some("(deprecated) Use 'search' instead. Search memories using semantic similarity.".to_string()),
                input_schema: tools::search::semantic_schema(),
            },
            ToolDescription {
                name: "hybrid_search".to_string(),
                description: Some("(deprecated) Use 'search' instead. Combined keyword + semantic search with RRF fusion.".to_string()),
                input_schema: tools::search::hybrid_schema(),
            },
            ToolDescription {
                name: "get_knowledge".to_string(),
                description: Some("(deprecated) Use 'memory' with action='get' instead. Retrieve a specific memory by ID.".to_string()),
                input_schema: tools::knowledge::get_schema(),
            },
            ToolDescription {
                name: "delete_knowledge".to_string(),
                description: Some("(deprecated) Use 'memory' with action='delete' instead. Delete a memory by ID.".to_string()),
                input_schema: tools::knowledge::delete_schema(),
            },
            ToolDescription {
                name: "mark_reviewed".to_string(),
                description: Some("Mark a memory as reviewed with FSRS rating (1=Again, 2=Hard, 3=Good, 4=Easy). Strengthens retention.".to_string()),
                input_schema: tools::review::schema(),
            },
            // Stats and maintenance
            ToolDescription {
                name: "get_stats".to_string(),
                description: Some("Get memory system statistics including total nodes, retention, and embedding status.".to_string()),
                input_schema: tools::stats::stats_schema(),
            },
            ToolDescription {
                name: "health_check".to_string(),
                description: Some("Check health status of the memory system.".to_string()),
                input_schema: tools::stats::health_schema(),
            },
            ToolDescription {
                name: "run_consolidation".to_string(),
                description: Some("Run memory consolidation cycle. Applies decay, promotes important memories, generates embeddings.".to_string()),
                input_schema: tools::consolidate::schema(),
            },
            // Codebase tools (deprecated - use unified 'codebase' tool)
            ToolDescription {
                name: "remember_pattern".to_string(),
                description: Some("(deprecated) Use 'codebase' with action='remember_pattern' instead. Remember a code pattern or convention.".to_string()),
                input_schema: tools::codebase::pattern_schema(),
            },
            ToolDescription {
                name: "remember_decision".to_string(),
                description: Some("(deprecated) Use 'codebase' with action='remember_decision' instead. Remember an architectural decision.".to_string()),
                input_schema: tools::codebase::decision_schema(),
            },
            ToolDescription {
                name: "get_codebase_context".to_string(),
                description: Some("(deprecated) Use 'codebase' with action='get_context' instead. Get remembered patterns and decisions.".to_string()),
                input_schema: tools::codebase::context_schema(),
            },
            // Prospective memory (intentions) - deprecated, use unified 'intention' tool
            ToolDescription {
                name: "set_intention".to_string(),
                description: Some("(deprecated) Use 'intention' with action='set' instead. Remember to do something in the future.".to_string()),
                input_schema: tools::intentions::set_schema(),
            },
            ToolDescription {
                name: "check_intentions".to_string(),
                description: Some("(deprecated) Use 'intention' with action='check' instead. Check if any intentions should be triggered.".to_string()),
                input_schema: tools::intentions::check_schema(),
            },
            ToolDescription {
                name: "complete_intention".to_string(),
                description: Some("(deprecated) Use 'intention' with action='update', status='complete' instead. Mark an intention as complete.".to_string()),
                input_schema: tools::intentions::complete_schema(),
            },
            ToolDescription {
                name: "snooze_intention".to_string(),
                description: Some("(deprecated) Use 'intention' with action='update', status='snooze' instead. Snooze an intention.".to_string()),
                input_schema: tools::intentions::snooze_schema(),
            },
            ToolDescription {
                name: "list_intentions".to_string(),
                description: Some("(deprecated) Use 'intention' with action='list' instead. List all intentions.".to_string()),
                input_schema: tools::intentions::list_schema(),
            },
            // Neuroscience tools
            ToolDescription {
                name: "get_memory_state".to_string(),
                description: Some("(deprecated) Use 'memory' with action='state' instead. Get the cognitive state of a memory.".to_string()),
                input_schema: tools::memory_states::get_schema(),
            },
            ToolDescription {
                name: "list_by_state".to_string(),
                description: Some("List memories grouped by cognitive state.".to_string()),
                input_schema: tools::memory_states::list_schema(),
            },
            ToolDescription {
                name: "state_stats".to_string(),
                description: Some("Get statistics about memory state distribution.".to_string()),
                input_schema: tools::memory_states::stats_schema(),
            },
            ToolDescription {
                name: "trigger_importance".to_string(),
                description: Some("Trigger retroactive importance to strengthen recent memories. Based on Synaptic Tagging & Capture (Frey & Morris 1997).".to_string()),
                input_schema: tools::tagging::trigger_schema(),
            },
            ToolDescription {
                name: "find_tagged".to_string(),
                description: Some("Find memories with high retention (tagged/strengthened memories).".to_string()),
                input_schema: tools::tagging::find_schema(),
            },
            ToolDescription {
                name: "tagging_stats".to_string(),
                description: Some("Get synaptic tagging and retention statistics.".to_string()),
                input_schema: tools::tagging::stats_schema(),
            },
            ToolDescription {
                name: "match_context".to_string(),
                description: Some("Search memories with context-dependent retrieval. Based on Tulving's Encoding Specificity Principle (1973).".to_string()),
                input_schema: tools::context::schema(),
            },
            // Feedback / preference learning
            ToolDescription {
                name: "promote_memory".to_string(),
                description: Some("Promote a memory (thumbs up). Use when a memory led to a good outcome. Increases retrieval strength so it surfaces more often.".to_string()),
                input_schema: tools::feedback::promote_schema(),
            },
            ToolDescription {
                name: "demote_memory".to_string(),
                description: Some("Demote a memory (thumbs down). Use when a memory led to a bad outcome or was wrong. Decreases retrieval strength so better alternatives surface. Does NOT delete.".to_string()),
                input_schema: tools::feedback::demote_schema(),
            },
            ToolDescription {
                name: "request_feedback".to_string(),
                description: Some("Ask the user if a memory was helpful. Use after applying advice from a memory. Returns options for the user to choose: helpful (promote), wrong (demote), or skip.".to_string()),
                input_schema: tools::feedback::request_feedback_schema(),
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
            Some(p) => serde_json::from_value(p).map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => return Err(JsonRpcError::invalid_params("Missing tool call parameters")),
        };

        let result = match request.name.as_str() {
            // ================================================================
            // UNIFIED TOOLS (v1.1+) - Preferred API
            // ================================================================
            "search" => tools::search_unified::execute(&self.storage, request.arguments).await,
            "memory" => tools::memory_unified::execute(&self.storage, request.arguments).await,
            "codebase" => tools::codebase_unified::execute(&self.storage, request.arguments).await,
            "intention" => tools::intention_unified::execute(&self.storage, request.arguments).await,

            // ================================================================
            // Core memory tools
            // ================================================================
            "ingest" => tools::ingest::execute(&self.storage, request.arguments).await,
            "smart_ingest" => tools::smart_ingest::execute(&self.storage, request.arguments).await,
            "mark_reviewed" => tools::review::execute(&self.storage, request.arguments).await,

            // ================================================================
            // DEPRECATED: Search tools - redirect to unified 'search'
            // ================================================================
            "recall" | "semantic_search" | "hybrid_search" => {
                warn!("Tool '{}' is deprecated. Use 'search' instead.", request.name);
                tools::search_unified::execute(&self.storage, request.arguments).await
            }

            // ================================================================
            // DEPRECATED: Memory tools - redirect to unified 'memory'
            // ================================================================
            "get_knowledge" => {
                warn!("Tool 'get_knowledge' is deprecated. Use 'memory' with action='get' instead.");
                // Transform arguments to unified format
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
                tools::memory_unified::execute(&self.storage, unified_args).await
            }
            "delete_knowledge" => {
                warn!("Tool 'delete_knowledge' is deprecated. Use 'memory' with action='delete' instead.");
                // Transform arguments to unified format
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
                tools::memory_unified::execute(&self.storage, unified_args).await
            }
            "get_memory_state" => {
                warn!("Tool 'get_memory_state' is deprecated. Use 'memory' with action='state' instead.");
                // Transform arguments to unified format
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("memory_id").cloned().unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "state",
                            "id": id
                        }))
                    }
                    None => None,
                };
                tools::memory_unified::execute(&self.storage, unified_args).await
            }

            // ================================================================
            // DEPRECATED: Codebase tools - redirect to unified 'codebase'
            // ================================================================
            "remember_pattern" => {
                warn!("Tool 'remember_pattern' is deprecated. Use 'codebase' with action='remember_pattern' instead.");
                // Transform arguments to unified format
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
                tools::codebase_unified::execute(&self.storage, unified_args).await
            }
            "remember_decision" => {
                warn!("Tool 'remember_decision' is deprecated. Use 'codebase' with action='remember_decision' instead.");
                // Transform arguments to unified format
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("remember_decision"));
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "remember_decision"})),
                };
                tools::codebase_unified::execute(&self.storage, unified_args).await
            }
            "get_codebase_context" => {
                warn!("Tool 'get_codebase_context' is deprecated. Use 'codebase' with action='get_context' instead.");
                // Transform arguments to unified format
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
                tools::codebase_unified::execute(&self.storage, unified_args).await
            }

            // ================================================================
            // DEPRECATED: Intention tools - redirect to unified 'intention'
            // ================================================================
            "set_intention" => {
                warn!("Tool 'set_intention' is deprecated. Use 'intention' with action='set' instead.");
                // Transform arguments to unified format
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
                tools::intention_unified::execute(&self.storage, unified_args).await
            }
            "check_intentions" => {
                warn!("Tool 'check_intentions' is deprecated. Use 'intention' with action='check' instead.");
                // Transform arguments to unified format
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
                tools::intention_unified::execute(&self.storage, unified_args).await
            }
            "complete_intention" => {
                warn!("Tool 'complete_intention' is deprecated. Use 'intention' with action='update', status='complete' instead.");
                // Transform arguments to unified format
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("intentionId").cloned().unwrap_or(serde_json::Value::Null);
                        Some(serde_json::json!({
                            "action": "update",
                            "id": id,
                            "status": "complete"
                        }))
                    }
                    None => None,
                };
                tools::intention_unified::execute(&self.storage, unified_args).await
            }
            "snooze_intention" => {
                warn!("Tool 'snooze_intention' is deprecated. Use 'intention' with action='update', status='snooze' instead.");
                // Transform arguments to unified format
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let id = args.get("intentionId").cloned().unwrap_or(serde_json::Value::Null);
                        let minutes = args.get("minutes").cloned().unwrap_or(serde_json::json!(30));
                        Some(serde_json::json!({
                            "action": "update",
                            "id": id,
                            "status": "snooze",
                            "snooze_minutes": minutes
                        }))
                    }
                    None => None,
                };
                tools::intention_unified::execute(&self.storage, unified_args).await
            }
            "list_intentions" => {
                warn!("Tool 'list_intentions' is deprecated. Use 'intention' with action='list' instead.");
                // Transform arguments to unified format
                let unified_args = match request.arguments {
                    Some(ref args) => {
                        let mut new_args = args.clone();
                        if let Some(obj) = new_args.as_object_mut() {
                            obj.insert("action".to_string(), serde_json::json!("list"));
                            // Rename 'status' to 'filter_status' if present
                            if let Some(status) = obj.remove("status") {
                                obj.insert("filter_status".to_string(), status);
                            }
                        }
                        Some(new_args)
                    }
                    None => Some(serde_json::json!({"action": "list"})),
                };
                tools::intention_unified::execute(&self.storage, unified_args).await
            }

            // ================================================================
            // Stats and maintenance (not deprecated)
            // ================================================================
            "get_stats" => tools::stats::execute_stats(&self.storage).await,
            "health_check" => tools::stats::execute_health(&self.storage).await,
            "run_consolidation" => tools::consolidate::execute(&self.storage).await,

            // ================================================================
            // Neuroscience tools (not deprecated, except get_memory_state above)
            // ================================================================
            "list_by_state" => tools::memory_states::execute_list(&self.storage, request.arguments).await,
            "state_stats" => tools::memory_states::execute_stats(&self.storage).await,
            "trigger_importance" => tools::tagging::execute_trigger(&self.storage, request.arguments).await,
            "find_tagged" => tools::tagging::execute_find(&self.storage, request.arguments).await,
            "tagging_stats" => tools::tagging::execute_stats(&self.storage).await,
            "match_context" => tools::context::execute(&self.storage, request.arguments).await,

            // ================================================================
            // Feedback / preference learning (not deprecated)
            // ================================================================
            "promote_memory" => tools::feedback::execute_promote(&self.storage, request.arguments).await,
            "demote_memory" => tools::feedback::execute_demote(&self.storage, request.arguments).await,
            "request_feedback" => tools::feedback::execute_request_feedback(&self.storage, request.arguments).await,

            name => {
                return Err(JsonRpcError::method_not_found_with_message(&format!(
                    "Unknown tool: {}",
                    name
                )));
            }
        };

        match result {
            Ok(content) => {
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: serde_json::to_string_pretty(&content).unwrap_or_else(|_| content.to_string()),
                    }],
                    is_error: Some(false),
                };
                serde_json::to_value(call_result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
            Err(e) => {
                let call_result = CallToolResult {
                    content: vec![crate::protocol::messages::ToolResultContent {
                        content_type: "text".to_string(),
                        text: serde_json::json!({ "error": e }).to_string(),
                    }],
                    is_error: Some(true),
                };
                serde_json::to_value(call_result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
        }
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
            // Prospective memory resources
            ResourceDescription {
                uri: "memory://intentions".to_string(),
                name: "Active Intentions".to_string(),
                description: Some("Future intentions (prospective memory) waiting to be triggered".to_string()),
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
            Some(p) => serde_json::from_value(p).map_err(|e| JsonRpcError::invalid_params(&e.to_string()))?,
            None => return Err(JsonRpcError::invalid_params("Missing resource URI")),
        };

        let uri = &request.uri;
        let content = if uri.starts_with("memory://") {
            resources::memory::read(&self.storage, uri).await
        } else if uri.starts_with("codebase://") {
            resources::codebase::read(&self.storage, uri).await
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
                serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(&e.to_string()))
            }
            Err(e) => Err(JsonRpcError::internal_error(&e)),
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
    async fn test_storage() -> (Arc<Mutex<Storage>>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(Mutex::new(storage)), dir)
    }

    /// Create a test server with temporary storage
    async fn test_server() -> (McpServer, TempDir) {
        let (storage, dir) = test_storage().await;
        let server = McpServer::new(storage);
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

        let request = make_request("initialize", Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })));

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
        let request = make_request("initialize", None);

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

        // Verify expected tools are present
        let tool_names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();

        // Unified tools (v1.1+)
        assert!(tool_names.contains(&"search"));
        assert!(tool_names.contains(&"memory"));
        assert!(tool_names.contains(&"codebase"));
        assert!(tool_names.contains(&"intention"));

        // Core tools
        assert!(tool_names.contains(&"ingest"));
        assert!(tool_names.contains(&"recall"));
        assert!(tool_names.contains(&"semantic_search"));
        assert!(tool_names.contains(&"hybrid_search"));
        assert!(tool_names.contains(&"get_knowledge"));
        assert!(tool_names.contains(&"delete_knowledge"));
        assert!(tool_names.contains(&"mark_reviewed"));
        assert!(tool_names.contains(&"get_stats"));
        assert!(tool_names.contains(&"health_check"));
        assert!(tool_names.contains(&"run_consolidation"));
        assert!(tool_names.contains(&"set_intention"));
        assert!(tool_names.contains(&"check_intentions"));
        assert!(tool_names.contains(&"complete_intention"));
        assert!(tool_names.contains(&"snooze_intention"));
        assert!(tool_names.contains(&"list_intentions"));
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
            assert!(tool["description"].is_string(), "Tool should have a description");
            assert!(tool["inputSchema"].is_object(), "Tool should have an input schema");
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
            assert!(resource["description"].is_string(), "Resource should have a description");
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

        let request = make_request("tools/call", Some(serde_json::json!({
            "name": "nonexistent_tool",
            "arguments": {}
        })));

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

        let request = make_request("tools/call", Some(serde_json::json!({
            "invalid": "params"
        })));

        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }
}
