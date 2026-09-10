//! MCP Protocol Messages
//!
//! Request and response types for MCP methods.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ============================================================================
// INITIALIZE
// ============================================================================

/// Initialize request from client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

impl Default for InitializeRequest {
    fn default() -> Self {
        Self {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "unknown".to_string(),
                version: "0.0.0".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Initialize response to client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub server_info: ServerInfo,
    pub capabilities: ServerCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<HashMap<String, Value>>,
    /// Declared so the server may send `notifications/message` (the MCP
    /// logging channel) for out-of-band events such as a first-run model
    /// download. Empty object per the spec.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<HashMap<String, Value>>,
}

// ============================================================================
// TOOLS
// ============================================================================

/// Behaviour hints for a tool, from the MCP `ToolAnnotations` schema.
///
/// Every hint is serialised explicitly, so a client that reads the list
/// never has to fall back on the spec defaults (which assume the worst:
/// writable, destructive, open-world). The hints describe the tool's effect
/// on the user's memory store, which is the environment an agent cares
/// about. `recall`, for example, records receipts and access statistics
/// as it runs, but it never changes what a memory says, so it is read-only
/// in the sense the hint exists to convey.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    /// The tool does not modify memory content.
    pub read_only_hint: bool,
    /// The tool can delete, merge, or overwrite existing memories.
    pub destructive_hint: bool,
    /// Calling the tool twice with the same arguments has no extra effect.
    pub idempotent_hint: bool,
    /// The tool reaches outside the local store (network, external systems).
    pub open_world_hint: bool,
}

/// Tool description for tools/list
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDescription {
    pub name: String,
    /// Human-readable display name; clients show it instead of `name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    /// Behaviour hints (read-only, destructive, idempotent, open-world).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
    /// Per-tool `_meta` annotations from the MCP wire spec.
    ///
    /// Notable keys recognized by Claude Code (v2.1.91+):
    /// - `anthropic/maxResultSizeChars` (integer, up to 500_000):
    ///   per-tool override of the 50K default `CallToolResult` truncation
    ///   ceiling. Pinned on the Tool definition; applies to every invocation.
    ///
    /// Free-form `serde_json::Value` (typically an object) so additional
    /// vendor-specific `_meta` keys can be added without further schema
    /// changes.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Result of tools/list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListToolsResult {
    pub tools: Vec<ToolDescription>,
}

/// Request for tools/call
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolRequest {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// Result of tools/call
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<ToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

// ============================================================================
// RESOURCES
// ============================================================================

/// Resource description for resources/list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDescription {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Result of resources/list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListResourcesResult {
    pub resources: Vec<ResourceDescription>,
}

/// Request for resources/read
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadResourceRequest {
    pub uri: String,
}

/// Result of resources/read
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContent {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}
