//! MCP JSON-RPC Types
//!
//! Core types for JSON-RPC 2.0 protocol used by MCP.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP Protocol Version
pub const MCP_VERSION: &str = "2025-11-25";

/// JSON-RPC version
pub const JSONRPC_VERSION: &str = "2.0";

// ============================================================================
// JSON-RPC REQUEST/RESPONSE
// ============================================================================

/// JSON-RPC Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}


/// JSON-RPC Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

// ============================================================================
// JSON-RPC ERROR
// ============================================================================

/// JSON-RPC Error Codes (standard + MCP-specific)
#[derive(Debug, Clone, Copy)]
pub enum ErrorCode {
    // Standard JSON-RPC errors
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,

    // MCP-specific errors (-32000 to -32099)
    ConnectionClosed = -32000,
    RequestTimeout = -32001,
    ResourceNotFound = -32002,
    ServerNotInitialized = -32003,
}

impl From<ErrorCode> for i32 {
    fn from(code: ErrorCode) -> Self {
        code as i32
    }
}

/// JSON-RPC Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    fn new(code: ErrorCode, message: &str) -> Self {
        Self {
            code: code.into(),
            message: message.to_string(),
            data: None,
        }
    }

    pub fn parse_error() -> Self {
        Self::new(ErrorCode::ParseError, "Parse error")
    }

    pub fn method_not_found() -> Self {
        Self::new(ErrorCode::MethodNotFound, "Method not found")
    }

    pub fn method_not_found_with_message(message: &str) -> Self {
        Self::new(ErrorCode::MethodNotFound, message)
    }

    pub fn invalid_params(message: &str) -> Self {
        Self::new(ErrorCode::InvalidParams, message)
    }

    pub fn internal_error(message: &str) -> Self {
        Self::new(ErrorCode::InternalError, message)
    }

    pub fn server_not_initialized() -> Self {
        Self::new(ErrorCode::ServerNotInitialized, "Server not initialized")
    }

    #[allow(dead_code)] // Reserved for future resource handling
    pub fn resource_not_found(uri: &str) -> Self {
        Self::new(ErrorCode::ResourceNotFound, &format!("Resource not found: {}", uri))
    }
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(Value::Number(1.into())),
            method: "test".to_string(),
            params: Some(serde_json::json!({"key": "value"})),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.method, "test");
        assert!(parsed.id.is_some()); // Has id, not a notification
    }

    #[test]
    fn test_notification() {
        let notification = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: None,
            method: "notify".to_string(),
            params: None,
        };

        assert!(notification.id.is_none()); // No id = notification
    }

    #[test]
    fn test_response_success() {
        let response = JsonRpcResponse::success(
            Some(Value::Number(1.into())),
            serde_json::json!({"result": "ok"}),
        );

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_response_error() {
        let response = JsonRpcResponse::error(
            Some(Value::Number(1.into())),
            JsonRpcError::method_not_found(),
        );

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }
}
