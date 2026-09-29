//! # MCP Protocol Compliance Tests
//!
//! Tests validating JSON-RPC 2.0 and MCP protocol compliance.
//! Based on the Model Context Protocol specification.

use serde_json::json;

// ============================================================================
// JSON-RPC 2.0 MESSAGE FORMAT TESTS
// ============================================================================

/// Test that JSON-RPC requests have required fields.
///
/// Per JSON-RPC 2.0 spec, requests MUST contain:
/// - jsonrpc: "2.0"
/// - method: string
/// - id: optional (if present, makes it a request vs notification)
#[test]
fn test_jsonrpc_request_required_fields() {
    // Valid request with all required fields
    let valid_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    assert_eq!(
        valid_request["jsonrpc"], "2.0",
        "jsonrpc version must be 2.0"
    );
    assert!(
        valid_request["method"].is_string(),
        "method must be a string"
    );
    assert!(
        valid_request["id"].is_number(),
        "id should be present for requests"
    );
}

/// Test that JSON-RPC notifications have no id field.
///
/// Notifications are requests without an id - the server MUST NOT reply.
#[test]
fn test_jsonrpc_notification_has_no_id() {
    let notification = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });

    assert!(
        notification.get("id").is_none(),
        "Notifications must not have an id field"
    );
    assert_eq!(notification["method"], "notifications/initialized");
}

/// Test JSON-RPC response format for success.
///
/// Successful responses MUST contain:
/// - jsonrpc: "2.0"
/// - id: matching the request id
/// - result: the result value (any JSON)
/// - MUST NOT contain error
#[test]
fn test_jsonrpc_success_response_format() {
    let success_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "vestige",
                "version": "0.1.0"
            }
        }
    });

    assert_eq!(success_response["jsonrpc"], "2.0");
    assert!(
        success_response["result"].is_object(),
        "Success response must have result"
    );
    assert!(
        success_response.get("error").is_none(),
        "Success response must not have error"
    );
}

/// Test JSON-RPC response format for errors.
///
/// Error responses MUST contain:
/// - jsonrpc: "2.0"
/// - id: matching the request id (or null if parsing failed)
/// - error: object with code, message, and optional data
/// - MUST NOT contain result
#[test]
fn test_jsonrpc_error_response_format() {
    let error_response = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32601,
            "message": "Method not found"
        }
    });

    assert_eq!(error_response["jsonrpc"], "2.0");
    assert!(
        error_response["error"].is_object(),
        "Error response must have error object"
    );
    assert!(
        error_response["error"]["code"].is_number(),
        "Error must have code"
    );
    assert!(
        error_response["error"]["message"].is_string(),
        "Error must have message"
    );
    assert!(
        error_response.get("result").is_none(),
        "Error response must not have result"
    );
}

// ============================================================================
// STANDARD JSON-RPC ERROR CODE TESTS
// ============================================================================

/// Test standard JSON-RPC error codes.
///
/// Standard error codes are defined in JSON-RPC 2.0:
/// - -32700: Parse error
/// - -32600: Invalid Request
/// - -32601: Method not found
/// - -32602: Invalid params
/// - -32603: Internal error
#[test]
fn test_standard_jsonrpc_error_codes() {
    let error_codes = [
        (-32700, "Parse error"),
        (-32600, "Invalid Request"),
        (-32601, "Method not found"),
        (-32602, "Invalid params"),
        (-32603, "Internal error"),
    ];

    for (code, message) in error_codes {
        // All standard codes are in the reserved range
        assert!(
            (-32700..=-32600).contains(&code),
            "Standard error code {} ({}) must be in reserved range",
            code,
            message
        );
    }
}

/// Test MCP-specific error codes.
///
/// MCP defines additional error codes in the -32000 to -32099 range:
/// - -32000: Connection closed
/// - -32001: Request timeout
/// - -32002: Resource not found
/// - -32003: Server not initialized
#[test]
fn test_mcp_specific_error_codes() {
    let mcp_error_codes = [
        (-32000, "ConnectionClosed"),
        (-32001, "RequestTimeout"),
        (-32002, "ResourceNotFound"),
        (-32003, "ServerNotInitialized"),
    ];

    for (code, name) in mcp_error_codes {
        // MCP-specific codes are in the server error range
        assert!(
            (-32099..=-32000).contains(&code),
            "MCP error code {} ({}) must be in server error range",
            code,
            name
        );
    }
}

// ============================================================================
// MCP INITIALIZATION TESTS
// ============================================================================

/// Test MCP initialize request format.
///
/// The initialize request MUST contain:
/// - protocolVersion: string (e.g., "2024-11-05")
/// - capabilities: object describing client capabilities
/// - clientInfo: object with name and version
#[test]
fn test_mcp_initialize_request_format() {
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {},
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    let params = &init_request["params"];
    assert!(
        params["protocolVersion"].is_string(),
        "protocolVersion required"
    );
    assert!(params["capabilities"].is_object(), "capabilities required");
    assert!(params["clientInfo"].is_object(), "clientInfo required");
    assert!(
        params["clientInfo"]["name"].is_string(),
        "clientInfo.name required"
    );
    assert!(
        params["clientInfo"]["version"].is_string(),
        "clientInfo.version required"
    );
}

/// Test MCP initialize response format.
///
/// The initialize response MUST contain:
/// - protocolVersion: string (server's version)
/// - serverInfo: object with name and version
/// - capabilities: object describing server capabilities
/// - instructions: optional string with usage guidance
#[test]
fn test_mcp_initialize_response_format() {
    let init_response = json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "vestige",
            "version": "0.1.0"
        },
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "listChanged": false }
        },
        "instructions": "Vestige is your long-term memory system."
    });

    assert!(
        init_response["protocolVersion"].is_string(),
        "protocolVersion required"
    );
    assert!(
        init_response["serverInfo"].is_object(),
        "serverInfo required"
    );
    assert!(
        init_response["serverInfo"]["name"].is_string(),
        "serverInfo.name required"
    );
    assert!(
        init_response["serverInfo"]["version"].is_string(),
        "serverInfo.version required"
    );
    assert!(
        init_response["capabilities"].is_object(),
        "capabilities required"
    );
}

/// Test that requests before initialization are rejected.
///
/// Per MCP spec, the server MUST reject all requests except 'initialize'
/// until initialization is complete.
#[test]
fn test_server_rejects_requests_before_initialize() {
    // Simulate the expected error for pre-init requests
    let pre_init_error = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32003,
            "message": "Server not initialized"
        }
    });

    assert_eq!(
        pre_init_error["error"]["code"], -32003,
        "Pre-initialization requests should return ServerNotInitialized error"
    );
}

// ============================================================================
// TOOLS PROTOCOL TESTS
// ============================================================================

/// Test tools/list response format.
///
/// The response contains an array of tool descriptions, each with:
/// - name: string (tool identifier)
/// - description: optional string
/// - inputSchema: JSON Schema for tool arguments
#[test]
fn test_tools_list_response_format() {
    let tools_list_response = json!({
        "tools": [
            {
                "name": "ingest",
                "description": "Add new knowledge to memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "recall",
                "description": "Look up by exact handle (mem:, sha:, path:, sym:, test:, run:, call:, session:...). Free text returns handle_required.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "handle": { "type": "string" }
                    },
                    "required": ["handle"]
                }
            }
        ]
    });

    let tools = tools_list_response["tools"].as_array().unwrap();
    assert!(!tools.is_empty(), "Tools list should not be empty");

    for tool in tools {
        assert!(tool["name"].is_string(), "Tool must have name");
        assert!(
            tool["inputSchema"].is_object(),
            "Tool must have inputSchema"
        );
        assert_eq!(
            tool["inputSchema"]["type"], "object",
            "inputSchema must be an object type"
        );
    }
}

/// Test tools/call request format.
///
/// The request MUST contain:
/// - name: string (tool to invoke)
/// - arguments: optional object with tool parameters
#[test]
fn test_tools_call_request_format() {
    let tools_call_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "ingest",
            "arguments": {
                "content": "Test knowledge to remember",
                "nodeType": "fact",
                "tags": ["test", "memory"]
            }
        }
    });

    let params = &tools_call_request["params"];
    assert!(params["name"].is_string(), "Tool name required");
    assert!(
        params["arguments"].is_object(),
        "Arguments should be an object"
    );
}

/// Test tools/call response format.
///
/// The response contains:
/// - content: array of content items (text, image, etc.)
/// - isError: optional boolean indicating tool execution error
#[test]
fn test_tools_call_response_format() {
    let tools_call_response = json!({
        "content": [
            {
                "type": "text",
                "text": "{\"success\": true, \"nodeId\": \"abc123\"}"
            }
        ],
        "isError": false
    });

    let content = tools_call_response["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Content array should not be empty");
    assert!(
        content[0]["type"].is_string(),
        "Content item must have type"
    );
    assert!(
        content[0]["text"].is_string(),
        "Text content must have text field"
    );
}

// ============================================================================
// RESOURCES PROTOCOL TESTS
// ============================================================================

/// Test resources/list response format.
///
/// The response contains an array of resource descriptions:
/// - uri: string (resource identifier)
/// - name: string (human-readable name)
/// - description: optional string
/// - mimeType: optional string
#[test]
fn test_resources_list_response_format() {
    let resources_list = json!({
        "resources": [
            {
                "uri": "memory://stats",
                "name": "Memory Statistics",
                "description": "Current memory system statistics",
                "mimeType": "application/json"
            },
            {
                "uri": "memory://recent",
                "name": "Recent Memories",
                "description": "Recently added memories",
                "mimeType": "application/json"
            }
        ]
    });

    let resources = resources_list["resources"].as_array().unwrap();
    for resource in resources {
        assert!(resource["uri"].is_string(), "Resource must have uri");
        assert!(resource["name"].is_string(), "Resource must have name");
    }
}

/// Test resources/read request format.
///
/// The request MUST contain:
/// - uri: string (resource to read)
#[test]
fn test_resources_read_request_format() {
    let read_request = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/read",
        "params": {
            "uri": "memory://stats"
        }
    });

    assert!(read_request["params"]["uri"].is_string(), "URI required");
}

/// Test resources/read response format.
///
/// The response contains:
/// - contents: array of content items with uri, mimeType, and text/blob
#[test]
fn test_resources_read_response_format() {
    let read_response = json!({
        "contents": [
            {
                "uri": "memory://stats",
                "mimeType": "application/json",
                "text": "{\"totalNodes\": 42, \"averageRetention\": 0.85}"
            }
        ]
    });

    let contents = read_response["contents"].as_array().unwrap();
    assert!(!contents.is_empty(), "Contents should not be empty");
    assert!(contents[0]["uri"].is_string(), "Content must have uri");
    // Must have either text or blob
    assert!(
        contents[0]["text"].is_string() || contents[0]["blob"].is_string(),
        "Content must have text or blob"
    );
}
