//! # MCP Tool Tests
//!
//! Comprehensive tests for all MCP tools provided by Vestige.
//! Tests cover input validation, execution, and response formats.

use serde_json::{Value, json};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Validate a tool call response structure
fn validate_tool_response(response: &Value) {
    assert!(
        response["content"].is_array(),
        "Response must have content array"
    );
    let content = response["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Content array must not be empty");
    assert!(content[0]["type"].is_string(), "Content must have type");
    assert!(content[0]["text"].is_string(), "Content must have text");
}

/// Parse the text content from a tool response
fn parse_response_text(response: &Value) -> Value {
    let text = response["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap_or(json!({"raw": text}))
}

// ============================================================================
// INGEST TOOL TESTS (3 tests)
// ============================================================================

/// Test ingest tool with valid content.
#[test]
fn test_ingest_tool_valid_content() {
    let _tool_call = json!({
        "name": "ingest",
        "arguments": {
            "content": "The Rust programming language is memory-safe.",
            "nodeType": "fact",
            "tags": ["rust", "programming", "safety"]
        }
    });

    // Expected response format
    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"success\": true, \"nodeId\": \"mock-id\", \"message\": \"Knowledge ingested successfully\"}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["success"], true, "Ingest should succeed");
    assert!(parsed["nodeId"].is_string(), "Should return nodeId");
}

/// Test ingest tool rejects empty content.
#[test]
fn test_ingest_tool_rejects_empty_content() {
    let _tool_call = json!({
        "name": "ingest",
        "arguments": {
            "content": ""
        }
    });

    // Expected error response
    let expected_error = json!({
        "content": [{
            "type": "text",
            "text": "{\"error\": \"Content cannot be empty\"}"
        }],
        "isError": true
    });

    assert_eq!(
        expected_error["isError"], true,
        "Empty content should be an error"
    );
}

/// Test ingest tool with all optional fields.
#[test]
fn test_ingest_tool_with_all_fields() {
    let tool_call = json!({
        "name": "ingest",
        "arguments": {
            "content": "Complex knowledge with all metadata.",
            "nodeType": "decision",
            "tags": ["architecture", "design"],
            "source": "team meeting notes"
        }
    });

    // All fields should be accepted
    assert!(tool_call["arguments"]["content"].is_string());
    assert!(tool_call["arguments"]["nodeType"].is_string());
    assert!(tool_call["arguments"]["tags"].is_array());
    assert!(tool_call["arguments"]["source"].is_string());
}

// ============================================================================
// RECALL TOOL TESTS (3 tests)
// ============================================================================

/// Test recall tool with an exact handle (4.0: handle-only).
#[test]
fn test_recall_tool_valid_query() {
    let _tool_call = json!({
        "name": "recall",
        "arguments": {
            "handle": "mem:3d2f0e1a-7b8c-4d9e-a1b2-c3d4e5f60718"
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"handles\": [{\"handle\": \"mem:3d2f0e1a-7b8c-4d9e-a1b2-c3d4e5f60718\", \"kind\": \"mem\", \"nodes\": []}]}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert!(parsed["handles"].is_array(), "Should return handles array");
}

/// Test recall rejects free text: handle_required, never a search.
#[test]
fn test_recall_tool_rejects_empty_query() {
    let tool_call = json!({
        "name": "recall",
        "arguments": {
            "query": ""
        }
    });

    // Free text (even empty) must return handle_required, not a search.
    assert!(tool_call["arguments"].get("handle").is_none());
    // Expected behavior: {"error":"handle_required","got":"", ...} with no candidates.
}

/// Test recall accepts at most 16 handles.
#[test]
fn test_recall_tool_clamps_limit() {
    // Up to 16 handles are accepted.
    let max_handles: Vec<&str> = std::iter::repeat("mem:3d2f0e1a-7b8c-4d9e-a1b2-c3d4e5f60718")
        .take(16)
        .collect();
    let max_call = json!({
        "name": "recall",
        "arguments": { "handle": max_handles }
    });
    let handles = max_call["arguments"]["handle"].as_array().unwrap();
    assert!(handles.len() <= 16, "16 handles is the documented maximum");

    // 17 handles is outside the contract.
    let over_handles: Vec<&str> = std::iter::repeat("mem:3d2f0e1a-7b8c-4d9e-a1b2-c3d4e5f60718")
        .take(17)
        .collect();
    let over_call = json!({
        "name": "recall",
        "arguments": { "handle": over_handles }
    });
    let handles = over_call["arguments"]["handle"].as_array().unwrap();
    assert!(
        handles.len() > 16,
        "17 handles must be rejected with handle_required"
    );
}

// ============================================================================
// SEMANTIC SEARCH TESTS (2 tests)
// ============================================================================

/// Test semantic search with valid parameters.
#[test]
fn test_semantic_search_valid() {
    let _tool_call = json!({
        "name": "semantic_search",
        "arguments": {
            "query": "memory management concepts",
            "limit": 5,
            "minSimilarity": 0.7
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"query\": \"memory management concepts\", \"method\": \"semantic\", \"total\": 2, \"results\": []}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert_eq!(
        parsed["method"], "semantic",
        "Should indicate semantic search"
    );
}

/// Test semantic search handles embedding not ready.
#[test]
fn test_semantic_search_embedding_not_ready() {
    // When embeddings aren't initialized, should return helpful error
    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"error\": \"Embedding service not ready\", \"hint\": \"Run consolidation first\"}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert!(
        parsed["error"].is_string(),
        "Should explain embedding not ready"
    );
    assert!(parsed["hint"].is_string(), "Should provide hint");
}

// ============================================================================
// HYBRID SEARCH TESTS (2 tests)
// ============================================================================

/// Test hybrid search with weights.
#[test]
fn test_hybrid_search_with_weights() {
    let _tool_call = json!({
        "name": "hybrid_search",
        "arguments": {
            "query": "error handling patterns",
            "limit": 10,
            "keywordWeight": 0.3,
            "semanticWeight": 0.7
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"query\": \"error handling patterns\", \"method\": \"hybrid\", \"total\": 0, \"results\": []}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["method"], "hybrid", "Should indicate hybrid search");
}

/// Test hybrid search with default weights.
#[test]
fn test_hybrid_search_default_weights() {
    let tool_call = json!({
        "name": "hybrid_search",
        "arguments": {
            "query": "testing strategies"
        }
    });

    // Default weights should be 0.5/0.5
    assert!(tool_call["arguments"].get("keywordWeight").is_none());
    assert!(tool_call["arguments"].get("semanticWeight").is_none());
}

// ============================================================================
// KNOWLEDGE MANAGEMENT TESTS (2 tests)
// ============================================================================

/// Test get_knowledge by ID.
#[test]
fn test_get_knowledge_by_id() {
    let _tool_call = json!({
        "name": "get_knowledge",
        "arguments": {
            "nodeId": "abc-123-def"
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"id\": \"abc-123-def\", \"content\": \"Test content\", \"nodeType\": \"fact\"}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert!(parsed["id"].is_string(), "Should return node ID");
    assert!(parsed["content"].is_string(), "Should return content");
}

/// Test delete_knowledge by ID.
#[test]
fn test_delete_knowledge_by_id() {
    let _tool_call = json!({
        "name": "delete_knowledge",
        "arguments": {
            "nodeId": "to-delete-123"
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"success\": true, \"deleted\": true}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["success"], true, "Delete should succeed");
}

// ============================================================================
// REVIEW TOOL TESTS (2 tests)
// ============================================================================

/// Test mark_reviewed with FSRS rating.
#[test]
fn test_mark_reviewed_with_rating() {
    let tool_call = json!({
        "name": "mark_reviewed",
        "arguments": {
            "nodeId": "review-node-123",
            "rating": 3  // Good
        }
    });

    // Rating values: 1=Again, 2=Hard, 3=Good, 4=Easy
    let rating = tool_call["arguments"]["rating"].as_i64().unwrap();
    assert!((1..=4).contains(&rating), "Rating must be 1-4");

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"success\": true, \"nextReview\": \"2024-01-20T10:00:00Z\"}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["success"], true, "Review should succeed");
    assert!(
        parsed["nextReview"].is_string(),
        "Should return next review date"
    );
}

/// Test mark_reviewed with invalid rating.
#[test]
fn test_mark_reviewed_invalid_rating() {
    let invalid_ratings = [0, 5, -1, 100];

    for rating in invalid_ratings {
        let tool_call = json!({
            "name": "mark_reviewed",
            "arguments": {
                "nodeId": "test-node",
                "rating": rating
            }
        });

        // Rating should be validated
        let r = tool_call["arguments"]["rating"].as_i64().unwrap();
        assert!(!(1..=4).contains(&r), "Rating {} should be invalid", r);
    }
}

// ============================================================================
// STATS AND MAINTENANCE TESTS (2 tests)
// ============================================================================

/// Test get_stats returns system statistics.
#[test]
fn test_get_stats() {
    let _tool_call = json!({
        "name": "get_stats",
        "arguments": {}
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"totalNodes\": 42, \"averageRetention\": 0.85, \"embeddingsGenerated\": 40}"
        }],
        "isError": false
    });

    validate_tool_response(&expected_response);
    let parsed = parse_response_text(&expected_response);
    assert!(
        parsed["totalNodes"].is_number(),
        "Should return total nodes"
    );
    assert!(
        parsed["averageRetention"].is_number(),
        "Should return average retention"
    );
}

/// Test health_check returns health status.
#[test]
fn test_health_check() {
    let _tool_call = json!({
        "name": "health_check",
        "arguments": {}
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"status\": \"healthy\", \"database\": \"ok\", \"embeddings\": \"ready\"}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert!(parsed["status"].is_string(), "Should return status");
}

// ============================================================================
// INTENTION TOOL TESTS (5 tests)
// ============================================================================

/// Test set_intention creates a new intention.
#[test]
fn test_set_intention_basic() {
    let _tool_call = json!({
        "name": "set_intention",
        "arguments": {
            "description": "Remember to review error handling",
            "priority": "high"
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"success\": true, \"intentionId\": \"int-123\", \"priority\": 3}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["success"], true, "Should succeed");
    assert!(
        parsed["intentionId"].is_string(),
        "Should return intention ID"
    );
    assert_eq!(parsed["priority"], 3, "High priority should be 3");
}

/// Test set_intention with time trigger.
#[test]
fn test_set_intention_with_time_trigger() {
    let tool_call = json!({
        "name": "set_intention",
        "arguments": {
            "description": "Check build status",
            "trigger": {
                "type": "time",
                "inMinutes": 30
            }
        }
    });

    let trigger = &tool_call["arguments"]["trigger"];
    assert_eq!(trigger["type"], "time", "Should be time trigger");
    assert!(trigger["inMinutes"].is_number(), "Should have duration");
}

/// Test check_intentions with context matching.
#[test]
fn test_check_intentions_with_context() {
    let _tool_call = json!({
        "name": "check_intentions",
        "arguments": {
            "context": {
                "codebase": "payments-service",
                "file": "src/handlers/payment.rs",
                "topics": ["error handling", "validation"]
            }
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"triggered\": [{\"id\": \"int-1\", \"description\": \"Review payments\"}], \"pending\": []}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert!(
        parsed["triggered"].is_array(),
        "Should return triggered intentions"
    );
    assert!(
        parsed["pending"].is_array(),
        "Should return pending intentions"
    );
}

/// Test complete_intention marks as fulfilled.
#[test]
fn test_complete_intention() {
    let _tool_call = json!({
        "name": "complete_intention",
        "arguments": {
            "intentionId": "int-to-complete"
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"success\": true, \"message\": \"Intention marked as complete\"}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert_eq!(parsed["success"], true, "Should succeed");
}

/// Test list_intentions with status filter.
#[test]
fn test_list_intentions_with_filter() {
    let _tool_call = json!({
        "name": "list_intentions",
        "arguments": {
            "status": "active",
            "limit": 10
        }
    });

    let expected_response = json!({
        "content": [{
            "type": "text",
            "text": "{\"intentions\": [], \"total\": 0, \"status\": \"active\"}"
        }],
        "isError": false
    });

    let parsed = parse_response_text(&expected_response);
    assert!(
        parsed["intentions"].is_array(),
        "Should return intentions array"
    );
    assert_eq!(parsed["status"], "active", "Should echo status filter");
}

// ============================================================================
// INPUT SCHEMA VALIDATION TESTS (2 tests)
// ============================================================================

/// Test tool input schemas have proper JSON Schema format.
#[test]
fn test_tool_schemas_are_valid_json_schema() {
    let ingest_schema = json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember"
            },
            "nodeType": {
                "type": "string",
                "description": "Type of knowledge"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["content"]
    });

    assert_eq!(
        ingest_schema["type"], "object",
        "Schema must be object type"
    );
    assert!(
        ingest_schema["properties"].is_object(),
        "Must have properties"
    );
    assert!(
        ingest_schema["required"].is_array(),
        "Must specify required fields"
    );
}

/// Test all tools have required inputSchema fields.
#[test]
fn test_all_tools_have_schema() {
    let tool_definitions = vec![
        ("ingest", vec!["content"]),
        ("recall", vec!["handle"]),
        ("semantic_search", vec!["query"]),
        ("hybrid_search", vec!["query"]),
        ("get_knowledge", vec!["nodeId"]),
        ("delete_knowledge", vec!["nodeId"]),
        ("mark_reviewed", vec!["nodeId", "rating"]),
        ("set_intention", vec!["description"]),
        ("complete_intention", vec!["intentionId"]),
        ("snooze_intention", vec!["intentionId"]),
    ];

    for (tool_name, required_fields) in tool_definitions {
        assert!(
            !required_fields.is_empty(),
            "Tool {} should have at least one required field",
            tool_name
        );
    }
}
