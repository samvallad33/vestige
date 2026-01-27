//! stdio Transport for MCP
//!
//! Handles JSON-RPC communication over stdin/stdout.

use std::io::{self, BufRead, BufReader, Write};
use tracing::{debug, error, warn};

use super::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::server::McpServer;

/// stdio Transport for MCP server
pub struct StdioTransport;

impl StdioTransport {
    pub fn new() -> Self {
        Self
    }

    /// Run the MCP server over stdio
    pub async fn run(self, mut server: McpServer) -> Result<(), io::Error> {
        let stdin = io::stdin();
        let stdout = io::stdout();

        let reader = BufReader::new(stdin.lock());
        let mut stdout = stdout.lock();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Failed to read line: {}", e);
                    break;
                }
            };

            if line.is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            // Parse JSON-RPC request
            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    warn!("Failed to parse request: {}", e);
                    let error_response = JsonRpcResponse::error(None, JsonRpcError::parse_error());
                    match serde_json::to_string(&error_response) {
                        Ok(response_json) => {
                            writeln!(stdout, "{}", response_json)?;
                            stdout.flush()?;
                        }
                        Err(e) => {
                            error!("Failed to serialize error response: {}", e);
                            // Send a minimal error response so client doesn't hang
                            let fallback = r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#;
                            let _ = writeln!(stdout, "{}", fallback);
                            let _ = stdout.flush();
                        }
                    }
                    continue;
                }
            };

            // Handle the request
            if let Some(response) = server.handle_request(request).await {
                match serde_json::to_string(&response) {
                    Ok(response_json) => {
                        debug!("Sending: {}", response_json);
                        writeln!(stdout, "{}", response_json)?;
                        stdout.flush()?;
                    }
                    Err(e) => {
                        error!("Failed to serialize response: {}", e);
                        // Send a minimal error response so client doesn't hang
                        let fallback = r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#;
                        let _ = writeln!(stdout, "{}", fallback);
                        let _ = stdout.flush();
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}
