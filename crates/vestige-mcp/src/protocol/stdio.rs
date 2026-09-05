//! stdio Transport for MCP
//!
//! Handles JSON-RPC communication over stdin/stdout.
//! v1.9.2: Async tokio I/O with error resilience.

use std::io;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::server::McpServer;

/// Maximum consecutive I/O errors before giving up
const MAX_CONSECUTIVE_ERRORS: u32 = 5;

/// Handle that background work (a first-run model download, a reranker load)
/// uses to push server-initiated `notifications/message` lines onto stdout
/// between responses. Clients that render MCP logging show them; others
/// ignore them. Sending never blocks and never fails the sender.
#[derive(Clone)]
pub struct Notifier {
    tx: mpsc::UnboundedSender<Value>,
}

impl Notifier {
    /// Queue one logging notification. `level` is an MCP log level
    /// (`info`, `warning`, ...), `logger` names the subsystem.
    pub fn log(&self, level: &str, logger: &str, data: Value) {
        let _ = self.tx.send(Self::message(level, logger, data));
    }

    /// The wire shape of a logging notification, kept pure for tests.
    pub fn message(level: &str, logger: &str, data: Value) -> Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": { "level": level, "logger": logger, "data": data }
        })
    }
}

/// Resolve the next queued notification, or wait forever when the transport
/// has no notification channel (or the channel closed and was dropped).
async fn next_notification(rx: &mut Option<mpsc::UnboundedReceiver<Value>>) -> Option<Value> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// stdio Transport for MCP server
pub struct StdioTransport {
    notifications: Option<mpsc::UnboundedReceiver<Value>>,
}

impl StdioTransport {
    pub fn new() -> Self {
        Self { notifications: None }
    }

    /// A transport plus the [`Notifier`] that feeds it.
    pub fn with_notifications() -> (Self, Notifier) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                notifications: Some(rx),
            },
            Notifier { tx },
        )
    }

    /// Run the MCP server over stdio with error resilience.
    pub async fn run(mut self, mut server: McpServer) -> Result<(), io::Error> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();

        let mut reader = BufReader::new(stdin);
        let mut stdout = stdout;
        let mut consecutive_errors: u32 = 0;
        let mut line_buf = String::new();

        loop {
            line_buf.clear();

            tokio::select! {
                result = reader.read_line(&mut line_buf) => {
                    match result {
                        Ok(0) => {
                            // Clean EOF — stdin closed
                            info!("stdin closed (EOF), shutting down");
                            break;
                        }
                        Ok(_) => {
                            consecutive_errors = 0;
                            let line = line_buf.trim();

                            if line.is_empty() {
                                continue;
                            }

                            debug!("Received: {} bytes", line.len());

                            // Parse JSON-RPC request
                            let request: JsonRpcRequest = match serde_json::from_str(line) {
                                Ok(r) => r,
                                Err(e) => {
                                    warn!("Failed to parse request: {}", e);
                                    let error_response = JsonRpcResponse::error(None, JsonRpcError::parse_error());
                                    match serde_json::to_string(&error_response) {
                                        Ok(response_json) => {
                                            let out = format!("{}\n", response_json);
                                            stdout.write_all(out.as_bytes()).await?;
                                            stdout.flush().await?;
                                        }
                                        Err(e) => {
                                            error!("Failed to serialize error response: {}", e);
                                            let fallback = "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"Internal error\"}}\n";
                                            let _ = stdout.write_all(fallback.as_bytes()).await;
                                            let _ = stdout.flush().await;
                                        }
                                    }
                                    continue;
                                }
                            };

                            // Handle the request
                            if let Some(response) = server.handle_request(request).await {
                                match serde_json::to_string(&response) {
                                    Ok(response_json) => {
                                        debug!("Sending: {} bytes", response_json.len());
                                        let out = format!("{}\n", response_json);
                                        stdout.write_all(out.as_bytes()).await?;
                                        stdout.flush().await?;
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize response: {}", e);
                                        let fallback = "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"Internal error\"}}\n";
                                        let _ = stdout.write_all(fallback.as_bytes()).await;
                                        let _ = stdout.flush().await;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            warn!(
                                "I/O error reading stdin ({}/{}): {}",
                                consecutive_errors, MAX_CONSECUTIVE_ERRORS, e
                            );
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                error!(
                                    "Too many consecutive I/O errors ({}), shutting down",
                                    consecutive_errors
                                );
                                break;
                            }
                            // Brief pause before retrying
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
                // Held until the handshake completes: a logging line before the
                // initialize response would desync a client that reads the next
                // line as its answer. The channel buffers meanwhile.
                notification = next_notification(&mut self.notifications), if server.is_initialized() => {
                    match notification {
                        Some(notification) => match serde_json::to_string(&notification) {
                            Ok(json) => {
                                let out = format!("{}\n", json);
                                stdout.write_all(out.as_bytes()).await?;
                                stdout.flush().await?;
                            }
                            Err(e) => warn!("Failed to serialize notification: {}", e),
                        },
                        // Every sender is gone: park the branch instead of
                        // spinning on a closed channel.
                        None => self.notifications = None,
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

#[cfg(test)]
mod tests {
    use super::Notifier;

    #[test]
    fn logging_notification_has_the_mcp_wire_shape() {
        let message = Notifier::message(
            "info",
            "vestige.embeddings",
            serde_json::json!({ "event": "model_download_started" }),
        );
        assert_eq!(message["jsonrpc"], "2.0");
        assert_eq!(message["method"], "notifications/message");
        assert_eq!(message["params"]["level"], "info");
        assert_eq!(message["params"]["logger"], "vestige.embeddings");
        assert_eq!(message["params"]["data"]["event"], "model_download_started");
        assert!(message.get("id").is_none(), "notifications carry no id");
    }

    #[tokio::test]
    async fn queued_notifications_are_delivered_in_order() {
        let (mut transport, notifier) = super::StdioTransport::with_notifications();
        notifier.log("info", "t", serde_json::json!({ "n": 1 }));
        notifier.log("info", "t", serde_json::json!({ "n": 2 }));
        let first = super::next_notification(&mut transport.notifications)
            .await
            .unwrap();
        let second = super::next_notification(&mut transport.notifications)
            .await
            .unwrap();
        assert_eq!(first["params"]["data"]["n"], 1);
        assert_eq!(second["params"]["data"]["n"], 2);
    }
}
