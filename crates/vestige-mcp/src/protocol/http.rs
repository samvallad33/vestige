//! Streamable HTTP transport for MCP.
//!
//! Implements the MCP Streamable HTTP transport specification:
//! - `POST /mcp` — JSON-RPC endpoint (initialize, tools/call, etc.)
//! - `DELETE /mcp` — session cleanup
//!
//! Each client gets a per-session `McpServer` instance (owns `initialized` state).
//! Shared state (Storage, CognitiveEngine, event bus) is shared across sessions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{delete, post};
use axum::{Json, Router};
use subtle::ConstantTimeEq;
use tokio::sync::{Mutex, RwLock, broadcast};
use tower::ServiceBuilder;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use crate::cognitive::CognitiveEngine;
use crate::dashboard::events::VestigeEvent;
use crate::protocol::types::JsonRpcRequest;
use crate::server::McpServer;
use vestige_core::Storage;

/// Maximum concurrent sessions.
const MAX_SESSIONS: usize = 100;

/// Sessions idle longer than this are reaped.
const SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// How often the reaper task runs.
const REAPER_INTERVAL: Duration = Duration::from_secs(5 * 60);

/// Concurrency limit for the tower middleware.
const CONCURRENCY_LIMIT: usize = 50;

/// Maximum request body size (256 KB — JSON-RPC requests should be small).
const MAX_BODY_SIZE: usize = 256 * 1024;

/// A per-client session holding its own McpServer instance.
struct Session {
    server: McpServer,
    last_active: Instant,
    protocol_version: String,
}

/// Shared state cloned into every axum handler.
#[derive(Clone)]
pub struct HttpTransportState {
    sessions: Arc<RwLock<HashMap<String, Arc<Mutex<Session>>>>>,
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    event_tx: broadcast::Sender<VestigeEvent>,
    auth_token: String,
    allowed_origins: Arc<Vec<String>>,
}

/// Start the HTTP MCP transport on `127.0.0.1:<port>`.
///
/// This function spawns a background tokio task and returns immediately.
pub async fn start_http_transport(
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    event_tx: broadcast::Sender<VestigeEvent>,
    auth_token: String,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = HttpTransportState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        storage,
        cognitive,
        event_tx,
        auth_token,
        allowed_origins: Arc::new(allowed_origins(port)),
    };

    // Spawn session reaper
    {
        let sessions = Arc::clone(&state.sessions);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(REAPER_INTERVAL).await;
                let mut map = sessions.write().await;
                let before = map.len();
                map.retain(|_id, session| {
                    // Try to check last_active without blocking; skip if locked
                    match session.try_lock() {
                        Ok(s) => s.last_active.elapsed() < SESSION_TIMEOUT,
                        Err(_) => true, // in-use, keep
                    }
                });
                let removed = before - map.len();
                if removed > 0 {
                    info!(
                        "Session reaper: removed {} idle sessions ({} active)",
                        removed,
                        map.len()
                    );
                }
            }
        });
    }

    let cors_origins = state
        .allowed_origins
        .iter()
        .filter_map(|origin| origin.parse().ok())
        .collect::<Vec<_>>();

    let app = Router::new()
        .route("/mcp", post(post_mcp))
        .route("/mcp", delete(delete_mcp))
        .layer(
            ServiceBuilder::new()
                .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
                .layer(ConcurrencyLimitLayer::new(CONCURRENCY_LIMIT))
                .layer(
                    CorsLayer::new()
                        .allow_origin(cors_origins)
                        .allow_methods([
                            axum::http::Method::POST,
                            axum::http::Method::DELETE,
                            axum::http::Method::OPTIONS,
                        ])
                        .allow_headers([
                            axum::http::header::CONTENT_TYPE,
                            axum::http::header::AUTHORIZATION,
                            axum::http::HeaderName::from_static("mcp-protocol-version"),
                            axum::http::HeaderName::from_static("mcp-session-id"),
                        ])
                        .expose_headers([
                            axum::http::HeaderName::from_static("mcp-protocol-version"),
                            axum::http::HeaderName::from_static("mcp-session-id"),
                        ]),
                ),
        )
        .with_state(state);

    // Bind to localhost only — use VESTIGE_HTTP_BIND=0.0.0.0 for remote access
    let bind_addr: std::net::IpAddr = std::env::var("VESTIGE_HTTP_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));

    let addr = std::net::SocketAddr::from((bind_addr, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("HTTP MCP transport listening on http://{}/mcp", addr);

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            warn!("HTTP transport error: {}", e);
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate the `Authorization: Bearer <token>` header using constant-time
/// comparison to prevent timing side-channel attacks.
fn validate_auth(headers: &HeaderMap, expected: &str) -> Result<(), (StatusCode, &'static str)> {
    let header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or((StatusCode::UNAUTHORIZED, "Missing Authorization header"))?;

    // The auth-scheme token is case-insensitive per RFC 7235/6750, so match
    // "Bearer" case-insensitively (accepting "bearer", "BEARER", ...) while
    // preserving the token's original case.
    let token = header
        .split_once(' ')
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, token)| token.trim_start())
        .ok_or((
            StatusCode::UNAUTHORIZED,
            "Invalid Authorization scheme (expected Bearer)",
        ))?;

    // Constant-time comparison: prevents timing side-channel attacks.
    // We first check lengths match (length itself is not secret since UUIDs
    // have a fixed public format), then compare bytes in constant time.
    let token_bytes = token.as_bytes();
    let expected_bytes = expected.as_bytes();

    if token_bytes.len() != expected_bytes.len()
        || token_bytes.ct_eq(expected_bytes).unwrap_u8() != 1
    {
        return Err((StatusCode::FORBIDDEN, "Invalid auth token"));
    }

    Ok(())
}

fn allowed_origins(port: u16) -> Vec<String> {
    if let Ok(configured) = std::env::var("VESTIGE_HTTP_ALLOWED_ORIGINS") {
        let origins: Vec<String> = configured
            .split(',')
            .map(str::trim)
            .filter(|origin| !origin.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        if !origins.is_empty() {
            return origins;
        }
    }

    vec![
        format!("http://127.0.0.1:{}", port),
        format!("http://localhost:{}", port),
    ]
}

fn validate_origin(
    headers: &HeaderMap,
    allowed_origins: &[String],
) -> Result<(), (StatusCode, &'static str)> {
    let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) else {
        return Ok(());
    };

    if allowed_origins.iter().any(|allowed| allowed == origin) {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "Origin not allowed"))
    }
}

fn validate_accept(headers: &HeaderMap) -> Result<(), (StatusCode, &'static str)> {
    let Some(accept) = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()) else {
        return Err((
            StatusCode::NOT_ACCEPTABLE,
            "Accept must include application/json and text/event-stream",
        ));
    };

    let mut accepts_json = false;
    let mut accepts_sse = false;
    for mime in accept
        .split(',')
        .map(|part| part.trim().split(';').next().unwrap_or("").trim())
    {
        // `*/*` accepts anything; `application/*` / `text/*` accept a whole type.
        // Honoring these lets generic HTTP clients (curl's default Accept: */*)
        // reach /mcp instead of getting a hard 406.
        accepts_json |=
            mime == "application/json" || mime == "application/*" || mime == "*/*";
        accepts_sse |= mime == "text/event-stream" || mime == "text/*" || mime == "*/*";
    }

    if accepts_json && accepts_sse {
        Ok(())
    } else {
        Err((
            StatusCode::NOT_ACCEPTABLE,
            "Accept must include application/json and text/event-stream",
        ))
    }
}

fn protocol_version_from_headers(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("mcp-protocol-version")
        .and_then(|v| v.to_str().ok())
}

fn validate_protocol_version(
    headers: &HeaderMap,
    expected: &str,
) -> Result<(), (StatusCode, &'static str)> {
    let Some(version) = protocol_version_from_headers(headers) else {
        return Err((
            StatusCode::BAD_REQUEST,
            "MCP-Protocol-Version header required",
        ));
    };

    if version == expected {
        Ok(())
    } else {
        Err((StatusCode::BAD_REQUEST, "MCP-Protocol-Version mismatch"))
    }
}

fn response_protocol_version(response: &crate::protocol::types::JsonRpcResponse) -> Option<String> {
    if response.error.is_some() {
        return None;
    }

    response
        .result
        .as_ref()
        .and_then(|result| result.get("protocolVersion"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

/// Extract and validate the `Mcp-Session-Id` header value.
///
/// Only accepts valid UUID v4 format (8-4-4-4-12 hex) to prevent header
/// injection and ensure session IDs match server-generated format.
fn session_id_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| uuid::Uuid::parse_str(s).is_ok())
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /mcp` — main JSON-RPC handler.
async fn post_mcp(
    State(state): State<HttpTransportState>,
    headers: HeaderMap,
    Json(request): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    if let Err((status, msg)) = validate_origin(&headers, &state.allowed_origins) {
        return (status, HeaderMap::new(), msg.to_string()).into_response();
    }
    if let Err((status, msg)) = validate_accept(&headers) {
        return (status, HeaderMap::new(), msg.to_string()).into_response();
    }

    // Auth check
    if let Err((status, msg)) = validate_auth(&headers, &state.auth_token) {
        return (status, HeaderMap::new(), msg.to_string()).into_response();
    }

    let is_initialize = request.method == "initialize";

    if is_initialize {
        // ── New session ──
        // Take write lock immediately to avoid TOCTOU race on MAX_SESSIONS check.
        let mut sessions = state.sessions.write().await;
        if sessions.len() >= MAX_SESSIONS {
            return (StatusCode::SERVICE_UNAVAILABLE, "Too many active sessions").into_response();
        }

        let server = McpServer::new_with_events(
            Arc::clone(&state.storage),
            Arc::clone(&state.cognitive),
            state.event_tx.clone(),
        );

        let session_id = uuid::Uuid::new_v4().to_string();

        let session = Arc::new(Mutex::new(Session {
            server,
            last_active: Instant::now(),
            protocol_version: crate::protocol::types::MCP_VERSION.to_string(),
        }));

        // Handle the initialize request
        let response = {
            let mut sess = session.lock().await;
            sess.server.handle_request(request).await
        };

        match response {
            Some(resp) => {
                let Some(protocol_version) = response_protocol_version(&resp) else {
                    drop(sessions);
                    return (StatusCode::OK, HeaderMap::new(), Json(resp)).into_response();
                };
                {
                    let mut sess = session.lock().await;
                    sess.protocol_version = protocol_version.clone();
                }
                sessions.insert(session_id.clone(), session);
                drop(sessions);
                let mut resp_headers = HeaderMap::new();
                resp_headers.insert(
                    "mcp-session-id",
                    session_id
                        .parse()
                        .unwrap_or_else(|_| axum::http::HeaderValue::from_static("invalid")),
                );
                if let Ok(value) = HeaderValue::from_str(&protocol_version) {
                    resp_headers.insert("mcp-protocol-version", value);
                }
                (StatusCode::OK, resp_headers, Json(resp)).into_response()
            }
            None => {
                drop(sessions);
                (StatusCode::ACCEPTED, HeaderMap::new()).into_response()
            }
        }
    } else {
        // ── Existing session ──
        let session_id = match session_id_from_headers(&headers) {
            Some(id) => id,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    "Missing or invalid Mcp-Session-Id header",
                )
                    .into_response();
            }
        };

        let session = {
            let sessions = state.sessions.read().await;
            sessions.get(&session_id).cloned()
        };

        let session = match session {
            Some(s) => s,
            None => {
                return (StatusCode::NOT_FOUND, "Session not found or expired").into_response();
            }
        };

        let session_protocol_version;
        let response = {
            let mut sess = session.lock().await;
            if let Err((status, msg)) = validate_protocol_version(&headers, &sess.protocol_version)
            {
                return (status, msg).into_response();
            }
            session_protocol_version = sess.protocol_version.clone();
            sess.last_active = Instant::now();
            sess.server.handle_request(request).await
        };

        let mut resp_headers = HeaderMap::new();
        resp_headers.insert(
            "mcp-session-id",
            session_id
                .parse()
                .unwrap_or_else(|_| axum::http::HeaderValue::from_static("invalid")),
        );
        if let Ok(value) = HeaderValue::from_str(&session_protocol_version) {
            resp_headers.insert("mcp-protocol-version", value);
        }

        match response {
            Some(resp) => (StatusCode::OK, resp_headers, Json(resp)).into_response(),
            None => (StatusCode::ACCEPTED, resp_headers).into_response(),
        }
    }
}

/// `DELETE /mcp` — explicit session cleanup.
async fn delete_mcp(
    State(state): State<HttpTransportState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err((status, msg)) = validate_origin(&headers, &state.allowed_origins) {
        return (status, msg).into_response();
    }
    if let Err((status, msg)) = validate_auth(&headers, &state.auth_token) {
        return (status, msg).into_response();
    }

    let session_id = match session_id_from_headers(&headers) {
        Some(id) => id,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing or invalid Mcp-Session-Id header",
            )
                .into_response();
        }
    };

    let session = {
        let sessions = state.sessions.read().await;
        sessions.get(&session_id).cloned()
    };
    let Some(session) = session else {
        return (StatusCode::NOT_FOUND, "Session not found").into_response();
    };

    let protocol_version = {
        let sess = session.lock().await;
        sess.protocol_version.clone()
    };
    if let Err((status, msg)) = validate_protocol_version(&headers, &protocol_version) {
        return (status, msg).into_response();
    }

    let mut sessions = state.sessions.write().await;
    if sessions.remove(&session_id).is_some() {
        info!("Session {} deleted via DELETE /mcp", &session_id[..8]);
        (StatusCode::OK, "Session deleted").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Session not found").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_validation_allows_absent_and_configured_origin() {
        let allowed = vec!["http://127.0.0.1:3928".to_string()];
        let mut headers = HeaderMap::new();
        assert!(validate_origin(&headers, &allowed).is_ok());

        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://127.0.0.1:3928"),
        );
        assert!(validate_origin(&headers, &allowed).is_ok());

        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://evil.example"),
        );
        assert_eq!(
            validate_origin(&headers, &allowed).unwrap_err().0,
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn accept_validation_rejects_incompatible_clients() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            validate_accept(&headers).unwrap_err().0,
            StatusCode::NOT_ACCEPTABLE
        );

        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        assert!(validate_accept(&headers).is_ok());

        headers.insert(header::ACCEPT, HeaderValue::from_static("application/json"));
        assert_eq!(
            validate_accept(&headers).unwrap_err().0,
            StatusCode::NOT_ACCEPTABLE
        );
    }

    #[test]
    fn protocol_header_must_match_session_when_present() {
        let mut headers = HeaderMap::new();
        assert_eq!(
            validate_protocol_version(&headers, "2025-11-25")
                .unwrap_err()
                .0,
            StatusCode::BAD_REQUEST
        );

        headers.insert(
            "mcp-protocol-version",
            HeaderValue::from_static("2025-11-25"),
        );
        assert!(validate_protocol_version(&headers, "2025-11-25").is_ok());

        headers.insert(
            "mcp-protocol-version",
            HeaderValue::from_static("2024-11-05"),
        );
        assert_eq!(
            validate_protocol_version(&headers, "2025-11-25")
                .unwrap_err()
                .0,
            StatusCode::BAD_REQUEST
        );
    }
}
