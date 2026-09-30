//! Memory Web Dashboard
//!
//! Self-contained web UI at localhost:3927 for browsing, searching,
//! and managing Vestige memories. Auto-starts inside the MCP server process.
//!
//! v2.0: WebSocket real-time events, CognitiveEngine access, new API endpoints.

pub mod events;
pub mod handlers;
pub mod state;
pub mod static_files;
pub mod websocket;

#[cfg(test)]
mod http_tests;

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::{Mutex, oneshot};
use tower::ServiceBuilder;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{info, warn};

use crate::cognitive::CognitiveEngine;
use state::AppState;
use vestige_core::Storage;

/// Build the axum router with all dashboard routes
pub fn build_router(
    storage: Arc<Storage>,
    cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    port: u16,
) -> (Router, AppState) {
    let state = AppState::new(storage, cognitive);
    build_router_inner(state, port)
}

/// Build the axum router sharing an external event broadcast channel.
pub fn build_router_with_event_tx(
    storage: Arc<Storage>,
    cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    event_tx: tokio::sync::broadcast::Sender<events::VestigeEvent>,
    port: u16,
) -> (Router, AppState) {
    let state = AppState::with_event_tx(storage, cognitive, event_tx);
    build_router_inner(state, port)
}

fn build_router_inner(state: AppState, port: u16) -> (Router, AppState) {
    build_guarded_router(state, AccessGuard::new(port))
}

fn build_guarded_router(state: AppState, guard: AccessGuard) -> (Router, AppState) {
    let port = guard.port;
    let origins: Vec<axum::http::HeaderValue> = guard
        .origins
        .iter()
        .map(|origin| origin.parse().expect("valid origin"))
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ]);

    // Security: restrict WebSocket connections to localhost only (prevents cross-site WS hijacking)
    let csp_value = format!(
        "default-src 'self'; \
         script-src 'self' 'unsafe-inline'; \
         style-src 'self' 'unsafe-inline'; \
         img-src 'self' blob: data:; \
         connect-src 'self' ws://127.0.0.1:{port} ws://localhost:{port}; \
         font-src 'self' data:; \
         frame-ancestors 'none'; \
         base-uri 'self'; \
         form-action 'self';"
    );
    let csp = SetResponseHeaderLayer::overriding(
        axum::http::header::CONTENT_SECURITY_POLICY,
        axum::http::HeaderValue::from_str(&csp_value).expect("valid CSP header"),
    );

    // Additional security headers
    let x_frame_options = SetResponseHeaderLayer::overriding(
        axum::http::header::X_FRAME_OPTIONS,
        axum::http::HeaderValue::from_static("DENY"),
    );
    let x_content_type_options = SetResponseHeaderLayer::overriding(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        axum::http::HeaderValue::from_static("nosniff"),
    );
    let referrer_policy = SetResponseHeaderLayer::overriding(
        axum::http::HeaderName::from_static("referrer-policy"),
        axum::http::HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    let permissions_policy = SetResponseHeaderLayer::overriding(
        axum::http::HeaderName::from_static("permissions-policy"),
        axum::http::HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );

    let router = Router::new()
        // SvelteKit Dashboard v2.0 (embedded static build)
        .route("/dashboard", get(static_files::serve_dashboard_spa))
        .route(
            "/dashboard/{*path}",
            get(static_files::serve_dashboard_asset),
        )
        // Legacy embedded HTML (keep for backward compat)
        .route("/", get(handlers::serve_dashboard))
        .route("/graph", get(handlers::serve_graph))
        // WebSocket for real-time events
        .route("/ws", get(websocket::ws_handler))
        // Memory CRUD
        .route("/api/memories", get(handlers::list_memories))
        .route("/api/memories/{id}", get(handlers::get_memory))
        .route("/api/memories/{id}", delete(handlers::delete_memory))
        .route("/api/memories/{id}/promote", post(handlers::promote_memory))
        .route("/api/memories/{id}/demote", post(handlers::demote_memory))
        // v2.0.7: active-forgetting HTTP surface. `suppress` was MCP-only
        // since v2.0.5 despite having full graph event handlers; this closes
        // the gap so dashboard users can trigger inhibition without dropping
        // to the MCP layer.
        .route(
            "/api/memories/{id}/suppress",
            post(handlers::suppress_memory),
        )
        .route(
            "/api/memories/{id}/unsuppress",
            post(handlers::unsuppress_memory),
        )
        // Search
        .route("/api/search", get(handlers::search_memories))
        // Stats & health
        .route("/api/stats", get(handlers::get_stats))
        .route("/api/health", get(handlers::health_check))
        // Timeline
        .route("/api/timeline", get(handlers::get_timeline))
        .route("/api/changelog", get(handlers::get_changelog))
        // Graph
        .route("/api/graph", get(handlers::get_graph))
        // Cognitive operations (v2.0)
        .route("/api/dream", post(handlers::trigger_dream))
        .route("/api/explore", post(handlers::explore_connections))
        .route("/api/predict", post(handlers::predict_memories))
        .route("/api/importance", post(handlers::score_importance))
        .route("/api/consolidate", post(handlers::trigger_consolidation))
        .route(
            "/api/retention-distribution",
            get(handlers::retention_distribution),
        )
        // Intentions (v2.0)
        .route("/api/intentions", get(handlers::list_intentions))
        // Reasoning Theater (v2.0.8) — 8-stage cognitive pipeline surface.
        // Wraps crate::tools::cross_reference::execute. Emits
        // DeepReferenceCompleted so Graph3D can glide, pulse, and arc.
        .route("/api/deep_reference", post(handlers::deep_reference_query))
        // Retroactive Salience Backfill — an explicit preview/promote surface
        // whose output is always persisted as a receipt before it reaches 3D.
        .route("/api/backfill", post(handlers::backfill_query))
        // Sanhedrin receipts: latest local hook verdict + appeal training.
        .route("/api/sanhedrin/latest", get(handlers::get_sanhedrin_latest))
        .route(
            "/api/sanhedrin/telemetry",
            get(handlers::get_sanhedrin_telemetry),
        )
        .route("/api/sanhedrin/appeal", post(handlers::appeal_sanhedrin))
        // ============================================================
        // AGENT BLACK BOX (v2.2) — replayable agent-run traces
        // ============================================================
        .route("/api/traces", get(handlers::list_traces))
        .route("/api/traces/{run_id}", get(handlers::get_trace))
        .route("/api/traces/{run_id}/export", get(handlers::export_trace))
        // ============================================================
        // MEMORY RECEIPTS (v2.2) — the nutrition label for a retrieval
        // ============================================================
        .route("/api/receipts", get(handlers::list_receipts))
        .route("/api/receipts/{receipt_id}", get(handlers::get_receipt))
        // ============================================================
        // MEMORY HYGIENE + INTELLIGENCE (live-wire) — duplicates,
        // contradictions, cross-project patterns, per-memory audit
        // ============================================================
        .route("/api/duplicates", get(handlers::list_duplicates))
        .route(
            "/api/duplicates/plan",
            post(handlers::plan_duplicates_merge),
        )
        .route(
            "/api/duplicates/apply",
            post(handlers::apply_duplicates_merge),
        )
        .route("/api/contradictions", get(handlers::list_contradictions))
        .route(
            "/api/patterns/cross-project",
            get(handlers::get_cross_project_patterns),
        )
        .route("/api/memories/{id}/audit", get(handlers::get_memory_audit))
        // ============================================================
        // MEMORY PRs (v2.2) — risk-gated brain-change review queue
        // ============================================================
        .route("/api/memory-prs", get(handlers::list_memory_prs))
        // Static `/mode` routes declared BEFORE the dynamic `/{id}` route (B7
        // hygiene). axum 0.8/matchit already prioritizes static segments, but
        // declaring them first makes the intent unambiguous and guards against
        // a future router that doesn't.
        .route("/api/memory-prs/mode", get(handlers::get_review_mode))
        .route("/api/memory-prs/mode", post(handlers::set_review_mode))
        .route("/api/memory-prs/{id}", get(handlers::get_memory_pr))
        .route(
            "/api/memory-prs/{id}/{action}",
            post(handlers::act_on_memory_pr),
        )
        .layer(
            ServiceBuilder::new()
                .concurrency_limit(50)
                .layer(csp)
                .layer(x_frame_options)
                .layer(x_content_type_options)
                .layer(referrer_policy)
                .layer(permissions_policy)
                // Before CORS and every handler: CORS only decides whether a
                // page may read a response, never whether the request runs.
                .layer(axum::middleware::from_fn_with_state(guard, guard_access))
                .layer(cors),
        )
        .with_state(state.clone());

    (router, state)
}

// ============================================================================
// ACCESS GUARD
// ============================================================================

/// Who may talk to the dashboard.
///
/// The listener binds 127.0.0.1 only, yet any web page open in the user's
/// browser can still send it requests, and a DNS-rebound hostname can read
/// the answers. CORS does not stop either: it only decides whether a page may
/// read a response, and a POST with no body or a form body runs without
/// asking. So every request must:
///
/// - name this dashboard in `Host` (127.0.0.1 or localhost, on its port),
///   which a rebound hostname cannot;
/// - carry no `Origin` but the dashboard's own;
/// - not be a load from another site (`Sec-Fetch-Site`), except a top-level
///   navigation to a page;
/// - when it changes anything (any method but GET, HEAD and OPTIONS), come
///   from the dashboard's own page (browsers always send `Origin` on those)
///   or carry `Authorization: Bearer <token>`, the token the HTTP MCP
///   transport takes (`VESTIGE_AUTH_TOKEN`, else the `auth_token` file).
#[derive(Clone)]
struct AccessGuard {
    port: u16,
    /// Accepted `Host` values, lowercase.
    hosts: Arc<[String]>,
    /// Accepted `Origin` values, lowercase.
    origins: Arc<[String]>,
    /// The bearer token for callers that are not the dashboard's page.
    /// Read (never created) the first time a request presents one, so a
    /// request without `Authorization` never touches the token file.
    token: Arc<OnceLock<Option<String>>>,
}

impl AccessGuard {
    fn new(port: u16) -> Self {
        let mut names = vec![("127.0.0.1", port), ("localhost", port)];
        // SvelteKit dev server (`vite dev` proxies /api and /ws here) — only
        // in debug builds, like the CORS list before it.
        if cfg!(debug_assertions) {
            names.extend([("127.0.0.1", 5173), ("localhost", 5173)]);
        }
        let mut hosts: Vec<String> = names
            .iter()
            .map(|(name, port)| format!("{name}:{port}"))
            .collect();
        if port == 80 {
            // A browser leaves the default port out of `Host`.
            hosts.extend(["127.0.0.1".to_string(), "localhost".to_string()]);
        }
        let origins = names
            .iter()
            .map(|(name, port)| format!("http://{name}:{port}"))
            .collect::<Vec<_>>();
        Self {
            port,
            hosts: hosts.into(),
            origins: origins.into(),
            token: Arc::new(OnceLock::new()),
        }
    }

    #[cfg(test)]
    fn with_token(port: u16, token: &str) -> Self {
        let guard = Self::new(port);
        let _ = guard.token.set(Some(token.to_string()));
        guard
    }

    fn token(&self) -> Option<&str> {
        self.token
            .get_or_init(crate::protocol::auth::read_auth_token)
            .as_deref()
    }

    fn check(&self, method: &Method, headers: &HeaderMap) -> Result<(), Refusal> {
        let host = headers
            .get(header::HOST)
            .and_then(|value| value.to_str().ok())
            .map(str::to_ascii_lowercase);
        if !host.is_some_and(|host| self.hosts.contains(&host)) {
            return Err(Refusal::forbidden(
                "forbidden_host",
                format!(
                    "this dashboard answers only to 127.0.0.1:{port} and localhost:{port}",
                    port = self.port
                ),
            ));
        }

        let origin = match headers.get(header::ORIGIN) {
            None => None,
            Some(value) => {
                let origin = value.to_str().map(str::to_ascii_lowercase);
                if !origin
                    .as_ref()
                    .is_ok_and(|origin| self.origins.contains(origin))
                {
                    return Err(Refusal::forbidden(
                        "forbidden_origin",
                        "only the dashboard's own page may call this API".to_string(),
                    ));
                }
                origin.ok()
            }
        };

        let safe = matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
        let fetch_site = headers
            .get("sec-fetch-site")
            .and_then(|value| value.to_str().ok());
        if matches!(fetch_site, Some("cross-site" | "same-site")) && origin.is_none() {
            let navigation = safe
                && headers
                    .get("sec-fetch-mode")
                    .and_then(|value| value.to_str().ok())
                    == Some("navigate");
            if !navigation {
                return Err(Refusal::forbidden(
                    "forbidden_cross_site",
                    "a page on another site cannot load this dashboard's data".to_string(),
                ));
            }
        }

        if safe || origin.is_some() {
            return Ok(());
        }
        // No Origin on a write: not the dashboard's page (a browser always
        // sends one), so a script, which needs the token.
        const HOW: &str = "a request that changes memory must come from the dashboard's own page or carry Authorization: Bearer <token> (VESTIGE_AUTH_TOKEN, else the auth_token file in the default Vestige data directory)";
        if !headers.contains_key(header::AUTHORIZATION) {
            return Err(Refusal {
                status: StatusCode::UNAUTHORIZED,
                code: "auth_required",
                message: HOW.to_string(),
            });
        }
        let Some(expected) = self.token() else {
            return Err(Refusal {
                status: StatusCode::UNAUTHORIZED,
                code: "auth_required",
                message: format!("no token is configured: {HOW}"),
            });
        };
        crate::protocol::http::validate_auth(headers, expected).map_err(|(status, why)| Refusal {
            status,
            code: if status == StatusCode::UNAUTHORIZED {
                "auth_required"
            } else {
                "auth_invalid"
            },
            message: format!("{why}: {HOW}"),
        })
    }
}

/// Why the guard turned a request away, as the JSON `{error, code}` body the
/// dashboard's fetcher shows.
struct Refusal {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl Refusal {
    fn forbidden(code: &'static str, message: String) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code,
            message,
        }
    }
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": format!("{}: {}", self.code, self.message),
            "code": self.code,
        });
        (self.status, axum::Json(body)).into_response()
    }
}

async fn guard_access(State(guard): State<AccessGuard>, request: Request, next: Next) -> Response {
    match guard.check(request.method(), request.headers()) {
        Ok(()) => next.run(request).await,
        Err(refusal) => {
            // debug, not warn: a hostile page can send these in a loop.
            tracing::debug!(
                method = %request.method(),
                path = request.uri().path(),
                code = refusal.code,
                "dashboard refused a request"
            );
            refusal.into_response()
        }
    }
}

/// Start the dashboard HTTP server (blocking — use in CLI mode)
pub async fn start_dashboard(
    storage: Arc<Storage>,
    cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    port: u16,
    open_browser: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (app, _state) = build_router(storage, cognitive, port);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    info!("Dashboard starting at http://127.0.0.1:{}", port);

    if open_browser {
        let url = format!("http://127.0.0.1:{}", port);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let _ = open::that(&url);
        });
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the dashboard as a background task (non-blocking — use in MCP server)
pub async fn start_background(
    storage: Arc<Storage>,
    cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    port: u16,
) -> Result<AppState, Box<dyn std::error::Error>> {
    let (app, state) = build_router(storage, cognitive, port);
    start_background_inner(app, state, port).await
}

/// This process's dashboard, started on demand: by `VESTIGE_DASHBOARD_ENABLED`
/// at startup, by `vestige dashboard` or `vestige serve --dashboard` in this
/// process, or by a `vestige dashboard` run elsewhere that asks through the
/// attach endpoint.
///
/// A dashboard this process asked for itself ([`DashboardOnDemand::ensure`])
/// serves until the process exits. One that only `vestige dashboard` runs
/// asked for ([`DashboardOnDemand::lease`]) serves while one of them still
/// runs: when the last one exits, the listener closes. An agent's server can
/// run for days, and the dashboard must not outlive the command that the
/// user believes they closed.
#[derive(Clone)]
pub struct DashboardOnDemand {
    running: Arc<Mutex<Option<Serving>>>,
    generations: Arc<AtomicU64>,
    storage: Arc<Storage>,
    cognitive: Arc<Mutex<CognitiveEngine>>,
    event_tx: tokio::sync::broadcast::Sender<events::VestigeEvent>,
}

/// The dashboard this process serves right now.
struct Serving {
    port: u16,
    state: AppState,
    /// Resolves once the listener has closed after `state.stop()`.
    closed: oneshot::Receiver<()>,
    /// This process asked for it itself: served until the process exits.
    pinned: bool,
    /// `vestige dashboard` runs holding it open.
    leases: usize,
    /// Tells a late lease release from one that belongs to this server.
    generation: u64,
}

impl DashboardOnDemand {
    pub fn new(
        storage: Arc<Storage>,
        cognitive: Arc<Mutex<CognitiveEngine>>,
        event_tx: tokio::sync::broadcast::Sender<events::VestigeEvent>,
    ) -> Self {
        Self {
            running: Arc::new(Mutex::new(None)),
            generations: Arc::new(AtomicU64::new(0)),
            storage,
            cognitive,
            event_tx,
        }
    }

    /// Serve the dashboard on `port` until this process exits. When it
    /// already runs, its port (which may differ from `port`).
    pub async fn ensure(&self, port: u16) -> Result<u16, String> {
        let mut running = self.running.lock().await;
        if let Some(serving) = running.as_mut() {
            serving.pinned = true;
            return Ok(serving.port);
        }
        let serving = self.start(port, true).await?;
        let port = serving.port;
        *running = Some(serving);
        Ok(port)
    }

    /// Serve the dashboard for as long as the returned lease lives. When it
    /// already runs, the lease names its port, which may differ from `port`.
    pub async fn lease(&self, port: u16) -> Result<DashboardLease, String> {
        let mut running = self.running.lock().await;
        if running.is_none() {
            *running = Some(self.start(port, false).await?);
        }
        let serving = running.as_mut().expect("started above");
        serving.leases += 1;
        Ok(DashboardLease {
            owner: self.clone(),
            generation: serving.generation,
            port: serving.port,
            released: false,
        })
    }

    async fn release(&self, generation: u64) {
        let mut running = self.running.lock().await;
        let Some(serving) = running.as_mut() else {
            return;
        };
        if serving.generation != generation {
            return;
        }
        serving.leases = serving.leases.saturating_sub(1);
        if serving.leases > 0 || serving.pinned {
            return;
        }
        let Some(serving) = running.take() else {
            return;
        };
        serving.state.stop();
        // Hold the lock until the port is free, so a lease that arrives now
        // binds it again instead of finding it still taken.
        if tokio::time::timeout(Duration::from_secs(5), serving.closed)
            .await
            .is_err()
        {
            warn!(
                port = serving.port,
                "the dashboard listener did not close within 5s"
            );
        }
        info!(
            port = serving.port,
            "dashboard stopped: the last `vestige dashboard` using it exited"
        );
    }

    async fn start(&self, port: u16, pinned: bool) -> Result<Serving, String> {
        let (app, state) = build_router_with_event_tx(
            Arc::clone(&self.storage),
            Some(Arc::clone(&self.cognitive)),
            self.event_tx.clone(),
            port,
        );
        let closed = serve_until_stopped(app, state.clone(), port)
            .await
            .map_err(|err| format!("the dashboard could not bind 127.0.0.1:{port}: {err}"))?;
        Ok(Serving {
            port,
            state,
            closed,
            pinned,
            leases: 0,
            generation: self.generations.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// For the attach endpoint: start (or find) the dashboard for one
    /// `vestige dashboard` run, as a URL plus the lease that keeps it serving.
    pub fn starter(&self) -> crate::attach::DashboardStarter {
        let this = self.clone();
        Arc::new(move |port| {
            let this = this.clone();
            Box::pin(async move {
                let lease = this.lease(port).await?;
                Ok(crate::attach::DashboardGrant {
                    url: format!("http://127.0.0.1:{}", lease.port),
                    hold: Box::new(lease),
                })
            })
        })
    }
}

/// Keeps a dashboard that [`DashboardOnDemand::lease`] started serving.
/// Dropping it (or [`DashboardLease::release`]) gives the claim back; after
/// the last one the dashboard stops, unless this process pinned it.
pub struct DashboardLease {
    owner: DashboardOnDemand,
    generation: u64,
    /// Where the dashboard answers.
    pub port: u16,
    released: bool,
}

impl DashboardLease {
    /// Give the claim back and wait until the dashboard has stopped, when
    /// this was the last one.
    pub async fn release(mut self) {
        self.released = true;
        self.owner.release(self.generation).await;
    }
}

impl Drop for DashboardLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let owner = self.owner.clone();
        let generation = self.generation;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move { owner.release(generation).await });
        }
    }
}

/// Start the dashboard sharing an external event broadcast channel.
pub async fn start_background_with_event_tx(
    storage: Arc<Storage>,
    cognitive: Option<Arc<Mutex<CognitiveEngine>>>,
    event_tx: tokio::sync::broadcast::Sender<events::VestigeEvent>,
    port: u16,
) -> Result<AppState, Box<dyn std::error::Error>> {
    let (app, state) = build_router_with_event_tx(storage, cognitive, event_tx, port);
    start_background_inner(app, state, port).await
}

async fn start_background_inner(
    app: Router,
    state: AppState,
    port: u16,
) -> Result<AppState, Box<dyn std::error::Error>> {
    serve_until_stopped(app, state.clone(), port)
        .await
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    Ok(state)
}

/// Bind 127.0.0.1:`port` and serve `app` in the background until
/// `state.stop()`. The receiver resolves once the listener has closed.
async fn serve_until_stopped(
    app: Router,
    state: AppState,
    port: u16,
) -> std::io::Result<oneshot::Receiver<()>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            warn!(
                "Dashboard could not bind to port {}: {} (MCP server continues without dashboard)",
                port, e
            );
            return Err(e);
        }
    };

    info!(
        "Dashboard available at http://127.0.0.1:{} (WebSocket at ws://127.0.0.1:{}/ws)",
        port, port
    );

    let (closed_tx, closed_rx) = oneshot::channel();
    let listener = ClosingListener {
        inner: Some(listener),
        closed: Some(closed_tx),
    };
    tokio::spawn(async move {
        let stop = state.clone();
        let served = axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop.stopped().await })
            .await;
        if let Err(e) = served {
            warn!("Dashboard server error: {}", e);
        }
        drop(state);
    });

    Ok(closed_rx)
}

/// A TCP listener that reports when it has closed, so a stopped dashboard's
/// port is known to be free before anything binds it again.
struct ClosingListener {
    inner: Option<tokio::net::TcpListener>,
    closed: Option<oneshot::Sender<()>>,
}

impl ClosingListener {
    fn listener(&mut self) -> &mut tokio::net::TcpListener {
        self.inner
            .as_mut()
            .unwrap_or_else(|| unreachable!("only Drop takes the listener"))
    }
}

impl axum::serve::Listener for ClosingListener {
    type Io = tokio::net::TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        axum::serve::Listener::accept(self.listener()).await
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        match &self.inner {
            Some(listener) => listener.local_addr(),
            None => Err(std::io::Error::other("the dashboard listener is closed")),
        }
    }
}

impl Drop for ClosingListener {
    fn drop(&mut self) {
        // Close the socket first, then report it closed.
        drop(self.inner.take());
        if let Some(closed) = self.closed.take() {
            let _ = closed.send(());
        }
    }
}
