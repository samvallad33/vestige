//! The dashboard's HTTP surface, driven through the real router over a
//! Strata log (the store every 4.0 install runs on).
//!
//! These tests send requests the way a browser or a local script does,
//! including the Host, Origin and Sec-Fetch headers a hostile page would
//! send, and check what comes back.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt;
use vestige_core::Storage;
use vestige_core::memory::IngestInput;

const PORT: u16 = 47512;
const HOST: &str = "127.0.0.1:47512";
const OWN_ORIGIN: &str = "http://127.0.0.1:47512";

fn strata() -> (tempfile::TempDir, Arc<Storage>) {
    let dir = tempfile::tempdir().unwrap();
    let storage = crate::strata_memory::open(dir.path()).unwrap();
    (dir, storage)
}

fn remember(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> String {
    storage
        .ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            ..Default::default()
        })
        .unwrap()
        .id
}

fn link(storage: &Arc<Storage>, source: &str, target: &str) {
    let now = chrono::Utc::now();
    storage
        .save_connection(&vestige_core::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: "derived_from".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 1,
        })
        .unwrap();
}

fn router(storage: &Arc<Storage>) -> Router {
    super::build_router(Arc::clone(storage), None, PORT).0
}

fn router_with_engine(storage: &Arc<Storage>) -> Router {
    let cognitive = Arc::new(tokio::sync::Mutex::new(
        crate::cognitive::CognitiveEngine::new(),
    ));
    super::build_router(Arc::clone(storage), Some(cognitive), PORT).0
}

struct Reply {
    status: StatusCode,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|err| panic!("not JSON ({err}): {:?}", self.body))
    }
}

async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> Reply {
    let mut request = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let request = request
        .body(body.map_or_else(Body::empty, |text| Body::from(text.to_string())))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    Reply {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
    }
}

/// A same-origin request from the dashboard's own page.
async fn own(app: &Router, method: &str, uri: &str, body: Option<&str>) -> Reply {
    let mut headers = vec![("host", HOST), ("origin", OWN_ORIGIN)];
    if body.is_some() {
        headers.push(("content-type", "application/json"));
    }
    send(app, method, uri, &headers, body).await
}

fn visible(storage: &Arc<Storage>, id: &str) -> bool {
    storage.get_node(id).unwrap().is_some()
}

// ---------------------------------------------------------------------------
// B1: a third-party page or a rebound hostname must not reach the store.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_cross_site_post_cannot_suppress_a_memory() {
    let (_dir, storage) = strata();
    let _first = remember(&storage, "the first memory", &[]);
    let target = remember(&storage, "a memory an attacker wants gone", &[]);
    let app = router(&storage);

    // No body: a CORS "simple request" that a browser sends without asking.
    let reply = send(
        &app,
        "POST",
        &format!("/api/memories/{target}/suppress"),
        &[("host", HOST), ("origin", "https://attacker.example")],
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::FORBIDDEN, "{}", reply.body);
    assert!(visible(&storage, &target), "a cross-site POST retired it");
}

#[tokio::test]
async fn a_cross_site_form_post_cannot_demote() {
    let (_dir, storage) = strata();
    let target = remember(&storage, "demote me from a form", &[]);
    let before = storage.get_node(&target).unwrap().unwrap().reps;
    let app = router(&storage);

    let reply = send(
        &app,
        "POST",
        &format!("/api/memories/{target}/demote"),
        &[
            ("host", HOST),
            ("origin", "http://evil.example"),
            ("content-type", "application/x-www-form-urlencoded"),
        ],
        Some("a=b"),
    )
    .await;
    assert_eq!(reply.status, StatusCode::FORBIDDEN, "{}", reply.body);
    assert_eq!(storage.get_node(&target).unwrap().unwrap().reps, before);
}

#[tokio::test]
async fn cross_site_dream_and_consolidate_are_refused() {
    let (_dir, storage) = strata();
    remember(&storage, "one", &[]);
    let app = router_with_engine(&storage);
    for path in ["/api/dream", "/api/consolidate"] {
        let reply = send(
            &app,
            "POST",
            path,
            &[("host", HOST), ("origin", "http://evil.example")],
            None,
        )
        .await;
        assert_eq!(
            reply.status,
            StatusCode::FORBIDDEN,
            "{path}: {}",
            reply.body
        );
    }
}

#[tokio::test]
async fn a_rebound_hostname_reads_nothing() {
    let (_dir, storage) = strata();
    remember(&storage, "private note: rebinding must not read this", &[]);
    let app = router(&storage);

    for host in [
        "rebind.attacker.example:47512",
        "rebind.attacker.example",
        "127.0.0.1:1",
        "localhost.attacker.example:47512",
    ] {
        let reply = send(&app, "GET", "/api/memories", &[("host", host)], None).await;
        assert_eq!(
            reply.status,
            StatusCode::FORBIDDEN,
            "Host {host}: {}",
            reply.body
        );
        assert!(
            !reply.body.contains("private note"),
            "Host {host} read content"
        );
    }
    for host in [HOST, "localhost:47512", "LOCALHOST:47512"] {
        let reply = send(&app, "GET", "/api/memories", &[("host", host)], None).await;
        assert_eq!(reply.status, StatusCode::OK, "Host {host}: {}", reply.body);
    }
}

#[tokio::test]
async fn a_write_with_no_origin_needs_the_bearer_token() {
    let (_dir, storage) = strata();
    let target = remember(&storage, "promote me from a script", &[]);
    let before = storage.get_node(&target).unwrap().unwrap().reps;
    let app = router(&storage);

    let reply = send(
        &app,
        "POST",
        &format!("/api/memories/{target}/promote"),
        &[("host", HOST)],
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::UNAUTHORIZED, "{}", reply.body);
    assert_eq!(storage.get_node(&target).unwrap().unwrap().reps, before);
}

#[tokio::test]
async fn a_cross_site_subresource_cannot_read_the_api() {
    let (_dir, storage) = strata();
    remember(&storage, "loaded by an img tag", &[]);
    let app = router(&storage);

    let reply = send(
        &app,
        "GET",
        "/api/memories",
        &[
            ("host", HOST),
            ("sec-fetch-site", "cross-site"),
            ("sec-fetch-mode", "no-cors"),
        ],
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::FORBIDDEN, "{}", reply.body);

    // A link from another site still opens the dashboard itself.
    let reply = send(
        &app,
        "GET",
        "/dashboard",
        &[
            ("host", HOST),
            ("sec-fetch-site", "cross-site"),
            ("sec-fetch-mode", "navigate"),
        ],
        None,
    )
    .await;
    assert_ne!(reply.status, StatusCode::FORBIDDEN, "{}", reply.body);
}

#[tokio::test]
async fn the_dashboards_own_origin_still_writes() {
    let (_dir, storage) = strata();
    let target = remember(&storage, "promote me from the dashboard", &[]);
    let before = storage.get_node(&target).unwrap().unwrap().reps;
    let app = router(&storage);

    for origin in [OWN_ORIGIN, "http://localhost:47512"] {
        let reply = send(
            &app,
            "POST",
            &format!("/api/memories/{target}/promote"),
            &[("host", HOST), ("origin", origin)],
            None,
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{origin}: {}", reply.body);
    }
    assert_eq!(storage.get_node(&target).unwrap().unwrap().reps, before + 2);
}

// ---------------------------------------------------------------------------
// C22: /api/graph?sort=connected on a Strata log.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn graph_sorted_by_connections_centers_on_the_hub() {
    let (_dir, storage) = strata();
    let hub = remember(&storage, "hub memory", &[]);
    let a = remember(&storage, "spoke a", &[]);
    let b = remember(&storage, "spoke b", &[]);
    let _newest = remember(&storage, "newest, isolated", &[]);
    link(&storage, &a, &hub);
    link(&storage, &b, &hub);
    let app = router(&storage);

    let reply = own(
        &app,
        "GET",
        "/api/graph?max_nodes=200&depth=3&sort=connected",
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let graph = reply.json();
    assert_eq!(graph["center_id"], hub, "{graph}");
    assert_eq!(graph["nodeCount"], 3, "{graph}");
    assert_eq!(graph["edgeCount"], 2, "{graph}");

    // The default (recent) view of a lonely newest memory falls back to the
    // densest cluster instead of one orb.
    let reply = own(&app, "GET", "/api/graph?max_nodes=200&depth=3", None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.json()["center_id"], hub);
}

#[tokio::test]
async fn graph_sorted_by_connections_falls_back_to_the_newest_memory() {
    let (_dir, storage) = strata();
    remember(&storage, "older", &[]);
    let newest = remember(&storage, "newest", &[]);
    let app = router(&storage);

    let reply = own(&app, "GET", "/api/graph?sort=connected", None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.json()["center_id"], newest);
}

// ---------------------------------------------------------------------------
// C23: the Memories search resolves exact handles across the whole store.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memory_search_finds_an_exact_id_or_tag_beyond_the_newest_page() {
    let (_dir, storage) = strata();
    let oldest = remember(&storage, "the oldest memory, tagged", &["ethena"]);
    for i in 0..45 {
        remember(&storage, &format!("filler memory {i}"), &[]);
    }
    let app = router(&storage);

    let reply = own(&app, "GET", &format!("/api/memories?handle={oldest}"), None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let found = reply.json();
    assert_eq!(found["memories"][0]["id"], oldest, "{found}");
    assert_eq!(found["total"], 1, "{found}");
    assert_eq!(found["resolution"]["kind"], "memory", "{found}");

    let reply = own(&app, "GET", "/api/memories?handle=ethena", None).await;
    let found = reply.json();
    assert_eq!(found["memories"][0]["id"], oldest, "{found}");
    assert_eq!(found["resolution"]["kind"], "tag", "{found}");
}

#[tokio::test]
async fn memory_search_never_matches_free_text() {
    let (_dir, storage) = strata();
    remember(&storage, "refund policy for the ethena project", &[]);
    let app = router(&storage);

    let reply = own(&app, "GET", "/api/memories?handle=refund%20policy", None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let found = reply.json();
    assert_eq!(found["total"], 0, "{found}");
    assert_eq!(found["memories"].as_array().unwrap().len(), 0, "{found}");
    assert!(
        found["resolution"]["handleRequired"].is_string(),
        "an unresolved search must say it needs a handle: {found}"
    );
}

// ---------------------------------------------------------------------------
// C24: totals cover the whole store, not the page that was returned.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memory_list_total_counts_the_whole_store() {
    let (_dir, storage) = strata();
    for i in 0..45 {
        remember(&storage, &format!("memory {i}"), &[]);
    }
    let app = router(&storage);

    let reply = own(&app, "GET", "/api/memories?limit=10", None).await;
    let page = reply.json();
    assert_eq!(page["memories"].as_array().unwrap().len(), 10, "{page}");
    assert_eq!(page["total"], 45, "{page}");
}

#[tokio::test]
async fn timeline_reports_the_window_total_and_rewrite_times() {
    let (_dir, storage) = strata();
    for i in 0..12 {
        remember(&storage, &format!("timeline memory {i}"), &[]);
    }
    let app = router(&storage);

    let reply = own(&app, "GET", "/api/timeline?days=365&limit=5", None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let timeline = reply.json();
    assert_eq!(
        timeline["days"], 365,
        "the 365-day view was cut: {timeline}"
    );
    assert_eq!(timeline["totalMemories"], 12, "{timeline}");
    assert_eq!(timeline["returned"], 5, "{timeline}");
    assert_eq!(timeline["truncated"], true, "{timeline}");
    let first = &timeline["timeline"][0]["memories"][0];
    assert!(first["updatedAt"].is_string(), "{first}");
}

#[tokio::test]
async fn retention_distribution_measures_every_memory() {
    let (_dir, storage) = strata();
    // One past the 1,000-row sample the endpoint used to take.
    for i in 0..1001 {
        remember(&storage, &format!("retention memory {i}"), &[]);
    }
    let app = router(&storage);

    let reply = own(&app, "GET", "/api/retention-distribution", None).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let distribution = reply.json();
    assert_eq!(distribution["total"], 1001, "{distribution}");
    let bucketed: u64 = distribution["distribution"]
        .as_array()
        .unwrap()
        .iter()
        .map(|bucket| bucket["count"].as_u64().unwrap())
        .sum();
    assert_eq!(bucketed, 1001);
}

// ---------------------------------------------------------------------------
// C26: a feature 4.0 withholds answers with the reason, never a bare 500.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn withheld_features_answer_with_a_structured_reason() {
    let (_dir, storage) = strata();
    let id = remember(&storage, "a memory with no recorded audit", &[]);
    let app = router_with_engine(&storage);

    let explore = format!(r#"{{"action":"associations","from_id":"{id}"}}"#);
    let calls: Vec<(&str, String, Option<&str>)> = vec![
        ("GET", "/api/changelog?limit=100".into(), None),
        ("GET", "/api/traces".into(), None),
        ("GET", "/api/receipts".into(), None),
        ("GET", "/api/search?q=x".into(), None),
        ("GET", "/api/memories?q=x".into(), None),
        ("GET", format!("/api/memories/{id}/audit"), None),
        (
            "POST",
            "/api/deep_reference".into(),
            Some(r#"{"query":"x"}"#),
        ),
        ("POST", "/api/backfill".into(), Some("{}")),
        ("POST", "/api/explore".into(), Some(explore.as_str())),
        ("POST", format!("/api/memories/{id}/unsuppress"), None),
    ];
    for (method, uri, body) in calls {
        let reply = own(&app, method, &uri, body).await;
        assert_eq!(
            reply.status,
            StatusCode::NOT_IMPLEMENTED,
            "{method} {uri}: {}",
            reply.body
        );
        let error = reply.json()["error"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(
            [
                "unavailable_in_4_0",
                "similarity_disabled",
                "pending_strata"
            ]
            .iter()
            .any(|code| error.starts_with(code)),
            "{method} {uri}: {error:?}"
        );
    }
}

/// The Stats page's "Consolidate memory" button. On a Strata log every phase
/// is a no-op, so an all-zero reply would show as a completed pass. It is
/// refused with the reason, and no consolidation is announced.
#[tokio::test]
async fn consolidate_on_a_strata_log_is_refused_not_reported_as_a_zero_pass() {
    let (_dir, storage) = strata();
    remember(&storage, "a memory consolidation would not touch", &[]);
    let (app, state) = super::build_router(Arc::clone(&storage), None, PORT);
    let mut events = state.subscribe();

    let reply = own(&app, "POST", "/api/consolidate", None).await;
    assert_eq!(reply.status, StatusCode::NOT_IMPLEMENTED, "{}", reply.body);
    let answer = reply.json();
    assert_eq!(answer["code"], "unavailable_in_4_0", "{answer}");
    let error = answer["error"].as_str().unwrap_or_default();
    assert!(error.starts_with("unavailable_in_4_0"), "{error}");
    assert!(error.contains("no-op on a Strata log"), "{error}");
    assert!(answer.get("nodesProcessed").is_none(), "{answer}");
    while let Ok(event) = events.try_recv() {
        assert!(
            !matches!(
                event,
                super::events::VestigeEvent::ConsolidationStarted { .. }
                    | super::events::VestigeEvent::ConsolidationCompleted { .. }
            ),
            "a refused consolidation was announced"
        );
    }
}

#[tokio::test]
async fn a_script_with_the_bearer_token_can_write() {
    let (_dir, storage) = strata();
    let target = remember(&storage, "promote me with the token", &[]);
    let before = storage.get_node(&target).unwrap().unwrap().reps;
    let token = "test-token-0123456789abcdef0123456789abcdef";
    let state = super::state::AppState::new(Arc::clone(&storage), None);
    let (app, _) = super::build_guarded_router(state, super::AccessGuard::with_token(PORT, token));
    let uri = format!("/api/memories/{target}/promote");

    let wrong = send(
        &app,
        "POST",
        &uri,
        &[("host", HOST), ("authorization", "Bearer not-the-token")],
        None,
    )
    .await;
    assert_eq!(wrong.status, StatusCode::FORBIDDEN, "{}", wrong.body);
    assert_eq!(wrong.json()["code"], "auth_invalid");

    for scheme in ["Bearer", "bearer"] {
        let bearer = format!("{scheme} {token}");
        let reply = send(
            &app,
            "POST",
            &uri,
            &[("host", HOST), ("authorization", bearer.as_str())],
            None,
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{scheme}: {}", reply.body);
    }
    assert_eq!(storage.get_node(&target).unwrap().unwrap().reps, before + 2);

    // The token does not excuse a foreign page: a browser sends its Origin.
    let bearer_token = format!("Bearer {token}");
    let reply = send(
        &app,
        "POST",
        &uri,
        &[
            ("host", HOST),
            ("origin", "https://attacker.example"),
            ("authorization", bearer_token.as_str()),
        ],
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::FORBIDDEN, "{}", reply.body);
}

#[tokio::test]
async fn a_strata_suppression_does_not_claim_it_can_be_undone() {
    let (_dir, storage) = strata();
    let target = remember(&storage, "suppressed from the dashboard", &[]);
    let app = router(&storage);

    let reply = own(
        &app,
        "POST",
        &format!("/api/memories/{target}/suppress"),
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    let answer = reply.json();
    assert_eq!(answer["suppressed"], true, "{answer}");
    assert_eq!(answer["reversible"], false, "{answer}");
    assert!(answer["reversibleUntil"].is_null(), "{answer}");
    assert!(!visible(&storage, &target));

    let reply = own(
        &app,
        "POST",
        &format!("/api/memories/{target}/unsuppress"),
        None,
    )
    .await;
    assert_eq!(reply.status, StatusCode::NOT_IMPLEMENTED, "{}", reply.body);
}

// ---------------------------------------------------------------------------
// B29: a dashboard only `vestige dashboard` runs asked for stops with the last.
// ---------------------------------------------------------------------------

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn answers(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
}

fn on_demand(storage: &Arc<Storage>) -> super::DashboardOnDemand {
    let cognitive = Arc::new(tokio::sync::Mutex::new(
        crate::cognitive::CognitiveEngine::new(),
    ));
    let (event_tx, _) = tokio::sync::broadcast::channel(16);
    super::DashboardOnDemand::new(Arc::clone(storage), cognitive, event_tx)
}

#[tokio::test]
async fn a_leased_dashboard_stops_with_its_last_lease() {
    let (_dir, storage) = strata();
    let dashboard = on_demand(&storage);
    let port = free_port();

    let first = dashboard.lease(port).await.unwrap();
    let second = dashboard.lease(free_port()).await.unwrap();
    assert_eq!(first.port, port);
    assert_eq!(second.port, port, "a second lease shares the running port");
    assert!(answers(port).await);

    second.release().await;
    assert!(answers(port).await, "one lease still holds it");
    first.release().await;
    assert!(!answers(port).await, "the listener outlived its last lease");

    // Dropping a lease (the attach connection closing) releases it too, and
    // the port serves again on the next request.
    let again = dashboard.lease(port).await.unwrap();
    assert!(answers(port).await);
    drop(again);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while answers(port).await {
        assert!(
            std::time::Instant::now() < deadline,
            "a dropped lease kept it serving"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_dashboard_this_process_asked_for_outlives_leases() {
    let (_dir, storage) = strata();
    let dashboard = on_demand(&storage);
    let port = free_port();

    assert_eq!(dashboard.ensure(port).await.unwrap(), port);
    let lease = dashboard.lease(free_port()).await.unwrap();
    assert_eq!(lease.port, port);
    lease.release().await;
    assert!(
        answers(port).await,
        "VESTIGE_DASHBOARD_ENABLED's dashboard stopped"
    );
}
