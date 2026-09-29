//! Sized-store performance sanity for every `memory_status` view.
//!
//! The status tool is the self-report surface; a dashboard an agent is
//! afraid to call because it might be O(n²) on a real store is a dashboard
//! that lies by absence. These tests build a 1000+ memory store, call every
//! view, and assert each completes inside a debug-mode-tolerant budget
//! (10s wall clock per view; the observed budget on this class of store is
//! well under 1s even unoptimized — the guard exists to catch pathological
//! regressions, not to police microsecond noise).

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use vestige_core::{IngestInput, OutputConfig, Storage};
use vestige_mcp::cognitive::CognitiveEngine;
use vestige_mcp::tools::memory_status;

const MEMORY_COUNT: usize = 1000;
/// Debug-build-tolerant per-view budget. Any view slower than this on 1000
/// rows is a regression worth investigating; none of the views should come
/// anywhere close (they are single-pass SQL scans plus bounded list work).
const VIEW_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

async fn sized_store() -> (Arc<Storage>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("temp dir");
    let storage = vestige_core::open_storage(Some(dir.path().join("sized.db"))).expect("storage");
    for index in 0..MEMORY_COUNT {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: format!(
                        "Sized store fixture memory number {index} about topic-{}",
                        index % 40
                    ),
                    tags: vec![format!("topic-{}", index % 40)],
                    ..Default::default()
                },
                "user",
            )
            .expect("ingest");
    }
    (storage, dir)
}

fn assert_within_budget(label: &str, started: Instant) {
    let elapsed = started.elapsed();
    assert!(
        elapsed <= VIEW_BUDGET,
        "{label} took {elapsed:?} on a {MEMORY_COUNT}-memory store; \
         budget {VIEW_BUDGET:?} — investigate for an accidental O(n²)"
    );
    eprintln!("{label}: {elapsed:?}");
}

#[tokio::test]
async fn every_view_completes_quickly_on_a_sized_store() {
    let (storage, _dir) = sized_store().await;
    let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
    let output_config = OutputConfig::default();

    // Health twice: first run may pay first-touch costs; the second proves
    // the steady-state read path is fast, not just the warm-up.
    let started = Instant::now();
    let health = memory_status::execute(&storage, &cognitive, &output_config, None)
        .await
        .expect("health view");
    assert_within_budget("view=health (cold)", started);

    let started = Instant::now();
    let health_again = memory_status::execute(
        &storage,
        &cognitive,
        &output_config,
        Some(serde_json::json!({ "view": "health" })),
    )
    .await
    .expect("health view");
    assert_within_budget("view=health (warm)", started);
    assert_eq!(health_again["totalMemories"], MEMORY_COUNT as i64);
    assert_eq!(health["stateDistribution"]["basis"], "full");

    for (view, args) in [
        ("retention", serde_json::json!({ "view": "retention" })),
        (
            "timeline",
            serde_json::json!({ "view": "timeline", "start": "2020-01-01" }),
        ),
        (
            "timeline-tagged",
            serde_json::json!({ "view": "timeline", "start": "2020-01-01", "tags": ["topic-1"] }),
        ),
        (
            "changelog",
            serde_json::json!({ "view": "changelog", "limit": 100 }),
        ),
        ("stats", serde_json::json!({ "view": "stats" })),
        (
            "stats-all-scopes",
            serde_json::json!({ "view": "stats", "all_scopes": true }),
        ),
    ] {
        let started = Instant::now();
        let value = memory_status::execute(&storage, &cognitive, &output_config, Some(args))
            .await
            .unwrap_or_else(|e| panic!("{view} view failed: {e}"));
        assert_within_budget(&format!("view={view}"), started);
        assert!(
            value.get("error").is_none(),
            "{view} returned an error payload: {value}"
        );
    }

    // The tools view is catalog work, not storage work; still budgeted.
    // `tool_guide` reads the live tools/list shape; a synthetic one-tool
    // catalog exercises both the inventory path and the full-schema unfold
    // (an empty catalog errors by design — "Unknown advertised tool").
    let catalog = serde_json::json!({
        "tools": [{
            "name": "recall",
            "description": "hybrid memory search",
            "annotations": { "readOnlyHint": true },
            "inputSchema": { "type": "object", "properties": { "query": { "type": "string" } } }
        }]
    });
    let started = Instant::now();
    let inventory =
        memory_status::tool_guide(&catalog, &serde_json::json!({})).expect("tools inventory");
    assert_eq!(inventory["tools"].as_array().unwrap().len(), 1);
    let unfolded = memory_status::tool_guide(&catalog, &serde_json::json!({ "tool": "recall" }))
        .expect("tools full-schema unfold");
    assert!(unfolded["tools"][0]["inputSchema"].is_object());
    assert_within_budget("view=tools (inventory + unfold)", started);
}

/// Scaling guard: doubling the store must not quadruple the stats view (the
/// view most at risk because it aggregates every row). Ratio is generous —
/// this catches O(n²), not constant-factor drift.
#[tokio::test]
async fn stats_view_scales_linearly_not_quadratically() {
    async fn timed_stats(count: usize) -> std::time::Duration {
        let dir = tempfile::tempdir().expect("temp dir");
        let storage =
            vestige_core::open_storage(Some(dir.path().join("scale.db"))).expect("storage");
        for index in 0..count {
            storage
                .ingest_in_scope(
                    IngestInput {
                        content: format!("Scaling fixture {index}"),
                        ..Default::default()
                    },
                    "user",
                )
                .expect("ingest");
        }
        let started = Instant::now();
        vestige_mcp::tools::hygiene_stats::execute(&storage, Some(serde_json::json!({})))
            .await
            .expect("stats");
        started.elapsed()
    }

    let small = timed_stats(250).await;
    let large = timed_stats(1000).await;
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(0.000_001);
    assert!(
        ratio < 25.0,
        "stats view grew {ratio:.1}x for a 4x store (small {small:?}, large {large:?}); \
         suspicious of quadratic behavior"
    );
    assert!(
        large <= VIEW_BUDGET,
        "stats view took {large:?} on 1000 rows; budget {VIEW_BUDGET:?}"
    );
}
