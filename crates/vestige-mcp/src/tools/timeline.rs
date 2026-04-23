//! Memory Timeline Tool
//!
//! Browse memories chronologically. Returns memories in a time range,
//! grouped by day. Defaults to last 7 days.

use chrono::{DateTime, NaiveDate, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

use vestige_core::Storage;

use super::search_unified::format_node;

/// Input schema for memory_timeline tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "start": {
                "type": "string",
                "description": "Start of time range (ISO 8601 date or datetime). Default: 7 days ago."
            },
            "end": {
                "type": "string",
                "description": "End of time range (ISO 8601 date or datetime). Default: now."
            },
            "node_type": {
                "type": "string",
                "description": "Filter by node type (e.g. 'fact', 'concept', 'decision')"
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Filter by tags (ANY match)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return (default: 50, max: 200)",
                "default": 50,
                "minimum": 1,
                "maximum": 200
            },
            "detail_level": {
                "type": "string",
                "description": "Level of detail: 'brief', 'summary' (default), or 'full'",
                "enum": ["brief", "summary", "full"],
                "default": "summary"
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimelineArgs {
    start: Option<String>,
    end: Option<String>,
    #[serde(alias = "node_type")]
    node_type: Option<String>,
    tags: Option<Vec<String>>,
    limit: Option<i32>,
    #[serde(alias = "detail_level")]
    detail_level: Option<String>,
}

/// Parse an ISO 8601 date or datetime string into a DateTime<Utc>.
/// Supports both `2026-02-01` and `2026-02-01T00:00:00Z` formats.
fn parse_datetime(s: &str) -> Result<DateTime<Utc>, String> {
    // Try full datetime first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Try date-only (YYYY-MM-DD)
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| format!("Invalid date: {}", s))?
            .and_utc();
        return Ok(dt);
    }
    Err(format!(
        "Invalid date/datetime '{}'. Use ISO 8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ",
        s
    ))
}

/// Execute memory_timeline tool
pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: TimelineArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => TimelineArgs {
            start: None,
            end: None,
            node_type: None,
            tags: None,
            limit: None,
            detail_level: None,
        },
    };

    // Validate detail_level
    let detail_level = match args.detail_level.as_deref() {
        Some("brief") => "brief",
        Some("full") => "full",
        Some("summary") | None => "summary",
        Some(invalid) => {
            return Err(format!(
                "Invalid detail_level '{}'. Must be 'brief', 'summary', or 'full'.",
                invalid
            ));
        }
    };

    // Parse time range
    let now = Utc::now();
    let start = match &args.start {
        Some(s) => Some(parse_datetime(s)?),
        None => Some(now - chrono::Duration::days(7)),
    };
    let end = match &args.end {
        Some(e) => Some(parse_datetime(e)?),
        None => Some(now),
    };

    let limit = args.limit.unwrap_or(50).clamp(1, 200);

    // Query memories in time range with filters pushed into SQL. Rust-side
    // `retain` after `LIMIT` was unsafe for sparse types/tags — a dominant
    // set could crowd the sparse matches out of the limit window and leave
    // the retain with 0 rows to keep.
    let results = storage
        .query_time_range(
            start,
            end,
            limit,
            args.node_type.as_deref(),
            args.tags.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    // Group by day
    let mut by_day: BTreeMap<NaiveDate, Vec<Value>> = BTreeMap::new();
    for node in &results {
        let date = node.created_at.date_naive();
        by_day
            .entry(date)
            .or_default()
            .push(format_node(node, detail_level));
    }

    // Build timeline (newest first)
    let timeline: Vec<Value> = by_day
        .into_iter()
        .rev()
        .map(|(date, memories)| {
            serde_json::json!({
                "date": date.to_string(),
                "count": memories.len(),
                "memories": memories,
            })
        })
        .collect();

    let total = results.len();
    let days = timeline.len();

    Ok(serde_json::json!({
        "tool": "memory_timeline",
        "range": {
            "start": start.map(|dt| dt.to_rfc3339()),
            "end": end.map(|dt| dt.to_rfc3339()),
        },
        "detailLevel": detail_level,
        "totalMemories": total,
        "days": days,
        "timeline": timeline,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_test_memory(storage: &Arc<Storage>, content: &str) {
        storage
            .ingest(vestige_core::IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["timeline-test".to_string()],
                valid_from: None,
                valid_until: None,
            })
            .unwrap();
    }

    /// Ingest with explicit node_type and tags. Used by the sparse-filter
    /// regression tests so the dominant and sparse sets can be told apart.
    async fn ingest_typed(
        storage: &Arc<Storage>,
        content: &str,
        node_type: &str,
        tags: &[&str],
    ) {
        storage
            .ingest(vestige_core::IngestInput {
                content: content.to_string(),
                node_type: node_type.to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();
    }

    #[test]
    fn test_schema_has_properties() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["start"].is_object());
        assert!(s["properties"]["end"].is_object());
        assert!(s["properties"]["node_type"].is_object());
        assert!(s["properties"]["tags"].is_object());
        assert!(s["properties"]["limit"].is_object());
        assert!(s["properties"]["detail_level"].is_object());
    }

    #[test]
    fn test_parse_datetime_rfc3339() {
        let result = parse_datetime("2026-02-18T10:30:00Z");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_datetime_date_only() {
        let result = parse_datetime("2026-02-18");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_datetime_invalid() {
        let result = parse_datetime("not-a-date");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid date/datetime"));
    }

    #[test]
    fn test_parse_datetime_empty() {
        let result = parse_datetime("");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_timeline_no_args_defaults() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["tool"], "memory_timeline");
        assert_eq!(value["detailLevel"], "summary");
        assert!(value["range"]["start"].is_string());
        assert!(value["range"]["end"].is_string());
    }

    #[tokio::test]
    async fn test_timeline_empty_database() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, None).await;
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 0);
        assert_eq!(value["days"], 0);
        assert!(value["timeline"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_timeline_with_memories() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Timeline test memory 1").await;
        ingest_test_memory(&storage, "Timeline test memory 2").await;
        let result = execute(&storage, None).await;
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 2);
        assert!(value["days"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_timeline_invalid_detail_level() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "detail_level": "invalid" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid detail_level"));
    }

    #[tokio::test]
    async fn test_timeline_detail_level_brief() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Brief test memory").await;
        let args = serde_json::json!({ "detail_level": "brief" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["detailLevel"], "brief");
    }

    #[tokio::test]
    async fn test_timeline_detail_level_full() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Full test memory").await;
        let args = serde_json::json!({ "detail_level": "full" });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["detailLevel"], "full");
    }

    #[tokio::test]
    async fn test_timeline_limit_clamped() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "limit": 0 });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok()); // limit clamped to 1, no error
    }

    #[tokio::test]
    async fn test_timeline_with_date_range() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Ranged memory").await;
        let args = serde_json::json!({
            "start": "2020-01-01",
            "end": "2030-12-31"
        });
        let result = execute(&storage, Some(args)).await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value["totalMemories"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_timeline_node_type_filter() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "A fact memory").await;
        let args = serde_json::json!({ "node_type": "concept" });
        let result = execute(&storage, Some(args)).await;
        let value = result.unwrap();
        // Ingested as "fact", filtering for "concept" should yield 0
        assert_eq!(value["totalMemories"], 0);
    }

    #[tokio::test]
    async fn test_timeline_tag_filter() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Tagged memory").await;
        let args = serde_json::json!({ "tags": ["timeline-test"] });
        let result = execute(&storage, Some(args)).await;
        let value = result.unwrap();
        assert!(value["totalMemories"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_timeline_tag_filter_no_match() {
        let (storage, _dir) = test_storage().await;
        ingest_test_memory(&storage, "Tagged memory").await;
        let args = serde_json::json!({ "tags": ["nonexistent-tag"] });
        let result = execute(&storage, Some(args)).await;
        let value = result.unwrap();
        assert_eq!(value["totalMemories"], 0);
    }

    /// Regression: `node_type` filter must work even when the sparse type is
    /// crowded out by a dominant type within the SQL `LIMIT`. Before the fix,
    /// `query_time_range` applied `LIMIT` before the Rust-side `retain`, so a
    /// limit of 5 against 10 dominant + 2 sparse rows returned 5 dominant,
    /// then filtered to 0 sparse.
    #[tokio::test]
    async fn test_timeline_node_type_filter_sparse() {
        let (storage, _dir) = test_storage().await;

        // Dominant set: 10 facts
        for i in 0..10 {
            ingest_typed(&storage, &format!("Dominant memory {}", i), "fact", &["alpha"]).await;
        }
        // Sparse set: 2 concepts
        for i in 0..2 {
            ingest_typed(&storage, &format!("Sparse memory {}", i), "concept", &["beta"]).await;
        }

        // Limit 5 against 12 total — before the fix, `retain` on `concept`
        // would operate on the 5 most recent rows (all `fact`) and find 0.
        let args = serde_json::json!({ "node_type": "concept", "limit": 5 });
        let value = execute(&storage, Some(args)).await.unwrap();
        assert_eq!(
            value["totalMemories"], 2,
            "Both sparse concepts should survive a limit smaller than the dominant set"
        );

        // Also verify the storage layer directly, so the contract is pinned
        // at the API boundary even if the tool wrapper shifts.
        let nodes = storage
            .query_time_range(None, None, 5, Some("concept"), None)
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().all(|n| n.node_type == "concept"));
    }

    /// Regression: `tags` filter must work even when the sparse tag is
    /// crowded out by a dominant tag within the SQL `LIMIT`. Parallel to
    /// the node_type sparse case — same `retain`-after-`LIMIT` bug.
    #[tokio::test]
    async fn test_timeline_tag_filter_sparse() {
        let (storage, _dir) = test_storage().await;

        // Dominant set: 10 memories with tag "common"
        for i in 0..10 {
            ingest_typed(&storage, &format!("Common memory {}", i), "fact", &["common"]).await;
        }
        // Sparse set: 2 memories with tag "rare"
        for i in 0..2 {
            ingest_typed(&storage, &format!("Rare memory {}", i), "fact", &["rare"]).await;
        }

        let args = serde_json::json!({ "tags": ["rare"], "limit": 5 });
        let value = execute(&storage, Some(args)).await.unwrap();
        assert_eq!(
            value["totalMemories"], 2,
            "Both sparse-tag matches should survive a limit smaller than the dominant set"
        );

        let tag_slice = vec!["rare".to_string()];
        let nodes = storage
            .query_time_range(None, None, 5, None, Some(&tag_slice))
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().all(|n| n.tags.iter().any(|t| t == "rare")));
    }

    /// Regression: tag filter must match exact tags, not substrings. Without
    /// the `"tag"`-wrapped `LIKE` pattern, a query for `alpha` would also
    /// match rows tagged `alphabet`. The pattern `%"alpha"%` keys off the
    /// JSON-array quote characters and rejects that.
    #[tokio::test]
    async fn test_timeline_tag_filter_exact_match() {
        let (storage, _dir) = test_storage().await;

        ingest_typed(&storage, "Exact tag hit", "fact", &["alpha"]).await;
        ingest_typed(&storage, "Substring decoy", "fact", &["alphabet"]).await;

        let tag_slice = vec!["alpha".to_string()];
        let nodes = storage
            .query_time_range(None, None, 50, None, Some(&tag_slice))
            .unwrap();
        assert_eq!(nodes.len(), 1, "Only the exact-tag match should return");
        assert_eq!(nodes[0].content, "Exact tag hit");
    }
}
