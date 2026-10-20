//! Deep Reference / Reason-Mode Tool
//!
//! Validates reason-mode arguments and delegates to the active storage backend
//! (returning `similarity_disabled` on Strata 4.0) while providing shared
//! trust and contradiction helpers for dashboard and retrieval views.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::scope::validate_scope;
use crate::cognitive::CognitiveEngine;
use vestige_core::Storage;

/// Input schema for deep_reference / cross_reference tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question, claim, or topic to reason about across all memories"
            },
            "depth": {
                "type": "integer",
                "description": "How many memories to analyze (default: 20, max: 50). Higher = more thorough.",
                "default": 20,
                "minimum": 5,
                "maximum": 50
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct DeepRefArgs {
    query: String,
    depth: Option<i32>,
    limit: Option<i32>,
    #[serde(alias = "min_retention")]
    min_retention: Option<f64>,
    #[serde(alias = "min_similarity")]
    min_similarity: Option<f32>,
    #[serde(alias = "exclude_types")]
    exclude_types: Option<Vec<String>>,
    #[serde(alias = "include_types")]
    include_types: Option<Vec<String>>,
    #[serde(alias = "token_budget")]
    token_budget: Option<i32>,
    #[serde(alias = "tag_prefix")]
    tag_prefix: Option<String>,
    scope: Option<String>,
    #[serde(alias = "include_cross_scope")]
    include_cross_scope: Option<bool>,
    #[serde(alias = "valid_at")]
    valid_at: Option<String>,
    #[serde(alias = "source_system")]
    source_system: Option<String>,
    #[serde(alias = "source_project")]
    source_project: Option<String>,
    #[serde(alias = "source_id")]
    source_id: Option<String>,
    #[serde(alias = "source_type")]
    source_type: Option<String>,
    #[serde(alias = "source_author")]
    source_author: Option<String>,
    #[serde(alias = "source_updated_after")]
    source_updated_after: Option<String>,
    #[serde(alias = "source_updated_before")]
    source_updated_before: Option<String>,
    #[serde(alias = "source_status")]
    source_status: Option<String>,
}

const UNSUPPORTED_REASON_FIELDS: &[&str] = &[
    "detail_level",
    "detailLevel",
    "context_topics",
    "contextTopics",
    "retrieval_mode",
    "retrievalMode",
    "concrete",
    "rank_native_fusion",
    "rankNativeFusion",
    "topic",
    "since",
    "min_trust",
    "minTrust",
];

const SUPPORTED_REASON_FIELDS: &[&str] = &[
    "mode",
    "query",
    "depth",
    "limit",
    "min_retention",
    "minRetention",
    "min_similarity",
    "minSimilarity",
    "exclude_types",
    "excludeTypes",
    "include_types",
    "includeTypes",
    "token_budget",
    "tokenBudget",
    "tag_prefix",
    "tagPrefix",
    "scope",
    "includeCrossScope",
    "include_cross_scope",
    "validAt",
    "valid_at",
    "source_system",
    "sourceSystem",
    "source_project",
    "sourceProject",
    "source_id",
    "sourceId",
    "source_type",
    "sourceType",
    "source_author",
    "sourceAuthor",
    "source_updated_after",
    "sourceUpdatedAfter",
    "source_updated_before",
    "sourceUpdatedBefore",
    "source_status",
    "sourceStatus",
    "runId",
    "run_id",
];

fn validate_reason_envelope(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Reason-mode arguments must be an object".to_string())?;

    let unsupported: Vec<&str> = UNSUPPORTED_REASON_FIELDS
        .iter()
        .copied()
        .filter(|field| object.contains_key(*field))
        .collect();
    if !unsupported.is_empty() {
        return Err(format!(
            "Unsupported recall reason-mode field(s): {}. Use mode='lookup' for these lookup-pipeline controls.",
            unsupported.join(", ")
        ));
    }

    let unknown: Vec<&str> = object
        .keys()
        .map(String::as_str)
        .filter(|field| {
            !SUPPORTED_REASON_FIELDS.contains(field) && !UNSUPPORTED_REASON_FIELDS.contains(field)
        })
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "Unknown recall reason-mode field(s): {}.",
            unknown.join(", ")
        ));
    }

    if let Some(mode) = object.get("mode").and_then(Value::as_str)
        && mode != "reason"
    {
        return Err(format!(
            "cross_reference executes reason mode; received mode='{mode}'"
        ));
    }

    Ok(())
}

fn validate_source_args(args: &DeepRefArgs) -> Result<(), String> {
    let parse_ts = |raw: &Option<String>, field: &str| -> Result<(), String> {
        match raw {
            None => Ok(()),
            Some(value) => DateTime::parse_from_rfc3339(value)
                .map(|_| ())
                .map_err(|_| format!("Invalid {field}: '{value}' is not an RFC3339 timestamp")),
        }
    };
    match args.source_status.as_deref() {
        None | Some("any") | Some("valid") | Some("tombstoned") => {}
        Some(other) => {
            return Err(format!(
                "Invalid source_status '{other}'. Must be 'any', 'valid', or 'tombstoned'."
            ));
        }
    }
    parse_ts(&args.source_updated_after, "source_updated_after")?;
    parse_ts(&args.source_updated_before, "source_updated_before")?;
    Ok(())
}

fn parse_valid_at(raw: Option<&str>) -> Result<Option<DateTime<Utc>>, String> {
    let Some(raw) = raw else { return Ok(None) };
    if raw == "now" {
        return Ok(Some(Utc::now()));
    }
    if raw.trim() != raw || raw.is_empty() {
        return Err(
            "Invalid validAt: expected 'now', RFC3339, or YYYY-MM-DD without surrounding whitespace"
                .to_string(),
        );
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(raw) {
        return Ok(Some(timestamp.with_timezone(&Utc)));
    }
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Ok(Some(Utc.from_utc_datetime(
            &date.and_hms_opt(0, 0, 0).expect("midnight is valid"),
        )));
    }
    Err("Invalid validAt: expected 'now', RFC3339, or YYYY-MM-DD".to_string())
}

/// Compute trust score from FSRS-6 memory state.
pub(crate) fn compute_trust(retention: f64, stability: f64, reps: i32, lapses: i32) -> f64 {
    let retention_factor = retention * 0.4;
    let stability_factor = (stability / 30.0).min(1.0) * 0.2;
    let reps_factor = (reps as f64 / 10.0).min(1.0) * 0.2;
    let lapses_penalty = (1.0 - (lapses as f64 / 5.0)).max(0.0) * 0.2;
    (retention_factor + stability_factor + reps_factor + lapses_penalty).clamp(0.0, 1.0)
}

pub(crate) fn appears_contradictory(a: &str, b: &str) -> bool {
    vestige_core::advanced::contradiction::appears_contradictory(
        a,
        b,
        vestige_core::advanced::contradiction::SubjectIdentity::FromTextOverlap,
    )
}

pub(crate) fn topic_overlap(a: &str, b: &str) -> f32 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_words: std::collections::HashSet<&str> =
        a_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    let b_words: std::collections::HashSet<&str> =
        b_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

pub async fn execute(
    storage: &Arc<Storage>,
    _cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let raw_args = args.ok_or_else(|| "Missing arguments".to_string())?;
    validate_reason_envelope(&raw_args)?;
    let args: DeepRefArgs =
        serde_json::from_value(raw_args).map_err(|e| format!("Invalid arguments: {}", e))?;

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let depth = args.depth.unwrap_or(20).clamp(5, 50);
    validate_scope(args.scope.as_deref())?;
    validate_source_args(&args)?;
    let _valid_at = parse_valid_at(args.valid_at.as_deref())?;
    let _superseded_ids = storage.superseded_node_ids().map_err(|e| e.to_string())?;

    storage
        .hybrid_search_filtered(
            &args.query,
            depth,
            0.3,
            0.7,
            args.include_types.as_deref(),
            args.exclude_types.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    Err(format!(
        "{}: cross_reference requires exact handles in 4.0",
        crate::tools::unavailable::SIMILARITY_DISABLED
    ))
}
