//! Unified Search Tool
//!
//! Validates lookup arguments and delegates to the active storage backend
//! (returning `similarity_disabled` on Strata 4.0) while providing shared
//! node formatting and output-mask helpers for recall and timeline tools.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::scope::validate_scope;
use crate::cognitive::CognitiveEngine;
use vestige_core::{OutputConfig, Storage};

/// Input schema for unified search tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "context_packet": {"type":"boolean", "default":false,
                "description":"[lookup only] Return stable evidence cards with a packet ID. Scores are omitted; cards are ordered by ID for client reuse. No provider cache or savings guarantee."},
            "known_packet_id": {"type":"string", "pattern":"^[a-f0-9]{64}$",
                "description":"[lookup only] Send only while the previous complete packet remains in the model context. Matching packets return notModified=true and no cards. Omit after context loss to refresh."},
            "query": {
                "type": "string",
                "description": "Free-text query. Legacy engine only; Strata returns similarity_disabled."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "min_retention": {
                "type": "number",
                "description": "Minimum retention strength, 0 to 1 (default 0).",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "abstain_floor": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Legacy engine only: confidence floor, 0 to 1; below it recall abstains instead of returning a weak answer. Default 0.35; 1 disables."
            },
            "include_superseded": {
                "type": "boolean",
                "description": "Include memories whose validity window closed. Default false: supersession is enforced (withheld), not down-ranked. As-of validAt queries always include history."
            },
            "min_similarity": {
                "type": "number",
                "description": "Score floor, 0 to 1 (default 0.5). Legacy engine only.",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "detail_level": {
                "type": "string",
                "description": "'brief': id, type, tags, createdAt (no content). 'summary': 240-char content preview. 'full' (default on handle recall): full node content; one-hop neighbors get a 200-char preview (80 when repeated; read one in full with memory(action='get')). 'full_neighbors': full content on both nodes and one-hop neighbors.",
                "enum": ["brief", "summary", "full", "full_neighbors"],
                "default": "summary"
            },
            "context_topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Context topics for the lookup. Legacy engine only."
            },
            "exclude_types": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Node types to exclude. Default: nothing is excluded; every type is searched."
            },
            "include_types": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Only these node types; overrides exclude_types."
            },
            "token_budget": {
                "type": "integer",
                "description": "Max response tokens; content is truncated to fit. memory(action='get') has the full text.",
                "minimum": 100,
                "maximum": 100000
            },
            "retrieval_mode": {
                "type": "string",
                "description": "Legacy engine only. 'precise': top hits only. 'balanced' (default). 'exhaustive': 5x overfetch, deep traversal.",
                "enum": ["precise", "balanced", "exhaustive"],
                "default": "balanced"
            },
            "concrete": {
                "type": "boolean",
                "description": "Literal lookup, no expansion or side effects. Auto-on for quoted strings, env vars, UUIDs, paths, identifiers. Legacy engine only.",
                "default": false
            },
            "rank_native_fusion": {
                "type": "boolean",
                "description": "Experimental, legacy engine only: weighted RRF across retrieval stages instead of multipliers.",
                "default": false
            },
            "tag_prefix": {
                "type": "string",
                "description": "Only results with a tag starting with this prefix (case-insensitive). Post-filter; raise 'limit' if it thins results."
            },
            "scope": {
                "type": "string",
                "description": "Project namespace (default 'user'); project memories never bleed into an unscoped recall."
            },
            "includeCrossScope": {
                "type": "boolean",
                "description": "Search every project namespace. Only when cross-project retrieval is intended.",
                "default": false
            },
            "validAt": {
                "type": "string",
                "description": "Only facts valid at this time ('now', RFC3339, YYYY-MM-DD). Omitted: expired and future facts are listed last, not hidden."
            },
            "source_system": {
                "type": "string",
                "description": "Only memories ingested from this external system ('github', 'redmine'). Post-filter."
            },
            "source_project": {
                "type": "string",
                "description": "Only this source project or repo, exact match."
            },
            "source_id": {
                "type": "string",
                "description": "Only this source record id; pair with source_system."
            },
            "source_type": {
                "type": "string",
                "description": "Source record type, e.g. 'issue', 'comment'."
            },
            "source_author": {
                "type": "string",
                "description": "Source author or reporter (not assignee)."
            },
            "source_updated_after": {
                "type": "string",
                "description": "Source updated at or after this RFC3339 time."
            },
            "source_updated_before": {
                "type": "string",
                "description": "Source updated at or before this RFC3339 time."
            },
            "source_status": {
                "type": "string",
                "enum": ["any", "valid", "tombstoned"],
                "description": "'any' (default), 'valid' (visible upstream), 'tombstoned' (removed upstream, kept for audit).",
                "default": "any"
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct SearchArgs {
    query: String,
    #[serde(alias = "context_packet")]
    context_packet: Option<bool>,
    #[serde(alias = "known_packet_id")]
    known_packet_id: Option<String>,
    limit: Option<i32>,
    #[serde(alias = "min_retention")]
    min_retention: Option<f64>,
    #[serde(alias = "min_similarity")]
    min_similarity: Option<f32>,
    #[serde(alias = "abstain_floor")]
    abstain_floor: Option<f64>,
    #[serde(alias = "include_superseded")]
    include_superseded: Option<bool>,
    #[serde(alias = "detail_level")]
    detail_level: Option<String>,
    #[serde(alias = "context_topics")]
    context_topics: Option<Vec<String>>,
    #[serde(alias = "exclude_types")]
    exclude_types: Option<Vec<String>>,
    #[serde(alias = "include_types")]
    include_types: Option<Vec<String>>,
    #[serde(alias = "token_budget")]
    token_budget: Option<i32>,
    #[serde(alias = "retrieval_mode")]
    retrieval_mode: Option<String>,
    #[serde(alias = "rank_native_fusion")]
    rank_native_fusion: Option<bool>,
    concrete: Option<bool>,
    #[serde(alias = "tag_prefix")]
    tag_prefix: Option<String>,
    scope: Option<String>,
    include_cross_scope: Option<bool>,
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

fn validate_source_args(args: &SearchArgs) -> Result<(), String> {
    let parse_ts = |s: &Option<String>, field: &str| -> Result<(), String> {
        match s {
            None => Ok(()),
            Some(v) => chrono::DateTime::parse_from_rfc3339(v)
                .map(|_| ())
                .map_err(|_| format!("Invalid {field}: '{v}' is not an RFC3339 timestamp")),
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

pub async fn execute(
    storage: &Arc<Storage>,
    _cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SearchArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if let Some(known) = args.known_packet_id.as_deref() {
        if !args.context_packet.unwrap_or(false) {
            return Err("known_packet_id requires context_packet=true".to_string());
        }
        if known.len() != 64
            || !known
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("known_packet_id must be 64 lowercase hex characters".to_string());
        }
    }

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let detail_level = args
        .detail_level
        .as_deref()
        .unwrap_or(&output_config.detail_level);
    if !matches!(detail_level, "brief" | "full" | "summary" | "full_neighbors") {
        return Err(format!(
            "Invalid detail_level '{}'. Must be 'brief', 'summary', 'full', or 'full_neighbors'.",
            detail_level
        ));
    }

    let retrieval_mode = args.retrieval_mode.as_deref().unwrap_or("balanced");
    if !matches!(retrieval_mode, "precise" | "exhaustive" | "balanced") {
        return Err(format!(
            "Invalid retrieval_mode '{}'. Must be 'precise', 'balanced', or 'exhaustive'.",
            retrieval_mode
        ));
    }

    validate_source_args(&args)?;
    validate_scope(args.scope.as_deref())?;
    let _valid_at = parse_valid_at(args.valid_at.as_deref())?;

    let limit = args.limit.unwrap_or(10).clamp(1, 100);
    storage
        .hybrid_search_filtered(
            &args.query,
            limit,
            1.0,
            0.0,
            args.include_types.as_deref(),
            args.exclude_types.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    Err(format!(
        "{}: free-text recall is disabled in 4.0",
        crate::tools::unavailable::SIMILARITY_DISABLED
    ))
}

const SCORE_FIELDS: &[&str] = &["combinedScore", "keywordScore", "semanticScore"];
const TIMESTAMP_FIELDS: &[&str] = &[
    "createdAt",
    "updatedAt",
    "lastAccessed",
    "nextReview",
    "validFrom",
    "validUntil",
];

pub fn apply_output_masks(results: &mut [Value], output_config: &OutputConfig) {
    if output_config.show_scores && output_config.show_timestamps {
        return;
    }
    for result in results.iter_mut() {
        if let Some(obj) = result.as_object_mut() {
            if !output_config.show_scores {
                for key in SCORE_FIELDS {
                    obj.remove(*key);
                }
            }
            if !output_config.show_timestamps {
                for key in TIMESTAMP_FIELDS {
                    obj.remove(*key);
                }
            }
        }
    }
}

fn source_provenance(node: &vestige_core::KnowledgeNode) -> Value {
    let Some(env) = node.source_envelope.as_ref() else {
        return Value::Null;
    };
    serde_json::json!({
        "system": env.source_system,
        "id": env.source_id,
        "url": env.source_url,
        "project": env.source_project,
        "type": env.source_type,
        "author": env.source_author,
        "sourceUpdatedAt": env.source_updated_at.map(|dt| dt.to_rfc3339()),
        "syncedAt": env.synced_at.map(|dt| dt.to_rfc3339()),
        "tombstoned": !node.is_currently_valid(),
    })
}

fn attach_source_record(value: &mut Value, node: &vestige_core::KnowledgeNode) {
    let provenance = source_provenance(node);
    if !provenance.is_null()
        && let Value::Object(map) = value
    {
        map.insert("sourceRecord".to_string(), provenance);
    }
}

pub fn format_node(node: &vestige_core::KnowledgeNode, detail_level: &str) -> Value {
    match detail_level {
        "brief" => serde_json::json!({
            "id": node.id,
            "nodeType": node.node_type,
            "tags": node.tags,
            "retentionStrength": node.retention_strength,
        }),
        "full" => {
            let mut v = serde_json::json!({
                "id": node.id,
                "content": node.content,
                "nodeType": node.node_type,
                "tags": node.tags,
                "retentionStrength": node.retention_strength,
                "storageStrength": node.storage_strength,
                "retrievalStrength": node.retrieval_strength,
                "source": node.source,
                "sentimentScore": node.sentiment_score,
                "sentimentMagnitude": node.sentiment_magnitude,
                "createdAt": node.created_at.to_rfc3339(),
                "updatedAt": node.updated_at.to_rfc3339(),
                "lastAccessed": node.last_accessed.to_rfc3339(),
                "nextReview": node.next_review.map(|dt| dt.to_rfc3339()),
                "stability": node.stability,
                "difficulty": node.difficulty,
                "reps": node.reps,
                "lapses": node.lapses,
                "validFrom": node.valid_from.map(|dt| dt.to_rfc3339()),
                "validUntil": node.valid_until.map(|dt| dt.to_rfc3339()),
            });
            attach_source_record(&mut v, node);
            v
        }
        _ => {
            let mut v = serde_json::json!({
                "id": node.id,
                "content": node.content,
                "nodeType": node.node_type,
                "tags": node.tags,
                "retentionStrength": node.retention_strength,
            });
            attach_source_record(&mut v, node);
            v
        }
    }
}
