//! Deep reference / reason mode.
//!
//! A Strata answer is an exact handle (id, unique prefix, or exact tag) or an
//! exact token, then the recorded neighborhood of those memories: one hop for
//! a token, the depth budget for a handle. Trust is FSRS state. Tension is a
//! recorded `corrects`, `supersedes`, `contradicts`, or `closed_by` edge.
//! Free text that is neither a handle nor an exact token is
//! `similarity_disabled` — the dashboard pins that refusal for a query such
//! as `"x"`. Nothing here is an embedding.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{DEFAULT_MEMORY_SCOPE, Storage};

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

fn validate_scope(raw_scope: Option<&str>) -> Result<(), String> {
    let scope = raw_scope.unwrap_or(DEFAULT_MEMORY_SCOPE).trim();
    if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
        return Err(
            "Invalid scope: expected a non-empty identifier of at most 200 visible characters"
                .to_string(),
        );
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

    let depth = args.depth.unwrap_or(20).clamp(5, 50) as usize;
    let limit = args.limit.unwrap_or(10).clamp(1, 50) as usize;
    validate_scope(args.scope.as_deref())?;
    validate_source_args(&args)?;
    let valid_at = parse_valid_at(args.valid_at.as_deref())?;
    let superseded_ids = storage.superseded_node_ids().map_err(|e| e.to_string())?;

    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        let trimmed = args.query.trim();
        let resolution = storage.resolve_handle(trimmed);
        if resolution.ids.is_empty() {
            return Err(format!(
                "{}: cross_reference requires exact handles in 4.0",
                crate::tools::unavailable::SIMILARITY_DISABLED
            ));
        }
        let scope = args.scope.as_deref().unwrap_or(DEFAULT_MEMORY_SCOPE).trim();
        let cross_scope = args.include_cross_scope.unwrap_or(false);
        let check_ts = valid_at.unwrap_or_else(Utc::now);
        let min_ret = args.min_retention.unwrap_or(0.0);

        let node_allowed = |n: &vestige_core::KnowledgeNode| -> bool {
            if superseded_ids.contains(&n.id) {
                return false;
            }
            if !cross_scope && !storage.node_is_in_scope(&n.id, scope).unwrap_or(false) {
                return false;
            }
            if n.valid_from.is_some_and(|t| t > check_ts) || n.valid_until.is_some_and(|t| t <= check_ts) {
                return false;
            }
            if n.retention_strength < min_ret {
                return false;
            }
            if let Some(inc) = args.include_types.as_ref()
                && !inc.is_empty()
                && !inc.iter().any(|t| t.eq_ignore_ascii_case(&n.node_type))
            {
                return false;
            }
            if let Some(exc) = args.exclude_types.as_ref()
                && exc.iter().any(|t| t.eq_ignore_ascii_case(&n.node_type))
            {
                return false;
            }
            if let Some(pref) = args.tag_prefix.as_deref()
                && !pref.is_empty()
                && !n.tags.iter().any(|t| t.starts_with(pref))
            {
                return false;
            }
            true
        };

        let mut seen = std::collections::HashSet::new();
        let mut evidence_nodes: Vec<(vestige_core::KnowledgeNode, &'static str, f64)> = Vec::new();
        let mut contradiction_pairs: Vec<Value> = Vec::new();

        for (idx, seed_id) in resolution.ids.iter().enumerate() {
            if evidence_nodes.len() >= depth {
                break;
            }
            let Some(seed) = storage.get_node(seed_id).map_err(|e| e.to_string())? else {
                continue;
            };
            if !node_allowed(&seed) || !seen.insert(seed.id.clone()) {
                continue;
            }
            let role = if idx == 0 { "primary" } else { "supporting" };
            let trust = compute_trust(seed.retention_strength, seed.stability, seed.reps, seed.lapses);
            evidence_nodes.push((seed, role, trust));
        }

        let seed_ids: Vec<String> = evidence_nodes.iter().map(|(n, _, _)| n.id.clone()).collect();
        const TENSION_KINDS: &[&str] = &["corrects", "supersedes", "contradicts", "closed_by"];
        for sid in &seed_ids {
            if let Ok(conns) = storage.get_connections_for_memory(sid) {
                for edge in conns {
                    if evidence_nodes.len() >= depth {
                        break;
                    }
                    let other_id = if edge.source_id == *sid {
                        &edge.target_id
                    } else {
                        &edge.source_id
                    };
                    let Some(other) = storage.get_node(other_id).map_err(|e| e.to_string())? else {
                        continue;
                    };
                    if !node_allowed(&other) {
                        continue;
                    }
                    let is_tension = TENSION_KINDS.contains(&edge.link_type.as_str());
                    if is_tension {
                        contradiction_pairs.push(serde_json::json!({
                            "a_id": edge.source_id,
                            "b_id": edge.target_id,
                            "relation": edge.link_type,
                            "strength": edge.strength,
                        }));
                    }
                    if seen.insert(other.id.clone()) {
                        let role = if is_tension { "contradicting" } else { "supporting" };
                        let trust = compute_trust(other.retention_strength, other.stability, other.reps, other.lapses);
                        evidence_nodes.push((other, role, trust));
                    }
                }
            }
        }

        // Also walk active (non-undone) Retroactive Salience Backfill receipts connected to seeds
        if let Ok(receipts) = storage.salience_receipts() {
            let undone: std::collections::HashSet<String> = receipts
                .iter()
                .filter_map(|r| r.get("undo_of").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string))
                .collect();
            let seed_set: std::collections::HashSet<&str> = seed_ids.iter().map(String::as_str).collect();
            for row in receipts {
                if evidence_nodes.len() >= depth {
                    break;
                }
                if row.get("undo_of").is_some_and(|v| !v.is_null()) {
                    continue;
                }
                if row.get("receipt_id").and_then(Value::as_str).is_some_and(|rid| undone.contains(rid)) {
                    continue;
                }
                let mem = row.get("memory_id").and_then(Value::as_str).unwrap_or("");
                let trig = row.get("trigger_id").and_then(Value::as_str).unwrap_or("");
                let other_id = if seed_set.contains(trig) {
                    mem
                } else if seed_set.contains(mem) {
                    trig
                } else {
                    continue;
                };
                if other_id.is_empty() {
                    continue;
                }
                if let Ok(Some(other)) = storage.get_node(other_id)
                    && node_allowed(&other)
                    && seen.insert(other.id.clone())
                {
                    let trust = compute_trust(other.retention_strength, other.stability, other.reps, other.lapses);
                    evidence_nodes.push((other, "supporting", trust));
                }
            }
        }

        evidence_nodes.sort_by(|a, b| {
            (a.1 == "primary")
                .cmp(&(b.1 == "primary"))
                .reverse()
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        let total_analyzed = evidence_nodes.len();
        evidence_nodes.truncate(limit);

        let recommended = evidence_nodes.first().map(|(n, _, trust)| {
            serde_json::json!({
                "memory_id": n.id,
                "trust": trust,
                "content": n.content,
            })
        });
        let avg_conf = if evidence_nodes.is_empty() {
            0.0
        } else {
            evidence_nodes.iter().map(|(_, _, t)| *t).sum::<f64>() / (evidence_nodes.len() as f64)
        };
        let evidence_json: Vec<Value> = evidence_nodes
            .into_iter()
            .map(|(n, role, trust)| {
                serde_json::json!({
                    "id": n.id,
                    "memory_id": n.id,
                    "role": role,
                    "trust": trust,
                    "nodeType": n.node_type,
                    "content": n.content,
                    "tags": n.tags,
                    "stability": n.stability,
                    "retentionStrength": n.retention_strength,
                    "createdAt": n.created_at.to_rfc3339(),
                })
            })
            .collect();

        return Ok(serde_json::json!({
            "query": args.query,
            "handleKind": resolution.kind.as_str(),
            "intent": "exact_handle_reason",
            "memoriesAnalyzed": total_analyzed,
            "confidence": avg_conf,
            "recommended": recommended,
            "evidence": evidence_json,
            "contradictions": contradiction_pairs,
        }));
    }

    storage
        .hybrid_search_filtered(
            &args.query,
            depth as i32,
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
