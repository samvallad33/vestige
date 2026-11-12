//! First-class contradiction surface.
//!
//! On Strata, a contradiction is a recorded tension edge (`corrects`,
//! `supersedes`, `contradicts`, `closed_by`). A `topic` narrows that scan to
//! an exact handle or an exact token. Pair text is never compared.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use vestige_core::{DEFAULT_MEMORY_SCOPE, Storage};

pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Optional topic/query to scope contradiction detection. If omitted, scans recent memories."
            },
            "since": {
                "type": "string",
                "description": "Optional RFC3339 timestamp; only memories updated after this time are considered."
            },
            "min_trust": {
                "type": "number",
                "description": "Minimum trust score for both sides of a contradiction.",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.3
            },
            "limit": {
                "type": "integer",
                "description": "Maximum memories to analyze before pairwise contradiction detection.",
                "minimum": 2,
                "maximum": 200,
                "default": 50
            },
            "scope": {
                "type": "string",
                "description": "Memory namespace. Defaults to 'user'."
            },
            "includeCrossScope": {
                "type": "boolean",
                "description": "Inspect all namespaces only when explicitly true.",
                "default": false
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContradictionArgs {
    #[allow(dead_code)]
    topic: Option<String>,
    since: Option<String>,
    #[serde(alias = "min_trust")]
    #[allow(dead_code)]
    min_trust: Option<f64>,
    #[allow(dead_code)]
    limit: Option<i32>,
    scope: Option<String>,
    #[serde(alias = "include_cross_scope")]
    include_cross_scope: Option<bool>,
}

const SUPPORTED_FIELDS: &[&str] = &[
    "mode",
    "topic",
    "since",
    "min_trust",
    "minTrust",
    "limit",
    "scope",
    "includeCrossScope",
    "include_cross_scope",
    "runId",
    "run_id",
];

fn validate_envelope(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Contradictions-mode arguments must be an object".to_string())?;
    let unsupported: Vec<&str> = object
        .keys()
        .map(String::as_str)
        .filter(|field| !SUPPORTED_FIELDS.contains(field))
        .collect();
    if !unsupported.is_empty() {
        return Err(format!(
            "Unsupported recall contradictions-mode field(s): {}. Use mode='lookup' or mode='reason' for those controls.",
            unsupported.join(", ")
        ));
    }
    Ok(())
}

fn parse_scope(args: &ContradictionArgs) -> Result<(String, bool), String> {
    let scope = args
        .scope
        .as_deref()
        .unwrap_or(DEFAULT_MEMORY_SCOPE)
        .trim()
        .to_string();
    if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
        return Err(
            "Invalid scope: expected a non-empty identifier of at most 200 visible characters"
                .to_string(),
        );
    }
    Ok((scope, args.include_cross_scope.unwrap_or(false)))
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ContradictionArgs = match args {
        Some(value) => {
            validate_envelope(&value)?;
            serde_json::from_value(value).map_err(|e| format!("Invalid arguments: {}", e))?
        }
        None => ContradictionArgs {
            topic: None,
            since: None,
            min_trust: None,
            limit: None,
            scope: None,
            include_cross_scope: None,
        },
    };

    let (scope, cross_scope) = parse_scope(&args)?;
    let since_dt = match args.since.as_deref() {
        Some(raw) => Some(
            DateTime::parse_from_rfc3339(raw)
                .map_err(|e| format!("Invalid since timestamp: {}", e))?
                .with_timezone(&Utc),
        ),
        None => None,
    };

    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        // On Strata, contradictions are explicit recorded tension edges (`corrects`,
        // `supersedes`, `contradicts`, `closed_by`). If `topic` is passed, it must resolve
        // as an exact handle (id, prefix, or exact tag); free-text similarity stays refused.
        let topic_filter: Option<std::collections::HashSet<String>> = match args.topic.as_deref().map(str::trim) {
            Some(t) if !t.is_empty() => {
                let res = storage.resolve_handle(t);
                if res.ids.is_empty() {
                    return Err(
                        "similarity_disabled: contradictions topic on Strata must be an exact handle (memory id or exact tag); embeddings and free-text similarity are disabled".into(),
                    );
                }
                Some(res.ids.into_iter().collect())
            }
            _ => None,
        };

        let retired = storage.superseded_node_ids().map_err(|e| e.to_string())?;
        let min_trust = args.min_trust.unwrap_or(0.0).clamp(0.0, 1.0);
        let max_pairs = args.limit.unwrap_or(50).clamp(2, 200) as usize;
        const TENSION_EDGES: &[&str] = &["corrects", "supersedes", "contradicts", "closed_by"];

        let mut edges: Vec<_> = storage
            .get_all_connections()
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter(|e| TENSION_EDGES.contains(&e.link_type.as_str()) && e.source_id != e.target_id)
            .collect();
        edges.sort_by_key(|e| std::cmp::Reverse(e.created_at));

        let mut seen = std::collections::HashSet::new();
        let mut pairs = Vec::new();
        for edge in edges {
            if pairs.len() >= max_pairs {
                break;
            }
            if retired.contains(&edge.source_id) || retired.contains(&edge.target_id) {
                continue;
            }
            if let Some(ref tf) = topic_filter
                && !tf.contains(&edge.source_id)
                && !tf.contains(&edge.target_id)
            {
                continue;
            }
            if !cross_scope {
                let s_ok = storage.node_is_in_scope(&edge.source_id, &scope).unwrap_or(false);
                let t_ok = storage.node_is_in_scope(&edge.target_id, &scope).unwrap_or(false);
                if !s_ok || !t_ok {
                    continue;
                }
            }
            let Some(source) = storage.get_node(&edge.source_id).map_err(|e| e.to_string())? else {
                continue;
            };
            let Some(target) = storage.get_node(&edge.target_id).map_err(|e| e.to_string())? else {
                continue;
            };
            if let Some(since) = since_dt
                && source.updated_at < since
                && target.updated_at < since
                && edge.created_at < since
            {
                continue;
            }
            let trust_src = super::cross_reference::compute_trust(
                source.retention_strength,
                source.stability,
                source.reps,
                source.lapses,
            );
            let trust_tgt = super::cross_reference::compute_trust(
                target.retention_strength,
                target.stability,
                target.reps,
                target.lapses,
            );
            if trust_src < min_trust || trust_tgt < min_trust {
                continue;
            }
            if !seen.insert((edge.source_id.clone(), edge.target_id.clone(), edge.link_type.clone())) {
                continue;
            }
            let (a, ta, b, tb) = if source.created_at <= target.created_at {
                (&source, trust_src, &target, trust_tgt)
            } else {
                (&target, trust_tgt, &source, trust_src)
            };
            pairs.push(serde_json::json!({
                "memory_a_id": a.id,
                "memory_b_id": b.id,
                "memory_a_preview": a.content.chars().take(200).collect::<String>(),
                "memory_b_preview": b.content.chars().take(200).collect::<String>(),
                "memory_a_type": a.node_type,
                "memory_b_type": b.node_type,
                "memory_a_created": a.created_at.to_rfc3339(),
                "memory_b_created": b.created_at.to_rfc3339(),
                "memory_a_tags": a.tags,
                "memory_b_tags": b.tags,
                "trust_a": (ta * 100.0).round() / 100.0,
                "trust_b": (tb * 100.0).round() / 100.0,
                "strength": edge.strength.clamp(0.0, 1.0),
                "relation": edge.link_type,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "recorded_at": edge.created_at.to_rfc3339(),
            }));
        }

        return Ok(serde_json::json!({
            "contradictions": pairs,
            "total": pairs.len(),
            "scope": if cross_scope { "all" } else { scope.as_str() },
            "source": "recorded_tension_edges",
        }));
    }

    Err(
        "similarity_disabled: contradictions are not a recorded edge; embeddings, cosine, BM25, FTS, Jaccard, and keyword or name matching are not Strata operations".into(),
    )
}
