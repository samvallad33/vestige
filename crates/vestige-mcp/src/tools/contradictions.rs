//! First-class contradiction surface.
//!
//! `deep_reference` already computes trust-weighted contradiction pairs while
//! answering a specific question. This tool exposes the same local logic as an
//! inspectable memory-health operation.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use crate::tools::cross_reference::{appears_contradictory, compute_trust, topic_overlap};
use vestige_core::{DEFAULT_MEMORY_SCOPE, KnowledgeNode, Storage};

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
    topic: Option<String>,
    since: Option<String>,
    #[serde(alias = "min_trust")]
    min_trust: Option<f64>,
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

    let limit = args.limit.unwrap_or(50).clamp(2, 200);
    let min_trust = args.min_trust.unwrap_or(0.3).clamp(0.0, 1.0);
    let (scope, include_cross_scope) = parse_scope(&args)?;
    let since = match args.since.as_deref() {
        Some(raw) => Some(
            DateTime::parse_from_rfc3339(raw)
                .map_err(|e| format!("Invalid since timestamp: {}", e))?
                .with_timezone(&Utc),
        ),
        None => None,
    };

    // Filtering must happen before the requested LIMIT. For topic search the
    // current hybrid index is global, so over-fetch a bounded pool and apply
    // the canonical node scope predicate before truncation. Recent mode uses
    // the scoped SQL reader directly, preventing other namespaces from
    // consuming the window at all.
    let fetch_limit = limit.saturating_mul(4).min(800);
    let mut memories = if let Some(topic) = args.topic.as_deref().filter(|s| !s.trim().is_empty()) {
        let results = storage
            .hybrid_search(topic, fetch_limit, 0.3, 0.7)
            .map_err(|e| e.to_string())?;
        results
            .into_iter()
            .filter_map(|result| {
                if include_cross_scope {
                    Some(Ok(result.node))
                } else {
                    match storage.node_is_in_scope(&result.node.id, &scope) {
                        Ok(true) => Some(Ok(result.node)),
                        Ok(false) => None,
                        Err(error) => Some(Err(error.to_string())),
                    }
                }
            })
            .collect::<Result<Vec<_>, String>>()?
    } else if include_cross_scope {
        storage
            .get_all_nodes(fetch_limit, 0)
            .map_err(|e| e.to_string())?
    } else {
        storage
            .get_all_nodes_in_scope(&scope, fetch_limit, 0)
            .map_err(|e| e.to_string())?
    };

    if let Some(since) = since {
        memories.retain(|memory| memory.updated_at >= since);
    }
    memories.truncate(limit as usize);

    let contradictions = find_contradictions(&memories, min_trust);

    Ok(serde_json::json!({
        "topic": args.topic,
        "scope": scope,
        "includeCrossScope": include_cross_scope,
        "memoriesAnalyzed": memories.len(),
        "minTrust": min_trust,
        "contradictionsFound": contradictions.len(),
        "contradictions": contradictions,
    }))
}

fn find_contradictions(memories: &[KnowledgeNode], min_trust: f64) -> Vec<Value> {
    let mut contradictions = Vec::new();

    for i in 0..memories.len() {
        for j in (i + 1)..memories.len() {
            let a = &memories[i];
            let b = &memories[j];
            let overlap = topic_overlap(&a.content, &b.content);
            if overlap < 0.4 || !appears_contradictory(&a.content, &b.content) {
                continue;
            }

            let a_trust = trust_for(a);
            let b_trust = trust_for(b);
            if a_trust.min(b_trust) < min_trust {
                continue;
            }

            let (stronger, stronger_trust, weaker, weaker_trust) = if a_trust >= b_trust {
                (a, a_trust, b, b_trust)
            } else {
                (b, b_trust, a, a_trust)
            };

            contradictions.push(serde_json::json!({
                "stronger": memory_card(stronger, stronger_trust),
                "weaker": memory_card(weaker, weaker_trust),
                "topicOverlap": overlap,
            }));
        }
    }

    contradictions.sort_by(|a, b| {
        let a_overlap = a["topicOverlap"].as_f64().unwrap_or(0.0);
        let b_overlap = b["topicOverlap"].as_f64().unwrap_or(0.0);
        b_overlap
            .partial_cmp(&a_overlap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    contradictions
}

fn trust_for(memory: &KnowledgeNode) -> f64 {
    compute_trust(
        memory.retention_strength,
        memory.stability,
        memory.reps,
        memory.lapses,
    )
}

fn memory_card(memory: &KnowledgeNode, trust: f64) -> Value {
    serde_json::json!({
        "id": memory.id.clone(),
        "preview": memory.content.chars().take(200).collect::<String>(),
        "trust": (trust * 100.0).round() / 100.0,
        "updatedAt": memory.updated_at.to_rfc3339(),
        "tags": memory.tags.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[tokio::test]
    async fn test_contradictions_reports_conflicting_memories() {
        let (storage, _dir) = test_storage().await;
        storage
            .ingest(IngestInput {
                content:
                    "For the release workflow we always run cargo test before publishing Vestige"
                        .to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content:
                    "Correction: for the release workflow we never run cargo test before publishing Vestige"
                        .to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();

        let result = execute(
            &storage,
            Some(serde_json::json!({
                "topic": "release workflow cargo test Vestige",
                "min_trust": 0.0
            })),
        )
        .await
        .unwrap();

        assert_eq!(result["contradictionsFound"], 1);
        assert!(result["contradictions"][0]["stronger"]["id"].is_string());
    }

    #[tokio::test]
    async fn contradictions_default_to_user_scope_before_limit() {
        let (storage, _dir) = test_storage().await;
        let user_a = storage
            .ingest_in_scope(
                IngestInput {
                    content: "Scope canary always enables cargo test before release".to_string(),
                    ..Default::default()
                },
                "user",
            )
            .unwrap()
            .id;
        let user_b = storage
            .ingest_in_scope(
                IngestInput {
                    content: "Scope canary never enables cargo test before release".to_string(),
                    ..Default::default()
                },
                "user",
            )
            .unwrap()
            .id;
        let private = storage
            .ingest_in_scope(
                IngestInput {
                    content: "Scope canary private project never enables cargo test before release"
                        .to_string(),
                    ..Default::default()
                },
                "private-project",
            )
            .unwrap()
            .id;

        let scoped = execute(
            &storage,
            Some(serde_json::json!({
                "topic": "scope canary cargo test release",
                "min_trust": 0.0,
                "limit": 10
            })),
        )
        .await
        .unwrap();
        let encoded = serde_json::to_string(&scoped).unwrap();
        assert!(encoded.contains(&user_a));
        assert!(encoded.contains(&user_b));
        assert!(!encoded.contains(&private));
        assert_eq!(scoped["scope"], "user");
        assert_eq!(scoped["includeCrossScope"], false);

        let cross_scope = execute(
            &storage,
            Some(serde_json::json!({
                "topic": "scope canary cargo test release",
                "min_trust": 0.0,
                "limit": 10,
                "includeCrossScope": true
            })),
        )
        .await
        .unwrap();
        assert!(
            serde_json::to_string(&cross_scope)
                .unwrap()
                .contains(&private)
        );
    }

    #[tokio::test]
    async fn contradictions_reject_inherited_unsupported_fields() {
        let (storage, _dir) = test_storage().await;
        let error = execute(
            &storage,
            Some(serde_json::json!({
                "topic": "release",
                "token_budget": 500
            })),
        )
        .await
        .unwrap_err();
        assert!(error.contains("Unsupported recall contradictions-mode field"));
        assert!(error.contains("token_budget"));
    }
}
