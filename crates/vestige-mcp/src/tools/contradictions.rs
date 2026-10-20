//! First-class contradiction surface.
//!
//! On Strata, contradictions are recorded only via explicit `corrects` edges;
//! free-text pairwise contradiction scanning returns `similarity_disabled`.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use super::scope::validate_scope;
use vestige_core::Storage;

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
    #[allow(dead_code)]
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

    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return Err(
            "similarity_disabled: contradictions are not a recorded edge; embeddings, cosine, BM25, FTS, Jaccard, and keyword or name matching are not Strata operations".into(),
        );
    }

    validate_scope(args.scope.as_deref())?;
    if let Some(raw) = args.since.as_deref() {
        let _ = DateTime::parse_from_rfc3339(raw)
            .map_err(|e| format!("Invalid since timestamp: {}", e))?
            .with_timezone(&Utc);
    }
    Err(
        "similarity_disabled: contradictions are not a recorded edge; embeddings, cosine, BM25, FTS, Jaccard, and keyword or name matching are not Strata operations".into(),
    )
}
