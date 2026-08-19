//! Importance Score Tool
//!
//! Exposes the 4-channel importance signaling system as an MCP tool.
//! Wraps ImportanceSignals::compute_importance() from vestige-core's
//! neuroscience module (dopamine/norepinephrine/acetylcholine/serotonin model).
//!
//! v1.5.0: Uses CognitiveEngine's persistent signals so novelty/reward/attention
//! accumulate across calls (not freshly created per call).

use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{ImportanceContext, Storage};

/// Input schema for importance_score tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to score for importance"
            },
            "context_topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional topics for novelty detection context"
            },
            "project": {
                "type": "string",
                "description": "Optional project/codebase name for context"
            }
        },
        "required": ["content"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportanceArgs {
    content: String,
    context_topics: Option<Vec<String>>,
    project: Option<String>,
}

pub async fn execute(
    _storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: ImportanceArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.content.trim().is_empty() {
        return Err("Content cannot be empty".to_string());
    }

    let mut context = ImportanceContext::current();
    if let Some(project) = args.project {
        context = context.with_project(project);
    }
    if let Some(topics) = args.context_topics {
        context = context.with_tags(topics);
    }

    // Use CognitiveEngine's persistent signals (novelty/reward/attention accumulate)
    let cog = cognitive.lock().await;
    let score = cog
        .importance_signals
        .compute_importance(&args.content, &context);

    // Also detect emotional markers for richer output
    let emotional_markers = cog.arousal_signal.detect_emotional_markers(&args.content);
    drop(cog);

    let markers_json: Vec<Value> = emotional_markers
        .iter()
        .map(|m| {
            serde_json::json!({
                "type": format!("{:?}", m.marker_type),
                "text": m.text,
                "intensity": m.intensity
            })
        })
        .collect();

    Ok(serde_json::json!({
        "composite": score.composite,
        "channels": {
            "novelty": score.novelty,
            "arousal": score.arousal,
            "reward": score.reward,
            "attention": score.attention
        },
        "encodingBoost": score.encoding_boost,
        "consolidationPriority": format!("{:?}", score.consolidation_priority),
        "weightsUsed": {
            "novelty": score.weights_used.novelty,
            "arousal": score.weights_used.arousal,
            "reward": score.weights_used.reward,
            "attention": score.weights_used.attention
        },
        "explanations": {
            "novelty": score.novelty_explanation.as_ref().map(|e| format!("{:?}", e)),
            "arousal": score.arousal_explanation.as_ref().map(|e| format!("{:?}", e)),
            "reward": score.reward_explanation.as_ref().map(|e| format!("{:?}", e)),
            "attention": score.attention_explanation.as_ref().map(|e| format!("{:?}", e))
        },
        "emotionalMarkers": markers_json,
        "summary": score.summary(),
        "dominantSignal": score.dominant_signal()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    #[test]
    fn test_schema_has_required_fields() {
        let schema = schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["content"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("content"))
        );
    }

    #[tokio::test]
    async fn test_empty_content_fails() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("importance.db"))).unwrap());
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({ "content": "" })),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_basic_importance_score() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("importance.db"))).unwrap());
        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "content": "CRITICAL: Production database migration failed with data loss!"
            })),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value["composite"].as_f64().is_some());
        assert!(value["channels"]["novelty"].as_f64().is_some());
        assert!(value["channels"]["arousal"].as_f64().is_some());
        assert!(value["dominantSignal"].is_string());
    }
}
