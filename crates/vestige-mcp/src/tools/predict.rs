//! Predict tool — Proactive memory prediction from exact code anchors.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::Storage;

pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "context": {
                "type": "object",
                "description": "Current context for prediction",
                "properties": {
                    "current_file": { "type": "string" },
                    "current_topics": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "codebase": { "type": "string" }
                }
            }
        }
    })
}

pub async fn execute(
    storage: &Arc<Storage>,
    _cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let context = args.as_ref().and_then(|a| a.get("context"));
    let mut base = crate::strata_memory::ghostlink::predict(storage.as_ref(), context)?;
    if let Some(file) = context
        .and_then(|c| c.get("current_file"))
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        let cwd = std::env::current_dir().ok();
        let report = crate::cognitive_kernel::simulate_edit_blast_radius(
            storage.as_ref(),
            file,
            cwd.as_deref(),
        );
        if let Some(obj) = base.as_object_mut() {
            obj.insert(
                "blastRadius".to_string(),
                serde_json::to_value(report).unwrap_or(serde_json::Value::Null),
            );
        }
    }
    Ok(base)
}
