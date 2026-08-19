//! # Retroactive Salience Backfill — MCP tool
//!
//! Memory with hindsight. When a salient FAILURE memory exists (a bug/crash/
//! regression — the "aversive event"), this reaches BACKWARD across history and
//! promotes the quiet earlier memory that caused it: the root cause a vector
//! search structurally cannot surface because it is not *similar* to the
//! failure, only causally upstream.
//!
//! Faithful port of Zaki/Cai et al. (2024) Nature 637:145-155. The core logic
//! lives in `vestige_core::advanced::retroactive_backfill`; this tool wires it
//! to real storage: builds candidates from `KnowledgeNode`s (entities drawn from
//! tags and content), runs the backward reach, and PROMOTES the surfaced cause
//! so it stops decaying and resurfaces next time.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use chrono::Utc;
use vestige_core::advanced::prediction_error::cosine_similarity;
use vestige_core::advanced::retroactive_backfill::{
    self, BackfillCandidate, FailureEvent, RetroactiveBackfill,
};
use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "failure_id": {
                "type": "string",
                "description": "ID of the failure/'aversive event' memory to backfill from. If omitted, the most recent memory that looks like a failure is used."
            },
            "manual": {
                "type": "boolean",
                "description": "Force the backfill even if the event isn't auto-detected as salient (manual override). Default false.",
                "default": false
            },
            "lookback_days": {
                "type": "integer",
                "description": "How many days back to reach for the cause. Default 30.",
                "minimum": 1,
                "maximum": 365,
                "default": 30
            },
            "promote": {
                "type": "boolean",
                "description": "Whether to actually promote (boost) the surfaced cause(s) in storage. Default true. Set false for a dry-run preview.",
                "default": true
            },
            "scan_limit": {
                "type": "integer",
                "description": "Max memories to scan as candidate causes. Default 500.",
                "minimum": 10,
                "maximum": 5000,
                "default": 500
            }
        }
    })
}

#[derive(Deserialize, Default)]
struct Args {
    failure_id: Option<String>,
    #[serde(default)]
    manual: bool,
    lookback_days: Option<i64>,
    promote: Option<bool>,
    scan_limit: Option<i32>,
}

/// Pull entities out of a memory: its tags, plus heuristic code-ish tokens from
/// content (UPPER_SNAKE env vars, dotted/slashed file paths). These are the
/// shared-entity join keys the backward reach follows.
///
/// Thin `&KnowledgeNode` adapter over the single core definition
/// [`retroactive_backfill::extract_entities`] so the MCP tool, CLI, and the
/// offline consolidation pass all extract entities identically (no drift).
fn extract_entities(node: &KnowledgeNode) -> Vec<String> {
    retroactive_backfill::extract_entities(&node.content, &node.tags)
}

/// Heuristic: does this memory read like a failure/"aversive event"? Checks both
/// content AND tags against the full FAILURE_MARKERS list. Public so the CLI and
/// any caller share ONE failure-detection definition (no drifting subsets).
///
/// Thin `&KnowledgeNode` adapter over [`retroactive_backfill::looks_like_failure`].
pub fn looks_like_failure(node: &KnowledgeNode) -> bool {
    retroactive_backfill::looks_like_failure(&node.content, &node.tags)
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    // Clamp numeric inputs to the documented schema bounds. The MCP dispatch
    // layer does NOT enforce the JSON-schema min/max, so a caller can send
    // scan_limit=-1 (SQLite treats a negative LIMIT as unbounded => full-table
    // fetch = DoS) or values above the 5000 cap. Clamp rather than trust.
    let lookback = args.lookback_days.unwrap_or(30).clamp(1, 365);
    let promote = args.promote.unwrap_or(true);
    let scan_limit = args.scan_limit.unwrap_or(500).clamp(10, 5000);

    // 1. Resolve the failure event.
    let failure_node = match &args.failure_id {
        Some(id) => storage
            .get_node(id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("failure memory '{id}' not found"))?,
        None => {
            // most recent memory that looks like a failure
            let recent = storage
                .get_all_nodes(scan_limit, 0)
                .map_err(|e| e.to_string())?;
            recent.into_iter().find(looks_like_failure).ok_or_else(|| {
                "no failure-like memory found to backfill from; pass failure_id or manual=true"
                    .to_string()
            })?
        }
    };

    let failure_entities = extract_entities(&failure_node);
    let failure_embedding = storage.get_node_embedding(&failure_node.id).ok().flatten();

    // surprise/prediction-error proxy: a failure-marked memory is treated as
    // high-salience; otherwise fall back to a neutral value (manual can force).
    let pe = if looks_like_failure(&failure_node) {
        0.9_f32
    } else {
        0.3_f32
    };

    let failure = FailureEvent {
        id: failure_node.id.clone(),
        content: failure_node.content.clone(),
        entities: failure_entities.clone(),
        tags: failure_node.tags.clone(),
        prediction_error: pe,
        manual: args.manual,
    };

    // 2. Build candidate causes from all OTHER memories (older than the failure).
    let all = storage
        .get_all_nodes(scan_limit, 0)
        .map_err(|e| e.to_string())?;
    let mut candidates: Vec<BackfillCandidate> = Vec::new();
    for node in &all {
        if node.id == failure_node.id {
            continue;
        }
        let age = (failure_node.created_at - node.created_at).num_seconds() as f64 / 86_400.0;
        // only consider memories strictly older than the failure (backward-only)
        if age <= 0.0 {
            continue;
        }
        let sim = match (
            &failure_embedding,
            storage.get_node_embedding(&node.id).ok().flatten(),
        ) {
            (Some(f), Some(c)) if f.len() == c.len() => Some(cosine_similarity(f, &c)),
            _ => None,
        };
        candidates.push(BackfillCandidate {
            id: node.id.clone(),
            content: node.content.clone(),
            entities: extract_entities(node),
            age_days_before_failure: age,
            stability: node.stability,
            similarity_to_failure: sim,
        });
    }

    // 3. Run the backward reach.
    let backfill = RetroactiveBackfill {
        lookback_days: lookback,
        ..RetroactiveBackfill::new()
    };
    let result = backfill.run(&failure, &candidates);

    if !result.triggered {
        return Ok(json!({
            "tool": "backfill",
            "triggered": false,
            "reason": "the event was not salient (not a detected failure and manual=false). Pass manual=true to force.",
            "failure_id": failure.id,
        }));
    }

    // 4. Promote the surfaced cause(s) so they stop decaying and resurface.
    let mut promoted = Vec::new();
    for cause in &result.causes {
        let content_preview = candidates
            .iter()
            .find(|c| c.id == cause.memory_id)
            .map(|c| c.content.chars().take(140).collect::<String>())
            .unwrap_or_default();
        // A Backfill result is explicit-entity evidence, not an omniscient
        // causal oracle. Always persist that evidence edge before saving the
        // receipt — including dry-run previews. `promote=false` means no FSRS
        // mutation, not "hide the evidence from the graph". The distinct link
        // type prevents any consumer from presenting this as a proven causal
        // relationship.
        let link_type = "backfill_candidate".to_string();
        let already_linked = storage
            .get_connections_for_memory(&cause.memory_id)
            .map(|connections| {
                connections.iter().any(|connection| {
                    connection.source_id == cause.memory_id
                        && connection.target_id == failure_node.id
                        && connection.link_type == link_type
                })
            })
            .unwrap_or(false);
        let candidate_edge_persisted = already_linked
            || storage
                .save_connection(&ConnectionRecord {
                    source_id: cause.memory_id.clone(),
                    target_id: failure_node.id.clone(),
                    strength: cause.score.clamp(0.0, 1.0),
                    link_type,
                    created_at: Utc::now(),
                    last_activated: Utc::now(),
                    activation_count: 0,
                })
                .is_ok();
        let did_promote = if promote && candidate_edge_persisted {
            // promote_memory_backfill boosts retrieval strength + reps (the
            // FSRS knob) with a bounded stability multiply.
            storage.promote_memory_backfill(&cause.memory_id).is_ok()
        } else {
            false
        };
        promoted.push(json!({
            "memory_id": cause.memory_id,
            "content_preview": content_preview,
            "shared_entities": cause.shared_entities,
            "age_days_before_failure": (cause.age_days * 10.0).round() / 10.0,
            "similarity_rank": cause.similarity_rank,
            "backfill_score": (cause.score * 100.0).round() / 100.0,
            "promoted": did_promote,
            "candidate_edge_persisted": candidate_edge_persisted,
            "reason": cause.reason,
        }));
    }

    Ok(json!({
        "tool": "backfill",
        "triggered": true,
        "lookback_days": lookback,
        "headline": format!(
            "Reached back across history from the failure and surfaced {} causal memor{} that semantic search would have missed.",
            result.causes.len(),
            if result.causes.len() == 1 { "y" } else { "ies" }
        ),
        "failure": {
            "id": failure.id,
            "content_preview": failure.content.chars().take(160).collect::<String>(),
            "entities": failure_entities,
        },
        "scanned": result.scanned,
        // This is an explicit direct evidence edge from the highest-ranked
        // candidate to the failure, not a topology inferred by the renderer.
        // The complete candidate set remains below for alternate branches.
        "path_ids": promoted.first()
            .and_then(|cause| cause.get("memory_id"))
            .and_then(|id| id.as_str())
            .map(|id| vec![id.to_string(), failure.id.clone()])
            .unwrap_or_default(),
        "causes": promoted,
        "note": "Causes are ranked by causal join (shared entities, backward in time), NOT semantic similarity. A high similarity_rank means a vector search would NOT have surfaced this — that is the point.",
    }))
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

    /// LIVE end-to-end: plant a quiet env-var cause, a semantic distractor, and a
    /// failure into a REAL SQLite store, then run the backfill MCP tool and assert
    /// it surfaces the causal env-var memory by the shared API_TIMEOUT entity —
    /// the root cause a vector search would never rank first. This is the
    /// reproducible receipt behind the demo.
    #[tokio::test]
    async fn live_backfill_surfaces_root_cause_through_storage() {
        let (storage, _dir) = test_storage().await;

        // 1) The quiet cause: an env-var edit (no failure words; not "similar" to a crash).
        //    Backdated 3 days so the backward reach can find it (the demo scenario).
        let cause = storage
            .ingest(IngestInput {
                content: "Set API_TIMEOUT=2 in the deploy env to speed up cold starts".to_string(),
                node_type: "decision".to_string(),
                tags: vec!["API_TIMEOUT".to_string(), "deploy-env".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&cause.id, chrono::Utc::now() - chrono::Duration::days(3))
            .unwrap();

        // 2) A semantic distractor: looks like the crash, but shares NO entity.
        //    Backdated 20 days (also in the past, so only the entity link decides).
        let distractor = storage
            .ingest(IngestInput {
                content: "A 500 Internal Server Error happened in the billing service last month"
                    .to_string(),
                node_type: "event".to_string(),
                tags: vec!["billing-service".to_string()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(
                &distractor.id,
                chrono::Utc::now() - chrono::Duration::days(20),
            )
            .unwrap();

        // 3) The failure, recorded last (most recent) — the "aversive event".
        let failure = storage
            .ingest(IngestInput {
                content: "Service crashed: 500 Internal Server Error on the auth endpoint"
                    .to_string(),
                node_type: "event".to_string(),
                tags: vec![
                    "auth-service".to_string(),
                    "API_TIMEOUT".to_string(),
                    "crash".to_string(),
                ],
                ..Default::default()
            })
            .unwrap();

        // Run the backfill tool against the real store (auto-finds the failure).
        let out = execute(&storage, Some(json!({ "promote": true, "manual": false })))
            .await
            .expect("backfill must run");

        assert_eq!(
            out["triggered"],
            json!(true),
            "the crash must trigger a backfill"
        );
        let causes = out["causes"].as_array().expect("causes array");
        assert!(!causes.is_empty(), "must surface at least one cause");

        // The top cause is the env-var memory, surfaced by the shared API_TIMEOUT entity.
        let top = &causes[0];
        let content = top["content_preview"].as_str().unwrap_or("");
        assert!(
            content.contains("API_TIMEOUT") && content.contains("deploy"),
            "top cause must be the env-var edit, got: {content}"
        );
        let shared = top["shared_entities"].as_array().unwrap();
        assert!(
            shared.iter().any(|e| e.as_str() == Some("api_timeout")),
            "must link via the shared API_TIMEOUT entity, got: {shared:?}"
        );
        assert_eq!(
            top["candidate_edge_persisted"],
            json!(true),
            "the candidate evidence edge must exist before a receipt can claim it"
        );
        assert_eq!(
            out["path_ids"],
            json!([cause.id, failure.id]),
            "the tool must publish the exact backend-authored candidate-to-failure path"
        );
        let edges = storage.get_connections_for_memory(&cause.id).unwrap();
        assert!(
            edges.iter().any(|edge| {
                edge.source_id == cause.id
                    && edge.target_id == failure.id
                    && edge.link_type == "backfill_candidate"
            }),
            "candidate evidence edge must be persisted"
        );
        // It was actually promoted in the real store.
        assert_eq!(
            top["promoted"],
            json!(true),
            "the cause must be promoted in storage"
        );
        // Sanity: the failure we ingested is the one that fired.
        assert_eq!(out["failure"]["id"], json!(failure.id));
    }
}
