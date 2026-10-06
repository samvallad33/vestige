//! Importance Score Tool
//!
//! `maintain(action='importance_score', id=...)` reports how much recorded
//! structure hangs on one memory: the typed edges that touch it and the FSRS
//! reviews the log holds for it. Every number is a count from the log and is
//! returned beside the score it feeds, and nothing is read from the memory's
//! content, tags or name, so the same store and the same id give the same
//! bytes.
//!
//! The v3 tool scored free text instead (`content`): novelty against a
//! learned word model, arousal from an emotion lexicon, combined with
//! weights. That is a word heuristic, and the model it learned made two
//! identical calls answer differently. It runs only in a legacy engine build
//! on a legacy store; a Strata log refuses it and names `id`.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::neuroscience::importance_signals::TEXT_SCORING_COMPILED_IN;
use vestige_core::{ImportanceContext, Storage};

/// Input schema for importance_score tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Full id of the memory to score from its recorded structure: typed-edge counts and FSRS review counts, each returned with the score."
            },
            "content": {
                "type": "string",
                "description": "Legacy engine only: free text to score with the v3 word heuristics. Refused on Strata; pass id."
            },
            "context_topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Legacy engine only, with content: topics for novelty detection."
            },
            "project": {
                "type": "string",
                "description": "Legacy engine only, with content: project name for context."
            }
        }
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportanceArgs {
    id: Option<String>,
    content: Option<String>,
    #[serde(alias = "context_topics")]
    context_topics: Option<Vec<String>>,
    project: Option<String>,
}

/// How the structural score is put together. Returned with every score.
const FORMULA: &str = "edges.total + reviews.count - reviews.lapses";

/// What the counts are, returned with every score.
const STRUCTURE_NOTE: &str = "Counted from the log: the typed edges that touch this memory, and its FSRS reviews (the save itself, each promote, each demote, each dream replay). A lapse is a review rated 1, which is what demote records. No content, tag or name is read.";

/// Why free text is not scored on a Strata log.
const TEXT_SCORING_REFUSAL: &str = "scoring free text needs word heuristics (novelty against a learned word model, an emotion lexicon), which are not Strata operations; pass id, the full id of a memory, to score it from its recorded edges and reviews";

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: ImportanceArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    let id = args
        .id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let has_content = args
        .content
        .as_deref()
        .is_some_and(|content| !content.trim().is_empty());

    if let Some(id) = id {
        if has_content {
            return Err(
                "Pass id or content, not both: id scores a stored memory from its recorded structure; content is the legacy text scorer"
                    .to_string(),
            );
        }
        return structural_score(storage, id);
    }
    if !has_content {
        return Err(
            "importance_score needs id: the full id of the memory to score from its recorded edges and reviews"
                .to_string(),
        );
    }
    if !TEXT_SCORING_COMPILED_IN || crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return Err(super::unavailable::withheld_in_4_0(
            "maintain action 'importance_score' with free-text 'content'",
            TEXT_SCORING_REFUSAL,
        ));
    }
    legacy_text_score(cognitive, args).await
}

/// Score one memory from what the log recorded about it.
fn structural_score(storage: &Arc<Storage>, id: &str) -> Result<Value, String> {
    let node = storage
        .get_node(id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| {
            format!(
                "Memory not found: {id}. importance_score takes a full memory id; recall with handle resolves a tag or an id prefix to one."
            )
        })?;
    let edges = storage
        .get_connections_for_memory(&node.id)
        .map_err(|e| e.to_string())?;

    // A BTreeMap, so the kinds come back in name order every time.
    let mut by_kind: BTreeMap<String, u64> = BTreeMap::new();
    let mut incoming = 0u64;
    let mut outgoing = 0u64;
    for edge in &edges {
        *by_kind.entry(edge.link_type.clone()).or_default() += 1;
        if edge.source_id == node.id {
            outgoing += 1;
        }
        if edge.target_id == node.id {
            incoming += 1;
        }
    }
    let total = edges.len() as u64;
    let reviews = u64::try_from(node.reps).unwrap_or(0);
    let lapses = u64::try_from(node.lapses).unwrap_or(0);
    let score = i64::try_from(total + reviews).unwrap_or(i64::MAX)
        - i64::try_from(lapses).unwrap_or(i64::MAX);
    let latest_receipt = storage
        .get_receipt(&node.id)
        .ok()
        .flatten()
        .map(|receipt| receipt.receipt_id);

    Ok(serde_json::json!({
        "id": node.id,
        "nodeType": node.node_type,
        "basis": "recorded_structure",
        "score": score,
        "formula": FORMULA,
        "computedFrom": {
            "edges": {
                "total": total,
                "incoming": incoming,
                "outgoing": outgoing,
                "byKind": by_kind,
            },
            "reviews": {
                "count": reviews,
                "lapses": lapses,
            },
        },
        "latestReceiptId": latest_receipt,
        "note": STRUCTURE_NOTE,
    }))
}

/// The v3 word-heuristic scorer. Reached only in a legacy engine build on a
/// legacy store.
async fn legacy_text_score(
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: ImportanceArgs,
) -> Result<Value, String> {
    let content = args.content.unwrap_or_default();
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
        .compute_importance(&content, &context);

    // Also detect emotional markers for richer output
    let emotional_markers = cog.arousal_signal.detect_emotional_markers(&content);
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
        "basis": "text_heuristics",
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

/// The structural score on a Strata log (the default build's store).
#[cfg(test)]
mod structural_tests {
    use super::*;
    use serde_json::json;
    use vestige_core::{ConnectionRecord, IngestInput};

    fn store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("data dir");
        let storage = crate::strata_memory::open(dir.path()).expect("strata log");
        (storage, dir)
    }

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    fn save(storage: &Arc<Storage>, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "decision".to_string(),
                ..Default::default()
            })
            .expect("ingest")
            .id
    }

    fn link(storage: &Arc<Storage>, source: &str, target: &str, kind: &str) {
        let at = chrono::DateTime::UNIX_EPOCH;
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: kind.to_string(),
                created_at: at,
                last_activated: at,
                activation_count: 0,
            })
            .expect("edge");
    }

    async fn score(storage: &Arc<Storage>, id: &str) -> Value {
        execute(storage, &cognitive(), Some(json!({ "id": id })))
            .await
            .expect("structural score")
    }

    #[tokio::test]
    async fn free_text_is_refused_and_the_refusal_names_the_exact_handle() {
        let (storage, _dir) = store();
        let error = execute(
            &storage,
            &cognitive(),
            Some(json!({"content": "CRITICAL: production database migration failed!"})),
        )
        .await
        .unwrap_err();
        assert!(error.starts_with("unavailable_in_4_0: "), "{error}");
        assert!(
            error.contains("pass id, the full id of a memory"),
            "{error}"
        );

        // The same refusal through the advertised surface.
        let error = super::super::maintain::execute(
            &storage,
            &cognitive(),
            Some(json!({"action": "importance_score", "content": "anything at all"})),
        )
        .await
        .unwrap_err();
        assert!(error.starts_with("unavailable_in_4_0: "), "{error}");

        let error = execute(&storage, &cognitive(), Some(json!({})))
            .await
            .unwrap_err();
        assert!(error.contains("needs id"), "{error}");
    }

    #[tokio::test]
    async fn the_score_is_counted_from_recorded_edges_and_reviews_with_its_inputs() {
        let (storage, _dir) = store();
        // Same words in both: only recorded structure may tell them apart.
        let hub = save(&storage, "Synthetic decision record.");
        let lone = save(&storage, "Synthetic decision record.");
        let a = save(&storage, "Synthetic evidence A.");
        let b = save(&storage, "Synthetic evidence B.");
        link(&storage, &a, &hub, "derived_from");
        link(&storage, &b, &hub, "derived_from");
        link(&storage, &hub, &lone, "evidence_of");

        let hub_score = score(&storage, &hub).await;
        let node = storage.get_node(&hub).unwrap().unwrap();
        assert_eq!(hub_score["basis"], "recorded_structure");
        assert_eq!(hub_score["nodeType"], "decision");
        assert_eq!(hub_score["formula"], FORMULA);
        assert_eq!(
            hub_score["computedFrom"]["edges"],
            json!({
                "total": 3, "incoming": 2, "outgoing": 1,
                "byKind": {"derived_from": 2, "evidence_of": 1},
            }),
            "{hub_score}"
        );
        assert_eq!(
            hub_score["computedFrom"]["reviews"],
            json!({"count": node.reps, "lapses": node.lapses}),
            "{hub_score}"
        );
        assert_eq!(
            hub_score["score"].as_i64().unwrap(),
            3 + i64::from(node.reps) - i64::from(node.lapses),
            "{hub_score}"
        );
        assert!(
            hub_score["latestReceiptId"]
                .as_str()
                .unwrap()
                .starts_with("eff-"),
            "{hub_score}"
        );
        // No text-derived field survives.
        for gone in [
            "composite",
            "channels",
            "emotionalMarkers",
            "dominantSignal",
        ] {
            assert!(hub_score.get(gone).is_none(), "{gone}: {hub_score}");
        }

        // Identical content, one edge instead of three: the score follows the
        // structure, not the words.
        let lone_score = score(&storage, &lone).await;
        assert_eq!(lone_score["computedFrom"]["edges"]["total"], 1);
        assert_eq!(
            hub_score["score"].as_i64().unwrap() - lone_score["score"].as_i64().unwrap(),
            2
        );

        // A promote is one more recorded review.
        storage.promote_memory(&lone).unwrap();
        let promoted = score(&storage, &lone).await;
        assert_eq!(
            promoted["computedFrom"]["reviews"]["count"]
                .as_u64()
                .unwrap(),
            lone_score["computedFrom"]["reviews"]["count"]
                .as_u64()
                .unwrap()
                + 1
        );
    }

    #[tokio::test]
    async fn the_same_store_and_id_give_the_same_bytes() {
        let (storage, _dir) = store();
        let id = save(&storage, "Synthetic decision record.");
        let other = save(&storage, "Synthetic evidence.");
        link(&storage, &other, &id, "derived_from");
        link(&storage, &id, &other, "evidence_of");

        // No time field to remove: nothing in the answer comes from a clock.
        let first = score(&storage, &id).await.to_string();
        let second = score(&storage, &id).await.to_string();
        assert_eq!(first, second);

        let through_maintain = super::super::maintain::execute(
            &storage,
            &cognitive(),
            Some(json!({"action": "importance_score", "id": id})),
        )
        .await
        .unwrap()
        .to_string();
        assert_eq!(first, through_maintain);
    }

    #[tokio::test]
    async fn an_unknown_id_says_what_handle_to_pass() {
        let (storage, _dir) = store();
        let error = execute(
            &storage,
            &cognitive(),
            Some(json!({"id": "mem-ffffffffffffffff"})),
        )
        .await
        .unwrap_err();
        assert!(error.contains("takes a full memory id"), "{error}");

        let id = save(&storage, "Synthetic decision record.");
        let error = execute(
            &storage,
            &cognitive(),
            Some(json!({"id": id, "content": "and some text"})),
        )
        .await
        .unwrap_err();
        assert!(error.contains("not both"), "{error}");
    }
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    #[test]
    fn test_schema_has_properties() {
        let schema = schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["content"].is_object());
        assert!(schema["properties"]["id"].is_object());
    }

    #[tokio::test]
    async fn test_empty_content_fails() {
        let dir = tempfile::tempdir().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("importance.db"))).unwrap();
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
        let storage = vestige_core::open_storage(Some(dir.path().join("importance.db"))).unwrap();
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
