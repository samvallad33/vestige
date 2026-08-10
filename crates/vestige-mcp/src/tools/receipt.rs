//! Unified receipt inspection and controlled context-ablation replay.

use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use vestige_core::{REPLAY_CLAIM_BOUNDARY, Receipt, ReceiptEvidence, ReplayPrivacyState, Storage};

const COUNTERFACTUAL_REPLAY_SCHEMA: &str =
    "https://vestige.dev/schemas/receipt/counterfactual-replay/v1";

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "replay"],
                "description": "get: inspect one persisted receipt and its safe replay-capsule summary. replay: withhold named receipt-local evidence slots from the exact frozen final context without rerunning retrieval."
            },
            "receipt_id": {
                "type": "string",
                "description": "Source receipt id. For replay this must be a retrieval receipt that has a frozen replay capsule."
            },
            "withheld_slots": {
                "type": "array",
                "items": { "type": "string", "pattern": "^evidence_[1-9][0-9]*$" },
                "uniqueItems": true,
                "description": "[replay] Receipt-local slots to remove from the frozen final context. Search is never rerun and removed candidates are never backfilled."
            }
        },
        "required": ["action", "receipt_id"]
    })
}

#[derive(Debug, Deserialize)]
struct ReceiptArgs {
    action: String,
    #[serde(alias = "receiptId")]
    receipt_id: String,
    #[serde(default, alias = "withheldSlots")]
    withheld_slots: Vec<String>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ReceiptArgs =
        serde_json::from_value(args.ok_or_else(|| "receipt requires arguments".to_string())?)
            .map_err(|error| format!("Invalid receipt arguments: {error}"))?;
    match args.action.as_str() {
        "get" => execute_get(storage, &args.receipt_id),
        "replay" => execute_replay(storage, &args.receipt_id, &args.withheld_slots),
        other => Err(format!("Unknown receipt action '{other}'. Use get|replay.")),
    }
}

fn execute_get(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    let receipt = storage
        .get_receipt(receipt_id)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    let capsule = storage
        .get_retrieval_replay_capsule(receipt_id)
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "action": "get",
        "receipt": receipt,
        "replayCapsule": capsule,
        "claimBoundary": REPLAY_CLAIM_BOUNDARY,
    }))
}

fn execute_replay(
    storage: &Arc<Storage>,
    source_receipt_id: &str,
    withheld_slots: &[String],
) -> Result<Value, String> {
    let durable = storage
        .create_context_ablation_replay(source_receipt_id, withheld_slots)
        .map_err(|error| error.to_string())?;
    if durable.replay.privacy_state != ReplayPrivacyState::Active {
        return Err("Replay evidence is no longer available under current privacy state".into());
    }
    let result = durable
        .replay
        .result
        .clone()
        .ok_or_else(|| "Replay evidence is unavailable".to_string())?;

    let receipt = if let Some(receipt_id) = &durable.replay.receipt_id {
        storage
            .get_receipt(receipt_id)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("Replay receipt '{receipt_id}' is missing"))?
    } else {
        let receipt = Receipt::build(
            Utc::now(),
            "replay",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[result.counterfactual.trust_floor],
            Vec::new(),
        )
        .with_evidence(ReceiptEvidence::CounterfactualReplay {
            schema: COUNTERFACTUAL_REPLAY_SCHEMA.to_string(),
            schema_version: 1,
            replay_id: durable.replay.replay_id.clone(),
            capsule_id: durable.replay.capsule_id.clone(),
            result: result.clone(),
        });
        storage
            .save_counterfactual_replay_receipt(
                &durable.replay.replay_id,
                &receipt,
                None,
                Some("receipt"),
            )
            .map_err(|error| error.to_string())?;
        receipt
    };

    Ok(json!({
        "action": "replay",
        "sourceReceiptId": source_receipt_id,
        "replayId": durable.replay.replay_id,
        "receiptId": receipt.receipt_id,
        "reusedExisting": durable.reused_existing,
        "receipt": receipt,
        "result": result,
        "claimBoundary": REPLAY_CLAIM_BOUNDARY,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_storage() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("receipt-tool.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    #[tokio::test]
    async fn legacy_receipt_gets_but_does_not_replay() {
        let (storage, _dir) = test_storage();
        let receipt = Receipt::build(
            Utc::now(),
            "legacy",
            vec!["memory_legacy".into()],
            Vec::new(),
            Vec::new(),
            &[0.8],
            Vec::new(),
        );
        storage
            .save_receipt(&receipt, None, Some("recall"), None)
            .unwrap();

        let get = execute(
            &storage,
            Some(json!({"action": "get", "receipt_id": receipt.receipt_id})),
        )
        .await
        .unwrap();
        assert!(get["replayCapsule"].is_null());
        assert!(
            execute(
                &storage,
                Some(json!({
                    "action": "replay",
                    "receipt_id": receipt.receipt_id,
                    "withheld_slots": ["evidence_1"]
                })),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn recorded_recall_replays_idempotently_with_typed_noncausal_receipt() {
        let (storage, dir) = test_storage();
        let recall_result = json!({
            "method": "hybrid+cognitive",
            "retrievalMode": "balanced",
            "tokenBudgetLimit": 256,
            "results": [
                {
                    "id": "memory_product_a",
                    "content": "private replay sentinel alpha",
                    "retentionStrength": 0.81
                },
                {
                    "id": "memory_product_b",
                    "content": "private replay sentinel beta",
                    "retentionStrength": 0.55
                }
            ],
            "expandable": ["memory_not_returned"]
        });
        let source = crate::trace_recorder::build_and_save_receipt(
            &storage,
            "run_receipt_product",
            "recall",
            &recall_result,
        )
        .expect("recall should atomically persist receipt and final capsule");
        let source_receipt_id = source["receipt_id"].as_str().unwrap();

        let args = Some(json!({
            "action": "replay",
            "receipt_id": source_receipt_id,
            "withheld_slots": ["evidence_2"]
        }));
        let first = execute(&storage, args.clone()).await.unwrap();
        assert_eq!(first["result"]["baseline"]["itemCount"], 2);
        assert_eq!(first["result"]["counterfactual"]["itemCount"], 1);
        assert_eq!(
            first["result"]["counterfactual"]["orderedSlots"],
            json!(["evidence_1"])
        );
        assert_eq!(first["claimBoundary"], REPLAY_CLAIM_BOUNDARY);
        assert_eq!(
            first["receipt"]["evidence"]["kind"],
            "counterfactual_replay"
        );
        assert_eq!(
            first["receipt"]["evidence"]["predicate"]["result"]["claimBoundary"],
            REPLAY_CLAIM_BOUNDARY
        );

        let public_json = serde_json::to_string(&first).unwrap();
        for forbidden in [
            "private replay sentinel alpha",
            "private replay sentinel beta",
            "memory_product_a",
            "memory_product_b",
            "memory_not_returned",
            "b3k:",
        ] {
            assert!(
                !public_json.contains(forbidden),
                "replay leaked {forbidden}"
            );
        }

        let second = execute(&storage, args).await.unwrap();
        assert_eq!(second["reusedExisting"], true);
        assert_eq!(second["replayId"], first["replayId"]);
        assert_eq!(second["receiptId"], first["receiptId"]);
        let reader = rusqlite::Connection::open(dir.path().join("receipt-tool.db")).unwrap();
        let replay_rows: i64 = reader
            .query_row("SELECT COUNT(*) FROM counterfactual_replays", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(replay_rows, 1);
    }

    #[test]
    fn schema_exposes_only_get_and_controlled_replay() {
        let actions = schema()["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(actions, vec![json!("get"), json!("replay")]);
        assert!(
            schema()["properties"]["withheld_slots"]["description"]
                .as_str()
                .unwrap()
                .contains("never rerun")
        );
    }
}
