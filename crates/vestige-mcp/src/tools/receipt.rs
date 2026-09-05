//! Unified receipt inspection and controlled context-ablation replay.

use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use vestige_core::storage::ReceiptAttestationStatus;
use vestige_core::{
    BACKFILL_RECEIPT_CLAIM_BOUNDARY, REPLAY_CLAIM_BOUNDARY, Receipt, ReceiptEvidence,
    ReplayPrivacyState, SYNAPTIC_CAPTURE_CLAIM_BOUNDARY, Storage,
};

const COUNTERFACTUAL_REPLAY_SCHEMA: &str =
    "https://vestige.dev/schemas/receipt/counterfactual-replay/v1";

const LEGACY_RECEIPT_CLAIM_BOUNDARY: &str = concat!(
    "This legacy receipt records retrieval metadata only; ",
    "it carries no controlled-replay or synaptic-capture claim."
);

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "replay"],
                "description": "'get': one persisted receipt with its safe replay-capsule summary. 'replay': withhold named evidence slots from the frozen final context without rerunning retrieval."
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
        "required": ["action", "receipt_id"],
        "additionalProperties": false,
        "oneOf": [
            {
                "properties": { "action": { "const": "get" } },
                "not": { "required": ["withheld_slots"] }
            },
            {
                "properties": { "action": { "const": "replay" } }
            }
        ]
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptArgs {
    action: String,
    #[serde(alias = "receiptId")]
    receipt_id: String,
    #[serde(alias = "withheldSlots")]
    withheld_slots: Option<Vec<String>>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: ReceiptArgs =
        serde_json::from_value(args.ok_or_else(|| "receipt requires arguments".to_string())?)
            .map_err(|error| format!("Invalid receipt arguments: {error}"))?;
    validate_args(&args)?;
    match args.action.as_str() {
        "get" => execute_get(storage, &args.receipt_id),
        "replay" => execute_replay(
            storage,
            &args.receipt_id,
            args.withheld_slots.as_deref().unwrap_or(&[]),
        ),
        other => Err(format!("Unknown receipt action '{other}'. Use get|replay.")),
    }
}

fn validate_args(args: &ReceiptArgs) -> Result<(), String> {
    if args.receipt_id.trim().is_empty() {
        return Err("receipt_id must not be empty".into());
    }
    if args.action == "get" && args.withheld_slots.is_some() {
        return Err("withheld_slots is only valid for action='replay'".into());
    }
    Ok(())
}

fn claim_boundary_for_receipt(receipt: &Receipt) -> &'static str {
    match &receipt.evidence {
        Some(ReceiptEvidence::CounterfactualReplay { .. }) => REPLAY_CLAIM_BOUNDARY,
        Some(ReceiptEvidence::SynapticCapture(_)) => SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
        Some(ReceiptEvidence::Backfill { .. }) => BACKFILL_RECEIPT_CLAIM_BOUNDARY,
        None => LEGACY_RECEIPT_CLAIM_BOUNDARY,
    }
}

fn safe_storage_error(operation: &str, error: &impl std::fmt::Display) -> String {
    tracing::warn!(%error, "receipt storage operation failed: {operation}");
    format!("Receipt {operation} is temporarily unavailable")
}

fn execute_get(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    let receipt = storage
        .get_receipt(receipt_id)
        .map_err(|error| safe_storage_error("lookup", &error))?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    let capsule = storage
        .get_retrieval_replay_capsule(receipt_id)
        .map_err(|error| safe_storage_error("lookup", &error))?;
    let claim_boundary = claim_boundary_for_receipt(&receipt);
    let attestation = receipt_attestation_view(storage, receipt_id)?;
    Ok(json!({
        "action": "get",
        "receipt": receipt,
        "replayCapsule": capsule,
        "claimBoundary": claim_boundary,
        "attestation": attestation,
    }))
}

/// Present cryptographic receipt state without treating local database row
/// checks as an external timestamp or non-equivocation proof.
fn receipt_attestation_view(storage: &Arc<Storage>, receipt_id: &str) -> Result<Value, String> {
    let status = storage
        .receipt_attestation_status(receipt_id)
        .map_err(|error| safe_storage_error("attestation lookup", &error))?
        .ok_or_else(|| format!("Receipt '{receipt_id}' was not found"))?;
    if status == ReceiptAttestationStatus::LegacyUnsigned {
        return Ok(json!({
            "status": "legacy_unsigned",
            "verification": {
                "locallyVerified": false,
                "claimBoundary": "No DSSE envelope exists for this legacy receipt."
            }
        }));
    }
    let envelope = storage
        .get_receipt_attestation_envelope(receipt_id)
        .map_err(|error| safe_storage_error("attestation envelope lookup", &error))?
        .ok_or_else(|| "Signed receipt is missing its immutable DSSE envelope".to_string())?;
    let verification = storage
        .verify_stored_receipt_attestation(receipt_id)
        .map_err(|error| safe_storage_error("attestation verification", &error))?
        .ok_or_else(|| "Signed receipt is missing its verifiable DSSE state".to_string())?;
    let locally_verified = verification.is_valid();
    let report = verification.report;
    Ok(json!({
        "status": "signed_v1",
        "envelope": envelope,
        "verification": {
            "locallyVerified": locally_verified,
            "signatureValid": report.signature_valid,
            "canonicalPayload": report.canonical_payload,
            "receiptBindingValid": verification.receipt_binding_valid,
            "localChainRowMatched": report.anchored,
            "predecessorRowMatched": report.predecessor_anchored,
            "terminalHeadRowMatched": report.terminal_head_matched,
            "verifiedKeyId": report.verified_key_id,
            "verifiedPublicKeyFingerprint": report.verified_public_key_fingerprint,
            "keyValidity": report.key_validity.map(|value| format!("{value:?}").to_lowercase()),
            "failures": report.failures.iter().map(|failure| format!("{failure:?}")).collect::<Vec<_>>(),
            "warnings": report.warnings.iter().map(|warning| format!("{warning:?}")).collect::<Vec<_>>(),
            "claimBoundary": "Local verification checks immutable rows and registered keys; it does not establish external anchoring, trusted time, truth, completeness, or non-equivocation."
        }
    }))
}

fn linked_replay_receipt(
    storage: &Arc<Storage>,
    replay_id: &str,
) -> Result<Option<Receipt>, String> {
    let replay = storage
        .get_context_ablation_replay(replay_id)
        .map_err(|error| safe_storage_error("recovery", &error))?;
    let Some(replay) = replay else {
        return Ok(None);
    };
    let Some(receipt_id) = replay.receipt_id else {
        return Ok(None);
    };
    storage
        .get_receipt(&receipt_id)
        .map_err(|error| safe_storage_error("recovery", &error))?
        .ok_or_else(|| "Replay receipt link is incomplete; retry the replay".to_string())
        .map(Some)
}

fn execute_replay(
    storage: &Arc<Storage>,
    source_receipt_id: &str,
    withheld_slots: &[String],
) -> Result<Value, String> {
    let durable = storage
        .create_context_ablation_replay(source_receipt_id, withheld_slots)
        .map_err(|error| safe_storage_error("creation", &error))?;
    if durable.replay.privacy_state != ReplayPrivacyState::Active {
        return Err("Replay evidence is no longer available under current privacy state".into());
    }
    let result = durable
        .replay
        .result
        .clone()
        .ok_or_else(|| "Replay evidence is unavailable".to_string())?;

    let (receipt, reused_existing_receipt) = if let Some(receipt_id) = &durable.replay.receipt_id {
        storage
            .get_receipt(receipt_id)
            .map_err(|error| safe_storage_error("lookup", &error))?
            .ok_or_else(|| "Replay receipt link is incomplete; retry the replay".to_string())
            .map(|receipt| (receipt, true))?
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
        match storage.save_counterfactual_replay_receipt(
            &durable.replay.replay_id,
            &receipt,
            None,
            Some("receipt"),
        ) {
            Ok(()) => (receipt, false),
            Err(error) => {
                tracing::warn!(
                    %error,
                    replay_id = %durable.replay.replay_id,
                    "replay receipt persistence failed; checking for a concurrently linked receipt"
                );
                linked_replay_receipt(storage, &durable.replay.replay_id)?
                    .ok_or_else(|| {
                        "Replay receipt could not be persisted or recovered; retry the replay"
                            .to_string()
                    })
                    .map(|receipt| (receipt, true))?
            }
        }
    };

    Ok(json!({
        "action": "replay",
        "sourceReceiptId": source_receipt_id,
        "replayId": durable.replay.replay_id,
        "receiptId": receipt.receipt_id,
        "reusedExisting": durable.reused_existing || reused_existing_receipt,
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
        assert_eq!(
            get["claimBoundary"], LEGACY_RECEIPT_CLAIM_BOUNDARY,
            "legacy receipts must not inherit the replay claim"
        );
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
    async fn get_rejects_replay_only_arguments() {
        let (storage, _dir) = test_storage();
        let error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "withheld_slots": ["evidence_1"]
            })),
        )
        .await
        .unwrap_err();
        assert_eq!(error, "withheld_slots is only valid for action='replay'");

        let empty_error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "withheld_slots": []
            })),
        )
        .await
        .unwrap_err();
        assert_eq!(
            empty_error,
            "withheld_slots is only valid for action='replay'"
        );
    }

    #[tokio::test]
    async fn receipt_arguments_reject_unknown_fields() {
        let (storage, _dir) = test_storage();
        let error = execute(
            &storage,
            Some(json!({
                "action": "get",
                "receipt_id": "r_test",
                "unexpected": true
            })),
        )
        .await
        .unwrap_err();
        assert!(error.contains("unknown field `unexpected`"));
    }

    #[test]
    fn claim_boundary_tracks_typed_receipt_evidence() {
        let legacy = Receipt::build(
            Utc::now(),
            "legacy",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[],
            Vec::new(),
        );
        assert_eq!(
            claim_boundary_for_receipt(&legacy),
            LEGACY_RECEIPT_CLAIM_BOUNDARY
        );

        let synaptic = legacy.with_evidence(ReceiptEvidence::SynapticCapture(
            vestige_core::SynapticCaptureEvidence {
                schema: "https://vestige.dev/schemas/receipt/synaptic-capture/v1".into(),
                schema_version: 1,
                algorithm_version: "test".into(),
                receipt_role: None,
                parent_receipt_id: None,
                evaluation_direction: None,
                trigger: vestige_core::SynapticCaptureTrigger {
                    event_id: "event_test".into(),
                    memory_id: "memory_test".into(),
                    event_type: "test".into(),
                    occurred_at: Utc::now(),
                    importance_score: 0.9,
                },
                capture_window: vestige_core::SynapticCaptureWindow {
                    evaluation_direction: "backward".into(),
                    backward_hours: 1.0,
                    forward_hours: 0.0,
                    tag_lifetime_hours: 1.0,
                    minimum_tag_strength: 0.0,
                    minimum_association_score: None,
                    maximum_captures: 1,
                    decay_function: "exponential".into(),
                    context_threshold: None,
                    context_algorithm_version: None,
                },
                candidates: Vec::new(),
                claim_boundary: "untrusted caller text is ignored".into(),
            },
        ));
        assert_eq!(
            claim_boundary_for_receipt(&synaptic),
            SYNAPTIC_CAPTURE_CLAIM_BOUNDARY,
            "the public boundary must be the canonical typed-evidence boundary"
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

        let replay_receipt = execute(
            &storage,
            Some(json!({"action": "get", "receipt_id": first["receiptId"]})),
        )
        .await
        .unwrap();
        assert_eq!(replay_receipt["claimBoundary"], REPLAY_CLAIM_BOUNDARY);
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
        assert_eq!(schema()["additionalProperties"], false);
    }
}
