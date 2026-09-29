//! `suppress` MCP Tool (v2.0.5) — Top-Down Active Forgetting
//!
//! Actively suppress a memory via top-down inhibitory control. Distinct from
//! `memory.delete` (which removes the row) and `memory.demote` (which is a
//! one-shot thumb-down). Each call compounds: suppression_count increments,
//! FSRS state is dealt a strong blow, and a background Rac1 cascade worker
//! (in the existing consolidation loop) will fade co-activated neighbors.
//!
//! Reversible within a 24-hour labile window via `reverse: true`.
//!
//! References:
//! - Anderson et al. (2025). Brain mechanisms underlying the inhibitory
//!   control of thought. Nat Rev Neurosci. DOI 10.1038/s41583-025-00929-y
//! - Cervantes-Sandoval & Davis (2020). Rac1 Impairs Forgetting-Induced
//!   Cellular Plasticity. Front Cell Neurosci. PMC7477079

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::Storage;
use vestige_core::neuroscience::active_forgetting::{ActiveForgettingSystem, DEFAULT_LABILE_HOURS};

/// Input schema for the `suppress` tool.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Top-down suppression (Anderson 2025 SIF, Davis Rac1): the memory persists but is inhibited from retrieval and decays faster. Each call compounds. A background worker spreads accelerated decay to co-activated neighbours over 72 hours. Local state is reversible within 24 hours when its snapshot still matches. Journaled cascade effects are reversed atomically when neighbor state still matches.",
        "properties": {
            "id": {
                "type": "string",
                "description": "Memory UUID to suppress (or reverse-suppress)"
            },
            "reason": {
                "type": "string",
                "description": "Optional note on why; logged."
            },
            "reverse": {
                "type": "boolean",
                "default": false,
                "description": "If true, reverse a previous suppression. Requires a matching snapshot within the 24-hour labile window; later state changes and legacy suppressions need review. Includes journaled neighbor cascades when their state still matches."
            },
            "cascade_derived_from": {
                "type": "boolean",
                "default": false,
                "description": "If true, extend this suppression along derived_from edges (blast traversal, depth cap 5, exact, cycle-safe) before suppressing. Each derived target is routed through the same Memory-PR review gate as a direct suppress: Fast mode suppresses them directly; Risk-Gated/Paranoid opens one pending Memory PR per target and suppresses NOTHING until approved. Ignored by reverse=true."
            }
        },
        "required": ["id"]
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SuppressArgs {
    id: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    reverse: bool,
    // Primary wire name is camelCase like the siblings; the snake_case
    // spelling (the name used in the schema description) is accepted too.
    #[serde(default, alias = "cascade_derived_from")]
    cascade_derived_from: bool,
}

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SuppressArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.id.trim().is_empty() {
        return Err("'id' must not be empty".to_string());
    }
    // Basic UUID sanity check — don't reject if missing, but warn
    if !crate::tools::memory_unified::is_memory_id(&args.id) {
        return Err(format!("Invalid memory ID format: {}", args.id));
    }

    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return execute_strata(storage, cognitive, &args).await;
    }

    let sys = ActiveForgettingSystem::new();

    if args.reverse {
        // Reverse path — only allowed within labile window.
        match storage.reverse_suppression(&args.id, sys.labile_hours) {
            Ok(node) => {
                let still_suppressed = node.suppression_count > 0;
                Ok(json!({
                    "success": true,
                    "action": "reverse",
                    "id": args.id,
                    "suppressionCount": node.suppression_count,
                    "stillSuppressed": still_suppressed,
                    "reversalScope": "local_state_and_journaled_cascades",
                    "journaledCascadeReversal": "atomic",
                    "unrecordedEffectsReversed": false,
                    "retentionStrength": node.retention_strength,
                    "retrievalStrength": node.retrieval_strength,
                    "stability": node.stability,
                    "message": if still_suppressed {
                        format!(
                            "Reversal applied. {} suppression(s) remain on this memory.",
                            node.suppression_count
                        )
                    } else {
                        "Local suppression reversed. Journaled cascade effects are reversed atomically when neighbor state still matches.".to_string()
                    },
                }))
            }
            Err(e) => Err(format!("Reverse failed: {}", e)),
        }
    } else {
        // Forward path — suppress + log reason + tell the user what will happen.
        let before_count = storage
            .get_node(&args.id)
            .map_err(|e| format!("Failed to load memory: {}", e))?
            .map(|n| n.suppression_count)
            .unwrap_or(0);

        // Optional derived_from cascade: compute the exact blast set FIRST
        // (scope before effect — the traversal happens before ANY
        // suppression), then route every derived target through the same
        // Memory-PR review gate a direct suppress hits (#296 purge
        // discipline). The primary id below is gated by the server pre-gate;
        // the cascade ids are new victims and are gated here, per id:
        // Fast mode executes directly, Risk-Gated/Paranoid holds each behind
        // a pending Memory PR and suppresses nothing until approved.
        let cascade = if args.cascade_derived_from {
            Some(derive_and_gate_cascade(storage, &args.id, args.reason.as_deref()).await?)
        } else {
            None
        };

        let node = storage
            .suppress_memory(&args.id)
            .map_err(|e| format!("Suppress failed: {}", e))?;

        // Count how many neighbors will be cascaded over the coming 72h.
        // We don't run the cascade synchronously — it happens in the
        // background consolidation loop via `run_rac1_cascade_sweep`. But we
        // can give the user an estimate.
        let edges = storage
            .get_connections_for_memory(&args.id)
            .unwrap_or_default();
        let estimated_cascade = edges.len().min(100);

        let reversible_until = node
            .suppressed_at
            .map(|t| sys.reversible_until(t))
            .unwrap_or_else(chrono::Utc::now);
        let retrieval_penalty = sys.retrieval_penalty(node.suppression_count);

        tracing::info!(
            id = %args.id,
            count = node.suppression_count,
            reason = args.reason.as_deref().unwrap_or(""),
            "Memory suppressed"
        );

        Ok(json!({
            "success": true,
            "action": "suppress",
            "id": args.id,
            "suppressionCount": node.suppression_count,
            "priorCount": before_count,
            "retrievalPenalty": retrieval_penalty,
            "retentionStrength": node.retention_strength,
            "retrievalStrength": node.retrieval_strength,
            "stability": node.stability,
            "estimatedCascadeNeighbors": estimated_cascade,
            "cascadeDerivedFrom": cascade,
            "reversibleUntil": reversible_until.to_rfc3339(),
            "labileWindowHours": DEFAULT_LABILE_HOURS,
            "reason": args.reason,
            "message": format!(
                "Actively forgetting. Suppression #{} applied. ~{} co-activated neighbors will fade over the next 72h via Rac1 cascade. Reversible for {}h.",
                node.suppression_count, estimated_cascade, DEFAULT_LABILE_HOURS
            ),
            "citation": "Anderson et al. 2025, Nat Rev Neurosci, DOI: 10.1038/s41583-025-00929-y"
        }))
    }
}

/// Strata suppress is one allowed RETIRE under the named `suppress` rule.
/// The node stays on the log and drops out of retrieval. Compounding counts
/// and reversal are not recorded ops.
async fn execute_strata(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: &SuppressArgs,
) -> Result<Value, String> {
    if args.reverse {
        return Err(
            "reverse is not admitted on the Strata log; suppression is one RETIRE".into(),
        );
    }
    if args.cascade_derived_from {
        return Err("cascade_derived_from is not admitted on the Strata log".into());
    }
    let node = storage
        .suppress_memory(&args.id)
        .map_err(|e| format!("Suppress failed: {}", e))?;
    let receipt_id = node.source.clone().unwrap_or_default();
    if !receipt_id.starts_with("eff-") {
        return Err("Suppress failed: retire returned no effect receipt".into());
    }
    tracing::info!(
        id = %args.id,
        receipt = %receipt_id,
        reason = args.reason.as_deref().unwrap_or(""),
        "Memory suppressed"
    );
    // Same drop as purge: the process graph was hydrated before the RETIRE.
    let mut rebuilt = CognitiveEngine::new();
    rebuilt.hydrate(storage);
    *cognitive.lock().await = rebuilt;
    Ok(json!({
        "success": true,
        "action": "suppress",
        "id": args.id,
        "nodeId": args.id,
        "receiptId": receipt_id,
        "rule": "suppress",
        "reason": args.reason,
        "message": "Suppressed; can't be retrieved.",
    }))
}

/// Exact derived_from blast from `id`, each target gated like a direct
/// suppress. Returns a JSON summary for the tool response; suppression of
/// held targets does NOT happen until their Memory PR is approved.
async fn derive_and_gate_cascade(
    storage: &Arc<Storage>,
    id: &str,
    reason: Option<&str>,
) -> Result<Value, String> {
    let report = storage
        .blast_radius_with_link_types(id, false, &["derived_from"])
        .map_err(|e| format!("cascade traversal failed: {}", e))?;
    let targets: Vec<_> = report
        .affected
        .into_iter()
        .filter(|a| a.id != id)
        .collect();

    let mode = crate::trace_recorder::read_review_mode(storage);
    let mut entries = Vec::with_capacity(targets.len());
    let mut allowed: Vec<String> = Vec::new();
    let mut held = 0usize;

    for target in targets {
        let gate_args = Some(json!({"id": target.id, "reason": reason}));
        match crate::trace_recorder::gate_pending_memory_mutation(
            storage,
            None,
            "suppress_cascade_derived_from",
            "suppress",
            &gate_args,
            mode,
        ) {
            Ok(None) => {
                allowed.push(target.id.clone());
                entries.push(json!({
                    "id": target.id,
                    "via": target.via,
                    "depth": target.depth,
                    "outcome": "suppressed",
                }));
            }
            Ok(Some(pending)) => {
                held += 1;
                entries.push(json!({
                    "id": target.id,
                    "via": target.via,
                    "depth": target.depth,
                    "outcome": "pending_review",
                    "requiresReview": true,
                    "memoryPr": pending["memoryPrsOpened"][0]["id"],
                }));
            }
            Err(error) => {
                // Fail closed: a broken gate must not degrade into an
                // ungated cascade suppression.
                held += 1;
                entries.push(json!({
                    "id": target.id,
                    "via": target.via,
                    "depth": target.depth,
                    "outcome": "gate_error",
                    "requiresReview": true,
                    "error": error,
                }));
            }
        }
    }

    if !allowed.is_empty() {
        let refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
        for outcome in storage.retire_affected(&refs, reason.unwrap_or("derived_from cascade")) {
            let entry = entries
                .iter_mut()
                .find(|e| e["id"] == json!(outcome.id))
                .expect("outcome id came from allowed set");
            if !outcome.suppressed {
                entry["outcome"] = json!("failed");
                entry["error"] = json!(outcome.error);
            }
        }
    }

    Ok(json!({
        "linkType": "derived_from",
        "targets": entries,
        "suppressed": entries.len() - held,
        "heldForReview": held,
        "note": if held > 0 {
            "Held targets were NOT suppressed; approve their Memory PRs (forget) or keep them (promote)."
        } else {
            "All derived_from targets suppressed through the existing suppression mechanism (no deletions)."
        },
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    async fn run(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        execute(storage, &cognitive, args).await
    }

    fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    fn ingest(storage: &Arc<Storage>, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["test".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    #[test]
    fn test_schema_is_valid() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["id"].is_object());
        assert!(s["properties"]["reverse"].is_object());
        assert_eq!(s["required"][0], "id");
    }

    #[tokio::test]
    async fn test_suppress_missing_args() {
        let (storage, _dir) = test_storage();
        let result = run(&storage, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_suppress_invalid_uuid() {
        let (storage, _dir) = test_storage();
        let args = json!({"id": "not-a-uuid"});
        let result = run(&storage, Some(args)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid memory ID"));
    }

    #[tokio::test]
    async fn test_suppress_increments_count() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake is my roommate");

        // First call
        let r1 = run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        assert_eq!(r1["suppressionCount"], 1);
        assert_eq!(r1["priorCount"], 0);

        // Second call — compounds
        let r2 = run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        assert_eq!(r2["suppressionCount"], 2);
        assert_eq!(r2["priorCount"], 1);
    }

    #[tokio::test]
    async fn test_suppress_applies_fsrs_penalty() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        let before = storage.get_node(&id).unwrap().unwrap();
        let result = run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Stability should be heavily reduced
        let after_stability = result["stability"].as_f64().unwrap();
        assert!(after_stability < before.stability);
        // Retention should be reduced
        let after_retention = result["retentionStrength"].as_f64().unwrap();
        assert!(after_retention < before.retention_strength);
    }

    #[tokio::test]
    async fn test_suppress_is_not_delete() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Memory must still be retrievable via get_node
        let node = storage.get_node(&id).unwrap();
        assert!(node.is_some(), "Suppressed memory must still exist");
        assert_eq!(node.unwrap().suppression_count, 1);
    }

    #[tokio::test]
    async fn test_reverse_within_window_decrements() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();
        run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        // Now reverse — count should drop from 2 to 1
        let r = run(&storage, Some(json!({"id": id.clone(), "reverse": true})))
            .await
            .unwrap();
        assert_eq!(r["suppressionCount"], 1);
        assert_eq!(r["stillSuppressed"], true);

        // Reverse again — should go to 0
        let r = run(&storage, Some(json!({"id": id.clone(), "reverse": true})))
            .await
            .unwrap();
        assert_eq!(r["suppressionCount"], 0);
        assert_eq!(r["stillSuppressed"], false);
    }

    #[tokio::test]
    async fn test_reverse_without_prior_suppression_fails() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Fresh memory");

        let result = run(&storage, Some(json!({"id": id.clone(), "reverse": true}))).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no active suppression"));
    }

    #[tokio::test]
    async fn test_suppress_records_timestamp() {
        let (storage, _dir) = test_storage();
        let id = ingest(&storage, "Jake");

        run(&storage, Some(json!({"id": id.clone()})))
            .await
            .unwrap();

        let node = storage.get_node(&id).unwrap().unwrap();
        assert!(node.suppressed_at.is_some(), "suppressed_at must be set");
    }

    // ================================================================
    // cascade_derived_from — blast traversal + per-id review gate
    // ================================================================

    fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 0.8,
                link_type: link_type.to_string(),
                created_at: chrono::Utc::now(),
                last_activated: chrono::Utc::now(),
                activation_count: 0,
            })
            .unwrap();
    }

    fn set_mode(storage: &Arc<Storage>, mode: &str) {
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            json!({"mode": mode}).to_string(),
        )
        .unwrap();
    }

    fn seed_derived_chain(storage: &Arc<Storage>) -> (String, String, String, String) {
        let root = ingest(storage, "root decision");
        let child = ingest(storage, "derived summary");
        let grandchild = ingest(storage, "derived postmortem");
        let unrelated = ingest(storage, "unrelated memory");
        link(storage, &root, &child, "derived_from");
        link(storage, &child, &grandchild, "derived_from");
        // a non-derived edge type must NOT be pulled in by the cascade
        link(storage, &root, &unrelated, "backfill_candidate");
        (root, child, grandchild, unrelated)
    }

    #[tokio::test]
    async fn test_cascade_fast_mode_suppresses_derived_chain() {
        let (storage, _dir) = test_storage();
        set_mode(&storage, "fast");
        let (root, child, grandchild, unrelated) = seed_derived_chain(&storage);

        let r = run(
            &storage,
            Some(json!({"id": root.clone(), "cascade_derived_from": true, "reason": "bad derivation"})),
        )
        .await
        .unwrap();

        let cascade = r["cascadeDerivedFrom"].as_object().unwrap();
        assert_eq!(cascade["suppressed"], 2);
        assert_eq!(cascade["heldForReview"], 0);

        for id in [&root, &child, &grandchild] {
            let node = storage.get_node(id).unwrap().unwrap();
            assert_eq!(node.suppression_count, 1, "{id} must be suppressed");
        }
        let stranger = storage.get_node(&unrelated).unwrap().unwrap();
        assert_eq!(stranger.suppression_count, 0, "non-derived edges must not cascade");
    }

    #[tokio::test]
    async fn test_cascade_risk_gated_holds_targets_behind_prs() {
        use vestige_core::MemoryPrStatus;

        let (storage, _dir) = test_storage();
        set_mode(&storage, "risk_gated");
        let (root, child, grandchild, _unrelated) = seed_derived_chain(&storage);

        // Direct execute() call: the server pre-gate would normally hold the
        // primary suppress itself; here we verify the per-id gate on the
        // cascade targets.
        let r = run(
            &storage,
            Some(json!({"id": root.clone(), "cascade_derived_from": true, "reason": "review me"})),
        )
        .await
        .unwrap();

        let cascade = r["cascadeDerivedFrom"].as_object().unwrap();
        assert_eq!(cascade["heldForReview"], 2);
        assert_eq!(cascade["suppressed"], 0);

        // Cascade targets were NOT suppressed; each has a pending PR.
        for id in [&child, &grandchild] {
            assert_eq!(
                storage.get_node(id).unwrap().unwrap().suppression_count, 0,
                "cascade target must wait for review"
            );
        }
        let prs = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(prs.len(), 2, "one PR per derived target");
        assert!(prs.iter().all(|pr| pr.diff["pendingAction"] == json!("suppress")));
    }

    #[tokio::test]
    async fn test_cascade_default_off() {
        let (storage, _dir) = test_storage();
        set_mode(&storage, "fast");
        let (root, child, _grandchild, _unrelated) = seed_derived_chain(&storage);

        let r = run(&storage, Some(json!({"id": root.clone()})))
            .await
            .unwrap();
        assert!(r["cascadeDerivedFrom"].is_null(), "no cascade unless asked");
        assert_eq!(
            storage.get_node(&child).unwrap().unwrap().suppression_count, 0,
            "default suppress must not touch derived targets"
        );
    }
}
