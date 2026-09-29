//! # Blast Radius — MCP tool
//!
//! Exact downstream reach of a cause/source memory: BFS over recorded
//! lineage edges (`derived_from`, `backfill_candidate`, `evidence_of`),
//! SOURCE -> TARGET only, plus depth-0 siblings sharing the root's
//! `commit <sha>` line. Exact traversal, zero scoring — the edges are
//! hypotheses recorded by `backfill`/lifecycle, and this tool only measures
//! how far they reach.
//!
//! Two actions:
//! - `report` (default, read-only): root_id or commit_sha prefix in, blast
//!   report out.
//! - `retire` (explicit, destructive-adjacent): suppress each affected id
//!   through the EXISTING review-gated path. Every id is routed through
//!   `gate_pending_memory_mutation` exactly like a direct `suppress` call
//!   (#296's purge discipline): under Risk-Gated/Paranoid review a pending
//!   Memory PR is opened per id and NOTHING is suppressed until approved;
//!   Fast mode keeps the historical direct-suppression behavior. Retire is
//!   never a delete — it flips the existing suppression mechanism
//!   (`retire_affected` -> `suppress_memory`), so rows survive, the
//!   operation is journaled, and the 24h labile reversal window applies.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::{Storage, BLAST_LINK_TYPES, BLAST_MAX_DEPTH};

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Exact blast radius of a cause/source memory: BFS over lineage edges (derived_from, backfill_candidate, evidence_of), source->target direction, depth cap 5, cycle-safe; records sharing the root's commit sha are depth-0 siblings. action='report' (default) is read-only. action='retire' suppresses listed ids through the same Memory-PR review gate as suppress/purge (never deletes).",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["report", "retire"],
                "default": "report",
                "description": "report: read-only reach. retire: explicit suppression of listed ids, each routed through the review gate."
            },
            "root_id": {
                "type": "string",
                "description": "Memory UUID of the cause/source record (the SOURCE side of recorded lineage edges). Required for report unless commit_sha is given."
            },
            "commit_sha": {
                "type": "string",
                "description": "Commit sha prefix (>= 6 hex chars); resolves to the newest record whose 'commit <sha>' line matches. Alternative to root_id."
            },
            "open_only": {
                "type": "boolean",
                "default": true,
                "description": "Filter non-root entries to valid_until IS NULL or > now. Default true."
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required for action='retire': memory ids to retire (e.g. from a prior report). Each is gated and suppressed individually."
            },
            "reason": {
                "type": "string",
                "description": "Required for action='retire': why the blast is being retired; recorded on the pending Memory PRs and the retire journal."
            }
        }
    })
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Args {
    #[serde(default)]
    action: Action,
    root_id: Option<String>,
    commit_sha: Option<String>,
    #[serde(default = "default_true")]
    open_only: bool,
    ids: Option<Vec<String>>,
    reason: Option<String>,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Action {
    #[default]
    Report,
    Retire,
}

fn default_true() -> bool {
    true
}

const RETIRE_NOTICE: &str = "Retire suppresses through the review-gated path (Memory PRs, not deletions): held ids stay live until their PR is approved with `forget`, kept with `promote`, or held suppressed with `quarantine`. Fast mode executes directly, matching `suppress`.";

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    match args.action {
        Action::Report => report(storage, args).await,
        Action::Retire => retire(storage, args).await,
    }
}

async fn report(storage: &Arc<Storage>, args: Args) -> Result<Value, String> {
    let trimmed_root = args
        .root_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let trimmed_sha = args
        .commit_sha
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let (root_id, resolved_via) = match (trimmed_root, trimmed_sha) {
        (Some(id), _) => (id.to_string(), "root_id".to_string()),
        (None, Some(sha)) => {
            let id = storage
                .resolve_commit_sha_root(sha)
                .map_err(|e| format!("commit_sha resolution failed: {}", e))?
                .ok_or_else(|| {
                    format!("no record has a 'commit <sha>' line starting with '{sha}'")
                })?;
            (id, format!("commit_sha:{sha}"))
        }
        (None, None) => {
            return Err("provide root_id or commit_sha".to_string());
        }
    };

    let blast = storage
        .blast_radius(&root_id, args.open_only)
        .map_err(|e| format!("blast radius failed: {}", e))?;

    let commit_sha = storage
        .get_node(&root_id)
        .ok()
        .flatten()
        .and_then(|node| vestige_core::commit_sha_of(&node.content));

    Ok(json!({
        "tool": "blast_radius",
        "action": "report",
        "rootId": blast.root_id,
        "resolvedVia": resolved_via,
        "commitSha": commit_sha,
        "openOnly": args.open_only,
        "maxDepth": BLAST_MAX_DEPTH,
        "linkTypes": BLAST_LINK_TYPES,
        "total": blast.total,
        "affected": blast
            .affected
            .iter()
            .map(|a| json!({"id": a.id, "via": a.via, "depth": a.depth}))
            .collect::<Vec<_>>(),
        "traversal": "exact BFS over source->target lineage edges; no scoring; edges are recorded hypotheses, not proven causes",
        "retireHint": "retire is a separate explicit action: {action:'retire', ids:[...], reason:'...'} — every id goes through the Memory-PR review gate before suppression.",
    }))
}

async fn retire(storage: &Arc<Storage>, args: Args) -> Result<Value, String> {
    let ids = args.ids.clone().ok_or_else(|| {
        "action='retire' requires 'ids' (memory ids, e.g. from a prior blast report)".to_string()
    })?;
    if ids.is_empty() {
        return Err("'ids' must not be empty".to_string());
    }
    let reason = args
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .ok_or_else(|| "action='retire' requires 'reason'".to_string())?
        .to_string();

    for id in &ids {
        if uuid::Uuid::parse_str(id.trim()).is_err() {
            return Err(format!("Invalid memory ID format: {}", id));
        }
    }

    // Gate discipline (#296 purge pattern): route EVERY id through the same
    // pre-call review gate a direct `suppress` call hits, by name. The gate
    // is tool-layer (`gate_pending_memory_mutation` in trace_recorder) and
    // wraps this storage-level suppress path, so calling it per id here is
    // the mirror of how standalone `purge` was wired. Ok(None) => proceed
    // (Fast mode or not review-class); Ok(Some) => held behind a Memory PR,
    // do NOT suppress; Err => fail closed.
    let mode = crate::trace_recorder::read_review_mode(storage);
    let run_id = "blast_radius_retire";

    let mut results = Vec::with_capacity(ids.len());
    let mut to_execute = Vec::new();
    let mut held_ids = Vec::new();

    for id in ids.iter().map(|id| id.trim().to_string()).collect::<Vec<_>>() {
        let gate_args = Some(json!({"id": id, "reason": reason}));
        match crate::trace_recorder::gate_pending_memory_mutation(
            storage,
            None,
            run_id,
            "suppress",
            &gate_args,
            mode,
        ) {
            Ok(None) => to_execute.push(id),
            Ok(Some(held)) => {
                let pr_id = held["memoryPrsOpened"][0]["id"].as_str().map(str::to_string);
                held_ids.push(id.clone());
                results.push(json!({
                    "id": id,
                    "outcome": "pending_review",
                    "requiresReview": true,
                    "memoryPr": pr_id,
                    "message": held["message"],
                }));
            }
            Err(error) => {
                // Fail closed: a broken gate must never degrade into an
                // ungated suppression.
                held_ids.push(id.clone());
                results.push(json!({
                    "id": id,
                    "outcome": "gate_error",
                    "requiresReview": true,
                    "error": error,
                }));
            }
        }
    }

    let executed = if to_execute.is_empty() {
        Vec::new()
    } else {
        let refs: Vec<&str> = to_execute.iter().map(String::as_str).collect();
        storage.retire_affected(&refs, &reason)
    };

    for outcome in executed {
        if outcome.suppressed {
            results.push(json!({
                "id": outcome.id,
                "outcome": "suppressed",
                "requiresReview": false,
                "suppressionCount": outcome.suppression_count,
            }));
        } else {
            results.push(json!({
                "id": outcome.id,
                "outcome": "failed",
                "requiresReview": false,
                "suppressionCount": outcome.suppression_count,
                "error": outcome.error,
            }));
        }
    }

    let executed_count = results
        .iter()
        .filter(|r| r["outcome"] == json!("suppressed"))
        .count();

    Ok(json!({
        "tool": "blast_radius",
        "action": "retire",
        "reason": reason,
        "reviewMode": format!("{:?}", mode).to_lowercase(),
        "executed": executed_count,
        "heldForReview": held_ids.len(),
        "results": results,
        "requiresReviewFor": held_ids,
        "notice": RETIRE_NOTICE,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::{ConnectionRecord, IngestInput, MemoryPrStatus};

    fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    fn set_mode(storage: &Arc<Storage>, mode: &str) {
        std::fs::write(
            storage.data_dir().join("review_mode.json"),
            json!({"mode": mode}).to_string(),
        )
        .unwrap();
    }

    fn ingest(storage: &Arc<Storage>, content: &str) -> String {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap()
            .id
    }

    fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
        storage
            .save_connection(&ConnectionRecord {
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

    fn seeded_cause_failure(storage: &Arc<Storage>) -> (String, String) {
        let cause = ingest(storage, "deploy shipped with stale config cache");
        let failure = ingest(storage, "prod outage: config never reloaded");
        let dependent = ingest(storage, "postmortem cites stale cache");
        link(storage, &cause, &failure, "backfill_candidate");
        link(storage, &failure, &dependent, "derived_from");
        (cause, dependent)
    }

    #[test]
    fn schema_has_report_and_retire() {
        let s = schema();
        assert_eq!(s["type"], "object");
        assert_eq!(s["properties"]["action"]["enum"][0], "report");
        assert_eq!(s["properties"]["action"]["enum"][1], "retire");
        assert_eq!(s["properties"]["open_only"]["default"], true);
    }

    #[tokio::test]
    async fn report_by_root_id_lists_descendants() {
        let (storage, _dir) = test_storage();
        let (cause, dependent) = seeded_cause_failure(&storage);

        let r = execute(
            &storage,
            Some(json!({"root_id": cause, "open_only": false})),
        )
        .await
        .unwrap();
        assert_eq!(r["action"], "report");
        assert_eq!(r["total"], 3);
        assert_eq!(r["affected"][0]["id"], cause);
        assert_eq!(r["affected"][0]["via"], "root");
        assert!(r["affected"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["id"] == json!(dependent)));
    }

    #[tokio::test]
    async fn report_by_commit_sha_prefix() {
        let (storage, _dir) = test_storage();
        let cause = ingest(&storage, "commit 0badc0ffee00 fix: reload config");
        let failure = ingest(&storage, "outage after that commit");
        link(&storage, &cause, &failure, "evidence_of");

        let r = execute(&storage, Some(json!({"commit_sha": "0badc0ff"})))
            .await
            .unwrap();
        assert_eq!(r["rootId"], cause);
        assert_eq!(r["total"], 2);
    }

    #[tokio::test]
    async fn report_requires_a_root() {
        let (storage, _dir) = test_storage();
        let err = execute(&storage, Some(json!({}))).await.unwrap_err();
        assert!(err.contains("root_id or commit_sha"));
        let err = execute(&storage, Some(json!({"commit_sha": "zz9"})))
            .await
            .unwrap_err();
        assert!(err.contains("commit <sha>"));
    }

    #[tokio::test]
    async fn retire_fast_mode_suppresses_through_storage_path() {
        let (storage, _dir) = test_storage();
        set_mode(&storage, "fast");
        let (cause, dependent) = seeded_cause_failure(&storage);
        let report = execute(
            &storage,
            Some(json!({"root_id": cause, "open_only": false})),
        )
        .await
        .unwrap();
        let ids: Vec<String> = report["affected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].as_str().unwrap().to_string())
            .collect();

        let r = execute(
            &storage,
            Some(json!({"action": "retire", "ids": ids, "reason": "test retire"})),
        )
        .await
        .unwrap();
        assert_eq!(r["executed"], 3);
        assert_eq!(r["heldForReview"], 0);
        for a in r["results"].as_array().unwrap() {
            assert_eq!(a["outcome"], "suppressed");
            let node = storage.get_node(a["id"].as_str().unwrap()).unwrap().unwrap();
            assert_eq!(node.suppression_count, 1, "suppressed, not deleted");
            assert!(node.suppressed_at.is_some());
        }
        // not a delete: every row still exists
        assert!(storage.get_node(&dependent).unwrap().is_some());
    }

    #[tokio::test]
    async fn retire_risk_gated_opens_prs_and_does_not_suppress() {
        let (storage, _dir) = test_storage();
        set_mode(&storage, "risk_gated");
        let (cause, dependent) = seeded_cause_failure(&storage);
        let report = execute(
            &storage,
            Some(json!({"root_id": cause, "open_only": false})),
        )
        .await
        .unwrap();
        let ids: Vec<String> = report["affected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].as_str().unwrap().to_string())
            .collect();

        let r = execute(
            &storage,
            Some(json!({"action": "retire", "ids": ids, "reason": "gated retire"})),
        )
        .await
        .unwrap();
        assert_eq!(r["executed"], 0, "nothing may suppress before review");
        assert_eq!(r["heldForReview"], 3);
        assert_eq!(r["requiresReviewFor"].as_array().unwrap().len(), 3);

        // PRs, not deletions nor suppressions
        let prs = storage
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(prs.len(), 3, "one pending Memory PR per id");
        assert!(prs.iter().all(|pr| pr.diff["pendingAction"] == json!("suppress")));
        for id in [&cause, &dependent] {
            let node = storage.get_node(id).unwrap().unwrap();
            assert_eq!(node.suppression_count, 0, "pre-gate must not suppress");
        }
    }

    #[tokio::test]
    async fn retire_requires_ids_and_reason() {
        let (storage, _dir) = test_storage();
        let err = execute(
            &storage,
            Some(json!({"action": "retire", "reason": "x"})),
        )
        .await
        .unwrap_err();
        assert!(err.contains("ids"));
        let err = execute(
            &storage,
            Some(json!({"action": "retire", "ids": []})),
        )
        .await
        .unwrap_err();
        assert!(err.contains("empty"));
        let err = execute(
            &storage,
            Some(json!({"action": "retire", "ids": ["not-a-uuid"], "reason": "x"})),
        )
        .await
        .unwrap_err();
        assert!(err.contains("Invalid memory ID"));
    }
}
