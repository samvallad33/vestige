//! `selftest` MCP tool.
//!
//! On a Strata log the tool only reads the user's directory. It checks the
//! segment chain, re-derives every receipt against the gate, and round-trips
//! one ingest through the gate in a scratch store outside that directory.
//!
//! The legacy SQLite path (below, `legacy-sqlite` tests) is an end-to-end
//! check that retroactive backfill reaches a quiet cause. It NEVER touches
//! the live store:
//!
//! 1. `backup_to` the live store into a fresh tempdir (the same consistent
//!    `VACUUM INTO` snapshot `vestige backup` uses — read-only on the source).
//! 2. Open the copy as a second `Storage`.
//! 3. Five rounds: plant a synthetic quiet cause (distinctive env-shaped
//!    `PLANTED_CAUSE_<k>` anchor + a file path, backdated 5 days via
//!    `set_created_at`) in an isolated `selftest-round-<k>` scope, ingest a
//!    synthetic failure sharing EXACTLY that one anchor, then run the real
//!    `tools::backfill::execute` against the COPY (`failure_id` + `manual=true`,
//!    preview) and score hit@1 / hit@3.
//! 4. A 6th round shares NO anchor: the backfill must fire its gap report and
//!    name the missing anchor (`planted_cause_6`) — the calibration metric
//!    that says "when the trail breaks, the tool says so instead of guessing".
//! 5. Delete the temp store.
//!
//! Deterministic by construction: each round lives in its own scope, the
//! only in-window candidate is the planted cause, ages are exact (5.0 days),
//! and every round runs as a preview (no graph or FSRS writes at all).

use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::{IngestInput, Storage};

pub fn schema() -> Value {
    json!({
        "type": "object",
        "description": "Read-only Strata selftest: log chain, receipt verification, and a gate round-trip in a scratch store.",
        "properties": {}
    })
}

/// One scored round: cause planted, failure referencing exactly one anchor.
struct RoundOutcome {
    round: u32,
    cause_id: String,
    hit_at_1: bool,
    hit_at_3: bool,
    rank: Option<usize>,
}

/// Plant a quiet, non-failure-looking cause carrying exactly two anchors: the
/// env-shaped `PLANTED_CAUSE_<k>` tag and a file path in the content.
fn plant_cause(storage: &Arc<Storage>, scope: &str, k: u32) -> Result<String, String> {
    let cause = storage
        .ingest_in_scope(
            IngestInput {
                content: format!("Tuned PLANTED_KNOB_{k} inside src/selftest/planted_{k}.rs"),
                node_type: "decision".to_string(),
                tags: vec![format!("PLANTED_CAUSE_{k}")],
                ..Default::default()
            },
            scope,
        )
        .map_err(|e| format!("round {k}: planting cause failed: {e}"))?;
    storage
        .set_created_at(&cause.id, chrono::Utc::now() - chrono::Duration::days(5))
        .map_err(|e| format!("round {k}: backdating cause failed: {e}"))?;
    Ok(cause.id)
}

/// Ingest a synthetic failure that shares EXACTLY the `PLANTED_CAUSE_<k>`
/// anchor with round k's cause (share_anchor=true) or none at all (round 6).
fn plant_failure(
    storage: &Arc<Storage>,
    scope: &str,
    k: u32,
    share_anchor: bool,
) -> Result<String, String> {
    let failure = storage
        .ingest_in_scope(
            IngestInput {
                content: if share_anchor {
                    format!("Outage on shard {k}: crash followed PLANTED_CAUSE_{k} change")
                } else {
                    "Outage with no recorded trigger: crash after PLANTED_CAUSE_6 flip".to_string()
                },
                node_type: "event".to_string(),
                tags: vec![format!("PLANTED_CAUSE_{k}")],
                ..Default::default()
            },
            scope,
        )
        .map_err(|e| format!("round {k}: planting failure failed: {e}"))?;
    Ok(failure.id)
}

/// Run the REAL backfill tool (manual=true, preview) against the copy.
async fn run_backfill(
    storage: &Arc<Storage>,
    scope: &str,
    failure_id: &str,
) -> Result<Value, String> {
    super::backfill::execute(
        storage,
        Some(json!({
            "failure_id": failure_id,
            "scope": scope,
            "manual": true,
            "promote": false,
            "lookback_days": 30,
            "scan_limit": 100,
        })),
    )
    .await
}

mod strata_selftest;

pub async fn execute(storage: &Arc<Storage>, _args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return strata_selftest::run(storage);
    }
    // 1. Consistent snapshot of the live store into a throwaway tempdir.
    let dir = tempfile::TempDir::new().map_err(|e| format!("tempdir failed: {e}"))?;
    let copy_path = dir.path().join("selftest-copy.db");
    storage
        .backup_to(&copy_path)
        .map_err(|e| format!("backing the store up for the selftest failed: {e}"))?;

    // 2. Open the copy as its own storage. Everything below mutates only this.
    let copy = Arc::new(
        vestige_core::open_storage(Some(copy_path))
            .map_err(|e| format!("opening the selftest copy failed: {e}"))?,
    );

    // 3. Five planted-cause rounds, each isolated in its own scope.
    let mut outcomes: Vec<RoundOutcome> = Vec::new();
    for k in 1..=5u32 {
        let scope = format!("selftest-round-{k}");
        let cause_id = plant_cause(&copy, &scope, k)?;
        let failure_id = plant_failure(&copy, &scope, k, true)?;
        let out = run_backfill(&copy, &scope, &failure_id).await?;
        let causes = out["causes"].as_array().cloned().unwrap_or_default();
        let rank = causes
            .iter()
            .position(|c| c["memory_id"].as_str() == Some(cause_id.as_str()));
        outcomes.push(RoundOutcome {
            round: k,
            cause_id,
            hit_at_1: rank == Some(0),
            hit_at_3: rank.is_some_and(|r| r < 3),
            rank,
        });
    }

    let hits = outcomes.iter().filter(|o| o.hit_at_1).count();
    let hits_at_3 = outcomes.iter().filter(|o| o.hit_at_3).count();
    let misses = outcomes.len() - hits;

    // 4. Gap-calibration round: NO shared anchor. The backfill must fire its
    //    gap report and name the failure's env-shaped anchor as the missing
    //    record class, instead of surfacing an unrelated cause. The only
    //    in-window record is an unrelated change that shares nothing.
    let gap_scope = "selftest-round-6".to_string();
    let unrelated = copy
        .ingest_in_scope(
            IngestInput {
                content: "Rotated an unrelated credential in src/selftest/other.rs".to_string(),
                node_type: "decision".to_string(),
                tags: vec!["SELFTEST_UNRELATED".to_string()],
                ..Default::default()
            },
            &gap_scope,
        )
        .map_err(|e| format!("round 6: planting unrelated record failed: {e}"))?;
    copy.set_created_at(
        &unrelated.id,
        chrono::Utc::now() - chrono::Duration::days(5),
    )
    .map_err(|e| format!("round 6: backdating failed: {e}"))?;
    let gap_failure_id = plant_failure(&copy, &gap_scope, 6, false)?;
    let gap_out = run_backfill(&copy, &gap_scope, &gap_failure_id).await?;

    let gap_fired = gap_out["triggered"] == json!(true)
        && gap_out["causes"].as_array().is_some_and(Vec::is_empty)
        && gap_out["gap"].is_object();
    let missing: Vec<String> = gap_out["gap"]["missing_entities"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut missing = missing;
    missing.sort();
    let named_missing_anchor = missing.iter().find(|e| *e == "planted_cause_6").cloned();
    let gap_calibration = gap_fired && named_missing_anchor.is_some();

    // 5. Drop the copy (closing its connections) and remove the temp store.
    drop(copy);
    let temp_store_deleted = dir.close().is_ok();

    Ok(json!({
        "tool": "selftest",
        "kind": "planted_cause_selftest",
        "rounds": outcomes.len(),
        "hits": hits,
        "misses": misses,
        "hit_at_3": hits_at_3,
        "hit_rate_1": (hits as f64 / outcomes.len() as f64 * 1000.0).round() / 1000.0,
        "hit_rate_3": (hits_at_3 as f64 / outcomes.len() as f64 * 1000.0).round() / 1000.0,
        "rounds_detail": outcomes.iter().map(|o| json!({
            "round": o.round,
            "cause_id": o.cause_id,
            "rank": o.rank.map(|r| r + 1),
            "hit_at_1": o.hit_at_1,
            "hit_at_3": o.hit_at_3,
        })).collect::<Vec<_>>(),
        "gap_calibration": gap_calibration,
        "gap": {
            "fired": gap_fired,
            "named_missing_anchor": named_missing_anchor,
            "missing_entities": missing,
        },
        "deterministic": true,
        "live_store_touched": false,
        "temp_store_deleted": temp_store_deleted,
        "note": "Planted 5 quiet causes (env-shaped anchor + file path, backdated 5d) plus failures sharing exactly one anchor, ran the real backfill against a temp copy of the store, and scored hit@1/hit@3. The 6th round shares no anchor and must fire the gap report naming the missing anchor. The live store is only read (backup_to snapshot).",
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    fn live_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("live.db"))).unwrap();
        (storage, dir)
    }

    /// The full planted-cause flow against a seeded (non-empty) store: every
    /// round must hit@1, the gap round must fire and name the missing
    /// anchor, the temp store must be deleted, and the LIVE store must be
    /// untouched (same nodes, no selftest scopes).
    #[tokio::test]
    async fn planted_cause_selftest_scores_perfectly_and_never_touches_the_live_store() {
        let (storage, _dir) = live_store();
        // Seed one ordinary memory so the backup copies real content.
        storage
            .ingest(IngestInput {
                content: "Ordinary note: the coffee machine is on the third floor".to_string(),
                ..Default::default()
            })
            .unwrap();

        let out = execute(&storage, None).await.expect("selftest must run");

        assert_eq!(out["rounds"], json!(5));
        assert_eq!(
            out["hits"],
            json!(5),
            "every planted cause must be rank 1: {out}"
        );
        assert_eq!(out["misses"], json!(0));
        assert_eq!(out["hit_at_3"], json!(5));
        assert_eq!(out["hit_rate_1"], json!(1.0));
        assert_eq!(out["deterministic"], json!(true));
        assert_eq!(out["temp_store_deleted"], json!(true));
        assert_eq!(out["live_store_touched"], json!(false));

        // Calibration metric: the no-shared-anchor round fires the gap report
        // and names the missing env-shaped anchor class.
        assert_eq!(out["gap_calibration"], json!(true), "{out}");
        assert_eq!(out["gap"]["fired"], json!(true));
        assert_eq!(out["gap"]["named_missing_anchor"], json!("planted_cause_6"));

        // Live store untouched: same single node, nothing in the selftest scopes.
        let nodes = storage.get_all_nodes(100, 0).unwrap();
        assert_eq!(
            nodes.len(),
            1,
            "no planted record may leak into the live store"
        );
        assert!(
            storage
                .get_all_nodes_in_scope("selftest-round-1", 100, 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            storage
                .get_all_nodes_in_scope("selftest-round-6", 100, 0)
                .unwrap()
                .is_empty()
        );
    }

    /// The selftest works on an empty store too (a fresh install's baseline).
    #[tokio::test]
    async fn planted_cause_selftest_runs_on_an_empty_store() {
        let (storage, _dir) = live_store();
        let out = execute(&storage, None).await.expect("selftest must run");
        assert_eq!(out["hits"], json!(5), "{out}");
        assert_eq!(out["gap_calibration"], json!(true));
    }
}
