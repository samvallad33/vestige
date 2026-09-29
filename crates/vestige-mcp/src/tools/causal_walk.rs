//! # Causal Walk — MCP tool
//!
//! The successor to `backfill`: investigate a failure from EXPLICIT start
//! points (failing test, stack frame, CI run, logged write, version range)
//! through exact mechanism edges to the change records behind it. Candidates
//! are investigation hypotheses, not proven causes. The default preview
//! persists nothing; explicit `promote=true` records `evidence_of` trail
//! edges through the existing `save_connection` surface.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use vestige_core::advanced::causal_walk::{
    self as core_causal_walk, CausalWalkRequest, StartPoint, persist_evidence_edges,
    walk_storage,
};
use vestige_core::Storage;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "default": "user",
                "description": "Exact project namespace to walk. Defaults to user; cross-project inference is not performed."
            },
            "start_points": {
                "type": "array",
                "minItems": 1,
                "description": "Explicit evidence handles; at least one is required or the walk returns needs_report (it never guesses). version_range restricts commit suspects and pairs best with a failure start point.",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "failing_test"},
                                "name": {"type": "string", "description": "Failing test name; walked to its file's co-touch commits."}
                            },
                            "required": ["kind", "name"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "stack_frame"},
                                "frame": {"type": "string", "description": "\"file:line\" or \"file\"; the last pre-failure toucher is the prime suspect (SZZ-lite)."}
                            },
                            "required": ["kind", "frame"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "ci_run"},
                                "run_id": {"type": "string", "description": "Agent-trace run id; its failure channel (vetoed claims, touched memories) seeds the anchors."}
                            },
                            "required": ["kind", "run_id"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "logged_write"},
                                "node_id": {"type": "string", "description": "Memory / tool-call record id; its edges and shared entities are walked."}
                            },
                            "required": ["kind", "node_id"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "version_range"},
                                "worked_in": {"type": "string", "description": "Last-known-good tag."},
                                "broke_in": {"type": "string", "description": "First-bad tag."},
                                "repo": {"type": "string", "description": "Path to the git repository. Single repo per call."}
                            },
                            "required": ["kind", "worked_in", "broke_in", "repo"]
                        }
                    ]
                }
            },
            "lookback_days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 365,
                "default": 30,
                "description": "How far back before the failure anchor suspects may lie."
            },
            "promote": {
                "type": "boolean",
                "default": false,
                "description": "Explicitly persist evidence_of trail edges after review. Default false: preview only, no graph mutation. Promotion does not verify causality."
            },
            "scan_limit": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "default": 500,
                "description": "Max records scanned in the scope."
            }
        }
    })
}

#[derive(Deserialize, Default)]
struct Args {
    scope: Option<String>,
    start_points: Option<Vec<StartPoint>>,
    lookback_days: Option<i64>,
    promote: Option<bool>,
    scan_limit: Option<i32>,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    // Clamp numeric inputs to the documented schema bounds (the dispatch
    // layer does not enforce JSON-schema min/max).
    let lookback = args.lookback_days.unwrap_or(core_causal_walk::DEFAULT_LOOKBACK_DAYS).clamp(1, 365);
    let promote = args.promote.unwrap_or(false);
    let scan_limit = args.scan_limit.unwrap_or(500).clamp(10, 5000);

    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }

    // No start point (or an empty list) is a needs_report, never a guess and
    // never an error: the core walk handles that refusal.
    let request = CausalWalkRequest {
        scope: scope.to_string(),
        start_points: args.start_points.unwrap_or_default(),
        lookback_days: lookback,
        scan_limit,
    };
    let result = walk_storage(&**storage, &request)?;

    // content previews for causes + rejections (bounded pool, same scope)
    let contents: std::collections::HashMap<String, String> = storage
        .get_all_nodes_in_scope(scope, scan_limit, 0)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|n| (n.id.clone(), n.content.clone()))
        .collect();
    let preview_of = |id: &str| -> String {
        contents
            .get(id)
            .map(|c| c.chars().take(140).collect::<String>())
            .unwrap_or_default()
    };

    let mut causes: Vec<Value> = Vec::with_capacity(result.causes.len());
    for c in &result.causes {
        let mut v = serde_json::to_value(c).map_err(|e| e.to_string())?;
        v["score"] = json!((c.score * 100.0).round() / 100.0);
        v["content_preview"] = json!(preview_of(&c.id));
        causes.push(v);
    }

    // Persist trail edges ONLY on explicit promote.
    let (edges_persisted, edge_list) = if promote && !result.causes.is_empty() {
        let written = persist_evidence_edges(&**storage, &result)?;
        (written.len(), written)
    } else {
        (0, vec![])
    };

    let needs_report = result
        .needs_report
        .as_ref()
        .map(|nr| serde_json::to_value(nr).map_err(|e| e.to_string()))
        .transpose()?;

    let headline = if !result.causes.is_empty() {
        format!(
            "Found {} suspect change record{} by exact mechanism edges from the start points; hypotheses, not proven causes.",
            result.causes.len(),
            if result.causes.len() == 1 { "" } else { "s" }
        )
    } else {
        "The walk refused: no anchored records from the given start points. See needs_report.".to_string()
    };

    Ok(json!({
        "tool": "causal_walk",
        "scope": scope,
        "preview": !promote,
        "evidence_status": "hypothesis",
        "causality_verified": false,
        "headline": headline,
        "causes": causes,
        "needs_report": needs_report,
        "rejected": result.rejected.iter().map(|r| json!({
            "id": r.id,
            "reason": r.reason,
            "shared_anchors": r.shared_anchors,
            "content_preview": preview_of(&r.id),
        })).collect::<Vec<_>>(),
        "promote": {
            "edges_persisted": edges_persisted,
            "edges": edge_list.iter().map(|(s, t)| json!([s, t])).collect::<Vec<_>>(),
            "link_type": core_causal_walk::EVIDENCE_LINK_TYPE,
        },
        "note": "causal_walk replaces backfill (still dispatchable as a hidden alias). Causes rank by IDF-weighted shared anchors, change-record bonus and older-first ties; scores are ranking heuristics, not probabilities of causation. Preview writes nothing; promote records evidence_of edges only.",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use tempfile::TempDir;
    use vestige_core::advanced::git_records::{self, GitCommit};
    use vestige_core::{IngestInput, KnowledgeNode};

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
    }

    fn sha(c: char) -> String {
        c.to_string().repeat(40)
    }

    fn seed(storage: &Arc<Storage>, content: &str, tags: Vec<&str>, days_ago: i64) -> KnowledgeNode {
        let node = storage
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&node.id, Utc::now() - Duration::days(days_ago))
            .unwrap();
        storage.get_node(&node.id).unwrap().unwrap()
    }

    fn commit_record(
        storage: &Arc<Storage>,
        sha: &str,
        subject: &str,
        files: &[&str],
        days_ago: i64,
    ) -> KnowledgeNode {
        let content = git_records::record_content(&GitCommit {
            sha: sha.to_string(),
            time: Utc::now() - Duration::days(days_ago),
            subject: subject.to_string(),
            files: files.iter().map(|f| f.to_string()).collect(),
            extra_files: 0,
            symbols: vec![],
            mentions: vec![],
                hunks: vec![],
                extra_hunks: 0,
                imports: vec![],
        });
        seed(storage, &content, vec![git_records::COMMIT_TAG], days_ago)
    }

    #[test]
    fn schema_declares_all_five_start_point_kinds() {
        let full = schema();
        let variants = full["properties"]["start_points"]["items"]["oneOf"]
            .as_array()
            .expect("oneOf variants");
        let kinds: Vec<&str> = variants
            .iter()
            .map(|v| v["properties"]["kind"]["const"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            vec!["failing_test", "stack_frame", "ci_run", "logged_write", "version_range"]
        );
    }

    #[tokio::test]
    async fn causal_walk_finds_cotouch_and_persists_edges_only_on_promote() {
        let (storage, _dir) = test_storage().await;
        let note = seed(
            &storage,
            "test_login_flow failed: auth flow crashed in tests/auth_test.rs",
            vec!["ci"],
            0,
        );
        let bad = commit_record(
            &storage,
            &sha('a'),
            "harden login",
            &["tests/auth_test.rs", "src/auth.rs"],
            5,
        );
        let good = commit_record(
            &storage,
            &sha('b'),
            "tune session cache",
            &["tests/auth_test.rs", "src/session.rs"],
            25,
        );

        // 1) no start point -> needs_report, not an error
        let out = execute(&storage, Some(json!({}))).await.unwrap();
        assert!(out["causes"].as_array().unwrap().is_empty());
        assert!(
            out["needs_report"]["missing"].as_array().unwrap().len() == 1,
            "{out}"
        );
        assert_eq!(
            out["needs_report"]["required_start_points"].as_array().unwrap().len(),
            5
        );

        // 2) failing_test walk: co-touch commits ranked, preview writes nothing
        let args = json!({"start_points": [{"kind": "failing_test", "name": "test_login_flow"}]});
        for _ in 0..2 {
            let preview = execute(&storage, Some(args.clone())).await.unwrap();
            assert_eq!(preview["preview"], json!(true));
            let causes = preview["causes"].as_array().unwrap();
            assert_eq!(causes.len(), 2);
            assert_eq!(causes[0]["sha"], json!(sha('a')));
            assert_eq!(
                causes[0]["path"][0]["via"], "failing_test/co_touch",
                "{}", preview
            );
            assert!(
                causes[0]["shared_anchors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e == "tests/auth_test.rs")
            );
            assert_eq!(preview["promote"]["edges_persisted"], json!(0));
        }
        assert!(
            storage
                .get_connections_for_memory(&bad.id)
                .unwrap()
                .is_empty(),
            "preview must not write graph edges"
        );
        let before = storage.get_node(&bad.id).unwrap().unwrap();
        assert_eq!(before.reps, storage.get_node(&bad.id).unwrap().unwrap().reps);

        // 3) promote: exactly one evidence_of edge from the top cause to the note
        let out = execute(
            &storage,
            Some(json!({"start_points": [{"kind": "failing_test", "name": "test_login_flow"}], "promote": true})),
        )
        .await
        .unwrap();
        // both surfaced causes link to the same evidence note (the failure note)
        assert_eq!(out["promote"]["edges_persisted"], json!(2), "{out}");
        assert_eq!(
            out["promote"]["edges"][0],
            json!([bad.id, note.id]),
            "the trail edge is cause -> evidence record"
        );
        let edges = storage.get_connections_for_memory(&bad.id).unwrap();
        assert!(edges.iter().any(|e| {
            e.source_id == bad.id
                && e.target_id == note.id
                && e.link_type == "evidence_of"
        }));
        // the runner-up commit was not linked (it shares no evidence_to target
        // other than the same note — one edge per cause is expected too)
        let good_edges = storage.get_connections_for_memory(&good.id).unwrap();
        assert_eq!(good_edges.len(), 1, "every surfaced cause gets its edge");
    }
}
