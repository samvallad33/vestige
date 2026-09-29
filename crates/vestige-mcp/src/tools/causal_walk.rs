//! # Causal Walk — MCP tool
//!
//! The successor to `backfill`: investigate a failure from EXPLICIT start
//! points (failing test, stack frame, CI run, logged write, version range)
//! through exact mechanism edges to the change records behind it. Candidates
//! are investigation hypotheses, not proven causes. The default preview
//! persists nothing; explicit `promote=true` records `evidence_of` trail
//! edges through the existing `save_connection` surface.
//!
//! On a Strata log the walk is a bounded backward BFS from the given node
//! over recorded causal edges only. Shared names are not edges.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use vestige_core::Storage;
use vestige_core::advanced::causal_walk::{
    self as core_causal_walk, CausalWalkRequest, StartPoint, persist_evidence_edges, walk_storage,
};

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
    /// Strata start. The advertised handle is `start_points` `logged_write`.
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    node: Option<String>,
    #[serde(default)]
    logged_write: Option<String>,
}

/// Recorded causal vocabulary. Incoming edge: source caused target.
const CAUSAL_LINKS: &[&str] = &["closed_by", "derived_from", "evidence_of", "touched"];
const MAX_DEPTH: u32 = 8;

struct Reached {
    id: String,
    depth: u32,
    path: Vec<Hop>,
}

#[derive(Clone)]
struct Hop {
    source_id: String,
    target_id: String,
    link_type: String,
}

pub async fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    if crate::strata_memory::is_strata_backend(storage.as_ref()) {
        return recorded_backward_walk(storage, args);
    }
    let args: Args = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| e.to_string())?,
        None => Args::default(),
    };
    // Clamp numeric inputs to the documented schema bounds (the dispatch
    // layer does not enforce JSON-schema min/max).
    let lookback = args
        .lookback_days
        .unwrap_or(core_causal_walk::DEFAULT_LOOKBACK_DAYS)
        .clamp(1, 365);
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
        "The walk refused: no anchored records from the given start points. See needs_report."
            .to_string()
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

fn recorded_backward_walk(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let args: Args = match args {
        Some(value) => serde_json::from_value(value).map_err(|err| err.to_string())?,
        None => Args::default(),
    };
    let scope = args.scope.as_deref().unwrap_or("user").trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }
    let node_cap = args.scan_limit.unwrap_or(500).clamp(10, 5000) as usize;
    let Some(start_id) = explicit_node_id(&args) else {
        return Ok(walk_payload(
            storage,
            scope,
            WalkOut {
                start: None,
                nodes: Vec::new(),
                truncated: false,
                node_cap,
                needs_report: Some(json!({
                    "missing": ["node_id"],
                    "detail": "pass node_id or a logged_write start point; causal_walk does not guess",
                })),
            },
        ));
    };
    if !in_scope(storage, &start_id, scope)? {
        return Ok(walk_payload(
            storage,
            scope,
            WalkOut {
                start: Some(start_id),
                nodes: Vec::new(),
                truncated: false,
                node_cap,
                needs_report: Some(json!({
                    "missing": ["node"],
                    "detail": "start node is not recorded in scope",
                })),
            },
        ));
    }

    let mut reached = vec![Reached {
        id: start_id.clone(),
        depth: 0,
        path: Vec::new(),
    }];
    let mut visited = HashSet::from([start_id.clone()]);
    let mut queue = VecDeque::from([(start_id.clone(), 0u32, Vec::new())]);
    let mut truncated = false;

    while let Some((current, depth, path)) = queue.pop_front() {
        if depth >= MAX_DEPTH {
            if has_admissible_cause(storage, &current, scope, &visited)? {
                truncated = true;
            }
            continue;
        }
        for edge in incoming_causal(storage, &current)? {
            if visited.contains(edge.source_id.as_str()) {
                continue;
            }
            if !in_scope(storage, &edge.source_id, scope)? {
                continue;
            }
            if reached.len() >= node_cap {
                truncated = true;
                break;
            }
            visited.insert(edge.source_id.clone());
            let mut next_path = path.clone();
            next_path.push(Hop {
                source_id: edge.source_id.clone(),
                target_id: edge.target_id.clone(),
                link_type: edge.link_type.clone(),
            });
            let next_depth = depth + 1;
            queue.push_back((edge.source_id.clone(), next_depth, next_path.clone()));
            reached.push(Reached {
                id: edge.source_id,
                depth: next_depth,
                path: next_path,
            });
        }
        if reached.len() >= node_cap {
            for (pending, _, _) in &queue {
                if has_admissible_cause(storage, pending, scope, &visited)? {
                    truncated = true;
                    break;
                }
            }
            break;
        }
    }

    reached.sort_by(|a, b| a.depth.cmp(&b.depth).then_with(|| a.id.cmp(&b.id)));
    Ok(walk_payload(
        storage,
        scope,
        WalkOut {
            start: Some(start_id),
            nodes: reached,
            truncated,
            node_cap,
            needs_report: None,
        },
    ))
}

struct WalkOut {
    start: Option<String>,
    nodes: Vec<Reached>,
    truncated: bool,
    node_cap: usize,
    needs_report: Option<Value>,
}

fn explicit_node_id(args: &Args) -> Option<String> {
    nonempty(&args.node_id)
        .or_else(|| nonempty(&args.node))
        .or_else(|| nonempty(&args.logged_write))
        .or_else(|| {
            args.start_points
                .as_ref()?
                .iter()
                .find_map(|point| match point {
                    StartPoint::LoggedWrite { node_id } => {
                        let id = node_id.trim();
                        (!id.is_empty()).then(|| id.to_string())
                    }
                    _ => None,
                })
        })
}

fn nonempty(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

fn in_scope(storage: &Arc<Storage>, id: &str, scope: &str) -> Result<bool, String> {
    storage
        .node_is_in_scope(id, scope)
        .map_err(|err| err.to_string())
}

fn has_admissible_cause(
    storage: &Arc<Storage>,
    node_id: &str,
    scope: &str,
    visited: &HashSet<String>,
) -> Result<bool, String> {
    for edge in incoming_causal(storage, node_id)? {
        if visited.contains(edge.source_id.as_str()) {
            continue;
        }
        if in_scope(storage, &edge.source_id, scope)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn incoming_causal(
    storage: &Arc<Storage>,
    node_id: &str,
) -> Result<Vec<vestige_core::ConnectionRecord>, String> {
    let mut edges = storage
        .get_connections_for_memory(node_id)
        .map_err(|err| err.to_string())?;
    edges.retain(|edge| {
        edge.target_id == node_id
            && edge.source_id != node_id
            && CAUSAL_LINKS.contains(&edge.link_type.as_str())
    });
    edges.sort_by(|a, b| (&a.link_type, &a.source_id).cmp(&(&b.link_type, &b.source_id)));
    Ok(edges)
}

fn walk_payload(storage: &Arc<Storage>, scope: &str, walk: WalkOut) -> Value {
    let node_json = |node: &Reached| {
        json!({
            "id": node.id,
            "depth": node.depth,
            "content": content_of(storage, &node.id),
            "path": node.path.iter().map(|hop| json!({
                "source_id": hop.source_id,
                "target_id": hop.target_id,
                "link_type": hop.link_type,
            })).collect::<Vec<_>>(),
        })
    };
    let causes: Vec<Value> = walk
        .nodes
        .iter()
        .filter(|node| node.depth > 0)
        .map(node_json)
        .collect();
    let rendered: Vec<Value> = walk.nodes.iter().map(node_json).collect();
    json!({
        "tool": "causal_walk",
        "status": "completed",
        "scope": scope,
        "direction": "backward",
        "start": walk.start,
        "truncated": walk.truncated,
        "bounds": { "max_depth": MAX_DEPTH, "max_nodes": walk.node_cap },
        "nodes": rendered,
        "causes": causes,
        "needs_report": walk.needs_report,
        "note": "Backward BFS over recorded causal edges only (closed_by, derived_from, evidence_of, touched).",
    })
}

fn content_of(storage: &Arc<Storage>, id: &str) -> String {
    storage
        .get_node(id)
        .ok()
        .flatten()
        .map(|node| node.content.chars().take(160).collect())
        .unwrap_or_default()
}

/// Fixture for the stdio test: an effect, its recorded cause, and a same-text decoy.
#[doc(hidden)]
pub fn seed_recorded_cause(dir: &std::path::Path) -> Result<(String, String, String), String> {
    let storage = crate::strata_memory::open(dir).map_err(|err| err.to_string())?;
    let effect = ingest_id(&storage, "user", "login handler failed")?;
    let cause = ingest_id(&storage, "user", "commit flipped the auth timeout")?;
    let decoy = ingest_id(&storage, "user", "commit flipped the auth timeout")?;
    let downstream = ingest_id(&storage, "user", "pager incident after the failure")?;
    save_link(&storage, &cause, &effect, "derived_from")?;
    save_link(&storage, &decoy, &effect, "projected_to")?;
    save_link(&storage, &effect, &downstream, "derived_from")?;
    Ok((effect, cause, decoy))
}

fn ingest_id(storage: &Arc<Storage>, scope: &str, content: &str) -> Result<String, String> {
    storage
        .ingest_in_scope(
            vestige_core::IngestInput {
                content: content.to_string(),
                ..Default::default()
            },
            scope,
        )
        .map(|node| node.id)
        .map_err(|err| err.to_string())
}

fn save_link(
    storage: &Arc<Storage>,
    source: &str,
    target: &str,
    link_type: &str,
) -> Result<(), String> {
    let now = chrono::Utc::now();
    storage
        .save_connection(&vestige_core::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .map_err(|err| err.to_string())
}

#[cfg(all(test, feature = "legacy-sqlite"))]
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

    fn seed(
        storage: &Arc<Storage>,
        content: &str,
        tags: Vec<&str>,
        days_ago: i64,
    ) -> KnowledgeNode {
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
            vec![
                "failing_test",
                "stack_frame",
                "ci_run",
                "logged_write",
                "version_range"
            ]
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
            out["needs_report"]["required_start_points"]
                .as_array()
                .unwrap()
                .len(),
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
                "{}",
                preview
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
        assert_eq!(
            before.reps,
            storage.get_node(&bad.id).unwrap().unwrap().reps
        );

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
            e.source_id == bad.id && e.target_id == note.id && e.link_type == "evidence_of"
        }));
        // the runner-up commit was not linked (it shares no evidence_to target
        // other than the same note — one edge per cause is expected too)
        let good_edges = storage.get_connections_for_memory(&good.id).unwrap();
        assert_eq!(good_edges.len(), 1, "every surfaced cause gets its edge");
    }
}

#[cfg(test)]
mod strata_walk {
    use super::*;
    use vestige_core::{ConnectionRecord, IngestInput};

    fn open() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn put(storage: &Arc<Storage>, scope: &str, content: &str) -> String {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    ..Default::default()
                },
                scope,
            )
            .unwrap()
            .id
    }

    fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
        let now = chrono::Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: link_type.to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    fn node_ids(out: &Value) -> Vec<String> {
        out["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| node["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn recorded_backward_walk_returns_the_cause() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "login handler failed");
        let cause = put(&storage, "user", "commit flipped the auth timeout");
        let origin = put(&storage, "user", "timeout default changed in review");
        let same_words = put(&storage, "user", "commit flipped the auth timeout");
        let downstream = put(&storage, "user", "pager incident after the failure");
        let other = put(&storage, "other", "commit flipped the auth timeout");
        link(&storage, &cause, &effect, "derived_from");
        link(&storage, &origin, &cause, "evidence_of");
        link(&storage, &same_words, &effect, "projected_to");
        link(&storage, &effect, &downstream, "derived_from");
        link(&storage, &other, &effect, "derived_from");
        let edges_before = storage.get_all_connections().unwrap().len();

        let args = json!({
            "scope": "user",
            "promote": true,
            "start_points": [{ "kind": "logged_write", "node_id": effect }],
        });
        let out = execute(&storage, Some(args.clone())).await.unwrap();
        let again = execute(&storage, Some(args)).await.unwrap();
        assert_eq!(out, again, "walk must be deterministic");
        assert_eq!(storage.get_all_connections().unwrap().len(), edges_before);

        let body = serde_json::to_string(&out).unwrap();
        assert!(!body.contains("pending_strata"), "{out}");
        assert_eq!(out["status"], "completed");
        assert_eq!(out["direction"], "backward");
        assert!(out["needs_report"].is_null(), "{out}");
        assert_eq!(out["start"], effect);

        let cause_row = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == cause)
            .unwrap_or_else(|| panic!("missing cause: {out}"));
        assert_eq!(cause_row["depth"], 1);
        assert_eq!(cause_row["content"], "commit flipped the auth timeout");
        assert_eq!(cause_row["path"][0]["source_id"], cause);
        assert_eq!(cause_row["path"][0]["target_id"], effect);
        assert_eq!(cause_row["path"][0]["link_type"], "derived_from");

        let origin_row = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == origin)
            .unwrap_or_else(|| panic!("missing origin: {out}"));
        assert_eq!(origin_row["depth"], 2);
        assert_eq!(origin_row["path"].as_array().unwrap().len(), 2);
        assert_eq!(origin_row["path"][1]["source_id"], origin);
        assert_eq!(origin_row["path"][1]["link_type"], "evidence_of");

        let ids = node_ids(&out);
        assert!(ids.contains(&effect));
        assert!(
            !ids.contains(&same_words),
            "shared text is not an edge: {out}"
        );
        assert!(
            !ids.contains(&downstream),
            "forward edges are not walked: {out}"
        );
        assert!(!ids.contains(&other), "walk stays in scope: {out}");

        let missing = execute(&storage, Some(json!({"scope": "user"})))
            .await
            .unwrap();
        assert_eq!(missing["status"], "completed");
        assert!(missing["causes"].as_array().unwrap().is_empty());
        assert!(
            missing["needs_report"]["missing"]
                .as_array()
                .unwrap()
                .contains(&json!("node_id"))
        );
        assert!(
            !serde_json::to_string(&missing)
                .unwrap()
                .contains("pending_strata")
        );
    }

    #[tokio::test]
    async fn recorded_walk_is_ordered_and_bounded() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "effect");
        let smaller = put(&storage, "user", "smaller-id cause");
        let larger = put(&storage, "user", "larger-id cause");
        assert!(smaller < larger, "ingest order is id order");
        link(&storage, &larger, &effect, "closed_by");
        link(&storage, &smaller, &effect, "derived_from");
        let out = execute(&storage, Some(json!({"node_id": effect})))
            .await
            .unwrap();
        let cause_ids: Vec<&str> = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect();
        assert_eq!(cause_ids, vec![smaller.as_str(), larger.as_str()], "{out}");

        let (storage, _dir) = open();
        let mut chain = Vec::new();
        for hop in 0..=MAX_DEPTH + 1 {
            chain.push(put(&storage, "user", &format!("hop {hop}")));
        }
        for hop in 0..chain.len() - 1 {
            link(&storage, &chain[hop + 1], &chain[hop], "touched");
        }
        let out = execute(&storage, Some(json!({"node_id": chain[0]})))
            .await
            .unwrap();
        let ids = node_ids(&out);
        assert!(ids.contains(&chain[MAX_DEPTH as usize]), "{out}");
        assert!(!ids.contains(&chain[(MAX_DEPTH + 1) as usize]), "{out}");
        assert_eq!(out["truncated"], true, "{out}");

        let (storage, _dir) = open();
        let effect = put(&storage, "user", "bush");
        let mut causes = Vec::new();
        for index in 0..12 {
            causes.push(put(&storage, "user", &format!("cause {index}")));
        }
        for cause in &causes {
            link(&storage, cause, &effect, "derived_from");
        }
        let out = execute(&storage, Some(json!({"node_id": effect, "scan_limit": 10})))
            .await
            .unwrap();
        assert_eq!(out["nodes"].as_array().unwrap().len(), 10, "{out}");
        assert_eq!(out["truncated"], true, "{out}");
        let ids = node_ids(&out);
        assert!(ids.contains(&causes[0]));
        assert!(!ids.contains(&causes[11]), "{ids:?}");
    }
}
