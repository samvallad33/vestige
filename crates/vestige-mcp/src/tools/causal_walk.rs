//! # Causal Walk — MCP tool
//!
//! Investigate a failure by walking recorded edges backward from a
//! `logged_write` node. Names are not a join. Candidates are hypotheses.
//! Preview persists nothing; explicit `promote=true` admits an `evidence_of`
//! `SaveEdge` and returns its receipt.

use serde::Deserialize;
use serde_json::{Value, json};
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
                "description": "Explicit evidence handles; at least one is required or the walk returns needs_report (it never guesses). Only logged_write names a record. The walk follows recorded edges backward from that record.",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "failing_test"},
                                "name": {"type": "string", "description": "Failing test name. Not a recorded edge; pass logged_write of the failure record."}
                            },
                            "required": ["kind", "name"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "stack_frame"},
                                "frame": {"type": "string", "description": "\"file:line\" or \"file\". Not a recorded edge; pass logged_write of the failure record."}
                            },
                            "required": ["kind", "frame"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "ci_run"},
                                "run_id": {"type": "string", "description": "Agent-trace run id. Not a recorded edge; pass logged_write of the failure record."}
                            },
                            "required": ["kind", "run_id"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "logged_write"},
                                "node_id": {"type": "string", "description": "Memory / tool-call record id. Recorded edges are walked backward from it."}
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
                "description": "Admit evidence_of SaveEdge trail edges after review and return their receipts. Default false: preview only, no graph mutation. Promotion does not verify causality."
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
    let strata = crate::strata_memory::is_strata_backend(storage.as_ref());
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

    // promote admits evidence_of through SaveEdge (PROPOSE -> GATE -> EFFECT).
    let (edges_persisted, edge_list) = if promote && !result.causes.is_empty() {
        let written = persist_evidence_edges(&**storage, &result)?;
        (written.len(), written)
    } else {
        (0, vec![])
    };
    let receipts: Vec<Value> = edge_list
        .iter()
        .map(|(source_id, target_id)| {
            json!({
                "source_id": source_id,
                "target_id": target_id,
                "link_type": core_causal_walk::EVIDENCE_LINK_TYPE,
            })
        })
        .collect();

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
            "receipts": receipts,
        },
        "note": if strata {
            "causal_walk follows recorded edges backward. promote admits an evidence_of SaveEdge and returns its receipt. Names are not a join."
        } else {
            "causal_walk follows recorded edges only. promote records evidence_of edges. Names are not a join."
        },
    }))
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use tempfile::TempDir;
    use vestige_core::{IngestInput, KnowledgeNode};

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("test.db"))).unwrap();
        (storage, dir)
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
    async fn causal_walk_follows_recorded_edges_and_persists_only_on_promote() {
        let (storage, _dir) = test_storage().await;
        let cause = seed(
            &storage,
            "Raised the pool wait inside the billing client before the quiet deploy window.",
            vec![],
            5,
        );
        let decoy = seed(
            &storage,
            "Checkout lane stored spare linen beside the stair.",
            vec![],
            4,
        );
        let symptom = seed(
            &storage,
            "Checkout lane returned 504. Gateway gave up.",
            vec![],
            1,
        );
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: cause.id.clone(),
                target_id: symptom.id.clone(),
                strength: 1.0,
                link_type: "derived_from".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();
        storage
            .save_connection(&vestige_core::ConnectionRecord {
                source_id: decoy.id.clone(),
                target_id: symptom.id.clone(),
                strength: 1.0,
                link_type: "reports".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();

        let out = execute(&storage, Some(json!({}))).await.unwrap();
        assert!(out["causes"].as_array().unwrap().is_empty());
        assert_eq!(
            out["needs_report"]["missing"].as_array().unwrap().len(),
            1,
            "{out}"
        );
        assert_eq!(
            out["needs_report"]["required_start_points"]
                .as_array()
                .unwrap()
                .len(),
            5
        );

        let named = execute(
            &storage,
            Some(json!({"start_points": [{"kind": "failing_test", "name": "Checkout"}]})),
        )
        .await
        .unwrap();
        assert!(named["causes"].as_array().unwrap().is_empty(), "{named}");
        assert!(
            named["needs_report"]["missing"]
                .to_string()
                .contains("not a recorded edge"),
            "{named}"
        );

        let args = json!({"start_points": [{"kind": "logged_write", "node_id": symptom.id}]});
        for _ in 0..2 {
            let preview = execute(&storage, Some(args.clone())).await.unwrap();
            assert_eq!(preview["preview"], json!(true));
            let causes = preview["causes"].as_array().unwrap();
            assert_eq!(causes.len(), 1, "{preview}");
            assert_eq!(causes[0]["id"], json!(cause.id));
            assert_eq!(causes[0]["shared_anchors"], json!([]));
            assert_eq!(causes[0]["path"][0]["via"], json!("edge/derived_from"));
            assert_eq!(preview["promote"]["edges_persisted"], json!(0));
            assert!(!preview.to_string().contains(&decoy.id), "{preview}");
        }
        assert!(
            storage
                .get_connections_for_memory(&cause.id)
                .unwrap()
                .iter()
                .all(|edge| edge.link_type != "evidence_of"),
            "preview must not write an evidence edge"
        );

        let out = execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "logged_write", "node_id": symptom.id}],
                "promote": true
            })),
        )
        .await
        .unwrap();
        assert_eq!(out["promote"]["edges_persisted"], json!(1), "{out}");
        assert_eq!(out["promote"]["edges"][0], json!([cause.id, symptom.id]));
        assert_eq!(
            out["promote"]["receipts"][0],
            json!({
                "source_id": cause.id,
                "target_id": symptom.id,
                "link_type": "evidence_of",
            })
        );
        let edges = storage.get_connections_for_memory(&cause.id).unwrap();
        assert!(edges.iter().any(|edge| {
            edge.source_id == cause.id
                && edge.target_id == symptom.id
                && edge.link_type == "evidence_of"
        }));
    }
}

#[cfg(test)]
mod strata_tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;
    use vestige_core::advanced::causal_walk::{
        CausalWalkOptions, RecordedEdge, StartPoint, WalkContext, WalkRecord,
    };
    use vestige_core::advanced::retroactive_backfill::extract_entities;

    const CAUSE: &str =
        "Raised the pool wait inside the billing client before the quiet deploy window.";
    const SYMPTOM: &str = "Checkout lane returned 504. Gateway gave up.";
    const LATER: &str = "Zephyr quilt inventory shifted after the quiet hour.";
    const DECOY: &str = "Checkout lane stored spare linen beside the stair.";

    fn names(content: &str) -> std::collections::BTreeSet<String> {
        extract_entities(content, &[]).into_iter().collect()
    }

    fn log_blake3(root: &Path) -> String {
        let mut files = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        files.sort();
        let mut hasher = blake3::Hasher::new();
        for path in files {
            let rel = path.strip_prefix(root).unwrap();
            hasher.update(rel.to_string_lossy().as_bytes());
            hasher.update(b"\0");
            hasher.update(&std::fs::read(&path).unwrap());
            hasher.update(b"\0");
        }
        hasher.finalize().to_hex().to_string()
    }

    fn seed(store: &mut strata_store::StrataStore, content: &str, at_ms: i64) -> String {
        store
            .ingest_in_scope(
                strata_store::IngestInput {
                    content: content.to_string(),
                    created_at_ms: Some(at_ms),
                    ..Default::default()
                },
                "user",
            )
            .unwrap()
    }

    fn edge(
        store: &mut strata_store::StrataStore,
        source_id: &str,
        target_id: &str,
        link_type: &str,
        at_ms: i64,
    ) {
        store
            .save_connection(&strata_store::ConnectionRecord {
                source_id: source_id.to_string(),
                target_id: target_id.to_string(),
                strength_milli: 1000,
                link_type: link_type.to_string(),
                meta_sha: None,
                created_at_ms: at_ms,
                activation_count: 0,
            })
            .unwrap();
    }

    fn open_empty() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn record(id: &str, content: &str, at_ms: i64) -> WalkRecord {
        WalkRecord {
            id: id.to_string(),
            content: content.to_string(),
            entities: Vec::new(),
            created_at: chrono::DateTime::from_timestamp_millis(at_ms).unwrap(),
            stability: 0.0,
            is_commit: false,
            sha: None,
            files: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Frames in one segment: header is 58 bytes, then
    /// `len u32 || kind u8 || payload || blake3[32] || prev[32]`.
    fn segment_frames(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let mut off = 58usize;
        let mut frames = Vec::new();
        while off + 5 <= bytes.len() {
            let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            let kind = bytes[off + 4];
            let start = off + 5;
            let end = start.saturating_add(len);
            let frame_end = end.saturating_add(64);
            if end > bytes.len() || frame_end > bytes.len() {
                break;
            }
            frames.push((kind, bytes[start..end].to_vec()));
            off = frame_end;
        }
        frames
    }

    fn log_frames(root: &Path) -> Vec<(u8, Vec<u8>)> {
        let mut segs: Vec<_> = std::fs::read_dir(root.join("log"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "seg"))
            .collect();
        segs.sort();
        segs.into_iter()
            .flat_map(|path| segment_frames(&std::fs::read(path).unwrap()))
            .collect()
    }

    #[test]
    fn recorded_edge_vocabulary_rejects_names_and_legacy_inferred() {
        for link_type in [
            "touched",
            "anchored_to",
            "derived_from",
            "supersedes",
            "corrects",
            "closed_by",
            "projected_to",
            "evidence_of",
        ] {
            assert!(
                vestige_core::advanced::causal_walk::is_recorded_edge(link_type),
                "{link_type}"
            );
        }
        assert!(!vestige_core::advanced::causal_walk::is_recorded_edge(
            "legacy_inferred"
        ));
        assert!(!vestige_core::advanced::causal_walk::is_recorded_edge(
            "reports"
        ));

        let symptom_ms = 1_700_000_000_000i64;
        let cause = record("cause", CAUSE, symptom_ms - 1);
        let symptom = record("symptom", SYMPTOM, symptom_ms);
        let mut ctx = WalkContext::default();
        ctx.edges.push(RecordedEdge {
            source_id: cause.id.clone(),
            target_id: symptom.id.clone(),
            link_type: "derived_from".into(),
            strength: 1.0,
            legacy_inferred: true,
        });
        let result = CausalWalkOptions::default().walk(
            &[StartPoint::LoggedWrite {
                node_id: symptom.id.clone(),
            }],
            &[cause, symptom],
            &ctx,
        );
        assert!(result.causes.is_empty(), "{:?}", result.causes);
        assert!(result.needs_report.is_some());
    }

    #[tokio::test]
    async fn strata_walk_follows_recorded_edge_and_drops_later_events() {
        let cause_names = names(CAUSE);
        let symptom_names = names(SYMPTOM);
        let shared: Vec<_> = cause_names.intersection(&symptom_names).collect();
        assert!(
            shared.is_empty(),
            "cause and symptom must share no name: {shared:?}"
        );
        assert!(
            !names(DECOY)
                .intersection(&symptom_names)
                .collect::<Vec<_>>()
                .is_empty(),
            "the name decoy must share a name the walk is forbidden to use"
        );

        let dir = TempDir::new().unwrap();
        let symptom_ms = 1_700_000_000_000i64;
        let (cause_id, symptom_id, later_id, decoy_id) = {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let decoy_id = seed(&mut store, DECOY, symptom_ms - 86_400_000);
            let cause_id = seed(&mut store, CAUSE, symptom_ms - 1);
            let symptom_id = seed(&mut store, SYMPTOM, symptom_ms);
            let later_id = seed(&mut store, LATER, symptom_ms + 1);
            edge(
                &mut store,
                &cause_id,
                &symptom_id,
                "derived_from",
                symptom_ms - 1,
            );
            edge(
                &mut store,
                &later_id,
                &symptom_id,
                "derived_from",
                symptom_ms + 1,
            );
            (cause_id, symptom_id, later_id, decoy_id)
        };
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let before = log_blake3(dir.path());
        let preview = execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "logged_write", "node_id": symptom_id}]
            })),
        )
        .await
        .unwrap();
        let again = execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "logged_write", "node_id": symptom_id}]
            })),
        )
        .await
        .unwrap();
        let named = execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "failing_test", "name": "Checkout"}]
            })),
        )
        .await
        .unwrap();
        let after = log_blake3(dir.path());
        assert_eq!(
            before, after,
            "promote:false must not append to the strata log"
        );

        assert_eq!(preview["preview"], json!(true));
        assert_eq!(again["preview"], json!(true));
        assert_eq!(preview["promote"]["edges_persisted"], json!(0));
        assert_eq!(preview["promote"]["receipts"], json!([]));
        let causes = preview["causes"].as_array().unwrap();
        assert_eq!(causes.len(), 1, "{preview}");
        assert_eq!(causes[0]["id"], json!(cause_id));
        assert_eq!(causes[0]["shared_anchors"], json!([]));
        assert_eq!(causes[0]["path"][0]["via"], json!("edge/derived_from"));
        assert!(named["causes"].as_array().unwrap().is_empty(), "{named}");
        assert!(
            named["needs_report"]["missing"]
                .to_string()
                .contains("not a recorded edge"),
            "{named}"
        );

        for body in [&preview, &again, &named] {
            let text = body.to_string();
            assert!(
                !text.contains(&later_id),
                "a later event must not be returned: {text}"
            );
            assert!(!text.contains(&decoy_id), "{text}");
        }
    }

    #[tokio::test]
    async fn strata_promote_admits_evidence_edge_and_receipt_verifies() {
        let dir = TempDir::new().unwrap();
        let symptom_ms = 1_700_000_000_000i64;
        let (cause_id, symptom_id) = {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let cause_id = seed(&mut store, CAUSE, symptom_ms - 1);
            let symptom_id = seed(&mut store, SYMPTOM, symptom_ms);
            edge(
                &mut store,
                &cause_id,
                &symptom_id,
                "derived_from",
                symptom_ms - 1,
            );
            (cause_id, symptom_id)
        };
        let promoted = {
            let storage = crate::strata_memory::open(dir.path()).unwrap();
            let promoted = execute(
                &storage,
                Some(json!({
                    "start_points": [{"kind": "logged_write", "node_id": symptom_id}],
                    "promote": true
                })),
            )
            .await
            .unwrap();
            let second = execute(
                &storage,
                Some(json!({
                    "start_points": [{"kind": "logged_write", "node_id": symptom_id}],
                    "promote": true
                })),
            )
            .await
            .unwrap();
            assert_eq!(second["promote"]["edges_persisted"], json!(0), "{second}");
            assert_eq!(second["promote"]["receipts"], json!([]));
            drop(storage);
            promoted
        };

        assert_eq!(promoted["preview"], json!(false));
        assert_eq!(
            promoted["promote"]["edges_persisted"],
            json!(1),
            "{promoted}"
        );
        assert_eq!(
            promoted["promote"]["receipts"][0],
            json!({
                "source_id": cause_id,
                "target_id": symptom_id,
                "link_type": "evidence_of",
            }),
            "{promoted}"
        );

        {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            let admitted = store.edges().into_iter().any(|edge| {
                edge.source_id == cause_id
                    && edge.target_id == symptom_id
                    && edge.link_type == "evidence_of"
            });
            assert!(
                admitted,
                "SaveEdge must land evidence_of in the derived edge list"
            );
            let verdicts = store.rederive_verdicts().unwrap();
            assert!(!verdicts.is_empty());
            assert!(
                verdicts
                    .iter()
                    .all(|(_, verdict)| format!("{verdict:?}") == "Allow"),
                "{verdicts:?}"
            );
            assert!(store.sweep().is_empty(), "{:?}", store.sweep());

            let frames = log_frames(dir.path());
            let write = frames.iter().find(|(kind, payload)| {
                *kind == 32
                    && payload
                        .windows(cause_id.len())
                        .any(|w| w == cause_id.as_bytes())
                    && payload
                        .windows(symptom_id.len())
                        .any(|w| w == symptom_id.as_bytes())
                    && payload
                        .windows(b"evidence_of".len())
                        .any(|w| w == b"evidence_of")
            });
            let Some((_, op_bytes)) = write else {
                panic!("no STORE_WRITE frame carries the evidence_of edge");
            };
            let digest = *blake3::hash(op_bytes).as_bytes();
            assert!(
                frames.iter().any(|(kind, payload)| {
                    *kind == 1 && payload.len() >= 32 && payload[..32] == digest
                }),
                "PROPOSE.action_hash must be blake3 of the SaveEdge bytes"
            );
            assert!(
                frames.iter().any(|(kind, payload)| {
                    *kind == 3 && payload.len() == 80 && payload[48..] == digest
                }),
                "EFFECT.payload_digest must be blake3 of the SaveEdge bytes"
            );
            store.seal_checkpoint().unwrap();
        }

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../strata-verify/Cargo.toml");
        let output = std::process::Command::new("cargo")
            .arg("run")
            .arg("--quiet")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--")
            .arg(dir.path())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("OK"),
            "strata-verify failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[tokio::test]
    async fn strata_walk_unknown_handle_does_not_guess() {
        let (storage, _dir) = open_empty();
        let missing = "mem-0123456789abcdef";
        let out = execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "logged_write", "node_id": missing}]
            })),
        )
        .await
        .unwrap();
        assert!(out["causes"].as_array().unwrap().is_empty(), "{out}");
        let missing_report = out["needs_report"]["missing"].to_string();
        assert!(
            missing_report.contains(missing),
            "unknown handle must refuse, not guess: {out}"
        );
    }

    #[tokio::test]
    async fn strata_walk_rejects_invalid_input() {
        let (storage, _dir) = open_empty();
        let empty_scope = execute(&storage, Some(json!({"scope": "  "})))
            .await
            .unwrap_err();
        assert!(empty_scope.contains("scope"), "{empty_scope}");
        let bad_start = execute(
            &storage,
            Some(json!({"start_points": [{"kind": "logged_write"}]})),
        )
        .await
        .unwrap_err();
        assert!(bad_start.contains("node_id"), "{bad_start}");
    }
}
