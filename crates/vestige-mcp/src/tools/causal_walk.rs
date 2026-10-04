//! # Causal Walk — MCP tool
//!
//! The successor to `backfill`: investigate a failure from EXPLICIT start
//! points (failing test, stack frame, CI run, logged write, version range)
//! through exact mechanism edges to the change records behind it. Candidates
//! are investigation hypotheses, not proven causes. On the legacy engine the
//! default preview persists nothing and explicit `promote=true` records
//! `evidence_of` trail edges through the existing `save_connection` surface; a
//! recorded walk has no inferred trail, so it reports promote as a no-op.
//!
//! On a Strata log the walk is a bounded backward BFS from each start node
//! over recorded causal edges only. A start point of any kind names its start
//! node with `node_id`. Shared names are not edges.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;

use vestige_core::Storage;
use vestige_core::advanced::causal_walk::{
    self as core_causal_walk, CausalWalkRequest, StartPoint, persist_evidence_edges, walk_storage,
};

/// The optional `node_id` every start-point variant accepts.
fn symptom_node_id() -> Value {
    json!({
        "type": "string",
        "description": "Optional. The id of the memory that records this symptom (what a symptom write returned). A Strata log is walked from recorded nodes, so a start point without node_id is reported unresolved in start_points instead of walked."
    })
}

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
                "description": "Explicit evidence handles; at least one is required or the walk returns needs_report (it never guesses). Every kind accepts node_id, the recorded symptom to walk from; on a Strata log a start point without one is reported unresolved. version_range restricts commit suspects and pairs best with a failure start point.",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "failing_test"},
                                "name": {"type": "string", "description": "Failing test name; walked to its file's co-touch commits."},
                                "node_id": symptom_node_id()
                            },
                            "required": ["kind", "name"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "stack_frame"},
                                "frame": {"type": "string", "description": "\"file:line\" or \"file\"; the last pre-failure toucher is the prime suspect (SZZ-lite)."},
                                "node_id": symptom_node_id()
                            },
                            "required": ["kind", "frame"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "ci_run"},
                                "run_id": {"type": "string", "description": "Agent-trace run id; its failure channel (vetoed claims, touched memories) seeds the anchors."},
                                "node_id": symptom_node_id()
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
                                "repo": {"type": "string", "description": "Path to the git repository. Single repo per call."},
                                "node_id": symptom_node_id()
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
                "description": "Legacy engine only: explicitly persist evidence_of trail edges after review. Default false: preview only, no graph mutation. Promotion does not verify causality. A Strata walk follows recorded edges only and records nothing."
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

// Sibling tools take camelCase (`repoPath`, `memoryId`), so a client that
// guesses `startPoints` must not have its start points silently ignored.
#[derive(Deserialize, Default)]
struct Args {
    scope: Option<String>,
    #[serde(alias = "startPoints")]
    start_points: Option<Vec<StartPoint>>,
    #[serde(alias = "lookbackDays")]
    lookback_days: Option<i64>,
    promote: Option<bool>,
    #[serde(alias = "scanLimit")]
    scan_limit: Option<i32>,
    /// Strata start. The advertised handle is `start_points` `logged_write`.
    #[serde(default, alias = "nodeId")]
    node_id: Option<String>,
    #[serde(default)]
    node: Option<String>,
    #[serde(default, alias = "loggedWrite")]
    logged_write: Option<String>,
}

/// Recorded causal vocabulary. Incoming edge: source caused target.
const CAUSAL_LINKS: &[&str] = &["closed_by", "derived_from", "evidence_of", "touched"];
const MAX_DEPTH: u32 = 8;

struct Reached {
    id: String,
    depth: u32,
    path: Vec<Hop>,
    /// The start nodes whose walk reached this node (filled when walks merge).
    from: Vec<String>,
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

    let mut rows = start_rows(&args);
    let starts = classify_starts(storage, scope, &mut rows)?;
    if starts.is_empty() {
        let needs_report = refusal(scope, &rows);
        return Ok(walk_payload(
            storage,
            scope,
            WalkOut {
                promote_requested: args.promote.unwrap_or(false),
                starts,
                rows,
                nodes: Vec::new(),
                causes: Vec::new(),
                truncated: false,
                node_cap,
                needs_report: Some(needs_report),
            },
        ));
    }

    let mut walks = Vec::with_capacity(starts.len());
    let mut truncated = false;
    for start in &starts {
        let (reached, cut) = walk_from(storage, scope, start, node_cap)?;
        truncated |= cut;
        walks.push((start.clone(), reached));
    }
    let (mut nodes, mut causes) = merge_walks(walks);
    if nodes.len() > node_cap {
        nodes.truncate(node_cap);
        truncated = true;
    }
    let kept: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    causes.retain(|node| kept.contains(node.id.as_str()));

    Ok(walk_payload(
        storage,
        scope,
        WalkOut {
            promote_requested: args.promote.unwrap_or(false),
            starts,
            rows,
            nodes,
            causes,
            truncated,
            node_cap,
            needs_report: None,
        },
    ))
}

/// Bounded backward BFS from one start node over recorded causal edges.
/// Returns every node reached (the start first) and whether a bound cut it.
fn walk_from(
    storage: &Arc<Storage>,
    scope: &str,
    start_id: &str,
    node_cap: usize,
) -> Result<(Vec<Reached>, bool), String> {
    let mut reached = vec![Reached {
        id: start_id.to_string(),
        depth: 0,
        path: Vec::new(),
        from: Vec::new(),
    }];
    let mut visited = HashSet::from([start_id.to_string()]);
    let mut queue = VecDeque::from([(start_id.to_string(), 0u32, Vec::new())]);
    let mut truncated = false;

    while let Some((current, depth, path)) = queue.pop_front() {
        if depth >= MAX_DEPTH {
            if has_admissible_cause(storage, &current, scope, &visited)? {
                truncated = true;
            }
            continue;
        }
        for (upstream, edge) in upstream_causal(storage, &current)? {
            if visited.contains(upstream.as_str()) {
                continue;
            }
            if !in_scope(storage, &upstream, scope)? {
                continue;
            }
            if reached.len() >= node_cap {
                truncated = true;
                break;
            }
            visited.insert(upstream.clone());
            let mut next_path = path.clone();
            next_path.push(Hop {
                source_id: edge.source_id.clone(),
                target_id: edge.target_id.clone(),
                link_type: edge.link_type.clone(),
            });
            let next_depth = depth + 1;
            queue.push_back((upstream.clone(), next_depth, next_path.clone()));
            reached.push(Reached {
                id: upstream,
                depth: next_depth,
                path: next_path,
                from: Vec::new(),
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
    Ok((reached, truncated))
}

/// Union the per-start walks. `nodes` holds every node once, at its shallowest
/// depth (ties keep the earlier start); `causes` holds every node reached at
/// depth > 0 from some start, so a start that is itself a recorded cause of
/// another start still shows the edge that says so. Both list, for each node,
/// the starts whose walk reached it.
fn merge_walks(walks: Vec<(String, Vec<Reached>)>) -> (Vec<Reached>, Vec<Reached>) {
    fn absorb(into: &mut BTreeMap<String, Reached>, start: &str, node: &Reached) {
        match into.get_mut(&node.id) {
            Some(existing) => {
                if !existing.from.iter().any(|from| from == start) {
                    existing.from.push(start.to_string());
                }
                if node.depth < existing.depth {
                    existing.depth = node.depth;
                    existing.path = node.path.clone();
                }
            }
            None => {
                into.insert(
                    node.id.clone(),
                    Reached {
                        id: node.id.clone(),
                        depth: node.depth,
                        path: node.path.clone(),
                        from: vec![start.to_string()],
                    },
                );
            }
        }
    }
    let mut nodes = BTreeMap::new();
    let mut causes = BTreeMap::new();
    for (start, reached) in &walks {
        for node in reached {
            absorb(&mut nodes, start, node);
            if node.depth > 0 {
                absorb(&mut causes, start, node);
            }
        }
    }
    let order = |map: BTreeMap<String, Reached>| {
        let mut rows: Vec<Reached> = map.into_values().collect();
        rows.sort_by(|a, b| a.depth.cmp(&b.depth).then_with(|| a.id.cmp(&b.id)));
        rows
    };
    (order(nodes), order(causes))
}

/// One start the caller gave, and what the recorded walk did with it. Every
/// start appears in the response, so none is dropped silently.
struct StartRow {
    /// Position in `start_points`; `None` for a top-level handle.
    index: Option<usize>,
    kind: &'static str,
    label: String,
    node_id: Option<String>,
    status: &'static str,
    reason: Option<String>,
}

impl StartRow {
    fn json(&self) -> Value {
        json!({
            "index": self.index,
            "kind": self.kind,
            "label": self.label,
            "node_id": self.node_id,
            "status": self.status,
            "reason": self.reason,
        })
    }
}

struct WalkOut {
    /// Did the caller ask for promote=true? A recorded walk has nothing to
    /// promote, and says so instead of ignoring the request.
    promote_requested: bool,
    /// Distinct in-scope start nodes that were walked, in the order given.
    starts: Vec<String>,
    rows: Vec<StartRow>,
    nodes: Vec<Reached>,
    causes: Vec<Reached>,
    truncated: bool,
    node_cap: usize,
    needs_report: Option<Value>,
}

/// Every start the caller gave, in order: top-level handles first (`node_id`,
/// `node`, `logged_write`), then `start_points`. A `node_id` on a start point
/// of ANY kind names the recorded node to walk from.
fn start_rows(args: &Args) -> Vec<StartRow> {
    let mut rows = Vec::new();
    for handle in [&args.node_id, &args.node, &args.logged_write] {
        if let Some(id) = nonempty(handle) {
            rows.push(StartRow {
                index: None,
                kind: "logged_write",
                label: format!("logged_write {id}"),
                node_id: Some(id),
                status: "pending",
                reason: None,
            });
        }
    }
    for (index, point) in args.start_points.iter().flatten().enumerate() {
        rows.push(StartRow {
            index: Some(index),
            kind: point.kind(),
            label: point.label(),
            node_id: point.record_id().map(str::to_string),
            status: "pending",
            reason: None,
        });
    }
    rows
}

/// Decide the fate of each start: walk it, or say exactly why not. Returns the
/// distinct in-scope nodes to walk, in the order given.
fn classify_starts(
    storage: &Arc<Storage>,
    scope: &str,
    rows: &mut [StartRow],
) -> Result<Vec<String>, String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut walkable = Vec::new();
    for row in rows.iter_mut() {
        let Some(id) = row.node_id.clone() else {
            row.status = "unresolved";
            row.reason = Some(format!(
                "this {} start point carries no node_id; the walk begins at a recorded node, so add node_id (the id of the memory that records this symptom)",
                row.kind
            ));
            continue;
        };
        if !seen.insert(id.clone()) {
            row.status = "duplicate";
            row.reason = Some("names a node an earlier start point already walks".to_string());
        } else if !in_scope(storage, &id, scope)? {
            row.status = "not_in_scope";
            row.reason = Some(format!("node is not recorded in scope '{scope}'"));
        } else {
            row.status = "walked";
            walkable.push(id);
        }
    }
    Ok(walkable)
}

/// The `needs_report` for a walk with nothing to start from. It names the
/// single thing missing and lists every start point with its reason.
fn refusal(scope: &str, rows: &[StartRow]) -> Value {
    let (missing, detail) = if rows.is_empty() {
        (
            "node_id",
            "no start point given; causal_walk does not guess. Pass start_points with a node_id on at least one start point (any kind), or a logged_write start point".to_string(),
        )
    } else if rows.iter().all(|row| row.node_id.is_none()) {
        (
            "node_id",
            "no start point carries a node_id. The walk begins at a recorded node, so add node_id (the id of the memory that records the symptom) to any start point, for example {\"kind\":\"stack_frame\",\"frame\":\"src/a.rs:10\",\"node_id\":\"mem-...\"}".to_string(),
        )
    } else {
        (
            "node",
            format!(
                "no start node is recorded in scope '{scope}'; pass the scope the symptom was written to"
            ),
        )
    };
    json!({
        "missing": [missing],
        "detail": detail,
        "unresolved_start_points": rows
            .iter()
            .filter(|row| row.status != "walked")
            .map(StartRow::json)
            .collect::<Vec<_>>(),
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
    for (upstream, _) in upstream_causal(storage, node_id)? {
        if visited.contains(upstream.as_str()) {
            continue;
        }
        if in_scope(storage, &upstream, scope)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Which end of a recorded causal edge is upstream of `node_id`, if the edge
/// leads upstream from it at all.
///
/// An edge is stored the way its writer names it, so the earlier record is
/// not always the source:
///
/// - `A derived_from B` (a link declared at save, a GhostLink weave): A came
///   from B, so from A the walk goes to B. The same edge seen from B leads
///   downstream, to something built on B, and is not followed.
/// - `E evidence_of S`, `I closed_by F`, `C touched S`: the source is the
///   earlier record, so from the target the walk goes to the source.
///
/// 4.1.0 followed every kind from target to source. For `derived_from` that
/// reported the memories derived from a symptom as its causes and never
/// reached what the symptom itself derives from.
pub(crate) fn upstream_end<'a>(
    edge: &'a vestige_core::ConnectionRecord,
    node_id: &str,
) -> Option<&'a str> {
    if edge.source_id == edge.target_id {
        return None;
    }
    match edge.link_type.as_str() {
        "derived_from" if edge.source_id == node_id => Some(edge.target_id.as_str()),
        "closed_by" | "evidence_of" | "touched" if edge.target_id == node_id => {
            Some(edge.source_id.as_str())
        }
        _ => None,
    }
}

/// The recorded causal edges that lead upstream from `node_id`, each with the
/// id of the memory at the upstream end. Ordered by kind, then id.
fn upstream_causal(
    storage: &Arc<Storage>,
    node_id: &str,
) -> Result<Vec<(String, vestige_core::ConnectionRecord)>, String> {
    let edges = storage
        .get_connections_for_memory(node_id)
        .map_err(|err| err.to_string())?;
    let mut out: Vec<(String, vestige_core::ConnectionRecord)> = edges
        .into_iter()
        .filter_map(|edge| {
            let upstream = upstream_end(&edge, node_id)?.to_string();
            Some((upstream, edge))
        })
        .collect();
    out.sort_by(|a, b| (&a.1.link_type, &a.0).cmp(&(&b.1.link_type, &b.0)));
    Ok(out)
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
            "from": node.from,
        })
    };
    let empty = (walk.needs_report.is_none() && walk.causes.is_empty())
        .then(|| why_empty(storage, scope, &walk.starts));
    json!({
        "tool": "causal_walk",
        "status": "completed",
        "scope": scope,
        "direction": "backward",
        "emptyBecause": empty.as_ref().map(|(why, _)| why.clone()),
        "incomingEdges": empty.map(|(_, edges)| edges),
        "promote": {
            "requested": walk.promote_requested,
            "edges_persisted": 0,
            "note": "A recorded walk follows only edges the log already holds, so there is no inferred trail to record; promote=true changes nothing here.",
        },
        "start": walk.starts.first(),
        "starts": walk.starts,
        "start_points": walk.rows.iter().map(StartRow::json).collect::<Vec<_>>(),
        "truncated": walk.truncated,
        "bounds": { "max_depth": MAX_DEPTH, "max_nodes": walk.node_cap },
        "nodes": walk.nodes.iter().map(&node_json).collect::<Vec<_>>(),
        "causes": walk.causes.iter().map(&node_json).collect::<Vec<_>>(),
        "needs_report": walk.needs_report,
        "note": "Backward BFS over recorded causal edges only, from every start node: from a memory to what it is derived_from, and to the records that are evidence_of it, that it closed, or that touched it. A node_id on any start point names the recorded symptom the walk begins at; start_points reports what happened to each one.",
    })
}

/// Why a walk from real start nodes found no cause, from what the log holds:
/// causal edges whose upstream end is in another scope, causal edges that
/// lead downstream from a start, and edges that touch a start but are not
/// causal. Facts only; nothing is inferred.
fn why_empty(storage: &Arc<Storage>, scope: &str, starts: &[String]) -> (String, Value) {
    let mut other_scope = 0usize;
    let mut downstream = 0usize;
    let mut non_causal: BTreeMap<String, usize> = BTreeMap::new();
    for start in starts {
        let Ok(edges) = storage.get_connections_for_memory(start) else {
            continue;
        };
        for edge in edges {
            if edge.source_id == edge.target_id {
                continue;
            }
            if let Some(upstream) = upstream_end(&edge, start) {
                if !in_scope(storage, upstream, scope).unwrap_or(false) {
                    other_scope += 1;
                }
            } else if CAUSAL_LINKS.contains(&edge.link_type.as_str()) {
                // A causal edge seen from its upstream end: something that
                // derives from, is evidenced by, or closed this memory.
                downstream += 1;
            } else if edge.target_id == *start {
                *non_causal.entry(edge.link_type).or_default() += 1;
            }
        }
    }
    let mut why = format!(
        "no recorded causal edge (closed_by, derived_from, evidence_of, touched) leads upstream from the start node{} in scope '{scope}'",
        if starts.len() == 1 { "" } else { "s" }
    );
    if other_scope > 0 {
        why.push_str(&format!(
            "; {other_scope} causal edge(s) lead upstream into another scope, which this walk does not cross"
        ));
    }
    if downstream > 0 {
        why.push_str(&format!(
            "; {downstream} causal edge(s) lead downstream from here (to memories that came after), which a backward walk does not follow"
        ));
    }
    if !non_causal.is_empty() {
        let kinds: Vec<String> = non_causal
            .iter()
            .map(|(kind, count)| format!("{count} {kind}"))
            .collect();
        why.push_str(&format!(
            "; incoming edges that are not causal: {}",
            kinds.join(", ")
        ));
    }
    why.push_str(". A cause becomes walkable once the symptom records a link to it: save the symptom with links [{kind: 'derived_from', to: <cause id>}], or save the evidence with links [{kind: 'evidence_of', to: <symptom id>}].");
    (
        why,
        json!({
            "causalFromOtherScopes": other_scope,
            "causalDownstream": downstream,
            "nonCausal": non_causal,
        }),
    )
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
    save_link(&storage, &effect, &cause, "derived_from")?;
    save_link(&storage, &decoy, &effect, "projected_to")?;
    save_link(&storage, &downstream, &effect, "derived_from")?;
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
        link(&storage, &effect, &cause, "derived_from");
        link(&storage, &origin, &cause, "evidence_of");
        link(&storage, &same_words, &effect, "projected_to");
        link(&storage, &downstream, &effect, "derived_from");
        link(&storage, &effect, &other, "derived_from");
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
        // The hop is the recorded edge: `effect derived_from cause`.
        assert_eq!(cause_row["path"][0]["source_id"], effect);
        assert_eq!(cause_row["path"][0]["target_id"], cause);
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
        link(&storage, &effect, &smaller, "derived_from");
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
            link(&storage, &effect, cause, "derived_from");
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
    fn causes_of(out: &Value) -> Vec<(String, u64)> {
        out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["id"].as_str().unwrap().to_string(),
                    row["depth"].as_u64().unwrap(),
                )
            })
            .collect()
    }

    fn statuses(out: &Value) -> Vec<&str> {
        out["start_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["status"].as_str().unwrap())
            .collect()
    }

    #[tokio::test]
    async fn node_id_on_every_start_point_kind_walks_the_recorded_cause() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "login handler failed");
        let cause = put(&storage, "user", "commit flipped the auth timeout");
        link(&storage, &effect, &cause, "derived_from");

        let baseline = execute(
            &storage,
            Some(json!({"start_points": [{"kind": "logged_write", "node_id": effect}]})),
        )
        .await
        .unwrap();
        let want = causes_of(&baseline);
        assert_eq!(want, vec![(cause.clone(), 1)], "{baseline}");

        let starts = [
            json!({"kind": "failing_test", "name": "test_login", "node_id": effect}),
            json!({"kind": "stack_frame", "frame": "src/auth.rs:10", "node_id": effect}),
            json!({"kind": "ci_run", "run_id": "run-1", "node_id": effect}),
            json!({"kind": "version_range", "worked_in": "v1", "broke_in": "v2", "repo": "/r", "node_id": effect}),
        ];
        for start in starts {
            let out = execute(&storage, Some(json!({"start_points": [start.clone()]})))
                .await
                .unwrap_or_else(|err| panic!("{start} must be accepted: {err}"));
            assert_eq!(out["status"], "completed", "{start}: {out}");
            assert!(out["needs_report"].is_null(), "{start}: {out}");
            assert_eq!(out["start"], effect, "{start}");
            assert_eq!(causes_of(&out), want, "{start}: {out}");
            assert_eq!(out["start_points"][0]["status"], "walked", "{start}");
            assert_eq!(out["start_points"][0]["kind"], start["kind"], "{start}");
            assert_eq!(out["start_points"][0]["node_id"], effect, "{start}");
        }
    }

    #[tokio::test]
    async fn a_refused_start_point_accepts_the_node_id_the_refusal_asks_for() {
        let (storage, _dir) = open();
        let symptom = put(&storage, "user", "stack frame src/auth.rs:10 panicked");
        let cause = put(&storage, "user", "commit touched src/auth.rs");
        link(&storage, &cause, &symptom, "touched");

        // 1. the bare start point is refused, and the refusal says why
        let bare = json!({"kind": "stack_frame", "frame": "src/auth.rs:10"});
        let refused = execute(&storage, Some(json!({"start_points": [bare.clone()]})))
            .await
            .unwrap();
        assert_eq!(refused["needs_report"]["missing"], json!(["node_id"]));
        let detail = refused["needs_report"]["detail"].as_str().unwrap();
        assert!(detail.contains("node_id"), "{detail}");
        assert_eq!(statuses(&refused), vec!["unresolved"], "{refused}");
        assert_eq!(
            refused["needs_report"]["unresolved_start_points"][0]["kind"],
            "stack_frame"
        );
        assert!(refused["start"].is_null(), "{refused}");
        assert!(causes_of(&refused).is_empty());

        // 2. the same start point carrying that node_id is accepted, not an
        //    "unknown field" error: the refusal and the schema agree
        let mut fixed = bare;
        fixed["node_id"] = json!(symptom);
        let out = execute(&storage, Some(json!({"start_points": [fixed]})))
            .await
            .unwrap();
        assert!(out["needs_report"].is_null(), "{out}");
        assert_eq!(causes_of(&out), vec![(cause.clone(), 1)], "{out}");
        assert_eq!(out["causes"][0]["path"][0]["link_type"], "touched");
        assert_eq!(out["causes"][0]["path"][0]["target_id"], symptom);
    }

    #[tokio::test]
    async fn several_starts_merge_with_provenance() {
        let (storage, _dir) = open();
        let a = put(&storage, "user", "failure A");
        let b = put(&storage, "user", "failure B");
        let cause_a = put(&storage, "user", "cause of A");
        let cause_b = put(&storage, "user", "cause of B");
        let root = put(&storage, "user", "root behind both");
        link(&storage, &a, &cause_a, "derived_from");
        link(&storage, &cause_b, &b, "closed_by");
        link(&storage, &root, &cause_a, "evidence_of");
        link(&storage, &root, &cause_b, "evidence_of");

        let out = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "failing_test", "name": "ta", "node_id": a},
                {"kind": "stack_frame", "frame": "f", "node_id": b},
            ]})),
        )
        .await
        .unwrap();
        assert_eq!(out["start"], a, "start stays the first walked node");
        assert_eq!(out["starts"], json!([a, b]), "{out}");
        assert_eq!(
            causes_of(&out),
            vec![
                (cause_a.clone(), 1),
                (cause_b.clone(), 1),
                (root.clone(), 2)
            ],
            "{out}"
        );
        let by_id = |id: &str| {
            out["causes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == id)
                .unwrap()
                .clone()
        };
        assert_eq!(by_id(&cause_a)["from"], json!([a]));
        assert_eq!(by_id(&cause_b)["from"], json!([b]));
        assert_eq!(
            by_id(&root)["from"],
            json!([a, b]),
            "reached from both starts"
        );
        assert_eq!(
            by_id(&root)["path"][0]["source_id"],
            a,
            "equal depth keeps the earlier start's route"
        );
        let nodes = node_ids(&out);
        for id in [&a, &b, &cause_a, &cause_b, &root] {
            assert!(nodes.contains(id), "{id} missing from {nodes:?}");
        }
        assert_eq!(nodes.len(), 5, "each node appears once: {nodes:?}");
    }

    #[tokio::test]
    async fn a_start_that_is_also_a_recorded_cause_keeps_its_edge() {
        let (storage, _dir) = open();
        let a = put(&storage, "user", "failing test A");
        let b = put(&storage, "user", "stack frame B");
        link(&storage, &a, &b, "derived_from");
        let out = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "failing_test", "name": "ta", "node_id": a},
                {"kind": "stack_frame", "frame": "f", "node_id": b},
            ]})),
        )
        .await
        .unwrap();
        // B is a start (depth 0) AND a recorded cause of A: the edge survives
        assert_eq!(causes_of(&out), vec![(b.clone(), 1)], "{out}");
        // recorded as `a derived_from b`
        assert_eq!(out["causes"][0]["path"][0]["source_id"], a);
        assert_eq!(out["causes"][0]["path"][0]["target_id"], b);
        assert_eq!(out["causes"][0]["from"], json!([a]));
        let nodes = out["nodes"].as_array().unwrap();
        let b_row = nodes.iter().find(|row| row["id"] == b).unwrap();
        assert_eq!(b_row["depth"], 0, "as a start it is shown at depth 0");
        assert_eq!(b_row["from"], json!([a, b]));
    }

    #[tokio::test]
    async fn unresolvable_starts_are_reported_not_dropped() {
        let (storage, _dir) = open();
        let symptom = put(&storage, "user", "good symptom");
        let cause = put(&storage, "user", "its cause");
        let elsewhere = put(&storage, "other", "symptom in another scope");
        link(&storage, &symptom, &cause, "derived_from");

        let out = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "failing_test", "name": "t", "node_id": symptom},
                {"kind": "stack_frame", "frame": "src/x.rs:1"},
                {"kind": "ci_run", "run_id": "r", "node_id": elsewhere},
                {"kind": "logged_write", "node_id": symptom},
            ]})),
        )
        .await
        .unwrap();
        assert!(out["needs_report"].is_null(), "one start walked: {out}");
        assert_eq!(
            statuses(&out),
            vec!["walked", "unresolved", "not_in_scope", "duplicate"],
            "{out}"
        );
        assert_eq!(out["starts"], json!([symptom]));
        assert_eq!(causes_of(&out), vec![(cause, 1)]);
        let reasons: Vec<&str> = out["start_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["reason"].as_str().unwrap_or(""))
            .collect();
        assert!(reasons[1].contains("node_id"), "{reasons:?}");
        assert!(reasons[2].contains("scope 'user'"), "{reasons:?}");
    }

    #[tokio::test]
    async fn nothing_to_walk_refuses_with_the_one_thing_missing() {
        let (storage, _dir) = open();
        let elsewhere = put(&storage, "other", "symptom in another scope");

        for args in [json!({}), json!({"start_points": []})] {
            let out = execute(&storage, Some(args)).await.unwrap();
            assert_eq!(out["needs_report"]["missing"], json!(["node_id"]), "{out}");
            assert!(out["start"].is_null());
            assert_eq!(out["start_points"], json!([]));
        }

        let only_names = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "failing_test", "name": "t"},
                {"kind": "version_range", "worked_in": "a", "broke_in": "b", "repo": "/r"},
            ]})),
        )
        .await
        .unwrap();
        assert_eq!(only_names["needs_report"]["missing"], json!(["node_id"]));
        assert_eq!(
            only_names["needs_report"]["unresolved_start_points"]
                .as_array()
                .unwrap()
                .len(),
            2,
            "{only_names}"
        );

        let wrong_scope = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "stack_frame", "frame": "f", "node_id": elsewhere},
            ]})),
        )
        .await
        .unwrap();
        assert_eq!(wrong_scope["needs_report"]["missing"], json!(["node"]));
        assert_eq!(
            wrong_scope["needs_report"]["unresolved_start_points"][0]["status"],
            "not_in_scope"
        );
        // and the scope the symptom lives in walks it
        let found = execute(
            &storage,
            Some(json!({"scope": "other", "start_points": [
                {"kind": "stack_frame", "frame": "f", "node_id": elsewhere},
            ]})),
        )
        .await
        .unwrap();
        assert!(found["needs_report"].is_null(), "{found}");
        assert_eq!(found["start"], elsewhere);
    }

    #[tokio::test]
    async fn a_misspelled_field_is_still_an_error_that_names_the_real_one() {
        let (storage, _dir) = open();
        let err = execute(
            &storage,
            Some(json!({"start_points": [
                {"kind": "stack_frame", "frame": "f", "nodeid": "mem-1"},
            ]})),
        )
        .await
        .unwrap_err();
        assert!(err.contains("unknown field `nodeid`"), "{err}");
        assert!(err.contains("node_id"), "{err}");
    }

    #[test]
    fn schema_lists_node_id_on_every_start_point_kind() {
        let schema = schema();
        let variants = schema["properties"]["start_points"]["items"]["oneOf"]
            .as_array()
            .unwrap();
        assert_eq!(variants.len(), 5);
        for variant in variants {
            let kind = variant["properties"]["kind"]["const"].as_str().unwrap();
            assert!(
                variant["properties"]["node_id"]["type"] == "string",
                "{kind} must advertise node_id"
            );
            let required: Vec<&str> = variant["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_str().unwrap())
                .collect();
            assert_eq!(
                required.contains(&"node_id"),
                kind == "logged_write",
                "{kind}: node_id is required only where it is the whole start point"
            );
        }
    }
    #[tokio::test]
    async fn camel_case_arguments_are_understood_not_ignored() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "symptom");
        let cause = put(&storage, "user", "cause");
        link(&storage, &effect, &cause, "derived_from");
        for args in [
            json!({"startPoints": [{"kind": "logged_write", "node_id": effect}]}),
            json!({"nodeId": effect}),
            json!({"loggedWrite": effect}),
        ] {
            let out = execute(&storage, Some(args.clone())).await.unwrap();
            assert!(out["needs_report"].is_null(), "{args}: {out}");
            assert_eq!(causes_of(&out), vec![(cause.clone(), 1)], "{args}");
        }
    }

    #[tokio::test]
    async fn promote_on_a_recorded_walk_is_reported_as_a_no_op() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "symptom");
        let cause = put(&storage, "user", "cause");
        link(&storage, &effect, &cause, "derived_from");
        let before = storage.get_all_connections().unwrap().len();
        let out = execute(&storage, Some(json!({"node_id": effect, "promote": true})))
            .await
            .unwrap();
        assert_eq!(out["promote"]["requested"], true, "{out}");
        assert_eq!(out["promote"]["edges_persisted"], 0);
        assert!(out["promote"]["note"].as_str().unwrap().contains("nothing"));
        assert_eq!(storage.get_all_connections().unwrap().len(), before);
        let plain = execute(&storage, Some(json!({"node_id": effect})))
            .await
            .unwrap();
        assert_eq!(plain["promote"]["requested"], false);
    }

    #[tokio::test]
    async fn a_walk_goes_to_what_the_symptom_derives_from_not_to_what_derives_from_it() {
        // The shape `smart_ingest` links record: `new derived_from existing`.
        let (storage, _dir) = open();
        let decision = put(
            &storage,
            "user",
            "decision: cache moved to an in-memory LRU",
        );
        let failure = put(&storage, "user", "failure: sessions dropped under load");
        let postmortem = put(&storage, "user", "postmortem written after the failure");
        let evidence = put(
            &storage,
            "user",
            "log line showing evictions before the drop",
        );
        link(&storage, &failure, &decision, "derived_from");
        link(&storage, &postmortem, &failure, "derived_from");
        link(&storage, &evidence, &failure, "evidence_of");

        let out = execute(&storage, Some(json!({"node_id": failure})))
            .await
            .unwrap();
        let mut found: Vec<String> = causes_of(&out).into_iter().map(|(id, _)| id).collect();
        found.sort();
        let mut expected = vec![decision.clone(), evidence.clone()];
        expected.sort();
        assert_eq!(found, expected, "upstream only: {out}");
        assert!(
            !out.to_string().contains(&postmortem),
            "a memory derived from the failure is downstream, not a cause: {out}"
        );

        // From the decision nothing is upstream, and the reason names the
        // downstream edge instead of calling it a cause.
        let out = execute(&storage, Some(json!({"node_id": decision})))
            .await
            .unwrap();
        assert!(causes_of(&out).is_empty(), "{out}");
        assert_eq!(out["incomingEdges"]["causalDownstream"], 1, "{out}");
        assert!(
            out["emptyBecause"]
                .as_str()
                .unwrap()
                .contains("lead downstream from here"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn an_empty_walk_says_why_from_the_edges_it_saw() {
        let (storage, _dir) = open();
        let symptom = put(&storage, "user", "symptom with no recorded cause");
        let lookalike = put(&storage, "user", "shares words, not causal");
        let elsewhere = put(&storage, "other", "a cause recorded in another scope");
        link(&storage, &lookalike, &symptom, "projected_to");
        link(&storage, &symptom, &elsewhere, "derived_from");

        let out = execute(&storage, Some(json!({"node_id": symptom})))
            .await
            .unwrap();
        assert!(causes_of(&out).is_empty());
        assert!(out["needs_report"].is_null(), "the start was walked: {out}");
        let why = out["emptyBecause"]
            .as_str()
            .expect("an empty walk explains itself");
        assert!(why.contains("no recorded causal edge"), "{why}");
        assert!(
            why.contains("1 causal edge(s) lead upstream into another scope"),
            "{why}"
        );
        assert!(why.contains("1 projected_to"), "{why}");
        assert_eq!(out["incomingEdges"]["causalFromOtherScopes"], 1);
        assert_eq!(out["incomingEdges"]["nonCausal"]["projected_to"], 1);

        // a bare symptom: still explained, with nothing extra claimed
        let bare = put(&storage, "user", "a symptom nothing points at");
        let out = execute(&storage, Some(json!({"node_id": bare})))
            .await
            .unwrap();
        let why = out["emptyBecause"].as_str().unwrap();
        assert!(
            !why.contains("another scope") && !why.contains("not causal"),
            "{why}"
        );

        // a walk that finds a cause does not claim emptiness
        let cause = put(&storage, "user", "the real cause");
        link(&storage, &cause, &symptom, "touched");
        let found = execute(&storage, Some(json!({"node_id": symptom})))
            .await
            .unwrap();
        assert_eq!(causes_of(&found), vec![(cause, 1)]);
        assert!(found["emptyBecause"].is_null(), "{found}");
        assert!(found["incomingEdges"].is_null());
    }
}
