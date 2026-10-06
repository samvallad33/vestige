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
//! On a Strata log the walk is a bounded backward BFS over recorded causal
//! edges only. `stack_frame` (`path:line`) and `failing_test` (a test file
//! path) resolve to `file:` anchors `ingest_repo` recorded, by exact path,
//! then a `/`-bounded suffix of at least two components, then blame of the
//! line at the failure revision. A registry or vendor path that names
//! `crate-version` resolves to the `pkg:` anchor of that exact version, and
//! a lockfile bump is reached from that anchor. That path resolver is the
//! one walk start resolver. `vestige connect` and auto-connect still join
//! memories on byte-exact identities; they do not resolve frames.
//! `node_id` names the failure memory. `ci_run` does not resolve by name.
//! `version_range` bounds candidates with `git rev-list --first-parent` and
//! is not a name search. Shared names are not edges. A child of a visited
//! commit that `corrects` an older commit contributes that older commit. A
//! recorded `corrects` edge whose target is an ancestor of the failure
//! revision contributes that target too, including when the revert sits on
//! another branch.
//!
//! ## One `touched` edge per path
//!
//! `derived_from`, `evidence_of`, `closed_by` and `corrects` are lineage: a
//! writer declared that one record came from, evidences, closed or corrected
//! another, and a chain of them is a chain of declarations. `touched` is
//! weaker. It records that two memories name the same exact thing (it is
//! what `vestige connect` and the ingest-time auto-connect write), and
//! naming the same thing is not transitive. A path therefore follows lineage
//! freely, up to the depth bound, and at most one `touched` edge. A walk
//! that starts at a `file:` or `pkg:` anchor uses that one edge to reach the
//! commits ingest recorded against it; further hops from those commits are
//! lineage. The `touched` edges the walk saw and did not follow, and the
//! memories that lie only behind them, are counted in `not_followed`.
//!
//! ## Order
//!
//! When a frame, a package anchor, or a failure revision resolved, candidates
//! are ordered by recorded structure: reverted after the failure, blame of
//! the line, the failing hunk, depth, edge kind (`corrects`, then `touched`,
//! then `derived_from`), the time gap to the failure, then id. Otherwise
//! they are ordered by depth, then by how many distinct exact identities
//! each shares with the start (`auto_connect::extract_identities`: tag,
//! path, commit, issue, url; byte-exact, never words), then by how rare
//! those identities are, then by id. Every candidate still carries the
//! identities it shares (`joined_on`). A hub tag is listed apart and not
//! counted. The order arranges hypotheses; it proves nothing.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::auto_connect::{self, Identity, IdentityKind, RankedCandidate};
use vestige_core::Storage;
use vestige_core::advanced::causal_walk::{
    self as core_causal_walk, CausalWalkRequest, StartPoint, persist_evidence_edges, walk_storage,
};

/// The optional `node_id` every start-point variant accepts.
fn symptom_node_id() -> Value {
    json!({
        "type": "string",
        "description": "Optional. The id of the memory that records this symptom. On a Strata log, stack_frame and failing_test also resolve to recorded file anchors by exact path; node_id is the failure memory whose derived_from commit is the revision blame uses. version_range bounds candidates with git rev-list and does not need its own node_id. ci_run still needs node_id."
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
                "description": "Explicit evidence handles; at least one is required or the walk returns needs_report (it never guesses). stack_frame and failing_test resolve by exact path to file anchors ingest_repo recorded. version_range limits commit candidates to git rev-list --first-parent worked_in..broke_in in the local repo. It is not a version-name search. ci_run walks the node_id memory.",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "failing_test"},
                                "name": {"type": "string", "description": "Test file path (repo-relative). Resolved to a recorded file anchor by exact path identity. A bare test name is not matched to a file."},
                                "node_id": symptom_node_id()
                            },
                            "required": ["kind", "name"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "const": "stack_frame"},
                                "frame": {"type": "string", "description": "\"file:line\" or \"file:line:col\". Resolved to a recorded file anchor by exact path identity; the line is blamed at the failure revision."},
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
                                "worked_in": {"type": "string", "description": "Last-known-good revision (tag or SHA). Exclusive start of git rev-list --first-parent."},
                                "broke_in": {"type": "string", "description": "First-bad revision (tag or SHA). Inclusive end of git rev-list --first-parent."},
                                "repo": {"type": "string", "description": "Path to the local git repository. Nothing is fetched."},
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

/// Recorded causal vocabulary.
///
/// `corrects` is included so a git revert (source = revert, target = reverted)
/// is an upstream edge. The same kind also stores a lesson as source = earlier.
const CAUSAL_LINKS: &[&str] = &[
    "closed_by",
    "corrects",
    "derived_from",
    "evidence_of",
    "touched",
];
const MAX_DEPTH: u32 = 8;
/// The co-touch kind: two memories name the same thing. Followed at most once
/// on a path (see the module docs).
const CO_TOUCH: &str = "touched";

struct Reached {
    id: String,
    depth: u32,
    path: Vec<Hop>,
    /// The start node `path` begins at.
    start: String,
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
    let git = resolve_git_frames(storage, scope, &starts, &mut rows)?;
    let walked_starts = if starts.is_empty() {
        git.anchors.clone()
    } else {
        starts.clone()
    };
    if starts.is_empty() && git.anchors.is_empty() {
        let needs_report = refusal(scope, &rows);
        // A version range is a bound, not a start. Report whether the local
        // rev-list resolved even when nothing was walked.
        let mut nodes = Vec::new();
        let mut causes = Vec::new();
        let range = apply_version_range(
            storage,
            args.start_points.as_deref(),
            &mut nodes,
            &mut causes,
        );
        return Ok(walk_payload(
            storage,
            scope,
            WalkOut {
                promote_requested: args.promote.unwrap_or(false),
                starts: walked_starts,
                rows,
                nodes,
                causes,
                ranking: Ranking::default(),
                not_followed: NotFollowed::default(),
                truncated: false,
                node_cap,
                needs_report: Some(needs_report),
                structure: BTreeMap::new(),
                upstream_note: None,
                ancestry_note: None,
                range,
                structural_rank: false,
            },
        ));
    }

    let mut walks = Vec::with_capacity(starts.len() + git.anchors.len());
    let mut truncated = false;
    let mut held: BTreeSet<HeldEdge> = BTreeSet::new();
    for start in starts.iter().chain(git.anchors.iter()) {
        let walked = walk_from(storage, scope, start, node_cap)?;
        truncated |= walked.truncated;
        held.extend(walked.held);
        walks.push((start.clone(), walked.reached));
    }
    let (mut nodes, mut causes) = merge_walks(walks);
    // An edge one start did not follow hides nothing if another start's walk
    // reached the memory behind it.
    let reached: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    held.retain(|edge| !reached.contains(edge.upstream.as_str()));
    let ancestry_note = extend_revert_ancestry(storage, scope, &git, &mut nodes, &mut causes);
    let reached_after: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    held.retain(|edge| !reached_after.contains(edge.upstream.as_str()));
    let held_shared = shared_on(storage, &held);
    // Structural rank applies when a frame, a package anchor, or the failure
    // revision resolved. Identity rank is the order otherwise. The identities
    // each cause shares are attached either way.
    let structural = git.active;
    let structure = if structural {
        let admissible: HashSet<String> = causes
            .iter()
            .filter(|cause| git_admissible(storage, &cause.id, &git))
            .map(|cause| cause.id.clone())
            .collect();
        causes.retain(|cause| admissible.contains(&cause.id));
        nodes.retain(|node| node.depth == 0 || admissible.contains(&node.id));
        let structure = git_structure(storage, &causes, &git);
        causes.sort_by(|a, b| {
            git_rank(storage, a, &git)
                .cmp(&git_rank(storage, b, &git))
                .then_with(|| a.id.cmp(&b.id))
        });
        structure
    } else {
        BTreeMap::new()
    };
    let upstream_note = upstream_note_for(storage, &rows, &causes);
    let range = apply_version_range(
        storage,
        args.start_points.as_deref(),
        &mut nodes,
        &mut causes,
    );
    if nodes.len() > node_cap {
        nodes.truncate(node_cap);
        truncated = true;
    }
    let kept: HashSet<&str> = nodes.iter().map(|node| node.id.as_str()).collect();
    causes.retain(|node| kept.contains(node.id.as_str()));
    // One read of the scope counts the carriers of everything reported: the
    // starts' identities and the ones the unfollowed edges rest on.
    let also_counted: BTreeSet<Identity> = held_shared.iter().flatten().cloned().collect();
    let ranking = rank_causes(storage, scope, &mut causes, also_counted, !structural)?;
    let mut not_followed = describe_not_followed(&held, &held_shared, &ranking);
    not_followed.held = held_leads(&held, &held_shared, &ranking, &nodes, node_cap);

    Ok(walk_payload(
        storage,
        scope,
        WalkOut {
            promote_requested: args.promote.unwrap_or(false),
            starts: walked_starts,
            rows,
            nodes,
            causes,
            ranking,
            not_followed,
            truncated,
            node_cap,
            needs_report: None,
            structure,
            upstream_note,
            ancestry_note,
            range,
            structural_rank: structural,
        },
    ))
}

/// A `touched` edge the walk saw and did not follow: the path it would have
/// extended had already followed one. `upstream` is the memory behind it,
/// `from` the reached memory it leads into.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HeldEdge {
    upstream: String,
    from: String,
}

/// One start's walk: every node reached (the start first), whether a bound
/// cut it, and the `touched` edges it did not follow whose upstream memory it
/// did not reach another way.
struct Walked {
    reached: Vec<Reached>,
    truncated: bool,
    held: BTreeSet<HeldEdge>,
}

/// Bounded backward BFS from one start node over recorded causal edges.
///
/// A path follows lineage edges freely and at most one `touched` edge, so the
/// search runs over `(memory, has this path used its touched edge)`. A memory
/// reached both ways is expanded both ways, the unused state covering the
/// used one; it is listed once, at the depth it was first reached. What is
/// reached is therefore every memory with an admissible path of at most
/// `MAX_DEPTH` hops, whatever order the edges are read in.
fn walk_from(
    storage: &Arc<Storage>,
    scope: &str,
    start_id: &str,
    node_cap: usize,
) -> Result<Walked, String> {
    let mut reached = vec![Reached {
        id: start_id.to_string(),
        depth: 0,
        path: Vec::new(),
        start: start_id.to_string(),
        from: Vec::new(),
    }];
    let mut listed = HashSet::from([start_id.to_string()]);
    let mut expanded = HashSet::from([(start_id.to_string(), false)]);
    let mut queue = VecDeque::from([(start_id.to_string(), 0u32, Vec::new(), false)]);
    let mut held: BTreeSet<HeldEdge> = BTreeSet::new();
    let mut truncated = false;

    while let Some((current, depth, path, used_touch)) = queue.pop_front() {
        if depth >= MAX_DEPTH {
            if has_admissible_cause(storage, &current, scope, used_touch, &listed)? {
                truncated = true;
            }
            continue;
        }
        let mut steps = Vec::new();
        for (upstream, edge) in upstream_causal(storage, &current)? {
            let co_touch = edge.link_type == CO_TOUCH;
            if co_touch && used_touch {
                if in_scope(storage, &upstream, scope)? {
                    held.insert(HeldEdge {
                        upstream,
                        from: current.clone(),
                    });
                }
                continue;
            }
            steps.push((
                upstream,
                Hop {
                    source_id: edge.source_id,
                    target_id: edge.target_id,
                    link_type: edge.link_type,
                },
                co_touch,
            ));
        }
        // A child's recorded `corrects` edge is lineage, not a second touch.
        for (upstream, hop) in child_corrections(storage, &current)? {
            steps.push((upstream, hop, false));
        }
        for (upstream, hop, co_touch) in steps {
            let next_used = used_touch || co_touch;
            // Already expanded with the touched edge unused (which covers
            // both states), or in this same state.
            if expanded.contains(&(upstream.clone(), false))
                || (next_used && expanded.contains(&(upstream.clone(), true)))
            {
                continue;
            }
            if !in_scope(storage, &upstream, scope)? {
                continue;
            }
            let unlisted = !listed.contains(upstream.as_str());
            if unlisted && reached.len() >= node_cap {
                truncated = true;
                break;
            }
            expanded.insert((upstream.clone(), next_used));
            let mut next_path = path.clone();
            next_path.push(hop);
            let next_depth = depth + 1;
            queue.push_back((upstream.clone(), next_depth, next_path.clone(), next_used));
            if unlisted {
                listed.insert(upstream.clone());
                reached.push(Reached {
                    id: upstream,
                    depth: next_depth,
                    path: next_path,
                    start: start_id.to_string(),
                    from: Vec::new(),
                });
            }
        }
        if reached.len() >= node_cap {
            for (pending, _, _, pending_used) in &queue {
                if has_admissible_cause(storage, pending, scope, *pending_used, &listed)? {
                    truncated = true;
                    break;
                }
            }
            break;
        }
    }
    // A held edge hides nothing if its upstream memory was reached anyway.
    held.retain(|edge| !listed.contains(edge.upstream.as_str()));
    Ok(Walked {
        reached,
        truncated,
        held,
    })
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
                    existing.start = node.start.clone();
                }
            }
            None => {
                into.insert(
                    node.id.clone(),
                    Reached {
                        id: node.id.clone(),
                        depth: node.depth,
                        path: node.path.clone(),
                        start: node.start.clone(),
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

/// What one candidate shares with the start memory its path begins at.
struct Evidence {
    /// The exact identities both record, each with the number of memories of
    /// the scope that record it. Hub tags are not here. Tags first, then
    /// sorted.
    shared: Vec<(Identity, usize)>,
    /// Distinct values among `shared` (a tag that is also a path counts once).
    count: usize,
    /// Tags both carry that more than half the scope carries too: listed,
    /// not counted.
    hub_tags: Vec<(Identity, usize)>,
}

/// How a walk's candidates were ordered: what each shares with its start,
/// the size of the scope the carrier counts are out of, and the carriers of
/// every identity the walk reports.
#[derive(Default)]
struct Ranking {
    scope_size: usize,
    evidence: BTreeMap<String, Evidence>,
    carriers: BTreeMap<Identity, usize>,
}

impl Ranking {
    fn carried_by(&self, identity: &Identity) -> usize {
        self.carriers.get(identity).copied().unwrap_or(0)
    }

    /// The hub rule, on the scope as it is now.
    fn is_hub(&self, identity: &Identity) -> bool {
        identity.kind == IdentityKind::Tag
            && auto_connect::is_hub_tag(self.carried_by(identity), self.scope_size)
    }
}

/// The order of a walk that did not resolve a frame, a package anchor, or a
/// failure revision.
const RANK_ORDER: &str = "depth, then distinct exact identities shared with the start (more first), then rarer identities first (fewer carriers in the scope), then id; hub tags are not counted";

/// The order of a walk whose frame, package anchor, or failure revision
/// resolved. Recorded structure only.
const STRUCTURAL_RANK_ORDER: &str = "structural: reverted after the failure, then blame of the reported line, then the failing hunk, then depth, then edge kind (corrects, then touched, then derived_from), then the time gap to the failure, then id";

/// What one candidate is ordered by within a depth: `RankedCandidate::strength`.
type Strength = (Reverse<usize>, Vec<usize>);

/// The exact identities one memory records. A memory that cannot be read
/// back records nothing.
fn identities_of(storage: &Arc<Storage>, id: &str) -> Result<BTreeSet<Identity>, String> {
    Ok(storage
        .get_node(id)
        .map_err(|err| err.to_string())?
        .map(|node| {
            auto_connect::extract_identities(&node.content, &node.tags)
                .into_iter()
                .collect()
        })
        .unwrap_or_default())
}

/// Order the candidates and return what each shares with its start.
///
/// Depth first, as the walk found them. Within a depth: more distinct exact
/// identities shared with the start first, then the rarer identities first,
/// then id (`RankedCandidate::strength`, the same order an ingest uses when
/// its edge budget has to choose). The scope is read once, to count the
/// carriers of the starts' identities (and of `also_counted`, which the
/// caller reports) and to learn which candidate records which of them.
fn rank_causes(
    storage: &Arc<Storage>,
    scope: &str,
    causes: &mut [Reached],
    also_counted: BTreeSet<Identity>,
    reorder: bool,
) -> Result<Ranking, String> {
    if causes.is_empty() {
        return Ok(Ranking::default());
    }
    // The identities of each start some candidate's path begins at.
    let mut of_start: BTreeMap<String, BTreeSet<Identity>> = BTreeMap::new();
    for cause in causes.iter() {
        if !of_start.contains_key(&cause.start) {
            of_start.insert(cause.start.clone(), identities_of(storage, &cause.start)?);
        }
    }
    let mut wanted = also_counted;
    wanted.extend(of_start.values().flatten().cloned());
    let scan = auto_connect::scan_scope(storage.as_ref(), scope, &wanted)?;
    let recorded: BTreeMap<&str, &[Identity]> = scan
        .holders
        .iter()
        .map(|holder| (holder.id.as_str(), holder.identities.as_slice()))
        .collect();
    let mut ranking = Ranking {
        scope_size: scan.scope_size,
        evidence: BTreeMap::new(),
        carriers: scan.carriers(),
    };

    let nothing = BTreeSet::new();
    let mut strength: Option<BTreeMap<String, Strength>> = reorder.then(BTreeMap::new);
    for cause in causes.iter() {
        let start = of_start.get(&cause.start).unwrap_or(&nothing);
        let mut shared: Vec<(Identity, usize)> = Vec::new();
        let mut hub_tags: Vec<(Identity, usize)> = Vec::new();
        for identity in recorded.get(cause.id.as_str()).copied().unwrap_or_default() {
            if !start.contains(identity) {
                continue;
            }
            let carried_by = ranking.carried_by(identity);
            if ranking.is_hub(identity) {
                hub_tags.push((identity.clone(), carried_by));
            } else {
                shared.push((identity.clone(), carried_by));
            }
        }
        let candidate = RankedCandidate {
            id: cause.id.clone(),
            shared,
        };
        let (count, rarity) = candidate.strength();
        ranking.evidence.insert(
            cause.id.clone(),
            Evidence {
                count: count.0,
                shared: candidate.shared,
                hub_tags,
            },
        );
        if let Some(strength) = strength.as_mut() {
            strength.insert(cause.id.clone(), (count, rarity));
        }
    }
    // A structurally ranked walk keeps the order `git_rank` already set.
    // The evidence above is still attached to each cause.
    if let Some(strength) = strength.as_ref() {
        causes.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then_with(|| strength.get(&a.id).cmp(&strength.get(&b.id)))
                .then_with(|| a.id.cmp(&b.id))
        });
    }
    Ok(ranking)
}

/// The `touched` edges a walk did not follow, described from what the log
/// holds: how many there are, how many memories lie only behind them, and
/// the exact identities the two ends of those edges share.
#[derive(Default)]
struct NotFollowed {
    edges: usize,
    memories: usize,
    /// `(identity, its carriers in the scope, edges whose two ends both
    /// record it)`. Most edges first, then by identity. Hub tags are left
    /// out: they join nothing.
    shared: Vec<(Identity, usize, usize)>,
    /// Edges whose two ends share no exact identity, or only hub tags.
    no_counted_identity: usize,
    /// The memories behind those edges, one per memory, listed so a tool
    /// that tests hypotheses (`vestige prove`) can try them after the
    /// causes. Not ranked with the causes and not followed.
    held: Vec<HeldLead>,
}

/// One memory behind a `touched` edge the walk did not follow: the reached
/// memory it hangs from, that memory's depth, and the exact identities the
/// two share (hub tags left out). Listed, never followed: nothing is said
/// about it until a test says it.
struct HeldLead {
    upstream: String,
    from: String,
    from_depth: u32,
    shared: Vec<(Identity, usize)>,
}

/// The held leads of a walk: for every memory behind an unfollowed edge,
/// the edge that hangs it closest to the start (smallest depth, then most
/// shared identities, then the id of the reached end). Ordered the way the
/// causes are: by the depth it would have, then more shared identities
/// first, then id. Capped at `cap`, the walk's own node bound.
fn held_leads(
    held: &BTreeSet<HeldEdge>,
    held_shared: &[BTreeSet<Identity>],
    ranking: &Ranking,
    nodes: &[Reached],
    cap: usize,
) -> Vec<HeldLead> {
    let depth_of: BTreeMap<&str, u32> = nodes
        .iter()
        .map(|node| (node.id.as_str(), node.depth))
        .collect();
    let mut best: BTreeMap<&str, HeldLead> = BTreeMap::new();
    for (edge, shared) in held.iter().zip(held_shared) {
        let Some(&from_depth) = depth_of.get(edge.from.as_str()) else {
            continue;
        };
        let mut shared: Vec<(Identity, usize)> = shared
            .iter()
            .filter(|identity| !ranking.is_hub(identity))
            .map(|identity| (identity.clone(), ranking.carried_by(identity)))
            .collect();
        shared.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let candidate = HeldLead {
            upstream: edge.upstream.clone(),
            from: edge.from.clone(),
            from_depth,
            shared,
        };
        let better = match best.get(edge.upstream.as_str()) {
            None => true,
            Some(current) => {
                (
                    candidate.from_depth,
                    Reverse(candidate.shared.len()),
                    &candidate.from,
                ) < (
                    current.from_depth,
                    Reverse(current.shared.len()),
                    &current.from,
                )
            }
        };
        if better {
            best.insert(edge.upstream.as_str(), candidate);
        }
    }
    let mut out: Vec<HeldLead> = best.into_values().collect();
    out.sort_by(|a, b| {
        (a.from_depth, Reverse(a.shared.len()), &a.upstream).cmp(&(
            b.from_depth,
            Reverse(b.shared.len()),
            &b.upstream,
        ))
    });
    out.truncate(cap);
    out
}

/// Most identities listed for the edges a walk did not follow. The count of
/// the rest is reported, so nothing is dropped silently.
const NOT_FOLLOWED_LISTED: usize = 10;

/// Why a `touched` edge was not followed, stated once.
const NOT_FOLLOWED_REASON: &str = "a touched edge records that two memories name the same exact thing, which does not carry over a second step, so a path follows at most one touched edge; these edges lead on from memories already reached over one";

/// For each edge the walk did not follow, in order, the exact identities its
/// two ends both record.
fn shared_on(storage: &Arc<Storage>, held: &BTreeSet<HeldEdge>) -> Vec<BTreeSet<Identity>> {
    let mut identities: BTreeMap<&str, BTreeSet<Identity>> = BTreeMap::new();
    for id in held
        .iter()
        .flat_map(|edge| [edge.upstream.as_str(), edge.from.as_str()])
    {
        identities
            .entry(id)
            .or_insert_with(|| identities_of(storage, id).unwrap_or_default());
    }
    held.iter()
        .map(|edge| {
            match (
                identities.get(edge.upstream.as_str()),
                identities.get(edge.from.as_str()),
            ) {
                (Some(upstream), Some(from)) => upstream.intersection(from).cloned().collect(),
                _ => BTreeSet::new(),
            }
        })
        .collect()
}

/// Count the edges a walk did not follow and what their ends share, with
/// the carriers `ranking` counted for those identities.
fn describe_not_followed(
    held: &BTreeSet<HeldEdge>,
    held_shared: &[BTreeSet<Identity>],
    ranking: &Ranking,
) -> NotFollowed {
    let mut on: BTreeMap<&Identity, usize> = BTreeMap::new();
    let mut no_counted_identity = 0usize;
    for shared in held_shared {
        let mut any = false;
        for identity in shared.iter().filter(|identity| !ranking.is_hub(identity)) {
            *on.entry(identity).or_insert(0) += 1;
            any = true;
        }
        if !any {
            no_counted_identity += 1;
        }
    }
    let mut shared: Vec<(Identity, usize, usize)> = on
        .into_iter()
        .map(|(identity, edges)| (identity.clone(), ranking.carried_by(identity), edges))
        .collect();
    shared.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    NotFollowed {
        edges: held.len(),
        memories: held
            .iter()
            .map(|edge| edge.upstream.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        shared,
        no_counted_identity,
        held: Vec::new(),
    }
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
    /// Repo-relative path a `stack_frame` or `failing_test` named.
    locator: Option<PathLoc>,
}

#[derive(Clone)]
struct PathLoc {
    path: String,
    line: Option<u32>,
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
    /// What each candidate shares with its start, by candidate id.
    ranking: Ranking,
    /// The `touched` edges the walk did not follow.
    not_followed: NotFollowed,
    truncated: bool,
    node_cap: usize,
    needs_report: Option<Value>,
    /// Structural rank facts for git-resolved causes, keyed by node id.
    structure: BTreeMap<String, Value>,
    /// Why an upstream crate range was not recorded. Local files only.
    upstream_note: Option<String>,
    /// `worked_in..broke_in` bound, including how many commits it removed.
    range: Option<Value>,
    /// Why a revert's ancestry was not checked. Local objects only.
    ancestry_note: Option<String>,
    /// True when candidates were ordered by recorded git structure. False
    /// when they were ordered by depth and exact identities.
    structural_rank: bool,
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
                locator: None,
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
            locator: locator_of(point),
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
            if row.locator.is_some() {
                // Exact path resolution runs after this pass.
                row.status = "pending";
                continue;
            }
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
    let file_miss = rows.iter().any(|row| {
        row.reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with("no recorded file anchor"))
    });
    let (missing, detail) = if rows.is_empty() {
        (
            "node_id",
            "no start point given; causal_walk does not guess. Pass start_points with a node_id on at least one start point (any kind), or a logged_write start point".to_string(),
        )
    } else if file_miss && rows.iter().all(|row| row.node_id.is_none()) {
        (
            "file",
            rows.iter()
                .find_map(|row| row.reason.clone())
                .unwrap_or_else(|| "no recorded file anchor matches the path".to_string()),
        )
    } else if rows.iter().all(|row| row.node_id.is_none()) {
        (
            "node_id",
            "no start point carries a node_id. The walk begins at a recorded node, so add node_id (the id of the memory that records the symptom) to any start point, for example {\"kind\":\"stack_frame\",\"frame\":\"src/a.rs:10\",\"node_id\":\"mem-...\"}. A stack_frame or failing_test that names a repo-relative path resolves without node_id once ingest_repo has recorded that file.".to_string(),
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

/// Does a recorded edge the walk may still follow lead from `node_id` to a
/// memory it has not listed? `used_touch` says whether the path to `node_id`
/// has used its one `touched` edge.
fn has_admissible_cause(
    storage: &Arc<Storage>,
    node_id: &str,
    scope: &str,
    used_touch: bool,
    listed: &HashSet<String>,
) -> Result<bool, String> {
    for (upstream, edge) in upstream_causal(storage, node_id)? {
        if listed.contains(upstream.as_str()) || (used_touch && edge.link_type == CO_TOUCH) {
            continue;
        }
        if in_scope(storage, &upstream, scope)? {
            return Ok(true);
        }
    }
    for (upstream, _) in child_corrections(storage, node_id)? {
        if listed.contains(upstream.as_str()) {
            continue;
        }
        if in_scope(storage, &upstream, scope)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The commit a child of `node_id` reverts. The child itself is not a cause:
/// the hop is the recorded `corrects` edge, from the revert to the older commit.
fn child_corrections(storage: &Arc<Storage>, node_id: &str) -> Result<Vec<(String, Hop)>, String> {
    let edges = storage
        .get_connections_for_memory(node_id)
        .map_err(|err| err.to_string())?;
    let mut children = Vec::new();
    for edge in &edges {
        if edge.link_type == "derived_from"
            && edge.target_id == node_id
            && edge.source_id != node_id
        {
            children.push(edge.source_id.clone());
        }
    }
    children.sort();
    children.dedup();
    let mut out = Vec::new();
    for child in children {
        let child_edges = storage
            .get_connections_for_memory(&child)
            .map_err(|err| err.to_string())?;
        for edge in child_edges {
            if edge.link_type != "corrects" || edge.source_id != child || edge.target_id == node_id
            {
                continue;
            }
            out.push((
                edge.target_id.clone(),
                Hop {
                    source_id: edge.source_id,
                    target_id: edge.target_id,
                    link_type: edge.link_type,
                },
            ));
        }
    }
    out.sort_by(|left, right| left.0.cmp(&right.0));
    out.dedup_by(|left, right| left.0 == right.0);
    Ok(out)
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
        // Git revert: source is the revert, target is the older commit.
        // Lesson: source is the earlier record, target is the failure.
        "corrects" if edge.source_id == node_id => Some(edge.target_id.as_str()),
        "corrects" if edge.target_id == node_id => Some(edge.source_id.as_str()),
        "closed_by" | "evidence_of" | "touched" if edge.target_id == node_id => {
            Some(edge.source_id.as_str())
        }
        // Package anchors are not memories. `commit supersedes pkg@old` is the
        // lockfile's old version; only that anchor walks back to the commit.
        "supersedes" if edge.target_id == node_id && node_id.starts_with("pkg:") => {
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
        let mut row = json!({
            "id": node.id,
            "depth": node.depth,
            "content": content_of(storage, &node.id),
            "path": node.path.iter().map(|hop| json!({
                "source_id": hop.source_id,
                "target_id": hop.target_id,
                "link_type": hop.link_type,
            })).collect::<Vec<_>>(),
            "from": node.from,
        });
        if let Some(structure) = walk.structure.get(&node.id) {
            row["structure"] = structure.clone();
        }
        row
    };
    // `kind:value` with the number of memories of the scope that record it.
    let with_carriers = |entries: &[(Identity, usize)]| -> Vec<Value> {
        entries
            .iter()
            .map(|(identity, carriers)| {
                json!({ "identity": identity.to_string(), "carriers": carriers })
            })
            .collect()
    };
    let cause_json = |(index, node): (usize, &Reached)| {
        let mut value = node_json(node);
        value["rank"] = json!(index + 1);
        if let Some(evidence) = walk.ranking.evidence.get(&node.id) {
            value["start"] = json!(node.start);
            value["shared_count"] = json!(evidence.count);
            value["joined_on"] = json!(with_carriers(&evidence.shared));
            value["not_counted_hub_tags"] = json!(with_carriers(&evidence.hub_tags));
        }
        value
    };
    let ranking = (!walk.causes.is_empty()).then(|| {
        json!({
            "order": if walk.structural_rank { STRUCTURAL_RANK_ORDER } else { RANK_ORDER },
            "scope_size": walk.ranking.scope_size,
            "joined_on": "per cause: the exact identities it records that its start records too, each with its carriers out of scope_size; a depth-1 cause is linked to the start itself, a deeper one is only compared with it",
            "touched": "a path follows lineage edges (derived_from, evidence_of, closed_by) freely and at most one touched edge",
            "evidence_status": "hypothesis",
        })
    });
    let not_followed = (walk.not_followed.edges > 0).then(|| {
        json!({
            "touched_edges": walk.not_followed.edges,
            "memories": walk.not_followed.memories,
            "shared": walk
                .not_followed
                .shared
                .iter()
                .take(NOT_FOLLOWED_LISTED)
                .map(|(identity, carriers, edges)| {
                    json!({
                        "identity": identity.to_string(),
                        "carriers": carriers,
                        "edges": edges,
                    })
                })
                .collect::<Vec<_>>(),
            "shared_identities": walk.not_followed.shared.len(),
            "no_counted_identity": walk.not_followed.no_counted_identity,
            "scope_size": walk.ranking.scope_size,
            "reason": NOT_FOLLOWED_REASON,
            "held": walk.not_followed.held.iter().map(|lead| json!({
                "memory": lead.upstream,
                "from": lead.from,
                "depth": lead.from_depth + 1,
                "shared": with_carriers(&lead.shared),
                "content": content_of(storage, &lead.upstream),
            })).collect::<Vec<_>>(),
            "held_note": "the memories behind those edges, one touched edge beyond a reached memory, in the order the causes use; not ranked with them and not followed. A tool that tests hypotheses may try them after the causes",
        })
    });
    let empty = (walk.needs_report.is_none() && walk.causes.is_empty())
        .then(|| why_empty(storage, scope, &walk.starts));
    let empty_because = empty.as_ref().map(|(why, _)| {
        let mut text = match &walk.upstream_note {
            Some(note) => format!("{why}. {note}"),
            None => why.clone(),
        };
        if let Some(note) = &walk.ancestry_note {
            text = format!("{text}. {note}");
        }
        text
    });
    json!({
        "tool": "causal_walk",
        "status": "completed",
        "scope": scope,
        "direction": "backward",
        "emptyBecause": empty_because,
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
        "causes": walk.causes.iter().enumerate().map(cause_json).collect::<Vec<_>>(),
        "ranking": ranking,
        "not_followed": not_followed,
        "needs_report": walk.needs_report,
        "upstreamSkipped": walk.upstream_note,
        "ancestrySkipped": walk.ancestry_note,
        "range": walk.range,
        "note": "Backward BFS over recorded causal edges only, from every start node and from file or package anchors a stack_frame or failing_test resolved. From a memory to what it is derived_from, along corrects (a git revert to the commit it names, or a lesson to the failure), and to the records that are evidence_of it, that it closed, or that touched it. A path follows lineage freely and at most one touched edge; not_followed counts the rest. When a frame, package anchor, or failure revision resolved, rank is structural (reverted after the failure, blame of the line, hunk, hop, edge kind). Otherwise rank is depth, then exact identities shared with the start. A node_id on any start point names the recorded symptom.",
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
        "no recorded causal edge (closed_by, corrects, derived_from, evidence_of, touched) leads upstream from the start node{} in scope '{scope}'",
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

struct GitQuery {
    anchors: Vec<String>,
    files: Vec<String>,
    line: Option<u32>,
    failure_commit_id: Option<String>,
    failure_time: Option<DateTime<Utc>>,
    blame_shas: HashSet<String>,
    failure_sha: Option<String>,
    root: Option<PathBuf>,
    active: bool,
}

struct Revision {
    id: String,
    sha: String,
    time: DateTime<Utc>,
    root: Option<PathBuf>,
}

/// Resolve `stack_frame` / `failing_test` paths to recorded `file:` anchors and,
/// when a walked memory is `derived_from` a commit, the failure revision.
fn resolve_git_frames(
    storage: &Arc<Storage>,
    scope: &str,
    starts: &[String],
    rows: &mut [StartRow],
) -> Result<GitQuery, String> {
    let recorded = file_anchors_in_scope(storage, scope)?;
    let recorded_packages = package_anchors_in_scope(storage, scope)?;
    let mut anchors = Vec::new();
    let mut anchor_seen = HashSet::new();
    let mut files = Vec::new();
    let mut file_seen = HashSet::new();
    let mut line = None;
    for row in rows.iter_mut() {
        let Some(loc) = row.locator.clone() else {
            continue;
        };
        let package = recorded_package(&loc.path, &recorded_packages);
        let matched = match_paths(&loc.path, &recorded);
        if matched.is_empty() && package.is_none() {
            if row.status == "pending" {
                row.status = "unresolved";
                row.reason = Some(file_miss_reason(&loc));
            }
            continue;
        }
        if row.status == "pending" {
            row.status = "walked";
            row.reason = None;
        }
        if line.is_none() {
            line = loc.line;
        }
        if let Some(package) = package
            && anchor_seen.insert(package.clone())
        {
            anchors.push(package);
        }
        for (anchor, path) in matched {
            if file_seen.insert(path.clone()) {
                files.push(path);
            }
            if anchor_seen.insert(anchor.clone()) {
                anchors.push(anchor);
            }
        }
    }

    let mut revision: Option<Revision> = None;
    for id in starts {
        if let Some(rev) = failure_revision(storage, id)? {
            let replace = match &revision {
                None => true,
                Some(have) => rev.time >= have.time,
            };
            if replace {
                revision = Some(rev);
            }
        }
    }
    let (failure_commit_id, failure_time, failure_sha, mut root) = match revision {
        Some(rev) => (Some(rev.id), Some(rev.time), Some(rev.sha), rev.root),
        None => (None, None, None, None),
    };
    if root.is_none() && !files.is_empty() {
        root = git_root_from_touchers(storage, scope, &files)?;
    }
    let mut blame_shas = HashSet::new();
    if let (Some(line), Some(sha), Some(root)) = (line, failure_sha.as_deref(), root.as_ref()) {
        for file in &files {
            if let Ok(Some(blamed)) = crate::tools::repo_ingest::blame_line(root, sha, file, line) {
                blame_shas.insert(blamed);
            }
        }
    }
    if files.is_empty()
        && let Some(failure_id) = failure_commit_id.as_deref()
    {
        for package in packages_on_parent_chain(storage, failure_id, MAX_DEPTH)? {
            if anchor_seen.insert(package.clone()) {
                anchors.push(package);
            }
        }
    }
    let active = !files.is_empty() || failure_sha.is_some() || !anchors.is_empty();
    Ok(GitQuery {
        anchors,
        files,
        line,
        failure_commit_id,
        failure_time,
        blame_shas,
        failure_sha,
        root,
        active,
    })
}

fn recorded_package(path: &str, recorded: &HashSet<String>) -> Option<String> {
    let (ecosystem, name, version) = vestige_core::advanced::git_records::registry_package(path)?;
    let id = vestige_core::advanced::git_records::package_anchor_id(&ecosystem, &name, &version);
    recorded.contains(&id).then_some(id)
}

fn package_anchors_in_scope(
    storage: &Arc<Storage>,
    scope: &str,
) -> Result<HashSet<String>, String> {
    let edges = storage
        .get_all_connections()
        .map_err(|err| err.to_string())?;
    let mut out = HashSet::new();
    for edge in edges {
        if edge.link_type != "touched" && edge.link_type != "supersedes" {
            continue;
        }
        if !edge.target_id.starts_with("pkg:") {
            continue;
        }
        if in_scope(storage, &edge.source_id, scope)? {
            out.insert(edge.target_id);
        }
    }
    Ok(out)
}

/// Package versions touched by `start` or by its recorded parents, within
/// `max_depth`. Used when the failure names no source file, so the lockfile
/// bump is the only recorded path to the changing commit.
fn packages_on_parent_chain(
    storage: &Arc<Storage>,
    start: &str,
    max_depth: u32,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut seen_nodes = HashSet::new();
    let mut seen_pkg = HashSet::new();
    let mut queue = VecDeque::from([(start.to_string(), 0u32)]);
    while let Some((id, depth)) = queue.pop_front() {
        if !seen_nodes.insert(id.clone()) || depth > max_depth {
            continue;
        }
        let edges = storage
            .get_connections_for_memory(&id)
            .map_err(|err| err.to_string())?;
        for edge in edges {
            if edge.link_type == "touched"
                && edge.source_id == id
                && edge.target_id.starts_with("pkg:")
                && seen_pkg.insert(edge.target_id.clone())
            {
                out.push(edge.target_id.clone());
            }
            if edge.link_type == "derived_from" && edge.source_id == id && depth < max_depth {
                queue.push_back((edge.target_id.clone(), depth + 1));
            }
        }
    }
    Ok(out)
}

fn file_miss_reason(loc: &PathLoc) -> String {
    let shown = match loc.line {
        Some(line) => format!("{}:{line}", loc.path),
        None => loc.path.clone(),
    };
    format!(
        "no recorded file anchor matches `{shown}` by exact path (a bare file name does not match). ingest_repo records file:<repo-relative path> from the diff; add node_id to walk a memory instead"
    )
}

fn file_anchors_in_scope(
    storage: &Arc<Storage>,
    scope: &str,
) -> Result<Vec<(String, String)>, String> {
    let edges = storage
        .get_all_connections()
        .map_err(|err| err.to_string())?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for edge in edges {
        if edge.link_type != "touched" {
            continue;
        }
        let Some(path) = edge.target_id.strip_prefix("file:") else {
            continue;
        };
        if path.is_empty() {
            continue;
        }
        if !in_scope(storage, &edge.source_id, scope)? {
            continue;
        }
        if seen.insert(edge.target_id.clone()) {
            out.push((edge.target_id.clone(), path.to_string()));
        }
    }
    Ok(out)
}

fn match_paths(reported: &str, recorded: &[(String, String)]) -> Vec<(String, String)> {
    let reported = normalize_path(reported);
    let exact: Vec<(String, String)> = recorded
        .iter()
        .filter(|(_, path)| path == &reported)
        .cloned()
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    recorded
        .iter()
        .filter(|(_, path)| paths_identify(&reported, path))
        .cloned()
        .collect()
}

/// Exact path, or a shared `/`-bounded suffix of at least two components.
///
/// `.../tokio-1.39.0/src/util/linked_list.rs` matches
/// `tokio/src/util/linked_list.rs` because both end in `src/util/linked_list.rs`.
/// `server.go` does not match `pkg/kubelet/server.go`: one component is a
/// basename, not a directory path.
fn paths_identify(reported: &str, recorded: &str) -> bool {
    if reported == recorded {
        return true;
    }
    let reported: Vec<&str> = reported
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    let recorded: Vec<&str> = recorded
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    let mut common = 0usize;
    for (left, right) in reported.iter().rev().zip(recorded.iter().rev()) {
        if left != right {
            break;
        }
        common += 1;
    }
    common >= 2
}

fn normalize_path(path: &str) -> String {
    path.trim()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

fn split_frame(frame: &str) -> (String, Option<u32>) {
    let frame = frame.trim();
    let Some((rest, last)) = frame.rsplit_once(':') else {
        return (frame.to_string(), None);
    };
    if parse_line(last).is_none() || rest.is_empty() {
        return (frame.to_string(), None);
    }
    if let Some((path, mid)) = rest.rsplit_once(':')
        && parse_line(mid).is_some()
        && !path.is_empty()
    {
        return (path.to_string(), parse_line(mid));
    }
    (rest.to_string(), parse_line(last))
}

fn parse_line(text: &str) -> Option<u32> {
    if text.is_empty() || !text.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    text.parse::<u32>().ok().filter(|line| *line > 0)
}

fn locator_of(point: &StartPoint) -> Option<PathLoc> {
    match point {
        StartPoint::StackFrame { frame, .. } => {
            let (path, line) = split_frame(frame);
            let path = normalize_path(&path);
            if path.is_empty() {
                None
            } else {
                Some(PathLoc { path, line })
            }
        }
        StartPoint::FailingTest { name, .. } => failing_test_path(name).map(|path| PathLoc {
            path: normalize_path(&path),
            line: None,
        }),
        _ => None,
    }
}

fn failing_test_path(name: &str) -> Option<String> {
    let file = name
        .trim()
        .split("::")
        .next()
        .unwrap_or("")
        .split_whitespace()
        .next()
        .unwrap_or("");
    if file.contains('/') || file.contains('\\') || file.contains('.') {
        Some(file.to_string())
    } else {
        None
    }
}

fn full_sha(sha: &str) -> bool {
    sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit())
}

fn commit_sha(node: &vestige_core::KnowledgeNode) -> Option<String> {
    node.tags.iter().find_map(|tag| {
        let sha = tag.strip_prefix("commit:")?;
        full_sha(sha).then(|| sha.to_ascii_lowercase())
    })
}

fn git_root_of(node: &vestige_core::KnowledgeNode) -> Option<PathBuf> {
    node.tags.iter().find_map(|tag| {
        tag.strip_prefix(crate::tools::repo_ingest::GIT_ROOT_PREFIX)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
    })
}

fn failure_revision(storage: &Arc<Storage>, id: &str) -> Result<Option<Revision>, String> {
    let edges = storage
        .get_connections_for_memory(id)
        .map_err(|err| err.to_string())?;
    let mut best: Option<Revision> = None;
    for edge in edges {
        if edge.link_type != "derived_from" || edge.source_id != id {
            continue;
        }
        let Some(node) = storage
            .get_node(&edge.target_id)
            .map_err(|err| err.to_string())?
        else {
            continue;
        };
        let Some(sha) = commit_sha(&node) else {
            continue;
        };
        let Some(time) = node.valid_from else {
            continue;
        };
        let rev = Revision {
            id: node.id.clone(),
            sha,
            time,
            root: git_root_of(&node),
        };
        let replace = match &best {
            None => true,
            Some(have) => rev.time >= have.time,
        };
        if replace {
            best = Some(rev);
        }
    }
    Ok(best)
}

fn git_root_from_touchers(
    storage: &Arc<Storage>,
    scope: &str,
    files: &[String],
) -> Result<Option<PathBuf>, String> {
    let edges = storage
        .get_all_connections()
        .map_err(|err| err.to_string())?;
    for edge in edges {
        if edge.link_type != "touched" {
            continue;
        }
        let Some(path) = edge.target_id.strip_prefix("file:") else {
            continue;
        };
        if !files.iter().any(|file| file == path) {
            continue;
        }
        if !in_scope(storage, &edge.source_id, scope)? {
            continue;
        }
        if let Some(node) = storage
            .get_node(&edge.source_id)
            .map_err(|err| err.to_string())?
            && let Some(root) = git_root_of(&node)
        {
            return Ok(Some(root));
        }
    }
    Ok(None)
}

fn git_admissible(storage: &Arc<Storage>, id: &str, git: &GitQuery) -> bool {
    if id.starts_with("file:")
        || id.starts_with("hunk:")
        || id.starts_with("pkg:")
        || id.starts_with("upstream:")
    {
        return false;
    }
    if git.failure_commit_id.as_deref() == Some(id) {
        return false;
    }
    if let Some(failure_time) = git.failure_time
        && let Ok(Some(node)) = storage.get_node(id)
        && node.valid_from.is_some_and(|valid| valid > failure_time)
    {
        return false;
    }
    true
}

fn reverted_after(storage: &Arc<Storage>, id: &str, failure_time: Option<DateTime<Utc>>) -> bool {
    let Some(failure_time) = failure_time else {
        return false;
    };
    let Ok(edges) = storage.get_connections_for_memory(id) else {
        return false;
    };
    for edge in edges {
        if edge.link_type != "corrects" || edge.target_id != id {
            continue;
        }
        let Ok(Some(revert)) = storage.get_node(&edge.source_id) else {
            continue;
        };
        if revert.valid_from.is_some_and(|valid| valid > failure_time) {
            return true;
        }
    }
    false
}

fn touched_line(storage: &Arc<Storage>, id: &str, files: &[String], line: Option<u32>) -> bool {
    let Some(line) = line else {
        return false;
    };
    let Ok(edges) = storage.get_connections_for_memory(id) else {
        return false;
    };
    for edge in edges {
        if edge.link_type != "touched" || edge.source_id != id {
            continue;
        }
        let Some(rest) = edge.target_id.strip_prefix("hunk:") else {
            continue;
        };
        let Some((file, span)) = rest.rsplit_once(':') else {
            continue;
        };
        if !files.iter().any(|candidate| candidate == file) {
            continue;
        }
        let Some((start, len)) = span.split_once('+') else {
            continue;
        };
        let (Ok(start), Ok(len)) = (start.parse::<u32>(), len.parse::<u32>()) else {
            continue;
        };
        if len > 0 && line >= start && line < start.saturating_add(len) {
            return true;
        }
    }
    false
}

fn edge_rank(path: &[Hop]) -> u8 {
    match path.last().map(|hop| hop.link_type.as_str()) {
        Some("corrects") => 0,
        Some("touched") => 1,
        Some("derived_from") => 2,
        _ => 3,
    }
}

fn git_rank(
    storage: &Arc<Storage>,
    cause: &Reached,
    git: &GitQuery,
) -> (u8, u8, u8, u32, u8, i64, String) {
    let node = storage.get_node(&cause.id).ok().flatten();
    let sha = node.as_ref().and_then(commit_sha);
    let reverted = u8::from(!reverted_after(storage, &cause.id, git.failure_time));
    let blame = u8::from(!sha.as_ref().is_some_and(|sha| git.blame_shas.contains(sha)));
    let hunk = u8::from(!touched_line(storage, &cause.id, &git.files, git.line));
    let gap = match (
        git.failure_time,
        node.as_ref().and_then(|node| node.valid_from),
    ) {
        (Some(failure), Some(commit)) if commit <= failure => {
            (failure - commit).num_milliseconds().max(0)
        }
        _ => i64::MAX,
    };
    (
        reverted,
        blame,
        hunk,
        cause.depth,
        edge_rank(&cause.path),
        gap,
        cause.id.clone(),
    )
}

fn git_structure(
    storage: &Arc<Storage>,
    causes: &[Reached],
    git: &GitQuery,
) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for cause in causes {
        let node = storage.get_node(&cause.id).ok().flatten();
        let sha = node.as_ref().and_then(commit_sha);
        out.insert(
            cause.id.clone(),
            json!({
                "sha": sha,
                "revertedAfterFailure": reverted_after(storage, &cause.id, git.failure_time),
                "blameOfLine": sha.as_ref().is_some_and(|sha| git.blame_shas.contains(sha)),
                "touchedFailingHunk": touched_line(storage, &cause.id, &git.files, git.line),
                "hop": cause.depth,
                "edge": cause.path.last().map(|hop| hop.link_type.as_str()),
                "lockfile": lockfile_of(storage, &cause.id),
            }),
        );
    }
    out
}

fn lockfile_of(storage: &Arc<Storage>, id: &str) -> Option<Value> {
    let edges = storage.get_connections_for_memory(id).ok()?;
    let mut touched = Vec::new();
    let mut replaced = Vec::new();
    for edge in edges {
        if edge.source_id != id {
            continue;
        }
        let Some(package) = parse_package_anchor(&edge.target_id) else {
            continue;
        };
        if edge.link_type == "touched" {
            touched.push(package);
        } else if edge.link_type == "supersedes" {
            replaced.push(package);
        }
    }
    for new in &touched {
        if let Some(old) = replaced.iter().find(|old| {
            old.ecosystem == new.ecosystem && old.name == new.name && old.version != new.version
        }) {
            return Some(json!({
                "ecosystem": new.ecosystem,
                "package": new.name,
                "from": old.version,
                "to": new.version,
            }));
        }
    }
    None
}

struct PackageAnchor {
    ecosystem: String,
    name: String,
    version: String,
}

fn parse_package_anchor(id: &str) -> Option<PackageAnchor> {
    let rest = id.strip_prefix("pkg:")?;
    let (head, version) = rest.rsplit_once('@')?;
    let (ecosystem, name) = head.split_once(':')?;
    if ecosystem.is_empty() || name.is_empty() || version.is_empty() {
        return None;
    }
    Some(PackageAnchor {
        ecosystem: ecosystem.to_string(),
        name: name.to_string(),
        version: version.to_string(),
    })
}

fn upstream_note_for(
    storage: &Arc<Storage>,
    rows: &[StartRow],
    causes: &[Reached],
) -> Option<String> {
    if rows.iter().any(|row| {
        row.locator
            .as_ref()
            .is_some_and(|loc| registry_path_without_local_vcs(&loc.path))
    }) {
        return Some(
            "no local .cargo_vcs_info.json or package.json gitHead for the registry path; the upstream commit range was not recorded and nothing was fetched"
                .to_string(),
        );
    }
    let lockfile_cause = causes.iter().any(|cause| {
        storage
            .get_connections_for_memory(&cause.id)
            .ok()
            .is_some_and(|edges| {
                edges.iter().any(|edge| {
                    edge.source_id == cause.id
                        && edge.link_type == "touched"
                        && edge.target_id.starts_with("pkg:")
                }) && !edges.iter().any(|edge| {
                    edge.source_id == cause.id
                        && edge.link_type == "anchored_to"
                        && edge.target_id.starts_with("upstream:")
                })
            })
    });
    lockfile_cause.then(|| {
        "no local .cargo_vcs_info.json or package.json gitHead for the lockfile bump; the upstream old..new range was not recorded and nothing was fetched"
            .to_string()
    })
}

/// When a recorded `corrects` edge names a commit that is an ancestor of the
/// failure revision, add that commit. The revert itself may sit on another
/// branch (it is not required to be a descendant of the failure). Missing git
/// objects are reported; nothing is fetched. A cause the walk already reached
/// is left as it is.
fn extend_revert_ancestry(
    storage: &Arc<Storage>,
    scope: &str,
    git: &GitQuery,
    nodes: &mut Vec<Reached>,
    causes: &mut Vec<Reached>,
) -> Option<String> {
    let (Some(failure_sha), Some(root), Some(failure_id)) = (
        git.failure_sha.as_deref(),
        git.root.as_ref(),
        git.failure_commit_id.as_deref(),
    ) else {
        return None;
    };
    if !git.active || !root.is_dir() {
        return None;
    }
    let edges = storage.get_all_connections().ok()?;
    let mut missing_objects = false;
    let mut added = Vec::new();
    for edge in edges {
        if edge.link_type != "corrects" || edge.source_id == edge.target_id {
            continue;
        }
        if edge.target_id == failure_id || causes.iter().any(|cause| cause.id == edge.target_id) {
            continue;
        }
        if !in_scope(storage, &edge.source_id, scope).unwrap_or(false)
            || !in_scope(storage, &edge.target_id, scope).unwrap_or(false)
        {
            continue;
        }
        let Ok(Some(revert)) = storage.get_node(&edge.source_id) else {
            continue;
        };
        let Ok(Some(cause)) = storage.get_node(&edge.target_id) else {
            continue;
        };
        let (Some(revert_sha), Some(cause_sha)) = (commit_sha(&revert), commit_sha(&cause)) else {
            continue;
        };
        if revert_sha == failure_sha || cause_sha == failure_sha {
            continue;
        }
        match crate::tools::repo_ingest::git_is_ancestor(root, &cause_sha, failure_sha) {
            Ok(true) => added.push((edge.target_id, edge.source_id)),
            Err(_) => missing_objects = true,
            Ok(false) => {}
        }
    }
    let from = nodes
        .iter()
        .find(|node| node.id == failure_id)
        .map(|node| node.from.clone())
        .unwrap_or_default();
    let depth = nodes
        .iter()
        .find(|node| node.id == failure_id)
        .map(|node| node.depth.saturating_add(1))
        .unwrap_or(1);
    for (cause_id, revert_id) in added {
        if nodes.iter().any(|node| node.id == cause_id) {
            continue;
        }
        let start = from
            .first()
            .cloned()
            .unwrap_or_else(|| failure_id.to_string());
        let reached = Reached {
            id: cause_id.clone(),
            depth,
            path: vec![Hop {
                source_id: revert_id,
                target_id: cause_id.clone(),
                link_type: "corrects".into(),
            }],
            start: start.clone(),
            from: from.clone(),
        };
        nodes.push(Reached {
            id: reached.id.clone(),
            depth: reached.depth,
            path: reached.path.clone(),
            start,
            from: reached.from.clone(),
        });
        causes.push(reached);
    }
    missing_objects.then(|| {
        "a revert's ancestry could not be checked because a commit object is not in the local repository; nothing was fetched".to_string()
    })
}

fn apply_version_range(
    storage: &Arc<Storage>,
    points: Option<&[StartPoint]>,
    nodes: &mut Vec<Reached>,
    causes: &mut Vec<Reached>,
) -> Option<Value> {
    let points = points?;
    let (worked_in, broke_in, repo) = points.iter().find_map(|point| match point {
        StartPoint::VersionRange {
            worked_in,
            broke_in,
            repo,
            ..
        } => Some((worked_in.as_str(), broke_in.as_str(), repo.as_str())),
        _ => None,
    })?;
    let window = match crate::tools::repo_ingest::first_parent_rev_list(
        std::path::Path::new(repo),
        worked_in,
        broke_in,
    ) {
        Ok(window) => window,
        Err(err) => {
            return Some(json!({
                "workedIn": worked_in,
                "brokeIn": broke_in,
                "repo": repo,
                "good": Value::Null,
                "bad": Value::Null,
                "excluded": 0,
                "excludedShas": [],
                "because": err,
            }));
        }
    };
    let mut excluded = Vec::new();
    causes.retain(|cause| match node_commit_sha(storage, &cause.id) {
        Some(sha) if !window.shas.contains(&sha) => {
            excluded.push(sha);
            false
        }
        _ => true,
    });
    let dropped: HashSet<&str> = excluded.iter().map(String::as_str).collect();
    nodes.retain(|node| match node_commit_sha(storage, &node.id) {
        Some(sha) => !dropped.contains(sha.as_str()),
        None => true,
    });
    let because = if excluded.is_empty() {
        format!(
            "no commit candidate was outside git rev-list --first-parent {}..{}",
            window.good, window.bad
        )
    } else {
        format!(
            "{} commit(s) excluded: outside git rev-list --first-parent {}..{}",
            excluded.len(),
            window.good,
            window.bad
        )
    };
    Some(json!({
        "workedIn": worked_in,
        "brokeIn": broke_in,
        "repo": repo,
        "good": window.good,
        "bad": window.bad,
        "excluded": excluded.len(),
        "excludedShas": excluded,
        "because": because,
    }))
}

fn node_commit_sha(storage: &Arc<Storage>, id: &str) -> Option<String> {
    storage
        .get_node(id)
        .ok()
        .flatten()
        .as_ref()
        .and_then(commit_sha)
}

fn registry_path_without_local_vcs(path: &str) -> bool {
    if vestige_core::advanced::git_records::registry_package(path).is_none() {
        return false;
    }
    local_crate_sha(path).is_none()
}

/// sha1 or gitHead from a crate directory that is already on disk.
/// Walks up from the frame's file. Does not search the network.
fn local_crate_sha(path: &str) -> Option<String> {
    let mut dir = std::path::Path::new(path).parent()?.to_path_buf();
    for _ in 0..8 {
        let vcs = dir.join(".cargo_vcs_info.json");
        if let Some(sha) = std::fs::read_to_string(&vcs)
            .ok()
            .as_deref()
            .and_then(|text| json_object_sha(text, "sha1"))
        {
            return Some(sha);
        }
        let package = dir.join("package.json");
        if let Some(sha) = std::fs::read_to_string(&package)
            .ok()
            .as_deref()
            .and_then(|text| json_object_sha(text, "gitHead"))
        {
            return Some(sha);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn json_object_sha(text: &str, key: &str) -> Option<String> {
    let value: Value = serde_json::from_str(text).ok()?;
    let sha = value.get(key)?.as_str()?.to_ascii_lowercase();
    (sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit())).then_some(sha)
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
            parents: vec![],
            reverts: None,
            cherry_picked_from: None,
            fixes: vec![],
            lock_bumps: vec![],
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
        // A lineage chain: each record is evidence of the one before it. (A
        // chain of `touched` edges is followed for one edge only.)
        for hop in 0..chain.len() - 1 {
            link(&storage, &chain[hop + 1], &chain[hop], "evidence_of");
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

        // 1. a bare frame with no recorded file anchor is refused, and the
        //    refusal names the path rather than a missing memory id.
        let bare = json!({"kind": "stack_frame", "frame": "src/auth.rs:10"});
        let refused = execute(&storage, Some(json!({"start_points": [bare.clone()]})))
            .await
            .unwrap();
        assert_eq!(refused["needs_report"]["missing"], json!(["file"]));
        let detail = refused["needs_report"]["detail"].as_str().unwrap();
        assert!(detail.contains("file anchor"), "{detail}");
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

    fn put_tagged(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> String {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    tags: tags.iter().map(|tag| tag.to_string()).collect(),
                    ..Default::default()
                },
                "user",
            )
            .unwrap()
            .id
    }

    #[tokio::test]
    async fn candidates_are_ordered_by_depth_then_shared_identities_then_rarity_then_id() {
        let (storage, _dir) = open();
        // Written in this order, so id order alone would list the depth-1
        // candidates one_common, two, one_rare, declared.
        let one_common = put_tagged(&storage, "commit a. Touched: lib/other.py", &["connection"]);
        let two = put_tagged(&storage, "commit b. Touched: lib/conn.py", &["connection"]);
        let one_rare = put_tagged(&storage, "commit c", &["retry"]);
        let declared = put_tagged(&storage, "decision d", &[]);
        let deeper = put_tagged(
            &storage,
            "commit e. Touched: lib/conn.py",
            &["connection", "retry"],
        );
        // Two more carriers of `connection`, linked to nothing: they only
        // make that tag less rare than `retry`.
        put_tagged(&storage, "bystander 1", &["connection"]);
        put_tagged(&storage, "bystander 2", &["connection"]);
        let start = put_tagged(
            &storage,
            "failure: connection retry storm, trace ends at lib/conn.py:88",
            &["connection", "retry"],
        );
        for cause in [&one_common, &two, &one_rare] {
            link(&storage, cause, &start, "touched");
        }
        link(&storage, &start, &declared, "derived_from");
        // Lineage behind a lead is followed: `one_common` declares what it
        // came from.
        link(&storage, &one_common, &deeper, "derived_from");

        let args = json!({"node_id": start});
        let out = execute(&storage, Some(args.clone())).await.unwrap();
        assert_eq!(
            causes_of(&out),
            vec![
                // Depth 1: two shared identities; then one shared, the rarer
                // tag first; then the declared link, which shares nothing.
                (two.clone(), 1),
                (one_rare.clone(), 1),
                (one_common.clone(), 1),
                (declared.clone(), 1),
                // Depth 2 comes after every depth-1 candidate, although it
                // shares the most with the start.
                (deeper.clone(), 2),
            ],
            "{out}"
        );
        assert_eq!(execute(&storage, Some(args)).await.unwrap(), out);

        let causes = out["causes"].as_array().unwrap();
        // Every candidate carries the exact identities it shares with the
        // start, each with its carriers in the scope of 8.
        assert_eq!(out["ranking"]["scope_size"], 8, "{out}");
        assert_eq!(causes[0]["rank"], 1);
        assert_eq!(causes[0]["start"], start);
        assert_eq!(causes[0]["shared_count"], 2);
        assert_eq!(
            causes[0]["joined_on"],
            json!([
                {"identity": "tag:connection", "carriers": 6},
                {"identity": "path:lib/conn.py", "carriers": 3},
            ])
        );
        assert_eq!(
            causes[1]["joined_on"],
            json!([{"identity": "tag:retry", "carriers": 3}])
        );
        assert_eq!(
            causes[2]["joined_on"],
            json!([{"identity": "tag:connection", "carriers": 6}])
        );
        // The words `connection` and `retry` in the start's text join
        // nothing: the declared decision shares no identity.
        assert_eq!(causes[3]["joined_on"], json!([]));
        assert_eq!(causes[3]["shared_count"], 0);
        assert_eq!(causes[4]["shared_count"], 3);
        assert_eq!(causes[4]["rank"], 5);
        assert_eq!(causes[4]["not_counted_hub_tags"], json!([]));
        assert!(
            out["ranking"]["order"]
                .as_str()
                .unwrap()
                .starts_with("depth, then distinct exact identities"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn a_hub_tag_is_listed_but_not_counted() {
        let (storage, _dir) = open();
        let by_hub_only = put_tagged(&storage, "commit a", &["campaign"]);
        let by_path = put_tagged(&storage, "commit b. Touched: src/gate.rs", &[]);
        for i in 0..13 {
            put_tagged(&storage, &format!("batch {i}"), &["campaign"]);
        }
        for i in 0..4 {
            put_tagged(&storage, &format!("other {i}"), &[]);
        }
        let start = put_tagged(&storage, "failure at src/gate.rs:9", &["campaign"]);
        link(&storage, &by_hub_only, &start, "touched");
        link(&storage, &by_path, &start, "touched");

        let out = execute(&storage, Some(json!({"node_id": start})))
            .await
            .unwrap();
        // 15 of the 20 memories carry `campaign`: a hub. The candidate that
        // shares only the hub tag ranks after the one sharing a path, though
        // its id is smaller.
        assert_eq!(
            causes_of(&out),
            vec![(by_path.clone(), 1), (by_hub_only.clone(), 1)],
            "{out}"
        );
        assert_eq!(out["ranking"]["scope_size"], 20);
        assert_eq!(
            out["causes"][0]["joined_on"],
            json!([{"identity": "path:src/gate.rs", "carriers": 2}])
        );
        assert_eq!(out["causes"][1]["joined_on"], json!([]));
        assert_eq!(out["causes"][1]["shared_count"], 0);
        assert_eq!(
            out["causes"][1]["not_counted_hub_tags"],
            json!([{"identity": "tag:campaign", "carriers": 15}])
        );
    }

    /// The rule that keeps a walk a narrowing: lineage edges chain, a
    /// `touched` edge does not. Two memories naming the same thing is a
    /// lead; the lead of a lead shares nothing with the start.
    #[tokio::test]
    async fn a_path_follows_at_most_one_touched_edge() {
        let (storage, _dir) = open();
        // The shape of a commit window: the failure shares `connection`
        // with two commits, and each of those shares `tests` with commits
        // the failure shares nothing with.
        let far_a = put_tagged(&storage, "commit far a. Touched: tests/a.py", &["tests"]);
        let far_b = put_tagged(&storage, "commit far b. Touched: tests/b.py", &["tests"]);
        let near_a = put_tagged(
            &storage,
            "commit near a. Touched: lib/conn.py tests/a.py",
            &["connection", "tests"],
        );
        let near_b = put_tagged(&storage, "commit near b", &["connection", "tests"]);
        let origin = put_tagged(&storage, "decision the near commit came from", &[]);
        let behind_origin = put_tagged(&storage, "commit behind the decision", &["tests"]);
        let start = put_tagged(&storage, "failure: connection storm", &["connection"]);
        link(&storage, &near_a, &start, "touched");
        link(&storage, &near_b, &start, "touched");
        // What `vestige connect` writes between the commits themselves.
        link(&storage, &far_a, &near_a, "touched");
        link(&storage, &far_b, &near_a, "touched");
        link(&storage, &far_a, &near_b, "touched");
        link(&storage, &far_a, &far_b, "touched");
        // Lineage behind a lead is still followed, but the path has used
        // its touched edge: nothing is reached over another one.
        link(&storage, &near_b, &origin, "derived_from");
        link(&storage, &behind_origin, &origin, "touched");

        let args = json!({"node_id": start});
        let out = execute(&storage, Some(args.clone())).await.unwrap();
        assert_eq!(
            causes_of(&out),
            vec![
                (near_a.clone(), 1),
                (near_b.clone(), 1),
                (origin.clone(), 2)
            ],
            "{out}"
        );
        assert_eq!(execute(&storage, Some(args)).await.unwrap(), out);
        for hidden in [&far_a, &far_b, &behind_origin] {
            assert!(!node_ids(&out).contains(hidden), "{hidden} in {out}");
        }
        assert_eq!(out["truncated"], false, "nothing was cut by a bound: {out}");

        // What was not followed is counted, with what the ends of those
        // edges share. `far_a -> far_b` is never seen: neither end was
        // reached.
        let held = &out["not_followed"];
        assert_eq!(held["touched_edges"], 4, "{out}");
        assert_eq!(held["memories"], 3, "{out}");
        assert_eq!(
            held["shared"],
            json!([
                {"identity": "tag:tests", "carriers": 5, "edges": 3},
                {"identity": "path:tests/a.py", "carriers": 2, "edges": 1},
            ]),
            "{out}"
        );
        assert_eq!(held["shared_identities"], 2);
        assert_eq!(held["scope_size"], 7);
        // `behind_origin -> origin`: the decision records no identity.
        assert_eq!(held["no_counted_identity"], 1);
        assert!(
            held["reason"]
                .as_str()
                .unwrap()
                .contains("at most one touched edge"),
            "{out}"
        );
    }

    /// One `touched` edge per PATH, not per walk: after lineage edges a
    /// path may still follow one, and a memory reached both over a touched
    /// edge and over lineage is expanded from the lineage side too.
    #[tokio::test]
    async fn a_touched_edge_is_followed_after_lineage_and_from_either_route() {
        let (storage, _dir) = open();
        let start = put(&storage, "user", "failure");
        let decision = put(&storage, "user", "decision the failure derives from");
        let both = put(&storage, "user", "record reached two ways");
        let beyond = put(&storage, "user", "commit that touched the record");
        let lead = put(&storage, "user", "commit that touched the decision");
        // start -> decision -> both is lineage; both -> start is also one
        // touched edge away.
        link(&storage, &start, &decision, "derived_from");
        link(&storage, &decision, &both, "derived_from");
        link(&storage, &both, &start, "touched");
        link(&storage, &lead, &decision, "touched");
        link(&storage, &beyond, &both, "touched");

        let out = execute(&storage, Some(json!({"node_id": start})))
            .await
            .unwrap();
        assert_eq!(
            causes_of(&out),
            vec![
                (decision.clone(), 1),
                // Listed at the depth of its shortest path, the touched edge.
                (both.clone(), 1),
                (lead.clone(), 2),
                // start -> decision -> both -> beyond: two lineage edges,
                // then the path's one touched edge.
                (beyond.clone(), 3),
            ],
            "{out}"
        );
        let beyond_row = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == beyond)
            .unwrap();
        let kinds: Vec<&str> = beyond_row["path"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hop| hop["link_type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["derived_from", "derived_from", "touched"]);
        assert!(
            out["not_followed"].is_null(),
            "nothing is hidden behind a touched edge: {out}"
        );
    }

    /// A walk stopped by the depth bound says so only for edges it could
    /// still follow: a `touched` edge behind a path that has used its own
    /// is not "more beyond", it is not followed and counted as such.
    #[tokio::test]
    async fn the_depth_bound_does_not_count_touched_edges_it_would_not_follow() {
        let (storage, _dir) = open();
        let start = put(&storage, "user", "failure");
        let lead = put(&storage, "user", "lead");
        link(&storage, &lead, &start, "touched");
        let mut last = lead.clone();
        for hop in 1..MAX_DEPTH {
            let next = put(&storage, "user", &format!("origin {hop}"));
            link(&storage, &last, &next, "derived_from");
            last = next;
        }
        let behind = put(&storage, "user", "behind the bound, over touched");
        link(&storage, &behind, &last, "touched");

        let out = execute(&storage, Some(json!({"node_id": start})))
            .await
            .unwrap();
        assert!(node_ids(&out).contains(&last), "{out}");
        assert!(!node_ids(&out).contains(&behind), "{out}");
        assert_eq!(out["truncated"], false, "{out}");

        // The same memory behind a lineage edge IS more beyond the bound.
        let further = put(&storage, "user", "behind the bound, over lineage");
        link(&storage, &last, &further, "derived_from");
        let out = execute(&storage, Some(json!({"node_id": start})))
            .await
            .unwrap();
        assert!(!node_ids(&out).contains(&further), "{out}");
        assert_eq!(out["truncated"], true, "{out}");
    }

    #[test]
    fn path_identity_rejects_a_bare_basename() {
        assert!(super::paths_identify(
            "src/util/linked_list.rs",
            "tokio/src/util/linked_list.rs"
        ));
        assert!(super::paths_identify(
            "/checkout/tokio/src/util/linked_list.rs",
            "tokio/src/util/linked_list.rs"
        ));
        assert!(!super::paths_identify(
            "linked_list.rs",
            "tokio/src/util/linked_list.rs"
        ));
        assert!(!super::paths_identify("server.go", "pkg/kubelet/server.go"));
        assert!(super::paths_identify("src/a.rs", "src/a.rs"));
        assert!(super::paths_identify(
            "/home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs",
            "tokio/src/util/linked_list.rs"
        ));
        let (path, line) = super::split_frame("tokio/src/util/linked_list.rs:123:9");
        assert_eq!(path, "tokio/src/util/linked_list.rs");
        assert_eq!(line, Some(123));
        assert!(super::failing_test_path("test_login").is_none());
        assert_eq!(
            super::failing_test_path("src/a.rs::test_login").as_deref(),
            Some("src/a.rs")
        );
    }

    #[tokio::test]
    async fn stack_frame_ranks_the_reverted_commit_above_the_blamed_line() {
        let dir = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str], date: &str| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args([
                    "-c",
                    "user.email=t@example.com",
                    "-c",
                    "user.name=t",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                ])
                .args(args)
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_SYSTEM", "/dev/null")
                .env("GIT_AUTHOR_DATE", date)
                .env("GIT_COMMITTER_DATE", date)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        git(&["init", "-q"], "2024-01-01T00:00:00Z");
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/a.rs"),
            "fn a() {\n    let introduced = 1;\n}\n",
        )
        .unwrap();
        git(&["add", "-A"], "2024-01-01T00:00:00Z");
        git(
            &["commit", "-q", "-m", "introduce the line"],
            "2024-01-01T00:00:00Z",
        );
        let introduced = git(&["rev-parse", "HEAD"], "2024-01-01T00:00:00Z");

        std::fs::write(
            dir.path().join("src/a.rs"),
            "fn a() {\n    let introduced = 1;\n    let touched = 2;\n}\n",
        )
        .unwrap();
        git(&["add", "-A"], "2024-02-01T00:00:00Z");
        git(
            &["commit", "-q", "-m", "touch the file"],
            "2024-02-01T00:00:00Z",
        );
        let cause = git(&["rev-parse", "HEAD"], "2024-02-01T00:00:00Z");

        std::fs::write(dir.path().join("README.md"), "observed\n").unwrap();
        git(&["add", "-A"], "2024-03-01T00:00:00Z");
        git(
            &["commit", "-q", "-m", "failure revision"],
            "2024-03-01T00:00:00Z",
        );
        let failure_rev = git(&["rev-parse", "HEAD"], "2024-03-01T00:00:00Z");
        git(&["revert", "--no-edit", &cause], "2024-04-01T00:00:00Z");

        let (storage, _store) = open();
        let ingested = crate::tools::repo_ingest::execute(
            &storage,
            crate::tools::repo_ingest::Request {
                repo_path: dir.path().to_path_buf(),
                codebase: Some("demo".into()),
                scope: Some("demo".into()),
                rev: None,
                since: None,
                until: None,
                limit: Some(20),
                dry_run: false,
                budget: None,
            },
        )
        .await
        .unwrap();
        assert!(
            ingested["edges"]["reverts"].as_u64().unwrap() >= 1,
            "{ingested}"
        );

        let commit_id = |sha: &str| {
            let tag = crate::tools::repo_ingest::commit_tag(sha);
            storage
                .current_code_context_nodes("event", Some(&tag), "demo", 5)
                .unwrap()
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("missing {sha}"))
                .id
        };
        let failure = put(&storage, "demo", "failure: panic at src/a.rs:2");
        link(&storage, &failure, &commit_id(&failure_rev), "derived_from");

        let out = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{
                    "kind": "stack_frame",
                    "frame": "src/a.rs:2",
                    "node_id": failure
                }]
            })),
        )
        .await
        .unwrap();
        let top = &out["causes"][0];
        assert_eq!(top["structure"]["sha"], json!(cause), "{out}");
        assert_eq!(top["structure"]["revertedAfterFailure"], true, "{out}");
        assert_eq!(
            top["structure"]["blameOfLine"], false,
            "line 2 was introduced by an earlier commit: {out}"
        );
        let shas: Vec<String> = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|cause| cause["structure"]["sha"].as_str().map(str::to_string))
            .collect();
        let cause_at = shas.iter().position(|sha| sha == &cause).unwrap();
        let introduced_at = shas
            .iter()
            .position(|sha| sha == &introduced)
            .unwrap_or_else(|| panic!("blamed commit missing: {out}"));
        assert!(cause_at < introduced_at, "{out}");

        let bare = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{"kind": "stack_frame", "frame": "a.rs:2"}]
            })),
        )
        .await
        .unwrap();
        assert_eq!(bare["needs_report"]["missing"], json!(["file"]), "{bare}");

        let by_test = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{
                    "kind": "failing_test",
                    "name": "src/a.rs",
                    "node_id": failure
                }]
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            by_test["causes"][0]["structure"]["sha"],
            json!(cause),
            "{by_test}"
        );

        let frame = format!("{}:2", dir.path().join("src/a.rs").display());
        let suffixed = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{
                    "kind": "stack_frame",
                    "frame": frame,
                    "node_id": failure
                }]
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            suffixed["causes"][0]["structure"]["sha"],
            json!(cause),
            "{suffixed}"
        );
    }

    #[tokio::test]
    async fn a_lockfile_bump_ranks_through_the_package_anchor() {
        let (storage, _dir) = open();
        let scope = "user";
        let commit = |sha: &str, days: i64| {
            storage
                .ingest_in_scope(
                    IngestInput {
                        content: format!("commit {sha}"),
                        node_type: "event".into(),
                        tags: vec![format!("commit:{sha}"), "git-commit".into()],
                        valid_from: Some(Utc::now() - chrono::Duration::days(days)),
                        ..Default::default()
                    },
                    scope,
                )
                .unwrap()
                .id
        };
        let observed_sha = "a".repeat(40);
        let bump_sha = "b".repeat(40);
        let other_sha = "c".repeat(40);
        let observed = commit(&observed_sha, 1);
        let bump = commit(&bump_sha, 2);
        let other = commit(&other_sha, 8);
        let failure = put(&storage, scope, "failure: error decoding response body");
        link(&storage, &failure, &observed, "derived_from");
        link(&storage, &observed, &bump, "derived_from");
        link(&storage, &observed, &other, "derived_from");
        link(&storage, &bump, "pkg:cargo:reqwest@0.12.9", "touched");
        link(&storage, &bump, "pkg:cargo:reqwest@0.12.8", "supersedes");
        link(&storage, &other, "pkg:cargo:serde@1.0.1", "touched");
        link(&storage, &other, "pkg:cargo:serde@1.0.0", "supersedes");

        let frame = "/home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/reqwest-0.12.9/src/lib.rs:10";
        let out = execute(
            &storage,
            Some(json!({
                "scope": scope,
                "start_points": [{
                    "kind": "stack_frame",
                    "frame": frame,
                    "node_id": failure
                }]
            })),
        )
        .await
        .unwrap();
        let top = &out["causes"][0];
        assert_eq!(top["id"], bump, "{out}");
        assert_eq!(top["structure"]["edge"], "touched", "{out}");
        assert_eq!(top["structure"]["lockfile"]["package"], "reqwest", "{out}");
        assert_eq!(top["structure"]["lockfile"]["from"], "0.12.8", "{out}");
        assert_eq!(top["structure"]["lockfile"]["to"], "0.12.9", "{out}");
        assert_ne!(top["id"], other, "an older lockfile bump is not the cause");
        let note = out["upstreamSkipped"].as_str().unwrap_or("");
        assert!(
            note.contains("nothing was fetched"),
            "missing local upstream metadata must be reported: {out}"
        );

        let ci = execute(
            &storage,
            Some(json!({
                "scope": scope,
                "start_points": [{
                    "kind": "ci_run",
                    "run_id": "https://example.test/actions/runs/1#step:8:12",
                    "node_id": failure
                }]
            })),
        )
        .await
        .unwrap();
        assert_eq!(ci["causes"][0]["id"], bump, "{ci}");
        assert_eq!(ci["causes"][0]["structure"]["edge"], "touched", "{ci}");
        assert_eq!(
            ci["causes"][0]["structure"]["lockfile"]["package"], "reqwest",
            "{ci}"
        );
    }

    #[tokio::test]
    async fn a_missing_version_range_does_not_drop_a_non_commit_cause() {
        let (storage, _dir) = open();
        let effect = put(&storage, "user", "login handler failed");
        let cause = put(&storage, "user", "commit flipped the auth timeout");
        link(&storage, &effect, &cause, "derived_from");
        let out = execute(
            &storage,
            Some(json!({
                "start_points": [
                    {"kind": "logged_write", "node_id": effect},
                    {"kind": "version_range", "worked_in": "no-such", "broke_in": "also-missing", "repo": "/no/such/repo"}
                ]
            })),
        )
        .await
        .unwrap();
        assert!(
            out["causes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == cause),
            "{out}"
        );
        assert_eq!(out["range"]["excluded"], 0, "{out}");
        let because = out["range"]["because"].as_str().unwrap();
        assert!(
            because.contains("not an available directory") || because.contains("could not resolve"),
            "{because}"
        );
    }

    #[tokio::test]
    async fn version_range_excludes_commits_outside_first_parent() {
        let repo = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str], date: &str| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(repo.path())
                .args([
                    "-c",
                    "user.email=t@example.com",
                    "-c",
                    "user.name=t",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                ])
                .args(args)
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_SYSTEM", "/dev/null")
                .env("GIT_AUTHOR_DATE", date)
                .env("GIT_COMMITTER_DATE", date)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        git(&["init", "-q"], "2020-01-01T00:00:00Z");
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/a.rs"), "fn a() { let v = 1; }\n").unwrap();
        git(&["add", "-A"], "2020-01-01T00:00:00Z");
        git(&["commit", "-q", "-m", "old line"], "2020-01-01T00:00:00Z");
        let old = git(&["rev-parse", "HEAD"], "2020-01-01T00:00:00Z");
        git(&["tag", "good"], "2020-01-01T00:00:00Z");
        std::fs::write(repo.path().join("src/a.rs"), "fn a() { let v = 2; }\n").unwrap();
        git(&["add", "-A"], "2024-06-01T00:00:00Z");
        git(&["commit", "-q", "-m", "the bump"], "2024-06-01T00:00:00Z");
        let mid = git(&["rev-parse", "HEAD"], "2024-06-01T00:00:00Z");
        std::fs::write(repo.path().join("src/a.rs"), "fn a() { let v = 3; }\n").unwrap();
        git(&["add", "-A"], "2024-07-01T00:00:00Z");
        git(
            &["commit", "-q", "-m", "failure revision"],
            "2024-07-01T00:00:00Z",
        );
        let bad = git(&["rev-parse", "HEAD"], "2024-07-01T00:00:00Z");
        git(&["tag", "bad"], "2024-07-01T00:00:00Z");

        let (storage, _log) = open();
        let scope = "user";
        let record = |sha: &str, when: &str| {
            storage
                .ingest_in_scope(
                    IngestInput {
                        content: format!("commit {sha}"),
                        node_type: "event".into(),
                        tags: vec![
                            format!("commit:{sha}"),
                            "git-commit".into(),
                            format!(
                                "{}{}",
                                crate::tools::repo_ingest::GIT_ROOT_PREFIX,
                                repo.path().display()
                            ),
                        ],
                        valid_from: Some(when.parse().unwrap()),
                        ..Default::default()
                    },
                    scope,
                )
                .unwrap()
                .id
        };
        let old_id = record(&old, "2020-01-01T00:00:00Z");
        let mid_id = record(&mid, "2024-06-01T00:00:00Z");
        let bad_id = record(&bad, "2024-07-01T00:00:00Z");
        for id in [&old_id, &mid_id, &bad_id] {
            link(&storage, id, "file:src/a.rs", "touched");
        }
        let failure = put(&storage, scope, "failure at src/a.rs:1");
        link(&storage, &failure, &bad_id, "derived_from");

        let out = execute(
            &storage,
            Some(json!({
                "scope": scope,
                "start_points": [
                    {"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": failure},
                    {
                        "kind": "version_range",
                        "worked_in": "good",
                        "broke_in": "bad",
                        "repo": repo.path().display().to_string()
                    }
                ]
            })),
        )
        .await
        .unwrap();
        let shas: Vec<&str> = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|row| row["structure"]["sha"].as_str())
            .collect();
        assert!(shas.contains(&mid.as_str()), "{out}");
        assert!(!shas.contains(&old.as_str()), "{out}");
        assert!(out["range"]["excluded"].as_u64().unwrap() >= 1, "{out}");
        let excluded = out["range"]["excludedShas"].as_array().unwrap();
        assert!(
            excluded
                .iter()
                .any(|sha| sha.as_str() == Some(old.as_str())),
            "{out}"
        );
        assert!(
            out["range"]["because"]
                .as_str()
                .unwrap()
                .contains("rev-list --first-parent"),
            "{out}"
        );
    }

    fn git_at(root: &std::path::Path, args: &[&str], date: &str) -> String {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args([
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
            ])
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    async fn ingest_repo(
        storage: &Arc<Storage>,
        root: &std::path::Path,
        rev: &str,
        limit: usize,
    ) -> Value {
        crate::tools::repo_ingest::execute(
            storage,
            crate::tools::repo_ingest::Request {
                repo_path: root.to_path_buf(),
                codebase: Some("demo".into()),
                scope: Some("demo".into()),
                rev: Some(rev.to_string()),
                since: None,
                until: None,
                limit: Some(limit),
                dry_run: false,
                budget: None,
            },
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn a_revert_child_reaches_the_corrected_commit_past_the_parent_gap() {
        let repo = tempfile::TempDir::new().unwrap();
        let root = repo.path();
        let day = |n: u32| format!("2024-01-{n:02}T00:00:00Z");
        git_at(root, &["init", "-q"], &day(1));
        std::fs::write(root.join("cause.rs"), "fn cause() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(1));
        git_at(root, &["commit", "-q", "-m", "cause"], &day(1));
        let cause = git_at(root, &["rev-parse", "HEAD"], &day(1));
        for n in 2..=10 {
            std::fs::write(root.join(format!("f{n}.rs")), format!("fn f{n}() {{}}\n")).unwrap();
            git_at(root, &["add", "-A"], &day(n));
            git_at(
                root,
                &["commit", "-q", "-m", &format!("filler {n}")],
                &day(n),
            );
        }
        let failure_rev = git_at(root, &["rev-parse", "HEAD"], &day(10));
        git_at(root, &["revert", "--no-edit", &cause], &day(11));

        let (storage, _log) = open();
        let ingested = crate::tools::repo_ingest::execute(
            &storage,
            crate::tools::repo_ingest::Request {
                repo_path: root.to_path_buf(),
                codebase: Some("demo".into()),
                scope: Some("demo".into()),
                rev: None,
                since: None,
                until: None,
                limit: Some(30),
                dry_run: false,
                budget: None,
            },
        )
        .await
        .unwrap();
        assert!(
            ingested["edges"]["reverts"].as_u64().unwrap() >= 1,
            "{ingested}"
        );
        let cause_id = {
            let tag = crate::tools::repo_ingest::commit_tag(&cause);
            storage
                .current_code_context_nodes("event", Some(&tag), "demo", 5)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
                .id
        };
        let failure_id = {
            let tag = crate::tools::repo_ingest::commit_tag(&failure_rev);
            storage
                .current_code_context_nodes("event", Some(&tag), "demo", 5)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
                .id
        };
        let failure = put(&storage, "demo", "failure observed with no path");
        link(&storage, &failure, &failure_id, "derived_from");
        let out = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{"kind": "logged_write", "node_id": failure}]
            })),
        )
        .await
        .unwrap();
        let top = &out["causes"][0];
        assert_eq!(top["structure"]["sha"], json!(cause), "{out}");
        assert_eq!(top["id"], cause_id, "{out}");
        assert_eq!(top["structure"]["edge"], "corrects", "{out}");
        assert_eq!(top["structure"]["revertedAfterFailure"], true, "{out}");
    }

    #[tokio::test]
    async fn revert_ancestry_reaches_a_cause_whose_parent_was_not_ingested() {
        let repo = tempfile::TempDir::new().unwrap();
        let root = repo.path();
        let day = |n: u32| format!("2024-02-{n:02}T00:00:00Z");
        git_at(root, &["init", "-q"], &day(1));
        std::fs::write(root.join("cause.rs"), "fn cause() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(1));
        git_at(root, &["commit", "-q", "-m", "cause"], &day(1));
        let cause = git_at(root, &["rev-parse", "HEAD"], &day(1));
        std::fs::write(root.join("between.rs"), "fn between() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(2));
        git_at(root, &["commit", "-q", "-m", "between"], &day(2));
        std::fs::write(root.join("README.md"), "observed\n").unwrap();
        git_at(root, &["add", "-A"], &day(3));
        git_at(root, &["commit", "-q", "-m", "failure revision"], &day(3));
        let failure_rev = git_at(root, &["rev-parse", "HEAD"], &day(3));
        std::fs::write(root.join("b.rs"), "fn b() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(4));
        git_at(root, &["commit", "-q", "-m", "later"], &day(4));
        std::fs::write(root.join("c.rs"), "fn c() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(5));
        git_at(
            root,
            &["commit", "-q", "-m", "parent of the revert"],
            &day(5),
        );
        git_at(root, &["revert", "--no-edit", &cause], &day(6));
        let revert = git_at(root, &["rev-parse", "HEAD"], &day(6));

        let (storage, _log) = open();
        ingest_repo(&storage, root, &cause, 1).await;
        ingest_repo(&storage, root, &failure_rev, 1).await;
        let pulled = ingest_repo(&storage, root, &revert, 1).await;
        assert!(
            pulled["edges"]["reverts"].as_u64().unwrap() >= 1,
            "{pulled}"
        );

        let id_of = |sha: &str| {
            let tag = crate::tools::repo_ingest::commit_tag(sha);
            storage
                .current_code_context_nodes("event", Some(&tag), "demo", 5)
                .unwrap()
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("missing {sha}"))
                .id
        };
        let failure = put(&storage, "demo", "failure observed with no path");
        link(&storage, &failure, &id_of(&failure_rev), "derived_from");
        let out = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{"kind": "logged_write", "node_id": failure}]
            })),
        )
        .await
        .unwrap();
        let shas: Vec<&str> = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|row| row["structure"]["sha"].as_str())
            .collect();
        assert!(shas.contains(&cause.as_str()), "{out}");
        let cause_row = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["structure"]["sha"] == cause)
            .unwrap();
        assert_eq!(cause_row["structure"]["edge"], "corrects", "{out}");
    }

    #[tokio::test]
    async fn a_revert_on_another_branch_still_names_an_ancestor_of_the_failure() {
        let repo = tempfile::TempDir::new().unwrap();
        let root = repo.path();
        let day = |n: u32| format!("2024-03-{n:02}T00:00:00Z");
        git_at(root, &["init", "-q"], &day(1));
        std::fs::write(root.join("cause.rs"), "fn cause() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(1));
        git_at(root, &["commit", "-q", "-m", "cause"], &day(1));
        let cause = git_at(root, &["rev-parse", "HEAD"], &day(1));
        let main = git_at(root, &["branch", "--show-current"], &day(1));
        std::fs::write(root.join("mid.rs"), "fn mid() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(2));
        git_at(root, &["commit", "-q", "-m", "between"], &day(2));
        std::fs::write(root.join("README.md"), "observed\n").unwrap();
        git_at(root, &["add", "-A"], &day(3));
        git_at(root, &["commit", "-q", "-m", "failure revision"], &day(3));
        let failure_rev = git_at(root, &["rev-parse", "HEAD"], &day(3));
        git_at(root, &["switch", "-q", "-c", "side", &cause], &day(4));
        std::fs::write(root.join("side.rs"), "fn side() {}\n").unwrap();
        git_at(root, &["add", "-A"], &day(4));
        git_at(root, &["commit", "-q", "-m", "side"], &day(4));
        git_at(root, &["revert", "--no-edit", &cause], &day(5));
        let revert = git_at(root, &["rev-parse", "HEAD"], &day(5));
        git_at(root, &["switch", "-q", &main], &day(5));
        let on_this_branch = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["merge-base", "--is-ancestor", &failure_rev, &revert])
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .status()
            .unwrap();
        assert!(
            !on_this_branch.success(),
            "the revert is not downstream of the failure revision"
        );

        let (storage, _log) = open();
        ingest_repo(&storage, root, &cause, 1).await;
        ingest_repo(&storage, root, &failure_rev, 1).await;
        let pulled = ingest_repo(&storage, root, &revert, 1).await;
        assert!(
            pulled["edges"]["reverts"].as_u64().unwrap() >= 1,
            "{pulled}"
        );
        let failure_id = {
            let tag = crate::tools::repo_ingest::commit_tag(&failure_rev);
            storage
                .current_code_context_nodes("event", Some(&tag), "demo", 5)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
                .id
        };
        let failure = put(&storage, "demo", "failure observed with no path");
        link(&storage, &failure, &failure_id, "derived_from");
        let out = execute(
            &storage,
            Some(json!({
                "scope": "demo",
                "start_points": [{"kind": "logged_write", "node_id": failure}]
            })),
        )
        .await
        .unwrap();
        let cause_row = out["causes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["structure"]["sha"] == cause)
            .unwrap_or_else(|| panic!("{out}"));
        assert_eq!(cause_row["structure"]["edge"], "corrects", "{out}");
    }
}
