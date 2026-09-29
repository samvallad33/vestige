//! # Causal Walk
//!
//! Follows recorded edges only. The start is an exact `logged_write` node id.
//! The other start kinds stay in the schema and return [`NeedsReport`]: a
//! name, a path, a run id, or a tag range is not an edge.
//!
//! An edge counts when its `link_type` is one of [`RECORDED_EDGE_TYPES`]
//! (`touched`, `anchored_to`, `derived_from`, `supersedes`, `corrects`,
//! `closed_by`, `projected_to`, `evidence_of`) and it is not
//! `legacy_inferred`. Shared names are not a join and are not a score.
//!
//! Time is a hard filter: a record at or after the start node is omitted.
//! Ranking is recorded edge strength, then older-first. At most [`MAX_HOPS`]
//! hops. Promote persists `evidence_of` through the existing `save_connection`
//! surface (`SaveEdge`). Candidates are hypotheses.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::git_records;

/// Link type persisted on promote: cause record → evidence record.
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
/// Recorded edge vocabulary the walk will follow. `legacy_inferred` is not one of them.
pub const RECORDED_EDGE_TYPES: &[&str] = &[
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

/// True for a v4 recorded edge. Name overlap is not an edge. `legacy_inferred` is skipped.
pub fn is_recorded_edge(link_type: &str) -> bool {
    link_type != "legacy_inferred" && RECORDED_EDGE_TYPES.contains(&link_type)
}
/// Hard hop cap per cause (MVP bound).
pub const MAX_HOPS: usize = 2;
pub const DEFAULT_LOOKBACK_DAYS: i64 = 30;
/// How many causes to surface.
pub const MAX_CAUSES: usize = 3;
/// How many why-not rejections to report.
pub const MAX_REJECTIONS: usize = 3;

/// The canonical start-point menu, echoed in every needs_report.
pub const REQUIRED_START_POINTS: &[&str] = &[
    "logged_write {node_id}: the failure record; recorded edges are walked backward from it",
    "failing_test {name}: not a recorded edge; pass logged_write of the failure record",
    "stack_frame {frame}: not a recorded edge; pass logged_write of the failure record",
    "ci_run {run_id}: not a recorded edge; pass logged_write of the failure record",
    "version_range {worked_in, broke_in, repo}: not a recorded edge; pass logged_write of the failure record",
];

// ============================================================================
// INPUT TYPES
// ============================================================================

/// One explicit start point. At least one is required; the walk never guesses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StartPoint {
    /// A failing test name. Walked to the records naming it, then to the
    /// commits touching its file (co-touch).
    FailingTest { name: String },
    /// `file:line` or `file` from a failure stack. The last pre-failure
    /// toucher of that file is the prime suspect (SZZ-lite).
    StackFrame { frame: String },
    /// An agent-trace run id. Its failure channel seeds the anchor entities.
    CiRun { run_id: String },
    /// A memory / tool-call record id. Recorded edges from this node are walked.
    LoggedWrite { node_id: String },
    /// `worked_in..broke_in` tag range in `repo`. Restricts commit suspects
    /// to the rev-list commit set.
    VersionRange {
        worked_in: String,
        broke_in: String,
        repo: String,
    },
}

impl StartPoint {
    /// Short human label for reports ("why not X" style output).
    pub fn label(&self) -> String {
        match self {
            StartPoint::FailingTest { name } => format!("failing_test {name}"),
            StartPoint::StackFrame { frame } => format!("stack_frame {frame}"),
            StartPoint::CiRun { run_id } => format!("ci_run {run_id}"),
            StartPoint::LoggedWrite { node_id } => format!("logged_write {node_id}"),
            StartPoint::VersionRange {
                worked_in,
                broke_in,
                repo,
            } => {
                format!("version_range {worked_in}..{broke_in} in {repo}")
            }
        }
    }
}

/// One resolved `git rev-list worked_in..broke_in` commit set.
#[derive(Debug, Clone)]
pub struct ResolvedRange {
    pub worked_in: String,
    pub broke_in: String,
    pub shas: HashSet<String>,
}

/// One persisted edge the walk may follow.
#[derive(Debug, Clone)]
pub struct RecordedEdge {
    pub source_id: String,
    pub target_id: String,
    pub link_type: String,
    pub strength: f64,
    /// Migration flag. The live `SaveEdge` record has no such bit; a link whose
    /// type is the literal `legacy_inferred` is still skipped.
    pub legacy_inferred: bool,
}

/// Failure-channel evidence harvested from one agent-trace run.
#[derive(Debug, Clone, Default)]
pub struct RunEvidence {
    /// Entities from vetoed-claim text (the run's failure vocabulary).
    pub entities: Vec<String>,
    /// Memory ids the run touched (retrieved / written / cited as evidence).
    pub referenced_ids: Vec<String>,
    /// Last trace event time, epoch millis.
    pub last_event_at: Option<i64>,
}

/// Everything the pure walk needs that is not in the record list itself.
#[derive(Debug, Default)]
pub struct WalkContext {
    pub now: DateTime<Utc>,
    /// run_id → harvested evidence (missing run = absent key).
    pub runs: HashMap<String, RunEvidence>,
    /// Recorded edges. Name overlap is not in this list. `legacy_inferred` is never inserted.
    pub edges: Vec<RecordedEdge>,
    /// start-point index → resolved rev-list (or the error that blocked it).
    pub ranges: HashMap<usize, Result<ResolvedRange, String>>,
    /// True when more than one distinct repo appeared in version_range starts.
    pub multiple_repos: bool,
}

/// Tuning knobs; mirrors the backfill's knobs where they overlap.
#[derive(Debug, Clone)]
pub struct CausalWalkOptions {
    pub lookback_days: i64,
    pub max_causes: usize,
    pub max_rejections: usize,
}

impl Default for CausalWalkOptions {
    fn default() -> Self {
        Self {
            lookback_days: DEFAULT_LOOKBACK_DAYS,
            max_causes: MAX_CAUSES,
            max_rejections: MAX_REJECTIONS,
        }
    }
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

/// One hop on the trail from a start point to a suspect.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathHop {
    /// Mechanism that produced this hop, e.g. `failing_test/co_touch`.
    pub via: String,
    /// What the hop passes through (a file, a frame, a run, a node id, a range).
    pub hop: String,
}

/// One suspect change record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalCause {
    /// Record id of the suspect.
    pub id: String,
    /// Commit sha when the suspect is a change record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
    /// Recorded edge strength. Names do not contribute.
    pub score: f64,
    /// The trail; at most [`MAX_HOPS`] hops.
    pub path: Vec<PathHop>,
    /// Always empty. Names are not a join; the field stays so older clients
    /// still parse the object.
    pub shared_anchors: Vec<String>,
    /// Records this cause should be linked to on promote (start-point
    /// evidence records). Not part of the wire output.
    #[serde(skip)]
    pub evidence_to: Vec<String>,
}

/// Returned when the walk refuses: no start point, or a start point that
/// anchored to nothing. NEVER guessed around.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeedsReport {
    /// What each dead start point was missing.
    pub missing: Vec<String>,
    /// What the caller must provide to make the walk runnable.
    pub required_start_points: Vec<String>,
}

/// One near-miss with the rule that excluded it (why-not top N).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalkRejection {
    pub id: String,
    pub reason: String,
    /// Shared-anchor count at rejection time.
    pub shared_anchors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalWalkResult {
    pub causes: Vec<CausalCause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs_report: Option<NeedsReport>,
    #[serde(default)]
    pub rejected: Vec<WalkRejection>,
}

// ============================================================================
// RECORD VIEW (storage-agnostic)
// ============================================================================

/// The minimal view of a record the walk needs. Built from `KnowledgeNode`s
/// by [`walk_storage`]; pure so the engine is testable without a store.
#[derive(Debug, Clone)]
pub struct WalkRecord {
    pub id: String,
    pub content: String,
    pub entities: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub stability: f64,
    /// Tagged `git-commit` (a change, not a report about one).
    pub is_commit: bool,
    /// Full/lowercased sha from the record's `commit <sha>` header.
    pub sha: Option<String>,
    /// Files from the record's `files:` line (commit records).
    pub files: Vec<String>,
    /// Path-qualified symbols from the record's `symbols:` line.
    pub symbols: Vec<String>,
}

/// The full sha a commit record carries (first line: `commit <sha> ...`).
/// Same pattern as the MCP backfill tool so both sides of the wire agree.
#[allow(dead_code)] // callers sit in the legacy-sqlite-gated surface; dead only in the no-embeddings profile
fn commit_sha_of(content: &str) -> Option<String> {
    let mut words = content.split_whitespace();
    if words.next() != Some("commit") {
        return None;
    }
    let sha = words.next()?;
    (sha.len() >= 7 && sha.chars().all(|c| c.is_ascii_hexdigit())).then(|| sha.to_ascii_lowercase())
}

/// Parse the `files:` / `symbols:` lines of a `git_records::record_content`
/// body. `( +N more)` truncation markers are dropped, never parsed as paths.
#[allow(dead_code)] // callers sit in the legacy-sqlite-gated surface; dead only in the no-embeddings profile
fn parse_prefixed_line(content: &str, prefix: &str) -> Vec<String> {
    content
        .lines()
        .find(|l| l.starts_with(prefix))
        .map(|l| {
            l[prefix.len()..]
                .split(", ")
                .map(|p| {
                    // strip a trailing " (+N more)" marker from the last entry
                    match p.find(" (+") {
                        Some(i) => p[..i].to_string(),
                        None => p.to_string(),
                    }
                })
                .filter(|p| !p.is_empty() && !p.starts_with('('))
                .collect()
        })
        .unwrap_or_default()
}

#[allow(dead_code)] // callers sit in the legacy-sqlite-gated surface; dead only in the no-embeddings profile
fn walk_record_of(
    id: &str,
    content: &str,
    tags: &[String],
    created_at: DateTime<Utc>,
    stability: f64,
) -> WalkRecord {
    let is_commit = tags.iter().any(|t| t == git_records::COMMIT_TAG);
    WalkRecord {
        id: id.to_string(),
        content: content.to_string(),
        entities: Vec::new(),
        created_at,
        stability,
        is_commit,
        sha: if is_commit {
            commit_sha_of(content)
        } else {
            None
        },
        files: if is_commit {
            parse_prefixed_line(content, "files: ")
        } else {
            vec![]
        },
        symbols: if is_commit {
            parse_prefixed_line(content, "symbols: ")
        } else {
            vec![]
        },
    }
}

impl CausalWalkOptions {
    /// Run the walk. `records` is the whole in-scope pool; `starts` the
    /// explicit evidence handles; `ctx` the out-of-band evidence (runs,
    /// edges, rev-lists).
    pub fn walk(
        &self,
        starts: &[StartPoint],
        records: &[WalkRecord],
        ctx: &WalkContext,
    ) -> CausalWalkResult {
        let now = if ctx.now.timestamp() == 0 {
            Utc::now()
        } else {
            ctx.now
        };
        let mut missing: Vec<String> = Vec::new();

        if starts.is_empty() {
            return CausalWalkResult {
                causes: vec![],
                needs_report: Some(NeedsReport {
                    missing: vec![
                        "no start point given; causal_walk never guesses a failure origin"
                            .to_string(),
                    ],
                    required_start_points: REQUIRED_START_POINTS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                }),
                rejected: vec![],
            };
        }
        if ctx.multiple_repos {
            missing.push(
                "single repo per call: multiple version_range start points name different repos"
                    .to_string(),
            );
        }

        // The start is an exact handle. Names, file paths, and shared tokens
        // are not a join. Only `logged_write` names a record; the other
        // start kinds do not become edges.
        let by_id: HashMap<&str, &WalkRecord> =
            records.iter().map(|r| (r.id.as_str(), r)).collect();
        let mut anchor_ids: Vec<String> = Vec::new();
        let mut anchor_time: Option<DateTime<Utc>> = None;
        for start in starts {
            match start {
                StartPoint::LoggedWrite { node_id } => match by_id.get(node_id.as_str()) {
                    None => {
                        missing.push(format!("logged write '{node_id}' not found in this scope"))
                    }
                    Some(r) => {
                        anchor_ids.push(r.id.clone());
                        anchor_time = Some(r.created_at);
                    }
                },
                other => missing.push(format!(
                    "{} is not a recorded edge; pass logged_write of the failure record",
                    other.label()
                )),
            }
        }
        let failure_time = anchor_time.unwrap_or(now);
        let anchor_set: HashSet<&str> = anchor_ids.iter().map(|id| id.as_str()).collect();

        // Follow recorded edges only, and only to a strictly earlier record.
        let mut best: HashMap<&str, (f64, Vec<PathHop>)> = HashMap::new();
        let mut rejections: Vec<WalkRejection> = Vec::new();
        let mut queue: Vec<(&str, usize)> = anchor_set.iter().copied().map(|id| (id, 0)).collect();
        let mut queued: HashSet<&str> = anchor_set.clone();
        let mut qi = 0;
        while qi < queue.len() {
            let (id, depth) = queue[qi];
            qi += 1;
            if depth >= MAX_HOPS {
                continue;
            }
            for edge in &ctx.edges {
                if edge.legacy_inferred || !is_recorded_edge(&edge.link_type) {
                    continue;
                }
                let other = if edge.source_id == id {
                    edge.target_id.as_str()
                } else if edge.target_id == id {
                    edge.source_id.as_str()
                } else {
                    continue;
                };
                if anchor_set.contains(other) {
                    continue;
                }
                let Some(rec) = by_id.get(other) else {
                    continue;
                };
                if rec.created_at >= failure_time {
                    continue;
                }
                let age_days = (failure_time - rec.created_at).num_seconds() as f64 / 86_400.0;
                if age_days > self.lookback_days as f64 {
                    rejections.push(WalkRejection {
                        id: rec.id.clone(),
                        reason: format!("outside the {}d lookback window", self.lookback_days),
                        shared_anchors: 0,
                    });
                    continue;
                }
                let hop = PathHop {
                    via: format!("edge/{}", edge.link_type),
                    hop: format!("{id} -> {other}"),
                };
                let slot = best.entry(other).or_insert_with(|| (0.0, Vec::new()));
                if edge.strength >= slot.0 {
                    slot.0 = edge.strength;
                }
                if slot.1.len() < MAX_HOPS {
                    slot.1.push(hop);
                }
                if queued.insert(other) {
                    queue.push((other, depth + 1));
                }
            }
        }

        let mut causes: Vec<CausalCause> = best
            .iter()
            .filter_map(|(id, (strength, hops))| {
                let rec = by_id.get(id)?;
                Some(CausalCause {
                    id: rec.id.clone(),
                    sha: None,
                    score: *strength,
                    path: hops.clone(),
                    shared_anchors: Vec::new(),
                    evidence_to: anchor_ids.clone(),
                })
            })
            .collect();
        causes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ta = by_id.get(a.id.as_str()).map(|r| r.created_at);
                    let tb = by_id.get(b.id.as_str()).map(|r| r.created_at);
                    ta.cmp(&tb)
                })
        });
        causes.truncate(self.max_causes);
        rejections.sort_by(|a, b| a.reason.cmp(&b.reason).then(a.id.cmp(&b.id)));
        rejections.truncate(self.max_rejections);

        let needs_report = if causes.is_empty() {
            if missing.is_empty() {
                missing.push(
                    "no recorded edge leads to an earlier record in the lookback window".into(),
                );
            }
            Some(NeedsReport {
                missing,
                required_start_points: REQUIRED_START_POINTS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            })
        } else {
            None
        };

        CausalWalkResult {
            causes,
            needs_report,
            rejected: rejections,
        }
    }
}

// ============================================================================
// STORAGE ASSEMBLY + PROMOTE
// ============================================================================

/// A fully-specified causal walk against a store.
#[derive(Debug, Clone)]
pub struct CausalWalkRequest {
    pub scope: String,
    pub start_points: Vec<StartPoint>,
    pub lookback_days: i64,
    /// Max records scanned (clamped to 10..=5000 like the backfill tool).
    pub scan_limit: i32,
}

impl Default for CausalWalkRequest {
    fn default() -> Self {
        Self {
            scope: "user".to_string(),
            start_points: vec![],
            lookback_days: DEFAULT_LOOKBACK_DAYS,
            scan_limit: 500,
        }
    }
}

/// Assemble the record pool and recorded edges from `storage`, then run
/// the pure walk. `Err` only for storage/IO failures; refusals are
/// `needs_report`, not errors.
pub fn walk_storage(
    storage: &crate::storage::Storage,
    req: &CausalWalkRequest,
) -> Result<CausalWalkResult, String> {
    let scope = req.scope.trim();
    if scope.is_empty() {
        return Err("scope must not be empty".into());
    }
    let scan_limit = req.scan_limit.clamp(10, 5000);
    let lookback = req.lookback_days.clamp(1, 365);

    let nodes = storage
        .get_all_nodes_in_scope(scope, scan_limit, 0)
        .map_err(|e| e.to_string())?;
    let records: Vec<WalkRecord> = nodes
        .iter()
        .map(|n| walk_record_of(&n.id, &n.content, &n.tags, n.created_at, n.stability))
        .collect();

    let edges = storage
        .get_all_connections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|edge| is_recorded_edge(&edge.link_type))
        .map(|edge| RecordedEdge {
            source_id: edge.source_id,
            target_id: edge.target_id,
            link_type: edge.link_type,
            strength: edge.strength,
            legacy_inferred: false,
        })
        .collect();
    let ctx = WalkContext {
        now: Utc::now(),
        edges,
        ..WalkContext::default()
    };

    Ok(CausalWalkOptions {
        lookback_days: lookback,
        ..CausalWalkOptions::default()
    }
    .walk(&req.start_points, &records, &ctx))
}

/// Persist the trail as `evidence_of` edges (cause → evidence records).
/// Called ONLY on promote; preview writes nothing. Returns the edges written.
pub fn persist_evidence_edges(
    storage: &crate::storage::Storage,
    result: &CausalWalkResult,
) -> Result<Vec<(String, String)>, String> {
    let mut written = Vec::new();
    for cause in &result.causes {
        let existing = storage
            .get_connections_for_memory(&cause.id)
            .map_err(|e| e.to_string())?;
        for target in &cause.evidence_to {
            if target == &cause.id {
                continue;
            }
            let already = existing.iter().any(|c| {
                c.source_id == cause.id
                    && c.target_id == *target
                    && c.link_type == EVIDENCE_LINK_TYPE
            });
            if already {
                continue;
            }
            storage
                .save_connection(&crate::ConnectionRecord {
                    source_id: cause.id.clone(),
                    target_id: target.clone(),
                    strength: cause.score.clamp(0.0, 1.0),
                    link_type: EVIDENCE_LINK_TYPE.to_string(),
                    created_at: Utc::now(),
                    last_activated: Utc::now(),
                    activation_count: 0,
                })
                .map_err(|e| e.to_string())?;
            written.push((cause.id.clone(), target.clone()));
        }
    }
    Ok(written)
}

// ============================================================================
// TESTS — temp store seeded with git-commit records fabricated through the
// real IngestInput + record_content pipeline.
// ============================================================================

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;
    use crate::storage::SqliteMemoryStore as Storage;
    use crate::{IngestInput, KnowledgeNode};
    use tempfile::TempDir;

    const SCOPE: &str = "walk-test";

    fn store() -> (Storage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("walk.db"))).unwrap();
        (storage, dir)
    }

    fn sha_of(c: char) -> String {
        c.to_string().repeat(40)
    }

    /// Seed one record into SCOPE, backdated `days_ago`, return its node.
    fn seed(storage: &Storage, content: &str, tags: Vec<&str>, days_ago: i64) -> KnowledgeNode {
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    ..Default::default()
                },
                SCOPE,
            )
            .unwrap();
        storage
            .set_created_at(&node.id, Utc::now() - chrono::Duration::days(days_ago))
            .unwrap();
        storage.get_node(&node.id).unwrap().unwrap()
    }

    fn commit_record(
        storage: &Storage,
        sha: &str,
        subject: &str,
        files: &[&str],
        symbols: &[&str],
        mentions: &[&str],
        days_ago: i64,
    ) -> KnowledgeNode {
        let content = git_records::record_content(&git_records::GitCommit {
            sha: sha.to_string(),
            time: Utc::now() - chrono::Duration::days(days_ago),
            subject: subject.to_string(),
            files: files.iter().map(|f| f.to_string()).collect(),
            extra_files: 0,
            symbols: symbols.iter().map(|s| s.to_string()).collect(),
            mentions: mentions.iter().map(|m| m.to_string()).collect(),
            hunks: vec![],
            extra_hunks: 0,
            imports: vec![],
        });
        seed(storage, &content, vec![git_records::COMMIT_TAG], days_ago)
    }

    fn req(starts: Vec<StartPoint>) -> CausalWalkRequest {
        CausalWalkRequest {
            scope: SCOPE.into(),
            start_points: starts,
            ..Default::default()
        }
    }

    #[test]
    fn name_start_points_are_not_edges() {
        let (storage, _dir) = store();
        let symptom = seed(
            &storage,
            "Checkout lane returned 504. Gateway gave up.",
            vec![],
            1,
        );
        let cause = seed(
            &storage,
            "Raised the pool wait inside the billing client before the quiet deploy window.",
            vec![],
            5,
        );
        storage
            .save_connection(&crate::ConnectionRecord {
                source_id: cause.id.clone(),
                target_id: symptom.id.clone(),
                strength: 1.0,
                link_type: "derived_from".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();

        for start in [
            StartPoint::FailingTest {
                name: "Checkout".into(),
            },
            StartPoint::StackFrame {
                frame: "lane:12".into(),
            },
            StartPoint::CiRun {
                run_id: "run-1".into(),
            },
            StartPoint::VersionRange {
                worked_in: "w1".into(),
                broke_in: "b1".into(),
                repo: "/nonexistent/repo".into(),
            },
        ] {
            let result = walk_storage(&storage, &req(vec![start])).unwrap();
            assert!(result.causes.is_empty(), "{:?}", result.causes);
            let report = result.needs_report.expect("a name is not an edge");
            assert!(
                report
                    .missing
                    .iter()
                    .any(|m| m.contains("not a recorded edge")),
                "{:?}",
                report.missing
            );
        }
    }

    #[test]
    fn logged_write_follows_recorded_edge_not_shared_names() {
        let (storage, _dir) = store();
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
        let later = seed(
            &storage,
            "Zephyr quilt inventory shifted after the quiet hour.",
            vec![],
            0,
        );
        let link = |source: &KnowledgeNode, target: &KnowledgeNode, link_type: &str| {
            storage
                .save_connection(&crate::ConnectionRecord {
                    source_id: source.id.clone(),
                    target_id: target.id.clone(),
                    strength: 1.0,
                    link_type: link_type.into(),
                    created_at: Utc::now(),
                    last_activated: Utc::now(),
                    activation_count: 0,
                })
                .unwrap();
        };
        link(&cause, &symptom, "derived_from");
        link(&decoy, &symptom, "reports");
        link(&later, &symptom, "derived_from");

        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::LoggedWrite {
                node_id: symptom.id.clone(),
            }]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        assert_eq!(result.causes.len(), 1, "{:?}", result.causes);
        let top = &result.causes[0];
        assert_eq!(top.id, cause.id);
        assert!(top.shared_anchors.is_empty());
        assert_eq!(top.path[0].via, "edge/derived_from");
        assert!(top.evidence_to.contains(&symptom.id));
        assert!(
            result
                .causes
                .iter()
                .all(|c| c.id != decoy.id && c.id != later.id)
        );
        let rendered = format!("{result:?}");
        assert!(!rendered.contains(&later.id), "{rendered}");
        assert!(!rendered.contains(&decoy.id), "{rendered}");
    }

    #[test]
    fn no_start_point_refuses_with_needs_report() {
        let (storage, _dir) = store();
        commit_record(
            &storage,
            &sha_of('f'),
            "harden login",
            &["tests/auth_test.rs"],
            &[],
            &[],
            3,
        );
        let result = walk_storage(&storage, &req(vec![])).unwrap();
        let report = result.needs_report.expect("must refuse, never guess");
        assert!(result.causes.is_empty());
        assert!(report.missing[0].contains("never guesses"));
        assert_eq!(report.required_start_points.len(), 5);
    }

    #[test]
    fn wrong_handle_refuses_with_needs_report() {
        let (storage, _dir) = store();
        seed(&storage, "unrelated note", vec![], 1);

        // a test name nothing mentions
        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::FailingTest {
                name: "test_ghost".into(),
            }]),
        )
        .unwrap();
        let report = result.needs_report.expect("unanchored test must refuse");
        assert!(result.causes.is_empty());
        assert!(
            report.missing.iter().any(|m| m.contains("test_ghost")),
            "{:?}",
            report.missing
        );
        assert!(!report.required_start_points.is_empty());

        // a logged write that does not exist in the scope
        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::LoggedWrite {
                node_id: "ghost-id".into(),
            }]),
        )
        .unwrap();
        let report = result.needs_report.expect("unknown node must refuse");
        assert!(result.causes.is_empty());
        assert!(report.missing.iter().any(|m| m.contains("ghost-id")));
    }

    #[test]
    fn legacy_inferred_and_freeform_links_are_skipped() {
        assert!(is_recorded_edge("derived_from"));
        assert!(!is_recorded_edge("legacy_inferred"));
        assert!(!is_recorded_edge("reports"));
        let (storage, _dir) = store();
        let cause = seed(&storage, "billing client raised the pool wait", vec![], 3);
        let symptom = seed(&storage, "gateway gave up on the quiet hour", vec![], 1);
        storage
            .save_connection(&crate::ConnectionRecord {
                source_id: cause.id.clone(),
                target_id: symptom.id.clone(),
                strength: 1.0,
                link_type: "derived_from".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();
        let records = vec![
            super::walk_record_of(
                &cause.id,
                &cause.content,
                &cause.tags,
                cause.created_at,
                cause.stability,
            ),
            super::walk_record_of(
                &symptom.id,
                &symptom.content,
                &symptom.tags,
                symptom.created_at,
                symptom.stability,
            ),
        ];
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
            &records,
            &ctx,
        );
        assert!(result.causes.is_empty(), "{:?}", result.causes);
        assert!(result.needs_report.is_some());
    }

    #[test]
    fn promote_persists_evidence_of_edges_and_preview_does_not() {
        let (storage, _dir) = store();
        let cause = seed(
            &storage,
            "Raised the pool wait inside the billing client before the quiet deploy window.",
            vec![],
            5,
        );
        let symptom = seed(
            &storage,
            "Checkout lane returned 504. Gateway gave up.",
            vec![],
            1,
        );
        storage
            .save_connection(&crate::ConnectionRecord {
                source_id: cause.id.clone(),
                target_id: symptom.id.clone(),
                strength: 1.0,
                link_type: "derived_from".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();
        let request = req(vec![StartPoint::LoggedWrite {
            node_id: symptom.id.clone(),
        }]);
        let preview = walk_storage(&storage, &request).unwrap();
        assert_eq!(preview.causes.len(), 1);
        assert_eq!(preview.causes[0].id, cause.id);
        assert!(
            storage
                .get_connections_for_memory(&cause.id)
                .unwrap()
                .iter()
                .all(|edge| edge.link_type != EVIDENCE_LINK_TYPE)
        );

        let written = persist_evidence_edges(&storage, &preview).unwrap();
        assert_eq!(written, vec![(cause.id.clone(), symptom.id.clone())]);
        let edges = storage.get_connections_for_memory(&cause.id).unwrap();
        assert!(edges.iter().any(|edge| {
            edge.source_id == cause.id
                && edge.target_id == symptom.id
                && edge.link_type == EVIDENCE_LINK_TYPE
        }));
        assert!(
            persist_evidence_edges(&storage, &preview)
                .unwrap()
                .is_empty()
        );
    }

    // ---- pure helpers ----

    #[test]
    fn commit_sha_and_prefixed_lines_parse_the_record_format() {
        let content = git_records::record_content(&git_records::GitCommit {
            sha: sha_of('1'),
            time: Utc::now(),
            subject: "big move".into(),
            files: (0..52).map(|i| format!("src/f{i}.rs")).collect(),
            extra_files: 2,
            symbols: vec!["src/f0.rs/handler_0".into()],
            mentions: vec!["API_TIMEOUT".into()],
            hunks: vec![],
            extra_hunks: 0,
            imports: vec![],
        });
        assert_eq!(commit_sha_of(&content), Some(sha_of('1')));
        assert_eq!(commit_sha_of("not a commit record"), None);
        // record_content prints every file the GitCommit carries (the 50-file
        // cap lives in parse_git_log's push_file, not here) plus the marker.
        let files = parse_prefixed_line(&content, "files: ");
        assert_eq!(files.len(), 52);
        assert_eq!(files[0], "src/f0.rs");
        assert!(!files.iter().any(|f| f.contains("(+")), "{files:?}");
        assert_eq!(
            parse_prefixed_line(&content, "symbols: "),
            vec!["src/f0.rs/handler_0".to_string()]
        );
    }
}
