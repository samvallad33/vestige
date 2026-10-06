//! # Causal Walk
//!
//! The successor to [`super::retroactive_backfill`]. Backfill starts from ONE
//! failure memory and reaches backward through shared entities; causal_walk
//! starts from EXPLICIT evidence handles the caller already holds — a failing
//! test, a stack frame, a CI run, a logged write, a version range — and walks
//! exact, mechanism-specific edges to the change records behind the failure.
//! The start point is never guessed: with no start point, or a start point
//! that anchors to nothing in the store, the walk refuses with a
//! [`NeedsReport`] instead of inventing a trail (the same no-fabrication rule
//! that keeps backfill from linking on vocabulary alone).
//!
//! ## Mechanisms (all exact, all backward-only)
//!
//! 1. **failing_test** → the test-file record(s) naming the test → the commit
//!    records touching that file → for each, its OTHER files are the suspects
//!    (co-touch: the regression rides in on a commit that also touched the
//!    test).
//! 2. **stack_frame** (`file:line` or `file`) → commit records whose hunk
//!    spans (files / path-qualified symbols) cover that file → the LAST commit
//!    touching the file before the failure anchor is the prime suspect
//!    (SZZ-lite; sha matching reuses the `commit <sha>` record-header pattern).
//! 3. **ci_run** → the agent-trace run's failure channel (vetoed claims plus
//!    the memories the run retrieved/wrote) → their entities → records.
//! 4. **logged_write** → the node → its persisted connection edges →
//!    neighbours, plus records sharing its entities.
//! 5. **version_range** (`worked_in`/`broke_in`/`repo`) → the `git rev-list`
//!    commit set between the two tags → a RESTRICTION on commit candidates
//!    (out-of-range commits are rejected with a why-not) plus a second hop for
//!    in-range commits overlapping the failure anchors.
//!
//! ## Ranking
//!
//! Ported from `run_trail`: IDF-weighted shared anchors (probabilistic IDF ×
//! identifier-tier weight — a shared file path is worth ~3x a shared word, and
//! a name every candidate carries is boilerplate, not a clue), a gentle
//! recency term, the change-record bonus (the change, not the chatter about
//! it), and older-first tie-breaks. Embedding similarity is not an input in
//! either direction, for the same reason as backfill.
//!
//! ## MVP bounds
//!
//! Single repo per call; at most [`MAX_HOPS`] hops per cause; supersession is
//! not followed (commit records are upserted idempotently, not superseded).
//! Trail edges persist ONLY on promote, as `evidence_of` links via the
//! existing `save_connection` surface.
//!
//! Candidates are hypotheses. Nothing here proves causation.

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::git_records;
use super::retroactive_backfill::{
    CHANGE_RECORD_BONUS, IdentifierTier, extract_entities, normalized_tier,
};

/// Link type persisted on promote: cause record → evidence record.
pub const EVIDENCE_LINK_TYPE: &str = "evidence_of";
/// Hard hop cap per cause (MVP bound).
pub const MAX_HOPS: usize = 2;
pub const DEFAULT_LOOKBACK_DAYS: i64 = 30;
/// How many causes to surface.
pub const MAX_CAUSES: usize = 3;
/// How many why-not rejections to report.
pub const MAX_REJECTIONS: usize = 3;

/// The canonical start-point menu, echoed in every needs_report.
pub const REQUIRED_START_POINTS: &[&str] = &[
    "failing_test {name}: a failing test whose file's co-touch commits become suspects",
    "stack_frame {frame: \"file:line\" or \"file\"}: blame the last pre-failure toucher",
    "ci_run {run_id}: the failure channel (vetoed claims / touched memories) of an agent-trace run",
    "logged_write {node_id}: a memory or tool-call record whose edges to walk",
    "version_range {worked_in, broke_in, repo}: restrict commit suspects to a tag range (pair with a failure start point)",
];

// ============================================================================
// INPUT TYPES
// ============================================================================

/// One explicit start point. At least one is required; the walk never guesses.
///
/// Every variant may carry `node_id`: the id of the memory that RECORDS this
/// symptom (the id its write returned). A start point is a description
/// of the failure; `node_id` is the handle the log can walk from. On a Strata
/// log the walk begins at recorded nodes only, so `node_id` is what makes a
/// `failing_test`, `stack_frame`, `ci_run` or `version_range` start walkable.
/// `logged_write` is the variant whose only content is that handle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StartPoint {
    /// A failing test name. Walked to the records naming it, then to the
    /// commits touching its file (co-touch).
    FailingTest {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
    },
    /// `file:line` or `file` from a failure stack. The last pre-failure
    /// toucher of that file is the prime suspect (SZZ-lite).
    StackFrame {
        frame: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
    },
    /// An agent-trace run id. Its failure channel seeds the anchor entities.
    CiRun {
        run_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
    },
    /// A memory / tool-call record id. Its edges and shared entities are
    /// walked.
    LoggedWrite { node_id: String },
    /// `worked_in..broke_in` tag range in `repo`. Restricts commit suspects
    /// to the rev-list commit set.
    VersionRange {
        worked_in: String,
        broke_in: String,
        repo: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
    },
}

impl StartPoint {
    /// The `kind` string this start point serializes with.
    pub fn kind(&self) -> &'static str {
        match self {
            StartPoint::FailingTest { .. } => "failing_test",
            StartPoint::StackFrame { .. } => "stack_frame",
            StartPoint::CiRun { .. } => "ci_run",
            StartPoint::LoggedWrite { .. } => "logged_write",
            StartPoint::VersionRange { .. } => "version_range",
        }
    }

    /// The recorded node this start point names, whatever its kind: the
    /// trimmed, non-empty `node_id`, or `None`.
    pub fn record_id(&self) -> Option<&str> {
        let raw = match self {
            StartPoint::FailingTest { node_id, .. }
            | StartPoint::StackFrame { node_id, .. }
            | StartPoint::CiRun { node_id, .. }
            | StartPoint::VersionRange { node_id, .. } => node_id.as_deref(),
            StartPoint::LoggedWrite { node_id } => Some(node_id.as_str()),
        };
        raw.map(str::trim).filter(|id| !id.is_empty())
    }

    /// Short human label for reports ("why not X" style output).
    pub fn label(&self) -> String {
        match self {
            StartPoint::FailingTest { name, .. } => format!("failing_test {name}"),
            StartPoint::StackFrame { frame, .. } => format!("stack_frame {frame}"),
            StartPoint::CiRun { run_id, .. } => format!("ci_run {run_id}"),
            StartPoint::LoggedWrite { node_id } => format!("logged_write {node_id}"),
            StartPoint::VersionRange {
                worked_in,
                broke_in,
                repo,
                ..
            } => {
                format!("version_range {worked_in}..{broke_in} in {repo}")
            }
        }
    }
}

/// Expand every `node_id` carried by a non-`logged_write` start point into an
/// explicit `logged_write` start, so the walk engine only ever sees the five
/// canonical shapes.
///
/// The original start points keep their positions (with `node_id` cleared), so
/// index-keyed evidence such as resolved version ranges stays aligned. The
/// synthesized `logged_write` starts are appended in first-seen order and a
/// node already named by an explicit `logged_write` is not repeated. The
/// function is idempotent: normalizing its own output changes nothing.
pub fn normalize_start_points(starts: &[StartPoint]) -> Vec<StartPoint> {
    // A node an explicit logged_write already names is never repeated, even
    // when the logged_write comes after the start point that carries it.
    let mut seen: HashSet<String> = starts
        .iter()
        .filter(|start| matches!(start, StartPoint::LoggedWrite { .. }))
        .filter_map(|start| start.record_id().map(str::to_string))
        .collect();
    let mut out: Vec<StartPoint> = Vec::with_capacity(starts.len());
    let mut appended: Vec<StartPoint> = Vec::new();
    for start in starts {
        let mut kept = start.clone();
        match &mut kept {
            StartPoint::FailingTest { node_id, .. }
            | StartPoint::StackFrame { node_id, .. }
            | StartPoint::CiRun { node_id, .. }
            | StartPoint::VersionRange { node_id, .. } => *node_id = None,
            StartPoint::LoggedWrite { node_id } => *node_id = node_id.trim().to_string(),
        }
        if !matches!(start, StartPoint::LoggedWrite { .. })
            && let Some(id) = start.record_id()
            && seen.insert(id.to_string())
        {
            appended.push(StartPoint::LoggedWrite {
                node_id: id.to_string(),
            });
        }
        out.push(kept);
    }
    out.extend(appended);
    out
}

/// One resolved `git rev-list worked_in..broke_in` commit set.
#[derive(Debug, Clone)]
pub struct ResolvedRange {
    pub worked_in: String,
    pub broke_in: String,
    pub shas: HashSet<String>,
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
    /// node_id → neighbour node ids (either side of a persisted edge).
    pub edges: HashMap<String, Vec<String>>,
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
    /// Ranking score (IDF-weighted anchors + recency + change bonus).
    pub score: f64,
    /// The trail; at most [`MAX_HOPS`] hops.
    pub path: Vec<PathHop>,
    /// Entities shared with the start-point anchors (the causal join).
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
        entities: extract_entities(content, tags),
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

/// Normalize a caller-supplied handle (test name, frame file) into entity
/// form using the SAME extractor the record side uses, so both ends of the
/// join agree. Returns 0..1 anchors (a single token is 0 or 1 entity).
fn normalized_anchors(raw: &str) -> Vec<String> {
    extract_entities(raw, &[])
}

/// Path-suffix match: `src/auth.rs` matches `src/auth.rs` and either side
/// being a deeper path of the other (`a/./b` normalization not needed for
/// repo-relative paths).
fn path_matches(file: &str, anchor: &str) -> bool {
    let (f, a) = (file.to_lowercase(), anchor.to_lowercase());
    f == a || f.ends_with(&format!("/{a}")) || a.ends_with(&format!("/{f}"))
}

// ============================================================================
// THE WALK (pure)
// ============================================================================

#[derive(Debug, Default, Clone)]
struct Seed {
    anchors: BTreeSet<String>,
    hops: Vec<PathHop>,
    /// Seeded by a persisted edge, not by vocabulary: survives with zero
    /// shared anchors because the edge itself is the evidence.
    via_edge: bool,
}

/// Record one mechanism reaching one record. Multiple mechanisms reaching the
/// same record merge anchors and hops (capped at [`MAX_HOPS`] at output).
fn add_seed<'a>(
    seeds: &mut HashMap<&'a str, Seed>,
    record: &'a WalkRecord,
    anchors_shared: Vec<String>,
    hop: PathHop,
    via_edge: bool,
) {
    let entry = seeds.entry(record.id.as_str()).or_default();
    entry.anchors.extend(anchors_shared);
    entry.hops.push(hop);
    entry.via_edge |= via_edge;
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
        // Idempotent: `walk_storage` already normalized, so for its callers
        // this is the same list. Direct callers may pass start points that
        // carry a node_id.
        let normalized = normalize_start_points(starts);
        let starts = normalized.as_slice();
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

        // ------------------------------------------------------------------
        // Pass A: anchor records / entities / failure time per start point.
        // The failure anchor is the NEWEST thing the start points pin down:
        // after that, the failure is live and nothing newer is a cause.
        // ------------------------------------------------------------------
        struct StartAnchors {
            entities: BTreeSet<String>,
            record_ids: BTreeSet<String>,
            time: Option<DateTime<Utc>>,
        }
        let by_id: HashMap<&str, &WalkRecord> =
            records.iter().map(|r| (r.id.as_str(), r)).collect();
        let mut anchors: Vec<Option<StartAnchors>> = Vec::with_capacity(starts.len());
        for start in starts {
            let mut a = StartAnchors {
                entities: BTreeSet::new(),
                record_ids: BTreeSet::new(),
                time: None,
            };
            match start {
                StartPoint::FailingTest { name, .. } => {
                    // test-file records: records naming the test
                    let name_anchors = normalized_anchors(name);
                    if name_anchors.is_empty() {
                        missing.push(format!(
                            "failing test '{name}' is not identifier-shaped; give the test's exact name"
                        ));
                    } else {
                        a.entities.extend(name_anchors.iter().cloned());
                    }
                    for r in records {
                        if name_anchors
                            .iter()
                            .any(|n| r.content.to_lowercase().contains(n))
                        {
                            a.entities.extend(r.entities.iter().cloned());
                            a.record_ids.insert(r.id.clone());
                            a.time = a.time.max(Some(r.created_at));
                        }
                    }
                    if a.record_ids.is_empty() && !name_anchors.is_empty() {
                        missing.push(format!(
                            "no record in scope names failing test '{name}' — ingest the failing run's note or its test-file commits"
                        ));
                    }
                }
                StartPoint::StackFrame { frame, .. } => {
                    let file = frame.split(':').next().unwrap_or(frame).trim();
                    let file_anchors = normalized_anchors(file);
                    if file_anchors.is_empty() {
                        missing.push(format!(
                            "stack frame '{frame}' names no identifier-shaped file"
                        ));
                    } else {
                        a.entities.extend(file_anchors.iter().cloned());
                    }
                    // no anchor records of its own: a note naming the file
                    // (if any) helps date the failure
                    for r in records {
                        if file_anchors.iter().any(|f| r.entities.contains(f)) {
                            a.time = a.time.max(Some(r.created_at));
                        }
                    }
                }
                StartPoint::CiRun { run_id, .. } => match ctx.runs.get(run_id) {
                    None => missing.push(format!(
                        "no agent trace for run '{run_id}' — the run recorded no events"
                    )),
                    Some(ev) => {
                        a.entities.extend(ev.entities.iter().cloned());
                        for id in &ev.referenced_ids {
                            if let Some(r) = by_id.get(id.as_str()) {
                                a.entities.extend(r.entities.iter().cloned());
                                a.record_ids.insert(r.id.clone());
                                a.time = a.time.max(Some(r.created_at));
                            }
                        }
                        if let Some(ms) = ev.last_event_at
                            && let Some(t) = DateTime::from_timestamp_millis(ms)
                        {
                            a.time = a.time.max(Some(t));
                        }
                    }
                },
                StartPoint::LoggedWrite { node_id } => match by_id.get(node_id.as_str()) {
                    None => {
                        missing.push(format!("logged write '{node_id}' not found in this scope"))
                    }
                    Some(r) => {
                        a.entities.extend(r.entities.iter().cloned());
                        a.record_ids.insert(r.id.clone());
                        a.time = Some(r.created_at);
                    }
                },
                StartPoint::VersionRange { .. } => {
                    // restriction only; anchor pool contribution is the
                    // in-range overlap handled in pass B
                }
            }
            anchors.push(if a.entities.is_empty() && a.record_ids.is_empty() {
                None
            } else {
                Some(a)
            });
        }

        let failure_time = anchors
            .iter()
            .flatten()
            .filter_map(|a| a.time)
            .max()
            .unwrap_or(now);

        // ------------------------------------------------------------------
        // Pass B: seeds + rejections per mechanism.
        // ------------------------------------------------------------------
        let anchor_pool: HashSet<&str> = anchors
            .iter()
            .flatten()
            .flat_map(|a| a.entities.iter().map(|e| e.as_str()))
            .collect();
        let anchor_record_ids: HashSet<&str> = anchors
            .iter()
            .flatten()
            .flat_map(|a| a.record_ids.iter().map(|e| e.as_str()))
            .collect();

        let mut seeds: HashMap<&str, Seed> = HashMap::new();
        let mut rejections: Vec<WalkRejection> = Vec::new();

        for (idx, start) in starts.iter().enumerate() {
            match start {
                StartPoint::FailingTest { name, .. } => {
                    let Some(a) = &anchors[idx] else { continue };
                    if a.record_ids.is_empty() {
                        continue; // pass A already reported the dead handle
                    }
                    let name_anchors = normalized_anchors(name);
                    // test files = path-tier anchors of the records naming the test
                    let test_files: Vec<&String> = a
                        .entities
                        .iter()
                        .filter(|e| normalized_tier(e) == IdentifierTier::Path)
                        .collect();
                    if test_files.is_empty() {
                        missing.push(format!(
                            "records naming failing test '{name}' carry no file path — ingest the failing run's note or its test-file commits"
                        ));
                        continue;
                    }
                    for r in records {
                        if anchor_record_ids.contains(r.id.as_str()) {
                            continue; // the evidence, not a suspect
                        }
                        let touched: Vec<&String> = test_files
                            .iter()
                            .filter(|f| r.files.iter().any(|rf| path_matches(rf, f)))
                            .cloned()
                            .collect();
                        if r.is_commit {
                            if !touched.is_empty() {
                                let others: Vec<String> = r
                                    .files
                                    .iter()
                                    .filter(|f| !test_files.iter().any(|t| path_matches(f, t)))
                                    .cloned()
                                    .collect();
                                let hop_file = touched[0].clone();
                                let others_s = if others.is_empty() {
                                    "no other file".to_string()
                                } else {
                                    others.join(", ")
                                };
                                add_seed(
                                    &mut seeds,
                                    r,
                                    touched.iter().map(|f| (*f).clone()).collect(),
                                    PathHop {
                                        via: "failing_test/co_touch".into(),
                                        hop: format!("{hop_file} -> {others_s}"),
                                    },
                                    false,
                                );
                            }
                        } else if test_files.iter().any(|f| r.entities.contains(f))
                            || name_anchors.iter().any(|n| r.entities.contains(n))
                        {
                            rejections.push(WalkRejection {
                                id: r.id.clone(),
                                reason: format!(
                                    "report about the change, not a commit co-touching the test file(s) {}",
                                    test_files.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
                                ),
                                shared_anchors: r
                                    .entities
                                    .iter()
                                    .filter(|e| anchor_pool.contains(e.as_str()))
                                    .count(),
                            });
                        }
                    }
                }
                StartPoint::StackFrame { frame, .. } => {
                    let Some(a) = &anchors[idx] else { continue };
                    let file = frame.split(':').next().unwrap_or(frame).trim();
                    let file_anchor = &a.entities.iter().next().cloned().unwrap_or_default();
                    let touches = |r: &WalkRecord| {
                        r.files.iter().any(|f| path_matches(f, file))
                            || r.symbols.iter().any(|s| {
                                s.rsplit_once('/')
                                    .is_some_and(|(p, _)| path_matches(p, file))
                            })
                    };
                    let mut touchers: Vec<&WalkRecord> = records
                        .iter()
                        .filter(|r| r.is_commit && touches(r))
                        .collect();
                    touchers.sort_by_key(|r| r.created_at);
                    if touchers.is_empty() {
                        missing.push(format!(
                            "no commit record's hunk spans cover '{file}' — ingest git history (`vestige ingest-git <repo>`)"
                        ));
                        continue;
                    }
                    let Some(prime) = touchers
                        .iter()
                        .rev()
                        .find(|r| r.created_at <= failure_time)
                        .copied()
                    else {
                        missing.push(format!(
                            "every commit touching '{file}' postdates the failure anchor; nothing pre-failure to blame"
                        ));
                        continue;
                    };
                    add_seed(
                        &mut seeds,
                        prime,
                        vec![file_anchor.clone()],
                        PathHop {
                            via: "stack_frame/last_toucher".into(),
                            hop: frame.clone(),
                        },
                        false,
                    );
                    for earlier in &touchers {
                        if earlier.created_at < prime.created_at {
                            rejections.push(WalkRejection {
                                id: earlier.id.clone(),
                                reason: format!(
                                    "earlier toucher of {file}; the last pre-failure toucher is the prime suspect (SZZ-lite)"
                                ),
                                shared_anchors: earlier
                                    .entities
                                    .iter()
                                    .filter(|e| anchor_pool.contains(e.as_str()))
                                    .count(),
                            });
                        }
                    }
                }
                StartPoint::CiRun { run_id, .. } => {
                    let Some(a) = &anchors[idx] else { continue };
                    for r in records {
                        if anchor_record_ids.contains(r.id.as_str()) {
                            continue;
                        }
                        let shared: Vec<String> = r
                            .entities
                            .iter()
                            .filter(|e| a.entities.contains(*e))
                            .cloned()
                            .collect();
                        if shared.is_empty() {
                            continue;
                        }
                        let hop_anchor = shared[0].clone();
                        add_seed(
                            &mut seeds,
                            r,
                            shared,
                            PathHop {
                                via: "ci_run/failed_calls".into(),
                                hop: format!("run {run_id} via {hop_anchor}"),
                            },
                            false,
                        );
                    }
                }
                StartPoint::LoggedWrite { node_id } => {
                    let Some(a) = &anchors[idx] else { continue };
                    let neighbours: Option<&Vec<String>> = ctx.edges.get(node_id);
                    for r in records {
                        if r.id == *node_id || anchor_record_ids.contains(r.id.as_str()) {
                            continue;
                        }
                        let is_neighbour =
                            neighbours.is_some_and(|ns| ns.iter().any(|n| n == &r.id));
                        let shared: Vec<String> = r
                            .entities
                            .iter()
                            .filter(|e| a.entities.contains(*e))
                            .cloned()
                            .collect();
                        if is_neighbour {
                            add_seed(
                                &mut seeds,
                                r,
                                shared.clone(),
                                PathHop {
                                    via: "logged_write/edge".into(),
                                    hop: format!("edge {node_id} -> {}", r.id),
                                },
                                true,
                            );
                        }
                        if !shared.is_empty() {
                            let hop_anchor = shared[0].clone();
                            add_seed(
                                &mut seeds,
                                r,
                                shared,
                                PathHop {
                                    via: "logged_write/shared_anchor".into(),
                                    hop: hop_anchor,
                                },
                                false,
                            );
                        }
                    }
                }
                StartPoint::VersionRange {
                    worked_in,
                    broke_in,
                    ..
                } => {
                    match ctx.ranges.get(&idx) {
                        Some(Err(msg)) => missing.push(msg.clone()),
                        Some(Ok(range)) => {
                            // in-range commits overlapping the anchor pool get a
                            // rev-list hop (second hop at most: MAX_HOPS caps)
                            for r in records {
                                if !r.is_commit {
                                    continue;
                                }
                                let in_range =
                                    r.sha.as_ref().is_some_and(|s| range.shas.contains(s));
                                if !in_range {
                                    continue;
                                }
                                let overlapping: Vec<String> = r
                                    .entities
                                    .iter()
                                    .filter(|e| anchor_pool.contains(e.as_str()))
                                    .cloned()
                                    .collect();
                                if overlapping.is_empty() {
                                    continue; // in range, but no failure overlap to join on
                                }
                                let hop_anchor = overlapping[0].clone();
                                add_seed(
                                    &mut seeds,
                                    r,
                                    overlapping,
                                    PathHop {
                                        via: "version_range/rev_list".into(),
                                        hop: format!(
                                            "{worked_in}..{broke_in} ({}) via {hop_anchor}",
                                            range.shas.len()
                                        ),
                                    },
                                    false,
                                );
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        // version-range restriction over ALL commit seeds: applied in the
        // candidate loop below (out-of-range commits are rejected with a
        // why-not, never silently dropped)
        let active_ranges: Vec<&ResolvedRange> = ctx
            .ranges
            .values()
            .filter_map(|r| r.as_ref().ok())
            .collect();

        // ------------------------------------------------------------------
        // Window + restriction + ranking (ported from run_trail).
        // ------------------------------------------------------------------
        let mut candidates: Vec<(&WalkRecord, Seed)> = Vec::new();
        for (id, s) in &seeds {
            let Some(r) = by_id.get(id) else { continue };
            let age_days = (failure_time - r.created_at).num_seconds() as f64 / 86_400.0;
            let shared_n = s.anchors.len();
            if age_days <= 0.0 {
                rejections.push(WalkRejection {
                    id: r.id.clone(),
                    reason: "record is newer than the failure anchor".into(),
                    shared_anchors: shared_n,
                });
                continue;
            }
            if age_days > self.lookback_days as f64 {
                rejections.push(WalkRejection {
                    id: r.id.clone(),
                    reason: format!("outside the {}d lookback window", self.lookback_days),
                    shared_anchors: shared_n,
                });
                continue;
            }
            if shared_n == 0 && !s.via_edge {
                rejections.push(WalkRejection {
                    id: r.id.clone(),
                    reason: "shares no anchor with the start points".into(),
                    shared_anchors: 0,
                });
                continue;
            }
            if !active_ranges.is_empty() && r.is_commit {
                let in_any = r
                    .sha
                    .as_ref()
                    .is_some_and(|sha| active_ranges.iter().any(|rg| rg.shas.contains(sha)));
                if !in_any {
                    let (w, b) = (&active_ranges[0].worked_in, &active_ranges[0].broke_in);
                    rejections.push(WalkRejection {
                        id: r.id.clone(),
                        reason: format!("outside version range {w}..{b}"),
                        shared_anchors: shared_n,
                    });
                    continue;
                }
            }
            candidates.push((r, s.clone()));
        }

        // IDF over the in-window candidate pool (probabilistic IDF × tier
        // weight — ported verbatim in spirit from run_trail).
        let n = candidates.len();
        let mut df: HashMap<&str, usize> = HashMap::new();
        for (_, s) in &candidates {
            for a in &s.anchors {
                *df.entry(a.as_str()).or_default() += 1;
            }
        }
        let idf = |e: &str| -> f64 {
            let d = df.get(e).copied().unwrap_or(0) as f64;
            ((1.0 + n as f64) / (1.0 + d)).ln().max(0.0) * normalized_tier(e).weight()
        };

        let mut causes: Vec<CausalCause> = candidates
            .iter()
            .map(|(r, s)| {
                let age_days = (failure_time - r.created_at).num_seconds() as f64 / 86_400.0;
                let entity_term: f64 = s.anchors.iter().map(|a| idf(a)).sum();
                let recency_term = 0.3 * (1.0 / (1.0 + age_days / self.lookback_days as f64));
                let change_term = if r.is_commit {
                    CHANGE_RECORD_BONUS
                } else {
                    0.0
                };
                let mut anchors_sorted = s.anchors.iter().cloned().collect::<Vec<_>>();
                anchors_sorted.sort_by(|a, b| {
                    idf(b)
                        .partial_cmp(&idf(a))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                CausalCause {
                    id: r.id.clone(),
                    sha: r.sha.clone(),
                    score: entity_term + recency_term + change_term,
                    path: s.hops.iter().take(MAX_HOPS).cloned().collect(),
                    shared_anchors: anchors_sorted,
                    evidence_to: anchor_record_ids.iter().map(|s| s.to_string()).collect(),
                }
            })
            .collect();

        causes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                // near-ties: the change, not the newest report about it
                .then(
                    by_id
                        .get(b.id.as_str())
                        .map(|r| r.is_commit)
                        .unwrap_or(false)
                        .cmp(
                            &by_id
                                .get(a.id.as_str())
                                .map(|r| r.is_commit)
                                .unwrap_or(false),
                        ),
                )
                // older-first ties (the older change had more time to be the origin)
                .then(
                    by_id
                        .get(a.id.as_str())
                        .map(|r| r.created_at)
                        .unwrap_or(failure_time)
                        .cmp(
                            &by_id
                                .get(b.id.as_str())
                                .map(|r| r.created_at)
                                .unwrap_or(failure_time),
                        ),
                )
        });
        causes.truncate(self.max_causes);

        rejections.sort_by(|a, b| {
            b.shared_anchors
                .cmp(&a.shared_anchors)
                .then(a.reason.cmp(&b.reason))
        });
        rejections.truncate(self.max_rejections);

        let needs_report = if causes.is_empty() {
            let required: Vec<String> = if starts.is_empty() {
                REQUIRED_START_POINTS
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                let mut req: Vec<String> = starts
                    .iter()
                    .map(|s| {
                        format!(
                            "an anchor for {} (a record that resolves it in this scope)",
                            s.label()
                        )
                    })
                    .collect();
                if starts
                    .iter()
                    .all(|s| matches!(s, StartPoint::VersionRange { .. }))
                {
                    req = REQUIRED_START_POINTS[..4]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                }
                req.sort();
                req.dedup();
                req
            };
            if missing.is_empty() {
                missing.push(
                    "the start points anchored no candidate records in the lookback window".into(),
                );
            }
            Some(NeedsReport {
                missing,
                required_start_points: required,
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

/// The stdout lines of `git -C <repo> <args>`, or `None` when git fails.
pub fn git_lines(repo: &str, git_args: &[&str]) -> Option<Vec<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(git_args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    )
}

/// Assemble the record pool + out-of-band evidence from `storage`, then run
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

    // Wall clock feeds the non-hashed display path only (H6 applies to the
    // hashed walk state, which is caller-timestamped).
    let mut ctx = WalkContext {
        now: Utc::now(),
        ..WalkContext::default()
    };

    // A node_id carried by any start point becomes an explicit logged_write
    // start; every index-keyed step below uses this list.
    let starts = normalize_start_points(&req.start_points);

    // ci_run evidence: the run's failure channel
    for start in &starts {
        if let StartPoint::CiRun { run_id, .. } = start {
            let events = storage.get_trace(run_id).map_err(|e| e.to_string())?;
            let mut ev = RunEvidence::default();
            for event in &events {
                use crate::trace::MemoryTraceEvent as E;
                match event {
                    E::SanhedrinVeto {
                        claim,
                        evidence_ids,
                        ..
                    } => {
                        ev.entities.extend(extract_entities(claim, &[]));
                        ev.referenced_ids.extend(evidence_ids.iter().cloned());
                    }
                    E::MemoryRetrieve { ids, .. } => ev.referenced_ids.extend(ids.iter().cloned()),
                    E::MemoryWrite { id, .. } => ev.referenced_ids.push(id.clone()),
                    _ => {}
                }
                ev.last_event_at = Some(ev.last_event_at.unwrap_or(i64::MIN).max(event.at()));
            }
            ev.entities.sort();
            ev.entities.dedup();
            ev.referenced_ids.sort();
            ev.referenced_ids.dedup();
            ctx.runs.insert(run_id.clone(), ev);
        }
    }

    // logged_write edges: neighbours across persisted connections
    for start in &starts {
        if let StartPoint::LoggedWrite { node_id } = start {
            let conns = storage
                .get_connections_for_memory(node_id)
                .map_err(|e| e.to_string())?;
            let neighbours: Vec<String> = conns
                .iter()
                .filter(|c| c.link_type != "legacy_inferred")
                .map(|c| {
                    if c.source_id == *node_id {
                        c.target_id.clone()
                    } else {
                        c.source_id.clone()
                    }
                })
                .collect();
            ctx.edges.insert(node_id.clone(), neighbours);
        }
    }

    // version ranges: rev-list resolution per start point (index-keyed)
    let repos: HashSet<&str> = starts
        .iter()
        .filter_map(|s| match s {
            StartPoint::VersionRange { repo, .. } => Some(repo.as_str()),
            _ => None,
        })
        .collect();
    ctx.multiple_repos = repos.len() > 1;
    for (idx, start) in starts.iter().enumerate() {
        if let StartPoint::VersionRange {
            worked_in,
            broke_in,
            repo,
            ..
        } = start
        {
            let resolved = match git_lines(repo, &["rev-list", &format!("{worked_in}..{broke_in}")])
            {
                None => Err(format!(
                    "could not resolve the version range {worked_in}..{broke_in} in {repo} (tag missing or not fetched?)"
                )),
                Some(lines) if lines.is_empty() => Err(format!(
                    "version range {worked_in}..{broke_in} is empty — broke_in must come after worked_in"
                )),
                Some(lines) => Ok(ResolvedRange {
                    worked_in: worked_in.clone(),
                    broke_in: broke_in.clone(),
                    shas: git_records::parse_rev_list(&lines.join("\n")),
                }),
            };
            ctx.ranges.insert(idx, resolved);
        }
    }

    Ok(CausalWalkOptions {
        lookback_days: lookback,
        ..CausalWalkOptions::default()
    }
    .walk(&starts, &records, &ctx))
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
    fn failing_test_walk_finds_the_cotouch_commit() {
        let (storage, _dir) = store();
        // the failure note (test-file record): newest, names test + file
        let note = seed(
            &storage,
            "test_login_flow failed: auth flow crashed in tests/auth_test.rs",
            vec!["ci"],
            0,
        );
        // BAD: touched the test file AND src/auth.rs, 5 days ago
        let bad = commit_record(
            &storage,
            &sha_of('a'),
            "harden login",
            &["tests/auth_test.rs", "src/auth.rs"],
            &[],
            &[],
            5,
        );
        // GOOD: touched the test file 25 days ago (older, loses on recency)
        let good = commit_record(
            &storage,
            &sha_of('b'),
            "tune session cache",
            &["tests/auth_test.rs", "src/session.rs"],
            &[],
            &[],
            25,
        );
        // chatter: a non-commit report quoting the test file
        let chatter = seed(
            &storage,
            "CI summary: tests/auth_test.rs red on the auth job",
            vec![],
            1,
        );
        // unrelated commit: shares nothing, must be invisible
        commit_record(&storage, &sha_of('c'), "docs", &["README.md"], &[], &[], 2);

        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::FailingTest {
                name: "test_login_flow".into(),
                node_id: None,
            }]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        assert_eq!(result.causes.len(), 2, "{:?}", result.causes);
        let top = &result.causes[0];
        assert_eq!(top.id, bad.id);
        assert_eq!(top.sha.as_deref(), Some(sha_of('a').as_str()));
        assert_eq!(top.path[0].via, "failing_test/co_touch");
        // the hop names the test file AND the co-touched suspect file
        assert!(
            top.path[0].hop.contains("tests/auth_test.rs"),
            "{}",
            top.path[0].hop
        );
        assert!(
            top.path[0].hop.contains("src/auth.rs"),
            "{}",
            top.path[0].hop
        );
        assert!(
            top.shared_anchors
                .contains(&"tests/auth_test.rs".to_string())
        );
        // the runner-up is the older co-touch, not the chatter
        assert_eq!(result.causes[1].id, good.id);
        assert_eq!(result.causes[1].sha.as_deref(), Some(sha_of('b').as_str()));
        // chatter is a why-not, never a cause
        assert!(
            !result.causes.iter().any(|c| c.id == chatter.id),
            "a report quoting the anchor must never outrank the change"
        );
        assert!(
            result
                .rejected
                .iter()
                .any(|r| r.id == chatter.id && r.reason.contains("report about the change")),
            "{:?}",
            result.rejected
        );
        // the evidence trail points at the failure note
        assert!(top.evidence_to.contains(&note.id));
    }

    #[test]
    fn stack_frame_blames_the_last_pre_failure_toucher() {
        let (storage, _dir) = store();
        seed(&storage, "crash at src/auth.rs:88 during login", vec![], 0);
        let old = commit_record(
            &storage,
            &sha_of('c'),
            "introduce auth",
            &["src/auth.rs"],
            &[],
            &[],
            10,
        );
        let last = commit_record(
            &storage,
            &sha_of('d'),
            "tweak timeout",
            &["src/auth.rs"],
            &[],
            &[],
            3,
        );

        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::StackFrame {
                frame: "src/auth.rs:88".into(),
                node_id: None,
            }]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        assert_eq!(result.causes.len(), 1, "{:?}", result.causes);
        let top = &result.causes[0];
        assert_eq!(top.id, last.id);
        assert_eq!(top.sha.as_deref(), Some(sha_of('d').as_str()));
        assert_eq!(top.path[0].via, "stack_frame/last_toucher");
        assert_eq!(top.path[0].hop, "src/auth.rs:88");
        // the earlier toucher is answered in why-not, not surfaced
        assert!(
            result
                .rejected
                .iter()
                .any(|r| r.id == old.id && r.reason.contains("earlier toucher")),
            "{:?}",
            result.rejected
        );
    }

    #[test]
    fn version_range_restricts_candidates_to_the_rev_list() {
        // a real repo so rev-list produces a real sha for the in-range record
        let repo = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(repo.path())
                .args(["-c", "user.email=t@t", "-c", "user.name=t"])
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{:?}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).to_string()
        };
        git(&["init", "-q"]);
        git(&["commit", "--allow-empty", "-q", "-m", "one"]);
        git(&["tag", "w1"]);
        git(&["commit", "--allow-empty", "-q", "-m", "two"]);
        git(&["tag", "b1"]);
        let in_range_sha = git(&["rev-list", "w1..b1"]).trim().to_string();
        assert_eq!(in_range_sha.len(), 40);

        let (storage, _dir) = store();
        seed(
            &storage,
            "test_login_flow failed in tests/auth_test.rs",
            vec![],
            0,
        );
        // out-of-range but newer + co-touching: would win without the range
        commit_record(
            &storage,
            &sha_of('e'),
            "harden login",
            &["tests/auth_test.rs", "src/auth.rs"],
            &[],
            &[],
            5,
        );
        // in-range co-touch (the rev-list sha), older
        let in_range = commit_record(
            &storage,
            &in_range_sha,
            "tune session cache",
            &["tests/auth_test.rs", "src/session.rs"],
            &[],
            &[],
            8,
        );

        let result = walk_storage(
            &storage,
            &req(vec![
                StartPoint::FailingTest {
                    name: "test_login_flow".into(),
                    node_id: None,
                },
                StartPoint::VersionRange {
                    worked_in: "w1".into(),
                    broke_in: "b1".into(),
                    repo: repo.path().display().to_string(),
                    node_id: None,
                },
            ]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        // the recency winner is rejected by the range; the in-range commit is it
        assert!(
            result
                .rejected
                .iter()
                .any(|r| r.reason == "outside version range w1..b1"),
            "{:?}",
            result.rejected
        );
        assert_eq!(result.causes.len(), 1, "{:?}", result.causes);
        assert_eq!(result.causes[0].sha.as_deref(), Some(in_range_sha.as_str()));
        assert!(result.causes[0].id == in_range.id);
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
                node_id: None,
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
    fn ci_run_walks_trace_entities_to_records() {
        let (storage, _dir) = store();
        // a note the failing run retrieved (referenced evidence)
        let note = seed(
            &storage,
            "API_TIMEOUT was changed in the deploy env",
            vec!["API_TIMEOUT"],
            2,
        );
        // the quiet change carrying the same env var, older than the run
        let change = commit_record(
            &storage,
            &sha_of('7'),
            "tweak deploy",
            &["src/deploy.rs"],
            &[],
            &["API_TIMEOUT"],
            4,
        );

        let now_ms = Utc::now().timestamp_millis();
        use crate::trace::MemoryTraceEvent as E;
        storage
            .append_trace_event(&E::MemoryRetrieve {
                run_id: "run-1".into(),
                ids: vec![note.id.clone()],
                activation: [("x".to_string(), 1.0)].into_iter().collect(),
                at: now_ms,
            })
            .unwrap();
        storage
            .append_trace_event(&E::SanhedrinVeto {
                run_id: "run-1".into(),
                claim: "vetoed: claimed the API_TIMEOUT regression was benign".into(),
                evidence_ids: vec![],
                confidence: 0.8,
                at: now_ms,
            })
            .unwrap();

        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::CiRun {
                run_id: "run-1".into(),
                node_id: None,
            }]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        assert_eq!(result.causes.len(), 1, "{:?}", result.causes);
        let top = &result.causes[0];
        assert_eq!(top.id, change.id);
        assert_eq!(top.path[0].via, "ci_run/failed_calls");
        assert!(top.shared_anchors.contains(&"api_timeout".to_string()));
    }

    #[test]
    fn a_start_points_node_id_walks_as_an_extra_logged_write() {
        let (storage, _dir) = store();
        let note = seed(&storage, "crash at src/auth.rs:88 during login", vec![], 0);
        commit_record(
            &storage,
            &sha_of('c'),
            "introduce auth",
            &["src/auth.rs"],
            &[],
            &[],
            10,
        );
        commit_record(
            &storage,
            &sha_of('d'),
            "tweak timeout",
            &["src/auth.rs"],
            &[],
            &[],
            3,
        );

        let carried = walk_storage(
            &storage,
            &req(vec![StartPoint::StackFrame {
                frame: "src/auth.rs:88".into(),
                node_id: Some(note.id.clone()),
            }]),
        )
        .unwrap();
        let spelled_out = walk_storage(
            &storage,
            &req(vec![
                StartPoint::StackFrame {
                    frame: "src/auth.rs:88".into(),
                    node_id: None,
                },
                StartPoint::LoggedWrite {
                    node_id: note.id.clone(),
                },
            ]),
        )
        .unwrap();

        assert!(carried.needs_report.is_none(), "{:?}", carried.needs_report);
        assert_eq!(carried.causes[0].path[0].via, "stack_frame/last_toucher");
        assert_eq!(
            carried, spelled_out,
            "a node_id on a start point is exactly an extra logged_write start"
        );

        // and a node_id that is not in this scope is reported, not guessed around
        let unknown = walk_storage(
            &storage,
            &req(vec![StartPoint::FailingTest {
                name: "test_login_flow".into(),
                node_id: Some("mem-does-not-exist".into()),
            }]),
        )
        .unwrap();
        let report = unknown.needs_report.expect("unknown node must refuse");
        assert!(
            report
                .missing
                .iter()
                .any(|m| m.contains("mem-does-not-exist")),
            "{:?}",
            report.missing
        );
    }

    #[test]
    fn logged_write_walks_edges_and_shared_anchors() {
        let (storage, _dir) = store();
        // the quiet cause, older, connected to the logged write by an edge
        let cause = seed(
            &storage,
            "Set API_TIMEOUT=2 in the deploy env to speed up cold starts",
            vec!["API_TIMEOUT"],
            5,
        );
        // the logged write under investigation (the start point)
        let write = seed(
            &storage,
            "deploy job failed after the API_TIMEOUT change",
            vec![],
            3,
        );
        storage
            .save_connection(&crate::ConnectionRecord {
                source_id: write.id.clone(),
                target_id: cause.id.clone(),
                strength: 0.5,
                link_type: "reports".into(),
                created_at: Utc::now(),
                last_activated: Utc::now(),
                activation_count: 0,
            })
            .unwrap();

        let result = walk_storage(
            &storage,
            &req(vec![StartPoint::LoggedWrite {
                node_id: write.id.clone(),
            }]),
        )
        .unwrap();

        assert!(result.needs_report.is_none(), "{:?}", result.needs_report);
        assert_eq!(result.causes.len(), 1, "{:?}", result.causes);
        let top = &result.causes[0];
        assert_eq!(top.id, cause.id);
        // two hops max: the edge plus the shared anchor
        assert!(top.path.len() <= MAX_HOPS);
        let vias: Vec<&str> = top.path.iter().map(|h| h.via.as_str()).collect();
        assert!(vias.contains(&"logged_write/edge"), "{vias:?}");
        assert!(vias.contains(&"logged_write/shared_anchor"), "{vias:?}");
        assert!(top.shared_anchors.contains(&"api_timeout".to_string()));
        assert!(top.evidence_to.contains(&write.id));
    }

    #[test]
    fn promote_persists_evidence_of_edges_and_preview_does_not() {
        let (storage, _dir) = store();
        let note = seed(
            &storage,
            "test_login_flow failed in tests/auth_test.rs",
            vec![],
            0,
        );
        let bad = commit_record(
            &storage,
            &sha_of('a'),
            "harden login",
            &["tests/auth_test.rs", "src/auth.rs"],
            &[],
            &[],
            5,
        );

        let request = req(vec![StartPoint::FailingTest {
            name: "test_login_flow".into(),
            node_id: None,
        }]);
        // preview: nothing written
        let preview = walk_storage(&storage, &request).unwrap();
        assert!(preview.causes[0].id == bad.id);
        assert!(
            storage
                .get_connections_for_memory(&bad.id)
                .unwrap()
                .is_empty()
        );

        // promote: exactly one evidence_of edge cause -> note
        let written = persist_evidence_edges(&storage, &preview).unwrap();
        assert_eq!(written, vec![(bad.id.clone(), note.id.clone())]);
        let edges = storage.get_connections_for_memory(&bad.id).unwrap();
        assert!(edges.iter().any(|e| {
            e.source_id == bad.id && e.target_id == note.id && e.link_type == EVIDENCE_LINK_TYPE
        }));
        // idempotent: a second promote writes nothing new
        assert!(
            persist_evidence_edges(&storage, &preview)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn empty_or_unresolvable_ranges_refuse_rather_than_widen() {
        let (storage, _dir) = store();
        seed(
            &storage,
            "test_login_flow failed in tests/auth_test.rs",
            vec![],
            0,
        );
        // no repo at all: the range cannot resolve
        let result = walk_storage(
            &storage,
            &req(vec![
                StartPoint::FailingTest {
                    name: "test_login_flow".into(),
                    node_id: None,
                },
                StartPoint::VersionRange {
                    worked_in: "w9".into(),
                    broke_in: "b9".into(),
                    repo: "/nonexistent/repo".into(),
                    node_id: None,
                },
            ]),
        )
        .unwrap();
        let report = result.needs_report.expect("unresolvable range must refuse");
        assert!(
            report
                .missing
                .iter()
                .any(|m| m.contains("could not resolve")),
            "{:?}",
            report.missing
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

// ============================================================================
// START POINT node_id — pure tests, no store, run on every feature set
// ============================================================================

#[cfg(test)]
mod start_point_node_id_tests {
    use super::*;
    use serde_json::json;

    fn logged_write(id: &str) -> StartPoint {
        StartPoint::LoggedWrite {
            node_id: id.to_string(),
        }
    }

    fn stack_frame(frame: &str, node_id: Option<&str>) -> StartPoint {
        StartPoint::StackFrame {
            frame: frame.to_string(),
            node_id: node_id.map(str::to_string),
        }
    }

    #[test]
    fn every_variant_accepts_node_id_and_round_trips() {
        let cases = [
            (
                json!({"kind": "failing_test", "name": "t", "node_id": "mem-1"}),
                "failing_test",
            ),
            (
                json!({"kind": "stack_frame", "frame": "src/a.rs:1", "node_id": "mem-1"}),
                "stack_frame",
            ),
            (
                json!({"kind": "ci_run", "run_id": "r1", "node_id": "mem-1"}),
                "ci_run",
            ),
            (
                json!({"kind": "logged_write", "node_id": "mem-1"}),
                "logged_write",
            ),
            (
                json!({"kind": "version_range", "worked_in": "a", "broke_in": "b", "repo": "/r", "node_id": "mem-1"}),
                "version_range",
            ),
        ];
        for (wire, kind) in cases {
            let point: StartPoint =
                serde_json::from_value(wire.clone()).unwrap_or_else(|err| panic!("{kind}: {err}"));
            assert_eq!(point.kind(), kind);
            assert_eq!(point.record_id(), Some("mem-1"), "{kind}");
            assert_eq!(
                serde_json::to_value(&point).unwrap(),
                wire,
                "{kind} must round-trip with its node_id"
            );
        }
    }

    #[test]
    fn node_id_is_optional_off_logged_write_and_omitted_when_absent() {
        let point: StartPoint =
            serde_json::from_value(json!({"kind": "stack_frame", "frame": "src/a.rs:1"})).unwrap();
        assert_eq!(point.record_id(), None);
        assert_eq!(
            serde_json::to_value(&point).unwrap(),
            json!({"kind": "stack_frame", "frame": "src/a.rs:1"}),
            "an absent node_id is not serialized as null"
        );
        assert!(
            serde_json::from_value::<StartPoint>(json!({"kind": "logged_write"})).is_err(),
            "logged_write is the variant whose only content is node_id"
        );
    }

    #[test]
    fn typos_are_still_rejected() {
        let err = serde_json::from_value::<StartPoint>(
            json!({"kind": "stack_frame", "frame": "src/a.rs:1", "nodeid": "mem-1"}),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown field `nodeid`"), "{err}");
        assert!(
            err.contains("node_id"),
            "the message names the real field: {err}"
        );
    }

    #[test]
    fn blank_node_id_names_no_record() {
        assert_eq!(stack_frame("f", Some("   ")).record_id(), None);
        assert_eq!(stack_frame("f", Some("")).record_id(), None);
        assert_eq!(
            stack_frame("f", Some("  mem-9 ")).record_id(),
            Some("mem-9")
        );
    }

    #[test]
    fn normalize_appends_logged_writes_and_keeps_original_positions() {
        let starts = vec![
            stack_frame("src/a.rs:1", Some("mem-1")),
            StartPoint::VersionRange {
                worked_in: "v1".into(),
                broke_in: "v2".into(),
                repo: "/r".into(),
                node_id: Some("mem-2".into()),
            },
            stack_frame("src/b.rs:2", None),
        ];
        let out = normalize_start_points(&starts);
        assert_eq!(
            out,
            vec![
                stack_frame("src/a.rs:1", None),
                StartPoint::VersionRange {
                    worked_in: "v1".into(),
                    broke_in: "v2".into(),
                    repo: "/r".into(),
                    node_id: None,
                },
                stack_frame("src/b.rs:2", None),
                logged_write("mem-1"),
                logged_write("mem-2"),
            ],
            "originals keep their index (resolved version ranges are index-keyed)"
        );
        assert_eq!(normalize_start_points(&out), out, "idempotent");
    }

    #[test]
    fn normalize_never_repeats_a_node_an_explicit_logged_write_names() {
        // the explicit logged_write comes AFTER the start point that carries it
        let starts = vec![
            stack_frame("src/a.rs:1", Some("mem-1")),
            stack_frame("src/b.rs:2", Some("mem-1")),
            logged_write("mem-1"),
        ];
        let out = normalize_start_points(&starts);
        assert_eq!(
            out.iter()
                .filter(|s| matches!(s, StartPoint::LoggedWrite { .. }))
                .count(),
            1,
            "{out:?}"
        );
        // and two carriers of the same node produce one synthesized start
        let out = normalize_start_points(&[
            stack_frame("src/a.rs:1", Some("mem-7")),
            stack_frame("src/b.rs:2", Some(" mem-7 ")),
        ]);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[2], logged_write("mem-7"));
    }

    #[test]
    fn normalize_leaves_canonical_input_alone() {
        let starts = vec![logged_write("mem-1"), stack_frame("src/a.rs:1", None)];
        assert_eq!(normalize_start_points(&starts), starts);
        assert!(normalize_start_points(&[]).is_empty());
    }
}
