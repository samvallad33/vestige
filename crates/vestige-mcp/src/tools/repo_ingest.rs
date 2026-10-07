//! `codebase action=ingest_repo`: the repo connector.
//!
//! Each commit of a local git checkout becomes one change record: an `event`
//! memory whose content is the commit's touched files, modules, hunk-header
//! symbols and changed-line identifiers (the same record format the legacy
//! `ingest-git` wrote, from `git_records::record_content`). What makes it a
//! record the log can reason about:
//!
//! * **Exact handles.** Tags `git-commit`, `codebase:<name>` and
//!   `commit:<sha>`, so a commit is found by `recall(handle="commit:<sha>")`
//!   and nothing is found by text.
//! * **Provenance.** The source envelope is `(git, <codebase>, <sha>)`, which
//!   is what `memory_status view=provenance` reports and what makes a re-run
//!   idempotent: a commit already in the scope is skipped, not duplicated.
//! * **Valid time.** `valid_from` is the commit's author time (clamped to now,
//!   so a future-dated commit is not hidden as "not yet valid"). The log's own
//!   write time stays the real write time.
//! * **Anchors.** Every hunk-header symbol the commit touched, and every
//!   touched file, that exists in the checkout is anchored with the same schema
//!   `remember_decision` uses, so `codebase action=verify` re-checks the change
//!   record against the working tree and reports drift for as long as it runs.
//!   A record's status is only "fresh" when *all* its anchors are, and a
//!   path-only anchor (or a symbol that no longer exists) can only ever read
//!   "unverifiable". So when a commit has any verifiable anchor, only the
//!   verifiable ones are recorded and the rest are counted, not recorded; a
//!   commit with none keeps its path-only anchors, so a deleted file still shows.
//!
//! `repoPath` must be the top of a working tree. git reports history paths
//! relative to that top while anchors resolve against `repoPath`, so a
//! subdirectory would anchor the wrong files; it is refused with the path to
//! pass instead, and so is a bare repository, which has no files to anchor.
//!
//! Git is run local-only: lazy fetching from a partial clone's remote is turned
//! off, so reading history never reaches the network. The checkout may be
//! untrusted, so settings in its own config that would make a read run a
//! program (a signature check's `gpg.program`, an fsmonitor hook, an external
//! diff or textconv) are overridden on the command line. If git dies partway
//! (a blobless clone whose older blobs are not on disk), the commits read
//! before the failure are kept and the stop is reported.
//!
//! Re-running is always safe and always continues: a commit already recorded is
//! skipped, and one whose anchors never landed (an interrupted run, a failed
//! anchor write) gets them on the next run. A write phase is bounded in time;
//! hitting the bound stops cleanly and says how much is left. Two writes of the
//! same codebase into the same scope would both read the recorded set before
//! either wrote, so a second one is refused while the first runs; previews are
//! never refused.
//!
//! Preview is the default. The log is append-only: a bulk write cannot be
//! undone, so the first call says what it would write and nothing else. The
//! records go to their own scope (the codebase name unless one is given), so a
//! third-party repository never lands in the shared `user` scope. Merge
//! commits are skipped: they carry no diff to anchor.
//!
//! **Exact path edges.** After the records land, two commit records in this
//! scope get one `touched` edge when they changed the same repo-relative file
//! path. The path is the exact string git recorded for the commit (the `b/`
//! side, capped the same way the commit record is). It is not split into
//! directory segments, not lowercased, and not taken from content tokens.
//! A path touched by more commits than `max(25, 5%)` of the commits in the
//! scope is a hub: it produces no edges and is listed in `skippedHubPaths`.
//! Every other path is a chain — each commit links only to the nearest earlier
//! and later commit that touched that path — so the edge count stays linear.
//! A later page reads the files of commits already in the scope and extends
//! the chain. A re-run writes an edge only when that pair does not already
//! have it.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use vestige_core::advanced::git_records::{self, GitCommit};
use vestige_core::codebase::{AnchorDraft, AnchorStatus, CodeAnchor, capture_anchor};
use vestige_core::{ConnectionRecord, IngestInput, SourceEnvelope, Storage, StorageError};

/// Commits read per call unless the caller says otherwise.
pub const DEFAULT_LIMIT: usize = 100;
/// Hard cap per call; page deeper with `rev` or `until`. Every record is one
/// durable log write (about 80 ms each), so a page stays well inside a client's
/// request timeout.
pub const MAX_LIMIT: usize = 500;
/// A write phase that runs past this stops cleanly; a rerun continues.
const WRITE_BUDGET: Duration = Duration::from_secs(45);
/// Anchors queued before one log write records them all.
const ANCHOR_BATCH: usize = 200;
const GIT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_GIT_OUTPUT: u64 = 256 * 1024 * 1024;
/// Anchors kept per commit; the overflow is counted, not silently dropped.
const MAX_ANCHORS_PER_COMMIT: usize = 24;
/// Files kept per commit by `git_records` (`MAX_FILES`). A later page must
/// not invent paths the commit record itself dropped.
const RECORDED_FILES: usize = 50;
/// A path touched by more commits than `max(HUB_FLOOR, 5% of the scope)` is
/// a hub and records no pairwise `touched` edges.
const HUB_FLOOR: usize = 25;
const HUB_SHARE_NUM: usize = 5;
const HUB_SHARE_DEN: usize = 100;
const SAMPLE_COMMITS: usize = 5;
const IDS_SHOWN: usize = 20;
/// Longest value accepted for a ref, date or name handed to git or the log.
const MAX_ARG_LEN: usize = 200;

/// What to ingest and where it goes.
pub struct Request {
    pub repo_path: PathBuf,
    pub codebase: Option<String>,
    pub scope: Option<String>,
    pub rev: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
    pub limit: Option<usize>,
    pub dry_run: bool,
    /// Write-phase time budget; `None` is [`WRITE_BUDGET`]. Tests shrink it.
    pub budget: Option<Duration>,
}

/// Ingest (or preview) the commits of `req.repo_path`. Blocking git and log
/// work runs off the async executor.
pub async fn execute(storage: &Arc<Storage>, req: Request) -> Result<Value, String> {
    let storage = Arc::clone(storage);
    tokio::task::spawn_blocking(move || run(&storage, req))
        .await
        .map_err(|e| format!("ingest_repo task failed: {e}"))?
}

fn run(storage: &Arc<Storage>, req: Request) -> Result<Value, String> {
    let root = std::fs::canonicalize(&req.repo_path).map_err(|_| {
        format!(
            "repoPath '{}' is not an available directory",
            req.repo_path.display()
        )
    })?;
    if !root.is_dir() {
        return Err("repoPath must be a directory".into());
    }

    let codebase = match trimmed(req.codebase.as_deref()) {
        Some(name) => name.to_string(),
        None => root
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .filter(|name| !name.is_empty())
            .ok_or("could not derive a codebase name from repoPath; pass codebase")?,
    };
    check_label("codebase", &codebase)?;
    let scope = trimmed(req.scope.as_deref())
        .map(str::to_string)
        .unwrap_or_else(|| codebase.clone());
    check_label("scope", &scope)?;
    let limit = req.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    for (name, value) in [
        ("rev", &req.rev),
        ("since", &req.since),
        ("until", &req.until),
    ] {
        if let Some(value) = value {
            check_git_arg(name, value)?;
        }
    }
    check_top_of_work_tree(&root)?;
    // Held until this call returns. A preview writes nothing, so it is never
    // refused for a write in flight.
    let _claim = if req.dry_run {
        None
    } else {
        Some(claim_write(storage, &scope, &codebase)?)
    };

    let (commits, stopped_early) = read_commits(&root, &req, limit)?;
    let head = run_git(&root, &["rev-parse".into(), "HEAD".into()])
        .ok()
        .filter(|run| run.failure.is_none())
        .map(|run| String::from_utf8_lossy(&run.stdout).trim().to_string())
        .filter(|sha| !sha.is_empty());

    let ingested = ingested_commits(storage, &scope, &codebase)?;
    let mut sha_to_id = ingested.clone();
    // git lists newest first; write oldest first so memory ids follow history.
    let fresh: Vec<&GitCommit> = commits
        .iter()
        .rev()
        .filter(|commit| !ingested.contains_key(&commit.sha))
        .collect();
    let already = commits.len() - fresh.len();

    // Commits recorded earlier whose anchors never landed: an interrupted run,
    // or an anchor write that failed. They get their anchors now.
    let recorded: Vec<(&GitCommit, &String)> = commits
        .iter()
        .filter_map(|commit| ingested.get(&commit.sha).map(|id| (commit, id)))
        .collect();
    let anchored = if recorded.is_empty() {
        HashMap::new()
    } else {
        let ids: Vec<String> = recorded.iter().map(|(_, id)| (*id).clone()).collect();
        storage
            .code_anchors_for_nodes(&ids)
            .map_err(|e| format!("could not read the anchors already recorded: {e}"))?
    };
    let to_repair: Vec<(&GitCommit, &String)> = recorded
        .into_iter()
        .filter(|(_, id)| !anchored.contains_key(id.as_str()))
        .collect();

    let now = Utc::now();
    let started = Instant::now();
    let budget = req.budget.unwrap_or(WRITE_BUDGET);
    let mut stats = Stats::default();
    let mut node_ids: Vec<String> = Vec::new();
    let mut failures = Failures::default();
    let mut queue: Vec<CodeAnchor> = Vec::new();

    for commit in &fresh {
        if !req.dry_run && started.elapsed() >= budget {
            stats.stopped_by_budget = true;
            break;
        }
        if commit.time > now {
            stats.future_dates_clamped += 1;
        }
        // Preview and write take the same path, so the preview's anchor
        // numbers are the numbers a write would produce.
        let captured = capture_anchors(commit, &root, "");
        stats.note(&captured);
        if req.dry_run {
            stats.anchors.add(&captured.anchors);
            continue;
        }

        let input = change_record(commit, &codebase, now);
        let node = match storage.ingest_in_scope(input, &scope) {
            Ok(node) => node,
            Err(StorageError::SecretDetected { .. }) => {
                stats.skipped_secret += 1;
                continue;
            }
            Err(e) => {
                failures.on_commit_refused(format!(
                    "stopped at commit {} ({}): {e}",
                    short(&commit.sha),
                    commit.subject
                ));
                break;
            }
        };
        stats.created += 1;
        sha_to_id.insert(commit.sha.clone(), node.id.clone());
        if node_ids.len() < IDS_SHOWN {
            node_ids.push(node.id.clone());
        }
        queue.extend(bind(captured.anchors, &node.id));
        if queue.len() >= ANCHOR_BATCH {
            flush_anchors(storage, &mut queue, &mut stats, &mut failures);
        }
    }

    for (commit, node_id) in &to_repair {
        if !req.dry_run && started.elapsed() >= budget {
            stats.stopped_by_budget = true;
            break;
        }
        let captured = capture_anchors(commit, &root, node_id);
        if captured.anchors.is_empty() {
            continue;
        }
        stats.anchors_repaired_commits += 1;
        if req.dry_run {
            stats.anchors.add(&captured.anchors);
            continue;
        }
        queue.extend(bind(captured.anchors, node_id));
        if queue.len() >= ANCHOR_BATCH {
            flush_anchors(storage, &mut queue, &mut stats, &mut failures);
        }
    }
    flush_anchors(storage, &mut queue, &mut stats, &mut failures);
    let path_edges = record_exact_path_edges(
        storage,
        &root,
        &commits,
        &sha_to_id,
        req.dry_run,
        started,
        budget,
        &mut stats,
    );
    let remaining = if req.dry_run {
        0
    } else {
        fresh.len() - stats.created - stats.skipped_secret
    };

    let oldest = commits.last().map(|commit| commit.sha.clone());
    let more = commits.len() >= limit && stopped_early.is_none();
    let sample: Vec<Value> = fresh
        .iter()
        .rev()
        .take(SAMPLE_COMMITS)
        .map(|commit| {
            json!({
                "sha": short(&commit.sha),
                "date": commit.time.to_rfc3339(),
                "subject": commit.subject,
                "files": commit.files.len(),
                "symbols": commit.symbols.len(),
            })
        })
        .collect();

    let anchors = json!({
        (if req.dry_run { "wouldRecord" } else { "recorded" }): stats.anchors.total,
        "verifiable": stats.anchors.verifiable,
        "unverifiableKept": stats.anchors.total - stats.anchors.verifiable,
        "droppedUnverifiable": stats.dropped_unverifiable,
        "pathsNotInCheckout": stats.paths_not_in_checkout,
        "pathsOutsideCheckout": stats.paths_outside_checkout,
        "commitsWithTruncatedAnchors": stats.commits_with_truncated_anchors,
        "commitsGivenMissingAnchors": stats.anchors_repaired_commits,
        "anchorWriteErrors": stats.anchor_errors,
    });

    let next_step = suggest_next_step(&Outcome {
        ending: Ending::of(req.dry_run, stats.stopped_by_budget, &failures),
        to_write: fresh.len(),
        remaining,
        skipped_secret: stats.skipped_secret,
        budget,
        codebase: &codebase,
        scope: &scope,
        root: &root,
    });

    Ok(json!({
        "action": "ingest_repo",
        "dryRun": req.dry_run,
        "codebase": codebase,
        "scope": scope,
        "limit": limit,
        "repo": {
            "path": root.display().to_string(),
            "head": head,
            "rev": req.rev.as_deref().unwrap_or("HEAD"),
            "since": req.since,
            "until": req.until,
            "mergeCommits": "skipped",
        },
        "commits": {
            "seen": commits.len(),
            "alreadyIngested": already,
            "new": fresh.len(),
            "created": stats.created,
            "remaining": remaining,
            "stoppedByBudget": stats.stopped_by_budget,
            "skippedSecret": stats.skipped_secret,
            "futureDatesClamped": stats.future_dates_clamped,
        },
        "edges": {
            (if req.dry_run { "wouldRecord" } else { "recorded" }): path_edges.touched,
            "touched": path_edges.touched,
            "commitsInScope": path_edges.commits_in_scope,
            "hubBound": path_edges.hub_bound,
            "skippedHubPaths": path_edges.skipped_hubs,
        },
        "anchors": anchors,
        "sample": sample,
        "nodeIds": node_ids,
        "nodeIdsTruncated": stats.created > IDS_SHOWN,
        "gitStoppedEarly": stopped_early.as_ref().map(|reason| json!({
            "commitsKept": commits.len(),
            "reason": reason,
            "hint": PARTIAL_CLONE_HINT,
        })),
        "more": more,
        "pageBackWith": if more { oldest.map(|sha| json!({"rev": format!("{sha}~1")})) } else { None },
        "partial": failures.first.is_some(),
        "error": failures.first,
        "nextStep": next_step,
    }))
}

/// Write ingests running in this process, keyed by store, scope and codebase.
/// Every agent on a machine shares one server process, and two writes for the
/// same key would both read the recorded set before either writes, then both
/// write every commit.
static ACTIVE_WRITES: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// One write ingest's hold on its key. Dropping it, on any exit path, frees
/// the key.
struct WriteClaim {
    key: String,
}

impl Drop for WriteClaim {
    fn drop(&mut self) {
        ACTIVE_WRITES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.key);
    }
}

/// Claim the right to write `codebase` into `scope` of this store, or refuse
/// while another write holds it. Labels carry no control characters, so NUL
/// separates the key's parts unambiguously.
fn claim_write(storage: &Arc<Storage>, scope: &str, codebase: &str) -> Result<WriteClaim, String> {
    let key = format!(
        "{}\u{0}{scope}\u{0}{codebase}",
        storage.data_dir().display()
    );
    let mut active = ACTIVE_WRITES.lock().unwrap_or_else(PoisonError::into_inner);
    if !active.insert(key.clone()) {
        return Err(format!(
            "an ingest of codebase '{codebase}' into scope '{scope}' is already running; run again when it finishes (recorded commits are skipped)"
        ));
    }
    Ok(WriteClaim { key })
}

/// What went wrong while writing. A commit the log refused stops the run; a
/// failed anchor write does not (its commits stay recorded, only their anchors
/// are missing), so the two lead to different next steps.
#[derive(Default)]
struct Failures {
    /// The first failure message, whichever kind came first.
    first: Option<String>,
    /// The log refused a commit, and the run stopped there.
    commit_refused: bool,
    /// An anchor write failed; its commits are recorded without anchors.
    anchors_failed: bool,
}

impl Failures {
    fn on_commit_refused(&mut self, message: String) {
        self.commit_refused = true;
        self.first.get_or_insert(message);
    }

    fn on_anchors_failed(&mut self, message: String) {
        self.anchors_failed = true;
        self.first.get_or_insert(message);
    }
}

/// How a call ended, which decides the next step it suggests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    /// Nothing was written.
    Preview,
    /// The write budget ran out with commits left to write.
    Budget,
    /// The log refused a commit, so the run stopped there.
    RefusedCommit,
    /// Every commit was written, but some anchor writes failed.
    AnchorsMissing,
    /// Everything was written.
    Complete,
}

impl Ending {
    fn of(dry_run: bool, stopped_by_budget: bool, failures: &Failures) -> Self {
        if dry_run {
            Ending::Preview
        } else if stopped_by_budget {
            Ending::Budget
        } else if failures.commit_refused {
            Ending::RefusedCommit
        } else if failures.anchors_failed {
            Ending::AnchorsMissing
        } else {
            Ending::Complete
        }
    }
}

/// The facts a next step names.
struct Outcome<'a> {
    ending: Ending,
    /// New commits a preview would write.
    to_write: usize,
    /// Commits a budget stop left unwritten.
    remaining: usize,
    skipped_secret: usize,
    budget: Duration,
    codebase: &'a str,
    scope: &'a str,
    /// The canonical checkout, so the suggested `verify` call runs as written.
    root: &'a Path,
}

/// The `nextStep` of an ingest response.
fn suggest_next_step(outcome: &Outcome) -> String {
    let Outcome {
        codebase, scope, ..
    } = outcome;
    match outcome.ending {
        Ending::Preview => format!(
            "Preview only: nothing was written. Call again with dryRun=false to record {} change records in scope '{scope}'. The log is append-only, so a bulk write cannot be undone.",
            outcome.to_write
        ),
        Ending::Budget => format!(
            "Stopped at the {}s write budget with {} commits left to write. Run the same call again to continue: the commits already recorded are skipped.",
            outcome.budget.as_secs(),
            outcome.remaining
        ),
        Ending::RefusedCommit => "The ingest stopped early. Nothing is lost: re-run with the same arguments and the commits already recorded are skipped.".to_string(),
        Ending::AnchorsMissing => {
            let recorded = match outcome.skipped_secret {
                0 => "Every commit was recorded".to_string(),
                refused => {
                    format!("Every commit except the {refused} the secret gate refused was recorded")
                }
            };
            format!(
                "{recorded}, but some of their anchors were not (see error). Run the same call again to add them: recorded commits are skipped and get their missing anchors."
            )
        }
        Ending::Complete => format!(
            "Check them against the checkout with codebase action='verify' codebase='{codebase}' scope='{scope}' repoPath='{}'. Find one commit with recall handle='commit:<sha>'.",
            outcome.root.display()
        ),
    }
}

#[derive(Default)]
struct Stats {
    created: usize,
    skipped_secret: usize,
    future_dates_clamped: usize,
    paths_not_in_checkout: usize,
    paths_outside_checkout: usize,
    commits_with_truncated_anchors: usize,
    dropped_unverifiable: usize,
    anchor_errors: usize,
    anchors_repaired_commits: usize,
    stopped_by_budget: bool,
    anchors: AnchorCounts,
}

impl Stats {
    fn note(&mut self, captured: &Captured) {
        self.paths_not_in_checkout += captured.not_in_checkout;
        self.paths_outside_checkout += captured.outside_checkout;
        self.dropped_unverifiable += captured.dropped_unverifiable;
        if captured.truncated {
            self.commits_with_truncated_anchors += 1;
        }
    }
}

#[derive(Default)]
struct AnchorCounts {
    total: usize,
    verifiable: usize,
}

impl AnchorCounts {
    fn add(&mut self, anchors: &[CodeAnchor]) {
        self.total += anchors.len();
        self.verifiable += anchors.iter().filter(|a| a.is_verifiable()).count();
    }
}

/// The change record for one commit.
fn change_record(commit: &GitCommit, codebase: &str, now: chrono::DateTime<Utc>) -> IngestInput {
    // non_exhaustive: default-then-mutate is the only cross-crate construction
    let mut envelope = SourceEnvelope::default();
    envelope.source_system = Some(git_records::SOURCE_SYSTEM.to_string());
    envelope.source_id = Some(commit.sha.clone());
    envelope.source_project = Some(codebase.to_string());
    envelope.source_type = Some("commit".to_string());
    envelope.source_updated_at = Some(commit.time);
    IngestInput {
        content: git_records::record_content(commit),
        node_type: "event".to_string(),
        source: Some(git_records::SOURCE_SYSTEM.to_string()),
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: vec![
            git_records::COMMIT_TAG.to_string(),
            format!("codebase:{codebase}"),
            commit_tag(&commit.sha),
        ],
        // A future author date would make the record "not yet valid" and hide it.
        valid_from: Some(commit.time.min(now)),
        valid_until: None,
        validity_inferred: false,
        source_envelope: Some(envelope),
    }
}

/// The exact tag that finds one commit record.
pub fn commit_tag(sha: &str) -> String {
    format!("commit:{sha}")
}

/// Commits already recorded in `scope` for `codebase`, by exact `commit:` tag:
/// sha to the id of its change record.
fn ingested_commits(
    storage: &Arc<Storage>,
    scope: &str,
    codebase: &str,
) -> Result<HashMap<String, String>, String> {
    let tag = format!("codebase:{codebase}");
    let nodes = storage
        .current_code_context_nodes("event", Some(&tag), scope, i32::MAX)
        .map_err(|e| format!("could not list the commits already recorded: {e}"))?;
    let mut recorded = HashMap::new();
    for node in &nodes {
        for sha in node
            .tags
            .iter()
            .filter_map(|tag| tag.strip_prefix("commit:"))
        {
            recorded.insert(sha.to_string(), node.id.clone());
        }
    }
    Ok(recorded)
}

struct PathEdgeReport {
    touched: usize,
    commits_in_scope: usize,
    hub_bound: usize,
    skipped_hubs: Vec<String>,
}

struct Touch {
    sha: String,
    id: Option<String>,
    time: DateTime<Utc>,
    files: Vec<String>,
}

/// `max(25, 5% of commits in scope)`, integer percent, rounded down.
fn hub_bound(commits_in_scope: usize) -> usize {
    let share = commits_in_scope.saturating_mul(HUB_SHARE_NUM) / HUB_SHARE_DEN;
    share.max(HUB_FLOOR)
}

fn path_is_hub(touchers: usize, commits_in_scope: usize) -> bool {
    touchers > hub_bound(commits_in_scope)
}

/// `touched` edges between commit records that changed one exact path.
///
/// Paths come from the commit's recorded file list, never from directory
/// segments or from [`crate::auto_connect::extract_entities`].
fn record_exact_path_edges(
    storage: &Arc<Storage>,
    root: &Path,
    commits: &[GitCommit],
    sha_to_id: &HashMap<String, String>,
    dry_run: bool,
    started: Instant,
    budget: Duration,
    stats: &mut Stats,
) -> PathEdgeReport {
    let page: HashSet<&str> = commits.iter().map(|commit| commit.sha.as_str()).collect();
    let mut scope_shas: HashSet<&str> = sha_to_id.keys().map(String::as_str).collect();
    if dry_run {
        for commit in commits {
            scope_shas.insert(commit.sha.as_str());
        }
    }
    let commits_in_scope = scope_shas.len();
    let bound = hub_bound(commits_in_scope);

    let mut touches: Vec<Touch> = Vec::new();
    for commit in commits {
        let id = sha_to_id.get(&commit.sha).cloned();
        if !dry_run && id.is_none() {
            continue;
        }
        touches.push(Touch {
            sha: commit.sha.clone(),
            id,
            time: commit.time,
            files: commit.files.clone(),
        });
    }
    let missing: Vec<String> = sha_to_id
        .keys()
        .filter(|sha| !page.contains(sha.as_str()))
        .cloned()
        .collect();
    let prior = read_recorded_files(root, &missing);
    for sha in &missing {
        let Some((time, files)) = prior.get(sha) else {
            continue;
        };
        touches.push(Touch {
            sha: sha.clone(),
            id: sha_to_id.get(sha).cloned(),
            time: *time,
            files: files.clone(),
        });
    }

    let mut by_path: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, touch) in touches.iter().enumerate() {
        let mut seen = HashSet::new();
        for file in &touch.files {
            if file.is_empty() || file.contains('\n') || file.contains('\u{1e}') {
                continue;
            }
            if !seen.insert(file.as_str()) {
                continue;
            }
            by_path.entry(file.clone()).or_default().push(index);
        }
    }

    let mut paths: Vec<String> = by_path.keys().cloned().collect();
    paths.sort();
    let mut skipped_hubs = Vec::new();
    let mut planned: Vec<(usize, usize)> = Vec::new();
    for path in &paths {
        let mut order = by_path[path].clone();
        order.sort_unstable();
        order.dedup();
        if path_is_hub(order.len(), commits_in_scope) {
            skipped_hubs.push(path.clone());
            continue;
        }
        order.sort_by(|&left, &right| {
            touches[left]
                .time
                .cmp(&touches[right].time)
                .then_with(|| touches[left].sha.cmp(&touches[right].sha))
        });
        order.dedup();
        for pair in order.windows(2) {
            planned.push((pair[0], pair[1]));
        }
    }

    let mut existing: HashSet<(String, String)> = storage
        .get_all_connections()
        .unwrap_or_default()
        .into_iter()
        .filter(|edge| edge.link_type == "touched")
        .map(|edge| (edge.source_id, edge.target_id))
        .collect();
    let mut touched = 0usize;
    for (earlier, later) in planned {
        if !dry_run && started.elapsed() >= budget {
            stats.stopped_by_budget = true;
            break;
        }
        let (Some(source), Some(target)) =
            (touches[earlier].id.as_deref(), touches[later].id.as_deref())
        else {
            // Preview of a commit that has no record yet. Count the pair by
            // sha so two shared paths do not double it. A write never counts
            // an edge it did not store.
            if dry_run {
                let dedup = (touches[earlier].sha.clone(), touches[later].sha.clone());
                if existing.insert(dedup) {
                    touched += 1;
                }
            }
            continue;
        };
        if !existing.insert((source.to_string(), target.to_string())) {
            continue;
        }
        if dry_run {
            touched += 1;
            continue;
        }
        let now = Utc::now();
        if storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: "touched".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .is_ok()
        {
            touched += 1;
        }
    }

    PathEdgeReport {
        touched,
        commits_in_scope,
        hub_bound: bound,
        skipped_hubs,
    }
}

/// Exact file lists for commits already recorded and not on this page.
/// `git log --name-only --no-walk` so a later page does not re-diff history
/// and does not split a path into segments.
fn read_recorded_files(
    root: &Path,
    shas: &[String],
) -> HashMap<String, (DateTime<Utc>, Vec<String>)> {
    let mut out = HashMap::new();
    for chunk in shas.chunks(40) {
        let mut args = vec![
            "log".to_string(),
            "--no-walk".into(),
            "--name-only".into(),
            "--no-merges".into(),
            "--no-color".into(),
            "--no-ext-diff".into(),
            "--no-textconv".into(),
            "--no-show-signature".into(),
            "--pretty=format:%x1e%H%x1f%aI".into(),
        ];
        args.extend(chunk.iter().cloned());
        let Ok(run) = run_git(root, &args) else {
            continue;
        };
        if run.failure.is_some() && run.stdout.is_empty() {
            continue;
        }
        for (sha, time, files) in parse_name_only(&String::from_utf8_lossy(&run.stdout)) {
            out.insert(sha, (time, files));
        }
    }
    out
}

fn parse_name_only(raw: &str) -> Vec<(String, DateTime<Utc>, Vec<String>)> {
    let mut out = Vec::new();
    let mut current: Option<(String, DateTime<Utc>, Vec<String>)> = None;
    for line in raw.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix('\u{1e}') {
            if let Some(done) = current.take() {
                out.push(done);
            }
            let mut fields = rest.split('\u{1f}');
            let sha = fields.next().unwrap_or("").trim().to_string();
            let time = fields
                .next()
                .unwrap_or("")
                .trim()
                .parse::<DateTime<Utc>>()
                .unwrap_or_default();
            if sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()) {
                current = Some((sha, time, Vec::new()));
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if let Some((_, _, files)) = current.as_mut()
            && !files.iter().any(|have| have == line)
            && files.len() < RECORDED_FILES
        {
            files.push(line.to_string());
        }
    }
    if let Some(done) = current {
        out.push(done);
    }
    out
}

/// Point captured anchors at their record.
fn bind(anchors: Vec<CodeAnchor>, node_id: &str) -> impl Iterator<Item = CodeAnchor> {
    let node_id = node_id.to_string();
    anchors.into_iter().map(move |mut anchor| {
        anchor.node_id = node_id.clone();
        anchor
    })
}

/// Record every queued anchor in one log write. A commit's anchors are queued
/// together, so a failed write leaves whole commits without anchors, which a
/// rerun repairs.
fn flush_anchors(
    storage: &Arc<Storage>,
    queue: &mut Vec<CodeAnchor>,
    stats: &mut Stats,
    failures: &mut Failures,
) {
    if queue.is_empty() {
        return;
    }
    match storage.record_code_anchors(queue) {
        Ok(_) => stats.anchors.add(queue),
        Err(e) => {
            stats.anchor_errors += 1;
            failures.on_anchors_failed(format!(
                "{} anchors were not recorded ({e}); run again and they are added",
                queue.len()
            ));
        }
    }
    queue.clear();
}

/// Commits from `git log`, newest first, and why git stopped early if it did.
fn read_commits(
    root: &Path,
    req: &Request,
    limit: usize,
) -> Result<(Vec<GitCommit>, Option<String>), String> {
    // Nothing the checkout's own config names may run: no external diff, no
    // textconv filter, and no signature check (log.showSignature would run
    // gpg.program, a path the repository chooses).
    let mut args: Vec<String> = [
        "log",
        "-p",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-show-signature",
        "--no-merges",
        "--pretty=format:%x1e%H%x1f%aI%x1f%s",
    ]
    .iter()
    .map(|arg| arg.to_string())
    .collect();
    args.push("-n".into());
    args.push(limit.to_string());
    if let Some(since) = &req.since {
        args.push(format!("--since={since}"));
    }
    if let Some(until) = &req.until {
        args.push(format!("--until={until}"));
    }
    if let Some(rev) = &req.rev {
        args.push(rev.clone());
    }
    let run = run_git(root, &args)?;
    let mut commits = git_records::parse_git_log(&String::from_utf8_lossy(&run.stdout));
    let Some(failure) = run.failure else {
        return Ok((commits, None));
    };
    // git streams commit by commit and died partway. The last record may be
    // cut mid-diff, so it is not trusted; everything before it is whole.
    commits.pop();
    if commits.is_empty() {
        return Err(git_failure(&failure));
    }
    Ok((commits, Some(git_failure(&failure))))
}

/// What a git run produced. A non-zero exit is `failure` (stderr), not an
/// error: git may have written whole commits before it died.
struct GitRun {
    stdout: Vec<u8>,
    failure: Option<String>,
}

/// Run git in `root`, local-only, bounded in time and output size, never
/// prompting. `Err` is for failing to run git at all, a timeout, or output
/// over the cap.
///
/// The checkout may be untrusted, and its `.git/config` can name programs for
/// git to run. Settings given with `-c` win over that file, so the ones that
/// would run a program on a read are pinned off here: `log.showSignature`
/// (which runs `gpg.program`) and `core.fsmonitor` (a hook command).
fn run_git(root: &Path, args: &[String]) -> Result<GitRun, String> {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "--no-pager",
            "-c",
            "core.quotepath=off",
            "-c",
            "log.showSignature=false",
            "-c",
            "core.fsmonitor=false",
        ])
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_OPTIONAL_LOCKS", "0")
        // a partial clone must not fetch missing objects from its remote
        .env("GIT_NO_LAZY_FETCH", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run git (is it installed?): {e}"))?;
    let mut stdout = child.stdout.take().ok_or("git stdout was not captured")?;
    let mut stderr = child.stderr.take().ok_or("git stderr was not captured")?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let read = (&mut stdout).take(MAX_GIT_OUTPUT + 1).read_to_end(&mut buf);
        let _ = tx.send(read.map(|_| buf));
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut stderr).take(64 * 1024).read_to_end(&mut buf);
        buf
    });

    let out = match rx.recv_timeout(GIT_TIMEOUT) {
        Ok(Ok(buf)) => buf,
        Ok(Err(e)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("reading git output failed: {e}"));
        }
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "git timed out after {}s; lower limit or narrow with since, until or rev",
                GIT_TIMEOUT.as_secs()
            ));
        }
    };
    if out.len() as u64 > MAX_GIT_OUTPUT {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "git output exceeded {} MiB; lower limit or narrow with since, until or rev",
            MAX_GIT_OUTPUT / (1024 * 1024)
        ));
    }
    let status = child
        .wait()
        .map_err(|e| format!("waiting for git failed: {e}"))?;
    let err = stderr_reader.join().unwrap_or_default();
    Ok(GitRun {
        stdout: out,
        failure: (!status.success()).then(|| String::from_utf8_lossy(&err).to_string()),
    })
}

/// Refuse a `root` that is not the top of a working tree. git reports history
/// paths relative to the top, while anchors resolve against `root`, so from a
/// subdirectory every anchor would name the wrong file (in a monorepo, often a
/// file that exists). A bare repository has no files at all.
fn check_top_of_work_tree(root: &Path) -> Result<(), String> {
    let run = run_git(root, &["rev-parse".into(), "--show-toplevel".into()])?;
    if let Some(stderr) = run.failure {
        if !stderr.contains("must be run in a work tree") {
            return Err(git_failure(&stderr));
        }
        let bare = run_git(root, &["rev-parse".into(), "--is-bare-repository".into()])
            .ok()
            .filter(|run| run.failure.is_none())
            .is_some_and(|run| String::from_utf8_lossy(&run.stdout).trim() == "true");
        return Err(if bare {
            "repoPath is a bare repository: it has no working tree, so there are no files to anchor. Ingest a checkout of it (a clone without --bare) and pass the top of that working tree.".to_string()
        } else {
            "repoPath is inside a .git directory, which has no files to anchor. Pass the top of the working tree (the directory that holds .git).".to_string()
        });
    }
    let shown = String::from_utf8_lossy(&run.stdout).trim().to_string();
    let top = std::fs::canonicalize(&shown).map_err(|e| {
        format!("git names '{shown}' as the top of the working tree, but it cannot be opened: {e}")
    })?;
    if top.as_path() != root {
        return Err(format!(
            "repoPath must be the top of the working tree: pass '{}'. git reports history paths from there, so anchoring from '{}' would point at the wrong files.",
            top.display(),
            root.display()
        ));
    }
    Ok(())
}

const PARTIAL_CLONE_HINT: &str = "Git could not read objects this repository does not have on disk. A partial (blobless) clone keeps only some file contents locally, and this tool never fetches. Fetch the history first (for example `git -C <repo> backfill`, or `git -C <repo> fetch --refetch`), or clone without --filter, then ingest again: the commits already recorded are skipped.";

/// A git failure in one line, with the common causes named.
fn git_failure(stderr: &str) -> String {
    let first = stderr
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("git exited with an error");
    if stderr.contains("does not have any commits yet") {
        return "the repository has no commits yet".to_string();
    }
    if stderr.contains("not a git repository") {
        return "repoPath is not a git repository".to_string();
    }
    let missing_objects = [
        "promisor",
        "attempting to fetch",
        "unable to read",
        "bad object",
        "missing blob",
        "not in the object database",
        "could not fetch",
        "lazy",
    ]
    .iter()
    .any(|marker| stderr.contains(marker));
    if missing_objects {
        format!("git failed: {first}. {PARTIAL_CLONE_HINT}")
    } else {
        format!("git failed: {first}")
    }
}

/// The anchors one commit's record keeps, and what was left out and why.
struct Captured {
    anchors: Vec<CodeAnchor>,
    not_in_checkout: usize,
    outside_checkout: usize,
    /// Captured but not kept: unverifiable beside a verifiable anchor.
    dropped_unverifiable: usize,
    truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Presence {
    Present,
    Missing,
    Outside,
}

/// Capture the anchors a commit's record keeps. `node_id` is stamped on each;
/// pass an empty id when the node does not exist yet and bind it later.
fn capture_anchors(commit: &GitCommit, root: &Path, node_id: &str) -> Captured {
    // every touched file, looked at once
    let presence: HashMap<&str, Presence> = commit
        .files
        .iter()
        .map(|file| (file.as_str(), presence_of(root, file)))
        .collect();
    let present = |file: &str| presence.get(file) == Some(&Presence::Present);

    let mut drafts: Vec<AnchorDraft> = Vec::new();
    let mut anchored_with_symbol: HashSet<&str> = HashSet::new();
    // `<file>/<symbol>`: the symbol is one identifier, so the last `/` splits it
    for qualified in &commit.symbols {
        let Some((file, symbol)) = qualified.rsplit_once('/') else {
            continue;
        };
        if present(file) {
            drafts.push(AnchorDraft::new(file).with_symbol(symbol));
            anchored_with_symbol.insert(file);
        }
    }
    // a file with no symbol is anchored by path, so its removal still shows
    for file in &commit.files {
        if !anchored_with_symbol.contains(file.as_str()) && present(file) {
            drafts.push(AnchorDraft::new(file.as_str()));
        }
    }

    let mut anchors: Vec<CodeAnchor> = drafts
        .iter()
        .map(|draft| capture_anchor(node_id, root, draft))
        .collect();
    // Hashed from the tree a moment ago, so it matches the tree: say so, and
    // the first `verify` has no verdict to write for it.
    for anchor in anchors.iter_mut().filter(|anchor| anchor.is_verifiable()) {
        anchor.last_status = Some(AnchorStatus::Verified);
        anchor.last_verified_at = Some(anchor.captured_at);
    }
    let mut dropped_unverifiable = 0;
    if anchors.iter().any(CodeAnchor::is_verifiable) {
        let before = anchors.len();
        anchors.retain(CodeAnchor::is_verifiable);
        dropped_unverifiable = before - anchors.len();
    }
    let truncated = anchors.len() > MAX_ANCHORS_PER_COMMIT;
    anchors.truncate(MAX_ANCHORS_PER_COMMIT);

    let count = |wanted: Presence| presence.values().filter(|found| **found == wanted).count();
    Captured {
        anchors,
        not_in_checkout: count(Presence::Missing),
        outside_checkout: count(Presence::Outside),
        dropped_unverifiable,
        truncated,
    }
}

/// Is `file` a regular file inside the canonical checkout `root`? A path that
/// resolves outside it (a symlink, `..`) is never read.
fn presence_of(root: &Path, file: &str) -> Presence {
    let relative = Path::new(file);
    if relative.is_absolute()
        || relative.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Presence::Outside;
    }
    match std::fs::canonicalize(root.join(relative)) {
        Ok(real) if !real.starts_with(root) => Presence::Outside,
        Ok(real) if real.is_file() => Presence::Present,
        _ => Presence::Missing,
    }
}

fn trimmed(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn short(sha: &str) -> &str {
    &sha[..sha.len().min(12)]
}

/// A codebase or scope name: visible text the log accepts.
fn check_label(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.len() > MAX_ARG_LEN || value.chars().any(char::is_control) {
        return Err(format!(
            "{name} must be a non-empty identifier of at most {MAX_ARG_LEN} visible characters"
        ));
    }
    Ok(())
}

/// A value handed to git: no option injection, no control characters.
fn check_git_arg(name: &str, value: &str) -> Result<(), String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if value.starts_with('-') {
        return Err(format!("{name} must not start with '-'"));
    }
    if value.len() > MAX_ARG_LEN || value.chars().any(char::is_control) {
        return Err(format!(
            "{name} must be at most {MAX_ARG_LEN} characters with no control characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use chrono::DateTime;
    use serde_json::json;
    use tokio::sync::Mutex;

    const AUTH_V1: &str = "\
pub fn load_config(path: &str) -> String {
    let raw = path.to_string();
    raw
}

pub fn parse_timeout(raw: &str) -> u32 {
    raw.len() as u32
}
";
    const DATES: [&str; 4] = [
        "2026-01-01T10:00:00+00:00",
        "2026-01-02T10:00:00+00:00",
        "2026-01-03T10:00:00+00:00",
        "2026-01-04T10:00:00+00:00",
    ];

    /// A real git repository in a temp dir, isolated from the host's git config.
    struct Repo {
        dir: tempfile::TempDir,
        shas: Vec<String>,
    }

    impl Repo {
        fn empty() -> Self {
            let repo = Repo {
                dir: tempfile::TempDir::new().unwrap(),
                shas: Vec::new(),
            };
            repo.git(&["init", "-q"], DATES[0]);
            repo
        }

        /// Four commits: add a file, change `load_config` and add a test, add a
        /// README, change `parse_timeout`.
        fn standard() -> Self {
            let mut repo = Repo::empty();
            repo.write("src/auth.rs", AUTH_V1);
            repo.commit("add auth", DATES[0]);
            repo.write(
                "src/auth.rs",
                &AUTH_V1.replace("path.to_string()", "path.trim().to_string()"),
            );
            repo.write("tests/auth_test.rs", "// placeholder\n");
            repo.commit("trim the path in load_config", DATES[1]);
            repo.write("README.md", "# demo\n");
            repo.commit("docs", DATES[2]);
            repo.write(
                "src/auth.rs",
                &AUTH_V1
                    .replace("path.to_string()", "path.trim().to_string()")
                    .replace("raw.len()", "raw.trim().len()"),
            );
            repo.commit("fix parse_timeout", DATES[3]);
            repo
        }

        fn git(&self, args: &[&str], date: &str) -> String {
            let out = Command::new("git")
                .arg("-C")
                .arg(self.dir.path())
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

        fn write(&self, rel: &str, body: &str) {
            let path = self.dir.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }

        fn commit(&mut self, message: &str, date: &str) {
            self.git(&["add", "-A"], date);
            self.git(&["commit", "-q", "-m", message], date);
            let sha = self.git(&["rev-parse", "HEAD"], date);
            self.shas.push(sha);
        }

        fn request(&self, dry_run: bool) -> Request {
            Request {
                repo_path: self.dir.path().to_path_buf(),
                codebase: Some("demo".into()),
                scope: None,
                rev: None,
                since: None,
                until: None,
                limit: None,
                dry_run,
                budget: None,
            }
        }
    }

    fn strata() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    async fn ingest(storage: &Arc<Storage>, request: Request) -> Value {
        execute(storage, request).await.unwrap()
    }

    fn nodes(storage: &Arc<Storage>, scope: &str) -> Vec<vestige_core::KnowledgeNode> {
        storage.get_all_nodes_in_scope(scope, 1000, 0).unwrap()
    }

    async fn verify(storage: &Arc<Storage>, repo: &Repo) -> Value {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        crate::tools::codebase_unified::execute(
            storage,
            &cognitive,
            &vestige_core::OutputConfig::default(),
            Some(json!({
                "action": "verify",
                "codebase": "demo",
                "scope": "demo",
                "repoPath": repo.dir.path().to_str().unwrap(),
            })),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn a_preview_writes_nothing_and_predicts_the_write_exactly() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();

        let preview = ingest(&storage, repo.request(true)).await;
        assert_eq!(preview["dryRun"], true);
        assert_eq!(preview["scope"], "demo", "isolated by default: {preview}");
        assert_eq!(preview["commits"]["seen"], 4);
        assert_eq!(preview["commits"]["new"], 4);
        assert_eq!(preview["commits"]["created"], 0);
        assert!(
            preview["nextStep"]
                .as_str()
                .unwrap()
                .contains("dryRun=false"),
            "{preview}"
        );
        assert!(nodes(&storage, "demo").is_empty(), "a preview wrote");
        assert!(nodes(&storage, "user").is_empty());
        let sample = preview["sample"].as_array().unwrap();
        assert_eq!(sample[0]["subject"], "fix parse_timeout", "newest first");

        let written = ingest(&storage, repo.request(false)).await;
        assert_eq!(written["commits"]["created"], 4, "{written}");
        assert_eq!(
            preview["anchors"]["wouldRecord"], written["anchors"]["recorded"],
            "the preview's anchor count is the write's"
        );
        assert_eq!(
            preview["anchors"]["verifiable"], written["anchors"]["verifiable"],
            "{preview} vs {written}"
        );
    }

    #[tokio::test]
    async fn each_commit_is_a_dated_provenanced_exact_handle_record() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let out = ingest(&storage, repo.request(false)).await;
        assert_eq!(out["commits"]["created"], 4, "{out}");
        assert_eq!(out["partial"], false);
        assert!(out["error"].is_null());
        // the suggested verify call names this checkout, ready to run
        let top = std::fs::canonicalize(repo.dir.path()).unwrap();
        let next = out["nextStep"].as_str().unwrap();
        assert!(
            next.contains(&format!("repoPath='{}'", top.display())),
            "{next}"
        );

        let recorded = nodes(&storage, "demo");
        assert_eq!(recorded.len(), 4);
        assert!(
            nodes(&storage, "user").is_empty(),
            "a repo never lands in the shared user scope by default"
        );
        for (index, sha) in repo.shas.iter().enumerate() {
            let tag = commit_tag(sha);
            let found = storage
                .current_code_context_nodes("event", Some(&tag), "demo", 10)
                .unwrap();
            assert_eq!(
                found.len(),
                1,
                "exact handle {tag} finds exactly its commit"
            );
            let node = &found[0];
            assert_eq!(node.node_type, "event");
            for want in ["git-commit", "codebase:demo", tag.as_str()] {
                assert!(
                    node.tags.iter().any(|t| t == want),
                    "{want} in {:?}",
                    node.tags
                );
            }
            assert!(
                node.content.starts_with(&format!("commit {sha}")),
                "{}",
                node.content
            );
            // valid time is the commit's time; the write time stays the real one
            let expected = DateTime::parse_from_rfc3339(DATES[index]).unwrap();
            assert_eq!(node.valid_from.unwrap(), expected, "valid_from of {sha}");
            assert!(node.created_at > expected, "write time is not backdated");
            // provenance
            let envelope = node.source_envelope.as_ref().expect("provenance");
            assert_eq!(envelope.source_system.as_deref(), Some("git"));
            assert_eq!(envelope.source_project.as_deref(), Some("demo"));
            assert_eq!(envelope.source_id.as_deref(), Some(sha.as_str()));
        }
        // oldest written first, so ids follow history
        let mut by_id = recorded.clone();
        by_id.sort_by(|a, b| a.id.cmp(&b.id));
        assert!(
            by_id[0].content.contains("add auth"),
            "{}",
            by_id[0].content
        );
        assert!(by_id[3].content.contains("fix parse_timeout"));
    }

    #[tokio::test]
    async fn a_rerun_skips_what_is_recorded_and_paging_resumes() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();

        let mut newest_two = repo.request(false);
        newest_two.limit = Some(2);
        let first = ingest(&storage, newest_two).await;
        assert_eq!(first["commits"]["created"], 2, "{first}");
        assert_eq!(first["more"], true);
        assert_eq!(
            first["pageBackWith"]["rev"],
            format!("{}~1", repo.shas[2]),
            "points just before the oldest commit read"
        );

        // read all four: the two already recorded are skipped, not duplicated
        let all = ingest(&storage, repo.request(false)).await;
        assert_eq!(all["commits"]["alreadyIngested"], 2, "{all}");
        assert_eq!(all["commits"]["created"], 2);
        assert_eq!(nodes(&storage, "demo").len(), 4);

        let again = ingest(&storage, repo.request(false)).await;
        assert_eq!(again["commits"]["new"], 0);
        assert_eq!(again["commits"]["created"], 0);
        assert_eq!(again["commits"]["alreadyIngested"], 4);
        assert_eq!(nodes(&storage, "demo").len(), 4, "a rerun wrote nothing");
        assert_eq!(again["more"], false);
        assert!(again["pageBackWith"].is_null());

        // paging backwards with the hint reads the older commits
        let (storage2, _dir2) = strata();
        let mut newest = repo.request(false);
        newest.limit = Some(2);
        ingest(&storage2, newest).await;
        let mut older = repo.request(false);
        older.limit = Some(2);
        older.rev = Some(format!("{}~1", repo.shas[2]));
        let page = ingest(&storage2, older).await;
        assert_eq!(page["commits"]["created"], 2, "{page}");
        assert_eq!(nodes(&storage2, "demo").len(), 4);
    }

    #[tokio::test]
    async fn verify_watches_the_change_records_and_sees_drift() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        ingest(&storage, repo.request(false)).await;

        // fresh: the two commits with a located symbol are verifiable; the
        // two with only a path (a new file, a README) can only be unverifiable
        let report = verify(&storage, &repo).await;
        assert_eq!(report["checked"], 4, "{report}");
        assert_eq!(report["checkedByType"]["event"], 4);
        assert_eq!(report["fresh"], 2, "{report}");
        assert_eq!(report["stale"], 0, "{report}");
        assert_eq!(report["unverifiable"], 2, "{report}");

        // the code a commit touched changes after ingestion: its record drifts
        repo.write(
            "src/auth.rs",
            "pub fn load_config(path: &str) -> String {\n    String::new()\n}\n\npub fn parse_timeout(raw: &str) -> u32 {\n    raw.trim().len() as u32\n}\n",
        );
        let drifted = verify(&storage, &repo).await;
        assert_eq!(drifted["stale"], 1, "{drifted}");
        let stale = &drifted["staleMemories"][0];
        assert_eq!(stale["nodeType"], "event");
        assert!(
            stale["content"]
                .as_str()
                .unwrap()
                .contains("trim the path in load_config"),
            "the commit that touched load_config drifted: {stale}"
        );
        assert_eq!(drifted["fresh"], 1, "parse_timeout is unchanged: {drifted}");

        // a touched file that is deleted is reported too
        std::fs::remove_file(repo.dir.path().join("README.md")).unwrap();
        let gone = verify(&storage, &repo).await;
        assert_eq!(gone["stale"], 2, "{gone}");
        assert!(
            gone["staleMemories"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["status"] == "missing"),
            "{gone}"
        );
    }

    #[tokio::test]
    async fn verify_without_a_codebase_does_not_read_every_event() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        ingest(&storage, repo.request(false)).await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let report = crate::tools::codebase_unified::execute(
            &storage,
            &cognitive,
            &vestige_core::OutputConfig::default(),
            Some(json!({
                "action": "verify",
                "scope": "demo",
                "repoPath": repo.dir.path().to_str().unwrap(),
            })),
        )
        .await
        .unwrap();
        assert_eq!(report["checked"], 0, "{report}");
    }

    #[tokio::test]
    async fn a_future_dated_commit_is_clamped_and_stays_visible() {
        let mut repo = Repo::standard();
        repo.write("src/future.rs", "pub fn from_the_future() {}\n");
        repo.commit("clock skew", "2099-01-01T00:00:00+00:00");
        let (storage, _dir) = strata();
        let out = ingest(&storage, repo.request(false)).await;
        assert_eq!(out["commits"]["futureDatesClamped"], 1, "{out}");
        assert_eq!(out["commits"]["created"], 5);
        let tag = commit_tag(repo.shas.last().unwrap());
        let found = storage
            .current_code_context_nodes("event", Some(&tag), "demo", 10)
            .unwrap();
        let node = &found[0];
        assert!(node.valid_from.unwrap() <= Utc::now());
        assert!(
            node.is_currently_valid(),
            "a future date must not hide the record"
        );
    }

    #[tokio::test]
    async fn hostile_or_broken_arguments_are_refused_before_anything_is_written() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let refuse = |mutate: &dyn Fn(&mut Request)| {
            let mut request = repo.request(false);
            mutate(&mut request);
            request
        };

        let err = execute(
            &storage,
            refuse(&|r| r.rev = Some("--output=/tmp/x".into())),
        )
        .await
        .unwrap_err();
        assert!(err.contains("must not start with '-'"), "{err}");
        let err = execute(&storage, refuse(&|r| r.since = Some("-1".into())))
            .await
            .unwrap_err();
        assert!(err.contains("since"), "{err}");
        let err = execute(&storage, refuse(&|r| r.until = Some("a\nb".into())))
            .await
            .unwrap_err();
        assert!(err.contains("control"), "{err}");
        let err = execute(
            &storage,
            refuse(&|r| r.codebase = Some("bad\u{7}name".into())),
        )
        .await
        .unwrap_err();
        assert!(err.contains("codebase"), "{err}");
        let err = execute(&storage, refuse(&|r| r.scope = Some("x".repeat(201))))
            .await
            .unwrap_err();
        assert!(err.contains("scope"), "{err}");
        let err = execute(
            &storage,
            refuse(&|r| r.repo_path = PathBuf::from("/definitely/not/here")),
        )
        .await
        .unwrap_err();
        assert!(err.contains("not an available directory"), "{err}");

        let plain = tempfile::TempDir::new().unwrap();
        let mut request = repo.request(false);
        request.repo_path = plain.path().to_path_buf();
        let err = execute(&storage, request).await.unwrap_err();
        assert!(err.contains("not a git repository"), "{err}");

        let empty = Repo::empty();
        let err = execute(&storage, empty.request(false)).await.unwrap_err();
        assert!(err.contains("no commits yet"), "{err}");

        assert!(nodes(&storage, "demo").is_empty(), "a refusal wrote");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_leading_outside_the_checkout_is_never_read_or_anchored() {
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("hosts");
        std::fs::write(&secret, "127.0.0.1 localhost\n").unwrap();

        let mut repo = Repo::empty();
        repo.write("src/ok.rs", "pub fn fine() {}\n");
        std::fs::create_dir_all(repo.dir.path().join("src")).unwrap();
        std::os::unix::fs::symlink(&secret, repo.dir.path().join("src/leak.rs")).unwrap();
        repo.commit("add a link", DATES[0]);

        let (storage, _dir) = strata();
        let out = ingest(&storage, repo.request(false)).await;
        assert_eq!(out["commits"]["created"], 1, "{out}");
        assert_eq!(out["anchors"]["pathsOutsideCheckout"], 1, "{out}");
        let id = &nodes(&storage, "demo")[0].id;
        let anchored: Vec<String> = storage
            .code_anchors_for_node(id)
            .unwrap()
            .into_iter()
            .map(|anchor| anchor.file_path)
            .collect();
        assert!(
            !anchored.iter().any(|p| p.ends_with("leak.rs")),
            "{anchored:?}"
        );
        assert!(
            anchored.iter().any(|p| p.ends_with("ok.rs")),
            "{anchored:?}"
        );
    }

    #[tokio::test]
    async fn a_commit_the_gate_refuses_is_skipped_not_fatal() {
        let mut repo = Repo::empty();
        repo.write("a.txt", "one\n");
        repo.commit("first", DATES[0]);
        repo.write("b.txt", "two\n");
        repo.commit("leaked key AKIAIOSFODNN7EXAMPLE in a message", DATES[1]);
        repo.write("c.txt", "three\n");
        repo.commit("third", DATES[2]);

        let (storage, _dir) = strata();
        let out = ingest(&storage, repo.request(false)).await;
        assert_eq!(out["commits"]["skippedSecret"], 1, "{out}");
        assert_eq!(out["commits"]["created"], 2);
        assert_eq!(out["partial"], false);
        assert!(out["error"].is_null());
        let leaked = commit_tag(&repo.shas[1]);
        assert!(
            storage
                .current_code_context_nodes("event", Some(&leaked), "demo", 10)
                .unwrap()
                .is_empty(),
            "the refused commit is not in the log"
        );
    }

    #[tokio::test]
    async fn merge_commits_carry_no_diff_and_are_skipped() {
        let mut repo = Repo::standard();
        repo.git(&["switch", "-q", "-c", "side"], DATES[3]);
        repo.write("side.txt", "side\n");
        repo.commit("on a side branch", DATES[3]);
        repo.git(&["switch", "-q", "-"], DATES[3]);
        repo.git(
            &["merge", "-q", "--no-ff", "side", "-m", "merge side"],
            DATES[3],
        );

        let (storage, _dir) = strata();
        let out = ingest(&storage, repo.request(true)).await;
        assert_eq!(
            out["commits"]["seen"], 5,
            "four plus the side commit: {out}"
        );
        assert_eq!(out["repo"]["mergeCommits"], "skipped");
    }

    #[tokio::test]
    async fn the_codebase_name_defaults_to_the_checkout_directory() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let mut request = repo.request(true);
        request.codebase = None;
        let out = ingest(&storage, request).await;
        let expected = repo
            .dir
            .path()
            .canonicalize()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert_eq!(out["codebase"], expected);
        assert_eq!(out["scope"], expected);
    }

    /// Delete a blob from the object database, as a partial clone lacks it.
    #[cfg(unix)]
    fn forget_blob(repo: &Repo, spec: &str) {
        let blob = repo.git(&["rev-parse", spec], DATES[0]);
        let path = repo
            .dir
            .path()
            .join(".git/objects")
            .join(&blob[..2])
            .join(&blob[2..]);
        std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn git_dying_partway_keeps_the_whole_commits_and_says_where_it_stopped() {
        let repo = Repo::standard();
        // the oldest version of auth.rs is gone: reading c2's diff needs it
        forget_blob(&repo, &format!("{}:src/auth.rs", repo.shas[0]));
        let (storage, _dir) = strata();

        let out = ingest(&storage, repo.request(false)).await;
        let stopped = &out["gitStoppedEarly"];
        assert_eq!(stopped["commitsKept"], 2, "c4 and c3 are whole: {out}");
        assert!(
            stopped["hint"].as_str().unwrap().contains("never fetches"),
            "{stopped}"
        );
        assert!(
            stopped["reason"].as_str().unwrap().contains("git failed"),
            "{stopped}"
        );
        assert_eq!(out["commits"]["created"], 2, "{out}");
        assert_eq!(out["more"], false, "a stopped read is not a full page");
        // the cut-off commit (c2) was dropped, not recorded half-read
        let recorded = nodes(&storage, "demo");
        assert_eq!(recorded.len(), 2);
        for node in &recorded {
            assert!(node.content.contains("files: "), "{}", node.content);
        }
        for dropped in [&repo.shas[0], &repo.shas[1]] {
            let tag = commit_tag(dropped);
            assert!(
                storage
                    .current_code_context_nodes("event", Some(&tag), "demo", 10)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn when_nothing_is_readable_the_error_names_the_cause_and_the_fix() {
        let repo = Repo::standard();
        // the newest file version is gone: even the first commit cannot be read
        forget_blob(&repo, "HEAD:src/auth.rs");
        let (storage, _dir) = strata();
        let err = execute(&storage, repo.request(false)).await.unwrap_err();
        assert!(err.contains("does not have on disk"), "{err}");
        assert!(err.contains("never fetches"), "{err}");
        assert!(nodes(&storage, "demo").is_empty());
    }

    #[test]
    fn git_failures_are_classified_without_inventing_causes() {
        assert_eq!(
            git_failure("fatal: your current branch 'main' does not have any commits yet\n"),
            "the repository has no commits yet"
        );
        assert_eq!(
            git_failure("fatal: not a git repository (or any parent)\n"),
            "repoPath is not a git repository"
        );
        let lazy =
            git_failure("fatal: You are attempting to fetch abc, which is in the commit graph\n");
        assert!(lazy.contains("never fetches"), "{lazy}");
        let other = git_failure("fatal: ambiguous argument 'nope'\n");
        assert_eq!(other, "git failed: fatal: ambiguous argument 'nope'");
        assert!(!other.contains("never fetches"));
        assert_eq!(git_failure(""), "git failed: git exited with an error");
    }

    #[tokio::test]
    async fn verifiable_anchors_carry_their_capture_verdict_so_verify_has_nothing_to_write() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        ingest(&storage, repo.request(false)).await;

        let mut verifiable = 0;
        for node in nodes(&storage, "demo") {
            for anchor in storage.code_anchors_for_node(&node.id).unwrap() {
                if anchor.is_verifiable() {
                    verifiable += 1;
                    assert_eq!(anchor.last_status, Some(AnchorStatus::Verified));
                    assert_eq!(anchor.last_verified_at, Some(anchor.captured_at));
                }
            }
        }
        assert!(verifiable >= 2, "the fixture has located symbols");

        // verify finds them fresh and leaves the stamps as they were
        verify(&storage, &repo).await;
        for node in nodes(&storage, "demo") {
            for anchor in storage.code_anchors_for_node(&node.id).unwrap() {
                if anchor.is_verifiable() {
                    assert_eq!(
                        anchor.last_verified_at,
                        Some(anchor.captured_at),
                        "verify rewrote a verdict that was already right"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn a_rerun_gives_missing_anchors_to_commits_an_earlier_run_left_bare() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        // as if a run died between the record write and the anchor write
        let now = Utc::now();
        let commits = read_commits(
            &std::fs::canonicalize(repo.dir.path()).unwrap(),
            &repo.request(true),
            10,
        )
        .unwrap()
        .0;
        for commit in commits.iter().filter(|c| c.subject != "docs") {
            storage
                .ingest_in_scope(change_record(commit, "demo", now), "demo")
                .unwrap();
        }
        assert_eq!(nodes(&storage, "demo").len(), 3);
        for node in nodes(&storage, "demo") {
            assert!(storage.code_anchors_for_node(&node.id).unwrap().is_empty());
        }

        let preview = ingest(&storage, repo.request(true)).await;
        assert_eq!(
            preview["anchors"]["commitsGivenMissingAnchors"], 3,
            "{preview}"
        );
        assert_eq!(preview["commits"]["new"], 1, "only docs is new");

        let out = ingest(&storage, repo.request(false)).await;
        assert_eq!(out["commits"]["created"], 1, "{out}");
        assert_eq!(out["commits"]["alreadyIngested"], 3);
        assert_eq!(out["anchors"]["commitsGivenMissingAnchors"], 3, "{out}");
        for node in nodes(&storage, "demo") {
            assert!(
                !storage.code_anchors_for_node(&node.id).unwrap().is_empty(),
                "{} still has no anchors",
                node.content.lines().next().unwrap()
            );
        }
        // and once repaired, a further run has nothing to repair
        let again = ingest(&storage, repo.request(false)).await;
        assert_eq!(again["anchors"]["commitsGivenMissingAnchors"], 0, "{again}");
        assert_eq!(report_fresh(&verify(&storage, &repo).await), (2, 0));
    }

    /// (fresh, stale) from a verify report.
    fn report_fresh(report: &Value) -> (u64, u64) {
        (
            report["fresh"].as_u64().unwrap(),
            report["stale"].as_u64().unwrap(),
        )
    }

    #[tokio::test]
    async fn a_spent_write_budget_stops_cleanly_and_a_rerun_finishes() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let mut request = repo.request(false);
        request.budget = Some(Duration::ZERO);
        let stopped = ingest(&storage, request).await;
        assert_eq!(stopped["commits"]["stoppedByBudget"], true, "{stopped}");
        assert_eq!(stopped["commits"]["created"], 0);
        assert_eq!(stopped["commits"]["remaining"], 4);
        assert_eq!(stopped["partial"], false, "a budget stop is not an error");
        assert!(
            stopped["nextStep"]
                .as_str()
                .unwrap()
                .contains("Run the same call again"),
            "{stopped}"
        );

        let done = ingest(&storage, repo.request(false)).await;
        assert_eq!(done["commits"]["created"], 4, "{done}");
        assert_eq!(done["commits"]["remaining"], 0);
        assert_eq!(done["commits"]["stoppedByBudget"], false);
    }

    #[tokio::test]
    async fn the_limit_is_capped_and_reported() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let mut request = repo.request(true);
        request.limit = Some(1_000_000);
        let out = ingest(&storage, request).await;
        assert_eq!(out["limit"], MAX_LIMIT as u64, "{out}");
        let mut request = repo.request(true);
        request.limit = Some(0);
        assert_eq!(ingest(&storage, request).await["limit"], 1);
    }

    #[test]
    fn git_arguments_are_validated_for_option_injection_and_control_bytes() {
        assert!(check_git_arg("rev", "v1.2..v1.3").is_ok());
        assert!(check_git_arg("rev", "abc123~1").is_ok());
        assert!(check_git_arg("since", "2 weeks ago").is_ok());
        for bad in ["", "   ", "-x", "--exec=sh", "a\u{0}b", "a\nb"] {
            assert!(check_git_arg("rev", bad).is_err(), "{bad:?}");
        }
        assert!(check_git_arg("rev", &"a".repeat(MAX_ARG_LEN + 1)).is_err());
    }

    /// Any `codebase` call, unwrapped.
    async fn codebase_call(storage: &Arc<Storage>, args: Value) -> Value {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        crate::tools::codebase_unified::execute(
            storage,
            &cognitive,
            &vestige_core::OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap()
    }

    /// git names history paths from the top of the working tree, so a
    /// subdirectory would anchor the wrong files. It is refused, naming the
    /// directory to pass instead.
    #[tokio::test]
    async fn a_subdirectory_of_the_checkout_is_refused_with_the_top_to_pass() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let top = std::fs::canonicalize(repo.dir.path()).unwrap();
        for dry_run in [true, false] {
            let mut request = repo.request(dry_run);
            request.repo_path = repo.dir.path().join("src");
            let err = execute(&storage, request).await.unwrap_err();
            assert!(
                err.contains(&format!(
                    "repoPath must be the top of the working tree: pass '{}'",
                    top.display()
                )),
                "{err}"
            );
        }
        assert!(nodes(&storage, "demo").is_empty(), "a refusal wrote");
        // the top itself is accepted
        let preview = ingest(&storage, repo.request(true)).await;
        assert_eq!(preview["commits"]["seen"], 4, "{preview}");
    }

    #[tokio::test]
    async fn a_bare_repository_is_refused_because_it_has_no_files_to_anchor() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let elsewhere = tempfile::TempDir::new().unwrap();
        let bare = elsewhere.path().join("demo.git");
        repo.git(
            &["clone", "-q", "--bare", ".", bare.to_str().unwrap()],
            DATES[0],
        );
        let mut request = repo.request(true);
        request.repo_path = bare;
        let err = execute(&storage, request).await.unwrap_err();
        assert!(err.contains("bare repository"), "{err}");
        assert!(err.contains("no files to anchor"), "{err}");

        // a checkout's .git directory has no files to anchor either
        let mut request = repo.request(true);
        request.repo_path = repo.dir.path().join(".git");
        let err = execute(&storage, request).await.unwrap_err();
        assert!(err.contains(".git directory"), "{err}");
        assert!(err.contains("no files to anchor"), "{err}");
    }

    /// Every agent shares one server process. A second write of the same
    /// codebase into the same scope would read the recorded set before the
    /// first wrote and duplicate every commit, so it waits its turn.
    #[tokio::test]
    async fn a_second_write_of_the_same_codebase_and_scope_is_refused_while_one_runs() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        let held = claim_write(&storage, "demo", "demo").unwrap();

        let err = execute(&storage, repo.request(false)).await.unwrap_err();
        assert!(
            err.contains("an ingest of codebase 'demo' into scope 'demo' is already running"),
            "{err}"
        );
        assert!(err.contains("run again when it finishes"), "{err}");
        assert!(
            nodes(&storage, "demo").is_empty(),
            "the refused write wrote"
        );

        // a preview only reads, so it is never refused
        let preview = ingest(&storage, repo.request(true)).await;
        assert_eq!(preview["commits"]["new"], 4, "{preview}");
        // another scope is a different key
        let mut other = repo.request(false);
        other.scope = Some("elsewhere".into());
        assert_eq!(ingest(&storage, other).await["commits"]["created"], 4);

        drop(held);
        let written = ingest(&storage, repo.request(false)).await;
        assert_eq!(written["commits"]["created"], 4, "{written}");
        // a finished write, or one that failed, frees its key
        let empty = Repo::empty();
        let err = execute(&storage, empty.request(false)).await.unwrap_err();
        assert!(err.contains("no commits yet"), "{err}");
        let again = ingest(&storage, repo.request(false)).await;
        assert_eq!(again["commits"]["alreadyIngested"], 4, "{again}");
        assert_eq!(nodes(&storage, "demo").len(), 4, "nothing was duplicated");
    }

    /// Replace HEAD with the same commit carrying a PGP signature block, as
    /// `git commit -S` writes it. Nothing checks it unless git is asked to.
    #[cfg(unix)]
    fn sign_head(repo: &Repo) {
        let raw = repo.git(&["cat-file", "commit", "HEAD"], DATES[0]);
        let (headers, message) = raw.split_once("\n\n").unwrap();
        let signed = format!(
            "{headers}\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n iQEzBAABCAAdFiEEAAAAAAAAAAAAAAAAAAAAAAAAAAAFAmZmZmYACgkQAAAAAAAAAAA=\n =AAAA\n -----END PGP SIGNATURE-----\n\n{message}\n"
        );
        let object = repo.dir.path().join(".git").join("signed-commit");
        std::fs::write(&object, signed).unwrap();
        let sha = repo.git(
            &[
                "hash-object",
                "-t",
                "commit",
                "-w",
                object.to_str().unwrap(),
            ],
            DATES[0],
        );
        repo.git(&["update-ref", "HEAD", sha.as_str()], DATES[0]);
    }

    /// An untrusted checkout's own config must not make a read run a program:
    /// with log.showSignature set, `git log` runs gpg.program on every signed
    /// commit it shows.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_checkout_config_cannot_make_an_ingest_run_a_program() {
        use std::os::unix::fs::PermissionsExt;

        let repo = Repo::standard();
        sign_head(&repo);
        let outside = tempfile::TempDir::new().unwrap();
        let marker = outside.path().join("ran");
        let program = outside.path().join("evil.sh");
        std::fs::write(
            &program,
            format!("#!/bin/sh\ntouch '{}'\nexit 1\n", marker.display()),
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        repo.git(
            &["config", "gpg.program", program.to_str().unwrap()],
            DATES[0],
        );
        repo.git(&["config", "log.showSignature", "true"], DATES[0]);

        let (storage, _dir) = strata();
        let preview = ingest(&storage, repo.request(true)).await;
        assert_eq!(preview["commits"]["seen"], 4, "{preview}");
        assert!(
            !marker.exists(),
            "the checkout's gpg.program ran during an ingest"
        );
    }

    #[test]
    fn the_next_step_follows_how_the_call_ended() {
        let root = Path::new("/checkouts/demo");
        let outcome = |ending, skipped_secret| Outcome {
            ending,
            to_write: 4,
            remaining: 2,
            skipped_secret,
            budget: Duration::from_secs(45),
            codebase: "demo",
            scope: "demo",
            root,
        };

        // an anchor write failed but every commit is in: not "stopped early"
        let anchors = suggest_next_step(&outcome(Ending::AnchorsMissing, 0));
        assert!(
            anchors.starts_with("Every commit was recorded, but some of their anchors were not"),
            "{anchors}"
        );
        assert!(
            anchors.contains("Run the same call again to add them"),
            "{anchors}"
        );
        assert!(!anchors.contains("stopped early"), "{anchors}");
        let anchors = suggest_next_step(&outcome(Ending::AnchorsMissing, 1));
        assert!(
            anchors.starts_with("Every commit except the 1 the secret gate refused was recorded"),
            "{anchors}"
        );
        let refused = suggest_next_step(&outcome(Ending::RefusedCommit, 0));
        assert!(refused.starts_with("The ingest stopped early"), "{refused}");
        let budget = suggest_next_step(&outcome(Ending::Budget, 0));
        assert!(
            budget.contains("45s write budget with 2 commits left"),
            "{budget}"
        );
        let preview = suggest_next_step(&outcome(Ending::Preview, 0));
        assert!(preview.contains("record 4 change records"), "{preview}");
        // the suggested verify call names the real checkout
        let done = suggest_next_step(&outcome(Ending::Complete, 0));
        assert!(
            done.contains(
                "codebase action='verify' codebase='demo' scope='demo' repoPath='/checkouts/demo'"
            ),
            "{done}"
        );
        assert!(!done.contains("<this checkout>"), "{done}");

        // which ending a call had, and the error it reports
        let mut failures = Failures::default();
        assert_eq!(Ending::of(false, false, &failures), Ending::Complete);
        failures.on_anchors_failed("anchors failed".into());
        assert_eq!(Ending::of(false, false, &failures), Ending::AnchorsMissing);
        failures.on_commit_refused("stopped at commit".into());
        assert_eq!(Ending::of(false, false, &failures), Ending::RefusedCommit);
        assert_eq!(
            failures.first.as_deref(),
            Some("anchors failed"),
            "error is the first message"
        );
        assert_eq!(Ending::of(false, true, &failures), Ending::Budget);
        assert_eq!(Ending::of(true, true, &failures), Ending::Preview);
    }

    /// get_context lists patterns and decisions; an ingested repository's
    /// change records are events. An empty answer points at them instead of
    /// saying the codebase has nothing anywhere.
    #[tokio::test]
    async fn get_context_points_at_the_change_records_it_does_not_list() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        ingest(&storage, repo.request(false)).await;

        let ctx = codebase_call(
            &storage,
            json!({"action": "get_context", "codebase": "demo"}),
        )
        .await;
        assert_eq!(ctx["scope"], "user");
        assert_eq!(ctx["patterns"]["count"], 0);
        assert_eq!(ctx["decisions"]["count"], 0);
        assert_eq!(
            ctx["scopes"],
            json!([
                {"scope": "demo", "patterns": 0, "decisions": 0, "events": 4, "requested": false}
            ]),
            "{ctx}"
        );
        let note = ctx["note"].as_str().expect("an empty answer has a note");
        assert!(!note.contains("in any scope"), "{note}");
        assert!(note.contains("patterns and decisions only"), "{note}");
        assert!(note.contains("in scope 'demo' (4)"), "{note}");
        assert!(
            note.contains(
                "codebase action='verify' codebase='demo' scope='demo' repoPath=<checkout>"
            ),
            "{note}"
        );
        assert!(note.contains("recall handle='commit:<sha>'"), "{note}");

        // reading the scope the records are in still lists none, and says why
        let ctx = codebase_call(
            &storage,
            json!({"action": "get_context", "codebase": "demo", "scope": "demo"}),
        )
        .await;
        assert_eq!(ctx["scopes"][0]["requested"], true, "{ctx}");
        let note = ctx["note"].as_str().unwrap();
        assert!(note.contains("in scope 'demo' (4)"), "{note}");

        // a decision in another scope keeps the existing pointer, events added
        codebase_call(
            &storage,
            json!({"action": "remember_decision", "codebase": "demo", "scope": "other",
                   "decision": "Walk only recorded edges", "rationale": "events note test"}),
        )
        .await;
        let ctx = codebase_call(
            &storage,
            json!({"action": "get_context", "codebase": "demo"}),
        )
        .await;
        let note = ctx["note"].as_str().unwrap();
        assert!(
            note.contains("but other scopes hold some: 'other' (0 patterns, 1 decisions)"),
            "{note}"
        );
        assert!(note.contains("in scope 'demo' (4)"), "{note}");

        // without a codebase no event is counted: most events are not code
        let ctx = codebase_call(&storage, json!({"action": "get_context"})).await;
        let rows = ctx["scopes"].as_array().unwrap();
        assert!(rows.iter().all(|row| row["events"] == 0), "{ctx}");
    }

    /// verify checks at most `limit` memories per type. A sweep that stopped
    /// short says how many it left out instead of reading as the whole scope.
    #[tokio::test]
    async fn verify_says_how_many_it_left_unchecked() {
        let repo = Repo::standard();
        let (storage, _dir) = strata();
        ingest(&storage, repo.request(false)).await;
        let checkout = repo.dir.path().to_str().unwrap();

        for scope in ["demo", " demo "] {
            let report = codebase_call(
                &storage,
                json!({"action": "verify", "codebase": "demo", "scope": scope,
                       "repoPath": checkout, "limit": 2}),
            )
            .await;
            assert_eq!(report["checked"], 2, "{scope:?}: {report}");
            assert_eq!(report["truncated"], true, "{report}");
            assert_eq!(report["totalByType"]["event"], 4, "{report}");
            assert_eq!(
                report["uncheckedByType"],
                json!({"event": 2}),
                "only types with something unchecked: {report}"
            );
            let message = report["message"].as_str().unwrap();
            assert!(
                message.contains("2 more were not checked; raise limit (max 1000)"),
                "{message}"
            );
        }

        let full = verify(&storage, &repo).await;
        assert_eq!(full["checked"], 4, "{full}");
        assert_eq!(full["truncated"], false, "{full}");
        assert_eq!(full["uncheckedByType"], json!({}));
        assert_eq!(
            full["totalByType"],
            json!({"decision": 0, "event": 4, "pattern": 0})
        );
        assert!(
            !full["message"].as_str().unwrap().contains("not checked"),
            "{full}"
        );
    }

    fn commit_node(storage: &Arc<Storage>, scope: &str, sha: &str) -> String {
        let tag = format!("commit:{sha}");
        nodes(storage, scope)
            .into_iter()
            .find(|node| node.tags.iter().any(|have| have == &tag))
            .unwrap_or_else(|| panic!("no commit {sha} in {scope}"))
            .id
    }

    fn touched(storage: &Arc<Storage>) -> Vec<vestige_core::ConnectionRecord> {
        storage
            .get_all_connections()
            .unwrap()
            .into_iter()
            .filter(|edge| edge.link_type == "touched")
            .collect()
    }

    #[test]
    fn hub_bound_is_the_larger_of_25_and_five_percent() {
        assert_eq!(hub_bound(3), 25);
        assert_eq!(hub_bound(26), 25);
        assert_eq!(hub_bound(500), 25);
        assert_eq!(hub_bound(577), 28);
        assert!(!path_is_hub(25, 26));
        assert!(path_is_hub(26, 26));
        assert!(!path_is_hub(28, 577));
        assert!(path_is_hub(29, 577));
    }

    #[test]
    fn name_only_lines_are_exact_paths() {
        let raw = "\u{1e}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\u{1f}2026-01-01T00:00:00+00:00\ncontracts/gov/Token.sol\n\n\u{1e}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\u{1f}2026-01-02T00:00:00+00:00\ncontracts/x.sol\n";
        let parsed = parse_name_only(raw);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].2, vec!["contracts/gov/Token.sol".to_string()]);
        assert_eq!(parsed[1].2, vec!["contracts/x.sol".to_string()]);
    }

    /// A and B share `contracts/x.sol`. C touches only `contracts/y.sol`.
    /// Two other commits share directory segments (`contracts`, `gov`) and
    /// not a path. Only A→B is a `touched` edge, including when B arrives
    /// on a later page. GhostLink bridge then admits that pair.
    #[tokio::test]
    async fn exact_file_path_links_neighbors_and_not_directory_segments() {
        let mut repo = Repo::empty();
        repo.write("contracts/x.sol", "contract X { uint v = 1; }\n");
        repo.commit("A touches x", "2026-04-01T00:00:00+00:00");
        repo.write("contracts/x.sol", "contract X { uint v = 2; }\n");
        repo.commit("B touches x", "2026-04-02T00:00:00+00:00");
        repo.write("contracts/y.sol", "contract Y {}\n");
        repo.commit("C touches y", "2026-04-03T00:00:00+00:00");
        repo.write("contracts/gov/Token.sol", "contract Token {}\n");
        repo.commit("segment contracts/gov", "2026-04-04T00:00:00+00:00");
        repo.write("src/gov/Token.sol", "contract Token {}\n");
        repo.commit("segment gov only", "2026-04-05T00:00:00+00:00");

        let (storage, _dir) = strata();
        let mut preview = repo.request(true);
        preview.codebase = Some("pairs".into());
        preview.scope = Some("pairs".into());
        let preview = ingest(&storage, preview).await;
        assert_eq!(preview["edges"]["wouldRecord"], 1, "{preview}");
        assert_eq!(preview["edges"]["skippedHubPaths"], json!([]), "{preview}");
        assert!(touched(&storage).is_empty(), "a preview wrote an edge");

        let mut only_a = repo.request(false);
        only_a.codebase = Some("pairs".into());
        only_a.scope = Some("pairs".into());
        only_a.limit = Some(1);
        only_a.rev = Some(repo.shas[0].clone());
        let page_a = ingest(&storage, only_a).await;
        assert_eq!(page_a["commits"]["created"], 1, "{page_a}");
        assert_eq!(page_a["edges"]["touched"], 0, "{page_a}");

        let mut only_b = repo.request(false);
        only_b.codebase = Some("pairs".into());
        only_b.scope = Some("pairs".into());
        only_b.limit = Some(1);
        only_b.rev = Some(repo.shas[1].clone());
        let page_b = ingest(&storage, only_b).await;
        assert_eq!(page_b["commits"]["created"], 1, "{page_b}");
        assert_eq!(page_b["edges"]["touched"], 1, "{page_b}");
        assert_eq!(page_b["edges"]["skippedHubPaths"], json!([]), "{page_b}");

        let mut rest = repo.request(false);
        rest.codebase = Some("pairs".into());
        rest.scope = Some("pairs".into());
        let rest = ingest(&storage, rest).await;
        assert_eq!(rest["commits"]["created"], 3, "{rest}");
        assert_eq!(rest["edges"]["touched"], 0, "no new pair: {rest}");

        let id = |sha: &str| commit_node(&storage, "pairs", sha);
        let (a, b, c) = (id(&repo.shas[0]), id(&repo.shas[1]), id(&repo.shas[2]));
        let (gov, other) = (id(&repo.shas[3]), id(&repo.shas[4]));
        let edges = touched(&storage);
        assert_eq!(edges.len(), 1, "{edges:?}");
        assert_eq!(
            edges[0].source_id, a,
            "earlier commit is the source: {edges:?}"
        );
        assert_eq!(edges[0].target_id, b, "{edges:?}");
        assert!(!edges[0].target_id.contains('/'));
        for lone in [&c, &gov, &other] {
            let neighbors = storage.get_connections_for_memory(lone).unwrap();
            assert!(
                neighbors.iter().all(|edge| edge.link_type != "touched"),
                "{lone} must not be linked by a directory segment: {neighbors:?}"
            );
        }

        let again = ingest(&storage, {
            let mut req = repo.request(false);
            req.codebase = Some("pairs".into());
            req.scope = Some("pairs".into());
            req
        })
        .await;
        assert_eq!(again["edges"]["touched"], 0, "{again}");
        assert_eq!(touched(&storage).len(), 1, "re-ingest duplicated the edge");

        let mut other_scope = repo.request(false);
        other_scope.codebase = Some("pairs".into());
        other_scope.scope = Some("other".into());
        let other_out = ingest(&storage, other_scope).await;
        assert_eq!(other_out["edges"]["touched"], 1, "{other_out}");
        let edges = touched(&storage);
        assert_eq!(edges.len(), 2, "{edges:?}");
        for edge in &edges {
            let same = ["pairs", "other"].into_iter().any(|scope| {
                storage.node_is_in_scope(&edge.source_id, scope).unwrap()
                    && storage.node_is_in_scope(&edge.target_id, scope).unwrap()
            });
            assert!(same, "edge crossed scopes: {edge:?}");
        }

        let proposed = crate::strata_memory::ghostlink::propose(
            storage.as_ref(),
            &crate::strata_memory::ghostlink::ProposeRequest {
                lens: crate::strata_memory::ghostlink::Lens::Bridge,
                scope: Some("pairs".into()),
                tags: Vec::new(),
                limit: 10,
                cursor: None,
            },
        )
        .unwrap();
        let candidates = proposed["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 1, "{proposed}");
        let pair = [
            candidates[0]["firstId"].as_str().unwrap(),
            candidates[0]["secondId"].as_str().unwrap(),
        ];
        assert!(
            pair.contains(&a.as_str()) && pair.contains(&b.as_str()),
            "{proposed}"
        );
        assert_eq!(candidates[0]["hops"], 1, "{proposed}");
    }

    /// A file every commit touches is a hub once it passes max(25, 5%).
    /// The chain for that path is not written. A path only two commits
    /// share still is.
    #[tokio::test]
    async fn a_hub_file_touched_by_every_commit_records_no_edges() {
        let mut repo = Repo::empty();
        for i in 0..26 {
            repo.write(
                "contracts/Hub.sol",
                &format!("contract Hub {{ uint v = {i}; }}\n"),
            );
            if i < 2 {
                repo.write(
                    "contracts/x.sol",
                    &format!("contract X {{ uint v = {i}; }}\n"),
                );
            }
            if i == 2 {
                repo.write("contracts/y.sol", "contract Y {}\n");
            }
            let date = format!("2026-05-{:02}T00:00:00+00:00", i + 1);
            repo.commit(&format!("hub {i}"), &date);
        }
        let (storage, _dir) = strata();
        let mut req = repo.request(false);
        req.codebase = Some("hub".into());
        req.scope = Some("hub".into());
        let out = ingest(&storage, req).await;
        assert_eq!(out["edges"]["commitsInScope"], 26, "{out}");
        assert_eq!(out["edges"]["hubBound"], 25, "{out}");
        assert_eq!(out["edges"]["touched"], 1, "{out}");
        assert_eq!(
            out["edges"]["skippedHubPaths"],
            json!(["contracts/Hub.sol"]),
            "{out}"
        );
        let edges = touched(&storage);
        assert_eq!(edges.len(), 1, "{edges:?}");
        let a = commit_node(&storage, "hub", &repo.shas[0]);
        let b = commit_node(&storage, "hub", &repo.shas[1]);
        let c = commit_node(&storage, "hub", &repo.shas[2]);
        assert_eq!(edges[0].source_id, a);
        assert_eq!(edges[0].target_id, b);
        assert!(
            storage
                .get_connections_for_memory(&c)
                .unwrap()
                .iter()
                .all(|edge| edge.link_type != "touched"),
            "the y.sol commit is not on the hub chain"
        );
        let again = ingest(&storage, {
            let mut req = repo.request(false);
            req.codebase = Some("hub".into());
            req.scope = Some("hub".into());
            req
        })
        .await;
        assert_eq!(again["edges"]["touched"], 0, "{again}");
        assert_eq!(touched(&storage).len(), 1);

        let proposed = crate::strata_memory::ghostlink::propose(
            storage.as_ref(),
            &crate::strata_memory::ghostlink::ProposeRequest {
                lens: crate::strata_memory::ghostlink::Lens::Bridge,
                scope: Some("hub".into()),
                tags: Vec::new(),
                limit: 10,
                cursor: None,
            },
        )
        .unwrap();
        assert_eq!(
            proposed["candidates"].as_array().unwrap().len(),
            1,
            "hub must not admit the clique: {proposed}"
        );
    }
}
