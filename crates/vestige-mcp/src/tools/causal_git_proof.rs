//! Real-repo proof that `causal_walk` reaches a ground-truth cause from
//! `ingest_repo` plus one failure memory, or misses it, with no hand-written
//! cause link.
//!
//! Not part of CI (the clones are local and the walk is slow). Run:
//!
//! ```text
//! cargo test -p vestige-mcp --lib real_repos -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Clones live under `VESTIGE_PROOF_REPOS` (default `/tmp/vestige-proof`).
//! The cause SHA is an oracle for scoring the walk. It is never written as an
//! edge, a tag, or a start point. `VESTIGE_PROOF_ONLY=name,name` runs a subset.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{Value, json};
use vestige_core::{ConnectionRecord, IngestInput, Storage};

use super::causal_walk;
use super::repo_ingest::{self, Request};

const SCAN_LIMIT: i32 = 2000;
const WRITE_BUDGET: Duration = Duration::from_secs(300);
const INGEST_ROUNDS: usize = 8;

#[derive(Clone, Copy)]
enum Expect {
    /// Ground-truth cause is this 1-based rank.
    Rank(usize),
    /// Cause was ingested and is still not reachable from the reported frame.
    Miss,
}

struct Case {
    name: &'static str,
    repo: &'static str,
    failure_sha: &'static str,
    cause_sha: &'static str,
    symptom: &'static str,
    pages: &'static [(&'static str, usize)],
    expect: Expect,
    /// A `corrects` edge must exist onto the cause (git revert trailer).
    expect_revert: bool,
    note: &'static str,
    starts: fn(&str) -> Value,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "tokio-6714",
            repo: "tokio",
            failure_sha: "48e35c11d924ffa3a009b89fbb6d36e57b835da4",
            cause_sha: "8480a180e6ffdd0ec0ec213a9bebcda2445fc541",
            symptom: "failure: tokio 1.39.0 causes quinn to panic. thread 'tests::stream_id_flow_control' panicked at /home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs:123:9: assertion `left != right` failed",
            pages: &[("47210a8e6eeb82b51aa778074fdc4d757b953b8c", 100)],
            expect: Expect::Rank(1),
            expect_revert: true,
            note: "issue #6714 panic path, observed at tokio-1.39.0. Blame of line 123 is not the cause; the cause is the commit the later git revert names.",
            starts: tokio_starts,
        },
        Case {
            name: "cargo-10682",
            repo: "cargo",
            failure_sha: "a028ae42fc1376571de836be702e840ca8e060c2",
            cause_sha: "f12f0256caedaeada2fecd8bc5e93feee7abb5b7",
            symptom: "failure: cargo 1.61.0 (a028ae42fc1376571de836be702e840ca8e060c2) local-registry missing-crate error regressed from 1.60. Issue cites src/cargo/sources/registry/local.rs#L57-L60.",
            pages: &[("a028ae42fc1376571de836be702e840ca8e060c2", 90)],
            expect: Expect::Rank(4),
            expect_revert: false,
            note: "issue #10682 cites local.rs L57-L60. Line 57 is blamed to a later refactor, and the cause's hunks start at line 58, so the cause is found by the file touch and ranks after the blamed line.",
            starts: cargo_starts,
        },
        Case {
            name: "uv-10186",
            repo: "uv",
            failure_sha: "351d602d86c484a39bc537f1eb99866ea2c25fc1",
            cause_sha: "d2f58d92991fa08b24596fcc6c6472dc5015d3bc",
            symptom: "failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
            pages: &[("b52d48973fe9ddb2e78b663ec48a1a68f7e7802d", 20)],
            expect: Expect::Rank(1),
            expect_revert: true,
            note: "issue #10186 has no source path. Start is the CI run URL. Failure revision is the 0.5.12 tag. The cause touched only Cargo.lock; the kept path is the lockfile touched edge, and the commit was git-reverted.",
            starts: uv_starts,
        },
        Case {
            name: "prometheus-mempostings",
            repo: "prometheus",
            failure_sha: "9700933d18954e82155f1abed712707d66027b1e",
            cause_sha: "50ef0dc954592666a13ff92ef20811f0127c3b49",
            symptom: "failure: Memory allocation goes so high in Prombench that the system is unusable.",
            pages: &[
                ("2fbbfc3da800d3b33c7f7b430e403f66b781b962", 1),
                ("9700933d18954e82155f1abed712707d66027b1e", 8),
            ],
            expect: Expect::Miss,
            expect_revert: true,
            note: "revert message names Prombench and no file. This page ingests the revert before the failure revision, so the parent edge is not recorded yet, and the shallow clone does not connect the cause to that revision. The follow-up proof ingests the failure revision first and ranks the cause via corrects.",
            starts: logged_only,
        },
        Case {
            name: "kubernetes-windows-kubelet",
            repo: "kubernetes",
            failure_sha: "9edcffcde5595e8a5b1a35f88c421764e575afce",
            cause_sha: "4060ee60c1d2e5ba1fba1f8729adfc211cee1b6f",
            symptom: "failure: kubelet fails to restart on Windows since v1.31.0. E0828 server.go:102 Failed to listen to socket while starting device plugin registry. E0828 kubelet.go:1566 Failed to start ContainerManager. listen unix kubelet.sock bind.",
            pages: &[
                ("3ca610757ed3e38f8a0a58798a0bdb23aebcc6bb", 1),
                ("9edcffcde5595e8a5b1a35f88c421764e575afce", 8),
            ],
            expect: Expect::Rank(1),
            expect_revert: true,
            note: "issue #126965 logs only basenames server.go:102 and kubelet.go:1566. A bare file name is not a path. The revert is on another branch from v1.31.0; the cause is an ancestor of v1.31.0, so the recorded corrects edge is walked and ranks first.",
            starts: kubelet_starts,
        },
        Case {
            name: "grafana-min-step",
            repo: "grafana",
            failure_sha: "4c8d14014dd8c40fd9f41df83c3ccaf107b0d346",
            cause_sha: "7ae2eed876fce88281ef88ba1e7edca97cfdfde6",
            symptom: "failure: Having the minStep take precedence introduced breaking changes for existing dashboards that were using minStep with the previous logic when their series returned a lot of data points.",
            pages: &[
                ("03d43277131d4310f2de02efcf81d168b132d273", 1),
                ("4c8d14014dd8c40fd9f41df83c3ccaf107b0d346", 8),
            ],
            expect: Expect::Miss,
            expect_revert: true,
            note: "revert describes the dashboard break and names no file. This page ingests the revert before the failure revision, so the parent edge is not recorded yet, and the shallow clone does not connect the cause to that revision. The follow-up proof ingests the failure revision first and ranks the cause via corrects.",
            starts: logged_only,
        },
    ]
}

fn tokio_starts(id: &str) -> Value {
    json!([{
        "kind": "stack_frame",
        "frame": "/home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs:123:9",
        "node_id": id,
    }])
}

fn cargo_starts(id: &str) -> Value {
    json!([{
        "kind": "stack_frame",
        "frame": "src/cargo/sources/registry/local.rs:57",
        "node_id": id,
    }])
}

fn uv_starts(id: &str) -> Value {
    json!([{
        "kind": "ci_run",
        "run_id": "https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
        "node_id": id,
    }])
}

fn kubelet_starts(id: &str) -> Value {
    json!([
        {"kind": "stack_frame", "frame": "server.go:102", "node_id": id},
        {"kind": "stack_frame", "frame": "kubelet.go:1566", "node_id": id},
    ])
}

fn logged_only(id: &str) -> Value {
    json!([{"kind": "logged_write", "node_id": id}])
}

fn repos_root() -> PathBuf {
    std::env::var_os("VESTIGE_PROOF_REPOS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/vestige-proof"))
}

fn commit_id(storage: &Arc<Storage>, scope: &str, sha: &str) -> Option<String> {
    let tag = repo_ingest::commit_tag(sha);
    storage
        .current_code_context_nodes("event", Some(&tag), scope, 5)
        .ok()?
        .into_iter()
        .next()
        .map(|node| node.id)
}

struct Report {
    name: String,
    ingested: bool,
    revert_edge: bool,
    rank: Option<usize>,
    cause_count: usize,
    truncated: bool,
    empty_because: String,
    top: String,
    cause_row: String,
    failure_links: String,
    ingest: String,
    note: String,
    passed: bool,
    why: String,
}

async fn ingest_pages(
    storage: &Arc<Storage>,
    repo: &Path,
    scope: &str,
    pages: &[(&str, usize)],
) -> Result<String, String> {
    let mut lines = Vec::new();
    for (rev, limit) in pages {
        let mut done = false;
        for round in 1..=INGEST_ROUNDS {
            let started = Instant::now();
            let out = repo_ingest::execute(
                storage,
                Request {
                    repo_path: repo.to_path_buf(),
                    codebase: Some(scope.to_string()),
                    scope: Some(scope.to_string()),
                    rev: Some((*rev).to_string()),
                    since: None,
                    until: None,
                    limit: Some(*limit),
                    dry_run: false,
                    budget: Some(WRITE_BUDGET),
                },
            )
            .await?;
            if let Some(err) = out["error"].as_str() {
                return Err(format!("{scope} rev {rev} ingest error: {err}"));
            }
            let remaining = out["commits"]["remaining"].as_u64().unwrap_or(1);
            let stopped = out["commits"]["stoppedByBudget"].as_bool().unwrap_or(true);
            let line = format!(
                "round {round} rev {} created={} remaining={remaining} stopped={stopped} pulled={} edges={} in {}s",
                &rev[..12],
                out["commits"]["created"],
                out["commits"]["pulledReverts"],
                out["edges"],
                started.elapsed().as_secs(),
            );
            eprintln!("  {scope} {line}");
            lines.push(line);
            if remaining == 0 && !stopped {
                done = true;
                break;
            }
        }
        if !done {
            return Err(format!(
                "{scope} rev {rev} did not finish within {INGEST_ROUNDS} rounds"
            ));
        }
    }
    Ok(lines.join(" | "))
}

fn link_failure(
    storage: &Arc<Storage>,
    scope: &str,
    symptom: &str,
    revision_id: &str,
) -> Result<String, String> {
    let node = storage
        .ingest_in_scope(
            IngestInput {
                content: symptom.to_string(),
                tags: vec!["failure".to_string()],
                ..Default::default()
            },
            scope,
        )
        .map_err(|err| err.to_string())?;
    let now = Utc::now();
    storage
        .save_connection(&ConnectionRecord {
            source_id: node.id.clone(),
            target_id: revision_id.to_string(),
            strength: 1.0,
            link_type: "derived_from".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .map_err(|err| err.to_string())?;
    Ok(node.id)
}

fn failure_links(storage: &Arc<Storage>, failure_id: &str) -> String {
    let Ok(edges) = storage.get_connections_for_memory(failure_id) else {
        return "unreadable".into();
    };
    edges
        .iter()
        .map(|edge| {
            format!(
                "{} -{}-> {}",
                edge.source_id, edge.link_type, edge.target_id
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn has_revert_edge(storage: &Arc<Storage>, cause_id: &str) -> bool {
    storage
        .get_connections_for_memory(cause_id)
        .map(|edges| {
            edges
                .iter()
                .any(|edge| edge.link_type == "corrects" && edge.target_id == cause_id)
        })
        .unwrap_or(false)
}

fn row_brief(cause: &Value) -> String {
    let sha = cause["structure"]["sha"].as_str().unwrap_or("?");
    let short = &sha[..sha.len().min(12)];
    format!(
        "{short} reverted={} blame={} hunk={} hop={} edge={}",
        cause["structure"]["revertedAfterFailure"],
        cause["structure"]["blameOfLine"],
        cause["structure"]["touchedFailingHunk"],
        cause["structure"]["hop"],
        cause["structure"]["edge"].as_str().unwrap_or("?"),
    )
}

fn only_derived_from_revision(
    storage: &Arc<Storage>,
    failure_id: &str,
    revision_id: &str,
    cause_id: &str,
) -> Result<(), String> {
    let edges = storage
        .get_connections_for_memory(failure_id)
        .map_err(|err| err.to_string())?;
    if edges.len() != 1 {
        return Err(format!(
            "failure has {} edges, want exactly one derived_from the observed revision",
            edges.len()
        ));
    }
    let edge = &edges[0];
    if edge.link_type != "derived_from"
        || edge.source_id != failure_id
        || edge.target_id != revision_id
    {
        return Err(format!(
            "failure link is {} -{}-> {}, want derived_from {revision_id}",
            edge.source_id, edge.link_type, edge.target_id
        ));
    }
    if edge.target_id == cause_id {
        return Err("failure is linked at the cause; that is a hand-written cause link".into());
    }
    Ok(())
}

async fn run_case(case: &Case, root: &Path) -> Report {
    let repo = root.join(case.repo);
    let dir = tempfile::tempdir().expect("tempdir");
    let storage = strata_open(dir.path());
    let scope = case.name;
    eprintln!("CASE {} ingest {}", case.name, repo.display());
    let ingest = match ingest_pages(&storage, &repo, scope, case.pages).await {
        Ok(summary) => summary,
        Err(why) => {
            return Report {
                name: case.name.into(),
                ingested: false,
                revert_edge: false,
                rank: None,
                cause_count: 0,
                truncated: false,
                empty_because: String::new(),
                top: String::new(),
                cause_row: String::new(),
                failure_links: String::new(),
                ingest: String::new(),
                note: case.note.into(),
                passed: false,
                why,
            };
        }
    };
    let Some(revision_id) = commit_id(&storage, scope, case.failure_sha) else {
        return Report {
            name: case.name.into(),
            ingested: false,
            revert_edge: false,
            rank: None,
            cause_count: 0,
            truncated: false,
            empty_because: String::new(),
            top: String::new(),
            cause_row: String::new(),
            failure_links: String::new(),
            ingest,
            note: case.note.into(),
            passed: false,
            why: format!("failure revision {} was not ingested", case.failure_sha),
        };
    };
    let Some(cause_id) = commit_id(&storage, scope, case.cause_sha) else {
        return Report {
            name: case.name.into(),
            ingested: false,
            revert_edge: false,
            rank: None,
            cause_count: 0,
            truncated: false,
            empty_because: String::new(),
            top: String::new(),
            cause_row: String::new(),
            failure_links: String::new(),
            ingest,
            note: case.note.into(),
            passed: false,
            why: format!("cause {} was not ingested", case.cause_sha),
        };
    };
    let failure_id = match link_failure(&storage, scope, case.symptom, &revision_id) {
        Ok(id) => id,
        Err(why) => {
            return Report {
                name: case.name.into(),
                ingested: true,
                revert_edge: has_revert_edge(&storage, &cause_id),
                rank: None,
                cause_count: 0,
                truncated: false,
                empty_because: String::new(),
                top: String::new(),
                cause_row: String::new(),
                failure_links: String::new(),
                ingest,
                note: case.note.into(),
                passed: false,
                why,
            };
        }
    };
    if let Err(why) = only_derived_from_revision(&storage, &failure_id, &revision_id, &cause_id) {
        return Report {
            name: case.name.into(),
            ingested: true,
            revert_edge: has_revert_edge(&storage, &cause_id),
            rank: None,
            cause_count: 0,
            truncated: false,
            empty_because: String::new(),
            top: String::new(),
            cause_row: String::new(),
            failure_links: failure_links(&storage, &failure_id),
            ingest,
            note: case.note.into(),
            passed: false,
            why,
        };
    }
    let walk = match causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "scan_limit": SCAN_LIMIT,
            "start_points": (case.starts)(&failure_id),
        })),
    )
    .await
    {
        Ok(value) => value,
        Err(why) => {
            return Report {
                name: case.name.into(),
                ingested: true,
                revert_edge: has_revert_edge(&storage, &cause_id),
                rank: None,
                cause_count: 0,
                truncated: false,
                empty_because: String::new(),
                top: String::new(),
                cause_row: String::new(),
                failure_links: failure_links(&storage, &failure_id),
                ingest,
                note: case.note.into(),
                passed: false,
                why: format!("causal_walk: {why}"),
            };
        }
    };
    let causes = walk["causes"].as_array().cloned().unwrap_or_default();
    let rank = causes.iter().position(|cause| {
        cause["structure"]["sha"].as_str() == Some(case.cause_sha)
            || cause["id"].as_str() == Some(cause_id.as_str())
    });
    let revert_edge = has_revert_edge(&storage, &cause_id);
    let top = causes
        .iter()
        .take(8)
        .map(row_brief)
        .collect::<Vec<_>>()
        .join(" || ");
    let cause_row = rank
        .and_then(|index| causes.get(index))
        .map(row_brief)
        .unwrap_or_else(|| "absent".into());
    let mut why = String::new();
    let mut passed = true;
    match case.expect {
        Expect::Rank(want) => {
            if rank != Some(want - 1) {
                passed = false;
                why = format!(
                    "want rank {want}, got {}",
                    rank.map(|index| index + 1)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "miss".into())
                );
            }
        }
        Expect::Miss => {
            if let Some(index) = rank {
                passed = false;
                why = format!("expected a miss, found at rank {}", index + 1);
            }
        }
    }
    if case.expect_revert && !revert_edge {
        passed = false;
        let extra = "git revert trailer did not record a corrects edge onto the cause";
        if why.is_empty() {
            why = extra.into();
        } else {
            why = format!("{why}; {extra}");
        }
    }
    Report {
        name: case.name.into(),
        ingested: true,
        revert_edge,
        rank: rank.map(|index| index + 1),
        cause_count: causes.len(),
        truncated: walk["truncated"].as_bool().unwrap_or(false),
        empty_because: walk["emptyBecause"].as_str().unwrap_or("").to_string(),
        top,
        cause_row,
        failure_links: failure_links(&storage, &failure_id),
        ingest,
        note: case.note.into(),
        passed,
        why,
    }
}

fn strata_open(dir: &Path) -> Arc<Storage> {
    crate::strata_memory::open(dir).expect("open strata log")
}

fn selected(name: &str) -> bool {
    match std::env::var("VESTIGE_PROOF_ONLY") {
        Ok(raw) if !raw.trim().is_empty() => raw.split(',').any(|item| item.trim() == name),
        _ => true,
    }
}

fn render(report: &Report) -> String {
    format!(
        "PROOF {name} passed={passed} ingested={ingested} revertEdge={revert} rank={rank} of {count} truncated={truncated} empty={empty}\n  cause: {cause}\n  top: {top}\n  failureLinks: {links}\n  ingest: {ingest}\n  note: {note}\n  why: {why}",
        name = report.name,
        passed = report.passed,
        ingested = report.ingested,
        revert = report.revert_edge,
        rank = report
            .rank
            .map(|n| n.to_string())
            .unwrap_or_else(|| "miss".into()),
        count = report.cause_count,
        truncated = report.truncated,
        empty = if report.empty_because.is_empty() {
            "-"
        } else {
            report.empty_because.as_str()
        },
        cause = report.cause_row,
        top = if report.top.is_empty() {
            "-"
        } else {
            report.top.as_str()
        },
        links = report.failure_links,
        ingest = report.ingest,
        note = report.note,
        why = if report.why.is_empty() {
            "-"
        } else {
            report.why.as_str()
        },
    )
}

#[tokio::test]
#[ignore = "needs local clones under VESTIGE_PROOF_REPOS; not a CI test"]
async fn real_repos_rank_or_miss_without_hand_written_cause_links() {
    let root = repos_root();
    let mut rendered = Vec::new();
    let mut failed = Vec::new();
    for case in cases() {
        if !selected(case.name) {
            continue;
        }
        let repo = root.join(case.repo);
        if !repo.join(".git").is_dir() {
            let why = format!("missing clone {}", repo.display());
            eprintln!("PROOF {} FAIL {why}", case.name);
            failed.push(format!("{}: {why}", case.name));
            continue;
        }
        let report = run_case(&case, &root).await;
        let text = render(&report);
        eprintln!("{text}\n");
        if !report.passed {
            failed.push(format!("{}: {}", report.name, report.why));
        }
        rendered.push(text);
    }
    let blob = rendered.join("\n\n");
    let out = root.join("proof-report.txt");
    if let Err(err) = std::fs::write(&out, &blob) {
        eprintln!("could not write {}: {err}", out.display());
    }
    assert!(
        failed.is_empty(),
        "causal proof failed:\n{}\n\n{blob}",
        failed.join("\n")
    );
}

/// Lockfile bump, first-parent range, and revert/patch-id reachability.
/// Same inputs as the rank proof: `ingest_repo` plus one failure memory
/// `derived_from` the observed revision. The tokio case also passes the
/// good..bad range the regression is bounded by. No cause link is written.
#[tokio::test]
#[ignore = "needs local clones under VESTIGE_PROOF_REPOS; not a CI test"]
async fn real_repos_followup_lockfile_range_and_revert() {
    let root = repos_root();
    let acf = "acf8a7da7a64bf08d578db9a9836a8e061765314";
    let mut rendered = Vec::new();
    let mut failed = Vec::new();
    for case in followup_cases() {
        if !selected(case.name) {
            continue;
        }
        let repo = root.join(case.repo);
        if !repo.join(".git").is_dir() {
            let why = format!("missing clone {}", repo.display());
            eprintln!("PROOF {} FAIL {why}", case.name);
            failed.push(format!("{}: {why}", case.name));
            continue;
        }
        let report = run_followup(&case, &root, acf).await;
        let text = report.render();
        eprintln!("{text}\n");
        if !report.passed {
            failed.push(format!("{}: {}", report.name, report.why));
        }
        rendered.push(text);
    }
    let blob = rendered.join("\n\n");
    let out = root.join("followup-proof-report.txt");
    if let Err(err) = std::fs::write(&out, &blob) {
        eprintln!("could not write {}: {err}", out.display());
    }
    assert!(
        failed.is_empty(),
        "follow-up proof failed:\n{}\n\n{blob}",
        failed.join("\n")
    );
}

struct FollowCase {
    name: &'static str,
    repo: &'static str,
    failure_sha: &'static str,
    cause_sha: &'static str,
    symptom: &'static str,
    pages: &'static [(&'static str, usize)],
    note: &'static str,
    signal: Signal,
}

enum Signal {
    /// Rank 1 through a lockfile `touched` edge for this package.
    Lockfile { package: &'static str },
    /// Cause stays; this full SHA is removed by `worked_in..broke_in`.
    Range {
        worked_in: &'static str,
        broke_in: &'static str,
        frame: &'static str,
    },
    /// Last hop is the recorded `corrects` edge (revert trailer or patch-id).
    Revert,
}

fn followup_cases() -> Vec<FollowCase> {
    vec![
        FollowCase {
            name: "uv-10186-lockfile",
            repo: "uv",
            failure_sha: "351d602d86c484a39bc537f1eb99866ea2c25fc1",
            cause_sha: "d2f58d92991fa08b24596fcc6c6472dc5015d3bc",
            symptom: "failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
            pages: &[("b52d48973fe9ddb2e78b663ec48a1a68f7e7802d", 20)],
            note: "no source path in the issue. The cause commit changes only Cargo.lock. Rank is the lockfile touched edge, not the prose.",
            signal: Signal::Lockfile { package: "reqwest" },
        },
        FollowCase {
            name: "tokio-6714-range",
            repo: "tokio",
            failure_sha: "48e35c11d924ffa3a009b89fbb6d36e57b835da4",
            cause_sha: "8480a180e6ffdd0ec0ec213a9bebcda2445fc541",
            symptom: "failure: tokio 1.39.0 causes quinn to panic. thread 'tests::stream_id_flow_control' panicked at /home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs:123:9: assertion `left != right` failed",
            pages: &[
                ("47210a8e6eeb82b51aa778074fdc4d757b953b8c", 100),
                ("acf8a7da7a64bf08d578db9a9836a8e061765314", 1),
            ],
            note: "panic path plus tokio-1.38.0..tokio-1.39.0. The 2020 commit that also touched linked_list.rs is ingested so the range can exclude it.",
            signal: Signal::Range {
                worked_in: "tokio-1.38.0",
                broke_in: "tokio-1.39.0",
                frame: "/home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs:123:9",
            },
        },
        FollowCase {
            name: "prometheus-mempostings-revert",
            repo: "prometheus",
            failure_sha: "9700933d18954e82155f1abed712707d66027b1e",
            cause_sha: "50ef0dc954592666a13ff92ef20811f0127c3b49",
            symptom: "failure: Memory allocation goes so high in Prombench that the system is unusable.",
            pages: &[
                ("9700933d18954e82155f1abed712707d66027b1e", 8),
                ("2fbbfc3da800d3b33c7f7b430e403f66b781b962", 1),
            ],
            note: "failure revision is the revert's parent. It is ingested first so the parent edge is recorded, then the child jump follows corrects.",
            signal: Signal::Revert,
        },
        FollowCase {
            name: "kubernetes-windows-kubelet-revert",
            repo: "kubernetes",
            failure_sha: "9edcffcde5595e8a5b1a35f88c421764e575afce",
            cause_sha: "4060ee60c1d2e5ba1fba1f8729adfc211cee1b6f",
            symptom: "failure: kubelet fails to restart on Windows since v1.31.0. E0828 server.go:102 Failed to listen to socket while starting device plugin registry. E0828 kubelet.go:1566 Failed to start ContainerManager. listen unix kubelet.sock bind.",
            pages: &[
                ("3ca610757ed3e38f8a0a58798a0bdb23aebcc6bb", 1),
                ("9edcffcde5595e8a5b1a35f88c421764e575afce", 8),
            ],
            note: "revert is not a descendant of v1.31.0. The cause is an ancestor, so the corrects edge is added without the parent chain.",
            signal: Signal::Revert,
        },
        FollowCase {
            name: "grafana-min-step-revert",
            repo: "grafana",
            failure_sha: "4c8d14014dd8c40fd9f41df83c3ccaf107b0d346",
            cause_sha: "7ae2eed876fce88281ef88ba1e7edca97cfdfde6",
            symptom: "failure: Having the minStep take precedence introduced breaking changes for existing dashboards that were using minStep with the previous logic when their series returned a lot of data points.",
            pages: &[
                ("4c8d14014dd8c40fd9f41df83c3ccaf107b0d346", 8),
                ("03d43277131d4310f2de02efcf81d168b132d273", 1),
            ],
            note: "failure revision is the revert's parent. It is ingested first so the parent edge is recorded, then the child jump follows corrects.",
            signal: Signal::Revert,
        },
    ]
}

struct FollowReport {
    name: String,
    passed: bool,
    why: String,
    text: String,
}

impl FollowReport {
    fn render(&self) -> String {
        self.text.clone()
    }
}

async fn run_followup(case: &FollowCase, root: &Path, excluded_sha: &str) -> FollowReport {
    let fail = |why: String| FollowReport {
        name: case.name.into(),
        passed: false,
        why: why.clone(),
        text: format!("PROOF {} FAIL {why}", case.name),
    };
    let repo = root.join(case.repo);
    let dir = tempfile::tempdir().expect("tempdir");
    let storage = strata_open(dir.path());
    let ingest = match ingest_pages(&storage, &repo, case.name, case.pages).await {
        Ok(summary) => summary,
        Err(why) => return fail(why),
    };
    let Some(revision_id) = commit_id(&storage, case.name, case.failure_sha) else {
        return fail(format!(
            "failure revision {} was not ingested",
            case.failure_sha
        ));
    };
    let Some(cause_id) = commit_id(&storage, case.name, case.cause_sha) else {
        return fail(format!("cause {} was not ingested", case.cause_sha));
    };
    let failure_id = match link_failure(&storage, case.name, case.symptom, &revision_id) {
        Ok(id) => id,
        Err(why) => return fail(why),
    };
    if let Err(why) = only_derived_from_revision(&storage, &failure_id, &revision_id, &cause_id) {
        return fail(why);
    }
    let starts = match case.signal {
        Signal::Lockfile { .. } => json!([{
            "kind": "ci_run",
            "run_id": "https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
            "node_id": failure_id,
        }]),
        Signal::Range {
            worked_in,
            broke_in,
            frame,
        } => json!([
            {"kind": "stack_frame", "frame": frame, "node_id": failure_id},
            {
                "kind": "version_range",
                "worked_in": worked_in,
                "broke_in": broke_in,
                "repo": repo.display().to_string(),
            }
        ]),
        Signal::Revert => json!([{"kind": "logged_write", "node_id": failure_id}]),
    };
    let walk = match causal_walk::execute(
        &storage,
        Some(json!({
            "scope": case.name,
            "scan_limit": SCAN_LIMIT,
            "start_points": starts,
        })),
    )
    .await
    {
        Ok(value) => value,
        Err(why) => return fail(format!("causal_walk: {why}")),
    };
    let causes = walk["causes"].as_array().cloned().unwrap_or_default();
    let rank = causes.iter().position(|cause| {
        cause["structure"]["sha"].as_str() == Some(case.cause_sha)
            || cause["id"].as_str() == Some(cause_id.as_str())
    });
    let cause = rank.and_then(|index| causes.get(index));
    let edge = cause
        .and_then(|row| row["structure"]["edge"].as_str())
        .unwrap_or("");
    let package = cause
        .and_then(|row| row["structure"]["lockfile"]["package"].as_str())
        .unwrap_or("");
    let brief = cause.map(row_brief).unwrap_or_else(|| "absent".into());
    let top = causes
        .iter()
        .take(6)
        .map(row_brief)
        .collect::<Vec<_>>()
        .join(" || ");
    let mut passed = true;
    let mut why = String::new();
    let mut extra = String::new();
    match case.signal {
        Signal::Lockfile { package: want } => {
            if rank != Some(0) {
                passed = false;
                why = format!(
                    "want rank 1 via lockfile, got {}",
                    rank.map(|index| (index + 1).to_string())
                        .unwrap_or_else(|| "miss".into())
                );
            }
            if edge != "touched" || package != want {
                passed = false;
                let detail = format!(
                    "want edge touched lockfile.package={want}, got edge={edge} package={package}"
                );
                if why.is_empty() {
                    why = detail;
                } else {
                    why = format!("{why}; {detail}");
                }
            }
        }
        Signal::Range { .. } => {
            if rank.is_none() {
                passed = false;
                why = "cause was not in the causes after the range filter".into();
            }
            let shas = walk["range"]["excludedShas"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let excluded = shas.iter().any(|sha| sha.as_str() == Some(excluded_sha));
            if !excluded {
                passed = false;
                let detail = format!("excludedShas does not contain {excluded_sha}");
                if why.is_empty() {
                    why = detail;
                } else {
                    why = format!("{why}; {detail}");
                }
            }
            if causes
                .iter()
                .any(|row| row["structure"]["sha"].as_str() == Some(excluded_sha))
            {
                passed = false;
                let detail = "the 2020 blame commit is still a cause".to_string();
                if why.is_empty() {
                    why = detail;
                } else {
                    why = format!("{why}; {detail}");
                }
            }
            extra = format!(
                " range.excluded={} excludedSha={} present={} range.because={}",
                walk["range"]["excluded"],
                excluded_sha,
                excluded,
                walk["range"]["because"].as_str().unwrap_or("")
            );
        }
        Signal::Revert => {
            if rank.is_none() || edge != "corrects" {
                passed = false;
                why = format!(
                    "want corrects, got {} edge={edge}",
                    rank.map(|index| format!("rank {}", index + 1))
                        .unwrap_or_else(|| "miss".into())
                );
            }
        }
    }
    let rank_label = rank
        .map(|index| (index + 1).to_string())
        .unwrap_or_else(|| "miss".into());
    let text = format!(
        "PROOF {name} passed={passed} rank={rank_label} of {count} edge={edge} pkg={pkg}{extra}\n  cause: {brief}\n  top: {top}\n  ingest: {ingest}\n  note: {note}\n  why: {why_text}",
        name = case.name,
        count = causes.len(),
        pkg = if package.is_empty() { "-" } else { package },
        top = if top.is_empty() { "-" } else { top.as_str() },
        note = case.note,
        why_text = if why.is_empty() { "-" } else { why.as_str() },
    );
    FollowReport {
        name: case.name.into(),
        passed,
        why,
        text,
    }
}
