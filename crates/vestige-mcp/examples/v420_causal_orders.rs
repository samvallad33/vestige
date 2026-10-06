//! Provisional runner for the six real-repo causal walks.
//!
//! Each case is `ingest_repo` plus one failure memory tagged `failure`,
//! linked `derived_from` only the revision where the failure was observed.
//! The cause SHA is an oracle used to score the printed rank. It is not
//! written as an edge, a tag, or a start point. No version range is stored
//! on the failure record.
//!
//! ```text
//! cargo run -p vestige-mcp --example v420_causal_orders --release -- --root /tmp/vestige-proof
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{Value, json};
use vestige_core::{ConnectionRecord, IngestInput, Storage};
use vestige_mcp::strata_memory;
use vestige_mcp::tools::{causal_walk, repo_ingest};

const WRITE_BUDGET: Duration = Duration::from_secs(300);
const INGEST_ROUNDS: usize = 8;
const SCAN_LIMIT: i32 = 2000;

struct Case {
    name: &'static str,
    repo: &'static str,
    failure_sha: &'static str,
    cause_sha: &'static str,
    symptom: &'static str,
    pages: &'static [(&'static str, usize)],
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
            starts: tokio_starts,
        },
        Case {
            name: "cargo-10682",
            repo: "cargo",
            failure_sha: "a028ae42fc1376571de836be702e840ca8e060c2",
            cause_sha: "f12f0256caedaeada2fecd8bc5e93feee7abb5b7",
            symptom: "failure: cargo 1.61.0 (a028ae42fc1376571de836be702e840ca8e060c2) local-registry missing-crate error regressed from 1.60. Issue cites src/cargo/sources/registry/local.rs#L57-L60.",
            pages: &[("a028ae42fc1376571de836be702e840ca8e060c2", 90)],
            starts: cargo_starts,
        },
        Case {
            name: "uv-10186",
            repo: "uv",
            failure_sha: "351d602d86c484a39bc537f1eb99866ea2c25fc1",
            cause_sha: "d2f58d92991fa08b24596fcc6c6472dc5015d3bc",
            symptom: "failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
            pages: &[("b52d48973fe9ddb2e78b663ec48a1a68f7e7802d", 20)],
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

struct Opts {
    root: PathBuf,
    only: Option<String>,
    fair_uv: bool,
}

fn opts() -> Opts {
    let mut args = std::env::args().skip(1);
    let mut root = PathBuf::from("/tmp/vestige-proof");
    let mut only: Option<String> = None;
    let mut fair_uv = std::env::var("VESTIGE_PROOF_FAIR_UV").ok().as_deref() == Some("1");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                root = PathBuf::from(args.next().expect("--root needs a directory"));
            }
            "--only" => {
                only = Some(args.next().expect("--only needs a name"));
            }
            "--fair-uv" => fair_uv = true,
            other => eprintln!("ignoring {other}"),
        }
    }
    Opts { root, only, fair_uv }
}

fn selected(only: &Option<String>, name: &str) -> bool {
    match only {
        Some(raw) if !raw.trim().is_empty() => raw.split(',').any(|item| item.trim() == name),
        _ => true,
    }
}

#[tokio::main]
async fn main() {
    let opts = opts();
    let root = opts.root;
    let mut failed = 0u32;
    for case in cases() {
        if !selected(&opts.only, case.name) {
            continue;
        }
        let orders: Vec<(&str, Vec<(&str, usize)>)> = if case.pages.len() < 2 {
            vec![("single-page", case.pages.to_vec())]
        } else {
            let forward = case.pages.to_vec();
            let mut reverse = forward.clone();
            reverse.reverse();
            vec![("forward", forward), ("reverse", reverse)]
        };
        for (order, pages) in orders {
            if let Err(err) = run_one(&case, &root, order, &pages).await {
                eprintln!("FAIL {} {order}: {err}", case.name);
                failed += 1;
            }
        }
    }
    if opts.fair_uv && selected(&opts.only, "uv-10186") {
        if let Err(err) = run_uv_fair(&root).await {
            eprintln!("FAIL uv-10186 fair-cutoff: {err}");
            failed += 1;
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}

async fn run_uv_fair(root: &Path) -> Result<(), String> {
    let repo = root.join("uv");
    let parent = git_rev_parse(&repo, "b52d48973fe9ddb2e78b663ec48a1a68f7e7802d^")?;
    let case = Case {
        name: "uv-10186-fair-cutoff",
        repo: "uv",
        failure_sha: "351d602d86c484a39bc537f1eb99866ea2c25fc1",
        cause_sha: "d2f58d92991fa08b24596fcc6c6472dc5015d3bc",
        symptom: "failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12",
        pages: &[],
        starts: uv_starts,
    };
    let pages = vec![(parent.as_str(), 20usize)];
    eprintln!(
        "FAIR uv history cut before b52d48973fe9ddb2e78b663ec48a1a68f7e7802d; ingest rev {parent}"
    );
    run_one(&case, root, "before-revert-b52d489", &pages).await
}

fn git_rev_parse(repo: &Path, spec: &str) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--verify", spec])
        .output()
        .map_err(|err| err.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "git rev-parse {spec} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.len() != 40 {
        return Err(format!("git rev-parse {spec} returned {sha}"));
    }
    Ok(sha)
}

async fn run_one(
    case: &Case,
    root: &Path,
    order: &str,
    pages: &[(&str, usize)],
) -> Result<(), String> {
    let repo = root.join(case.repo);
    if !repo.join(".git").is_dir() {
        return Err(format!("missing clone {}", repo.display()));
    }
    let dir = tempfile::tempdir().map_err(|err| err.to_string())?;
    let storage = strata_memory::open(dir.path()).map_err(|err| err.to_string())?;
    let scope = format!("{}-{order}", case.name);
    let ingest_started = Instant::now();
    let ingest = ingest_pages(&storage, &repo, &scope, pages).await?;
    let ingest_s = ingest_started.elapsed().as_secs_f64();
    let revision_id = commit_id(&storage, &scope, case.failure_sha)
        .ok_or_else(|| format!("failure revision {} was not ingested", case.failure_sha))?;
    let cause_present = commit_id(&storage, &scope, case.cause_sha);
    let failure_id = link_failure(&storage, &scope, case.symptom, &revision_id)?;
    let edges = storage
        .get_connections_for_memory(&failure_id)
        .map_err(|err| err.to_string())?;
    if edges.len() != 1
        || edges[0].link_type != "derived_from"
        || edges[0].target_id != revision_id
        || edges[0].target_id == cause_present.clone().unwrap_or_default()
    {
        return Err(format!(
            "failure record edges are not exactly one derived_from the observed revision: {edges:?}"
        ));
    }
    let walk_started = Instant::now();
    let walk = causal_walk::execute(
        &storage,
        Some(json!({
            "scope": scope,
            "scan_limit": SCAN_LIMIT,
            "start_points": (case.starts)(&failure_id),
        })),
    )
    .await?;
    let walk_s = walk_started.elapsed().as_secs_f64();
    let causes = walk["causes"].as_array().cloned().unwrap_or_default();
    let rank = causes.iter().position(|cause| {
        cause["structure"]["sha"].as_str() == Some(case.cause_sha)
            || cause["id"].as_str() == cause_present.as_deref()
    });
    let row = rank.and_then(|index| causes.get(index));
    println!(
        "RESULT name={} order={} rank={} of={} walk_s={:.3} ingest_s={:.1} failure_edges={} version_range=no derived_from=observed-revision cause_ingested={} edge={} reverted={} hop={} truncated={} empty={}",
        case.name,
        order,
        rank.map(|index| (index + 1).to_string())
            .unwrap_or_else(|| "miss".into()),
        causes.len(),
        walk_s,
        ingest_s,
        edges.len(),
        cause_present.is_some(),
        row.and_then(|row| row["structure"]["edge"].as_str()).unwrap_or("-"),
        row.and_then(|row| row["structure"]["revertedAfterFailure"].as_bool())
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        row.and_then(|row| row["structure"]["hop"].as_u64())
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into()),
        walk["truncated"].as_bool().unwrap_or(false),
        walk["emptyBecause"].as_str().unwrap_or("-"),
    );
    let top = causes
        .iter()
        .take(6)
        .map(|cause| {
            let sha = cause["structure"]["sha"].as_str().unwrap_or("?");
            format!(
                "{} {} {}",
                &sha[..sha.len().min(12)],
                cause["structure"]["edge"].as_str().unwrap_or("?"),
                cause["structure"]["hop"].as_u64().unwrap_or(0),
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    println!("TOP {order} {}: {}", case.name, if top.is_empty() { "-" } else { &top });
    println!("INGEST {order} {}: {ingest}", case.name);
    Ok(())
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
                repo_ingest::Request {
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
                &rev[..12.min(rev.len())],
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

fn commit_id(storage: &Arc<Storage>, scope: &str, sha: &str) -> Option<String> {
    let tag = repo_ingest::commit_tag(sha);
    storage
        .current_code_context_nodes("event", Some(&tag), scope, 5)
        .ok()?
        .into_iter()
        .next()
        .map(|node| node.id)
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
