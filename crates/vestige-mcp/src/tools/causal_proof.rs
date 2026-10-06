//! Ignored end-to-end proof: `ingest_repo` plus one failure memory, then
//! `causal_walk`. No cause edge is written by the harness.
//!
//! ```sh
//! cargo test -p vestige-mcp --lib causal_proof -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use vestige_core::Storage;

use super::repo_ingest::{self, Request};

struct Case {
    name: &'static str,
    url: &'static str,
    /// History fetched from this commit. For a revert, this is the revert,
    /// so the trailer is in the page and the reverted commit is pulled in
    /// even when it sits further back than `limit`.
    ingest_rev: &'static str,
    depth: u32,
    limit: usize,
    /// Revision the failure was observed at. `derived_from` points here only.
    observe: &'static str,
    cause: &'static str,
    /// What the issue actually printed, before the repo-relative frame below.
    reported: &'static str,
    frame: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "tokio-6714",
        url: "https://github.com/tokio-rs/tokio.git",
        ingest_rev: "ea521e17f5030e7909120828e1231f32be98d9f4",
        depth: 80,
        limit: 80,
        observe: "48e35c11d924ffa3a009b89fbb6d36e57b835da4",
        cause: "8480a180e6ffdd0ec0ec213a9bebcda2445fc541",
        reported: "tokio-1.39.0/src/util/linked_list.rs:123:9 (panic in tests::stream_id_flow_control)",
        frame: "tokio/src/util/linked_list.rs:123",
    },
    Case {
        name: "cargo-10682",
        url: "https://github.com/rust-lang/cargo.git",
        ingest_rev: "a028ae42fc1376571de836be702e840ca8e060c2",
        depth: 200,
        limit: 200,
        observe: "a028ae42fc1376571de836be702e840ca8e060c2",
        cause: "f12f0256caedaeada2fecd8bc5e93feee7abb5b7",
        reported: "src/cargo/sources/registry/local.rs#L57-L60 (cargo 1.61.0 a028ae42)",
        frame: "src/cargo/sources/registry/local.rs:57",
    },
    Case {
        name: "uv-10186",
        url: "https://github.com/astral-sh/uv.git",
        ingest_rev: "1807088b7879491ddddd916034696f022785d9d3",
        depth: 40,
        limit: 20,
        observe: "351d602d86c484a39bc537f1eb99866ea2c25fc1",
        cause: "d2f58d92991fa08b24596fcc6c6472dc5015d3bc",
        reported: "uv 0.5.12 error decoding response body; no source path in the issue. Frame is Cargo.lock, the only file the reverted commit records.",
        frame: "Cargo.lock",
    },
    Case {
        name: "prometheus-mempostings",
        url: "https://github.com/prometheus/prometheus.git",
        ingest_rev: "2fbbfc3da800d3b33c7f7b430e403f66b781b962",
        depth: 250,
        limit: 180,
        observe: "9700933d18954e82155f1abed712707d66027b1e",
        cause: "50ef0dc954592666a13ff92ef20811f0127c3b49",
        reported: "Prombench memory regression; no stack. Frame is tsdb/index/postings.go (MemPostings).",
        frame: "tsdb/index/postings.go",
    },
    Case {
        name: "kubernetes-windows-kubelet",
        url: "https://github.com/kubernetes/kubernetes.git",
        ingest_rev: "3ca610757ed3e38f8a0a58798a0bdb23aebcc6bb",
        depth: 2200,
        limit: 40,
        observe: "bbd83d86444d7b325a51f4daa0d65163b795b70e",
        cause: "4060ee60c1d2e5ba1fba1f8729adfc211cee1b6f",
        reported: "server.go:102 Failed to listen to socket while starting device plugin registry; kubelet.go:1566 Failed to start ContainerManager",
        frame: "pkg/kubelet/cm/devicemanager/plugin/v1beta1/server.go:102",
    },
    Case {
        name: "grafana-min-step",
        url: "https://github.com/grafana/grafana.git",
        ingest_rev: "03d43277131d4310f2de02efcf81d168b132d273",
        depth: 450,
        limit: 400,
        observe: "4c8d14014dd8c40fd9f41df83c3ccaf107b0d346",
        cause: "7ae2eed876fce88281ef88ba1e7edca97cfdfde6",
        reported: "Min Step precedence broke existing Prometheus dashboards; the revert cites no stack. Frame is pkg/promlib/models/query.go.",
        frame: "pkg/promlib/models/query.go",
    },
];

fn say(line: impl AsRef<str>) {
    eprintln!("{}", line.as_ref());
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

fn proof_root() -> PathBuf {
    std::env::var_os("VESTIGE_PROOF_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("vestige-proof"))
}

fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|err| format!("git: {err}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn has_commit(dir: &Path, sha: &str) -> bool {
    dir.join(".git").is_dir() && git(dir, &["cat-file", "-e", sha]).is_ok()
}

fn prepare(case: &Case) -> Result<PathBuf, String> {
    let dir = proof_root().join(case.name);
    if has_commit(&dir, case.ingest_rev)
        && has_commit(&dir, case.observe)
        && has_commit(&dir, case.cause)
    {
        if git(&dir, &["rev-parse", "HEAD"])? != case.ingest_rev {
            git(&dir, &["checkout", "--detach", "--force", case.ingest_rev])?;
        }
        return Ok(dir);
    }
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    if !dir.join(".git").is_dir() {
        git(&dir, &["init", "-q"])?;
        git(&dir, &["remote", "add", "origin", case.url])?;
    }
    let depth = case.depth.to_string();
    git(
        &dir,
        &["fetch", "--depth", &depth, "origin", case.ingest_rev],
    )?;
    git(&dir, &["checkout", "--detach", "--force", "FETCH_HEAD"])?;
    for sha in [case.observe, case.cause] {
        if !has_commit(&dir, sha) {
            git(&dir, &["fetch", "--depth", &depth, "origin", sha])?;
        }
    }
    Ok(dir)
}

async fn ingest_page(
    storage: &Arc<Storage>,
    dir: &Path,
    name: &str,
    rev: &str,
    limit: usize,
) -> Result<Value, String> {
    let mut last = Value::Null;
    for try_n in 1..=12 {
        let out = repo_ingest::execute(
            storage,
            Request {
                repo_path: dir.to_path_buf(),
                codebase: Some(name.to_string()),
                scope: Some(name.to_string()),
                rev: Some(rev.to_string()),
                since: None,
                until: None,
                limit: Some(limit),
                dry_run: false,
                budget: Some(Duration::from_secs(180)),
            },
        )
        .await?;
        say(format!(
            "ingest {name} try {try_n} seen={} created={} remaining={} stopped={} outside={} derived={} corrects={} touched={} edge_err={} secret={} partial={} err={}",
            out["commits"]["seen"],
            out["commits"]["created"],
            out["commits"]["remaining"],
            out["commits"]["stoppedByBudget"],
            out["commits"]["outsidePage"],
            out["edges"]["derivedFrom"],
            out["edges"]["corrects"],
            out["edges"]["touched"],
            out["edges"]["errors"],
            out["commits"]["skippedSecret"],
            out["partial"],
            out["error"],
        ));
        let stopped = out["commits"]["stoppedByBudget"] == true;
        let remaining = out["commits"]["remaining"].as_u64().unwrap_or(0);
        last = out;
        if !stopped && remaining == 0 {
            break;
        }
        if !stopped && remaining > 0 && last["commits"]["created"].as_u64() == Some(0) {
            return Err(format!("ingest made no progress: {last}"));
        }
    }
    Ok(last)
}

fn node_for(storage: &Arc<Storage>, sha: &str) -> Option<String> {
    storage
        .resolve_handle(&format!("commit:{sha}"))
        .ids
        .into_iter()
        .next()
}

fn failure_at(storage: &Arc<Storage>, scope: &str, observed: &str, text: &str) -> String {
    let failure = storage
        .ingest_in_scope(
            vestige_core::IngestInput {
                content: format!("failure\n{text}"),
                node_type: "event".into(),
                tags: vec!["failure".into()],
                ..Default::default()
            },
            scope,
        )
        .unwrap();
    let now = chrono::Utc::now();
    storage
        .save_connection(&vestige_core::ConnectionRecord {
            source_id: failure.id.clone(),
            target_id: observed.to_string(),
            strength: 1.0,
            link_type: "derived_from".into(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .unwrap();
    failure.id
}

fn sha_of(row: &Value) -> String {
    row["content"]
        .as_str()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string()
}

async fn prove(case: &Case) -> Result<String, String> {
    let dir = prepare(case)?;
    let data = tempfile::TempDir::new().map_err(|err| err.to_string())?;
    let storage = crate::strata_memory::open(data.path()).map_err(|err| err.to_string())?;
    let ingested = ingest_page(&storage, &dir, case.name, case.ingest_rev, case.limit).await?;
    let cause_id = node_for(&storage, case.cause).ok_or_else(|| {
        format!(
            "cause {} was not ingested (outside={}, corrects={})",
            case.cause, ingested["commits"]["outsidePage"], ingested["edges"]["corrects"]
        )
    })?;
    let observe_id = node_for(&storage, case.observe)
        .ok_or_else(|| format!("observation {} was not ingested", case.observe))?;
    let failure = failure_at(&storage, case.name, &observe_id, case.reported);
    let linked_to_cause = storage
        .get_connections_for_memory(&failure)
        .map_err(|err| err.to_string())?
        .iter()
        .any(|edge| edge.target_id == cause_id || edge.source_id == cause_id);
    if linked_to_cause {
        return Err("harness wrote an edge to the cause".into());
    }

    let mut points = vec![json!({
        "kind": "stack_frame",
        "frame": case.frame,
        "node_id": failure,
    })];
    if case.name == "kubernetes-windows-kubelet" {
        points.push(json!({
            "kind": "stack_frame",
            "frame": "pkg/kubelet/kubelet.go:1566",
            "node_id": failure,
        }));
    }
    let out = crate::tools::causal_walk::execute(
        &storage,
        Some(json!({
            "scope": case.name,
            "scan_limit": 2000,
            "start_points": points,
        })),
    )
    .await?;
    let causes = out["causes"].as_array().cloned().unwrap_or_default();
    let rank = causes.iter().position(|row| sha_of(row) == case.cause);
    let top: Vec<String> = causes
        .iter()
        .take(5)
        .map(|row| {
            let content = row["content"].as_str().unwrap_or("");
            let mut parts = content.split_whitespace();
            let sha = parts.nth(1).unwrap_or("");
            let subject: String = content
                .split_once(' ')
                .and_then(|(_, rest)| rest.split_once(' '))
                .map(|(_, subject)| subject)
                .unwrap_or("")
                .chars()
                .take(80)
                .collect();
            format!("{} {subject}", &sha[..sha.len().min(12)])
        })
        .collect();
    let reasons: Vec<String> = out["start_points"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|row| {
            format!(
                "{}={}",
                row["status"].as_str().unwrap_or(""),
                row["reason"].as_str().unwrap_or("")
            )
        })
        .collect();
    let line = format!(
        "PROOF {} cause={} rank={} of {} found={} reported={} frame={} starts=[{}] top=[{}]",
        case.name,
        &case.cause[..12],
        rank.map(|n| n.to_string()).unwrap_or_else(|| "miss".into()),
        causes.len(),
        rank.is_some(),
        case.reported,
        case.frame,
        reasons.join(" | "),
        top.join(" | "),
    );
    say(&line);
    Ok(line)
}

#[tokio::test]
#[ignore = "clones real git history and runs ingest_repo + causal_walk"]
async fn real_regressions_from_ingest_and_one_failure() {
    let mut lines = Vec::new();
    for case in CASES {
        match prove(case).await {
            Ok(line) => lines.push(line),
            Err(err) => {
                let line = format!("PROOF {} HARNESS_ERROR {err}", case.name);
                say(&line);
                lines.push(line);
            }
        }
    }
    say("--- proof summary ---");
    for line in &lines {
        say(line);
    }
    assert!(
        lines.iter().all(|line| !line.contains("HARNESS_ERROR")),
        "{lines:?}"
    );
}
