//! The processes `git bisect run` starts.
//!
//! `git bisect run` needs a command that tests the commit it has checked
//! out. That command is this binary again, through a hidden subcommand with
//! two modes:
//!
//! * `probe` tests the commit (or reuses its recorded verdict), hands the
//!   result back to the parent through a file and exits 0 for good, 1 for
//!   bad, 125 for cannot test;
//! * `sim` answers from the first bad commit already found and counts the
//!   call, which is how many runs plain bisect needs.
//!
//! Neither opens the store: the parent holds it and is its only writer.
//!
//! Any failure of the subcommand itself (a configuration it cannot read, an
//! I/O error, a panic) exits with 255. git reads 1 to 127 as "this commit
//! is bad", so an internal failure reported that way would silently corrupt
//! the bisection; on 128 and above `git bisect run` aborts instead, and the
//! parent reports why.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::CHILD_COMMAND;
use super::git::{git_out, git_test, subject_of, watch_signals};
use super::probe::{Mode, Session, Test, append_line, measure, new_entry, text_of};
use super::stats::Stats;

/// What the subcommand exits with when it fails itself. `git bisect run`
/// aborts on any code from 128 up.
pub const CHILD_FAILED: i32 = 255;

// git reads 1..=127 as a verdict on the commit and aborts from 128.
const _: () = assert!(CHILD_FAILED >= 128);

/// What the parent tells the processes `git bisect run` starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ChildConfig {
    pub repo: PathBuf,
    pub worktree: PathBuf,
    pub oracle: PathBuf,
    pub oracle_sha256: String,
    /// Seconds one test run may take; `None` for no limit.
    pub timeout_seconds: Option<u64>,
    /// The probe log: verdicts already recorded are reused, not re-run.
    pub session: PathBuf,
    /// Where a child hands a new result back to the parent.
    pub pending: PathBuf,
    /// `--flaky`: the rates measured on the two ends.
    pub stats: Option<StatsBits>,
    /// Set for the replay of plain bisect.
    pub replay: Option<Replay>,
}

/// [`Stats`] with each float as its bit pattern, so the child tests with
/// exactly the numbers the parent measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct StatsBits {
    p0: u64,
    p1: u64,
    alpha: u64,
    beta: u64,
    max_runs: u64,
}

impl From<Stats> for StatsBits {
    fn from(stats: Stats) -> Self {
        Self {
            p0: stats.p0.to_bits(),
            p1: stats.p1.to_bits(),
            alpha: stats.alpha.to_bits(),
            beta: stats.beta.to_bits(),
            max_runs: stats.max_runs,
        }
    }
}

impl From<StatsBits> for Stats {
    fn from(bits: StatsBits) -> Self {
        Self {
            p0: f64::from_bits(bits.p0),
            p1: f64::from_bits(bits.p1),
            alpha: f64::from_bits(bits.alpha),
            beta: f64::from_bits(bits.beta),
            max_runs: bits.max_runs,
        }
    }
}

/// The answer a replay gives, and where it counts its calls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Replay {
    pub first_bad: String,
    pub counter: PathBuf,
}

impl ChildConfig {
    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let text = serde_json::to_string(self).context("cannot write the run configuration")?;
        fs::write(path, text)
            .with_context(|| format!("cannot write the run configuration {}", path.display()))
    }

    fn read(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("cannot read the run configuration {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("bad run configuration {}", path.display()))
    }

    fn test(&self) -> Test {
        Test {
            oracle: self.oracle.clone(),
            worktree: self.worktree.clone(),
            timeout: self.timeout_seconds.map(Duration::from_secs),
        }
    }
}

/// Entry point of the hidden subcommand: the process exit code.
pub fn child(mode: &str, cfg: &Path) -> i32 {
    // The test runs in its own process group; on Ctrl-C this process has to
    // be the one that stops it.
    watch_signals();
    run_child(mode, cfg)
}

fn run_child(mode: &str, cfg: &Path) -> i32 {
    // A panic must not leave with the code of a bad commit either.
    let outcome = std::panic::catch_unwind(|| {
        ChildConfig::read(cfg).and_then(|cfg| match mode {
            "probe" => child_probe(&cfg),
            "sim" => child_sim(&cfg),
            other => anyhow::bail!("unknown mode {other}"),
        })
    });
    match outcome {
        Ok(Ok(code)) => code,
        Ok(Err(err)) => {
            eprintln!("vestige {CHILD_COMMAND} {mode} failed: {err:#}");
            CHILD_FAILED
        }
        Err(_) => {
            eprintln!("vestige {CHILD_COMMAND} {mode} failed: it panicked");
            CHILD_FAILED
        }
    }
}

/// The exit code that tells git a verdict. Only these three leave a probe:
/// a test that exits with 128 or more, or dies of a signal, is bad by the
/// frozen rule, and handing git its raw code would abort the bisection.
fn code_of(verdict: &str) -> anyhow::Result<i32> {
    match verdict {
        "good" => Ok(0),
        "bad" => Ok(1),
        "skip" => Ok(125),
        other => anyhow::bail!("a recorded verdict reads {other:?}"),
    }
}

fn child_probe(cfg: &ChildConfig) -> anyhow::Result<i32> {
    let sha = git_out(&cfg.worktree, &["rev-parse", "HEAD"])?;

    let recorded = Session::read(&cfg.session);
    if let Some(hit) = recorded
        .iter()
        .find(|entry| text_of(entry, "commit") == sha)
    {
        let code = code_of(text_of(hit, "verdict"))?;
        append_line(
            &cfg.pending,
            &json!({"reused": hit.get("n"), "commit": sha}),
        )
        .context("cannot hand the reused verdict back")?;
        return Ok(code);
    }
    // A run handed back earlier in this bisect and not chained yet.
    let handed_back = fs::read_to_string(&cfg.pending).unwrap_or_default();
    for line in handed_back.lines() {
        if let Ok(record) = serde_json::from_str::<Value>(line)
            && let Some(entry) = record.get("entry").and_then(Value::as_object)
            && text_of(entry, "commit") == sha
        {
            return code_of(text_of(entry, "verdict"));
        }
    }

    let stats: Option<Stats> = cfg.stats.map(Stats::from);
    let mode = match &stats {
        Some(stats) => Mode::Sequential(stats),
        None => Mode::Once,
    };
    let outcome = measure(&cfg.test(), mode)?;
    let code = code_of(outcome.verdict)?;
    let subject = subject_of(&cfg.repo, &sha);
    let entry = new_entry(&sha, &subject, &outcome, &cfg.oracle_sha256, "bisect");
    append_line(&cfg.pending, &json!({"entry": entry})).context("cannot hand the result back")?;
    Ok(code)
}

fn child_sim(cfg: &ChildConfig) -> anyhow::Result<i32> {
    let replay = cfg
        .replay
        .as_ref()
        .context("the run configuration has no replay")?;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&replay.counter)
        .and_then(|mut counter| counter.write_all(b"x"))
        .with_context(|| format!("cannot count in {}", replay.counter.display()))?;
    // Bad exactly when the checked-out commit contains the first bad one.
    let bad = git_test(
        &cfg.worktree,
        &["merge-base", "--is-ancestor", &replay.first_bad, "HEAD"],
    )?;
    Ok(i32::from(bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(dir: &Path) -> ChildConfig {
        ChildConfig {
            repo: dir.join("re po"),
            worktree: dir.join("check out"),
            oracle: dir.join("test.sh"),
            oracle_sha256: "feed".repeat(16),
            timeout_seconds: Some(600),
            session: dir.join("probes.jsonl"),
            pending: dir.join("handed-back.jsonl"),
            stats: Some(Stats::measured((0, 30), (10, 30), 0.01, 80).into()),
            replay: Some(Replay {
                first_bad: "4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270".to_string(),
                counter: dir.join("sim.count"),
            }),
        }
    }

    #[test]
    fn the_configuration_round_trips_with_exact_floats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.json");
        let written = config(dir.path());
        written.write(&path).unwrap();
        let read = ChildConfig::read(&path).unwrap();
        assert_eq!(read, written);
        // The rates come back bit for bit, not through a decimal.
        let stats: Stats = read.stats.unwrap().into();
        assert_eq!(stats, Stats::measured((0, 30), (10, 30), 0.01, 80));
        assert_eq!(stats.p1.to_bits(), (10.5f64 / 31.0).to_bits());
        assert_eq!(read.test().timeout, Some(Duration::from_secs(600)));
    }

    #[test]
    fn a_failure_of_the_subcommand_itself_exits_where_git_bisect_aborts() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-cfg.json");
        assert_eq!(run_child("probe", &missing), CHILD_FAILED);
        assert_eq!(run_child("sim", &missing), CHILD_FAILED);
        let garbled = dir.path().join("cfg.json");
        fs::write(&garbled, "{\"repo\": 3").unwrap();
        assert_eq!(run_child("probe", &garbled), CHILD_FAILED);
        // A configuration that reads, in a place where nothing works: no
        // worktree to ask for HEAD, no replay to answer from.
        let mut cfg = config(dir.path());
        cfg.write(&garbled).unwrap();
        assert_eq!(run_child("probe", &garbled), CHILD_FAILED);
        assert_eq!(run_child("sim", &garbled), CHILD_FAILED);
        assert_eq!(run_child("nonsense", &garbled), CHILD_FAILED);
        cfg.replay = None;
        cfg.write(&garbled).unwrap();
        assert_eq!(run_child("sim", &garbled), CHILD_FAILED);
    }

    #[test]
    fn only_the_three_verdict_codes_leave_a_probe() {
        assert_eq!(code_of("good").unwrap(), 0);
        assert_eq!(code_of("bad").unwrap(), 1);
        assert_eq!(code_of("skip").unwrap(), 125);
        assert!(code_of("").is_err());
        assert!(code_of("GOOD").is_err());
    }
}
