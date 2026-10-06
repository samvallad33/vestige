//! # `vestige prove`: the walk proposes, the test decides
//!
//! A causal walk returns leads: commits the failure memory reaches over
//! recorded edges. A lead is not a cause. `prove` takes those leads and runs
//! the user's own test on them, so the answer it prints is a tested one:
//!
//! 1. the causal walk from the failure memory (`--logged-write`), in
//!    process, the same walk `vestige causal-walk --logged-write` prints;
//! 2. a time gate: a lead committed after the failure was reported, or
//!    outside `good..bad`, is dropped;
//! 3. the protocol is frozen before any test runs: the test's sha256, the
//!    two ends, the leads in order and the limits are hashed and saved as a
//!    memory, so none of them can be chosen after seeing a result. Then the
//!    test must pass on `--good` and fail on `--bad`;
//! 4. a bisect over all the leads, oldest to newest, then the parent of the
//!    earliest failing one. "fails, parent passes" is a tested boundary;
//! 5. stock `git bisect run` over the whole range as confirmation, reusing
//!    every verdict already recorded, and a replay that counts how many
//!    runs plain bisect needs;
//! 6. why: the smallest set of the commit's changes that still fails on its
//!    parent (Zeller's ddmin), the rest of the commit without that set
//!    (must pass), and an undo on the bad ref (must pass);
//! 7. with `--flaky`, a fixed number of runs with and without the commit.
//!
//! The result ends in a verdict card (see [`card`]): LEAD, BOUNDARY,
//! CONFIRMED, ISOLATED, REVERSED, and REPEATED under `--flaky`, each with
//! whether it holds and the run numbers that back it.
//!
//! Every test run is written twice: appended to a probe log in which each
//! entry carries the sha256 of the one before it, and saved in the store as
//! an `event` memory through the calls `vestige ingest` makes. The report
//! file holds the whole chain and `vestige prove --check` re-verifies it.
//!
//! Two kinds of line are printed and they are labelled:
//! `[recorded link]` is a lead from the walk, `[tested]` is a test run.
//!
//! The test follows `git bisect run`: exit 0 is good, 125 is "cannot test
//! this commit", any other code is bad.
//!
//! ## A bug that only shows some of the time
//!
//! `--flaky` replaces "one run, one verdict". The two ends are measured in
//! batches until a one-sided Fisher exact test puts their difference beyond
//! chance (p < 0.001); every commit is then judged by Wald's sequential
//! test between the two measured failure rates, and a commit the test
//! cannot decide within `--max-runs` counts as "cannot test". See [`stats`].
//!
//! ## What it touches
//!
//! The user's checkout, index, branches and bisect state are never
//! touched. Tests run in a detached worktree under a scratch directory in
//! the system temp directory; both are removed on every way out, error and
//! Ctrl-C included (see [`git`]). The report and the undo patch are only
//! ever created, never overwritten.
//!
//! ## Byte compatibility
//!
//! This is a port of the `walk-verify.py` reference tool and its reports
//! are interchangeable with that tool's: either `check` verifies the
//! other's report. The chain and protocol hashes are taken over Python's
//! canonical JSON form (see [`json`]).
//!
//! ## One writer
//!
//! A CLI command that opens a Strata log is its only writer until it exits,
//! so the processes `git bisect run` starts cannot write to the store. They
//! run the test, hand the result back through a file, and this process
//! saves the memory and extends the chain as each result arrives (see
//! [`child`]).

use std::path::PathBuf;
use std::sync::Arc;

use vestige_core::Storage;

pub mod card;
pub mod check;
pub mod child;
pub mod git;
pub mod hunks;
pub mod json;
pub mod probe;
pub mod run;
pub mod stats;
mod text;

pub use check::check;
pub use child::child;
pub use run::prove;

/// Name of the hidden subcommand `git bisect run` calls back into.
pub const CHILD_COMMAND: &str = "_prove";

/// Arguments of `vestige prove`.
#[derive(Debug, Clone, clap::Args)]
pub struct ProveArgs {
    /// The failure memory the causal walk starts from
    #[arg(long, value_name = "MEMORY_ID", required_unless_present = "check")]
    pub logged_write: Option<String>,
    /// Git repository the failure is in. Tests run in a temporary worktree
    /// of it, never in this checkout
    #[arg(long, value_name = "DIR", required_unless_present = "check")]
    pub repo: Option<PathBuf>,
    /// A ref where the test passes
    #[arg(long, value_name = "REF", required_unless_present = "check")]
    pub good: Option<String>,
    /// A ref where the test fails
    #[arg(long, value_name = "REF", required_unless_present = "check")]
    pub bad: Option<String>,
    /// One shell command to use as the test: exit 0 good, 125 cannot test,
    /// any other code bad
    #[arg(long, value_name = "COMMAND", conflicts_with = "oracle")]
    pub test: Option<String>,
    /// An executable test script, same exit codes as --test
    #[arg(long, value_name = "SCRIPT")]
    pub oracle: Option<PathBuf>,
    /// Another file the test depends on; its sha256 is frozen in the
    /// protocol with the test's own (repeatable)
    #[arg(long, value_name = "FILE")]
    pub also_hash: Vec<PathBuf>,
    /// When the failure was reported (RFC 3339). A commit made after this
    /// cannot be its cause
    #[arg(long, value_name = "RFC3339", required_unless_present = "check")]
    pub reported_at: Option<String>,
    /// Where to write the JSON report (refuses to overwrite)
    #[arg(long, value_name = "FILE", required_unless_present = "check")]
    pub report: Option<PathBuf>,
    /// Tag carried by every memory this run writes
    #[arg(long, default_value = "walk-verify")]
    pub slug: String,
    /// At most this many test runs for the bisect over the leads. Every
    /// lead is in that bisect; when the cap is reached git bisect decides
    #[arg(long, default_value_t = 12, value_name = "RUNS")]
    pub max_candidates: usize,
    /// At most this many test runs for the search inside the commit
    #[arg(long, default_value_t = 24, value_name = "RUNS")]
    pub max_line_runs: u32,
    /// Kill a test run that takes longer than this and count it as cannot
    /// test (0 for no limit)
    #[arg(long, default_value_t = 600, value_name = "SECONDS")]
    pub timeout: u64,
    /// Stop at the first bad commit; skip the search inside it
    #[arg(long)]
    pub no_why: bool,
    /// The bug only shows some of the time: repeat the test on each commit
    /// until the evidence is decisive
    #[arg(long)]
    pub flaky: bool,
    /// --flaky: accepted chance of a wrong verdict on one commit
    #[arg(long, default_value_t = 0.01, value_name = "P")]
    pub alpha: f64,
    /// --flaky: at most this many runs on one commit; undecided by then, it
    /// counts as cannot test
    #[arg(long, default_value_t = 80, value_name = "RUNS")]
    pub max_runs: u64,
    /// --flaky: at most this many runs on each end while measuring how
    /// often the test fails there
    #[arg(long, default_value_t = 100, value_name = "RUNS")]
    pub baseline_max: u64,
    /// --flaky: runs with and runs without the first bad commit at the end
    #[arg(long, default_value_t = 50, value_name = "RUNS")]
    pub strength_runs: u64,
    /// How many leads to list
    #[arg(long, default_value_t = 6)]
    pub show: usize,
    /// Re-verify a report offline: the hash chain, the frozen protocol, the
    /// verdict card against the recorded runs, and the undo patch
    #[arg(
        long,
        value_name = "REPORT",
        conflicts_with_all = ["logged_write", "repo", "good", "bad", "test", "oracle", "reported_at", "report"]
    )]
    pub check: Option<PathBuf>,
}

/// `vestige prove`: `--check` needs no store, everything else does.
/// `open_store` is the CLI's own way of opening the store it chose.
/// Returns the process exit code.
pub fn run<F>(args: &ProveArgs, open_store: F) -> anyhow::Result<i32>
where
    F: FnOnce() -> anyhow::Result<(Arc<Storage>, PathBuf)>,
{
    if let Some(report) = &args.check {
        return check(&text::expand_user(report));
    }
    let (storage, data_dir) = open_store()?;
    prove(&storage, &data_dir, args)
}
