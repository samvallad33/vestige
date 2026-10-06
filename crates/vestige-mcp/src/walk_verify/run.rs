//! The run: walk, time gate, frozen protocol, the tests, the result.

use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};
use vestige_core::{IngestInput, SecretPolicy, Storage};

use super::card::{CardFacts, Rung, Side, Strength, Why, verdict_card};
use super::child::{ChildConfig, Replay};
use super::git::{
    PLAIN_DIFF, ScratchGuard, commit_field, first_bad_logged, first_bad_named, git_bytes,
    git_command, git_fed, git_ok, git_out, git_test, interrupted, parents_of, remove_worktree,
    resolve_commit, subject_of,
};
use super::hunks::{Unit, ddmin, names_of, patch_of, split_hunks};
use super::json::{Entry, pretty_json, protocol_hash, sha256_hex};
use super::probe::{
    Mode, Outcome, Session, Test, count_of, measure, new_entry, probe_text, run_test, text_of,
    verdict_of,
};
use super::stats::{BASELINE_P, MAX_FIXED_RUNS, Stats, fisher_p};
use super::text::{
    Palette, Stop, abspath, expand_user, file_name, format_g, head, now_stamp, opt_display,
    py_display, stop, strip, utf8,
};
use super::{CHILD_COMMAND, ProveArgs};

/// Report format written here: the fields `walk-verify.py` version 7 writes.
const TOOL: &str = "vestige prove (walk-verify 7)";

/// The exit-code rule, as the frozen protocol states it.
const PROTOCOL_RULE: &str = "exit 0 good, 125 cannot test, any other code bad";

const READING: &str = "walk candidates are recorded links (leads). probes and first_bad_commit are tested results of this repro script. why.minimal_failing_changes is tested too: those changes alone, applied to the parent, fail it.";

/// Runs on each end in one batch of the `--flaky` baseline.
const BASELINE_BATCH: u64 = 10;

/// How much of git's own output is kept to show when a bisect names nothing.
const KEPT_TRANSCRIPT: usize = 8192;

// ---------------------------------------------------------------------------
// Leads
// ---------------------------------------------------------------------------

/// One lead: a memory the walk reached whose text names a commit.
#[derive(Debug, Clone)]
pub(super) struct Lead {
    pub rank: usize,
    pub memory: String,
    pub depth: u64,
    /// The sha as the memory wrote it.
    short: String,
    commit: Option<String>,
}

/// The sha a memory names when its text starts `Commit <sha>:` or
/// `Commit <sha> `: 7 to 40 lowercase hex characters. This is the line
/// `vestige causal-walk` prints under each cause, read the same way.
fn commit_named(content: &str) -> Option<String> {
    let rest = content.trim_start().strip_prefix("Commit ")?;
    let sha: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || ('a'..='f').contains(c))
        .collect();
    if !(7..=40).contains(&sha.len()) {
        return None;
    }
    match rest[sha.len()..].chars().next() {
        Some(':' | ' ' | '\n') => Some(sha),
        _ => None,
    }
}

/// The leads of a recorded walk: its causes, in rank order, that name a commit.
fn leads_of(walk: &Value) -> Vec<Lead> {
    walk["causes"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, cause)| {
            Some(Lead {
                rank: index + 1,
                memory: cause["id"].as_str()?.to_string(),
                depth: cause["depth"].as_u64().unwrap_or(0),
                short: commit_named(cause["content"].as_str()?)?,
                commit: None,
            })
        })
        .collect()
}

/// The causal walk from the failure memory, in process: the same call
/// `vestige causal-walk --logged-write` makes, so the leads are that
/// command's causes in its order.
fn walk_from(storage: &Arc<Storage>, failure: &str) -> anyhow::Result<Value> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .context("cannot start the runtime for the walk")?;
    let walk = runtime
        .block_on(crate::tools::causal_walk::execute(
            storage,
            Some(json!({
                "scope": vestige_core::DEFAULT_MEMORY_SCOPE,
                "start_points": [{"kind": "logged_write", "node_id": failure}],
            })),
        ))
        .map_err(|err| stop(1, format!("The walk could not run from {failure}: {err}")))?;
    if let Some(refusal) = walk["needs_report"].as_object() {
        let detail = refusal
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("it gave no reason");
        return Err(stop(
            1,
            format!(
                "The walk could not start from {failure}: {detail}. Check --logged-write and the data directory; nothing was tested."
            ),
        ));
    }
    Ok(walk)
}

// ---------------------------------------------------------------------------
// Memories
// ---------------------------------------------------------------------------

/// Save one `event` memory the way `vestige ingest` does: the default-scope
/// ingest with the secret gate on, then auto-connect on exact identities.
/// The error is why the store did not take it.
fn remember(storage: &Storage, text: &str, tags: &[&str]) -> Result<String, String> {
    let input = IngestInput {
        content: text.to_string(),
        node_type: "event".to_string(),
        source: Some("walk-verify".to_string()),
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
        valid_from: None,
        valid_until: None,
        validity_inferred: false,
        source_envelope: None,
    };
    let node = storage
        .ingest_with_secret_policy(input, SecretPolicy::Reject)
        .map_err(|err| err.to_string())?;
    // The memory is saved; a failed auto-connect is reported, never fatal,
    // as `vestige ingest` has it.
    if let Err(err) = crate::auto_connect::auto_connect_new_memory(
        storage,
        &node.id,
        vestige_core::DEFAULT_MEMORY_SCOPE,
        &node.content,
        &node.tags,
    ) {
        eprintln!("  (auto-connect skipped for {}: {err})", node.id);
    }
    Ok(node.id)
}

// ---------------------------------------------------------------------------
// The prover: everything that needs the worktree
// ---------------------------------------------------------------------------

/// The files the parent shares with the processes `git bisect run` starts.
struct ChildFiles {
    cfg: PathBuf,
    pending: PathBuf,
    /// What git and the hidden subcommand wrote to stderr during a bisect.
    errors: PathBuf,
    counter: PathBuf,
}

/// What a `git bisect run` came to.
struct Bisected {
    first_bad: Option<String>,
    /// Why it named no commit, in git's and the subcommand's own words.
    aborted: Option<String>,
}

/// The cap on new test runs in step 4 (`--max-candidates`).
struct Cap {
    most: usize,
    spent: usize,
    reached: bool,
}

struct Prover<'a> {
    storage: &'a Storage,
    pal: Palette,
    repo: PathBuf,
    test: Test,
    oracle_sha256: String,
    slug: String,
    session: Session,
    /// The rates `--flaky` measured on the two ends, once it has.
    stats: Option<Stats>,
    /// Runs the store did not take as a memory.
    unsaved: usize,
}

impl Prover<'_> {
    /// Save the run as a memory, then chain it. A run the store does not
    /// take is still chained, with no memory, and is counted.
    fn admit(&mut self, mut entry: Entry, on_commit: bool) -> anyhow::Result<Entry> {
        let Palette { y, o, .. } = self.pal;
        let text = probe_text(&entry, on_commit);
        let verdict_tag = format!("probe-{}", text_of(&entry, "verdict"));
        let tags = ["oracle-probe", verdict_tag.as_str(), self.slug.as_str()];
        let memory = match remember(self.storage, &text, &tags) {
            Ok(id) => Value::String(id),
            Err(err) => {
                self.unsaved += 1;
                println!(
                    "  {y}warning: this run could not be saved as a memory ({err}); it is still in the report{o}"
                );
                Value::Null
            }
        };
        entry.insert("memory".to_string(), memory);
        self.session.append(entry)
    }

    fn print_tested(&self, entry: &Entry) {
        let Palette { b, d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {b}[tested]{o} {} {}{:<4}{o} {}  {d}-> {}{o}",
            head(text_of(entry, "commit"), 10),
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            text_of(entry, "oracle_said"),
            py_display(entry.get("memory")),
        );
    }

    fn print_reused(&self, entry: &Entry) {
        let Palette { d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {d}[tested]{o} {} {}{:<4}{o} {d}reused probe {}{o}",
            head(text_of(entry, "commit"), 10),
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            py_display(entry.get("n")),
        );
    }

    fn show(&self, what: &str, entry: &Entry) {
        let Palette { b, d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {b}[tested]{o} {what}: {}{:<4}{o} {}  {d}-> {}{o}",
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            text_of(entry, "oracle_said"),
            py_display(entry.get("memory")),
        );
    }

    /// One run decides, or under `--flaky` as many as the evidence needs.
    fn mode(&self) -> Mode<'_> {
        match &self.stats {
            Some(stats) => Mode::Sequential(stats),
            None => Mode::Once,
        }
    }

    /// Put the worktree on `commit`. `force` also drops what a patch or a
    /// test changed in tracked files.
    fn checkout(&self, commit: &str, force: bool) -> anyhow::Result<()> {
        let mut command = git_command(&self.test.worktree);
        command.args(["checkout", "-q", "--detach"]);
        if force {
            command.arg("-f");
        }
        let output = command.arg(commit).output().context("cannot run git")?;
        if !output.status.success() {
            return Err(stop(
                2,
                format!(
                    "checkout of {} failed: {}",
                    head(commit, 10),
                    head(strip(&String::from_utf8_lossy(&output.stderr)), 200)
                ),
            ));
        }
        Ok(())
    }

    /// Test one commit, or reuse its recorded verdict. A reused verdict
    /// prints nothing: its line is already above.
    fn probe(&mut self, sha: &str, phase: &str) -> anyhow::Result<Entry> {
        if let Some(hit) = self.session.cached(sha) {
            return Ok(hit.clone());
        }
        self.checkout(sha, false)?;
        let outcome = measure(&self.test, self.mode())?;
        let subject = subject_of(&self.repo, sha);
        let entry = new_entry(sha, &subject, &outcome, &self.oracle_sha256, phase);
        let entry = self.admit(entry, true)?;
        self.print_tested(&entry);
        Ok(entry)
    }

    /// [`Self::probe`] under the cap of step 4: `None`, with the cap
    /// marked as reached, when the commit would need a new run and none is
    /// left. A reused verdict costs nothing.
    fn probe_within(
        &mut self,
        cap: &mut Cap,
        sha: &str,
        phase: &str,
    ) -> anyhow::Result<Option<Entry>> {
        if self.session.cached(sha).is_none() {
            if cap.spent >= cap.most {
                cap.reached = true;
                return Ok(None);
            }
            cap.spent += 1;
        }
        self.probe(sha, phase).map(Some)
    }

    /// Step 4: bisect the leads, closest links first. The leads one link
    /// from the report are bisected oldest to newest (git bisect's own
    /// assumption: one first bad commit, everything after it bad) and the
    /// parent of the earliest failing one is tested; "fails, parent passes"
    /// is the boundary. Only when that tier holds no boundary are the leads
    /// two links away added, then three, each time reusing every run
    /// already made. `ordered` is every kept lead in the range, oldest
    /// first.
    fn bisect_leads(&mut self, ordered: &[Lead], most: usize) -> anyhow::Result<Option<Lead>> {
        let Palette { d, g, y, o, .. } = self.pal;
        let mut depths: Vec<u64> = ordered.iter().map(|lead| lead.depth).collect();
        depths.sort_unstable();
        depths.dedup();
        let mut cap = Cap {
            most,
            spent: 0,
            reached: false,
        };
        let mut boundary = None;
        'tiers: for depth in &depths {
            let mut line_up: Vec<&Lead> =
                ordered.iter().filter(|lead| lead.depth <= *depth).collect();
            if depths.len() > 1 {
                println!(
                    "  {d}leads within {depth} link{} of the report: {}{o}",
                    if *depth == 1 { "" } else { "s" },
                    line_up.len()
                );
            }
            // Leads at `low..high` are still open.
            let (mut low, mut high) = (0usize, line_up.len());
            let mut earliest_bad: Option<&Lead> = None;
            while low < high {
                let middle = (low + high - 1) / 2;
                let lead = line_up[middle];
                let commit = lead.commit.as_deref().unwrap_or_default();
                let Some(entry) = self.probe_within(&mut cap, commit, "candidate")? else {
                    break 'tiers;
                };
                match text_of(&entry, "verdict") {
                    "bad" => {
                        earliest_bad = Some(lead);
                        high = middle;
                    }
                    "good" => low = middle + 1,
                    _ => {
                        // Cannot be tested: it leaves the line-up.
                        line_up.remove(middle);
                        high -= 1;
                    }
                }
            }
            let Some(lead) = earliest_bad else {
                continue;
            };
            let commit = lead.commit.as_deref().unwrap_or_default();
            // A root commit has no parent to test: no boundary in this tier.
            let Some(parent) = parents_of(&self.repo, commit)?.into_iter().next() else {
                continue;
            };
            let Some(on_parent) = self.probe_within(&mut cap, &parent, "parent")? else {
                break 'tiers;
            };
            if text_of(&on_parent, "verdict") == "good" {
                println!("  {g}{} fails and its parent passes.{o}", lead.short);
                boundary = Some(lead.clone());
                break 'tiers;
            }
        }
        if cap.reached {
            println!(
                "  {y}the cap of {most} test runs on leads is reached; the full git bisect takes over{o}"
            );
        } else if boundary.is_none() {
            println!(
                "  {y}The first bad commit is not among the leads; the full git bisect takes over.{o}"
            );
        }
        Ok(boundary)
    }

    /// Test whatever is in the worktree now and record it under `label`.
    /// `fixed` runs the test exactly that many times.
    fn probe_state(
        &mut self,
        label: &str,
        subject: &str,
        phase: &str,
        fixed: Option<u64>,
    ) -> anyhow::Result<Entry> {
        let mode = fixed.map_or(self.mode(), Mode::Fixed);
        let outcome = measure(&self.test, mode)?;
        let entry = new_entry(label, subject, &outcome, &self.oracle_sha256, phase);
        self.admit(entry, false)
    }

    /// `--flaky` step 3: measure how often the test fails on each end, in
    /// batches, until the difference cannot be chance or `baseline_max`
    /// runs an end are spent. Returns the Fisher p of the last batch and
    /// keeps the two rates for every later verdict.
    fn flaky_baseline(
        &mut self,
        good: (&str, &str),
        bad: (&str, &str),
        baseline_max: u64,
        alpha: f64,
        max_runs: u64,
    ) -> anyhow::Result<f64> {
        let Palette { b, d, r, o, .. } = self.pal;
        let started = now_stamp();
        // (failures, runs) on each end.
        let (mut on_bad, mut on_good) = ((0u64, 0u64), (0u64, 0u64));
        let mut p_value = 1.0;
        while on_bad.1 < baseline_max {
            for ((commit, name), counts) in [(bad, &mut on_bad), (good, &mut on_good)] {
                self.checkout(commit, false)?;
                let before = counts.1;
                for _ in 0..BASELINE_BATCH {
                    let run = run_test(&self.test)?;
                    match verdict_of(run.exit) {
                        "skip" => {}
                        verdict => {
                            counts.1 += 1;
                            counts.0 += u64::from(verdict == "bad");
                        }
                    }
                }
                if counts.1 == before {
                    // No run of a whole batch could be tested. Repeating
                    // would never end on the bad end and decides nothing on
                    // the good one.
                    return Err(stop(
                        1,
                        format!(
                            "{r}The test could not be run on {name}: {BASELINE_BATCH} runs in a row ended as cannot test (exit 125 or timed out), so nothing can be decided with it.{o}"
                        ),
                    ));
                }
            }
            p_value = fisher_p(on_bad.0, on_bad.1, on_good.0, on_good.1);
            if p_value < BASELINE_P {
                break;
            }
        }
        let decided = p_value < BASELINE_P;
        for ((commit, _), (fails, runs), end) in [(good, on_good, "good"), (bad, on_bad, "bad")] {
            let verdict = if decided { end } else { "skip" };
            let exit = i64::from(end == "bad");
            let outcome = Outcome::counted(verdict, exit, runs, fails, started.clone());
            let subject = subject_of(&self.repo, commit);
            let entry = new_entry(commit, &subject, &outcome, &self.oracle_sha256, "baseline");
            let entry = self.admit(entry, true)?;
            println!(
                "  {b}[tested]{o} {} {}{:<4}{o} failed {fails} of {runs} runs  {d}-> {}{o}",
                head(commit, 10),
                self.pal.verdict(end),
                end.to_uppercase(),
                py_display(entry.get("memory")),
            );
        }
        self.stats = Some(Stats::measured(on_good, on_bad, alpha, max_runs));
        Ok(p_value)
    }

    /// Take the results the bisect children have handed back since the last
    /// call: save and chain each new run, print each line.
    fn take_handed_back(&mut self, pending: &Path, taken: &mut usize) -> anyhow::Result<()> {
        let Ok(text) = fs::read_to_string(pending) else {
            return Ok(());
        };
        // A line still being written has no newline yet; it waits.
        let complete = text.rfind('\n').map_or("", |end| &text[..=end]);
        for line in complete.lines().skip(*taken) {
            *taken += 1;
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if let Some(n) = record.get("reused").and_then(Value::as_u64) {
                let hit = usize::try_from(n)
                    .ok()
                    .and_then(|n| n.checked_sub(1))
                    .and_then(|index| self.session.entries.get(index))
                    .cloned();
                if let Some(hit) = hit {
                    self.print_reused(&hit);
                }
            } else if let Some(entry) = record.get("entry").and_then(Value::as_object) {
                let entry = self.admit(entry.clone(), true)?;
                self.print_tested(&entry);
            }
        }
        Ok(())
    }

    /// Start a bisect in the worktree. The error is git's own.
    fn bisect_start(&self, good: &str, bad: &str) -> Result<(), String> {
        let output = git_command(&self.test.worktree)
            .args(["bisect", "start", bad, good])
            .output()
            .map_err(|err| format!("cannot run git: {err}"))?;
        if output.status.success() {
            return Ok(());
        }
        git_ok(&self.test.worktree, &["bisect", "reset"]);
        Err(format!(
            "git bisect start failed: {}",
            head(strip(&String::from_utf8_lossy(&output.stderr)), 400)
        ))
    }

    /// Stock `git bisect run` over `good..bad`, with this binary's hidden
    /// subcommand as the script. Every result a child hands back is saved
    /// and chained here, as it arrives.
    fn bisect(&mut self, files: &ChildFiles, good: &str, bad: &str) -> anyhow::Result<Bisected> {
        let exe = std::env::current_exe().context("cannot find this executable")?;
        if let Err(why) = self.bisect_start(good, bad) {
            return Ok(Bisected {
                first_bad: None,
                aborted: Some(why),
            });
        }
        let errors = fs::File::create(&files.errors).context("cannot create the bisect log")?;
        let mut run = git_command(&self.test.worktree)
            .args(["bisect", "run"])
            .arg(&exe)
            .args([CHILD_COMMAND, "probe"])
            .arg(&files.cfg)
            .stdout(Stdio::piped())
            .stderr(Stdio::from(errors))
            .spawn()
            .context("cannot run git bisect")?;
        let mut transcript = String::new();
        let mut named = None;
        let mut taken = 0usize;
        let mut read = || -> anyhow::Result<()> {
            let Some(stdout) = run.stdout.take() else {
                return Ok(());
            };
            // git prints after each run, so each line is a cue to pick up
            // what the child just handed back.
            for line in BufReader::new(stdout).split(b'\n') {
                let line = String::from_utf8_lossy(&line?).into_owned();
                self.take_handed_back(&files.pending, &mut taken)?;
                if let Some(sha) = first_bad_named(&line) {
                    named = Some(sha);
                }
                transcript.push_str(&line);
                transcript.push('\n');
                if transcript.len() > 2 * KEPT_TRANSCRIPT {
                    let cut = transcript.len() - KEPT_TRANSCRIPT;
                    let cut = (cut..transcript.len())
                        .find(|&at| transcript.is_char_boundary(at))
                        .unwrap_or(transcript.len());
                    transcript.drain(..cut);
                }
            }
            Ok(())
        };
        let outcome = read();
        if outcome.is_err() {
            let _ = run.kill();
        }
        let finished = run.wait();
        let outcome = outcome.and_then(|()| self.take_handed_back(&files.pending, &mut taken));
        // The log names the commit in English whatever the locale; it is
        // gone after the reset.
        let log = git_out(&self.test.worktree, &["bisect", "log"]).unwrap_or_default();
        git_ok(&self.test.worktree, &["bisect", "reset"]);
        outcome?;
        interrupted()?;
        let first_bad = first_bad_logged(&log).or(named);
        let aborted = first_bad.is_none().then(|| {
            // What git printed, then what git and the hidden subcommand
            // wrote to stderr: the reason is in one of the two.
            let said = fs::read_to_string(&files.errors).unwrap_or_default();
            let how = match finished {
                Ok(status) => format!("git bisect run ended with {status}"),
                Err(err) => format!("git bisect run could not be waited for: {err}"),
            };
            format!(
                "{}\n{}\n{how}",
                tail(strip(&transcript), 600),
                tail(strip(&said), 400)
            )
        });
        Ok(Bisected { first_bad, aborted })
    }

    /// How many runs plain `git bisect` needs on this range: replay it
    /// against the tested answer, without running the test. `None`, with
    /// the reason printed, when the replay does not arrive at that answer:
    /// a count from a replay that stopped early would be wrong.
    fn replay_plain_bisect(
        &mut self,
        files: &ChildFiles,
        good: &str,
        bad: &str,
        first_bad: &str,
    ) -> anyhow::Result<Option<u64>> {
        let Palette { y, o, .. } = self.pal;
        let exe = std::env::current_exe().context("cannot find this executable")?;
        let unreported = |why: &str| -> anyhow::Result<Option<u64>> {
            println!(
                "  {y}The replay of plain git bisect did not finish, so the number of runs it needs is not reported: {why}{o}"
            );
            Ok(None)
        };
        // A counter left by an earlier replay would be added to.
        if files.counter.exists() {
            fs::remove_file(&files.counter).context("cannot reset the replay counter")?;
        }
        if let Err(why) = self.bisect_start(good, bad) {
            return unreported(&why);
        }
        let run = git_command(&self.test.worktree)
            .args(["bisect", "run"])
            .arg(&exe)
            .args([CHILD_COMMAND, "sim"])
            .arg(&files.cfg)
            .output();
        let log = git_out(&self.test.worktree, &["bisect", "log"]).unwrap_or_default();
        git_ok(&self.test.worktree, &["bisect", "reset"]);
        interrupted()?;
        let run = match run {
            Ok(run) => run,
            Err(err) => return unreported(&format!("cannot run git: {err}")),
        };
        let replayed = first_bad_logged(&log).or_else(|| {
            String::from_utf8_lossy(&run.stdout)
                .lines()
                .find_map(first_bad_named)
        });
        if replayed.as_deref() != Some(first_bad) {
            let said = String::from_utf8_lossy(&run.stderr);
            return unreported(&format!(
                "git bisect run ended with {}. {}",
                run.status,
                tail(strip(&said), 400)
            ));
        }
        Ok(fs::read(&files.counter)
            .ok()
            .map(|marks| marks.len() as u64))
    }

    /// Put the worktree on `commit`, clean: no revert in progress, no
    /// change to a tracked file, no untracked file a patch or the test left.
    fn reset_to(&self, commit: &str) -> anyhow::Result<()> {
        let worktree = &self.test.worktree;
        // Fails when no revert is in progress, which is the usual case.
        git_ok(worktree, &["revert", "--abort"]);
        self.checkout(commit, true)?;
        git_out(worktree, &["reset", "-q", "--hard"])?;
        git_out(worktree, &["clean", "-fdq"])?;
        Ok(())
    }

    /// Apply a patch to the index and the worktree, plainly or else
    /// three-way. `false`, with the worktree as it was, when neither applies.
    fn apply(&self, patch: &[u8], reverse: bool) -> anyhow::Result<bool> {
        for three_way in [false, true] {
            let mut args = vec!["apply", "--whitespace=nowarn", "--index"];
            if reverse {
                args.push("-R");
            }
            if three_way {
                args.push("--3way");
            }
            args.push("-");
            if git_fed(&self.test.worktree, &args, patch)? {
                return Ok(true);
            }
            git_out(&self.test.worktree, &["reset", "-q", "--hard"])?;
        }
        Ok(false)
    }

    /// What is staged against HEAD, as a patch `git apply` takes. Both ways
    /// of undoing stage what they change, files added and removed included.
    fn staged_diff(&self) -> Option<Vec<u8>> {
        let mut args = PLAIN_DIFF.to_vec();
        args.extend(["--cached", "HEAD"]);
        git_bytes(&self.test.worktree, &args)
            .ok()
            .filter(|patch| !patch.is_empty())
    }

    /// Step 6, three tests on the first bad commit. `lines`: ddmin over its
    /// changes applied to its parent, the smallest set that still fails.
    /// `without`: the commit's other changes, without that set, which must
    /// pass. `undo`: on the bad ref, undo the whole commit or, when later
    /// commits are in the way, only the lines found, which must pass.
    fn explain(
        &mut self,
        bad: &str,
        first_bad: &str,
        parents: &[String],
        max_line_runs: u32,
        bad_name: &str,
    ) -> anyhow::Result<Why> {
        let Palette { d, o, .. } = self.pal;
        let short = head(first_bad, 10).to_string();
        let mut why = Why::default();
        match parents.first() {
            None => {
                why.root = true;
                println!(
                    "  {d}{short} is a root commit: it has no parent to apply its changes to, so the search inside it is skipped{o}"
                );
            }
            Some(parent) => {
                if parents.len() > 1 {
                    println!(
                        "  {d}{short} is a merge: its changes are read against its first parent {}{o}",
                        head(parent, 10)
                    );
                }
                self.search_changes(parent, first_bad, max_line_runs, &mut why)?;
            }
        }
        self.undo(bad, first_bad, parents.len() > 1, bad_name, &mut why)?;
        self.reset_to(bad)?;
        Ok(why)
    }

    /// The `lines` and `without` tests of [`Self::explain`].
    fn search_changes(
        &mut self,
        parent: &str,
        first_bad: &str,
        max_line_runs: u32,
        why: &mut Why,
    ) -> anyhow::Result<()> {
        let mut args = PLAIN_DIFF.to_vec();
        args.extend([parent, first_bad]);
        let units = split_hunks(&git_bytes(&self.repo, &args)?);
        let total = units.len();
        let pick =
            |indexes: &[usize]| -> Vec<&Unit> { indexes.iter().map(|&i| &units[i]).collect() };

        let budget = Cell::new(i64::from(max_line_runs));
        let mut cache: HashMap<Vec<usize>, bool> = HashMap::new();
        let mut failure: Option<anyhow::Error> = None;
        let all: Vec<usize> = (0..total).collect();
        let minimal = if total >= 2 {
            ddmin(all.clone(), &budget, |subset| {
                let mut key = subset.to_vec();
                key.sort_unstable();
                if let Some(&known) = cache.get(&key) {
                    return known;
                }
                let ordered = pick(&key);
                let mut test = || -> anyhow::Result<bool> {
                    self.reset_to(parent)?;
                    if !self.apply(&patch_of(&ordered), false)? {
                        // A set that does not apply cannot be said to fail.
                        return Ok(false);
                    }
                    budget.set(budget.get() - 1);
                    let label = format!(
                        "{} + {} of {total} changes",
                        head(parent, 10),
                        ordered.len()
                    );
                    let entry = self.probe_state(&label, &names_of(&ordered), "lines", None)?;
                    self.show(
                        &format!("parent + {:2} of {total} changes", ordered.len()),
                        &entry,
                    );
                    Ok(text_of(&entry, "verdict") == "bad")
                };
                match test() {
                    Ok(fails) => {
                        cache.insert(key, fails);
                        fails
                    }
                    Err(err) => {
                        // Nothing more can be tested; end the search.
                        failure.get_or_insert(err);
                        budget.set(0);
                        false
                    }
                }
            })
        } else {
            all.clone()
        };
        if let Some(err) = failure {
            return Err(err);
        }
        why.units = total;
        why.minimal = minimal.iter().map(|&i| units[i].clone()).collect();
        why.complete = budget.get() > 0;

        let rest: Vec<usize> = all
            .iter()
            .copied()
            .filter(|i| !minimal.contains(i))
            .collect();
        if !minimal.is_empty() && !rest.is_empty() {
            self.reset_to(parent)?;
            if self.apply(&patch_of(&pick(&rest)), false)? {
                let entry = self.probe_state(
                    &format!("{} + the other {} changes", head(parent, 10), rest.len()),
                    &format!("the commit without: {}", names_of(&pick(&minimal))),
                    "without",
                    None,
                )?;
                why.without = Some(text_of(&entry, "verdict").to_string());
                self.show(
                    &format!(
                        "parent + the other {} changes, without the {} found",
                        rest.len(),
                        minimal.len()
                    ),
                    &entry,
                );
            } else {
                why.without = Some("does not apply".to_string());
            }
        }
        Ok(())
    }

    /// The `undo` test of [`Self::explain`].
    fn undo(
        &mut self,
        bad: &str,
        first_bad: &str,
        is_merge: bool,
        bad_name: &str,
        why: &mut Why,
    ) -> anyhow::Result<()> {
        let Palette { d, o, .. } = self.pal;
        let short = head(first_bad, 10);
        self.reset_to(bad)?;
        let mut revert = vec!["revert", "--no-commit", "--no-edit"];
        if is_merge {
            // Against the first parent, as its changes were read.
            revert.extend(["-m", "1"]);
        }
        revert.push(first_bad);
        if git_ok(&self.test.worktree, &revert) {
            why.undo_patch = self.staged_diff();
            let entry = self.probe_state(
                &format!("{} with {short} undone", head(bad, 10)),
                &format!("undo of the whole first bad commit on {bad_name}"),
                "undo",
                None,
            )?;
            why.revert = Some(text_of(&entry, "verdict").to_string());
            why.undo_how = Some("whole commit");
            self.show(&format!("{bad_name} with the whole commit undone"), &entry);
            return Ok(());
        }
        self.reset_to(bad)?;
        let found: Vec<&Unit> = why.minimal.iter().collect();
        if !found.is_empty() && self.apply(&patch_of(&found), true)? {
            why.undo_patch = self.staged_diff();
            let entry = self.probe_state(
                &format!(
                    "{} with {} found change(s) of {short} undone",
                    head(bad, 10),
                    found.len()
                ),
                &format!("undo of: {}", names_of(&found)),
                "undo",
                None,
            )?;
            why.revert = Some(text_of(&entry, "verdict").to_string());
            why.undo_how = Some("found lines only");
            println!(
                "  {d}the whole commit no longer undoes cleanly on {bad_name} (later commits touched other parts of it), so only the lines found are undone{o}"
            );
            self.show(&format!("{bad_name} with only those lines undone"), &entry);
        } else {
            why.revert = Some("rewritten".to_string());
            println!(
                "  {d}later commits rewrote these exact lines, so they cannot be undone mechanically on {bad_name}{o}"
            );
        }
        Ok(())
    }

    /// `--flaky` step 7: a fixed number of runs with the first bad commit
    /// and the same number without it.
    fn strength(&mut self, first_bad: &str, parent: &str, runs: u64) -> anyhow::Result<Strength> {
        let Palette { b, d, o, .. } = self.pal;
        let mut sides = Vec::with_capacity(2);
        for (name, commit) in [("with", first_bad), ("without", parent)] {
            self.checkout(commit, true)?;
            let entry = self.probe_state(
                &format!("{} x{runs}", head(commit, 10)),
                &format!("fixed-size run {name} the first bad commit"),
                "strength",
                Some(runs),
            )?;
            let side = Side {
                fails: count_of(&entry, "fails"),
                runs: count_of(&entry, "runs"),
                probe: count_of(&entry, "n"),
            };
            println!(
                "  {b}[tested]{o} {name} the commit ({}): failed {b}{} of {}{o} runs  {d}-> {}{o}",
                head(commit, 10),
                side.fails,
                side.runs,
                py_display(entry.get("memory")),
            );
            sides.push(side);
        }
        match sides.as_slice() {
            [with, without] => Ok(Strength::measured(*with, *without)),
            _ => anyhow::bail!("the strength runs did not give two sides"),
        }
    }
}

/// The last `n` characters of `s`.
fn tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    s.chars().skip(count.saturating_sub(n)).collect()
}

// ---------------------------------------------------------------------------
// What the command was asked, checked before anything is created
// ---------------------------------------------------------------------------

/// The `--flaky` settings, validated.
#[derive(Debug, Clone, Copy)]
struct Flaky {
    alpha: f64,
    max_runs: u64,
    baseline_max: u64,
    strength_runs: u64,
}

/// Everything the run was asked, resolved and checked.
struct Plan {
    failure: String,
    repo: PathBuf,
    store: PathBuf,
    out: PathBuf,
    patch_out: PathBuf,
    good_ref: String,
    bad_ref: String,
    good: String,
    bad: String,
    window: u64,
    good_is_ancestor: bool,
    reported_at: String,
    reported: DateTime<FixedOffset>,
    slug: String,
    flaky: Option<Flaky>,
    timeout: Option<Duration>,
    /// `--also-hash`: file name and sha256, in name order.
    also_hashed: Vec<(String, PathBuf, String)>,
}

fn refuse<T>(message: impl Into<String>) -> anyhow::Result<T> {
    Err(stop(1, message))
}

/// Where the undo patch goes: beside the report, `.json` replaced.
fn patch_path(out: &Path) -> anyhow::Result<PathBuf> {
    let text = utf8(out)?;
    Ok(PathBuf::from(match text.strip_suffix(".json") {
        Some(stem) => format!("{stem}.undo.patch"),
        None => format!("{text}.undo.patch"),
    }))
}

fn sha256_of_file(path: &Path, what: &str) -> anyhow::Result<String> {
    match fs::read(path) {
        Ok(bytes) => Ok(sha256_hex(&bytes)),
        Err(err) => refuse(format!("cannot read {what} {}: {err}", path.display())),
    }
}

fn flaky_settings(args: &ProveArgs) -> anyhow::Result<Option<Flaky>> {
    if !args.flaky {
        return Ok(None);
    }
    if !(args.alpha > 0.0 && args.alpha < 0.5) {
        return refuse(format!(
            "--alpha is the accepted chance of a wrong verdict on one commit: more than 0 and less than 0.5, not {}",
            args.alpha
        ));
    }
    if args.max_runs == 0 {
        return refuse("--max-runs must be at least 1");
    }
    for (flag, value) in [
        ("--baseline-max", args.baseline_max),
        ("--strength-runs", args.strength_runs),
    ] {
        if !(1..=MAX_FIXED_RUNS).contains(&value) {
            return refuse(format!(
                "{flag} must be from 1 to {MAX_FIXED_RUNS}, not {value}"
            ));
        }
    }
    Ok(Some(Flaky {
        alpha: args.alpha,
        max_runs: args.max_runs,
        baseline_max: args.baseline_max,
        strength_runs: args.strength_runs,
    }))
}

fn plan(data_dir: &Path, args: &ProveArgs) -> anyhow::Result<Plan> {
    let (Some(failure), Some(repo), Some(good_ref), Some(bad_ref), Some(reported_at), Some(report)) = (
        args.logged_write.as_deref(),
        args.repo.as_deref(),
        args.good.as_deref(),
        args.bad.as_deref(),
        args.reported_at.as_deref(),
        args.report.as_deref(),
    ) else {
        anyhow::bail!(
            "prove needs --logged-write, --repo, --good, --bad, --reported-at and --report, or --check <report>"
        );
    };
    let slug = args.slug.trim();
    if slug.is_empty() || slug.contains(',') {
        return refuse(
            "--slug becomes one tag on every memory of the run: it cannot be empty or hold a comma",
        );
    }
    let Ok(reported) = DateTime::parse_from_rfc3339(reported_at) else {
        return refuse(format!(
            "--reported-at wants RFC 3339, like 2026-04-06T19:21:50Z, not {reported_at}"
        ));
    };
    let flaky = flaky_settings(args)?;

    let out = abspath(&expand_user(report))?;
    let patch_out = patch_path(&out)?;
    for path in [&out, &patch_out] {
        if path.exists() {
            return refuse(format!("refusing to overwrite {}", path.display()));
        }
    }
    if !out.parent().is_some_and(Path::is_dir) {
        return refuse(format!(
            "the directory for --report does not exist: {}",
            out.display()
        ));
    }

    let repo = abspath(&expand_user(repo))?;
    if !repo.is_dir() {
        return refuse(format!("--repo is not a directory: {}", repo.display()));
    }
    if let Err(err) = git_out(&repo, &["rev-parse", "--git-dir"]) {
        return refuse(format!(
            "--repo is not a git repository: {} ({err:#})",
            repo.display()
        ));
    }
    let (Some(good), Some(bad)) = (
        resolve_commit(&repo, good_ref),
        resolve_commit(&repo, bad_ref),
    ) else {
        return refuse("cannot resolve --good/--bad");
    };
    let window: u64 = git_out(&repo, &["rev-list", "--count", &format!("{good}..{bad}")])?
        .parse()
        .context("git rev-list --count did not print a number")?;
    if window == 0 {
        return refuse(format!(
            "there is no commit in {good_ref}..{bad_ref}: --bad must hold commits --good does not. Are the two the wrong way round?"
        ));
    }
    let good_is_ancestor = git_test(&repo, &["merge-base", "--is-ancestor", &good, &bad])?;

    let mut also_hashed: Vec<(String, PathBuf, String)> = Vec::new();
    for file in &args.also_hash {
        let path = abspath(&expand_user(file))?;
        let name = file_name(&path);
        if name.is_empty() || also_hashed.iter().any(|(known, _, _)| *known == name) {
            return refuse(format!(
                "--also-hash files are recorded by file name, and {} has none or one already given",
                path.display()
            ));
        }
        let sha256 = sha256_of_file(&path, "the --also-hash file")?;
        also_hashed.push((name, path, sha256));
    }
    also_hashed.sort();

    Ok(Plan {
        failure: failure.to_string(),
        repo,
        store: abspath(data_dir)?,
        out,
        patch_out,
        good_ref: good_ref.to_string(),
        bad_ref: bad_ref.to_string(),
        good,
        bad,
        window,
        good_is_ancestor,
        reported_at: reported_at.to_string(),
        reported,
        slug: slug.to_string(),
        flaky,
        timeout: (args.timeout > 0).then(|| Duration::from_secs(args.timeout)),
        also_hashed,
    })
}

/// The test script: the `--oracle` file, or `--test` written out as one.
/// The user's own command is the only text here that a shell reads.
fn test_script(args: &ProveArgs, scratch: &Path) -> anyhow::Result<PathBuf> {
    if let Some(command) = &args.test {
        if command.trim().is_empty() {
            return refuse("--test is empty; give the one command that tests the bug");
        }
        let script = scratch.join("test-command.sh");
        fs::write(&script, format!("#!/bin/sh\n{command}\n"))
            .context("cannot write the test script")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&script, fs::Permissions::from_mode(0o755))
                .context("cannot make the test script executable")?;
        }
        return Ok(script);
    }
    let Some(script) = &args.oracle else {
        return refuse("give --test 'one command' or --oracle script");
    };
    let script = abspath(&expand_user(script))?;
    let Ok(meta) = fs::metadata(&script) else {
        return refuse(format!("there is no test script at {}", script.display()));
    };
    if !meta.is_file() {
        return refuse(format!(
            "the test script is not a file: {}",
            script.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return refuse(format!(
                "the test script is not executable: {} (chmod +x it)",
                script.display()
            ));
        }
    }
    Ok(script)
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// What steps 3 to 7 established, all of it in the worktree.
struct Tested {
    boundary: Option<Lead>,
    runs_before_bisect: usize,
    first_bad: Option<String>,
    /// The parents of the first bad commit.
    parents: Vec<String>,
    aborted: Option<String>,
    plain_runs: Option<u64>,
    why: Option<Why>,
    strength: Option<Strength>,
}

/// Run `vestige prove`. Returns the process exit code: 0 when a first bad
/// commit was named and the report written, 1 when the run was refused,
/// could not decide, or git bisect named no commit.
pub fn prove(storage: &Arc<Storage>, data_dir: &Path, args: &ProveArgs) -> anyhow::Result<i32> {
    match prove_inner(storage, data_dir, args) {
        Ok(code) => Ok(code),
        Err(err) => match err.downcast::<Stop>() {
            Ok(stop) => {
                println!("{}", stop.message);
                Ok(stop.code)
            }
            Err(err) => Err(err),
        },
    }
}

fn prove_inner(storage: &Arc<Storage>, data_dir: &Path, args: &ProveArgs) -> anyhow::Result<i32> {
    let pal = Palette::detect();
    let Palette {
        b,
        d,
        c,
        g,
        r,
        y,
        o,
    } = pal;
    let plan = plan(data_dir, args)?;
    let Plan {
        failure,
        repo,
        good_ref,
        bad_ref,
        good,
        bad,
        window,
        ..
    } = &plan;

    let tmp = tempfile::Builder::new()
        .prefix("walk-verify-")
        .tempdir()
        .context("cannot create a temporary directory")?
        .keep();
    let scratch = ScratchGuard::arm(repo.clone(), tmp.clone());

    let oracle = test_script(args, &tmp)?;
    let oracle_sha256 = sha256_of_file(&oracle, "the test script")?;
    let worktree = tmp.join("checkout");
    let files = ChildFiles {
        cfg: tmp.join("cfg.json"),
        pending: tmp.join("handed-back.jsonl"),
        errors: tmp.join("bisect.stderr"),
        counter: tmp.join("sim.count"),
    };
    let session_path = tmp.join("probes.jsonl");
    let mut child_config = ChildConfig {
        repo: repo.clone(),
        worktree: worktree.clone(),
        oracle: oracle.clone(),
        oracle_sha256: oracle_sha256.clone(),
        timeout_seconds: plan.timeout.map(|limit| limit.as_secs()),
        session: session_path.clone(),
        pending: files.pending.clone(),
        stats: None,
        replay: None,
    };
    child_config.write(&files.cfg)?;

    println!(
        "\n{b}Step 1. The walk proposes{o}  {d}$ vestige causal-walk --logged-write {failure}{o}"
    );
    let walk = walk_from(storage, failure)?;
    let mut leads = leads_of(&walk);
    println!(
        "  {window} commits between {good_ref} and {bad_ref}. The walk reaches {} of them over recorded links.",
        leads.len()
    );
    if leads.is_empty()
        && let Some(reason) = walk["emptyBecause"].as_str()
    {
        println!("  {d}{reason}{o}");
    }
    if !plan.good_is_ancestor {
        println!(
            "  {d}{good_ref} is not an ancestor of {bad_ref}: git bisect tests their merge base first, and stops if the test already fails there{o}"
        );
    }

    let mut kept: Vec<Lead> = Vec::new();
    let mut dropped: Vec<(Lead, String)> = Vec::new();
    for lead in &mut leads {
        let Some(full) = resolve_commit(repo, &lead.short) else {
            dropped.push((lead.clone(), "not a commit in this repo".to_string()));
            continue;
        };
        lead.commit = Some(full.clone());
        let when = commit_field(repo, &full, "%cI")?;
        let Ok(committed) = DateTime::parse_from_rfc3339(&when) else {
            dropped.push((
                lead.clone(),
                "its commit date could not be read".to_string(),
            ));
            continue;
        };
        if committed > plan.reported {
            dropped.push((
                lead.clone(),
                format!("committed {}, after the report", head(&when, 10)),
            ));
            continue;
        }
        let in_bad = git_test(repo, &["merge-base", "--is-ancestor", &full, bad])?;
        let in_good = git_test(repo, &["merge-base", "--is-ancestor", &full, good])?;
        if !in_bad || in_good {
            dropped.push((lead.clone(), format!("outside {good_ref}..{bad_ref}")));
            continue;
        }
        kept.push(lead.clone());
    }
    println!("\n{b}Step 2. Time gate{o}  a cause cannot come after its effect");
    for (lead, reason) in &dropped {
        println!("  {d}dropped {}: {reason}{o}", lead.short);
    }
    println!(
        "  {} candidate(s) kept, {} dropped.",
        kept.len(),
        dropped.len()
    );
    for lead in kept.iter().take(args.show) {
        let commit = lead.commit.as_deref().unwrap_or(&lead.short);
        println!(
            "  {d}[recorded link]{o} #{} depth {}  {c}{} {}{o}",
            lead.rank,
            lead.depth,
            lead.short,
            head(&subject_of(repo, commit), 86)
        );
    }
    if kept.len() > args.show {
        println!("  {d}... and {} more{o}", kept.len() - args.show);
    }

    // Frozen before the first test: what is tested, with what, on which leads.
    let kept_commits: Vec<&str> = kept
        .iter()
        .filter_map(|lead| lead.commit.as_deref())
        .collect();
    let also_hashed: Map<String, Value> = plan
        .also_hashed
        .iter()
        .map(|(name, _, sha256)| (name.clone(), json!(sha256)))
        .collect();
    let mut protocol = json!({
        "slug": plan.slug,
        "failure_memory": failure,
        "good": good,
        "bad": bad,
        "oracle_sha256": oracle_sha256,
        "rule": PROTOCOL_RULE,
        "candidates": kept_commits,
        "max_candidates": args.max_candidates,
        "max_line_runs": args.max_line_runs,
        "also_hashed": also_hashed,
        "flaky": plan.flaky.map(|flaky| json!({
            "alpha": flaky.alpha,
            "max_runs_per_commit": flaky.max_runs,
            "baseline_max": flaky.baseline_max,
            "strength_runs": flaky.strength_runs,
        })),
        "frozen_at": now_stamp(),
    });
    let protocol_sha256 = protocol.as_object().map(protocol_hash).unwrap_or_default();
    let protocol_text = format!(
        "Protocol ({}), frozen before any test run: failure {failure}, good {}, bad {}, repro script sha256 {}, rule: {PROTOCOL_RULE}. {} candidates in walk order: {}. Protocol sha256 {protocol_sha256}. Nothing in this protocol may change after the first test.",
        plan.slug,
        head(good, 10),
        head(bad, 10),
        head(&oracle_sha256, 16),
        kept.len(),
        kept_commits
            .iter()
            .map(|commit| head(commit, 10))
            .collect::<Vec<_>>()
            .join(" "),
    );
    let protocol_memory = match remember(
        storage.as_ref(),
        &protocol_text,
        &["oracle-protocol", plan.slug.as_str()],
    ) {
        Ok(id) => Some(id),
        Err(err) => {
            println!(
                "\n  {y}warning: the protocol was not saved as a memory: {err}. Its sha256 is still written to the report.{o}"
            );
            None
        }
    };
    println!(
        "\n{b}Protocol frozen before any test{o}  the script, the two ends and the {} candidates are on record  {d}-> {}  sha256 {}{o}",
        kept.len(),
        opt_display(protocol_memory.as_deref()),
        head(&protocol_sha256, 16)
    );
    protocol["sha256"] = json!(protocol_sha256);
    protocol["memory"] = json!(protocol_memory);

    // Oldest to newest. Step 4 lines the leads up in this order, under git
    // bisect's own assumption: one first bad commit, everything after it bad.
    let order = git_out(
        repo,
        &[
            "rev-list",
            "--topo-order",
            "--reverse",
            &format!("{good}..{bad}"),
        ],
    )?;
    let position: HashMap<&str, usize> = order
        .split_whitespace()
        .enumerate()
        .map(|(index, sha)| (sha, index))
        .collect();

    let added = git_command(repo)
        .args(["worktree", "add", "-q", "--detach"])
        .arg(&worktree)
        .arg(bad)
        .output()
        .context("cannot run git")?;
    if !added.status.success() {
        return refuse(format!(
            "worktree failed: {}",
            strip(&String::from_utf8_lossy(&added.stderr))
        ));
    }
    scratch.worktree_added(&worktree);

    let mut prover = Prover {
        storage: storage.as_ref(),
        pal,
        repo: repo.clone(),
        test: Test {
            oracle: oracle.clone(),
            worktree: worktree.clone(),
            timeout: plan.timeout,
        },
        oracle_sha256: oracle_sha256.clone(),
        slug: plan.slug.clone(),
        session: Session::new(session_path),
        stats: None,
        unsaved: 0,
    };

    // Steps 3 to 7 need the worktree; it is removed before the result is
    // written, whatever they return.
    let mut steps = || -> anyhow::Result<Option<Tested>> {
        println!(
            "\n{b}Step 3. The repro script must tell the two ends apart{o}  {d}{}{o}",
            file_name(&oracle)
        );
        if let Some(flaky) = plan.flaky {
            let p_value = prover.flaky_baseline(
                (good.as_str(), good_ref.as_str()),
                (bad.as_str(), bad_ref.as_str()),
                flaky.baseline_max,
                flaky.alpha,
                flaky.max_runs,
            )?;
            // A NaN p (counts too large) is not below the bar either.
            let beyond_chance = p_value < BASELINE_P;
            let separates = prover.stats.is_some_and(|stats| stats.separates());
            if !beyond_chance || !separates {
                println!(
                    "{r}After {} runs on each end the failure rates are not clearly different (p = {}), so nothing can be decided with this test.{o}",
                    flaky.baseline_max,
                    format_g(p_value, 3)
                );
                return Ok(None);
            }
            child_config.stats = prover.stats.map(Into::into);
            child_config.write(&files.cfg)?;
            println!(
                "  {d}The two ends differ beyond chance (p = {}). Each commit is now tested repeatedly until the evidence is decisive at {}.{o}",
                format_g(p_value, 2),
                format_g(flaky.alpha, 6)
            );
        } else {
            let on_good = prover.probe(good, "baseline")?;
            let on_bad = prover.probe(bad, "baseline")?;
            if text_of(&on_good, "verdict") != "good" || text_of(&on_bad, "verdict") != "bad" {
                println!(
                    "{r}The script does not pass on {good_ref} and fail on {bad_ref}, so nothing can be decided with it.{o}"
                );
                return Ok(None);
            }
        }

        println!(
            "\n{b}Step 4. Bisect the leads, closest links first, then test the parent{o}"
        );
        let mut ordered: Vec<(usize, Lead)> = kept
            .iter()
            .filter_map(|lead| {
                let at = position.get(lead.commit.as_deref()?)?;
                Some((*at, lead.clone()))
            })
            .collect();
        ordered.sort_by_key(|(at, _)| *at);
        let ordered: Vec<Lead> = ordered.into_iter().map(|(_, lead)| lead).collect();
        let boundary = prover.bisect_leads(&ordered, args.max_candidates)?;
        let runs_before_bisect = prover.session.entries.len();

        println!(
            "\n{b}Step 5. Confirm with stock git bisect over all {window} commits{o}  {d}$ git bisect run{o}"
        );
        let Bisected { first_bad, aborted } = prover.bisect(&files, good, bad)?;
        let mut tested = Tested {
            boundary,
            runs_before_bisect,
            first_bad: None,
            parents: Vec::new(),
            aborted,
            plain_runs: None,
            why: None,
            strength: None,
        };
        let Some(first_bad) = first_bad else {
            return Ok(Some(tested));
        };
        tested.parents = parents_of(repo, &first_bad)?;
        child_config.replay = Some(Replay {
            first_bad: first_bad.clone(),
            counter: files.counter.clone(),
        });
        child_config.write(&files.cfg)?;
        tested.plain_runs = prover.replay_plain_bisect(&files, good, bad, &first_bad)?;
        if !args.no_why {
            println!(
                "\n{b}Step 6. Why: find the lines, test the commit without them, undo them on the broken version{o}"
            );
            tested.why = Some(prover.explain(
                bad,
                &first_bad,
                &tested.parents,
                args.max_line_runs,
                bad_ref,
            )?);
        }
        if let Some(flaky) = plan.flaky {
            let runs = flaky.strength_runs;
            println!(
                "\n{b}Step 7. How strong: {runs} runs with the commit, {runs} runs without{o}"
            );
            match tested.parents.first() {
                Some(parent) => {
                    tested.strength = Some(prover.strength(&first_bad, parent, runs)?);
                }
                None => println!(
                    "  {d}{} is a root commit: there is no version without it to compare against{o}",
                    head(&first_bad, 10)
                ),
            }
        }
        tested.first_bad = Some(first_bad);
        Ok(Some(tested))
    };
    let tested = steps();
    remove_worktree();
    let Some(tested) = tested? else {
        return Ok(1);
    };
    let entries = prover.session.entries.clone();
    let unsaved = prover.unsaved;
    let stats = prover.stats;
    let Tested {
        boundary,
        runs_before_bisect,
        first_bad,
        parents,
        aborted,
        plain_runs,
        why,
        strength,
    } = tested;

    // The protocol said what the test was. If the file is not that any
    // more, the runs above were not all runs of one test.
    let mut frozen = vec![("the test script", oracle.clone(), oracle_sha256.clone())];
    frozen.extend(
        plan.also_hashed
            .iter()
            .map(|(_, path, sha256)| ("the --also-hash file", path.clone(), sha256.clone())),
    );
    for (what, path, sha256) in &frozen {
        if fs::read(path).ok().map(|bytes| sha256_hex(&bytes)).as_ref() != Some(sha256) {
            return refuse(format!(
                "{r}{what} {} changed while the run was in progress, so its runs are not runs of the frozen protocol. No report was written.{o}",
                path.display()
            ));
        }
    }

    println!("\n{b}Result{o}");
    let mut subject = String::new();
    if let Some(first_bad) = &first_bad {
        subject = subject_of(repo, first_bad);
        let when = commit_field(repo, first_bad, "%cs").unwrap_or_default();
        println!(
            "  {b}[tested]{o} git bisect: {g}{}{o} is the first bad commit",
            head(first_bad, 10)
        );
        println!("           {b}{}{o}  {d}({when}){o}", head(&subject, 100));
    } else {
        println!(
            "  {r}git bisect did not name a first bad commit.{o} Its output:\n{}",
            aborted.as_deref().unwrap_or_default()
        );
    }
    let agree = boundary
        .as_ref()
        .is_some_and(|lead| first_bad.is_some() && lead.commit == first_bad);
    let reached = first_bad.as_ref().and_then(|first_bad| {
        leads
            .iter()
            .find(|lead| lead.commit.as_ref() == Some(first_bad))
    });
    if let Some(lead) = boundary.as_ref().filter(|_| agree) {
        println!(
            "  The walk had this commit as lead #{} of {}. git bisect, run on its own, names the same commit.",
            lead.rank,
            leads.len()
        );
    } else if let Some(lead) = reached {
        println!(
            "  The walk reached this commit as lead #{} of {} (depth {}).",
            lead.rank,
            leads.len(),
            lead.depth
        );
    } else if first_bad.is_some() {
        println!(
            "  {y}The walk did not reach this commit. The tested result stands; the walk missed it.{o}"
        );
    }
    let found_in = runs_before_bisect.saturating_sub(2);
    if agree && let Some(plain) = plain_runs.filter(|plain| *plain > 0) {
        let what = if plan.flaky.is_some() {
            format!("{found_in} commits tested")
        } else {
            format!("{found_in} test runs")
        };
        println!(
            "  Found in {b}{what}{o} on the walk's {} leads. Plain git bisect needs {plain} on the same {window} commits {d}(replayed against the tested answer){o}.",
            kept.len()
        );
    }
    if let Some(strength) = &strength {
        println!(
            "  {b}[tested]{o} With this commit the test fails {b}{} of {}{o} times. Without it, {b}{} of {}{o}.",
            strength.with.fails, strength.with.runs, strength.without.fails, strength.without.runs
        );
        println!(
            "  {b}[tested]{o} Chance of that split if the commit made no difference: {b}{}{o}. Failure is at least {b}{:.1} times{o} more likely with it {d}(95% confidence, exact binomial bounds){o}.",
            format_g(strength.fisher_p, 2),
            strength.at_least_times.unwrap_or(0.0)
        );
    }
    if let Some(why) = &why {
        let count = why.minimal.len();
        let that = if count == 1 {
            "that change"
        } else {
            "those changes"
        };
        if count > 0 && why.units > 1 {
            println!(
                "  {b}[tested]{o} Of its {} changes, {b}{count} {} enough to cause it{o}{}:",
                why.units,
                if count == 1 { "is" } else { "are" },
                if why.complete {
                    ""
                } else {
                    " (run budget reached, may shrink further)"
                }
            );
            for unit in &why.minimal {
                println!("    {c}{} line {}{o}", unit.file, unit.start);
                for line in unit.added.iter().take(8) {
                    println!("      {g}+ {}{o}", head(line, 110));
                }
            }
        }
        match why.without.as_deref() {
            Some("good") => {
                println!("  {b}[tested]{o} The rest of the commit, without {that}, passes.")
            }
            Some("bad") => println!(
                "  {b}[tested]{o} The rest of the commit still fails without {that}, so more than one part of it carries the bug."
            ),
            _ => {}
        }
        match (why.revert.as_deref(), why.undo_how) {
            (Some("good"), Some("whole commit")) => {
                println!("  {b}[tested]{o} Undo this one commit on {bad_ref} and the bug is gone.");
            }
            (Some("good"), _) => {
                println!("  {b}[tested]{o} Undo just {that} on {bad_ref} and the bug is gone.");
            }
            (Some("bad"), _) => println!(
                "  {b}[tested]{o} Undoing it on {bad_ref} does not fix it; something later also carries the bug."
            ),
            _ => {}
        }
    }
    let total_runs: u64 = entries
        .iter()
        .map(|entry| entry.get("runs").and_then(Value::as_u64).unwrap_or(1))
        .sum();
    if plan.flaky.is_some() {
        println!(
            "  {d}{} results recorded from {total_runs} test runs, including the two ends, the git bisect confirmation and the line search.{o}",
            entries.len()
        );
    } else {
        println!(
            "  {d}{} runs recorded in total, including the two ends, the git bisect confirmation and the line search.{o}",
            entries.len()
        );
    }

    // The rungs, each with whether it holds and the runs that back it.
    let mut card: Vec<Rung> = Vec::new();
    if let Some(first_bad) = &first_bad {
        card = verdict_card(&CardFacts {
            entries: &entries,
            first_bad,
            parent: parents.first().map(String::as_str),
            lead: reached.map(|lead| (lead.rank, leads.len(), lead.memory.as_str())),
            window: *window,
            bad_ref,
            why: why.as_ref(),
            strength: strength.as_ref(),
        });
        let newest_first = position.len() - position.get(first_bad.as_str()).copied().unwrap_or(0);
        let held = card.iter().filter(|rung| rung.holds).count();
        println!(
            "\n{b}Verdict{o}  commit {g}{}{o}  {b}{held} of {} rungs hold{o}",
            head(first_bad, 10),
            card.len()
        );
        for rung in &card {
            println!(
                "  {}{:<9}{o} {}  {}  {d}{}{o}",
                if rung.holds { g } else { y },
                rung.name,
                if rung.holds { "yes" } else { "no " },
                rung.statement,
                rung.proof
            );
        }
        println!(
            "  {d}In plain git log order this is commit {newest_first} of {window}. Protocol {} was frozen before the first test.{o}",
            opt_display(protocol_memory.as_deref())
        );
        println!(
            "  {d}Not claimed: why the authors made the change, whether other inputs fail too, or what the right fix is.{o}"
        );
    }

    let saved = entries.len() - unsaved.min(entries.len());
    let mut result_memory = None;
    let mut result_unsaved = false;
    if let Some(first_bad) = &first_bad {
        let walk_said = match reached {
            Some(lead) => format!("had reached this commit as lead #{}", lead.rank),
            None => "had not reached this commit".to_string(),
        };
        let lines_said = match &why {
            Some(why) => {
                let changes = why
                    .minimal
                    .iter()
                    .map(|unit| format!("{} line {}", unit.file, unit.start))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "Undo on the bad ref ({}) tested {}; the commit without the minimal changes tested {}. Minimal failing changes: {}.",
                    opt_display(why.undo_how),
                    opt_display(why.revert.as_deref()).to_uppercase(),
                    opt_display(why.without.as_deref()).to_uppercase(),
                    if changes.is_empty() {
                        "not narrowed"
                    } else {
                        &changes
                    }
                )
            }
            None => "Lines not searched.".to_string(),
        };
        let recorded = if unsaved == 0 {
            "each recorded as its own memory tagged oracle-probe".to_string()
        } else {
            format!("{saved} of them recorded as memories tagged oracle-probe")
        };
        let text = format!(
            "Tested result ({}): Commit {} ({}) is the first bad commit between {good_ref} and {bad_ref} according to git bisect run with repro script sha256 {}. {} probes, {recorded}. The causal walk from {failure} {walk_said}. This says which commit first makes the repro script fail. {lines_said}",
            plan.slug,
            head(first_bad, 10),
            head(&subject, 100),
            head(&oracle_sha256, 16),
            entries.len(),
        );
        match remember(
            storage.as_ref(),
            &text,
            &["oracle-result", plan.slug.as_str()],
        ) {
            Ok(id) => result_memory = Some(id),
            Err(err) => {
                result_unsaved = true;
                println!("  {y}warning: the result was not saved as a memory: {err}{o}");
            }
        }
    }

    let mut why_report = Value::Null;
    let mut undo_patch: Option<&[u8]> = None;
    if let Some(why) = &why {
        why_report = json!({
            "undo_on_bad": why.revert,
            "undo_how": why.undo_how,
            "commit_without_minimal": why.without,
            "changes_in_commit": why.units,
            "search_complete": why.complete,
            "minimal_failing_changes": why.minimal.iter().map(|unit| json!({
                "file": unit.file,
                "line": unit.start,
                "added": unit.added,
            })).collect::<Vec<_>>(),
        });
        if let Some(patch) = why.undo_patch.as_deref()
            && why.revert.as_deref() == Some("good")
        {
            why_report["undo_patch"] = json!({
                "file": file_name(&plan.patch_out),
                "applies_to": bad_ref,
                "sha256": sha256_hex(patch),
            });
            undo_patch = Some(patch);
        }
    }
    let flaky_report = match plan.flaky {
        Some(_) => json!({
            "stats": stats.map(|stats| json!({
                "p0": stats.p0,
                "p1": stats.p1,
                "alpha": stats.alpha,
                "beta": stats.beta,
                "max_runs": stats.max_runs,
            })),
            "strength": strength.as_ref().map(Strength::json),
            "test_runs_total": total_runs,
        }),
        None => Value::Null,
    };
    let report = json!({
        "tool": TOOL,
        "slug": plan.slug,
        "repo": utf8(repo)?,
        "good": {"ref": good_ref, "commit": good},
        "bad": {"ref": bad_ref, "commit": bad},
        "window_commits": window,
        "failure_memory": failure,
        "reported_at": plan.reported_at,
        "store": utf8(&plan.store)?,
        "oracle": {"file": file_name(&oracle), "sha256": oracle_sha256},
        "walk": {
            "reached": leads.len(),
            "kept": kept.len(),
            "dropped": dropped.iter().map(|(lead, reason)| json!({"commit": lead.short, "why": reason})).collect::<Vec<_>>(),
            "candidates": kept.iter().map(|lead| json!({
                "rank": lead.rank,
                "depth": lead.depth,
                "commit": lead.commit,
                "memory": lead.memory,
            })).collect::<Vec<_>>(),
        },
        "flaky": flaky_report,
        "protocol": protocol,
        "verdict_card": card.iter().map(Rung::json).collect::<Vec<_>>(),
        "probes": entries,
        "first_bad_commit": first_bad,
        "walk_lead_rank": reached.map(|lead| lead.rank),
        "plain_bisect_runs_replayed": plain_runs,
        "found_in_runs": found_in,
        "why": why_report,
        "result_memory": result_memory,
        "chain_head": entries.last().map(|entry| text_of(entry, "hash")),
        "reading": READING,
    });
    write_new(&plan.out, pretty_json(&report).as_bytes())?;
    if let Some(patch) = undo_patch {
        if let Err(err) = write_new(&plan.patch_out, patch) {
            // A report that names a patch nobody can find is worse than none.
            let _ = fs::remove_file(&plan.out);
            return Err(err);
        }
        println!(
            "  The undo that was tested, as a patch on {bad_ref}: {b}{}{o}",
            plan.patch_out.display()
        );
    }
    let memories = if unsaved == 0 {
        "Every run is a memory in the store".to_string()
    } else {
        format!(
            "{saved} of {} runs were saved as memories in the store",
            entries.len()
        )
    };
    let result = match (&result_memory, result_unsaved) {
        (Some(id), _) => format!(", result {id}"),
        (None, true) => ", the result was not".to_string(),
        (None, false) => String::new(),
    };
    println!(
        "\n  {memories}{result}. Report: {b}{}{o}",
        plan.out.display()
    );
    println!(
        "  {d}Anyone can re-check the report offline: vestige prove --check {}{o}\n",
        file_name(&plan.out)
    );
    drop(scratch);
    Ok(if first_bad.is_some() { 0 } else { 1 })
}

/// Write a file that must not exist yet. The report is refused up front
/// when it exists; this closes the gap between that check and the write.
fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            return refuse(format!(
                "refusing to overwrite {}, which appeared while the run was in progress",
                path.display()
            ));
        }
        Err(err) => {
            return Err(err).with_context(|| format!("cannot write {}", path.display()));
        }
    };
    file.write_all(bytes)
        .with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lead_is_a_memory_that_starts_with_a_commit_sha() {
        assert_eq!(
            commit_named("Commit 4a6c2c0ff8: Adding retries").as_deref(),
            Some("4a6c2c0ff8")
        );
        assert_eq!(
            commit_named("  Commit abcdef1 fix").as_deref(),
            Some("abcdef1")
        );
        assert_eq!(
            commit_named("Commit abcdef1\nbody").as_deref(),
            Some("abcdef1")
        );
        let full = "a".repeat(40);
        assert_eq!(commit_named(&format!("Commit {full}: x")), Some(full));
        assert_eq!(commit_named("Commit abcdef: too short"), None);
        assert_eq!(
            commit_named(&format!("Commit {}: too long", "a".repeat(41))),
            None
        );
        assert_eq!(commit_named("Commit abcdef1"), None);
        assert_eq!(commit_named("Commit ABCDEF1: upper"), None);
        assert_eq!(commit_named("Commit abcdef1\u{e9}"), None);
        assert_eq!(commit_named("Issue 4026: Commit abcdef1: x"), None);
        assert_eq!(commit_named(""), None);

        let walk = json!({"causes": [
            {"id": "mem-01", "depth": 1, "content": "Commit 1111111: one"},
            {"id": "mem-02", "depth": 1, "content": "Issue report, not a commit"},
            {"id": "mem-03", "depth": 2, "content": "Commit 3333333: three"},
        ]});
        let leads = leads_of(&walk);
        assert_eq!(leads.len(), 2);
        assert_eq!((leads[0].rank, leads[0].short.as_str()), (1, "1111111"));
        assert_eq!(
            (leads[1].rank, leads[1].depth, leads[1].memory.as_str()),
            (3, 2, "mem-03")
        );
        // A walk with no causes, or none at all, gives no leads.
        assert!(leads_of(&json!({"causes": []})).is_empty());
        assert!(leads_of(&json!({})).is_empty());
        assert!(leads_of(&json!({"causes": "not a list"})).is_empty());
    }

    #[test]
    fn the_undo_patch_goes_beside_the_report() {
        assert_eq!(
            patch_path(Path::new("/r e p/redis-171333.json")).unwrap(),
            Path::new("/r e p/redis-171333.undo.patch")
        );
        assert_eq!(
            patch_path(Path::new("/reports/plain")).unwrap(),
            Path::new("/reports/plain.undo.patch")
        );
    }

    #[test]
    fn a_report_is_never_written_over_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        write_new(&path, b"first").unwrap();
        let err = write_new(&path, b"second").unwrap_err();
        assert!(err.downcast_ref::<Stop>().is_some(), "{err:#}");
        assert!(err.to_string().contains("refusing to overwrite"), "{err}");
        assert_eq!(fs::read(&path).unwrap(), b"first");
        // A directory that is not there is an error, not a refusal.
        let err = write_new(&dir.path().join("no/such/dir.json"), b"x").unwrap_err();
        assert!(err.downcast_ref::<Stop>().is_none());
    }

    #[test]
    fn the_tail_is_cut_on_characters() {
        assert_eq!(tail("abcdef", 3), "def");
        assert_eq!(tail("h\u{e9}llo", 4), "\u{e9}llo");
        assert_eq!(tail("ab", 5), "ab");
        assert_eq!(tail("", 5), "");
    }
}
