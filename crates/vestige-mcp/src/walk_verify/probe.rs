//! One test run, one verdict, and the hash-chained log they are written to.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::{Value, json};

use super::git::REPO_ENV;
use super::json::{Entry, ZERO_HASH, chain_hash};
use super::stats::{Stats, Wald};
use super::text::{head, last_line, now_stamp};

/// The exit code that means "cannot test this commit", as for `git bisect`.
pub(super) const CANNOT_TEST: i64 = 125;

/// A commit is "cannot test" once this many of its runs could not be tested.
const MAX_SKIPS: u32 = 5;

/// What a test's exit code means, as `git bisect run` reads it.
pub(super) fn verdict_of(exit: i64) -> &'static str {
    match exit {
        0 => "good",
        CANNOT_TEST => "skip",
        _ => "bad",
    }
}

/// The exit code that stands for a verdict.
fn exit_of(verdict: &str) -> i64 {
    match verdict {
        "good" => 0,
        "bad" => 1,
        _ => CANNOT_TEST,
    }
}

// ---------------------------------------------------------------------------
// The probe log
// ---------------------------------------------------------------------------

/// The probe log of one run: a JSONL file the bisect child processes read,
/// and the same entries in memory. Only the parent process appends.
pub(super) struct Session {
    path: PathBuf,
    pub entries: Vec<Entry>,
}

impl Session {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            entries: Vec::new(),
        }
    }

    /// The entries of a log file. A line that does not parse (one still
    /// being written) is not an entry yet.
    pub fn read(path: &Path) -> Vec<Entry> {
        let Ok(text) = fs::read_to_string(path) else {
            return Vec::new();
        };
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
            .collect()
    }

    pub fn cached(&self, commit: &str) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|entry| entry.get("commit").and_then(Value::as_str) == Some(commit))
    }

    /// Number the entry, link it to the one before, hash it, write it.
    pub fn append(&mut self, mut entry: Entry) -> anyhow::Result<Entry> {
        let prev = self
            .entries
            .last()
            .and_then(|last| last.get("hash"))
            .and_then(Value::as_str)
            .unwrap_or(ZERO_HASH)
            .to_string();
        entry.insert("n".to_string(), json!(self.entries.len() + 1));
        entry.insert("prev".to_string(), json!(prev));
        entry.remove("hash");
        let hash = chain_hash(&prev, &entry);
        entry.insert("hash".to_string(), json!(hash));
        append_line(&self.path, &Value::Object(entry.clone()))
            .with_context(|| format!("cannot write the probe log {}", self.path.display()))?;
        self.entries.push(entry.clone());
        Ok(entry)
    }
}

/// Append one JSON value as one line.
pub(super) fn append_line(path: &Path, value: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_string(value).map_err(std::io::Error::other)?;
    line.push('\n');
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())
}

pub(super) fn text_of<'a>(entry: &'a Entry, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or("")
}

pub(super) fn count_of(entry: &Entry, key: &str) -> u64 {
    entry.get(key).and_then(Value::as_u64).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// One run of the test
// ---------------------------------------------------------------------------

/// The test and where it runs.
#[derive(Debug, Clone)]
pub(super) struct Test {
    /// The executable test script.
    pub oracle: PathBuf,
    /// The temporary worktree the test runs in.
    pub worktree: PathBuf,
    /// A run that takes longer is killed and counts as "cannot test".
    /// `None` waits for as long as the test takes.
    pub timeout: Option<Duration>,
}

/// One run of the test.
pub(super) struct TestRun {
    pub exit: i64,
    /// The last line the test printed, or that it was killed for time.
    pub said: String,
    pub at: String,
}

#[cfg(unix)]
fn signal_of(status: &ExitStatus) -> i64 {
    use std::os::unix::process::ExitStatusExt;
    i64::from(status.signal().unwrap_or(1))
}

#[cfg(not(unix))]
fn signal_of(_status: &ExitStatus) -> i64 {
    1
}

/// The exit code, or minus the signal that ended the process.
fn exit_code(status: &ExitStatus) -> i64 {
    status.code().map_or_else(|| -signal_of(status), i64::from)
}

/// How much of the test's output is kept. Only the last line is used, so a
/// test that prints gigabytes costs a megabyte.
const KEPT_OUTPUT: usize = 1 << 20;

/// Read a stream to its end into `kept`, holding on to its tail only.
fn keep_tail(mut stream: impl Read, kept: &Mutex<Vec<u8>>) {
    let mut chunk = [0u8; 8192];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return,
            Ok(read) => {
                let mut kept = kept.lock().unwrap_or_else(PoisonError::into_inner);
                kept.extend_from_slice(&chunk[..read]);
                if kept.len() > 2 * KEPT_OUTPUT {
                    let cut = kept.len() - KEPT_OUTPUT;
                    kept.drain(..cut);
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return,
        }
    }
}

/// Start reading a stream on its own thread. The buffer fills as the test
/// prints; the receiver hears when the stream has ended.
fn tail_reader<R: Read + Send + 'static>(stream: R) -> (Arc<Mutex<Vec<u8>>>, mpsc::Receiver<()>) {
    let kept = Arc::new(Mutex::new(Vec::new()));
    let (done, ended) = mpsc::channel();
    let filled = Arc::clone(&kept);
    std::thread::spawn(move || {
        keep_tail(stream, &filled);
        let _ = done.send(());
    });
    (kept, ended)
}

/// Kill the test and everything it started that stayed in its group.
#[cfg(unix)]
fn kill_group(child: &mut std::process::Child) {
    if let Ok(group) = libc::pid_t::try_from(child.id()) {
        // SAFETY: the child has not been waited for yet, so its id still
        // names it and the process group it leads; killpg has no other
        // precondition.
        unsafe {
            libc::killpg(group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill_group(child: &mut std::process::Child) {
    let _ = child.kill();
}

/// The command of one test run: the script, in the worktree, in its own
/// process group, without the variables that tell git which repository to
/// act on. `git bisect run` exports those, and a test that itself runs git
/// must see the same environment in every phase.
fn test_command(test: &Test) -> Command {
    #[cfg(unix)]
    let mut command = Command::new(&test.oracle);
    #[cfg(not(unix))]
    let mut command = {
        let mut command = Command::new("sh");
        command.arg(&test.oracle);
        command
    };
    command.current_dir(&test.worktree).stdin(Stdio::null());
    for name in REPO_ENV {
        command.env_remove(name);
    }
    command.env_remove(super::git::CONFIG_ENV);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
}

/// A run that gave no verdict: it counts as "cannot test", and says why.
fn cannot_test(said: String, at: String) -> TestRun {
    TestRun {
        exit: CANNOT_TEST,
        said: head(&said, 160).to_string(),
        at,
    }
}

/// Run the test once in the worktree as it stands.
///
/// Only an exit code from 0 to 127 is a verdict. A test that cannot be
/// started, runs past the time limit (it is killed, with its process
/// group), or dies of a signal or with a code above 127 is "cannot test",
/// with the reason as its result line. An error is returned only when the
/// run itself cannot go on (interrupted, or the pipe cannot be made).
pub(super) fn run_test(test: &Test) -> anyhow::Result<TestRun> {
    let at = now_stamp();
    // One pipe for both streams, so the last line is the last line printed.
    let (output, writer) = std::io::pipe().context("cannot make a pipe for the test's output")?;
    let mut command = test_command(test);
    command
        .stdout(
            writer
                .try_clone()
                .context("cannot share the test's output pipe")?,
        )
        .stderr(writer);
    let spawned = command.spawn();
    // The command holds the writing ends; the reader sees the end of the
    // stream only once they are closed here too.
    drop(command);
    let mut child = match spawned {
        Ok(child) => child,
        Err(err) => {
            return Ok(cannot_test(
                format!("the test could not be started: {err}"),
                at,
            ));
        }
    };
    let (kept, ended) = tail_reader(output);

    let started = Instant::now();
    let mut pause = Duration::from_millis(1);
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().context("cannot wait for the test")? {
            break status;
        }
        if super::git::interrupted().is_err() {
            kill_group(&mut child);
            let _ = child.wait();
            super::git::interrupted()?;
        }
        if test.timeout.is_some_and(|limit| started.elapsed() >= limit) {
            kill_group(&mut child);
            timed_out = true;
            break child.wait().context("cannot wait for the test")?;
        }
        std::thread::sleep(pause);
        pause = (pause * 2).min(Duration::from_millis(50));
    };
    super::git::interrupted()?;
    if timed_out {
        let limit = test.timeout.unwrap_or_default().as_secs();
        return Ok(cannot_test(
            format!("timed out after {limit} seconds, counted as cannot test"),
            at,
        ));
    }

    // The stream ends when the last process holding it exits. A process the
    // test left running may hold it open for good; its output is not waited
    // for past a short grace.
    let _ = ended.recv_timeout(Duration::from_secs(2));
    let output = std::mem::take(&mut *kept.lock().unwrap_or_else(PoisonError::into_inner));
    let last = last_line(&output);
    let exit = exit_code(&status);
    if !(0..=127).contains(&exit) {
        return Ok(cannot_test(
            format!(
                "the test died abnormally (exit {exit}), counted as cannot test: {}",
                head(&last, 90)
            ),
            at,
        ));
    }
    Ok(TestRun {
        exit,
        said: last,
        at,
    })
}

// ---------------------------------------------------------------------------
// One verdict
// ---------------------------------------------------------------------------

/// How a verdict on the checked-out state is reached.
#[derive(Debug, Clone, Copy)]
pub(super) enum Mode<'a> {
    /// One run decides.
    Once,
    /// `--flaky`: repeat until Wald's sequential test is decisive.
    Sequential(&'a Stats),
    /// Exactly this many runs; any failure makes it bad.
    Fixed(u64),
}

/// A verdict on one state, from one run or from many.
pub(super) struct Outcome {
    pub verdict: &'static str,
    pub exit: i64,
    pub said: String,
    pub at: String,
    /// `(runs, fails)` when the verdict comes from repeated runs.
    pub counts: Option<(u64, u64)>,
}

impl Outcome {
    /// A verdict that stands for `fails` failures in `runs` runs.
    pub fn counted(verdict: &'static str, exit: i64, runs: u64, fails: u64, at: String) -> Self {
        Self {
            verdict,
            exit,
            said: format!("failed {fails} of {runs} runs"),
            at,
            counts: Some((runs, fails)),
        }
    }
}

/// One verdict for whatever is checked out now. Runs the test once, or
/// repeats it: until the evidence is decisive, or a fixed number of times.
/// A run that cannot be tested is not counted; after [`MAX_SKIPS`] of them
/// the repeating stops.
pub(super) fn measure(test: &Test, mode: Mode<'_>) -> anyhow::Result<Outcome> {
    let at = now_stamp();
    let (limit, mut wald) = match mode {
        Mode::Once => {
            let run = run_test(test)?;
            return Ok(Outcome {
                verdict: verdict_of(run.exit),
                exit: run.exit,
                said: run.said,
                at: run.at,
                counts: None,
            });
        }
        Mode::Sequential(stats) => (stats.max_runs, Some(Wald::new(stats))),
        Mode::Fixed(runs) => (runs, None),
    };
    let (mut runs, mut fails, mut skips) = (0u64, 0u64, 0u32);
    let mut last_failure = String::new();
    let mut last_skip = String::new();
    let mut decided = None;
    while runs < limit && skips < MAX_SKIPS {
        let run = run_test(test)?;
        let failed = match verdict_of(run.exit) {
            "skip" => {
                skips += 1;
                last_skip = run.said;
                continue;
            }
            verdict => verdict == "bad",
        };
        runs += 1;
        if failed {
            fails += 1;
            last_failure = run.said;
        }
        if let Some(wald) = wald.as_mut() {
            decided = wald.observe(failed);
            if decided.is_some() {
                break;
            }
        }
    }
    let verdict = match wald {
        Some(_) => decided.unwrap_or("skip"),
        None if fails > 0 => "bad",
        None => "good",
    };
    let mut said = format!("failed {fails} of {runs} runs");
    if !last_failure.is_empty() {
        said.push_str(", e.g. ");
        said.push_str(&last_failure);
    } else if runs == 0 && !last_skip.is_empty() {
        // Nothing could be tested: say what the test said instead.
        said.push_str(", ");
        said.push_str(&last_skip);
    }
    Ok(Outcome {
        verdict,
        exit: exit_of(verdict),
        said: head(&said, 160).to_string(),
        at,
        counts: Some((runs, fails)),
    })
}

/// A probe entry before it is saved and chained.
pub(super) fn new_entry(
    commit: &str,
    subject: &str,
    outcome: &Outcome,
    oracle_sha256: &str,
    phase: &str,
) -> Entry {
    let mut entry = Entry::new();
    entry.insert("commit".to_string(), json!(commit));
    entry.insert("subject".to_string(), json!(head(subject, 120)));
    entry.insert("verdict".to_string(), json!(outcome.verdict));
    entry.insert("exit".to_string(), json!(outcome.exit));
    entry.insert("oracle_said".to_string(), json!(outcome.said));
    entry.insert("oracle_sha256".to_string(), json!(oracle_sha256));
    entry.insert("phase".to_string(), json!(phase));
    entry.insert("at".to_string(), json!(outcome.at));
    if let Some((runs, fails)) = outcome.counts {
        entry.insert("runs".to_string(), json!(runs));
        entry.insert("fails".to_string(), json!(fails));
    }
    entry
}

/// The memory text of a probe. `on_commit` is a run on one checked-out
/// commit; otherwise the entry's `commit` is a label for a worktree state.
pub(super) fn probe_text(entry: &Entry, on_commit: bool) -> String {
    let what = if on_commit {
        format!("Commit {}", head(text_of(entry, "commit"), 10))
    } else {
        text_of(entry, "commit").to_string()
    };
    format!(
        "Oracle probe: {what} ({}) tested {}. Repro script exit code {}, result: {}. Script sha256 {}, run at {}, phase {}. A tested result for this one state, not a claim about any other.",
        head(text_of(entry, "subject"), 100),
        text_of(entry, "verdict").to_uppercase(),
        entry.get("exit").and_then(Value::as_i64).unwrap_or(0),
        text_of(entry, "oracle_said"),
        head(text_of(entry, "oracle_sha256"), 16),
        text_of(entry, "at"),
        text_of(entry, "phase"),
    )
}

#[cfg(test)]
mod tests {
    use super::super::json::tests::{FIRST_HASH, SECOND_HASH, first_entry, second_entry};
    use super::*;

    #[test]
    fn verdicts_follow_git_bisect() {
        assert_eq!(verdict_of(0), "good");
        assert_eq!(verdict_of(125), "skip");
        for exit in [1, 2, 124, 126, 127] {
            assert_eq!(verdict_of(exit), "bad", "exit {exit}");
        }
        // Outside 0..=127 the rule still reads bad; git bisect itself stops
        // on such a code during the confirmation.
        assert_eq!(verdict_of(137), "bad");
        assert_eq!(verdict_of(-9), "bad");
        assert_eq!(
            (exit_of("good"), exit_of("bad"), exit_of("skip")),
            (0, 1, 125)
        );
    }

    #[test]
    fn the_session_numbers_links_and_hashes_each_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::new(dir.path().join("probes.jsonl"));
        let mut one = first_entry();
        one.remove("n");
        one.remove("prev");
        let one = session.append(one).unwrap();
        assert_eq!(one["n"], 1);
        assert_eq!(one["prev"], ZERO_HASH);
        assert_eq!(one["hash"], FIRST_HASH);
        let mut two = second_entry();
        two.remove("n");
        two.remove("prev");
        let two = session.append(two).unwrap();
        assert_eq!(two["n"], 2);
        assert_eq!(two["prev"], FIRST_HASH);
        assert_eq!(two["hash"], SECOND_HASH);
        // What the bisect child reads back is what was appended.
        let read = Session::read(&session.path);
        assert_eq!(read, session.entries);
        assert!(
            session
                .cached("4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270")
                .is_some()
        );
        assert!(session.cached("4a6c2c0ff8").is_none());
        // A line cut off mid-write is not an entry.
        let mut cut = fs::read_to_string(&session.path).unwrap();
        cut.push_str("{\"commit\": \"half");
        fs::write(&session.path, cut).unwrap();
        assert_eq!(Session::read(&session.path).len(), 2);
        assert!(Session::read(&dir.path().join("missing.jsonl")).is_empty());
    }

    #[cfg(unix)]
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    fn test_in(dir: &Path, body: &str, timeout: Option<Duration>) -> Test {
        Test {
            oracle: script(dir, "test.sh", body),
            worktree: dir.to_path_buf(),
            timeout,
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_run_reports_the_exit_code_and_the_last_line() {
        let dir = tempfile::tempdir().unwrap();
        // Both streams, in the order they were printed.
        let run = run_test(&test_in(dir.path(), "echo one; echo two >&2; exit 3", None)).unwrap();
        assert_eq!((run.exit, run.said.as_str()), (3, "two"));
        let run = run_test(&test_in(dir.path(), "echo warn >&2; echo result", None)).unwrap();
        assert_eq!((run.exit, run.said.as_str()), (0, "result"));
        let run = run_test(&test_in(dir.path(), "pwd", None)).unwrap();
        assert_eq!(run.exit, 0);
        assert!(
            run.said
                .ends_with(&crate::walk_verify::text::file_name(dir.path()))
        );
        let run = run_test(&test_in(dir.path(), "exit 127", None)).unwrap();
        assert_eq!((run.exit, verdict_of(run.exit)), (127, "bad"));
        // More output than is kept does not block the test or the reader.
        let noisy =
            "i=0; while [ $i -lt 4000 ]; do printf '%01000d\\n' $i; i=$((i+1)); done; echo done";
        let run = run_test(&test_in(dir.path(), noisy, None)).unwrap();
        assert_eq!((run.exit, run.said.as_str()), (0, "done"));
    }

    #[cfg(unix)]
    #[test]
    fn only_an_exit_code_up_to_127_is_a_verdict() {
        let dir = tempfile::tempdir().unwrap();
        // Killed by a signal.
        let run = run_test(&test_in(dir.path(), "echo dying; kill -9 $$", None)).unwrap();
        assert_eq!((run.exit, verdict_of(run.exit)), (CANNOT_TEST, "skip"));
        assert_eq!(
            run.said,
            "the test died abnormally (exit -9), counted as cannot test: dying"
        );
        // A code git bisect would abort on.
        let run = run_test(&test_in(dir.path(), "exit 200", None)).unwrap();
        assert_eq!(run.exit, CANNOT_TEST);
        assert_eq!(
            run.said,
            "the test died abnormally (exit 200), counted as cannot test: "
        );
        // A test that cannot be started.
        let missing = Test {
            oracle: dir.path().join("no-such-test.sh"),
            worktree: dir.path().to_path_buf(),
            timeout: None,
        };
        let run = run_test(&missing).unwrap();
        assert_eq!(run.exit, CANNOT_TEST);
        assert!(
            run.said.starts_with("the test could not be started: "),
            "{}",
            run.said
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_run_does_not_get_the_variables_git_bisect_exports() {
        let dir = tempfile::tempdir().unwrap();
        let test = test_in(dir.path(), "true", None);
        let command = test_command(&test);
        // Exactly these nine are removed from what the test inherits, and
        // nothing else is set or removed: the eight that point git at a
        // repository, and `GIT_CONFIG_PARAMETERS` (the `-c` hooksPath
        // setting prove's own git calls carry must not reach the user's
        // test). (The end-to-end test runs a test that looks for them
        // under a real `git bisect run`.)
        let removed: Vec<&str> = command
            .get_envs()
            .map(|(name, value)| {
                assert!(value.is_none(), "{name:?} is set, not removed");
                name.to_str().unwrap()
            })
            .collect();
        assert_eq!(
            removed,
            [
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_COMMON_DIR",
                "GIT_CONFIG_PARAMETERS",
                "GIT_DIR",
                "GIT_INDEX_FILE",
                "GIT_NAMESPACE",
                "GIT_OBJECT_DIRECTORY",
                "GIT_PREFIX",
                "GIT_WORK_TREE",
            ]
        );
        assert_eq!(command.get_current_dir(), Some(dir.path()));
    }

    #[cfg(unix)]
    #[test]
    fn a_run_past_the_timeout_is_killed_with_its_children_and_cannot_test() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("child.pid");
        // The test starts a child of its own and then hangs.
        let body = format!(
            "sleep 300 & echo $! > '{}'; echo started; sleep 300",
            pid_file.display()
        );
        let started = Instant::now();
        let run = run_test(&test_in(dir.path(), &body, Some(Duration::from_secs(1)))).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(run.exit, CANNOT_TEST);
        assert_eq!(verdict_of(run.exit), "skip");
        assert_eq!(
            run.said,
            "timed out after 1 seconds, counted as cannot test"
        );
        // The child it started is gone too.
        let pid: libc::pid_t = fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let mut gone = false;
        for _ in 0..200 {
            // SAFETY: signal 0 only asks whether the process exists.
            if unsafe { libc::kill(pid, 0) } != 0 {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(gone, "the test's own child {pid} survived the timeout");
        // A test that finishes in time is not touched by the limit.
        let run = run_test(&test_in(
            dir.path(),
            "echo quick",
            Some(Duration::from_secs(30)),
        ))
        .unwrap();
        assert_eq!((run.exit, run.said.as_str()), (0, "quick"));
    }

    #[cfg(unix)]
    #[test]
    fn a_process_the_test_leaves_behind_does_not_hang_the_run() {
        let dir = tempfile::tempdir().unwrap();
        // The background sleep keeps the test's output open after it exits.
        let started = Instant::now();
        let run = run_test(&test_in(dir.path(), "sleep 6 & echo left; exit 0", None)).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert_eq!((run.exit, run.said.as_str()), (0, "left"));
    }

    /// A test that fails on every `period`-th run, counted in a file.
    #[cfg(unix)]
    fn every_nth(dir: &Path, period: u32) -> Test {
        let counter = dir.join("count");
        let body = format!(
            "n=$(cat '{0}' 2>/dev/null || echo 0); n=$((n + 1)); echo $n > '{0}'; if [ $((n % {period})) -eq 0 ]; then echo \"failed on run $n\"; exit 1; fi; echo ok",
            counter.display()
        );
        test_in(dir, &body, None)
    }

    #[cfg(unix)]
    #[test]
    fn repeated_runs_stop_when_the_evidence_is_decisive() {
        let stats = Stats::measured((0, 20), (7, 20), 0.01, 80);
        // Never fails: good after 33 runs.
        let dir = tempfile::tempdir().unwrap();
        let outcome = measure(
            &test_in(dir.path(), "echo ok", None),
            Mode::Sequential(&stats),
        )
        .unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("good", 0));
        assert_eq!(outcome.counts, Some((33, 0)));
        assert_eq!(outcome.said, "failed 0 of 33 runs");
        // Fails on every third run: bad after 9 runs, 3 of them failures.
        let dir = tempfile::tempdir().unwrap();
        let outcome = measure(&every_nth(dir.path(), 3), Mode::Sequential(&stats)).unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("bad", 1));
        assert_eq!(outcome.counts, Some((9, 3)));
        assert_eq!(outcome.said, "failed 3 of 9 runs, e.g. failed on run 9");
        // Out of runs before the evidence is decisive: cannot test.
        let capped = Stats {
            max_runs: 5,
            ..stats
        };
        let dir = tempfile::tempdir().unwrap();
        let outcome = measure(
            &test_in(dir.path(), "echo ok", None),
            Mode::Sequential(&capped),
        )
        .unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("skip", 125));
        assert_eq!(outcome.counts, Some((5, 0)));
    }

    #[cfg(unix)]
    #[test]
    fn a_fixed_number_of_runs_counts_every_failure() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = measure(&every_nth(dir.path(), 3), Mode::Fixed(30)).unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("bad", 1));
        assert_eq!(outcome.counts, Some((30, 10)));
        assert_eq!(outcome.said, "failed 10 of 30 runs, e.g. failed on run 30");
        let dir = tempfile::tempdir().unwrap();
        let outcome = measure(&test_in(dir.path(), "echo ok", None), Mode::Fixed(7)).unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("good", 0));
        assert_eq!(outcome.counts, Some((7, 0)));
    }

    #[cfg(unix)]
    #[test]
    fn runs_that_cannot_be_tested_are_not_counted_and_end_the_repeating() {
        let stats = Stats::measured((0, 20), (7, 20), 0.01, 80);
        let dir = tempfile::tempdir().unwrap();
        let test = test_in(dir.path(), "echo 'no compiler here'; exit 125", None);
        let outcome = measure(&test, Mode::Sequential(&stats)).unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("skip", 125));
        assert_eq!(outcome.counts, Some((0, 0)));
        assert_eq!(outcome.said, "failed 0 of 0 runs, no compiler here");
        let outcome = measure(&test, Mode::Fixed(50)).unwrap();
        assert_eq!(outcome.counts, Some((0, 0)));
        // One run decides without --flaky, whatever it says.
        let outcome = measure(&test, Mode::Once).unwrap();
        assert_eq!((outcome.verdict, outcome.exit), ("skip", 125));
        assert_eq!(outcome.counts, None);
        assert_eq!(outcome.said, "no compiler here");
    }

    #[test]
    fn an_entry_carries_run_counts_only_when_runs_were_repeated() {
        let once = Outcome {
            verdict: "bad",
            exit: 2,
            said: "boom".to_string(),
            at: "2026-10-05T23:45:31+00:00".to_string(),
            counts: None,
        };
        let entry = new_entry(
            &"a".repeat(40),
            &"s".repeat(200),
            &once,
            "feed",
            "candidate",
        );
        assert_eq!(entry.len(), 8);
        assert_eq!(text_of(&entry, "subject").len(), 120);
        assert_eq!(entry["exit"], 2);
        assert!(!entry.contains_key("runs"));
        assert_eq!(
            probe_text(&entry, true),
            format!(
                "Oracle probe: Commit aaaaaaaaaa ({}) tested BAD. Repro script exit code 2, result: boom. Script sha256 feed, run at 2026-10-05T23:45:31+00:00, phase candidate. A tested result for this one state, not a claim about any other.",
                "s".repeat(100)
            )
        );
        let counted = Outcome::counted("bad", 1, 30, 10, "2026-10-05T23:45:31+00:00".to_string());
        let entry = new_entry("abc x30", "fixed-size run", &counted, "feed", "strength");
        assert_eq!(
            (count_of(&entry, "runs"), count_of(&entry, "fails")),
            (30, 10)
        );
        assert_eq!(text_of(&entry, "oracle_said"), "failed 10 of 30 runs");
        assert!(
            probe_text(&entry, false)
                .starts_with("Oracle probe: abc x30 (fixed-size run) tested BAD.")
        );
    }

    #[ignore]
    #[cfg(unix)]
    #[test]
    fn catalog_item_59_a_hang_is_a_fail() {
        let root = PathBuf::from(std::env::var("HOME").unwrap())
            .join("Downloads/vestige-blind-spots-975d/item-59");
        std::fs::create_dir_all(&root).unwrap();
        let run = run_test(&test_in(
            &root,
            "sleep 30",
            Some(Duration::from_millis(200)),
        ))
        .unwrap();
        assert_eq!(verdict_of(run.exit), "bad", "catalog 59: {}", run.said);
        assert!(
            !run.said.contains("cannot test"),
            "catalog 59: a timeout is a fail: {}",
            run.said
        );
    }
}
