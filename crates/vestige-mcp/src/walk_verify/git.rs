//! Every `git` call of `prove`, and the cleanup of what it creates.
//!
//! git is always run as an argument vector, never through a shell, and
//! always with `-C <dir>` and without the variables that would point it at
//! another repository. The only things created outside the store are a
//! scratch directory under the system temp directory and, inside it, one
//! detached worktree of the user's repository; [`ScratchGuard`] removes
//! both on every way out.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use anyhow::Context;

use super::text::{head, stop, strip};

/// The variable git passes `-c` settings to its own subprocesses in. The
/// test is not one of them: it runs with the configuration the user has.
pub(super) const CONFIG_ENV: &str = "GIT_CONFIG_PARAMETERS";

/// The variables that tell git which repository, worktree, index or object
/// store to act on. `git bisect run` exports them to its command; inherited
/// from a hook or an alias they would turn `git -C <worktree> checkout -f`
/// onto some other checkout. Neither git as run here nor the user's test
/// ever sees them.
pub(super) const REPO_ENV: [&str; 8] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
];

/// Where `core.hooksPath` points for every git call of `prove`: nowhere.
#[cfg(not(windows))]
const NO_HOOKS: &str = "core.hooksPath=/dev/null";
#[cfg(windows)]
const NO_HOOKS: &str = "core.hooksPath=NUL";

/// `git -C <dir>`, reading nothing from stdin and running no hook.
///
/// A worktree shares its repository's hooks, so every checkout made here or
/// by `git bisect` would fire the user's `post-checkout` (and an undo their
/// commit hooks). The setting travels to the git processes `git bisect run`
/// starts through `GIT_CONFIG_PARAMETERS`; the user's test does not see it
/// (see [`CONFIG_ENV`]).
pub(super) fn git_command(dir: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .args(["-c", NO_HOOKS])
        .arg("-C")
        .arg(dir)
        .stdin(Stdio::null());
    for name in REPO_ENV {
        command.env_remove(name);
    }
    command
}

fn failure(args: &[&str], output: &Output) -> String {
    let said = String::from_utf8_lossy(&output.stderr);
    let said = strip(&said);
    format!(
        "git {} failed{}{}",
        args.join(" "),
        if said.is_empty() { "" } else { ": " },
        head(said, 400)
    )
}

/// Run git and return what it wrote to stdout, as bytes.
pub(super) fn git_bytes(dir: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let output = git_command(dir)
        .args(args)
        .output()
        .context("cannot run git; is it installed and on PATH?")?;
    anyhow::ensure!(output.status.success(), failure(args, &output));
    Ok(output.stdout)
}

/// Run git and return its stdout, stripped.
pub(super) fn git_out(dir: &Path, args: &[&str]) -> anyhow::Result<String> {
    let stdout = git_bytes(dir, args)?;
    Ok(strip(&String::from_utf8_lossy(&stdout)).to_string())
}

/// Run git for a yes or no: exit 0 is yes, exit 1 is no, anything else is
/// an error (a bad object name must not read as "no").
pub(super) fn git_test(dir: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let output = git_command(dir)
        .args(args)
        .output()
        .context("cannot run git; is it installed and on PATH?")?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(anyhow::anyhow!(failure(args, &output))),
    }
}

/// Run git where failing is one of the expected outcomes; whether it
/// succeeded.
pub(super) fn git_ok(dir: &Path, args: &[&str]) -> bool {
    git_command(dir)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Run git with `input` on its stdin; whether it succeeded.
pub(super) fn git_fed(dir: &Path, args: &[&str], input: &[u8]) -> anyhow::Result<bool> {
    let mut child = git_command(dir)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("cannot run git; is it installed and on PATH?")?;
    if let Some(mut stdin) = child.stdin.take() {
        // git may stop reading input it rejects; its exit status is the
        // answer either way.
        let _ = stdin.write_all(input);
    }
    Ok(child.wait().context("cannot wait for git")?.success())
}

/// Whether `text` is a full object name, SHA-1 or SHA-256.
pub(super) fn is_sha(text: &str) -> bool {
    matches!(text.len(), 40 | 64)
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The commit a revision names, or `None` when it names none. The revision
/// is user input. No revision starts with a dash, so one that does is
/// refused here instead of reaching git, where it would read as an option.
pub(super) fn resolve_commit(repo: &Path, revision: &str) -> Option<String> {
    if revision.is_empty() || revision.starts_with('-') {
        return None;
    }
    let name = format!("{revision}^{{commit}}");
    let output = git_command(repo)
        .args(["rev-parse", "--verify", "--quiet", &name])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let sha = strip(&String::from_utf8_lossy(&output.stdout)).to_string();
    (output.status.success() && is_sha(&sha)).then_some(sha)
}

/// The parents of a commit: none for a root commit, two or more for a merge.
pub(super) fn parents_of(repo: &Path, commit: &str) -> anyhow::Result<Vec<String>> {
    let line = git_out(repo, &["rev-list", "--parents", "-n", "1", commit])?;
    Ok(line
        .split_whitespace()
        .skip(1)
        .map(str::to_string)
        .collect())
}

/// One `--format` field of a commit. `--no-show-signature` keeps a
/// `log.showSignature` setting from adding lines to it.
pub(super) fn commit_field(repo: &Path, commit: &str, format: &str) -> anyhow::Result<String> {
    git_out(
        repo,
        &[
            "show",
            "-s",
            "--no-show-signature",
            &format!("--format={format}"),
            commit,
        ],
    )
}

/// A commit's subject line; empty when it cannot be read. It is a label on
/// a probe entry, never something a verdict depends on.
pub(super) fn subject_of(repo: &Path, commit: &str) -> String {
    commit_field(repo, commit, "%s").unwrap_or_default()
}

/// The options that make `git diff` write the same patch whatever the
/// user's configuration says: no color, no external diff or textconv
/// driver, `a/` and `b/` prefixes, three lines of context, rename detection
/// as git ships it, full paths, and binary changes as applicable patches.
pub(super) const PLAIN_DIFF: [&str; 12] = [
    "-c",
    "core.quotePath=false",
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--binary",
    "-M",
    "-U3",
    "--inter-hunk-context=0",
    "--src-prefix=a/",
    "--dst-prefix=b/",
];

/// The first bad commit a bisect log names. git writes this comment line
/// in English whatever the locale, unlike what it prints.
pub(super) fn first_bad_logged(log: &str) -> Option<String> {
    log.lines().rev().find_map(|line| {
        let rest = line.strip_prefix("# first bad commit: [")?;
        let (sha, _) = rest.split_once(']')?;
        is_sha(sha).then(|| sha.to_string())
    })
}

/// The commit a line of `git bisect` output names as first bad (English
/// locale only; [`first_bad_logged`] is the one relied on).
pub(super) fn first_bad_named(line: &str) -> Option<String> {
    let sha = line.strip_suffix(" is the first bad commit")?;
    is_sha(sha).then(|| sha.to_string())
}

// ---------------------------------------------------------------------------
// Cleanup: the worktree and the scratch directory always go away
// ---------------------------------------------------------------------------

struct Scratch {
    repo: PathBuf,
    dir: PathBuf,
    worktree: Option<PathBuf>,
}

static SCRATCH: Mutex<Option<Scratch>> = Mutex::new(None);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Remove the temporary worktree, and with it any bisect state: that lives
/// in the worktree's own administrative directory, which goes too.
pub(super) fn remove_worktree() {
    let taken = {
        let mut guard = SCRATCH.lock().unwrap_or_else(PoisonError::into_inner);
        guard
            .as_mut()
            .and_then(|scratch| Some((scratch.repo.clone(), scratch.worktree.take()?)))
    };
    let Some((repo, worktree)) = taken else {
        return;
    };
    let removed = git_command(&repo)
        .args(["worktree", "remove", "--force"])
        .arg(&worktree)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !removed || worktree.exists() {
        // The directory goes first; prune then drops the stale registration.
        let _ = fs::remove_dir_all(&worktree);
        git_ok(&repo, &["worktree", "prune"]);
    }
    if worktree.exists() {
        eprintln!(
            "warning: could not remove the temporary worktree {}; delete it and run `git worktree prune` in {}",
            worktree.display(),
            repo.display()
        );
    }
}

fn remove_scratch() {
    remove_worktree();
    let scratch = SCRATCH
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(scratch) = scratch {
        let _ = fs::remove_dir_all(&scratch.dir);
    }
}

/// Removes the worktree and the scratch directory when dropped, on every
/// path out of the run. A panic hook covers a build that aborts on panic,
/// and a signal flag turns Ctrl-C, SIGTERM and SIGHUP into an ordinary
/// error return. Nothing can be done about SIGKILL: what is left then is a
/// directory under the system temp directory and a worktree registration
/// that `git worktree prune` drops.
pub(super) struct ScratchGuard;

impl ScratchGuard {
    pub fn arm(repo: PathBuf, dir: PathBuf) -> Self {
        *SCRATCH.lock().unwrap_or_else(PoisonError::into_inner) = Some(Scratch {
            repo,
            dir,
            worktree: None,
        });
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            remove_scratch();
            previous(info);
        }));
        watch_signals();
        ScratchGuard
    }

    /// From here on there is a worktree to remove.
    pub fn worktree_added(&self, worktree: &Path) {
        if let Some(scratch) = SCRATCH
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_mut()
        {
            scratch.worktree = Some(worktree.to_path_buf());
        }
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        remove_scratch();
    }
}

#[cfg(unix)]
extern "C" fn on_signal(_signal: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Note Ctrl-C, SIGTERM and SIGHUP instead of dying on them. The test runs
/// in its own process group, so the run that is waiting for it sees the
/// flag, kills that group and returns; the cleanup then runs as usual.
#[cfg(unix)]
pub(super) fn watch_signals() {
    let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGHUP, handler);
    }
}

#[cfg(not(unix))]
pub(super) fn watch_signals() {}

/// An error once a signal asked the run to stop.
pub(super) fn interrupted() -> anyhow::Result<()> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(stop(
            130,
            "interrupted; the temporary worktree was removed.",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_is_never_pointed_at_another_repository_by_the_environment() {
        let command = git_command(Path::new("/some where/repo"));
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(args, ["-C", "/some where/repo"]);
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name.to_str().unwrap())
            .collect();
        let mut expected = REPO_ENV.to_vec();
        expected.sort_unstable();
        assert_eq!(removed, expected);
    }

    #[test]
    fn a_first_bad_commit_is_read_from_the_bisect_log() {
        let sha = "4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270";
        let log = format!(
            "# bad: [{bad}] Updating lib version\n# good: [{good}] Changing current version\ngit bisect start '{bad}' '{good}'\n# good: [{good}] x\ngit bisect good {good}\n# first bad commit: [{sha}] Adding retries [for] the connect\n",
            bad = "b".repeat(40),
            good = "a".repeat(40),
        );
        assert_eq!(first_bad_logged(&log).as_deref(), Some(sha));
        // A bisect that stopped early names none.
        let stopped = "# bad: [bbbb] x\ngit bisect start\n# only skipped commits left to test\n# possible first bad commit: [4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270] y\n";
        assert_eq!(first_bad_logged(stopped), None);
        assert_eq!(first_bad_logged(""), None);
        assert_eq!(first_bad_logged("# first bad commit: [nothex] x"), None);
        assert_eq!(first_bad_logged("# first bad commit: [4a6c2c0"), None);

        assert_eq!(
            first_bad_named(&format!("{sha} is the first bad commit")).as_deref(),
            Some(sha)
        );
        assert_eq!(
            first_bad_named("Bisecting: 3 revisions left to test after this"),
            None
        );
        assert_eq!(first_bad_named("4a6c2c0 is the first bad commit"), None);
    }

    /// A repository with two commits, the second a child of the first.
    fn tiny_repo(dir: &Path) -> (String, String) {
        let run = |args: &[&str]| {
            let output = git_command(dir)
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(args)
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .output()
                .expect("spawn git");
            assert!(output.status.success(), "git {args:?}");
        };
        run(&["init", "-q", "-b", "main"]);
        fs::write(dir.join("a.txt"), "one\n").unwrap();
        run(&["add", "-A"]);
        run(&["commit", "-q", "-m", "first subject"]);
        fs::write(dir.join("a.txt"), "two\n").unwrap();
        run(&["commit", "-q", "-am", "second subject"]);
        (
            git_out(dir, &["rev-parse", "HEAD~1"]).unwrap(),
            git_out(dir, &["rev-parse", "HEAD"]).unwrap(),
        )
    }

    #[test]
    fn revisions_parents_and_yes_no_questions_are_answered_or_refused() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("re po");
        fs::create_dir(&repo).unwrap();
        let (first, second) = tiny_repo(&repo);

        assert_eq!(
            resolve_commit(&repo, "HEAD").as_deref(),
            Some(second.as_str())
        );
        assert_eq!(
            resolve_commit(&repo, "main~1").as_deref(),
            Some(first.as_str())
        );
        assert_eq!(resolve_commit(&repo, "no-such-ref"), None);
        // Input that looks like an option is a revision that does not exist.
        assert_eq!(resolve_commit(&repo, "--all"), None);
        assert_eq!(resolve_commit(&repo, "-h"), None);
        assert_eq!(resolve_commit(&repo, ""), None);
        assert_eq!(resolve_commit(&dir.path().join("not-a-repo"), "HEAD"), None);

        assert_eq!(
            parents_of(&repo, &second).unwrap(),
            std::slice::from_ref(&first)
        );
        assert!(parents_of(&repo, &first).unwrap().is_empty());
        assert!(parents_of(&repo, &"0".repeat(40)).is_err());
        assert_eq!(subject_of(&repo, &second), "second subject");
        assert_eq!(subject_of(&repo, &"0".repeat(40)), "");

        let ancestor = |a: &str, b: &str| git_test(&repo, &["merge-base", "--is-ancestor", a, b]);
        assert!(ancestor(&first, &second).unwrap());
        assert!(!ancestor(&second, &first).unwrap());
        // An object that does not exist is an error, not a "no".
        assert!(ancestor(&"0".repeat(40), &second).is_err());

        let err = git_out(&repo, &["rev-parse", "--verify", "nope"]).unwrap_err();
        assert!(
            err.to_string()
                .starts_with("git rev-parse --verify nope failed"),
            "{err}"
        );
        assert!(git_ok(&repo, &["rev-parse", "--verify", "--quiet", "HEAD"]));
        assert!(!git_ok(
            &repo,
            &["rev-parse", "--verify", "--quiet", "nope"]
        ));

        // The plain diff of the second commit, and feeding it back.
        let mut args = PLAIN_DIFF.to_vec();
        args.extend([first.as_str(), second.as_str()]);
        let diff = git_bytes(&repo, &args).unwrap();
        let text = String::from_utf8(diff.clone()).unwrap();
        assert!(
            text.contains("--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-one\n+two\n"),
            "{text}"
        );
        assert!(git_fed(&repo, &["apply", "--check", "-R", "-"], &diff).unwrap());
        assert!(!git_fed(&repo, &["apply", "--check", "-"], &diff).unwrap());
        assert!(!git_fed(&repo, &["apply", "--check", "-"], b"not a patch").unwrap());
    }
}
