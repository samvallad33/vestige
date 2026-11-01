//! Tree hashes from local git. Every tree here leaves `.vestige/proof` out, so
//! a bundle committed beside the code never changes the tree it proves.
//!
//! The working tree is hashed through a private copy of the index kept inside
//! the git directory (`GIT_INDEX_FILE`): every change, untracked files
//! included and ignored files left out, is staged into the copy and written
//! with `git write-tree`. The user's own index is only ever read.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{PROOF_DIR, ProofError, err, is_object_id};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn git(dir: &Path, args: &[&str], index: Option<&Path>) -> Result<String, ProofError> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(dir).args(args);
    cmd.env("GIT_OPTIONAL_LOCKS", "0");
    cmd.env_remove("GIT_INDEX_FILE");
    if let Some(index) = index {
        cmd.env("GIT_INDEX_FILE", index);
    }
    let out = cmd
        .output()
        .map_err(|e| ProofError(format!("git could not run: {e}")))?;
    if !out.status.success() {
        return err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The repository root holding `dir`.
pub fn toplevel(dir: &Path) -> Result<PathBuf, ProofError> {
    git(dir, &["rev-parse", "--show-toplevel"], None).map(PathBuf::from)
}

/// The full id of the commit `rev` names.
pub fn commit_id(dir: &Path, rev: &str) -> Result<String, ProofError> {
    if rev.starts_with('-') {
        return err(format!("{rev:?} is not a revision"));
    }
    let id = git(
        dir,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{rev}^{{commit}}"),
        ],
        None,
    )?;
    if !is_object_id(&id) {
        return err(format!("{rev:?} does not name a commit"));
    }
    Ok(id)
}

/// A scratch index beside the real one, removed when dropped.
struct ScratchIndex(PathBuf);

impl ScratchIndex {
    fn new(dir: &Path, copy_real: bool) -> Result<Self, ProofError> {
        let real = git(dir, &["rev-parse", "--git-path", "index"], None)?;
        let real = if Path::new(&real).is_absolute() {
            PathBuf::from(real)
        } else {
            dir.join(real)
        };
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!(
            "index.vestige-witness-{}-{}-{nanos}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let path = real.with_file_name(name);
        if copy_real && real.is_file() {
            std::fs::copy(&real, &path)
                .map_err(|e| ProofError(format!("could not copy the index: {e}")))?;
        }
        Ok(Self(path))
    }
}

impl Drop for ScratchIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let mut lock = self.0.clone().into_os_string();
        lock.push(".lock");
        let _ = std::fs::remove_file(lock);
    }
}

fn write_tree(dir: &Path, index: &ScratchIndex) -> Result<String, ProofError> {
    git(
        dir,
        &[
            "rm",
            "-r",
            "--cached",
            "-q",
            "--ignore-unmatch",
            "--",
            &format!(":/{PROOF_DIR}"),
        ],
        Some(&index.0),
    )?;
    let tree = git(dir, &["write-tree"], Some(&index.0))?;
    if !is_object_id(&tree) {
        return err(format!("git write-tree gave {tree:?}"));
    }
    Ok(tree)
}

/// The tree of everything in the working directory now, as `git add -A`
/// followed by a commit would record it, without `.vestige/proof`.
pub fn working_tree(dir: &Path) -> Result<String, ProofError> {
    let index = ScratchIndex::new(dir, true)?;
    git(dir, &["add", "-A", "--", ":/"], Some(&index.0))?;
    write_tree(dir, &index)
}

/// A commit's tree without `.vestige/proof`.
pub fn commit_tree(dir: &Path, commit: &str) -> Result<String, ProofError> {
    let has_proof = !git(
        dir,
        &["ls-tree", "--full-tree", commit, "--", PROOF_DIR],
        None,
    )?
    .is_empty();
    if !has_proof {
        return git(dir, &["rev-parse", &format!("{commit}^{{tree}}")], None);
    }
    let index = ScratchIndex::new(dir, false)?;
    git(dir, &["read-tree", commit], Some(&index.0))?;
    write_tree(dir, &index)
}

/// The text of one file at a commit, `None` when the commit has no such file.
pub fn show_file(dir: &Path, commit: &str, path: &str) -> Option<String> {
    git(dir, &["show", &format!("{commit}:{path}")], None).ok()
}

/// The bundle stored as a git note on `commit`, if there is one.
pub fn read_note(dir: &Path, commit: &str) -> Option<String> {
    git(
        dir,
        &["notes", "--ref", crate::NOTES_REF, "show", commit],
        None,
    )
    .ok()
}

/// Store `text` as the git note on `commit`, replacing any earlier one.
pub fn write_note(dir: &Path, commit: &str, text: &str) -> Result<(), ProofError> {
    git(
        dir,
        &[
            "notes",
            "--ref",
            crate::NOTES_REF,
            "add",
            "-f",
            "-m",
            text,
            commit,
        ],
        None,
    )
    .map(|_| ())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn repo() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
            vec!["config", "commit.gpgsign", "false"],
        ] {
            git(dir.path(), &args, None).unwrap();
        }
        std::fs::write(dir.path().join("a.txt"), "one\n").unwrap();
        std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
        git(dir.path(), &["add", "-A"], None).unwrap();
        git(dir.path(), &["commit", "-q", "-m", "first"], None).unwrap();
        dir
    }

    pub(crate) fn run(dir: &Path, args: &[&str]) -> String {
        git(dir, args, None).unwrap()
    }

    #[test]
    fn a_clean_checkout_hashes_to_its_commit_tree_and_the_index_is_untouched() {
        let r = repo();
        let index = r.path().join(".git/index");
        let before = std::fs::read(&index).unwrap();
        let head = commit_id(r.path(), "HEAD").unwrap();
        let tree = working_tree(r.path()).unwrap();
        assert_eq!(tree, run(r.path(), &["rev-parse", "HEAD^{tree}"]));
        assert_eq!(commit_tree(r.path(), &head).unwrap(), tree);
        assert_eq!(
            std::fs::read(&index).unwrap(),
            before,
            "the user's index is never written"
        );
        let leftovers: Vec<_> = std::fs::read_dir(r.path().join(".git"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains("vestige-witness"))
            .collect();
        assert!(leftovers.is_empty(), "the scratch index is removed");
    }

    #[test]
    fn edits_and_untracked_files_change_the_tree_and_ignored_ones_do_not() {
        let r = repo();
        let clean = working_tree(r.path()).unwrap();
        std::fs::create_dir_all(r.path().join("target")).unwrap();
        std::fs::write(r.path().join("target/out.bin"), "x").unwrap();
        assert_eq!(
            working_tree(r.path()).unwrap(),
            clean,
            "ignored files are left out"
        );
        std::fs::write(r.path().join("new.txt"), "untracked\n").unwrap();
        let with_new = working_tree(r.path()).unwrap();
        assert_ne!(with_new, clean);
        assert_eq!(
            run(r.path(), &["status", "--porcelain"]),
            "?? new.txt",
            "nothing was staged"
        );
        // Committing exactly that state gives the same tree.
        run(r.path(), &["add", "-A"]);
        run(r.path(), &["commit", "-q", "-m", "second"]);
        let head = commit_id(r.path(), "HEAD").unwrap();
        assert_eq!(commit_tree(r.path(), &head).unwrap(), with_new);
    }

    #[test]
    fn a_committed_proof_folder_never_changes_the_tree() {
        let r = repo();
        let sub = r.path().join("src");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("lib.rs"), "fn main() {}\n").unwrap();
        let tested = working_tree(&sub).unwrap();
        run(r.path(), &["add", "-A"]);
        run(r.path(), &["commit", "-q", "-m", "code"]);
        std::fs::create_dir_all(r.path().join(PROOF_DIR)).unwrap();
        std::fs::write(r.path().join(PROOF_DIR).join("x.intoto.jsonl"), "{}\n").unwrap();
        assert_eq!(working_tree(r.path()).unwrap(), tested);
        run(r.path(), &["add", "-A"]);
        run(r.path(), &["commit", "-q", "-m", "proof"]);
        let head = commit_id(r.path(), "HEAD").unwrap();
        assert_ne!(run(r.path(), &["rev-parse", "HEAD^{tree}"]), tested);
        assert_eq!(commit_tree(r.path(), &head).unwrap(), tested);
    }

    #[test]
    fn a_note_round_trips_and_a_dash_revision_is_refused() {
        let r = repo();
        let head = commit_id(r.path(), "HEAD").unwrap();
        assert_eq!(read_note(r.path(), &head), None);
        write_note(r.path(), &head, "{\"a\":1}").unwrap();
        assert_eq!(read_note(r.path(), &head).as_deref(), Some("{\"a\":1}"));
        assert!(commit_id(r.path(), "--all").is_err());
        assert!(commit_id(r.path(), "no-such-ref").is_err());
    }
}
