//! Copy-only store open.
//!
//! The copy skips `strata.lock` and never seals or rewrites the source.
//! `StrataStore::backup_to` is not used: that path mutates the source.

use std::fs;
use std::io;
use std::path::Path;

use strata_store::{StoreError, StrataStore};

/// A store opened on a private copy of a log directory.
pub struct CopiedStore {
    _dir: tempfile::TempDir,
    /// Fold of the copy.
    pub store: StrataStore,
}

/// Recursively copy `src` onto `dst`, omitting every `strata.lock`.
pub fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    let mut entries: Vec<_> = fs::read_dir(src)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if entry.file_name() == "strata.lock" {
            continue;
        }
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Open a copy of `src_dir`. The source directory is not opened and not locked.
pub fn open_copy(src_dir: &Path) -> Result<CopiedStore, StoreError> {
    let dir = tempfile::tempdir().map_err(StoreError::Io)?;
    copy_tree(src_dir, dir.path()).map_err(StoreError::Io)?;
    let store = StrataStore::open(dir.path())?;
    Ok(CopiedStore { _dir: dir, store })
}
