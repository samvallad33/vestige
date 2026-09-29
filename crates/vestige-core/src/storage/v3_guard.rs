//! v3 SQLite guard.
//!
//! 4.0 never opens a v3 SQLite store read-write. Every read-write storage
//! entry point calls [`detect_v3`] first: a file that carries the SQLite
//! magic — any schema version, since 4.0 itself writes no SQLite — is opened
//! only through a read-only `immutable=1` URI to read its schema version,
//! and refused with [`StorageError::V3StoreNeedsMigration`].
//!
//! The magic sniff and header fallback parse raw bytes with std only, so the
//! guard compiles (and still refuses) in builds without the legacy engine.
//! When `legacy-sqlite` is compiled in, the schema version is read from the
//! store's own `schema_version` table through the mandated read-only URI.

use std::path::{Path, PathBuf};

use crate::storage::types::{Result, StorageError};

/// First 16 bytes of every SQLite 3 database file.
pub const SQLITE_MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// Actionable next step embedded in every refusal.
pub const MIGRATION_HINT: &str =
    "run: vestige migrate-to-strata --from <path>  (the original is never modified)";

/// A detected v3 SQLite store. `schema_version` is the highest
/// `schema_version.version` row (0 when it could not be read).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct V3Info {
    /// Path of the SQLite file. Never opened read-write.
    pub path: PathBuf,
    /// Highest schema version recorded in the store; 0 = unknown.
    pub schema_version: u32,
}

/// Detect a v3 SQLite file at `path`.
///
/// Reads at most the 100-byte database header (std file read, no write
/// handle of any kind). Returns `Ok(None)` for missing files, empty files,
/// and any file that does not carry the SQLite magic.
pub fn detect_v3(path: &Path) -> Result<Option<V3Info>> {
    let mut header = [0u8; 100];
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        // A missing file is not a v3 store; the real open path reports it.
        Err(_) => return Ok(None),
    };
    let mut filled = 0usize;
    while filled < header.len() {
        match std::io::Read::read(&mut file, &mut header[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return Ok(None),
        }
    }
    if filled < SQLITE_MAGIC.len() || &header[..16] != SQLITE_MAGIC {
        return Ok(None);
    }
    let schema_version = read_schema_version(path, &header);
    Ok(Some(V3Info {
        path: path.to_path_buf(),
        schema_version,
    }))
}

/// Build the canonical refusal for a detected v3 store.
pub fn refuse_v3(info: &V3Info) -> StorageError {
    StorageError::V3StoreNeedsMigration {
        path: info.path.display().to_string(),
        schema_version: info.schema_version,
        hint: MIGRATION_HINT.to_string(),
    }
}

/// Refuse a v3 SQLite file at `path`; pass anything else through.
pub fn ensure_not_v3(path: &Path) -> Result<()> {
    match detect_v3(path)? {
        Some(info) => Err(refuse_v3(&info)),
        None => Ok(()),
    }
}

/// True when the read-write v3 refusal is compiled in.
///
/// Workspace tests unify `v3-engine`, which compiles the constructor guard
/// out. The MCP pre-check must use the same switch or a racing start treats
/// the in-progress file (schema version 0) as a migration refusal.
pub fn v3_rw_guard_armed() -> bool {
    cfg!(all(feature = "legacy-sqlite", not(feature = "v3-engine")))
}

#[cfg_attr(not(feature = "legacy-sqlite"), allow(unused_variables))]
fn read_schema_version(path: &Path, header: &[u8; 100]) -> u32 {
    #[cfg(feature = "legacy-sqlite")]
    if let Some(v) = query_schema_version_table(path) {
        return v;
    }
    // Header fallback without rusqlite: bytes 60..64 hold the database
    // `user_version` (big-endian). Vestige tracks its migration version in
    // the `schema_version` table, so this is best-effort provenance only —
    // refusal does not depend on it.
    let _ = header;
    0
}

/// Highest `schema_version.version` row, read strictly through the
/// mandated read-only immutable URI. The connection physically cannot
/// write, journal, or create `-wal`/`-shm` siblings.
#[cfg(feature = "legacy-sqlite")]
fn query_schema_version_table(path: &Path) -> Option<u32> {
    let uri = format!("file:{}?mode=ro&immutable=1", uri_encode_path(path));
    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
        | rusqlite::OpenFlags::SQLITE_OPEN_URI
        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(uri, flags).ok()?;
    conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
        row.get::<_, u32>(0)
    })
    .ok()
}

/// Percent-encode a path for use inside a `file:` URI. SQLite treats `?`
/// and `#` as URI structure and `%` as an escape; everything outside the
/// unreserved set is encoded so spaces and shell-hostile names survive.
#[cfg(feature = "legacy-sqlite")]
fn uri_encode_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut out = String::with_capacity(text.len());
    for byte in text.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'.' | b'_' | b'-' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_none_for_missing_and_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detect_v3(&dir.path().join("absent.db")).unwrap().is_none());
        let empty = dir.path().join("empty.db");
        std::fs::write(&empty, b"").unwrap();
        assert!(detect_v3(&empty).unwrap().is_none());
    }

    #[test]
    fn detect_returns_none_for_non_sqlite_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let junk = dir.path().join("junk.db");
        std::fs::write(&junk, b"definitely not a database header at all").unwrap();
        assert!(detect_v3(&junk).unwrap().is_none());
    }

    #[test]
    fn refusal_error_carries_path_schema_and_hint() {
        let info = V3Info {
            path: PathBuf::from("/tmp/vestige.db"),
            schema_version: 38,
        };
        match refuse_v3(&info) {
            StorageError::V3StoreNeedsMigration {
                path,
                schema_version,
                hint,
            } => {
                assert_eq!(path, "/tmp/vestige.db");
                assert_eq!(schema_version, 38);
                assert!(hint.contains("migrate-to-strata"));
            }
            other => panic!("wrong error: {other:?}"),
        }
    }
}
