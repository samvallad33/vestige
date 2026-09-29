//! Source loading: turn `<src>` into a migration snapshot — strictly
//! read-only (H10).
//!
//! Path A (preferred): `<src>` is a portable-archive JSON file produced by
//! `vestige portable-export`. Decode it directly.
//!
//! Path B (direct): `<src>` is a SQLite database file (or a data directory
//! containing `vestige.db`). It is opened ONLY through the
//! `file:<p>?mode=ro&immutable=1` URI with `SQLITE_OPEN_READ_ONLY`: SQLite
//! physically cannot write, journal, create `-wal`/`-shm` siblings, run
//! migrations, or chmod the file. The source's BLAKE3 is taken before the
//! first SQL read and re-checked after the migration; any difference is a
//! hard failure.
//!
//! A non-empty `-wal` sibling means the last commits may still sit in the
//! WAL: the migration refuses unless `--accept-wal-snapshot` is given, in
//! which case the db + sidecars are copied to a scratch directory and the
//! COPY is read (the original stays untouched).

use std::path::{Path, PathBuf};

use chrono::DateTime;
use rusqlite::OpenFlags;
use vestige_core::storage::PortableArchive;
use vestige_core::storage::PortableTable;
pub use vestige_core::storage::PortableValue;

/// Portable-archive format identifier (mirrors the ungated constant the v3
/// engine writes; declared locally so this crate needs no sqlite features).
pub const PORTABLE_ARCHIVE_FORMAT: &str = "vestige.portable.v1";

use crate::MigrationError;

/// SQLite file magic (first 16 bytes of every SQLite 3 database).
pub const SQLITE_MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// Tables mirrored into the snapshot, in canonical order. Mirrors
/// `PORTABLE_TABLES` in `vestige-core/src/storage/sqlite/mod.rs` — the v3
/// row-decoding contract — without opening a vestige-core storage handle.
const PORTABLE_TABLES: &[&str] = &[
    "knowledge_nodes",
    // node_embeddings is intentionally NOT selected: its vector blobs are
    // dropped (H1) and must never enter the snapshot. Only COUNT(*) is read
    // (dropped_vectors). (Audit finding: SELECT * hex-encoded blobs into
    // the snapshot while the CLI claimed vector values were never read.)
    "fsrs_cards",
    "memory_states",
    "memory_connections",
    "memory_access_log",
    "state_transitions",
    "intentions",
    "insights",
    "sessions",
    "fsrs_config",
    "consolidation_history",
    "dream_history",
    "retention_snapshots",
    "sync_tombstones",
    "deletion_tombstones",
    "composition_events",
    "composition_members",
    "composition_outcomes",
];

/// Everything the migration needs to know about one source store.
pub struct SourceSnapshot {
    /// Portable-archive view of the mapped tables (same shape the
    /// `vestige portable-export` path produces).
    pub archive: PortableArchive,
    /// Highest `schema_version.version` row.
    pub schema_version: u32,
    /// Last verified `receipt_envelopes` entry digest (`None` = no
    /// envelopes / legacy-unsigned store). The chain is fully verified
    /// before this is produced.
    pub envelope_head: Option<String>,
    /// `node_embeddings` rows (their vector values are never read; H1).
    pub dropped_vectors: u64,
    /// V40 `walk_receipts` rows, ordered by (created_at, receipt_id).
    pub walk_receipts: Vec<WalkReceiptRow>,
    /// Row count per snapshot table (sorted by table name).
    pub table_counts: Vec<(String, u64)>,
    /// EVERY user table with rows, from sqlite_master (audit finding:
    /// skipped_tables must be driven by the schema, not by a hand list, so
    /// the receipt never claims a fuller copy than it is).
    pub all_nonempty_tables: Vec<(String, u64)>,
}

/// One V40 `walk_receipts` row; becomes a reference node in the log.
pub struct WalkReceiptRow {
    pub receipt_id: String,
    pub digest: String,
    pub canonical_json: String,
    pub engine_version: String,
    pub created_ms: i64,
}

/// Resolved source files (after any `--accept-wal-snapshot` copy).
pub struct SourceFiles {
    /// The database actually read (the original, or the snapshot copy).
    pub db: PathBuf,
    /// `-wal` sidecar next to `db`, when present.
    pub wal: Option<PathBuf>,
    /// `-shm` sidecar next to `db`, when present.
    pub shm: Option<PathBuf>,
}

impl SourceFiles {
    /// Canonical BLAKE3 over db → wal → shm (only existing files).
    pub fn blake3_hex(&self) -> Result<String, MigrationError> {
        let mut hasher = blake3::Hasher::new();
        let candidates = [
            ("db", Some(&self.db)),
            ("wal", self.wal.as_ref()),
            ("shm", self.shm.as_ref()),
        ];
        for (tag, path) in candidates {
            let Some(path) = path else { continue };
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            hasher.update(format!("{tag}:{len}:", len = bytes.len()).as_bytes());
            hasher.update(&bytes);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

/// Resolve `<src>` to concrete files, enforce the WAL policy, and apply the
/// snapshot copy when `accept_wal` is set. `scratch` is only used when a
/// snapshot copy is made.
pub fn prepare_source(
    source: &Path,
    accept_wal: bool,
    scratch: &Path,
) -> Result<(SourceFiles, bool), MigrationError> {
    let db = resolve_source(source)?;
    if !is_sqlite_file(&db)? {
        // Portable-archive JSON: no SQLite files involved.
        return Ok((
            SourceFiles {
                db,
                wal: None,
                shm: None,
            },
            false,
        ));
    }

    let wal = sidecar(&db, "-wal");
    let shm = sidecar(&db, "-shm");

    let wal_nonempty = wal
        .as_ref()
        .is_some_and(|p| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false));
    if wal_nonempty && !accept_wal {
        return Err(MigrationError::WalPresent {
            path: wal.as_ref().unwrap().display().to_string(),
        });
    }

    if wal_nonempty {
        // Snapshot: copy db + wal + shm into scratch and read the copy so
        // the immutable read sees a consistent image. The originals are
        // only ever read (std::fs::copy opens for reading).
        std::fs::create_dir_all(scratch)?;
        let copy =
            |src: &Option<PathBuf>, suffix: &str| -> Result<Option<PathBuf>, MigrationError> {
                Ok(match src {
                    Some(p) if p.exists() => {
                        let dst = scratch.join(format!("snapshot{suffix}"));
                        std::fs::copy(p, &dst)?;
                        Some(dst)
                    }
                    _ => None,
                })
            };
        // Sidecar names MUST line up with the snapshot db name or SQLite
        // will not associate the copied -wal with it (audit finding: the
        // first cut copied `snapshot-wal`, so the checkpoint saw nothing).
        let db_copy = scratch.join("snapshot.db");
        let wal_copy = scratch.join("snapshot.db-wal");
        let shm_copy = scratch.join("snapshot.db-shm");
        std::fs::copy(&db, &db_copy)?;
        if let Some(wal) = &wal {
            std::fs::copy(wal, &wal_copy)?;
        }
        if let Some(shm) = &shm {
            std::fs::copy(shm, &shm_copy)?;
        }
        {
            // immutable=1 does NOT see WAL-resident commits (audit finding:
            // a row living only in the -wal was silently dropped). The COPY
            // is checkpointed so its main db is complete; the ORIGINAL is
            // never opened read-write.
            let conn = rusqlite::Connection::open(&db_copy)
                .map_err(|e| MigrationError::Source(format!("checkpoint snapshot copy: {e}")))?;
            let _ = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                row.get::<_, i64>(0)
            });
        }
        let files = SourceFiles {
            db: db_copy,
            wal: wal_copy.exists().then_some(wal_copy),
            shm: shm_copy.exists().then_some(shm_copy),
        };
        Ok((files, true))
    } else {
        Ok((SourceFiles { db, wal, shm }, false))
    }
}

/// Load `<src>` as a snapshot, auto-detecting the path shape. SQLite
/// sources are opened strictly read-only.
pub fn load_snapshot(files: &SourceFiles) -> Result<SourceSnapshot, MigrationError> {
    if !is_sqlite_file(&files.db)? {
        // Path A: portable archive JSON.
        let bytes = std::fs::read(&files.db)?;
        let archive: PortableArchive = parse_archive_json(&bytes)?;
        if archive.archive_format != PORTABLE_ARCHIVE_FORMAT {
            return Err(MigrationError::UnsupportedSource(format!(
                "archive format {:?} is not {:?}",
                archive.archive_format, PORTABLE_ARCHIVE_FORMAT
            )));
        }
        let table_counts = archive
            .tables
            .iter()
            .filter(|t| !t.rows.is_empty())
            .map(|t| (t.name.clone(), t.rows.len() as u64))
            .collect::<Vec<_>>();
        return Ok(SourceSnapshot {
            schema_version: archive.schema_version,
            archive,
            envelope_head: None,
            dropped_vectors: 0,
            walk_receipts: Vec::new(),
            table_counts: table_counts.clone(),
            all_nonempty_tables: table_counts,
        });
    }

    let conn = open_readonly(&files.db)?;
    let schema_version = conn
        .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
            row.get::<_, u32>(0)
        })
        .unwrap_or(0);

    let envelope_head = verify_envelope_chain(&conn)?;

    let dropped_vectors = conn
        .query_row("SELECT COUNT(*) FROM node_embeddings", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n.max(0) as u64)
        .unwrap_or(0);

    let mut walk_receipts = Vec::new();
    {
        // Real v3.1.1 stores (schema <= 39) have no walk_receipts table; it
        // only exists from V40. Missing table = empty vec, same as read_table.
        let has_walk_receipts: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='walk_receipts'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|n| n > 0)
            .unwrap_or(false);
        if !has_walk_receipts {
            return finish_snapshot(
                &conn,
                schema_version,
                envelope_head,
                dropped_vectors,
                walk_receipts,
            );
        }
        let mut stmt = conn
            .prepare(
                "SELECT receipt_id, digest, canonical_json, engine_version, created_at
                 FROM walk_receipts ORDER BY created_at, receipt_id",
            )
            .map_err(|e| MigrationError::Source(format!("read walk_receipts: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(WalkReceiptRow {
                    receipt_id: row.get(0)?,
                    digest: row.get(1)?,
                    canonical_json: row.get(2)?,
                    engine_version: row.get(3)?,
                    created_ms: crate::source::timestamp_ms(&row.get::<_, String>(4)?).unwrap_or(0),
                })
            })
            .map_err(|e| MigrationError::Source(format!("read walk_receipts: {e}")))?;
        for row in rows {
            walk_receipts
                .push(row.map_err(|e| MigrationError::Corrupt(format!("walk_receipts row: {e}")))?);
        }
    }

    finish_snapshot(
        &conn,
        schema_version,
        envelope_head,
        dropped_vectors,
        walk_receipts,
    )
}

/// Every user table with at least one row, per sqlite_master.
fn all_nonempty_tables(conn: &rusqlite::Connection) -> Vec<(String, u64)> {
    let Ok(mut stmt) = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table'
         AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'knowledge_fts%'
         ORDER BY name",
    ) else {
        return Vec::new();
    };
    let names: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map(|rows| rows.filter_map(Result::ok).collect())
        .unwrap_or_default();
    names
        .into_iter()
        .filter_map(|name| {
            let quoted = format!("\"{name}\"");
            conn.query_row(&format!("SELECT COUNT(*) FROM {quoted}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .ok()
            .filter(|n| *n > 0)
            .map(|n| (name, n as u64))
        })
        .collect()
}

/// Build the SourceSnapshot from the already-read scalars: the portable
/// tables (v3 contract list), per-table counts, and the walk receipts.
fn finish_snapshot(
    conn: &rusqlite::Connection,
    schema_version: u32,
    envelope_head: Option<String>,
    dropped_vectors: u64,
    walk_receipts: Vec<WalkReceiptRow>,
) -> Result<SourceSnapshot, MigrationError> {
    let all_nonempty = all_nonempty_tables(conn);
    let mut tables = Vec::new();
    let mut table_counts = Vec::new();
    for table_name in PORTABLE_TABLES {
        let Some(table) = read_table(conn, table_name)? else {
            continue;
        };
        table_counts.push((table.name.clone(), table.rows.len() as u64));
        tables.push(table);
    }
    // Vectors are dropped (H1) and the blob column is never selected, but
    // the receipt must still name them.
    if dropped_vectors > 0 {
        table_counts.push(("node_embeddings".to_string(), dropped_vectors));
        table_counts.sort();
    }

    Ok(SourceSnapshot {
        schema_version,
        archive: PortableArchive {
            archive_format: PORTABLE_ARCHIVE_FORMAT.to_string(),
            // Fixed provenance string: the source store carries no usable
            // version column, and this field must be deterministic across
            // runs (identical sources → identical logs).
            vestige_version: "v3-migration-source".to_string(),
            schema_version,
            exported_at: DateTime::from_timestamp(0, 0).unwrap_or_default(),
            mode: "exact".to_string(),
            tables,
        },
        envelope_head,
        dropped_vectors,
        walk_receipts,
        table_counts,
        all_nonempty_tables: all_nonempty,
    })
}

/// Open a SQLite file strictly read-only through the immutable URI. The
/// connection cannot write, journal, or create sidecars — SQLite enforces
/// it at the VFS level, not by convention.
pub fn open_readonly(path: &Path) -> Result<rusqlite::Connection, MigrationError> {
    let uri = format!("file:{}?mode=ro&immutable=1", uri_encode_path(path));
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    rusqlite::Connection::open_with_flags(uri, flags)
        .map_err(|e| MigrationError::Source(format!("open read-only: {e}")))
}

/// Verify the `receipt_envelopes` hash chain. Any break aborts the
/// migration with a report. Returns the head (`entry_digest` of the last
/// row in `(chain_id, sequence)` order), or `None` for a store without
/// envelopes (the explicit legacy-unsigned state).
pub fn verify_envelope_chain(
    conn: &rusqlite::Connection,
) -> Result<Option<String>, MigrationError> {
    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='receipt_envelopes'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|n| n > 0)
        .unwrap_or(false);
    if !exists {
        return Ok(None);
    }

    let mut stmt = conn
        .prepare(
            "SELECT chain_id, sequence, previous_entry_digest, payload_type, envelope_json,
                    payload_digest, entry_digest, signing_key_id
             FROM receipt_envelopes
             ORDER BY chain_id, sequence",
        )
        .map_err(|e| MigrationError::Source(format!("read receipt_envelopes: {e}")))?;

    let mut rows = stmt
        .query([])
        .map_err(|e| MigrationError::Source(format!("read receipt_envelopes: {e}")))?;

    let mut head: Option<String> = None;
    let mut current_chain: Option<String> = None;
    let mut expected_seq: i64 = 0;
    let mut prev_digest = String::new();

    while let Some(row) = rows
        .next()
        .map_err(|e| MigrationError::Source(format!("read receipt_envelopes: {e}")))?
    {
        let cell = |idx: usize| -> Result<String, MigrationError> {
            row.get::<_, String>(idx).map_err(|e| {
                MigrationError::Corrupt(format!("receipt_envelopes column {idx}: {e}"))
            })
        };
        let chain_id = cell(0)?;
        let sequence: i64 = row
            .get::<_, i64>(1)
            .map_err(|e| MigrationError::Corrupt(format!("receipt_envelopes column 1: {e}")))?;
        let previous: Option<String> = row
            .get::<_, Option<String>>(2)
            .map_err(|e| MigrationError::Corrupt(format!("receipt_envelopes column 2: {e}")))?;
        let payload_type = cell(3)?;
        let envelope_json = cell(4)?;
        let payload_digest_col = cell(5)?;
        let entry_digest_col = cell(6)?;
        let _signing_key_id = cell(7)?;

        if current_chain.as_deref() != Some(chain_id.as_str()) {
            current_chain = Some(chain_id.clone());
            expected_seq = 0;
            prev_digest.clear();
        }
        if sequence != expected_seq {
            return Err(MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id}: expected sequence {expected_seq}, found {sequence}"
            )));
        }
        match (sequence, previous.as_deref()) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(MigrationError::BrokenEnvelopeChain(format!(
                    "chain {chain_id}: first entry carries a previous digest"
                )));
            }
            (_, None) => {
                return Err(MigrationError::BrokenEnvelopeChain(format!(
                    "chain {chain_id} seq {sequence}: previous digest missing"
                )));
            }
            (_, Some(prev)) if prev == prev_digest => {}
            (_, Some(prev)) => {
                return Err(MigrationError::BrokenEnvelopeChain(format!(
                    "chain {chain_id} seq {sequence}: previous digest {prev} does not link to {prev_digest}"
                )));
            }
        }

        // Recompute both digests from the envelope's own DSSE payload.
        let env: DsseWire = serde_json::from_str(&envelope_json).map_err(|e| {
            MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id} seq {sequence}: envelope_json is not DSSE: {e}"
            ))
        })?;
        let payload = decode_b64(&env.payload).map_err(|e| {
            MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id} seq {sequence}: payload base64: {e}"
            ))
        })?;
        let (key_id, sig) = match env.signatures.first() {
            Some(s) => (s.keyid.clone().unwrap_or_default(), s.sig.clone()),
            None => (String::new(), String::new()),
        };
        let sig_bytes = decode_b64(&sig).map_err(|e| {
            MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id} seq {sequence}: signature base64: {e}"
            ))
        })?;

        if vestige_core::storage::receipt_attestation::payload_digest(&payload)
            != payload_digest_col
        {
            return Err(MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id} seq {sequence}: payload_digest mismatch"
            )));
        }
        let recomputed = vestige_core::storage::receipt_attestation::entry_digest(
            &payload_type,
            &payload,
            &key_id,
            &sig_bytes,
        );
        if recomputed != entry_digest_col {
            return Err(MigrationError::BrokenEnvelopeChain(format!(
                "chain {chain_id} seq {sequence}: entry_digest mismatch"
            )));
        }

        prev_digest = entry_digest_col.clone();
        expected_seq = sequence + 1;
        head = Some(entry_digest_col);
    }
    Ok(head)
}

fn decode_b64(raw: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(raw.trim())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(raw.trim()))
}

/// Minimal DSSE envelope shape for digest recomputation.
#[derive(serde::Deserialize)]
struct DsseWire {
    #[serde(rename = "payloadType")]
    #[allow(dead_code)]
    payload_type: String,
    payload: String,
    signatures: Vec<DsseWireSig>,
}

#[derive(serde::Deserialize)]
struct DsseWireSig {
    #[serde(default)]
    keyid: Option<String>,
    sig: String,
}

/// Read one table exactly the way `export_portable_archive` does:
/// `SELECT * ORDER BY rowid`, columns by name.
fn read_table(
    conn: &rusqlite::Connection,
    table_name: &str,
) -> Result<Option<PortableTable>, MigrationError> {
    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
            [table_name],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| MigrationError::Source(format!("probe {table_name}: {e}")))?
        > 0;
    if !exists {
        return Ok(None);
    }

    let quoted = format!("\"{table_name}\"");
    let mut stmt = conn
        .prepare(&format!("SELECT * FROM {quoted} ORDER BY rowid"))
        .map_err(|e| MigrationError::Source(format!("read {table_name}: {e}")))?;
    let columns: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let column_count = columns.len();

    let rows = stmt
        .query_map([], |row| {
            let mut values = Vec::with_capacity(column_count);
            for idx in 0..column_count {
                values.push(portable_value(row.get_ref(idx)?));
            }
            Ok(values)
        })
        .map_err(|e| MigrationError::Source(format!("read {table_name}: {e}")))?;

    let mut portable_rows = Vec::new();
    for row in rows {
        portable_rows
            .push(row.map_err(|e| MigrationError::Corrupt(format!("{table_name} row: {e}")))?);
    }
    Ok(Some(PortableTable {
        name: (*table_name).to_string(),
        columns,
        rows: portable_rows,
    }))
}

/// rusqlite value → snapshot value (blobs hex-encoded, matching the
/// portable-archive convention).
fn portable_value(value: rusqlite::types::ValueRef<'_>) -> PortableValue {
    use rusqlite::types::ValueRef;
    match value {
        ValueRef::Null => PortableValue::Null,
        ValueRef::Integer(v) => PortableValue::Integer(v),
        ValueRef::Real(v) => PortableValue::Real(v),
        ValueRef::Text(t) => PortableValue::Text(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => PortableValue::Blob(hex_encode(b)),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn parse_archive_json(bytes: &[u8]) -> Result<PortableArchive, MigrationError> {
    serde_json::from_slice(bytes)
        .map_err(|e| MigrationError::UnsupportedSource(format!("not a portable archive: {e}")))
}

/// Resolve a directory source to its `vestige.db`; pass files through.
fn resolve_source(source: &Path) -> Result<PathBuf, MigrationError> {
    if source.is_dir() {
        let db = source.join("vestige.db");
        if db.is_file() {
            Ok(db)
        } else {
            Err(MigrationError::UnsupportedSource(format!(
                "directory {} has no vestige.db",
                source.display()
            )))
        }
    } else if source.is_file() {
        Ok(source.to_path_buf())
    } else {
        Err(MigrationError::SourceNotFound(source.display().to_string()))
    }
}

fn sidecar(db: &Path, suffix: &str) -> Option<PathBuf> {
    let mut name = db.file_name()?.to_os_string();
    name.push(suffix);
    let path = db.with_file_name(name);
    path.exists().then_some(path)
}

/// Sniff the SQLite file magic without loading the file.
pub fn is_sqlite_file(path: &Path) -> Result<bool, MigrationError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 16];
    let read = file.read(&mut magic)?;
    Ok(read == 16 && &magic == SQLITE_MAGIC)
}

/// Percent-encode a path for use inside a `file:` URI.
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

/// Find a table by exact name.
pub fn table<'a>(archive: &'a PortableArchive, name: &str) -> Option<&'a PortableTable> {
    archive.tables.iter().find(|t| t.name == name)
}

/// Value accessor for one row of a [`PortableTable`].
pub struct Row<'a> {
    table: &'a PortableTable,
    index: usize,
}

impl<'a> Row<'a> {
    /// Wrap row `index` of `table`.
    pub fn new(table: &'a PortableTable, index: usize) -> Self {
        Self { table, index }
    }

    /// The table's column names, in order.
    pub fn columns(&self) -> &'a [String] {
        &self.table.columns
    }

    /// Raw value of column `name`.
    pub fn get(&self, name: &str) -> Result<&'a PortableValue, MigrationError> {
        let column = self
            .table
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| {
                MigrationError::Corrupt(format!("table {} has no column {}", self.table.name, name))
            })?;
        self.table
            .rows
            .get(self.index)
            .and_then(|row| row.get(column))
            .ok_or_else(|| {
                MigrationError::Corrupt(format!(
                    "table {} row {} is missing column {}",
                    self.table.name, self.index, name
                ))
            })
    }

    /// TEXT column.
    pub fn text(&self, name: &str) -> Result<&'a str, MigrationError> {
        match self.get(name)? {
            PortableValue::Text(s) => Ok(s.as_str()),
            other => Err(type_error(&self.table.name, name, "text", other)),
        }
    }

    /// INTEGER column.
    pub fn integer(&self, name: &str) -> Result<i64, MigrationError> {
        match self.get(name)? {
            PortableValue::Integer(v) => Ok(*v),
            other => Err(type_error(&self.table.name, name, "integer", other)),
        }
    }

    /// REAL column.
    pub fn real(&self, name: &str) -> Result<f64, MigrationError> {
        match self.get(name)? {
            PortableValue::Real(v) => Ok(*v),
            PortableValue::Integer(v) => Ok(*v as f64),
            other => Err(type_error(&self.table.name, name, "real", other)),
        }
    }

    /// Nullable TEXT column.
    pub fn opt_text(&self, name: &str) -> Result<Option<&'a str>, MigrationError> {
        match self.get(name)? {
            PortableValue::Null => Ok(None),
            PortableValue::Text(s) => Ok(Some(s.as_str())),
            other => Err(type_error(&self.table.name, name, "text?", other)),
        }
    }

    /// INTEGER column with SQL NULL read as the given default.
    pub fn integer_or(&self, name: &str, default: i64) -> Result<i64, MigrationError> {
        match self.get(name)? {
            PortableValue::Null => Ok(default),
            PortableValue::Integer(v) => Ok(*v),
            other => Err(type_error(&self.table.name, name, "integer?", other)),
        }
    }

    /// REAL column with SQL NULL read as the given default.
    pub fn real_or(&self, name: &str, default: f64) -> Result<f64, MigrationError> {
        match self.get(name)? {
            PortableValue::Null => Ok(default),
            PortableValue::Real(v) => Ok(*v),
            PortableValue::Integer(v) => Ok(*v as f64),
            other => Err(type_error(&self.table.name, name, "real?", other)),
        }
    }
}

fn type_error(table: &str, column: &str, wanted: &str, got: &PortableValue) -> MigrationError {
    MigrationError::Corrupt(format!(
        "table {table} column {column}: expected {wanted}, found {got:?}"
    ))
}

/// Parse a legacy RFC3339 timestamp into Unix epoch milliseconds.
///
/// Vestige writes `DateTime<Utc>::to_rfc3339()` everywhere, but be liberal
/// about the exact offset spelling (`Z` vs `+00:00` both parse with
/// `parse_from_rfc3339`).
pub fn timestamp_ms(raw: &str) -> Result<i64, MigrationError> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.timestamp_millis())
        .map_err(|e| MigrationError::Corrupt(format!("timestamp {raw:?} is not RFC3339: {e}")))
}

/// Parse a legacy JSON tag array (`'[]'`, `'["a","b"]'`). Tolerant: a NULL,
/// non-array, or malformed value migrates as no tags rather than failing the
/// whole migration — tags are metadata, not structure.
pub fn parse_tags(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else { return Vec::new() };
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}
