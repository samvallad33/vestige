//! # strata-migrate — one-shot Vestige SQLite → STRATA migration
//!
//! `vestige migrate-to-strata --from <src> [--to <dst>]` reads a Vestige
//! SQLite store STRICTLY READ-ONLY and replays it into an append-only
//! STRATA log through a signed MIGRATION_RECEIPT. It is a MIGRATION, not a
//! sync: run it once, keep the SQLite file (byte-identical, never opened
//! read-write) as the pre-migration record, and write new memories to
//! STRATA.
//!
//! ## What lands in the log
//!
//! | SQLite source                    | STRATA record                                  |
//! |----------------------------------|------------------------------------------------|
//! | `GENESIS` / `PARAMS v4-migrate/1`| provenance + parameter frames on a fresh log   |
//! | `knowledge_nodes` rows           | `NODE` frames (kernel_id v1 + legacy UUID)      |
//! | V40 `walk_receipts` rows         | reference `NODE` frames tagged migrated_from_v4 |
//! | `knowledge_nodes.superseded_by`  | `SUPERSESSION` frames                           |
//! | `memory_connections` rows        | `EDGE` frames (declared 8-type vocabulary;      |
//! |                                  | inferred types become `legacy_inferred`)        |
//! | `fsrs_cards` rows                | `FSRS_REVIEW` frames (kernel `ReviewEvent`       |
//! |                                  | plus `borsh(Option<i64>)` `reviewed_at_ms`)      |
//! | `sync_tombstones` / `deletion_tombstones` | `TOMBSTONE` frames                     |
//! | everything else with rows        | counted in `MigrationReport::skipped_tables`    |
//! | final frame                      | signed `MIGRATION_RECEIPT` (kind 46)            |
//!
//! The run re-hashes the source after the replay and refuses to seal if a
//! single byte changed (the reader is read-only at the SQLite VFS level,
//! so this is belt-and-suspenders). `strata_kernel::verify_with_head` plus
//! `StrataLog::verify_tail` must both pass before `verify_passed` is
//! reported true.
//!
//! ## FSRS fold semantics (read before relying on it)
//!
//! SQLite stores only FINAL FSRS state (`reps`, `lapses`, floats) — the
//! review history that produced it is gone. Migration therefore synthesizes
//! a deterministic event series per card: `reps - lapses` rating-3 (good)
//! events followed by `lapses` rating-1 (again) events. The kernel fold
//! reproduces `review_count == reps` and `lapse_count == lapses` EXACTLY;
//! stability/difficulty are recomputed by the deterministic fold and become
//! the new truth (the legacy floats were not reproducible from any log).
//!
//! ## Event seqs
//!
//! `ReviewEvent::event_seq` is the frame seq the event lands at (the kernel
//! requires `event.seq() == record seq`), predicted from the log head before
//! the batch append and asserted against the returned acks afterward.
//!
//! Every `FSRS_REVIEW` ends with `borsh(Option<i64>)` `reviewed_at_ms`.
//! Intermediate synthetic reviews are `None`. The last one is the card's
//! `fsrs_cards.last_review` (unix ms), or `None` when that column is NULL,
//! empty, or missing. The kernel `ReviewEvent` prefix is unchanged, so
//! checkpoint hashes stay valid. Retrievability uses this clock instead of
//! the import-time frame seq.
//!
//! ## Determinism
//!
//! Every timestamp in the log comes from the source rows (a replay clock:
//! the migration never reads the wall clock into hashed state), the log is
//! opened seeded from the source BLAKE3 unless a seed is pinned, and all
//! hashed collections are ordered Vecs. Two runs over one source produce
//! byte-identical segments.

pub mod records;
pub mod snapshot;
// v3 comparison against a source SQLite store. Absent unless `sqlite-reader`.
#[cfg(feature = "sqlite-reader")]
pub mod source;

#[cfg(feature = "sqlite-reader")]
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
#[cfg(feature = "sqlite-reader")]
use std::time::Instant;

#[cfg(feature = "sqlite-reader")]
use strata::StrataLog;
#[cfg(feature = "sqlite-reader")]
use strata_kernel::checkpoint::{checkpoint_hash, Checkpoint};
#[cfg(any(feature = "sqlite-reader", test))]
use strata_kernel::event::ReviewEvent;
#[cfg(any(feature = "sqlite-reader", test))]
use strata_kernel::fsrs::ALGO_V1;
#[cfg(any(feature = "sqlite-reader", test))]
use strata_kernel::kernel::Kernel;
#[cfg(feature = "sqlite-reader")]
use strata_kernel::verify::verify_with_head;
#[cfg(feature = "sqlite-reader")]
use vestige_core::storage::PortableArchive;
#[cfg(feature = "sqlite-reader")]
use vestige_core::storage::PortableValue;

pub use records::{
    EdgeRecord, GenesisRecord, MigrationReceipt, NodeRecord, ParamsRecord, ReceiptBody, SourceKey,
    SupersessionRecord, TombstoneRecord, KIND_MIGRATION_RECEIPT, RECEIPT_SIGNING_KEY_ID,
    RECORD_VERSION,
};
pub use snapshot::{read_snapshot, Snapshot};

/// Parameter set implemented by this migrator. Written as the `PARAMS`
/// frame on a fresh log.
pub const PARAMS_ID: &str = "v4-migrate/1";

/// Frames per `append_batch` call: bounds peak memory on huge stores while
/// staying far above the log's own 64-frame group-commit cap.
#[cfg(feature = "sqlite-reader")]
const BATCH_FRAMES: usize = 1024;

/// Tables this migration maps into STRATA records. Every other source table
/// that contains rows is reported in `skipped_tables`.
#[cfg(feature = "sqlite-reader")]
const MAPPED_TABLES: &[&str] = &[
    "knowledge_nodes",
    "memory_connections",
    "fsrs_cards",
    "sync_tombstones",
    "deletion_tombstones",
];

/// Declared edge kinds: the 8 types a user or tool writes through the typed
/// edge API. Anything else on a v3 `memory_connections` row is inferred
/// history and migrates as [`LEGACY_INFERRED_KIND`], never as one of these.
pub const STRATA_EDGE_VOCABULARY: [&str; 8] = [
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

/// Kind stored in the existing `EdgeRecord.link_type` string for every
/// inferred v3 link (similarity, entity, keyword, or any other type that
/// was not an explicit declaration). Not a causal kind, and not a new field.
pub const LEGACY_INFERRED_KIND: &str = "legacy_inferred";

/// Hook invoked on the staging directory after the receipt is sealed and
/// before the rename. Only the lock holder runs it.
pub type BeforePublish = Box<dyn FnOnce(&Path) -> Result<(), String>>;

/// Options for one migration run.
#[derive(Default)]
pub struct MigrateOptions {
    /// Read and verify the source, report counts, write nothing.
    pub dry_run: bool,
    /// Allow migrating a source with a non-empty `-wal` by snapshot-copying
    /// db + sidecars to a scratch directory first. The original is still
    /// never modified.
    pub accept_wal_snapshot: bool,
    /// Pin the strata log seed (signing key + segment ids derive from it).
    /// `None` derives the seed from the source BLAKE3, which already makes
    /// two runs over one source byte-identical.
    pub seed: Option<[u8; 32]>,
    /// Runs after the staging lock is held and before any log is created.
    /// Only the lock holder runs it. An error deletes staging and leaves
    /// the destination untouched.
    pub before_import: Option<BeforePublish>,
    /// Runs on the staging directory after the receipt is sealed and before
    /// the rename. Only the process holding the staging lock runs it. An
    /// error deletes staging and leaves the destination untouched.
    pub before_publish: Option<BeforePublish>,
}

impl std::fmt::Debug for MigrateOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MigrateOptions")
            .field("dry_run", &self.dry_run)
            .field("accept_wal_snapshot", &self.accept_wal_snapshot)
            .field("seed", &self.seed)
            .field(
                "before_import",
                &self.before_import.as_ref().map(|_| "Some"),
            )
            .field(
                "before_publish",
                &self.before_publish.as_ref().map(|_| "Some"),
            )
            .finish()
    }
}

/// Everything that can stop a migration. Nothing is ever half-written: the
/// strata log is append-only, so a failed run leaves already-appended frames
/// durable and simply reports the error.
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    /// The source path does not exist.
    #[error("migration source not found: {0}")]
    SourceNotFound(String),
    /// The source exists but is neither a portable archive, a SQLite db,
    /// nor a directory containing `vestige.db`.
    #[error("unsupported migration source: {0}")]
    UnsupportedSource(String),
    /// Opening or reading the source failed.
    #[error("source store error: {0}")]
    Source(String),
    /// The source has a non-empty `-wal`; rerun with `--accept-wal-snapshot`
    /// to migrate from a consistent snapshot copy.
    #[error("refusing non-empty WAL {path}; rerun with --accept-wal-snapshot")]
    WalPresent {
        /// Path of the offending `-wal` file.
        path: String,
    },
    /// The `receipt_envelopes` hash chain is broken. The message names the
    /// first break; nothing is written.
    #[error("broken receipt_envelopes hash chain: {0}")]
    BrokenEnvelopeChain(String),
    /// The destination exists and is not a finished migration of this source.
    /// Decided before any file is created in the destination.
    #[error("destination {path} is not empty and is not a completed migration of this source")]
    DestinationNotEmpty {
        /// The refusing destination directory.
        path: String,
    },
    /// The source's BLAKE3 changed while frames were being appended into
    /// the staging directory. Staging is removed; the destination is untouched.
    #[error(
        "source changed during migration (before {before}, after {after}); the destination log is incomplete and must not be trusted"
    )]
    SourceTampered {
        /// BLAKE3 taken before the first read.
        before: String,
        /// BLAKE3 taken before the seal.
        after: String,
    },
    /// A source row could not be decoded into a record.
    #[error("corrupt source data: {0}")]
    Corrupt(String),
    /// The strata log refused or failed an operation.
    #[error("strata log error: {0}")]
    Strata(String),
    /// The determinism kernel refused an operation.
    #[error("strata kernel error: {0}")]
    Kernel(String),
    /// Filesystem error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Outcome of one migration run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MigrationReport {
    /// NODE frames appended (knowledge_nodes + walk_receipt references).
    pub nodes: u64,
    /// EDGE frames appended.
    pub edges: u64,
    /// FSRS_REVIEW frames appended.
    pub fsrs_events: u64,
    /// Source tables that contained rows but have no STRATA mapping.
    pub skipped_tables: Vec<String>,
    /// Whether kernel replay verification AND log tail verification both
    /// passed over the finished log.
    pub verify_passed: bool,
    /// `node_embeddings` rows whose vector values were never read.
    pub dropped_vectors: u64,
    /// Last verified `receipt_envelopes` entry digest (empty = none).
    pub envelope_head: String,
    /// BLAKE3 hex of the source files (identical before and after; the run
    /// aborts otherwise). Empty for portable-archive sources.
    pub source_blake3: String,
    /// The sealed receipt digest, hex (`None` under `--dry-run`).
    pub receipt_digest: Option<String>,
    /// Whether the sealed receipt verifies (checksum + ed25519 signature).
    pub receipt_verified: bool,
    /// True when the destination already carried a receipt for this source:
    /// nothing was written; the existing receipt is echoed.
    pub idempotent_reuse: bool,
    /// Wall-clock duration of the migration.
    #[serde(serialize_with = "ser_duration_secs", rename = "durationSeconds")]
    pub duration: Duration,
}

fn ser_duration_secs<S: serde::Serializer>(
    duration: &Duration,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(duration.as_secs_f64())
}

/// Migrate from `<src>` (portable archive JSON, SQLite db file, or data
/// directory containing `vestige.db`) into a STRATA log at `<dst>`, with
/// default options.
#[cfg(feature = "sqlite-reader")]
pub fn migrate(source: &Path, strata_dir: &Path) -> Result<MigrationReport, MigrationError> {
    migrate_with_options(source, strata_dir, MigrateOptions::default())
}

/// Migrate with explicit options. See [`MigrateOptions`].
#[cfg(feature = "sqlite-reader")]
pub fn migrate_with_options(
    source: &Path,
    strata_dir: &Path,
    options: MigrateOptions,
) -> Result<MigrationReport, MigrationError> {
    let started = Instant::now();
    let scratch = tempfile::tempdir()?;
    let (files, _snapshotted) =
        source::prepare_source(source, options.accept_wal_snapshot, scratch.path())?;

    if !source::is_sqlite_file(&files.db)? {
        // Portable-archive JSON path: no SQLite, no guard, no source hash.
        let snapshot = source::load_snapshot(&files)?;
        if options.dry_run {
            return Ok(dry_run_report(&snapshot, "", started));
        }
        // Portable archives have no source hash, so a finished log is not
        // treated as an idempotent re-run of "this" source.
        return stage_import(
            strata_dir,
            options.seed,
            "",
            false,
            snapshot,
            started,
            None,
            options.before_import,
            options.before_publish,
        );
    }

    // ---- hash BEFORE any SQL read --------------------------------------
    let blake3_before = files.blake3_hex()?;

    // ---- read-only snapshot (verifies the envelope chain) ---------------
    let snapshot = source::load_snapshot(&files)?;

    if options.dry_run {
        return Ok(dry_run_report(&snapshot, &blake3_before, started));
    }

    // Writes go to a sibling staging directory and land in `strata_dir`
    // only via rename, after the source re-hash. A non-empty destination
    // is classified by reading it, never by opening a log in it.
    stage_import(
        strata_dir,
        options.seed,
        &blake3_before,
        true,
        snapshot,
        started,
        Some(files),
        options.before_import,
        options.before_publish,
    )
}

/// Appended to the destination directory's file name. The first-launch
/// destination is `<data-dir>/log`, so its staging directory is
/// `<data-dir>/log.strata-staging`.
pub const STAGING_SUFFIX: &str = ".strata-staging";

/// Lock file inside the staging directory. A dotfile, so destination
/// occupancy and the migrator ignore it. Held with `File::try_lock`.
#[cfg(feature = "sqlite-reader")]
const STAGING_LOCK_NAME: &str = ".upgrade.lock";

/// How long a loser looks for the lock file after `create_dir` and before
/// `try_lock`. This covers that gap only. A held lock is not abandoned on a timer.
#[cfg(feature = "sqlite-reader")]
const LOCK_FILE_APPEAR: Duration = Duration::from_millis(200);

/// Sibling of `dest`: `<dest>`'s file name plus [`STAGING_SUFFIX`].
#[cfg(feature = "sqlite-reader")]
fn staging_path(dest: &Path) -> std::path::PathBuf {
    let name = dest.file_name().unwrap_or(std::ffi::OsStr::new("strata"));
    let mut staging_name = name.to_os_string();
    staging_name.push(STAGING_SUFFIX);
    dest.with_file_name(staging_name)
}

/// True when `dir` contains anything other than dotfiles. A missing path
/// is empty. A non-directory is occupied.
#[cfg(feature = "sqlite-reader")]
fn destination_occupied(dir: &Path) -> Result<bool, MigrationError> {
    if !dir.exists() {
        return Ok(false);
    }
    if !dir.is_dir() {
        return Ok(true);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Read a MIGRATION_RECEIPT for `source_blake3` without opening the log.
/// Opening would create `strata.key` and, on a sealed tail, a new segment.
#[cfg(feature = "sqlite-reader")]
fn receipt_matching(dir: &Path, source_blake3: &str) -> Option<records::MigrationReceipt> {
    use borsh::BorshDeserialize;

    if !dir.is_dir() {
        return None;
    }
    let mut segs: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "seg"))
        .collect();
    segs.sort();
    for path in segs {
        let bytes = std::fs::read(&path).ok()?;
        let mut cursor = bytes.as_slice();
        if strata::SegmentHeader::deserialize_reader(&mut cursor).is_err() {
            return None;
        }
        loop {
            if cursor.is_empty() {
                break;
            }
            let frame = match strata::Frame::deserialize_reader(&mut cursor) {
                Ok(frame) => frame,
                Err(_) => break,
            };
            if frame.kind != records::KIND_MIGRATION_RECEIPT {
                continue;
            }
            if let Ok(receipt) = records::decode_receipt(&frame.payload) {
                if receipt.body.source_blake3_before == source_blake3 {
                    return Some(receipt);
                }
            }
        }
    }
    None
}

/// Import into `dest` via a sibling staging directory, then rename.
///
/// `allow_idempotent` is set for SQLite sources, whose receipt is keyed by
/// the source BLAKE3. Nothing is written to `dest` until every pre-check
/// has passed. The winner is `create_dir` of the staging directory plus
/// `File::try_lock` on a file inside it. The kernel drops that lock when
/// the process dies, including SIGKILL. A later process that can take the
/// lock wipes the staging directory and starts over. A process that finds
/// the lock held waits until the rename publishes `dest`.
#[cfg(feature = "sqlite-reader")]
#[allow(clippy::too_many_arguments)]
fn stage_import(
    dest: &Path,
    seed: Option<[u8; 32]>,
    source_blake3: &str,
    allow_idempotent: bool,
    snapshot: source::SourceSnapshot,
    started: Instant,
    files: Option<source::SourceFiles>,
    mut before_import: Option<BeforePublish>,
    mut before_publish: Option<BeforePublish>,
) -> Result<MigrationReport, MigrationError> {
    let staging = staging_path(dest);
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent)?;
    }
    loop {
        if destination_occupied(dest)? {
            if allow_idempotent {
                if let Some(receipt) = receipt_matching(dest, source_blake3) {
                    return Ok(idempotent_report(&snapshot, &receipt, started));
                }
            }
            return Err(MigrationError::DestinationNotEmpty {
                path: dest.display().to_string(),
            });
        }
        match std::fs::create_dir(&staging) {
            Ok(()) => {
                let lock = match lock_new_staging(&staging) {
                    Ok(lock) => lock,
                    Err(err) => {
                        let _ = std::fs::remove_dir_all(&staging);
                        return Err(err);
                    }
                };
                return import_holding_lock(
                    lock,
                    &staging,
                    dest,
                    seed,
                    source_blake3,
                    snapshot,
                    started,
                    files,
                    before_import.take(),
                    before_publish.take(),
                );
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                wait_or_reclaim(&staging, dest)?;
            }
            Err(err) => return Err(err.into()),
        }
    }
}

#[cfg(feature = "sqlite-reader")]
#[allow(clippy::too_many_arguments)]
fn import_holding_lock(
    lock: std::fs::File,
    staging: &Path,
    dest: &Path,
    seed: Option<[u8; 32]>,
    source_blake3: &str,
    snapshot: source::SourceSnapshot,
    started: Instant,
    files: Option<source::SourceFiles>,
    before_import: Option<BeforePublish>,
    before_publish: Option<BeforePublish>,
) -> Result<MigrationReport, MigrationError> {
    if let Some(hook) = before_import {
        if let Err(detail) = hook(staging) {
            return Err(discard_staging(
                lock,
                staging,
                MigrationError::Strata(detail),
            ));
        }
    }
    let log = match open_log(staging, seed, source_blake3) {
        Ok(log) => log,
        Err(err) => return Err(discard_staging(lock, staging, err)),
    };
    let outcome = match migrate_snapshot_into(&snapshot, &log, source_blake3) {
        Ok(outcome) => outcome,
        Err(err) => {
            drop(log);
            return Err(discard_staging(lock, staging, err));
        }
    };

    // Test seam: the kill-mid-import test sets this so SIGKILL lands after
    // frames are durable in staging and before the receipt is sealed.
    if std::env::var_os("STRATA_MIGRATE_SIGKILL_WINDOW").is_some() {
        std::thread::sleep(Duration::from_secs(60));
    }

    let sealed_hash = if let Some(files) = &files {
        let blake3_after = match files.blake3_hex() {
            Ok(hash) => hash,
            Err(err) => {
                drop(log);
                return Err(discard_staging(lock, staging, err));
            }
        };
        if source_blake3 != blake3_after {
            drop(log);
            return Err(discard_staging(
                lock,
                staging,
                MigrationError::SourceTampered {
                    before: source_blake3.to_string(),
                    after: blake3_after,
                },
            ));
        }
        blake3_after
    } else {
        source_blake3.to_string()
    };

    let report = match finish(log, staging, outcome, snapshot, &sealed_hash, started) {
        Ok(report) => report,
        Err(err) => return Err(discard_staging(lock, staging, err)),
    };
    if !report.verify_passed {
        return Err(discard_staging(
            lock,
            staging,
            MigrationError::Strata("replay verification failed".into()),
        ));
    }
    if let Some(hook) = before_publish {
        if let Err(detail) = hook(staging) {
            return Err(discard_staging(
                lock,
                staging,
                MigrationError::Strata(detail),
            ));
        }
    }
    if let Err(err) = publish(staging, dest) {
        return Err(discard_staging(lock, staging, err));
    }
    let _ = std::fs::remove_file(dest.join(STAGING_LOCK_NAME));
    drop(lock);
    Ok(report)
}

#[cfg(feature = "sqlite-reader")]
fn discard_staging(lock: std::fs::File, staging: &Path, err: MigrationError) -> MigrationError {
    let _ = std::fs::remove_dir_all(staging);
    drop(lock);
    err
}

#[cfg(feature = "sqlite-reader")]
fn lock_new_staging(staging: &Path) -> Result<std::fs::File, MigrationError> {
    let path = staging.join(STAGING_LOCK_NAME);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(MigrationError::Io(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "staging lock already held",
        ))),
        Err(std::fs::TryLockError::Error(err)) => Err(err.into()),
    }
}

#[cfg(feature = "sqlite-reader")]
enum Held {
    Acquired(std::fs::File),
    Busy,
    Missing,
}

#[cfg(feature = "sqlite-reader")]
fn try_hold(staging: &Path) -> Result<Held, MigrationError> {
    let path = staging.join(STAGING_LOCK_NAME);
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Held::Missing),
        Err(err) => return Err(err.into()),
    };
    match file.try_lock() {
        Ok(()) => Ok(Held::Acquired(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(Held::Busy),
        Err(std::fs::TryLockError::Error(err)) => Err(err.into()),
    }
}

/// Staging already exists. Take the lock and wipe it when the owner is
/// dead, or wait while a live owner still holds it.
#[cfg(feature = "sqlite-reader")]
fn wait_or_reclaim(staging: &Path, dest: &Path) -> Result<(), MigrationError> {
    match try_hold(staging)? {
        Held::Acquired(lock) => {
            let _ = std::fs::remove_dir_all(staging);
            drop(lock);
            return Ok(());
        }
        Held::Missing if !lock_file_appears(staging, dest) => {
            return reclaim_if_abandoned(staging, dest);
        }
        Held::Missing | Held::Busy => {}
    }
    loop {
        if destination_occupied(dest)? || !staging.exists() {
            return Ok(());
        }
        match try_hold(staging)? {
            Held::Acquired(lock) => {
                let _ = std::fs::remove_dir_all(staging);
                drop(lock);
                return Ok(());
            }
            Held::Busy => std::thread::sleep(Duration::from_millis(20)),
            Held::Missing => {
                if !lock_file_appears(staging, dest) {
                    return reclaim_if_abandoned(staging, dest);
                }
            }
        }
    }
}

#[cfg(feature = "sqlite-reader")]
fn lock_file_appears(staging: &Path, dest: &Path) -> bool {
    let path = staging.join(STAGING_LOCK_NAME);
    let started = Instant::now();
    while started.elapsed() < LOCK_FILE_APPEAR {
        if path.exists() {
            return true;
        }
        if !staging.exists() || destination_occupied(dest).unwrap_or(false) {
            return path.exists();
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    path.exists()
}

#[cfg(feature = "sqlite-reader")]
fn reclaim_if_abandoned(staging: &Path, dest: &Path) -> Result<(), MigrationError> {
    if destination_occupied(dest)? || !staging.exists() {
        return Ok(());
    }
    // The directory exists and nobody holds the lock file. The creator died
    // between `create_dir` and `try_lock`, or the file never appeared.
    let _ = std::fs::remove_dir_all(staging);
    Ok(())
}

#[cfg(feature = "sqlite-reader")]
fn publish(staging: &Path, dest: &Path) -> Result<(), MigrationError> {
    if dest.exists() {
        // Occupancy ignores dotfiles. Keep them; the rename replaces the shell.
        for entry in std::fs::read_dir(dest)? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                std::fs::rename(entry.path(), staging.join(&name))?;
            }
        }
        std::fs::remove_dir_all(dest)?;
    }
    std::fs::rename(staging, dest)?;
    Ok(())
}

#[cfg(feature = "sqlite-reader")]
#[allow(clippy::too_many_arguments)]
fn idempotent_report(
    snapshot: &source::SourceSnapshot,
    receipt: &records::MigrationReceipt,
    started: Instant,
) -> MigrationReport {
    MigrationReport {
        nodes: table_rows(snapshot, "knowledge_nodes"),
        edges: table_rows(snapshot, "memory_connections"),
        fsrs_events: table_rows(snapshot, "fsrs_cards"),
        skipped_tables: skipped_tables_for(snapshot),
        verify_passed: true,
        dropped_vectors: snapshot.dropped_vectors,
        envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
        source_blake3: receipt.body.source_blake3_before.clone(),
        receipt_digest: Some(hex32(&receipt.checksum)),
        receipt_verified: receipt.verify_checksum() && receipt.verify_signature(),
        duration: started.elapsed(),
        idempotent_reuse: true,
    }
}

#[cfg(feature = "sqlite-reader")]
fn open_log(
    strata_dir: &Path,
    pinned_seed: Option<[u8; 32]>,
    _source_blake3: &str,
) -> Result<StrataLog, MigrationError> {
    // The log signing seed comes from OS entropy unless a test pins it.
    // Deriving it from the source BLAKE3 put the seed inside the log itself
    // (the PARAMS frame), so any reader could re-derive strata.key and forge
    // segments (audit finding (a): CONFIRMED forge).
    let seed = match pinned_seed {
        Some(seed) => seed,
        None => {
            let mut seed = [0u8; 32];
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| {
                    use std::io::Read;
                    f.read_exact(&mut seed)?;
                    Ok(())
                })
                .map_err(|e| {
                    MigrationError::Io(std::io::Error::new(
                        e.kind(),
                        format!("no OS entropy for the log signing seed: {e}"),
                    ))
                })?;
            seed
        }
    };
    StrataLog::open_seeded(strata_dir, seed).map_err(|e| MigrationError::Strata(e.to_string()))
}

#[cfg(feature = "sqlite-reader")]
fn dry_run_report(
    snapshot: &source::SourceSnapshot,
    source_blake3: &str,
    started: Instant,
) -> MigrationReport {
    MigrationReport {
        nodes: table_rows(snapshot, "knowledge_nodes"),
        edges: table_rows(snapshot, "memory_connections"),
        fsrs_events: table_rows(snapshot, "fsrs_cards"),
        skipped_tables: skipped_tables_for(snapshot),
        // A dry run writes nothing; the envelope chain is verified during
        // the read, so reaching this point means the chain held.
        verify_passed: true,
        dropped_vectors: snapshot.dropped_vectors,
        envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
        source_blake3: source_blake3.to_string(),
        receipt_digest: None,
        receipt_verified: false,
        idempotent_reuse: false,
        duration: started.elapsed(),
    }
}

#[cfg(feature = "sqlite-reader")]
fn table_rows(snapshot: &source::SourceSnapshot, table: &str) -> u64 {
    source::table(&snapshot.archive, table)
        .map(|t| t.rows.len() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "sqlite-reader")]
fn skipped_tables_for(snapshot: &source::SourceSnapshot) -> Vec<String> {
    // Driven by sqlite_master (audit finding): every nonempty table without
    // a STRATA mapping is named, so the receipt never overclaims.
    snapshot
        .all_nonempty_tables
        .iter()
        .filter(|(name, _)| !MAPPED_TABLES.contains(&name.as_str()))
        .map(|(name, _)| name.clone())
        .collect()
}

/// Everything produced during one replay.
#[cfg(feature = "sqlite-reader")]
struct ReplayOutcome {
    nodes: u64,
    edges: u64,
    fsrs_events: u64,
    /// Hash of the last fold checkpoint (the replay anchor), or `[0; 32]`
    /// when the log carries no checkpoint.
    anchor: [u8; 32],
}

/// Replay a snapshot into an already-open log: GENESIS/PARAMS on a fresh
/// log, then nodes (+ walk-receipt reference nodes), edges, tombstones,
/// supersessions, FSRS folds, and the fold checkpoint. The
/// MIGRATION_RECEIPT is appended by [`finish`] once the source re-hash has
/// been confirmed.
#[cfg(feature = "sqlite-reader")]
fn migrate_snapshot_into(
    snapshot: &source::SourceSnapshot,
    log: &StrataLog,
    source_blake3: &str,
) -> Result<ReplayOutcome, MigrationError> {
    let archive = &snapshot.archive;

    let (mut node_records, kernel_ids, supersessions) = extract_nodes(archive)?;
    let edge_records = extract_edges(archive, &kernel_ids)?;
    let tombstones = extract_tombstones(archive)?;
    let walk_nodes = extract_walk_receipts(snapshot, &kernel_ids)?;
    attach_fsrs_legacy(archive, &kernel_ids, &mut node_records)?;

    let mut nodes = 0u64;
    let mut edges = 0u64;
    let mut fsrs_events = 0u64;

    let mut writer = Writer::new(log);
    if log.head().frames_total == 0 {
        writer.push(
            records::KIND_GENESIS,
            borsh::to_vec(&GenesisRecord {
                record_version: RECORD_VERSION,
                archive_format: archive.archive_format.clone(),
                vestige_version: archive.vestige_version.clone(),
                schema_version: snapshot.schema_version,
            }),
        )?;
        writer.push(
            records::KIND_PARAMS,
            borsh::to_vec(&ParamsRecord {
                record_version: RECORD_VERSION,
                params_id: PARAMS_ID.to_string(),
                schema_version: snapshot.schema_version,
                source_blake3: source_blake3.to_string(),
                envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
            }),
        )?;
    }
    for record in node_records.iter().chain(walk_nodes.iter()) {
        writer.push(records::KIND_NODE, borsh::to_vec(record))?;
        nodes += 1;
    }
    for record in &edge_records {
        writer.push(records::KIND_EDGE, borsh::to_vec(record))?;
        edges += 1;
    }
    for record in &tombstones {
        writer.push(records::KIND_TOMBSTONE, borsh::to_vec(record))?;
    }
    for record in &supersessions {
        writer.push(records::KIND_SUPERSESSION, borsh::to_vec(record))?;
    }

    // ---- fsrs_cards -> review-event folds --------------------------------
    // event_seq MUST equal the frame seq, so predict it from the writer's
    // view of the log head and assert the acks below.
    if let Some(table) = source::table(archive, "fsrs_cards") {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            let memory_id = row.text("memory_id")?.to_string();
            let kernel_id = *kernel_ids.get(&memory_id).ok_or_else(|| {
                MigrationError::Corrupt(format!(
                    "fsrs_cards row references unknown memory {memory_id}"
                ))
            })?;
            let reps = row.integer_or("reps", 0)?.clamp(0, u32::MAX as i64);
            let lapses = row.integer_or("lapses", 0)?.clamp(0, reps);
            let reviewed_at_ms = last_review_ms(&row)?;
            let ratings = fsrs_ratings_for(reps, lapses);
            let last = ratings.len().saturating_sub(1);
            for (index, rating) in ratings.into_iter().enumerate() {
                let event = ReviewEvent {
                    card_id: kernel_id,
                    rating,
                    event_seq: writer.next_seq(),
                };
                let mut payload = borsh::to_vec(&event)
                    .map_err(|e| MigrationError::Corrupt(format!("borsh encode review: {e}")))?;
                // Only the card's latest review has a known wall clock.
                // The option tag is always written.
                let clock = if index == last { reviewed_at_ms } else { None };
                payload.extend(
                    borsh::to_vec(&clock).map_err(|e| {
                        MigrationError::Corrupt(format!("borsh encode review: {e}"))
                    })?,
                );
                writer.push(records::KIND_FSRS_REVIEW, Ok(payload))?;
                fsrs_events += 1;
            }
        }
    }
    writer.flush()?;

    // ---- fold + checkpoint ------------------------------------------------
    // Fold ALL review events in the log (an earlier run into the same
    // directory may have contributed some); the checkpoint state covers the
    // whole log, not just this run's events.
    let before_checkpoint = read_snapshot(log)?;
    let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1)
        .map_err(|e| MigrationError::Kernel(e.to_string()))?;
    let mut state = strata_kernel::state::State::default();
    kernel.apply_all(&mut state, before_checkpoint.reviews.iter());

    let anchor = if writer.appended > 0 {
        let prev = before_checkpoint
            .checkpoints
            .last()
            .map_or([0u8; 32], checkpoint_hash);
        let checkpoint = Checkpoint::seal(ALGO_V1, writer.last_frame_seq, prev, &state);
        let anchor = checkpoint_hash(&checkpoint);
        let bytes = borsh::to_vec(&checkpoint)
            .map_err(|e| MigrationError::Corrupt(format!("borsh encode checkpoint: {e}")))?;
        log.append(records::KIND_CHECKPOINT, &bytes)
            .map_err(|e| MigrationError::Strata(e.to_string()))?;
        anchor
    } else {
        // Nothing new was appended; keep the existing head checkpoint as the
        // anchor instead of duplicating it (duplicate log_seq would trip the
        // verifier's strict checkpoint ordering).
        before_checkpoint
            .checkpoints
            .last()
            .map_or([0u8; 32], checkpoint_hash)
    };

    Ok(ReplayOutcome {
        nodes,
        edges,
        fsrs_events,
        anchor,
    })
}

/// Append the sealed MIGRATION_RECEIPT, close the segment, verify the log,
/// and assemble the report.
#[cfg(feature = "sqlite-reader")]
fn finish(
    log: StrataLog,
    strata_dir: &Path,
    outcome: ReplayOutcome,
    snapshot: source::SourceSnapshot,
    source_blake3: &str,
    started: Instant,
) -> Result<MigrationReport, MigrationError> {
    // Per-table source counts plus the walk receipts, sorted by table name
    // (H6: ordered collections only in hashed state).
    let mut table_counts = snapshot.table_counts.clone();
    if !snapshot.walk_receipts.is_empty() {
        table_counts.push((
            "walk_receipts".to_string(),
            snapshot.walk_receipts.len() as u64,
        ));
        table_counts.sort();
    }

    let signing = records::load_or_create_receipt_key(&receipt_key_dir_of(strata_dir))?;
    let receipt = MigrationReceipt::seal(
        ReceiptBody {
            record_version: RECORD_VERSION,
            source_blake3_before: source_blake3.to_string(),
            source_blake3_after: source_blake3.to_string(),
            schema_version: snapshot.schema_version,
            envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
            counts: table_counts,
            dropped_vectors: snapshot.dropped_vectors,
            dropped_columns: vec![
                "node_embeddings.embedding (H1: vector values are dropped and counted)".to_string(),
                "schema_version.applied_at".to_string(),
                "receipt_envelopes.receipt_id/projection_json/issued_at/stored_at".to_string(),
            ],
            signing_key_id: RECEIPT_SIGNING_KEY_ID.to_string(),
        },
        &signing,
    );
    let receipt_bytes = borsh::to_vec(&receipt)
        .map_err(|e| MigrationError::Corrupt(format!("borsh encode receipt: {e}")))?;
    log.append(records::KIND_MIGRATION_RECEIPT, &receipt_bytes)
        .map_err(|e| MigrationError::Strata(e.to_string()))?;

    log.seal()
        .map_err(|e| MigrationError::Strata(e.to_string()))?;

    let verify_passed = match verify_migrated(&log, outcome.anchor) {
        Ok(passed) => passed,
        Err(reason) => {
            eprintln!("strata-migrate: verification error: {reason}");
            false
        }
    };

    Ok(MigrationReport {
        nodes: outcome.nodes,
        edges: outcome.edges,
        fsrs_events: outcome.fsrs_events,
        skipped_tables: skipped_tables_for(&snapshot),
        verify_passed,
        dropped_vectors: snapshot.dropped_vectors,
        envelope_head: snapshot.envelope_head.unwrap_or_default(),
        source_blake3: source_blake3.to_string(),
        receipt_digest: Some(hex32(&receipt.checksum)),
        receipt_verified: receipt.verify_checksum() && receipt.verify_signature(),
        idempotent_reuse: false,
        duration: started.elapsed(),
    })
}

/// The receipt-signing key lives NEXT TO the destination log, never inside
/// the log: `<parent-of---to>/receipt-signing.key`. The log carries only
/// the verifying key.
#[cfg(feature = "sqlite-reader")]
fn receipt_key_dir_of(strata_dir: &Path) -> std::path::PathBuf {
    strata_dir.parent().unwrap_or(strata_dir).to_path_buf()
}

#[cfg(feature = "sqlite-reader")]
fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Batched append helper: predicts frame seqs from the log head, asserts the
/// returned acks match, and never holds more than `BATCH_FRAMES` frames.
#[cfg(feature = "sqlite-reader")]
struct Writer<'a> {
    log: &'a StrataLog,
    batch: Vec<(u8, Vec<u8>)>,
    /// Seq the next queued frame will receive.
    next: u64,
    /// Frames appended by this writer so far.
    appended: u64,
    /// Seq of the last frame this writer appended (0 if none).
    last_frame_seq: u64,
}

#[cfg(feature = "sqlite-reader")]
impl<'a> Writer<'a> {
    fn new(log: &'a StrataLog) -> Self {
        Self {
            log,
            batch: Vec::new(),
            next: log.head().next_seq,
            appended: 0,
            last_frame_seq: 0,
        }
    }

    /// Predicted seq of the next frame.
    fn next_seq(&self) -> u64 {
        self.next
    }

    fn push(
        &mut self,
        kind: u8,
        payload: Result<Vec<u8>, borsh::io::Error>,
    ) -> Result<(), MigrationError> {
        let payload = payload
            .map_err(|e| MigrationError::Corrupt(format!("borsh encode kind {kind:#04x}: {e}")))?;
        if self.batch.len() >= BATCH_FRAMES {
            self.flush()?;
        }
        self.batch.push((kind, payload));
        self.last_frame_seq = self.next;
        self.next += 1;
        self.appended += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MigrationError> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let expected_base = self.next - self.batch.len() as u64;
        let batch = std::mem::take(&mut self.batch);
        let acks = self
            .log
            .append_batch(batch)
            .map_err(|e| MigrationError::Strata(e.to_string()))?;
        for (offset, ack) in acks.iter().enumerate() {
            if ack.seq != expected_base + offset as u64 {
                return Err(MigrationError::Strata(format!(
                    "append ack mismatch: frame {} acked at seq {}",
                    expected_base + offset as u64,
                    ack.seq
                )));
            }
        }
        Ok(())
    }
}

/// Kernel replay + log tail verification over the finished log.
#[cfg(feature = "sqlite-reader")]
fn verify_migrated(log: &StrataLog, anchor: [u8; 32]) -> Result<bool, MigrationError> {
    let tail_ok = log.verify_tail().is_ok();
    let snapshot = read_snapshot(log)?;
    let events = snapshot
        .reviews
        .iter()
        .map(|event| {
            let bytes = borsh::to_vec(event)
                .map_err(|e| MigrationError::Corrupt(format!("re-encode review event: {e}")))?;
            Ok((event.event_seq, blake3::hash(&bytes).into(), *event))
        })
        .collect::<Result<Vec<_>, MigrationError>>()?;
    let replay = verify_with_head(&snapshot.checkpoints, Some(anchor), events.into_iter());
    let replay_ok = replay.is_ok();
    if let Err(error) = replay {
        eprintln!("strata-migrate: kernel verify failed: {error}");
    }
    Ok(tail_ok && replay_ok)
}

/// `fsrs_cards.last_review` as unix epoch ms.
///
/// NULL, `""`, or a table without the column is `None` (the option tag is
/// still written). A non-empty value that is not RFC3339 is corrupt.
#[cfg(feature = "sqlite-reader")]
fn last_review_ms(row: &source::Row<'_>) -> Result<Option<i64>, MigrationError> {
    if !row.columns().iter().any(|column| column == "last_review") {
        return Ok(None);
    }
    match row.opt_text("last_review")? {
        Some(raw) if !raw.is_empty() => Ok(Some(source::timestamp_ms(raw)?)),
        _ => Ok(None),
    }
}

/// Deterministic rating series reproducing an fsrs_cards row exactly:
/// `reps - lapses` good (3) reviews, then `lapses` again (1) reviews. The
/// fold yields `review_count == reps` and `lapse_count == lapses`; the final
/// phase is Relearning iff the card ever lapsed, else Review.
#[cfg(any(feature = "sqlite-reader", test))]
fn fsrs_ratings_for(reps: i64, lapses: i64) -> Vec<u8> {
    if reps <= 0 {
        return Vec::new();
    }
    let reps = reps.min(u32::MAX as i64);
    let again = lapses.clamp(0, reps) as usize;
    let good = (reps as usize) - again;
    let mut ratings = Vec::with_capacity(good + again);
    ratings.resize(good, 3);
    ratings.resize(good + again, 1);
    ratings
}

#[cfg(feature = "sqlite-reader")]
const FSRS_MAPPED_COLUMNS: &[&str] = &["memory_id", "reps", "lapses"];

/// FSRS final state the fold cannot reproduce (stability/difficulty floats,
/// due dates, phase) rides on the node's legacy capture (blocker 4).
#[cfg(feature = "sqlite-reader")]
fn attach_fsrs_legacy(
    archive: &PortableArchive,
    kernel_ids: &HashMap<String, u64>,
    node_records: &mut [NodeRecord],
) -> Result<(), MigrationError> {
    let Some(table) = source::table(archive, "fsrs_cards") else {
        return Ok(());
    };
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let memory_id = row.text("memory_id")?.to_string();
        let Some(kernel_id) = kernel_ids.get(&memory_id) else {
            continue;
        };
        let legacy = capture_legacy("fsrs_cards", &row, FSRS_MAPPED_COLUMNS)?;
        if let Some(node) = node_records.iter_mut().find(|n| n.kernel_id == *kernel_id) {
            node.legacy.extend(legacy);
        }
    }
    Ok(())
}

/// V40 `walk_receipts` rows become reference nodes tagged `migrated_from_v4`
/// (PR-0a spec 4d). Kernel ids continue after the knowledge nodes so the
/// dense identity space stays total.
#[cfg(feature = "sqlite-reader")]
fn extract_walk_receipts(
    snapshot: &source::SourceSnapshot,
    kernel_ids: &HashMap<String, u64>,
) -> Result<Vec<NodeRecord>, MigrationError> {
    let base_kernel_id = kernel_ids.values().copied().max().unwrap_or(0);
    let records = snapshot
        .walk_receipts
        .iter()
        .enumerate()
        .map(|(offset, row)| NodeRecord {
            record_version: RECORD_VERSION,
            legacy_id: row.receipt_id.clone(),
            kernel_id: base_kernel_id + offset as u64 + 1,
            content: row.canonical_json.clone(),
            node_type: "walk_receipt".to_string(),
            tags: vec!["migrated_from_v4".to_string()],
            created_ms: row.created_ms,
            updated_ms: row.created_ms,
            last_accessed_ms: row.created_ms,
            legacy: Vec::new(),
            source: None,
            source_updated_at_ms: None,
        })
        .collect();
    Ok(records)
}

/// Canonical string form of a snapshot value for legacy column capture:
/// ints/floats/text verbatim (floats are Display'd, deterministic), blobs hex.
#[cfg(feature = "sqlite-reader")]
fn legacy_value(value: &source::PortableValue) -> String {
    match value {
        PortableValue::Null => String::new(),
        PortableValue::Integer(v) => v.to_string(),
        PortableValue::Real(v) => v.to_string(),
        PortableValue::Text(t) => t.clone(),
        PortableValue::Blob(hex) => hex.clone(),
    }
}

/// Columns with dedicated fields on the record (everything else is captured
/// verbatim in `legacy`).
#[cfg(feature = "sqlite-reader")]
const NODE_MAPPED_COLUMNS: &[&str] = &[
    "id",
    "content",
    "node_type",
    "tags",
    "created_at",
    "updated_at",
    "last_accessed",
    "superseded_by",
];

#[cfg(feature = "sqlite-reader")]
fn capture_legacy(
    table: &str,
    row: &source::Row<'_>,
    mapped: &[&str],
) -> Result<Vec<(String, String)>, MigrationError> {
    let columns = row.columns();
    let mut out = Vec::new();
    for column in columns {
        if mapped.contains(&column.as_str()) {
            continue;
        }
        // `column` is already `&String`; `get` takes `&str`.
        let value = row.get(column)?;
        out.push((format!("{table}.{column}"), legacy_value(value)));
    }
    Ok(out)
}

/// Decoded `knowledge_nodes`: node records, the legacy→kernel id map (dense,
/// 1-based, source row order), and supersession pointers.
#[cfg(feature = "sqlite-reader")]
type NodeSet = (
    Vec<NodeRecord>,
    HashMap<String, u64>,
    Vec<SupersessionRecord>,
);

/// TEXT column, or `None` when the column is absent or empty.
#[cfg(feature = "sqlite-reader")]
fn column_text<'a>(row: &source::Row<'a>, name: &str) -> Result<Option<&'a str>, MigrationError> {
    if !row.columns().iter().any(|column| column == name) {
        return Ok(None);
    }
    Ok(row.opt_text(name)?.filter(|text| !text.is_empty()))
}

/// Lift the v3 source onto the store provenance fields.
///
/// A `(source_system, source_id)` pair becomes a [`SourceKey`]. A free-form
/// `source` string with no pair becomes `system = <text>`. `source_updated_at`
/// is the timestamp when that column is set. The v3.1.1 fixture has no
/// `source_updated_at` column, so a sourced row keeps its `updated_at`
/// instead. A row with neither a source nor a source timestamp yields
/// `(None, None)`.
#[cfg(feature = "sqlite-reader")]
fn node_provenance(
    row: &source::Row<'_>,
) -> Result<(Option<SourceKey>, Option<i64>), MigrationError> {
    let system = column_text(row, "source_system")?;
    let project = column_text(row, "source_project")?.unwrap_or("");
    let id = column_text(row, "source_id")?;
    let label = column_text(row, "source")?;
    let source = if let (Some(system), Some(id)) = (system, id) {
        Some(SourceKey {
            system: system.to_string(),
            project: project.to_string(),
            id: id.to_string(),
        })
    } else {
        label.map(|label| SourceKey {
            system: label.to_string(),
            project: String::new(),
            id: String::new(),
        })
    };
    let source_updated_at_ms = if row
        .columns()
        .iter()
        .any(|column| column == "source_updated_at")
    {
        match column_text(row, "source_updated_at")? {
            Some(raw) => Some(source::timestamp_ms(raw)?),
            None => None,
        }
    } else if source.is_some() {
        Some(source::timestamp_ms(row.text("updated_at")?)?)
    } else {
        None
    };
    Ok((source, source_updated_at_ms))
}

/// Decode `knowledge_nodes` into node records, the legacy→kernel id map
/// (dense, 1-based, source row order), and supersession pointers.
#[cfg(feature = "sqlite-reader")]
fn extract_nodes(archive: &PortableArchive) -> Result<NodeSet, MigrationError> {
    let Some(table) = source::table(archive, "knowledge_nodes") else {
        return Ok((Vec::new(), HashMap::new(), Vec::new()));
    };
    let has_superseded_by = table.columns.iter().any(|c| c == "superseded_by");

    let mut records = Vec::with_capacity(table.rows.len());
    let mut kernel_ids: HashMap<String, u64> = HashMap::with_capacity(table.rows.len());
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let legacy_id = row.text("id")?.to_string();
        let kernel_id = (index as u64) + 1;
        kernel_ids.insert(legacy_id.clone(), kernel_id);
        let legacy = capture_legacy("knowledge_nodes", &row, NODE_MAPPED_COLUMNS)?;
        let (source, source_updated_at_ms) = node_provenance(&row)?;
        records.push(NodeRecord {
            record_version: RECORD_VERSION,
            kernel_id,
            content: row.text("content")?.to_string(),
            node_type: row.text("node_type")?.to_string(),
            tags: source::parse_tags(row.opt_text("tags")?),
            created_ms: source::timestamp_ms(row.text("created_at")?)?,
            updated_ms: source::timestamp_ms(row.text("updated_at")?)?,
            last_accessed_ms: source::timestamp_ms(row.text("last_accessed")?)?,
            legacy,
            legacy_id,
            source,
            source_updated_at_ms,
        });
    }

    let mut supersessions = Vec::new();
    if has_superseded_by {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            if let Some(target) = row.opt_text("superseded_by")? {
                let superseded_legacy_id = row.text("id")?.to_string();
                let superseded_by_legacy_id = target.to_string();
                supersessions.push(SupersessionRecord {
                    record_version: RECORD_VERSION,
                    superseded_kernel_id: kernel_ids
                        .get(&superseded_legacy_id)
                        .copied()
                        .unwrap_or(0),
                    superseded_by_kernel_id: kernel_ids
                        .get(&superseded_by_legacy_id)
                        .copied()
                        .unwrap_or(0),
                    superseded_legacy_id,
                    superseded_by_legacy_id,
                });
            }
        }
    }
    Ok((records, kernel_ids, supersessions))
}

/// Decode `memory_connections` into edge records. FK cascades make dangling
/// edges impossible in a consistent store; a dangling edge in an archive is
/// corruption and stops the migration (fail-stop, never silently dropped).
///
/// A v3 `link_type` in [`STRATA_EDGE_VOCABULARY`] is a declared edge and
/// passes through. Every other type is inferred: the existing `link_type`
/// field becomes [`LEGACY_INFERRED_KIND`] (never `derived_from` or another
/// causal kind), `legacy_inferred` is set, and the original type is kept
/// for provenance.
#[cfg(feature = "sqlite-reader")]
fn extract_edges(
    archive: &PortableArchive,
    kernel_ids: &HashMap<String, u64>,
) -> Result<Vec<EdgeRecord>, MigrationError> {
    let Some(table) = source::table(archive, "memory_connections") else {
        return Ok(Vec::new());
    };
    let mut records = Vec::with_capacity(table.rows.len());
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let source_legacy_id = row.text("source_id")?.to_string();
        let target_legacy_id = row.text("target_id")?.to_string();
        let source_kernel_id = *kernel_ids.get(&source_legacy_id).ok_or_else(|| {
            MigrationError::Corrupt(format!(
                "memory_connections row references unknown source {source_legacy_id}"
            ))
        })?;
        let target_kernel_id = *kernel_ids.get(&target_legacy_id).ok_or_else(|| {
            MigrationError::Corrupt(format!(
                "memory_connections row references unknown target {target_legacy_id}"
            ))
        })?;
        let legacy_link_type = row.text("link_type")?.to_string();
        let (link_type, legacy_inferred) =
            if STRATA_EDGE_VOCABULARY.contains(&legacy_link_type.as_str()) {
                (legacy_link_type.clone(), false)
            } else {
                (LEGACY_INFERRED_KIND.to_string(), true)
            };
        let legacy = capture_legacy(
            "memory_connections",
            &row,
            &[
                "source_id",
                "target_id",
                "strength",
                "link_type",
                "created_at",
                "last_activated",
                "activation_count",
            ],
        )?;
        records.push(EdgeRecord {
            record_version: RECORD_VERSION,
            source_kernel_id,
            target_kernel_id,
            strength_q32: strata_kernel::canonical::to_q32_32(row.real("strength")?),
            link_type,
            legacy_inferred,
            legacy_link_type,
            created_ms: source::timestamp_ms(row.text("created_at")?)?,
            last_activated_ms: source::timestamp_ms(row.text("last_activated")?)?,
            activation_count: row.integer_or("activation_count", 0)? as i32,
            legacy,
            source_legacy_id,
            target_legacy_id,
        });
    }
    Ok(records)
}

/// Decode `sync_tombstones` and `deletion_tombstones`.
#[cfg(feature = "sqlite-reader")]
fn extract_tombstones(archive: &PortableArchive) -> Result<Vec<TombstoneRecord>, MigrationError> {
    let mut records = Vec::new();
    if let Some(table) = source::table(archive, "sync_tombstones") {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            records.push(TombstoneRecord {
                record_version: RECORD_VERSION,
                origin_table: "sync_tombstones".to_string(),
                source_table: row.opt_text("table_name")?.map(str::to_string),
                row_id: row.text("row_id")?.to_string(),
                deleted_ms: source::timestamp_ms(row.text("deleted_at")?)?,
                reason: row.opt_text("reason")?.map(str::to_string),
                node_type: None,
                tags: Vec::new(),
            });
        }
    }
    if let Some(table) = source::table(archive, "deletion_tombstones") {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            records.push(TombstoneRecord {
                record_version: RECORD_VERSION,
                origin_table: "deletion_tombstones".to_string(),
                source_table: None,
                row_id: row.text("memory_id")?.to_string(),
                deleted_ms: source::timestamp_ms(row.text("deleted_at")?)?,
                reason: row.opt_text("reason")?.map(str::to_string),
                node_type: row.opt_text("node_type")?.map(str::to_string),
                tags: source::parse_tags(row.opt_text("tags")?),
            });
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rating_series_reproduces_counters() {
        assert!(fsrs_ratings_for(0, 0).is_empty());
        assert_eq!(fsrs_ratings_for(3, 0), vec![3, 3, 3]);
        assert_eq!(fsrs_ratings_for(5, 2), vec![3, 3, 3, 1, 1]);
        assert_eq!(fsrs_ratings_for(2, 2), vec![1, 1]);
        // lapses clamped to reps
        assert_eq!(fsrs_ratings_for(2, 9), vec![1, 1]);
        // negative reps -> nothing
        assert!(fsrs_ratings_for(-3, 1).is_empty());
    }

    #[test]
    fn rating_series_folds_to_exact_counters() {
        let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1).unwrap();
        let mut state = strata_kernel::state::State::default();
        let events: Vec<ReviewEvent> = fsrs_ratings_for(6, 2)
            .into_iter()
            .enumerate()
            .map(|(i, rating)| ReviewEvent {
                card_id: 7,
                rating,
                event_seq: i as u64 + 1,
            })
            .collect();
        kernel.apply_all(&mut state, events.iter());
        let card = state.cards.get(&7).unwrap();
        assert_eq!(card.review_count, 6);
        assert_eq!(card.lapse_count, 2);
        assert_eq!(card.phase, strata_kernel::fsrs::CardPhase::Relearning);
    }

    #[test]
    fn rating_series_without_lapses_ends_in_review_phase() {
        let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1).unwrap();
        let mut state = strata_kernel::state::State::default();
        let events: Vec<ReviewEvent> = fsrs_ratings_for(2, 0)
            .into_iter()
            .enumerate()
            .map(|(i, rating)| ReviewEvent {
                card_id: 1,
                rating,
                event_seq: i as u64 + 1,
            })
            .collect();
        kernel.apply_all(&mut state, events.iter());
        let card = state.cards.get(&1).unwrap();
        assert_eq!(card.review_count, 2);
        assert_eq!(card.lapse_count, 0);
        assert_eq!(card.phase, strata_kernel::fsrs::CardPhase::Review);
    }

    #[cfg(feature = "sqlite-reader")]
    #[test]
    fn timestamps_parse_from_rfc3339_variants() {
        assert_eq!(
            source::timestamp_ms("2026-09-28T12:00:00+00:00").unwrap(),
            1_790_596_800_000
        );
        assert!(source::timestamp_ms("2026-09-28T12:00:00Z").is_ok());
        assert!(source::timestamp_ms("not a date").is_err());
    }

    #[cfg(feature = "sqlite-reader")]
    #[test]
    fn tags_parse_tolerantly() {
        assert_eq!(source::parse_tags(None), Vec::<String>::new());
        assert_eq!(source::parse_tags(Some("[]")), Vec::<String>::new());
        assert_eq!(
            source::parse_tags(Some(r#"["rust","memory"]"#)),
            vec!["rust".to_string(), "memory".to_string()]
        );
        assert_eq!(source::parse_tags(Some("null")), Vec::<String>::new());
        assert_eq!(source::parse_tags(Some("{broken")), Vec::<String>::new());
    }

    #[test]
    fn inferred_link_types_are_outside_the_causal_vocabulary() {
        // Similarity, entity, and keyword products, plus the rest of the
        // free-form v3 graph, are not declared causal kinds.
        for legacy in ["causal", "semantic", "temporal", "user_defined", "pattern"] {
            assert!(!STRATA_EDGE_VOCABULARY.contains(&legacy));
        }
        assert!(
            !STRATA_EDGE_VOCABULARY.contains(&LEGACY_INFERRED_KIND),
            "legacy_inferred is not a causal kind"
        );
        assert_eq!(STRATA_EDGE_VOCABULARY.len(), 8);
    }

    #[test]
    fn receipt_seals_and_verifies() {
        use ed25519_dalek::SigningKey;
        let body = ReceiptBody {
            record_version: RECORD_VERSION,
            source_blake3_before: "aa".repeat(32),
            source_blake3_after: "aa".repeat(32),
            schema_version: 38,
            envelope_head: String::new(),
            counts: vec![("knowledge_nodes".to_string(), 3)],
            dropped_vectors: 7,
            dropped_columns: Vec::new(),
            signing_key_id: RECEIPT_SIGNING_KEY_ID.to_string(),
        };
        let key = SigningKey::from_bytes(&[9u8; 32]);
        let receipt = MigrationReceipt::seal(body, &key);
        assert!(receipt.verify_checksum());
        assert!(receipt.verify_signature());
        // A mutated body breaks BOTH the checksum and the signature.
        let mut mutated = receipt.clone();
        mutated.body.dropped_vectors = 8;
        assert!(!mutated.verify_checksum());
        assert!(!mutated.verify_signature());
        // A signature that does not match the receipt's claimed verifying
        // key fails authorship: swap the in-log key (as a forger would) and
        // the signature no longer matches.
        let mut wrong_key = receipt.clone();
        let other = SigningKey::from_bytes(&[8u8; 32]);
        wrong_key.verifying_key = other.verifying_key().to_bytes();
        assert!(
            !wrong_key.verify_signature(),
            "wrong key must fail authorship"
        );
        assert!(
            wrong_key.verify_checksum(),
            "the checksum is key-independent"
        );
    }
}
