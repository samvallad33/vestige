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
//! | `GENESIS` / `PARAMS v4-migrate/2`| provenance + parameter frames on a fresh log   |
//! | `knowledge_nodes` rows           | `NODE` frames (kernel_id v1 + legacy UUID)      |
//! | `knowledge_nodes` FSRS columns   | `FSRS_STATE` frames, one per row with no        |
//! |                                  | `fsrs_cards` row (see FSRS state below)         |
//! | V40 `walk_receipts` rows         | reference `NODE` frames tagged migrated_from_v4 |
//! | `knowledge_nodes.superseded_by`  | `SUPERSESSION` frames                           |
//! | `memory_connections` rows        | `EDGE` frames (declared 8-type vocabulary;      |
//! |                                  | inferred types become `legacy_inferred`)        |
//! | `fsrs_cards` rows                | `FSRS_REVIEW` frames (kernel `ReviewEvent`       |
//! |                                  | plus `borsh(Option<i64>)` `reviewed_at_ms`)      |
//! | `sync_tombstones` / `deletion_tombstones` | `TOMBSTONE` frames                     |
//! | `intentions` rows                | no migration frame: decoded into [`Carryover`]  |
//! |                                  | for [`MigrateOptions::carry_over`], which admits |
//! |                                  | them as store writes after the receipt           |
//! | everything else with rows        | counted in `MigrationReport::skipped_tables`    |
//! | final frame                      | signed `MIGRATION_RECEIPT` (kind 46)            |
//!
//! Intentions are store state, not history: on Strata they only exist as
//! gate-admitted `UpsertIntentions` writes, which is what gives each row a
//! receipt. This crate cannot encode a store op (the store depends on it),
//! so it reads the rows from the same snapshot the receipt hashes and hands
//! them to the caller's `carry_over` hook, which runs on the staged log
//! before the rename. Without a hook, `intentions` stays in `skipped_tables`.
//!
//! Only frames before the receipt are migration frames. `0x20` and `0x21`
//! are shared with the store's `STORE_WRITE` and `STORE_CHECKPOINT`, so a
//! frame after the receipt (a carried intention, or any later store write)
//! is store data that the receipt never counted.
//!
//! The run re-hashes the source after the replay and refuses to seal if a
//! single byte changed (the reader is read-only at the SQLite VFS level,
//! so this is belt-and-suspenders). `strata_kernel::verify_with_head` plus
//! `StrataLog::verify_tail` must both pass before `verify_passed` is
//! reported true.
//!
//! ## FSRS fold semantics (read before relying on it)
//!
//! An `fsrs_cards` row stores only FINAL FSRS state (`reps`, `lapses`,
//! floats) — the review history that produced it is gone. Migration
//! therefore synthesizes a deterministic event series per card:
//! `reps - lapses` rating-3 (good) events followed by `lapses` rating-1
//! (again) events. The kernel fold reproduces `review_count == reps` and
//! `lapse_count == lapses` EXACTLY;
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
//! ## FSRS state from `knowledge_nodes` (read before relying on it)
//!
//! Real v3 stores leave `fsrs_cards` empty. Scheduling lives on
//! `knowledge_nodes` (`stability`, `difficulty`, `reps`, `lapses`,
//! `learning_state`, `last_accessed`, `sentiment_magnitude`), and v3 raised
//! stability on access without counting a review, so almost every row has
//! `reps = 0`. A rating series would carry nothing for those rows. Every
//! `knowledge_nodes` row without an `fsrs_cards` row therefore gets one
//! `FSRS_STATE` frame ([`FsrsStateRecord`]) carrying the card itself.
//!
//! v3 computed retrievability as `vestige_core::fsrs::retrievability_with_decay`
//! over `stability * (1 + 0.5 * sentiment_magnitude)` and the days since
//! `last_accessed`, with the store's personalized `fsrs_config.w20` (default
//! [`vestige_core::fsrs::DEFAULT_DECAY`]). Strata's pinned curve decays at a
//! different rate, so the importer refits stability: at the fit clock (the
//! latest `knowledge_nodes` timestamp in the source, or one day after the
//! row's `last_accessed` when that is later), Strata's retrievability for the
//! card equals v3's for the same row. The clock comes from the source, never
//! the wall clock, so the frame depends only on the source. After the fit,
//! the card decays on Strata's curve. `reps`, `lapses`, and `difficulty`
//! carry over, and the last review clock is `last_accessed`.
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
    EdgeRecord, FsrsStateRecord, GenesisRecord, MigrationReceipt, NodeRecord, ParamsRecord,
    ReceiptBody, SourceKey, SupersessionRecord, TombstoneRecord, KIND_FSRS_STATE,
    KIND_MIGRATION_RECEIPT, RECEIPT_SIGNING_KEY_ID, RECORD_VERSION,
};
pub use snapshot::{read_snapshot, Snapshot};

/// Parameter set implemented by this migrator. Written as the `PARAMS`
/// frame on a fresh log. `v4-migrate/2` adds `FSRS_STATE` frames.
pub const PARAMS_ID: &str = "v4-migrate/2";

/// The first parameter set. Its logs carry no `FSRS_STATE` frames.
pub const PARAMS_ID_V1: &str = "v4-migrate/1";

/// Whether a log written under `params_id` carries one `FSRS_STATE` frame
/// per `knowledge_nodes` row without an `fsrs_cards` row.
pub fn params_carry_fsrs_states(params_id: &str) -> bool {
    params_id != PARAMS_ID_V1
}

/// Algorithm version the `FSRS_STATE` stability fit targets: the version
/// strata-store folds reviews and derives retrievability under.
pub const FSRS_STATE_ALGO: u32 = strata_kernel::fsrs::ALGO_V2;

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

/// Source table whose rows ride in [`Carryover::intentions`].
pub const INTENTIONS_TABLE: &str = "intentions";
/// v3 code anchors: carried like intentions, as admitted store writes.
pub const CODE_ANCHORS_TABLE: &str = "code_memory_anchors";

/// One v3 `intentions` row, decoded from the snapshot the receipt hashes.
///
/// Field for field this is `strata_store::IntentionRecord` (this crate does
/// not depend on the store). Timestamps are unix milliseconds; v3 wrote
/// microseconds, so sub-millisecond digits are dropped. Text is verbatim:
/// statuses, trigger JSON, and scope are not normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentionRow {
    pub id: String,
    pub content: String,
    pub trigger_type: String,
    pub trigger_data: String,
    /// v3 default when NULL or absent: 2 (normal).
    pub priority: i32,
    /// v3 default when NULL or absent: `active`.
    pub status: String,
    pub created_at_ms: i64,
    pub deadline_ms: Option<i64>,
    pub fulfilled_at_ms: Option<i64>,
    pub reminder_count: i32,
    pub last_reminded_at_ms: Option<i64>,
    pub notes: Option<String>,
    pub tags: Vec<String>,
    pub related_memories: Vec<String>,
    pub snoozed_until_ms: Option<i64>,
    /// v3 default when NULL or absent: `api`.
    pub source_type: String,
    pub source_data: Option<String>,
    /// `None` for rows written before v3 schema 37 added the column.
    pub scope: Option<String>,
}

/// One v3 `code_memory_anchors` row. Field for field this is
/// `strata_store::AnchorRecord`; timestamps are unix milliseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorRow {
    pub id: String,
    pub node_id: String,
    pub file_path: String,
    pub symbol: Option<String>,
    pub symbol_kind: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    pub span_lines: Option<u32>,
    pub content_hash: Option<String>,
    pub captured_at_ms: i64,
    pub last_verified_at_ms: Option<i64>,
    pub last_status: Option<String>,
}

/// v3 rows the migration log has no frame kind for, in source row order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Carryover {
    /// Every `intentions` row.
    pub intentions: Vec<IntentionRow>,
    /// Every `code_memory_anchors` row. The hook admits the ones whose
    /// memory is live on the staged log.
    pub anchors: Vec<AnchorRow>,
}

/// Admits a [`Carryover`] into the staged log and returns how many
/// intention rows the staged log now holds. Runs after
/// [`MigrateOptions::before_publish`] passes and before the rename. A count
/// short of `carryover.intentions.len()`, or an error, deletes staging and
/// leaves the destination untouched.
pub type CarryOverHook = Box<dyn FnOnce(&Path, &Carryover) -> Result<u64, String>>;

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
    /// Receives the source's `intentions` rows after `before_publish`
    /// passes, on the same staging directory. See [`CarryOverHook`].
    /// `None` leaves `intentions` in `skipped_tables`.
    pub carry_over: Option<CarryOverHook>,
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
            .field("carry_over", &self.carry_over.as_ref().map(|_| "Some"))
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
    /// `intentions` rows the `carry_over` hook admitted into the log (0 when
    /// no hook ran).
    pub intentions_carried: u64,
    /// Source tables that contained rows but have no STRATA mapping.
    /// `intentions` is listed unless `carry_over` admitted every row.
    pub skipped_tables: Vec<String>,
    /// Whether kernel replay verification AND log tail verification both
    /// passed over the finished log.
    pub verify_passed: bool,
    /// `node_embeddings` rows whose vector values were never read.
    pub dropped_vectors: u64,
    /// FSRS_STATE frames: `knowledge_nodes` rows with no `fsrs_cards` row.
    pub fsrs_states: u64,
    /// `memory_connections` rows whose source or target memory no longer
    /// exists (left by older v3 builds). Skipped, not imported.
    #[serde(default)]
    pub skipped_dangling_edges: u64,
    /// `fsrs_cards` rows whose memory no longer exists. Skipped, not imported.
    #[serde(default)]
    pub skipped_dangling_cards: u64,
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
    // Snapshot copies live in the data directory beside `strata_dir`, owner
    // only, and are removed when this function returns.
    let scratch_parent = match strata_dir.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut scratch = source::Scratch::new(scratch_parent);
    let (files, _snapshotted) =
        source::prepare_source(source, options.accept_wal_snapshot, &mut scratch)?;

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
            Hooks {
                before_import: options.before_import,
                before_publish: options.before_publish,
                carry_over: options.carry_over,
            },
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
        Hooks {
            before_import: options.before_import,
            before_publish: options.before_publish,
            carry_over: options.carry_over,
        },
    )
}

/// The caller's staging hooks, moved into the lock holder.
#[cfg(feature = "sqlite-reader")]
struct Hooks {
    before_import: Option<BeforePublish>,
    before_publish: Option<BeforePublish>,
    carry_over: Option<CarryOverHook>,
}

/// Appended to the destination directory's file name. The first-launch
/// destination is `<data-dir>/log`, so its staging directory is
/// `<data-dir>/log.strata-staging`.
pub const STAGING_SUFFIX: &str = ".strata-staging";

/// How often a process that finds the staging lock held checks again.
#[cfg(feature = "sqlite-reader")]
const STAGING_LOCK_POLL: Duration = Duration::from_millis(20);

/// Sibling of `dest`: `<dest>`'s file name plus [`STAGING_SUFFIX`].
#[cfg(feature = "sqlite-reader")]
fn staging_path(dest: &Path) -> std::path::PathBuf {
    let name = dest.file_name().unwrap_or(std::ffi::OsStr::new("strata"));
    let mut staging_name = name.to_os_string();
    staging_name.push(STAGING_SUFFIX);
    dest.with_file_name(staging_name)
}

/// The staging lock: a dotfile beside the staging directory, `.<staging>.lock`.
/// Beside it, not inside: Windows cannot rename a directory while a file in it
/// is open, and the lock stays held until staging is renamed onto `dest`.
#[cfg(feature = "sqlite-reader")]
fn staging_lock_path(staging: &Path) -> std::path::PathBuf {
    let name = staging
        .file_name()
        .unwrap_or(std::ffi::OsStr::new("strata-staging"));
    let mut lock_name = std::ffi::OsString::from(".");
    lock_name.push(name);
    lock_name.push(".lock");
    staging.with_file_name(lock_name)
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
            if let Ok(receipt) = records::decode_receipt(&frame.payload)
                && receipt.body.source_blake3_before == source_blake3
            {
                return Some(receipt);
            }
        }
    }
    None
}

/// Import into `dest` via a sibling staging directory, then rename.
///
/// `allow_idempotent` is set for SQLite sources, whose receipt is keyed by
/// the source BLAKE3. Nothing is written to `dest` until every pre-check
/// has passed. The winner is whoever takes the staging lock
/// ([`staging_lock_path`], `File::try_lock`); the kernel drops that lock
/// when the process dies, including SIGKILL. The holder wipes any staging
/// directory an earlier importer left and starts over. A process that finds
/// the lock held waits for it, then sees `dest` published or takes over.
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
    hooks: Hooks,
) -> Result<MigrationReport, MigrationError> {
    let staging = staging_path(dest);
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_path = staging_lock_path(&staging);
    loop {
        if destination_occupied(dest)? {
            if allow_idempotent
                && let Some(receipt) = receipt_matching(dest, source_blake3)
            {
                return Ok(idempotent_report(&snapshot, &receipt, started));
            }
            return Err(MigrationError::DestinationNotEmpty {
                path: dest.display().to_string(),
            });
        }
        let Some(lock) = try_staging_lock(&lock_path)? else {
            // Another importer holds it. It publishes `dest` (checked at the
            // top of the loop) or dies, and the kernel frees the lock.
            std::thread::sleep(STAGING_LOCK_POLL);
            continue;
        };
        // Held now, so no importer is running: `dest` may have been
        // published while this process waited, and any staging directory
        // is an earlier importer's leftover.
        if destination_occupied(dest)? {
            drop(lock);
            continue;
        }
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        std::fs::create_dir(&staging)?;
        return import_holding_lock(
            lock,
            &staging,
            dest,
            seed,
            source_blake3,
            snapshot,
            started,
            files,
            hooks,
        );
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
    hooks: Hooks,
) -> Result<MigrationReport, MigrationError> {
    let Hooks {
        before_import,
        before_publish,
        carry_over,
    } = hooks;
    // Decoded before the backup and the import, so a corrupt row stops the
    // run early. Nobody reads the rows without a hook.
    let carryover = if carry_over.is_some() {
        match extract_carryover(&snapshot.archive) {
            Ok(carryover) => carryover,
            Err(err) => return Err(discard_staging(lock, staging, err)),
        }
    } else {
        Carryover::default()
    };
    if let Some(hook) = before_import
        && let Err(detail) = hook(staging)
    {
        return Err(discard_staging(
            lock,
            staging,
            MigrationError::Strata(detail),
        ));
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

    let mut report = match finish(log, staging, outcome, snapshot, &sealed_hash, started) {
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
    if let Some(hook) = before_publish
        && let Err(detail) = hook(staging)
    {
        return Err(discard_staging(
            lock,
            staging,
            MigrationError::Strata(detail),
        ));
    }
    if let Some(hook) = carry_over {
        let expected = carryover.intentions.len() as u64;
        match hook(staging, &carryover) {
            Ok(carried) if carried == expected => {
                report.intentions_carried = carried;
                report
                    .skipped_tables
                    .retain(|table| table != INTENTIONS_TABLE && table != CODE_ANCHORS_TABLE);
            }
            Ok(carried) => {
                return Err(discard_staging(
                    lock,
                    staging,
                    MigrationError::Strata(format!(
                        "carry-over admitted {carried} of {expected} intentions"
                    )),
                ));
            }
            Err(detail) => {
                return Err(discard_staging(
                    lock,
                    staging,
                    MigrationError::Strata(detail),
                ));
            }
        }
    }
    if let Err(err) = publish(staging, dest) {
        return Err(discard_staging(lock, staging, err));
    }
    // The lock file stays: deleting it would let one waiter lock the old
    // file while another creates and locks a new one.
    drop(lock);
    Ok(report)
}

#[cfg(feature = "sqlite-reader")]
fn discard_staging(lock: std::fs::File, staging: &Path, err: MigrationError) -> MigrationError {
    let _ = std::fs::remove_dir_all(staging);
    drop(lock);
    err
}

/// Take the staging lock without waiting. `None` when another process holds it.
#[cfg(feature = "sqlite-reader")]
fn try_staging_lock(path: &Path) -> Result<Option<std::fs::File>, MigrationError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(err)) => Err(err.into()),
    }
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
    let dangling = dangling_rows(&snapshot.archive);
    MigrationReport {
        nodes: table_rows(snapshot, "knowledge_nodes"),
        edges: table_rows(snapshot, "memory_connections") - dangling.edges,
        fsrs_events: table_rows(snapshot, "fsrs_cards") - dangling.cards,
        intentions_carried: 0,
        skipped_tables: skipped_tables_for(snapshot),
        verify_passed: true,
        dropped_vectors: snapshot.dropped_vectors,
        fsrs_states: planned_fsrs_states(&snapshot.archive),
        skipped_dangling_edges: dangling.edges,
        skipped_dangling_cards: dangling.cards,
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
            getrandom::fill(&mut seed).map_err(|e| {
                MigrationError::Io(std::io::Error::other(format!(
                    "no OS entropy for the log signing seed: {e}"
                )))
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
    let dangling = dangling_rows(&snapshot.archive);
    MigrationReport {
        nodes: table_rows(snapshot, "knowledge_nodes"),
        edges: table_rows(snapshot, "memory_connections") - dangling.edges,
        fsrs_events: table_rows(snapshot, "fsrs_cards") - dangling.cards,
        intentions_carried: 0,
        skipped_tables: skipped_tables_for(snapshot),
        // A dry run writes nothing; the envelope chain is verified during
        // the read, so reaching this point means the chain held.
        verify_passed: true,
        dropped_vectors: snapshot.dropped_vectors,
        fsrs_states: planned_fsrs_states(&snapshot.archive),
        skipped_dangling_edges: dangling.edges,
        skipped_dangling_cards: dangling.cards,
        envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
        source_blake3: source_blake3.to_string(),
        receipt_digest: None,
        receipt_verified: false,
        idempotent_reuse: false,
        duration: started.elapsed(),
    }
}

/// Rows the importer skips because the memory they point at is gone.
#[cfg(feature = "sqlite-reader")]
#[derive(Default)]
struct DanglingRows {
    edges: u64,
    cards: u64,
}

/// Count the dangling `memory_connections` and `fsrs_cards` rows of an
/// archive without building records (dry-run and idempotent reports).
#[cfg(feature = "sqlite-reader")]
fn dangling_rows(archive: &PortableArchive) -> DanglingRows {
    let mut out = DanglingRows::default();
    let ids: std::collections::HashSet<&str> = source::table(archive, "knowledge_nodes")
        .map(|table| {
            (0..table.rows.len())
                .filter_map(|i| source::Row::new(table, i).text("id").ok())
                .collect()
        })
        .unwrap_or_default();
    if let Some(table) = source::table(archive, "memory_connections") {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            let live = |column: &str| row.text(column).is_ok_and(|id| ids.contains(id));
            if !(live("source_id") && live("target_id")) {
                out.edges += 1;
            }
        }
    }
    if let Some(table) = source::table(archive, "fsrs_cards") {
        for index in 0..table.rows.len() {
            let row = source::Row::new(table, index);
            if !row.text("memory_id").is_ok_and(|id| ids.contains(id)) {
                out.cards += 1;
            }
        }
    }
    out
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
    // a STRATA mapping is named, so the receipt never overclaims. A table
    // whose COUNT(*) failed is still named (`None` count): dropping the
    // error used to erase it from this list.
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
    fsrs_states: u64,
    skipped_dangling_edges: u64,
    skipped_dangling_cards: u64,
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
    let (edge_records, skipped_dangling_edges) = extract_edges(archive, &kernel_ids)?;
    let tombstones = extract_tombstones(archive)?;
    let walk_nodes = extract_walk_receipts(snapshot, &kernel_ids)?;
    attach_fsrs_legacy(archive, &kernel_ids, &mut node_records)?;

    let mut nodes = 0u64;
    let mut edges = 0u64;
    let mut fsrs_events = 0u64;
    let mut skipped_dangling_cards = 0u64;

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
            // A card whose memory was deleted by an older v3 build has
            // nothing to attach to: skip it and count it.
            let Some(kernel_id) = kernel_ids.get(&memory_id).copied() else {
                skipped_dangling_cards += 1;
                continue;
            };
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

    // ---- knowledge_nodes scheduling columns -> imported card states ------
    // Rows with an fsrs_cards row already have a rating series above.
    let mut fsrs_states = 0u64;
    for record in extract_fsrs_states(archive, &node_records)? {
        writer.push(records::KIND_FSRS_STATE, borsh::to_vec(&record))?;
        fsrs_states += 1;
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
        fsrs_states,
        skipped_dangling_edges,
        skipped_dangling_cards,
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
        intentions_carried: 0,
        skipped_tables: skipped_tables_for(&snapshot),
        verify_passed,
        dropped_vectors: snapshot.dropped_vectors,
        fsrs_states: outcome.fsrs_states,
        skipped_dangling_edges: outcome.skipped_dangling_edges,
        skipped_dangling_cards: outcome.skipped_dangling_cards,
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

/// One day in unix milliseconds (Strata measures review age in whole days).
#[cfg(feature = "sqlite-reader")]
const DAY_MS: i64 = 86_400_000;

/// v3 `apply_decay` stretched stability for emotional memories:
/// `stability * (1 + sentiment_magnitude * 0.5)`.
#[cfg(feature = "sqlite-reader")]
const V3_SENTIMENT_STABILITY_BOOST: f64 = 0.5;

/// Memory ids that have an `fsrs_cards` row (and so a rating series).
#[cfg(feature = "sqlite-reader")]
fn carded_memory_ids(
    archive: &PortableArchive,
) -> Result<std::collections::HashSet<String>, MigrationError> {
    let mut ids = std::collections::HashSet::new();
    if let Some(table) = source::table(archive, "fsrs_cards") {
        for index in 0..table.rows.len() {
            ids.insert(
                source::Row::new(table, index)
                    .text("memory_id")?
                    .to_string(),
            );
        }
    }
    Ok(ids)
}

/// `FSRS_STATE` frames a run over this archive appends: `knowledge_nodes`
/// rows without an `fsrs_cards` row. Used by the dry-run and idempotent
/// reports, which write nothing.
#[cfg(feature = "sqlite-reader")]
fn planned_fsrs_states(archive: &PortableArchive) -> u64 {
    let Some(table) = source::table(archive, "knowledge_nodes") else {
        return 0;
    };
    let carded = carded_memory_ids(archive).unwrap_or_default();
    (0..table.rows.len())
        .filter(|index| {
            source::Row::new(table, *index)
                .text("id")
                .is_ok_and(|id| !carded.contains(id))
        })
        .count() as u64
}

/// v3's forgetting-curve decay: `fsrs_config.w20` when set to a positive
/// finite number, else `vestige_core::fsrs::DEFAULT_DECAY` (what v3's
/// `apply_decay` fell back to).
#[cfg(feature = "sqlite-reader")]
fn v3_decay(archive: &PortableArchive) -> f64 {
    let default = vestige_core::fsrs::DEFAULT_DECAY;
    let Some(table) = source::table(archive, "fsrs_config") else {
        return default;
    };
    (0..table.rows.len())
        .map(|index| source::Row::new(table, index))
        .find(|row| row.text("key").is_ok_and(|key| key == "w20"))
        .and_then(|row| row.get("value").ok().and_then(portable_f64))
        .filter(|w20| w20.is_finite() && *w20 > 0.0)
        .unwrap_or(default)
}

/// A numeric SQLite value. v3 columns are dynamically typed, so a numeric
/// string also counts. NULL, blobs, and other text are `None`.
#[cfg(feature = "sqlite-reader")]
fn portable_f64(value: &PortableValue) -> Option<f64> {
    match value {
        PortableValue::Real(v) => Some(*v),
        PortableValue::Integer(v) => Some(*v as f64),
        PortableValue::Text(text) => text.trim().parse().ok(),
        PortableValue::Null | PortableValue::Blob(_) => None,
    }
}

/// Scheduling column as `f64`. A missing column, NULL, non-number, or
/// non-finite value reads as v3's column default: these are scheduling
/// metadata, and one bad float should not stop the whole upgrade.
#[cfg(feature = "sqlite-reader")]
fn fsrs_column(row: &source::Row<'_>, name: &str, default: f64) -> f64 {
    if !row.columns().iter().any(|column| column == name) {
        return default;
    }
    row.get(name)
        .ok()
        .and_then(portable_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

/// v3 `learning_state` as a kernel phase. The kernel has no `new` phase;
/// a never-reviewed card is `Learning`.
#[cfg(feature = "sqlite-reader")]
fn v3_phase(row: &source::Row<'_>) -> strata_kernel::fsrs::CardPhase {
    use strata_kernel::fsrs::CardPhase;
    let state = if row.columns().iter().any(|c| c == "learning_state") {
        row.opt_text("learning_state").ok().flatten().unwrap_or("")
    } else {
        ""
    };
    match state.trim().to_ascii_lowercase().as_str() {
        "review" => CardPhase::Review,
        "relearning" => CardPhase::Relearning,
        _ => CardPhase::Learning,
    }
}

/// One `FSRS_STATE` record per `knowledge_nodes` row without an
/// `fsrs_cards` row, in source row order. See the crate docs.
///
/// `node_records[i]` must be row `i` of `knowledge_nodes`, as
/// [`extract_nodes`] builds it.
#[cfg(feature = "sqlite-reader")]
fn extract_fsrs_states(
    archive: &PortableArchive,
    node_records: &[NodeRecord],
) -> Result<Vec<FsrsStateRecord>, MigrationError> {
    use strata_kernel::canonical::to_q32_32;
    use strata_kernel::fsrs::{FsrsFold, D_MAX, D_MIN};

    let Some(table) = source::table(archive, "knowledge_nodes") else {
        return Ok(Vec::new());
    };
    if node_records.len() != table.rows.len() {
        return Err(MigrationError::Corrupt(format!(
            "knowledge_nodes has {} rows but {} node records",
            table.rows.len(),
            node_records.len()
        )));
    }
    let carded = carded_memory_ids(archive)?;
    let w20 = v3_decay(archive);
    // The source's own "now": its latest node timestamp, never later than
    // the moment of import (a future-dated row must not move every card's
    // fit point).
    let import_clock_ms = chrono::Utc::now().timestamp_millis();
    let source_clock = node_records
        .iter()
        .map(|node| {
            node.created_ms
                .max(node.updated_ms)
                .max(node.last_accessed_ms)
        })
        .max()
        .unwrap_or(0)
        .min(import_clock_ms);

    let mut records = Vec::new();
    for (index, node) in node_records.iter().enumerate() {
        let row = source::Row::new(table, index);
        if row.text("id")? != node.legacy_id {
            return Err(MigrationError::Corrupt(format!(
                "knowledge_nodes row {index} is not node {}",
                node.legacy_id
            )));
        }
        if carded.contains(&node.legacy_id) {
            continue;
        }
        let stability = fsrs_column(&row, "stability", 1.0);
        let difficulty = fsrs_column(&row, "difficulty", 5.0);
        let sentiment = fsrs_column(&row, "sentiment_magnitude", 0.0);
        let reps = (fsrs_column(&row, "reps", 0.0) as i64).clamp(0, i64::from(u32::MAX));
        let lapses = (fsrs_column(&row, "lapses", 0.0) as i64).clamp(0, reps);

        // Fit at the source clock, or one day after the last access when
        // that is later: Strata measures whole days, and a card younger
        // than a day reads 1.0 whatever its stability.
        let reviewed_at_ms = node.last_accessed_ms;
        let fitted_at_ms = source_clock.max(reviewed_at_ms.saturating_add(DAY_MS));
        let elapsed_ms = fitted_at_ms.saturating_sub(reviewed_at_ms);
        // v3: (now - last_accessed).num_seconds() / 86400.
        let v3_days = (elapsed_ms / 1000) as f64 / 86_400.0;
        let v3_retrievability = vestige_core::fsrs::retrievability_with_decay(
            stability * (1.0 + sentiment * V3_SENTIMENT_STABILITY_BOOST),
            v3_days,
            w20,
        );
        let whole_days = FsrsFold::elapsed_review_days(reviewed_at_ms, fitted_at_ms);
        let fitted =
            FsrsFold::stability_for_retrievability(v3_retrievability, whole_days, FSRS_STATE_ALGO)
                .map_err(|e| MigrationError::Kernel(e.to_string()))?;

        records.push(FsrsStateRecord {
            record_version: RECORD_VERSION,
            kernel_id: node.kernel_id,
            legacy_id: node.legacy_id.clone(),
            algo_version: FSRS_STATE_ALGO,
            stability_q: to_q32_32(fitted),
            difficulty_q: to_q32_32(difficulty.clamp(D_MIN, D_MAX)),
            review_count: reps as u32,
            lapse_count: lapses as u32,
            phase: v3_phase(&row),
            reviewed_at_ms,
            v3_retrievability_q: to_q32_32(v3_retrievability),
            v3_decay_q: to_q32_32(w20),
            fitted_at_ms,
        });
    }
    Ok(records)
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

/// Decode `memory_connections` into edge records. Older v3 builds could
/// leave rows whose source or target memory was deleted; such an edge has no
/// node to attach to, so it is skipped and counted (returned second), and
/// the report names the count.
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
) -> Result<(Vec<EdgeRecord>, u64), MigrationError> {
    let Some(table) = source::table(archive, "memory_connections") else {
        return Ok((Vec::new(), 0));
    };
    let mut records = Vec::with_capacity(table.rows.len());
    let mut skipped = 0u64;
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let source_legacy_id = row.text("source_id")?.to_string();
        let target_legacy_id = row.text("target_id")?.to_string();
        let (Some(&source_kernel_id), Some(&target_kernel_id)) = (
            kernel_ids.get(&source_legacy_id),
            kernel_ids.get(&target_legacy_id),
        ) else {
            skipped += 1;
            continue;
        };
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
    Ok((records, skipped))
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

/// Decode `intentions` into [`Carryover`] rows, source row order.
///
/// Mirrors v3's `row_to_intention`: a column older schemas lack reads as its
/// v3 default, and `tags` / `related_memories` are tolerant JSON arrays.
/// Unlike v3, a non-empty timestamp that does not parse is corrupt instead
/// of becoming "now" or `None`, and the run stops.
#[cfg(feature = "sqlite-reader")]
fn extract_carryover(archive: &PortableArchive) -> Result<Carryover, MigrationError> {
    Ok(Carryover {
        intentions: extract_intention_rows(archive)?,
        anchors: extract_anchor_rows(archive)?,
    })
}

/// Decode `code_memory_anchors` rows, source row order. A missing table (v3
/// before code anchors) is no rows.
#[cfg(feature = "sqlite-reader")]
fn extract_anchor_rows(archive: &PortableArchive) -> Result<Vec<AnchorRow>, MigrationError> {
    let Some(table) = source::table(archive, CODE_ANCHORS_TABLE) else {
        return Ok(Vec::new());
    };
    let has = |name: &str| table.columns.iter().any(|column| column == name);
    let mut anchors = Vec::with_capacity(table.rows.len());
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let opt_text = |name: &str| -> Result<Option<String>, MigrationError> {
            if !has(name) {
                return Ok(None);
            }
            Ok(row.opt_text(name)?.map(str::to_string))
        };
        let opt_line = |name: &str| -> Result<Option<u32>, MigrationError> {
            if !has(name) {
                return Ok(None);
            }
            match row.integer_or(name, -1)? {
                -1 => Ok(None),
                value => u32::try_from(value).map(Some).map_err(|_| {
                    MigrationError::Corrupt(format!(
                        "code_memory_anchors row {index} column {name}: {value} is out of range"
                    ))
                }),
            }
        };
        let opt_ms = |name: &str| -> Result<Option<i64>, MigrationError> {
            match opt_text(name)? {
                Some(raw) if !raw.is_empty() => Ok(Some(source::timestamp_ms(&raw)?)),
                _ => Ok(None),
            }
        };
        anchors.push(AnchorRow {
            id: row.text("id")?.to_string(),
            node_id: row.text("node_id")?.to_string(),
            file_path: row.text("file_path")?.to_string(),
            symbol: opt_text("symbol")?,
            symbol_kind: opt_text("symbol_kind")?,
            start_line: opt_line("start_line")?,
            end_line: opt_line("end_line")?,
            span_lines: opt_line("span_lines")?,
            content_hash: opt_text("content_hash")?,
            captured_at_ms: source::timestamp_ms(row.text("captured_at")?)?,
            last_verified_at_ms: opt_ms("last_verified_at")?,
            last_status: opt_text("last_status")?,
        });
    }
    Ok(anchors)
}

#[cfg(feature = "sqlite-reader")]
fn extract_intention_rows(archive: &PortableArchive) -> Result<Vec<IntentionRow>, MigrationError> {
    let Some(table) = source::table(archive, INTENTIONS_TABLE) else {
        return Ok(Vec::new());
    };
    let has = |name: &str| table.columns.iter().any(|column| column == name);
    let mut intentions = Vec::with_capacity(table.rows.len());
    for index in 0..table.rows.len() {
        let row = source::Row::new(table, index);
        let opt_text = |name: &str| -> Result<Option<String>, MigrationError> {
            if !has(name) {
                return Ok(None);
            }
            Ok(row.opt_text(name)?.map(str::to_string))
        };
        let opt_ms = |name: &str| -> Result<Option<i64>, MigrationError> {
            match opt_text(name)? {
                Some(raw) if !raw.is_empty() => Ok(Some(source::timestamp_ms(&raw)?)),
                _ => Ok(None),
            }
        };
        let int32 = |name: &str, default: i64| -> Result<i32, MigrationError> {
            let value = if has(name) {
                row.integer_or(name, default)?
            } else {
                default
            };
            i32::try_from(value).map_err(|_| {
                MigrationError::Corrupt(format!(
                    "intentions row {index} column {name}: {value} is out of range"
                ))
            })
        };
        intentions.push(IntentionRow {
            id: row.text("id")?.to_string(),
            content: row.text("content")?.to_string(),
            trigger_type: row.text("trigger_type")?.to_string(),
            trigger_data: row.text("trigger_data")?.to_string(),
            priority: int32("priority", 2)?,
            status: opt_text("status")?.unwrap_or_else(|| "active".to_string()),
            created_at_ms: source::timestamp_ms(row.text("created_at")?)?,
            deadline_ms: opt_ms("deadline")?,
            fulfilled_at_ms: opt_ms("fulfilled_at")?,
            reminder_count: int32("reminder_count", 0)?,
            last_reminded_at_ms: opt_ms("last_reminded_at")?,
            notes: opt_text("notes")?,
            tags: source::parse_tags(opt_text("tags")?.as_deref()),
            related_memories: source::parse_tags(opt_text("related_memories")?.as_deref()),
            snoozed_until_ms: opt_ms("snoozed_until")?,
            source_type: opt_text("source_type")?.unwrap_or_else(|| "api".to_string()),
            source_data: opt_text("source_data")?,
            scope: opt_text("scope")?,
        });
    }
    Ok(intentions)
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

    #[cfg(feature = "sqlite-reader")]
    fn portable_table(
        name: &str,
        columns: &[&str],
        rows: Vec<Vec<PortableValue>>,
    ) -> PortableArchive {
        PortableArchive {
            archive_format: source::PORTABLE_ARCHIVE_FORMAT.to_string(),
            vestige_version: "3.1.1".to_string(),
            schema_version: 40,
            exported_at: chrono::Utc::now(),
            mode: "exact".to_string(),
            tables: vec![vestige_core::storage::PortableTable {
                name: name.to_string(),
                columns: columns.iter().map(|c| c.to_string()).collect(),
                rows,
            }],
        }
    }

    #[cfg(feature = "sqlite-reader")]
    #[test]
    fn scheduling_columns_read_tolerantly_with_v3_defaults() {
        use PortableValue::{Blob, Integer, Null, Real, Text};
        let archive = portable_table(
            "knowledge_nodes",
            &["stability", "reps", "learning_state"],
            vec![
                vec![Real(2.5), Integer(4), Text("review".into())],
                vec![Text(" 7.25 ".into()), Real(2.0), Text("Relearning".into())],
                vec![Null, Null, Null],
                vec![Real(f64::NAN), Text("x".into()), Text("new".into())],
                vec![Blob("00".into()), Integer(1), Text("learning".into())],
            ],
        );
        let table = &archive.tables[0];
        let read = |index: usize| {
            let row = source::Row::new(table, index);
            (
                fsrs_column(&row, "stability", 1.0),
                fsrs_column(&row, "reps", 0.0),
                fsrs_column(&row, "difficulty", 5.0),
                v3_phase(&row),
            )
        };
        use strata_kernel::fsrs::CardPhase::{Learning, Relearning, Review};
        assert_eq!(read(0), (2.5, 4.0, 5.0, Review));
        assert_eq!(read(1), (7.25, 2.0, 5.0, Relearning));
        assert_eq!(read(2), (1.0, 0.0, 5.0, Learning));
        assert_eq!(read(3), (1.0, 0.0, 5.0, Learning));
        assert_eq!(read(4), (1.0, 1.0, 5.0, Learning));
    }

    #[cfg(feature = "sqlite-reader")]
    #[test]
    fn v3_decay_reads_w20_or_falls_back_to_the_v3_default() {
        use PortableValue::{Real, Text};
        let config = |value: PortableValue| {
            portable_table(
                "fsrs_config",
                &["key", "value", "updated_at"],
                vec![
                    vec![Text("w17".into()), Real(9.0), Text(String::new())],
                    vec![Text("w20".into()), value, Text(String::new())],
                ],
            )
        };
        assert_eq!(v3_decay(&config(Real(0.0803))), 0.0803);
        assert_eq!(v3_decay(&config(Text("0.2".into()))), 0.2);
        let default = vestige_core::fsrs::DEFAULT_DECAY;
        assert_eq!(v3_decay(&config(Real(0.0))), default);
        assert_eq!(v3_decay(&config(Real(-1.0))), default);
        assert_eq!(v3_decay(&config(PortableValue::Null)), default);
        assert_eq!(v3_decay(&portable_table("other", &[], Vec::new())), default);
    }

    #[test]
    fn params_v1_logs_carry_no_fsrs_states() {
        assert!(!params_carry_fsrs_states(PARAMS_ID_V1));
        assert!(params_carry_fsrs_states(PARAMS_ID));
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
