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
//! | `memory_connections` rows        | `EDGE` frames for the 8-type vocabulary;        |
//! |                                  | lookalike legacy types become                   |
//! |                                  | `derived_from{legacy_inferred=1}`;              |
//! |                                  | `semantic` / `similarity` are `LEGACY_LINK`     |
//! |                                  | frames and are not causal edges                 |
//! | `fsrs_cards` rows                | `FSRS_REVIEW` frames (kernel `ReviewEvent`)     |
//! | `sync_tombstones` / `deletion_tombstones` | `TOMBSTONE` frames                     |
//! | everything else with rows        | counted in `MigrationReport::skipped_tables`    |
//! | final frame                      | signed `MIGRATION_RECEIPT` (kind 46)            |
//!
//! The run re-hashes the source after the replay and refuses to seal if a
//! single byte changed (the reader is read-only at the SQLite VFS level,
//! so this is belt-and-suspenders). Sealing rolls an empty active segment,
//! so verification re-checks every sealed segment's frames and trailer,
//! replays the log, and compares per-kind counts to the MIGRATION_RECEIPT.
//! `frames_verified` must equal the total frame count; any mismatch is a
//! hard error.
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
//! ## Determinism
//!
//! Every timestamp in the log comes from the source rows (a replay clock:
//! the migration never reads the wall clock into hashed state), the log is
//! opened seeded from the source BLAKE3 unless a seed is pinned, and all
//! hashed collections are ordered Vecs. Two runs over one source produce
//! byte-identical segments.

pub mod records;
pub mod snapshot;
pub mod source;
mod verify;

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use strata::StrataLog;
use strata_kernel::checkpoint::{checkpoint_hash, Checkpoint};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::ALGO_V1;
use strata_kernel::kernel::Kernel;
use vestige_core::storage::PortableArchive;
use vestige_core::storage::PortableValue;

pub use records::{
    EdgeRecord, GenesisRecord, MigrationReceipt, NodeRecord, ParamsRecord, ReceiptBody,
    SupersessionRecord, TombstoneRecord, KIND_MIGRATION_RECEIPT, RECEIPT_SIGNING_KEY_ID,
    RECORD_VERSION,
};
pub use snapshot::{read_snapshot, Snapshot};
pub use verify::{verify_migrated_dir, LogVerify};

/// Private log key, stored next to `--to` rather than inside it.
pub fn log_signing_key_path(strata_dir: &Path) -> std::path::PathBuf {
    let parent = strata_dir.parent().unwrap_or_else(|| Path::new("."));
    let mut name = strata_dir.file_name().unwrap_or_default().to_os_string();
    name.push(".strata.key");
    parent.join(name)
}

/// Copies the external log key into `dir` for `StrataLog::open`, and removes
/// that copy when dropped.
pub struct StagedLogKey {
    staged: Option<std::path::PathBuf>,
}

impl Drop for StagedLogKey {
    fn drop(&mut self) {
        if let Some(path) = self.staged.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Place the external log key where `StrataLog::open` expects it.
pub fn stage_log_key(dir: &Path) -> Result<StagedLogKey, MigrationError> {
    let internal = dir.join("strata.key");
    if internal.exists() {
        return Ok(StagedLogKey { staged: None });
    }
    let external = log_signing_key_path(dir);
    std::fs::copy(&external, &internal).map_err(|e| {
        MigrationError::Strata(format!("log signing key {}: {e}", external.display()))
    })?;
    set_mode_0600(&internal)?;
    Ok(StagedLogKey {
        staged: Some(internal),
    })
}

fn set_mode_0600(path: &Path) -> Result<(), MigrationError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Open a migrated log, staging the external key for the duration.
pub struct OpenedLog {
    /// The opened log. Dropped before the staged key file is removed.
    pub log: StrataLog,
    _key: StagedLogKey,
}

/// Open `dir` using the key stored outside it.
pub fn open_migrated(dir: &Path) -> Result<OpenedLog, MigrationError> {
    let key = stage_log_key(dir)?;
    let log = StrataLog::open(dir).map_err(|e| MigrationError::Strata(e.to_string()))?;
    Ok(OpenedLog { log, _key: key })
}

/// Parameter set implemented by this migrator. Written as the `PARAMS`
/// frame on a fresh log.
pub const PARAMS_ID: &str = "v4-migrate/1";

/// Frames per `append_batch` call: bounds peak memory on huge stores while
/// staying far above the log's own 64-frame group-commit cap.
const BATCH_FRAMES: usize = 1024;

/// Tables emitted as their own frames. Every other nonempty table is named
/// in `skipped_tables`. `walk_receipts` becomes NODE frames.
const EMITTED_TABLES: &[&str] = &[
    "knowledge_nodes",
    "memory_connections",
    "fsrs_cards",
    "sync_tombstones",
    "deletion_tombstones",
    "walk_receipts",
];

/// The only edge vocabulary STRATA carries (H4). A legacy `link_type`
/// outside this set migrates as `derived_from` with `legacy_inferred = 1`,
/// except association rows ([`is_noncausal_association`]), which are not
/// edges at all.
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

/// v3 association rows. They must not become EDGE frames: a causal walk
/// and Backfill only follow `KIND_EDGE`.
pub fn is_noncausal_association(link_type: &str) -> bool {
    link_type.eq_ignore_ascii_case("semantic") || link_type.eq_ignore_ascii_case("similarity")
}

/// Options for one migration run.
#[derive(Debug, Clone, Default)]
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
    /// A killed run left files in `--to`. `--rollback` removes them.
    #[error(
        "destination {path} is a partial migration; rerun with --rollback to remove it before migrating again"
    )]
    DestinationPartial {
        /// The refusing destination directory.
        path: String,
    },
    /// `--to` already holds a sealed MIGRATION_RECEIPT. It is not extended.
    #[error("destination {path} already holds a completed migration; refusing to extend it")]
    DestinationComplete {
        /// The refusing destination directory.
        path: String,
    },
    /// `--rollback` does not delete a finished log.
    #[error("refusing to roll back completed destination {path}")]
    RollbackRefused {
        /// The destination that still holds a receipt.
        path: String,
    },
    /// The source's BLAKE3 changed while frames were being appended. The
    /// log is incomplete; the destination is poisoned (further runs refuse).
    #[error("source changed during migration (before {before}, after {after}); the destination log is incomplete and must not be trusted")]
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
    /// Sealed-segment, replay, or receipt-count verification failed.
    #[error("migration verification failed: {0}")]
    Verify(String),
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
    /// Whether sealed-segment, replay, and receipt-count verification passed.
    pub verify_passed: bool,
    /// Frames covered by sealed-segment frame and trailer checks.
    pub frames_verified: u64,
    /// Frames replayed from the log. Equals `frames_verified` when verification passed.
    pub frames_total: u64,
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
    /// Always false. A destination that already holds a receipt is refused
    /// (`DestinationComplete`) instead of being reused.
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
pub fn migrate(source: &Path, strata_dir: &Path) -> Result<MigrationReport, MigrationError> {
    migrate_with_options(source, strata_dir, MigrateOptions::default())
}

/// Migrate with explicit options. See [`MigrateOptions`].
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
        refuse_occupied(strata_dir)?;
        let (log, verifying_key) = open_log(strata_dir, options.seed)?;
        let outcome = migrate_snapshot_into(&snapshot, &log, "", verifying_key)?;
        return finish(log, strata_dir, outcome, snapshot, "", started);
    }

    // ---- hash BEFORE any SQL read --------------------------------------
    let blake3_before = files.blake3_hex()?;

    // ---- read-only snapshot (verifies the envelope chain) ---------------
    let snapshot = source::load_snapshot(&files)?;

    if options.dry_run {
        return Ok(dry_run_report(&snapshot, &blake3_before, started));
    }

    // Empty destination: proceed. A partial destination names --rollback.
    // A completed destination is refused; it is not silently reused.
    std::fs::create_dir_all(strata_dir)?;
    refuse_occupied(strata_dir)?;

    // ---- replay ----------------------------------------------------------
    let (log, verifying_key) = open_log(strata_dir, options.seed)?;
    let outcome = migrate_snapshot_into(&snapshot, &log, &blake3_before, verifying_key)?;

    // ---- re-hash the source: a single changed byte stops the seal -------
    let blake3_after = files.blake3_hex()?;
    if blake3_before != blake3_after {
        // Frames already landed; the log is incomplete and the destination
        // policy above refuses any further run into this directory. Report
        // as an error instead of panicking past the write (audit finding).
        return Err(MigrationError::SourceTampered {
            before: blake3_before,
            after: blake3_after,
        });
    }

    finish(log, strata_dir, outcome, snapshot, &blake3_after, started)
}

enum DestState {
    Empty,
    Partial,
    Complete,
}

fn dest_state(dir: &Path) -> DestState {
    let occupied = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| !e.file_name().to_string_lossy().starts_with('.'))
        })
        .unwrap_or(false);
    if !occupied {
        return DestState::Empty;
    }
    let Ok(opened) = open_migrated(dir) else {
        return DestState::Partial;
    };
    let has_receipt = opened.log.read_frames(1).ok().is_some_and(|frames| {
        frames
            .iter()
            .any(|f| f.kind == records::KIND_MIGRATION_RECEIPT)
    });
    if has_receipt {
        DestState::Complete
    } else {
        DestState::Partial
    }
}

fn refuse_occupied(dir: &Path) -> Result<(), MigrationError> {
    let path = dir.display().to_string();
    match dest_state(dir) {
        DestState::Empty => Ok(()),
        DestState::Partial => Err(MigrationError::DestinationPartial { path }),
        DestState::Complete => Err(MigrationError::DestinationComplete { path }),
    }
}

/// Remove a partial `--to` (and its external log key). A completed
/// destination is refused.
pub fn rollback(dest: &Path) -> Result<(), MigrationError> {
    match dest_state(dest) {
        DestState::Empty => Ok(()),
        DestState::Complete => Err(MigrationError::RollbackRefused {
            path: dest.display().to_string(),
        }),
        DestState::Partial => {
            if dest.exists() {
                std::fs::remove_dir_all(dest)?;
            }
            let key = log_signing_key_path(dest);
            if key.exists() {
                std::fs::remove_file(key)?;
            }
            Ok(())
        }
    }
}

fn open_log(
    strata_dir: &Path,
    pinned_seed: Option<[u8; 32]>,
) -> Result<(StrataLog, [u8; 32]), MigrationError> {
    // OS entropy unless a test pins the seed. The seed is not derived from
    // the source hash: that hash is published in PARAMS.
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
    let verifying_key = ed25519_dalek::SigningKey::from_bytes(&seed)
        .verifying_key()
        .to_bytes();
    let log = StrataLog::open_seeded(strata_dir, seed)
        .map_err(|e| MigrationError::Strata(e.to_string()))?;
    Ok((log, verifying_key))
}

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
        frames_verified: 0,
        frames_total: 0,
        dropped_vectors: snapshot.dropped_vectors,
        envelope_head: snapshot.envelope_head.clone().unwrap_or_default(),
        source_blake3: source_blake3.to_string(),
        receipt_digest: None,
        receipt_verified: false,
        idempotent_reuse: false,
        duration: started.elapsed(),
    }
}

fn table_rows(snapshot: &source::SourceSnapshot, table: &str) -> u64 {
    source::table(&snapshot.archive, table)
        .map(|t| t.rows.len() as u64)
        .unwrap_or(0)
}

fn skipped_tables_for(snapshot: &source::SourceSnapshot) -> Vec<String> {
    // Nonempty tables that do not become frames. Driven by sqlite_master
    // (or the archive's own tables), never a hardcoded name list.
    let mut names: Vec<String> = snapshot
        .all_nonempty_tables
        .iter()
        .filter(|(name, _)| !EMITTED_TABLES.contains(&name.as_str()))
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    names
}

/// Columns present on a nonempty source table whose values are not written
/// into a frame. Fully captured tables (`knowledge_nodes`,
/// `memory_connections`, `fsrs_cards`) contribute nothing.
fn dropped_columns_for(snapshot: &source::SourceSnapshot) -> Vec<String> {
    let mut out = Vec::new();
    for (table, columns) in &snapshot.nonempty_columns {
        for column in columns {
            if column_carried(table, column) {
                continue;
            }
            out.push(format!("{table}.{column}"));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `true` when this column's value is stored on a migrated record.
fn column_carried(table: &str, column: &str) -> bool {
    match table {
        "knowledge_nodes" | "memory_connections" | "fsrs_cards" => true,
        "sync_tombstones" => matches!(column, "table_name" | "row_id" | "deleted_at" | "reason"),
        "deletion_tombstones" => {
            matches!(
                column,
                "memory_id" | "deleted_at" | "reason" | "node_type" | "tags"
            )
        }
        "walk_receipts" => matches!(column, "receipt_id" | "canonical_json" | "created_at"),
        "schema_version" => column == "version",
        "receipt_envelopes" => column == "entry_digest",
        _ => false,
    }
}

/// Everything produced during one replay.
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
fn migrate_snapshot_into(
    snapshot: &source::SourceSnapshot,
    log: &StrataLog,
    source_blake3: &str,
    log_verifying_key: [u8; 32],
) -> Result<ReplayOutcome, MigrationError> {
    let archive = &snapshot.archive;

    let (mut node_records, kernel_ids, supersessions) = extract_nodes(archive)?;
    let (edge_records, legacy_links) = extract_edges(archive, &kernel_ids)?;
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
                log_verifying_key,
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
    for record in &legacy_links {
        writer.push(records::KIND_LEGACY_LINK, borsh::to_vec(record))?;
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
            for rating in fsrs_ratings_for(reps, lapses) {
                let event = ReviewEvent {
                    card_id: kernel_id,
                    rating,
                    event_seq: writer.next_seq(),
                };
                writer.push(records::KIND_FSRS_REVIEW, borsh::to_vec(&event))?;
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
            dropped_columns: dropped_columns_for(&snapshot),
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

    let verified = verify::verify_open_log(&log, strata_dir, outcome.anchor)?;
    relocate_log_key(strata_dir)?;

    Ok(MigrationReport {
        nodes: outcome.nodes,
        edges: outcome.edges,
        fsrs_events: outcome.fsrs_events,
        skipped_tables: skipped_tables_for(&snapshot),
        verify_passed: true,
        frames_verified: verified.frames_verified,
        frames_total: verified.frames_total,
        dropped_vectors: snapshot.dropped_vectors,
        envelope_head: snapshot.envelope_head.unwrap_or_default(),
        source_blake3: source_blake3.to_string(),
        receipt_digest: Some(hex32(&receipt.checksum)),
        receipt_verified: receipt.verify_checksum() && receipt.verify_signature(),
        idempotent_reuse: false,
        duration: started.elapsed(),
    })
}

/// Move `<dest>/strata.key` to [`log_signing_key_path`] and force mode 0600.
fn relocate_log_key(strata_dir: &Path) -> Result<(), MigrationError> {
    let src = strata_dir.join("strata.key");
    let dst = log_signing_key_path(strata_dir);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(&src, &dst)?;
    set_mode_0600(&dst)?;
    Ok(())
}

/// The receipt-signing key lives NEXT TO the destination log, never inside
/// the log: `<parent-of---to>/receipt-signing.key`. The log carries only
/// the verifying key.
fn receipt_key_dir_of(strata_dir: &Path) -> std::path::PathBuf {
    strata_dir.parent().unwrap_or(strata_dir).to_path_buf()
}

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Batched append helper: predicts frame seqs from the log head, asserts the
/// returned acks match, and never holds more than `BATCH_FRAMES` frames.
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

/// Deterministic rating series reproducing an fsrs_cards row exactly:
/// `reps - lapses` good (3) reviews, then `lapses` again (1) reviews. The
/// fold yields `review_count == reps` and `lapse_count == lapses`; the final
/// phase is Relearning iff the card ever lapsed, else Review.
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

const FSRS_MAPPED_COLUMNS: &[&str] = &["memory_id", "reps", "lapses"];

/// FSRS final state the fold cannot reproduce (stability/difficulty floats,
/// due dates, phase) rides on the node's legacy capture (blocker 4).
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
        })
        .collect();
    Ok(records)
}

/// Canonical string form of a snapshot value for legacy column capture:
/// ints/floats/text verbatim (floats are Display'd, deterministic), blobs hex.
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
        let value = row.get(&column)?;
        out.push((format!("{table}.{column}"), legacy_value(value)));
    }
    Ok(out)
}

/// Decoded `knowledge_nodes`: node records, the legacy→kernel id map (dense,
/// 1-based, source row order), and supersession pointers.
type NodeSet = (
    Vec<NodeRecord>,
    HashMap<String, u64>,
    Vec<SupersessionRecord>,
);

/// Decode `knowledge_nodes` into node records, the legacy→kernel id map
/// (dense, 1-based, source row order), and supersession pointers.
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
/// Legacy link types are folded into the 8-type STRATA vocabulary: anything
/// outside [`STRATA_EDGE_VOCABULARY`] becomes `derived_from` with
/// `legacy_inferred = 1`, except association rows, which are returned as
/// legacy-link records and never as edges.
fn extract_edges(
    archive: &PortableArchive,
    kernel_ids: &HashMap<String, u64>,
) -> Result<(Vec<EdgeRecord>, Vec<records::LegacyLinkRecord>), MigrationError> {
    let Some(table) = source::table(archive, "memory_connections") else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut records = Vec::with_capacity(table.rows.len());
    let mut legacy_links = Vec::new();
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
        let strength_q32 = strata_kernel::canonical::to_q32_32(row.real("strength")?);
        let created_ms = source::timestamp_ms(row.text("created_at")?)?;
        let last_activated_ms = source::timestamp_ms(row.text("last_activated")?)?;
        let activation_count = row.integer_or("activation_count", 0)? as i32;
        if is_noncausal_association(&legacy_link_type) {
            legacy_links.push(records::LegacyLinkRecord {
                record_version: RECORD_VERSION,
                source_kernel_id,
                target_kernel_id,
                source_legacy_id,
                target_legacy_id,
                legacy_link_type,
                strength_q32,
                created_ms,
                last_activated_ms,
                activation_count,
                legacy,
            });
            continue;
        }
        let (link_type, legacy_inferred) =
            if STRATA_EDGE_VOCABULARY.contains(&legacy_link_type.as_str()) {
                (legacy_link_type.clone(), false)
            } else {
                ("derived_from".to_string(), true)
            };
        records.push(EdgeRecord {
            record_version: RECORD_VERSION,
            source_kernel_id,
            target_kernel_id,
            strength_q32,
            link_type,
            legacy_inferred,
            legacy_link_type,
            created_ms,
            last_activated_ms,
            activation_count,
            legacy,
            source_legacy_id,
            target_legacy_id,
        });
    }
    Ok((records, legacy_links))
}

/// Decode `sync_tombstones` and `deletion_tombstones`.
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

    #[test]
    fn timestamps_parse_from_rfc3339_variants() {
        assert_eq!(
            source::timestamp_ms("2026-09-28T12:00:00+00:00").unwrap(),
            1_790_596_800_000
        );
        assert!(source::timestamp_ms("2026-09-28T12:00:00Z").is_ok());
        assert!(source::timestamp_ms("not a date").is_err());
    }

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
    fn legacy_link_types_fold_into_derived_from() {
        for legacy in ["causal", "temporal", "user_defined", "pattern"] {
            assert!(!STRATA_EDGE_VOCABULARY.contains(&legacy));
            assert!(!is_noncausal_association(legacy));
        }
        assert!(is_noncausal_association("semantic"));
        assert!(is_noncausal_association("similarity"));
        assert!(is_noncausal_association("Similarity"));
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
