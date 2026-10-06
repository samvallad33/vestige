//! Vestige CLI
//!
//! Command-line interface for managing cognitive memory system.

// Supplies the `__isoc23_*` and `__cxa_call_terminate` symbols that the
// statically linked ONNX Runtime archive imports from glibc >= 2.38 and
// libstdc++ >= GCC 13. Compiled into each binary root rather than into the
// library so the definitions are always part of the final link instead of
// being subject to archive member selection. See the module docs.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[path = "../glibc_compat.rs"]
mod glibc_compat;

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use chrono::{NaiveDate, Utc};
use clap::{Args, Parser, Subcommand};
use colored::Colorize;
use vestige_core::{
    ConnectionRecord, IngestInput, PortableImportMode, SecretConfidence, SecretPolicy,
    SourceEnvelope, SourceUpsertOutcome, Storage, scan_secrets,
};

/// Vestige - Cognitive Memory System CLI
#[derive(Parser)]
#[command(name = "vestige")]
#[command(author = "samvallad33")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CLI for the Vestige cognitive memory system")]
#[command(
    long_about = "Vestige is a local-first memory system for coding agents.\n\nVestige 4.0 keeps memories in a Strata log inside the data directory: an append-only log where every write passes a gate and FSRS-6 schedules review. Recall is by exact handle (memory id, id prefix, tag); a Strata log runs no similarity search. Builds with the legacy-sqlite feature keep the v3 SQLite engine."
)]
struct Cli {
    /// Use a specific Vestige data directory for this command.
    #[arg(long, global = true, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

static CLI_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// `.serve.lock`, held from the first `open_storage` to exit. The Strata log
/// has one writer; a CLI command that opens it is that writer while it runs.
static CLI_SERVE_LOCK: OnceLock<fs::File> = OnceLock::new();

#[derive(Debug, Clone, Default, Args)]
struct SandwichInstallOptions {
    /// Overwrite existing staged Vestige hook and agent files.
    #[arg(long)]
    force: bool,

    /// Wire optional UserPromptSubmit preflight hooks.
    #[arg(long)]
    enable_preflight: bool,

    /// Wire both optional preflight hooks and the optional Sanhedrin verifier.
    #[arg(long)]
    enable_sandwich: bool,

    /// Wire optional Sanhedrin Stop hook.
    #[arg(long)]
    enable_sanhedrin: bool,

    /// On Apple Silicon, auto-start the local MLX Sanhedrin backend.
    #[arg(long)]
    with_launchd: bool,

    /// Also stage the large memory-loader hook file.
    #[arg(long)]
    include_memory_loader: bool,

    /// OpenAI-compatible chat completions endpoint for optional Sanhedrin.
    #[arg(long, value_name = "URL")]
    sanhedrin_endpoint: Option<String>,

    /// Model name passed to the optional Sanhedrin endpoint.
    #[arg(long, value_name = "MODEL")]
    sanhedrin_model: Option<String>,

    /// Use a local checkout/release root containing hooks/ and agents/.
    #[arg(long, value_name = "DIR", hide = true)]
    src: Option<PathBuf>,
}

#[derive(Subcommand)]
enum SandwichCommands {
    /// Install/update Cognitive Sandwich companion files without enabling hooks by default.
    Install {
        /// Install files from a specific release tag instead of latest.
        #[arg(long)]
        version: Option<String>,

        #[command(flatten)]
        options: SandwichInstallOptions,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Show memory statistics
    Stats {
        /// Show tagging/retention distribution
        #[arg(long)]
        tagging: bool,

        /// Show cognitive state distribution
        #[arg(long)]
        states: bool,
    },

    /// Run health check with warnings and recommendations
    Health,

    /// Run the memory consolidation cycle (a no-op on a Strata log in 4.0)
    ///
    /// A legacy SQLite store runs decay, promotion, pruning and dedup. A
    /// Strata log runs none of those passes in 4.0, so nothing changes.
    Consolidate,

    /// Import a v3 vestige.db into a Strata log by running vestige-upgrade
    ///
    /// vestige.db is read and left byte-identical. Once a Strata log exists in
    /// the data directory it is the live store and there is nothing to import.
    Upgrade {
        /// Report what would be imported without running vestige-upgrade.
        #[arg(long)]
        dry_run: bool,
    },

    /// Update Vestige binaries from the latest GitHub release
    Update {
        /// Install a specific release tag instead of latest (example: v2.1.27)
        #[arg(long)]
        version: Option<String>,

        /// Override install directory (defaults to the current vestige binary's directory)
        #[arg(long)]
        install_dir: Option<PathBuf>,

        /// Print what would be updated without changing files
        #[arg(long)]
        dry_run: bool,

        /// Deprecated: companion updates are skipped by default.
        #[arg(long)]
        no_sandwich: bool,

        /// Also refresh optional Claude Code Cognitive Sandwich companion files.
        #[arg(long)]
        sandwich_companion: bool,

        #[command(flatten)]
        sandwich: SandwichInstallOptions,
    },

    /// Manage optional Claude Code Cognitive Sandwich companion files.
    Sandwich {
        #[command(subcommand)]
        command: SandwichCommands,
    },

    /// Re-ingest memories from a JSON file (for example a `vestige export`)
    ///
    /// Each memory in a `vestige export --format json` file (or an MCP recall
    /// result) is ingested as a new record: new id, created now, fresh review
    /// state. Ids, timestamps, review history and edges in the file are not
    /// restored. Portable archives import only into a legacy SQLite store.
    ///
    /// A Strata backup made by `vestige backup` is a directory. Restore it by
    /// stopping Vestige and copying the backup's log/ (and store.meta, when
    /// present) into the data directory in place of its log/. This command
    /// does not do that.
    Restore {
        /// Path to the JSON file
        file: PathBuf,
    },

    /// Back up the live store
    ///
    /// On a Strata log (the 4.0 store) this seals the log and copies log/
    /// (plus store.meta, when present) into a new directory. vestige.db is
    /// never copied: after an upgrade it is the old v3 file, not the live
    /// store. Restore by stopping Vestige and copying the backup's log/ into
    /// the data directory. A legacy SQLite store is written as one consistent
    /// snapshot file (VACUUM INTO).
    Backup {
        /// Destination: a new (or empty) directory for a Strata log, a file
        /// for a legacy SQLite store
        output: PathBuf,
    },

    /// Migrate a v3 SQLite store into a STRATA log (read-only over the source)
    MigrateToStrata {
        /// Path to the v3 SQLite database (or its data directory)
        #[arg(long)]
        from: PathBuf,

        /// Destination STRATA directory (defaults to <data-dir>/strata)
        #[arg(long)]
        to: Option<PathBuf>,

        /// Verify the source and report counts without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Migrate a source with a non-empty WAL from a consistent snapshot
        /// copy (the original is still never modified)
        #[arg(long)]
        accept_wal_snapshot: bool,
    },

    /// Verify a STRATA directory. Read-only: creates no files.
    ///
    /// Accepts a migrated log (no `kernel.log`), a live store (`log/` plus
    /// `store.meta`), or the kernel/gate layout.
    ///
    /// The key fingerprint is the lowercase blake3 hex of the 32-byte
    /// ed25519 verifying key. A migration receipt is trusted only when that
    /// key matches `receipt-signing.key` beside the log. With no receipt,
    /// the fingerprint is `strata.key` and the report says so.
    StrataVerify {
        /// Directory to verify
        dir: PathBuf,
        /// Require this key fingerprint (blake3 hex of the ed25519 verifying key)
        #[arg(long)]
        expect_key: Option<String>,
    },

    /// Export memories in JSON or JSONL format
    Export {
        /// Output file path
        output: PathBuf,
        /// Export format: json or jsonl
        #[arg(long, default_value = "json")]
        format: String,
        /// Filter by tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Only export memories created after this date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
    },

    /// Export an exact portable archive (legacy SQLite stores only)
    ///
    /// A Strata log in 4.0 does not write portable archives and refuses. Use
    /// `vestige export` for the memories or `vestige backup` for an exact copy
    /// of the log.
    PortableExport {
        /// Output archive path
        output: PathBuf,
    },

    /// Import an exact portable archive (legacy SQLite stores only)
    ///
    /// A Strata log in 4.0 does not read portable archives and refuses.
    PortableImport {
        /// Input archive path
        input: PathBuf,
        /// Merge into the current database instead of requiring an empty target
        #[arg(long)]
        merge: bool,
    },

    /// Two-way sync with a portable archive file or Vestige Cloud (legacy
    /// SQLite stores only)
    ///
    /// Sync merges portable archives, which a Strata log in 4.0 does not write
    /// or read, so it refuses there. Use `vestige export` or `vestige backup`.
    Sync {
        /// Sync archive path, often in Dropbox/iCloud/Syncthing/Git.
        /// Omit when using --cloud.
        archive: Option<PathBuf>,
        /// Sync with the hosted Vestige Cloud managed-sync service instead of a
        /// file. Requires a sync key (VESTIGE_CLOUD_SYNC_KEY) and endpoint
        /// (--endpoint or VESTIGE_CLOUD_ENDPOINT).
        #[arg(long)]
        cloud: bool,
        /// Vestige Cloud base endpoint (e.g. https://sync.vestige.dev).
        /// Defaults to the VESTIGE_CLOUD_ENDPOINT env var.
        #[arg(long)]
        endpoint: Option<String>,
    },

    /// Delete stale memories below a retention threshold (legacy SQLite stores)
    ///
    /// Deletion is withheld on a Strata log in 4.0: the log is append-only.
    /// --dry-run still lists the memories below the threshold.
    Gc {
        /// Minimum retention strength to keep (delete below this)
        #[arg(long, default_value = "0.1")]
        min_retention: f64,
        /// Maximum age in days (delete memories older than this AND below retention threshold)
        #[arg(long)]
        max_age_days: Option<u64>,
        /// Dry run - show what would be deleted without actually deleting
        #[arg(long)]
        dry_run: bool,
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Launch the memory web dashboard
    Dashboard {
        /// Port to bind the dashboard server to
        #[arg(long, default_value = "3927")]
        port: u16,
        /// Don't automatically open the browser
        #[arg(long)]
        no_open: bool,
    },

    /// Ingest a memory as a new record
    ///
    /// Nothing is merged by similarity. On a Strata log the write passes the
    /// log's gate before it is admitted, then auto-connects to earlier
    /// memories that record the same exact identity: an exact tag, or an
    /// exact file path, commit sha, `owner/repo#123` reference or URL. Shared
    /// words never join (the ingest-time share of `vestige connect`).
    Ingest {
        /// Content to remember
        content: String,
        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Node type (fact, concept, event, person, place, note, pattern, decision)
        #[arg(long, default_value = "fact")]
        node_type: String,
        /// Source reference
        #[arg(long)]
        source: Option<String>,
        /// Backdate this memory N days in the past (legacy SQLite stores; a
        /// Strata log refuses before writing)
        #[arg(long)]
        ago_days: Option<i64>,
        /// Exact creation time (RFC 3339) of an external record such as an
        /// issue or a commit (legacy SQLite stores; a Strata log refuses
        /// before writing)
        #[arg(long)]
        created_at: Option<String>,
        /// Deliberately allow a detected credential to be stored. Prefer a
        /// secret-manager reference; this disables the default safety guard.
        #[arg(long)]
        allow_secrets: bool,
    },

    /// Ingest git commits as memory records (legacy SQLite stores only)
    ///
    /// One record per commit, upserted by repo and sha and dated to the commit
    /// time, so re-running is idempotent. The Strata store in 4.0 exposes no
    /// source upsert or creation-time rewrite, so it refuses before writing.
    IngestGit {
        /// Path to the git repository
        path: PathBuf,
        /// Only commits on/after this date (RFC 3339 or YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
        /// Only commits before this date (RFC 3339 or YYYY-MM-DD) — bound the
        /// window above the failure date so post-cause commits don't eat the
        /// max-commits budget
        #[arg(long)]
        until: Option<String>,
        /// Stop after this many commits (newest first)
        #[arg(long, default_value = "2000")]
        max_commits: usize,
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },

    /// Create typed edges between memories that record the same exact identity
    ///
    /// Scans all memories of a scope and creates a `touched` edge between
    /// each pair that records the same exact identity: an exact tag
    /// (case-sensitive), an exact file path token, a whole-token commit sha
    /// (40 hex, or 7+ hex), an `owner/repo#123` reference, or a URL. Shared
    /// words never join two memories. A tag carried by more than 14
    /// memories of the scope is too common to be evidence and is skipped;
    /// file paths always join. Every pair is printed with the identities
    /// that joined it. An edge records that two memories name the same
    /// thing, not that one caused the other: a causal walk over these edges
    /// returns hypotheses. Ingest auto-connects pairs as they land, and this
    /// full scan also catches pairs that share a path only in their text.
    Connect {
        /// Show what edges would be created without writing them
        #[arg(long)]
        dry_run: bool,
        /// Minimum distinct shared identities required (default 1)
        #[arg(long, default_value = "1")]
        min_shared: usize,
        /// Maximum edges to create (safety cap)
        #[arg(long, default_value = "100")]
        max_edges: usize,
        /// Project namespace to connect (default: user)
        #[arg(long, default_value = "user")]
        scope: String,
    },

    /// Read-only audit for credential-shaped values already in the local store.
    ScanSecrets {
        /// Include high-entropy review candidates as well as blocking matches.
        #[arg(long)]
        include_suspected: bool,
        /// Emit a machine-readable JSON report. No memory content is printed.
        #[arg(long)]
        json: bool,
        /// Stop after this many findings (default: scan the entire store).
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Retroactive Salience Backfill (legacy SQLite stores; use causal-walk)
    ///
    /// Reaches backward from a failure and lists earlier memories that share
    /// an exact entity (env var, path, identifier) with it. Candidates are
    /// associations, not proven causes. A shared name is not a recorded edge,
    /// so a Strata log in 4.0 refuses: use `vestige causal-walk
    /// --logged-write <memory-id>`, which walks recorded causal edges.
    Backfill {
        /// ID of the failure memory; if omitted, the latest failure-like memory is used
        #[arg(long)]
        failure_id: Option<String>,
        /// Force the backfill even if the event isn't auto-detected as salient
        #[arg(long)]
        manual: bool,
        /// How many days back to reach
        #[arg(long, default_value = "30")]
        lookback_days: i64,
        /// Dry run: don't actually promote the surfaced cause
        #[arg(long)]
        no_promote: bool,
        /// Demo mode: first show what a plain keyword (BM25) search returns
        /// for the failure (the lookalike, NOT the cause), then the backfill.
        #[arg(long)]
        contrast: bool,
        /// Machine-readable: print the raw backfill result as JSON (for tooling /
        /// benchmarks). Suppresses the human-formatted output.
        #[arg(long)]
        json: bool,
        /// Git repository backing this scope's commit records; enables
        /// version-range mapping from the failure text ("broke in X, worked in Y").
        #[arg(long)]
        git_repo: Option<PathBuf>,
        /// Last-known-good tag for the version range (with --git-repo)
        #[arg(long)]
        worked_in: Option<String>,
        /// First-bad tag for the version range (with --git-repo)
        #[arg(long)]
        broke_in: Option<String>,
        /// Memory id (or commit sha prefix): report the exact rule that
        /// excluded it, or its rank among the surfaced causes
        #[arg(long)]
        why_not: Option<String>,
    },

    /// Causal walk: investigate a failure from an explicit start point
    ///
    /// Successor to `backfill`. It refuses with a needs_report instead of
    /// guessing when no start point is given.
    ///
    /// On a Strata log (4.0) the walk starts at a recorded memory: a bounded
    /// backward walk over recorded causal edges (closed_by, derived_from,
    /// evidence_of, touched). It writes nothing. --logged-write names that
    /// memory directly; --node-id attaches it to a --failing-test,
    /// --stack-frame, --ci-run or version range. Those start points resolve
    /// through shared names, which are not recorded edges, so a Strata log
    /// refuses them unless --node-id says which recorded memory they are.
    ///
    /// A legacy SQLite store walks every start point through shared exact
    /// anchors to change records and, unless --no-promote, records
    /// evidence_of edges. Results are hypotheses, not proven causes.
    CausalWalk {
        /// Failing test name, walked to its file's co-touch commits (legacy
        /// SQLite stores)
        #[arg(long)]
        failing_test: Option<String>,
        /// Stack frame "file:line" or "file"; the last pre-failure toucher is
        /// the prime suspect, SZZ-lite (legacy SQLite stores)
        #[arg(long)]
        stack_frame: Option<String>,
        /// Agent-trace run id whose failure channel seeds the anchors (legacy
        /// SQLite stores)
        #[arg(long)]
        ci_run: Option<String>,
        /// Memory / tool-call record id whose recorded edges are walked
        #[arg(long)]
        logged_write: Option<String>,
        /// The memory that records this symptom. Attached to every start
        /// point given, which lets a Strata log walk it; alone it walks that
        /// memory, like --logged-write
        #[arg(long)]
        node_id: Option<String>,
        /// Git repository for --worked-in/--broke-in, single repo per call
        /// (legacy SQLite stores, or a Strata log with --node-id)
        #[arg(long)]
        git_repo: Option<PathBuf>,
        /// Last-known-good tag (with --git-repo)
        #[arg(long)]
        worked_in: Option<String>,
        /// First-bad tag (with --git-repo)
        #[arg(long)]
        broke_in: Option<String>,
        /// How many days back from the failure anchor suspects may lie (legacy
        /// SQLite stores; a Strata walk is bounded by depth and node count)
        #[arg(long, default_value = "30")]
        lookback_days: i64,
        /// Dry run: don't persist evidence_of trail edges (a Strata walk never
        /// writes)
        #[arg(long)]
        no_promote: bool,
        /// Project namespace to walk (default: user)
        #[arg(long, default_value = "user")]
        scope: String,
        /// Machine-readable: print the raw causal walk result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Recall memories by exact handle (--handle), or by free text on a legacy
    /// SQLite store
    ///
    /// On a Strata log (4.0) recall is by exact handle only: a memory id, a
    /// unique id prefix of 8 or more characters, or an exact tag. It prints
    /// the matching memories and their one-hop recorded edges, like the MCP
    /// recall tool's `handle` argument. A free-text QUERY is refused there,
    /// with any handles found in the text.
    ///
    /// On a legacy SQLite store a QUERY runs the v3 deep_reference engine
    /// (keyword search, FSRS-6 trust, spreading activation, supersession and
    /// contradiction analysis) and prints its answer, evidence and confidence.
    Recall {
        /// Free-text query or claim to reason about (legacy SQLite stores)
        #[arg(required_unless_present = "handle", conflicts_with = "handle")]
        query: Option<String>,
        /// Exact handle: memory id, unique id prefix (8+ chars), or exact tag
        #[arg(long)]
        handle: Option<String>,
        /// How many memories to analyze for a QUERY (candidate depth)
        #[arg(long, default_value = "20")]
        depth: i64,
        /// Output raw JSON instead of the human-readable summary
        #[arg(long)]
        json: bool,
    },

    /// Compose: list NEVER-COMPOSED memory pairs as leads, not findings
    ///
    /// On a Strata log (4.0) this is GhostLink `propose`. The bridge lens
    /// (default) lists pairs within three recorded typed-edge hops (touched,
    /// derived_from, closed_by) that were never woven, ranked by hop distance,
    /// composition novelty and retention. The divergent lens lists pairs no
    /// recorded edge joins. Every pair carries its proof; nothing is ranked by
    /// text or vector similarity. When nothing qualifies, it says why.
    ///
    /// On a legacy SQLite store a pair is linked within three recorded
    /// causal-edge hops but never joined by a composition event.
    Compose {
        /// How many pairs to list
        #[arg(long, default_value = "5")]
        limit: i32,
        /// Strata log only: `bridge` (default) or `divergent`
        #[arg(long)]
        lens: Option<String>,
        /// Optional tag filter (comma-separated) to focus a domain
        #[arg(long)]
        tags: Option<String>,
        /// Exact project namespace. Default: `user` on a Strata log, every
        /// scope on a legacy SQLite store.
        #[arg(long)]
        scope: Option<String>,
        /// Output raw JSON instead of the human-readable summary
        #[arg(long)]
        json: bool,
    },

    /// Project the durable subset of a scope (decisions, patterns, rule-tagged facts)
    /// into a fenced region of a client rule file such as CLAUDE.md or MEMORY.md, one
    /// memory id per line. Prints the diff; writes only with --write, and only the fence.
    Project {
        /// Target file (created if missing; only the fenced region is ever replaced)
        #[arg(long, value_name = "FILE", default_value = "CLAUDE.md")]
        out: PathBuf,
        /// 'claude-md' (grouped section) or 'memory-md' (one line per memory)
        #[arg(long, default_value = "claude-md")]
        format: String,
        /// Project namespace to project
        #[arg(long, default_value = "user")]
        scope: String,
        /// Leave out memories below this retention
        #[arg(long, default_value = "0.3")]
        min_retention: f64,
        /// At most this many memories
        #[arg(long, default_value = "60")]
        max_items: usize,
        /// Apply the change instead of printing the diff
        #[arg(long)]
        write: bool,
        /// Output raw JSON instead of the diff
        #[arg(long)]
        json: bool,
    },

    /// Start standalone HTTP MCP server (no stdio, for remote access)
    Serve {
        /// HTTP transport port
        #[arg(long, default_value = "3928")]
        port: u16,
        /// Also start the dashboard
        #[arg(long)]
        dashboard: bool,
        /// Dashboard port
        #[arg(long, default_value = "3927")]
        dashboard_port: u16,
    },

    /// Run the planted-cause selftest in a throwaway store (the live store is
    /// only read)
    ///
    /// On a Strata log it plants a cause, an intermediate and a symptom with
    /// recorded derived_from edges plus distractors in a temp log, walks back
    /// over recorded causal edges, and deletes the temp log. A legacy SQLite
    /// store runs backfill hit@1/hit@3 and gap calibration on a temp copy.
    Selftest,

    /// Find decayed fix/lesson memories linked to a failure
    ///
    /// On a Strata log (4.0) the link is a recorded causal edge (corrects,
    /// derived_from, evidence_of, closed_by), walked back from the failure. A
    /// legacy SQLite store matches a shared exact anchor instead.
    ForgottenLesson {
        /// Failure memory id to inspect
        failure_id: String,
        /// Exact project namespace of the failure (default: user)
        #[arg(long)]
        scope: Option<String>,
        /// Output raw JSON (same payload as the MCP tool)
        #[arg(long)]
        json: bool,
    },

    /// Prove which commit broke a test: the causal walk proposes, the test decides
    ///
    /// Walks back from the failure memory (--logged-write) to the commits it
    /// reaches over recorded edges, drops any committed after --reported-at
    /// or outside good..bad, and freezes the protocol (the test's sha256,
    /// the two ends, the leads) before any test runs. Then it runs your test
    /// on those leads only and on the parent of the earliest failing one.
    /// Stock `git bisect run` over the whole range confirms it, reusing the
    /// verdicts already recorded. Then it finds the smallest set of the
    /// commit's changes that still fails, tests the commit without them, and
    /// undoes them on the bad ref. The result is a verdict card: LEAD,
    /// BOUNDARY, CONFIRMED, ISOLATED, REVERSED, each with whether it holds
    /// and the runs behind it.
    ///
    /// With --flaky, for a bug that only shows some of the time, each commit
    /// is tested repeatedly until the evidence is decisive, and the card
    /// gains REPEATED.
    ///
    /// Output lines are labelled `[recorded link]` (a lead from the walk) or
    /// `[tested]` (a test run). Every run is saved as an `event` memory and
    /// written to the report, where each entry carries the sha256 of the one
    /// before it. The test runs in a temporary git worktree, never in your
    /// checkout. Its exit code is read as git bisect reads it: 0 good, 125
    /// cannot test, any other code bad.
    ///
    /// `vestige prove --check <report.json>` re-verifies a report offline.
    Prove(vestige_mcp::walk_verify::ProveArgs),

    /// What `git bisect run` calls during `prove`
    #[command(name = vestige_mcp::walk_verify::CHILD_COMMAND, hide = true)]
    ProveChild {
        /// `probe` or `sim`
        mode: String,
        /// The run configuration `prove` wrote
        cfg: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(data_dir) = cli.data_dir {
        CLI_DATA_DIR
            .set(expand_tilde(data_dir))
            .map_err(|_| anyhow::anyhow!("data directory was initialized more than once"))?;
    }

    match cli.command {
        Commands::Stats { tagging, states } => run_stats(tagging, states),
        Commands::Health => run_health(),
        Commands::Consolidate => run_consolidate(),
        Commands::Upgrade { dry_run } => run_upgrade(dry_run),
        Commands::Update {
            version,
            install_dir,
            dry_run,
            no_sandwich,
            sandwich_companion,
            sandwich,
        } => run_update(
            version,
            install_dir,
            dry_run,
            no_sandwich,
            sandwich_companion,
            sandwich,
        ),
        Commands::Sandwich { command } => match command {
            SandwichCommands::Install { version, options } => {
                run_sandwich_install(version.as_deref(), &options)
            }
        },
        Commands::Restore { file } => run_restore(file),
        Commands::Backup { output } => run_backup(output),
        Commands::MigrateToStrata {
            from,
            to,
            dry_run,
            accept_wal_snapshot,
        } => run_migrate_to_strata(from, to, dry_run, accept_wal_snapshot),
        Commands::StrataVerify { dir, expect_key } => run_strata_verify(dir, expect_key),
        Commands::Export {
            output,
            format,
            tags,
            since,
        } => run_export(output, format, tags, since),
        Commands::PortableExport { output } => run_portable_export(output),
        Commands::PortableImport { input, merge } => run_portable_import(input, merge),
        Commands::Sync {
            archive,
            cloud,
            endpoint,
        } => run_sync(archive, cloud, endpoint),
        Commands::Gc {
            min_retention,
            max_age_days,
            dry_run,
            yes,
        } => run_gc(min_retention, max_age_days, dry_run, yes),
        Commands::Dashboard { port, no_open } => run_dashboard(port, !no_open),
        Commands::Ingest {
            content,
            tags,
            node_type,
            source,
            ago_days,
            created_at,
            allow_secrets,
        } => run_ingest(
            content,
            tags,
            node_type,
            source,
            ago_days,
            created_at,
            allow_secrets,
        ),
        Commands::IngestGit {
            path,
            since,
            until,
            max_commits,
            json,
        } => run_ingest_git(path, since, until, max_commits, json),
        Commands::Connect {
            dry_run,
            min_shared,
            max_edges,
            scope,
        } => run_connect(dry_run, min_shared, max_edges, scope),
        Commands::ScanSecrets {
            include_suspected,
            json,
            limit,
        } => run_scan_secrets(include_suspected, json, limit),
        Commands::Backfill {
            failure_id,
            manual,
            lookback_days,
            no_promote,
            contrast,
            json,
            git_repo,
            worked_in,
            broke_in,
            why_not,
        } => run_backfill(
            failure_id,
            manual,
            lookback_days,
            !no_promote,
            contrast,
            json,
            git_repo,
            worked_in,
            broke_in,
            why_not,
        ),
        Commands::CausalWalk {
            failing_test,
            stack_frame,
            ci_run,
            logged_write,
            node_id,
            git_repo,
            worked_in,
            broke_in,
            lookback_days,
            no_promote,
            scope,
            json,
        } => run_causal_walk(
            failing_test,
            stack_frame,
            ci_run,
            logged_write,
            node_id,
            git_repo,
            worked_in,
            broke_in,
            lookback_days,
            !no_promote,
            scope,
            json,
        ),
        Commands::Recall {
            query,
            handle,
            depth,
            json,
        } => run_recall(query, handle, depth, json),
        Commands::Compose {
            limit,
            lens,
            tags,
            scope,
            json,
        } => run_compose(limit, lens, tags, scope, json),
        Commands::Project {
            out,
            format,
            scope,
            min_retention,
            max_items,
            write,
            json,
        } => run_project(out, format, scope, min_retention, max_items, write, json),
        Commands::Serve {
            port,
            dashboard,
            dashboard_port,
        } => run_serve(port, dashboard, dashboard_port),
        Commands::Selftest => run_selftest(),
        Commands::ForgottenLesson {
            failure_id,
            scope,
            json,
        } => run_forgotten_lesson(failure_id, scope, json),
        Commands::Prove(args) => {
            let code =
                vestige_mcp::walk_verify::run(&args, || Ok((open_storage()?, cli_data_dir()?)))?;
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
        Commands::ProveChild { mode, cfg } => {
            std::process::exit(vestige_mcp::walk_verify::child(&mode, &cfg))
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReleaseAsset {
    target: &'static str,
    archive_ext: &'static str,
    binary_suffix: &'static str,
}

struct UpdateTempDir {
    path: PathBuf,
}

impl UpdateTempDir {
    fn create() -> anyhow::Result<Self> {
        let path = env::temp_dir().join(format!(
            "vestige-update-{}-{}",
            std::process::id(),
            Utc::now().timestamp_millis()
        ));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create temp directory {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for UpdateTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn release_asset_for(os: &str, arch: &str) -> anyhow::Result<ReleaseAsset> {
    match (os, arch) {
        ("macos", "aarch64") => Ok(ReleaseAsset {
            target: "aarch64-apple-darwin",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("macos", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-apple-darwin",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("linux", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-unknown-linux-gnu",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("linux", "aarch64") => Ok(ReleaseAsset {
            target: "aarch64-unknown-linux-gnu",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("windows", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-pc-windows-msvc",
            archive_ext: "zip",
            binary_suffix: ".exe",
        }),
        _ => anyhow::bail!(
            "unsupported platform for vestige update: {}-{}. Download manually from https://github.com/samvallad33/vestige/releases",
            os,
            arch
        ),
    }
}

fn current_release_asset() -> anyhow::Result<ReleaseAsset> {
    release_asset_for(env::consts::OS, env::consts::ARCH)
}

fn release_download_url(asset: ReleaseAsset, version: Option<&str>) -> String {
    let archive_name = format!("vestige-mcp-{}.{}", asset.target, asset.archive_ext);
    match version {
        Some(version) => {
            let tag = normalize_release_tag(version);
            format!(
                "https://github.com/samvallad33/vestige/releases/download/{}/{}",
                tag, archive_name
            )
        }
        None => format!(
            "https://github.com/samvallad33/vestige/releases/latest/download/{}",
            archive_name
        ),
    }
}

fn normalize_release_tag(version: &str) -> String {
    if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{}", version)
    }
}

fn source_archive_url(tag: &str) -> String {
    format!(
        "https://github.com/samvallad33/vestige/archive/refs/tags/{}.tar.gz",
        tag
    )
}

fn download_file(url: &str, output: &Path, action: &str) -> anyhow::Result<()> {
    run_command(
        Command::new("curl")
            .arg("-fsSL")
            .arg("-A")
            .arg("vestige-cli")
            .arg(url)
            .arg("-o")
            .arg(output),
        action,
    )
}

fn parse_sha256(text: &str) -> anyhow::Result<String> {
    let hash = text
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow::anyhow!("checksum file is empty"))?
        .to_ascii_lowercase();
    if hash.len() != 64 || !hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        anyhow::bail!("checksum file does not contain a valid SHA-256 hash");
    }
    Ok(hash)
}

fn sha256_from_command(command: &mut Command) -> anyhow::Result<Option<String>> {
    match command.output() {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            Ok(Some(parse_sha256(&text)?))
        }
        Ok(_) => Ok(None),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).context("failed to run checksum command"),
    }
}

fn compute_sha256(path: &Path) -> anyhow::Result<String> {
    #[cfg(windows)]
    {
        if let Some(hash) = sha256_from_command(
            Command::new("powershell")
                .arg("-NoProfile")
                .arg("-Command")
                .arg("(Get-FileHash -Algorithm SHA256 -LiteralPath $args[0]).Hash.ToLowerInvariant()")
                .arg(path),
        )? {
            return Ok(hash);
        }
    }

    #[cfg(not(windows))]
    {
        if let Some(hash) =
            sha256_from_command(Command::new("shasum").arg("-a").arg("256").arg(path))?
        {
            return Ok(hash);
        }
        if let Some(hash) = sha256_from_command(Command::new("sha256sum").arg(path))? {
            return Ok(hash);
        }
    }

    anyhow::bail!("no SHA-256 command available to verify release archive");
}

fn verify_release_checksum(archive_path: &Path, checksum_path: &Path) -> anyhow::Result<()> {
    let expected = parse_sha256(&fs::read_to_string(checksum_path).with_context(|| {
        format!(
            "failed to read release checksum file {}",
            checksum_path.display()
        )
    })?)?;
    let actual = compute_sha256(archive_path)?;
    if actual != expected {
        anyhow::bail!(
            "release archive checksum mismatch for {}",
            archive_path.display()
        );
    }
    Ok(())
}

fn latest_release_tag() -> anyhow::Result<String> {
    let temp_dir = UpdateTempDir::create()?;
    let metadata_path = temp_dir.path.join("latest-release.json");
    download_file(
        "https://api.github.com/repos/samvallad33/vestige/releases/latest",
        &metadata_path,
        "checking latest Vestige release",
    )?;
    let file = fs::File::open(&metadata_path)?;
    let metadata: serde_json::Value =
        serde_json::from_reader(file).context("failed to parse latest Vestige release metadata")?;
    metadata
        .get("tag_name")
        .and_then(|tag| tag.as_str())
        .map(|tag| tag.to_string())
        .ok_or_else(|| anyhow::anyhow!("latest Vestige release metadata did not include tag_name"))
}

fn release_tag_for_source(version: Option<&str>) -> anyhow::Result<String> {
    match version {
        Some(version) => Ok(normalize_release_tag(version)),
        None => latest_release_tag(),
    }
}

fn find_sandwich_source_root(root: &Path) -> Option<PathBuf> {
    if root.join("hooks").is_dir() && root.join("agents").is_dir() {
        return Some(root.to_path_buf());
    }

    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("hooks").is_dir() && path.join("agents").is_dir() {
            return Some(path);
        }
    }

    None
}

fn download_sandwich_source(version: Option<&str>, output_dir: &Path) -> anyhow::Result<PathBuf> {
    let tag = release_tag_for_source(version)?;
    let archive_path = output_dir.join(format!("vestige-source-{}.tar.gz", tag));
    let url = source_archive_url(&tag);

    println!("{}: {}", "Sandwich source".white().bold(), tag);
    download_file(&url, &archive_path, "downloading Vestige source archive")?;
    extract_source_archive(&archive_path, output_dir)?;
    find_sandwich_source_root(output_dir).ok_or_else(|| {
        anyhow::anyhow!("Vestige source archive did not contain hooks/ and agents/ directories")
    })
}

fn home_dir() -> anyhow::Result<PathBuf> {
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("failed to locate home directory"))
}

fn is_vestige_hook_command(command: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "synthesis-preflight.sh",
        "cwd-state-injector.sh",
        "vestige-pulse-daemon.sh",
        "preflight-swarm.sh",
        "load-all-memory.sh",
        "veto-detector.sh",
        "sanhedrin.sh",
        "synthesis-stop-validator.sh",
        "synthesis-gate.sh",
    ];
    NEEDLES.iter().any(|needle| command.contains(needle))
}

fn scrub_vestige_hooks(settings: &mut serde_json::Value) {
    let Some(hooks) = settings
        .get_mut("hooks")
        .and_then(|hooks| hooks.as_object_mut())
    else {
        return;
    };

    for event_name in ["UserPromptSubmit", "Stop"] {
        let Some(groups) = hooks
            .get_mut(event_name)
            .and_then(|groups| groups.as_array_mut())
        else {
            continue;
        };

        for group in groups.iter_mut() {
            if let Some(commands) = group
                .get_mut("hooks")
                .and_then(|hooks| hooks.as_array_mut())
            {
                commands.retain(|hook| {
                    !hook
                        .get("command")
                        .and_then(|command| command.as_str())
                        .is_some_and(is_vestige_hook_command)
                });
            }
        }

        groups.retain(|group| {
            group
                .get("hooks")
                .and_then(|hooks| hooks.as_array())
                .is_some_and(|hooks| !hooks.is_empty())
        });
    }

    hooks.retain(|_, value| match value {
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(items) => !items.is_empty(),
        serde_json::Value::Null => false,
        _ => true,
    });

    if hooks.is_empty()
        && let Some(root) = settings.as_object_mut()
    {
        root.remove("hooks");
    }
}

fn merge_json(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base), serde_json::Value::Object(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(existing) => merge_json(existing, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

fn merge_settings_fragment(
    settings: &mut serde_json::Value,
    fragment_path: &Path,
) -> anyhow::Result<()> {
    let file = fs::File::open(fragment_path)
        .with_context(|| format!("failed to open {}", fragment_path.display()))?;
    let fragment: serde_json::Value = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", fragment_path.display()))?;
    merge_json(settings, fragment);
    Ok(())
}

fn copy_companion_files(
    source_dir: &Path,
    destination_dir: &Path,
    allowed_extensions: &[&str],
    _mode: u32,
    options: &SandwichInstallOptions,
) -> anyhow::Result<(usize, usize)> {
    fs::create_dir_all(destination_dir)?;
    let mut copied = 0;
    let mut skipped = 0;

    for entry in fs::read_dir(source_dir)
        .with_context(|| format!("failed to read {}", source_dir.display()))?
    {
        let entry = entry?;
        let source = entry.path();
        if !source.is_file() {
            continue;
        }

        let extension = source
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        if !allowed_extensions.contains(&extension) {
            continue;
        }

        let Some(file_name) = source.file_name() else {
            continue;
        };
        if file_name.to_string_lossy() == "load-all-memory.sh" && !options.include_memory_loader {
            continue;
        }

        let destination = destination_dir.join(file_name);
        if destination.exists() && !options.force {
            skipped += 1;
            continue;
        }

        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to copy {} to {}",
                source.display(),
                destination.display()
            )
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&destination)?.permissions();
            perms.set_mode(_mode);
            fs::set_permissions(&destination, perms)?;
        }

        copied += 1;
    }

    Ok((copied, skipped))
}

fn quote_shell_env(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn write_sanhedrin_env(
    hooks_dir: &Path,
    endpoint: &str,
    model: &str,
    dashboard_port: &str,
) -> anyhow::Result<()> {
    let env_path = hooks_dir.join("vestige-sanhedrin.env");
    let contents = format!(
        "VESTIGE_SANHEDRIN_ENABLED=1\nVESTIGE_SANHEDRIN_ENDPOINT={}\nVESTIGE_SANHEDRIN_MODEL={}\nVESTIGE_DASHBOARD_PORT={}\nVESTIGE_SANHEDRIN_CLAIM_MODE=1\nVESTIGE_SANHEDRIN_OUTPUT=json\n",
        quote_shell_env(endpoint),
        quote_shell_env(model),
        quote_shell_env(dashboard_port)
    );
    fs::write(&env_path, contents)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&env_path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&env_path, perms)?;
    }

    println!("{}: {}", "Sanhedrin env".white().bold(), env_path.display());
    Ok(())
}

fn install_launchd_job(source_root: &Path, home: &Path, model: &str) -> anyhow::Result<()> {
    let launchd_dir = home.join("Library").join("LaunchAgents");
    fs::create_dir_all(&launchd_dir)?;

    let template_path = source_root
        .join("launchd")
        .join("com.vestige.mlx-server.plist.template");
    let template = fs::read_to_string(&template_path)
        .with_context(|| format!("failed to read {}", template_path.display()))?;
    // XML-escape interpolated values: this plist is XML, and an unescaped model
    // string containing &, <, >, " or ' would corrupt the plist (or inject
    // elements). Escape before substitution.
    let xml_escape = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    };
    let rendered = template
        .replace("__HOME__", &xml_escape(&home.display().to_string()))
        .replace("__MODEL__", &xml_escape(model));

    let plist = launchd_dir.join("com.vestige.mlx-server.plist");
    fs::write(&plist, rendered)?;
    let _ = Command::new("launchctl").arg("unload").arg(&plist).status();
    run_command(
        Command::new("launchctl").arg("load").arg(&plist),
        "loading Vestige MLX launchd job",
    )?;
    println!("{}: {}", "launchd".white().bold(), plist.display());
    Ok(())
}

fn remove_legacy_launchd_job(home: &Path) {
    if env::consts::OS != "macos" {
        return;
    }

    let plist = home
        .join("Library")
        .join("LaunchAgents")
        .join("com.vestige.mlx-server.plist");
    if plist.exists() {
        let _ = Command::new("launchctl").arg("unload").arg(&plist).status();
        if fs::remove_file(&plist).is_ok() {
            println!(
                "{}: removed old Sanhedrin launchd job",
                "launchd".white().bold()
            );
        }
    }
}

/// Load the Claude Code settings file for a merge. A missing or blank file is an
/// empty object; a file that is not a JSON object is an error and is never
/// replaced.
fn read_settings_for_update(settings_path: &Path) -> anyhow::Result<serde_json::Value> {
    let raw = match fs::read_to_string(settings_path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(serde_json::json!({}));
        }
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", settings_path.display()));
        }
    };
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    let settings: serde_json::Value = serde_json::from_str(&raw).with_context(|| {
        format!(
            "{} is not valid JSON; fix or move it and re-run (it was left unchanged)",
            settings_path.display()
        )
    })?;
    if !settings.is_object() {
        anyhow::bail!(
            "{} does not hold a JSON object; fix or move it and re-run (it was left unchanged)",
            settings_path.display()
        );
    }
    Ok(settings)
}

/// Copy the current settings file aside before it is rewritten. The first
/// backup ever taken is kept as-is; a second backup is refreshed on every run.
fn backup_settings_before_rewrite(claude_dir: &Path, settings_path: &Path) -> anyhow::Result<()> {
    if !settings_path.exists() {
        return Ok(());
    }
    let first = claude_dir.join("settings.json.bak.pre-sandwich");
    let latest = claude_dir.join("settings.json.bak.last-sandwich");
    let mut targets = vec![latest];
    if !first.exists() {
        targets.push(first);
    }
    for target in targets {
        fs::copy(settings_path, &target)
            .with_context(|| format!("failed to back up to {}", target.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&target)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&target, perms)?;
        }
    }
    Ok(())
}

fn install_sandwich_from_source(
    source_root: &Path,
    options: &SandwichInstallOptions,
) -> anyhow::Result<()> {
    let home = home_dir()?;
    let claude_dir = home.join(".claude");
    let hooks_dir = claude_dir.join("hooks");
    let agents_dir = claude_dir.join("agents");
    let settings_path = claude_dir.join("settings.json");
    let source_root =
        find_sandwich_source_root(source_root).unwrap_or_else(|| source_root.to_path_buf());

    if !source_root.join("hooks").is_dir() || !source_root.join("agents").is_dir() {
        anyhow::bail!(
            "Cognitive Sandwich source missing hooks/ or agents/: {}",
            source_root.display()
        );
    }

    let enable_preflight = options.enable_preflight || options.enable_sandwich;
    let mut enable_sanhedrin =
        options.enable_sanhedrin || options.enable_sandwich || options.with_launchd;
    let mut with_launchd = options.with_launchd;

    if with_launchd && (env::consts::OS != "macos" || env::consts::ARCH != "aarch64") {
        println!(
            "{}",
            "--with-launchd is Apple Silicon only; using endpoint-backed Sanhedrin instead."
                .yellow()
        );
        with_launchd = false;
        enable_sanhedrin = true;
    }

    // Read the settings first: an unreadable or unparseable file stops the
    // install before anything on disk has changed.
    let mut settings = read_settings_for_update(&settings_path)?;

    fs::create_dir_all(&claude_dir)?;
    let (hooks_copied, hooks_skipped) = copy_companion_files(
        &source_root.join("hooks"),
        &hooks_dir,
        &["sh", "py"],
        0o755,
        options,
    )?;
    let (json_copied, json_skipped) = copy_companion_files(
        &source_root.join("hooks"),
        &hooks_dir,
        &["json"],
        0o644,
        options,
    )?;
    let (agents_copied, agents_skipped) = copy_companion_files(
        &source_root.join("agents"),
        &agents_dir,
        &["md"],
        0o644,
        options,
    )?;

    println!(
        "{}: {} installed, {} skipped",
        "Hooks".white().bold(),
        hooks_copied + json_copied,
        hooks_skipped + json_skipped
    );
    println!(
        "{}: {} installed, {} skipped",
        "Agents".white().bold(),
        agents_copied,
        agents_skipped
    );

    if !with_launchd {
        remove_legacy_launchd_job(&home);
    }

    let dashboard_port = env::var("VESTIGE_DASHBOARD_PORT").unwrap_or_else(|_| "3927".to_string());
    let mut endpoint = options
        .sanhedrin_endpoint
        .clone()
        .or_else(|| env::var("VESTIGE_SANHEDRIN_ENDPOINT").ok())
        .or_else(|| env::var("MLX_ENDPOINT").ok())
        .unwrap_or_default()
        .trim_end_matches('/')
        .to_string();
    let mut model = options
        .sanhedrin_model
        .clone()
        .or_else(|| env::var("VESTIGE_SANHEDRIN_MODEL").ok())
        .or_else(|| env::var("VESTIGE_SANDWICH_MODEL").ok())
        .unwrap_or_default();
    if with_launchd {
        if endpoint.is_empty() {
            endpoint = "http://127.0.0.1:8080/v1/chat/completions".to_string();
        }
        if model.is_empty() {
            model = "mlx-community/Qwen3.6-35B-A3B-4bit".to_string();
        }
    }

    if enable_sanhedrin {
        if endpoint.is_empty() || model.is_empty() {
            println!(
                "{}",
                "Sanhedrin enabled without a verifier model; it will fail open until VESTIGE_SANHEDRIN_ENDPOINT and VESTIGE_SANHEDRIN_MODEL are set."
                    .yellow()
            );
        }
        write_sanhedrin_env(&hooks_dir, &endpoint, &model, &dashboard_port)?;
    }
    if with_launchd {
        install_launchd_job(&source_root, &home, &model)?;
    }

    backup_settings_before_rewrite(&claude_dir, &settings_path)?;
    scrub_vestige_hooks(&mut settings);

    if enable_preflight {
        merge_settings_fragment(
            &mut settings,
            &source_root
                .join("hooks")
                .join("settings.preflight.fragment.json"),
        )?;
    }
    if enable_sanhedrin {
        merge_settings_fragment(
            &mut settings,
            &source_root
                .join("hooks")
                .join("settings.sanhedrin.fragment.json"),
        )?;
    }

    let mut rendered = serde_json::to_vec_pretty(&settings)?;
    rendered.push(b'\n');
    fs::write(&settings_path, rendered)
        .with_context(|| format!("failed to write {}", settings_path.display()))?;

    if enable_preflight || enable_sanhedrin {
        let mut layers = Vec::new();
        if enable_preflight {
            layers.push("preflight");
        }
        if enable_sanhedrin {
            layers.push("sanhedrin");
        }
        println!(
            "{}: enabled optional layer(s): {}",
            "Settings".white().bold(),
            layers.join(", ")
        );
    } else {
        println!(
            "{}: no Vestige Claude Code hooks enabled by default",
            "Settings".white().bold()
        );
    }

    Ok(())
}

fn run_sandwich_install(
    version: Option<&str>,
    options: &SandwichInstallOptions,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "=== Vestige Cognitive Sandwich Install ===".cyan().bold()
    );
    println!();

    if let Some(source_root) = &options.src {
        install_sandwich_from_source(source_root, options)?;
    } else {
        let temp_dir = UpdateTempDir::create()?;
        let source_root = download_sandwich_source(version, &temp_dir.path)?;
        install_sandwich_from_source(&source_root, options)?;
    }

    println!();
    let optional_layers_enabled = options.enable_preflight
        || options.enable_sandwich
        || options.enable_sanhedrin
        || options.with_launchd;
    let message = if optional_layers_enabled {
        "Cognitive Sandwich files updated. Restart Claude Code to use enabled optional hooks."
    } else {
        "Cognitive Sandwich files updated. No hooks enabled; no automatic model calls."
    };
    println!("{}", message.green().bold());
    Ok(())
}

fn run_command(command: &mut Command, action: &str) -> anyhow::Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to start {}", action))?;
    if !status.success() {
        anyhow::bail!("{} failed with status {}", action, status);
    }
    Ok(())
}

fn create_private_file(path: &Path) -> std::io::Result<fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
    }

    #[cfg(not(unix))]
    {
        fs::File::create(path)
    }
}

fn command_output(command: &mut Command, action: &str) -> anyhow::Result<String> {
    let output = command
        .output()
        .with_context(|| format!("failed to start {}", action))?;
    if !output.status.success() {
        anyhow::bail!("{} failed with status {}", action, output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn powershell_quote(value: &Path) -> String {
    format!("'{}'", value.display().to_string().replace('\'', "''"))
}

fn normalize_archive_entry(entry: &str) -> anyhow::Result<String> {
    let normalized = entry.trim().replace('\\', "/");
    let normalized = normalized.strip_prefix("./").unwrap_or(&normalized);
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.get(1..2) == Some(":")
        || normalized
            .split('/')
            .any(|part| part.is_empty() || part == "..")
    {
        anyhow::bail!("archive contains unsafe entry: {}", entry);
    }
    Ok(normalized.to_string())
}

fn archive_listing(archive_path: &Path, archive_ext: &str) -> anyhow::Result<String> {
    let listing = match archive_ext {
        "tar.gz" => command_output(
            Command::new("tar").arg("-tzf").arg(archive_path),
            "listing Vestige archive with tar",
        )?,
        "zip" => {
            let script = format!(
                "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                 $zip = [System.IO.Compression.ZipFile]::OpenRead({}); \
                 try {{ $zip.Entries | ForEach-Object {{ $_.FullName }} }} finally {{ $zip.Dispose() }}",
                powershell_quote(archive_path)
            );
            command_output(
                Command::new("powershell")
                    .arg("-NoProfile")
                    .arg("-Command")
                    .arg(script),
                "listing Vestige archive with PowerShell",
            )?
        }
        other => anyhow::bail!("unsupported release archive extension: {}", other),
    };
    Ok(listing)
}

fn validate_archive_safety(archive_path: &Path, archive_ext: &str) -> anyhow::Result<()> {
    let listing = archive_listing(archive_path, archive_ext)?;
    for entry in listing.lines().filter(|line| !line.trim().is_empty()) {
        normalize_archive_entry(entry)?;
    }
    Ok(())
}

fn validate_archive_entries(
    archive_path: &Path,
    archive_ext: &str,
    expected_members: &[String],
) -> anyhow::Result<()> {
    let listing = archive_listing(archive_path, archive_ext)?;

    let expected: HashSet<&str> = expected_members.iter().map(String::as_str).collect();
    for entry in listing.lines().filter(|line| !line.trim().is_empty()) {
        let normalized = normalize_archive_entry(entry)?;
        if !expected.contains(normalized.as_str()) {
            anyhow::bail!("release archive contains unexpected entry: {}", entry);
        }
    }
    Ok(())
}

fn extract_source_archive(archive_path: &Path, output_dir: &Path) -> anyhow::Result<()> {
    validate_archive_safety(archive_path, "tar.gz")?;
    run_command(
        Command::new("tar")
            .arg("-xzf")
            .arg(archive_path)
            .arg("-C")
            .arg(output_dir),
        "extracting Vestige source archive with tar",
    )
}

fn extract_archive(
    archive_path: &Path,
    output_dir: &Path,
    archive_ext: &str,
    expected_members: &[String],
) -> anyhow::Result<()> {
    validate_archive_entries(archive_path, archive_ext, expected_members)?;
    match archive_ext {
        "tar.gz" => run_command(
            Command::new("tar")
                .arg("-xzf")
                .arg(archive_path)
                .arg("-C")
                .arg(output_dir),
            "extracting Vestige release archive with tar",
        ),
        "zip" => run_command(
            Command::new("powershell")
                .arg("-NoProfile")
                .arg("-Command")
                .arg(format!(
                    "Expand-Archive -LiteralPath {} -DestinationPath {} -Force",
                    powershell_quote(archive_path),
                    powershell_quote(output_dir)
                )),
            "extracting Vestige release archive with PowerShell",
        ),
        other => anyhow::bail!("unsupported release archive extension: {}", other),
    }
}

fn replace_binary(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid destination path {}", destination.display()))?;
    let temp_destination = destination.with_file_name(format!(
        ".{}.vestige-update-{}",
        file_name,
        std::process::id()
    ));

    fs::copy(source, &temp_destination).with_context(|| {
        format!(
            "failed to stage {} for install at {}",
            source.display(),
            temp_destination.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&temp_destination)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&temp_destination, perms)?;
    }

    #[cfg(windows)]
    if destination.exists() {
        fs::remove_file(destination).with_context(|| {
            format!(
                "failed to replace {}. Close running Vestige processes and retry",
                destination.display()
            )
        })?;
    }

    fs::rename(&temp_destination, destination).with_context(|| {
        let _ = fs::remove_file(&temp_destination);
        format!(
            "failed to install {}. If this is a system directory, retry with: sudo vestige update",
            destination.display()
        )
    })?;

    Ok(())
}

fn run_update(
    version: Option<String>,
    install_dir: Option<PathBuf>,
    dry_run: bool,
    no_sandwich: bool,
    sandwich_companion: bool,
    sandwich: SandwichInstallOptions,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Update ===".cyan().bold());
    println!();

    let asset = current_release_asset()?;
    let current_exe = env::current_exe().context("failed to locate current vestige executable")?;
    let install_dir = match install_dir {
        Some(path) => path,
        None => current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("current executable has no parent directory"))?
            .to_path_buf(),
    };

    let url = release_download_url(asset, version.as_deref());
    let archive_name = format!("vestige-mcp-{}.{}", asset.target, asset.archive_ext);

    println!(
        "{}: {}",
        "Current version".white().bold(),
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "{}: {}",
        "Release".white().bold(),
        version.as_deref().unwrap_or("latest")
    );
    println!("{}: {}", "Target".white().bold(), asset.target);
    println!(
        "{}: {}",
        "Install dir".white().bold(),
        install_dir.display()
    );
    println!("{}: {}", "Download".white().bold(), url);

    if dry_run {
        println!();
        println!("{}", "Dry run: no files changed.".yellow().bold());
        return Ok(());
    }

    fs::create_dir_all(&install_dir).with_context(|| {
        format!(
            "failed to create install directory {}",
            install_dir.display()
        )
    })?;

    let temp_dir = UpdateTempDir::create()?;
    let archive_path = temp_dir.path.join(&archive_name);
    let checksum_path = temp_dir.path.join(format!("{}.sha256", archive_name));

    println!();
    println!("{}", "Downloading release archive...".cyan());
    download_file(&url, &archive_path, "downloading Vestige release archive")?;
    download_file(
        &format!("{}.sha256", url),
        &checksum_path,
        "downloading Vestige release checksum",
    )?;
    verify_release_checksum(&archive_path, &checksum_path)?;

    let binaries = [
        "vestige",
        "vestige-mcp",
        "vestige-restore",
        "vestige-upgrade",
    ];
    let expected_members = binaries
        .iter()
        .map(|binary| format!("{}{}", binary, asset.binary_suffix))
        .collect::<Vec<_>>();

    println!("{}", "Extracting release archive...".cyan());
    extract_archive(
        &archive_path,
        &temp_dir.path,
        asset.archive_ext,
        &expected_members,
    )?;

    for binary in binaries {
        let filename = format!("{}{}", binary, asset.binary_suffix);
        let source = temp_dir.path.join(&filename);
        if !source.exists() {
            anyhow::bail!("release archive is missing expected binary: {}", filename);
        }

        let destination = install_dir.join(&filename);
        println!("  {} {}", "install".dimmed(), destination.display());
        replace_binary(&source, &destination)?;
    }

    println!();
    let installed_mcp = install_dir.join(format!("vestige-mcp{}", asset.binary_suffix));
    if let Ok(output) = Command::new(&installed_mcp).arg("--version").output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !version.is_empty() {
            println!("{}: {}", "Installed".white().bold(), version.green());
        }
    }

    println!(
        "{}",
        "Binary update complete. Restart your MCP client to pick up the new binary."
            .green()
            .bold()
    );

    if sandwich_companion && !no_sandwich {
        println!();
        println!(
            "{}",
            "Updating Cognitive Sandwich companion files...".cyan()
        );
        run_sandwich_install(version.as_deref(), &sandwich)?;
    } else if no_sandwich {
        println!(
            "{}",
            "Skipped Cognitive Sandwich companion update (--no-sandwich).".yellow()
        );
    } else {
        println!(
            "{}",
            "Skipped Cognitive Sandwich companion update (default). Pass --sandwich-companion to refresh Claude Code companion files."
                .yellow()
        );
    }

    Ok(())
}

/// Run stats command
fn run_stats(show_tagging: bool, show_states: bool) -> anyhow::Result<()> {
    let storage = open_storage()?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Memory Statistics ===".cyan().bold());
    println!();

    // Basic stats
    println!("{}: {}", "Total Memories".white().bold(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white().bold(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white().bold(),
        stats.average_retention * 100.0
    );
    println!(
        "{}: {:.2}",
        "Average Storage Strength".white().bold(),
        stats.average_storage_strength
    );
    println!(
        "{}: {:.2}",
        "Average Retrieval Strength".white().bold(),
        stats.average_retrieval_strength
    );
    println!(
        "{}: {}",
        "With Embeddings".white().bold(),
        stats.nodes_with_embeddings
    );

    if let Some(model) = &stats.active_embedding_model {
        println!("{}: {}", "Active Embedding Model".white().bold(), model);
    }

    if let Some(model) = &stats.embedding_model {
        println!("{}: {}", "Stored Embedding Model".white().bold(), model);
    }

    if stats.nodes_with_mismatched_embeddings > 0 {
        println!(
            "{}: {}",
            "Mismatched Embeddings".white().bold(),
            stats.nodes_with_mismatched_embeddings
        );
    }

    if let Some(oldest) = stats.oldest_memory {
        println!(
            "{}: {}",
            "Oldest Memory".white().bold(),
            oldest.format("%Y-%m-%d %H:%M:%S")
        );
    }
    if let Some(newest) = stats.newest_memory {
        println!(
            "{}: {}",
            "Newest Memory".white().bold(),
            newest.format("%Y-%m-%d %H:%M:%S")
        );
    }

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{}: {:.1}%",
        "Active Embedding Coverage".white().bold(),
        embedding_coverage
    );

    // Tagging distribution (retention levels)
    if show_tagging {
        println!();
        println!("{}", "=== Retention Distribution ===".yellow().bold());

        // Every memory, not the first page: a sample would misstate the
        // distribution on a store of thousands.
        let memories = fetch_all_nodes(&storage)?;
        let total = memories.len();

        if total > 0 {
            let high = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.7)
                .count();
            let medium = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7)
                .count();
            let low = memories
                .iter()
                .filter(|m| m.retention_strength < 0.4)
                .count();

            print_distribution_bar("High (>=70%)", high, total, "green");
            print_distribution_bar("Medium (40-70%)", medium, total, "yellow");
            print_distribution_bar("Low (<40%)", low, total, "red");
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    // State distribution
    if show_states {
        println!();
        println!(
            "{}",
            "=== Cognitive State Distribution ===".magenta().bold()
        );

        // Every memory, not the first page: a sample would misstate the
        // distribution on a store of thousands.
        let memories = fetch_all_nodes(&storage)?;
        let total = memories.len();

        if total > 0 {
            let (active, dormant, silent, unavailable) = compute_state_distribution(&memories);

            print_distribution_bar("Active", active, total, "green");
            print_distribution_bar("Dormant", dormant, total, "yellow");
            print_distribution_bar("Silent", silent, total, "red");
            print_distribution_bar("Unavailable", unavailable, total, "magenta");

            println!();
            println!("{}", "State Thresholds:".dimmed());
            println!("  {} >= 0.70 accessibility", "Active".green());
            println!("  {} >= 0.40 accessibility", "Dormant".yellow());
            println!("  {} >= 0.10 accessibility", "Silent".red());
            println!("  {} < 0.10 accessibility", "Unavailable".magenta());
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    Ok(())
}

/// Compute cognitive state distribution for memories
fn compute_state_distribution(
    memories: &[vestige_core::KnowledgeNode],
) -> (usize, usize, usize, usize) {
    let mut active = 0;
    let mut dormant = 0;
    let mut silent = 0;
    let mut unavailable = 0;

    for memory in memories {
        // Accessibility = 0.5*retention + 0.3*retrieval + 0.2*storage
        let accessibility = memory.retention_strength * 0.5
            + memory.retrieval_strength * 0.3
            + memory.storage_strength * 0.2;

        if accessibility >= 0.7 {
            active += 1;
        } else if accessibility >= 0.4 {
            dormant += 1;
        } else if accessibility >= 0.1 {
            silent += 1;
        } else {
            unavailable += 1;
        }
    }

    (active, dormant, silent, unavailable)
}

/// Print a distribution bar
fn print_distribution_bar(label: &str, count: usize, total: usize, color: &str) {
    let percentage = if total > 0 {
        (count as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let bar_width: usize = 30;
    let filled = ((percentage / 100.0) * bar_width as f64) as usize;
    let empty = bar_width.saturating_sub(filled);

    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(empty));
    let colored_bar = match color {
        "green" => bar.green(),
        "yellow" => bar.yellow(),
        "red" => bar.red(),
        "magenta" => bar.magenta(),
        _ => bar.white(),
    };

    println!(
        "  {:15} [{:30}] {:>4} ({:>5.1}%)",
        label, colored_bar, count, percentage
    );
}

/// Run health check
fn run_health() -> anyhow::Result<()> {
    let storage = open_storage()?;
    let stats = storage.get_stats()?;
    // A Strata log has no embeddings, no keyword search and a no-op
    // consolidation in 4.0, so none of the v3 embedding or consolidation
    // advice applies to it.
    let strata = is_strata(&storage);

    println!("{}", "=== Vestige Health Check ===".cyan().bold());
    println!();

    // Determine health status
    let (status, status_color) = if stats.total_nodes == 0 {
        ("EMPTY", "white")
    } else if stats.average_retention < 0.3 {
        ("CRITICAL", "red")
    } else if stats.average_retention < 0.5 {
        ("DEGRADED", "yellow")
    } else {
        ("HEALTHY", "green")
    };

    let colored_status = match status_color {
        "green" => status.green().bold(),
        "yellow" => status.yellow().bold(),
        "red" => status.red().bold(),
        _ => status.white().bold(),
    };

    println!("{}: {}", "Status".white().bold(), colored_status);
    println!("{}: {}", "Total Memories".white(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white(),
        stats.average_retention * 100.0
    );

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    if strata {
        println!(
            "{}: {}",
            "Retrieval".white(),
            "exact handles only (Strata log: no embeddings, no keyword search)".yellow()
        );
    } else {
        println!(
            "{}: {:.1}%",
            "Active Embedding Coverage".white(),
            embedding_coverage
        );
        // w1b: the embedding runtime was removed; keyword search is the only
        // engine. Saying anything else here would read as a store-health
        // verdict (issue #191) when it is a statement about the build.
        println!(
            "{}: {}",
            "Embedding Service".white(),
            "removed from this build (keyword search only)".yellow()
        );
    }

    // Warnings
    let mut warnings = Vec::new();

    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings.push(if strata {
            "Low average retention - review the memories you still need"
        } else {
            "Low average retention - consider running consolidation or reviewing memories"
        });
    }

    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }

    // Embedding warnings describe a SQLite store; a Strata log never has
    // vectors to cover.
    if !strata {
        if stats.total_nodes > 0 && stats.nodes_with_active_embeddings == 0 {
            warnings.push(if vestige_mcp::embeddings_compiled_in() {
                "No active-model embeddings generated - semantic search unavailable"
            } else {
                "Built without embeddings - semantic search is unavailable in this build by construction"
            });
        }

        if embedding_coverage < 50.0 && stats.total_nodes > 10 {
            warnings.push("Low embedding coverage - run consolidation to improve semantic search");
        }

        if stats.nodes_with_mismatched_embeddings > 0 {
            warnings.push("Stored embeddings from another model are present - run consolidation after changing embedding models");
        }
    }

    if !warnings.is_empty() {
        println!();
        println!("{}", "Warnings:".yellow().bold());
        for warning in &warnings {
            println!("  {} {}", "!".yellow().bold(), warning.yellow());
        }
    }

    // Recommendations
    let mut recommendations = Vec::new();

    if status == "CRITICAL" {
        recommendations
            .push("CRITICAL: Many memories have very low retention. Review important memories.");
    }

    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }

    // Consolidation is a no-op on a Strata log in 4.0, so recommending it
    // there would promise work that does not happen.
    if !strata && stats.nodes_with_active_embeddings < stats.total_nodes {
        recommendations
            .push("Run 'vestige consolidate' to generate active-model embeddings for better semantic search.");
    }

    if !strata && stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider running periodic consolidation to maintain memory health.");
    }

    if recommendations.is_empty() && status == "HEALTHY" {
        recommendations.push("Memory system is healthy!");
    }

    println!();
    println!("{}", "Recommendations:".cyan().bold());
    for rec in &recommendations {
        let icon = if rec.starts_with("CRITICAL") {
            "!".red().bold()
        } else {
            ">".cyan()
        };
        let text = if rec.starts_with("CRITICAL") {
            rec.red().to_string()
        } else {
            rec.to_string()
        };
        println!("  {} {}", icon, text);
    }

    Ok(())
}

/// Data directory for this invocation (`--data-dir`, then `VESTIGE_DATA_DIR`,
/// then the platform directory). The Strata log lives here. `vestige.db` is
/// only the v3 file we refuse to open.
fn cli_data_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = CLI_DATA_DIR.get() {
        return Ok(path.clone());
    }
    if let Some(value) = std::env::var_os("VESTIGE_DATA_DIR")
        && !value.is_empty()
    {
        return Ok(expand_tilde(PathBuf::from(value)));
    }
    let proj = directories::ProjectDirs::from("com", "vestige", "core")
        .ok_or_else(|| anyhow::anyhow!("Could not determine project directories"))?;
    Ok(proj.data_dir().to_path_buf())
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let rest = {
        let mut components = path.components();
        match components.next() {
            Some(std::path::Component::Normal(first)) if first == "~" => {
                Some(components.as_path().to_path_buf())
            }
            _ => None,
        }
    };
    match rest {
        Some(rest) => directories::BaseDirs::new()
            .map(|dirs| dirs.home_dir().join(rest))
            .unwrap_or(path),
        None => path,
    }
}

/// The v3 SQLite file that would have lived in the data directory.
fn cli_db_path() -> anyhow::Result<PathBuf> {
    Ok(cli_data_dir()?.join("vestige.db"))
}

/// Import a v3 `vestige.db` by running `vestige-upgrade`. The file is not opened.
fn run_upgrade(dry_run: bool) -> anyhow::Result<()> {
    let source = cli_db_path()?;
    if !vestige_mcp::v3_launch::db_present(&source) {
        anyhow::bail!("no store at {} (nothing to upgrade)", source.display());
    }
    // The v3 file stays after a successful upgrade. Once the Strata log is
    // published it is the live store and vestige-upgrade is not run again,
    // so neither mode may report an import.
    if vestige_mcp::v3_launch::strata_log_published(&source) {
        println!(
            "Already upgraded: the Strata log in {} is the live store. {} is kept byte-identical and is not read; nothing to import.",
            cli_data_dir()?.join("log").display(),
            source.display()
        );
        return Ok(());
    }
    if dry_run {
        println!(
            "Dry run: vestige-upgrade would import {} and leave that file byte-identical.",
            source.display()
        );
        return Ok(());
    }
    vestige_mcp::v3_launch::upgrade_or_refuse(&source)?;
    println!(
        "{} vestige.db was not modified.",
        "Upgraded.".green().bold()
    );
    Ok(())
}

/// Run consolidation cycle
fn run_consolidate() -> anyhow::Result<()> {
    println!("{}", "=== Vestige Consolidation ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    if is_strata(&storage) {
        println!(
            "{}",
            "Strata log: consolidation is a no-op in Vestige 4.0. No decay, promotion, pruning, dedup or embedding pass runs on the log, so nothing was changed."
                .yellow()
        );
        return Ok(());
    }

    println!("Running memory consolidation cycle...");
    println!();
    let result = storage.run_consolidation()?;

    println!(
        "{}: {}",
        "Nodes Processed".white().bold(),
        result.nodes_processed
    );
    println!(
        "{}: {}",
        "Nodes Promoted".white().bold(),
        result.nodes_promoted
    );
    println!("{}: {}", "Nodes Pruned".white().bold(), result.nodes_pruned);
    println!(
        "{}: {}",
        "Duplicates Merged".white().bold(),
        result.duplicates_merged
    );
    println!(
        "{}: {}",
        "Decay Applied".white().bold(),
        result.decay_applied
    );
    println!(
        "{}: {}",
        "Embeddings Generated".white().bold(),
        result.embeddings_generated
    );
    println!("{}: {}ms", "Duration".white().bold(), result.duration_ms);
    if result.embeddings_generated == 0 {
        // w1b: the embedding runtime was removed; there are no vectors to generate.
        println!(
            "  {}",
            "(this build has no embedding runtime, so there are no vectors to generate)".yellow()
        );
    }

    println!();
    println!(
        "{}",
        format!(
            "Consolidation complete: {} nodes processed, {} embeddings generated in {}ms",
            result.nodes_processed, result.embeddings_generated, result.duration_ms
        )
        .green()
    );

    Ok(())
}

/// Run restore from backup
fn run_restore(backup_path: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Restore ===".cyan().bold());
    println!();
    println!("Loading backup from: {}", backup_path.display());

    // A Strata backup (`vestige backup`) is a directory. Restoring one means
    // replacing the live log while Vestige is stopped, which this command
    // must not do behind a running server.
    let meta = std::fs::metadata(&backup_path)
        .with_context(|| format!("cannot read {}", backup_path.display()))?;
    if meta.is_dir() {
        if backup_path.join("log").is_dir() {
            anyhow::bail!(
                "unavailable_in_4_0: restore does not load a Strata backup directory. {} is a copy of a Strata log: stop Vestige, then copy its log/ (and store.meta, when present) into the data directory in place of the existing log/.",
                backup_path.display()
            );
        }
        anyhow::bail!(
            "{} is a directory, not a JSON restore file",
            backup_path.display()
        );
    }

    let storage = open_storage()?;
    let strata = is_strata(&storage);

    // Read and parse backup
    let backup_bytes = std::fs::read(&backup_path)?;
    if backup_bytes.starts_with(b"SQLite format 3\0") {
        if strata {
            anyhow::bail!(
                "unavailable_in_4_0: {} is a v3 SQLite database, and restore does not write SQLite into a Strata log. To import it, save it as vestige.db in a data directory that has no log/ yet and run `vestige --data-dir <that dir> upgrade` (it runs vestige-upgrade and leaves the file unchanged).",
                backup_path.display()
            );
        }
        anyhow::bail!(
            "{} is a raw SQLite database backup, not a JSON restore file. Use portable-export/portable-import for cross-device transfer, or replace the database file manually while Vestige is stopped.",
            backup_path.display()
        );
    }
    let backup_content = String::from_utf8(backup_bytes)
        .with_context(|| format!("{} is not UTF-8 JSON", backup_path.display()))?;

    if let Ok(archive) = serde_json::from_str::<vestige_core::PortableArchive>(&backup_content)
        && archive.archive_format == vestige_core::PORTABLE_ARCHIVE_FORMAT
    {
        if strata {
            anyhow::bail!(
                "unavailable_in_4_0: restore of a portable archive is not available on Strata in Vestige 4.0: a Strata log does not read portable archives yet. {STRATA_EXACT_RESTORE_HINT}"
            );
        }
        println!("Detected portable archive.");
        println!("{}: {}", "Format".white().bold(), archive.archive_format);
        println!("{}: {}", "Schema".white().bold(), archive.schema_version);
        println!("{}: {}", "Tables".white().bold(), archive.tables.len());
        println!("{}: {}", "Rows".white().bold(), archive.total_rows());
        println!();

        let report = storage.import_portable_archive(&archive, PortableImportMode::EmptyOnly)?;

        println!(
            "{}: {}",
            "Tables imported".white().bold(),
            report.tables_imported
        );
        println!(
            "{}: {}",
            "Rows imported".white().bold(),
            report.rows_imported
        );
        println!(
            "{}: {}",
            "Tables skipped".white().bold(),
            report.tables_skipped
        );
        println!("{}: {}", "FTS rebuilt".white().bold(), report.fts_rebuilt);
        println!();
        println!("{}", "Portable restore complete.".green().bold());
        return Ok(());
    }

    #[derive(serde::Deserialize)]
    struct BackupWrapper {
        #[serde(rename = "type")]
        _type: String,
        text: String,
    }

    #[derive(serde::Deserialize)]
    struct RecallResult {
        results: Vec<MemoryBackup>,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MemoryBackup {
        content: String,
        node_type: Option<String>,
        tags: Option<Vec<String>>,
        source: Option<String>,
    }

    let memories = if let Ok(wrapper) = serde_json::from_str::<Vec<BackupWrapper>>(&backup_content)
    {
        let first = wrapper.first().context("backup wrapper is empty")?;
        let recall_result: RecallResult = serde_json::from_str(&first.text)?;
        recall_result.results
    } else if let Ok(recall_result) = serde_json::from_str::<RecallResult>(&backup_content) {
        recall_result.results
    } else if let Ok(memories) = serde_json::from_str::<Vec<MemoryBackup>>(&backup_content) {
        memories
    } else {
        anyhow::bail!(
            "Unrecognized backup format. Expected portable archive, MCP wrapper, RecallResult, or array of memories."
        );
    };

    println!("Found {} memories to restore", memories.len());
    println!();
    if strata {
        println!(
            "{}",
            "Strata log: each memory is ingested as a new record (new id, created now, fresh review state). Ids, timestamps, review history and edges in the file are not restored."
                .yellow()
        );
        println!("{}", STRATA_EXACT_RESTORE_HINT.dimmed());
        println!();
    }

    // 4.0 builds have no embedding runtime: restore only ingests.
    println!("Ingesting memories...");
    println!();

    let total = memories.len();
    let mut success_count = 0;

    for (i, memory) in memories.into_iter().enumerate() {
        let input = IngestInput {
            content: memory.content.clone(),
            node_type: memory.node_type.unwrap_or_else(|| "fact".to_string()),
            source: memory.source,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: memory.tags.unwrap_or_default(),
            valid_from: None,
            valid_until: None,
            validity_inferred: false,
            source_envelope: None,
        };

        match storage.ingest(input) {
            Ok(_node) => {
                success_count += 1;
                println!(
                    "[{}/{}] {} {}",
                    i + 1,
                    total,
                    "OK".green(),
                    truncate(&memory.content, 60)
                );
            }
            Err(e) => {
                println!("[{}/{}] {} {}", i + 1, total, "FAIL".red(), e);
            }
        }
    }

    println!();
    println!(
        "Restore complete: {}/{} memories {}",
        success_count.to_string().green().bold(),
        total,
        if strata {
            "ingested as new records"
        } else {
            "restored"
        }
    );

    // Show stats
    let stats = storage.get_stats()?;
    println!();
    println!("{}: {}", "Total Nodes".white(), stats.total_nodes);
    if !strata {
        println!(
            "{}: {}",
            "Active Embeddings".white(),
            stats.nodes_with_active_embeddings
        );
    }

    Ok(())
}

/// How an exact Strata restore works: the backup is a directory copy of the
/// log, put back while no Vestige process holds the store.
const STRATA_EXACT_RESTORE_HINT: &str = "For an exact restore, stop Vestige and copy the log/ of a `vestige backup` directory (and its store.meta, when present) into the data directory in place of the existing log/.";

/// The durable store this invocation opened is a Strata log.
fn is_strata(storage: &Arc<Storage>) -> bool {
    vestige_mcp::strata_memory::is_strata_backend(storage.as_ref())
}

/// `path` made absolute with its deepest existing ancestor canonicalized, so
/// a destination that does not exist yet still compares against real paths.
fn resolve_existing_prefix(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut existing = absolute.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                missing.push(name.to_os_string());
                existing = parent;
            }
            _ => break,
        }
    }
    let mut resolved = std::fs::canonicalize(existing).unwrap_or_else(|_| existing.to_path_buf());
    for name in missing.into_iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

/// Total bytes under `path` (a file, or a directory walked recursively).
fn path_size(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| path_size(&entry.path()))
                .sum()
        })
        .unwrap_or(0)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Get the default database path
fn get_default_db_path() -> anyhow::Result<PathBuf> {
    cli_db_path()
}

/// Verify a STRATA directory. Same report as the `strata-verify` binary.
/// Creates nothing in `dir`. Log and receipt checks only. The v3 SQLite
/// comparison is `sqlite-reader`, which this path does not enable.
fn run_strata_verify(dir: PathBuf, expect_key: Option<String>) -> anyhow::Result<()> {
    let report = strata_verify::verify_path(&dir);
    println!("{}", report.json);
    if !report.key_fingerprint.is_empty() {
        println!("key fingerprint: {}", report.key_fingerprint);
    }
    let mut failed = !report.ok;
    if let Some(expected) = expect_key.as_deref() {
        let actual = report.key_fingerprint.to_ascii_lowercase();
        let expected = expected.trim().to_ascii_lowercase();
        if actual != expected {
            eprintln!("signing key fingerprint {actual} does not match --expect-key {expected}");
            failed = true;
        }
    }
    if !failed {
        println!("OK");
        return Ok(());
    }
    println!("FAILED");
    for failure in &report.failures {
        eprintln!("  {failure}");
    }
    std::process::exit(1);
}

/// Open storage using the CLI-selected data directory, if one was provided.
/// Read-only migration of a v3 SQLite store into a STRATA log. The source
/// is never opened read-write, migrated in place, or modified in any way;
/// its path and BLAKE3 are printed and keeping the file is recommended.
fn run_migrate_to_strata(
    from: PathBuf,
    to: Option<PathBuf>,
    dry_run: bool,
    accept_wal_snapshot: bool,
) -> anyhow::Result<()> {
    #[cfg(not(feature = "migrate-to-strata"))]
    {
        let _ = (from, to, dry_run, accept_wal_snapshot);
        anyhow::bail!(
            "migrate-to-strata is not linked into this binary; rebuild with --features migrate-to-strata"
        );
    }
    #[cfg(feature = "migrate-to-strata")]
    run_migrate_to_strata_linked(from, to, dry_run, accept_wal_snapshot)
}

#[cfg(feature = "migrate-to-strata")]
fn run_migrate_to_strata_linked(
    from: PathBuf,
    to: Option<PathBuf>,
    dry_run: bool,
    accept_wal_snapshot: bool,
) -> anyhow::Result<()> {
    let destination = match to {
        Some(dir) => dir,
        None => cli_db_path()?.with_file_name("strata"),
    };
    let options = strata_migrate::MigrateOptions {
        dry_run,
        accept_wal_snapshot,
        ..Default::default()
    };
    let report = strata_migrate::migrate_with_options(&from, &destination, options)?;

    println!("{}", "=== Vestige migrate-to-strata ===".cyan().bold());
    println!(
        "{} {} (keep this file; it is your pre-migration record)",
        "Source (never modified):".bold(),
        from.display()
    );
    if !report.source_blake3.is_empty() {
        println!("{} {}", "Source BLAKE3:".bold(), report.source_blake3);
    }
    println!("{} {}", "Destination:".bold(), destination.display());
    println!(
        "{} {} nodes, {} edges, {} fsrs events, {} fsrs cards",
        "Migrated:".green().bold(),
        report.nodes,
        report.edges,
        report.fsrs_events,
        report.fsrs_states
    );
    if report.dropped_vectors > 0 {
        println!(
            "{} {} (vector values were never read)",
            "Dropped vectors:".yellow().bold(),
            report.dropped_vectors
        );
    }
    if !report.skipped_tables.is_empty() {
        println!(
            "{} {}",
            "Skipped tables (counted, not mapped):".bold(),
            report.skipped_tables.join(", ")
        );
    }
    if dry_run {
        println!("Dry run: nothing was written.");
        return Ok(());
    }
    println!(
        "{} {} (signature verified: {})",
        "MIGRATION_RECEIPT:".green().bold(),
        report.receipt_digest.clone().unwrap_or_default(),
        report.receipt_verified
    );
    println!("{} {}", "Replay verification:".bold(), report.verify_passed);
    if !report.verify_passed || !report.receipt_verified {
        anyhow::bail!("migration log failed verification");
    }
    Ok(())
}

fn open_storage() -> anyhow::Result<std::sync::Arc<Storage>> {
    let dir = cli_data_dir()?;
    // Same check `vestige-mcp` runs before stdio. `vestige.db` is not opened.
    // It runs before the lock: the upgrade helper takes that lock itself.
    vestige_mcp::v3_launch::upgrade_or_refuse(&dir.join("vestige.db"))?;
    if !take_cli_lock(&dir)? {
        return Err(served_elsewhere(&dir));
    }
    Ok(vestige_mcp::strata_memory::open(&dir)?)
}

/// Take the store's serve lock for the rest of this process. `false` when a
/// Vestige server (or another command) holds it.
fn take_cli_lock(dir: &Path) -> anyhow::Result<bool> {
    if CLI_SERVE_LOCK.get().is_some() {
        return Ok(true);
    }
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create the data directory {}", dir.display()))?;
    match vestige_mcp::attach::try_serve_lock(dir)? {
        Some(lock) => {
            let _ = CLI_SERVE_LOCK.set(lock);
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Who holds the store, for messages. A pid is named only when that process
/// answers the attach handshake; the endpoint file can outlive it.
fn store_holder(dir: &Path) -> String {
    match vestige_mcp::attach::probe_owner_blocking(dir) {
        Some(pid) => format!("vestige-mcp (pid {pid})"),
        None => "another Vestige process (a vestige command, vestige-upgrade, or a server that is not answering)".to_string(),
    }
}

/// Why a command that opens the log cannot run while the store is served.
fn served_elsewhere(dir: &Path) -> anyhow::Error {
    let holder = store_holder(dir);
    anyhow::anyhow!(
        "{holder} is serving {}. This command opens the log directly, and the log has one \
         writer, so it runs only while no Vestige server holds the store. Use the matching \
         MCP tool through your agent, or stop the Vestige server and run it again. \
         `vestige backup` works while a server runs.",
        dir.display()
    )
}

/// Fetch all nodes from storage using pagination
fn fetch_all_nodes(storage: &Arc<Storage>) -> anyhow::Result<Vec<vestige_core::KnowledgeNode>> {
    let mut all_nodes = Vec::new();
    let page_size = 500;
    let mut offset = 0;

    loop {
        let batch = storage.get_all_nodes(page_size, offset)?;
        let batch_len = batch.len();
        all_nodes.extend(batch);
        if batch_len < page_size as usize {
            break;
        }
        offset += page_size;
    }

    Ok(all_nodes)
}

/// Back up the live store. A Strata log is copied as a directory through
/// `Storage::backup_to`, the same path as the MCP `maintain backup`; a
/// legacy SQLite store is written as one consistent snapshot file.
fn run_backup(output: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Backup ===".cyan().bold());
    println!();

    let data_dir = cli_data_dir()?;
    let db_path = get_default_db_path()?;
    // Opening creates an empty store, and a backup of that would hide a
    // wrong --data-dir behind a success line.
    if !data_dir.join("log").is_dir() && !vestige_mcp::v3_launch::db_present(&db_path) {
        anyhow::bail!(
            "no Vestige store in {}: no Strata log/ and no vestige.db (nothing to back up)",
            data_dir.display()
        );
    }

    // A running server holds the store: back up through it.
    vestige_mcp::v3_launch::upgrade_or_refuse(&db_path)?;
    if !take_cli_lock(&data_dir)? {
        return run_backup_through_server(&data_dir, &output);
    }

    let storage = open_storage()?;
    if is_strata(&storage) {
        return run_strata_backup(&storage, &data_dir, &output);
    }

    if !db_path.exists() {
        anyhow::bail!("Database not found at: {}", db_path.display());
    }

    // Create parent directories if needed
    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    // `VACUUM INTO` produces a transactionally consistent snapshot, including
    // committed frames that still live in the source WAL. Do not copy the main
    // database file directly: a busy checkpoint can otherwise omit those frames
    // while still making the backup command look successful.
    println!("Creating a consistent SQLite snapshot...");
    println!("  {} {}", "From:".dimmed(), db_path.display());
    println!("  {}   {}", "To:".dimmed(), output.display());
    storage.backup_to(&output)?;

    let size_display = format_size(std::fs::metadata(&output)?.len());

    println!();
    println!(
        "{}",
        format!("Backup complete: {} ({})", output.display(), size_display)
            .green()
            .bold()
    );

    Ok(())
}

/// Copy a Strata log into a new directory: the log is sealed, then every
/// log file except its lock (plus `store.meta`) is copied. `vestige.db` is
/// never part of it, even when the v3 file is still beside the log.
fn run_strata_backup(storage: &Arc<Storage>, data_dir: &Path, output: &Path) -> anyhow::Result<()> {
    check_strata_backup_destination(data_dir, output)?;
    let log_dir = data_dir.join("log");
    println!("Sealing and copying the Strata log...");
    println!("  {} {}", "From:".dimmed(), log_dir.display());
    println!("  {}   {}", "To:".dimmed(), output.display());
    storage
        .backup_to(output)
        .map_err(|err| anyhow::anyhow!("Strata backup failed: {err}"))?;
    print_strata_backup_summary(data_dir, output);
    Ok(())
}

/// The same backup, made by the `vestige-mcp` that serves the store (its
/// `maintain backup`, into `<data-dir>/backups`) and then moved to `output`.
fn run_backup_through_server(data_dir: &Path, output: &Path) -> anyhow::Result<()> {
    check_strata_backup_destination(data_dir, output)?;
    let holder = store_holder(data_dir);
    println!("{holder} holds this store; backing up through it...");
    let rt = tokio::runtime::Runtime::new()?;
    let made = rt
        .block_on(vestige_mcp::attach::call_tool(
            data_dir,
            "maintain",
            serde_json::json!({"action": "backup"}),
        ))
        .map_err(|err| {
            anyhow::anyhow!(
                "{err}. Stop the Vestige server and run `vestige backup` again, or ask your agent to run maintain backup."
            )
        })?;
    let made = made
        .get("path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("the server's backup answer named no path: {made}"))?;
    println!("  {} {}", "Made:".dimmed(), made.display());
    println!("  {}   {}", "To:".dimmed(), output.display());
    if made != output {
        // The destination check allows an empty directory; `rename` onto one
        // is not portable, so it goes first.
        if output.is_dir() {
            fs::remove_dir(output)?;
        }
        if fs::rename(&made, output).is_err() {
            // Another filesystem: copy into a sibling, sync it, then rename
            // it into place, so a failed copy never leaves a partial backup
            // at the requested path.
            let staged = output.with_file_name(format!(
                ".{}.partial-{}",
                output
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "backup".to_string()),
                std::process::id()
            ));
            let copied = copy_dir_all(&made, &staged).and_then(|()| fs::rename(&staged, output));
            if let Err(err) = copied {
                let _ = fs::remove_dir_all(&staged);
                anyhow::bail!(
                    "failed to copy the backup to {}: {err}. The server's copy is still at {}",
                    output.display(),
                    made.display()
                );
            }
            fs::remove_dir_all(&made).with_context(|| {
                format!("copied the backup, but could not remove {}", made.display())
            })?;
        }
    }
    print_strata_backup_summary(data_dir, output);
    Ok(())
}

/// Copy a directory tree (a Strata backup: plain files in plain directories).
///
/// The copy is owner-only on unix (directories 0700, files 0600), like the
/// backup it is copied from.
fn copy_dir_all(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(to, fs::Permissions::from_mode(0o700))?;
    }
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            let mut options = fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut out = options.open(&target)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                out.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            std::io::copy(&mut fs::File::open(entry.path())?, &mut out)?;
            out.sync_all()?;
        }
    }
    Ok(())
}

/// A Strata backup goes to a new (or empty) directory outside the live log.
fn check_strata_backup_destination(data_dir: &Path, output: &Path) -> anyhow::Result<()> {
    let log_dir = data_dir.join("log");
    if let Ok(meta) = std::fs::metadata(output) {
        let empty_dir = meta.is_dir() && std::fs::read_dir(output)?.next().is_none();
        if !empty_dir {
            anyhow::bail!(
                "{} already exists. A Strata backup is a new directory; pass a path that does not exist yet (or an empty directory) so no older files mix into the copy.",
                output.display()
            );
        }
    }
    // A destination inside the live log would be read back as log files
    // while they are being copied.
    let canonical_log = std::fs::canonicalize(&log_dir).unwrap_or_else(|_| log_dir.clone());
    if resolve_existing_prefix(output)?.starts_with(&canonical_log) {
        anyhow::bail!(
            "{} is inside the live log {}; pick a destination outside log/",
            output.display(),
            log_dir.display()
        );
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn print_strata_backup_summary(data_dir: &Path, output: &Path) {
    let files = std::fs::read_dir(output.join("log"))
        .map(|entries| entries.flatten().count())
        .unwrap_or(0);
    let size_display = format_size(path_size(output));

    println!();
    println!(
        "{}",
        format!(
            "Backup complete: {} ({}, {} files in log/)",
            output.display(),
            size_display,
            files
        )
        .green()
        .bold()
    );
    println!(
        "  {} log/{}",
        "Contains:".dimmed(),
        if output.join("store.meta").exists() {
            " and store.meta"
        } else {
            ""
        }
    );
    let db_path = data_dir.join("vestige.db");
    if vestige_mcp::v3_launch::db_present(&db_path) {
        println!(
            "  {} {} is the v3 file kept after the upgrade, not the live store; it was not copied.",
            "Not copied:".dimmed(),
            db_path.display()
        );
    }
    let keys: Vec<&str> = ["receipt-signing.key", "actor.key"]
        .into_iter()
        .filter(|name| data_dir.join(name).is_file())
        .collect();
    if !keys.is_empty() {
        println!(
            "  {} {} beside log/ in {} (keep them with the data directory).",
            "Not copied:".dimmed(),
            keys.join(", "),
            data_dir.display()
        );
    }
    println!(
        "  {} stop Vestige, then copy {} into the data directory in place of its log/.",
        "Restore:".dimmed(),
        output.join("log").display()
    );
}

/// Run the planted-cause selftest (the MCP `selftest` tool) from the CLI.
/// Snapshots the store to a tempdir copy, plants causes/failures there, runs
/// the real backfill against the copy, and prints the same payload the MCP
/// tool returns. The live store is only read.
fn run_selftest() -> anyhow::Result<()> {
    println!("{}", "=== Planted-Cause Selftest ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt
        .block_on(vestige_mcp::tools::selftest::execute(&storage, None))
        .map_err(|e| anyhow::anyhow!(e))?;

    if result["kind"] == "recorded_edge_walk" {
        // Strata payload: named checks over a planted recorded-edge chain,
        // not the backfill hit@k rounds a legacy SQLite store reports.
        let line = format!(
            "recorded-edge walk: {}/{} checks passed",
            result["checks_passed"], result["checks_total"]
        );
        if result["all_passed"] == true {
            println!("{}", line.green().bold());
        } else {
            println!("{}", line.yellow());
        }
    } else if result["hits"] == serde_json::json!(result["rounds"])
        && result["gap_calibration"] == true
    {
        println!(
            "{}",
            format!(
                "hit@1 {}/{} · hit@3 {}/{} · gap calibration OK",
                result["hits"], result["rounds"], result["hit_at_3"], result["rounds"]
            )
            .green()
            .bold()
        );
    } else {
        println!(
            "{}",
            format!(
                "hit@1 {}/{} · hit@3 {}/{} · gap calibration {}",
                result["hits"],
                result["rounds"],
                result["hit_at_3"],
                result["rounds"],
                result["gap_calibration"]
            )
            .yellow()
        );
    }
    if result["kind"] == "recorded_edge_walk" {
        println!(
            "{}",
            format!(
                "(live store touched: {}; temp log deleted: {})",
                result["live_store_touched"], result["temp_store_deleted"]
            )
            .dimmed()
        );
    } else {
        println!("{}", "(live store untouched; temp copy deleted)".dimmed());
    }
    println!();
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// Run the forgotten-lesson scan (the MCP `forgotten_lesson` tool) from the
/// CLI: decayed fix/lesson memories sharing an exact anchor with a failure.
fn run_forgotten_lesson(
    failure_id: String,
    scope: Option<String>,
    json: bool,
) -> anyhow::Result<()> {
    let storage = open_storage()?;
    let args = serde_json::json!({"failure_id": failure_id, "scope": scope});
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt
        .block_on(vestige_mcp::tools::forgotten_lesson::execute(
            &storage,
            Some(args),
        ))
        .map_err(|e| anyhow::anyhow!(e))?;

    // Machine-readable path: the raw tool payload, byte-for-byte the MCP shape.
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    println!("{}", "=== Forgotten Lessons ===".cyan().bold());
    println!();
    // A Strata log links lessons by recorded causal edges (`edge_path`); a
    // legacy SQLite store by a shared exact anchor (`shared_anchor`).
    let strata = is_strata(&storage);
    let lessons = result["forgotten_lessons"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if lessons.is_empty() {
        println!(
            "{}",
            if strata {
                "No decayed lesson is reached from this failure over recorded causal edges."
            } else {
                "No decayed lesson shares an anchor with this failure."
            }
            .dimmed()
        );
    } else {
        for lesson in &lessons {
            let link = if strata {
                let hops: Vec<String> = lesson["edge_path"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|hop| {
                        format!(
                            "{} -[{}]-> {}",
                            hop["source_id"].as_str().unwrap_or("?"),
                            hop["link_type"].as_str().unwrap_or("?"),
                            hop["target_id"].as_str().unwrap_or("?")
                        )
                    })
                    .collect();
                format!("edges {}", hops.join(", "))
            } else {
                format!("anchor {}", lesson["shared_anchor"].as_str().unwrap_or("?"))
            };
            println!(
                "  {} · retention {}% · {}",
                lesson["lesson_id"].as_str().unwrap_or("?").normal(),
                lesson["retention_pct"].to_string().yellow(),
                link.cyan(),
            );
            let preview = lesson["content_preview"].as_str().unwrap_or("");
            if !preview.is_empty() {
                println!("    {}", preview.dimmed());
            }
        }
        println!();
        println!(
            "{}",
            "Recorded fixes the store can no longer retrieve — review before relearning.".dimmed()
        );
    }
    Ok(())
}

/// Run export command - exports memories in JSON or JSONL format
fn run_export(
    output: PathBuf,
    format: String,
    tags: Option<String>,
    since: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Export ===".cyan().bold());
    println!();

    // Validate format
    if format != "json" && format != "jsonl" {
        anyhow::bail!("Invalid format '{}'. Must be 'json' or 'jsonl'.", format);
    }

    // Parse since date if provided
    let since_date = match &since {
        Some(date_str) => {
            let naive = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|e| {
                anyhow::anyhow!("Invalid date '{}': {}. Use YYYY-MM-DD format.", date_str, e)
            })?;
            Some(
                naive
                    .and_hms_opt(0, 0, 0)
                    .expect("midnight is always valid")
                    .and_utc(),
            )
        }
        None => None,
    };

    // Parse tags filter
    let tag_filter: Vec<String> = tags
        .as_deref()
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let storage = open_storage()?;
    let all_nodes = fetch_all_nodes(&storage)?;

    // Apply filters
    let filtered: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            // Date filter
            if let Some(ref since_dt) = since_date
                && node.created_at < *since_dt
            {
                return false;
            }
            // Tag filter: node must contain ALL specified tags
            if !tag_filter.is_empty() {
                for tag in &tag_filter {
                    if !node.tags.iter().any(|t| t == tag) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    println!("{}: {}", "Format".white().bold(), format);
    if !tag_filter.is_empty() {
        println!("{}: {}", "Tag filter".white().bold(), tag_filter.join(", "));
    }
    if let Some(ref date_str) = since {
        println!("{}: {}", "Since".white().bold(), date_str);
    }
    println!(
        "{}: {} / {} total",
        "Matching".white().bold(),
        filtered.len(),
        all_nodes.len()
    );
    println!();

    // Create parent directories if needed
    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = create_private_file(&output)?;
    let mut writer = BufWriter::new(file);

    match format.as_str() {
        "json" => {
            serde_json::to_writer_pretty(&mut writer, &filtered)?;
            writer.write_all(b"\n")?;
        }
        "jsonl" => {
            for node in &filtered {
                serde_json::to_writer(&mut writer, node)?;
                writer.write_all(b"\n")?;
            }
        }
        _ => unreachable!(),
    }

    writer.flush()?;

    let file_size = std::fs::metadata(&output)?.len();
    let size_display = if file_size >= 1024 * 1024 {
        format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
    } else if file_size >= 1024 {
        format!("{:.1} KB", file_size as f64 / 1024.0)
    } else {
        format!("{} bytes", file_size)
    };

    println!(
        "{}",
        format!(
            "Exported {} memories to {} ({}, {})",
            filtered.len(),
            output.display(),
            format,
            size_display
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Run exact portable archive export.
fn run_portable_export(output: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Portable Export ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    // Refuse before creating anything at the destination.
    if is_strata(&storage) {
        anyhow::bail!(
            "unavailable_in_4_0: portable-export is not available on Strata in Vestige 4.0: a Strata log does not write portable archives yet. {STRATA_PORTABLE_ALTERNATIVES}"
        );
    }

    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let archive = storage.export_portable_archive_to_path(&output)?;

    let size_display = format_size(std::fs::metadata(&output)?.len());

    println!("{}: {}", "Archive".white().bold(), output.display());
    println!("{}: {}", "Format".white().bold(), archive.archive_format);
    println!("{}: {}", "Schema".white().bold(), archive.schema_version);
    println!("{}: {}", "Tables".white().bold(), archive.tables.len());
    println!("{}: {}", "Rows".white().bold(), archive.total_rows());
    println!();
    println!(
        "{}",
        format!(
            "Portable export complete: {} ({})",
            output.display(),
            size_display
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Run exact portable archive import.
fn run_portable_import(input: PathBuf, merge: bool) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Portable Import ===".cyan().bold());
    println!();
    println!("{}: {}", "Archive".white().bold(), input.display());
    let mode = if merge {
        PortableImportMode::Merge
    } else {
        PortableImportMode::EmptyOnly
    };
    println!(
        "{}",
        if merge {
            "Mode: merge into existing database".yellow()
        } else {
            "Mode: empty database only".yellow()
        }
    );
    println!();

    let storage = open_storage()?;
    if is_strata(&storage) {
        anyhow::bail!(
            "unavailable_in_4_0: portable-import is not available on Strata in Vestige 4.0: a Strata log does not read portable archives yet. {STRATA_EXACT_RESTORE_HINT} `vestige restore <file.json>` re-ingests a `vestige export` file as new memories."
        );
    }
    let report = storage.import_portable_archive_from_path(&input, mode)?;

    println!(
        "{}: {}",
        "Tables imported".white().bold(),
        report.tables_imported
    );
    println!(
        "{}: {}",
        "Rows imported".white().bold(),
        report.rows_imported
    );
    println!(
        "{}: {}",
        "Tables skipped".white().bold(),
        report.tables_skipped
    );
    println!("{}: {}", "FTS rebuilt".white().bold(), report.fts_rebuilt);
    if merge {
        println!(
            "{}: {} inserted, {} updated, {} deleted, {} skipped, {} kept local",
            "Merge".white().bold(),
            report.rows_inserted,
            report.rows_updated,
            report.rows_deleted,
            report.rows_skipped,
            report.conflicts_kept_local
        );
    }
    println!();
    println!("{}", "Portable import complete.".green().bold());

    Ok(())
}

/// Run file-backed two-way sync.
fn run_sync(archive: Option<PathBuf>, cloud: bool, endpoint: Option<String>) -> anyhow::Result<()> {
    if cloud {
        run_sync_cloud(endpoint)
    } else {
        let archive = archive.ok_or_else(|| {
            anyhow::anyhow!(
                "no sync target: pass an archive path for file sync, or --cloud for Vestige Cloud"
            )
        })?;
        run_sync_file(archive)
    }
}

fn run_sync_file(archive: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige File Sync ===".cyan().bold());
    println!();
    println!("{}: {}", "Archive".white().bold(), archive.display());

    let storage = open_storage()?;
    refuse_sync_on_strata(&storage)?;
    let report = storage.sync_portable_archive_file(&archive)?;
    print_sync_report(&report);
    Ok(())
}

/// Sync (file or cloud) merges portable archives, which a Strata log does not
/// write or read in 4.0.
fn refuse_sync_on_strata(storage: &Arc<Storage>) -> anyhow::Result<()> {
    if is_strata(storage) {
        anyhow::bail!(
            "unavailable_in_4_0: sync is not available on Strata in Vestige 4.0: sync merges portable archives, which a Strata log does not write or read yet. Nothing was read or written. {STRATA_PORTABLE_ALTERNATIVES}"
        );
    }
    Ok(())
}

/// What works on a Strata log in place of a portable archive.
const STRATA_PORTABLE_ALTERNATIVES: &str = "Use `vestige export <file> --format json` (or jsonl) for the memories, or `vestige backup <dir>` for an exact copy of the log.";

#[cfg(feature = "cloud-sync")]
fn run_sync_cloud(endpoint: Option<String>) -> anyhow::Result<()> {
    let endpoint = endpoint
        .or_else(|| std::env::var("VESTIGE_CLOUD_ENDPOINT").ok())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Vestige Pro syncs your agent's memory across machines, end-to-end encrypted \
                 ($19/month).\n  Subscribe: https://github.com/samvallad33/vestige#vestige-pro\n  \
                 Already subscribed? Pass --endpoint or set VESTIGE_CLOUD_ENDPOINT from your \
                 welcome email."
            )
        })?;
    let sync_key = std::env::var("VESTIGE_CLOUD_SYNC_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no sync key set. Your local memory is free forever; syncing it across machines \
                 is Vestige Pro ($19/month, zero-knowledge encrypted).\n  Subscribe: \
                 https://github.com/samvallad33/vestige#vestige-pro\n  Already subscribed? Set \
                 VESTIGE_CLOUD_SYNC_KEY from your welcome email."
            )
        })?;

    // Required zero-knowledge encryption passphrase. Never sent to the server.
    let encryption_key = std::env::var("VESTIGE_CLOUD_ENCRYPTION_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no encryption key: set VESTIGE_CLOUD_ENCRYPTION_KEY to a strong passphrase. \
                 Vestige Pro refuses plaintext cloud sync. Use the same passphrase on every \
                 device; it cannot be recovered by Vestige."
            )
        })?;

    println!("{}", "=== Vestige Cloud Sync ===".cyan().bold());
    println!();
    println!("{}: {}", "Endpoint".white().bold(), endpoint);
    println!(
        "{}: {}",
        "Encryption".white().bold(),
        "zero-knowledge (XChaCha20-Poly1305) — your data is encrypted before upload".green()
    );

    let storage = open_storage()?;
    refuse_sync_on_strata(&storage)?;
    let report = storage.sync_portable_archive_cloud(&endpoint, &sync_key, Some(encryption_key))?;
    print_sync_report(&report);
    Ok(())
}

#[cfg(not(feature = "cloud-sync"))]
fn run_sync_cloud(_endpoint: Option<String>) -> anyhow::Result<()> {
    anyhow::bail!(
        "this build was compiled without the `cloud-sync` feature. 4.0 builds leave \
         it off. Building from source? Add --features cloud-sync."
    )
}

fn print_sync_report(report: &vestige_core::PortableSyncReport) {
    if let Some(pull) = &report.pull {
        println!("{}", "Pull: merged remote archive".yellow());
        println!(
            "  {} inserted, {} updated, {} deleted, {} skipped, {} kept local",
            pull.rows_inserted,
            pull.rows_updated,
            pull.rows_deleted,
            pull.rows_skipped,
            pull.conflicts_kept_local
        );
    } else {
        println!(
            "{}",
            "Pull: archive does not exist yet; creating it".yellow()
        );
    }

    println!("{}", "Push: wrote merged local state".yellow());
    println!(
        "{}",
        format!(
            "Sync complete: {} tables, {} rows",
            report.pushed_tables, report.pushed_rows
        )
        .green()
        .bold()
    );
}

/// Run garbage collection command
fn run_gc(
    min_retention: f64,
    max_age_days: Option<u64>,
    dry_run: bool,
    yes: bool,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Garbage Collection ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    // Deletion is erasure-class and withheld on an append-only Strata log.
    // Refuse before the confirmation prompt, which would promise a delete
    // that cannot happen. A dry run is read-only and still lists candidates.
    let strata = is_strata(&storage);
    if strata && !dry_run {
        anyhow::bail!(vestige_mcp::strata_memory::withheld_message("gc deletion"));
    }
    let all_nodes = fetch_all_nodes(&storage)?;
    let now = Utc::now();

    // Find candidates for deletion
    let candidates: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            // Must be below retention threshold
            if node.retention_strength >= min_retention {
                return false;
            }
            // If max_age_days specified, must also be older than that
            if let Some(max_days) = max_age_days {
                let age_days = (now - node.created_at).num_days();
                if age_days < 0 || (age_days as u64) < max_days {
                    return false;
                }
            }
            true
        })
        .collect();

    println!(
        "{}: {}",
        "Min retention threshold".white().bold(),
        min_retention
    );
    if let Some(max_days) = max_age_days {
        println!("{}: {} days", "Max age".white().bold(), max_days);
    }
    println!(
        "{}: {} / {} total",
        if strata {
            "Below threshold"
        } else {
            "Candidates for deletion"
        }
        .white()
        .bold(),
        candidates.len(),
        all_nodes.len()
    );

    if candidates.is_empty() {
        println!();
        println!(
            "{}",
            "No memories match the garbage collection criteria.".green()
        );
        return Ok(());
    }

    // Show sample of what would be deleted
    println!();
    println!(
        "{}",
        if strata {
            "Sample of memories below the threshold:"
        } else {
            "Sample of memories to be removed:"
        }
        .yellow()
        .bold()
    );
    let sample_count = candidates.len().min(10);
    for node in candidates.iter().take(sample_count) {
        let age_days = (now - node.created_at).num_days();
        println!(
            "  {} [ret={:.3}, age={}d] {}",
            node.id.get(..8).unwrap_or(&node.id).dimmed(),
            node.retention_strength,
            age_days,
            truncate(&node.content, 60).dimmed()
        );
    }
    if candidates.len() > sample_count {
        println!(
            "  {} ... and {} more",
            "".dimmed(),
            candidates.len() - sample_count
        );
    }

    if dry_run {
        println!();
        let line = if strata {
            format!(
                "Dry run: {} memories are below the threshold. Deleting them is withheld on Strata in Vestige 4.0 (the log is append-only), so gc cannot remove them.",
                candidates.len()
            )
        } else {
            format!(
                "Dry run: {} memories would be deleted. Re-run without --dry-run to delete.",
                candidates.len()
            )
        };
        println!("{}", line.yellow().bold());
        return Ok(());
    }

    // Confirmation prompt (unless --yes)
    if !yes {
        println!();
        print!(
            "{} Delete {} memories? This cannot be undone. [y/N] ",
            "WARNING:".red().bold(),
            candidates.len()
        );
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        if input != "y" && input != "yes" {
            println!("{}", "Aborted.".yellow());
            return Ok(());
        }
    }

    // Perform deletion
    let mut deleted = 0;
    let mut errors = 0;
    let total_candidates = candidates.len();

    for node in &candidates {
        match storage.delete_node(&node.id) {
            Ok(true) => deleted += 1,
            Ok(false) => errors += 1, // node was already gone
            Err(e) => {
                eprintln!(
                    "  {} Failed to delete {}: {}",
                    "ERR".red(),
                    node.id.get(..8).unwrap_or(&node.id),
                    e
                );
                errors += 1;
            }
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Garbage collection complete: {}/{} memories deleted{}",
            deleted,
            total_candidates,
            if errors > 0 {
                format!(" ({} errors)", errors)
            } else {
                String::new()
            }
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Ingest a memory via CLI (routes through smart_ingest / PE Gating)
fn run_ingest(
    content: String,
    tags: Option<String>,
    node_type: String,
    source: Option<String>,
    ago_days: Option<i64>,
    created_at: Option<String>,
    allow_secrets: bool,
) -> anyhow::Result<()> {
    if content.trim().is_empty() {
        anyhow::bail!("Content cannot be empty");
    }
    // Parse/validate BEFORE any storage write: a bad timestamp must not leave a
    // node behind with the wrong created_at and no key to retry with.
    let created_at_ts = created_at
        .map(|ts| {
            chrono::DateTime::parse_from_rfc3339(&ts)
                .map(|t| t.with_timezone(&chrono::Utc))
                .map_err(|e| anyhow::anyhow!("--created-at wants RFC 3339: {e}"))
        })
        .transpose()?;
    if ago_days.is_some() && created_at_ts.is_some() {
        anyhow::bail!("--ago-days and --created-at are mutually exclusive");
    }

    let tag_list: Vec<String> = tags
        .as_deref()
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let input = IngestInput {
        content: content.clone(),
        node_type,
        source,
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: tag_list,
        valid_from: None,
        valid_until: None,
        validity_inferred: false,
        source_envelope: None,
    };

    let storage = open_storage()?;
    // The Strata adapter does not expose a creation-time rewrite in 4.0.
    // Refuse before ingesting, or the memory would land with the current
    // time and the command would still fail.
    if is_strata(&storage) && (ago_days.is_some() || created_at_ts.is_some()) {
        anyhow::bail!(
            "unavailable_in_4_0: ingest --ago-days/--created-at is not available on Strata in Vestige 4.0: the Strata store does not expose a creation-time rewrite yet, so the memory would keep the time it was written. Nothing was written; drop the flag to ingest with the current time."
        );
    }
    let secret_policy = if allow_secrets {
        SecretPolicy::AllowExplicitly
    } else {
        SecretPolicy::Reject
    };

    {
        let node = storage.ingest_with_secret_policy(input, secret_policy)?;
        if let Some(days) = ago_days {
            // Duration::days panics on overflow for extreme inputs; try_days
            // returns None instead. The subtraction itself can ALSO overflow the
            // DateTime range, so use checked_sub_signed rather than `-` (which
            // panics). Both the construction and the subtraction are guarded.
            let delta = chrono::Duration::try_days(days).ok_or_else(|| {
                anyhow::anyhow!("--ago-days value {days} is out of the supported range")
            })?;
            let when = chrono::Utc::now()
                .checked_sub_signed(delta)
                .ok_or_else(|| {
                    anyhow::anyhow!("--ago-days value {days} is out of the supported range")
                })?;
            storage.set_created_at(&node.id, when)?;
        }
        if let Some(when) = created_at_ts {
            storage.set_created_at(&node.id, when)?;
        }
        println!("{}", "=== Vestige Ingest ===".cyan().bold());
        println!();
        println!("{}: create", "Decision".white().bold());
        println!("{}: {}", "Node ID".white().bold(), node.id);
        println!();
        let confirmation = if allow_secrets {
            "Memory created with explicit credential override (content redacted)".to_string()
        } else {
            format!("Memory created ({})", truncate(&content, 60))
        };
        println!("{}", confirmation.green().bold());

        // Auto-connect: the ingest-time share of `vestige connect`, run on
        // just this memory (the write goes to the default scope, like the
        // ingest above). The memory is already saved, so a failure here is
        // reported, never fatal — the same rule the post-ingest hooks keep.
        match vestige_mcp::auto_connect::auto_connect_new_memory(
            storage.as_ref(),
            &node.id,
            vestige_core::DEFAULT_MEMORY_SCOPE,
            &node.content,
            &node.tags,
        ) {
            Ok(report) => {
                if report.edges > 0 {
                    println!(
                        "{}",
                        format!(
                            "Auto-connected: {} edge(s) created on exact identities: {}",
                            report.edges,
                            report.shared_identities.join(", ")
                        )
                        .green()
                        .bold()
                    );
                    for pair in &report.pairs {
                        println!(
                            "  {} -[touched]-> {}  joined on: {}",
                            pair.source_id,
                            pair.target_id,
                            pair.identities.join(", ")
                        );
                    }
                }
                if !report.skipped_common_tags.is_empty() {
                    println!(
                        "Auto-connect skipped tag(s) carried by more than {} memories: {}",
                        vestige_mcp::auto_connect::MAX_TAG_CARRIERS,
                        report.skipped_common_tags.join(", ")
                    );
                }
            }
            Err(err) => eprintln!("{} auto-connect skipped: {err}", "WARN".yellow()),
        }
    }

    Ok(())
}

/// Findings across `texts`, each (kind, fingerprint) once.
fn credential_findings<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    include_suspected: bool,
) -> Vec<vestige_core::SecretFinding> {
    let mut findings: Vec<vestige_core::SecretFinding> = Vec::new();
    for text in texts {
        for finding in scan_secrets(text) {
            if !findings.contains(&finding) {
                findings.push(finding);
            }
        }
    }
    findings
        .retain(|finding| include_suspected || finding.confidence == SecretConfidence::Blocking);
    findings
}

/// Read-only audit of already-persisted text for credential shapes.
///
/// On a Strata log it covers every record the log holds: live, suppressed and
/// retired memories (their bytes stay in the append-only log), scopes,
/// provenance, tags, and intentions. Deliberately emits IDs, detector
/// classes, and short fingerprints only. It never prints the matching
/// content, source, or surrounding context.
fn run_scan_secrets(
    include_suspected: bool,
    json_output: bool,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let storage = open_storage()?;
    let mut scanned = 0_usize;
    let mut hits = Vec::new();

    let hit_json = |id: &str,
                    record_kind: &str,
                    retired: bool,
                    created_at: chrono::DateTime<chrono::Utc>,
                    findings: Vec<vestige_core::SecretFinding>| {
        serde_json::json!({
            "nodeId": id,
            "recordKind": record_kind,
            "retired": retired,
            "createdAt": created_at.to_rfc3339(),
            "findings": findings.into_iter().map(|finding| serde_json::json!({
                "kind": finding.kind.as_str(),
                "confidence": finding.confidence.to_string(),
                "fingerprint": finding.fingerprint,
            })).collect::<Vec<_>>(),
        })
    };

    if let Some(records) = vestige_mcp::strata_memory::secret_audit_records(storage.as_ref()) {
        for record in records {
            scanned += 1;
            let findings =
                credential_findings(record.texts.iter().map(String::as_str), include_suspected);
            if findings.is_empty() {
                continue;
            }
            let record_kind = match record.kind {
                vestige_mcp::strata_memory::AuditKind::Memory => "memory",
                vestige_mcp::strata_memory::AuditKind::Intention => "intention",
            };
            hits.push(hit_json(
                &record.id,
                record_kind,
                record.retired,
                record.created_at,
                findings,
            ));
            if limit.is_some_and(|max| hits.len() >= max) {
                break;
            }
        }
    } else {
        let mut offset = 0_i32;
        loop {
            let nodes = storage.get_all_nodes(100, offset)?;
            if nodes.is_empty() {
                break;
            }
            offset += nodes.len() as i32;

            for node in nodes {
                scanned += 1;
                let mut texts: Vec<&str> = vec![node.content.as_str()];
                texts.extend(node.source.as_deref());
                texts.extend(node.tags.iter().map(String::as_str));
                if let Some(envelope) = node.source_envelope.as_ref() {
                    texts.extend(
                        [
                            envelope.source_url.as_deref(),
                            envelope.source_project.as_deref(),
                            envelope.source_type.as_deref(),
                            envelope.source_author.as_deref(),
                        ]
                        .into_iter()
                        .flatten(),
                    );
                }
                let findings = credential_findings(texts, include_suspected);
                if findings.is_empty() {
                    continue;
                }
                hits.push(hit_json(
                    &node.id,
                    "memory",
                    false,
                    node.created_at,
                    findings,
                ));
                if limit.is_some_and(|max| hits.len() >= max) {
                    break;
                }
            }

            if limit.is_some_and(|max| hits.len() >= max) {
                break;
            }
        }
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "scanned": scanned,
                "hits": hits,
                "truncated": limit.is_some_and(|max| hits.len() >= max),
            }))?
        );
    } else if hits.is_empty() {
        println!("No potential credentials found across {scanned} records.");
    } else {
        println!(
            "Potential credentials found in {} of {scanned} scanned records:",
            hits.len()
        );
        for hit in &hits {
            let node_id = hit["nodeId"].as_str().unwrap_or("unknown");
            let note = match (hit["recordKind"].as_str(), hit["retired"].as_bool()) {
                (Some("intention"), _) => " | intention",
                (_, Some(true)) => " | suppressed or retired",
                _ => "",
            };
            for finding in hit["findings"].as_array().into_iter().flatten() {
                println!(
                    "{node_id} | {} | {} | {}{note}",
                    finding["kind"].as_str().unwrap_or("unknown"),
                    finding["confidence"].as_str().unwrap_or("unknown"),
                    finding["fingerprint"].as_str().unwrap_or("unknown"),
                );
            }
        }
        if is_strata(&storage) {
            // Erasure is withheld on the append-only log: suppress hides a
            // memory from reads, but its bytes stay in log/ and in backups.
            println!(
                "Rotate live credentials. On a Strata log the memory's bytes stay in the append-only log (and in every backup of it): suppress hides it from reads, and erasure is withheld in 4.0, so treat these credentials as exposed."
            );
        } else {
            println!("Rotate live credentials, then remove or replace affected memories manually.");
        }
    }

    Ok(())
}

/// Run Retroactive Salience Backfill from the CLI (the demo's payoff command).
#[allow(clippy::too_many_arguments)]
fn run_backfill(
    failure_id: Option<String>,
    manual: bool,
    lookback_days: i64,
    promote: bool,
    contrast: bool,
    json: bool,
    git_repo: Option<PathBuf>,
    worked_in: Option<String>,
    broke_in: Option<String>,
    why_not: Option<String>,
) -> anyhow::Result<()> {
    let storage = open_storage()?;
    // Backfill joins memories by shared entity names. A shared name is not a
    // recorded edge, so a Strata log refuses and names the successor, which
    // walks recorded causal edges only.
    if is_strata(&storage) {
        let start = failure_id.as_deref().unwrap_or("<memory-id>");
        anyhow::bail!(
            "unavailable_in_4_0: backfill is not available on Strata in Vestige 4.0: it joins memories by shared entity names, which are not recorded edges. Use its successor, `vestige causal-walk --logged-write {start}`, which walks recorded causal edges (closed_by, derived_from, evidence_of, touched) backward from that memory."
        );
    }

    // Resolve the failure text up front (used by the contrast baseline).
    // Use the SAME failure detector the backfill tool uses (content + tags, full
    // marker list) so the CLI's pick and the tool's pick never diverge.
    let failure_text: Option<String> = match &failure_id {
        Some(id) => storage.get_node(id).ok().flatten().map(|n| n.content),
        None => storage
            .get_all_nodes(500, 0)
            .ok()
            .and_then(|nodes| {
                nodes
                    .into_iter()
                    .find(vestige_mcp::tools::backfill::looks_like_failure)
            })
            .map(|n| n.content),
    };

    // CONTRAST: show what a SIMILARITY SEARCH returns for the failure first — the
    // lookalike it ranks at the top, which is NOT the cause. Same store, same
    // query. Keyword search ranks by RESEMBLANCE, which is exactly the blind spot.
    if contrast && let Some(ftext) = &failure_text {
        // Generic salient-words query: keep alphanumerics, drop a leading
        // "<word>:" label if present (e.g. "Service crashed:"). No hardcoding.
        let query = match ftext.split_once(": ") {
            Some((lead, rest)) if lead.split_whitespace().count() <= 2 => rest,
            _ => ftext.as_str(),
        };

        // w1b: keyword-only. The engine label stays explicit so the contrast
        // never presents lexical resemblance as semantic ranking (the audit's
        // top finding: never present keyword search as "semantic").
        let engine = "keyword (BM25)";
        let mut shown = false;
        println!(
            "{}",
            format!("── 1. SIMILARITY SEARCH · {engine} ──")
                .dimmed()
                .bold()
        );
        println!("   query: {}", truncate(query, 60).dimmed());

        // best OTHER match (exclude the failure itself, which trivially matches).
        // keyword/BM25 — ranks by lexical resemblance.
        if !shown {
            // keyword/BM25 (always works) — still ranks by lexical resemblance.
            if let Ok(hits) = storage.search(query, 6) {
                let others: Vec<_> = hits
                    .iter()
                    .filter(|h| h.content != *ftext)
                    .take(3)
                    .collect();
                for (i, h) in others.iter().enumerate() {
                    let tag = if i == 0 {
                        " ← top match".red().bold().to_string()
                    } else {
                        String::new()
                    };
                    println!("   {}. {}{}", i + 1, truncate(&h.content, 60).normal(), tag);
                    shown = true;
                }
            }
        }
        if shown {
            println!(
                "   {}",
                "→ ranked by RESEMBLANCE. its top hit is a lookalike, not the cause.".red()
            );
        } else {
            println!(
                "   {}",
                "(no lookalikes — nothing resembles the crash)".dimmed()
            );
        }
        println!();
        println!(
            "{}",
            "── 2. POSTDICT (reach backward for the CAUSE) ──"
                .magenta()
                .bold()
        );
    }

    let args = serde_json::json!({
        "failure_id": failure_id,
        "manual": manual,
        "lookback_days": lookback_days,
        "promote": promote,
        "git_repo": git_repo.as_ref().map(|p| p.display().to_string()),
        "worked_in": worked_in,
        "broke_in": broke_in,
        "why_not": why_not,
    });

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt
        .block_on(vestige_mcp::tools::backfill::execute(&storage, Some(args)))
        .map_err(|e| anyhow::anyhow!(e))?;

    // Machine-readable path: dump the raw tool result (includes per-cause
    // memory_id, shared_entities, similarity_rank) and stop. Used by tooling
    // and by any scoring harness that needs real engine output rather than the
    // human-readable rendering. (It once named CauseBench, which is retracted.)
    if json {
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }

    println!(
        "{}",
        "=== Retroactive Salience Backfill ===".magenta().bold()
    );
    println!();
    if result["triggered"] != serde_json::json!(true) {
        println!(
            "{} {}",
            "Not triggered:".yellow().bold(),
            result["reason"].as_str().unwrap_or("event not salient")
        );
        return Ok(());
    }
    if let Some(f) = result["failure"].as_object() {
        println!(
            "{} {}",
            "Failure:".red().bold(),
            f.get("content_preview")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        );
    }
    println!();
    println!(
        "{}",
        "Reached BACKWARD and surfaced associated candidates through shared entities:".white()
    );
    println!();
    if let Some(causes) = result["causes"].as_array() {
        for (i, c) in causes.iter().enumerate() {
            let age = c["age_days_before_failure"].as_f64().unwrap_or(0.0);
            let rank = c["similarity_rank"].as_u64();
            println!(
                "  {} {}",
                format!("#{}", i + 1).cyan().bold(),
                c["content_preview"].as_str().unwrap_or("").green().bold()
            );
            println!(
                "     {} {:.1} days before the failure",
                "↩ reached back".magenta(),
                age
            );
            if let Some(shared) = c["shared_entities"].as_array() {
                let ents: Vec<&str> = shared.iter().filter_map(|e| e.as_str()).collect();
                println!("     {} {}", "🔗 causal join:".magenta(), ents.join(", "));
            }
            if let Some(r) = rank {
                println!(
                    "     {} ranked #{} on similarity {}",
                    "🔍".magenta(),
                    r,
                    "(quiet on similarity at backfill time; association, not proven cause)"
                        .dimmed()
                );
            }
            if c["promoted"] == serde_json::json!(true) {
                println!(
                    "     {} promoted — it will resurface next time",
                    "✅".green()
                );
            }
            println!();
        }
    }
    // why-not-X: name a suspect, get the rule that excluded it
    if let Some(w) = result["why_not"].as_object() {
        println!(
            "{} {} — {}",
            "Why not".yellow().bold(),
            w.get("target").and_then(|v| v.as_str()).unwrap_or(""),
            w.get("detail").and_then(|v| v.as_str()).unwrap_or("")
        );
        println!();
    }
    // strongest rejections, so a miss is explainable instead of silent
    if let Some(rejected) = result["rejected"].as_array().filter(|r| !r.is_empty()) {
        println!(
            "{}",
            "Rejected (top candidates that failed a rule):".white()
        );
        for r in rejected {
            println!(
                "  {} {} — {}",
                "✗".red(),
                r["content_preview"].as_str().unwrap_or("").dimmed(),
                r["reason"].as_str().unwrap_or("")
            );
        }
        println!();
    }
    // trail-break report: what to record so the chain closes next time
    if let Some(gap) = result["gap"].as_object() {
        println!("{}", "TRAIL GAP:".yellow().bold());
        println!("  {}", gap["note"].as_str().unwrap_or(""));
        if let Some(s) = gap["suggestion"].as_str() {
            println!("  {} {}", "→".magenta(), s);
        }
        println!();
    }
    Ok(())
}

/// Run a causal walk from the CLI: explicit start points -> exact mechanism
/// edges -> ranked suspect change records. Mirrors the MCP `causal_walk`
/// tool (same core engine); `--json` prints the raw result for tooling.
#[allow(clippy::too_many_arguments)]
fn run_causal_walk(
    failing_test: Option<String>,
    stack_frame: Option<String>,
    ci_run: Option<String>,
    logged_write: Option<String>,
    node_id: Option<String>,
    git_repo: Option<PathBuf>,
    worked_in: Option<String>,
    broke_in: Option<String>,
    lookback_days: i64,
    promote: bool,
    scope: String,
    json: bool,
) -> anyhow::Result<()> {
    use vestige_core::advanced::causal_walk as cw;

    let storage = open_storage()?;
    let node_id = node_id
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty());
    if is_strata(&storage) {
        // These start points resolve through shared names (a test's file, a
        // frame's path, a run's anchors, a tag range's commits), which are not
        // recorded edges. On their own they are refused; with --node-id naming
        // the recorded memory they describe, the walk starts at that memory.
        let name_based: Vec<&str> = [
            ("--failing-test", failing_test.is_some()),
            ("--stack-frame", stack_frame.is_some()),
            ("--ci-run", ci_run.is_some()),
            ("--git-repo", git_repo.is_some()),
            ("--worked-in", worked_in.is_some()),
            ("--broke-in", broke_in.is_some()),
        ]
        .into_iter()
        .filter_map(|(flag, given)| given.then_some(flag))
        .collect();
        if !name_based.is_empty() && node_id.is_none() {
            anyhow::bail!(
                "unavailable_in_4_0: causal-walk {} resolves through shared names, which are not recorded edges, so a Strata log cannot walk it by itself. Add --node-id <memory-id> (the memory that records this symptom) to walk from that memory, or pass --logged-write <memory-id> to walk a memory directly.",
                name_based.join(", ")
            );
        }
        let version_flags = [git_repo.is_some(), worked_in.is_some(), broke_in.is_some()];
        if version_flags.iter().any(|given| *given) && !version_flags.iter().all(|given| *given) {
            anyhow::bail!(
                "causal-walk: a version range needs --git-repo, --worked-in and --broke-in together"
            );
        }
        let mut start_points: Vec<serde_json::Value> = Vec::new();
        if let Some(name) = &failing_test {
            start_points.push(
                serde_json::json!({"kind": "failing_test", "name": name, "node_id": node_id}),
            );
        }
        if let Some(frame) = &stack_frame {
            start_points.push(
                serde_json::json!({"kind": "stack_frame", "frame": frame, "node_id": node_id}),
            );
        }
        if let Some(run_id) = &ci_run {
            start_points
                .push(serde_json::json!({"kind": "ci_run", "run_id": run_id, "node_id": node_id}));
        }
        if let (Some(worked), Some(broke), Some(repo)) = (&worked_in, &broke_in, &git_repo) {
            start_points.push(serde_json::json!({
                "kind": "version_range",
                "worked_in": worked,
                "broke_in": broke,
                "repo": repo.display().to_string(),
                "node_id": node_id,
            }));
        }
        // --node-id with no descriptive start point, or an explicit
        // --logged-write, is a plain walk from that memory.
        if start_points.is_empty()
            && let Some(id) = &node_id
        {
            start_points.push(serde_json::json!({"kind": "logged_write", "node_id": id}));
        }
        if let Some(id) = &logged_write {
            start_points.push(serde_json::json!({"kind": "logged_write", "node_id": id}));
        }
        return run_causal_walk_strata(&storage, start_points, scope, json);
    }

    // Assemble start points; the walk refuses (needs_report) rather than
    // guessing when none resolve. A --node-id rides on every start point given
    // and stands alone as a logged_write when there is none.
    let mut start_points: Vec<cw::StartPoint> = Vec::new();
    if let Some(name) = failing_test {
        start_points.push(cw::StartPoint::FailingTest {
            name,
            node_id: node_id.clone(),
        });
    }
    if let Some(frame) = stack_frame {
        start_points.push(cw::StartPoint::StackFrame {
            frame,
            node_id: node_id.clone(),
        });
    }
    if let Some(run_id) = ci_run {
        start_points.push(cw::StartPoint::CiRun {
            run_id,
            node_id: node_id.clone(),
        });
    }
    if let Some(node_id) = logged_write {
        start_points.push(cw::StartPoint::LoggedWrite { node_id });
    }
    if let (Some(worked), Some(broke), Some(repo)) = (worked_in, broke_in, git_repo) {
        start_points.push(cw::StartPoint::VersionRange {
            worked_in: worked,
            broke_in: broke,
            repo: repo.display().to_string(),
            node_id: node_id.clone(),
        });
    }
    if start_points.is_empty()
        && let Some(id) = node_id
    {
        start_points.push(cw::StartPoint::LoggedWrite { node_id: id });
    }

    #[cfg(vestige_embeddings_removed)]
    {
        let _ = storage.init_embeddings();
    }

    let request = cw::CausalWalkRequest {
        scope,
        start_points,
        lookback_days,
        scan_limit: 500,
    };
    let result = cw::walk_storage(&*storage, &request).map_err(anyhow::Error::msg)?;

    if json {
        let mut payload = serde_json::to_value(&result)?;
        if promote && !result.causes.is_empty() {
            payload["promoted_edges"] = serde_json::to_value(
                cw::persist_evidence_edges(&*storage, &result).unwrap_or_default(),
            )?;
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("{}", "=== Causal Walk ===".magenta().bold());
    println!(
        "  {} hypotheses, not proven causes — investigate before attributing cause",
        "note:".dimmed()
    );
    println!();

    if let Some(report) = &result.needs_report {
        println!("{}", "NEEDS REPORT (the walk refused):".yellow().bold());
        for m in &report.missing {
            println!("  {} {m}", "!".red());
        }
        println!("{}", "Provide one of:".white());
        for r in &report.required_start_points {
            println!("  {} {r}", "->".cyan());
        }
        return Ok(());
    }

    for (rank, cause) in result.causes.iter().enumerate() {
        println!(
            "{} {} score {:.2}",
            format!("#{}", rank + 1).green().bold(),
            cause.sha.as_deref().unwrap_or(&cause.id),
            cause.score
        );
        for hop in &cause.path {
            println!("  {} via {} — {}", "->".cyan(), hop.via, hop.hop);
        }
        println!(
            "  {} {}",
            "anchors:".dimmed(),
            cause.shared_anchors.join(", ")
        );
        println!();
    }

    if !result.rejected.is_empty() {
        println!("{}", "Rejected (why-not, top candidates):".white());
        for r in &result.rejected {
            println!("  {} {} — {}", "✗".red(), r.id, r.reason);
        }
        println!();
    }

    if promote && !result.causes.is_empty() {
        let written = cw::persist_evidence_edges(&*storage, &result).map_err(anyhow::Error::msg)?;
        println!(
            "{} {} evidence_of trail edge{} persisted",
            "→".magenta(),
            written.len(),
            if written.len() == 1 { "" } else { "s" }
        );
    } else {
        println!(
            "  {}",
            "(preview: nothing persisted; pass without --no-promote to record evidence_of edges)"
                .dimmed()
        );
    }
    Ok(())
}

/// Causal walk on a Strata log: the MCP `causal_walk` tool's recorded-edge
/// path, a bounded backward BFS from one memory over recorded causal edges.
/// Read-only: nothing is persisted, whatever --no-promote says.
fn run_causal_walk_strata(
    storage: &Arc<Storage>,
    start_points: Vec<serde_json::Value>,
    scope: String,
    json: bool,
) -> anyhow::Result<()> {
    let args = serde_json::json!({ "scope": scope, "start_points": start_points });
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt
        .block_on(vestige_mcp::tools::causal_walk::execute(
            storage,
            Some(args),
        ))
        .map_err(anyhow::Error::msg)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    println!("{}", "=== Causal Walk ===".magenta().bold());
    println!(
        "  {} backward over recorded causal edges only (closed_by, derived_from, evidence_of, touched); hypotheses, not proven causes",
        "note:".dimmed()
    );
    println!();

    if let Some(report) = result["needs_report"].as_object() {
        println!("{}", "NEEDS REPORT (the walk refused):".yellow().bold());
        for missing in report
            .get("missing")
            .and_then(|m| m.as_array())
            .into_iter()
            .flatten()
        {
            println!("  {} {}", "!".red(), missing.as_str().unwrap_or("?"));
        }
        if let Some(detail) = report.get("detail").and_then(|d| d.as_str()) {
            println!("  {detail}");
        }
        println!(
            "{} {}",
            "Provide:".white(),
            "--node-id <memory-id> (or --logged-write <memory-id>)".cyan()
        );
        return Ok(());
    }

    let preview = |value: &serde_json::Value| truncate(value.as_str().unwrap_or(""), 100);
    if let Some(start) = result["nodes"].as_array().and_then(|nodes| nodes.first()) {
        println!(
            "{} {}  {}",
            "Start:".white().bold(),
            start["id"].as_str().unwrap_or("?"),
            preview(&start["content"]).dimmed()
        );
        println!();
    }
    let causes = result["causes"].as_array().cloned().unwrap_or_default();
    if causes.is_empty() {
        println!(
            "{}",
            "No recorded causal edge leads upstream from this memory.".dimmed()
        );
    }
    for (rank, cause) in causes.iter().enumerate() {
        println!(
            "{} {} depth {}",
            format!("#{}", rank + 1).green().bold(),
            cause["id"].as_str().unwrap_or("?"),
            cause["depth"]
        );
        println!("  {}", preview(&cause["content"]));
        for hop in cause["path"].as_array().into_iter().flatten() {
            println!(
                "  {} {} -[{}]-> {}",
                "->".cyan(),
                hop["source_id"].as_str().unwrap_or("?"),
                hop["link_type"].as_str().unwrap_or("?"),
                hop["target_id"].as_str().unwrap_or("?")
            );
        }
        println!();
    }
    if result["truncated"] == true {
        println!(
            "  {} stopped at the walk bounds (depth {}, {} nodes); more recorded causes lie beyond",
            "truncated:".yellow(),
            result["bounds"]["max_depth"],
            result["bounds"]["max_nodes"]
        );
    }
    println!(
        "  {}",
        "(read-only: a Strata walk persists nothing)".dimmed()
    );
    Ok(())
}

/// Normalized remote identity for a repo: `git config remote.origin.url`
/// stripped of protocol and `.git` (`https://github.com/a/b.git`,
/// `git@github.com:a/b` -> `github.com/a/b`). None when no remote is set.
fn git_repo_identity(path: &Path) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if url.is_empty() {
        return None;
    }
    let stripped = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("git@")
        .trim_start_matches("ssh://")
        .trim_end_matches(".git")
        .replace(':', "/");
    Some(stripped)
}

/// Ingest git commits as memory records. Each commit becomes one event whose
/// content carries the files, module dirs and hunk-header symbols — the
/// query-time entity extractor turns those into the causal join keys. Records
/// upsert on `(git, "<repo>#<sha>")`, so re-running is free, and created_at is
/// the commit time so the backward reach is exact.
fn run_ingest_git(
    path: PathBuf,
    since: Option<String>,
    until: Option<String>,
    max_commits: usize,
    json: bool,
) -> anyhow::Result<()> {
    let storage = open_storage()?;
    if is_strata(&storage) {
        anyhow::bail!(
            "unavailable_in_4_0: ingest-git is not available on Strata in Vestige 4.0: it upserts each commit by source key and dates it to the commit time, and the Strata store exposes neither operation yet. Nothing was written."
        );
    }
    let repo_display = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    // Identity = remote URL (normalized), not the directory basename: two
    // clones named "vestige" in different paths would otherwise share one
    // source-key namespace and project tag.
    let repo_name = git_repo_identity(&path).unwrap_or_else(|| {
        path.canonicalize()
            .map(|p| p.display().to_string())
            .unwrap_or(repo_display.clone())
    });
    let mut git_args = vec![
        "log".to_string(),
        "-p".to_string(),
        "--unified=0".to_string(),
        "--no-color".to_string(),
        "-n".to_string(),
        max_commits.to_string(),
        "--pretty=format:%x1e%H%x1f%aI%x1f%s".to_string(),
    ];
    if let Some(s) = &since {
        git_args.push(format!("--since={s}"));
    }
    if let Some(u) = &until {
        git_args.push(format!("--until={u}"));
    }
    let out = Command::new("git")
        .arg("-C")
        .arg(&path)
        .args(&git_args)
        .output()
        .context("running git log — is `git` installed and is this a git repo?")?;
    if !out.status.success() {
        anyhow::bail!("git log failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let commits =
        vestige_core::advanced::git_records::parse_git_log(&String::from_utf8_lossy(&out.stdout));

    let mut created = 0usize;
    let mut updated = 0usize;
    let mut unchanged = 0usize;
    for c in &commits {
        // non_exhaustive: default-then-mutate is the only cross-crate construction
        let mut envelope = SourceEnvelope::default();
        envelope.source_system = Some(vestige_core::advanced::git_records::SOURCE_SYSTEM.into());
        envelope.source_id = Some(format!("{repo_name}#{}", c.sha));
        envelope.source_updated_at = Some(c.time);
        envelope.content_hash = Some(format!("git-{}", c.sha));
        envelope.synced_at = Some(Utc::now());
        envelope.source_project = Some(repo_name.clone());
        envelope.source_type = Some("commit".into());
        let input = IngestInput {
            content: vestige_core::advanced::git_records::record_content(c),
            node_type: "event".to_string(),
            source: Some("git".to_string()),
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec![
                vestige_core::advanced::git_records::COMMIT_TAG.to_string(),
                repo_name.clone(),
            ],
            valid_from: Some(c.time),
            valid_until: None,
            validity_inferred: false,
            source_envelope: Some(envelope),
        };
        let result = storage.upsert_by_source(input)?;
        match result.outcome {
            SourceUpsertOutcome::Created => {
                storage.set_created_at(&result.node_id, c.time)?;
                created += 1;
            }
            SourceUpsertOutcome::Updated => {
                storage.set_created_at(&result.node_id, c.time)?;
                updated += 1;
            }
            SourceUpsertOutcome::Unchanged => unchanged += 1,
        }
    }
    if json {
        println!(
            "{}",
            serde_json::json!({
                "repo": repo_display,
                "repo_identity": repo_name,
                "commits_seen": commits.len(),
                "created": created,
                "updated": updated,
                "unchanged": unchanged,
            })
        );
    } else {
        println!("{}", "=== Vestige Ingest Git ===".cyan().bold());
        println!();
        println!("{}: {}", "Repo".white().bold(), repo_display);
        println!("{}: {}", "Identity".white().bold(), repo_name);
        println!("{}: {}", "Commits seen".white().bold(), commits.len());
        println!("{}: {}", "Created".white().bold(), created);
        println!("{}: {}", "Updated".white().bold(), updated);
        println!("{}: {}", "Unchanged".white().bold(), unchanged);
    }
    Ok(())
}

/// One candidate `touched` edge: the earlier memory as the source, the
/// later one as the target, with the exact identities both record
/// (`kind:value`).
struct ConnectPair {
    source_id: String,
    target_id: String,
    source_content: String,
    target_content: String,
    shared: Vec<String>,
}

/// Create typed edges between memories that record the same exact identity.
///
/// Memories ingested one by one used to land as isolated nodes: tags, no
/// edges, so `causal-walk --logged-write` said no recorded causal edge leads
/// upstream even when two memories recorded the same `src/path.py`. The
/// ingest path now runs its share of this automatically
/// (`vestige_mcp::auto_connect`, on the memories a save just wrote); this
/// command remains the full-scan catch-up, joining pairs the ingest-time
/// handles cannot see. It extracts the exact identities of every memory in
/// the scope (`auto_connect::extract_identities`: tags, file paths, commit
/// shas, issue references, URLs; no words, no ML, no similarity) and writes
/// a `touched` edge for each pair sharing at least `--min-shared` of them,
/// through `Storage::save_connection`, which on a Strata log is
/// `StrataStore::save_connection` behind the gate. A tag carried by more
/// than `auto_connect::MAX_TAG_CARRIERS` memories of the scope is skipped
/// and named; every pair is printed with the identities that joined it.
///
/// Direction follows the walk's rule for `touched` (causal_walk.rs: the
/// source is the earlier record, so from the target the walk goes to the
/// source): the older memory of a pair is the source. Pairs already joined
/// by a recorded edge — either direction — are skipped, so re-running is
/// free.
fn run_connect(
    dry_run: bool,
    min_shared: usize,
    max_edges: usize,
    scope: String,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Connect ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    let mut nodes = fetch_nodes_in_scope(&storage, &scope)?;
    // Oldest first, so a pair's source is its earlier memory.
    nodes.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));

    println!("{}: {}", "Scope".white().bold(), scope);
    println!("{}: {}", "Memories scanned".white().bold(), nodes.len());

    if nodes.len() < 2 {
        println!();
        println!(
            "{}",
            "Nothing to connect: fewer than two memories in this scope.".green()
        );
        return Ok(());
    }

    use vestige_mcp::auto_connect::{self, Identity};
    let identity_sets: Vec<std::collections::BTreeSet<Identity>> = nodes
        .iter()
        .map(|node| {
            auto_connect::extract_identities(&node.content, &node.tags)
                .into_iter()
                .collect()
        })
        .collect();

    // The guard: a tag carried by more than MAX_TAG_CARRIERS memories of
    // this scope joins everything to everything and is not evidence.
    let common_tags = auto_connect::too_common_tags(nodes.iter().map(|node| node.tags.as_slice()));
    if !common_tags.is_empty() {
        println!(
            "{}: {}",
            format!(
                "Tags skipped (carried by more than {} memories)",
                auto_connect::MAX_TAG_CARRIERS
            )
            .white()
            .bold(),
            common_tags
                .iter()
                .map(|(tag, carriers)| format!("{tag} ({carriers})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Pairs already joined by any recorded edge (either direction) are left
    // alone: re-running connect must not stack parallel edges.
    let joined: HashSet<(String, String)> = storage
        .get_all_connections()?
        .into_iter()
        .filter_map(|edge| {
            (edge.source_id != edge.target_id).then(|| {
                if edge.source_id < edge.target_id {
                    (edge.source_id, edge.target_id)
                } else {
                    (edge.target_id, edge.source_id)
                }
            })
        })
        .collect();

    // An edge needs at least one shared identity; --min-shared 0 is read as 1.
    let min_shared = min_shared.max(1);
    let mut pairs: Vec<ConnectPair> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        for (later, set) in nodes.iter().zip(&identity_sets).skip(i + 1) {
            let joining = auto_connect::joining_identities(&identity_sets[i], set, &common_tags);
            if auto_connect::distinct_values(&joining) < min_shared {
                continue;
            }
            let shared: Vec<String> = joining.iter().map(Identity::to_string).collect();
            let key = if node.id < later.id {
                (node.id.clone(), later.id.clone())
            } else {
                (later.id.clone(), node.id.clone())
            };
            if joined.contains(&key) {
                continue;
            }
            pairs.push(ConnectPair {
                source_id: node.id.clone(),
                target_id: later.id.clone(),
                source_content: node.content.clone(),
                target_content: later.content.clone(),
                shared,
            });
        }
    }

    println!("{}: {}", "Candidate pairs".white().bold(), pairs.len());
    println!();

    let capped = pairs.len() > max_edges;
    pairs.truncate(max_edges);

    if pairs.is_empty() {
        println!(
            "{}",
            "No new edges: no memory pair shares an exact identity that is not already joined by a recorded edge."
                .green()
        );
        return Ok(());
    }

    for pair in &pairs {
        println!(
            "  {} -[touched]-> {}  {}",
            pair.source_id.dimmed(),
            pair.target_id.dimmed(),
            format!("joined on: {}", pair.shared.join(", ")).dimmed()
        );
        println!("      {}", truncate(&pair.source_content, 72).dimmed());
        println!("      {}", truncate(&pair.target_content, 72).dimmed());
    }

    if dry_run {
        println!();
        println!(
            "{}",
            format!(
                "Dry run: {} touched edge(s) would be created. Re-run without --dry-run to write them.",
                pairs.len()
            )
            .yellow()
            .bold()
        );
        return Ok(());
    }

    let now = Utc::now();
    let mut created = 0usize;
    let mut errors = 0usize;
    for pair in &pairs {
        // strength 0.5 -> strength_milli 500: a co-touch is a moderate link.
        let edge = ConnectionRecord {
            source_id: pair.source_id.clone(),
            target_id: pair.target_id.clone(),
            strength: 0.5,
            link_type: "touched".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        };
        match storage.save_connection(&edge) {
            Ok(()) => created += 1,
            Err(err) => {
                eprintln!(
                    "  {} Failed to connect {} -> {}: {}",
                    "ERR".red(),
                    pair.source_id,
                    pair.target_id,
                    err
                );
                errors += 1;
            }
        }
    }

    println!();
    if capped {
        println!(
            "{} stopped at the --max-edges cap ({max_edges}); more candidate pairs remain.",
            "truncated:".yellow()
        );
    }
    println!(
        "{}",
        format!(
            "Connect complete: {}/{} touched edge(s) created{}",
            created,
            pairs.len(),
            if errors > 0 {
                format!(" ({} errors)", errors)
            } else {
                String::new()
            }
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Fetch every node in one scope using pagination (the scoped sibling of
/// [`fetch_all_nodes`]; connect keeps its edges within one scope, the same
/// invariant `check_links` enforces for declared links).
fn fetch_nodes_in_scope(
    storage: &Arc<Storage>,
    scope: &str,
) -> anyhow::Result<Vec<vestige_core::KnowledgeNode>> {
    let mut all_nodes = Vec::new();
    let page_size = 500;
    let mut offset = 0;

    loop {
        let batch = storage.get_all_nodes_in_scope(scope, page_size, offset)?;
        let batch_len = batch.len();
        all_nodes.extend(batch);
        if batch_len < page_size as usize {
            break;
        }
        offset += page_size;
    }

    Ok(all_nodes)
}

/// Recall by exact handle (both stores), or by free text through the real
/// deep_reference engine (legacy SQLite stores only).
fn run_recall(
    query: Option<String>,
    handle: Option<String>,
    depth: i64,
    json: bool,
) -> anyhow::Result<()> {
    use vestige_mcp::cognitive::CognitiveEngine;

    let storage = open_storage()?;
    let rt = tokio::runtime::Runtime::new()?;

    // Handle mode goes through the MCP recall tool's `handle` argument, so
    // the CLI resolves exactly what the tool resolves.
    let recall_handle = |args: serde_json::Value| -> anyhow::Result<serde_json::Value> {
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        rt.block_on(vestige_mcp::tools::recall::execute(
            &storage,
            &cognitive,
            &vestige_core::OutputConfig::default(),
            Some(args),
        ))
        .map_err(|e| anyhow::anyhow!("recall error: {e}"))
    };

    if let Some(handle) = handle {
        let value = recall_handle(serde_json::json!({ "handle": handle }))?;
        return print_handle_recall(&handle, &value, json);
    }
    let query = query.context("pass a QUERY or --handle")?;

    if is_strata(&storage) {
        // Free text needs similarity, which a Strata log does not run.
        if json {
            // The MCP tool's handle_required payload, candidates included.
            let value = recall_handle(serde_json::json!({ "handle": "", "query": query }))?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        let suggestions = handle_suggestions(&storage, &query);
        let found = if suggestions.is_empty() {
            "No word in the text is a handle in this store.".to_string()
        } else {
            format!("Handles in your text: {}.", suggestions.join("; "))
        };
        anyhow::bail!(
            "similarity_disabled: free-text recall is not a Strata operation in Vestige 4.0 (no embeddings, BM25, FTS or keyword matching). Recall by exact handle instead: vestige recall --handle <memory id | unique id prefix of 8+ chars | exact tag>. {found}"
        );
    }

    let result = rt.block_on(async {
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        {
            let mut cog = cognitive.lock().await;
            cog.hydrate(&storage);
        }
        let args = serde_json::json!({ "query": query, "depth": depth });
        vestige_mcp::tools::cross_reference::execute(&storage, &cognitive, Some(args)).await
    });

    let value = result.map_err(|e| anyhow::anyhow!("recall error: {}", e))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    // Human-readable summary of the real engine output.
    let conf = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let intent = value
        .get("intent")
        .and_then(|v| v.as_str())
        .unwrap_or("Synthesis");
    let analyzed = value
        .get("memoriesAnalyzed")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    println!(
        "{}  intent={}  confidence={:.0}%  memories_analyzed={}",
        "Recall".cyan().bold(),
        intent,
        conf * 100.0,
        analyzed
    );

    if let Some(rec) = value.get("recommended") {
        let ans = rec
            .get("answer_preview")
            .or_else(|| rec.get("preview"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !ans.is_empty() {
            println!("\n{}", "Recommended:".white().bold());
            for line in ans.lines().take(6) {
                println!("  {}", line);
            }
        }
    }

    if let Some(ev) = value.get("evidence").and_then(|v| v.as_array()) {
        println!("\n{} ({})", "Evidence".white().bold(), ev.len());
        for (i, e) in ev.iter().take(5).enumerate() {
            let pv = e
                .get("preview")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .replace('\n', " ");
            let pv: String = pv.chars().take(78).collect();
            println!("  {}. {}", i + 1, pv);
        }
    }

    Ok(())
}

/// Words of `text` that resolve as exact handles, as `--handle` arguments.
/// Same tokens the MCP tool mines (the whole text, then up to eight
/// identifier-shaped words of 3+ chars), resolved exactly: no fuzzy match.
fn handle_suggestions(storage: &Arc<Storage>, text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut queries: Vec<&str> = vec![text];
    queries.extend(
        text.split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '/' | '-')))
            .filter(|token| token.len() >= 3)
            .take(8),
    );
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for query in queries {
        if !seen.insert(query) {
            continue;
        }
        let resolution = storage.resolve_handle(query);
        let count = resolution.ids.len();
        if count == 0 {
            continue;
        }
        out.push(format!(
            "--handle {query} ({}, {count} memor{})",
            resolution.kind.as_str(),
            if count == 1 { "y" } else { "ies" }
        ));
    }
    out
}

/// Most resolved memories printed for one handle; --json prints every one.
const HANDLE_PRINT_LIMIT: usize = 20;

/// Render the MCP recall tool's handle payload. An ambiguous or unmatched
/// handle is an error (non-zero exit), after the JSON when --json is set.
fn print_handle_recall(handle: &str, value: &serde_json::Value, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(value)?);
    }
    let candidates: Vec<&str> = value["candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["id"].as_str())
        .collect();
    match value["error"].as_str() {
        Some("ambiguous") => anyhow::bail!(
            "ambiguous: handle '{handle}' is a prefix of {} memory ids; pass a longer prefix or the full id. Candidates: {}",
            candidates.len(),
            candidates.join(", ")
        ),
        Some(error) => anyhow::bail!(
            "{error}: nothing matches handle '{handle}'. A handle is a memory id, a unique id prefix of 8+ characters, or an exact tag (case-sensitive); legacy SQLite stores also resolve commit shas, files, symbols, tests and run ids."
        ),
        None => {}
    }
    if json {
        return Ok(());
    }

    let nodes = value["nodes"].as_array().cloned().unwrap_or_default();
    println!(
        "{}  handle={}  kind={}  exact={}  {} memor{}",
        "Recall".cyan().bold(),
        handle,
        value["kind"].as_str().unwrap_or("?"),
        value["exact"],
        nodes.len(),
        if nodes.len() == 1 { "y" } else { "ies" }
    );
    for node in nodes.iter().take(HANDLE_PRINT_LIMIT) {
        let tags: Vec<&str> = node["tags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| t.as_str())
            .collect();
        println!();
        println!(
            "{}  [{}]  {} {}",
            node["id"].as_str().unwrap_or("?").white().bold(),
            node["type"].as_str().unwrap_or("?"),
            "tags:".dimmed(),
            tags.join(", ")
        );
        println!(
            "  {}",
            truncate(node["content"].as_str().unwrap_or(""), 300)
        );
    }
    if nodes.len() > HANDLE_PRINT_LIMIT {
        println!();
        println!(
            "  {} ... and {} more (pass --json for every memory)",
            "".dimmed(),
            nodes.len() - HANDLE_PRINT_LIMIT
        );
    }

    let neighbors = value["neighbors"].as_array().cloned().unwrap_or_default();
    println!();
    println!(
        "{} ({})",
        "Recorded edges, one hop".white().bold(),
        neighbors.len()
    );
    for edge in &neighbors {
        println!(
            "  {} -[{}]-> {}  {}",
            edge["from"].as_str().unwrap_or("?"),
            edge["link_type"].as_str().unwrap_or("?"),
            edge["to"].as_str().unwrap_or("?"),
            truncate(edge["node"]["content"].as_str().unwrap_or(""), 70).dimmed()
        );
    }
    Ok(())
}

/// Project the durable subset of a scope into a fenced rule-file region.
fn run_project(
    out: PathBuf,
    format: String,
    scope: String,
    min_retention: f64,
    max_items: usize,
    write: bool,
    json: bool,
) -> anyhow::Result<()> {
    use vestige_core::projection::{self, ProjectionFormat, ProjectionOptions};

    let format = ProjectionFormat::parse(&format)
        .ok_or_else(|| anyhow::anyhow!("unknown format '{format}'; use claude-md or memory-md"))?;
    let storage = open_storage()?;
    let projection = projection::project(
        &storage,
        &ProjectionOptions {
            scope: scope.clone(),
            format,
            min_retention: min_retention.clamp(0.0, 1.0),
            max_items: max_items.clamp(1, 500),
        },
    )?;
    let existed = out.exists();
    let existing = if existed {
        std::fs::read_to_string(&out)?
    } else {
        String::new()
    };
    let new_text = projection::splice(&existing, &projection.region);
    let diff = projection::line_diff(&existing, &new_text);
    let (added, removed) = projection::diff_summary(&diff);

    if write && (added > 0 || removed > 0) {
        projection::write_projection(&out, existed.then_some(existing.as_str()), &new_text)?;
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": out.display().to_string(),
                "format": format.label(),
                "scope": scope,
                "itemCount": projection.items.len(),
                "items": projection.items.iter().map(|i| serde_json::json!({
                    "id": i.id, "nodeType": i.node_type, "tags": i.tags,
                })).collect::<Vec<_>>(),
                "added": added,
                "removed": removed,
                "written": write && (added > 0 || removed > 0),
            }))?
        );
        return Ok(());
    }

    if added == 0 && removed == 0 {
        println!(
            "{} already holds this projection ({} memories from scope {}); nothing to change.",
            out.display(),
            projection.items.len(),
            scope
        );
        return Ok(());
    }
    if write {
        println!(
            "Wrote {} memories from scope {} into the fenced region of {} (+{} -{} lines). Everything outside the fence is untouched.",
            projection.items.len(),
            scope,
            out.display(),
            added,
            removed
        );
    } else {
        println!(
            "Projection of {} memories from scope {} for {} (+{} -{} lines). Re-run with --write to apply; only the fenced region changes.\n",
            projection.items.len(),
            scope,
            out.display(),
            added,
            removed
        );
        print!("{}", projection::unified(&diff, 400));
    }
    Ok(())
}

/// `vestige compose` on a Strata log: GhostLink `propose`, proofs included.
fn run_ghostlink_propose(
    storage: &Arc<Storage>,
    limit: i32,
    lens: Option<String>,
    tags: Option<String>,
    scope: Option<String>,
    json: bool,
) -> anyhow::Result<()> {
    use vestige_mcp::strata_memory::ghostlink::{Lens, ProposeRequest, propose};
    let lens = Lens::parse(lens.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    let scope = scope.map(|s| s.trim().to_string());
    if scope.as_deref().is_some_and(str::is_empty) {
        anyhow::bail!("--scope must not be empty");
    }
    let tags: Vec<String> = tags
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let request = ProposeRequest {
        lens,
        scope: Some(scope.unwrap_or_else(|| vestige_core::DEFAULT_MEMORY_SCOPE.to_string())),
        tags,
        limit: usize::try_from(limit.clamp(1, 100)).unwrap_or(5),
        cursor: None,
    };
    let answer =
        propose(storage.as_ref(), &request).map_err(|e| anyhow::anyhow!("compose error: {e}"))?;
    let candidates = answer["candidates"].as_array().cloned().unwrap_or_default();
    if json {
        // The 4.0.0 `--json` contract: an array of pairs with a_id / b_id and
        // the legacy keys. Each pair also carries its lens, lane and proof.
        let pairs: Vec<serde_json::Value> = candidates
            .iter()
            .map(|c| {
                serde_json::json!({
                    "a_id": c["firstId"],
                    "b_id": c["secondId"],
                    "score": c["score"],
                    "novelty": c.get("noveltyScore").cloned().unwrap_or(serde_json::Value::Null),
                    "bridge": c.get("bridgeScore").cloned().unwrap_or(serde_json::Value::Null),
                    "trust": c.get("trustScore").cloned().unwrap_or(serde_json::Value::Null),
                    "a": c["firstPreview"],
                    "b": c["secondPreview"],
                    "shared_tags": [],
                    "question": c["compositionQuestion"],
                    "reason": c["reason"],
                    "lens": c["lens"],
                    "lane": c.get("lane").cloned().unwrap_or(serde_json::Value::Null),
                    "hops": c.get("hops").cloned().unwrap_or(serde_json::Value::Null),
                    "pathMin": c.get("pathMin").cloned().unwrap_or(serde_json::Value::Null),
                    "proof": c["proof"],
                })
            })
            .collect();
        if pairs.is_empty()
            && let Some(why) = answer["admission"]["emptyBecause"].as_str()
        {
            eprintln!("compose: {why}");
        }
        println!("{}", serde_json::to_string_pretty(&pairs)?);
        return Ok(());
    }
    let scope_label = request.scope.as_deref().unwrap_or("user");
    if candidates.is_empty() {
        let why = answer["admission"]["emptyBecause"]
            .as_str()
            .unwrap_or("nothing qualifies under this lens");
        println!(
            "{}  {} lens: no pairs in scope {}: {}",
            "Compose".magenta().bold(),
            lens.as_str(),
            scope_label,
            why
        );
        return Ok(());
    }
    println!(
        "{}  {} lens: {} pair{} in scope {} (log seq {}; leads, not findings):\n",
        "Compose".magenta().bold(),
        lens.as_str(),
        candidates.len(),
        if candidates.len() == 1 { "" } else { "s" },
        scope_label,
        answer["headSeq"]
    );
    for (i, c) in candidates.iter().enumerate() {
        let score = c["score"]
            .as_f64()
            .map_or_else(|| "unmeasured".to_string(), |v| format!("{v:.2}"));
        println!(
            "{} {} / {}  [{}]",
            format!("{}.", i + 1).cyan().bold(),
            c["firstId"].as_str().unwrap_or("?"),
            c["secondId"].as_str().unwrap_or("?"),
            score
        );
        println!(
            "   A: {}",
            truncate(c["firstPreview"].as_str().unwrap_or(""), 70)
        );
        println!(
            "   B: {}",
            truncate(c["secondPreview"].as_str().unwrap_or(""), 70)
        );
        println!("   why: {}", c["reason"].as_str().unwrap_or(""));
        if let Some(path) = c["proof"]["path"]
            .as_array()
            .filter(|path| !path.is_empty())
        {
            // The proof itself: each recorded edge, in walk order.
            let mut line = path[0]["from"].as_str().unwrap_or("?").to_string();
            for step in path {
                let kind = step["kind"].as_str().unwrap_or("?");
                let arrow = if step["reversed"].as_bool().unwrap_or(false) {
                    format!(" <-{kind}- ")
                } else {
                    format!(" -{kind}-> ")
                };
                line.push_str(&arrow);
                line.push_str(step["to"].as_str().unwrap_or("?"));
            }
            println!("   path: {line}");
        }
        if let Some(q) = c["compositionQuestion"].as_str().filter(|q| !q.is_empty()) {
            println!("   Q: {q}");
        }
        println!();
    }
    Ok(())
}

/// Compose: list never-composed memory pairs.
fn run_compose(
    limit: i32,
    lens: Option<String>,
    tags: Option<String>,
    scope: Option<String>,
    json: bool,
) -> anyhow::Result<()> {
    let storage = open_storage()?;
    let strata = is_strata(&storage);
    if strata {
        return run_ghostlink_propose(&storage, limit, lens, tags, scope, json);
    }
    if lens.is_some() {
        anyhow::bail!(
            "--lens needs a Strata log (Vestige 4.0); a legacy SQLite store has one engine"
        );
    }

    let tag_vec: Option<Vec<String>> = tags.map(|t| {
        t.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let scope = scope.map(|s| s.trim().to_string());
    if scope.as_deref().is_some_and(str::is_empty) {
        anyhow::bail!("--scope must not be empty");
    }

    // The scoped call is the one the MCP `graph never_composed` action makes.
    // With no scope a Strata log lists `user` (its unscoped call lists
    // nothing), and a legacy SQLite store considers every scope, as before.
    let candidates = storage
        .get_never_composed_candidates_in_scope(limit, tag_vec.as_deref(), scope.as_deref())
        .map_err(|e| anyhow::anyhow!("compose error: {}", e))?;
    let scope_label = scope.clone().unwrap_or_else(|| {
        if strata {
            vestige_core::DEFAULT_MEMORY_SCOPE.to_string()
        } else {
            "every scope".to_string()
        }
    });

    if json {
        let arr: Vec<_> = candidates
            .iter()
            .map(|c| {
                serde_json::json!({
                    "a_id": c.first_id,
                    "b_id": c.second_id,
                    "score": c.score,
                    "novelty": c.novelty_score,
                    "bridge": c.bridge_score,
                    "trust": c.trust_score,
                    "a": c.first_preview,
                    "b": c.second_preview,
                    "shared_tags": c.shared_tags,
                    "question": c.composition_question,
                    "reason": c.reason,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }

    if candidates.is_empty() {
        println!(
            "{}  no never-composed pairs in {} (try a wider --limit, another --scope, or remove --tags)",
            "Compose".magenta().bold(),
            scope_label
        );
        return Ok(());
    }

    println!(
        "{}  {} never-composed pair{} in {}, linked by recorded causal edges but never composed:\n",
        "Compose".magenta().bold(),
        candidates.len(),
        if candidates.len() == 1 { "" } else { "s" },
        scope_label
    );

    for (i, c) in candidates.iter().enumerate() {
        let a: String = c
            .first_preview
            .replace('\n', " ")
            .chars()
            .take(70)
            .collect();
        let b: String = c
            .second_preview
            .replace('\n', " ")
            .chars()
            .take(70)
            .collect();
        let idx = format!("{}.", i + 1).cyan().bold();
        let metrics = format!(
            "{:.2}  (novelty {:.2}, bridge {:.2})",
            c.score, c.novelty_score, c.bridge_score
        );
        println!("{} {} {}", idx, "score".white(), metrics);
        println!("   A: {}", a);
        println!("   B: {}", b);
        let q: String = c
            .composition_question
            .replace('\n', " ")
            .chars()
            .take(120)
            .collect();
        if !q.is_empty() {
            println!("   {} {}", "?".yellow().bold(), q.yellow());
        }
        println!();
    }

    Ok(())
}

/// Run the dashboard web server
fn run_dashboard(port: u16, open_browser: bool) -> anyhow::Result<()> {
    use vestige_mcp::cognitive::CognitiveEngine;

    println!("{}", "=== Vestige Dashboard ===".cyan().bold());
    println!();

    let dir = cli_data_dir()?;
    // Same check `vestige-mcp` runs before stdio. It runs before the lock:
    // the upgrade helper takes that lock itself.
    vestige_mcp::v3_launch::upgrade_or_refuse(&dir.join("vestige.db"))?;
    let rt = tokio::runtime::Runtime::new()?;
    let mut open_browser = open_browser;

    // Usually an agent's vestige-mcp holds the store. That process serves the
    // dashboard on request, for as long as this command runs (it stops the
    // dashboard when the last `vestige dashboard` using it exits); if it
    // exits, this process takes the store and serves the dashboard itself.
    let wait = vestige_mcp::attach::election_wait();
    let mut deadline = std::time::Instant::now() + wait;
    while !take_cli_lock(&dir)? {
        match rt.block_on(vestige_mcp::attach::request_dashboard(&dir, port)) {
            Ok(lease) => {
                println!("Dashboard: {}", lease.url.cyan());
                println!(
                    "  {} served by vestige-mcp (pid {}), the Vestige server your agents use",
                    ">".cyan(),
                    lease.owner_pid
                );
                // That server runs one dashboard. When it already serves one
                // on another port, say so rather than drop --port silently.
                if let Some(serving) = url_port(&lease.url)
                    && serving != port
                {
                    println!(
                        "  {} --port {port} was not used: that server already serves the dashboard on port {serving}",
                        "!".yellow()
                    );
                }
                if open_browser {
                    let _ = open::that(&lease.url);
                    open_browser = false;
                }
                println!("{}", "Press Ctrl+C to stop.".dimmed());
                rt.block_on(lease.closed());
                println!("That Vestige server exited; moving the dashboard here...");
                deadline = std::time::Instant::now() + wait;
            }
            Err(err) => {
                // A definite refusal (the port is taken, say) is not retried.
                if err.kind() == std::io::ErrorKind::ConnectionRefused {
                    anyhow::bail!(
                        "the Vestige server holding {} could not start the dashboard: {err}",
                        dir.display()
                    );
                }
                if std::time::Instant::now() >= deadline {
                    anyhow::bail!(
                        "{} holds {} and did not start the dashboard: {err}",
                        store_holder(&dir),
                        dir.display()
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }

    println!(
        "Starting dashboard at {}...",
        format!("http://127.0.0.1:{}", port).cyan()
    );
    let storage = open_storage()?;
    rt.block_on(async move {
        // Initialize cognitive engine for dream and other cognitive features
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        {
            let mut cog = cognitive.lock().await;
            cog.hydrate(&storage); // Load persisted connections
        }
        let (event_tx, _) = tokio::sync::broadcast::channel::<
            vestige_mcp::dashboard::events::VestigeEvent,
        >(vestige_mcp::dashboard::state::EVENT_CHANNEL_CAPACITY);
        let dashboard = vestige_mcp::dashboard::DashboardOnDemand::new(
            Arc::clone(&storage),
            Arc::clone(&cognitive),
            event_tx.clone(),
        );
        let running = dashboard
            .ensure(port)
            .await
            .map_err(|e| anyhow::anyhow!("Dashboard error: {}", e))?;
        // This process holds the store, so MCP clients started meanwhile
        // attach here instead of waiting for it to exit.
        let _attach_point =
            open_cli_attach_point(&storage, &cognitive, &event_tx, Some(dashboard.starter())).await;
        let url = format!("http://127.0.0.1:{running}");
        println!("Dashboard: {}", url.cyan());
        if open_browser {
            let _ = open::that(&url);
        }
        println!("{}", "Press Ctrl+C to stop.".dimmed());
        tokio::signal::ctrl_c().await.ok();
        Ok(())
    })
}

/// The port in a `http://host:port` URL.
fn url_port(url: &str) -> Option<u16> {
    url.rsplit_once(':')?.1.trim_end_matches('/').parse().ok()
}

/// Start standalone HTTP MCP server (no stdio transport)
fn run_serve(port: u16, with_dashboard: bool, dashboard_port: u16) -> anyhow::Result<()> {
    use vestige_mcp::cognitive::CognitiveEngine;

    println!("{}", "=== Vestige HTTP Server ===".cyan().bold());
    println!();

    let storage = open_storage()?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        {
            let mut cog = cognitive.lock().await;
            cog.hydrate(&storage);
        }

        let (event_tx, _) = tokio::sync::broadcast::channel::<
            vestige_mcp::dashboard::events::VestigeEvent,
        >(vestige_mcp::dashboard::state::EVENT_CHANNEL_CAPACITY);

        let dashboard = vestige_mcp::dashboard::DashboardOnDemand::new(
            Arc::clone(&storage),
            Arc::clone(&cognitive),
            event_tx.clone(),
        );
        // Optionally start dashboard
        if with_dashboard {
            let dashboard = dashboard.clone();
            tokio::spawn(async move {
                match dashboard.ensure(dashboard_port).await {
                    Ok(port) => println!("  {} Dashboard: http://127.0.0.1:{}", ">".cyan(), port),
                    Err(e) => eprintln!("  {} Dashboard failed: {}", "!".yellow(), e),
                }
            });
        }

        // This process holds the store, so MCP stdio clients started
        // meanwhile attach here instead of waiting for it to exit.
        let _attach_point =
            open_cli_attach_point(&storage, &cognitive, &event_tx, Some(dashboard.starter())).await;

        // Get auth token
        let token = vestige_mcp::protocol::auth::get_or_create_auth_token()
            .map_err(|e| anyhow::anyhow!("Failed to create auth token: {}", e))?;

        let bind = std::env::var("VESTIGE_HTTP_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
        println!(
            "  {} HTTP transport: http://{}:{}/mcp",
            ">".cyan(),
            bind,
            port
        );
        if let Ok(path) = vestige_mcp::protocol::auth::token_path() {
            println!("  {} Auth token file: {}", ">".cyan(), path.display());
        }
        println!();
        println!("{}", "Press Ctrl+C to stop.".dimmed());

        // Start HTTP transport (blocks on the server, no stdio)
        vestige_mcp::protocol::http::start_http_transport(
            Arc::clone(&storage),
            Arc::clone(&cognitive),
            event_tx,
            token,
            port,
        )
        .await
        .map_err(|e| anyhow::anyhow!("HTTP transport failed: {}", e))?;

        // Keep the process alive (the HTTP server runs in a spawned task)
        tokio::signal::ctrl_c().await.ok();
        println!();
        println!("{}", "Shutting down...".dimmed());

        Ok(())
    })
}

/// Accept attached MCP sessions for as long as this long-running command
/// holds the store. A failure only means clients wait for this to exit.
async fn open_cli_attach_point(
    storage: &Arc<Storage>,
    cognitive: &Arc<tokio::sync::Mutex<vestige_mcp::cognitive::CognitiveEngine>>,
    event_tx: &tokio::sync::broadcast::Sender<vestige_mcp::dashboard::events::VestigeEvent>,
    dashboard: Option<vestige_mcp::attach::DashboardStarter>,
) -> Option<vestige_mcp::attach::AttachPoint> {
    let dir = cli_data_dir().ok()?;
    let storage = Arc::clone(storage);
    let cognitive = Arc::clone(cognitive);
    let event_tx = event_tx.clone();
    match vestige_mcp::attach::AttachPoint::open(
        &dir,
        move || {
            vestige_mcp::server::McpServer::new_with_events(
                Arc::clone(&storage),
                Arc::clone(&cognitive),
                event_tx.clone(),
            )
        },
        dashboard,
    )
    .await
    {
        Ok(point) => Some(point),
        Err(err) => {
            eprintln!(
                "  {} MCP clients cannot attach while this runs: {}",
                "!".yellow(),
                err
            );
            None
        }
    }
}

/// Truncate a string for display (UTF-8 safe)
fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_chars {
        s
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    #[test]
    fn update_asset_mapping_matches_release_names() {
        let mac_arm = release_asset_for("macos", "aarch64").unwrap();
        assert_eq!(mac_arm.target, "aarch64-apple-darwin");
        assert_eq!(mac_arm.archive_ext, "tar.gz");
        assert_eq!(mac_arm.binary_suffix, "");

        let linux = release_asset_for("linux", "x86_64").unwrap();
        assert_eq!(linux.target, "x86_64-unknown-linux-gnu");
        assert_eq!(linux.archive_ext, "tar.gz");

        let linux_arm = release_asset_for("linux", "aarch64").unwrap();
        assert_eq!(linux_arm.target, "aarch64-unknown-linux-gnu");
        assert_eq!(linux_arm.archive_ext, "tar.gz");
        assert_eq!(linux_arm.binary_suffix, "");

        let windows = release_asset_for("windows", "x86_64").unwrap();
        assert_eq!(windows.target, "x86_64-pc-windows-msvc");
        assert_eq!(windows.archive_ext, "zip");
        assert_eq!(windows.binary_suffix, ".exe");
    }

    #[test]
    fn update_url_uses_latest_or_normalized_tag() {
        let asset = release_asset_for("macos", "aarch64").unwrap();
        assert_eq!(
            release_download_url(asset, None),
            "https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            release_download_url(asset, Some("2.1.0")),
            "https://github.com/samvallad33/vestige/releases/download/v2.1.0/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            release_download_url(asset, Some("v2.1.0")),
            "https://github.com/samvallad33/vestige/releases/download/v2.1.0/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    fn source_archive_url_uses_normalized_tag() {
        assert_eq!(normalize_release_tag("2.1.1"), "v2.1.1");
        assert_eq!(normalize_release_tag("v2.1.1"), "v2.1.1");
        assert_eq!(
            source_archive_url("v2.1.1"),
            "https://github.com/samvallad33/vestige/archive/refs/tags/v2.1.1.tar.gz"
        );
    }

    #[test]
    fn strata_verify_is_a_subcommand() {
        let cli = Cli::try_parse_from(["vestige", "strata-verify", "/tmp/store"]).unwrap();
        assert!(matches!(cli.command, Commands::StrataVerify { .. }));
    }

    #[test]
    fn scrub_vestige_hooks_removes_only_vestige_commands() {
        let mut settings = serde_json::json!({
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            { "type": "command", "command": "/tmp/synthesis-preflight.sh" },
                            { "type": "command", "command": "/tmp/custom-user-hook.sh" }
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            { "type": "command", "command": "/tmp/sanhedrin.sh" }
                        ]
                    }
                ]
            },
            "other": true
        });

        scrub_vestige_hooks(&mut settings);

        let user_hooks = settings["hooks"]["UserPromptSubmit"][0]["hooks"]
            .as_array()
            .unwrap();
        assert_eq!(user_hooks.len(), 1);
        assert_eq!(user_hooks[0]["command"], "/tmp/custom-user-hook.sh");
        assert!(settings["hooks"].get("Stop").is_none());
        assert_eq!(settings["other"], true);
    }
}

/// Help text and argument shape for the 4.0 (Strata) build. Runs without
/// `legacy-sqlite`; the behavior itself is covered by `tests/cli_strata.rs`.
#[cfg(test)]
mod strata_cli_tests {
    use super::*;
    use clap::CommandFactory;

    fn long_help(path: &[&str]) -> String {
        let mut command = Cli::command();
        for name in path {
            command = command
                .find_subcommand(name)
                .unwrap_or_else(|| panic!("no subcommand {name}"))
                .clone();
        }
        command.render_long_help().to_string()
    }

    #[test]
    fn help_text_makes_no_v3_only_claims() {
        let banned = [
            "semantic-band",
            "hybrid search",
            "full backup of the SQLite database",
            "Prediction Error Gating",
            "vector search",
            "SEMANTIC SEARCH",
            "synaptic tagging",
            "rehearse them on a copy",
            "entities the backfill joins on",
        ];
        let mut pages = vec![long_help(&[])];
        for sub in Cli::command().get_subcommands() {
            pages.push(long_help(&[sub.get_name()]));
        }
        for page in &pages {
            for claim in banned {
                assert!(
                    !page.contains(claim),
                    "help still claims {claim:?}:\n{page}"
                );
            }
        }
    }

    #[test]
    fn strata_help_names_what_works() {
        let backup = long_help(&["backup"]);
        assert!(backup.contains("Strata log"), "{backup}");
        assert!(backup.contains("vestige.db is never copied"), "{backup}");
        let backfill = long_help(&["backfill"]);
        assert!(backfill.contains("vestige causal-walk"), "{backfill}");
        let recall = long_help(&["recall"]);
        assert!(recall.contains("--handle"), "{recall}");
        assert!(recall.contains("exact tag"), "{recall}");
        for sub in ["portable-export", "portable-import", "sync"] {
            let page = long_help(&[sub]);
            assert!(page.contains("legacy SQLite stores only"), "{sub}: {page}");
        }
        let compose = long_help(&["compose"]);
        assert!(compose.contains("no recorded edge"), "{compose}");
    }

    #[test]
    fn recall_takes_a_query_or_a_handle_but_not_both() {
        let handle = Cli::try_parse_from(["vestige", "recall", "--handle", "mem-1"]).unwrap();
        assert!(matches!(
            handle.command,
            Commands::Recall {
                query: None,
                handle: Some(_),
                ..
            }
        ));
        let query = Cli::try_parse_from(["vestige", "recall", "what broke"]).unwrap();
        assert!(matches!(
            query.command,
            Commands::Recall {
                query: Some(_),
                handle: None,
                ..
            }
        ));
        assert!(Cli::try_parse_from(["vestige", "recall"]).is_err());
        assert!(Cli::try_parse_from(["vestige", "recall", "q", "--handle", "h"]).is_err());
    }

    #[test]
    fn compose_takes_an_optional_scope() {
        let cli = Cli::try_parse_from(["vestige", "compose", "--scope", "proj"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Compose { scope: Some(ref s), .. } if s == "proj"
        ));
    }

    #[test]
    fn backup_destination_inside_the_log_is_detected_before_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        std::fs::create_dir_all(&log).unwrap();
        let canonical_log = std::fs::canonicalize(&log).unwrap();
        let inside = resolve_existing_prefix(&log.join("a").join("b")).unwrap();
        assert!(inside.starts_with(&canonical_log), "{inside:?}");
        let beside = resolve_existing_prefix(&dir.path().join("backups").join("x")).unwrap();
        assert!(!beside.starts_with(&canonical_log), "{beside:?}");
        assert!(
            !dir.path().join("backups").exists(),
            "resolving created a directory"
        );
    }

    #[cfg(unix)]
    #[test]
    fn copied_backup_tree_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("made");
        std::fs::create_dir_all(from.join("log")).unwrap();
        std::fs::write(from.join("log").join("a.seg"), [1u8; 4]).unwrap();
        std::fs::write(from.join("store.meta"), [2u8; 4]).unwrap();
        let to = dir.path().join("copy");
        copy_dir_all(&from, &to).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&to), 0o700);
        assert_eq!(mode(&to.join("log")), 0o700);
        assert_eq!(mode(&to.join("log").join("a.seg")), 0o600);
        assert_eq!(mode(&to.join("store.meta")), 0o600);
        assert_eq!(std::fs::read(to.join("store.meta")).unwrap(), [2u8; 4]);
    }

    #[test]
    fn path_size_sums_a_directory_copy() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("log")).unwrap();
        std::fs::write(dir.path().join("log").join("a.seg"), [0u8; 10]).unwrap();
        std::fs::write(dir.path().join("store.meta"), [0u8; 5]).unwrap();
        assert_eq!(path_size(dir.path()), 15);
        assert_eq!(format_size(15), "15 bytes");
        assert_eq!(format_size(2048), "2.0 KB");
    }
}
