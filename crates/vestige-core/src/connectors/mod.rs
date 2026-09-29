//! External-source connectors (#57).
//!
//! A connector turns records in a long-lived external system (a ticket tracker,
//! an issue board, a support queue) into source-aware Vestige memories, so an
//! investigative agent can search and reason over years of history **offline**,
//! **semantically**, and **cited back to the canonical record** — something no
//! live ticket-system MCP proxy can do.
//!
//! ## Layering
//!
//! - The [`Connector`] contract, [`NormalizedRecord`] shape, and the stable
//!   [`content_hash`] are pure (no network) and always compiled, so the sync
//!   semantics are unit-testable without hitting an API.
//! - Network-backed reference connectors ([`github`] and [`redmine`]) live
//!   behind the `connectors` cargo feature so the default local-first build
//!   links no HTTP client.
//!
//! ## Sync contract (the part that makes re-running safe)
//!
//! Every connector produces [`NormalizedRecord`]s. Each carries a
//! [`SourceEnvelope`](crate::memory::SourceEnvelope) whose
//! `(source_system, source_id)` is the idempotency key and whose `content_hash`
//! is the change detector. The driver routes each record through
//! [`upsert_by_source`](crate::storage::SqliteMemoryStore::upsert_by_source):
//!
//! - unseen record → insert
//! - changed `content_hash` → update in place (+ re-embed)
//! - same `content_hash` → no-op (only liveness advances)
//!
//! Because neither GitHub nor Redmine expose a deletion feed, deletions are
//! handled out-of-band by a periodic reconcile pass
//! ([`reconcile_source_tombstones`](crate::storage::SqliteMemoryStore::reconcile_source_tombstones)).

use chrono::{DateTime, Utc};

use crate::memory::{IngestInput, SourceEnvelope};
// Connector cursor state persists through the legacy SQLite store
// (build/t5-legacy-isolation).
#[cfg(feature = "legacy-sqlite")]
use crate::storage::ConnectorCursor;

#[cfg(feature = "connectors")]
pub mod github;

#[cfg(feature = "connectors")]
pub mod redmine;

/// A single external record, already normalized into the fields Vestige needs.
///
/// The connector is responsible for flattening a possibly-rich source record
/// (an issue plus its comments / journals / status changes) into a single
/// retrievable `content` blob plus the structured envelope. Keeping one memory
/// per logical record (rather than per comment) keeps retrieval coherent and
/// the idempotency key simple.
#[derive(Debug, Clone)]
pub struct NormalizedRecord {
    /// Human-readable content to embed and search over.
    pub content: String,
    /// Tags for categorization (e.g. `["github", "issue", "state:open"]`).
    pub tags: Vec<String>,
    /// The provenance envelope. `source_system`, `source_id`, and `content_hash`
    /// MUST be set for idempotent upsert.
    pub envelope: SourceEnvelope,
}

impl NormalizedRecord {
    /// Convert into an [`IngestInput`] ready for `upsert_by_source`.
    pub fn into_ingest_input(self) -> IngestInput {
        IngestInput {
            content: self.content,
            node_type: "event".to_string(),
            source: self.envelope.source_url.clone(),
            tags: self.tags,
            source_envelope: Some(self.envelope),
            ..Default::default()
        }
    }
}

/// One page of records plus the cursor needed to fetch the next page.
#[derive(Debug, Clone, Default)]
pub struct FetchPage {
    pub records: Vec<NormalizedRecord>,
    /// Opaque token to resume after this page, or `None` when exhausted.
    pub next_cursor: Option<String>,
    /// Records the connector dropped mid-page for a NON-FATAL reason (e.g. the
    /// comment fetch for one issue failed after a retry). A skipped record is
    /// NOT persisted this run; the driver clamps the run cursor to before the
    /// record's `source_updated_at` so the next run re-fetches it instead of
    /// silently losing it. Fatal page failures still abort via `Err`.
    pub skipped: Vec<SkippedRecord>,
}

/// One record dropped mid-page for a non-fatal reason (see [`FetchPage::skipped`]).
#[derive(Debug, Clone)]
pub struct SkippedRecord {
    /// The record's `source_updated_at`, used to clamp the run cursor so the
    /// next sync re-fetches it. `None` means "unknown" — the driver then does
    /// not advance the cursor past the oldest failure it does know about.
    pub source_updated_at: Option<DateTime<Utc>>,
    /// Human-readable reason, surfaced in `SyncReport::warnings`.
    pub reason: String,
}

/// Errors a connector can surface.
#[derive(Debug, thiserror::Error)]
pub enum ConnectorError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("rate limited by upstream (Retry-After {0:?}; None means the server gave no header — back off and retry later)")]
    RateLimited(Option<std::time::Duration>),
    #[error("source error ({status}): {message}")]
    Source { status: u16, message: String },
}

pub type ConnectorResult<T> = Result<T, ConnectorError>;

/// The contract every external-source connector implements.
///
/// Intentionally minimal: fetch a window of records updated since a cursor,
/// page through them, and (separately) enumerate currently-live ids for the
/// deletion-reconcile pass. The driver owns persistence, embedding, and cursor
/// checkpointing — a connector is just a typed, incremental reader.
#[allow(async_fn_in_trait)]
pub trait Connector {
    /// Stable system identifier written into every envelope (`github`, …).
    fn source_system(&self) -> &str;

    /// The scope this connector instance is bound to (`owner/repo`, project id).
    fn scope(&self) -> &str;

    /// Fetch one page of records whose source-updated time is `>= since`
    /// (inclusive on purpose — see the overlap note below), resuming from
    /// `cursor` when provided. Records should be returned in ascending
    /// update-time order so a mid-run interruption resumes safely.
    ///
    /// Callers pass `since = checkpoint − overlap` (a few minutes) so a record
    /// written with a slightly-behind upstream clock, or one sharing the exact
    /// boundary second, is never skipped. The `content_hash` short-circuit in
    /// `upsert_by_source` makes the resulting re-scan free.
    async fn fetch_updated(
        &self,
        since: Option<DateTime<Utc>>,
        cursor: Option<String>,
    ) -> ConnectorResult<FetchPage>;

    /// Enumerate the ids currently visible upstream for this scope, for the
    /// deletion-reconcile pass. Cheap (ids only). `None` means the connector
    /// cannot enumerate, so the driver must skip reconciliation rather than
    /// tombstone everything.
    async fn list_live_ids(&self) -> ConnectorResult<Option<Vec<String>>> {
        Ok(None)
    }
}

/// Recommended overlap subtracted from the saved cursor before the next fetch,
/// to absorb clock skew and same-second boundary updates (the `>=` window).
pub const CURSOR_OVERLAP_SECS: i64 = 120;

/// Longest rate-limit backoff the driver will sleep through inside one tool
/// call. GitHub's *primary* limit can take up to an hour to reset — waiting
/// that long inside an MCP request is hostile to the caller, so anything
/// longer than this aborts the run with a `RateLimited` error that names the
/// wait, and the saved per-page checkpoints make the retry a cheap resume.
#[allow(dead_code)] // used by the github test/legacy-sqlite profile; dead only in the no-embeddings build
const MAX_RATE_LIMIT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);

/// One page fetch with rate-limit respect: a single bounded retry after the
/// server's `Retry-After` when that wait is short enough to sleep through.
/// Anything longer (or a second consecutive 429) aborts so the caller decides.
#[allow(dead_code)] // same profile note as MAX_RATE_LIMIT_BACKOFF
async fn fetch_page_with_retry<C: Connector>(
    connector: &C,
    since: Option<DateTime<Utc>>,
    cursor: Option<String>,
) -> ConnectorResult<FetchPage> {
    match connector.fetch_updated(since, cursor.clone()).await {
        Err(ConnectorError::RateLimited(Some(wait))) if wait <= MAX_RATE_LIMIT_BACKOFF => {
            tracing::warn!(wait_secs = wait.as_secs(), "rate limited; backing off once");
            tokio::time::sleep(wait).await;
            connector.fetch_updated(since, cursor).await
        }
        other => other,
    }
}

/// Summary of one sync run, returned to the caller / surfaced by the MCP tool.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SyncReport {
    pub source_system: String,
    pub scope: String,
    pub created: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub tombstoned: usize,
    /// New high-water mark persisted as the cursor for the next run.
    pub new_cursor: Option<DateTime<Utc>>,
    /// Whether a deletion-reconcile pass ran this time.
    pub reconciled: bool,
    /// Non-fatal warnings (e.g. a page that failed and was skipped).
    pub warnings: Vec<String>,
}

/// Drive a full incremental sync of one connector into the store (#57).
///
/// This is the orchestration the MCP `source_sync` tool calls. It:
/// 1. loads the saved checkpoint and starts from `cursor − overlap` (the `>=`
///    window that prevents missing same-second / clock-skewed updates);
/// 2. pages the connector forward in update order, routing each record through
///    [`upsert_by_source`](crate::storage::SqliteMemoryStore::upsert_by_source)
///    (insert / update-in-place / no-op by content hash);
/// 3. advances the cursor to the max `source_updated_at` actually observed,
///    persisting the checkpoint after every completed page, so an interrupted
///    run (network drop, mid-sync shutdown) resumes from the last good page
///    instead of re-fetching the whole window; a crash re-scans rather than
///    skips;
/// 4. optionally reconciles deletions when `reconcile` is set and the connector
///    can enumerate live ids.
///
/// `max_pages` bounds a single run (so a first sync of a 15-year tracker can be
/// resumed across calls rather than blocking on one enormous fetch).
#[cfg(feature = "legacy-sqlite")]
pub async fn run_sync<C: Connector>(
    store: &crate::storage::Storage,
    connector: &C,
    reconcile: bool,
    max_pages: usize,
) -> ConnectorResult<SyncReport> {
    use crate::storage::SourceUpsertOutcome;

    let source_system = connector.source_system().to_string();
    let scope = connector.scope().to_string();

    let mut report = SyncReport {
        source_system: source_system.clone(),
        scope: scope.clone(),
        ..Default::default()
    };

    // 1. Load checkpoint, apply the overlap window.
    let checkpoint = store
        .get_connector_cursor(&source_system, &scope)
        .map_err(|e| ConnectorError::Transport(e.to_string()))?;
    let since = checkpoint
        .cursor_updated_at
        .map(|c| c - chrono::Duration::seconds(CURSOR_OVERLAP_SECS));

    // 2. Page forward, upserting each record. The checkpoint is persisted after
    // EVERY completed page (not just at the end of the run): every record is
    // already durably upserted by then, so saving per page makes an interrupted
    // multi-page sync resume from the last good page instead of re-fetching the
    // whole window.
    let mut cursor: Option<String> = None;
    let mut max_seen = checkpoint.cursor_updated_at;
    // Oldest source_updated_at among records that FAILED to upsert (or were
    // skipped mid-page) this run. We must not advance the persisted cursor past
    // this, or the failed record — fetched in ascending update order — would
    // fall outside the next run's `since` window and never be retried (a silent
    // permanent gap).
    let mut oldest_failure: Option<DateTime<Utc>> = None;
    // Count of genuinely new records (Created). Unchanged re-scans of the
    // overlap window must not inflate the running total.
    let mut created_this_run = 0i64;

    let save_checkpoint = |max_seen: Option<DateTime<Utc>>,
                           created_this_run: i64,
                           reconciled_at: Option<DateTime<Utc>>|
     -> ConnectorResult<()> {
        let new_checkpoint = ConnectorCursor {
            source_system: source_system.clone(),
            scope: scope.clone(),
            cursor_updated_at: max_seen,
            last_synced_at: Some(Utc::now()),
            last_full_reconcile_at: reconciled_at.or(checkpoint.last_full_reconcile_at),
            // Accumulate only NEW records, so re-scanning the overlap window
            // (which reports Unchanged) does not inflate the running total.
            records_seen: checkpoint.records_seen + created_this_run,
        };
        store
            .save_connector_cursor(&new_checkpoint)
            .map_err(|e| ConnectorError::Transport(e.to_string()))
    };

    for _ in 0..max_pages.max(1) {
        let page = fetch_page_with_retry(connector, since, cursor.clone()).await?;
        for record in page.records {
            let observed = record.envelope.source_updated_at;
            match store.upsert_by_source(record.into_ingest_input()) {
                Ok(res) => {
                    match res.outcome {
                        SourceUpsertOutcome::Created => {
                            report.created += 1;
                            created_this_run += 1;
                        }
                        SourceUpsertOutcome::Updated => report.updated += 1,
                        SourceUpsertOutcome::Unchanged => report.unchanged += 1,
                    }
                    if let Some(ts) = observed
                        && max_seen.map(|m| ts > m).unwrap_or(true)
                    {
                        max_seen = Some(ts);
                    }
                }
                Err(e) => {
                    report.warnings.push(format!("upsert failed: {e}"));
                    if let Some(ts) = observed
                        && oldest_failure.map(|f| ts < f).unwrap_or(true)
                    {
                        oldest_failure = Some(ts);
                    }
                }
            }
        }
        for skipped in page.skipped {
            report.warnings.push(format!(
                "skipped record (will retry next sync): {}",
                skipped.reason
            ));
            if let Some(ts) = skipped.source_updated_at
                && oldest_failure.map(|f| ts < f).unwrap_or(true)
            {
                oldest_failure = Some(ts);
            }
        }

        // Clamp for the per-page save so a mid-run abort never leaves the
        // cursor past a record that failed on an earlier page.
        let clamped = clamp_cursor(max_seen, oldest_failure);
        if let Err(e) = save_checkpoint(clamped, created_this_run, None) {
            report
                .warnings
                .push(format!("checkpoint save failed mid-run: {e}"));
            break;
        }

        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }

    // Clamp the cursor so we never advance past a record that failed this run.
    // Subtract one second so the next run's inclusive `since` re-includes it.
    max_seen = clamp_cursor(max_seen, oldest_failure);

    // 3. Optional deletion reconciliation.
    let mut reconciled = false;
    if reconcile {
        match connector.list_live_ids().await {
            // CATASTROPHIC-DATA-LOSS GUARD: an empty live-id set would tombstone
            // EVERY stored memory for this source (none of them appear in the
            // empty list). An empty result almost always means a transient/auth
            // failure or an over-narrow scope, not "the source truly has zero
            // issues". Treat it like None (cannot safely enumerate) and skip.
            Ok(Some(live_ids)) if live_ids.is_empty() => report.warnings.push(
                "list_live_ids returned an empty set; skipping reconcile to avoid \
                 mass-tombstoning the entire source"
                    .to_string(),
            ),
            Ok(Some(live_ids)) => {
                match store.reconcile_source_tombstones(&source_system, &scope, &live_ids) {
                    Ok(r) => {
                        report.tombstoned = r.tombstoned.len();
                        reconciled = true;
                    }
                    Err(e) => report.warnings.push(format!("reconcile failed: {e}")),
                }
            }
            Ok(None) => report
                .warnings
                .push("connector cannot enumerate live ids; skipped reconcile".to_string()),
            Err(e) => report.warnings.push(format!("list_live_ids failed: {e}")),
        }
    }
    report.reconciled = reconciled;
    report.new_cursor = max_seen;

    // 4. Persist the final checkpoint (the authoritative one, including
    // reconcile bookkeeping).
    let now = Utc::now();
    save_checkpoint(max_seen, created_this_run, reconciled.then_some(now))?;

    Ok(report)
}

/// Highest cursor value it is safe to persist: never past the oldest record
/// that failed this run (minus one second so the next inclusive-`since` run
/// re-fetches it).
#[allow(dead_code)] // same profile note as MAX_RATE_LIMIT_BACKOFF
fn clamp_cursor(
    max_seen: Option<DateTime<Utc>>,
    oldest_failure: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    let Some(failed_at) = oldest_failure else {
        return max_seen;
    };
    let clamp_to = failed_at - chrono::Duration::seconds(1);
    Some(match max_seen {
        Some(m) if m < clamp_to => m,
        _ => clamp_to,
    })
}

/// Compute a stable content hash over the record's meaning.
///
/// Stability requirements (so re-syncing an unchanged record is a true no-op):
/// - **key order independent** — callers pass `(field, value)` pairs which we
///   sort before hashing, so map/field ordering never changes the digest;
/// - **volatile fields excluded** — the caller must omit the cursor timestamp,
///   view/comment counts, and ephemeral permission flags (hash the meaning,
///   not the metadata);
/// - **collision-resistant** — BLAKE3 (already a Vestige dependency).
///
/// Comment/journal arrays should be flattened into the pairs in a stable order
/// (sorted by their own id) by the caller before hashing.
pub fn content_hash(fields: &[(&str, &str)]) -> String {
    let mut pairs: Vec<(&str, &str)> = fields.to_vec();
    pairs.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));

    let mut hasher = blake3::Hasher::new();
    for (k, v) in pairs {
        // Length-prefix each field so ("ab","c") can't collide with ("a","bc").
        hasher.update(&(k.len() as u64).to_le_bytes());
        hasher.update(k.as_bytes());
        hasher.update(&(v.len() as u64).to_le_bytes());
        hasher.update(v.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(all(test, feature = "legacy-sqlite"))]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_order_independent() {
        let a = content_hash(&[
            ("title", "Crash"),
            ("body", "stacktrace"),
            ("state", "open"),
        ]);
        let b = content_hash(&[
            ("state", "open"),
            ("title", "Crash"),
            ("body", "stacktrace"),
        ]);
        assert_eq!(a, b, "reordering fields must not change the hash");
    }

    #[test]
    fn content_hash_changes_with_content() {
        let a = content_hash(&[("body", "v1")]);
        let b = content_hash(&[("body", "v2")]);
        assert_ne!(a, b, "different content must hash differently");
    }

    #[test]
    fn content_hash_no_boundary_collision() {
        // ("ab","c") vs ("a","bc") must differ thanks to length prefixing.
        let a = content_hash(&[("ab", "c")]);
        let b = content_hash(&[("a", "bc")]);
        assert_ne!(a, b);
    }

    #[test]
    fn normalized_record_carries_envelope_into_input() {
        let rec = NormalizedRecord {
            content: "issue body".to_string(),
            tags: vec!["github".to_string()],
            envelope: SourceEnvelope {
                source_system: Some("github".to_string()),
                source_id: Some("42".to_string()),
                source_url: Some("https://example/42".to_string()),
                content_hash: Some("h".to_string()),
                ..Default::default()
            },
        };
        let input = rec.into_ingest_input();
        assert_eq!(input.content, "issue body");
        assert_eq!(input.source.as_deref(), Some("https://example/42"));
        let env = input.source_envelope.unwrap();
        assert!(env.has_key());
        assert_eq!(env.source_id.as_deref(), Some("42"));
    }
}

// ===================== Driver tests (mock connector, no network) ==================
// `run_sync` owns the mutation paths: cursor checkpointing, the overlap window,
// failure clamping, rate-limit retry, and reconcile routing. Until now only its
// pure helpers and the storage layer had tests; these pin the driver itself.

#[cfg(all(test, feature = "legacy-sqlite"))]
mod driver_tests {
    use super::*;
    use crate::storage::SqliteMemoryStore;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn ts(secs: i64) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-06-19T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
            + chrono::Duration::seconds(secs)
    }

    fn rec(id: &str, updated: DateTime<Utc>, body: &str) -> NormalizedRecord {
        NormalizedRecord {
            content: format!("issue {id}: {body}"),
            tags: vec!["mock".to_string()],
            envelope: SourceEnvelope {
                source_system: Some("mock".to_string()),
                source_id: Some(id.to_string()),
                source_url: Some(format!("https://example.test/{id}")),
                source_updated_at: Some(updated),
                content_hash: Some(format!("h-{id}-{body}")),
                synced_at: Some(Utc::now()),
                source_project: Some("mock/scope".to_string()),
                source_type: Some("issue".to_string()),
                source_author: Some("tester".to_string()),
                ..Default::default()
            },
        }
    }

    /// Scripted connector: each `fetch_updated` call consumes the next script
    /// entry (panics if exhausted — tests must script every call they expect).
    /// Every call's `(since, cursor)` arguments are recorded for assertions.
    type CallLog = Mutex<Vec<(Option<DateTime<Utc>>, Option<String>)>>;

    struct MockConnector {
        script: Mutex<Vec<ConnectorResult<FetchPage>>>,
        calls: CallLog,
        call_count: AtomicUsize,
        live_ids: Option<Vec<String>>,
    }

    impl MockConnector {
        fn new(script: Vec<ConnectorResult<FetchPage>>) -> Self {
            Self {
                script: Mutex::new(script),
                calls: Mutex::new(Vec::new()),
                call_count: AtomicUsize::new(0),
                live_ids: None,
            }
        }

        fn with_live_ids(mut self, ids: Vec<String>) -> Self {
            self.live_ids = Some(ids);
            self
        }

        fn ok(records: Vec<NormalizedRecord>, next: Option<&str>) -> ConnectorResult<FetchPage> {
            Ok(FetchPage {
                records,
                next_cursor: next.map(str::to_string),
                skipped: Vec::new(),
            })
        }

    }

    impl Connector for MockConnector {
        fn source_system(&self) -> &str {
            "mock"
        }

        fn scope(&self) -> &str {
            "mock/scope"
        }

        async fn fetch_updated(
            &self,
            since: Option<DateTime<Utc>>,
            cursor: Option<String>,
        ) -> ConnectorResult<FetchPage> {
            self.calls.lock().unwrap().push((since, cursor.clone()));
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut script = self.script.lock().unwrap();
            assert!(
                !script.is_empty(),
                "mock script exhausted by call {}",
                self.call_count.load(Ordering::SeqCst)
            );
            script.remove(0)
        }

        async fn list_live_ids(&self) -> ConnectorResult<Option<Vec<String>>> {
            Ok(self.live_ids.clone())
        }
    }

    // The store must outlive the tempdir guard, so tests keep both handles.
    fn store_kept() -> (tempfile::TempDir, SqliteMemoryStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = SqliteMemoryStore::new(Some(dir.path().join("t.db"))).unwrap();
        (dir, s)
    }

    #[tokio::test]
    async fn run_sync_pages_creates_and_then_reports_unchanged() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let t2 = ts(60);
        let t3 = ts(120);

        let conn = MockConnector::new(vec![
            MockConnector::ok(vec![rec("1", t1, "a"), rec("2", t2, "b")], Some("p2")),
            MockConnector::ok(vec![rec("2", t2, "b-EDITED"), rec("3", t3, "c")], None),
        ]);
        let report = run_sync(&store, &conn, false, 10).await.unwrap();
        assert_eq!(report.created, 3);
        assert_eq!(report.updated, 1, "issue 2 was edited between pages");
        assert_eq!(report.unchanged, 0);
        assert_eq!(report.new_cursor, Some(t3));

        // Second run: everything already known and unchanged.
        let conn2 = MockConnector::new(vec![MockConnector::ok(vec![
            rec("1", t1, "a"),
            rec("2", t2, "b-EDITED"),
            rec("3", t3, "c"),
        ], None)]);
        let report2 = run_sync(&store, &conn2, false, 10).await.unwrap();
        assert_eq!(report2.created, 0, "re-running must not duplicate");
        assert_eq!(report2.unchanged, 3);
    }

    #[tokio::test]
    async fn run_sync_applies_the_overlap_window_to_the_saved_cursor() {
        let (_dir, store) = store_kept();
        let t3 = ts(120);

        let conn = MockConnector::new(vec![MockConnector::ok(vec![rec("3", t3, "c")], None)]);
        run_sync(&store, &conn, false, 10).await.unwrap();

        let conn2 = MockConnector::new(vec![MockConnector::ok(vec![], None)]);
        run_sync(&store, &conn2, false, 10).await.unwrap();
        let calls = conn2.calls.lock().unwrap();
        assert_eq!(
            calls[0].0,
            Some(t3 - chrono::Duration::seconds(CURSOR_OVERLAP_SECS)),
            "next run resumes from cursor minus the overlap window"
        );
    }

    #[tokio::test]
    async fn run_sync_clamps_the_cursor_below_a_failed_upsert() {
        let (_dir, store) = store_kept();
        let t2 = ts(60);
        let t3 = ts(120);
        // Issue 2's content trips the secret policy → upsert fails. Issue 3
        // (later in the same window) still succeeds.
        let secret = format!("ghp_{}", "A".repeat(36));
        let bad = rec("2", t2, &secret);
        let good = rec("3", t3, "c");

        let conn = MockConnector::new(vec![MockConnector::ok(vec![bad, good], None)]);
        let report = run_sync(&store, &conn, false, 10).await.unwrap();
        assert_eq!(report.created, 1, "only the clean record persists");
        assert!(
            !report.warnings.is_empty(),
            "the failed upsert must surface as a warning"
        );

        let cursor = store
            .get_connector_cursor("mock", "mock/scope")
            .unwrap()
            .cursor_updated_at;
        assert_eq!(
            cursor,
            Some(t2 - chrono::Duration::seconds(1)),
            "cursor must be clamped so the failed record is re-fetched next run"
        );
    }

    #[tokio::test]
    async fn run_sync_persists_the_checkpoint_after_every_page_so_interruptions_resume() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let t2 = ts(300);

        // Page 1 succeeds; page 2 dies mid-run. The pre-per-page-checkpoint
        // behavior lost the whole run's progress; now page 1's checkpoint
        // survives the abort.
        let conn = MockConnector::new(vec![
            MockConnector::ok(vec![rec("1", t1, "a")], Some("p2")),
            Err(ConnectorError::Transport(
                "GET https://api.example.test/page2: connection reset".to_string(),
            )),
        ]);
        let err = run_sync(&store, &conn, false, 10).await.unwrap_err();
        assert!(err.to_string().contains("page2"), "error names the call");

        let cursor = store
            .get_connector_cursor("mock", "mock/scope")
            .unwrap()
            .cursor_updated_at;
        assert_eq!(
            cursor,
            Some(t1),
            "page 1's checkpoint must survive a page-2 failure"
        );
        // t2 (the failed page's high-water mark) was never persisted.
        assert_ne!(cursor, Some(t2));
    }

    #[tokio::test]
    async fn run_sync_sleeps_through_a_short_rate_limit_and_retries_once() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let conn = MockConnector::new(vec![
            Err(ConnectorError::RateLimited(Some(
                std::time::Duration::from_millis(1),
            ))),
            MockConnector::ok(vec![rec("1", t1, "a")], None),
        ]);
        let report = run_sync(&store, &conn, false, 10).await.unwrap();
        assert_eq!(report.created, 1);
        assert_eq!(conn.call_count.load(Ordering::SeqCst), 2, "one retry");
    }

    #[tokio::test]
    async fn run_sync_aborts_on_a_long_rate_limit_instead_of_blocking() {
        let (_dir, store) = store_kept();
        let conn = MockConnector::new(vec![Err(ConnectorError::RateLimited(Some(
            std::time::Duration::from_secs(3600),
        )))]);
        let err = run_sync(&store, &conn, false, 10).await.unwrap_err();
        assert!(
            matches!(err, ConnectorError::RateLimited(Some(d)) if d.as_secs() == 3600),
            "long backoffs abort with the wait surfaced: {err}"
        );
    }

    #[tokio::test]
    async fn run_sync_turns_skipped_records_into_warnings_and_clamps_the_cursor() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let t2 = ts(60);
        let t3 = ts(120);

        let page = FetchPage {
            records: vec![rec("1", t1, "a"), rec("3", t3, "c")],
            next_cursor: None,
            skipped: vec![SkippedRecord {
                source_updated_at: Some(t2),
                reason: "mock/scope#2 comments: 500".to_string(),
            }],
        };
        let conn = MockConnector::new(vec![Ok(page)]);
        let report = run_sync(&store, &conn, false, 10).await.unwrap();
        assert_eq!(report.created, 2, "unskipped records still persist");
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("skipped record") && w.contains("#2")),
            "skips surface as warnings: {:?}",
            report.warnings
        );
        let cursor = store
            .get_connector_cursor("mock", "mock/scope")
            .unwrap()
            .cursor_updated_at;
        assert_eq!(
            cursor,
            Some(t2 - chrono::Duration::seconds(1)),
            "cursor clamps below the skipped record so the next run retries it"
        );
    }

    #[tokio::test]
    async fn run_sync_reconcile_tombstones_only_records_missing_upstream() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let t2 = ts(60);
        let t3 = ts(120);

        // First sync: issues 1, 2, 3.
        let conn = MockConnector::new(vec![MockConnector::ok(vec![
            rec("1", t1, "a"),
            rec("2", t2, "b"),
            rec("3", t3, "c"),
        ], None)]);
        run_sync(&store, &conn, false, 10).await.unwrap();

        // Issue 2 vanished upstream. A reconcile that only sees {1, 3} must
        // tombstone exactly issue 2's memory — and nothing else.
        let conn2 = MockConnector::new(vec![MockConnector::ok(vec![rec("1", t1, "a")], None)])
            .with_live_ids(vec!["1".to_string(), "3".to_string()]);
        let report = run_sync(&store, &conn2, true, 10).await.unwrap();
        assert_eq!(report.tombstoned, 1);
        assert!(report.reconciled);

        let still_valid: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM knowledge_nodes WHERE source_system='mock' \
                     AND valid_until IS NULL",
                    [],
                    |r| r.get(0),
                )
                .unwrap()
        };
        assert_eq!(still_valid, 2, "1 and 3 stay valid; only 2 tombstoned");

        // Issue 2 reappears upstream → the next upsert un-tombstones it.
        let conn3 = MockConnector::new(vec![MockConnector::ok(vec![rec("2", t2, "b")], None)])
            .with_live_ids(vec!["1".into(), "2".into(), "3".into()]);
        let report3 = run_sync(&store, &conn3, false, 10).await.unwrap();
        assert_eq!(report3.unchanged, 1, "same hash → Unchanged path");
        let now_valid: i64 = {
            let reader = store.reader.lock().unwrap();
            reader
                .query_row(
                    "SELECT COUNT(*) FROM knowledge_nodes WHERE source_system='mock' \
                     AND valid_until IS NULL",
                    [],
                    |r| r.get(0),
                )
                .unwrap()
        };
        assert_eq!(now_valid, 3, "reappearing record is un-tombstoned");
    }

    #[tokio::test]
    async fn run_sync_reconcile_refuses_to_tombstone_everything_on_an_empty_live_set() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let conn = MockConnector::new(vec![MockConnector::ok(vec![rec("1", t1, "a")], None)]);
        run_sync(&store, &conn, false, 10).await.unwrap();

        // An empty live-id enumeration is a transient/auth failure signal, not
        // "the source is empty" — it must skip reconcile, not wipe the scope.
        let conn2 = MockConnector::new(vec![MockConnector::ok(vec![], None)])
            .with_live_ids(vec![]);
        let report = run_sync(&store, &conn2, true, 10).await.unwrap();
        assert_eq!(report.tombstoned, 0);
        assert!(!report.reconciled);
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("empty set")),
            "the guard must explain itself: {:?}",
            report.warnings
        );
    }

    #[tokio::test]
    async fn run_sync_respects_max_pages_and_leaves_a_resumable_cursor() {
        let (_dir, store) = store_kept();
        let t1 = ts(0);
        let t2 = ts(600);

        let conn = MockConnector::new(vec![
            MockConnector::ok(vec![rec("1", t1, "a")], Some("p2")),
            MockConnector::ok(vec![rec("2", t2, "b")], None),
        ]);
        let report = run_sync(&store, &conn, false, 1).await.unwrap();
        assert_eq!(report.created, 1, "only page 1 within max_pages=1");
        assert_eq!(
            store
                .get_connector_cursor("mock", "mock/scope")
                .unwrap()
                .cursor_updated_at,
            Some(t1),
            "the saved cursor lets the next call resume where this one stopped"
        );
        assert_eq!(conn.call_count.load(Ordering::SeqCst), 1);
    }
}
