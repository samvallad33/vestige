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
}

/// Errors a connector can surface.
#[derive(Debug, thiserror::Error)]
pub enum ConnectorError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("rate limited; retry after {0:?}")]
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
///    persisting it only after the run so a crash re-scans rather than skips;
/// 4. optionally reconciles deletions when `reconcile` is set and the connector
///    can enumerate live ids.
///
/// `max_pages` bounds a single run (so a first sync of a 15-year tracker can be
/// resumed across calls rather than blocking on one enormous fetch).
pub async fn run_sync<C: Connector>(
    store: &crate::storage::SqliteMemoryStore,
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

    // 2. Page forward, upserting each record.
    let mut cursor: Option<String> = None;
    let mut max_seen = checkpoint.cursor_updated_at;
    // Oldest source_updated_at among records that FAILED to upsert this run. We
    // must not advance the persisted cursor past this, or the failed record —
    // fetched in ascending update order — would fall outside the next run's
    // `since` window and never be retried (a silent permanent gap).
    let mut oldest_failure: Option<DateTime<Utc>> = None;
    // Count of genuinely new records (Created). Unchanged re-scans of the
    // overlap window must not inflate the running total.
    let mut created_this_run = 0i64;

    for _ in 0..max_pages.max(1) {
        let page = connector.fetch_updated(since, cursor.clone()).await?;
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
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }

    // Clamp the cursor so we never advance past a record that failed this run.
    // Subtract one second so the next run's inclusive `since` re-includes it.
    if let Some(failed_at) = oldest_failure {
        let clamp_to = failed_at - chrono::Duration::seconds(1);
        max_seen = Some(match max_seen {
            Some(m) if m < clamp_to => m,
            _ => clamp_to,
        });
    }

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

    // 4. Persist the checkpoint (only after the run).
    let now = Utc::now();
    let new_checkpoint = ConnectorCursor {
        source_system: source_system.clone(),
        scope: scope.clone(),
        cursor_updated_at: max_seen,
        last_synced_at: Some(now),
        last_full_reconcile_at: if reconciled {
            Some(now)
        } else {
            checkpoint.last_full_reconcile_at
        },
        // Accumulate only NEW records, so re-scanning the overlap window (which
        // reports Unchanged) does not inflate the running total.
        records_seen: checkpoint.records_seen + created_this_run,
    };
    store
        .save_connector_cursor(&new_checkpoint)
        .map_err(|e| ConnectorError::Transport(e.to_string()))?;

    Ok(report)
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

#[cfg(test)]
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
