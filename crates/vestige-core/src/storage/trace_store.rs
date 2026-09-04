//! # Black Box / Receipts / Memory PRs — persistence
//!
//! CRUD for the three V18 tables (`agent_traces` + `agent_runs`,
//! `memory_receipts`, `memory_prs`) on [`SqliteMemoryStore`]. The pure data
//! model lives in [`crate::trace`]; this file is the storage half of the
//! Black Box, immune system, and cinematic debugger for agent memory.
//!
//! Every method follows the established store idiom: lock the writer/reader
//! `Mutex<Connection>`, `params![]`-bind, store timestamps as RFC3339 (and
//! event millis as INTEGER), serialize structured fields with `serde_json`, and
//! map rows back through a small closure.

use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use uuid::Uuid;

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};
use crate::trace::{MemoryPr, MemoryPrAction, MemoryPrStatus, MemoryTraceEvent, Receipt};

/// Side effect applied while atomically deciding a pre-execution mutation PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingMemoryMutationEffect {
    /// The reviewer kept the memory unchanged.
    Kept,
    /// The reviewer approved the pending purge/delete.
    Purged,
    /// The reviewer held the memory under active suppression.
    Suppressed,
}

/// Result of deciding a PR created before a destructive mutation.
#[derive(Debug, Clone)]
pub struct PendingMemoryMutationDecision {
    /// Final PR state returned even when an approved purge removed its row.
    pub pr: MemoryPr,
    /// Mutation side effect committed with the decision.
    pub effect: PendingMemoryMutationEffect,
}

/// Trace retention window used when `VESTIGE_TRACE_RETENTION_DAYS` is unset or
/// unusable.
const DEFAULT_TRACE_RETENTION_DAYS: i64 = 30;

/// Upper bound on the configurable trace retention window (100 years). Beyond
/// this the window is meaningless — `0` is the documented "keep forever" — and
/// unbounded values overflow the `chrono::Duration` the sweep builds from them,
/// which panics (and the release profile aborts on panic).
const MAX_TRACE_RETENTION_DAYS: i64 = 36_500;

fn is_receipt_local_slot(id: &str) -> bool {
    [
        "candidate_",
        "pair_",
        "evidence_",
        "trigger_",
        "redacted_",
        "purged_",
    ]
    .iter()
    .any(|prefix| id.starts_with(prefix))
}

/// Parse the `VESTIGE_TRACE_RETENTION_DAYS` value into a usable retention
/// window. Unset, empty, negative, and malformed values fall back to
/// [`DEFAULT_TRACE_RETENTION_DAYS`]; values above
/// [`MAX_TRACE_RETENTION_DAYS`] are clamped. Split out from
/// [`SqliteMemoryStore::prune_agent_traces`] so the parsing rules are testable
/// without touching process-global environment state.
fn resolve_trace_retention_days(raw: Option<&str>) -> i64 {
    raw.and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|d| *d >= 0)
        .map(|d| d.min(MAX_TRACE_RETENTION_DAYS))
        .unwrap_or(DEFAULT_TRACE_RETENTION_DAYS)
}

/// A roll-up summary of one agent run, for the Black Box run list.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AgentRunSummary {
    /// The run id.
    pub run_id: String,
    /// The first tool invoked in the run (the run's "entry point").
    pub first_tool: Option<String>,
    /// Total events recorded.
    pub event_count: i64,
    /// Memories retrieved across the run.
    pub retrieved_count: i64,
    /// Memories suppressed across the run.
    pub suppressed_count: i64,
    /// Memory writes across the run.
    pub write_count: i64,
    /// Sanhedrin vetoes across the run.
    pub veto_count: i64,
    /// Millis of the first event.
    pub started_at: i64,
    /// Millis of the most recent event.
    pub last_at: i64,
}

impl SqliteMemoryStore {
    // ========================================================================
    // BLACK BOX — trace events + run roll-up
    // ========================================================================

    /// Append one trace event to a run (append-only) and update the run
    /// roll-up. Returns the assigned sequence number within the run.
    ///
    /// `seq` is `MAX(seq)+1` for the run, computed under the writer lock so a
    /// run's events stay totally ordered even under concurrent tool calls.
    pub fn append_trace_event(&self, event: &MemoryTraceEvent) -> Result<i64> {
        let now = Utc::now();
        let run_id = event.run_id().to_string();
        let event_type = event.kind();
        let at = event.at();
        let payload = serde_json::to_string(event)
            .map_err(|e| StorageError::Init(format!("trace event serialize: {e}")))?;
        let tool = match event {
            MemoryTraceEvent::McpCall { tool, .. } => Some(tool.clone()),
            _ => None,
        };

        // Roll-up deltas this event contributes.
        let (d_retrieved, d_suppressed, d_write, d_veto) = match event {
            MemoryTraceEvent::MemoryRetrieve { ids, .. } => (ids.len() as i64, 0, 0, 0),
            MemoryTraceEvent::MemorySuppress { .. } => (0, 1, 0, 0),
            MemoryTraceEvent::MemoryWrite { .. } => (0, 0, 1, 0),
            MemoryTraceEvent::SanhedrinVeto { .. } => (0, 0, 0, 1),
            _ => (0, 0, 0, 0),
        };

        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

        // Propagate a seq-query failure instead of defaulting to 0: swallowing
        // the error with unwrap_or(0) could write a duplicate seq=0 for a run
        // that already has events, corrupting Black Box replay ordering. On an
        // empty run COALESCE(...,-1)+1 already yields 0 correctly.
        let seq: i64 = writer.query_row(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_traces WHERE run_id = ?1",
            params![run_id],
            |r| r.get(0),
        )?;

        writer.execute(
            "INSERT INTO agent_traces (id, run_id, seq, event_type, tool, payload, at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                Uuid::new_v4().to_string(),
                run_id,
                seq,
                event_type,
                tool,
                payload,
                at,
                now.to_rfc3339(),
            ],
        )?;

        // Upsert the run roll-up. On first event the row is created with the
        // event's tool as the entry point; subsequent events accumulate counts
        // and advance `last_at`.
        writer.execute(
            "INSERT INTO agent_runs (run_id, first_tool, event_count, retrieved_count,
                 suppressed_count, write_count, veto_count, started_at, last_at, created_at)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, ?7, ?8)
             ON CONFLICT(run_id) DO UPDATE SET
                 first_tool = COALESCE(agent_runs.first_tool, excluded.first_tool),
                 event_count = agent_runs.event_count + 1,
                 retrieved_count = agent_runs.retrieved_count + ?3,
                 suppressed_count = agent_runs.suppressed_count + ?4,
                 write_count = agent_runs.write_count + ?5,
                 veto_count = agent_runs.veto_count + ?6,
                 last_at = MAX(agent_runs.last_at, ?7)",
            params![
                run_id,
                tool,
                d_retrieved,
                d_suppressed,
                d_write,
                d_veto,
                at,
                now.to_rfc3339(),
            ],
        )?;

        Ok(seq)
    }

    /// Fetch every event of a run, in sequence order. The black-box replay.
    pub fn get_trace(&self, run_id: &str) -> Result<Vec<MemoryTraceEvent>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader
            .prepare("SELECT payload FROM agent_traces WHERE run_id = ?1 ORDER BY seq ASC")?;
        let rows = stmt.query_map(params![run_id], |row| {
            let payload: String = row.get(0)?;
            Ok(payload)
        })?;
        let mut out = Vec::new();
        for r in rows {
            let payload = r?;
            if let Ok(ev) = serde_json::from_str::<MemoryTraceEvent>(&payload) {
                out.push(ev);
            }
        }
        drop(stmt);
        Self::redact_trace_for_current_state(&reader, &mut out)?;
        Ok(out)
    }

    /// Resolve every trace-carried memory id against current public validity.
    /// This makes suppression immediately effective even for an older run.
    fn redact_trace_for_current_state(
        conn: &rusqlite::Connection,
        events: &mut [MemoryTraceEvent],
    ) -> Result<()> {
        let mut ids = Vec::<String>::new();
        for event in events.iter() {
            for id in event.referenced_memory_ids() {
                if !is_receipt_local_slot(id) && !ids.iter().any(|existing| existing == id) {
                    ids.push(id.to_string());
                }
            }
        }

        let now_ms = Utc::now().timestamp_millis();
        for (index, id) in ids.into_iter().enumerate() {
            let publicly_eligible: Option<i64> = conn
                .query_row(
                    "SELECT CASE
                        WHEN suppression_count = 0
                         AND superseded_by IS NULL
                         AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?2)
                         AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?2)
                        THEN 1 ELSE 0 END
                     FROM knowledge_nodes WHERE id = ?1",
                    params![id, now_ms],
                    |row| row.get(0),
                )
                .optional()?;
            if publicly_eligible != Some(1) {
                let replacement = format!("redacted_{}", index + 1);
                for event in events.iter_mut() {
                    event.redact_memory_id(&id, &replacement);
                }
            }
        }
        Ok(())
    }

    /// List recent runs, newest activity first.
    pub fn list_agent_runs(&self, limit: usize) -> Result<Vec<AgentRunSummary>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT run_id, first_tool, event_count, retrieved_count, suppressed_count,
                    write_count, veto_count, started_at, last_at
             FROM agent_runs ORDER BY last_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_run_summary)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch one run summary.
    pub fn get_agent_run(&self, run_id: &str) -> Result<Option<AgentRunSummary>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        reader
            .query_row(
                "SELECT run_id, first_tool, event_count, retrieved_count, suppressed_count,
                        write_count, veto_count, started_at, last_at
                 FROM agent_runs WHERE run_id = ?1",
                params![run_id],
                Self::row_to_run_summary,
            )
            .optional()
            .map_err(StorageError::from)
    }

    fn row_to_run_summary(row: &rusqlite::Row) -> rusqlite::Result<AgentRunSummary> {
        Ok(AgentRunSummary {
            run_id: row.get("run_id")?,
            first_tool: row.get("first_tool").ok().flatten(),
            event_count: row.get("event_count")?,
            retrieved_count: row.get("retrieved_count")?,
            suppressed_count: row.get("suppressed_count")?,
            write_count: row.get("write_count")?,
            veto_count: row.get("veto_count")?,
            started_at: row.get("started_at")?,
            last_at: row.get("last_at")?,
        })
    }

    /// Prune old Black Box trace events (bounded retention).
    ///
    /// The trace recorder appends rows on every MCP tool call, so without a
    /// sweep `agent_traces` grows without bound. Called from the periodic
    /// consolidation cycle (mirroring `prune_access_log`): deletes trace
    /// events older than the retention window, then drops any `agent_runs`
    /// roll-up left with no events (orphaned).
    ///
    /// Retention defaults to 30 days. `VESTIGE_TRACE_RETENTION_DAYS` overrides
    /// it; `0` keeps traces forever (sweep disabled); unset, empty, negative,
    /// or malformed values fall back to the default, and absurdly large values
    /// are clamped to [`MAX_TRACE_RETENTION_DAYS`]. Returns the number of trace
    /// events deleted.
    pub fn prune_agent_traces(&self) -> Result<i64> {
        let days = resolve_trace_retention_days(
            std::env::var("VESTIGE_TRACE_RETENTION_DAYS")
                .ok()
                .as_deref(),
        );
        self.prune_agent_traces_older_than_days(days)
    }

    /// Env-independent core of [`Self::prune_agent_traces`], so tests can
    /// exercise retention deterministically. `days == 0` means keep forever.
    pub(crate) fn prune_agent_traces_older_than_days(&self, days: i64) -> Result<i64> {
        if days == 0 {
            return Ok(0);
        }
        // Belt and braces against an out-of-range window: `Duration::days`
        // panics on overflow and the release profile is panic = "abort", so an
        // unclamped value would hard-kill the process from inside the periodic
        // consolidation sweep. Skip the sweep instead — a cutoff that far back
        // could not have deleted anything anyway.
        let Some(cutoff) = chrono::Duration::try_days(days)
            .and_then(|window| Utc::now().checked_sub_signed(window))
        else {
            tracing::warn!("Trace retention of {days} days is out of range; skipping trace sweep");
            return Ok(0);
        };
        let cutoff_ms = cutoff.timestamp_millis();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let deleted =
            writer.execute("DELETE FROM agent_traces WHERE at < ?1", params![cutoff_ms])? as i64;
        if deleted > 0 {
            // Drop run roll-ups whose every event was just swept, so the Black
            // Box run list never shows runs that can no longer be replayed.
            writer.execute(
                "DELETE FROM agent_runs
                 WHERE run_id NOT IN (SELECT DISTINCT run_id FROM agent_traces)",
                [],
            )?;
        }
        Ok(deleted)
    }

    // ========================================================================
    // MEMORY RECEIPTS
    // ========================================================================

    /// Persist a retrieval receipt. `run_id`/`tool`/`query` are denormalized
    /// context for the dashboard; the full [`Receipt`] is stored as JSON.
    pub fn save_receipt(
        &self,
        receipt: &Receipt,
        run_id: Option<&str>,
        tool: Option<&str>,
        query: Option<&str>,
    ) -> Result<()> {
        let payload = serde_json::to_string(receipt)
            .map_err(|e| StorageError::Init(format!("receipt serialize: {e}")))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO memory_receipts
                 (receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
                  trust_floor, decay_risk, payload, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                receipt.receipt_id,
                run_id,
                tool,
                query,
                receipt.retrieved.len() as i64,
                receipt.suppressed.len() as i64,
                receipt.trust_floor,
                receipt.decay_risk.as_str(),
                payload,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Associate a receipt committed inside a tool transaction with the Black
    /// Box run that invoked that tool. The receipt payload stays immutable; the
    /// denormalized run column is filled once after dispatch returns.
    pub fn link_receipt_to_run(&self, receipt_id: &str, run_id: &str) -> Result<bool> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let changed = writer.execute(
            "UPDATE memory_receipts SET run_id = ?1
             WHERE receipt_id = ?2 AND run_id IS NULL",
            params![run_id, receipt_id],
        )?;
        Ok(changed == 1)
    }

    /// Fetch one receipt by id.
    pub fn get_receipt(&self, receipt_id: &str) -> Result<Option<Receipt>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let payload: Option<String> = reader
            .query_row(
                "SELECT payload FROM memory_receipts WHERE receipt_id = ?1",
                params![receipt_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(payload) = payload else {
            return Ok(None);
        };
        let mut receipt: Receipt = serde_json::from_str(&payload)
            .map_err(|e| StorageError::Init(format!("receipt deserialize: {e}")))?;
        Self::redact_receipt_for_current_state(&reader, &mut receipt)?;
        Ok(Some(receipt))
    }

    /// List recent receipts, newest first.
    pub fn list_receipts(&self, limit: usize) -> Result<Vec<Receipt>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader
            .prepare("SELECT payload FROM memory_receipts ORDER BY created_at DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let p: String = row.get(0)?;
            Ok(p)
        })?;
        let mut payloads = Vec::new();
        for r in rows {
            payloads.push(r?);
        }
        drop(stmt);
        let mut out = Vec::new();
        for payload in payloads {
            let mut receipt: Receipt = serde_json::from_str(&payload)
                .map_err(|e| StorageError::Init(format!("receipt deserialize: {e}")))?;
            Self::redact_receipt_for_current_state(&reader, &mut receipt)?;
            out.push(receipt);
        }
        Ok(out)
    }

    /// List the receipts belonging to one run, newest first (B5). The Black Box
    /// receipts panel uses this so the receipts it shows actually belong to the
    /// selected run, not the global latest.
    pub fn list_receipts_for_run(&self, run_id: &str, limit: usize) -> Result<Vec<Receipt>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT payload FROM memory_receipts WHERE run_id = ?1
             ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![run_id, limit as i64], |row| {
            let p: String = row.get(0)?;
            Ok(p)
        })?;
        let mut payloads = Vec::new();
        for r in rows {
            payloads.push(r?);
        }
        drop(stmt);
        let mut out = Vec::new();
        for payload in payloads {
            let mut receipt: Receipt = serde_json::from_str(&payload)
                .map_err(|e| StorageError::Init(format!("receipt deserialize: {e}")))?;
            Self::redact_receipt_for_current_state(&reader, &mut receipt)?;
            out.push(receipt);
        }
        Ok(out)
    }

    /// Resolve stable evidence ids against current suppression/validity state
    /// before a receipt crosses a public API boundary. Stored history remains
    /// auditable, but a later suppress/purge cannot resurrect a correlatable id.
    fn redact_receipt_for_current_state(
        conn: &rusqlite::Connection,
        receipt: &mut Receipt,
    ) -> Result<()> {
        let mut ids = Vec::<String>::new();
        let mut push_id = |id: &str| {
            if !is_receipt_local_slot(id) && !ids.iter().any(|existing| existing == id) {
                ids.push(id.to_string());
            }
        };
        for id in &receipt.retrieved {
            push_id(id);
        }
        for entry in &receipt.suppressed {
            push_id(&entry.id);
        }
        for mutation in &receipt.mutations {
            push_id(&mutation.id);
        }
        if let Some(crate::trace::ReceiptEvidence::SynapticCapture(evidence)) = &receipt.evidence {
            push_id(&evidence.trigger.memory_id);
            for candidate in &evidence.candidates {
                if let Some(id) = &candidate.memory_id {
                    push_id(id);
                }
            }
        }

        let now_ms = Utc::now().timestamp_millis();
        for (index, id) in ids.into_iter().enumerate() {
            let publicly_eligible: Option<i64> = conn
                .query_row(
                    "SELECT CASE
                        WHEN suppression_count = 0
                         AND superseded_by IS NULL
                         AND (valid_from IS NULL OR unixepoch(valid_from) * 1000 <= ?2)
                         AND (valid_until IS NULL OR unixepoch(valid_until) * 1000 > ?2)
                        THEN 1 ELSE 0 END
                     FROM knowledge_nodes WHERE id = ?1",
                    params![id, now_ms],
                    |row| row.get(0),
                )
                .optional()?;
            if publicly_eligible != Some(1) {
                receipt.redact_memory_id(&id, &format!("redacted_{}", index + 1));
            }
        }
        Ok(())
    }

    // ========================================================================
    // MEMORY PRs — the risk-gated review queue
    // ========================================================================

    /// Open (insert) a Memory PR.
    pub fn save_memory_pr(&self, pr: &MemoryPr) -> Result<()> {
        let diff = serde_json::to_string(&pr.diff).unwrap_or_else(|_| "{}".to_string());
        let signals = serde_json::to_string(&pr.signals).unwrap_or_else(|_| "[]".to_string());
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO memory_prs
                 (id, kind, status, title, subject_id, run_id, diff, signals,
                  decision, created_at, decided_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                pr.id,
                pr.kind.as_str(),
                pr.status.as_str(),
                pr.title,
                pr.subject_id,
                pr.run_id,
                diff,
                signals,
                pr.decision
                    .and_then(|d| serde_json::to_value(d).ok())
                    .and_then(|v| v.as_str().map(|s| s.to_string())),
                pr.created_at,
                pr.decided_at,
            ],
        )?;
        Ok(())
    }

    /// Fetch one Memory PR by id.
    pub fn get_memory_pr(&self, id: &str) -> Result<Option<MemoryPr>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        reader
            .query_row(
                "SELECT id, kind, status, title, subject_id, run_id, diff, signals,
                        decision, created_at, decided_at
                 FROM memory_prs WHERE id = ?1",
                params![id],
                Self::row_to_memory_pr,
            )
            .optional()
            .map_err(StorageError::from)
    }

    /// List Memory PRs, optionally filtered by status, newest first.
    pub fn list_memory_prs(
        &self,
        status: Option<MemoryPrStatus>,
        limit: usize,
    ) -> Result<Vec<MemoryPr>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let (sql, with_filter) = match status {
            Some(_) => (
                "SELECT id, kind, status, title, subject_id, run_id, diff, signals,
                        decision, created_at, decided_at
                 FROM memory_prs WHERE status = ?1 ORDER BY created_at DESC LIMIT ?2",
                true,
            ),
            None => (
                "SELECT id, kind, status, title, subject_id, run_id, diff, signals,
                        decision, created_at, decided_at
                 FROM memory_prs ORDER BY created_at DESC LIMIT ?1",
                false,
            ),
        };
        let mut stmt = reader.prepare(sql)?;
        let mut out = Vec::new();
        if with_filter {
            let st = status.unwrap();
            let rows =
                stmt.query_map(params![st.as_str(), limit as i64], Self::row_to_memory_pr)?;
            for r in rows {
                out.push(r?);
            }
        } else {
            let rows = stmt.query_map(params![limit as i64], Self::row_to_memory_pr)?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// Count pending Memory PRs (for the nav badge).
    pub fn count_pending_memory_prs(&self) -> Result<i64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let n: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM memory_prs WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(n)
    }

    /// Record a decision on a Memory PR, moving it out of `pending`. Returns the
    /// updated PR. `AskAgentWhy` is read-only and never reaches here.
    pub fn decide_memory_pr(&self, id: &str, action: MemoryPrAction) -> Result<MemoryPr> {
        let new_status = action.resulting_status().ok_or_else(|| {
            StorageError::Init("ask_agent_why is read-only and decides nothing".into())
        })?;
        let decision = serde_json::to_value(action)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();
        let now = Utc::now().to_rfc3339();
        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // Only a still-pending PR may be decided. The `AND status = 'pending'`
            // guard makes decisions final: re-POSTing an action on an already
            // promoted/forgotten/merged PR cannot flip its status, re-run its
            // side effects (e.g. release_quarantine resurrecting a rejected
            // memory), or overwrite the audit ledger (decision/decided_at).
            let changed = writer.execute(
                "UPDATE memory_prs SET status = ?1, decision = ?2, decided_at = ?3
                 WHERE id = ?4 AND status = 'pending'",
                params![new_status.as_str(), decision, now, id],
            )?;
            if changed == 0 {
                // Distinguish "no such PR" from "already decided" so callers get
                // a truthful error instead of a misleading NotFound.
                return Err(match self.get_memory_pr(id)? {
                    Some(_) => StorageError::Init(format!(
                        "memory PR {id} is already decided and cannot be re-decided"
                    )),
                    None => StorageError::NotFound(id.to_string()),
                });
            }
        }
        self.get_memory_pr(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    /// Atomically decide and, when approved, apply a mutation that was held
    /// before execution. Returns `None` for ordinary post-commit Memory PRs.
    ///
    /// `Forget` approves the requested purge/suppression, `Promote` (and the
    /// other accept actions) keeps the current memory unchanged, and
    /// `Quarantine` keeps the row but applies suppression. The PR transition
    /// and mutation share one SQLite transaction, so neither can commit alone.
    pub fn decide_pending_memory_mutation(
        &self,
        id: &str,
        action: MemoryPrAction,
    ) -> Result<Option<PendingMemoryMutationDecision>> {
        let pr = self
            .get_memory_pr(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))?;
        let Some(pending_action) = pr
            .diff
            .get("pendingAction")
            .and_then(serde_json::Value::as_str)
        else {
            return Ok(None);
        };
        let subject_id = pr.subject_id.clone().ok_or_else(|| {
            StorageError::Init(format!("pending mutation PR {id} has no subject"))
        })?;
        let new_status = action.resulting_status().ok_or_else(|| {
            StorageError::Init("ask_agent_why is read-only and decides nothing".into())
        })?;
        let decision = serde_json::to_value(action)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let now = Utc::now();

        let effect = {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            // Writer transactions begin IMMEDIATE through the shared helper so
            // the 5 s busy timeout applies to the write lock and BUSY/LOCKED
            // past it is retried with a logged reason, same as every other
            // writer in the storage layer.
            let tx = Self::begin_write_transaction(&writer, "decide_pending_memory_mutation")?;

            let changed = tx.execute(
                "UPDATE memory_prs SET status = ?1, decision = ?2, decided_at = ?3
                 WHERE id = ?4 AND status = 'pending'",
                params![new_status.as_str(), decision, now.to_rfc3339(), id],
            )?;
            if changed == 0 {
                return Err(StorageError::Init(format!(
                    "memory PR {id} is already decided and cannot be re-decided"
                )));
            }

            let effect = match action {
                MemoryPrAction::Forget if matches!(pending_action, "purge" | "delete") => {
                    let deleted =
                        Self::purge_node_in_transaction(&tx, &subject_id, now, true)?.is_some();
                    if !deleted {
                        return Err(StorageError::NotFound(subject_id.to_string()));
                    }
                    PendingMemoryMutationEffect::Purged
                }
                MemoryPrAction::Forget | MemoryPrAction::Quarantine => {
                    let changed = tx.execute(
                        "UPDATE knowledge_nodes SET
                            last_accessed = ?1,
                            suppression_count = COALESCE(suppression_count, 0) + 1,
                            suppressed_at = ?1,
                            retrieval_strength = MAX(0.05, retrieval_strength - 0.35),
                            retention_strength = MAX(0.05, retention_strength - 0.20),
                            stability = stability * 0.4
                         WHERE id = ?2",
                        params![now.to_rfc3339(), &subject_id],
                    )?;
                    if changed == 0 {
                        return Err(StorageError::NotFound(subject_id.to_string()));
                    }
                    Self::invalidate_replay_evidence_for_memory_in_transaction(
                        &tx,
                        &subject_id,
                        crate::storage::ReplayInvalidationReason::Suppressed,
                    )?;
                    PendingMemoryMutationEffect::Suppressed
                }
                _ => PendingMemoryMutationEffect::Kept,
            };

            tx.commit()?;
            effect
        };

        let mut pr = pr;
        pr.status = new_status;
        pr.decision = Some(action);
        pr.decided_at = Some(now.to_rfc3339());
        if effect == PendingMemoryMutationEffect::Purged {
            self.remove_purged_node_from_vector_index(&subject_id);
        } else if effect == PendingMemoryMutationEffect::Suppressed {
            let _ = self.log_access(&subject_id, "suppress");
        }
        Ok(Some(PendingMemoryMutationDecision { pr, effect }))
    }

    fn row_to_memory_pr(row: &rusqlite::Row) -> rusqlite::Result<MemoryPr> {
        let kind_s: String = row.get("kind")?;
        let status_s: String = row.get("status")?;
        let diff_s: String = row.get("diff")?;
        let signals_s: String = row.get("signals")?;
        let decision_s: Option<String> = row.get("decision").ok().flatten();

        let kind = crate::trace::MemoryPrKind::from_label(&kind_s)
            .unwrap_or(crate::trace::MemoryPrKind::NewFact);
        let status = serde_json::from_value(serde_json::Value::String(status_s))
            .unwrap_or(MemoryPrStatus::Pending);
        let diff: serde_json::Value =
            serde_json::from_str(&diff_s).unwrap_or(serde_json::json!({}));
        let signals = serde_json::from_str(&signals_s).unwrap_or_default();
        let decision =
            decision_s.and_then(|s| serde_json::from_value(serde_json::Value::String(s)).ok());

        Ok(MemoryPr {
            id: row.get("id")?,
            kind,
            status,
            title: row.get("title")?,
            diff,
            signals,
            subject_id: row.get("subject_id").ok().flatten(),
            run_id: row.get("run_id").ok().flatten(),
            created_at: row.get("created_at")?,
            decided_at: row.get("decided_at").ok().flatten(),
            decision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;
    use crate::trace::{
        DecayRisk, MemoryPrKind, MemoryTraceEvent, Receipt, RiskSignal, SuppressReason,
        SuppressedReceiptEntry,
    };

    fn store() -> SqliteMemoryStore {
        // Temp-file store for isolated, fast tests (mirrors the existing
        // sqlite.rs test helpers; there is no in-memory constructor).
        let dir = tempfile::tempdir().unwrap();
        SqliteMemoryStore::new(Some(dir.path().join("trace_test.db"))).expect("test store")
    }

    #[test]
    fn receipt_local_slots_are_never_resolved_as_memory_ids() {
        for slot in [
            "candidate_1",
            "pair_fedcba98",
            "evidence_2",
            "trigger_1",
            "redacted_3",
            "purged_1",
        ] {
            assert!(is_receipt_local_slot(slot), "{slot} must stay opaque");
        }
        assert!(!is_receipt_local_slot(
            "550e8400-e29b-41d4-a716-446655440000"
        ));
    }

    #[test]
    fn trace_append_orders_and_rolls_up() {
        let s = store();
        let run = "run_abc";
        s.append_trace_event(&MemoryTraceEvent::McpCall {
            run_id: run.into(),
            tool: "deep_reference".into(),
            args_hash: "h".into(),
            at: 100,
        })
        .unwrap();
        let mut activation = std::collections::BTreeMap::new();
        activation.insert("m1".to_string(), 0.9);
        s.append_trace_event(&MemoryTraceEvent::MemoryRetrieve {
            run_id: run.into(),
            ids: vec!["m1".into(), "m2".into()],
            activation,
            at: 110,
        })
        .unwrap();
        s.append_trace_event(&MemoryTraceEvent::MemorySuppress {
            run_id: run.into(),
            id: "m3".into(),
            reason: SuppressReason::Contradicted,
            at: 120,
        })
        .unwrap();

        let events = s.get_trace(run).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].kind(), "mcp.call");
        assert_eq!(events[2].kind(), "memory.suppress");

        let summary = s.get_agent_run(run).unwrap().unwrap();
        assert_eq!(summary.first_tool.as_deref(), Some("deep_reference"));
        assert_eq!(summary.event_count, 3);
        assert_eq!(summary.retrieved_count, 2);
        assert_eq!(summary.suppressed_count, 1);
        assert_eq!(summary.started_at, 100);
        assert_eq!(summary.last_at, 120);

        let runs = s.list_agent_runs(10).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, run);
    }

    #[test]
    fn trace_reads_redact_suppressed_ids_and_purge_scrubs_raw_payloads() {
        let s = store();
        let node = s
            .ingest(IngestInput {
                content: "trace identity must follow current privacy state".into(),
                ..Default::default()
            })
            .unwrap();
        let mut activation = std::collections::BTreeMap::new();
        activation.insert(node.id.clone(), 0.9);
        s.append_trace_event(&MemoryTraceEvent::MemoryRetrieve {
            run_id: "run_privacy".into(),
            ids: vec![node.id.clone()],
            activation,
            at: 100,
        })
        .unwrap();
        s.append_trace_event(&MemoryTraceEvent::MemoryWrite {
            run_id: "run_privacy".into(),
            id: node.id.clone(),
            diff: serde_json::json!({ "memoryId": node.id }),
            source: crate::trace::WriteSource::Agent,
            at: 110,
        })
        .unwrap();

        let visible = serde_json::to_string(&s.get_trace("run_privacy").unwrap()).unwrap();
        assert!(visible.contains(&node.id));

        s.suppress_memory(&node.id).unwrap();
        let suppressed = serde_json::to_string(&s.get_trace("run_privacy").unwrap()).unwrap();
        assert!(!suppressed.contains(&node.id));
        assert!(suppressed.contains("redacted_1"));

        s.purge_node(&node.id, Some("trace privacy test")).unwrap();
        let raw_payloads: Vec<String> = {
            let reader = s.reader.lock().unwrap();
            let mut stmt = reader
                .prepare("SELECT payload FROM agent_traces WHERE run_id = 'run_privacy'")
                .unwrap();
            stmt.query_map([], |row| row.get(0))
                .unwrap()
                .map(|row| row.unwrap())
                .collect()
        };
        assert!(
            raw_payloads
                .iter()
                .all(|payload| !payload.contains(&node.id))
        );
        assert!(
            raw_payloads
                .iter()
                .all(|payload| payload.contains("purged_1"))
        );
    }

    #[test]
    fn prune_agent_traces_sweeps_old_events_and_orphaned_runs() {
        let s = store();
        let now_ms = Utc::now().timestamp_millis();
        let old_ms = now_ms - 40 * 24 * 60 * 60 * 1000; // 40 days ago

        s.append_trace_event(&MemoryTraceEvent::McpCall {
            run_id: "run_old".into(),
            tool: "search".into(),
            args_hash: "h".into(),
            at: old_ms,
        })
        .unwrap();
        s.append_trace_event(&MemoryTraceEvent::McpCall {
            run_id: "run_new".into(),
            tool: "search".into(),
            args_hash: "h".into(),
            at: now_ms,
        })
        .unwrap();

        // 0 = keep forever: the sweep is disabled entirely.
        assert_eq!(s.prune_agent_traces_older_than_days(0).unwrap(), 0);
        assert_eq!(s.list_agent_runs(10).unwrap().len(), 2);

        // 30-day sweep: the old event goes, and its now-orphaned run roll-up
        // goes with it; the fresh run is untouched.
        let deleted = s.prune_agent_traces_older_than_days(30).unwrap();
        assert_eq!(deleted, 1, "exactly the 40-day-old event is deleted");
        assert!(s.get_trace("run_old").unwrap().is_empty());
        assert!(
            s.get_agent_run("run_old").unwrap().is_none(),
            "orphaned run roll-up must be swept with its events"
        );
        assert_eq!(s.get_trace("run_new").unwrap().len(), 1);
        assert!(s.get_agent_run("run_new").unwrap().is_some());
    }

    /// Retention parsing must survive hostile / fat-fingered env values. The
    /// upper bound is the load-bearing one: an unclamped huge value overflows
    /// the `chrono::Duration` the sweep builds, and `panic = "abort"` in the
    /// release profile turns that into a hard process kill.
    #[test]
    fn trace_retention_env_value_is_clamped_and_never_overflows() {
        // Sane values pass through untouched.
        assert_eq!(resolve_trace_retention_days(Some("7")), 7);
        assert_eq!(resolve_trace_retention_days(Some(" 90 ")), 90);
        // 0 stays 0: the documented "keep forever" switch.
        assert_eq!(resolve_trace_retention_days(Some("0")), 0);
        // Unset / empty / malformed / negative fall back to the default.
        for raw in [None, Some(""), Some("   "), Some("forever"), Some("-5")] {
            assert_eq!(
                resolve_trace_retention_days(raw),
                DEFAULT_TRACE_RETENTION_DAYS,
                "unusable value {raw:?} must fall back to the default"
            );
        }
        // Absurdly large values are clamped instead of overflowing.
        for raw in ["999999999999", "9223372036854775807"] {
            assert_eq!(
                resolve_trace_retention_days(Some(raw)),
                MAX_TRACE_RETENTION_DAYS,
                "{raw} must be clamped, not passed through"
            );
        }
    }

    /// End-to-end: a hostile `VESTIGE_TRACE_RETENTION_DAYS` must not panic the
    /// process, and must not sweep anything (the clamped window is 100 years,
    /// far older than any trace). Also exercises the raw sweep at `i64::MAX` to
    /// prove the second line of defence inside
    /// `prune_agent_traces_older_than_days` holds even if a caller bypasses the
    /// env clamp.
    #[test]
    fn prune_agent_traces_survives_hostile_retention_value() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        const KEY: &str = "VESTIGE_TRACE_RETENTION_DAYS";

        let s = store();
        s.append_trace_event(&MemoryTraceEvent::McpCall {
            run_id: "run_hostile".into(),
            tool: "search".into(),
            args_hash: "h".into(),
            at: Utc::now().timestamp_millis(),
        })
        .unwrap();

        // Direct call: no clamp in play, so this is the overflow path itself.
        assert_eq!(
            s.prune_agent_traces_older_than_days(i64::MAX).unwrap(),
            0,
            "an out-of-range window must skip the sweep, not panic"
        );

        // Env path: process env is global and unsafe to mutate under Rust 2024,
        // so serialize on a local lock and restore the previous value.
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var_os(KEY);
        unsafe { std::env::set_var(KEY, "999999999999") };
        let deleted = s.prune_agent_traces();
        unsafe {
            match previous {
                Some(prev) => std::env::set_var(KEY, prev),
                None => std::env::remove_var(KEY),
            }
        }

        assert_eq!(
            deleted.unwrap(),
            0,
            "a 100-year clamped window sweeps nothing"
        );
        assert_eq!(
            s.get_trace("run_hostile").unwrap().len(),
            1,
            "the fresh trace must survive"
        );
    }

    #[test]
    fn receipt_roundtrips() {
        let s = store();
        let m1 = s
            .ingest(IngestInput {
                content: "receipt evidence one".into(),
                ..Default::default()
            })
            .unwrap();
        let m2 = s
            .ingest(IngestInput {
                content: "receipt evidence two".into(),
                ..Default::default()
            })
            .unwrap();
        let m3 = s
            .ingest(IngestInput {
                content: "receipt evidence three".into(),
                ..Default::default()
            })
            .unwrap();
        let receipt = Receipt {
            receipt_id: "r_2026_06_22_abc".into(),
            retrieved: vec![m1.id, m2.id],
            suppressed: vec![SuppressedReceiptEntry::new(m3.id, SuppressReason::LowTrust)],
            activation_path: vec!["a -> b".into()],
            trust_floor: 0.62,
            decay_risk: DecayRisk::Medium,
            mutations: vec![],
            evidence: None,
        };
        s.save_receipt(&receipt, Some("run_abc"), Some("search"), Some("q"))
            .unwrap();
        let got = s.get_receipt("r_2026_06_22_abc").unwrap().unwrap();
        assert_eq!(got, receipt);
        assert_eq!(s.list_receipts(10).unwrap().len(), 1);
    }

    #[test]
    fn receipts_are_listable_per_run_b5() {
        let s = store();
        let mk = |id: &str| Receipt {
            receipt_id: id.into(),
            retrieved: vec!["m1".into()],
            suppressed: vec![],
            activation_path: vec![],
            trust_floor: 0.9,
            decay_risk: DecayRisk::Low,
            mutations: vec![],
            evidence: None,
        };
        s.save_receipt(&mk("r_a1"), Some("run_a"), Some("search"), None)
            .unwrap();
        s.save_receipt(&mk("r_a2"), Some("run_a"), Some("search"), None)
            .unwrap();
        s.save_receipt(&mk("r_b1"), Some("run_b"), Some("search"), None)
            .unwrap();

        let run_a = s.list_receipts_for_run("run_a", 10).unwrap();
        assert_eq!(run_a.len(), 2, "run_a has exactly its 2 receipts");
        assert!(run_a.iter().all(|r| r.receipt_id.starts_with("r_a")));

        let run_b = s.list_receipts_for_run("run_b", 10).unwrap();
        assert_eq!(run_b.len(), 1, "run_b has only its own receipt");
        assert_eq!(run_b[0].receipt_id, "r_b1");

        // Global list still sees all three.
        assert_eq!(s.list_receipts(10).unwrap().len(), 3);
    }

    #[test]
    fn memory_pr_lifecycle() {
        let s = store();
        let pr = MemoryPr {
            id: "pr_1".into(),
            kind: MemoryPrKind::ContradictionDetected,
            status: MemoryPrStatus::Pending,
            title: "Agent wants to overwrite a high-trust fact".into(),
            diff: serde_json::json!({"before": "x", "after": "y"}),
            signals: vec![RiskSignal {
                code: "contradicts_high_trust".into(),
                detail: "Contradicts trust 0.9.".into(),
            }],
            subject_id: Some("m_old".into()),
            run_id: Some("run_abc".into()),
            created_at: Utc::now().to_rfc3339(),
            decided_at: None,
            decision: None,
        };
        s.save_memory_pr(&pr).unwrap();

        assert_eq!(s.count_pending_memory_prs().unwrap(), 1);
        let pending = s
            .list_memory_prs(Some(MemoryPrStatus::Pending), 10)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].signals[0].code, "contradicts_high_trust");

        let decided = s.decide_memory_pr("pr_1", MemoryPrAction::Promote).unwrap();
        assert_eq!(decided.status, MemoryPrStatus::Promoted);
        assert_eq!(decided.decision, Some(MemoryPrAction::Promote));
        assert!(decided.decided_at.is_some());
        assert_eq!(s.count_pending_memory_prs().unwrap(), 0);
    }

    #[test]
    fn promote_releases_a_quarantined_memory_end_to_end() {
        // B1 regression: the full quarantine→release cycle at the storage layer.
        // gate_writes suppresses a risky write; an accept action must reverse it.
        let s = store();
        let node = s
            .ingest(crate::IngestInput {
                content: "Risky write that got quarantined.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .expect("ingest");
        assert_eq!(node.suppression_count, 0, "fresh node not suppressed");

        // Quarantine it (what gate_writes does for a risky write).
        let suppressed = s.suppress_memory(&node.id).expect("suppress");
        assert_eq!(
            suppressed.suppression_count, 1,
            "quarantined write is suppressed (held out of retrieval)"
        );

        // Promote = release. (The action releases_memory() == true; the handler
        // calls release_quarantine on the subject.)
        assert!(crate::MemoryPrAction::Promote.releases_memory());
        let released = s.release_quarantine(&node.id).expect("release quarantine");
        assert_eq!(
            released.suppression_count, 0,
            "promoting the PR must release the memory — not leave it suppressed"
        );
        assert!(
            released.suppressed_at.is_none(),
            "release must clear suppressed_at"
        );
    }

    #[test]
    fn release_quarantine_works_past_the_labile_window_c1() {
        // C1: a PR reviewed LATE (past the 24h active-forgetting labile window)
        // must still release the memory. reverse_suppression refuses after the
        // window; release_quarantine must not.
        let s = store();
        let node = s
            .ingest(crate::IngestInput {
                content: "Risky write quarantined and reviewed days later.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .expect("ingest");
        s.suppress_memory(&node.id).expect("suppress");

        // Backdate suppressed_at to 100h ago — well past any labile window.
        s.set_suppressed_at_for_test(&node.id, chrono::Utc::now() - chrono::Duration::hours(100));

        // reverse_suppression refuses (window expired)...
        assert!(
            s.reverse_suppression(&node.id, 24).is_err(),
            "reverse_suppression must refuse past the labile window"
        );
        // ...but release_quarantine still releases it.
        let released = s.release_quarantine(&node.id).expect("release past window");
        assert_eq!(released.suppression_count, 0);
        assert!(released.suppressed_at.is_none());
    }

    #[test]
    fn release_quarantine_is_idempotent_on_unsuppressed() {
        let s = store();
        let node = s
            .ingest(crate::IngestInput {
                content: "Never suppressed.".to_string(),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .expect("ingest");
        // No-op when not suppressed — must not error.
        let same = s.release_quarantine(&node.id).expect("idempotent release");
        assert_eq!(same.suppression_count, 0);
    }

    #[test]
    fn ask_agent_why_is_not_a_decision() {
        let s = store();
        let pr = MemoryPr {
            id: "pr_2".into(),
            kind: MemoryPrKind::NewFact,
            status: MemoryPrStatus::Pending,
            title: "t".into(),
            diff: serde_json::json!({}),
            signals: vec![],
            subject_id: None,
            run_id: None,
            created_at: Utc::now().to_rfc3339(),
            decided_at: None,
            decision: None,
        };
        s.save_memory_pr(&pr).unwrap();
        assert!(
            s.decide_memory_pr("pr_2", MemoryPrAction::AskAgentWhy)
                .is_err()
        );
        // Still pending.
        assert_eq!(s.count_pending_memory_prs().unwrap(), 1);
    }
}
