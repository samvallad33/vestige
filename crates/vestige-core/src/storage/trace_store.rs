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
use rusqlite::{params, OptionalExtension};
use uuid::Uuid;

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};
use crate::trace::{MemoryPr, MemoryPrAction, MemoryPrStatus, MemoryTraceEvent, Receipt};

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

        let seq: i64 = writer
            .query_row(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_traces WHERE run_id = ?1",
                params![run_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

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
        let mut stmt = reader.prepare(
            "SELECT payload FROM agent_traces WHERE run_id = ?1 ORDER BY seq ASC",
        )?;
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
        Ok(out)
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
        Ok(payload.and_then(|p| serde_json::from_str(&p).ok()))
    }

    /// List recent receipts, newest first.
    pub fn list_receipts(&self, limit: usize) -> Result<Vec<Receipt>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(
            "SELECT payload FROM memory_receipts ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let p: String = row.get(0)?;
            Ok(p)
        })?;
        let mut out = Vec::new();
        for r in rows {
            if let Ok(rc) = serde_json::from_str::<Receipt>(&r?) {
                out.push(rc);
            }
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
        let mut out = Vec::new();
        for r in rows {
            if let Ok(rc) = serde_json::from_str::<Receipt>(&r?) {
                out.push(rc);
            }
        }
        Ok(out)
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
            let changed = writer.execute(
                "UPDATE memory_prs SET status = ?1, decision = ?2, decided_at = ?3 WHERE id = ?4",
                params![new_status.as_str(), decision, now, id],
            )?;
            if changed == 0 {
                return Err(StorageError::NotFound(id.to_string()));
            }
        }
        self.get_memory_pr(id)?
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
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
        let diff: serde_json::Value = serde_json::from_str(&diff_s).unwrap_or(serde_json::json!({}));
        let signals = serde_json::from_str(&signals_s).unwrap_or_default();
        let decision = decision_s
            .and_then(|s| serde_json::from_value(serde_json::Value::String(s)).ok());

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
    fn receipt_roundtrips() {
        let s = store();
        let receipt = Receipt {
            receipt_id: "r_2026_06_22_abc".into(),
            retrieved: vec!["m1".into(), "m2".into()],
            suppressed: vec![SuppressedReceiptEntry::new("m3", SuppressReason::LowTrust)],
            activation_path: vec!["a -> b".into()],
            trust_floor: 0.62,
            decay_risk: DecayRisk::Medium,
            mutations: vec![],
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
        // calls reverse_suppression on the subject.)
        assert!(crate::MemoryPrAction::Promote.releases_memory());
        let released = s
            .reverse_suppression(&node.id, 24)
            .expect("reverse suppression within labile window");
        assert_eq!(
            released.suppression_count, 0,
            "promoting the PR must release the memory — not leave it suppressed"
        );
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
        assert!(s
            .decide_memory_pr("pr_2", MemoryPrAction::AskAgentWhy)
            .is_err());
        // Still pending.
        assert_eq!(s.count_pending_memory_prs().unwrap(), 1);
    }
}
