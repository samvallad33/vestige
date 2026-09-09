//! Atomic local persistence and reproducible replay for evidence-aware intentions.
//!
//! The journal records caller assertions, not externally verified truth. It never
//! invokes providers or performs the action described by an intention.

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::sqlite::SqliteMemoryStore;
use crate::intention_graph::{Command, IntentionGraph};

type MemoryCommitmentRow = (String, Option<String>, Option<String>, Option<String>, i64);

const MAX_GRAPH_BYTES: usize = 16 * 1024 * 1024;
const MAX_COMMAND_BYTES: usize = 128 * 1024;
const MAX_JOURNAL_ENTRIES: i64 = 20_000;

fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn validate_scope(scope: &str) -> Result<(), String> {
    if scope.is_empty()
        || scope.len() > 128
        || !scope
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-/".contains(&c))
    {
        return Err(
            "scope must contain 1..128 ASCII letters, digits, '.', '_', '-', or '/'".into(),
        );
    }
    Ok(())
}

fn storage_error(error: impl std::fmt::Display) -> String {
    format!("Intention graph storage: {error}")
}

impl SqliteMemoryStore {
    /// Read a content commitment for an available memory in the same scope.
    /// Raw memory text is never copied into intention history. A commitment
    /// proves a local snapshot changed, not that the memory's claim is true.
    pub fn intention_memory_snapshot(
        &self,
        scope: &str,
        memory_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_scope(scope)?;
        uuid::Uuid::parse_str(memory_id).map_err(|_| "memory_id must be a UUID")?;
        let reader = self.reader.lock().map_err(storage_error)?;
        let row: Option<MemoryCommitmentRow> = reader.query_row(
            "SELECT content,valid_from,valid_until,superseded_by,suppression_count FROM knowledge_nodes
             WHERE id=?1 AND COALESCE(NULLIF(trim(scope),''),'user')=?2",
            params![memory_id, scope],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)),
        ).optional().map_err(storage_error)?;
        let mut value = None;
        if let Some((content, from, until, superseded, suppressed)) = row {
            let parse = |raw: &str| -> Result<DateTime<Utc>, String> {
                DateTime::parse_from_rfc3339(raw)
                    .map(|t| t.with_timezone(&Utc))
                    .or_else(|_| {
                        chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f")
                            .map(|t| t.and_utc())
                    })
                    .map_err(|_| "memory validity timestamp is malformed".into())
            };
            let from = from.as_deref().map(parse).transpose()?;
            let until = until.as_deref().map(parse).transpose()?;
            if superseded.is_none()
                && suppressed == 0
                && from.is_none_or(|at| at <= now)
                && until.is_none_or(|at| at > now)
            {
                value = Some(digest(&content));
            }
        }
        Ok(
            json!({"source_key":format!("memory:{memory_id}"),"memory_id":memory_id,
            "value":value,"observed_at":now,"boundary":"Local scoped content commitment; not external fact verification."}),
        )
    }

    /// Apply a command to one scope, atomically committing snapshot and journal.
    /// IMMEDIATE transactions serialize writers across processes using this DB.
    /// Scope names are namespaces, not an authentication or tenancy boundary.
    pub fn apply_intention_graph(
        &self,
        scope: &str,
        command: Command,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_scope(scope)?;
        let command_json = serde_json::to_string(&command).map_err(storage_error)?;
        if command_json.len() > MAX_COMMAND_BYTES {
            return Err("intention graph command exceeds 128 KiB".into());
        }
        let mut writer = self.writer.lock().map_err(storage_error)?;
        let tx = writer
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        let prior: Option<String> = tx
            .query_row(
                "SELECT state_json FROM intention_graph_state WHERE scope = ?1",
                params![scope],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage_error)?;
        let mut graph: IntentionGraph = match &prior {
            Some(state) => serde_json::from_str(state).map_err(storage_error)?,
            None => IntentionGraph::default(),
        };
        let before = serde_json::to_string(&graph).map_err(storage_error)?;
        let output = graph.apply(command, now)?;
        let after = serde_json::to_string(&graph).map_err(storage_error)?;
        if after.len() > MAX_GRAPH_BYTES {
            return Err("intention graph exceeds the 16 MiB local scope limit".into());
        }
        let mut committed_seq = None;
        if before != after {
            let seq: i64 = tx.query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM intention_graph_journal WHERE scope = ?1",
                params![scope], |row| row.get(0),
            ).map_err(storage_error)?;
            if seq > MAX_JOURNAL_ENTRIES {
                return Err("intention graph journal limit reached; preserve this scope and create a new scope with reviewed plan definitions".into());
            }
            let at = now.to_rfc3339();
            let output_json = serde_json::to_string(&output).map_err(storage_error)?;
            tx.execute(
                "INSERT INTO intention_graph_journal(scope,seq,command_json,evaluated_at,output_digest,state_digest)
                 VALUES (?1,?2,?3,?4,?5,?6)",
                params![scope, seq, command_json, at, digest(&output_json), digest(&after)],
            ).map_err(storage_error)?;
            tx.execute(
                "INSERT INTO intention_graph_state(scope,state_json,updated_at) VALUES (?1,?2,?3)
                 ON CONFLICT(scope) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
                params![scope, after, at],
            ).map_err(storage_error)?;
            committed_seq = Some(seq);
        }
        tx.commit().map_err(storage_error)?;
        Ok(
            json!({"scope": scope, "journal_seq": committed_seq, "result": output,
            "boundary": "Local deterministic evaluation of supplied evidence; no external action or independent fact verification."}),
        )
    }

    /// Replay the exact committed command history against an empty evaluator.
    /// Digests detect divergence, not adversarial rewriting of the whole DB.
    pub fn replay_intention_graph(&self, scope: &str) -> Result<Value, String> {
        validate_scope(scope)?;
        let mut writer = self.writer.lock().map_err(storage_error)?;
        let tx = writer.transaction().map_err(storage_error)?;
        let mut graph = IntentionGraph::default();
        let mut count = 0_i64;
        {
            let mut stmt = tx
                .prepare(
                    "SELECT seq,command_json,evaluated_at,output_digest,state_digest
                 FROM intention_graph_journal WHERE scope=?1 ORDER BY seq",
                )
                .map_err(storage_error)?;
            let mut rows = stmt.query(params![scope]).map_err(storage_error)?;
            while let Some(row) = rows.next().map_err(storage_error)? {
                let seq: i64 = row.get(0).map_err(storage_error)?;
                if seq != count + 1 || seq > MAX_JOURNAL_ENTRIES {
                    return Err("intention graph journal sequence is invalid".into());
                }
                let command: String = row.get(1).map_err(storage_error)?;
                let at: String = row.get(2).map_err(storage_error)?;
                let expected_output: String = row.get(3).map_err(storage_error)?;
                let expected_state: String = row.get(4).map_err(storage_error)?;
                let command: Command = serde_json::from_str(&command).map_err(storage_error)?;
                let now = DateTime::parse_from_rfc3339(&at)
                    .map_err(storage_error)?
                    .with_timezone(&Utc);
                let output = graph.apply(command, now)?;
                let actual_output = digest(&serde_json::to_string(&output).map_err(storage_error)?);
                let actual_state = digest(&serde_json::to_string(&graph).map_err(storage_error)?);
                if actual_output != expected_output || actual_state != expected_state {
                    return Err(format!("intention graph replay diverged at sequence {seq}"));
                }
                count = seq;
            }
        }
        let stored: Option<String> = tx
            .query_row(
                "SELECT state_json FROM intention_graph_state WHERE scope=?1",
                params![scope],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage_error)?;
        let replayed = serde_json::to_string(&graph).map_err(storage_error)?;
        if stored.as_deref().is_some_and(|state| state != replayed)
            || (stored.is_none() && count != 0)
        {
            return Err("intention graph replay differs from the current snapshot".into());
        }
        tx.commit().map_err(storage_error)?;
        Ok(
            json!({"scope":scope,"matched":true,"commands":count,"state_digest":digest(&replayed),
            "boundary":"Deterministic local replay, not proof of external facts, delivery, or cryptographic authenticity."}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> DateTime<Utc> {
        "2026-10-01T09:00:00Z".parse().unwrap()
    }
    fn plan(id: &str) -> Command {
        serde_json::from_value(
            json!({"action":"plan","id":id,"description":"Synthetic projector intention",
            "requirements":[],"conflict_keys":[]}),
        )
        .unwrap()
    }

    #[test]
    fn graph_journal_failure_rolls_back_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        store
            .writer
            .lock()
            .unwrap()
            .execute_batch(
                "CREATE TRIGGER reject_intention_journal BEFORE INSERT ON intention_graph_journal
             BEGIN SELECT RAISE(ABORT, 'injected journal failure'); END;",
            )
            .unwrap();
        assert!(
            store
                .apply_intention_graph("user", plan("p1"), at())
                .is_err()
        );
        let count: i64 = store
            .reader
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM intention_graph_state", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
        store
            .writer
            .lock()
            .unwrap()
            .execute_batch("DROP TRIGGER reject_intention_journal")
            .unwrap();
        assert!(
            store
                .apply_intention_graph("user", plan("p1"), at())
                .is_ok()
        );
        assert_eq!(
            store.replay_intention_graph("user").unwrap()["matched"],
            true
        );
    }

    #[test]
    fn graph_two_connections_preserve_both_commands() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let a = SqliteMemoryStore::new(Some(path.clone())).unwrap();
        let b = SqliteMemoryStore::new(Some(path)).unwrap();
        std::thread::scope(|s| {
            let first = s.spawn(|| a.apply_intention_graph("user", plan("a"), at()).unwrap());
            let second = s.spawn(|| b.apply_intention_graph("user", plan("b"), at()).unwrap());
            first.join().unwrap();
            second.join().unwrap();
        });
        assert_eq!(a.replay_intention_graph("user").unwrap()["commands"], 2);
        let output = a
            .apply_intention_graph(
                "user",
                serde_json::from_value(json!({"action":"portfolio"})).unwrap(),
                at(),
            )
            .unwrap();
        assert!(output.to_string().contains("\"a\""));
        assert!(output.to_string().contains("\"b\""));
        assert_eq!(a.replay_intention_graph("other").unwrap()["commands"], 0);
    }

    #[test]
    fn graph_replay_detects_recorded_output_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        store
            .apply_intention_graph("user", plan("p1"), at())
            .unwrap();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE intention_graph_journal SET output_digest='changed'",
                [],
            )
            .unwrap();
        assert!(
            store
                .replay_intention_graph("user")
                .unwrap_err()
                .contains("diverged")
        );
    }

    #[test]
    fn graph_rejects_bad_scope_before_storage() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        assert!(
            store
                .apply_intention_graph("user\nother", plan("p1"), at())
                .is_err()
        );
        assert!(store.replay_intention_graph("").is_err());
    }

    #[test]
    fn graph_corrupt_persisted_plan_returns_error_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        store
            .apply_intention_graph("user", plan("p1"), at())
            .unwrap();
        let writer = store.writer.lock().unwrap();
        let raw: String = writer
            .query_row(
                "SELECT state_json FROM intention_graph_state WHERE scope='user'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let mut state: Value = serde_json::from_str(&raw).unwrap();
        state["plans"]["p1"]["versions"] = json!([]);
        writer
            .execute(
                "UPDATE intention_graph_state SET state_json=?1 WHERE scope='user'",
                params![state.to_string()],
            )
            .unwrap();
        drop(writer);
        let command = serde_json::from_value(json!({"action":"explain","id":"p1"})).unwrap();
        let error = store
            .apply_intention_graph("user", command, at())
            .unwrap_err();
        assert!(error.contains("versions"), "{error}");
        assert!(store.replay_intention_graph("user").is_err());
    }

    #[test]
    fn memory_commitments_change_without_copying_content_or_crossing_scope() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        let node = store
            .ingest_in_scope(
                crate::IngestInput {
                    content: "Synthetic projector requirement alpha".into(),
                    ..Default::default()
                },
                "project-a",
            )
            .unwrap();
        let first = store
            .intention_memory_snapshot("project-a", &node.id, at())
            .unwrap();
        assert!(first["value"].is_string());
        assert!(!first.to_string().contains("Synthetic projector"));
        assert!(
            store
                .intention_memory_snapshot("project-b", &node.id, at())
                .unwrap()["value"]
                .is_null()
        );
        store.writer.lock().unwrap().execute("UPDATE knowledge_nodes SET content='Synthetic projector requirement beta' WHERE id=?1",params![node.id]).unwrap();
        let changed = store
            .intention_memory_snapshot("project-a", &node.id, at())
            .unwrap();
        assert_ne!(first["value"], changed["value"]);
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE knowledge_nodes SET valid_until='2026-09-01T00:00:00Z' WHERE id=?1",
                params![node.id],
            )
            .unwrap();
        assert!(
            store
                .intention_memory_snapshot("project-a", &node.id, at())
                .unwrap()["value"]
                .is_null()
        );
    }
}
