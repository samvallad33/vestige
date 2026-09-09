//! Atomic compare-and-swap of reminder occurrences selected by a check.
use super::sqlite::{IntentionRecord, SqliteMemoryStore};
use rusqlite::{TransactionBehavior, params};

impl SqliteMemoryStore {
    /// Commit a complete check batch or none of it. A concurrent check or user
    /// edit invalidates the old snapshot and asks the caller to retry. This
    /// claims local occurrences; it cannot guarantee external delivery.
    pub fn commit_intention_check(
        &self,
        changes: &[(IntentionRecord, IntentionRecord)],
    ) -> Result<(), String> {
        if changes.is_empty() {
            return Ok(());
        }
        let mut writer = self.writer.lock().map_err(|e| e.to_string())?;
        let tx = writer
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| e.to_string())?;
        for (old, new) in changes {
            if old.id != new.id {
                return Err("intention check cannot change record identity".into());
            }
            let changed = tx
                .execute(
                    "UPDATE intentions SET trigger_type=?1,trigger_data=?2,status=?3,
                    reminder_count=?4,last_reminded_at=?5,snoozed_until=?6
                 WHERE id=?7 AND trigger_type=?8 AND trigger_data=?9 AND status=?10
                    AND reminder_count=?11 AND last_reminded_at IS ?12
                    AND snoozed_until IS ?13 AND content=?14 AND priority=?15
                    AND deadline IS ?16 AND fulfilled_at IS ?17",
                    params![
                        new.trigger_type,
                        new.trigger_data,
                        new.status,
                        new.reminder_count,
                        new.last_reminded_at.map(|t| t.to_rfc3339()),
                        new.snoozed_until.map(|t| t.to_rfc3339()),
                        old.id,
                        old.trigger_type,
                        old.trigger_data,
                        old.status,
                        old.reminder_count,
                        old.last_reminded_at.map(|t| t.to_rfc3339()),
                        old.snoozed_until.map(|t| t.to_rfc3339()),
                        old.content,
                        old.priority,
                        old.deadline.map(|t| t.to_rfc3339()),
                        old.fulfilled_at.map(|t| t.to_rfc3339())
                    ],
                )
                .map_err(|e| format!("Intention check commit failed: {e}"))?;
            if changed != 1 {
                return Err(format!(
                    "Intention '{}' changed during check; retry the check",
                    old.id
                ));
            }
        }
        tx.commit().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str) -> IntentionRecord {
        IntentionRecord {
            id: id.into(),
            content: "Synthetic reminder".into(),
            trigger_type: "time".into(),
            trigger_data: "{}".into(),
            priority: 2,
            status: "active".into(),
            created_at: "2026-10-01T09:00:00Z".parse().unwrap(),
            deadline: None,
            fulfilled_at: None,
            reminder_count: 0,
            last_reminded_at: None,
            notes: None,
            tags: vec![],
            related_memories: vec![],
            snoozed_until: None,
            source_type: "user".into(),
            source_data: None,
        }
    }

    fn delivered(old: &IntentionRecord) -> IntentionRecord {
        let mut new = old.clone();
        new.reminder_count += 1;
        new.last_reminded_at = Some(old.created_at);
        new
    }

    #[test]
    fn intention_claim_has_one_winner_across_connections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let a = SqliteMemoryStore::new(Some(path.clone())).unwrap();
        let b = SqliteMemoryStore::new(Some(path)).unwrap();
        let old = record("a");
        a.save_intention(&old).unwrap();
        let changes = vec![(old.clone(), delivered(&old))];
        let outcomes = std::thread::scope(|s| {
            let first = s.spawn(|| a.commit_intention_check(&changes));
            let second = s.spawn(|| b.commit_intention_check(&changes));
            [first.join().unwrap(), second.join().unwrap()]
        });
        assert_eq!(outcomes.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(a.get_intention("a").unwrap().unwrap().reminder_count, 1);
    }

    #[test]
    fn intention_claim_batch_conflict_rolls_back_prior_claims() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("test.db"))).unwrap();
        let a = record("a");
        let b = record("b");
        store.save_intention(&a).unwrap();
        store.save_intention(&b).unwrap();
        let mut cancelled = b.clone();
        cancelled.status = "cancelled".into();
        store.save_intention(&cancelled).unwrap();
        assert!(
            store
                .commit_intention_check(&[(a.clone(), delivered(&a)), (b.clone(), delivered(&b))])
                .is_err()
        );
        assert_eq!(store.get_intention("a").unwrap().unwrap().reminder_count, 0);
        assert_eq!(
            store.get_intention("b").unwrap().unwrap().status,
            "cancelled"
        );
    }
}
