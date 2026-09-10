//! Bounded, transactional maintenance pages. Cursors describe live scans.
use super::{Result, SqliteMemoryStore, StorageError};
use chrono::{DateTime, Duration, Utc};
use rusqlite::params;
use serde_json::{Value, json};
use std::time::Instant;

fn bounds(limit: usize, after: Option<&str>, budget_ms: u64) -> Result<()> {
    if !(1..=1000).contains(&limit) || !(1..=10_000).contains(&budget_ms) {
        return Err(StorageError::Init(
            "batchSize must be 1..1000 and budgetMs 1..10000".into(),
        ));
    }
    if after.is_some_and(|id| uuid::Uuid::parse_str(id).is_err()) {
        return Err(StorageError::Init("after must be a memory UUID".into()));
    }
    Ok(())
}

impl SqliteMemoryStore {
    /// Stable ID page for bounded dream input. Selection is namespace-specific.
    pub fn maintenance_memory_page(
        &self,
        limit: usize,
        after: Option<&str>,
        scope: &str,
    ) -> Result<(Vec<crate::KnowledgeNode>, bool)> {
        bounds(limit, after, 10_000)?;
        let ids = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader.prepare("SELECT id FROM knowledge_nodes WHERE id>?1 AND COALESCE(NULLIF(trim(scope),''),'user')=?2
                AND COALESCE(suppression_count,0)=0 ORDER BY id LIMIT ?3")?
                .query_map(params![after.unwrap_or(""),scope,(limit+1) as i64],|r|r.get::<_,String>(0))?
                .collect::<std::result::Result<Vec<_>,_>>()?
        };
        let more = ids.len() > limit;
        let mut nodes = Vec::new();
        for id in ids.iter().take(limit) {
            if let Some(node) = self.get_node(id)? {
                nodes.push(node);
            }
        }
        Ok((nodes, more))
    }

    /// Clear only waking tags actually covered by the completed dream snapshot.
    pub fn clear_dream_page_tags(
        &self,
        ids: &[String],
        started_at: DateTime<Utc>,
    ) -> Result<usize> {
        if ids.len() > 500 {
            return Err(StorageError::Init("dream page exceeds 500 memories".into()));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "clear_dream_page_tags")?;
        let mut cleared = 0;
        for id in ids {
            cleared += tx.execute(
                "UPDATE knowledge_nodes SET waking_tag=FALSE,waking_tag_at=NULL
            WHERE id=?1 AND waking_tag=TRUE AND waking_tag_at<=?2",
                params![id, started_at.to_rfc3339()],
            )?;
        }
        tx.commit()?;
        Ok(cleared)
    }

    /// Decay, promotion and ACT-R activation share one current-state transaction.
    pub fn maintain_lifecycle_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
    ) -> Result<Value> {
        bounds(limit, after, budget_ms)?;
        let started = Instant::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "maintain_lifecycle_batch")?;
        let ids = tx
            .prepare("SELECT id FROM knowledge_nodes WHERE id > ?1 ORDER BY id LIMIT ?2")?
            .query_map(params![after.unwrap_or(""), (limit + 1) as i64], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut processed = 0;
        let mut changed = 0;
        let mut cursor = after.map(str::to_string);
        let now = Utc::now();
        let w20: f64 = tx
            .query_row(
                "SELECT value FROM fsrs_config WHERE key = 'w20'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(crate::fsrs::DEFAULT_DECAY);
        let sleep = crate::SleepConsolidation::new();
        for id in ids.iter().take(limit) {
            if started.elapsed().as_millis() >= u128::from(budget_ms) {
                break;
            }
            let (last, storage, sentiment, stability, suppressed, protected): (String, f64, f64, f64, i64, i64) = tx.query_row(
                "SELECT last_accessed, storage_strength, sentiment_magnitude, stability,
                 COALESCE(suppression_count,0), COALESCE(protected,0) FROM knowledge_nodes WHERE id = ?1",
                params![id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)))?;
            if !dry_run && suppressed == 0 && protected == 0 {
                let last = DateTime::parse_from_rfc3339(&last)
                    .map_err(|e| StorageError::Init(e.to_string()))?;
                let days = (now.signed_duration_since(last)).num_seconds().max(0) as f64 / 86400.0;
                let stability = stability.min(crate::fsrs::MAX_STABILITY);
                let retrieval = crate::fsrs::retrievability_with_decay(
                    stability * (1.0 + sentiment * 0.5),
                    days,
                    w20,
                );
                let promoted = if sleep.should_promote(sentiment, storage) {
                    sleep.promotion_boost(storage)
                } else {
                    storage
                };
                let times = tx.prepare("SELECT accessed_at FROM memory_access_log WHERE node_id = ?1
                    AND access_type NOT IN ('search_hit','retrieval_shown') ORDER BY accessed_at DESC LIMIT 500")?
                    .query_map(params![id], |row| row.get::<_, String>(0))?.collect::<std::result::Result<Vec<_>, _>>()?;
                let mut sum = 0.0_f64;
                for time in times {
                    let time = DateTime::parse_from_rfc3339(&time)
                        .map_err(|e| StorageError::Init(e.to_string()))?;
                    let days = (now.signed_duration_since(time)).num_seconds() as f64 / 86400.0;
                    sum += days.max(0.001).powf(-0.5);
                }
                tx.execute(
                    "UPDATE knowledge_nodes SET stability=?1, storage_strength=?2,
                    retrieval_strength=?3, retention_strength=?4, activation=?5 WHERE id=?6",
                    params![
                        stability,
                        promoted,
                        retrieval,
                        sleep.calculate_retention(promoted, retrieval),
                        if sum > 0.0 { sum.ln() } else { 0.0 },
                        id
                    ],
                )?;
                changed += 1;
            }
            processed += 1;
            cursor = Some(id.clone());
        }
        tx.commit()?;
        Ok(
            json!({"phase":"lifecycle", "dryRun":dry_run, "selected":ids.len().min(limit),
            "processed":processed, "changed":changed, "hasMore":ids.len()>processed,
            "nextCursor":cursor, "durationMs":started.elapsed().as_millis(),
            "budgetMs":budget_ms, "deadlineKind":"cooperative; an in-flight SQLite operation may finish after the budget",
            "checkpoint":"transaction committed; repeat an uncompleted cursor after failure"}),
        )
    }

    /// Garbage collection reevaluates current eligibility under its write transaction.
    pub fn maintain_gc_batch(
        &self,
        limit: usize,
        after: Option<&str>,
        budget_ms: u64,
        dry_run: bool,
        min_retention: f64,
        max_age_days: Option<u64>,
    ) -> Result<Value> {
        bounds(limit, after, budget_ms)?;
        if !min_retention.is_finite()
            || !(0.0..=1.0).contains(&min_retention)
            || max_age_days.is_some_and(|days| days == 0 || days > 365_000)
        {
            return Err(StorageError::Init(
                "invalid retention threshold or max_age_days".into(),
            ));
        }
        let started = Instant::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "maintain_gc_batch")?;
        let ids = tx
            .prepare("SELECT id FROM knowledge_nodes WHERE id > ?1 ORDER BY id LIMIT ?2")?
            .query_map(params![after.unwrap_or(""), (limit + 1) as i64], |r| {
                r.get::<_, String>(0)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut processed = 0;
        let mut candidates = 0;
        let mut deleted = 0;
        let mut cursor = after.map(str::to_string);
        let now = Utc::now();
        for id in ids.iter().take(limit) {
            if started.elapsed().as_millis() >= u128::from(budget_ms) {
                break;
            }
            let (retention, created, protected):(f64,String,i64)=tx.query_row(
                "SELECT retention_strength,created_at,COALESCE(protected,0) FROM knowledge_nodes WHERE id=?1",
                params![id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
            let created = DateTime::parse_from_rfc3339(&created)
                .map_err(|e| StorageError::Init(e.to_string()))?;
            let eligible = protected == 0
                && retention < min_retention
                && max_age_days
                    .is_none_or(|d| now.signed_duration_since(created) >= Duration::days(d as i64));
            if eligible {
                candidates += 1;
                if !dry_run && Self::purge_node_in_transaction(&tx, id, now, true)?.is_some() {
                    deleted += 1;
                }
            }
            processed += 1;
            cursor = Some(id.clone());
        }
        tx.commit()?;
        Ok(
            json!({"tool":"gc","dryRun":dry_run,"candidateCount":candidates,"deleted":deleted,
            "errors":0,"processed":processed,"hasMore":ids.len()>processed,"nextCursor":cursor,
            "minRetention":min_retention,"maxAgeDays":max_age_days,"durationMs":started.elapsed().as_millis(),
            "budgetMs":budget_ms,"atomic":true,"checkpoint":"committed page; restart cursor after a completed sweep"}),
        )
    }

    /// Trim bounded access-history rows; callers repeat until hasMore is false.
    pub fn maintain_log_batch(&self, limit: usize, dry_run: bool) -> Result<Value> {
        bounds(limit, None, 10_000)?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "maintain_log_batch")?;
        let cutoff = (Utc::now() - Duration::days(super::ACCESS_LOG_RETENTION_DAYS)).to_rfc3339();
        let ids=tx.prepare("SELECT rowid FROM memory_access_log WHERE accessed_at < ?1 ORDER BY rowid LIMIT ?2")?
            .query_map(params![cutoff,(limit+1) as i64],|r|r.get::<_,i64>(0))?.collect::<std::result::Result<Vec<_>,_>>()?;
        if !dry_run {
            for id in ids.iter().take(limit) {
                tx.execute("DELETE FROM memory_access_log WHERE rowid=?1", params![id])?;
            }
        }
        tx.commit()?;
        Ok(
            json!({"phase":"logs","dryRun":dry_run,"selected":ids.len().min(limit),
            "deleted":if dry_run{0}else{ids.len().min(limit)},"hasMore":ids.len()>limit}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;
    fn fixture() -> (SqliteMemoryStore, tempfile::TempDir, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("batch.db"))).unwrap();
        let mut ids = Vec::new();
        for content in ["first page", "second page", "third page"] {
            ids.push(
                store
                    .ingest(IngestInput {
                        content: content.into(),
                        node_type: "fact".into(),
                        ..Default::default()
                    })
                    .unwrap()
                    .id,
            );
        }
        ids.sort();
        (store, dir, ids)
    }
    #[test]
    fn dream_pages_preserve_other_scopes_and_newer_waking_tags() {
        let (store, _dir, ids) = fixture();
        let started = Utc::now();
        {
            let writer = store.writer.lock().unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET waking_tag=1,waking_tag_at=?1",
                    params![(started - Duration::seconds(1)).to_rfc3339()],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET scope='other' WHERE id=?1",
                    params![ids[2]],
                )
                .unwrap();
            writer
                .execute(
                    "UPDATE knowledge_nodes SET waking_tag_at=?1 WHERE id=?2",
                    params![(started + Duration::seconds(1)).to_rfc3339(), ids[1]],
                )
                .unwrap();
        }
        let (page, more) = store.maintenance_memory_page(1, None, "user").unwrap();
        assert!(more);
        assert_eq!(page[0].id, ids[0]);
        let (next, more) = store
            .maintenance_memory_page(1, Some(&ids[0]), "user")
            .unwrap();
        assert!(!more);
        assert_eq!(next[0].id, ids[1]);
        assert_eq!(store.clear_dream_page_tags(&ids[..2], started).unwrap(), 1);
        let reader = store.reader.lock().unwrap();
        for (id, expected) in [(&ids[0], false), (&ids[1], true), (&ids[2], true)] {
            let tagged: bool = reader
                .query_row(
                    "SELECT waking_tag FROM knowledge_nodes WHERE id=?1",
                    params![id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(tagged, expected);
        }
    }
    #[test]
    fn lifecycle_pages_preserve_protected_and_suppressed_state() {
        let (store, _dir, ids) = fixture();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE knowledge_nodes SET retention_strength=0.01, retrieval_strength=0.02",
                [],
            )
            .unwrap();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE knowledge_nodes SET protected=1 WHERE id=?1",
                params![ids[0]],
            )
            .unwrap();
        store.suppress_memory(&ids[1]).unwrap();
        let suppressed = store.get_node(&ids[1]).unwrap().unwrap();
        let preview = store
            .maintain_lifecycle_batch(2, None, 10000, true)
            .unwrap();
        assert_eq!(preview["changed"], 0);
        assert_eq!(preview["nextCursor"], ids[1]);
        assert_eq!(preview["hasMore"], true);
        let applied = store
            .maintain_lifecycle_batch(3, None, 10000, false)
            .unwrap();
        assert_eq!(applied["changed"], 1);
        assert_eq!(
            store.get_node(&ids[0]).unwrap().unwrap().retention_strength,
            0.01
        );
        assert_eq!(
            store.get_node(&ids[1]).unwrap().unwrap().stability,
            suppressed.stability
        );
        assert!(
            store
                .maintain_lifecycle_batch(1001, None, 1, false)
                .is_err()
        );
    }
    #[test]
    fn gc_page_rolls_back_on_failure_and_never_deletes_protected() {
        let (store, _dir, ids) = fixture();
        store
            .writer
            .lock()
            .unwrap()
            .execute("UPDATE knowledge_nodes SET retention_strength=0.0", [])
            .unwrap();
        store
            .writer
            .lock()
            .unwrap()
            .execute(
                "UPDATE knowledge_nodes SET protected=1 WHERE id=?1",
                params![ids[2]],
            )
            .unwrap();
        let preview = store
            .maintain_gc_batch(3, None, 10000, true, 0.1, None)
            .unwrap();
        assert_eq!(preview["candidateCount"], 2);
        store
            .writer
            .lock()
            .unwrap()
            .execute_batch(
                "CREATE TEMP TRIGGER fail_gc BEFORE DELETE ON knowledge_nodes
            BEGIN SELECT RAISE(ABORT,'injected failure'); END;",
            )
            .unwrap();
        assert!(
            store
                .maintain_gc_batch(3, None, 10000, false, 0.1, None)
                .is_err()
        );
        assert_eq!(store.get_stats().unwrap().total_nodes, 3);
        store
            .writer
            .lock()
            .unwrap()
            .execute_batch("DROP TRIGGER fail_gc")
            .unwrap();
        let applied = store
            .maintain_gc_batch(3, None, 10000, false, 0.1, None)
            .unwrap();
        assert_eq!(applied["deleted"], 2);
        assert!(store.get_node(&ids[2]).unwrap().is_some());
    }
}
