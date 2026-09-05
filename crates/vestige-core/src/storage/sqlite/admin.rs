//! Store open and administration: data directory resolution, connection
//! pragmas, durability profiles, WAL checkpoints, integrity checks, schema
//! introspection, stats and backups.

use super::*;

impl SqliteMemoryStore {
    fn data_dir_from_env() -> Option<PathBuf> {
        std::env::var_os(DATA_DIR_ENV).and_then(|value| {
            if value.is_empty() {
                None
            } else {
                Some(PathBuf::from(value))
            }
        })
    }

    fn expand_tilde(path: PathBuf) -> PathBuf {
        let rest = {
            let mut components = path.components();
            match components.next() {
                Some(Component::Normal(first)) if first == "~" => {
                    Some(components.as_path().to_path_buf())
                }
                _ => None,
            }
        };

        match rest {
            Some(rest) => BaseDirs::new()
                .map(|dirs| dirs.home_dir().join(rest))
                .unwrap_or(path),
            None => path,
        }
    }

    fn prepare_data_dir(data_dir: PathBuf) -> Result<PathBuf> {
        let data_dir = Self::expand_tilde(data_dir);
        std::fs::create_dir_all(&data_dir)?;
        // Restrict directory permissions to owner-only on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            let _ = std::fs::set_permissions(&data_dir, perms);
        }
        Ok(data_dir.join(DATABASE_FILE))
    }

    /// Resolve a Vestige database path from an explicit data directory.
    pub fn db_path_for_data_dir(data_dir: PathBuf) -> Result<PathBuf> {
        Self::prepare_data_dir(data_dir)
    }

    /// Resolve the default Vestige database path.
    ///
    /// `VESTIGE_DATA_DIR` is treated as a directory and wins over the platform
    /// per-user data directory. The database file is always `vestige.db` inside
    /// that directory.
    pub fn default_db_path() -> Result<PathBuf> {
        if let Some(data_dir) = Self::data_dir_from_env() {
            return Self::prepare_data_dir(data_dir);
        }

        let proj_dirs = ProjectDirs::from("com", "vestige", "core").ok_or_else(|| {
            StorageError::Init("Could not determine project directories".to_string())
        })?;

        Self::prepare_data_dir(proj_dirs.data_dir().to_path_buf())
    }

    /// Apply PRAGMAs and optional encryption to a connection.
    pub(super) fn configure_connection(
        conn: &Connection,
        profile: SqliteDurabilityProfile,
        writer: bool,
    ) -> Result<()> {
        // Apply encryption key if SQLCipher is enabled and key is provided
        #[cfg(feature = "encryption")]
        {
            if let Ok(key) = std::env::var("VESTIGE_ENCRYPTION_KEY") {
                if !key.is_empty() {
                    conn.pragma_update(None, "key", &key)?;
                }
            }
        }

        // WAL is persistent database state, so only the writer requests the
        // transition. Every connection still receives its own synchronous,
        // foreign-key, timeout, and full-fsync settings.
        if writer {
            conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        }

        let durability_pragmas = match profile {
            SqliteDurabilityProfile::Hardened => {
                "PRAGMA synchronous = FULL;
                 PRAGMA fullfsync = ON;
                 PRAGMA checkpoint_fullfsync = ON;"
            }
            SqliteDurabilityProfile::Balanced => {
                "PRAGMA synchronous = NORMAL;
                 PRAGMA fullfsync = OFF;
                 PRAGMA checkpoint_fullfsync = OFF;"
            }
        };
        conn.execute_batch(durability_pragmas)?;
        conn.execute_batch(
            "PRAGMA cache_size = -64000;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;
             PRAGMA mmap_size = 268435456;
             PRAGMA wal_autocheckpoint = 1000;
             PRAGMA journal_size_limit = 67108864;",
        )?;

        Ok(())
    }

    fn read_effective_pragmas(conn: &Connection) -> Result<SqliteConnectionPragmas> {
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        let synchronous: i64 = conn.query_row("PRAGMA synchronous", [], |row| row.get(0))?;
        let fullfsync: i64 = conn.query_row("PRAGMA fullfsync", [], |row| row.get(0))?;
        let checkpoint_fullfsync: i64 =
            conn.query_row("PRAGMA checkpoint_fullfsync", [], |row| row.get(0))?;
        let wal_autocheckpoint_pages: i64 =
            conn.query_row("PRAGMA wal_autocheckpoint", [], |row| row.get(0))?;
        let foreign_keys: i64 = conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
        let busy_timeout_ms: i64 = conn.query_row("PRAGMA busy_timeout", [], |row| row.get(0))?;
        let synchronous_label = match synchronous {
            0 => "off",
            1 => "normal",
            2 => "full",
            3 => "extra",
            _ => "unknown",
        }
        .to_string();

        Ok(SqliteConnectionPragmas {
            journal_mode: journal_mode.to_ascii_lowercase(),
            synchronous,
            synchronous_label,
            fullfsync_enabled: fullfsync != 0,
            fullfsync_meaningful_on_this_platform: cfg!(target_os = "macos"),
            checkpoint_fullfsync_enabled: checkpoint_fullfsync != 0,
            wal_autocheckpoint_pages,
            foreign_keys_enabled: foreign_keys != 0,
            busy_timeout_ms,
        })
    }

    pub(super) fn verify_effective_pragmas(
        profile: SqliteDurabilityProfile,
        role: &str,
        pragmas: &SqliteConnectionPragmas,
    ) -> Result<()> {
        if !pragmas.foreign_keys_enabled {
            return Err(StorageError::Init(format!(
                "SQLite {role} connection refused foreign_keys=ON"
            )));
        }
        if profile == SqliteDurabilityProfile::Hardened {
            if pragmas.journal_mode != "wal" {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} journal_mode is '{}' instead of WAL",
                    pragmas.journal_mode
                )));
            }
            if pragmas.synchronous != 2 {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} synchronous is '{}' instead of FULL",
                    pragmas.synchronous_label
                )));
            }
            #[cfg(target_os = "macos")]
            if !pragmas.fullfsync_enabled || !pragmas.checkpoint_fullfsync_enabled {
                return Err(StorageError::Init(format!(
                    "Hardened SQLite startup refused durability downgrade: {role} fullfsync={} checkpoint_fullfsync={} instead of both enabled",
                    pragmas.fullfsync_enabled, pragmas.checkpoint_fullfsync_enabled
                )));
            }
        }
        Ok(())
    }

    fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2
             )",
            params![table, column],
            |row| row.get(0),
        )?;
        Ok(exists != 0)
    }

    fn count_foreign_key_violations(conn: &Connection) -> Result<u64> {
        let mut stmt = conn.prepare("PRAGMA foreign_key_check")?;
        let mut rows = stmt.query([])?;
        let mut count = 0_u64;
        while rows.next()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Delete orphaned child rows whose own foreign key is declared
    /// `ON DELETE CASCADE`. Such a row is unreachable -- its parent is already
    /// gone and the schema states it should have gone with it -- so removing it
    /// restores the invariant without discarding anything a reader could reach.
    /// Rows whose FK is NOT cascade-declared are deliberately left alone so the
    /// caller still fails loudly on genuine corruption.
    fn repair_cascade_orphans(conn: &Connection) -> Result<u64> {
        // (child_table, rowid) pairs reported by the checker.
        let violations: Vec<(String, Option<i64>)> = {
            let mut stmt = conn.prepare("PRAGMA foreign_key_check")?;
            let mapped = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
            })?;
            let mut v = Vec::new();
            for r in mapped {
                v.push(r?);
            }
            v
        };
        if violations.is_empty() {
            return Ok(0);
        }

        // A table is repairable only if EVERY one of its foreign keys that
        // points at knowledge_nodes is ON DELETE CASCADE.
        let mut cascade_ok: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        let mut repaired = 0_u64;
        // This transaction reads (`PRAGMA foreign_key_list`) before it writes
        // (`DELETE`), which is exactly the shape that returns
        // `SQLITE_BUSY_SNAPSHOT` under DEFERRED without consulting the busy
        // handler. It runs while the store is being opened, so a CLI writing
        // beside the server must not be able to fail the open.
        let tx = Self::begin_write_transaction(conn, "repair_cascade_orphans")?;
        for (table, rowid) in violations {
            let Some(rowid) = rowid else { continue };
            let ok = match cascade_ok.get(&table) {
                Some(v) => *v,
                None => {
                    let mut stmt = tx.prepare(&format!(
                        "PRAGMA foreign_key_list(\"{}\")",
                        table.replace('"', "\"\"")
                    ))?;
                    let rows = stmt.query_map([], |row| {
                        Ok((row.get::<_, String>(2)?, row.get::<_, String>(6)?))
                    })?;
                    let mut all_cascade = false;
                    for r in rows {
                        let (parent, on_delete) = r?;
                        if parent.eq_ignore_ascii_case("knowledge_nodes") {
                            all_cascade = on_delete.eq_ignore_ascii_case("CASCADE");
                            if !all_cascade {
                                break;
                            }
                        }
                    }
                    cascade_ok.insert(table.clone(), all_cascade);
                    all_cascade
                }
            };
            if !ok {
                continue;
            }
            let n = tx.execute(
                &format!(
                    "DELETE FROM \"{}\" WHERE rowid = ?1",
                    table.replace('"', "\"\"")
                ),
                params![rowid],
            )?;
            repaired += n as u64;
        }
        tx.commit()?;
        Ok(repaired)
    }

    /// Run `PRAGMA quick_check` and return its rows.
    fn quick_check_rows(conn: &Connection) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let mut stmt = conn.prepare("PRAGMA quick_check")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Names of every FTS5 virtual table in the schema.
    fn fts5_table_names(conn: &Connection) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND sql LIKE '%USING fts5%'",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Is every failing quick_check row attributable to an FTS5 index we can
    /// rebuild? A single row we cannot attribute means real damage, and the
    /// caller must fail rather than paper over it.
    fn quick_check_failure_is_only_fts5(rows: &[String], fts_tables: &[String]) -> bool {
        if fts_tables.is_empty() {
            return false;
        }
        rows.iter().filter(|r| r.as_str() != "ok").all(|r| {
            let lower = r.to_lowercase();
            lower.contains("fts5") && fts_tables.iter().any(|t| lower.contains(&t.to_lowercase()))
        })
    }

    fn run_integrity_checks(conn: &Connection, phase: &str) -> Result<SqliteIntegrityStatus> {
        let mut quick_rows = Self::quick_check_rows(conn)?;

        // An FTS5 external-content index is DERIVED STATE. `knowledge_fts` is
        // declared `content='knowledge_nodes'`, so every token in it is
        // reconstructible from a table quick_check just verified. Refusing to
        // open the whole store because a rebuildable index is damaged strands
        // the user's memories behind an index we can regenerate in seconds --
        // and that is exactly what happened in the field: a store with 2,926
        // intact memories became unopenable over one corrupt fts5 blob.
        //
        // So: rebuild and re-check. Only if the rebuild fails, or the re-check
        // still fails, is this real corruption worth refusing over. This
        // deliberately mirrors the CASCADE-orphan repair below -- repair derived
        // state, fail loudly on genuine damage.
        //
        // Writer phases only. The runtime reader must never attempt a write.
        let repairable_phase = phase == "pre-migration" || phase == "post-migration";
        let quick_ok =
            |rows: &[String]| rows.len() == 1 && rows.first().map(String::as_str) == Some("ok");
        if !quick_ok(&quick_rows) && repairable_phase {
            let fts_tables = Self::fts5_table_names(conn)?;
            if Self::quick_check_failure_is_only_fts5(&quick_rows, &fts_tables) {
                let detail = quick_rows.join("; ");
                let mut rebuilt = Vec::new();
                for table in &fts_tables {
                    // Identifier is read back from sqlite_master, not user input.
                    let quoted = table.replace('"', "\"\"");
                    match conn.execute_batch(&format!(
                        "INSERT INTO \"{quoted}\"(\"{quoted}\") VALUES('rebuild');"
                    )) {
                        Ok(()) => rebuilt.push(table.clone()),
                        Err(error) => {
                            return Err(StorageError::Init(format!(
                                "SQLite {phase} quick_check failed ({detail}) and rebuilding \
                                 FTS index '{table}' also failed: {error}"
                            )));
                        }
                    }
                }
                quick_rows = Self::quick_check_rows(conn)?;
                if quick_ok(&quick_rows) {
                    tracing::warn!(
                        phase,
                        rebuilt = ?rebuilt,
                        detail,
                        "rebuilt corrupt FTS index from its content table; store opened normally"
                    );
                }
            }
        }

        let quick_check = quick_rows.join("; ");
        if !quick_ok(&quick_rows) {
            return Err(StorageError::Init(format!(
                "SQLite {phase} quick_check failed: {quick_check}"
            )));
        }

        let mut foreign_key_violations = Self::count_foreign_key_violations(conn)?;
        if foreign_key_violations != 0 && phase == "pre-migration" {
            // Deletion residue from older builds (and from any delete path that
            // ran without `PRAGMA foreign_keys = ON`) leaves child rows whose
            // knowledge_nodes parent is already gone. Those rows are unreachable
            // by construction, and their own schema says ON DELETE CASCADE --
            // "if the parent goes, I go". Refusing to open the database over
            // them bricks every store that predates FK enforcement, with no
            // recovery path short of manual SQLite surgery. Repair them here,
            // BEFORE migrations, then re-check. Only CASCADE-declared FKs are
            // repaired; anything else still fails loudly below.
            let repaired = Self::repair_cascade_orphans(conn)?;
            foreign_key_violations = Self::count_foreign_key_violations(conn)?;
            if repaired > 0 {
                tracing::warn!(
                    repaired,
                    remaining = foreign_key_violations,
                    "repaired orphaned child rows left by an earlier delete (ON DELETE CASCADE residue)"
                );
            }
        }
        if foreign_key_violations != 0 {
            return Err(StorageError::Init(format!(
                "SQLite {phase} foreign_key_check found {foreign_key_violations} violation(s)"
            )));
        }

        let synaptic_tables = [
            "synaptic_tags",
            "synaptic_events",
            "synaptic_capture_items",
            "memory_receipts",
        ];
        let mut synaptic_checks_applied = true;
        for table in synaptic_tables {
            if !Self::table_exists(conn, table)? {
                synaptic_checks_applied = false;
                break;
            }
        }

        let synaptic_consistency_violations = if synaptic_checks_applied {
            let missing_receipts: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_events e
                 LEFT JOIN memory_receipts r ON r.receipt_id = e.receipt_id
                 WHERE e.receipt_id IS NULL OR r.receipt_id IS NULL",
                [],
                |row| row.get(0),
            )?;
            let invalid_event_receipt_predicates =
                if Self::table_has_column(conn, "synaptic_events", "public_event_id")? {
                    conn.query_row(
                        "SELECT COUNT(*)
                     FROM synaptic_events e
                     JOIN memory_receipts r ON r.receipt_id = e.receipt_id
                     WHERE CASE json_extract(r.payload, '$.evidence.predicate.schemaVersion')
                         WHEN 1 THEN
                                json_extract(r.payload, '$.evidence.kind')
                                    IS NOT 'synaptic_capture'
                             OR e.algorithm_version IS NOT 'vestige.synaptic_capture.v1'
                             OR e.public_event_id IS NULL
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v1'
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                         WHEN 2 THEN
                                json_extract(r.payload, '$.evidence.kind')
                                    IS NOT 'synaptic_capture'
                             OR e.algorithm_version IS NOT 'vestige.synaptic_capture.v2'
                             OR e.public_event_id IS NULL
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v2'
                             OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                    IS NOT 'root'
                             OR json_type(
                                    r.payload,
                                    '$.evidence.predicate.parentReceiptId'
                                ) IS NOT NULL
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                         ELSE 1
                     END",
                        [],
                        |row| row.get::<_, i64>(0),
                    )?
                } else {
                    0
                };
            let invalid_items: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_capture_items i
                 LEFT JOIN synaptic_events e ON e.event_id = i.event_id
                 LEFT JOIN synaptic_tags t ON t.tag_id = i.tag_id
                 LEFT JOIN memory_receipts r ON r.receipt_id = i.receipt_id
                 WHERE e.event_id IS NULL OR t.tag_id IS NULL OR r.receipt_id IS NULL
                    OR i.memory_id IS NOT t.memory_id",
                [],
                |row| row.get(0),
            )?;
            // V21 stores one root receipt id on both the event and every item.
            // V22 may store a per-pair child receipt on an item, so the startup
            // invariant becomes predicate-version aware once the V22 columns
            // exist. Preparing this SQL conditionally keeps pre-V22 databases
            // valid during checks that run before pending migrations.
            let invalid_item_receipt_predicates = if Self::table_has_column(
                conn,
                "synaptic_events",
                "public_event_id",
            )? && Self::table_has_column(
                conn,
                "synaptic_capture_items",
                "evaluation_direction",
            )? {
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM synaptic_capture_items i
                     JOIN synaptic_events e ON e.event_id = i.event_id
                     JOIN memory_receipts r ON r.receipt_id = i.receipt_id
                     WHERE CASE json_extract(r.payload, '$.evidence.predicate.schemaVersion')
                         WHEN 1 THEN
                                i.receipt_id <> e.receipt_id
                             OR i.evaluation_direction IS NOT 'backward'
                             OR i.algorithm_version IS NOT 'vestige.synaptic_capture.v1'
                         WHEN 2 THEN
                                json_extract(r.payload, '$.evidence.kind') IS NOT 'synaptic_capture'
                             OR i.algorithm_version IS NOT 'vestige.synaptic_capture.v2'
                             OR i.evaluation_direction NOT IN ('backward', 'forward')
                             OR json_extract(r.payload, '$.evidence.predicate.algorithmVersion')
                                    IS NOT 'vestige.synaptic_capture.v2'
                             OR CASE i.evaluation_direction
                                  WHEN 'backward' THEN
                                         i.receipt_id <> e.receipt_id
                                      OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                             IS NOT 'root'
                                      OR json_type(
                                             r.payload,
                                             '$.evidence.predicate.parentReceiptId'
                                         ) IS NOT NULL
                                  WHEN 'forward' THEN
                                         i.receipt_id = e.receipt_id
                                      OR json_extract(r.payload, '$.evidence.predicate.receiptRole')
                                             IS NOT 'pair'
                                      OR json_extract(r.payload, '$.evidence.predicate.parentReceiptId')
                                             IS NOT e.receipt_id
                                  ELSE 1
                                END
                             OR e.public_event_id IS NULL
                             OR json_type(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT 'text'
                             OR json_extract(r.payload, '$.evidence.predicate.trigger.eventId')
                                    IS NOT e.public_event_id
                             OR json_extract(r.payload, '$.evidence.predicate.evaluationDirection')
                                    IS NOT i.evaluation_direction
                             OR json_array_length(
                                    json_extract(r.payload, '$.evidence.predicate.candidates')
                                ) IS NOT 1
                             OR json_extract(
                                    r.payload,
                                    '$.evidence.predicate.candidates[0].evidenceSlot'
                                ) IS NOT i.evidence_slot
                         ELSE 1
                     END",
                    [],
                    |row| row.get::<_, i64>(0),
                )?
            } else {
                conn.query_row(
                    "SELECT COUNT(*)
                     FROM synaptic_capture_items i
                     JOIN synaptic_events e ON e.event_id = i.event_id
                     WHERE i.receipt_id <> e.receipt_id",
                    [],
                    |row| row.get::<_, i64>(0),
                )?
            };
            let duplicate_active_tags: i64 = conn.query_row(
                "SELECT COUNT(*) FROM (
                     SELECT memory_id
                     FROM synaptic_tags
                     WHERE state = 'active'
                     GROUP BY memory_id
                     HAVING COUNT(*) > 1
                 )",
                [],
                |row| row.get(0),
            )?;
            let invalid_captured_tags: i64 = conn.query_row(
                "SELECT COUNT(*)
                 FROM synaptic_tags t
                 LEFT JOIN synaptic_events e ON e.event_id = t.capture_event_id
                 LEFT JOIN synaptic_capture_items i
                   ON i.event_id = t.capture_event_id
                  AND i.tag_id = t.tag_id
                  AND i.disposition = 'captured'
                 WHERE (t.state = 'captured' AND (
                           t.capture_event_id IS NULL
                        OR t.captured_at_ms IS NULL
                        OR e.event_id IS NULL
                        OR i.tag_id IS NULL
                       ))
                    OR (t.state <> 'captured' AND (
                           t.capture_event_id IS NOT NULL
                        OR t.captured_at_ms IS NOT NULL
                       ))",
                [],
                |row| row.get(0),
            )?;
            (missing_receipts
                + invalid_event_receipt_predicates
                + invalid_items
                + invalid_item_receipt_predicates
                + duplicate_active_tags
                + invalid_captured_tags) as u64
        } else {
            0
        };
        if synaptic_consistency_violations != 0 {
            return Err(StorageError::Init(format!(
                "SQLite {phase} synaptic receipt consistency checks found {synaptic_consistency_violations} violation(s)"
            )));
        }

        Ok(SqliteIntegrityStatus {
            quick_check,
            foreign_key_violations,
            synaptic_checks_applied,
            synaptic_consistency_violations,
        })
    }

    fn checkpoint_connection(
        conn: &Connection,
        mode: WalCheckpointMode,
    ) -> Result<WalCheckpointStatus> {
        let sql = match mode {
            WalCheckpointMode::Passive => "PRAGMA wal_checkpoint(PASSIVE)",
            WalCheckpointMode::Truncate => "PRAGMA wal_checkpoint(TRUNCATE)",
        };
        let (busy, log_frames, checkpointed_frames) =
            conn.query_row(sql, [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        Ok(WalCheckpointStatus {
            busy,
            log_frames,
            checkpointed_frames,
        })
    }

    /// Create new storage instance
    pub fn new(db_path: Option<PathBuf>) -> Result<Self> {
        Self::new_with_durability(db_path, SqliteDurabilityProfile::from_env()?)
    }

    /// Create storage with an explicit durability policy.
    ///
    /// This is primarily useful for controlled benchmarks and embedded callers
    /// that cannot use process environment configuration.
    pub fn new_with_durability(
        db_path: Option<PathBuf>,
        profile: SqliteDurabilityProfile,
    ) -> Result<Self> {
        let path = match db_path {
            Some(p) => p,
            None => Self::default_db_path()?,
        };

        // Open writer connection
        let writer_conn = Connection::open(&path)?;

        // Restrict database file permissions to owner-only on Unix
        #[cfg(unix)]
        if path.exists() {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perms);
        }

        Self::configure_connection(&writer_conn, profile, true)?;
        let writer_pragmas = Self::read_effective_pragmas(&writer_conn)?;
        Self::verify_effective_pragmas(profile, "writer", &writer_pragmas)?;

        // Opening the database lets SQLite recover a committed WAL. Validate
        // that recovered state before migrations can change the schema.
        let before_migrations = Self::run_integrity_checks(&writer_conn, "pre-migration")?;

        // Apply migrations on writer only
        crate::storage::migrations::apply_migrations(&writer_conn)?;
        // Issue #191: heal v1.x raw-768 vectors copied verbatim under the
        // 256-dim legacy profile before any strict dimension check can abort
        // startup. No-op on clean stores.
        crate::storage::migrations::repair_legacy_raw_profile_vectors(
            &writer_conn,
            LEGACY_EMBEDDING_PROFILE_ID,
        )?;
        writer_conn.execute_batch("PRAGMA optimize = 0x10002;")?;
        let after_migrations = Self::run_integrity_checks(&writer_conn, "post-migration")?;
        let startup_checkpoint =
            Self::checkpoint_connection(&writer_conn, WalCheckpointMode::Passive)?;

        // Open reader connection to same path
        let reader_conn = Connection::open(&path)?;
        Self::configure_connection(&reader_conn, profile, false)?;
        let reader_pragmas = Self::read_effective_pragmas(&reader_conn)?;
        Self::verify_effective_pragmas(profile, "reader", &reader_pragmas)?;

        let durability_status = SqliteDurabilityStatus {
            profile,
            writer: writer_pragmas,
            reader: reader_pragmas,
            before_migrations,
            after_migrations,
            startup_checkpoint,
            commit_acknowledgement: match profile {
                SqliteDurabilityProfile::Hardened => {
                    "tx.commit() returned after SQLite FULL WAL synchronization"
                }
                SqliteDurabilityProfile::Balanced => {
                    "tx.commit() returned under SQLite NORMAL WAL synchronization"
                }
            }
            .to_string(),
            claim_boundary: "Process-crash tests prove transaction atomicity and recovery at the tested commit boundaries. Power-loss durability still depends on the operating system, filesystem, controller, and storage device honoring completed flush requests; WAL requires local shared-memory and locking semantics."
                .to_string(),
        };

        #[cfg(feature = "embeddings")]
        let embedding_service = EmbeddingService::new();

        #[cfg(feature = "vector-search")]
        let vector_index = if Self::vector_search_enabled_by_cpu() {
            let vector_index = VectorIndex::new()
                .map_err(|e| StorageError::Init(format!("Failed to create vector index: {}", e)))?;
            Some(Mutex::new(vector_index))
        } else {
            tracing::warn!(
                "Vector search disabled: {}",
                Self::vector_search_unavailable_reason().unwrap_or("manual override"),
            );
            None
        };

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let query_cache = if vector_index.is_some() {
            Some(Mutex::new(LruCache::new(
                NonZeroUsize::new(100).expect("100 is non-zero"),
            )))
        } else {
            None
        };

        let storage = Self {
            db_path: path,
            durability_status,
            writer: Mutex::new(writer_conn),
            reader: Mutex::new(reader_conn),
            scheduler: Mutex::new(FSRSScheduler::default()),
            #[cfg(feature = "embeddings")]
            embedding_service,
            #[cfg(feature = "vector-search")]
            vector_index,
            #[cfg(feature = "vector-search")]
            vector_index_watermark: Mutex::new(VectorIndexWatermark::default()),
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            query_cache,
            #[cfg(all(feature = "embeddings", feature = "vector-search"))]
            attached_profile_runtime: RwLock::new(None),
            registered_model: std::sync::RwLock::new(None),
        };

        // V20 seeds a minimal SQL row so old databases migrate atomically.
        // Replace that bootstrap with the complete, serializable manifest before
        // exposing the store to callers.
        storage.ensure_legacy_embedding_profile_manifest()?;

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        if storage.vector_index.is_some() {
            storage.load_embeddings_into_index()?;
        }

        Ok(storage)
    }

    /// Absolute path of the SQLite database this storage instance uses.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Verified durability profile and startup-recovery results.
    pub fn durability_status(&self) -> &SqliteDurabilityStatus {
        &self.durability_status
    }

    /// Run an explicit SQLite WAL checkpoint and return SQLite's raw counters.
    ///
    /// `Passive` is safe for live status/recovery workflows. `Truncate` should
    /// be used only after application writers have stopped (for example, at a
    /// quiesced backup or graceful-shutdown boundary); it is not what makes an
    /// already-acknowledged hardened commit durable.
    pub fn checkpoint_wal(&self, mode: WalCheckpointMode) -> Result<WalCheckpointStatus> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        Self::checkpoint_connection(&writer, mode)
    }

    /// Re-run integrity and V21 consistency checks against the live database.
    pub fn verify_integrity(&self) -> Result<SqliteIntegrityStatus> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        Self::run_integrity_checks(&reader, "runtime")
    }

    /// Data directory containing the SQLite database and sidecar folders.
    pub fn data_dir(&self) -> &Path {
        self.db_path.parent().unwrap_or_else(|| Path::new("."))
    }

    /// Sidecar directory for files belonging to this storage instance.
    pub fn sidecar_dir(&self, name: &str) -> PathBuf {
        self.data_dir().join(name)
    }

    /// Return the profile-scoped HNSW sidecar location. The profile ID is
    /// validated before being placed in a path, preventing traversal through a
    /// manifest or CLI argument.
    pub fn embedding_profile_index_dir(&self, profile_id: &EmbeddingProfileId) -> Result<PathBuf> {
        EmbeddingProfileId::new(profile_id.as_str().to_string())
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        Ok(self
            .sidecar_dir("embedding-profiles")
            .join(profile_id.as_str())
            .join("hnsw"))
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats> {
        let now = Utc::now().to_rfc3339();

        // Resolve the active pointer before taking the shared reader lock.
        // `active_embedding_profile` reads through that same mutex; calling it
        // below after acquiring `reader` would self-deadlock every stats read.
        #[cfg(feature = "embeddings")]
        let active_profile = self.active_embedding_profile()?;

        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let total: i64 =
            reader.query_row("SELECT COUNT(*) FROM knowledge_nodes", [], |row| row.get(0))?;

        let due: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE next_review <= ?1",
            params![now],
            |row| row.get(0),
        )?;

        let avg_retention: f64 = reader.query_row(
            "SELECT COALESCE(AVG(retention_strength), 0) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_storage: f64 = reader.query_row(
            "SELECT COALESCE(AVG(storage_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let avg_retrieval: f64 = reader.query_row(
            "SELECT COALESCE(AVG(retrieval_strength), 1) FROM knowledge_nodes",
            [],
            |row| row.get(0),
        )?;

        let oldest: Option<String> = reader
            .query_row("SELECT MIN(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let newest: Option<String> = reader
            .query_row("SELECT MAX(created_at) FROM knowledge_nodes", [], |row| {
                row.get(0)
            })
            .ok();

        let nodes_with_embeddings: i64 = reader.query_row(
            "SELECT COUNT(*) FROM knowledge_nodes WHERE has_embedding = 1",
            [],
            |row| row.get(0),
        )?;

        let embedding_model: Option<String> = reader
            .query_row(
                "SELECT model
                 FROM node_embeddings
                 GROUP BY model
                 ORDER BY COUNT(*) DESC, model ASC
                 LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;

        #[cfg(feature = "embeddings")]
        let active_embedding_model = active_profile.as_ref().and_then(|active| {
            reader
                .query_row(
                    "SELECT model_id FROM embedding_profiles WHERE profile_id = ?1",
                    params![active.profile_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .ok()
        });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model: Option<String> = None;

        #[cfg(feature = "embeddings")]
        let (nodes_with_active_embeddings, nodes_with_mismatched_embeddings) = {
            let active_profile_id = active_profile
                .as_ref()
                .map(|active| active.profile_id.as_str());
            let active_model = active_embedding_model.as_deref();
            let active_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv
                       WHERE epv.node_id = kn.id
                         AND epv.profile_id = ?1
                         AND epv.model = ?2
                         AND epv.dimensions = (
                             SELECT embedding_dimension FROM embedding_profiles
                             WHERE profile_id = ?1
                         )
                   )",
                params![active_profile_id, active_model],
                |row| row.get(0),
            )?;
            let mismatched_count: i64 = reader.query_row(
                "SELECT COUNT(*)
                 FROM knowledge_nodes kn
                 WHERE (kn.has_embedding = 1 OR EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv WHERE epv.node_id = kn.id
                   ))
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_profile_vectors epv
                       WHERE epv.node_id = kn.id
                         AND epv.profile_id = ?1
                         AND epv.model = ?2
                         AND epv.dimensions = (
                             SELECT embedding_dimension FROM embedding_profiles
                             WHERE profile_id = ?1
                         )
                   )",
                params![active_profile_id, active_model],
                |row| row.get(0),
            )?;
            (active_count, mismatched_count)
        };
        #[cfg(not(feature = "embeddings"))]
        let (nodes_with_active_embeddings, nodes_with_mismatched_embeddings) =
            (nodes_with_embeddings, 0);

        Ok(MemoryStats {
            total_nodes: total,
            nodes_due_for_review: due,
            average_retention: avg_retention,
            average_storage_strength: avg_storage,
            average_retrieval_strength: avg_retrieval,
            oldest_memory: oldest.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }),
            newest_memory: newest.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }),
            nodes_with_embeddings,
            nodes_with_active_embeddings,
            nodes_with_mismatched_embeddings,
            embedding_model,
            active_embedding_model,
        })
    }

    /// Introspect the live SQLite schema: schema version + per-table row/column
    /// shape + embedding-coverage convenience fields.
    ///
    /// This is the v2.1.24+ replacement for the direct-SQLite reads that
    /// audit scripts and migration guards previously had to perform. The set
    /// of tables walked matches `PORTABLE_USER_DATA_TABLES` — the same
    /// canonical set used by portable export / import — so the surface stays
    /// stable across migrations rather than chasing arbitrary
    /// `sqlite_master` rows.
    ///
    /// Cost: O(N_tables) `COUNT(*)` queries + one PRAGMA per table. Negligible
    /// at the table cardinalities Vestige carries (~15 tables, all indexed).
    /// Safe to call on every MCP `system_status` invocation when the flag is
    /// set; callers wanting to limit cost should leave the flag off (default).
    pub fn schema_introspection(&self) -> Result<crate::SchemaIntrospection> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        let schema_version = Self::current_schema_version(&reader)?;

        // schema_version has the row (version PK + applied_at TEXT). Read the
        // applied_at for the current version row; tolerate failure (legacy
        // databases may have skipped the applied_at fill on early upgrades).
        let applied_at_str: Option<String> = reader
            .query_row(
                "SELECT applied_at FROM schema_version WHERE version = ?1",
                params![schema_version as i64],
                |row| row.get(0),
            )
            .optional()?;
        let schema_version_applied_at = applied_at_str.and_then(|s| {
            // The migration scripts use `datetime('now')` which yields
            // SQLite's "YYYY-MM-DD HH:MM:SS" UTC form (NOT RFC3339).
            // Try the SQLite form first, fall back to RFC3339 for any
            // future migrations that switch.
            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                .map(|naive| naive.and_utc())
                .or_else(|_| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
                .ok()
        });

        let mut tables = Vec::with_capacity(PORTABLE_USER_DATA_TABLES.len());
        for table_name in PORTABLE_USER_DATA_TABLES {
            if Self::table_exists(&reader, table_name)? {
                let rows = Self::table_row_count(&reader, table_name)?;
                let columns = Self::table_columns(&reader, table_name)?;
                tables.push(crate::TableIntrospection {
                    name: (*table_name).to_string(),
                    rows,
                    columns,
                });
            }
        }

        // Convenience: active-profile coverage is the number of nodes with no
        // vector in the currently selected isolated vector space.
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?;
        let embedding_null_count: i64 = reader
            .query_row(
                "SELECT COUNT(*) FROM knowledge_nodes kn
                 WHERE NOT EXISTS (
                     SELECT 1 FROM embedding_profile_vectors epv
                     WHERE epv.node_id = kn.id AND epv.profile_id = ?1
                 )",
                params![active_profile_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        #[cfg(feature = "embeddings")]
        let active_embedding_model = active_profile_id.as_deref().and_then(|profile_id| {
            reader
                .query_row(
                    "SELECT model_id FROM embedding_profiles WHERE profile_id = ?1",
                    params![profile_id],
                    |row| row.get::<_, String>(0),
                )
                .ok()
        });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_model: Option<String> = None;

        #[cfg(feature = "embeddings")]
        let active_embedding_dimensions: Option<u32> =
            active_profile_id.as_deref().and_then(|profile_id| {
                reader
                    .query_row(
                        "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                        params![profile_id],
                        |row| row.get::<_, i64>(0),
                    )
                    .ok()
                    .and_then(|dimension| u32::try_from(dimension).ok())
            });
        #[cfg(not(feature = "embeddings"))]
        let active_embedding_dimensions: Option<u32> = None;

        Ok(crate::SchemaIntrospection {
            schema_version,
            schema_version_applied_at,
            tables,
            embedding_null_count,
            active_embedding_model,
            active_embedding_dimensions,
        })
    }

    fn scan_last_backup_timestamp(backup_dir: &Path) -> Option<DateTime<Utc>> {
        if !backup_dir.exists() {
            return None;
        }

        let mut latest: Option<DateTime<Utc>> = None;

        if let Ok(entries) = std::fs::read_dir(backup_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // Parse vestige-YYYYMMDD-HHMMSS.db
                if let Some(ts_part) = name_str
                    .strip_prefix("vestige-")
                    .and_then(|s| s.strip_suffix(".db"))
                    && let Ok(naive) =
                        chrono::NaiveDateTime::parse_from_str(ts_part, "%Y%m%d-%H%M%S")
                {
                    let dt = naive.and_utc();
                    if latest.as_ref().is_none_or(|l| dt > *l) {
                        latest = Some(dt);
                    }
                }
            }
        }

        latest
    }

    /// Get last backup timestamp for this storage instance.
    /// Parses `vestige-YYYYMMDD-HHMMSS.db` filenames.
    pub fn last_backup_timestamp(&self) -> Option<DateTime<Utc>> {
        Self::scan_last_backup_timestamp(&self.sidecar_dir("backups"))
    }

    /// Get last backup timestamp in the default backups directory.
    /// Kept for compatibility with older callers.
    pub fn get_last_backup_timestamp() -> Option<DateTime<Utc>> {
        let backup_dir = Self::default_db_path().ok()?.parent()?.join("backups");
        Self::scan_last_backup_timestamp(&backup_dir)
    }

    /// Create a consistent backup using VACUUM INTO
    pub fn backup_to(&self, path: &std::path::Path) -> Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| StorageError::Init("Invalid backup path encoding".to_string()))?;
        // Validate path: reject control characters (except tab) for defense-in-depth
        if path_str.bytes().any(|b| b < 0x20 && b != b'\t') {
            return Err(StorageError::Init(
                "Backup path contains invalid characters".to_string(),
            ));
        }
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        // VACUUM INTO doesn't support parameterized queries; escape single quotes
        reader.execute_batch(&format!("VACUUM INTO '{}'", path_str.replace('\'', "''")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(path, perms)?;
        }
        Ok(())
    }
}
