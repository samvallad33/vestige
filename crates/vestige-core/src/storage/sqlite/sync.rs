//! Portable archive export, import and device sync, including the merge
//! rules for combining two stores.

use super::*;

impl SqliteMemoryStore {
    /// Export an exact portable archive preserving raw Vestige storage rows.
    ///
    /// Unlike the user-facing JSON export, this preserves IDs, timestamps,
    /// FSRS state, graph edges, suppression state, history tables, and raw
    /// embedding blobs. It is intended for Vestige-to-Vestige device transfer.
    pub fn export_portable_archive(&self) -> Result<PortableArchive> {
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let tx = reader.transaction()?;

        let schema_version = Self::current_schema_version(&tx)?;
        let mut tables = Vec::new();

        for table_name in PORTABLE_TABLES {
            if !Self::table_exists(&tx, table_name)? {
                continue;
            }

            let quoted_table = Self::quote_ident(table_name);
            let mut stmt = tx.prepare(&format!("SELECT * FROM {} ORDER BY rowid", quoted_table))?;
            let columns: Vec<String> = stmt
                .column_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect();
            let column_count = columns.len();

            let rows = stmt.query_map([], |row| {
                let mut values = Vec::with_capacity(column_count);
                for idx in 0..column_count {
                    values.push(Self::portable_value_from_ref(row.get_ref(idx)?)?);
                }
                Ok(values)
            })?;

            let mut portable_rows = Vec::new();
            for row in rows {
                portable_rows.push(row?);
            }

            tables.push(PortableTable {
                name: (*table_name).to_string(),
                columns,
                rows: portable_rows,
            });
        }

        let archive = PortableArchive {
            archive_format: PORTABLE_ARCHIVE_FORMAT.to_string(),
            vestige_version: crate::VERSION.to_string(),
            schema_version,
            exported_at: Utc::now(),
            mode: "exact".to_string(),
            tables,
        };
        tx.commit()?;
        Ok(archive)
    }

    /// Write an exact portable archive to a JSON file.
    pub fn export_portable_archive_to_path(
        &self,
        path: &std::path::Path,
    ) -> Result<PortableArchive> {
        let archive = self.export_portable_archive()?;
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("vestige-portable.json");
        let temp_path = parent.join(format!(".{}.tmp-{}", filename, Uuid::new_v4()));

        #[cfg(unix)]
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp_path)?
        };
        #[cfg(not(unix))]
        let mut file = std::fs::File::create(&temp_path)?;
        if let Err(e) = serde_json::to_writer_pretty(&mut file, &archive) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(StorageError::Init(format!(
                "Failed to write portable archive: {}",
                e
            )));
        }
        file.flush()?;
        file.sync_all()?;
        drop(file);

        if let Err(rename_err) = std::fs::rename(&temp_path, path) {
            if path.exists() {
                std::fs::remove_file(path)?;
                std::fs::rename(&temp_path, path)?;
            } else {
                let _ = std::fs::remove_file(&temp_path);
                return Err(rename_err.into());
            }
        }
        Ok(archive)
    }

    /// Import an exact portable archive.
    ///
    /// `EmptyOnly` preserves the conservative migration path. `Merge` is used
    /// by portable sync to combine non-empty databases with tombstones and
    /// newer-local conflict handling.
    pub fn import_portable_archive(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
    ) -> Result<PortableImportReport> {
        self.import_portable_archive_with_secret_policy(archive, mode, SecretPolicy::Reject)
    }

    /// Import an exact archive using an explicit credential-storage policy.
    ///
    /// The archive is preflighted before a writer or transaction is opened, so
    /// a rejected archive cannot partially import safe sibling rows.
    pub fn import_portable_archive_with_secret_policy(
        &self,
        archive: &PortableArchive,
        mode: PortableImportMode,
        policy: SecretPolicy,
    ) -> Result<PortableImportReport> {
        if archive.archive_format != PORTABLE_ARCHIVE_FORMAT {
            return Err(StorageError::Init(format!(
                "Unsupported portable archive format '{}'",
                archive.archive_format
            )));
        }
        if archive.mode != "exact" {
            return Err(StorageError::Init(format!(
                "Unsupported portable archive mode '{}'",
                archive.mode
            )));
        }

        Self::enforce_secret_policy_for_portable_archive(archive, policy)?;

        let mut seen_tables = std::collections::HashSet::new();
        let mut tables_by_name = std::collections::HashMap::new();
        for table in &archive.tables {
            if !PORTABLE_TABLES.contains(&table.name.as_str()) {
                return Err(StorageError::Init(format!(
                    "Portable archive contains unsupported table '{}'",
                    table.name
                )));
            }
            if !seen_tables.insert(table.name.as_str()) {
                return Err(StorageError::Init(format!(
                    "Portable archive contains duplicate table '{}'",
                    table.name
                )));
            }
            tables_by_name.insert(table.name.as_str(), table);
        }

        let mut report = PortableImportReport {
            tables_imported: 0,
            rows_imported: 0,
            tables_skipped: 0,
            fts_rebuilt: false,
            rows_inserted: 0,
            rows_updated: 0,
            rows_skipped: 0,
            rows_deleted: 0,
            conflicts_kept_local: 0,
        };

        {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

            let current_schema = Self::current_schema_version(&writer)?;
            if archive.schema_version > current_schema {
                return Err(StorageError::Init(format!(
                    "Archive schema version {} is newer than this Vestige database schema {}",
                    archive.schema_version, current_schema
                )));
            }

            match mode {
                PortableImportMode::EmptyOnly => {
                    Self::ensure_portable_import_target_empty(&writer)?
                }
                PortableImportMode::Merge => {}
            }

            let tx = Self::begin_write_transaction(
                &writer,
                "import_portable_archive_with_secret_policy",
            )?;
            let mut merge_state = PortableMergeState::default();

            for table_name in PORTABLE_TABLES {
                let Some(table) = tables_by_name.get(table_name) else {
                    continue;
                };

                if !Self::table_exists(&tx, table_name)? {
                    report.tables_skipped += 1;
                    continue;
                }

                if mode == PortableImportMode::Merge {
                    Self::merge_portable_table(
                        &tx,
                        table_name,
                        table,
                        &mut report,
                        &mut merge_state,
                    )?;
                    report.tables_imported += 1;
                    continue;
                }

                let target_columns = Self::table_columns(&tx, table_name)?;
                let mut insert_columns = Vec::new();
                let mut source_indexes = Vec::new();

                for (idx, column) in table.columns.iter().enumerate() {
                    if target_columns.iter().any(|target| target == column) {
                        insert_columns.push(column.clone());
                        source_indexes.push(idx);
                    }
                }

                if insert_columns.is_empty() {
                    report.tables_skipped += 1;
                    continue;
                }

                let quoted_table = Self::quote_ident(table_name);
                let quoted_columns = insert_columns
                    .iter()
                    .map(|column| Self::quote_ident(column))
                    .collect::<Vec<_>>()
                    .join(", ");
                let placeholders = std::iter::repeat_n("?", insert_columns.len())
                    .collect::<Vec<_>>()
                    .join(", ");
                let verb = if *table_name == "fsrs_config" {
                    "INSERT OR REPLACE"
                } else {
                    "INSERT"
                };
                let sql = format!(
                    "{} INTO {} ({}) VALUES ({})",
                    verb, quoted_table, quoted_columns, placeholders
                );

                for row in &table.rows {
                    if row.len() != table.columns.len() {
                        return Err(StorageError::Init(format!(
                            "Portable archive row in table '{}' has {} values for {} columns",
                            table_name,
                            row.len(),
                            table.columns.len()
                        )));
                    }

                    let values = source_indexes
                        .iter()
                        .map(|idx| row[*idx].to_sql_value())
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(|e| {
                            StorageError::Init(format!("Invalid portable value: {}", e))
                        })?;
                    tx.execute(&sql, params_from_iter(values))?;
                    report.rows_imported += 1;
                    report.rows_inserted += 1;
                }

                report.tables_imported += 1;
            }

            if Self::table_exists(&tx, "knowledge_fts")? {
                tx.execute(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')",
                    [],
                )?;
                report.fts_rebuilt = true;
            }

            tx.commit()?;
        }

        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        self.load_embeddings_into_index()?;

        Ok(report)
    }

    /// Read and import an exact portable archive JSON file.
    pub fn import_portable_archive_from_path(
        &self,
        path: &std::path::Path,
        mode: PortableImportMode,
    ) -> Result<PortableImportReport> {
        let file = std::fs::File::open(path)?;
        let archive: PortableArchive = serde_json::from_reader(file)
            .map_err(|e| StorageError::Init(format!("Failed to parse portable archive: {}", e)))?;
        self.import_portable_archive(&archive, mode)
    }

    /// Synchronize this database with a pluggable portable archive backend.
    ///
    /// Sync is pull-merge-push:
    /// 1. read remote archive if present,
    /// 2. merge it into the local database using tombstones and conflict rules,
    /// 3. export the merged local database,
    /// 4. write the archive back through the backend.
    pub fn sync_portable_archive<B: PortableSyncBackend>(
        &self,
        backend: &B,
    ) -> Result<PortableSyncReport> {
        let (pulled, pull) = match backend.read_archive()? {
            Some(remote) => (
                true,
                Some(self.import_portable_archive(&remote, PortableImportMode::Merge)?),
            ),
            None => (false, None),
        };

        let archive = self.export_portable_archive()?;
        let pushed_tables = archive.tables.len();
        let pushed_rows = archive.total_rows();
        let archive_format = archive.archive_format.clone();
        backend.write_archive(&archive)?;

        Ok(PortableSyncReport {
            backend: backend.label(),
            pulled,
            pull,
            pushed_tables,
            pushed_rows,
            archive_format,
        })
    }

    /// Synchronize this database with a file-backed portable archive.
    pub fn sync_portable_archive_file(&self, path: &std::path::Path) -> Result<PortableSyncReport> {
        let backend = FilePortableSyncBackend::new(path);
        self.sync_portable_archive(&backend)
    }

    /// Synchronize this database with the hosted Vestige Cloud managed-sync
    /// service. `endpoint` is the base URL (e.g. `https://sync.vestige.dev`) and
    /// `sync_key` is the per-user key issued at purchase. Pull-merge-push is
    /// identical to file sync — only the transport differs.
    ///
    /// When `encryption_key` is `Some`, the archive is encrypted client-side
    /// (XChaCha20-Poly1305) before upload, so the server only stores ciphertext
    /// (zero-knowledge). The passphrase never leaves this process.
    #[cfg(feature = "cloud-sync")]
    pub fn sync_portable_archive_cloud(
        &self,
        endpoint: &str,
        sync_key: &str,
        encryption_key: Option<String>,
    ) -> Result<PortableSyncReport> {
        let backend = crate::storage::cloud_sync::HttpPortableSyncBackend::new_with_encryption(
            endpoint,
            sync_key,
            encryption_key,
        )?;
        self.sync_portable_archive(&backend)
    }

    fn merge_portable_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        report: &mut PortableImportReport,
        state: &mut PortableMergeState,
    ) -> Result<()> {
        match table_name {
            "sync_tombstones" => Self::merge_sync_tombstones(tx, table, report),
            "knowledge_nodes" => Self::merge_knowledge_nodes(tx, table, report, state),
            "memory_access_log"
            | "state_transitions"
            | "consolidation_history"
            | "dream_history"
            | "retention_snapshots" => Self::merge_append_only_table(tx, table_name, table, report),
            "composition_events" | "composition_outcomes" => {
                Self::merge_keyed_table(tx, table_name, table, &["id"], report, state)
            }
            "composition_members" => Self::merge_keyed_table(
                tx,
                table_name,
                table,
                &["event_id", "memory_id", "role"],
                report,
                state,
            ),
            "node_embeddings" => {
                Self::merge_keyed_table(tx, table_name, table, &["node_id"], report, state)
            }
            "fsrs_cards" | "memory_states" => {
                Self::merge_keyed_table(tx, table_name, table, &["memory_id"], report, state)
            }
            "deletion_tombstones" => Self::merge_deletion_tombstones(tx, table, report),
            "memory_connections" => Self::merge_keyed_table(
                tx,
                table_name,
                table,
                &["source_id", "target_id"],
                report,
                state,
            ),
            "intentions" | "insights" | "sessions" => {
                Self::merge_keyed_table(tx, table_name, table, &["id"], report, state)
            }
            "fsrs_config" => {
                Self::merge_keyed_table(tx, table_name, table, &["key"], report, state)
            }
            _ => {
                report.tables_skipped += 1;
                Ok(())
            }
        }
    }

    fn merge_knowledge_nodes(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
        state: &mut PortableMergeState,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(id) = Self::portable_text(table, row, "id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_updated = Self::portable_timestamp(table, row, "updated_at");

            // An opaque marker represents an explicit purge. Unlike legacy raw
            // tombstones, it is intentionally permanent: no timestamp from a
            // later archive can resurrect the same stable id.
            let rejected_by_opaque_tombstone =
                Self::has_opaque_tombstone(tx, "knowledge_nodes", id)?;
            let rejected_by_legacy_tombstone =
                Self::tombstone_timestamp(tx, "knowledge_nodes", id)?.is_some_and(|deleted_at| {
                    incoming_updated.is_some_and(|updated| deleted_at >= updated)
                });
            if rejected_by_opaque_tombstone || rejected_by_legacy_tombstone {
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }

            let existing_updated: Option<String> = tx
                .query_row(
                    "SELECT updated_at FROM knowledge_nodes WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()?;

            if let (Some(existing), Some(incoming)) = (
                existing_updated
                    .as_deref()
                    .and_then(Self::parse_rfc3339_opt),
                incoming_updated,
            ) && existing > incoming
            {
                state.locally_newer_nodes.insert(id.to_string());
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }

            let affected = Self::insert_or_replace_row(tx, "knowledge_nodes", table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn merge_sync_tombstones(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(table_name) = Self::portable_text(table, row, "table_name") else {
                report.rows_skipped += 1;
                continue;
            };
            let Some(row_id) = Self::portable_text(table, row, "row_id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_deleted_at = Self::portable_timestamp(table, row, "deleted_at");
            let existing_tombstone: Option<String> = tx
                .query_row(
                    "SELECT deleted_at FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                    params![table_name, row_id],
                    |row| row.get(0),
                )
                .optional()?;
            let existing_deleted_at = existing_tombstone
                .as_ref()
                .and_then(|deleted_at| Self::parse_rfc3339_opt(deleted_at));
            let incoming_wins = match (existing_deleted_at, incoming_deleted_at) {
                (Some(existing), Some(incoming)) => incoming >= existing,
                (Some(_), None) => false,
                (None, _) => true,
            };

            let effective_deleted_at = if incoming_wins {
                let affected = Self::insert_or_replace_row(tx, "sync_tombstones", table, row)?;
                report.rows_imported += 1;
                if affected == MergeWrite::Inserted {
                    report.rows_inserted += 1;
                } else {
                    report.rows_updated += 1;
                }
                incoming_deleted_at
            } else {
                report.rows_skipped += 1;
                existing_deleted_at
            };

            if table_name == "knowledge_nodes" {
                let Some(target_id) = Self::resolve_tombstone_memory_id(tx, row_id)? else {
                    // The target may arrive in a later archive, but this merge
                    // has no raw identifier to delete. The opaque tombstone is
                    // still persisted; a future node merge consults it by
                    // deriving the same marker from the candidate's local id.
                    continue;
                };
                let local_updated: Option<String> = tx
                    .query_row(
                        "SELECT updated_at FROM knowledge_nodes WHERE id = ?1",
                        params![target_id],
                        |row| row.get(0),
                    )
                    .optional()?;
                let should_delete = match (
                    local_updated.as_deref().and_then(Self::parse_rfc3339_opt),
                    effective_deleted_at,
                ) {
                    (Some(local), Some(deleted)) => {
                        row_id.starts_with("opaque:") || deleted >= local
                    }
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if should_delete {
                    // The remote marker has already been persisted above.
                    // Reuse the local coordinator without rewriting its
                    // timestamp so merge performs the full evidence cleanup
                    // atomically with the portable import.
                    if Self::purge_node_in_transaction(
                        tx,
                        &target_id,
                        effective_deleted_at.unwrap_or_else(Utc::now),
                        false,
                    )?
                    .is_some()
                    {
                        report.rows_deleted += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn merge_deletion_tombstones(
        tx: &rusqlite::Transaction<'_>,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            let Some(memory_id) = Self::portable_text(table, row, "memory_id") else {
                report.rows_skipped += 1;
                continue;
            };
            let incoming_deleted_at = Self::portable_timestamp(table, row, "deleted_at");
            let existing_deleted_at: Option<String> = tx
                .query_row(
                    "SELECT deleted_at FROM deletion_tombstones WHERE memory_id = ?1",
                    params![memory_id],
                    |row| row.get(0),
                )
                .optional()?;

            if let (Some(existing), Some(incoming)) = (
                existing_deleted_at
                    .as_deref()
                    .and_then(Self::parse_rfc3339_opt),
                incoming_deleted_at,
            ) && existing > incoming
            {
                report.rows_skipped += 1;
                continue;
            }

            let affected = Self::insert_or_replace_row(tx, "deletion_tombstones", table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn merge_keyed_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        key_columns: &[&str],
        report: &mut PortableImportReport,
        state: &PortableMergeState,
    ) -> Result<()> {
        for row in &table.rows {
            if !Self::parent_rows_exist(tx, table_name, table, row)? {
                report.rows_skipped += 1;
                continue;
            }
            if key_columns
                .iter()
                .any(|column| Self::portable_value(table, row, column).is_none())
            {
                report.rows_skipped += 1;
                continue;
            }
            if Self::row_references_locally_newer_node(table_name, table, row, state) {
                report.conflicts_kept_local += 1;
                report.rows_skipped += 1;
                continue;
            }
            let affected = Self::insert_or_replace_row(tx, table_name, table, row)?;
            report.rows_imported += 1;
            if affected == MergeWrite::Inserted {
                report.rows_inserted += 1;
            } else {
                report.rows_updated += 1;
            }
        }
        Ok(())
    }

    fn row_references_locally_newer_node(
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
        state: &PortableMergeState,
    ) -> bool {
        match table_name {
            "node_embeddings" => Self::portable_text(table, row, "node_id")
                .is_some_and(|id| state.locally_newer_nodes.contains(id)),
            "fsrs_cards" | "memory_states" => Self::portable_text(table, row, "memory_id")
                .is_some_and(|id| state.locally_newer_nodes.contains(id)),
            "memory_connections" => {
                Self::portable_text(table, row, "source_id")
                    .is_some_and(|id| state.locally_newer_nodes.contains(id))
                    || Self::portable_text(table, row, "target_id")
                        .is_some_and(|id| state.locally_newer_nodes.contains(id))
            }
            _ => false,
        }
    }

    fn merge_append_only_table(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        report: &mut PortableImportReport,
    ) -> Result<()> {
        for row in &table.rows {
            if !Self::parent_rows_exist(tx, table_name, table, row)? {
                report.rows_skipped += 1;
                continue;
            }

            let insert_columns: Vec<String> = table
                .columns
                .iter()
                .filter(|column| column.as_str() != "id")
                .cloned()
                .collect();
            if insert_columns.is_empty() {
                report.rows_skipped += 1;
                continue;
            }

            let values = Self::row_values_for_columns(table, row, &insert_columns)?;
            if Self::row_exists_by_values(tx, table_name, &insert_columns, &values)? {
                report.rows_skipped += 1;
                continue;
            }

            Self::insert_row_with_columns(tx, table_name, &insert_columns, values)?;
            report.rows_imported += 1;
            report.rows_inserted += 1;
        }
        Ok(())
    }

    fn parent_rows_exist(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<bool> {
        match table_name {
            "node_embeddings" | "memory_access_log" => Self::portable_text(table, row, "node_id")
                .map(|id| Self::node_exists(tx, id))
                .transpose()
                .map(|v| v.unwrap_or(false)),
            "fsrs_cards" | "memory_states" | "state_transitions" => {
                Self::portable_text(table, row, "memory_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()
                    .map(|v| v.unwrap_or(false))
            }
            "memory_connections" => {
                let source_exists = Self::portable_text(table, row, "source_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                let target_exists = Self::portable_text(table, row, "target_id")
                    .map(|id| Self::node_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(source_exists && target_exists)
            }
            "composition_members" => {
                let event_exists = Self::portable_text(table, row, "event_id")
                    .map(|id| Self::composition_event_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(event_exists)
            }
            "composition_outcomes" => {
                let event_exists = Self::portable_text(table, row, "event_id")
                    .map(|id| Self::composition_event_exists(tx, id))
                    .transpose()?
                    .unwrap_or(false);
                Ok(event_exists)
            }
            _ => Ok(true),
        }
    }

    fn insert_or_replace_row(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<MergeWrite> {
        let key_exists = Self::merge_row_exists(tx, table_name, table, row)?;
        let values = Self::row_values_for_columns(table, row, &table.columns)?;
        Self::upsert_row_with_columns(tx, table_name, &table.columns, values)?;
        Ok(if key_exists {
            MergeWrite::Updated
        } else {
            MergeWrite::Inserted
        })
    }

    fn merge_key_columns(table_name: &str) -> &'static [&'static str] {
        match table_name {
            "knowledge_nodes" | "intentions" | "insights" | "sessions" => &["id"],
            "composition_events" | "composition_outcomes" => &["id"],
            "composition_members" => &["event_id", "memory_id", "role"],
            "node_embeddings" => &["node_id"],
            "fsrs_cards" | "memory_states" | "deletion_tombstones" => &["memory_id"],
            "memory_connections" => &["source_id", "target_id"],
            "fsrs_config" => &["key"],
            "sync_tombstones" => &["table_name", "row_id"],
            _ => &[],
        }
    }

    fn upsert_row_with_columns(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: Vec<Value>,
    ) -> Result<()> {
        let key_columns = Self::merge_key_columns(table_name);
        if key_columns.is_empty() {
            return Self::insert_row_with_columns(tx, table_name, columns, values);
        }

        let quoted_table = Self::quote_ident(table_name);
        let quoted_columns = columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders = std::iter::repeat_n("?", columns.len())
            .collect::<Vec<_>>()
            .join(", ");
        let conflict_target = key_columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let update_columns = columns
            .iter()
            .filter(|column| !key_columns.iter().any(|key| key == &column.as_str()))
            .map(|column| {
                let quoted = Self::quote_ident(column);
                format!("{quoted} = excluded.{quoted}")
            })
            .collect::<Vec<_>>();

        let conflict_action = if update_columns.is_empty() {
            "DO NOTHING".to_string()
        } else {
            format!("DO UPDATE SET {}", update_columns.join(", "))
        };

        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT({}) {}",
            quoted_table, quoted_columns, placeholders, conflict_target, conflict_action
        );
        tx.execute(&sql, params_from_iter(values))?;
        Ok(())
    }

    fn insert_row_with_columns(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: Vec<Value>,
    ) -> Result<()> {
        let quoted_table = Self::quote_ident(table_name);
        let quoted_columns = columns
            .iter()
            .map(|column| Self::quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders = std::iter::repeat_n("?", columns.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT OR REPLACE INTO {} ({}) VALUES ({})",
            quoted_table, quoted_columns, placeholders
        );
        tx.execute(&sql, params_from_iter(values))?;
        Ok(())
    }

    fn merge_row_exists(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        table: &PortableTable,
        row: &[PortableValue],
    ) -> Result<bool> {
        let key_columns = Self::merge_key_columns(table_name);
        if key_columns.is_empty() {
            return Ok(false);
        }
        let mut columns = Vec::new();
        for key in key_columns {
            columns.push((*key).to_string());
        }
        let values = Self::row_values_for_columns(table, row, &columns)?;
        Self::row_exists_by_values(tx, table_name, &columns, &values)
    }

    fn row_exists_by_values(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        columns: &[String],
        values: &[Value],
    ) -> Result<bool> {
        let quoted_table = Self::quote_ident(table_name);
        let where_clause = columns
            .iter()
            .map(|column| format!("{} IS ?", Self::quote_ident(column)))
            .collect::<Vec<_>>()
            .join(" AND ");
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE {}",
            quoted_table, where_clause
        );
        let count: i64 = tx.query_row(&sql, params_from_iter(values.iter()), |row| row.get(0))?;
        Ok(count > 0)
    }

    fn row_values_for_columns(
        table: &PortableTable,
        row: &[PortableValue],
        columns: &[String],
    ) -> Result<Vec<Value>> {
        columns
            .iter()
            .map(|column| {
                Self::portable_value(table, row, column)
                    .ok_or_else(|| {
                        StorageError::Init(format!(
                            "Portable archive row in table '{}' is missing column '{}'",
                            table.name, column
                        ))
                    })?
                    .to_sql_value()
                    .map_err(|e| StorageError::Init(format!("Invalid portable value: {}", e)))
            })
            .collect()
    }

    fn portable_value<'a>(
        table: &PortableTable,
        row: &'a [PortableValue],
        column: &str,
    ) -> Option<&'a PortableValue> {
        table
            .columns
            .iter()
            .position(|name| name == column)
            .and_then(|idx| row.get(idx))
    }

    fn portable_text<'a>(
        table: &PortableTable,
        row: &'a [PortableValue],
        column: &str,
    ) -> Option<&'a str> {
        match Self::portable_value(table, row, column) {
            Some(PortableValue::Text(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    fn portable_timestamp(
        table: &PortableTable,
        row: &[PortableValue],
        column: &str,
    ) -> Option<DateTime<Utc>> {
        Self::portable_text(table, row, column).and_then(Self::parse_rfc3339_opt)
    }

    fn parse_rfc3339_opt(value: &str) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(value)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    }

    fn tombstone_timestamp(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        row_id: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        let opaque_marker = if table_name == "knowledge_nodes" {
            Some(Self::opaque_tombstone_marker(row_id))
        } else {
            None
        };
        let deleted_at: Option<String> = tx
            .query_row(
                "SELECT deleted_at FROM sync_tombstones
                 WHERE table_name = ?1 AND (row_id = ?2 OR row_id = ?3)
                 ORDER BY deleted_at DESC LIMIT 1",
                params![table_name, row_id, opaque_marker],
                |row| row.get(0),
            )
            .optional()?;
        Ok(deleted_at.as_deref().and_then(Self::parse_rfc3339_opt))
    }

    fn has_opaque_tombstone(
        tx: &rusqlite::Transaction<'_>,
        table_name: &str,
        row_id: &str,
    ) -> Result<bool> {
        if table_name != "knowledge_nodes" {
            return Ok(false);
        }
        let marker = Self::opaque_tombstone_marker(row_id);
        let exists: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM sync_tombstones WHERE table_name = ?1 AND row_id = ?2",
                params![table_name, marker],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    pub(super) fn current_schema_version(conn: &Connection) -> Result<u32> {
        let version: i64 = conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )?;
        Ok(version as u32)
    }

    fn ensure_portable_import_target_empty(conn: &Connection) -> Result<()> {
        for table_name in PORTABLE_USER_DATA_TABLES {
            if Self::table_exists(conn, table_name)? {
                let count = Self::table_row_count(conn, table_name)?;
                if count > 0 {
                    return Err(StorageError::Init(format!(
                        "Portable import requires an empty target database; table '{}' has {} rows",
                        table_name, count
                    )));
                }
            }
        }
        Ok(())
    }

    pub(super) fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
        let exists: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1",
            params![table_name],
            |row| row.get(0),
        )?;
        Ok(exists > 0)
    }

    pub(super) fn table_row_count(conn: &Connection, table_name: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {}", Self::quote_ident(table_name));
        Ok(conn.query_row(&sql, [], |row| row.get(0))?)
    }

    pub(super) fn table_columns(conn: &Connection, table_name: &str) -> Result<Vec<String>> {
        let sql = format!("PRAGMA table_info({})", Self::quote_ident(table_name));
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;

        let mut columns = Vec::new();
        for row in rows {
            columns.push(row?);
        }
        Ok(columns)
    }

    fn portable_value_from_ref(value: ValueRef<'_>) -> rusqlite::Result<PortableValue> {
        Ok(match value {
            ValueRef::Null => PortableValue::Null,
            ValueRef::Integer(value) => PortableValue::Integer(value),
            ValueRef::Real(value) => PortableValue::Real(value),
            ValueRef::Text(value) => PortableValue::Text(
                std::str::from_utf8(value)
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(e))
                    })?
                    .to_string(),
            ),
            ValueRef::Blob(value) => PortableValue::Blob(encode_hex(value)),
        })
    }

    fn quote_ident(identifier: &str) -> String {
        format!("\"{}\"", identifier.replace('"', "\"\""))
    }
}
