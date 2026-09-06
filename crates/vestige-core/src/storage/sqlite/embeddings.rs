//! Embedding profiles and the vector index: profile manifests and migration
//! checkpoints, index build from `embedding_profile_vectors`, journal-driven
//! refresh and reconcile against peer processes, and embedding (re)generation.

use super::*;

impl SqliteMemoryStore {
    #[cfg(feature = "vector-search")]
    pub(super) fn vector_search_enabled_by_cpu() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let has_required_features = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let has_required_features = true;

        has_required_features && !Self::vector_search_disable_requested()
    }

    /// The runtime opt-out, read the same way by the gate and by the
    /// "why is it off" report so the two can never disagree.
    #[cfg(feature = "vector-search")]
    fn vector_search_disable_requested() -> bool {
        #[cfg(test)]
        if VECTOR_SEARCH_DISABLED_FOR_TEST.with(|cell| cell.get()) {
            return true;
        }
        std::env::var_os(VESTIGE_DISABLE_VECTOR_SEARCH)
            .is_some_and(|value| env_value_disables_vector_search(&value))
    }

    #[cfg(feature = "vector-search")]
    pub(super) fn vector_search_unavailable_reason() -> Option<&'static str> {
        if Self::vector_search_disable_requested() {
            return Some("disabled by VESTIGE_DISABLE_VECTOR_SEARCH");
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return Some("unsupported CPU: AVX2 required");
            }
            if !std::arch::is_x86_feature_detected!("fma") {
                return Some("unsupported CPU: FMA required");
            }
        }

        None
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn vector_search_available(&self) -> bool {
        self.vector_index.is_some()
    }

    /// Load existing embeddings into vector index
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn load_embeddings_into_index(&self) -> Result<()> {
        let Some(index) = self.vector_index.as_ref() else {
            return Ok(());
        };
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        drop(reader);
        let (rebuilt, journal_seq) = self.build_embedding_profile_index(&active_profile_id)?;
        {
            let mut index = index
                .lock()
                .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
            *index = rebuilt;
        }
        self.reset_vector_index_watermark(journal_seq);
        Ok(())
    }

    /// Build an isolated exact-dimension HNSW index without touching the live
    /// index. Activation uses this preflight so an invalid destination can
    /// never become the visible database pointer.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn build_embedding_profile_index(&self, profile_id: &str) -> Result<(VectorIndex, i64)> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        // One read snapshot for the rows AND the journal head, so the watermark
        // handed back describes exactly the rows that went into this index (#181).
        let snapshot = begin_read_snapshot(&reader)?;
        let profile_dimension: usize = snapshot
            .query_row(
                "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                params![profile_id],
                |row| row.get::<_, i64>(0),
            )?
            .try_into()
            .map_err(|_| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' has an invalid embedding dimension",
                    profile_id
                ))
            })?;
        let mut stmt = snapshot.prepare(
            "SELECT node_id, embedding, model
             FROM embedding_profile_vectors
             WHERE profile_id = ?1",
        )?;

        // Never drop rows silently: a row this rebuild cannot read is a memory
        // that stays invisible to semantic search until it is re-embedded, and
        // the operator must be able to see that in the log.
        let mut unreadable_rows = 0_usize;
        let embeddings: Vec<(String, Vec<u8>, String)> = stmt
            .query_map(params![profile_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| match r {
                Ok(row) => Some(row),
                Err(error) => {
                    unreadable_rows += 1;
                    tracing::warn!(
                        %error,
                        profile_id,
                        "Skipping an unreadable embedding_profile_vectors row during vector index rebuild"
                    );
                    None
                }
            })
            .collect();
        if unreadable_rows > 0 {
            tracing::warn!(
                unreadable_rows,
                profile_id,
                "Vector index rebuild skipped rows; those memories stay keyword-searchable only until re-embedded"
            );
        }

        drop(stmt);
        let journal_seq: i64 = snapshot.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
            [],
            |row| row.get(0),
        )?;
        drop(snapshot);
        drop(reader);

        // An index is a profile-scoped structure. In particular, never
        // Matryoshka-truncate a 1024/native profile into the legacy 256d index.
        let mut index = VectorIndex::with_config(VectorIndexConfig {
            dimensions: profile_dimension,
            ..VectorIndexConfig::default()
        })
        .map_err(|e| {
            StorageError::Init(format!("Failed to rebuild vector index before load: {}", e))
        })?;

        for (node_id, embedding_bytes, _model_name) in embeddings {
            let embedding = Embedding::from_bytes(&embedding_bytes).ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' contains unreadable vector '{}'",
                    profile_id, node_id
                ))
            })?;
            if embedding.dimensions != profile_dimension {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' declares {} dimensions but vector '{}' has {}",
                    profile_id, profile_dimension, node_id, embedding.dimensions
                )));
            }
            index.add(&node_id, &embedding.vector).map_err(|error| {
                StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' failed to build index for vector '{}': {}",
                    profile_id, node_id, error
                ))
            })?;
        }
        Ok((index, journal_seq))
    }

    /// Get the embedding vector for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_node_embedding(&self, node_id: &str) -> Result<Option<Vec<f32>>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        let mut stmt = reader.prepare(
            "SELECT embedding FROM embedding_profile_vectors
             WHERE profile_id = ?1 AND node_id = ?2",
        )?;

        let embedding_row: Option<Vec<u8>> = stmt
            .query_row(params![&active_profile_id, node_id], |row| row.get(0))
            .optional()?;

        // Direct table writes remain a supported test and migration fixture.
        // Only the legacy profile may consult that compatibility mirror; every
        // non-legacy profile is strictly isolated from it.
        let embedding_row =
            if embedding_row.is_none() && active_profile_id == LEGACY_EMBEDDING_PROFILE_ID {
                reader
                    .query_row(
                        "SELECT embedding FROM node_embeddings WHERE node_id = ?1",
                        params![node_id],
                        |row| row.get(0),
                    )
                    .optional()?
            } else {
                embedding_row
            };

        Ok(embedding_row
            .and_then(|bytes| Embedding::from_bytes(&bytes).map(|embedding| embedding.vector)))
    }

    /// Get all embedding vectors for duplicate detection
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let active_profile_id = Self::active_profile_id_from_conn(&reader)?
            .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
        let mut stmt = reader.prepare(
            "SELECT node_id, embedding FROM embedding_profile_vectors WHERE profile_id = ?1",
        )?;

        let mut unreadable_rows = 0_usize;
        let mut undecodable_rows = 0_usize;
        let mut results: Vec<(String, Vec<f32>)> = stmt
            .query_map(params![&active_profile_id], |row| {
                let node_id: String = row.get(0)?;
                let embedding_bytes: Vec<u8> = row.get(1)?;
                Ok((node_id, embedding_bytes))
            })?
            .filter_map(|r| match r {
                Ok(row) => Some(row),
                Err(error) => {
                    unreadable_rows += 1;
                    tracing::warn!(
                        %error,
                        profile_id = %active_profile_id,
                        "Skipping an unreadable embedding row while loading vectors"
                    );
                    None
                }
            })
            .filter_map(|(id, bytes)| match Embedding::from_bytes(&bytes) {
                Some(embedding) => Some((id, embedding.vector)),
                None => {
                    undecodable_rows += 1;
                    tracing::warn!(
                        node_id = %id,
                        profile_id = %active_profile_id,
                        "Skipping an undecodable embedding blob while loading vectors"
                    );
                    None
                }
            })
            .collect();
        // Keep direct writes to the historic table readable for the legacy
        // profile only. The anti-join avoids duplicate node ids after V20's
        // one-time copy, while non-legacy profiles never see this mirror.
        //
        // This mirror is the branch an unmigrated store actually takes, so it
        // gets the same "never drop a row silently" treatment as the profile
        // table above: a row skipped here is a memory that stays invisible to
        // semantic search, and the operator has to be able to see it.
        if active_profile_id == LEGACY_EMBEDDING_PROFILE_ID {
            drop(stmt);
            let mut legacy_stmt = reader.prepare(
                "SELECT ne.node_id, ne.embedding
                 FROM node_embeddings ne
                 WHERE NOT EXISTS (
                     SELECT 1 FROM embedding_profile_vectors pv
                     WHERE pv.profile_id = ?1 AND pv.node_id = ne.node_id
                 )",
            )?;
            results.extend(
                legacy_stmt
                    .query_map(params![LEGACY_EMBEDDING_PROFILE_ID], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
                    })?
                    .filter_map(|row| match row {
                        Ok(row) => Some(row),
                        Err(error) => {
                            unreadable_rows += 1;
                            tracing::warn!(
                                %error,
                                profile_id = %active_profile_id,
                                "Skipping an unreadable node_embeddings row while loading legacy vectors"
                            );
                            None
                        }
                    })
                    .filter_map(|(id, bytes)| match Embedding::from_bytes(&bytes) {
                        Some(embedding) => Some((id, embedding.vector)),
                        None => {
                            undecodable_rows += 1;
                            tracing::warn!(
                                node_id = %id,
                                profile_id = %active_profile_id,
                                "Skipping an undecodable node_embeddings blob while loading legacy vectors"
                            );
                            None
                        }
                    }),
            );
        }

        // Summarised after the legacy mirror so one line covers both sources.
        if unreadable_rows + undecodable_rows > 0 {
            tracing::warn!(
                unreadable_rows,
                undecodable_rows,
                profile_id = %active_profile_id,
                "Vector load skipped rows; those memories stay keyword-searchable only until re-embedded"
            );
        }

        Ok(results)
    }

    /// Fallback for builds without local embeddings/vector search.
    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn get_node_embedding(&self, _node_id: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Generate embedding for a node
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn generate_embedding_for_node(&self, node_id: &str, content: &str) -> Result<()> {
        if !self.active_embedding_runtime_ready()? {
            return Ok(());
        }

        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let encoded_content = manifest
            .profile
            .encode_document(content)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;

        let (embedding_bytes, embedding_dimensions, model_name, vector) =
            if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
                let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                    StorageError::Init(format!("Create local embedding runtime: {error}"))
                })?;
                let vector = runtime
                    .block_on(embedder.embed_document(content))
                    .map_err(|error| StorageError::Init(format!("Embedding failed: {error}")))?;
                let bytes = vector
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                (bytes, vector.len(), active.profile_id.to_string(), vector)
            } else {
                let embedding = self
                    .embedding_service
                    .embed(&encoded_content)
                    .map_err(|e| StorageError::Init(format!("Embedding failed: {e}")))?;
                (
                    embedding.to_bytes(),
                    embedding.dimensions,
                    self.embedding_service.model_name().to_string(),
                    embedding.vector,
                )
            };
        if embedding_dimensions != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id, manifest.profile.embedding_dimension, embedding_dimensions
            )));
        }

        self.persist_node_embedding(
            node_id,
            &embedding_bytes,
            embedding_dimensions,
            &model_name,
            &vector,
            active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID,
        )
    }

    /// Write one node's vector everywhere semantic search reads it: the
    /// active profile's vector table (and the historic `node_embeddings`
    /// mirror while the legacy profile is active), the node's
    /// `has_embedding` flag, and the in-memory vector index. Shared by the
    /// embedder path and by [`MemoryStoreSend::insert`] for caller-supplied
    /// vectors, so no path can accept an embedding and drop it.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn persist_node_embedding(
        &self,
        node_id: &str,
        embedding_bytes: &[u8],
        embedding_dimensions: usize,
        model_name: &str,
        vector: &[f32],
        mirror_to_legacy_table: bool,
    ) -> Result<()> {
        let now = Utc::now();

        // One transaction for the three rows, with the journal head read inside
        // it. We hold the write lock, so no peer can commit between our INSERT
        // and that read: `journal_head` is the seq the trigger just appended.
        let journal_head: i64 = {
            let writer = self
                .writer
                .lock()
                .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
            let tx = Self::begin_write_transaction(&writer, "persist_node_embedding")?;
            if mirror_to_legacy_table {
                tx.execute(
                    "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimensions, model, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        node_id,
                        embedding_bytes,
                        embedding_dimensions as i32,
                        model_name,
                        now.to_rfc3339(),
                    ],
                )?;
            }

            let active_profile_id = Self::active_profile_id_from_conn(&tx)?
                .unwrap_or_else(|| LEGACY_EMBEDDING_PROFILE_ID.to_string());
            tx.execute(
                "INSERT OR REPLACE INTO embedding_profile_vectors
                    (profile_id, node_id, embedding, dimensions, model, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    active_profile_id,
                    node_id,
                    embedding_bytes,
                    embedding_dimensions as i32,
                    model_name,
                    now.to_rfc3339(),
                ],
            )?;

            tx.execute(
                "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = ?2 WHERE id = ?1",
                params![node_id, model_name],
            )?;
            let head: i64 = tx.query_row(
                "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
                [],
                |row| row.get(0),
            )?;
            tx.commit()?;
            head
        };

        if let Some(index) = self.vector_index.as_ref() {
            let mut index = index
                .lock()
                .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;
            index
                .add(node_id, vector)
                .map_err(|e| StorageError::Init(format!("Vector index add failed: {}", e)))?;
        }

        // Our own write bumps the reader's data_version exactly like a peer's
        // would, but the vector is already in the index. If ours is the only
        // journal row since the last refresh, absorb it now so the next search
        // does not re-add it. Anything else in between (a peer's row, an unknown
        // watermark) is left for the refresh, which re-adds ours harmlessly.
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq + 1 == journal_head
        {
            watermark.journal_seq = journal_head;
        }

        Ok(())
    }

    /// Index a caller-supplied embedding under the active profile, or refuse
    /// it loudly. Used by [`MemoryStoreSend::insert`]: a record that carries
    /// a vector is either searchable when the call returns or the call fails.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn index_supplied_embedding(
        &self,
        node_id: &str,
        vector: &[f32],
        model_name: Option<&str>,
    ) -> crate::storage::memory_store::MemoryStoreResult<()> {
        use crate::storage::memory_store::MemoryStoreError;
        let active = self
            .active_embedding_profile()
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?
            .ok_or_else(|| {
                MemoryStoreError::InvalidInput(
                    "record carries an embedding but the store has no active embedding profile to index it under"
                        .to_string(),
                )
            })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?
            .ok_or_else(|| {
                MemoryStoreError::Backend(format!(
                    "active embedding profile '{}' has no manifest",
                    active.profile_id
                ))
            })?;
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(MemoryStoreError::InvalidInput(format!(
                "embedding length {} != active profile '{}' dimension {}",
                vector.len(),
                active.profile_id,
                manifest.profile.embedding_dimension
            )));
        }
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        let model = model_name
            .map(str::to_string)
            .unwrap_or_else(|| active.profile_id.to_string());
        self.persist_node_embedding(
            node_id,
            &bytes,
            vector.len(),
            &model,
            vector,
            active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID,
        )
        .map_err(|e| MemoryStoreError::Backend(e.to_string()))
    }

    /// Read the active profile pointer from a caller-held connection. The
    /// pointer has one row and is changed in the same SQLite transaction as
    /// profile status, so readers can never observe a half-switch.
    pub(super) fn active_profile_id_from_conn(conn: &Connection) -> Result<Option<String>> {
        conn.query_row(
            "SELECT active_profile_id FROM embedding_profile_state WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(StorageError::from)
    }

    fn profile_state_text(state: EmbeddingProfileState) -> Result<String> {
        serde_json::to_value(state)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(
                    "profile state must serialize to a string".to_string(),
                )
            })
    }

    fn migration_state_text(state: EmbeddingMigrationState) -> Result<String> {
        serde_json::to_value(state)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                StorageError::InvalidEmbeddingProfile(
                    "migration state must serialize to a string".to_string(),
                )
            })
    }

    fn parse_rfc3339(value: String, field: &str) -> Result<DateTime<Utc>> {
        // V20's SQL bootstrap uses SQLite `datetime('now')` while subsequent
        // Rust writes use RFC3339. Reuse the store's tolerant parser so either
        // durable timestamp representation round-trips through profile APIs.
        Self::parse_timestamp(&value, field).map_err(StorageError::from)
    }

    pub(super) fn ensure_legacy_embedding_profile_manifest(&self) -> Result<()> {
        // Normal opens must be completely idempotent. In particular, never
        // rewrite a preserved legacy profile to Active after an explicit Qwen
        // activation has moved the durable pointer elsewhere.
        let existing_manifest = {
            let reader = self
                .reader
                .lock()
                .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
            reader
                .query_row(
                    "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                    params![LEGACY_EMBEDDING_PROFILE_ID],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
        };
        if existing_manifest
            .as_deref()
            .is_some_and(|json| serde_json::from_str::<EmbeddingProfileManifest>(json).is_ok())
        {
            return Ok(());
        }

        let mut manifest = EmbeddingProfileManifest::not_installed(
            BuiltinEmbeddingProfile::NomicLegacyRaw256.profile(),
        )
        .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let active_is_legacy = self
            .active_embedding_profile()?
            .is_none_or(|active| active.profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID);
        manifest.state = if active_is_legacy {
            EmbeddingProfileState::Active
        } else {
            // This only upgrades V20's bootstrap '{}' placeholder. Existing
            // valid manifests already returned above, so no lifecycle receipt
            // or user choice is ever overwritten on reopen.
            EmbeddingProfileState::Ready
        };
        self.save_embedding_profile_manifest(&manifest)
    }

    /// Persist a full profile contract and its lifecycle receipt. This is a
    /// metadata operation only: saving an Installed manifest never downloads,
    /// migrates, or activates a model.
    pub fn save_embedding_profile_manifest(
        &self,
        manifest: &EmbeddingProfileManifest,
    ) -> Result<()> {
        manifest
            .validate()
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let profile = &manifest.profile;
        let manifest_json = serde_json::to_string(manifest)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let artifact_hashes = serde_json::to_string(&profile.verified_model_artifact_hashes)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let runtime = serde_json::to_string(&manifest.runtime)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let verification = serde_json::to_string(&manifest.verification)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let evaluation = serde_json::to_string(&manifest.evaluation)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let failure = serde_json::to_string(&manifest.failure)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let state = Self::profile_state_text(manifest.state)?;
        let now = Utc::now();
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_embedding_profile_manifest")?;
        let existing: Option<(String, i64)> = tx
            .query_row(
                "SELECT pm.manifest_json, pm.vector_count
                 FROM embedding_profile_manifests pm WHERE pm.profile_id = ?1",
                params![profile.profile_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((existing_json, vector_count)) = existing {
            // The V20 bootstrap row is deliberately replaced once with the
            // canonical legacy manifest. After that, a profile ID is an
            // immutable vector-space identity, not a mutable model selector.
            if let Ok(existing_manifest) =
                serde_json::from_str::<EmbeddingProfileManifest>(&existing_json)
                && vector_count > 0
                && existing_manifest.profile != manifest.profile
            {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' already owns {} vectors; changing its encoding contract requires a new profile ID",
                    profile.profile_id, vector_count
                )));
            }
        }
        if manifest.state == EmbeddingProfileState::Active {
            let pointer = Self::active_profile_id_from_conn(&tx)?;
            if pointer.as_deref() != Some(profile.profile_id.as_str()) {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "profile '{}' may become active only through activate_embedding_profile",
                    profile.profile_id
                )));
            }
        }
        tx.execute(
            "INSERT INTO embedding_profiles (
                profile_id, model_id, immutable_model_revision, verified_model_artifact_hashes,
                runtime_backend, embedding_dimension, normalization_method,
                document_encoding_template, query_encoding_template, maximum_token_limit,
                chunking_strategy, status, installed_at, last_verified_at, runtime_metadata,
                verification, evaluation, failure, created_at, updated_at
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
             ) ON CONFLICT(profile_id) DO UPDATE SET
                model_id = excluded.model_id,
                immutable_model_revision = excluded.immutable_model_revision,
                verified_model_artifact_hashes = excluded.verified_model_artifact_hashes,
                runtime_backend = excluded.runtime_backend,
                embedding_dimension = excluded.embedding_dimension,
                normalization_method = excluded.normalization_method,
                document_encoding_template = excluded.document_encoding_template,
                query_encoding_template = excluded.query_encoding_template,
                maximum_token_limit = excluded.maximum_token_limit,
                chunking_strategy = excluded.chunking_strategy,
                status = excluded.status,
                installed_at = excluded.installed_at,
                last_verified_at = excluded.last_verified_at,
                runtime_metadata = excluded.runtime_metadata,
                verification = excluded.verification,
                evaluation = excluded.evaluation,
                failure = excluded.failure,
                updated_at = excluded.updated_at",
            params![
                profile.profile_id.as_str(),
                profile.model_id,
                profile.immutable_model_revision,
                artifact_hashes,
                serde_json::to_string(&profile.runtime_backend).unwrap_or_default(),
                profile.embedding_dimension as i64,
                serde_json::to_string(&profile.normalization_method).unwrap_or_default(),
                serde_json::to_string(&profile.document_encoding_template).unwrap_or_default(),
                serde_json::to_string(&profile.query_encoding_template).unwrap_or_default(),
                profile.maximum_token_limit as i64,
                serde_json::to_string(&profile.chunking_strategy).unwrap_or_default(),
                state,
                manifest.installed_at.map(|value| value.to_rfc3339()),
                manifest.last_verified_at.map(|value| value.to_rfc3339()),
                runtime,
                verification,
                evaluation,
                failure,
                profile.created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_manifests (
                profile_id, manifest_json, manifest_hash, vector_count, index_member_count, index_integrity_hash, updated_at
             ) VALUES (
                ?1, ?2, ?3,
                (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                0, NULL, ?4
             ) ON CONFLICT(profile_id) DO UPDATE SET
                manifest_json = excluded.manifest_json,
                manifest_hash = excluded.manifest_hash,
                vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = excluded.profile_id),
                updated_at = excluded.updated_at",
            params![profile.profile_id.as_str(), manifest_json, manifest.manifest_hash(), now.to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Read a profile's complete persisted contract and lifecycle state.
    pub fn embedding_profile_manifest(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<EmbeddingProfileManifest>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let manifest: Option<String> = reader
            .query_row(
                "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        manifest
            .map(|json| {
                serde_json::from_str(&json)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))
            })
            .transpose()
    }

    /// List known profiles without triggering any install/runtime work.
    pub fn list_embedding_profile_manifests(&self) -> Result<Vec<EmbeddingProfileManifest>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut statement = reader
            .prepare("SELECT manifest_json FROM embedding_profile_manifests ORDER BY profile_id")?;
        statement
            .query_map([], |row| row.get::<_, String>(0))?
            .map(|row| {
                let json = row?;
                serde_json::from_str(&json).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(error))
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Read the active semantic-retrieval profile pointer.
    pub fn active_embedding_profile(&self) -> Result<Option<ActiveEmbeddingProfile>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(String, Option<String>, String)> = reader
            .query_row(
                "SELECT active_profile_id, previous_profile_id, activated_at
                 FROM embedding_profile_state WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        row.map(|(profile_id, previous_profile_id, activated_at)| {
            Ok(ActiveEmbeddingProfile {
                profile_id: EmbeddingProfileId::new(profile_id)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
                previous_profile_id: previous_profile_id
                    .map(EmbeddingProfileId::new)
                    .transpose()
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
                activated_at: Self::parse_rfc3339(activated_at, "profile activation timestamp")?,
            })
        })
        .transpose()
    }

    /// Change only the active-profile pointer after the caller has explicitly
    /// installed, evaluated, migrated, and validated the destination. The
    /// pointer and both status updates are one SQLite transaction; no vector
    /// rows are copied, removed, or re-embedded during activation.
    pub fn activate_embedding_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<ActiveEmbeddingProfile> {
        let target_state = Self::profile_state_text(EmbeddingProfileState::Ready)?;
        let active_state = Self::profile_state_text(EmbeddingProfileState::Active)?;
        let now = Utc::now();
        // Prebuild outside the write transaction. A malformed vector, mixed
        // dimensions, or unbuildable index fails here while the old pointer is
        // still live. Once the index lock is acquired, searches block until the
        // pointer transaction and in-memory index swap have both completed.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let rebuilt_index = if self.vector_index.is_some() {
            Some(self.build_embedding_profile_index(profile_id.as_str())?)
        } else {
            None
        };
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        let mut live_index = match self.vector_index.as_ref() {
            Some(index) => Some(
                index
                    .lock()
                    .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?,
            ),
            None => None,
        };
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "activate_embedding_profile")?;
        let current: Option<String> = tx
            .query_row(
                "SELECT active_profile_id FROM embedding_profile_state WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let stored_state: Option<String> = tx
            .query_row(
                "SELECT status FROM embedding_profiles WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        let Some(stored_state) = stored_state else {
            return Err(StorageError::NotFound(profile_id.to_string()));
        };
        let manifest_json: String = tx
            .query_row(
                "SELECT manifest_json FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                StorageError::NotFound(format!("embedding profile manifest {profile_id}"))
            })?;
        let manifest: EmbeddingProfileManifest = serde_json::from_str(&manifest_json)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let legacy_rollback = profile_id.as_str() == LEGACY_EMBEDDING_PROFILE_ID
            && current.is_some()
            && current.as_deref() != Some(profile_id.as_str());
        if stored_state != target_state && stored_state != active_state {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' is '{}' and cannot be activated; only a validated ready profile may change live semantic retrieval",
                profile_id, stored_state
            )));
        }
        if current.as_deref() == Some(profile_id.as_str()) {
            tx.commit()?;
            return Ok(ActiveEmbeddingProfile {
                profile_id: profile_id.clone(),
                previous_profile_id: None,
                activated_at: now,
            });
        }
        if !legacy_rollback
            && (manifest.state != EmbeddingProfileState::Ready
                || manifest.verification.status != VerificationStatus::Verified
                || manifest.verification.verified_artifacts.is_empty()
                || manifest
                    .runtime
                    .as_ref()
                    .is_none_or(|runtime| !runtime.local_only)
                || manifest.evaluation.is_none())
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' must have a ready, locally verified runtime and completed evaluation before activation",
                profile_id
            )));
        }
        let completed_state = Self::migration_state_text(EmbeddingMigrationState::Completed)?;
        let completed_migration: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profile_migrations
             WHERE destination_profile_id = ?1 AND state = ?2",
            params![profile_id.as_str(), completed_state],
            |row| row.get(0),
        )?;
        let integrity: Option<(i64, i64, String)> = tx
            .query_row(
                "SELECT vector_count, index_member_count, manifest_hash
                 FROM embedding_profile_manifests WHERE profile_id = ?1",
                params![profile_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((vector_count, index_member_count, manifest_hash)) = integrity else {
            return Err(StorageError::InvalidEmbeddingProfile(
                "missing profile integrity manifest".to_string(),
            ));
        };
        if !legacy_rollback
            && (completed_migration == 0
                || vector_count != index_member_count
                || manifest_hash != manifest.manifest_hash())
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' lacks a completed migration with a validated matching index manifest",
                profile_id
            )));
        }
        let wrong_dimension_vectors: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profile_vectors
             WHERE profile_id = ?1 AND dimensions != ?2",
            params![
                profile_id.as_str(),
                manifest.profile.embedding_dimension as i64
            ],
            |row| row.get(0),
        )?;
        if wrong_dimension_vectors != 0 {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' has {} vectors incompatible with its declared dimension",
                profile_id, wrong_dimension_vectors
            )));
        }
        if let Some(current_id) = &current {
            // A prior active profile remains validated and rollback-ready. It
            // becomes Ready (not Inactive), so any future activation still
            // passes the exact same validation gate.
            tx.execute(
                "UPDATE embedding_profiles SET status = ?1, updated_at = ?2 WHERE profile_id = ?3",
                params![target_state, now.to_rfc3339(), current_id],
            )?;
        }
        tx.execute(
            "UPDATE embedding_profiles SET status = ?1, updated_at = ?2 WHERE profile_id = ?3",
            params![active_state, now.to_rfc3339(), profile_id.as_str()],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_state (
                singleton, active_profile_id, previous_profile_id, activated_at, updated_at
             ) VALUES (1, ?1, ?2, ?3, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                active_profile_id = excluded.active_profile_id,
                previous_profile_id = excluded.previous_profile_id,
                activated_at = excluded.activated_at,
                updated_at = excluded.updated_at",
            params![profile_id.as_str(), current, now.to_rfc3339()],
        )?;
        tx.commit()?;
        // The index lock blocks semantic search across the committed-pointer /
        // in-memory-index handoff. The replacement was built and fully checked
        // before the pointer was ever visible.
        #[cfg(all(feature = "embeddings", feature = "vector-search"))]
        {
            let swapped_journal_seq =
                if let (Some(live_index), Some((rebuilt_index, journal_seq))) =
                    (live_index.as_deref_mut(), rebuilt_index)
                {
                    *live_index = rebuilt_index;
                    Some(journal_seq)
                } else {
                    None
                };
            // Release the index before touching the watermark: the refresh path
            // never holds both locks at once, so neither may this one.
            drop(live_index);
            if let Some(journal_seq) = swapped_journal_seq {
                self.reset_vector_index_watermark(journal_seq);
            }
        }
        Ok(ActiveEmbeddingProfile {
            profile_id: profile_id.clone(),
            previous_profile_id: current
                .map(EmbeddingProfileId::new)
                .transpose()
                .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?,
            activated_at: now,
        })
    }

    /// Instant rollback is exactly another explicit pointer change; the old
    /// profile's isolated vectors and sidecar remain intact.
    pub fn rollback_embedding_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<ActiveEmbeddingProfile> {
        self.activate_embedding_profile(profile_id)
    }

    /// Store a vector in one profile's private vector space. Dimension and
    /// profile identity are checked at write time, preventing a migration from
    /// accidentally contaminating its destination profile.
    pub fn put_embedding_profile_vector(&self, vector: &EmbeddingProfileVector) -> Result<()> {
        if vector.profile_id.trim().is_empty()
            || vector.node_id.trim().is_empty()
            || vector.dimensions == 0
        {
            return Err(StorageError::InvalidEmbeddingProfile(
                "profile vector requires a profile ID, node ID, and positive dimensions"
                    .to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "put_embedding_profile_vector")?;
        let declared_dimension: Option<i64> = tx
            .query_row(
                "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
                params![&vector.profile_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(declared_dimension) = declared_dimension else {
            return Err(StorageError::NotFound(vector.profile_id.clone()));
        };
        if declared_dimension != i64::from(vector.dimensions) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' declares {} dimensions but attempted vector has {}",
                vector.profile_id, declared_dimension, vector.dimensions
            )));
        }
        tx.execute(
            "INSERT INTO embedding_profile_vectors
                (profile_id, node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(profile_id, node_id) DO UPDATE SET
                embedding = excluded.embedding, dimensions = excluded.dimensions,
                model = excluded.model, created_at = excluded.created_at",
            params![
                &vector.profile_id,
                &vector.node_id,
                &vector.embedding,
                vector.dimensions as i64,
                &vector.model,
                vector.created_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "UPDATE embedding_profile_manifests
             SET vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                 updated_at = ?2
             WHERE profile_id = ?1",
            params![&vector.profile_id, Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Read one profile-scoped vector. This never falls back to another profile.
    pub fn embedding_profile_vector(
        &self,
        profile_id: &EmbeddingProfileId,
        node_id: &str,
    ) -> Result<Option<EmbeddingProfileVector>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<(Vec<u8>, i64, String, String)> = reader
            .query_row(
                "SELECT embedding, dimensions, model, created_at
                 FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
                params![profile_id.as_str(), node_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        row.map(|(embedding, dimensions, model, created_at)| {
            Ok(EmbeddingProfileVector {
                profile_id: profile_id.to_string(),
                node_id: node_id.to_string(),
                embedding,
                dimensions: dimensions.try_into().map_err(|_| {
                    StorageError::InvalidEmbeddingProfile("negative vector dimensions".to_string())
                })?,
                model,
                created_at: Self::parse_rfc3339(created_at, "profile vector timestamp")?,
            })
        })
        .transpose()
    }

    /// Record the validated vector/index membership for a profile. This is the
    /// final integrity receipt required before `activate_embedding_profile` can
    /// move live semantic retrieval to the profile.
    pub fn save_embedding_profile_integrity_manifest(
        &self,
        integrity: &EmbeddingProfileIntegrityManifest,
    ) -> Result<()> {
        let profile_id = EmbeddingProfileId::new(integrity.profile_id.clone())
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let manifest = self
            .embedding_profile_manifest(&profile_id)?
            .ok_or_else(|| StorageError::NotFound(profile_id.to_string()))?;
        if integrity.manifest_hash != manifest.manifest_hash() {
            return Err(StorageError::InvalidEmbeddingProfile(
                "integrity receipt hash does not match the persisted profile manifest".to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let actual_vector_count: i64 = writer.query_row(
            "SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1",
            params![profile_id.as_str()],
            |row| row.get(0),
        )?;
        if integrity.vector_count != actual_vector_count as u64
            || integrity.index_member_count != integrity.vector_count
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "integrity receipt for '{}' does not match stored profile vectors",
                profile_id
            )));
        }
        writer.execute(
            "UPDATE embedding_profile_manifests SET
                manifest_json = manifest_json,
                manifest_hash = ?2,
                vector_count = ?3,
                index_member_count = ?4,
                index_integrity_hash = ?5,
                updated_at = ?6
             WHERE profile_id = ?1",
            params![
                profile_id.as_str(),
                &integrity.manifest_hash,
                integrity.vector_count as i64,
                integrity.index_member_count as i64,
                &integrity.index_integrity_hash,
                integrity.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Persist (or resume) a migration checkpoint. The active-profile pointer
    /// is deliberately untouched; migration is not activation.
    pub fn save_profile_migration_checkpoint(
        &self,
        checkpoint: &ProfileMigrationCheckpoint,
    ) -> Result<()> {
        if checkpoint.source_profile_id == checkpoint.destination_profile_id {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration source and destination profiles must differ".to_string(),
            ));
        }
        if checkpoint.completed_memories > checkpoint.total_memories {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration completed memories cannot exceed total memories".to_string(),
            ));
        }
        let state = Self::migration_state_text(checkpoint.state)?;
        let failed_memory_ids = serde_json::to_string(&checkpoint.failed_memory_ids)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(&writer, "save_profile_migration_checkpoint")?;
        let profiles: i64 = tx.query_row(
            "SELECT COUNT(*) FROM embedding_profiles WHERE profile_id IN (?1, ?2)",
            params![
                checkpoint.source_profile_id.as_str(),
                checkpoint.destination_profile_id.as_str()
            ],
            |row| row.get(0),
        )?;
        if profiles != 2 {
            return Err(StorageError::NotFound(
                "migration source or destination embedding profile".to_string(),
            ));
        }
        tx.execute(
            "INSERT INTO embedding_profile_migrations (
                migration_id, source_profile_id, destination_profile_id, state,
                total_memories, completed_memories, failed_memory_ids, last_memory_id,
                started_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(migration_id) DO UPDATE SET
                source_profile_id = excluded.source_profile_id,
                destination_profile_id = excluded.destination_profile_id,
                state = excluded.state,
                total_memories = excluded.total_memories,
                completed_memories = excluded.completed_memories,
                failed_memory_ids = excluded.failed_memory_ids,
                last_memory_id = excluded.last_memory_id,
                updated_at = excluded.updated_at",
            params![
                checkpoint.migration_id.to_string(),
                checkpoint.source_profile_id.as_str(),
                checkpoint.destination_profile_id.as_str(),
                state,
                checkpoint.total_memories as i64,
                checkpoint.completed_memories as i64,
                failed_memory_ids,
                checkpoint.last_memory_id.map(|value| value.to_string()),
                checkpoint.started_at.to_rfc3339(),
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch a resumable migration checkpoint by immutable migration ID.
    pub fn profile_migration_checkpoint(
        &self,
        migration_id: Uuid,
    ) -> Result<Option<ProfileMigrationCheckpoint>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let row: Option<EmbeddingProfileMigrationRow> = reader
            .query_row(
                "SELECT source_profile_id, destination_profile_id, state, total_memories,
                        completed_memories, failed_memory_ids, last_memory_id, started_at, updated_at
                 FROM embedding_profile_migrations WHERE migration_id = ?1",
                params![migration_id.to_string()],
                |row| Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?,
                    row.get(5)?, row.get(6)?, row.get(7)?, row.get(8)?,
                )),
            )
            .optional()?;
        row.map(
            |(source, destination, state, total, completed, failed, last, started, updated)| {
                Ok(ProfileMigrationCheckpoint {
                    migration_id,
                    source_profile_id: EmbeddingProfileId::new(source).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                    destination_profile_id: EmbeddingProfileId::new(destination).map_err(
                        |error| StorageError::InvalidEmbeddingProfile(error.to_string()),
                    )?,
                    state: serde_json::from_value(serde_json::Value::String(state)).map_err(
                        |error| StorageError::InvalidEmbeddingProfile(error.to_string()),
                    )?,
                    total_memories: total.try_into().map_err(|_| {
                        StorageError::InvalidEmbeddingProfile(
                            "negative migration total".to_string(),
                        )
                    })?,
                    completed_memories: completed.try_into().map_err(|_| {
                        StorageError::InvalidEmbeddingProfile(
                            "negative migration completed count".to_string(),
                        )
                    })?,
                    failed_memory_ids: serde_json::from_str(&failed).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                    last_memory_id: last
                        .map(|value| {
                            Uuid::parse_str(&value).map_err(|error| {
                                StorageError::InvalidEmbeddingProfile(error.to_string())
                            })
                        })
                        .transpose()?,
                    started_at: Self::parse_rfc3339(started, "migration start timestamp")?,
                    updated_at: Self::parse_rfc3339(updated, "migration update timestamp")?,
                })
            },
        )
        .transpose()
    }

    /// Upsert one durable work item for a migration repair/resume queue.
    pub fn save_embedding_profile_migration_node_checkpoint(
        &self,
        checkpoint: &EmbeddingProfileMigrationNodeCheckpoint,
    ) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT INTO embedding_profile_migration_checkpoints
                (migration_id, node_id, state, error, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(migration_id, node_id) DO UPDATE SET
                state = excluded.state, error = excluded.error, updated_at = excluded.updated_at",
            params![
                &checkpoint.migration_id,
                &checkpoint.node_id,
                &checkpoint.state,
                &checkpoint.error,
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Atomically persist one destination-profile vector and its durable
    /// per-node migration checkpoint. A crash therefore leaves either neither
    /// record or both records—never a vector that a resume cursor believes was
    /// not written (or vice versa).
    pub fn put_embedding_profile_vector_with_migration_checkpoint(
        &self,
        vector: &EmbeddingProfileVector,
        checkpoint: &EmbeddingProfileMigrationNodeCheckpoint,
    ) -> Result<()> {
        if vector.node_id != checkpoint.node_id {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration checkpoint node ID must match its vector node ID".to_string(),
            ));
        }
        if vector.dimensions == 0 {
            return Err(StorageError::InvalidEmbeddingProfile(
                "profile vector dimensions must be positive".to_string(),
            ));
        }
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = Self::begin_write_transaction(
            &writer,
            "put_embedding_profile_vector_with_migration_checkpoint",
        )?;
        let destination_profile: Option<String> = tx
            .query_row(
                "SELECT destination_profile_id FROM embedding_profile_migrations WHERE migration_id = ?1",
                params![&checkpoint.migration_id],
                |row| row.get(0),
            )
            .optional()?;
        if destination_profile.as_deref() != Some(vector.profile_id.as_str()) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "migration '{}' does not target profile '{}'",
                checkpoint.migration_id, vector.profile_id
            )));
        }
        let declared_dimension: i64 = tx.query_row(
            "SELECT embedding_dimension FROM embedding_profiles WHERE profile_id = ?1",
            params![&vector.profile_id],
            |row| row.get(0),
        )?;
        if declared_dimension != i64::from(vector.dimensions) {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' declares {} dimensions but attempted vector has {}",
                vector.profile_id, declared_dimension, vector.dimensions
            )));
        }
        tx.execute(
            "INSERT INTO embedding_profile_vectors
                (profile_id, node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(profile_id, node_id) DO UPDATE SET
                embedding = excluded.embedding, dimensions = excluded.dimensions,
                model = excluded.model, created_at = excluded.created_at",
            params![
                &vector.profile_id,
                &vector.node_id,
                &vector.embedding,
                vector.dimensions as i64,
                &vector.model,
                vector.created_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "INSERT INTO embedding_profile_migration_checkpoints
                (migration_id, node_id, state, error, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(migration_id, node_id) DO UPDATE SET
                state = excluded.state, error = excluded.error, updated_at = excluded.updated_at",
            params![
                &checkpoint.migration_id,
                &checkpoint.node_id,
                &checkpoint.state,
                &checkpoint.error,
                checkpoint.updated_at.to_rfc3339(),
            ],
        )?;
        tx.execute(
            "UPDATE embedding_profile_manifests
             SET vector_count = (SELECT COUNT(*) FROM embedding_profile_vectors WHERE profile_id = ?1),
                 updated_at = ?2
             WHERE profile_id = ?1",
            params![&vector.profile_id, Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Return the latest resumable migration checkpoint for a destination
    /// profile, ordered deterministically by update time and migration ID.
    pub fn latest_profile_migration_checkpoint_for_destination(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<ProfileMigrationCheckpoint>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let migration_id: Option<String> = reader
            .query_row(
                "SELECT migration_id FROM embedding_profile_migrations
                 WHERE destination_profile_id = ?1
                 ORDER BY updated_at DESC, migration_id DESC LIMIT 1",
                params![profile_id.as_str()],
                |row| row.get(0),
            )
            .optional()?;
        drop(reader);
        migration_id
            .map(|id| {
                Uuid::parse_str(&id)
                    .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))
                    .and_then(|id| {
                        self.profile_migration_checkpoint(id)?.ok_or_else(|| {
                            StorageError::NotFound(format!("migration checkpoint {id}"))
                        })
                    })
            })
            .transpose()
    }

    /// Persist a migration snapshot receipt without ever storing a private
    /// absolute path. The path is relative to `data_dir()` and the report must
    /// bind both the snapshot bytes and the stable corpus snapshot by SHA-256.
    pub fn save_profile_migration_snapshot_receipt(
        &self,
        migration_id: Uuid,
        relative_snapshot_path: &Path,
        validation_report: &serde_json::Value,
    ) -> Result<()> {
        if relative_snapshot_path.is_absolute()
            || relative_snapshot_path.as_os_str().is_empty()
            || relative_snapshot_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(StorageError::InvalidEmbeddingProfile(
                "migration snapshot path must be a non-empty relative path under the Vestige data directory".to_string(),
            ));
        }
        let required_sha256 = ["snapshot_sha256", "corpus_sha256"];
        for key in required_sha256 {
            let valid = validation_report
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| {
                    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
            if !valid {
                return Err(StorageError::InvalidEmbeddingProfile(format!(
                    "migration validation report requires a 64-character SHA-256 '{key}'"
                )));
            }
        }
        let path = relative_snapshot_path.to_str().ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("snapshot path must be UTF-8".to_string())
        })?;
        let report = serde_json::to_string(validation_report)
            .map_err(|error| StorageError::InvalidEmbeddingProfile(error.to_string()))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let changed = writer.execute(
            "UPDATE embedding_profile_migrations
             SET snapshot_path = ?1, validation_report = ?2, updated_at = ?3
             WHERE migration_id = ?4",
            params![
                path,
                report,
                Utc::now().to_rfc3339(),
                migration_id.to_string()
            ],
        )?;
        if changed != 1 {
            return Err(StorageError::NotFound(format!("migration {migration_id}")));
        }
        Ok(())
    }

    /// Attach a verified process-local runtime to the currently active profile.
    ///
    /// This deliberately stores no artifact path and never initializes or
    /// downloads a model. The profile contract must exactly match the persisted
    /// active profile, which prevents a runner for one vector space from being
    /// used to query another.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(crate) fn attach_active_profile_embedder(
        &self,
        profile_id: &EmbeddingProfileId,
        embedder: Arc<ProfiledEmbedder>,
    ) -> Result<()> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        if active.profile_id != *profile_id {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "cannot attach runtime for '{}' while '{}' is active",
                profile_id, active.profile_id
            )));
        }
        let manifest = self
            .embedding_profile_manifest(profile_id)?
            .ok_or_else(|| StorageError::NotFound(profile_id.to_string()))?;
        if manifest.profile != *embedder.profile()
            || manifest.verification.status != VerificationStatus::Verified
            || manifest
                .runtime
                .as_ref()
                .is_none_or(|runtime| !runtime.local_only)
        {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "profile '{}' does not have a matching verified local runtime contract",
                profile_id
            )));
        }
        let mut attached = self.attached_profile_runtime.write().map_err(|_| {
            StorageError::Init("Attached profile runtime lock poisoned".to_string())
        })?;
        *attached = Some(AttachedProfileRuntime {
            profile_id: profile_id.clone(),
            embedder,
        });
        if let Some(cache) = &self.query_cache {
            cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?
                .clear();
        }
        Ok(())
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn attached_embedder_for(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Option<Arc<ProfiledEmbedder>>> {
        let attached = self.attached_profile_runtime.read().map_err(|_| {
            StorageError::Init("Attached profile runtime lock poisoned".to_string())
        })?;
        Ok(attached.as_ref().and_then(|runtime| {
            (runtime.profile_id == *profile_id).then(|| runtime.embedder.clone())
        }))
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn active_embedding_runtime_ready(&self) -> Result<bool> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        if self.attached_embedder_for(&active.profile_id)?.is_some() {
            return Ok(true);
        }
        Ok(
            manifest.profile.runtime_backend != EmbeddingRuntimeBackend::FastembedCandle
                && self.embedding_service.is_ready(),
        )
    }

    /// Check if the active profile has a usable local query runtime. Optional
    /// Qwen profiles return false until a verified runner is explicitly
    /// attached to this process.
    #[cfg(feature = "embeddings")]
    pub fn is_embedding_ready(&self) -> bool {
        #[cfg(feature = "vector-search")]
        {
            self.active_embedding_runtime_ready().unwrap_or(false)
        }
        #[cfg(not(feature = "vector-search"))]
        self.embedding_service.is_ready()
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn is_embedding_ready(&self) -> bool {
        false
    }

    /// Initialize the released Nomic default without widening optional profile
    /// activation into an implicit model-selection path.
    ///
    /// Existing installs have always initialized the active legacy Nomic
    /// runtime from normal CLI/MCP startup. Preserve that contract exactly.
    /// Every non-legacy profile, including all Qwen variants, remains an
    /// explicit artifact-backed workflow and cannot be initialized here.
    #[cfg(feature = "embeddings")]
    pub fn init_embeddings(&self) -> Result<()> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        if active.profile_id.as_str() != LEGACY_EMBEDDING_PROFILE_ID {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "direct embedding initialization is supported only for the released legacy Nomic profile; '{}' requires the explicit profile workflow",
                active.profile_id
            )));
        }
        self.embedding_service.init().map_err(|error| {
            StorageError::Init(format!("Initialize legacy Nomic embeddings: {error}"))
        })
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn init_embeddings(&self) -> Result<()> {
        Ok(()) // No-op when embeddings feature is disabled
    }

    /// Get query embedding from cache or compute it
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let cache_key = format!("{}\0{}", active.profile_id, query);
        // Check cache first
        let Some(index_cache) = self.query_cache.as_ref() else {
            return Err(StorageError::Init("Query cache unavailable".to_string()));
        };
        {
            let mut cache = index_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Never fall back from an active optional profile to the legacy
        // service. Qwen vectors and Nomic vectors are different spaces; a
        // missing explicit attachment is an availability error, not permission
        // to issue a semantically invalid query.
        let vector = if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
            let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                StorageError::Init(format!("Create local query runtime: {error}"))
            })?;
            runtime
                .block_on(embedder.embed_query(query))
                .map_err(|error| StorageError::Init(format!("Failed to embed query: {error}")))?
        } else if manifest.profile.runtime_backend == EmbeddingRuntimeBackend::FastembedCandle {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires an explicitly attached verified local runtime; supply its artifact directory for this process",
                active.profile_id
            )));
        } else {
            self.embedding_service
                .embed(
                    &manifest.profile.encode_query(query).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                )
                .map_err(|e| StorageError::Init(format!("Failed to embed query: {e}")))?
                .vector
        };
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id,
                manifest.profile.embedding_dimension,
                vector.len()
            )));
        }

        // Store in cache
        {
            let mut cache = index_cache
                .lock()
                .map_err(|_| StorageError::Init("Query cache lock poisoned".to_string()))?;
            cache.put(cache_key, vector.clone());
        }

        Ok(vector)
    }

    /// Compute one document vector for the active profile without populating
    /// the query cache. Document and query templates are distinct parts of a
    /// profile contract, particularly for Qwen retrieval profiles.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn get_document_embedding(&self, content: &str) -> Result<Vec<f32>> {
        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let vector = if let Some(embedder) = self.attached_embedder_for(&active.profile_id)? {
            let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                StorageError::Init(format!("Create local document runtime: {error}"))
            })?;
            runtime
                .block_on(embedder.embed_document(content))
                .map_err(|error| StorageError::Init(format!("Failed to embed document: {error}")))?
        } else if manifest.profile.runtime_backend == EmbeddingRuntimeBackend::FastembedCandle {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires an explicitly attached verified local runtime; supply its artifact directory for this process",
                active.profile_id
            )));
        } else {
            self.embedding_service
                .embed(
                    &manifest.profile.encode_document(content).map_err(|error| {
                        StorageError::InvalidEmbeddingProfile(error.to_string())
                    })?,
                )
                .map_err(|error| StorageError::Init(format!("Failed to embed document: {error}")))?
                .vector
        };
        if vector.len() != manifest.profile.embedding_dimension {
            return Err(StorageError::InvalidEmbeddingProfile(format!(
                "active profile '{}' requires {} dimensions but its runtime produced {}",
                active.profile_id,
                manifest.profile.embedding_dimension,
                vector.len()
            )));
        }
        Ok(vector)
    }

    /// Bring the in-process vector index up to date with vectors written by OTHER
    /// processes since this one last looked. Returns the number of index
    /// mutations applied: vectors added, replaced or removed.
    ///
    /// THE BUG THIS FIXES (#181). The HNSW index is process-local: it is built once
    /// at startup from `embedding_profile_vectors` and thereafter only ever appended
    /// to by THIS process's own ingests. A second MCP server writing to the same
    /// SQLite file is therefore invisible to it. In a normal setup, a desktop
    /// client, an editor integration, a CLI and a dashboard all pointed at one store,
    /// every long-lived process is semantically blind to everything its peers have
    /// written since it booted. The consequences are silent: the prediction-error
    /// gate sees no similar candidate and creates a duplicate instead of reinforcing,
    /// and recall returns an incomplete answer with no indication anything is missing.
    /// The FTS5 leg reads SQLite directly and is unaffected, which is exactly why the
    /// failure is partial and hard to notice.
    ///
    /// THE SIGNAL. `PRAGMA data_version` is incremented on a connection whenever a
    /// DIFFERENT connection commits. Reading it is a single pragma with no table
    /// access, so this check is affordable on every query, and when nothing has
    /// changed it costs one integer comparison. It only says THAT something changed.
    ///
    /// WHAT CHANGED comes from `vector_journal` (migration V32). Three triggers append
    /// one row per insert, update or delete of `embedding_profile_vectors`, keyed by
    /// an AUTOINCREMENT `seq` that is allocated inside the writer's transaction: so it
    /// is monotonic in commit order, never reused, and independent of wall clocks.
    /// The index remembers the last `seq` it absorbed and reads exactly the rows past
    /// it. A peer re-embedding an existing node is an upsert row, so the stale vector
    /// is replaced; a peer's purge is a delete row, so the dead vector leaves the
    /// index. The first version of this refresh rescanned every vector row, blob
    /// included, on every external commit, and could not see re-embeddings at all
    /// because it skipped any id the index already held.
    ///
    /// RECONCILE. If the watermark is unknown, or the journal has been pruned past
    /// it, the index is compared against the table instead: one covering scan of
    /// node ids for the active profile, add what is missing, drop what is gone. That
    /// is O(N) over ids only, and it runs in exactly those two cases.
    ///
    /// LOCK DISCIPLINE. This acquires the reader lock, the watermark lock and the
    /// index lock SEQUENTIALLY and never holds two at once. `semantic_search_raw`
    /// holds only the index lock, so no ordering cycle exists and this cannot
    /// deadlock against it.
    ///
    /// FAILS OPEN. A refresh problem must degrade to a possibly-stale index, never
    /// break the query: returning an error here would turn a peer's write into an
    /// outage.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn refresh_vector_index_if_stale(&self) -> usize {
        let Some(index_mutex) = self.vector_index.as_ref() else {
            return 0;
        };

        // --- reader lock: has anyone else committed? ---
        let current_version: i64 = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            match reader.query_row("PRAGMA data_version", [], |row| row.get(0)) {
                Ok(v) => v,
                Err(_) => return 0,
            }
        };
        // --- watermark lock: compare, and take the journal position ---
        let last_seq = {
            let Ok(mut watermark) = self.vector_index_watermark.lock() else {
                return 0;
            };
            if watermark.data_version == current_version {
                return 0; // nothing has changed since we last looked
            }
            watermark.data_version = current_version;
            watermark.journal_seq
        };

        let Ok(Some(active)) = self.active_embedding_profile() else {
            return 0;
        };
        let profile_id = active.profile_id.as_str();

        // --- reader lock, one snapshot: what changed past the watermark? ---
        let plan = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(snapshot) = begin_read_snapshot(&reader) else {
                return 0;
            };
            match Self::vector_refresh_plan(&snapshot, profile_id, last_seq) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "vector index refresh could not read the journal; the index may be stale until the next query"
                    );
                    return 0;
                }
            }
        };

        let (changes, head) = match plan {
            VectorRefreshPlan::Reconcile => {
                return self.reconcile_vector_index(index_mutex, profile_id);
            }
            VectorRefreshPlan::Incremental { changes, head } => (changes, head),
        };

        // --- index lock: apply exactly what the journal named ---
        let mut applied = 0usize;
        if !changes.is_empty() {
            let Ok(mut index) = index_mutex.lock() else {
                return 0;
            };
            for (node_id, blob) in changes {
                let mutated = match blob {
                    None => matches!(index.remove(&node_id), Ok(true)),
                    Some(blob) => Self::add_journaled_vector(&mut index, &node_id, &blob),
                };
                if mutated {
                    applied += 1;
                }
            }
        }
        // --- watermark lock: current through `head` ---
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq < head
        {
            watermark.journal_seq = head;
        }
        if applied > 0 {
            tracing::debug!(
                applied,
                head,
                data_version = current_version,
                "refreshed vector index with memories written by another process"
            );
        }
        applied
    }

    /// Read the journal past `last_seq` for `profile_id` inside `snapshot`, and
    /// decide whether the index can follow it or must reconcile.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn vector_refresh_plan(
        snapshot: &Connection,
        profile_id: &str,
        last_seq: i64,
    ) -> rusqlite::Result<VectorRefreshPlan> {
        if last_seq < 0 {
            return Ok(VectorRefreshPlan::Reconcile);
        }
        let (oldest, head): (Option<i64>, i64) = snapshot.query_row(
            "SELECT MIN(seq), COALESCE(MAX(seq), 0) FROM vector_journal",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        // Pruned past us: rows between our watermark and the oldest survivor are
        // gone, or the journal was emptied after we had already seen rows.
        let pruned_past_us = match oldest {
            Some(oldest) => oldest > last_seq + 1,
            None => last_seq > 0,
        };
        if pruned_past_us {
            return Ok(VectorRefreshPlan::Reconcile);
        }

        // Last op per node wins; the journal is read in seq order.
        let mut latest: HashMap<String, bool> = HashMap::new();
        let mut stmt = snapshot.prepare(
            "SELECT node_id, op FROM vector_journal
             WHERE profile_id = ?1 AND seq > ?2
             ORDER BY seq",
        )?;
        let rows = stmt.query_map(params![profile_id, last_seq], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (node_id, op) = row?;
            latest.insert(node_id, op == "delete");
        }
        drop(stmt);

        let mut fetch = snapshot.prepare(
            "SELECT embedding FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
        )?;
        let mut changes = Vec::with_capacity(latest.len());
        for (node_id, deleted) in latest {
            if deleted {
                changes.push((node_id, None));
                continue;
            }
            // Same snapshot as the journal read, so an upsert whose row is
            // nevertheless absent can only be a later delete we also saw.
            let blob: Option<Vec<u8>> = fetch
                .query_row(params![profile_id, &node_id], |row| row.get(0))
                .optional()?;
            changes.push((node_id, blob));
        }
        Ok(VectorRefreshPlan::Incremental { changes, head })
    }

    /// Decode and add one journaled vector. Same decoder the startup builder
    /// uses, so a vector added here is identical to one added by a rebuild. A
    /// vector this index cannot hold (unreadable, wrong dimension) is skipped;
    /// the memory stays keyword-searchable and never fails a query.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn add_journaled_vector(index: &mut VectorIndex, node_id: &str, blob: &[u8]) -> bool {
        let Some(embedding) = Embedding::from_bytes(blob) else {
            tracing::warn!(
                node_id,
                "skipping an unreadable vector during index refresh"
            );
            return false;
        };
        if embedding.dimensions != index.dimensions() {
            return false; // another profile's width: not ours to hold
        }
        index.add(node_id, &embedding.vector).is_ok()
    }

    /// Compare the index against `embedding_profile_vectors` for `profile_id`
    /// and fix the difference. Used when the journal cannot be trusted to be
    /// complete: an unknown watermark, or a journal pruned past it. O(N) over
    /// node ids (a covering index scan), fetching only the vectors that are
    /// missing. Returns the number of index mutations.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    fn reconcile_vector_index(&self, index_mutex: &Mutex<VectorIndex>, profile_id: &str) -> usize {
        // --- reader lock, one snapshot: every id the table holds, and the head ---
        let (present, head): (HashSet<String>, i64) = {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(snapshot) = begin_read_snapshot(&reader) else {
                return 0;
            };
            let Ok(mut stmt) = snapshot
                .prepare("SELECT node_id FROM embedding_profile_vectors WHERE profile_id = ?1")
            else {
                return 0;
            };
            let Ok(rows) = stmt.query_map(params![profile_id], |row| row.get::<_, String>(0))
            else {
                return 0;
            };
            let present: HashSet<String> = rows
                .filter_map(warn_skipped_row("reconcile_vector_index"))
                .collect();
            drop(stmt);
            let head: i64 = match snapshot.query_row(
                "SELECT COALESCE(MAX(seq), 0) FROM vector_journal",
                [],
                |row| row.get(0),
            ) {
                Ok(head) => head,
                Err(_) => return 0,
            };
            (present, head)
        };

        // --- index lock: the two differences ---
        let (missing, gone): (Vec<String>, Vec<String>) = {
            let Ok(index) = index_mutex.lock() else {
                return 0;
            };
            let missing = present
                .iter()
                .filter(|id| !index.contains(id))
                .cloned()
                .collect();
            let gone = index
                .keys()
                .filter(|key| !present.contains(*key))
                .map(str::to_string)
                .collect();
            (missing, gone)
        };

        // --- reader lock: fetch only what is missing ---
        let blobs: Vec<(String, Vec<u8>)> = if missing.is_empty() {
            Vec::new()
        } else {
            let Ok(reader) = self.reader.lock() else {
                return 0;
            };
            let Ok(mut stmt) = reader.prepare(
                "SELECT embedding FROM embedding_profile_vectors WHERE profile_id = ?1 AND node_id = ?2",
            ) else {
                return 0;
            };
            missing
                .into_iter()
                .filter_map(|node_id| {
                    stmt.query_row(params![profile_id, &node_id], |row| {
                        row.get::<_, Vec<u8>>(0)
                    })
                    .optional()
                    .ok()
                    .flatten()
                    .map(|blob| (node_id, blob))
                })
                .collect()
        };

        // --- index lock: apply ---
        let mut applied = 0usize;
        {
            let Ok(mut index) = index_mutex.lock() else {
                return 0;
            };
            for node_id in gone {
                if matches!(index.remove(&node_id), Ok(true)) {
                    applied += 1;
                }
            }
            for (node_id, blob) in blobs {
                if Self::add_journaled_vector(&mut index, &node_id, &blob) {
                    applied += 1;
                }
            }
        }
        // A vector this process wrote between the snapshot and the apply above
        // may have been removed as `gone`; its journal row sits past `head`, so
        // the next refresh puts it back.
        if let Ok(mut watermark) = self.vector_index_watermark.lock()
            && watermark.journal_seq < head
        {
            watermark.journal_seq = head;
        }
        tracing::info!(
            applied,
            head,
            profile_id,
            "reconciled the vector index against the database"
        );
        applied
    }

    /// Point the watermark at a freshly built index: it holds every vector row
    /// as of journal position `journal_seq`, and the next search must look at
    /// the journal once regardless of what the reader's data_version says.
    #[cfg(feature = "vector-search")]
    fn reset_vector_index_watermark(&self, journal_seq: i64) {
        if let Ok(mut watermark) = self.vector_index_watermark.lock() {
            watermark.journal_seq = journal_seq;
            watermark.data_version = -1;
        }
    }

    /// Trim `vector_journal` (#181). A row is needed only until every peer has
    /// absorbed it, and a peer that has been away long enough to miss trimmed
    /// rows reconciles against the table, so this keeps the newest 10,000 rows
    /// plus everything younger than seven days and deletes the rest. Ids only;
    /// there is no content in this table to protect. Returns rows deleted.
    pub(crate) fn prune_vector_journal(&self) -> Result<usize> {
        const KEEP_ROWS: i64 = 10_000;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let deleted = writer.execute(
            "DELETE FROM vector_journal
             WHERE seq <= (SELECT COALESCE(MAX(seq), 0) FROM vector_journal) - ?1
               AND at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-7 days')",
            params![KEEP_ROWS],
        )?;
        Ok(deleted)
    }

    /// Semantic search returning scores
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn semantic_search_raw(
        &self,
        query: &str,
        limit: i32,
    ) -> Result<Vec<(String, f32)>> {
        if !self.vector_search_available() {
            return Ok(vec![]);
        }
        if !self.active_embedding_runtime_ready()? {
            return Err(StorageError::InvalidEmbeddingProfile(
                "active embedding profile has no explicitly attached local query runtime"
                    .to_string(),
            ));
        }

        // HyDE query expansion: for conceptual queries, embed expanded variants
        // and use the centroid for broader semantic coverage
        let intent = hyde::classify_intent(query);
        let query_embedding = match intent {
            hyde::QueryIntent::Definition
            | hyde::QueryIntent::HowTo
            | hyde::QueryIntent::Reasoning
            | hyde::QueryIntent::Lookup => {
                let variants = hyde::expand_query(query);
                let embeddings: Vec<Vec<f32>> = variants
                    .iter()
                    .filter_map(|v| self.get_query_embedding(v).ok())
                    .collect();
                if embeddings.len() > 1 {
                    hyde::centroid_embedding(&embeddings)
                } else {
                    self.get_query_embedding(query)?
                }
            }
            _ => self.get_query_embedding(query)?,
        };

        // Pick up anything a peer process wrote since we last searched (#181).
        // Cheap when nothing changed: one PRAGMA and an integer comparison. Runs
        // BEFORE the index lock is taken, and takes its own locks sequentially,
        // so it cannot deadlock against the search below.
        self.refresh_vector_index_if_stale();

        let index = self.vector_index.as_ref().unwrap();
        let index = index
            .lock()
            .map_err(|_| StorageError::Init("Vector index lock poisoned".to_string()))?;

        index
            .search(&query_embedding, limit as usize)
            .map_err(|e| StorageError::Init(format!("Vector search failed: {}", e)))
    }

    /// Generate embeddings for nodes
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn generate_embeddings(
        &self,
        node_ids: Option<&[String]>,
        force: bool,
    ) -> Result<EmbeddingResult> {
        if !self.active_embedding_runtime_ready()? {
            // Generating vectors is never authority to download or initialize
            // a model. Explicit profile installation/runtime preparation must
            // happen first; callers receive an honest empty result meanwhile.
            tracing::debug!("Skipping embedding generation: active runtime is not installed/ready");
            return Ok(EmbeddingResult::default());
        }

        let active = self.active_embedding_profile()?.ok_or_else(|| {
            StorageError::InvalidEmbeddingProfile("no active embedding profile pointer".to_string())
        })?;
        let active_manifest = self
            .embedding_profile_manifest(&active.profile_id)?
            .ok_or_else(|| StorageError::NotFound(active.profile_id.to_string()))?;
        let active_model = active_manifest.profile.model_id.as_str();
        let mut result = EmbeddingResult::default();
        let nodes = self.embedding_regeneration_candidates(
            &active.profile_id,
            active_manifest.profile.embedding_dimension,
            active_model,
            node_ids,
            force,
        )?;

        for (id, content, stored_model) in nodes {
            if !force {
                let stored_model: Option<String> = self
                    .reader
                    .lock()
                    .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?
                    .query_row(
                        "SELECT model FROM embedding_profile_vectors
                         WHERE profile_id = ?1 AND node_id = ?2",
                        params![active.profile_id.as_str(), &id],
                        |row| row.get(0),
                    )
                    .optional()?
                    .or(stored_model);

                if stored_model.as_deref() == Some(active_model) {
                    result.skipped += 1;
                    continue;
                }
            }

            match self.generate_embedding_for_node(&id, &content) {
                Ok(()) => result.successful += 1,
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(format!("{}: {}", id, e));
                }
            }
        }

        Ok(result)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn embedding_regeneration_candidates(
        &self,
        profile_id: &EmbeddingProfileId,
        profile_dimension: usize,
        profile_model: &str,
        node_ids: Option<&[String]>,
        force: bool,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;

        if let Some(ids) = node_ids {
            if ids.is_empty() {
                return Ok(Vec::new());
            }

            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT kn.id, kn.content, epv.model
                 FROM knowledge_nodes kn
                 LEFT JOIN embedding_profile_vectors epv
                   ON epv.node_id = kn.id AND epv.profile_id = ?
                 WHERE kn.id IN ({})",
                placeholders
            );

            let mut stmt = reader.prepare(&query)?;
            let profile = profile_id.as_str();
            let mut params: Vec<&dyn rusqlite::ToSql> = vec![&profile];
            params.extend(ids.iter().map(|id| id as &dyn rusqlite::ToSql));
            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows
                .filter_map(warn_skipped_row("embedding_regeneration_candidates"))
                .collect());
        }

        if force {
            let mut stmt = reader.prepare(
                "SELECT kn.id, kn.content, epv.model
                 FROM knowledge_nodes kn
                 LEFT JOIN embedding_profile_vectors epv
                   ON epv.node_id = kn.id AND epv.profile_id = ?1",
            )?;
            let rows = stmt.query_map(params![profile_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            return Ok(rows
                .filter_map(warn_skipped_row("embedding_regeneration_candidates"))
                .collect());
        }

        let mut stmt = reader.prepare(
            "SELECT kn.id, kn.content, epv.model
             FROM knowledge_nodes kn
             LEFT JOIN embedding_profile_vectors epv
               ON epv.node_id = kn.id AND epv.profile_id = ?1
             WHERE epv.node_id IS NULL OR epv.dimensions != ?2 OR epv.model != ?3",
        )?;
        let rows = stmt.query_map(
            params![profile_id.as_str(), profile_dimension as i64, profile_model],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )?;
        Ok(rows
            .filter_map(warn_skipped_row("embedding_regeneration_candidates"))
            .collect())
    }

    /// Generate all missing or active-model-mismatched embeddings.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub(super) fn generate_missing_embeddings(&self) -> Result<i64> {
        if !self.active_embedding_runtime_ready()? {
            tracing::debug!(
                "Skipping consolidation embedding generation: active profile runtime is unavailable"
            );
            return Ok(0);
        }

        let result = self.generate_embeddings(None, false)?;
        if result.failed > 0 {
            tracing::warn!(
                failed = result.failed,
                "Some embeddings could not be regenerated during consolidation"
            );
        }

        Ok(result.successful)
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search", test))]
    pub(super) fn embedding_model_matches_active(stored_model: &str, active_model: &str) -> bool {
        // Profile-aware retrieval never uses model-family matching. This helper
        // remains solely for legacy vector-repair bookkeeping.
        stored_model == active_model
    }

    #[cfg(all(feature = "embeddings", feature = "vector-search", test))]
    pub(super) fn embedding_vector_for_active_model(
        embedding_bytes: &[u8],
        stored_model: &str,
        active_model: &str,
    ) -> Option<Vec<f32>> {
        if !Self::embedding_model_matches_active(stored_model, active_model) {
            return None;
        }
        Embedding::from_bytes(embedding_bytes).map(|embedding| embedding.vector)
    }
}
