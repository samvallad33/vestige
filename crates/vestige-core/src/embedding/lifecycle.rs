//! Explicit, offline lifecycle runner for embedding profiles.
//!
//! This module is deliberately separate from the command-line and dashboard
//! surfaces.  Its only model input is a caller-supplied artifact directory;
//! the directory is verified for each invocation and is never persisted.  A
//! migration writes exclusively to its destination profile and cannot move the
//! active-profile pointer.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    embedder::{Embedder, EmbedderError},
    storage::{
        EmbeddingProfileIntegrityManifest, EmbeddingProfileMigrationNodeCheckpoint,
        EmbeddingProfileVector, Storage, StorageError,
    },
};

use super::{
    EmbeddingEvaluationSummary, EmbeddingMigrationState, EmbeddingProfile, EmbeddingProfileError,
    EmbeddingProfileId, EmbeddingProfileManifest, EmbeddingProfileState, EmbeddingRuntimeMetadata,
    ProfileMigrationCheckpoint, ProfileRuntimeRegistry, ProfiledEmbedder, VerifiedLocalArtifact,
};

/// A verified local lifecycle failure. No variant retries, downloads, or
/// activates a profile as a side effect.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingLifecycleError {
    #[error(transparent)]
    Profile(#[from] EmbeddingProfileError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Embedder(#[from] EmbedderError),
    #[error(
        "local Qwen runner is not included in this Vestige build; rebuild with --features qwen3-embeddings"
    )]
    RunnerUnavailable,
    #[error(
        "profile migration requires the vector-search feature so Vestige can build and verify its HNSW sidecar"
    )]
    SidecarUnavailable,
    #[error("profile '{0}' is not a locally verified installed profile")]
    NotInstalled(EmbeddingProfileId),
    #[error("profile '{0}' must be evaluated before migration")]
    NotReady(EmbeddingProfileId),
    #[error("migration '{0}' does not match the requested source/destination profiles")]
    MigrationMismatch(Uuid),
    #[error("migration '{migration_id}' intentionally stopped after {completed} new memories")]
    Interrupted { migration_id: Uuid, completed: u64 },
    #[error("invalid Agent Memory Eval fixture: {0}")]
    InvalidFixture(String),
    #[error("filesystem operation failed: {0}")]
    Filesystem(#[from] std::io::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Durable evaluation evidence written under the store sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingLifecycleEvaluationReceipt {
    pub evaluation: EmbeddingEvaluationSummary,
    pub report_path: String,
    pub report_sha256: String,
    pub baseline_available: bool,
}

/// Evidence for a migration that has either completed or stopped safely.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingLifecycleMigrationReceipt {
    pub migration_id: Uuid,
    pub state: EmbeddingMigrationState,
    pub snapshot_path: String,
    pub snapshot_sha256: String,
    pub corpus_sha256: String,
    pub total_memories: u64,
    pub completed_memories: u64,
    pub failed_memory_ids: Vec<Uuid>,
    pub active_profile_unchanged: bool,
    pub integrity_hash: Option<String>,
}

/// Profile lifecycle service. The registry is process-local by design: callers
/// must explicitly re-supply local artifacts on a later CLI/dashboard process.
pub struct EmbeddingProfileLifecycle<'a> {
    storage: &'a Storage,
    registry: ProfileRuntimeRegistry,
}

impl<'a> EmbeddingProfileLifecycle<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self {
            storage,
            registry: ProfileRuntimeRegistry::new(),
        }
    }

    /// Resolve and re-hash the exact artifact contract beneath `artifact_root`.
    /// The root never enters a persisted manifest or receipt.
    pub fn verified_artifacts(
        profile: &EmbeddingProfile,
        artifact_root: &Path,
    ) -> Result<Vec<VerifiedLocalArtifact>, EmbeddingLifecycleError> {
        profile.validate()?;
        if profile.verified_model_artifact_hashes.is_empty() {
            return Err(EmbeddingProfileError::InvalidManifest(
                "profile has no pinned local artifacts".to_string(),
            )
            .into());
        }
        profile
            .verified_model_artifact_hashes
            .iter()
            .cloned()
            .map(|artifact| VerifiedLocalArtifact::from_root(artifact, artifact_root))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Persist an explicit verified install. This never evaluates, migrates,
    /// or changes the active pointer.
    pub fn install_verified(
        &self,
        profile: EmbeddingProfile,
        artifacts: &[VerifiedLocalArtifact],
        runtime: EmbeddingRuntimeMetadata,
        embedder: Arc<dyn Embedder>,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        let manifest = self
            .registry
            .install_verified(profile, artifacts, runtime, embedder)?;
        self.storage.save_embedding_profile_manifest(&manifest)?;
        Ok(manifest)
    }

    /// Explicit Qwen 0.6B install from a verified local artifact directory.
    #[cfg(feature = "qwen3-embeddings")]
    pub fn install_qwen3_local(
        &self,
        profile: EmbeddingProfile,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        let manifest = self.register_qwen3_local_verified(profile, artifact_root)?;
        self.storage.save_embedding_profile_manifest(&manifest)?;
        Ok(manifest)
    }

    #[cfg(not(feature = "qwen3-embeddings"))]
    pub fn install_qwen3_local(
        &self,
        _profile: EmbeddingProfile,
        _artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Explicit Granite install from a verified local artifact directory.
    /// Same contract as the Qwen path: artifacts are hash-verified against
    /// the profile, and no Hub or network client is ever invoked.
    #[cfg(feature = "embeddings")]
    pub fn install_granite_onnx(
        &self,
        profile: EmbeddingProfile,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        let manifest = self.register_granite_onnx_verified(profile, artifact_root)?;
        self.storage.save_embedding_profile_manifest(&manifest)?;
        Ok(manifest)
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn install_granite_onnx(
        &self,
        _profile: EmbeddingProfile,
        _artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Recreate the Granite runner from the caller-supplied local artifact
    /// directory and attach it to the currently active matching profile.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn attach_active_granite_onnx(
        &self,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileId, EmbeddingLifecycleError> {
        let active = self.storage.active_embedding_profile()?.ok_or_else(|| {
            EmbeddingLifecycleError::InvalidFixture(
                "no active embedding profile is configured".to_string(),
            )
        })?;
        let profile = self.installed_profile(&active.profile_id)?;
        self.register_granite_onnx_verified(profile, artifact_root)?;
        self.attach_registered_active_profile(&active.profile_id)?;
        Ok(active.profile_id)
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    pub fn attach_active_granite_onnx(
        &self,
        _artifact_root: &Path,
    ) -> Result<EmbeddingProfileId, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Migrate to a Granite profile using a process-local verified runner.
    #[cfg(feature = "embeddings")]
    pub fn migrate_granite_onnx(
        &self,
        destination: &EmbeddingProfileId,
        source: &EmbeddingProfileId,
        artifact_root: &Path,
        migration_id: Option<Uuid>,
        interrupt_after: Option<u64>,
    ) -> Result<EmbeddingLifecycleMigrationReceipt, EmbeddingLifecycleError> {
        let profile = self.installed_profile(destination)?;
        self.register_granite_onnx_verified(profile, artifact_root)?;
        self.migrate_registered(destination, source, migration_id, interrupt_after)
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn migrate_granite_onnx(
        &self,
        _destination: &EmbeddingProfileId,
        _source: &EmbeddingProfileId,
        _artifact_root: &Path,
        _migration_id: Option<Uuid>,
        _interrupt_after: Option<u64>,
    ) -> Result<EmbeddingLifecycleMigrationReceipt, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Evaluate a currently registered profile against the committed Agent
    /// Memory Eval fixture. It records the candidate's raw rankings and never
    /// invents a baseline score when no baseline runtime was supplied.
    pub fn evaluate_registered(
        &self,
        profile_id: &EmbeddingProfileId,
        fixture_dir: &Path,
        compared_against: EmbeddingProfileId,
        baseline_available: bool,
    ) -> Result<EmbeddingLifecycleEvaluationReceipt, EmbeddingLifecycleError> {
        let persisted = self
            .storage
            .embedding_profile_manifest(profile_id)?
            .ok_or_else(|| EmbeddingLifecycleError::NotInstalled(profile_id.clone()))?;
        ensure_locally_verified(&persisted)?;
        let embedder = self.registry.embedder(profile_id)?;
        let outcome = evaluate_fixture(
            &embedder,
            &persisted,
            fixture_dir,
            compared_against.clone(),
            baseline_available,
        )?;
        let manifest = self
            .registry
            .record_evaluation(profile_id, outcome.summary.clone())?;
        self.storage.save_embedding_profile_manifest(&manifest)?;
        self.write_evaluation_report(profile_id, outcome.evaluation_id, &outcome.report)?;
        Ok(EmbeddingLifecycleEvaluationReceipt {
            evaluation: outcome.summary,
            report_path: format!(
                "embedding-profiles/evaluations/{}/{}.json",
                profile_id, outcome.evaluation_id
            ),
            report_sha256: outcome.report_hash,
            baseline_available,
        })
    }

    /// Reconstruct the Qwen runner from local artifacts for a fresh process,
    /// evaluate it, and leave it only in this process-local registry.
    #[cfg(feature = "qwen3-embeddings")]
    pub fn evaluate_qwen3_local(
        &self,
        profile_id: &EmbeddingProfileId,
        artifact_root: &Path,
        fixture_dir: &Path,
        compared_against: EmbeddingProfileId,
        baseline_available: bool,
    ) -> Result<EmbeddingLifecycleEvaluationReceipt, EmbeddingLifecycleError> {
        let profile = self.installed_profile(profile_id)?;
        // Re-registering proves the explicit path hashes still match. The
        // existing persisted Ready receipt remains untouched until the new
        // evaluation receipt replaces it.
        self.register_qwen3_local_verified(profile, artifact_root)?;
        self.evaluate_registered(
            profile_id,
            fixture_dir,
            compared_against,
            baseline_available,
        )
    }

    #[cfg(not(feature = "qwen3-embeddings"))]
    pub fn evaluate_qwen3_local(
        &self,
        _profile_id: &EmbeddingProfileId,
        _artifact_root: &Path,
        _fixture_dir: &Path,
        _compared_against: EmbeddingProfileId,
        _baseline_available: bool,
    ) -> Result<EmbeddingLifecycleEvaluationReceipt, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Attach a process-local, already registered profile runtime to storage
    /// for live semantic retrieval. The storage layer verifies that this is
    /// exactly the active, locally verified profile and clears its query cache.
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    pub fn attach_registered_active_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<(), EmbeddingLifecycleError> {
        let embedder = self.registry.embedder(profile_id)?;
        self.storage
            .attach_active_profile_embedder(profile_id, embedder)?;
        Ok(())
    }

    /// Recreate Qwen from the caller-supplied local artifact directory and
    /// attach it only to the currently active matching profile. This is the
    /// explicit fresh-process path for semantic recall; the directory is never
    /// persisted and no Hub/network client is invoked.
    #[cfg(all(feature = "qwen3-embeddings", feature = "vector-search"))]
    pub fn attach_active_qwen3_local(
        &self,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileId, EmbeddingLifecycleError> {
        let active = self.storage.active_embedding_profile()?.ok_or_else(|| {
            EmbeddingLifecycleError::InvalidFixture(
                "no active embedding profile is configured".to_string(),
            )
        })?;
        let profile = self.installed_profile(&active.profile_id)?;
        self.register_qwen3_local_verified(profile, artifact_root)?;
        self.attach_registered_active_profile(&active.profile_id)?;
        Ok(active.profile_id)
    }

    #[cfg(not(all(feature = "qwen3-embeddings", feature = "vector-search")))]
    pub fn attach_active_qwen3_local(
        &self,
        _artifact_root: &Path,
    ) -> Result<EmbeddingProfileId, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    /// Migrate using a process-local, profile-scoped embedder. `migration_id`
    /// is optional when starting; callers must provide it to resume the same
    /// immutable snapshot. The active pointer is read before and after as a
    /// hard assertion.
    pub fn migrate_registered(
        &self,
        destination: &EmbeddingProfileId,
        source: &EmbeddingProfileId,
        migration_id: Option<Uuid>,
        interrupt_after: Option<u64>,
    ) -> Result<EmbeddingLifecycleMigrationReceipt, EmbeddingLifecycleError> {
        let destination_manifest = self
            .storage
            .embedding_profile_manifest(destination)?
            .ok_or_else(|| EmbeddingLifecycleError::NotInstalled(destination.clone()))?;
        ensure_locally_verified(&destination_manifest)?;
        if destination_manifest.state != EmbeddingProfileState::Ready {
            return Err(EmbeddingLifecycleError::NotReady(destination.clone()));
        }
        let embedder = self.registry.embedder(destination)?;
        let active_before = self
            .storage
            .active_embedding_profile()?
            .map(|value| value.profile_id);
        let migration_id = migration_id.unwrap_or_else(Uuid::new_v4);
        let paths = MigrationPaths::new(self.storage, migration_id)?;
        let existing = self.storage.profile_migration_checkpoint(migration_id)?;
        let checkpoint = if let Some(checkpoint) = existing {
            if checkpoint.source_profile_id != *source
                || checkpoint.destination_profile_id != *destination
            {
                return Err(EmbeddingLifecycleError::MigrationMismatch(migration_id));
            }
            checkpoint
        } else {
            paths.create_parent()?;
            self.storage.backup_to(&paths.snapshot)?;
            let snapshot = Storage::new(Some(paths.snapshot.clone()))?;
            let nodes = all_snapshot_nodes(&snapshot)?;
            let now = Utc::now();
            let checkpoint = ProfileMigrationCheckpoint {
                migration_id,
                source_profile_id: source.clone(),
                destination_profile_id: destination.clone(),
                state: EmbeddingMigrationState::Pending,
                total_memories: nodes.len() as u64,
                completed_memories: 0,
                failed_memory_ids: Vec::new(),
                last_memory_id: None,
                started_at: now,
                updated_at: now,
            };
            self.storage
                .save_profile_migration_checkpoint(&checkpoint)?;
            checkpoint
        };
        if !paths.snapshot.is_file() {
            return Err(EmbeddingLifecycleError::InvalidFixture(format!(
                "migration snapshot '{}' is missing; refusing to enumerate live mutable data",
                paths.snapshot.display()
            )));
        }
        let snapshot_bytes = fs::read(&paths.snapshot)?;
        let snapshot_sha256 = sha256_hex(&snapshot_bytes);
        let snapshot = Storage::new(Some(paths.snapshot.clone()))?;
        let mut nodes = all_snapshot_nodes(&snapshot)?;
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let corpus_sha256 = corpus_hash(&nodes);
        self.storage.save_profile_migration_snapshot_receipt(
            migration_id,
            Path::new(&paths.relative_path),
            &serde_json::json!({
                "snapshot_sha256": snapshot_sha256,
                "corpus_sha256": corpus_sha256,
                "corpus_count": nodes.len(),
                "snapshot_format": "sqlite-vacuum-into/v1",
            }),
        )?;
        let total = nodes.len() as u64;
        if total != checkpoint.total_memories {
            return Err(EmbeddingLifecycleError::InvalidFixture(
                "snapshot corpus count changed after checkpoint creation".to_string(),
            ));
        }

        let now = Utc::now();
        let mut running = checkpoint.clone();
        running.state = EmbeddingMigrationState::Running;
        running.updated_at = now;
        self.storage.save_profile_migration_checkpoint(&running)?;
        let mut newly_completed = 0_u64;
        let mut failed = BTreeSet::new();
        let mut completed = 0_u64;
        let mut last_memory_id = None;
        for node in &nodes {
            let node_uuid = Uuid::parse_str(&node.id).map_err(|_| {
                EmbeddingLifecycleError::InvalidFixture(format!(
                    "snapshot contains non-UUID knowledge node ID '{}'",
                    node.id
                ))
            })?;
            last_memory_id = Some(node_uuid);
            // A previously persisted correctly-sized destination vector is a
            // durable completion; never read or overwrite a source vector.
            let already_done = self
                .storage
                .embedding_profile_vector(destination, &node.id)?
                .is_some_and(|vector| {
                    vector.dimensions as usize == destination_manifest.profile.embedding_dimension
                });
            if already_done {
                completed += 1;
                continue;
            }
            match embed_one(
                &embedder,
                &node.content,
                destination_manifest.profile.embedding_dimension,
            ) {
                Ok(embedding) => {
                    let vector = EmbeddingProfileVector {
                        profile_id: destination.to_string(),
                        node_id: node.id.clone(),
                        embedding: f32_bytes(&embedding),
                        dimensions: embedding.len() as u32,
                        model: destination_manifest.profile.model_id.clone(),
                        created_at: Utc::now(),
                    };
                    self.storage
                        .put_embedding_profile_vector_with_migration_checkpoint(
                            &vector,
                            &EmbeddingProfileMigrationNodeCheckpoint {
                                migration_id: migration_id.to_string(),
                                node_id: node.id.clone(),
                                state: "completed".to_string(),
                                error: None,
                                updated_at: Utc::now(),
                            },
                        )?;
                    completed += 1;
                    newly_completed += 1;
                }
                Err(error) => {
                    failed.insert(node_uuid);
                    self.storage
                        .save_embedding_profile_migration_node_checkpoint(
                            &EmbeddingProfileMigrationNodeCheckpoint {
                                migration_id: migration_id.to_string(),
                                node_id: node.id.clone(),
                                state: "failed".to_string(),
                                error: Some(error.to_string()),
                                updated_at: Utc::now(),
                            },
                        )?;
                }
            }
            let mut progress = running.clone();
            progress.completed_memories = completed;
            progress.failed_memory_ids = failed.iter().copied().collect();
            progress.last_memory_id = last_memory_id;
            progress.updated_at = Utc::now();
            self.storage.save_profile_migration_checkpoint(&progress)?;
            running = progress;
            if interrupt_after.is_some_and(|limit| newly_completed >= limit) {
                running.state = EmbeddingMigrationState::Paused;
                running.updated_at = Utc::now();
                self.storage.save_profile_migration_checkpoint(&running)?;
                return Err(EmbeddingLifecycleError::Interrupted {
                    migration_id,
                    completed: newly_completed,
                });
            }
        }

        // Recompute from destination rows rather than trusting counters. This
        // makes a crash between vector and work-item writes safely resumable.
        let missing = nodes
            .iter()
            .filter(|node| {
                self.storage
                    .embedding_profile_vector(destination, &node.id)
                    .ok()
                    .flatten()
                    .is_none()
            })
            .count() as u64;
        if !failed.is_empty() || missing != 0 {
            running.state = EmbeddingMigrationState::Failed;
            running.completed_memories = total.saturating_sub(missing);
            running.failed_memory_ids = failed.iter().copied().collect();
            running.updated_at = Utc::now();
            self.storage.save_profile_migration_checkpoint(&running)?;
            return Ok(EmbeddingLifecycleMigrationReceipt {
                migration_id,
                state: running.state,
                snapshot_path: paths.relative_path,
                snapshot_sha256,
                corpus_sha256,
                total_memories: total,
                completed_memories: running.completed_memories,
                failed_memory_ids: running.failed_memory_ids,
                active_profile_unchanged: self
                    .storage
                    .active_embedding_profile()?
                    .map(|v| v.profile_id)
                    == active_before,
                integrity_hash: None,
            });
        }
        let vector_hash = destination_vector_hash(self.storage, destination, &nodes)?;
        let sidecar = build_and_verify_sidecar(
            self.storage,
            destination,
            destination_manifest.profile.embedding_dimension,
            &nodes,
        )?;
        let integrity = EmbeddingProfileIntegrityManifest {
            profile_id: destination.to_string(),
            manifest_json: serde_json::json!({
                "migration_id": migration_id,
                "snapshot_sha256": snapshot_sha256,
                "corpus_sha256": corpus_sha256,
                "vector_sha256": vector_hash,
                "sidecar": sidecar.relative_path,
                "sidecar_sha256": sidecar.integrity_hash,
                "index": "hnsw-usearch-profile-sidecar/v1",
            }),
            manifest_hash: destination_manifest.manifest_hash(),
            vector_count: total,
            index_member_count: total,
            index_integrity_hash: Some(sidecar.integrity_hash.clone()),
            updated_at: Utc::now(),
        };
        self.storage
            .save_embedding_profile_integrity_manifest(&integrity)?;
        running.state = EmbeddingMigrationState::Completed;
        running.completed_memories = total;
        running.failed_memory_ids.clear();
        running.last_memory_id = last_memory_id;
        running.updated_at = Utc::now();
        self.storage.save_profile_migration_checkpoint(&running)?;
        let active_after = self
            .storage
            .active_embedding_profile()?
            .map(|value| value.profile_id);
        if active_after != active_before {
            return Err(EmbeddingLifecycleError::Storage(
                StorageError::InvalidEmbeddingProfile(
                    "migration changed the active profile pointer".to_string(),
                ),
            ));
        }
        Ok(EmbeddingLifecycleMigrationReceipt {
            migration_id,
            state: running.state,
            snapshot_path: paths.relative_path,
            snapshot_sha256,
            corpus_sha256,
            total_memories: total,
            completed_memories: total,
            failed_memory_ids: Vec::new(),
            active_profile_unchanged: true,
            integrity_hash: integrity.index_integrity_hash,
        })
    }

    #[cfg(feature = "qwen3-embeddings")]
    pub fn migrate_qwen3_local(
        &self,
        destination: &EmbeddingProfileId,
        source: &EmbeddingProfileId,
        artifact_root: &Path,
        migration_id: Option<Uuid>,
        interrupt_after: Option<u64>,
    ) -> Result<EmbeddingLifecycleMigrationReceipt, EmbeddingLifecycleError> {
        let profile = self.installed_profile(destination)?;
        self.register_qwen3_local_verified(profile, artifact_root)?;
        self.migrate_registered(destination, source, migration_id, interrupt_after)
    }

    #[cfg(not(feature = "qwen3-embeddings"))]
    pub fn migrate_qwen3_local(
        &self,
        _destination: &EmbeddingProfileId,
        _source: &EmbeddingProfileId,
        _artifact_root: &Path,
        _migration_id: Option<Uuid>,
        _interrupt_after: Option<u64>,
    ) -> Result<EmbeddingLifecycleMigrationReceipt, EmbeddingLifecycleError> {
        Err(EmbeddingLifecycleError::RunnerUnavailable)
    }

    #[cfg(feature = "embeddings")]
    fn installed_profile(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<EmbeddingProfile, EmbeddingLifecycleError> {
        let manifest = self
            .storage
            .embedding_profile_manifest(profile_id)?
            .ok_or_else(|| EmbeddingLifecycleError::NotInstalled(profile_id.clone()))?;
        ensure_locally_verified(&manifest)?;
        Ok(manifest.profile)
    }

    #[cfg(feature = "embeddings")]
    fn register_granite_onnx_verified(
        &self,
        profile: EmbeddingProfile,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        let artifacts = Self::verified_artifacts(&profile, artifact_root)?;
        let runner = Arc::new(
            crate::embedder::GraniteOnnxEmbedder::from_verified_local_artifacts(
                profile.clone(),
                &artifacts,
            )?,
        );
        self.registry
            .install_verified(
                profile.clone(),
                &artifacts,
                granite_runtime(&profile),
                runner,
            )
            .map_err(Into::into)
    }

    #[cfg(feature = "qwen3-embeddings")]
    fn register_qwen3_local_verified(
        &self,
        profile: EmbeddingProfile,
        artifact_root: &Path,
    ) -> Result<EmbeddingProfileManifest, EmbeddingLifecycleError> {
        let artifacts = Self::verified_artifacts(&profile, artifact_root)?;
        let runner = Arc::new(
            crate::embedder::Qwen3LocalEmbedder::from_verified_local_artifacts(
                profile.clone(),
                &artifacts,
            )?,
        );
        self.registry
            .install_verified(profile.clone(), &artifacts, local_runtime(&profile), runner)
            .map_err(Into::into)
    }

    fn write_evaluation_report(
        &self,
        profile_id: &EmbeddingProfileId,
        id: Uuid,
        report: &serde_json::Value,
    ) -> Result<(), EmbeddingLifecycleError> {
        let path = self
            .storage
            .sidecar_dir("embedding-profiles")
            .join("evaluations")
            .join(profile_id.as_str());
        fs::create_dir_all(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        }
        let report_path = path.join(format!("{id}.json"));
        fs::write(&report_path, serde_json::to_vec_pretty(report)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&report_path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }
}

#[cfg(feature = "embeddings")]
fn granite_runtime(profile: &EmbeddingProfile) -> EmbeddingRuntimeMetadata {
    use super::EmbeddingDevice;
    EmbeddingRuntimeMetadata {
        backend: profile.runtime_backend,
        device: EmbeddingDevice::Cpu,
        runtime_version: "fastembed=5.13.2;user-defined-onnx".to_string(),
        initialized_at: Utc::now(),
        local_only: true,
    }
}

#[cfg(feature = "qwen3-embeddings")]
fn local_runtime(profile: &EmbeddingProfile) -> EmbeddingRuntimeMetadata {
    use super::EmbeddingDevice;
    EmbeddingRuntimeMetadata {
        backend: profile.runtime_backend,
        device: EmbeddingDevice::Cpu,
        runtime_version: "fastembed=5.13.4;candle=0.10.2".to_string(),
        initialized_at: Utc::now(),
        local_only: true,
    }
}

fn ensure_locally_verified(
    manifest: &EmbeddingProfileManifest,
) -> Result<(), EmbeddingLifecycleError> {
    if manifest.verification.status != super::VerificationStatus::Verified
        || manifest
            .runtime
            .as_ref()
            .is_none_or(|runtime| !runtime.local_only)
    {
        return Err(EmbeddingLifecycleError::NotInstalled(
            manifest.profile.profile_id.clone(),
        ));
    }
    Ok(())
}

struct MigrationPaths {
    snapshot: PathBuf,
    relative_path: String,
}
impl MigrationPaths {
    fn new(storage: &Storage, migration_id: Uuid) -> Result<Self, EmbeddingLifecycleError> {
        let relative_path = format!("embedding-profiles/migrations/{migration_id}/snapshot.sqlite");
        Ok(Self {
            snapshot: storage.data_dir().join(&relative_path),
            relative_path,
        })
    }
    fn create_parent(&self) -> Result<(), EmbeddingLifecycleError> {
        let parent = self.snapshot.parent().expect("snapshot has parent");
        fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
        }
        Ok(())
    }
}

fn all_snapshot_nodes(
    snapshot: &Storage,
) -> Result<Vec<crate::memory::KnowledgeNode>, EmbeddingLifecycleError> {
    let mut offset = 0;
    let mut nodes = Vec::new();
    loop {
        let page = snapshot.get_all_nodes(500, offset)?;
        if page.is_empty() {
            break;
        }
        offset += page.len() as i32;
        nodes.extend(page);
    }
    Ok(nodes)
}

fn embed_one(
    embedder: &ProfiledEmbedder,
    content: &str,
    dimensions: usize,
) -> Result<Vec<f32>, EmbeddingLifecycleError> {
    let runtime = tokio::runtime::Runtime::new().map_err(EmbeddingLifecycleError::Filesystem)?;
    let vector = runtime.block_on(embedder.embed_document(content))?;
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(EmbeddingLifecycleError::Embedder(
            EmbedderError::EmbedFailed(format!(
                "expected {dimensions} finite dimensions, got {}",
                vector.len()
            )),
        ));
    }
    Ok(vector)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}
fn corpus_hash(nodes: &[crate::memory::KnowledgeNode]) -> String {
    let mut h = Sha256::new();
    for node in nodes {
        h.update(node.id.as_bytes());
        h.update([0]);
        h.update(node.content.as_bytes());
        h.update([0]);
    }
    format!("{:x}", h.finalize())
}
fn destination_vector_hash(
    storage: &Storage,
    profile: &EmbeddingProfileId,
    nodes: &[crate::memory::KnowledgeNode],
) -> Result<String, EmbeddingLifecycleError> {
    let mut h = Sha256::new();
    for node in nodes {
        let vector = storage
            .embedding_profile_vector(profile, &node.id)?
            .ok_or_else(|| {
                EmbeddingLifecycleError::InvalidFixture(format!(
                    "missing destination vector for {}",
                    node.id
                ))
            })?;
        h.update(node.id.as_bytes());
        h.update([0]);
        h.update(&vector.embedding);
        h.update([0]);
    }
    Ok(format!("{:x}", h.finalize()))
}

struct SidecarReceipt {
    relative_path: String,
    integrity_hash: String,
}

#[cfg(feature = "vector-search")]
fn build_and_verify_sidecar(
    storage: &Storage,
    profile: &EmbeddingProfileId,
    dimensions: usize,
    nodes: &[crate::memory::KnowledgeNode],
) -> Result<SidecarReceipt, EmbeddingLifecycleError> {
    use crate::search::{VectorIndex, VectorIndexConfig};

    let directory = storage.embedding_profile_index_dir(profile)?;
    fs::create_dir_all(&directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    }
    let final_path = directory.join("index.usearch");
    let staged_path = directory.join("index.staged.usearch");
    let staged_mappings = staged_path.with_extension("mappings.json");
    let final_mappings = final_path.with_extension("mappings.json");
    let _ = fs::remove_file(&staged_path);
    let _ = fs::remove_file(&staged_mappings);
    let config = VectorIndexConfig {
        dimensions,
        ..VectorIndexConfig::default()
    };
    let mut index = VectorIndex::with_config(config.clone()).map_err(|error| {
        EmbeddingLifecycleError::InvalidFixture(format!("create destination HNSW sidecar: {error}"))
    })?;
    index.reserve(nodes.len()).map_err(|error| {
        EmbeddingLifecycleError::InvalidFixture(format!(
            "reserve destination HNSW sidecar: {error}"
        ))
    })?;
    for node in nodes {
        let row = storage
            .embedding_profile_vector(profile, &node.id)?
            .ok_or_else(|| {
                EmbeddingLifecycleError::InvalidFixture(format!(
                    "missing destination vector for sidecar: {}",
                    node.id
                ))
            })?;
        let vector = bytes_to_f32(&row.embedding, dimensions)?;
        index.add(&node.id, &vector).map_err(|error| {
            EmbeddingLifecycleError::InvalidFixture(format!("add destination HNSW vector: {error}"))
        })?;
    }
    index.save(&staged_path).map_err(|error| {
        EmbeddingLifecycleError::InvalidFixture(format!("save destination HNSW sidecar: {error}"))
    })?;
    let verified = VectorIndex::load(&staged_path, config).map_err(|error| {
        EmbeddingLifecycleError::InvalidFixture(format!("reload destination HNSW sidecar: {error}"))
    })?;
    if verified.dimensions() != dimensions
        || verified.len() != nodes.len()
        || nodes.iter().any(|node| !verified.contains(&node.id))
    {
        return Err(EmbeddingLifecycleError::InvalidFixture(
            "destination HNSW sidecar failed membership/dimension verification".to_string(),
        ));
    }
    fs::rename(&staged_path, &final_path)?;
    fs::rename(&staged_mappings, &final_mappings)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&final_path, fs::Permissions::from_mode(0o600))?;
        fs::set_permissions(&final_mappings, fs::Permissions::from_mode(0o600))?;
    }
    let mut hash = Sha256::new();
    hash.update(fs::read(&final_path)?);
    hash.update([0]);
    hash.update(fs::read(&final_mappings)?);
    Ok(SidecarReceipt {
        relative_path: format!("embedding-profiles/{}/hnsw/index.usearch", profile),
        integrity_hash: format!("{:x}", hash.finalize()),
    })
}

#[cfg(not(feature = "vector-search"))]
fn build_and_verify_sidecar(
    _storage: &Storage,
    _profile: &EmbeddingProfileId,
    _dimensions: usize,
    _nodes: &[crate::memory::KnowledgeNode],
) -> Result<SidecarReceipt, EmbeddingLifecycleError> {
    Err(EmbeddingLifecycleError::SidecarUnavailable)
}

fn bytes_to_f32(bytes: &[u8], dimensions: usize) -> Result<Vec<f32>, EmbeddingLifecycleError> {
    if bytes.len() != dimensions * std::mem::size_of::<f32>() {
        return Err(EmbeddingLifecycleError::InvalidFixture(format!(
            "destination vector byte length {} does not match {dimensions} dimensions",
            bytes.len()
        )));
    }
    let vector = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(EmbeddingLifecycleError::InvalidFixture(
            "destination vector contains non-finite value".to_string(),
        ));
    }
    Ok(vector)
}

#[derive(Deserialize)]
struct FixtureManifest {
    fixture_version: String,
    files: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct FixtureMemory {
    memory_id: String,
    content: String,
}
#[derive(Deserialize)]
struct FixtureQuery {
    query_id: String,
    category: String,
    query: String,
    relevance: BTreeMap<String, i32>,
    #[serde(default)]
    expected_literals: Vec<String>,
    #[serde(default)]
    forbidden_memory_ids: Vec<String>,
}

struct EvaluationOutcome {
    evaluation_id: Uuid,
    summary: EmbeddingEvaluationSummary,
    report: serde_json::Value,
    report_hash: String,
}
fn evaluate_fixture(
    embedder: &ProfiledEmbedder,
    manifest: &EmbeddingProfileManifest,
    fixture_dir: &Path,
    compared_against: EmbeddingProfileId,
    baseline_available: bool,
) -> Result<EvaluationOutcome, EmbeddingLifecycleError> {
    let manifest_bytes = fs::read(fixture_dir.join("manifest.json"))?;
    let fixture: FixtureManifest = serde_json::from_slice(&manifest_bytes)?;
    if fixture.fixture_version != "v1" {
        return Err(EmbeddingLifecycleError::InvalidFixture(
            "only v1 Agent Memory Eval fixtures are supported".to_string(),
        ));
    }
    for file in ["corpus.jsonl", "queries.jsonl"] {
        let actual = sha256_hex(&fs::read(fixture_dir.join(file))?);
        if fixture.files.get(file) != Some(&actual) {
            return Err(EmbeddingLifecycleError::InvalidFixture(format!(
                "fixture hash mismatch for {file}"
            )));
        }
    }
    let corpus: Vec<FixtureMemory> = read_jsonl(&fixture_dir.join("corpus.jsonl"))?;
    let queries: Vec<FixtureQuery> = read_jsonl(&fixture_dir.join("queries.jsonl"))?;
    if corpus.is_empty() || queries.is_empty() {
        return Err(EmbeddingLifecycleError::InvalidFixture(
            "fixture corpus and queries must be non-empty".to_string(),
        ));
    }
    let start = Instant::now();
    let corpus_texts = corpus
        .iter()
        .map(|memory| memory.content.as_str())
        .collect::<Vec<_>>();
    let runtime = tokio::runtime::Runtime::new().map_err(EmbeddingLifecycleError::Filesystem)?;
    let corpus_vectors = runtime.block_on(embedder.embed_document_batch(&corpus_texts))?;
    if corpus_vectors.len() != corpus.len() {
        return Err(EmbeddingLifecycleError::Embedder(
            EmbedderError::EmbedFailed("embedder returned an incomplete corpus batch".to_string()),
        ));
    }
    let mut rankings = BTreeMap::new();
    let mut query_latencies = Vec::new();
    for query in &queries {
        let begin = Instant::now();
        let vector = runtime.block_on(embedder.embed_query(&query.query))?;
        query_latencies.push(begin.elapsed().as_millis() as u64);
        let mut scored = corpus
            .iter()
            .zip(&corpus_vectors)
            .map(|(memory, document)| (memory.memory_id.clone(), cosine(&vector, document)))
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rankings.insert(query.query_id.clone(), scored);
    }
    let report = score_fixture(
        &corpus,
        &queries,
        &rankings,
        fixture.fixture_version,
        sha256_hex(&manifest_bytes),
        manifest,
        baseline_available,
    )?;
    let report_hash = sha256_hex(&serde_json::to_vec(&report)?);
    let overall = report.get("overall").ok_or_else(|| {
        EmbeddingLifecycleError::InvalidFixture("evaluator produced no overall metrics".to_string())
    })?;
    let mut latencies = query_latencies;
    latencies.sort_unstable();
    let p = |pct: usize| {
        latencies
            .get((latencies.len().saturating_sub(1) * pct) / 100)
            .copied()
    };
    let summary = EmbeddingEvaluationSummary {
        evaluation_id: Uuid::new_v4(),
        compared_against,
        completed_at: Utc::now(),
        corpus_size: corpus.len() as u64,
        recall_at_5: number(overall, "recall_at_5"),
        recall_at_10: number(overall, "recall_at_10"),
        ndcg_at_10: number(overall, "ndcg_at_10"),
        exact_match_preservation: number(overall, "exact_match_preservation_at_5"),
        false_positive_rate: number(overall, "false_positive_retrieval_rate_at_5"),
        p50_query_latency_ms: p(50),
        p95_query_latency_ms: p(95),
        ingestion_throughput_per_second: Some(
            corpus.len() as f64 / start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE),
        ),
        report_hash: report_hash.clone(),
    };
    Ok(EvaluationOutcome {
        evaluation_id: summary.evaluation_id,
        summary,
        report,
        report_hash,
    })
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<Vec<T>, EmbeddingLifecycleError> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum::<f64>();
    let an = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let bn = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if an > 0.0 && bn > 0.0 {
        dot / (an * bn)
    } else {
        f64::NEG_INFINITY
    }
}
fn number(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64()
}

fn score_fixture(
    corpus: &[FixtureMemory],
    queries: &[FixtureQuery],
    rankings: &BTreeMap<String, Vec<(String, f64)>>,
    fixture_version: String,
    fixture_manifest_sha256: String,
    profile_manifest: &EmbeddingProfileManifest,
    baseline_available: bool,
) -> Result<serde_json::Value, EmbeddingLifecycleError> {
    let by_id = corpus
        .iter()
        .map(|memory| (memory.memory_id.as_str(), memory.content.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut categories: BTreeMap<String, MetricCounts> = BTreeMap::new();
    let mut failures = Vec::new();
    for query in queries {
        let ranked = rankings.get(&query.query_id).ok_or_else(|| {
            EmbeddingLifecycleError::InvalidFixture(format!(
                "missing ranking for {}",
                query.query_id
            ))
        })?;
        let count = categories.entry(query.category.clone()).or_default();
        count.queries += 1;
        let ids = ranked.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>();
        for (k, target) in [
            (5, &mut count.recall_hits_5),
            (10, &mut count.recall_hits_10),
        ] {
            if ids
                .iter()
                .take(k)
                .any(|id| query.relevance.contains_key(*id))
            {
                *target += 1;
            }
        }
        let received = ids
            .iter()
            .take(10)
            .map(|id| *query.relevance.get(*id).unwrap_or(&0))
            .collect::<Vec<_>>();
        let mut ideal = query.relevance.values().copied().collect::<Vec<_>>();
        ideal.sort_by(|a, b| b.cmp(a));
        count.ndcg += dcg(&received) / dcg(&ideal).max(f64::MIN_POSITIVE);
        if !query.expected_literals.is_empty() {
            count.exact_queries += 1;
            if ids.iter().take(5).any(|id| {
                query.relevance.contains_key(*id)
                    && query.expected_literals.iter().all(|literal| {
                        by_id
                            .get(*id)
                            .is_some_and(|content| content.contains(literal))
                    })
            }) {
                count.exact_hits += 1;
            }
        }
        let top = ids.iter().take(5).collect::<Vec<_>>();
        count.top_five_positions += top.len() as u64;
        count.forbidden_hits += top
            .iter()
            .filter(|id| {
                query
                    .forbidden_memory_ids
                    .iter()
                    .any(|forbidden| forbidden == **id)
            })
            .count() as u64;
        if query.category == "duplicate_near_miss" {
            count.duplicate_positions += top.len() as u64;
            count.duplicate_hits += top
                .iter()
                .filter(|id| {
                    query
                        .forbidden_memory_ids
                        .iter()
                        .any(|forbidden| forbidden == **id)
                })
                .count() as u64;
        }
        let no_relevant = !ids
            .iter()
            .take(5)
            .any(|id| query.relevance.contains_key(*id));
        let forbidden = top.iter().any(|id| {
            query
                .forbidden_memory_ids
                .iter()
                .any(|forbidden| forbidden == **id)
        });
        if no_relevant || forbidden {
            let mut failure_codes = Vec::new();
            if no_relevant {
                failure_codes.push("no_relevant_result_at_5");
            }
            if forbidden {
                failure_codes.push("forbidden_result_at_5");
            }
            failures.push(serde_json::json!({"query_id":query.query_id,"category":query.category,"failures": failure_codes, "top_five":top}));
        }
    }
    let overall = categories
        .values()
        .fold(MetricCounts::default(), |mut total, next| {
            total.add(next);
            total
        });
    Ok(
        serde_json::json!({ "spec_version":"agent-memory-eval/v1", "fixture_version":fixture_version, "fixture_manifest_sha256":fixture_manifest_sha256, "profile_manifest":profile_manifest, "baseline": {"available":baseline_available, "note": if baseline_available {"baseline comparison supplied by caller"} else {"no baseline runtime was supplied; candidate-only metrics"}}, "overall": overall.value(), "by_category": categories.into_iter().map(|(name, value)| (name, value.value())).collect::<BTreeMap<_,_>>(), "raw_rankings":rankings, "failures":failures }),
    )
}
#[derive(Default, Clone)]
struct MetricCounts {
    queries: u64,
    recall_hits_5: u64,
    recall_hits_10: u64,
    ndcg: f64,
    exact_queries: u64,
    exact_hits: u64,
    top_five_positions: u64,
    forbidden_hits: u64,
    duplicate_positions: u64,
    duplicate_hits: u64,
}
impl MetricCounts {
    fn add(&mut self, other: &Self) {
        self.queries += other.queries;
        self.recall_hits_5 += other.recall_hits_5;
        self.recall_hits_10 += other.recall_hits_10;
        self.ndcg += other.ndcg;
        self.exact_queries += other.exact_queries;
        self.exact_hits += other.exact_hits;
        self.top_five_positions += other.top_five_positions;
        self.forbidden_hits += other.forbidden_hits;
        self.duplicate_positions += other.duplicate_positions;
        self.duplicate_hits += other.duplicate_hits;
    }
    fn value(&self) -> serde_json::Value {
        let ratio = |a, b| {
            if b == 0 {
                serde_json::Value::Null
            } else {
                serde_json::json!(a as f64 / b as f64)
            }
        };
        serde_json::json!({"queries":self.queries,"recall_at_5":ratio(self.recall_hits_5,self.queries),"recall_at_10":ratio(self.recall_hits_10,self.queries),"ndcg_at_10":if self.queries==0 {serde_json::Value::Null} else {serde_json::json!(self.ndcg/self.queries as f64)},"exact_match_preservation_at_5":ratio(self.exact_hits,self.exact_queries),"false_positive_retrieval_rate_at_5":ratio(self.forbidden_hits,self.top_five_positions),"duplicate_near_miss_retrieval_rate_at_5":ratio(self.duplicate_hits,self.duplicate_positions)})
    }
}
fn dcg(values: &[i32]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, v)| (2_f64.powi(*v) - 1.0) / ((i + 2) as f64).log2())
        .sum()
}

#[cfg(all(test, feature = "embeddings", feature = "vector-search"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        embedding::{
            ChunkingStrategy, EmbeddingNormalization, EmbeddingRuntimeBackend, EncodingTemplate,
            ModelArtifactHash,
        },
        memory::IngestInput,
    };

    struct TinyEmbedder;
    impl crate::embedder::EmbedderSend for TinyEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            // Deterministic finite values are enough to exercise the isolation,
            // snapshot/resume, and sidecar gates without a model download.
            Ok(vec![text.len() as f32 + 1.0, 1.0])
        }
        fn model_name(&self) -> &str {
            "test-local-runner"
        }
        fn dimension(&self) -> usize {
            2
        }
        fn model_hash(&self) -> String {
            "0".repeat(64)
        }
        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            let mut result = Vec::with_capacity(texts.len());
            for text in texts {
                result.push(vec![text.len() as f32 + 1.0, 1.0]);
            }
            Ok(result)
        }
    }

    fn profile(artifact: ModelArtifactHash) -> EmbeddingProfile {
        EmbeddingProfile {
            profile_id: EmbeddingProfileId::new("test-local-profile-2d").unwrap(),
            display_name: "Test Local Profile".to_string(),
            model_id: "test/local".to_string(),
            immutable_model_revision: "immutable-test-revision".to_string(),
            verified_model_artifact_hashes: vec![artifact],
            runtime_backend: EmbeddingRuntimeBackend::FastembedCandle,
            embedding_dimension: 2,
            normalization_method: EmbeddingNormalization::L2,
            document_encoding_template: EncodingTemplate::Raw,
            query_encoding_template: EncodingTemplate::Raw,
            maximum_token_limit: 64,
            chunking_strategy: ChunkingStrategy::WholeDocument,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn migration_is_resumable_isolated_and_builds_a_verified_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(temp.path().join("store.sqlite"))).unwrap();
        storage
            .ingest(IngestInput {
                content: "first migration memory".to_string(),
                ..Default::default()
            })
            .unwrap();
        storage
            .ingest(IngestInput {
                content: "second migration memory".to_string(),
                ..Default::default()
            })
            .unwrap();
        let source = EmbeddingProfileId::new("nomic-v1.5-legacy-raw-256").unwrap();
        // Seed the source profile explicitly: a migration must never delete
        // an existing rollback vector, but it also must not invent one for a
        // node whose historic Nomic embedding was never generated.
        for node in storage.get_all_nodes(10, 0).unwrap() {
            storage
                .put_embedding_profile_vector(&EmbeddingProfileVector {
                    profile_id: source.to_string(),
                    node_id: node.id,
                    embedding: f32_bytes(&vec![0.25; 256]),
                    dimensions: 256,
                    model: "nomic-ai/nomic-embed-text-v1.5".to_string(),
                    created_at: Utc::now(),
                })
                .unwrap();
        }
        let artifact_path = temp.path().join("runner.bin");
        fs::write(&artifact_path, b"local test artifact").unwrap();
        let artifact = ModelArtifactHash::sha256("runner.bin", sha256_hex(b"local test artifact"));
        let profile = profile(artifact.clone());
        let artifacts = vec![VerifiedLocalArtifact::from_root(artifact, temp.path()).unwrap()];
        let lifecycle = EmbeddingProfileLifecycle::new(&storage);
        lifecycle
            .install_verified(
                profile.clone(),
                &artifacts,
                EmbeddingRuntimeMetadata {
                    backend: EmbeddingRuntimeBackend::FastembedCandle,
                    device: crate::embedding::EmbeddingDevice::Cpu,
                    runtime_version: "test".to_string(),
                    initialized_at: Utc::now(),
                    local_only: true,
                },
                Arc::new(TinyEmbedder),
            )
            .unwrap();
        let evaluation = EmbeddingEvaluationSummary {
            evaluation_id: Uuid::new_v4(),
            compared_against: EmbeddingProfileId::new("nomic-v1.5-legacy-raw-256").unwrap(),
            completed_at: Utc::now(),
            corpus_size: 0,
            recall_at_5: None,
            recall_at_10: None,
            ndcg_at_10: None,
            exact_match_preservation: None,
            false_positive_rate: None,
            p50_query_latency_ms: None,
            p95_query_latency_ms: None,
            ingestion_throughput_per_second: None,
            report_hash: "1".repeat(64),
        };
        let ready = lifecycle
            .registry
            .record_evaluation(&profile.profile_id, evaluation)
            .unwrap();
        storage.save_embedding_profile_manifest(&ready).unwrap();
        let migration_id = Uuid::new_v4();
        let interrupted = lifecycle
            .migrate_registered(&profile.profile_id, &source, Some(migration_id), Some(1))
            .unwrap_err();
        assert!(
            matches!(interrupted, EmbeddingLifecycleError::Interrupted { migration_id: id, .. } if id == migration_id)
        );
        assert_eq!(
            storage
                .active_embedding_profile()
                .unwrap()
                .unwrap()
                .profile_id,
            source
        );
        let receipt = lifecycle
            .migrate_registered(&profile.profile_id, &source, Some(migration_id), None)
            .unwrap();
        assert_eq!(receipt.state, EmbeddingMigrationState::Completed);
        assert!(receipt.active_profile_unchanged);
        assert_eq!(receipt.total_memories, 2);
        assert!(
            storage
                .embedding_profile_index_dir(&profile.profile_id)
                .unwrap()
                .join("index.usearch")
                .is_file()
        );
        assert!(
            storage
                .embedding_profile_vector(&source, &storage.get_all_nodes(1, 0).unwrap()[0].id)
                .unwrap()
                .is_some(),
            "migration must retain the source profile vector so rollback can switch pointers without re-embedding"
        );
        assert!(
            storage
                .embedding_profile_manifest(&profile.profile_id)
                .unwrap()
                .unwrap()
                .evaluation
                .is_some()
        );
        storage
            .activate_embedding_profile(&profile.profile_id)
            .unwrap();
        // Candle-backed profiles must not silently query through the legacy
        // service after activation. They become usable only after the
        // process-local verified runner is attached.
        assert!(!storage.is_embedding_ready());
        lifecycle
            .attach_registered_active_profile(&profile.profile_id)
            .unwrap();
        assert!(storage.is_embedding_ready());
        assert!(
            !storage
                .semantic_search("first memory", 5, -1.0)
                .unwrap()
                .is_empty()
        );
    }
}
