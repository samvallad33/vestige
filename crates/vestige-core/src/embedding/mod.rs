//! Durable, local-first embedding profile contracts.
//!
//! A profile identifies the *meaning* of a vector, not merely the model that
//! produced it.  In particular, document and query encoding are part of the
//! identity.  Storage and search must only compare vectors which share the
//! same [`EmbeddingProfileId`].
//!
//! This module deliberately contains no downloader, environment-variable
//! selector, or activation side effect.  Installing, evaluating, migrating,
//! and activating a profile are separate operations owned by higher layers.

mod lifecycle;
mod profile;

pub use lifecycle::{
    EmbeddingLifecycleError, EmbeddingLifecycleEvaluationReceipt,
    EmbeddingLifecycleMigrationReceipt, EmbeddingProfileLifecycle,
};

pub use profile::{
    ActiveEmbeddingProfile, BuiltinEmbeddingProfile, ChunkingStrategy,
    EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION, EmbeddingDevice, EmbeddingEvaluationSummary,
    EmbeddingMigrationState, EmbeddingNormalization, EmbeddingProfile, EmbeddingProfileError,
    EmbeddingProfileFailure, EmbeddingProfileId, EmbeddingProfileManifest, EmbeddingProfileState,
    EmbeddingRuntimeBackend, EmbeddingRuntimeMetadata, EmbeddingVerification, EncodingTemplate,
    ModelArtifactHash, ProfileMigrationCheckpoint, ProfileRuntimeRegistry, ProfiledEmbedder,
    VerificationStatus, VerifiedLocalArtifact, builtin_embedding_profile_by_id,
    builtin_embedding_profiles,
};
