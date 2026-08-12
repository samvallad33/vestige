use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
};

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::embedder::{Embedder, EmbedderError, EmbedderResult};

/// Schema version for a persisted [`EmbeddingProfileManifest`].
pub const EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// A stable identifier for a vector-space and encoding contract.
///
/// A profile ID, rather than a model ID, is the foreign key that identifies a
/// stored vector. Vectors from distinct profile IDs are not comparable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingProfileId(pub String);

impl EmbeddingProfileId {
    pub fn new(value: impl Into<String>) -> Result<Self, EmbeddingProfileError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EmbeddingProfileError::InvalidProfileId(
                "profile ID cannot be empty".to_string(),
            ));
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'.')
        }) {
            return Err(EmbeddingProfileError::InvalidProfileId(
                "profile ID must use lowercase ASCII letters, digits, hyphens, and dots"
                    .to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EmbeddingProfileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<EmbeddingProfileId> for String {
    fn from(value: EmbeddingProfileId) -> Self {
        value.0
    }
}

/// Encoding instructions which are persisted as part of the profile identity.
///
/// `Template` supports exactly one `{input}` placeholder.  This prevents a
/// model instruction from being changed invisibly at call sites.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EncodingTemplate {
    Raw,
    Prefix { prefix: String },
    Template { template: String },
}

impl EncodingTemplate {
    pub fn render(&self, input: &str) -> Result<String, EmbeddingProfileError> {
        match self {
            Self::Raw => Ok(input.to_string()),
            Self::Prefix { prefix } => Ok(format!("{prefix}{input}")),
            Self::Template { template } => {
                let placeholders = template.matches("{input}").count();
                if placeholders != 1 {
                    return Err(EmbeddingProfileError::InvalidEncodingTemplate(
                        "template must contain exactly one `{input}` placeholder".to_string(),
                    ));
                }
                Ok(template.replacen("{input}", input, 1))
            }
        }
    }

    fn validate(&self) -> Result<(), EmbeddingProfileError> {
        match self {
            Self::Prefix { prefix } if prefix.is_empty() => {
                Err(EmbeddingProfileError::InvalidEncodingTemplate(
                    "prefix cannot be empty".to_string(),
                ))
            }
            Self::Template { .. } => self.render("").map(|_| ()),
            _ => Ok(()),
        }
    }
}

/// Normalization applied before a vector is stored or searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingNormalization {
    L2,
    None,
}

/// The local inference implementation required by a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingRuntimeBackend {
    FastembedOnnx,
    FastembedCandle,
}

/// Document chunking is profile identity because it changes vector meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChunkingStrategy {
    WholeDocument,
    CharacterWindow {
        max_characters: usize,
        overlap_characters: usize,
    },
}

impl ChunkingStrategy {
    fn validate(&self) -> Result<(), EmbeddingProfileError> {
        match self {
            Self::WholeDocument => Ok(()),
            Self::CharacterWindow {
                max_characters,
                overlap_characters,
            } if *max_characters == 0 || *overlap_characters >= *max_characters => {
                Err(EmbeddingProfileError::InvalidChunkingStrategy(
                    "window size must be positive and overlap must be smaller than the window"
                        .to_string(),
                ))
            }
            Self::CharacterWindow { .. } => Ok(()),
        }
    }
}

/// A hash for one downloaded model or tokenizer artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactHash {
    /// Relative artifact name, never an absolute local path.
    pub artifact: String,
    /// Hash algorithm, currently expected to be `sha256`.
    pub algorithm: String,
    /// Lowercase hexadecimal digest.
    pub digest: String,
}

impl ModelArtifactHash {
    pub fn sha256(artifact: impl Into<String>, digest: impl Into<String>) -> Self {
        Self {
            artifact: artifact.into(),
            algorithm: "sha256".to_string(),
            digest: digest.into(),
        }
    }

    fn validate(&self) -> Result<(), EmbeddingProfileError> {
        let path = Path::new(&self.artifact);
        if self.artifact.trim().is_empty()
            || self.artifact.split('/').any(str::is_empty)
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::CurDir
                        | Component::ParentDir
                        | Component::RootDir
                        | Component::Prefix(_)
                )
            })
            || self
                .artifact
                .chars()
                .any(|character| character == '\\' || character.is_control())
        {
            return Err(EmbeddingProfileError::InvalidArtifactHash(
                "artifact must be a normalized relative path without traversal, prefixes, backslashes, or control characters".to_string(),
            ));
        }
        if self.algorithm != "sha256"
            || self.digest.len() != 64
            || !self
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EmbeddingProfileError::InvalidArtifactHash(
                "artifact hashes must be lowercase sha256 digests".to_string(),
            ));
        }
        Ok(())
    }
}

/// Immutable identity and encoding contract for one isolated vector profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfile {
    pub profile_id: EmbeddingProfileId,
    pub display_name: String,
    pub model_id: String,
    pub immutable_model_revision: String,
    /// Hashes verified at install time. Built-in catalog profiles start empty;
    /// an installed manifest must record the hashes it actually verified.
    #[serde(default)]
    pub verified_model_artifact_hashes: Vec<ModelArtifactHash>,
    pub runtime_backend: EmbeddingRuntimeBackend,
    pub embedding_dimension: usize,
    pub normalization_method: EmbeddingNormalization,
    pub document_encoding_template: EncodingTemplate,
    pub query_encoding_template: EncodingTemplate,
    pub maximum_token_limit: usize,
    pub chunking_strategy: ChunkingStrategy,
    pub created_at: DateTime<Utc>,
}

impl EmbeddingProfile {
    pub fn validate(&self) -> Result<(), EmbeddingProfileError> {
        EmbeddingProfileId::new(self.profile_id.0.clone())?;
        if self.display_name.trim().is_empty()
            || self.model_id.trim().is_empty()
            || self.immutable_model_revision.trim().is_empty()
        {
            return Err(EmbeddingProfileError::InvalidProfile(
                "display name, model ID, and immutable model revision are required".to_string(),
            ));
        }
        if self.embedding_dimension == 0 || self.maximum_token_limit == 0 {
            return Err(EmbeddingProfileError::InvalidProfile(
                "embedding dimension and maximum token limit must be positive".to_string(),
            ));
        }
        self.document_encoding_template.validate()?;
        self.query_encoding_template.validate()?;
        self.chunking_strategy.validate()?;
        for hash in &self.verified_model_artifact_hashes {
            hash.validate()?;
        }
        Ok(())
    }

    pub fn encode_document(&self, document: &str) -> Result<String, EmbeddingProfileError> {
        if document.is_empty() {
            return Err(EmbeddingProfileError::EmptyInput("document"));
        }
        self.document_encoding_template.render(document)
    }

    pub fn encode_query(&self, query: &str) -> Result<String, EmbeddingProfileError> {
        if query.is_empty() {
            return Err(EmbeddingProfileError::EmptyInput("query"));
        }
        self.query_encoding_template.render(query)
    }

    /// Stable content hash for receipts, manifests, and migration validation.
    pub fn contract_hash(&self) -> String {
        let encoded = serde_json::to_vec(self)
            .expect("EmbeddingProfile is serializable; contract serialization cannot fail");
        blake3::hash(&encoded).to_hex().to_string()
    }
}

/// Profiles shipped in the built-in catalog. They are definitions only; using
/// a Qwen entry never downloads it or changes the active profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinEmbeddingProfile {
    NomicLegacyRaw256,
    NomicRetrieval256,
    QwenBalanced256,
    QwenBalanced1024,
    QwenMax1024,
    QwenMaxNative,
}

impl BuiltinEmbeddingProfile {
    pub const fn id(self) -> &'static str {
        match self {
            Self::NomicLegacyRaw256 => "nomic-v1.5-legacy-raw-256",
            Self::NomicRetrieval256 => "nomic-v1.5-retrieval-v1-256",
            Self::QwenBalanced256 => "qwen3-0.6b-retrieval-v1-256",
            Self::QwenBalanced1024 => "qwen3-0.6b-retrieval-v1-1024",
            Self::QwenMax1024 => "qwen3-4b-retrieval-v1-1024",
            Self::QwenMaxNative => "qwen3-4b-retrieval-v1-native",
        }
    }

    pub fn profile(self) -> EmbeddingProfile {
        let created_at = Utc
            .with_ymd_and_hms(2026, 8, 11, 0, 0, 0)
            .single()
            .expect("valid built-in profile timestamp");
        let profile_id = EmbeddingProfileId::new(self.id()).expect("valid built-in profile ID");
        let qwen_query = EncodingTemplate::Template {
            template: "Instruct: Given an agent-memory request, retrieve the most relevant confirmed memories, code facts, decisions, and incident evidence.\nQuery: {input}".to_string(),
        };

        match self {
            Self::NomicLegacyRaw256 => EmbeddingProfile {
                profile_id,
                display_name: "Nomic Compact (Legacy)".to_string(),
                model_id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
                immutable_model_revision: "fastembed-catalog-5.11".to_string(),
                verified_model_artifact_hashes: Vec::new(),
                runtime_backend: EmbeddingRuntimeBackend::FastembedOnnx,
                embedding_dimension: 256,
                normalization_method: EmbeddingNormalization::L2,
                document_encoding_template: EncodingTemplate::Raw,
                query_encoding_template: EncodingTemplate::Raw,
                maximum_token_limit: 8192,
                chunking_strategy: ChunkingStrategy::WholeDocument,
                created_at,
            },
            Self::NomicRetrieval256 => EmbeddingProfile {
                profile_id,
                display_name: "Nomic Compact".to_string(),
                model_id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
                immutable_model_revision: "fastembed-catalog-5.11".to_string(),
                verified_model_artifact_hashes: Vec::new(),
                runtime_backend: EmbeddingRuntimeBackend::FastembedOnnx,
                embedding_dimension: 256,
                normalization_method: EmbeddingNormalization::L2,
                document_encoding_template: EncodingTemplate::Prefix {
                    prefix: "search_document: ".to_string(),
                },
                query_encoding_template: EncodingTemplate::Prefix {
                    prefix: "search_query: ".to_string(),
                },
                maximum_token_limit: 8192,
                chunking_strategy: ChunkingStrategy::WholeDocument,
                created_at,
            },
            Self::QwenBalanced256 => qwen_profile(
                profile_id,
                qwen_query,
                created_at,
                QwenProfileSpec::balanced_compact(),
            ),
            Self::QwenBalanced1024 => qwen_profile(
                profile_id,
                qwen_query,
                created_at,
                QwenProfileSpec::balanced(),
            ),
            Self::QwenMax1024 => qwen_profile(
                profile_id,
                qwen_query,
                created_at,
                QwenProfileSpec::max_balanced(),
            ),
            Self::QwenMaxNative => qwen_profile(
                profile_id,
                qwen_query,
                created_at,
                QwenProfileSpec::max_native(),
            ),
        }
    }
}

struct QwenProfileSpec {
    display_name: &'static str,
    model_id: &'static str,
    embedding_dimension: usize,
    immutable_model_revision: &'static str,
    artifact_hashes: fn() -> Vec<ModelArtifactHash>,
}

impl QwenProfileSpec {
    const fn balanced_compact() -> Self {
        Self {
            display_name: "Qwen Balanced 0.6B (Compact)",
            model_id: "Qwen/Qwen3-Embedding-0.6B",
            embedding_dimension: 256,
            immutable_model_revision: QWEN_06B_REVISION,
            artifact_hashes: qwen_06b_artifacts,
        }
    }

    const fn balanced() -> Self {
        Self {
            display_name: "Qwen Balanced 0.6B",
            model_id: "Qwen/Qwen3-Embedding-0.6B",
            embedding_dimension: 1024,
            immutable_model_revision: QWEN_06B_REVISION,
            artifact_hashes: qwen_06b_artifacts,
        }
    }

    const fn max_balanced() -> Self {
        Self {
            display_name: "Qwen Max 4B (Balanced)",
            model_id: "Qwen/Qwen3-Embedding-4B",
            embedding_dimension: 1024,
            immutable_model_revision: QWEN_4B_REVISION,
            artifact_hashes: qwen_4b_artifacts,
        }
    }

    const fn max_native() -> Self {
        Self {
            display_name: "Qwen Max 4B (Native)",
            model_id: "Qwen/Qwen3-Embedding-4B",
            embedding_dimension: 2560,
            immutable_model_revision: QWEN_4B_REVISION,
            artifact_hashes: qwen_4b_artifacts,
        }
    }
}

fn qwen_profile(
    profile_id: EmbeddingProfileId,
    query_encoding_template: EncodingTemplate,
    created_at: DateTime<Utc>,
    spec: QwenProfileSpec,
) -> EmbeddingProfile {
    EmbeddingProfile {
        profile_id,
        display_name: spec.display_name.to_string(),
        model_id: spec.model_id.to_string(),
        immutable_model_revision: spec.immutable_model_revision.to_string(),
        verified_model_artifact_hashes: (spec.artifact_hashes)(),
        runtime_backend: EmbeddingRuntimeBackend::FastembedCandle,
        embedding_dimension: spec.embedding_dimension,
        normalization_method: EmbeddingNormalization::L2,
        document_encoding_template: EncodingTemplate::Raw,
        query_encoding_template,
        maximum_token_limit: 8192,
        chunking_strategy: ChunkingStrategy::WholeDocument,
        created_at,
    }
}

const QWEN_06B_REVISION: &str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3";
const QWEN_4B_REVISION: &str = "5cf2132abc99cad020ac570b19d031efec650f2b";

fn qwen_06b_artifacts() -> Vec<ModelArtifactHash> {
    vec![
        ModelArtifactHash::sha256(
            "model.safetensors",
            "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        ),
        ModelArtifactHash::sha256(
            "tokenizer.json",
            "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        ),
        ModelArtifactHash::sha256(
            "config.json",
            "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
        ),
    ]
}

fn qwen_4b_artifacts() -> Vec<ModelArtifactHash> {
    vec![
        ModelArtifactHash::sha256(
            "model-00001-of-00002.safetensors",
            "e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8",
        ),
        ModelArtifactHash::sha256(
            "model-00002-of-00002.safetensors",
            "ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1",
        ),
        ModelArtifactHash::sha256(
            "model.safetensors.index.json",
            "9d130c7f24fa1f9a2a7e19fad42c7d6d2d6fea31b180bdf3e8aac1924c26c39a",
        ),
        ModelArtifactHash::sha256(
            "tokenizer.json",
            "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
        ),
        ModelArtifactHash::sha256(
            "config.json",
            "78d2861cbbfd80eee05839200c5a3b7ed64c789f6c1cab4fbb84cc4eae33eaf5",
        ),
    ]
}

pub fn builtin_embedding_profiles() -> Vec<EmbeddingProfile> {
    [
        BuiltinEmbeddingProfile::NomicLegacyRaw256,
        BuiltinEmbeddingProfile::NomicRetrieval256,
        BuiltinEmbeddingProfile::QwenBalanced256,
        BuiltinEmbeddingProfile::QwenBalanced1024,
        BuiltinEmbeddingProfile::QwenMax1024,
        BuiltinEmbeddingProfile::QwenMaxNative,
    ]
    .into_iter()
    .map(BuiltinEmbeddingProfile::profile)
    .collect()
}

pub fn builtin_embedding_profile_by_id(id: &str) -> Option<EmbeddingProfile> {
    [
        BuiltinEmbeddingProfile::NomicLegacyRaw256,
        BuiltinEmbeddingProfile::NomicRetrieval256,
        BuiltinEmbeddingProfile::QwenBalanced256,
        BuiltinEmbeddingProfile::QwenBalanced1024,
        BuiltinEmbeddingProfile::QwenMax1024,
        BuiltinEmbeddingProfile::QwenMaxNative,
    ]
    .into_iter()
    .find(|profile| profile.id() == id)
    .map(BuiltinEmbeddingProfile::profile)
}

/// Explicit install/evaluation/migration/activation lifecycle. No transition
/// is performed implicitly by this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProfileState {
    NotInstalled,
    Installing,
    Installed,
    Evaluating,
    Migrating,
    Ready,
    Active,
    Inactive,
    RetryableError,
    RepairNeeded,
}

/// Actual runtime selected after an explicit local install.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingDevice {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRuntimeMetadata {
    pub backend: EmbeddingRuntimeBackend,
    pub device: EmbeddingDevice,
    pub runtime_version: String,
    pub initialized_at: DateTime<Utc>,
    /// The embedder is local-only when true. Built-in profiles must never use
    /// a hosted embedding endpoint.
    pub local_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    NotAttempted,
    Verified,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingVerification {
    pub status: VerificationStatus,
    #[serde(default)]
    pub verified_artifacts: Vec<ModelArtifactHash>,
    pub verified_at: Option<DateTime<Utc>>,
    pub detail: Option<String>,
}

impl EmbeddingVerification {
    pub fn not_attempted() -> Self {
        Self {
            status: VerificationStatus::NotAttempted,
            verified_artifacts: Vec::new(),
            verified_at: None,
            detail: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingEvaluationSummary {
    pub evaluation_id: Uuid,
    pub compared_against: EmbeddingProfileId,
    pub completed_at: DateTime<Utc>,
    pub corpus_size: u64,
    pub recall_at_5: Option<f64>,
    pub recall_at_10: Option<f64>,
    pub ndcg_at_10: Option<f64>,
    pub exact_match_preservation: Option<f64>,
    pub false_positive_rate: Option<f64>,
    pub p50_query_latency_ms: Option<u64>,
    pub p95_query_latency_ms: Option<u64>,
    pub ingestion_throughput_per_second: Option<f64>,
    pub report_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfileFailure {
    pub code: String,
    pub message: String,
    pub occurred_at: DateTime<Utc>,
    pub retryable: bool,
}

/// Persisted status and verification receipt for a profile contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingProfileManifest {
    pub schema_version: u32,
    pub profile: EmbeddingProfile,
    pub state: EmbeddingProfileState,
    pub installed_at: Option<DateTime<Utc>>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub runtime: Option<EmbeddingRuntimeMetadata>,
    pub verification: EmbeddingVerification,
    pub evaluation: Option<EmbeddingEvaluationSummary>,
    pub failure: Option<EmbeddingProfileFailure>,
}

impl EmbeddingProfileManifest {
    pub fn not_installed(profile: EmbeddingProfile) -> Result<Self, EmbeddingProfileError> {
        profile.validate()?;
        Ok(Self {
            schema_version: EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION,
            profile,
            state: EmbeddingProfileState::NotInstalled,
            installed_at: None,
            last_verified_at: None,
            runtime: None,
            verification: EmbeddingVerification::not_attempted(),
            evaluation: None,
            failure: None,
        })
    }

    /// Creates a durable, recoverable failure receipt. It does not retry a
    /// download or initialize a model; a user must explicitly retry or repair
    /// the profile through the caller's install workflow.
    pub fn failed(
        profile: EmbeddingProfile,
        failure: EmbeddingProfileFailure,
    ) -> Result<Self, EmbeddingProfileError> {
        profile.validate()?;
        Ok(Self {
            schema_version: EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION,
            profile,
            state: if failure.retryable {
                EmbeddingProfileState::RetryableError
            } else {
                EmbeddingProfileState::RepairNeeded
            },
            installed_at: None,
            last_verified_at: None,
            runtime: None,
            verification: EmbeddingVerification::not_attempted(),
            evaluation: None,
            failure: Some(failure),
        })
    }

    pub fn manifest_hash(&self) -> String {
        let encoded = serde_json::to_vec(self)
            .expect("EmbeddingProfileManifest is serializable; manifest serialization cannot fail");
        blake3::hash(&encoded).to_hex().to_string()
    }

    pub fn validate(&self) -> Result<(), EmbeddingProfileError> {
        if self.schema_version != EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION {
            return Err(EmbeddingProfileError::UnsupportedManifestVersion(
                self.schema_version,
            ));
        }
        self.profile.validate()?;
        if let Some(runtime) = &self.runtime
            && (!runtime.local_only || runtime.backend != self.profile.runtime_backend)
        {
            return Err(EmbeddingProfileError::InvalidManifest(
                "runtime must be local-only and match the profile backend".to_string(),
            ));
        }
        if self.verification.status == VerificationStatus::Verified
            && (self.verification.verified_at.is_none()
                || self.verification.verified_artifacts.is_empty())
        {
            return Err(EmbeddingProfileError::InvalidManifest(
                "verified profiles require a timestamp and at least one verified artifact"
                    .to_string(),
            ));
        }
        for artifact in &self.verification.verified_artifacts {
            artifact.validate()?;
        }
        Ok(())
    }
}

/// The atomic, storage-owned pointer to the profile used for semantic search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveEmbeddingProfile {
    pub profile_id: EmbeddingProfileId,
    pub activated_at: DateTime<Utc>,
    pub previous_profile_id: Option<EmbeddingProfileId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingMigrationState {
    Pending,
    Running,
    Paused,
    Validating,
    Completed,
    Failed,
    Cancelled,
}

/// Resumable migration progress. Persistence, snapshots, and repair queues are
/// intentionally implemented by storage rather than the profile contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileMigrationCheckpoint {
    pub migration_id: Uuid,
    pub source_profile_id: EmbeddingProfileId,
    pub destination_profile_id: EmbeddingProfileId,
    pub state: EmbeddingMigrationState,
    pub total_memories: u64,
    pub completed_memories: u64,
    #[serde(default)]
    pub failed_memory_ids: Vec<Uuid>,
    pub last_memory_id: Option<Uuid>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Profile-scoped encoding facade around an existing embedder.
///
/// The wrapped embedder only receives fully rendered input. This keeps legacy
/// `Embedder::embed` source-compatible while making the document/query split
/// explicit for all new retrieval paths.
pub struct ProfiledEmbedder {
    profile: EmbeddingProfile,
    inner: Arc<dyn Embedder>,
}

/// One local artifact which an explicit installer has staged for verification.
///
/// `path` is intentionally not persisted in an [`EmbeddingProfileManifest`]:
/// it can contain private local filesystem information. The manifest records
/// only the verified relative artifact names and digests.
#[derive(Debug, Clone)]
pub struct VerifiedLocalArtifact {
    pub artifact: ModelArtifactHash,
    pub path: PathBuf,
}

impl VerifiedLocalArtifact {
    pub fn new(artifact: ModelArtifactHash, path: impl Into<PathBuf>) -> Self {
        Self {
            artifact,
            path: path.into(),
        }
    }

    /// Safely resolves a manifest-relative artifact under an explicit install
    /// directory. Canonical-path containment prevents a symlink inside the
    /// model directory from escaping the verified install root.
    pub fn from_root(
        artifact: ModelArtifactHash,
        root: impl AsRef<Path>,
    ) -> Result<Self, EmbeddingProfileError> {
        artifact.validate()?;
        let root = root.as_ref().canonicalize().map_err(|error| {
            EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: artifact.artifact.clone(),
                reason: error.to_string(),
            }
        })?;
        let path = root
            .join(&artifact.artifact)
            .canonicalize()
            .map_err(|error| EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: artifact.artifact.clone(),
                reason: error.to_string(),
            })?;
        if !path.starts_with(&root) {
            return Err(EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: artifact.artifact.clone(),
                reason: "artifact resolves outside the explicit install root".to_string(),
            });
        }
        Ok(Self { artifact, path })
    }

    pub fn verify(&self) -> Result<ModelArtifactHash, EmbeddingProfileError> {
        self.artifact.validate()?;
        let metadata = std::fs::metadata(&self.path).map_err(|error| {
            EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: self.artifact.artifact.clone(),
                reason: error.to_string(),
            }
        })?;
        if !metadata.is_file() {
            return Err(EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: self.artifact.artifact.clone(),
                reason: "artifact path is not a regular file".to_string(),
            });
        }

        let mut file = File::open(&self.path).map_err(|error| {
            EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: self.artifact.artifact.clone(),
                reason: error.to_string(),
            }
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                EmbeddingProfileError::ArtifactVerificationFailed {
                    artifact: self.artifact.artifact.clone(),
                    reason: error.to_string(),
                }
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != self.artifact.digest {
            return Err(EmbeddingProfileError::ArtifactVerificationFailed {
                artifact: self.artifact.artifact.clone(),
                reason: "sha256 digest does not match the pinned manifest".to_string(),
            });
        }
        Ok(self.artifact.clone())
    }
}

struct RegisteredProfile {
    manifest: EmbeddingProfileManifest,
    embedder: Arc<ProfiledEmbedder>,
}

/// Explicit, process-local registry for verified embedding runtimes.
///
/// This is deliberately a registry, not a model manager: it never reads an
/// environment variable, probes hardware, downloads weights, or substitutes a
/// different profile. Callers first explicitly install and verify artifacts,
/// then register the runner they created. Persistent activation remains an
/// atomic storage operation represented by [`ActiveEmbeddingProfile`].
#[derive(Default)]
pub struct ProfileRuntimeRegistry {
    profiles: RwLock<HashMap<EmbeddingProfileId, RegisteredProfile>>,
}

impl ProfileRuntimeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an already-created local runner only after every pinned model
    /// artifact has been verified from disk. This operation is explicit and
    /// cannot cause a network request.
    pub fn install_verified(
        &self,
        profile: EmbeddingProfile,
        artifacts: &[VerifiedLocalArtifact],
        runtime: EmbeddingRuntimeMetadata,
        embedder: Arc<dyn Embedder>,
    ) -> Result<EmbeddingProfileManifest, EmbeddingProfileError> {
        profile.validate()?;
        if !runtime.local_only || runtime.backend != profile.runtime_backend {
            return Err(EmbeddingProfileError::InvalidManifest(
                "installed runner must be local-only and match the profile backend".to_string(),
            ));
        }
        if artifacts.is_empty() {
            return Err(EmbeddingProfileError::InvalidManifest(
                "an explicit install must verify at least one pinned artifact".to_string(),
            ));
        }

        let verified_artifacts = artifacts
            .iter()
            .map(VerifiedLocalArtifact::verify)
            .collect::<Result<Vec<_>, _>>()?;
        if profile.verified_model_artifact_hashes.is_empty() {
            return Err(EmbeddingProfileError::InvalidManifest(
                "profile must contain pinned artifact hashes before it can be installed"
                    .to_string(),
            ));
        }
        if profile.verified_model_artifact_hashes != verified_artifacts {
            return Err(EmbeddingProfileError::InvalidManifest(
                "verified artifacts do not exactly match the profile's pinned artifact manifest"
                    .to_string(),
            ));
        }

        let now = Utc::now();
        let manifest = EmbeddingProfileManifest {
            schema_version: EMBEDDING_PROFILE_MANIFEST_SCHEMA_VERSION,
            profile: profile.clone(),
            state: EmbeddingProfileState::Installed,
            installed_at: Some(now),
            last_verified_at: Some(now),
            runtime: Some(runtime),
            verification: EmbeddingVerification {
                status: VerificationStatus::Verified,
                verified_artifacts,
                verified_at: Some(now),
                detail: None,
            },
            evaluation: None,
            failure: None,
        };
        manifest.validate()?;
        let registered = RegisteredProfile {
            manifest: manifest.clone(),
            embedder: Arc::new(ProfiledEmbedder::new(profile, embedder)?),
        };
        self.profiles
            .write()
            .expect("profile runtime registry lock poisoned")
            .insert(manifest.profile.profile_id.clone(), registered);
        Ok(manifest)
    }

    /// Stores a completed evaluation receipt without changing the active
    /// profile. The storage layer decides whether and when to activate it.
    pub fn record_evaluation(
        &self,
        profile_id: &EmbeddingProfileId,
        evaluation: EmbeddingEvaluationSummary,
    ) -> Result<EmbeddingProfileManifest, EmbeddingProfileError> {
        let mut profiles = self
            .profiles
            .write()
            .expect("profile runtime registry lock poisoned");
        let registered = profiles
            .get_mut(profile_id)
            .ok_or_else(|| EmbeddingProfileError::ProfileNotInstalled(profile_id.clone()))?;
        registered.manifest.evaluation = Some(evaluation);
        registered.manifest.state = EmbeddingProfileState::Ready;
        registered.manifest.failure = None;
        Ok(registered.manifest.clone())
    }

    pub fn manifest(&self, profile_id: &EmbeddingProfileId) -> Option<EmbeddingProfileManifest> {
        self.profiles
            .read()
            .expect("profile runtime registry lock poisoned")
            .get(profile_id)
            .map(|registered| registered.manifest.clone())
    }

    /// Gets only the requested profile's encoder. It never falls back to a
    /// default or active profile, which prevents cross-profile scoring.
    pub fn embedder(
        &self,
        profile_id: &EmbeddingProfileId,
    ) -> Result<Arc<ProfiledEmbedder>, EmbeddingProfileError> {
        self.profiles
            .read()
            .expect("profile runtime registry lock poisoned")
            .get(profile_id)
            .map(|registered| registered.embedder.clone())
            .ok_or_else(|| EmbeddingProfileError::ProfileNotInstalled(profile_id.clone()))
    }
}

impl ProfiledEmbedder {
    pub fn new(
        profile: EmbeddingProfile,
        inner: Arc<dyn Embedder>,
    ) -> Result<Self, EmbeddingProfileError> {
        profile.validate()?;
        Ok(Self { profile, inner })
    }

    pub fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }

    pub async fn embed_document(&self, document: &str) -> EmbedderResult<Vec<f32>> {
        let encoded = self
            .profile
            .encode_document(document)
            .map_err(profile_error_to_embedder_error)?;
        self.inner.embed(&encoded).await
    }

    pub async fn embed_query(&self, query: &str) -> EmbedderResult<Vec<f32>> {
        let encoded = self
            .profile
            .encode_query(query)
            .map_err(profile_error_to_embedder_error)?;
        self.inner.embed(&encoded).await
    }

    pub async fn embed_document_batch(&self, documents: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let encoded = documents
            .iter()
            .map(|document| {
                self.profile
                    .encode_document(document)
                    .map_err(profile_error_to_embedder_error)
            })
            .collect::<EmbedderResult<Vec<_>>>()?;
        let encoded_refs = encoded.iter().map(String::as_str).collect::<Vec<_>>();
        self.inner.embed_batch(&encoded_refs).await
    }
}

fn profile_error_to_embedder_error(error: EmbeddingProfileError) -> EmbedderError {
    EmbedderError::InvalidInput(error.to_string())
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingProfileError {
    #[error("invalid embedding profile ID: {0}")]
    InvalidProfileId(String),
    #[error("invalid embedding profile: {0}")]
    InvalidProfile(String),
    #[error("invalid encoding template: {0}")]
    InvalidEncodingTemplate(String),
    #[error("invalid chunking strategy: {0}")]
    InvalidChunkingStrategy(String),
    #[error("invalid model artifact hash: {0}")]
    InvalidArtifactHash(String),
    #[error("{0} cannot be empty")]
    EmptyInput(&'static str),
    #[error("unsupported embedding profile manifest schema version {0}")]
    UnsupportedManifestVersion(u32),
    #[error("invalid embedding profile manifest: {0}")]
    InvalidManifest(String),
    #[error("model artifact `{artifact}` failed verification: {reason}")]
    ArtifactVerificationFailed { artifact: String, reason: String },
    #[error("embedding profile `{0}` is not installed in this runtime")]
    ProfileNotInstalled(EmbeddingProfileId),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingEmbedder {
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingEmbedder {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for RecordingEmbedder {
        fn embed<'a>(
            &'a self,
            text: &'a str,
        ) -> crate::embedder::BoxedEmbedderFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(text.to_string());
                Ok(vec![1.0, 0.0])
            })
        }

        fn embed_batch<'a>(
            &'a self,
            texts: &'a [&'a str],
        ) -> crate::embedder::BoxedEmbedderFuture<'a, Vec<Vec<f32>>> {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap()
                    .extend(texts.iter().map(|text| (*text).to_string()));
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            })
        }

        fn model_name(&self) -> &str {
            "recording"
        }

        fn dimension(&self) -> usize {
            2
        }

        fn model_hash(&self) -> String {
            "recording".to_string()
        }

        fn signature(&self) -> crate::storage::ModelSignature {
            crate::storage::ModelSignature {
                name: self.model_name().to_string(),
                dimension: self.dimension(),
                hash: self.model_hash(),
            }
        }
    }

    #[test]
    fn builtins_have_unique_valid_ids_and_contracts() {
        let profiles = builtin_embedding_profiles();
        let ids = profiles
            .iter()
            .map(|profile| profile.profile_id.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), 6);
        for profile in profiles {
            profile.validate().unwrap();
            assert_eq!(profile.contract_hash().len(), 64);
        }
    }

    #[test]
    fn nomic_retrieval_profile_uses_document_and_query_prefixes() {
        let profile = BuiltinEmbeddingProfile::NomicRetrieval256.profile();
        assert_eq!(
            profile.encode_document("memory").unwrap(),
            "search_document: memory"
        );
        assert_eq!(
            profile.encode_query("request").unwrap(),
            "search_query: request"
        );
    }

    #[test]
    fn qwen_contract_is_raw_document_and_versioned_instruction_query() {
        let profile = BuiltinEmbeddingProfile::QwenBalanced1024.profile();
        assert_eq!(profile.encode_document("memory").unwrap(), "memory");
        assert_eq!(
            profile.encode_query("request").unwrap(),
            "Instruct: Given an agent-memory request, retrieve the most relevant confirmed memories, code facts, decisions, and incident evidence.\nQuery: request"
        );
    }

    #[test]
    fn qwen_profiles_pin_hub_revisions_and_all_runner_weight_artifacts() {
        let balanced = BuiltinEmbeddingProfile::QwenBalanced1024.profile();
        assert_eq!(balanced.immutable_model_revision, QWEN_06B_REVISION);
        assert_eq!(balanced.verified_model_artifact_hashes.len(), 3);
        assert!(
            balanced
                .verified_model_artifact_hashes
                .iter()
                .any(|artifact| artifact.artifact == "model.safetensors")
        );

        let maximum = BuiltinEmbeddingProfile::QwenMaxNative.profile();
        assert_eq!(maximum.immutable_model_revision, QWEN_4B_REVISION);
        assert_eq!(maximum.verified_model_artifact_hashes.len(), 5);
        assert!(
            maximum
                .verified_model_artifact_hashes
                .iter()
                .any(|artifact| {
                    artifact.artifact == "model.safetensors.index.json"
                        && artifact.digest
                            == "9d130c7f24fa1f9a2a7e19fad42c7d6d2d6fea31b180bdf3e8aac1924c26c39a"
                })
        );
    }

    #[tokio::test]
    async fn profiled_embedder_never_sends_raw_nomic_retrieval_input() {
        let inner = Arc::new(RecordingEmbedder::new());
        let embedder = ProfiledEmbedder::new(
            BuiltinEmbeddingProfile::NomicRetrieval256.profile(),
            inner.clone(),
        )
        .unwrap();

        embedder.embed_document("stored fact").await.unwrap();
        embedder.embed_query("find fact").await.unwrap();

        assert_eq!(
            *inner.calls.lock().unwrap(),
            vec!["search_document: stored fact", "search_query: find fact"]
        );
    }

    #[test]
    fn manifest_is_explicitly_not_installed_until_a_verified_install() {
        let manifest = EmbeddingProfileManifest::not_installed(
            BuiltinEmbeddingProfile::QwenBalanced1024.profile(),
        )
        .unwrap();
        assert_eq!(manifest.state, EmbeddingProfileState::NotInstalled);
        assert_eq!(
            manifest.verification.status,
            VerificationStatus::NotAttempted
        );
        manifest.validate().unwrap();
    }

    #[test]
    fn verified_manifest_requires_a_real_artifact_receipt() {
        let mut manifest = EmbeddingProfileManifest::not_installed(
            BuiltinEmbeddingProfile::NomicRetrieval256.profile(),
        )
        .unwrap();
        manifest.verification.status = VerificationStatus::Verified;
        manifest.verification.verified_at = Some(Utc::now());
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn builtin_lookup_is_exact_and_does_not_select_a_profile_implicitly() {
        assert!(builtin_embedding_profile_by_id("qwen3-0.6b").is_none());
        assert_eq!(
            builtin_embedding_profile_by_id("qwen3-0.6b-retrieval-v1-1024")
                .unwrap()
                .profile_id
                .as_str(),
            "qwen3-0.6b-retrieval-v1-1024"
        );
    }

    #[test]
    fn artifact_manifest_rejects_traversal_separators_and_uppercase_sha256() {
        let digest = "a".repeat(64);
        for artifact in [
            "../weights.onnx",
            "/weights.onnx",
            "C:\\weights.onnx",
            "dir//weights.onnx",
        ] {
            assert!(
                ModelArtifactHash::sha256(artifact, &digest)
                    .validate()
                    .is_err()
            );
        }
        assert!(
            ModelArtifactHash::sha256("weights.onnx", "A".repeat(64))
                .validate()
                .is_err()
        );
        ModelArtifactHash::sha256("weights.onnx", digest)
            .validate()
            .unwrap();
        ModelArtifactHash::sha256("onnx/model.onnx", "b".repeat(64))
            .validate()
            .unwrap();
    }

    #[test]
    fn explicit_install_verifies_the_pinned_local_artifact_without_auto_selection() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weights.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let artifact = ModelArtifactHash::sha256(
            "weights.onnx",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        let mut profile = BuiltinEmbeddingProfile::NomicRetrieval256.profile();
        profile.verified_model_artifact_hashes = vec![artifact.clone()];
        let registry = ProfileRuntimeRegistry::new();
        let manifest = registry
            .install_verified(
                profile.clone(),
                &[VerifiedLocalArtifact::new(artifact, path)],
                EmbeddingRuntimeMetadata {
                    backend: EmbeddingRuntimeBackend::FastembedOnnx,
                    device: EmbeddingDevice::Cpu,
                    runtime_version: "test".to_string(),
                    initialized_at: Utc::now(),
                    local_only: true,
                },
                Arc::new(RecordingEmbedder::new()),
            )
            .unwrap();

        assert_eq!(manifest.state, EmbeddingProfileState::Installed);
        assert_eq!(
            registry
                .manifest(&profile.profile_id)
                .unwrap()
                .verification
                .status,
            VerificationStatus::Verified
        );
        assert!(
            registry
                .embedder(&EmbeddingProfileId::new("qwen3-0.6b-retrieval-v1-256").unwrap())
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rooted_artifact_resolution_rejects_symlink_escapes() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("weights.onnx");
        std::fs::write(&outside_file, b"abc").unwrap();
        std::os::unix::fs::symlink(&outside_file, root.path().join("weights.onnx")).unwrap();
        let artifact = ModelArtifactHash::sha256(
            "weights.onnx",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        assert!(VerifiedLocalArtifact::from_root(artifact, root.path()).is_err());
    }
}
