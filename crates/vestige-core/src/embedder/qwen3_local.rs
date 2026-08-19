//! Explicit local-only Qwen3 embedding runner.
//!
//! This module intentionally has no Hub client. The caller must first resolve
//! every artifact below a user-approved directory and verify its pinned hash;
//! this runner then opens only those verified paths.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fastembed::{Qwen3Config, Qwen3Model, Qwen3TextEmbedding};
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::embedding::{EmbeddingProfile, EmbeddingRuntimeBackend, VerifiedLocalArtifact};

use super::{EmbedderError, EmbedderResult, EmbedderSend};

const QWEN_06B_NATIVE_DIMENSIONS: usize = 1024;
const QWEN_4B_NATIVE_DIMENSIONS: usize = 2560;
const QWEN_4B_WEIGHT_SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

/// The local runner supports exactly the pinned Qwen contracts in the profile
/// catalog. Adding a checkpoint requires an explicit profile, native-vector
/// contract, and artifact-manifest review; model IDs are not an escape hatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenModelSpec {
    native_dimensions: usize,
    weight_artifacts: &'static [&'static str],
    shard_index_artifact: Option<&'static str>,
}

fn model_spec(profile: &EmbeddingProfile) -> EmbedderResult<QwenModelSpec> {
    match profile.model_id.as_str() {
        "Qwen/Qwen3-Embedding-0.6B" => {
            if !matches!(
                profile.embedding_dimension,
                256 | QWEN_06B_NATIVE_DIMENSIONS
            ) {
                return Err(EmbedderError::Init(format!(
                    "Qwen 0.6B profile '{}' requests unsupported {} dimensions",
                    profile.profile_id, profile.embedding_dimension
                )));
            }
            Ok(QwenModelSpec {
                native_dimensions: QWEN_06B_NATIVE_DIMENSIONS,
                weight_artifacts: &["model.safetensors"],
                shard_index_artifact: None,
            })
        }
        "Qwen/Qwen3-Embedding-4B" => {
            if !matches!(
                profile.embedding_dimension,
                1024 | QWEN_4B_NATIVE_DIMENSIONS
            ) {
                return Err(EmbedderError::Init(format!(
                    "Qwen 4B profile '{}' requests unsupported {} dimensions; expected 1024 or {QWEN_4B_NATIVE_DIMENSIONS}",
                    profile.profile_id, profile.embedding_dimension
                )));
            }
            Ok(QwenModelSpec {
                native_dimensions: QWEN_4B_NATIVE_DIMENSIONS,
                weight_artifacts: &QWEN_4B_WEIGHT_SHARDS,
                shard_index_artifact: Some("model.safetensors.index.json"),
            })
        }
        _ => Err(EmbedderError::Init(format!(
            "local Qwen runner supports only the pinned Qwen3 0.6B and 4B embedding profiles, not '{}'",
            profile.model_id
        ))),
    }
}

/// A Qwen3 embedder loaded from a verified, explicitly supplied local artifact
/// set. CPU is selected deliberately; there is no hardware probe, automatic
/// device selection, model fallback, or Hub lookup.
pub struct Qwen3LocalEmbedder {
    profile: EmbeddingProfile,
    runner: Mutex<Qwen3TextEmbedding>,
    model_hash: String,
}

impl Qwen3LocalEmbedder {
    /// Construct the local runner after artifact verification. The profile is
    /// deliberately passed in so its output dimension remains part of the
    /// vector-space contract rather than an ambient runtime setting.
    pub fn from_verified_local_artifacts(
        profile: EmbeddingProfile,
        artifacts: &[VerifiedLocalArtifact],
    ) -> EmbedderResult<Self> {
        profile
            .validate()
            .map_err(|error| EmbedderError::Init(error.to_string()))?;
        if profile.runtime_backend != EmbeddingRuntimeBackend::FastembedCandle {
            return Err(EmbedderError::Init(
                "local Qwen runner requires the fastembed_candle profile backend".to_string(),
            ));
        }
        let spec = model_spec(&profile)?;

        let verified = artifacts
            .iter()
            .map(VerifiedLocalArtifact::verify)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| EmbedderError::Init(error.to_string()))?;
        if verified != profile.verified_model_artifact_hashes {
            return Err(EmbedderError::Init(
                "verified artifacts do not exactly match the profile contract".to_string(),
            ));
        }

        let config_path = artifact_path(artifacts, "config.json")?;
        let weight_paths = spec
            .weight_artifacts
            .iter()
            .map(|name| artifact_path(artifacts, name))
            .collect::<EmbedderResult<Vec<_>>>()?;
        if let Some(index_name) = spec.shard_index_artifact {
            validate_shard_index(artifact_path(artifacts, index_name)?, spec.weight_artifacts)?;
        }
        let tokenizer_path = artifact_path(artifacts, "tokenizer.json")?;
        let config: Qwen3Config = serde_json::from_slice(
            &std::fs::read(config_path)
                .map_err(|error| EmbedderError::Init(format!("read Qwen config: {error}")))?,
        )
        .map_err(|error| EmbedderError::Init(format!("parse Qwen config: {error}")))?;
        if config.hidden_size != spec.native_dimensions {
            return Err(EmbedderError::Init(format!(
                "Qwen profile '{}' expects {} native dimensions, artifact config declares {}",
                profile.profile_id, spec.native_dimensions, config.hidden_size
            )));
        }

        let device = Device::Cpu;
        // SAFETY: every path has been canonicalized below the explicit artifact
        // root and hash-verified before memory mapping.
        let variables = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_paths, DType::F32, &device)
        }
        .map_err(|error| {
            EmbedderError::Init(format!(
                "map Qwen weights: {error}; preflight this profile on an explicitly provisioned CPU device with sufficient available RAM (Qwen 4B is large). Vestige does not probe or select hardware automatically"
            ))
        })?;
        let model = Qwen3Model::new(config, variables).map_err(|error| {
            EmbedderError::Init(format!(
                "load Qwen model: {error}; preflight the selected CPU device and model capacity manually. Vestige does not probe or select hardware automatically"
            ))
        })?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| EmbedderError::Init(format!("load Qwen tokenizer: {error}")))?;
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length: profile.maximum_token_limit,
            ..Default::default()
        }));

        let model_hash = profile.contract_hash();
        Ok(Self {
            profile,
            runner: Mutex::new(Qwen3TextEmbedding::new(model, tokenizer)),
            model_hash,
        })
    }

    fn project(&self, native: Vec<f32>) -> EmbedderResult<Vec<f32>> {
        project_embedding(&self.profile, native)
    }
}

fn project_embedding(profile: &EmbeddingProfile, native: Vec<f32>) -> EmbedderResult<Vec<f32>> {
    let spec = model_spec(profile).map_err(|error| match error {
        EmbedderError::Init(message) => EmbedderError::EmbedFailed(message),
        other => other,
    })?;
    if native.len() != spec.native_dimensions {
        return Err(EmbedderError::EmbedFailed(format!(
            "Qwen runtime produced {} dimensions; expected {} for profile '{}'",
            native.len(),
            spec.native_dimensions,
            profile.profile_id
        )));
    }
    let mut vector = native;
    vector.truncate(profile.embedding_dimension);
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(EmbedderError::EmbedFailed(
            "Qwen runtime produced a non-normalizable embedding".to_string(),
        ));
    }
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn artifact_path<'a>(
    artifacts: &'a [VerifiedLocalArtifact],
    name: &str,
) -> EmbedderResult<&'a Path> {
    artifacts
        .iter()
        .find(|artifact| artifact.artifact.artifact == name)
        .map(|artifact| artifact.path.as_path())
        .ok_or_else(|| {
            EmbedderError::Init(format!("required verified artifact '{name}' is missing"))
        })
}

#[derive(Debug, Deserialize)]
struct SafetensorsShardIndex {
    weight_map: BTreeMap<String, String>,
}

/// Ensure the pinned index only references the verified shard set, and that
/// both required 4B shards actually participate in the model. The index hash
/// remains the primary immutable-contract check; this rejects a malformed
/// local index before it could become an accidental partial model load.
fn validate_shard_index(index_path: &Path, expected_shards: &[&str]) -> EmbedderResult<()> {
    let bytes = std::fs::read(index_path)
        .map_err(|error| EmbedderError::Init(format!("read Qwen shard index: {error}")))?;
    let index: SafetensorsShardIndex = serde_json::from_slice(&bytes)
        .map_err(|error| EmbedderError::Init(format!("parse Qwen shard index: {error}")))?;
    if index.weight_map.is_empty() {
        return Err(EmbedderError::Init(
            "Qwen shard index has no weight mappings".to_string(),
        ));
    }
    let referenced = index
        .weight_map
        .values()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if referenced.len() != expected_shards.len()
        || expected_shards
            .iter()
            .any(|shard| !referenced.contains(*shard))
        || referenced
            .iter()
            .any(|shard| !expected_shards.contains(shard))
    {
        return Err(EmbedderError::Init(
            "Qwen shard index does not exactly reference the profile's verified weight shards"
                .to_string(),
        ));
    }
    Ok(())
}

impl EmbedderSend for Qwen3LocalEmbedder {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        let runner = self
            .runner
            .lock()
            .map_err(|_| EmbedderError::EmbedFailed("Qwen runner lock poisoned".to_string()))?;
        let native = runner
            .embed(&[text])
            .map_err(|error| EmbedderError::EmbedFailed(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::EmbedFailed("Qwen returned no embedding".to_string()))?;
        self.project(native)
    }

    fn model_name(&self) -> &str {
        &self.profile.model_id
    }

    fn dimension(&self) -> usize {
        self.profile.embedding_dimension
    }

    fn model_hash(&self) -> String {
        self.model_hash.clone()
    }

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let runner = self
            .runner
            .lock()
            .map_err(|_| EmbedderError::EmbedFailed("Qwen runner lock poisoned".to_string()))?;
        runner
            .embed(texts)
            .map_err(|error| EmbedderError::EmbedFailed(error.to_string()))?
            .into_iter()
            .map(|native| self.project(native))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::embedder::Embedder;
    use crate::embedding::BuiltinEmbeddingProfile;

    use super::*;

    #[test]
    fn qwen_4b_profiles_require_both_pinned_weight_shards() {
        for builtin in [
            BuiltinEmbeddingProfile::QwenMax1024,
            BuiltinEmbeddingProfile::QwenMaxNative,
        ] {
            let spec = model_spec(&builtin.profile()).expect("valid Qwen 4B profile");
            assert_eq!(spec.native_dimensions, QWEN_4B_NATIVE_DIMENSIONS);
            assert_eq!(spec.weight_artifacts, QWEN_4B_WEIGHT_SHARDS);
            assert_eq!(
                spec.shard_index_artifact,
                Some("model.safetensors.index.json")
            );
        }
    }

    #[test]
    fn qwen_4b_prefix_projection_is_deterministic_and_normalized() {
        let profile = BuiltinEmbeddingProfile::QwenMax1024.profile();
        let native = (0..QWEN_4B_NATIVE_DIMENSIONS)
            .map(|index| index as f32 + 1.0)
            .collect();
        let vector = project_embedding(&profile, native).expect("project 4B native embedding");
        assert_eq!(vector.len(), 1024);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected L2 norm, got {norm}");
        assert!(
            vector[0] < vector[1],
            "projection preserves native prefix ordering"
        );
    }

    #[test]
    fn qwen_4b_native_projection_keeps_2560_dimensions_and_normalizes() {
        let profile = BuiltinEmbeddingProfile::QwenMaxNative.profile();
        let vector = project_embedding(&profile, vec![1.0; QWEN_4B_NATIVE_DIMENSIONS])
            .expect("preserve 4B native embedding");
        assert_eq!(vector.len(), QWEN_4B_NATIVE_DIMENSIONS);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected L2 norm, got {norm}");
    }

    #[test]
    fn qwen_4b_projection_rejects_non_native_runtime_dimension() {
        let error = project_embedding(
            &BuiltinEmbeddingProfile::QwenMax1024.profile(),
            vec![0.0; QWEN_06B_NATIVE_DIMENSIONS],
        )
        .expect_err("4B must not accept an 0.6B-length embedding");
        assert!(error.to_string().contains("expected 2560"));
    }

    #[test]
    fn shard_index_must_reference_exactly_the_verified_shards() {
        let temp_dir = tempfile::tempdir().expect("temporary shard-index directory");
        let index_path = temp_dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .expect("write valid shard index");
        validate_shard_index(&index_path, &QWEN_4B_WEIGHT_SHARDS)
            .expect("index names both verified shards");

        std::fs::write(
            &index_path,
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"other.safetensors"}}"#,
        )
        .expect("write invalid shard index");
        assert!(validate_shard_index(&index_path, &QWEN_4B_WEIGHT_SHARDS).is_err());
    }

    #[test]
    fn real_local_qwen_embedding_respects_the_profile_dimension() {
        let Ok(root) = std::env::var("VESTIGE_QWEN_ARTIFACT_ROOT") else {
            eprintln!("skipping local Qwen runner test; VESTIGE_QWEN_ARTIFACT_ROOT is unset");
            return;
        };
        let profile = BuiltinEmbeddingProfile::QwenBalanced1024.profile();
        let artifacts = profile
            .verified_model_artifact_hashes
            .iter()
            .cloned()
            .map(|artifact| VerifiedLocalArtifact::from_root(artifact, PathBuf::from(&root)))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let runner = Qwen3LocalEmbedder::from_verified_local_artifacts(profile, &artifacts)
            .expect("load Qwen strictly from verified local artifacts");
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let vector = runtime
            .block_on(Embedder::embed(
                &runner,
                "Vestige must preserve a user-approved decision.",
            ))
            .expect("embed with local Qwen runner");
        assert_eq!(vector.len(), 1024);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected L2 norm, got {norm}");
    }
}
