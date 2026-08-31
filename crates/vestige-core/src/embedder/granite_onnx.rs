//! Granite embedding runner: IBM granite-embedding-311m-multilingual-r2
//! through fastembed's user-defined ONNX path.
//!
//! Granite is not in fastembed's model catalogue, so this runner loads the
//! ONNX graph and tokenizer files from an explicitly installed, hash-verified
//! local artifact directory — the same verified-local contract the Qwen
//! runner uses. No Hub or network client is ever invoked here.
//!
//! Model facts, verified against the pinned revision
//! `44399559930365213510b1ee2eb15ded83374f0e` (Apache-2.0):
//! ModernBERT encoder, 768 native dimensions, 32,768-token context (4x the
//! Nomic profile's 8,192), CLS pooling (`1_Pooling/config.json` sets
//! `pooling_mode_cls_token: true`), no query/document prefixes. The profile
//! stores the official Matryoshka truncation to 256 dimensions, so existing
//! 256d vector storage is reused as-is.

use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};

use super::{EmbedderError, EmbedderResult, EmbedderSend};
use crate::embedding::{EmbeddingProfile, EmbeddingRuntimeBackend, VerifiedLocalArtifact};

/// Native output width of the granite-embedding-311m encoder.
const GRANITE_NATIVE_DIMENSIONS: usize = 768;

/// Artifact names the profile contract pins, relative to the install root.
pub const GRANITE_ARTIFACTS: [&str; 5] = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

pub struct GraniteOnnxEmbedder {
    profile: EmbeddingProfile,
    // fastembed's embed() requires &mut self.
    engine: Mutex<TextEmbedding>,
    model_hash: String,
}

impl GraniteOnnxEmbedder {
    /// Build the runner from an explicitly installed, verified artifact set.
    /// Every artifact must hash-match the profile contract exactly; a partial
    /// or substituted install refuses to start rather than silently embedding
    /// with different weights.
    pub fn from_verified_local_artifacts(
        profile: EmbeddingProfile,
        artifacts: &[VerifiedLocalArtifact],
    ) -> EmbedderResult<Self> {
        profile
            .validate()
            .map_err(|error| EmbedderError::Init(error.to_string()))?;
        if profile.runtime_backend != EmbeddingRuntimeBackend::FastembedOnnx {
            return Err(EmbedderError::Init(
                "Granite runner requires the fastembed_onnx profile backend".to_string(),
            ));
        }
        if profile.embedding_dimension > GRANITE_NATIVE_DIMENSIONS {
            return Err(EmbedderError::Init(format!(
                "Granite profile '{}' requests {} dimensions; the encoder produces {}",
                profile.profile_id, profile.embedding_dimension, GRANITE_NATIVE_DIMENSIONS
            )));
        }

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

        let read = |name: &str| -> EmbedderResult<Vec<u8>> {
            let path = artifacts
                .iter()
                .find(|artifact| artifact.artifact.artifact == name)
                .map(|artifact| artifact.path.as_path())
                .ok_or_else(|| {
                    EmbedderError::Init(format!("profile contract is missing artifact '{name}'"))
                })?;
            std::fs::read(path)
                .map_err(|error| EmbedderError::Init(format!("read Granite '{name}': {error}")))
        };

        let model = UserDefinedEmbeddingModel::new(
            read("onnx/model.onnx")?,
            TokenizerFiles {
                tokenizer_file: read("tokenizer.json")?,
                config_file: read("config.json")?,
                special_tokens_map_file: read("special_tokens_map.json")?,
                tokenizer_config_file: read("tokenizer_config.json")?,
            },
        )
        // Verified from the pinned revision's 1_Pooling/config.json:
        // pooling_mode_cls_token = true, all other modes false.
        .with_pooling(Pooling::Cls);

        // InitOptionsUserDefined is #[non_exhaustive]; construct via default
        // and mutate the public field.
        let mut options = InitOptionsUserDefined::default();
        options.max_length = profile.maximum_token_limit;
        let engine = TextEmbedding::try_new_from_user_defined(model, options)
            .map_err(|error| EmbedderError::Init(format!("load Granite ONNX: {error}")))?;

        let model_hash = profile.contract_hash();
        Ok(Self {
            profile,
            engine: Mutex::new(engine),
            model_hash,
        })
    }

    /// Official Matryoshka projection: truncate the native vector to the
    /// profile dimension, then L2-renormalize.
    fn project(&self, native: Vec<f32>) -> EmbedderResult<Vec<f32>> {
        if native.len() != GRANITE_NATIVE_DIMENSIONS {
            return Err(EmbedderError::EmbedFailed(format!(
                "Granite runtime produced {} dimensions; expected {} for profile '{}'",
                native.len(),
                GRANITE_NATIVE_DIMENSIONS,
                self.profile.profile_id
            )));
        }
        let mut vector = native;
        vector.truncate(self.profile.embedding_dimension);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err(EmbedderError::EmbedFailed(
                "Granite runtime produced a non-normalizable embedding".to_string(),
            ));
        }
        for value in &mut vector {
            *value /= norm;
        }
        Ok(vector)
    }

    fn embed_all(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| EmbedderError::EmbedFailed("Granite engine lock poisoned".to_string()))?;
        engine
            .embed(texts, None)
            .map_err(|error| EmbedderError::EmbedFailed(error.to_string()))?
            .into_iter()
            .map(|native| self.project(native))
            .collect()
    }
}

impl EmbedderSend for GraniteOnnxEmbedder {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        self.embed_all(&[text])?
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::EmbedFailed("Granite returned no embedding".to_string()))
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
        self.embed_all(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{BuiltinEmbeddingProfile, VerifiedLocalArtifact};

    /// Real-model integration: loads the actual Granite ONNX artifacts from
    /// `GRANITE_ARTIFACT_ROOT`, hash-verifies them against the profile
    /// contract, and embeds. Requires the ~1.2 GB artifact set on disk.
    #[test]
    #[ignore = "loads the real 1.2 GB Granite ONNX artifact set; set GRANITE_ARTIFACT_ROOT and run with --ignored"]
    fn granite_embeds_real_text_at_256_dimensions() {
        let root = std::env::var("GRANITE_ARTIFACT_ROOT")
            .expect("set GRANITE_ARTIFACT_ROOT to the downloaded artifact directory");
        let profile = BuiltinEmbeddingProfile::GraniteMultilingual256.profile();
        let artifacts = profile
            .verified_model_artifact_hashes
            .iter()
            .map(|hash| VerifiedLocalArtifact::from_root(hash.clone(), &root))
            .collect::<Result<Vec<_>, _>>()
            .expect("resolve artifacts under the install root");
        let embedder =
            GraniteOnnxEmbedder::from_verified_local_artifacts(profile.clone(), &artifacts)
                .expect("hash-verify and load the Granite runner");

        let vectors = embedder
            .embed_all(&[
                "A canine companion enjoys sprinting through the meadow at dawn",
                "El modelo de memoria olvida lo que nunca se consulta",
            ])
            .expect("embed real multilingual text");
        assert_eq!(vectors.len(), 2);
        for vector in &vectors {
            assert_eq!(vector.len(), 256);
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "Matryoshka projection must renormalize");
        }
        // Different sentences must not embed identically.
        let dot: f32 = vectors[0].iter().zip(&vectors[1]).map(|(a, b)| a * b).sum();
        assert!(dot < 0.99, "distinct texts collapsed to the same vector");
    }
}
