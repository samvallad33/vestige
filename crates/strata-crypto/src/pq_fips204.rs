//! Real NIST FIPS 204 (ML-DSA-65) + Ed25519 Hybrid Post-Quantum Signing.
//!
//! Pure Rust (`fips204` `#![forbid(unsafe_code)]`, ACVP/Wycheproof verified)
//! paired with classical `ed25519-dalek` v3 so Strata receipts and checkpoints
//! stay verifiable even if classical elliptic-curve discrete log is broken.
//!
//! Domain-separation binds both signatures over a single BLAKE3-256 pre-hash
//! tagged with `DOMAIN_TAG` plus the caller-supplied context label.

use ed25519_dalek::{Signature as Ed25519Sig, Signer, SigningKey, Verifier, VerifyingKey};
use fips204::ml_dsa_65;
use fips204::traits::{KeyGen, SerDes, Signer as PqSigner, Verifier as PqVerifier};
use serde::{Deserialize, Serialize};

const DOMAIN_TAG: &[u8] = b"vestige.strata.pq-hybrid.v1.ed25519+ml-dsa-65";

/// Verification failure modes for [`Fips204HybridSignature`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridPqError {
    EntropyUnavailable,
    PqKeyGenFailed,
    PqSignFailed,
    MalformedEd25519Key,
    MalformedEd25519Signature,
    MalformedMlDsaPublicKey,
    MalformedMlDsaSignature,
    Ed25519VerificationFailed,
    MlDsaVerificationFailed,
}

impl std::fmt::Display for HybridPqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for HybridPqError {}

/// Public verification key bundle (Ed25519 32B + ML-DSA-65 1,952B).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fips204HybridPublicKey {
    pub ed25519_pk: Vec<u8>,
    pub ml_dsa_65_pk: Vec<u8>,
}

/// Hybrid detached signature (Ed25519 64B + ML-DSA-65 3,309B).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fips204HybridSignature {
    pub algorithm: String,
    pub context: String,
    pub payload_blake3_hex: String,
    pub ed25519_sig: Vec<u8>,
    pub ml_dsa_65_sig: Vec<u8>,
}

/// Ephemeral or persisted hybrid signing keypair (Ed25519 + NIST FIPS 204 ML-DSA-65).
pub struct Fips204HybridKeypair {
    ed_sk: SigningKey,
    pq_pk: ml_dsa_65::PublicKey,
    pq_sk: ml_dsa_65::PrivateKey,
}

impl Fips204HybridKeypair {
    /// Generate a fresh hybrid keypair from the OS CSPRNG (`getrandom`).
    pub fn generate() -> Result<Self, HybridPqError> {
        let mut ed_seed = [0u8; 32];
        let mut pq_seed = [0u8; 32];
        getrandom::fill(&mut ed_seed).map_err(|_| HybridPqError::EntropyUnavailable)?;
        getrandom::fill(&mut pq_seed).map_err(|_| HybridPqError::EntropyUnavailable)?;
        Self::from_seeds(ed_seed, pq_seed)
    }

    /// Deterministically derive a hybrid keypair from two 32-byte seeds.
    pub fn from_seeds(ed_seed: [u8; 32], pq_seed: [u8; 32]) -> Result<Self, HybridPqError> {
        let ed_sk = SigningKey::from_bytes(&ed_seed);
        let (pq_pk, pq_sk) = ml_dsa_65::KG::keygen_from_seed(&pq_seed);
        Ok(Self {
            ed_sk,
            pq_pk,
            pq_sk,
        })
    }

    /// Export the serialized public key bundle.
    pub fn public_key(&self) -> Fips204HybridPublicKey {
        Fips204HybridPublicKey {
            ed25519_pk: self.ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_65_pk: self.pq_pk.clone().into_bytes().to_vec(),
        }
    }

    /// Sign `payload` under domain `context` with BOTH Ed25519 and ML-DSA-65.
    pub fn sign(&self, context: &str, payload: &[u8]) -> Result<Fips204HybridSignature, HybridPqError> {
        let digest = domain_digest(context, payload);
        let ed_sig = self.ed_sk.sign(&digest);
        let ctx_bytes = context.as_bytes();
        let pq_ctx = if ctx_bytes.len() <= 255 {
            ctx_bytes
        } else {
            &ctx_bytes[..255]
        };
        let pq_sig = self
            .pq_sk
            .try_sign(&digest, pq_ctx)
            .map_err(|_| HybridPqError::PqSignFailed)?;
        Ok(Fips204HybridSignature {
            algorithm: "ed25519+ml-dsa-65".to_string(),
            context: context.to_string(),
            payload_blake3_hex: blake3::hash(payload).to_hex().to_string(),
            ed25519_sig: ed_sig.to_bytes().to_vec(),
            ml_dsa_65_sig: pq_sig.to_vec(),
        })
    }
}

/// Verify BOTH classical Ed25519 and post-quantum ML-DSA-65 signatures over `payload`.
/// Succeeds if and only if both cryptographic proofs verify under the exact same context.
pub fn verify_hybrid_signature(
    pk: &Fips204HybridPublicKey,
    sig: &Fips204HybridSignature,
    expected_context: &str,
    payload: &[u8],
) -> Result<(), HybridPqError> {
    if sig.context != expected_context {
        return Err(HybridPqError::Ed25519VerificationFailed);
    }
    let digest = domain_digest(expected_context, payload);

    let ed_pk_bytes: [u8; 32] = pk
        .ed25519_pk
        .as_slice()
        .try_into()
        .map_err(|_| HybridPqError::MalformedEd25519Key)?;
    let ed_vk =
        VerifyingKey::from_bytes(&ed_pk_bytes).map_err(|_| HybridPqError::MalformedEd25519Key)?;
    let ed_sig_bytes: [u8; 64] = sig
        .ed25519_sig
        .as_slice()
        .try_into()
        .map_err(|_| HybridPqError::MalformedEd25519Signature)?;
    let ed_sig = Ed25519Sig::from_bytes(&ed_sig_bytes);
    ed_vk
        .verify(&digest, &ed_sig)
        .map_err(|_| HybridPqError::Ed25519VerificationFailed)?;

    let pq_pk_bytes: [u8; ml_dsa_65::PK_LEN] = pk
        .ml_dsa_65_pk
        .as_slice()
        .try_into()
        .map_err(|_| HybridPqError::MalformedMlDsaPublicKey)?;
    let pq_vk = ml_dsa_65::PublicKey::try_from_bytes(pq_pk_bytes)
        .map_err(|_| HybridPqError::MalformedMlDsaPublicKey)?;
    let pq_sig_bytes: [u8; ml_dsa_65::SIG_LEN] = sig
        .ml_dsa_65_sig
        .as_slice()
        .try_into()
        .map_err(|_| HybridPqError::MalformedMlDsaSignature)?;
    let ctx_bytes = expected_context.as_bytes();
    let pq_ctx = if ctx_bytes.len() <= 255 {
        ctx_bytes
    } else {
        &ctx_bytes[..255]
    };
    if !pq_vk.verify(&digest, &pq_sig_bytes, pq_ctx) {
        return Err(HybridPqError::MlDsaVerificationFailed);
    }
    Ok(())
}

fn domain_digest(context: &str, payload: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(DOMAIN_TAG);
    h.update(&(context.len() as u64).to_le_bytes());
    h.update(context.as_bytes());
    h.update(&(payload.len() as u64).to_le_bytes());
    h.update(payload);
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips204_hybrid_sign_verify_and_tamper_rejection() {
        let kp = Fips204HybridKeypair::from_seeds([0x11; 32], [0x22; 32]).expect("keygen");
        let pk = kp.public_key();
        assert_eq!(pk.ed25519_pk.len(), 32);
        assert_eq!(pk.ml_dsa_65_pk.len(), ml_dsa_65::PK_LEN);

        let payload = b"strata-checkpoint-root-v7";
        let sig = kp.sign("strata.checkpoint", payload).expect("sign");
        assert_eq!(sig.ed25519_sig.len(), 64);
        assert_eq!(sig.ml_dsa_65_sig.len(), ml_dsa_65::SIG_LEN);

        // Valid verification
        assert_eq!(
            verify_hybrid_signature(&pk, &sig, "strata.checkpoint", payload),
            Ok(())
        );

        // Payload tamper rejected
        assert!(verify_hybrid_signature(&pk, &sig, "strata.checkpoint", b"tampered").is_err());

        // Cross-context replay rejected
        assert!(verify_hybrid_signature(&pk, &sig, "strata.receipt", payload).is_err());

        // Post-quantum signature bit-flip rejected even when classical Ed25519 is intact
        let mut pq_tampered = sig.clone();
        pq_tampered.ml_dsa_65_sig[0] ^= 0xFF;
        assert_eq!(
            verify_hybrid_signature(&pk, &pq_tampered, "strata.checkpoint", payload),
            Err(HybridPqError::MlDsaVerificationFailed)
        );

        // Classical signature bit-flip rejected even when ML-DSA-65 is intact
        let mut ed_tampered = sig;
        ed_tampered.ed25519_sig[0] ^= 0xFF;
        assert_eq!(
            verify_hybrid_signature(&pk, &ed_tampered, "strata.checkpoint", payload),
            Err(HybridPqError::Ed25519VerificationFailed)
        );
    }
}
