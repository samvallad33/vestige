//! Offline NIST FIPS 204 ML-DSA-65 + Ed25519 hybrid verification for DSSE envelopes.
//!
//! Pure-Rust (`#![forbid(unsafe_code)]`), zero network dependencies, zero store access.
//! Enables CNSA 2.0 / post-quantum dual-signature attestation bundles verified offline in CI.

#![forbid(unsafe_code)]

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use fips204::ml_dsa_65;
use fips204::traits::{SerDes, Signer as _, Verifier as _};
use serde::{Deserialize, Serialize};

use crate::didkey::TrustedKey;
use crate::dsse::{self, Envelope};
use crate::{ProofError, err};

/// Domain separator bound into ML-DSA-65 verification over DSSE PAE bytes.
pub const PQ_DSSE_CONTEXT: &[u8] = b"vestige-proof-dsse-v1";
/// Canonical algorithm identifier for hybrid ML-DSA-65 + Ed25519 DSSE co-signatures.
pub const PQ_HYBRID_ALG: &str = "ml-dsa-65+ed25519";

/// Post-quantum ML-DSA-65 co-signature over a DSSE envelope's exact PAE bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PqCosignature {
    pub algorithm: String,
    pub public_key_b64: String,
    pub signature_b64: String,
    pub key_fingerprint_blake3: String,
}

/// A DSSE envelope paired with a NIST FIPS 204 ML-DSA-65 hybrid co-signature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HybridEnvelope {
    pub envelope: Envelope,
    pub pq: PqCosignature,
}

/// Deterministic BLAKE3 fingerprint of a pinned ML-DSA-65 public key.
pub fn ml_dsa_65_fingerprint(pk_bytes: &[u8; ml_dsa_65::PK_LEN]) -> String {
    blake3::hash(pk_bytes).to_hex().to_string()
}

/// Seal `body` into a classical DSSE envelope AND co-sign its exact PAE bytes with ML-DSA-65.
pub fn seal_hybrid(
    payload_type: &str,
    body: &[u8],
    keyid: &str,
    ed25519_sign: impl FnOnce(&[u8]) -> [u8; 64],
    ml_dsa_pk: &ml_dsa_65::PublicKey,
    ml_dsa_sk: &ml_dsa_65::PrivateKey,
) -> Result<HybridEnvelope, ProofError> {
    let envelope = dsse::seal(payload_type, body, keyid, ed25519_sign);
    let pae_bytes = dsse::pae(payload_type, body);
    let sig_bytes = ml_dsa_sk
        .try_sign(&pae_bytes, PQ_DSSE_CONTEXT)
        .map_err(|e| ProofError(format!("ML-DSA-65 signing failed: {e}")))?;
    let pk_bytes = ml_dsa_pk.clone().into_bytes();
    Ok(HybridEnvelope {
        envelope,
        pq: PqCosignature {
            algorithm: PQ_HYBRID_ALG.to_string(),
            public_key_b64: STANDARD.encode(pk_bytes),
            signature_b64: STANDARD.encode(sig_bytes),
            key_fingerprint_blake3: ml_dsa_65_fingerprint(&pk_bytes),
        },
    })
}

/// Offline hybrid verification: BOTH classical Ed25519 (against `trusted_ed25519`) AND
/// NIST FIPS 204 ML-DSA-65 (against `trusted_pq_fingerprints`) must verify over the exact DSSE PAE.
pub fn open_hybrid(
    hybrid: &HybridEnvelope,
    trusted_ed25519: &[TrustedKey],
    trusted_pq_fingerprints: &[String],
    payload_type: &str,
) -> Result<(Vec<u8>, String, String), ProofError> {
    let (body, ed_did) = dsse::open(&hybrid.envelope, trusted_ed25519, payload_type)?;
    if hybrid.pq.algorithm != PQ_HYBRID_ALG {
        return err(format!(
            "unsupported PQ algorithm {:?}, expected {PQ_HYBRID_ALG:?}",
            hybrid.pq.algorithm
        ));
    }
    let pk_vec = STANDARD
        .decode(&hybrid.pq.public_key_b64)
        .map_err(|e| ProofError(format!("invalid ML-DSA-65 public key base64: {e}")))?;
    let pk_arr: [u8; ml_dsa_65::PK_LEN] = pk_vec
        .try_into()
        .map_err(|_| ProofError("ML-DSA-65 public key length mismatch".into()))?;
    let fp = ml_dsa_65_fingerprint(&pk_arr);
    if fp != hybrid.pq.key_fingerprint_blake3 {
        return err("ML-DSA-65 public key fingerprint does not match key bytes");
    }
    if !trusted_pq_fingerprints.iter().any(|t| t == &fp) {
        return err(format!("ML-DSA-65 key {fp} is not in trusted PQ fingerprints"));
    }
    let sig_vec = STANDARD
        .decode(&hybrid.pq.signature_b64)
        .map_err(|e| ProofError(format!("invalid ML-DSA-65 signature base64: {e}")))?;
    let sig_arr: [u8; ml_dsa_65::SIG_LEN] = sig_vec
        .try_into()
        .map_err(|_| ProofError("ML-DSA-65 signature length mismatch".into()))?;
    let pk = ml_dsa_65::PublicKey::try_from_bytes(pk_arr)
        .map_err(|e| ProofError(format!("malformed ML-DSA-65 public key: {e}")))?;
    let pae_bytes = dsse::pae(payload_type, &body);
    if !pk.verify(&pae_bytes, &sig_arr, PQ_DSSE_CONTEXT) {
        return err("ML-DSA-65 post-quantum signature failed verification");
    }
    Ok((body, ed_did, fp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsse::tests::key;
    use crate::dsse::INTOTO_PAYLOAD_TYPE;
    use ed25519_dalek::Signer as _;

    #[test]
    fn hybrid_fips204_dsse_verifies_offline_and_rejects_any_tamper() {
        let (sk, tk) = key(42);
        let (pq_pk, pq_sk) = ml_dsa_65::try_keygen().expect("ml-dsa-65 keygen");
        let fp = ml_dsa_65_fingerprint(&pq_pk.clone().into_bytes());
        let body = br#"{"_type":"https://in-toto.io/Statement/v1","ok":true}"#;

        let hybrid = seal_hybrid(
            INTOTO_PAYLOAD_TYPE,
            body,
            &tk.did,
            |m| sk.sign(m).to_bytes(),
            &pq_pk,
            &pq_sk,
        )
        .expect("seal hybrid");

        // 1. Valid verification succeeds offline.
        let (opened, did, verified_fp) = open_hybrid(
            &hybrid,
            std::slice::from_ref(&tk),
            std::slice::from_ref(&fp),
            INTOTO_PAYLOAD_TYPE,
        )
        .expect("verify hybrid");
        assert_eq!(opened, body);
        assert_eq!(did, tk.did);
        assert_eq!(verified_fp, fp);

        // 2. Untrusted PQ fingerprint is rejected even if Ed25519 is trusted.
        assert!(open_hybrid(&hybrid, std::slice::from_ref(&tk), &[], INTOTO_PAYLOAD_TYPE).is_err());

        // 3. Tampered ML-DSA-65 signature is rejected.
        let mut tampered_sig = hybrid.clone();
        let mut raw_sig = STANDARD.decode(&tampered_sig.pq.signature_b64).unwrap();
        raw_sig[0] ^= 0xff;
        tampered_sig.pq.signature_b64 = STANDARD.encode(raw_sig);
        assert!(
            open_hybrid(
                &tampered_sig,
                std::slice::from_ref(&tk),
                std::slice::from_ref(&fp),
                INTOTO_PAYLOAD_TYPE
            )
            .is_err()
        );
    }
}
