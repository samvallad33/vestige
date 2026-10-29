//! DSSE v1 envelopes (github.com/secure-systems-lab/dsse).
//!
//! The signature covers `PAE(payloadType, payload)`. Opening an envelope is
//! strict, so that any changed byte of a stored envelope fails it:
//!
//! - exactly one signature, from a pinned key;
//! - its `keyid` must be that key's `did:key` (DSSE treats keyid as a hint;
//!   pinning it as well makes the whole envelope tamper-evident);
//! - both base64 fields must be the one canonical standard encoding of what
//!   they decode to;
//! - no field beyond the three DSSE names.
//!
//! A signature proves which exact bytes the pinned key signed. It does not
//! prove the key stayed out of an agent's reach, and it cannot rule out a
//! second, different bundle signed by the same key.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};

use crate::didkey::TrustedKey;
use crate::{ProofError, err};

/// The payload type of an in-toto Statement.
pub const INTOTO_PAYLOAD_TYPE: &str = "application/vnd.in-toto+json";
/// The payload type of one witness record.
pub const WITNESS_PAYLOAD_TYPE: &str = "application/vnd.vestige.witness.v1+json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    #[serde(rename = "payloadType")]
    pub payload_type: String,
    pub payload: String,
    pub signatures: Vec<Signature>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signature {
    pub keyid: String,
    pub sig: String,
}

/// DSSE's pre-authentication encoding.
pub fn pae(payload_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "DSSEv1 {} {} {} ",
        payload_type.len(),
        payload_type,
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

/// Sign `body` as one envelope. `sign` is handed the PAE bytes and returns the
/// Ed25519 signature; the secret key never enters this crate.
pub fn seal(
    payload_type: &str,
    body: &[u8],
    keyid: &str,
    sign: impl FnOnce(&[u8]) -> [u8; 64],
) -> Envelope {
    let sig = sign(&pae(payload_type, body));
    Envelope {
        payload_type: payload_type.to_string(),
        payload: STANDARD.encode(body),
        signatures: vec![Signature {
            keyid: keyid.to_string(),
            sig: STANDARD.encode(sig),
        }],
    }
}

fn canonical_base64(field: &str, text: &str) -> Result<Vec<u8>, ProofError> {
    let bytes = STANDARD
        .decode(text)
        .map_err(|e| ProofError(format!("{field} is not standard base64: {e}")))?;
    if STANDARD.encode(&bytes) != text {
        return err(format!("{field} is not the canonical base64 of its bytes"));
    }
    Ok(bytes)
}

/// Check the envelope against the pinned keys and return the payload bytes and
/// the `did:key` that signed them.
pub fn open(
    env: &Envelope,
    trusted: &[TrustedKey],
    payload_type: &str,
) -> Result<(Vec<u8>, String), ProofError> {
    if env.payload_type != payload_type {
        return err(format!(
            "payloadType is {:?}, expected {payload_type:?}",
            env.payload_type
        ));
    }
    let [signature] = env.signatures.as_slice() else {
        return err(format!(
            "the envelope carries {} signatures, exactly one is accepted",
            env.signatures.len()
        ));
    };
    let Some(key) = trusted.iter().find(|k| k.did == signature.keyid) else {
        return err(format!(
            "signed by {:?}, which is not a trusted key",
            signature.keyid
        ));
    };
    let body = canonical_base64("payload", &env.payload)?;
    let sig = canonical_base64("sig", &signature.sig)?;
    let sig: [u8; 64] = sig
        .try_into()
        .map_err(|_| ProofError("sig is not 64 bytes".into()))?;
    let sig = ed25519_dalek::Signature::from_bytes(&sig);
    key.key
        .verify_strict(&pae(&env.payload_type, &body), &sig)
        .map_err(|_| ProofError(format!("the signature by {} does not verify", key.did)))?;
    Ok((body, key.did.clone()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::didkey;
    use ed25519_dalek::Signer as _;

    pub(crate) fn key(seed: u8) -> (ed25519_dalek::SigningKey, TrustedKey) {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
        let did = didkey::from_public_key(&sk.verifying_key().to_bytes());
        let tk = TrustedKey::parse(&did).unwrap();
        (sk, tk)
    }

    pub(crate) fn sealed(
        sk: &ed25519_dalek::SigningKey,
        did: &str,
        payload_type: &str,
        body: &[u8],
    ) -> Envelope {
        seal(payload_type, body, did, |m| sk.sign(m).to_bytes())
    }

    #[test]
    fn pae_matches_the_dsse_spec_example() {
        assert_eq!(
            pae("http://example.com/HelloWorld", b"hello world"),
            b"DSSEv1 29 http://example.com/HelloWorld 11 hello world".to_vec()
        );
    }

    #[test]
    fn an_envelope_opens_only_for_its_type_and_its_pinned_key() {
        let (sk, tk) = key(1);
        let (_, other) = key(2);
        let env = sealed(&sk, &tk.did, WITNESS_PAYLOAD_TYPE, b"{\"a\":1}");
        let (body, did) = open(&env, std::slice::from_ref(&tk), WITNESS_PAYLOAD_TYPE).unwrap();
        assert_eq!(body, b"{\"a\":1}");
        assert_eq!(did, tk.did);
        assert!(open(&env, &[other], WITNESS_PAYLOAD_TYPE).is_err());
        assert!(open(&env, std::slice::from_ref(&tk), INTOTO_PAYLOAD_TYPE).is_err());
    }

    #[test]
    fn a_second_signature_a_changed_payload_or_a_loose_base64_is_refused() {
        let (sk, tk) = key(3);
        let env = sealed(&sk, &tk.did, INTOTO_PAYLOAD_TYPE, b"payload");
        let trusted = std::slice::from_ref(&tk);
        let mut two = env.clone();
        two.signatures.push(two.signatures[0].clone());
        assert!(open(&two, trusted, INTOTO_PAYLOAD_TYPE).is_err());
        let mut changed = env.clone();
        changed.payload = STANDARD.encode(b"payloaD");
        assert!(open(&changed, trusted, INTOTO_PAYLOAD_TYPE).is_err());
        let mut loose = env.clone();
        loose.payload = loose.payload.trim_end_matches('=').to_string();
        assert!(open(&loose, trusted, INTOTO_PAYLOAD_TYPE).is_err());
    }
}
