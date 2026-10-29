//! `did:key` for Ed25519, the form a trusted key is pinned in: multibase
//! base58btc (a leading `z`) of the multicodec prefix `0xed 0x01` and the
//! 32-byte public key. The same encoding `vestige-core` gives the store's
//! actor, so a bundle's `keyid` is the store's own identity.

use crate::{ProofError, err};

const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const ED25519_PREFIX: [u8; 2] = [0xed, 0x01];

fn base58_encode(bytes: &[u8]) -> String {
    let zeros = bytes.iter().take_while(|b| **b == 0).count();
    let mut digits: Vec<u8> = Vec::new();
    for &byte in &bytes[zeros..] {
        let mut carry = u32::from(byte);
        for d in digits.iter_mut() {
            carry += u32::from(*d) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let mut out = String::with_capacity(zeros + digits.len());
    out.extend(std::iter::repeat_n('1', zeros));
    out.extend(digits.iter().rev().map(|d| ALPHABET[*d as usize] as char));
    out
}

fn base58_decode(text: &str) -> Option<Vec<u8>> {
    let zeros = text.bytes().take_while(|b| *b == b'1').count();
    let mut bytes: Vec<u8> = Vec::new();
    for c in text.bytes().skip(zeros) {
        let mut carry = ALPHABET.iter().position(|a| *a == c)? as u32;
        for b in bytes.iter_mut() {
            carry += u32::from(*b) * 58;
            *b = (carry & 0xff) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            bytes.push((carry & 0xff) as u8);
            carry >>= 8;
        }
    }
    let mut out = vec![0u8; zeros];
    out.extend(bytes.iter().rev());
    Some(out)
}

/// `did:key:z6Mk…` for an Ed25519 public key.
pub fn from_public_key(public_key: &[u8; 32]) -> String {
    let mut payload = ED25519_PREFIX.to_vec();
    payload.extend_from_slice(public_key);
    format!("did:key:z{}", base58_encode(&payload))
}

/// The Ed25519 public key a `did:key` names. Anything else is refused, and the
/// text must be the one canonical spelling of that key.
pub fn to_public_key(did: &str) -> Result<[u8; 32], ProofError> {
    let Some(body) = did.strip_prefix("did:key:z") else {
        return err(format!("{did:?} is not a base58btc did:key"));
    };
    let Some(payload) = base58_decode(body) else {
        return err(format!("{did:?} is not valid base58btc"));
    };
    if payload.len() != 34 || payload[..2] != ED25519_PREFIX {
        return err(format!("{did:?} is not an Ed25519 did:key"));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&payload[2..]);
    if from_public_key(&key) != did {
        return err(format!("{did:?} is not the canonical spelling of its key"));
    }
    Ok(key)
}

/// A key a verifier trusts, pinned by its `did:key`.
#[derive(Clone, Debug)]
pub struct TrustedKey {
    pub did: String,
    pub key: ed25519_dalek::VerifyingKey,
}

impl TrustedKey {
    pub fn parse(did: &str) -> Result<Self, ProofError> {
        let did = did.trim();
        let bytes = to_public_key(did)?;
        let key = ed25519_dalek::VerifyingKey::from_bytes(&bytes)
            .map_err(|e| ProofError(format!("{did:?} is not a valid Ed25519 point: {e}")))?;
        Ok(Self {
            did: did.to_string(),
            key,
        })
    }

    /// Keys from a text file: one `did:key` per line, `#` starts a comment.
    pub fn parse_list(text: &str) -> Result<Vec<Self>, ProofError> {
        text.lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .filter(|l| !l.is_empty())
            .map(Self::parse)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base58_matches_the_reference_vector() {
        assert_eq!(base58_encode(b"Hello World!"), "2NEpo7TZRRrLZSi2U");
        assert_eq!(
            base58_decode("2NEpo7TZRRrLZSi2U").unwrap(),
            b"Hello World!".to_vec()
        );
        assert_eq!(base58_encode(&[0, 0, 1]), "112");
        assert_eq!(base58_decode("112").unwrap(), vec![0, 0, 1]);
        assert!(base58_decode("0OIl").is_none());
    }

    #[test]
    fn a_did_key_round_trips_and_starts_z6mk() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]).verifying_key();
        let did = from_public_key(&key.to_bytes());
        assert!(did.starts_with("did:key:z6Mk"), "{did}");
        assert_eq!(to_public_key(&did).unwrap(), key.to_bytes());
        assert_eq!(TrustedKey::parse(&did).unwrap().key, key);
    }

    #[test]
    fn foreign_or_altered_did_keys_are_refused() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]).verifying_key();
        let did = from_public_key(&key.to_bytes());
        assert!(to_public_key("did:web:example.com").is_err());
        assert!(to_public_key(&did.replacen("z6Mk", "z6Mj", 1)).is_err());
        assert!(to_public_key(&format!("{did}1")).is_err());
        let list = format!("# owner key\n{did}  # Sam's Mac\n\n");
        assert_eq!(TrustedKey::parse_list(&list).unwrap().len(), 1);
    }
}
