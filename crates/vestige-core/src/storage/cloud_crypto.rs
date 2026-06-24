//! Zero-knowledge client-side encryption for Vestige Cloud sync.
//!
//! The portable archive is encrypted on the client **before** it is uploaded
//! and decrypted **after** it is downloaded, so the hosted service only ever
//! stores ciphertext. The encryption passphrase is supplied by the user
//! (`VESTIGE_CLOUD_ENCRYPTION_KEY`) and is **never** sent to the server — it is
//! independent of the bearer sync key. This is what makes the "we hold no keys"
//! guarantee literally true: a server breach yields only random noise.
//!
//! Construction:
//! - KDF: Argon2id over (passphrase, random 16-byte salt) → 32-byte key.
//! - AEAD: XChaCha20-Poly1305 (192-bit nonce) over the archive bytes.
//! - Envelope (all non-secret framing prepended to the ciphertext):
//!   `MAGIC(8) | VERSION(1) | salt(16) | nonce(24) | ciphertext+tag`
//!
//! Tradeoff (by design, and a selling point): if the user loses the passphrase,
//! the synced data is unrecoverable. We cannot reset it — we never have it.

use argon2::Argon2;
use chacha20poly1305::aead::rand_core::RngCore;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};

use super::sqlite::{Result, StorageError};

/// Magic marker identifying a Vestige zero-knowledge envelope.
const MAGIC: &[u8; 8] = b"VSTGENC1";
const VERSION: u8 = 1;
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 24; // XChaCha20-Poly1305 nonce is 192-bit.
const KEY_LEN: usize = 32;
const HEADER_LEN: usize = MAGIC.len() + 1 + SALT_LEN + NONCE_LEN;

/// Derive a 32-byte key from the passphrase and salt using Argon2id (defaults).
fn derive_key(passphrase: &[u8], salt: &[u8]) -> Result<[u8; KEY_LEN]> {
    let mut key = [0u8; KEY_LEN];
    Argon2::default()
        .hash_password_into(passphrase, salt, &mut key)
        .map_err(|e| StorageError::Init(format!("key derivation failed: {e}")))?;
    Ok(key)
}

/// Encrypt `plaintext` under `passphrase`, returning the self-describing envelope.
///
/// A fresh random salt and nonce are generated per call, so re-encrypting the
/// same archive yields different ciphertext (no deterministic leakage).
pub fn encrypt(passphrase: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
    let mut salt = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut salt);

    let key_bytes = derive_key(passphrase.as_bytes(), &salt)?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key_bytes));
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|_| StorageError::Init("encryption failed".to_string()))?;

    let mut out = Vec::with_capacity(HEADER_LEN + ciphertext.len());
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&salt);
    out.extend_from_slice(nonce.as_slice());
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// True if `bytes` start with the Vestige encryption magic. Lets the sync
/// backend distinguish an encrypted envelope from a legacy plaintext archive.
pub fn is_encrypted(bytes: &[u8]) -> bool {
    bytes.len() >= MAGIC.len() && &bytes[..MAGIC.len()] == MAGIC
}

/// Decrypt a Vestige envelope under `passphrase`. Fails on a wrong passphrase or
/// any tampering (the Poly1305 tag is verified).
pub fn decrypt(passphrase: &str, envelope: &[u8]) -> Result<Vec<u8>> {
    if envelope.len() < HEADER_LEN {
        return Err(StorageError::Init(
            "cloud archive is too short to be a valid encrypted envelope".to_string(),
        ));
    }
    if &envelope[..MAGIC.len()] != MAGIC {
        return Err(StorageError::Init(
            "cloud archive is not a Vestige encrypted envelope".to_string(),
        ));
    }
    let version = envelope[MAGIC.len()];
    if version != VERSION {
        return Err(StorageError::Init(format!(
            "unsupported cloud encryption version {version}"
        )));
    }

    let salt_start = MAGIC.len() + 1;
    let nonce_start = salt_start + SALT_LEN;
    let ct_start = nonce_start + NONCE_LEN;
    let salt = &envelope[salt_start..nonce_start];
    let nonce = XNonce::from_slice(&envelope[nonce_start..ct_start]);
    let ciphertext = &envelope[ct_start..];

    let key_bytes = derive_key(passphrase.as_bytes(), salt)?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key_bytes));

    cipher.decrypt(nonce, ciphertext).map_err(|_| {
        StorageError::Init(
            "cloud decryption failed: wrong VESTIGE_CLOUD_ENCRYPTION_KEY or corrupted data"
                .to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let pass = "correct horse battery staple";
        let msg = b"the entire cognitive graph, in plaintext, before upload";
        let env = encrypt(pass, msg).unwrap();
        assert!(is_encrypted(&env));
        assert_ne!(&env[..], &msg[..], "envelope must not contain plaintext");
        let back = decrypt(pass, &env).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn wrong_passphrase_fails() {
        let env = encrypt("right-pass", b"secret").unwrap();
        assert!(decrypt("wrong-pass", &env).is_err());
    }

    #[test]
    fn tamper_is_detected() {
        let mut env = encrypt("pass", b"important memory").unwrap();
        // Flip a byte in the ciphertext region.
        let last = env.len() - 1;
        env[last] ^= 0xff;
        assert!(decrypt("pass", &env).is_err(), "AEAD must reject tampering");
    }

    #[test]
    fn ciphertext_is_nondeterministic() {
        // Same input encrypted twice → different envelopes (random salt+nonce).
        let a = encrypt("p", b"x").unwrap();
        let b = encrypt("p", b"x").unwrap();
        assert_ne!(a, b);
        // Both still decrypt correctly.
        assert_eq!(decrypt("p", &a).unwrap(), b"x");
        assert_eq!(decrypt("p", &b).unwrap(), b"x");
    }

    #[test]
    fn plaintext_is_not_misdetected_as_envelope() {
        assert!(!is_encrypted(b"{\"archiveFormat\":\"vestige.portable.v1\"}"));
        assert!(!is_encrypted(b""));
    }

    #[test]
    fn rejects_short_or_foreign_envelope() {
        assert!(decrypt("p", b"too short").is_err());
        assert!(decrypt("p", &[0u8; 100]).is_err());
    }
}
