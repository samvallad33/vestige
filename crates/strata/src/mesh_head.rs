//! Signed tree head over one writer's frames.
//!
//! Issue #401's signed tree head is not in this tree. A mesh head is an
//! ed25519 signature, from the writer's existing log key, over the acked
//! length, the last frame hash, and the RFC 6962 merkle root [`crate::format::merkle_root`]
//! already computes. The replica checks it with the pinned public key and
//! never sees the seed.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::format::merkle_root;

/// Domain separator for [`head_message`].
pub const MESH_HEAD_DOMAIN: &[u8] = b"vestige.mesh.signed-head.v1\0";

/// A writer's signed head. The signature covers [`head_message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedHead {
    /// Acked frame count. Seq of the last frame, or 0 when the log is empty.
    pub seq: u64,
    /// Chain hash of the last frame. All zeros when `seq` is 0.
    pub frame_hash: [u8; 32],
    /// Merkle root over every frame's `payload_blake3`, in order.
    pub merkle_root: [u8; 32],
    /// Ed25519 signature by the writer key.
    pub signature: [u8; 64],
}

/// Bytes the writer signs and the replica verifies.
pub fn head_message(seq: u64, frame_hash: &[u8; 32], merkle_root: &[u8; 32]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(MESH_HEAD_DOMAIN.len() + 8 + 64);
    msg.extend_from_slice(MESH_HEAD_DOMAIN);
    msg.extend_from_slice(&seq.to_le_bytes());
    msg.extend_from_slice(frame_hash);
    msg.extend_from_slice(merkle_root);
    msg
}

/// Sign a head with the log's signing key.
pub fn sign_head(
    signing: &SigningKey,
    seq: u64,
    frame_hash: [u8; 32],
    payload_hashes: &[[u8; 32]],
) -> SignedHead {
    let root = merkle_root(payload_hashes);
    let msg = head_message(seq, &frame_hash, &root);
    let signature = signing.sign(&msg).to_bytes();
    SignedHead {
        seq,
        frame_hash,
        merkle_root: root,
        signature,
    }
}

/// Verify `head` under `key`. The merkle root and frame hash are the caller's.
pub fn verify_head(key: &VerifyingKey, head: &SignedHead) -> bool {
    let Ok(signature) = Signature::from_slice(&head.signature) else {
        return false;
    };
    let msg = head_message(head.seq, &head.frame_hash, &head.merkle_root);
    key.verify(&msg, &signature).is_ok()
}
