#![forbid(unsafe_code)]

//! Receipt-frame names the mesh fold does not expose.
//!
//! `bf561a07` folds per-writer store chains in the private
//! `strata_store::mesh::fold_chains`. That function takes `ChainIn` values
//! (`writer`, `FrameRecord`s, `admitted_to`) and returns `MeshProjection`
//! (`digest`, `frontier`, `nodes`, `reinforcements`, `conflicts`,
//! `review_counts`). It does not return `admitted_frames`, and a `strata::Frame`
//! has no `lamport`, no `timestamp_ms`, and no per-frame signature.
//!
//! This module is the thin name alignment for those missing receipt types.
//! `WriterId::did` is [`crate::did_key_from_ed25519_public_key`], the same
//! multicodec `strata_store::writer_did` documents. `GENESIS_PREV_HASH` is the
//! same 32 zero bytes as `strata::GENESIS_PREV_SEGMENT_HASH`.
//! `fold_writer_chains` checks the ed25519 signature and the `prev_hash` link.
//! It does not merge store effects and it does not compute a mesh digest.

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::did_key_from_ed25519_public_key;

/// Mesh hashes are `[u8; 32]`. This is the name the receipt code uses.
pub type Hash32 = [u8; 32];

/// 32 zero bytes, the same value as `strata::GENESIS_PREV_SEGMENT_HASH`.
///
/// A provenance chain's first frame uses this as `prev_hash`. Inside a Strata
/// segment the first frame's `prev_frame_hash` is the segment header hash.
pub const GENESIS_PREV_HASH: Hash32 = [0u8; 32];

/// Not `strata::payload_blake3`: that preimage is `kind || payload`.
const PAYLOAD_DOMAIN: &[u8] = b"strata.provenance.payload.v1\0";
/// Not `strata::frame_hash`: that preimage is the borsh frame wire.
const FRAME_DOMAIN: &[u8] = b"strata.provenance.signed-frame.v1\0";

/// 32-byte ed25519 verifying key of one writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WriterId(pub [u8; 32]);

impl WriterId {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// `did:key` for this verifying key.
    pub fn did(self) -> String {
        did_key_from_ed25519_public_key(&self.0)
    }
}

/// Why `SignedWriterFrame::verify_cryptography` refused a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameVerifyError {
    pub reason: &'static str,
}

/// One signed provenance frame.
///
/// Field names match the receipt drop. They are not `strata::Frame` fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedWriterFrame {
    pub writer_id: WriterId,
    pub seq: u64,
    pub lamport: u64,
    pub timestamp_ms: u64,
    pub prev_hash: Hash32,
    pub payload_digest: Hash32,
    pub frame_digest: Hash32,
    pub signature: Vec<u8>,
    pub payload: Vec<u8>,
}

impl SignedWriterFrame {
    pub fn compute_payload_digest(payload: &[u8]) -> Hash32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PAYLOAD_DOMAIN);
        hasher.update(payload);
        *hasher.finalize().as_bytes()
    }

    pub fn compute_frame_digest(
        writer_id: &WriterId,
        seq: u64,
        lamport: u64,
        timestamp_ms: u64,
        prev_hash: &Hash32,
        payload_digest: &Hash32,
    ) -> Hash32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(FRAME_DOMAIN);
        hasher.update(writer_id.as_bytes());
        hasher.update(&seq.to_le_bytes());
        hasher.update(&lamport.to_le_bytes());
        hasher.update(&timestamp_ms.to_le_bytes());
        hasher.update(prev_hash);
        hasher.update(payload_digest);
        *hasher.finalize().as_bytes()
    }

    pub fn verify_cryptography(&self) -> Result<(), FrameVerifyError> {
        let payload_digest = Self::compute_payload_digest(&self.payload);
        if payload_digest != self.payload_digest {
            return Err(FrameVerifyError {
                reason: "payload_digest",
            });
        }
        let frame_digest = Self::compute_frame_digest(
            &self.writer_id,
            self.seq,
            self.lamport,
            self.timestamp_ms,
            &self.prev_hash,
            &self.payload_digest,
        );
        if frame_digest != self.frame_digest {
            return Err(FrameVerifyError {
                reason: "frame_digest",
            });
        }
        let key = VerifyingKey::from_bytes(self.writer_id.as_bytes()).map_err(|_| {
            FrameVerifyError {
                reason: "writer_id",
            }
        })?;
        let sig_bytes: [u8; 64] = self.signature.as_slice().try_into().map_err(|_| {
            FrameVerifyError {
                reason: "signature_len",
            }
        })?;
        let signature = Signature::from_bytes(&sig_bytes);
        key.verify(&self.frame_digest, &signature)
            .map_err(|_| FrameVerifyError {
                reason: "signature",
            })
    }
}

/// A provenance frame the admission step refused.
///
/// Mesh equivocation stays `ImportReport.equivocation` on a diverging store
/// prefix. This receipt is only a provenance frame that failed its signature
/// or its `prev_hash` link, which `fold_chains` never sees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquivocationReceipt {
    pub writer_id: WriterId,
    pub seq: u64,
    pub frame_digest: Hash32,
    pub reason: String,
}

/// Last admitted provenance frame for one writer.
///
/// `MeshProjection.frontier` is `did -> seq`. `writer_did` is that did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriterChainTip {
    pub writer_id: WriterId,
    pub writer_did: String,
    pub seq: u64,
    pub frame_digest: Hash32,
}

/// Admission result. `admitted_frames` is the field the provenance tests read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterChainFold {
    pub admitted_frames: Vec<SignedWriterFrame>,
    pub equivocations: Vec<EquivocationReceipt>,
    pub tips: Vec<WriterChainTip>,
}

/// Admit frames whose signature verifies and whose `prev_hash` extends the
/// writer's admitted tip, in `seq` order. A frame that fails either check is
/// an [`EquivocationReceipt`] and does not extend the tip, so a later frame
/// whose `prev_hash` names it is not admitted.
pub fn fold_writer_chains(frames: &[SignedWriterFrame]) -> WriterChainFold {
    let mut by_writer: BTreeMap<WriterId, Vec<&SignedWriterFrame>> = BTreeMap::new();
    for frame in frames {
        by_writer.entry(frame.writer_id).or_default().push(frame);
    }

    let mut admitted_frames = Vec::new();
    let mut equivocations = Vec::new();
    let mut tips = Vec::new();

    for (writer_id, mut group) in by_writer {
        group.sort_by_key(|frame| (frame.seq, frame.frame_digest));
        let mut tip_hash: Option<Hash32> = None;
        let mut admitted_here = Vec::new();
        for frame in group {
            if let Err(err) = frame.verify_cryptography() {
                equivocations.push(EquivocationReceipt {
                    writer_id,
                    seq: frame.seq,
                    frame_digest: frame.frame_digest,
                    reason: err.reason.to_owned(),
                });
                continue;
            }
            let linked = match tip_hash {
                None => frame.prev_hash == GENESIS_PREV_HASH,
                Some(prev) => frame.prev_hash == prev,
            };
            if !linked {
                equivocations.push(EquivocationReceipt {
                    writer_id,
                    seq: frame.seq,
                    frame_digest: frame.frame_digest,
                    reason: "prev_hash does not extend the admitted tip".to_owned(),
                });
                continue;
            }
            admitted_here.push(frame.clone());
            tip_hash = Some(frame.frame_digest);
        }
        if let Some(last) = admitted_here.last() {
            tips.push(WriterChainTip {
                writer_id,
                writer_did: writer_id.did(),
                seq: last.seq,
                frame_digest: last.frame_digest,
            });
        }
        admitted_frames.extend(admitted_here);
    }

    WriterChainFold {
        admitted_frames,
        equivocations,
        tips,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_id_did_matches_the_actor_encoding_and_genesis_is_zero() {
        let id = WriterId::from_bytes([9u8; 32]);
        assert_eq!(id.did(), did_key_from_ed25519_public_key(id.as_bytes()));
        assert_eq!(GENESIS_PREV_HASH, [0u8; 32]);
    }
}
