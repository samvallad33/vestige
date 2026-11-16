//! Wire format for segment files and `head.state`, plus every hash rule.
//!
//! All structs use borsh canonical encoding (little-endian fixed-width
//! integers, fixed-size arrays as raw bytes, no maps in wire structs).
//! [`Frame`] implements the borsh traits manually so the payload carries
//! exactly ONE `u32` length prefix — a derived `Vec<u8>` field would emit a
//! second length prefix of its own.
//!
//! Hash definitions (blake3 everywhere):
//! * `payload_blake3(frame) = blake3(kind || payload)` — the kind byte is
//!   bound into the frame's payload hash.
//! * `frame_hash(frame) = blake3(frame wire bytes)` — the chain hash; each
//!   frame's `prev_frame_hash` is the previous frame's `frame_hash`, and the
//!   first frame's `prev_frame_hash` is the segment `header_hash`.
//! * Merkle tree over frames (RFC6962-shaped): `leaf = H(0x00 || payload_blake3)`,
//!   `node = H(0x01 || left || right)`, split at the largest power of two
//!   strictly below the node count; the empty tree hashes the empty string.
//! * `segment_hash = blake3(entire sealed segment file, header..=trailer)`;
//!   the next segment's `prev_segment_hash` must equal it.

use borsh::io;
use borsh::{BorshDeserialize, BorshSerialize};

pub const SEGMENT_MAGIC: [u8; 8] = *b"STRTSEG1";
pub const SEGMENT_VERSION: u16 = 1;
/// `prev_segment_hash` of segment 0 (all zeros, by definition).
pub const GENESIS_PREV_SEGMENT_HASH: [u8; 32] = [0u8; 32];

pub const HEADER_WIRE_SIZE: usize = 8 + 2 + 16 + 32; // 58
pub const TRAILER_WIRE_SIZE: usize = 8 + 32 + 64; // 104
/// Frame bytes minus payload: len u32 + kind u8 + payload_blake3 + prev_frame_hash.
pub const FRAME_FIXED_WIRE_SIZE: usize = 4 + 1 + 32 + 32; // 69

/// Segment header. Wire: `magic[8] || version u16 || segment_id[16] || prev_segment_hash[32]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct SegmentHeader {
    pub magic: [u8; 8],
    pub version: u16,
    pub segment_id: [u8; 16],
    pub prev_segment_hash: [u8; 32],
}

/// Segment trailer, written by `seal()`. Wire: `frame_count u64 || merkle_root[32] || signature[64]`.
/// The signature is ed25519 over `segment_id || prev_segment_hash || merkle_root`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct SegmentTrailer {
    pub frame_count: u64,
    pub merkle_root: [u8; 32],
    pub signature: [u8; 64],
}

/// A single frame. Wire: `len u32 || kind u8 || payload[len] || payload_blake3[32] || prev_frame_hash[32]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: u8,
    pub payload: Vec<u8>,
    pub payload_blake3: [u8; 32],
    pub prev_frame_hash: [u8; 32],
}

impl BorshSerialize for Frame {
    fn serialize<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let Ok(len) = u32::try_from(self.payload.len()) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "strata: payload length exceeds u32",
            ));
        };
        len.serialize(writer)?;
        writer.write_all(&[self.kind])?;
        writer.write_all(&self.payload)?;
        writer.write_all(&self.payload_blake3)?;
        writer.write_all(&self.prev_frame_hash)?;
        Ok(())
    }
}

impl BorshDeserialize for Frame {
    fn deserialize_reader<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let len = u32::deserialize_reader(reader)?;
        let mut kind = [0u8; 1];
        reader.read_exact(&mut kind)?;
        let mut payload = vec![0u8; len as usize];
        reader.read_exact(&mut payload)?;
        let mut payload_blake3 = [0u8; 32];
        reader.read_exact(&mut payload_blake3)?;
        let mut prev_frame_hash = [0u8; 32];
        reader.read_exact(&mut prev_frame_hash)?;
        Ok(Frame {
            kind: kind[0],
            payload,
            payload_blake3,
            prev_frame_hash,
        })
    }
}

/// A decoded frame as returned by [`crate::StrataLog::read_frames`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRecord {
    /// Global sequence number (first frame in the log is seq 1).
    pub seq: u64,
    pub kind: u8,
    pub payload: Vec<u8>,
    pub payload_blake3: [u8; 32],
    pub prev_frame_hash: [u8; 32],
    /// blake3 of the frame's full wire encoding — the chain hash.
    pub frame_hash: [u8; 32],
}

pub(crate) fn hash_slice(bytes: &[u8]) -> [u8; 32] {
    blake3::hash(bytes).into()
}

/// `payload_blake3 = blake3(kind || payload)`. The kind byte is bound in.
pub fn payload_blake3(kind: u8, payload: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[kind]);
    h.update(payload);
    h.finalize().into()
}

/// Chain hash of a frame: blake3 over its full wire encoding.
pub fn frame_hash(frame: &Frame) -> [u8; 32] {
    hash_slice(&borsh::to_vec(frame).expect("strata: frame serializes"))
}

/// Chain hash of a segment header: blake3 over its wire encoding.
pub fn header_hash(header: &SegmentHeader) -> [u8; 32] {
    hash_slice(&borsh::to_vec(header).expect("strata: header serializes"))
}

fn leaf_hash(payload_blake3: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[0x00]);
    h.update(payload_blake3);
    h.finalize().into()
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[0x01]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// RFC6962-style merkle root over the frames' `payload_blake3` values.
/// Inputs are the raw per-frame payload hashes; leaves are domain-separated
/// inside. The empty-input root is blake3 of the empty string.
pub fn merkle_root(payload_hashes: &[[u8; 32]]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = payload_hashes.iter().map(leaf_hash).collect();
    merkle_fold(&leaves)
}

fn merkle_fold(hs: &[[u8; 32]]) -> [u8; 32] {
    match hs.len() {
        0 => hash_slice(&[]),
        1 => hs[0],
        n => {
            // Largest power of two strictly below n (RFC6962 split).
            let mut k = 1usize;
            while k * 2 < n {
                k *= 2;
            }
            node_hash(&merkle_fold(&hs[..k]), &merkle_fold(&hs[k..]))
        }
    }
}

/// Ed25519 signature message: `segment_id || prev_segment_hash || merkle_root`.
pub fn signature_message(
    segment_id: &[u8; 16],
    prev_segment_hash: &[u8; 32],
    merkle_root: &[u8; 32],
) -> [u8; 80] {
    let mut msg = [0u8; 80];
    msg[..16].copy_from_slice(segment_id);
    msg[16..48].copy_from_slice(prev_segment_hash);
    msg[48..80].copy_from_slice(merkle_root);
    msg
}

/// Parse one frame from the front of `buf`.
/// Returns the frame and the number of wire bytes consumed.
pub fn parse_frame(buf: &[u8]) -> io::Result<(Frame, usize)> {
    let mut cursor: &[u8] = buf;
    let frame = Frame::deserialize_reader(&mut cursor)?;
    let consumed = buf.len() - cursor.len();
    Ok((frame, consumed))
}

/// RFC 9162 consistency proof from the first `old_len` leaves to the full
/// list. Leaves are the frames' `payload_blake3` values, in chain order.
/// The tree shape is the same one [`merkle_root`] uses. An empty proof means
/// `old_len` is 0 or already the full list.
pub fn consistency_proof(payload_hashes: &[[u8; 32]], old_len: usize) -> Vec<[u8; 32]> {
    if old_len == 0 || old_len >= payload_hashes.len() {
        return Vec::new();
    }
    let leaves: Vec<[u8; 32]> = payload_hashes.iter().map(leaf_hash).collect();
    subproof(&leaves, old_len, true)
}

fn subproof(leaves: &[[u8; 32]], m: usize, complete: bool) -> Vec<[u8; 32]> {
    let n = leaves.len();
    if m == n {
        if complete {
            Vec::new()
        } else {
            vec![merkle_fold(leaves)]
        }
    } else {
        let mut k = 1usize;
        while k * 2 < n {
            k *= 2;
        }
        if m <= k {
            let mut proof = subproof(&leaves[..k], m, complete);
            proof.push(merkle_fold(&leaves[k..]));
            proof
        } else {
            let mut proof = subproof(&leaves[k..], m - k, false);
            proof.push(merkle_fold(&leaves[..k]));
            proof
        }
    }
}

fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Verify a [`consistency_proof`] between two merkle roots of this log's tree.
/// `first == second` is consistent only when the roots match and the proof is empty.
pub fn verify_consistency(
    first: usize,
    second: usize,
    first_hash: &[u8; 32],
    second_hash: &[u8; 32],
    proof: &[[u8; 32]],
) -> bool {
    if first > second {
        return false;
    }
    if first == second {
        return proof.is_empty() && first_hash == second_hash;
    }
    if first == 0 || proof.is_empty() {
        return false;
    }
    let mut path = proof.to_vec();
    if is_pow2(first) {
        path.insert(0, *first_hash);
    }
    let mut fn_idx = first - 1;
    let mut sn = second - 1;
    while fn_idx & 1 == 1 {
        fn_idx >>= 1;
        sn >>= 1;
    }
    let mut nodes = path.into_iter();
    let Some(seed) = nodes.next() else {
        return false;
    };
    let mut fr = seed;
    let mut sr = seed;
    for node in nodes {
        if sn == 0 {
            return false;
        }
        if fn_idx & 1 == 1 || fn_idx == sn {
            fr = node_hash(&node, &fr);
            sr = node_hash(&node, &sr);
            if fn_idx & 1 == 0 {
                while fn_idx != 0 && fn_idx & 1 == 0 {
                    fn_idx >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            sr = node_hash(&sr, &node);
        }
        fn_idx >>= 1;
        sn >>= 1;
    }
    fr == *first_hash && sr == *second_hash && sn == 0
}

#[cfg(test)]
mod consistency_tests {
    use super::{consistency_proof, merkle_root, verify_consistency};

    #[test]
    fn consistency_proofs_hold_for_every_prefix_up_to_24_leaves() {
        let leaves: Vec<[u8; 32]> = (0..24u8)
            .map(|i| {
                let mut leaf = [0u8; 32];
                leaf[0] = i;
                leaf[31] = i.wrapping_mul(3);
                leaf
            })
            .collect();
        for n in 1..=leaves.len() {
            let new_root = merkle_root(&leaves[..n]);
            for m in 1..n {
                let old_root = merkle_root(&leaves[..m]);
                let proof = consistency_proof(&leaves[..n], m);
                assert!(
                    verify_consistency(m, n, &old_root, &new_root, &proof),
                    "proof failed for {m} -> {n} ({} nodes)",
                    proof.len()
                );
                let mut bad = old_root;
                bad[0] ^= 0xff;
                assert!(!verify_consistency(m, n, &bad, &new_root, &proof));
            }
            assert!(verify_consistency(n, n, &new_root, &new_root, &[]));
        }
    }
}
