//! Full verification of a migrated log.
//!
//! `StrataLog::verify_tail` only scans the active segment. After `seal` that
//! segment is the new empty one, so its `frames_verified` is 0 while the
//! data lives in the sealed predecessor. This module re-checks every sealed
//! segment (frame chain + trailer signature), replays the log, and compares
//! per-kind frame counts to the MIGRATION_RECEIPT. Any mismatch is a hard error.

use std::path::Path;

use borsh::BorshDeserialize;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use strata::{Frame, FrameRecord, SegmentHeader, SegmentTrailer, StrataLog};
use strata_kernel::checkpoint::checkpoint_hash;
use strata_kernel::verify::verify_with_head;

use crate::records::{
    decode_edge, MigrationReceipt, KIND_CHECKPOINT, KIND_EDGE, KIND_FSRS_REVIEW, KIND_GENESIS,
    KIND_LEGACY_LINK, KIND_MIGRATION_RECEIPT, KIND_NODE, KIND_PARAMS, KIND_SUPERSESSION,
    KIND_TOMBSTONE,
};
use crate::{read_snapshot, MigrationError};

const HEADER_WIRE_SIZE: usize = 58;
const TRAILER_WIRE_SIZE: usize = 104;

/// Result of a passing full-log verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogVerify {
    /// Frames whose bytes, chain link, and (for sealed segments) trailer checked out.
    pub frames_verified: u64,
    /// Frames `read_frames` replayed. Must equal `frames_verified`.
    pub frames_total: u64,
}

/// Re-open `dir` and run the full check. The log lock must not be held.
pub fn verify_migrated_dir(dir: &Path) -> Result<LogVerify, MigrationError> {
    let opened =
        crate::open_migrated(dir).map_err(|e| MigrationError::Verify(format!("log open: {e}")))?;
    let anchor = read_snapshot(&opened.log)?
        .checkpoints
        .last()
        .map(checkpoint_hash)
        .unwrap_or([0u8; 32]);
    verify_open_log(&opened.log, dir, anchor)
}

pub(crate) fn verify_open_log(
    log: &StrataLog,
    dir: &Path,
    anchor: [u8; 32],
) -> Result<LogVerify, MigrationError> {
    let scanned = scan_sealed_segments(dir)?;
    let frames = log
        .read_frames(1)
        .map_err(|e| MigrationError::Verify(format!("frame replay: {e}")))?;
    let frames_total = frames.len() as u64;
    if scanned.frames_verified != frames_total {
        return Err(MigrationError::Verify(format!(
            "frames_verified {} != replayed frame count {frames_total}",
            scanned.frames_verified
        )));
    }
    if log.head().frames_total != frames_total {
        return Err(MigrationError::Verify(format!(
            "frames_verified {frames_total} != log head frames_total {}",
            log.head().frames_total
        )));
    }

    let snapshot = read_snapshot(log)?;
    let receipt = snapshot.receipt.as_ref().ok_or_else(|| {
        MigrationError::Verify("MIGRATION_RECEIPT missing from the replayed log".into())
    })?;
    if !receipt.verify_checksum() {
        return Err(MigrationError::Verify(
            "MIGRATION_RECEIPT checksum does not bind its body".into(),
        ));
    }
    if !receipt.verify_signature() {
        return Err(MigrationError::Verify(
            "MIGRATION_RECEIPT signature does not verify".into(),
        ));
    }
    check_kind_counts(&frames, receipt)?;

    let events = snapshot
        .reviews
        .iter()
        .map(|event| {
            let bytes = borsh::to_vec(event)
                .map_err(|e| MigrationError::Verify(format!("re-encode review event: {e}")))?;
            Ok((event.event_seq, blake3::hash(&bytes).into(), *event))
        })
        .collect::<Result<Vec<_>, MigrationError>>()?;
    verify_with_head(&snapshot.checkpoints, Some(anchor), events.into_iter())
        .map_err(|e| MigrationError::Verify(format!("kernel replay: {e}")))?;

    Ok(LogVerify {
        frames_verified: scanned.frames_verified,
        frames_total,
    })
}

struct Scan {
    frames_verified: u64,
}

fn scan_sealed_segments(dir: &Path) -> Result<Scan, MigrationError> {
    let segs = list_segments(dir)?;
    let vk = log_verifying_key(dir)?;
    let mut prev = [0u8; 32];
    let mut frames_verified = 0u64;
    let last = segs.len() - 1;
    for (idx, path) in segs.iter().enumerate() {
        let bytes = std::fs::read(path)?;
        let header = parse_header(&bytes).ok_or_else(|| {
            MigrationError::Verify(format!("segment {} header unreadable", path.display()))
        })?;
        if header.prev_segment_hash != prev {
            return Err(MigrationError::Verify(format!(
                "segment {} prev hash does not chain",
                path.display()
            )));
        }
        let header_hash = hash_slice(
            &borsh::to_vec(&header)
                .map_err(|e| MigrationError::Verify(format!("header encode: {e}")))?,
        );
        let scanned = scan_one(&bytes, header_hash)?;
        let is_last = idx == last;
        match scanned.stop {
            Stop::Trailer { trailer } => {
                if trailer.frame_count != scanned.frame_count {
                    return Err(MigrationError::Verify(format!(
                        "segment {} trailer frame_count {} != scanned {}",
                        path.display(),
                        trailer.frame_count,
                        scanned.frame_count
                    )));
                }
                let root = merkle_root(&scanned.leaves);
                if trailer.merkle_root != root {
                    return Err(MigrationError::Verify(format!(
                        "segment {} trailer merkle root mismatch",
                        path.display()
                    )));
                }
                let msg = signature_message(
                    &header.segment_id,
                    &header.prev_segment_hash,
                    &trailer.merkle_root,
                );
                let sig = Signature::from_bytes(&trailer.signature);
                if vk.verify(&msg, &sig).is_err() {
                    return Err(MigrationError::Verify(format!(
                        "segment {} trailer signature invalid",
                        path.display()
                    )));
                }
                frames_verified += scanned.frame_count;
                prev = hash_slice(&bytes);
            }
            Stop::Clean if is_last && scanned.frame_count == 0 => {
                // The segment rolled by seal(): no frames, no trailer.
            }
            Stop::Clean if is_last => {
                return Err(MigrationError::Verify(format!(
                    "segment {} holds {} unsealed frame(s); frames_verified would not cover them",
                    path.display(),
                    scanned.frame_count
                )));
            }
            Stop::Clean => {
                return Err(MigrationError::Verify(format!(
                    "sealed segment {} is missing its trailer",
                    path.display()
                )));
            }
            Stop::Torn { reason } => {
                return Err(MigrationError::Verify(format!(
                    "segment {} damaged: {reason}",
                    path.display()
                )));
            }
        }
    }
    Ok(Scan { frames_verified })
}

fn log_verifying_key(dir: &Path) -> Result<VerifyingKey, MigrationError> {
    let path = dir.join("strata.key");
    let bytes = std::fs::read(&path)
        .map_err(|e| MigrationError::Verify(format!("log signing key {}: {e}", path.display())))?;
    let seed: [u8; 32] = bytes
        .try_into()
        .map_err(|_| MigrationError::Verify(format!("{} is not 32 bytes", path.display())))?;
    let sk = ed25519_dalek::SigningKey::from_bytes(&seed);
    Ok(sk.verifying_key())
}

fn list_segments(dir: &Path) -> Result<Vec<std::path::PathBuf>, MigrationError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "seg") {
            out.push(path);
        }
    }
    out.sort();
    if out.is_empty() {
        return Err(MigrationError::Verify(format!(
            "no segment files in {}",
            dir.display()
        )));
    }
    Ok(out)
}

fn parse_header(bytes: &[u8]) -> Option<SegmentHeader> {
    if bytes.len() < HEADER_WIRE_SIZE {
        return None;
    }
    let header: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).ok()?;
    if header.magic != *b"STRTSEG1" || header.version != 1 {
        return None;
    }
    Some(header)
}

enum Stop {
    Clean,
    Trailer { trailer: SegmentTrailer },
    Torn { reason: String },
}

struct OneScan {
    frame_count: u64,
    leaves: Vec<[u8; 32]>,
    stop: Stop,
}

fn scan_one(bytes: &[u8], header_hash: [u8; 32]) -> Result<OneScan, MigrationError> {
    let mut frame_count = 0u64;
    let mut last_hash = header_hash;
    let mut leaves = Vec::new();
    let mut off = HEADER_WIRE_SIZE;
    loop {
        let rem = bytes.len().saturating_sub(off);
        if rem == 0 {
            return Ok(OneScan {
                frame_count,
                leaves,
                stop: Stop::Clean,
            });
        }
        if rem == TRAILER_WIRE_SIZE {
            let trailer = borsh::from_slice::<SegmentTrailer>(&bytes[off..])
                .map_err(|e| MigrationError::Verify(format!("trailer parse: {e}")))?;
            return Ok(OneScan {
                frame_count,
                leaves,
                stop: Stop::Trailer { trailer },
            });
        }
        if rem < 69 {
            return Ok(OneScan {
                frame_count,
                leaves,
                stop: Stop::Torn {
                    reason: "short frame tail".into(),
                },
            });
        }
        let (frame, used) = match parse_frame(&bytes[off..]) {
            Ok(parsed) => parsed,
            Err(reason) => {
                return Ok(OneScan {
                    frame_count,
                    leaves,
                    stop: Stop::Torn { reason },
                });
            }
        };
        if frame.payload_blake3 != payload_blake3(frame.kind, &frame.payload) {
            return Ok(OneScan {
                frame_count,
                leaves,
                stop: Stop::Torn {
                    reason: "payload blake3 mismatch".into(),
                },
            });
        }
        if frame.prev_frame_hash != last_hash {
            return Ok(OneScan {
                frame_count,
                leaves,
                stop: Stop::Torn {
                    reason: "frame chain link mismatch".into(),
                },
            });
        }
        last_hash = frame_hash(&frame)?;
        leaves.push(frame.payload_blake3);
        frame_count += 1;
        off += used;
    }
}

fn parse_frame(buf: &[u8]) -> Result<(Frame, usize), String> {
    let mut cursor = std::io::Cursor::new(buf);
    let frame = Frame::deserialize_reader(&mut cursor).map_err(|e| format!("frame parse: {e}"))?;
    let used = cursor.position() as usize;
    if used == 0 || used > buf.len() {
        return Err("frame parse consumed nothing".into());
    }
    Ok((frame, used))
}

fn check_kind_counts(
    frames: &[FrameRecord],
    receipt: &MigrationReceipt,
) -> Result<(), MigrationError> {
    let mut counts = std::collections::BTreeMap::<u8, u64>::new();
    for frame in frames {
        *counts.entry(frame.kind).or_default() += 1;
    }
    let count_of = |name: &str| -> u64 {
        receipt
            .body
            .counts
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    };
    let expect = [
        (
            "KIND_NODE",
            KIND_NODE,
            count_of("knowledge_nodes") + count_of("walk_receipts"),
        ),
        (
            "KIND_TOMBSTONE",
            KIND_TOMBSTONE,
            count_of("sync_tombstones") + count_of("deletion_tombstones"),
        ),
        ("KIND_GENESIS", KIND_GENESIS, 1),
        ("KIND_PARAMS", KIND_PARAMS, 1),
        ("KIND_CHECKPOINT", KIND_CHECKPOINT, 1),
        ("KIND_MIGRATION_RECEIPT", KIND_MIGRATION_RECEIPT, 1),
    ];
    for (name, kind, expected) in expect {
        let actual = counts.get(&kind).copied().unwrap_or(0);
        if actual != expected {
            return Err(MigrationError::Verify(format!(
                "{name}: log has {actual} frame(s), receipt accounting says {expected}"
            )));
        }
    }
    let edges = counts.get(&KIND_EDGE).copied().unwrap_or(0);
    let legacy_links = counts.get(&KIND_LEGACY_LINK).copied().unwrap_or(0);
    let connections = count_of("memory_connections");
    if edges + legacy_links != connections {
        return Err(MigrationError::Verify(format!(
            "KIND_EDGE ({edges}) + KIND_LEGACY_LINK ({legacy_links}) != memory_connections ({connections})"
        )));
    }
    for frame in frames.iter().filter(|f| f.kind == KIND_EDGE) {
        let edge = decode_edge(&frame.payload).map_err(|e| {
            MigrationError::Verify(format!("edge decode at seq {}: {e}", frame.seq))
        })?;
        if crate::is_noncausal_association(&edge.legacy_link_type) {
            return Err(MigrationError::Verify(format!(
                "seq {}: association link '{}' was written as a causal EDGE",
                frame.seq, edge.legacy_link_type
            )));
        }
    }
    let reviews = counts.get(&KIND_FSRS_REVIEW).copied().unwrap_or(0);
    let supers = counts.get(&KIND_SUPERSESSION).copied().unwrap_or(0);
    let accounted: u64 =
        expect.iter().map(|(_, _, n)| n).sum::<u64>() + edges + legacy_links + reviews + supers;
    if accounted != frames.len() as u64 {
        return Err(MigrationError::Verify(format!(
            "per-kind accounting covers {accounted} frame(s), log has {}",
            frames.len()
        )));
    }
    Ok(())
}

fn hash_slice(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn payload_blake3(kind: u8, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[kind]);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn frame_hash(frame: &Frame) -> Result<[u8; 32], MigrationError> {
    let bytes =
        borsh::to_vec(frame).map_err(|e| MigrationError::Verify(format!("frame encode: {e}")))?;
    Ok(hash_slice(&bytes))
}

fn merkle_root(payload_hashes: &[[u8; 32]]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = payload_hashes
        .iter()
        .map(|payload| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&[0x00]);
            hasher.update(payload);
            *hasher.finalize().as_bytes()
        })
        .collect();
    merkle_fold(&leaves)
}

fn merkle_fold(nodes: &[[u8; 32]]) -> [u8; 32] {
    match nodes.len() {
        0 => hash_slice(&[]),
        1 => nodes[0],
        n => {
            let mut k = 1usize;
            while k * 2 < n {
                k *= 2;
            }
            let mut hasher = blake3::Hasher::new();
            hasher.update(&[0x01]);
            hasher.update(&merkle_fold(&nodes[..k]));
            hasher.update(&merkle_fold(&nodes[k..]));
            *hasher.finalize().as_bytes()
        }
    }
}

fn signature_message(
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
