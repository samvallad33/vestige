//! Read-only segment scan. Never creates, truncates, or locks anything in
//! the directory it reads: no `StrataLog::open`, no `strata.key`, no segment.

use std::fs;
use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, SigningKey, Verifier};
use strata::{
    FRAME_FIXED_WIRE_SIZE, GENESIS_PREV_SEGMENT_HASH, HEADER_WIRE_SIZE, SEGMENT_MAGIC,
    SEGMENT_VERSION, SegmentHeader, SegmentTrailer, TRAILER_WIRE_SIZE, frame_hash, header_hash,
    merkle_root, parse_frame, payload_blake3, signature_message,
};

/// One durable frame, in log order.
pub struct ScannedFrame {
    /// Frame kind byte.
    pub kind: u8,
    /// Frame payload. The scanner already checked its hash.
    pub payload: Vec<u8>,
}

/// A chain-valid scan of every segment in a directory.
pub struct Scan {
    /// Frames in log order.
    pub frames: Vec<ScannedFrame>,
    /// Segment files that passed the chain check.
    pub segments: u32,
}

/// True when `dir` itself (not a child) contains a `*.seg` file.
pub(crate) fn dir_has_segments(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "seg"))
}

/// Walk every segment in `dir`, checking the hash chain, payload hashes,
/// and sealed-trailer signatures. A torn tail is an error: this scanner
/// does not truncate it.
pub fn scan_log(dir: &Path) -> Result<Scan, String> {
    if !dir.is_dir() {
        return Err(format!("not a directory: {}", dir.display()));
    }
    let verifying = read_verifying_key(dir)?;
    let segs = list_segments(dir)?;
    if segs.is_empty() {
        return Err(format!("no segment files in {}", dir.display()));
    }

    let mut prev_hash = GENESIS_PREV_SEGMENT_HASH;
    let mut frames = Vec::new();
    let last = segs.len() - 1;

    for (idx, (no, path)) in segs.iter().enumerate() {
        let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let header = parse_header(&bytes)
            .ok_or_else(|| format!("segment {no}: header unreadable (bad magic/version/length)"))?;
        if header.prev_segment_hash != prev_hash {
            return Err(format!(
                "segment {no}: prev_segment_hash does not match the previous segment"
            ));
        }
        let (stop, scanned) = scan_frames(&bytes, header_hash(&header), &mut frames, *no)?;
        let is_last = idx == last;
        match stop {
            Tail::Trailer { trailer } => {
                validate_trailer(&trailer, &scanned, &header, &verifying)
                    .map_err(|reason| format!("segment {no}: {reason}"))?;
                prev_hash = *blake3::hash(&bytes).as_bytes();
            }
            Tail::Clean if is_last => {}
            Tail::Clean => {
                return Err(format!(
                    "segment {no}: sealed segment is missing its trailer"
                ));
            }
            Tail::Torn { offset, reason } => {
                return Err(format!(
                    "segment {no}: damaged at offset {offset}: {reason}"
                ));
            }
        }
    }

    Ok(Scan {
        frames,
        segments: segs.len() as u32,
    })
}

fn read_verifying_key(dir: &Path) -> Result<ed25519_dalek::VerifyingKey, String> {
    let path = dir.join("strata.key");
    let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let seed: [u8; 32] = bytes
        .try_into()
        .map_err(|_| "strata.key is not 32 bytes".to_string())?;
    let signing = SigningKey::from_bytes(&seed);
    Ok(signing.verifying_key())
}

fn list_segments(dir: &Path) -> Result<Vec<(u32, PathBuf)>, String> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(stem) = name.strip_suffix(".seg") else {
            continue;
        };
        let (num, hex) = stem
            .split_once('-')
            .ok_or_else(|| format!("bad segment name: {name}"))?;
        if num.len() != 8
            || !num.bytes().all(|b| b.is_ascii_digit())
            || hex.len() != 32
            || !hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(format!("bad segment name: {name}"));
        }
        let no: u32 = num
            .parse()
            .map_err(|_| format!("segment number out of range: {name}"))?;
        out.push((no, entry.path()));
    }
    out.sort_by_key(|(no, _)| *no);
    for (i, (no, _)) in out.iter().enumerate() {
        if *no != i as u32 {
            return Err(format!(
                "segment numbering not dense: expected {i:08}, found {no:08}"
            ));
        }
    }
    Ok(out)
}

fn parse_header(bytes: &[u8]) -> Option<SegmentHeader> {
    if bytes.len() < HEADER_WIRE_SIZE {
        return None;
    }
    let header: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).ok()?;
    if header.magic != SEGMENT_MAGIC || header.version != SEGMENT_VERSION {
        return None;
    }
    Some(header)
}

struct ScannedTail {
    frame_count: u64,
    leaves: Vec<[u8; 32]>,
}

enum Tail {
    Clean,
    Trailer { trailer: SegmentTrailer },
    Torn { offset: usize, reason: String },
}

fn scan_frames(
    bytes: &[u8],
    header_hash: [u8; 32],
    frames: &mut Vec<ScannedFrame>,
    segment_no: u32,
) -> Result<(Tail, ScannedTail), String> {
    let mut last_frame_hash = header_hash;
    let mut leaves = Vec::new();
    let mut frame_count = 0u64;
    let mut off = HEADER_WIRE_SIZE;
    loop {
        let rem = bytes.len().saturating_sub(off);
        if rem == 0 {
            return Ok((
                Tail::Clean,
                ScannedTail {
                    frame_count,
                    leaves,
                },
            ));
        }
        if rem == TRAILER_WIRE_SIZE {
            let trailer = borsh::from_slice::<SegmentTrailer>(&bytes[off..])
                .map_err(|_| format!("segment {segment_no}: trailer-sized tail failed to parse"))?;
            return Ok((
                Tail::Trailer { trailer },
                ScannedTail {
                    frame_count,
                    leaves,
                },
            ));
        }
        if rem < FRAME_FIXED_WIRE_SIZE {
            return Ok((
                Tail::Torn {
                    offset: off,
                    reason: "short frame header".into(),
                },
                ScannedTail {
                    frame_count,
                    leaves,
                },
            ));
        }
        let (frame, used) = match parse_frame(&bytes[off..]) {
            Ok(parsed) => parsed,
            Err(e) => {
                return Ok((
                    Tail::Torn {
                        offset: off,
                        reason: format!("frame parse: {e}"),
                    },
                    ScannedTail {
                        frame_count,
                        leaves,
                    },
                ));
            }
        };
        if frame.payload_blake3 != payload_blake3(frame.kind, &frame.payload) {
            return Ok((
                Tail::Torn {
                    offset: off,
                    reason: "payload blake3 mismatch".into(),
                },
                ScannedTail {
                    frame_count,
                    leaves,
                },
            ));
        }
        if frame.prev_frame_hash != last_frame_hash {
            return Ok((
                Tail::Torn {
                    offset: off,
                    reason: "frame chain link mismatch".into(),
                },
                ScannedTail {
                    frame_count,
                    leaves,
                },
            ));
        }
        last_frame_hash = frame_hash(&frame);
        leaves.push(frame.payload_blake3);
        frames.push(ScannedFrame {
            kind: frame.kind,
            payload: frame.payload,
        });
        frame_count += 1;
        off += used;
    }
}

fn validate_trailer(
    trailer: &SegmentTrailer,
    scanned: &ScannedTail,
    header: &SegmentHeader,
    verifying: &ed25519_dalek::VerifyingKey,
) -> Result<(), String> {
    if trailer.frame_count != scanned.frame_count {
        return Err(format!(
            "trailer frame_count {} != scanned {}",
            trailer.frame_count, scanned.frame_count
        ));
    }
    let root = merkle_root(&scanned.leaves);
    if trailer.merkle_root != root {
        return Err("trailer merkle root mismatch".into());
    }
    let msg = signature_message(
        &header.segment_id,
        &header.prev_segment_hash,
        &trailer.merkle_root,
    );
    let sig = Signature::from_bytes(&trailer.signature);
    verifying
        .verify(&msg, &sig)
        .map_err(|_| "trailer signature invalid".to_string())
}
