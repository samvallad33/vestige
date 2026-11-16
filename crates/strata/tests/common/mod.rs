//! Shared harness for the Strata log property tests.
//!
//! Deterministic xorshift64* — no `proptest` or `rand` crate. The lane owns
//! new test files only, so the generator lives here instead of a dev-dependency.
//! `STRATA_PROP_CASES` raises the pinned case count (the scheduled larger run).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use borsh::BorshDeserialize;
use strata::{
    frame_hash, header_hash, parse_frame, payload_blake3, read_head_state, Frame, SegmentHeader,
    StrataLog, GENESIS_PREV_SEGMENT_HASH, HEADER_WIRE_SIZE, SEGMENT_MAGIC, SEGMENT_VERSION,
    TRAILER_WIRE_SIZE,
};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn gen_range(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() as usize) % n
    }
}

pub struct TmpDir {
    pub path: PathBuf,
}

impl TmpDir {
    pub fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "strata-prop-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn cases(default: u64) -> u64 {
    std::env::var("STRATA_PROP_CASES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

pub fn list_segments(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).expect("read log dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("seg") {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

pub fn only_segment(dir: &Path) -> PathBuf {
    let segs = list_segments(dir);
    assert_eq!(segs.len(), 1, "expected one segment in {}", dir.display());
    segs[0].clone()
}

pub fn splice_tail(dir: &Path, extra: &[u8]) {
    let seg = only_segment(dir);
    let mut file = OpenOptions::new()
        .append(true)
        .open(&seg)
        .expect("open segment");
    file.write_all(extra).expect("splice torn tail");
    file.sync_all().expect("sync torn tail");
}

/// Executable spec of the on-disk chain.
///
/// Seqs from [`StrataLog::read_frames`] are contiguous from 1. Every frame's
/// `payload_blake3` is `blake3(kind || payload)`. Inside a segment the first
/// frame links to the header hash and each later frame links to the previous
/// frame hash. A sealed segment's successor header names `blake3` of the
/// sealed file. When the log has frames, `head.state` equals the last seq.
pub fn assert_open_log_chain(dir: &Path, log: &StrataLog) {
    let frames = log.read_frames(1).expect("read frames");
    let mut prev_seq = 0u64;
    for (i, rec) in frames.iter().enumerate() {
        assert!(
            rec.seq > prev_seq,
            "seq did not strictly increase at index {i}: {} then {}",
            prev_seq,
            rec.seq
        );
        assert_eq!(rec.seq, i as u64 + 1, "gap in seq at index {i}");
        prev_seq = rec.seq;
        assert_eq!(
            rec.payload_blake3,
            payload_blake3(rec.kind, &rec.payload),
            "payload hash mismatch at seq {}",
            rec.seq
        );
    }
    let watermark = read_head_state(dir).expect("read head.state");
    if frames.is_empty() {
        assert!(
            watermark.is_none() || watermark == Some(0),
            "empty log watermark {watermark:?}"
        );
        assert_eq!(log.head().last_acked_seq, 0);
    } else {
        let last = frames.last().expect("nonempty").seq;
        assert_eq!(
            watermark,
            Some(last),
            "head.state missing or behind the log"
        );
        assert_eq!(log.head().last_acked_seq, last);
        assert_eq!(log.head().frames_total, frames.len() as u64);
    }
    assert_bytes_chain(dir, &frames);
}

fn assert_bytes_chain(dir: &Path, api_frames: &[strata::FrameRecord]) {
    let segs = list_segments(dir);
    assert!(!segs.is_empty(), "log has no segments");
    let mut expected_prev = GENESIS_PREV_SEGMENT_HASH;
    let mut seen: Vec<Vec<u8>> = Vec::new();
    for (index, path) in segs.iter().enumerate() {
        let is_last = index + 1 == segs.len();
        let bytes = fs::read(path).expect("read segment");
        assert!(
            bytes.len() >= HEADER_WIRE_SIZE,
            "{} shorter than a header",
            path.display()
        );
        let header = SegmentHeader::try_from_slice(&bytes[..HEADER_WIRE_SIZE])
            .unwrap_or_else(|err| panic!("{} header: {err}", path.display()));
        assert_eq!(header.magic, SEGMENT_MAGIC);
        assert_eq!(header.version, SEGMENT_VERSION);
        assert_eq!(
            header.prev_segment_hash,
            expected_prev,
            "{} segment link",
            path.display()
        );
        let (frames, consumed) = walk_frames(&bytes, &header, is_last, path);
        seen.extend(frames);
        if is_last {
            assert_eq!(
                consumed,
                bytes.len(),
                "{} active segment has an unparsed tail",
                path.display()
            );
        } else {
            assert_eq!(
                bytes.len() - consumed,
                TRAILER_WIRE_SIZE,
                "{} sealed segment trailer",
                path.display()
            );
            expected_prev = *blake3::hash(&bytes).as_bytes();
        }
    }
    assert_eq!(seen.len(), api_frames.len(), "byte walk vs read_frames");
    for (byte_payload, rec) in seen.iter().zip(api_frames) {
        assert_eq!(
            byte_payload, &rec.payload,
            "payload mismatch at seq {}",
            rec.seq
        );
    }
}

fn walk_frames(
    bytes: &[u8],
    header: &SegmentHeader,
    is_last: bool,
    path: &Path,
) -> (Vec<Vec<u8>>, usize) {
    let mut off = HEADER_WIRE_SIZE;
    let mut prev = header_hash(header);
    let mut payloads = Vec::new();
    while off < bytes.len() {
        let remain = bytes.len() - off;
        if !is_last && remain == TRAILER_WIRE_SIZE && !frame_chains(&bytes[off..], prev) {
            break;
        }
        let (frame, used) = parse_frame(&bytes[off..]).unwrap_or_else(|err| {
            panic!(
                "{} offset {off}: frame does not parse ({err}); remain {remain}",
                path.display()
            )
        });
        assert!(
            frame_links(&frame, prev),
            "{} offset {off}: frame hash or chain link",
            path.display()
        );
        prev = frame_hash(&frame);
        payloads.push(frame.payload);
        off += used;
    }
    (payloads, off)
}

fn frame_chains(bytes: &[u8], prev: [u8; 32]) -> bool {
    matches!(
        parse_frame(bytes),
        Ok((frame, used)) if used == bytes.len() && frame_links(&frame, prev)
    )
}

fn frame_links(frame: &Frame, prev: [u8; 32]) -> bool {
    frame.prev_frame_hash == prev
        && frame.payload_blake3 == payload_blake3(frame.kind, &frame.payload)
}
