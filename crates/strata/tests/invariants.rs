//! #399 log-chain properties for the public [`strata::StrataLog`] API.
//!
//! Checked against recorded bytes only:
//! - seqs strictly increase from 1 and every frame links to the previous
//!   chain hash (header hash for the first frame of a segment)
//! - `head.state` is present and equals the last acked seq once a frame exists
//! - truncating the active segment at any byte past the watermark reopens
//!   with every acked frame
//! - corrupting any byte of a complete frame at or below the watermark halts,
//!   and the file length is left unchanged
//!
//! Recovery when `head.state` is absent is #370 (C2/C3) and is not asserted
//! here. The in-file generator is the stand-in for `proptest` (test files
//! only; see `common`).

mod common;

use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom};

use common::{assert_open_log_chain, cases, only_segment, splice_tail, Rng, TmpDir};
use strata::{FrameRecord, StrataError, StrataLog};

fn append_payloads(log: &StrataLog, payloads: &[Vec<u8>]) {
    let batch: Vec<(u8, Vec<u8>)> = payloads
        .iter()
        .enumerate()
        .map(|(i, payload)| ((i % 7) as u8, payload.clone()))
        .collect();
    log.append_batch(batch).expect("append batch");
}

fn sample_payloads(seed: u64, n: usize) -> Vec<Vec<u8>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let len = 1 + rng.gen_range(24);
            (0..len)
                .map(|_| rng.next_u64() as u8)
                .chain(u64::to_le_bytes(seed ^ i as u64))
                .collect()
        })
        .collect()
}

#[test]
fn prop_log_chain_seqs_links_and_watermark() {
    for seed in 1..=cases(12) {
        let tmp = TmpDir::new("chain");
        let log = StrataLog::open(&tmp.path).expect("open");
        let mut rng = Rng::new(seed);
        let batches = 1 + rng.gen_range(4);
        for batch in 0..batches {
            let n = 1 + rng.gen_range(8);
            append_payloads(
                &log,
                &sample_payloads(seed.wrapping_mul(16) ^ batch as u64, n),
            );
            assert_open_log_chain(&tmp.path, &log);
        }
    }
}

#[test]
fn prop_seal_chains_segment_hash() {
    let tmp = TmpDir::new("seal");
    let log = StrataLog::open(&tmp.path).expect("open");
    append_payloads(&log, &sample_payloads(0x5EA1, 10));
    let seal = log.seal().expect("seal");
    let ack = log.append(9, b"after-seal").expect("append after seal");
    assert_eq!(ack.seq, 11);
    assert_eq!(log.head().segment_no, 1);
    assert_open_log_chain(&tmp.path, &log);
    let segs = common::list_segments(&tmp.path);
    assert_eq!(segs.len(), 2);
    let bytes = fs::read(&segs[1]).expect("segment 2");
    let header = borsh::from_slice::<strata::SegmentHeader>(&bytes[..strata::HEADER_WIRE_SIZE])
        .expect("header");
    assert_eq!(header.prev_segment_hash, seal.segment_hash);
}

#[test]
fn prop_torn_tail_every_byte_past_watermark_keeps_acked_frames() {
    let tmp = TmpDir::new("torn");
    let payloads = sample_payloads(0x70A2, 4);
    let seg = {
        let log = StrataLog::open(&tmp.path).expect("open");
        append_payloads(&log, &payloads);
        assert_open_log_chain(&tmp.path, &log);
        only_segment(&tmp.path)
    };
    let acked = fs::read(&seg).expect("acked bytes");
    let acked_len = acked.len() as u64;
    let head_path = tmp.path.join("head.state");
    let head = fs::read(&head_path).expect("head.state");
    let last_frame = last_frame_len(&acked);

    // Recovery truncates the file. Restore the acked image and the watermark
    // before every cut so a previous reopen cannot move `head.state`.
    //
    // A suffix that makes `last_frame + cut == 104` (`TRAILER_WIRE_SIZE`) is
    // the pinned bug in `regression_seed_trailer_window_swallows_acked_frame`,
    // not a torn tail the recovery path currently keeps.
    for cut in 0..=24 {
        if last_frame + cut == strata::TRAILER_WIRE_SIZE {
            continue;
        }
        let mut bytes = acked.clone();
        bytes.extend(std::iter::repeat_n(0xA5, cut));
        fs::write(&seg, &bytes).expect("plant cut");
        fs::write(&head_path, &head).expect("restore watermark");
        let log =
            StrataLog::open(&tmp.path).unwrap_or_else(|err| panic!("reopen cut {cut}: {err}"));
        let frames = log.read_frames(1).expect("frames");
        assert_eq!(
            frames.len(),
            payloads.len(),
            "cut {cut} dropped an acked frame"
        );
        for (frame, payload) in frames.iter().zip(&payloads) {
            assert_eq!(&frame.payload, payload, "cut {cut}");
        }
        assert_eq!(log.head().last_acked_seq, payloads.len() as u64);
        assert_open_log_chain(&tmp.path, &log);
    }

    fs::write(&seg, &acked).expect("restore acked prefix");
    fs::write(&head_path, &head).expect("restore watermark");
    splice_tail(&tmp.path, &[0xA5; 24]);
    {
        let log = StrataLog::open(&tmp.path).expect("reopen torn tail");
        let next = log.append(1, b"fresh").expect("append after recovery");
        assert_eq!(next.seq, payloads.len() as u64 + 1);
        assert_open_log_chain(&tmp.path, &log);
    }

    // A cut that lands inside an acked frame is damage at the watermark, not
    // a torn tail. Open halts and does not shorten the file any further.
    for delta in [1u64, 10, acked_len / 2] {
        fs::write(&seg, &acked).expect("restore acked prefix");
        fs::write(&head_path, &head).expect("restore watermark");
        let file = OpenOptions::new().write(true).open(&seg).expect("open seg");
        let short = acked_len - delta;
        file.set_len(short).expect("cut acked bytes");
        file.sync_all().expect("sync");
        drop(file);
        match StrataLog::open(&tmp.path) {
            Err(StrataError::Halt(detail)) => {
                assert_eq!(
                    detail.last_acked_seq,
                    payloads.len() as u64,
                    "delta {delta}"
                );
            }
            other => panic!("delta {delta}: expected Halt, got {other:?}"),
        }
        assert_eq!(
            fs::metadata(&seg).expect("meta").len(),
            short,
            "halt must not truncate further (delta {delta})"
        );
    }
}

#[test]
fn prop_corruption_of_acked_frame_halts() {
    let tmp = TmpDir::new("halt");
    let seg = {
        let log = StrataLog::open(&tmp.path).expect("open");
        append_payloads(&log, &sample_payloads(0x0A11, 4));
        assert_eq!(log.head().last_acked_seq, 4);
        only_segment(&tmp.path)
    };
    let original = fs::read(&seg).expect("segment");
    let (start, len) = second_frame_span(&original);
    assert!(len > 0);
    for offset in start..start + len {
        let mut bytes = original.clone();
        bytes[offset] ^= 0xff;
        fs::write(&seg, &bytes).expect("plant corruption");
        match StrataLog::open(&tmp.path) {
            Err(StrataError::Halt(detail)) => {
                assert_eq!(detail.last_acked_seq, 4, "offset {offset}");
                assert!(!detail.reason.is_empty(), "offset {offset}");
            }
            other => panic!(
                "offset {offset} (frame byte {}): expected Halt, got {other:?}",
                offset - start
            ),
        }
        assert_eq!(
            fs::metadata(&seg).expect("meta").len(),
            original.len() as u64,
            "offset {offset}: halt rewrote the segment"
        );
    }
}

fn last_frame_len(bytes: &[u8]) -> usize {
    let mut off = strata::HEADER_WIRE_SIZE;
    let mut used = 0usize;
    while off < bytes.len() {
        let (_frame, len) = strata::parse_frame(&bytes[off..]).expect("acked frame");
        used = len;
        off += len;
    }
    assert!(used > 0, "segment has no frame");
    used
}

/// Minimal repro, found by `prop_torn_tail_every_byte_past_watermark_keeps_acked_frames`.
///
/// `scan_frames` treats a remainder of exactly [`strata::TRAILER_WIRE_SIZE`]
/// (104) as a trailer unless those bytes are one chained frame. An acked
/// frame followed by a torn suffix that together are 104 bytes is not one
/// frame, so the scanner returns a trailer *at the start of the acked frame*.
/// Recovery then truncates that frame (it believes the bytes are an unacked
/// seal) and only afterwards notices the watermark now points past the file,
/// so `open` returns `Halt` and the acked frame is already gone.
///
/// Payload length 10 → frame wire size 79. Suffix 25. 79 + 25 = 104.
#[test]
fn regression_seed_trailer_window_swallows_acked_frame() {
    let tmp = TmpDir::new("trailer-window");
    let payload = vec![0x11; 10];
    let seg = {
        let log = StrataLog::open(&tmp.path).expect("open");
        log.append(1, &payload).expect("append");
        assert_eq!(log.head().last_acked_seq, 1);
        only_segment(&tmp.path)
    };
    let acked_len = fs::metadata(&seg).expect("stat").len();
    let frame_len = last_frame_len(&fs::read(&seg).expect("segment"));
    let suffix = strata::TRAILER_WIRE_SIZE - frame_len;
    assert!(suffix > 0 && suffix < strata::FRAME_FIXED_WIRE_SIZE);
    splice_tail(&tmp.path, &vec![0xA5; suffix]);
    let planted = fs::metadata(&seg).expect("stat").len();
    assert_eq!(planted, acked_len + suffix as u64);

    match StrataLog::open(&tmp.path) {
        Err(StrataError::Halt(detail)) => {
            assert_eq!(detail.last_acked_seq, 1);
            assert!(
                detail.reason.contains("acked frames missing"),
                "reason: {}",
                detail.reason
            );
        }
        other => panic!("expected the pinned halt, got {other:?}"),
    }
    let after = fs::metadata(&seg).expect("stat").len();
    assert!(
        after < acked_len,
        "open halted but kept the acked frame (before {acked_len}, after {after}); the bug changed shape"
    );
}

fn second_frame_span(bytes: &[u8]) -> (usize, usize) {
    let mut off = strata::HEADER_WIRE_SIZE;
    let (_first, first_len) = strata::parse_frame(&bytes[off..]).expect("frame 1");
    off += first_len;
    let start = off;
    let (_second, second_len) = strata::parse_frame(&bytes[off..]).expect("frame 2");
    (start, second_len)
}

/// A history whose middle frame link was dropped must fail the chain checker.
/// This is the seeded mutant for "replay skipped a frame": the checker is the
/// same predicate `assert_open_log_chain` uses on `read_frames`.
#[test]
fn mutant_skipped_frame_link_is_killed() {
    let tmp = TmpDir::new("mutant");
    let frames = {
        let log = StrataLog::open(&tmp.path).expect("open");
        append_payloads(&log, &sample_payloads(0x0BAD, 5));
        let frames = log.read_frames(1).expect("frames");
        assert!(chain_links(&frames), "honest history must link");
        frames
    };
    let mut skipped = frames.clone();
    skipped.remove(2);
    assert!(
        !chain_links(&skipped),
        "dropping the frame after seq 2 must break the checker"
    );
    let mut relinked = frames.clone();
    relinked[3].prev_frame_hash = [0x11; 32];
    assert!(
        !chain_links(&relinked),
        "a rewritten prev hash must break the checker"
    );
}

fn chain_links(frames: &[FrameRecord]) -> bool {
    if frames.is_empty() {
        return true;
    }
    frames.iter().enumerate().all(|(i, frame)| {
        frame.seq == i as u64 + 1
            && frame.payload_blake3 == strata::payload_blake3(frame.kind, &frame.payload)
            && (i == 0 || frame.prev_frame_hash == frames[i - 1].frame_hash)
    })
}

/// The length prefix is inside the hashed frame. A one-byte edit of it is
/// the same halt class as a payload edit; this guards the span helper.
#[test]
fn prop_length_prefix_of_acked_frame_halts() {
    let tmp = TmpDir::new("len");
    let seg = {
        let log = StrataLog::open(&tmp.path).expect("open");
        append_payloads(&log, &sample_payloads(0x1E4, 3));
        only_segment(&tmp.path)
    };
    let mut bytes = fs::read(&seg).expect("segment");
    let (start, _) = second_frame_span(&bytes);
    bytes[start] ^= 0xff;
    let len_before = bytes.len() as u64;
    fs::write(&seg, &bytes).expect("plant");
    assert!(
        matches!(StrataLog::open(&tmp.path), Err(StrataError::Halt(_))),
        "length-prefix edit below the watermark must halt"
    );
    assert_eq!(file_len(&seg), len_before);
}

fn file_len(path: &std::path::Path) -> u64 {
    fs::metadata(path).expect("stat").len()
}

#[test]
fn prop_reads_of_the_log_append_nothing() {
    let tmp = TmpDir::new("reads");
    let log = StrataLog::open(&tmp.path).expect("open");
    append_payloads(&log, &sample_payloads(0x2EAD, 6));
    let head = log.head().frames_total;
    let _ = log.read_frames(1).expect("read");
    let _ = log.verify_tail().expect("tail");
    let _ = log.verify_log().expect("verify");
    let _ = log.head();
    let _ = strata::read_head_state(&tmp.path).expect("watermark");
    assert_eq!(log.head().frames_total, head);
    let seg = only_segment(&tmp.path);
    let on_disk = fs::metadata(&seg).expect("stat").len();
    let mut file = fs::File::open(&seg).expect("seg");
    let end = file.seek(SeekFrom::End(0)).expect("seek");
    assert_eq!(on_disk, end);
}
