//! In-crate tests. Each test uses its own temp directory and resets the
//! failpoint hooks; segment syncs are counted through `SYNC_COUNT`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use crate::format::{self, SegmentHeader, HEADER_WIRE_SIZE};
use crate::sync::{
    reset_failpoints, set_frame_write_budget, set_site_full, Site, FAIL_ON_SYNC_N, SYNC_COUNT,
};
use crate::{SeqAck, StrataError, StrataLog};

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The failpoint statics (`SYNC_COUNT`, `FAIL_ON_SYNC_N`) are process-wide, so
/// every test takes this lock for its whole body: `cargo test` stays
/// deterministic even though libtest runs tests on parallel threads.
static SERIAL_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    match SERIAL_TESTS.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "strata-test-{}-{tag}-{}",
        std::process::id(),
        DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn only_segment(dir: &Path) -> PathBuf {
    let segs = crate::log::list_segments(dir).unwrap();
    assert_eq!(segs.len(), 1, "expected exactly one segment");
    segs[0].1.clone()
}

fn append_many(log: &StrataLog, n: usize) -> Vec<SeqAck> {
    let batch: Vec<(u8, Vec<u8>)> = (0..n)
        .map(|i| ((i % 7) as u8, format!("payload-{i}").into_bytes()))
        .collect();
    log.append_batch(batch).unwrap()
}

#[test]
fn round_trip_and_reopen() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("roundtrip");
    {
        let log = StrataLog::open(&dir).unwrap();
        let a1 = log.append(1, b"hello").unwrap();
        let a2 = log.append(2, b"").unwrap(); // empty payload is legal
        let a3 = log.append(3, b"world").unwrap();
        assert_eq!((a1.seq, a2.seq, a3.seq), (1, 2, 3));
        assert_ne!(a1.frame_hash, a2.frame_hash);
        let head = log.head();
        assert_eq!(head.last_acked_seq, 3);
        assert_eq!(head.next_seq, 4);
        assert_eq!(head.frames_total, 3);
        assert_eq!(head.segment_no, 0);
        let frames = log.read_frames(1).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].payload, b"hello");
        assert!(frames[1].payload.is_empty());
        assert_eq!(frames[2].kind, 3);
        assert_eq!(log.verify_tail().unwrap().frames_verified, 3);
        assert_eq!(log.read_frames(3).unwrap().len(), 1);
    } // DirLock released on drop
    {
        let log = StrataLog::open(&dir).unwrap();
        let frames = log.read_frames(1).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].payload, b"hello");
        let a4 = log.append(9, b"after-reopen").unwrap();
        assert_eq!(a4.seq, 4);
        assert_eq!(log.head().last_acked_seq, 4);
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn chain_integrity_1000_frames_and_seal() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("chain1000");
    let log = StrataLog::open(&dir).unwrap();
    for _ in 0..10 {
        append_many(&log, 100);
    }
    let frames = log.read_frames(1).unwrap();
    assert_eq!(frames.len(), 1000);
    for (i, rec) in frames.iter().enumerate() {
        assert_eq!(rec.seq, i as u64 + 1);
        assert_eq!(
            rec.payload_blake3,
            format::payload_blake3(rec.kind, &rec.payload)
        );
        if i > 0 {
            assert_eq!(
                rec.prev_frame_hash,
                frames[i - 1].frame_hash,
                "chain break at index {i}"
            );
        }
    }
    // First frame chains off the segment header hash.
    let seg = only_segment(&dir);
    let header_bytes = fs::read(&seg).unwrap();
    let header_hash = format::hash_slice(&header_bytes[..HEADER_WIRE_SIZE]);
    assert_eq!(frames[0].prev_frame_hash, header_hash);
    assert_eq!(log.read_frames(999).unwrap().len(), 2);

    // Seal: signed trailer lands, next segment chains to the sealed hash.
    let seal = log.seal().unwrap();
    assert_eq!(seal.sealed_segment_no, 0);
    assert_eq!(seal.frame_count, 1000);
    let ack = log.append(42, b"seg2").unwrap();
    assert_eq!(ack.seq, 1001);
    let segs = crate::log::list_segments(&dir).unwrap();
    assert_eq!(segs.len(), 2);
    let b2 = fs::read(&segs[1].1).unwrap();
    let h2: SegmentHeader = borsh::from_slice(&b2[..HEADER_WIRE_SIZE]).unwrap();
    assert_eq!(h2.prev_segment_hash, seal.segment_hash);
    drop(log);

    // Reopen: sealed trailer + signature verified, chain continues.
    let log = StrataLog::open(&dir).unwrap();
    let frames = log.read_frames(1).unwrap();
    assert_eq!(frames.len(), 1001);
    assert_eq!(frames[1000].payload, b"seg2");
    let tail = log.verify_tail().unwrap();
    assert_eq!(tail.segment_no, 1);
    assert_eq!(tail.frames_verified, 1);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_tail_recovery_truncates_unacked_frames() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("tornA");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 4);
        let head = log.head();
        assert_eq!(head.last_acked_seq, 4);
        // Hand-craft a torn frame 5: valid encoding, cut mid-way, never acked.
        let frame = format::Frame {
            kind: 9,
            payload: b"never-acked".to_vec(),
            payload_blake3: format::payload_blake3(9, b"never-acked"),
            prev_frame_hash: head.last_frame_hash,
        };
        let wire = borsh::to_vec(&frame).unwrap();
        let seg = only_segment(&dir);
        let mut f = fs::OpenOptions::new().append(true).open(&seg).unwrap();
        f.write_all(&wire[..25]).unwrap(); // torn: mid-frame
        f.sync_all().unwrap();
    }
    {
        let log = StrataLog::open(&dir).unwrap();
        let frames = log.read_frames(1).unwrap();
        assert_eq!(frames.len(), 4, "acked prefix must survive");
        assert_eq!(log.head().last_acked_seq, 4);
        assert_eq!(log.head().next_seq, 5);
        let a = log.append(5, b"fresh").unwrap();
        assert_eq!(a.seq, 5);
        assert_eq!(log.read_frames(1).unwrap().len(), 5);
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_tail_short_read_truncates() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("tornB");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
        drop(log);
        // Frame 3's write "tore": chop bytes off the file, and the durable
        // watermark only covers frames 1..=2.
        let seg = only_segment(&dir);
        let len = fs::metadata(&seg).unwrap().len();
        let f = fs::OpenOptions::new().write(true).open(&seg).unwrap();
        f.set_len(len - 10).unwrap();
        drop(f);
        crate::log::write_head_state(&dir, 2).unwrap();
    }
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 2);
    assert_eq!(log.head().last_acked_seq, 2);
    assert_eq!(log.append(7, b"next").unwrap().seq, 3);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn corruption_below_watermark_halts() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("halt");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
        drop(log);
    }
    // Flip a payload byte of frame 2 (acked; watermark is 3).
    let seg = only_segment(&dir);
    let mut bytes = fs::read(&seg).unwrap();
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    let f2_start = HEADER_WIRE_SIZE + n1;
    bytes[f2_start + 5] ^= 0xff; // inside frame 2's payload ("payload-1")
    fs::write(&seg, &bytes).unwrap();

    let err = StrataLog::open(&dir).unwrap_err();
    match &err {
        StrataError::Halt(d) => {
            assert_eq!(d.last_acked_seq, 3);
            assert!(d.reason.contains("blake3"), "reason: {}", d.reason);
        }
        other => panic!("expected Halt, got {other:?}"),
    }
    // History is never truncated: the damaged bytes are still on disk.
    let after = fs::read(&seg).unwrap();
    assert_eq!(after.len(), bytes.len());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn damaged_segment_without_head_state_refuses_open() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("nohead-flip");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 4);
        drop(log);
    }
    // Watermark gone: a mid-segment flip must still halt, not truncate.
    fs::remove_file(dir.join("head.state")).unwrap();
    let seg = only_segment(&dir);
    let mut bytes = fs::read(&seg).unwrap();
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    let f2_start = HEADER_WIRE_SIZE + n1;
    bytes[f2_start + 5] ^= 0xff; // payload byte of frame 2, mid-segment
    fs::write(&seg, &bytes).unwrap();

    let err = StrataLog::open(&dir).unwrap_err();
    match &err {
        StrataError::Halt(d) => {
            assert!(
                d.reason.contains("blake3") || d.reason.contains("damaged"),
                "reason: {}",
                d.reason
            );
        }
        other => panic!("expected open to fail, got {other:?}"),
    }
    assert_eq!(
        fs::read(&seg).unwrap(),
        bytes,
        "open must not truncate a damaged segment"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_key_on_existing_log_refuses_open() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("nokey");
    {
        let log = StrataLog::open(&dir).unwrap();
        log.append(1, b"durable-frame").unwrap();
        drop(log);
    }
    let key = dir.join("strata.key");
    assert!(key.is_file(), "a new log writes its key");
    fs::remove_file(&key).unwrap();
    let err = StrataLog::open(&dir).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("signing key missing") && msg.contains("refusing to mint"),
        "{msg}"
    );
    assert!(
        !key.exists(),
        "reopen must not mint a replacement key into the log directory"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn group_commit_batch_shares_one_sync() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("group");
    let log = StrataLog::open(&dir).unwrap();
    reset_failpoints(); // count only post-open segment syncs

    let acks = append_many(&log, 10);
    let seqs: Vec<u64> = acks.iter().map(|a| a.seq).collect();
    assert_eq!(seqs, (1..=10).collect::<Vec<_>>());
    assert_eq!(
        SYNC_COUNT.load(Ordering::SeqCst),
        1,
        "one batch must share exactly one segment sync"
    );
    assert_eq!(log.head().last_acked_seq, 10);

    // Batches larger than MAX_BATCH_FRAMES chunk: 130 frames -> 64+64+2.
    let acks2 = append_many(&log, 130);
    assert_eq!(acks2.len(), 130);
    assert_eq!(acks2.last().unwrap().seq, 140);
    assert_eq!(SYNC_COUNT.load(Ordering::SeqCst), 1 + 3);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn group_commit_concurrent_appenders_coalesce() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("gthreads");
    let log = Arc::new(StrataLog::open(&dir).unwrap());
    reset_failpoints(); // count only post-open segment syncs

    const THREADS: usize = 8;
    const PER_THREAD: usize = 4;
    let total: u64 = (THREADS * PER_THREAD) as u64;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let log = Arc::clone(&log);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            (0..PER_THREAD)
                .map(|i| {
                    log.append(t as u8, format!("t{t}-i{i}").as_bytes())
                        .unwrap()
                })
                .collect::<Vec<_>>()
        }));
    }
    let mut all: Vec<SeqAck> = Vec::new();
    for h in handles {
        all.extend(h.join().unwrap());
    }
    assert_eq!(all.len() as u64, total);
    let mut seqs: Vec<u64> = all.iter().map(|a| a.seq).collect();
    seqs.sort_unstable();
    assert_eq!(
        seqs,
        (1..=total).collect::<Vec<_>>(),
        "seqs are unique and dense"
    );
    assert_eq!(log.head().last_acked_seq, total);

    let syncs = SYNC_COUNT.load(Ordering::SeqCst);
    assert!(
        syncs < total as usize,
        "group commit must coalesce concurrent appends (syncs={syncs})"
    );
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len() as u64, total);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn failpoint_third_sync_panics_fail_stop() {
    let _serial = serialize();
    let dir = test_dir("failpoint");
    let log = StrataLog::open(&dir).unwrap();
    reset_failpoints();

    log.append(1, b"one").unwrap(); // segment sync 1
    log.append(2, b"two").unwrap(); // segment sync 2
    FAIL_ON_SYNC_N.store(3, Ordering::SeqCst);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        log.append(3, b"three") // segment sync 3 -> injected failure
    }));
    let payload = result.expect_err("the 3rd segment sync must panic");
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("panic payload is a String");
    assert!(msg.contains("strata fail-stop"), "{msg}");
    assert!(msg.contains("op: sync"), "{msg}");
    assert!(msg.contains("last_acked_seq: 2"), "{msg}");

    // Watermark on disk still covers only the first two frames: the failed
    // commit never reached DURABLE. Frame 3's bytes were fully written
    // before the sync failure, so the recovery scan legitimately retains
    // the valid frame — its ACK was lost, not its bytes — and the sequence
    // continues past it.
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(
        log.head().last_acked_seq,
        2,
        "failed commit must not advance the watermark"
    );
    assert_eq!(log.read_frames(1).unwrap().len(), 3);
    assert_eq!(log.append(4, b"four").unwrap().seq, 4);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn single_writer_lock_and_stale_takeover() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("lock");
    let log = StrataLog::open(&dir).unwrap();
    match StrataLog::open(&dir) {
        // Windows cannot read a locked file's pid, so it may be unknown (0).
        Err(StrataError::Locked { pid }) => assert!(pid > 0 || cfg!(windows)),
        other => panic!("expected Locked, got {other:?}"),
    }
    drop(log); // releases strata.lock
    let log2 = StrataLog::open(&dir).unwrap();
    log2.append(1, b"x").unwrap();
    drop(log2);

    // A stale lock (pid that cannot exist) is detected and taken over.
    fs::write(
        dir.join(crate::lockfile::LOCK_NAME),
        4_000_000u64.to_le_bytes(),
    )
    .unwrap();
    let log3 = StrataLog::open(&dir).unwrap();
    log3.append(2, b"y").unwrap();
    assert_eq!(log3.read_frames(1).unwrap().len(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn lock_file_without_a_holder_never_blocks_open() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("lock-no-holder");
    fs::create_dir_all(&dir).unwrap();
    let lock_path = dir.join(crate::lockfile::LOCK_NAME);

    // A process killed between creating the lock file and writing its owner
    // bytes leaves an empty file behind.
    fs::write(&lock_path, []).unwrap();
    let log = StrataLog::open(&dir).expect("an empty lock file has no holder");
    log.append(1, b"a").unwrap();
    drop(log);

    // A lock file naming a pid that now belongs to an unrelated live process.
    fs::write(&lock_path, u64::from(std::process::id()).to_le_bytes()).unwrap();
    let log = StrataLog::open(&dir).expect("a live pid in a leftover file is not a holder");
    log.append(2, b"b").unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 2);
    drop(log);

    // Bytes that are not a pid at all.
    fs::write(&lock_path, b"garbage").unwrap();
    StrataLog::open(&dir).expect("an unparseable lock file is not a holder");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn lock_file_stays_in_place_while_a_writer_holds_the_log() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("lock-stays");
    let lock_path = dir.join(crate::lockfile::LOCK_NAME);
    let log = StrataLog::open(&dir).unwrap();
    assert!(lock_path.exists(), "the lock file must outlive open");
    log.append(1, b"x").unwrap();
    match StrataLog::open(&dir) {
        Err(StrataError::Locked { pid }) => assert!(
            pid == u64::from(std::process::id()) || (cfg!(windows) && pid == 0),
            "unexpected holder pid {pid}"
        ),
        other => panic!("expected Locked, got {other:?}"),
    }
    // Still refused after the failed attempt: a refused opener must not
    // disturb the lock file.
    assert!(lock_path.exists());
    assert!(matches!(
        StrataLog::open(&dir),
        Err(StrataError::Locked { .. })
    ));
    // Release keeps the file: removing it would let an opener that already
    // holds the old file and one that creates a new file both succeed.
    drop(log);
    assert!(lock_path.exists(), "release must not unlink the lock file");
    StrataLog::open(&dir).unwrap();
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn merkle_tree_shapes() {
    let _serial = serialize();
    // Independent reference recomputation of the RFC6962-shaped tree.
    fn leaf(p: &[u8; 32]) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&[0x00]);
        h.update(p);
        h.finalize().into()
    }
    fn node(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&[0x01]);
        h.update(l);
        h.update(r);
        h.finalize().into()
    }
    fn reference(hs: &[[u8; 32]]) -> [u8; 32] {
        match hs.len() {
            0 => blake3::hash(&[]).into(),
            1 => hs[0],
            n => {
                let mut k = 1usize;
                while k * 2 < n {
                    k *= 2;
                }
                node(&reference(&hs[..k]), &reference(&hs[k..]))
            }
        }
    }

    let l: Vec<[u8; 32]> = (1u8..=5).map(|i| [i; 32]).collect();
    // `format::merkle_root` takes raw payload hashes and leaf-hashes inside,
    // so the reference folds the leaf-hashed values.
    let leaves: Vec<[u8; 32]> = l.iter().map(leaf).collect();
    let empty_root: [u8; 32] = blake3::hash(&[]).into();
    assert_eq!(format::merkle_root(&[]), empty_root);
    assert_eq!(format::merkle_root(&l[..1]), leaf(&l[0]));
    assert_eq!(
        format::merkle_root(&l[..2]),
        node(&leaf(&l[0]), &leaf(&l[1]))
    );
    assert_eq!(format::merkle_root(&l[..3]), reference(&leaves[..3]));
    assert_eq!(format::merkle_root(&l), reference(&leaves));
    assert_ne!(format::merkle_root(&l[..2]), format::merkle_root(&l[..3]));
}

/// The log signing key must come from the OS CSPRNG on every platform. A
/// derivable seed (clock plus pid) would let anyone re-derive `strata.key`
/// and forge segments, so two fresh logs must get distinct, non-trivial keys.
#[test]
fn fresh_logs_get_distinct_os_entropy_signing_keys() {
    let _serial = serialize();
    reset_failpoints();
    let a = test_dir("entropy-a");
    let b = test_dir("entropy-b");
    drop(StrataLog::open(&a).unwrap());
    drop(StrataLog::open(&b).unwrap());
    let key_a = fs::read(a.join("strata.key")).unwrap();
    let key_b = fs::read(b.join("strata.key")).unwrap();
    assert_eq!(key_a.len(), 32);
    assert_eq!(key_b.len(), 32);
    assert_ne!(key_a, key_b, "two fresh logs shared a signing key");
    assert!(key_a.iter().any(|&x| x != 0), "signing key is all zero");
    assert!(key_b.iter().any(|&x| x != 0), "signing key is all zero");
}

/// Power fails inside a group commit before its sync: a later frame's page
/// landed and an earlier one did not, so a complete frame above the recorded
/// watermark fails its hash. It was never acked; recovery drops it.
#[test]
fn corrupt_complete_frame_above_the_watermark_is_truncated() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("above-wm");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
        let head = log.head();
        let bad = format::Frame {
            kind: 9,
            payload: b"never-acked".to_vec(),
            payload_blake3: format::payload_blake3(9, b"a different payload"),
            prev_frame_hash: head.last_frame_hash,
        };
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(only_segment(&dir))
            .unwrap();
        f.write_all(&borsh::to_vec(&bad).unwrap()).unwrap();
        f.sync_all().unwrap();
    }
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(
        log.read_frames(1).unwrap().len(),
        3,
        "acked history survives"
    );
    assert_eq!(log.append(4, b"after").unwrap().seq, 4);
    fs::remove_dir_all(&dir).unwrap();
}

fn segments(dir: &Path) -> Vec<PathBuf> {
    crate::log::list_segments(dir)
        .unwrap()
        .into_iter()
        .map(|(_, path)| path)
        .collect()
}

/// Power fails while `seal` writes the trailer: the file covers every
/// trailer byte but half are zeros, and the next segment was never created.
/// Every frame before it is acked, so the torn trailer is dropped.
#[test]
fn a_seal_torn_by_a_crash_reopens_with_every_frame() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("torn-seal");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
        log.seal().unwrap();
    }
    let segs = segments(&dir);
    assert_eq!(segs.len(), 2);
    fs::remove_file(&segs[1]).unwrap();
    let mut bytes = fs::read(&segs[0]).unwrap();
    let len = bytes.len();
    for byte in &mut bytes[len - 52..] {
        *byte = 0;
    }
    fs::write(&segs[0], &bytes).unwrap();

    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 3);
    assert_eq!(log.append(4, b"after").unwrap().seq, 4);
    fs::remove_dir_all(&dir).unwrap();
}

/// An unacked tear that leaves exactly a trailer's worth of bytes is still a
/// tear, not a seal.
#[test]
fn an_unacked_tear_of_trailer_size_is_truncated() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("tear-104");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(only_segment(&dir))
            .unwrap();
        f.write_all(&[0xA5; format::TRAILER_WIRE_SIZE]).unwrap();
        f.sync_all().unwrap();
    }
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 3);
    assert_eq!(log.append(4, b"after").unwrap().seq, 4);
    fs::remove_dir_all(&dir).unwrap();
}

/// A frame whose payload is 35 bytes is exactly as long as a trailer. It is
/// a frame: acked, and still there after a reopen.
#[test]
fn a_trailer_sized_frame_is_read_as_a_frame() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("frame-104");
    let payload = [7u8; 35];
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 2);
        log.append(3, &payload).unwrap();
        let tail = fs::read(only_segment(&dir)).unwrap();
        let (_, used) =
            format::parse_frame(&tail[tail.len() - format::TRAILER_WIRE_SIZE..]).unwrap();
        assert_eq!(
            used,
            format::TRAILER_WIRE_SIZE,
            "the frame is trailer-sized"
        );
    }
    let log = StrataLog::open(&dir).unwrap();
    let frames = log.read_frames(1).unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[2].payload, payload);
    assert_eq!(log.append(4, b"after").unwrap().seq, 4);
    fs::remove_dir_all(&dir).unwrap();
}

/// `seal` wrote its trailer, then could not create the next segment. The log
/// fails stop and wakes everyone; an append afterwards returns at once
/// instead of waiting forever.
#[cfg(unix)]
#[test]
fn a_seal_that_cannot_roll_the_segment_does_not_hang_appends() {
    use std::os::unix::fs::PermissionsExt;
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("seal-roll");
    let log = Arc::new(StrataLog::open(&dir).unwrap());
    append_many(&log, 2);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o500)).unwrap();
    let sealed = log.seal();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        sealed.is_err(),
        "creating the next segment should have failed"
    );

    let (tx, rx) = std::sync::mpsc::channel();
    let appender = Arc::clone(&log);
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            appender.append(1, b"after a failed seal")
        }));
        let _ = tx.send(outcome.is_err() || outcome.is_ok_and(|r| r.is_err()));
    });
    let failed_fast = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("append after a failed seal hung");
    assert!(failed_fast, "append after a failed seal must not succeed");
    drop(log);
    let reopened = StrataLog::open(&dir).unwrap();
    assert_eq!(reopened.read_frames(1).unwrap().len(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

// ---------------------------------------------------------------------------
// Recovery with no recorded watermark fails closed
// ---------------------------------------------------------------------------

/// A log of `n` acked frames with `head.state` removed, plus the bytes of its
/// only segment.
fn log_without_watermark(tag: &str, n: usize) -> (PathBuf, PathBuf, Vec<u8>) {
    let dir = test_dir(tag);
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, n);
        drop(log);
    }
    fs::remove_file(dir.join("head.state")).unwrap();
    let seg = only_segment(&dir);
    let bytes = fs::read(&seg).unwrap();
    (dir, seg, bytes)
}

fn assert_halt_and_untouched(dir: &Path, seg: &Path, expected: &[u8]) {
    match StrataLog::open(dir) {
        Err(StrataError::Halt(_)) => {}
        Err(other) => panic!("expected Halt, got {other:?}"),
        Ok(_) => panic!("open must not succeed over a damaged log"),
    }
    assert!(seg.exists(), "open must not delete the damaged segment");
    assert_eq!(
        fs::read(seg).unwrap(),
        expected,
        "open must leave the damaged bytes in place"
    );
}

#[test]
fn corrupt_active_header_without_head_state_halts() {
    let _serial = serialize();
    reset_failpoints();
    let (dir, seg, mut bytes) = log_without_watermark("hdr-nohead", 4);
    bytes[0] ^= 0xff; // magic
    fs::write(&seg, &bytes).unwrap();
    assert_halt_and_untouched(&dir, &seg, &bytes);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_header_creation_still_recovers() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("hdr-torn-create");
    {
        let log = StrataLog::open(&dir).unwrap();
        drop(log);
    }
    // First-use crash: the segment file exists but its header write tore.
    let seg = only_segment(&dir);
    let f = fs::OpenOptions::new().write(true).open(&seg).unwrap();
    f.set_len(10).unwrap();
    drop(f);
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.append(1, b"first").unwrap().seq, 1);
    assert_eq!(log.read_frames(1).unwrap().len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn flipped_length_in_middle_frame_without_head_state_halts() {
    let _serial = serialize();
    reset_failpoints();
    let (dir, seg, mut bytes) = log_without_watermark("len-mid-nohead", 4);
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    let f2_start = HEADER_WIRE_SIZE + n1;
    bytes[f2_start + 3] ^= 0x40; // high byte of frame 2's length prefix
    fs::write(&seg, &bytes).unwrap();
    assert_halt_and_untouched(&dir, &seg, &bytes);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn flipped_length_in_final_frame_without_head_state_halts() {
    let _serial = serialize();
    reset_failpoints();
    let (dir, seg, mut bytes) = log_without_watermark("len-last-nohead", 4);
    let mut off = HEADER_WIRE_SIZE;
    for _ in 0..3 {
        let (_f, n) = format::parse_frame(&bytes[off..]).unwrap();
        off += n;
    }
    bytes[off + 3] ^= 0x40; // high byte of the last frame's length prefix
    fs::write(&seg, &bytes).unwrap();
    assert_halt_and_untouched(&dir, &seg, &bytes);
    fs::remove_dir_all(&dir).unwrap();
}

/// A final write cut anywhere inside its frame, with no watermark on disk,
/// still recovers to the frames before it.
#[test]
fn torn_final_write_without_head_state_still_recovers() {
    let _serial = serialize();
    reset_failpoints();
    let probe = test_dir("torn-nohead-probe");
    let (wire, last_frame_len) = {
        let log = StrataLog::open(&probe).unwrap();
        append_many(&log, 4);
        let last_frame_len = format::FRAME_FIXED_WIRE_SIZE + "payload-3".len();
        let head = log.head();
        let frame = format::Frame {
            kind: 9,
            payload: b"never-acked".to_vec(),
            payload_blake3: format::payload_blake3(9, b"never-acked"),
            prev_frame_hash: head.last_frame_hash,
        };
        (borsh::to_vec(&frame).unwrap(), last_frame_len)
    };
    let _ = fs::remove_dir_all(&probe);
    for cut in 1..wire.len() {
        // A tail that happens to be exactly trailer-sized is read as a
        // possible seal and is refused without a watermark; that path is
        // separate from torn-frame recovery.
        if last_frame_len + cut == format::TRAILER_WIRE_SIZE {
            continue;
        }
        let (dir, seg, _bytes) = log_without_watermark("torn-nohead", 4);
        let mut f = fs::OpenOptions::new().append(true).open(&seg).unwrap();
        f.write_all(&wire[..cut]).unwrap();
        f.sync_all().unwrap();
        drop(f);
        let log = StrataLog::open(&dir)
            .unwrap_or_else(|e| panic!("cut at {cut} of {} must recover: {e:?}", wire.len()));
        assert_eq!(log.read_frames(1).unwrap().len(), 4, "cut at {cut}");
        assert_eq!(log.append(5, b"fresh").unwrap().seq, 5, "cut at {cut}");
        drop(log);
        fs::remove_dir_all(&dir).unwrap();
    }
}

/// Flip one payload byte of the second frame of the only segment.
fn flip_second_frame(dir: &Path) {
    let seg = only_segment(dir);
    let mut bytes = fs::read(&seg).unwrap();
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    bytes[HEADER_WIRE_SIZE + n1 + 5] ^= 0xff;
    fs::write(&seg, &bytes).unwrap();
}

#[test]
fn seal_refuses_to_sign_over_a_damaged_active_segment() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("seal-damaged");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    flip_second_frame(&dir);
    let seg = only_segment(&dir);
    let before = fs::read(&seg).unwrap();

    let err = log.seal().expect_err("seal must not sign damaged bytes");
    assert!(matches!(err, StrataError::Halt(_)), "got {err:?}");
    assert_eq!(
        fs::read(&seg).unwrap(),
        before,
        "a refused seal must leave the segment bytes untouched"
    );
    assert_eq!(
        crate::log::list_segments(&dir).unwrap().len(),
        1,
        "a refused seal must not roll a new segment"
    );
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

// Full disk: a failed write is an error to the caller, not a process abort
// ---------------------------------------------------------------------------

fn is_full(e: &StrataError) -> bool {
    matches!(e, StrataError::Io(io) if io.kind() == std::io::ErrorKind::StorageFull)
}

#[test]
fn full_disk_frame_write_is_an_error_and_the_log_stays_usable() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("full-write");
    let log = StrataLog::open(&dir).unwrap();
    log.append(1, b"one").unwrap();
    log.append(2, b"two").unwrap();

    // The second frame of the batch hits the full disk after the first was
    // already written: the whole batch is refused and nothing of it remains.
    set_frame_write_budget(Some(1));
    let err = log
        .append_batch(vec![(3, b"three".to_vec()), (4, b"four".to_vec())])
        .expect_err("a full disk refuses the write");
    assert!(is_full(&err), "{err:?}");
    assert_eq!(log.head().last_acked_seq, 2);
    assert_eq!(log.head().next_seq, 3, "refused frames free their seqs");

    set_frame_write_budget(None);
    assert_eq!(log.append(5, b"five").unwrap().seq, 3);
    assert_eq!(log.read_frames(1).unwrap().len(), 3);
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    let kinds: Vec<u8> = log.read_frames(1).unwrap().iter().map(|f| f.kind).collect();
    assert_eq!(kinds, vec![1, 2, 5]);
    assert_eq!(log.head().last_acked_seq, 3);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn damage_found_on_read_stops_further_appends() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("read-damaged");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    flip_second_frame(&dir);

    let err = log
        .read_frames(1)
        .expect_err("a read over acked damage must fail, not return a prefix");
    assert!(matches!(err, StrataError::Halt(_)), "got {err:?}");

    let err = log
        .append(1, b"after-damage")
        .expect_err("a log that found acked damage must refuse writes");
    assert!(matches!(err, StrataError::Halt(_)), "got {err:?}");
    let err = log
        .append_batch(vec![(1, b"x".to_vec())])
        .expect_err("batch appends are refused too");
    assert!(matches!(err, StrataError::Halt(_)), "got {err:?}");
    assert_eq!(log.head().last_acked_seq, 3, "no new frame was acked");
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn full_disk_watermark_write_is_an_error_and_the_frames_are_not_kept() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("full-watermark");
    let log = StrataLog::open(&dir).unwrap();
    log.append(1, b"one").unwrap();

    set_site_full(Site::HeadState, true);
    let err = log
        .append(2, b"two")
        .expect_err("watermark cannot be written");
    assert!(is_full(&err), "{err:?}");
    assert_eq!(log.head().last_acked_seq, 1);

    set_site_full(Site::HeadState, false);
    assert_eq!(log.append(3, b"three").unwrap().seq, 2);
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    let kinds: Vec<u8> = log.read_frames(1).unwrap().iter().map(|f| f.kind).collect();
    assert_eq!(kinds, vec![1, 3]);
    assert!(!dir.join("head.state.tmp").exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn damage_found_by_verify_tail_stops_further_appends() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("tail-damaged");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    flip_second_frame(&dir);

    assert!(log.verify_tail().is_err());
    let err = log
        .append(1, b"after-damage")
        .expect_err("a log that found acked damage must refuse writes");
    assert!(matches!(err, StrataError::Halt(_)), "got {err:?}");
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn full_disk_fails_every_queued_appender_and_keeps_seqs_dense() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("full-concurrent");
    let log = StrataLog::open(&dir).unwrap();
    log.append(1, b"seed").unwrap();

    set_frame_write_budget(Some(0));
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8u8)
        .map(|i| {
            let log = log.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                log.append(10 + i, b"x").map(|a| a.seq)
            })
        })
        .collect();
    for h in handles {
        let res = h.join().expect("no appender may panic on a full disk");
        let err = res.expect_err("every appender is refused");
        assert!(is_full(&err), "{err:?}");
    }
    assert_eq!(log.head().next_seq, 2);

    set_frame_write_budget(None);
    let seqs: Vec<u64> = (0..3)
        .map(|i| log.append(20 + i, b"y").unwrap().seq)
        .collect();
    assert_eq!(seqs, vec![2, 3, 4]);
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 4);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_healthy_log_keeps_accepting_writes_after_reads_and_checks() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("healthy-reads");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    log.seal().unwrap();
    append_many(&log, 2);
    assert_eq!(log.read_frames(1).unwrap().len(), 5);
    log.verify_tail().unwrap();
    assert_eq!(log.append(1, b"more").unwrap().seq, 6);
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn full_disk_seal_trailer_write_is_an_error_and_the_segment_stays_open() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("full-seal");
    let log = StrataLog::open(&dir).unwrap();
    log.append(1, b"one").unwrap();

    set_site_full(Site::Trailer, true);
    let err = log.seal().expect_err("trailer cannot be written");
    assert!(is_full(&err), "{err:?}");

    set_site_full(Site::Trailer, false);
    assert_eq!(log.append(2, b"two").unwrap().seq, 2);
    log.seal().unwrap();
    drop(log);
    let log = StrataLog::open(&dir).unwrap();
    assert_eq!(log.read_frames(1).unwrap().len(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn damage_in_a_sealed_segment_fails_reads_and_stops_appends() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("sealed-damaged");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    log.seal().unwrap();
    append_many(&log, 2);
    let report = log.verify_log().unwrap();
    assert_eq!(
        (report.segments, report.sealed_segments, report.frames),
        (2, 1, 5)
    );

    // Flip a payload byte of frame 2 inside the sealed segment.
    let sealed = crate::log::list_segments(&dir).unwrap()[0].1.clone();
    let mut bytes = fs::read(&sealed).unwrap();
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    bytes[HEADER_WIRE_SIZE + n1 + 5] ^= 0xff;
    fs::write(&sealed, &bytes).unwrap();

    assert!(matches!(log.verify_log(), Err(StrataError::Halt(_))));
    assert!(matches!(log.read_frames(1), Err(StrataError::Halt(_))));
    assert!(matches!(log.append(1, b"x"), Err(StrataError::Halt(_))));
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_forged_sealed_trailer_fails_verification() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("sealed-trailer");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    log.seal().unwrap();

    // Zero the trailer signature: frames and merkle root still agree, the
    // signature does not.
    let sealed = crate::log::list_segments(&dir).unwrap()[0].1.clone();
    let mut bytes = fs::read(&sealed).unwrap();
    let n = bytes.len();
    for b in &mut bytes[n - 8..] {
        *b ^= 0xff;
    }
    fs::write(&sealed, &bytes).unwrap();

    assert!(matches!(log.verify_log(), Err(StrataError::Halt(_))));
    assert!(matches!(log.append(1, b"x"), Err(StrataError::Halt(_))));
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn frames_missing_from_disk_fail_verification() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("frames-missing");
    let log = StrataLog::open(&dir).unwrap();
    append_many(&log, 3);
    // Cut the active segment back to frame 2, on a frame boundary.
    let seg = only_segment(&dir);
    let bytes = fs::read(&seg).unwrap();
    let (_f1, n1) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
    let (_f2, n2) = format::parse_frame(&bytes[HEADER_WIRE_SIZE + n1..]).unwrap();
    fs::write(&seg, &bytes[..HEADER_WIRE_SIZE + n1 + n2]).unwrap();

    assert!(matches!(log.verify_log(), Err(StrataError::Halt(_))));
    assert!(matches!(log.append(1, b"x"), Err(StrataError::Halt(_))));
    drop(log);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn full_disk_during_open_leaves_no_residue_that_blocks_the_next_open() {
    let _serial = serialize();
    reset_failpoints();

    // Lock pid cannot be written: the half-made lock file must not remain.
    let dir = test_dir("full-open-lock");
    set_site_full(Site::Lock, true);
    let err = StrataLog::open(&dir).expect_err("no room for the lock");
    assert!(is_full(&err), "{err:?}");
    set_site_full(Site::Lock, false);
    StrataLog::open(&dir).expect("space is back: the log opens");
    fs::remove_dir_all(&dir).unwrap();

    // Signing key cannot be written.
    let dir = test_dir("full-open-key");
    set_site_full(Site::Key, true);
    let err = StrataLog::open(&dir).expect_err("no room for the key");
    assert!(is_full(&err), "{err:?}");
    set_site_full(Site::Key, false);
    StrataLog::open(&dir).expect("space is back: the log opens");
    fs::remove_dir_all(&dir).unwrap();

    // First segment header cannot be written.
    let dir = test_dir("full-open-segment");
    set_site_full(Site::SegmentHeader, true);
    let err = StrataLog::open(&dir).expect_err("no room for the segment");
    assert!(is_full(&err), "{err:?}");
    set_site_full(Site::SegmentHeader, false);
    let log = StrataLog::open(&dir).expect("space is back: the log opens");
    assert_eq!(log.append(1, b"one").unwrap().seq, 1);
    fs::remove_dir_all(&dir).unwrap();
}

/// A copy through exFAT or a network share leaves a `._<name>` sidecar beside
/// each file. One next to a segment ends in `.seg` but is not a segment: the
/// log opens with it there and reads back every frame.
#[test]
fn a_hidden_sidecar_named_like_a_segment_does_not_stop_the_log_opening() {
    let _serial = serialize();
    reset_failpoints();
    let dir = test_dir("hidden-sidecar");
    {
        let log = StrataLog::open(&dir).unwrap();
        append_many(&log, 3);
    }
    let segment = only_segment(&dir);
    let name = segment.file_name().unwrap().to_string_lossy().into_owned();
    fs::write(dir.join(format!("._{name}")), b"\x00\x05\x16\x07sidecar").unwrap();
    fs::write(
        dir.join(format!("._00000000-{}.seg", "ab".repeat(16))),
        b"not a segment",
    )
    .unwrap();
    fs::write(dir.join(".DS_Store"), b"finder").unwrap();

    assert_eq!(
        segments(&dir),
        vec![segment],
        "hidden files are not segments"
    );
    let log = StrataLog::open(&dir).expect("the log opens beside the sidecars");
    assert_eq!(log.read_frames(1).unwrap().len(), 3);
    assert_eq!(log.append(1, b"after").unwrap().seq, 4);
    drop(log);

    // The same name without the leading dot is still refused as a bad segment.
    fs::write(dir.join("stray.seg"), b"not a segment").unwrap();
    assert!(matches!(
        crate::log::list_segments(&dir),
        Err(StrataError::Corrupt(_))
    ));
    fs::remove_dir_all(&dir).unwrap();
}
