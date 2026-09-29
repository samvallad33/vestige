//! The log engine: open/recovery, the fail-stop commit pipeline, group
//! commit, sealing, reads, and tail verification.
//!
//! Locking model: `group` (batch queue, watermarks, seqs) and `files` (the
//! active segment handle) are separate mutexes. The commit path NEVER holds
//! `group` while acquiring `files` — the leader drops `group` before doing
//! I/O — so there is no lock-order cycle anywhere.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use borsh::{BorshDeserialize, BorshSerialize};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier};

use crate::error::{HaltDetail, StrataError};
use crate::format::{
    self, Frame, FrameRecord, SegmentHeader, SegmentTrailer, GENESIS_PREV_SEGMENT_HASH,
    HEADER_WIRE_SIZE, SEGMENT_MAGIC, SEGMENT_VERSION, TRAILER_WIRE_SIZE,
};
use crate::lockfile::DirLock;
use crate::sync::{self, SyncPurpose};

/// Maximum frames sharing one sync (group-commit upper bound).
pub const MAX_BATCH_FRAMES: usize = 64;
/// Group-commit coalescing window, in milliseconds.
pub const GROUP_COMMIT_WINDOW_MS: u64 = 2;
const GROUP_COMMIT_WINDOW: Duration = Duration::from_millis(GROUP_COMMIT_WINDOW_MS);

const KEY_FILE: &str = "strata.key";
const HEAD_STATE: &str = "head.state";
const HEAD_STATE_TMP: &str = "head.state.tmp";

/// Acknowledgement for one durably committed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqAck {
    /// Global sequence number of the acked frame (first frame in the log is 1).
    pub seq: u64,
    /// blake3 of the frame's full wire encoding — the chain hash.
    pub frame_hash: [u8; 32],
}

/// Point-in-time snapshot of log state (advisory; never holds both locks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadInfo {
    /// Highest seq whose frame is synced, verified, and watermark-recorded.
    pub last_acked_seq: u64,
    /// Seq that the next appended frame will receive.
    pub next_seq: u64,
    pub segment_no: u32,
    pub segment_id: [u8; 16],
    pub prev_segment_hash: [u8; 32],
    pub frames_in_segment: u64,
    /// Total frames across all segments.
    pub frames_total: u64,
    /// Chain hash of the last frame in the active segment (header hash if none).
    pub last_frame_hash: [u8; 32],
}

/// Result of sealing the active segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealInfo {
    pub sealed_segment_no: u32,
    pub segment_id: [u8; 16],
    pub frame_count: u64,
    pub merkle_root: [u8; 32],
    /// blake3 over the sealed segment file (header..=trailer); equals the new
    /// active segment's `prev_segment_hash`.
    pub segment_hash: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrailerCheck {
    pub frame_count: u64,
    pub merkle_root: [u8; 32],
    pub signature_valid: bool,
}

/// Result of [`StrataLog::verify_tail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailReport {
    pub segment_no: u32,
    pub frames_verified: u64,
    /// Seq of the first frame in the active segment (empty segment: last_seq + 1).
    pub first_seq: u64,
    /// Seq of the last frame in the active segment (0 when frames_total is 0).
    pub last_seq: u64,
    pub last_frame_hash: [u8; 32],
    /// Present when the active segment file ends in a valid trailer
    /// (a sealed segment that was never rolled — recovery rolls it).
    pub trailer: Option<TrailerCheck>,
}

pub struct StrataLog {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for StrataLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrataLog")
            .field("dir", &self.inner.dir.display().to_string())
            .finish_non_exhaustive()
    }
}

impl Clone for StrataLog {
    fn clone(&self) -> Self {
        StrataLog {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct Inner {
    dir: PathBuf,
    _lock: DirLock,
    signing: SigningKey,
    files: Mutex<WriterState>,
    group: Mutex<Group>,
    cv: Condvar,
}

impl Inner {
    fn lock_group(&self) -> MutexGuard<'_, Group> {
        match self.group.lock() {
            Ok(g) => g,
            Err(_) => panic!("strata: group lock poisoned after a fail-stop abort"),
        }
    }
    fn lock_files(&self) -> MutexGuard<'_, WriterState> {
        match self.files.lock() {
            Ok(f) => f,
            Err(_) => panic!("strata: writer lock poisoned after a fail-stop abort"),
        }
    }
    fn wait_for_flush<'a>(&self, g: MutexGuard<'a, Group>) -> MutexGuard<'a, Group> {
        match self.cv.wait(g) {
            Ok(g) => g,
            Err(_) => panic!("strata: group lock poisoned after a fail-stop abort"),
        }
    }
    fn wait_until<'a>(
        &self,
        g: MutexGuard<'a, Group>,
        deadline: Instant,
    ) -> MutexGuard<'a, Group> {
        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            return g;
        }
        match self.cv.wait_timeout(g, timeout) {
            Ok((g2, _)) => g2,
            Err(_) => panic!("strata: group lock poisoned after a fail-stop abort"),
        }
    }
}

struct Group {
    next_seq: u64,
    last_acked_seq: u64,
    queued: Vec<PendingFrame>,
    completed: Vec<SeqAck>,
    flushing: bool,
    /// Fail-stop latch set when a commit panicked; later appends also panic.
    poisoned: Option<String>,
}

struct PendingFrame {
    seq: u64,
    kind: u8,
    payload: Vec<u8>,
}

struct WriterState {
    file: File,
    path: PathBuf,
    segment_no: u32,
    segment_id: [u8; 16],
    prev_segment_hash: [u8; 32],
    /// Next append position (end of valid frames).
    offset: u64,
    frame_count: u64,
    /// Chain hash of the last frame in this segment (header hash if none).
    last_frame_hash: [u8; 32],
    /// payload_blake3 per frame — merkle leaves, kept for seal().
    leaves: Vec<[u8; 32]>,
    /// Total frames across all segments.
    frames_total: u64,
}

// ---------------------------------------------------------------------------
// Fail-stop helpers
// ---------------------------------------------------------------------------

fn fail_stop_io(last_acked_seq: u64, op: &str, err: &io::Error) -> ! {
    panic!(
        "strata fail-stop {{ op: {op}, errno: {:?}, detail: {err}, last_acked_seq: {last_acked_seq} }} \
         — deliberate abort; no retry, no fallback",
        err.raw_os_error()
    )
}

fn fail_stop_msg(last_acked_seq: u64, op: &str, detail: &str) -> ! {
    panic!(
        "strata fail-stop {{ op: {op}, errno: None, detail: {detail}, last_acked_seq: {last_acked_seq} }} \
         — deliberate abort; no retry, no fallback"
    )
}

fn halt_err(last_acked_seq: u64, segment: u32, offset: u64, reason: impl Into<String>) -> StrataError {
    StrataError::Halt(HaltDetail {
        last_acked_seq,
        segment,
        offset,
        reason: reason.into(),
    })
}

// ---------------------------------------------------------------------------
// Small file helpers
// ---------------------------------------------------------------------------

fn fill_random(buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut f = File::open("/dev/urandom")?;
        f.read_exact(buf)
    }
    #[cfg(not(unix))]
    {
        // Weak std-only fallback for non-target platforms.
        let mut seed = blake3::Hasher::new();
        // SystemTimeError is not an io::Error. A clock before the epoch just
        // yields a zero seed; this path is a weak fallback, not entropy.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes();
        seed.update(&nanos);
        seed.update(&(std::process::id() as u64).to_le_bytes());
        seed.update(&(buf.len() as u64).to_le_bytes());
        let mut block = seed.finalize().as_slice().to_vec();
        let mut taken = 0;
        while taken < buf.len() {
            for b in block.iter() {
                if taken == buf.len() {
                    break;
                }
                buf[taken] = *b;
                taken += 1;
            }
            let mut h = blake3::Hasher::new();
            h.update(&block);
            block = h.finalize().as_slice().to_vec();
        }
        Ok(())
    }
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        let _ = offset;
        let mut handle: &File = file;
        handle.seek(SeekFrom::Start(0))?; // offset applied by callers below
        handle.read_exact(buf)
    }
}

fn load_or_create_key(dir: &Path) -> Result<SigningKey, StrataError> {
    let path = dir.join(KEY_FILE);
    if path.exists() {
        let bytes = fs::read(&path)?;
        let seed: [u8; 32] = bytes
            .try_into()
            .map_err(|_| StrataError::Corrupt(format!("{KEY_FILE} is not 32 bytes")))?;
        Ok(SigningKey::from_bytes(&seed))
    } else {
        let mut seed = [0u8; 32];
        fill_random(&mut seed)?;
        #[cfg(unix)]
        let mut f = {
            use std::os::unix::fs::OpenOptionsExt;
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)?
        };
        #[cfg(not(unix))]
        let mut f = OpenOptions::new().write(true).create_new(true).open(&path)?;
        f.write_all(&seed)?;
        sync::sync_file(&f, SyncPurpose::Metadata)?;
        sync::sync_dir(dir)?;
        Ok(SigningKey::from_bytes(&seed))
    }
}

#[derive(BorshSerialize, BorshDeserialize)]
struct HeadState {
    last_acked_seq: u64,
}

fn read_head_state(dir: &Path) -> Result<Option<u64>, StrataError> {
    match fs::read(dir.join(HEAD_STATE)) {
        Ok(b) => {
            let s: HeadState = borsh::from_slice(&b)
                .map_err(|e| StrataError::Corrupt(format!("{HEAD_STATE}: {e}")))?;
            Ok(Some(s.last_acked_seq))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Atomic watermark replace: temp file + fsync + rename + dir fsync. The temp
/// path is exclusive to the single writer; a leftover `head.state.tmp` after
/// a crash is simply overwritten next time and ignored on open.
pub(crate) fn write_head_state(dir: &Path, seq: u64) -> io::Result<()> {
    let tmp = dir.join(HEAD_STATE_TMP);
    let mut f = File::create(&tmp)?;
    f.write_all(&borsh::to_vec(&HeadState { last_acked_seq: seq })?)?;
    sync::sync_file(&f, SyncPurpose::Metadata)?;
    drop(f);
    fs::rename(&tmp, dir.join(HEAD_STATE))?;
    sync::sync_dir(dir)?;
    Ok(())
}

fn hex32(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn segment_filename(no: u32, id: &[u8; 16]) -> String {
    format!("{no:08}-{}.seg", hex32(id))
}

/// All `NNNNNNNN-<uuid>.seg` files, sorted by number, numbering validated dense.
pub(crate) fn list_segments(dir: &Path) -> Result<Vec<(u32, PathBuf)>, StrataError> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(stem) = name.strip_suffix(".seg") else {
            continue;
        };
        let (num, hex) = stem.split_once('-').ok_or_else(|| {
            StrataError::Corrupt(format!("bad segment name: {name}"))
        })?;
        if num.len() != 8
            || !num.bytes().all(|b| b.is_ascii_digit())
            || hex.len() != 32
            || !hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(StrataError::Corrupt(format!("bad segment name: {name}")));
        }
        let no: u32 = num
            .parse()
            .map_err(|_| StrataError::Corrupt(format!("segment number out of range: {name}")))?;
        out.push((no, entry.path()));
    }
    out.sort_by_key(|(no, _)| *no);
    for (i, (no, _)) in out.iter().enumerate() {
        if *no != i as u32 {
            return Err(StrataError::Corrupt(format!(
                "segment numbering not dense: expected {i:08}, found {no:08}"
            )));
        }
    }
    Ok(out)
}

fn create_segment(
    dir: &Path,
    no: u32,
    prev_segment_hash: [u8; 32],
) -> Result<(PathBuf, SegmentHeader, File), StrataError> {
    let mut id = [0u8; 16];
    fill_random(&mut id)?;
    let header = SegmentHeader {
        magic: SEGMENT_MAGIC,
        version: SEGMENT_VERSION,
        segment_id: id,
        prev_segment_hash,
    };
    let path = dir.join(segment_filename(no, &id));
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    f.write_all(&borsh::to_vec(&header)?)?;
    sync::sync_file(&f, SyncPurpose::Segment)?;
    sync::sync_dir(dir)?;
    Ok((path, header, f))
}

fn truncate_segment(path: &Path, len: u64) -> Result<(), StrataError> {
    let f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(len)?;
    sync::sync_file(&f, SyncPurpose::Segment)?;
    Ok(())
}

fn parse_header(bytes: &[u8]) -> Option<SegmentHeader> {
    if bytes.len() < HEADER_WIRE_SIZE {
        return None;
    }
    let h: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).ok()?;
    if h.magic != SEGMENT_MAGIC || h.version != SEGMENT_VERSION {
        return None;
    }
    Some(h)
}

// ---------------------------------------------------------------------------
// Recovery scan
// ---------------------------------------------------------------------------

struct ScanState {
    frame_count: u64,
    last_frame_hash: [u8; 32],
    leaves: Vec<[u8; 32]>,
    /// End of the valid frame region (start of trailer / torn tail / EOF).
    end_offset: usize,
}

enum TailStop {
    /// Frames run cleanly to EOF with no trailer.
    Clean,
    /// Exactly TRAILER_WIRE_SIZE bytes remain and they parse as a trailer.
    Trailer { trailer: SegmentTrailer, offset: usize },
    /// First bad frame: torn, short, zero, blake3 mismatch, or chain break.
    Torn { offset: usize, reason: String },
}

fn scan_frames(bytes: &[u8], header_hash: [u8; 32]) -> (TailStop, ScanState) {
    let mut st = ScanState {
        frame_count: 0,
        last_frame_hash: header_hash,
        leaves: Vec::new(),
        end_offset: HEADER_WIRE_SIZE,
    };
    let mut off = HEADER_WIRE_SIZE;
    loop {
        let rem = bytes.len() - off;
        if rem == 0 {
            st.end_offset = off;
            return (TailStop::Clean, st);
        }
        if rem == TRAILER_WIRE_SIZE {
            st.end_offset = off;
            if let Ok(trailer) = borsh::from_slice::<SegmentTrailer>(&bytes[off..]) {
                return (TailStop::Trailer { trailer, offset: off }, st);
            }
            return (
                TailStop::Torn {
                    offset: off,
                    reason: "trailer-sized tail failed to parse".into(),
                },
                st,
            );
        }
        if rem < format::FRAME_FIXED_WIRE_SIZE {
            st.end_offset = off;
            return (
                TailStop::Torn {
                    offset: off,
                    reason: "short frame header".into(),
                },
                st,
            );
        }
        match format::parse_frame(&bytes[off..]) {
            Err(e) => {
                st.end_offset = off;
                return (
                    TailStop::Torn {
                        offset: off,
                        reason: format!("frame parse: {e}"),
                    },
                    st,
                );
            }
            Ok((frame, used)) => {
                if frame.payload_blake3 != format::payload_blake3(frame.kind, &frame.payload) {
                    st.end_offset = off;
                    return (
                        TailStop::Torn {
                            offset: off,
                            reason: "payload blake3 mismatch".into(),
                        },
                        st,
                    );
                }
                if frame.prev_frame_hash != st.last_frame_hash {
                    st.end_offset = off;
                    return (
                        TailStop::Torn {
                            offset: off,
                            reason: "frame chain link mismatch".into(),
                        },
                        st,
                    );
                }
                st.last_frame_hash = format::frame_hash(&frame);
                st.leaves.push(frame.payload_blake3);
                st.frame_count += 1;
                off += used;
            }
        }
    }
}

fn validate_trailer(
    trailer: &SegmentTrailer,
    st: &ScanState,
    header: &SegmentHeader,
    signing: &SigningKey,
) -> Result<(), String> {
    if trailer.frame_count != st.frame_count {
        return Err(format!(
            "trailer frame_count {} != scanned {}",
            trailer.frame_count, st.frame_count
        ));
    }
    let merkle = format::merkle_root(&st.leaves);
    if trailer.merkle_root != merkle {
        return Err("trailer merkle root mismatch".into());
    }
    let msg = format::signature_message(
        &header.segment_id,
        &header.prev_segment_hash,
        &trailer.merkle_root,
    );
    let sig = Signature::from_bytes(&trailer.signature);
    if signing.verifying_key().verify(&msg, &sig).is_err() {
        return Err("trailer signature invalid".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The commit pipeline: WRITING -> SYNCING -> VERIFYING -> DURABLE -> ACK
// ---------------------------------------------------------------------------

struct WrittenFrame {
    seq: u64,
    offset: u64,
    wire: Vec<u8>,
    frame_hash: [u8; 32],
}

fn commit_batch(
    dir: &Path,
    w: &mut WriterState,
    batch: Vec<PendingFrame>,
    watermark_before: u64,
) -> Vec<SeqAck> {
    let mut written: Vec<WrittenFrame> = Vec::with_capacity(batch.len());

    // WRITING: serialize and append each frame at the end of the segment.
    for pf in batch {
        let payload_blake3 = format::payload_blake3(pf.kind, &pf.payload);
        let frame = Frame {
            kind: pf.kind,
            payload: pf.payload,
            payload_blake3,
            prev_frame_hash: w.last_frame_hash,
        };
        let wire = match borsh::to_vec(&frame) {
            Ok(b) => b,
            Err(e) => fail_stop_io(watermark_before, "serialize", &e),
        };
        if let Err(e) = w.file.write_all(&wire) {
            fail_stop_io(watermark_before, "write", &e);
        }
        let frame_hash = format::hash_slice(&wire);
        let offset = w.offset;
        w.offset += wire.len() as u64;
        w.frame_count += 1;
        w.frames_total += 1;
        w.leaves.push(payload_blake3);
        w.last_frame_hash = frame_hash;
        written.push(WrittenFrame {
            seq: pf.seq,
            offset,
            wire,
            frame_hash,
        });
    }

    // SYNCING: one durability barrier for the whole group.
    if let Err(e) = sync::sync_file(&w.file, SyncPurpose::Segment) {
        fail_stop_io(watermark_before, "sync", &e);
    }

    // VERIFYING: pread every frame back, recompute blake3, compare.
    let mut readback: Vec<u8> = Vec::new();
    for wr in &written {
        readback.clear();
        readback.resize(wr.wire.len(), 0);
        if let Err(e) = read_exact_at(&w.file, &mut readback, wr.offset) {
            fail_stop_io(watermark_before, "verify-pread", &e);
        }
        if readback != wr.wire {
            fail_stop_msg(
                watermark_before,
                "verify-bytes",
                &format!(
                    "readback differs from written bytes at seq {} offset {}",
                    wr.seq, wr.offset
                ),
            );
        }
        let (frame, used) = match format::parse_frame(&readback) {
            Ok(x) => x,
            Err(e) => fail_stop_io(watermark_before, "verify-parse", &e),
        };
        if used != wr.wire.len() {
            fail_stop_msg(
                watermark_before,
                "verify-length",
                &format!("frame decode length drift at seq {}", wr.seq),
            );
        }
        if frame.payload_blake3 != format::payload_blake3(frame.kind, &frame.payload) {
            fail_stop_msg(
                watermark_before,
                "verify-blake3",
                &format!("recomputed payload blake3 mismatch at seq {}", wr.seq),
            );
        }
    }

    // DURABLE: atomically replace the acked watermark.
    let new_watermark = written.last().expect("non-empty batch").seq;
    if let Err(e) = write_head_state(dir, new_watermark) {
        fail_stop_io(watermark_before, "watermark", &e);
    }

    // ACK (returned to callers only after verify + watermark).
    written
        .into_iter()
        .map(|wr| SeqAck {
            seq: wr.seq,
            frame_hash: wr.frame_hash,
        })
        .collect()
}

fn take_completed(completed: &mut Vec<SeqAck>, mine: &[u64]) -> Option<Vec<SeqAck>> {
    if mine.iter().all(|s| completed.iter().any(|a| a.seq == *s)) {
        let mut out = Vec::with_capacity(mine.len());
        for s in mine {
            if let Some(pos) = completed.iter().position(|a| a.seq == *s) {
                out.push(completed.remove(pos));
            }
        }
        Some(out)
    } else {
        None
    }
}

fn poisoned_panic(reason: &str, last_acked_seq: u64) -> ! {
    panic!(
        "strata fail-stop {{ op: group, errno: None, detail: {reason}, \
         last_acked_seq: {last_acked_seq} }} — deliberate abort; no retry, no fallback"
    )
}

// ---------------------------------------------------------------------------
// StrataLog
// ---------------------------------------------------------------------------

impl StrataLog {
    /// Open (and recover) a log directory. Creates the directory, the signing
    /// key, and segment 0 on first use.
    ///
    /// Recovery scans every segment: sealed segments must carry a valid
    /// trailer and signature; the active segment's tail is truncated at the
    /// first torn/short/zero/corrupt frame when that frame is above the acked
    /// watermark, otherwise [`StrataError::Halt`] is returned and nothing is
    /// modified.
    pub fn open(dir: impl AsRef<Path>) -> Result<StrataLog, StrataError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let lock = DirLock::acquire(&dir)?;
        let signing = load_or_create_key(&dir)?;
        let last_acked_seq = read_head_state(&dir)?.unwrap_or(0);

        let mut segs = list_segments(&dir)?;
        if segs.is_empty() {
            create_segment(&dir, 0, GENESIS_PREV_SEGMENT_HASH)?;
            segs = list_segments(&dir)?;
        }

        let mut prev_hash_expected = GENESIS_PREV_SEGMENT_HASH;
        let mut frames_total: u64 = 0;

        enum ActiveSetup {
            Existing {
                no: u32,
                path: PathBuf,
                header: SegmentHeader,
                state: ScanState,
            },
            New {
                no: u32,
                prev: [u8; 32],
            },
        }
        let mut active: Option<ActiveSetup> = None;

        let last_idx = segs.len() - 1;
        for (idx, (no, path)) in segs.iter().enumerate() {
            let is_last = idx == last_idx;
            let first_seq_here = frames_total + 1;
            let bytes = fs::read(path)?;

            let header = match parse_header(&bytes) {
                Some(h) => h,
                None => {
                    if is_last && frames_total >= last_acked_seq {
                        // No acked frame lives in this segment; all of it is
                        // expendable. Recreate it in place.
                        let _ = fs::remove_file(path);
                        active = Some(ActiveSetup::New {
                            no: *no,
                            prev: prev_hash_expected,
                        });
                        break;
                    }
                    return Err(halt_err(
                        last_acked_seq,
                        *no,
                        0,
                        "segment header unreadable (bad magic/version/length)",
                    ));
                }
            };
            let hh = format::header_hash(&header);

            if header.prev_segment_hash != prev_hash_expected {
                return Err(halt_err(
                    last_acked_seq,
                    *no,
                    0,
                    format!(
                        "segment chain: prev_segment_hash does not match segment {no}'s hash"
                    ),
                ));
            }

            let (stop, st) = scan_frames(&bytes, hh);

            if !is_last {
                // Sealed segments are history: any damage halts.
                match stop {
                    TailStop::Trailer { trailer, .. } => {
                        if let Err(reason) = validate_trailer(&trailer, &st, &header, &signing) {
                            return Err(halt_err(
                                last_acked_seq,
                                *no,
                                st.end_offset as u64,
                                format!("sealed segment trailer: {reason}"),
                            ));
                        }
                    }
                    TailStop::Clean => {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            bytes.len() as u64,
                            "sealed segment is missing its trailer",
                        ))
                    }
                    TailStop::Torn { offset, reason } => {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            format!("sealed segment damaged: {reason}"),
                        ))
                    }
                }
                prev_hash_expected = format::hash_slice(&bytes);
                frames_total += st.frame_count;
                continue;
            }

            // Active (last) segment.
            match stop {
                TailStop::Clean => {
                    frames_total += st.frame_count;
                    active = Some(ActiveSetup::Existing {
                        no: *no,
                        path: path.clone(),
                        header,
                        state: st,
                    });
                }
                TailStop::Trailer { trailer, offset } => {
                    if validate_trailer(&trailer, &st, &header, &signing).is_ok() {
                        // Sealed, but the follow-on segment was never created
                        // (crash between trailer sync and segment create).
                        prev_hash_expected = format::hash_slice(&bytes);
                        frames_total += st.frame_count;
                        active = Some(ActiveSetup::New {
                            no: no + 1,
                            prev: prev_hash_expected,
                        });
                    } else if first_seq_here + st.frame_count <= last_acked_seq {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            "damaged trailer covers acked frames",
                        ));
                    } else {
                        truncate_segment(path, offset as u64)?;
                        frames_total += st.frame_count;
                        active = Some(ActiveSetup::Existing {
                            no: *no,
                            path: path.clone(),
                            header,
                            state: st,
                        });
                    }
                }
                TailStop::Torn { offset, reason } => {
                    let bad_seq = first_seq_here + st.frame_count;
                    if bad_seq <= last_acked_seq {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            format!("damage at/below acked watermark (seq {bad_seq}): {reason}"),
                        ));
                    }
                    // Unacked tail frames may vanish: truncate, fsync, continue.
                    truncate_segment(path, offset as u64)?;
                    frames_total += st.frame_count;
                    active = Some(ActiveSetup::Existing {
                        no: *no,
                        path: path.clone(),
                        header,
                        state: st,
                    });
                }
            }
        }

        if frames_total < last_acked_seq {
            return Err(halt_err(
                last_acked_seq,
                0,
                0,
                format!(
                    "acked frames missing: watermark {last_acked_seq} but only \
                     {frames_total} frames survived recovery"
                ),
            ));
        }

        let (w_no, w_path, w_header, w_state, w_file) = match active
            .expect("recovery always terminates on the last segment")
        {
            ActiveSetup::Existing {
                no,
                path,
                header,
                state,
            } => {
                let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
                file.seek(SeekFrom::Start(state.end_offset as u64))?;
                (no, path, header, state, file)
            }
            ActiveSetup::New { no, prev } => {
                let (path, header, file) = create_segment(&dir, no, prev)?;
                let state = ScanState {
                    frame_count: 0,
                    last_frame_hash: format::header_hash(&header),
                    leaves: Vec::new(),
                    end_offset: HEADER_WIRE_SIZE,
                };
                (no, path, header, state, file)
            }
        };

        let writer = WriterState {
            file: w_file,
            path: w_path,
            segment_no: w_no,
            segment_id: w_header.segment_id,
            prev_segment_hash: w_header.prev_segment_hash,
            offset: w_state.end_offset as u64,
            frame_count: w_state.frame_count,
            last_frame_hash: w_state.last_frame_hash,
            leaves: w_state.leaves,
            frames_total,
        };
        let group = Group {
            next_seq: frames_total + 1,
            last_acked_seq,
            queued: Vec::new(),
            completed: Vec::new(),
            flushing: false,
            poisoned: None,
        };
        Ok(StrataLog {
            inner: Arc::new(Inner {
                dir,
                _lock: lock,
                signing,
                files: Mutex::new(writer),
                group: Mutex::new(group),
                cv: Condvar::new(),
            }),
        })
    }

    /// Append one frame; the ack returns only after sync + verify + watermark.
    pub fn append(&self, kind: u8, payload: &[u8]) -> Result<SeqAck, StrataError> {
        let mut acks = self.append_batch(vec![(kind, payload.to_vec())])?;
        Ok(acks.remove(0))
    }

    /// Group-commit batch API: every frame in the call shares syncs (chunked
    /// at [`MAX_BATCH_FRAMES`]); all acks return only after verify + watermark.
    pub fn append_batch(&self, frames: Vec<(u8, Vec<u8>)>) -> Result<Vec<SeqAck>, StrataError> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let mut g = self.inner.lock_group();
        let mut mine: Vec<u64> = Vec::with_capacity(frames.len());
        for (kind, payload) in frames {
            let seq = g.next_seq;
            g.next_seq = g
                .next_seq
                .checked_add(1)
                .expect("strata: sequence space exhausted");
            g.queued.push(PendingFrame { seq, kind, payload });
            mine.push(seq);
        }
        let mut leader = false;
        let mut first_round = true;
        loop {
            if !leader {
                if let Some(reason) = g.poisoned.clone() {
                    poisoned_panic(&reason, g.last_acked_seq);
                }
                if let Some(acks) = take_completed(&mut g.completed, &mine) {
                    return Ok(acks);
                }
                if g.flushing {
                    g = self.inner.wait_for_flush(g);
                    continue;
                }
                leader = true;
                g.flushing = true;
                if first_round {
                    first_round = false;
                    // Coalescing window: <=2ms or <=64 frames.
                    let deadline = Instant::now() + GROUP_COMMIT_WINDOW;
                    while g.queued.len() < MAX_BATCH_FRAMES {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        g = self.inner.wait_until(g, deadline);
                    }
                }
            }
            // Leader: flush one chunk of at most MAX_BATCH_FRAMES.
            if g.queued.is_empty() {
                if let Some(acks) = take_completed(&mut g.completed, &mine) {
                    g.flushing = false;
                    self.inner.cv.notify_all();
                    return Ok(acks);
                }
                fail_stop_msg(
                    g.last_acked_seq,
                    "invariant",
                    "no queued frames but pending appends are unacked",
                );
            }
            let n = g.queued.len().min(MAX_BATCH_FRAMES);
            let batch: Vec<PendingFrame> = g.queued.drain(..n).collect();
            let watermark_before = g.last_acked_seq;
            drop(g); // followers may queue while the pipeline runs

            let result = {
                let mut files = self.inner.lock_files();
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    commit_batch(&self.inner.dir, &mut files, batch, watermark_before)
                }))
            };

            let acks = match result {
                Ok(acks) => acks,
                Err(payload) => {
                    // Latch the failure for followers, then keep panicking.
                    let mut g = self.inner.lock_group();
                    g.flushing = false;
                    g.poisoned =
                        Some("commit batch failed (see the sibling fail-stop panic)".to_string());
                    self.inner.cv.notify_all();
                    std::panic::resume_unwind(payload);
                }
            };
            g = self.inner.lock_group();
            let batch_last = acks.last().map(|a| a.seq).unwrap_or(watermark_before);
            if batch_last > g.last_acked_seq {
                g.last_acked_seq = batch_last;
            }
            g.completed.extend(acks.iter().copied());
            if mine.iter().all(|s| g.completed.iter().any(|a| a.seq == *s)) {
                let acks = take_completed(&mut g.completed, &mine).expect("just checked");
                g.flushing = false;
                self.inner.cv.notify_all();
                return Ok(acks);
            }
            // Still leader: more of this call's (or followers') frames remain.
        }
    }

    /// Seal the active segment: flush pending frames, write + sync + verify
    /// the signed trailer, then roll a fresh segment chained to the sealed
    /// one's hash.
    pub fn seal(&self) -> Result<SealInfo, StrataError> {
        // Take leadership so no commit can interleave with the trailer.
        let mut g = self.inner.lock_group();
        if let Some(reason) = g.poisoned.clone() {
            poisoned_panic(&reason, g.last_acked_seq);
        }
        while g.flushing {
            g = self.inner.wait_for_flush(g);
        }
        g.flushing = true;
        loop {
            if g.queued.is_empty() {
                break;
            }
            let n = g.queued.len().min(MAX_BATCH_FRAMES);
            let batch: Vec<PendingFrame> = g.queued.drain(..n).collect();
            let watermark_before = g.last_acked_seq;
            drop(g);
            let result = {
                let mut files = self.inner.lock_files();
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    commit_batch(&self.inner.dir, &mut files, batch, watermark_before)
                }))
            };
            match result {
                Err(payload) => {
                    let mut g = self.inner.lock_group();
                    g.flushing = false;
                    g.poisoned =
                        Some("seal flush failed (see the sibling fail-stop panic)".to_string());
                    self.inner.cv.notify_all();
                    std::panic::resume_unwind(payload);
                }
                Ok(acks) => {
                    g = self.inner.lock_group();
                    let last = acks.last().map(|a| a.seq).unwrap_or(g.last_acked_seq);
                    if last > g.last_acked_seq {
                        g.last_acked_seq = last;
                    }
                    g.completed.extend(acks.iter().copied());
                }
            }
        }
        let wm = g.last_acked_seq;
        drop(g);

        // Trailer phase — files lock only (all appenders are parked on the cv).
        let mut files = self.inner.lock_files();
        let merkle = format::merkle_root(&files.leaves);
        let msg = format::signature_message(
            &files.segment_id,
            &files.prev_segment_hash,
            &merkle,
        );
        let signature: Signature = self.inner.signing.sign(&msg);
        let trailer = SegmentTrailer {
            frame_count: files.frame_count,
            merkle_root: merkle,
            signature: signature.to_bytes(),
        };
        let wire = borsh::to_vec(&trailer)?;
        let trailer_at = files.offset;
        if let Err(e) = files.file.write_all(&wire) {
            fail_stop_io(wm, "seal-write", &e);
        }
        if let Err(e) = sync::sync_file(&files.file, SyncPurpose::Segment) {
            fail_stop_io(wm, "seal-sync", &e);
        }
        let mut readback = vec![0u8; wire.len()];
        if let Err(e) = read_exact_at(&files.file, &mut readback, trailer_at) {
            fail_stop_io(wm, "seal-verify", &e);
        }
        if readback != wire {
            fail_stop_msg(wm, "seal-verify", "trailer readback differs from written bytes");
        }
        files.offset += wire.len() as u64;
        let mut whole = vec![0u8; files.offset as usize];
        if let Err(e) = read_exact_at(&files.file, &mut whole, 0) {
            fail_stop_io(wm, "seal-hash", &e);
        }
        let segment_hash = format::hash_slice(&whole);

        let sealed_no = files.segment_no;
        let sealed_id = files.segment_id;
        let frame_count = files.frame_count;
        drop(files);

        let (path, header, file) = create_segment(&self.inner.dir, sealed_no + 1, segment_hash)?;
        let mut files = self.inner.lock_files();
        files.file = file;
        files.path = path;
        files.segment_no = sealed_no + 1;
        files.segment_id = header.segment_id;
        files.prev_segment_hash = segment_hash;
        files.offset = HEADER_WIRE_SIZE as u64;
        files.frame_count = 0;
        files.last_frame_hash = format::header_hash(&header);
        files.leaves = Vec::new();
        drop(files);

        let mut g = self.inner.lock_group();
        g.flushing = false;
        self.inner.cv.notify_all();
        drop(g);

        Ok(SealInfo {
            sealed_segment_no: sealed_no,
            segment_id: sealed_id,
            frame_count,
            merkle_root: merkle,
            segment_hash,
        })
    }

    /// Read frames with `seq >= from_seq`, oldest first, from the durable log.
    /// Holds the writer lock, so it sees a consistent snapshot.
    pub fn read_frames(&self, from_seq: u64) -> Result<Vec<FrameRecord>, StrataError> {
        let _guard = self.inner.lock_files();
        let segs = list_segments(&self.inner.dir)?;
        let mut out = Vec::new();
        let mut seq: u64 = 1;
        for (_no, path) in &segs {
            let bytes = fs::read(path)?;
            let Some(header) = parse_header(&bytes) else {
                return Err(StrataError::Corrupt(format!(
                    "unreadable segment header in {}",
                    path.display()
                )));
            };
            let hh = format::header_hash(&header);
            let (_stop, st) = scan_frames(&bytes, hh);
            let mut off = HEADER_WIRE_SIZE;
            while off < st.end_offset {
                let (frame, used) = format::parse_frame(&bytes[off..])
                    .map_err(|e| StrataError::Corrupt(format!("frame decode: {e}")))?;
                let fh = format::frame_hash(&frame);
                if seq >= from_seq {
                    out.push(FrameRecord {
                        seq,
                        kind: frame.kind,
                        payload: frame.payload,
                        payload_blake3: frame.payload_blake3,
                        prev_frame_hash: frame.prev_frame_hash,
                        frame_hash: fh,
                    });
                }
                seq += 1;
                off += used;
            }
        }
        Ok(out)
    }

    /// Advisory point-in-time head snapshot.
    pub fn head(&self) -> HeadInfo {
        // Never hold both locks: snapshot group state, then writer state.
        let (next_seq, last_acked_seq) = {
            let g = self.inner.lock_group();
            (g.next_seq, g.last_acked_seq)
        };
        let f = self.inner.lock_files();
        HeadInfo {
            last_acked_seq,
            next_seq,
            segment_no: f.segment_no,
            segment_id: f.segment_id,
            prev_segment_hash: f.prev_segment_hash,
            frames_in_segment: f.frame_count,
            frames_total: f.frames_total,
            last_frame_hash: f.last_frame_hash,
        }
    }

    /// Re-verify the active segment's tail from disk: recompute every frame's
    /// blake3 and chain link, and validate the trailer if one is present.
    /// Read-only — never truncates. Returns [`StrataError::Halt`] on any
    /// damage found.
    pub fn verify_tail(&self) -> Result<TailReport, StrataError> {
        let wm = self.inner.lock_group().last_acked_seq;
        let f = self.inner.lock_files();
        let bytes = fs::read(&f.path)?;
        let Some(header) = parse_header(&bytes) else {
            return Err(halt_err(wm, f.segment_no, 0, "verify_tail: header unreadable"));
        };
        let hh = format::header_hash(&header);
        let (stop, st) = scan_frames(&bytes, hh);
        let first_seq = f.frames_total.saturating_sub(st.frame_count) + 1;
        let last_seq = f.frames_total;
        match stop {
            TailStop::Clean => Ok(TailReport {
                segment_no: f.segment_no,
                frames_verified: st.frame_count,
                first_seq,
                last_seq,
                last_frame_hash: st.last_frame_hash,
                trailer: None,
            }),
            TailStop::Trailer { trailer, offset } => match validate_trailer(&trailer, &st, &header, &self.inner.signing) {
                Ok(()) => Ok(TailReport {
                    segment_no: f.segment_no,
                    frames_verified: st.frame_count,
                    first_seq,
                    last_seq,
                    last_frame_hash: st.last_frame_hash,
                    trailer: Some(TrailerCheck {
                        frame_count: trailer.frame_count,
                        merkle_root: trailer.merkle_root,
                        signature_valid: true,
                    }),
                }),
                Err(reason) => Err(halt_err(
                    wm,
                    f.segment_no,
                    offset as u64,
                    format!("verify_tail: trailer: {reason}"),
                )),
            },
            TailStop::Torn { offset, reason } => Err(halt_err(
                wm,
                f.segment_no,
                offset as u64,
                format!("verify_tail: {reason}"),
            )),
        }
    }
}
