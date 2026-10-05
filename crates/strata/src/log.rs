//! The log engine: open/recovery, the fail-stop commit pipeline, group
//! commit, sealing, reads, and tail verification.
//!
//! Locking model: `group` (batch queue, watermarks, seqs) and `files` (the
//! active segment handle) are separate mutexes. The commit path NEVER holds
//! `group` while acquiring `files` — the leader drops `group` before doing
//! I/O — so there is no lock-order cycle anywhere.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
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
use crate::sync::{self, Site, SyncPurpose};

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

/// Result of [`StrataLog::verify_log`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogReport {
    /// Segments scanned, sealed and active.
    pub segments: u32,
    /// Sealed segments whose signed trailer was checked.
    pub sealed_segments: u32,
    /// Frames read back and re-hashed across every segment.
    pub frames: u64,
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
    /// Deterministic-log seed (`open_seeded`). `None` = ordinary random-key,
    /// random-segment-id log. Present: the key and every segment id derive
    /// from it, so two runs over identical frames produce identical bytes.
    seed: Option<[u8; 32]>,
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
    /// A commit was refused because the volume is full and its writes were
    /// rolled back. Discard every queued frame (their seqs would otherwise
    /// leave a hole), reopen the seq space at the log tail, and wake the
    /// parked appenders so each sees the refusal.
    fn refuse_queued(&self, g: &mut Group, err: &io::Error, next_seq: u64) {
        g.queued.clear();
        g.epoch += 1;
        g.next_seq = next_seq;
        g.refusal = Some((err.kind(), err.to_string()));
        g.flushing = false;
        self.cv.notify_all();
    }

    fn wait_for_flush<'a>(&self, g: MutexGuard<'a, Group>) -> MutexGuard<'a, Group> {
        match self.cv.wait(g) {
            Ok(g) => g,
            Err(_) => panic!("strata: group lock poisoned after a fail-stop abort"),
        }
    }
    fn wait_until<'a>(&self, g: MutexGuard<'a, Group>, deadline: Instant) -> MutexGuard<'a, Group> {
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
    /// Set when a read of durable history found damage. Later appends and
    /// seals return the halt instead of extending a log that no longer
    /// verifies.
    halted: Option<HaltDetail>,
    /// Bumped each time a commit is refused for lack of space. An appender
    /// whose frames were queued under an older epoch and were not acked had
    /// them discarded with the refused commit.
    epoch: u64,
    /// The most recent refusal, reported to the appenders it discarded.
    refusal: Option<(io::ErrorKind, String)>,
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

fn halt_err(
    last_acked_seq: u64,
    segment: u32,
    offset: u64,
    reason: impl Into<String>,
) -> StrataError {
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
    // OS CSPRNG on every platform (getrandom(2)/getentropy/BCryptGenRandom).
    // No weak fallback: a platform without entropy fails the open instead of
    // minting a derivable log signing key.
    getrandom::fill(buf).map_err(|e| io::Error::other(e.to_string()))
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        // seek_read moves the file cursor on Windows (pread does not), and
        // appends write at the cursor, so put it back where it was.
        use std::os::windows::fs::FileExt;
        let mut handle: &File = file;
        let resume = handle.stream_position()?;
        let mut done = 0usize;
        let read = loop {
            if done == buf.len() {
                break Ok(());
            }
            match file.seek_read(&mut buf[done..], offset + done as u64) {
                Ok(0) => {
                    break Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "positioned read reached the end of the segment",
                    ));
                }
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => break Err(e),
            }
        };
        handle.seek(SeekFrom::Start(resume))?;
        read
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::Read;
        let mut handle: &File = file;
        let resume = handle.stream_position()?;
        handle.seek(SeekFrom::Start(offset))?;
        let read = handle.read_exact(buf);
        handle.seek(SeekFrom::Start(resume))?;
        read
    }
}

/// True when any segment already holds bytes past its header. That log
/// has a key; open must not write a replacement into the log directory.
fn existing_log_has_writes(dir: &Path) -> Result<bool, StrataError> {
    if !dir.exists() {
        return Ok(false);
    }
    for (_, path) in list_segments(dir)? {
        if fs::metadata(&path)?.len() > HEADER_WIRE_SIZE as u64 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn load_or_create_key(dir: &Path, log_seed: Option<&[u8; 32]>) -> Result<SigningKey, StrataError> {
    let path = dir.join(KEY_FILE);
    if path.exists() {
        let bytes = fs::read(&path)?;
        let seed: [u8; 32] = bytes
            .try_into()
            .map_err(|_| StrataError::Corrupt(format!("{KEY_FILE} is not 32 bytes")))?;
        Ok(SigningKey::from_bytes(&seed))
    } else if existing_log_has_writes(dir)? {
        // Frames are already durable. A fresh key cannot attest them, and
        // must not be minted into the log directory.
        Err(StrataError::Corrupt(
            "signing key missing for an existing log; refusing to mint a new key".into(),
        ))
    } else {
        // Seeded logs derive the signing key from the caller-supplied seed so
        // two seeded runs produce identical signatures and segment bytes.
        let mut seed = [0u8; 32];
        match log_seed {
            Some(s) => seed.copy_from_slice(s),
            None => fill_random(&mut seed)?,
        }
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
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let written = sync::guard_space(Site::Key)
            .and_then(|_| f.write_all(&seed))
            .and_then(|_| sync::sync_file(&f, SyncPurpose::Metadata))
            .and_then(|_| sync::sync_dir(dir));
        if let Err(e) = written {
            // A key file that is empty or short would fail every later open.
            drop(f);
            let _ = fs::remove_file(&path);
            return Err(e.into());
        }
        Ok(SigningKey::from_bytes(&seed))
    }
}

#[derive(BorshSerialize, BorshDeserialize)]
struct HeadState {
    last_acked_seq: u64,
}

/// The recorded acked watermark in `dir`, or `None` when `head.state` does
/// not exist. An unreadable file is [`StrataError::Corrupt`].
pub fn read_head_state(dir: &Path) -> Result<Option<u64>, StrataError> {
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
#[cfg(test)]
pub(crate) fn write_head_state(dir: &Path, seq: u64) -> io::Result<()> {
    stage_head_state(dir, seq)?;
    publish_head_state(dir)
}

/// Write and sync the replacement watermark beside the live one. On failure
/// the temp file is removed and the live watermark is untouched.
fn stage_head_state(dir: &Path, seq: u64) -> io::Result<()> {
    let tmp = dir.join(HEAD_STATE_TMP);
    let staged = (|| {
        sync::guard_space(Site::HeadState)?;
        let mut f = File::create(&tmp)?;
        f.write_all(&borsh::to_vec(&HeadState {
            last_acked_seq: seq,
        })?)?;
        sync::sync_file(&f, SyncPurpose::Metadata)
    })();
    if staged.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    staged
}

/// Atomically replace the live watermark with the staged one.
fn publish_head_state(dir: &Path) -> io::Result<()> {
    fs::rename(dir.join(HEAD_STATE_TMP), dir.join(HEAD_STATE))?;
    sync::sync_dir(dir)?;
    Ok(())
}

/// True when the volume (or the user's quota on it) has no room left.
pub(crate) fn is_storage_full(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
    )
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
        if name.starts_with('.') {
            // a file-manager or archive sidecar (`._…seg`, `.DS_Store`), never a segment
            continue;
        }
        let Some(stem) = name.strip_suffix(".seg") else {
            continue;
        };
        let (num, hex) = stem
            .split_once('-')
            .ok_or_else(|| StrataError::Corrupt(format!("bad segment name: {name}")))?;
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

/// Deterministic segment id for seeded logs: derived from the log seed and
/// the segment number so two seeded runs produce byte-identical segments.
fn seeded_segment_id(seed: &[u8; 32], no: u32) -> [u8; 16] {
    let hash = blake3::Hasher::new_derive_key("vestige strata segment id v1")
        .update(seed)
        .update(&no.to_be_bytes())
        .finalize();
    let full = *hash.as_bytes();
    let mut id = [0u8; 16];
    id.copy_from_slice(&full[..16]);
    id
}

fn create_segment(
    dir: &Path,
    no: u32,
    prev_segment_hash: [u8; 32],
    seed: Option<&[u8; 32]>,
) -> Result<(PathBuf, SegmentHeader, File), StrataError> {
    let id = match seed {
        Some(seed) => seeded_segment_id(seed, no),
        None => {
            let mut id = [0u8; 16];
            fill_random(&mut id)?;
            id
        }
    };
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
    let wire = borsh::to_vec(&header)?;
    let written = sync::guard_space(Site::SegmentHeader)
        .and_then(|_| f.write_all(&wire))
        .and_then(|_| sync::sync_file(&f, SyncPurpose::Segment))
        .and_then(|_| sync::sync_dir(dir));
    if let Err(e) = written {
        // A segment without a complete header is not part of the log.
        drop(f);
        let _ = fs::remove_file(&path);
        return Err(e.into());
    }
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
    Trailer {
        trailer: SegmentTrailer,
        offset: usize,
    },
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
            // A frame with a 35-byte payload is also 104 bytes long. A frame
            // that parses, hashes and chains is a frame; only otherwise are
            // the bytes read as a trailer (a fixed-size trailer parses from
            // any 104 bytes).
            let chained = match format::parse_frame(&bytes[off..]) {
                Ok((frame, used))
                    if used == rem
                        && frame.payload_blake3
                            == format::payload_blake3(frame.kind, &frame.payload)
                        && frame.prev_frame_hash == st.last_frame_hash =>
                {
                    Some((frame, used))
                }
                _ => None,
            };
            if let Some((frame, used)) = chained {
                st.last_frame_hash = format::frame_hash(&frame);
                st.leaves.push(frame.payload_blake3);
                st.frame_count += 1;
                off += used;
                continue;
            }
            st.end_offset = off;
            if let Ok(trailer) = borsh::from_slice::<SegmentTrailer>(&bytes[off..]) {
                return (
                    TailStop::Trailer {
                        trailer,
                        offset: off,
                    },
                    st,
                );
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

/// A torn final write ends mid-frame. A complete frame that fails blake3
/// or the chain link is corruption, not a tear, and must not be truncated.
///
/// A damaged length prefix makes an intact frame look cut off. `chain_tip` is
/// the hash the next frame must link to: when the tail holds a complete frame
/// that hashes and links under some other length, the declared length is the
/// damage, the frame (and anything after it) may have been acked, and the tail
/// is not a tear.
fn is_torn_final_write(tail: &[u8], chain_tip: &[u8; 32]) -> bool {
    if tail.len() < format::FRAME_FIXED_WIRE_SIZE {
        return !tail.is_empty();
    }
    let Ok(prefix) = <[u8; 4]>::try_from(&tail[..4]) else {
        return true;
    };
    let declared = u32::from_le_bytes(prefix) as u64;
    let need = 4u64 + 1 + declared + 32 + 32;
    if (tail.len() as u64) >= need {
        return false;
    }
    !holds_intact_frame(tail, chain_tip)
}

/// True when `tail` begins with a complete frame, of any declared length,
/// whose payload hash verifies and whose link is `chain_tip`. Frames end with
/// `payload_blake3 || prev_frame_hash`, so candidate ends are found by looking
/// for the chain tip in the link position and hashing only on a match.
fn holds_intact_frame(tail: &[u8], chain_tip: &[u8; 32]) -> bool {
    let kind = tail[4];
    for end in format::FRAME_FIXED_WIRE_SIZE..=tail.len() {
        if tail[end - 32..end] != chain_tip[..] {
            continue;
        }
        let payload = &tail[5..end - 64];
        if tail[end - 64..end - 32] == format::payload_blake3(kind, payload)[..] {
            return true;
        }
    }
    false
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
// Whole-log verification
// ---------------------------------------------------------------------------

fn halted_for_writes(detail: HaltDetail) -> StrataError {
    StrataError::Halt(HaltDetail {
        reason: format!(
            "writes are refused after damage was found in the durable log: {}",
            detail.reason
        ),
        ..detail
    })
}

/// Walk every segment in order, verifying as it goes, and hand each frame to
/// `on_frame`. Damage anywhere, in a sealed segment or the active one, is a
/// halt. `expected_frames` is the writer's own frame count.
fn walk_segments(
    dir: &Path,
    signing: &SigningKey,
    expected_frames: u64,
    wm: u64,
    mut on_frame: impl FnMut(FrameRecord),
) -> Result<LogReport, StrataError> {
    let segs = list_segments(dir)?;
    if segs.is_empty() {
        return Err(halt_err(wm, 0, 0, "log has no segments"));
    }
    let last_idx = segs.len() - 1;
    let mut prev_hash_expected = GENESIS_PREV_SEGMENT_HASH;
    let mut seq: u64 = 0;
    let mut sealed_segments = 0u32;
    for (idx, (no, path)) in segs.iter().enumerate() {
        let is_last = idx == last_idx;
        let bytes = fs::read(path)?;
        let Some(header) = parse_header(&bytes) else {
            return Err(halt_err(
                wm,
                *no,
                0,
                "segment header unreadable (bad magic/version/length)",
            ));
        };
        if header.prev_segment_hash != prev_hash_expected {
            return Err(halt_err(
                wm,
                *no,
                0,
                format!("segment chain: prev_segment_hash does not match segment {no}'s hash"),
            ));
        }
        let (stop, st) = scan_frames(&bytes, format::header_hash(&header));
        match stop {
            TailStop::Trailer { trailer, offset } => {
                if let Err(reason) = validate_trailer(&trailer, &st, &header, signing) {
                    return Err(halt_err(
                        wm,
                        *no,
                        offset as u64,
                        format!("segment trailer: {reason}"),
                    ));
                }
                sealed_segments += 1;
                prev_hash_expected = format::hash_slice(&bytes);
            }
            TailStop::Clean if is_last => {}
            TailStop::Clean => {
                return Err(halt_err(
                    wm,
                    *no,
                    bytes.len() as u64,
                    "sealed segment is missing its trailer",
                ));
            }
            TailStop::Torn { offset, reason } => {
                return Err(halt_err(
                    wm,
                    *no,
                    offset as u64,
                    format!(
                        "segment damaged at frame {}: {reason}",
                        seq + st.frame_count + 1
                    ),
                ));
            }
        }
        let mut off = HEADER_WIRE_SIZE;
        while off < st.end_offset {
            let (frame, used) = format::parse_frame(&bytes[off..])
                .map_err(|e| StrataError::Corrupt(format!("frame decode: {e}")))?;
            seq += 1;
            let fh = format::frame_hash(&frame);
            on_frame(FrameRecord {
                seq,
                kind: frame.kind,
                payload: frame.payload,
                payload_blake3: frame.payload_blake3,
                prev_frame_hash: frame.prev_frame_hash,
                frame_hash: fh,
            });
            off += used;
        }
    }
    if seq != expected_frames {
        return Err(halt_err(
            wm,
            0,
            0,
            format!("log holds {seq} frames on disk but {expected_frames} were committed"),
        ));
    }
    Ok(LogReport {
        segments: segs.len() as u32,
        sealed_segments,
        frames: seq,
    })
}

/// The active segment on disk must hold exactly the frames the writer
/// committed, with no trailer and no stray bytes.
fn check_active_segment_on_disk(w: &WriterState, wm: u64) -> Result<(), StrataError> {
    let bytes = fs::read(&w.path)?;
    let Some(header) = parse_header(&bytes) else {
        return Err(halt_err(
            wm,
            w.segment_no,
            0,
            "seal: active segment header unreadable",
        ));
    };
    let (stop, st) = scan_frames(&bytes, format::header_hash(&header));
    match stop {
        TailStop::Clean => {}
        TailStop::Trailer { offset, .. } => {
            return Err(halt_err(
                wm,
                w.segment_no,
                offset as u64,
                "seal: active segment already ends in a trailer",
            ));
        }
        TailStop::Torn { offset, reason } => {
            return Err(halt_err(
                wm,
                w.segment_no,
                offset as u64,
                format!("seal: active segment damaged: {reason}"),
            ));
        }
    }
    if st.frame_count != w.frame_count
        || st.leaves != w.leaves
        || st.last_frame_hash != w.last_frame_hash
        || st.end_offset as u64 != w.offset
    {
        return Err(halt_err(
            wm,
            w.segment_no,
            st.end_offset as u64,
            "seal: active segment on disk differs from the frames committed to it",
        ));
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

/// Writer state at the start of a commit, kept so a refused commit can be
/// undone exactly.
struct WriterSnapshot {
    offset: u64,
    frame_count: u64,
    frames_total: u64,
    leaves_len: usize,
    last_frame_hash: [u8; 32],
}

impl WriterSnapshot {
    fn take(w: &WriterState) -> Self {
        WriterSnapshot {
            offset: w.offset,
            frame_count: w.frame_count,
            frames_total: w.frames_total,
            leaves_len: w.leaves.len(),
            last_frame_hash: w.last_frame_hash,
        }
    }

    /// Cut the segment back to the snapshot. Nothing in the undone range was
    /// acked, and shrinking a file needs no free space. If the cut itself
    /// fails the segment's state is unknown: fail-stop.
    fn restore(&self, w: &mut WriterState, last_acked_seq: u64) {
        w.offset = self.offset;
        w.frame_count = self.frame_count;
        w.frames_total = self.frames_total;
        w.leaves.truncate(self.leaves_len);
        w.last_frame_hash = self.last_frame_hash;
        let cut = w
            .file
            .set_len(self.offset)
            .and_then(|_| w.file.seek(SeekFrom::Start(self.offset)).map(|_| ()))
            .and_then(|_| sync::sync_file(&w.file, SyncPurpose::Segment));
        if let Err(e) = cut {
            fail_stop_io(last_acked_seq, "rollback", &e);
        }
    }
}

/// Run one commit group. A full volume before the group is durable is
/// returned as `Err` after the segment is restored; every other failure is
/// fail-stop.
fn commit_batch(
    dir: &Path,
    w: &mut WriterState,
    batch: Vec<PendingFrame>,
    watermark_before: u64,
) -> Result<Vec<SeqAck>, io::Error> {
    let snapshot = WriterSnapshot::take(w);
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
        if let Err(e) = sync::guard_space(Site::Frame).and_then(|_| w.file.write_all(&wire)) {
            if is_storage_full(&e) {
                snapshot.restore(w, watermark_before);
                return Err(e);
            }
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
    if let Err(e) = stage_head_state(dir, new_watermark) {
        if is_storage_full(&e) {
            // Nothing is published yet: the group was never acked, so it is
            // undone like a refused write.
            snapshot.restore(w, watermark_before);
            return Err(e);
        }
        fail_stop_io(watermark_before, "watermark", &e);
    }
    if let Err(e) = publish_head_state(dir) {
        fail_stop_io(watermark_before, "watermark", &e);
    }

    // ACK (returned to callers only after verify + watermark).
    Ok(written
        .into_iter()
        .map(|wr| SeqAck {
            seq: wr.seq,
            frame_hash: wr.frame_hash,
        })
        .collect())
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

/// The error handed to an appender whose frames a refused commit discarded.
fn refusal_error(g: &Group) -> StrataError {
    match &g.refusal {
        Some((kind, msg)) => StrataError::Io(io::Error::new(*kind, msg.clone())),
        None => StrataError::Io(io::Error::from(io::ErrorKind::StorageFull)),
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
    /// Recovery scans every segment. A sealed segment's trailer is verified
    /// on open and any mismatch halts, whether or not `head.state` exists.
    /// Only the unsealed active tail may be truncated, and only at a torn
    /// final write above the acked watermark. Anything else returns
    /// [`StrataError::Halt`] and leaves the bytes untouched.
    pub fn open(dir: impl AsRef<Path>) -> Result<StrataLog, StrataError> {
        Self::open_inner(dir, None)
    }

    /// Open a deterministic log: the signing key and every segment id derive
    /// from `seed`, so two seeded runs over identical frames produce
    /// byte-identical segments. Used by the migration tool (replay equality)
    /// and by tests that pin log bytes.
    ///
    /// If the directory already carries a `strata.key`, that key wins and the
    /// seed is ignored (existing logs are never re-keyed). If frames exist
    /// and the key file is missing, open fails instead of minting one.
    pub fn open_seeded(dir: impl AsRef<Path>, seed: [u8; 32]) -> Result<StrataLog, StrataError> {
        Self::open_inner(dir, Some(seed))
    }

    fn open_inner(dir: impl AsRef<Path>, seed: Option<[u8; 32]>) -> Result<StrataLog, StrataError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let lock = DirLock::acquire(&dir)?;
        let signing = load_or_create_key(&dir, seed.as_ref())?;
        // Missing head.state reads as watermark 0, but only a recorded
        // watermark lets recovery drop a damaged tail as never acked.
        let recorded_watermark = read_head_state(&dir)?;
        let watermark_known = recorded_watermark.is_some();
        let last_acked_seq = recorded_watermark.unwrap_or(0);

        let mut segs = list_segments(&dir)?;
        if segs.is_empty() {
            create_segment(&dir, 0, GENESIS_PREV_SEGMENT_HASH, seed.as_ref())?;
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
                    if is_last && frames_total >= last_acked_seq && bytes.len() <= HEADER_WIRE_SIZE
                    {
                        // Creation was cut off before any frame could follow
                        // the header: no acked frame lives in this segment
                        // and none of it is worth keeping. A file that holds
                        // bytes beyond a header-sized prefix may hold acked
                        // frames behind a damaged header, so it halts below
                        // whatever the watermark says.
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
                    format!("segment chain: prev_segment_hash does not match segment {no}'s hash"),
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
                    if let Err(reason) = validate_trailer(&trailer, &st, &header, &signing) {
                        // No segment follows this one, so the trailer was the
                        // last write: a seal cut off by a crash, or a torn
                        // unacked frame that happens to be trailer-sized.
                        // Seal only runs after every frame before it is
                        // acked, so with a recorded watermark covering those
                        // frames, dropping these bytes loses nothing acked.
                        let last_seq_here = (first_seq_here + st.frame_count).saturating_sub(1);
                        if watermark_known && last_seq_here <= last_acked_seq {
                            truncate_segment(path, offset as u64)?;
                            frames_total += st.frame_count;
                            active = Some(ActiveSetup::Existing {
                                no: *no,
                                path: path.clone(),
                                header,
                                state: st,
                            });
                            continue;
                        }
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            format!(
                                "last segment trailer: {reason} (frames through seq {last_seq_here}, \
                                 acked watermark {last_acked_seq}{})",
                                if watermark_known { "" } else { ", head.state missing" }
                            ),
                        ));
                    }
                    // Sealed, but the follow-on segment was never created
                    // (crash between trailer sync and segment create).
                    prev_hash_expected = format::hash_slice(&bytes);
                    frames_total += st.frame_count;
                    active = Some(ActiveSetup::New {
                        no: no + 1,
                        prev: prev_hash_expected,
                    });
                }
                TailStop::Torn { offset, reason } => {
                    let bad_seq = first_seq_here + st.frame_count;
                    // A complete frame that fails its checks is a tear only
                    // when it lies above a recorded watermark: never acked,
                    // as when power fails inside a group commit before its
                    // sync. At or below it, or with no watermark to trust,
                    // halt and do not rebuild a shorter log.
                    let unacked = watermark_known && bad_seq > last_acked_seq;
                    if !is_torn_final_write(&bytes[offset..], &st.last_frame_hash) && !unacked {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            format!(
                                "segment damaged at seq {bad_seq}: {reason} (acked watermark \
                                 {last_acked_seq}{})",
                                if watermark_known {
                                    ""
                                } else {
                                    ", head.state missing"
                                }
                            ),
                        ));
                    }
                    if bad_seq <= last_acked_seq {
                        return Err(halt_err(
                            last_acked_seq,
                            *no,
                            offset as u64,
                            format!("damage at/below acked watermark (seq {bad_seq}): {reason}"),
                        ));
                    }
                    // Unacked torn tail only: truncate, fsync, continue.
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

        let (w_no, w_path, w_header, w_state, w_file) =
            match active.expect("recovery always terminates on the last segment") {
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
                    let (path, header, file) = create_segment(&dir, no, prev, seed.as_ref())?;
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
            halted: None,
            epoch: 0,
            refusal: None,
        };
        Ok(StrataLog {
            inner: Arc::new(Inner {
                dir,
                _lock: lock,
                signing,
                seed,
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
        if let Some(detail) = g.halted.clone() {
            return Err(halted_for_writes(detail));
        }
        let my_epoch = g.epoch;
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
                if g.epoch != my_epoch {
                    // A commit was refused for lack of space and took this
                    // call's unacked frames with it.
                    g.completed.retain(|a| !mine.contains(&a.seq));
                    return Err(refusal_error(&g));
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

            let (result, tail_seq) = {
                let mut files = self.inner.lock_files();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    commit_batch(&self.inner.dir, &mut files, batch, watermark_before)
                }));
                (result, files.frames_total + 1)
            };

            let acks = match result {
                Ok(Ok(acks)) => acks,
                Ok(Err(refused)) => {
                    let mut g = self.inner.lock_group();
                    self.inner.refuse_queued(&mut g, &refused, tail_seq);
                    // Earlier chunks of this call were acked and stay durable;
                    // their acks are not handed back for a failed call.
                    g.completed.retain(|a| !mine.contains(&a.seq));
                    return Err(refused.into());
                }
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
        if let Some(detail) = g.halted.clone() {
            return Err(halted_for_writes(detail));
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
            let (result, tail_seq) = {
                let mut files = self.inner.lock_files();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    commit_batch(&self.inner.dir, &mut files, batch, watermark_before)
                }));
                (result, files.frames_total + 1)
            };
            match result {
                Ok(Err(refused)) => {
                    let mut g = self.inner.lock_group();
                    self.inner.refuse_queued(&mut g, &refused, tail_seq);
                    return Err(refused.into());
                }
                Err(payload) => {
                    let mut g = self.inner.lock_group();
                    g.flushing = false;
                    g.poisoned =
                        Some("seal flush failed (see the sibling fail-stop panic)".to_string());
                    self.inner.cv.notify_all();
                    std::panic::resume_unwind(payload);
                }
                Ok(Ok(acks)) => {
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
        // The trailer signs the in-memory merkle root. Sign it only over bytes
        // that still match: a segment altered since its frames were acked is
        // history that no longer verifies, and a fresh signature must not
        // vouch for it.
        if let Err(err) = check_active_segment_on_disk(&files, wm) {
            drop(files);
            self.latch_halt(&err);
            let mut g = self.inner.lock_group();
            g.flushing = false;
            self.inner.cv.notify_all();
            return Err(err);
        }
        let merkle = format::merkle_root(&files.leaves);
        let msg = format::signature_message(&files.segment_id, &files.prev_segment_hash, &merkle);
        let signature: Signature = self.inner.signing.sign(&msg);
        let trailer = SegmentTrailer {
            frame_count: files.frame_count,
            merkle_root: merkle,
            signature: signature.to_bytes(),
        };
        let wire = match borsh::to_vec(&trailer) {
            Ok(wire) => wire,
            Err(err) => {
                // Nothing written yet: release the writer so appends go on.
                drop(files);
                let mut g = self.inner.lock_group();
                g.flushing = false;
                self.inner.cv.notify_all();
                return Err(err.into());
            }
        };
        let trailer_at = files.offset;
        if let Err(e) = sync::guard_space(Site::Trailer).and_then(|_| files.file.write_all(&wire)) {
            if is_storage_full(&e) {
                // Cut the partial trailer off and leave the segment open.
                let cut = files
                    .file
                    .set_len(trailer_at)
                    .and_then(|_| files.file.seek(SeekFrom::Start(trailer_at)).map(|_| ()))
                    .and_then(|_| sync::sync_file(&files.file, SyncPurpose::Segment));
                if let Err(cut_err) = cut {
                    fail_stop_io(wm, "seal-rollback", &cut_err);
                }
                drop(files);
                let mut g = self.inner.lock_group();
                g.flushing = false;
                self.inner.cv.notify_all();
                return Err(e.into());
            }
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
            fail_stop_msg(
                wm,
                "seal-verify",
                "trailer readback differs from written bytes",
            );
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

        let (path, header, file) = match create_segment(
            &self.inner.dir,
            sealed_no + 1,
            segment_hash,
            self.inner.seed.as_ref(),
        ) {
            Ok(created) => created,
            Err(err) => {
                // A full volume is not damage: take the trailer back off so
                // the segment is open again and appends go on once there is
                // room. Anything else, or an undo that cannot be shown exact,
                // leaves the durable trailer in place and fail-stops like any
                // other post-write failure; reopening rolls a fresh segment
                // forward from the sealed one.
                let still_sealed: Option<String> = if err.is_storage_full() {
                    self.undo_seal(sealed_no, trailer_at, wm)
                        .err()
                        .map(|undo_err| format!(" (the seal could not be undone: {undo_err})"))
                } else {
                    Some(String::new())
                };
                let mut g = self.inner.lock_group();
                g.flushing = false;
                if let Some(why) = still_sealed {
                    g.poisoned = Some(format!(
                        "seal could not create the next segment: {err}{why}"
                    ));
                }
                self.inner.cv.notify_all();
                return Err(err);
            }
        };
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

    /// Undo a seal whose trailer is durable but whose follow-on segment could
    /// not be created: cut the trailer off so segment `sealed_no` is the open
    /// active segment again, exactly as before the seal. The trailer was the
    /// only thing the seal appended, and no frame can have been committed
    /// since (the caller still holds commit leadership).
    ///
    /// `Err` means the undo is not known to be exact and the caller must
    /// fail-stop. Up to the cut the trailer is still in place, and a sealed
    /// last segment is a state `open` already recovers from.
    fn undo_seal(&self, sealed_no: u32, trailer_at: u64, wm: u64) -> Result<(), StrataError> {
        // An open segment followed by another file halts the next open, so
        // nothing of the failed create may remain, now or after a crash:
        // check for a leftover and make its removal durable before the cut.
        let segs = list_segments(&self.inner.dir)?;
        if segs.last().map(|(no, _)| *no) != Some(sealed_no) {
            return Err(StrataError::Corrupt(format!(
                "a segment file follows segment {sealed_no:08}"
            )));
        }
        sync::sync_dir(&self.inner.dir)?;

        let mut files = self.inner.lock_files();
        files.file.set_len(trailer_at)?;
        files.file.seek(SeekFrom::Start(trailer_at))?;
        sync::sync_file(&files.file, SyncPurpose::Segment)?;
        files.offset = trailer_at;
        // Prove it: the bytes on disk are again exactly the committed frames.
        check_active_segment_on_disk(&files, wm)
    }

    /// Read frames with `seq >= from_seq`, oldest first, from the durable log.
    /// Holds the writer lock, so it sees a consistent snapshot.
    ///
    /// Every segment is checked on the way: frame hashes, the frame chain,
    /// the segment chain, and each sealed segment's signed trailer. Damage
    /// anywhere is a [`StrataError::Halt`] and also stops further appends;
    /// a damaged log never yields a shorter prefix.
    pub fn read_frames(&self, from_seq: u64) -> Result<Vec<FrameRecord>, StrataError> {
        let wm = self.inner.lock_group().last_acked_seq;
        let mut out = Vec::new();
        let result = {
            let files = self.inner.lock_files();
            walk_segments(
                &self.inner.dir,
                &self.inner.signing,
                files.frames_total,
                wm,
                |record| {
                    if record.seq >= from_seq {
                        out.push(record);
                    }
                },
            )
        };
        match result {
            Ok(_) => Ok(out),
            Err(err) => {
                self.latch_halt(&err);
                Err(err)
            }
        }
    }

    /// Re-verify every segment from disk against this process's view of the
    /// log: header and frame chains, payload hashes, each sealed segment's
    /// signed trailer, and the frame count. Read-only. Damage returns
    /// [`StrataError::Halt`] and stops further appends.
    pub fn verify_log(&self) -> Result<LogReport, StrataError> {
        let wm = self.inner.lock_group().last_acked_seq;
        let result = {
            let files = self.inner.lock_files();
            walk_segments(
                &self.inner.dir,
                &self.inner.signing,
                files.frames_total,
                wm,
                |_| {},
            )
        };
        if let Err(err) = &result {
            self.latch_halt(err);
        }
        result
    }

    /// `Ok` while the log accepts writes; the recorded halt once a check has
    /// found damage in durable history. Callers whose write path has no error
    /// channel of its own use this to refuse before starting.
    pub fn ensure_writable(&self) -> Result<(), StrataError> {
        match self.inner.lock_group().halted.clone() {
            Some(detail) => Err(halted_for_writes(detail)),
            None => Ok(()),
        }
    }

    /// Remember that durable history no longer verifies, so appends and seals
    /// are refused. Only [`StrataError::Halt`] latches; an I/O error reading
    /// the files says nothing about the bytes.
    fn latch_halt(&self, err: &StrataError) {
        if let StrataError::Halt(detail) = err {
            let mut g = self.inner.lock_group();
            if g.halted.is_none() {
                g.halted = Some(detail.clone());
            }
        }
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
        let result = self.verify_tail_inner();
        if let Err(err) = &result {
            self.latch_halt(err);
        }
        result
    }

    fn verify_tail_inner(&self) -> Result<TailReport, StrataError> {
        let wm = self.inner.lock_group().last_acked_seq;
        let f = self.inner.lock_files();
        let bytes = fs::read(&f.path)?;
        let Some(header) = parse_header(&bytes) else {
            return Err(halt_err(
                wm,
                f.segment_no,
                0,
                "verify_tail: header unreadable",
            ));
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
            TailStop::Trailer { trailer, offset } => {
                match validate_trailer(&trailer, &st, &header, &self.inner.signing) {
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
                }
            }
            TailStop::Torn { offset, reason } => Err(halt_err(
                wm,
                f.segment_no,
                offset as u64,
                format!("verify_tail: {reason}"),
            )),
        }
    }
}
