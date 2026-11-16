//! Replica of one remote writer's segment chain.
//!
//! A replica pins the writer's verifying key and never mints or reads a
//! signing seed. `strata.key` in the replica directory is a hard error: the
//! file is not opened. Frames are admitted only when `payload_blake3`
//! recomputes and `prev_frame_hash` extends the replica tail. A trailer is
//! admitted only when the pinned key verifies `signature_message`.
//!
//! The commit of an accepted frame follows the log's protocol: write, sync,
//! read back, then advance `head.state`. A frame that fails its checks is
//! refused before any byte of it is written.

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::StrataError;
use crate::format::{
    self, frame_hash, payload_blake3, Frame, FrameRecord, SegmentHeader, SegmentTrailer,
    GENESIS_PREV_SEGMENT_HASH, HEADER_WIRE_SIZE, SEGMENT_MAGIC, SEGMENT_VERSION,
    TRAILER_WIRE_SIZE,
};
use crate::lockfile::DirLock;
use crate::log::{self, list_segments, parse_header, scan_frames, write_head_state, SeqAck, TailStop};
use crate::sync::{self, SyncPurpose};

const WRITER_PUB: &str = "writer.pub";
const KEY_FILE: &str = "strata.key";

/// One writer's chain, verified under a pinned public key.
pub struct ReplicaLog {
    dir: PathBuf,
    _lock: DirLock,
    verifying: VerifyingKey,
}

impl ReplicaLog {
    /// Open a replica directory, creating it if needed.
    ///
    /// `writer_verifying_key` is the key the grant pinned. It is recorded in
    /// `writer.pub`. This function does not call the log's key minting path
    /// and does not read `strata.key`.
    pub fn open(
        dir: impl AsRef<Path>,
        writer_verifying_key: VerifyingKey,
    ) -> Result<Self, StrataError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        refuse_signing_seed(&dir)?;
        let lock = DirLock::acquire(&dir)?;
        pin_writer_key(&dir, &writer_verifying_key)?;
        let replica = Self {
            dir,
            _lock: lock,
            verifying: writer_verifying_key,
        };
        let _ = replica.read_frames()?;
        Ok(replica)
    }

    /// The pinned verifying key. This is not a signing key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.verifying
    }

    /// Directory holding the replica segments and `writer.pub`.
    pub fn directory(&self) -> &Path {
        &self.dir
    }

    /// Highest acked frame seq, from `head.state`. Missing means 0.
    pub fn last_acked_seq(&self) -> Result<u64, StrataError> {
        Ok(log::read_head_state(&self.dir)?.unwrap_or(0))
    }

    /// Read every acked frame, re-checking links, payload hashes, and trailers.
    pub fn read_frames(&self) -> Result<Vec<FrameRecord>, StrataError> {
        read_frames_in(&self.dir, &self.verifying)
    }

    /// Start a segment with the peer's header. The header is synced before
    /// any frame is accepted into it.
    pub fn begin_segment(&mut self, header: &SegmentHeader) -> Result<(), StrataError> {
        if header.magic != SEGMENT_MAGIC || header.version != SEGMENT_VERSION {
            return Err(StrataError::Corrupt(
                "replica refused a segment header with a bad magic or version".into(),
            ));
        }
        let segs = list_segments(&self.dir).unwrap_or_default();
        let expected_prev = match segs.last() {
            None => GENESIS_PREV_SEGMENT_HASH,
            Some((_, path)) => {
                let bytes = fs::read(path)?;
                if !matches!(stop_of(&bytes), TailStop::Trailer { .. }) {
                    return Err(StrataError::Corrupt(
                        "replica cannot open a new segment before the active one is sealed"
                            .into(),
                    ));
                }
                format::hash_slice(&bytes)
            }
        };
        if header.prev_segment_hash != expected_prev {
            return Err(StrataError::Corrupt(
                "replica refused a segment header whose prev_segment_hash does not chain".into(),
            ));
        }
        let no = segs.len() as u32;
        let path = self.dir.join(segment_filename(no, &header.segment_id));
        let wire = borsh::to_vec(header).map_err(|err| {
            StrataError::Corrupt(format!("replica header does not serialize: {err}"))
        })?;
        if wire.len() != HEADER_WIRE_SIZE {
            return Err(StrataError::Corrupt(
                "replica header is not the fixed wire size".into(),
            ));
        }
        durable_create(&path, &self.dir, &wire)?;
        Ok(())
    }

    /// Append one frame that already carries its chain link and payload hash.
    ///
    /// A bad link or a bad payload hash returns an error and leaves the
    /// segment file the length it had before the call.
    pub fn append_frame(&mut self, frame: &Frame) -> Result<SeqAck, StrataError> {
        if frame.payload_blake3 != payload_blake3(frame.kind, &frame.payload) {
            return Err(StrataError::Corrupt(
                "replica refused a frame whose payload_blake3 does not recompute".into(),
            ));
        }
        let (path, bytes) = self.active_unsealed()?;
        let (_stop, state) = scan_frames(&bytes, header_hash_of(&bytes)?);
        if frame.prev_frame_hash != state.last_frame_hash {
            return Err(StrataError::Corrupt(
                "replica refused a frame whose prev_frame_hash does not extend the tail".into(),
            ));
        }
        let wire = borsh::to_vec(frame).map_err(|err| {
            StrataError::Corrupt(format!("replica frame does not serialize: {err}"))
        })?;
        let offset = bytes.len() as u64;
        let before = bytes.len() as u64;
        if let Err(err) = extend_file(&path, offset, &wire) {
            let _ = truncate_to(&path, before);
            return Err(err);
        }
        let seq = frames_on_disk(&self.dir, &self.verifying)?;
        write_head_state(&self.dir, seq)?;
        Ok(SeqAck {
            seq,
            frame_hash: frame_hash(frame),
        })
    }

    /// Append a trailer. A bad signature or a bad merkle root is refused
    /// before the trailer bytes are written.
    pub fn append_trailer(&mut self, trailer: &SegmentTrailer) -> Result<(), StrataError> {
        let (path, bytes) = self.active_unsealed()?;
        let header = parse_header(&bytes).ok_or_else(|| {
            StrataError::Corrupt("replica active segment header does not decode".into())
        })?;
        let (_stop, state) = scan_frames(&bytes, format::header_hash(&header));
        if trailer.frame_count != state.frame_count {
            return Err(StrataError::Corrupt(
                "replica refused a trailer whose frame_count does not match".into(),
            ));
        }
        let root = format::merkle_root(&state.leaves);
        if trailer.merkle_root != root {
            return Err(StrataError::Corrupt(
                "replica refused a trailer whose merkle root does not match".into(),
            ));
        }
        let msg = format::signature_message(
            &header.segment_id,
            &header.prev_segment_hash,
            &trailer.merkle_root,
        );
        let signature = Signature::from_slice(&trailer.signature).map_err(|_| {
            StrataError::Corrupt("replica refused a trailer with a malformed signature".into())
        })?;
        if self.verifying.verify(&msg, &signature).is_err() {
            return Err(StrataError::Corrupt(
                "replica refused a trailer whose signature does not verify".into(),
            ));
        }
        let wire = borsh::to_vec(trailer).map_err(|err| {
            StrataError::Corrupt(format!("replica trailer does not serialize: {err}"))
        })?;
        if wire.len() != TRAILER_WIRE_SIZE {
            return Err(StrataError::Corrupt(
                "replica trailer is not the fixed wire size".into(),
            ));
        }
        let before = bytes.len() as u64;
        if let Err(err) = extend_file(&path, before, &wire) {
            let _ = truncate_to(&path, before);
            return Err(err);
        }
        Ok(())
    }

    /// Verify `bytes` as one segment, then write those exact bytes.
    ///
    /// A bad link, payload hash, or trailer signature returns before the
    /// segment file is created.
    pub fn ingest_segment(&mut self, bytes: &[u8]) -> Result<u64, StrataError> {
        let header = parse_header(bytes).ok_or_else(|| {
            StrataError::Corrupt("replica refused a segment whose header does not decode".into())
        })?;
        let (stop, state) = scan_frames(bytes, format::header_hash(&header));
        let frame_count = match &stop {
            TailStop::Clean => state.frame_count,
            TailStop::Trailer { trailer, .. } => {
                if let Err(reason) = check_trailer(trailer, &state, &header, &self.verifying) {
                    return Err(StrataError::Corrupt(format!(
                        "replica refused a segment trailer: {reason}"
                    )));
                }
                state.frame_count
            }
            TailStop::Torn { reason, .. } => {
                return Err(StrataError::Corrupt(format!(
                    "replica refused a damaged segment: {reason}"
                )));
            }
        };
        let segs = list_segments(&self.dir).unwrap_or_default();
        let expected_prev = match segs.last() {
            None => GENESIS_PREV_SEGMENT_HASH,
            Some((_, path)) => format::hash_slice(&fs::read(path)?),
        };
        if header.prev_segment_hash != expected_prev {
            return Err(StrataError::Corrupt(
                "replica refused a segment that does not extend the chain".into(),
            ));
        }
        if segs.last().is_some_and(|(_, path)| {
            fs::read(path)
                .ok()
                .is_some_and(|existing| !matches!(stop_of(&existing), TailStop::Trailer { .. }))
        }) {
            return Err(StrataError::Corrupt(
                "replica refused a segment while the active tail is unsealed".into(),
            ));
        }
        let no = segs.len() as u32;
        let path = self.dir.join(segment_filename(no, &header.segment_id));
        durable_create(&path, &self.dir, bytes)?;
        let acked = self.last_acked_seq()? + frame_count;
        if let Err(err) = write_head_state(&self.dir, acked) {
            let _ = fs::remove_file(&path);
            return Err(err.into());
        }
        Ok(frame_count)
    }

    fn active_unsealed(&self) -> Result<(PathBuf, Vec<u8>), StrataError> {
        let segs = list_segments(&self.dir)?;
        let Some((_, path)) = segs.into_iter().next_back() else {
            return Err(StrataError::Corrupt(
                "replica has no segment to append to".into(),
            ));
        };
        let bytes = fs::read(&path)?;
        if matches!(stop_of(&bytes), TailStop::Trailer { .. }) {
            return Err(StrataError::Corrupt(
                "replica active segment is sealed; begin a new segment first".into(),
            ));
        }
        if matches!(stop_of(&bytes), TailStop::Torn { .. }) {
            return Err(StrataError::Corrupt(
                "replica active segment is damaged".into(),
            ));
        }
        Ok((path, bytes))
    }

    /// Bring this replica up to the segment files in `src`.
    ///
    /// New segments are stored byte-for-byte. An unsealed tail that grew is
    /// extended one frame at a time, then sealed when the source file has a
    /// trailer. A source that is not an extension of the replica is reported
    /// as equivocation and no further segment is written.
    pub fn sync_from_dir(&mut self, src: &Path) -> Result<(), StrataError> {
        let sources = seg_files(src)?;
        let locals = list_segments(&self.dir).unwrap_or_default();
        if sources.len() < locals.len() {
            return Err(StrataError::Corrupt(
                "equivocation: source chain is shorter than the replica".into(),
            ));
        }
        for (index, (_, local_path)) in locals.iter().enumerate() {
            let local = fs::read(local_path)?;
            let incoming = fs::read(&sources[index])?;
            if incoming == local {
                continue;
            }
            let last = index + 1 == locals.len();
            let unsealed = !matches!(stop_of(&local), TailStop::Trailer { .. });
            if last && unsealed && incoming.starts_with(&local) {
                self.extend_unsealed(&incoming)?;
                continue;
            }
            return Err(StrataError::Corrupt(
                "equivocation: source segment diverges from the replica prefix".into(),
            ));
        }
        for source in sources.iter().skip(locals.len()) {
            let bytes = fs::read(source)?;
            self.ingest_segment(&bytes)?;
        }
        Ok(())
    }

    fn extend_unsealed(&mut self, incoming: &[u8]) -> Result<(), StrataError> {
        let header = parse_header(incoming).ok_or_else(|| {
            StrataError::Corrupt("replica extension header does not decode".into())
        })?;
        let (stop, state) = scan_frames(incoming, format::header_hash(&header));
        if matches!(stop, TailStop::Torn { .. }) {
            return Err(StrataError::Corrupt(
                "replica refused a damaged extension".into(),
            ));
        }
        let locals = list_segments(&self.dir).unwrap_or_default();
        let have = match locals.last() {
            Some((_, path)) => {
                let old = fs::read(path)?;
                let old_header = parse_header(&old).ok_or_else(|| {
                    StrataError::Corrupt("replica tail header does not decode".into())
                })?;
                scan_frames(&old, format::header_hash(&old_header)).1.frame_count
            }
            None => 0,
        };
        let mut off = HEADER_WIRE_SIZE;
        let mut index = 0u64;
        while off < state.end_offset {
            let (frame, used) = format::parse_frame(&incoming[off..]).map_err(|err| {
                StrataError::Corrupt(format!("replica extension frame does not decode: {err}"))
            })?;
            if index >= have {
                self.append_frame(&frame)?;
            }
            index += 1;
            off += used;
        }
        if let TailStop::Trailer { trailer, .. } = stop {
            self.append_trailer(&trailer)?;
        }
        Ok(())
    }

    /// Copy at most `max_new_frames` frames from `src`.
    ///
    /// Each appended frame is durable before this returns its seq. A kill
    /// after the returned ack leaves that frame on disk and leaves every
    /// later frame unwritten. A trailer is written only after its frames.
    pub fn sync_frames(&mut self, src: &Path, max_new_frames: u64) -> Result<ReplicaStep, StrataError> {
        if max_new_frames == 0 {
            return Ok(ReplicaStep {
                frames_added: 0,
                last_ack: None,
                caught_up: false,
            });
        }
        let sources = seg_files(src)?;
        let locals = list_segments(&self.dir).unwrap_or_default();
        if sources.len() < locals.len() {
            return Err(StrataError::Corrupt(
                "equivocation: source chain is shorter than the replica".into(),
            ));
        }
        let mut budget = max_new_frames;
        let mut added = 0u64;
        let mut last_ack = None;
        for (index, source) in sources.iter().enumerate() {
            let incoming = fs::read(source)?;
            let locals = list_segments(&self.dir).unwrap_or_default();
            if index < locals.len() {
                let local = fs::read(&locals[index].1)?;
                if incoming == local {
                    continue;
                }
                let last = index + 1 == locals.len();
                let unsealed = !matches!(stop_of(&local), TailStop::Trailer { .. });
                if !(last && unsealed && incoming.starts_with(&local)) {
                    return Err(StrataError::Corrupt(
                        "equivocation: source segment diverges from the replica prefix".into(),
                    ));
                }
            } else if index == locals.len() {
                let header = parse_header(&incoming).ok_or_else(|| {
                    StrataError::Corrupt("replica refused a segment whose header does not decode".into())
                })?;
                self.begin_segment(&header)?;
            } else {
                return Err(StrataError::Corrupt(
                    "equivocation: source chain skips a segment".into(),
                ));
            }
            let (n, ack, done) = self.extend_budget(&incoming, budget)?;
            added += n;
            budget -= n;
            if ack.is_some() {
                last_ack = ack;
            }
            if !done || budget == 0 {
                return Ok(ReplicaStep {
                    frames_added: added,
                    last_ack,
                    caught_up: false,
                });
            }
        }
        Ok(ReplicaStep {
            frames_added: added,
            last_ack,
            caught_up: self.matches_source(&sources)?,
        })
    }

    fn extend_budget(
        &mut self,
        incoming: &[u8],
        budget: u64,
    ) -> Result<(u64, Option<(u64, [u8; 32])>, bool), StrataError> {
        let header = parse_header(incoming).ok_or_else(|| {
            StrataError::Corrupt("replica extension header does not decode".into())
        })?;
        let (stop, state) = scan_frames(incoming, format::header_hash(&header));
        if matches!(stop, TailStop::Torn { .. }) {
            return Err(StrataError::Corrupt(
                "replica refused a damaged extension".into(),
            ));
        }
        let locals = list_segments(&self.dir).unwrap_or_default();
        let have = match locals.last() {
            Some((_, path)) => {
                let old = fs::read(path)?;
                let old_header = parse_header(&old).ok_or_else(|| {
                    StrataError::Corrupt("replica tail header does not decode".into())
                })?;
                scan_frames(&old, format::header_hash(&old_header)).1.frame_count
            }
            None => 0,
        };
        let mut off = HEADER_WIRE_SIZE;
        let mut index = 0u64;
        let mut added = 0u64;
        let mut last = None;
        while off < state.end_offset {
            let (frame, used) = format::parse_frame(&incoming[off..]).map_err(|err| {
                StrataError::Corrupt(format!("replica extension frame does not decode: {err}"))
            })?;
            if index >= have {
                if added >= budget {
                    return Ok((added, last, false));
                }
                let ack = self.append_frame(&frame)?;
                last = Some((ack.seq, ack.frame_hash));
                added += 1;
            }
            index += 1;
            off += used;
        }
        if let TailStop::Trailer { trailer, .. } = stop {
            let locals = list_segments(&self.dir).unwrap_or_default();
            let sealed = locals.last().is_some_and(|(_, path)| {
                fs::read(path)
                    .ok()
                    .is_some_and(|bytes| matches!(stop_of(&bytes), TailStop::Trailer { .. }))
            });
            if !sealed {
                self.append_trailer(&trailer)?;
            }
        }
        Ok((added, last, true))
    }

    fn matches_source(&self, sources: &[PathBuf]) -> Result<bool, StrataError> {
        let locals = list_segments(&self.dir).unwrap_or_default();
        if locals.len() != sources.len() {
            return Ok(false);
        }
        for (local, source) in locals.iter().zip(sources.iter()) {
            if fs::read(&local.1)? != fs::read(source)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

}

/// One paced replica step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaStep {
    /// Frames made durable by this call.
    pub frames_added: u64,
    /// Last frame this call acked, when it appended at least one.
    pub last_ack: Option<(u64, [u8; 32])>,
    /// The replica's segment bytes match `src`.
    pub caught_up: bool,
}

fn seg_files(dir: &Path) -> Result<Vec<PathBuf>, StrataError> {
    let mut files = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(files),
        Err(err) => return Err(err.into()),
    };
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("seg") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn stop_of(bytes: &[u8]) -> TailStop {
    match parse_header(bytes) {
        Some(header) => scan_frames(bytes, format::header_hash(&header)).0,
        None => TailStop::Torn {
            offset: 0,
            reason: "header".into(),
        },
    }
}

fn header_hash_of(bytes: &[u8]) -> Result<[u8; 32], StrataError> {
    let header = parse_header(bytes).ok_or_else(|| {
        StrataError::Corrupt("replica active segment header does not decode".into())
    })?;
    Ok(format::header_hash(&header))
}

fn check_trailer(
    trailer: &SegmentTrailer,
    state: &log::ScanState,
    header: &SegmentHeader,
    key: &VerifyingKey,
) -> Result<(), String> {
    if trailer.frame_count != state.frame_count {
        return Err(format!(
            "frame_count {} != scanned {}",
            trailer.frame_count, state.frame_count
        ));
    }
    let root = format::merkle_root(&state.leaves);
    if trailer.merkle_root != root {
        return Err("merkle root mismatch".into());
    }
    let msg = format::signature_message(
        &header.segment_id,
        &header.prev_segment_hash,
        &trailer.merkle_root,
    );
    let signature = Signature::from_slice(&trailer.signature)
        .map_err(|_| "malformed signature".to_string())?;
    key.verify(&msg, &signature)
        .map_err(|_| "signature invalid".to_string())
}

fn frames_on_disk(dir: &Path, key: &VerifyingKey) -> Result<u64, StrataError> {
    let segs = list_segments(dir).unwrap_or_default();
    let mut total = 0u64;
    for (_, path) in segs {
        let bytes = fs::read(path)?;
        let header = parse_header(&bytes).ok_or_else(|| {
            StrataError::Corrupt("replica segment header does not decode".into())
        })?;
        let (stop, state) = scan_frames(&bytes, format::header_hash(&header));
        if let TailStop::Trailer { trailer, .. } = &stop {
            check_trailer(trailer, &state, &header, key)
                .map_err(|reason| StrataError::Corrupt(format!("replica trailer: {reason}")))?;
        }
        if matches!(stop, TailStop::Torn { .. }) {
            return Err(StrataError::Corrupt(
                "replica segment is damaged".into(),
            ));
        }
        total += state.frame_count;
    }
    Ok(total)
}

fn read_frames_in(dir: &Path, key: &VerifyingKey) -> Result<Vec<FrameRecord>, StrataError> {
    refuse_signing_seed(dir)?;
    let segs = match list_segments(dir) {
        Ok(segs) => segs,
        Err(StrataError::Io(err)) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let acked = log::read_head_state(dir)?.unwrap_or(0);
    let mut out = Vec::new();
    let mut prev_file_hash = GENESIS_PREV_SEGMENT_HASH;
    for (idx, (_, path)) in segs.iter().enumerate() {
        let bytes = fs::read(path)?;
        let header = parse_header(&bytes).ok_or_else(|| {
            StrataError::Corrupt("replica segment header does not decode".into())
        })?;
        if header.prev_segment_hash != prev_file_hash {
            return Err(StrataError::Corrupt(
                "replica segment prev_segment_hash does not match the previous file".into(),
            ));
        }
        let (stop, state) = scan_frames(&bytes, format::header_hash(&header));
        let last = idx + 1 == segs.len();
        match &stop {
            TailStop::Trailer { trailer, .. } => {
                if let Err(reason) = check_trailer(trailer, &state, &header, key) {
                    return Err(StrataError::Corrupt(format!(
                        "replica trailer: {reason}"
                    )));
                }
            }
            TailStop::Clean if last => {}
            TailStop::Clean => {
                return Err(StrataError::Corrupt(
                    "replica sealed segment is missing its trailer".into(),
                ));
            }
            TailStop::Torn { reason, .. } => {
                return Err(StrataError::Corrupt(format!(
                    "replica segment is damaged: {reason}"
                )));
            }
        }
        // Re-walk frames for the records. scan_frames already proved them.
        let mut off = HEADER_WIRE_SIZE;
        for _ in 0..state.frame_count {
            let (frame, used) = format::parse_frame(&bytes[off..]).map_err(|err| {
                StrataError::Corrupt(format!("replica frame reread failed: {err}"))
            })?;
            let seq = out.len() as u64 + 1;
            let hashed = frame_hash(&frame);
            if acked == 0 || seq <= acked {
                out.push(FrameRecord {
                    seq,
                    kind: frame.kind,
                    payload: frame.payload,
                    payload_blake3: frame.payload_blake3,
                    prev_frame_hash: frame.prev_frame_hash,
                    frame_hash: hashed,
                });
            }
            off += used;
        }
        prev_file_hash = format::hash_slice(&bytes);
    }
    if acked > out.len() as u64 {
        return Err(StrataError::Halt(crate::error::HaltDetail {
            last_acked_seq: acked,
            segment: 0,
            offset: 0,
            reason: format!(
                "replica watermark {acked} exceeds the {} frames that verify",
                out.len()
            ),
        }));
    }
    if acked > 0 {
        out.truncate(acked as usize);
    }
    Ok(out)
}

/// Read a replica directory without holding it open.
pub fn read_replica_frames(
    dir: &Path,
    writer_verifying_key: &VerifyingKey,
) -> Result<Vec<FrameRecord>, StrataError> {
    read_frames_in(dir, writer_verifying_key)
}

fn refuse_signing_seed(dir: &Path) -> Result<(), StrataError> {
    if dir.join(KEY_FILE).exists() {
        return Err(StrataError::Corrupt(
            "replica refuses a directory that contains strata.key; the peer signing seed is not read"
                .into(),
        ));
    }
    Ok(())
}

fn pin_writer_key(dir: &Path, key: &VerifyingKey) -> Result<(), StrataError> {
    let path = dir.join(WRITER_PUB);
    let bytes = key.to_bytes();
    if path.exists() {
        let existing = fs::read(&path)?;
        if existing.as_slice() != bytes {
            return Err(StrataError::Corrupt(
                "replica writer.pub does not match the pinned verifying key".into(),
            ));
        }
        return Ok(());
    }
    durable_create(&path, dir, &bytes)
}

fn segment_filename(no: u32, id: &[u8; 16]) -> String {
    let hex: String = id.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("{no:08}-{hex}.seg")
}

fn durable_create(path: &Path, dir: &Path, bytes: &[u8]) -> Result<(), StrataError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    sync::sync_file(&file, SyncPurpose::Segment)?;
    sync::sync_dir(dir)?;
    let read_back = fs::read(path)?;
    if read_back.as_slice() != bytes {
        let _ = fs::remove_file(path);
        return Err(StrataError::Corrupt(
            "replica write did not read back".into(),
        ));
    }
    Ok(())
}

fn extend_file(path: &Path, offset: u64, bytes: &[u8]) -> Result<(), StrataError> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let end = file.seek(SeekFrom::End(0))?;
    if end != offset {
        return Err(StrataError::Corrupt(
            "replica append offset is not the end of the segment".into(),
        ));
    }
    file.write_all(bytes)?;
    sync::sync_file(&file, SyncPurpose::Segment)?;
    if let Some(parent) = path.parent() {
        sync::sync_dir(parent)?;
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut got = vec![0u8; bytes.len()];
    file.read_exact(&mut got)?;
    if got != bytes {
        return Err(StrataError::Corrupt(
            "replica frame did not read back".into(),
        ));
    }
    Ok(())
}

fn truncate_to(path: &Path, len: u64) -> Result<(), StrataError> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    sync::sync_file(&file, SyncPurpose::Segment)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{payload_blake3, SegmentTrailer};
    use crate::StrataLog;
    use ed25519_dalek::SigningKey;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "strata-replica-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn writer_with_frames(n: usize) -> (PathBuf, VerifyingKey) {
        let dir = temp_dir("writer");
        let log = StrataLog::open(&dir).unwrap();
        for i in 0..n {
            log.append(32, &[i as u8, 1, 2, 3]).unwrap();
        }
        log.seal().unwrap();
        let key = log.verifying_key();
        drop(log);
        (dir, key)
    }

    fn only_segment(dir: &Path) -> Vec<u8> {
        let mut segs: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
            .collect();
        segs.sort();
        fs::read(&segs[0]).unwrap()
    }

    #[test]
    fn replica_pins_the_public_key_and_never_creates_a_signing_seed() {
        let (src, verifying) = writer_with_frames(2);
        let dir = temp_dir("replica");
        let mut replica = StrataLog::open_replica(&dir, verifying).unwrap();
        let bytes = only_segment(&src);
        replica.ingest_segment(&bytes).unwrap();
        assert!(!dir.join("strata.key").exists());
        assert_eq!(
            fs::read(dir.join("writer.pub")).unwrap(),
            verifying.to_bytes()
        );
        assert_eq!(replica.read_frames().unwrap().len(), 2);
        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_link_bad_payload_hash_and_bad_trailer_write_nothing() {
        let (src, verifying) = writer_with_frames(2);
        let bytes = only_segment(&src);
        let dir = temp_dir("refuse");
        let mut replica = StrataLog::open_replica(&dir, verifying).unwrap();

        let header: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).unwrap();
        replica.begin_segment(&header).unwrap();
        let (frame, consumed) = format::parse_frame(&bytes[HEADER_WIRE_SIZE..]).unwrap();
        replica.append_frame(&frame).unwrap();
        let seg = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
            .unwrap();
        let len_after_good = fs::metadata(&seg).unwrap().len();

        let (mut bad_link, _) = format::parse_frame(&bytes[HEADER_WIRE_SIZE + consumed..]).unwrap();
        bad_link.prev_frame_hash[0] ^= 0x5a;
        let err = replica.append_frame(&bad_link).unwrap_err();
        assert!(err.to_string().contains("prev_frame_hash"), "{err}");
        assert_eq!(fs::metadata(&seg).unwrap().len(), len_after_good);

        let (mut bad_hash, _) =
            format::parse_frame(&bytes[HEADER_WIRE_SIZE + consumed..]).unwrap();
        bad_hash.payload_blake3 = payload_blake3(bad_hash.kind, b"nope");
        let err = replica.append_frame(&bad_hash).unwrap_err();
        assert!(err.to_string().contains("payload_blake3"), "{err}");
        assert_eq!(fs::metadata(&seg).unwrap().len(), len_after_good);

        let (good_second, _) = format::parse_frame(&bytes[HEADER_WIRE_SIZE + consumed..]).unwrap();
        replica.append_frame(&good_second).unwrap();
        let len_before_trailer = fs::metadata(&seg).unwrap().len();
        let mut trailer: SegmentTrailer =
            borsh::from_slice(&bytes[bytes.len() - TRAILER_WIRE_SIZE..]).unwrap();
        trailer.signature[0] ^= 0xff;
        let err = replica.append_trailer(&trailer).unwrap_err();
        assert!(err.to_string().contains("signature"), "{err}");
        assert_eq!(fs::metadata(&seg).unwrap().len(), len_before_trailer);
        assert_eq!(replica.read_frames().unwrap().len(), 2);

        let empty = temp_dir("refuse-whole");
        let mut fresh = StrataLog::open_replica(&empty, verifying).unwrap();
        let mut flipped = bytes.clone();
        let sig_at = flipped.len() - 1;
        flipped[sig_at] ^= 0xff;
        assert!(fresh.ingest_segment(&flipped).is_err());
        assert!(fs::read_dir(&empty).unwrap().all(|entry| {
            entry.unwrap().path().extension().and_then(|ext| ext.to_str()) != Some("seg")
        }));
        assert!(!empty.join(KEY_FILE).exists());

        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&empty);
    }

    #[test]
    fn open_replica_refuses_a_directory_that_already_holds_a_signing_seed() {
        let dir = temp_dir("poison");
        fs::write(dir.join(KEY_FILE), [7u8; 32]).unwrap();
        let mut seed = [0u8; 32];
        seed[0] = 9;
        let verifying = SigningKey::from_bytes(&seed).verifying_key();
        let err = match StrataLog::open_replica(&dir, verifying) {
            Ok(_) => panic!("open_replica accepted a directory that holds strata.key"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("strata.key"), "{err}");
        assert_eq!(fs::read(dir.join(KEY_FILE)).unwrap(), [7u8; 32]);
        let _ = fs::remove_dir_all(&dir);
    }
}
