//! # strata — the Causal Proof Engine's log layer
//!
//! STRATA is not a database. It is the Causal Proof Engine: it commits,
//! attests, and testifies — never serves queries. This crate is the
//! append-only, fail-stop foundation.
//!
//! Segment files (`NNNNNNNN-<uuid>.seg`) hold a borsh-encoded header, a chain
//! of hash-linked frames, and a signed trailer (see [`crate::format`]).
//!
//! ## Commit protocol (fail-stop, TigerBeetle-style)
//!
//! `WRITING -> SYNCING -> VERIFYING -> DURABLE -> ACK`
//!
//! * **WRITING**: frame bytes are appended at the end of the active segment.
//! * **SYNCING**: one durability barrier per commit group. Linux:
//!   `fdatasync`. macOS: `fcntl(F_FULLFSYNC)` — on APFS a plain `fsync` only
//!   drains the kernel page cache and does not force the drive to acknowledge
//!   the write to stable storage, so we pay for the full barrier.
//! * **VERIFYING**: every frame is pread back and its blake3 recomputed; any
//!   mismatch or I/O error triggers a deliberate `panic!` carrying a
//!   fail-stop report `{last_acked_seq, op, errno}`. No retry, no fallback.
//! * **DURABLE**: the `head.state` watermark (`last_acked_seq`) is replaced
//!   atomically (temp file + fsync + rename + directory fsync).
//! * **ACK**: [`SeqAck`] is returned only after verify + watermark.
//!
//! A full volume is the one I/O failure that is not fail-stop. If writing a
//! frame, the head watermark, or a seal trailer fails with a storage-full
//! error, nothing of the commit has been acked: the segment is cut back to
//! its state before the commit, every queued appender gets the error
//! ([`StrataError::is_storage_full`]), the seq space is unchanged, and the
//! log stays usable once space is freed. Sync, verify, and every other I/O
//! error stay fail-stop. Opening also leaves no half-written lock, key, or
//! segment file behind when space runs out.
//!
//! In production binaries, build with `panic = "abort"` so a fail-stop panic
//! takes the process down; the library uses `panic!` (not a hard `abort`) so
//! tests can exercise the path with `catch_unwind`.
//!
//! ## Group commit (v1)
//!
//! Frames queued within a <=2ms coalescing window or <=64 frames share one
//! sync. v1 is a synchronous leader/follower design behind one mutex plus one
//! condvar: the first appender becomes the batch leader, holds the window open
//! for more frames, then runs the whole pipeline while later appenders queue
//! and wait. There is no dedicated flusher thread yet (documented v1
//! simplification), so a lone append also pays the 2ms window. Batch-heavy
//! callers should use [`StrataLog::append_batch`].
//!
//! ## Recovery
//!
//! On open every segment is scanned: frame blake3s and chain links are
//! recomputed. A sealed segment's trailer is verified whether or not
//! `head.state` exists; any mismatch halts. Only the unsealed active tail
//! may be truncated, and only at a torn final write above the acked
//! watermark. Any other damage returns [`StrataError::Halt`] — history is
//! never truncated.
//! A fully-written, chain-valid frame above the watermark that survived the
//! crash is retained: its bytes are durable, only its ack was lost, and the
//! watermark is not advanced for it.
//!
//! ## Single writer
//!
//! Enforced by an OS advisory lock on `strata.lock`, held on an open file for
//! the life of the log and released by the kernel when the holder exits, so a
//! killed writer never leaves the directory locked and a recycled pid never
//! looks like a holder. The file records the holder's pid for diagnostics
//! only, and is never removed while a writer has the log open.

mod error;
mod format;
mod lockfile;
mod log;
mod mesh_head;
mod replica;
pub mod segment_lock;
mod sync;

#[cfg(test)]
mod tests;

pub use error::{HaltDetail, StrataError};
pub use format::{
    consistency_proof, frame_hash, header_hash, merkle_root, parse_frame, payload_blake3,
    signature_message, verify_consistency, Frame, FrameRecord, SegmentHeader, SegmentTrailer,
    FRAME_FIXED_WIRE_SIZE, GENESIS_PREV_SEGMENT_HASH, HEADER_WIRE_SIZE, SEGMENT_MAGIC,
    SEGMENT_VERSION, TRAILER_WIRE_SIZE,
};
pub use log::{
    read_head_state, HeadInfo, LogReport, SealInfo, SeqAck, StrataLog, TailReport, TrailerCheck,
    GROUP_COMMIT_WINDOW_MS, MAX_BATCH_FRAMES,
};
pub use mesh_head::{head_message, sign_head, verify_head, SignedHead, MESH_HEAD_DOMAIN};
pub use replica::{read_replica_frames, ReplicaLog, ReplicaStep};
pub use segment_lock::{
    seal_segment, unseal_segment_owner, verify_segment_seal, SealLevel, SealReceipt,
};
#[cfg(feature = "failpoints")]
pub use sync::failpoints;
