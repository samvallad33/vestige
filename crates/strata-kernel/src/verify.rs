//! Replay verifier: recompute everything, trust nothing.
//!
//! [`verify`] takes stored checkpoints and a stream of `(seq, event_hash,
//! event)` records and proves, in order:
//!
//! 1. every checkpoint has the right [`crate::checkpoint::MAGIC`] and a
//!    known algorithm version;
//! 2. checkpoint `log_seq`s are strictly increasing;
//! 3. the hash chain holds — `checkpoint_hash(c_{i-1}) == c_i.prev`
//!    (genesis chains from `[0; 32]`); with [`verify_with_head`], the head
//!    checkpoint's recomputed hash must equal the externally anchored
//!    `head_hash`;
//! 4. each record's borsh bytes hash to its stored `event_hash`, the
//!    record's seq field agrees with `event.seq()`, and seqs are strictly
//!    increasing;
//! 5. events fold under the version-dispatched kernel of the checkpoint
//!    segment they fall in (`prev.log_seq < seq <= log_seq` folds under
//!    THIS checkpoint's version — that is the upgrade path: a log may move
//!    to a new algorithm version at a checkpoint boundary);
//! 6. at each checkpoint's `log_seq`, the recomputed
//!    [`crate::checkpoint::state_root`] equals the stored one BIT FOR BIT;
//! 7. no trailing events exist past the final checkpoint.
//!
//! Every error names the offending `seq` (or checkpoint `log_seq` and
//! index). Verification allocates one fold state and performs no I/O.

use core::fmt;

use crate::checkpoint::{checkpoint_hash, state_root, Checkpoint, MAGIC};
use crate::kernel::{kernel_for, VersionedFold};
use crate::state::State;

/// Why a replay failed. Every variant names the offending seq (or the
/// checkpoint's `log_seq` plus its index in the slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Checkpoint has the wrong magic bytes (wrong format version).
    BadMagic {
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
    },
    /// Checkpoint names an algorithm version with no pinned kernel.
    UnknownAlgoVersion {
        /// The unknown version value.
        version: u32,
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
    },
    /// Checkpoint `log_seq`s are not strictly increasing.
    CheckpointOutOfOrder {
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
        /// The previous checkpoint's log position.
        prev_log_seq: u64,
    },
    /// Recomputed hash of the previous checkpoint does not equal this
    /// checkpoint's `prev_checkpoint` (chain tamper).
    ChainBreak {
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
    },
    /// The head checkpoint's recomputed hash differs from the externally
    /// anchored hash (only checked by [`verify_with_head`]).
    CheckpointHashMismatch {
        /// Index of the head checkpoint (last in the slice).
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
    },
    /// The record's seq field disagrees with `event.seq()`.
    EventSeqMismatch {
        /// The record's claimed seq.
        seq: u64,
    },
    /// `blake3(borsh(event))` differs from the stored per-event hash.
    EventHashMismatch {
        /// The offending record's seq.
        seq: u64,
    },
    /// Event seqs are not strictly increasing.
    OutOfOrderEvent {
        /// The offending record's seq.
        seq: u64,
        /// The previously applied seq.
        applied_seq: u64,
    },
    /// Recomputed state root differs from the checkpoint's stored root.
    StateRootMismatch {
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
        /// Root stored in the checkpoint.
        expected: [u8; 32],
        /// Root recomputed by replay.
        actual: [u8; 32],
    },
    /// Events remain after the final checkpoint was verified.
    TrailingEvents {
        /// Seq of the first surplus event.
        first_extra_seq: u64,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::BadMagic { index, log_seq } => {
                write!(f, "checkpoint[{index}] (log_seq {log_seq}): bad magic")
            }
            VerifyError::UnknownAlgoVersion {
                version,
                index,
                log_seq,
            } => {
                write!(
                    f,
                    "checkpoint[{index}] (log_seq {log_seq}): unknown algorithm version {version}"
                )
            }
            VerifyError::CheckpointOutOfOrder {
                index,
                log_seq,
                prev_log_seq,
            } => {
                write!(f, "checkpoint[{index}]: log_seq {log_seq} not greater than previous {prev_log_seq}")
            }
            VerifyError::ChainBreak { index, log_seq } => {
                write!(f, "checkpoint[{index}] (log_seq {log_seq}): prev_checkpoint does not match recomputed hash of predecessor")
            }
            VerifyError::CheckpointHashMismatch { index, log_seq } => {
                write!(f, "checkpoint[{index}] (head, log_seq {log_seq}): recomputed hash differs from anchored head hash")
            }
            VerifyError::EventSeqMismatch { seq } => {
                write!(f, "event seq {seq}: record seq disagrees with event.seq()")
            }
            VerifyError::EventHashMismatch { seq } => {
                write!(f, "event seq {seq}: borsh bytes do not hash to stored hash")
            }
            VerifyError::OutOfOrderEvent { seq, applied_seq } => {
                write!(
                    f,
                    "event seq {seq}: not strictly increasing (previous {applied_seq})"
                )
            }
            VerifyError::StateRootMismatch {
                index,
                log_seq,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "checkpoint[{index}] (log_seq {log_seq}): state root mismatch, stored {expected:02x?} != replayed {actual:02x?}"
                )
            }
            VerifyError::TrailingEvents { first_extra_seq } => {
                write!(
                    f,
                    "trailing event past final checkpoint, first extra seq {first_extra_seq}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VerifyError {}

/// Verify a checkpointed event log (no head anchor). See [`verify_with_head`].
pub fn verify<E: VersionedFold>(
    checkpoints: &[Checkpoint],
    events: impl Iterator<Item = (u64, [u8; 32], E)>,
) -> Result<(), VerifyError> {
    verify_with_head(checkpoints, None, events)
}

/// Verify a checkpointed event log, optionally pinning the head checkpoint's
/// hash to an externally stored anchor.
///
/// The chain links prove every non-head checkpoint's stored bytes are exact
/// (any field tamper changes the recomputed hash and breaks the next
/// link). The HEAD checkpoint has no successor, so without an anchor only
/// its `state_root` / magic / version claims are checked by replay; pass
/// `head_hash` — the `checkpoint_hash` value the log layer persists next to
/// the checkpoint — to make head tamper (e.g. raising `log_seq`) detectable
/// as [`VerifyError::CheckpointHashMismatch`].
pub fn verify_with_head<E: VersionedFold>(
    checkpoints: &[Checkpoint],
    head_hash: Option<[u8; 32]>,
    events: impl Iterator<Item = (u64, [u8; 32], E)>,
) -> Result<(), VerifyError> {
    let mut state = State::default();
    let mut prev_hash: [u8; 32] = [0; 32];
    let mut prev_log_seq: Option<u64> = None;
    let mut last_applied: Option<u64> = None;
    let mut pending: Option<(u64, [u8; 32], E)> = None;
    let mut events = events;

    for (index, cp) in checkpoints.iter().enumerate() {
        if cp.magic != MAGIC {
            return Err(VerifyError::BadMagic {
                index,
                log_seq: cp.log_seq,
            });
        }
        if let Some(pls) = prev_log_seq
            && cp.log_seq <= pls
        {
            return Err(VerifyError::CheckpointOutOfOrder {
                index,
                log_seq: cp.log_seq,
                prev_log_seq: pls,
            });
        }
        if cp.prev_checkpoint != prev_hash {
            return Err(VerifyError::ChainBreak {
                index,
                log_seq: cp.log_seq,
            });
        }
        // Fail fast on unknown versions before touching the event stream.
        kernel_for::<E>(cp.algo_version).map_err(|e| VerifyError::UnknownAlgoVersion {
            version: e.0,
            index,
            log_seq: cp.log_seq,
        })?;

        // Fold every event belonging to this checkpoint's segment.
        loop {
            let record = match pending.take() {
                Some(r) => r,
                None => match events.next() {
                    Some(r) => r,
                    None => break,
                },
            };
            if record.0 > cp.log_seq {
                pending = Some(record);
                break;
            }
            let (seq, stored_hash, event) = record;
            if event.seq() != seq {
                return Err(VerifyError::EventSeqMismatch { seq });
            }
            if let Some(applied) = last_applied
                && seq <= applied
            {
                return Err(VerifyError::OutOfOrderEvent {
                    seq,
                    applied_seq: applied,
                });
            }
            let bytes =
                borsh::to_vec(&event).expect("borsh serialize of StrataEvent is infallible");
            let computed: [u8; 32] = *blake3::hash(&bytes).as_bytes();
            if computed != stored_hash {
                return Err(VerifyError::EventHashMismatch { seq });
            }
            kernel_for::<E>(cp.algo_version).expect("version was checked above")(
                &mut state, &event,
            );
            last_applied = Some(seq);
        }

        let actual_root = state_root(&state);
        if actual_root != cp.state_root {
            return Err(VerifyError::StateRootMismatch {
                index,
                log_seq: cp.log_seq,
                expected: cp.state_root,
                actual: actual_root,
            });
        }
        prev_hash = checkpoint_hash(cp);
        prev_log_seq = Some(cp.log_seq);
    }

    if let Some(hash) = head_hash
        && let Some(cp) = checkpoints.last()
        && checkpoint_hash(cp) != hash
    {
        return Err(VerifyError::CheckpointHashMismatch {
            index: checkpoints.len() - 1,
            log_seq: cp.log_seq,
        });
    }

    if let Some((seq, _, _)) = pending {
        return Err(VerifyError::TrailingEvents {
            first_extra_seq: seq,
        });
    }
    if let Some(record) = events.next() {
        return Err(VerifyError::TrailingEvents {
            first_extra_seq: record.0,
        });
    }

    Ok(())
}
