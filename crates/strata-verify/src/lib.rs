//! # strata-verify — the determinism/parity gate for the SQLite→STRATA flip
//!
//! `verify_store(dir)` recomputes EVERYTHING a strata store claims about
//! itself and never trusts a single byte of stored summary state:
//!
//! * **kernel side** — `kernel.log` (framed `(seq, event_hash, borsh event)`
//!   records), `kernel.checkpoints` (`borsh(Vec<Checkpoint>)`), and
//!   `kernel.head` (the 32-byte external anchor of the head checkpoint) are
//!   re-read as raw bytes; the checkpoint chain links, the head anchor, every
//!   per-event hash, seq monotonicity, segment versioning, and the state root
//!   at every checkpoint boundary are all RECOMPUTED via
//!   `strata_kernel::verify::verify_with_head` plus an independent explicit
//!   fold pass.
//! * **gate side** — `gate.log` (framed `(seq, kind, payload)` records),
//!   `gate.policy` (`borsh(Policy)`), and `gate.head` (the 32-byte frame-hash
//!   anchor of the last gate frame) are re-read as raw bytes; frame seqs must
//!   be dense from 0, the last frame hash must equal the anchor, every stored
//!   GATE verdict must be re-derived bit-for-bit under the stored policy
//!   (`strata_gate::rederive_verdicts`), and the structural sweep
//!   (`strata_gate::sweep`) must come back empty.
//!
//! This store layout is the on-disk contract the sibling `strata-store` crate
//! must emit; until that crate merges, the differential harness
//! (`tests/differential` in the vestige repo) materializes the layout
//! directly from the gate/kernel crates, and this crate verifies it.
//!
//! Every failure names the offending `seq` (or checkpoint `log_seq` + index)
//! in its `Display`, so a tampered byte is reported against the exact record
//! that tampering broke.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod layout;
mod live;
pub mod migration;
mod readonly;

use std::fmt;
use std::path::Path;
use std::time::Instant;

use layout::{
    decode_gate_frames, decode_kernel_records, gate_frame_hash, read_bytes_exact_32, store_paths,
};
use serde::Serialize;
use strata_gate::Policy;
use strata_gate::log::{EventLog, MemLog};
use strata_gate::record::{GateRecord, RecordKind};
use strata_gate::rederive_verdicts;
use strata_gate::sweep;
use strata_kernel::checkpoint::{Checkpoint, MAGIC, checkpoint_hash, state_root};
use strata_kernel::event::{ReviewEvent, StrataEvent};
use strata_kernel::kernel::kernel_for;
use strata_kernel::state::State;
use strata_kernel::verify::{VerifyError, verify_with_head};

/// Structural violation kinds reported by the gate sweep, mirrored from
/// `strata_gate::record::DutyKind` so the report stays plain data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GapKind {
    /// An EFFECT whose admission check over the window before it fails
    /// (e.g. forged with no admitting GATE).
    OrphanEffect,
    /// A PROPOSE context id with no producing WRITE effect in the prefix.
    ReadNoReceipt,
    /// A hole in the frame sequence counters.
    DutySeqGap,
}

impl fmt::Display for GapKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GapKind::OrphanEffect => f.write_str("orphan_effect"),
            GapKind::ReadNoReceipt => f.write_str("read_no_receipt"),
            GapKind::DutySeqGap => f.write_str("duty_seq_gap"),
        }
    }
}

/// Why verification failed. Every variant names the offending seq (or the
/// checkpoint's `log_seq` plus its index) so tampering is pinpointed.
///
/// Serialized as its `Display` string (which always carries the seq) so the
/// CLI JSON report stays stable across internal refactors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    /// A store file could not be read.
    Io {
        /// File name within the store dir (e.g. `kernel.log`).
        file: &'static str,
        /// Lower-level error text.
        error: String,
    },
    /// A store file could not be decoded (truncated frame, bad borsh, wrong
    /// anchor length). `seq` is the offending record when known.
    Malformed {
        /// File name within the store dir.
        file: &'static str,
        /// Offending record seq, when the failure is frame-local.
        seq: Option<u64>,
        /// What failed to decode.
        what: String,
    },
    /// The kernel replay verifier failed; names the seq (or checkpoint
    /// log_seq + index) per its own `Display`.
    Kernel(VerifyError),
    /// An explicit checkpoint-chain recomputation failed.
    CheckpointChain {
        /// Index of the offending checkpoint.
        index: usize,
        /// Its claimed log position.
        log_seq: u64,
        /// What failed.
        what: &'static str,
    },
    /// The recomputed kernel head hash differs from the `kernel.head` anchor.
    KernelHeadAnchorMismatch,
    /// The recomputed last gate frame hash differs from the `gate.head` anchor.
    GateHeadAnchorMismatch,
    /// Gate frame seqs are not dense from 0.
    GateSeqNotDense {
        /// Offending frame seq.
        seq: u64,
        /// Expected dense seq at that position.
        expected: u64,
    },
    /// A stored GATE verdict differs from the re-derived one under the stored
    /// policy.
    VerdictMismatch {
        /// The GATE record's seq.
        gate_seq: u64,
    },
    /// Rederivation itself failed (unknown proposal / malformed record).
    Rederive(String),
    /// The structural sweep flagged a record (e.g. a forged EFFECT with no
    /// admitting GATE before it in the log).
    StructuralGap {
        /// The violation kind.
        kind: GapKind,
        /// The offending record's seq.
        seq: u64,
    },
}

impl fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyFailure::Io { file, error } => write!(f, "{file}: read failed: {error}"),
            VerifyFailure::Malformed { file, seq, what } => match seq {
                Some(seq) => write!(f, "{file}: malformed record at seq {seq}: {what}"),
                None => write!(f, "{file}: malformed: {what}"),
            },
            VerifyFailure::Kernel(e) => write!(f, "kernel replay: {e}"),
            VerifyFailure::CheckpointChain {
                index,
                log_seq,
                what,
            } => write!(f, "checkpoint[{index}] (log_seq {log_seq}): {what}"),
            VerifyFailure::KernelHeadAnchorMismatch => {
                write!(
                    f,
                    "kernel.head anchor != recomputed hash of head checkpoint"
                )
            }
            VerifyFailure::GateHeadAnchorMismatch => {
                write!(f, "gate.head anchor != recomputed hash of last gate frame")
            }
            VerifyFailure::GateSeqNotDense { seq, expected } => {
                write!(f, "gate log seq {seq}: not dense (expected {expected})")
            }
            VerifyFailure::VerdictMismatch { gate_seq } => write!(
                f,
                "GATE at seq {gate_seq}: stored verdict differs from re-derived verdict"
            ),
            VerifyFailure::Rederive(e) => write!(f, "rederive failed: {e}"),
            VerifyFailure::StructuralGap { kind, seq } => {
                write!(f, "gate log seq {seq}: structural violation {kind}")
            }
        }
    }
}

impl Serialize for VerifyFailure {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

/// The store verification report. Every boolean is the result of a
/// recomputation from raw bytes in this call — nothing is read from stored
/// "ok" flags or summaries.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    /// `true` when the kernel replay verifier passes end to end (event
    /// hashes, ordering, segment versions, no trailing events past the head
    /// checkpoint) AND the head checkpoint hash matches the `kernel.head`
    /// anchor AND the gate log frames are dense with the last frame hash
    /// matching the `gate.head` anchor. This is the strata-layer
    /// `verify_tail` concern.
    pub log_tail_ok: bool,
    /// `true` when the checkpoint chain recomputes cleanly: correct magic,
    /// known algorithm versions, strictly increasing `log_seq`s, and every
    /// `prev_checkpoint` equal to the recomputed hash of its predecessor.
    pub checkpoint_chain_ok: bool,
    /// `true` when the independent replay fold's `state_root` equals the
    /// stored root at EVERY checkpoint boundary, bit for bit.
    pub state_root_matches: bool,
    /// `true` when every stored GATE verdict is re-derived bit-for-bit under
    /// the stored policy over the same log prefix.
    pub gate_verdicts_rederived: bool,
    /// Structural violations found by the gate sweep (empty on a clean log).
    pub gaps: Vec<GapKind>,
    /// Wall time spent inside `verify_store`, milliseconds.
    pub duration_ms: u64,
    /// Every failure, each naming the offending seq / checkpoint position.
    /// Empty iff [`VerifyReport::ok`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<VerifyFailure>,
}

impl VerifyReport {
    /// All checks passed: every recomputed boolean is `true`, the sweep found
    /// no gaps, and no failure was recorded.
    pub fn ok(&self) -> bool {
        self.log_tail_ok
            && self.checkpoint_chain_ok
            && self.state_root_matches
            && self.gate_verdicts_rederived
            && self.gaps.is_empty()
            && self.failures.is_empty()
    }
}

/// Verify the strata store rooted at `dir` (see [`layout`] for the file
/// contract). Recomputes everything; never trusts stored summaries.
pub fn verify_store(dir: &Path) -> VerifyReport {
    let started = Instant::now();
    let mut failures: Vec<VerifyFailure> = Vec::new();
    let paths = store_paths(dir);

    // ---- Read + decode every artifact from raw bytes. ----------------------
    let kernel_bytes = match std::fs::read(&paths.kernel_log) {
        Ok(b) => b,
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "kernel.log",
                error: e.to_string(),
            });
            Vec::new()
        }
    };
    let kernel_records = match decode_kernel_records(&kernel_bytes) {
        Ok(records) => records,
        Err((seq, what)) => {
            failures.push(VerifyFailure::Malformed {
                file: "kernel.log",
                seq,
                what,
            });
            Vec::new()
        }
    };

    let checkpoint_bytes = match std::fs::read(&paths.kernel_checkpoints) {
        Ok(b) => b,
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "kernel.checkpoints",
                error: e.to_string(),
            });
            Vec::new()
        }
    };
    let checkpoints: Vec<Checkpoint> = match borsh::from_slice(&checkpoint_bytes) {
        Ok(cps) => cps,
        Err(e) => {
            failures.push(VerifyFailure::Malformed {
                file: "kernel.checkpoints",
                seq: None,
                what: format!("borsh Vec<Checkpoint>: {e}"),
            });
            Vec::new()
        }
    };

    let kernel_head = match read_bytes_exact_32(&paths.kernel_head) {
        Ok(h) => Some(h),
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "kernel.head",
                error: e,
            });
            None
        }
    };

    let gate_bytes = match std::fs::read(&paths.gate_log) {
        Ok(b) => b,
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "gate.log",
                error: e.to_string(),
            });
            Vec::new()
        }
    };
    let gate_frames = match decode_gate_frames(&gate_bytes) {
        Ok(frames) => frames,
        Err((seq, what)) => {
            failures.push(VerifyFailure::Malformed {
                file: "gate.log",
                seq,
                what,
            });
            Vec::new()
        }
    };

    let policy_bytes = match std::fs::read(&paths.gate_policy) {
        Ok(b) => b,
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "gate.policy",
                error: e.to_string(),
            });
            Vec::new()
        }
    };
    let policy: Policy = match borsh::from_slice(&policy_bytes) {
        Ok(p) => p,
        Err(e) => {
            failures.push(VerifyFailure::Malformed {
                file: "gate.policy",
                what: format!("borsh Policy: {e}"),
                seq: None,
            });
            Policy::default()
        }
    };

    let gate_head = match read_bytes_exact_32(&paths.gate_head) {
        Ok(h) => Some(h),
        Err(e) => {
            failures.push(VerifyFailure::Io {
                file: "gate.head",
                error: e,
            });
            None
        }
    };

    // ---- Checkpoint chain: recomputed from the checkpoint bytes alone. -----
    let mut checkpoint_chain_ok = !checkpoints.is_empty();
    if checkpoint_chain_ok {
        let mut prev_hash: [u8; 32] = [0; 32];
        let mut prev_log_seq: Option<u64> = None;
        for (index, cp) in checkpoints.iter().enumerate() {
            let broken = |what: &'static str| VerifyFailure::CheckpointChain {
                index,
                log_seq: cp.log_seq,
                what,
            };
            if cp.magic != MAGIC {
                failures.push(broken("bad magic"));
                checkpoint_chain_ok = false;
            }
            if strata_kernel::fsrs::weights_for(cp.algo_version).is_err() {
                failures.push(broken("unknown algorithm version"));
                checkpoint_chain_ok = false;
            }
            if let Some(pls) = prev_log_seq
                && cp.log_seq <= pls
            {
                failures.push(broken("log_seq not strictly increasing"));
                checkpoint_chain_ok = false;
            }
            if cp.prev_checkpoint != prev_hash {
                failures.push(broken("prev_checkpoint != recomputed hash of predecessor"));
                checkpoint_chain_ok = false;
            }
            prev_hash = checkpoint_hash(cp);
            prev_log_seq = Some(cp.log_seq);
        }
        // Head anchor: the externally stored hash of the head checkpoint.
        if let Some(anchor) = kernel_head {
            let recomputed = checkpoints.last().map(checkpoint_hash).unwrap_or([0; 32]);
            if recomputed != anchor {
                failures.push(VerifyFailure::KernelHeadAnchorMismatch);
                checkpoint_chain_ok = false;
            }
        } else {
            checkpoint_chain_ok = false;
        }
    } else if checkpoints.is_empty() {
        failures.push(VerifyFailure::Malformed {
            file: "kernel.checkpoints",
            seq: None,
            what: "no checkpoints (at least a genesis checkpoint is required)".to_string(),
        });
    }

    // ---- Independent replay fold: state roots at every boundary. -----------
    let mut state_root_matches = !checkpoints.is_empty() && !kernel_records.is_empty();
    {
        let mut state = State::default();
        let mut record_iter = kernel_records.iter().peekable();
        let mut last_applied: Option<u64> = None;
        for (index, cp) in checkpoints.iter().enumerate() {
            // Segment membership mirrors the kernel verifier: events with
            // prev.log_seq < seq <= log_seq fold under THIS checkpoint's
            // version.
            let Ok(apply) = kernel_for::<ReviewEvent>(cp.algo_version) else {
                // Already reported by the chain pass / kernel verify below.
                state_root_matches = false;
                continue;
            };
            while let Some(record) = record_iter.peek() {
                if record.seq > cp.log_seq {
                    break;
                }
                let seq = record.seq;
                let stored_hash = record.event_hash;
                let payload = record.payload.clone();
                let event: ReviewEvent = match borsh::from_slice::<ReviewEvent>(&payload) {
                    Ok(e) if e.seq() == seq => e,
                    Ok(e) => {
                        failures.push(VerifyFailure::Malformed {
                            file: "kernel.log",
                            seq: Some(seq),
                            what: format!("event.seq() {} != frame seq", e.seq()),
                        });
                        state_root_matches = false;
                        record_iter.next();
                        continue;
                    }
                    Err(e) => {
                        failures.push(VerifyFailure::Malformed {
                            file: "kernel.log",
                            seq: Some(seq),
                            what: format!("borsh ReviewEvent: {e}"),
                        });
                        state_root_matches = false;
                        record_iter.next();
                        continue;
                    }
                };
                let computed: [u8; 32] = *blake3::hash(&payload).as_bytes();
                if computed != stored_hash {
                    // The kernel verifier below also reports this with the
                    // seq; record it here too since this pass is independent.
                    failures.push(VerifyFailure::Kernel(VerifyError::EventHashMismatch {
                        seq,
                    }));
                    state_root_matches = false;
                }
                if let Some(prev) = last_applied
                    && seq <= prev
                {
                    failures.push(VerifyFailure::Kernel(VerifyError::OutOfOrderEvent {
                        seq,
                        applied_seq: prev,
                    }));
                    state_root_matches = false;
                }
                apply(&mut state, &event);
                last_applied = Some(seq);
                record_iter.next();
            }
            let actual = state_root(&state);
            if actual != cp.state_root {
                failures.push(VerifyFailure::Kernel(VerifyError::StateRootMismatch {
                    index,
                    log_seq: cp.log_seq,
                    expected: cp.state_root,
                    actual,
                }));
                state_root_matches = false;
            }
        }
        if record_iter.peek().is_some() {
            let seq = record_iter.peek().map(|r| r.seq).unwrap_or(0);
            failures.push(VerifyFailure::Kernel(VerifyError::TrailingEvents {
                first_extra_seq: seq,
            }));
            state_root_matches = false;
        }
    }
    if kernel_records.is_empty() {
        // An empty kernel log is only legal with a lone genesis checkpoint
        // whose root is the empty-state root.
        if let Some(cp) = checkpoints.first() {
            let empty_ok = cp.log_seq == 0 && cp.state_root == state_root(&State::default());
            if !empty_ok {
                failures.push(VerifyFailure::CheckpointChain {
                    index: 0,
                    log_seq: cp.log_seq,
                    what: "empty log but genesis checkpoint is not the empty-state root",
                });
                state_root_matches = false;
            } else {
                state_root_matches = true;
            }
        }
    }

    // ---- Canonical kernel verify (the reference recomputation). ------------
    let events = kernel_records.iter().map(|record| {
        let event: ReviewEvent = borsh::from_slice(&record.payload).unwrap_or(ReviewEvent {
            card_id: 0,
            rating: 0,
            event_seq: 0,
        });
        (record.seq, record.event_hash, event)
    });
    let kernel_ok = match verify_with_head(&checkpoints, kernel_head, events) {
        Ok(()) => true,
        Err(e) => {
            // Dedup: this pass may re-report what the independent passes
            // already recorded; keep one copy of each distinct failure.
            if !failures
                .iter()
                .any(|f| f == &VerifyFailure::Kernel(e.clone()))
            {
                failures.push(VerifyFailure::Kernel(e.clone()));
            }
            false
        }
    };

    // ---- Gate log: dense frames + head anchor. -----------------------------
    let mut gate_tail_ok = true;
    {
        let mut expected: u64 = 0;
        for frame in &gate_frames {
            if frame.seq != expected {
                failures.push(VerifyFailure::GateSeqNotDense {
                    seq: frame.seq,
                    expected,
                });
                gate_tail_ok = false;
            }
            expected = frame.seq.saturating_add(1);
        }
        if let Some(anchor) = gate_head {
            let recomputed = gate_frames
                .last()
                .map(|f| gate_frame_hash(f.seq, f.kind, &f.payload))
                .unwrap_or(*blake3::hash(b"").as_bytes());
            if recomputed != anchor {
                failures.push(VerifyFailure::GateHeadAnchorMismatch);
                gate_tail_ok = false;
            }
        } else {
            gate_tail_ok = false;
        }
    }

    // ---- Gate verdict rederivation under the stored policy. ----------------
    let mut gate_verdicts_rederived = true;
    {
        // Rebuild the log view with the SAME frame-hash recipe the writer
        // used (dense seqs reproduce identical acks).
        let mut view = MemLog::new();
        for frame in &gate_frames {
            view.append(frame.kind, frame.payload.clone());
        }
        let stored: Vec<(u64, strata_gate::Verdict)> = gate_frames
            .iter()
            .filter(|f| f.kind == RecordKind::Gate)
            .filter_map(|f| {
                let record: GateRecord = borsh::from_slice(&f.payload).ok()?;
                Some((f.seq, record.verdict))
            })
            .collect();
        match rederive_verdicts(&view, &policy) {
            Ok(rederived) => {
                if rederived != stored {
                    let mut rd = rederived.clone();
                    let mut st = stored.clone();
                    rd.sort_by_key(|(seq, _)| *seq);
                    st.sort_by_key(|(seq, _)| *seq);
                    for (r, s) in rd.iter().zip(st.iter()) {
                        if r != s {
                            failures.push(VerifyFailure::VerdictMismatch { gate_seq: r.0 });
                            gate_verdicts_rederived = false;
                        }
                    }
                    if rd.len() != st.len() {
                        gate_verdicts_rederived = false;
                    }
                }
            }
            Err(e) => {
                failures.push(VerifyFailure::Rederive(e.to_string()));
                gate_verdicts_rederived = false;
            }
        }
    }

    // ---- Structural sweep. --------------------------------------------------
    let mut gaps: Vec<GapKind> = Vec::new();
    {
        let mut view = MemLog::new();
        for frame in &gate_frames {
            view.append(frame.kind, frame.payload.clone());
        }
        for gap in sweep(&view) {
            let (kind, seq) = match &gap.detail {
                strata_gate::GapDetail::OrphanEffect { effect_seq, .. } => {
                    (GapKind::OrphanEffect, *effect_seq)
                }
                strata_gate::GapDetail::ReadNoReceipt { reader_seq, .. } => {
                    (GapKind::ReadNoReceipt, *reader_seq)
                }
                strata_gate::GapDetail::DutySeqGap {
                    expected, found, ..
                } => {
                    failures.push(VerifyFailure::GateSeqNotDense {
                        seq: *found,
                        expected: *expected,
                    });
                    (GapKind::DutySeqGap, *found)
                }
            };
            failures.push(VerifyFailure::StructuralGap { kind, seq });
            if !gaps.contains(&kind) {
                gaps.push(kind);
            }
        }
    }

    let log_tail_ok = kernel_ok && gate_tail_ok;

    VerifyReport {
        log_tail_ok,
        checkpoint_chain_ok,
        state_root_matches,
        gate_verdicts_rederived,
        gaps,
        duration_ms: started.elapsed().as_millis() as u64,
        failures,
    }
}

/// Recompute-and-compare helper for callers holding already-decoded artifacts
/// (used by the differential harness to double-check a store it just wrote).
pub fn verify_decoded(
    checkpoints: &[Checkpoint],
    kernel_head: [u8; 32],
    records: &[(u64, [u8; 32], Vec<u8>)],
) -> Result<(), VerifyError> {
    let events = records.iter().map(|(seq, hash, payload)| {
        let event: ReviewEvent = borsh::from_slice(payload).expect("frame payload decodes");
        (*seq, *hash, event)
    });
    verify_with_head(checkpoints, Some(kernel_head), events)
}

/// Re-export so CLI/tests can mirror the writer's framing without depending
/// on the layout module path explicitly.
pub use layout::{
    FRAME_LEN_BYTES as VERIFY_FRAME_LEN_BYTES, GateFrame, KernelRecord,
    MAGIC_BYTES as VERIFY_STORE_MAGIC, StoreFiles, StorePaths,
};

/// One verification of a directory. `json` is the pretty-printed report.
/// The check never creates or modifies a file under `dir`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathReport {
    /// True when every check for the detected layout passed.
    pub ok: bool,
    /// Pretty JSON of the layout-specific report.
    pub json: String,
    /// Human-readable failures. Empty when `ok` is true.
    pub failures: Vec<String>,
}

/// Verify `dir` without writing.
///
/// Layout detection, in order:
/// * segment files in `dir` — a migrated log when a receipt frame is
///   present, otherwise a raw strata log (chain only);
/// * `log/*.seg` — a live strata-store (`store.meta` only after a seal);
/// * otherwise the kernel.log / gate.log layout.
///
/// A successful migrated-log check does not continue into the kernel
/// layout. A missing path is a failure and is not created.
pub fn verify_path(dir: &Path) -> PathReport {
    if !dir.exists() {
        return path_failure(format!("path does not exist: {}", dir.display()));
    }
    if readonly::dir_has_segments(dir) {
        return verify_segment_dir(dir);
    }
    if live::is_live_store(dir) {
        let report = live::verify_live_store(dir);
        let failures = report.failures.clone();
        return path_from(report.ok, &report, failures);
    }
    let report = verify_store(dir);
    let failures = report.failures.iter().map(|f| f.to_string()).collect();
    path_from(report.ok(), &report, failures)
}

fn verify_segment_dir(dir: &Path) -> PathReport {
    let scan = match readonly::scan_log(dir) {
        Ok(scan) => scan,
        Err(err) => return path_failure(err),
    };
    let has_receipt = scan
        .frames
        .iter()
        .any(|frame| frame.kind == strata_migrate::records::KIND_MIGRATION_RECEIPT);
    if has_receipt {
        return match migration::verify_migrated_log(dir) {
            Ok(report) => {
                let failures = report.failures.clone();
                let ok = report.ok;
                path_from(ok, &report, failures)
            }
            Err(err) => path_failure(err),
        };
    }
    let report = live::LiveVerifyReport {
        ok: true,
        frames_total: scan.frames.len() as u64,
        segments: scan.segments,
        failures: Vec::new(),
    };
    path_from(true, &report, Vec::new())
}

fn path_failure(err: String) -> PathReport {
    let body = serde_json::json!({ "ok": false, "failures": [err.clone()] });
    PathReport {
        ok: false,
        json: serde_json::to_string_pretty(&body).expect("failure report serializes"),
        failures: vec![err],
    }
}

fn path_from(ok: bool, body: &impl serde::Serialize, failures: Vec<String>) -> PathReport {
    PathReport {
        ok,
        json: serde_json::to_string_pretty(body).expect("verify report serializes"),
        failures,
    }
}
