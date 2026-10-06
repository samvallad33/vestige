//! Record kinds and borsh payloads. Integers only — no strings, no floats,
//! no clocks. Byte-exact layouts are documented on each type below.

use borsh::{BorshDeserialize, BorshSerialize};

/// Record kind codes as they appear on the wire (the `kind` byte of a frame
/// and the first byte of every `GAP`-adjacent dispatch).
///
/// 1..=7 by design: 0 stays reserved as "unknown".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    Propose,
    Gate,
    Effect,
    Gap,
    LessonAlarm,
    Canary,
    Alert,
}

impl RecordKind {
    pub const fn to_u8(self) -> u8 {
        match self {
            RecordKind::Propose => 1,
            RecordKind::Gate => 2,
            RecordKind::Effect => 3,
            RecordKind::Gap => 4,
            RecordKind::LessonAlarm => 5,
            RecordKind::Canary => 6,
            RecordKind::Alert => 7,
        }
    }

    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(RecordKind::Propose),
            2 => Some(RecordKind::Gate),
            3 => Some(RecordKind::Effect),
            4 => Some(RecordKind::Gap),
            5 => Some(RecordKind::LessonAlarm),
            6 => Some(RecordKind::Canary),
            7 => Some(RecordKind::Alert),
            _ => None,
        }
    }
}

/// `PROPOSE.action_kind` codes.
pub mod action_kind {
    pub const WRITE: u8 = 0;
    pub const RETIRE: u8 = 1;
    pub const GRANT: u8 = 2;
    pub const EFFECT: u8 = 3;
}

/// Alias so integrations can spell `action_kind::WRITE` or
/// `ActionKindCode::WRITE` interchangeably.
pub use action_kind as ActionKindCode;

/// Gate verdict. Borsh wire codes by declaration order: Allow=0, Deny=1,
/// Hold=2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum Verdict {
    Allow,
    Deny,
    Hold,
}

/// `PROPOSE` — the model proposes; nothing changes yet.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct Propose {
    pub action_hash: [u8; 32],
    /// One of [`action_kind`]: WRITE / RETIRE / GRANT / EFFECT.
    pub action_kind: u8,
    pub params_hash: [u8; 32],
    /// Referenced memory/edge ids (fact ids are WRITE-effect seqs).
    pub context: Vec<u64>,
}

/// `GATE` — the deterministic checker's decision, pinned to the policy and
/// the signed inputs it was made over.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct GateRecord {
    pub propose_seq: u64,
    pub verdict: Verdict,
    /// blake3(borsh(policy)) of the policy that produced this verdict.
    pub policy_hash: [u8; 32],
    pub inputs: GateInputs,
}

/// `EFFECT` — a landed change. Only reachable through
/// `GateRuntime::commit_effect`, which requires an admitting GATE earlier in
/// the log.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct EffectRecord {
    pub propose_seq: u64,
    /// The GATE seq that justifies this effect. Must be the latest gate for
    /// the proposal at append time.
    pub gate_seq: u64,
    pub action_hash: [u8; 32],
    pub payload_digest: [u8; 32],
}

/// `LESSON_ALARM` — a lesson whose retention fell below the forget floor.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct LessonAlarmRecord {
    pub propose_seq: u64,
    pub lesson_id: u64,
    pub retention_milli: i64,
}

/// `CANARY` — a planted canary id. Any later record referencing the id trips
/// an `ALERT` and auto-holds descendants.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct CanaryRecord {
    pub canary_id: u64,
}

/// `ALERT` — a canary was referenced by the record at `reader_seq`.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct AlertRecord {
    pub canary_id: u64,
    pub reader_seq: u64,
}

/// `GAP.duty` — which structural duty was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum DutyKind {
    /// Effect with no admit-able gate before it.
    OrphanEffect,
    /// A read (PROPOSE context id) referencing a fact with no producing
    /// WRITE effect in the prefix.
    ReadNoReceipt,
    /// A hole in the per-source duty sequence counters.
    DutySeqGap,
}

/// `GAP.detail` — typed, borsh.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum GapDetail {
    OrphanEffect {
        effect_seq: u64,
        propose_seq: u64,
        /// `Rejected::code()` of the failed admission check.
        reason: u8,
    },
    ReadNoReceipt {
        reader_seq: u64,
        dangling_id: u64,
    },
    DutySeqGap {
        /// Duty counter owner. Source 0 = the single append path.
        source: u64,
        expected: u64,
        found: u64,
    },
}

/// `GAP` — structural violation found by [`crate::sweep`].
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct GapRecord {
    pub duty: DutyKind,
    pub detail: GapDetail,
}

use crate::inputs::GateInputs;

/// One frame of the event log (in-memory view; the payload is the borsh of
/// the kind-specific record).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateEvent {
    pub seq: u64,
    pub kind: RecordKind,
    pub payload: Vec<u8>,
}

/// Errors raised while reading the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    /// No PROPOSE record exists at the given seq.
    UnknownProposal { seq: u64 },
    /// Payload failed to decode as the kind-specific record.
    MalformedRecord { seq: u64 },
}

impl core::fmt::Display for GateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GateError::UnknownProposal { seq } => write!(f, "no PROPOSE at seq {seq}"),
            GateError::MalformedRecord { seq } => write!(f, "malformed record at seq {seq}"),
        }
    }
}

impl std::error::Error for GateError {}

impl GateEvent {
    /// Decode the payload as the kind-specific record `T`.
    pub fn decode<T: BorshDeserialize>(&self) -> Result<T, GateError> {
        T::try_from_slice(&self.payload).map_err(|_| GateError::MalformedRecord { seq: self.seq })
    }

    /// Typed PROPOSE, or `None` if this frame is not a well-formed PROPOSE.
    pub fn propose(&self) -> Option<Propose> {
        (self.kind == RecordKind::Propose)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn gate(&self) -> Option<GateRecord> {
        (self.kind == RecordKind::Gate)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn effect(&self) -> Option<EffectRecord> {
        (self.kind == RecordKind::Effect)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn lesson_alarm(&self) -> Option<LessonAlarmRecord> {
        (self.kind == RecordKind::LessonAlarm)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn canary(&self) -> Option<CanaryRecord> {
        (self.kind == RecordKind::Canary)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn alert(&self) -> Option<AlertRecord> {
        (self.kind == RecordKind::Alert)
            .then(|| self.decode().ok())
            .flatten()
    }

    pub fn gap(&self) -> Option<GapRecord> {
        (self.kind == RecordKind::Gap)
            .then(|| self.decode().ok())
            .flatten()
    }
}
