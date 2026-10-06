//! # strata-gate — the Causal Proof Engine's admission runtime
//!
//! No verdict, no write. The security layer between thinking and doing.
//!
//! The centerpiece invariant: **the model proposes, the deterministic checker
//! decides, change+verdict land as one signed record; a write with no
//! approving verdict before it in the log is REJECTED at admission.**
//!
//! Pure functions over event streams. No clocks, no floats, no environment
//! access, no randomness — every output is a pure function of the log prefix
//! and the pinned policy.
//!
//! Module map:
//! - [`log`] — integration shims (`SeqAck`, `EventLog`) plus an in-memory
//!   `MemLog` used for tests and wiring examples.
//! - [`record`] — borsh record payloads (`PROPOSE`, `GATE`, `EFFECT`, `GAP`,
//!   `LESSON_ALARM`, `CANARY`, `ALERT`) and kind codes.
//! - [`inputs`] — [`GateInputs`] and [`compute_inputs`]: the signed gate
//!   context derived purely from the event-log prefix.
//! - [`policy`] — the deterministic policy VM: [`Policy`], [`Rule`],
//!   [`evaluate`], first-match-wins, default Deny.
//! - [`admit`] — the admission rule: [`admit`] with enum'd rejection reasons.
//! - [`runtime`] — [`GateRuntime`], the only append path for effects
//!   ([`GateRuntime::commit_effect`]).
//! - [`rederive`] — [`rederive_verdicts`]: replay and re-evaluate every GATE.
//! - [`sweep`] — [`sweep`]: emit `GAP` records for structural violations.
//!
//! Byte-exact layouts, enum codes, and integration decisions are documented in
//! the byte-exact layouts in `record.rs`.

pub mod admit;
pub mod inputs;
pub mod log;
pub mod policy;
pub mod record;
pub mod rederive;
pub mod runtime;
pub mod sweep;
#[cfg(test)]
mod tests;

pub use admit::{Rejected, admit};
pub use inputs::{BlastRadius, FORGET_FLOOR_MILLI, GateInputs, compute_inputs};
pub use log::{EventLog, MemLog, SeqAck};
pub use policy::{
    ANY_KIND, Policy, Rule, Veto, WILDCARD_PREFIX, evaluate, evaluate_detailed, evaluate_for,
    gate_verdict, policy_hash,
};
pub use record::{
    ActionKindCode, AlertRecord, CanaryRecord, DutyKind, EffectRecord, GapDetail, GapRecord,
    GateError, GateEvent, GateRecord, LessonAlarmRecord, Propose, RecordKind, Verdict, action_kind,
};
pub use rederive::{RederiveError, rederive_verdicts};
pub use runtime::GateRuntime;
pub use sweep::sweep;
