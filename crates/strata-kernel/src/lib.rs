//! # strata-kernel — the Causal Proof Engine's determinism core
//!
//! Integers-only hashed state, pinned kernels, bit-for-bit replay.
//!
//! The Strata determinism kernel: a pure, I/O-free fold of review events into
//! hashed state, plus the checkpoint and replay-verification machinery that
//! makes the fold's output bit-for-bit reproducible.
//!
//! ## Determinism contract
//!
//! * All state is canonical: [`borsh`] everywhere, integer fields only
//!   (`u8`/`u32`/`u64`/`i64`, `[u8; N]`, `Vec<T>`, `Option<T>`, field-only
//!   enums, maps are `BTreeMap`). No floats and no `String`-keyed maps ever
//!   enter hashed state — see [`lint_state_type!`] and
//!   [`canonical::HashableState`].
//! * Real-valued model quantities (stability, difficulty) live in state ONLY
//!   as quantized `i64` in Q32.32 fixed point, produced by the documented
//!   round-ties-to-even formula in [`canonical::to_q32_32`].
//! * Transcendental math (`exp`, `pow`, `rint`) goes through the [`libm`]
//!   crate — the pinned musl software port — never the host libm, so results
//!   are identical on every platform and toolchain. IEEE-754 basic operations
//!   (`+ - * /`) used here are exactly specified and rustc performs no FMA
//!   contraction by default.
//! * Time is SEQ: events carry a monotonically increasing `u64` sequence
//!   number. The kernel never reads a wall clock.
//! * Folding is dispatched by `ALGORITHM_VERSION`
//!   ([`kernel::kernel_for`]); old logs replay under the old constants
//!   forever. Changing any weight or formula is a NEW version, never an edit.
//!
//! ## no_std status
//!
//! Built with `std` for now (allowed by scope); the code restricts itself to
//! `core`/`alloc` plus `borsh`/`blake3`/`libm`, and all float math already
//! lives behind `libm`, so a future `no_std` port only needs feature-flag
//! flips on the I/O-free dependencies.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod canonical;
pub mod checkpoint;
pub mod event;
pub mod fsrs;
pub mod kernel;
pub mod state;
pub mod verify;

/// Everything the parallel strata crates (log, gate) need to code against.
pub mod prelude {
    pub use crate::canonical::{from_q32_32, lint_roundtrip, to_q32_32, HashableState, Q_ONE};
    pub use crate::checkpoint::{checkpoint_hash, state_root, Checkpoint, MAGIC};
    pub use crate::event::{ReviewEvent, StrataEvent};
    pub use crate::fsrs::{
        fold_v3, fold_v4, is_known_algo_version, weights_for, CardPhase, CardState, FsrsFold,
        ALGO_V1, ALGO_V2, ALGO_V3, ALGO_V4, FSRS6_OFFICIAL_WEIGHTS, FSRS7_WEIGHTS,
        V1_WEIGHTS_MILLI, V2_WEIGHTS_MILLI,
    };
    pub use crate::kernel::{
        kernel_for, ApplyFn, Kernel, KernelTableEntry, UnknownAlgoVersion, VersionedFold,
    };
    pub use crate::state::State;
    pub use crate::verify::{verify, verify_with_head, VerifyError};
}
