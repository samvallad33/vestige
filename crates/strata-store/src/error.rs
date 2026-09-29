//! Store error surface. One enum, manual `Display`/`Error` impls — the crate
//! keeps its dependency footprint at borsh + blake3 + the three strata
//! siblings.

use strata::StrataError;
use strata_kernel::verify::VerifyError;

/// Every fallible `StrataStore` operation returns this.
#[derive(Debug)]
pub enum StoreError {
    /// The underlying log refused the operation (lock, metadata, halt).
    Log(StrataError),
    /// A gate-record decode/lookup failed mid-write.
    Gate(String),
    /// The pinned policy evaluated the proposal to `Deny`.
    Denied {
        /// Gate seq of the rejecting GATE record.
        propose_seq: u64,
    },
    /// The pinned policy evaluated the proposal to `Hold` (a RETIRE the
    /// admission context did not authorize).
    Held {
        /// Gate seq of the holding GATE record.
        propose_seq: u64,
    },
    /// `GateRuntime::commit_effect` admission rejected the effect.
    Rejected(String),
    /// Caller-supplied data is invalid (empty content, bad edge type, ...).
    InvalidInput(String),
    /// Referenced node id does not exist.
    NotFound(String),
    /// Checkpoint chain / replay verification failed on open or seal.
    Verify(String),
    /// borsh encode/decode failure (should be unreachable for store types).
    Encode(String),
    /// Filesystem error outside the log's own fail-stop path.
    Io(std::io::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Log(e) => write!(f, "log error: {e}"),
            StoreError::Gate(e) => write!(f, "gate error: {e}"),
            StoreError::Denied { propose_seq } => {
                write!(f, "gate denied proposal at seq {propose_seq}")
            }
            StoreError::Held { propose_seq } => {
                write!(
                    f,
                    "gate held proposal at seq {propose_seq} (destructive action)"
                )
            }
            StoreError::Rejected(e) => write!(f, "effect rejected at admission: {e}"),
            StoreError::InvalidInput(e) => write!(f, "invalid input: {e}"),
            StoreError::NotFound(e) => write!(f, "not found: {e}"),
            StoreError::Verify(e) => write!(f, "verification failed: {e}"),
            StoreError::Encode(e) => write!(f, "encoding error: {e}"),
            StoreError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Log(e) => Some(e),
            StoreError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StrataError> for StoreError {
    fn from(e: StrataError) -> Self {
        StoreError::Log(e)
    }
}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

impl From<VerifyError> for StoreError {
    fn from(e: VerifyError) -> Self {
        StoreError::Verify(e.to_string())
    }
}
