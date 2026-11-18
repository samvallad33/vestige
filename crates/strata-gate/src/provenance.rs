//! Wave 14M (#774) Ingest Provenance & Causal-Anchor Gate +
//! Security Fortress (#836 Vector 5) Monotonic Causal Clock Guard.
//!
//! Ensures:
//! 1. Every admitted `remember` / `SignedWriterFrame` write carries a verifiable
//!    [`BirthCertificate`] linking it to a writer, session, and at least one
//!    causal anchor (explicit `links`/`files`, or auto-bound from active
//!    session/Reflex receipts).
//! 2. Writes with zero resolvable anchors are quarantined with a deterministic
//!    [`MissingProvenanceReceipt`] so Retroactive Salience Backfill (RSB) and
//!    `causal_walk` never encounter silent orphans.
//! 3. Session timestamps (`session_ts`) enforce strict monotonicity `T_{n+1} >= T_n`;
//!    any backward clock jump is rejected as [`ClockSpoofAttempt`].

use borsh::{BorshDeserialize, BorshSerialize};

/// A deterministic causal anchor binding a memory write to its origin.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum CausalAnchor {
    /// Prior memory or symptom handle (e.g. `mem-...`).
    MemoryId(String),
    /// Touched source file or symbol anchor (e.g. `src/auth.rs#verify_token`).
    FilePath(String),
    /// Active turn/tool/error receipt hash from Reflex or session state.
    ToolReceiptHash([u8; 32]),
    /// Exact git commit SHA (full 40-char or >=8-char unique prefix).
    GitCommitSha(String),
}

/// Mandatory Birth Certificate attached to every admitted memory write.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct BirthCertificate {
    pub writer_id: String,
    pub passport_digest: Option<[u8; 32]>,
    pub session_id: String,
    pub turn_receipt_hash: Option<[u8; 32]>,
    pub session_ts: u64,
    pub anchors: Vec<CausalAnchor>,
    pub auto_bound: bool,
}

/// Deterministic quarantine receipt emitted when a write has zero causal anchors.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct MissingProvenanceReceipt {
    pub code: String,
    pub reason: String,
    pub writer_id: String,
    pub session_id: String,
    pub attempted_ts: u64,
    pub quarantined: bool,
}

/// Tamper receipt emitted when a backward clock jump (`T_{n+1} < T_n`) is detected.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ClockSpoofAttempt {
    pub prev_ts: u64,
    pub attempted_ts: u64,
    pub session_id: String,
}

impl core::fmt::Display for ClockSpoofAttempt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ClockSpoofAttempt in session '{}': attempted_ts {} < prev_ts {}",
            self.session_id, self.attempted_ts, self.prev_ts
        )
    }
}

impl std::error::Error for ClockSpoofAttempt {}

/// Result of provenance gate evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceOutcome {
    Admitted(BirthCertificate),
    Quarantined(MissingProvenanceReceipt),
    ClockSpoofRejected(ClockSpoofAttempt),
}

/// Input bundle passed to the provenance gate on write.
#[derive(Debug, Clone)]
pub struct ProvenanceInput {
    pub writer_id: String,
    pub passport_digest: Option<[u8; 32]>,
    pub session_id: String,
    pub prev_session_ts: Option<u64>,
    pub session_ts: u64,
    pub explicit_links: Vec<String>,
    pub explicit_files: Vec<String>,
    pub active_tool_receipt: Option<[u8; 32]>,
}

/// Enforce monotonic causal clock invariant (`T_{n+1} >= T_n`).
pub fn check_monotonic_ts(
    session_id: &str,
    prev_ts: Option<u64>,
    attempted_ts: u64,
) -> Result<(), ClockSpoofAttempt> {
    if let Some(prev) = prev_ts
        && attempted_ts < prev
    {
        return Err(ClockSpoofAttempt {
            prev_ts: prev,
            attempted_ts,
            session_id: session_id.to_string(),
        });
    }
    Ok(())
}

/// Verify explicit causal anchors or auto-bind from active session/Reflex receipt.
///
/// Rejects backward clock jumps with `ClockSpoofRejected` and quarantines
/// completely unanchored writes with `MissingProvenanceReceipt`.
pub fn verify_or_autobind(input: ProvenanceInput) -> ProvenanceOutcome {
    if let Err(spoof) = check_monotonic_ts(
        &input.session_id,
        input.prev_session_ts,
        input.session_ts,
    ) {
        return ProvenanceOutcome::ClockSpoofRejected(spoof);
    }

    let mut anchors = Vec::new();
    for link in input.explicit_links {
        let trimmed = link.trim();
        if !trimmed.is_empty() {
            anchors.push(CausalAnchor::MemoryId(trimmed.to_string()));
        }
    }
    for file in input.explicit_files {
        let trimmed = file.trim();
        if !trimmed.is_empty() {
            anchors.push(CausalAnchor::FilePath(trimmed.to_string()));
        }
    }

    let mut auto_bound = false;
    if anchors.is_empty()
        && let Some(receipt_hash) = input.active_tool_receipt
    {
        anchors.push(CausalAnchor::ToolReceiptHash(receipt_hash));
        auto_bound = true;
    }

    if anchors.is_empty() {
        return ProvenanceOutcome::Quarantined(MissingProvenanceReceipt {
            code: "MissingProvenanceReceipt".to_string(),
            reason: "no explicit links, files, or active session tool receipt available to anchor memory".to_string(),
            writer_id: input.writer_id,
            session_id: input.session_id,
            attempted_ts: input.session_ts,
            quarantined: true,
        });
    }

    ProvenanceOutcome::Admitted(BirthCertificate {
        writer_id: input.writer_id,
        passport_digest: input.passport_digest,
        session_id: input.session_id,
        turn_receipt_hash: input.active_tool_receipt,
        session_ts: input.session_ts,
        anchors,
        auto_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_spoof_rejected_on_backward_ts() {
        let outcome = verify_or_autobind(ProvenanceInput {
            writer_id: "agent-1".to_string(),
            passport_digest: None,
            session_id: "sess-1".to_string(),
            prev_session_ts: Some(500),
            session_ts: 499,
            explicit_links: vec!["mem-001".to_string()],
            explicit_files: vec![],
            active_tool_receipt: None,
        });
        assert!(matches!(
            outcome,
            ProvenanceOutcome::ClockSpoofRejected(ClockSpoofAttempt {
                prev_ts: 500,
                attempted_ts: 499,
                ..
            })
        ));
    }

    #[test]
    fn autobinds_active_tool_receipt_when_links_omitted() {
        let hash = [0xAB; 32];
        let outcome = verify_or_autobind(ProvenanceInput {
            writer_id: "agent-1".to_string(),
            passport_digest: None,
            session_id: "sess-1".to_string(),
            prev_session_ts: Some(100),
            session_ts: 105,
            explicit_links: vec![],
            explicit_files: vec![],
            active_tool_receipt: Some(hash),
        });
        match outcome {
            ProvenanceOutcome::Admitted(cert) => {
                assert!(cert.auto_bound);
                assert_eq!(cert.anchors, vec![CausalAnchor::ToolReceiptHash(hash)]);
            }
            other => panic!("expected Admitted, got {other:?}"),
        }
    }

    #[test]
    fn quarantines_unanchored_orphan_with_missing_provenance_receipt() {
        let outcome = verify_or_autobind(ProvenanceInput {
            writer_id: "agent-1".to_string(),
            passport_digest: None,
            session_id: "sess-1".to_string(),
            prev_session_ts: Some(100),
            session_ts: 100,
            explicit_links: vec![],
            explicit_files: vec![],
            active_tool_receipt: None,
        });
        match outcome {
            ProvenanceOutcome::Quarantined(receipt) => {
                assert_eq!(receipt.code, "MissingProvenanceReceipt");
                assert!(receipt.quarantined);
            }
            other => panic!("expected Quarantined, got {other:?}"),
        }
    }
}
