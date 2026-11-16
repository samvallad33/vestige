#![forbid(unsafe_code)]

//! Deterministic 7-Field Strata Provenance Receipt
//!
//! Enables `causal_walk(action="prove")` to trace any symptom deterministically
//! back to its exact `writer_id`, `passport_digest`, `(session_id, turn_seq, prompt_hash)`,
//! `(git_head_sha, files)`, `(tool_call_digest, exit_code, error_sig)`, causal edge,
//! BLAKE3/Ed25519 signature (#404), and Operator `policy_verdict`.

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

use crate::chain_fold::{Hash32, SignedWriterFrame, WriterId};

/// Domain separation tag for canonical provenance payloads.
pub const DOMAIN_STRATA_PROVENANCE_V1: &[u8] = b"strata.provenance.receipt.v1\0";

/// Domain separation tag for missing provenance rejection receipts (#774).
pub const DOMAIN_MISSING_PROVENANCE_V1: &[u8] = b"strata.missing_provenance.v1\0";

/// Explicit causal relationship linking a receipt to its predecessor frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalEdge {
    /// Genesis or standalone root observation (`parent_frame_digest == [0u8; 32]`).
    Root,
    /// Derived from an earlier decision, edit anchor, or prompt turn.
    DerivedFrom,
    /// Empirical evidence (e.g., failing test output, edit hunk, stack trace) for a claim.
    EvidenceOf,
    /// Closes an active `OpenFailure` / bug symptom.
    Closes,
    /// Supersedes/retracts a prior decision or memory frame.
    Supersedes,
}

/// Touched file path and optional changed line hunk range (`[start_line, end_line]`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TouchedFileHunk {
    pub path: String,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
}

/// Operator policy gate outcome recorded at the moment of the action.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum PolicyVerdict {
    Allowed { rule_id: String },
    Stopped { rule_id: String, reason: String },
}

/// Kind of provenance event recorded into Strata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceEventKind {
    EditAnchor,
    CommitAnchor,
    OpenFailure,
    FixResolved,
    RememberSave,
    OperatorStop,
}

/// Deterministic rejection receipt emitted when a `remember` save has no session turn,
/// tool, file, or causal link anchor (#774).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MissingProvenanceReceipt {
    pub session_id: String,
    pub summary_digest: Hash32,
    pub reason: String,
    pub receipt_digest: Hash32,
}

impl MissingProvenanceReceipt {
    pub fn new(session_id: impl Into<String>, summary: &str, reason: impl Into<String>) -> Self {
        let session_id = session_id.into();
        let reason = reason.into();
        let summary_digest = *blake3::hash(summary.as_bytes()).as_bytes();

        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN_MISSING_PROVENANCE_V1);
        hasher.update(session_id.as_bytes());
        hasher.update(b"\0");
        hasher.update(&summary_digest);
        hasher.update(b"\0");
        hasher.update(reason.as_bytes());
        let receipt_digest = *hasher.finalize().as_bytes();

        Self {
            session_id,
            summary_digest,
            reason,
            receipt_digest,
        }
    }
}

/// Reason stored in every orphan quarantine body. The digest covers these bytes.
pub const ORPHAN_QUARANTINE_REASON: &str =
    "save rejected: no session turn, tool, file, or causal link anchor present";

const QUARANTINE_MARK: &str = "vestige.provenance.quarantine.v1";
const RELEASE_MARK: &str = "vestige.provenance.release.v1";

/// One orphan body recovered from the log, plus the summary those bytes carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineRecord {
    pub receipt: MissingProvenanceReceipt,
    pub summary: String,
}

/// An appended anchor frame. It names a quarantine receipt and does not replace it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRelease {
    pub receipt_digest: Hash32,
    pub anchor_digest: Hash32,
    pub summary: String,
}

/// Quarantine projection over log bodies. Released receipts stay in `retained`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceView {
    pub quarantined: Vec<QuarantineRecord>,
    pub released: Vec<ProvenanceRelease>,
    pub retained: Vec<QuarantineRecord>,
}

/// Lowercase hex. Used for receipt digests in the log body.
pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Decode 32 bytes of lowercase or uppercase hex.
pub fn decode_hex_32(text: &str) -> Option<Hash32> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    let bytes = text.as_bytes();
    for index in 0..32 {
        let hi = hex_nibble(bytes[index * 2])?;
        let lo = hex_nibble(bytes[index * 2 + 1])?;
        out[index] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn single_line(value: &str) -> bool {
    !value.contains('\n') && !value.contains('\r')
}

/// Canonical quarantine body and the receipt those bytes recompute to.
///
/// `session_id` is empty for an orphan with no session. `summary` is one line.
pub fn quarantine_body(
    session_id: &str,
    summary: &str,
) -> Result<(String, MissingProvenanceReceipt), &'static str> {
    if !single_line(session_id) || !single_line(summary) || summary.is_empty() {
        return Err("session and summary must be single lines, and summary must not be empty");
    }
    let receipt = MissingProvenanceReceipt::new(session_id, summary, ORPHAN_QUARANTINE_REASON);
    let body = format!("{QUARANTINE_MARK}\n{session_id}\n{ORPHAN_QUARANTINE_REASON}\n{summary}");
    Ok((body, receipt))
}

/// Canonical release body. The quarantine body is a different record and stays.
pub fn release_body(receipt_digest: Hash32, anchor_digest: Hash32) -> String {
    format!(
        "{RELEASE_MARK}\n{}\n{}",
        encode_hex(&receipt_digest),
        encode_hex(&anchor_digest)
    )
}

/// Parse one log body. Anything that is not a quarantine or release record is ignored.
pub fn parse_quarantine_body(body: &str) -> Option<QuarantineRecord> {
    let mut lines = body.split('\n');
    if lines.next()? != QUARANTINE_MARK {
        return None;
    }
    let session_id = lines.next()?;
    let reason = lines.next()?;
    let summary = lines.next()?;
    if lines.next().is_some() || reason != ORPHAN_QUARANTINE_REASON || summary.is_empty() {
        return None;
    }
    let receipt = MissingProvenanceReceipt::new(session_id, summary, reason);
    Some(QuarantineRecord {
        receipt,
        summary: summary.to_string(),
    })
}

fn parse_release_body(body: &str) -> Option<(Hash32, Hash32)> {
    let mut lines = body.split('\n');
    if lines.next()? != RELEASE_MARK {
        return None;
    }
    let receipt = decode_hex_32(lines.next()?)?;
    let anchor = decode_hex_32(lines.next()?)?;
    if lines.next().is_some() {
        return None;
    }
    Some((receipt, anchor))
}

/// Project quarantine state from log bodies.
///
/// A release frame removes a receipt from `quarantined` and leaves it in
/// `retained`. Bodies that are not provenance records are skipped.
pub fn project_provenance_bodies<S: AsRef<str>>(bodies: &[S]) -> ProvenanceView {
    let mut retained = Vec::new();
    let mut releases = Vec::new();
    for body in bodies {
        let body = body.as_ref();
        if let Some(record) = parse_quarantine_body(body) {
            retained.push(record);
        } else if let Some((receipt_digest, anchor_digest)) = parse_release_body(body) {
            releases.push((receipt_digest, anchor_digest));
        }
    }
    retained.sort_by(|left, right| {
        left.receipt
            .receipt_digest
            .cmp(&right.receipt.receipt_digest)
    });
    retained.dedup_by(|left, right| left.receipt.receipt_digest == right.receipt.receipt_digest);
    releases.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    releases.dedup();

    let mut released = Vec::new();
    for (receipt_digest, anchor_digest) in &releases {
        if let Some(record) = retained
            .iter()
            .find(|record| record.receipt.receipt_digest == *receipt_digest)
        {
            released.push(ProvenanceRelease {
                receipt_digest: *receipt_digest,
                anchor_digest: *anchor_digest,
                summary: record.summary.clone(),
            });
        }
    }
    let quarantined = retained
        .iter()
        .filter(|record| {
            !released
                .iter()
                .any(|release| release.receipt_digest == record.receipt.receipt_digest)
        })
        .cloned()
        .collect();
    ProvenanceView {
        quarantined,
        released,
        retained,
    }
}

/// Canonical 7-field Strata provenance receipt body (fields 1–5 & 7 stored in payload;
/// field 1 `writer_id` + field 6 `payload_digest`/`frame_digest`/`signature` live on `SignedWriterFrame`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrataProvenanceReceipt {
    pub writer_id: WriterId,
    /// BLAKE3 digest of the active Biscuit SubagentPassport (or `[0u8; 32]` for root owner).
    pub passport_digest: Hash32,

    pub session_id: String,
    pub turn_seq: u64,
    pub prompt_hash: Hash32,

    pub git_head_sha: String,
    pub files: Vec<TouchedFileHunk>,

    pub tool_call_digest: Hash32,
    pub exit_code: Option<i32>,
    pub error_sig: Option<String>,

    pub parent_frame_digest: Hash32,
    pub causal_edge: CausalEdge,

    pub policy_verdict: PolicyVerdict,

    pub kind: ProvenanceEventKind,
    pub summary: String,
}

impl StrataProvenanceReceipt {
    /// Computes the canonical `tool_call_digest` from `(tool_name, canonical_input)`.
    pub fn compute_tool_call_digest(tool_name: &str, normalized_input: &str) -> Hash32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"strata.tool_call.v1\0");
        hasher.update(tool_name.as_bytes());
        hasher.update(b"\0");
        hasher.update(normalized_input.as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Computes the canonical `prompt_hash` from a `UserPromptSubmit` prompt string.
    pub fn compute_prompt_hash(prompt_text: &str) -> Hash32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"strata.prompt.v1\0");
        hasher.update(prompt_text.as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Computes `passport_digest` from serialized Biscuit token bytes (`[0u8; 32]` if owner/root).
    pub fn compute_passport_digest(passport_bytes: Option<&[u8]>) -> Hash32 {
        match passport_bytes {
            Some(bytes) if !bytes.is_empty() => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"strata.passport.v1\0");
                hasher.update(bytes);
                *hasher.finalize().as_bytes()
            }
            _ => [0u8; 32],
        }
    }

    /// Rejects orphan receipts that lack a session turn, tool call, file anchor, and causal link (#774).
    pub fn require_anchor(&self) -> Result<(), MissingProvenanceReceipt> {
        let has_session = !self.session_id.trim().is_empty()
            && (self.turn_seq > 0 || self.prompt_hash != [0u8; 32]);
        let has_tool = self.tool_call_digest != [0u8; 32];
        let has_files = !self.files.is_empty();
        let has_link = self.parent_frame_digest != [0u8; 32];

        if has_session || has_tool || has_files || has_link {
            Ok(())
        } else {
            Err(MissingProvenanceReceipt::new(
                &self.session_id,
                &self.summary,
                ORPHAN_QUARANTINE_REASON,
            ))
        }
    }

    /// Serializes and signs this provenance receipt into a `#404` `SignedWriterFrame`.
    pub fn into_signed_frame(
        &self,
        signing_key: &SigningKey,
        seq: u64,
        lamport: u64,
        timestamp_ms: u64,
        prev_hash: Hash32,
    ) -> Result<SignedWriterFrame, serde_json::Error> {
        let writer_id = WriterId::from_bytes(signing_key.verifying_key().to_bytes());
        let payload = serde_json::to_vec(self)?;
        let payload_digest = SignedWriterFrame::compute_payload_digest(&payload);
        let frame_digest = SignedWriterFrame::compute_frame_digest(
            &writer_id,
            seq,
            lamport,
            timestamp_ms,
            &prev_hash,
            &payload_digest,
        );
        let sig = signing_key.sign(&frame_digest);

        Ok(SignedWriterFrame {
            writer_id,
            seq,
            lamport,
            timestamp_ms,
            prev_hash,
            payload_digest,
            frame_digest,
            signature: sig.to_bytes().to_vec(),
            payload,
        })
    }

    /// Decodes a `StrataProvenanceReceipt` from a `SignedWriterFrame`'s payload.
    pub fn from_signed_frame(frame: &SignedWriterFrame) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(&frame.payload)
    }
}

/// Deterministic backward provenance proof returned by `causal_walk(action="prove")`.
pub fn prove_causal_chain(
    frames: &[SignedWriterFrame],
    symptom_frame_digest: Hash32,
) -> Vec<(Hash32, StrataProvenanceReceipt)> {
    let mut by_digest: BTreeMap<Hash32, StrataProvenanceReceipt> = BTreeMap::new();
    for frame in frames {
        if frame.verify_cryptography().is_ok()
            && let Ok(receipt) = StrataProvenanceReceipt::from_signed_frame(frame)
        {
            by_digest.insert(frame.frame_digest, receipt);
        }
    }

    let mut chain = Vec::new();
    let mut visited = BTreeSet::new();
    let mut cursor = symptom_frame_digest;

    while visited.insert(cursor) {
        let Some(receipt) = by_digest.get(&cursor).cloned() else {
            break;
        };
        let parent = receipt.parent_frame_digest;
        chain.push((cursor, receipt));
        if parent == [0u8; 32] {
            break;
        }
        cursor = parent;
    }

    chain
}
