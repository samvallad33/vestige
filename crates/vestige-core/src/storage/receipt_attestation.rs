//! Privacy-safe, cryptographically verifiable decision-receipt attestations.
//!
//! The mutable [`crate::trace::Receipt`] payload is deliberately not signed:
//! purge redaction rewrites that payload. Instead, this module signs an
//! immutable, identity-free predicate containing receipt-local evidence slots
//! and salted commitments. The deletable [`DisclosureMapping`] that connects a
//! slot to a memory id is kept outside the signed envelope.
//!
//! A valid envelope proves that trusted key material signed the exact canonical
//! predicate bytes. A valid linked chain proves continuity of the observed
//! entries. Neither proves memory truth, causal correctness, completeness, nor
//! rollback resistance without a previously exported or externally anchored
//! checkpoint.

use std::collections::HashSet;
use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
use chrono::{DateTime, SecondsFormat, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use uuid::{Uuid, Version};

/// RFC 8785/JCS schema identifier for the immutable payload.
pub const RECEIPT_ATTESTATION_SCHEMA_V1: &str = "urn:vestige:receipt-attestation:v1";
/// DSSE payload type signed by this module.
pub const RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1: &str =
    "application/vnd.vestige.receipt+json;version=1";
/// Signature algorithm used by v1 attestations.
pub const RECEIPT_ATTESTATION_SIGNATURE_ALGORITHM_V1: &str = "ed25519";
/// Current immutable payload schema version.
pub const RECEIPT_ATTESTATION_VERSION_V1: u32 = 1;

const PAYLOAD_DIGEST_CONTEXT: &str = "vestige.receipt.payload.v1";
const DECISION_DIGEST_CONTEXT: &str = "vestige.receipt.decision.v1";
const ENTRY_DIGEST_CONTEXT: &str = "vestige.receipt.entry.v1";
const DISCLOSURE_COMMITMENT_CONTEXT: &str = "vestige.receipt.disclosure.v1";
const PUBLIC_KEY_FINGERPRINT_CONTEXT: &str = "vestige.receipt.ed25519-key.v1";
const DISCLOSURE_NONCE_CONTEXT: &str = "vestige.receipt.disclosure-nonce.v1";

/// Defensive application limits. These are deliberately below the protocol's
/// theoretical limits so untrusted envelopes cannot force unbounded work.
pub const MAX_DSSE_PAYLOAD_BYTES: usize = 1_048_576;
pub const MAX_DSSE_SIGNATURES: usize = 64;
pub const MAX_TRUSTED_SIGNING_KEYS: usize = 1_024;
pub const MAX_ATTESTATION_EVIDENCE: usize = 1_024;
pub const MAX_CLOSED_CODE_BYTES: usize = 128;
pub const MAX_PRODUCER_FIELD_BYTES: usize = 128;
pub const MAX_PRIVATE_IDENTIFIER_BYTES: usize = 1_024;

/// Largest integer exactly representable by every RFC 8785 implementation.
///
/// Chain sequences are serialized as decimal strings anyway, but retaining the
/// I-JSON bound prevents downstream consumers from accidentally coercing a
/// larger value through an IEEE-754 number.
pub const MAX_SAFE_CHAIN_SEQUENCE: u64 = 9_007_199_254_740_991;

macro_rules! opaque_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            fn random() -> Self {
                Self(format!("{}_{}", $prefix, Uuid::new_v4().simple()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            fn validate(&self, field: &'static str) -> Result<(), AttestationError> {
                validate_opaque_v4_id(field, &self.0, $prefix)
            }
        }
    };
}

opaque_id!(OpaqueReceiptId, "ratt");
opaque_id!(OpaqueRunId, "run");
opaque_id!(OpaqueChainId, "chain");
opaque_id!(OpaqueEvidenceSlot, "slot");

/// Closed decision kinds supported by the immutable v1 schema.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptPredicateKind {
    SynapticCapture,
    CounterfactualReplayInfluence,
    VerifiedLocalDisclosureErasure,
}

/// Fixed epistemic boundaries. Callers cannot inject a stronger free-form
/// claim into immutable signed bytes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptClaimBoundary {
    /// Exact signed inputs and output are attested; truth and causality are not.
    DecisionEvidenceNotTruthOrCausality,
    /// Only influence in a controlled replay is measured.
    ControlledReplayInfluenceNotRealWorldCausality,
    /// Only verified Vestige-controlled local disclosure erasure is covered.
    VerifiedVestigeLocalErasureOnly,
}

impl ReceiptPredicateKind {
    const fn operation_kind(self) -> &'static str {
        match self {
            Self::SynapticCapture => "synaptic_capture",
            Self::CounterfactualReplayInfluence => "counterfactual_replay_influence",
            Self::VerifiedLocalDisclosureErasure => "verified_local_disclosure_erasure",
        }
    }

    const fn claim_boundary(self) -> ReceiptClaimBoundary {
        match self {
            Self::SynapticCapture => ReceiptClaimBoundary::DecisionEvidenceNotTruthOrCausality,
            Self::CounterfactualReplayInfluence => {
                ReceiptClaimBoundary::ControlledReplayInfluenceNotRealWorldCausality
            }
            Self::VerifiedLocalDisclosureErasure => {
                ReceiptClaimBoundary::VerifiedVestigeLocalErasureOnly
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureDirection {
    Backward,
    Forward,
}

/// Redaction-safe, identifier-free projections accepted by the v1 builder.
/// Stable memory, user, session, event, and receipt identifiers have no field
/// in this type and therefore cannot accidentally enter `decisionDigest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RedactionSafeDecisionProjectionV1 {
    SynapticCapture {
        direction: CaptureDirection,
        evaluated_count: u32,
        captured_count: u32,
        withheld_count: u32,
    },
    CounterfactualReplayInfluence {
        baseline_count: u32,
        counterfactual_count: u32,
        membership_changed: bool,
        ordering_changed: bool,
        decision_changed: bool,
        withheld_slot_count: u32,
    },
    VerifiedLocalDisclosureErasure {
        generation: u64,
    },
}

impl RedactionSafeDecisionProjectionV1 {
    pub const fn kind(&self) -> ReceiptPredicateKind {
        match self {
            Self::SynapticCapture { .. } => ReceiptPredicateKind::SynapticCapture,
            Self::CounterfactualReplayInfluence { .. } => {
                ReceiptPredicateKind::CounterfactualReplayInfluence
            }
            Self::VerifiedLocalDisclosureErasure { .. } => {
                ReceiptPredicateKind::VerifiedLocalDisclosureErasure
            }
        }
    }

    fn digest(&self) -> Result<String, AttestationError> {
        if let Self::VerifiedLocalDisclosureErasure { generation } = self
            && (*generation == 0 || *generation > MAX_SAFE_CHAIN_SEQUENCE)
        {
            return Err(AttestationError::InvalidProjection(
                "erasure generation must be a positive I-JSON-safe integer",
            ));
        }
        let canonical = serde_json_canonicalizer::to_vec(self)?;
        Ok(decision_digest(&canonical))
    }
}

/// A new chain mints a random chain id; successors can only be built from a
/// chain entry produced by successful signing or verification.
#[derive(Debug, Clone, Copy)]
pub enum AttestationChainPosition<'a> {
    Genesis,
    Successor(&'a ChainEntry),
}

/// Result of privacy-safe construction. Disclosures are deletable and never
/// serialized inside the signed attestation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedReceiptAttestation {
    attestation: ReceiptAttestationV1,
    disclosures: Vec<DisclosureMapping>,
}

impl PreparedReceiptAttestation {
    pub fn attestation(&self) -> &ReceiptAttestationV1 {
        &self.attestation
    }

    pub fn disclosures(&self) -> &[DisclosureMapping] {
        &self.disclosures
    }

    pub fn into_parts(self) -> (ReceiptAttestationV1, Vec<DisclosureMapping>) {
        (self.attestation, self.disclosures)
    }
}

/// Immutable signed predicate for one decision receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReceiptAttestationV1 {
    /// Must equal [`RECEIPT_ATTESTATION_SCHEMA_V1`].
    schema: String,
    /// Must equal [`RECEIPT_ATTESTATION_VERSION_V1`].
    schema_version: u32,
    /// Algorithm expected for the DSSE signature in this schema version.
    signature_algorithm: String,
    /// Random receipt-local identifier. It must not encode a memory or user id.
    receipt_id: OpaqueReceiptId,
    /// Normalized UTC time claimed by the signer.
    #[serde(with = "normalized_utc")]
    issued_at: DateTime<Utc>,
    /// Software/build identity that produced the decision.
    producer: ProducerIdentity,
    /// Opaque operation/run identity. Callers must not place user ids here.
    operation: OperationIdentity,
    /// Append-only chain location authenticated inside the signed payload.
    chain: ChainLink,
    /// Typed, privacy-safe decision predicate.
    predicate: ReceiptPredicate,
}

impl ReceiptAttestationV1 {
    /// Construct an identity-free attestation and fresh per-slot disclosure
    /// nonces from the operating system CSPRNG used by UUID v4 generation.
    pub fn build<I, S>(
        issued_at: DateTime<Utc>,
        producer: ProducerIdentity,
        chain_position: AttestationChainPosition<'_>,
        algorithm_version: impl Into<String>,
        projection: RedactionSafeDecisionProjectionV1,
        private_memory_ids: I,
    ) -> Result<PreparedReceiptAttestation, AttestationError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        producer.validate()?;
        let algorithm_version = algorithm_version.into();
        validate_closed_code("predicate.algorithmVersion", &algorithm_version)?;
        let private_memory_ids: Vec<String> =
            private_memory_ids.into_iter().map(Into::into).collect();
        if private_memory_ids.len() > MAX_ATTESTATION_EVIDENCE {
            return Err(AttestationError::TooMuchEvidence(private_memory_ids.len()));
        }
        for memory_id in &private_memory_ids {
            require_bounded_nonempty("privateMemoryId", memory_id, MAX_PRIVATE_IDENTIFIER_BYTES)?;
        }

        let receipt_id = OpaqueReceiptId::random();
        let (chain_id, sequence, previous_entry_digest) = match chain_position {
            AttestationChainPosition::Genesis => (OpaqueChainId::random(), 0, None),
            AttestationChainPosition::Successor(previous) => {
                let sequence = previous
                    .sequence
                    .checked_add(1)
                    .ok_or(AttestationError::SequenceOutOfRange(u64::MAX))?;
                (
                    previous.chain_id.clone(),
                    sequence,
                    Some(previous.entry_digest.clone()),
                )
            }
        };
        let kind = projection.kind();
        let mut disclosures = Vec::with_capacity(private_memory_ids.len());
        let mut evidence = Vec::with_capacity(private_memory_ids.len());
        for memory_id in private_memory_ids {
            let disclosure = DisclosureMapping {
                receipt_id: receipt_id.clone(),
                evidence_slot: OpaqueEvidenceSlot::random(),
                memory_id,
                nonce: random_disclosure_nonce(),
            };
            evidence.push(disclosure.evidence_commitment());
            disclosures.push(disclosure);
        }
        let attestation = Self {
            schema: RECEIPT_ATTESTATION_SCHEMA_V1.to_string(),
            schema_version: RECEIPT_ATTESTATION_VERSION_V1,
            signature_algorithm: RECEIPT_ATTESTATION_SIGNATURE_ALGORITHM_V1.to_string(),
            receipt_id,
            issued_at,
            producer,
            operation: OperationIdentity {
                kind: kind.operation_kind().to_string(),
                opaque_run_id: OpaqueRunId::random(),
            },
            chain: ChainLink {
                chain_id,
                sequence,
                previous_entry_digest,
            },
            predicate: ReceiptPredicate {
                kind,
                algorithm_version,
                decision_digest: projection.digest()?,
                evidence,
                claim_boundary: kind.claim_boundary(),
            },
        };
        attestation.validate()?;
        Ok(PreparedReceiptAttestation {
            attestation,
            disclosures,
        })
    }

    pub fn receipt_id(&self) -> &OpaqueReceiptId {
        &self.receipt_id
    }

    pub fn issued_at(&self) -> DateTime<Utc> {
        self.issued_at
    }

    pub fn producer(&self) -> &ProducerIdentity {
        &self.producer
    }

    pub fn operation(&self) -> &OperationIdentity {
        &self.operation
    }

    pub fn chain(&self) -> &ChainLink {
        &self.chain
    }

    pub fn predicate(&self) -> &ReceiptPredicate {
        &self.predicate
    }

    /// Validate invariants that are stricter than Serde's type checks.
    pub fn validate(&self) -> Result<(), AttestationError> {
        if self.schema != RECEIPT_ATTESTATION_SCHEMA_V1 {
            return Err(AttestationError::UnsupportedSchema(self.schema.clone()));
        }
        if self.schema_version != RECEIPT_ATTESTATION_VERSION_V1 {
            return Err(AttestationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.signature_algorithm != RECEIPT_ATTESTATION_SIGNATURE_ALGORITHM_V1 {
            return Err(AttestationError::UnsupportedSignatureAlgorithm(
                self.signature_algorithm.clone(),
            ));
        }
        self.receipt_id.validate("receiptId")?;
        self.producer.validate()?;
        require_nonempty("operation.kind", &self.operation.kind)?;
        if self.operation.kind != self.predicate.kind.operation_kind() {
            return Err(AttestationError::OperationKindMismatch);
        }
        self.operation
            .opaque_run_id
            .validate("operation.opaqueRunId")?;
        self.chain.chain_id.validate("chain.chainId")?;
        validate_closed_code(
            "predicate.algorithmVersion",
            &self.predicate.algorithm_version,
        )?;
        validate_digest("predicate.decisionDigest", &self.predicate.decision_digest)?;
        if self.predicate.claim_boundary != self.predicate.kind.claim_boundary() {
            return Err(AttestationError::ClaimBoundaryMismatch);
        }

        if self.chain.sequence > MAX_SAFE_CHAIN_SEQUENCE {
            return Err(AttestationError::SequenceOutOfRange(self.chain.sequence));
        }
        match (self.chain.sequence, &self.chain.previous_entry_digest) {
            (0, None) => {}
            (0, Some(_)) => return Err(AttestationError::GenesisHasPredecessor),
            (_, None) => return Err(AttestationError::MissingPreviousEntryDigest),
            (_, Some(digest)) => validate_digest("chain.previousEntryDigest", digest)?,
        }

        if self.predicate.evidence.len() > MAX_ATTESTATION_EVIDENCE {
            return Err(AttestationError::TooMuchEvidence(
                self.predicate.evidence.len(),
            ));
        }
        let mut slots = HashSet::with_capacity(self.predicate.evidence.len());
        for evidence in &self.predicate.evidence {
            evidence
                .evidence_slot
                .validate("predicate.evidence.evidenceSlot")?;
            validate_digest("predicate.evidence.commitment", &evidence.commitment)?;
            if !slots.insert(evidence.evidence_slot.as_str()) {
                return Err(AttestationError::DuplicateEvidenceSlot(
                    evidence.evidence_slot.as_str().to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Identity of the producer binary. Values must be deployment identities, not
/// hostnames or user identifiers, because signed payloads are immutable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProducerIdentity {
    name: String,
    version: String,
    build: String,
}

impl ProducerIdentity {
    /// Create a deployment identity. Every component is a bounded closed code;
    /// hostnames, paths, e-mail addresses, and whitespace are rejected.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        build: impl Into<String>,
    ) -> Result<Self, AttestationError> {
        let value = Self {
            name: name.into(),
            version: version.into(),
            build: build.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), AttestationError> {
        for (field, value) in [
            ("producer.name", self.name.as_str()),
            ("producer.version", self.version.as_str()),
            ("producer.build", self.build.as_str()),
        ] {
            validate_producer_code(field, value)?;
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn build(&self) -> &str {
        &self.build
    }
}

/// Receipt-local operation identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OperationIdentity {
    kind: String,
    /// Random or committed per-run value; never a stable user/session id.
    opaque_run_id: OpaqueRunId,
}

impl OperationIdentity {
    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn opaque_run_id(&self) -> &OpaqueRunId {
        &self.opaque_run_id
    }
}

/// Signed chain position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChainLink {
    chain_id: OpaqueChainId,
    /// Serialized as a canonical decimal string, not a JSON number.
    #[serde(with = "decimal_u64")]
    sequence: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_entry_digest: Option<String>,
}

impl ChainLink {
    pub fn chain_id(&self) -> &OpaqueChainId {
        &self.chain_id
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn previous_entry_digest(&self) -> Option<&str> {
        self.previous_entry_digest.as_deref()
    }
}

/// Privacy-safe evidence for the decision that produced the receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReceiptPredicate {
    kind: ReceiptPredicateKind,
    algorithm_version: String,
    /// Digest of a caller-defined, redaction-safe decision projection.
    decision_digest: String,
    /// Opaque receipt-local slots and salted commitments only.
    evidence: Vec<EvidenceCommitment>,
    /// Explicitly states what this evidence does and does not establish.
    claim_boundary: ReceiptClaimBoundary,
}

impl ReceiptPredicate {
    pub const fn kind(&self) -> ReceiptPredicateKind {
        self.kind
    }

    pub fn algorithm_version(&self) -> &str {
        &self.algorithm_version
    }

    pub fn decision_digest(&self) -> &str {
        &self.decision_digest
    }

    pub fn evidence(&self) -> &[EvidenceCommitment] {
        &self.evidence
    }

    pub const fn claim_boundary(&self) -> ReceiptClaimBoundary {
        self.claim_boundary
    }
}

/// One immutable evidence slot. It deliberately contains no stable memory id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvidenceCommitment {
    evidence_slot: OpaqueEvidenceSlot,
    commitment: String,
}

impl EvidenceCommitment {
    pub fn evidence_slot(&self) -> &OpaqueEvidenceSlot {
        &self.evidence_slot
    }

    pub fn commitment(&self) -> &str {
        &self.commitment
    }
}

/// Deletable mapping that can selectively disclose one evidence commitment.
///
/// Purge deletes this complete record, including the high-entropy nonce. The
/// attestation and its signature remain verifiable, while the commitment can no
/// longer be resolved from local state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureMapping {
    receipt_id: OpaqueReceiptId,
    evidence_slot: OpaqueEvidenceSlot,
    memory_id: String,
    nonce: [u8; 32],
}

impl DisclosureMapping {
    /// Compute the immutable commitment stored in the signed predicate.
    pub fn evidence_commitment(&self) -> EvidenceCommitment {
        EvidenceCommitment {
            evidence_slot: self.evidence_slot.clone(),
            commitment: disclosure_commitment(
                self.receipt_id.as_str(),
                self.evidence_slot.as_str(),
                &self.memory_id,
                &self.nonce,
            ),
        }
    }

    pub fn receipt_id(&self) -> &OpaqueReceiptId {
        &self.receipt_id
    }

    pub fn evidence_slot(&self) -> &OpaqueEvidenceSlot {
        &self.evidence_slot
    }

    pub(crate) fn memory_id(&self) -> &str {
        &self.memory_id
    }

    pub(crate) fn nonce(&self) -> &[u8; 32] {
        &self.nonce
    }
}

/// Standard DSSE JSON envelope. Writers emit padded standard Base64; readers
/// accept both standard and URL-safe alphabets as required by the DSSE spec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DsseEnvelope {
    #[serde(rename = "payloadType")]
    pub payload_type: String,
    pub payload: String,
    pub signatures: Vec<DsseSignature>,
}

/// One DSSE signature. `keyid` is an unauthenticated lookup hint; trust always
/// comes from independently provisioned [`TrustedSigningKey`] material.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DsseSignature {
    /// Optional and unauthenticated under DSSE. When absent, verification tries
    /// all independently trusted keys within the configured bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyid: Option<String>,
    pub sig: String,
}

/// Output of deterministic canonicalization and signing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedReceiptAttestation {
    pub envelope: DsseEnvelope,
    pub payload_digest: String,
    pub entry_digest: String,
    pub public_key: [u8; 32],
    chain_entry: ChainEntry,
}

impl SignedReceiptAttestation {
    pub fn chain_entry(&self) -> &ChainEntry {
        &self.chain_entry
    }
}

/// Lifecycle state held in the trusted key registry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SigningKeyStatus {
    Active,
    /// Historical signatures remain eligible for verification.
    Retired,
    /// Signatures at or after `revoked_at` are rejected. Without a revocation
    /// time, all signatures made by this key are rejected.
    Revoked,
    /// Administrative hard stop; no signature is accepted.
    Disabled,
}

/// Independently trusted Ed25519 key record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrustedSigningKey {
    pub key_id: String,
    pub public_key: [u8; 32],
    pub status: SigningKeyStatus,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

impl TrustedSigningKey {
    /// Evaluate policy at the signed, claimed issuance time.
    ///
    /// The returned result does not independently prove that wall-clock time;
    /// external timestamping or anchoring is needed to prevent backdating.
    pub fn validity_at(&self, claimed_signing_time: DateTime<Utc>) -> SigningKeyValidity {
        if self.status == SigningKeyStatus::Disabled {
            return SigningKeyValidity::Disabled;
        }
        if claimed_signing_time < self.valid_from {
            return SigningKeyValidity::NotYetValid;
        }
        if self
            .valid_until
            .is_some_and(|until| claimed_signing_time >= until)
        {
            return SigningKeyValidity::Expired;
        }
        if self.status == SigningKeyStatus::Revoked {
            match self.revoked_at {
                Some(revoked_at) if claimed_signing_time < revoked_at => {
                    return SigningKeyValidity::ValidHistorical;
                }
                _ => return SigningKeyValidity::Revoked,
            }
        }
        if self
            .revoked_at
            .is_some_and(|revoked_at| claimed_signing_time >= revoked_at)
        {
            return SigningKeyValidity::Revoked;
        }
        if self.status == SigningKeyStatus::Retired {
            SigningKeyValidity::ValidHistorical
        } else {
            SigningKeyValidity::Valid
        }
    }

    /// Stable fingerprint for registry and checkpoint comparisons.
    pub fn public_key_fingerprint(&self) -> String {
        public_key_fingerprint(&self.public_key)
    }
}

/// Result of evaluating a key at the claimed signing time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigningKeyValidity {
    Valid,
    ValidHistorical,
    NotYetValid,
    Expired,
    Revoked,
    Disabled,
}

/// Row and chain expectations supplied by trusted storage or an exported
/// checkpoint. All fields are optional so callers can perform staged checks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerificationContext {
    pub expected_receipt_id: Option<String>,
    pub expected_payload_digest: Option<String>,
    pub expected_entry_digest: Option<String>,
    pub expected_public_key_fingerprint: Option<String>,
    pub expected_chain_id: Option<String>,
    pub expected_sequence: Option<u64>,
    pub predecessor: PredecessorExpectation,
    /// An externally trusted immediately preceding entry. This authenticates
    /// linkage only and is never compared with the current signer's key.
    pub predecessor_anchor: Option<TrustedPredecessorAnchor>,
    /// An independently held expectation for the entry being verified.
    pub expected_terminal_head: Option<ExpectedTerminalHead>,
}

/// What the caller knows about the immediately preceding entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PredecessorExpectation {
    /// Linkage was not checked. Verification reports an unanchored warning.
    #[default]
    Unchecked,
    /// This entry must be the chain genesis.
    Genesis,
    /// The predecessor was required but unavailable (a visible gap).
    Missing,
    /// Metadata from a separately verified predecessor.
    Previous(ChainEntry),
}

/// Previously exported or externally anchored immediately preceding entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedPredecessorAnchor {
    pub chain_id: String,
    pub sequence: u64,
    pub entry_digest: String,
}

/// Independently held expectation for the terminal entry of an observed
/// segment. Unlike a predecessor anchor, its optional key fingerprint refers
/// to this exact entry's signer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTerminalHead {
    pub chain_id: String,
    pub sequence: u64,
    pub entry_digest: String,
    pub public_key_fingerprint: Option<String>,
}

/// Detailed result for one immutable envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub attestation: Option<ReceiptAttestationV1>,
    pub payload_digest: Option<String>,
    pub entry_digest: Option<String>,
    pub signature_valid: bool,
    pub canonical_payload: bool,
    pub key_validity: Option<SigningKeyValidity>,
    pub verified_key_id: Option<String>,
    pub verified_public_key_fingerprint: Option<String>,
    pub verified_signature_index: Option<usize>,
    pub anchored: bool,
    pub predecessor_anchored: bool,
    pub terminal_head_matched: bool,
    pub failures: Vec<VerificationFailure>,
    pub warnings: Vec<VerificationWarning>,
}

impl VerificationReport {
    fn new() -> Self {
        Self {
            attestation: None,
            payload_digest: None,
            entry_digest: None,
            signature_valid: false,
            canonical_payload: false,
            key_validity: None,
            verified_key_id: None,
            verified_public_key_fingerprint: None,
            verified_signature_index: None,
            anchored: false,
            predecessor_anchored: false,
            terminal_head_matched: false,
            failures: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// True when every requested structural, cryptographic, row, and link
    /// check succeeded. External anchoring is intentionally reported separately.
    pub fn is_valid(&self) -> bool {
        self.failures.is_empty()
    }

    /// True only when the entry is valid and tied to a supplied trusted checkpoint.
    pub fn is_anchored_valid(&self) -> bool {
        self.is_valid() && self.anchored
    }

    /// Return chain metadata only after all requested verification checks pass.
    pub fn chain_entry(&self) -> Option<ChainEntry> {
        if !self.is_valid() || !self.signature_valid {
            return None;
        }
        let attestation = self.attestation.as_ref()?;
        Some(ChainEntry {
            receipt_id: attestation.receipt_id.clone(),
            chain_id: attestation.chain.chain_id.clone(),
            sequence: attestation.chain.sequence,
            previous_entry_digest: attestation.chain.previous_entry_digest.clone(),
            entry_digest: self.entry_digest.clone()?,
            signer_public_key_fingerprint: self.verified_public_key_fingerprint.clone()?,
        })
    }

    fn fail_once(&mut self, failure: VerificationFailure) {
        if !self.failures.contains(&failure) {
            self.failures.push(failure);
        }
    }

    fn warn_once(&mut self, warning: VerificationWarning) {
        if !self.warnings.contains(&warning) {
            self.warnings.push(warning);
        }
    }
}

/// A failure is specific enough for an API or dashboard to explain exactly
/// which assurance was not established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationFailure {
    UnsupportedPayloadType {
        actual: String,
    },
    MalformedPayloadBase64,
    PayloadTooLarge,
    MissingSignature,
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    TooManyTrustedKeys {
        actual: usize,
        maximum: usize,
    },
    UnknownSigningKey {
        key_id: Option<String>,
    },
    KeyIdMismatch {
        envelope: String,
        trusted: String,
    },
    MalformedSignatureBase64,
    MalformedSignature,
    MalformedPublicKey,
    InvalidEd25519Signature,
    PayloadJsonPolicyViolation {
        reason: String,
    },
    CanonicalPayloadMismatch,
    PayloadSchemaInvalid {
        reason: String,
    },
    SchemaMismatch {
        actual: String,
    },
    SchemaVersionMismatch {
        actual: u32,
    },
    SignatureAlgorithmMismatch {
        actual: String,
    },
    PayloadDigestMismatch {
        expected: String,
        actual: String,
    },
    EntryDigestMismatch {
        expected: String,
        actual: String,
    },
    ReceiptIdMismatch {
        expected: String,
        actual: String,
    },
    PublicKeyMismatch {
        expected: String,
        actual: String,
    },
    ChainIdMismatch {
        expected: String,
        actual: String,
    },
    WrongSequence {
        expected: u64,
        actual: u64,
    },
    WrongPredecessor {
        expected: String,
        actual: Option<String>,
    },
    MissingPredecessor {
        sequence: u64,
    },
    WrongGenesis,
    KeyNotYetValid,
    KeyExpired,
    KeyRevoked,
    KeyDisabled,
    CheckpointMismatch,
    PredecessorAnchorMismatch,
    TerminalHeadMismatch,
}

/// Non-fatal assurance boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationWarning {
    /// The observed entry or chain was not connected to a caller-held checkpoint.
    UnanchoredChain,
    /// A retired or subsequently revoked key was valid at the claimed signing time.
    HistoricallyValidKey,
}

/// Structural construction/canonicalization errors.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("unsupported attestation schema: {0}")]
    UnsupportedSchema(String),
    #[error("unsupported attestation schema version: {0}")]
    UnsupportedSchemaVersion(u32),
    #[error("unsupported signature algorithm: {0}")]
    UnsupportedSignatureAlgorithm(String),
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("field exceeds its byte bound: {field} ({actual} > {maximum})")]
    FieldTooLong {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("field is not a closed implementation code: {0}")]
    InvalidClosedCode(&'static str),
    #[error("field is not a random opaque v4 identifier: {0}")]
    InvalidOpaqueId(&'static str),
    #[error("unsafe Unicode noncharacter in field: {0}")]
    UnicodeNoncharacter(&'static str),
    #[error("invalid lowercase BLAKE3 digest in {0}")]
    InvalidDigest(&'static str),
    #[error("chain sequence exceeds the I-JSON safe range: {0}")]
    SequenceOutOfRange(u64),
    #[error("genesis entry must not have a predecessor")]
    GenesisHasPredecessor,
    #[error("non-genesis entry requires a previous entry digest")]
    MissingPreviousEntryDigest,
    #[error("duplicate evidence slot: {0}")]
    DuplicateEvidenceSlot(String),
    #[error("operation kind does not match the typed predicate kind")]
    OperationKindMismatch,
    #[error("claim boundary does not match the typed predicate kind")]
    ClaimBoundaryMismatch,
    #[error("attestation evidence count exceeds the bound: {0}")]
    TooMuchEvidence(usize),
    #[error("invalid redaction-safe decision projection: {0}")]
    InvalidProjection(&'static str),
    #[error("key id must not be empty")]
    EmptyKeyId,
    #[error("RFC 8785 canonicalization failed: {0}")]
    Canonicalization(#[from] serde_json::Error),
}

/// Produce RFC 8785 canonical bytes after enforcing the v1 payload invariants.
pub fn canonical_attestation_bytes(
    attestation: &ReceiptAttestationV1,
) -> Result<Vec<u8>, AttestationError> {
    attestation.validate()?;
    Ok(serde_json_canonicalizer::to_vec(attestation)?)
}

/// Exact DSSE v1 pre-authentication encoding:
/// `DSSEv1 SP len(type) SP type SP len(payload) SP payload`.
pub fn dsse_pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let payload_type = payload_type.as_bytes();
    let mut encoded = Vec::with_capacity(payload_type.len() + payload.len() + 32);
    encoded.extend_from_slice(b"DSSEv1 ");
    encoded.extend_from_slice(payload_type.len().to_string().as_bytes());
    encoded.push(b' ');
    encoded.extend_from_slice(payload_type);
    encoded.push(b' ');
    encoded.extend_from_slice(payload.len().to_string().as_bytes());
    encoded.push(b' ');
    encoded.extend_from_slice(payload);
    encoded
}

/// Canonicalize and sign an attestation from a caller-managed 32-byte Ed25519 seed.
///
/// Secure seed generation, at-rest protection, rotation, and deletion belong to
/// the keystore/integration layer rather than this deterministic core.
pub fn sign_attestation(
    attestation: &ReceiptAttestationV1,
    key_id: &str,
    signing_seed: &[u8; 32],
) -> Result<SignedReceiptAttestation, AttestationError> {
    if key_id.trim().is_empty() {
        return Err(AttestationError::EmptyKeyId);
    }
    let payload = canonical_attestation_bytes(attestation)?;
    let pae = dsse_pae(RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1, &payload);
    let signing_key = SigningKey::from_bytes(signing_seed);
    let signature = signing_key.sign(&pae).to_bytes();
    let payload_digest = payload_digest(&payload);
    let entry_digest = entry_digest(
        RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1,
        &payload,
        key_id,
        &signature,
    );
    let public_key = signing_key.verifying_key().to_bytes();
    let chain_entry = ChainEntry {
        receipt_id: attestation.receipt_id.clone(),
        chain_id: attestation.chain.chain_id.clone(),
        sequence: attestation.chain.sequence,
        previous_entry_digest: attestation.chain.previous_entry_digest.clone(),
        entry_digest: entry_digest.clone(),
        signer_public_key_fingerprint: public_key_fingerprint(&public_key),
    };
    Ok(SignedReceiptAttestation {
        envelope: DsseEnvelope {
            payload_type: RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1.to_string(),
            payload: STANDARD.encode(&payload),
            signatures: vec![DsseSignature {
                keyid: Some(key_id.to_string()),
                sig: STANDARD.encode(signature),
            }],
        },
        payload_digest,
        entry_digest,
        public_key,
        chain_entry,
    })
}

/// Strictly verify one immutable envelope and all caller-supplied expectations.
pub fn verify_envelope(
    envelope: &DsseEnvelope,
    trusted_key: &TrustedSigningKey,
    context: &VerificationContext,
) -> VerificationReport {
    verify_envelope_with_keys(envelope, std::slice::from_ref(trusted_key), context)
}

struct VerifiedSignatureCandidate<'a> {
    signature_index: usize,
    trusted_key: &'a TrustedSigningKey,
    key_fingerprint: String,
    entry_digest: String,
}

/// Verify an envelope against an independently provisioned trust set.
///
/// DSSE `keyid` is only a lookup hint. Keyless signatures are tried against all
/// trusted keys, unknown/malformed extra signatures do not eclipse a later
/// valid signature, and all Ed25519 checks use strict verification.
pub fn verify_envelope_with_keys(
    envelope: &DsseEnvelope,
    trusted_keys: &[TrustedSigningKey],
    context: &VerificationContext,
) -> VerificationReport {
    let mut report = VerificationReport::new();
    if envelope.payload_type != RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1 {
        report.fail_once(VerificationFailure::UnsupportedPayloadType {
            actual: envelope.payload_type.clone(),
        });
    }

    let payload = match decode_dsse_base64_bounded(&envelope.payload, MAX_DSSE_PAYLOAD_BYTES) {
        Ok(payload) => payload,
        Err(Base64DecodeError::Malformed) => {
            report.fail_once(VerificationFailure::MalformedPayloadBase64);
            return report;
        }
        Err(Base64DecodeError::TooLarge) => {
            report.fail_once(VerificationFailure::PayloadTooLarge);
            return report;
        }
    };
    let actual_payload_digest = payload_digest(&payload);
    report.payload_digest = Some(actual_payload_digest.clone());
    if let Some(expected) = &context.expected_payload_digest
        && expected != &actual_payload_digest
    {
        report.fail_once(VerificationFailure::PayloadDigestMismatch {
            expected: expected.clone(),
            actual: actual_payload_digest,
        });
    }

    if envelope.signatures.is_empty() {
        report.fail_once(VerificationFailure::MissingSignature);
        return finish_unanchored(report, context);
    }
    if envelope.signatures.len() > MAX_DSSE_SIGNATURES {
        report.fail_once(VerificationFailure::TooManySignatures {
            actual: envelope.signatures.len(),
            maximum: MAX_DSSE_SIGNATURES,
        });
        return finish_unanchored(report, context);
    }
    if trusted_keys.len() > MAX_TRUSTED_SIGNING_KEYS {
        report.fail_once(VerificationFailure::TooManyTrustedKeys {
            actual: trusted_keys.len(),
            maximum: MAX_TRUSTED_SIGNING_KEYS,
        });
        return finish_unanchored(report, context);
    }

    let pae = dsse_pae(&envelope.payload_type, &payload);
    let mut candidates = Vec::new();
    let mut first_unknown_key: Option<Option<String>> = None;
    let mut saw_malformed_base64 = false;
    let mut saw_malformed_signature = false;
    let mut saw_malformed_public_key = false;
    let mut saw_invalid_signature = false;
    for (signature_index, signature_record) in envelope.signatures.iter().enumerate() {
        let signature_bytes = match decode_dsse_base64_bounded(&signature_record.sig, 64) {
            Ok(signature) => signature,
            Err(Base64DecodeError::Malformed | Base64DecodeError::TooLarge) => {
                saw_malformed_base64 = true;
                continue;
            }
        };
        let signature = match Signature::from_slice(&signature_bytes) {
            Ok(signature) => signature,
            Err(_) => {
                saw_malformed_signature = true;
                continue;
            }
        };
        let matching_keys: Vec<&TrustedSigningKey> = match signature_record.keyid.as_deref() {
            Some(key_id) => {
                let matches: Vec<_> = trusted_keys
                    .iter()
                    .filter(|key| key.key_id == key_id)
                    .collect();
                if matches.is_empty() {
                    if first_unknown_key.is_none() {
                        first_unknown_key = Some(Some(key_id.to_string()));
                    }
                    continue;
                }
                matches
            }
            None => {
                if trusted_keys.is_empty() {
                    if first_unknown_key.is_none() {
                        first_unknown_key = Some(None);
                    }
                    continue;
                }
                trusted_keys.iter().collect()
            }
        };
        for trusted_key in matching_keys {
            let verifying_key = match VerifyingKey::from_bytes(&trusted_key.public_key) {
                Ok(key) => key,
                Err(_) => {
                    saw_malformed_public_key = true;
                    continue;
                }
            };
            if verifying_key.verify_strict(&pae, &signature).is_err() {
                saw_invalid_signature = true;
                continue;
            }
            let key_id_component = signature_record.keyid.as_deref().unwrap_or("");
            candidates.push(VerifiedSignatureCandidate {
                signature_index,
                trusted_key,
                key_fingerprint: trusted_key.public_key_fingerprint(),
                entry_digest: entry_digest(
                    &envelope.payload_type,
                    &payload,
                    key_id_component,
                    &signature_bytes,
                ),
            });
        }
    }
    if candidates.is_empty() {
        if let Some(key_id) = first_unknown_key {
            report.fail_once(VerificationFailure::UnknownSigningKey { key_id });
        } else if saw_malformed_base64 {
            report.fail_once(VerificationFailure::MalformedSignatureBase64);
        } else if saw_malformed_signature {
            report.fail_once(VerificationFailure::MalformedSignature);
        } else if saw_malformed_public_key {
            report.fail_once(VerificationFailure::MalformedPublicKey);
        } else if saw_invalid_signature {
            report.fail_once(VerificationFailure::InvalidEd25519Signature);
        } else {
            report.fail_once(VerificationFailure::MissingSignature);
        }
        return finish_unanchored(report, context);
    }
    // At least one strict Ed25519 signature over the exact PAE bytes is valid,
    // even if payload policy/schema parsing below fails.
    report.signature_valid = true;

    let value = match parse_json_no_duplicates(&payload) {
        Ok(value) => value,
        Err(reason) => {
            report.fail_once(VerificationFailure::PayloadJsonPolicyViolation { reason });
            return finish_unanchored(report, context);
        }
    };
    match serde_json_canonicalizer::to_vec(&value) {
        Ok(canonical) if canonical == payload => report.canonical_payload = true,
        Ok(_) => report.fail_once(VerificationFailure::CanonicalPayloadMismatch),
        Err(error) => report.fail_once(VerificationFailure::PayloadJsonPolicyViolation {
            reason: error.to_string(),
        }),
    }

    let attestation: ReceiptAttestationV1 = match serde_json::from_value(value) {
        Ok(attestation) => attestation,
        Err(error) => {
            report.fail_once(VerificationFailure::PayloadSchemaInvalid {
                reason: error.to_string(),
            });
            return finish_unanchored(report, context);
        }
    };

    // Prefer a candidate that satisfies independently held row/head metadata,
    // then one whose key is policy-valid at the claimed issuance time. Envelope
    // order is the stable final tie-breaker.
    if let Some(expected) = &context.expected_entry_digest
        && candidates
            .iter()
            .any(|candidate| &candidate.entry_digest == expected)
    {
        candidates.retain(|candidate| &candidate.entry_digest == expected);
    }
    let expected_fingerprint = context
        .expected_public_key_fingerprint
        .as_ref()
        .or_else(|| {
            context
                .expected_terminal_head
                .as_ref()
                .and_then(|head| head.public_key_fingerprint.as_ref())
        });
    if let Some(expected) = expected_fingerprint
        && candidates
            .iter()
            .any(|candidate| &candidate.key_fingerprint == expected)
    {
        candidates.retain(|candidate| &candidate.key_fingerprint == expected);
    }
    if candidates.iter().any(|candidate| {
        matches!(
            candidate.trusted_key.validity_at(attestation.issued_at),
            SigningKeyValidity::Valid | SigningKeyValidity::ValidHistorical
        )
    }) {
        candidates.retain(|candidate| {
            matches!(
                candidate.trusted_key.validity_at(attestation.issued_at),
                SigningKeyValidity::Valid | SigningKeyValidity::ValidHistorical
            )
        });
    }
    let candidate = candidates
        .into_iter()
        .min_by_key(|candidate| candidate.signature_index)
        .expect("non-empty verified signature candidates");
    let actual_entry_digest = candidate.entry_digest.clone();
    let actual_key_fingerprint = candidate.key_fingerprint.clone();
    report.entry_digest = Some(actual_entry_digest.clone());
    report.verified_key_id = Some(candidate.trusted_key.key_id.clone());
    report.verified_public_key_fingerprint = Some(actual_key_fingerprint.clone());
    report.verified_signature_index = Some(candidate.signature_index);
    if let Some(expected) = &context.expected_entry_digest
        && expected != &actual_entry_digest
    {
        report.fail_once(VerificationFailure::EntryDigestMismatch {
            expected: expected.clone(),
            actual: actual_entry_digest.clone(),
        });
    }
    if let Some(expected) = &context.expected_public_key_fingerprint
        && expected != &actual_key_fingerprint
    {
        report.fail_once(VerificationFailure::PublicKeyMismatch {
            expected: expected.clone(),
            actual: actual_key_fingerprint.clone(),
        });
    }
    if attestation.schema != RECEIPT_ATTESTATION_SCHEMA_V1 {
        report.fail_once(VerificationFailure::SchemaMismatch {
            actual: attestation.schema.clone(),
        });
    }
    if attestation.schema_version != RECEIPT_ATTESTATION_VERSION_V1 {
        report.fail_once(VerificationFailure::SchemaVersionMismatch {
            actual: attestation.schema_version,
        });
    }
    if attestation.signature_algorithm != RECEIPT_ATTESTATION_SIGNATURE_ALGORITHM_V1 {
        report.fail_once(VerificationFailure::SignatureAlgorithmMismatch {
            actual: attestation.signature_algorithm.clone(),
        });
    }
    if let Err(error) = attestation.validate() {
        match error {
            AttestationError::UnsupportedSchema(actual) => {
                report.fail_once(VerificationFailure::SchemaMismatch { actual });
            }
            AttestationError::UnsupportedSchemaVersion(actual) => {
                report.fail_once(VerificationFailure::SchemaVersionMismatch { actual });
            }
            AttestationError::UnsupportedSignatureAlgorithm(actual) => {
                report.fail_once(VerificationFailure::SignatureAlgorithmMismatch { actual });
            }
            other => report.fail_once(VerificationFailure::PayloadSchemaInvalid {
                reason: other.to_string(),
            }),
        }
    }

    if let Some(expected) = &context.expected_receipt_id
        && expected != attestation.receipt_id.as_str()
    {
        report.fail_once(VerificationFailure::ReceiptIdMismatch {
            expected: expected.clone(),
            actual: attestation.receipt_id.as_str().to_string(),
        });
    }
    if let Some(expected) = &context.expected_chain_id
        && expected != attestation.chain.chain_id.as_str()
    {
        report.fail_once(VerificationFailure::ChainIdMismatch {
            expected: expected.clone(),
            actual: attestation.chain.chain_id.as_str().to_string(),
        });
    }
    if let Some(expected) = context.expected_sequence
        && expected != attestation.chain.sequence
    {
        report.fail_once(VerificationFailure::WrongSequence {
            expected,
            actual: attestation.chain.sequence,
        });
    }

    let key_validity = candidate.trusted_key.validity_at(attestation.issued_at);
    report.key_validity = Some(key_validity);
    match key_validity {
        SigningKeyValidity::Valid => {}
        SigningKeyValidity::ValidHistorical => {
            report.warn_once(VerificationWarning::HistoricallyValidKey);
        }
        SigningKeyValidity::NotYetValid => report.fail_once(VerificationFailure::KeyNotYetValid),
        SigningKeyValidity::Expired => report.fail_once(VerificationFailure::KeyExpired),
        SigningKeyValidity::Revoked => report.fail_once(VerificationFailure::KeyRevoked),
        SigningKeyValidity::Disabled => report.fail_once(VerificationFailure::KeyDisabled),
    }

    check_predecessor(&attestation, &context.predecessor, &mut report);
    check_predecessor_anchor(
        &attestation,
        context.predecessor_anchor.as_ref(),
        &mut report,
    );
    check_terminal_head(
        &attestation,
        &actual_entry_digest,
        &actual_key_fingerprint,
        context.expected_terminal_head.as_ref(),
        &mut report,
    );
    report.anchored = report.predecessor_anchored || report.terminal_head_matched;
    report.attestation = Some(attestation);
    finish_unanchored(report, context)
}

/// Domain-separated BLAKE3 digest of exact DSSE payload bytes.
pub fn payload_digest(payload: &[u8]) -> String {
    derive_digest(PAYLOAD_DIGEST_CONTEXT, &[payload])
}

/// Domain-separated BLAKE3 digest for the closed, identifier-free projection
/// accepted by [`ReceiptAttestationV1::build`]. Kept private so callers cannot
/// bypass the typed projection boundary with arbitrary bytes.
fn decision_digest(decision_projection: &[u8]) -> String {
    derive_digest(DECISION_DIGEST_CONTEXT, &[decision_projection])
}

/// Versioned digest of the immutable logical DSSE entry.
///
/// Outer JSON whitespace and Base64 alphabet do not affect the digest. The
/// length-prefixed preimage contains DSSE PAE bytes, key id, and raw signature.
pub fn entry_digest(payload_type: &str, payload: &[u8], key_id: &str, signature: &[u8]) -> String {
    let pae = dsse_pae(payload_type, payload);
    derive_digest(ENTRY_DIGEST_CONTEXT, &[&pae, key_id.as_bytes(), signature])
}

/// Stable fingerprint for raw Ed25519 public-key bytes.
pub fn public_key_fingerprint(public_key: &[u8; 32]) -> String {
    derive_digest(PUBLIC_KEY_FINGERPRINT_CONTEXT, &[public_key])
}

/// Salted, domain-separated disclosure commitment.
pub fn disclosure_commitment(
    receipt_id: &str,
    evidence_slot: &str,
    memory_id: &str,
    nonce: &[u8; 32],
) -> String {
    derive_digest(
        DISCLOSURE_COMMITMENT_CONTEXT,
        &[
            receipt_id.as_bytes(),
            evidence_slot.as_bytes(),
            memory_id.as_bytes(),
            nonce,
        ],
    )
}

/// Outcome of resolving a deletable disclosure against an immutable predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisclosureVerification {
    Verified,
    /// Neutral absence: no mapping is currently available and no cause has
    /// been cryptographically/operationally established.
    MissingDisclosure,
    /// The mapping is absent and a matching verified-local-erasure capability
    /// was supplied by the unlearning verifier.
    UnavailableAfterVerifiedErasure,
    ErasureProofMismatch,
    EvidenceSlotNotCommitted,
    ReceiptIdMismatch,
    CommitmentMismatch,
}

/// Capability produced only inside `vestige-core` after the complete local
/// unlearning verifier has validated its signed ledger/postconditions. This
/// type is intentionally not constructible by API consumers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedDisclosureErasure {
    receipt_id: OpaqueReceiptId,
    evidence_slot: OpaqueEvidenceSlot,
    commitment: String,
    erasure_proof_digest: String,
}

impl VerifiedDisclosureErasure {
    /// Integration seam for the verified-local-unlearning implementation.
    /// Calling code must have verified both the erasure ledger signature and
    /// the complete in-scope postcondition set before minting this capability.
    #[allow(dead_code)] // wired by the integration layer in the same crate
    pub(crate) fn after_verified_local_erasure(
        receipt_id: OpaqueReceiptId,
        evidence_slot: OpaqueEvidenceSlot,
        commitment: impl Into<String>,
        erasure_proof_digest: impl Into<String>,
    ) -> Result<Self, AttestationError> {
        let commitment = commitment.into();
        let erasure_proof_digest = erasure_proof_digest.into();
        validate_digest("erasure.commitment", &commitment)?;
        validate_digest("erasure.proofDigest", &erasure_proof_digest)?;
        Ok(Self {
            receipt_id,
            evidence_slot,
            commitment,
            erasure_proof_digest,
        })
    }

    pub fn erasure_proof_digest(&self) -> &str {
        &self.erasure_proof_digest
    }
}

/// Verify a disclosure, preserving intentional erasure as distinct from tamper.
pub fn verify_disclosure(
    attestation: &ReceiptAttestationV1,
    evidence_slot: &str,
    disclosure: Option<&DisclosureMapping>,
) -> DisclosureVerification {
    verify_disclosure_with_erasure_proof(attestation, evidence_slot, disclosure, None)
}

/// Verify a disclosure while treating absence as erasure only when a matching
/// capability from the verified-local-unlearning path is supplied.
pub fn verify_disclosure_with_erasure_proof(
    attestation: &ReceiptAttestationV1,
    evidence_slot: &str,
    disclosure: Option<&DisclosureMapping>,
    verified_erasure: Option<&VerifiedDisclosureErasure>,
) -> DisclosureVerification {
    let Some(expected) = attestation
        .predicate
        .evidence
        .iter()
        .find(|evidence| evidence.evidence_slot.as_str() == evidence_slot)
    else {
        return DisclosureVerification::EvidenceSlotNotCommitted;
    };
    let Some(disclosure) = disclosure else {
        return match verified_erasure {
            Some(proof)
                if proof.receipt_id == attestation.receipt_id
                    && proof.evidence_slot.as_str() == evidence_slot
                    && proof.commitment == expected.commitment =>
            {
                DisclosureVerification::UnavailableAfterVerifiedErasure
            }
            Some(_) => DisclosureVerification::ErasureProofMismatch,
            None => DisclosureVerification::MissingDisclosure,
        };
    };
    if disclosure.receipt_id != attestation.receipt_id {
        return DisclosureVerification::ReceiptIdMismatch;
    }
    if disclosure.evidence_slot.as_str() != evidence_slot {
        return DisclosureVerification::EvidenceSlotNotCommitted;
    }
    let actual = disclosure_commitment(
        disclosure.receipt_id.as_str(),
        disclosure.evidence_slot.as_str(),
        &disclosure.memory_id,
        &disclosure.nonce,
    );
    if actual == expected.commitment {
        DisclosureVerification::Verified
    } else {
        DisclosureVerification::CommitmentMismatch
    }
}

/// Chain metadata derived only after an envelope has been verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainEntry {
    receipt_id: OpaqueReceiptId,
    chain_id: OpaqueChainId,
    sequence: u64,
    previous_entry_digest: Option<String>,
    entry_digest: String,
    signer_public_key_fingerprint: String,
}

impl ChainEntry {
    pub fn receipt_id(&self) -> &OpaqueReceiptId {
        &self.receipt_id
    }

    pub fn chain_id(&self) -> &OpaqueChainId {
        &self.chain_id
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn previous_entry_digest(&self) -> Option<&str> {
        self.previous_entry_digest.as_deref()
    }

    pub fn entry_digest(&self) -> &str {
        &self.entry_digest
    }

    pub fn signer_public_key_fingerprint(&self) -> &str {
        &self.signer_public_key_fingerprint
    }
}

/// Full result for an ordered observed chain segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainVerificationReport {
    pub anchored: bool,
    pub predecessor_anchored: bool,
    pub terminal_head_matched: bool,
    pub complete_from_genesis: bool,
    pub failures: Vec<ChainFailure>,
    pub warnings: Vec<VerificationWarning>,
}

impl ChainVerificationReport {
    pub fn is_valid(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Link-level failure for an observed ordered chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainFailure {
    EmptyChain,
    WrongGenesis,
    MissingPredecessor {
        sequence: u64,
    },
    ChainIdMismatch {
        expected: String,
        actual: String,
    },
    SequenceGap {
        expected: u64,
        actual: u64,
    },
    DuplicateOrReorderedSequence {
        previous: u64,
        actual: u64,
    },
    PreviousDigestMismatch {
        expected: String,
        actual: Option<String>,
    },
    InvalidEntryDigest {
        receipt_id: String,
    },
    CheckpointMismatch,
    PredecessorAnchorMismatch,
    TerminalHeadMismatch,
}

/// Verify sequence and digest linkage for an ordered chain segment.
///
/// Inputs must come from successful envelope verification. This function does
/// not itself re-check signatures.
pub fn verify_chain(
    entries: &[ChainEntry],
    predecessor_anchor: Option<&TrustedPredecessorAnchor>,
) -> ChainVerificationReport {
    verify_chain_segment(entries, predecessor_anchor, None)
}

/// Verify an ordered chain segment with separate predecessor and terminal-head
/// expectations. This separation permits signer rotation at any link.
pub fn verify_chain_segment(
    entries: &[ChainEntry],
    predecessor_anchor: Option<&TrustedPredecessorAnchor>,
    expected_terminal_head: Option<&ExpectedTerminalHead>,
) -> ChainVerificationReport {
    let mut report = ChainVerificationReport {
        anchored: false,
        predecessor_anchored: false,
        terminal_head_matched: false,
        complete_from_genesis: false,
        failures: Vec::new(),
        warnings: Vec::new(),
    };
    let Some(first) = entries.first() else {
        report.failures.push(ChainFailure::EmptyChain);
        return report;
    };
    for entry in entries {
        if !is_digest(&entry.entry_digest) {
            report.failures.push(ChainFailure::InvalidEntryDigest {
                receipt_id: entry.receipt_id.as_str().to_string(),
            });
        }
    }

    if first.sequence == 0 {
        report.complete_from_genesis = first.previous_entry_digest.is_none();
        if first.previous_entry_digest.is_some() {
            report.failures.push(ChainFailure::WrongGenesis);
        }
    } else if predecessor_anchor.is_none() {
        report.failures.push(ChainFailure::MissingPredecessor {
            sequence: first.sequence,
        });
    }

    if let Some(anchor) = predecessor_anchor {
        let direct_successor = anchor.chain_id == first.chain_id.as_str()
            && anchor
                .sequence
                .checked_add(1)
                .is_some_and(|sequence| sequence == first.sequence)
            && first.previous_entry_digest.as_deref() == Some(anchor.entry_digest.as_str());
        if direct_successor {
            report.predecessor_anchored = true;
        } else {
            report
                .failures
                .push(ChainFailure::PredecessorAnchorMismatch);
        }
    }

    for pair in entries.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if current.chain_id != previous.chain_id {
            report.failures.push(ChainFailure::ChainIdMismatch {
                expected: previous.chain_id.as_str().to_string(),
                actual: current.chain_id.as_str().to_string(),
            });
        }
        match previous.sequence.checked_add(1) {
            Some(expected) if current.sequence == expected => {}
            Some(expected) if current.sequence > previous.sequence => {
                report.failures.push(ChainFailure::SequenceGap {
                    expected,
                    actual: current.sequence,
                });
            }
            _ => report
                .failures
                .push(ChainFailure::DuplicateOrReorderedSequence {
                    previous: previous.sequence,
                    actual: current.sequence,
                }),
        }
        if current.previous_entry_digest.as_deref() != Some(previous.entry_digest.as_str()) {
            report.failures.push(ChainFailure::PreviousDigestMismatch {
                expected: previous.entry_digest.clone(),
                actual: current.previous_entry_digest.clone(),
            });
        }
    }

    if let Some(head) = expected_terminal_head {
        let terminal = entries.last().expect("non-empty chain");
        let matched = head.chain_id == terminal.chain_id.as_str()
            && head.sequence == terminal.sequence
            && head.entry_digest == terminal.entry_digest
            && head
                .public_key_fingerprint
                .as_ref()
                .is_none_or(|expected| expected == &terminal.signer_public_key_fingerprint);
        if matched {
            report.terminal_head_matched = true;
        } else {
            report.failures.push(ChainFailure::TerminalHeadMismatch);
        }
    }

    report.anchored = report.predecessor_anchored || report.terminal_head_matched;

    if !report.anchored {
        report.warnings.push(VerificationWarning::UnanchoredChain);
    }
    if !report.failures.is_empty() {
        report.complete_from_genesis = false;
    }
    report
}

fn check_predecessor(
    attestation: &ReceiptAttestationV1,
    predecessor: &PredecessorExpectation,
    report: &mut VerificationReport,
) {
    match predecessor {
        PredecessorExpectation::Unchecked => {}
        PredecessorExpectation::Genesis => {
            if attestation.chain.sequence != 0 || attestation.chain.previous_entry_digest.is_some()
            {
                report.fail_once(VerificationFailure::WrongGenesis);
            }
        }
        PredecessorExpectation::Missing => {
            if attestation.chain.sequence > 0 {
                report.fail_once(VerificationFailure::MissingPredecessor {
                    sequence: attestation.chain.sequence,
                });
            }
        }
        PredecessorExpectation::Previous(previous) => {
            if previous.chain_id != attestation.chain.chain_id {
                report.fail_once(VerificationFailure::ChainIdMismatch {
                    expected: previous.chain_id.as_str().to_string(),
                    actual: attestation.chain.chain_id.as_str().to_string(),
                });
            }
            let expected_sequence = previous.sequence.saturating_add(1);
            if attestation.chain.sequence != expected_sequence {
                report.fail_once(VerificationFailure::WrongSequence {
                    expected: expected_sequence,
                    actual: attestation.chain.sequence,
                });
            }
            if attestation.chain.previous_entry_digest.as_deref()
                != Some(previous.entry_digest.as_str())
            {
                report.fail_once(VerificationFailure::WrongPredecessor {
                    expected: previous.entry_digest.clone(),
                    actual: attestation.chain.previous_entry_digest.clone(),
                });
            }
        }
    }
}

fn check_predecessor_anchor(
    attestation: &ReceiptAttestationV1,
    anchor: Option<&TrustedPredecessorAnchor>,
    report: &mut VerificationReport,
) {
    let Some(anchor) = anchor else {
        return;
    };
    let direct_successor = anchor.chain_id == attestation.chain.chain_id.as_str()
        && anchor.sequence.checked_add(1) == Some(attestation.chain.sequence)
        && attestation.chain.previous_entry_digest.as_deref() == Some(anchor.entry_digest.as_str());
    if direct_successor {
        report.predecessor_anchored = true;
    } else {
        report.fail_once(VerificationFailure::PredecessorAnchorMismatch);
    }
}

fn check_terminal_head(
    attestation: &ReceiptAttestationV1,
    entry_digest: &str,
    public_key_fingerprint: &str,
    expected: Option<&ExpectedTerminalHead>,
    report: &mut VerificationReport,
) {
    let Some(expected) = expected else {
        return;
    };
    let matched = expected.chain_id == attestation.chain.chain_id.as_str()
        && expected.sequence == attestation.chain.sequence
        && expected.entry_digest == entry_digest
        && expected
            .public_key_fingerprint
            .as_deref()
            .is_none_or(|fingerprint| fingerprint == public_key_fingerprint);
    if matched {
        report.terminal_head_matched = true;
    } else {
        report.fail_once(VerificationFailure::TerminalHeadMismatch);
    }
}

fn finish_unanchored(
    mut report: VerificationReport,
    _context: &VerificationContext,
) -> VerificationReport {
    if !report.anchored {
        report.warn_once(VerificationWarning::UnanchoredChain);
    }
    report
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), AttestationError> {
    if value.trim().is_empty() {
        Err(AttestationError::EmptyField(field))
    } else if contains_unicode_noncharacter(value) {
        Err(AttestationError::UnicodeNoncharacter(field))
    } else {
        Ok(())
    }
}

fn require_bounded_nonempty(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), AttestationError> {
    require_nonempty(field, value)?;
    if value.len() > maximum {
        Err(AttestationError::FieldTooLong {
            field,
            actual: value.len(),
            maximum,
        })
    } else {
        Ok(())
    }
}

fn validate_closed_code(field: &'static str, value: &str) -> Result<(), AttestationError> {
    require_bounded_nonempty(field, value, MAX_CLOSED_CODE_BYTES)?;
    if value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
    }) {
        Ok(())
    } else {
        Err(AttestationError::InvalidClosedCode(field))
    }
}

fn validate_producer_code(field: &'static str, value: &str) -> Result<(), AttestationError> {
    require_bounded_nonempty(field, value, MAX_PRODUCER_FIELD_BYTES)?;
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'+'))
    {
        Ok(())
    } else {
        Err(AttestationError::InvalidClosedCode(field))
    }
}

fn validate_opaque_v4_id(
    field: &'static str,
    value: &str,
    prefix: &str,
) -> Result<(), AttestationError> {
    let Some(encoded) = value
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('_'))
    else {
        return Err(AttestationError::InvalidOpaqueId(field));
    };
    if encoded.len() != 32
        || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit())
        || Uuid::parse_str(encoded)
            .ok()
            .and_then(|uuid| uuid.get_version())
            != Some(Version::Random)
    {
        return Err(AttestationError::InvalidOpaqueId(field));
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), AttestationError> {
    if is_digest(value) {
        Ok(())
    } else {
        Err(AttestationError::InvalidDigest(field))
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn derive_digest(context: &str, parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    for part in parts {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

fn random_disclosure_nonce() -> [u8; 32] {
    // `Uuid::new_v4` draws from the operating-system random source. Four
    // independent draws are compressed through a domain-separated BLAKE3 XOF,
    // avoiding UUID version/variant fixed bits in the resulting nonce.
    let mut hasher = blake3::Hasher::new_derive_key(DISCLOSURE_NONCE_CONTEXT);
    for _ in 0..4 {
        hasher.update(Uuid::new_v4().as_bytes());
    }
    let mut nonce = [0_u8; 32];
    hasher.finalize_xof().fill(&mut nonce);
    nonce
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Base64DecodeError {
    Malformed,
    TooLarge,
}

fn decode_dsse_base64_bounded(
    encoded: &str,
    maximum_decoded_bytes: usize,
) -> Result<Vec<u8>, Base64DecodeError> {
    // Four Base64 characters encode at most three bytes. Reject before
    // allocation/decoding, with a small allowance for padding.
    let maximum_encoded = maximum_decoded_bytes
        .checked_add(2)
        .and_then(|value| value.checked_div(3))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(4))
        .ok_or(Base64DecodeError::TooLarge)?;
    if encoded.len() > maximum_encoded {
        return Err(Base64DecodeError::TooLarge);
    }
    STANDARD
        .decode(encoded)
        .or_else(|_| STANDARD_NO_PAD.decode(encoded))
        .or_else(|_| URL_SAFE.decode(encoded))
        .or_else(|_| URL_SAFE_NO_PAD.decode(encoded))
        .map_err(|_| Base64DecodeError::Malformed)
        .and_then(|decoded| {
            if decoded.len() > maximum_decoded_bytes {
                Err(Base64DecodeError::TooLarge)
            } else {
                Ok(decoded)
            }
        })
}

fn contains_unicode_noncharacter(value: &str) -> bool {
    value.chars().any(|character| {
        let code = character as u32;
        (0xFDD0..=0xFDEF).contains(&code) || code & 0xFFFF == 0xFFFE || code & 0xFFFF == 0xFFFF
    })
}

/// Parse JSON while rejecting duplicate object member names. `serde_json`'s
/// normal `Value` parser uses last-wins semantics, which is unsafe for signed
/// policy objects even though canonical reserialization would be deterministic.
fn parse_json_no_duplicates(bytes: &[u8]) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = DuplicateCheckedValue::deserialize(&mut deserializer)
        .map_err(|error| error.to_string())?
        .0;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(value)
}

struct DuplicateCheckedValue(Value);

impl<'de> Deserialize<'de> for DuplicateCheckedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateCheckedVisitor)
    }
}

struct DuplicateCheckedVisitor;

impl<'de> Visitor<'de> for DuplicateCheckedVisitor {
    type Value = DuplicateCheckedValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an I-JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(DuplicateCheckedValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.unsigned_abs() > MAX_SAFE_CHAIN_SEQUENCE {
            return Err(de::Error::custom("integer exceeds the I-JSON safe range"));
        }
        Ok(DuplicateCheckedValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value > MAX_SAFE_CHAIN_SEQUENCE {
            return Err(de::Error::custom("integer exceeds the I-JSON safe range"));
        }
        Ok(DuplicateCheckedValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.abs() > MAX_SAFE_CHAIN_SEQUENCE as f64 {
            return Err(E::custom("number exceeds the I-JSON safe range"));
        }
        let number = Number::from_f64(value)
            .ok_or_else(|| E::custom("non-finite JSON number is forbidden"))?;
        Ok(DuplicateCheckedValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_string())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if contains_unicode_noncharacter(&value) {
            return Err(de::Error::custom(
                "Unicode noncharacters are forbidden by policy",
            ));
        }
        Ok(DuplicateCheckedValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(DuplicateCheckedValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(DuplicateCheckedValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        DuplicateCheckedValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<DuplicateCheckedValue>()? {
            values.push(value.0);
        }
        Ok(DuplicateCheckedValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if contains_unicode_noncharacter(&key) {
                return Err(de::Error::custom(
                    "Unicode noncharacters are forbidden in object keys",
                ));
            }
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate object key: {key}")));
            }
            let value = object.next_value::<DuplicateCheckedValue>()?;
            values.insert(key, value.0);
        }
        Ok(DuplicateCheckedValue(Value::Object(values)))
    }
}

mod normalized_utc {
    use super::*;

    pub fn serialize<S>(value: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&value.to_rfc3339_opts(SecondsFormat::AutoSi, true))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        let parsed = DateTime::parse_from_rfc3339(&encoded)
            .map_err(de::Error::custom)?
            .with_timezone(&Utc);
        let normalized = parsed.to_rfc3339_opts(SecondsFormat::AutoSi, true);
        if encoded != normalized {
            return Err(de::Error::custom(format!(
                "timestamp must be normalized UTC: expected {normalized}"
            )));
        }
        Ok(parsed)
    }
}

mod decimal_u64 {
    use super::*;

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        if encoded.is_empty()
            || (encoded.len() > 1 && encoded.starts_with('0'))
            || !encoded.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(de::Error::custom(
                "sequence must be an unsigned decimal string without leading zeros",
            ));
        }
        encoded.parse().map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    const SEED: [u8; 32] = [7; 32];

    fn issued_at() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 10, 12, 34, 56)
            .single()
            .unwrap()
    }

    fn disclosure() -> DisclosureMapping {
        DisclosureMapping {
            receipt_id: OpaqueReceiptId("ratt_00000000000040008000000000000001".to_string()),
            evidence_slot: OpaqueEvidenceSlot("slot_00000000000040008000000000000004".to_string()),
            memory_id: "mem_private_stable_id".to_string(),
            nonce: [19; 32],
        }
    }

    fn attestation() -> ReceiptAttestationV1 {
        ReceiptAttestationV1 {
            schema: RECEIPT_ATTESTATION_SCHEMA_V1.to_string(),
            schema_version: RECEIPT_ATTESTATION_VERSION_V1,
            signature_algorithm: RECEIPT_ATTESTATION_SIGNATURE_ALGORITHM_V1.to_string(),
            receipt_id: disclosure().receipt_id,
            issued_at: issued_at(),
            producer: ProducerIdentity::new("vestige-core", "2.3.0", "98f302d").unwrap(),
            operation: OperationIdentity {
                kind: "synaptic_capture".to_string(),
                opaque_run_id: OpaqueRunId("run_00000000000040008000000000000002".to_string()),
            },
            chain: ChainLink {
                chain_id: OpaqueChainId("chain_00000000000040008000000000000003".to_string()),
                sequence: 0,
                previous_entry_digest: None,
            },
            predicate: ReceiptPredicate {
                kind: ReceiptPredicateKind::SynapticCapture,
                algorithm_version: "synaptic-capture-v1".to_string(),
                decision_digest: decision_digest(b"redaction-safe-decision"),
                evidence: vec![disclosure().evidence_commitment()],
                claim_boundary: ReceiptClaimBoundary::DecisionEvidenceNotTruthOrCausality,
            },
        }
    }

    fn trusted_key() -> TrustedSigningKey {
        TrustedSigningKey {
            key_id: "local-key-1".to_string(),
            public_key: SigningKey::from_bytes(&SEED).verifying_key().to_bytes(),
            status: SigningKeyStatus::Active,
            valid_from: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            valid_until: None,
            revoked_at: None,
        }
    }

    fn sign_raw(payload_type: &str, payload: &[u8], key_id: &str) -> DsseEnvelope {
        let key = SigningKey::from_bytes(&SEED);
        let signature = key.sign(&dsse_pae(payload_type, payload)).to_bytes();
        DsseEnvelope {
            payload_type: payload_type.to_string(),
            payload: STANDARD.encode(payload),
            signatures: vec![DsseSignature {
                keyid: Some(key_id.to_string()),
                sig: STANDARD.encode(signature),
            }],
        }
    }

    #[test]
    fn dsse_pae_matches_protocol_encoding() {
        assert_eq!(
            dsse_pae("text/plain", b"hello"),
            b"DSSEv1 10 text/plain 5 hello"
        );
    }

    #[test]
    fn canonical_payload_is_stable_and_sequence_is_a_string() {
        let bytes = canonical_attestation_bytes(&attestation()).unwrap();
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(text.contains("\"sequence\":\"0\""));
        assert_eq!(
            serde_json_canonicalizer::to_vec(&serde_json::from_slice::<Value>(&bytes).unwrap())
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn signs_and_strictly_verifies_with_specific_metadata() {
        let signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let context = VerificationContext {
            expected_receipt_id: Some("ratt_00000000000040008000000000000001".to_string()),
            expected_payload_digest: Some(signed.payload_digest.clone()),
            expected_entry_digest: Some(signed.entry_digest.clone()),
            expected_public_key_fingerprint: Some(public_key_fingerprint(&signed.public_key)),
            expected_chain_id: Some("chain_00000000000040008000000000000003".to_string()),
            expected_sequence: Some(0),
            predecessor: PredecessorExpectation::Genesis,
            predecessor_anchor: None,
            expected_terminal_head: Some(ExpectedTerminalHead {
                chain_id: "chain_00000000000040008000000000000003".to_string(),
                sequence: 0,
                entry_digest: signed.entry_digest.clone(),
                public_key_fingerprint: Some(public_key_fingerprint(&signed.public_key)),
            }),
        };
        let report = verify_envelope(&signed.envelope, &trusted_key(), &context);
        assert!(report.is_anchored_valid(), "{:#?}", report.failures);
        assert!(report.signature_valid);
        assert!(report.canonical_payload);
    }

    #[test]
    fn a_swapped_key_id_is_not_trusted() {
        let mut signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        signed.envelope.signatures[0].keyid = Some("attacker-key".to_string());
        let report = verify_envelope(
            &signed.envelope,
            &trusted_key(),
            &VerificationContext::default(),
        );
        assert!(
            report
                .failures
                .iter()
                .any(|failure| matches!(failure, VerificationFailure::UnknownSigningKey { .. }))
        );
        assert!(!report.signature_valid);
        assert!(!report.is_valid());
    }

    #[test]
    fn url_safe_unpadded_dsse_base64_is_accepted() {
        let mut signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let payload = STANDARD.decode(&signed.envelope.payload).unwrap();
        let signature = STANDARD.decode(&signed.envelope.signatures[0].sig).unwrap();
        signed.envelope.payload = URL_SAFE_NO_PAD.encode(payload);
        signed.envelope.signatures[0].sig = URL_SAFE_NO_PAD.encode(signature);
        let report = verify_envelope(
            &signed.envelope,
            &trusted_key(),
            &VerificationContext::default(),
        );
        assert!(report.signature_valid);
        assert!(report.is_valid(), "{:#?}", report.failures);
    }

    #[test]
    fn valid_signature_over_noncanonical_json_is_application_invalid() {
        let canonical = canonical_attestation_bytes(&attestation()).unwrap();
        let value: Value = serde_json::from_slice(&canonical).unwrap();
        let pretty = serde_json::to_vec_pretty(&value).unwrap();
        let envelope = sign_raw(RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1, &pretty, "local-key-1");
        let report = verify_envelope(&envelope, &trusted_key(), &VerificationContext::default());
        assert!(report.signature_valid);
        assert!(!report.canonical_payload);
        assert!(
            report
                .failures
                .contains(&VerificationFailure::CanonicalPayloadMismatch)
        );
    }

    #[test]
    fn payload_tamper_breaks_signature_and_row_digest() {
        let mut signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let expected_payload_digest = signed.payload_digest.clone();
        let mut changed = attestation();
        changed.predicate.algorithm_version = "attacker-policy".to_string();
        signed.envelope.payload = STANDARD.encode(canonical_attestation_bytes(&changed).unwrap());
        let report = verify_envelope(
            &signed.envelope,
            &trusted_key(),
            &VerificationContext {
                expected_payload_digest: Some(expected_payload_digest),
                ..VerificationContext::default()
            },
        );
        assert!(
            report
                .failures
                .contains(&VerificationFailure::InvalidEd25519Signature)
        );
        assert!(
            report.failures.iter().any(|failure| matches!(
                failure,
                VerificationFailure::PayloadDigestMismatch { .. }
            ))
        );
    }

    #[test]
    fn normalized_utc_timestamp_is_an_application_invariant() {
        let canonical = canonical_attestation_bytes(&attestation()).unwrap();
        let encoded = String::from_utf8(canonical)
            .unwrap()
            .replace("2026-08-10T12:34:56Z", "2026-08-10T12:34:56+00:00");
        let value: Value = serde_json::from_str(&encoded).unwrap();
        let payload = serde_json_canonicalizer::to_vec(&value).unwrap();
        let envelope = sign_raw(RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1, &payload, "local-key-1");
        let report = verify_envelope(&envelope, &trusted_key(), &VerificationContext::default());
        assert!(report.signature_valid);
        assert!(report.canonical_payload);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            VerificationFailure::PayloadSchemaInvalid { reason }
                if reason.contains("normalized UTC")
        )));
    }

    #[test]
    fn duplicate_json_keys_are_rejected_even_when_signature_is_valid() {
        let payload = br#"{"schema":"a","schema":"b"}"#;
        let envelope = sign_raw(RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1, payload, "local-key-1");
        let report = verify_envelope(&envelope, &trusted_key(), &VerificationContext::default());
        assert!(report.signature_valid);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            VerificationFailure::PayloadJsonPolicyViolation { reason }
                if reason.contains("duplicate object key")
        )));
    }

    #[test]
    fn disclosure_is_verifiable_then_intentionally_unavailable_after_purge() {
        let attestation = attestation();
        let disclosure = disclosure();
        let slot = disclosure.evidence_slot.as_str();
        assert_eq!(
            verify_disclosure(&attestation, slot, Some(&disclosure)),
            DisclosureVerification::Verified
        );
        assert_eq!(
            verify_disclosure(&attestation, slot, None),
            DisclosureVerification::MissingDisclosure
        );
        let serialized =
            String::from_utf8(canonical_attestation_bytes(&attestation).unwrap()).unwrap();
        assert!(!serialized.contains(&disclosure.memory_id));
    }

    #[test]
    fn disclosure_commitment_binds_every_field() {
        let original = disclosure();
        let expected = original.evidence_commitment().commitment;
        let mut changed = original.clone();
        changed.evidence_slot =
            OpaqueEvidenceSlot("slot_00000000000040008000000000000005".to_string());
        assert_ne!(expected, changed.evidence_commitment().commitment);
        changed = original.clone();
        changed.memory_id = "another-memory".to_string();
        assert_ne!(expected, changed.evidence_commitment().commitment);
        changed = original.clone();
        changed.nonce[0] ^= 1;
        assert_ne!(expected, changed.evidence_commitment().commitment);
    }

    #[test]
    fn chain_verification_detects_gaps_and_supports_external_anchor() {
        let digest0 = payload_digest(b"entry-0");
        let digest1 = payload_digest(b"entry-1");
        let digest3 = payload_digest(b"entry-3");
        let entries = vec![
            ChainEntry {
                receipt_id: OpaqueReceiptId("ratt_00000000000040008000000000000011".to_string()),
                chain_id: OpaqueChainId("chain_00000000000040008000000000000010".to_string()),
                sequence: 1,
                previous_entry_digest: Some(digest0.clone()),
                entry_digest: digest1.clone(),
                signer_public_key_fingerprint: public_key_fingerprint(&[1; 32]),
            },
            ChainEntry {
                receipt_id: OpaqueReceiptId("ratt_00000000000040008000000000000013".to_string()),
                chain_id: OpaqueChainId("chain_00000000000040008000000000000010".to_string()),
                sequence: 3,
                previous_entry_digest: Some(digest1.clone()),
                entry_digest: digest3,
                signer_public_key_fingerprint: public_key_fingerprint(&[2; 32]),
            },
        ];
        let checkpoint = TrustedPredecessorAnchor {
            chain_id: "chain_00000000000040008000000000000010".to_string(),
            sequence: 0,
            entry_digest: digest0,
        };
        let report = verify_chain(&entries, Some(&checkpoint));
        assert!(report.anchored);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            ChainFailure::SequenceGap {
                expected: 2,
                actual: 3
            }
        )));
    }

    #[test]
    fn key_validity_preserves_pre_revocation_history() {
        let mut key = trusted_key();
        key.status = SigningKeyStatus::Revoked;
        key.revoked_at = Some(Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap());
        assert_eq!(
            key.validity_at(issued_at()),
            SigningKeyValidity::ValidHistorical
        );
        assert_eq!(
            key.validity_at(Utc.with_ymd_and_hms(2026, 10, 1, 0, 0, 0).unwrap()),
            SigningKeyValidity::Revoked
        );
    }

    #[test]
    fn verification_rejects_a_key_not_valid_at_claimed_signing_time() {
        let signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let mut key = trusted_key();
        key.valid_from = Utc.with_ymd_and_hms(2026, 8, 11, 0, 0, 0).unwrap();
        let report = verify_envelope(&signed.envelope, &key, &VerificationContext::default());
        assert!(report.signature_valid);
        assert!(
            report
                .failures
                .contains(&VerificationFailure::KeyNotYetValid)
        );
    }

    #[test]
    fn envelope_verification_detects_wrong_predecessor() {
        let mut linked = attestation();
        linked.chain.sequence = 2;
        linked.chain.previous_entry_digest = Some(payload_digest(b"wrong"));
        let signed = sign_attestation(&linked, "local-key-1", &SEED).unwrap();
        let predecessor = ChainEntry {
            receipt_id: OpaqueReceiptId("ratt_00000000000040008000000000000012".to_string()),
            chain_id: linked.chain.chain_id.clone(),
            sequence: 1,
            previous_entry_digest: Some(payload_digest(b"entry-0")),
            entry_digest: payload_digest(b"right"),
            signer_public_key_fingerprint: public_key_fingerprint(&[3; 32]),
        };
        let report = verify_envelope(
            &signed.envelope,
            &trusted_key(),
            &VerificationContext {
                predecessor: PredecessorExpectation::Previous(predecessor),
                ..VerificationContext::default()
            },
        );
        assert!(report.signature_valid);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| matches!(failure, VerificationFailure::WrongPredecessor { .. }))
        );
    }

    #[test]
    fn entry_digest_ignores_base64_representation_but_binds_key_id() {
        let signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let payload = STANDARD.decode(&signed.envelope.payload).unwrap();
        let signature = STANDARD.decode(&signed.envelope.signatures[0].sig).unwrap();
        assert_eq!(
            signed.entry_digest,
            entry_digest(
                RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1,
                &payload,
                "local-key-1",
                &signature
            )
        );
        assert_ne!(
            signed.entry_digest,
            entry_digest(
                RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1,
                &payload,
                "other-key",
                &signature
            )
        );
    }

    #[test]
    fn dsse_optional_keyid_extensions_and_later_valid_signature_are_supported() {
        let signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let payload = STANDARD.decode(&signed.envelope.payload).unwrap();
        let valid_signature = signed.envelope.signatures[0].sig.clone();
        let encoded = serde_json::json!({
            "payloadType": RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1,
            "payload": STANDARD.encode(payload),
            "ignoredEnvelopeExtension": {"future": true},
            "signatures": [
                {"keyid": "unknown-key", "sig": "AAAA", "ignored": 1},
                {"sig": "not base64***"},
                {"sig": valid_signature, "ignoredSignatureExtension": "ok"}
            ]
        });
        let envelope: DsseEnvelope = serde_json::from_value(encoded).unwrap();
        let report =
            verify_envelope_with_keys(&envelope, &[trusted_key()], &VerificationContext::default());
        assert!(report.is_valid(), "{:#?}", report.failures);
        assert!(report.signature_valid);
        assert_eq!(report.verified_signature_index, Some(2));
        assert_eq!(report.verified_key_id.as_deref(), Some("local-key-1"));
    }

    #[test]
    fn unknown_key_is_distinct_from_an_invalid_known_key_signature() {
        let envelope = sign_raw(
            RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1,
            &canonical_attestation_bytes(&attestation()).unwrap(),
            "not-in-registry",
        );
        let report =
            verify_envelope_with_keys(&envelope, &[trusted_key()], &VerificationContext::default());
        assert!(
            report
                .failures
                .contains(&VerificationFailure::UnknownSigningKey {
                    key_id: Some("not-in-registry".to_string()),
                })
        );
        assert!(!report.signature_valid);
    }

    #[test]
    fn predecessor_anchor_and_terminal_head_are_separate_across_key_rotation() {
        const ROTATED_SEED: [u8; 32] = [9; 32];
        let predecessor_signed = sign_attestation(&attestation(), "local-key-1", &SEED).unwrap();
        let predecessor = predecessor_signed.chain_entry().clone();
        let prepared = ReceiptAttestationV1::build(
            issued_at(),
            ProducerIdentity::new("vestige-core", "2.3.0", "rotation-test").unwrap(),
            AttestationChainPosition::Successor(&predecessor),
            "synaptic-capture-v1",
            RedactionSafeDecisionProjectionV1::SynapticCapture {
                direction: CaptureDirection::Forward,
                evaluated_count: 1,
                captured_count: 1,
                withheld_count: 0,
            },
            ["private-memory"],
        )
        .unwrap();
        let successor =
            sign_attestation(prepared.attestation(), "rotated-key", &ROTATED_SEED).unwrap();
        let rotated_key = TrustedSigningKey {
            key_id: "rotated-key".to_string(),
            public_key: SigningKey::from_bytes(&ROTATED_SEED)
                .verifying_key()
                .to_bytes(),
            status: SigningKeyStatus::Active,
            valid_from: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            valid_until: None,
            revoked_at: None,
        };
        let predecessor_anchor = TrustedPredecessorAnchor {
            chain_id: predecessor.chain_id.as_str().to_string(),
            sequence: predecessor.sequence,
            entry_digest: predecessor.entry_digest.clone(),
        };
        let terminal = ExpectedTerminalHead {
            chain_id: predecessor.chain_id.as_str().to_string(),
            sequence: 1,
            entry_digest: successor.entry_digest.clone(),
            public_key_fingerprint: Some(rotated_key.public_key_fingerprint()),
        };
        let report = verify_envelope(
            &successor.envelope,
            &rotated_key,
            &VerificationContext {
                predecessor_anchor: Some(predecessor_anchor),
                expected_terminal_head: Some(terminal),
                ..VerificationContext::default()
            },
        );
        assert!(report.is_anchored_valid(), "{:#?}", report.failures);
        assert!(report.predecessor_anchored);
        assert!(report.terminal_head_matched);
    }

    #[test]
    fn safe_builder_mints_random_ids_nonces_and_fixed_claims() {
        let build = || {
            ReceiptAttestationV1::build(
                issued_at(),
                ProducerIdentity::new("vestige-core", "2.3.0", "builder-test").unwrap(),
                AttestationChainPosition::Genesis,
                "replay-v1",
                RedactionSafeDecisionProjectionV1::CounterfactualReplayInfluence {
                    baseline_count: 2,
                    counterfactual_count: 1,
                    membership_changed: true,
                    ordering_changed: false,
                    decision_changed: true,
                    withheld_slot_count: 0,
                },
                ["private-memory-a", "private-memory-b"],
            )
            .unwrap()
        };
        let first = build();
        let second = build();
        assert_ne!(first.attestation.receipt_id, second.attestation.receipt_id);
        assert_ne!(
            first.attestation.operation.opaque_run_id,
            second.attestation.operation.opaque_run_id
        );
        assert_ne!(first.disclosures[0].nonce, first.disclosures[1].nonce);
        assert_eq!(
            first.attestation.predicate.claim_boundary,
            ReceiptClaimBoundary::ControlledReplayInfluenceNotRealWorldCausality
        );
        let payload =
            String::from_utf8(canonical_attestation_bytes(first.attestation()).unwrap()).unwrap();
        assert!(!payload.contains("private-memory"));
    }

    #[test]
    fn missing_disclosure_is_neutral_without_matching_verified_erasure() {
        let attestation = attestation();
        let disclosure = disclosure();
        let expected = disclosure.evidence_commitment();
        assert_eq!(
            verify_disclosure(&attestation, expected.evidence_slot.as_str(), None),
            DisclosureVerification::MissingDisclosure
        );
        let proof = VerifiedDisclosureErasure::after_verified_local_erasure(
            disclosure.receipt_id.clone(),
            disclosure.evidence_slot.clone(),
            expected.commitment,
            payload_digest(b"verified-erasure-proof"),
        )
        .unwrap();
        assert_eq!(
            verify_disclosure_with_erasure_proof(
                &attestation,
                disclosure.evidence_slot.as_str(),
                None,
                Some(&proof),
            ),
            DisclosureVerification::UnavailableAfterVerifiedErasure
        );
    }

    #[test]
    fn unicode_noncharacters_and_resource_exhaustion_inputs_are_rejected() {
        let canonical = canonical_attestation_bytes(&attestation()).unwrap();
        let encoded = String::from_utf8(canonical)
            .unwrap()
            .replace("98f302d", "bad\u{fdd0}build");
        let value: Value = serde_json::from_str(&encoded).unwrap();
        let payload = serde_json_canonicalizer::to_vec(&value).unwrap();
        let envelope = sign_raw(RECEIPT_ATTESTATION_PAYLOAD_TYPE_V1, &payload, "local-key-1");
        let report = verify_envelope(&envelope, &trusted_key(), &VerificationContext::default());
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            VerificationFailure::PayloadJsonPolicyViolation { reason }
                if reason.contains("noncharacters")
        )));

        let mut too_many = sign_attestation(&attestation(), "local-key-1", &SEED)
            .unwrap()
            .envelope;
        too_many.signatures = vec![too_many.signatures[0].clone(); MAX_DSSE_SIGNATURES + 1];
        let report = verify_envelope(&too_many, &trusted_key(), &VerificationContext::default());
        assert!(
            report
                .failures
                .iter()
                .any(|failure| matches!(failure, VerificationFailure::TooManySignatures { .. }))
        );
    }
}
