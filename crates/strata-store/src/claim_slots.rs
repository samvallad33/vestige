//! Claim slots (PR 9): a contradiction is exactly two live claims on the
//! same exact `(subject_handle, field_id)` slot whose canonical values
//! differ and whose validity overlaps at the decision point `as_of`.
//!
//! Ported from `vestige-spacetime/src/leg5_claim_lattice.rs` with the PR-9
//! fixes:
//! - no `Freeform` slot kind (untyped prose is never a slot; word-level
//!   disagreement is not a contradiction — H2),
//! - numeric tolerance is an integer in the field's unit (no floats — H6),
//! - no confidence field (authority comes from receipts, not a float),
//! - `subject_handle` and `value_bytes` are used verbatim: no trimming, no
//!   case folding (`"Prod"` != `"prod"`),
//! - `BTreeMap` everywhere (no `HashMap` in hashed state — H6).
//!
//! The lattice's quarantine/PR machinery is intentionally not ported: a
//! conflict is recorded as a signed CONFLICT_RECEIPT (kind 44) and the
//! later conflicting claim is admitted with a [`ClaimOutcomeKind::Held`]
//! outcome (the admission.rs Held shape) until an explicit
//! `corrects`/`supersedes` resolves it.
//!
//! Frame kinds: the claim record itself is a KIND_CLAIM (48) receipt frame
//! appended through the gate via [`StrataStore::append_receipt`]; conflicts
//! are KIND_CONFLICT_RECEIPT (44). A frame only lands on an `Allow`
//! verdict, so every CLAIM frame in the log is gate-admitted.
//!
//! Determinism (H6): borsh payloads of ints/bytes/strings only, no floats,
//! no wall clock (caller-supplied `AsOf`), and every listing ends in a
//! total order — `(subject bytes ASC, field_id ASC, a.seq ASC, b.seq ASC)`.

use std::collections::BTreeMap;

use borsh::{BorshDeserialize, BorshSerialize};

use crate::error::StoreError;
use crate::kinds::{AsOf, ReceiptHeader, KIND_CLAIM, KIND_CONFLICT_RECEIPT};
use crate::store::StrataStore;

/// PARAMS set id holding the claim-field table
/// (`field.<id>.kind` / `field.<id>.tolerance` knobs).
pub const CLAIM_FIELDS_PARAMS_ID: &str = "claim-fields/1";

/// Field-table kind code: one exact value; any differing value conflicts.
pub const FIELD_KIND_EXCLUSIVE: i64 = 0;
/// Field-table kind code: multi-valued; differing members are additive and
/// never conflict.
pub const FIELD_KIND_SET: i64 = 1;
/// Field-table kind code: an integer with an integer tolerance knob
/// (`field.<id>.tolerance`, default 0), in the field's unit.
pub const FIELD_KIND_NUMERIC: i64 = 2;
/// Field-table kind code: dotted numeric version components
/// (`parse_version` + `encode_version` canonical bytes).
pub const FIELD_KIND_VERSION: i64 = 3;

/// Per-field comparison semantics. Port of the lattice's `ClaimSemantics`
/// without `Freeform` and without float tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSemantics {
    /// One exact value (a handle, a status): differing bytes conflict.
    Exclusive,
    /// Set-valued: differing members are additive, never a conflict.
    Set,
    /// Integer value; two values within `tolerance` (in the field's unit)
    /// are the same value.
    Numeric {
        /// Maximum absolute difference that still counts as the same value.
        tolerance: i64,
    },
    /// Dotted numeric version components; differing components conflict.
    Version,
}

fn decode_semantics(kind_code: i64, tolerance: i64) -> Option<FieldSemantics> {
    match kind_code {
        FIELD_KIND_EXCLUSIVE => Some(FieldSemantics::Exclusive),
        FIELD_KIND_SET => Some(FieldSemantics::Set),
        FIELD_KIND_NUMERIC => Some(FieldSemantics::Numeric {
            // A negative tolerance is meaningless; clamp, never error mid-table.
            tolerance: tolerance.max(0),
        }),
        FIELD_KIND_VERSION => Some(FieldSemantics::Version),
        _ => None,
    }
}

/// The claim-field table decoded from the current `claim-fields/1` PARAMS
/// set: `field_id -> semantics`. Knob keys are `field.<id>.kind` (codes
/// above) and `field.<id>.tolerance` (numeric only, default 0). Field ids
/// are append-only in spirit: removing a previously used id makes
/// `conflicts_at` fail closed.
pub fn field_table(store: &StrataStore) -> BTreeMap<i64, FieldSemantics> {
    let mut table: BTreeMap<i64, FieldSemantics> = BTreeMap::new();
    let Some(params) = store.params(CLAIM_FIELDS_PARAMS_ID) else {
        return table;
    };
    for (key, kind_code) in &params.knobs {
        if let Some(id) = key
            .strip_prefix("field.")
            .and_then(|rest| rest.strip_suffix(".kind"))
        {
            if let (Ok(id), Some(sem)) = (id.parse::<i64>(), decode_semantics(*kind_code, 0)) {
                table.insert(id, sem);
            }
        }
    }
    for (key, tolerance) in &params.knobs {
        if let Some(id) = key
            .strip_prefix("field.")
            .and_then(|rest| rest.strip_suffix(".tolerance"))
        {
            if let Ok(id) = id.parse::<i64>() {
                if let Some(FieldSemantics::Numeric { tolerance: knob }) = table.get_mut(&id) {
                    *knob = (*tolerance).max(0);
                }
            }
        }
    }
    table
}

/// Canonical value encodings per field kind.
///
/// - Numeric: borsh of one `i64` (the field's unit).
/// - Version: borsh of one `Vec<u64>` (the dotted components).
/// - Exclusive / Set: the caller's exact bytes (handles, statuses); the
///   store never normalizes them.
pub fn encode_numeric(value: i64) -> Vec<u8> {
    borsh::to_vec(&value).expect("borsh i64 is infallible")
}

/// Canonical version value bytes (see [`encode_numeric`]).
pub fn encode_version(parts: &[u64]) -> Vec<u8> {
    borsh::to_vec(&parts.to_vec()).expect("borsh Vec<u64> is infallible")
}

/// Port of the lattice's version parser, stricter: every non-digit run
/// splits, every part must parse, and an empty component list is rejected
/// (fail closed) instead of silently skipped.
pub fn parse_version(value: &str) -> Option<Vec<u64>> {
    let parts: Vec<u64> = value
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!parts.is_empty()).then_some(parts)
}

fn decode_numeric(bytes: &[u8]) -> Option<i64> {
    i64::try_from_slice(bytes).ok()
}

fn decode_version(bytes: &[u8]) -> Option<Vec<u64>> {
    Vec::<u64>::try_from_slice(bytes).ok()
}

/// Reject non-canonical value bytes at admission (fail closed, H8).
fn ensure_canonical(sem: &FieldSemantics, value: &[u8]) -> Result<(), StoreError> {
    match sem {
        FieldSemantics::Exclusive | FieldSemantics::Set => Ok(()),
        FieldSemantics::Numeric { .. } => {
            if decode_numeric(value).is_some() {
                Ok(())
            } else {
                Err(StoreError::InvalidInput(
                    "numeric field value must be canonical borsh i64 (encode_numeric)".into(),
                ))
            }
        }
        FieldSemantics::Version => {
            if decode_version(value).is_some() {
                Ok(())
            } else {
                Err(StoreError::InvalidInput(
                    "version field value must be canonical borsh Vec<u64> \
                     (parse_version + encode_version)"
                        .into(),
                ))
            }
        }
    }
}

/// Do two values on the same slot conflict under `sem`? Equal bytes never
/// conflict; the field type decides the rest. `value_bytes` are exact
/// bytes: there is no case folding, trimming, or parsing of prose (H2/H5).
pub fn values_conflict(sem: &FieldSemantics, a: &[u8], b: &[u8]) -> bool {
    if a == b {
        return false;
    }
    match sem {
        FieldSemantics::Exclusive => true,
        FieldSemantics::Set => false,
        FieldSemantics::Numeric { tolerance } => {
            match (decode_numeric(a), decode_numeric(b)) {
                (Some(x), Some(y)) => {
                    // i128 cannot overflow on any i64 pair.
                    let diff = i128::from(x) - i128::from(y);
                    diff.abs() > i128::from(*tolerance)
                }
                // Non-canonical bytes never reach a live claim
                // (record_claim rejects them); treat them as conflicting,
                // never as equal.
                _ => true,
            }
        }
        FieldSemantics::Version => match (decode_version(a), decode_version(b)) {
            (Some(x), Some(y)) => x != y,
            _ => true,
        },
    }
}

/// Caller-side claim input (the durable record adds the receipt header).
#[derive(Debug, Clone)]
pub struct ClaimInput {
    /// Exact subject handle; used verbatim (no trim, no case fold, H5).
    pub subject_handle: String,
    /// Field id from the `claim-fields/1` PARAMS table.
    pub field_id: i64,
    /// Canonical value bytes per the field's kind.
    pub value_bytes: Vec<u8>,
    /// Validity start (caller-supplied decision point, H6).
    pub valid_from: AsOf,
    /// The receipt this claim cites as its source.
    pub source_receipt: [u8; 32],
}

/// KIND_CLAIM (48) frame payload: one agent-declared claim on one exact
/// slot, gate-admitted through `append_receipt`.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ClaimRecord {
    /// Receipt header (active params + the recording decision point).
    pub header: ReceiptHeader,
    /// Exact subject handle, verbatim bytes.
    pub subject_handle: String,
    /// Field id from the PARAMS `claim-fields/1` table.
    pub field_id: i64,
    /// Canonical value bytes per the field kind.
    pub value_bytes: Vec<u8>,
    /// Validity start.
    pub valid_from: AsOf,
    /// The receipt this claim cites as its source.
    pub source_receipt: [u8; 32],
}

/// One live claim with its CLAIM frame's log seq (the claim's `seq`).
#[derive(Debug, Clone)]
pub struct LiveClaim {
    /// Log seq of the CLAIM frame.
    pub seq: u64,
    /// The decoded claim record.
    pub claim: ClaimRecord,
}

/// KIND_CONFLICT_RECEIPT (44) payload: two live claims on one exact slot
/// with differing canonical values, recorded at admission time.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ConflictReceipt {
    /// Receipt header (active params + the detection decision point).
    pub header: ReceiptHeader,
    /// The exact subject handle of the contested slot.
    pub subject: String,
    /// The contested field id.
    pub field_id: i64,
    /// CLAIM frame seq of the earlier (incumbent) claim.
    pub a_seq: u64,
    /// CLAIM frame seq of the later (held) claim.
    pub b_seq: u64,
}

/// One derived conflict pair at an `as_of` (see `StrataStore::conflicts_at`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictPair {
    /// The exact subject handle of the contested slot.
    pub subject: String,
    /// The contested field id.
    pub field_id: i64,
    /// Earlier claim's CLAIM frame seq (`a.seq < b.seq`).
    pub a_seq: u64,
    /// Later claim's CLAIM frame seq.
    pub b_seq: u64,
}

/// Outcome kind of [`StrataStore::record_claim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimOutcomeKind {
    /// No live claim conflicts; the claim is authoritative on its slot.
    Recorded,
    /// The later claim conflicts with an earlier live claim: it is admitted
    /// (the CLAIM frame landed) but HELD — not authoritative — until an
    /// explicit `corrects`/`supersedes` resolves the slot. The conflict is
    /// receipted (CONFLICT_RECEIPT 44).
    Held,
}

/// What one [`StrataStore::record_claim`] call decided.
#[derive(Debug, Clone)]
pub struct ClaimOutcome {
    /// The outcome kind.
    pub outcome: ClaimOutcomeKind,
    /// Log seq of the landed CLAIM frame.
    pub claim_seq: u64,
    /// One `(receipt_frame_seq, a_seq, b_seq)` per emitted CONFLICT_RECEIPT,
    /// in conflict order (`a.seq` ascending); empty unless Held.
    pub conflicts: Vec<(u64, u64, u64)>,
}

fn borsh_vec<T: BorshSerialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    borsh::to_vec(value).map_err(|e| StoreError::Encode(e.to_string()))
}

fn hash32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

impl StrataStore {
    /// Every CLAIM frame in the log with its frame seq. Frames only land on
    /// an `Allow` verdict, so everything here is gate-admitted; a frame
    /// that fails to decode is log corruption and fails closed.
    fn claim_frames(&self) -> Result<Vec<(u64, ClaimRecord)>, StoreError> {
        let mut out = Vec::new();
        for frame in self.log().read_frames(1)? {
            if frame.kind == KIND_CLAIM {
                let record = ClaimRecord::try_from_slice(&frame.payload).map_err(|e| {
                    StoreError::Verify(format!("undecodable CLAIM frame at seq {}: {e}", frame.seq))
                })?;
                out.push((frame.seq, record));
            }
        }
        Ok(out)
    }

    /// All live claims at `as_of`, in the canonical order
    /// `(subject bytes ASC, field_id ASC, seq ASC)`.
    ///
    /// A claim is live at `as_of` when its CLAIM frame is in the log by
    /// `as_of.seq` (`seq <= as_of.seq`) and its validity has started
    /// (`valid_from.seq <= as_of.seq`); retirement via explicit
    /// `corrects`/`supersedes` narrows this in a later PR.
    pub fn live_claims_at(&self, as_of: AsOf) -> Result<Vec<LiveClaim>, StoreError> {
        let mut live: Vec<LiveClaim> = self
            .claim_frames()?
            .into_iter()
            .filter(|(seq, claim)| *seq <= as_of.seq && claim.valid_from.seq <= as_of.seq)
            .map(|(seq, claim)| LiveClaim { seq, claim })
            .collect();
        live.sort_by(|x, y| {
            (x.claim.subject_handle.as_bytes(), x.claim.field_id, x.seq).cmp(&(
                y.claim.subject_handle.as_bytes(),
                y.claim.field_id,
                y.seq,
            ))
        });
        Ok(live)
    }

    /// All conflict pairs over live claims at `as_of`: same exact
    /// `(subject_handle, field_id)`, differing canonical values per the
    /// field's semantics, both valid at `as_of`. Ordered
    /// `(subject bytes ASC, field_id ASC, a.seq ASC, b.seq ASC)`.
    pub fn conflicts_at(&self, as_of: AsOf) -> Result<Vec<ConflictPair>, StoreError> {
        let live = self.live_claims_at(as_of)?;
        let table = field_table(self);
        let mut groups: BTreeMap<(String, i64), Vec<&LiveClaim>> = BTreeMap::new();
        for entry in &live {
            groups
                .entry((entry.claim.subject_handle.clone(), entry.claim.field_id))
                .or_default()
                .push(entry);
        }
        let mut pairs = Vec::new();
        for ((subject, field_id), members) in &groups {
            let sem = table.get(field_id).ok_or_else(|| {
                StoreError::InvalidInput(format!(
                    "field_id {field_id} has live claims but left the {CLAIM_FIELDS_PARAMS_ID} \
                     table (field ids are append-only)"
                ))
            })?;
            for (i, a) in members.iter().enumerate() {
                for b in members.iter().skip(i + 1) {
                    if values_conflict(sem, &a.claim.value_bytes, &b.claim.value_bytes) {
                        pairs.push(ConflictPair {
                            subject: subject.clone(),
                            field_id: *field_id,
                            a_seq: a.seq,
                            b_seq: b.seq,
                        });
                    }
                }
            }
        }
        pairs.sort_by(|x, y| {
            (x.subject.as_bytes(), x.field_id, x.a_seq, x.b_seq).cmp(&(
                y.subject.as_bytes(),
                y.field_id,
                y.a_seq,
                y.b_seq,
            ))
        });
        Ok(pairs)
    }

    /// Record one claim on an exact slot (gate-admitted, H7).
    ///
    /// Fails closed when the subject handle is empty or padded, the field
    /// id is not in the `claim-fields/1` PARAMS table, or the value bytes
    /// are not canonical for the field kind. When an earlier live claim
    /// holds a conflicting value on the same exact slot at `as_of`, the
    /// CLAIM frame still lands (H9: assertions are appended, never
    /// deleted) but the outcome is [`ClaimOutcomeKind::Held`] — the claim
    /// is not authoritative — and one CONFLICT_RECEIPT (44) is emitted per
    /// conflicting incumbent, citing both CLAIM frame seqs
    /// (`a_seq` = incumbent, `b_seq` = the new claim). This is the
    /// admission.rs Held shape: a recorded outcome, not an error.
    pub fn record_claim(
        &mut self,
        claim: ClaimInput,
        as_of: AsOf,
    ) -> Result<ClaimOutcome, StoreError> {
        if claim.subject_handle.is_empty() || claim.subject_handle.trim() != claim.subject_handle {
            return Err(StoreError::InvalidInput(
                "subject_handle must be a non-empty exact handle (no padding)".into(),
            ));
        }
        let semantics = field_table(self).remove(&claim.field_id).ok_or_else(|| {
            StoreError::InvalidInput(format!(
                "field_id {} is not in the {} PARAMS table",
                claim.field_id, CLAIM_FIELDS_PARAMS_ID
            ))
        })?;
        ensure_canonical(&semantics, &claim.value_bytes)?;

        // Conflicting incumbents, a.seq ascending (live_claims_at order).
        let incumbents: Vec<u64> = self
            .live_claims_at(as_of)?
            .into_iter()
            .filter(|live| {
                live.claim.subject_handle == claim.subject_handle
                    && live.claim.field_id == claim.field_id
                    && values_conflict(&semantics, &live.claim.value_bytes, &claim.value_bytes)
            })
            .map(|live| live.seq)
            .collect();

        let record = ClaimRecord {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of,
            },
            subject_handle: claim.subject_handle.clone(),
            field_id: claim.field_id,
            value_bytes: claim.value_bytes,
            valid_from: claim.valid_from,
            source_receipt: claim.source_receipt,
        };
        let bytes = borsh_vec(&record)?;
        let (_, claim_seq) = self.append_receipt(KIND_CLAIM, &bytes, Vec::new())?;

        let mut conflicts = Vec::with_capacity(incumbents.len());
        for a_seq in incumbents {
            let receipt = ConflictReceipt {
                header: ReceiptHeader {
                    params_hash: self.current_params_hash(),
                    as_of,
                },
                subject: claim.subject_handle.clone(),
                field_id: claim.field_id,
                a_seq,
                b_seq: claim_seq,
            };
            let receipt_bytes = borsh_vec(&receipt)?;
            let (_, receipt_seq) =
                self.append_receipt(KIND_CONFLICT_RECEIPT, &receipt_bytes, Vec::new())?;
            conflicts.push((receipt_seq, a_seq, claim_seq));
        }

        Ok(ClaimOutcome {
            outcome: if conflicts.is_empty() {
                ClaimOutcomeKind::Recorded
            } else {
                ClaimOutcomeKind::Held
            },
            claim_seq,
            conflicts,
        })
    }
}

/// Could-not-have-known proof (`receipt{action:"cnhk"}` store-side core):
/// proves `claim.valid_from.seq > decision_as_of_seq`, i.e. the claim's
/// validity started strictly after the decision's decision point, so the
/// decision could not have known it. Rejects (fail closed) when the claim
/// predates (or is contemporaneous with) the decision.
///
/// Pure and deterministic: no store, no clock. `claim_ref` commits to the
/// exact borsh CLAIM record (so a verifier can re-fetch the claim by
/// `claim_seq`, re-hash it, and re-check the ordering);
/// `decision_ref` is the caller's commitment to the decision receipt.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct CnhkProof {
    /// blake3 of the borsh-encoded CLAIM record.
    pub claim_ref: [u8; 32],
    /// Caller's commitment to the decision receipt.
    pub decision_ref: [u8; 32],
    /// Log seq of the CLAIM frame.
    pub claim_seq: u64,
    /// The decision's `as_of.seq`.
    pub decision_as_of_seq: u64,
    /// blake3 (derive-key `vestige strata cnhk/1`) over the four fields
    /// above; makes any tampering detectable without the originals.
    pub checksum: [u8; 32],
}

fn cnhk_checksum(
    claim_ref: &[u8; 32],
    decision_ref: &[u8; 32],
    claim_seq: u64,
    decision_as_of_seq: u64,
) -> [u8; 32] {
    let bytes = borsh::to_vec(&(*claim_ref, *decision_ref, claim_seq, decision_as_of_seq))
        .expect("borsh cnhk fields are infallible");
    *blake3::Hasher::new_derive_key("vestige strata cnhk/1")
        .update(&bytes)
        .finalize()
        .as_bytes()
}

impl CnhkProof {
    /// Recompute and compare the proof's checksum.
    pub fn checksum_valid(&self) -> bool {
        self.checksum
            == cnhk_checksum(
                &self.claim_ref,
                &self.decision_ref,
                self.claim_seq,
                self.decision_as_of_seq,
            )
    }
}

/// Build the could-not-have-known proof. Rejects when
/// `claim.valid_from.seq <= decision_as_of_seq` (the claim was knowable at
/// the decision).
pub fn cnhk(
    claim: &ClaimRecord,
    claim_seq: u64,
    decision_as_of_seq: u64,
    decision_ref: [u8; 32],
) -> Result<CnhkProof, String> {
    if claim.valid_from.seq <= decision_as_of_seq {
        return Err(format!(
            "claim predates the decision: valid_from.seq {} <= decision as_of.seq {}",
            claim.valid_from.seq, decision_as_of_seq
        ));
    }
    let claim_ref = hash32(&borsh::to_vec(claim).map_err(|e| e.to_string())?);
    let checksum = cnhk_checksum(&claim_ref, &decision_ref, claim_seq, decision_as_of_seq);
    Ok(CnhkProof {
        claim_ref,
        decision_ref,
        claim_seq,
        decision_as_of_seq,
        checksum,
    })
}
