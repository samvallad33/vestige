//! Frame-kind registry (append-only, never renumber).
//!
//! Payloads serialize with borsh (
//!
//! Allocation:
//! - `1..=7` strata-gate records (`Propose, Gate, Effect, Gap, LessonAlarm,
//!   Canary, Alert`).
//! - `31..=38` (0x1F/0x26) the strata-migrate record family (GENESIS, NODE,
//!   EDGE, FSRS_REVIEW, TOMBSTONE, SUPERSESSION, CHECKPOINT, PARAMS) and
//!   `46` its MIGRATION_RECEIPT. Migration frames live only in migration
//!   logs, so the store's decimal `38` (TOOL_CALL) never shares a log with
//!   the migration family's `0x26`; the migration receipt is decimal 46 in
//!   both registries by design (PR-0a).
//! - `32/33` the store's own data frames (`STORE_WRITE`, `STORE_CHECKPOINT`).
//! - `34..=52` the 4.0 receipt/record registry defined below.
//!
//! New kinds ALWAYS take the next free number; a kind is never renumbered or
//! reused. Payloads are borsh over ints and bytes only, and every receipt
//! payload carries a `params_hash: [u8; 32]` and an `as_of: (u64 seq, i64
//! valid_time_us)` header so any verdict is replayable at its decision point.

/// RECALL_RECEIPT: one handle walk (produced by PR 1 `recall_walk`).
pub const KIND_RECALL_RECEIPT: u8 = 34;
/// WALK_RECEIPT: one causal walk/1 from structured seeds (PR 3).
pub const KIND_WALK_RECEIPT: u8 = 35;
/// ADMISSION_RECEIPT: one ingest decision (PR 2).
pub const KIND_ADMISSION_RECEIPT: u8 = 36;
/// CLAIM_VERDICT: the deterministic verdict over one CLAIM (PR 4).
pub const KIND_CLAIM_VERDICT: u8 = 37;
/// TOOL_CALL: posted before host-tool dispatch (PR 4).
pub const KIND_TOOL_CALL: u8 = 38;
/// TOOL_RESULT: posted after host-tool dispatch (PR 4).
pub const KIND_TOOL_RESULT: u8 = 39;
/// DERIVE: a compaction derivation over exact member ids (PR 7).
pub const KIND_DERIVE: u8 = 40;
/// COUNTERFACTUAL_RECEIPT: a walk re-run with edges excluded (PR 3).
pub const KIND_COUNTERFACTUAL_RECEIPT: u8 = 41;
/// SELFTEST_RECEIPT: a planted-cause scratch selftest (PR 7).
pub const KIND_SELFTEST_RECEIPT: u8 = 42;
/// IMPACT_RECEIPT: the retire/purge blast-radius fixed point (PR 8).
pub const KIND_IMPACT_RECEIPT: u8 = 43;
/// CONFLICT_RECEIPT: two live claims on one exact slot (PR 9).
pub const KIND_CONFLICT_RECEIPT: u8 = 44;
/// GHOSTLINK_RECEIPT: structural bridging pairs (PR 10).
pub const KIND_GHOSTLINK_RECEIPT: u8 = 45;
/// MIGRATION_RECEIPT: the read-only SQLite → STRATA migration seal (PR 0a).
pub const KIND_MIGRATION_RECEIPT: u8 = 46;
/// PARAMS: one signed decision-knob set (PR 0c).
pub const KIND_PARAMS: u8 = 47;
/// CLAIM: an agent's declared, receipt-cited claim (PR 4).
pub const KIND_CLAIM: u8 = 48;
/// SESSION_MARK: a pinned session anchor written by the host (PR 1).
pub const KIND_SESSION_MARK: u8 = 49;
/// REVIEW: one graded review event over a node (PR 6).
pub const KIND_REVIEW: u8 = 50;
/// INTENTION_FIRED: a structured intention trigger matched (PR 11f).
pub const KIND_INTENTION_FIRED: u8 = 51;
/// JOB: a periodic-scan record naming the input seq range it covered (PR 2).
pub const KIND_JOB: u8 = 52;

/// Caller-supplied decision point (H6): the log seq the request sees plus
/// its valid-time in microseconds. Nothing below the MCP handler reads a
/// clock; the handler fills `valid_time_us` from the request.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    borsh::BorshSerialize,
    borsh::BorshDeserialize,
    PartialOrd,
    Ord,
)]
pub struct AsOf {
    /// Log head seq the request is answered at.
    pub seq: u64,
    /// Valid time in microseconds since the Unix epoch.
    pub valid_time_us: i64,
}

impl Default for AsOf {
    fn default() -> Self {
        Self {
            seq: 0,
            valid_time_us: 0,
        }
    }
}

/// Common header of every receipt payload: the PARAMS hash that governed the
/// decision and the decision point it was made at.
#[derive(Debug, Clone, PartialEq, Eq, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct ReceiptHeader {
    /// blake3 of the canonical PARAMS payload in force.
    pub params_hash: [u8; 32],
    /// Caller-supplied decision point.
    pub as_of: AsOf,
}

/// PARAMS frame payload: one signed decision-knob set. Knobs are integers
/// (or integer tables); changing any knob appends a NEW PARAMS frame whose
/// hash then appears in every receipt that used it. Env vars carry only
/// secrets and paths, never knobs (H7).
#[derive(Debug, Clone, PartialEq, Eq, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct ParamsRecord {
    /// Common receipt header (as_of = when the set was appended).
    pub header: ReceiptHeader,
    /// Parameter set id, e.g. `walk/1`, `ghostlink/1`, `lesson-floor/1`.
    pub params_id: String,
    /// Knobs in canonical (sorted-key) order: `(key, int_value)` pairs.
    /// Table-valued knobs flatten to `key.idx` keys.
    pub knobs: Vec<(String, i64)>,
}

impl ParamsRecord {
    /// Canonical blake3 of this parameter set (the `params_hash` receipts
    /// cite). Hashes the borsh encoding with `knobs` sorted, so insertion
    /// order never changes the hash.
    pub fn hash(&self) -> [u8; 32] {
        let mut canonical = self.clone();
        canonical.header = ReceiptHeader {
            params_hash: [0u8; 32],
            as_of: AsOf::default(),
        };
        canonical.knobs.sort();
        let bytes = borsh::to_vec(&canonical).expect("borsh encode params");
        *blake3::hash(&bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_hash_is_canonical_over_knobs() {
        let a = ParamsRecord {
            header: ReceiptHeader {
                params_hash: [0; 32],
                as_of: AsOf {
                    seq: 1,
                    valid_time_us: 100,
                },
            },
            params_id: "walk/1".into(),
            knobs: vec![("k".into(), 3), ("hub_cap".into(), 200)],
        };
        let mut b = a.clone();
        b.knobs.reverse();
        b.header.as_of = AsOf {
            seq: 99,
            valid_time_us: 999,
        };
        // Insertion order and as_of never change the knob hash.
        assert_eq!(a.hash(), b.hash());

        let mut c = a.clone();
        c.knobs[0].1 = 4;
        assert_ne!(a.hash(), c.hash(), "a changed knob changes the hash");
    }

    #[test]
    fn kind_numbers_are_stable() {
        // The registry is append-only; these pins are the append-only test.
        assert_eq!(KIND_RECALL_RECEIPT, 34);
        assert_eq!(KIND_WALK_RECEIPT, 35);
        assert_eq!(KIND_ADMISSION_RECEIPT, 36);
        assert_eq!(KIND_CLAIM_VERDICT, 37);
        assert_eq!(KIND_TOOL_CALL, 38);
        assert_eq!(KIND_TOOL_RESULT, 39);
        assert_eq!(KIND_DERIVE, 40);
        assert_eq!(KIND_COUNTERFACTUAL_RECEIPT, 41);
        assert_eq!(KIND_SELFTEST_RECEIPT, 42);
        assert_eq!(KIND_IMPACT_RECEIPT, 43);
        assert_eq!(KIND_CONFLICT_RECEIPT, 44);
        assert_eq!(KIND_GHOSTLINK_RECEIPT, 45);
        assert_eq!(KIND_MIGRATION_RECEIPT, 46);
        assert_eq!(KIND_PARAMS, 47);
        assert_eq!(KIND_CLAIM, 48);
        assert_eq!(KIND_SESSION_MARK, 49);
        assert_eq!(KIND_REVIEW, 50);
        assert_eq!(KIND_INTENTION_FIRED, 51);
        assert_eq!(KIND_JOB, 52);
    }

    #[test]
    fn asof_orders_by_seq_then_valid_time() {
        let a = AsOf {
            seq: 1,
            valid_time_us: 5,
        };
        let b = AsOf {
            seq: 1,
            valid_time_us: 6,
        };
        let c = AsOf {
            seq: 2,
            valid_time_us: 0,
        };
        assert!(a < b && b < c);
    }
}
