//! Receipt-bound claims (PR 4): TOOL_CALL (38) / TOOL_RESULT (39) /
//! CLAIM (48) records plus the pure claim checker and its signed
//! CLAIM_VERDICT (37) receipt.
//!
//! A claim is admissible iff EVERY cited id resolves to a TOOL_RESULT frame
//! that is at or before the claim's seq, in the same session, whose
//! `tool`/`target_handle` appears in the claim kind's exact family table,
//! and whose outcome passes (`exit_status == 0` OR all test outcomes equal
//! `pass`). Every match is exact string equality over structured fields
//! (H5): a tool named `cargo-test-helper` never satisfies a `tests_pass`
//! family entry of `cargo`. The claim kind is a FIELD of the CLAIM record,
//! never inferred from prose (H2); no LLM sits anywhere in this path (H3)
//! and every failure denies or gaps, never allows (H8).
//!
//! The family table is plain `(claim_kind, tool)` pairs supplied by the
//! caller — PARAMS-shaped structured data, never regex, never prose. The
//! spec's "family table is PARAMS" intent is honored by keeping the table
//! out of code decisions (it is data the caller pins, and its canonical
//! hash is bound into every verdict), but a string-valued table cannot ride
//! in a [`crate::kinds::ParamsRecord`] (knobs are `(String, i64)`), so the
//! signed-PARAMS home for it is a later record (see the PR report).
//!
//! GAP records: the spec's "missing citation writes a GAP (strata-gate kind
//! 4)" is carried as the typed `GapRecord` (strata-gate's kind-4 payload
//! vocabulary: `ReadNoReceipt`) inside the signed CLAIM_VERDICT receipt's
//! `gap` field. Landing a kind-4 DATA frame instead would bypass the
//! gate-log cache on the live append path (`gate_log.rs` only learns about
//! gate frames appended through the `EventLog` trait), so live and reopened
//! gate-seq spaces would diverge and bit-identical replay (H6) would break.
//! The GAP therefore rides in the gate-admitted, signed verdict receipt;
//! promoting it to a standalone frame needs a cache-consistent store append
//! path first.

use borsh::{BorshDeserialize, BorshSerialize};

use crate::error::StoreError;
use crate::kinds::{AsOf, ReceiptHeader};
use crate::store::StrataStore;
use strata_gate::{DutyKind, GapDetail, GapRecord};

/// The only test outcome that counts as passing (exact, lowercase).
pub const OUTCOME_PASS: &str = "pass";

/// Claim kind of an advisory-veto annotation: a record a model may append
/// OUTSIDE the decision path. It never enters a verdict — the checker
/// consumes only TOOL_RESULT frames — and an unknown claim kind always
/// denies, so the annotation can never allow anything (H3/H8).
pub const ADVISORY_VETO_KIND: &str = "advisory_veto";

/// Verdict rule codes (integers, tabled here; stringified only at display
/// edges). Allow codes are positive evidence; deny codes name the failed
/// admissibility condition.
pub mod rule {
    /// An admissible citation passed via `exit_status == 0`.
    pub const PASS_EXIT_ZERO: i32 = 1;
    /// An admissible citation passed via all outcomes `pass`.
    pub const PASS_OUTCOMES: i32 = 2;
    /// The claim cited nothing (GAP + deny).
    pub const NO_CITATION: i32 = 3;
    /// A cited id does not resolve to a TOOL_RESULT frame.
    pub const CITATION_NOT_FOUND: i32 = 4;
    /// A cited TOOL_RESULT frame is newer than the claim's seq.
    pub const CITATION_FUTURE: i32 = 5;
    /// A cited TOOL_RESULT belongs to a different session.
    pub const CROSS_SESSION: i32 = 6;
    /// A cited result's tool/target is not in the claim kind's family table.
    pub const FAMILY_MISMATCH: i32 = 7;
    /// A cited result failed (`exit_status != 0` and an outcome != `pass`).
    pub const FAILED_RESULT: i32 = 8;
    /// The claim kind has no family-table entries (unknown or unconfigured).
    pub const UNKNOWN_CLAIM_KIND: i32 = 9;
}

/// Display name for a [`rule`] code (display edge only; decisions are the
/// integer codes).
pub fn rule_name(code: i32) -> &'static str {
    match code {
        rule::PASS_EXIT_ZERO => "pass_exit_zero",
        rule::PASS_OUTCOMES => "pass_outcomes",
        rule::NO_CITATION => "no_citation",
        rule::CITATION_NOT_FOUND => "citation_not_found",
        rule::CITATION_FUTURE => "citation_future",
        rule::CROSS_SESSION => "cross_session",
        rule::FAMILY_MISMATCH => "family_mismatch",
        rule::FAILED_RESULT => "failed_result",
        rule::UNKNOWN_CLAIM_KIND => "unknown_claim_kind",
        _ => "unknown_rule",
    }
}

/// TOOL_CALL frame payload (kind 38): posted BEFORE host-tool dispatch.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ToolCallRecord {
    /// Receipt header. The store overwrites `params_hash` with the active
    /// PARAMS at append; `as_of` is the caller-supplied decision point.
    pub header: ReceiptHeader,
    /// Session the call belongs to (claims must cite the same session).
    pub session: String,
    /// Caller's dispatch id; TOOL_CALL and its TOOL_RESULT share it.
    pub call_id: String,
    /// Exact host tool name (no normalization, H5).
    pub tool: String,
    /// Exact target handle the tool acts on (may be empty).
    pub target_handle: String,
    /// blake3 over the canonical arguments (arguments are never stored).
    pub args_hash: [u8; 32],
}

/// TOOL_RESULT frame payload (kind 39): posted AFTER host-tool dispatch.
/// This is the receipt a claim must cite.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ToolResultRecord {
    /// Receipt header (see [`ToolCallRecord::header`]).
    pub header: ReceiptHeader,
    /// Session the call belongs to (must match the citing claim's).
    pub session: String,
    /// The dispatch id of the paired TOOL_CALL.
    pub call_id: String,
    /// Exact host tool name (family tables match this exactly).
    pub tool: String,
    /// Exact target handle the tool acted on (family tables may match this).
    pub target_handle: String,
    /// Process exit status (0 = success).
    pub exit_status: i32,
    /// Per-test outcomes as `(test, outcome)` pairs; `outcome` compares
    /// exactly against [`OUTCOME_PASS`]. Sorted copies only — never a map.
    pub test_outcomes: Vec<(String, String)>,
}

/// CLAIM frame payload (kind 48): the agent's declared, receipt-cited
/// claim. `claim_kind` is a FIELD — prose is never consulted (H2).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ClaimRecord {
    /// Receipt header. The store overwrites `params_hash` at append;
    /// `as_of.seq` is the claim's decision point: citations must resolve to
    /// TOOL_RESULT frames at or before it.
    pub header: ReceiptHeader,
    /// Session the claim was made in (citations must match exactly).
    pub session: String,
    /// One of the configured claim kinds (e.g. `tests_pass`, `build_ok`,
    /// `lint_ok`, `typecheck_ok`, `deployed`, `file_written`); exact string,
    /// never inferred from prose.
    pub claim_kind: String,
    /// Exact target the claim is about.
    pub target_handle: String,
    /// TOOL_RESULT frame seqs this claim is bound to.
    pub cited_record_ids: Vec<u64>,
}

/// CLAIM_VERDICT frame payload (kind 37): `{claim, allow, rule_ids[], gap?}`
/// — the deterministic checker's decision, bound to the claim digest, the
/// family table hash, and the decision point, so replay re-derives it bit
/// for bit.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct ClaimVerdictRecord {
    /// Receipt header (as_of = the claim's decision point).
    pub header: ReceiptHeader,
    /// blake3(borsh(ClaimRecord)) of the checked claim.
    pub claim_digest: [u8; 32],
    /// Log seq of the CLAIM frame this verdict decides.
    pub claim_frame_seq: u64,
    /// The verdict.
    pub allow: bool,
    /// [`rule`] codes that fired (sorted, deduped).
    pub rule_ids: Vec<i32>,
    /// The typed GAP record (strata-gate kind-4 vocabulary, `ReadNoReceipt`)
    /// when the claim cited nothing. See the module docs for why the GAP
    /// rides in this receipt rather than a standalone frame.
    pub gap: Option<GapRecord>,
    /// Canonical hash of the exact family table the verdict was made over.
    pub family_table_hash: [u8; 32],
}

impl ClaimVerdictRecord {
    /// Display projection of the GAP: `(duty name, reader_seq, dangling_id)`,
    /// or `None`. Lets CLI/hook surfaces render the gap without depending on
    /// strata-gate types directly.
    pub fn gap_summary(&self) -> Option<(&'static str, u64, u64)> {
        self.gap.as_ref().map(|gap| match gap.detail {
            GapDetail::ReadNoReceipt {
                reader_seq,
                dangling_id,
            } => ("read_no_receipt", reader_seq, dangling_id),
            // The claim checker only ever writes ReadNoReceipt gaps; the
            // other duty kinds come from the structural sweep.
            GapDetail::OrphanEffect { effect_seq, .. } => ("orphan_effect", effect_seq, 0),
            GapDetail::DutySeqGap { expected, .. } => ("duty_seq_gap", expected, 0),
        })
    }
}

/// Outcome of the pure checker: the verdict plus the rules that fired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// Is the claim admissible?
    pub allow: bool,
    /// [`rule`] codes that fired (sorted, deduped).
    pub rule_ids: Vec<i32>,
    /// A GAP record must be written with this verdict (missing citation).
    pub gap_required: bool,
}

/// The initial family table: exact `(claim_kind, tool/argv[0])` pairs. This
/// is a starting point owned by the caller — pass an explicit table (the
/// CLI's `--family kind:tool`) to override it; a kind with no entries
/// denies everything (`unknown_claim_kind`), which is the fail-closed
/// default for `deployed` and any future kind.
pub const DEFAULT_FAMILY_TABLE: &[(&str, &str)] = &[
    ("tests_pass", "cargo"),
    ("tests_pass", "cargo-nextest"),
    ("tests_pass", "npm"),
    ("tests_pass", "pnpm"),
    ("tests_pass", "yarn"),
    ("tests_pass", "pytest"),
    ("tests_pass", "vitest"),
    ("tests_pass", "jest"),
    ("tests_pass", "go"),
    ("build_ok", "cargo"),
    ("build_ok", "npm"),
    ("build_ok", "pnpm"),
    ("build_ok", "yarn"),
    ("build_ok", "go"),
    ("build_ok", "tsc"),
    ("lint_ok", "cargo"),
    ("lint_ok", "clippy"),
    ("lint_ok", "eslint"),
    ("lint_ok", "ruff"),
    ("typecheck_ok", "cargo"),
    ("typecheck_ok", "tsc"),
    ("typecheck_ok", "mypy"),
    ("typecheck_ok", "pyright"),
    ("file_written", "Write"),
    ("file_written", "Edit"),
];

/// Canonical blake3 of a family table: sorted, deduped pairs, borsh, blake3.
/// Insertion order never changes the hash.
pub fn family_table_hash(family_table: &[(String, String)]) -> [u8; 32] {
    let mut pairs = family_table.to_vec();
    pairs.sort();
    pairs.dedup();
    let bytes = borsh::to_vec(&pairs).expect("borsh family table");
    *blake3::hash(&bytes).as_bytes()
}

/// blake3(borsh(ClaimRecord)) — the digest a verdict binds to.
pub fn claim_digest(claim: &ClaimRecord) -> [u8; 32] {
    let bytes = borsh::to_vec(claim).expect("borsh claim");
    *blake3::hash(&bytes).as_bytes()
}

/// The typed GAP written for an uncited claim: a `ReadNoReceipt` gap at the
/// claim's frame seq with dangling id 0 ("cited nothing").
pub fn gap_for(claim_frame_seq: u64) -> GapRecord {
    GapRecord {
        duty: DutyKind::ReadNoReceipt,
        detail: GapDetail::ReadNoReceipt {
            reader_seq: claim_frame_seq,
            dangling_id: 0,
        },
    }
}

/// The pure claim checker (spec `gate-policy/claims-1`). `tool_results` are
/// `(frame_seq, record)` candidates read from the log; `family_table` are
/// exact `(claim_kind, tool)` pairs. A claim is admissible iff every cited
/// id resolves to a TOOL_RESULT that is at or before the claim's seq, in
/// the same session, whose `tool`/`target_handle` is in the claim kind's
/// exact family table, and whose outcome passes (`exit_status == 0` OR all
/// outcomes [`OUTCOME_PASS`]). Pure: same inputs, same verdict, always.
pub fn verify_claim(
    claim: &ClaimRecord,
    tool_results: &[(u64, ToolResultRecord)],
    family_table: &[(String, String)],
) -> Verdict {
    let deny = |rule_ids: Vec<i32>, gap: bool| Verdict {
        allow: false,
        rule_ids,
        gap_required: gap,
    };

    let allowed: Vec<&str> = family_table
        .iter()
        .filter(|(kind, _)| kind == &claim.claim_kind)
        .map(|(_, tool)| tool.as_str())
        .collect();
    if allowed.is_empty() {
        return deny(vec![rule::UNKNOWN_CLAIM_KIND], false);
    }
    if claim.cited_record_ids.is_empty() {
        return deny(vec![rule::NO_CITATION], true);
    }

    let mut rules: Vec<i32> = Vec::new();
    for &id in &claim.cited_record_ids {
        let Some((seq, result)) = tool_results.iter().find(|(frame_seq, _)| *frame_seq == id)
        else {
            return deny(vec![rule::CITATION_NOT_FOUND], false);
        };
        if *seq > claim.header.as_of.seq {
            return deny(vec![rule::CITATION_FUTURE], false);
        }
        if result.session != claim.session {
            return deny(vec![rule::CROSS_SESSION], false);
        }
        // Exact equality on structured fields only (H5): a tool named
        // `cargo-test-helper` is not `cargo`, and no fuzzy fallback exists.
        if !allowed.contains(&result.tool.as_str())
            && !allowed.contains(&result.target_handle.as_str())
        {
            return deny(vec![rule::FAMILY_MISMATCH], false);
        }
        let all_outcomes_pass = result
            .test_outcomes
            .iter()
            .all(|(_, outcome)| outcome == OUTCOME_PASS);
        if result.exit_status != 0 && !all_outcomes_pass {
            return deny(vec![rule::FAILED_RESULT], false);
        }
        rules.push(if result.exit_status == 0 {
            rule::PASS_EXIT_ZERO
        } else {
            rule::PASS_OUTCOMES
        });
    }
    rules.sort_unstable();
    rules.dedup();
    Verdict {
        allow: true,
        rule_ids: rules,
        gap_required: false,
    }
}

/// What one [`StrataStore::check_claim`] call decided and where it landed.
#[derive(Debug, Clone)]
pub struct ClaimCheckOutcome {
    /// Log seq of the appended CLAIM frame.
    pub claim_frame_seq: u64,
    /// Log seq of the appended CLAIM_VERDICT frame.
    pub verdict_frame_seq: u64,
    /// The verdict record as appended (also re-derivable from the log).
    pub verdict: ClaimVerdictRecord,
}

/// One replay-verified verdict: the stored receipt next to the verdict
/// re-derived from the log alone, with byte equality precomputed.
#[derive(Debug, Clone)]
pub struct RederivedVerdict {
    /// Log seq of the CLAIM_VERDICT frame.
    pub frame_seq: u64,
    /// The stored record.
    pub stored: ClaimVerdictRecord,
    /// The record re-derived by replaying the log.
    pub rederived: ClaimVerdictRecord,
    /// borsh bytes of stored and rederived are identical.
    pub bit_identical: bool,
}

impl StrataStore {
    fn encode<T: BorshSerialize>(value: &T) -> Result<Vec<u8>, StoreError> {
        borsh::to_vec(value).map_err(|e| StoreError::Encode(e.to_string()))
    }

    /// Append one gate-admitted TOOL_CALL receipt (kind 38). The record's
    /// `header.params_hash` is bound to the store's active PARAMS; `as_of`
    /// stays caller-supplied. Returns the frame's log seq.
    pub fn record_tool_call(&mut self, record: &ToolCallRecord) -> Result<u64, StoreError> {
        let mut record = record.clone();
        record.header.params_hash = self.current_params_hash();
        let bytes = Self::encode(&record)?;
        let (_, frame_seq) =
            self.append_receipt(crate::kinds::KIND_TOOL_CALL, &bytes, Vec::new())?;
        Ok(frame_seq)
    }

    /// Append one gate-admitted TOOL_RESULT receipt (kind 39) — the receipt
    /// type claims cite. Returns the frame's log seq (the citation id).
    pub fn record_tool_result(&mut self, record: &ToolResultRecord) -> Result<u64, StoreError> {
        let mut record = record.clone();
        record.header.params_hash = self.current_params_hash();
        let bytes = Self::encode(&record)?;
        let (_, frame_seq) =
            self.append_receipt(crate::kinds::KIND_TOOL_RESULT, &bytes, Vec::new())?;
        Ok(frame_seq)
    }

    /// Check one claim: append the CLAIM frame (48), run the pure checker
    /// over every TOOL_RESULT in the log, and append the signed
    /// CLAIM_VERDICT receipt (37) — with the typed GAP when the claim cited
    /// nothing. Deny is a normal outcome (Ok), never an error; only store
    /// failures error.
    ///
    /// The claim's `header.params_hash` is bound to the active PARAMS;
    /// `header.as_of` is the claim's decision point (citations must resolve
    /// at or before it).
    pub fn check_claim(
        &mut self,
        claim: &ClaimRecord,
        family_table: &[(String, String)],
    ) -> Result<ClaimCheckOutcome, StoreError> {
        let results = self.read_tool_results()?;
        let mut claim = claim.clone();
        claim.header.params_hash = self.current_params_hash();
        let decision = verify_claim(&claim, &results, family_table);

        let claim_bytes = Self::encode(&claim)?;
        let (_, claim_frame_seq) =
            self.append_receipt(crate::kinds::KIND_CLAIM, &claim_bytes, Vec::new())?;

        let record = ClaimVerdictRecord {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of: claim.header.as_of,
            },
            claim_digest: claim_digest(&claim),
            claim_frame_seq,
            allow: decision.allow,
            rule_ids: decision.rule_ids,
            gap: decision.gap_required.then(|| gap_for(claim_frame_seq)),
            family_table_hash: family_table_hash(family_table),
        };
        let verdict_bytes = Self::encode(&record)?;
        let (_, verdict_frame_seq) =
            self.append_receipt(crate::kinds::KIND_CLAIM_VERDICT, &verdict_bytes, Vec::new())?;
        Ok(ClaimCheckOutcome {
            claim_frame_seq,
            verdict_frame_seq,
            verdict: record,
        })
    }

    /// Append an advisory-veto annotation (a CLAIM frame with
    /// [`ADVISORY_VETO_KIND`]) — the only record a model may add outside the
    /// decision path. It never enters a verdict: the checker reads only
    /// TOOL_RESULT frames, and the annotation's unknown kind would deny if
    /// it were ever checked.
    pub fn record_advisory_veto(
        &mut self,
        session: &str,
        target_handle: &str,
        as_of: AsOf,
    ) -> Result<u64, StoreError> {
        let claim = ClaimRecord {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of,
            },
            session: session.to_string(),
            claim_kind: ADVISORY_VETO_KIND.to_string(),
            target_handle: target_handle.to_string(),
            cited_record_ids: Vec::new(),
        };
        let bytes = Self::encode(&claim)?;
        let (_, frame_seq) = self.append_receipt(crate::kinds::KIND_CLAIM, &bytes, Vec::new())?;
        Ok(frame_seq)
    }

    /// Every TOOL_RESULT frame as `(frame_seq, record)`, in log order.
    pub fn read_tool_results(&self) -> Result<Vec<(u64, ToolResultRecord)>, StoreError> {
        let frames = self.log().read_frames(1)?;
        Ok(frames
            .into_iter()
            .filter(|f| f.kind == crate::kinds::KIND_TOOL_RESULT)
            .filter_map(|f| {
                ToolResultRecord::try_from_slice(&f.payload)
                    .ok()
                    .map(|record| (f.seq, record))
            })
            .collect())
    }

    /// Every CLAIM_VERDICT frame as `(frame_seq, record)`, in log order.
    pub fn read_claim_verdicts(&self) -> Result<Vec<(u64, ClaimVerdictRecord)>, StoreError> {
        let frames = self.log().read_frames(1)?;
        Ok(frames
            .into_iter()
            .filter(|f| f.kind == crate::kinds::KIND_CLAIM_VERDICT)
            .filter_map(|f| {
                ClaimVerdictRecord::try_from_slice(&f.payload)
                    .ok()
                    .map(|record| (f.seq, record))
            })
            .collect())
    }

    /// Replay re-derivation: re-run the pure checker over the log alone and
    /// compare each re-derived verdict with the stored receipt byte for
    /// byte. Claims without a verdict frame (advisory annotations) are
    /// skipped; a verdict without its claim frame is tamper evidence.
    pub fn rederive_claim_verdicts(
        &self,
        family_table: &[(String, String)],
    ) -> Result<Vec<RederivedVerdict>, StoreError> {
        let frames = self.log().read_frames(1)?;
        let mut results: Vec<(u64, ToolResultRecord)> = Vec::new();
        let mut claims: std::collections::BTreeMap<u64, ClaimRecord> = Default::default();
        let mut verdicts: Vec<(u64, ClaimVerdictRecord)> = Vec::new();
        for frame in frames {
            match frame.kind {
                crate::kinds::KIND_TOOL_RESULT => {
                    if let Ok(record) = ToolResultRecord::try_from_slice(&frame.payload) {
                        results.push((frame.seq, record));
                    }
                }
                crate::kinds::KIND_CLAIM => {
                    if let Ok(record) = ClaimRecord::try_from_slice(&frame.payload) {
                        claims.insert(frame.seq, record);
                    }
                }
                crate::kinds::KIND_CLAIM_VERDICT => {
                    if let Ok(record) = ClaimVerdictRecord::try_from_slice(&frame.payload) {
                        verdicts.push((frame.seq, record));
                    }
                }
                _ => {}
            }
        }

        let table_hash = family_table_hash(family_table);
        let mut out = Vec::new();
        for (frame_seq, stored) in verdicts {
            let claim = claims.get(&stored.claim_frame_seq).ok_or_else(|| {
                StoreError::Verify(format!(
                    "claim verdict at seq {frame_seq} cites missing CLAIM frame {}",
                    stored.claim_frame_seq
                ))
            })?;
            let decision = verify_claim(claim, &results, family_table);
            let rederived = ClaimVerdictRecord {
                header: stored.header.clone(),
                claim_digest: claim_digest(claim),
                claim_frame_seq: stored.claim_frame_seq,
                allow: decision.allow,
                rule_ids: decision.rule_ids,
                gap: decision
                    .gap_required
                    .then(|| gap_for(stored.claim_frame_seq)),
                family_table_hash: table_hash,
            };
            let bit_identical = Self::encode(&rederived)? == Self::encode(&stored)?;
            out.push(RederivedVerdict {
                frame_seq,
                stored,
                rederived,
                bit_identical,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(kind: &str, session: &str, cited: &[u64]) -> ClaimRecord {
        ClaimRecord {
            header: ReceiptHeader {
                params_hash: [0; 32],
                as_of: AsOf {
                    seq: 100,
                    valid_time_us: 0,
                },
            },
            session: session.to_string(),
            claim_kind: kind.to_string(),
            target_handle: "repo://x".to_string(),
            cited_record_ids: cited.to_vec(),
        }
    }

    fn result(
        seq: u64,
        session: &str,
        tool: &str,
        exit_status: i32,
        outcome: &str,
    ) -> (u64, ToolResultRecord) {
        (
            seq,
            ToolResultRecord {
                header: ReceiptHeader {
                    params_hash: [0; 32],
                    as_of: AsOf {
                        seq,
                        valid_time_us: 0,
                    },
                },
                session: session.to_string(),
                call_id: format!("call-{seq}"),
                tool: tool.to_string(),
                target_handle: "repo://x".to_string(),
                exit_status,
                test_outcomes: vec![("t".to_string(), outcome.to_string())],
            },
        )
    }

    fn family() -> Vec<(String, String)> {
        vec![
            ("tests_pass".to_string(), "cargo".to_string()),
            ("lint_ok".to_string(), "cargo".to_string()),
        ]
    }

    #[test]
    fn exact_family_match_only() {
        let results = vec![
            result(4, "s", "cargo", 0, OUTCOME_PASS),
            result(5, "s", "cargo-test-helper", 0, OUTCOME_PASS),
        ];
        let table = family();
        let ok = verify_claim(&claim("tests_pass", "s", &[4]), &results, &table);
        assert!(ok.allow, "exact family match allows");
        let lookalike = verify_claim(&claim("tests_pass", "s", &[5]), &results, &table);
        assert!(
            !lookalike.allow && lookalike.rule_ids == vec![rule::FAMILY_MISMATCH],
            "a lookalike tool name never satisfies the family table"
        );
    }

    #[test]
    fn pass_is_exit_zero_or_all_outcomes_pass() {
        let table = family();
        // exit 1 but every outcome passes: admissible (the OR branch).
        let results = vec![result(4, "s", "cargo", 1, OUTCOME_PASS)];
        assert!(verify_claim(&claim("tests_pass", "s", &[4]), &results, &table).allow);
        // exit 1 with a failing outcome: denied.
        let results = vec![result(4, "s", "cargo", 1, "fail")];
        let v = verify_claim(&claim("tests_pass", "s", &[4]), &results, &table);
        assert!(!v.allow && v.rule_ids == vec![rule::FAILED_RESULT]);
    }

    #[test]
    fn family_table_hash_is_canonical() {
        let a = vec![
            ("tests_pass".to_string(), "cargo".to_string()),
            ("lint_ok".to_string(), "cargo".to_string()),
        ];
        let mut b = a.clone();
        b.reverse();
        assert_eq!(family_table_hash(&a), family_table_hash(&b));
        let mut c = a.clone();
        c[0].1 = "npm".to_string();
        assert_ne!(family_table_hash(&a), family_table_hash(&c));
    }

    #[test]
    fn uncited_claim_gaps_and_unknown_kind_denies() {
        let table = family();
        let v = verify_claim(&claim("tests_pass", "s", &[]), &[], &table);
        assert!(!v.allow && v.gap_required && v.rule_ids == vec![rule::NO_CITATION]);
        let v = verify_claim(&claim("deployed", "s", &[4]), &[], &table);
        assert!(!v.allow && v.rule_ids == vec![rule::UNKNOWN_CLAIM_KIND]);
    }
}
