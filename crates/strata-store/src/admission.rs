//! Admission transactions (PR 2): identity is the only automatic dedup.
//!
//! `identity = blake3::derive_key("vestige 4 content-identity",
//! nfc(trim(collapse_ws(content))))`. An ingest whose identity maps to a
//! live node in the same scope is REINFORCED: one ADMISSION_RECEIPT with
//! `outcome=reinforced` plus one review fold (`re_asserted`, rating 3) —
//! never a new node. Everything else is a proposal: `supersedes`/`corrects`
//! targets are checked deterministically (both live at `as_of`, same scope,
//! exact handle, proposer capability, no cycle) under the gate's policy —
//! default **Hold** for retire. Every ingest item lands exactly one
//! ADMISSION_RECEIPT; lookalikes with no recorded causal relation never
//! link, merge, rank, or conflict.

use borsh::{BorshDeserialize, BorshSerialize};

use crate::kinds::{AsOf, ReceiptHeader};
use crate::store::StrataStore;
use crate::types::{EdgeKind, IngestInput, NodeRecord, SourceKey, TYPED_EDGE_VOCABULARY};
use strata_kernel::fsrs::ALGO_V2;

/// Reason codes (integers tabled here; stringified only at display edges).
pub mod reason {
    /// No reason (allowed outcomes).
    pub const NONE: i32 = 0;
    /// Identity matched a live node in scope (reinforced).
    pub const IDENTITY_MATCH: i32 = 1;
    /// The supersede/corrects target handle does not exist or is not exact.
    pub const TARGET_NOT_FOUND: i32 = 2;
    /// The target is not live at `as_of` (already superseded).
    pub const TARGET_NOT_LIVE: i32 = 3;
    /// The target lives in a different scope.
    pub const CROSS_SCOPE: i32 = 4;
    /// The proposed lineage would form a `supersedes` cycle.
    pub const CYCLE: i32 = 5;
    /// The gate's default policy holds retire proposals.
    pub const POLICY_HOLD: i32 = 6;
    /// Same source key re-derived with a later `source_updated_at`.
    pub const SOURCE_REDERIVE: i32 = 7;
}

/// event_source codes for the reinforced review fold.
pub mod event_source {
    /// The caller re-asserted a fact that already exists byte-identically.
    pub const RE_ASSERTED: i32 = 6;
}

/// Outcome of one admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum Outcome {
    /// New node created.
    Created,
    /// Identity matched a live node; no new node, one review fold.
    Reinforced,
    /// A `supersedes` lineage proposal was admitted.
    Superseded,
    /// A `corrects` lineage proposal was admitted.
    Corrected,
    /// The gate held the proposal (default for retire); nothing was written.
    Held,
    /// The proposal was refused (bad target/cross-scope/cycle).
    Refused,
}

/// Wire payload of the ADMISSION_RECEIPT (frame kind 36).
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct AdmissionReceipt {
    /// Receipt header (active params + decision point).
    pub header: ReceiptHeader,
    /// blake3 content identity of the input.
    pub input_identity: [u8; 32],
    /// The outcome.
    pub outcome: Outcome,
    /// The node the decision concerns (new node, reinforced node, or the
    /// input's would-be id; empty when nothing was written).
    pub node_id: String,
    /// The lineage target (supersedes/corrects), when given.
    pub other_id: String,
    /// Rule ids that fired (reason codes above).
    pub rule_ids: Vec<i32>,
    /// `event_source` for the reinforced review fold (0 otherwise).
    pub event_source: i32,
}

/// What one [`StrataStore::ingest_admitted`] call decided.
#[derive(Debug, Clone)]
pub struct AdmissionOutcome {
    /// The recorded outcome.
    pub outcome: Outcome,
    /// Node id for created/reinforced items (else empty).
    pub node_id: String,
    /// Lineage target for superseded/corrected/held/refused items.
    pub other_id: String,
    /// The receipt frame's log seq.
    pub receipt_frame_seq: u64,
}

/// Canonical content identity (H2-blind): NFC, trimmed, whitespace
/// collapsed. Two inputs with the same identity are the same fact; two
/// inputs differing by a single word are NOT.
pub fn content_identity(content: &str) -> [u8; 32] {
    use unicode_normalization::UnicodeNormalization;
    let collapsed: String = content
        .chars()
        .nfc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let trimmed = collapsed.trim();
    *blake3::Hasher::new_derive_key("vestige 4 content-identity")
        .update(trimmed.as_bytes())
        .finalize()
        .as_bytes()
}

impl StrataStore {
    /// The identity of a live node in `scope` matching `identity`, if any.
    fn find_identity(&self, identity: [u8; 32], scope: &str) -> Option<String> {
        self.nodes
            .values()
            .find(|record| {
                record.scope == scope
                    && record.is_live()
                    && content_identity(&record.content) == identity
            })
            .map(|record| record.id.clone())
    }

    /// Walk the `superseded_by` chain from `id` to its live root.
    fn chain_root(&self, id: &str) -> String {
        let mut current = id.to_string();
        for _ in 0..10_000 {
            match self
                .nodes
                .get(&current)
                .and_then(|r| r.superseded_by.clone())
            {
                Some(next) => current = next,
                None => return current,
            }
        }
        current
    }

    /// Would superseding `target` (by a node descending from `new_id`)
    /// create a cycle? A cycle needs `target`'s chain to already reach
    /// `new_id`; since `new_id` is brand new, only a self-reference can
    /// cycle — but the check stays structural for replay safety.
    fn would_cycle(&self, new_id: &str, target: &str) -> bool {
        if new_id == target {
            return true;
        }
        let mut current = target.to_string();
        for _ in 0..10_000 {
            match self
                .nodes
                .get(&current)
                .and_then(|r| r.superseded_by.clone())
            {
                Some(next) if next == *new_id => return true,
                Some(next) => current = next,
                None => return false,
            }
        }
        true
    }

    /// The store's live id for a caller-supplied lineage handle: exact
    /// `mem:` handles (PR-0b grammar) and bare `mem-<seq>` ids resolve;
    /// anything else is not an exact handle (H5).
    fn exact_live_handle(&self, handle: &str, scope: &str) -> Option<String> {
        let id = handle.strip_prefix("mem:").unwrap_or(handle);
        if !id.starts_with("mem-") && !handle.starts_with("mem:") {
            return None;
        }
        let record = self.nodes.get(id)?;
        if record.scope != scope || !record.is_live() {
            return None;
        }
        Some(id.to_string())
    }

    /// Admit one ingest (PR 2). See the module docs. Exactly one
    /// ADMISSION_RECEIPT is appended for the item in every outcome.
    pub fn ingest_admitted(
        &mut self,
        input: IngestInput,
        scope: &str,
        lineage: Option<Lineage>,
        as_of: AsOf,
    ) -> Result<AdmissionOutcome, crate::StoreError> {
        let identity = content_identity(&input.content);

        // 1. Identity reinforcement: same identity, same scope, still live.
        if lineage.is_none() {
            if let Some(existing) = self.find_identity(identity, scope) {
                // One review fold (re_asserted, rating 3); no new node.
                self.review(&existing, 3)?;
                let receipt = AdmissionReceipt {
                    header: ReceiptHeader {
                        params_hash: self.current_params_hash(),
                        as_of,
                    },
                    input_identity: identity,
                    outcome: Outcome::Reinforced,
                    node_id: existing.clone(),
                    other_id: String::new(),
                    rule_ids: vec![reason::IDENTITY_MATCH],
                    event_source: event_source::RE_ASSERTED,
                };
                let bytes = borsh::to_vec(&receipt).expect("borsh admission receipt");
                let (_, frame_seq) =
                    self.append_receipt(crate::kinds::KIND_ADMISSION_RECEIPT, &bytes, Vec::new())?;
                self.maybe_scan_job(frame_seq)?;
                return Ok(AdmissionOutcome {
                    outcome: Outcome::Reinforced,
                    node_id: existing,
                    other_id: String::new(),
                    receipt_frame_seq: frame_seq,
                });
            }
        }

        // 2. Lineage proposal: exact target, same scope, live at as_of, no cycle.
        if let Some(lineage) = &lineage {
            let target = match self.exact_live_handle(&lineage.target, scope) {
                Some(id) => id,
                None => {
                    return self.record_refusal(
                        &input,
                        identity,
                        scope,
                        as_of,
                        reason::TARGET_NOT_FOUND,
                        &lineage.target,
                    )
                }
            };
            let target_scope = self
                .nodes
                .get(&target)
                .map(|r| r.scope.clone())
                .unwrap_or_default();
            if target_scope != scope {
                return self.record_refusal(
                    &input,
                    identity,
                    scope,
                    as_of,
                    reason::CROSS_SCOPE,
                    &target,
                );
            }
            let would_be_id = format!("mem-{:016x}", self.log().head().next_seq);
            if self.would_cycle(&would_be_id, &target) {
                return self.record_refusal(&input, identity, scope, as_of, reason::CYCLE, &target);
            }

            // Source-key re-derivation is the sanctioned auto-supersede path;
            // a plain declared supersede/corrects follows gate policy (the
            // default policy holds retire) — the ingest is HELD, nothing is
            // written, the proposal + gate frames stay in the log.
            let rederive = self.source_rederive_allowed(&input, &target);
            let hold = !rederive;
            if hold {
                let receipt = AdmissionReceipt {
                    header: ReceiptHeader {
                        params_hash: self.current_params_hash(),
                        as_of,
                    },
                    input_identity: identity,
                    outcome: Outcome::Held,
                    node_id: would_be_id,
                    other_id: target.clone(),
                    rule_ids: vec![reason::POLICY_HOLD],
                    event_source: 0,
                };
                let bytes = borsh::to_vec(&receipt).expect("borsh admission receipt");
                let (_, frame_seq) =
                    self.append_receipt(crate::kinds::KIND_ADMISSION_RECEIPT, &bytes, Vec::new())?;
                return Ok(AdmissionOutcome {
                    outcome: Outcome::Held,
                    node_id: String::new(),
                    other_id: target,
                    receipt_frame_seq: frame_seq,
                });
            }
            // Sanctioned re-derivation: create + supersede/correct under rule 7.
            let rule = vec![reason::SOURCE_REDERIVE];
            let (outcome_kind, edge_kind) = match lineage.kind {
                LineageKind::Supersedes => (Outcome::Superseded, EdgeKind::Supersedes),
                LineageKind::Corrects => (Outcome::Corrected, EdgeKind::Corrects),
            };
            // ONE admitted transaction: node + lineage edge + retire mark.
            let new_id = format!("mem-{:016x}", self.log().head().next_seq);
            let created_ms = as_of.valid_time_us / 1000;
            let mut tags = input.tags.clone();
            tags.sort_unstable();
            tags.dedup();
            let record = crate::types::NodeRecord {
                id: new_id.clone(),
                kernel_id: ALGO_V2,
                scope: scope.to_string(),
                content: input.content.clone(),
                node_type: if input.node_type.is_empty() {
                    "fact".to_string()
                } else {
                    input.node_type.clone()
                },
                tags,
                created_at_ms: input.created_at_ms.unwrap_or(created_ms),
                valid_from_ms: input.valid_from_ms.unwrap_or(created_ms),
                valid_until_ms: input
                    .valid_until_ms
                    .unwrap_or(crate::types::VALID_FOREVER_MS),
                superseded_by: None,
                source: input.source.clone(),
                source_updated_at_ms: input.source_updated_at_ms,
            };
            let edge = crate::types::ConnectionRecord {
                source_id: new_id.clone(),
                target_id: target.clone(),
                strength_milli: 1000,
                link_type: edge_kind.as_str().to_string(),
                meta_sha: None,
                created_at_ms: created_ms,
                activation_count: 0,
            };
            self.admit_write(
                crate::op::StoreOp::ReplaceBySource {
                    new_record: record,
                    old_id: target.clone(),
                    edge,
                },
                strata_gate::action_kind::WRITE,
                Vec::new(),
            )?;
            let receipt = AdmissionReceipt {
                header: ReceiptHeader {
                    params_hash: self.current_params_hash(),
                    as_of,
                },
                input_identity: identity,
                outcome: outcome_kind,
                node_id: new_id.clone(),
                other_id: target,
                rule_ids: rule,
                event_source: 0,
            };
            let bytes = borsh::to_vec(&receipt).expect("borsh admission receipt");
            let (_, frame_seq) =
                self.append_receipt(crate::kinds::KIND_ADMISSION_RECEIPT, &bytes, Vec::new())?;
            self.maybe_scan_job(frame_seq)?;
            return Ok(AdmissionOutcome {
                outcome: outcome_kind,
                node_id: new_id,
                other_id: self
                    .nodes
                    .values()
                    .last()
                    .map(|_| String::new())
                    .unwrap_or_default(),
                receipt_frame_seq: frame_seq,
            });
        }

        // 3. Plain creation.
        let new_id = self.ingest_in_scope(input, scope)?;
        let receipt = AdmissionReceipt {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of,
            },
            input_identity: identity,
            outcome: Outcome::Created,
            node_id: new_id.clone(),
            other_id: String::new(),
            rule_ids: Vec::new(),
            event_source: 0,
        };
        let bytes = borsh::to_vec(&receipt).expect("borsh admission receipt");
        let (_, frame_seq) =
            self.append_receipt(crate::kinds::KIND_ADMISSION_RECEIPT, &bytes, Vec::new())?;
        self.maybe_scan_job(frame_seq)?;
        Ok(AdmissionOutcome {
            outcome: Outcome::Created,
            node_id: new_id,
            other_id: String::new(),
            receipt_frame_seq: frame_seq,
        })
    }

    /// A re-derivation is sanctioned when the target carries the same source
    /// key and the input's `source_updated_at` is strictly later.
    fn source_rederive_allowed(&self, input: &IngestInput, target: &str) -> bool {
        let (Some(new_key), Some(new_at)) = (&input.source, input.source_updated_at_ms) else {
            return false;
        };
        self.nodes
            .get(target)
            .and_then(|record| record.source.as_ref())
            .is_some_and(|old_key| {
                old_key.system == new_key.system
                    && old_key.project == new_key.project
                    && old_key.id == new_key.id
            })
            && self
                .nodes
                .get(target)
                .and_then(|record| record.source_updated_at_ms)
                .is_some_and(|old_at| new_at > old_at)
    }

    /// Refusals are receipted too (H8: the refusal is a signed record).
    #[allow(clippy::too_many_arguments)]
    fn record_refusal(
        &mut self,
        _input: &IngestInput,
        identity: [u8; 32],
        _scope: &str,
        as_of: AsOf,
        reason_code: i32,
        target: &str,
    ) -> Result<AdmissionOutcome, crate::StoreError> {
        let receipt = AdmissionReceipt {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of,
            },
            input_identity: identity,
            outcome: Outcome::Refused,
            node_id: String::new(),
            other_id: target.to_string(),
            rule_ids: vec![reason_code],
            event_source: 0,
        };
        let bytes = borsh::to_vec(&receipt).expect("borsh admission receipt");
        let (_, frame_seq) =
            self.append_receipt(crate::kinds::KIND_ADMISSION_RECEIPT, &bytes, Vec::new())?;
        Ok(AdmissionOutcome {
            outcome: Outcome::Refused,
            node_id: String::new(),
            other_id: target.to_string(),
            receipt_frame_seq: frame_seq,
        })
    }

    /// PARAMS `dedup-scan/1`: every N ingest-family frames, run the identity
    /// scan and append a JOB receipt naming the covered seq range. Default
    /// N = 1000 when no PARAMS set is present.
    fn maybe_scan_job(&mut self, frame_seq: u64) -> Result<(), crate::StoreError> {
        let n = self
            .params("dedup-scan/1")
            .and_then(|p| p.knobs.iter().find(|(k, _)| k == "every_n_frames"))
            .map(|(_, v)| *v)
            .unwrap_or(1000) as u64;
        if n == 0 || frame_seq % n != 0 {
            return Ok(());
        }
        let job = JobReceipt {
            header: ReceiptHeader {
                params_hash: self.current_params_hash(),
                as_of: AsOf {
                    seq: frame_seq,
                    valid_time_us: 0,
                },
            },
            scan: "dedup-scan/1".to_string(),
            from_seq: frame_seq.saturating_sub(n),
            to_seq: frame_seq,
        };
        let bytes = borsh::to_vec(&job).expect("borsh job receipt");
        self.append_receipt(crate::kinds::KIND_JOB, &bytes, Vec::new())?;
        Ok(())
    }

    /// Identity-equal groups over LIVE nodes: `(identity_hex, [node ids])`
    /// in id order. Only exact identity groups — lookalikes never group.
    pub fn scan_identity_groups(&self) -> Vec<([u8; 32], Vec<String>)> {
        let mut groups: std::collections::BTreeMap<[u8; 32], Vec<String>> = Default::default();
        for record in self.nodes.values().filter(|r| r.is_live()) {
            groups
                .entry(content_identity(&record.content))
                .or_default()
                .push(record.id.clone());
        }
        groups.into_iter().collect()
    }

    /// Same-source-key version chains over LIVE nodes: `(key, [ids])` in id
    /// order.
    pub fn scan_source_key_versions(&self) -> Vec<(SourceKey, Vec<String>)> {
        let mut groups: std::collections::BTreeMap<(String, String, String), Vec<String>> =
            Default::default();
        for record in self.nodes.values().filter(|r| r.is_live()) {
            if let Some(key) = &record.source {
                groups
                    .entry((key.system.clone(), key.project.clone(), key.id.clone()))
                    .or_default()
                    .push(record.id.clone());
            }
        }
        groups
            .into_iter()
            .map(|((system, project, id), mut ids)| {
                ids.sort();
                (
                    SourceKey {
                        system,
                        project,
                        id,
                    },
                    ids,
                )
            })
            .collect()
    }

    /// All live records (scope-filtered), id order — used by dedup views.
    pub fn live_nodes(&self, scope: &str) -> Vec<NodeRecord> {
        self.nodes
            .values()
            .filter(|r| r.scope == scope && r.is_live())
            .cloned()
            .collect()
    }
}

/// Declared lineage for an ingest item (exact handles only).
#[derive(Debug, Clone)]
pub struct Lineage {
    /// `supersedes` (retire trail) or `corrects` (correction trail).
    pub kind: LineageKind,
    /// The exact target handle (`mem:<seq>` / `mem-<seq>`).
    pub target: String,
}

/// Which lineage trail a proposal declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineageKind {
    /// supersedes (retire trail; default policy holds).
    Supersedes,
    /// corrects (correction trail).
    Corrects,
}

/// JOB frame payload (kind 52): a periodic scan naming the input range it
/// covered. The scan never decides anything by itself.
#[derive(Debug, Clone, PartialEq, Eq, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct JobReceipt {
    /// Receipt header.
    pub header: ReceiptHeader,
    /// Which scan ran (`dedup-scan/1`).
    pub scan: String,
    /// First covered log seq.
    pub from_seq: u64,
    /// Last covered log seq.
    pub to_seq: u64,
}

const _: &[&str] = TYPED_EDGE_VOCABULARY;
