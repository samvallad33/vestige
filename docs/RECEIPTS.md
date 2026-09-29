# Receipts — the 4.0 frame-kind registry

Every mutation and every decision-informing answer in Vestige 4.0 lands as a
frame in a signed, append-only STRATA log. A mutation goes
**PROPOSE → GATE → EFFECT → data frame**: the deterministic gate admits or
refuses, the data frame lands only on `Allow`, and the refusal itself is also
recorded. Frames are borsh-encoded over ints and bytes only — no floats, no
wall clock, no HashMaps in hashed state (H6).

Source of truth: `crates/strata-store/src/kinds.rs` (registry),
`crates/strata-gate/src/record.rs` (gate records),
`crates/strata-store/src/store.rs` (admission path),
`crates/strata-migrate/src/records.rs` (migration family),
`crates/strata-store/src/admission.rs` (admission receipts).

## Kind-byte allocation

| Range | Owner |
|---|---|
| `0` | Reserved ("unknown") by the log layer. |
| `1..=7` | strata-gate records (see below). |
| `0x1F..=0x26` (decimal 31..=38) | strata-migrate record family — lives only in migration logs, so it never shares a log with gate records. |
| `32` / `33` | The store's own data frames: `STORE_WRITE` (32), `STORE_CHECKPOINT` (33). |
| `34..=52` | The 4.0 receipt/record registry (table below). |

The registry is append-only: a kind is never renumbered or reused, and a test
(`registry_append_only` in `crates/vestige-mcp/tests/forbidden_mechanisms.rs`)
pins every number.

## The common receipt header

Every receipt payload carries a `ReceiptHeader`:

| Field | Type | Meaning |
|---|---|---|
| `params_hash` | `[u8; 32]` | blake3 of the canonical PARAMS set(s) in force — the knobs the decision ran under. |
| `as_of` | `(seq u64, valid_time_us i64)` | The caller-supplied decision point (log seq + valid time). Nothing below the MCP handler reads a clock. |

Any verdict is therefore replayable at its decision point: re-run the log
through that `as_of` under the cited PARAMS and you get the same answer.

## Gate records (kinds 1..=7)

| # | Kind | Payload fields | Role |
|---|---|---|---|
| 1 | `PROPOSE` | `action_hash`, `action_kind` (WRITE=0 / RETIRE=1 / GRANT=2 / EFFECT=3), `params_hash`, `context: Vec<u64>` | The model proposes; nothing changes yet. |
| 2 | `GATE` | `propose_seq`, `verdict` (Allow=0 / Deny=1 / Hold=2), `policy_hash` (blake3 of the policy), `inputs` | The deterministic checker's decision. |
| 3 | `EFFECT` | `propose_seq`, `gate_seq`, `action_hash`, `payload_digest` | A landed change; only appendable when an admitting GATE exists earlier in the log. |
| 4 | `GAP` | gap record | Structural violation surfaced by the gate sweep (orphan effect, read-without-receipt, duty-seq hole). |
| 5 | `LESSON_ALARM` | `propose_seq`, `lesson_id`, `retention_milli` | A lesson whose retention fell below the forget floor. |
| 6 | `CANARY` | `canary_id` | A planted canary id. |
| 7 | `ALERT` | alert record | A canary was referenced by the record at `reader_seq`; descendants auto-hold. |

The store's default policy holds every `RETIRE` action (supersession) and
allows everything else under a blast-radius cap
(`default_policy()` in `crates/strata-store/src/store.rs`).

## The 4.0 receipt/record registry (kinds 34..=52)

"Producing PR" is the series entry that owns the kind (from the registry
comments in `kinds.rs`). Fields are listed from the payload struct where one
exists at this HEAD; kinds whose payload struct ships with a later PR carry
only the `ReceiptHeader` contract until then.

| # | Kind | Producing PR / surface | Key fields | Status at this HEAD |
|---|---|---|---|---|
| 34 | `RECALL_RECEIPT` | PR 1 — `recall` handle walk | header + walk payload (PR 1) | Registry entry; payload lands with PR 1 |
| 35 | `WALK_RECEIPT` | PR 3 — `causal_walk` | header + walk payload (PR 3) | Registry entry; payload lands with PR 3 |
| 36 | `ADMISSION_RECEIPT` | PR 2 — every ingest decision (`ingest_admitted`) | `header`, `input_identity` (blake3 content identity), `outcome` (Created / Reinforced / Superseded / Corrected / Held / Refused), `node_id`, `other_id`, `rule_ids` (reason codes), `event_source` | **Written by shipped code** — exactly one per ingest item, in every outcome |
| 37 | `CLAIM_VERDICT` | PR 4 — deterministic verdict over a CLAIM | header + verdict payload (PR 4) | Registry entry; payload lands with PR 4 |
| 38 | `TOOL_CALL` | PR 4 — posted before host-tool dispatch | header + call payload (PR 4) | Registry entry; payload lands with PR 4 |
| 39 | `TOOL_RESULT` | PR 4 — posted after host-tool dispatch | header + result payload (PR 4) | Registry entry; payload lands with PR 4 |
| 40 | `DERIVE` | PR 7 — compaction derivation over exact member ids | header + derive payload (PR 7) | Registry entry; payload lands with PR 7 |
| 41 | `COUNTERFACTUAL_RECEIPT` | PR 3 — walk re-run with edges excluded | header + replay payload (PR 3) | Registry entry; payload lands with PR 3 |
| 42 | `SELFTEST_RECEIPT` | PR 7 — planted-cause scratch selftest | header + selftest payload (PR 7) | Registry entry; payload lands with PR 7 |
| 43 | `IMPACT_RECEIPT` | PR 8 — retire/purge blast-radius fixed point | header + impact payload (PR 8) | Registry entry; payload lands with PR 8 |
| 44 | `CONFLICT_RECEIPT` | PR 9 — two live claims on one exact slot | header + conflict payload (PR 9) | Registry entry; payload lands with PR 9 |
| 45 | `GHOSTLINK_RECEIPT` | PR 10 — structural bridging pairs | header + bridge payload (PR 10) | Registry entry; payload lands with PR 10 |
| 46 | `MIGRATION_RECEIPT` | PR 0a — `vestige migrate-to-strata` (see [MIGRATING-TO-4.0.md](MIGRATING-TO-4.0.md)) | `record_version`, `body` (see below), `verifying_key: [u8; 32]`, `signature: [u8; 64]`, `checksum: [u8; 32]` | **Written by shipped code** — final frame of every migration log |
| 47 | `PARAMS` | PR 0c — one signed decision-knob set | `header`, `params_id` (e.g. `walk/1`), `knobs: Vec<(String, i64)>` | **Written by shipped code** (`StrataStore::append_params`) |
| 48 | `CLAIM` | PR 4 — an agent's declared, receipt-cited claim | header + claim payload (PR 4) | Registry entry; payload lands with PR 4 |
| 49 | `SESSION_MARK` | PR 1 — a pinned session anchor written by the host | header + anchor payload (PR 1) | Registry entry; payload lands with PR 1 |
| 50 | `REVIEW` | PR 6 — one graded review event over a node | header + review payload (PR 6) | Registry entry; payload lands with PR 6 |
| 51 | `INTENTION_FIRED` | PR 11f — a structured intention trigger matched | header + trigger payload (PR 11f) | Registry entry; payload lands with PR 11f |
| 52 | `JOB` | PR 2 — a periodic scan naming the input seq range it covered | `header`, `scan` (e.g. `dedup-scan/1`), `from_seq`, `to_seq` | **Written by shipped code** (`maybe_scan_job` after every N ingest-family frames, default N = 1000) |

### ADMISSION_RECEIPT reason codes

`rule_ids` carries integer reason codes (`crates/strata-store/src/admission.rs`):
`1` identity match (reinforced), `2` target not found / not exact, `3` target
not live at `as_of`, `4` cross-scope, `5` cycle, `6` policy hold (default for
retire), `7` source re-derive (same source key, strictly later
`source_updated_at` — the only sanctioned automatic supersede).

Admission dedup is identity-only: `identity = blake3::derive_key("vestige
4 content-identity", nfc(trim(collapse_ws(content))))`. Lookalikes with no
recorded causal relation never link, merge, rank, or conflict — there is no
similarity term anywhere in the decision.

### MIGRATION_RECEIPT body (kind 46)

`ReceiptBody` fields (`crates/strata-migrate/src/records.rs`):
`source_blake3_before` and `source_blake3_after` (equal in any sealed receipt
— the migrator refuses to seal otherwise), `schema_version`, `envelope_head`,
`counts` (per-table source row counts, sorted by table name),
`dropped_vectors` (vector values never read; vectors do not survive into
STRATA), `dropped_columns` (source columns that did not ride into the log),
`signing_key_id` (`vestige-migrate-receipt-v1`).

The ed25519 signature over `borsh(body)` is the authorship proof, made by a
random key stored OUTSIDE the log (`receipt-signing.key`, 0600, next to the
destination log); the log carries only the verifying key. The plain blake3
`checksum` over the body is tamper-evidence, not authorship.

## Every receipt is gate-admitted

Receipts and PARAMS frames do not bypass the gate: `StrataStore::append_receipt`
and `StrataStore::append_params` run the same PROPOSE → GATE → EFFECT
transaction as writes (`crates/strata-store/src/store.rs`, `admit_frame`).

- The data frame lands only when the gate says `Allow`.
- On reopen, the store replays the log and honors recorded verdicts;
  `rederive_verdicts()` re-derives every verdict under the pinned policy.
- A data frame whose payload digest no admitted EFFECT cites is counted as an
  orphan write and ignored — a forged frame cannot become state.

## How verification works

Three layers, all fail-closed:

1. **Log chain (StrataLog).** Frames are chain-hashed and segments sealed with
   signed trailers. `StrataLog::open` halts on a flipped sealed byte, on
   truncation, or on a wrong signing key (`strata.key`). `verify_tail` plus a
   full segment re-scan re-check the chain.
2. **Store replay (strata-store).** Every open replays the log into derived
   state and verifies the checkpoint chain against the `store.meta` anchor
   (`verify_checkpoint_chain`). `state_digest()` is proven bit-identical
   across open/close/open cycles.
3. **External verifier (strata-verify).** The standalone binary
   `strata-verify <dir>` recomputes everything and trusts no summary state:
   - kernel side — checkpoint chain links, the external head anchor, every
     per-event hash, seq monotonicity, and the state root at every checkpoint
     boundary recomputed via `verify_with_head` plus an independent explicit
     fold;
   - gate side — dense frame seqs, the last frame hash against the anchor,
     every stored GATE verdict re-derived bit-for-bit under the stored policy,
     and a structural sweep that must come back empty;
   - for a directory whose log carries a MIGRATION_RECEIPT (kind 46) —
     `verify_migrated_log`: chain check, receipt blake3 checksum, receipt
     ed25519 signature under the in-log verifying key, and a frame-count
     replay against the receipt's per-table counts (a dropped frame fails even
     with the signing key, because the counts no longer match).

Every failure names the offending seq (or checkpoint `log_seq` + index), so a
tampered byte is reported against the exact record that broke.
