# FULL LOOP analysis

Checked Gemini's drop against `main` (`52030d69`) and `origin/feat/strata-mesh-iroh` at `bf561a07e1d867c5cafa8281acdcccbef8a2f816` (commits `ab9d9999` fold, `aed1ed49` iroh sync, `bf561a07` unused-import cleanup). The #404 agent `bc-e26544ea-16dd-5037-b82d-10bd7eafa336` is IDLE. The branch is pushed. Nothing below is inferred from an unpushed tree.

`vestige_core::chain_fold` does not exist on either branch. The fold that landed is `fn fold_chains` in `crates/strata-store/src/mesh.rs` (private), fed by `crates/strata/src/replica.rs` (`ReplicaLog`) and headed by `crates/strata/src/mesh_head.rs` (`SignedHead`). A new crate `crates/strata-mesh` is the iroh transport and SPAKE2 pairing. It is not a provenance receipt type.

Counts below are references the drop expects the repo to already have. Symbols the drop defines itself are marked NEW and are not in the counts.

## Counts

| Set | MATCH | MISMATCH | MISSING |
| --- | ---: | ---: | ---: |
| `chain_fold` contract (21 rows) | 1 | 10 | 10 |
| Other external references | 14 | 8 | 4 |
| **Total** | **15** | **18** | **14** |

Top fixes, in the order they block a verbatim paste:

1. Do not overwrite `hooks/operator-stop-memory.py` or `hooks/settings.stop-memory.fragment.json`. Both are the shipped Operator-stop relay. `packages/vestige-init/test/kernel.test.js` and `memory-setup.test.js` require the repo hook to be byte-identical to the package copy, and `six-tool-memory-hooks.test.js` calls `ingest()` on it. `benchmarks/real_engine/integrity.py` reads the fragment by that filename. The save-gate script lands at `hooks/save-gate.py`. The fragment lands at `hooks/settings.save-gate.fragment.json`.
2. Do not add `ed25519-dalek` 2. The workspace already depends on `ed25519-dalek = "3"` (`SigningKey::from_bytes` is the call the rest of the tree uses). `blake3 = "1"`, `serde`, and `serde_json` are already on `vestige-core`. `biscuit-auth` is absent and is added only on the passport branch, at major 6.
3. `crates/vestige-mcp/src/tools/reflex/mod.rs` does not exist. The module file is `crates/vestige-mcp/src/tools/reflex.rs`, which already has `mod pairing`. `pub mod auto_bind` goes there.
4. Appending Gemini's `pub struct OpenFailure` to `pairing.rs` does not compile. The file already has `pub(super) struct OpenFailure` with fields `sig`, `node`, `tool`, `words`, `agent`, `edited`. The new session types go in `pairing::save_gate` and the names `auto_bind` uses are re-exported. The existing pairing functions stay.
5. `SignedWriterFrame`, `verify_cryptography`, `lamport`, `timestamp_ms`, `EquivocationReceipt`, `WriterChainTip`, and `admitted_frames` are not on the mesh fold. `fold_chains` is private, takes `&[ChainIn]` of store effects, and returns `MeshProjection` (`digest`, `frontier`, `nodes`, `reinforcements`, `conflicts`, `review_counts`). The thin adapter is `vestige_core::chain_fold`. It does not reimplement `fold_chains` and it does not edit `mesh.rs`.

## `chain_fold` contract

| Item Gemini imports | Verdict | Real item |
| --- | --- | --- |
| `WriterId` | MISSING | No newtype. Writer identity is `strata_store::mesh::writer_did(&[u8; 32]) -> String` (`crates/strata-store/src/mesh.rs:64`), the same multicodec as `vestige_core::did_key_from_ed25519_public_key` (`crates/vestige-core/src/actor.rs:140`). `PinnedWriter.verifying_key` and `ChainManifest.writer_did` / `verifying_key` are the grant record. |
| `SignedWriterFrame` | MISSING | `strata::Frame` (`crates/strata/src/format.rs:54`) is `{kind, payload, payload_blake3, prev_frame_hash}`. `strata::FrameRecord` (`format.rs:100`) adds `seq` and `frame_hash`. Neither is a per-frame signed receipt. |
| `writer_id` field | MISSING | Not on `Frame` or `FrameRecord`. The writer key is pinned beside the log (`ReplicaLog`, `writer.pub`). |
| `seq` field | MISMATCH | `FrameRecord.seq: u64`. First acked frame is seq 1. Gemini's tests sign seq 0. |
| `lamport` field | MISSING | No lamport clock on a frame, a head, or `MeshProjection`. |
| `timestamp_ms` field | MISSING | No per-frame timestamp on `Frame` / `FrameRecord` / `SignedHead`. |
| `prev_hash` field | MISMATCH | `Frame.prev_frame_hash`. Inside a segment the first frame's prev is the segment `header_hash`, not 32 zero bytes. |
| `payload_digest` field | MISMATCH | `Frame.payload_blake3`. Algorithm is `blake3(kind \|\| payload)` (`payload_blake3`, `format.rs:116`), not a provenance-domain hash of the JSON bytes. |
| `frame_digest` field | MISMATCH | `FrameRecord.frame_hash`, `blake3` of the borsh wire (`frame_hash`, `format.rs:124`). |
| `signature` field | MISMATCH | Not a frame field. `SegmentTrailer.signature` (`format.rs:46`) signs `segment_id \|\| prev_segment_hash \|\| merkle_root`. `SignedHead.signature` (`crates/strata/src/mesh_head.rs:18`) signs `MESH_HEAD_DOMAIN \|\| seq \|\| frame_hash \|\| merkle_root`. |
| `payload` field | MATCH | `Frame.payload: Vec<u8>`. |
| `compute_payload_digest(&[u8])` | MISMATCH | `strata::payload_blake3(kind, payload)`. Different preimage. Not a method. |
| `compute_frame_digest(&writer_id, seq, lamport, ts, &prev, &payload_digest)` | MISMATCH | `strata::frame_hash(&Frame)`. It does not take writer, lamport, or timestamp. |
| `verify_cryptography()` | MISSING | `ReplicaLog` recomputes `payload_blake3`, checks `prev_frame_hash`, and verifies the trailer with the pinned key. `verify_head` checks `SignedHead`. No per-frame method. |
| `WriterId::from_bytes([u8; 32])` | MISSING | No constructor. `writer_did` takes `&[u8; 32]`. |
| `Hash32` | MISMATCH | Every mesh hash is a bare `[u8; 32]` (`MeshProjection.digest`, `FrameRecord.frame_hash`, `SignedHead.frame_hash`). No alias. |
| `GENESIS_PREV_HASH` | MISMATCH | `strata::GENESIS_PREV_SEGMENT_HASH: [u8; 32] = [0u8; 32]` (`format.rs:27`). Same bytes. Different role: prev hash of segment 0, not the prev of the first frame. |
| `fold_writer_chains(&[SignedWriterFrame]) -> { admitted_frames }` | MISMATCH | `fn fold_chains(chains: &[ChainIn]) -> MeshProjection` at `crates/strata-store/src/mesh.rs:1480`. Private. `ChainIn` is `{writer: String, frames: Vec<FrameRecord>, admitted_to: u64}`. Return has no `admitted_frames`. Public read is `StrataStore::mesh_projection`. |
| `.admitted_frames` | MISSING | `MeshProjection` (`mesh.rs:282`) exposes `digest`, `frontier: BTreeMap<String, u64>`, `nodes`, `reinforcements`, `conflicts`, `review_counts`. |
| `EquivocationReceipt` | MISSING | `ImportReport.equivocation: bool` (`mesh.rs:312`). `record_equivocation` is private. `OpenConflict` is a two-claim slot, not a frame-signature receipt. |
| `WriterChainTip` | MISSING | Nearest are `MeshProjection.frontier` (did to last included seq) and `SignedHead {seq, frame_hash, merkle_root, signature}`. |

Adapter, and why it is not a second fold: `crates/vestige-core/src/chain_fold.rs` names the missing receipt types, aliases `Hash32` to `[u8; 32]`, sets `GENESIS_PREV_HASH` to the same 32 zero bytes as `GENESIS_PREV_SEGMENT_HASH`, and builds `WriterId` so `WriterId::did()` is `did_key_from_ed25519_public_key` (the function `writer_did` documents as the same encoding). `fold_writer_chains` only checks the ed25519 signature Gemini's tests flip and the `prev_hash` link those tests quarantine on. It does not linearize store effects, does not compute a mesh digest, and does not call into `mesh.rs`. Calling `fold_chains` would require making a private function public and encoding provenance JSON as `StoreOp`s, which is a different machine than the one `bf561a07` shipped.

## Per file

### `hooks/settings.stop-memory.fragment.json`

| Reference | Verdict | Real |
| --- | --- | --- |
| path `hooks/settings.stop-memory.fragment.json` | MISMATCH | File exists. Events are `PostToolUse` matcher `*`, `UserPromptSubmit`, `Stop`, `SessionStart` matcher `startup\|resume\|clear\|compact`. Command is `python3 ~/.vestige/hooks/operator-stop-memory.py`, timeout 30, no `--event`. |
| hook command events `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` | MATCH | Those event names are the Claude Code hook events the existing fragment already uses (plus `SessionStart`, which Gemini omits). |

NEW: the save-gate fragment is `hooks/settings.save-gate.fragment.json`. Matchers and timeout 5 match the drop. The command path is `hooks/save-gate.py`.

### `hooks/operator-stop-memory.py`

The drop's module is a different program from the file on disk. The on-disk hook reads `~/.operator/receipts`, writes memories through MCP, never blocks a tool, and always exits 0. Gemini's script denies `git commit` and blocks Stop.

| Reference | Verdict | Real |
| --- | --- | --- |
| path `hooks/operator-stop-memory.py` | MISMATCH | 460-line Operator-stop relay. Byte-identical copy at `packages/vestige-init/hooks/operator-stop-memory.py`. |
| `VESTIGE_REFLEX_DIR` | MISSING | Not read by the existing hook. It uses `OPERATOR_HOME`, `VESTIGE_MCP_BIN`, `VESTIGE_DATA_DIR`. |
| `~/.local/state/vestige/reflex` | MISSING | Not a path in the existing hook. |
| `hashlib.blake2b` | MISSING | The existing hook does not hash prompts. Strata frame hashes are blake3. The drop's hook receipts are blake2b by the spec. They are not the Rust receipt hashes. |
| `GIT_COMMIT_RE`, `MAX_STOP_BLOCKS_PER_TURN` | NEW | Not in the existing hook. |

Functions the drop defines (`state_dir`, `session_state_path`, `receipt_log_path`, `digest_hex`, `current_git_head_sha`, `load_state`, `save_state`, `append_event_receipt`, `extract_edited_hunks`, `normalize_error_sig`, `extract_exit_code`, `unsaved_files`, `unanchored_resolved_errors`, `format_save_gate_reason`, `handle_user_prompt_submit`, `handle_post_tool_use`, `handle_pre_tool_use`, `handle_stop`, `main`) are NEW. No `unsafe`. No bare `unwrap`. Failures exit 0. Logic notes, not compile failures:

- A prompt updates session state and does not append a frame. The signed chain's first parent is 32 zero bytes, not a prompt frame.
- Any `Bash` exit 0 while `open_failures` is non-empty closes every open failure, including a successful `git commit`.
- `remember` with an empty `files` list marks every edited path saved and anchors every resolved pair.

The script is stored verbatim at `hooks/save-gate.py`. `hooks/test_save_gate.py` passed: 3 tests, including bad JSON exiting 0.

### `hooks/test_save_gate.py`

NEW file. One reference change: `Path(__file__).with_name("operator-stop-memory.py")` would load the Operator relay and fail the deny assertion. It now loads `save-gate.py`. Logged in `docs/TWEAKS_FULL_LOOP.md`.

### `crates/vestige-core/src/provenance.rs`

Imports:

| Reference | Verdict | Real |
| --- | --- | --- |
| `crate::chain_fold::{Hash32, SignedWriterFrame, WriterId}` | MISSING | No `chain_fold` module on `main` or `bf561a07`. |
| `ed25519_dalek::{Signer, SigningKey}` | MATCH | `ed25519-dalek = "3"` in `crates/vestige-core/Cargo.toml`. `SigningKey::from_bytes(&[u8; 32])`, `Signer::sign`, `VerifyingKey::to_bytes` are the calls in `crates/vestige-core/src/actor.rs`. |
| `serde::{Deserialize, Serialize}` | MATCH | Already a dependency. |
| `blake3::Hasher` / `blake3::hash` | MATCH | `blake3 = "1"` already a dependency. |

NEW: `DOMAIN_STRATA_PROVENANCE_V1`, `DOMAIN_MISSING_PROVENANCE_V1`, `CausalEdge`, `TouchedFileHunk`, `PolicyVerdict`, `ProvenanceEventKind`, `MissingProvenanceReceipt`, `StrataProvenanceReceipt`, `compute_tool_call_digest`, `compute_prompt_hash`, `compute_passport_digest`, `require_anchor`, `into_signed_frame`, `from_signed_frame`, `prove_causal_chain`.

Nearest names that are not these types:

| Drop name | Verdict | Real |
| --- | --- | --- |
| `CausalEdge::{DerivedFrom, Closes}` | MISSING | String constants `W_DERIVED_FROM` and `W_CLOSED_BY` in `strata_store::ghostlink`. No enum. |
| `PolicyVerdict` | MISSING | `strata_gate::record::Verdict` is `Allow`, `Deny`, `Hold` with no `rule_id` or `reason`. |

`require_anchor` / `MissingProvenanceReceipt` are already in the drop. They are the #774 reject path. No second edit is required for that behavior. `prove_causal_chain` nests `if verify { if let Ok(receipt) }`, which `clippy -D warnings` warns as `collapsible_if`. That is fixed only if the lint fires, and the lint text is logged.

No `unsafe`. No `unwrap` outside the mental model of `Result`. `into_signed_frame` returns `serde_json::Error`.

### `PATCH crates/vestige-core/Cargo.toml`

| Dependency the drop adds | Verdict | Workspace |
| --- | --- | --- |
| `biscuit-auth = "6.0"` | MISSING | Not in root `Cargo.toml` or any member. `docs.rs` current 6.x exports `BiscuitBuilder`, `AuthorizerBuilder`, `BlockBuilder`, `builder_ext::{BuilderExt, AuthorizerExt}`. Added on `feat/subagent-passport` only. |
| `blake3 = "1"` | MATCH | `crates/vestige-core/Cargo.toml` line 52. Not added again. |
| `ed25519-dalek = { version = "2", features = ["std"] }` | MISMATCH | Existing line is `ed25519-dalek = "3"`. Version 2 is not added. Dalek 3 has no `std` feature to turn on. |
| `serde` with `derive` | MATCH | Already present. |
| `serde_json = "1"` | MATCH | Already present. |

Integration tests under `crates/vestige-core/tests/` cannot see a normal dependency. `ed25519-dalek = "3"` is added to `[dev-dependencies]` so `save_gate_and_provenance.rs` and `passport_tests.rs` can name `SigningKey`. That is a Cargo visibility rule, not a second crypto stack.

### `PATCH crates/vestige-core/src/lib.rs`

| Reference | Verdict | Real |
| --- | --- | --- |
| `pub mod chain_fold` "alongside" | MISSING | `lib.rs` has no `chain_fold` module. |
| `pub mod passport` | NEW | Added on the passport branch. |
| `pub mod provenance` | NEW | Added on the provenance branch. |

### `PATCH crates/vestige-mcp/src/tools/reflex/pairing.rs`

Would the existing file break if the patch were pasted at the bottom? Yes. `pub struct OpenFailure` collides with `pub(super) struct OpenFailure` at `pairing.rs:31`. The rest of the file (`classify`, `segment`, fix list, faded lessons) is not edited.

| Reference | Verdict | Real |
| --- | --- | --- |
| `OpenFailure` | MISMATCH | Existing `pub(super) struct OpenFailure { sig, node, tool, words, agent, edited }`. Gemini's struct is `{failure_frame_digest, command, error_sig, exit_code, tool_call_digest, git_head_sha, files, turn_seq, ts_ms}`. |
| `vestige_core::chain_fold::Hash32` | MISSING | Module absent until the adapter lands. |
| `vestige_core::provenance::{CausalEdge, TouchedFileHunk}` | MISSING | Module absent until provenance lands. |
| `serde` | MATCH | `pairing.rs` already uses `Serialize` / `Deserialize` via `use super::*` from `reflex.rs`. |
| `BTreeMap` / `BTreeSet` | MATCH | `std`. Not imported in `pairing.rs` today; the snippet imports them. |

NEW, nested in `pairing::save_gate` and re-exported except `OpenFailure`: `ResolvedFailurePair`, `FileAnchorEvent`, `SaveGateVerdict`, `RememberAutoBindContext`, `ReflexSessionState`, and the methods `new`, `on_user_prompt`, `record_file_anchor`, `record_command_outcome`, `unsaved_edited_files`, `unanchored_resolved_errors`, `check_save_gate`, `build_auto_bind_context`, `mark_remember_saved`.

`record_command_outcome` already carries `#[allow(clippy::too_many_arguments)]`. `build_auto_bind_context` still returns `CausalEdge::Root` when the session has no anchor. `execute_remember_save_with_autobind` rejects that case before it signs. The Root arm is not the save path.

Clippy risk: `error_sig.unwrap_or_else` is a library `Option` unwrap, not `Result::unwrap`. Allowed.

### `crates/vestige-mcp/src/tools/reflex/auto_bind.rs`

NEW module. External references:

| Reference | Verdict | Real |
| --- | --- | --- |
| `super::pairing::ReflexSessionState` | MISSING | Created by the pairing patch. |
| `vestige_core::chain_fold::{Hash32, SignedWriterFrame, WriterId, GENESIS_PREV_HASH}` | MISSING | Adapter. |
| `ed25519_dalek::SigningKey` | MATCH | `vestige-mcp` already depends on `ed25519-dalek = "3"`. |
| `crate::tools::reflex::pairing::SaveGateVerdict` in the unit test | MISMATCH | `mod pairing` was private (`reflex.rs`). `pub mod pairing` exposes `save_gate`. The test imports `crate::tools::reflex::pairing::save_gate::SaveGateVerdict` because `super` inside `mod tests` is `auto_bind`. `remember_unified::execute` does not call `execute_remember_save_with_autobind`. The proving tests are the unit test and `save_gate_and_provenance.rs`, not a live `remember` tool call. |

NEW: `RememberSaveInput`, `AutoBoundRememberResult`, `RememberAutoBindError`, `execute_remember_save_with_autobind`. Rule id `remember.autobind.v1` is a string in the drop. No `unsafe`. The only `expect` / `expect_err` calls are inside `#[cfg(test)]`.

### `PATCH crates/vestige-mcp/src/tools/reflex/mod.rs`

| Reference | Verdict | Real |
| --- | --- | --- |
| `crates/vestige-mcp/src/tools/reflex/mod.rs` | MISSING | Parent module is `crates/vestige-mcp/src/tools/reflex.rs`. Submodules `compaction`, `pairing`, `stale`, `witness` already live in `crates/vestige-mcp/src/tools/reflex/`. |
| `pub mod auto_bind` | NEW | Declared from `reflex.rs`. `tools/mod.rs` already has `pub mod reflex` (line 130). |

### `crates/vestige-core/tests/save_gate_and_provenance.rs`

NEW integration test. References `fold_writer_chains`, `WriterId`, `GENESIS_PREV_HASH`, `prove_causal_chain`, and the receipt enums. All of those are the adapter plus provenance.

The f2 → f1 → f0 test uses `git_head_sha` `"abc123def"`, signs with `SigningKey::from_bytes(&[42u8; 32])`, and expects `folded.admitted_frames.len() == 3`. The tamper test flips `signature[0]` and expects `fold_writer_chains` to drop f2 from the admitted set so `prove_causal_chain` from f2 is empty. That is prev-link quarantine, which `fold_chains` does not do for these bytes. The orphan test calls `require_anchor`.

`#![forbid(unsafe_code)]`. `expect` is test-only.

What the test does not prove: six distinct signed frames (commit, remember, fix, failure, edit, prompt). The prompt is a `prompt_hash` on the edit receipt. The hook JSONL holds `commit_anchor` and `fix_resolved` as blake2b lines, not as `SignedWriterFrame`s.

### `crates/vestige-core/src/passport.rs`

NEW. External references:

| Reference | Verdict | Real |
| --- | --- | --- |
| `biscuit_auth::builder_ext::{AuthorizerExt, BuilderExt}` | MISSING | Crate not in the workspace. Present on biscuit-auth 6 docs. `BuilderExt` is implemented for `BiscuitBuilder` and `AuthorizerBuilder`. The drop imports the traits and does not call `resource` / `check_operation`. Unused imports fail `dead_code` only as warnings if the traits are unused. `unused_imports` is a rustc warn, so `-D warnings` fails. Dropped if the compiler says they are unused. |
| `AuthorizerBuilder`, `Biscuit`, `BiscuitBuilder`, `BlockBuilder`, `KeyPair`, `PublicKey` | MISSING | Same crate. `AuthorizerBuilder::fact/check/code` take `self` and return `Result<Self, _>` on current docs, which matches the drop's chain. |
| `crate::chain_fold::Hash32` | MISSING | Adapter alias. |
| `iroh-willow` / Meadowcap | MISSING | Not in the drop and not in `bf561a07`. #405 shipped `crates/strata-mesh` (iroh QUIC + SPAKE2 tickets), not Willow. Not added. |

`#![forbid(unsafe_code)]`. No `unwrap`. Datalog is built with `format!` after `validate_datalog_atom` rejects quotes, backslashes, and control characters. `unix_now` is not a biscuit builtin. The authorizer injects `unix_now(<seconds>)` as a fact and the token checks that fact. That is ordinary datalog, not a clock builtin. If biscuit rejects `.starts_with` or the `or` shape, the passport tests fail and the Datalog string is the only edit.

#762 lines this file can prove: a child cannot outgrow a block it already carries, and a denied check returns a deterministic `PassportDenialReceipt`. It does not sync a Willow subspace.

### `crates/vestige-core/tests/passport_tests.rs`

NEW. `test_readonly_docs_allows_docs_read_and_denies_src_write_with_deterministic_receipt` and `test_child_cannot_widen_scope_or_ttl_and_expired_ttl_is_denied`. The second test also builds a `StrataProvenanceReceipt` whose `passport_digest` is `child.passport_digest()` and checks `verify_cryptography`.

#523 is not this test. #523 wants `SubagentStart` to inject exact memory ids and `SubagentStop` to write `derived_from` / `part_of` edges. No such hook or edge is in the drop. Not claimed.

#559 "a narrowed subagent's write outside its scope is refused" is the `src/app.rs` deny. "the receipt shows the token chain" is not met: the denial receipt stores `passport_digest` and the failing check's coordinates, not the biscuit block list.

### `demo/full_loop.sh`

NEW. It drives `hooks/operator-stop-memory.py`. That path is retargeted to `hooks/save-gate.py` for the same reason as the unit test. Keys are written under a temp dir with mode `0600` and never printed. Steps 10 and 11 shell out to `cargo test -p vestige-core --test passport_tests` and `--test save_gate_and_provenance`. Step 11 does not walk commit → remember → fix → failure → edit → prompt as six frames. It runs the three-frame test.

### `docs/FULL_LOOP_DEMO.md`

The paste lost the closing fence after `bash demo/full_loop.sh` and the heading markers on the CISO lines. The file in the tree is those same sentences with the fence closed and the headings restored. Wording check: the required line is present ("Vestige stops the destructive calls it recognizes before they run and keeps a receipt for every decision."). The drop does not say "every command", "whole chain", "proven", or "runs safely in production".

### `docs/TWEAKS_FULL_LOOP.md` in the drop

The three entries describe edits Gemini already baked into the pasted sources (fail-open `main`, `MissingProvenanceReceipt`, `SubagentPassport`). They are not diffs against this repo. The file checked in here records only edits this wiring made, with the compiler or test text that forced each one.

## Clippy, unwrap, unsafe

| Risk | Where | Treatment |
| --- | --- | --- |
| `clippy::collapsible_if` | `prove_causal_chain` | Fix only after `cargo clippy --workspace -- -D warnings` prints it. |
| `clippy::too_many_arguments` | `record_command_outcome` | Already allowed in the drop. `compute_frame_digest` has 6 parameters; the default lint fires at 7. |
| `unused_imports` | `passport.rs` `BuilderExt` / `AuthorizerExt` | Remove if rustc reports them. |
| `dead_code` on adapter tips / equivocation receipts | `chain_fold.rs` | Fields are public and written by `fold_writer_chains`, which the integration test reads via `admitted_frames`. |
| `unwrap` outside tests | none in the drop's library paths | Adapter uses `Result`. |
| `unsafe` | forbidden in `provenance.rs`, `auto_bind.rs`, `passport.rs`, and the two integration tests | Adapter is `forbid(unsafe_code)` as well. |
| Logic | hook blake2b vs receipt blake3 | Left as specified. The demo does not compare them. |
| Logic | `remember` MCP handler never calls `execute_remember_save_with_autobind` | Not in the drop. #774's function is tested. The issue stays open until `remember_unified` uses it. |
| Logic | frame `writer_id` is taken from the signing key, not cross-checked against the JSON body's `writer_id` | `verify_cryptography` checks the signature under `frame.writer_id`. A body that names a different writer still verifies. Not exercised by the tests. Not changed. |

## Branches

- `feat/save-gate-hooks` from `main`: hook, fragment, unit test, this analysis.
- `feat/provenance-autobind` from `bf561a07`: adapter, provenance, auto-bind, the same hook files (the python gate runs on every branch).
- `feat/subagent-passport` from the provenance tip: passport, biscuit-auth 6, demo.

No PR. No merge. No edits under `benchmarks/real_engine/`, `rsb_seat_behind`, `HYPERNYM_LATTICE`, or `rungs.rs`.
