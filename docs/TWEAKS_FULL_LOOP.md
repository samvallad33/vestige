# Tweaks log

Gemini's pasted `docs/TWEAKS_FULL_LOOP.md` describes edits already present in the drop (fail-open `main`, `MissingProvenanceReceipt`, `SubagentPassport`). Those are not diffs against this repo. This file records only changes made while wiring the drop in.

## 1. `hooks/save-gate.py` instead of `hooks/operator-stop-memory.py`

- **File**: `hooks/operator-stop-memory.py` (drop path) → `hooks/save-gate.py`
- **Original text**: the drop is a new program saved at `hooks/operator-stop-memory.py`
- **New text**: the same bytes, at `hooks/save-gate.py`. `hooks/operator-stop-memory.py` is untouched.
- **Error that forced it**: the path is the Operator-stop relay. `packages/vestige-init/test/kernel.test.js` and `memory-setup.test.js` require it to be byte-identical to `packages/vestige-init/hooks/operator-stop-memory.py`, and `six-tool-memory-hooks.test.js` calls `ingest()` on it. Pointing `hooks/test_save_gate.py` at the existing file fails:

```
FAIL: test (__main__.T.test)
AssertionError: unexpectedly None
```

`PreToolUse` on `git commit -m 'fix'` after an edit produced no JSON (`rc 0`, empty stdout). The save-gate assertion `permissionDecision == deny` never ran.

## 2. `hooks/test_save_gate.py` hook path

- **File**: `hooks/test_save_gate.py`
- **Original text**: `HOOK = Path(__file__).with_name("operator-stop-memory.py")`
- **New text**: `HOOK = Path(__file__).with_name("save-gate.py")`
- **Error that forced it**: the assertion in tweak 1. `Path(__file__).with_name` loads the sibling of the test, which is the Operator relay.

## 3. `hooks/settings.save-gate.fragment.json` instead of `hooks/settings.stop-memory.fragment.json`

- **File**: `hooks/settings.stop-memory.fragment.json` (drop path) → `hooks/settings.save-gate.fragment.json`
- **Original text**: `"command": "python3 hooks/operator-stop-memory.py --event UserPromptSubmit"` (and the same command for `PreToolUse`, `PostToolUse`, `Stop`)
- **New text**: `"command": "python3 hooks/save-gate.py --event UserPromptSubmit"` (and the same command for the other three events). Matchers and `"timeout": 5` are unchanged.
- **Error that forced it**: `benchmarks/real_engine/integrity.py` `_registered_hook_catalog` reads `hooks/settings.stop-memory.fragment.json` by name. The file on `main` registers `python3 ~/.vestige/hooks/operator-stop-memory.py` at timeout 30, including `SessionStart`. Replacing it changes the catalog the benchmark checks and the fragment `npx @vestige/init` ships. Not edited.

## 4. Docstring path left as the drop wrote it

- **File**: `hooks/save-gate.py`
- **Original text**: `Vestige Save-Gate & 7-Field Strata Provenance Hook (hooks/operator-stop-memory.py)`
- **New text**: unchanged
- **Error that forced it**: none. The docstring is not a path the test or the fragment executes.

## 5. `prove_causal_chain` nested if

- **File**: `crates/vestige-core/src/provenance.rs`
- **Original text**:

```rust
        if frame.verify_cryptography().is_ok() {
            if let Ok(receipt) = StrataProvenanceReceipt::from_signed_frame(frame) {
                by_digest.insert(frame.frame_digest, receipt);
            }
        }
```

- **New text**:

```rust
        if frame.verify_cryptography().is_ok()
            && let Ok(receipt) = StrataProvenanceReceipt::from_signed_frame(frame)
        {
            by_digest.insert(frame.frame_digest, receipt);
        }
```

- **Error that forced it**: `cargo clippy -p vestige-core -- -D warnings`

```
error: this `if` statement can be collapsed
   --> crates/vestige-core/src/provenance.rs:232:9
    |
232 | / ...   if frame.verify_cryptography().is_ok() {
233 | | ...       if let Ok(receipt) = StrataProvenanceReceipt::from_signed_f...
234 | | ...           by_digest.insert(frame.frame_digest, receipt);
235 | | ...       }
236 | | ...   }
    |
    = note: `-D clippy::collapsible-if` implied by `-D warnings`
help: collapse nested if block
```

## 6. Close `impl ReflexSessionState` inside `pairing::save_gate`

- **File**: `crates/vestige-mcp/src/tools/reflex/pairing.rs`
- **Original text**: the drop's `impl` closing `}` is the line immediately before `FILE: crates/vestige-mcp/src/tools/reflex/auto_bind.rs` (upload line 1341). The first paste into the nested module stopped on the `}` of `mark_remember_saved`.
- **New text**: that impl-closing `}` is present, and one more `}` closes `pub mod save_gate`.
- **Error that forced it**:

```
error: this file contains an unclosed delimiter
   --> crates/vestige-mcp/src/tools/reflex/pairing.rs:1038:4
    |
725 | pub mod save_gate {
    |                   - unclosed delimiter
```

The nesting itself is tweak 7.

## 8. Edition 2024 reference depth in `unsaved_edited_files`

- **File**: `crates/vestige-mcp/src/tools/reflex/pairing.rs`
- **Original text**: `*edited_turn > saved_turn` and `.map(|(path, _)| path.clone())`
- **New text**: `**edited_turn > saved_turn` and `.map(|(path, _)| (*path).clone())`
- **Error that forced it**: `cargo check -p vestige-mcp --tests`

```
error[E0308]: mismatched types
   --> crates/vestige-mcp/src/tools/reflex/pairing.rs:916:32
    |
916 |                 *edited_turn > saved_turn
    |                                ^^^^^^^^^^ expected `&u64`, found `u64`
    |
help: consider dereferencing the borrow
    |
916 |                 **edited_turn > saved_turn
```

`BTreeMap::iter` yields `(&String, &u64)`, and `Iterator::filter` passes `&Item`. On edition 2024 that pattern binds `&&u64`. `get(*path)` already type-checked, so `*path` is `&String`.

## 9. Re-export only `ReflexSessionState`

- **File**: `crates/vestige-mcp/src/tools/reflex/pairing.rs` and the unit test in `auto_bind.rs`
- **Original text**: `pub use` of `FileAnchorEvent`, `ReflexSessionState`, `RememberAutoBindContext`, `ResolvedFailurePair`, `SaveGateVerdict`. Test: `use crate::tools::reflex::pairing::SaveGateVerdict;`
- **New text**: `pub use save_gate::ReflexSessionState;` and `use super::pairing::save_gate::SaveGateVerdict;`
- **Error that forced it**: `cargo check -p vestige-mcp`

```
warning: unused imports: `FileAnchorEvent`, `RememberAutoBindContext`, `ResolvedFailurePair`, and `SaveGateVerdict`
    --> crates/vestige-mcp/src/tools/reflex/pairing.rs:1037:5
```

`unused_imports` is a rustc warning. `cargo clippy -- -D warnings` denies it. `mod pairing` is private, so a `pub use` that nothing in the non-test crate names is unused. `auto_bind` names `ReflexSessionState`. The test names `SaveGateVerdict` through `save_gate` directly.

## 10. `pub mod auto_bind` and `pub mod pairing`

- **File**: `crates/vestige-mcp/src/tools/reflex.rs`
- **Original text**: `mod auto_bind;` (first wiring, matching `mod pairing`) and the test import `use super::pairing::save_gate::SaveGateVerdict;`
- **New text**: `pub mod auto_bind;` (the drop's words), `pub mod pairing;`, test import `use crate::tools::reflex::pairing::save_gate::SaveGateVerdict;`
- **Error that forced it**: `cargo check -p vestige-mcp --tests` on the private module:

```
error[E0433]: cannot find `pairing` in `super`
   --> crates/vestige-mcp/src/tools/reflex/auto_bind.rs:165:16
    |
165 |     use super::pairing::save_gate::SaveGateVerdict;
    |                ^^^^^^^ could not find `pairing` in `super`
```

`super` inside `mod tests` is `auto_bind`, not `reflex`. The same check warned `dead_code` on `execute_remember_save_with_autobind`, `RememberSaveInput`, and `OpenFailure` because `mod auto_bind` is private and `reflex.rs` never names those items. `pub mod auto_bind` is what the drop asked to add to the parent module. `pub mod pairing` is required so that public function can name `ReflexSessionState` and its public field types (`OpenFailure`, `ResolvedFailurePair`, `FileAnchorEvent`). Existing pairing items stay `pub(super)`.

## 11. `drain(..).collect()` to `mem::take`

- **File**: `crates/vestige-mcp/src/tools/reflex/pairing.rs`
- **Original text**: `let drained: Vec<OpenFailure> = self.open_failures.drain(..).collect();`
- **New text**: `let drained = std::mem::take(&mut self.open_failures);`
- **Error that forced it**: `cargo clippy -p vestige-mcp --all-targets -- -D warnings`

```
error: draining all elements of a collection into a new collection of the same type
   --> crates/vestige-mcp/src/tools/reflex/pairing.rs:897:45
    |
897 | ...Failure> = self.open_failures.drain(..).collect();
    |
    = note: `-D clippy::drain-collect` implied by `-D warnings`
help: use `mem::take`
```

## 7. Nest the new pairing types in `save_gate`

- **File**: `crates/vestige-mcp/src/tools/reflex/pairing.rs`
- **Original text**: the drop appends `pub struct OpenFailure` at file scope.
- **New text**: the same items are inside `pub mod save_gate`, and `ReflexSessionState`, `SaveGateVerdict`, `RememberAutoBindContext`, `ResolvedFailurePair`, and `FileAnchorEvent` are re-exported. `OpenFailure` is not re-exported.
- **Error that forced it**: a file-scope paste duplicates `OpenFailure`. The existing type is `pub(super) struct OpenFailure` at `pairing.rs` with fields `sig`, `node`, `tool`, `words`, `agent`, `edited`. Rust rejects the second struct. The compile was not re-run in that shape; the unclosed-delimiter build above is the build that included the nest. The duplicate-name failure is the reason the nest exists.

## 12. Quarantine bodies ride the admitted store log

- **File**: `crates/vestige-core/src/provenance.rs`, `crates/vestige-mcp/src/provenance_log.rs`, `crates/vestige-mcp/src/bin/cli.rs`
- **Original text**: `execute_remember_save_with_autobind` returns `MissingProvenanceReceipt` and writes nothing. `fold_chains` has no memory-quarantine receipt.
- **New text**: `quarantine_body` / `release_body` / `project_provenance_bodies`. `vestige provenance quarantine|release|status` appends those bodies as `UpsertNode` frames (`created_at_ms = 0`) and leaves the quarantine node in place when the release node is appended.
- **Error that forced it**: #774's done-when is a quarantined orphan that syncs over the iroh mesh and is freed by an appended frame. `MeshProjection` has no such receipt, and a second fold would not be what `bf561a07` syncs. An admitted store frame is already in `fold_chains`, so both devices recompute the same receipt and the same `mesh_digest`.

## 13. `status` must take the mutable store guard

- **File**: `crates/vestige-mcp/src/provenance_log.rs`
- **Original text**: `with_store(storage, status_store)`
- **New text**: `with_store(storage, |store| status_store(store))`
- **Error that forced it**: `cargo test -p vestige-mcp --lib autobind`

```
error[E0631]: type mismatch in function arguments
   --> crates/vestige-mcp/src/provenance_log.rs:33:25
    |
 33 |     with_store(storage, status_store)
    |
    = note: expected function signature `for<'a> fn(&'a mut StrataStore) -> _`
               found function signature `fn(&StrataStore) -> _`
```

## 14. One-sided quarantine sync does not use `sync_both`

- **File**: `crates/vestige-mcp/tests/mesh_sync.rs`
- **Original text**: the new test called `sync_both`, which asserts `dialed["merges"]` is non-empty.
- **New text**: `sync_ok` checks that the listener and the dialer both exit 0.
- **Error that forced it**: `cargo test -p vestige-mcp --features mesh --test mesh_sync -- --test-threads=1`

```
test orphan_quarantine_syncs_over_loopback_iroh_and_releases_to_one_mesh_digest ... FAILED
assertion failed: dialed["merges"].as_array().is_some_and(|rows| !rows.is_empty())
```

The dialer is the device that ingested. `import_chain` writes no `MergeChain` when the other chain's fresh effects are all mesh receipts (`mesh.rs` receipt-only return). The listener is the side that admits the quarantine. The other three mesh tests still use `sync_both`.

## 15. Peer quarantine is read from the replica chain

- **File**: `crates/strata-store/src/mesh.rs`, `crates/vestige-mcp/src/provenance_log.rs`
- **Original text**: `status` listed `store.nodes()`, the local registry only.
- **New text**: `StrataStore::mesh_upsert_contents` walks the same chains as `project_with` (`chains`) and returns admitted live `UpsertNode` contents. `status` projects those.
- **Error that forced it**: `cargo test -p vestige-mcp --features mesh --test mesh_sync orphan_quarantine -- --test-threads=1 --nocapture`

```
assertion `left == right` failed
  left: Null
  right: "8a29795ba2c5c578584e1fcbd5559f1a99561aab01d47f683558bd42f1f3f6fd"
at crates/vestige-mcp/tests/mesh_sync.rs:507
assert_eq!(right["quarantined"][0]["receipt_digest"], receipt);
```

The mesh digests already matched. The receiver keeps the sender's frames in `mesh/chains/<fingerprint>`, not in its node registry. `fold_chains` was not given a second implementation.

`strata-store` is edition 2021. `if let` + `&&` is a let-chain and does not compile there (`let chains are only allowed in Rust 2024 or later` at `mesh.rs:513`). The live check stays a nested `if`.

## 16. Mesh-feature clippy rejected an identity `map_err`

- **File**: `crates/vestige-mcp/src/mesh.rs` (`sync`, dial path)
- **Original text**: `})?.map_err(|err| err)?;`
- **New text**: `})??;`
- **Error that forced it**: `cargo clippy --workspace --all-targets --features mesh -- -D warnings` and the same command with `-p vestige-mcp`. Both exited 101.

```
error: unnecessary map of the identity function
   --> crates/vestige-mcp/src/mesh.rs:221:20
    |
221 |                   })?
222 | |                 .map_err(|err| err)?;
    | |___________________________________^
    = note: `-D clippy::map-identity` implied by `-D warnings`
```

`with_store` already returns `Result<Result<_, String>, String>` here, so the second `?` is the same conversion. Default-feature clippy does not typecheck this `cfg(feature = "mesh")` block, which is why `cargo clippy --workspace --all-targets -- -D warnings` exited 0 before the change.

## 17. Mesh-feature clippy rejected a nested `if` in `wait_file`

- **File**: `crates/vestige-mcp/tests/mesh_sync.rs`
- **Original text**: `if let Ok(text) = fs::read_to_string(path) { if !text.trim().is_empty() { ... } }`
- **New text**: `if let Ok(text) = fs::read_to_string(path) && !text.trim().is_empty()`
- **Error that forced it**: the re-run of `cargo clippy --workspace --all-targets --features mesh -- -D warnings` after section 16, exit 101. `vestige-mcp` is edition 2024, so the let-chain compiles. The first mesh clippy run stopped on section 16 and did not reach this test target.

```
error: this `if` statement can be collapsed
   --> crates/vestige-mcp/tests/mesh_sync.rs:141:9
    = note: `-D clippy::collapsible-if` implied by `-D warnings`
```
