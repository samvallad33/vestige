# Wave 14F/14H tweaks

Gemini's `### FILE:` blocks for `crates/vestige-fmindex` (#766) and
`crates/vestige-codegraph` (#768) were extracted verbatim, then both crates
were added to the workspace. Versions were not rewritten. `cargo` resolved:

| Requirement | Resolved | Registry source checked |
|---|---|---|
| `gix` `0.87` | `0.87.1` | `gix-0.87.1/src/repository/reference.rs` `head_id`, `revision.rs` `rev_walk`, `repository/object.rs` `empty_tree`, `object/tree/diff/mod.rs` `Change::{Addition,Deletion,Modification,Rewrite}`, `object/tree/diff/for_each.rs` `for_each_to_obtain_tree` |
| `tree-sitter` `0.25` | `0.25.10` | `tree-sitter-0.25.10/binding_rust/lib.rs` `Parser::set_language`, `InputEdit`, `Node::named_children` |
| `tree-sitter-rust` `0.24` | `0.24.2` | `tree-sitter-rust-0.24.2/bindings/rust/lib.rs` `LANGUAGE: LanguageFn` (`.into()` → `tree_sitter::Language`) |
| `tree-sitter-python` `0.25` | `0.25.0` | optional, not in the default feature set |
| `tree-sitter-typescript` `0.23` | `0.23.2` | optional, not in the default feature set |
| `blake3` `1` | `1.8.7` | `Hasher::{new,update,finalize}` |
| `serde` `1` | `1.0.229` | derive |
| `tempfile` `3` | `3.27.0` | dev-dependency |

Those signatures match the calls Gemini wrote. No gix or tree-sitter edit was required.

The FM-index is not referenced from ranking, retrieval, or MCP.

## Changes

### `Cargo.toml:5-10`

Gemini wrote a members patch listing the two new crates. Inserted
`"crates/vestige-fmindex"` and `"crates/vestige-codegraph"` into the existing
`[workspace.members]` array.

### `crates/vestige-fmindex/src/lib.rs:1`

Gemini wrote `#![forbid(unsafe_code)]`. Changed to `#![deny(unsafe_code)]`.

`crates/vestige-fmindex/src/wavelet.rs:1` is `#![allow(unsafe_code)]` around the
`core::arch::x86_64::_popcnt64` fast path (`#[cfg(all(target_arch = "x86_64", target_feature = "popcnt"))]`).
A crate-level `forbid` cannot be overridden (E0453). `deny` still rejects
`unsafe` in every other module. The scalar SWAR `popcount_scalar` path is unchanged.

### `crates/vestige-fmindex/src/lib.rs:24`

Gemini wrote `use crate::error::FmError;`. Changed to `pub use crate::error::FmError;`.

`crates/vestige-fmindex/tests/fm_exact.rs:2` imports `vestige_fmindex::FmError`.
The private import made the enum unreachable from the integration test (E0603).
`pub mod error` was already public; the re-export is what the test calls.

### `crates/vestige-fmindex/src/suffix.rs:119-125`

Gemini wrote:

```rust
for i in 1..n {
    if is_lms(&is_s, i) {
        let c = s[i] as usize;
```

Changed to `for (i, &sym) in s.iter().enumerate().skip(1)` and `let c = sym as usize`.

`cargo clippy -- -D warnings` rejects `clippy::needless_range_loop` on Rust 1.98.1.
Same indices, same LMS placement.

### `crates/vestige-fmindex/src/wavelet.rs:90` and `:95`

Gemini wrote `ones_seen % SELECT_SAMPLE_STEP == 0` and
`zeros_seen % SELECT_SAMPLE_STEP == 0`. Changed both to
`is_multiple_of(SELECT_SAMPLE_STEP)`.

`clippy::manual_is_multiple_of` on Rust 1.98.1. `SELECT_SAMPLE_STEP` is 256, so
this matches the `%` test, including the `0` sample.

### `crates/vestige-fmindex/tests/fm_exact.rs:194-235`

Gemini printed raw bytes, index bytes, ratio, and run count, then asserted
`count` only. The ignored 10MB test now also times `count` and full `locate`
of `shard=17 op=chain_fold`, times `count` of `seq=1234 status=ADMITTED`, and
prints `VmRSS` / `VmHWM` from `/proc/self/status`.

Added `assert_eq!(located.len(), 2_500)` next to the existing count asserts.
`f64` is used only in the `eprintln` ratio. The pass/fail comparison stays integer.

### `crates/vestige-codegraph/src/congruence.rs:82-104`

Gemini held `self.by_kind_shape.entry(shape_key)` and then called `self.union`
while that mutable borrow was live (E0499). The peer id is copied out with
`get` + `first().copied()` before `union`. The `entry` borrow is taken only
afterwards, to push the id. Same union rule, same witness, same insertion.

### `crates/vestige-codegraph/src/parse.rs:171-188`

Gemini reused the cached `Tree` whenever the path had been parsed, and applied
`Tree::edit` only if an `InputEdit` was present. `CodeGraph::index_repo` always
passes `edit: None` with a full new buffer. tree-sitter then kept the previous
span, so the commit that appends `fn altered` still exposed only `fn sum`.

The cached tree is now reused only when `edit` is `Some`. A full-buffer reparse
passes `None` as the old tree. `Tree::edit` is unchanged on the incremental path.

### `crates/vestige-codegraph/src/parse.rs:298-304`

Gemini's return type was `Option<(SymbolKind, String, Option<(usize, usize)>)>`.
`clippy::type_complexity` rejects that signature. Aliased as `SymbolHeader` and
used `Option<SymbolHeader>`. The tuple layout is unchanged.

### `crates/vestige-codegraph/src/why.rs:58-68`

Gemini's hunk match had an `or_else` that accepted any hunk on the same path
when none overlapped the symbol's line range. The commit that adds `fn altered`
(hunk lines 4–7) was then attached to unchanged `fn sum` (lines 1–3), so
`why("src/math.rs", 2)` returned 3 commits. The test requires 2: the rename and
the original add.

The `or_else` is removed. A hop is recorded only when `new_range` overlaps the
symbol's line range. A pure git rename still synthesizes a full-file range in
`git.rs`, so a moved file still overlaps.
