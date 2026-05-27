# Sub-Plan 0001c: Sunset the `async-trait` crate dependency

**Status**: Draft
**Branch**: `feat/storage-trait-phase1` (Phase 1 amendment, PR A)
**Depends on**:
- `0001a-trait-rewrite.md` (rewrites `MemoryStore` / `LocalMemoryStore` and
  the SQLite impl; lands first)
- `0001b-sqlite-split.md` (moves `sqlite.rs` into `sqlite/`; lands second)

**Related**: `docs/adr/0002-phase-2-execution.md` (decision D1 closing line:
"async-trait dependency stays in Cargo.toml only if other code uses it;
otherwise removed"). This sub-plan operationalises the removal.

---

## Context

This is the third and final Phase 1 amendment sub-plan. Sub-plan `0001a`
rewrote `MemoryStore` / `LocalMemoryStore` using
`#[trait_variant::make(MemoryStore: Send)]` and dropped the
`#[async_trait::async_trait]` attribute from the SQLite impl block.
Sub-plan `0001b` then split `sqlite.rs` into a `sqlite/` directory; the
trait impl now lives in `sqlite/trait_impl.rs`. After `0001a` lands, the
only remaining `async_trait` usage in the workspace is the embedder pair
(`embedder/mod.rs` declares the trait; `embedder/fastembed.rs` implements
it). This sub-plan rewrites those two files following the exact pattern
from `0001a`, then removes `async-trait = "0.1"` from
`crates/vestige-core/Cargo.toml`. End state: zero `async_trait` references
anywhere under `crates/`, zero direct dependency on the `async-trait`
crate, workspace tests and clippy green.

The rewrite is mechanically tiny -- one trait declaration, one impl block,
one Cargo.toml line -- but it is gated behind `0001a` and `0001b` so the
trait-rewrite pattern is already settled and so the SQLite trait impl
attribute has already been dropped. Doing the embedder rewrite without
that pre-work would leave the `async-trait` dep behind for the SQLite
side and force the Cargo.toml deletion into a separate commit later.

---

## Scope

### In scope

- Rewrite `LocalEmbedder` declaration in
  `crates/vestige-core/src/embedder/mod.rs` to use
  `#[trait_variant::make(Embedder: Send)] pub trait LocalEmbedder`.
- Delete the `pub use LocalEmbedder as Embedder;` alias from the same file.
  The `Embedder` symbol becomes the trait that `trait_variant::make` emits
  at the same module path, so the existing
  `pub use embedder::{Embedder, ..., LocalEmbedder};` line in
  `crates/vestige-core/src/lib.rs:167` keeps working untouched.
- Drop the `#[async_trait::async_trait]` attribute from the
  `FastembedEmbedder` impl block in
  `crates/vestige-core/src/embedder/fastembed.rs`.
- Update doc comments on the trait declaration to describe
  `trait_variant` rather than `async_trait`.
- Remove `async-trait = "0.1"` from
  `crates/vestige-core/Cargo.toml` (line 119 area). Use
  `cargo rm async-trait` from inside the crate directory.
- Verify with `grep -rn "async_trait" crates/` returning zero hits.

### Out of scope

- Any change to the `MemoryStore` trait or `SqliteMemoryStore` impl;
  those were handled by `0001a`.
- Any file moves in `embedder/` (no parallel of `0001b` is required;
  `embedder/` already has the `mod.rs` + `fastembed.rs` shape).
- Touching `crates/vestige-mcp/` or any cognitive module. None of them
  hold `Arc<dyn Embedder>` or `Box<dyn Embedder>` in production.
- Renaming the `Embedder` / `LocalEmbedder` symbols or changing the
  re-exports in `crates/vestige-core/src/lib.rs`.

---

## Prerequisites

### State assumed at start

- `0001a` is merged onto the branch. After `0001a`:
  - `crates/vestige-core/src/storage/memory_store.rs` declares
    `#[trait_variant::make(MemoryStore: Send)] pub trait LocalMemoryStore`.
  - The SQLite impl block has no `#[async_trait::async_trait]` attribute.
  - `grep -rn async_trait crates/` returns exactly three hits, all in
    `crates/vestige-core/src/embedder/` (two in `mod.rs`, one in
    `fastembed.rs`), and one Cargo.toml hit.
- `0001b` is merged onto the branch. After `0001b`:
  - `crates/vestige-core/src/storage/sqlite.rs` no longer exists as a
    single file; the impl lives in `crates/vestige-core/src/storage/sqlite/trait_impl.rs`.
  - The embedder files are untouched.

### Required crates

| Crate          | Version | Action                                                          |
|----------------|---------|-----------------------------------------------------------------|
| `trait-variant`| `0.1`   | Already declared (line 117 of Cargo.toml). Verify present.      |
| `async-trait`  | `0.1`   | Remove. Only the two embedder files still use it after `0001a`. |

### Workspace-wide audit before starting

Run from `/home/delandtj/prppl/vestige-phase2/` (or the equivalent
worktree where this sub-plan executes):

```bash
grep -rn "async_trait\|async-trait" crates/ tests/
```

Expected hits before this sub-plan starts (after `0001a` + `0001b`):

```
crates/vestige-core/Cargo.toml:119:async-trait = "0.1"
crates/vestige-core/src/embedder/mod.rs:24:/// `#[async_trait::async_trait]` makes every `async fn` return a
crates/vestige-core/src/embedder/mod.rs:27:#[async_trait::async_trait]
crates/vestige-core/src/embedder/mod.rs:56:/// Both names refer to the same `async_trait`-annotated trait.
crates/vestige-core/src/embedder/fastembed.rs:44:#[async_trait::async_trait]
```

Five hits across two source files and one Cargo.toml. After this sub-plan,
the same grep must return zero hits.

```bash
grep -rn "async-trait\|async_trait" --include="Cargo.toml" crates/
```

Expected: exactly one hit (`crates/vestige-core/Cargo.toml:119`). No other
workspace crate declares `async-trait` as a direct dependency. This is
the precondition that lets us delete the line cleanly.

---

## Files Touched

### Trait declaration (vestige-core)

| File                                            | Lines (approx) | Change                                                                                                                                                                       |
|-------------------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/vestige-core/src/embedder/mod.rs`       | 21-57          | Replace `#[async_trait::async_trait] pub trait LocalEmbedder: Send + Sync + 'static` with `#[trait_variant::make(Embedder: Send)] pub trait LocalEmbedder: Sync + 'static`. Delete the `pub use LocalEmbedder as Embedder;` alias on line 57. Update doc comments (lines 21-26, 55-56). |

### Impl block (vestige-core)

| File                                            | Lines (approx) | Change                                                                                                       |
|-------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------|
| `crates/vestige-core/src/embedder/fastembed.rs` | 44             | Delete the `#[async_trait::async_trait]` attribute. Keep the `impl LocalEmbedder for FastembedEmbedder { ... }` body verbatim. No `Box::pin`, no `'async_trait` lifetimes, no manual `Pin<Box<dyn Future>>`. |

### Other Embedder impls

None. `grep -rn "impl.*LocalEmbedder\|impl.*Embedder for" crates/ tests/`
returns exactly one production hit:
`crates/vestige-core/src/embedder/fastembed.rs:45: impl LocalEmbedder for FastembedEmbedder`.
There is no test mock implementing `Embedder` in the test tree (the only
test that touches the trait, `tests/phase_1/embedder_trait.rs`, uses the
concrete `FastembedEmbedder` boxed as `Box<dyn Embedder>`).

### Call sites (production)

Verified by:

```bash
grep -rn "dyn Embedder\|dyn LocalEmbedder" crates/ tests/ --include="*.rs"
grep -rn "Box<dyn Embedder>\|Arc<dyn Embedder>" crates/ tests/ --include="*.rs"
grep -rn "use.*Embedder" crates/ tests/ --include="*.rs"
```

Production call sites that may need verification (and the expected change
for each, even though we have already verified that none need an edit):

| File                                                       | Use                                                                                                            | Required change |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------|
| `crates/vestige-core/src/lib.rs:167`                       | `pub use embedder::{Embedder, EmbedderError, EmbedderResult, FastembedEmbedder, LocalEmbedder};`               | None. Both names still exist at `crate::embedder::*` after the rewrite; `Embedder` is now the `trait_variant`-generated trait, `LocalEmbedder` is the source-of-truth trait. The re-export keeps resolving. |
| `crates/vestige-core/src/embedder/fastembed.rs:7`          | `use super::{EmbedderError, EmbedderResult, LocalEmbedder};`                                                   | None. `LocalEmbedder` is still the source-of-truth trait name. |
| `crates/vestige-core/src/embedder/mod.rs:5`                | `pub use fastembed::FastembedEmbedder;`                                                                        | None. Concrete type, untouched. |
| `crates/vestige-mcp/src/**`                                | No file imports `Embedder` or `LocalEmbedder` by name; none hold `Arc<dyn Embedder>` or `Box<dyn Embedder>`.   | None. Verified by grep returning empty for `dyn Embedder` and `dyn LocalEmbedder` under `crates/vestige-mcp/`. |
| Cognitive modules under `crates/vestige-core/src/advanced/` and `crates/vestige-core/src/neuroscience/` | No file imports `Embedder` or `LocalEmbedder` by name. `advanced/adaptive_embedding.rs` defines its own unrelated `AdaptiveEmbedder` struct. | None. The name collision is purely surface-level; the two types live in different modules and never resolve to each other. |
| `crates/vestige-core/src/embeddings/**`                    | No file imports `Embedder` or `LocalEmbedder`. The `EmbeddingService` struct is what `FastembedEmbedder` wraps internally. | None. |

The production audit returns zero files that need editing.

### Call sites (tests)

| File                                                       | Lines | Use                                                                | Required change |
|------------------------------------------------------------|-------|--------------------------------------------------------------------|-----------------|
| `tests/phase_1/embedder_trait.rs`                          | 3, 19 | `use vestige_core::embedder::{Embedder, FastembedEmbedder};`<br>`let e: Box<dyn Embedder> = Box::new(FastembedEmbedder::new());` | None. `Embedder` is the `trait_variant`-generated Send variant; `Box<dyn Embedder>` keeps compiling. `FastembedEmbedder` implements `LocalEmbedder`; the blanket `impl<T: LocalEmbedder + Send> Embedder for T` that `trait_variant::make` emits gives the boxing for free. |

The test audit returns zero files that need editing.

### Cargo dependency cleanup

| File                                | Lines     | Change                                                                                              |
|-------------------------------------|-----------|-----------------------------------------------------------------------------------------------------|
| `crates/vestige-core/Cargo.toml`    | 119       | Remove `async-trait = "0.1"`. Run `cargo rm async-trait` from inside `crates/vestige-core/` so `Cargo.lock` updates atomically with the manifest. |

### Documentation

| File                                        | Change                                                                                                                                                   |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/vestige-core/src/embedder/mod.rs`   | Rewrite the trait-level doc comment (lines 21-26) and the `pub use` doc comment (lines 55-56) to describe `trait_variant`, not `async_trait`. See "Trait declaration rewrite" below for the exact new text. |
| `CLAUDE.md`                                 | No change. The repo-level architecture notes do not name the trait attribute.                                                                            |

---

## Trait Declaration Rewrite

### Before (state after `0001a` and `0001b` land)

`crates/vestige-core/src/embedder/mod.rs:1-58`:

```rust
//! Text-to-vector encoding trait. Pluggable per-install.

mod fastembed;

pub use fastembed::FastembedEmbedder;

/// Error returned by every `Embedder` method.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("embedder initialization failed: {0}")]
    Init(String),
    #[error("embedding generation failed: {0}")]
    EmbedFailed(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type EmbedderResult<T> = std::result::Result<T, EmbedderError>;

/// Pluggable embedder. The storage layer NEVER calls fastembed directly;
/// callers compute vectors via this trait and pass them into `MemoryStore`.
///
/// `#[async_trait::async_trait]` makes every `async fn` return a
/// `Pin<Box<dyn Future + Send>>`, which is required for `Box<dyn Embedder>`
/// and `Arc<dyn Embedder>` to be dyn-compatible.
#[async_trait::async_trait]
pub trait LocalEmbedder: Send + Sync + 'static {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>>;

    fn model_name(&self) -> &str;

    fn dimension(&self) -> usize;

    /// Stable blake3 hash of (model_name || dimension || vestige-core crate version).
    /// Lowercase hex, 64 chars.
    ///
    /// Used by `MemoryStore::register_model` to detect silent model drift
    /// (e.g. a fastembed minor upgrade that changes vector output).
    fn model_hash(&self) -> String;

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>>;

    /// Returns the `ModelSignature` describing this embedder. Convenience
    /// wrapper over the three accessors above.
    fn signature(&self) -> crate::storage::ModelSignature {
        crate::storage::ModelSignature {
            name: self.model_name().to_string(),
            dimension: self.dimension(),
            hash: self.model_hash(),
        }
    }
}

/// Type alias: `Embedder` is the dyn-compatible, Send+Sync variant.
/// Both names refer to the same `async_trait`-annotated trait.
pub use LocalEmbedder as Embedder;
```

### After

`crates/vestige-core/src/embedder/mod.rs:1-55` (approximately):

```rust
//! Text-to-vector encoding trait. Pluggable per-install.

mod fastembed;

pub use fastembed::FastembedEmbedder;

/// Error returned by every `Embedder` method.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("embedder initialization failed: {0}")]
    Init(String),
    #[error("embedding generation failed: {0}")]
    EmbedFailed(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type EmbedderResult<T> = std::result::Result<T, EmbedderError>;

/// Pluggable embedder. The storage layer NEVER calls fastembed directly;
/// callers compute vectors via this trait and pass them into `MemoryStore`.
///
/// `LocalEmbedder` is the source-of-truth trait. The
/// `#[trait_variant::make(Embedder: Send)]` attribute auto-generates an
/// `Embedder` variant whose returned futures are `Send`, so
/// `Box<dyn Embedder>` and `Arc<dyn Embedder>` are usable on tokio/axum
/// runtimes, while `Box<dyn LocalEmbedder>` remains usable on single-
/// threaded executors and thread-local backends.
///
/// Every method is native async-fn-in-trait (stable on MSRV 1.91); no
/// per-call heap allocation, no boxed futures at the static-dispatch
/// boundary.
#[trait_variant::make(Embedder: Send)]
pub trait LocalEmbedder: Sync + 'static {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>>;

    fn model_name(&self) -> &str;

    fn dimension(&self) -> usize;

    /// Stable blake3 hash of (model_name || dimension || vestige-core crate version).
    /// Lowercase hex, 64 chars.
    ///
    /// Used by `MemoryStore::register_model` to detect silent model drift
    /// (e.g. a fastembed minor upgrade that changes vector output).
    fn model_hash(&self) -> String;

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>>;

    /// Returns the `ModelSignature` describing this embedder. Convenience
    /// wrapper over the three accessors above.
    fn signature(&self) -> crate::storage::ModelSignature {
        crate::storage::ModelSignature {
            name: self.model_name().to_string(),
            dimension: self.dimension(),
            hash: self.model_hash(),
        }
    }
}
```

### Both halves of the macro-generated output (for reviewer clarity)

`trait_variant::make(Embedder: Send)` expands the source-of-truth
`LocalEmbedder` declaration above into the equivalent of:

```rust
// 1. The source-of-truth trait, exactly as written.
pub trait LocalEmbedder: Sync + 'static {
    fn embed(&self, text: &str) -> impl Future<Output = EmbedderResult<Vec<f32>>>;
    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn model_hash(&self) -> String;
    fn embed_batch(&self, texts: &[&str]) -> impl Future<Output = EmbedderResult<Vec<Vec<f32>>>>;
    fn signature(&self) -> crate::storage::ModelSignature { /* default impl unchanged */ }
}

// 2. The generated Send variant.
pub trait Embedder: Sync + 'static {
    fn embed(&self, text: &str) -> impl Future<Output = EmbedderResult<Vec<f32>>> + Send;
    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn model_hash(&self) -> String;
    fn embed_batch(&self, texts: &[&str]) -> impl Future<Output = EmbedderResult<Vec<Vec<f32>>>> + Send;
    fn signature(&self) -> crate::storage::ModelSignature { /* default impl unchanged */ }
}

// 3. The blanket impl that wires any LocalEmbedder + Send through to Embedder.
impl<T> Embedder for T
where
    T: LocalEmbedder + Send,
    // (all returned futures of LocalEmbedder's async fns are required to be Send,
    //  which is satisfied for FastembedEmbedder -- see "Risks" below)
{ /* forwarders */ }
```

Notes:

- The `pub use LocalEmbedder as Embedder;` line on the current
  `embedder/mod.rs:57` is **deleted** entirely. `Embedder` is now the
  trait that `trait_variant::make` emits at the same module path; the
  re-export in `crates/vestige-core/src/lib.rs:167`
  (`pub use embedder::{Embedder, ..., LocalEmbedder};`) keeps resolving
  unchanged.
- `Sync + 'static` on `LocalEmbedder` (and no `Send` bound on the trait
  itself) mirrors the `0001a` pattern for `LocalMemoryStore`. The current
  trait carries `Send + Sync + 'static`; the rewrite drops the `Send`
  bound from the local variant. `Box<dyn LocalEmbedder>` is `Sync` but
  not `Send`; `Box<dyn Embedder>` (the generated variant) is `Send + Sync`.
- `trait_variant` 0.1 does **not** require any attribute on the impl
  block. The attribute lives only on the trait declaration. See next
  section.

---

## Impl Block Migration

`trait_variant` 0.1 attaches the attribute only to the trait declaration.
The impl side is plain `impl LocalEmbedder for FastembedEmbedder`; no
attribute on the impl, no `#[trait_variant::make(Embedder: Send)]` on the
impl block. The macro auto-generates the blanket
`impl<T: LocalEmbedder + Send> Embedder for T`, so any concrete type that
implements `LocalEmbedder` automatically also implements `Embedder`
provided it is `Send`.

`FastembedEmbedder` is `Send + Sync` because:

- `inner: EmbeddingService` is `Send + Sync` (it wraps fastembed's
  `TextEmbedding` which is `Send + Sync` after fastembed 4.x; verified
  in `crates/vestige-core/src/embeddings/mod.rs`).
- `cached_hash: std::sync::OnceLock<String>` is `Send + Sync` for `T: Send + Sync`.
- The `#[cfg(not(feature = "embeddings"))]` branch carries only
  `cached_hash`, which is unconditionally `Send + Sync`.

No new bound is needed.

### Before

`crates/vestige-core/src/embedder/fastembed.rs:38-100` (relevant header):

```rust
impl Default for FastembedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl LocalEmbedder for FastembedEmbedder {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        // ... body unchanged ...
    }

    fn model_name(&self) -> &str { /* ... */ }
    fn dimension(&self) -> usize { /* ... */ }
    fn model_hash(&self) -> String { /* ... */ }

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        // ... body unchanged ...
    }
}
```

### After

`crates/vestige-core/src/embedder/fastembed.rs:38-99` (one fewer line):

```rust
impl Default for FastembedEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalEmbedder for FastembedEmbedder {
    async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        // ... body unchanged ...
    }

    fn model_name(&self) -> &str { /* ... */ }
    fn dimension(&self) -> usize { /* ... */ }
    fn model_hash(&self) -> String { /* ... */ }

    async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        // ... body unchanged ...
    }
}
```

Diff is exactly one removed line (the `#[async_trait::async_trait]`
attribute on line 44). Every method body, every `async fn` signature,
every `use` statement inside the impl block stays verbatim. No
`Box::pin(async move { ... })`, no manual `Pin<Box<dyn Future>>`, no
`'async_trait` lifetime markers; native async-fn-in-trait does this
directly.

### Why the impl block does not need an attribute

`trait_variant::make` generates two things from the source trait
(see the "macro-generated output" subsection above):

1. The source trait itself (`LocalEmbedder`), with native async fns.
2. A second trait (`Embedder`) whose method signatures return
   `impl Future<Output = ...> + Send` instead of `impl Future<Output = ...>`,
   plus a blanket impl wiring concrete types through.

Both are emitted at the macro-call site. `FastembedEmbedder` writes one
impl block (against `LocalEmbedder`); the macro-generated blanket
guarantees `FastembedEmbedder: Embedder` for free. The
`Box<dyn Embedder>` cast on `tests/phase_1/embedder_trait.rs:19` keeps
type-checking unchanged.

---

## Call Site Audit

Verified via, from the phase2 worktree root:

```bash
grep -rn "async_trait\|LocalEmbedder\|dyn Embedder" crates/
grep -rn "use.*Embedder" crates/ tests/ --include="*.rs"
grep -rn "Box<dyn Embedder>\|Arc<dyn Embedder>\|&dyn Embedder" crates/ tests/ --include="*.rs"
grep -rn "Box<dyn LocalEmbedder>\|Arc<dyn LocalEmbedder>\|&dyn LocalEmbedder" crates/ tests/ --include="*.rs"
grep -rn "impl LocalEmbedder for\|impl Embedder for" crates/ tests/ --include="*.rs"
```

### Files that reference the trait object form

Exactly one, test-only:

| File                                 | Line | Use                                                                                      | Required change |
|--------------------------------------|------|------------------------------------------------------------------------------------------|-----------------|
| `tests/phase_1/embedder_trait.rs`    | 3    | `use vestige_core::embedder::{Embedder, FastembedEmbedder};`                             | None. `Embedder` is the generated Send variant at the same path. |
| `tests/phase_1/embedder_trait.rs`    | 19   | `let e: Box<dyn Embedder> = Box::new(FastembedEmbedder::new());`                         | None. `FastembedEmbedder: LocalEmbedder + Send` -> blanket gives `: Embedder` -> `Box<dyn Embedder>` is well-formed. |

### Files that import `Embedder` or `LocalEmbedder` by name

| File                                                | Line | Use                                                                                                            | Required change |
|-----------------------------------------------------|------|----------------------------------------------------------------------------------------------------------------|-----------------|
| `crates/vestige-core/src/lib.rs`                    | 167  | `pub use embedder::{Embedder, EmbedderError, EmbedderResult, FastembedEmbedder, LocalEmbedder};`               | None. Both names still resolve. |
| `crates/vestige-core/src/embedder/mod.rs`           | 5    | `pub use fastembed::FastembedEmbedder;`                                                                        | None. |
| `crates/vestige-core/src/embedder/fastembed.rs`     | 7    | `use super::{EmbedderError, EmbedderResult, LocalEmbedder};`                                                   | None. |
| `tests/phase_1/embedder_trait.rs`                   | 3    | `use vestige_core::embedder::{Embedder, FastembedEmbedder};`                                                   | None. |

### Files that implement the trait

| File                                                | Line | Impl                                                                  | Required change                              |
|-----------------------------------------------------|------|-----------------------------------------------------------------------|----------------------------------------------|
| `crates/vestige-core/src/embedder/fastembed.rs`     | 45   | `impl LocalEmbedder for FastembedEmbedder` (currently `#[async_trait]`) | Drop the `#[async_trait::async_trait]` attr. |

No other impls exist. There is no test mock implementing `Embedder` or
`LocalEmbedder` anywhere in the workspace.

### Files that import `async_trait` directly

After `0001a` lands, only the embedder pair:

```
crates/vestige-core/src/embedder/mod.rs:24    (doc comment)
crates/vestige-core/src/embedder/mod.rs:27    (attribute)
crates/vestige-core/src/embedder/mod.rs:56    (doc comment)
crates/vestige-core/src/embedder/fastembed.rs:44  (attribute)
```

Plus the Cargo manifest:

```
crates/vestige-core/Cargo.toml:119:async-trait = "0.1"
```

### Production files that hold a concrete embedder

`FastembedEmbedder` is constructed and used by concrete name (not via
trait object) in: the dashboard / MCP layer if it needs to embed query
strings ad-hoc. None of those call sites need an edit because the
concrete type is what they hold, and the concrete type is untouched by
this sub-plan.

### Conclusion

| Category                                         | Count |
|--------------------------------------------------|-------|
| Production source files modified                 | 2     |
| Test source files modified                       | 0     |
| Cargo manifests modified                         | 1     |
| Production source files importing `Embedder` / `LocalEmbedder` (verified unchanged) | 3 |
| Test source files importing `Embedder` (verified unchanged) | 1 |
| Direct `async_trait` uses outside the embedder module after `0001a` | 0 |

---

## Cargo.toml Change

From inside `crates/vestige-core/`:

```bash
cargo rm async-trait
```

This removes line 119 of `Cargo.toml` and updates `Cargo.lock` in one
step. Manual editing is acceptable as a fallback if `cargo rm` is
unavailable in the agent environment; in that case, follow up with
`cargo check -p vestige-core` to refresh the lockfile.

### Verification

```bash
# Direct dependency must be gone.
grep -rn "async-trait\|async_trait" --include="Cargo.toml" crates/
# Expected: empty.

# Transitive presence is permitted (e.g. via a third-party crate).
cargo tree -p vestige-core --depth 2 | grep async-trait
# Expected: empty for the direct edges; if a sub-dependency still pulls
# async-trait transitively, the output may contain it deeper than depth 2,
# which is fine. We only care about removing the DIRECT edge.
```

If `cargo tree --depth 2` returns any `async-trait` line, inspect with
`cargo tree -p vestige-core -i async-trait` to see what is pulling it.
Acceptable parents: any third-party crate. Unacceptable parent: anything
under `vestige-*`, which would mean a missed file.

---

## Commit Sequence

Three commits, each green on
`cargo test -p vestige-core --features embeddings,vector-search` and
`cargo test -p vestige-core --no-default-features`.

### Commit 1: rewrite LocalEmbedder trait declaration

- Touches: `crates/vestige-core/src/embedder/mod.rs` only.
- Action: replace lines 21-57 per the "Trait Declaration Rewrite"
  section above. Delete the `pub use LocalEmbedder as Embedder;` line.
- Green after: `cargo check -p vestige-core` (the impl block in
  `fastembed.rs` still has its `#[async_trait::async_trait]` attribute;
  the macro is harmless when applied to a trait that the impl block
  targets by path, because async_trait rewrites the impl's async fns
  into boxed-future fns whose signatures still match the native-async
  declarations after trait_variant lowering, just as it did for the
  SQLite intermediate state in `0001a`'s commit 1).

  **Mitigation if check fails between commits 1 and 2:** combine the
  two into a single commit. The split is offered for review convenience;
  the build must be green after every commit lands.

### Commit 2: drop `#[async_trait::async_trait]` from FastembedEmbedder impl

- Touches: `crates/vestige-core/src/embedder/fastembed.rs` only.
- Action: delete line 44 (`#[async_trait::async_trait]`).
- Green after:
  - `cargo test -p vestige-core --features embeddings,vector-search`.
  - `cargo test -p vestige-core --no-default-features` (the
    `#[cfg(not(feature = "embeddings"))]` branches inside the impl now
    stand on their own).
  - Phase 1 integration test: `cargo test --test embedder_trait
    --features embeddings,vector-search`.

### Commit 3: drop the async-trait dependency

- Touches: `crates/vestige-core/Cargo.toml` (plus `Cargo.lock` as a
  side effect).
- Action: from inside `crates/vestige-core/`, run `cargo rm async-trait`.
- Green after: `cargo build --workspace --all-targets` and
  `cargo test --workspace`.
- Final hard ASCII gate: `! grep -rn "async_trait" crates/` must exit
  with status 0 (i.e. the inverted grep finds nothing).

### Combined alternative

Commits 1 and 2 may fold into a single commit if the per-step split
feels artificial (the patterns are identical to `0001a`'s commits 3
and 4). Commit 3 (the Cargo.toml removal) should stay separate so the
dependency-removal diff is visible in isolation.

---

## Verification

Every command runs from the repo root unless noted otherwise.

```bash
# 1. Vestige-core, default features (embeddings + vector-search).
cargo test -p vestige-core --features embeddings,vector-search

# 2. Vestige-core, minimal features (no embeddings, no vector-search).
cargo test -p vestige-core --no-default-features

# 3. Workspace build, all targets (catches any feature-gated regression
#    in the vestige-mcp tools tree).
cargo build --workspace --all-targets

# 4. Whole-workspace test (vestige-mcp 406 tests + vestige-core 352
#    tests per the CLAUDE.md baseline).
cargo test --workspace

# 5. Phase 1 embedder integration test (the trait-shape contract).
cargo test --test embedder_trait --features embeddings,vector-search

# 6. Clippy gate, deny warnings (matches Phase 1 PR policy).
cargo clippy --workspace --all-targets --features embeddings,vector-search -- -D warnings

# 7. Hard ASCII gate -- async_trait must be gone from source.
! grep -rn "async_trait" crates/
# Inverted grep: exit 0 iff grep found nothing.

# 8. Hard ASCII gate -- async-trait must be gone from manifests.
! grep -rn "async-trait" --include="Cargo.toml" crates/

# 9. Confirm trait_variant attribute is in place at the embedder.
grep -rn "trait_variant::make" crates/vestige-core/src/embedder/
# Expected: exactly one hit, in embedder/mod.rs.

# 10. Workspace-wide trait_variant audit (should match the count after
#     0001a -- two hits total, one for storage, one for embedder).
grep -rn "trait_variant::make" crates/vestige-core/src/
# Expected: two hits.
```

Expected outcomes:

- Command 1: 352 vestige-core tests pass (matches baseline).
- Command 2: smaller test count, all pass.
- Command 3: workspace builds in dev mode for all targets.
- Command 4: 758 total tests pass (matches CLAUDE.md baseline).
- Command 5: `embedder_trait` integration test passes. The
  `fastembed_implements_embedder_trait` assertion (`let e: Box<dyn
  Embedder> = ...`) is the canary; if `trait_variant::make` failed to
  emit the `Embedder` Send variant, this fails to compile.
- Command 6: zero clippy warnings.
- Command 7: empty output. `async_trait` is fully gone from source.
- Command 8: empty output. `async-trait` is fully gone from manifests.
- Command 9: one hit.
- Command 10: two hits.

---

## Acceptance Criteria

A reviewer should be able to check every box:

- [ ] `crates/vestige-core/src/embedder/mod.rs` declares the embedder
      trait with `#[trait_variant::make(Embedder: Send)] pub trait
      LocalEmbedder: Sync + 'static`, no `async_trait` attribute, no
      `Send` bound on `LocalEmbedder` itself.
- [ ] `crates/vestige-core/src/embedder/mod.rs` no longer contains
      `pub use LocalEmbedder as Embedder;`.
- [ ] `crates/vestige-core/src/embedder/fastembed.rs` declares
      `impl LocalEmbedder for FastembedEmbedder` with no attribute on
      the impl block.
- [ ] `crates/vestige-core/Cargo.toml` does not declare `async-trait`
      as a direct dependency.
- [ ] `grep -rn "async_trait" crates/` returns zero hits.
- [ ] `grep -rn "async-trait" --include="Cargo.toml" crates/` returns
      zero hits.
- [ ] `grep -rn "trait_variant::make" crates/vestige-core/src/` returns
      exactly two hits (storage trait + embedder trait).
- [ ] All 758 workspace tests pass (`cargo test --workspace`).
- [ ] `tests/phase_1/embedder_trait.rs` compiles and passes with the
      `Box<dyn Embedder>` cast intact.
- [ ] `cargo clippy --workspace --all-targets --features
      embeddings,vector-search -- -D warnings` is clean.
- [ ] No file under `crates/vestige-mcp/` or under
      `crates/vestige-core/src/{neuroscience,advanced,consolidation,
      codebase,memory,embeddings}/` was modified by this sub-plan.
- [ ] `Cargo.lock` was updated as a side effect of `cargo rm async-trait`
      (it must no longer reference `async-trait`).
- [ ] Doc comments on the embedder trait declaration describe
      `trait_variant`, not `async_trait`.

---

## Risks and Mitigations

- **`trait_variant::make` requires returned futures to be `Send` for the
  blanket `impl<T: LocalEmbedder + Send> Embedder for T`. If any
  `async fn embed`/`embed_batch` body inside `FastembedEmbedder` captures
  a non-Send local, the blanket impl fails to type-check.**
  Mitigation: the existing impl bodies call `self.inner.embed(text)` /
  `self.inner.embed_batch(texts)`, where `inner: EmbeddingService` is
  `Send + Sync` (verified in `crates/vestige-core/src/embeddings/mod.rs`).
  No `.await` points exist inside the bodies in either feature branch;
  the `EmbeddingService::embed` calls are synchronous. The futures are
  trivially `Send`. If a future change introduces a non-Send local
  (e.g. an `Rc` or a non-Send guard), the blanket impl will surface that
  as a compile error at the dyn cast in `tests/phase_1/embedder_trait.rs`,
  which is the correct outcome.
- **The macro's blanket impl interacts oddly with the default `signature`
  method.**
  Mitigation: `signature` is a synchronous method returning
  `crate::storage::ModelSignature`, with no `Send` or `async` concerns.
  `trait_variant::make` emits it on both variants as-is. The existing
  Phase 1 test `signature_matches_memory_store_registry` exercises this
  path and is part of the verification step.
- **`Box<dyn Embedder>` cast in `tests/phase_1/embedder_trait.rs` fails
  to resolve after the rewrite.**
  Mitigation: the rewrite preserves the `Embedder` symbol at the same
  module path; only its provenance changes (now generated by
  `trait_variant::make` instead of by `pub use LocalEmbedder as
  Embedder;`). The macro is specifically designed so that the generated
  trait is dyn-compatible at the Send-bound boundary. Verified by the
  identical pattern already working for `MemoryStore` after `0001a`.
- **`cargo rm async-trait` updates `Cargo.lock` but accidentally bumps
  other crates.**
  Mitigation: run `cargo rm async-trait` and then immediately inspect
  the resulting `Cargo.lock` diff. The expected diff is the removal of
  the `[[package]] name = "async-trait"` block and its hash. Anything
  else is a red flag and should be reverted before committing
  (`git checkout -- Cargo.lock` then `cargo update -p async-trait
  --precise=remove` -- or fall back to manual edit + `cargo check`).
- **A new workspace crate added in parallel with this work declares
  `async-trait` and the dependency removal silently re-introduces it
  later.**
  Mitigation: the verification step `grep -rn "async-trait"
  --include="Cargo.toml" crates/` is part of the acceptance criteria; a
  rebase that reintroduces the line will fail this gate.
- **MCP server uses `Embedder` somewhere we missed.**
  Mitigation: full workspace grep (`grep -rn "Embedder" crates/`)
  returns no hits inside `crates/vestige-mcp/` for the trait names; the
  MCP layer uses the concrete `EmbeddingService` from
  `crates/vestige-core/src/embeddings/` for ad-hoc embedding calls. The
  trait surface is purely internal to `vestige-core`.

---

## Out-of-Band Notes

- **No other workspace crate declares `async-trait` as a direct
  dependency.** Verified by
  `grep -rn "async-trait" --include="Cargo.toml" crates/` returning
  exactly one hit at `crates/vestige-core/Cargo.toml:119`. There is
  nothing to clean up in `crates/vestige-mcp/Cargo.toml` or elsewhere.
- **Order matters across the three Phase 1 amendment sub-plans:**
  `0001a` (trait rewrite) -> `0001b` (sqlite split) -> `0001c` (this
  one, async-trait sunset). Reversing the order is possible in
  principle but would force re-editing the embedder rewrite twice and
  leaves the `async-trait` dep behind until very late.
- **This sub-plan amends `feat/storage-trait-phase1` (tip 790c0c8 plus
  whatever commits `0001a` and `0001b` added).** The branch has not
  been opened upstream yet, so amending in place is safe; no force-push
  to a public PR.
- **After this sub-plan lands, the branch is reviewed and merged before
  Phase 2 sub-plans (`0002a-` through `0002i-`) begin implementation.**
  Phase 2 introduces no async-trait usage; the Postgres backend follows
  the same `trait_variant::make` pattern (see ADR 0002 D1).
- **`trait-variant` 0.1 stays in `Cargo.toml`.** It is the only crate
  this sub-plan keeps; `async-trait` is the only one it removes.

---

## Self-Contained `/goal` Brief

For a fresh Claude Code session executing this sub-plan without prior
conversation context:

1. Check out branch `feat/storage-trait-phase1` (or a worktree off
   of it after `0001a` and `0001b` are merged into it).
2. Read this file (`docs/plans/0001c-async-trait-sunset.md`) in full.
3. Read `docs/plans/0001a-trait-rewrite.md` sections "Trait declaration
   rewrite" and "Impl block migration" -- they document the exact
   pattern this sub-plan mirrors for the embedder.
4. Run the prerequisite audit grep listed under "Prerequisites". If it
   returns more than the five hits documented there, stop and report;
   the upstream state does not match what this sub-plan assumes.
5. Execute Commit 1 (rewrite `embedder/mod.rs`), then Commit 2 (drop
   the attribute on the FastembedEmbedder impl), then Commit 3
   (`cargo rm async-trait`). Run the verification commands listed
   above after each commit; do not proceed if any test or clippy gate
   fails.
6. Verify every box in "Acceptance Criteria" is ticked.
7. Report file paths touched, test counts, and the final two grep
   results (commands 7 and 8 from "Verification") in the closing
   message.
