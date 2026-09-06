# The v2.6.1 → v2.7 Plan — code strength, bleeding edge, uniqueness

Written 2026-08-30, hours after v2.6.0 shipped. Two evidence streams: a
line-level read of every module that makes Vestige unique (plus mechanical
metrics over the ~90k-line core), and a full frontier scour of the August
2026 Rust/dependency/research/MCP landscape, sourced claim by claim. Sliced
into releases per the new cadence: 2-3 releases a week, spread out; only
data-safety ships same-day.

## Ground truth from the line-level read

Hygiene is already elite: zero unwrap/expect in non-test code across the
entire neuroscience and advanced arsenal, 3 TODOs, 18 audited allows. The
one hygiene hotspot is `embedding/lifecycle.rs` (26 unwraps). The real debt
is structural:

- `storage/sqlite.rs`: 21,456 lines, 516 functions, five functions over 200
  lines (`run_consolidation` 468, `smart_ingest_*` 384,
  `run_integrity_checks` 294, `write_archive` 285, `auto_dedup` 229).
- ~10 independent cosine/dot implementations scattered across the tree
  (embeddings, synaptic store, hippocampal index, prediction error, dreams,
  compression, lifecycle, spacetime). Any optimization applied to one is
  wasted on the other nine.
- `spreading_activation.rs` defects: the "BFS" is a stack-based DFS;
  `activate()` can return duplicate memory ids that crowd real associations
  out of top-N; `allow_cycles` is dead config; edge decay has no caller.
- Backfill promotion has no MAX_STABILITY clamp at the write site (the #121
  compounding family). One stale FSRS comment (says D0(3), code correctly
  does D0(4)).
- Zero Rust micro-benchmarks exist. Every perf claim to date is end-to-end.
- The July `feat/consolidation-cleanup` branch already proved the
  consolidation decomposition (commit 005109a) and a −479-line mechanical
  modernization (3376ecc); both re-derivable on current main.

## Frontier verdicts (sourced in the Aug 30 scour)

- **Rust 1.98: DO NOT BUMP YET.** Open critical stable-to-stable miscompile
  rust-lang/rust#161441 — vacant vtable slot for boxed async services,
  reproduced on aarch64-apple-darwin — is our exact axum/tower shape on our
  primary target. Re-evaluate at the first 1.98.x point release or 1.99
  (Oct 1). The new deny-by-default runtime-symbol lint does NOT affect our
  bundled SQLite / linked ONNX Runtime (it fires on Rust code only).
- **Algebraic float methods** (1.98) are the right eventual tool for our
  scalar-accumulator vector loops (published results: 3.9x on summation
  shapes; plain iterator float loops provably do not auto-vectorize) — but
  chunked multi-accumulator kernels bank most of that win on stable 1.97.1
  today, deterministically. Consolidate first, swap `algebraic_*` in behind
  one seam when the toolchain moves. `std::simd` remains nightly-only; the
  `wide` crate is the stable fallback if benches demand more.
- **fastembed 5.13.2 → 6.0.2**: fixes Qwen3 and nomic-v2 dtype bugs in
  models we feature-gate; adds ORT placement controls aimed exactly at
  user-defined loaders like our Granite runner; only documented breaking
  change is the Error type becoming a real enum. Brings ort 2.0.0-rc.13
  (ONNX Runtime 1.28) and candle 0.11 as one coherent unit.
- **usearch =2.23.0 → 2.26.1, gated**: NumKong v7 SIMD kernels, tombstone
  reclamation, and 2.26.1's bounded file-controlled sizes (memory-safety
  hardening for parsing user-disk index files). The Windows MSVC breakage
  that forced the pin (unum-cloud/usearch#746) is still open — attempt the
  unpin behind a manual MSVC build check, keep fp16lib, revert freely.
- **rusqlite 0.40.2 is current**; SQLite 3.53.3/3.53.4 are upstream patch
  releases not yet bundled by libsqlite3-sys; 3.54.0 drafted for Oct 15.
  Stay; take the bundle bump when it exists.
- **MCP 2026-07-28** removed the initialize handshake, sessions, and
  server-initiated sampling in favor of `_meta` self-describing requests,
  `server/discover` (which we already ship), and `resultType`/MRTR. Formal
  deprecation guarantees ≥12 months, and our 2025-11-25 revision stays
  negotiable — but the client SDKs (rmcp 3.x, TS v2, C# v2) all moved
  day-and-date. Dual-revision serving is the intended compat path.
- **Research wave (Jun-Aug 2026) validates our lanes**: deterministic
  conflict resolution beats LLM judgment (arXiv:2606.01435); bitemporal
  operator algebra for contradictions (TOKI, 2606.06240); superseded-memory
  penalties (2606.27472); forgetting-placement studies (2606.15903); and
  Lin et al. (Science Advances) shows retroactive salience enhancement is
  GRADED and NONMONOTONIC — low-intensity salience spreads further, and
  spread follows embedding proximity. Direct tuning guidance for backfill.
- **Competitive movement**: ZenBrain (arXiv:2604.23878) is the first
  project publicly claiming FSRS + consolidation for agent memory (Python,
  research-grade) — the unqualified "only FSRS memory system" line retires;
  the durable claim is "FSRS-6 + deterministic contradiction resolution +
  retroactive salience in one local Rust binary." MemOS shipped local
  SQLite+FTS5+vector plugins and is the direct local-first rival. mem0's
  own 2026 survey names staleness an open problem and shows its scores
  collapsing at 10M tokens — the lifecycle thesis remains ours.

## Release slicing

### v2.6.1 — "Sharper" (quality + dependencies; next release slot)

1. `vector_math` consolidation: one module, chunked-accumulator kernels,
   all ~10 call sites migrated; criterion benches (cosine, truncate+
   renormalize, fuse_rank_native, appears_contradictory) land first and
   gate the numbers; `algebraic_*` seam documented for the toolchain day.
2. fastembed 6.0.2 (+ ort rc.13, candle 0.11) as one tested unit across
   all feature combos; Granite loader error-type touch-ups.
3. usearch 2.26.1 attempt, gated on a manual Windows MSVC build; fp16lib
   pinned; independently revertable commit.
4. Correctness batch: spreading-activation dedup + honest best-first
   traversal + implement-or-remove allow_cycles + wire-or-remove edge
   decay; backfill MAX_STABILITY clamp; FSRS comment fix; lifecycle.rs
   unwrap audit.
5. Tooling: cargo-nextest in CI; proptest invariants for vector math and
   FSRS-6 (approximate assertions from day one); cargo-shear check.
   cargo-mutants as a weekly scheduled job scoped to contradiction + FSRS.
6. NO toolchain bump (see frontier verdict).

### v2.6.2 — "Lighter" (structure; pure code motion, same week)

1. Split `sqlite.rs` along its 46 region markers into cohesive storage
   submodules, one module per commit, tests moving with their subjects.
2. Re-derive the July consolidation decomposition (orchestrator + 9 named
   cognitive methods) and give the other four god-functions the same
   treatment.
3. Re-run the mechanical modernization (−479 lines in July) with the
   recorded double-gate for the map_or/is_none_or cascade.

### v2.7.0 — "Unmistakable" (uniqueness; following week)

1. IDF-weighted causal join for backfill (entity rarity from a store
   frequency index) + Lin-informed graded weighting: proximity-weighted,
   capped salience multiplier. Measured on plant-cause probes on the real
   store before merge.
2. Entity extraction v2: camelCase + lowercase_snake identifiers (most
   Rust/Python symbols are currently invisible to the causal join);
   entity-pool precision measured before/after.
3. ACT-R fan effect (Anderson 1974) in spreading activation: divide by
   out-degree, S − ln(fan); MemConflict-gated like every ranking change.
4. MCP 2026-07-28 dual-revision support: `_meta` negotiation +
   `resultType` alongside the legacy handshake; full 14-tool regression
   against both revisions.
5. Reversible auto-merge (apply_merge_op extraction) — completes #142.
6. Wire-or-cull the two dormant channels (#124 recurring intentions, #137
   predict channels).

### Parked for v2.7.1+, explicitly

- Bitemporal validity metadata for the contradiction path — the strongest
  research-validated headline candidate (2606.01435, TOKI), but it touches
  schema and as-of semantics we just stabilized. It deserves its own
  release with its own measurement story.
- Toolchain 1.98.x/1.99 + algebraic swap-in, the day the miscompile fix
  ships.
- loom modeling of the storage lock graph (pairs with the stage-5B
  try_lock decision — its own session).

The through-line stands: v2.5.0 proved the audit discipline, v2.6.0 proved
the data-safety discipline, and this arc makes the code itself carry the
signature — one vector kernel instead of ten, one storage module per
concern instead of a god-file, every claim benched, every ranking change
gated, and the science citations running all the way down into the commit
messages.
