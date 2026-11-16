# Vestige Retrieval Experiment Log

Monotonic Zero-Regression Gate tracking across Steps 0–4.

## Pre-Flight: Deduplicate C2 Hooks + Add Capped-All@5 Metric
- **Changes**:
  - Removed duplicate calls to `apply_c2_pool_c_absent_session_key` and `apply_c2_anaphoric_tail` in `crates/vestige-core/src/retrieval.rs`.
  - Added `Capped-All@5` (`|retrieved_top5 ∩ oracle| / min(5, |oracle|)`) to `benchmarks/run_stage1_retrieval.mjs` and the BEAM evaluation script.
  - Note: `vestige-bench-1115\c2-w2001` (`Envelope-seal-calendar`) was checked across all local worktrees and `origin` remotes and was not present; skipped cleanly.
- **Verification (`cargo test -p vestige-core retrieval::tests`: 55 passed, 0 failed; `cargo build --release` zero warnings)**:
  - **LongMemEval_S (`470 non-abs`)**: `RecallAny@1=91.49% | RecallAny@5=98.51% | RecallAny@10=99.79% | RecallAll@5=89.79% | Capped-All@5=95.10% | RecallAll@10=94.89% | NDCG@10=0.9385` (`avg 1679.77 ms/q`)
  - **LoCoMo-10 (`1982 scored`)**: `CORE Session R@1=73.76% R@5=92.45% | Turn Exact R@1=52.67% R@5=73.44% R@10=80.34%` (`avg 30.70 ms/q`)
  - **BEAM-100K (`355/400`)**: `Sess Any@1/5/10=37.46%/70.14%/77.46% All@5=38.59% Capped-All@5=51.36% | Turn @1/5/10=46.20%/67.89%/76.62% All@5=35.21% Capped-All@5=48.38% All@10=41.13%` (`55.05s`)
- **Gate Verdict**: **ACCEPTED** (exact 0-diff parity with `4de0f6ff`).

## Step 0: Parity Speed & Cache Foundation (Sedgewick `u32` Postings + Burton Bloom `u64` + `#[repr(C)] TurnMeta`)
- **Changes**:
  - Added Sedgewick dense `u32` `TokenId` inverted postings (`s_postings`, `w_postings`, `t_postings`) + zero-alloc `PreparedPlan` / `bm25f_prepared` in `crates/vestige-core/src/retrieval.rs`.
  - Added Burton Bloom 64-bit per-session & per-turn membership filters (`session_bloom`, `turn_bloom`).
  - Added cache-aligned `#[repr(C)] struct TurnMeta` (`<= 24 bytes`, compile-time asserted) in `Index.turn_meta`.
- **Verification (`cargo test -p vestige-core retrieval::tests`: 55 passed, 0 failed; `cargo build --release` zero warnings)**:
  - **LongMemEval_S (`470 non-abs`)**: `RecallAny@1=91.49% | RecallAny@5=98.51% | RecallAny@10=99.79% | RecallAll@5=89.79% | Capped-All@5=95.10% | RecallAll@10=94.89% | NDCG@10=0.9385` (RSS `322 MB`)
  - **LoCoMo-10 (`1982 scored`)**: `CORE Session R@1=73.76% R@5=92.45% | Turn Exact R@1=52.67% R@5=73.44% R@10=80.34%`
  - **BEAM-100K (`355/400`)**: `Sess Any@1/5/10=37.46%/70.14%/77.46% All@5=38.59% Capped-All@5=51.36% | Turn @1/5/10=46.20%/67.89%/76.62% All@5=35.21% Capped-All@5=48.38% All@10=41.13%`
- **Gate Verdict**: **ACCEPTED** (exact 0-diff parity across all three suites).

