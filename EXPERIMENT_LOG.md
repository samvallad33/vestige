# Vestige Retrieval Experiment Log

Monotonic Zero-Regression Gate tracking across Steps 0–4.

## Pre-Flight (`894c7d44`): Deduplicate C2 Hooks + Add Capped-All@5 Metric
- **Lines Touched**:
  - `crates/vestige-core/src/retrieval.rs`: lines `6442-6445` (dedup `apply_c2_pool_c_absent_session_key`), lines `6475-6479` (dedup `apply_c2_anaphoric_tail`).
  - `benchmarks/run_stage1_retrieval.mjs`: lines `213-214`, `228`, `233-240`, `248`, `256`, `260`, `270`.
  - `EXPERIMENT_LOG.md`: lines `1-14`.
- **Stage Funnel**: `LME-S` `@1=91.49%`, `All@5=89.79%`, `Capped-All@5=95.10%`, `All@10=94.89%` | `BEAM-100K` `Turn@1=46.20%`, `Turn@5=67.89%`, `All@5=35.21%` (baseline `pool@50`/`pre_rank@5`/`post_kemeny@5` parity).

## Step 0 (`45fa0769`): Parity Speed & Cache Foundation (Sedgewick `u32` Postings + Burton Bloom `u64` + `#[repr(C)] TurnMeta`)
- **Lines Touched**:
  - `crates/vestige-core/src/retrieval.rs`: lines `1990-2165` (`s_postings`/`w_postings`/`t_postings`, `session_bloom`/`turn_bloom`, `#[repr(C)] TurnMeta` <=24B, `bloom_mask64`, `build_postings`, `collect_posting_candidates`), lines `4009-4015` & `4046-4051` (`build_index`), lines `4453-4533` (`PreparedPlan`, `prepare_plan`, `bm25f_prepared`), lines `4804-4816` & `4941-4946` (`rank_one`).
  - `EXPERIMENT_LOG.md`: lines `15-26`.
- **Gate Verdict**: **ACCEPTED** (exact 0-diff parity across `LME-S`, `LoCoMo`, `BEAM-100K`; RSS `322 MB`).

## Step 1: Zero-Overlap & Open-Domain Bridge (Ounis QPP + Wilkins `u64` + Mooers `u128` Zatocode + Hovy/Goolsbey/Lieberman + Forbus SME)
- **Lines Touched**:
  - `crates/vestige-core/src/zero_overlap.rs`: lines `1-410` (pure integer zero-float taxonomic/Zatocode/SME module).
  - `crates/vestige-core/src/lib.rs`: line `123` (`pub(crate) mod zero_overlap;`).
  - `crates/vestige-core/src/retrieval.rs`: lines `2002-2004` (`turn_zero_overlap`, `zero_overlap_candidates`), lines `3185-3186` & `3531-3535` & `4062-4063` (`build_index`), lines `5046-5157` (Ounis QPP gate + top-15 zero-overlap bridge), lines `6829-6839` (final slot-0 seating for zero-overlap hits).
- **Verification (`cargo test`: 55/55 passed, 0 warnings)**:
  - **NoLiMa 32K (`needle_set`, n=58, `16.82s`)**: `0.00% -> R@1=79.31% R@5=93.10% R@10=93.10%` (`world_knowledge 90.5%`, `commonsense_knowledge 50.0%@1 / 100.0%@5`)
  - **NoLiMa MC Guard (`n=58`, `16.87s`)**: `R@1=96.55% R@5=100.00% R@10=100.00%` (PASS >= 96.55%)
  - **LongMemEval_S (`470 non-abs`)**: `RecallAny@1=91.49% | RecallAny@5=98.51% | RecallAny@10=99.79% | RecallAll@5=89.79% | Capped-All@5=95.10% | RecallAll@10=94.89% | NDCG@10=0.9385` (0-diff parity)
  - **LoCoMo-10 (`1982 scored`)**: `CORE Session R@1=74.02% (+0.26%) R@5=92.58% (+0.13%) | Turn Exact R@1=52.86% (+0.19%) R@5=73.57% (+0.13%) R@10=80.47% (+0.13%)`
  - **BEAM-100K (`355/400`)**: `Sess Any@1/5/10=37.46%/70.14%/77.46% All@5=38.59% Capped-All@5=51.36% | Turn @1/5/10=46.20%/67.89%/76.62% All@5=35.21% Capped-All@5=48.38% All@10=41.13%` (0-diff parity)
- **Gate Verdict**: **ACCEPTED** (`NoLiMa 32K +79.31% R@1 / +93.10% R@5`, `LoCoMo +0.26%`, zero regressions).

