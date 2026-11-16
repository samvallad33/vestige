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

## RULE 1 funnel backfill

`tools/c2_bench.py` drives `retrieval_bench --worker` with `VESTIGE_FUNNEL_DUMP` set. The dump is not read by any seat. Each cell is any-gold on the gate questions: LME_S n=470 non-abstention, LoCoMo n=1536 (category 1–4 with evidence), BEAM-100K n=355 (non-empty `source_chat_ids`).

`pool@50` is `scene_pool50` immediately before submodular selection. `pre_rank@5` is the turn list cloned immediately before `apply_step11_kemeny_wh_slot_voter`. `post_kemeny@5` is the turn list cloned immediately after `rank1_gatekeeper_keep_rsb_head`. `@1` is `final_ids[0]`. Session `pre_rank@5` and session `post_kemeny@5` are `final_session_indices[:5]`: that binding is immutable, and the harness aborts if `&mut final_session_indices` appears.

One LoCoMo evidence id never enters the corpus, on every commit measured here. `conv-50:69` cites `D30:05` and the dialogue id is `D30:5`. Gold-in-corpus is 1535/1536. The evidence regex was not loosened.

Capped-All, NDCG, NoLiMa, and the logged BEAM session All@5 were not recomputed by this harness. The percentages below are this harness only.

### Pre-flight `894c7d44` and Step 0 `45fa0769`

`c2_bench.py --diff` across the combined runs: 2361 gate questions, losses=0, gains=0. Step 0 is an exact 0-diff of pre-flight under this harness. Both binaries were the historical commit plus the measurement dump and nothing else.

LME_S n=470. Turn any-gold: corpus 470/470, pool@50 464/470, pre_rank@5 406/470, post_kemeny@5 415/470, @1 360/470. Session any-gold: corpus 470/470, pool@50 470/470, pre_rank@5 463/470, post_kemeny@5 463/470, @1 430/470. Scoreboard turn any@1/@5/@10 all@5 = 360/428/441/323. Session any@1/@5/@10 all@5/all@10 = 430/463/469/422/446 (91.49% / 98.51% / 99.79% / 89.79% / 94.89%).

LoCoMo n=1536. Turn any-gold: corpus 1535/1536, pool@50 1422/1536, pre_rank@5 1037/1536, post_kemeny@5 1098/1536, @1 809/1536. Session any-gold: corpus 1536/1536, pool@50 1536/1536, pre_rank@5 1420/1536, post_kemeny@5 1420/1536, @1 1133/1536. Turn any@1/@5/@10 = 809/1128/1234. Session any@1/@5 = 1133/1420 (73.76% / 92.45%).

BEAM n=355. Turn any-gold: corpus 355/355, pool@50 327/355, pre_rank@5 229/355, post_kemeny@5 234/355, @1 164/355. Session any-gold: corpus 355/355, pool@50 341/355, pre_rank@5 249/355, post_kemeny@5 249/355, @1 133/355. Turn any@1/@5/@10 all@5/all@10 = 164/241/272/125/146 (46.20% / 67.89% / 76.62% / 35.21% / 41.13%). Session any@1/@5/@10 = 133/249/275 (37.46% / 70.14% / 77.46%).

### Step 1 `e81a99b2`

Same dump, ranker unchanged. LME (470) and BEAM (355) are a 0-diff of Step 0, funnel counts included. LoCoMo losses=0, 13 hit-flags gained, four questions: `conv-47:29` (cat 2) turn @1/@5/@10 and session @1/@5; `conv-47:31` (cat 2) turn @1 and session @1/@5; `conv-48:19` (cat 1) turn @1/@5/@10 and session @1; `conv-48:68` (cat 2) session @1.

LoCoMo turn any@1/@5/@10 = 812/1130/1236 (52.86% / 73.57% / 80.47%). Session any@1/@5 = 1137/1422 (74.02% / 92.58%). That is the logged Step 1 CORE: session @1 1133→1137 (+0.26%), turn @1 809→812 (+0.19%), turn @5/@10 and session @5 each +2 (+0.13%).

LoCoMo turn any-gold: corpus 1535/1536, pool@50 1424/1536, pre_rank@5 1039/1536, post_kemeny@5 1100/1536, @1 812/1536. Session any-gold: corpus 1536/1536, pool@50 1536/1536, pre_rank@5 1422/1536, post_kemeny@5 1422/1536, @1 1137/1536.

## Step 2 (`perf/step2-multihop-overwrite`): Nominal terminal promotion

Blackburn nominal anchor plus Wallace terminal-hop promotion. The speaker field is copied at ingest. A turn is a terminal question when its prose, before a code fence, ends on `?`. After every earlier seat, including the Step 1 zero-overlap slot-0 insert, if slot 0 is a terminal question and exactly one of slots 1..4 is a non-question spoken by a different nominal the query names, that turn moves to slot 0. `user` and `assistant` are role labels, not nominals. Two competing nominals do not fire. Top-5 membership is unchanged. `final_session_indices` is not rewritten. `rsb_seat_behind` is not touched. The module has no floating-point values, no `HashMap`, and no domain word list. Ranking reads `ix.multihop` and the query, not turn body.

The attached WIP was not landed. It used `std::collections::HashMap`, domain phrase tables, and floating-point score scalers. A numeric-overwrite simulation of that family on the Step 1 dump lost 5 BEAM questions and gained 0. A broader `is_question` flag lost `conv-49:141`. BEAM top-5 digit and overlap stumps lost questions. Those attempts are in `C2_STEP2_LOG.md`. They were not committed.

Gate, `tools/c2_bench.py --diff` against the Step 1 dump above. LME 470 losses=0 gains=0. BEAM 355 losses=0 gains=0. LoCoMo 1536 losses=0 gains=4, all turn @1: `conv-26:103` (cat 4) `D6:9` → `D6:10`; `conv-30:57` (cat 4) `D7:4` → `D7:5`; `conv-48:41` (cat 1) `D11:6` → `D28:18`; `conv-49:131` (cat 4) `D18:7` → `D18:8`. Turn @5/@10 and every session metric stayed put. LME session had 0 drops.

LoCoMo turn any@1 812/1536 → 816/1536 (52.86% → 53.12%). Turn @5/@10 stay 1130/1236. Session @1/@5 stay 1137/1422. LME and BEAM scoreboards and funnels stay on the Step 1 numbers. LoCoMo funnel stages above the seat are unchanged (pool@50 1424, pre_rank@5 1039, post_kemeny@5 1100). @1 is 816/1536.

`cargo test -p vestige-core`: 586 passed, 0 failed. `cargo clippy -p vestige-core --all-targets -- -D warnings`: exit 0. `cargo check -p vestige-mcp`: exit 0. `tools/c2_hashcheck.py --n 500`: `hash-check 0/500`.

**Gate Verdict**: **WIN**. One strict uplift (LoCoMo Turn@1), zero per-question losses on LME_S, LoCoMo, and BEAM-100K.

