# Vestige Retrieval Engine (`crates/vestige-core/src/retrieval/`) — Agent Guide

Read this before creating, moving, or editing ANY file under `crates/vestige-core/src/retrieval/`.

---

## 1. Pipeline Directory Map (Put Files in the Right Stage)

The engine runs as a deterministic **7-stage pipeline + mirrored test suite**. Zero `.rs` files belong directly at `retrieval/` root besides `mod.rs`.

| Stage / Folder | Owns | Put a NEW file here when building... | Register & Wire In |
| :--- | :--- | :--- | :--- |
| **`query/`** *(Stage 1)* | `tokens`, `stemming`, `dates`, `intent`, `plan`, `wh_slots`, `coref`, `centering`, `focus`, `hobbs`, `rst` | Query parsing, normalization, pronoun/anaphora resolution, intent classification, WH-slot typing, temporal parsing | `query/mod.rs` -> call from `query/plan.rs` or `stage01..03` |
| **`index/`** *(Stage 2)* | `mod.rs` (`Vocab`/`Bag`/`Index`), `normalize`, `postings`, `rsb`, `policy`, `kite_typed` | Corpus/turn preprocessing, index structures, postings lists, alias/orthographic normalization, RSB index links | `index/mod.rs` -> call in `build_index` (`index/mod.rs`) |
| **`lexical/`** *(Stage 3)* | `bm25f`, `bm25l_dfr`, `bm25tp`, `dph`, `qld`, `wand`, `rocchio`, `rbp`, `rungs`, `steps`, `lgd_q16`, `ppm_surprisal`, `query_frame_ic`, `sinkhorn_ot_q16`, `speaker_translation_q16`, `translation_lm` | Exact-token/stem/proximity lexical scorers, IDF/burstiness weighting, query-likelihood or divergence models, classic IR expansion | `lexical/mod.rs` -> call from `rank/stage01..03` |
| **`graph/`** *(Stage 4)* | Causal, temporal & bridge structures (`bridge`, `causal`, `supersession`, `taxa`, `torrens`, `stn_floyd_warshall`, `bitemporal_heideltime`, `temporal_allen`, `ppr_rwr`, `viterbi`, `swanson_abc`, `actr`, `ac3`, etc.) | Cross-turn/cross-session graphs, temporal constraint intervals, causal chains, entity-bridge walks, supersession/contradiction DAGs | `graph/mod.rs` -> call from `rank/stage04`, `stage07`, `stage11`, or `stage12` |
| **`fusion/`** *(Stage 5)* | `lin_bilmes`, `kemeny`, `kemeny_mrf`, `psl_ilp`, `top10_rrf`, `aspect_coverage_q16`, `hypergraph_mdl_q16`, `matroid_intersection_q16`, `pit_commensuration_q16`, `scaldpp_q16`, `summary_silhouette_q16` | Multi-voter rank aggregation, submodular session/turn selection, matroid/ILP cover solvers, RRF/Kemeny consensus | `fusion/mod.rs` -> call from `rank/stage05`, `stage06`, or `stage12` |
| **`rank/`** *(Stage 6)* | Straight-line 13-stage `rank_one`: `context.rs`, `stage01`–`stage13`, `event_ordering/`, `stage04/`, `stage12/`, `implicit_bridge.rs`, `supersession_and_guard.rs`, `tool_receipt.rs` | Orchestrating new passes inside `rank_one`, adding shared per-query fields to `RankCtx`, or splitting large `stageXX_*.rs` helpers into `rank/stageXX/` | `rank/mod.rs` (or `rank/stageXX/mod.rs`) -> call cleanly from `stageXX_*.rs` |
| **`seating/`** *(Stage 7)* | Final turn seating & `@1` tie-breakers: `gatekeeper`, `at1_scorer_q16` + `at1/`, `c2/`, `w2_*`, `cues`, `primacy`, `opener_and_hearsay`, `florilegium`, `slot_promotions`, `preference_lattice`, `vault_closers`, `step11_wh_voter`, `step15_comparative`, etc. | Deep-pool `@1` features, Condorcet/C2/W2 seating rules, slot promotions, preference/revision closers, final turn selection | `seating/mod.rs` (`seating/at1/mod.rs` or `seating/c2/mod.rs`) -> wire in `stage08`, `stage09`, or `stage12` |
| **`tests/`** | 1-to-1 mirrored tests: `query_tests.rs`, `index_tests.rs`, `lexical_tests.rs`, `graph_tests.rs`, `fusion_tests.rs`, `rank_tests.rs`, `seating_tests.rs` (+ `tests/mod.rs` builders) | Unit tests for the corresponding stage module | `tests/mod.rs` -> add tests to matching `<stage>_tests.rs` (or subfolder) |

> **Static lexicons stay OUTSIDE `retrieval/`:** Compiled linguistic knowledge tables (`wordnet.rs`, `roget.rs`, `conceptnet.rs`, `wikidata.rs`, `geonames.rs`, `zero_overlap.rs`) live in `crates/vestige-core/src/` and are consumed by `retrieval/` as read-only lookups.

---

## 2. Hard Organization Rules (Zero Sprawl, Zero Bloat)

1. **NEVER pile new logic into existing mega-files.**  
   Do **NOT** append new subsystems, passes, or helper structs directly onto:
   - `seating/at1_scorer_q16.rs` -> put AT1 query/feature/gatekeeper helpers in **`seating/at1/`**
   - `rank/stage12_session_sync_and_fusion.rs` -> put Stage 12 helper passes in **`rank/stage12/`**
   - `rank/stage04_floors_and_bridge.rs` -> put Stage 4 bridge/lift passes in **`rank/stage04/`**
   - `tests/seating_tests.rs` -> add a clean submodule or split large test groups into `tests/<area>_tests.rs`
2. **Keep every new module small & single-purpose (`<= ~500` lines).**  
   If a `stageXX_*.rs` or domain module grows past ~700–1,000 lines, extract cohesive helper passes into a subdirectory (`<stage>/<feature_or_stage>/mod.rs` + focused submodules), matching `rank/event_ordering/`, `rank/stage04/`, `rank/stage12/`, `seating/at1/`, and `seating/c2/`.
3. **Avoid adding new glob re-exports (`use self::...::*;`).**  
   Older files are glob-imported at the `retrieval/` root, which makes collisions easy. Keep new modules encapsulated (`pub(in crate::retrieval) mod my_module;`) and import them explicitly by path (`crate::retrieval::graph::my_module::score_fn` or `super::stageXX::helper::run`).
4. **PROTECT `seating/at1_scorer_q16.rs` GBDT Markers & Path.**  
   `scripts/refit_at1_convs1_5.py` parses the exact file path `crates/vestige-core/src/retrieval/seating/at1_scorer_q16.rs` and asserts `// === BEGIN FROZEN GBDT WEIGHTS ... ===` / `// === END FROZEN GBDT WEIGHTS ===` plus `canonical_gbdt_bytes()`.
   - NEVER move `at1_scorer_q16.rs` or rename those markers.
   - NEVER hand-edit frozen GBDT weights (`eval_at1_gbdt_q16` weights are SHA-256 pinned in unit tests).
   - Put feature extraction, query specs, Condorcet ballots, and gatekeeper logic inside `seating/at1/`.
5. **No orphaned / dead files.**  
   Every `.rs` file under `retrieval/` MUST be declared in its `mod.rs` and invoked by an active pipeline stage (or `#[cfg(test)]`). Remove dead prototypes rather than leaving uncalled modules in `graph/` or `fusion/`.

---

## 3. Core Engine Invariants (Zero-Leak & Pure Deterministic Math)

1. **Zero-Leak Input Contract**: Keep `#[serde(deny_unknown_fields)]` on `TurnLink`, `Turn`, `Session`, and `Query` in `retrieval/mod.rs`. Benchmark labels (`category`, `question_type`, `answer`, `has_answer`) are rejected at parse time and must never enter retrieval scoring.
2. **Zero Vectors, Zero LLMs, Zero Network**: Every score is deterministic arithmetic over caller-supplied text and pinned local resources. Prefer integer **`Q16.16`** fixed-point (`65_536 = 1.0`, helpers in `fusion/kemeny_mrf.rs`) for new cross-platform deterministic scorers.
3. **Hardcode & Open-Class Guard (`scripts/hardcode_scan.sh`)**:
   - NEVER embed dataset/benchmark names in non-test Rust string literals.
   - NEVER write hand-typed open-class topic/entity word arrays.
   - Any closed-class grammar/function-word array with `>= 8` lowercase word literals in non-test code MUST either use a listed `ALLOWED_CONSTS` name or carry `// hardcode-scan: closed-class (<brief linguistic reason>)` on the line immediately above it.
   - NEVER modify pinned files under `crates/vestige-core/data/` (SHA-256 verified by `hardcode_scan.sh`).
4. **Preservation & Tie-Breaking**: When adding a late stage or tie-breaker, explicitly guard Position 0 / Rank-1 (`top0`) crisp lexical winners and state cleanly in module docs whether the pass affects `session` ranking, `turn` ranking, or both.

---

## 4. Checklist: Adding or Editing a Retrieval Module

1. **Pick the exact stage folder** from Section 1 (`query`, `index`, `lexical`, `graph`, `fusion`, `rank`, `seating`, or `rank/stageXX/`).
2. **Create a focused file** (`my_pass_q16.rs` or `<subdir>/mod.rs`) with a module `//!` docstring explaining inputs, invariants, and caller stage.
3. **Declare it** in that folder's `mod.rs` (`pub(in crate::retrieval) mod my_pass_q16;`).
4. **Wire it once** at the exact pipeline point in `rank/stageXX_*.rs` or `seating/`.
5. **Add unit tests** in the mirrored `retrieval/tests/<stage>_tests.rs` (using shared builders in `tests/mod.rs`), or a local `#[cfg(test)] mod tests` if testing private module internals.
6. **Verify before finishing** (run all 3 gates; never skip `hardcode_scan.sh` or `-D warnings`):
   ```sh
   bash scripts/hardcode_scan.sh
   cargo clippy -p vestige-core --all-targets -- -D warnings
   cargo test -p vestige-core --lib retrieval:: --no-fail-fast
   ```
   For full benchmark validation, run `bash scripts/run_master_ratchet.sh <step_name> core`.
