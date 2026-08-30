# Vestige Upgrade Plan — August 30, 2026

Synthesized from seven parallel research lanes. Ranked by user-visible benefit ÷ migration cost. Every claim traced to a primary source or to verified repo source. Where lanes contradicted each other, the contradiction is stated, not smoothed.

---

## The headline

**Three of the seven lanes independently concluded "measure on Vestige's own corpus before locking this in."** Reranker depth, RRF `k`, HyDE gating, temporal-contiguity weight, and any int8 model swap are all blocked on the same missing thing: a retrieval eval harness. CauseBench is retracted (CHANGELOG.md:170-181) and LongMemEval v1 was invalidated. **The harness is the true item zero** — without it, four Tier-1 tunables become guesses, and you have no way to prove any of this worked.

**Two corrections to the stack description, both verified in source:**

1. Vestige does **not** use fixed 0.3/0.7 weighted fusion. `crates/vestige-core/src/storage/sqlite.rs:6042` reads `let _ = (keyword_weight, semantic_weight);` and line 6044 calls `reciprocal_rank_fusion(&keyword_results, &semantic_results, 60.0)`. The 0.3/0.7 constants are dead outside benches. Any advice to "move to RRF" is advice to redo finished work.
2. Vestige does **not** advertise MCP 2026-07-28. `crates/vestige-mcp/src/protocol/types.rs:9` is `2025-11-25`; `server.rs:68` lists four supported revisions, none of them 2026-07-28. It answers `server/discover` as a compat probe with a doc comment (server.rs:315-328) explaining the deliberate exclusion. That is correct and honest — do not "fix" it.

---

## POST-2.5.0 — the next phase (agreed Aug 30 2026)

v2.5.0 shipped Tier 1 items 0, 1 and 2 (eval harness, reranker depth, #181). Four
items were explicitly deferred to the release after it. Items 1 to 3 were agreed
in conversation; item 4 was measured afterwards and is the best-evidenced of the
four.

### 1. Granite-embedding-311m-multilingual-r2 as an opt-in profile

Full write-up in **TIER 2 #2** below, including the zero-downtime staging plan on
PR #168's reversible profiles. Nothing to restate here.

### 2. RRF-native fusion

Replace the remaining multiplicative score blending with rank-based fusion, on the
principle that independent signals should join as ranked lists and never as
multipliers: a multiplier lets one signal's scale silently dominate, while a rank
is scale-free by construction.

Validate with **team-draft interleaving**, which is 10-100x more sensitive than
split A/B for ranker comparison because both rankers serve the same query and only
the click attribution differs.

Note the constraint already recorded in the headline above: core hybrid search is
ALREADY RRF (`sqlite.rs:6044`, `k=60`), and the 0.3/0.7 weights are dead outside
benches. So this item is about the stages layered on top of that fusion, not about
the fusion itself. Scope it precisely before starting, or it becomes a rewrite of
finished work. **TIER 2 #6** (make RRF `k` configurable and measure) is the
adjacent, smaller piece.

### 3. Staleness prediction via backcalculation

Brookmeyer & Gail 1986, the epidemiological method for inferring current unobserved
prevalence from an observed lag distribution. Applied here: infer which memories
have most likely gone stale from the distribution of observed staleness discoveries,
rather than waiting to be told a memory is wrong.

Speculative and unmeasured. It should be gated behind the eval harness like every
other ranking change, and it is the weakest-evidenced of these four. Sequence it
last unless it acquires supporting measurement.

### 4. The accessibility state machine runs 2 of its 4 states

**Measured Aug 30 2026, read-only against the live 2,930-memory store.** This
supersedes an earlier, wrong claim in conversation that the stage was inert and
changed no ordering; that came from a 50-result sample, not the corpus.

| bucket | multiplier | count | share |
|---|---|---|---|
| Active | x1.00 | 873 | 29.8% |
| Dormant | x0.70 | 2,057 | 70.2% |
| Silent | x0.30 | **0** | 0% |
| Unavailable | x0.05 | **0** | 0% |

`retention_strength` spans **0.6129 to 1.0**, mean 0.7005. State is derived from it
alone (`search_unified.rs:700-708`) at thresholds 0.7 / 0.4 / 0.1.

Two consequences, and they point in opposite directions:

- The stage is **not** inert. A binary x1.00 / x0.70 across a 70/30 split does
  reorder results: a Dormant hit at score 1.0 falls to 0.7 and loses to an Active
  hit at 0.8. It discriminates.
- But **half the designed model is unreachable.** Silent and Unavailable cannot
  fire, because nothing in the store falls below 0.61. Memories reach **217 days**
  since last access (mean 37.9) and still do not decay into them. A four-state
  forgetting model is operating as a two-state one, and the two states carrying
  the actual forgetting semantics are decorative.

Proximate cause, in source: the FSRS review path hardcodes
`let new_retrieval_strength = 1.0;` (`sqlite.rs:4095`), so
`retention = 0.7 + min(storage/10, 1.0) * 0.3` is **floored at 0.7** for anything
reviewed. `apply_decay` pulls it down from there, but empirically never past 0.61.

**The work is a decision, not a retune.** Determine whether this is a defect (decay
too flat, or `retrieval_strength` wrongly pinned) or deliberate conservatism, then
either fix the decay curve or delete the two dead states and stop advertising a
four-state model. Do not blind-tune the constants.

Two hard constraints:

- This changes ranking for **every** memory of **every** user. It is gated on the
  eval harness (`benchmarks/memconflict/`) like every other ranking change, and it
  needs a before/after on the same seeds.
- It interacts with the v2.4.0 suppression fix. `suppress` and `demote` deliberately
  no longer stamp `last_accessed`, because `apply_decay` recomputes retention from
  `days_since(last_accessed)`. Any change to that recomputation must not silently
  re-break inhibition. See the v2.4.0 CHANGELOG entry before touching it.

**Recommended order:** 4, then 1, then 2, then 3. Item 4 is measured, self-contained,
and affects every query today. Item 1 is well-specified and independent of ranking.
Item 2 needs scoping before it can be estimated. Item 3 needs evidence before it
deserves a slot.

---

## TIER 1 — do next

### 0. Stand up the retrieval eval harness (medium, prerequisite for 4 items below)

Not an upgrade; the instrument that makes the rest of this plan executable. LongMemEval_S (~115K tokens, 40 sessions) as the sanity baseline, **MemConflict (arXiv 2605.20926) as the flagship**. MemConflict is where the entire field fails: MemOS 0.5539, Letta 0.4871, A-Mem 0.4452, Mem0 0.3612, Memobase 0.3553, LangMem 0.2822, and the best Conflict Recognition Score across all six is 0.2501. Vestige ships contradiction inspection as a first-class tool; none of the six does.

Run the no-memory and naive-BM25 controls **first** and publish them alongside — MemDelta (arXiv 2606.29914) shows memory systems routinely underperform controlled baselines, and after the CauseBench retraction a second unreproducible number is fatal. Preregister the protocol and ship the harness in-repo.

### 1. Fix reranker starvation — `overfetch_multiplier` 3→5, precise 1→3 (small, largest measured recall delta per line changed)

`crates/vestige-mcp/src/tools/search_unified.rs:463-467`. Verified in source:

```rust
let overfetch_multiplier = match retrieval_mode {
    "precise" => 1,    // No overfetch — return exactly what's asked
    "exhaustive" => 5, // Deep overfetch for maximum recall
    _ => 3,            // Balanced default
};
```

T2-RAGBench (arXiv 2604.01733, Apr 2 2026, 23,088 queries / 7,318 docs) reranker-depth ablation: **20 candidates → Recall@5 0.458; 50 → 0.826; 100 → 0.888.** There is a cliff below 50. At the default `limit` of 10, balanced mode feeds the cross-encoder 30 candidates and precise mode feeds it **10** — precise mode runs a cross-encoder over a pool small enough that it can only reorder, never rescue.

There is an internal inconsistency worth naming: `crates/vestige-core/src/search/reranker.rs:23` already declares `DEFAULT_RETRIEVAL_COUNT: usize = 50`. The reranker's own config says 50; the MCP tool path never gives it 50. The `.min(100)` cap at line 479 already bounds DB load.

Cost: latency only, +67% pairs scored in balanced mode. Precise mode is documented as "fast, token-efficient" — use 3x there, not 5x. Single benchmark, single domain (financial text-and-table), so validate on the harness from item 0 before locking the constant.

### 2. Fix #181 — stale per-process vector index via `PRAGMA data_version` watermark (small, closes an open bug)

Verified: the index is rebuilt from `embedding_profile_vectors` at startup (`sqlite.rs:1413`, `build_embedding_profile_index` at :1425) and is **never persisted** — `save()`/`load()`/`view()` in `search/vector.rs` have zero production callers. The index is pure derived state over SQLite, so there is no index file to reconcile, only rows past a watermark to `add()`. `usearch` 2.23.0 already supports incremental `add()` on a live index. `PRAGMA data_version` increments on a connection when another commits, so the staleness check is effectively free per query.

Today, concurrent MCP server processes are semantically blind to each other's writes — silent duplicates through the prediction-error gate, incomplete recall. The FTS5 leg reads SQLite directly and is unaffected, which is exactly why this failure is partial and hard to notice.

Risk: `semantic_search_raw` (sqlite.rs:6308) holds the reader lock and the index `Mutex` in a fixed order; a reload path inside it can deadlock against non-reentrant lock ordering. Decide explicitly on reload failure — degrade to keyword-only **and say so in the response**, never silently return a partial result.

### 3. Bump fastembed 5.11 → 6.0.2, candle 0.10.2 → 0.11.0 (small, enabler for everything downstream)

`crates/vestige-core/Cargo.toml:143-146`, verified. No re-embed. This is the gate on every subsequent embeddings/runtime lever, because 5.11 exposes none of `with_intra_threads`, `with_execution_providers`, `with_max_length`, `with_dimension_override`. Vestige currently passes exactly `.with_show_download_progress(true).with_cache_dir(cache_dir)` at `embeddings/local.rs:122` and `search/reranker.rs:131` — nothing else.

6.0.2 pins `ort = "=2.0.0-rc.13"` (ONNX Runtime 1.28, opset 24) vs the 1.24-era runtime behind 5.11.

Three things to handle:
- **Breaking:** `fastembed::Error` moved from an `anyhow::Error` alias to a `thiserror` enum (PR #278). Two call sites, both currently plain `match` on Ok/Err.
- Align Vestige's own `candle-core`/`candle-nn` pins to 0.11.0 in the same PR or you compile two copies of candle.
- Both Nomic builtin profiles pin `immutable_model_revision: "fastembed-catalog-5.11"`. That string is load-bearing — revise it deliberately, not incidentally.
- **Re-run `scripts/check-glibc-floor.sh`.** A fastembed major can pull a new `ort-sys`, which can move the symbol floor this branch exists to hold at GLIBC_2.34.
- Keep `tokenizers` at 0.22.2 — it exactly matches fastembed 6.0.2's own pin. Moving to 0.23.1 ahead of fastembed buys nothing.

**Known sharp edge:** rc.13 has a teardown SIGSEGV under `load-dynamic` when the loaded libonnxruntime is older than 1.23 (pykeio/ort#614; exit 139 on ORT 1.20/1.22, exit 0 on 1.23.2). The fix (ort#610) merged 2026-08-23, **after** rc.13 shipped — it lands in rc.14. Vestige ships this exact configuration as its `ort-dynamic` feature. Until rc.14, document a minimum system ONNX Runtime of 1.23.

### 4. Unpin usearch =2.23.0 → 2.26.1 (medium — **two lanes contradict each other here**)

`Cargo.toml:170`, with an in-file comment reading "unum-cloud/usearch#746. Unpin when the upstream fix lands." **Issue #746 closed 2026-04-18**, fixed in v2.25.0; four releases have shipped since without reopening. The comment's own condition is met. 2.26.1 additionally carries "Fixed Windows CRT selection via Cargo," directly relevant to the Windows MSVC release job, plus hash-lookup tombstone reclamation under churn (which matters — Vestige removes/re-adds vectors on suppress and purge).

**Unresolved contradiction, do not skip:** the rust-runtime lane says preserve `features = ["fp16lib"]` through the bump (citing the Cargo.toml:160-163 comment recording PR #94 / issue #93, MSVC fatal C1021 from a `#warning` branch). The vector-index lane read usearch's **main-branch** Cargo.toml and reports `[features]` is now `numkong / openmp / simsimd` with `default = ["numkong"]` and **no `fp16lib` at all** — meaning the current feature line would fail to resolve.

Resolve empirically before writing any code: pull the manifest for **2.26.1 specifically** (not main) and read it. If `fp16lib` is gone, both original constraints must be re-derived against NumKong v7 — (a) issue #71, does the config still avoid AVX2/FMA dispatch on pre-Haswell x86_64, and (b) issue #93, does Windows MSVC stay clean. Test matrix must include Windows MSVC, which `ci.yml` still does not cover (only `release.yml` compiles it). Also verify the on-disk index format is unchanged across 2.23→2.26, or every user's index silently invalidates.

When this lands, **rewrite** the Cargo.toml:150-170 comment block for NumKong v7 rather than deleting it. It encodes two real production incidents; delete it and the next dependency cleanup re-breaks Windows exactly as before.

### 5. Document and guard the x86-64-v3 CPU baseline (small, latent SIGILL affecting real users today)

This is not on anyone's radar and it should be. ort's own docs: *"All x86-64 binaries are compiled with a baseline requirement of x86-64-v3"* — Intel Haswell 2013+, Gracemont, AMD Excavator or any Ryzen — and *"Linux binaries are compiled with Clang, not GCC, and depend on libc++."* Pre-2013 Intel and low-end Pentium/Celeron users get an **illegal instruction at first inference**, not a clean startup link error. That presents as a mystery crash report, not as the tidy glibc failure that got diagnosed on this branch.

Cheap now: a documented CPU floor in README + release notes, and ideally a startup capability check that fails loudly with an actionable message. The structural fix is Tier 2 #1.

### 6. Add `outputSchema` to the 14 advertised tools (small, real spec gap at the *current* revision)

Vestige emits `structuredContent` (server.rs:644, 659, 1461, 1474) while declaring no `outputSchema` in `tools/list` (handle_tools_list, server.rs:367+). Spec: *"Clients SHOULD validate structured results against this schema."* No client can validate, no client gets type info, the LLM gets no parsing guidance. This is a gap at 2025-11-25, not a 2026-07-28 requirement.

Sharp edge: once `outputSchema` is declared, servers **MUST** return conforming structured results. The error paths at server.rs:659/1474 also set `structured_content`, so the schema must admit the error shape. Start with `recall` and `smart_ingest`, not all 14 at once.

### 7. Run the official MCP conformance suite (small, converts a claim into a number)

`npx @modelcontextprotocol/conformance server --url http://localhost:PORT/mcp --requirements 2025-11-25`. Same suite that scored the Rust SDK 67/67 server / 50/50 client for its Tier 1 promotion on 2026-08-21. Read-only external client, breaks nothing. Run against the Streamable HTTP listener (`protocol/http.rs`, POST /mcp); dispatch is shared with stdio so findings apply to both. Then run it again with `--requirements 2026-07-28` to get an itemized gap list, which replaces guesswork about Tier 3 #1's scope.

### 8. Fix the `metal` feature — it accelerates nothing Vestige ships (small, honesty fix)

`Cargo.toml:87` declares `metal = ["fastembed/metal"]`. fastembed's `metal` resolves to `["qwen3", "nomic-v2-moe", "candle-core/metal", "candle-nn/metal"]` — **no `ort/*` entry at all**. Vestige's production embedder is `EmbeddingModel::NomicEmbedTextV15` on the ONNX path and its reranker is `RerankerModel::JINARerankerV1TurboEn`, also ONNX. Building with `--features metal` today pulls in candle + Metal kernels, inflates build time and binary size, and delivers zero acceleration to the two models that actually run.

Either delete it or rename/document it as applying only to the optional qwen3 / nomic-v2-moe candle embedders. Call it out in CHANGELOG.md — it is a visible cargo feature change. The real Apple Silicon lever on the ONNX path is `with_intra_threads`, unlocked by item 3.

---

## TIER 1 sequencing

```
0. Eval harness (MemConflict + LongMemEval_S + controls)   ── unblocks 1, and Tier 2 measurement
   │
   ├─ 1. overfetch_multiplier          search_unified.rs      ── measure on harness, then land
   ├─ 2. #181 watermark                sqlite.rs, vector.rs   ── independent
   ├─ 6. outputSchema                  server.rs              ── independent
   ├─ 7. conformance run               (external, no code)    ── run today, no dependencies
   └─ 5. CPU baseline doc              README, release notes  ── independent

3. fastembed 6.0.2 + candle 0.11 ──┬── 8. metal feature fix   (same Cargo.toml block)
   (re-run check-glibc-floor.sh)   └── 4. usearch 2.26.1      (same Cargo.toml block)
                                        ↑ resolve fp16lib contradiction FIRST
```

**File conflicts to plan around:**
- Items 3, 4, 8 all edit `crates/vestige-core/Cargo.toml` lines 87 and 139-170. Land them as **one dependency PR**, not three — and per the project's own "don't stack is competition-only" rule, batching product fixes into one mega-PR with the suite green is the correct pattern here.
- Items 1 and 2 touch different crates (`vestige-mcp/tools/search_unified.rs` vs `vestige-core/storage/sqlite.rs`) but both change retrieval behavior. Land **2 before 1** — fixing the stale index changes what candidates exist before you tune how many get reranked, and doing it the other way round confounds the depth measurement.
- Item 7 has zero code dependencies. Run it today, in parallel with everything.
- Items 3+4 must precede any Tier 2 embedding or runtime work.

---

## TIER 2 — worth doing, needs a migration plan

### 1. Replace pyke prebuilts with Microsoft's own ONNX Runtime tarball (medium — structurally fixes the bug this branch exists for)

**Root cause of the GLIBC_2.38 break, primary source:** every Linux row in `pykeio/ort-artifacts/.github/workflows/build.yml` is `runs-on: ubuntu-24.04`. Confirmed by the maintainer in pykeio/ort#523 (opened 2026-01-29, closed the next day): *"I believe I had to switch to the Ubuntu 24.04 runner because ONNX Runtime started requiring it… I'll look into downgrading to 22.04 for future builds, but that's only glibc 2.35."*

Microsoft builds its own Linux artifacts in a **manylinux_2_28** container — glibc 2.28 floor, covering RHEL/Alma/Rocky 8, Amazon Linux 2023, Debian 10+, every Ubuntu from 18.04. Upstream's best offer is 2.35. Switching to Microsoft's artifact is strictly better than waiting, and it kills the x86-64-v3 SIGILL from Tier 1 #5 at the same time. ONNX Runtime 1.29.0 (2026-08-12): Linux x64 11.1 MB, aarch64 10.0 MB, osx-arm64 41.6 MB.

Cost: Vestige stops shipping one self-contained static binary on Linux. `release.yml` must fetch and unpack per target. **`ORT_DYLIB_PATH` resolves relative to the executable**, and cargo examples/tests build into `target/<profile>/examples` and `.../deps` — a classic way to break CI without breaking the release. Prefer `ort::init_from(dylib_path)` at startup over the env var. You also trade pyke's build attestation for Microsoft's checksums.

Verify before committing (I did not download the tarball):
```
objdump -T libonnxruntime.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1
```
Do **not** use the `alfatraining/ort-artifacts` fork — stale since 2026-01-26, still building ORT 1.23.2, 4 stars.

### 2. Granite-embedding-311m-multilingual-r2 as a new opt-in profile (medium, forces re-embed — staging plan below)

`ibm-granite/granite-embedding-311m-multilingual-r2`, arXiv 2605.13521, May 14 2026. Apache 2.0. **32,768-token context (4× nomic's 8,192).** 768d native with official Matryoshka to exactly 256d. Official ONNX + OpenVINO weights. No query/document prefixes.

Paper Table 5: **65.2 MMTEB Multilingual Retrieval vs EmbeddingGemma-300m's 62.5** in the same table. Table 8 gives explicit 256d truncation results: 64.7 multilingual (−0.5 vs full 768d), 51.6 English, 63.4 code — so Vestige's existing 256d storage decision costs almost nothing on this model.

**Honest gap:** I could not find a same-source benchmark comparing Granite R2 directly to nomic-embed-text-v1.5. Vestige's prior research recorded nomic at MTEB v1 62.28@768d / 61.04@256d, which is **not comparable** to MMTEB Retrieval numbers. The defensible public claim is the head-to-head win over EmbeddingGemma plus 4× context at identical licence — not a numeric delta against nomic.

Integration cost: Granite is **not** in fastembed's catalogue (I enumerated all 46 `EmbeddingModel` variants in 6.0.2; absent). Needs `TextEmbedding::try_new_from_user_defined` with `UserDefinedEmbeddingModel` + `InitOptionsUserDefined`. Architecture is ModernBERT (alternating attention, GeGLU, RoPE, 262,152-token Gemma-3 vocab) — a real ONNX-export risk in principle, **de-risked twice**: IBM ships official ONNX itself, and fastembed already carries `ModernBertEmbedLarge`, proving the ModernBERT-through-ort path works. Download grows from ~90MB to a 311M-param ONNX; use the quantized build.

### How PR #168's reversible profiles stage this with zero downtime

`crates/vestige-core/src/embedding/profile.rs` already contains everything a model swap needs. **Do not build new migration machinery.**

1. **Add, don't replace.** Register Granite as an *additional* builtin `EmbeddingProfileId`. `NomicRetrieval256` stays the default. No existing user is disturbed; no upgrade re-embeds anything.
2. **Pin the artifacts.** The registry *rejects* a profile whose `verified_model_artifact_hashes` is empty, so pin the ONNX hashes and `immutable_model_revision` exactly the way `QWEN_06B_REVISION` / `qwen_06b_artifacts` already do. This is the mechanism that makes the migration reversible rather than lossy.
3. **Set `EncodingTemplate::Raw`.** Granite needs no prefixes, unlike nomic's `search_query:`/`search_document:`. Because templates are per-profile, prefix conventions cannot drift invisibly at call sites.
4. **Opt in per user, resume via cursor.** `EmbeddingProfileMigration` already carries `source_profile_id`, `destination_profile_id`, and `last_memory_id` — a resumable re-embed cursor. Drive the corpus through it in the background.
5. **Scope search to one `profile_id` until the cursor completes.** `EmbeddingProfileId` is the foreign key on stored vectors, so Nomic and Granite vectors never mix in a single ANN query. During migration, semantic search stays on the source profile; it flips atomically when `last_memory_id` reaches the end.
6. **Rollback is free.** The source profile's vectors were never deleted. Reverting is a default-profile flip, not a restore.

Same machinery covers **any** future re-embed, including the int8 nomic swap in Tier 3 — which is exactly why that must never be a silent config flip.

### 3. MemSecBench — publish a Selective Repair Success Rate (medium, security differentiator on existing machinery)

arXiv 2607.27080, Jul 29 2026, Zhejiang UT. 24 configurations, 310 cases from 48 contexts: **84.2% memory-poisoning success, 50.3% end-to-end attack success, 56.1% selective repair success.** The decisive number is a **30.2-point gap** between removing the poisoned memory (86.3%) and repairing without collateral damage to benign memories (56.1%). Current systems cannot delete one bad memory without destroying good ones.

Vestige's purge (content removed, embeddings removed, content-free tombstone retained) plus suppress is precisely the primitive that gap describes, and the paper concludes no current backend has it.

**Run this AFTER Tier 1 #2 and #4.** Running it honestly will expose whether purge actually removes embeddings from the in-process HNSW index rather than only from SQLite. HNSW deletes are soft; usearch 2.23.0 tombstones rather than compacts (2.26.0 adds hash-lookup tombstone reclamation, which is why the version bump comes first). If purge leaves a recoverable vector, the benchmark will find it — fix that before publishing, not after.

### 4. Async write path (medium)

"Async writes by default" is production requirement #1 in mem0's State of AI Agent Memory 2026 (Aug 29 2026). MemOS 2.0 "Stardust" (Aug 17 2026, Apache 2.0) ships MemScheduler; Letta's sleep-time agents fire every N steps (default 5). Vestige's FSRS-6 scheduling machinery means the scheduler concept is half-built; what's missing is decoupling the write from the agent turn.

Breaks read-after-write consistency — any test or hook that ingests then immediately recalls goes flaky. Needs an explicit flush/await path for tests and the save-guard hook, plus a durable queue so a crash between accept and index does not silently lose a memory.

**Blocked on #180.** See the open-bugs section.

### 5. Adaptive IDF fusion weighting (medium)

vstash (arXiv 2604.15484, Apr 20 2026) is a near-exact architectural twin of Vestige — local-first, single SQLite file, FTS5 + vector, RRF, small local embedding model. Adaptive per-query weighting from mean IDF of stemmed query terms, read from SQLite's `fts5vocab` virtual table. No LLM, no model, no network.

NDCG@10 across 5 BEIR datasets, zero regressions: SciFact +0.1%, NFCorpus +1.8%, SciDocs +1.7%, FiQA +3.4%, ArguAna +21.4%. **The paper itself states the ArguAna number comes primarily from a companion distance-cutoff change, not IDF weighting.** The defensible IDF-only band is +1.7% to +3.4%.

Requires moving from rank-based RRF to weighted RRF or a normalized convex combination. Interacts with the existing `keyword_bypass_threshold` (search_unified.rs:581) — both are lexical-confidence mechanisms; do not double-count.

### 6. Make RRF `k` configurable and measure (small, but do NOT blind-swap)

`HybridSearchConfig` already carries an `rrf_k` field defaulting to 60.0 that the live sqlite.rs:6044 path ignores. Wiring it through is the change.

Three sources say 60 is not optimal: T2-RAGBench (k=60 → Recall@5 0.695; k=10 → 0.716), OpenSearch's 6-dataset BEIR benchmark (RRF 3.86% *lower* NDCG@10 than tuned score-normalization fusion), and Qdrant's guidance that best `k` tracks relevant-docs-per-query (k=2–5 when ~1 relevant doc). Memory recall typically has few relevant memories per query, arguing for low `k`.

**Counter-evidence, reported honestly:** Bruch et al. (arXiv 2210.11934) find NDCG swings wildly with RRF parameters and that tuned parametric RRF does not generalize out-of-domain. Hence "configurable and measure," not "set k=10."

### 7. Gate HyDE behind measurement (small)

T2-RAGBench Table I: HyDE was the **worst of ten methods** — Recall@5 0.544 vs vanilla dense 0.587 vs BM25 0.644 vs hybrid RRF 0.695. Explicit recommendation: *"Avoid HyDE for domains with precise numerical or entity-centric queries."* Vestige's protected exact-lookup semantics are entity-centric by definition.

**Scope limit, stated honestly:** Vestige's HyDE (sqlite.rs:6321-6333) is *template-based query-variant expansion averaged into a centroid*, not LLM-generated pseudo-documents. The paper's causal mechanism — an LLM hallucinating wrong figures and dragging the query embedding off-target — does not directly apply. This is strong suggestive evidence to A/B, not proof to delete. The literal/concrete bypass at search_unified.rs:94 already protects the highest-risk cases. Keep `centroid_embedding()` and `expand_query()` exported; `benches/search_bench.rs` uses them.

### 8. Multi-signal ranking: recency + trust + graph proximity (small, with one hard constraint)

Both independent 2026 surveys (Vectorize Mar 14, bymar Apr 16) list multi-strategy search as an expected capability. Reference implementation on the identical substrate: engram-mcp scores `0.6 × cosine + 0.2 × recency + 0.2 × importance` over SQLite FTS5 + ONNX. Vestige already computes FSRS-6 retention and holds a graph — the signals exist and are simply not in the ranker.

**Hard constraint:** recency weighting actively harms exact-lookup (env vars, UUIDs, paths, quoted strings) — a behavior CLAUDE.md explicitly protects. The new signals must be zero-weighted on the exact-lookup path. Gate behind a config flag and A/B it.

### 9. Temporal contiguity / context reinstatement (small, genuine gap, unmeasured benefit)

Verified gap: zero matches for `contiguity` or `temporal_neighbor` across all crates. `neuroscience/context_memory.rs` implements Tulving & Thomson (1973) encoding specificity as static context *matching*, which cannot surface a memory temporally adjacent to a hit but semantically dissimilar.

Mechanism: Howard & Kahana (2002) Temporal Context Model, confirmed to operate in modern LLMs at ICML 2026 (Pink et al., poster Jul 8 2026) — localized to a one-dimensional temporal code reinstated by a single attention head, whose ablation significantly degraded temporal-order performance. Same class of win as `retroactive_backfill.rs`: surfaces episodically related memories that pure semantic similarity structurally cannot reach.

Reads `created_at` timestamps that already exist, adds a re-ranking boost. No schema change, no re-embed, pure Rust, no new dependency. **The ICML paper measures LLM internals, not Vestige-shaped retrieval** — the gain size is unmeasured. Default the weight low, validate on the item-0 harness before default-on.

### 10. Recency-weighted FSRS-6 optimizer loss (small, small measured benefit)

`crates/vestige-core/src/fsrs/optimizer.rs` trains 21 weights with no sample weighting (grep for recency/sample_weight returns nothing). srs-benchmark (9,999 collections, 349.9M reviews, table re-run 2026-08-29): FSRS-rs (FSRS-6 + recency weighting) LogLoss **0.3443** / RMSE(bins) **0.0635** / AUC **0.7074** vs plain FSRS-6 at 0.3460 / 0.0653 / 0.7034. Same 21 parameters, same model — the gain is purely the training loss weighting. The effect replicates independently within FSRS-7 (0.3370 recency vs 0.3401 plain).

Zero stored-state change; `ReviewLog` already carries `elapsed_days` as f64. Users who have already optimized get slightly different weights on their next run. Do this when you next touch the optimizer.

### 11. Qwen3-Reranker-0.6B as an opt-in high-accuracy tier (medium, NOT a default swap)

BEIR nDCG@10 **56.94 vs 49.60** for jina-reranker-v1-turbo-en, roughly +7.3. **Comparability caveat:** 56.94 is from Jina's July 2026 v3.5 table, 49.60 is Jina's own 17-dataset figure on the v1-turbo model card — different protocols. Treat the delta as directionally large, not exact. Apache 2.0, commercially clean.

Not in fastembed's `RerankerModel` enum (verified in the vendored source: exactly `BGERerankerBase`, `BGERerankerV2M3`, `JINARerankerV1TurboEn`, `JINARerankerV2BaseMultiligual`), so it needs `TextRerank::try_new_from_user_defined`. Harder: Qwen3-Reranker is a **decoder** with left-padding that scores via a yes/no token logit, not a sequence-classification head — fastembed's logit extraction needs a shim. fp32/fp16 ONNX are deliberately absent upstream (single-file size limit), so you are committed to int8 or q4. Download grows ~150MB → 400MB–1.2GB.

**Latency is the blocker to default adoption.** ~380ms/query CPU reported (weak source) vs ~85ms on a T4; mxbai's comparable 0.5B needs 0.67s on an A100. Benchmark on the M1 Max at 50 candidates before this goes anywhere near the default path. **No re-embed** — rerankers score (query, document-text) pairs at query time and never touch stored vectors. A user with 10,000 memories migrates for free.

---

## TIER 3 — interesting, speculative, or blocked

| Item | Status |
|---|---|
| **MCP 2026-07-28 implementation** | Large. Every result needs `resultType`; sessions/`Mcp-Session-Id` become dead (SEP-2567); `handle_initialize` bypassed (SEP-2575); per-request `_meta` parsing is new code. Vestige has **zero exposure** to every deprecation (no roots/sampling/logging, already on Streamable HTTP not HTTP+SSE, already uses -32602, already deterministic tools/list). Nothing urgent. Run Tier 1 #7 with `--requirements 2026-07-28` first to get the exact scenario list. |
| **rmcp 3.1.4 adoption** | Large, and the strategic *alternative* to the above, not an addition. Rust SDK promoted to Tier 1 on 2026-08-21 (67/67 server, 50/50 client, same-day spec tracking). Would outsource all future spec churn. But rmcp owns the server loop, transport, and dispatch — Vestige's `match method.as_str()`, its 22 hidden deprecated-tool redirects, `anthropic/maxResultSizeChars` handling, `vestige/` custom methods, and `protocol/auth.rs` all need re-homing, into a project that currently has **zero** MCP dependencies. Prototype behind a feature flag and diff conformance scores before committing. |
| **FSRS-7** | **Blocked on ecosystem.** Measurably better (LogLoss 0.3370 vs 0.3460, AUC 0.7220 vs 0.7034) but **no shipped Rust implementation** — crates.io `fsrs` max is 6.6.2, and the benchmark itself labels "FSRS-rs" as the Rust port of FSRS-6. Work is in unmerged fsrs-rs PRs #395/#426, last activity 2026-07-06. Anki 26.09b1 still ships FSRS-6. **Do not hand-roll it**: FSRS-7 redefines stability (S is no longer the interval at R=90%), and its 8-parameter dual-trace power-law mixture has no closed-form inverse, making interval computation a root-find. That is a from-scratch reimplementation plus migration of every stored stability value for ~2.6% LogLoss. Watch PR #426. **Note when it lands:** FSRS-7 fits Vestige *better* than Anki — its headline change is fractional intervals and realistic same-day predictions, and Vestige's "reviews" are accesses at arbitrary sub-day times, the exact regime FSRS-6 handles worst. `ReviewLog` already stores `elapsed_days` as f64. |
| **MemReranker-0.6B** | **Blocked on licence.** The single most domain-matched model found: distilled specifically for agent-memory retrieval (temporal constraints, causal reasoning, coreference), LoCoMo MAP 0.7150 / nDCG@10 0.7540 — beating BGE-reranker-v2-m3 by 4.4pp MAP **and** the 7× larger Qwen3-Reranker-4B. ~200ms. **The weights licence is unverified** — MemOS repo is Apache 2.0, the paper is silent, and the HF model page returned HTTP 401. No ONNX export exists. Verify the licence field, then benchmark, then decide. LoCoMo is dialogue-memory, not code/technical memory, so gains may not transfer. |
| **int8 nomic (`NomicEmbedTextV15Q`)** | Already in fastembed's registry, no new dependency. M1 Max meets the hardware precondition (ARMv8.2 SDOT/UDOT, the exact instruction class ONNX Runtime's quantization doc names), and ORT has been adding Arm64 low-precision kernels through 2026. **But ONNX Runtime publishes no headline int8 CPU speedup and warns "it is not rare to get worse performance on old devices."** And this is a **silent migration event**: dimensionality is unchanged (768→256), so nothing errors and no schema check fires — the corpus just degrades as new memories land in a slightly different space. Must go through the PR #168 profile machinery, never a config flip. Measure recall@k, not just latency; a 30% latency win that costs recall is a bad trade for a product whose value proposition is retrieval quality. |
| **ort-tract / ort-candle** | Timeboxed spike only. `alternative-backend` sets `ort-sys/disable-linking` — nothing downloaded, nothing linked against libc++, no glibc floor, no x86-64-v3 baseline. Structurally deletes the entire packaging problem. **But** fastembed pins ort with `api-24` while both backends declare `ort-sys` at `api-17` — anything above that returns a null function pointer **at runtime, not compile time**. Both backends document only a partial API and "Only Tensor types are supported." ort-tract demands rust-version 1.91, raising MSRV. Budget a spike answering exactly two questions: do both models load, and do the vectors match fp32 ONNX within tolerance. |
| **arroy 0.8.0** | Only if the #181 watermark fix proves unworkable under the existing lock ordering. LMDB-backed, structurally multi-process (*"many processes may share the same data and atomically modify the vectors"*), eliminates the startup rebuild. But it is a random-projection forest, not HNSW — `connectivity`/`expansion_add`/`expansion_search` have no equivalent, `n_trees`/`search_k` need re-tuning from scratch, and it adds a second on-disk store, so purge/unlearning must stay honest across two backends. Do not lead with this. |
| **sqlite-vec** | Architecturally the best long-term answer (index inside the same SQLite file, so #181 cannot recur) but **not ready**: stable is 0.1.9, ANN indexing only exists in the 0.1.10 alpha cycle with `ivf` marked experimental and DiskANN still getting statement-cleanup fixes as of 0.1.10-alpha.4 (2026-05-18). Also adds a loadable-extension distribution story across seven release targets. Revisit at 0.1.10 stable. |
| **Consolidation / reflection layer** | The recommendation to be **most** skeptical of. Hindsight leads BEAM at every tier (73.4% @100K, 64.1% @10M vs Honcho 40.6%, RAG 24.9%) on the strength of synthesizing higher-order knowledge. But it requires an LLM in the write path — collides head-on with local-first — and it manufactures derived memories that can be wrong, compounding the contradiction problem MemConflict measures. **See RecMem (arXiv 2605.16045, ACL 2026 Findings) for the right shape if you ever do it:** recurrence-gated consolidation cuts memory-construction token cost by up to 87% while exceeding accuracy, and the recurrence detector is a clustering pass over embeddings Vestige already computes. Do this after the MemConflict and MemSecBench numbers land, never before. |
| **EmbeddingGemma-300m** | Deliberately **de-prioritized** despite being the obvious pick and a one-line change in fastembed 6. Two disqualifiers: **2,048-token context vs nomic's 8,192** — a 4× regression against profiles declaring `maximum_token_limit: 8192` with `ChunkingStrategy::WholeDocument`, silently truncating any memory over 2K tokens; and the model card licence is **`gemma`** (ai.google.dev/gemma/terms), **not Apache 2.0**. The HF launch blog says Apache 2.0 and is **wrong**; the model card is authoritative. Gemma terms carry a Prohibited Use Policy and redistribution obligations — a live issue for a self-contained binary with a paid tier. Keep as the fallback if Granite's custom-ONNX integration stalls. |

---

## Do NOT change — verified still current

**Architecture and design decisions that are correct today.** This list is as load-bearing as the upgrade list.

- **The two-stage architecture (hybrid recall → cross-encoder rerank).** This is the single highest-leverage thing Vestige already owns. "Beyond the Reranker" (arXiv 2606.28367): removing the cross-encoder collapses retrieval outright and costs **30 points of answer F1**, an effect no other pipeline enhancement approaches. T2-RAGBench: Hybrid+Rerank 0.816 Recall@5 vs hybrid RRF alone 0.695, BM25 0.644, dense 0.587. Fix the depth; do not redesign the stage.
- **RRF over linear fusion in the live path.** Already done and correctly reasoned (comment at sqlite.rs:6038-6041). The old linear path used max-only normalization (`score / max_keyword`, hybrid.rs:78-88) — the weakest normalization choice and the likely cause of the paraphrase-burying it replaced.
- **256d Matryoshka truncation.** Validated, not merely tolerated. Granite Table 8: 64.7@256d vs 65.2@768d (−0.5). EmbeddingGemma card: 68.37@256d vs 69.67@768d. Truncation is close to free on modern MRL-trained models. Keep the layer-norm → truncate → L2-normalize implementation.
- **The embedding-profile system itself.** Needs no redesign. `EmbeddingProfileId` as the vector foreign key, pinned `immutable_model_revision` + `verified_model_artifact_hashes` with registry enforcement, per-profile encoding templates, and `EmbeddingProfileMigration` with a `last_memory_id` cursor. Every model swap in this plan rides on it.
- **The existing Qwen3-Embedding profiles.** Apache 2.0, 32K context, MRL 32–1024d, 64.33 MTEB multilingual, correctly built with pinned revisions and hashes. Note for later: Qwen3 is decoder-only on the candle backend, not the ONNX path, and at least one practitioner benchmark has EmbeddingGemma beating Qwen3-0.6B at comparable size — so 0.6B is a fine opt-in, not an obvious default.
- **HNSW as the algorithm, usearch as the implementation.** At 256d with a low-thousands corpus, brute force is already sub-millisecond; no index change produces a user-visible win. Six releases Dec 2025–Aug 2026, still in-process, still dependency-light.
- **f32 vector storage — do not quantize.** i8 saves ~0.75MB per 1000 memories at 256d, noise on a 64GB machine, and costs measurable accuracy: MonaVec (arXiv 2606.19458) measures usearch-i8 at 0.928 vs 0.960 full-precision on AG News. Vestige is already lossy once via Matryoshka. Revisit above ~1M vectors.
- **ONNX Runtime as the runtime.** Checked against burn (training framework, 0.22 still pre-release), mistral.rs (crates.io stranded at 0.8.1 from April, and it's an LLM serving stack), llama-cpp-2 (GGUF-only, trades one C++ packaging problem for another), mlx-rs (last release 2025-12-16, self-described "unofficial"). Nothing is a drop-in for a 137M BERT encoder + 37M cross-encoder.
- **CoreML EP — do NOT enable.** This is the upgrade I expected to recommend and the evidence says no. Measured on M2 Max (k2-fsa/sherpa-onnx#2910, 2025-12-18, Transformer/Conformer): CoreML RTF 0.470 vs CPU 0.427 at 2 threads; 0.457 vs **0.372** at 10 threads. CoreML lost at both and lost by *more* as CPU threads scaled. ORT's only CoreML performance note is *"performance may be negatively impacted by inputs with dynamic shapes"* — exactly a text embedder's regime.
- **No LLM-built knowledge graph.** GraphRAG-Bench (arXiv 2506.02404v3, 1,018 questions, 7M-word corpus, nine SOTA systems): best GraphRAG (HippoRAG) 72.64 vs **plain BM25 71.66**; two methods score *below* the no-retrieval baseline. Construction cost 33M–84M LLM tokens, 1.4–5.7 hours. Paying 33M tokens for +0.98 points is a local-first non-starter. Vestige's LLM-free `spreading_activation.rs` (stage 6 of search_unified.rs:796-814) is the correct call.
- **No query decomposition.** arXiv 2606.05658: on MuSiQue, decomposition collapsed MRR from 0.469 to **0.102** and Success@5 from 1.0 to 0.063; even where it helped, latency went 21s → 48s. Corroborated by T2-RAGBench (multi-query 0.640 vs BM25 0.644, a wash).
- **Late chunking is a non-issue.** Vestige does not chunk memories — atomic, embedded whole to MAX_TEXT_LENGTH 8192 (`embeddings/local.rs:26,321`). And the late-chunking paper's own nomic column (arXiv 2409.04701v3, Table 2) shows SciFact 70.7 → 70.6 (a *regression*) and NFCorpus 35.3 → 35.3 (flat). The widely-quoted +6.5pp is the jina-v2-small column, not nomic.
- **No late-interaction / ColBERT rerank stage.** Late interaction's payoff is in *first-stage* retrieval with a multi-vector index — many vectors per memory, a storage-engine and usearch redesign that *would* force a re-embed, which none of the Tier 1/2 reranker items do. As a rerank stage over 50 candidates it gives up the query-document attention that is why a cross-encoder wins.
- **The BM25-like term-overlap fallback when the cross-encoder is unavailable.** Correct local-first graceful degradation. Keep regardless of model choice.
- **Do NOT move to the BGE reranker line.** `bge-reranker-v2-m3` is a one-line change in fastembed's enum and it is tempting, but it dates to **February 2024** — same generation as the model it would replace. 568M params for a result the MemReranker paper shows a 0.6B distilled model beats by 4.4pp MAP. BAAI has shipped nothing newer than bge-reranker-v2.5-gemma2-lightweight (July 2024). The line is stagnant. A 15× latency hit to land on a two-year-old model is the worst trade in the stack.
- **Eager LLM consolidation — actively do not add it.** The strongest evidence in the whole synthesis: "Useful Memories Become Faulty When Continuously Updated by LLMs" (arXiv 2605.12978, May 13 2026) shows memory utility rises then degrades *below* the no-memory baseline — GPT-5.4 failed **54%** of ARC-AGI problems it had previously solved without memory, after consolidating from verified solutions. Their prescription is to keep raw episodes as first-class evidence and gate consolidation explicitly, which is exactly what Vestige does with preserved raw memories + explicit `advanced/merge_supersede.rs`. External validation of an existing design decision.
- **Retrieval-induced forgetting, interference, encoding specificity, retroactive salience backfill.** All already shipped (`neuroscience/memory_states.rs` with RIF + competition_loss persisted in migrations.rs:405; `active_forgetting.rs` citing Anderson 1994; `context_memory.rs`; `advanced/retroactive_backfill.rs` as a faithful port of Zaki/Cai 2024 Nature 637:145-155 including the backward-only asymmetry). No 2025-2026 result supersedes any of them.
- **FSRS-6 itself.** `fsrs/algorithm.rs` implements the correct 21 weights and the correct power forgetting curve including personalizable decay w20. Matches what Anki 26.09b1 ships today.
- **SM-20 — non-starter.** Proprietary cloud API, early-access rate limits (100 reps/day free, 10,000 historical import cap), no Rust binding, not in srs-benchmark so no independent comparison. Cannot back a local-first core.
- **Neural schedulers (RWKV/LSTM/GRU) — do not adopt despite topping the benchmark.** RWKV-Instant leads (LogLoss 0.2773, AUC 0.8329) but uses 2.76M parameters, trains across 5,000 users rather than per-user, needs features Vestige doesn't have (answer duration, sibling cards, deck hierarchy), and is not evaluated with TimeSeriesSplit unlike FSRS — not like-for-like. GRU needs Reptile pre-training on 100 users' histories, which a single-user local system cannot obtain.
- **MCP version negotiation as written.** Do not add 2026-07-28 to `supported_protocol_versions()` before the work is actually done — that would be a false claim and would fail conformance.
- **SQLite as canonical store with FTS5 hybrid.** Convergent with where the field landed. MemOS 2.0's own local plugin (Aug 17 2026) is SQLite + hybrid FTS5 + vector — the architecture Vestige has run for a year. Nobody credible argues for a heavyweight vector DB in a local agent memory server anymore.
- **Local-first single Rust binary, MCP-native, purge-with-tombstone, exact-lookup preservation.** All still correct and increasingly rare. The 2026 field is Python + Docker + external DBs (cognee needs PostgreSQL/LanceDB/Kuzu; Zep retired self-hosted CE entirely). Rust competitors that exist are toys — palace-rs has 4 stars, engram-mcp 15. MemSecBench just turned purge semantics from a hygiene feature into a measured security capability the field lacks.
- **LanceDB, DiskANN — rejected.** lancedb 0.37.1 pulls ~184MB of Arrow 58 / DataFusion 54 / Tokio transitive deps. DiskANN exists to keep billion-scale indexes off RAM; Vestige's entire index fits in a few megabytes.

**One thing to write down in the repo so nobody burns a week on it in six months:** jina-reranker **v2, v3, and v3.5 are all CC BY-NC 4.0**. jina-reranker-v3.5 (0.6B, Jul 20 2026, BEIR 63.20, arXiv 2607.18152) is the best reranker in the world at its size and is **permanently unavailable to Vestige** while it has any commercial tier. v1-turbo-en is the last permissively-licensed Jina reranker. Someone will find that 63.20 number.

**One thing to react to correctly:** MemPalace hit 43,458 GitHub stars in **eight days** (OSS Insight, Apr 13 2026), later ~54k, positioned as local-first MCP memory. Vectorize's Apr 12 2026 teardown found its 96.6% LongMemEval measured raw ChromaDB over uncompressed text, its LoCoMo 100% used `top_k=50` against 19-32-session datasets (retrieving the whole corpus), its "30× lossless" compression is lossy and regresses 12.4 points, and its advertised contradiction detection and multi-hop traversal **do not exist in the code**. Do not react architecturally. Do note that a well-marketed local-first MCP memory server can take 40k stars in a week — that is a distribution lesson, not a technical one.

---

## Open bugs

### #181 — stale per-process vector index: **FIXED by Tier 1 #2**

Direct fix, small cost, no new dependency, using API already present in the pinned usearch 2.23.0. Detailed above. Tier 2 #1 (arroy) is the structural fallback if lock ordering defeats the watermark; do not reach for it first.

Secondary benefit: **usearch 2.26.0's hash-lookup tombstone reclamation under high churn** (Tier 1 #4) directly serves the suppress/purge path, which removes and re-adds vectors. That interaction is also what makes Tier 2 #3 (MemSecBench) honest — soft HNSW deletes are exactly what a poisoning benchmark probes for.

### #180 — apply_plan atomicity: **NOT fixed by anything in this plan**

Being explicit because it matters. No lane surfaced a dependency, model, or protocol upgrade that touches it, and none will. Verified in source: `SqliteStorage::apply_plan` (sqlite.rs:12198) performs a sequence of independent `self.*` calls — `get_plan`, `plan_status`, `get_node`, `read_bitemporal` per invalidated id, then mutations — with **no enclosing transaction**. A crash or error partway through the `PlanKind::Merge` arm leaves the survivor mutated and some absorbed nodes not, with a partial undo blob. The `plan_status` "already applied" guard (12210-12220) is a check-then-act with no lock, so it is also racy under the concurrency #181 exposes.

**Two hard sequencing consequences:**

1. **#181's fix makes #180 more reachable, not less.** Once concurrent processes actually see each other's writes, concurrent `apply_plan` calls on overlapping node sets become a live race rather than a theoretical one.
2. **Tier 2 #4 (async write path) must not land before #180 is fixed.** Moving writes off the request path multiplies the number of interleaved mutations against exactly this non-transactional code. Fix #180 first, or explicitly exclude merge/supersede from the async path.

Fixing #180 is cheap relative to everything above — wrap the whole `apply_plan` body in an `IMMEDIATE` transaction so the status check and the mutations commit or roll back together. It is not in Tier 1 only because no lane's research produced it; on benefit/cost it belongs alongside Tier 1 #2, and I would land the two in the same PR since they touch the same file and the same concurrency story.

---

## What I could not verify

Stated plainly so nothing here gets quoted as settled:

- **The usearch `fp16lib` feature contradiction** (Tier 1 #4). Two lanes read different manifests. Resolve against the 2.26.1 manifest specifically before writing code.
- **Granite R2 vs nomic-embed-text-v1.5 head-to-head.** No same-source benchmark exists. The MTEB v1 numbers Vestige has recorded for nomic are not comparable to MMTEB Retrieval.
- **MemReranker-0.6B weights licence.** HF page returned 401.
- **Microsoft's ONNX Runtime 1.29.0 Linux tarball glibc floor.** manylinux_2_28 is the documented build container; confirm with `objdump -T` on the extracted artifact.
- **usearch on-disk index format stability across 2.23 → 2.26.** A silent serialization change would invalidate every user's index.
- **ort-tract/ort-candle `api-17` vs fastembed's `api-24`.** Incompatibility surfaces as a null function pointer at runtime, not a compile error.
- **Microsoft Harrier-OSS-v1** (announced Mar 30 2026, MTEB Multilingual v2 SOTA claim) — single secondary source (MarkTechPost), licence, MRL support, and ONNX availability all unverified; its 640d output does not truncate cleanly to 256 without confirmed MRL. Found but not recommended.
- **The ONNX Runtime plugin-EP transition.** Separate release tags are now being cut (plugin-ep-webgpu/v0.3.0 Aug 24, plugin-ep-cuda/v0.1.0 Aug 17). No CoreML plugin exists today, but if one ships, the "one archive per EP combination" constraint that shapes ort's whole distribution model relaxes and the Tier 2 #1 packaging calculus changes. Re-check in a few months.