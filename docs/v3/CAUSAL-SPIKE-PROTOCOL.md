# v3.0 Causal Graph — Spike Protocol (PREREGISTERED)

Preregistered: 2026-08-31, before any spike run was executed.
Branch: `spike/causal-graph`. Harness: `benchmarks/causal-spike/`.

## Why this document exists

v3.0 is gated on this spike. The gate: **blind recovery of known
cause→failure pairs**. A prior causal benchmark in this project was formally
withdrawn after adversarial review reproduced indefensible headline claims
(see CHANGELOG, CauseBench retraction). This spike therefore preregisters its
dataset construction, systems under test, metrics, and pass thresholds BEFORE
the first run, and reports every number regardless of outcome. Nothing in this
protocol may be edited after the first `run` invocation except to append
results and errata.

## Claim boundary

The spike measures recovery of **planted candidate causes** under blinding. A
pass licenses the claim "the causal layer surfaces receipt-backed upstream
candidates that similarity search misses." It never licenses "automatic root
cause" — the epistemic boundary shipped in `BACKFILL_RECEIPT_CLAIM_BOUNDARY`
applies verbatim to everything downstream of this spike.

## Hypothesis

H1: Over stores with realistic distractor density, an entity-temporal causal
recovery layer blindly recovers planted cause→failure pairs at materially
higher recall than lexical/semantic similarity search, specifically on pairs
constructed so the true cause shares **no failure vocabulary** with the
failure event.

H2 (the v3.0 primitive): a **persistent causal graph** — accumulated
`backfill_candidate` evidence edges plus temporal-precedence traversal —
matches or beats per-query backfill while enabling multi-hop recovery
(cause-of-cause) that per-query backfill cannot express.

## Dataset construction (deterministic, seed=20260831)

- **Stores**: fresh temporary SQLite stores only, built by `causal-spike seed`.
  The harness MUST refuse to run against any pre-existing store path. (Origin:
  the Jul 15 data-integrity incident where a dry-run auto-fired on an audit
  note in a live corpus.)
- **N = 30 planted pairs** per store, 3 stores (90 pairs total). Each pair:
  - one *cause* memory: an innocuous decision/config change carrying 1–2
    identifier entities (env-var style, path style, version-pin style),
    ZERO failure vocabulary, backdated uniformly 7–45 days before its failure.
  - one *failure* memory: crash/error text that shares the identifier
    entity/entities with its cause but no other content words.
- **Distractors, per pair**:
  - 5 semantic decoys: failure-flavored text sharing vocabulary with the
    failure but NO entity link (these are what similarity search prefers).
  - 5 entity chatter: memories sharing the entity but ingested AFTER the
    failure or carrying no causal relation (tests temporal precedence).
  - 5 neutral background memories.
  So each failure sits in a field of ≥15 distractors; a store holds ~480
  memories.
- **Multi-hop subset**: 10 of the 90 pairs additionally plant a *root* cause
  one hop upstream (root → intermediate → failure, entity-chained) to probe
  H2's multi-hop recovery. Scored separately; not part of the H1 gate.
- **Blinding**: ground truth manifest (pair ids, roles) is written to a JSON
  file OUTSIDE the store and never ingested. `run` receives only the store
  and the list of failure ids. `score` is the only stage that opens the
  manifest.

## Systems under test

| Arm | System | Description |
|-----|--------|-------------|
| A | `lexical` | `hybrid_search` over the failure text, top-k (the similarity baseline; where the embedding model is unavailable this degrades to FTS hybrid and is reported as such) |
| B | `backfill` | current per-query Retroactive Salience Backfill (`RetroactiveBackfill::run` over backward candidates) |
| C | `causal-graph` | the v3.0 primitive: persist `backfill_candidate` edges for ALL failures first (graph accumulation pass), then answer each query by traversing the graph with temporal-precedence + edge-strength scoring; multi-hop allowed |

Arm C is the moonshot under test. Arms A and B are controls: A is the
industry default, B is what already ships.

## Metrics (all reported, no selection)

- recall@1, recall@3, MRR over the 90 planted pairs, per arm.
- **Separation rate**: fraction of pairs where C (or B) succeeds at k=3 AND A
  fails at k=3 — the moat number.
- Multi-hop recall@3 on the 10-root subset (C only; B structurally cannot).
- Wall-clock per query, per arm.

## Pass gate (locked before first run)

The spike PASSES and v3.0 scope locks iff, aggregated over the 3 stores:

1. Arm C recall@1 ≥ 0.60 AND recall@3 ≥ 0.80 on planted pairs;
2. Arm C separation rate vs Arm A ≥ 0.40;
3. Arm C ≥ Arm B on recall@3 (the graph must not lose to per-query backfill);
4. Multi-hop recall@3 ≥ 0.50 on the 10-root subset.

Any other outcome is a FAIL for the gate as posed. A fail does not end v3.0;
it forces a redesign of the recovery layer before scope locks, and the fail
numbers get published to this document unedited.

## Anti-p-hacking rules

- One seeded dataset generation. No regeneration after seeing results.
- Thresholds above may not move. If the harness has a bug, fix the bug, note
  the erratum here, rerun ALL arms.
- Every run appends its commit hash, store seeds, and full metrics JSON to
  `benchmarks/causal-spike/results/` (committed).
- The scored comparison always includes ALL arms — no dropping the baseline.

## Results

### Run 2026-09-01T04:51:09Z — seed=20260831 (official)

- HEAD at measurement: `7a2dc8da1308e6bc6e08a01447cb2a0d8c183a60` (protocol + crate scaffold). Harness sources used for this run were uncommitted on `spike/causal-graph` at measurement time.
- Stores: 3 × 30 planted pairs (90), 10 multi-hop. Dataset written under `benchmarks/causal-spike/data/` (gitignored).
- Arm A search_mode: `fts_keyword` (embedding hybrid compiled in vestige-core; this run's `SearchResult`s had no semantic scores).
- Metrics JSON: `benchmarks/causal-spike/results/*-7a2dc8d-20260901T045109Z.json`

| Arm | recall@1 | recall@3 | MRR | sep vs A | multi-hop r@3 | mean ms/query |
|-----|----------|----------|-----|----------|---------------|---------------|
| A lexical | 0.00 | 0.00 | 0.00 | — | — | 1.94 |
| B backfill | 1.00 | 1.00 | 1.00 | 1.00 | — | 3.75 |
| C causal-graph | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (10/10) | 0.005 |

Pass gate (locked): C r@1 ≥ 0.60 ✓, C r@3 ≥ 0.80 ✓, C sep vs A ≥ 0.40 ✓, C r@3 ≥ B r@3 ✓, C multi-hop r@3 ≥ 0.50 ✓.

**Outcome: PASS.** Licenses: receipt-backed upstream candidates that similarity search misses. Does not license automatic root cause.

Post-hoc (see erratum run 2026-09-01T05:21:40Z): Arm A candidate lists were empty for all 90 queries. The printed PASS is arithmetically true against a null baseline and does not lock v3.0 scope.

A prior smoke with seed=1 (not the preregistered dataset) is in `results/*-20260901T045017Z.json` and is not this gate.

### Run 2026-09-01T05:21:40Z — same frozen stores, harness errata (not a new seed)

- Same seed=20260831 stores. `dataset_id=8e199470e4807ecd`. No regeneration.
- HEAD stamp still `7a2dc8d` (harness still uncommitted). `embedding_ready=false` on every store (`node_embeddings=0`).
- Arm A kept as protocol `hybrid_search` (now labeled `fts_keyword_and`). Extra disclosure arm `lexical-or` = `Storage::search` (FTS OR+BM25); not in the locked gate.
- Metrics JSON: `benchmarks/causal-spike/results/*-7a2dc8d-20260901T052140Z.json`
- Runs: `benchmarks/causal-spike/data/runs-erratum-harness/` (gitignored)

| Arm | recall@1 | recall@3 | MRR | sep vs A | sep vs OR | multi-hop r@3 | mean list len | empty-list rate | mean ms/query | amortized ms/query |
|-----|----------|----------|-----|----------|-----------|---------------|---------------|-----------------|---------------|--------------------|
| A lexical (`fts_keyword_and`) | 0.00 | 0.00 | 0.00 | — | — | 0.00 | 0.00 | 1.00 | 2.03 | — |
| A′ lexical-or (`fts_or_bm25`) | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 10.00 | 0.00 | 2.08 | — |
| B backfill | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 (0/10) | 1.00 | 0.00 | 3.96 | — |
| C causal-graph | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (10/10) | 1.11 | 0.00 | 0.006 | 21.70 (incl. 1953 ms accumulation) |

Locked gate arithmetic (unchanged thresholds): C r@1 ≥ 0.60 ✓, C r@3 ≥ 0.80 ✓, C sep vs A ≥ 0.40 ✓, C r@3 ≥ B r@3 ✓, C multi-hop r@3 ≥ 0.50 ✓.

**Outcome: INVALID-BASELINE.** Arm A is still a null function (empty lists). The OR+BM25 disclosure arm returns 10 candidates per query and still recovers 0/90 planted causes, so the *ranking* moat vs a live lexical path is real — but the locked gate compares against Arm A, and a PASS against empty lists is refused. B's list length is exactly 1.00 (only the planted cause is backward-and-entity-eligible). C's 1.11 is that cause plus the root on the 10 multi-hop pairs. Claim licensed if this were a pass: receipt-backed upstream candidates that similarity search misses — NEVER automatic root cause.

### Errata

- SQLite row IDs are UUID v4 from `Storage::ingest`; identical *content* across reseeds, not identical DB bytes.
- Arm C accumulation is two passes (pre-run disclosure). Pass 1 persists `backfill_candidate` edges from every `looks_like_failure` memory. Pass 2 re-runs backfill with `manual=true` from every quiet memory promoted in pass 1, so `root → intermediate` edges exist for intermediates that carry no failure vocabulary. Pass 2 uses no ground truth: its source set is derived solely from pass-1 output. Without pass 2, multi-hop recovery is unreachable by construction and gate check 4 is undefined. This is not PCMCI / transfer entropy.
- Protocol Arm A is `hybrid_search`. Its keyword leg is FTS5 **implicit AND**. On this dataset that returns zero candidates after self-filtering. `Storage::search` (OR+BM25) is reported as extra arm `lexical-or` and is not substituted into the locked gate.
- If Arm A `empty_list_rate == 1.0`, `score` emits `INVALID-BASELINE` rather than PASS, even when the five locked checks are arithmetically true.
- Persisted edge strength is `score / (1 + score)` so `EVIDENCE_FLOOR=0.2` is on a (0,1) scale. Typical backfill scores are ≥ 1.0, so squashed strength is ≥ 0.5; the floor still does not reject those edges.
- Arm C query timer excludes the accumulation pass. Publish both `mean_wall_clock_ms` and `amortized_ms_per_query`.
- Each arm runs against a per-arm scratch copy of the store (`{stem}.{arm}.scratch.db`) so Arm C cannot mutate the seeded artifact.
- Multi-hop recall@3 is reported for every arm when the 10-root subset is present. The locked gate still reads C's number. Measured: B = 0/10, C = 10/10.
- `score` hard-errors on unparseable run JSON, missing protocol arms, duplicate `(arm, failure_id)`, extra/missing failure ids, and `dataset_id` mismatch. Silent skips are a p-hacking vector and are not allowed.
- `.gitignore` ignores `*.db` / wal / shm / `data/runs/` / `data/runs-*/` only. `manifest.json` and `store-*/failures.json` are committable.
- Official dataset fingerprint `dataset_id=8e199470e4807ecd` (FNV-1a of the sorted 90 failure ids) is recorded on the manifest, each `failures.json`, and every run. Metadata only — stores were not regenerated.
- Verdict artifacts carry `claim_boundary` verbatim plus `claim_never_licensed: "automatic root cause"`.
- A store holds 510 pair-memories plus 3–4 roots (513 or 514), 1540 across the three stores — not ~480.
- **Not changed (needs a new preregistered generation, Sam's call):** entity chatter is still all *after* the failure, so B/C have no backward competitor sharing the entity. Cause/failure templates still share stopwords; failure bodies keep the identifier in tags only. Do not rewrite planted text on the frozen 20260831 stores.
