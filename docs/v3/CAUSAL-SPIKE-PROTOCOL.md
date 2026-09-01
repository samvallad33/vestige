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

*(append-only; empty at preregistration)*
