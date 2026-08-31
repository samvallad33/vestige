# MemConflict benchmark harness

Retrieval benchmark for Vestige against
[MemConflict](https://arxiv.org/abs/2605.20926) (arXiv:2605.20926), with
mandatory no-memory, random, and BM25 controls.

**Read [`../../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) before quoting any
number from this harness.** It documents what is and is not measured, and why
absolute values here are not comparable to the paper's published tables.

## Quick start

```sh
cargo build -p vestige-mcp
python3 benchmarks/memconflict/fetch_dataset.py
python3 benchmarks/memconflict/run.py --instances 2 --sessions 25 --top-k 5
```

No `pip install`. The harness is standard library only, by design: a benchmark
that needs a dependency resolver is a benchmark that stops reproducing.

## Files

| File | Purpose |
| --- | --- |
| `DATASET.lock.json` | Pinned upstream revision + SHA-256 of every data file. |
| `fetch_dataset.py` | Downloads the pinned data and verifies hashes. Hard-fails on mismatch. |
| `judge.py` | Faithful port of upstream `Evaluation/eval_scoring.py` rule-based judge. No LLM. |
| `bm25.py` | Okapi BM25 control. Pure stdlib. |
| `mcp_client.py` | JSON-RPC-over-stdio client for `vestige-mcp`. |
| `run.py` | Orchestrator. Runs every arm, writes results JSON. |
| `LONGMEMEVAL.lock.json` | Pinned LongMemEval_S (cleaned) revision + hash. |
| `longmemeval.py` | LongMemEval_S **sanity check only** — see below. |
| `results/` | Timestamped run outputs. |

## The four arms

All four run every time. None can be silently dropped.

- **`nomem`** — no retrieval. The floor.
- **`random`** — K random memories, seeded. The blob-inflation control: the
  judge awards partial credit for token overlap, so any K memories score above
  zero by chance. Beat this or you are reporting corpus statistics.
- **`bm25`** — Okapi BM25 on the identical corpus. The earned-complexity bar.
- **`vestige`** — live `recall()` over MCP.

Same corpus, same reader, same judge, same K for every arm. One variable
changes: retrieval.

## Useful flags

```
--instances N        simulated users to evaluate (default 1)
--sessions N         max sessions ingested per user (default 10)
--top-k N            memories retrieved per question, identical for all arms (default 5)
--recall-mode        lookup (default) | reason
--warmup SECONDS     embedding warmup before first tools/call (default 45)
--arms a,b,c         subset of arms; use for fast offline iteration
--seed N             seed for the random control (default 1234)
--out PATH           results JSON path
```

Static-conflict questions (the ones that exercise CRS) first appear around
session 10-17 depending on the instance. `--sessions 25` is the practical
floor for covering all three conflict types.

Fast offline iteration, no server needed:

```sh
python3 benchmarks/memconflict/run.py --arms nomem,random,bm25 --instances 1 --sessions 12
```

## Gotchas

- **Do not skip the warmup.** The embedding service warms up asynchronously
  after `initialize` returns. Calling a tool early silently measures a degraded
  keyword-only path. Confirm a run was healthy by checking `semanticScore` is
  non-null in retrieval output.
- **stdin must stay open** for the server's lifetime; closing it ends the run.
- Each simulated user gets its own `scope` namespace so memories cannot bleed
  between users.
- Run size is bounded by ingest throughput (~7 memory units/s on the reference
  machine).


## LongMemEval_S sanity check

```sh
python3 benchmarks/memconflict/longmemeval.py --questions 5
```

**Not a headline benchmark and never quotable as a LongMemEval score.** It runs
the same four arms on an independent dataset and reports one thing:
`evidence_recall@k` — does the retrieved text contain the gold answer string?
It exists to catch a silently broken harness, not to score the product.

It pins `longmemeval-cleaned`. The original `xiaowu0162/longmemeval` dataset is
**deprecated upstream** (it contains noisy history sessions that interfere with
answer correctness), so any number computed against the original is invalid.
