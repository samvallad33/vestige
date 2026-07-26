# Preregistration — LongMemEval-S retrieval-only evaluation of Vestige

Committed **before** the v2 results are produced. The commit adding this file
contains the exact harness that generates them; any later change to harness or
config appears as a later commit touching this directory.

## Fixed configuration

| choice | value |
|---|---|
| dataset | `longmemeval_s_cleaned.json` from `xiaowu0162/longmemeval-cleaned` (MIT), 500 instances |
| exclusions | the 30 abstention instances (`question_id` ending `_abs`), matching upstream's retrieval evaluation → **470 scored** |
| retrieval | `Storage::hybrid_search` — the shipped path: FTS5/BM25 + USearch HNSW cosine, RRF k=60 fusion. The legacy keyword/semantic weight parameters are no-ops in the shipped code and play no role |
| embedder | Nomic Embed Text v1.5 (ONNX, 768d → 256d Matryoshka), the shipped default; input truncated by the shipped code at 8192 **bytes** |
| k values | 1, 3, 5, 10 |
| candidates fetched | 10 per query, one query per instance (the raw question text, unmodified) |
| store | fresh temp SQLite per instance; no cross-question state |
| tuning | none — same config for all instances; no per-type or per-question adjustment |

## Modes (both will be run and both will be published)

1. **whole-session** (production-faithful): one node per session. Disclosure:
   63.5% of all sessions and 92.8% of gold sessions exceed the embedder's
   8192-byte truncation, so semantic matching sees only their heads; FTS
   indexes full text. This mode measures what a user of the shipped ingest
   path would actually get.
2. **chunked-session**: sessions split at ≤7500 bytes, never mid-turn; scoring
   collapses chunks back to sessions (first-appearance rank). This mode removes
   truncation blindness while keeping session-level scoring.

## Metrics

- `recall@k` — at least one gold session in the top k (lenient / any-hit).
- `recall_all@k` — **every** gold session in the top k (strict; multi-session
  questions have up to 4 gold sessions and only 10 candidates are fetched, so
  this is a hard metric by construction).

Primary published headline: **whole-session `recall@5` (any-hit)** with the
chunked number and both `recall_all` columns alongside. Per-question rows are
dumped to JSONL and published with the aggregates.

## Baseline

Median 48 sessions per instance, median 2 gold → random session-level
recall@1 ≈ 4.2%. Results are read against that, not against zero.

## Known fidelity gaps (disclosed, unfixed by design)

- Gold-correlated tokens: dataset gold session ids all contain the literal
  token `answer`; the harness therefore never stores session ids — nodes carry
  opaque `s<idx>` tags and the gold mapping lives outside the store.
- Instances where any node fails to embed are skipped and listed, never
  silently scored.
- Nomic task prefixes (`search_document:` / `search_query:`) are not applied,
  because the shipped ingest/search path does not apply them.
- Session timestamps are not indexed; temporal-reasoning questions get no
  recency signal beyond content.
- RRF ties can order-flip between runs (hash-map iteration inside the shipped
  fusion); at the top-10 cut this can move a tied item in or out.
- `recall@k` here is the any-hit variant; upstream does not formally define a
  single Recall@K, so cross-paper comparisons should quote definitions, not
  bare numbers.

## What this is not

Not end-to-end QA accuracy (no generation, no LLM judge, no API key), not
LongMemEval-M, and not an abstention measurement. Never report these numbers
as a bare "LongMemEval score".
