# LongMemEval — Vestige retrieval harness

Measures **retrieval only**: given a question and its haystack of chat sessions,
how often does Vestige surface the session that actually contains the answer?

No generation stage, no LLM judge, no API key, no network. `cargo run` and a
dataset file are the whole dependency list.

## What these numbers are, and are not

**These are recall@k. They are NOT end-to-end QA accuracy.**

Most published LongMemEval results quote QA accuracy from the benchmark's
generation + LLM-judge stages, which need an OpenAI key. This harness runs
neither. A recall number and a QA-accuracy number are not comparable, and
reporting recall as a bare "LongMemEval score" would be exactly the failure the
[ai-memory-comparison](https://github.com/carsteneu/ai-memory-comparison)
benchmark column already suffers from, where entries mix R@1, R@5, session-level
and turn-level without saying which.

Always report as: `LongMemEval-S, session-level recall@5 = X%, retrieval-only`.

## Pre-registered configuration

Fixed before the first full run, so nothing here can be tuned to flatter a
result after seeing one:

| choice | value | why |
|---|---|---|
| retrieval | `Storage::hybrid_search` | the shipped path: FTS5/BM25 + HNSW cosine fused with **RRF k=60**. The legacy keyword/semantic weight parameters in the signature are no-ops in the shipped code (`let _ = (keyword_weight, semantic_weight)`) — nothing here is tunable, which is the point |
| k values | 1, 3, 5, 10 | fixed up front |
| candidates fetched | 10 | one query per instance, scored at every k |
| modes | whole-session (primary) + chunked-session | whole-session is production-faithful; chunked (≤7500-byte chunks, never mid-turn, session-level scoring) removes the embedder's 8192-BYTE truncation blindness — measured on LongMemEval-S: 63.5% of all sessions and **92.8% of gold sessions** exceed 8192 bytes, so whole-session semantic matching sees only their heads (FTS still indexes full text) |
| exclusions | 30 `_abs` abstention instances | upstream skips them in retrieval evaluation; scoring them would make numbers non-comparable → 470 scored |
| metrics | `recall@k` (any gold in top k) and `recall_all@k` (all gold in top k) | any-hit is lenient for multi-session questions; both are always reported |
| store | fresh temp SQLite per instance | no cross-question leakage |
| per-question tuning | none | same config for all 470 scored instances |

`init_embeddings()` is called after opening each store. Without it embeddings are
never generated and hybrid search silently degrades to keyword-only, which would
quietly understate the result.

## Baseline for interpretation

On LongMemEval-S: median 48 sessions per instance, median 2 of them gold.
**Random session-level recall@1 is therefore about 4.2%.** Any reported number
should be read against that, not against zero.

## Data

`xiaowu0162/longmemeval-cleaned` on HuggingFace (MIT). Note the older
`xiaowu0162/longmemeval` dataset is **deprecated** by its author because it
contained noisy history sessions that interfered with answer correctness. Use
the cleaned one.

```sh
curl -L -o longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

`longmemeval_oracle.json` (14.7 MB) contains **only the answer sessions**, so
session-level recall against it is trivially ~100%. It is useful for validating
the harness end to end, and useless as a performance measurement. Turn
granularity on oracle *is* discriminating, since only about 8% of turns are gold.

## Running

```sh
cargo build --release
./target/release/lme-retrieval longmemeval_s_cleaned.json --granularity session
./target/release/lme-retrieval longmemeval_s_cleaned.json --granularity turn --limit 50
```

Scale: a full session-granularity run is 23,867 ingests. Turn granularity over
the full set is 246,750 ingests and takes considerably longer.

## What is not measured here

- End-to-end QA accuracy (needs the generation + judge stages, and an API key)
- The harder LongMemEval-M set (~500 sessions per instance)
- Abstention, which LongMemEval scores through the QA stage

Upstream benchmark: <https://github.com/xiaowu0162/LongMemEval>, ICLR 2025,
[arXiv:2410.10813](https://arxiv.org/abs/2410.10813).
