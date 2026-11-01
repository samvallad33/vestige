# MemoryAgentBench offline adapter

Official code: https://github.com/HUST-AI-HYZ/MemoryAgentBench at
`538026089d1a8a8eff05121d0db89b388f360eba` (MIT). Official data:
https://huggingface.co/datasets/ai-hyz/MemoryAgentBench at
`7ea066982b140a19337e17e60d45d4076e042faf` (MIT dataset card).
All four official Parquet splits and `entity2id.json` are downloaded locally in
ignored `data/`. Parquet files are checked against their published LFS SHA-256.
Preparation needs `pyarrow` for those files; JSON fixture inputs use stdlib only.
It does not import the upstream agent stack, dotenv, models, or provider clients.

```bash
python3 -m campaigns.memoryagentbench.prepare --help
python3 -m campaigns.memoryagentbench.prepare \
  --upstream upstream/memoryagentbench \
  --output evidence/<fresh-name>/memoryagentbench --limit 1
```

`--limit` selects the first N contexts **per source variant** before seeing any
model output; omit it to prepare all 146 contexts / 3,671 questions. The prepared
manifest records actual files, hashes, sizes, selected source counts, adapter and
scorer source identities. `history.jsonl` contains only lossless chunks of source
context. The original numbered fact updates remain intact. There is no invented
timestamp or automatic gold-based causal edge. Chunking uses 4,096 Unicode
characters and is explicitly different from upstream's tokenizer/sentence chunks.
Every arm must use this same frozen chunk sequence for this adapter comparison.

`questions.jsonl` carries raw questions and a common upstream query template.
`oracle.jsonl` alone carries answers, source categories, original IDs, evidence
annotations, keypoints and abstention labels. Runtime question IDs and chunk tags
are neutral indices. A native host must discover memory handles from write
receipts and those neutral scopes; oracle metadata must never become retrieval
tags, prompts or state writes. Preparing these files does not execute native
hooks or qualify any kernel mechanism.

```bash
PYTHONHASHSEED=0 python3 -m campaigns.memoryagentbench.metrics \
  --oracle <prepared>/oracle.jsonl --predictions <predictions.jsonl> \
  --output <fresh-results.json>
```

Predictions are JSONL `{"question_id":"mab-c00000-q00000","hypothesis":"..."}`.
The primary EM/substr metrics execute exact selected upstream pure functions,
including source-specific parsing. Recommendation Recall@5 uses the upstream
fuzzy parser and entity mapping with equivalent stdlib Levenshtein/argmin
primitives. Upstream `set` iteration makes tie order process-seed-dependent;
`PYTHONHASHSEED=0` is required for this path. Missing seed produces a pending
record rather than a score. The parser's raw-URI and fuzzy-match quirks are
preserved; a canonical movie-name fixture passed, and a raw-URI fixture scored
zero without being rewritten into a win.

LongMemEval-style tasks need one judge call; summarization needs the upstream
fluency, recall and precision rubrics. This adapter only creates question-bearing
payloads with `JUDGE_NOT_EXECUTED`, zero budget and zero requests. Summary rubrics
get the actual question prepended, a disclosed change from the upstream rubric.
`score_supplied_judge_outputs` performs the original summary F1 calculation only
when all raw judge records and identities are supplied. It validates evidence
shape; provider authenticity belongs to the separately audited transport.
Missing judge results never become correctness scores. No paid run is authorized
by preparation or by access to an existing credential.
