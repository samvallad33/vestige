# LongMemEval-S offline adapter

Official code: https://github.com/xiaowu0162/LongMemEval at
`9e0b455f4ef0e2ab8f2e582289761153549043fc` (MIT). Official cleaned data:
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned at
`98d7416c24c778c2fee6e6f3006e7a073259d48f` (MIT dataset card).
The complete `longmemeval_s_cleaned.json` is downloaded locally: 277,383,467
bytes, SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The earlier bounded 8MiB prefix is preserved in ignored data as an unused access
artifact; all finalized preparation uses the complete hash-verified file.

```bash
python3 -m campaigns.longmemeval.prepare --help
python3 -m campaigns.longmemeval.prepare \
  --upstream upstream/longmemeval \
  --output evidence/<fresh-name>/longmemeval --limit 10
```

Omit `--limit` to prepare all 500 questions and 23,867 history sessions, including
all six question types and 30 abstention cases. A limited sample follows a
predeclared category/abstention stratification, then fills by source order and
emits in source order. It never inspects model outputs or answers for selection.
Category counts and abstention counts are recorded in the manifest, and those
labels remain hidden from runtime prompts and retrieval tags.

`history.jsonl` retains original session order, exact timestamp strings and
user/assistant turn content. Both old and updated facts remain; no reference
answer determines overwrites or causal edges. Original `has_answer`, source
question IDs, category labels, abstention suffixes and evidence session IDs live
only in `oracle.jsonl`. `questions.jsonl` uses neutral runtime IDs and carries the
actual question/date, never the answer. All 13 source records with repeated
session IDs retain every occurrence under unique ordinal handles. Oracle-only
alias groups credit either occurrence as the same original session for retrieval.

```bash
python3 -m campaigns.longmemeval.metrics \
  --oracle <prepared>/oracle.jsonl --predictions <predictions.jsonl> \
  --output <fresh-judge-payloads.jsonl>
```

Predictions use JSONL `{"question_id":"lme-q00000","hypothesis":"..."}`.
This command extracts the exact upstream `get_anscheck_prompt` definition without
executing provider imports and produces actual-question-bearing payloads. It
never calls a model. Outputs stay `JUDGE_NOT_EXECUTED`; no heuristic exact-match
substitute is called official QA accuracy. Knowledge-update, temporal and
abstention rules stay in the upstream judge template.

`aggregate_judgements` aggregates explicit raw yes/no labels with judge and request
identities; supplied evidence authenticity still needs the audited launch-layer
transport. `retrieval_metrics` reports session precision/recall and original
recall-any/all semantics, excluding abstention cases as upstream does. A prepared
corpus is not native kernel qualification, a live agent run, or a benchmark win.
