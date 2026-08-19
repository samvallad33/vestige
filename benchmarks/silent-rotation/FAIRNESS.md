# Fairness retest — was Silent Rotation rigged?

Recounted 2026-08-19 from the arm JSON and the transcript files on
`benchmark/silent-rotation`. Reproduce the model split with
`python3 tests/by_model_tables.py`.

Two independent slices: [`GPT-5.6-SOL.md`](GPT-5.6-SOL.md) and
[`KIMI-K3.md`](KIMI-K3.md). If a claim only holds after mixing models, it
does not belong in a post.

## What “rigged” would mean

A reviewer is entitled to suspect:

1. The model was chosen because Vestige looks good on it.
2. Only Vestige got a write/coordination tool.
3. The causal edge was planted, so “memory architecture” did no work.
4. Failed competitor cells were dropped.
5. Scores exist for runs whose transcripts do not.

## What I rechecked, from files

**GPT-5.6 Sol is clean.** 5 trials × 3 arms × 3 agents = 45 transcripts. All 45
are on disk. Five distinct live keys. Memory layer alive on every RAG and
Vestige cell. Anarchy 0/5 (all converged wrong). RAG 0/5 (4 wrong, 1 split).
Vestige 5/5. First `vestige_backfill` contained the live key 15/15. First
`rag_search` contained it 0/15. `vestige_log` count is 3 on every sync fleet
and 0 on every anarchy and rag fleet.

**Kimi K3 is clean except one trial’s missing traces.** SuperMemory 5/5 with
transcripts. Vestige 4/4 transcript-backed (the JSON 5/5 includes
`runB-trial-1/sync`, which lists three transcript filenames that are not in
the folder). RAG 3/4 transcript-backed, never converged-wrong. Hindsight 0/3
all split. Dead memory layers in this slice: none.

**Withheld cells are labeled, not silently dropped.**
`results/WITHHELD-contaminated/README.md`: Zep runB-2 was a dirty FalkorDB
(previous trial’s live key served as current). mem0 runB-1 was `~/.mem0` not
cleared. Hindsight runB-3 was a dead backend (0/9 retrievals). Those are
harness bugs. They are not in either model table.

**The production oracle is outside the agent repo.** Local tests go green for
any matching key. That trap is identical across arms. It is why anarchy can
look like a win until replay.

## Real inequalities — say these out loud

1. **Tool parity.** `harness/agent/fleet_runner.py` gives sync
   `vestige_backfill` **and** `vestige_log`. RAG / mem0 / SuperMemory /
   Hindsight / Zep are read-only retrieval. Anarchy has neither. The ablation
   `rag-bus` (bus on cosine) went 0/5, so the bus alone does not create the
   win. The transcripts also show the kid in the first backfill payload before
   any log. Still: the tools are not equal. Do not claim “only the memory
   backend changed.”
2. **The causal edge is hand-authored.** `prepare_trial.py` tags cause and
   failure with a shared `active_key` entity. This measures traversal, not
   discovery. `sync-noedge` went 0/5, `causes: []`. Conjunction, not a single
   magic trick. See `ABLATION.md`.
3. **Kimi is a query-grinder.** Mixing Kimi RAG (3/4 correct) with GPT RAG
   (0/5) into one “RAG 4/23” table hides that. That is why these two files
   exist.
4. **SuperMemory also wins on Kimi.** 5/5. A Vestige-only victory lap on that
   model is false.
5. **n is thin on Hindsight (3) and Zep (2).** Do not lead with those as a
   bake-off.
6. **The original 25-trial sweep was not pre-registered.** The ablation was.
   Do not reverse that.

## What is not rigged

- Same failing test, same keyring size (50), same blank configs, same
  production replay, across both models.
- Live key not in any file the agents can read.
- GPT and transcript-backed Kimi Vestige first-call: 15/15 and 12/12.
- Query-based first-call on those same agents: GPT RAG 0/15; Kimi RAG 0/12;
  Kimi SuperMemory 1/15; Kimi mem0/hindsight/zep 0.
- Competitor arms did not receive `vestige_log`. Confirmed in the arm JSON
  `agents_used_vestige_log` field.
- Contaminated and dead-backend cells were withheld with receipts, then
  re-run.

## How to use this in a post

Lead with **GPT-5.6 Sol 0/5 / 0/5 / 5/5** (anarchy / rag / Vestige). Complete
traces. Then **Kimi K3**, where SuperMemory also goes 5/5 and first-call is
the thing that still separates Vestige (12/12) from everyone else. Then the
ablation: three axes, 0/5 each.

Do not ship a mixed-model headline as if it were one experiment on one brain.
