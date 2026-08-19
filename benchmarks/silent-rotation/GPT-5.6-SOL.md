# Silent Rotation — GPT-5.6 Sol only

Five trials. One model. Three arms. Every transcript on disk.

This slice exists so the result cannot be dismissed as “it only works on Kimi.”
The task, the 50-key keyring, the local-green-any-key trap, and the production
replay oracle are the same as the rest of the benchmark. The only thing that
changes is the model, and the memory tool.

Recount: `python3 tests/by_model_tables.py`

Keys, randomized per trial: `k_yildun`, `k_bellatrix`, `k_wren`, `k_indus`, `k_dorado`.
All five unique. Directories: `results/gpt-5.6-sol-trial-{1-5}/`.

## Headline

Did all three agents write the same key, and was it the live one?

| arm | correct | wrong (tests green, prod voids) | split | n | transcripts |
|---|---|---|---|---|---|
| anarchy (no memory) | **0/5** | **5/5** | 0/5 | 5 | 15/15 |
| rag (dense cosine) | **0/5** | 4/5 | 1/5 | 5 | 15/15 |
| **sync (Vestige)** | **5/5** | **0/5** | 0/5 | 5 | 15/15 |

First memory-tool call contained the live key:

| arm | first call correct | n |
|---|---|---|
| **sync (Vestige, no query)** | **15/15** | 15 |
| rag | 0/15 | 15 |

Memory layer alive on every RAG and Vestige cell. Zero retrieval errors. Zero
missing transcript files. `vestige_log` was used only on the sync arm (3/3
agents per trial). RAG never got a write bus.

## Per trial

| trial | live key | anarchy | rag | Vestige |
|---|---|---|---|---|
| 1 | k_yildun | green_but_voids_prod | green_but_voids_prod | **fixed_correctly** |
| 2 | k_bellatrix | green_but_voids_prod | green_but_voids_prod | **fixed_correctly** |
| 3 | k_wren | green_but_voids_prod | green_but_voids_prod | **fixed_correctly** |
| 4 | k_indus | green_but_voids_prod | green_but_voids_prod | **fixed_correctly** |
| 5 | k_dorado | green_but_voids_prod | failed_merge_conflict | **fixed_correctly** |

Study trial 1: `results/gpt-5.6-sol-trial-1/`. RAG agent a1 read “staging only”
and still called `k_wezen` the “authoritative active platform key.” Vestige a0
called `vestige_backfill {}` on turn 1, in parallel with `run_tests`, and all
three agents wrote `k_yildun`.

## What this slice does not contain

No mem0, SuperMemory, Hindsight, or Zep. Those were wired later and ran on Kimi.
Do not quote a 7-arm GPT table. There isn’t one.

n=5 is small. It is also complete: every claimed transcript is in the repo.

![GPT-5.6 Sol — three arms, five trials](figures/silent-rotation-gpt-5.6-sol.png)
