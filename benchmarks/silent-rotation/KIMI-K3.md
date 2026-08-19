# Silent Rotation — Kimi K3 only

Five trials. One model. Up to seven arms. This is the competitive slice.

Recount: `python3 tests/by_model_tables.py`

Trials: `results/runA-trial-{1,2}/`, `results/runB-trial-{1,2,3}/`.
Keys: `k_nashira`, `k_regulus`, `k_nashira` (reused), `k_regulus` (reused), `k_orion`.

## Headline (transcript-backed)

A cell counts here only if every transcript the arm JSON names is actually on
disk. That drops `runB-trial-1` anarchy / rag / sync, which have scores and no
traces. SuperMemory and Hindsight on that same trial **do** have transcripts
and stay in.

| arm | correct | wrong | split | n | first-call correct |
|---|---|---|---|---|---|
| anarchy | **0/4** | 4/4 | 0/4 | 4 | n/a |
| rag | **3/4** | 0/4 | 1/4 | 4 | 0/12 |
| **sync (Vestige)** | **4/4** | **0/4** | 0/4 | 4 | **12/12** |
| SuperMemory | **5/5** | **0/5** | 0/5 | 5 | 1/15 |
| mem0 | 2/4 | 0/4 | 2/4 | 4 | 0/12 |
| hindsight | 0/3 | 0/3 | 3/3 | 3 | 0/9 |
| zep | 0/2 | 1/2 | 1/2 | 2 | 0/6 |

If you take the arm JSON at face value, including the three runB-trial-1 cells
with no transcripts, Vestige is 5/5 and anarchy is 0/5 with one split. Do not
do that in a post. The missing files are listed by `tests/by_model_tables.py`.

## The thing this slice forces you to say

On Kimi, **SuperMemory is also 5/5.** Dense RAG never converged wrong (3
correct, 1 split of 4). Vestige is not “the only arm that worked.” Vestige is
the arm whose **first** memory call already contained the live key (12/12),
while every query-based first call on this model was 1/15 SuperMemory and 0
everywhere else.

Kimi grinds follow-up queries. That is why RAG and SuperMemory can still land
the fleet on the right key after a decoy first hit. GPT-5.6 Sol, in the sister
benchmark, mostly does not: RAG 0/5.

## Per trial (JSON verdicts; `[NO TX]` = score without traces)

| trial | live key | anarchy | rag | Vestige | SuperMemory | mem0 | hindsight | zep |
|---|---|---|---|---|---|---|---|---|
| runA-1 | k_nashira | wrong | **correct** | **correct** | **correct** | split | split | split |
| runA-2 | k_regulus | wrong | **correct** | **correct** | **correct** | **correct** | — | — |
| runB-1 | k_nashira | split `[NO TX]` | split `[NO TX]` | **correct `[NO TX]`** | **correct** | withheld | split | — |
| runB-2 | k_regulus | wrong | split | **correct** | **correct** | split | split | — |
| runB-3 | k_orion | wrong | **correct** | **correct** | **correct** | **correct** | — | wrong |

Study runA-trial-1 first. All seven arms. The Mem0/Hindsight 2-right-1-wrong
pair lives here. Vestige a0/a1/a2 all wrote `k_nashira`.

## Cleanliness

- Memory layer alive on every Kimi cell in this table. Zero dead backends.
- `vestige_log` used only on sync (3/3 agents). Competitors: 0.
- First sync memory tool, on every transcript-backed agent: `vestige_backfill`,
  payload already contains the live key.
- Withheld (not in this table): contaminated Zep runB-2, dirty `~/.mem0`
  runB-1, dead Hindsight runB-3. Those folders are under
  `results/WITHHELD-contaminated/`.
- Key reuse: `k_nashira` and `k_regulus` each appear twice. Not ideal. Still
  four distinct keys across five trials, and GPT’s five keys are all unique.

![Kimi K3 — transcript-backed](figures/silent-rotation-kimi-k3.png)
