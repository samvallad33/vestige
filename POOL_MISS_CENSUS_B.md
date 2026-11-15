# POOL_MISS_CENSUS_B

Baseline is the combined Step 31+32+33+34 tip `95ef2f228de94e6b4ab8d8c3566bac3847b5c3ca`, binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`. No ranker edit is in these numbers. Official scores are the no-dump `c2_bench.py` run (`/workspace/c2-runs/baseline.json`). Stage counts are the replay dump (`VESTIGE_REPLAY_DUMP`), which is a second ranking of the same requests.

## What the dump fields are

Verified in `write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`):

- `pre_ids` / `candidates` is the pool. It is the seated turn list first, then a `t_scene`-descending fill up to 50 among turns with `t_scene > 0`.
- `rank_pre == 0` is therefore `seated[0]`. It is not the scene argmax. Scene rank 0 below is recomputed as argmax `t_scene` over the pool (tie broken by turn index).
- `final_ids` / `seated_ids` is the official turn order returned to the bench.
- `t_scene` is the score vector snapshotted at `maybe_dump_replay`, after seating has reordered the list. The vector itself is not reordered.
- `t_own` is the turn's own lexical score copied into that same snap. The standing floor writes `t_scene`, not `t_own`.
- `final_session_ids` is the official session list.
- `session_turns` would be every turn of a session that is in the turn top 10 or the session top 10. `c2_dumps.py` drops that array before writing the jsonl, so a gold turn that is not in the pool has no score in the file even when its session was kept.
- Turns with `t_scene <= 0` that are outside the seated list never enter the pool. Sessions that lose the session cut are absent from `final_session_ids`, and the dump has no `s_score` for them, so "filtered at `s_score > 0 || bridge_boost > 0`" cannot be separated from "ranked 11+".

The dump re-rank and the no-dump bench disagree on Turn any-hit @1/@5/@10 for 16 of 355 BEAM questions (instruction_following 8, event_ordering 4, multi_session_reasoning 2, temporal_reasoning 1, knowledge_update 1). The scoreboard below is the no-dump bench. The funnel is the dump. Do not treat a dump hit-count as the published metric.

## Official baseline (no-dump bench)

| Bench | Turn @1/@5/@10 | Turn All@5 | Session @1/@5/@10 |
| --- | --- | --- | --- |
| LME_S n=470 | 353/428/437 | 314 (All@10 447) | 430/463/468 (All@5 424) |
| LoCoMo n=1536 | 808/1127/1226 |  | 1133/1416 |
| BEAM-100K n=355 | 160/238/267 | 118 (All@10 138) | 133/249/275 (All@5 137, All@10 178) |

BEAM by category, Turn @1/@5/@10 and All@5 / All@10, then Session @1/@5/@10:

| Category | n | T@1/@5/@10 | All@5/10 | S@1/@5/@10 |
| --- | --- | --- | --- | --- |
| contradiction_resolution | 40 | 32/40/40 | 22/27 | 29/40/40 |
| event_ordering | 40 | 7/14/22 | 0/0 | 7/22/27 |
| information_extraction | 40 | 20/27/29 | 20/21 | 15/25/27 |
| instruction_following | 40 | 6/18/21 | 13/16 | 4/16/17 |
| knowledge_update | 40 | 31/37/39 | 24/27 | 24/37/38 |
| multi_session_reasoning | 40 | 22/34/36 | 4/7 | 20/33/36 |
| preference_following | 39 | 7/14/16 | 11/13 | 4/14/24 |
| summarization | 36 | 6/17/25 | 0/0 | 6/24/26 |
| temporal_reasoning | 40 | 29/37/39 | 24/27 | 24/38/40 |

## Five-line funnel (any gold in the stage)

BEAM, from `baseline-beam.jsonl`. Corpus is the chat turn id set.

```
[BEAM all] n=355
  gold in corpus      355
  gold in pool@50     329
  gold at scene rank0 125
  gold at final rank0 159
  gold in final top5  240
```

Per category, pool / scene0 / final0 / final5 / final10:

| Category | pool@50 | scene0 | final0 | final5 | final10 |
| --- | --- | --- | --- | --- | --- |
| contradiction_resolution | 40 | 32 | 32 | 40 | 40 |
| event_ordering | 40 | 5 | 7 | 14 | 20 |
| information_extraction | 32 | 16 | 20 | 27 | 29 |
| instruction_following | 34 | 4 | 7 | 20 | 23 |
| knowledge_update | 39 | 21 | 30 | 37 | 39 |
| multi_session_reasoning | 40 | 19 | 22 | 34 | 36 |
| preference_following | 30 | 2 | 7 | 14 | 16 |
| summarization | 34 | 7 | 6 | 17 | 25 |
| temporal_reasoning | 40 | 19 | 28 | 37 | 39 |

LoCoMo, from `c2_funnel.py` on the same binary (one gold evidence id in Temporal is not a turn id in the conversation, so corpus is 320/321):

```
[LoCoMo Multi-Hop] n=282
  gold in corpus     282
  gold in pool@50    243
  gold at scene rank0 72
  gold at final rank0 97
  gold in final top5 158
[LoCoMo Temporal] n=321
  gold in corpus     320
  gold in pool@50    299
  gold at scene rank0 182
  gold at final rank0 191
  gold in final top5 248
[LoCoMo Open-Domain] n=92
  gold in corpus     92
  gold in pool@50    63
  gold at scene rank0 14
  gold at final rank0 20
  gold in final top5 35
[LoCoMo Single-Hop] n=841
  gold in corpus     841
  gold in pool@50    810
  gold at scene rank0 440
  gold at final rank0 500
  gold in final top5 686
```

LME, from `baseline-lme.jsonl`. Gold turns are `has_answer` turns inside the indexed haystack, so corpus membership is by construction (470/470). Pool / scene0 / final0 / final5 / final10: 463 / 265 / 353 / 428 / 437. 23 questions have a gold session outside session top 10 (multi-session 10, temporal-reasoning 13).

## BEAM gold turns outside pool@50

368 gold-turn instances are outside `pre_ids`. The dump has no `t_scene` for them (`session_turns` was stripped, and they are not in the pool). The split below is surface-token overlap of the raw turn text with the question after a closed-class stop list. It is not the engine stemmer and not a postings proof.

| Category | in pool | absent, no shared surface token | absent, some shared surface token | of which session is in session top 10 |
| --- | --- | --- | --- | --- |
| contradiction_resolution | 83 | 1 | 4 | 2 |
| event_ordering | 113 | 24 | 93 | 11 |
| information_extraction | 54 | 2 | 12 | 2 |
| instruction_following | 42 | 6 | 4 | 4 |
| knowledge_update | 81 | 4 | 2 | 0 |
| multi_session_reasoning | 127 | 5 | 33 | 1 |
| preference_following | 37 | 0 | 10 | 0 |
| summarization | 115 | 20 | 148 | 1 |
| temporal_reasoning | 74 | 0 | 0 | 0 |

Death stage that the dump can actually support: a gold turn not in `pre_ids` was not seated and was not among the `t_scene > 0` turns that filled the pool to 50. The "no shared surface token" rows are the ones a lexical postings pass is least likely to have scored; standing-policy floors can still lift a zero-overlap imperative, so an empty surface overlap is not "never posted."

## BEAM gold sessions outside session top 10

383 gold-session instances (a session that holds at least one gold turn) are outside `final_session_ids[:10]`. The dump cannot say whether `s_score` was zero.

| Category | in session top 10 | outside |
| --- | --- | --- |
| contradiction_resolution | 67 | 10 |
| event_ordering | 42 | 136 |
| information_extraction | 27 | 13 |
| instruction_following | 21 | 21 |
| knowledge_update | 66 | 11 |
| multi_session_reasoning | 63 | 41 |
| preference_following | 24 | 15 |
| summarization | 39 | 129 |
| temporal_reasoning | 66 | 7 |

## Where a new seat can move a published number

Instruction following is the BEAM cell furthest from the lexical head: pool holds a gold on 34/40 questions, final top 5 on 20, final rank 0 on 7 in the dump, and the bench Turn @1 is 6. The misses that are already inside the top 5 can change Turn @1 without changing the top-5 set. Promoting an arbitrary user non-question session opener over a question head is not that seat: on the dump it gains instruction Turn @1 and loses preference, temporal, knowledge, and information-extraction heads that are themselves the gold. Seating every zero-lexical standing-rule winner on a task question also loses: `10:knowledge_update:1` is a question that states the deadline, and the channel's winner is a zero-specificity editing-progress rule. The seat that holds is narrower. The winner must carry an explicit conditional, and one of its stored content terms must be an inflection of a query focal term (`implement` / `implementation`). That is the change in `SCOREBOARD_POOL_B.md`.
