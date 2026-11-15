# Pool miss census A (LoCoMo)

Binary: Step 31 + 32 + 33 + 34 tip on `perf/c2-pool-locomo`, before the buried-lexical-peak seat. Measured with `retrieval_bench` and `VESTIGE_REPLAY_DUMP`.

## What the dump fields are

Verified in `write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`):

- `pre_ids` is pool@50. It is the seated list (`final_turn_indices`) first, then other turns with `t_scene > 0` in scene order (`order(t_scene, session_ts)`), stopped at 50.
- `final_ids` is the official seated turn order. `final_ids[0]` is final rank 0. `rank_pre` on a candidate is the index in `pre_ids`, so `pre_ids[0]` is `final_ids[0]` whenever the seated list is non-empty. Scene-rank 0 is not that index. It is the argmax of `t_scene` over the pool, tie broken to the lower turn index.
- `final_session_ids` is the official session order. Session top 10 is the first 10.
- `score_rows` (dump-only, every indexed turn) is `[id, session_id, session_order, t_scene_bits, t_own_bits, t_turn_raw_bits, flags]`. Flag bit 0 is `is_question`. Ranking does not read `score_rows`.
- `session_rows` is `[session_id, s_score_bits, bridge_boost_bits]` for every indexed session.
- A turn is dropped by the session candidate set when its session has `s_score == 0` and `bridge_boost == 0`. That is the filter at the submodular session loop (`s_score[si] > 0.0 || bridge_boost[si] > 0.0`, around the `remaining` collect near line 5489).
- A turn is dropped by postings when it is indexed and `t_scene == t_own == t_turn_raw == 0`. Own and carried BM25 never left zero, and neighbor scene carry did not either. The posting loop that would have given it features is the content-token ingest around lines 3125–3274.
- A turn is below the pool cut when `t_scene > 0` and its id is absent from `pre_ids`.

## Stage funnel (before any new seat)

`tools/c2_funnel.py` on this tip:

| Category | n | gold in corpus | pool@50 | scene rank 0 | final rank 0 | final top 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-Hop | 282 | 282 | 243 | 72 | 97 | 158 |
| Temporal | 321 | 320 | 299 | 182 | 191 | 248 |
| Open-Domain | 92 | 92 | 63 | 14 | 20 | 35 |
| Single-Hop | 841 | 841 | 810 | 440 | 500 | 686 |

Pool@50 here is questions with at least one gold turn in `pre_ids`. 90% of n is 254 / 289 / 83 / 757. Open-Domain is short by 20 questions. Multi-Hop is short by 11. Temporal and Single-Hop already clear 90% at pool@50.

## Gold turns absent from pool@50

Every such turn, by earliest stage. Counts are turns, not questions. A multi-gold question contributes one row per missing evidence id.

| Category | never indexed | dropped by postings | scored then zeroed | below pool cut |
| --- | ---: | ---: | ---: | ---: |
| Multi-Hop | 3 | 0 | 0 | 344 |
| Temporal | 1 | 0 | 0 | 40 |
| Open-Domain | 0 | 0 | 0 | 113 |
| Single-Hop | 0 | 0 | 0 | 36 |
| Total | 4 | 0 | 0 | 533 |

No indexed gold turn has a zero scene score. The old miss-map `no_pool` bucket is not a zero-posting stage on this binary: the golds outside pool@50 are scored and then cut.

Same-session distance from a below-cut gold to the nearest pool member:

| Category | adjacent | within 3 | no pool mate in session |
| --- | ---: | ---: | ---: |
| Multi-Hop | 44 | 48 | 252 |
| Temporal | 5 | 5 | 30 |
| Open-Domain | 14 | 12 | 87 |
| Single-Hop | 14 | 7 | 15 |

Open-Domain and Multi-Hop misses are mostly not adjacency misses. The turn is scored, in another part of the dialogue, and loses the 50-cut on `t_scene` even when `t_own` is large. Example: LoCoMo `1:13` gold `D6:6` has `t_own = 65.32` and `t_scene = 1.46`, and the pool floor on that question is `t_scene = 5.47`.

Never indexed (the id is not in `score_rows`):

- `3:58` `D10:19` (Multi-Hop)
- `4:18` `D:11:26` (Multi-Hop; the id has an extra colon, so the session key is not `D11`)
- `6:38` `D4:36` (Multi-Hop)
- `9:69` `D30:05` (Temporal; zero-padded turn index, the index stores `D30:5`)

## Gold sessions absent from the session top 10

| Category | never indexed | dropped by session candidate set | below session cut |
| --- | ---: | ---: | ---: |
| Multi-Hop | 1 | 0 | 185 |
| Open-Domain | 0 | 0 | 54 |
| Single-Hop | 0 | 0 | 11 |
| Temporal | 0 | 0 | 11 |
| Total | 1 | 0 | 261 |

The one never-indexed session is `4:18` session `D`, from the malformed id `D:11:26`. Every other missing gold session has positive `s_score` or `bridge_boost` and lost the top-10 cut. The candidate-set filter at line 5489 does not remove them.
