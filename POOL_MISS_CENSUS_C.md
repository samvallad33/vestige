# Pool miss census C (LongMemEval focus)

Baseline binary: combined Steps 31–34, `retrieval_bench` sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`. Dump: `VESTIGE_REPLAY_DUMP` on all 2,361 scored questions (LME 470, LoCoMo 1,536, BEAM 355). No ranker edit had been measured yet; this census is that binary.

## What the dump fields are

Verified in `write_one_replay_dump` (`crates/vestige-core/src/retrieval.rs`, pool build at the `pool.len() >= 50` loop, emit at `pre_ids` / `final_ids`).

- `final_ids` and `seated_ids` are `snap.seated`, the official turn order after the gatekeeper and every C2 seat including Steps 32, 33, and 34. `rank` is the index in that list. Rank 0 is slot 1.
- `pre_ids` is that seated list first, then turns with `t_scene > 0` in `order()` (higher `t_scene`, then higher session timestamp, then lower turn index) until 50. `rank_pre` is the index in `pre_ids`. `pre_ids[0]` is `final_ids[0]` whenever the seated list is non-empty, so gold at pre_rank 0 and gold at final rank 0 are the same count. Scene rank 0 is a different number: argmax of `t_scene` over the pool, tie broken by lower `idx`.
- `t_scene` is the scene-pooled score copied onto the snapshot. Seats do not rewrite it. `t_own` is the uncarried turn score. `s_score` and `bridge_boost` are the session score and its bridge term.
- The wide scored pool inside `rank_one` is every turn with `t_scene > 0` (`remaining` at the submodular turn loop). That set is larger than `pre_ids`. A turn with `t_scene <= 0` never enters it.
- Session candidates are sessions with `s_score > 0` or `bridge_boost > 0` (the `remaining` filter in the session submodular loop). A session with both at 0 is not a candidate.
- `session_turns` lists every turn of sessions that placed in the official turn top 10 or the official session top 10. A gold absent from both `pre_ids` and `session_turns` has no score in the dump.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body`. They are dump-only. Ranking does not.

Ingest postings are the feat loop that interns content tokens onto each turn (the `for w in raw.iter().filter(|w| is_content(w))` walk). Every haystack turn is indexed. "Never posted" here means the turn was indexed and still drew `t_scene <= 0` and `t_own <= 0`: no query term survived that posting walk, and scene neighbours did not lift it.

## Five-line funnel (any gold)

`pre_rank 0` equals `final rank 0` on every row below, as the writer requires.

```
[LME_S] n=470
  gold in corpus       470
  gold in pool@50      463
  gold at pre_rank 0   353
  gold at final rank 0 353
  gold in final top5   428
  gold at scene rank0  265
[LoCoMo] n=1536
  gold in corpus       1532   (every gold id is a turn id; 1535 have at least one)
  gold in pool@50      1415
  gold at pre_rank 0   808
  gold at final rank 0 808
  gold in final top5   1127
  gold at scene rank0  708
[BEAM-100K] n=355
  gold in corpus       355
  gold in pool@50      328
  gold at pre_rank 0   160
  gold at final rank 0 160
  gold in final top5   238
  gold at scene rank0  124
```

LME by category, same five lines (corpus / pool@50 / pre_rank 0 / final rank 0 / top 5), scene rank 0 last:

| Category | n | corpus | pool@50 | pre0 | final0 | top5 | scene0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi-session | 121 | 121 | 119 | 89 | 89 | 113 | 66 |
| temporal-reasoning | 127 | 127 | 124 | 93 | 93 | 111 | 67 |
| knowledge-update | 72 | 72 | 72 | 62 | 62 | 71 | 47 |
| single-session-user | 64 | 64 | 64 | 55 | 55 | 62 | 46 |
| single-session-assistant | 56 | 56 | 56 | 42 | 42 | 52 | 35 |
| single-session-preference | 30 | 30 | 28 | 12 | 12 | 19 | 4 |

LoCoMo: Multi-Hop 282 / 279 / 243 / 97 / 97 / 158 (scene 72). Temporal 321 / 320 / 299 / 191 / 191 / 248 (scene 182). Open-Domain 92 / 92 / 63 / 20 / 20 / 35 (scene 14). Single-Hop 841 / 841 / 810 / 500 / 500 / 686 (scene 440). The four corpus misses are evidence ids that are not turn ids; one question is missing every gold id (1535 any-gold), three more are missing one id of several (1532 all-golds).

## Gold turns not in the top 5

Stage of each gold turn. `in_top5` is shown so the complement is the miss.

| Bench / category | in top 5 | in k, not top 5 | in pool@50, not in k | t_scene > 0 below the 50 cut | absent from dump | session not a candidate (s_score = bridge = 0) |
|---|---:|---:|---:|---:|---:|---:|
| LME multi-session | 197 | 12 | 86 | 21 | 7 | 1 of the dumped turns |
| LME temporal-reasoning | 172 | 21 | 45 | 8 | 7 | 0 |
| LME knowledge-update | 130 | 6 | 8 | 0 | 0 | 0 |
| LME single-session-user | 64 | 1 | 1 | 0 | 0 | 0 |
| LME single-session-assistant | 52 | 2 | 2 | 0 | 0 | 0 |
| LME single-session-preference | 22 | 0 | 18 | 4 | 0 | 0 |
| LoCoMo Multi-Hop | 217 | 74 | 247 | 161 | 186 | |
| LoCoMo Temporal | 263 | 18 | 53 | 30 | 11 | |
| LoCoMo Open-Domain | 43 | 11 | 41 | 55 | 58 | |
| LoCoMo Single-Hop | 713 | 62 | 84 | 27 | 9 | |
| BEAM contradiction_resolution | 63 | 5 | 15 | 2 | 3 | |
| BEAM event_ordering | 18 | 13 | 58 | 23 | 118 | |
| BEAM information_extraction | 32 | 3 | 19 | 2 | 12 | |
| BEAM instruction_following | 19 | 3 | 20 | 4 | 6 | |
| BEAM knowledge_update | 62 | 5 | 14 | 0 | 6 | |
| BEAM multi_session_reasoning | 52 | 13 | 58 | 5 | 37 | |
| BEAM preference_following | 15 | 2 | 20 | 1 | 9 | |
| BEAM summarization | 21 | 18 | 76 | 1 | 167 | |
| BEAM temporal_reasoning | 55 | 5 | 14 | 0 | 0 | |

Questions whose gold session is outside the official session top 10: LME multi-session 10, temporal-reasoning 13, and 0 for the other four LME categories.

LME turns that are in pool@50 but not top 5, by geometry:

| Category | user opener | user later | assistant |
|---|---:|---:|---:|
| multi-session | 61 | 36 | 1 |
| temporal-reasoning | 49 | 17 | 0 |
| knowledge-update | 1 | 13 | 0 |
| single-session-preference | 4 | 14 | 0 |

The LME hole is not "never indexed." Haystack length matches `n_turns` on every LME row (corpus 470/470). One multi-session gold turn sits in a session the candidate filter drops. Seven multi-session and seven temporal golds are outside both the 50-list and the top-10 session dump, so their score is not in the file. Everything else that misses the top 5 was scored. Most of those misses are user session-openers already inside pool@50.

## Reading

All@5 on multi-session (53/121) and temporal-reasoning (74/127) fails because a second gold turn, usually the user turn at session position 0, is in the pool and not in the top 5. The top 5 is holding an assistant elaboration from that same session, or a later user turn, at a higher scene score. Session candidate generation is not the drop.
