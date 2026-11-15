# Step 34 Bertillonage scoreboard

Measured with `retrieval_bench` and `tools/c2_bench.py` on the Computer 2 files:

- LongMemEval_S `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- LoCoMo `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K chats under `BEAM/chats/100K`

`bd2693e2` below is the algo-44 tree (`b1e4509e`) produced by applying `steps7_10a` then `algos44` onto `d121d063`. Its scores match the published bd2693e2 board. The combined tip is `3345f79f` (Step 31 Exchequer + Step 32 trial balance). Step 34 is the commit on top of that tip.

## Headline

| Bench | Metric | bd2693e2 | Step 31+32 tip | Step 34 |
| --- | --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 | 424 | 424 | 424 |
| LME_S | Turn @1/@5/@10 | 350/428/437 | 352/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 796/1127/1226 | 801/1127/1226 | 805/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 156/238/267 | 159/238/267 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 | 133/249/275 |

Step 34 versus the combined tip: LME Turn @1 352→353, LoCoMo Turn @1 801→805, BEAM Turn @1 159→160. Every other headline cell is unchanged. Per-question losses versus the tip: 0 on every metric of all three benches. LME Session drops: 0.

## LME by type (Turn @1)

| Type | bd2693e2 | tip | Step 34 |
| --- | --- | --- | --- |
| single-session-user | 54 | 54 | 55 |
| multi-session | 88 | 89 | 89 |
| single-session-preference | 12 | 12 | 12 |
| temporal-reasoning | 93 | 93 | 93 |
| knowledge-update | 62 | 62 | 62 |
| single-session-assistant | 41 | 42 | 42 |

## LoCoMo by category (Turn @1)

| Category | bd2693e2 | tip | Step 34 |
| --- | --- | --- | --- |
| Multi-Hop | 97 | 97 | 97 |
| Temporal | 187 | 188 | 190 |
| Open-Domain | 18 | 19 | 19 |
| Single-Hop | 494 | 497 | 499 |

## BEAM by category (Turn @1)

| Category | bd2693e2 | tip | Step 34 |
| --- | --- | --- | --- |
| contradiction_resolution | 32 | 32 | 32 |
| event_ordering | 7 | 7 | 7 |
| information_extraction | 20 | 20 | 20 |
| instruction_following | 6 | 6 | 6 |
| knowledge_update | 30 | 31 | 31 |
| multi_session_reasoning | 21 | 21 | 22 |
| preference_following | 7 | 7 | 7 |
| summarization | 6 | 6 | 6 |
| temporal_reasoning | 27 | 29 | 29 |

## Questions Step 34 flipped to Turn @1

All six were top-5-not-#1 on the tip. None of them lost any other metric.

- LME `ccb36322` (single-session-user)
- LoCoMo `1:42` (Single-Hop), `5:105` (Single-Hop), `9:9` (Temporal), `9:36` (Temporal)
- BEAM `19:multi_session_reasoning:0`

Step 31 `e8a79c70` and Step 32 `c4a1ceb8`, LoCoMo `2:106` `3:143` `5:13` `6:25` `9:147`, and BEAM `9:temporal_reasoning:1` `11:knowledge_update:0` `20:temporal_reasoning:1` are still Turn @1 hits.
