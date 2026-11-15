# Scoreboard — Step 36 Antaranga-bahiranga

Baseline is the combined Step 31+32+33+34 tip `d4e61d94` (binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`). After is `retrieval_bench` sha256 `dd223baa27ac55950c3d9971203a1786dd32c4323a1b3b82c3434faf967eac9d`.

`bd2693e2` is the algo-44 tree `5c246f3a` benched in this same run. Its aggregates match the published bd2693e2 board. Per-question losses of the tip against that bench: 0. Per-question losses of Step 36 against the tip: 0.

| Bench | Metric | bd2693e2 | Tip 31+32+33+34 | Step 36 |
| --- | --- | --- | --- | --- |
| LME_S | Session @1/@5/@10 | 430/463/468 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 350/428/437 | 353/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 | 314 |
| LoCoMo | Turn @1/@5/@10 | 796/1127/1226 | 808/1127/1226 | **811**/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 | 1133/1416 |
| BEAM-100K | Turn @1/@5/@10 | 156/238/267 | 160/238/267 | **161**/238/267 |
| BEAM-100K | All@5 / All@10 | 118 / 138 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 | 133/249/275 |

n = 470 / 1536 / 355.

## LME by type (Session @1, Turn @1/@5)

| Type | bd2693e2 | tip | Step 36 |
| --- | --- | --- | --- |
| knowledge-update | 71, 62/71 | 71, 62/71 | 71, 62/71 |
| multi-session | 110, 88/113 | 110, 89/113 | 110, 89/113 |
| single-session-assistant | 56, 41/52 | 56, 42/52 | 56, 42/52 |
| single-session-preference | 17, 12/19 | 17, 12/19 | 17, 12/19 |
| single-session-user | 62, 54/62 | 62, 55/62 | 62, 55/62 |
| temporal-reasoning | 114, 93/111 | 114, 93/111 | 114, 93/111 |

## LoCoMo by category (Turn @1/@5, Session @1)

| Category | bd2693e2 | tip | Step 36 |
| --- | --- | --- | --- |
| Multi-Hop | 97/158, 156 | 97/158, 156 | 97/158, 156 |
| Temporal | 187/248, 242 | 191/248, 242 | 191/248, 242 |
| Open-Domain | 18/35, 36 | 20/35, 36 | 20/35, 36 |
| Single-Hop | 494/686, 699 | 500/686, 699 | **503**/686, 699 |

## BEAM by category (Turn @1/@5, All@5)

| Category | bd2693e2 | tip | Step 36 |
| --- | --- | --- | --- |
| contradiction_resolution | 32/40, 22 | 32/40, 22 | 32/40, 22 |
| event_ordering | 7/14, 0 | 7/14, 0 | 7/14, 0 |
| information_extraction | 20/27, 20 | 20/27, 20 | 20/27, 20 |
| instruction_following | 6/18, 13 | 6/18, 13 | 6/18, 13 |
| knowledge_update | 30/37, 24 | 31/37, 24 | **32**/37, 24 |
| multi_session_reasoning | 21/34, 4 | 22/34, 4 | 22/34, 4 |
| preference_following | 7/14, 11 | 7/14, 11 | 7/14, 11 |
| summarization | 6/17, 0 | 6/17, 0 | 6/17, 0 |
| temporal_reasoning | 27/37, 24 | 29/37, 24 | 29/37, 24 |

## Flips vs the combined tip

Turn@1 only. Losses: 0.

- LoCoMo `0:97`, `3:174`, `9:86`
- BEAM `16:knowledge_update:1`

Steps 31–34 gains are still Turn@1 hits: LME `e8a79c70`, `c4a1ceb8`, `ccb36322`; LoCoMo `2:106`, `3:143`, `5:13`, `6:25`, `9:147`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`.
