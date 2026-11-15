# Per-question diff: Step 36 versus the Step 31+32+33+34 tip

Baseline file: combined tip `d4e61d94`.
Candidate: Step 36 Antaranga-bahiranga, binary sha256 `dd223baa27ac55950c3d9971203a1786dd32c4323a1b3b82c3434faf967eac9d`.

Vectors are the `c2_bench.py` per-question rows.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TA@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

Losses on any coordinate, all three benches: **0**.

Bit counts: LME +0/−0 (470 questions). LoCoMo +3/−0 (1536). BEAM +1/−0 (355). Every gain is Turn@1 only.

## Gains

| Bench | Question | Tip | Step 36 |
| --- | --- | --- | --- |
| LoCoMo | `0:97` | `[0, 1, 0, 1, 1]` | `[0, 1, 1, 1, 1]` |
| LoCoMo | `3:174` | `[0, 1, 0, 1, 1]` | `[0, 1, 1, 1, 1]` |
| LoCoMo | `9:86` | `[0, 1, 0, 1, 1]` | `[0, 1, 1, 1, 1]` |
| BEAM | `16:knowledge_update:1` | `[0, 1, 1, 1, 0, 1, 1]` | `[1, 1, 1, 1, 0, 1, 1]` |

## Prior wins still hitting (Turn@1 = 1)

LME `e8a79c70`, `c4a1ceb8`, `ccb36322`.

LoCoMo `2:106`, `3:143`, `5:13`, `6:25`, `9:147`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`.

BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`.
