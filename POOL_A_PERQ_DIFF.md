# Per-question diff: buried lexical peak versus the Step 31+32+33+34 tip

Vectors are the `c2_bench.py` per-question rows.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TA@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

Losses on any coordinate, all three benches: **0**.

LME coordinate changes: 0. BEAM coordinate changes: 0. LME session-coordinate drops: 0.

## Gains

| Bench | Question | Tip | After |
| --- | --- | --- | --- |
| LoCoMo | `9:137` | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |

Single-Hop Turn@5 686→687, Turn@10 737→738. Turn@1 and both session metrics stay put on that row.
