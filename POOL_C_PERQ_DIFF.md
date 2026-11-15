# Per-question diff — Pool C versus the Step 31+32+33+34 tip

Stored bits only. A loss is a 1 that became 0. There are none.

LME bits: S@1, S@5, S@10, T@1, T@5, T@10, Turn All@5.
LoCoMo bits: S@1, S@5, T@1, T@5, T@10.
BEAM bits: T@1, T@5, T@10, All@5, S@1, S@5, Session All@5.

470 + 1,536 + 355 questions compared. Two gains.

| Bench | Question | Type | Before | After | Delta |
|---|---|---|---|---|---|
| LME | `gpt4_fe651585` | temporal-reasoning | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| BEAM | `20:knowledge_update:1` | knowledge_update | `[1,1,1,0,1,1,1]` | `[1,1,1,1,1,1,1]` | All@5 |

BEAM Turn All@10 is not in the per-question vector. The aggregate moved 138 → 139 with the same question.
