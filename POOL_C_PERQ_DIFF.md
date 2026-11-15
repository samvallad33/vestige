# Per-question diff — Pool C versus the Step 31+32+33+34 tip

Stored bits only. A loss is a 1 that became 0. There are none, on this tip and against attempt 2 (`82c45088`).

LME bits: S@1, S@5, S@10, T@1, T@5, T@10, Turn All@5.
LoCoMo bits: S@1, S@5, T@1, T@5, T@10.
BEAM bits: T@1, T@5, T@10, All@5, S@1, S@5, Session All@5.

470 + 1,536 + 355 questions compared. Twelve gains. LoCoMo: no question moved. Losses versus attempt 3 (`ca63ca03`): 0.

| Bench | Question | Type | Before | After | Delta |
|---|---|---|---|---|---|
| LME | `2b8f3739` | multi-session | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| LME | `d3ab962e` | multi-session | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| LME | `099778bb` | multi-session | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| LME | `gpt4_1e4a8aec` | temporal-reasoning | `[1,1,1,0,0,0,0]` | `[1,1,1,0,1,1,1]` | Turn@5, Turn@10, All@5 |
| LME | `dcfa8644` | temporal-reasoning | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| LME | `gpt4_fe651585` | temporal-reasoning | `[1,1,1,1,1,1,0]` | `[1,1,1,1,1,1,1]` | Turn All@5 |
| BEAM | `6:temporal_reasoning:0` | temporal_reasoning | `[0,1,1,0,0,1,1]` | `[0,1,1,1,0,1,1]` | All@5 |
| BEAM | `11:knowledge_update:0` | knowledge_update | `[1,1,1,0,1,1,1]` | `[1,1,1,1,1,1,1]` | All@5 |
| BEAM | `15:multi_session_reasoning:0` | multi_session_reasoning | `[0,0,0,0,0,0,0]` | `[0,1,1,0,0,0,0]` | Turn@5, Turn@10 |
| BEAM | `19:multi_session_reasoning:0` | multi_session_reasoning | `[1,1,1,0,0,1,1]` | `[1,1,1,1,0,1,1]` | All@5 |
| BEAM | `15:temporal_reasoning:1` | temporal_reasoning | `[0,0,0,0,0,0,0]` | `[0,1,1,0,0,0,0]` | Turn@5, Turn@10 |
| BEAM | `20:knowledge_update:1` | knowledge_update | `[1,1,1,0,1,1,1]` | `[1,1,1,1,1,1,1]` | All@5 |

`ac031881` (single-session-assistant) stays `[1,1,1,0,1,1,1]`.

BEAM Turn All@10 is not in the per-question vector. The aggregate moved 138 → 142. `6:temporal_reasoning:0`, `11:knowledge_update:0`, and `20:knowledge_update:1` account for three. `19:multi_session_reasoning:0` is the fourth: its stored All@5 bit is the one that flipped, and All@10 moved with it (7 → 8 on multi_session_reasoning).
