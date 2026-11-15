# Per-question diff vs the combined Step 31–36 tip

Score vectors:

- LME: `[session@1, session@5, session@10, turn@1, turn@5, turn@10, turn All@5]`
- LoCoMo: `[session@1, session@5, turn@1, turn@5, turn@10]`
- BEAM: `[turn@1, turn@5, turn@10, turn All@5, session@1, session@5, session All@5]`

Compared against `/tmp` runs of the same `c2_bench.py` driver on `b5d2e130` and on Step 38. LME 0 diffs / 470. BEAM 0 diffs / 355. LoCoMo 1 diff / 1536.

| Key | Before | After | What changed |
| --- | --- | --- | --- |
| LoCoMo `6:58` | `[0, 1, 0, 1, 1]` | `[0, 1, 1, 1, 1]` | Turn@1 0→1. Slot 1 `D26:10` → gold `D28:27`. |

No scored losses.

Score-neutral reorder (not a diff in the vectors above): LoCoMo `5:31` slot 1 `D25:5` → `D13:13`. Both non-gold. Vector stays `[0, 1, 0, 1, 1]`. Top-5 membership unchanged.

Listed flips that must stay hits, all still Turn@1 hits after Step 38:

- LME `e8a79c70`, `c4a1ceb8`, `ccb36322`, `gpt4_2ba83207`
- LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`, `0:97`, `3:174`, `9:86`
- BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`, `16:knowledge_update:1`
