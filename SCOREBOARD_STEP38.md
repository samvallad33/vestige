# Scoreboard — Step 38 Hermeneutic circle

Baseline is the combined Step 31+32+33+34+35+36 tip (`b5d2e130`), measured in one binary before this step. After is Step 38 on that tip.

| Bench | Metric | Before | After |
| --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5/All@10 | 424/447 | 424/447 |
| LME_S | Turn @1/@5/@10 | 354/428/437 | 354/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 811/1127/1226 | **812**/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| LoCoMo Multi-Hop | Turn@1 / Session@1 | 97 / 156 | 97 / 156 |
| LoCoMo Temporal | Turn@1 / Session@1 | 191 / 242 | **192** / 242 |
| LoCoMo Open-Domain | Turn@1 / Session@1 | 20 / 36 | 20 / 36 |
| LoCoMo Single-Hop | Turn@1 / Session@1 | 503 / 699 | 503 / 699 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 161/238/267 | 161/238/267 |
| BEAM-100K | Turn All@5/All@10 | 118/138 | 118/138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |

BEAM Turn@1 by category, unchanged: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 6, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

Per-question score diff vs the combined tip: LoCoMo `6:58` Turn@1 0→1. No other scored bit changed on 470 + 1536 + 355 questions. Detail: `STEP38_PERQ_DIFF.md`.
