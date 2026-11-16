# SCOREBOARD Step 42

Combined pack-43 tip versus W2-006 witness-margin session swap. Counts are from `retrieval_bench` (release) with the replay dump off for the ranked response. The tip was measured in the same harness before the ranker edit. Turns were not rewritten.

Pack-43 replay tip before this commit: `31c17e6a189b81a7d11ffbf4005758f25aceebb7`. Computer 2's recorded tip `19a8e30aa1226706e7fffab863bf327149bb3269` is not this clone's SHA: Steps 39 and 40 conflicted at the hook and were resolved in order. The measured cells match the composed Steps 31–40 + pools + W2-001 figures (LME Session@1 432, LoCoMo Turn@1 813 / Turn@5 1136 / Session@1 1138, BEAM Session@1 140).

| Bench | Metric | Before | After | Delta |
| --- | --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 432/463/468 | 434/463/468 | @1 +2 |
| LME_S | Session All@5/All@10 | 424/447 | 424/447 | 0 |
| LME_S | Turn @1/@5/@10 | 355/429/438 | 355/429/438 | 0 |
| LME_S | Turn All@5/All@10 | 320/346 | 320/346 | 0 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 813/1136/1229 | 813/1136/1229 | 0 |
| LoCoMo | Session @1/@5/@10 | 1138/1416/1488 | 1138/1416/1488 | 0 |
| LoCoMo | Session All@5/All@10 | 1246/1361 | 1246/1361 | 0 |
| LoCoMo | Turn All@5/All@10 | 964/1046 | 964/1046 | 0 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 | 162/240/269 | 0 |
| BEAM-100K | Turn All@5/All@10 | 122/142 | 122/142 | 0 |
| BEAM-100K | Session @1/@5/@10 | 140/249/275 | 148/249/275 | @1 +8 |
| BEAM-100K | Session All@5/All@10 | 137/178 | 137/178 | 0 |

Per-question: 10 Session@1 gains, 0 losses, 0 other bit flips. LoCoMo 1536/1536 unchanged. See `PERQ_DIFF_STEP42.md`.

LME Session@1 by type: multi-session 110→111, temporal-reasoning 114→115. knowledge-update stays 71/72. single-session-assistant 56, single-session-preference 18, single-session-user 63.

BEAM Session@1 by category: summarization 7→8, contradiction_resolution 29→30, temporal_reasoning 24→27, knowledge_update 27→28, instruction_following 4→5, preference_following 4→5. event_ordering 7, information_extraction 18, multi_session_reasoning 20 unchanged.

90% need, not met by this step: LME Turn@1 423, LME Turn All@5 423, LoCoMo Turn@1 1383, LoCoMo Session@1 1383, BEAM Turn@1 320, BEAM Turn All@5 320 (structural max 305), BEAM Session@1 320 (top-10 admission cap 275; this step only reorders inside the top 3).
