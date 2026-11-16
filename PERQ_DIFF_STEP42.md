# Per-question diff, Step 42 vs combined pack-43 tip

Tip `31c17e6a189b81a7d11ffbf4005758f25aceebb7`. After is W2-006.

Bit order: `[S@1, S@5, S@10, S All@5, S All@10, T@1, T@5, T@10, T All@5, T All@10]`.

## Gains

| Bench | Question | Type | Before | After | Flip |
| --- | --- | --- | --- | --- | --- |
| BEAM-100K | `1:summarization:1` | summarization | `[0,1,1,0,0,1,1,1,0,0]` | `[1,1,1,0,0,1,1,1,0,0]` | Session@1 |
| BEAM-100K | `4:contradiction_resolution:0` | contradiction_resolution | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| BEAM-100K | `8:temporal_reasoning:0` | temporal_reasoning | `[0,1,1,0,0,1,1,1,0,0]` | `[1,1,1,0,0,1,1,1,0,0]` | Session@1 |
| BEAM-100K | `10:knowledge_update:1` | knowledge_update | `[0,1,1,0,0,1,1,1,0,0]` | `[1,1,1,0,0,1,1,1,0,0]` | Session@1 |
| BEAM-100K | `10:temporal_reasoning:1` | temporal_reasoning | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| BEAM-100K | `11:instruction_following:1` | instruction_following | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| BEAM-100K | `16:preference_following:1` | preference_following | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| BEAM-100K | `17:temporal_reasoning:0` | temporal_reasoning | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| LME_S | `b5ef892d` | multi-session | `[0,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` | Session@1 |
| LME_S | `gpt4_d6585ce9` | temporal-reasoning | `[0,1,1,0,1,0,0,0,0,0]` | `[1,1,1,0,1,0,0,0,0,0]` | Session@1 |

## Losses

None. Every bit that was 1 on the tip is still 1, on all 470 LME_S, 1536 LoCoMo, and 355 BEAM-100K questions. LoCoMo 0/1536 changed. Turn ids did not change on any question (the hook does not write the turn list). Listed Steps 31–40, pool, and W2-001 flips therefore remain hits.
