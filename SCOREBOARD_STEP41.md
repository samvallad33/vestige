# SCOREBOARD Step 41 — Queried-speaker reply seat

Baseline is the combined pack-43 tip (Steps 31–40, Pools C/B/A, W2-001), replayed with `git am -3` from `d121d063` onto `perf/c2-step41-queried-speaker-seat`. Replayed tip before this step: `1a0794b49e67e020f4540d7b727ce11e141d6097`. The pack's named Computer 2 tip `19a8e30aa1226706e7fffab863bf327149bb3269` was rewritten by `git am` and is not an object here. Baseline `retrieval_bench` sha256 `911ffb8951c0db69903dc060c340f8e1702dafc7133aafce59537d1464beaede`. After sha256 `8fea3e4702c388c5bedaadba1b9a1b8e2c5e86d4fcbb8b34bdca8892da6ff28d`.

Four LoCoMo questions change, all Turn@1 0→1. Every other aggregate below is unchanged. Zero losses.

| bench | metric | before | after |
|---|---|---:|---:|
| LME_S n=470 | Session @1/@5/@10 | 432/463/468 | 432/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 355/429/438 | 355/429/438 |
| LME_S | Turn All@5 | 320 | 320 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 813/1136/1229 | **817**/1136/1229 |
| LoCoMo | Session @1/@5 | 1138/1416 | 1138/1416 |
| LoCoMo | Multi-Hop Turn@1 / Session@1 | 97 / 156 | 97 / 156 |
| LoCoMo | Temporal Turn@1 / Session@1 | 192 / 243 | **195** / 243 |
| LoCoMo | Open-Domain Turn@1 / Session@1 | 20 / 36 | 20 / 36 |
| LoCoMo | Single-Hop Turn@1 / Session@1 | 504 / 703 | **505** / 703 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 | 162/240/269 |
| BEAM-100K | Turn All@5 / All@10 | 122 / 142 | 122 / 142 |
| BEAM-100K | Session @1/@5/@10 | 140/249/275 | 140/249/275 |
| BEAM-100K | Session All@5 / All@10 | 137 / 178 | 137 / 178 |

LME Turn@1 by type, unchanged: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 94/127, knowledge-update 62/72, single-session-assistant 42/56.

BEAM Turn@1, unchanged: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 7, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

Per-question: LoCoMo `0:68`, `6:23`, `7:3`, `7:49` Turn@1 0→1. LME 0 diffs / 470. BEAM 0 diffs / 355. See `STEP41_PERQ_DIFF.md`.
