# Scoreboard — Pool C session-key opener

Baseline is the combined Step 31 + 32 + 33 + 34 tip (this branch before the Pool C commit). Binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`.

Winning commit on `perf/c2-pool-lme`, parent `57b4e023` (Step 34). Binary sha256 `58a129f4309549385bf176976a555842643dd2b635650bfc708aecc00d51cc15`.

Per-question losses versus that baseline: 0 on every stored bit (LME S@1/S@5/S@10/T@1/T@5/T@10/All@5, LoCoMo S@1/S@5/T@1/T@5/T@10, BEAM T@1/T@5/T@10/All@5/S@1/S@5/SAll@5). LoCoMo is byte-identical. Two questions flipped, both gains.

## Headline

| | Baseline 31–34 | Pool C | Delta |
|---|---:|---:|---:|
| LME Session @1/@5/@10 | 430/463/468 | 430/463/468 | 0 |
| LME Session All@5/All@10 | 424/447 | 424/447 | 0 |
| LME Turn @1/@5/@10 | 353/428/437 | 353/428/437 | 0 |
| LME Turn All@5 | 314 | 315 | +1 |
| LoCoMo Turn @1/@5/@10 | 808/1127/1226 | 808/1127/1226 | 0 |
| LoCoMo Session @1/@5 | 1133/1416 | 1133/1416 | 0 |
| BEAM Turn @1/@5/@10 | 160/238/267 | 160/238/267 | 0 |
| BEAM Turn All@5/All@10 | 118/138 | 119/139 | +1/+1 |
| BEAM Session @1/@5/@10 | 133/249/275 | 133/249/275 | 0 |
| BEAM Session All@5/All@10 | 137/178 | 137/178 | 0 |

## LME by category (baseline → Pool C)

| Category | n | S@1 | T@1 | T@5 | T@10 | All@5 |
|---|---:|---:|---:|---:|---:|---:|
| multi-session | 121 | 110 | 89 | 113 | 113 | 53 |
| temporal-reasoning | 127 | 114 | 93 | 111 | 116 | 74 → 75 |
| knowledge-update | 72 | 71 | 62 | 71 | 72 | 59 |
| single-session-user | 64 | 62 | 55 | 62 | 63 | 62 |
| single-session-assistant | 56 | 56 | 42 | 52 | 54 | 52 |
| single-session-preference | 30 | 17 | 12 | 19 | 19 | 14 |

Unmarked cells did not move.

## BEAM category that moved

knowledge_update All@5 24 → 25, All@10 27 → 28. Turn@1/@5/@10 and Session did not move. Every other BEAM category is unchanged on the stored bits.

## Questions that flipped

| Bench | Question | Bit |
|---|---|---|
| LME temporal-reasoning | `gpt4_fe651585` | Turn All@5 0 → 1 |
| BEAM knowledge_update | `20:knowledge_update:1` | All@5 0 → 1 |

`gpt4_fe651585` ("Who became a parent first, Rachel or Alex?"): both golds were already in pool@50. Slot 1 stayed `answer_65600ff6_1#8`. The session-2 user opener `answer_65600ff6_2#0` took the seat of that session's assistant elaboration `answer_65600ff6_2#9`.
