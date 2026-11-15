# Scoreboard — Pool C session-key opener

Baseline is the combined Step 31 + 32 + 33 + 34 tip. Binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`.

This board is attempt 3, stacked on attempt 2 (`82c45088c489cc72399c3031fa39896bb266964b`). The top 5 must already span two sessions, and a staying turn may attest a query group the opener does not. Binary sha256 `4c764e0ed91d1ba6a37a48c50d366f0723c26d7eb23412767ca52d8758df9b05`.

Per-question losses versus the Step 31–34 tip: 0 on every stored bit. LoCoMo is byte-identical. Eleven questions flipped, all gains. Losses versus attempt 2: 0.

## Headline

| | Baseline 31–34 | Pool C | Delta |
|---|---:|---:|---:|
| LME Session @1/@5/@10 | 430/463/468 | 430/463/468 | 0 |
| LME Session All@5/All@10 | 424/447 | 424/447 | 0 |
| LME Turn @1/@5/@10 | 353/428/437 | 353/429/438 | 0 / +1 / +1 |
| LME Turn All@5 | 314 | 320 | +6 |
| LoCoMo Turn @1/@5/@10 | 808/1127/1226 | 808/1127/1226 | 0 |
| LoCoMo Session @1/@5 | 1133/1416 | 1133/1416 | 0 |
| BEAM Turn @1/@5/@10 | 160/238/267 | 160/240/269 | 0 / +2 / +2 |
| BEAM Turn All@5/All@10 | 118/138 | 121/141 | +3 / +3 |
| BEAM Session @1/@5/@10 | 133/249/275 | 133/249/275 | 0 |
| BEAM Session All@5/All@10 | 137/178 | 137/178 | 0 |

## LME by category (baseline → Pool C)

| Category | n | S@1 | T@1 | T@5 | T@10 | All@5 |
|---|---:|---:|---:|---:|---:|---:|
| multi-session | 121 | 110 | 89 | 113 | 113 | 53 → 56 |
| temporal-reasoning | 127 | 114 | 93 | 111 → 112 | 116 → 117 | 74 → 77 |
| knowledge-update | 72 | 71 | 62 | 71 | 72 | 59 |
| single-session-user | 64 | 62 | 55 | 62 | 63 | 62 |
| single-session-assistant | 56 | 56 | 42 | 52 | 54 | 52 |
| single-session-preference | 30 | 17 | 12 | 19 | 19 | 14 |

Unmarked cells did not move. Single-session-assistant Turn@5 stays 52: `ac031881` is one session, and the seat does not run.

## BEAM categories that moved

| Category | What moved |
|---|---|
| knowledge_update | All@5 24 → 26, All@10 27 → 29 |
| multi_session_reasoning | Turn@5 34 → 35, Turn@10 36 → 37 |
| temporal_reasoning | Turn@5 37 → 38, Turn@10 39 → 40, All@5 24 → 25, All@10 27 → 28 |

Turn@1 and Session did not move in any BEAM category.

## Questions that flipped

| Bench | Question | Bit |
|---|---|---|
| LME multi-session | `2b8f3739` | Turn All@5 0 → 1 |
| LME multi-session | `d3ab962e` | Turn All@5 0 → 1 |
| LME multi-session | `099778bb` | Turn All@5 0 → 1 |
| LME temporal-reasoning | `gpt4_1e4a8aec` | Turn@5, Turn@10, All@5 0 → 1 |
| LME temporal-reasoning | `dcfa8644` | Turn All@5 0 → 1 |
| LME temporal-reasoning | `gpt4_fe651585` | Turn All@5 0 → 1 |
| BEAM temporal_reasoning | `6:temporal_reasoning:0` | All@5 0 → 1 |
| BEAM knowledge_update | `11:knowledge_update:0` | All@5 0 → 1 |
| BEAM multi_session_reasoning | `15:multi_session_reasoning:0` | Turn@5, Turn@10 0 → 1 |
| BEAM temporal_reasoning | `15:temporal_reasoning:1` | Turn@5, Turn@10 0 → 1 |
| BEAM knowledge_update | `20:knowledge_update:1` | All@5 0 → 1 |

`2b8f3739`: both of the other gold openers were already in the top 5. Session-3 assistant `answer_23759615_3#7` yielded to session-3 opener `answer_23759615_3#0`.

`gpt4_1e4a8aec`: the gold opener `answer_16bd5ea6_2#0` was outside the top 5. It took the seat of assistant `answer_16bd5ea6_2#5`. Slot 1 stayed.
