# SCOREBOARD Step 37 — Kalevala parallelism

Baseline is the combined Step 31+32+33+34+35 tip `fd883f580d91cc7bb6e62ac5b4d8804d4e30f7a2` (binary sha256 `5cc9d0f9e0e43247e64cf46c17e0a230b855686ab28418ff949c4d4b64ca3b69`). After is `retrieval_bench` sha256 `1d7363188e0576173a5d6af36c4af5aa7db7b9863b019d40467979cd4e717b9f`. Deltas are LME Turn@1 only. Every other aggregate below is unchanged, and the per-question diff has zero losses.

| bench | metric | before | after |
|---|---|---:|---:|
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 354/428/437 | 355/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 808/1127/1226 | 808/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| LoCoMo | Multi-Hop Turn@1 / Session@1 | 97 / 156 | 97 / 156 |
| LoCoMo | Temporal Turn@1 / Session@1 | 191 / 242 | 191 / 242 |
| LoCoMo | Open-Domain Turn@1 / Session@1 | 20 / 36 | 20 / 36 |
| LoCoMo | Single-Hop Turn@1 / Session@1 | 500 / 699 | 500 / 699 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 160/238/267 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |
| BEAM-100K | Session All@5 / All@10 | 137 / 178 | 137 / 178 |

LME Turn@1 by type: temporal-reasoning 93→94. Every other LME type, every LoCoMo category, and every BEAM category is unchanged on Turn@1, Turn@5, and All@5.

Per-question: LME `gpt4_fa19884d` Turn@1 0→1. No other question changes on any recorded bit.
