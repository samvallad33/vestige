# Scoreboard — Step 32 trial balance

Baseline is the bd2693e2 tree (algos #44, `c06dbfc3770d17dee228291adcd1c136ce8f5cb7`), reproduced before the Step 32 edit. After is one `retrieval_bench` release run of `apply_c2_step32_trial_balance`. Deltas are Turn@1 only. Every other aggregate below is unchanged, and the per-question diff has zero losses.

## LME_S (n=470)

| | Session @1 | @5 | @10 | All@5 | Turn @1 | @5 | @10 | All@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 430 | 463 | 468 | 424 | 350 | 428 | 437 | 314 |
| after | 430 | 463 | 468 | 424 | 351 | 428 | 437 | 314 |

Turn@1 by type: multi-session 88 → 89. The other five types are unchanged.

## LoCoMo (n=1536)

| | Turn @1 | @5 | @10 | Session @1 | @5 |
|---|---:|---:|---:|---:|---:|
| before | 796 | 1127 | 1226 | 1133 | 1416 |
| after | 801 | 1127 | 1226 | 1133 | 1416 |

Turn@1 by category: Single-Hop 494 → 497, Temporal 187 → 188, Open-Domain 18 → 19, Multi-Hop 97 unchanged. Session@1 unchanged in every category.

## BEAM-100K (n=355)

| | Turn @1 | @5 | @10 | All@5 | All@10 | Session @1 | @5 | @10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 156 | 238 | 267 | 118 | 138 | 133 | 249 | 275 |
| after | 159 | 238 | 267 | 118 | 138 | 133 | 249 | 275 |

Turn@1 by category: temporal_reasoning 27 → 29, knowledge_update 30 → 31. The other seven categories are unchanged. All@5 is unchanged in every category.

## Gates

clippy `-D warnings` clean. `cargo check -p vestige-mcp` clean. `cargo test -p vestige-core` 560 passed. Hash-check 0/500.
