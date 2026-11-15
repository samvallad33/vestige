# Pool-B scoreboard

Standing-rule seat on `perf/c2-pool-beam`, measured with `retrieval_bench` and `c2_bench.py` against the combined Step 31+32+33+34 tip `95ef2f228de94e6b4ab8d8c3566bac3847b5c3ca`.

Datasets:

- LongMemEval_S `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- LoCoMo `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K chats under `BEAM/chats/100K`

Winning binary sha256 `d1f19e876e25c07dd500ec77161a0303407da9b29b4b84d42ff4edd0f16e335e`.

## Headline

| Bench | Metric | combined tip | pool-B |
| --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 353/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 808/1127/1226 | 808/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 160/238/267 | 161/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |
| BEAM-100K | Session All@5 / All@10 | 137 / 178 | 137 / 178 |

BEAM Turn @1 is the only headline cell that moves, 160→161. Per-question losses versus the tip: 0 on every metric of all three benches. LME Session drops: 0.

## The flip

`1:instruction_following:0` Turn @1 0→1. The question is "Could you show me how to implement a login feature?" The seat moves the user opener "Always format all code snippets with syntax highlighting when I ask about implementation details." from inside the top 5 to slot 1. That opener's own lexical score is zero. `implement` and `implementation` differ by the suffix `ation`. Top-5 membership is unchanged, so Turn @5, Turn @10, and All@5 stay put. No other question changes on any bench.

## BEAM instruction_following

| | tip | pool-B |
| --- | --- | --- |
| Turn @1/@5/@10 | 6/18/21 | 7/18/21 |
| All@5 / All@10 | 13/16 | 13/16 |
| Session @1/@5/@10 | 4/16/17 | 4/16/17 |

Every other BEAM category, every LME type, and every LoCoMo category matches the tip.

## Gates

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean
- `cargo check -p vestige-mcp` finished
- `cargo test -p vestige-core` 564 passed, 0 failed
- hash-check ranked ids 0/500, full responses 0/500
