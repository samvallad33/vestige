# C2 pool-B log

Branch `perf/c2-pool-beam`. Parent of the win is `95ef2f228de94e6b4ab8d8c3566bac3847b5c3ca` (Step 34 Bertillonage on top of Steps 31–33 and the 44 algo patches). Baseline binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`. Measured binary for the win sha256 `d1f19e876e25c07dd500ec77161a0303407da9b29b4b84d42ff4edd0f16e335e`.

Baseline funnel (BEAM dump, any gold): pool@50 329, scene-rank0 125, final @1 159, final top 5 240, final top 10 267. Dump `rank_pre == 0` is the seated head, so pre-rank @1 equals final @1 by construction. Official no-dump Turn @1 is 160. The dump and the no-dump bench disagree on Turn any-hit for 16/355 BEAM questions; the scoreboard is the no-dump bench. Census: `POOL_MISS_CENSUS_B.md`.

## Attempt 1 — standing-rule winner, zero own score, no inflection gate

Not committed.

Move `rank_standing_rules` head to slot 1 when it is already in the top 5, is a user session opener, `t_own == 0`, the sitting head is a different user question, and the query is task-under-rules and not preference, summary, temporal, event-sequence, multi-hop, a when/where/why/count sniper, current-state, or assistant-attribution.

BEAM no-dump: Turn @1 stayed 160. One gain, one loss.

- Gain: `1:instruction_following:0` Turn @1 0→1. The channel winner is the opener "Always format all code snippets with syntax highlighting when I ask about implementation details."
- Loss: `10:knowledge_update:1` Turn @1 1→0. The sitting head is the gold question that states the April 20 deadline. The channel winner is a zero-specificity editing-progress rule. The channel keeps every task-under-rules policy when none of them hit a focal id, then sorts by timestamp.

LoCoMo and LME were not run. A loss on BEAM is enough to drop the predicate.

## Attempt 2 — explicit conditional plus an inflection of a stored content term

This commit. `apply_c2_pool_b_standing_seat` in `crates/vestige-core/src/algos_c2.rs` (function at 3827, inflection helper `c2_policy_inflects_query` at 3916, unit test `pool_b_standing_seat_promotes_zero_own_opener_on_task_question` at 7703). Call site `crates/vestige-core/src/retrieval.rs` 6287–6298, after Step 34 and before `maybe_dump_replay`.

Same gates as attempt 1, plus: the winner has an explicit conditional scope, and one of its stored content terms (`TurnIx.seq`, not the fan-cut scope set and not the turn body) differs from a query focal term by only a closed inflectional suffix. `implement` meets `implementation`. `date` does not meet `deadline`. Top-5 membership does not change. `rsb_seat_behind` is not touched. The session list is not rewritten.

An exact focal-id intersection against `StandingRule.scope` was a no-op on the real corpus. The engine stemmer leaves `implement` and `implementation` as `implement` and `implementat`, and the scope set is fan-cut, so the conditional word is often absent from `scope` even though `has_explicit_scope` is set. The content-term inflection is what still sees that pair.

### Scoreboard versus `95ef2f22`

| Bench | Metric | before | after |
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

BEAM instruction_following Turn @1 6→7. Every other category cell is unchanged. LME by type is unchanged. LoCoMo by category is unchanged.

Gains: `1:instruction_following:0` Turn @1 0→1. Losses: none, on every metric of all three benches. LME Session drops: 0.

Gates: `cargo clippy -p vestige-core --all-targets -- -D warnings` clean. `cargo check -p vestige-mcp` finished. `cargo test -p vestige-core` 564 passed, 0 failed. Hash-check ranked ids 0/500, full f32 responses 0/500.
