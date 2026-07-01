# Tool Consolidation v2.2.0

> Reduce the Vestige MCP tool surface so an agent can reliably pick the right
> tool, then make the few always-on tools deterministic. Two layers: Layer 1
> (this release) collapses 34 advertised tools to 12; Layer 2 (follow-up) shrinks
> the *default* surface and enforces the memory loop with hooks.
>
> Note: v2.2.0 also ships one **net-new** tool from a separate value stream — the
> flagship `backfill` (Retroactive Salience, Cai 2024 *Nature*). It is not part of
> this fold, so the shipped advertised surface is **13** (12 from consolidation +
> `backfill`), as asserted by `test_tools_list_returns_all_tools`.

## Why (frontier evidence)

More advertised tools actively degrade tool selection — the 30 tools an agent
ignores make the 5 it uses harder to choose:

- **RAG-MCP** (arXiv 2505.03275): selection accuracy collapses 43% → 14% when the
  full tool catalog is dumped into context; stays >90% under ~30 tools.
- **Anthropic tool-deferral**: deferring tool schemas moved Opus 4 from 49% → 74%
  on a tool-heavy benchmark.
- **GitHub Copilot**: 40 → 13 tools gave +2–5pp accuracy and −400ms latency.
- **OpenAI** guidance: aim for <20 functions visible at the start of a turn.
- **RoTBench** (2401.08326): tool *names* are load-bearing — renaming drops GPT-4
  80 → 58. So renames are deliberate and every old name keeps working.

Vestige had **34** advertised tools. This is the correction.

## Layer 1 — Count reduction (THIS RELEASE): 34 → 12 (fold result)

Principle: **one consolidation per commit, one change per submission.** Each
consolidation is its own commit, landed in a safe order with the hot retrieval
path touched last. Every old tool name remains a hidden `warn!` + redirect alias
for at least one minor release (so existing `.mcp.json` configs, hooks, and agent
habits keep working) and is removed in **v2.3.0**.

### Safe order (as committed)

| # | Commit | Folds | Into | Count |
|---|--------|-------|------|------:|
| 1 | `dedup` | find_duplicates + merge_candidates + plan_merge + plan_supersede + apply_plan + merge_undo + protect + merge_policy (8) | `dedup` | 34 → 27 |
| 2 | `session_start` | session_context (rename) | `session_start` | 27 |
| 3a | `memory_status` | system_status + memory_health + memory_timeline + memory_changelog (4) | `memory_status` | 27 → 24 |
| 3b | `graph` | explore_connections + predict + memory_graph + composed_graph (4) | `graph` | 24 → 21 |
| 4 | `maintain` | consolidate + dream + gc + importance_score + backup + export + restore (7) | `maintain` | 21 → 15 |
| 5 | `recall` | search + deep_reference + cross_reference + contradictions (4) | `recall` | 15 → 12 |

`recall` is committed **last** because it is the hot path.

### Final advertised surface (13)

The fold itself lands at 12; the release adds the net-new flagship `backfill`,
for **13 advertised**.

| Standalone (7) | Consolidated (6) |
|---|---|
| `smart_ingest` | `recall` |
| `memory` | `dedup` |
| `codebase` | `memory_status` |
| `intention` | `graph` |
| `source_sync` | `maintain` |
| `suppress` | `session_start` |
| `backfill` | |

### Action / mode / view maps

- **`recall`** — `mode`: `lookup` (default) · `reason` · `contradictions`
- **`dedup`** — `action`: `scan` (default) · `plan_merge` · `plan_supersede` · `apply` · `undo` · `protect` · `policy`
- **`memory_status`** — `view`: `health` (default) · `retention` · `timeline` · `changelog`
- **`graph`** — `action`: `chain` · `associations` · `bridges` · `predict` · `memory_graph` · `recent` · `get` · `memory` · `neighbors` · `never_composed` · `bounty_mode` · `label`
- **`maintain`** — `action`: `consolidate` · `dream` · `gc` · `importance_score` · `backup` · `export` · `restore`

### Resolved design decisions

- **`search` is folded, not kept standalone.** `recall` with no `mode` (the
  default) *is* search — a zero-overhead pass-through to `search_unified`. Keeping
  both `search` and `recall` advertised would be the exact RAG-MCP anti-pattern.
  The fold lands at a clean **12**; adding the flagship `backfill` brings the
  shipped surface to **13**, still well inside the ~30-tool safe zone and leaving
  headroom toward a future always-on `save` surface.
- **`graph` actions are flat peers, not nested.** `explore`'s `chain` /
  `associations` / `bridges` sit alongside `predict` / `memory_graph` /
  `composed_graph` actions in a single `action` enum — matching the existing
  `memory` / `codebase` flat-action convention and avoiding a translation layer.

### Invariants preserved (with the test that proves each)

- **bitemporal-never-delete** (`dedup`): plan → apply → undo, confirm-gating, and
  invalidation-not-deletion delegate to `merge::execute` verbatim.
- **`system_status` response shape** (`memory_status` view=`health`): byte-for-byte
  — `test_default_view_is_health`.
- **`gc` dry-run default** + **`restore` path-confinement** (`maintain`):
  `test_maintain_actions_and_safety`.
- **`recall` lookup = search, no reasoning cost** (hot path):
  `test_recall_lookup_matches_search_shape`.
- **Dashboard events** (consolidate/dream/importance_score Started + Completed,
  SearchPerformed): preserved by re-emitting in the new dispatch arms and by
  `emit_tool_event` normalizing the unified tool name to its effective sub-action.

### Result-size annotations (moved with their tools)

`memory_timeline` (200k) → `memory_status`; `search` (300k) → `recall`; new
`dedup` 150k and `graph` 250k. Kept in sync across the annotation loop, the
`expected_max_result_size` helper, and both annotation guard tests.

### Deprecation timeline

Aliases `warn!` in v2.2.x and are hard-removed in **v2.3.0**. Full alias list (31
names) lives in the dispatch redirects in `crates/vestige-mcp/src/server.rs`.

## Layer 2 — Default-surface + hooks (FOLLOW-UP, NOT in v2.2.0)

Count reduction is necessary but not sufficient: what matters most is how few
tools are visible *at the start of a turn*, plus making the memory loop fire
deterministically instead of hoping the model remembers.

- **Tiny always-on surface (~3)**: `recall` @ session start, `save` (=`smart_ingest`)
  @ session end, `recall` on-demand for facts. Everything else (`dedup`, `graph`,
  `maintain`, `memory_status`, …) deferred off the default surface, loaded on
  demand.
- **Deterministic hooks**: a `SessionStart` hook fires `recall`; a `Stop` hook
  fires `save` (async, fire-and-forget — synchronous heavy work in `Stop` causes
  loops + per-turn lag). "If the model fails to save, it's gone" — move save out
  of the model hot loop.
- This is what turns 12-advertised into ~3-default. Status: **design guidance
  only; no code in v2.2.0.**

## Verification

Per-commit gates (all green for every commit):

```sh
cargo test --workspace --no-fail-fast
cargo clippy --workspace -- -D warnings
```

Release gates before tagging v2.2.0:

```sh
pnpm --filter @vestige/dashboard check
pnpm --filter @vestige/dashboard build
```

Plus a `tools/list` smoke check asserting exactly **12** advertised names
(`test_tools_list_returns_all_tools`).
