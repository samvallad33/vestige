# Vestige State And Plan

This document is a public, sanitized replacement for an older internal planning
snapshot. It intentionally omits private local paths, personal operating
context, unpublished roadmap notes, and private repository locations.

For current user-facing release information, use:

- `README.md`
- `CHANGELOG.md`
- `docs/STORAGE.md`
- `docs/COGNITIVE_SANDWICH.md`
- `docs/AGENT-MEMORY-PROTOCOL.md`
- `docs/CLAUDE-SETUP.md`

## Current Release Shape

Vestige v2.2.0 is the "Retroactive Salience + Tool Consolidation" release. Its
public scope is:

- Retroactive Salience Backfill (Cai 2024 *Nature*): on a recorded failure,
  reach backward in time to promote the causally-upstream root cause. Auto-fires
  in the consolidation pass; also exposed as the `backfill` MCP tool and
  `vestige backfill` CLI
- MCP tool consolidation: 34 tools collapse to 13 advertised
  (`recall`, `smart_ingest`, `memory`, `graph`, `maintain`, `dedup`,
  `memory_status`, `suppress`, `backfill`, `codebase`, `intention`,
  `source_sync`, `session_start`); the pre-consolidation names remain
  dispatchable as hidden back-compat aliases
- `deep_reference`/`recall` retrieval engine upgrades: F32 embeddings (was I8),
  Reciprocal Rank Fusion, and claim-vs-memory contradiction checks

It carries forward the local-first hardening baseline from prior releases:

- stdio MCP as the default agent transport, with HTTP MCP opt-in only
- binary-only `vestige update` by default
- delete and purge confirmation parity for destructive memory removal
- portable sync handling for purge tombstones, UPSERT merge, and vector index
  reloads
- release packaging with dashboard freshness checks and checksums
- agent-neutral memory instructions for any MCP-compatible client

The release keeps the local-first baseline intact. Heavy model hooks, local
verifier models, and preflight automation remain optional.

## Release Gates

Before tagging a release, run:

```sh
cargo test --workspace --no-fail-fast
cargo clippy --workspace -- -D warnings
pnpm --filter @vestige/dashboard check
pnpm --filter @vestige/dashboard build
git diff --check
```

For dashboard route changes, rebuild and stage `apps/dashboard/build/` so the
embedded static assets match `apps/dashboard/src/`.

## Product Principles

- Exact things should stay exact. Literal identifiers should not lose to
  semantic expansion.
- Forgetting should be honest. A hard purge should remove content, embeddings,
  graph edges, and derived references while retaining only non-content proof
  that deletion happened.
- Contradictions should be visible. Trust-weighted disagreement should be
  inspectable directly instead of hidden inside a broader reasoning tool.
- Installation should remain boring. Users should not need a large local model
  or background hook system just to use memory.
- Pro features should add managed convenience without weakening local-first
  ownership.

## Public Architecture Summary

Vestige is organized as:

- `crates/vestige-core`: storage, search, embeddings, memory lifecycle, FSRS,
  graph, dream, and cognitive modules
- `crates/vestige-mcp`: MCP server, CLI, dashboard backend, tools, update flow
- `apps/dashboard`: SvelteKit dashboard source
- `packages/vestige-mcp-npm`: npm wrapper for the MCP binary
- `packages/vestige-init`: installer helper
- `docs`: user and integration documentation

## v2.2.0 Implementation Notes

The advertised MCP surface is 13 tools (see `handle_tools_list` in
`crates/vestige-mcp/src/server.rs`, verified by `test_tools_list_returns_all_tools`).
Consolidation is a facade: the ~22 folded/renamed names still dispatch through
`handle_tools_call` as hidden aliases, so existing client configs keep working.
`backfill` is deliberately advertised as its own tool rather than folded into
`maintain`, because backward causal promotion is a distinct cognitive primitive.

HTTP MCP is disabled unless the user passes `--http`, passes `--http-port`, or
sets `VESTIGE_HTTP_ENABLED=1`. The stdio MCP server remains the portable default
for Claude Code, Codex, Cursor, VS Code, Xcode, OpenCode, JetBrains, Windsurf,
and other clients.

Purge is implemented transactionally in storage and surfaced through the MCP
`memory` tool. `memory(action="purge", confirm=true)` is the explicit hard
delete path. `delete` remains a backwards-compatible alias but also requires
`confirm=true`.

Portable merge imports preserve both sync tombstones and non-content deletion
tombstones. Keyed table writes use UPSERT rather than `INSERT OR REPLACE` so
related rows are not accidentally cascaded away.

Claude Code Cognitive Sandwich files are optional companion files, not the
default Vestige setup path. Use `vestige update --sandwich-companion` or
`vestige sandwich install` only when that hook layer is wanted.

## 15. Autopilot Rationale

The backend event bus exists so dashboard and MCP activity can be observed by
the cognitive engine without making user-facing agent hooks mandatory. Any
autonomous behavior should be conservative, rate-limited, and local-first.

Autopilot-style routing should never require a remote model, a heavy local
model, or a Claude hook to make normal memory useful. It should only connect
already-emitted Vestige events to existing cognitive modules when that improves
maintenance, retrieval quality, or dashboard fidelity without surprising the
user.
