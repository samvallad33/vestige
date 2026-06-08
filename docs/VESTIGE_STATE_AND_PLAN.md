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

Vestige v2.1.21 is the "Agent-Neutral Hardening" release. Its public scope is:

- stdio MCP as the default agent transport, with HTTP MCP opt-in only
- binary-only `vestige update` by default
- delete and purge confirmation parity for destructive memory removal
- portable sync fixes for purge tombstones, UPSERT merge, and vector index
  reloads
- safer release packaging with dashboard freshness checks and checksums
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

## v2.1.21 Implementation Notes

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
