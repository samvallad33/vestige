# Vestige MCP Server

Local cognitive memory for MCP-compatible AI agents.

This crate provides the `vestige-mcp` stdio MCP server plus the `vestige` CLI.
The cognitive engine lives in `vestige-core`; this crate owns protocol handling,
tool dispatch, optional dashboard serving, backups, restore, update, and
portable import/export commands.

## Install

For normal users, prefer the release package:

```bash
npm install -g vestige-mcp-server
```

For local development:

```bash
cargo build --release -p vestige-mcp
```

## Register With An MCP Client

Use the command `vestige-mcp` in any stdio MCP client:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

Examples:

```bash
claude mcp add vestige vestige-mcp -s user
codex mcp add vestige -- vestige-mcp
```

## Transports

- Default: JSON-RPC 2.0 over stdio.
- Optional: MCP-over-HTTP on `/mcp`, enabled only with `--http`,
  `--http-port`, or `VESTIGE_HTTP_ENABLED=1`.
- Dashboard: `vestige dashboard` or `VESTIGE_DASHBOARD_ENABLED=1`.

HTTP and dashboard bearer tokens are generated locally; see
[`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md).

## Current Tool Surface

As of v2.2, the server advertises **13 consolidated tools** from
[`src/server.rs`](src/server.rs):

- `recall` — unified retrieval (folds `search` + `deep_reference` +
  `cross_reference` + `contradictions`); modes `lookup`/`reason`/`contradictions`
- `smart_ingest` — Prediction-Error-gated ingestion (single + batch)
- `memory` — get / edit / promote / demote / state / purge
- `graph` — chains, associations, bridges, predictions, force-directed export
  (folds `explore_connections` + `predict` + `memory_graph` + `composed_graph`)
- `maintain` — consolidate / dream / gc / importance_score / backup / export / restore
- `dedup` — scan / plan_merge / plan_supersede / apply / undo / protect / policy
  (folds `find_duplicates` + the 7 Phase-3 merge tools)
- `memory_status` — health / retention / timeline / changelog
  (folds `system_status` + `memory_health` + `memory_timeline` + `memory_changelog`)
- `suppress` — top-down active forgetting (Anderson 2025 + Davis Rac1)
- `backfill` — Retroactive Salience Backfill (Cai 2024 Nature), the flagship
- `codebase` — per-project patterns and decisions
- `intention` — set / check / update / list future triggers
- `source_sync` — external-source connectors (GitHub, Redmine)
- `session_start` — one-call session initialization (renamed from `session_context`)

The ~22 pre-consolidation names (`search`, `deep_reference`, `dream`,
`consolidate`, `memory_graph`, …) still work as **hidden back-compat aliases**
dispatched in `handle_tools_call`, so existing configs keep working.

See the root [`README.md`](../../README.md) and
[`docs/AGENT-MEMORY-PROTOCOL.md`](../../docs/AGENT-MEMORY-PROTOCOL.md) for
agent instructions.

## License

AGPL-3.0-only
