# OpenCode Adoption Plan

Status: Vestige was tested with OpenCode `1.16.2` on June 8, 2026. The working config uses OpenCode's top-level `mcp.vestige` schema, not `mcpServers`.

Public promotion started:

- Vestige PR #70: `https://github.com/samvallad33/vestige/pull/70`
- OpenCode issue: `https://github.com/anomalyco/opencode/issues/31402`
- OpenCode docs/ecosystem PR: `https://github.com/anomalyco/opencode/pull/31405`
- awesome-opencode PR: `https://github.com/awesome-opencode/awesome-opencode/pull/418`
- opencode.cafe listing request: `https://github.com/R44VC0RP/opencode.cafe/issues/6`
- OpenCode persistent memory comment: `https://github.com/anomalyco/opencode/issues/16077#issuecomment-4652064625`

## Release Gate

- PR #67 is merged upstream and should be treated as the contributor-driven starting point.
- Ship the corrected OpenCode config docs and `@vestige/init` migration from stale `mcpServers.vestige` to `mcp.vestige`.
- Ship the background embedding initialization fix before making direct `npx` the main OpenCode install path. A cold published `2.1.23` package can still time out while OpenCode waits for tools.
- After release, verify all three OpenCode paths again:
  - installed binary: `command: ["vestige-mcp"]`
  - project memory: `command: ["vestige-mcp", "--data-dir", "./.vestige"]`
  - direct npm: `command: ["npx", "-y", "-p", "vestige-mcp-server@latest", "vestige-mcp"]` with `timeout: 60000`

## Official OpenCode PR

Target repo: `https://github.com/anomalyco/opencode`

Files:

- `packages/web/src/content/docs/mcp-servers.mdx`
- `packages/web/src/content/docs/ecosystem.mdx`

MCP docs snippet:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vestige": {
      "type": "local",
      "command": ["npx", "-y", "-p", "vestige-mcp-server@latest", "vestige-mcp"],
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

Ecosystem row:

```md
| [Vestige](https://github.com/samvallad33/vestige) | Local MCP memory server for OpenCode that remembers project decisions, preferences, and previous fixes across sessions |
```

Positioning: local, inspectable MCP memory for OpenCode. Avoid claiming Vestige fixes OpenCode's process memory or session resume behavior.

## Awesome OpenCode

Target repo: `https://github.com/awesome-opencode/awesome-opencode`

Suggested entry, with category to confirm against maintainer preference (`data/projects/vestige.yaml` or `data/resources/vestige.yaml`):

```yaml
name: Vestige
repo: https://github.com/samvallad33/vestige
tagline: Local persistent memory for OpenCode
description: Local MCP server that lets OpenCode remember project decisions, preferences, architecture context, and previous fixes across sessions.
scope:
  - global
  - project
tags:
  - mcp
  - memory
  - local-first
  - sqlite
  - opencode
min_version: 1.16.2
homepage: https://github.com/samvallad33/vestige/blob/main/docs/integrations/opencode.md
installation: |
  npm install -g vestige-mcp-server@latest
  npx @vestige/init
```

## MCP Directories

Current state:

- Official MCP Registry already lists `io.github.samvallad33/vestige` at `https://registry.modelcontextprotocol.io/v0/servers?search=vestige`.
- Smithery already lists Vestige and indexes 25 tools: `https://smithery.ai/server/@samvallad33/vestige`.
- Glama already lists Vestige, but the listing needs a refresh/fix if it shows no tools: `https://glama.ai/mcp/servers/samvallad33/vestige`.
- `mcp.so` does not show Vestige under the expected slugs yet; submit manually at `https://mcp.so/submit`.

Priority order:

1. Official MCP Registry: `https://github.com/modelcontextprotocol/registry`
2. Awesome MCP Servers: `https://github.com/punkpeye/awesome-mcp-servers`
3. Glama MCP directory: `https://glama.ai/mcp/servers`
4. Smithery: `https://smithery.ai`
5. PulseMCP: `https://www.pulsemcp.com`

Registry metadata is mostly ready: `server.json` exists and `packages/vestige-mcp-npm/package.json` has `mcpName: "io.github.samvallad33/vestige"`. Publish only when the package version and `server.json` version match the released npm package.

## Community Launch

Use tested technical copy, not hype:

> Vestige now works with OpenCode as a local MCP memory server. It gives OpenCode persistent memory for project decisions, preferences, architecture context, and previous fixes across sessions. Install with `npm install -g vestige-mcp-server@latest`, run `npx @vestige/init`, then verify with `opencode mcp list`.

High-signal channels after release:

- OpenCode Discord: `https://opencode.ai/discord`
- opencode.cafe MCP Server listing: `https://opencode.cafe`
- OpenCode memory-related GitHub issues, only where directly relevant
- Hacker News and Lobsters with a technical post about the tested OpenCode integration and failure modes
- npm keyword/discovery after the next package release includes `opencode`

## Proof Checklist

- `opencode debug config` accepts `mcp.vestige`.
- `opencode mcp list` shows `vestige connected`.
- Stale `mcpServers.vestige` examples fail in OpenCode and are migrated by `@vestige/init`.
- OpenCode tools are prefixed as `vestige_search`, `vestige_smart_ingest`, `vestige_session_context`, and `vestige_deep_reference`.
- The OpenCode guide says `timeout: 60000` for direct `npx` and `timeout: 10000` for installed binaries.
