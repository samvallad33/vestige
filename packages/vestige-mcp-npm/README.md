# vestige-mcp-server

**The Causal Proof Engine. The operating system for AI agents.**

**Zero vectors. Zero RAG. No lookalike text.** Nothing in the engine is found, ranked, paired or explained by embeddings, keyword scoring or similarity. Every answer comes from what Strata recorded and carries its proof: a memory id, an edge path, or a receipt.

Strata is the kernel: an append-only, hash-chained, signed log on your machine. One install gives every agent on that machine the same kernel, the same gate, the same debugger and the same shell. Nothing leaves your machine by default.

This package installs the Vestige binaries. Full documentation: [github.com/samvallad33/vestige](https://github.com/samvallad33/vestige).

## Install

```bash
npm install -g vestige-mcp-server
```

The installer downloads the release archive for your platform from the GitHub Release and checks its `.sha256` before it installs anything. Node 18 or newer. You can also download the archive yourself from [the latest release](https://github.com/samvallad33/vestige/releases/latest).

```bash
vestige-mcp --version
```

| Command | What it is |
|---|---|
| `vestige-mcp` | The MCP server your agents run |
| `vestige` | The CLI: `backup`, `strata-verify`, `dashboard` and more |
| `vestige-upgrade` | One-shot importer for a v3 store |
| `vestige-restore` | Re-imports an export as new records |

## Connect your agents

**Claude Code**

```bash
claude mcp add vestige vestige-mcp -s user
```

**Codex**

```bash
codex mcp add vestige -- vestige-mcp
```

**Claude Desktop and any other client**

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

Desktop apps do not read your shell's PATH, so use the full path from `which vestige-mcp` there. Then restart the client. Claude Desktop's config lives at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS and `%APPDATA%\Claude\claude_desktop_config.json` on Windows.

Every agent on the machine can run Vestige at once. The first to start serves the store and the others attach to it, so they all read and write through one writer.

## What is in the system

- **Strata**, the kernel: an append-only, hash-chained, signed log. State is re-derived by replaying it.
- **Admission**: every write is proposed, checked by the gate and admitted, and returns a receipt you can replay.
- **Recall by handle**: an agent finds a record by id, unique id prefix or exact tag, never by resemblance.
- **The debugger**: `causal_walk` walks a failure backward along recorded links to the decision behind it. `forgotten_lesson` and `selftest` work the same way.
- **GhostLink**: finds pairs of records nobody has combined, each with its proof.
- **Lifecycle**: FSRS scheduling, dreaming over recorded edges, and suppression.
- **Source awareness**: decisions anchored to code, drift flagged, and git history turned into change records.

## Dashboard and backups

```bash
vestige dashboard         # http://127.0.0.1:3927
vestige backup <new-dir>  # works while your agents are running
```

## Upgrading from v3

Quit every app that runs Vestige v3, point them at the 4.x `vestige-mcp`, and start them again. The first launch builds a Strata log from your `vestige.db`, verifies it against a signed migration receipt, and never modifies the v3 file. Details: [Upgrading from v3](https://github.com/samvallad33/vestige#upgrading-from-v3).

## Environment

| Variable | What it does |
|---|---|
| `VESTIGE_DATA_DIR` | Data directory when `--data-dir` is absent. The Strata log lives in `log/` inside it |
| `VESTIGE_DASHBOARD_PORT` | Dashboard port (default 3927) |
| `VESTIGE_AUTH_TOKEN` | Bearer token for the dashboard API and HTTP MCP |
| `RUST_LOG` | Log filter |

More in [docs/CONFIGURATION.md](https://github.com/samvallad33/vestige/blob/main/docs/CONFIGURATION.md).

## Platforms

macOS (Apple silicon and Intel), Linux x64 and arm64 (glibc 2.35 or newer), Windows x64.

## License

AGPL-3.0-only.
