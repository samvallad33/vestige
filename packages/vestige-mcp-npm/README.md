# vestige-mcp-server

Vestige MCP Server - A synthetic hippocampus for AI assistants.

Built on 130 years of cognitive science research, Vestige provides biologically-inspired memory that decays, strengthens, and consolidates like the human mind.

## Installation

```bash
npm install -g vestige-mcp-server
```

This automatically downloads the correct binary for your platform (macOS, Linux, Windows) from GitHub releases.

Already installed? Update without copying release URLs:

```bash
vestige update
```

This refreshes the binaries only. Optional Claude Code Cognitive Sandwich
companion files are refreshed with `vestige update --sandwich-companion` or
`vestige sandwich install`.

### What gets installed

| Command | Description |
|---------|-------------|
| `vestige-mcp` | MCP server for local agent memory |
| `vestige` | CLI for stats, health checks, and maintenance |
| `vestige-restore` | Restore helper for backup recovery |

### Verify installation

```bash
vestige health
```

## Usage with MCP Clients

Vestige works with any client that can register a stdio MCP server.

**Claude Code**

```bash
claude mcp add vestige vestige-mcp -s user
```

**Codex**

```bash
codex mcp add vestige -- vestige-mcp
```

Then restart your MCP client.

**OpenCode**

Add to `~/.config/opencode/opencode.json` or a project-local `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vestige": {
      "type": "local",
      "command": ["vestige-mcp"],
      "enabled": true,
      "timeout": 10000
    }
  }
}
```

Prefer the installed `vestige-mcp` command for OpenCode. If you run Vestige directly through `npx`, use a longer first-run timeout because npm may need to download the package before OpenCode can connect:

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

## Usage with Claude Desktop

Add to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

## CLI Commands

```bash
vestige stats          # Memory statistics
vestige stats --states # Cognitive state distribution
vestige health         # System health check
vestige consolidate    # Run memory maintenance cycle
vestige update         # Update binaries
vestige update --sandwich-companion # Also refresh optional Claude Code files
vestige sandwich install # Manage optional Claude Code hook files
```

## Features

- **FSRS-6 Algorithm**: State-of-the-art spaced repetition for optimal memory retention
- **Dual-Strength Memory**: Bjork & Bjork (1992) - Storage + Retrieval strength model
- **Synaptic Tagging**: Memories become important retroactively (Frey & Morris 1997)
- **Semantic Search**: Local embeddings via nomic-embed-text-v1.5 (768 dimensions)
- **Local-First**: All data stays on your machine - no cloud, no API costs

## Storage & Memory

Vestige uses SQLite for storage. Your memories are stored on **disk**, not in RAM.

- **Database limit**: 216TB (SQLite theoretical max)
- **RAM usage**: ~64MB cache (configurable)
- **Typical usage**: 1 million memories ≈ 1-2GB on disk

You'll never run out of space. A heavy user creating 100 memories/day would use ~1.5GB after 10 years.

## Embeddings

On first use, Vestige downloads the nomic-embed-text-v1.5 model (~130MB). This is a one-time download and all subsequent operations are fully offline.

The model is stored in Vestige's OS cache directory, or you can set a global location:

```bash
export FASTEMBED_CACHE_PATH="$HOME/.fastembed_cache"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log verbosity + per-module filter | `info` |
| `FASTEMBED_CACHE_PATH` | Embeddings model cache override | OS cache dir |
| `VESTIGE_DATA_DIR` | Storage directory fallback; database lives at `<dir>/vestige.db` | OS data dir |
| `VESTIGE_DASHBOARD_PORT` | Dashboard port | `3927` |
| `VESTIGE_AUTH_TOKEN` | Bearer auth for dashboard + HTTP MCP | auto-generated |

Storage precedence is `--data-dir <path>`, then `VESTIGE_DATA_DIR`, then your OS's per-user data directory.

## Troubleshooting

### "Could not attach to MCP server vestige"

1. Verify binary exists: `which vestige-mcp`
2. Test directly: `vestige-mcp` (should wait for stdio input)
3. Check your MCP client's server logs.

### "vestige: command not found"

Reinstall the package:
```bash
npm install -g vestige-mcp-server
```

### Embeddings not downloading

The model downloads on first memory ingest or search operation. If your MCP
client cannot connect to the MCP server, no memory operations happen and no
model downloads.

Fix the MCP connection first, then the model will download automatically.

## Supported Platforms

| Platform | Architecture |
|----------|--------------|
| macOS | ARM64 (Apple Silicon), x86_64 (Intel) |
| Linux | x86_64 |
| Windows | x86_64 |

## License

AGPL-3.0-only
