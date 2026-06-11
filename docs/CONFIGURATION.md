# Configuration Reference

> Environment variables, CLI commands, and setup options

---

## First-Run Network Requirement

Vestige downloads the **Nomic Embed Text v1.5** model (~130MB) from Hugging Face on first use. Qwen3 embeddings are opt-in and download their own Hugging Face model when selected.

**All subsequent runs are fully offline.**

### Model Cache Location

The embedding model is cached in platform-specific directories:

| Platform | Cache Location |
|----------|----------------|
| macOS | `~/Library/Caches/vestige/fastembed` |
| Linux | `~/.cache/vestige/fastembed` |
| Windows | `%LOCALAPPDATA%\vestige\cache\fastembed` |

Override with environment variable:
```bash
export FASTEMBED_CACHE_PATH="/custom/path"
```

Qwen3 currently uses Hugging Face Hub's Candle loader directly, so use the standard Hugging Face cache environment such as `HF_HOME` if you need to relocate that larger model cache.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_DATA_DIR` | OS per-user data directory | Storage directory fallback; overridden by `--data-dir`; database lives at `<dir>/vestige.db` |
| `VESTIGE_EMBEDDING_MODEL` | `nomic-v1.5` | Embedding backend selector. Use `qwen3-0.6b` with a build that enables `qwen3-embeddings` |
| `RUST_LOG` | `info` (via tracing-subscriber) | Log verbosity + per-module filtering |
| `FASTEMBED_CACHE_PATH` | Platform cache directory; `./.fastembed_cache` fallback | Embedding model cache location |
| `VESTIGE_DASHBOARD_PORT` | `3927` | Dashboard HTTP + WebSocket port |
| `VESTIGE_HTTP_ENABLED` | `false` | Set `true` or `1` to enable optional MCP-over-HTTP |
| `VESTIGE_HTTP_PORT` | `3928` | Optional MCP-over-HTTP port; `--http-port` also enables HTTP |
| `VESTIGE_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `VESTIGE_HTTP_ALLOWED_ORIGINS` | localhost origins for the HTTP port | Comma-separated browser origins allowed to call MCP-over-HTTP |
| `VESTIGE_AUTH_TOKEN` | auto-generated | Dashboard + MCP HTTP bearer auth |
| `VESTIGE_DASHBOARD_ENABLED` | `false` | Set `true` or `1` to enable the web dashboard |
| `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` | `6` | FSRS-6 decay cycle cadence |

> **Storage location precedence:** `--data-dir <path>` wins over `VESTIGE_DATA_DIR`; if neither is set, Vestige uses your OS's per-user data directory: `~/Library/Application Support/com.vestige.core/` on macOS, `~/.local/share/vestige/core/` on Linux, `%APPDATA%\vestige\core\` on Windows. Custom paths are directories, are created if missing, expand a leading `~`, and store the database at `<dir>/vestige.db`.

---

## Output Configuration (`vestige.toml`)

> Added in **v2.1.24** (Roadmap Phase 2: Configurable Output).

You can control the default shape and size of high-traffic MCP responses with an
optional config file. It is **local-first** — no cloud service is involved — and
**fully backward-compatible**: with no file present, Vestige behaves exactly as
it did before.

### Location

The config file lives in the active Vestige data directory, alongside the
database:

```
<data_dir>/vestige.toml      # e.g. ~/Library/Application Support/com.vestige.core/vestige.toml
```

The data directory is resolved with the same precedence as storage
(`--data-dir` > `VESTIGE_DATA_DIR` > OS per-user data dir). A missing file, or a
file with no recognized keys, falls back to built-in defaults. The parser is
lenient: unknown keys and unknown sections are ignored, so the file can grow in
future releases without breaking older binaries.

### `[defaults]` table

```toml
[defaults]
# Detail level for high-traffic tools: "brief" | "summary" | "full"
detail_level = "summary"

# Default result count for high-traffic tools (positive integer)
limit = 10

# Output profile: "lean" | "default" | "audit" | "research"
profile = "default"
```

All three keys are optional. `detail_level` and `limit`, when set, override the
selected profile's presets.

### Output profiles

A profile presets a coherent bundle of detail level, default limit, and whether
scores and timestamps are included:

| Profile | Detail | Default limit | Scores | Timestamps | Use when |
|---------|--------|---------------|--------|------------|----------|
| `lean` | `brief` | 5 | dropped | dropped | Context budget matters most |
| `default` | `summary` | tool default | shown | shown | **Historical behavior (unchanged)** |
| `audit` | `full` | tool default | shown | shown | Reviewing or debugging memory state |
| `research` | `full` | 25 | shown | shown | Wide, detailed result sets |

### Precedence

Resolved per call, highest to lowest:

1. **Explicit MCP parameter** (e.g. `detail_level` / `limit` on a `search`
   call) — always wins.
2. **`vestige.toml`** — the `[defaults]` keys and the selected profile.
3. **Built-in default** — the `default` profile, identical to pre-v2.1.24
   behavior.

### Affected tools

`search`, `memory_timeline`, `codebase` (`get_context`), and `session_context`
resolve their default detail level and result limit through this config. Each of
these tools also echoes the active `profile` in its response so you can confirm
what was applied. Tools that take no `detail_level`/`limit` are unaffected.

### Example: minimize context cost

```toml
[defaults]
profile = "lean"
```

### Example: detailed audits without changing the profile

```toml
[defaults]
detail_level = "full"
limit = 50
```

---

## Command-Line Options

```bash
vestige-mcp --data-dir /custom/path   # Custom storage location
VESTIGE_DATA_DIR=~/.vestige vestige-mcp # Env fallback storage location
VESTIGE_DATA_DIR=./.vestige vestige stats # Point the CLI at the same custom DB
vestige-mcp --help                     # Show all options
```

---

## CLI Commands (v1.1+)

Stats and maintenance were moved from MCP to CLI to minimize context window usage:

```bash
vestige stats              # Memory statistics
vestige stats --tagging    # Retention distribution
vestige stats --states     # Cognitive state distribution
vestige health             # System health check
vestige consolidate        # Run memory maintenance
vestige restore <file>     # Restore from backup
vestige portable-export <file>         # Exact Vestige-to-Vestige archive
vestige portable-import <file>         # Import exact archive into an empty database
vestige portable-import <file> --merge # Merge exact archive into this database
vestige sync <file>                    # Pull/merge/push through a file backend
```

---

## Client Configuration

### Codex (One-liner)

```bash
codex mcp add vestige -- /usr/local/bin/vestige-mcp
```

### Codex (Manual)

Add to `~/.codex/config.toml`:
```toml
[mcp_servers.vestige]
command = "/usr/local/bin/vestige-mcp"
```

### Claude Code (One-liner)

```bash
claude mcp add vestige vestige-mcp -s user
```

### Claude Code (Manual)

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### Claude Desktop (macOS)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### Claude Desktop (Windows)

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### OpenCode

OpenCode supports global and project-local config. For a project-local setup, add to `opencode.json`:
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

For isolated per-project memory, pass the data directory in the command array:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vestige": {
      "type": "local",
      "command": ["vestige-mcp", "--data-dir", "./.vestige"],
      "enabled": true,
      "timeout": 10000
    }
  }
}
```

See the [OpenCode integration guide](integrations/opencode.md) for global config, verification, and troubleshooting.

---

## Custom Data Directory

For per-project or custom storage:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "/path/to/custom/dir"]
    }
  }
}
```

For a shell-level default:

```bash
export VESTIGE_DATA_DIR="/path/to/custom/dir"
```

`--data-dir` takes precedence over `VESTIGE_DATA_DIR`, so you can keep a global env default and still isolate one client or project with an explicit CLI argument.

See [Storage Modes](STORAGE.md) for more options.

---

## Updating Vestige

**Latest version:**
```bash
vestige update
```

This updates `vestige`, `vestige-mcp`, and `vestige-restore`. It does not mutate
Claude Code Cognitive Sandwich companion files unless you explicitly request it.

**Also refresh optional Claude Code companion files:**
```bash
vestige update --sandwich-companion
```

**Pin to specific version:**
```bash
vestige update --version v2.1.21
```

**Manage the optional Cognitive Sandwich layer without updating binaries:**
```bash
vestige sandwich install
vestige sandwich install --enable-preflight
vestige sandwich install --enable-sanhedrin --sanhedrin-endpoint=http://127.0.0.1:11434/v1/chat/completions
```

**Check your version:**
```bash
vestige-mcp --version
```

---

## Development

```bash
# Run tests
cargo test --all-features

# Run with logging
RUST_LOG=debug cargo run --release

# Build optimized binary
cargo build --release --all-features
```
