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
| `VESTIGE_BACKFILL_AUTOFIRE` | `on` | Retroactive Salience Backfill auto-fire during consolidation. On by default; set `0`/`false`/`off`/`no` to disable. The manual `backfill` tool + CLI stay available either way. When on, promotion is bounded (`stability = MIN(stability * 1.5, stability + 365)`) |
| `VESTIGE_AUTO_CONSOLIDATE_MERGE` | `on` | Auto concat-merge of near-duplicate memories during consolidation (keeps the strongest, folds the rest in as `[MERGED]` blocks, deletes the originals). On by default; set `0`/`false`/`off`/`no` to disable. Protected (`dedup protect`) memories are never absorbed or deleted by this pass, on or off. |
| `VESTIGE_TRACE` | `on` | Agent Black Box trace recording. **On by default**: every MCP tool call writes rows to `agent_traces`/`agent_runs` in your local database. Set `0`/`false`/`off`/`no` to turn the recorder off. Read once per process, so changing it mid-process has no effect |
| `VESTIGE_TRACE_RETENTION_DAYS` | `30` | How long Black Box traces are kept. The consolidation cycle deletes trace events older than this and drops any `agent_runs` roll-up left with no events. `0` keeps traces forever (sweep disabled); unset, empty, negative, or malformed values fall back to `30` |
| `VESTIGE_DISABLE_VECTOR_SEARCH` | unset (vector search on) | Kill switch for the HNSW vector index. Set to `1`/`true`/`yes`/`on`/`enable`/`enabled` to force semantic/vector search off and fall back to keyword search. Useful on older x86 CPUs — the index also disables itself automatically when AVX2+FMA are missing |

> **Storage location precedence:** `--data-dir <path>` wins over `VESTIGE_DATA_DIR`; if neither is set, Vestige uses your OS's per-user data directory: `~/Library/Application Support/com.vestige.core/` on macOS, `~/.local/share/vestige/core/` on Linux, `%APPDATA%\vestige\core\` on Windows. Custom paths are directories, are created if missing, expand a leading `~`, and store the database at `<dir>/vestige.db`.

### Vestige Pro (hosted cloud sync)

These are read only by `vestige sync --cloud`. Leave them unset and Vestige stays fully local — nothing is uploaded and no network call is made.

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_CLOUD_ENDPOINT` | unset | Hosted managed-sync endpoint, issued when you subscribe. `--endpoint` on `vestige sync --cloud` takes precedence |
| `VESTIGE_CLOUD_SYNC_KEY` | unset | Per-user bearer key for the hosted service, issued when you subscribe. Authenticates the transport only — it is **not** the encryption passphrase |
| `VESTIGE_CLOUD_ENCRYPTION_KEY` | unset (**required** for cloud sync) | Passphrase for client-side zero-knowledge encryption (Argon2id KDF → XChaCha20-Poly1305, `VSTGENC1` envelope). Use the same passphrase on every device |

> **The passphrase never leaves your machine.** The archive is encrypted on-device
> before upload and decrypted after download, so the hosted service only ever
> stores ciphertext. Vestige has no copy of `VESTIGE_CLOUD_ENCRYPTION_KEY` and
> **cannot reset or recover it** — if you lose it, the synced blob is
> unrecoverable by design. Encryption is mandatory: the client refuses to upload
> an unencrypted archive and rejects a plaintext archive on download.

---

## Output Configuration (`vestige.toml`)

> Added in **v2.1.26** (Roadmap Phase 2: Configurable Output).

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
3. **Built-in default** — the `default` profile, identical to pre-v2.1.26
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
vestige sync --cloud                   # Pull/merge/push through Vestige Pro (see cloud env vars)
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
