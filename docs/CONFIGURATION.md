# Configuration Reference

> Environment variables, CLI commands, and setup options for Vestige 4.2.0

---

## Offline by default

A default 4.x build has no embedding model, so there is **no first-run download**, and it
makes no startup version check. Nothing is fetched when it starts. The calls that can
use the network are all explicit:

| Call | Network use | In a 4.x release build |
|------|-------------|------------------------|
| `vestige update` | Downloads a release archive and its `.sha256` from GitHub, verifies the checksum, replaces the binaries | Yes, only when you run it |
| `source_sync` tool (`--features connectors`) | Indexes GitHub or Redmine issues | No. The feature is off, and the name is an unknown tool |
| `vestige sync --cloud` (`--features cloud-sync`) | Hosted sync | No. The feature is off, and `sync` is refused on a Strata log |
| `ghostlink` `weave` `evidence` | None. Vestige records the URL and hash you give it and never fetches | Yes |

The dashboard listens on `127.0.0.1` only. The optional HTTP transport binds `127.0.0.1`
unless you set `VESTIGE_HTTP_BIND`.

---

## Environment Variables

These are read by the 4.x default build.

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_DATA_DIR` | OS per-user data directory | Data directory fallback; overridden by `--data-dir`. The Strata log lives in `<dir>/log/` |
| `RUST_LOG` | `info` | Log verbosity and per-module filtering. Logs go to stderr, because stdout carries the protocol |
| `VESTIGE_DASHBOARD_PORT` | `3927` | Dashboard HTTP and WebSocket port. `vestige dashboard --port` sets it for one run |
| `VESTIGE_DASHBOARD_ENABLED` | `false` | Set `true` or `1` to have `vestige-mcp` serve the web dashboard for as long as it runs. Without it, `vestige dashboard` asks the running server for a dashboard, and that dashboard stops when the last `vestige dashboard` exits |
| `VESTIGE_HTTP_ENABLED` | `false` | Set `true` or `1` to enable the optional MCP-over-HTTP transport. `--http` does the same |
| `VESTIGE_HTTP_PORT` | `3928` | MCP-over-HTTP port. `--http-port` also enables HTTP |
| `VESTIGE_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `VESTIGE_HTTP_ALLOWED_ORIGINS` | localhost origins for the HTTP port | Comma-separated browser origins allowed to call MCP-over-HTTP |
| `VESTIGE_AUTH_TOKEN` | auto-generated | Bearer token for the HTTP transport and for dashboard writes that do not come from the dashboard's own page (a script's POST or DELETE). When unset, both read the `auth_token` file in the default per-user data directory, even with `--data-dir`. The HTTP transport creates that file; the dashboard never does. See [Dashboard access](#dashboard-access) |
| `VESTIGE_SYSTEM_PROMPT_MODE` | `minimal` | `minimal` or `full`. `full` sends the longer composition instructions in the MCP handshake. Any other value falls back to `minimal` |
| `VESTIGE_ATTACH_WAIT_SECS` | `120` | How long a new `vestige-mcp` waits for the process holding the store to accept an attach, and how long `vestige-upgrade` waits for `.serve.lock`. A first launch on a large v3 store holds the lock for the whole import |
| `VESTIGE_STATE_TTL_DAYS` | `30` | Lifetime of a `state` record when you give no `validUntil`. `0` disables the default |

`vestige-mcp --help` prints the same list.

### Dashboard access

The dashboard binds `127.0.0.1` only. Any web page open in your browser can
still send it requests, and CORS only decides whether that page may read the
answer. So the dashboard also checks each request itself:

- `Host` must be `127.0.0.1:<port>` or `localhost:<port>`. A DNS-rebound
  hostname is refused, so it cannot read your store through the dashboard.
  If you forward the dashboard over SSH, use the same port number on both
  ends.
- A request that carries an `Origin` must carry the dashboard's own
  (`http://127.0.0.1:<port>` or `http://localhost:<port>`). A load from
  another site (`Sec-Fetch-Site: cross-site`/`same-site`) is refused, except
  a link that opens a dashboard page.
- A request that changes anything (any method but GET, HEAD and OPTIONS)
  must come from the dashboard's own page or carry
  `Authorization: Bearer <token>`. The token comes from `VESTIGE_AUTH_TOKEN`,
  or else from the `auth_token` file in the data directory in use (`--data-dir`, then
  `VESTIGE_DATA_DIR`, then the per-user default). A custom directory with no token yet
  takes over the one the default location already holds.

A refused request gets `401 auth_required` or `403` with a JSON `error` that
says why. Reading the API from a local script (`curl http://127.0.0.1:3927/api/health`)
needs no token.

Dashboard routes for features a Strata log withholds answer `501` with a JSON `error` of
`unavailable_in_4_0`, `similarity_disabled` or `pending_strata`, the same words an MCP
caller sees. The Consolidate button is one of them.

> **Storage location precedence:** `--data-dir <path>` wins over `VESTIGE_DATA_DIR`; if
> neither is set, Vestige uses your OS's per-user data directory. Custom paths are
> directories, are created if missing, and expand a leading `~`. The defaults per platform
> are in [Storage](STORAGE.md#data-directory).

### Variables that no longer do anything

Most variables on earlier versions of this page configured the v3 engine. A default 4.x
build ignores them, or reads them and finds nothing to act on:

| Variable | Why it has no effect on a Strata log |
|----------|--------------------------------------|
| `FASTEMBED_CACHE_PATH`, `ORT_DYLIB_PATH`, `VESTIGE_DISABLE_VECTOR_SEARCH`, `HF_HOME` | There is no embedding model or vector index. The code no longer reads the first three |
| `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` | The periodic consolidation pass is a no-op on Strata. It runs and changes nothing |
| `VESTIGE_BACKFILL_AUTOFIRE`, `VESTIGE_FAILURE_FEEDBACK` | The v3 failure hooks do not run. Use `causal_walk` |
| `VESTIGE_AUTO_CONSOLIDATE_MERGE` | Merging near duplicates needs embeddings |
| `VESTIGE_DREAM_COMPILE_AUTOFIRE` | The automatic pass fails with "not implemented by this backend" and files nothing. Run `maintain` `dream_compile` yourself |
| `VESTIGE_TRACE`, `VESTIGE_TRACE_RETENTION_DAYS` | The Black Box trace recorder records nothing on a Strata log in 4.x |
| `VESTIGE_CLOUD_ENDPOINT`, `VESTIGE_CLOUD_SYNC_KEY`, `VESTIGE_CLOUD_ENCRYPTION_KEY` | Read only by `vestige sync --cloud`, which no 4.x release build includes and which a Strata log refuses |

---

## Review Modes (Memory PR write gating)

Memory PR review modes (`risk_gated`, `paranoid`) are **not available on a Strata log**.
A `review_mode.json` carried over from v3 is read as `fast`: it is logged once and the
file is left as written. The dashboard's Memory PR list says review is unavailable.

Every write still passes the log's gate and returns a receipt. A write that fails the
gate, such as one carrying a live credential, is refused and writes nothing.

---

## Output Configuration (`vestige.toml`)

You can control the default shape and size of high-traffic MCP responses with an
optional config file. It is **local-first**, and **fully backward-compatible**: with no
file present, Vestige behaves exactly as it does by default.

### Location

The config file lives in the active data directory, beside `log/`:

```
<data_dir>/vestige.toml      # e.g. ~/Library/Application Support/com.vestige.core/vestige.toml
```

The data directory is resolved with the same precedence as storage. A missing file, or a
file with no recognized keys, falls back to built-in defaults. The file is read at most
1 MiB. The parser is lenient: unknown keys and unknown sections are ignored, and it
reads section headers and `key = value` lines with string or integer values.

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
| `audit` | `full` | tool default | shown | shown | Reviewing or debugging the store |
| `research` | `full` | 25 | shown | shown | Wide, detailed result sets |

### Precedence

Resolved per call, highest to lowest:

1. **Explicit MCP parameter** (for example `detail_level` or `limit`) always wins.
2. **`vestige.toml`**: the `[defaults]` keys and the selected profile.
3. **Built-in default**: the `default` profile.

### Affected tools

On a Strata log, `session_start`, `memory_status` (the `timeline` view) and `codebase`
(`get_context`) resolve their default detail level and limit through this config, and
each echoes the active `profile` in its response so you can confirm what was applied.
`recall` by handle returns the same shape whatever the profile.

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
vestige-mcp --http --http-port 8080   # Enable the HTTP transport on a port
vestige-mcp --no-http                 # Force the HTTP transport off
VESTIGE_DATA_DIR="$HOME/.vestige" vestige-mcp # Env fallback (shell); GUI JSON does not expand ~
VESTIGE_DATA_DIR=./.vestige vestige stats # Point the CLI at the same custom directory
vestige-mcp --help                    # Show all options
```

---

## CLI Commands

The CLI is for inspection and maintenance. `backup`, `strata-verify` and `dashboard` work
while an agent's server holds the store. Every command that opens the log directly
exits with an error that names the holder while a server runs.

```bash
vestige stats                         # Counts, due for review, average retention
vestige stats --tagging               # Retention distribution
vestige stats --states                # Accessibility state distribution
vestige health                        # Health check
vestige ingest "<text>" --tags a,b    # Save a record through the gate (--node-type, --source)
vestige recall --handle <handle>      # Find by id, unique prefix, or exact tag
vestige compose --lens bridge         # GhostLink propose (or --lens divergent)
vestige causal-walk --logged-write <id>   # Walk back over recorded edges (or --node-id)
vestige connect --dry-run             # Join memories that record the same exact identity (touched edges)
vestige prove --logged-write <id> --repo <dir> --good <ref> --bad <ref> --test '<cmd>' --reported-at <rfc3339> --report out.json
                                      # Run your test on the walk's leads: frozen protocol, bisect, why, verdict card
vestige prove --check out.json        # Re-verify a report offline
vestige forgotten-lesson <failure-id> # Faded fix or lesson records behind a failure
vestige selftest                      # Planted-cause check on a throwaway copy
vestige project --out CLAUDE.md       # Preview the projected fence; --write applies it
vestige scan-secrets                  # Audit the log for credential-shaped values
vestige export <file> --format jsonl  # Export live records
vestige backup <new-folder>           # Owner-only copy of the log
vestige strata-verify <data-dir>      # Verify the log and its migration receipt
vestige gc --dry-run                  # List records below a retention threshold; deletion is refused
vestige dashboard                     # Serve http://127.0.0.1:3927
vestige serve                         # Standalone HTTP MCP server (--dashboard to add the dashboard)
```

Every command takes `--data-dir <DIR>`. `vestige consolidate` is a no-op on a Strata log.
`vestige restore <file.json>` re-ingests an export as new records; it does not restore a
backup. `vestige backfill`, `ingest-git`, `portable-export`, `portable-import` and `sync`
work on a legacy SQLite store only. `vestige project` writes only with `--write`, and
always pass `--out`, because the default target is `./CLAUDE.md`.

---

## Client Configuration

### Codex (One-liner)

```bash
codex mcp add vestige -- vestige-mcp
```

### Codex (Manual)

Add to `~/.codex/config.toml`:
```toml
[mcp_servers.vestige]
command = "vestige-mcp"
```

If `vestige-mcp` is not on the PATH Codex sees, use the absolute path from `which vestige-mcp`.

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

Claude Desktop is a GUI app: it does not inherit your shell PATH and does not expand `~` in JSON. After you unpack the release archive, paste the absolute path to `vestige-mcp`.

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>"
    }
  }
}
```

Per-project store: the flag is `--data-dir`, and the directory must be absolute:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": ["--data-dir", "/Users/you/projects/my-app/.vestige"]
    }
  }
}
```

An Intel Mac needs nothing extra. There is no ONNX Runtime to point at, so `ORT_DYLIB_PATH` is not used.

Drop-in skeleton: [`claude-desktop-config.json`](claude-desktop-config.json).

### Claude Desktop (Windows)

Same GUI PATH rule: paste the absolute path from `where vestige-mcp`. Install from the release `.zip` or with `npm install -g vestige-mcp-server`.

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from where vestige-mcp>"
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

For an isolated per-project store, pass the data directory in the command array:

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

See [Storage](STORAGE.md) for more options.

---

## Updating Vestige

`vestige update` downloads the release archive and its `.sha256` file from GitHub,
checks the checksum, and replaces `vestige`, `vestige-mcp`, `vestige-restore` and
`vestige-upgrade` in the directory of the running `vestige` binary. It is the one
command here that uses the network by default, so look first:

```bash
vestige update --dry-run              # Print what would change
vestige update                        # Update to the latest release
vestige update --version v4.1.1       # Install a specific release tag
```

`--install-dir` overrides the destination. Optional Claude Code companion files
(`vestige sandwich install`, `vestige update --sandwich-companion`) are not touched
unless you ask. A running server keeps its old build until its app restarts, so restart
every agent afterward.

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
cargo build --release
```

The default features are `codebase-git` only. `embeddings` and `vector-search` are empty
stubs that select no crates. `connectors` and `cloud-sync` stay off in release builds,
and the release workflow rejects a target feature set that links `reqwest`.
`codebase-git` pulls in libgit2, OpenSSL and libssh2; build without it on targets that
cannot carry them (Android and Termux, see [INSTALL-TERMUX.md](INSTALL-TERMUX.md)), and
the `codebase` tool then reports git history as unavailable.
