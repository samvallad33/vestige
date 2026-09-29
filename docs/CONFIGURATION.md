# Configuration Reference

> Environment variables, CLI commands, and setup options

---

## First run

4.0 downloads nothing. There is no embedding model, no reranker, and no
network fetch on first use — the store is the STRATA log on your machine.
A v3 SQLite store is refused at startup with the migration hint; see
[MIGRATING-TO-4.0.md](MIGRATING-TO-4.0.md).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_DATA_DIR` | OS per-user data directory | Storage directory fallback; overridden by `--data-dir`; a migrated STRATA directory defaults to `<dir>/strata` |
| `RUST_LOG` | `info` (via tracing-subscriber) | Log verbosity + per-module filtering |
| `VESTIGE_DASHBOARD_PORT` | `3927` | Dashboard HTTP + WebSocket port |
| `VESTIGE_HTTP_ENABLED` | `false` | Set `true` or `1` to enable optional MCP-over-HTTP |
| `VESTIGE_HTTP_PORT` | `3928` | Optional MCP-over-HTTP port; `--http-port` also enables HTTP |
| `VESTIGE_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `VESTIGE_HTTP_ALLOWED_ORIGINS` | localhost origins for the HTTP port | Comma-separated browser origins allowed to call MCP-over-HTTP |
| `VESTIGE_AUTH_TOKEN` | auto-generated | Dashboard + MCP HTTP bearer auth |
| `VESTIGE_DASHBOARD_ENABLED` | `false` | Set `true` or `1` to enable the web dashboard |
| `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` | `6` | FSRS-6 decay cycle cadence |
| `VESTIGE_FAILURE_FEEDBACK` | `off` | Post-retrieval failure feedback (2.8.0): when a failure-shaped memory is ingested, the memories retrieved in the previous thirty minutes of receipts lose retrieval strength by rank (at most 0.10 each, floor 0.05, same scope, once per failure). Opt in with `1`/`true`/`on`/`yes`. Every delta is written to the `failure_feedback` ledger and `revert_failure_feedback` undoes it exactly. |
| `VESTIGE_AUTO_CONSOLIDATE_MERGE` | `off` | Auto concat-merge of near-duplicate memories during consolidation (keeps the strongest, folds the rest in as `[MERGED]` blocks, deletes the originals). **Off by default since v2.6.0** — unattended destruction is opt-in: set `1`/`true`/`on`/`yes` to enable; anything else (including typos) stays off. Protected (`dedup protect`) memories are never absorbed or deleted by this pass. The `dedup` tool remains the previewable, reversible path. |
| `VESTIGE_TRACE` | `on` | Agent Black Box trace recording. **On by default**: every MCP tool call writes rows to `agent_traces`/`agent_runs` in your local database. Set `0`/`false`/`off`/`no` to turn the recorder off. Read once per process, so changing it mid-process has no effect |
| `VESTIGE_TRACE_RETENTION_DAYS` | `30` | How long Black Box traces are kept. The consolidation cycle deletes trace events older than this and drops any `agent_runs` roll-up left with no events. `0` keeps traces forever (sweep disabled); unset, empty, negative, or malformed values fall back to `30` |

> **Storage location precedence:** `--data-dir <path>` wins over `VESTIGE_DATA_DIR`; if neither is set, Vestige uses your OS's per-user data directory: `~/Library/Application Support/com.vestige.core/` on macOS, `~/.local/share/vestige/core/` on Linux, `%APPDATA%\vestige\core\` on Windows. Custom paths are directories, are created if missing, and expand a leading `~`. 4.0 never creates a SQLite store and refuses to open one; a migrated STRATA directory defaults to `<dir>/strata`.

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

## Review Modes (Memory PR write gating)

Vestige applies memory writes automatically by default, without an approval
step. Users who choose review can hold flagged writes. Each held write is suppressed (excluded from normal retrieval) and
opens a **Memory PR** you decide in the dashboard (Memory PRs tab) or via
`GET /api/memory-prs`.

| Mode | Behavior |
|------|----------|
| `fast` | **Default (Automatic).** Every write auto-commits without approval. |
| `risk_gated` | **Opt-in.** Ordinary writes auto-commit; risky ones (contradicting high-trust memories, destructive ops, sensitive topics) open a Memory PR. A write counts as touching a sensitive topic when a tag names it, a credential-shaped value sits next to it, the write is short, the topic leads the text, or two distinct topics appear. One sensitive word buried in a long note is a mention, not a subject, and does not gate. |
| `paranoid` | **Opt-in.** Gate every write. Nothing enters the brain without approval. |

The mode is stored in `<data_dir>/review_mode.json` and set from the dashboard
(`POST /api/memory-prs/mode`). A missing or corrupt file falls back to
`fast`; malformed settings produce a warning. Invalid API mode names are rejected
without changing the saved setting. Existing valid settings survive upgrades.
Select **Automatic (default)** in Memory PRs to stop holding future writes; this
does not apply historical pending proposals.

When a normal risky write is gated, the tool response carries `memoryPrs` and a
`memoryPrNotice` describing the quarantine. Confirmed purge/delete calls and
direct suppression are different: in `risk_gated` and `paranoid` modes Vestige
durably opens a pending Memory PR **before** the mutation and returns without
changing the memory. If the PR cannot be saved, the call fails closed.

For a pending destructive PR, `forget` approves and executes the requested
purge or suppression, `promote` keeps the memory unchanged, and `quarantine`
keeps the row but suppresses it. `fast` is the default direct-execution mode. Purge still requires
`confirm=true`; automatic memory writes do not remove that tool contract.

> **Note:** `VESTIGE_TRACE=0` disables Black Box trace/receipt recording, but it
> does not disable this pre-execution safety gate. Review mode, not tracing,
> controls destructive mutation policy.

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

1. **Explicit MCP parameter** (e.g. `detail_level` / `limit` on a `recall`
   call) — always wins.
2. **`vestige.toml`** — the `[defaults]` keys and the selected profile.
3. **Built-in default** — the `default` profile, identical to pre-v2.1.26
   behavior.

### Affected tools

`recall`, `memory_status` (`timeline` view), `codebase` (`get_context`), and `session_start`
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
VESTIGE_DATA_DIR="$HOME/.vestige" vestige-mcp # Env fallback (shell); GUI JSON does not expand ~
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

Claude Desktop is a GUI app: it does not inherit your shell PATH and does not expand `~` in JSON. After you unpack the GitHub Release archive, paste the absolute path to `vestige-mcp`.

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

Per-project memory — live storage flag is `--data-dir`, and the directory must be absolute:

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

**Intel Mac:** Claude Desktop does not inherit `.zshrc`. Put `ORT_DYLIB_PATH` in the MCP `env` block. Run `brew --prefix onnxruntime` and paste the result — do not hardcode `/opt/homebrew` vs `/usr/local`. See [Intel Mac install](INSTALL-INTEL-MAC.md).

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {
        "ORT_DYLIB_PATH": "<brew --prefix onnxruntime>/lib/libonnxruntime.dylib"
      }
    }
  }
}
```

Drop-in skeleton: [`claude-desktop-config.json`](claude-desktop-config.json).

### Claude Desktop (Windows)

Same GUI PATH rule: paste the absolute path from `where vestige-mcp`. Official install is npm, not cargo.

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
vestige update --version v3.1.1
```

**Manage the optional Cognitive Sandwich layer without updating binaries:**
```bash
vestige sandwich install
vestige sandwich install --enable-preflight
```

**Check your version:**
```bash
vestige-mcp --version
```

---

## Development

### Building without the legacy engine

Android/Termux targets drop `codebase-git` (libgit2, OpenSSL, libssh2) so the
build does not need those C libraries; the `codebase` tool reports git history
as unavailable. 4.0 has no embedding features to drop — `embeddings` and
`vector-search` are declared always-off stubs kept only so legacy
`#[cfg(feature = ...)]` sites compile.

```bash
cargo build --release -p vestige-mcp --no-default-features --features connectors,cloud-sync
```

`recall` is exact-handle only in every build. See [INSTALL-TERMUX.md](INSTALL-TERMUX.md).

```bash
# Run tests
cargo test --all-features

# Run with logging
RUST_LOG=debug cargo run --release

# Build optimized binary
cargo build --release --all-features
```
