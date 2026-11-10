# Configuration Reference

> Environment variables, CLI commands, and setup options for Vestige v6.2.1

---

## Offline by default

A default build has no embedding model, so there is **no first-run download**, and it
makes no startup version check. Nothing is fetched when it starts. The calls that can
use the network are all explicit:

| Call | Network use | In a release build |
|------|-------------|--------------------|
| `vestige update` | Downloads a release archive and its `.sha256` from GitHub, verifies the checksum, replaces the binaries | Yes, only when you run it |
| `vestige backup` | Creates an owner-only, Ed25519-sealed segment backup on local disk | No network |
| `causal_walk(action="prove")` | Runs deterministic local bisect and generates cryptographic proof cards | No network |

The dashboard listens on `127.0.0.1` only. The optional HTTP transport binds `127.0.0.1`
unless you set `VESTIGE_HTTP_BIND`.

---

## Environment Variables

These are read by the default v6.2 build:

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_DATA_DIR` | OS per-user data directory | Data directory fallback; overridden by `--data-dir`. The Strata log lives in `<dir>/log/` |
| `RUST_LOG` | `info` | Log verbosity and per-module filtering. Logs go to stderr, because stdout carries the protocol |
| `VESTIGE_DASHBOARD_PORT` | `3927` | Dashboard HTTP and WebSocket port. `vestige dashboard --port` sets it for one run |
| `VESTIGE_DASHBOARD_ENABLED` | `false` | Set `true` or `1` to have `vestige-mcp` serve the web dashboard for as long as it runs |
| `VESTIGE_HTTP_ENABLED` | `false` | Set `true` or `1` to enable the optional MCP-over-HTTP transport. `--http` does the same |
| `VESTIGE_HTTP_PORT` | `3928` | MCP-over-HTTP port. `--http-port` also enables HTTP |
| `VESTIGE_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `VESTIGE_HTTP_ALLOWED_ORIGINS` | localhost origins for the HTTP port | Comma-separated browser origins allowed to call MCP-over-HTTP |
| `VESTIGE_AUTH_TOKEN` | auto-generated | Bearer token for the HTTP transport |
| `VESTIGE_STATE_TTL_DAYS` | `30` | Lifetime of a `state` record when you give no `validUntil`. `0` disables the default |

See [UPGRADING.md](UPGRADING.md) for the table of variables and v3 SQLite subcommands retired in v6.2.

---

## CLI Commands

In v6.2, the host-boundary commands manage the process lifecycle, while analysis and
verification commands automatically route over `~/.vestige/reflex.sock` when `vestige-mcp`
is active to eliminate `.serve.lock` collisions.

### Host-Boundary Commands

```bash
vestige serve                         # Starts vestige-mcp and binds reflex.sock
vestige reflex <event>                # Sub-millisecond reflex hook client
vestige backup <new-folder>           # Owner-only Ed25519 segment sealed copy
vestige update                        # Updates Vestige binaries from latest GitHub release
vestige dashboard                     # Serves http://127.0.0.1:3927
```

### Zero-Lock IPC & Verification Commands

```bash
vestige stats                         # Counts, due for review, average retention
vestige health                        # Health check with warnings and recommendations
vestige prove --logged-write <id> --repo <dir> --good <ref> --bad <ref> --test '<cmd>'
                                      # Deterministic bisect and signed cryptographic proof card
vestige strata-verify <data-dir>      # Live BLAKE3 & Ed25519 segment cryptographic verification
vestige scan-secrets                  # Audit the log and workspace for credential leaks
vestige connect --dry-run             # Join memories recording the exact same byte identities
vestige project --repo <dir>          # Lock-free CLAUDE.md / AGENTS.md rule fence compilation
vestige selftest                      # Self-test verification suite
```

---

## Client Configuration

### Claude Code

```bash
claude mcp add vestige vestige-mcp -s user
```

### Cursor

Add to `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### VS Code (Copilot)

Add to `.vscode/mcp.json` or user settings:
```json
{
  "servers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### OpenCode

Add to `opencode.json`:
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
