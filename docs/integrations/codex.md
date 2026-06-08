# Codex

> Give Codex a brain that remembers between sessions.

Codex has native MCP support through the `codex mcp` CLI. Add Vestige once and Codex can carry project preferences, architecture decisions, and past fixes across sessions.

---

## Prerequisites

- **Codex CLI** installed and authenticated
- **vestige-mcp** binary installed ([Installation guide](../../README.md#quick-start))

---

## Setup

### 1. Add Vestige

```bash
codex mcp add vestige -- /usr/local/bin/vestige-mcp
```

> **Use an absolute path.** Run `which vestige-mcp` to find the installed binary.

### 2. Verify

```bash
codex mcp list
```

You should see a `vestige` entry with `enabled` status.

### 3. Test it in Codex

Start Codex and ask:

> "What MCP tools do you have access to?"

You should see Vestige's tools listed (`search`, `smart_ingest`, `memory`, and others).

---

## First Use

In Codex:

> "Remember that this project uses Rust with Axum and SQLite"

Start a **new session**, then ask:

> "What stack does this project use?"

It remembers.

---

## Manual Configuration

Codex stores MCP servers in `~/.codex/config.toml`.

Minimal config:

```toml
[mcp_servers.vestige]
command = "/usr/local/bin/vestige-mcp"
```

After saving, restart Codex or start a new session.

---

## Project-Specific Memory

Use `--data-dir` to isolate memory per repo or workspace:

```bash
codex mcp remove vestige
codex mcp add vestige -- /usr/local/bin/vestige-mcp --data-dir /Users/you/projects/my-app/.vestige
```

Equivalent manual config:

```toml
[mcp_servers.vestige]
command = "/usr/local/bin/vestige-mcp"
args = ["--data-dir", "/Users/you/projects/my-app/.vestige"]
```

---

## Intelligent Memory Protocol

MCP registration makes Vestige tools available to Codex. It does not, by itself,
force Codex to call those tools before answering.

For workspaces where Codex should behave like it has persistent cognitive
memory, add an `AGENTS.md` file at the workspace or repo root:

```markdown
Before answering substantive prompts, consult Vestige using the current prompt
plus project and user context. Use `session_context` for broad context, `search`
for quick memory checks, and `deep_reference` for decisions, contradictions, or
accuracy-sensitive questions. Compose memories into actions; do not summarize
retrievals.
```

Then use the full protocol in
[`codex-intelligent-memory.md`](./codex-intelligent-memory.md).

---

## Troubleshooting

<details>
<summary>Vestige tools do not appear in Codex</summary>

1. Verify the server is registered:
   ```bash
   codex mcp list
   ```
2. Check the binary path:
   ```bash
   which vestige-mcp
   ```
3. Ensure the config entry exists in `~/.codex/config.toml`.
4. Start a fresh Codex session after adding the server.
</details>

<details>
<summary>Need to remove or re-add the server</summary>

```bash
codex mcp remove vestige
codex mcp add vestige -- /usr/local/bin/vestige-mcp
```
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Xcode 26.3 | [Setup](./xcode.md) |
| Cursor | [Setup](./cursor.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Your AI remembers everything, everywhere.
