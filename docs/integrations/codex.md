# Codex

> Written for Vestige 4.x.

> Give Codex a record of what you decided, that carries across sessions.

Codex has native MCP support through the `codex mcp` CLI. Add Vestige once and Codex can reload project decisions, corrections and past fixes in every session.

---

## Prerequisites

- **Codex CLI** installed and authenticated
- **vestige-mcp** binary installed ([Installation guide](../REFERENCE.md#install))

---

## Setup

### 1. Add Vestige

```bash
codex mcp add vestige -- "$(which vestige-mcp)"
```

> **Use an absolute path.** The README install puts the binaries in `~/.local/bin`, and Homebrew puts them under its own prefix. Run `which vestige-mcp` and use what it prints.

### 2. Verify

```bash
codex mcp list
```

You should see a `vestige` entry with `enabled` status.

### 3. Test it in Codex

Start Codex and ask:

> "What MCP tools do you have access to?"

You should see Vestige's 16 tools listed, including `session_start`, `recall`, `smart_ingest`, `memory` and `receipt`.

---

## First Use

In Codex:

> "Remember that this project uses Rust with Axum. Tag it `stack`."

Start a **new session**, then ask:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige 4.x finds a record by an exact handle (its id or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Manual Configuration

Codex stores MCP servers in `~/.codex/config.toml`.

Minimal config:

```toml
[mcp_servers.vestige]
command = "<absolute path from which vestige-mcp>"
```

After saving, restart Codex or start a new session.

---

## Project-Specific Data Directory

Use `--data-dir` to keep a separate store per repo or workspace:

```bash
codex mcp remove vestige
codex mcp add vestige -- "$(which vestige-mcp)" --data-dir /Users/you/projects/my-app/.vestige
```

Equivalent manual config:

```toml
[mcp_servers.vestige]
command = "<absolute path from which vestige-mcp>"
args = ["--data-dir", "/Users/you/projects/my-app/.vestige"]
```

---

## Agent Protocol

MCP registration makes Vestige tools available to Codex. It does not, by itself, make Codex call them before it answers.

For workspaces where Codex should consult Vestige, add an `AGENTS.md` file at the workspace or repo root:

```markdown
At the start of a task, call Vestige `session_start` with include_intentions and
include_status, and with context.codebase and context.repoPath for the checkout you
are editing. Then call `recall` with handle set to the task's narrow topic tag.
Recall by exact tag or id only. Compose what you find into actions; do not
summarize retrievals.
```

Then use the full protocol in [`codex-intelligent-memory.md`](./codex-intelligent-memory.md).

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
codex mcp add vestige -- "$(which vestige-mcp)"
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

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.
