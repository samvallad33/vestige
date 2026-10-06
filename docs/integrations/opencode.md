# OpenCode

> Written for Vestige 4.x.

> Give OpenCode a local record of your decisions across TUI, CLI, and desktop sessions.

OpenCode supports local MCP servers through its `mcp` config. Add Vestige once and your OpenCode agents can reload project decisions, architecture context, preferences, and previous fixes between sessions.

Verified with OpenCode `1.16.2` on June 8, 2026.

---

## Why OpenCode Users Add Vestige

OpenCode is strong at driving real coding work from the terminal. The painful gap is continuity: the next session often has to rediscover what the previous session already learned. Vestige gives OpenCode a local, signed log through MCP, so the agent can reuse the project context that should not be trapped in one chat transcript.

Useful records include:

- project decisions: "we use Axum handlers thinly and keep database logic in storage modules"
- preferences: "prefer small focused PRs and explicit verification receipts"
- architecture context: "the dashboard talks to the MCP server through the Axum backend and WebSocket events"
- bug fixes: "OpenCode rejects `mcpServers`; use top-level `mcp.vestige` with a command array"
- workflow state: "PR #67 was merged, but the config shape needed correction before promotion"

Vestige is local-first. Records are stored in a Strata log on your machine, can be scoped globally or per project, and are found by exact handle (an id or an exact tag) with tools like `vestige_session_start`, `vestige_recall`, and `vestige_smart_ingest`.

---

## Setup

### 1. Install Vestige

Install Vestige as described in the [README](../../README.md#install) and put `vestige-mcp` on your PATH.

Verify the binary:

```bash
vestige-mcp --version
```

Point OpenCode at the unpacked binary:

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

### 2. Add Vestige To OpenCode

For global use across projects, create or edit:

```bash
mkdir -p ~/.config/opencode
${EDITOR:-vi} ~/.config/opencode/opencode.json
```

Add:

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

OpenCode also supports project-local config. Put the same block in `opencode.json` at the repo root when you want the setting checked in with a project.

For a custom config file, set `OPENCODE_CONFIG=/path/to/opencode.json` before launching OpenCode.

### 3. Verify

Restart OpenCode, then validate the resolved config and MCP server list:

```bash
opencode debug config
opencode mcp list
```

You should see `vestige` listed. In a session, ask:

> "What MCP tools can you use?"

Vestige tools should be available with the `vestige_` prefix, such as `vestige_recall`, `vestige_smart_ingest`, and `vestige_session_start`.

---

## First Use

In OpenCode:

> "Remember that this project uses Rust with Axum and SQLite. Tag it `stack`."

Start a new OpenCode session, then ask:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige 4.x finds a record by an exact handle (its id or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Project-Specific Data Directory

To keep a separate store per repo, add `--data-dir` to OpenCode's command array:

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

For an absolute path:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vestige": {
      "type": "local",
      "command": ["<absolute path from which vestige-mcp>", "--data-dir", "/Users/you/projects/my-app/.vestige"],
      "enabled": true,
      "timeout": 10000
    }
  }
}
```

---

## Automatic Setup

If `opencode` is installed or `~/.config/opencode` exists, Vestige's installer can add the global config automatically:

```bash
npx @vestige/init
```

The installer writes a backup before modifying an existing config file. It also migrates Vestige entries copied from older `mcpServers` examples into OpenCode's current `mcp.vestige` shape.

---

## Troubleshooting

<details>
<summary>Vestige tools do not appear</summary>

1. Verify OpenCode can see configured MCP servers:
   ```bash
   opencode debug config
   opencode mcp list
   ```
2. Verify the binary is on your path:
   ```bash
   which vestige-mcp
   ```
3. Use an absolute binary path if OpenCode cannot resolve `vestige-mcp`.
4. Restart OpenCode after changing `opencode.json`.
5. Keep `timeout` at `10000` or higher for installed binaries. If you use the direct `npx` command, use `60000` so the first cold npm download does not fail OpenCode startup.
</details>

<details>
<summary>Config does not validate</summary>

OpenCode uses the top-level `mcp` key. Do not use the `mcpServers` shape from Claude Desktop, Cursor, or Windsurf.

If you copied an older Vestige example that used `mcpServers`, rerun:

```bash
npx @vestige/init
```

Correct:

```json
{
  "mcp": {
    "vestige": {
      "type": "local",
      "command": ["vestige-mcp"],
      "timeout": 10000
    }
  }
}
```
</details>

<details>
<summary>Too many MCP tools in context</summary>

OpenCode loads MCP tools alongside built-in tools. If you have many MCP servers enabled, disable unused servers or restrict MCP tools per agent in your OpenCode config.
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Codex | [Setup](./codex.md) |
| Cursor | [Setup](./cursor.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |
| Xcode 26.3 | [Setup](./xcode.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |
