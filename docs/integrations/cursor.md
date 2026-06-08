# Cursor

> Give Cursor a brain that remembers between sessions.

Cursor has native MCP support. Add Vestige and your AI assistant remembers your architecture, preferences, and past fixes across every session.

---

## Setup

### 1. Create or edit the config file

**Global (all projects):**

| Platform | Path |
|----------|------|
| macOS / Linux | `~/.cursor/mcp.json` |
| Windows | `%USERPROFILE%\.cursor\mcp.json` |

```bash
# macOS / Linux
mkdir -p ~/.cursor
open -e ~/.cursor/mcp.json
```

### 2. Add Vestige

```json
{
  "mcpServers": {
    "vestige": {
      "command": "/usr/local/bin/vestige-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

> **Use absolute paths.** Cursor does not reliably resolve relative paths or `~`. Run `which vestige-mcp` to find your binary location.

**Windows:**
```json
{
  "mcpServers": {
    "vestige": {
      "command": "C:\\Users\\you\\.cargo\\bin\\vestige-mcp.exe",
      "args": [],
      "env": {}
    }
  }
}
```

### 3. Restart Cursor

Fully quit and reopen Cursor. The MCP server loads on startup.

### 4. Verify

Open Cursor's AI chat and ask:

> "What MCP tools do you have access to?"

You should see Vestige's tools listed (search, smart_ingest, memory, etc.).

---

## First Use

Ask Cursor's AI:

> "Remember that this project uses React with TypeScript and Tailwind CSS"

Start a **new chat session**, then:

> "What tech stack does this project use?"

It remembers.

---

## Project-Specific Memory

To isolate memory per project, use `--data-dir`:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "/usr/local/bin/vestige-mcp",
      "args": ["--data-dir", "/Users/you/projects/my-app/.vestige"],
      "env": {}
    }
  }
}
```

Or place a `.cursor/mcp.json` in the project root for project-level config.

---

## Troubleshooting

<details>
<summary>Vestige tools not appearing</summary>

1. Verify the binary exists:
   ```bash
   which vestige-mcp
   ```
2. Test the binary manually:
   ```bash
   echo '{}' | vestige-mcp
   ```
3. Check the config is valid JSON:
   ```bash
   cat ~/.cursor/mcp.json | python3 -m json.tool
   ```
4. Fully restart Cursor (Cmd+Q / Alt+F4, not just close window).
</details>

<details>
<summary>Silent failures</summary>

Cursor does not surface MCP server errors in the UI. Test by running the command directly in your terminal to see actual error output.
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Xcode 26.3 | [Setup](./xcode.md) |
| Codex | [Setup](./codex.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Your AI remembers everything, everywhere.
