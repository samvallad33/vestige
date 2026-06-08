# Windsurf

> Give Cascade a brain that remembers between sessions.

Windsurf has native MCP support through its Cascade AI. Add Vestige and Cascade remembers your architecture, preferences, and past decisions across every session.

---

## Setup

### 1. Open the config file

**Option A — Via UI:**

1. Open **Windsurf > Settings > Advanced Settings**
2. Scroll to the **"Cascade"** section
3. Click **"view the raw JSON config file"**

**Option B — Direct path:**

| Platform | Path |
|----------|------|
| macOS / Linux | `~/.codeium/windsurf/mcp_config.json` |
| Windows | `%USERPROFILE%\.codeium\windsurf\mcp_config.json` |

```bash
# macOS / Linux
open -e ~/.codeium/windsurf/mcp_config.json
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

**With environment variable expansion** (Windsurf-specific feature):

```json
{
  "mcpServers": {
    "vestige": {
      "command": "${env:HOME}/.cargo/bin/vestige-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

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

### 3. Restart Windsurf

Restart the IDE or refresh the Cascade panel.

### 4. Verify

Open Cascade and ask:

> "What MCP tools do you have?"

You should see Vestige's tools listed.

---

## First Use

In Cascade:

> "Remember that this project uses Next.js 15 with the App Router and Drizzle ORM"

Start a **new Cascade session**, then:

> "What framework does this project use?"

It remembers.

---

## Project-Specific Memory

```json
{
  "mcpServers": {
    "vestige": {
      "command": "/usr/local/bin/vestige-mcp",
      "args": ["--data-dir", "${env:HOME}/projects/my-app/.vestige"],
      "env": {}
    }
  }
}
```

---

## Important: Tool Limit

Windsurf has a **hard cap of 100 tools** across all MCP servers. Vestige uses 25 tools, leaving plenty of room for other servers.

---

## Troubleshooting

<details>
<summary>Vestige not appearing in Cascade</summary>

1. Verify the config file is valid JSON:
   ```bash
   cat ~/.codeium/windsurf/mcp_config.json | python3 -m json.tool
   ```
2. Ensure you're using absolute paths (or `${env:HOME}` expansion).
3. Check the Cascade panel for error messages.
4. Fully restart Windsurf.
</details>

<details>
<summary>Tool limit exceeded</summary>

If you have many MCP servers and exceed 100 total tools, Cascade will ignore excess servers. Remove unused servers or use Vestige's unified tools (each handles multiple operations).
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Xcode 26.3 | [Setup](./xcode.md) |
| Cursor | [Setup](./cursor.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| Codex | [Setup](./codex.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Your AI remembers everything, everywhere.
