# VS Code (GitHub Copilot)

> Give Copilot a brain that remembers between sessions.

VS Code supports MCP servers through GitHub Copilot's agent mode. Vestige plugs directly in, giving Copilot persistent memory across every coding session.

---

## Prerequisites

- **VS Code 1.99+** (or latest stable)
- **GitHub Copilot** extension installed and active
- **vestige-mcp** binary installed ([Installation guide](../../README.md#quick-start))

---

## Setup

### 1. Create the config file

**Workspace (recommended — shareable with team):**

Create `.vscode/mcp.json` in your project root:

```bash
mkdir -p .vscode
```

**User-level (all projects):**

Open Command Palette (`Cmd+Shift+P`) and run:

```
MCP: Open User Configuration
```

### 2. Add Vestige

Note: VS Code uses `"servers"` (not `"mcpServers"`).

```json
{
  "servers": {
    "vestige": {
      "command": "/usr/local/bin/vestige-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

> **Use absolute paths.** Run `which vestige-mcp` to find your binary.

**Windows:**
```json
{
  "servers": {
    "vestige": {
      "command": "C:\\Users\\you\\.cargo\\bin\\vestige-mcp.exe",
      "args": [],
      "env": {}
    }
  }
}
```

### 3. Verify

VS Code auto-detects config changes — no restart needed.

Open **Copilot Chat** (agent mode) and ask:

> "What MCP tools do you have?"

Vestige's tools (search, smart_ingest, memory, etc.) should appear.

---

## First Use

In Copilot Chat:

> "Remember that this project uses Express.js with PostgreSQL and follows REST conventions"

Start a **new chat**, then:

> "What's the tech stack for this project?"

It remembers.

---

## Secure API Keys (Optional)

VS Code supports input variables to avoid hardcoding secrets:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "vestige-data-dir",
      "description": "Vestige data directory"
    }
  ],
  "servers": {
    "vestige": {
      "command": "/usr/local/bin/vestige-mcp",
      "args": ["--data-dir", "${input:vestige-data-dir}"],
      "env": {}
    }
  }
}
```

---

## Share Memory Config With Your Team

Since `.vscode/mcp.json` lives in the project, you can commit it:

```bash
git add .vscode/mcp.json
git commit -m "Add Vestige memory server for Copilot"
```

Every team member with Vestige installed will automatically get memory-enabled Copilot.

---

## Troubleshooting

<details>
<summary>Vestige not showing in Copilot</summary>

1. Ensure you're using **agent mode** in Copilot Chat (not inline completions).
2. Verify VS Code version is 1.99+.
3. Check the config file is at `.vscode/mcp.json` (not `.vscode/settings.json`).
4. Verify the key is `"servers"` not `"mcpServers"`.
5. Test the binary manually:
   ```bash
   which vestige-mcp && echo "Found" || echo "Not found"
   ```
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Xcode 26.3 | [Setup](./xcode.md) |
| Cursor | [Setup](./cursor.md) |
| Codex | [Setup](./codex.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Your AI remembers everything, everywhere.
