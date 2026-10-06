# VS Code (GitHub Copilot)

> Written for Vestige 4.x.

> Give Copilot a record of what you decided, that carries across sessions.

VS Code supports MCP servers through GitHub Copilot's agent mode. Vestige plugs directly in, so Copilot can reload your decisions and past fixes in every coding session.

---

## Prerequisites

- **VS Code 1.99+** (or latest stable)
- **GitHub Copilot** extension installed and active
- **vestige-mcp** binary installed ([Installation guide](../REFERENCE.md#install))

---

## Setup

### 1. Create the config file

**Workspace (recommended, shareable with the team):**

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
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {}
    }
  }
}
```

> **Use absolute paths.** Run `which vestige-mcp` and paste what it prints. The README install puts the binary in `~/.local/bin`.

**Windows:**
```json
{
  "servers": {
    "vestige": {
      "command": "<absolute path from where vestige-mcp>",
      "args": [],
      "env": {}
    }
  }
}
```

### 3. Verify

VS Code auto-detects config changes, so no restart is needed.

Open **Copilot Chat** (agent mode) and ask:

> "What MCP tools do you have?"

Vestige's tools (recall, smart_ingest, memory, etc.) should appear.

---

## First Use

In Copilot Chat:

> "Remember that this project uses Express.js with PostgreSQL and follows REST conventions. Tag it `stack`."

Start a **new chat**, then:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige 4.x finds a memory by an exact handle (its id or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Prompt for the data directory (optional)

VS Code supports input variables, so a value such as the data directory is asked for instead of hardcoded:

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
      "command": "<absolute path from which vestige-mcp>",
      "args": ["--data-dir", "${input:vestige-data-dir}"],
      "env": {}
    }
  }
}
```

---

## Share the Config With Your Team

Since `.vscode/mcp.json` lives in the project, you can commit it:

```bash
git add .vscode/mcp.json
git commit -m "Add Vestige MCP config for Copilot"
```

Every team member with Vestige installed gets the same MCP entry. Each person keeps their own local store.

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

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.
