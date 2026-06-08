# JetBrains (IntelliJ, WebStorm, PyCharm, etc.)

> Give your JetBrains AI assistant a brain that remembers.

JetBrains IDEs (2025.2+) have built-in MCP support. Vestige integrates through the MCP server settings, giving your AI assistant persistent memory across sessions.

---

## Prerequisites

- **JetBrains IDE 2025.2+** (IntelliJ IDEA, WebStorm, PyCharm, GoLand, etc.)
- **vestige-mcp** binary installed ([Installation guide](../../README.md#quick-start))

---

## Setup

### Option A: Auto-Configure (Recommended)

JetBrains can auto-configure MCP servers for connected clients:

1. Open **Settings** (`Cmd+,` / `Ctrl+Alt+S`)
2. Navigate to **Tools > MCP Server**
3. Click **"+"** to add a new MCP server
4. Configure:
   - **Name:** `vestige`
   - **Command:** `/usr/local/bin/vestige-mcp`
   - **Arguments:** (leave empty)
5. Click **Apply**

### Option B: Junie AI Config

If using JetBrains Junie AI, add Vestige to the Junie MCP config:

**User-level (all projects):**

```bash
mkdir -p ~/.junie/mcp
```

Edit `~/.junie/mcp/mcp.json`:

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

**Project-level:**

Create `.junie/mcp/mcp.json` in your project root with the same format.

### Option C: Via External Client

JetBrains exposes its own tools via MCP. You can also use Vestige through an external client (Claude Code, Cursor) that connects to JetBrains:

1. In JetBrains: **Settings > Tools > MCP Server**
2. Click **Auto-Configure** for your preferred client
3. Add Vestige to that client's config (see [Cursor](./cursor.md), [VS Code](./vscode.md))

---

## Verify

After configuration, the MCP server should appear in **Settings > Tools > MCP Server** with a green status indicator.

Test by asking your AI assistant:

> "Remember that this project uses Spring Boot with Kotlin and follows hexagonal architecture"

---

## Project-Specific Memory

Isolate memory per project:

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

---

## Troubleshooting

<details>
<summary>MCP server not appearing in settings</summary>

1. Verify your IDE version is 2025.2 or later.
2. Check that the binary path is absolute:
   ```bash
   which vestige-mcp
   ```
3. Restart the IDE after adding the configuration.
</details>

<details>
<summary>Finding your client's config file</summary>

In **Settings > Tools > MCP Server**, click the expansion arrow next to your client, then select **"Open Client Settings File"** to see the exact config path.
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
| Windsurf | [Setup](./windsurf.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Your AI remembers everything, everywhere.
