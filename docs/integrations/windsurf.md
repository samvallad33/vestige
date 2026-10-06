# Windsurf

> Written for Vestige 4.x.

> Give Cascade a record of what you decided, that carries across sessions.

Windsurf has native MCP support through its Cascade AI. Add Vestige and Cascade can reload your architecture decisions, preferences, and past fixes in every session.

---

## Setup

### 1. Open the config file

**Option A: Via UI:**

1. Open **Windsurf > Settings > Advanced Settings**
2. Scroll to the **"Cascade"** section
3. Click **"view the raw JSON config file"**

**Option B: Direct path:**

| Platform | Path |
|----------|------|
| macOS / Linux | `~/.codeium/windsurf/mcp_config.json` |
| Windows | `%USERPROFILE%\.codeium\windsurf\mcp_config.json` |

```bash
# macOS / Linux
open -e ~/.codeium/windsurf/mcp_config.json
```

### 2. Add Vestige

Install Vestige first: [Install](../REFERENCE.md#install). Run `which vestige-mcp` and paste what it prints; the install puts the binary in `~/.local/bin`.

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
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
      "command": "${env:HOME}/.local/bin/vestige-mcp",
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
      "command": "<absolute path from where vestige-mcp>",
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

> "Remember that this project uses Next.js 15 with the App Router and Drizzle ORM. Tag it `stack`."

Start a **new Cascade session**, then:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige 4.x finds a memory by an exact handle (its id or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Project-Specific Data Directory

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": ["--data-dir", "${env:HOME}/projects/my-app/.vestige"],
      "env": {}
    }
  }
}
```

---

## Important: Tool Limit

Windsurf has a **hard cap of 100 tools** across all MCP servers. Vestige uses 16 tools, leaving plenty of room for other servers.

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

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.
