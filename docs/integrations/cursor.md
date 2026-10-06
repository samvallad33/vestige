# Cursor

> Written for Vestige 4.x.

> Give Cursor a brain that remembers between sessions.

Cursor has native MCP support. Add Vestige and your AI assistant remembers your architecture, preferences, and past fixes across every session.

---

## Setup

Install Vestige first: [Install](../REFERENCE.md#install). Cursor's GUI does not reliably inherit that PATH and does not expand `~`. Paste the absolute path; do not guess `/usr/local/bin`.

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

### 2. Resolve the binary, then add Vestige

```bash
which vestige-mcp          # macOS / Linux
where vestige-mcp          # Windows
```

The README install puts the binaries in `~/.local/bin`, and Homebrew puts them under its own prefix. Paste whatever the command above prints.

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": []
    }
  }
}
```

**Windows:** same shape. Install from the release zip (see the README), then paste the absolute path from `where vestige-mcp`. Do not install 4.x with npm: the npm package still serves 3.0.0.

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from where vestige-mcp>",
      "args": []
    }
  }
}
```

**Intel Mac:** nothing extra. Vestige 4.x has no ONNX runtime, so no `ORT_DYLIB_PATH` is needed.

### 3. Restart Cursor

Fully quit and reopen Cursor. The MCP server loads on startup.

### 4. Verify

Open Cursor's AI chat and ask:

> "What MCP tools do you have access to?"

You should see Vestige's 16 tools listed, including `smart_ingest`, `recall` and `receipt`.

---

## First Use

Ask Cursor's AI:

> "Remember that this project uses React with TypeScript and Tailwind CSS. Tag it `stack`."

Start a **new chat session**, then:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige 4.x finds a memory by an exact handle (its id or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Project-Specific Memory

To isolate memory per project, pass `--data-dir` with an **absolute** directory (Cursor does not expand `~` or relative paths in `args`):

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": ["--data-dir", "/Users/you/projects/my-app/.vestige"]
    }
  }
}
```

Or place a `.cursor/mcp.json` in the project root for project-level config.

---

## Troubleshooting

<details>
<summary>Vestige tools not appearing</summary>

1. Verify the binary exists and copy that exact path into `command`:
   ```bash
   which vestige-mcp          # macOS / Linux
   where vestige-mcp          # Windows
   ```
2. Test the binary manually. It should print `vestige-mcp 4.1.0` or newer:
   ```bash
   vestige-mcp --version
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
