# Cursor

> Give Cursor a brain that remembers between sessions.

Cursor has native MCP support. Add Vestige and your AI assistant remembers your architecture, preferences, and past fixes across every session.

---

## Setup

After `npm install -g vestige-mcp-server@latest`, the `vestige-mcp` binary is on your **shell** PATH. Cursor's GUI does not reliably inherit that PATH and does not expand `~`. Paste the absolute path; do not guess `/usr/local/bin`.

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

nvm, fnm, and Homebrew npm almost never install into `/usr/local/bin`. Paste whatever the command above prints.

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

**Windows:** same shape. Official install is npm, not cargo — paste the absolute path from `where vestige-mcp`. A `.cargo\bin` path is only correct if you built from source.

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

**Intel Mac:** Cursor does not inherit `.zshrc`. Put `ORT_DYLIB_PATH` in this same `env` block. Run `brew --prefix onnxruntime` and paste the result — do not hardcode `/opt/homebrew` vs `/usr/local`. See [Intel Mac install](../INSTALL-INTEL-MAC.md).

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {
        "ORT_DYLIB_PATH": "<brew --prefix onnxruntime>/lib/libonnxruntime.dylib"
      }
    }
  }
}
```

### 3. Restart Cursor

Fully quit and reopen Cursor. The MCP server loads on startup.

### 4. Verify

Open Cursor's AI chat and ask:

> "What MCP tools do you have access to?"

You should see Vestige's tools listed (`smart_ingest`, `recall`, `backfill`).

---

## First Use

Ask Cursor's AI:

> "Remember that this project uses React with TypeScript and Tailwind CSS"

Start a **new chat session**, then:

> "What tech stack does this project use?"

It remembers.

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
