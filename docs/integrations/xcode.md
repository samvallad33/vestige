# Xcode 26.3

> Written for Vestige 4.x.

> Give Xcode's AI agent a record of what you decided, that carries across sessions.

Xcode 26.3 supports [agentic coding](https://developer.apple.com/documentation/xcode/giving-agentic-coding-tools-access-to-xcode) with full MCP (Model Context Protocol) integration. Vestige plugs into Xcode's Claude Agent, so it can reload your decisions and past fixes in every coding session.

---

## Quick Start (30 seconds)

### 1. Install Vestige

Install Vestige as described in the [README](../../README.md#install). Xcode's agent does not read your shell PATH, so note the absolute path that `which vestige-mcp` prints.

### 2. Add to your Xcode project

Create a `.mcp.json` file in your project root:

```bash
cat > /path/to/your/project/.mcp.json << 'EOF'
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
EOF
```

Or use the setup script:

```bash
curl -sSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/xcode-setup.sh -o xcode-setup.sh
bash xcode-setup.sh
```

### 3. Restart Xcode

Quit Xcode completely (Cmd+Q) and reopen your project.

### 4. Verify

Type `/context` in the Agent panel. You should see `vestige` listed with 16 tools.

---

## Why `.mcp.json` instead of the global config?

Xcode 26.3's Claude Agent has a feature gate (`claudeai-mcp`) that blocks custom MCP servers configured in the global config at `~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig/.claude`.

**Project-level `.mcp.json` files bypass this gate entirely.** This is the method that actually works. Drop the file in your project root and Xcode loads it on the next session.

> **Important:** Xcode runs agents in a sandboxed environment that does **not** inherit your shell configuration (`.zshrc`, `.bashrc`, etc.). You **must** use absolute paths for the `command` field.

---

## What Vestige Does for Xcode

| Without Vestige | With Vestige |
|-----------------|--------------|
| Every session starts from zero | Agent recalls your architecture, patterns, and preferences by tag |
| Re-explain SwiftUI conventions each time | Agent reloads your conventions from the tag you saved them under |
| Bug fixes are forgotten | Agent can recall past fixes and walk a failure back along recorded links |
| No context between Xcode and other IDEs | One store shared across Xcode, Cursor, VS Code, and more |
| A wrong record keeps resurfacing | Agent demotes it and saves a corrected successor |

### Example Workflows

**Architecture decisions:**
> "Remember: we chose Observation framework over Combine for state management because it's simpler and Apple-recommended for iOS 17+. Tag it `swiftui-state`."

**Bug documentation:**
> The agent fixes a Core Data migration crash and saves the fix with `smart_ingest`, tagged `core-data`. Next time you hit a migration issue, ask it to recall the `core-data` tag.

**Reminders:**
> An intention with a deadline or a trigger shows up when `session_start` runs, right inside Xcode's Agent panel.

**Correcting a record:**
> The agent finds the record behind a wrong detail by its id, demotes it, and saves a corrected successor with `memory` action `edit`. Nothing is deleted from the log.

**One store across IDEs:**
> Fix a backend bug in VS Code. Open the iOS app in Xcode. The agent can recall the tag you saved the API change under, because every agent on the machine shares one store.

---

## Add Vestige to Every Project

Run the setup script with `a` to install into all detected projects:

```bash
curl -sSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/xcode-setup.sh -o xcode-setup.sh
bash xcode-setup.sh
```

Or manually drop `.mcp.json` into any project:

```bash
# From inside your project directory
cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
EOF
```

### Per-project data directory

To give each project its own store:

```json
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "<absolute path from which vestige-mcp>",
      "args": ["--data-dir", "/Users/you/Developer/MyApp/.vestige"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

---

## Tips

### Use a CLAUDE.md so the agent calls Vestige

Place a `CLAUDE.md` in your project root to make the agent use Vestige without being asked:

```markdown
## Vestige

At the start of every session:
1. Call `session_start` with the project's `context.codebase` and `context.repoPath`
2. Call `recall` with `handle` set to this project's topic tag
3. Save important decisions and bug fixes with `smart_ingest`, tagged with the project and one topic
```

See [CLAUDE.md templates](../CLAUDE-SETUP.md) for a full setup.

---

## Troubleshooting

<details>
<summary>"vestige" not showing in /context</summary>

1. Make sure `.mcp.json` is in your **project root** (same directory as `.xcodeproj` or `Package.swift`):
   ```bash
   ls -la /path/to/project/.mcp.json
   ```

2. Verify the binary path is correct and absolute:
   ```bash
   ls -la "$(which vestige-mcp)"
   ```

3. Check that `.mcp.json` is valid JSON:
   ```bash
   cat /path/to/project/.mcp.json | python3 -m json.tool
   ```

4. Fully quit and restart Xcode (Cmd+Q, not just close window).

5. Check debug logs:
   ```bash
   cat ~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig/debug/latest | grep -i vestige
   ```

</details>

<details>
<summary>"Agent has been closed" or "Your request couldn't be completed"</summary>

This is a known issue with Xcode 26.3's Claude Agent that can happen independently of MCP configuration.

**Nuclear fix:** Delete the agent config and let Xcode recreate it:
```bash
mv ~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig \
   ~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig.bak
```
Then restart Xcode and sign back into Claude in Settings > Intelligence.

</details>

<details>
<summary>Global .claude config not loading MCP servers</summary>

Xcode 26.3 has a feature gate (`claudeai-mcp`) that may block custom MCP servers from the global config file at `~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig/.claude`.

**Solution:** Use project-level `.mcp.json` instead. This bypasses the gate. See the [Quick Start](#quick-start-30-seconds) above.

</details>

---

## Also Works With

Vestige uses the MCP standard, so the same store works across all your tools:

| IDE | Guide |
|-----|-------|
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Codex | [Setup](./codex.md) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |
| Cursor | [Setup](./cursor.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.

---

<p align="center">
  <a href="../../README.md">Back to README</a>
</p>
