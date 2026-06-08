# Xcode 26.3

> Give Xcode's AI agent a brain that remembers.

Xcode 26.3 supports [agentic coding](https://developer.apple.com/documentation/xcode/giving-agentic-coding-tools-access-to-xcode) with full MCP (Model Context Protocol) integration. Vestige plugs directly into Xcode's Claude Agent, giving it persistent memory across every coding session.

**Vestige is the first cognitive memory server for Xcode.**

---

## Quick Start (30 seconds)

### 1. Install Vestige

```bash
npm install -g vestige-mcp-server@latest
```

### 2. Add to your Xcode project

Create a `.mcp.json` file in your project root:

```bash
cat > /path/to/your/project/.mcp.json << 'EOF'
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "vestige-mcp",
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

Type `/context` in the Agent panel. You should see `vestige` listed with 25 tools.

---

## Why `.mcp.json` instead of the global config?

Xcode 26.3's Claude Agent has a feature gate (`claudeai-mcp`) that blocks custom MCP servers configured in the global config at `~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig/.claude`.

**Project-level `.mcp.json` files bypass this gate entirely.** This is the method that actually works. Drop the file in your project root and Xcode loads it on the next session.

> **Important:** Xcode runs agents in a sandboxed environment that does **not** inherit your shell configuration (`.zshrc`, `.bashrc`, etc.). You **must** use absolute paths for the `command` field.

---

## What Vestige Does for Xcode

| Without Vestige | With Vestige |
|-----------------|--------------|
| Every session starts from zero | Agent recalls your architecture, patterns, and preferences |
| Re-explain SwiftUI conventions each time | Agent knows your conventions from day one |
| Bug fixes are forgotten | Agent remembers past fixes and avoids regressions |
| No context between Xcode and other IDEs | Shared memory across Xcode, Cursor, VS Code, and more |
| AI hallucinations persist forever | Agent detects and self-corrects bad memories |

### Example Workflows

**Architecture decisions:**
> "Remember: we chose Observation framework over Combine for state management because it's simpler and Apple-recommended for iOS 17+."

**Bug documentation:**
> The agent fixes a Core Data migration crash? Vestige automatically stores the fix. Next time it encounters a migration issue, it remembers the solution.

**Proactive reminders:**
> The agent surfaces your pending deadlines, hackathon dates, and concert tickets — right inside Xcode's Agent panel.

**Self-correcting memory:**
> The agent traces a hallucinated detail back to a specific memory, identifies it as wrong, and deletes it autonomously.

**Cross-IDE memory:**
> Fix a backend bug in VS Code. Open the iOS app in Xcode. The agent already knows about the API change because Vestige shares memory across all your tools.

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
      "command": "/usr/local/bin/vestige-mcp",
      "args": [],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
EOF
```

### Per-project isolated memory

To give each project its own memory database:

```json
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "/usr/local/bin/vestige-mcp",
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

### Use a CLAUDE.md for proactive memory

Place a `CLAUDE.md` in your project root to make the agent use Vestige automatically:

```markdown
## Memory

At the start of every session:
1. Search Vestige for this project's context
2. Recall architecture decisions and coding patterns
3. Save important decisions and bug fixes without being asked
```

See [CLAUDE.md templates](../CLAUDE-SETUP.md) for a full setup.

### Embedding model cache

The first time Vestige runs, it downloads the embedding model (~130MB). In Xcode's sandboxed environment, the cache location is:

```
~/Library/Caches/vestige/fastembed
```

If the download fails behind a corporate proxy, pre-download by running `vestige-mcp` once from your terminal.

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
   ls -la /usr/local/bin/vestige-mcp
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

<details>
<summary>Embedding model fails to download</summary>

The first run downloads ~130MB. If Xcode's sandbox blocks the download:

1. Run `vestige-mcp` once from your terminal to cache the model
2. The cache at `~/Library/Caches/vestige/fastembed` will be available to the sandboxed instance

Behind a proxy:
```bash
HTTPS_PROXY=your-proxy:port vestige-mcp
```

</details>

---

## Also Works With

Vestige uses the MCP standard — the same memory works across all your tools:

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

Your AI remembers everything, everywhere.

---

<p align="center">
  <a href="../../README.md">Back to README</a>
</p>
