# I Gave Xcode's AI Agent a Memory. It Remembered My Deadlines.

*Xcode 26.3's AI coding assistant forgets everything between sessions. I fixed that in 30 seconds with one file.*

---

Xcode 26.3 shipped with something big: native support for AI coding agents. Claude and Codex can now read your project, run builds, write code, and iterate on fixes — all inside Xcode.

But there's a problem nobody's talking about: **it forgets everything.**

Every time you start a new session, the agent has zero context. Your architecture decisions? Gone. That bug you spent an hour debugging yesterday? Gone. Your coding conventions, project patterns, library preferences? All gone. You start from scratch every single time.

I fixed this with one file and 30 seconds.

## The Setup

[Vestige](https://github.com/samvallad33/vestige) is a cognitive memory system built on 130 years of memory research. It uses FSRS-6 spaced repetition (the algorithm behind modern Anki), prediction error gating, synaptic tagging, and spreading activation — all running in a single Rust binary, 100% local.

It speaks MCP (Model Context Protocol), the same protocol Xcode 26.3 uses for tool integration. So connecting them was trivial.

**Step 1:** Install Vestige
```bash
npm install -g vestige-mcp-server
```

**Step 2:** Drop one file in your project root
```bash
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

**Step 3:** Restart Xcode.

That's it. Type `/context` in the Agent panel and you'll see 23 Vestige tools loaded alongside Xcode's built-in tools.

## What Happened Next

I told the agent: *"Remember that SoulVault is the first offline-first journaling app with enhanced encryption and emotional AI, launching in 2026."*

It responded:

> Remembering...

Then it searched Vestige for existing context (`mcp__vestige__search`), checked my intentions (`mcp__vestige__intention`), and since it didn't find SoulVault context yet, it saved the information (`mcp__vestige__smart_ingest`).

Then — without me asking — it surfaced my pending reminders. Product launch deadlines, hackathon dates, even personal events. All pulled from Vestige's memory, displayed right inside Xcode's Agent panel.

My AI coding assistant just pulled my deadlines. Inside my IDE. While I was coding.

## It Gets Better: Self-Correcting Memory

Then I tested something. I asked the agent about a detail I suspected was wrong — a cat named "Whiskers" that had been hallucinated by a previous integration.

The agent:
1. Searched Vestige — didn't find "Whiskers"
2. Searched deeper — found a memory about *planning* to get a cat
3. Traced the hallucination to a specific memory ID from an older integration
4. **Deleted the bad memory on its own**
5. Corrected the record: *"you're planning to get a cat — you don't currently have one"*

The AI debugging its own memory, autonomously, inside Xcode. No other memory system does this — the flat JSON knowledge graphs would just keep the wrong data forever.

## Why This Matters

Every other MCP "memory server" is a JSON file with `create_entity` and `create_relation`. They "remember" the way a notepad remembers — they store text. There's no decay, no consolidation, no deduplication, no self-correction.

Vestige remembers the way a brain does:

- **FSRS-6 spaced repetition** — memories naturally decay and strengthen based on usage, trained on 700M+ reviews
- **Prediction error gating** — automatically deduplicates and decides whether to create, update, or supersede memories
- **Spreading activation** — searching for one memory surfaces related memories without changing their durable strength
- **Synaptic tagging** — important memories get tagged for long-term consolidation
- **23 cognitive tools** — search, ingest, dream, predict, explore connections, garbage collect, and more

All running locally in a single Rust binary. No cloud. No API keys. No data leaves your machine.

## The Technical Gotcha

Here's something I discovered that isn't documented anywhere: Xcode 26.3's Claude Agent may not load custom MCP servers configured in the global config file at `~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig/.claude`. There's an internal feature gate that can block them.

**The solution:** Use a project-level `.mcp.json` file instead. Xcode's agent reliably loads MCP servers from project-root `.mcp.json` files. That's why the setup above uses `.mcp.json` in the project root rather than the global Claude config.

## One More Thing

If Xcode's agent ever gives you "Prompt stream failed: Agent has been closed" — that's usually a corrupted config directory. The nuclear fix:

```bash
mv ~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig \
   ~/Library/Developer/Xcode/CodingAssistant/ClaudeAgentConfig.bak
```

Restart Xcode, sign back in, re-add your `.mcp.json`. Fresh start.

## Try It

The full setup takes 30 seconds:

```bash
# Install Vestige
npm install -g vestige-mcp-server

# Add to your project (run from project root)
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

# Restart Xcode. Done.
```

Or use the one-liner setup script:

```bash
curl -sSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/xcode-setup.sh -o xcode-setup.sh
bash xcode-setup.sh
```

**Your AI coding assistant forgets everything between sessions. Give it a brain.**

[GitHub](https://github.com/samvallad33/vestige) | [Full Xcode Guide](https://github.com/samvallad33/vestige/blob/main/docs/integrations/xcode.md)

---

*Vestige is open source (AGPL-3.0). 100% local. No cloud. No API keys. Your memories never leave your machine.*
