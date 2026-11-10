# Cursor

> Written for Vestige 6.x.

> Give Cursor a record of what you decided, that carries across sessions.

Cursor has native MCP support. Add Vestige and your AI assistant can reload your architecture decisions, preferences, and past fixes in every session.

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
      "command": "<absolute path from which vestige-mcp or where vestige-mcp>",
      "args": []
    }
  }
}
```

### 3. Restart Cursor

Fully quit and reopen Cursor. The MCP server loads on startup.

### 4. Verify

Open Cursor's AI chat and ask:

> "What MCP tools do you have access to?"

You should see Vestige's 6 tools listed: `session`, `remember`, `recall`, `causal_walk`, `curate`, and `precedent`.

---

## First Use

Ask Cursor's AI:

> "Remember that this project uses React with TypeScript and Tailwind CSS. Tag it `stack`."

Start a **new chat session**, then:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige finds a memory by an exact handle (its id, an 8+ character prefix, or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Agent Protocol & Rules Enforcement

MCP registration makes Vestige tools available to Cursor, but does not alone force Cursor to drive them on every turn.

To enforce continuous Vestige utilization, place the protocol in `.cursorrules` or `.cursor/rules/vestige-protocol.mdc` at your repository root:

```markdown
---
description: Vestige Cognitive Protocol mandatory for all agent sessions
globs: *
alwaysApply: true
---

# ⚡ VESTIGE COGNITIVE PROTOCOL (MANDATORY FOR ALL AGENTS)

You are connected to Vestige (vestige-mcp), a deterministic local safety kernel and causal memory ledger (Strata). While the autonomic Reflex hooks (~/.vestige/reflex.sock) run automatically in the background, you are strictly required to drive the 6 Vestige MCP tools (session, remember, recall, causal_walk, curate, precedent) as your primary cognitive loop.

Never rely on raw context-window guessing, grep/rg fishing, or unrecorded state mutations when a Vestige tool applies.

1. Boot Sequence (First Action of Every Turn/Session)
- MANDATORY FIRST CALL: At the start of any task or session, before reading files or running shell commands, call session(action="start") to boot lifecycle state, check active intentions, and load projected rules.
- Pre-Flight Handle Discovery: Before modifying any module or debugging a subsystem, call recall (using exact handles or discovered_handles) to pull existing architectural invariants and prior decisions.

2. Debugging & Failure Recovery (causal_walk + prove)
Whenever a test fails, a build breaks, a runtime error appears, or an unexpected regression occurs:
- STOP GUESSING: Do NOT immediately edit code or run blind grep searches based on stack-trace text.
- Step 1 — Trace the Root Cause: Immediately call causal_walk pointed at the symptom or failing test (from_handle or start_points) to walk backward across recorded Strata edges (symptom -> failing test -> tool call -> git commit -> memory write) and surface forgotten lessons.
- Step 2 — Prove the Cause: When a suspect commit or historical mutation is identified, invoke causal_walk(action="prove") to run a deterministic bisect witness and verify the root cause before writing a fix.

3. Gated Writes & Retroactive Salience (remember + curate)
Whenever you make a non-trivial architectural decision, discover a root cause, or ship a verified fix:
- Anchor Decisions to Code: Call remember(action="save", ...) to perform a gated write into Strata with explicit codebase anchoring and inline Retroactive Salience Backfill (RSB) linking the fix to the failure.
- Maintain the Descent (Zero Stale Noise): If a new change invalidates or replaces an older assumption, you MUST call curate to supersede, demote, or bridge the outdated memory node so future subagents never converge on stale beliefs.

4. Security & Operator Compliance (precedent)
- Never Bypass the Gate: Every shell command, git operation, and memory write is gated by the 114-rule Operator. If a tool call is blocked by Reflex or Operator, do not attempt workarounds or shell flags to bypass it.
- Inspect the Receipt: Immediately call precedent to inspect the Operator guard stop, read the BLAKE3/Ed25519 receipt, or run secret scanning before proposing a compliant alternative.

5. Hard Execution Constraints (Anti-Drift Checklist)
- No Unanchored Fixes: A bug fix is NOT complete until causal_walk has traced the cause and remember has anchored the resolution in Strata.
- No Silent Supersession: Never leave contradictory architectural facts active; always run curate when changing an invariant.
- Tool Priority: Always prefer recall and causal_walk over raw filesystem scanning when investigating "why" code was written a certain way.
```

---

## Project-Specific Data Directory

To keep a separate store per project, pass `--data-dir` with an **absolute** directory (Cursor does not expand `~` or relative paths in `args`):

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
2. Test the binary manually. It should print `vestige-mcp 6.2.0` or newer:
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

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.
