# Codex

> Written for Vestige 6.x.

> Give Codex a record of what you decided, that carries across sessions.

Codex has native MCP support through the `codex mcp` CLI. Add Vestige once and Codex can reload project decisions, corrections and past fixes in every session.

---

## Prerequisites

- **Codex CLI** installed and authenticated
- **vestige-mcp** binary installed ([Installation guide](../REFERENCE.md#install))

---

## Setup

### 1. Add Vestige

```bash
codex mcp add vestige -- "$(which vestige-mcp)"
```

> **Use an absolute path.** The README install puts the binaries in `~/.local/bin`, and Homebrew puts them under its own prefix. Run `which vestige-mcp` and use what it prints.

### 2. Verify

```bash
codex mcp list
```

You should see a `vestige` entry with `enabled` status.

### 3. Test it in Codex

Start Codex and ask:

> "What MCP tools do you have access to?"

You should see Vestige's 6 tools listed: `session`, `remember`, `recall`, `causal_walk`, `curate`, and `precedent`.

---

## First Use

In Codex:

> "Remember that this project uses Rust with Axum. Tag it `stack`."

Start a **new session**, then ask:

> "Recall the `stack` tag from Vestige."

It comes back, with its memory id. Vestige finds a record by an exact handle (its id, an 8+ character prefix, or an exact tag), not by resemblance, so name the tag when you save and when you ask. See [Getting Started](../GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

---

## Manual Configuration

Codex stores MCP servers in `~/.codex/config.toml`.

Minimal config:

```toml
[mcp_servers.vestige]
command = "<absolute path from which vestige-mcp or where vestige-mcp>"
```

After saving, restart Codex or start a new session.

---

## Project-Specific Data Directory

Use `--data-dir` to keep a separate store per repo or workspace:

```bash
codex mcp remove vestige
codex mcp add vestige -- "$(which vestige-mcp)" --data-dir /Users/you/projects/my-app/.vestige
```

Equivalent manual config:

```toml
[mcp_servers.vestige]
command = "<absolute path from which vestige-mcp>"
args = ["--data-dir", "/Users/you/projects/my-app/.vestige"]
```

---

## Agent Protocol

MCP registration makes Vestige tools available to Codex. It does not, by itself, make Codex call them before it answers.

Codex reads `AGENTS.md` at the workspace root or repo root. Embed the mandatory **VESTIGE COGNITIVE PROTOCOL** directly into `AGENTS.md`:

```markdown
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

## Troubleshooting

<details>
<summary>Vestige tools do not appear in Codex</summary>

1. Verify the server is registered:
   ```bash
   codex mcp list
   ```
2. Check the binary path:
   ```bash
   which vestige-mcp
   ```
3. Ensure the config entry exists in `~/.codex/config.toml`.
4. Start a fresh Codex session after adding the server.
</details>

<details>
<summary>Need to remove or re-add the server</summary>

```bash
codex mcp remove vestige
codex mcp add vestige -- "$(which vestige-mcp)"
```
</details>

---

## Also Works With

| IDE | Guide |
|-----|-------|
| Xcode 26.3 | [Setup](./xcode.md) |
| Cursor | [Setup](./cursor.md) |
| VS Code (Copilot) | [Setup](./vscode.md) |
| OpenCode | [Setup](./opencode.md) |
| JetBrains | [Setup](./jetbrains.md) |
| Windsurf | [Setup](./windsurf.md) |
| Claude Code | [Setup](../CONFIGURATION.md#claude-code-one-liner) |
| Claude Desktop | [Setup](../CONFIGURATION.md#claude-desktop-macos) |

Every agent on the machine shares one store through one writer. The first to start serves it, and the others attach.
