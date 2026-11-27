# Vestige Multi-Runtime Hook Bundles (24 Runtimes · 2026)

Drop-in, schema-verified Vestige hook configurations, native plugins, and SDK adapters for **every major 2026 AI coding runtime**. Every bundle routes through Vestige's single shared engine ([packages/vestige-init/hooks/vestige-bridge.cjs](../../packages/vestige-init/hooks/vestige-bridge.cjs)):

- **Zero-Prompt Recall + Capsule + Active Intentions** on session start & prompt submit
- **Call Coach (CC-001..CC-013)** pre-execution alias & token-budget optimization
- **Operator Guard + `precedent()`** fail-closed safety & secret-scan commit gates
- **Anti-Spiral, Chesterton's Fence & Blast-Radius** warnings across long runs
- **Proof-Carrying Finish** at Stop + git `post-commit` receipt capture

## Quick Install

```sh
# Install into one runtime (globally)
node hooks/runtimes/install.js --runtime codex

# Install into a specific project/repo
node hooks/runtimes/install.js --runtime copilot --project /path/to/repo

# Install into ALL detected runtimes on this machine + enable Operator Guard
node hooks/runtimes/install.js --all --guard
```

Or use `@vestige/init`:

```sh
npx @vestige/init --guard
```

## Runtime Matrix (24 Runtimes)

| Runtime | Folder | Target Config Location | Events |
|---|---|---|---|
| **OpenAI Codex CLI** | [`codex/`](./codex/) | `~/.codex/hooks.json` | 12 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **Cursor Agent** | [`cursor/`](./cursor/) | `~/.cursor/hooks.json` | 16 events (`sessionStart`, `beforeSubmitPrompt`, `preToolUse`, ...) |
| **Gemini CLI** | [`gemini-cli/`](./gemini-cli/) | `~/.gemini/settings.json` | 7 events (`SessionStart`, `BeforeAgent`, `BeforeTool`, ...) |
| **Claude Code** | [`claude-code/`](./claude-code/) | `~/.claude/settings.json` | 12 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **Antigravity** | [`antigravity/`](./antigravity/) | `~/.gemini/config/plugins/vestige/hooks.json` | 6 events (`SessionStart`, `PreInvocation`, `PreToolUse`, ...) |
| **GitHub Copilot CLI & VS Code** | [`copilot/`](./copilot/) | `~/.copilot/hooks/vestige.json` | 10 events (`sessionStart`, `userPromptSubmitted`, `preToolUse`, ...) |
| **Windsurf Cascade** | [`windsurf/`](./windsurf/) | `~/.codeium/windsurf/hooks.json` | 9 events (`pre_user_prompt`, `pre_read_code`, `pre_write_code`, ...) |
| **Devin Local & CLI** | [`devin/`](./devin/) | `~/.config/devin/hooks.v1.json` | 6 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **OpenCode** | [`opencode/`](./opencode/) | `~/.config/opencode/plugins/vestige/index.js` | 4 events (`experimental.chat.system.transform`, `tool.execute.before`, `tool.execute.after`, ...) |
| **Cline (v3.36+)** | [`cline/`](./cline/) | `~/.cline/hooks/PreToolUse` | 6 events (`TaskStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **AWS Kiro IDE & CLI** | [`kiro/`](./kiro/) | `~/.kiro/hooks/vestige.json` | 5 events (`UserPromptSubmit`, `PreToolUse`, `PostToolUse`, ...) |
| **Factory AI Droid** | [`factory-droid/`](./factory-droid/) | `~/.factory/hooks.json` | 6 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **Roo Code** | [`roo-code/`](./roo-code/) | `~/.roo/hooks.json` | 6 events (`sessionStart`, `beforeSubmitPrompt`, `preToolUse`, ...) |
| **Kilo Code** | [`kilo-code/`](./kilo-code/) | `~/.kilocode/hooks.json` | 5 events (`sessionStart`, `beforeSubmitPrompt`, `preToolUse`, ...) |
| **Block Goose (v1.41+)** | [`goose/`](./goose/) | `~/.config/goose/hooks.json` | 5 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **Charm Crush** | [`crush/`](./crush/) | `~/.config/crush/crush.json` | 2 events (`PreToolUse`, `PostToolUse`) |
| **OpenHands** | [`openhands/`](./openhands/) | `~/.openhands/hooks.json` | 5 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
| **OpenClaw** | [`openclaw/`](./openclaw/) | `~/.openclaw/extensions/vestige-hooks/index.js` | 4 events (`before_prompt_build`, `before_tool_call`, `after_tool_call`, ...) |
| **Amazon Q Developer CLI** | [`amazon-q/`](./amazon-q/) | `~/.aws/amazonq/cli-agents/q_cli_default.json` | 5 events (`agentSpawn`, `userPromptSubmit`, `preToolUse`, ...) |
| **Continue** | [`continue/`](./continue/) | `~/.continue/hooks.json` | 5 events (`sessionStart`, `beforeSubmitPrompt`, `preToolUse`, ...) |
| **Zed** | [`zed/`](./zed/) | `~/.config/zed/hooks.json` | 4 events (`sessionStart`, `preToolUse`, `postToolUse`, ...) |
| **Sourcegraph Amp** | [`amp/`](./amp/) | `~/.config/amp/hooks.json` | 5 events (`sessionStart`, `beforeSubmitPrompt`, `preToolUse`, ...) |
| **Aider CLI Wrapper** | [`aider/`](./aider/) | `aider-vestige.mjs` | 4 events (`SessionStart`, `UserPromptSubmit`, `PostToolUse`, ...) |
| **Python Agent SDKs (CrewAI, OpenAI Agents SDK, LangChain / LangGraph)** | [`python-sdks/`](./python-sdks/) | `vestige_hooks.py` | 5 events (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, ...) |
