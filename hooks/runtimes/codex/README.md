# Vestige Hooks — OpenAI Codex CLI

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime codex
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime codex --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.codex/hooks.json`
- **Project / Repo Path**: `<repo>/.codex/hooks.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PreCompact`, `PostCompact`, `Stop`, `SubagentStart`, `SubagentStop`, `Interrupt`, `SessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Restart Codex or run `/hooks` in Codex CLI to review & approve hook commands.
