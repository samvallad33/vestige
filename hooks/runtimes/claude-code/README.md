# Vestige Hooks — Claude Code

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime claude-code
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime claude-code --guard
```

- **Config / Adapter File**: `settings.json`
- **User / Global Path**: `~/.claude/settings.json`
- **Project / Repo Path**: `<repo>/.claude/settings.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, `PostToolUseFailure`, `PreCompact`, `PostCompact`, `SubagentStart`, `SubagentStop`, `Stop`, `SessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Restart Claude Code or open `/hooks` to inspect registered hooks.
