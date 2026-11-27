# Vestige Hooks — Antigravity

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime antigravity
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime antigravity --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.gemini/config/plugins/vestige/hooks.json`
- **Project / Repo Path**: `N/A (global/SDK)`
- **Events Wired**: `SessionStart`, `PreInvocation`, `PreToolUse`, `PostToolUse`, `PostInvocation`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Restart Antigravity if plugin hooks changed. Use `npx @vestige/init` for full bridge bundle deployment.
