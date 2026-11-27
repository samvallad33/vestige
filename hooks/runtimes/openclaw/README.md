# Vestige Hooks — OpenClaw

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime openclaw
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime openclaw --guard
```

- **Config / Adapter File**: `index.js`
- **User / Global Path**: `~/.openclaw/extensions/vestige-hooks/index.js`
- **Project / Repo Path**: `N/A (global/SDK)`
- **Events Wired**: `before_prompt_build`, `before_tool_call`, `after_tool_call`, `agent_end`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Install plugin folder into `~/.openclaw/extensions/vestige-hooks/` (`before_tool_call`, `after_tool_call`, `before_prompt_build`, `agent_end`).
