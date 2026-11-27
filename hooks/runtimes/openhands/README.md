# Vestige Hooks — OpenHands

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime openhands
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime openhands --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.openhands/hooks.json`
- **Project / Repo Path**: `<repo>/.openhands/hooks.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Place `.openhands/hooks.json` in the workspace or `~/.openhands/hooks.json`.
