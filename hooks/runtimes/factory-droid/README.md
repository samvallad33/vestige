# Vestige Hooks — Factory AI Droid

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime factory-droid
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime factory-droid --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.factory/hooks.json`
- **Project / Repo Path**: `<repo>/.factory/hooks.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Inspect and enable hooks in Factory Droid via `/hooks`.
