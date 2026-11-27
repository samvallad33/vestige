# Vestige Hooks — Block Goose (v1.41+)

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime goose
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime goose --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.config/goose/hooks.json`
- **Project / Repo Path**: `<repo>/.goose/hooks.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Enable lifecycle hooks in Goose config (`GOOSE_ENABLE_HOOKS=true`); blocks on exit code 2 + stderr.
