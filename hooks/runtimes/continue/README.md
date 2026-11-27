# Vestige Hooks — Continue

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime continue
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime continue --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.continue/hooks.json`
- **Project / Repo Path**: `<repo>/.continue/hooks.json`
- **Events Wired**: `sessionStart`, `beforeSubmitPrompt`, `preToolUse`, `postToolUse`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Copy into `~/.continue/hooks.json` or repo `.continue/hooks.json`.
