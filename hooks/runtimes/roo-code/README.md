# Vestige Hooks — Roo Code

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime roo-code
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime roo-code --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.roo/hooks.json`
- **Project / Repo Path**: `<repo>/.roo/hooks.json`
- **Events Wired**: `sessionStart`, `beforeSubmitPrompt`, `preToolUse`, `postToolUse`, `preCompact`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Merge into `~/.roo/hooks.json` or repo `.roo/hooks.json`.
