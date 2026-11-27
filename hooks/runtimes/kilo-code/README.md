# Vestige Hooks — Kilo Code

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime kilo-code
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime kilo-code --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.kilocode/hooks.json`
- **Project / Repo Path**: `<repo>/.kilocode/hooks.json`
- **Events Wired**: `sessionStart`, `beforeSubmitPrompt`, `preToolUse`, `postToolUse`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Merge into `~/.kilocode/hooks.json` or repo `.kilocode/hooks.json`.
