# Vestige Hooks — Sourcegraph Amp

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime amp
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime amp --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.config/amp/hooks.json`
- **Project / Repo Path**: `<repo>/.amp/hooks.json`
- **Events Wired**: `sessionStart`, `beforeSubmitPrompt`, `preToolUse`, `postToolUse`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Copy into `~/.config/amp/hooks.json` (`XDG_CONFIG_HOME/amp/hooks.json`).
