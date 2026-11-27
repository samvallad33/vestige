# Vestige Hooks — Devin Local & CLI

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime devin
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime devin --guard
```

- **Config / Adapter File**: `hooks.v1.json`
- **User / Global Path**: `~/.config/devin/hooks.v1.json`
- **Project / Repo Path**: `<repo>/.devin/hooks.v1.json`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Place in `.devin/hooks.v1.json` in your repo or merge under `hooks` in `~/.config/devin/config.json`.
