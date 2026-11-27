# Vestige Hooks — OpenCode

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime opencode
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime opencode --guard
```

- **Config / Adapter File**: `index.js`
- **User / Global Path**: `~/.config/opencode/plugins/vestige/index.js`
- **Project / Repo Path**: `<repo>/.opencode/plugins/vestige/index.js`
- **Events Wired**: `experimental.chat.system.transform`, `tool.execute.before`, `tool.execute.after`, `experimental.session.compacting`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: OpenCode auto-loads JS/TS plugins from `.opencode/plugins/` or `~/.config/opencode/plugins/`.
