# Vestige Hooks — Zed

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime zed
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime zed --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.config/zed/hooks.json`
- **Project / Repo Path**: `<repo>/.zed/hooks.json`
- **Events Wired**: `sessionStart`, `preToolUse`, `postToolUse`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Also add `"context_servers": { "vestige": { "command": "vestige-mcp", "args": [] } }` to `~/.config/zed/settings.json`.
