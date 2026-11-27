# Vestige Hooks — Charm Crush

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime crush
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime crush --guard
```

- **Config / Adapter File**: `crush.json`
- **User / Global Path**: `~/.config/crush/crush.json`
- **Project / Repo Path**: `<repo>/crush.json`
- **Events Wired**: `PreToolUse`, `PostToolUse`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Flat `{ name, matcher, command, timeout }` entries inside `crush.json` `hooks`.
