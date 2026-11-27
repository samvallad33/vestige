# Vestige Hooks — AWS Kiro IDE & CLI

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime kiro
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime kiro --guard
```

- **Config / Adapter File**: `vestige.json`
- **User / Global Path**: `~/.kiro/hooks/vestige.json`
- **Project / Repo Path**: `<repo>/.kiro/hooks/vestige.json`
- **Events Wired**: `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostFileSave`, `SessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Uses Kiro 1.x `"version": "v1"` schema in `.kiro/hooks/vestige.json` (`exit 2` blocks PreToolUse).
