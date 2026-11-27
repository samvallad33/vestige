# Vestige Hooks — Aider CLI Wrapper

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime aider
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime aider --guard
```

- **Config / Adapter File**: `aider-vestige.mjs`
- **User / Global Path**: `Wrapper / SDK import (see below)`
- **Project / Repo Path**: `N/A (global/SDK)`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PostToolUse`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Run `node hooks/runtimes/aider/aider-vestige.mjs` instead of bare `aider` to wrap session recall, git receipts, and Stop auto-save.
