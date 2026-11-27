# Vestige Hooks — Cline (v3.36+)

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime cline
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime cline --guard
```

- **Config / Adapter File**: `PreToolUse`
- **User / Global Path**: `~/.cline/hooks/PreToolUse`
- **Project / Repo Path**: `<repo>/.clinerules/hooks/PreToolUse`
- **Events Wired**: `TaskStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `TaskComplete`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Copy the 6 executable files (`TaskStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `TaskComplete`) into `~/.cline/hooks/` (and `~/Documents/Cline/Hooks/`) or `.clinerules/hooks/`.
