# Vestige Hooks — Cursor Agent

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime cursor
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime cursor --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.cursor/hooks.json`
- **Project / Repo Path**: `<repo>/.cursor/hooks.json`
- **Events Wired**: `sessionStart`, `beforeSubmitPrompt`, `preToolUse`, `beforeShellExecution`, `beforeMCPExecution`, `beforeReadFile`, `postToolUse`, `postToolUseFailure`, `afterShellExecution`, `afterMCPExecution`, `afterFileEdit`, `subagentStart`, `subagentStop`, `preCompact`, `stop`, `sessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Reload Cursor window after installing ~/.cursor/hooks.json or .cursor/hooks.json.
