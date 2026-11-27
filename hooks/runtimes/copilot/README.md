# Vestige Hooks — GitHub Copilot CLI & VS Code

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime copilot
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime copilot --guard
```

- **Config / Adapter File**: `vestige.json`
- **User / Global Path**: `~/.copilot/hooks/vestige.json`
- **Project / Repo Path**: `<repo>/.github/hooks/vestige.json`
- **Events Wired**: `sessionStart`, `userPromptSubmitted`, `preToolUse`, `postToolUse`, `errorOccurred`, `preCompact`, `subagentStart`, `subagentStop`, `agentStop`, `sessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Requires a trusted workspace in VS Code (`chat.useHooks` enabled; on Copilot CLI ~/.copilot/hooks/*.json loads automatically).
