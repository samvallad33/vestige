# Vestige Hooks — Amazon Q Developer CLI

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime amazon-q
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime amazon-q --guard
```

- **Config / Adapter File**: `q_cli_default.json`
- **User / Global Path**: `~/.aws/amazonq/cli-agents/q_cli_default.json`
- **Project / Repo Path**: `<repo>/.amazonq/cli-agents/vestige.json`
- **Events Wired**: `agentSpawn`, `userPromptSubmit`, `preToolUse`, `postToolUse`, `stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Sets `hooks.agentSpawn`, `hooks.userPromptSubmit`, `hooks.preToolUse`, `hooks.postToolUse`, `hooks.stop` in Q CLI default agent.
