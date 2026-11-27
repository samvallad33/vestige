# Vestige Hooks — Windsurf Cascade

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime windsurf
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime windsurf --guard
```

- **Config / Adapter File**: `hooks.json`
- **User / Global Path**: `~/.codeium/windsurf/hooks.json`
- **Project / Repo Path**: `<repo>/.windsurf/hooks.json`
- **Events Wired**: `pre_user_prompt`, `pre_read_code`, `pre_write_code`, `pre_run_command`, `pre_mcp_tool_use`, `post_write_code`, `post_run_command`, `post_mcp_tool_use`, `post_cascade_response`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Restart Windsurf after adding ~/.codeium/windsurf/hooks.json or workspace .windsurf/hooks.json.
