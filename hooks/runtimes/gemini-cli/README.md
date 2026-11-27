# Vestige Hooks — Gemini CLI

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime gemini-cli
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime gemini-cli --guard
```

- **Config / Adapter File**: `settings.json`
- **User / Global Path**: `~/.gemini/settings.json`
- **Project / Repo Path**: `<repo>/.gemini/settings.json`
- **Events Wired**: `SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `PreCompress`, `AfterAgent`, `SessionEnd`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Run `/hooks panel` in Gemini CLI to verify active hooks, or link directly with `gemini extensions link hooks/runtimes/gemini-cli`.
