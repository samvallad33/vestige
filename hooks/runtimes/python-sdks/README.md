# Vestige Hooks — Python Agent SDKs (CrewAI, OpenAI Agents SDK, LangChain / LangGraph)

One-command install across OSes:

```sh
node hooks/runtimes/install.js --runtime python-sdks
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime python-sdks --guard
```

- **Config / Adapter File**: `vestige_hooks.py`
- **User / Global Path**: `Wrapper / SDK import (see below)`
- **Project / Repo Path**: `N/A (global/SDK)`
- **Events Wired**: `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`
- **Capabilities Included**:
  - Zero-Prompt Recall (`SessionStart` / `UserPromptSubmit`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + `precedent(query="stops"|"scan_secrets")` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git `post-commit` receipt automation + Stop proof-carrying finish
- **Activation Note**: Import `register_crewai_hooks`, `langchain_wrap_tool_call`, or `openai_agents_tool_guard` from `vestige_hooks.py`.
