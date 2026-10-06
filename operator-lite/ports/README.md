# Operator Lite ports — one gate, every host

The gate is one stdlib-only Python file: `../operator-gate.py`. It is
host-agnostic by contract:

- reads a JSON payload on stdin: `{"tool_name", "tool_input": {"command"}, "cwd", "session_id"?}`
- exit 0 = allow (silent), exit 2 = block (the reason is on stderr, and the
  host surfaces it to the model, which must change course)
- writes its own hash-chained receipts under `~/.operator/receipts/`
- shadow vs enforce is a file: `~/.operator/mode` (`shadow` default)

Every port in this directory is a thin adapter that carries that contract to
a different agent host. No port contains rule logic. No port re-implements
analysis. One gate, verified once against the 43-case GuardFall bypass
corpus, everywhere.

## The matrix

| Host | Mechanism | Port | Status |
|---|---|---|---|
| OpenClaw | `before_tool_call` plugin | [../openclaw-plugin/](../openclaw-plugin/) (ClawHub: `clawhub install vestige-operator-lite`) | published, inspector PASS |
| Claude Code | `PreToolUse` hook | [../README.md](../README.md#install-one-command) | shipped |
| Codex | hooks.json | [../README.md](../README.md#install-one-command) | shipped |
| OpenHands | `.openhands/hooks.json` PreToolUse | [openhands/](openhands/) | gate-tested, suite PASS |
| OpenCode | `.opencode/plugins/` `tool.execute.before` | [opencode/](opencode/) | gate-tested, corpus 46/46 via adapter, boots in opencode CLI |
| Cline | file hooks `~/.cline/hooks/PreToolUse` (fires before approval policies — covers YOLO) + SDK plugin | [cline/](cline/) | gate-tested, suite PASS, receipt chain OK |
| Goose | lifecycle hooks `PreToolUse` (exit-2 block, goose v1.41.0+) | [goose/](goose/) | gate-tested, 29/29 |
| Crush | `hooks.PreToolUse[]` in crush.json — flat `{name, matcher?, command, timeout}` entries; payload is the gate's contract verbatim (exit-2 block, stderr reason) | [crush/](crush/) | gate-tested, suite 40/40 |
| Gemini CLI | `BeforeTool` hook in `~/.gemini/settings.json` (exit-2 deny) | [gemini-cli/](gemini-cli/) | gate-tested, suite PASS, corpus 46/46 |
| Amazon Q CLI | `preToolUse` hook in `~/.aws/amazonq/cli-agents/q_cli_default.json` (exit-2 block, stderr to model) | [amazon-q/](amazon-q/) | gate-tested, suite PASS, corpus 46/46 via adapter |
| Cursor | agent hooks in `~/.cursor/hooks.json` — `beforeShellExecution` + `beforeMCPExecution` + `preToolUse` (disjoint matchers); deny is stdout JSON `{"permission":"deny"}` at exit 0 (invalid output blocks too); `failClosed: true` shipped | [cursor/](cursor/) | gate-tested, suite 44/44 |
| Windsurf (Cascade) | `pre_run_command` / `pre_write_code` / `pre_mcp_tool_use` hooks in `~/.codeium/windsurf/hooks.json` (workspace `.windsurf/hooks.json` / `.devin/hooks.json` optional) — payload reshaped by a shim (`tool_info.command_line` → `tool_input.command`; exit-2 block, stderr reason) | [windsurf/](windsurf/) | gate-tested, suite 57/57 PASS, corpus 46/46 via adapter |
| CrewAI | execution hooks — global `register_hook(PRE_TOOL_CALL, fn)`; block is `HookAborted(reason, source)` (special-cased; other hook exceptions fail open), LLM sees CrewAI's generic blocked message | [crewai/](crewai/) | gate-tested, suite PASS incl. real crewai 1.15.23 dispatch |
| OpenAI Agents SDK | tool input guardrails (`reject_content`) | [openai-agents/](openai-agents/) | live-tested on openai-agents 0.22.3, 30 PASS |
| LangChain / LangGraph (Python) | `@wrap_tool_call` / `AgentMiddleware` middleware for LangChain v1 `create_agent` (block = returned `ToolMessage`, handler never runs); hand-rolled LangGraph `ToolNode` via `decide()` + `ToolException` in the tool | [langchain/](langchain/) | gate-tested, suite PASS, corpus 46/46 via adapter, live `create_agent` loop (scripted model) PASS |
| Aider | no hook API — terminal wrapper | [aider/](aider/) | see port README |

Status values: `shipped` (tested end-to-end) · `gate-tested` (gate contract
verified locally; host wiring follows documented, unmodified upstream
behavior) · `experimental`.

## Install any port

```
sh ports/<host>/install.sh
```

Every install.sh: installs the gate to `~/.operator/gate/` (never
overwrites a newer copy), wires the host hook, sets shadow mode, prints the
flip command: `echo enforce > ~/.operator/mode`.

## The rule

Forks die; adapters compound. The platforms upstream control their release
cycles — we ride the hooks they already shipped. If a platform adds a
better interception point later, the port shrinks; the gate stays.
