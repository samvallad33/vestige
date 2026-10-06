# Operator Lite — LangChain / LangGraph port (official)

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[LangChain](https://docs.langchain.com) agents (LangChain v1 `create_agent`,
which is a LangGraph graph under the hood) and, with one extra line of user
code, into hand-rolled LangGraph `StateGraph`s. One stdlib-only Python file
does all analysis; this port is a thin adapter — a middleware module plus
install/test scripts. It contains no rule logic and no analysis.

## The verified mechanism (and why)

Verified against the current (2026) LangChain v1 middleware API, in the docs
and the source — not inferred:

- **Hook** — `wrap_tool_call` ("Wrap and intercept tool calls"), as a
  decorator or as a method on
  [`AgentMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/types/wrap_tool_call).
  The wrapper receives `(request, handler)`; `request` is a
  [`ToolCallRequest`](https://docs.langchain.com/oss/python/langchain/middleware/custom)
  whose `request.tool_call` is the model's call dict `{"name", "args", "id"}`
  (plus `request.tool` / `request.state` / `request.runtime`). It must return
  a `ToolMessage` or `Command`.
- **Blocking** — the docs' own short-circuit pattern: return a `ToolMessage`
  and call `handler` **zero times**. Raising is deliberately *not* used as the
  block form: per the `AgentMiddleware.wrap_tool_call` contract, "exceptions
  propagate unless `handle_tool_errors` is configured on `ToolNode`", so a
  raise can be swallowed into a generic error message — a returned
  `ToolMessage` is deterministic, and its `content` is what the LLM sees:

  ```
  Operator Lite blocked: OPERATOR: STOPPED (OP-003 no-blind-recursive-delete)
  Why: Recursive delete outside scratch/build directories needs ...
  No action was taken. Do not retry this action or a variation of it.
  Only the owner can allow it once ... operator-gate approve <digest>
  ```

- **Attachment** — `create_agent(model, tools, middleware=[...])`
  ([built-in middleware docs](https://docs.langchain.com/oss/python/langchain/middleware/built-in)).
- **LangGraph difference** — `langgraph.prebuilt.ToolNode` takes **no**
  middleware; it does support interceptors on current versions, but the
  portable path for a hand-rolled `StateGraph` is to guard the tool function
  itself with this module's `decide()` and raise `ToolException` (snippet
  below). `HumanInTheLoopMiddleware` (per-tool `interrupt_on` with
  `RejectDecision {"type": "reject", "message": ...}`) is the interactive
  complement for calls a deterministic gate should *pause* rather than decide.

So the middleware consults the gate on every shell-ish tool call and turns the
gate's exit code into LangChain's native vocabulary:

```
LangChain tool call (request.tool_call)
  └─ operator_middleware.decide(name, args)
      ├─ not shell-ish -> handler runs untouched
      └─ spawn: python3 ~/.operator/gate/operator-gate.py hook --source langchain
          stdin: {"tool_name", "tool_input": {"command"}, "cwd", "session_id"}
          ├─ exit 0 -> handler(request)            (receipt already written)
          └─ exit 2 -> ToolMessage("Operator Lite blocked: <gate stderr>")
```

A tool is *shell-ish* when its name contains "shell" (LangChain's ShellTool /
ShellToolMiddleware) or its args carry a `command` key (plus `cmd`/`script`
fallbacks, mirroring the OpenClaw adapter) — the gate, not the port, decides
relevance. On spawn failure the port fails **open**, except when the raw
command matches the destructive-like regex (the OpenClaw adapter's, extended
with shell-rc writes), in which case it fails **closed**.

## What it stops

Eleven STOP rules (blocking in enforce mode):

| Rule | Catches |
|---|---|
| OP-000 | edits to the gate itself, hook registrations, agent settings files |
| OP-001 | delete/move of a workspace root (or a parent of one) |
| OP-002 | wiping the Vestige store (sqlite3 writes, `vestige gc/purge`) |
| OP-003 | blind recursive deletes, sweeps from `/`, fork bombs |
| OP-004 | force-push to shared branches, remote-main deletion, `.git` removal |
| OP-005 | unreviewed publishes (npm/cargo/twine/docker/gh release) |
| OP-006 | paid deploys and live-billing actions |
| OP-007 | destructive SQL (DROP/TRUNCATE/unscoped DELETE) |
| OP-008 | shell-init writes (`>> ~/.zshrc` = code execution by install) |
| OP-009 | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |

Plus shadow-only OP-S01..S17 (work-loss git, outgoing comms, public mutation,
credential-shaped writes, pipe-to-interpreter, persistence, env hijacks, exfil
shapes, invisible characters, MCP argument exfil, agent-config writes,
paste/tunnel egress, new git remotes, sandbox-escape primitives), recorded in
receipts and promotable per rule via `commitments.json`. The analyzer sees
through quote reassembly (`r''m`), `$IFS`, `$(echo rm)`, ANSI-C quoting,
base64 bodies, brace/glob enumeration, heredocs, session variables and
subshell time-bombs; every STOP names the transforms it applied.

**GuardFall-proof:** the gate replays a 43-case corpus of bypass techniques
and must pass 43/43:

```
python3 ~/.operator/gate/operator-gate.py corpus guardfall
```

Every decision appends a hash-chained receipt to `~/.operator/receipts/`
(`operator-gate verify` walks the chain). Receipts are digests, not
signatures.

## Install (2 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/langchain/install.sh
sh ports/langchain/test.sh                    # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub unless an existing copy is same-or-newer; never overwrites a newer
  gate; refuses to wire a gate that does not parse)
- installs the middleware to `~/.operator/ports/langchain/operator_middleware.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- prints the usage snippet and next steps; requires `python3` (the gate and
  the adapter are stdlib-only — no `pip install` of anything)

## Usage

**LangChain v1 `create_agent`** (the minimal wiring):

```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/.operator/ports/langchain"))

from langchain.agents import create_agent
from operator_middleware import operator_middleware

agent = create_agent(
    "<your-model>",
    tools=[run_shell_command],      # any tool whose args carry "command"
    middleware=[operator_middleware()],
)
agent.invoke({"messages": [{"role": "user", "content": "tidy the build dir"}]})
# a stopped call comes back to the LLM as: "Operator Lite blocked: OPERATOR: STOPPED (OP-...)"
```

Or copy `operator_middleware.py` into your project next to your agent code and
`from operator_middleware import operator_middleware` — the module is the
distribution, and it imports cleanly with no langchain installed (the
langchain imports happen lazily inside `operator_middleware()`).

**LangGraph (`StateGraph` + `ToolNode`)** — `ToolNode` takes no middleware, so
guard the tool itself; `ToolException` text reaches the LLM when the node's
error handling is enabled (`handle_tool_errors=True` on `ToolNode`):

```python
from langchain_core.tools import tool, ToolException
from operator_middleware import decide          # stdlib-only core

@tool
def run_shell_command(command: str) -> str:
    """Run a shell command."""
    verdict, message = decide("shell", {"command": command})
    if verdict == "block":
        raise ToolException(message)            # the LLM sees the gate's verdict
    import subprocess
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
```

## Shadow → enforce

Start in shadow: every decision is analyzed and receipted, nothing is blocked.
When the log looks right, flip:

```
echo enforce > ~/.operator/mode
```

In enforce mode a blocked call reaches the LLM as a `ToolMessage` carrying the
gate's full verdict — rule, why, the compliant path, and the one-time
`operator-gate approve` digest the owner can grant from their own terminal.
The middleware applies the same verdict to sync (`wrap_tool_call`) and async
(`awrap_tool_call`) agents.

## Tested

`test.sh` is hermetic (a throwaway HOME rooted under the real HOME — never
under `/tmp`, which the gate classifies as scratch; the real `~/.operator` is
never touched) and runs the real `install.sh` into it, then drives the
middleware's `decide()` — the exact function `wrap_tool_call` applies to every
tool call — with args in the exact shape LangChain passes:

- `rm -rf ~/demo-factory/canary` → shadow: allowed + OP-003 `SHADOW_STOP`
  receipt; enforce: blocked with `Operator Lite blocked: OPERATOR: STOPPED`
  naming OP-003, plus a `STOP` receipt
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → blocked in enforce
- `echo OPERATOR_CANARY_STOP_7f3a` → blocked (the canary proves the whole
  chain is wired)
- `ls -la` → allowed in both modes, receipted with no rule hits
- `rm -rf ./src` inside the session cwd → allowed (the gate is precise, not
  trigger-happy)
- `tee ~/.zshrc` with the gate unreachable → blocked (fail closed, the
  destructive-like extension); `ls -la` with the gate unreachable → allowed
  (fail open)
- non-shell tools (`read_file`) and shell-named tools without a command key →
  passed through with no gate spawn and no receipt
- the **GuardFall corpus replayed through the adapter**: all 43 verdicts
  survive `decide()` (same semantics as the gate's own corpus runner)
- the module imports with langchain absent (stdlib-only adapter), install.sh
  is idempotent and never clobbers an existing mode, the receipt chain
  verifies, and no payload directory is ever created — nothing is executed

When `langchain` is importable, `test.sh` additionally drives the real
middleware object — and a real agent loop:

- `wrap_tool_call` returning a `ToolMessage` (never running the handler) for a
  blocked call, handler pass-through for allowed and non-shell calls, and the
  same via `awrap_tool_call`
- a **live `create_agent` graph** with this middleware attached and a scripted
  model (`GenericFakeChatModel` with a no-op `bind_tools` — the standard
  test-double, no API key): a destructive tool call surfaces to the "LLM" as
  `Operator Lite blocked: OPERATOR: STOPPED (OP-003 ...)` and the tool double
  never runs; a benign call reaches the tool

## Honest limitations

- **No session against a real hosted LLM was run.** Everything short of the
  model itself is tested (middleware, block form, async path, and the full
  `create_agent` loop with a scripted model); what a hosted model adds is
  beyond this port's control. Confirm end-to-end on your stack by asking a
  real agent to run `echo OPERATOR_CANARY_STOP_7f3a` and checking
  `~/.operator/receipts/` for the OP-CANARY receipt.
- **Only command-carrying tools are gated.** A tool whose args do not carry a
  `command`/`cmd`/`script` string is passed through untouched; arbitrary
  Python tools (a `@tool` that calls `os.remove` directly, notebook-style
  code execution) are out of scope — the gate never sees them. Wrap such
  tools with `decide()` yourself, as in the LangGraph snippet.
- **The cwd is the agent process's cwd.** LangChain tool calls carry no
  per-call working directory, so the port reports `os.getcwd()`; relative
  paths in a command are judged against that. A tool that internally `cd`s
  elsewhere before acting is invisible to that judgment (gate boundary, not a
  port defect). Receipts record `session_id` only if the caller passes one;
  the middleware leaves it unset (LangChain thread ids live in RunnableConfig,
  which `ToolCallRequest` does not reliably expose).
- **A broken install fails open** (spawn error, gate missing) except for
  destructive-like text, which fails closed. `install.sh` narrows the window
  by parse-checking the gate and the middleware before declaring success.
- **Composition order matters to other middleware.** LangChain composes
  middleware "first defined = outermost"; a middleware *outside* this one can
  still short-circuit calls before the gate sees them. Put
  `operator_middleware()` first in the list.
- **Gate-owning actions stay interactive.** The middleware sets
  `OPERATOR_AGENT_SESSION=1` when spawning the gate, so `operator-gate
  approve/mode/install` invoked *through* an agent are refused by the gate
  itself; approvals happen in the owner's terminal.
- **Receipts name every decision** — allowed, shadow-stopped, or blocked.
  That is the audit trail, not noise; `operator-gate verify` walks the chain,
  and the chain is digests, not signatures.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
