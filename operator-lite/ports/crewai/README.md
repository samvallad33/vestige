# Operator Lite — CrewAI port (official)

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[CrewAI](https://docs.crewai.com)'s Python agents through its **execution
hooks** — the `PRE_TOOL_CALL` interception point every tool call passes
through. One stdlib-only Python file is the whole adapter (`operator_hooks.py`);
it contains no rule logic: it reshapes the hook context into the gate's payload,
spawns the gate once per shell-ish call, and translates the exit code into
CrewAI's blocking dialect.

## The verified mechanism (and why)

Verified against the official docs
([docs.crewai.com/en/learn/execution-hooks](https://docs.crewai.com/en/learn/execution-hooks))
**and** against the crewAI source itself (tested live against crewAI
`1.15.23`, `lib/crewai/src/crewai/hooks/{dispatch,decorators,types,tool_hooks}.py`):

- **Registration** — `from crewai.hooks import register_hook,
  InterceptionPoint, HookAborted`; `register_hook(InterceptionPoint.PRE_TOOL_CALL, fn)`
  registers a **global** hook on the dispatcher's queue. The decorator form is
  `@on(InterceptionPoint.PRE_TOOL_CALL)` (global on a plain function; crew-scoped
  as a method inside a `@CrewBase` class). The legacy
  `register_before_tool_call_hook` registry is **aliased onto the same queue**
  (`tool_hooks.py`), so both dialects share one ordered chain. The port uses the
  programmatic `register_hook` (no decorator import-time magic), auto-detects the
  dialect, and degrades to `register_before_tool_call_hook` (block = `return
  False`, no reason) on crewAI versions that only have the legacy API.
- **Hook signature** — every hook takes a single context. At the tool points it
  is a `ToolCallHookContext`: `.tool_name` (sanitized lowercase a-z0-9_),
  `.tool_input` (the mutable kwargs dict), `.tool`, `.agent`, `.task`, `.crew`.
  No `cwd` is carried, so the adapter passes `os.getcwd()` of the crew process.
- **Blocking** — raise `HookAborted(reason, source)`. Verified special-cased in
  `dispatch._invoke_hook`: `HookAborted` **propagates by design** (and lands in
  `HookDispatchedEvent` telemetry with `abort_reason`/`abort_source`);
  `run_before_tool_call_hooks` catches it and returns `True`, and the tool never
  executes. **Every other exception a hook raises is swallowed (fail-open)** —
  the docs say so and the source confirms (`except Exception: ... return False`).
  This is why the port never relies on an ordinary exception to block: all four
  outcomes (allow / block / unreachable+benign / unreachable+destructive) are
  decided explicitly, and a residual adapter bug fails open exactly like an
  unreachable gate — the same boundary every other port documents.
- **How the block surfaces** — honest asymmetry, verified in the source: the
  tool result handed back to the crew/LLM is CrewAI's generic
  `"Tool execution blocked by hook. Tool: <name>"` (all three execution paths:
  `utilities/tool_utils.py`, `agents/crew_agent_executor.py`,
  `utilities/agent_utils.py`). The gate's full verdict — rule, why, compliant
  path, the owner's `operator-gate approve` digest — rides in the
  `HookAborted.reason` (visible in hook telemetry/event listeners), on the
  adapter's stderr, and in the receipt. The LLM is told it was blocked; the
  human can read exactly why.
- **Allow** — return `None` (proceed); receipts are already written by the gate.

```
CrewAI tool call (any agent, any task)
  └─ operator_hooks._hook(ctx)                      # PRE_TOOL_CALL, global
      └─ python3 ~/.operator/gate/operator-gate.py hook --source crewai
          stdin: {"tool_name", "tool_input": {"command"}, "cwd": os.getcwd()}
                 (+ "session_id" = crew id when readable)
          ├─ exit 0 -> return None        -> tool runs (receipt already written)
          ├─ exit 2 -> raise HookAborted("Operator Lite blocked: <gate stderr>",
          │                              source="operator-lite") -> tool never runs
          └─ unreachable -> fail-open, unless the command looks destructive
             (or writes a shell rc/init file): fail closed instead
```

The gate is spawned with `OPERATOR_AGENT_SESSION=1`, so `operator-gate approve`
refuses to grant permits from inside an agent session — approvals stay
owner-tty-only, as on every other host. Timeout per call: 8s.

## What it protects against

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
| OP-008 | shell-init writes (`tee ~/.zshrc` = code execution by install) |
| OP-009 | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |

Plus shadow-only OP-S01..S17 (work-loss git, outgoing comms, public mutation,
credential-shaped writes, pipe-to-interpreter, persistence, env hijacks, exfil
shapes, invisible characters, MCP argument exfil, agent-config writes,
paste/tunnel egress, new git remotes, sandbox-escape primitives), recorded in
receipts and promotable per rule via `commitments.json`. The analyzer sees
through quote reassembly (`r''m`), `$IFS`, `$(echo rm)`, ANSI-C quoting, base64
bodies, brace/glob enumeration, heredocs, session variables and subshell
time-bombs.

**GuardFall-proof:** the gate replays a 43-case corpus of bypass techniques and
must pass 43/43: `python3 ~/.operator/gate/operator-gate.py corpus guardfall`.

Every decision appends a hash-chained receipt to `~/.operator/receipts/`
(`operator-gate verify` walks the chain). Receipts are digests, not signatures.

## Usage (one import)

```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/.operator/ports/crewai"))
import operator_hooks   # registers the global PRE_TOOL_CALL gate hook (shadow)

from crewai import Agent, Task, Crew
researcher = Agent(role="Researcher", goal="Research the topic", backstory="...", verbose=True)
task = Task(description="Investigate {topic}", agent=researcher, expected_output="A report.")
Crew(agents=[researcher], tasks=[task]).kickoff(inputs={"topic": "Operator Lite"})
# every shell-ish tool call now passes the gate; flip: echo enforce > ~/.operator/mode
```

Prefer explicit control? Set `OPERATOR_CREWAI_AUTOINSTALL=0` before the import
and call `operator_hooks.install()` / `.uninstall()` / `print(.status())`
yourself. Crew-scoped instead of global: wrap the same idea in an `@on` method
inside your `@CrewBase` class (see the docs link above).

## Install (2 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/crewai/install.sh
sh ports/crewai/test.sh                    # optional: proves the chain end-to-end
```

`install.sh` is idempotent and:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub unless an existing copy is same-or-newer; never overwrites a newer
  gate; refuses to wire a gate that does not parse)
- installs the adapter to `~/.operator/ports/crewai/operator_hooks.py`
  (backing up any previous copy) and smoke-checks that it resolves the gate
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- prints the wiring snippet and next steps; requires `python3`

There is no hooks.json to merge: CrewAI hooks are registered from Python, so
the wiring is the one import above (put the path on `sys.path`, or copy the
module into your project — it is dependency-free).

## Shadow → enforce

Start in shadow: every decision is analyzed and receipted, nothing is blocked.
When the log looks right, flip:

```
echo enforce > ~/.operator/mode
```

In enforce mode a blocked call never runs; the `HookAborted` reason carries the
gate's full verdict, and the receipt carries the one-time `operator-gate
approve` digest the owner can grant from their own terminal.

## Tested

`test.sh` is hermetic (a throwaway HOME rooted under the real HOME — never
under `/tmp`, which the gate classifies as scratch; the real `~/.operator` and
your crews are never touched) and drives the adapter's **exact decision path**
(the same `_hook` body CrewAI calls, via `operator_hooks.py hookcheck`), plus —
when a Python with `crewai` importable is available (`OPERATOR_CREWAI_PYTHON`
or a probe) — the **real dispatch** through crewAI 1.15.23:

- install into the throwaway HOME: gate + adapter + `mode=shadow` + smoke status
- shadow: `rm -rf ~/demo-factory/canary` → allowed + `SHADOW_STOP` receipt with
  OP-003, `source=crewai`, tool `bash`, cwd = adapter cwd; obfuscated `r''m`
  variant receipted identically (gate sees through quote reassembly); `tee
  ~/.zshrc` receipted OP-008; benign `ls -la` receipted with no rule hits
- enforce: `rm -rf` → exit 2 with the `Operator Lite blocked:` reason carrying
  `OPERATOR: STOPPED` + OP-003; `r''m` and the `OPERATOR_CANARY_STOP_7f3a`
  canary blocked; in-cwd `rm -rf ./src` still allowed (precise, not
  trigger-happy); STOP receipt recorded
- gate unreachable: `tee ~/.zshrc` and `rm -rf` fail **closed**; `ls -la` fails
  open with a loud stderr note; a gate that exits 7 (not 0/2) counts as
  unreachable; no `~/.zshrc` is ever created
- real crewAI dispatch (`crewai` 1.15.23): hook sits on the
  `PRE_TOOL_CALL` queue; `run_before_tool_call_hooks` returns False in shadow
  and True in enforce for the same live process; `HookAborted.reason` starts
  with `Operator Lite blocked:` and `.source == "operator-lite"`; a tool with
  no command-shaped payload is not gated; unreachable fail-closed/fail-open
  hold through the real dispatcher
- receipt chain verifies (`chain=OK`); nothing is ever executed

## Honest limitations

- **Only command-carrying tool calls are gated.** The predicate is a tool name
  containing `shell`/`command`/`terminal` or a `tool_input` with a
  `command`/`cmd`/`script` key. Tools that act through other keys (a
  `write_file` path write, an MCP `url` fetch) are not routed through the gate
  by this port — the same boundary as the OpenClaw plugin, narrower than the
  Claude Code port (which also hooks its write tools).
- **The exceptions-swallowed caveat, and the answer.** CrewAI fails open on any
  hook exception other than `HookAborted`. The port leans on the one mechanism
  that is special-cased (`HookAborted`), decides every failure explicitly, and
  accepts that a residual adapter bug is invisible to CrewAI — failing open
  with a stderr note, exactly like an unreachable gate. The receipt trail, not
  the framework, is what makes silence visible.
- **The LLM sees CrewAI's generic block message**, not the gate's verdict
  (verified: all three tool-execution paths synthesize
  `"Tool execution blocked by hook. Tool: <name>"`). The verdict is in hook
  telemetry, stderr and receipts — for the human, not the model. Other hosts
  show the model the reason; CrewAI's hook contract does not, so the model
  learns "blocked" and must change course without the why.
- **Only calls through CrewAI's tool execution are seen.** A Flow step or
  custom code that calls a tool function directly in Python never fires
  `PRE_TOOL_CALL`; commands the owner runs in their own terminal are invisible
  (the gate's boundary on every host).
- **Latency is on us.** CrewAI demands hooks that don't block on heavy I/O.
  Each gated call pays one gate spawn (tens of ms); the 8s timeout bounds a
  wedged gate. Non-command tool calls pay one dict lookup.
- **Dialect coverage.** Verified on crewAI 1.15.23 (dispatcher dialect, global
  `register_hook`). Older crewAI without the dispatcher degrades to the legacy
  `before_tool_call` API where blocking is `return False` and the reason is
  dropped (tested logic path, not against a live old version). Future CrewAI
  hook-API changes are upstream's to break; the port is one dependency-free
  glue file.
- **No per-call human prompt.** CrewAI's `ctx.request_human_input()` exists,
  but Operator Lite's model is shadow-then-enforce with owner-side
  `operator-gate approve` permits, not interactive prompts mid-run.
- **`import operator_hooks` registers globally** — every crew in the process
  is gated. That is the feature; use `OPERATOR_CREWAI_AUTOINSTALL=0` +
  `install()`/`uninstall()` if you need finer control.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
