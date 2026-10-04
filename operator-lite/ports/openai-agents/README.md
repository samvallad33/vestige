# Operator Lite — OpenAI Agents SDK (Python) port

The gate for the OpenAI Agents SDK: every command-carrying function tool call
passes `operator-gate.py` **before** the tool runs. Gate says no → the model
sees `Operator Lite blocked: <reason>` instead of a tool result and must
change course. One gate, the same closed rule set as every other port — no
rule logic lives in this port.

Mechanism (verified against the SDK docs, 2026-09-29):

- **Tool input guardrails** — `function_tool(tool_input_guardrails=[...])`
  runs before the tool is invoked; the guardrail reads the raw arguments off
  `data.context.tool_arguments` ([ref/tool_guardrails/](https://openai.github.io/openai-agents-python/ref/tool_guardrails/),
  [ref/decorators/](https://openai.github.io/openai-agents-python/ref/decorators/)).
- **Deny form** — `ToolGuardrailFunctionOutput.reject_content(message)`:
  "Message to send to the model instead of the tool result" — the run stays
  alive, the model sees the reason. (`.raise_exception()` exists for a
  hard halt; we deliberately do not use it — the gate's contract is "the
  agent must change course", not "the run dies".)
- **Human approval** is the separate, orthogonal layer:
  `needs_approval=True` pauses the run (`result.interruptions`), then
  `state = result.to_state()`, `state.approve(item)` /
  `state.reject(item, rejection_message=...)`, resume with
  `Runner.run(agent, state)` ([human_in_the_loop/](https://openai.github.io/openai-agents-python/human_in_the_loop/)).
  The gate is deterministic and runs first either way.

## What it stops

The gate's 11 enforce-mode STOP rules, unchanged (27 rules total live in the
gate; the rest are shadow candidates):

| Rule | Name | Stops |
|---|---|---|
| OP-000 | protect-the-gate | Editing the gate, its rules, permits, mode or hook registrations |
| OP-001 | protect-workspaces | Deleting or moving a registered workspace root (or a parent of one) |
| OP-002 | protect-memory | Wiping, deleting or rewriting the Vestige store |
| OP-003 | no-blind-recursive-delete | Recursive delete outside scratch/build directories |
| OP-004 | no-history-destruction | Force-pushing over a shared branch, deleting `.git` |
| OP-005 | no-unreviewed-publish | Publishing a release/package, deleting a public repo |
| OP-006 | no-paid-deploy | Deploying the billing/control plane, acting on live billing |
| OP-007 | no-destructive-sql | `DROP`/`TRUNCATE`/unscoped `DELETE` against a database |
| OP-008 | no-shell-init-write | Writing shell rc/init files (code execution by install) |
| OP-009 | no-reverse-shell | `/dev/tcp`, `nc -e`, DNS-tunnel tools |
| OP-010 | no-cloud-metadata | Querying cloud metadata credential endpoints |

The gate parses the command, not the vibe: it tracks `cd`, variable
assignment, heredocs, command substitution, base64 blobs, ANSI-C quoting and
brace/glob expansion before judging targets. Receipts are hash-chained
digests under `~/.operator/receipts/` (integrity = reference digest, **not**
a signature).

## Install

```sh
sh ports/openai-agents/install.sh
```

Installs the gate to `~/.operator/gate/` (never overwrites a newer copy),
the module to `~/.operator/ports/openai-agents/`, and sets **shadow mode**
(log-only) unless you already chose a mode.

## Minimal usage

```python
from agents import Agent, Runner
from operator_guard import gated_shell_tool

@gated_shell_tool(name="run_shell")
async def run_shell(command: str) -> str:
    """Run a shell command locally. Every call passes the Operator gate first."""
    import subprocess
    p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return (p.stdout + p.stderr)[-4000:]

agent = Agent(name="ops", tools=[run_shell])
print(Runner.run_sync(agent, "list files in the current directory"))
```

Point Python at the installed module first:
`export PYTHONPATH="$PYTHONPATH:$HOME/.operator/ports/openai-agents"` (or copy
`operator_guard.py` beside your script — it is stdlib-only and imports fine
without the SDK installed).

Other entry points, for tools you do not create yourself:

- `operator_guard.input_guardrail()` — the raw
  `ToolInputGuardrail`, attach via `function_tool(tool_input_guardrails=[...])`
  or to an MCP server's `tool_input_guardrails`.
- `operator_guard.wrap_tool(existing_tool)` — gates any prebuilt
  `FunctionTool` in place by composing `on_invoke_tool`.
- `gated_shell_tool(..., needs_approval=True)` — combine the deterministic
  gate with a human pause: the run interrupts with a `ToolApprovalItem`, you
  `state.approve(item)` or `state.reject(item, rejection_message=...)`, then
  resume with `Runner.run(agent, state)`.

## Shadow → enforce

Shadow mode classifies, receipts, and lets everything through — read the log
first:

```sh
cat ~/.operator/receipts/$(date -u +%F).jsonl
echo enforce > ~/.operator/mode     # flip
```

`$OPERATOR_GATE_MODE` overrides the file; `$OPERATOR_GATE` overrides the gate
path (the module resolves `$OPERATOR_GATE` → its sibling
`operator-gate.py` → `~/.operator/gate/operator-gate.py`).

## Prove it

```sh
sh ports/openai-agents/test.sh
```

Hermetic (throwaway HOME under your real HOME), executes nothing: drives
`rm -rf ~/demo-factory/canary` through the module in shadow (allowed +
receipt) and enforce (blocked + reason), the `r''m` obfuscation, the
unreachable-gate fail-closed path on `tee ~/.zshrc`, and the receipt chain
verification. If the `openai-agents` package is installed, it also builds a
real gated tool and asserts the guardrail's `reject_content` deny form;
otherwise it says `UNTESTED` honestly.

## Limitations, stated honestly

- **Only command-carrying calls are gated.** A tool whose arguments carry no
  `command`/`cmd` key passes untouched. File-write tools (the SDK's file
  tools, MCP `write_file`, …) are not covered by this port — the gate's
  write-tool rules (OP-000/002/008/S14) fire only on write *tools it is
  wired to*, and this port wires shell-shaped calls.
- **Guardrails only guard what they are attached to.** A shell call made by
  any other tool (an unguarded MCP server, raw `LocalShellTool` executor
  code, model-builtin shell on other surfaces) bypasses this port entirely.
  `wrap_tool()` narrows that for prebuilt `FunctionTool`s.
- **`needs_approval` pauses, the gate decides.** They are different layers;
  approving around the gate is the owner's prerogative (`operator-gate
  approve`), not the agent's.
- The gate's own boundary applies verbatim: it blocks what its closed rule
  set recognises, routed through gated calls. Shell obfuscation it cannot
  parse is a residual gap; receipts are digest chains, not signatures.

Main repo: https://github.com/samvallad33/vestige — see
[operator-lite/README.md](../../README.md) and the
[port matrix](../README.md).
