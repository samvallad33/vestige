# Operator Lite — OpenHands port

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
OpenHands' PreToolUse hooks. One stdlib-only Python file does all analysis;
this port is a thin adapter: a payload reshaper plus install/test scripts. It
contains no rule logic.

## Install (3 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/openhands/install.sh                # gate + adapter + .openhands/hooks.json, mode=shadow
sh ports/openhands/test.sh                   # optional: proves the chain end-to-end
```

`install.sh [project-dir]` (default: the current directory) is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub; never overwrites an existing same-or-newer copy)
- installs the adapter to `~/.operator/gate/openhands-wrapper.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- merges the hook into `<project>/.openhands/hooks.json` (existing hooks are
  preserved; an invalid existing file is left untouched and reported)

## Shadow → enforce

Start in shadow: every decision is analyzed and receipted, nothing is blocked.
When the log looks right, flip:

```
echo enforce > ~/.operator/mode
```

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
| OP-008 | shell-init writes (`>> ~/.zshrc` = code execution by install) |
| OP-009 | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |

Plus shadow-only OP-S01..S17 (work-loss git, outgoing comms, public mutation,
credential-shaped writes, pipe-to-interpreter, persistence, env hijacks,
exfil shapes, invisible characters, MCP argument exfil, agent-config writes,
paste/tunnel egress, new git remotes, sandbox-escape primitives), recorded in
receipts and promotable per rule via `commitments.json`.

**GuardFall-proof:** the gate replays a 43-case corpus of GuardFall bypass
techniques (quote reassembly, `$IFS`, command substitution, ANSI-C escapes,
base64 bodies, brace/glob enumeration, subshell time-bombs, session variables,
cd tracking, heredocs, fork bombs, `mv x /dev/null`, …) and must pass 43/43:

```
python3 ~/.operator/gate/operator-gate.py corpus guardfall
```

Every decision appends a hash-chained receipt to `~/.operator/receipts/`
(`operator-gate verify` walks the chain). Receipts are digests, not
signatures.

## How the wiring works (verified against the docs)

Source: [docs.openhands.dev — Hooks](https://docs.openhands.dev/openhands/usage/customization/hooks)

- File: `.openhands/hooks.json` in the project root; `pre_tool_use` entries of
  the shape `{"matcher", "hooks": [{"type": "command", "command", "timeout"}]}`.
  (`matcher: "*"` sends every tool call through the gate; the gate itself
  decides which calls carry shell or write semantics.)
- OpenHands pipes the hook this payload on stdin:

  ```json
  {"event_type": "PreToolUse", "tool_name": "terminal",
   "tool_input": {"command": "rm -rf /tmp/data"},
   "session_id": "abc-123", "working_dir": "/workspace"}
  ```

- Exit codes: 0 = proceed, 2 = deny, any other code = error (proceed + log).
  The gate emits exactly 0 and 2.

The adapter (`openhands-wrapper.py`, installed next to the gate) makes two
reshapes and nothing else:

1. OpenHands names the directory `working_dir`; the gate reads `cwd`. Without
   this, relative delete targets are unresolvable and get downgraded to the
   unresolved shadow rule instead of being judged.
2. OpenHands' mutating file tool `file_editor` (ops `create`, `str_replace`,
   `insert`) is renamed to `str_replace_editor`, a write-tool name the gate
   already knows, so file writes get full write-path coverage (OP-000, OP-002,
   OP-008, OP-S14). Read-only `view` calls are left alone.

Hooks load at conversation start — restart a conversation after installing.
In the OpenHands CLI, `/skills` lists the loaded hooks.

## Tested

`test.sh` is hermetic (a throwaway HOME and project; the real `~/.operator`
is never touched) and drives the real `hooks.json` command with payloads in
the exact stdin shape above:

- `rm -rf ~/demo-factory/canary` → shadow: exit 0 + OP-003 receipt; enforce: exit 2
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → same verdicts
- `ls -la` → exit 0 in both modes, receipted with no rule hits
- `rm -rf vestige` with `working_dir: ~/Developer` → OP-001 receipt whose
  `cwd` equals the OpenHands `working_dir` (proves the reshape end-to-end)
- receipt chain verifies; no `~/demo-factory` directory is ever created;
  payloads are never executed — the gate is analyzer-only

## Honest limitations

- **Host wiring is untested against live OpenHands.** The mechanism, schema,
  payload and exit codes above come from the official docs (linked); the gate
  contract is tested end-to-end locally by `test.sh`, but this port has not
  been run inside a live OpenHands conversation. Status: `gate-tested`.
- Hooks only see calls routed through hooked tools. A tool that bypasses the
  hook chain is invisible to the gate (this is the gate's boundary on every
  host, not OpenHands-specific).
- Tool-name coverage: `terminal` is the gate's shell path; `execute_bash` and
  any tool whose input carries a `command` key are caught by the gate's
  fallback. `file_editor` mutations are covered via the reshape above. Tools
  the gate does not recognize pass through unjudged. MCP tool coverage
  (OP-S02/S13) assumes `mcp__`-prefixed tool names; if OpenHands names MCP
  tools differently, those rules may not fire.
- OpenHands treats any hook exit other than 0/2 as an error and proceeds —
  the same fail-open direction as the gate's own degraded mode. A wrapper or
  gate crash therefore fails open (except that the gate itself fails closed
  for plainly destructive text after an internal error).
- The gate's closed rule set has residual gaps (shell obfuscation it cannot
  parse, actions outside hooked tools). It raises the floor; it is not a
  sandbox.
- The SDK's `SecurityAnalyzer` / confirmation-policy API is a separate
  mechanism (analyzers classify risk; they do not hard-deny). This port uses
  the hooks path, which is the documented hard-deny (`exit 2`) mechanism.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
