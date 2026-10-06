# Operator Lite — Cursor port (official)

The Operator gate, wired into [Cursor](https://cursor.com)'s **agent hooks** —
the same host-agnostic gate that ships for Claude Code, Codex, Gemini CLI,
Goose, OpenClaw, OpenHands, OpenCode and Cline. Part of
[Vestige](https://github.com/samvallad33/vestige); see the
[main Operator Lite README](../../README.md) for the full rule table
(OP-000..OP-010 enforce, OP-S01..S17 shadow).

## The verified mechanism (and why)

Sources: [cursor.com/docs/agent/hooks](https://cursor.com/docs/agent/hooks)
(re-verified against the live docs while writing this port). Cursor reads hook
configs from `<project>/.cursor/hooks.json` and `~/.cursor/hooks.json` (user
scope; priority Enterprise → Team → Project → User), watches the files and
reloads them automatically. The schema is a required `"version": 1` plus a
`"hooks"` object keyed by event name; each entry is
`{command, timeout (SECONDS), matcher?, failClosed?, type?}`.

The payload is JSON on stdin — a common envelope
(`conversation_id`, `generation_id`, `model`, `hook_event_name`,
`cursor_version`, `workspace_roots`, `user_email`, `transcript_path`) merged
with event-specific fields:

```json
// beforeShellExecution
{ "command": "<full terminal command>", "cwd": "<working directory>", "sandbox": false }

// beforeMCPExecution — note: tool_input is a JSON STRING, not an object
{ "tool_name": "<tool name>", "tool_input": "<json params string>",
  "mcp_server_name": "<server key from mcp.json>", "url": "<server url>" }

// preToolUse — "fires for all tool types (Shell, Read, Write, MCP, Task, etc.)"
{ "tool_name": "Write", "tool_input": { "file_path": "..." }, "cwd": "...", "tool_use_id": "..." }
```

Permission hooks decide through **stdout JSON at exit 0**:

```json
{ "continue": true, "permission": "deny",
  "user_message": "<shown in the client>",
  "agent_message": "<sent to the agent>" }
```

Exit code 2 also blocks ("equivalent to returning `permission: deny`"), other
exit codes fail open, and — the sharp edge — **permission hooks block on
invalid or empty stdout even when `failClosed` is false**, so an allow must
still print schema-valid JSON. `failClosed: true` makes crashes, timeouts,
non-zero exits and empty output block instead of allow (the docs recommend it
for security-critical hooks). When multiple hooks match, all run and "any
`deny` wins over `ask`, and `ask` wins over `allow`".

The gate's own contract is different (exit 2 + stderr verdict, camelCase-free
field names), so this port is a thin adapter — `cursor-hook.py` — that
translates both directions and nothing else:

```
Cursor hook payload (stdin)
  └─ cursor-hook.py          field mapping (below) + verdict translation:
                             gate exit 0  -> {"permission": "allow"}  on stdout
                             gate exit 2  -> {"permission": "deny"}   on stdout,
                                             gate's stderr verdict becomes
                                             agent_message; user_message carries
                                             the rule id + approve digest
      └─ operator-gate.py hook --source cursor
          ├─ exit 0  -> allow (receipt already written)
          └─ exit 2  -> deny (the agent sees the full OPERATOR: STOPPED verdict)
```

| Cursor sends | Adapter maps to the gate |
|---|---|
| `command` (top level, beforeShellExecution) | `tool_name: "Shell"` + `tool_input: {"command": ...}` |
| `conversation_id` | `session_id` |
| `cwd`, else `workspace_roots[0]` | `cwd` (MCP events carry no cwd) |
| `tool_name` + `mcp_server_name` (beforeMCPExecution) | `mcp__<server>__<tool>` so the gate's MCP rules (OP-S02, OP-S13) fire |
| `tool_input` as a JSON **string** (beforeMCPExecution) | parsed into an object; unparsable strings pass through for the gate to wrap |
| `MCP:<tool>` spelling (preToolUse) | `mcp__<tool>` |

Three hooks are registered, with **disjoint surfaces**:

- `beforeShellExecution` (matcher `*`) — every agent shell command
- `beforeMCPExecution` — every MCP tool call (the docs list no matcher: all calls)
- `preToolUse` (matcher `^(Write|Edit|MultiEdit|Delete|NotebookEdit)$`) —
  native file tools only, which is what the dedicated hooks don't cover.
  Forwarding the same action through two hooks would consume a single-use
  `operator-gate approve` permit on the first invocation and block the second,
  so Shell and MCP are deliberately excluded here. The gate, not the adapter,
  decides relevance.

Hook commands use the adapter's **absolute path**, substituted at install time:
the docs specify that relative command paths resolve against the hooks.json
directory, and `~` expansion is not documented — an absolute path works under
either spawn model.

## What it stops

Everything the gate stops on any host: edits to the gate/hook/permit files
itself (OP-000), workspace-root deletion (OP-001), Vestige store wipes
(OP-002), blind recursive deletes and sweeps from `/` (OP-003), git history
destruction (OP-004), unreviewed publishes (OP-005), paid deploys (OP-006),
destructive SQL (OP-007), shell-init writes (OP-008), reverse shells (OP-009),
cloud-metadata exfil (OP-010) — plus the shadow-only classes (work-loss git,
outgoing comms, public mutation, credential-shaped writes,
pipe-to-interpreter, persistence, env hijacks, exfil shapes, invisible
characters, MCP argument exfil, agent-config writes, paste/tunnel egress, new
git remotes, sandbox-escape primitives), recorded in receipts and promotable
per rule via `commitments.json`.

**GuardFall-proof:** the gate replays a 43-case corpus of bypass techniques
(quote reassembly, `$IFS`, command substitution, ANSI-C escapes, base64
bodies, brace/glob enumeration, subshell time-bombs, cd tracking, heredocs,
fork bombs, `mv x /dev/null`, …) and must pass 43/43:

```
python3 ~/.operator/gate/operator-gate.py corpus guardfall
```

Every decision appends a hash-chained receipt to `~/.operator/receipts/`
(`operator-gate verify` walks the chain). Receipts are digests, not
signatures.

## Install (2 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/cursor/install.sh               # gate + adapter + ~/.cursor/hooks.json hooks, mode=shadow
sh ports/cursor/test.sh                  # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub; never overwrites an existing same-or-newer copy)
- installs the adapter to `~/.operator/gate/cursor-hook.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- merges the three hooks into `~/.cursor/hooks.json` (user scope, every
  project; a timestamped backup is written first; existing hooks and settings
  are preserved; an invalid existing file is left untouched and manual steps
  are printed)

Requires `python3` on PATH. Cursor watches hooks config files and reloads
them automatically; if the hooks do not load, restart Cursor. The bundled
`hooks.json` in this directory is the reference template
(`REPLACE_WITH_HOME` is substituted by the installer).

## Shadow → enforce

Start in shadow: every decision is analyzed and receipted, nothing is blocked.
When the log looks right, flip:

```
echo enforce > ~/.operator/mode
```

In enforce mode a blocked call returns the documented deny JSON: the model
sees the gate's full verdict (`agent_message`) — rule, why, the compliant
path — and the `user_message` carries the one-time `operator-gate approve
<digest>` the owner can grant from their own terminal.

## Tested

`test.sh` is hermetic (a throwaway HOME; the real `~/.cursor` and
`~/.operator` are never touched) and drives the real installed adapter with
payloads in the exact stdin shapes documented above:

- pre-existing `hooks.json` content (an unrelated `beforeReadFile` hook)
  survives the merge; a timestamped backup is written; `version: 1`, the
  absolute adapter path, `failClosed: true` and the matchers all verified
- `rm -rf ~/demo-factory/canary` → shadow: exit 0 + allow JSON + OP-003
  SHADOW_STOP receipt; enforce: exit 0 + `permission: "deny"` JSON whose
  `agent_message` carries `OPERATOR: STOPPED` and whose `user_message`
  carries OP-003 and the approve digest
- `r''m -rf` (quote reassembly) → same verdicts
- `ls -la` → allow JSON in both modes, receipted with no rule hits
- `rm -rf ./src` inside the session cwd → allow (the gate is precise, not
  trigger-happy)
- `beforeMCPExecution` with `tool_input` as a JSON string → receipted as
  `mcp__postmark__send_email`, OP-S13 for the `~/.ssh/id_rsa` argument,
  `workspace_roots[0]` used as the cwd fallback
- `preToolUse` `Write` to `~/.zshrc` → OP-008 deny in enforce
- gate missing + plainly destructive text → deny (fail closed); gate missing +
  benign → allow; malformed and empty stdin → allow — and in every case the
  stdout is schema-valid JSON (Cursor blocks invalid output on permission
  hooks even without `failClosed`)
- no payload is ever executed — no `demo-factory`, no `./src`, no `~/.zshrc`
  is ever created; the gate is analyzer-only. Receipt chain verifies.

## Honest limitations

- **Host wiring is untested against a live Cursor session.** The schema,
  payload shapes, deny JSON, exit codes, `failClosed` and merge semantics
  above come from the official docs (linked); the adapter and the gate
  contract are tested end-to-end locally by `test.sh`, but this port has not
  been run inside Cursor itself. Status: `gate-tested`. To confirm: run
  `install.sh`, ask the agent to run `echo OPERATOR_CANARY_STOP_7f3a`, and
  check `~/.operator/receipts/<today>.jsonl`.
- **MCP coverage is receipt-grade, not block-grade.** `beforeMCPExecution`
  is verified and wired, but the gate rules that key on MCP calls (OP-S02
  message/send, OP-S13 secret-shaped args and sensitive paths) are
  shadow-class: in enforce mode an MCP call carrying `~/.ssh/id_rsa` is
  receipted, not blocked. Blocking MCP surface needs an upstream gate change
  (promoting OP-S13 via `commitments.json` mode overrides works per rule);
  this port never modifies the gate. `failClosed: true` still blocks MCP
  calls outright if the adapter cannot run at all.
- **preToolUse tool names are partially documented.** The docs confirm the
  hook "fires for all tool types" and list `Write`, `Delete`, `Shell`, `Read`,
  `Grep`, `Task`; `Edit` is plausible but not listed. The matcher covers the
  known mutating names; unknown ones pass ungated by this hook (and shell/MCP
  remain covered by their dedicated hooks regardless).
- **Agent-run commands, as documented.** The docs describe
  `beforeShellExecution` as firing before commands the agent executes; they
  do not promise that commands typed by the human in Cursor's own terminal
  fire hooks. Treat the human's terminal as outside the gate — the same
  boundary every port has.
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but `~/.cursor/hooks.json` (where removing an entry could unregister
  this port) is not yet in `self_protected_files()` — a rogue agent could try
  to disable the port by editing it. The receipts make the silence visible;
  wiring that path into the gate is an upstream gate change, and this port
  never modifies the gate.
- **failClosed is shipped ON.** If the adapter cannot run at all (python3
  missing, file deleted), Cursor blocks the action instead of allowing it —
  the strict posture the docs recommend for security hooks. Remove the
  `"failClosed": true` lines in `~/.cursor/hooks.json` to relax.
- **No human-approval bridge through the Cursor UI.** There is no documented
  external approval command a program could answer, so `operator-gate
  approve` one-time permits are granted from the owner's terminal only; the
  deny `user_message` surfaces the digest to make that copy-pasteable.
- The gate's closed rule set has residual gaps (shell obfuscation it cannot
  parse, actions outside hooked tools). It raises the floor; it is not a
  sandbox.

## Files

| File | Role |
|---|---|
| `hooks.json` | Reference template (installer writes the absolute path into `~/.cursor/hooks.json`) |
| `cursor-hook.py` | Payload translator + verdict translation + gate runner (stdlib only) |
| `install.sh` | gate install + hooks.json merge (backup, idempotent), shadow default |
| `test.sh` | hermetic end-to-end tests (real install.sh, real adapter, exact payload shapes) |

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
