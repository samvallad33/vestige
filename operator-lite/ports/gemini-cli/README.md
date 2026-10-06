# Operator Lite — Gemini CLI port

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[gemini-cli](https://github.com/google-gemini/gemini-cli)'s `BeforeTool` hooks.
One stdlib-only Python file does all analysis; this port is a thin adapter: a
payload reshaper plus install/test scripts. It contains no rule logic.

## Install (2 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/gemini-cli/install.sh               # gate + adapter + ~/.gemini/settings.json hook, mode=shadow
sh ports/gemini-cli/test.sh                  # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub; never overwrites an existing same-or-newer copy)
- installs the adapter to `~/.operator/gate/gemini-wrapper.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- merges the `BeforeTool` hook into `~/.gemini/settings.json` (a timestamped
  backup is written first; existing settings and existing hooks are preserved;
  an invalid existing file is left untouched and manual steps are printed)

Requires `python3` on PATH. Hooks are on by default in gemini-cli
(`hooksConfig.enabled` defaults to `true`).

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

## How the wiring works (verified against the docs and the source)

Sources: [geminicli.com/docs/hooks](https://geminicli.com/docs/hooks),
[the hooks reference](https://geminicli.com/docs/hooks/reference), and the
gemini-cli source (`packages/core/src/hooks/{hookRunner,hookPlanner,types}.ts`,
`packages/cli/src/config/settingsSchema.ts`,
`packages/core/src/tools/definitions/base-declarations.ts`).

- Hooks live in `settings.json` under a top-level `"hooks"` key, object keyed
  by PascalCase event name; `hooks.BeforeTool` is an array of matcher groups
  `{"matcher", "hooks": [{"type": "command", "command", "timeout", "name"}]}`.
  User scope is `~/.gemini/settings.json` (project `.gemini/settings.json`
  takes precedence if a project also defines hooks). The `matcher` is a
  regular expression against the tool name; `"*"` or `""` matches everything.
  `timeout` is in milliseconds (default 60000; this port uses 10000).
- gemini-cli pipes the hook this payload on stdin:

  ```json
  {"session_id": "…", "transcript_path": "…", "cwd": "/workspace",
   "hook_event_name": "BeforeTool", "timestamp": "…",
   "tool_name": "run_shell_command",
   "tool_input": {"command": "rm -rf /tmp/data"}}
  ```

- Exit codes (verified in `hookRunner.ts`): 0 = allow; 1 = non-fatal warning
  (proceeds, shows stderr as a warning); **any other code, including 2, denies
  the call and uses the stderr text as the reason sent to the agent**. The
  gate emits exactly 0 and 2, so the mapping is exact — the gate's
  `OPERATOR: STOPPED (...)` verdict becomes the tool denial the model sees. A
  stdout `{"decision": "deny"}` JSON path also exists; the gate doesn't need
  it because its exit-code contract already lands on deny.

The adapter (`gemini-wrapper.py`, installed next to the gate) makes one
reshape and nothing else:

1. Gemini's editing tool is `replace` (args `file_path`, `old_string`,
   `new_string`); the gate's write-tool registry knows that shape as `edit`.
   Renaming it gives file edits full write-path coverage (OP-000, OP-002,
   OP-008, OP-S14). Gemini's other mutating tool, `write_file`, is already a
   name the gate knows, and the shell tool `run_shell_command` is in the
   gate's shell registry, so both pass through untouched. (The shell path
   would in fact work with no wrapper at all — every payload field already
   matches by name — the wrapper exists for the `replace` coverage.)

Hook commands are run through the user's shell (`spawn(executable,
[argsPrefix, command])` in `hookRunner.ts`), so `~` in the hook command
expands. Hooks are read at session start — open a new gemini session after
installing; the `/hooks` command lists what registered. To disable without
uninstalling: `"hooksConfig": {"disabled": ["operator-lite"]}` in
`~/.gemini/settings.json`.

## Tested

`test.sh` is hermetic (a throwaway HOME; the real `~/.gemini` and
`~/.operator` are never touched) and drives the real installed hook command
with payloads in the exact stdin shape above:

- pre-existing `settings.json` content (a `theme` key and an unrelated
  `AfterTool` hook) survives the merge; the `BeforeTool` entry is appended
- `rm -rf ~/demo-factory/canary` → shadow: exit 0 + OP-003 receipt; enforce: exit 2
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → same verdicts
- `ls -la` → exit 0 in both modes, receipted with no rule hits
- a `replace` tool call editing `~/.zshrc` → OP-008 receipt whose
  `action_preview` starts with `edit` (proves the reshape end-to-end)
- receipt `cwd` equals the gemini payload's `cwd`; receipt chain verifies;
  no `~/demo-factory` directory is ever created; payloads are never executed
  — the gate is analyzer-only

## Honest limitations

- **Host wiring is untested against a live gemini session.** The mechanism,
  schema, payload and exit codes above come from the official docs and the
  gemini-cli source (linked); the gate contract is tested end-to-end locally
  by `test.sh`, but this port has not been run inside a live `gemini` CLI
  session. Status: `gate-tested`.
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but `~/.gemini/settings.json` (where `hooksConfig.disabled` or a
  deleted hook could unregister this port) is not yet in
  `self_protected_files()` — a rogue agent could try to disable the port by
  editing it. The receipts make the silence visible; wiring that path into
  the gate is an upstream gate change, and this port never modifies the gate.
- **MCP tool names.** Gemini CLI namespaces discovered MCP tools as
  `mcp_{server}_{tool}` (verified in `mcp-tool.ts`), not the `mcp__` prefix
  the gate's OP-S02/S13 rules key on. MCP calls with a `command` argument
  still get analyzed by the gate's command-key fallback; message/send and
  sensitive-path MCP rules may not fire. As with every port, tools the gate
  does not recognize pass through unjudged.
- **Other native paths.** gemini-cli also runs shell through its interactive
  `!` prompt and background-shell utilities (`list_background_processes`,
  `read_background_output` — read-only). Anything that does not arrive as a
  `BeforeTool` hook call is invisible to the gate; this is the gate's
  boundary on every host, not gemini-specific.
- **Fail-open direction.** gemini-cli treats exit 1 (and any hook spawn
  failure) as a warning and proceeds — the same fail-open direction as the
  gate's own degraded mode; the adapter fails open with a loud stderr note
  when the gate is missing. The gate itself fails closed only for plainly
  destructive text after an internal error.
- The gate's closed rule set has residual gaps (shell obfuscation it cannot
  parse, actions outside hooked tools). It raises the floor; it is not a
  sandbox.
- **No human-approval bridge.** gemini-cli has no external approval command a
  program could answer, so `operator-gate approve` one-time permits are
  granted from the owner's terminal only.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
