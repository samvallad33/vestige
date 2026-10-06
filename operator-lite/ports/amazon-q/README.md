# Operator Lite — Amazon Q Developer CLI port

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[Amazon Q Developer CLI](https://github.com/aws/amazon-q-developer-cli) (now
branded Kiro CLI) `preToolUse` agent hooks. One stdlib-only Python file does
all analysis; this port is a thin adapter: a payload reshaper plus
install/test scripts. It contains no rule logic.

## Install (2 commands)

```
git clone https://github.com/samvallad33/vestige && cd vestige/operator-lite   # skip if already cloned
sh ports/amazon-q/install.sh               # gate + adapter + agent hook, mode=shadow
sh ports/amazon-q/test.sh                  # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub; never overwrites an existing same-or-newer copy)
- installs the adapter to `~/.operator/gate/q-wrapper.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- wires the `preToolUse` hook into Q's agent config — your configured default
  agent's file if `chat.defaultAgent` points at an existing one, otherwise
  `~/.aws/amazonq/cli-agents/q_cli_default.json` (created mirroring Q's
  built-in default if it doesn't exist; a timestamped backup is written
  before any merge; existing fields and hooks are preserved; an invalid
  existing file is left untouched and manual steps are printed)
- warns if a workspace `.amazonq/cli-agents/` copy shadows the global file it
  wired (workspace agents win by name)

Requires `python3` on PATH. Hook entries use `timeout_ms: 10000` (Q's default
is 30000; the gate is stdlib and finishes in well under a second).

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
cd tracking, heredocs, fork bombs, `mv x /dev/null`, …) and must pass 43/43 —
`test.sh` replays the whole corpus through this adapter, so the pass is
adapter-inclusive:

```
python3 ~/.operator/gate/operator-gate.py corpus guardfall   # gate alone (repo checkout)
sh ports/amazon-q/test.sh                                    # corpus through the adapter
```

Every decision appends a hash-chained receipt to `~/.operator/receipts/`
(`operator-gate verify` walks the chain). Receipts are digests, not
signatures.

## How the wiring works (verified against the docs and the source)

Sources: [docs/hooks.md](https://github.com/aws/amazon-q-developer-cli/blob/main/docs/hooks.md),
[docs/agent-format.md](https://github.com/aws/amazon-q-developer-cli/blob/main/docs/agent-format.md),
[docs/agent-file-locations.md](https://github.com/aws/amazon-q-developer-cli/blob/main/docs/agent-file-locations.md),
[docs/default-agent-behavior.md](https://github.com/aws/amazon-q-developer-cli/blob/main/docs/default-agent-behavior.md),
and the Q CLI source (`crates/chat-cli/src/cli/agent/hook.rs`,
`crates/chat-cli/src/cli/chat/cli/hooks.rs`,
`crates/chat-cli/src/cli/chat/mod.rs`,
`crates/chat-cli/src/cli/agent/mod.rs`,
`crates/chat-cli/src/cli/chat/tools/mod.rs`,
`crates/chat-cli/src/cli/chat/tools/tool_index.json`,
`crates/chat-cli/src/util/paths.rs`,
`crates/chat-cli/src/database/settings.rs`).

- Hooks live in an **agent file** (JSON), under the top-level `"hooks"`
  object keyed by camelCase event names: `agentSpawn`, `userPromptSubmit`,
  `preToolUse`, `postToolUse`, `stop`. Each entry is
  `{"command", "timeout_ms"?, "max_output_size"?, "cache_ttl_seconds"?,
  "matcher"?}` — snake_case keys, exactly the `Hook` struct in
  `agent/hook.rs` (no casing rename, `deny_unknown_fields` on the agent
  schema). Global agent files: `~/.aws/amazonq/cli-agents/<name>.json`;
  workspace: `.amazonq/cli-agents/<name>.json` (workspace wins on name
  conflict). Plain `q chat` runs Q's in-memory default agent, which a file
  named `q_cli_default.json` overrides — that (or your own
  `chat.defaultAgent` file) is where install wires the hook.
- `matcher` is optional; **no matcher (or `"*"`) means the hook runs for
  every tool** (`hook_matches_tool` in `chat/cli/hooks.rs`). This port omits
  the matcher so every call is seen. `cache_ttl_seconds` defaults to 0, so
  the hook re-executes on every call (no caching of verdicts).
- Q spawns each hook as `bash -c <command>` (unix) with the payload on
  stdin, this shape (verbatim construction in `hooks.rs` — note there is no
  `session_id` field):

  ```json
  {"hook_event_name": "preToolUse", "cwd": "/workspace",
   "tool_name": "execute_bash",
   "tool_input": {"command": "rm -rf /tmp/data", "summary": "..."}}
  ```

  MCP tools arrive namespaced as `@server/tool`.
- Exit codes (verified in `chat/mod.rs`, which comments the contract verbatim):
  0 = allow; **2 = block, stderr returned to the model as
  "PreToolHook blocked the tool execution: <stderr>"**; any other code = the
  CLI shows a warning and allows the tool. The gate emits exactly 0 and 2,
  so the mapping is exact — the gate's `OPERATOR: STOPPED (...)` verdict
  becomes the block the model sees.
- Built-in tool shapes (from `tools/tool_index.json`): `execute_bash` input
  is `{command (required), summary?}`; `fs_write` input is
  `{command: create|str_replace|insert|append, path (both required),
  file_text?, old_str?, new_str?, insert_line?}`.

The adapter (`q-wrapper.py`, installed next to the gate) makes two reshapes
and nothing else:

1. Q's shell tool `execute_bash` (Windows twin: `execute_cmd`) is renamed to
   `bash`, a name in the gate's shell registry, so the shell branch always
   engages deterministically. (The shell path would in fact work with no
   wrapper at all — `tool_input.command` already matches by name — the
   wrapper exists to make it explicit.)
2. Q's file-write tool `fs_write` is renamed to `write` (a name in the
   gate's write-tool registry), which gives file writes full write-path
   coverage (OP-000, OP-002, OP-008, OP-S14) via its `path` field. Without
   this, `fs_write` would be actively misjudged: its enum `command` field
   (`create`/`str_replace`/…) collides with the gate's command-key fallback.
   The body is re-exposed as `content` / `new_string` — the names the
   gate's credential-shaped-write check (OP-S04) reads.

Everything else (`fs_read`, `use_aws`, `knowledge`, `thinking`, `todo_list`,
`gh_issue`, `introspect`, `delegate`, `@server/tool`) passes through
untouched. Agent files are read at session start — open a new `q chat`
session after installing.

## Tested

`test.sh` is hermetic (a throwaway HOME; the real `~/.aws` and `~/.operator`
are never touched) and drives the real installed hook command
(`bash -c 'python3 "$HOME/.operator/gate/q-wrapper.py"'` — Q's exact spawn)
with payloads in the exact stdin shape above:

- fresh install creates `q_cli_default.json` mirroring Q's built-in default
  (`tools: ["*"]`, workspace resources + rules glob, `useLegacyMcpJson`) plus
  the hook; a re-install over a pre-existing agent file merges without
  clobbering and writes a backup
- `execute_bash: rm -rf ~/demo-factory/canary` → shadow: exit 0 + OP-003
  receipt; enforce: exit 2 with `OPERATOR: STOPPED` + `OP-003` on stderr
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → same verdicts
- `ls -la` → exit 0 in both modes, receipted with no rule hits
- an `fs_write` create of `~/.zshrc` → receipt whose preview starts with
  `write` and carries OP-008 (proves the rename end-to-end); enforce: exit 2
- an `fs_write` with a credential-shaped `file_text` → OP-S04 shadow receipt
  (proves the body remap)
- all 43 GuardFall corpus cases replayed through the adapter with Q-shaped
  payloads (reshape → gate classify), 43/43 expected verdicts
- receipt `cwd` equals the Q payload's `cwd`; receipt chain verifies;
  no `~/demo-factory` directory is ever created; payloads are never executed
  — the gate is analyzer-only

## Honest limitations

- **Host wiring is untested against a live `q chat` session.** The mechanism,
  schema, payload, exit codes and tool shapes above come from the official
  docs and the amazon-q-developer-cli source (linked); the gate contract is
  tested end-to-end locally by `test.sh`, but this port has not been run
  inside a live Q CLI session. Status: `gate-tested`.
- **Only the wired agent is gated.** The hook lives in one agent file
  (`q_cli_default.json` or your `chat.defaultAgent` target). Other agent
  files, workspace agent files that shadow the global one (install warns),
  and `q chat --agent <other>` sessions do not run the gate until the same
  `hooks` entry is added to those files (fragment:
  [`agent-hook.json`](agent-hook.json)).
- **MCP tool names.** Q namespaces MCP tools as `@server/tool`, not the
  `mcp__` prefix the gate's OP-S02/S13 rules key on. MCP calls with a
  `command` argument still get analyzed by the gate's command-key fallback;
  message/send and sensitive-path MCP rules may not fire. As with every
  port, tools the gate does not recognize pass through unjudged — notably
  `use_aws` API calls are not judged by any rule.
- **No session id in receipts.** Q's hook payload has no `session_id`
  field, so receipts record an empty session string for this host.
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but `~/.aws/amazonq/cli-agents/*.json` (where a removed `hooks`
  entry could unregister this port) is not yet in `self_protected_files()`
  — a rogue agent could try to disable the port by editing it. The receipts
  make the silence visible; wiring those paths into the gate is an upstream
  gate change, and this port never modifies the gate.
- **Fail-open direction.** Q CLI treats any nonzero exit other than 2 as a
  warning and proceeds (hook spawn failures, timeouts, exit 1) — the same
  fail-open direction as the gate's own degraded mode; the adapter fails
  open with a loud stderr note when the gate is missing. The gate itself
  fails closed only for plainly destructive text after an internal error.
- **Subagents.** `delegate`-spawned subagent tool calls are not verified to
  pass through the parent's preToolUse hooks; treat subagent actions as
  outside the gate until tested against a live session.
- The gate's closed rule set has residual gaps (shell obfuscation it cannot
  parse, actions outside hooked tools). It raises the floor; it is not a
  sandbox.
- **No human-approval bridge.** Q CLI has no external approval command a
  program could answer, so `operator-gate approve` one-time permits are
  granted from the owner's terminal only.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
