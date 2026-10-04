# Operator Lite — Windsurf (Cascade) port (official)

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[Windsurf](https://windsurf.com)'s Cascade agent through its **Cascade Hooks**
(pre-hooks on `pre_run_command`, `pre_write_code` and `pre_mcp_tool_use` —
the three blockable hook events that carry tool actions). One stdlib-only
Python file does all analysis; this port is a thin adapter: a payload
reshaper plus config/install/test scripts. It contains no rule logic.

## The verified mechanism (and why)

Verified against the official docs at
[docs.devin.ai/desktop/cascade/hooks](https://docs.devin.ai/desktop/cascade/hooks)
(Windsurf's agent docs now live under the Devin Desktop name — same product,
same Cascade hooks):

- **Config locations** — JSON files at three levels, all merged and all
  executed (system → user → workspace):
  - **User level** (what `install.sh` wires by default):
    `~/.codeium/windsurf/hooks.json` (Devin Desktop IDE). The JetBrains
    plugin variant is `~/.codeium/hooks.json` — same schema, one path differs.
  - **Workspace level** (optional `install.sh <workspace-dir>` argument):
    `.devin/hooks.json` in the workspace root, with the legacy
    `.windsurf/hooks.json` used only when the `.devin` file is absent or
    defines no hooks. The installer merges into an existing
    `.devin/hooks.json` when there is one, and never creates a
    `.windsurf/hooks.json` that a `.devin` file would shadow.
  - **System level** (enterprise/MDM deployment, not touched by this port):
    `/Library/Application Support/Devin/hooks.json` (macOS), `/etc/devin/hooks.json`
    (Linux/WSL), `C:\ProgramData\Devin\hooks.json` (Windows), each with a
    legacy Windsurf path fallback.
- **Entry schema** — a top-level `"hooks"` object keyed by snake_case event
  name; each event maps to an **array of flat objects**:
  `{"command" (required), "powershell"?, "show_output"?, "working_directory"?}`.
  There is **no matcher, no timeout and no name field** — every call on a
  wired event is forwarded, and the gate (not the config) decides relevance.
  `command` runs via `bash -c` on macOS/Linux, so `"$HOME"` in the wired
  command expands. Hooks don't run in Restricted Mode.
- **Stdin payload** — *not* the gate's contract; this is why the port needs a
  wrapper (Crush's payload is byte-identical to the gate's; Windsurf's is not):

  ```json
  {"agent_action_name": "pre_run_command",
   "trajectory_id": "…", "execution_id": "…", "timestamp": "ISO 8601",
   "model_name": "…",
   "tool_info": {"command_line": "npm install package-name",
                 "cwd": "/Users/yourname/project"}}
  ```

  `pre_write_code` carries `tool_info: {file_path, edits: [{old_string, new_string}]}`;
  `pre_mcp_tool_use` carries `tool_info: {mcp_server_name, mcp_tool_name,
  mcp_tool_arguments}`. Common fields on every event: `agent_action_name`,
  `trajectory_id`, `execution_id`, `timestamp`, `model_name`, `tool_info`.
- **Exit codes** — `0` = success, action proceeds; `2` = **the action is
  blocked and Cascade shows the stderr text to the agent** (pre-hooks only);
  *any other code* = error, **the action proceeds normally** (fail open, with
  the error visible). Only pre-hooks can block:
  `pre_user_prompt`, `pre_read_code`, `pre_write_code`, `pre_run_command`,
  `pre_mcp_tool_use`; post-hooks never can.
- **No timeout knob** — the docs set no per-hook timeout and ask for
  sub-100ms hooks. The wrapper adds one Python startup plus the gate's
  millisecond analysis per wired call; see limitations.

The adapter (`windsurf-wrapper.py`, installed next to the gate) makes exactly
three reshapes and nothing else:

1. `pre_run_command` → gate tool `run_terminal_cmd` (a name already in the
   gate's shell registry) with `tool_input.command := tool_info.command_line`
   and `cwd := tool_info.cwd`.
2. `pre_write_code` → gate tool `edit` (a write-tool name the gate knows) with
   `file_path` and the joined `edits[]` strings, so file writes get full
   write-path coverage (OP-000 / OP-002 / OP-008 / OP-S14).
3. `pre_mcp_tool_use` → gate tool name `mcp__<server>__<tool>` with
   `tool_input := mcp_tool_arguments`, so the gate's MCP rules (OP-S02
   outbound comms, OP-S13 secret/sensitive-path arguments) fire by prefix.

`session_id := trajectory_id` (the conversation id; `execution_id` is per-turn).
For write/MCP events the payload carries no `cwd`, so the wrapper passes its
own process cwd — Windsurf runs hook commands from the workspace root by
default. Every call still lands in the receipt either way.

```
Windsurf pre-hook payload (stdin)
  └─ python3 "$HOME/.operator/gate/windsurf-wrapper.py"
      └─ python3 "$HOME/.operator/gate/operator-gate.py" hook --source windsurf
          ├─ exit 0, empty stdout -> allow (receipt already written)
          └─ exit 2, stderr "OPERATOR: STOPPED ..." -> block; Cascade shows
             the agent that stderr as the block reason
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
sh ports/windsurf/install.sh                # user level: every workspace
sh ports/windsurf/install.sh <workspace>    # or one workspace only
sh ports/windsurf/test.sh                   # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub unless an existing copy is same-or-newer; never overwrites a newer
  gate; refuses to wire a gate that does not parse)
- installs the adapter to `~/.operator/gate/windsurf-wrapper.py`
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- merges the three pre-hooks into `~/.codeium/windsurf/hooks.json` (or the
  workspace's `.windsurf/hooks.json` / existing `.devin/hooks.json`): existing
  hooks are preserved, the original file is backed up to
  `hooks.json.bak-<timestamp>` first, and a file that is not valid JSON is
  left untouched with manual merge instructions printed
- requires `python3`; needs `curl`, `wget`, or Python's urllib for the
  download when no sibling gate exists

The reference fragment is [hooks.json](hooks.json).

## Manual merge

If an existing `hooks.json` cannot be merged automatically (not valid JSON,
or `hooks` has an unexpected shape), nothing is modified. Add this alongside
your existing keys:

```json
{
  "hooks": {
    "pre_run_command": [
      { "command": "python3 \"$HOME/.operator/gate/windsurf-wrapper.py\"" }
    ],
    "pre_write_code": [
      { "command": "python3 \"$HOME/.operator/gate/windsurf-wrapper.py\"" }
    ],
    "pre_mcp_tool_use": [
      { "command": "python3 \"$HOME/.operator/gate/windsurf-wrapper.py\"" }
    ]
  }
}
```

## Shadow → enforce

Start in shadow: every decision is analyzed and receipted, nothing is blocked.
When the log looks right, flip:

```
echo enforce > ~/.operator/mode
```

In enforce mode a blocked call shows the Cascade agent the gate's full
verdict — rule, why, the compliant path — and receipts carry the one-time
`operator-gate approve` digest the owner can grant from their own terminal.

## Tested

`test.sh` is hermetic (a throwaway HOME, workspace and projects rooted under
the real HOME — never under `/tmp`, which the gate classifies as scratch; the
real `~/.operator` and `~/.codeium` are never touched) and drives the **exact
hook command read back out of the installed `hooks.json`** with payloads in
the exact stdin shapes above:

- `rm -rf ~/demo-factory/canary` via `pre_run_command` → shadow: exit 0 +
  OP-003 receipt; enforce: exit 2 with `OPERATOR: STOPPED` on stderr
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → exit 2 in enforce
- `echo OPERATOR_CANARY_STOP_7f3a` → exit 2 in enforce (whole chain proven)
- `ls -la` → exit 0 in both modes, stdout and stderr empty, receipted with no
  rule hits
- `pre_write_code` editing `~/.zshrc` → OP-008, receipt tool `edit` (proves
  the reshape end-to-end)
- `pre_mcp_tool_use` `mcp__slack__send_message` → OP-S02 receipt;
  an MCP argument naming `~/.ssh/id_rsa` → OP-S13 receipt (both shadow-only)
- `rm -rf ./src` inside the session cwd → exit 0 (the gate is precise, not
  trigger-happy)
- the 43-case GuardFall bypass corpus replayed **through the adapter** as real
  `pre_run_command` payloads — verdicts must survive the reshape (43/43,
  semantics mirroring the gate's own `corpus guardfall` runner: a shadow
  technique that co-fires a STOP rule blocks in enforce, as it should)
- merge behavior: existing user hooks and an existing team `.devin/hooks.json`
  preserved, backups written, rerun is idempotent, invalid JSON left untouched
- workspace install without existing hooks → legacy `.windsurf/hooks.json`
  written; with an existing `.devin/hooks.json` → merged there instead, no
  shadowing `.windsurf` directory created
- receipt `cwd` equals the payload's `tool_info.cwd`, receipt `session`
  equals `trajectory_id`, receipt chain verifies; no `~/demo-factory`
  directory is ever created; payloads are never executed — the gate is
  analyzer-only

## Honest limitations

- **Host wiring is untested against a live Windsurf session.** Windsurf is
  not installed on the machine that authored this port. The config paths,
  flat entry schema, `bash -c` execution, payload shapes and exit-code
  semantics are verified against the official docs (linked above), and the
  gate contract is tested end-to-end locally by `test.sh`; what needs the
  real app is hook discovery of `hooks.json`, the stderr-reason display to
  Cascade, and when hook configs are re-read (the docs don't state reload
  timing — start a new Cascade session after installing, and prove the
  wiring with the canary step). Status: `gate-tested`.
- **Hooks only see calls routed through the wired events.** Cascade actions
  that never fire `pre_run_command` / `pre_write_code` / `pre_mcp_tool_use`,
  worktree bootstrap (`post_setup_worktree` is a post-hook and cannot block),
  and commands the owner runs in their own terminal are invisible to the
  gate (this is the gate's boundary on every host, not Windsurf-specific).
- **Restricted Mode disables hooks entirely** — the gate is off there by
  definition.
- **A broken install fails open.** If `python3` or the wrapper cannot run at
  all, the hook exits non-0/2, which Windsurf treats as an error and the call
  proceeds (the stderr note is visible). The wrapper writes a loud
  "gate missing" note in that case. (When the gate process itself runs but
  hits an internal error, its own degraded mode fail-closes on plainly
  destructive text.)
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but `~/.codeium/windsurf/hooks.json` and workspace hooks files (where
  the hook could be unregistered) are not yet in `self_protected_files()`.
  Receipts make the silence visible; wiring those paths into the gate is an
  upstream gate change, and this port never modifies the gate.
- **No timeout knob, performance is on us.** Windsurf sets no per-hook
  timeout and recommends sub-100ms hooks. Each wired call pays one Python
  startup (~tens of ms) plus millisecond-scale analysis. If Cascade feels
  slow with the port, unwire `pre_write_code` first — it is the highest-
  frequency event — and keep the command gate.
- **No matcher means everything on the wired events is forwarded** — read-only
  MCP calls included. That is deliberate (the gate, not the config, decides
  relevance), but it is why `pre_read_code` and `pre_user_prompt` are *not*
  wired: the gate has no rules for them, so wiring them would only add
  latency.
- **Receipts name every decision** — allowed, shadow-stopped, or blocked.
  That is the audit trail, not noise; `operator-gate verify` walks the chain,
  and the chain is digests, not signatures.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
