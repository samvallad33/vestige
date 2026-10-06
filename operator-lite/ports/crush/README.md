# Operator Lite — Crush port (official)

The Operator gate ([../../operator-gate.py](../../operator-gate.py)) wired into
[Crush](https://github.com/charmbracelet/crush) (Charm's terminal AI agent)
through Crush's **`PreToolUse` hooks** — currently its only hook event. One
stdlib-only Python file does all analysis; this port is a thin adapter: a
config fragment plus install/test scripts. It contains no rule logic and no
payload shim, because none is needed (see below).

## The verified mechanism (and why)

Verified in Crush's source, not inferred:

- **Config** — hooks live in `crush.json` (project root; global
  `~/.config/crush/crushrc` also merges) under a top-level `"hooks"` map keyed
  by event. `PreToolUse` is the only supported event. Each entry is a **flat**
  object: `{"name", "matcher"?, "command" (required), "timeout"?}`
  (`HookConfig` in
  [`internal/config/config.go`](https://github.com/charmbracelet/crush/blob/main/internal/config/config.go);
  default timeout 30 s). This is *not* Claude Code's nested
  `{matcher, hooks:[{type,command}]}` shape — Crush's Claude Code
  compatibility is on the payload and decision-JSON side, not the config side.
  Naming the hook `operator-lite` is what makes
  `crush hook remove PreToolUse --name operator-lite` possible.
- **Stdin payload** — the hook receives exactly the gate's contract
  ([`internal/hooks/input.go`](https://github.com/charmbracelet/crush/blob/main/internal/hooks/input.go)):

  ```json
  {"event": "PreToolUse", "session_id": "abc-123", "cwd": "/path/to/project",
   "tool_name": "bash", "tool_input": {"command": "ls -la"}}
  ```

  `tool_input` is emitted as a parsed JSON object "for compatibility with
  Claude Code hooks". `CRUSH_EVENT`, `CRUSH_TOOL_NAME`, `CRUSH_SESSION_ID`,
  `CRUSH_CWD`, `CRUSH_PROJECT_DIR` (+ `CRUSH_TOOL_INPUT_COMMAND`,
  `CRUSH_TOOL_INPUT_FILE_PATH`) are exported as environment variables.
- **Execution** — the `command` string runs through Crush's embedded POSIX
  shell (`sh -c` semantics), so `"$HOME"` expands and the absolute-gate
  lookup works from any project directory.
- **Exit codes**
  ([`internal/hooks/runner.go`](https://github.com/charmbracelet/crush/blob/main/internal/hooks/runner.go)):
  `0` = success, stdout parsed as decision JSON (empty stdout = no decision =
  the tool proceeds — the gate writes nothing to stdout in hook mode);
  `2` = **the tool call is blocked and stderr is the deny reason**; `49` =
  halt the whole turn (the gate never emits this); any other code = logged
  non-blocking error, the call proceeds. Deny wins over allow across parallel
  hooks; identical command strings are deduplicated per event.

That exit-code contract (0 silent / 2 + stderr reason) is byte-for-byte the
gate's own, and every payload field name matches, so the hook is the gate
invoked directly:

```
crush PreToolUse payload (stdin)
  └─ python3 "$HOME/.operator/gate/operator-gate.py" hook --source crush
      ├─ exit 0, empty stdout -> allow (receipt already written)
      └─ exit 2, stderr "OPERATOR: STOPPED ..." -> block; that stderr is the
         deny reason Crush shows the model
```

Goose and OpenHands needed an adapter to rename `working_dir` → `cwd`; Crush
already says `cwd`. No matcher is set, so every tool call is forwarded — the
gate, not the config, decides relevance. Crush names its shell tool `bash`
and its file-edit tool `edit`; both are in the gate's routing tables, and any
tool whose input carries a `command` key is caught by the gate's fallback.

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
sh ports/crush/install.sh [project-dir]      # default: current directory
sh ports/crush/test.sh                       # optional: proves the chain end-to-end
```

`install.sh` is idempotent:

- installs the gate to `~/.operator/gate/operator-gate.py` (downloads from
  GitHub unless an existing copy is same-or-newer; never overwrites a newer
  gate; refuses to wire a gate that does not parse)
- sets `~/.operator/mode` to `shadow` if no mode file exists yet
- merges the `PreToolUse` hook into `<project-dir>/crush.json`: existing keys
  are preserved, the original file is backed up to
  `crush.json.bak-<timestamp>` first, and a file that is not valid JSON is
  left untouched with manual merge instructions printed
- requires `python3`; needs `curl`, `wget`, or Python's urllib for the
  download when no sibling gate exists

To gate every project instead of one, add the same hook to the global config
(`~/.config/crush/crushrc`) with Crush's own command:

```
crush hook add PreToolUse --command 'python3 "$HOME/.operator/gate/operator-gate.py" hook --source crush' --name operator-lite
```

## Manual merge

If an existing `crush.json` cannot be merged automatically (it is not valid
JSON, or `hooks`/`hooks.PreToolUse` has an unexpected shape), nothing is
modified. Add this alongside your existing keys — the reference fragment is
[crush.json](crush.json):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "name": "operator-lite",
        "command": "python3 \"$HOME/.operator/gate/operator-gate.py\" hook --source crush",
        "timeout": 10
      }
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

In enforce mode a blocked call shows the model the gate's full verdict — rule,
why, the compliant path, and the one-time `operator-gate approve` digest the
owner can grant from their own terminal.

## Tested

`test.sh` is hermetic (a throwaway HOME and project rooted under the real
HOME — never under `/tmp`, which the gate classifies as scratch; the real
`~/.operator` is never touched) and drives the **exact command string read
back out of the installed `crush.json`** with payloads in the exact stdin
shape above:

- `rm -rf ~/demo-factory/canary` → shadow: exit 0 + OP-003 receipt; enforce: exit 2 with `OPERATOR: STOPPED` on stderr
- `r''m -rf ~/demo-factory/canary` (quote reassembly) → exit 2 in enforce
- `ls -la` → exit 0 in both modes, stdout empty, receipted with no rule hits
- `edit` targeting `~/.zshrc` → OP-008 (write-path coverage for Crush's file tool)
- `rm -rf ./src` inside the session cwd → exit 0 (the gate is precise, not trigger-happy)
- merge behavior: existing `crush.json` keys preserved, backup written,
  rerun is idempotent, invalid JSON left untouched
- receipt chain verifies; no `~/demo-factory` directory is ever created;
  payloads are never executed — the gate is analyzer-only

## Honest limitations

- **Host wiring is untested against a live crush session.** crush is not
  installed on the machine that authored this port. The config schema, stdin
  payload, `sh -c` execution and exit-code semantics are verified in Crush's
  source (linked above), and the gate contract is tested end-to-end locally
  by `test.sh`; what needs a real crush binary is hook discovery of
  `crush.json` at startup and Crush-side display of the stderr deny reason.
  Status: `gate-tested`.
- **Hooks only see calls routed through hooked tools.** A tool that bypasses
  hook emission, or commands the owner runs in their own terminal, are
  invisible to the gate (this is the gate's boundary on every host, not
  Crush-specific).
- **A broken install fails open.** If `python3` or the gate cannot run at
  all, the hook exits with a non-0/2 code, which Crush logs as a non-blocking
  error and the call proceeds. (When the gate process itself runs but hits an
  internal error, its own degraded mode fail-closes on plainly destructive
  text.) `install.sh` narrows this window by syntax-checking the gate before
  wiring it; Goose's port compensates further with a fail-closed shim, which
  this port omits because Crush's payload needs no shim and duplicating
  rule-shaped logic in an adapter is against the ports rule.
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but project `crush.json` / `~/.config/crush/crushrc` (where the hook
  could be unregistered) are not yet in `self_protected_files()`. Receipts
  make the silence visible; wiring those paths into the gate is an upstream
  gate change, and this port never modifies the gate.
- **Receipts name every decision** — allowed, shadow-stopped, or blocked.
  That is the audit trail, not noise; `operator-gate verify` walks the chain,
  and the chain is digests, not signatures.
- **Timeouts:** the hook is wired with `timeout: 10` (the gate's analysis is
  milliseconds; Crush's default would be 30 s). On timeout Crush records a
  non-blocking error and proceeds.

## Back to the main repo

[operator-lite](../../README.md) · [ports matrix](../README.md) ·
[Vestige](https://github.com/samvallad33/vestige)
