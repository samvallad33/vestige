# Operator Lite for OpenCode

Official OpenCode port of [Operator Lite](https://github.com/samvallad33/vestige)
(`operator-lite/` in the main repo). One stdlib-only Python gate, one thin
adapter. No rule logic lives here; the port carries the gate's contract to
OpenCode's plugin hooks.

```
model calls the bash tool
  -> opencode "tool.execute.before" hook (this port's index.js)
       -> python3 ~/.operator/gate/operator-gate.py hook --source opencode
            stdin:  {"tool_name", "tool_input": {"command"}, "cwd", "session_id"}
            exit 0  -> allow (silent; gate already wrote a hash-chained receipt)
            exit 2  -> the plugin throws "Operator Lite blocked: <gate reason>"
```

The thrown error aborts the tool call before execution and its message text is
what the model sees as the tool error, so it can change course.

## What it stops

In enforce mode the gate blocks (exit 2) on eleven closed, deterministic rules:

| Rule | Stops |
|---|---|
| OP-000 | editing the gate itself, its rules, permits, mode or hook registrations |
| OP-001 | deleting or moving a registered workspace root (or any parent of one) |
| OP-002 | wiping or rewriting the Vestige store |
| OP-003 | blind recursive deletes outside scratch/build directories |
| OP-004 | force-pushing over a shared branch; deleting `.git` |
| OP-005 | publishing releases/packages; deleting public repos |
| OP-006 | paid/production deploys and live billing actions |
| OP-007 | destructive SQL (DROP/TRUNCATE, unscoped DELETE) |
| OP-008 | writing shell rc/init files (code execution by install) |
| OP-009 | reverse shells and DNS tunnels (`/dev/tcp`, `nc -e`, dnscat2...) |
| OP-010 | cloud metadata credential endpoints (169.254.169.254 et al.) |

A canary rule (`OP-CANARY`) always stops when present, purely to prove the gate
is wired. Seventeen further shadow-only candidates (OP-S01..S17: git work loss,
outbound comms, secret-shaped writes, pipe-to-interpreter, persistence writes,
env hijacking, exfil shapes, invisible characters, MCP argument exfiltration,
agent-config writes, sandbox escapes...) are recorded to the receipt log but
never block; promote the ones you want after reviewing the log. The gate also
sees through common bypass attempts: `cd` tracking, variable expansion, heredoc
bodies, `$'..'`/base64 encoding, quote-split commands like `r''m`, brace/glob
expansion, subshells and command substitution. Approving a blocked action once
is owner-only and interactive: `operator-gate approve <digest>`.

## Verified mechanism (opencode 1.16.x)

- Hook: `"tool.execute.before"` — `(input: { tool, sessionID, callID }, output: { args }) => Promise<void>`.
  Async is supported. Source of truth: https://opencode.ai/docs/plugins and
  `packages/plugin/src/index.ts` / `packages/opencode/src/session/tools.ts` in
  [sst/opencode](https://github.com/sst/opencode).
- The shell tool's id is `bash` and the command string is `args.command`.
- A thrown `Error` in the hook prevents execution; the message becomes the
  tool error the model sees (`failToolCall` in `session/processor.ts`).
- Plugin locations: project `.opencode/plugin/` (or `.opencode/plugins/`,
  deprecated) and global `~/.config/opencode/plugin/` (or `plugins/`,
  deprecated). The loader treats every module export as a plugin function, so
  `index.js` exports exactly one function.

## Install (3 commands, owner-side)

```sh
cd operator-lite/ports/opencode
sh install.sh            # add --project for project-only scope
sh test.sh               # 10-check harness, expect "10 passed, 0 failed"
```

`install.sh` puts the gate at `~/.operator/gate/operator-gate.py` (bundled copy
preferred, otherwise downloaded from the main repo; never overwrites an
existing gate), copies `index.js` to `~/.config/opencode/plugin/operator-lite.js`,
sets shadow mode, and prints next steps. Requires `python3` on PATH.

The gate starts in **shadow** mode: every decision is classified and written to
`~/.operator/receipts/` as a hash-chained digest log, but nothing is blocked.
Review the log, then flip one file:

```sh
echo enforce > ~/.operator/mode     # shadow -> enforce
echo off     > ~/.operator/mode     # or disable entirely
```

`OPERATOR_GATE_MODE=enforce|shadow|off` overrides the file; `$OPERATOR_GATE`
overrides the gate path; `$OPERATOR_PYTHON` overrides the interpreter.

## Honest limitations

- **Coverage boundary** (stated by the gate itself): it blocks actions routed
  through the hooked tool call that its closed rule set recognises. It cannot
  stop actions that bypass the bash tool, and shell obfuscation it does not
  parse is a residual gap.
- **Shell calls only**: this port forwards shell-type tool calls (`bash` and
  anything carrying a command string). OpenCode's `write`/`edit` tool calls are
  not forwarded; the same hook could carry them if the gate's write-path
  argument keys are normalized. MCP tool calls are not forwarded.
- **Fail-open on gate failure**: if the gate cannot run (missing python3,
  unreadable gate file, 8 s timeout), the port allows the call — unless the
  command plainly looks destructive (`rm -r*`, `mv` onto workspace/config
  paths, `push --force`, `DROP TABLE`, `fly deploy`, `mkfs`, `dd if=`,
  the Vestige store, the canary), in which case it blocks and says so.
- **Receipts are digests, not signatures**: the hash chain detects accidental
  edits; it does not prove authorship or resist a determined tamperer.
- **No inline execution**: test strings are only ever fed to the gate as JSON;
  the gate analyzes, it never executes.

## Test status

`test.sh` (10/10): gate contract with OpenCode-shaped payloads (plain
`rm -rf` -> STOP, `r''m` obfuscation -> STOP, benign `ls` -> ALLOW), the 43-case
GuardFall bypass corpus (43/43), adapter hook contract driven directly with node
(block / canary / benign / gate-unreachable), and — with the `opencode` CLI
installed — plugin discovery via `opencode debug config` plus a clean
`opencode debug startup` smoke run. **UNTESTED**: a live blocking decision
inside a real OpenCode session (needs a configured model). Verify manually with
the canary: ask OpenCode to run `echo OPERATOR_CANARY_STOP_7f3a` — in shadow
mode it logs an OP-CANARY STOP receipt without interrupting the session.
