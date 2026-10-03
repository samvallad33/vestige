# Operator Lite — the free agent-safety gate

One stdlib-only Python file that sits between your AI agent and your machine as a
PreToolUse hook, and **blocks** destructive, polluting and exfiltrating commands
*before they run* — with an explain-mode verdict and a hash-chained receipt for
every decision.

Part of [Vestige](https://github.com/samvallad33/vestige). Free and standalone:
copy one file, wire one hook, and your agent has a deterministic gate.

## Install (3 hosts, one script)

    mkdir -p ~/.operator/gate
    cp operator-gate.py ~/.operator/gate/
    chmod 755 ~/.operator/gate/operator-gate.py
    echo shadow > ~/.operator/mode          # always shadow first

**Claude Code** (`~/.claude/settings.json`):
```json
{ "hooks": { "PreToolUse": [{ "matcher": "*",
  "hooks": [{ "type": "command",
  "command": "python3 ~/.operator/gate/operator-gate.py hook --source claude" }] }] } }
```
**Codex** (`~/.codex/hooks.json`): same shape, `--source codex`.
**OpenClaw**: the gate is already bundled in a ClawHub plugin — one line:

    clawhub install vestige-operator-lite

**Cline**: file-based `PreToolUse` hook + SDK plugin — `sh operator-lite/ports/cline/install.sh`
(2 commands + 1 settings toggle; see [ports/cline/README.md](ports/cline/README.md)).

(Or any `before_tool_call` plugin that pipes the command through the hook
binary: gate exit 0 = allow, exit 2 = block with the reason.)

ALLOW = exit 0, silent. STOP = exit 2, reason on stderr (the agent sees it and
must change course). Flip to blocking with: `echo enforce > ~/.operator/mode`.

## What it stops

| Rule | Mode | Catches |
|---|---|---|
| OP-000 | STOP | edits to the gate itself, hook registrations, settings files |
| OP-001 | STOP | delete/move of a workspace root (or parent) |
| OP-002 | STOP | wiping the memory store (sqlite3 writes, `vestige gc`) |
| OP-003 | STOP | blind recursive deletes, sweeps from `/`, fork bombs |
| OP-004 | STOP | force-push to shared branches, remote-main deletion, `.git` removal |
| OP-005 | STOP | unreviewed publishes (npm/cargo/twine/docker/gh release) |
| OP-006 | STOP | paid deploys and live-billing actions |
| OP-007 | STOP | destructive SQL (DROP/TRUNCATE/unscoped DELETE) |
| OP-008 | STOP | shell-init writes (`>> ~/.zshrc` = code execution by install) |
| OP-009 | STOP | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | STOP | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |
| OP-S01..S17 | SHADOW | work-loss git, comms, public mutation, secret writes, pipe-to-interpreter, persistence, env-hijack (LD_PRELOAD/PATH=), exfil shapes, invisible chars, MCP argument exfil, agent-config writes, paste/tunnel egress, new git remotes, sandbox-escape primitives |

## What the analyzer sees through

Quote reassembly (`r''m`, `r\m`), `$IFS` expansion, `$(echo rm)` as program,
ANSI-C `$'\x72m'`, base64-decoded bodies, brace + glob enumeration against the
live filesystem, subshell time-bombs (`(sleep 300; rm -rf) &`), session variables
(`T=~/x; rm -rf $T`), cd tracking, heredocs, data-vs-code pipe discrimination,
fork bombs, `mv x /dev/null`. Every STOP tells you which transforms it applied.

## Commands

    operator-gate status | verify | corpus guardfall | install | upgrade

`verify` walks the receipt hash chain. `corpus guardfall` replays 46 adapted
GuardFall bypass cases (must pass 46/46). `upgrade` describes the paid Operator
gate and where to get it. Every verdict appends to
`~/.operator/receipts/<date>.jsonl`.

## Honest boundary

It blocks what its closed rule set recognizes, routed through hooked tools.
It cannot see around hooks, and parse failures fail open except for plainly
destructive text. Receipts are hash-chained digests, not signatures.
