# Vestige Operator Lite for OpenClaw

A deterministic pre-tool gate. Every shell-like tool call your OpenClaw agent
makes is classified against a closed rule set **before it runs**. The gate is
bundled in this package — no clone, no copy step. Intelligence does not equal
authority.

## What it does

Subscribes to `before_tool_call` and pipes each command through the Operator
Lite gate (one stdlib-only Python file, analyzer-only — it never executes what
it inspects):

- **Deterministic verdicts** — 11 STOP rules + 15 SHADOW rules evaluated by a
  shell walker that follows `cd`, expands shell variables, reads heredocs,
  and simulates subshells before matching.
- **GuardFall-proof** — sees through the obfuscation classes from CSA's
  GuardFall research: `$IFS` splitting, ANSI-C quoting, quote reassembly
  (`r''m -rf`), command substitution as the program name, `base64 -d | sh`,
  brace and glob expansion against the live filesystem, subshell time-bombs,
  fork bombs. 46/46 on the adapted bypass corpus.
- **Hash-chained receipts** — every verdict (allow, block, shadow) is
  appended to `~/.operator/receipts/<date>.jsonl` as a tamper-evident chain.
  `operator-gate verify` walks the chain.

## Shadow mode first

The plugin ships in shadow mode: the gate classifies and receipts every call
but blocks nothing. Watch the verdict stream, then flip:

```sh
echo enforce > ~/.operator/mode
```

Back to logging only: `echo shadow > ~/.operator/mode`.

## What the STOP rules catch

| Rule | Catches |
|---|---|
| OP-000 | edits to the gate itself, hook registrations, settings files |
| OP-001 | delete/move of a workspace root (or parent) |
| OP-002 | wiping the memory store |
| OP-003 | blind recursive deletes, sweeps from `/`, fork bombs |
| OP-004 | force-push to shared branches, remote-main deletion, `.git` removal |
| OP-005 | unreviewed publishes (npm/cargo/twine/docker/gh release) |
| OP-006 | paid deploys and live-billing actions |
| OP-007 | destructive SQL (DROP/TRUNCATE/unscoped DELETE) |
| OP-008 | shell-init writes (`>> ~/.zshrc` = code execution by install) |
| OP-009 | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |

SHADOW rules log work-loss git patterns, comms, public mutation, secret
writes, pipe-to-interpreter, persistence, env hijacks, exfil shapes,
invisible characters, and sandbox-escape primitives — without blocking.

## Exit contract

| gate exit | plugin behavior |
|-----------|-----------------|
| 0 | allow — normal flow continues |
| 2 | block — the gate's reason is surfaced to the model, which must change course |
| spawn failure | fail-open, unless the raw text plainly looks destructive |

## Also works without OpenClaw

The same file is a PreToolUse hook for Claude Code and Codex — see the
[Operator Lite README](https://github.com/samvallad33/vestige/tree/main/operator-lite).

Local-first. No telemetry. No dependencies beyond python3.
