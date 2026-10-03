# Operator Lite — the free agent-safety gate

One stdlib-only Python file that sits between your AI agent and your machine as a
PreToolUse hook, and **blocks** destructive, polluting and exfiltrating commands
*before they run* — with an explain-mode verdict and a hash-chained receipt for
every decision.

Part of [Vestige](https://github.com/samvallad33/vestige). Free and standalone:
copy one file, wire one hook, and your agent has a deterministic gate.

## Install (one command)

    curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o /tmp/operator-gate.py && python3 /tmp/operator-gate.py install

From a clone, `python3 operator-gate.py install` does the same. It copies the gate to
`~/.operator/gate`, registers the Claude Code hook, starts in shadow mode (records
every verdict, blocks nothing) and then replays your last 30 days of Claude Code
history through the rules. Nothing in that history is executed.

## Other hosts, or by hand

    mkdir -p ~/.operator/gate
    cp operator-gate.py ~/.operator/gate/
    chmod 755 ~/.operator/gate/operator-gate.py
    echo shadow > ~/.operator/mode          # always shadow first

**Claude Code** (`~/.claude/settings.json`), which `install` writes for you:
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

## See it on your own history first

    python3 operator-gate.py replay            # last 30 days of Claude Code history on this machine
    python3 operator-gate.py replay --here     # only calls made under the current directory

`replay` runs every tool call in the Claude Code transcripts on this machine through
the same classifier the hook uses. It prints:

- a scoreboard: calls made, calls a built-in rule would have stopped, calls flagged
  in shadow, calls no built-in rule decides;
- the most recent would-be stops, each with its date, its project, the rule, the
  reason and the part of the command the rule fired on;
- the actions no built-in rule decides: pushes, hook-skipping commits, package
  installs by name, deploys, database commands, CI config and env-file writes;
- the laws your own history drafts from those actions, most frequent first.

Classification only: nothing in the history is executed and no receipt is written.
It needs no install, and `install` runs it once at the end. `--days N`, `--all`,
`--budget SECONDS` and `--json` change the window and the output. `--share` prints
three lines of counts and nothing from the history itself. It reads Claude Code
transcripts only.

From a made-up history:

```
operator-gate replay: the last 30 days on this machine. Nothing was executed.

       23  tool calls your agents made (2 Claude Code sessions, 2 projects)
        3  a built-in rule would have stopped
        1  flagged in shadow: recorded, not stopped
       14  no built-in rule decides: only you can

Would have been stopped (all of them):
  Mar 21  shop-api           OP-004 force push to a shared branch
                             git push --force origin main
  Mar 19  shop-api           OP-007 destructive SQL
                             psql $DATABASE_URL -c 'DROP TABLE sessions'
  Mar 14  infra              OP-003 recursive delete of ~/Documents/old-terraform-state
                             rm -rf ~/Documents/old-terraform-state
By rule:
      1  OP-003 no-blind-recursive-delete   recursive delete of ~/Documents/old-terraform-state
      1  OP-004 no-history-destruction      force push to a shared branch
      1  OP-007 no-destructive-sql          destructive SQL

Flagged in shadow, recorded and not stopped:
      1  OP-S01 work-loss                   git reset --hard HEAD~1

No built-in rule decides these. They ran:
      4  pushed to a remote                 git push origin main
      3  installed packages by name         npm install left-pad
      2  ran a database client or migration psql -c 'select count(*) from users' shop
      2  deployed or changed infrastructure terraform apply -auto-approve
      1  wrote CI or deploy config          Write .github/workflows/ci.yml
      1  wrote an env file                  Write .env
      1  skipped git hooks                  git commit --no-verify -m wip

Your first laws, drafted from this history:
  "No push without my permit."                           4 times
  "No new package without my review."                    3 times
  "Deploys and infrastructure changes are mine."         2 times
  "No database client or migration without my permit."   2 times
  "Never skip git hooks."                                1 time
  "CI and deploy config are mine to change."             1 time
  "Env files are mine to change."                        1 time
```

The built-in rules are free. Enforcing the drafted laws is what Operator, the paid
gate, does: `operator-gate upgrade`.

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
fork bombs, `mv x /dev/null`, a command written after a redirection
(`>log rm -rf x`). Every STOP tells you which transforms it applied.

## Commands

    operator-gate status | verify | corpus guardfall | install | upgrade | replay

`verify` walks the receipt hash chain. `corpus guardfall` replays 46 adapted
GuardFall bypass cases (must pass 46/46). `upgrade` describes the paid Operator
gate and where to get it. `python3 test_parser.py` and `python3 test_replay.py`
run the tests; CI runs them on Linux and macOS. `corpus` needs a clone: the corpus
file ships in the repository, not with the single gate file. Every verdict appends to
`~/.operator/receipts/<date>.jsonl`.

## Honest boundary

It blocks what its closed rule set recognizes, routed through hooked tools.
It cannot see around hooks, and parse failures fail open except for plainly
destructive text. Receipts are hash-chained digests, not signatures.
