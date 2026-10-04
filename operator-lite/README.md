# Operator Lite — the free agent-safety gate

One stdlib-only Python file that sits between your AI agent and your machine as a
PreToolUse hook, and **blocks** destructive, polluting and exfiltrating commands
*before they run* — with an explain-mode verdict and a hash-chained receipt for
every decision.

Part of [Vestige](https://github.com/samvallad33/vestige). Free and standalone:
copy one file, wire one hook, and your agent has a deterministic gate.

## Install (one command)

    npx operator-lite

The same on macOS, Linux and Windows. It needs Node for `npx` and Python 3.9 or newer
for the gate. Run it in your own terminal: the gate refuses an install started by an
agent.

It copies the gate to `~/.operator/gate`, registers the Claude Code hook, then wires
each other agent it finds on the machine (Cursor, Gemini CLI, Windsurf, Cline, Goose,
opencode, Amazon Q). On Windows it wires Claude Code and lists the others with their
manual steps. It starts in shadow mode (records every verdict, blocks nothing) and then
replays your last 30 days of Claude Code history through the rules. Nothing in that
history is executed. Everything else passes through to the gate: `npx operator-lite
status`, `npx operator-lite replay`, `npx operator-lite mode enforce`.

The package carries the gate and the host ports and downloads nothing else. It is
published from [`npm/`](npm/) in this directory.

### Without Node

The gate is one Python file and installs itself. This registers Claude Code only:

    curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o /tmp/operator-gate.py && python3 /tmp/operator-gate.py install

On Windows, in PowerShell, with Python 3.9 or newer installed:

    curl.exe -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o "$env:TEMP\operator-gate.py"; python "$env:TEMP\operator-gate.py" install

From a clone, `python3 operator-gate.py install` does the same.

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
must change course). Flip to blocking with `operator-gate mode enforce`.

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

## Platforms

macOS and Linux. Windows support is new: the gate reads Git Bash and PowerShell
commands, Windows paths in every spelling (`/c/Users/me`, `C:\Users\me`, the 8.3
short name), the `cmd /c rmdir /s` family, and `install` registers a hook
Windows can run. On Windows it is tested by `test_windows.py` on a CI runner.

## What it stops

| Rule | Mode | Catches |
|---|---|---|
| OP-000 | STOP | edits to the gate itself, hook registrations, settings files |
| OP-001 | STOP | delete/move of a workspace root (or parent) |
| OP-002 | STOP | wiping the memory store (sqlite3 writes, `vestige gc`) |
| OP-003 | STOP | blind recursive deletes, sweeps from `/`, fork bombs, `rsync --delete` over a folder outside the project |
| OP-004 | STOP | force-push to shared branches, deleting a shared remote branch (`--delete main`, `:main`, `--mirror`), removing `.git` or a whole repository |
| OP-005 | STOP | unreviewed publishes (npm/cargo/twine/docker/gh release) |
| OP-006 | STOP | paid deploys, live-billing actions, infrastructure teardown (`terraform destroy`, `kubectl delete namespace`, `aws s3 rb`, `helm uninstall`) |
| OP-007 | STOP | destructive SQL (DROP/TRUNCATE/unscoped DELETE) and database resets (`migrate:fresh`, `prisma migrate reset`, `rails db:drop`, `manage.py flush`, `dropdb`, `docker compose down -v`) |
| OP-008 | STOP | shell-init writes (`>> ~/.zshrc` = code execution by install) |
| OP-009 | STOP | reverse shells (`/dev/tcp`, `nc -e`, DNS-tunnel tools) |
| OP-010 | STOP | cloud metadata endpoints (169.254.169.254, metadata.google.internal) |
| OP-011 | STOP | deleting or discarding work nothing can bring back, read from the repository's own state: uncommitted changes, untracked files, `.env` files, keys and local databases (`rm`, `git reset --hard`, `git checkout -- .`, `git restore`, `git clean -f`, `git stash drop`) |
| OP-S01..S17 | SHADOW | work-loss git, comms, public mutation, secret writes, pipe-to-interpreter, persistence, env-hijack (LD_PRELOAD/PATH=), exfil shapes, invisible chars, MCP argument exfil, agent-config writes, paste/tunnel egress, new git remotes, sandbox-escape primitives |

OP-011 is new in 0.3.8 and it changes a default: `git reset --hard`, `git checkout -- .`,
`git restore`, `git clean -f` and a plain `rm` now stop when they would lose work that
exists nowhere else. With nothing uncommitted they pass, as does anything stashed or
committed first in the same command. Files made in the last hour count as the session's
own and are not protected. To record instead of stop, set
`"mode_overrides": {"OP-011": "shadow"}` in `~/.operator/commitments.json`.

## A file is judged by what is inside it

`python3 no_bugs.py` is judged by what `no_bugs.py` does, not by its name. The gate
opens the file an agent is about to run and reads it the way it reads a typed command:

- every way a file gets run: an interpreter and a file, `./file`, `bash < file`, a file
  piped into a shell, `npm`/`yarn`/`pnpm`/`bun` scripts (with their `pre` and `post`
  scripts), `make` targets and their prerequisites, `composer` scripts, `python -m`,
  `uv run`, `npx tsx`, `go run`, a local `setup.py`, SQL files handed to a database
  client, notebooks run by `nbconvert` or `papermill`, PowerShell and batch files, the
  git hooks a `git commit` fires, and a file the same command writes before running it;
- what the code does: the commands it hands to a shell, the folders it deletes, the
  database calls that drop or empty, payloads it decodes at run time, the local files
  it imports, and the arguments the script was run with.

Python is read by its syntax tree and other languages as text with strings and
comments set apart, so a file that only prints a dangerous-looking string passes.
`test_files.py` holds both halves: 84 files and commands that must stop, 26 that
must pass.

## What the analyzer sees through

Quote reassembly (`r''m`, `r\m`), `$IFS` expansion, `$(echo rm)` as program,
ANSI-C `$'\x72m'`, base64-decoded bodies, brace + glob enumeration against the
live filesystem, subshell time-bombs (`(sleep 300; rm -rf) &`), session variables
(`T=~/x; rm -rf $T`), cd tracking, heredocs, data-vs-code pipe discrimination,
fork bombs, `mv x /dev/null`, a command written after a redirection
(`>log rm -rf x`). Every STOP tells you which transforms it applied.

## Commands

    operator-gate status | replay | check | mode | approve | verify | install | uninstall | upgrade | corpus guardfall

`check '<command>'` judges one command and prints the verdict, the rule and the
reason. It runs nothing and records nothing, and exits 2 when the gate would stop
the command. `--shell powershell`, `--cwd DIR` and `--json` are accepted.
`replay --stops` lists every stop in the window instead of the most recent five, so
you can count the false ones in your own history.

`status` is the log: the mode, whether the hook is registered, the last call
checked, the counts for the last 7 days, and the most recent commands the gate
stopped or would have stopped, each with its project and reason. `status --rules`
prints the rule table. `mode enforce` switches from recording to blocking, and
`mode shadow` switches back; it is an owner command and needs your own terminal.

`install` puts an `operator-gate` command on your PATH when `~/.local/bin` or
`~/bin` is already on it. Otherwise run it as
`python3 ~/.operator/gate/operator-gate.py`; every hint the gate prints uses the
form that works on your machine.

`verify` walks the receipt hash chain. `corpus guardfall` replays 46 adapted
GuardFall bypass cases (must pass 46/46). `upgrade` describes the paid Operator
gate and where to get it. `test_parser.py`, `test_replay.py`, `test_files.py`,
`test_database.py`, `test_lostwork.py` and `test_powershell.py` are the tests; CI runs
them on Linux and macOS, and `test_windows.py` on Windows as well. `corpus` needs a clone: the corpus
file ships in the repository, not with the single gate file. Every verdict appends to
`~/.operator/receipts/<date>.jsonl`.

## What it does not stop

It blocks what its closed rule set recognizes, routed through hooked tools. It cannot
see around hooks, and parse failures fail open except for plainly destructive text.
Receipts are hash-chained digests, not signatures. Known gaps in 0.3.8:

- A recursive delete whose target is a variable nothing sets (`rm -rf "$DIR"/*`) is
  recorded, not stopped.
- Commands sent to another machine or into a container (`ssh`, `docker exec`,
  `kubectl exec`) are read for database resets and teardown, not for file deletes.
  Commands wrapped by `tmux`, `screen` or `watch` are not opened.
- Disk formatting tools and recursive permission changes over a home folder pass.
- `git branch -D` and `git reset --hard <older commit>` are recorded, not stopped;
  the commits stay in the reflog.
- Code fetched from the network and piped into a shell is recorded, not stopped.
- Compiled programs cannot be read, and code whose target is only known at run time
  is not judged.
- What the Write and Edit tools put into a file is not judged, only where they write.
- State-aware rules read the repository as it is now. `replay` judges old commands
  against today's files.
