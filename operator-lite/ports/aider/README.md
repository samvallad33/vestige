# Operator Lite for Aider — the PATH-shim port

Aider is the one major agent host with **no hook or plugin API**, so this port
does not register anything with Aider at all. Instead it puts the gate where
Aider's shell commands already go: your `PATH`.

```
aider's shell command ("rm -rf ~/demo-factory/canary")
        |
        v
  your PATH -- ~/.operator/shim/bin/rm  (a ~20-line POSIX sh wrapper)
        |   builds {"tool_name":"shim:rm","tool_input":{"command":"rm -rf ..."},"cwd":$PWD}
        v
  ~/.operator/gate/operator-gate.py        (the one real gate, unmodified)
        |--- exit 0  -> shim execs the REAL /bin/rm (absolute path baked in)
        |--- exit 2  -> "Operator Lite blocked: <reason>" on stderr, exit 126
        '--- unreachable -> fail-open UNLESS the text looks destructive
```

The gate is the same stdlib-only Python file every other port uses:
[`../../operator-gate.py`](../../operator-gate.py). No rule logic lives in this
port; the shims and `operator-dispatch.py` only transport the payload and map
exit codes. One gate, verified once, everywhere.

## Why shims: the verified Aider landscape

Checked against the live docs and changelog (2026-09):

- **Options reference** (https://aider.chat/docs/config/options.html): there is
  **no hook or plugin API** in the options list. The closest mechanisms are
  `--suggest-shell-commands` ("Enable/disable suggesting shell commands
  (default: True)"), `--yes-always` ("Always say yes to every confirmation"),
  `--lint-cmd` / `--test-cmd` (external commands run *after* edits), `--load`
  (run `/commands` from a file on launch) and `--notifications-command`. None
  of these can intercept a shell command before execution.
- **Config reference** (https://aider.chat/docs/config/aider_conf.html):
  `.aider.conf.yml` has no hooks, plugins, lifecycle events or pre/post
  callbacks; the only "hook"-adjacent option is `git-commit-verify`, which just
  toggles whether aider *bypasses git's own* pre-commit hooks.
- **Release history** (https://raw.githubusercontent.com/Aider-AI/aider/main/HISTORY.md):
  grepping the whole changelog for hook/plugin/intercept mechanisms surfaces
  only `--shell-completions`, the `--watch-files` source-file watcher (watches
  for AI comment instructions in files — not command interception), and the
  Python scripting API (https://aider.chat/docs/scripting.html), which scripts
  aider itself rather than hooking its shell execution. As of the v0.86.x
  series, no interception point has been added.

What Aider *does* do is execute its (suggested and confirmed) shell commands in
a real shell that inherits the environment you launched it with. So we prefix
the `PATH` for that one invocation — a mechanism Aider cannot not use.

## What it stops

The shims route the covered binaries through the gate's closed rule set —
the same **11 STOP rules** every host gets:

| Rule | Stops |
|---|---|
| OP-000 | editing the gate, its rules, permits, mode or hook registrations |
| OP-001 | deleting/moving a registered workspace root (or a parent of one) |
| OP-002 | wiping or rewriting the Vestige store |
| OP-003 | blind recursive deletes outside scratch/build dirs |
| OP-004 | force-pushing over a shared branch; deleting `.git` |
| OP-005 | unreviewed publishes/releases (`npm publish`, `gh release create`, repo delete) |
| OP-006 | paid/billing-plane deploys (fly, vercel, wrangler, supabase, stripe live) |
| OP-007 | destructive SQL (`DROP`/`TRUNCATE`/unscoped `DELETE`) |
| OP-008 | writing shell rc/init files (code execution by install) |
| OP-009 | reverse shells and DNS tunnels (`/dev/tcp`, `nc -e`) |
| OP-010 | cloud metadata credential endpoints (169.254.169.254 et al.) |

plus the shadow-only log-first rules OP-S01..S17 (work loss, outbound comms,
exfil shapes, persistence writes, env hijacks, ...). Shadow vs enforce is the
gate's own `~/.operator/mode`, shared with every other port.

## Install (3 commands)

```sh
git clone https://github.com/samvallad33/vestige          # or reuse your checkout
sh vestige/operator-lite/ports/aider/install.sh
PATH="$HOME/.operator/shim/bin:$PATH" aider
```

`install.sh` (POSIX sh, stdlib-only): installs the gate to `~/.operator/gate/`
(downloading from GitHub only if the existing copy is older — never overwrites
a newer gate), installs the shim dispatcher, generates the shims with absolute
real-binary paths baked in (binaries missing on your machine are skipped and
listed), defaults to shadow mode, smoke-tests one shimmed call, and prints the
exact `PATH=` line and alias for your setup.

The shim set covers what the gate's rules can judge through a binary:
`ls rm rmdir unlink mv cp ln install rsync scp dd truncate chmod chown chflags
xattr mkfifo tee sed find git gh npm pnpm yarn cargo docker fly flyctl stripe
vercel supabase psql duckdb mongosh redis-cli sqlite3 curl wget nc base64
mkfs crontab at launchctl tmux screen ...` — the authoritative list is the
`SHIM_BINS` variable in `install.sh`; edit it and re-run the installer to
regenerate. Re-run `install.sh` any time to pick up new binaries (idempotent).

## Shadow → enforce

Install starts in **shadow** (log-only): every shimmed command is judged and
receipted, nothing blocks. Watch the log, then flip:

```sh
python3 ~/.operator/gate/operator-gate.py status     # rules + mode
python3 ~/.operator/gate/operator-gate.py verify     # receipt chain
echo enforce > ~/.operator/mode                      # flip to blocking
```

Blocked actions print `Operator Lite blocked:` plus the gate's reason and the
one-time permit command (`operator-gate approve <digest>`), which only the
owner can run from their own interactive terminal.

## Coverage boundary — read this

Stated plainly, because a half-honest security claim is worse than none:

- **PATH invocations only.** `rm -rf x` and even quote-obfuscated `r''m -rf x`
  land on the shim (the shell resolves the name *after* quote removal, at any
  shell nesting depth). But an **absolute-path call like `/bin/rm -rf x`
  bypasses the shims completely.** So do `sudo rm` (sudo resets PATH), shell
  builtins, and redirects from unshimmed primaries (`echo x > ~/.zshrc` is
  invisible; `tee ~/.zshrc` is caught).
- **Only shimmed binaries are seen.** `python3 -c "shutil.rmtree(...)"` never
  touches a shim. The gate's inline-code analysis only helps on commands that
  reach it through one.
- **Fail-open on gate trouble**, mirroring the OpenClaw reference adapter: if
  the gate is unreachable (missing python3/gate, crash, 8s timeout) a shim
  fails open *unless* the command matches the reference adapter's
  `DESTRUCTIVE_LIKE` regex (`rm -r`, force-push, `DROP TABLE`, `mkfs`,
  `dd if=`, ...), in which case it fails closed. This port extends that filter
  with one extra class — shell-rc writes (`tee ~/.zshrc`, `>> ~/.bashrc`) fail
  closed even with the gate away, because a truncated rc file is code
  execution by install. That filter is the last line, not the first — the
  gate always wins when reachable.
- **Latency**: each shimmed call costs two Python startups (~0.1s on an M-series
  Mac). Fine for agent sessions; you will feel it if you `export PATH` in your
  own interactive shell.
- **Obfuscation the gate cannot parse** remains a residual gap — same boundary
  as every other port; the gate's docstring says it first.

## The universal fallback

Because the mechanism is a `PATH` prefix, the exact same install covers
**literally any terminal agent** — Aider, OpenHands CLI, goose, opencode, or a
friend's bespoke agent — anything that execs shell commands from your PATH:

```sh
PATH="$HOME/.operator/shim/bin:$PATH" <any-agent>
```

It is the fallback for hosts without hooks, and the second layer for hosts that
have them. If Aider ships a real hook API someday, this port shrinks to an
adapter; the gate stays.

## Test

```sh
sh ports/aider/test.sh
```

Runs the real installer into disposable homes under `$HOME` (never `/tmp` —
the gate exempts scratch dirs from OP-003), then proves: benign `ls` passes
and really executes; enforce-mode `rm -rf ~/demo-factory/canary` and the
quote-obfuscated `r''m -rf` are blocked at exit 126 with the real binary never
executed; shadow mode passes through with a receipt; gate-unreachable fails
open for benign text, closed for destructive text and closed for shell-rc
writes. Nothing real is ever deleted — every canary path is nonexistent and a
keeper marker must survive.

Live check (aider 0.86.2 on the machine that authored this port): launched
under the shim PATH, aider's own startup commands — `ls`, `git version`,
`ls ~/.operator` — landed on the shims and receipted with `source=aider`,
`tool=shim:ls` / `shim:git`. What is **not** yet exercised: an LLM-suggested
shell command mid-session (that flow needs an API key on this machine) —
mechanically it is the same `run_cmd` → inherited-PATH path the startup
commands already proved.

## Links

- Main repo: https://github.com/samvallad33/vestige
- The gate: [`../../operator-gate.py`](../../operator-gate.py) — storage, rules,
  receipts: [`../../../docs/STORAGE.md`](../../../docs/STORAGE.md)
- Port matrix: [`../README.md`](../README.md) — one gate, every host
- Reference adapter (fail-open policy source): `openclaw-plugin/index.js` in
  the sibling repo `vestige-operator` (not part of this repository)
