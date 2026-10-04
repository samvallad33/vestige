# Operator Lite — Goose port (official)

The Operator gate, wired into [Goose](https://github.com/block/goose) (Block's
open-source AI agent) through Goose's **lifecycle hooks** — the same
host-agnostic gate that ships for Claude Code, Codex, OpenClaw, OpenHands,
OpenCode and Cline. Part of
[Vestige](https://github.com/samvallad33/vestige); see the
[main Operator Lite README](../../README.md) for the full rule table
(OP-000..OP-010 enforce, OP-S01..S17 shadow).

## The verified mechanism (and why)

Goose ships a hook system following the Open Plugins spec:

- plugins live in `~/.agents/plugins/<name>/` (user scope) or
  `<project>/.agents/plugins/<name>/` (project scope), auto-discovered at
  startup; each provides `hooks/hooks.json`
- a `PreToolUse` hook **blocks the tool call** when it exits with code 2 —
  the reason is taken from stderr — or when it prints
  `{"decision":"block","reason":...}` on stdout; exit 0 with clean stdout
  allows. Hooks receive the payload as JSON on stdin:
  `{"event", "session_id", "tool_name", "tool_input", "working_dir"}`
- sources: [hooks guide](https://goose-docs.ai/docs/guides/context-engineering/hooks),
  [hooks announcement (2026-05-14)](https://goose-docs.ai/blog/2026/05/14/goose-hooks),
  release notes: hooks + PreToolUse denial shipped in
  [v1.41.0](https://github.com/block/goose) (2026-07-03, PRs #9093/#9304),
  `on_failure: "block"` in v1.48.0 (2026-08-27)

That exit-code contract is byte-for-byte the gate's own (exit 0 = allow,
exit 2 = block, reason on stderr), so the port is a thin adapter:

```
goose PreToolUse payload (stdin)
  └─ operator-hook.py      maps working_dir -> cwd (the one field goose
                           names differently), keeps stdout empty
      └─ operator-gate.py hook --source goose
          ├─ exit 0  -> allow (receipt already written)
          └─ exit 2  -> block; gate's stderr ("OPERATOR: STOPPED ...")
                        becomes the tool denial goose shows the model
```

Every `PreToolUse` call is forwarded — the gate, not the adapter, decides
relevance — so shell commands, file writes and MCP extension tools are all
receipted and judged. Goose's `developer` extension sends the unprefixed tool
name `shell` (which the gate's shell routing knows), other extensions
namespace as `{extension}__{tool}`, and both route correctly (verified against
goose PR #12260 and live payloads).

What was **not** viable, checked honestly:

- **GOOSE_MODE=approve** asks the human via Allow/Deny UI buttons; there is
  no documented custom/external approval command that a program could answer
  (checked the [permissions guide](https://goose-docs.ai/docs/guides/managing-tools/goose-permissions)),
  so the gate's one-time `operator-gate approve` permit flow cannot be hooked
  into Goose's approval prompt.
- **An MCP extension proxy** (registering a gated shell tool as an extension)
  cannot be the primary path: the built-in `developer` extension's `shell`
  tool would remain ungated, and Goose gives an extension no authority over
  other tools' calls. Hooks cover every tool; the proxy covers almost none of
  what matters.

## What it stops

Everything the gate stops on any host: edits to the gate/hook/permit files
itself (OP-000), workspace-root deletion (OP-001), Vestige store wipes
(OP-002), blind recursive deletes and sweeps from `/` (OP-003), git history
destruction (OP-004), unreviewed publishes (OP-005), paid deploys (OP-006),
destructive SQL (OP-007), shell-init writes (OP-008), reverse shells (OP-009),
cloud-metadata exfil (OP-010), plus the shadow-only classes (work-loss git,
pipe-to-interpreter, persistence, env hijack, exfil shapes, ...). The
analyzer sees through quote reassembly (`r''m`), `$IFS`, `$(echo rm)`,
ANSI-C quoting, base64 bodies, heredocs, session variables and subshell
time-bombs. Full table: [../../README.md](../../README.md#what-it-stops).

## Install

```sh
sh operator-lite/ports/goose/install.sh
```

The script installs the gate to `~/.operator/gate/` (downloaded from the
Vestige repo if no local copy exists; never overwrites an existing gate),
copies the hook plugin into `~/.agents/plugins/operator-lite/`, sets
**shadow mode** if no mode file exists, and prints next steps. Requires
goose v1.41.0+ and `python3` on PATH.

Verify the wiring:

```sh
sh operator-lite/ports/goose/test.sh    # 29 hermetic checks, installs into a throwaway HOME
```

Then inside goose (shadow mode: the STOP is logged, nothing is blocked):

> Run `echo OPERATOR_CANARY_STOP_7f3a`

and check `~/.operator/receipts/<today>.jsonl` for the `OP-CANARY` receipt.

## Shadow -> enforce

```sh
echo enforce > ~/.operator/mode     # blocks for real; receipts keep flowing
echo shadow  > ~/.operator/mode     # back to log-only
```

In enforce mode a blocked call shows the model the gate's full verdict —
rule, why, the compliant path, and the one-time `operator-gate approve`
digest the owner can grant from their own terminal.

## Honest limitations

- **Hooks are event-shaped, not universal.** Only events goose emits can be
  gated. `PreToolUse` covers tool calls; commands run by other paths goose
  may add (or tools that bypass hook emission) are invisible. `on_failure`
  defaults to fail-open: if the hook cannot run at all, goose proceeds — the
  adapter compensates by fail-**closing** on plainly destructive text when
  the gate is unreachable, mirroring the gate's own degraded mode. For a
  strict fail-closed posture add `"on_failure": "block"` to the hook object
  (goose v1.48.0+).
- **The hook registration itself is outside OP-000's protected list.** The
  gate protects `~/.operator/` and the Claude/ZCode/Codex config files by
  rule, but `~/.agents/plugins/operator-lite/hooks/hooks.json` and
  `~/.config/goose/settings.json` (where `disabledPlugins` could unregister
  this port) are not yet in `self_protected_files()` — a rogue agent could
  try to disable the port by editing them. The receipts make the silence
  visible; wiring those paths into the gate is an upstream gate change, and
  this port never modifies the gate.
- **No human-approval bridge.** Goose has no external approval command
  (see above), so `operator-gate approve` one-time permits are granted from
  the owner's terminal only, never through Goose's UI.
- **Receipts name every decision.** The gate writes a receipt for every
  forwarded call — allowed, shadow-stopped, or blocked. That is the audit
  trail, not noise; `operator-gate verify` walks the chain.
- **UNTESTED against a live goose session** on the machine that authored
  this port (goose not installed). The gate contract, the adapter, the
  `sh -c` + `PLUGIN_ROOT` invocation shape and the payload mapping are all
  tested hermetically (29/29, `test.sh`); what needs a real goose binary is
  hook auto-discovery of `~/.agents/plugins/` and goose-side display of the
  stderr denial — both directly documented upstream behavior.

## Files

| File | Role |
|---|---|
| `hooks.json` | Open Plugins hook config (template, `${PLUGIN_ROOT}`-relative) |
| `operator-hook.py` | stdin payload translator + gate runner (stdlib only) |
| `install.sh` | gate install + plugin wiring, shadow default |
| `test.sh` | hermetic end-to-end tests (real install.sh, real hook command) |
