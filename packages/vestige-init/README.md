# @vestige/init

Local memory setup for supported AI clients, with an optional Operator gate of 102 STOP rules.

## Three steps

1. Install the v5.0.1 binaries from [GitHub Releases](https://github.com/samvallad33/vestige/releases) and put `vestige-mcp` on PATH. A missing or older binary gets clear instructions and no configuration changes.
2. Run `npx @vestige/init`. Choose **Memory only** (default) or **Memory plus guard**, review every file and appended hook, then confirm once. `--yes` uses memory only; `--guard --yes` also requests the guard. A new guard installation requires your own terminal.
3. Restart your client and approve/trust new hooks in the client's settings when required. Run `npx @vestige/init doctor`. Try: **Tell me something about a friend, then ask me about them later.**

## Memory from the first chat

Say “Maya felt off after dinner at Bangkok Garden”, then “Maya has a peanut
allergy”. The hook's explicit-declaration parser records the exact words,
`person:Maya`, `place:Bangkok Garden` and `allergy:peanut` through `smart_ingest`.
Backfill raises the earlier reaction through Maya, with a change receipt.
“Where should we take Maya for her birthday dinner?” selects the recorded
`person:Maya` identity and recalls the raised reaction. Nothing infers that
the restaurant serves peanuts or that another person's reaction concerns Maya.

The declaration parser covers a small explicit grammar. It does not run a
model or extract arbitrary facts from assistant prose. Session context tells
the connected agent to save richer durable facts with `smart_ingest` and
explicit entities/attributes. Saved statements retain user-statement provenance;
they are remembered reports, not independently verified health facts.

## Hook coverage

| Client | Session | Prompt recall | Turn save | Before compaction | Session end |
|---|---|---|---|---|---|
| Claude Code | `SessionStart` | `UserPromptSubmit` | `Stop` | `PreCompact` | `SessionEnd` |
| Codex | `SessionStart` | `UserPromptSubmit` | `Stop` | `PreCompact` | `SessionEnd` |
| Gemini CLI | `SessionStart` | `BeforeAgent` | `AfterAgent` | `PreCompress` | `SessionEnd` |
| Cursor | `sessionStart` | Capture/recall; inject on `postToolUse` | `afterAgentResponse` | `preCompact` | `sessionEnd` |

Cursor's documented `beforeSubmitPrompt` output cannot carry context. Its
session hook and post-tool hook inject context using supported fields. On
MCP-only clients (Claude Desktop, VS Code, OpenCode, Windsurf, Junie, Zed,
Kiro, Antigravity, and Xcode on macOS), this installer connects tools; lifecycle
automation requires a supported client adapter. Detection uses known config
locations and executable names, not an exhaustive scan of arbitrary applications.

Codex requires review/trust of each changed hook definition through `/hooks`.
The installer does not edit that trust record. Actual client execution is a
separate check from registration; `doctor` labels it accordingly.

## Recall and receipts

`~/.vestige/memory/triggers.json` contains the editable question, memory,
time, planning, preference and back-reference phrase lists. Matching is
case-insensitive with Unicode word boundaries; phrases are literal, not regexes.
Exact recorded entity surfaces also trigger recall. With no match, light recall
returns the top three memories. Selection by entity is separate from salience
ranking; no embeddings, fuzzy search or inferred identity links are used.

`~/.vestige/memory/receipts.jsonl` records hook events, trigger labels, memory
handles, Backfill receipt handles and elapsed time. This local hook log is an
unsigned activity log. The admitted salience changes have their own Strata
receipts. Errors go to `errors.log` and never become conversation text.

Hooks reserve time for process startup, bound their work, and fail open. A
pending declaration stays queued when MCP is unavailable and retries with a
stable idempotency key. Pre-compaction and end hooks flush pending declarations
and revisit their exact Backfill triggers; every admitted save already runs
Backfill. Queueing does not promise that a crashed client emitted its end hook.
The optional guard has a separate fail-closed adapter and keeps the existing
Operator decision/permit logic. `--guard` writes `operator-gate.py` (102 STOP
rules) and the pinned `lite_core.py` only when `~/.operator/gate/operator-gate.py`
is absent. An existing gate, including a paid or newer one, is preserved.
Existing native gate registrations are kept single, so one permit is consumed
once. The adapter and its routing config live under the gate-protected
`~/.operator/state/vestige-memory/`; editable memory settings cannot redirect
the guard. The guard's stop-to-Strata reflex is registered for Claude Code and
Codex through a bounded fail-open relay. A deadline leaves the gate's original
receipts available for retry; exact receipt tags prevent duplicate stop memories.

## Maintenance

```bash
npx @vestige/init doctor
npx @vestige/init status
npx @vestige/init uninstall --dry-run
npx @vestige/init uninstall
```

`doctor` checks the binary, each registered hook, runtime assets, store response
and autofire in the responding process. `status` also lists memory count and
recent Backfill receipts. These commands do not prove a host has trusted or
executed a hook. Uninstall restores byte-identical backups when a file is still
exactly what Vestige installed. If a user edited it later, it removes only the
owned entries, preserves other edits, and retains backups and the memory store.
Re-running setup is idempotent. A client editing a target after preview makes
setup refuse before writing; rerun to review a fresh preview.

The initializer has native fixture coverage on macOS. Windows paths and quoting
have unit coverage; execution inside native Windows and Linux clients requires
separate verification. Registration fixtures do not establish client execution.

## Follow one real stop around the loop

```bash
npx @vestige/init precedent
```

One stop from your own machine: what the agent tried and the gate stopping it, its receipt re-derived from its own bytes, the stop as a Vestige memory, and the precedent the next agent reads.

## Options

| Flag | Behavior |
|---|---|
| `--dry-run` | Preview all planned files and write nothing |
| `--yes` | Memory-only setup without prompts |
| `--guard` | Install the Operator gate when none is present: 102 STOP rules (OP-000 through OP-114, with OP-086–OP-097 and OP-100 reserved) plus the pinned Operator Lite core. An existing gate file is kept. |
| `--client <id>` | Limit setup to selected detected clients |
| `--json` | Machine-readable result |
| `--clients-only` | Compatibility path for the previous connection-only wizard |

Docs and primary client hook references:

- https://learn.chatgpt.com/docs/hooks
- https://code.claude.com/docs/en/hooks
- https://cursor.com/docs/hooks
- https://geminicli.com/docs/hooks/reference/
- https://github.com/samvallad33/vestige
