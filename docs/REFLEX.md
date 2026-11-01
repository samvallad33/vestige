# Reflex

Reflex is the Vestige tool that agent hooks call directly. Every hook firing reaches the server that owns
your store, and every firing leaves a line in a hash-chained Hook Ledger.

Its first job is the Operator stop. The Operator gate decides and blocks. Reflex reads what the gate
decided, saves each stop as a memory linked to its rule, tells the stopped agent once, and remembers how
each stop was answered. The gate stays the only thing that can block a call, and Reflex never allows,
retries or routes around a stop.

Its second job is the context manifest: the exact memories each agent holds at every call. A memory the
agent already holds word for word is folded out of later results, a hook never injects it twice, and
every Vestige hook shares one budget per turn.

## What happens on a hook firing

On every firing Reflex reads the gate's day files under `~/.operator/receipts` from where it last
stopped. `$OPERATOR_HOME` moves that folder. Each new line is checked against the digest chain the gate
writes: it must name the digest of the line before it, and its own digest must match its content. A line
that breaks the chain is counted and never becomes a memory.

| Gate receipt | What Reflex records |
|---|---|
| `STOP` | An `operator-stop` memory tagged `op-receipt-<16 hex>`, `derived_from` the rule's memory, which is tagged `op-rule-<rule>`. These are the same tags and text the older `operator-stop-memory.py` hook writes, so `recall(handle='op-receipt-…')` works either way. |
| `PERMIT_GRANTED` | An `operator-resolution` memory with a `closes` edge to the stop it answers, found by the permit's `stop_receipt`. |
| `ALLOW` by a law's own compliant path, or by a used permit | An `operator-resolution` memory with a `closes` edge to the newest open stop of that rule or action in the same session. |

A stop already in memory, written by any other hook, is reused and never written twice.

Only the calling session's rows are written before the hook answers. Every other row, including the
history Reflex finds the first time it runs, drains in one background task per store, so a backlog never
slows a tool call.

## What the agent is told

| Event | What Reflex adds to the agent's context | Budget |
|---|---|---|
| `PreToolUse` | A stop of this session the agent has not heard about yet, then a warning when this exact command was stopped before | 600 characters |
| `PostToolUse`, `PostToolUseFailure`, `PostToolBatch` | Stops of this session the agent has not heard about yet | 600 characters |
| `UserPromptSubmit` | The same, and the turn budget resets | 2,000 characters |
| `SubagentStart` | The stops already told in this session, which hold for the subagent too | 1,500 characters |
| `SessionStart` | The newest stops on this machine from the last seven days, each with the handle to recall it; after compaction, the stops this session was already told | 2,000 characters |
| `Stop` and every other event | Nothing; Reflex records only | 0 |

Each stop is told once per session. Every Vestige hook together adds at most 8,000 characters to a turn,
and the memory hook leaves 1,500 of those for Reflex's notices.

When a later stop hits a rule that was answered before, the notice names that answer and its memory. A
command's fingerprint is the first 16 hex characters of the SHA-256 of the command with its whitespace
collapsed, so the exact command is recognized without storing it again.

## What the stop memory holds

Besides the gate's own fields, a stop memory names the memories that were in the agent's context at that
exact tool call. Reflex learns them from the agent's own Vestige reads on the same MCP connection, and
takes the snapshot at `PreToolUse`. The gate's receipt names that call by its `tool_use_id`.

## What each agent holds

Reflex keeps a context for every session: one for the main agent and one for each subagent. A memory joins
an agent's context when one of that agent's own Vestige reads returns it, when the memory hook injects
it, or when Reflex tells the agent about a stop saved as it. Ids are read from the JSON keys of the
result, never matched in text, and a body is remembered by the SHA-256 of its exact text.

Claude Code sends each tool call's `tool_use_id` in the call's `_meta`. The PreToolUse firing just
before the call records the same id with the agent that made it, so every read lands in exactly that
agent's context. A subagent starts with nothing and its context ends with it. Compaction and `/clear`
empty every context of the session. A read Reflex cannot tie to an agent this way is left as it was.

| What happens | When |
|---|---|
| A memory body is folded to its opening and `memory(action='get', id=...)`, the call that returns it whole | The same agent already holds that exact body and the call did not name that memory itself |
| The result carries a `reflexFold` block with the ids folded, both sizes and the SHA-256 of the result before folding | Any body was folded |
| The memory hook skips a memory | The agent already holds at least that much of it |
| The memory hook drops a memory | It would take the turn past its budget |
| A firing is answered empty and only ledgered | The host reported the same firing twice: the same event for the same call, or the same once-per-turn event within five seconds |

Tags, types, dates, edge paths and receipt ids are never folded. Receipts and the dashboard see every
result whole, because the fold runs last.

Every PreToolUse snapshot records the agent's manifest: the SHA-256 over the sorted ids it holds. A stop
memory names that manifest and the ids, and every Hook Ledger line carries the manifest of the agent it
fired for.

To measure the fold on your own sessions, replay recorded Claude Code transcripts through it offline:

```bash
vestige reflex --replay ~/.claude/projects/*/*.jsonl
```

It reports, per tool, the characters each Vestige result sent and what it would have sent with the fold,
with one context per transcript and per subagent sidechain, emptied at each compaction. It writes nothing.

## Failure Pairing

When a call fails, Reflex reads the tool, the command, the exit code and the first error line, and
classifies the failure by a fixed table. No model is involved.

| Not recorded | Why |
|---|---|
| `ls`, `test`, `stat` and similar on a missing path | A probe that answers by failing |
| `grep`, `rg` and `pgrep` finding nothing, `git diff --quiet`, `diff` | Exit code 1 is their answer, not a failure |
| An interrupt, a refused tool call, an Operator stop | Not a failure of the call; stops have their own memories |

Every other failure gets a signature: the tool, the first words of the command and the first error
line, with paths, numbers and hashes masked, hashed to 16 hex characters. The failure is saved once as
an `event` memory tagged `fail-<signature>`, with the command, the error line, the working directory,
the checked-out commit and the call id, and a `derived_from` edge to the last memory the session wrote.
Anything shaped like a credential is masked before the command is saved.

| What happens | When |
|---|---|
| The agent is told the fix right away, with the failure, the fix memory and the command that worked | An earlier failure with the same signature has a recorded fix |
| A fix memory tagged `fix-<signature>` is saved with a `closes` edge to the failure, naming the command that worked and the files edited in between | The same command next succeeds for the same agent in the same session |
| A `causal_walk` and a `forgotten_lesson` walk run off the hook's path, and a linked lesson below 50 percent retention is told at the next firing | Every recorded failure |

A first-time failure puts nothing in front of the agent.

Failure, fix, stop and resolution memories also carry `reflex-session-<session id>`, so one exact tag
lists everything Reflex recorded for a session.

## Test witnesses

When the whole shell command equals an entry on the witness allowlist, Reflex records a signed test
witness. The allowlist is the `[witness]` table of `vestige.toml` in the data folder plus the
repository's own `.vestige/witness.toml`.

| Event | What Reflex does |
|---|---|
| `PreToolUse` | Takes the git tree of the working directory through a scratch copy of the index, so the user's index is only read |
| `PostToolUse`, `PostToolUseFailure` | Takes the tree again, then signs and saves the witness: the command, the exit code, the SHA-256 of the output the host reported, both trees, and the test count when the runner reported one in JSON or junit |

The witness is a DSSE envelope signed by the store's actor key, saved as an `event` memory tagged
`reflex-witness`, `tree-<tree>` and `reflex-session-<session id>`. The agent is told one line naming the
command, green, red or zero tests, the tree and the memory. A background run, an interrupted run and any
command that is not exactly on the allowlist are not witnessed. `receipt(action='prove')` turns the
witnesses of one commit into a proof bundle, see [Proof-carrying pull requests](PROOF-CARRYING-PRS.md).

## Stale-belief push

Several agents share one store. When one of them retires a memory, every other agent already holding
that memory hears about it before its next action. Reflex records each retirement in order from the
result of the call that made it:

| Call | Recorded as |
|---|---|
| `memory(action='edit')` | The old memory retired for its successor, with the successor's first sentence |
| `smart_ingest` with a `supersedes` link | The old memory superseded by the new one |
| `memory(action='demote')` | Demoted as wrong or unhelpful |
| `suppress`, and each target its cascade suppressed | Suppressed |

On the next PreToolUse, PostToolUse, UserPromptSubmit or SubagentStart, Reflex compares the new
retirements with the agent's context. An agent that held the memory before it was retired is told once,
with the old id, the successor, when it changed and the receipt, and the old id leaves its context. The
session that made the change and agents that never held the memory are told nothing.

## Compaction survival

Compaction rewrites a conversation into a summary, and whatever the summary drops is gone for the rest of
the session. Reflex checks it and repairs it by handle.

| Event | What Reflex does |
|---|---|
| `PreCompact` | Saves a checkpoint memory, tagged `compaction-checkpoint`, listing by exact id every memory the main agent holds |
| `PostCompact` | Reads the summary for those exact ids and saves an audit memory, tagged `compaction-audit` and derived from the checkpoint, naming what was kept and what was dropped |
| `SessionStart` after compaction | Gives back the dropped memories by handle, decisions first, each as its current version, with its type and opening, inside the 2,000 character budget |

An id the summary paraphrased or cut counts as dropped. A memory retired since the checkpoint comes back
as its successor, and a suppressed one does not come back. Hosts without `PostCompact` get every
checkpointed memory back. The checkpoint survives a server restart in `sessions.json`.

## The Hook Ledger

Every firing appends one line to `<data dir>/reflex/ledger.jsonl` with the event, the session, the
characters added, the receipts told, the memories written and the time it took. Each line carries
`prev`, the digest of the line before it, and `digest`, the SHA-256 of the line's compact JSON with sorted
keys and without `digest`. Any edit to a line breaks the chain from that point on. This is a reference
digest that detects edits, not a signature.

To read it:

```bash
vestige reflex --since 24h
```

It reads only `<data dir>/reflex/`, never the log, so it works while a server owns the store. It reports
firings by event, characters told, stops and resolutions recorded, folds and the characters they saved by
tool, memories the memory hook injected, skipped and dropped, duplicate firings, whether the ledger chain
is intact, and the receipt reader's own counters.

## Installing

`npx @vestige/init` registers Reflex when the installed `vestige-mcp` is 6.0.0 or newer.

- **Claude Code** calls Reflex from `mcp_tool` hooks on `PreToolUse`, `PostToolUse`,
  `PostToolUseFailure`, `UserPromptSubmit`, `SubagentStart`, `SubagentStop` and `Stop`, each passing
  the session, the call and the agent id. `SessionStart` at launch fires before MCP servers connect, so it
  uses the command shim `~/.vestige/memory/hooks/reflex.cjs`.
- **The memory hook** names its session when it asks the server what to inject, so the server can skip
  what the agent already holds. Older servers ignore the field.
- **Codex, Gemini CLI and Cursor** use the same command shim on their matching events.

None of these hooks can block a call. The Operator gate stays a separate command hook that fails closed.

While a Reflex server is active it refreshes `<data dir>/reflex/heartbeat`, and the older
`operator-stop-memory.py` hook steps aside so a stop is never handled twice. With an older server, or
none, that hook works exactly as before.

## Files

| Path | Holds |
|---|---|
| `<data dir>/reflex/state.json` | Read positions in the gate's day files, the last digest read, recorded stops, rules, open stops and the background queue |
| `<data dir>/reflex/sessions.json` | What each agent of each recent session holds, so a resumed session gets no repeats after a server restart |
| `<data dir>/reflex/ledger.jsonl` | The Hook Ledger; it rotates to `ledger.jsonl.1` past 8 MB |
| `<data dir>/reflex/heartbeat` | The time of the last firing |
