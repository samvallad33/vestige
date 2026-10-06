# Frequently Asked Questions

> Answers for Vestige 4.1.1

Vestige is the Causal Proof Engine and the operating system for AI agents. Its kernel is
Strata, an append-only, hash-chained, signed log. In the engine there are no vectors, no
RAG and no similarity, and every output carries its proof: a record id, an edge path,
or a receipt.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Identity & Persona](#identity--persona)
- [How Records Strengthen and Fade](#how-records-strengthen-and-fade)
- [Advanced Features](#advanced-features)
- [Power User Tips](#power-user-tips)
- [Use Cases](#use-cases)
- [Technical Deep-Dives](#technical-deep-dives)
- [Comparisons](#comparisons)
- [Things Most People Miss](#things-most-people-miss)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

<details>
<summary><b>"Can Vestige support multiple agents or MCP clients?"</b></summary>

**Yes, with no setup.** Claude Code in three terminals, Cursor, Codex and Claude Desktop
can all run `vestige-mcp` at once. The first one to start takes the store's lock and serves
it. The others connect to it and relay their client's stdio there, so every agent reads and
writes the same log through one writer. If the serving process quits, another takes over
and its client's session continues. See
[Storage](STORAGE.md#option-3-multi-agent-household).

For separate identities, give each agent its own `--data-dir`. Two data directories are two
stores.
</details>

<details>
<summary><b>"What's the learning curve for a non-technical human?"</b></summary>

**Honest answer:** Installation requires terminal basics (copy-paste commands). Daily use
requires none.

1. Have a technical friend do the 5-minute install, or follow [Getting Started](GETTING-STARTED.md).
2. Add the session-start routine from [CLAUDE-SETUP.md](CLAUDE-SETUP.md) to your client's
   instruction file.
3. Talk normally. Ask the agent to save things and to tag them. The agent makes the calls.

The one habit that matters: **tag on purpose.** A record is found by its id or an exact
tag, so the tag is how you get it back.
</details>

<details>
<summary><b>"What input do you feed it? How are records created?"</b></summary>

Your agent creates records through tool calls. Four ways:

1. **Explicit.** You say "save that I prefer dark mode" and the agent calls `smart_ingest`.
2. **Proactive.** The agent decides a fact is worth keeping and calls `smart_ingest`.
3. **Code.** The agent records a pattern or decision with `codebase` action
   `remember_pattern` or `remember_decision`, anchored to files and symbols.
4. **History.** `codebase` action `ingest_repo` turns each commit of a local checkout into
   a change record.

From a terminal you can use `vestige ingest "<text>" --tags a,b`.

Every write passes the log's gate and returns a receipt.
</details>

<details>
<summary><b>"Can it be filled with a conversation stream in realtime?"</b></summary>

No. Vestige is **tool-based**, not stream-based. The agent decides what is worth saving.
Saving everything would bury the records you need. Nothing is deduplicated by
similarity in 4.x either, so every duplicate you save is kept.
</details>

---

## Identity & Persona

<details>
<summary><b>"Can I store preferences, identity or persona notes?"</b></summary>

Yes, as tagged records. Vestige does not treat identity specially. You choose the type and
the tags, and you load them by tag at session start:

```
smart_ingest(content="User prefers tabs over spaces", node_type="fact", tags=["preferences"])
smart_ingest(content="I value continuity and plain answers.", node_type="note", tags=["identity"])

recall(handle="identity")
```

`recall` takes an exact handle, so `identity` returns every live record with that tag
and `Identity` returns nothing. See [CLAUDE-SETUP.md](CLAUDE-SETUP.md).
</details>

<details>
<summary><b>"What happens when two records conflict?"</b></summary>

Both stay. 4.x does not detect conflicts by comparing text, and it does not merge or
reinforce on similarity. You decide:

- `memory` action `edit` admits a corrected successor and retires the old record, which
  stays on the log.
- `suppress` hides the wrong record from every read. It cannot be undone.
- `memory` action `demote` marks it wrong and lets it fade faster. It is not deleted.

`recall` modes `reason` and `contradictions` returned lookalike-based answers in v3. In
4.x they return `similarity_disabled`.
</details>

---

## How Records Strengthen and Fade

<details>
<summary><b>"When records fade, do you delete them?"</b></summary>

**No.** Every record has an FSRS card. Its retrievability falls with time and rises when you
review it. Four bands describe where it sits: Active (0.7 and above), Dormant (0.4),
Silent (0.1) and Unavailable (below that). A faded record is still on the log and still
found by its id and tags. Handle recall returns it whatever its retention.

Fading changes what ranks and what is flagged: GhostLink's scores, `forgotten_lesson`
(which looks for fix and lesson records that decayed below 0.5), `dream`, and `project`
(which leaves out records below a retention floor, default 0.3).

Nothing in 4.x deletes a record. `purge` is withheld, because on an append-only signed log
it could hide a record but not erase its bytes.

How the curve works, and where its constants come from: [The Science](SCIENCE.md).
</details>

<details>
<summary><b>"Does using a record keep it strong?"</b></summary>

Only an explicit review does. A read never changes a card, because a result being shown
is not proof that it was right. `memory` action `promote` folds a review rated Easy and
`demote` one rated Again. Both return a receipt.
</details>

<details>
<summary><b>"What happened to Spreading Activation, Synaptic Tagging and Dual-Strength?"</b></summary>

They belong to the v3 engine and are not part of a Strata log:

- **Spreading activation** primed related records by embedding similarity. 4.x follows only
  edges the log recorded.
- **Synaptic tagging** strengthened a time window around something important. A Strata log
  records no capture events. `maintain` action `importance_score` still scores text you
  pass it, but it writes nothing.
- **Dual-strength** tracked storage and retrieval strength separately. A Strata log has
  one derived value, retrievability, next to FSRS stability and difficulty.

The research grounding stays documented in [The Science](SCIENCE.md), with each mechanism
labeled implemented, inspired by, or v3 only.
</details>

---

## Advanced Features

<details>
<summary><b>"What happened to Prediction Error Gating?"</b></summary>

It compared each new save to existing records by similarity and created, merged or
reinforced. That needs embeddings, which 4.x does not ship. A 4.x save never merges. Since
4.2.0 a save whose text already exists in the scope (after NFC, case and whitespace
normalization) reinforces the original with a small echo record and answers
`decision: "reinforce"`; different text is always a new record.

What you have instead: `dedup` action `scan` lists **exact** duplicates (identical content
hash or identical declared source), and `memory` action `edit` retires an old version in
favor of a successor.
</details>

<details>
<summary><b>"What are Intentions?"</b></summary>

Reminders that surface when a session opens or when you call `check`:

```
intention(action="set", description="Review the auth refactor",
          trigger={type: "context", file_pattern: "auth", codebase: "my-project"}, priority="high")

intention(action="check", context={codebase: "my-project", file: "src/auth/login.ts"})
```

**Trigger types:** `time` (`at` or `in_minutes`), `context` (`codebase`, `file_pattern`,
`topic`), `event` (`condition`), plus recurring and compound.

Matching is plain: a context field fires when it is a case-insensitive substring of what
`check` is given, and an event fires only on an exact key such as `build_finished`. Nothing
runs in the background. A trigger fires when something calls `check`, and `session_start`
does. An intention that is past due shows `OVERDUE`. See [INTENTIONS.md](INTENTIONS.md).
</details>

<details>
<summary><b>"What's the difference between all the search tools?"</b></summary>

There is no search. `recall` takes a handle:

| Handle | Finds |
|--------|-------|
| A full id such as `mem-0000000000000001` | That record |
| A unique id prefix of 8 or more characters | That record, or an `ambiguous` refusal listing candidates |
| An exact tag | Every live record carrying it, from every scope |
| `commit:<sha>` | The change record `ingest_repo` made for that commit |
| `evidence:<sha256>` | The GhostLink composition record that cites that hash |

A free-text query returns `similarity_disabled`, and a miss returns `handle_required`.
</details>

<details>
<summary><b>"How do I make certain records sticky?"</b></summary>

1. **Promote** it after it proves useful: `memory(action="promote", id=...)`.
2. **Tag** it narrowly so it loads at session start: `recall(handle="release-checklist")`.
3. **Project** the durable ones into `CLAUDE.md` or `MEMORY.md` with the `project` tool, so
   other clients read them without calling Vestige.

Fading never removes a record, so "never forget" is the default for what is on the log.
</details>

<details>
<summary><b>"What does the consolidation cycle do?"</b></summary>

On a Strata log, nothing. `maintain` action `consolidate` is withheld and refused with
`unavailable_in_4_0`, because every phase was a no-op and the zeros read as a completed
pass. `vestige consolidate` is also a no-op.

What does work:

- `maintain` action `dream` replays recorded edges and folds one FSRS review per endpoint.
- `maintain` action `dream_compile` re-weights recorded edges among the top records.
- `session_start` tells you when a backup or a dream is due.

Both dream actions need at least 5 live records and say so when they have nothing to replay.
</details>

---

## Power User Tips

<details>
<summary><b>"What node types should I use?"</b></summary>

| Node Type | Use For | Example |
|-----------|---------|---------|
| `fact` | Objective information | "The user's timezone is PST" |
| `concept` | Ideas and principles | "This codebase prefers composition" |
| `decision` | Choices and their reasons | "We chose Postgres because..." |
| `pattern` | Recurring code patterns | "All endpoints use this error handler" |
| `event` | Things that happened | "Deployed v2.0 on March 15" |
| `person`, `place`, `note` | As named | "Alex prefers async communication" |
| `state` | A snapshot that expires | "Build is at 80%" (valid 30 days unless you set `validUntil`) |

`node_type` is free text and is not validated, so a typo creates a new type. Types do not
affect any ranking. They matter to `project` (decisions, patterns and rule-tagged facts),
to `memory_status` counts, and to GhostLink, which groups its sampler by type.
</details>

<details>
<summary><b>"How should I structure tags?"</b></summary>

Tags are the only index, and they are exact and case-sensitive. Tag recall returns **every**
record under the tag, with its full content, from every scope, and you cannot intersect two
tags. So:

- Give each record one project tag and one or two **narrow topic tags**, spelled the same
  way every time.
- Do not recall a broad tag such as `decision` or `verified` in a large store. Pick the
  narrow one.
- Put dates in the content, not in tags.
- Check `memory_status` view `stats` (`counts.byTag`) before inventing a new tag.

`dedup` actions `tag_rename` and `tag_merge` preview exact tag cleanups.
</details>

<details>
<summary><b>"Can I query the store with SQL?"</b></summary>

No. The store is a signed append-only log, not a SQLite file. Use the tools:

| To see | Use |
|--------|-----|
| Counts and health | `memory_status` views `stats`, `health`, `retention`, or `vestige stats` |
| A record and its origin | `memory` action `get`, `memory_status` view `provenance` with `memoryId` |
| What a write did | `receipt` action `get` |
| Everything, as data | `vestige export file.jsonl --format jsonl`, or `maintain` action `export` |
| The log's integrity | `vestige strata-verify <data directory>` |
</details>

<details>
<summary><b>"What are the fixed thresholds?"</b></summary>

None are tunable by a setting. They are in the code:

| Parameter | Value |
|-----------|-------|
| Causal walk bound | 8 hops, 500 nodes |
| GhostLink bridge radius | 3 hops over `touched`, `derived_from`, `closed_by` |
| GhostLink divergent radius | 6 hops (a farther pair counts as 7) |
| `forgotten_lesson` threshold | retrievability below 0.5 |
| `dream` edge-strength floor | 0.5 (the `min_similarity` argument sets it) |
| Accessibility bands | 0.7, 0.4, 0.1 |
| `smart_ingest` | 20 items per batch, 16 links per record |
| `session_start` budget | 1000 (bytes divided by four) |
| `project` defaults | retention 0.3, 60 items |
</details>

<details>
<summary><b>"How do I debug when recall isn't finding what I expect?"</b></summary>

1. **Check the handle.** It must be exact: `Incident` is not `incident`. A prefix needs
   8 or more characters.
2. **Check the tag you meant.** `memory_status(view="stats")` lists tags by count. A typo
   at save time created a different tag.
3. **Check the record exists and is live.** `memory(action="get", id=...)`. A suppressed,
   edited or superseded record is retired and not found by handle.
4. **Check the scope.** Handles resolve across scopes, but walks and GhostLink stay in one.
5. **Check the log.** `vestige strata-verify <data directory>`, and `vestige health`.

Common causes: a case difference in the tag, a prefix that is too short, a record saved
under an untagged call, or a record that was edited (its id changed to the successor's).
</details>

---

## Use Cases

<details>
<summary><b>"How do developers use Vestige?"</b></summary>

- **Decisions with reasons.** `codebase` action `remember_decision`, anchored to the files
  that carry it. `verify` flags the decision when the code drifts.
- **Debugging.** Save the failure as a record, declare its cause with `links`, and run
  `causal_walk` from the failure. `forgotten_lesson` finds fixes you already wrote and
  have since faded.
- **History.** `ingest_repo` makes every commit a change record, so `verify` shows which
  past changes the code has drifted from.
- **Across sessions.** `session_start` opens with open intentions and what is due.
</details>

<details>
<summary><b>"Can Vestige be used for team knowledge?"</b></summary>

With care. There is no access control, no sync service and no merge. Options:

1. **One store per person.** Each developer's own data directory. Nothing is shared.
2. **A store per project in `.vestige/`.** Do not commit the log. It holds everything saved
   to it, unencrypted. Add `.vestige/` to `.gitignore`.
3. **Share the projection.** The `project` tool writes the durable subset of a scope into
   a fenced region of `CLAUDE.md` or `MEMORY.md`, one record id per line. That file is safe
   to commit once you have read it.
</details>

<details>
<summary><b>"How is Vestige different from a notes app?"</b></summary>

| | Notes app | Vestige |
|---|---|---|
| Finding things | You search | An exact handle, by you or your agent |
| Age | Everything stays equal | Records fade under FSRS and are never deleted |
| Duplicates | You manage them | `dedup scan` lists exact ones |
| Proof | None | A receipt per write, replayable against the log |
| Failures | A note you hope you find | A walk backward along links you declared |
</details>

<details>
<summary><b>"Can Vestige help an agent act as a coach or advisor?"</b></summary>

It can keep continuity between sessions: what was discussed and what helped. Important caveats:

- Vestige is not HIPAA compliant and gives no compliance guarantees.
- The log is stored locally and is **not encrypted** on disk. Use full-disk encryption.
- For anything clinical, consult professionals. Records do not replace them.
</details>

---

## Technical Deep-Dives

<details>
<summary><b>"How does the scheduling work?"</b></summary>

Each record has an FSRS card in the Strata kernel. Retrievability follows a power-law curve,
`R = (1 + FACTOR × t / S) ^ -0.5`, with stability `S` growing on each review. The constants
are pinned in the source and versioned, the arithmetic is fixed-point, and a review update
measures elapsed time in log positions rather than days. Details and honest limits:
[The Science](SCIENCE.md#fsrs-scheduling-in-the-strata-kernel).
</details>

<details>
<summary><b>"What embedding model does Vestige use?"</b></summary>

None. A default 4.x build has no embedding model, no vector index, no BM25 and no FTS, and
nothing downloads on first start. `memory_status` still reports fields like
`embeddingModel: null` and `embeddingsCompiledIn: false`, left over from v3.
</details>

<details>
<summary><b>"What replaced hybrid search?"</b></summary>

Nothing ranks by resemblance, so there is nothing to fuse. Recall is by exact handle. The
only ordering is deterministic: GhostLink's scores come from recorded edges, FSRS state and
log positions, and `causal_walk` is a breadth-first search over recorded edges.
</details>

<details>
<summary><b>"What's the performance like?"</b></summary>

There is no 4.x benchmark on this page, and the old latency table described v3's vector
search. What is documented: `vestige-mcp` is about 8.0 MB on macOS arm64, and the first
launch on a v3 store ran the upgrade in about 17 seconds on a 297 MB store before the MCP
handshake answered. A walk is bounded at 8 hops and 500 nodes. See
[Benchmarks](BENCHMARKS.md) for what is measured and what is not.
</details>

<details>
<summary><b>"Is there any network activity?"</b></summary>

**Not by default.** A default 4.x build has no model download, no startup version check
and no telemetry. The calls that use the network are explicit: `vestige update`
(downloads a release and verifies its checksum), and, in builds with extra features that
no release includes, `source_sync` and `vestige sync --cloud`. `ghostlink` `weave` can
record a URL and hash you give it as evidence, but Vestige never fetches the URL. The
HTTP transport is off unless you enable it, and it binds `127.0.0.1`. See
[Configuration](CONFIGURATION.md#offline-by-default) and [SECURITY.md](../SECURITY.md).
</details>

---

## Comparisons

<details>
<summary><b>"How is Vestige different from RAG?"</b></summary>

| Aspect | Traditional RAG | Vestige 4.x |
|--------|-----------------|-------------|
| Storage | Chunk and embed everything | Records you or your agent chose to save, in a signed log |
| Retrieval | Top-k similarity | An exact handle |
| What counts as a link | Anything that scores close | Only an edge the log recorded |
| Updates | Re-embed documents | A successor admitted through the gate, with a receipt |
| Age | Nothing fades | FSRS scheduling |
| Proof | None | An id, an edge path or a receipt on every output |

Resemblance search hands back the nearest-sounding record, and the cause of a bug rarely
shares words with its error. Vestige follows the links that were declared.
</details>

<details>
<summary><b>"Do I need to switch off my AI client's built-in memory?"</b></summary>

No. They are independent. A client's own feature lives on the vendor's side and Vestige
lives on your machine, so you can run both. Vestige gives you explicit control over what is
saved, receipts, causal walks, and records anchored to your code.
</details>

<details>
<summary><b>"Why not just use a vector database?"</b></summary>

A vector database answers "what is closest to this text." Vestige answers "what did we
record that led here." It has no embeddings at all. What it adds is a gate and a receipt on
every write, edges that only count if declared, scheduled fading, and a debugger that walks
failures backward.
</details>

---

## Things Most People Miss

<details>
<summary><b>"What features exist that most people don't know about?"</b></summary>

1. **Declare the cause when you save.** `smart_ingest` takes `links` (`derived_from`,
   `evidence_of`, `closes`) to existing records. `causal_walk` and `forgotten_lesson` follow
   them.
2. **Your git history is a source.** `codebase` action `ingest_repo` previews first, then
   records commits as anchored change records.
3. **`receipt` replay.** It rebuilds the state from the log and reports any mismatch.
4. **`selftest`.** Plants a known cause in a throwaway copy and checks the walk finds it.
5. **`project`.** Writes the durable subset of a scope into `CLAUDE.md` or `MEMORY.md`.
6. **Provenance.** `source` on a save, and `memory_status` view `provenance` for the frame
   that admitted a record.
7. **GhostLink `weave` with `evidence`.** Record what a tested pair showed, with up to 8
   outside findings (URL, sha256, time) that Vestige stores but never fetches.
8. **`vestige scan-secrets`.** Audits the log for credential-shaped values already stored.
</details>

<details>
<summary><b>"What's planned for future versions?"</b></summary>

The main open item is real erasure as crypto-erasure
([#402](https://github.com/samvallad33/vestige/issues/402)), because `purge` is withheld on an
append-only log. See the [changelog](../CHANGELOG.md) for what shipped and the
[roadmap](ROADMAP.md) for older plans.
</details>

<details>
<summary><b>"What's the 'magic prompt' to get the most out of Vestige?"</b></summary>

See [CLAUDE-SETUP.md](CLAUDE-SETUP.md) for the full template. The key elements:

**Session start.**
1. `session_start(include_intentions=true, include_status=true, context={codebase, repoPath})`.
   Do not pass `queries`.
2. `recall(handle="<narrow topic tag>")` for the task's history.

**During work.**
- Decision made: `codebase(action="remember_decision", repoPath=...)` with a rationale.
- Fact or correction established: `smart_ingest` with a narrow tag.
- Failure: save it as an `event`, then `causal_walk` from it.

**Hygiene.**
- A record helped: `memory(action="promote", id=...)`.
- A record misled: `memory(action="demote", id=...)`.
</details>

---

## Troubleshooting

### "Command not found" after installation

Make sure `vestige-mcp` is on your PATH:
```bash
which -a vestige-mcp
```

If it is not found, add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` (or `~/.bashrc`),
open a new terminal, or use the full path:
```bash
claude mcp add vestige /full/path/to/vestige-mcp -s user
```

If it prints an older version, an older install comes first on your PATH.

### "Tools not showing" in Claude

1. Check the config file syntax (valid JSON).
2. Restart the client completely, not just a reload.
3. Desktop apps do not read your shell's PATH. Use the absolute path from `which vestige-mcp`.

### "`vestige-mcp (pid N) is serving ...`"

A CLI command that opens the log directly exited with this because a Vestige server holds the
store, and the log has one writer. Use the matching tool through your agent, or stop the
server and run the command again. `vestige backup`, `vestige strata-verify` and
`vestige dashboard` work while a server runs.

### `similarity_disabled` or `handle_required`

You passed free text. `recall` takes an id, a unique prefix of 8 or more characters, or an
exact tag. The error text lists handles it found in your words, if any. See
[Getting Started](GETTING-STARTED.md#2-the-one-rule-you-find-a-memory-by-its-handle).

### `unavailable_in_4_0`

The action is withheld on a Strata log, and the message says why and what to use instead.
The list is in [Tool contracts](TOOL-CONTRACTS.md#withheld-in-4x).

### The first launch after upgrading from v3 is slow

The first 4.x launch on a v3 data directory runs the import before the MCP handshake
answers: about 17 seconds on a 297 MB store. A second agent starting at the same time waits
for it. `VESTIGE_ATTACH_WAIT_SECS` (default 120) bounds the wait. If a v3 server is still
running, quit it first, because it keeps writing to `vestige.db`, which 4.x no longer
reads. If `vestige-upgrade` is missing next to `vestige`, 4.x refuses to start and leaves
`vestige.db` untouched. See [Migrating to Vestige 4.0](MIGRATING-v4.md).

### The log fails verification

```bash
vestige strata-verify "<data directory>"
```

It reports which segment or frame failed. A damaged log refuses appends rather than
guessing. Restore from a backup by copying its `log/` over the data directory's `log/` with
every Vestige client stopped. See [Storage](STORAGE.md#backups).

### Windows notes

- Use the prebuilt release binaries.
- Paths in `~/.claude.json` and other MCP configs want forward slashes (`C:/Users/you/...`).
- Configuration changes take effect on the next MCP connect, not immediately.
